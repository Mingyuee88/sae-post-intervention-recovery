import argparse
import importlib.util
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

OPEN_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("SAEBENCH_ROOT", str(OPEN_SOURCE_ROOT)))
SRC_ROOT = OPEN_SOURCE_ROOT / "src"
for _path in (ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sae_bench.recovery_core import cast_for_sae, project_to_encoder_null


BASELINE = Path(os.environ.get("REFUSAL_BASELINE_PATH", str(Path(__file__).resolve().with_name("baseline_refusal_sae_recovery.py"))))


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("refusal_recovery_base", BASELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refusal_recovery_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def dump_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_mean(vals: Sequence[Any]) -> Optional[float]:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return mean(xs) if xs else None


def total_delta_norm(delta_by_layer: Dict[int, torch.Tensor]) -> torch.Tensor:
    if not delta_by_layer:
        return torch.tensor(0.0)
    device = next(iter(delta_by_layer.values())).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for delta in delta_by_layer.values():
        total = total + delta.float().pow(2).sum()
    return total.sqrt()


def scale_deltas_to_norm(delta_by_layer: Dict[int, torch.Tensor], max_norm: float):
    norm = total_delta_norm(delta_by_layer)
    if float(norm.item()) > float(max_norm):
        scale = float(max_norm) / (float(norm.item()) + 1e-8)
        for delta in delta_by_layer.values():
            delta.data.mul_(scale)



def _ordered_delta_items(delta_by_layer: Dict[int, torch.Tensor]):
    return [(int(k), delta_by_layer[int(k)]) for k in sorted(int(x) for x in delta_by_layer)]


def _flat_grad(delta_items) -> torch.Tensor:
    chunks = []
    for _, delta in delta_items:
        if delta.grad is None:
            chunks.append(torch.zeros_like(delta, dtype=torch.float32).reshape(-1))
        else:
            chunks.append(delta.grad.detach().float().reshape(-1))
    if not chunks:
        return torch.zeros((), dtype=torch.float32)
    return torch.cat(chunks)


def _assign_flat_grad(delta_items, flat: torch.Tensor):
    offset = 0
    for _, delta in delta_items:
        n = delta.numel()
        if delta.grad is None:
            delta.grad = torch.zeros_like(delta)
        view = flat[offset : offset + n].view_as(delta).to(delta.grad.device, dtype=delta.grad.dtype)
        delta.grad.copy_(view)
        offset += n


def _flat_selected_state(refs: Dict[int, torch.Tensor], curr: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_chunks = []
    cur_chunks = []
    for layer in sorted(refs):
        if layer not in curr or refs[layer].numel() == 0 or curr[layer].numel() == 0:
            continue
        ref_chunks.append(refs[layer].to(curr[layer].device, dtype=torch.float32).reshape(-1))
        cur_chunks.append(curr[layer].float().reshape(-1))
    if not cur_chunks:
        device = next((v.device for v in curr.values() if isinstance(v, torch.Tensor)), torch.device('cpu'))
        z = torch.zeros((), device=device, dtype=torch.float32)
        return z, z.detach()
    return torch.cat(cur_chunks), torch.cat(ref_chunks).detach()


def build_jacobian_basis(
    delta_by_layer: Dict[int, torch.Tensor],
    refs: Dict[int, torch.Tensor],
    curr: Dict[int, torch.Tensor],
    *,
    probe_count: int,
    include_drift_probe: bool,
    seed: int,
) -> List[torch.Tensor]:
    # Low-rank basis for the row space of d selected SAE activations / d Delta.
    delta_items = _ordered_delta_items(delta_by_layer)
    variables = [v for _, v in delta_items]
    if not variables:
        return []
    z_vec, ref_vec = _flat_selected_state(refs, curr)
    if z_vec.numel() == 0:
        return []
    probes: List[torch.Tensor] = []
    if include_drift_probe:
        drift = (z_vec - ref_vec).detach()
        drift_norm = drift.norm(p=2)
        if float(drift_norm.item()) > 1e-8:
            probes.append(drift / drift_norm)
    gen = torch.Generator(device=z_vec.device)
    gen.manual_seed(int(seed))
    for _ in range(max(0, int(probe_count))):
        r = torch.randn(z_vec.shape, generator=gen, device=z_vec.device, dtype=z_vec.dtype)
        r = r / (r.norm(p=2) + 1e-8)
        probes.append(r)
    basis: List[torch.Tensor] = []
    for r in probes:
        scalar = (z_vec * r).sum()
        grads = torch.autograd.grad(scalar, variables, retain_graph=True, allow_unused=True)
        chunks = []
        for var, grad in zip(variables, grads):
            chunks.append((torch.zeros_like(var) if grad is None else grad).float().reshape(-1))
        q = torch.cat(chunks).detach()
        for b in basis:
            q = q - torch.dot(q, b) * b
        q_norm = q.norm(p=2)
        if float(q_norm.item()) > 1e-7:
            basis.append(q / q_norm)
    return basis


def project_grads_against_basis(delta_by_layer: Dict[int, torch.Tensor], basis: Sequence[torch.Tensor]):
    if not basis:
        return
    delta_items = _ordered_delta_items(delta_by_layer)
    g = _flat_grad(delta_items)
    for b in basis:
        b = b.to(g.device, dtype=g.dtype)
        g = g - torch.dot(g, b) * b
    _assign_flat_grad(delta_items, g)


def project_deltas_in_place(delta_by_layer: Dict[int, torch.Tensor], saes, feature_idx_by_layer, ridge: float):
    with torch.no_grad():
        for layer, delta in delta_by_layer.items():
            sae = saes[f"blocks.{int(layer)}.hook_resid_post"]
            feature_idx = feature_idx_by_layer[int(layer)]
            rows = [project_to_encoder_null(row.float(), sae, feature_idx, ridge=ridge) for row in delta.data]
            delta.data.copy_(torch.stack(rows, dim=0).to(delta.device, dtype=delta.dtype))


def project_grads_in_place(delta_by_layer: Dict[int, torch.Tensor], saes, feature_idx_by_layer, ridge: float):
    with torch.no_grad():
        for layer, delta in delta_by_layer.items():
            if delta.grad is None:
                continue
            sae = saes[f"blocks.{int(layer)}.hook_resid_post"]
            feature_idx = feature_idx_by_layer[int(layer)]
            rows = [project_to_encoder_null(row.float(), sae, feature_idx, ridge=ridge) for row in delta.grad]
            delta.grad.copy_(torch.stack(rows, dim=0).to(delta.grad.device, dtype=delta.grad.dtype))


class CaptureSelectedFeatures:
    def __init__(
        self,
        *,
        mod,
        sae,
        layer: int,
        feature_idx: torch.Tensor,
        positions: Sequence[int],
        state: Dict[int, torch.Tensor],
        detach: bool = True,
    ):
        self.mod = mod
        self.sae = sae
        self.layer = int(layer)
        self.feature_idx = feature_idx
        self.positions = [int(p) for p in positions]
        self.state = state
        self.detach = bool(detach)

    def __call__(self, resid: torch.Tensor, hook=None, **kwargs):
        z = self.sae.encode(cast_for_sae(resid, self.sae)).float()[0, :, self.feature_idx]
        if self.positions:
            valid_positions = [p for p in self.positions if 0 <= p < z.shape[0]]
            z = z[valid_positions] if valid_positions else z[:0]
        z = z.float()
        self.state[self.layer] = z.detach() if self.detach else z
        return resid


class MultiLayerDeltaHook:
    def __init__(
        self,
        *,
        layer: int,
        delta: Optional[torch.Tensor],
        positions: Sequence[int],
        fixed_delta: Optional[torch.Tensor],
        fixed_positions: Sequence[int],
    ):
        self.layer = int(layer)
        self.delta = delta
        self.positions = [int(p) for p in positions]
        self.fixed_delta = fixed_delta
        self.fixed_positions = [int(p) for p in fixed_positions]

    def _apply(self, out: torch.Tensor, delta: Optional[torch.Tensor], positions: Sequence[int]):
        if delta is None:
            return
        values = delta.to(out.device, dtype=torch.float32)
        for row_idx, pos in enumerate(positions):
            if row_idx >= values.shape[0]:
                break
            if 0 <= pos < out.shape[1]:
                out[:, pos, :] = out[:, pos, :] + values[row_idx].unsqueeze(0)

    def __call__(self, resid: torch.Tensor, hook=None, **kwargs):
        out = resid.float().clone()
        self._apply(out, self.fixed_delta, self.fixed_positions)
        self._apply(out, self.delta, self.positions)
        return out.to(resid.dtype)


def capture_postclamp_refs(mod, model, saes, circuit, runtime, clamp_hook, tokens, positions):
    state: Dict[int, torch.Tensor] = {}
    hooks = [(runtime["resid_name_filter"], clamp_hook)]
    for layer, feats in sorted(circuit.items()):
        hook_name = f"blocks.{int(layer)}.hook_resid_post"
        hooks.append(
            (
                hook_name,
                CaptureSelectedFeatures(
                    mod=mod,
                    sae=saes[hook_name],
                    layer=int(layer),
                    feature_idx=torch.tensor(feats, device=tokens.device, dtype=torch.long),
                    positions=positions,
                    state=state,
                ),
            )
        )
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return logits, state


def capture_with_deltas(
    mod,
    model,
    saes,
    circuit,
    runtime,
    clamp_hook,
    tokens,
    capture_positions,
    delta_by_layer: Optional[Dict[int, torch.Tensor]] = None,
    delta_positions: Optional[Sequence[int]] = None,
    fixed_delta_by_layer: Optional[Dict[int, torch.Tensor]] = None,
    fixed_positions: Optional[Sequence[int]] = None,
    delta_layers: Optional[Sequence[int]] = None,
    requires_grad: bool = False,
):
    capture_state: Dict[int, torch.Tensor] = {}
    hooks = [(runtime["resid_name_filter"], clamp_hook)]
    delta_by_layer = delta_by_layer or {}
    fixed_delta_by_layer = fixed_delta_by_layer or {}
    delta_positions = list(delta_positions or [])
    fixed_positions = list(fixed_positions or [])
    delta_hook_layers = sorted(set(int(x) for x in (delta_layers or [])) | set(int(x) for x in delta_by_layer) | set(int(x) for x in fixed_delta_by_layer))
    for layer in sorted(set(int(k) for k in circuit) | set(delta_hook_layers)):
        hook_name = f"blocks.{int(layer)}.hook_resid_post"
        if int(layer) in delta_hook_layers:
            hooks.append((hook_name, MultiLayerDeltaHook(layer=int(layer), delta=delta_by_layer.get(int(layer)), positions=delta_positions, fixed_delta=fixed_delta_by_layer.get(int(layer)), fixed_positions=fixed_positions)))
        if int(layer) in circuit:
            feats = circuit[int(layer)]
            hooks.append((hook_name, CaptureSelectedFeatures(mod=mod, sae=saes[hook_name], layer=int(layer), feature_idx=torch.tensor(feats, device=tokens.device, dtype=torch.long), positions=capture_positions, state=capture_state, detach=not requires_grad)))
    if requires_grad:
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    else:
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return logits, capture_state


def drift_from_refs(refs: Dict[int, torch.Tensor], curr: Dict[int, torch.Tensor]) -> Dict[str, Any]:
    per_layer = {}
    all_ref = []
    all_curr = []
    for layer in sorted(refs):
        if layer not in curr:
            continue
        r = refs[layer].float()
        c = curr[layer].float()
        if r.numel() == 0 or c.numel() == 0:
            continue
        diff = c - r
        floor_gap = torch.relu(r - c)
        per_layer[str(layer)] = {
            "l2": float(diff.norm(p=2).item()),
            "rmse": float(diff.pow(2).mean().sqrt().item()),
            "linf": float(diff.abs().max().item()),
            "floor_violation_l2": float(floor_gap.norm(p=2).item()),
            "floor_violation_rmse": float(floor_gap.pow(2).mean().sqrt().item()),
            "floor_violation_mean": float(floor_gap.mean().item()),
            "floor_violation_relative_l2": float(floor_gap.norm(p=2).item() / (r.norm(p=2).item() + 1e-8)),
            "floor_violation_frac": float((floor_gap > 1e-6).float().mean().item()),
            "n_values": int(diff.numel()),
        }
        all_ref.append(r.reshape(-1))
        all_curr.append(c.reshape(-1))
    if not all_ref:
        return {"l2": None, "rmse": None, "linf": None, "floor_violation_l2": None, "floor_violation_rmse": None, "floor_violation_mean": None, "floor_violation_relative_l2": None, "floor_violation_frac": None, "per_layer": per_layer, "n_values": 0}
    ref_vec = torch.cat(all_ref)
    curr_vec = torch.cat(all_curr)
    diff = curr_vec - ref_vec
    floor_gap = torch.relu(ref_vec - curr_vec)
    return {
        "l2": float(diff.norm(p=2).item()),
        "rmse": float(diff.pow(2).mean().sqrt().item()),
        "linf": float(diff.abs().max().item()),
        "relative_l2": float(diff.norm(p=2).item() / (ref_vec.norm(p=2).item() + 1e-8)),
        "floor_violation_l2": float(floor_gap.norm(p=2).item()),
        "floor_violation_rmse": float(floor_gap.pow(2).mean().sqrt().item()),
        "floor_violation_mean": float(floor_gap.mean().item()),
        "floor_violation_relative_l2": float(floor_gap.norm(p=2).item() / (ref_vec.norm(p=2).item() + 1e-8)),
        "floor_violation_frac": float((floor_gap > 1e-6).float().mean().item()),
        "n_values": int(diff.numel()),
        "per_layer": per_layer,
    }


def activation_drift_loss(
    refs: Dict[int, torch.Tensor],
    curr: Dict[int, torch.Tensor],
    *,
    mode: str,
    eps: float,
) -> torch.Tensor:
    device = next((v.device for v in curr.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))
    total = torch.zeros((), device=device, dtype=torch.float32)
    n_layers = 0
    for layer in sorted(refs):
        if layer not in curr or curr[layer].numel() == 0 or refs[layer].numel() == 0:
            continue
        ref = refs[layer].to(curr[layer].device, dtype=torch.float32)
        cur = curr[layer].float()
        mse = (cur - ref).pow(2).mean()
        if mode == "raw_mse":
            layer_loss = mse
        elif mode == "relative_mse":
            layer_loss = mse / (ref.pow(2).mean().detach() + float(eps))
        elif mode == "relative_l2":
            layer_loss = (cur - ref).pow(2).sum() / (ref.pow(2).sum().detach() + float(eps))
        else:
            raise ValueError(f"unknown act_loss_mode={mode}")
        total = total + layer_loss
        n_layers += 1
    if n_layers == 0:
        return total
    return total / float(n_layers)


def make_delta_dict(layers: Sequence[int], n_positions: int, d_model: int, device: str):
    return {
        int(layer): torch.zeros(n_positions, d_model, device=device, dtype=torch.float32, requires_grad=True)
        for layer in layers
    }


def select_circuit(mod, feature_cache, args) -> Dict[int, List[int]]:
    if args.single_layer is not None:
        if args.single_features:
            feats = [int(x) for x in args.single_features.split(",") if x.strip()]
        else:
            base = mod.select_feature_circuit(
                feature_cache,
                feature_source=args.feature_source,
                dataset_name=args.dataset_name,
                feature_scope=args.feature_scope,
            )
            feats = [int(x) for x in base[int(args.single_layer)]]
            if args.single_feature_top_k > 0:
                feats = feats[: int(args.single_feature_top_k)]
        return {int(args.single_layer): feats}
    return mod.select_feature_circuit(
        feature_cache,
        feature_source=args.feature_source,
        dataset_name=args.dataset_name,
        feature_scope=args.feature_scope,
    )


def optimize_multilayer_delta(
    *,
    mod,
    model,
    saes,
    circuit,
    runtime,
    clamp_hook,
    tokens,
    objective_fn,
    refs,
    capture_positions,
    delta_positions,
    fixed_delta_by_layer=None,
    fixed_positions=None,
    delta_layers: Optional[Sequence[int]] = None,
    projection_mode: str = "encoder",
    num_steps: int,
    lr: float,
    max_delta_norm: float,
    lambda_act: float,
    act_loss_mode: str,
    act_loss_eps: float,
    best_checkpoint_mode: str,
    best_checkpoint_every: int,
    best_drift_relative_l2_threshold: float,
    best_drift_penalty_weight: float,
    ridge: float,
    seed: int,
    jacobian_probe_count: int = 0,
    jacobian_include_drift_probe: bool = False,
    jacobian_correction_steps: int = 0,
    jacobian_correction_lr: float = 0.05,
    jacobian_correction_every: int = 1,
):
    torch.manual_seed(seed)
    monitor_layers = sorted(int(k) for k in circuit)
    layers = sorted(int(k) for k in (delta_layers if delta_layers is not None else monitor_layers))
    feature_idx_by_layer = {int(layer): torch.tensor(circuit[int(layer)], device=tokens.device, dtype=torch.long) for layer in monitor_layers}
    if projection_mode == "encoder" and any(layer not in feature_idx_by_layer for layer in layers):
        raise ValueError("encoder projection requires delta layers to be part of the monitored circuit")
    delta_by_layer = make_delta_dict(layers, len(delta_positions), model.cfg.d_model, str(tokens.device))
    if projection_mode == "encoder":
        project_deltas_in_place(delta_by_layer, saes, feature_idx_by_layer, ridge)
    optimizer = torch.optim.Adam(list(delta_by_layer.values()), lr=lr)
    history = []
    best: Optional[Dict[str, Any]] = None

    def snapshot_checkpoint(step: int, label: str):
        nonlocal best
        logits_eval, state_eval = capture_with_deltas(
            mod,
            model,
            saes,
            circuit,
            runtime,
            clamp_hook,
            tokens,
            capture_positions,
            delta_by_layer=delta_by_layer,
            delta_positions=delta_positions,
            fixed_delta_by_layer=fixed_delta_by_layer,
            fixed_positions=fixed_positions,
            delta_layers=layers,
            requires_grad=False,
        )
        objective_eval = float(objective_fn(logits_eval).detach().item())
        drift_eval = drift_from_refs(refs, state_eval)
        rel = drift_eval.get("relative_l2")
        rel_float = float(rel) if isinstance(rel, (int, float)) and math.isfinite(float(rel)) else float("inf")
        if best_checkpoint_mode == "final":
            return
        if best_checkpoint_mode == "drift_constrained":
            feasible = rel_float <= float(best_drift_relative_l2_threshold)
            score = objective_eval if feasible else -float("inf")
            fallback_score = objective_eval - float(best_drift_penalty_weight) * rel_float
        elif best_checkpoint_mode == "penalty_score":
            feasible = True
            excess = max(0.0, rel_float - float(best_drift_relative_l2_threshold))
            score = objective_eval - float(best_drift_penalty_weight) * excess
            fallback_score = score
        else:
            raise ValueError(f"unknown best_checkpoint_mode={best_checkpoint_mode}")
        if best is None:
            should_update = True
        elif score == -float("inf") and best.get("score") == -float("inf"):
            should_update = fallback_score > float(best.get("fallback_score", -float("inf")))
        else:
            should_update = score > float(best.get("score", -float("inf")))
        if should_update:
            best = {
                "step": int(step),
                "label": label,
                "score": float(score),
                "fallback_score": float(fallback_score),
                "objective": objective_eval,
                "drift": drift_eval,
                "feasible": bool(feasible),
                "delta_by_layer": {int(k): v.detach().cpu().clone() for k, v in delta_by_layer.items()},
            }

    for step in range(num_steps):
        model.reset_hooks()
        model.zero_grad(set_to_none=True)
        logits, state = capture_with_deltas(
            mod,
            model,
            saes,
            circuit,
            runtime,
            clamp_hook,
            tokens,
            capture_positions,
            delta_by_layer=delta_by_layer,
            delta_positions=delta_positions,
            fixed_delta_by_layer=fixed_delta_by_layer,
            fixed_positions=fixed_positions,
            delta_layers=layers,
            requires_grad=True,
        )
        objective = objective_fn(logits)
        drift = drift_from_refs(refs, state)
        drift_penalty = activation_drift_loss(
            refs,
            state,
            mode=act_loss_mode,
            eps=act_loss_eps,
        ).to(tokens.device)
        loss = -objective + float(lambda_act) * drift_penalty
        optimizer.zero_grad(set_to_none=True)
        if projection_mode == "jacobian":
            basis = build_jacobian_basis(delta_by_layer, refs, state, probe_count=jacobian_probe_count, include_drift_probe=jacobian_include_drift_probe, seed=int(seed) * 100000 + int(step))
            (-objective).backward(retain_graph=True)
            project_grads_against_basis(delta_by_layer, basis)
            if float(lambda_act) != 0.0:
                (float(lambda_act) * drift_penalty).backward()
        else:
            loss.backward()
            if projection_mode == "encoder":
                project_grads_in_place(delta_by_layer, saes, feature_idx_by_layer, ridge)
            elif projection_mode != "none":
                raise ValueError(f"unknown projection_mode={projection_mode}")
        optimizer.step()
        if projection_mode == "encoder":
            project_deltas_in_place(delta_by_layer, saes, feature_idx_by_layer, ridge)
        scale_deltas_to_norm(delta_by_layer, max_delta_norm)
        if (
            projection_mode == "jacobian"
            and int(jacobian_correction_steps) > 0
            and int(jacobian_correction_every) > 0
            and ((step + 1) % int(jacobian_correction_every) == 0)
        ):
            for _correction_step in range(int(jacobian_correction_steps)):
                model.reset_hooks()
                model.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                _, correction_state = capture_with_deltas(
                    mod,
                    model,
                    saes,
                    circuit,
                    runtime,
                    clamp_hook,
                    tokens,
                    capture_positions,
                    delta_by_layer=delta_by_layer,
                    delta_positions=delta_positions,
                    fixed_delta_by_layer=fixed_delta_by_layer,
                    fixed_positions=fixed_positions,
                    delta_layers=layers,
                    requires_grad=True,
                )
                correction_loss = activation_drift_loss(
                    refs,
                    correction_state,
                    mode=act_loss_mode,
                    eps=act_loss_eps,
                ).to(tokens.device)
                correction_loss.backward()
                with torch.no_grad():
                    for _layer, _delta in delta_by_layer.items():
                        if _delta.grad is not None:
                            _delta.data.add_(_delta.grad, alpha=-float(jacobian_correction_lr))
                scale_deltas_to_norm(delta_by_layer, max_delta_norm)
                optimizer.zero_grad(set_to_none=True)
        if step == 0 or step == num_steps - 1 or (step + 1) % 20 == 0:
            history.append(
                {
                    "step": int(step),
                    "objective": float(objective.detach().item()),
                    "loss": float(loss.detach().item()),
                    "act_loss": float(drift_penalty.detach().item()),
                    "drift_l2": drift["l2"],
                    "drift_relative_l2": drift.get("relative_l2"),
                    "drift_rmse": drift["rmse"],
                    "delta_norm": float(total_delta_norm(delta_by_layer).detach().item()),
                }
            )
        if best_checkpoint_mode != "final" and int(best_checkpoint_every) > 0 and (
            step == 0 or step == num_steps - 1 or (step + 1) % int(best_checkpoint_every) == 0
        ):
            snapshot_checkpoint(step, "periodic")
    if best is not None:
        with torch.no_grad():
            for layer, value in best["delta_by_layer"].items():
                delta_by_layer[int(layer)].data.copy_(value.to(delta_by_layer[int(layer)].device, dtype=delta_by_layer[int(layer)].dtype))
    final_logits, final_state = capture_with_deltas(
        mod,
        model,
        saes,
        circuit,
        runtime,
        clamp_hook,
        tokens,
        capture_positions,
        delta_by_layer=delta_by_layer,
        delta_positions=delta_positions,
        fixed_delta_by_layer=fixed_delta_by_layer,
        fixed_positions=fixed_positions,
        requires_grad=False,
    )
    final_objective = objective_fn(final_logits)
    final_drift = drift_from_refs(refs, final_state)
    return {
        "delta_by_layer": {int(k): v.detach().cpu() for k, v in delta_by_layer.items()},
        "final_objective": float(final_objective.detach().item()),
        "final_drift": final_drift,
        "final_delta_norm": float(total_delta_norm(delta_by_layer).detach().item()),
        "best_checkpoint": {
            k: v for k, v in (best or {}).items() if k != "delta_by_layer"
        } if best is not None else None,
        "history": history,
    }


def run_sample(mod, model, saes, runtime, feature_cache, args, row, sample_idx):
    circuit = select_circuit(mod, feature_cache, args)
    layers = sorted(int(k) for k in circuit)
    requested_delta_layers = [int(x) for x in str(args.delta_layers).split(",") if str(x).strip()] or None
    mod.ensure_sae_layers_loaded(saes, model_name=args.model_name, layers=layers, device=args.device, keep_only=True)
    clamp_hook = partial(mod.clamp_sae_safe, saes=saes, circuit=circuit, val=args.clamp_value, multiply=True, ind=False)
    extra_fwd_hooks = [(runtime["resid_name_filter"], clamp_hook)]

    formatted_prompt = runtime["format_prompt"](model.tokenizer, row["instruction"])
    base_target = row.get("target_answer") or row.get("base_response")
    preflight = mod.evaluate_preflight_sample(
        model=model,
        extra_fwd_hooks=extra_fwd_hooks,
        instruction=row["instruction"],
        target_answer=base_target,
        max_new_tokens=args.max_new_tokens,
    )
    target_answer = preflight["base_response"] if args.recovery_target_mode == "base_response_valid_case" else base_target
    prompt_tokens = model.to_tokens(formatted_prompt)
    prompt_len = int(prompt_tokens.shape[1])
    tokens = model.to_tokens(formatted_prompt + target_answer)
    answer_positions = list(range(prompt_len, min(int(tokens.shape[1]), prompt_len + int(args.answer_token_limit))))
    if not answer_positions:
        answer_positions = list(range(prompt_len, int(tokens.shape[1])))
    prompt_anchor = max(prompt_len - 1, 0)

    refs_logits, refs = capture_postclamp_refs(mod, model, saes, circuit, runtime, clamp_hook, tokens, answer_positions)
    refusal_prefix_token_ids = mod.build_refusal_prefix_token_ids(model.tokenizer)
    safety_prefix_token_ids = mod.build_safety_cue_prefix_token_ids(model.tokenizer) if args.discourage_safety_prefixes else []
    discouraged_prefix_token_ids = list(refusal_prefix_token_ids) + list(safety_prefix_token_ids)
    decision_tokens = model.to_tokens(formatted_prompt + target_answer)
    decision_prefix_token_ids = [int(x) for x in decision_tokens[0, prompt_len : prompt_len + int(args.target_prefix_token_limit)].tolist()]
    stage1_objective = mod.build_recovery_objective(
        tokens,
        answer_start=prompt_len,
        decision_prefix_token_ids=decision_prefix_token_ids,
        refusal_prefix_token_ids=refusal_prefix_token_ids,
        discouraged_prefix_token_ids=discouraged_prefix_token_ids,
        anti_refusal_weight=args.boundary_stage1_anti_refusal_weight,
        anti_refusal_prefix_len=args.anti_refusal_prefix_len,
        first_token_margin_weight=args.boundary_stage1_first_token_margin_weight,
        first_token_logprob_weight=args.boundary_stage1_first_token_logprob_weight,
        target_prefix_logprob_weight=args.boundary_stage1_target_prefix_logprob_weight,
        target_prefix_token_limit=args.target_prefix_token_limit,
        answer_logprob_weight=0.0,
    )
    stage2_objective = mod.build_recovery_objective(
        tokens,
        answer_start=prompt_len,
        decision_prefix_token_ids=decision_prefix_token_ids,
        refusal_prefix_token_ids=refusal_prefix_token_ids,
        discouraged_prefix_token_ids=discouraged_prefix_token_ids,
        anti_refusal_weight=args.anti_refusal_weight,
        anti_refusal_prefix_len=args.anti_refusal_prefix_len,
        first_token_margin_weight=0.0,
        first_token_logprob_weight=0.0,
        target_prefix_logprob_weight=0.0,
        target_prefix_token_limit=args.target_prefix_token_limit,
        answer_logprob_weight=args.answer_logprob_weight,
        answer_token_limit=args.answer_token_limit,
        answer_prefix_token_limit=args.answer_prefix_token_limit,
        answer_prefix_token_weight=args.answer_prefix_token_weight,
    )
    stage1 = optimize_multilayer_delta(
        mod=mod,
        model=model,
        saes=saes,
        circuit=circuit,
        runtime=runtime,
        clamp_hook=clamp_hook,
        tokens=tokens,
        objective_fn=stage1_objective,
        refs=refs,
        capture_positions=answer_positions,
        delta_positions=[prompt_anchor],
        delta_layers=requested_delta_layers,
        projection_mode=args.projection_mode,
        num_steps=args.boundary_stage1_steps,
        lr=args.boundary_stage1_lr,
        max_delta_norm=args.boundary_stage1_max_delta_norm,
        lambda_act=args.lambda_act,
        act_loss_mode=args.act_loss_mode,
        act_loss_eps=args.act_loss_eps,
        best_checkpoint_mode=args.best_checkpoint_mode,
        best_checkpoint_every=args.best_checkpoint_every,
        best_drift_relative_l2_threshold=args.best_drift_relative_l2_threshold,
        best_drift_penalty_weight=args.best_drift_penalty_weight,
        ridge=args.ridge,
        seed=sample_idx + args.seed,
        jacobian_probe_count=args.jacobian_probe_count,
        jacobian_include_drift_probe=args.jacobian_include_drift_probe,
        jacobian_correction_steps=args.jacobian_correction_steps,
        jacobian_correction_lr=args.jacobian_correction_lr,
        jacobian_correction_every=args.jacobian_correction_every,
    )
    fixed_delta_by_layer = {int(k): v.to(tokens.device) for k, v in stage1["delta_by_layer"].items()}
    if args.stage2_delta_scope == "answer":
        stage2_delta_positions = answer_positions
    elif args.stage2_delta_scope == "prompt_anchor":
        stage2_delta_positions = [prompt_anchor]
    elif args.stage2_delta_scope == "prompt_and_answer":
        stage2_delta_positions = [prompt_anchor] + answer_positions
    else:
        raise ValueError(f"unknown stage2_delta_scope={args.stage2_delta_scope}")

    stage2 = optimize_multilayer_delta(
        mod=mod,
        model=model,
        saes=saes,
        circuit=circuit,
        runtime=runtime,
        clamp_hook=clamp_hook,
        tokens=tokens,
        objective_fn=stage2_objective,
        refs=refs,
        capture_positions=answer_positions,
        delta_positions=stage2_delta_positions,
        delta_layers=requested_delta_layers,
        fixed_delta_by_layer=fixed_delta_by_layer,
        fixed_positions=[prompt_anchor],
        projection_mode=args.projection_mode,
        num_steps=args.num_steps,
        lr=args.lr,
        max_delta_norm=args.max_delta_norm,
        lambda_act=args.lambda_act,
        act_loss_mode=args.act_loss_mode,
        act_loss_eps=args.act_loss_eps,
        best_checkpoint_mode=args.best_checkpoint_mode,
        best_checkpoint_every=args.best_checkpoint_every,
        best_drift_relative_l2_threshold=args.best_drift_relative_l2_threshold,
        best_drift_penalty_weight=args.best_drift_penalty_weight,
        ridge=args.ridge,
        seed=sample_idx + args.seed,
        jacobian_probe_count=args.jacobian_probe_count,
        jacobian_include_drift_probe=args.jacobian_include_drift_probe,
        jacobian_correction_steps=args.jacobian_correction_steps,
        jacobian_correction_lr=args.jacobian_correction_lr,
        jacobian_correction_every=args.jacobian_correction_every,
    )
    stage2_delta_by_layer = {int(k): v.to(tokens.device) for k, v in stage2["delta_by_layer"].items()}
    hooks = [(runtime["resid_name_filter"], clamp_hook)]
    final_delta_layers = sorted(set(layers) | set(int(k) for k in stage2_delta_by_layer) | set(int(k) for k in fixed_delta_by_layer))
    for layer in final_delta_layers:
        hooks.append((f"blocks.{int(layer)}.hook_resid_post", MultiLayerDeltaHook(layer=int(layer), delta=stage2_delta_by_layer.get(int(layer)), positions=answer_positions, fixed_delta=fixed_delta_by_layer.get(int(layer)), fixed_positions=[prompt_anchor])))
    with model.hooks(fwd_hooks=hooks):
        recovered_full = model.generate(formatted_prompt, max_new_tokens=args.max_new_tokens, temperature=0.0, verbose=False)
    recovered_response = mod.extract_new_text(recovered_full, formatted_prompt)
    row_out = {
        "sample_idx": sample_idx,
        "instruction": row["instruction"],
        "target_answer": target_answer,
        "base_response": preflight["base_response"],
        "clamped_response": preflight["clamped_response"],
        "recovered_response": recovered_response,
        "base_refusal": bool(preflight["base_refusal"]),
        "clamped_refusal": bool(preflight["clamped_refusal"]),
        "recovered_refusal": bool(runtime["substring_matching_judge_fn"](recovered_response)),
        "recovered_safety_cue": bool(mod.contains_safety_cue(recovered_response)),
        "recovered_cooperative": (not bool(runtime["substring_matching_judge_fn"](recovered_response))) and (not bool(mod.contains_safety_cue(recovered_response))),
        "valid_recovery_case": bool((not preflight["base_refusal"]) and preflight["clamped_refusal"]),
        "feature_source": args.feature_source,
        "feature_scope": args.feature_scope,
        "circuit": {str(k): [int(x) for x in v] for k, v in sorted(circuit.items())},
        "n_circuit_features": sum(len(v) for v in circuit.values()),
        "projection_mode": args.projection_mode,
        "delta_layers": requested_delta_layers or layers,
        "jacobian_probe_count": args.jacobian_probe_count,
        "jacobian_include_drift_probe": args.jacobian_include_drift_probe,
        "jacobian_correction_steps": args.jacobian_correction_steps,
        "jacobian_correction_lr": args.jacobian_correction_lr,
        "jacobian_correction_every": args.jacobian_correction_every,
        "recovery_mode": "single_layer" if args.single_layer is not None else "multi_layer",
        "stage1_final_objective": stage1["final_objective"],
        "stage1_delta_norm": stage1["final_delta_norm"],
        "stage1_drift": stage1["final_drift"],
        "stage1_best_checkpoint": stage1.get("best_checkpoint"),
        "stage2_final_objective": stage2["final_objective"],
        "stage2_delta_norm": stage2["final_delta_norm"],
        "stage2_delta_scope": args.stage2_delta_scope,
        "stage2_delta_positions": stage2_delta_positions,
        "final_drift": stage2["final_drift"],
        "stage2_best_checkpoint": stage2.get("best_checkpoint"),
        "history": {"stage1": stage1["history"], "stage2": stage2["history"]},
    }
    return mod.annotate_recovery_quality(row_out)


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_clamp = sum(1 for r in rows if r.get("clamped_refusal"))
    n_nonref = sum(1 for r in rows if not r.get("recovered_refusal"))
    n_base_like = sum(
        1 for r in rows if bool(r.get("recovered_base_like") or r.get("base_like_recovery"))
    )
    n_strict_base_like = sum(
        1
        for r in rows
        if bool(r.get("recovered_strict_base_like") or r.get("strict_base_like_recovery"))
    )
    return {
        "n_samples": n,
        "n_clamp_refusal": n_clamp,
        "n_recovered_nonrefusal": n_nonref,
        "n_recovered_base_like": n_base_like,
        "n_recovered_strict_base_like": n_strict_base_like,
        "nonrefusal_rate": n_nonref / n if n else None,
        "base_like_rate": n_base_like / n if n else None,
        "avg_final_drift_l2": safe_mean([r["final_drift"]["l2"] for r in rows]),
        "avg_final_drift_rmse": safe_mean([r["final_drift"]["rmse"] for r in rows]),
        "avg_final_drift_relative_l2": safe_mean([r["final_drift"].get("relative_l2") for r in rows]),
        "avg_final_floor_violation_l2": safe_mean([r["final_drift"].get("floor_violation_l2") for r in rows]),
        "avg_final_floor_violation_rmse": safe_mean([r["final_drift"].get("floor_violation_rmse") for r in rows]),
        "avg_final_floor_violation_mean": safe_mean([r["final_drift"].get("floor_violation_mean") for r in rows]),
        "avg_final_floor_violation_relative_l2": safe_mean([r["final_drift"].get("floor_violation_relative_l2") for r in rows]),
        "avg_final_floor_violation_frac": safe_mean([r["final_drift"].get("floor_violation_frac") for r in rows]),
        "avg_stage2_delta_norm": safe_mean([r["stage2_delta_norm"] for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="advbench")
    parser.add_argument("--model_name", default="gemma-2b")
    parser.add_argument("--feature_source", default="benchmark_our")
    parser.add_argument("--feature_scope", default="global")
    parser.add_argument("--target_pairs_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=24)
    parser.add_argument("--clamp_value", type=float, default=3.0)
    parser.add_argument("--projection_mode", choices=["encoder", "none", "jacobian"], default="encoder")
    parser.add_argument("--delta_layers", default="")
    parser.add_argument("--jacobian_probe_count", type=int, default=0)
    parser.add_argument("--jacobian_include_drift_probe", action="store_true")
    parser.add_argument("--jacobian_correction_steps", type=int, default=0)
    parser.add_argument("--jacobian_correction_lr", type=float, default=0.05)
    parser.add_argument("--jacobian_correction_every", type=int, default=1)
    parser.add_argument("--single_layer", type=int, default=None)
    parser.add_argument("--single_features", default="")
    parser.add_argument("--single_feature_top_k", type=int, default=1)
    parser.add_argument("--recovery_target_mode", choices=["provided_target", "base_response_valid_case"], default="provided_target")
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max_delta_norm", type=float, default=40.0)
    parser.add_argument("--lambda_act", type=float, default=0.02)
    parser.add_argument("--act_loss_mode", choices=["raw_mse", "relative_mse", "relative_l2"], default="relative_mse")
    parser.add_argument("--act_loss_eps", type=float, default=1e-6)
    parser.add_argument("--best_checkpoint_mode", choices=["final", "drift_constrained", "penalty_score"], default="drift_constrained")
    parser.add_argument("--best_checkpoint_every", type=int, default=10)
    parser.add_argument("--best_drift_relative_l2_threshold", type=float, default=0.25)
    parser.add_argument("--best_drift_penalty_weight", type=float, default=10.0)
    parser.add_argument("--stage2_delta_scope", choices=["answer", "prompt_anchor", "prompt_and_answer"], default="answer")
    parser.add_argument("--answer_logprob_weight", type=float, default=5.0)
    parser.add_argument("--answer_token_limit", type=int, default=48)
    parser.add_argument("--answer_prefix_token_limit", type=int, default=0)
    parser.add_argument("--answer_prefix_token_weight", type=float, default=1.0)
    parser.add_argument("--anti_refusal_weight", type=float, default=0.1)
    parser.add_argument("--anti_refusal_prefix_len", type=int, default=8)
    parser.add_argument("--target_prefix_token_limit", type=int, default=8)
    parser.add_argument("--boundary_stage1_steps", type=int, default=20)
    parser.add_argument("--boundary_stage1_lr", type=float, default=0.08)
    parser.add_argument("--boundary_stage1_max_delta_norm", type=float, default=30.0)
    parser.add_argument("--boundary_stage1_anti_refusal_weight", type=float, default=0.2)
    parser.add_argument("--boundary_stage1_first_token_margin_weight", type=float, default=0.1)
    parser.add_argument("--boundary_stage1_first_token_logprob_weight", type=float, default=4.0)
    parser.add_argument("--boundary_stage1_target_prefix_logprob_weight", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--discourage_safety_prefixes", action="store_true")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = load_baseline_module()
    feature_cache = mod.load_pickle(ROOT.parent / "cache" / f"benchmark_{args.model_name}_feats.pkl")
    model, saes = mod.load_model_and_saes(args.model_name, args.device)
    runtime = mod.get_runtime()
    rows = json.load(open(args.target_pairs_json, "r", encoding="utf-8"))
    rows = rows[: args.max_samples] if args.max_samples > 0 else rows
    sample_rows = []
    out_dir = Path(args.output_dir)
    for idx, row in enumerate(rows):
        sample = run_sample(mod, model, saes, runtime, feature_cache, args, row, idx)
        sample_rows.append(sample)
        dump_json(sample_rows, out_dir / "samples.json")
        print(
            idx,
            "nonref",
            int(not sample["recovered_refusal"]),
            "base_like",
            int(bool(sample.get("recovered_base_like") or sample.get("base_like_recovery"))),
            "drift_l2",
            f"{sample['final_drift']['l2']:.4f}",
            flush=True,
        )
    aggregate = summarize(sample_rows)
    aggregate.update(
        {
            "dataset_name": args.dataset_name,
            "feature_source": args.feature_source,
            "feature_scope": args.feature_scope,
            "projection_mode": args.projection_mode,
            "jacobian_correction_steps": args.jacobian_correction_steps,
            "jacobian_correction_lr": args.jacobian_correction_lr,
            "jacobian_correction_every": args.jacobian_correction_every,
            "single_layer": args.single_layer,
            "single_features": args.single_features,
            "clamp_value": args.clamp_value,
            "lambda_act": args.lambda_act,
            "act_loss_mode": args.act_loss_mode,
            "best_checkpoint_mode": args.best_checkpoint_mode,
            "best_drift_relative_l2_threshold": args.best_drift_relative_l2_threshold,
            "best_drift_penalty_weight": args.best_drift_penalty_weight,
            "stage2_delta_scope": args.stage2_delta_scope,
            "target_pairs_json": args.target_pairs_json,
        }
    )
    dump_json(aggregate, out_dir / "aggregate.json")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
