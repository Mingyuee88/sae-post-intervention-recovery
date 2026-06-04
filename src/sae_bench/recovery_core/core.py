from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from sae_lens import SAE


def sae_input_dtype(sae: SAE) -> torch.dtype:
    return getattr(getattr(sae, "W_enc", None), "dtype", torch.float32)


def cast_for_sae(x: torch.Tensor, sae: SAE) -> torch.Tensor:
    return x.to(dtype=sae_input_dtype(sae))


def feature_decode_contrib(sae: SAE, feature_idx: torch.Tensor, feature_acts: torch.Tensor) -> torch.Tensor:
    w = sae.W_dec[feature_idx].float()
    return feature_acts.float() @ w


def feature_decode_contrib_seq(sae: SAE, feature_idx: torch.Tensor, feature_acts_seq: torch.Tensor) -> torch.Tensor:
    w = sae.W_dec[feature_idx].float()
    return feature_acts_seq.float() @ w


def get_choice_token_ids(model, letters: List[str]) -> List[int]:
    return [model.to_single_token(" " + x) for x in letters]


def target_margin(logits_last: torch.Tensor, target_token_id: int) -> torch.Tensor:
    target = logits_last[target_token_id]
    mask = torch.ones_like(logits_last, dtype=torch.bool)
    mask[target_token_id] = False
    others_mean = logits_last[mask].mean()
    return target - others_mean


def choice_margin(logits_last: torch.Tensor, target_token_id: int, choice_token_ids: List[int]) -> torch.Tensor:
    choice_logits = logits_last[choice_token_ids]
    target_pos = choice_token_ids.index(target_token_id)
    target = choice_logits[target_pos]
    mask = torch.ones_like(choice_logits, dtype=torch.bool)
    mask[target_pos] = False
    others_mean = choice_logits[mask].mean()
    return target - others_mean


def choice_ce_loss(logits_last: torch.Tensor, target_token_id: int, choice_token_ids: List[int]) -> torch.Tensor:
    choice_logits = logits_last[choice_token_ids].unsqueeze(0)
    target_pos = torch.tensor([choice_token_ids.index(target_token_id)], device=logits_last.device)
    return F.cross_entropy(choice_logits, target_pos)


def objective_value(
    logits_last: torch.Tensor,
    target_token_id: int,
    choice_token_ids: Optional[List[int]],
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode == "vocab_margin":
        return target_margin(logits_last, target_token_id)
    if loss_mode == "choice_margin":
        if choice_token_ids is None:
            raise ValueError("choice_token_ids required for choice_margin")
        return choice_margin(logits_last, target_token_id, choice_token_ids)
    if loss_mode == "choice_ce":
        if choice_token_ids is None:
            raise ValueError("choice_token_ids required for choice_ce")
        return -choice_ce_loss(logits_last, target_token_id, choice_token_ids)
    raise ValueError(f"Unknown loss_mode: {loss_mode}")


def apply_feature_clamp(
    z: torch.Tensor,
    feature_idx: torch.Tensor,
    clamp_value: float,
    *,
    multiply: bool = False,
    positive_only: bool = True,
) -> torch.Tensor:
    z_def = z.clone()
    selected = z_def[..., feature_idx]
    if multiply:
        replacement = selected * float(clamp_value)
    else:
        replacement = torch.full_like(selected, float(clamp_value))
    if positive_only:
        selected = torch.where(selected > 0, replacement, selected)
    else:
        selected = replacement
    z_def[..., feature_idx] = selected
    return z_def


@torch.no_grad()
def build_sae_clamped_reference(
    model,
    sae: SAE,
    hook_name: str,
    tokens: torch.Tensor,
    feature_idx: torch.Tensor,
    clamp_value: float,
    *,
    multiply: bool = False,
    positive_only: bool = True,
    extra_fwd_hooks: Optional[List[tuple[str, Callable]]] = None,
):
    ref_state: Dict[str, Any] = {}

    def hook_fn(resid: torch.Tensor, hook=None, **kwargs):
        x_sae = cast_for_sae(resid, sae)
        z = sae.encode(x_sae)
        recon = sae.decode(z)
        error = resid.float() - recon.float()

        z_def = apply_feature_clamp(
            z,
            feature_idx,
            clamp_value,
            multiply=multiply,
            positive_only=positive_only,
        )
        recon_def = sae.decode(z_def)
        x_def = error + recon_def.float()

        z_seq_ref = sae.encode(cast_for_sae(x_def, sae)).float()[0, :, feature_idx]
        dec_seq_ref = feature_decode_contrib_seq(sae, feature_idx, z_seq_ref)
        x_last_def = x_def[:, -1, :]
        z_last = z_seq_ref[-1]
        dec_last = dec_seq_ref[-1]

        ref_state["x_def_all"] = x_def.detach().float()
        ref_state["x_def_last"] = x_last_def[0].detach().float()
        ref_state["act_ref_seq"] = z_seq_ref.detach().float()
        ref_state["act_ref"] = z_last.detach().float()
        ref_state["decode_ref_seq"] = dec_seq_ref.detach().float()
        ref_state["decode_ref"] = dec_last.detach().float()
        ref_state["z_def_last"] = z_def[0, -1, feature_idx].detach().float()
        return x_def.to(resid.dtype)

    fwd_hooks = list(extra_fwd_hooks or []) + [(hook_name, hook_fn)]
    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return logits, ref_state


@torch.no_grad()
def build_direct_defended_reference(
    model,
    sae: SAE,
    hook_name: str,
    tokens: torch.Tensor,
    feature_idx: torch.Tensor,
    multiplier: float,
):
    return build_sae_clamped_reference(
        model=model,
        sae=sae,
        hook_name=hook_name,
        tokens=tokens,
        feature_idx=feature_idx,
        clamp_value=-float(multiplier),
        multiply=False,
        positive_only=True,
    )


class FixedDirectDefendedPlusDeltaHook:
    def __init__(
        self,
        sae: SAE,
        feature_idx: torch.Tensor,
        defended_ref: Dict[str, torch.Tensor],
        delta_last: Optional[torch.Tensor],
        state: Optional[Dict[str, Any]] = None,
        delta_position: Optional[int] = None,
        delta_positions: Optional[List[int]] = None,
        fixed_delta: Optional[torch.Tensor] = None,
        fixed_delta_positions: Optional[List[int]] = None,
    ):
        self.sae = sae
        self.feature_idx = feature_idx
        self.defended_ref = defended_ref
        self.delta_last = delta_last
        self.state = state if state is not None else {}
        self.delta_position = delta_position
        self.delta_positions = [int(p) for p in delta_positions] if delta_positions is not None else None
        self.fixed_delta = fixed_delta
        self.fixed_delta_positions = [int(p) for p in fixed_delta_positions] if fixed_delta_positions is not None else None

    def __call__(self, resid: torch.Tensor, hook=None, **kwargs):
        out = resid.clone()
        seq_len = out.shape[1]
        x_def_full = self.defended_ref["x_def_all"].to(out.device, dtype=torch.float32)
        x_curr = x_def_full[:, :seq_len, :].clone()
        def apply_delta(delta_tensor, positions, single_position):
            if delta_tensor is None:
                return
            delta_values = delta_tensor.to(x_curr.device, dtype=x_curr.dtype)
            if delta_values.dim() == 1:
                delta_values = delta_values.unsqueeze(0)
            if positions is not None:
                if delta_values.shape[0] != len(positions):
                    raise ValueError(f"delta shape {tuple(delta_values.shape)} does not match delta_positions {positions}")
                for row_idx, pos in enumerate(positions):
                    delta_idx = int(pos)
                    if delta_idx < 0:
                        delta_idx = seq_len + delta_idx
                    if 0 <= delta_idx < seq_len:
                        x_curr[:, delta_idx, :] = x_curr[:, delta_idx, :] + delta_values[row_idx].unsqueeze(0)
            else:
                delta_idx = seq_len - 1 if single_position is None else int(single_position)
                if delta_idx < 0:
                    delta_idx = seq_len + delta_idx
                if 0 <= delta_idx < seq_len:
                    x_curr[:, delta_idx, :] = x_curr[:, delta_idx, :] + delta_values[0].unsqueeze(0)

        apply_delta(self.fixed_delta, self.fixed_delta_positions, None)
        apply_delta(self.delta_last, self.delta_positions, self.delta_position)

        z_seq = self.sae.encode(cast_for_sae(x_curr, self.sae)).float()[0, :, self.feature_idx]
        dec_seq = feature_decode_contrib_seq(self.sae, self.feature_idx, z_seq)
        x_last = x_curr[:, -1, :]
        z_last = z_seq[-1]
        dec_last = dec_seq[-1]

        act_ref_seq = self.defended_ref["act_ref_seq"].to(z_seq.device, dtype=z_seq.dtype)
        if act_ref_seq.dim() == 3 and act_ref_seq.shape[0] == 1:
            act_ref_seq = act_ref_seq[0]
        act_ref_seq = act_ref_seq[:seq_len]

        dec_ref_seq = self.defended_ref["decode_ref_seq"].to(dec_seq.device, dtype=dec_seq.dtype)
        if dec_ref_seq.dim() == 3 and dec_ref_seq.shape[0] == 1:
            dec_ref_seq = dec_ref_seq[0]
        dec_ref_seq = dec_ref_seq[:seq_len]
        dec_ref = dec_ref_seq[-1]

        act_delta_seq = z_seq - act_ref_seq
        dec_delta_seq = dec_seq - dec_ref_seq
        dec_delta = dec_last - dec_ref

        self.state["feat_act_curr_seq"] = z_seq.detach().float()
        self.state["feat_act_curr"] = z_last.detach().float()
        self.state["feat_decode_curr_seq"] = dec_seq.detach().float()
        self.state["feat_decode_curr"] = dec_last.detach().float()
        self.state["feat_act_drift_l2_seq"] = act_delta_seq.norm(p=2)
        self.state["feat_act_drift_linf_seq"] = act_delta_seq.abs().max()
        self.state["feat_decode_drift_l2_seq"] = dec_delta_seq.norm(p=2)
        self.state["feat_decode_drift_linf_seq"] = dec_delta_seq.abs().max()
        # Backward-compatible aliases for older callers.
        self.state["feat_act_drift_l2"] = self.state["feat_act_drift_l2_seq"]
        self.state["feat_act_drift_linf"] = self.state["feat_act_drift_linf_seq"]
        self.state["feat_decode_drift_l2"] = self.state["feat_decode_drift_l2_seq"]
        self.state["feat_decode_drift_linf"] = self.state["feat_decode_drift_linf_seq"]
        self.state["feat_decode_drift_l2_last"] = dec_delta.norm(p=2)
        self.state["feat_decode_drift_linf_last"] = dec_delta.abs().max()
        self.state["x_last_curr"] = x_last[0].detach().float()

        out[:, :, :] = x_curr.to(out.dtype)
        return out


def get_selected_encoder_columns(sae: SAE, feature_idx: torch.Tensor) -> torch.Tensor:
    return sae.W_enc[:, feature_idx].detach().float()


def project_to_encoder_null(vec: torch.Tensor, sae: SAE, feature_idx: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
    cols = get_selected_encoder_columns(sae, feature_idx).to(vec.device, dtype=vec.dtype)
    if cols.numel() == 0:
        return vec
    gram = cols.T @ cols
    eye = torch.eye(gram.shape[0], device=vec.device, dtype=vec.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, cols.T @ vec)
    proj = cols @ coeff
    return vec - proj


def optimize_delta(
    model,
    sae: SAE,
    hook_name: str,
    tokens: torch.Tensor,
    feature_idx: torch.Tensor,
    defended_ref: Dict[str, torch.Tensor],
    *,
    objective_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    objective_name: str = "objective",
    last_idx: Optional[int] = None,
    target_token_id: Optional[int] = None,
    choice_token_ids: Optional[List[int]] = None,
    loss_mode: str = "choice_margin",
    num_steps: int = 100,
    lr: float = 0.1,
    max_delta_norm: float = 20.0,
    projection_mode: str = "encoder",
    delta_position: Optional[int] = None,
    delta_positions: Optional[List[int]] = None,
    lambda_act: float = 0.0,
    lambda_decode: float = 0.0,
    lambda_delta: float = 0.0,
    ridge: float = 1e-4,
    seed: int = 0,
    extra_fwd_hooks: Optional[List[tuple[str, Callable]]] = None,
    init_delta: Optional[torch.Tensor] = None,
    fixed_delta: Optional[torch.Tensor] = None,
    fixed_delta_positions: Optional[List[int]] = None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    state: Dict[str, Any] = {}
    delta_rows = len(delta_positions) if delta_positions is not None else 1
    delta_init = torch.zeros(delta_rows, model.cfg.d_model, device=tokens.device, dtype=torch.float32)
    if init_delta is not None:
        init = init_delta.detach().to(device=tokens.device, dtype=torch.float32)
        if init.dim() == 1:
            init = init.unsqueeze(0)
        if init.shape[1] != model.cfg.d_model:
            raise ValueError(f"init_delta has width {init.shape[1]}, expected {model.cfg.d_model}")
        rows = min(init.shape[0], delta_rows)
        delta_init[:rows] = init[:rows]
    delta = delta_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    history: List[Dict[str, float]] = []

    if objective_fn is None:
        if last_idx is None or target_token_id is None:
            raise ValueError("Either objective_fn or (last_idx, target_token_id) must be provided.")

        def objective_from_last(logits: torch.Tensor) -> torch.Tensor:
            return objective_value(logits[0, last_idx, :].float(), target_token_id, choice_token_ids, loss_mode)

        objective_fn = objective_from_last
        objective_name = loss_mode

    for step in range(num_steps):
        model.reset_hooks()
        model.zero_grad(set_to_none=True)
        hook_obj = FixedDirectDefendedPlusDeltaHook(
            sae=sae,
            feature_idx=feature_idx,
            defended_ref=defended_ref,
            delta_last=delta,
            state=state,
            delta_position=delta_position,
            delta_positions=delta_positions,
            fixed_delta=fixed_delta,
            fixed_delta_positions=fixed_delta_positions,
        )
        fwd_hooks = list(extra_fwd_hooks or []) + [(hook_name, hook_obj)]
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        objective = objective_fn(logits)
        delta_penalty = delta.norm(p=2)
        loss = (
            -objective
            + lambda_act * state["feat_act_drift_l2_seq"]
            + lambda_decode * state["feat_decode_drift_l2_seq"]
            + lambda_delta * delta_penalty
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            raw_grad = delta.grad.detach().float().clone()
            if projection_mode == "encoder":
                proj_rows = [project_to_encoder_null(row, sae, feature_idx, ridge=ridge) for row in raw_grad]
                proj_grad = torch.stack(proj_rows, dim=0)
                delta.grad.copy_(proj_grad.to(delta.grad.dtype))
                proj_cos = F.cosine_similarity(raw_grad.reshape(1, -1), proj_grad.reshape(1, -1), dim=1).item()
            elif projection_mode == "none":
                proj_cos = 1.0
            else:
                raise ValueError(f"Unknown projection_mode: {projection_mode}")

        optimizer.step()

        with torch.no_grad():
            dn = delta.data.norm().item()
            if dn > max_delta_norm:
                delta.data.mul_(max_delta_norm / (dn + 1e-8))

        history.append(
            {
                "step": float(step),
                "objective": float(objective.item()),
                "objective_name": objective_name,
                "score": float(objective.item()),
                "loss": float(loss.item()),
                "act_drift_l2_seq": float(state["feat_act_drift_l2_seq"].item()),
                "decode_drift_l2_seq": float(state["feat_decode_drift_l2_seq"].item()),
                "act_drift_l2": float(state["feat_act_drift_l2_seq"].item()),
                "decode_drift_l2": float(state["feat_decode_drift_l2_seq"].item()),
                "delta_norm": float(delta.detach().norm().item()),
                "delta_penalty": float(delta_penalty.item()),
                "proj_cos": float(proj_cos),
            }
        )

    final_state: Dict[str, Any] = {}
    final_hook = FixedDirectDefendedPlusDeltaHook(
        sae=sae,
        feature_idx=feature_idx,
        defended_ref=defended_ref,
        delta_last=delta.detach(),
        state=final_state,
        delta_position=delta_position,
        delta_positions=delta_positions,
        fixed_delta=fixed_delta,
        fixed_delta_positions=fixed_delta_positions,
    )
    with torch.no_grad():
        final_fwd_hooks = list(extra_fwd_hooks or []) + [(hook_name, final_hook)]
        final_logits = model.run_with_hooks(tokens, fwd_hooks=final_fwd_hooks)

    final_score = objective_fn(final_logits)
    result = {
        "delta": delta.detach().cpu(),
        "history": history,
        "final_logits": final_logits.detach().float().cpu(),
        "final_score": float(final_score.item()),
        "final_objective": float(final_score.item()),
        "final_act_drift_l2_seq": float(final_state["feat_act_drift_l2_seq"].item()),
        "final_act_drift_linf_seq": float(final_state["feat_act_drift_linf_seq"].item()),
        "final_decode_drift_l2_seq": float(final_state["feat_decode_drift_l2_seq"].item()),
        "final_decode_drift_linf_seq": float(final_state["feat_decode_drift_linf_seq"].item()),
        "final_act_drift_l2": float(final_state["feat_act_drift_l2_seq"].item()),
        "final_act_drift_linf": float(final_state["feat_act_drift_linf_seq"].item()),
        "final_decode_drift_l2": float(final_state["feat_decode_drift_l2_seq"].item()),
        "final_decode_drift_linf": float(final_state["feat_decode_drift_linf_seq"].item()),
        "final_decode_drift_l2_last": float(final_state["feat_decode_drift_l2_last"].item()),
        "final_decode_drift_linf_last": float(final_state["feat_decode_drift_linf_last"].item()),
        "final_delta_norm": float(delta.detach().norm().item()),
        "final_x_last": final_state["x_last_curr"].detach().cpu(),
        "final_feat_act_seq": final_state["feat_act_curr_seq"].detach().cpu(),
        "final_feat_decode_seq": final_state["feat_decode_curr_seq"].detach().cpu(),
    }
    if last_idx is not None:
        result["final_logits_last"] = final_logits[0, last_idx, :].detach().float().cpu()
    return result
