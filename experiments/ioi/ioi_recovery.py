from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_bench.recovery_core import build_direct_defended_reference, cast_for_sae, optimize_delta


def load_official_ioi_dataset_class(source_path: str | Path):
    source_path = Path(source_path)
    spec = importlib.util.spec_from_file_location("official_easy_transformer_ioi_dataset", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load IOIDataset source from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IOIDataset



def build_official_ioi_dataset(
    source_path: str | Path,
    tokenizer,
    *,
    prompt_type: str | list[str] = "BABA",
    n_prompts: int = 128,
    nb_templates: int | None = 1,
    seed: int = 0,
    prepend_bos: bool = False,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_cls = load_official_ioi_dataset_class(source_path)
    return dataset_cls(
        prompt_type=prompt_type,
        N=n_prompts,
        tokenizer=tokenizer,
        nb_templates=nb_templates,
        prepend_bos=prepend_bos,
    )



def require_shared_answer_position(answer_positions: torch.Tensor) -> int:
    positions = torch.as_tensor(answer_positions, dtype=torch.long)
    uniq = torch.unique(positions)
    if uniq.numel() != 1:
        raise ValueError(
            "All prompts must share the same answer position for the current shared-delta IOI recovery setup. "
            f"Got positions {uniq.tolist()}"
        )
    return int(uniq.item())



def _gather_answer_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    answer_positions: torch.Tensor,
) -> torch.Tensor:
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    answer_positions = torch.as_tensor(answer_positions, device=logits.device, dtype=torch.long)
    token_ids = torch.as_tensor(token_ids, device=logits.device, dtype=torch.long)
    return logits[batch_idx, answer_positions, token_ids]



def compute_ioi_metrics(
    logits: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    answer_positions: torch.Tensor,
) -> dict[str, Any]:
    io_logits = _gather_answer_logits(logits, io_token_ids, answer_positions).float()
    s_logits = _gather_answer_logits(logits, s_token_ids, answer_positions).float()
    logit_diff = io_logits - s_logits
    correct = logit_diff > 0
    return {
        "accuracy": float(correct.float().mean().item()),
        "correct_count": int(correct.sum().item()),
        "total_count": int(correct.numel()),
        "mean_logit_diff": float(logit_diff.mean().item()),
        "mean_io_logit": float(io_logits.mean().item()),
        "mean_s_logit": float(s_logits.mean().item()),
        "per_prompt_logit_diff": [float(x) for x in logit_diff.detach().cpu().tolist()],
    }



def mean_ioi_logit_diff(
    logits: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    answer_positions: torch.Tensor,
) -> torch.Tensor:
    io_logits = _gather_answer_logits(logits, io_token_ids, answer_positions)
    s_logits = _gather_answer_logits(logits, s_token_ids, answer_positions)
    return (io_logits - s_logits).mean()



def compute_reactivation_metrics(
    *,
    sae: SAE,
    defended_ref: dict[str, torch.Tensor],
    feature_idx: torch.Tensor,
    delta: torch.Tensor,
    answer_positions: torch.Tensor,
) -> dict[str, Any]:
    x_def_all = defended_ref["x_def_all"].to(device=sae.W_enc.device, dtype=torch.float32)
    x_curr = x_def_all.clone()
    delta_t = delta.detach().to(device=x_curr.device, dtype=torch.float32)
    if delta_t.dim() == 1:
        delta_t = delta_t.unsqueeze(0)

    batch_idx = torch.arange(x_curr.shape[0], device=x_curr.device)
    pos = torch.as_tensor(answer_positions, device=x_curr.device, dtype=torch.long)
    if delta_t.shape[0] == 1:
        x_curr[batch_idx, pos, :] = x_curr[batch_idx, pos, :] + delta_t[0].unsqueeze(0)
    elif delta_t.shape[0] == x_curr.shape[0]:
        x_curr[batch_idx, pos, :] = x_curr[batch_idx, pos, :] + delta_t
    else:
        raise ValueError(
            f"delta shape {tuple(delta_t.shape)} is incompatible with batch size {x_curr.shape[0]}"
        )

    with torch.no_grad():
        z_def = sae.encode(cast_for_sae(x_def_all, sae)).float()
        z_curr = sae.encode(cast_for_sae(x_curr, sae)).float()

    feat_idx = feature_idx.to(device=x_curr.device, dtype=torch.long)
    feat_def = z_def[batch_idx, pos][:, feat_idx]
    feat_curr = z_curr[batch_idx, pos][:, feat_idx]

    eligible_mask = feat_def <= 0
    reactivated_mask = (feat_curr > 0) & eligible_mask
    curr_positive_mask = feat_curr > 0
    def_positive_mask = feat_def > 0

    eligible_per_prompt = eligible_mask.float().sum(dim=1).clamp_min(1.0)
    total_count = int(feat_curr.numel())
    eligible_count = int(eligible_mask.sum().item())
    reactivated_count = int(reactivated_mask.sum().item())

    reactivated_positive = feat_curr.clamp_min(0.0) * eligible_mask.float()
    current_positive = feat_curr.clamp_min(0.0)
    defended_positive = feat_def.clamp_min(0.0)

    return {
        "selected_feature_positive_fraction": float(curr_positive_mask.float().mean().item()),
        "selected_feature_positive_mass_mean": float(current_positive.mean().item()),
        "defended_positive_fraction": float(def_positive_mask.float().mean().item()),
        "defended_positive_mass_mean": float(defended_positive.mean().item()),
        "eligible_nonpositive_fraction": float(eligible_mask.float().mean().item()),
        "eligible_nonpositive_count": eligible_count,
        "selected_feature_total_count": total_count,
        "reactivated_fraction": float(reactivated_mask.float().mean().item()),
        "reactivated_count": reactivated_count,
        "reactivated_positive_fraction_within_eligible": float(
            reactivated_mask.float().sum().item() / max(1, eligible_count)
        ),
        "reactivated_positive_mass_mean": float(reactivated_positive.mean().item()),
        "reactivated_positive_mass_mean_within_eligible": float(
            reactivated_positive.sum().item() / max(1, eligible_count)
        ),
        "per_prompt_reactivated_fraction": [
            float(x) for x in (reactivated_mask.float().sum(dim=1) / eligible_per_prompt).detach().cpu().tolist()
        ],
        "per_prompt_positive_fraction": [
            float(x) for x in curr_positive_mask.float().mean(dim=1).detach().cpu().tolist()
        ],
        "per_prompt_reactivated_positive_mass": [
            float(x) for x in (reactivated_positive.sum(dim=1) / eligible_per_prompt).detach().cpu().tolist()
        ],
    }



def select_top_ioi_features(
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    tokens: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    answer_positions: torch.Tensor,
    *,
    topk: int = 16,
) -> dict[str, Any]:
    saved: dict[str, torch.Tensor] = {}

    def passthrough_sae_hook(resid: torch.Tensor, hook=None, **kwargs):
        x_sae = cast_for_sae(resid, sae)
        z = sae.encode(x_sae)
        z.retain_grad()
        recon = sae.decode(z)
        passthrough = recon + (resid.float() - recon.float()).detach()
        saved["z"] = z
        return passthrough.to(resid.dtype)

    model.reset_hooks()
    model.zero_grad(set_to_none=True)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, passthrough_sae_hook)])
    objective = mean_ioi_logit_diff(logits, io_token_ids, s_token_ids, answer_positions)
    objective.backward()

    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    pos = torch.as_tensor(answer_positions, device=tokens.device, dtype=torch.long)
    z_end = saved["z"][batch_idx, pos].float()
    grad_end = saved["z"].grad[batch_idx, pos].float()
    mean_act = z_end.mean(dim=0)
    mean_grad = grad_end.mean(dim=0)
    attr = (z_end * grad_end).mean(dim=0)

    positive_idx = (attr > 0).nonzero(as_tuple=False).squeeze(-1)
    if positive_idx.numel() > 0:
        positive_attr = attr[positive_idx]
        k = min(topk, positive_attr.numel())
        local_rank = torch.topk(positive_attr, k=k).indices
        feature_idx = positive_idx[local_rank]
    else:
        k = min(topk, attr.numel())
        feature_idx = torch.topk(attr.abs(), k=k).indices

    return {
        "feature_idx": feature_idx.detach().cpu(),
        "attribution": attr.detach().cpu(),
        "mean_activation": mean_act.detach().cpu(),
        "mean_gradient": mean_grad.detach().cpu(),
        "objective": float(objective.item()),
    }



def run_ioi_recovery_experiment(
    *,
    model: HookedTransformer,
    sae: SAE,
    hook_name: str,
    tokens: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
    answer_positions: torch.Tensor,
    topk_features: int = 16,
    clamp_multiplier: float = 0.0,
    modes: list[str] | tuple[str, ...] = ("none", "encoder"),
    num_steps: int = 100,
    lr: float = 0.1,
    max_delta_norm: float = 20.0,
    lambda_act: float = 0.0,
    lambda_decode: float = 0.0,
    ridge: float = 1e-4,
    seed: int = 0,
) -> dict[str, Any]:
    answer_position = require_shared_answer_position(answer_positions)

    with torch.no_grad():
        baseline_logits = model(tokens, return_type="logits")
    baseline_metrics = compute_ioi_metrics(baseline_logits, io_token_ids, s_token_ids, answer_positions)

    feat_info = select_top_ioi_features(
        model=model,
        sae=sae,
        hook_name=hook_name,
        tokens=tokens,
        io_token_ids=io_token_ids,
        s_token_ids=s_token_ids,
        answer_positions=answer_positions,
        topk=topk_features,
    )
    feature_idx = feat_info["feature_idx"].to(tokens.device)

    with torch.no_grad():
        defended_logits, defended_ref = build_direct_defended_reference(
            model=model,
            sae=sae,
            hook_name=hook_name,
            tokens=tokens,
            feature_idx=feature_idx,
            multiplier=clamp_multiplier,
        )
    suppressed_metrics = compute_ioi_metrics(defended_logits, io_token_ids, s_token_ids, answer_positions)

    def objective_fn(logits: torch.Tensor) -> torch.Tensor:
        return mean_ioi_logit_diff(logits, io_token_ids, s_token_ids, answer_positions)

    recovery_results: dict[str, Any] = {}
    for mode in modes:
        result = optimize_delta(
            model=model,
            sae=sae,
            hook_name=hook_name,
            tokens=tokens,
            feature_idx=feature_idx,
            defended_ref=defended_ref,
            objective_fn=objective_fn,
            objective_name="mean_ioi_logit_diff",
            num_steps=num_steps,
            lr=lr,
            max_delta_norm=max_delta_norm,
            projection_mode=mode,
            delta_position=answer_position,
            lambda_act=lambda_act,
            lambda_decode=lambda_decode,
            ridge=ridge,
            seed=seed,
        )
        reactivation = compute_reactivation_metrics(
            sae=sae,
            defended_ref=defended_ref,
            feature_idx=feature_idx,
            delta=result["delta"],
            answer_positions=answer_positions,
        )
        recovery_results[mode] = {
            "metrics": compute_ioi_metrics(result["final_logits"], io_token_ids, s_token_ids, answer_positions),
            "final_act_drift_l2": result["final_act_drift_l2"],
            "final_decode_drift_l2": result["final_decode_drift_l2"],
            "final_delta_norm": result["final_delta_norm"],
            "reactivation": reactivation,
            "history": result["history"],
            "selected_feature_idx": [int(x) for x in feature_idx.detach().cpu().tolist()],
        }

    return {
        "answer_position": answer_position,
        "baseline_metrics": baseline_metrics,
        "suppressed_metrics": suppressed_metrics,
        "selected_feature_idx": [int(x) for x in feature_idx.detach().cpu().tolist()],
        "feature_objective": feat_info["objective"],
        "recovery": recovery_results,
    }



def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
