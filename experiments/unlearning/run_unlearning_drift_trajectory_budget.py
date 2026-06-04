#!/usr/bin/env python3
"""Diagnostic unlearning recovery runs for optimization drift and norm budgets.

This script intentionally runs a small matched slice, not the full 91 flips.
It records post-hoc choice-readout defended-feature drift at every optimization
step for Encoder-projected and unconstrained recovery, then sweeps the delta
norm budget and summarizes recovery/drift trade-offs.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from permutation_seqwide import LETTERS, format_prompt, parse_prompt
from sae_bench.recovery_core import (
    FixedDirectDefendedPlusDeltaHook,
    build_direct_defended_reference,
    get_choice_token_ids,
    objective_value,
    project_to_encoder_null,
)


PALETTE = {
    "blue": "#0F4D92",
    "red": "#B64342",
    "grid": "#E6E8EF",
    "text": "#4B5563",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_module_from_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_sample_ids(text: Optional[str]) -> Optional[set[int]]:
    if not text:
        return None
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def resolve_manifest_file(manifest_path: Path, file_text: str) -> Path:
    p = Path(file_text)
    if p.is_absolute() or p.exists():
        return p
    for cand in [manifest_path.parent / p, manifest_path.parent.parent / p, REPO_ROOT / p]:
        if cand.exists():
            return cand
    return manifest_path.parent / p


def greedy_choice_letter(model, logits_pos: torch.Tensor) -> str:
    scores = []
    for letter in LETTERS:
        tok = model.to_single_token(" " + letter)
        scores.append((letter, float(logits_pos[tok].item())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def metric_from_state(
    state: Dict[str, torch.Tensor],
    defended_ref: Dict[str, torch.Tensor],
    readout_pos: int,
    eps: float,
    direction_sign: float,
) -> Dict[str, float]:
    z_seq = state["feat_act_curr_seq"].detach().float()
    z_rec = z_seq[int(readout_pos)]
    z_ref_seq = defended_ref["act_ref_seq"].to(z_rec.device, dtype=z_rec.dtype)
    if z_ref_seq.dim() == 3 and z_ref_seq.shape[0] == 1:
        z_ref_seq = z_ref_seq[0]
    z0 = z_ref_seq[int(readout_pos)].detach().float()
    diff = z_rec - z0
    norm0 = float(z0.norm(p=2).item())
    drift_abs = float(diff.norm(p=2).item())
    violation_abs = float(torch.clamp(float(direction_sign) * diff, min=0.0).norm(p=2).item())
    return {
        "defended_norm": norm0,
        "drift_abs_l2": drift_abs,
        "drift_relative_l2": drift_abs / (norm0 + eps),
        "violation_abs_l2": violation_abs,
        "violation_relative_l2": violation_abs / (norm0 + eps),
    }


def optimize_with_step_history(
    *,
    model,
    sae,
    hook_name: str,
    tokens: torch.Tensor,
    readout_pos: int,
    feature_idx: torch.Tensor,
    defended_ref: Dict[str, torch.Tensor],
    target_token_id: int,
    choice_token_ids: List[int],
    gold_letter: str,
    loss_mode: str,
    projection_mode: str,
    max_delta_norm: float,
    num_steps: int,
    lr: float,
    ridge: float,
    seed: int,
    eps: float,
    direction_sign: float,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    delta = torch.zeros(1, model.cfg.d_model, device=tokens.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    history: List[Dict[str, Any]] = []

    def forward_with_delta(delta_tensor: torch.Tensor):
        state: Dict[str, Any] = {}
        hook_obj = FixedDirectDefendedPlusDeltaHook(
            sae=sae,
            feature_idx=feature_idx,
            defended_ref=defended_ref,
            delta_last=delta_tensor,
            state=state,
        )
        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_obj)])
        return logits, state

    for step in range(num_steps):
        model.reset_hooks()
        model.zero_grad(set_to_none=True)
        logits, state = forward_with_delta(delta)
        logits_pos = logits[0, readout_pos, :].float()
        objective = objective_value(logits_pos, target_token_id, choice_token_ids, loss_mode)
        loss = -objective
        choice = greedy_choice_letter(model, logits_pos.detach().cpu())
        metrics = metric_from_state(state, defended_ref, readout_pos, eps, direction_sign)
        delta_norm_before = float(delta.detach().norm().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            raw_grad = delta.grad.detach().float().clone()
            if projection_mode == "encoder":
                proj_rows = [project_to_encoder_null(row, sae, feature_idx, ridge=ridge) for row in raw_grad]
                proj_grad = torch.stack(proj_rows, dim=0)
                delta.grad.copy_(proj_grad.to(delta.grad.dtype))
                proj_cos = float(F.cosine_similarity(raw_grad.reshape(1, -1), proj_grad.reshape(1, -1), dim=1).item())
            elif projection_mode == "none":
                proj_cos = 1.0
            else:
                raise ValueError(f"Unknown projection_mode={projection_mode}")
        optimizer.step()
        with torch.no_grad():
            dn = delta.data.norm().item()
            if dn > max_delta_norm:
                delta.data.mul_(max_delta_norm / (dn + 1e-8))

        history.append(
            {
                "step": step,
                "projection_mode": projection_mode,
                "objective": float(objective.item()),
                "loss": float(loss.item()),
                "choice": choice,
                "correct": choice == gold_letter,
                "delta_norm_before": delta_norm_before,
                "delta_norm_after": float(delta.detach().norm().item()),
                "proj_cos": proj_cos,
                **metrics,
            }
        )

    with torch.no_grad():
        final_logits, final_state = forward_with_delta(delta.detach())
        final_logits_pos = final_logits[0, readout_pos, :].detach().float().cpu()
        final_choice = greedy_choice_letter(model, final_logits_pos)
        final_metrics = metric_from_state(final_state, defended_ref, readout_pos, eps, direction_sign)
        final_score = float(objective_value(final_logits[0, readout_pos, :].float(), target_token_id, choice_token_ids, loss_mode).item())

    return {
        "delta": delta.detach().cpu(),
        "history": history,
        "final_choice": final_choice,
        "final_correct": final_choice == gold_letter,
        "final_score": final_score,
        "final_delta_norm": float(delta.detach().norm().item()),
        **{f"final_{k}": v for k, v in final_metrics.items()},
    }


def collect_valid_flips(
    *,
    model,
    base,
    sae,
    hook_name: str,
    manifest: Dict[str, Any],
    manifest_path: Path,
    sample_ids: Optional[set[int]],
    strict_base_24_24: bool,
    max_valid_flips_per_question: Optional[int],
) -> List[Dict[str, Any]]:
    feature_idx = torch.tensor([int(x) for x in manifest["feature_ids"]], device=model.cfg.device, dtype=torch.long)
    multiplier = float(manifest["multiplier"])
    rows: List[Dict[str, Any]] = []
    for item in manifest["items"]:
        sample_id = int(item["sample_id"])
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        prompt_text = resolve_manifest_file(manifest_path, item["prompt_file"]).read_text(encoding="utf-8")
        question, choices = parse_prompt(prompt_text)
        gold_orig = item["gold_letter"]
        gold_idx_orig = LETTERS.index(gold_orig)

        q_rows: List[Dict[str, Any]] = []
        for perm_id, perm in enumerate(itertools.permutations(range(4))):
            perm_choices = [choices[i] for i in perm]
            gold_idx = perm.index(gold_idx_orig)
            gold_letter = LETTERS[gold_idx]
            prompt_perm = format_prompt(question, perm_choices)
            tokens = model.to_tokens(prompt_perm)
            readout_pos = tokens.shape[1] - 1
            target_token_id = model.to_single_token(" " + gold_letter)
            with torch.no_grad():
                base_logits = model(tokens)
                direct_def_logits, defended_ref = build_direct_defended_reference(
                    model=model,
                    sae=sae,
                    hook_name=hook_name,
                    tokens=tokens,
                    feature_idx=feature_idx,
                    multiplier=multiplier,
                )
            base_choice = greedy_choice_letter(model, base_logits[0, readout_pos, :].detach().float().cpu())
            defended_choice = greedy_choice_letter(model, direct_def_logits[0, readout_pos, :].detach().float().cpu())
            q_rows.append(
                {
                    "sample_id": sample_id,
                    "orig_dataset_idx": int(item["idx"]),
                    "perm_id": perm_id,
                    "perm": list(perm),
                    "prompt": prompt_perm,
                    "gold_letter_perm": gold_letter,
                    "base_choice": base_choice,
                    "direct_defended_choice": defended_choice,
                    "base_correct": base_choice == gold_letter,
                    "valid_flip": (base_choice == gold_letter and defended_choice != gold_letter),
                    "_tokens": tokens,
                    "_readout_pos": readout_pos,
                    "_target_token_id": target_token_id,
                    "_defended_ref": defended_ref,
                }
            )
        if strict_base_24_24 and sum(1 for r in q_rows if r["base_correct"]) != 24:
            continue
        valid = [r for r in q_rows if r["valid_flip"]]
        if max_valid_flips_per_question is not None:
            valid = valid[: max_valid_flips_per_question]
        rows.extend(valid)
    return rows


def summarize_budget(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["projection_mode"], float(r["max_delta_norm"]))].append(r)
    out: List[Dict[str, Any]] = []
    for (method, budget), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(vals)
        n_rec = sum(1 for v in vals if v["recovered_correct"])
        out.append(
            {
                "projection_mode": method,
                "max_delta_norm": budget,
                "n_valid_flips": n,
                "n_recovered": n_rec,
                "recovery_rate": n_rec / n if n else None,
                "mean_drift_abs_l2": float(np.mean([v["drift_abs_l2"] for v in vals])) if vals else None,
                "mean_violation_abs_l2": float(np.mean([v["violation_abs_l2"] for v in vals])) if vals else None,
                "mean_final_delta_norm": float(np.mean([v["final_delta_norm"] for v in vals])) if vals else None,
            }
        )
    return out


def plot_trajectory(rows: List[Dict[str, Any]], out_base: Path) -> List[str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    for method, color, label in [("encoder", PALETTE["blue"], "Encoder proj."), ("none", PALETTE["red"], "No projection")]:
        vals = [r for r in rows if r["projection_mode"] == method]
        vals.sort(key=lambda r: r["step"])
        ax.plot([r["step"] for r in vals], [r["drift_abs_l2"] for r in vals], color=color, lw=1.8, label=label)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Choice-readout drift (abs. L2)")
    ax.set_title("Drift during optimization", loc="left", pad=4)
    ax.grid(axis="both", color=PALETTE["grid"], linewidth=0.7)
    ax.legend(frameon=False, loc="upper left")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ["pdf", "png"]:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=450 if ext == "png" else 300)
        saved.append(str(p))
    plt.close(fig)
    return saved


def plot_budget(summary: List[Dict[str, Any]], out_base: Path) -> List[str]:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.25), gridspec_kw={"wspace": 0.34})
    methods = [("encoder", PALETTE["blue"], "Encoder proj."), ("none", PALETTE["red"], "No projection")]
    for ax, ykey, ylabel, title in [
        (axes[0], "recovery_rate", "Recovery rate (%)", "(a) Budget vs recovery"),
        (axes[1], "mean_drift_abs_l2", "Choice-readout drift (abs. L2)", "(b) Budget vs drift"),
    ]:
        for method, color, label in methods:
            vals = [r for r in summary if r["projection_mode"] == method]
            vals.sort(key=lambda r: r["max_delta_norm"])
            xs = [r["max_delta_norm"] for r in vals]
            if ykey == "recovery_rate":
                ys = [100.0 * r[ykey] for r in vals]
            else:
                ys = [r[ykey] for r in vals]
            ax.plot(xs, ys, marker="o", lw=1.8, ms=4.5, color=color, label=label)
        ax.set_xlabel("Delta norm budget")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=4)
        ax.grid(axis="both", color=PALETTE["grid"], linewidth=0.7)
    axes[0].set_ylim(-4, 104)
    axes[0].legend(frameon=False, loc="lower right")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ["pdf", "png"]:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=450 if ext == "png" else 300)
        saved.append(str(p))
    plt.close(fig)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="unlearning_task/prompt_pool/manifest.json")
    ap.add_argument("--base_recovery_script", default="Aout/unlearning/recovery_unlearning_choice_only_seqwide_act.py")
    ap.add_argument("--sample_ids", default="1")
    ap.add_argument("--max_valid_flips_per_question", type=int, default=6)
    ap.add_argument("--budgets", default="0,1,2,5,10,20")
    ap.add_argument("--trajectory_budget", type=float, default=20.0)
    ap.add_argument("--num_steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss_mode", default="choice_margin", choices=["choice_margin", "choice_ce", "vocab_margin"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--direction_sign", type=float, default=1.0)
    ap.add_argument("--output_dir", default="Aout/unlearning/posthoc_eval_runs/drift_budget_diagnostic")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = load_module_from_path(str(REPO_ROOT / args.base_recovery_script), "base_recovery_module")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_dtype = base.dtype_from_string("bfloat16")
    model, sae, hook_name = base.load_model_and_sae_from_release(
        model_name=manifest["model_name"],
        sae_release=manifest["sae_release"],
        sae_id=manifest["sae_id"],
        device=device,
        llm_dtype=llm_dtype,
    )
    feature_idx = torch.tensor([int(x) for x in manifest["feature_ids"]], device=device, dtype=torch.long)
    choice_token_ids = get_choice_token_ids(model, LETTERS)
    sample_ids = parse_sample_ids(args.sample_ids)
    max_flips = 1 if args.smoke else args.max_valid_flips_per_question
    num_steps = min(args.num_steps, 5) if args.smoke else args.num_steps
    budgets = [0.0, 1.0] if args.smoke else [float(x) for x in args.budgets.split(",") if x.strip()]

    flips = collect_valid_flips(
        model=model,
        base=base,
        sae=sae,
        hook_name=hook_name,
        manifest=manifest,
        manifest_path=manifest_path,
        sample_ids=sample_ids,
        strict_base_24_24=True,
        max_valid_flips_per_question=max_flips,
    )
    if not flips:
        raise RuntimeError("No valid flips selected.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest": str(manifest_path),
        "sample_ids": sorted(sample_ids) if sample_ids else None,
        "n_selected_valid_flips": len(flips),
        "selected_keys": [{"sample_id": r["sample_id"], "perm_id": r["perm_id"]} for r in flips],
        "num_steps": num_steps,
        "lr": args.lr,
        "budgets": budgets,
        "trajectory_budget": args.trajectory_budget,
        "loss_mode": args.loss_mode,
        "drift_metric": "choice_readout_posthoc_abs_l2",
        "direction_sign": args.direction_sign,
    }
    write_json(metadata, out_root / "metadata.json")

    trajectory_rows: List[Dict[str, Any]] = []
    traj_flip = flips[0]
    for method in ["encoder", "none"]:
        result = optimize_with_step_history(
            model=model,
            sae=sae,
            hook_name=hook_name,
            tokens=traj_flip["_tokens"],
            readout_pos=traj_flip["_readout_pos"],
            feature_idx=feature_idx,
            defended_ref=traj_flip["_defended_ref"],
            target_token_id=traj_flip["_target_token_id"],
            choice_token_ids=choice_token_ids,
            gold_letter=traj_flip["gold_letter_perm"],
            loss_mode=args.loss_mode,
            projection_mode=method,
            max_delta_norm=args.trajectory_budget,
            num_steps=num_steps,
            lr=args.lr,
            ridge=args.ridge,
            seed=args.seed,
            eps=args.eps,
            direction_sign=args.direction_sign,
        )
        for h in result["history"]:
            trajectory_rows.append(
                {
                    "sample_id": traj_flip["sample_id"],
                    "perm_id": traj_flip["perm_id"],
                    "gold_letter": traj_flip["gold_letter_perm"],
                    **h,
                }
            )
    write_jsonl(trajectory_rows, out_root / "trajectory_rows.jsonl")

    budget_rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for method in ["encoder", "none"]:
            for flip in flips:
                result = optimize_with_step_history(
                    model=model,
                    sae=sae,
                    hook_name=hook_name,
                    tokens=flip["_tokens"],
                    readout_pos=flip["_readout_pos"],
                    feature_idx=feature_idx,
                    defended_ref=flip["_defended_ref"],
                    target_token_id=flip["_target_token_id"],
                    choice_token_ids=choice_token_ids,
                    gold_letter=flip["gold_letter_perm"],
                    loss_mode=args.loss_mode,
                    projection_mode=method,
                    max_delta_norm=budget,
                    num_steps=num_steps,
                    lr=args.lr,
                    ridge=args.ridge,
                    seed=args.seed,
                    eps=args.eps,
                    direction_sign=args.direction_sign,
                )
                budget_rows.append(
                    {
                        "sample_id": flip["sample_id"],
                        "perm_id": flip["perm_id"],
                        "gold_letter": flip["gold_letter_perm"],
                        "projection_mode": method,
                        "max_delta_norm": budget,
                        "recovered_choice": result["final_choice"],
                        "recovered_correct": result["final_correct"],
                        "final_score": result["final_score"],
                        "final_delta_norm": result["final_delta_norm"],
                        "drift_abs_l2": result["final_drift_abs_l2"],
                        "drift_relative_l2": result["final_drift_relative_l2"],
                        "violation_abs_l2": result["final_violation_abs_l2"],
                        "violation_relative_l2": result["final_violation_relative_l2"],
                    }
                )
                print(
                    json.dumps(
                        {
                            "method": method,
                            "budget": budget,
                            "sample_id": flip["sample_id"],
                            "perm_id": flip["perm_id"],
                            "correct": result["final_correct"],
                            "drift_abs_l2": result["final_drift_abs_l2"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    budget_summary = summarize_budget(budget_rows)
    write_jsonl(budget_rows, out_root / "budget_rows.jsonl")
    write_json({"metadata": metadata, "budget_summary": budget_summary}, out_root / "budget_aggregate.json")

    fig_dir = REPO_ROOT / "figures"
    traj_figs = plot_trajectory(trajectory_rows, fig_dir / "unlearning_stepwise_drift_trajectory")
    budget_figs = plot_budget(budget_summary, fig_dir / "unlearning_budget_sweep")
    print(
        json.dumps(
            {
                "output_dir": str(out_root),
                "trajectory_rows": str(out_root / "trajectory_rows.jsonl"),
                "budget_rows": str(out_root / "budget_rows.jsonl"),
                "budget_aggregate": str(out_root / "budget_aggregate.json"),
                "trajectory_figures": traj_figs,
                "budget_figures": budget_figs,
                "budget_summary": budget_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
