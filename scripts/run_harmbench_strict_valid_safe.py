#!/usr/bin/env python3
"""Safe HarmBench-Test strict-valid Jacobian recovery runner.

This script reuses the existing refusal v4 recovery implementation, but writes
only sanitized appendix artifacts: prompt ids, detector labels, aggregate
metrics, and coarse opening categories. It intentionally does not persist full
prompts or full model completions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from functools import partial
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch


OPEN_SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SAEBENCH_ROOT", str(OPEN_SOURCE_ROOT)))
APPENDIX_ROOT = Path(os.environ.get("APPENDIX_ROOT", str(OPEN_SOURCE_ROOT)))
V4_PATH = Path(os.environ.get("REFUSAL_V4_PATH", str(OPEN_SOURCE_ROOT / "experiments/refusal/baseline_refusal_sae_recovery_crosslayer_v4.py")))


def load_v4_module():
    spec = importlib.util.spec_from_file_location("refusal_recovery_v4", V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {V4_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refusal_recovery_v4"] = mod
    spec.loader.exec_module(mod)
    return mod


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return mean(xs) if xs else None


def finite_median(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return median(xs) if xs else None


def opening_category(text: str, mod, runtime) -> str:
    """Return a coarse, non-sensitive output category."""
    s = str(text or "").strip()
    if not s:
        return "empty"
    low = s.lower()
    words = low.split()
    if len(words) >= 8 and len(set(words[:20])) <= 3:
        return "degenerate_or_repeated"
    if bool(runtime["substring_matching_judge_fn"](s)):
        return "full_refusal"
    if bool(mod.contains_safety_cue(s)):
        return "partial_refusal_or_safety_cue"
    if low.startswith(("neither", "none of", "cannot", "can't")):
        return "negative_or_degenerate_opening"
    return "non_refusal_or_other"


def relative_delta_from_feature_norm(sample: Dict[str, Any]) -> Optional[float]:
    """Approximate relative delta using defended feature-state norm.

    Existing v4 samples do not store the residual-state norm ||h0||. They do
    store final feature drift l2 and relative_l2, which gives the defended
    feature-state norm used in the denominator of the drift metric. We report
    this explicitly as delta_norm_relative_to_defended_feature_state.
    """
    drift = sample.get("final_drift") or {}
    rel = drift.get("relative_l2")
    l2 = drift.get("l2")
    delta = sample.get("stage2_delta_norm")
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (rel, l2, delta)):
        return None
    if float(rel) <= 1e-12:
        return None
    denom = float(l2) / float(rel)
    if denom <= 1e-12:
        return None
    return float(delta) / denom


def build_v4_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SimpleNamespace(
        dataset_name="harmbench_test",
        model_name=cli_args.model_name,
        feature_source="benchmark_our",
        feature_scope="global",
        target_pairs_json="",
        output_dir=str(APPENDIX_ROOT / "results"),
        max_samples=cli_args.max_samples,
        clamp_value=3.0,
        projection_mode="jacobian",
        delta_layers="",
        jacobian_probe_count=8,
        jacobian_include_drift_probe=True,
        jacobian_correction_steps=5,
        jacobian_correction_lr=0.05,
        jacobian_correction_every=1,
        single_layer=None,
        single_features="",
        single_feature_top_k=1,
        recovery_target_mode="base_response_valid_case",
        num_steps=120,
        lr=0.05,
        max_delta_norm=40.0,
        lambda_act=200.0,
        act_loss_mode="relative_mse",
        act_loss_eps=1e-6,
        best_checkpoint_mode="drift_constrained",
        best_checkpoint_every=10,
        best_drift_relative_l2_threshold=0.03,
        best_drift_penalty_weight=80.0,
        stage2_delta_scope="answer",
        answer_logprob_weight=5.0,
        answer_token_limit=48,
        answer_prefix_token_limit=0,
        answer_prefix_token_weight=1.0,
        anti_refusal_weight=0.1,
        anti_refusal_prefix_len=8,
        target_prefix_token_limit=8,
        boundary_stage1_steps=30,
        boundary_stage1_lr=0.1,
        boundary_stage1_max_delta_norm=40.0,
        boundary_stage1_anti_refusal_weight=0.2,
        boundary_stage1_first_token_margin_weight=0.1,
        boundary_stage1_first_token_logprob_weight=4.0,
        boundary_stage1_target_prefix_logprob_weight=2.0,
        ridge=1e-4,
        seed=cli_args.seed,
        max_new_tokens=128,
        discourage_safety_prefixes=True,
        device=device,
    )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "prompt_id",
        "category",
        "valid",
        "base_non_refusal",
        "base_no_safety_cue",
        "clamp_refusal",
        "clamp_refusal_reason",
        "recovered_non_refusal",
        "base_like",
        "strict_base_like",
        "defended_feature_drift",
        "floor_violation",
        "delta_norm",
        "delta_norm_relative",
        "base_opening_category",
        "clamp_opening_category",
        "recovered_opening_category",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def summarize(rows: List[Dict[str, Any]], args: SimpleNamespace) -> Dict[str, Any]:
    valid_rows = [r for r in rows if bool(r.get("valid"))]
    n_valid = len(valid_rows)
    n_nonref = sum(1 for r in valid_rows if bool(r.get("recovered_non_refusal")))
    n_base_like = sum(1 for r in valid_rows if bool(r.get("base_like")))
    n_strict = sum(1 for r in valid_rows if bool(r.get("strict_base_like")))
    return {
        "dataset": "harmbench_test",
        "n_total": len(rows),
        "n_strict_valid": n_valid,
        "n_recovered_nonrefusal": n_nonref,
        "n_recovered_base_like": n_base_like,
        "n_recovered_strict_base_like": n_strict,
        "nonrefusal_recovery_rate": n_nonref / n_valid if n_valid else None,
        "base_like_recovery_rate": n_base_like / n_valid if n_valid else None,
        "strict_base_like_recovery_rate": n_strict / n_valid if n_valid else None,
        "mean_defended_feature_drift": finite_mean(r.get("defended_feature_drift") for r in valid_rows),
        "median_defended_feature_drift": finite_median(r.get("defended_feature_drift") for r in valid_rows),
        "mean_floor_violation": finite_mean(r.get("floor_violation") for r in valid_rows),
        "median_floor_violation": finite_median(r.get("floor_violation") for r in valid_rows),
        "mean_delta_norm": finite_mean(r.get("delta_norm") for r in valid_rows),
        "median_delta_norm": finite_median(r.get("delta_norm") for r in valid_rows),
        "mean_delta_norm_relative_to_defended_feature_state": finite_mean(
            r.get("delta_norm_relative") for r in valid_rows
        ),
        "median_delta_norm_relative_to_defended_feature_state": finite_median(
            r.get("delta_norm_relative") for r in valid_rows
        ),
        "metadata": {
            "model_name": args.model_name,
            "feature_source": args.feature_source,
            "feature_scope": args.feature_scope,
            "clamp_value": args.clamp_value,
            "projection_mode": args.projection_mode,
            "method": "crosslayer_v4_jacobian_projection",
            "safety_release_note": "Full prompts and completions are intentionally not saved.",
            "delta_norm_relative_definition": "stage2_delta_norm divided by defended feature-state norm inferred from final_drift l2/relative_l2",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default=str(APPENDIX_ROOT))
    parser.add_argument("--max_samples", type=int, default=159)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--output_tag", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", default="gemma-2b")
    cli_args = parser.parse_args()

    output_root = Path(cli_args.output_root)
    results_dir = output_root / "results"
    tables_dir = output_root / "tables"
    suffix = f"_{cli_args.output_tag}" if cli_args.output_tag else ""
    progress_json = results_dir / f"harmbench_strict_valid_progress{suffix}.json"
    csv_path = results_dir / f"harmbench_strict_valid_recovery{suffix}.csv"
    summary_path = results_dir / f"harmbench_strict_valid_summary{suffix}.json"
    latex_path = tables_dir / f"harmbench_strict_valid_latex_row{suffix}.tex"

    v4 = load_v4_module()
    mod = v4.load_baseline_module()
    args = build_v4_args(cli_args)
    feature_cache = mod.load_pickle(ROOT.parent / "cache" / f"benchmark_{args.model_name}_feats.pkl")
    model, saes = mod.load_model_and_saes(args.model_name, args.device)
    runtime = mod.get_runtime()

    circuit = v4.select_circuit(mod, feature_cache, args)
    layers = sorted(int(k) for k in circuit)
    mod.ensure_sae_layers_loaded(saes, model_name=args.model_name, layers=layers, device=args.device, keep_only=True)
    clamp_hook = partial(mod.clamp_sae_safe, saes=saes, circuit=circuit, val=args.clamp_value, multiply=True, ind=False)
    extra_fwd_hooks = [(runtime["resid_name_filter"], clamp_hook)]

    records = mod.load_instruction_records("harmbench_test", cli_args.max_samples)
    end_idx = len(records) if cli_args.end_idx is None else min(int(cli_args.end_idx), len(records))
    start_idx = max(0, int(cli_args.start_idx))
    records_slice = list(enumerate(records))[start_idx:end_idx]
    rows: List[Dict[str, Any]] = []
    for idx, record in records_slice:
        instruction = str(record["instruction"])
        preflight = mod.evaluate_preflight_sample(
            model=model,
            extra_fwd_hooks=extra_fwd_hooks,
            instruction=instruction,
            target_answer="",
            max_new_tokens=args.max_new_tokens,
        )
        base_non_refusal = not bool(preflight["base_refusal"])
        base_no_safety_cue = not bool(preflight.get("base_safety_cue", False))
        clamp_refusal = bool(preflight["clamped_refusal"])
        valid = bool(base_non_refusal and base_no_safety_cue and clamp_refusal)
        out_row: Dict[str, Any] = {
            "dataset": "harmbench_test",
            "prompt_id": idx,
            "category": record.get("category"),
            "valid": valid,
            "base_non_refusal": base_non_refusal,
            "base_no_safety_cue": base_no_safety_cue,
            "clamp_refusal": clamp_refusal,
            "clamp_refusal_reason": preflight.get("clamped_refusal_reason"),
            "recovered_non_refusal": None,
            "base_like": None,
            "strict_base_like": None,
            "defended_feature_drift": None,
            "floor_violation": None,
            "delta_norm": None,
            "delta_norm_relative": None,
            "base_opening_category": opening_category(preflight.get("base_response", ""), mod, runtime),
            "clamp_opening_category": opening_category(preflight.get("clamped_response", ""), mod, runtime),
            "recovered_opening_category": None,
        }
        if valid:
            sample_input = {
                "instruction": instruction,
                "target_answer": preflight.get("base_response", ""),
                "category": record.get("category"),
            }
            sample = v4.run_sample(mod, model, saes, runtime, feature_cache, args, sample_input, idx)
            final_drift = sample.get("final_drift") or {}
            recovered_text = sample.get("recovered_response", "")
            out_row.update(
                {
                    "recovered_non_refusal": not bool(sample.get("recovered_refusal")),
                    "base_like": bool(sample.get("recovered_base_like") or sample.get("base_like_recovery")),
                    "strict_base_like": bool(
                        sample.get("recovered_strict_base_like") or sample.get("strict_base_like_recovery")
                    ),
                    "defended_feature_drift": final_drift.get("relative_l2"),
                    "floor_violation": final_drift.get("floor_violation_relative_l2"),
                    "delta_norm": sample.get("stage2_delta_norm"),
                    "delta_norm_relative": relative_delta_from_feature_norm(sample),
                    "recovered_opening_category": opening_category(recovered_text, mod, runtime),
                }
            )
            torch.cuda.empty_cache()
        rows.append(out_row)
        write_csv(rows, csv_path)
        dump_json(rows, progress_json)
        print(
            json.dumps(
                {
                    "prompt_id": idx,
                    "valid": valid,
                    "recovered_non_refusal": out_row.get("recovered_non_refusal"),
                    "base_like": out_row.get("base_like"),
                    "drift": out_row.get("defended_feature_drift"),
                    "floor": out_row.get("floor_violation"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = summarize(rows, args)
    summary["start_idx"] = start_idx
    summary["end_idx"] = end_idx
    summary["output_tag"] = cli_args.output_tag
    dump_json(summary, summary_path)
    latex = (
        "HarmBench-Test & "
        f"{summary['n_strict_valid']} & "
        f"{summary['n_recovered_nonrefusal']}/{summary['n_strict_valid']} & "
        f"{summary['n_recovered_base_like']}/{summary['n_strict_valid']} & "
        f"{summary['mean_defended_feature_drift']:.3f} & "
        f"{summary['mean_floor_violation']:.3f} & "
        f"{summary['mean_delta_norm_relative_to_defended_feature_state']:.3f} \\\\"
    )
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
