#!/usr/bin/env python3
"""Prepare per-flip Encoder/None interventions for unified post-hoc eval.

This is intentionally a thin wrapper around the existing unlearning recovery
code. It saves one delta.pt per valid answer-choice permutation so that the
post-hoc evaluator can rerun the same final intervention under a shared metric.
"""

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from permutation_seqwide import LETTERS, format_prompt, mean_or_none, parse_prompt, save_json, score_value


def load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("base_recovery_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_sample_ids(text: Optional[str]) -> Optional[set[int]]:
    if text is None or text.strip() == "":
        return None
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def resolve_manifest_file(manifest_path: Path, file_text: str) -> Path:
    p = Path(file_text)
    if p.is_absolute() or p.exists():
        return p
    candidates = [
        manifest_path.parent / p,
        manifest_path.parent.parent / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Save per-valid-flip deltas for post-hoc unlearning recovery eval.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base_recovery_script", required=True)
    ap.add_argument("--projection_mode", default="encoder", choices=["none", "encoder"])
    ap.add_argument("--lambda_act", type=float, default=0.0)
    ap.add_argument("--lambda_decode", type=float, default=0.0)
    ap.add_argument("--num_steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--max_delta_norm", type=float, default=20.0)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss_mode", default="choice_margin", choices=["choice_margin", "choice_ce", "vocab_margin"])
    ap.add_argument("--strict_base_24_24", action="store_true")
    ap.add_argument("--sample_ids", default=None, help="Comma-separated sample IDs to include, e.g. 1,2,8.")
    ap.add_argument("--limit_questions", type=int, default=None)
    ap.add_argument("--max_valid_flips_per_question", type=int, default=None)
    ap.add_argument("--max_total_valid_flips", type=int, default=None)
    ap.add_argument("--output_dir", default="posthoc_interventions")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = load_module_from_path(args.base_recovery_script)
    sample_filter = parse_sample_ids(args.sample_ids)

    model_name = manifest["model_name"]
    sae_release = manifest["sae_release"]
    sae_id = manifest["sae_id"]
    retain_threshold = float(manifest["retain_threshold"])
    n_features = int(manifest["n_features"])
    multiplier = float(manifest["multiplier"])
    feature_ids = [int(x) for x in manifest["feature_ids"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_dtype = base.dtype_from_string("bfloat16")
    model, sae, hook_name = base.load_model_and_sae_from_release(
        model_name=model_name,
        sae_release=sae_release,
        sae_id=sae_id,
        device=device,
        llm_dtype=llm_dtype,
    )
    feature_idx = torch.tensor(feature_ids, device=device, dtype=torch.long)
    choice_token_ids = base.get_choice_token_ids(model, LETTERS)

    items = manifest["items"]
    if args.limit_questions is not None:
        items = items[: args.limit_questions]

    out_root = Path(args.output_dir) / args.projection_mode
    rows: List[Dict[str, Any]] = []
    per_question: List[Dict[str, Any]] = []
    total_prepared = 0

    for item in items:
        sample_id = int(item["sample_id"])
        if sample_filter is not None and sample_id not in sample_filter:
            continue

        prompt_text = resolve_manifest_file(manifest_path, item["prompt_file"]).read_text(encoding="utf-8")
        question, choices = parse_prompt(prompt_text)
        gold_letter_orig = item["gold_letter"]
        gold_idx_orig = LETTERS.index(gold_letter_orig)
        correct_choice_text = choices[gold_idx_orig]

        q_rows: List[Dict[str, Any]] = []
        for perm_id, perm in enumerate(itertools.permutations(range(4))):
            perm_choices = [choices[i] for i in perm]
            new_gold_idx = perm.index(gold_idx_orig)
            gold_letter_perm = LETTERS[new_gold_idx]
            target_token = " " + gold_letter_perm
            prompt_perm = format_prompt(question, perm_choices)
            tokens = model.to_tokens(prompt_perm)
            last_idx = tokens.shape[1] - 1
            target_token_id = model.to_single_token(target_token)

            with torch.no_grad():
                base_logits = model(tokens)
                direct_def_logits, defended_ref = base.build_direct_defended_reference(
                    model=model,
                    sae=sae,
                    hook_name=hook_name,
                    tokens=tokens,
                    feature_idx=feature_idx,
                    multiplier=multiplier,
                )
                fixed_def_hook = base.FixedDirectDefendedPlusDeltaHook(
                    sae=sae,
                    feature_idx=feature_idx,
                    defended_ref=defended_ref,
                    delta_last=None,
                    state={},
                )
                fixed_def_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, fixed_def_hook)])

            base_logits_last = base_logits[0, last_idx, :].detach().float().cpu()
            direct_def_logits_last = direct_def_logits[0, last_idx, :].detach().float().cpu()
            fixed_def_logits_last = fixed_def_logits[0, last_idx, :].detach().float().cpu()

            base_choice = base.greedy_choice_letter(model, base_logits_last, LETTERS)
            direct_def_choice = base.greedy_choice_letter(model, direct_def_logits_last, LETTERS)
            fixed_def_choice = base.greedy_choice_letter(model, fixed_def_logits_last, LETTERS)
            base_correct = base_choice == gold_letter_perm
            direct_defended_wrong = direct_def_choice != gold_letter_perm
            fixed_defended_wrong = fixed_def_choice != gold_letter_perm
            valid_flip = base_correct and direct_defended_wrong

            q_rows.append(
                {
                    "sample_id": sample_id,
                    "orig_dataset_idx": int(item["idx"]),
                    "perm_id": perm_id,
                    "perm": list(perm),
                    "prompt": prompt_perm,
                    "gold_letter_original": gold_letter_orig,
                    "gold_letter_perm": gold_letter_perm,
                    "correct_choice_text": correct_choice_text,
                    "base_choice": base_choice,
                    "direct_defended_choice": direct_def_choice,
                    "fixed_defended_choice": fixed_def_choice,
                    "base_score": score_value(base, base_logits[0, last_idx, :], target_token_id, choice_token_ids, args.loss_mode),
                    "direct_defended_score": score_value(base, direct_def_logits[0, last_idx, :], target_token_id, choice_token_ids, args.loss_mode),
                    "fixed_defended_score": score_value(base, fixed_def_logits[0, last_idx, :], target_token_id, choice_token_ids, args.loss_mode),
                    "base_correct": base_correct,
                    "direct_defended_wrong": direct_defended_wrong,
                    "fixed_defended_wrong": fixed_defended_wrong,
                    "valid_flip": valid_flip,
                    "recovered_choice": None,
                    "recovered_score": None,
                    "final_delta_norm": None,
                    "recovered_success": False,
                    "_tokens": tokens,
                    "_last_idx": last_idx,
                    "_target_token_id": target_token_id,
                    "_defended_ref": defended_ref,
                }
            )

        base_correct_count = sum(1 for r in q_rows if r["base_correct"])
        in_strict_subset = base_correct_count == 24
        if args.strict_base_24_24 and not in_strict_subset:
            continue

        valid_seen = 0
        valid_prepared = 0
        for row in q_rows:
            if not row["valid_flip"]:
                continue
            if args.max_valid_flips_per_question is not None and valid_seen >= args.max_valid_flips_per_question:
                continue
            if args.max_total_valid_flips is not None and total_prepared >= args.max_total_valid_flips:
                continue
            valid_seen += 1

            result = base.optimize_delta(
                model=model,
                sae=sae,
                hook_name=hook_name,
                tokens=row["_tokens"],
                last_idx=row["_last_idx"],
                feature_idx=feature_idx,
                defended_ref=row["_defended_ref"],
                target_token_id=row["_target_token_id"],
                choice_token_ids=choice_token_ids,
                loss_mode=args.loss_mode,
                num_steps=args.num_steps,
                lr=args.lr,
                max_delta_norm=args.max_delta_norm,
                projection_mode=args.projection_mode,
                lambda_act=args.lambda_act,
                lambda_decode=args.lambda_decode,
                ridge=args.ridge,
                seed=args.seed,
            )
            final_logits_last = result["final_logits_last"]
            recovered_choice = base.greedy_choice_letter(model, final_logits_last, LETTERS)

            case_dir = out_root / "questions" / f"sample_{row['sample_id']:03d}" / f"perm_{row['perm_id']:02d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result["delta"], case_dir / "delta.pt")
            torch.save(row["_defended_ref"], case_dir / "defended_ref.pt")

            for private_key in ["_tokens", "_last_idx", "_target_token_id", "_defended_ref"]:
                row.pop(private_key, None)
            row.update(
                {
                    "recovered_choice": recovered_choice,
                    "recovered_score": float(result["final_score"]),
                    "final_delta_norm": float(result["final_delta_norm"]),
                    "final_act_drift_l2": float(result["final_act_drift_l2"]),
                    "final_decode_drift_l2": float(result["final_decode_drift_l2"]),
                    "recovered_success": recovered_choice == row["gold_letter_perm"],
                    "delta_pt": str(case_dir / "delta.pt"),
                    "defended_ref_pt": str(case_dir / "defended_ref.pt"),
                    "summary_json": str(case_dir / "summary.json"),
                }
            )
            save_json(row, case_dir / "summary.json")
            rows.append(row)
            total_prepared += 1
            valid_prepared += 1

        per_question.append(
            {
                "sample_id": sample_id,
                "orig_dataset_idx": int(item["idx"]),
                "n_base_correct": base_correct_count,
                "in_strict_subset": in_strict_subset,
                "n_valid_flips_prepared": valid_prepared,
                "n_recovered": sum(1 for r in rows if r["sample_id"] == sample_id and r["recovered_success"]),
            }
        )

    aggregate = {
        "model_name": model_name,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "retain_threshold": retain_threshold,
        "n_features": n_features,
        "multiplier": multiplier,
        "feature_ids": feature_ids,
        "projection_mode": args.projection_mode,
        "strict_base_24_24": args.strict_base_24_24,
        "n_questions_after_filter": len(per_question),
        "n_valid_flips": len(rows),
        "n_recovered": sum(1 for r in rows if r["recovered_success"]),
        "recovery_success_rate_over_valid_flips": (
            sum(1 for r in rows if r["recovered_success"]) / len(rows) if rows else None
        ),
        "avg_final_delta_norm": mean_or_none([r["final_delta_norm"] for r in rows if r["final_delta_norm"] is not None]),
        "per_question": per_question,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    save_json(aggregate, out_root / "aggregate.json")
    with open(out_root / "permutation_rows.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"aggregate_json": str(out_root / "aggregate.json"), "rows_jsonl": str(out_root / "permutation_rows.jsonl"), **aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
