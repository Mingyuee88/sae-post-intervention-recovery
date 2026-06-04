import argparse
import importlib.util
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

LETTERS = ["A", "B", "C", "D"]


def load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("base_recovery_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_prompt(prompt: str) -> Tuple[str, List[str]]:
    text = prompt.strip()
    pattern = re.compile(r"(?s)^(.*?)\n\nA\. (.*?)\nB\. (.*?)\nC\. (.*?)\nD\. (.*?)\n\nAnswer:\s*$")
    m = pattern.match(text)
    if not m:
        raise ValueError("Prompt does not match expected MCQ format.")
    question = m.group(1)
    choices = [m.group(2), m.group(3), m.group(4), m.group(5)]
    return question, choices


def format_prompt(question: str, choices: List[str]) -> str:
    return (
        f"{question}\n\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        f"Answer:"
    )


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def mean_or_none(xs: List[float]):
    if not xs:
        return None
    return sum(xs) / len(xs)


def score_value(base, logits_last, target_token_id, choice_token_ids, loss_mode):
    return float(base.objective_value(logits_last.float(), target_token_id, choice_token_ids, loss_mode).item())


def main():
    ap = argparse.ArgumentParser(description="24-permutation evaluator with strict 24/24 filtering and choice-only objective.")
    ap.add_argument("--manifest", required=True, help="prompt_pool/manifest.json")
    ap.add_argument("--base_recovery_script", required=True, help="Path to recovery_unlearning_choice_only.py")
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
    ap.add_argument("--output_dir", default="perm_eval_choice_outputs")
    ap.add_argument("--limit_questions", type=int, default=None)
    ap.add_argument("--save_permutation_prompts", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    base = load_module_from_path(args.base_recovery_script)

    model_name = manifest["model_name"]
    sae_release = manifest["sae_release"]
    sae_id = manifest["sae_id"]
    retain_threshold = manifest["retain_threshold"]
    n_features = manifest["n_features"]
    multiplier = manifest["multiplier"]
    feature_ids = manifest["feature_ids"]

    device = base.setup_environment()
    llm_dtype = base.dtype_from_string("bfloat16")

    model, sae, hook_name = base.load_model_and_sae_from_release(
        model_name=model_name,
        sae_release=sae_release,
        sae_id=sae_id,
        device=device,
        llm_dtype=llm_dtype,
    )
    feature_idx = torch.tensor(feature_ids, device=device, dtype=torch.long)

    items = manifest["items"]
    if args.limit_questions is not None:
        items = items[: args.limit_questions]

    all_rows: List[Dict[str, Any]] = []
    per_question_summary: List[Dict[str, Any]] = []

    for item in items:
        sample_id = int(item["sample_id"])
        prompt_text = Path(item["prompt_file"]).read_text(encoding="utf-8")
        question, choices = parse_prompt(prompt_text)
        gold_letter_orig = item["gold_letter"]
        gold_idx_orig = LETTERS.index(gold_letter_orig)
        correct_choice_text = choices[gold_idx_orig]
        choice_token_ids = base.get_choice_token_ids(model, LETTERS)

        q_rows = []
        perms = list(itertools.permutations(range(4)))
        prompt_save_dir = Path(args.output_dir) / args.projection_mode / f"sample_{sample_id:03d}" / "permuted_prompts"

        for perm_id, perm in enumerate(perms):
            perm_choices = [choices[i] for i in perm]
            new_gold_idx = perm.index(gold_idx_orig)
            gold_letter_perm = LETTERS[new_gold_idx]
            target_token = " " + gold_letter_perm
            prompt_perm = format_prompt(question, perm_choices)

            if args.save_permutation_prompts:
                prompt_save_dir.mkdir(parents=True, exist_ok=True)
                (prompt_save_dir / f"perm_{perm_id:02d}.txt").write_text(prompt_perm, encoding="utf-8")

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

            base_score = score_value(base, base_logits[0, last_idx, :], target_token_id, choice_token_ids, args.loss_mode)
            direct_def_score = score_value(base, direct_def_logits[0, last_idx, :], target_token_id, choice_token_ids, args.loss_mode)
            fixed_def_score = score_value(base, fixed_def_logits[0, last_idx, :], target_token_id, choice_token_ids, args.loss_mode)

            base_correct = (base_choice == gold_letter_perm)
            direct_defended_wrong = (direct_def_choice != gold_letter_perm)
            fixed_defended_wrong = (fixed_def_choice != gold_letter_perm)
            valid_flip = base_correct and direct_defended_wrong

            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "orig_dataset_idx": int(item["idx"]),
                "perm_id": perm_id,
                "perm": list(perm),
                "gold_letter_original": gold_letter_orig,
                "gold_letter_perm": gold_letter_perm,
                "correct_choice_text": correct_choice_text,
                "base_choice": base_choice,
                "direct_defended_choice": direct_def_choice,
                "fixed_defended_choice": fixed_def_choice,
                "base_score": base_score,
                "direct_defended_score": direct_def_score,
                "fixed_defended_score": fixed_def_score,
                "base_correct": base_correct,
                "direct_defended_wrong": direct_defended_wrong,
                "fixed_defended_wrong": fixed_defended_wrong,
                "valid_flip": valid_flip,
                "recovered_choice": None,
                "recovered_score": None,
                "final_delta_norm": None,
                "final_act_drift_l2": None,
                "final_decode_drift_l2": None,
                "recovered_success": False,
            }
            q_rows.append(row)

        base_correct_count = sum(1 for r in q_rows if r["base_correct"])
        in_strict_subset = (base_correct_count == 24)
        if args.strict_base_24_24 and not in_strict_subset:
            per_question_summary.append({
                "sample_id": sample_id,
                "orig_dataset_idx": int(item["idx"]),
                "gold_letter_original": gold_letter_orig,
                "correct_choice_text": correct_choice_text,
                "n_permutations": len(q_rows),
                "n_base_correct": base_correct_count,
                "n_valid_flips": 0,
                "n_recovered": 0,
                "base_correct_rate": base_correct_count / len(q_rows),
                "recovery_rate_over_valid_flips": None,
                "in_strict_subset": False,
            })
            continue

        for row in q_rows:
            if row["valid_flip"]:
                perm = row["perm"]
                perm_choices = [choices[i] for i in perm]
                gold_letter_perm = row["gold_letter_perm"]
                target_token = " " + gold_letter_perm
                prompt_perm = format_prompt(question, perm_choices)
                tokens = model.to_tokens(prompt_perm)
                last_idx = tokens.shape[1] - 1
                target_token_id = model.to_single_token(target_token)

                with torch.no_grad():
                    _, defended_ref = base.build_direct_defended_reference(
                        model=model,
                        sae=sae,
                        hook_name=hook_name,
                        tokens=tokens,
                        feature_idx=feature_idx,
                        multiplier=multiplier,
                    )

                result = base.optimize_delta(
                    model=model,
                    sae=sae,
                    hook_name=hook_name,
                    tokens=tokens,
                    last_idx=last_idx,
                    feature_idx=feature_idx,
                    defended_ref=defended_ref,
                    target_token_id=target_token_id,
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
                row["recovered_choice"] = recovered_choice
                row["recovered_score"] = float(result["final_score"])
                row["final_delta_norm"] = float(result["final_delta_norm"])
                row["final_act_drift_l2"] = float(result["final_act_drift_l2"])
                row["final_decode_drift_l2"] = float(result["final_decode_drift_l2"])
                row["recovered_success"] = (recovered_choice == row["gold_letter_perm"])

        valid_flip_count = sum(1 for r in q_rows if r["valid_flip"])
        recovered_count = sum(1 for r in q_rows if r["recovered_success"])
        per_question_summary.append({
            "sample_id": sample_id,
            "orig_dataset_idx": int(item["idx"]),
            "gold_letter_original": gold_letter_orig,
            "correct_choice_text": correct_choice_text,
            "n_permutations": len(q_rows),
            "n_base_correct": base_correct_count,
            "n_valid_flips": valid_flip_count,
            "n_recovered": recovered_count,
            "base_correct_rate": base_correct_count / len(q_rows),
            "recovery_rate_over_valid_flips": (recovered_count / valid_flip_count) if valid_flip_count > 0 else None,
            "in_strict_subset": in_strict_subset,
        })
        all_rows.extend(q_rows)

    n_total = len(all_rows)
    n_base_correct = sum(1 for r in all_rows if r["base_correct"])
    n_valid_flips = sum(1 for r in all_rows if r["valid_flip"])
    n_recovered = sum(1 for r in all_rows if r["recovered_success"])
    kept_questions = [q for q in per_question_summary if (not args.strict_base_24_24) or q["in_strict_subset"]]

    aggregate = {
        "model_name": model_name,
        "sae_release": sae_release,
        "sae_id": sae_id,
        "retain_threshold": retain_threshold,
        "n_features": n_features,
        "multiplier": multiplier,
        "projection_mode": args.projection_mode,
        "loss_mode": args.loss_mode,
        "strict_base_24_24": args.strict_base_24_24,
        "lambda_act": args.lambda_act,
        "lambda_decode": args.lambda_decode,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "max_delta_norm": args.max_delta_norm,
        "n_questions_before_filter": len(per_question_summary),
        "n_questions_after_filter": len(kept_questions),
        "n_total_permutations": n_total,
        "n_base_correct": n_base_correct,
        "n_valid_flips": n_valid_flips,
        "n_recovered": n_recovered,
        "base_correct_rate_over_all_permutations": (n_base_correct / n_total) if n_total > 0 else None,
        "recovery_success_rate_over_valid_flips": (n_recovered / n_valid_flips) if n_valid_flips > 0 else None,
        "avg_base_score_over_valid_flips": mean_or_none([r["base_score"] for r in all_rows if r["valid_flip"]]),
        "avg_fixed_defended_score_over_valid_flips": mean_or_none([r["fixed_defended_score"] for r in all_rows if r["valid_flip"]]),
        "avg_recovered_score_over_valid_flips": mean_or_none([r["recovered_score"] for r in all_rows if r["valid_flip"] and r["recovered_score"] is not None]),
        "avg_score_gain_vs_defended_over_valid_flips": mean_or_none([r["recovered_score"] - r["fixed_defended_score"] for r in all_rows if r["valid_flip"] and r["recovered_score"] is not None]),
        "avg_final_act_drift_l2_over_valid_flips": mean_or_none([r["final_act_drift_l2"] for r in all_rows if r["valid_flip"] and r["final_act_drift_l2"] is not None]),
        "avg_final_decode_drift_l2_over_valid_flips": mean_or_none([r["final_decode_drift_l2"] for r in all_rows if r["valid_flip"] and r["final_decode_drift_l2"] is not None]),
        "per_question": per_question_summary,
    }

    out_dir = Path(args.output_dir) / args.projection_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(aggregate, out_dir / "aggregate.json")
    with open(out_dir / "permutation_rows.jsonl", "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "aggregate_json": str(out_dir / "aggregate.json"),
        "rows_jsonl": str(out_dir / "permutation_rows.jsonl"),
        "n_questions_after_filter": aggregate["n_questions_after_filter"],
        "n_total_permutations": n_total,
        "n_base_correct": n_base_correct,
        "n_valid_flips": n_valid_flips,
        "n_recovered": n_recovered,
        "base_correct_rate_over_all_permutations": aggregate["base_correct_rate_over_all_permutations"],
        "recovery_success_rate_over_valid_flips": aggregate["recovery_success_rate_over_valid_flips"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
