#!/usr/bin/env python3
"""Unified post-hoc evaluator for unlearning recovery methods.

The evaluator reruns each saved final intervention under one shared protocol:
same valid flips, same defended SAE feature set, choice-readout token scope, and
the same relative-L2 normalization for drift/reactivation.
"""

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_bench.recovery_core import FixedDirectDefendedPlusDeltaHook, build_direct_defended_reference, cast_for_sae
from permutation_seqwide import LETTERS


def load_module_from_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def key_of(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["sample_id"]), int(row["perm_id"])


def resolve_existing_path(path_text: Optional[str], search_roots: Iterable[Path]) -> Optional[Path]:
    if not path_text:
        return None
    p = Path(path_text)
    if p.exists():
        return p
    for root in search_roots:
        cand = root / p.name
        if cand.exists():
            return cand
    return p


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


class CaptureSelectedFeatureHook:
    def __init__(self, sae, feature_idx: torch.Tensor, readout_pos: int, state: Dict[str, torch.Tensor]):
        self.sae = sae
        self.feature_idx = feature_idx
        self.readout_pos = int(readout_pos)
        self.state = state

    def __call__(self, resid: torch.Tensor, hook=None, **kwargs):
        z = self.sae.encode(cast_for_sae(resid, self.sae)).float()[0, self.readout_pos, self.feature_idx]
        self.state["z_readout"] = z.detach().float().cpu()
        return resid


def choice_from_logits(model, logits_pos: torch.Tensor) -> str:
    scores = []
    for letter in LETTERS:
        tok = model.to_single_token(" " + letter)
        scores.append((letter, float(logits_pos[tok].item())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


@torch.no_grad()
def defended_baseline(model, sae, hook_name: str, feature_idx: torch.Tensor, multiplier: float, prompt: str):
    tokens = model.to_tokens(prompt)
    readout_pos = tokens.shape[1] - 1
    logits, defended_ref = build_direct_defended_reference(
        model=model,
        sae=sae,
        hook_name=hook_name,
        tokens=tokens,
        feature_idx=feature_idx,
        multiplier=multiplier,
    )
    z0 = defended_ref["act_ref"].detach().float().cpu()
    logits_pos = logits[0, readout_pos, :].detach().float().cpu()
    return tokens, readout_pos, defended_ref, z0, logits_pos


@torch.no_grad()
def eval_delta_method(
    model,
    sae,
    hook_name: str,
    feature_idx: torch.Tensor,
    defended_ref: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    readout_pos: int,
    delta_pt: Path,
):
    delta = torch.load(delta_pt, map_location="cpu").to(tokens.device)
    state: Dict[str, Any] = {}
    hook_obj = FixedDirectDefendedPlusDeltaHook(
        sae=sae,
        feature_idx=feature_idx,
        defended_ref=defended_ref,
        delta_last=delta,
        state={},
    )
    capture = CaptureSelectedFeatureHook(sae, feature_idx, readout_pos, state)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_obj), (hook_name, capture)])
    logits_pos = logits[0, readout_pos, :].detach().float().cpu()
    return logits_pos, state["z_readout"]


@torch.no_grad()
def eval_oabd_method(
    model,
    clamp_sae,
    hook_name: str,
    feature_idx: torch.Tensor,
    multiplier: float,
    prompt: str,
    gold_letter: str,
    soft_suffix_pt: Path,
    oabd,
    suffix_len: int,
    suffix_placeholder_token: str,
):
    placeholder_id = model.to_single_token(suffix_placeholder_token)
    token_pack = oabd.build_suffix_training_tokens(
        model=model,
        prompt=prompt,
        target_continuation=" " + gold_letter,
        suffix_len=suffix_len,
        placeholder_id=placeholder_id,
    )
    prompt_tokens = token_pack["prompt_tokens"]
    full_tokens = token_pack["full_tokens"]
    prompt_positions = token_pack["prompt_positions"]
    target_positions = token_pack["target_positions"]
    answer_positions = token_pack["answer_positions"]
    attack_positions = token_pack["attack_positions"]
    suffix_positions = token_pack["suffix_positions"]
    suffix_start = int(suffix_positions[0].item())

    _, prompt_only_ref = oabd.build_prompt_only_defended_reference(
        model=model,
        sae=clamp_sae,
        hook_name=hook_name,
        prompt_tokens=prompt_tokens,
        clamp_feature_idx=feature_idx,
        multiplier=multiplier,
    )
    soft_suffix = torch.load(soft_suffix_pt, map_location="cpu").to(device=full_tokens.device, dtype=torch.float32)
    method_state: Dict[str, Any] = {}
    defended_hook = oabd.FixedDefendedPrefixWithSoftSuffixHook(
        sae=clamp_sae,
        clamp_feature_idx=feature_idx,
        prompt_positions=prompt_positions,
        answer_positions=answer_positions,
        attack_positions=attack_positions,
        prompt_only_ref=prompt_only_ref,
        state=method_state,
    )
    readout_pos = int(target_positions[0].item())
    capture_state: Dict[str, torch.Tensor] = {}
    capture = CaptureSelectedFeatureHook(clamp_sae, feature_idx, readout_pos, capture_state)
    logits = model.run_with_hooks(
        full_tokens,
        fwd_hooks=[
            ("hook_embed", oabd.SoftSuffixEmbedHook(soft_suffix=soft_suffix, start_pos=suffix_start)),
            (hook_name, defended_hook),
            (hook_name, capture),
        ],
    )
    logits_pos = logits[0, readout_pos, :].detach().float().cpu()
    return logits_pos, capture_state["z_readout"]


def metric_values(z0: torch.Tensor, zrec: torch.Tensor, direction_sign: float, eps: float) -> Dict[str, float]:
    diff = zrec.float() - z0.float()
    defended_norm = float(z0.float().norm(p=2).item())
    denom = defended_norm + eps
    drift_abs = float(diff.norm(p=2).item())
    drift = drift_abs / denom
    violation_vec = torch.clamp(float(direction_sign) * diff, min=0.0)
    violation_abs = float(violation_vec.norm(p=2).item())
    violation = violation_abs / denom
    self_drift = float((z0.float() - z0.float()).norm(p=2).item()) / denom
    self_violation = float(torch.clamp(float(direction_sign) * (z0.float() - z0.float()), min=0.0).norm(p=2).item()) / denom
    return {
        "defended_norm": defended_norm,
        "drift_abs_l2": drift_abs,
        "violation_abs_l2": violation_abs,
        "drift_relative_l2": drift,
        "violation_relative_l2": violation,
        "self_drift": self_drift,
        "self_violation": self_violation,
    }


def aggregate_method(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_q: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[int(row["sample_id"])].append(row)
    per_question = []
    for sample_id, qrows in sorted(by_q.items()):
        n_valid = len(qrows)
        n_rec = sum(1 for r in qrows if r["recovered_correct"])
        per_question.append(
            {
                "question_id": f"sample_{sample_id:03d}",
                "sample_id": sample_id,
                "orig_dataset_idx": qrows[0].get("orig_dataset_idx"),
                "n_valid_flips": n_valid,
                "n_recovered": n_rec,
                "recovery_rate": n_rec / n_valid if n_valid else None,
                "mean_defended_feature_drift": mean([r["defended_feature_drift"] for r in qrows]),
                "mean_reactivation_or_floor_violation": mean([r["reactivation_or_floor_violation"] for r in qrows]),
            }
        )
    n = len(rows)
    n_rec = sum(1 for r in rows if r["recovered_correct"])
    return {
        "n_valid_flips": n,
        "n_recovered": n_rec,
        "recovery_rate": n_rec / n if n else None,
        "mean_defended_feature_drift": mean([r["defended_feature_drift"] for r in rows]),
        "median_defended_feature_drift": median([r["defended_feature_drift"] for r in rows]),
        "mean_reactivation_or_floor_violation": mean([r["reactivation_or_floor_violation"] for r in rows]),
        "median_reactivation_or_floor_violation": median([r["reactivation_or_floor_violation"] for r in rows]),
        "mean_defended_feature_norm": mean([r["defended_feature_norm"] for r in rows]),
        "median_defended_feature_norm": median([r["defended_feature_norm"] for r in rows]),
        "n_near_zero_defended_norm": sum(1 for r in rows if r["defended_feature_norm"] <= 1e-8),
        "mean_defended_feature_drift_abs_l2": mean([r["defended_feature_drift_abs_l2"] for r in rows]),
        "median_defended_feature_drift_abs_l2": median([r["defended_feature_drift_abs_l2"] for r in rows]),
        "mean_reactivation_or_floor_violation_abs_l2": mean([r["reactivation_or_floor_violation_abs_l2"] for r in rows]),
        "median_reactivation_or_floor_violation_abs_l2": median([r["reactivation_or_floor_violation_abs_l2"] for r in rows]),
        "fraction_violation_gt_1e-3": mean([1.0 if r["reactivation_or_floor_violation"] > 1e-3 else 0.0 for r in rows]),
        "per_question": per_question,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified post-hoc unlearning recovery evaluator.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base_recovery_script", default="Aout/unlearning/recovery_unlearning_choice_only_seqwide_act.py")
    ap.add_argument("--oabd_script", default="Aout/unlearning/baseline_oabd_defended_suffix_fixedprefix_choiceonly_dualsummary.py")
    ap.add_argument("--oabd_rows_jsonl", required=True)
    ap.add_argument("--encoder_rows_jsonl", required=True)
    ap.add_argument("--none_rows_jsonl", required=True)
    ap.add_argument("--oabd_config_json", default=None)
    ap.add_argument("--suffix_len", type=int, default=None)
    ap.add_argument("--suffix_placeholder_token", default=" !")
    ap.add_argument("--direction_sign", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--rows_jsonl", required=True)
    args = ap.parse_args()

    base = load_module_from_path(str(REPO_ROOT / args.base_recovery_script), "base_recovery_module")
    oabd = load_module_from_path(str(REPO_ROOT / args.oabd_script), "oabd_module")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    oabd_rows_raw = [r for r in read_jsonl(Path(args.oabd_rows_jsonl)) if r.get("valid_flip")]
    encoder_rows_raw = [r for r in read_jsonl(Path(args.encoder_rows_jsonl)) if r.get("valid_flip")]
    none_rows_raw = [r for r in read_jsonl(Path(args.none_rows_jsonl)) if r.get("valid_flip")]
    oabd_by_key = {key_of(r): r for r in oabd_rows_raw}
    encoder_by_key = {key_of(r): r for r in encoder_rows_raw}
    none_by_key = {key_of(r): r for r in none_rows_raw}
    keys = sorted(set(oabd_by_key) & set(encoder_by_key) & set(none_by_key))
    if not keys:
        raise RuntimeError("No shared valid flips across OABD/Encoder/None rows.")

    suffix_len = args.suffix_len
    if suffix_len is None and args.oabd_config_json:
        cfg = json.loads(Path(args.oabd_config_json).read_text(encoding="utf-8"))
        suffix_len = int(cfg.get("suffix_len", 16))
        args.suffix_placeholder_token = cfg.get("suffix_placeholder_token", args.suffix_placeholder_token)
    if suffix_len is None:
        suffix_len = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_dtype = base.dtype_from_string("bfloat16")
    model, sae, hook_name = base.load_model_and_sae_from_release(
        model_name=manifest["model_name"],
        sae_release=manifest["sae_release"],
        sae_id=manifest["sae_id"],
        device=device,
        llm_dtype=llm_dtype,
    )
    feature_ids = [int(x) for x in manifest["feature_ids"]]
    feature_idx = torch.tensor(feature_ids, device=device, dtype=torch.long)
    multiplier = float(manifest["multiplier"])

    method_rows: Dict[str, List[Dict[str, Any]]] = {"OABD": [], "Encoder": [], "None": []}
    self_drifts = []
    self_violations = []

    for sample_id, perm_id in keys:
        ref_row = oabd_by_key[(sample_id, perm_id)]
        prompt = ref_row["prompt"]
        gold = ref_row["gold_letter_perm"]
        tokens, readout_pos, defended_ref, z0, _ = defended_baseline(
            model=model,
            sae=sae,
            hook_name=hook_name,
            feature_idx=feature_idx,
            multiplier=multiplier,
            prompt=prompt,
        )

        eval_specs = [
            ("Encoder", encoder_by_key[(sample_id, perm_id)].get("delta_pt"), encoder_by_key[(sample_id, perm_id)]),
            ("None", none_by_key[(sample_id, perm_id)].get("delta_pt"), none_by_key[(sample_id, perm_id)]),
        ]
        for method, delta_path_text, row_src in eval_specs:
            if not delta_path_text:
                raise RuntimeError(f"{method} row lacks delta_pt for sample={sample_id} perm={perm_id}.")
            logits_pos, zrec = eval_delta_method(
                model=model,
                sae=sae,
                hook_name=hook_name,
                feature_idx=feature_idx,
                defended_ref=defended_ref,
                tokens=tokens,
                readout_pos=readout_pos,
                delta_pt=Path(delta_path_text),
            )
            recovered_choice = choice_from_logits(model, logits_pos)
            metrics = metric_values(z0, zrec, args.direction_sign, args.eps)
            self_drifts.append(metrics["self_drift"])
            self_violations.append(metrics["self_violation"])
            method_rows[method].append(
                {
                    "method": method,
                    "question_id": f"sample_{sample_id:03d}",
                    "sample_id": sample_id,
                    "orig_dataset_idx": ref_row.get("orig_dataset_idx"),
                    "permutation_id": perm_id,
                    "gold_choice": gold,
                    "recovered_choice": recovered_choice,
                    "recovered_correct": recovered_choice == gold,
                    "defended_feature_drift": metrics["drift_relative_l2"],
                    "reactivation_or_floor_violation": metrics["violation_relative_l2"],
                    "defended_feature_norm": metrics["defended_norm"],
                    "defended_feature_drift_abs_l2": metrics["drift_abs_l2"],
                    "reactivation_or_floor_violation_abs_l2": metrics["violation_abs_l2"],
                    "choice_readout_position_baseline": int(readout_pos),
                    "choice_readout_position_recovered": int(readout_pos),
                    "feature_dim": int(z0.numel()),
                    "source_recovered_choice": row_src.get("recovered_choice"),
                }
            )

        oabd_row = oabd_by_key[(sample_id, perm_id)]
        soft_suffix_path = resolve_existing_path(
            oabd_row.get("soft_suffix_pt"),
            [
                Path(args.oabd_rows_jsonl).parent,
                REPO_ROOT,
                REPO_ROOT / "Aout/unlearning",
            ],
        )
        if soft_suffix_path is None or not soft_suffix_path.exists():
            raise RuntimeError(f"Missing OABD soft suffix for sample={sample_id} perm={perm_id}: {oabd_row.get('soft_suffix_pt')}")
        logits_pos, zrec = eval_oabd_method(
            model=model,
            clamp_sae=sae,
            hook_name=hook_name,
            feature_idx=feature_idx,
            multiplier=multiplier,
            prompt=prompt,
            gold_letter=gold,
            soft_suffix_pt=soft_suffix_path,
            oabd=oabd,
            suffix_len=suffix_len,
            suffix_placeholder_token=args.suffix_placeholder_token,
        )
        recovered_choice = choice_from_logits(model, logits_pos)
        metrics = metric_values(z0, zrec, args.direction_sign, args.eps)
        self_drifts.append(metrics["self_drift"])
        self_violations.append(metrics["self_violation"])
        method_rows["OABD"].append(
            {
                "method": "OABD",
                "question_id": f"sample_{sample_id:03d}",
                "sample_id": sample_id,
                "orig_dataset_idx": ref_row.get("orig_dataset_idx"),
                "permutation_id": perm_id,
                "gold_choice": gold,
                "recovered_choice": recovered_choice,
                "recovered_correct": recovered_choice == gold,
                "defended_feature_drift": metrics["drift_relative_l2"],
                "reactivation_or_floor_violation": metrics["violation_relative_l2"],
                "defended_feature_norm": metrics["defended_norm"],
                "defended_feature_drift_abs_l2": metrics["drift_abs_l2"],
                "reactivation_or_floor_violation_abs_l2": metrics["violation_abs_l2"],
                "choice_readout_position_baseline": int(readout_pos),
                "choice_readout_position_recovered": "soft_suffix_last",
                "feature_dim": int(z0.numel()),
                "source_recovered_choice": oabd_row.get("recovered_choice"),
            }
        )

    output = {
        "metadata": {
            "model_name": manifest["model_name"],
            "sae_release": manifest["sae_release"],
            "sae_id": manifest["sae_id"],
            "strict_base_24_24": True,
            "n_questions_after_filter": len({k[0] for k in keys}),
            "n_valid_flips": len(keys),
            "token_scope": "choice_readout",
            "drift_type": "posthoc_choice_readout_defended_feature_drift",
            "normalization": "relative_l2_by_defended_norm",
            "reactivation_direction_sign": args.direction_sign,
            "eps": args.eps,
            "same_feature_set": True,
            "same_valid_flips": True,
            "same_token_scope": True,
            "same_posthoc_evaluator": True,
            "feature_dim": len(feature_ids),
            "feature_ids": feature_ids,
            "valid_flip_keys": [{"sample_id": s, "perm_id": p} for s, p in keys],
        },
        "sanity_checks": {
            "baseline_self_drift_max": max(self_drifts) if self_drifts else None,
            "baseline_self_violation_max": max(self_violations) if self_violations else None,
            "all_methods_share_keys": True,
        },
        "methods": {name: aggregate_method(rows) for name, rows in method_rows.items()},
    }
    write_json(output, Path(args.output_json))
    rows_path = Path(args.rows_jsonl)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rows_path, "w", encoding="utf-8") as f:
        for name in ["OABD", "Encoder", "None"]:
            for row in method_rows[name]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output_json": args.output_json, "rows_jsonl": args.rows_jsonl, "methods": output["methods"], "sanity_checks": output["sanity_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
