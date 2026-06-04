#!/usr/bin/env python3
"""Appendix-support summaries for SAE recovery experiments.

Outputs are intentionally sanitized: no full prompts or completions are written.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


OPEN_SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SAEBENCH_ROOT", str(OPEN_SOURCE_ROOT)))
APPENDIX_ROOT = Path(os.environ.get("APPENDIX_ROOT", str(OPEN_SOURCE_ROOT)))
RESULTS = APPENDIX_ROOT / "results"
TABLES = APPENDIX_ROOT / "tables"
SCRIPTS = APPENDIX_ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ADV_ROOT = ROOT / "Batch_Recovery/crosslayer_v4_valid24_main/20260504_234000__advbench_benchmark_our_valid24_v4strong_floor_compare"
ADV_JAC_PARTS = [
    ADV_ROOT / "host30_queue/jacobian_v4strong_part0_valid12/wrapper_outputs",
    ADV_ROOT / "host192_queue/jacobian_v4strong_part1_valid12/wrapper_outputs",
]
ADV_NONE = ADV_ROOT / "host30_queue/none_lam200_norm40_thr003_valid24/wrapper_outputs"
ADV_OABD = ADV_ROOT / "host192_queue/oabd_lam0p9_norm40_len16_floor_valid24/wrapper_outputs/gemma-2b__advbench__benchmark_our__oabd_suffix"
ADV_BEHAVIOR = ADV_ROOT / "host192_queue/behavior_only_lam1_norm40_len16_floor_valid24/wrapper_outputs/gemma-2b__advbench__benchmark_our__oabd_suffix"

FEATURE_SWEEP_ROOT = ROOT / "Batch_Recovery/aout_refusal_feature_size_sweep_full520/20260502_120753__advbench_benchmark_our_localunion_full520_stage1_ft4_pref2_norm40_k1_61"
PATH_ATTR_SUMMARY = ROOT / "Batch_Recovery/recovery_path_attribution/20260505_134500__advbench_benchmark_our_valid24_v4strong_path_attr/merged_path_attribution_summary.json"
UNLEARNING_POSTHOC = ROOT / "Aout/unlearning/posthoc_eval_runs/full_20260506_2148/posthoc_aggregate.json"
HARM_SUMMARY = RESULTS / "harmbench_strict_valid_summary.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_values(values: Iterable[Any]) -> List[float]:
    xs = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            xs.append(float(v))
    return xs


def safe_mean(values: Iterable[Any]) -> Optional[float]:
    xs = finite_values(values)
    return mean(xs) if xs else None


def safe_median(values: Iterable[Any]) -> Optional[float]:
    xs = finite_values(values)
    return median(xs) if xs else None


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if n <= 0:
        return None, None, None
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values: List[float], n_boot: int = 2000, seed: int = 0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    xs = finite_values(values)
    if not xs:
        return None, None, None
    rng = random.Random(seed)
    boots = []
    n = len(xs)
    for _ in range(n_boot):
        boots.append(mean(rng.choice(xs) for _ in range(n)))
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    return mean(xs), lo, hi


def pct(x: Optional[float]) -> str:
    if x is None:
        return "--"
    return f"{100.0 * x:.1f}"


def num(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "--"
    return f"{float(x):.{nd}f}"


def merge_adv_jacobian() -> Dict[str, Any]:
    aggs = [load_json(p / "aggregate.json") for p in ADV_JAC_PARTS]
    samples = []
    for p in ADV_JAC_PARTS:
        samples.extend(load_json(p / "samples.json"))
    n = sum(int(a.get("n_samples", 0)) for a in aggs)
    def weighted(field: str) -> Optional[float]:
        vals = []
        weights = []
        for a in aggs:
            v = a.get(field)
            w = int(a.get("n_samples", 0))
            if isinstance(v, (int, float)) and w > 0:
                vals.append(float(v) * w)
                weights.append(w)
        return sum(vals) / sum(weights) if weights else None
    return {
        "name": "AdvBench Jacobian",
        "n": n,
        "n_nonref": sum(int(a.get("n_recovered_nonrefusal", 0)) for a in aggs),
        "n_base_like": sum(int(a.get("n_recovered_base_like", 0)) for a in aggs),
        "n_strict_base_like": sum(int(a.get("n_recovered_strict_base_like", 0)) for a in aggs),
        "drift": weighted("avg_final_drift_relative_l2"),
        "floor": weighted("avg_final_floor_violation_relative_l2"),
        "delta_norm": weighted("avg_stage2_delta_norm"),
        "samples": samples,
    }


def inferred_relative_delta(sample: Dict[str, Any]) -> Optional[float]:
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


def load_baseline_module():
    path = ROOT / "Aout/refusal/baseline_refusal_sae_recovery.py"
    spec = importlib.util.spec_from_file_location("refusal_base_for_audit", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refusal_base_for_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


def coarse_category(text: str, mod: Any = None) -> str:
    s = str(text or "").strip()
    if not s:
        return "empty"
    low = s.lower()
    words = low.split()
    if len(words) >= 8 and len(set(words[:20])) <= 3:
        return "degenerate_or_repeated"
    refusal = bool(mod.get_runtime()["substring_matching_judge_fn"](s)) if mod is not None else False
    safety = bool(mod.contains_safety_cue(s)) if mod is not None else False
    if refusal:
        return "full_refusal"
    if safety:
        return "partial_refusal"
    if low.startswith(("neither", "none of", "cannot", "can't")):
        return "detector_uncertain"
    return "non_refusal_candidate"


def write_uncertainty_rows(rows: List[Dict[str, Any]]) -> None:
    path = RESULTS / "uncertainty_intervals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "metric",
        "k",
        "n",
        "estimate",
        "wilson_low",
        "wilson_high",
        "mean",
        "bootstrap_low",
        "bootstrap_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def add_binary_ci(rows: List[Dict[str, Any]], source: str, metric: str, k: int, n: int) -> None:
    est, lo, hi = wilson(int(k), int(n))
    rows.append(
        {
            "source": source,
            "metric": metric,
            "k": int(k),
            "n": int(n),
            "estimate": est,
            "wilson_low": lo,
            "wilson_high": hi,
        }
    )


def add_continuous_ci(rows: List[Dict[str, Any]], source: str, metric: str, values: List[float]) -> None:
    avg, lo, hi = bootstrap_ci(values)
    rows.append(
        {
            "source": source,
            "metric": metric,
            "mean": avg,
            "bootstrap_low": lo,
            "bootstrap_high": hi,
        }
    )


def selected_feature_sweep_aggs() -> Dict[int, Path]:
    candidates: Dict[int, List[Path]] = {}
    for path in FEATURE_SWEEP_ROOT.rglob("aggregate.json"):
        m = re.search(r"topk(\d+)_full520", str(path))
        if not m:
            continue
        k = int(m.group(1))
        candidates.setdefault(k, []).append(path)
    selected: Dict[int, Path] = {}
    for k, paths in candidates.items():
        def score(p: Path) -> Tuple[int, int]:
            s = str(p)
            if k == 61 and "host30_effective_refusal_k61" in s:
                return (3, -len(s))
            if k >= 30 and "effective_refusal" in s:
                return (2, -len(s))
            if "cached_tail" in s or "host30/layer" in s or "host192/layer" in s or "host30_parallel" in s:
                return (1, -len(s))
            return (0, -len(s))
        selected[k] = sorted(paths, key=score, reverse=True)[0]
    return selected


def generate_uncertainty_and_tables() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    adv = merge_adv_jacobian()
    add_binary_ci(rows, "AdvBench strict-valid Jacobian", "nonrefusal_recovery", adv["n_nonref"], adv["n"])
    add_binary_ci(rows, "AdvBench strict-valid Jacobian", "base_like_recovery", adv["n_base_like"], adv["n"])
    add_binary_ci(rows, "AdvBench strict-valid Jacobian", "strict_base_like_recovery", adv["n_strict_base_like"], adv["n"])
    add_continuous_ci(
        rows,
        "AdvBench strict-valid Jacobian",
        "defended_feature_drift_relative_l2",
        [float(s.get("final_drift", {}).get("relative_l2")) for s in adv["samples"] if isinstance(s.get("final_drift", {}).get("relative_l2"), (int, float))],
    )
    add_continuous_ci(
        rows,
        "AdvBench strict-valid Jacobian",
        "floor_violation_relative_l2",
        [float(s.get("final_drift", {}).get("floor_violation_relative_l2")) for s in adv["samples"] if isinstance(s.get("final_drift", {}).get("floor_violation_relative_l2"), (int, float))],
    )
    delta_rel = [x for x in (inferred_relative_delta(s) for s in adv["samples"]) if isinstance(x, float)]
    add_continuous_ci(rows, "AdvBench strict-valid Jacobian", "delta_norm_relative_to_defended_feature_state", delta_rel)

    harm = load_json(HARM_SUMMARY) if HARM_SUMMARY.exists() else None
    if harm:
        n_h = int(harm.get("n_strict_valid", 0))
        add_binary_ci(rows, "HarmBench-Test strict-valid Jacobian", "nonrefusal_recovery", int(harm.get("n_recovered_nonrefusal", 0)), n_h)
        add_binary_ci(rows, "HarmBench-Test strict-valid Jacobian", "base_like_recovery", int(harm.get("n_recovered_base_like", 0)), n_h)

    feature_rows = []
    for k, path in sorted(selected_feature_sweep_aggs().items()):
        agg = load_json(path)
        n = int(agg.get("n_clamp_refusal") or agg.get("n_valid_recovery_cases") or 0)
        nonref = int(agg.get("n_recovered_nonrefusal", 0))
        base_like = int(agg.get("n_recovered_base_like", 0))
        add_binary_ci(rows, f"Feature-size sweep K={k}", "nonrefusal_recovery", nonref, n)
        add_binary_ci(rows, f"Feature-size sweep K={k}", "base_like_recovery", base_like, n)
        feature_rows.append(
            {
                "K": k,
                "path": str(path),
                "valid": n,
                "n_total": int(agg.get("n_preflight_total", 520)),
                "nonref": nonref,
                "base_like": base_like,
                "nonref_rate": nonref / n if n else None,
                "base_like_rate": base_like / n if n else None,
                "drift": agg.get("avg_final_act_drift_l2_seq") or agg.get("avg_final_drift_relative_l2"),
            }
        )
    dump_json(feature_rows, RESULTS / "feature_size_sweep_selected_summary.json")

    if PATH_ATTR_SUMMARY.exists():
        path_attr = load_json(PATH_ATTR_SUMMARY)
        for name, metric in path_attr.get("variant_metrics", {}).items():
            n = int(metric.get("n", 0))
            add_binary_ci(rows, f"Recovery-path attribution {name}", "nonrefusal_recovery", int(metric.get("n_nonrefusal", 0)), n)
            add_binary_ci(rows, f"Recovery-path attribution {name}", "base_like_recovery", int(metric.get("n_base_like", 0)), n)

    if UNLEARNING_POSTHOC.exists():
        unlearn = load_json(UNLEARNING_POSTHOC)
        for method, metric in unlearn.get("methods", {}).items():
            n = int(metric.get("n_valid_flips", 0))
            add_binary_ci(rows, f"WMDP-Bio posthoc {method}", "choice_recovery", int(metric.get("n_recovered", 0)), n)

    write_uncertainty_rows(rows)
    make_latex_ci_table(rows)
    make_main_refusal_table(adv, harm)
    make_delta_norm_table(adv, harm)
    make_experiment_detail_tables()
    make_responsible_release_paragraph()


def make_latex_ci_table(rows: List[Dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Source & Metric & Count & Estimate & 95\% CI \\",
        r"\midrule",
    ]
    for row in rows:
        if row.get("k") == "" or row.get("k") is None:
            continue
        source = str(row["source"]).replace("_", r"\_")
        metric = str(row["metric"]).replace("_", r"\_")
        k = int(row["k"])
        n = int(row["n"])
        lines.append(
            f"{source} & {metric} & {k}/{n} & {pct(row.get('estimate'))}\\% & "
            f"[{pct(row.get('wilson_low'))}, {pct(row.get('wilson_high'))}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "uncertainty_intervals_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_main_refusal_table(adv: Dict[str, Any], harm: Optional[Dict[str, Any]]) -> None:
    rows = [
        {
            "Dataset": "AdvBench",
            "Strict valid": adv["n"],
            "Non-ref. recovery": f"{adv['n_nonref']}/{adv['n']}",
            "Base-like recovery": f"{adv['n_base_like']}/{adv['n']}",
            "Drift": num(adv["drift"]),
            "Floor violation": num(adv["floor"]),
            "Rel. Delta norm": "--",
        }
    ]
    if harm:
        n_h = int(harm.get("n_strict_valid", 0))
        rows.append(
            {
                "Dataset": "HarmBench-Test",
                "Strict valid": n_h,
                "Non-ref. recovery": f"{int(harm.get('n_recovered_nonrefusal', 0))}/{n_h}",
                "Base-like recovery": f"{int(harm.get('n_recovered_base_like', 0))}/{n_h}",
                "Drift": num(harm.get("mean_defended_feature_drift")),
                "Floor violation": num(harm.get("mean_floor_violation")),
                "Rel. Delta norm": num(harm.get("mean_delta_norm_relative_to_defended_feature_state")),
            }
        )
    dump_json(rows, RESULTS / "main_refusal_strict_valid_table.json")
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Dataset & Strict valid & Non-ref. recovery & Base-like recovery & Drift & Floor viol. & Rel. $\Delta$ norm \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Dataset']} & {row['Strict valid']} & {row['Non-ref. recovery']} & {row['Base-like recovery']} & "
            f"{row['Drift']} & {row['Floor violation']} & {row['Rel. Delta norm']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "main_refusal_strict_valid_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_delta_norm_table(adv: Dict[str, Any], harm: Optional[Dict[str, Any]]) -> None:
    rows = []
    for sample in adv["samples"]:
        rows.append(
            {
                "dataset": "advbench",
                "method": "Jacobian",
                "sample_id": sample.get("sample_idx"),
                "delta_norm": sample.get("stage2_delta_norm"),
                "delta_norm_relative_to_defended_feature_state": inferred_relative_delta(sample),
                "defended_feature_drift": sample.get("final_drift", {}).get("relative_l2"),
                "floor_violation": sample.get("final_drift", {}).get("floor_violation_relative_l2"),
            }
        )
    harm_csv = RESULTS / "harmbench_strict_valid_recovery.csv"
    if harm_csv.exists():
        with harm_csv.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if str(r.get("valid")).lower() != "true":
                    continue
                rows.append(
                    {
                        "dataset": "harmbench_test",
                        "method": "Jacobian",
                        "sample_id": r.get("prompt_id"),
                        "delta_norm": r.get("delta_norm"),
                        "delta_norm_relative_to_defended_feature_state": r.get("delta_norm_relative"),
                        "defended_feature_drift": r.get("defended_feature_drift"),
                        "floor_violation": r.get("floor_violation"),
                    }
                )
    path = RESULTS / "refusal_delta_norms.csv"
    fields = [
        "dataset",
        "method",
        "sample_id",
        "delta_norm",
        "delta_norm_relative_to_defended_feature_state",
        "defended_feature_drift",
        "floor_violation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = {
        "note": "Exact residual-state-normalized delta is unavailable for older AdvBench runs because final deltas and h0 norms were not persisted. The reported relative value uses the defended feature-state norm inferred from drift l2/relative_l2.",
        "n_rows": len(rows),
        "mean_delta_norm": safe_mean(r.get("delta_norm") for r in rows),
        "median_delta_norm": safe_median(r.get("delta_norm") for r in rows),
        "mean_delta_norm_relative_to_defended_feature_state": safe_mean(
            r.get("delta_norm_relative_to_defended_feature_state") for r in rows
        ),
        "median_delta_norm_relative_to_defended_feature_state": safe_median(
            r.get("delta_norm_relative_to_defended_feature_state") for r in rows
        ),
    }
    dump_json(summary, RESULTS / "refusal_delta_norms_summary.json")


def make_evaluator_audit() -> None:
    mod = load_baseline_module()
    audit_rows = []
    for part_idx, folder in enumerate(ADV_JAC_PARTS):
        samples_path = folder / "samples.json"
        if not samples_path.exists():
            continue
        for sample in load_json(samples_path):
            sid = f"part{part_idx}_{sample.get('sample_idx')}"
            entries = [
                ("base", sample.get("base_response", ""), sample.get("base_refusal")),
                ("clamp", sample.get("clamped_response", ""), sample.get("clamped_refusal")),
                ("jacobian_recovery", sample.get("recovered_response", ""), sample.get("recovered_refusal")),
            ]
            for output_type, text, detector_refusal in entries:
                audit_rows.append(
                    {
                        "dataset": "advbench",
                        "sample_id": sid,
                        "output_type": output_type,
                        "original_detector_refusal": bool(detector_refusal),
                        "auto_opening_category": coarse_category(text, mod),
                        "manual_audit_label": "",
                        "manual_audit_notes": "",
                    }
                )
    path = RESULTS / "refusal_evaluator_audit.csv"
    fields = [
        "dataset",
        "sample_id",
        "output_type",
        "original_detector_refusal",
        "auto_opening_category",
        "manual_audit_label",
        "manual_audit_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)
    counts: Dict[str, Dict[str, int]] = {}
    for row in audit_rows:
        key = str(row["output_type"])
        cat = str(row["auto_opening_category"])
        counts.setdefault(key, {})
        counts[key][cat] = counts[key].get(cat, 0) + 1
    dump_json(
        {
            "audit_status": "manual_labels_pending",
            "safety_note": "No full harmful completions are saved in this audit artifact.",
            "counts_by_output_type_and_auto_category": counts,
        },
        RESULTS / "refusal_evaluator_audit_summary.json",
    )


def make_experiment_detail_tables() -> None:
    rows = [
        ["TPP", "Gemma-2B", "Gemma Scope", "task layer", "task SAE features", "task clamp", "answer token", "base-answer recovery", "Adam", "reported in logs", "reported in logs", "task norm budget", "encoder/none", "task scorer", "see task result dir"],
        ["WMDP-Bio unlearning", "Gemma-2B", "Gemma Scope", "unlearning layer", "strict matched defended features", "unlearning clamp", "choice readout", "choice-logit recovery", "Adam", "posthoc full slice", "see logs", "budget sweep available", "OABD/Encoder/None", "choice argmax A/B/C/D", "private run; public sanitized summary in results/sanitized"],
        ["IOI", "Gemma-2B", "Gemma Scope", "task layer", "IOI feature set", "task clamp", "answer token", "name-token recovery", "Adam", "reported in logs", "reported in logs", "task norm budget", "encoder/none", "IOI answer readout", "see IOI result dir"],
        ["Refusal AdvBench", "Gemma-2B", "benchmark_our", "cross-layer", "global refusal features", "3.0", "answer tokens", "base-answer fidelity + anti-refusal boundary", "Adam", "30 + 120", "0.1 / 0.05", "40", "Jacobian projection", "substring + safety cue detector", "private run; public sanitized summary in results/sanitized"],
        ["Refusal HarmBench-Test", "Gemma-2B", "benchmark_our", "cross-layer", "same as AdvBench", "3.0", "answer tokens", "same as AdvBench", "Adam", "30 + 120", "0.1 / 0.05", "40", "Jacobian projection", "same detector", "results/sanitized/harmbench_strict_valid_summary.json"],
        ["Feature-set size sweep", "Gemma-2B", "benchmark_our", "layer 11 recovery", "local-union top-K", "3.0", "answer tokens", "stage1 ft4/prefix2 norm40", "Adam", "full 520 preflight", "0.05", "40", "encoder", "same detector", "results/sanitized/feature_size_sweep_selected_summary.json"],
        ["Budget diagnostic", "Gemma-2B", "Gemma Scope", "WMDP layer", "unlearning defended features", "unlearning clamp", "choice readout", "choice-logit recovery", "Adam", "small diagnostic", "see logs", "swept", "encoder/none", "choice argmax", "Aout/unlearning"],
    ]
    lines = [
        r"\begin{tabular}{lllllllllllllll}",
        r"\toprule",
        r"Experiment & Model & SAE & Layer(s) & Feature set & Clamp & Token scope & Recovery loss & Opt. & Steps & LR & Budget & Projection & Evaluator & Output \\",
        r"\midrule",
    ]
    for row in rows:
        safe = [str(x).replace("_", r"\_") for x in row]
        lines.append(" & ".join(safe) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "appendix_experimental_details.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    compute_rows = [
        ["Refusal AdvBench v4 comparison", "2x RTX 6000 Ada, 48GB each", "about 2--3 h for valid24 comparison", "4 methods / split Jacobian", "about 45GB", "about 6--10 GPU-h", "Jacobian split across two shared-disk hosts"],
        ["Refusal HarmBench-Test strict-valid", "2x RTX 6000 Ada, 48GB each", "about 2.8 h wall-clock split run", "1 Jacobian run split into two shards", "about 34GB per GPU", "about 5.6 GPU-h", "43 strict-valid prompts from 159; safe sanitized outputs only"],
        ["Feature-size sweep", "2x RTX 6000 Ada, 48GB each", "multi-hour full-520 sweep", "representative K values 1--61", "about 45GB", "tens of GPU-h incl. preliminary sweeps", "base preflight cached across K after engineering fix"],
        ["Recovery-path attribution", "2x RTX 6000 Ada, 48GB each", "about 2--3 h", "valid24 split", "about 45GB", "about 4--6 GPU-h", "decomposes full delta into SAE residual/features"],
        ["WMDP-Bio posthoc evaluator", "RTX 6000 Ada, 48GB", "minutes to <1 h", "91 matched flips", "lower than generation runs", "<1 GPU-h posthoc", "shared evaluator for OABD/Encoder/None"],
    ]
    lines = [
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Experiment & Hardware & Runtime per run & Runs & Peak memory & Total GPU-hours & Notes \\",
        r"\midrule",
    ]
    for row in compute_rows:
        safe = [str(x).replace("_", r"\_") for x in row]
        lines.append(" & ".join(safe) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "appendix_compute_resources.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_responsible_release_paragraph() -> None:
    text = (
        "We treat recovery optimization as a diagnostic for SAE intervention robustness, "
        "not as a deployable bypass method. For safety-relevant refusal experiments, "
        "paper-ready artifacts report aggregate detector labels, drift statistics, and "
        "redacted output categories only; full harmful completions are not included in "
        "appendix tables or released examples. If code is released, we will keep the "
        "diagnostic framing explicit, avoid packaging harmful generations, and include "
        "guardrails that prevent accidental logging of unsafe completions."
    )
    (TABLES / "responsible_release_paragraph.tex").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    make_evaluator_audit()
    generate_uncertainty_and_tables()
    print(json.dumps({"results": str(RESULTS), "tables": str(TABLES)}, indent=2))


if __name__ == "__main__":
    main()
