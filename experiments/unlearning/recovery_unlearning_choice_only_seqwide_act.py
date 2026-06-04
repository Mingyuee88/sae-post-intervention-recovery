import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_bench.evals.unlearning.utils.feature_activation import get_top_features
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench.sae_bench_utils.general_utils import load_and_format_sae, setup_environment
from sae_bench.recovery_core import (
    FixedDirectDefendedPlusDeltaHook,
    build_direct_defended_reference,
    get_choice_token_ids,
    objective_value,
    optimize_delta,
)


@dataclass
class DefendedConfig:
    retain_threshold: float
    n_features: int
    multiplier: float
    layer: int
    sae_release: str
    sae_id: str
    sae_name: str
    wmdp_bio: float
    all_side_effects_mcq: float


def get_params_from_filename(filename: str):
    pattern = r"multiplier(\d+)_nfeatures(\d+)_layer(\d+)_retainthres(\d+(?:\.\d+)?).pkl"
    m = re.search(pattern, filename)
    if not m:
        return None
    multiplier, n_features, layer, retain_thres = m.groups()
    return float(multiplier), int(n_features), int(layer), float(retain_thres)


def read_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics_df(metrics_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pkl_path in sorted(metrics_dir.glob("*.pkl")):
        metrics = read_pickle(pkl_path)
        parsed = get_params_from_filename(pkl_path.name)
        if parsed is None:
            continue
        multiplier, n_features, layer, retain_thres = parsed
        row: Dict[str, Any] = {
            "file": str(pkl_path),
            "multiplier": multiplier,
            "n_features": n_features,
            "layer": layer,
            "retain_thres": retain_thres,
        }
        n_se_questions = 0
        n_se_correct_questions = 0
        for dataset_name, dataset_metrics in metrics.items():
            if dataset_name == "ablate_params":
                continue
            row[dataset_name] = dataset_metrics["mean_correct"]
            if dataset_name not in ["wmdp-bio", "college_biology"]:
                n_se_correct_questions += dataset_metrics["total_correct"]
                n_se_questions += len(dataset_metrics["is_correct"])
        row["all_side_effects_mcq"] = (
            n_se_correct_questions / n_se_questions if n_se_questions > 0 else 0.0
        )
        rows.append(row)
    rows.sort(
        key=lambda x: (
            x.get("all_side_effects_mcq", 0.0) < 0.99,
            x.get("wmdp-bio", 1.0),
            x.get("multiplier", 0.0),
        )
    )
    return rows


def choose_defended_config(
    rows: List[Dict[str, Any]],
    min_side_effects: float = 0.99,
    target_mode: str = "balanced",
) -> DefendedConfig:
    candidates = [r for r in rows if r.get("all_side_effects_mcq", 0.0) >= min_side_effects]
    if not candidates:
        raise ValueError(f"No configs satisfy all_side_effects_mcq >= {min_side_effects}.")

    if target_mode == "aggressive":
        chosen = min(
            candidates,
            key=lambda r: (
                r.get("wmdp-bio", 1.0),
                -r.get("all_side_effects_mcq", 0.0),
                r.get("multiplier", 0.0),
            ),
        )
    elif target_mode == "conservative":
        chosen = min(
            candidates,
            key=lambda r: (
                abs(r.get("wmdp-bio", 1.0) - 0.70),
                -r.get("all_side_effects_mcq", 0.0),
                r.get("multiplier", 0.0),
            ),
        )
    else:
        def key_fn(r: Dict[str, Any]):
            w = r.get("wmdp-bio", 1.0)
            se = r.get("all_side_effects_mcq", 0.0)
            mult = r.get("multiplier", 0.0)
            nfeat = r.get("n_features", 0)
            return (abs(w - 0.60), -se, abs(mult - 50.0), abs(nfeat - 10))
        chosen = min(candidates, key=key_fn)

    return DefendedConfig(
        retain_threshold=float(chosen["retain_thres"]),
        n_features=int(chosen["n_features"]),
        multiplier=float(chosen["multiplier"]),
        layer=int(chosen["layer"]),
        sae_release="",
        sae_id="",
        sae_name="",
        wmdp_bio=float(chosen["wmdp-bio"]),
        all_side_effects_mcq=float(chosen["all_side_effects_mcq"]),
    )


def load_sparsities(sparsity_dir: Path):
    txts = list(sparsity_dir.glob("*.txt"))
    if len(txts) < 2:
        raise FileNotFoundError(f"Expected two txt sparsity files under {sparsity_dir}, found {txts}")
    lower_map = {p.name.lower(): p for p in txts}
    forget_path = None
    retain_path = None
    for name, p in lower_map.items():
        if "forget" in name:
            forget_path = p
        if "retain" in name:
            retain_path = p
    if forget_path is None or retain_path is None:
        raise FileNotFoundError(f"Could not identify forget/retain sparsity files in {sparsity_dir}")
    forget = np.loadtxt(forget_path, dtype=float)
    retain = np.loadtxt(retain_path, dtype=float)
    return forget, retain


def recover_feature_ids(sparsity_dir: Path, retain_threshold: float, n_features: int) -> List[int]:
    forget, retain = load_sparsities(sparsity_dir)
    top_features = get_top_features(forget, retain, retain_threshold=retain_threshold)
    top_features = [int(x) for x in list(top_features[:n_features])]
    if len(top_features) == 0:
        raise ValueError("Recovered empty feature set from sparsity files.")
    return top_features


def dtype_from_string(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_sae_from_release(model_name: str, sae_release: str, sae_id: str, device: str, llm_dtype: torch.dtype):
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=llm_dtype)
    _, sae, _ = load_and_format_sae(sae_release, sae_id, device)
    sae = sae.to(device=device, dtype=llm_dtype)
    hook_name = sae.cfg.hook_name
    return model, sae, hook_name



@torch.no_grad()
def greedy_choice_letter(model: HookedTransformer, logits_last: torch.Tensor, letters: List[str]) -> str:
    scores = []
    for letter in letters:
        tok = model.to_single_token(" " + letter)
        scores.append((letter, float(logits_last[tok].item())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


@torch.no_grad()
def argmax_token_text(model: HookedTransformer, logits_last: torch.Tensor) -> str:
    tid = int(torch.argmax(logits_last).item())
    return model.tokenizer.decode([tid])


@torch.no_grad()
def generate_one_token(model: HookedTransformer, prompt: str, hook_name: Optional[str] = None, hook_fn=None) -> str:
    if hook_name is None or hook_fn is None:
        out = model.generate(prompt, max_new_tokens=1, temperature=0.0, verbose=False)
        return out
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        out = model.generate(prompt, max_new_tokens=1, temperature=0.0, verbose=False)
    return out


def extract_new_text(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):]
    return full_text



def dump_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Recover behavior under fixed post-clamp defended residual.")
    parser.add_argument("--model_name", type=str, default="gemma-2-2b-it")
    parser.add_argument("--sae_regex_pattern", type=str, required=True)
    parser.add_argument("--sae_block_pattern", type=str, required=True)
    parser.add_argument("--artifacts_path", type=str, default="artifacts")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--target_token", type=str, default=" A")
    parser.add_argument("--letters", type=str, default="A,B,C,D")
    parser.add_argument("--output_dir", type=str, default="recovery_outputs")
    parser.add_argument("--llm_dtype", type=str, default="bfloat16")
    parser.add_argument("--min_side_effects", type=float, default=0.99)
    parser.add_argument("--pick_mode", type=str, default="aggressive", choices=["balanced", "aggressive", "conservative"])
    parser.add_argument("--retain_threshold", type=float, default=None)
    parser.add_argument("--n_features", type=int, default=50)
    parser.add_argument("--multiplier", type=float, default=10.0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max_delta_norm", type=float, default=20.0)
    parser.add_argument("--projection_mode", type=str, default="encoder", choices=["none", "encoder"])
    parser.add_argument("--lambda_act", type=float, default=0.0)
    parser.add_argument("--lambda_decode", type=float, default=0.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require_defended_flip", action="store_true")
    parser.add_argument("--loss_mode", type=str, default="choice_margin", choices=["choice_margin", "choice_ce", "vocab_margin"])
    args = parser.parse_args()

    device = setup_environment()
    llm_dtype = dtype_from_string(args.llm_dtype)

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    if len(selected_saes) == 0:
        raise ValueError("No SAEs matched your regex/block pattern.")
    if len(selected_saes) > 1:
        raise ValueError(f"Expected exactly one SAE for recovery, got {selected_saes}")

    sae_release, sae_id = selected_saes[0]
    sae_name = f"{sae_release}_{sae_id}"

    base = Path(args.artifacts_path) / "unlearning" / args.model_name
    nested_results = base / sae_release / sae_id / "results"
    flat_results = base / f"{sae_release}_{sae_id.replace('/', '_')}" / "results"
    if nested_results.exists():
        results_dir = nested_results
    elif flat_results.exists():
        results_dir = flat_results
    else:
        parts = [p for p in sae_id.split("/") if p]
        matches = []
        for pkl in base.glob("**/results/metrics/*.pkl"):
            sp = str(pkl)
            if sae_release in sp and all(part in sp for part in parts):
                matches.append(pkl.parent.parent)
        matches = sorted(set(matches))
        if not matches:
            raise FileNotFoundError(f"Could not locate results dir under {base} for sae_release={sae_release}, sae_id={sae_id}")
        results_dir = matches[0]

    metrics_dir = results_dir / "metrics"
    sparsity_dir = results_dir / "sparsities"
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics dir not found: {metrics_dir}")
    if not sparsity_dir.exists():
        raise FileNotFoundError(f"Sparsity dir not found: {sparsity_dir}")

    rows = load_metrics_df(metrics_dir)
    if len(rows) == 0:
        raise ValueError(f"No metric rows found in {metrics_dir}")

    if args.retain_threshold is not None and args.n_features is not None and args.multiplier is not None:
        matched = [
            r for r in rows
            if abs(float(r["retain_thres"]) - float(args.retain_threshold)) < 1e-12
            and int(r["n_features"]) == int(args.n_features)
            and abs(float(r["multiplier"]) - float(args.multiplier)) < 1e-12
        ]
        if not matched:
            raise ValueError("Requested defended config not found in saved metrics.")
        chosen_row = matched[0]
        defended_cfg = DefendedConfig(
            retain_threshold=float(chosen_row["retain_thres"]),
            n_features=int(chosen_row["n_features"]),
            multiplier=float(chosen_row["multiplier"]),
            layer=int(chosen_row["layer"]),
            sae_release=sae_release,
            sae_id=sae_id,
            sae_name=sae_name,
            wmdp_bio=float(chosen_row["wmdp-bio"]),
            all_side_effects_mcq=float(chosen_row["all_side_effects_mcq"]),
        )
    else:
        defended_cfg = choose_defended_config(rows, min_side_effects=args.min_side_effects, target_mode=args.pick_mode)
        defended_cfg.sae_release = sae_release
        defended_cfg.sae_id = sae_id
        defended_cfg.sae_name = sae_name

    feature_ids = recover_feature_ids(sparsity_dir=sparsity_dir, retain_threshold=defended_cfg.retain_threshold, n_features=defended_cfg.n_features)

    model, sae, hook_name = load_model_and_sae_from_release(
        model_name=args.model_name,
        sae_release=sae_release,
        sae_id=sae_id,
        device=device,
        llm_dtype=llm_dtype,
    )

    prompt = args.prompt
    tokens = model.to_tokens(prompt)
    last_idx = tokens.shape[1] - 1
    target_token_id = model.to_single_token(args.target_token)
    letters = [x.strip() for x in args.letters.split(",") if x.strip()]
    choice_token_ids = get_choice_token_ids(model, letters)
    feature_idx = torch.tensor(feature_ids, device=device, dtype=torch.long)

    with torch.no_grad():
        base_logits = model(tokens)
        direct_def_logits, defended_ref = build_direct_defended_reference(
            model=model,
            sae=sae,
            hook_name=hook_name,
            tokens=tokens,
            feature_idx=feature_idx,
            multiplier=defended_cfg.multiplier,
        )
        fixed_def_hook = FixedDirectDefendedPlusDeltaHook(sae=sae, feature_idx=feature_idx, defended_ref=defended_ref, delta_last=None, state={})
        fixed_def_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, fixed_def_hook)])

    base_logits_last = base_logits[0, last_idx, :].detach().float().cpu()
    direct_def_logits_last = direct_def_logits[0, last_idx, :].detach().float().cpu()
    fixed_def_logits_last = fixed_def_logits[0, last_idx, :].detach().float().cpu()

    base_choice = greedy_choice_letter(model, base_logits_last, letters)
    direct_def_choice = greedy_choice_letter(model, direct_def_logits_last, letters)
    fixed_def_choice = greedy_choice_letter(model, fixed_def_logits_last, letters)

    base_score = float(objective_value(base_logits[0, last_idx, :].float(), target_token_id, choice_token_ids, args.loss_mode).item())
    direct_def_score = float(objective_value(direct_def_logits[0, last_idx, :].float(), target_token_id, choice_token_ids, args.loss_mode).item())
    fixed_def_score = float(objective_value(fixed_def_logits[0, last_idx, :].float(), target_token_id, choice_token_ids, args.loss_mode).item())

    if args.require_defended_flip and direct_def_choice == base_choice:
        raise RuntimeError(
            f"Direct defended state not reproduced: base_choice={base_choice}, direct_defended_choice={direct_def_choice}."
        )

    result = optimize_delta(
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
    recovered_choice = greedy_choice_letter(model, final_logits_last, letters)

    gen_fixed_def_hook = FixedDirectDefendedPlusDeltaHook(sae=sae, feature_idx=feature_idx, defended_ref=defended_ref, delta_last=None, state={})
    gen_recovered_hook = FixedDirectDefendedPlusDeltaHook(sae=sae, feature_idx=feature_idx, defended_ref=defended_ref, delta_last=result["delta"].to(device), state={})

    base_gen = generate_one_token(model, prompt)
    fixed_def_gen = generate_one_token(model, prompt, hook_name, gen_fixed_def_hook)
    recovered_gen = generate_one_token(model, prompt, hook_name, gen_recovered_hook)

    summary = {
        "sae_release": sae_release,
        "sae_id": sae_id,
        "sae_name": sae_name,
        "results_dir": str(results_dir),
        "hook_name": hook_name,
        "defended_config": {
            "retain_threshold": defended_cfg.retain_threshold,
            "n_features": defended_cfg.n_features,
            "multiplier": defended_cfg.multiplier,
            "layer": defended_cfg.layer,
            "wmdp_bio": defended_cfg.wmdp_bio,
            "all_side_effects_mcq": defended_cfg.all_side_effects_mcq,
        },
        "feature_ids": feature_ids,
        "prompt": prompt,
        "target_token": args.target_token,
        "letters": letters,
        "loss_mode": args.loss_mode,
        "base_score": base_score,
        "direct_defended_score": direct_def_score,
        "fixed_defended_score": fixed_def_score,
        "recovered_score": result["final_score"],
        "base_choice": base_choice,
        "direct_defended_choice": direct_def_choice,
        "fixed_defended_choice": fixed_def_choice,
        "recovered_choice": recovered_choice,
        "base_top_token": argmax_token_text(model, base_logits_last),
        "direct_defended_top_token": argmax_token_text(model, direct_def_logits_last),
        "fixed_defended_top_token": argmax_token_text(model, fixed_def_logits_last),
        "recovered_top_token": argmax_token_text(model, final_logits_last),
        "base_generated": extract_new_text(base_gen, prompt),
        "fixed_defended_generated": extract_new_text(fixed_def_gen, prompt),
        "recovered_generated": extract_new_text(recovered_gen, prompt),
        "preflight": {
            "base_choice": base_choice,
            "direct_defended_choice": direct_def_choice,
            "fixed_defended_choice": fixed_def_choice,
            "base_score": base_score,
            "direct_defended_score": direct_def_score,
            "fixed_defended_score": fixed_def_score,
        },
        "projection_mode": args.projection_mode,
        "lambda_act": args.lambda_act,
        "lambda_decode": args.lambda_decode,
        "num_steps": args.num_steps,
        "lr": args.lr,
        "max_delta_norm": args.max_delta_norm,
        "final_act_drift_l2": result["final_act_drift_l2"],
        "final_act_drift_linf": result["final_act_drift_linf"],
        "final_decode_drift_l2": result["final_decode_drift_l2"],
        "final_decode_drift_linf": result["final_decode_drift_linf"],
        "final_act_drift_l2_seq": result["final_act_drift_l2_seq"],
        "final_act_drift_linf_seq": result["final_act_drift_linf_seq"],
        "final_decode_drift_l2_seq": result["final_decode_drift_l2_seq"],
        "final_decode_drift_linf_seq": result["final_decode_drift_linf_seq"],
        "final_decode_drift_l2_last": result["final_decode_drift_l2_last"],
        "final_decode_drift_linf_last": result["final_decode_drift_linf_last"],
        "final_delta_norm": result["final_delta_norm"],
        "history": result["history"],
    }

    out_dir = Path(args.output_dir) / f"{sae_name.replace('/', '__')}__retain{defended_cfg.retain_threshold}__n{defended_cfg.n_features}__m{int(defended_cfg.multiplier)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out_dir / "summary.json")
    torch.save(result["delta"], out_dir / "delta.pt")
    torch.save(defended_ref, out_dir / "defended_ref.pt")

    print(json.dumps({
        "summary_json": str(out_dir / "summary.json"),
        "delta_pt": str(out_dir / "delta.pt"),
        "chosen_config": summary["defended_config"],
        "base_score": summary["base_score"],
        "direct_defended_score": summary["direct_defended_score"],
        "fixed_defended_score": summary["fixed_defended_score"],
        "recovered_score": summary["recovered_score"],
        "base_choice": summary["base_choice"],
        "direct_defended_choice": summary["direct_defended_choice"],
        "fixed_defended_choice": summary["fixed_defended_choice"],
        "recovered_choice": summary["recovered_choice"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
