# Paper-aligned artifact map

This repository is organized around the final paper version. It contains the recovery code used for the reported experiments, adapters, sanitized metrics, redacted manifests, and release-safety checks. It intentionally excludes raw harmful prompts/completions, model weights, SAE weights, private logs, exploratory script versions, old experiment variants, and figure-generation-only scripts.

## Main method / Section 4

- Shared single-layer recovery core: `src/sae_bench/recovery_core/core.py`
- Main cross-layer Jacobian refusal recovery: `experiments/refusal/baseline_refusal_sae_recovery_crosslayer_v4.py`
- Refusal single-layer/base utilities required by the cross-layer and OABD adapters: `experiments/refusal/baseline_refusal_sae_recovery.py`

## TPP

- Public adapter: `experiments/tpp/run_tpp_recovery_external.sh`
- Paper asset helper: `experiments/tpp/make_paper_assets.py`
- Sanitized metrics: `results/sanitized/neurips_main_table.csv`, `overall_summary_*.csv`, `dataset_summary_unweighted.csv`, `paired_target_deltas.csv`

## WMDP-Bio unlearning

- Recovery runner: `experiments/unlearning/recovery_unlearning_choice_only_seqwide_act.py`
- Permutation runner: `experiments/unlearning/permutation_seqwide.py`
- Full-run preparation helper used by the final post-hoc run: `experiments/unlearning/prepare_unlearning_posthoc_interventions.py`
- Shared post-hoc evaluator: `experiments/unlearning/posthoc_eval_unlearning_recovery.py`
- OABD-style adapter: `experiments/unlearning/baseline_oabd_defended_suffix_fixedprefix_choiceonly_dualsummary.py`
- Budget diagnostic runner: `experiments/unlearning/run_unlearning_drift_trajectory_budget.py`
- Sanitized aggregate: `results/sanitized/unlearning_posthoc_aggregate.json`
- Strict-slice manifest: `manifests/wmdp_bio_strict_valid_91_flips.json`

## IOI

- Code: `experiments/ioi/ioi_recovery.py`
- Sanitized metrics: `results/sanitized/ioi_aggregate.json`, `ioi_mode_summary.csv`, `ioi_ioi_per_prompt_mechanism_stats.json`

## Refusal case study

- Main code: `experiments/refusal/baseline_refusal_sae_recovery_crosslayer_v4.py`
- Shared/base utilities: `experiments/refusal/baseline_refusal_sae_recovery.py`
- OABD-style adapter: `experiments/refusal/baseline_oabd_refusal_recovery_crosslayer.py`
- Path attribution: `experiments/refusal/recovery_path_attribution.py`
- Sanitized metrics: `results/sanitized/main_refusal_strict_valid_table.json`, `results/sanitized/refusal_compare/`, `results/sanitized/refusal_recovery_path_attribution_summary.json`
- Redacted manifests: `manifests/refusal_advbench_strict_valid_24.json`, `manifests/refusal_harmbench_strict_valid_43.json`, `manifests/refusal_feature_size_sweep_k_values.json`

## Appendix checks

- HarmBench strict-valid: `results/sanitized/harmbench_strict_valid_*`
- Feature-size sweep: `results/sanitized/feature_size_sweep_selected_summary.json`
- Budget diagnostic: `results/sanitized/advbench_budget_sweep*`
- Uncertainty intervals: `results/sanitized/uncertainty_intervals.csv`
- Public-release safety test: `tests/test_public_artifact_safety.py`
