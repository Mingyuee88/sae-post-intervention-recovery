# WMDP-Bio Unlearning Recovery

This directory contains the final WMDP-Bio recovery code used by the paper.

## Main entrypoints

- `recovery_unlearning_choice_only_seqwide_act.py`: per-example choice-level recovery under an active SAE clamp.
- `permutation_seqwide.py`: strict 24-choice permutation runner.
- `prepare_unlearning_posthoc_interventions.py`: preparation helper used by the final full post-hoc run.
- `posthoc_eval_unlearning_recovery.py`: shared post-hoc evaluator for encoder, unconstrained, and OABD-style rows.
- `baseline_oabd_defended_suffix_fixedprefix_choiceonly_dualsummary.py`: OABD-style soft-suffix baseline adapter.
- `run_unlearning_drift_trajectory_budget.py`: budget and trajectory diagnostic runner used for the appendix diagnostic.

Figure-generation-only scripts and exploratory variants are intentionally omitted from the public artifact.

## Public reproduction wrapper

From the repository root:

```bash
export WMDP_BIO_PROMPT_POOL_MANIFEST=/path/to/local/wmdp_bio_prompt_pool_manifest.json
bash scripts/reproduce_unlearning.sh
```

Generated outputs go to `runs/unlearning/` by default and are ignored by git.

## Released artifacts

The strict-slice aggregate used in the paper is released as `results/sanitized/unlearning_posthoc_aggregate.json`. The public manifest is `manifests/wmdp_bio_strict_valid_91_flips.json`.
