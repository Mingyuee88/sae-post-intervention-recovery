# Refusal Recovery Experiments

This directory contains the final refusal-recovery code used by the paper.

## Main entrypoints

- `baseline_refusal_sae_recovery_crosslayer_v4.py`: final AdvBench/HarmBench cross-layer Jacobian-projected recovery runner used by the paper and appendix wrappers.
- `baseline_refusal_sae_recovery.py`: shared refusal utilities used by the final cross-layer runner and OABD-style adapter.
- `baseline_oabd_refusal_recovery_crosslayer.py`: OABD-style cross-layer baseline adapter used for comparison.
- `recovery_path_attribution.py`: replay and decomposition analysis for the optimized recovery path.

Older exploratory variants, including later scratch versions not used by the final logged runs, are intentionally omitted from the public artifact.

## Public reproduction wrapper

From the repository root:

```bash
export ADV_BENCH_STRICT_VALID_TARGET_PAIRS_JSON=/path/to/local/advbench_target_pairs.json
bash scripts/reproduce_refusal.sh
```

Generated outputs go to `runs/refusal/` by default and are ignored by git.

## Safety notes

These scripts can generate full model outputs when run locally. Do not commit raw refusal outputs, full harmful prompts, full completions, answer records, preflight rows, or sample files. Public results should be aggregated or redacted as in `results/sanitized/` and `manifests/`.
