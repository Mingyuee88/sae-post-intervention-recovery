# IOI Recovery

This directory contains the circuit-level IOI recovery experiment.

## Main entrypoint

- `ioi_recovery.py`: utilities for loading the official IOI dataset class, selecting SAE features, clamping them, optimizing recovery perturbations, and reporting recovery/reactivation metrics.

## Public reproduction wrapper

From the repository root:

```bash
export IOI_DATASET_SOURCE=/path/to/official/ioi_dataset.py
bash scripts/reproduce_ioi.sh
```

Generated outputs go to `runs/ioi/` by default and are ignored by git.

## Released artifacts

The paper aggregate is released as `results/sanitized/ioi_aggregate.json`, with mode summaries in `results/sanitized/ioi_mode_summary.csv`.
