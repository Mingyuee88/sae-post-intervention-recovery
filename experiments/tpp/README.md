# TPP Recovery

This directory contains the public TPP adapter and paper-asset utilities. The official TPP benchmark infrastructure comes from a separate SAEBench checkout.

## Public reproduction wrapper

From the repository root:

```bash
export SAEBENCH_EXTERNAL_ROOT=/path/to/SAEBench
bash scripts/reproduce_tpp.sh
```

The wrapper sets `PYTHONPATH` so the external SAEBench checkout can import the local recovery utilities from `src/sae_bench/recovery_core/`.

## Released artifacts

The public artifact includes aggregate TPP tables under `results/sanitized/`, including `neurips_main_table.csv`, `dataset_summary_unweighted.csv`, `overall_summary_unweighted.csv`, and `overall_summary_weighted.csv`.
