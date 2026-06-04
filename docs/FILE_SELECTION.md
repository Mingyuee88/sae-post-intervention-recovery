# File Selection for Open Source Release

## Included

- Core recovery utilities from `sae_bench/recovery_core`.
- Refusal, unlearning, IOI, and TPP experiment source files.
- Safe appendix scripts and aggregate/sanitized result tables.
- TPP paper-asset scripts and aggregate CSV/TEX outputs.

## Excluded

- `Aout/refusal/Batch_Recovery` raw outputs because they contain full prompts and model completions.
- Unlearning prompt-pool text and raw per-example outputs.
- TPP `recovery_per_example.csv` files because they are large raw per-example artifacts; aggregate CSVs are retained.
- `__pycache__`, logs, PIDs, checkpoints, pickle/numpy caches, and host-specific shell scripts.

## Files that still need review before public GitHub release

- Refusal baseline scripts can still write full `samples.json` if run directly. Public workflows should use safe wrappers or add a stricter `--public_safe_outputs` flag before release.
- Dataset manifests and target-pair files should be regenerated as redacted ID-only manifests.
- Add a real license after confirming compatibility with upstream SAEBench and SAE Lens licenses.
