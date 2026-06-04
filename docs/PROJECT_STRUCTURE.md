# Project Structure

This repository is organized by paper claim rather than by the original research-run directories.

## Core method

`src/sae_bench/recovery_core/` contains the reusable implementation of defended-state construction, SAE feature clamping, residual perturbation optimization, and encoder-null projection. Experiment scripts should import from this package rather than duplicating recovery logic.

## Experiments

- `experiments/tpp/`: latent-level recovery under official SAEBench TPP feature ablation.
- `experiments/unlearning/`: output-level WMDP-Bio answer-choice recovery after SAE-based unlearning clamps.
- `experiments/ioi/`: circuit-level IOI recovery under a fixed SAE clamp.
- `experiments/refusal/`: safety refusal recovery, cross-layer projection, baselines, and recovery-path attribution.

## Results and manifests

`results/sanitized/` stores aggregate metrics used by the paper. It should not contain full prompt text, full model completions, raw answer records, or cached activations.

`manifests/` stores ID-only or redacted manifests for strict-valid examples. These files are intended to document evaluation slices without publishing harmful content.

## Release tooling

- `Makefile`: top-level test and release-check commands.
- `scripts/safety_check_release.py`: artifact scanner for raw outputs, private paths, credentials, and unexpected placeholders.
- `scripts/inspect_results.py`: compact result-summary printer for reviewers and artifact users.
- `docs/RELEASE_CHECKLIST.md`: manual fields and publication checklist.
