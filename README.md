# SAE Interventions are Unreliable: Post-Intervention Recovery

This repository contains the public code and sanitized artifacts for the paper **"SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior"**.

The central question is: after an SAE feature clamp has already suppressed a behavior, can the behavior still be recovered from the defended residual state while the clamped SAE features remain close to their defended values?

## What Is Included

- Shared recovery utilities for SAE-clamped residual states.
- Encoder-orthogonal projected recovery for single-layer interventions.
- Cross-layer Jacobian-projected recovery for refusal-feature clamps.
- Final experiment code for TPP, WMDP-Bio unlearning, IOI, and refusal recovery.
- Sanitized aggregate results and redacted manifests used by the paper.

## Repository Map

```text
src/sae_bench/recovery_core/      Shared recovery and projection utilities
experiments/tpp/                  Final TPP adapter and paper-asset helper
experiments/unlearning/           Final WMDP-Bio recovery, post-hoc, and budget diagnostic code
experiments/ioi/                  Final IOI recovery script
experiments/refusal/              Final refusal recovery, cross-layer projection, OABD adapter, attribution
configs/                          Environment-variable templates for local paths and outputs
scripts/                          Reproduction wrappers and release checks
results/sanitized/                Aggregate metrics only
manifests/                        ID-only or redacted strict-valid manifests
docs/                             Release notes, third-party handling, artifact policy
```

## Installation

Create the environment and install the local package from the repository root:

```bash
conda env create -f environment.yml
conda activate sae-intervention-recovery
pip install -e ".[dev]"
```

For full experiments, install the experiment extras:

```bash
pip install -e ".[experiments,dev]"
```

The quotes around `.[experiments,dev]` are intentional: they work in bash and avoid zsh treating `[]` as a filename pattern. Nothing inside the command needs to be replaced when you run it from the repository root.

Model weights, SAE weights, and benchmark datasets are not redistributed here. Download them from their official sources and comply with their terms.

## Quick Checks

Run lightweight checks from the repository root:

```bash
make test
make inspect-results
make release-check
```

`make release-check` runs unit tests, scans the public artifact for unsafe files or private strings, and prints a compact summary of sanitized result files.

## Reproducing Paper Results

The scripts in `scripts/` are the intended public entrypoints. Copy the matching config template, edit local paths, source it, then run the reproduction script:

```bash
cp configs/refusal.env.example configs/refusal.env
# Edit configs/refusal.env for your local files.
source configs/refusal.env
bash scripts/reproduce_refusal.sh
```

Available templates:

```text
configs/tpp.env.example
configs/unlearning.env.example
configs/ioi.env.example
configs/refusal.env.example
```

For all final command templates, see `configs/final_commands.md`.

For a fast artifact inspection without rerunning GPU experiments:

```bash
python scripts/inspect_results.py
```

## Paper Claim to Code Map

| Paper setting | Code | Sanitized artifacts |
| --- | --- | --- |
| TPP latent recovery | `experiments/tpp/` | `results/sanitized/neurips_main_table.csv`, `overall_summary_*.csv` |
| WMDP-Bio unlearning recovery | `experiments/unlearning/` | `results/sanitized/unlearning_posthoc_aggregate.json` |
| IOI circuit recovery | `experiments/ioi/` | `results/sanitized/ioi_aggregate.json`, `ioi_mode_summary.csv` |
| Refusal recovery | `experiments/refusal/baseline_refusal_sae_recovery_crosslayer_v4.py` | `results/sanitized/main_refusal_strict_valid_table.json`, `refusal_compare/` |
| Recovery-path attribution | `experiments/refusal/recovery_path_attribution.py` | `results/sanitized/refusal_recovery_path_attribution_summary.json` |

## Responsible Release

This repository is a diagnostic artifact for evaluating whether SAE interventions form complete behavioral bottlenecks. It is not intended as a turnkey jailbreak or deployment attack package.

For safety-relevant refusal experiments, the public release includes aggregate statistics, detector labels, prompt IDs, and coarse redacted categories. It does not include full harmful prompts, full harmful completions, raw `samples.json`, answer-record files, private logs, cached model activations, model weights, or SAE weights.

## Acknowledgements

This codebase builds on and adapts infrastructure from SAEBench and Obfuscated Activations / OABD. See `ACKNOWLEDGEMENTS.md` and `docs/THIRD_PARTY.md` for details.

## Citation
```bibtex
@misc{cui2026saeinterventionsunreliable,
      title={SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior},
      author={Mingyue Cui and Linghui Shen and Xingyi Yang},
      year={2026},
      eprint={xxxx.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/xxxx.xxxxx}
}
```
