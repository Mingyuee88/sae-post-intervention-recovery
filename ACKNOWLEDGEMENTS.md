# Acknowledgements and Upstream Code

This repository contains research code for post-intervention recovery of SAE-based interventions. Some experiment infrastructure was adapted from existing open-source research repositories, while the recovery diagnostics, projection utilities, safety-filtered artifact structure, and paper-specific experiment wrappers were developed for this project.

## Upstream repositories

- **SAEBench**: https://github.com/adamkarvonen/SAEBench
  - Used as the basis for SAE benchmark evaluation infrastructure, especially the official TPP-related workflow.
  - Our release includes paper-specific recovery adapters and sanitized outputs rather than a full vendored upstream checkout.

- **Obfuscated Activations / OABD**: https://github.com/LukeBailey181/obfuscated-activations
  - Used as the basis or reference point for OABD-style soft-suffix baseline comparisons.
  - Scripts in this repository labelled `OABD-style` are experiment adapters for our comparison setting and should not be cited as a clean copy of the upstream implementation.

- **SAE Lens / TransformerLens / Gemma Scope**
  - Used for model and SAE loading, activation hooks, and SAE feature computations.
  - Model weights and SAE weights are not redistributed in this repository.

## Original contributions in this repository

- Post-intervention residual-space recovery diagnostic.
- Encoder-orthogonal projected recovery for single-layer SAE interventions.
- Cross-layer Jacobian-projected recovery for refusal-feature clamps.
- Recovery-path attribution and replay analyses.
- Paper-specific wrappers for TPP, WMDP-Bio unlearning, IOI, and refusal recovery.
- Sanitized result artifacts and responsible-release checks.

## Citation notes

When using this repository, please cite the accompanying paper and acknowledge the upstream repositories above where relevant. Replace the placeholder citation in `CITATION.cff` once the paper has an arXiv ID, conference URL, or DOI.
