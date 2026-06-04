# Third-party code and acknowledgement handling

This repository contains code adapted from or inspired by upstream research repositories. The public release should make that provenance clear in `README.md` and `ACKNOWLEDGEMENTS.md`.

## SAEBench

- Upstream: https://github.com/adamkarvonen/SAEBench
- Used for the official TPP benchmark and related SAE benchmark infrastructure.
- Handling in this artifact: users can clone SAEBench separately when needed. This repository includes paper-specific recovery adapters and sanitized outputs.

## Obfuscated Activations / OABD

- Upstream: https://github.com/LukeBailey181/obfuscated-activations
- Used conceptually and experimentally as an OABD-style soft-suffix baseline.
- Handling in this artifact: scripts labelled `OABD-style` are task adapters for comparison and should not be represented as the original upstream implementation.

## Gemma Scope / SAEs / model weights

We do not redistribute model weights or SAE weights. Download them through their official distribution channels and comply with the relevant model/data terms.

## Acknowledgement text

A concise version suitable for README or paper artifact notes:

> This codebase builds on and adapts infrastructure from SAEBench and Obfuscated Activations / OABD. We thank the authors of those projects for releasing their code and benchmark infrastructure.
