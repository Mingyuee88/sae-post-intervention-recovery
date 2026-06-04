# Responsible Release Notes

This artifact studies recovery from SAE interventions as a diagnostic for intervention completeness. It is not intended as a turnkey bypass toolkit.

## Public artifact policy

The public release should contain only aggregate recovery metrics, prompt or row identifiers, detector labels, drift/floor-violation metrics, and redacted opening categories where needed.

The public release should not contain full harmful prompts or completions, raw `samples.json`, `preflight_rows.json`, answer-record files from refusal experiments, SSH hosts, passwords, API keys, local cache paths, private logs, model weights, or SAE weights.

## Refusal experiments

The refusal recovery scripts can generate full model outputs during local evaluation. For paper-ready or public outputs, use safe wrappers and inspect generated files before release. The `.gitignore` blocks common raw-output filenames by default.
