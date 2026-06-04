#!/usr/bin/env bash
set -euo pipefail

: "${SAEBENCH_EXTERNAL_ROOT:?Set SAEBENCH_EXTERNAL_ROOT to a separately cloned SAEBench repo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${SAEBENCH_EXTERNAL_ROOT}:${PYTHONPATH:-}"

# Configuration template only. The external SAEBench checkout provides the
# official TPP pipeline; this artifact provides recovery utilities and sanitized
# paper outputs. If you have permission to use/modify the upstream TPP runner,
# connect its defended activations to src/sae_bench/recovery_core/core.py.
python -u "${SAEBENCH_EXTERNAL_ROOT}/sae_bench/evals/scr_and_tpp/main.py" --help
echo "See experiments/tpp/README.md for the paper's TPP recovery configuration."
