#!/usr/bin/env bash
set -euo pipefail

# Reproduce the TPP recovery experiment using a separate SAEBench checkout.
# Required environment variables:
#   SAEBENCH_EXTERNAL_ROOT: path to a local clone of https://github.com/adamkarvonen/SAEBench
# Optional:
#   OUTPUT_DIR: output directory for generated runs

: "${SAEBENCH_EXTERNAL_ROOT:?Set SAEBENCH_EXTERNAL_ROOT to your local SAEBench checkout}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/tpp}"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PWD/src:$SAEBENCH_EXTERNAL_ROOT:${PYTHONPATH:-}"

bash experiments/tpp/run_tpp_recovery_external.sh
