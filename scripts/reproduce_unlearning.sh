#!/usr/bin/env bash
set -euo pipefail

# Reproduce WMDP-Bio strict-slice recovery.
# Required environment variables:
#   WMDP_BIO_PROMPT_POOL_MANIFEST: local manifest for the WMDP-Bio prompt pool
# Optional:
#   OUTPUT_DIR: output directory for generated rows

: "${WMDP_BIO_PROMPT_POOL_MANIFEST:?Set WMDP_BIO_PROMPT_POOL_MANIFEST to a local prompt-pool manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/unlearning}"
mkdir -p "$OUTPUT_DIR/encoder" "$OUTPUT_DIR/none"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python experiments/unlearning/permutation_seqwide.py   --manifest "$WMDP_BIO_PROMPT_POOL_MANIFEST"   --base_recovery_script experiments/unlearning/recovery_unlearning_choice_only_seqwide_act.py   --projection_mode encoder   --strict_base_24_24   --num_steps 150   --lr 0.08   --max_delta_norm 20   --loss_mode choice_margin   --output_dir "$OUTPUT_DIR/encoder"

python experiments/unlearning/permutation_seqwide.py   --manifest "$WMDP_BIO_PROMPT_POOL_MANIFEST"   --base_recovery_script experiments/unlearning/recovery_unlearning_choice_only_seqwide_act.py   --projection_mode none   --strict_base_24_24   --num_steps 150   --lr 0.08   --max_delta_norm 20   --loss_mode choice_margin   --output_dir "$OUTPUT_DIR/none"
