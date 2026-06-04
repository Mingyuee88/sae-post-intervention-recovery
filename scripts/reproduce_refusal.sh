#!/usr/bin/env bash
set -euo pipefail

# Reproduce the main AdvBench refusal recovery run.
# Required environment variables:
#   ADV_BENCH_STRICT_VALID_TARGET_PAIRS_JSON: local target-pairs JSON for the strict-valid AdvBench slice
# Optional:
#   OUTPUT_DIR: output directory for generated refusal outputs

: "${ADV_BENCH_STRICT_VALID_TARGET_PAIRS_JSON:?Set ADV_BENCH_STRICT_VALID_TARGET_PAIRS_JSON to a local target-pairs JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/refusal/advbench_jacobian_v4}"
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python experiments/refusal/baseline_refusal_sae_recovery_crosslayer_v4.py   --dataset_name advbench   --model_name gemma-2b   --feature_source benchmark_our   --feature_scope global   --target_pairs_json "$ADV_BENCH_STRICT_VALID_TARGET_PAIRS_JSON"   --output_dir "$OUTPUT_DIR"   --max_samples 24   --clamp_value 3.0   --projection_mode jacobian   --jacobian_probe_count 8   --jacobian_include_drift_probe   --jacobian_correction_steps 5   --jacobian_correction_lr 0.05   --lambda_act 200   --act_loss_mode relative_l2   --max_delta_norm 40   --best_checkpoint_mode drift_constrained   --best_drift_relative_l2_threshold 0.03   --recovery_target_mode base_response_valid_case   --boundary_stage1_steps 20   --boundary_stage1_lr 0.08   --boundary_stage1_max_delta_norm 30   --boundary_stage1_first_token_logprob_weight 4   --boundary_stage1_target_prefix_logprob_weight 2   --answer_logprob_weight 5   --answer_token_limit 48   --answer_prefix_token_limit 16   --answer_prefix_token_weight 2   --discourage_safety_prefixes
