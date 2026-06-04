# Reproducibility Notes

## Experiments

- TPP: `experiments/tpp/run_tpp_recovery_external.py`
- WMDP-Bio unlearning: `experiments/unlearning/recovery_unlearning_choice_only_seqwide_act.py`
- Refusal SAE recovery: `experiments/refusal/baseline_refusal_sae_recovery_crosslayer_v4.py`
- IOI: `experiments/ioi/ioi_recovery.py`

## Metrics

The key shared metrics are recovery rate, base-like recovery, defended-feature drift, floor violation, and relative delta norm. For unlearning, use `experiments/unlearning/posthoc_eval_unlearning_recovery.py` to recompute metrics under one shared post-hoc evaluator.

## Sanitized results

Sanitized aggregate results are under `results/sanitized`. These files are sufficient to regenerate paper tables and appendix summaries without releasing raw harmful generations.

## Private artifacts required for full reruns

Full reruns require access to the original datasets, the Gemma model variant, SAE releases, and target-pair manifests. The public repository should document how to obtain them, but should not redistribute restricted artifacts.
