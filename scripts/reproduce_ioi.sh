#!/usr/bin/env bash
set -euo pipefail

# Reproduce IOI recovery.
# Required environment variables:
#   IOI_DATASET_SOURCE: path to the official IOI dataset Python source file
# Optional:
#   OUTPUT_JSON: output path for the aggregate JSON

: "${IOI_DATASET_SOURCE:?Set IOI_DATASET_SOURCE to the official IOI dataset Python file}"
OUTPUT_JSON="${OUTPUT_JSON:-runs/ioi/ioi_recovery.json}"
mkdir -p "$(dirname "$OUTPUT_JSON")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python - <<'PY_IOI'
import json
import os
from pathlib import Path
from sae_lens import SAE
from transformer_lens import HookedTransformer
from experiments.ioi.ioi_recovery import build_official_ioi_dataset, run_ioi_recovery_experiment, save_json

output_json = Path(os.environ.get('OUTPUT_JSON', 'runs/ioi/ioi_recovery.json'))
model = HookedTransformer.from_pretrained('gpt2', device='cuda')
sae, _, _ = SAE.from_pretrained(release='gpt2-small-res-jb', sae_id='blocks.4.hook_resid_pre', device='cuda')
dataset = build_official_ioi_dataset(
    source_path=os.environ['IOI_DATASET_SOURCE'],
    tokenizer=model.tokenizer,
    prompt_type='BABA',
    n_prompts=64,
    nb_templates=1,
    seed=0,
)
result = run_ioi_recovery_experiment(
    model=model,
    sae=sae,
    hook_name='blocks.4.hook_resid_pre',
    tokens=dataset.toks.cuda(),
    io_token_ids=dataset.io_tokenIDs.cuda(),
    s_token_ids=dataset.s_tokenIDs.cuda(),
    answer_positions=dataset.word_idx['end'],
    topk_features=64,
    clamp_multiplier=5.0,
    modes=('none', 'encoder'),
    num_steps=100,
    lr=0.1,
    max_delta_norm=20.0,
    seed=0,
)
save_json(result, output_json)
print(json.dumps({'output': str(output_json)}, indent=2))
PY_IOI
