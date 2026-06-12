<div align="center">
      
# SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![Paper](https://img.shields.io/badge/Paper-PDF-orange.svg)](https://mingyuee88.github.io/sae-post-intervention-recovery/static/pdf/SAE_preprint.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://mingyuee88.github.io/sae-post-intervention-recovery/)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/Mingyuee88/sae-post-intervention-recovery)

<br>



<video src="https://mingyuee88.github.io/sae-post-intervention-recovery/static/images/Animation.mp4" width="80%" autoplay loop muted playsinline controls>

  Your browser does not support the video tag.

</video>



</div>



<br>


> **SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior**  
> [Mingyue Cui](https://github.com/Mingyuee88), [Linghui Shen](https://github.com/LinghuiiShen), [Xingyi Yang](https://adamdad.github.io/)  
> The Hong Kong Polytechnic University  

This repository contains the public code and sanitized artifacts for the paper **"SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior"**.

The central question is: after an SAE feature clamp has already suppressed a behavior, can the behavior still be recovered from the defended residual state while the clamped SAE features remain close to their defended values?

In short, this project tests whether SAE feature interventions form reliable behavioral bottlenecks. Across latent-level, unlearning, circuit, and refusal settings, we intervene on model activations, then ask whether a constrained recovery direction can restore the suppressed behavior without simply undoing the clamped SAE features.
<img width="917" height="563" alt="image" src="https://github.com/user-attachments/assets/f1a5b20f-efb2-4dd1-9e22-5cbbfb3b3912" />



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

## External Sources

This repository does not redistribute model weights, SAE weights, or benchmark datasets. Download them from the original sources and comply with their terms.

| Used in | Resource | Source |
| --- | --- | --- |
| TPP, WMDP-Bio, refusal | Gemma 2 2B base model (`google/gemma-2-2b`) | https://huggingface.co/google/gemma-2-2b |
| TPP, WMDP-Bio, refusal | Gemma Scope residual SAEs (`google/gemma-scope-2b-pt-res`) | https://huggingface.co/google/gemma-scope-2b-pt-res |
| IOI | GPT-2 Small (`gpt2` / `openai-community/gpt2`) | https://huggingface.co/openai-community/gpt2 |
| IOI | GPT-2 Small residual SAEs (`gpt2-small-res-jb`, e.g. `blocks.4.hook_resid_pre`) | https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted |
| TPP | SAEBench official benchmark infrastructure | https://github.com/adamkarvonen/SAEBench |
| WMDP-Bio | WMDP multiple-choice dataset (`cais/wmdp`, WMDP-Bio split) | https://huggingface.co/datasets/cais/wmdp |
| Refusal | AdvBench harmful behaviors | https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv |
| Refusal appendix | HarmBench-Test | https://github.com/centerforaisafety/HarmBench |
| IOI | Official IOI dataset/code source from Easy-Transformer | https://github.com/redwoodresearch/Easy-Transformer |

## Quick Start

### 1. Targeted Probe Perturbation (TPP) latent-level recovery

This experiment uses the external SAEBench TPP pipeline plus the recovery utilities in this repository.

```bash
cp configs/tpp.env.example configs/tpp.env
# Edit configs/tpp.env so SAEBENCH_EXTERNAL_ROOT points to your local SAEBench checkout.
source configs/tpp.env
bash scripts/reproduce_tpp.sh
```


### 2. WMDP-Bio unlearning recovery

This experiment evaluates output-level recovery on strict WMDP-Bio multiple-choice flips.

```bash
cp configs/unlearning.env.example configs/unlearning.env
# Edit configs/unlearning.env so WMDP_BIO_PROMPT_POOL_MANIFEST points to your local manifest.
source configs/unlearning.env
bash scripts/reproduce_unlearning.sh
```

### 3. Indirect Object Identification (IOI) circuit-level recovery

This experiment uses GPT-2 Small, GPT-2 Small residual SAEs, and the official IOI dataset source from Easy-Transformer.

```bash
cp configs/ioi.env.example configs/ioi.env
# Edit configs/ioi.env so IOI_DATASET_SOURCE points to easy_transformer/ioi_dataset.py.
source configs/ioi.env
bash scripts/reproduce_ioi.sh
```

### 4. Refusal recovery and recovery-path attribution

This experiment uses Gemma 2 2B, Gemma Scope residual SAEs, AdvBench strict-valid target pairs, and the final cross-layer Jacobian recovery runner.

```bash
cp configs/refusal.env.example configs/refusal.env
# Edit configs/refusal.env so ADV_BENCH_STRICT_VALID_TARGET_PAIRS_JSON points to your local target-pairs JSON.
source configs/refusal.env
bash scripts/reproduce_refusal.sh
```


## Responsible Release

This repository is a diagnostic artifact for evaluating whether SAE interventions form complete behavioral bottlenecks. It is not intended as a turnkey jailbreak or deployment attack package.

For safety-relevant refusal experiments, the public release includes aggregate statistics, detector labels, prompt IDs, and coarse redacted categories. It does not include full harmful prompts, full harmful completions, raw `samples.json`, answer-record files, private logs, cached model activations, model weights, or SAE weights.

## Acknowledgements

This codebase builds on and adapts infrastructure from SAEBench, Obfuscated Activations / OABD, and the Easy-Transformer IOI dataset code. See [ACKNOWLEDGEMENTS.md](https://github.com/Mingyuee88/sae-post-intervention-recovery/blob/main/ACKNOWLEDGEMENTS.md) for details.

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
