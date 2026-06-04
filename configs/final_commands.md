# Final Reproduction Commands

Run all commands from the repository root.

## 1. Environment

For artifact inspection and release checks:

```bash
conda env create -f environment.yml
conda activate sae-intervention-recovery
pip install -e ".[dev]"
make release-check
python scripts/inspect_results.py
```

For full GPU experiments:

```bash
pip install -e ".[experiments,dev]"
```

The quoted extras are intentional. They work in bash and avoid zsh treating `[]` as a filename pattern.

## 2. TPP

```bash
cp configs/tpp.env.example configs/tpp.env
# Edit configs/tpp.env so SAEBENCH_EXTERNAL_ROOT points to your local SAEBench checkout.
source configs/tpp.env
bash scripts/reproduce_tpp.sh
```

## 3. WMDP-Bio Unlearning

```bash
cp configs/unlearning.env.example configs/unlearning.env
# Edit configs/unlearning.env so WMDP_BIO_PROMPT_POOL_MANIFEST points to your local manifest.
source configs/unlearning.env
bash scripts/reproduce_unlearning.sh
```

## 4. IOI

```bash
cp configs/ioi.env.example configs/ioi.env
# Edit configs/ioi.env so IOI_DATASET_SOURCE points to the official IOI dataset Python source.
source configs/ioi.env
bash scripts/reproduce_ioi.sh
```

## 5. Refusal Recovery

```bash
cp configs/refusal.env.example configs/refusal.env
# Edit configs/refusal.env so ADV_BENCH_STRICT_VALID_TARGET_PAIRS_JSON points to your local target-pairs JSON.
source configs/refusal.env
bash scripts/reproduce_refusal.sh
```

Generated outputs are written under `runs/` by default and are ignored by git.
