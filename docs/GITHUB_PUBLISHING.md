# GitHub Publishing Guide

This directory is intended to be published as its own repository, separate from the larger research checkout.

## Prepare metadata

The repository URL is already set to:

```text
https://github.com/Mingyuee88/sae-post-intervention-recovery
```

The paper URL is intentionally left out for now. Add it to `CITATION.cff`, `pyproject.toml`, and README after the paper has an arXiv URL, conference URL, or project page.

## Local release check

Run from this directory after creating the environment:

```bash
conda env create -f environment.yml
conda activate sae-intervention-recovery
pip install -e ".[dev]"
make clean-generated
make release-check
```

If you do not want to install the full experiment environment, at least run:

```bash
python3 scripts/safety_check_release.py
python3 scripts/inspect_results.py
bash -n scripts/reproduce_tpp.sh scripts/reproduce_unlearning.sh scripts/reproduce_ioi.sh scripts/reproduce_refusal.sh
```

## Push this directory to GitHub

From inside `OpenSource/`:

```bash
git init
git add .
git status --short
git commit -m "Initial public release"
git branch -M main
git remote add origin git@github.com:Mingyuee88/sae-post-intervention-recovery.git
git push -u origin main
```

If `git remote add origin` says the remote already exists, update it instead:

```bash
git remote set-url origin git@github.com:Mingyuee88/sae-post-intervention-recovery.git
git push -u origin main
```

## After publishing

- Check the GitHub Actions release-check workflow.
- Confirm README links render correctly.
- Confirm `results/sanitized/` files are visible but raw outputs are absent.
- Create a release tag such as `v0.1.0` after the first successful CI run.
