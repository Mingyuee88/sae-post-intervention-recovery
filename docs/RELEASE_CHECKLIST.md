# Release Checklist

Use this checklist before pushing the public repository.

## Required manual fields

- [ ] Confirm final author spelling and affiliation formatting.
- [ ] Add the paper URL to `CITATION.cff`, `pyproject.toml`, and `README.md` once the paper URL exists.

## Safety and artifact review

- [ ] Confirm no raw harmful prompts or completions are included.
- [ ] Confirm refusal manifests contain IDs/categories/labels only.
- [ ] Confirm no model weights, SAE weights, checkpoints, or cache files are included.
- [ ] Confirm no SSH hosts, passwords, API keys, Hugging Face tokens, or private filesystem paths are included.
- [ ] Confirm upstream repositories are acknowledged in `ACKNOWLEDGEMENTS.md` and `docs/THIRD_PARTY.md`.

## Commands

Run from the repository root:

```bash
make clean-generated
make release-check
```

The release check should run unit tests, scan for private artifacts, and print a compact sanitized-results summary.
