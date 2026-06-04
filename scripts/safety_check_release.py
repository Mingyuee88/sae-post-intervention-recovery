#!/usr/bin/env python3
"""Release-safety checks for the public artifact."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NAME_PARTS = (
    "samples",
    "preflight_rows",
    "answer_records",
    "raw_outputs",
)
FORBIDDEN_SUFFIXES = (
    ".log",
    ".pid",
    ".pt",
    ".pth",
    ".pkl",
    ".safetensors",
)
FORBIDDEN_TEXT = (
    "/" + "nf" + "shdd" + "/",
    "Cmy" + "cmy",
    "OPENAI" + "_API_KEY",
    "HF" + "_TOKEN",
    "WANDB" + "API_KEY",
)
RAW_RESULT_KEYS = (
    "base_response",
    "clamped_response",
    "recovered_response",
    "target_answer",
    "instruction",
)
ALLOWED_PLACEHOLDER_FILES = {
    "CITATION.cff",
    "pyproject.toml",
    "docs/RELEASE_CHECKLIST.md",
}


def iter_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
            yield path


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_names() -> list[str]:
    offenders = []
    for path in iter_files():
        name = rel(path).lower()
        if path.name.startswith("._"):
            offenders.append(rel(path))
        if any(part in name for part in FORBIDDEN_NAME_PARTS):
            offenders.append(rel(path))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(rel(path))
    return sorted(set(offenders))


def check_private_text() -> list[str]:
    offenders = []
    allowed_suffixes = {".py", ".md", ".tex", ".json", ".csv", ".yml", ".yaml", ".sh", ".txt", ".toml", ".cff"}
    for path in iter_files():
        if path.suffix.lower() not in allowed_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in FORBIDDEN_TEXT):
            offenders.append(rel(path))
    return offenders


def check_raw_result_fields() -> list[str]:
    offenders = []
    results = ROOT / "results" / "sanitized"
    if not results.exists():
        return ["results/sanitized is missing"]
    for path in results.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".tex", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(key in text for key in RAW_RESULT_KEYS):
            offenders.append(rel(path))
    return offenders


def check_placeholders() -> list[str]:
    offenders = []
    markers = ("TODO:", "TODO ")
    for path in iter_files():
        r = rel(path)
        if r in ALLOWED_PLACEHOLDER_FILES:
            continue
        if path.suffix.lower() not in {".md", ".toml", ".cff", ".yml", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in markers):
            offenders.append(r)
    return offenders


def main() -> int:
    checks = {
        "forbidden artifact names": check_names(),
        "private paths or credentials": check_private_text(),
        "raw result fields": check_raw_result_fields(),
        "unexpected placeholders": check_placeholders(),
    }
    failed = {name: offenders for name, offenders in checks.items() if offenders}
    if failed:
        print(json.dumps(failed, indent=2, ensure_ascii=False))
        return 1
    print("Release-safety checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
