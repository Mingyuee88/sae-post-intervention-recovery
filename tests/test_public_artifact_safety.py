"""Smoke checks for the public release artifact.

These tests are intentionally lightweight: they verify that the checked-in
artifact contains code, sanitized metrics, and documentation, but not raw model
outputs, private paths, credentials, or bulky model artifacts.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NAME_PARTS = (
    "samples",
    "preflight_rows",
    "answer_records",
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
)
RAW_RESULT_KEYS = (
    "base_response",
    "clamped_response",
    "recovered_response",
    "target_answer",
    "instruction",
)


def iter_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path


def test_no_raw_or_bulk_artifact_names():
    offenders = []
    for path in iter_files():
        rel = path.relative_to(ROOT).as_posix()
        lower = rel.lower()
        if any(part in lower for part in FORBIDDEN_NAME_PARTS):
            offenders.append(rel)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            offenders.append(rel)
    assert not offenders


def test_no_private_paths_or_credentials():
    offenders = []
    for path in iter_files():
        if path.suffix.lower() not in {".py", ".md", ".tex", ".json", ".csv", ".yml", ".yaml", ".sh", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in FORBIDDEN_TEXT):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders


def test_sanitized_results_do_not_include_full_answer_fields():
    results = ROOT / "results" / "sanitized"
    offenders = []
    for path in results.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".tex"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(key in text for key in RAW_RESULT_KEYS):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders
