#!/usr/bin/env python3
"""Print a compact summary of sanitized paper result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "sanitized"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def maybe(path: str) -> Path | None:
    p = RESULTS / path
    return p if p.exists() else None


def main() -> int:
    print("Sanitized artifact summary")
    print(f"root: {RESULTS}")

    for name in [
        "main_refusal_strict_valid_table.json",
        "harmbench_strict_valid_summary.json",
        "unlearning_posthoc_aggregate.json",
        "ioi_aggregate.json",
        "refusal_recovery_path_attribution_summary.json",
    ]:
        path = maybe(name)
        if path is None:
            print(f"missing: {name}")
            continue
        data = load_json(path)
        if isinstance(data, dict):
            print(f"json: {name} keys={list(data)[:8]}")
        else:
            print(f"json: {name} type={type(data).__name__}")

    for name in [
        "neurips_main_table.csv",
        "dataset_summary_unweighted.csv",
        "overall_summary_unweighted.csv",
        "uncertainty_intervals.csv",
    ]:
        path = maybe(name)
        if path is None:
            print(f"missing: {name}")
            continue
        print(f"csv:  {name} rows={count_csv_rows(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
