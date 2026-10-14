#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_summary(path: Path) -> dict[str, dict[str, str]]:
    if path.is_dir():
        path = path / "perf_benchmark_summary.csv"
    rows = _read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get("benchmark", "")).strip()
        if name:
            out[name] = row
    return out


def compare(*, before: Path, after: Path, markdown: Path | None) -> int:
    before_rows = _load_summary(before)
    after_rows = _load_summary(after)
    names = sorted(set(before_rows) | set(after_rows))
    lines = [
        "# EIEL Perf Compare",
        "",
        "| benchmark | before sec/gen | after sec/gen | speedup | before top | after top |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name in names:
        b = before_rows.get(name, {})
        a = after_rows.get(name, {})
        b_sec = _float(b.get("sec_gen"))
        a_sec = _float(a.get("sec_gen"))
        speedup = (b_sec / a_sec) if a_sec > 0.0 else 0.0
        lines.append(
            f"| {name} | {b_sec:.3f} | {a_sec:.3f} | {speedup:.3f}x | "
            f"{b.get('top_bottleneck', '')} | {a.get('top_bottleneck', '')} |"
        )
    text = "\n".join(lines) + "\n"
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two EIEL perf aggregate summaries")
    parser.add_argument("--before", required=True, help="Before aggregate dir or perf_benchmark_summary.csv")
    parser.add_argument("--after", required=True, help="After aggregate dir or perf_benchmark_summary.csv")
    parser.add_argument("--markdown", default="", help="Optional markdown output")
    args = parser.parse_args()
    return compare(
        before=Path(args.before),
        after=Path(args.after),
        markdown=Path(args.markdown) if str(args.markdown).strip() else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
