#!/usr/bin/env python3
"""
Phase 3 determinism helper: assert CSV equivalence.

This is a small verification tool used in FDP v3 validation:
- Compares metrics.csv and behaviors.csv between two run directories.
- Provides stable, deterministic reporting.
- Exits non-zero on mismatch.

Usage:
    python -m tests.harnesses.phase3_resume.verify.phase3_assert_csv_equivalence --run-a <dir> --run-b <dir>

Notes:
- This script is intentionally dependency-light (stdlib only).
- It compares both:
  (1) byte-for-byte file content, and
  (2) normalized CSV structure (header order + row count) if bytes differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Iterable, Tuple


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_shape(path: Path) -> Tuple[int, int, Tuple[str, ...]]:
    # row_count excludes header
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = 0
        for _ in reader:
            rows += 1
    return rows, len(header), tuple(header)


def _compare_one(label: str, a: Path, b: Path) -> int:
    if not a.exists():
        print(f"[CSV_EQ] FAIL {label}: missing A={a}", file=sys.stderr)
        return 2
    if not b.exists():
        print(f"[CSV_EQ] FAIL {label}: missing B={b}", file=sys.stderr)
        return 2

    sha_a = _sha256_file(a)
    sha_b = _sha256_file(b)

    if sha_a == sha_b:
        print(f"[CSV_EQ] PASS {label}: sha256={sha_a}")
        return 0

    rows_a, cols_a, hdr_a = _read_csv_shape(a)
    rows_b, cols_b, hdr_b = _read_csv_shape(b)

    print(f"[CSV_EQ] FAIL {label}: sha256 A={sha_a} B={sha_b}", file=sys.stderr)
    print(f"[CSV_EQ] A shape rows={rows_a} cols={cols_a}", file=sys.stderr)
    print(f"[CSV_EQ] B shape rows={rows_b} cols={cols_b}", file=sys.stderr)

    if hdr_a != hdr_b:
        print(f"[CSV_EQ] header mismatch", file=sys.stderr)
        # Print a small diff window, stable order
        max_show = 20
        only_a = [h for h in hdr_a if h not in hdr_b][:max_show]
        only_b = [h for h in hdr_b if h not in hdr_a][:max_show]
        if only_a:
            print(f"[CSV_EQ] header only in A (first {len(only_a)}): {only_a}", file=sys.stderr)
        if only_b:
            print(f"[CSV_EQ] header only in B (first {len(only_b)}): {only_b}", file=sys.stderr)

    return 1


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-a", required=True, type=Path, help="Run directory A")
    p.add_argument("--run-b", required=True, type=Path, help="Run directory B")
    args = p.parse_args(list(argv) if argv is not None else None)

    rc = 0
    rc |= _compare_one("metrics.csv", args.run_a / "metrics.csv", args.run_b / "metrics.csv")
    rc |= _compare_one("behaviors.csv", args.run_a / "behaviors.csv", args.run_b / "behaviors.csv")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
