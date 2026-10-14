#!/usr/bin/env python3
"""Summarize Patch/Jungle Step 14 registry batches.

This script is reporting-only. It reads an existing harness batch directory and
writes deterministic JSON with scenario summaries and raw comparison deltas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_import_path() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Patch/Jungle V0 Step 14 batch")
    parser.add_argument("--batch-dir", required=True, help="Harness batch directory containing matrix.csv")
    parser.add_argument("--out", required=True, help="JSON summary output path")
    parser.add_argument(
        "--tail-n",
        type=int,
        default=1,
        help="Average the final N metrics rows per run; default uses the final row only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ensure_import_path()
    from experiments.patch_jungle_eval import summarize_batch, write_summary_json

    args = parse_args(argv)
    summary = summarize_batch(args.batch_dir, tail_n=max(1, int(args.tail_n)))
    write_summary_json(args.out, summary)
    print(f"[patch-jungle-eval] wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
