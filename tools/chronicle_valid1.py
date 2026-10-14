#!/usr/bin/env python3
from __future__ import annotations

"""Run deterministic Scientific Chronicle VALID1 quality analysis."""

import argparse
from pathlib import Path

from simulation.chronicle.quality import analyze_validation_matrix
from simulation.chronicle.schema import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Scientific Chronicle episode quality.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Review a completed validation matrix.")
    analyze.add_argument("--matrix", type=Path, required=True)
    analyze.add_argument("--runs-root", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "analyze":
        result = analyze_validation_matrix(args.matrix, args.runs_root, args.output_dir)
        print(canonical_json(result))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
