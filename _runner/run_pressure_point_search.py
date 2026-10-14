#!/usr/bin/env python3
"""Project-local adapter for the controlled pressure-point search supervisor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_runtime_import_paths() -> None:
    repo = str(_repo_root().resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Emergence pressure-point search supervisor")
    parser.add_argument("--campaign_id", required=True)
    parser.add_argument("--runs_root", default="")
    parser.add_argument("--duration_days", type=int, default=28)
    parser.add_argument("--max_active_children", type=int, default=1)
    parser.add_argument("--profile", default="evolvability_band")
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _ensure_runtime_import_paths()
    from experiments.projects.emergence.tools.pressure_search import main as delegate_main

    cli: List[str] = [
        "--campaign_id",
        str(args.campaign_id),
        "--duration_days",
        str(int(args.duration_days)),
        "--max_active_children",
        str(int(args.max_active_children)),
        "--profile",
        str(args.profile),
        "--poll_seconds",
        str(int(args.poll_seconds)),
    ]
    if str(args.runs_root).strip():
        cli.extend(["--runs_root", str(args.runs_root).strip()])
    if args.dry_run:
        cli.append("--dry_run")
    if args.once:
        cli.append("--once")
    return int(delegate_main(cli))


if __name__ == "__main__":
    raise SystemExit(main())
