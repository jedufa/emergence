#!/usr/bin/env python3
"""
experiments.runs.run_registry_case

Purpose
-------
Execute exactly ONE run from a harness registry, identified by run_id.

This module exists to bridge:
- harness registry matrix expansion (seeds x axes)
- Phase 2.5 manifest runner parallelization (explicit jobs)

Usage
-----
This module is designed to be called by the Phase 1 wrapper (tools.run_job)
via a manifest job definition, with args like:

  python -m experiments.runs.run_registry_case \
    --registry _spec/registry/v1/tier4_harsh_exposure_nightly.yaml \
    --only_run_id tier4_harsh_exposure__exposure_E1_static_gradient__scenario_baseline__seed000001

Notes
-----
- Do NOT set --runs_root here. The wrapper sets EMERGENCE_RUNS_DIR/EMERGENCE_OUTPUTS_DIR
  so harness output lands under the wrapper payload folder.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description="Run a single registry case via the harness")
    p.add_argument("--registry", required=True, help="Path to registry YAML (relative to repo root or absolute)")
    p.add_argument("--only_run_id", required=True, help="Run only this run_id from the registry matrix")
    p.add_argument("--batch_tag", default="", help="Optional harness batch tag (appended to batch folder)")
    p.add_argument("--overwrite", action="store_true", help="Allow harness to write into an existing batch directory")
    p.add_argument("--print_cmds", action="store_true", help="Print resolved CLI command(s) for this case")
    p.add_argument("--dry_run", action="store_true", help="Expand and write artifacts, but do not execute")
    p.add_argument("--perf", action="store_true", help="Enable per-generation perf buckets (propagates --perf)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entrypoint."""
    args = _parse_args(argv)

    # Import inside main so tools can still lint without harness dependencies.
    from experiments.projects.emergence.experiments.harness.run_experiments import main as harness_main  # noqa: WPS433 (repo-local import)

    cli: List[str] = [
        "--registry",
        str(args.registry),
        "--only_run_id",
        str(args.only_run_id),
    ]

    if str(args.batch_tag).strip():
        cli.extend(["--batch_tag", str(args.batch_tag).strip()])

    if args.overwrite:
        cli.append("--overwrite")
    if args.print_cmds:
        cli.append("--print_cmds")
    if args.dry_run:
        cli.append("--dry_run")
    if args.perf:
        cli.append("--perf")

    return int(harness_main(cli))


if __name__ == "__main__":
    raise SystemExit(main())
