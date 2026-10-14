#!/usr/bin/env python3
"""Project-local adapter for launching one Emergence registry case.

This is the first explicit Causyl-facing runner seam for the Emergence project.
It resolves registry references against ``_spec/registry`` first, then delegates
to the retained Emergence harness entrypoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_runtime_import_paths() -> None:
    repo_root = str(_repo_root().resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one Emergence registry case via the Causyl project adapter")
    p.add_argument(
        "--registry",
        required=True,
        help="Registry path. Relative paths resolve against _spec/registry.",
    )
    p.add_argument("--only_run_id", required=True, help="Run only this run_id from the registry matrix")
    p.add_argument("--batch_tag", default="", help="Optional harness batch tag")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing batch directory")
    p.add_argument("--print_cmds", action="store_true", help="Print resolved CLI command(s) for this case")
    p.add_argument("--dry_run", action="store_true", help="Expand and write artifacts, but do not execute")
    p.add_argument("--perf", action="store_true", help="Enable per-generation perf buckets")
    return p.parse_args(argv)


def _resolve_registry(raw: str) -> Path:
    from experiments_shared.runner_utils import resolve_registry_path

    return resolve_registry_path(project_root=_project_root(), raw=raw)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _ensure_runtime_import_paths()
    from experiments_shared.runner_utils import ensure_project_output_env
    from experiments.projects.emergence.tools.entrypoints.run_registry_case import main as delegate_main

    for key, value in ensure_project_output_env(
        project_id="emergence",
        repo_root=_repo_root(),
        project_root=_project_root(),
    ).items():
        if key.endswith("_DIR"):
            os.environ[key] = value

    cli: List[str] = [
        "--registry",
        str(_resolve_registry(str(args.registry))),
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
    return int(delegate_main(cli))


if __name__ == "__main__":
    raise SystemExit(main())
