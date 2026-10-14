#!/usr/bin/env python3
"""Project-local adapter for launching an Emergence registry batch.

This adapter keeps the plane-facing launch seam small and explicit while the
retained Emergence harness continues to own matrix expansion and run execution.
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
    path = Path(__file__).resolve()
    return path.parents[4] if len(path.parents) > 4 else _project_root()


def _ensure_runtime_import_paths() -> None:
    repo_root = str(_repo_root().resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an Emergence registry batch via the project-local adapter"
    )
    p.add_argument(
        "--registry",
        required=True,
        help="Registry path. Relative paths resolve against _spec/registry.",
    )
    p.add_argument(
        "--runs_root",
        default="",
        help="Optional runs root. If omitted, the retained harness default/env resolution applies.",
    )
    p.add_argument("--batch_tag", default="", help="Optional harness batch tag")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing batch directory")
    p.add_argument("--dry_run", action="store_true", help="Expand and write artifacts, but do not execute")
    p.add_argument("--print_cmds", action="store_true", help="Print resolved CLI commands")
    p.add_argument(
        "--only_run_id",
        action="append",
        default=[],
        help="Restrict execution to one or more run_id values from the registry matrix",
    )
    p.add_argument(
        "--only_seed",
        action="append",
        default=[],
        help="Restrict execution to one or more seed values",
    )
    p.add_argument(
        "--knob",
        action="append",
        default=[],
        help="Pass-through knob override applied to every child run",
    )
    p.add_argument(
        "--performance_profile",
        "--performance-profile",
        dest="performance_profile",
        default="",
        help="Pass-through named simulation performance profile applied after generic --knob overrides",
    )
    p.add_argument(
        "--campaign_output_profile",
        "--campaign-output-profile",
        dest="campaign_output_profile",
        default="compact_default",
        help="Pass-through named campaign output/storage profile",
    )
    p.add_argument(
        "--allow_large_output",
        "--allow-large-output",
        dest="allow_large_output",
        action="store_true",
        help="Acknowledge large debug/full-replay output profile risk",
    )
    p.add_argument(
        "--allow_generation_checkpointing",
        "--allow-generation-checkpointing",
        dest="allow_generation_checkpointing",
        action="store_true",
        help="Allow per-generation checkpointing when full_replay is selected",
    )
    p.add_argument("--perf", action="store_true", help="Enable per-generation perf buckets")
    return p.parse_args(argv)


def _resolve_registry(raw: str) -> Path:
    from experiments_shared.runner_utils import resolve_registry_path

    return resolve_registry_path(project_root=_project_root(), raw=raw)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    _ensure_runtime_import_paths()
    from experiments_shared.runner_utils import ensure_project_output_env, normalize_project_runs_root
    try:
        from experiments.projects.emergence.experiments.harness.run_experiments import main as delegate_main
    except ModuleNotFoundError:
        from experiments.harness.run_experiments import main as delegate_main

    normalized_runs_root = ""
    if str(args.runs_root).strip():
        normalized_runs_root = str(
            normalize_project_runs_root(
                repo_root=_repo_root(),
                project_root=_project_root(),
                project_id="emergence",
                raw=str(args.runs_root),
            )
        )
    for key, value in ensure_project_output_env(
        project_id="emergence",
        repo_root=_repo_root(),
        project_root=_project_root(),
        runs_root=str(args.runs_root),
    ).items():
        if key.endswith("_DIR"):
            os.environ[key] = value

    cli: List[str] = ["--registry", str(_resolve_registry(str(args.registry)))]
    if normalized_runs_root:
        cli.extend(["--runs_root", normalized_runs_root])
    if str(args.batch_tag).strip():
        cli.extend(["--batch_tag", str(args.batch_tag).strip()])
    if args.overwrite:
        cli.append("--overwrite")
    if args.dry_run:
        cli.append("--dry_run")
    if args.print_cmds:
        cli.append("--print_cmds")
    if args.perf:
        cli.append("--perf")
    performance_profile = str(getattr(args, "performance_profile", "") or "").strip()
    if performance_profile:
        cli.extend(["--performance_profile", performance_profile])
    campaign_output_profile = str(getattr(args, "campaign_output_profile", "") or "").strip()
    if campaign_output_profile:
        cli.extend(["--campaign_output_profile", campaign_output_profile])
    if bool(getattr(args, "allow_large_output", False)):
        cli.append("--allow_large_output")
    if bool(getattr(args, "allow_generation_checkpointing", False)):
        cli.append("--allow_generation_checkpointing")
    for run_id in args.only_run_id:
        value = str(run_id).strip()
        if value:
            cli.extend(["--only_run_id", value])
    for seed in args.only_seed:
        value = str(seed).strip()
        if value:
            cli.extend(["--only_seed", value])
    for knob in args.knob:
        value = str(knob).strip()
        if value:
            cli.extend(["--knob", value])
    return int(delegate_main(cli))


if __name__ == "__main__":
    raise SystemExit(main())
