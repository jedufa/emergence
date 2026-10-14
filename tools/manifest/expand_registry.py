#!/usr/bin/env python3
"""
tools.manifest.expand_registry

Stability: EXPERIMENTAL (tooling-only)

Purpose
-------
Expand an experiments/harness registry YAML into explicit Phase 2.5 manifest jobs,
so the *manifest runner* can schedule seeds/cases in parallel.

Why this exists
---------------
The harness registry format can contain many seeds and axis combinations inside a
single registry file. If you run that via a single wrapper job, the manifest runner
cannot parallelize those internal runs (it sees only one job).

This tool explodes:
  registry.yaml (axes x seeds) -> N manifest jobs (typically 1 seed per job)

and can optionally append the resulting jobs to an existing manifest with safe
idempotent dedupe (fingerprint-based).

Key behaviors
-------------
- Deterministic expansion order (matrix_order from the registry).
- Deterministic job_id suffix based on a stable fingerprint.
- Idempotent append: re-running against the same manifest will skip duplicates.

Output
------
Either:
- --out-jobs: write a YAML fragment with a 'jobs:' list
- --append-to: append jobs into an existing manifest file

Notes
-----
This tool relies on the harness supporting:
  --only_run_id <run_id>
which is implemented in experiments/harness/run_experiments.py.

Execution entrypoint used by expanded jobs:
  tools.entrypoints.run_registry_case
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

_yaml_import_error: str | None = None
try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore
    _yaml_import_error = str(e)

from tools.manifest.manifest_ops import append_jobs, compute_job_fingerprint, dump_manifest_yaml, load_manifest, write_manifest


def _repo_root_from_this_file() -> Path:
    """Best-effort repo root discovery.

    This module has shown up in multiple layouts across snapshots
    (e.g. tools/, tools/scripts/, tools/manifest/). Using a fixed parent index
    is brittle, so we probe upward for the directories we expect at repo root.
    """

    here = Path(__file__).resolve().parent
    cands = [
        here.parent,  # one up
        here.parent.parent,  # two up
        here.parent.parent.parent,  # three up
    ]
    for cand in cands:
        try:
            if (cand / "tools").is_dir() and (cand / "scripts").is_dir() and (cand / "experiments").is_dir():
                return cand
        except Exception:
            continue
    return cands[0]


def _load_registry(path: str) -> Dict[str, Any]:
    """Load registry YAML (as dict)."""
    repo_root = _repo_root_from_this_file()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Registry not found: {p}")

    if yaml is None:  # type: ignore[name-defined]
        raise SystemExit(
            "PyYAML is required for tools/expand_registry.py. "
            "Install it in your venv (pip install pyyaml). "
            f"Import error: {_yaml_import_error}"
        )

    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError("Registry must parse into a mapping/dict")
    doc["__resolved_path"] = str(p)
    # Keep original string argument for reproducible identity.
    doc["__arg_path"] = str(Path(path))
    return doc


def _expand_rows(reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand registry matrix into ordered rows of axis values + seed."""
    axes = reg.get("axes") or {}
    matrix_order = list(reg.get("matrix_order") or [])
    seeds = list(reg.get("seeds") or [])
    if not matrix_order:
        raise ValueError("Registry missing matrix_order")
    if not seeds:
        raise ValueError("Registry missing seeds")

    # Build per-axis value lists
    axis_values: Dict[str, List[Any]] = {}
    for name in matrix_order:
        if name == "seed":
            continue
        ax = axes.get(name) or {}
        vals = ax.get("values")
        if vals is None:
            # Some registries may define intent.axes but not axes.values; refuse silently guessing.
            raise ValueError(f"Registry axis '{name}' missing axes.{name}.values")
        axis_values[name] = list(vals)

    ordered_axes = [n for n in matrix_order if n != "seed"]

    rows: List[Dict[str, Any]] = []

    def rec(i: int, ctx: Dict[str, Any]) -> None:
        if i >= len(ordered_axes):
            for s in seeds:
                row = dict(ctx)
                row["seed"] = int(s)
                rows.append(row)
            return
        name = ordered_axes[i]
        for v in axis_values[name]:
            ctx2 = dict(ctx)
            ctx2[name] = v
            rec(i + 1, ctx2)

    rec(0, {})
    return rows


def _format_template(tpl: str, ctx: Dict[str, Any]) -> str:
    """Format a template string using ctx (supports {seed:06d})."""
    return tpl.format(**ctx)


def _job_for_row(reg: Dict[str, Any], row: Dict[str, Any], *, entry_module: str) -> Dict[str, Any]:
    """Create a manifest job def for a single registry row."""
    naming = reg.get("naming") or {}
    name = str(reg.get("name") or "registry")
    run_tag_prefix = str(naming.get("run_tag_prefix") or name)

    # Build ctx for naming templates
    ctx: Dict[str, Any] = {
        "name": name,
        "run_tag_prefix": run_tag_prefix,
    }
    for k, v in row.items():
        ctx[k] = v

    run_id_tpl = str(naming.get("run_id_format") or "{name}__seed{seed:06d}")
    run_id = _format_template(run_id_tpl, ctx)

    # Identity for fingerprinting (stable + minimal)
    identity = {
        "registry": str(reg.get("__arg_path") or reg.get("__resolved_path") or ""),
        "registry_name": name,
        "registry_version": int(reg.get("version") or 0),
        "run_id": run_id,
        "seed": int(row.get("seed", 0)),
        "axes": {k: row[k] for k in row.keys() if k != "seed"},
        "module": entry_module,
    }
    fp = compute_job_fingerprint(identity)
    short = fp.split(":", 1)[1][:7]

    # Compact deterministic job_id (path-safe and bounded).
    seed_i = int(row.get("seed", 0))
    job_id = f"j{seed_i:06d}_{short}"

    job: Dict[str, Any] = {
        "job_id": job_id,
        "module": "tools.entrypoints.run_registry_case",
        "args": {
            "registry": str(reg.get("__arg_path") or reg.get("__resolved_path") or ""),
            "only_run_id": str(run_id),
        },
        "meta": {
            "registry_entry": name,
            "registry_path": str(reg.get("__arg_path") or reg.get("__resolved_path") or ""),
            "run_id": str(run_id),
            "seed": int(row.get("seed", 0)),
            "axes": {k: row[k] for k in row.keys() if k != "seed"},
            "fingerprint": fp,
        },
    }
    return job


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expand a harness registry into Phase 2.5 manifest jobs")
    p.add_argument("--registry", required=True, help="Path to registry YAML (relative to repo root or absolute)")
    p.add_argument("--out-jobs", default="", help="Write YAML fragment containing jobs: [...] to this path")
    p.add_argument("--append-to", default="", help="Append expanded jobs into an existing manifest YAML/JSON")
    p.add_argument(
        "--dedupe",
        default="fingerprint",
        choices=["fingerprint", "job_id"],
        help="Dedupe strategy when appending (default: fingerprint)",
    )
    p.add_argument(
        "--entry-module",
        default="tools.entrypoints.run_registry_case",
        help="Entrypoint module for each expanded job (default: tools.entrypoints.run_registry_case)",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    if bool(args.out_jobs) == bool(args.append_to):
        raise SystemExit("Provide exactly one of --out-jobs or --append-to")

    reg = _load_registry(args.registry)
    rows = _expand_rows(reg)

    jobs = [_job_for_row(reg, r, entry_module=str(args.entry_module)) for r in rows]
    expanded = len(jobs)

    if args.out_jobs:
        frag = {"jobs": jobs}
        Path(args.out_jobs).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_jobs).write_text(dump_manifest_yaml(frag), encoding="utf-8")
        print(f"Wrote jobs fragment: {args.out_jobs} (expanded={expanded})")
        return 0

    manifest = load_manifest(args.append_to)
    appended, skipped = append_jobs(manifest, jobs, dedupe_policy=str(args.dedupe))
    write_manifest(args.append_to, manifest)

    # IMPORTANT: "total_new" means "actually appended into the manifest", not "expanded candidates".
    total_new = appended
    print(
        f"Appended jobs into: {args.append_to} "
        f"(expanded={expanded} appended={appended} skipped={skipped} total_new={total_new})"
    )
    return 0


# ============================================================
# Safe CLI wrapper (no Python traceback by default)
# ============================================================

def _main_safe() -> int:
    """Run main() with friendly error reporting.

    Set environment variable EMERGENCE_TRACEBACK=1 to print full tracebacks.
    """
    import os
    import traceback
    try:
        return int(main())
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # SystemExit with message/string
        print(f"[ERROR] {code}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[ERROR] Interrupted (KeyboardInterrupt).", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        if os.environ.get("EMERGENCE_TRACEBACK", "") == "1":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(_main_safe())
