#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_env_action_hotpath_probe_v1"

PROFILES: dict[str, tuple[str, str]] = {
    "A101": (
        "v1/phaseIV_crb_x1_prediction_readiness_bridge.yaml",
        "phaseIV_crb_x1_prediction_readiness_bridge__ecology_crbx1_theme12_clean_pressure__init_init_founder_balanced__prediction_prediction_normal__seed000101",
    ),
    "B101": (
        "v1/lbc1_lineage_bottleneck_closure_source.yaml",
        "lbc1_lineage_bottleneck_closure_source__scenario_lbc1_00_tuned_full_jungle_recovery__seed000101",
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_summary(root: Path, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob("env_hotpath_expression_summary.json"):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1]


def _latest_decomposition(root: Path, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob("env_apply_decomposition_summary.json"):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1]


def _latest_phenotype_decomposition(root: Path, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob("envhot3_phenotype_decomposition_summary.json"):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1]


def _latest_developmental_residual(root: Path, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob("envhot7_developmental_residual_summary.json"):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1]


def _latest_developmental_cache_equality(root: Path, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob("envhot8_developmental_cache_equality_summary.json"):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1]


def _latest_developmental_effective_projection(root: Path, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob("envhot9_developmental_effective_projection_summary.json"):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1]


def run_one(*, label: str, runs_root: Path, batch_prefix: str, knobs: list[str]) -> dict[str, Any]:
    registry, run_id = PROFILES[label]
    cmd = [
        "python",
        "_runner/run_registry_batch.py",
        "--registry",
        registry,
        "--runs_root",
        str(runs_root),
        "--batch_tag",
        f"{batch_prefix}_{label.lower()}",
        "--overwrite",
        "--only_run_id",
        run_id,
        "--perf",
    ]
    for knob in knobs:
        cmd.extend(["--knob", str(knob)])

    started_wall = time.time()
    started = time.perf_counter()
    cp = subprocess.run(cmd, cwd=_project_root(), text=True, capture_output=True, check=False)
    wall_s = time.perf_counter() - started
    summary_path = _latest_summary(runs_root, started_wall)
    decomposition_path = _latest_decomposition(runs_root, started_wall)
    phenotype_decomposition_path = _latest_phenotype_decomposition(runs_root, started_wall)
    developmental_residual_path = _latest_developmental_residual(runs_root, started_wall)
    developmental_cache_equality_path = _latest_developmental_cache_equality(runs_root, started_wall)
    developmental_effective_projection_path = _latest_developmental_effective_projection(runs_root, started_wall)
    summary: dict[str, Any] | None = None
    decomposition: dict[str, Any] | None = None
    phenotype_decomposition: dict[str, Any] | None = None
    developmental_residual: dict[str, Any] | None = None
    developmental_cache_equality: dict[str, Any] | None = None
    developmental_effective_projection: dict[str, Any] | None = None
    if summary_path is not None:
        try:
            summary = _read_json(summary_path)
        except Exception:
            summary = None
    if decomposition_path is not None:
        try:
            decomposition = _read_json(decomposition_path)
        except Exception:
            decomposition = None
    if phenotype_decomposition_path is not None:
        try:
            phenotype_decomposition = _read_json(phenotype_decomposition_path)
        except Exception:
            phenotype_decomposition = None
    if developmental_residual_path is not None:
        try:
            developmental_residual = _read_json(developmental_residual_path)
        except Exception:
            developmental_residual = None
    if developmental_cache_equality_path is not None:
        try:
            developmental_cache_equality = _read_json(developmental_cache_equality_path)
        except Exception:
            developmental_cache_equality = None
    if developmental_effective_projection_path is not None:
        try:
            developmental_effective_projection = _read_json(developmental_effective_projection_path)
        except Exception:
            developmental_effective_projection = None
    return {
        "label": str(label),
        "registry": str(registry),
        "run_id": str(run_id),
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "wall_s": float(wall_s),
        "summary_path": str(summary_path) if summary_path is not None else "",
        "summary": summary,
        "decomposition_path": str(decomposition_path) if decomposition_path is not None else "",
        "decomposition": decomposition,
        "phenotype_decomposition_path": (
            str(phenotype_decomposition_path) if phenotype_decomposition_path is not None else ""
        ),
        "phenotype_decomposition": phenotype_decomposition,
        "developmental_residual_path": (
            str(developmental_residual_path) if developmental_residual_path is not None else ""
        ),
        "developmental_residual": developmental_residual,
        "developmental_cache_equality_path": (
            str(developmental_cache_equality_path) if developmental_cache_equality_path is not None else ""
        ),
        "developmental_cache_equality": developmental_cache_equality,
        "developmental_effective_projection_path": (
            str(developmental_effective_projection_path) if developmental_effective_projection_path is not None else ""
        ),
        "developmental_effective_projection": developmental_effective_projection,
        "stdout_tail": cp.stdout[-4000:],
        "stderr_tail": cp.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A101/B101 env/action expression hotpath diagnostics.")
    parser.add_argument("--runs-root", required=True, help="Runtime output root")
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--label", action="append", default=[], help=f"Known labels: {', '.join(sorted(PROFILES))}")
    parser.add_argument("--batch-prefix", default="env_action_hotpath", help="Batch tag prefix")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for prompt symmetry; run_registry_batch always receives overwrite.")
    parser.add_argument(
        "--knob",
        action="append",
        default=[
            "sim_cfg.generations=10",
            "sim_cfg.steps_per_generation=80",
            "sim_cfg.env_hotpath_expression_diagnostics_enable=true",
            "sim_cfg.env_hotpath_expression_diagnostics_schema_version=1",
            "sim_cfg.env_hotpath_expression_diagnostics_write_events=false",
            "sim_cfg.env_hotpath_apply_decomposition_enable=true",
            "sim_cfg.env_hotpath_apply_decomposition_schema_version=1",
            "sim_cfg.env_hotpath_phenotype_decomposition_enable=true",
            "sim_cfg.env_hotpath_phenotype_decomposition_schema_version=1",
            "sim_cfg.env_hotpath_developmental_residual_diagnostics_enable=true",
            "sim_cfg.env_hotpath_developmental_residual_diagnostics_schema_version=1",
            "sim_cfg.env_hotpath_developmental_cache_equality_diagnostics_enable=true",
            "sim_cfg.env_hotpath_developmental_cache_equality_schema_version=1",
            "sim_cfg.env_hotpath_developmental_effective_projection_diagnostics_enable=true",
            "sim_cfg.env_hotpath_developmental_effective_projection_schema_version=1",
        ],
        help="Knob override passed to every child run.",
    )
    args = parser.parse_args()

    labels = [str(label).strip() for label in (args.label or []) if str(label).strip()] or ["A101", "B101"]
    unknown = sorted(set(labels) - set(PROFILES))
    if unknown:
        raise SystemExit(f"unknown labels: {', '.join(unknown)}")

    runs_root = Path(str(args.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = [
        run_one(label=label, runs_root=runs_root, batch_prefix=str(args.batch_prefix), knobs=[str(k) for k in args.knob])
        for label in labels
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runs_root": str(runs_root),
        "labels": labels,
        "total_wall_s": float(time.perf_counter() - started),
        "results": results,
        "failures": [
            row
            for row in results
            if int(row.get("returncode", 1)) != 0
            or not str(row.get("summary_path", ""))
            or not str(row.get("decomposition_path", ""))
            or not str(row.get("phenotype_decomposition_path", ""))
            or not str(row.get("developmental_residual_path", ""))
            or not str(row.get("developmental_cache_equality_path", ""))
            or not str(row.get("developmental_effective_projection_path", ""))
        ],
    }
    out = Path(str(args.out)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "total_wall_s": payload["total_wall_s"], "failures": len(payload["failures"])}, sort_keys=True))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
