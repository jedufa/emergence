#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_phenostruct1_expression_profile_v1"
csv.field_size_limit(sys.maxsize)

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

FINAL_METRIC_FIELDS = (
    "generation",
    "alive",
    "deaths",
    "successful_births",
    "failed_births",
    "lineage_active_count",
    "lineage_effective_alive_lineages",
    "lineage_top1_alive_share",
    "final_effective_alive_lineages",
    "final_top1_alive_share",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_named_file(root: Path, name: str, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob(name):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1] if matches else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _final_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return {"path": str(metrics_path), "row_count": 0, "final": {}}
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final = rows[-1] if rows else {}
    return {
        "path": str(metrics_path),
        "row_count": len(rows),
        "final": {field: final.get(field, "") for field in FINAL_METRIC_FIELDS},
    }


def _checkpoint_summary(run_dir: Path) -> dict[str, Any]:
    bundle_root = run_dir / "checkpoint_bundles_v2"
    metadata_paths = sorted(bundle_root.glob("bundle_gen*/metadata.json")) if bundle_root.exists() else []
    metadata_hashes = {path.parent.name: _sha256(path) for path in metadata_paths}
    metadata_digest = hashlib.sha256(
        json.dumps(metadata_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "bundle_root": str(bundle_root),
        "bundle_count": len(metadata_paths),
        "metadata_digest": metadata_digest,
        "index_present": bool((run_dir / "viz" / "checkpoints_index.json").exists()),
    }


def _run_host_command(cmd: list[str]) -> dict[str, Any]:
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "stdout": cp.stdout[-4000:],
        "stderr": cp.stderr[-4000:],
    }


def _host_snapshot() -> dict[str, Any]:
    return {
        "processes": _run_host_command(
            [
                "bash",
                "-lc",
                'pgrep -af "run_registry|run_registry_batch|eiel_|_runner/run_registry_batch.py|tools/scripts" | head -40',
            ]
        ),
        "gpu": _run_host_command(
            [
                "bash",
                "-lc",
                "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true",
            ]
        ),
        "uptime": _run_host_command(["bash", "-lc", "date && uptime"]),
    }


def _summarize_run(
    run_dir: Path,
    summary_path: Path | None,
    decomposition_path: Path | None,
    phenotype_path: Path | None,
) -> dict[str, Any]:
    decomposition = _read_json(decomposition_path)
    phenotype = _read_json(phenotype_path)
    return {
        "run_dir": str(run_dir),
        "metrics": _final_metrics(run_dir),
        "checkpoints": _checkpoint_summary(run_dir),
        "manifest_present": bool((run_dir / "manifest.json").exists()),
        "run_config_present": bool((run_dir / "run_config.csv").exists() or (run_dir / "config.json").exists()),
        "summary_path": str(summary_path) if summary_path is not None else "",
        "decomposition_path": str(decomposition_path) if decomposition_path is not None else "",
        "phenotype_decomposition_path": str(phenotype_path) if phenotype_path is not None else "",
        "decomposition_seconds": decomposition.get("seconds", {}) if isinstance(decomposition, dict) else {},
        "decomposition_counts": decomposition.get("counts", {}) if isinstance(decomposition, dict) else {},
        "direct_cost_scalar": decomposition.get("direct_cost_scalar", {}) if isinstance(decomposition, dict) else {},
        "native_cognitive_budget_cost_aggregation": (
            decomposition.get("native_cognitive_budget_cost_aggregation", {}) if isinstance(decomposition, dict) else {}
        ),
        "phenotype_seconds": phenotype.get("seconds", {}) if isinstance(phenotype, dict) else {},
        "phenotype_counts": phenotype.get("counts", {}) if isinstance(phenotype, dict) else {},
        "phenotype_inactive_paths": phenotype.get("inactive_paths", {}) if isinstance(phenotype, dict) else {},
        "phenotype_inactive_default_strategy": (
            phenotype.get("inactive_default_strategy", {}) if isinstance(phenotype, dict) else {}
        ),
        "phenotype_derived": phenotype.get("derived", {}) if isinstance(phenotype, dict) else {},
    }


def _run_one(*, label: str, runs_root: Path, batch_prefix: str, knobs: list[str]) -> dict[str, Any]:
    registry, run_id = PROFILES[label]
    label_root = runs_root / label.lower()
    cmd = [
        "python",
        "_runner/run_registry_batch.py",
        "--registry",
        registry,
        "--runs_root",
        str(label_root),
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
    summary_path = _latest_named_file(label_root, "env_hotpath_expression_summary.json", started_wall)
    decomposition_path = _latest_named_file(label_root, "env_apply_decomposition_summary.json", started_wall)
    phenotype_path = _latest_named_file(label_root, "envhot3_phenotype_decomposition_summary.json", started_wall)
    run_dir = decomposition_path.parent.parent if decomposition_path is not None else Path("")
    return {
        "label": str(label),
        "registry": str(registry),
        "run_id": str(run_id),
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "wall_s": float(wall_s),
        "run": _summarize_run(run_dir, summary_path, decomposition_path, phenotype_path)
        if decomposition_path is not None
        else {},
        "stdout_tail": cp.stdout[-4000:],
        "stderr_tail": cp.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PHENOSTRUCT-1 phenotype expression residual profile.")
    parser.add_argument("--runs-root", required=True, help="Runtime output root")
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--label", action="append", default=[], help=f"Known labels: {', '.join(sorted(PROFILES))}")
    parser.add_argument("--batch-prefix", default="phenostruct1_expression", help="Batch tag prefix")
    parser.add_argument(
        "--knob",
        action="append",
        default=[
            "sim_cfg.generations=10",
            "sim_cfg.steps_per_generation=80",
            "env_cfg.width=16",
            "env_cfg.height=16",
            "pop_cfg.population_size=64",
            "sim_cfg.env_hotpath_expression_diagnostics_enable=true",
            "sim_cfg.env_hotpath_expression_diagnostics_schema_version=1",
            "sim_cfg.env_hotpath_expression_diagnostics_write_events=false",
            "sim_cfg.env_hotpath_apply_decomposition_enable=true",
            "sim_cfg.env_hotpath_apply_decomposition_schema_version=1",
            "sim_cfg.env_hotpath_phenotype_decomposition_enable=true",
            "sim_cfg.env_hotpath_phenotype_decomposition_schema_version=1",
            "sim_cfg.env_hotpath_inactive_phenotype_defaults_enable=true",
            "sim_cfg.native_cognitive_budget_cost_aggregation_enable=false",
            "sim_cfg.env_hotpath_direct_cost_scalar_enable=false",
            "sim_cfg.env_hotpath_direct_cost_scalar_mismatch_guard=false",
        ],
        help="Knob override passed to every child run.",
    )
    parser.add_argument(
        "--timing-note",
        default="Another W6 campaign may be co-resident; timing is interpreted as contaminated unless host snapshots are idle.",
    )
    args = parser.parse_args()

    labels = [str(label).strip() for label in (args.label or []) if str(label).strip()] or ["A101", "B101"]
    unknown = sorted(set(labels) - set(PROFILES))
    if unknown:
        raise SystemExit(f"unknown labels: {', '.join(unknown)}")

    runs_root = Path(str(args.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before_host = _host_snapshot()
    results = [
        _run_one(label=label, runs_root=runs_root, batch_prefix=str(args.batch_prefix), knobs=[str(k) for k in args.knob])
        for label in labels
    ]
    after_host = _host_snapshot()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runs_root": str(runs_root),
        "labels": labels,
        "total_wall_s": float(time.perf_counter() - started),
        "timing_note": str(args.timing_note),
        "host_snapshot_before": before_host,
        "host_snapshot_after": after_host,
        "results": results,
        "failures": [
            row
            for row in results
            if int(row.get("returncode", 1)) != 0
            or not row.get("run", {}).get("summary_path")
            or not row.get("run", {}).get("decomposition_path")
            or not row.get("run", {}).get("phenotype_decomposition_path")
        ],
    }
    out = Path(str(args.out)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "total_wall_s": payload["total_wall_s"], "failures": len(payload["failures"])}, sort_keys=True))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
