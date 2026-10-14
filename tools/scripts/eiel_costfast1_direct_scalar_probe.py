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


SCHEMA_VERSION = "eiel_costfast1_direct_scalar_probe_v1"
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

ARMS: dict[str, dict[str, Any]] = {
    "baseline": {
        "direct_enable": False,
        "guard": False,
        "guard_mode": "sample",
    },
    "optimized_guard_all": {
        "direct_enable": True,
        "guard": True,
        "guard_mode": "all",
    },
    "optimized_fast": {
        "direct_enable": True,
        "guard": False,
        "guard_mode": "sample",
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
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


def _summarize_run(run_dir: Path, summary_path: Path | None, decomposition_path: Path | None, phenotype_path: Path | None) -> dict[str, Any]:
    decomposition = _read_json(decomposition_path) if decomposition_path is not None else {}
    phenotype = _read_json(phenotype_path) if phenotype_path is not None else {}
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
        "phenotype_inactive_default_strategy": (
            phenotype.get("inactive_default_strategy", {}) if isinstance(phenotype, dict) else {}
        ),
    }


def _arm_knobs(common: list[str], arm: str) -> list[str]:
    spec = ARMS[str(arm)]
    out = list(common)
    out.extend(
        [
            "sim_cfg.native_cognitive_budget_cost_aggregation_enable=false",
            f"sim_cfg.env_hotpath_direct_cost_scalar_enable={'true' if spec['direct_enable'] else 'false'}",
            f"sim_cfg.env_hotpath_direct_cost_scalar_mismatch_guard={'true' if spec['guard'] else 'false'}",
            f"sim_cfg.env_hotpath_direct_cost_scalar_guard_mode={spec['guard_mode']}",
            "sim_cfg.env_hotpath_direct_cost_scalar_guard_sample_period=1",
        ]
    )
    return out


def run_one(*, label: str, arm: str, runs_root: Path, batch_prefix: str, knobs: list[str]) -> dict[str, Any]:
    registry, run_id = PROFILES[label]
    arm_root = runs_root / arm
    cmd = [
        "python",
        "_runner/run_registry_batch.py",
        "--registry",
        registry,
        "--runs_root",
        str(arm_root),
        "--batch_tag",
        f"{batch_prefix}_{arm}_{label.lower()}",
        "--overwrite",
        "--only_run_id",
        run_id,
        "--perf",
    ]
    for knob in _arm_knobs(knobs, arm):
        cmd.extend(["--knob", str(knob)])

    started_wall = time.time()
    started = time.perf_counter()
    cp = subprocess.run(cmd, cwd=_project_root(), text=True, capture_output=True, check=False)
    wall_s = time.perf_counter() - started
    summary_path = _latest_named_file(arm_root, "env_hotpath_expression_summary.json", started_wall)
    decomposition_path = _latest_named_file(arm_root, "env_apply_decomposition_summary.json", started_wall)
    phenotype_path = _latest_named_file(arm_root, "envhot3_phenotype_decomposition_summary.json", started_wall)
    run_dir = decomposition_path.parent.parent if decomposition_path is not None else Path("")
    return {
        "label": str(label),
        "arm": str(arm),
        "registry": str(registry),
        "run_id": str(run_id),
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "wall_s": float(wall_s),
        "run": _summarize_run(run_dir, summary_path, decomposition_path, phenotype_path) if decomposition_path is not None else {},
        "stdout_tail": cp.stdout[-4000:],
        "stderr_tail": cp.stderr[-4000:],
    }


def _compare_pair(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    b_run = baseline.get("run", {})
    o_run = optimized.get("run", {})
    b_metrics = b_run.get("metrics", {})
    o_metrics = o_run.get("metrics", {})
    b_final = b_metrics.get("final", {})
    o_final = o_metrics.get("final", {})
    metric_diffs = {
        field: {"baseline": b_final.get(field, ""), "optimized": o_final.get(field, "")}
        for field in FINAL_METRIC_FIELDS
        if str(b_final.get(field, "")) != str(o_final.get(field, ""))
    }
    b_checkpoints = b_run.get("checkpoints", {})
    o_checkpoints = o_run.get("checkpoints", {})
    checkpoint_diffs = {}
    for field in ("bundle_count", "metadata_digest", "index_present"):
        if b_checkpoints.get(field) != o_checkpoints.get(field):
            checkpoint_diffs[field] = {"baseline": b_checkpoints.get(field), "optimized": o_checkpoints.get(field)}
    b_wall = float(baseline.get("wall_s", 0.0) or 0.0)
    o_wall = float(optimized.get("wall_s", 0.0) or 0.0)
    speedup = float(b_wall / o_wall) if o_wall > 0.0 else 0.0
    return {
        "label": str(baseline.get("label", "")),
        "optimized_arm": str(optimized.get("arm", "")),
        "parity_pass": not metric_diffs and not checkpoint_diffs and b_metrics.get("row_count") == o_metrics.get("row_count"),
        "metric_row_counts": {"baseline": b_metrics.get("row_count"), "optimized": o_metrics.get("row_count")},
        "metric_diffs": metric_diffs,
        "checkpoint_diffs": checkpoint_diffs,
        "wall_s": {"baseline": b_wall, "optimized": o_wall, "speedup_ratio": speedup},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run COSTFAST-1 direct scalar baseline/optimized probes.")
    parser.add_argument("--runs-root", required=True, help="Runtime output root")
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--label", action="append", default=[], help=f"Known labels: {', '.join(sorted(PROFILES))}")
    parser.add_argument("--batch-prefix", default="costfast1_direct_scalar", help="Batch tag prefix")
    parser.add_argument(
        "--arm",
        action="append",
        choices=sorted(ARMS),
        default=[],
        help="Arm to run. Defaults to baseline, optimized_guard_all, optimized_fast.",
    )
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
        ],
        help="Knob override passed to every child run.",
    )
    parser.add_argument(
        "--timing-note",
        default="Another W6 campaign may be co-resident; timing is interpreted with that caveat.",
    )
    args = parser.parse_args()

    labels = [str(label).strip() for label in (args.label or []) if str(label).strip()] or ["A101", "B101"]
    unknown = sorted(set(labels) - set(PROFILES))
    if unknown:
        raise SystemExit(f"unknown labels: {', '.join(unknown)}")
    arms = [str(arm) for arm in (args.arm or [])] or ["baseline", "optimized_guard_all", "optimized_fast"]
    if "baseline" not in arms:
        raise SystemExit("baseline arm is required for pair comparison")

    runs_root = Path(str(args.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = []
    for label in labels:
        for arm in arms:
            results.append(run_one(label=label, arm=arm, runs_root=runs_root, batch_prefix=str(args.batch_prefix), knobs=[str(k) for k in args.knob]))

    pairs = []
    for label in labels:
        baseline = next(row for row in results if row["label"] == label and row["arm"] == "baseline")
        for arm in arms:
            if arm == "baseline":
                continue
            optimized = next(row for row in results if row["label"] == label and row["arm"] == arm)
            pairs.append(_compare_pair(baseline, optimized))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runs_root": str(runs_root),
        "labels": labels,
        "arms": arms,
        "total_wall_s": float(time.perf_counter() - started),
        "timing_note": str(args.timing_note),
        "results": results,
        "pairs": pairs,
        "failures": [
            row
            for row in results
            if int(row.get("returncode", 1)) != 0
            or not row.get("run", {}).get("summary_path")
            or not row.get("run", {}).get("decomposition_path")
        ],
    }
    out = Path(str(args.out)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "total_wall_s": payload["total_wall_s"], "failures": len(payload["failures"])}, sort_keys=True))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
