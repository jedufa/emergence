#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eiel_costsurf2_validation_probe_v1"
csv.field_size_limit(sys.maxsize)

PRED_REGISTRY = "v1/phaseIV_crb_x1_prediction_readiness_bridge.yaml"
LINEAGE_REGISTRY = "v1/lbc1_lineage_bottleneck_closure_source.yaml"
PRED_RUN_FMT = (
    "phaseIV_crb_x1_prediction_readiness_bridge__"
    "ecology_crbx1_theme12_clean_pressure__init_init_founder_balanced__"
    "prediction_prediction_normal__seed{seed:06d}"
)
LINEAGE_RUN_FMT = (
    "lbc1_lineage_bottleneck_closure_source__"
    "scenario_lbc1_00_tuned_full_jungle_recovery__seed{seed:06d}"
)
PROFILE_SEEDS = (101, 202, 303, 404, 505)
PROFILES: dict[str, tuple[str, str]] = {
    **{f"A{seed}": (PRED_REGISTRY, PRED_RUN_FMT.format(seed=seed)) for seed in PROFILE_SEEDS},
    **{f"B{seed}": (LINEAGE_REGISTRY, LINEAGE_RUN_FMT.format(seed=seed)) for seed in PROFILE_SEEDS},
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
    "default": {"surface": False, "guard": False, "guard_mode": "sample", "envhot4": False},
    "baseline": {"surface": False, "guard": False, "guard_mode": "sample", "envhot4": True},
    "guard_all": {"surface": True, "guard": True, "guard_mode": "all", "envhot4": True},
    "fast": {"surface": True, "guard": False, "guard_mode": "sample", "envhot4": True},
    "surface_only": {"surface": True, "guard": False, "guard_mode": "sample", "envhot4": False},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path | None) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path is not None else {}


def _latest_named_file(root: Path, name: str, started_at: float) -> Path | None:
    matches = []
    for path in root.rglob(name):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                matches.append(path)
        except OSError:
            continue
    return sorted(matches, key=lambda path: path.stat().st_mtime)[-1] if matches else None


def _run_dir_from_output(stdout: str, stderr: str) -> Path | None:
    text = f"{stdout}\n{stderr}"
    matches = re.findall(r"Metrics written to:\s*(/.+?/metrics\.csv)", text)
    if not matches:
        return None
    metrics_path = Path(matches[-1])
    return metrics_path.parent if metrics_path.name == "metrics.csv" else None


def _named_file_in_run_dir(run_dir: Path, name: str) -> Path | None:
    for rel in (
        Path("viz") / name,
        Path(name),
    ):
        path = run_dir / rel
        if path.exists():
            return path
    matches = sorted(run_dir.rglob(name))
    return matches[-1] if matches else None


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


def _run_host_command(cmd: str) -> dict[str, Any]:
    cp = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, check=False)
    return {"cmd": cmd, "returncode": int(cp.returncode), "stdout": cp.stdout[-4000:], "stderr": cp.stderr[-4000:]}


def _host_snapshot() -> dict[str, Any]:
    return {
        "processes": _run_host_command(
            'pgrep -af "run_registry|run_registry_batch|eiel_|_runner/run_registry_batch.py|tools/scripts" | head -40'
        ),
        "gpu": _run_host_command(
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true"
        ),
        "uptime": _run_host_command("date && uptime"),
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
        "cost_charge_surface": decomposition.get("cost_charge_surface", {}) if isinstance(decomposition, dict) else {},
        "phenotype_seconds": phenotype.get("seconds", {}) if isinstance(phenotype, dict) else {},
        "phenotype_inactive_default_strategy": (
            phenotype.get("inactive_default_strategy", {}) if isinstance(phenotype, dict) else {}
        ),
    }


def _arm_knobs(common: list[str], arm: str) -> list[str]:
    spec = ARMS[str(arm)]
    return [
        *common,
        "sim_cfg.native_cognitive_budget_cost_aggregation_enable=false",
        "sim_cfg.env_hotpath_direct_cost_scalar_enable=false",
        "sim_cfg.env_hotpath_direct_cost_scalar_mismatch_guard=false",
        f"sim_cfg.env_hotpath_inactive_phenotype_defaults_enable={'true' if spec['envhot4'] else 'false'}",
        f"sim_cfg.env_hotpath_cost_charge_surface_enable={'true' if spec['surface'] else 'false'}",
        f"sim_cfg.env_hotpath_cost_charge_surface_guard_enable={'true' if spec['guard'] else 'false'}",
        f"sim_cfg.env_hotpath_cost_charge_surface_guard_mode={spec['guard_mode']}",
        "sim_cfg.env_hotpath_cost_charge_surface_guard_sample_period=1",
    ]


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
    run_dir = _run_dir_from_output(cp.stdout, cp.stderr)
    if run_dir is not None:
        summary_path = _named_file_in_run_dir(run_dir, "env_hotpath_expression_summary.json")
        decomposition_path = _named_file_in_run_dir(run_dir, "env_apply_decomposition_summary.json")
        phenotype_path = _named_file_in_run_dir(run_dir, "envhot3_phenotype_decomposition_summary.json")
    else:
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
        "run": _summarize_run(run_dir, summary_path, decomposition_path, phenotype_path)
        if decomposition_path is not None
        else {},
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
    return {
        "label": str(baseline.get("label", "")),
        "optimized_arm": str(optimized.get("arm", "")),
        "parity_pass": not metric_diffs and not checkpoint_diffs and b_metrics.get("row_count") == o_metrics.get("row_count"),
        "metric_row_counts": {"baseline": b_metrics.get("row_count"), "optimized": o_metrics.get("row_count")},
        "metric_diffs": metric_diffs,
        "checkpoint_diffs": checkpoint_diffs,
        "wall_s": {"baseline": b_wall, "optimized": o_wall, "speedup_ratio": float(b_wall / o_wall) if o_wall > 0.0 else 0.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run COSTSURF-2 validation probes.")
    parser.add_argument("--runs-root", required=True, help="Runtime output root")
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--label", action="append", default=[], help=f"Known labels: {', '.join(sorted(PROFILES))}")
    parser.add_argument("--labels", default="", help="Comma-separated labels")
    parser.add_argument("--arm", action="append", choices=sorted(ARMS), default=[], help="Arm to run")
    parser.add_argument("--baseline-arm", choices=sorted(ARMS), default="baseline", help="Reference arm for pair comparison")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent worker count")
    parser.add_argument("--batch-prefix", default="costsurf2_validation", help="Batch tag prefix")
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
        ],
        help="Knob override passed to every child run.",
    )
    parser.add_argument("--timing-note", default="Timing is clean only if host snapshots show no co-resident performance run.")
    args = parser.parse_args()

    labels = [str(label).strip() for label in (args.label or []) if str(label).strip()]
    labels.extend([part.strip() for part in str(args.labels or "").split(",") if part.strip()])
    labels = labels or ["A101", "B101"]
    unknown = sorted(set(labels) - set(PROFILES))
    if unknown:
        raise SystemExit(f"unknown labels: {', '.join(unknown)}")
    arms = [str(arm) for arm in (args.arm or [])] or ["baseline", "guard_all", "fast"]
    baseline_arm = str(args.baseline_arm)
    if baseline_arm not in arms:
        raise SystemExit(f"{baseline_arm} arm is required for pair comparison")

    runs_root = Path(str(args.runs_root)).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before_host = _host_snapshot()
    jobs = [(label, arm) for label in labels for arm in arms]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                run_one,
                label=label,
                arm=arm,
                runs_root=runs_root,
                batch_prefix=str(args.batch_prefix),
                knobs=[str(k) for k in args.knob],
            )
            for label, arm in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results = sorted(results, key=lambda row: (str(row.get("label")), str(row.get("arm"))))
    after_host = _host_snapshot()

    pairs = []
    for label in labels:
        baseline = next(row for row in results if row["label"] == label and row["arm"] == baseline_arm)
        for arm in arms:
            if arm == baseline_arm:
                continue
            optimized = next(row for row in results if row["label"] == label and row["arm"] == arm)
            pairs.append(_compare_pair(baseline, optimized))

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "runs_root": str(runs_root),
        "labels": labels,
        "arms": arms,
        "baseline_arm": baseline_arm,
        "workers": int(args.workers),
        "total_wall_s": float(time.perf_counter() - started),
        "timing_note": str(args.timing_note),
        "host_snapshot_before": before_host,
        "host_snapshot_after": after_host,
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
