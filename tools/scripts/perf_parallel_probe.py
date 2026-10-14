#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROFILES: dict[str, tuple[str, str]] = {
    "A101": (
        "v1/phaseIV_crb_x1_prediction_readiness_bridge.yaml",
        "phaseIV_crb_x1_prediction_readiness_bridge__ecology_crbx1_theme12_clean_pressure__init_init_founder_balanced__prediction_prediction_normal__seed000101",
    ),
    "A202": (
        "v1/phaseIV_crb_x1_prediction_readiness_bridge.yaml",
        "phaseIV_crb_x1_prediction_readiness_bridge__ecology_crbx1_theme12_clean_pressure__init_init_founder_balanced__prediction_prediction_normal__seed000202",
    ),
    "B101": (
        "v1/lbc1_lineage_bottleneck_closure_source.yaml",
        "lbc1_lineage_bottleneck_closure_source__scenario_lbc1_00_tuned_full_jungle_recovery__seed000101",
    ),
    "B202": (
        "v1/lbc1_lineage_bottleneck_closure_source.yaml",
        "lbc1_lineage_bottleneck_closure_source__scenario_lbc1_00_tuned_full_jungle_recovery__seed000202",
    ),
    "C101": (
        "v1/phaseII_b1_founder_balance_validation.yaml",
        "phaseII_b1_founder_balance_validation__scenario_b1v_01_tuned_full_jungle_recovery_founder_balanced__seed000101",
    ),
    "C202": (
        "v1/phaseII_b1_founder_balance_validation.yaml",
        "phaseII_b1_founder_balance_validation__scenario_b1v_01_tuned_full_jungle_recovery_founder_balanced__seed000202",
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _capture_runtime_snapshot() -> dict[str, Any]:
    commands = {
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        "uptime": ["uptime"],
        "free": ["free", "-h"],
    }
    snapshot: dict[str, Any] = {}
    for name, cmd in commands.items():
        try:
            cp = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
            snapshot[name] = {
                "returncode": int(cp.returncode),
                "stdout": cp.stdout.strip(),
                "stderr": cp.stderr.strip(),
            }
        except Exception as exc:  # pragma: no cover - host dependent
            snapshot[name] = {"error": str(exc)}
    return snapshot


def _run_one(label: str, *, runs_root: Path, batch_prefix: str, knobs: list[str]) -> dict[str, Any]:
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
        cmd.extend(["--knob", knob])
    started = time.perf_counter()
    cp = subprocess.run(cmd, cwd=_project_root(), text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "registry": registry,
        "run_id": run_id,
        "cmd": cmd,
        "returncode": int(cp.returncode),
        "wall_s": elapsed,
        "stdout_tail": cp.stdout[-4000:],
        "stderr_tail": cp.stderr[-4000:],
    }


def run_probe(
    *,
    labels: list[str],
    workers: int,
    runs_root: Path,
    batch_prefix: str,
    knobs: list[str],
    capture_snapshot: bool = False,
    baseline_wall_s: float | None = None,
) -> dict[str, Any]:
    snapshot_before = _capture_runtime_snapshot() if bool(capture_snapshot) else None
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_run_one, label, runs_root=runs_root, batch_prefix=batch_prefix, knobs=knobs) for label in labels]
        for future in as_completed(futures):
            results.append(future.result())
    total_wall = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema_version": "eiel_perf_parallel_probe_v1",
        "workers": int(workers),
        "labels": labels,
        "runs_root": str(runs_root),
        "batch_prefix": batch_prefix,
        "total_campaign_wall_s": total_wall,
        "results": sorted(results, key=lambda row: str(row.get("label"))),
        "failures": [r for r in results if int(r.get("returncode", 1)) != 0],
    }
    if baseline_wall_s is not None and float(total_wall) > 0.0:
        payload["baseline_campaign_wall_s"] = float(baseline_wall_s)
        payload["throughput_ratio_vs_baseline"] = float(baseline_wall_s) / float(total_wall)
    if bool(capture_snapshot):
        payload["runtime_snapshot_before"] = snapshot_before
        payload["runtime_snapshot_after"] = _capture_runtime_snapshot()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run independent EIEL perf cells concurrently")
    parser.add_argument("--runs-root", required=True, help="Runtime output root")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent worker count")
    parser.add_argument("--label", action="append", default=[], help=f"Profile label. Known: {', '.join(sorted(PROFILES))}")
    parser.add_argument("--labels", default="", help="Comma-separated profile labels; merged with repeated --label")
    parser.add_argument("--batch-prefix", default="perf_parallel_probe", help="Batch tag prefix")
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--baseline-json", default="", help="Optional prior probe JSON used to compute throughput ratio")
    parser.add_argument("--capture-snapshot", action="store_true", help="Capture lightweight GPU/host snapshots before and after")
    parser.add_argument(
        "--knob",
        action="append",
        default=[
            "sim_cfg.generations=10",
            "sim_cfg.steps_per_generation=80",
            "sim_cfg.enable_profiler=false",
        ],
        help="Knob override passed to every child run",
    )
    args = parser.parse_args()
    labels = [str(x).strip() for x in (args.label or []) if str(x).strip()]
    labels.extend([part.strip() for part in str(args.labels or "").split(",") if part.strip()])
    labels = labels or ["A101", "A202", "B101", "B202"]
    unknown = sorted(set(labels) - set(PROFILES))
    if unknown:
        raise SystemExit(f"unknown labels: {', '.join(unknown)}")
    baseline_wall_s = None
    if str(args.baseline_json or "").strip():
        baseline_payload = json.loads(Path(str(args.baseline_json)).read_text(encoding="utf-8"))
        baseline_wall_s = float(baseline_payload["total_campaign_wall_s"])
    payload = run_probe(
        labels=labels,
        workers=max(1, int(args.workers)),
        runs_root=Path(args.runs_root),
        batch_prefix=str(args.batch_prefix),
        knobs=[str(x) for x in args.knob],
        capture_snapshot=bool(args.capture_snapshot),
        baseline_wall_s=baseline_wall_s,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "total_campaign_wall_s": payload["total_campaign_wall_s"], "failures": len(payload["failures"])}, sort_keys=True))
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
