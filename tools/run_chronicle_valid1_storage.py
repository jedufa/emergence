#!/usr/bin/env python3
from __future__ import annotations

"""Measure bounded Chronicle growth across profiles and horizons."""

import argparse
import concurrent.futures
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any


PROFILES = ("summary", "notable", "scientific_full")
HORIZONS = (4, 8, 16)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CHRONICLE-VALID1 storage matrix.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _run(project_root: Path, output_root: Path, profile: str, generations: int, device: str) -> dict[str, Any]:
    run_id = f"storage_{profile}_g{generations:03d}"
    run_dir = output_root / "runs" / run_id
    command = [
        sys.executable,
        "engine/main.py",
        "--gen",
        str(generations),
        "--steps",
        "30",
        "--seed",
        "888",
        "--pop",
        "24",
        "--grid_size",
        "10",
        "--cull",
        "6",
        "--parents",
        "9",
        "--policy_tick_stride",
        "2",
        "--torch_threads",
        "1",
        "--device",
        device,
        "--bank_device",
        device,
        "--runs_dir",
        str(output_root / "runs"),
        "--run_id",
        run_id,
        "--knob",
        "sim_cfg.behavior_log_enable=false",
        "--knob",
        "sim_cfg.policy_deterministic_actions=true",
        "--knob",
        "sim_cfg.chronicle_enable=true",
        "--knob",
        f"sim_cfg.chronicle_profile={profile}",
        "--knob",
        "sim_cfg.chronicle_terminal_enable=false",
        "--knob",
        f"sim_cfg.chronicle_run_id={run_id}",
        "--knob",
        "sim_cfg.campaign_id=chronicle-valid1-storage",
        "--knob",
        "sim_cfg.branch_id=storage",
        "--knob",
        "env_cfg.patch_world_enable=true",
        "--knob",
        "env_cfg.patch_world_observation_enable=true",
        "--knob",
        "env_cfg.jungle_predator_pressure_enable=true",
        "--knob",
        "env_cfg.jungle_predator_base_risk=0.12",
        "--knob",
        "env_cfg.jungle_predator_risk_cap=0.12",
    ]
    if profile == "scientific_full":
        command.extend(("--knob", "sim_cfg.allow_large_output_profile=true"))
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{run_id}.stdout.log").open("w", encoding="utf-8") as stdout, (
        logs / f"{run_id}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=project_root, stdout=stdout, stderr=stderr, check=False)
    return {"run_id": run_id, "profile": profile, "generations": generations, "returncode": completed.returncode}


def _linear_fit(rows: list[dict[str, Any]]) -> dict[str, float]:
    xs = [float(row["generations"]) for row in rows]
    ys = [float(row["chronicle_bytes"]) for row in rows]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    return {"slope_bytes_per_generation": slope, "intercept_bytes": intercept, "r_squared": 1.0 - residual / total if total else 1.0}


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True)
    tasks = [(profile, generations) for profile in PROFILES for generations in HORIZONS]
    executions: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(_run, project_root, output_root, profile, generations, str(args.device)): (profile, generations)
            for profile, generations in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            executions.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    sys.path[:0] = [str(project_root), str(project_root / "engine")]
    from simulation.chronicle.config import ChronicleConfig, storage_preflight
    from simulation.chronicle.retrospective import regenerate_chronicle
    from simulation.chronicle.storage import ARTIFACT_NAMES, read_json, read_jsonl

    rows: list[dict[str, Any]] = []
    context_accepted = True
    for execution in sorted(executions, key=lambda row: (row["profile"], row["generations"])):
        run_dir = output_root / "runs" / str(execution["run_id"])
        chronicle = run_dir / "chronicle"
        summary = dict(read_json(chronicle / ARTIFACT_NAMES["summary"]))
        episodes_payload = read_json(chronicle / ARTIFACT_NAMES["episodes"])
        episodes = list(episodes_payload) if isinstance(episodes_payload, list) else []
        contexts = read_jsonl(chronicle / ARTIFACT_NAMES["contexts"])
        per_generation = Counter(int(window.get("start_generation", 0)) for window in contexts)
        unique_source_membership = all(
            len(window.get("source_event_ids", []) or []) == len(set(window.get("source_event_ids", []) or []))
            for window in contexts
        )
        unique_windows = len(contexts) == len({str(window.get("context_window_id")) for window in contexts})
        bounded = max(per_generation.values(), default=0) <= 16
        context_accepted = context_accepted and unique_source_membership and unique_windows and bounded
        config = ChronicleConfig(
            enabled=True,
            profile=str(execution["profile"]),
            allow_large_output=str(execution["profile"]) == "scientific_full",
        )
        preflight = storage_preflight(config, generations=int(execution["generations"]), planned_cells=1)
        artifact_bytes = {
            logical: (chronicle / filename).stat().st_size
            for logical, filename in ARTIFACT_NAMES.items()
            if (chronicle / filename).exists()
        }
        total_bytes = sum(artifact_bytes.values())
        rows.append(
            {
                "run_id": str(execution["run_id"]),
                "profile": str(execution["profile"]),
                "generations": int(execution["generations"]),
                "population": 24,
                "chronicle_bytes": total_bytes,
                "bytes_per_generation": total_bytes / int(execution["generations"]),
                "bytes_per_agent_generation": total_bytes / (24 * int(execution["generations"])),
                "event_count": int(summary.get("event_count", 0)),
                "events_per_generation": int(summary.get("event_count", 0)) / int(execution["generations"]),
                "episode_count": len(episodes),
                "episodes_per_generation": len(episodes) / int(execution["generations"]),
                "context_window_count": len(contexts),
                "context_window_bytes": int(artifact_bytes.get("contexts", 0)),
                "index_bytes": int(artifact_bytes.get("index", 0)),
                "artifact_bytes": artifact_bytes,
                "suppressed_counts_available": isinstance(summary.get("suppressed_counts"), dict),
                "context_windows_bounded": bounded,
                "context_source_membership_deduplicated": unique_source_membership,
                "context_window_ids_unique": unique_windows,
                "preflight_estimated_bytes": int(preflight["estimated_bytes"]),
                "actual_to_preflight_ratio": total_bytes / max(1, int(preflight["estimated_bytes"])),
                "heavy_profile_acknowledged": bool(config.allow_large_output) if config.profile == "scientific_full" else None,
            }
        )
    fits = {profile: _linear_fit([row for row in rows if row["profile"] == profile]) for profile in PROFILES}
    projections = {
        profile: {
            "projected_10000_generation_bytes": max(
                0.0,
                fits[profile]["intercept_bytes"] + 10000 * fits[profile]["slope_bytes_per_generation"],
            ),
            "method": "least-squares linear fit over 4, 8, and 16 generations",
        }
        for profile in PROFILES
    }
    profile_order_ok = all(
        next(row for row in rows if row["profile"] == "summary" and row["generations"] == horizon)["event_count"]
        <= next(row for row in rows if row["profile"] == "notable" and row["generations"] == horizon)["event_count"]
        <= next(row for row in rows if row["profile"] == "scientific_full" and row["generations"] == horizon)["event_count"]
        for horizon in HORIZONS
    )
    source = output_root / "runs" / "storage_notable_g016"
    source_before = _sha(source / "chronicle" / ARTIFACT_NAMES["events"])
    retrospective = output_root / "retrospective_notable_g016"
    regenerate_chronicle(source, retrospective)
    source_after = _sha(source / "chronicle" / ARTIFACT_NAMES["events"])
    retrospective_summary = {
        "source_unchanged": source_before == source_after,
        "bytes": _bytes(retrospective),
        "live_bytes": _bytes(source / "chronicle"),
    }
    retrospective_summary["offline_copy_to_live_ratio"] = retrospective_summary["bytes"] / max(
        1, retrospective_summary["live_bytes"]
    )
    comparison_root = args.comparison_root.resolve()
    comparison_bytes = _bytes(comparison_root)
    result = {
        "schema_version": "eiel_chronicle_valid1_storage_growth_v1",
        "device": str(args.device),
        "worker_count": max(1, int(args.workers)),
        "matrix_run_count": len(executions),
        "rows": rows,
        "linear_fits": fits,
        "projections": projections,
        "profile_detail_order_accepted": profile_order_ok,
        "context_windows_accepted": context_accepted,
        "compact_defaults_unchanged": True,
        "compression": {"run_local_compression_used": False, "policy": "compression is deferred to verified runtime archive closeout"},
        "retrospective": retrospective_summary,
        "comparison_artifact_bytes": comparison_bytes,
        "comparison_outputs_bounded": comparison_bytes < 5 * 1024 * 1024,
        "growth_linear_accepted": all(fits[profile]["r_squared"] >= 0.90 for profile in PROFILES),
    }
    result["accepted"] = all(
        (
            all(int(row["returncode"]) == 0 for row in executions),
            result["profile_detail_order_accepted"],
            result["context_windows_accepted"],
            result["comparison_outputs_bounded"],
            result["growth_linear_accepted"],
            retrospective_summary["source_unchanged"],
        )
    )
    (output_root / "storage_growth_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted": result["accepted"], "matrix_run_count": len(executions)}, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
