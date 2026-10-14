#!/usr/bin/env python3
from __future__ import annotations

"""Run the bounded CHRONICLE-VALID1 rehearsal or main validation matrix."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SCENARIOS: dict[str, dict[str, Any]] = {
    "quiet_control": {
        "description": "Patch, Jungle, WDL, and stochastic shock surfaces disabled; expected ecology/hazard quiet control.",
        "knobs": [
            "env_cfg.patch_world_enable=false",
            "sim_cfg.shock_start_prob=0.0",
        ],
        "expected_positive_families": [],
        "negative_control_families": [
            "regional_food_depletion",
            "crowding_waste_crisis",
            "predator_pressure_wave",
            "ecological_recovery",
            "population_bottleneck",
            "delayed_hazard_cycle",
        ],
    },
    "flat_control": {
        "description": "Normal flat runner control with no Patch/Jungle/WDL substrate.",
        "knobs": ["env_cfg.patch_world_enable=false"],
        "expected_positive_families": [],
        "negative_control_families": [
            "regional_food_depletion",
            "crowding_waste_crisis",
            "predator_pressure_wave",
            "ecological_recovery",
            "delayed_hazard_cycle",
        ],
    },
    "patch_ecology": {
        "description": "Patch-only resource depletion, occupancy, waste, and regrowth diagnostics.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
        ],
        "expected_positive_families": ["regional_food_depletion", "crowding_waste_crisis"],
        "negative_control_families": ["predator_pressure_wave", "delayed_hazard_cycle"],
    },
    "crowding_waste": {
        "description": "Patch crowding and waste pressure with direct regrowth suppression enabled.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.patch_world_waste_consequence_enable=true",
            "env_cfg.patch_waste_presence_emit=0.05",
            "env_cfg.patch_waste_consume_emit=0.15",
            "env_cfg.patch_waste_density_emit_scale=0.02",
            "env_cfg.patch_waste_regrowth_penalty_enable=true",
            "env_cfg.patch_density_regrowth_penalty_enable=true",
        ],
        "expected_positive_families": ["crowding_waste_crisis"],
        "negative_control_families": ["predator_pressure_wave", "delayed_hazard_cycle"],
    },
    "jungle_predator": {
        "description": "Full Patch/Jungle predator pressure at a bounded non-catastrophic risk.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.jungle_predator_pressure_enable=true",
            "env_cfg.jungle_predator_base_risk=0.18",
            "env_cfg.jungle_predator_risk_cap=0.18",
            "env_cfg.jungle_predator_health_damage=0.12",
        ],
        "expected_positive_families": ["predator_pressure_wave"],
        "negative_control_families": ["delayed_hazard_cycle"],
    },
    "ecological_recovery": {
        "description": "Patch waste consequence plus the accepted Jungle ecological-recovery substrate.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.patch_world_waste_consequence_enable=true",
            "env_cfg.jungle_ecological_recovery_enable=true",
            "env_cfg.jungle_recovery_activity_initial=0.5",
            "env_cfg.jungle_recovery_activity_base_growth=0.02",
        ],
        "expected_positive_families": ["ecological_recovery", "crowding_waste_crisis"],
        "negative_control_families": ["predator_pressure_wave", "delayed_hazard_cycle"],
    },
    "delayed_hazard_wdl": {
        "description": "Patch World with the accepted delayed-hazard demand-loop substrate.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.wdl_delayed_hazard_enable=true",
            "env_cfg.wdl_delayed_hazard_observation_enable=true",
            "env_cfg.wdl_delayed_hazard_delay_steps=2",
            "env_cfg.wdl_delayed_hazard_cue_duration_steps=1",
            "env_cfg.wdl_delayed_hazard_active_duration_steps=3",
            "env_cfg.wdl_delayed_hazard_event_period_steps=8",
            "env_cfg.wdl_delayed_hazard_radius=2",
        ],
        "expected_positive_families": ["delayed_hazard_cycle"],
        "negative_control_families": ["predator_pressure_wave"],
    },
    "lineage_sensitive": {
        "description": "Four deterministic founder lineages for lineage transition and identity review.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "sim_cfg.lineage_founder_balance_enable=true",
            "sim_cfg.lineage_founder_group_count=4",
            "sim_cfg.lineage_founder_min_agents_per_group=4",
            "sim_cfg.lineage_founder_assignment_mode=deterministic_round_robin",
        ],
        "expected_positive_families": ["lineage_expansion", "lineage_decline", "lineage_extinction_sequence"],
        "negative_control_families": ["predator_pressure_wave", "delayed_hazard_cycle"],
    },
    "bottleneck_pressure": {
        "description": "High but bounded Jungle pressure intended to exercise cohort-survival bottleneck handling.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.jungle_predator_pressure_enable=true",
            "env_cfg.jungle_predator_base_risk=0.55",
            "env_cfg.jungle_predator_risk_cap=0.55",
            "env_cfg.jungle_predator_health_damage=0.5",
        ],
        "expected_positive_families": ["predator_pressure_wave", "population_bottleneck"],
        "negative_control_families": ["delayed_hazard_cycle"],
    },
    "multi_episode": {
        "description": "Combined Patch waste, Jungle pressure, recovery, WDL, and founder-sensitive context.",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.patch_world_waste_consequence_enable=true",
            "env_cfg.patch_waste_presence_emit=0.04",
            "env_cfg.patch_waste_consume_emit=0.12",
            "env_cfg.jungle_predator_pressure_enable=true",
            "env_cfg.jungle_predator_base_risk=0.12",
            "env_cfg.jungle_predator_risk_cap=0.12",
            "env_cfg.jungle_predator_health_damage=0.12",
            "env_cfg.jungle_ecological_recovery_enable=true",
            "env_cfg.jungle_recovery_activity_initial=0.4",
            "env_cfg.jungle_recovery_activity_base_growth=0.02",
            "env_cfg.wdl_delayed_hazard_enable=true",
            "env_cfg.wdl_delayed_hazard_observation_enable=true",
            "env_cfg.wdl_delayed_hazard_delay_steps=2",
            "env_cfg.wdl_delayed_hazard_active_duration_steps=3",
            "env_cfg.wdl_delayed_hazard_event_period_steps=10",
            "sim_cfg.lineage_founder_balance_enable=true",
            "sim_cfg.lineage_founder_group_count=4",
            "sim_cfg.lineage_founder_min_agents_per_group=4",
            "sim_cfg.lineage_founder_assignment_mode=deterministic_round_robin",
        ],
        "expected_positive_families": [
            "regional_food_depletion",
            "crowding_waste_crisis",
            "predator_pressure_wave",
            "ecological_recovery",
            "lineage_expansion",
            "lineage_decline",
            "lineage_extinction_sequence",
            "delayed_hazard_cycle",
        ],
        "negative_control_families": [],
    },
}


PHASES = {
    "rehearsal": {"seeds": [901, 902], "generations": 5, "steps": 30, "population": 24, "grid": 10},
    "main": {"seeds": [101, 202, 303], "generations": 24, "steps": 60, "population": 32, "grid": 12},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CHRONICLE-VALID1 validation cells in parallel.")
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_cell(project_root: Path, runs_root: Path, logs_root: Path, cell: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    run_id = str(cell["run_id"])
    run_dir = runs_root / run_id
    if overwrite:
        shutil.rmtree(run_dir, ignore_errors=True)
    elif (run_dir / "chronicle" / "chronicle_manifest.json").exists():
        return {"run_id": run_id, "returncode": 0, "elapsed_seconds": 0.0, "status": "existing"}
    command = [
        sys.executable,
        "engine/main.py",
        "--gen",
        str(cell["generations"]),
        "--steps",
        str(cell["steps"]),
        "--seed",
        str(cell["seed"]),
        "--pop",
        str(cell["population"]),
        "--grid_size",
        str(cell["grid"]),
        "--cull",
        str(max(4, int(cell["population"]) // 4)),
        "--parents",
        str(max(8, int(cell["population"]) * 3 // 8)),
        "--policy_tick_stride",
        "2",
        "--torch_threads",
        "1",
        "--device",
        str(cell["device"]),
        "--bank_device",
        str(cell["device"]),
        "--runs_dir",
        str(runs_root),
        "--run_id",
        run_id,
    ]
    for knob in cell["knobs"]:
        command.extend(("--knob", str(knob)))
    logs_root.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_root / f"{run_id}.stdout.log"
    stderr_path = logs_root / f"{run_id}.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=project_root, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.perf_counter() - started
    manifest = run_dir / "chronicle" / "chronicle_manifest.json"
    return {
        "run_id": run_id,
        "returncode": int(completed.returncode),
        "elapsed_seconds": elapsed,
        "status": "completed" if completed.returncode == 0 and manifest.exists() else "failed",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    runs_root = output_root / "runs"
    logs_root = output_root / "logs"
    phase = dict(PHASES[args.phase])
    base_knobs = [
        "sim_cfg.behavior_log_enable=false",
        "sim_cfg.policy_deterministic_actions=true",
        "sim_cfg.chronicle_enable=true",
        "sim_cfg.chronicle_profile=notable",
        "sim_cfg.chronicle_terminal_enable=false",
        f"sim_cfg.campaign_id=chronicle-valid1-{args.phase}",
        "sim_cfg.branch_id=default",
    ]
    cells: list[dict[str, Any]] = []
    for scenario, spec in SCENARIOS.items():
        for seed in phase["seeds"]:
            cells.append(
                {
                    "run_id": f"{args.phase}_{scenario}_s{seed}",
                    "phase": args.phase,
                    "scenario": scenario,
                    "description": spec["description"],
                    "seed": int(seed),
                    "generations": int(phase["generations"]),
                    "steps": int(phase["steps"]),
                    "population": int(phase["population"]),
                    "grid": int(phase["grid"]),
                    "device": str(args.device),
                    "chronicle_profile": "notable",
                    "knobs": [*base_knobs, *spec["knobs"]],
                    "expected_positive_families": list(spec["expected_positive_families"]),
                    "negative_control_families": list(spec["negative_control_families"]),
                }
            )
    matrix = {
        "schema_version": "eiel_chronicle_valid1_validation_matrix_v1",
        "phase": args.phase,
        "worker_count": max(1, int(args.workers)),
        "cell_count": len(cells),
        "paired_seeds": list(phase["seeds"]),
        "shape": {
            "generations": int(phase["generations"]),
            "steps_per_generation": int(phase["steps"]),
            "population": int(phase["population"]),
            "grid": int(phase["grid"]),
            "device": str(args.device),
        },
        "cells": cells,
    }
    _write_json(output_root / "validation_matrix.json", matrix)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_by_id = {
            executor.submit(_run_cell, project_root, runs_root, logs_root, cell, bool(args.overwrite)): cell["run_id"]
            for cell in cells
        }
        for future in concurrent.futures.as_completed(future_by_id):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    results.sort(key=lambda row: str(row["run_id"]))
    summary = {
        "schema_version": "eiel_chronicle_valid1_matrix_execution_v1",
        "phase": args.phase,
        "planned_run_count": len(cells),
        "completed_run_count": sum(1 for row in results if row["status"] in {"completed", "existing"}),
        "failed_run_count": sum(1 for row in results if row["status"] == "failed"),
        "wall_elapsed_seconds": time.perf_counter() - started,
        "worker_count": max(1, int(args.workers)),
        "results": results,
    }
    _write_json(output_root / "run_completion_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("phase", "completed_run_count", "failed_run_count", "wall_elapsed_seconds")}, sort_keys=True))
    return 0 if summary["failed_run_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
