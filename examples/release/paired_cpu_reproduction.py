"""Materialize the bounded CPU usage pair; execute only through ``eiel execute``.

Engineering demonstration, not Full Evolution, scientific replication, or
Qualification. The two treatments are one matched seed block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from engine.campaigns.artifact_lifecycle import lifecycle_profile_contract_v1
from engine.campaigns.compiler import compile_campaign
from engine.campaigns.execution_front_door_v1 import ExecutionRequestV2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    request_dir = args.request_dir.resolve()
    run_root = args.run_root.resolve()
    if request_dir.exists() or run_root.exists():
        parser.error("request-dir and run-root must both be fresh")
    spec = {
        "schema_version": "eiel_campaign_v1",
        "campaign_id": args.campaign_id,
        "description": "One engineering pair: reproduction attempt probability.",
        "no_claims": "No scientific discovery, Full Evolution or Qualification grant.",
        "world": {"base": "patch", "overlays": []},
        "policy": {"family": "fixed_mlp"},
        "selection": {"mode": "standard"},
        "parameters": {"patch.grid_size": 32, "patch.initial_population": 256,
                       "patch.capacity": 512, "patch.founder_placement": "balanced"},
        "matrix": {
            "dimensions": [{"id": "try_probability", "values": [0.0, 0.15]},
                           {"id": "ecology_seed", "values": [101]},
                           {"id": "task_seed", "values": [1001]}],
            "parameter_bindings": {"try_probability": "env_cfg.reproduce_try_prob"},
            "seed_bindings": {"ecology_seed": "ecology", "task_seed": "task"},
            "include": [], "exclude": [], "expected_cells": 2, "max_cells": 2,
        },
        "execution": {
            "backend": "cpu_native", "execution_route_identity": "legacy_generation_route_v1",
            "generations": 6, "ticks_per_generation": 160,
            "batch": {"mode": "bounded", "maximum": 1},
            "budgets": {"vram_bytes": 2 * 1024**3, "disk_bytes": 2 * 1024**3,
                        "wall_seconds": 1800}, "cancellation": False,
        },
        "checkpoints": {"milestones": [3, 6], "digest_every_generation": True,
                        "storage_mode": "disk"},
        "comparisons": {"fresh_repeats": 0, "continuation_cells": 2,
                        "continuation_splits": [3]},
        "artifacts": {"installed_distribution": {
            "path": str(wheel), "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "kind": "private_python_distribution", "abi": "python314",
        }},
        "analysis": {
            "required_metrics": ["population.births", "population.alive", "population.deaths"],
            "optional_metrics": [],
            "rules": [{"id": "birth-difference", "metric": "population.births",
                       "aggregate": "paired_delta", "group_by": ["try_probability"],
                       "pair_by": ["seed.ecology", "seed.task"], "operator": ">=",
                       "threshold": 0, "minimum_coverage": 1.0}],
        },
        "output": {"runtime_root": str(run_root), "reporting_profile": "compact"},
        "artifact_lifecycle": lifecycle_profile_contract_v1(
            "SCIENCE_COMPACT", total_max_bytes=2 * 1024**3, checkpoint_schedule=[3, 6]),
    }
    compile_campaign(spec)  # Static validation only; admission belongs to eiel.
    deviations = {
        "evolution.full_evolution": "Bounded plain Patch CPU reproduction demonstration.",
        "runtime.production_cuda": "Explicit CPU native route; no fallback.",
        "time.absolute_native_tick": "Existing explicit legacy generation route.",
        "checkpoint.native_streamed": "Existing CPU projection checkpoint.",
        "science.wrc_variability": "One supported primitive and matched seed block.",
        "authority.generation_runtime": "Legacy generation engineering exception.",
    }
    request = ExecutionRequestV2.from_mapping({
        "schema_version": "eiel_execution_request_v2", "purpose": "engineering",
        "campaign_spec": str(request_dir / "campaign.json"), "output_root": str(run_root),
        "restore_process_isolation": "inline",
        "baseline": {"profile_id": "EIEL_PRODUCTION_BASELINE_V1", "intent": "development",
                     "deviations": [{"requirement_id": key, "reason": value}
                                    for key, value in deviations.items()]},
    })
    request_dir.mkdir(parents=True)
    for name, value in (("campaign.json", spec), ("request.json", request.payload())):
        (request_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(request_dir / "request.json")


if __name__ == "__main__":
    main()
