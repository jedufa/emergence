#!/usr/bin/env python3
from __future__ import annotations

"""Run representative Chronicle no-influence and byte-parity checks."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


CELLS: tuple[dict[str, Any], ...] = (
    {"name": "flat_control", "seed": 111, "profile": "notable", "knobs": ["env_cfg.patch_world_enable=false"]},
    {
        "name": "predator_pressure",
        "seed": 222,
        "profile": "notable",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.jungle_predator_pressure_enable=true",
            "env_cfg.jungle_predator_base_risk=0.18",
            "env_cfg.jungle_predator_risk_cap=0.18",
        ],
    },
    {
        "name": "wdl_delayed_hazard",
        "seed": 333,
        "profile": "notable",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
            "env_cfg.wdl_delayed_hazard_enable=true",
            "env_cfg.wdl_delayed_hazard_observation_enable=true",
            "env_cfg.wdl_delayed_hazard_delay_steps=2",
            "env_cfg.wdl_delayed_hazard_event_period_steps=8",
        ],
    },
    {
        "name": "lineage_sensitive",
        "seed": 444,
        "profile": "notable",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "sim_cfg.lineage_founder_balance_enable=true",
            "sim_cfg.lineage_founder_group_count=4",
            "sim_cfg.lineage_founder_min_agents_per_group=4",
            "sim_cfg.lineage_founder_assignment_mode=deterministic_round_robin",
        ],
    },
    {
        "name": "scientific_full",
        "seed": 555,
        "profile": "scientific_full",
        "knobs": [
            "env_cfg.patch_world_enable=true",
            "env_cfg.patch_world_observation_enable=true",
        ],
    },
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CHRONICLE-VALID1 parity matrix.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_map(root: Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    ignored = excluded or set()
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in ignored
    }


def _run(
    project_root: Path,
    output_root: Path,
    cell: dict[str, Any],
    arm: str,
    device: str,
) -> dict[str, Any]:
    name = str(cell["name"])
    run_id = f"parity_{name}_{arm}"
    run_dir = output_root / "runs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    enabled = arm != "off"
    command = [
        sys.executable,
        "engine/main.py",
        "--gen",
        "4",
        "--steps",
        "20",
        "--seed",
        str(cell["seed"]),
        "--pop",
        "20",
        "--grid_size",
        "10",
        "--cull",
        "5",
        "--parents",
        "8",
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
        "--checkpoint_bundle_v2_period_gen",
        "4",
        "--checkpoint_rotate",
        "0",
        "--knob",
        "sim_cfg.behavior_log_enable=false",
        "--knob",
        "sim_cfg.policy_deterministic_actions=true",
        "--knob",
        f"sim_cfg.chronicle_enable={'true' if enabled else 'false'}",
        "--knob",
        f"sim_cfg.chronicle_profile={cell['profile']}",
        "--knob",
        "sim_cfg.chronicle_terminal_enable=false",
        "--knob",
        f"sim_cfg.chronicle_run_id=parity-{name}",
        "--knob",
        "sim_cfg.campaign_id=chronicle-valid1-parity",
        "--knob",
        f"sim_cfg.branch_id={name}",
    ]
    if cell["profile"] == "scientific_full":
        command.extend(("--knob", "sim_cfg.allow_large_output_profile=true"))
    for knob in cell["knobs"]:
        command.extend(("--knob", str(knob)))
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{run_id}.stdout.log").open("w", encoding="utf-8") as stdout, (
        logs / f"{run_id}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=project_root, stdout=stdout, stderr=stderr, check=False)
    bundles = sorted((run_dir / "checkpoint_bundles_v2").glob("bundle_gen*"))
    return {
        "cell": name,
        "arm": arm,
        "run_id": run_id,
        "returncode": int(completed.returncode),
        "run_dir": str(run_dir),
        "checkpoint": str(bundles[-1]) if bundles else None,
    }


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True)
    tasks = [(cell, arm) for cell in CELLS for arm in ("off", "on", "repeat")]
    executions: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(_run, project_root, output_root, cell, arm, str(args.device)): (cell, arm)
            for cell, arm in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            executions.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    by_key = {(row["cell"], row["arm"]): row for row in executions}
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        name = str(cell["name"])
        off = by_key[(name, "off")]
        on = by_key[(name, "on")]
        repeat = by_key[(name, "repeat")]
        complete = all(int(row["returncode"]) == 0 and row["checkpoint"] for row in (off, on, repeat))
        if not complete:
            rows.append({"cell": name, "complete": False, "accepted": False})
            continue
        excluded = {"metadata.json", "runner_state.json", "events_hash.txt"}
        off_checkpoint = _file_map(Path(str(off["checkpoint"])), excluded=excluded)
        on_checkpoint = _file_map(Path(str(on["checkpoint"])), excluded=excluded)
        on_chronicle = _file_map(Path(str(on["run_dir"])) / "chronicle")
        repeat_chronicle = _file_map(Path(str(repeat["run_dir"])) / "chronicle")
        metrics_off = _sha(Path(str(off["run_dir"])) / "metrics.csv")
        metrics_on = _sha(Path(str(on["run_dir"])) / "metrics.csv")
        rng_name = "rng_snapshot_v2.json"
        mechanics_equal = off_checkpoint == on_checkpoint
        rng_equal = off_checkpoint.get(rng_name) == on_checkpoint.get(rng_name)
        row = {
            "cell": name,
            "seed": int(cell["seed"]),
            "profile": str(cell["profile"]),
            "complete": True,
            "metrics_byte_equal": metrics_off == metrics_on,
            "mechanics_checkpoint_equal": mechanics_equal,
            "rng_snapshot_equal": rng_equal,
            "off_chronicle_absent": not (Path(str(off["run_dir"])) / "chronicle").exists(),
            "repeat_chronicle_byte_equal": on_chronicle == repeat_chronicle,
            "episode_byte_equal": on_chronicle.get("episodes.json") == repeat_chronicle.get("episodes.json"),
            "formal_event_byte_equal": on_chronicle.get("formal_events.jsonl")
            == repeat_chronicle.get("formal_events.jsonl"),
            "checkpoint_artifact_count": len(off_checkpoint),
        }
        row["accepted"] = all(
            bool(row[key])
            for key in (
                "metrics_byte_equal",
                "mechanics_checkpoint_equal",
                "rng_snapshot_equal",
                "off_chronicle_absent",
                "repeat_chronicle_byte_equal",
                "episode_byte_equal",
                "formal_event_byte_equal",
            )
        )
        rows.append(row)

    sys.path[:0] = [str(project_root), str(project_root / "engine")]
    from simulation.chronicle.retrospective import regenerate_chronicle

    retrospective_rows: list[dict[str, Any]] = []
    for cell in CELLS:
        name = str(cell["name"])
        source = Path(str(by_key[(name, "on")]["run_dir"]))
        output_a = output_root / "retrospective" / name / "a"
        output_b = output_root / "retrospective" / name / "b"
        regenerate_chronicle(source, output_a)
        regenerate_chronicle(source, output_b)
        files_a = _file_map(output_a)
        files_b = _file_map(output_b)
        retrospective_rows.append(
            {
                "cell": name,
                "artifact_count": len(files_a),
                "byte_equal": files_a == files_b,
            }
        )
    result = {
        "schema_version": "eiel_chronicle_valid1_determinism_parity_v1",
        "device": str(args.device),
        "worker_count": max(1, int(args.workers)),
        "cell_count": len(CELLS),
        "simulation_run_count": len(executions),
        "cells": rows,
        "retrospective": retrospective_rows,
        "all_no_influence_passed": all(bool(row.get("accepted")) for row in rows),
        "all_retrospective_byte_parity_passed": all(row["byte_equal"] for row in retrospective_rows),
    }
    result["accepted"] = bool(
        result["all_no_influence_passed"] and result["all_retrospective_byte_parity_passed"]
    )
    (output_root / "determinism_parity.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"accepted": result["accepted"], "cell_count": len(rows)}, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
