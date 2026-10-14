from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics_pkg.metrics_logger import MetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


SCENARIOS = {
    "disabled_control": {
        "path2_roz_reporting_enable": False,
    },
    "static_broad_zone": {
        "path2_roz_reporting_enable": True,
        "path2_roz_schedule_mode": "static_periodic_compat",
        "path2_roz_schedule_id": "static_broad_zone",
        "path2_roz_window_period_steps": 1,
        "path2_roz_window_duration_steps": 1,
        "path2_roz_zone_radius": 16,
    },
    "periodic_single_zone": {
        "path2_roz_reporting_enable": True,
        "path2_roz_schedule_mode": "periodic_single_zone",
        "path2_roz_schedule_id": "periodic_single_zone",
        "path2_roz_window_period_steps": 4,
        "path2_roz_window_duration_steps": 1,
        "path2_roz_zone_radius": 3,
    },
    "round_robin_multi_zone": {
        "path2_roz_reporting_enable": True,
        "path2_roz_schedule_mode": "multi_zone_round_robin",
        "path2_roz_schedule_id": "round_robin_multi_zone",
        "path2_roz_window_period_steps": 4,
        "path2_roz_window_duration_steps": 1,
        "path2_roz_schedule_zones_json": json.dumps(
            [
                {"id": "north_west", "center_y": 3, "center_x": 3, "radius": 3},
                {"id": "south_east", "center_y": 12, "center_x": 12, "radius": 3},
            ],
            sort_keys=True,
            separators=(",", ":"),
        ),
    },
}


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _final_fingerprint(sim: SimulationRunner) -> str:
    rows = []
    for agent in sorted(sim.pop.agents.values(), key=lambda item: int(item.agent_id)):
        rows.append(
            [
                int(agent.agent_id),
                str(agent.stats.lineage_id),
                bool(agent.alive),
                int(getattr(agent, "y", -1)),
                int(getattr(agent, "x", -1)),
            ]
        )
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _event_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "event_jsonl_rows": len(rows),
        "roz_enabled_rows": 0,
        "roz_active_rows": 0,
        "scan_rows": 0,
        "active_window_scan_rows": 0,
        "selected_parent_in_zone_rows": 0,
        "eligibility_in_zone_rows": 0,
        "attempt_in_zone_rows": 0,
        "success_in_zone_rows": 0,
        "action_offered_rows": 0,
        "action_taken_rows": 0,
    }
    zone_ids: set[str] = set()
    phase_steps: set[int] = set()
    for row in rows:
        event_type = str(row.get("event_type") or "")
        roz_on = int(row.get("path2_roz_reporting_enable", 0) or 0) > 0
        active = int(row.get("path2_roz_window_active", 0) or 0) > 0
        parent_in_zone = int(row.get("path2_roz_parent_in_zone", 0) or 0) > 0
        selected_parent_in_zone = int(row.get("path2_roz_selected_parent_scan_in_zone", 0) or 0) > 0
        if roz_on:
            counts["roz_enabled_rows"] += 1
            zone_id = str(row.get("path2_roz_zone_id") or "")
            if zone_id:
                zone_ids.add(zone_id)
            phase = row.get("path2_roz_schedule_phase_step", "")
            if phase not in ("", None):
                phase_steps.add(int(phase))
        if active:
            counts["roz_active_rows"] += 1
        if event_type.startswith("reproduction_scan"):
            counts["scan_rows"] += 1
            if active:
                counts["active_window_scan_rows"] += 1
        if selected_parent_in_zone:
            counts["selected_parent_in_zone_rows"] += 1
        if event_type == "reproduction_scan_eligible" and parent_in_zone:
            counts["eligibility_in_zone_rows"] += 1
        if event_type == "reproduction_scan_attempt_recorded" and parent_in_zone:
            counts["attempt_in_zone_rows"] += 1
        if event_type == "birth_success" and int(row.get("path2_roz_birth_in_zone", 0) or 0) > 0:
            counts["success_in_zone_rows"] += 1
        if event_type == "reproduction_action_offered":
            counts["action_offered_rows"] += 1
        if event_type == "reproduction_action_taken":
            counts["action_taken_rows"] += 1
    counts["distinct_zone_count"] = len(zone_ids)
    counts["distinct_phase_step_count"] = len(phase_steps)
    return counts


def _configure_cell(
    *,
    run_dir: Path,
    seed: int,
    scenario_id: str,
    generations: int,
    steps: int,
    population: int,
    grid: int,
    device: str,
) -> SimulationRunner:
    env_cfg = EnvConfig(width=int(grid), height=int(grid))
    env_cfg.reproduce_requires_mate = False
    env_cfg.reproduce_try_prob = 1.0
    env_cfg.reproduce_cooldown_steps = 1
    env_cfg.reproduce_parent_energy_share = 0.1
    env_cfg.reproduce_min_energy_buffer = 0.0
    pop_cfg = PopConfig(population_size=int(population), parent_fraction=0.5, cull_fraction=0.25)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device=str(device),
        bank_device=str(device),
        behavior_log_enable=False,
        policy_deterministic_actions=True,
    )
    sim_cfg.lineage_reproduction_funnel_reporting_enable = True
    sim_cfg.lineage_reproduction_funnel_write_events = True
    sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    for key, value in SCENARIOS[str(scenario_id)].items():
        setattr(sim_cfg, key, value)
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(run_dir)),
        run_dir=str(run_dir),
        perf=False,
    )


def _run_cell(
    *,
    raw_root: Path,
    seed: int,
    scenario_id: str,
    phase: str,
    generations: int,
    steps: int,
    population: int,
    grid: int,
    device: str,
    repeat_index: int = 0,
) -> dict[str, Any]:
    run_id = f"{phase}_{scenario_id}_seed{int(seed)}"
    if repeat_index:
        run_id += f"_repeat{repeat_index}"
    run_dir = raw_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sim = _configure_cell(
        run_dir=run_dir,
        seed=int(seed),
        scenario_id=str(scenario_id),
        generations=int(generations),
        steps=int(steps),
        population=int(population),
        grid=int(grid),
        device=str(device),
    )
    sim.run(start_gen=0, skip_init=False)
    try:
        sim.logger.close()
    except Exception:
        pass
    rows = _iter_events(run_dir)
    summary = _read_json(run_dir / "metrics" / "lineage_reproduction_funnel_run_summary.json")
    counts = _event_counts(rows)
    return {
        "phase": str(phase),
        "scenario_id": str(scenario_id),
        "seed": int(seed),
        "repeat_index": int(repeat_index),
        "run_dir": str(run_dir),
        "completed": True,
        "generations": int(generations),
        "steps_per_generation": int(steps),
        "population": int(population),
        "grid": int(grid),
        "device": str(device),
        "metrics_sha256": _sha256(run_dir / "metrics.csv"),
        "config_sha256": _sha256(run_dir / "config.json"),
        "run_config_sha256": _sha256(run_dir / "run_config.csv"),
        "final_fingerprint_sha256": _final_fingerprint(sim),
        "summary_counter_mismatches": _summary_mismatches(counts, summary),
        **counts,
    }


def _summary_mismatches(counts: dict[str, int], summary: dict[str, Any]) -> dict[str, dict[str, int]]:
    mapping = [
        ("roz_enabled_rows", "path2_roz_event_rows"),
        ("roz_active_rows", "path2_roz_window_active_rows"),
        ("roz_active_rows", "path2_roz_schedule_window_active_rows"),
        ("selected_parent_in_zone_rows", "path2_roz_selected_parent_scan_in_zone_rows"),
        ("action_offered_rows", "reproduction_action_offered_rows"),
        ("action_taken_rows", "reproduction_action_taken_rows"),
    ]
    mismatches: dict[str, dict[str, int]] = {}
    for count_key, summary_key in mapping:
        expected = int(counts.get(count_key, 0) or 0)
        actual = int(summary.get(summary_key, 0) or 0)
        if actual != expected:
            mismatches[summary_key] = {"summary": actual, "computed": expected}
    return mismatches


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _decision_summary(cells: list[dict[str, Any]], parity: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in cells if row.get("summary_counter_mismatches")]
    disabled_bad = [
        row
        for row in cells
        if row["scenario_id"] == "disabled_control" and int(row.get("roz_enabled_rows", 0) or 0) != 0
    ]
    action_bad = [
        row
        for row in cells
        if int(row.get("action_offered_rows", 0) or 0) or int(row.get("action_taken_rows", 0) or 0)
    ]
    enabled = [row for row in cells if row["scenario_id"] != "disabled_control"]
    intersects = [row for row in enabled if int(row.get("active_window_scan_rows", 0) or 0) > 0]
    parity_bad = [row for row in parity if not row.get("passed", False)]
    if disabled_bad or action_bad or parity_bad:
        decision = "PATH2-ROZ1-SCHEDULES-D"
        recommendation = "REPAIR-REPORTING-DETERMINISM"
    elif failed:
        decision = "PATH2-ROZ1-SCHEDULES-C"
        recommendation = "REPAIR-REPORTING-SCHEMA"
    elif len(intersects) == len(enabled):
        decision = "PATH2-ROZ1-SCHEDULES-A"
        recommendation = "ROZ2-PATCH-DERIVED-ZONE-DESIGN"
    else:
        decision = "PATH2-ROZ1-SCHEDULES-B"
        recommendation = "ROZ1-DIAGNOSTIC-CAMPAIGN"
    return {
        "decision": decision,
        "roz2_recommendation": recommendation,
        "disabled_contamination_count": len(disabled_bad),
        "action_semantics_contamination_count": len(action_bad),
        "summary_mismatch_cell_count": len(failed),
        "parity_failure_count": len(parity_bad),
        "enabled_intersection_cell_count": len(intersects),
        "enabled_cell_count": len(enabled),
    }


def _parity(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in cells:
        grouped.setdefault((str(row["scenario_id"]), int(row["seed"])), []).append(row)
    out = []
    for (scenario_id, seed), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        base = sorted(rows, key=lambda item: int(item["repeat_index"]))[0]
        for repeat in sorted(rows, key=lambda item: int(item["repeat_index"]))[1:]:
            passed = all(
                str(base.get(key, "")) == str(repeat.get(key, ""))
                for key in ("metrics_sha256", "config_sha256", "run_config_sha256", "final_fingerprint_sha256")
            )
            out.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "repeat_index": int(repeat["repeat_index"]),
                    "passed": bool(passed),
                    "metrics_match": base.get("metrics_sha256") == repeat.get("metrics_sha256"),
                    "config_match": base.get("config_sha256") == repeat.get("config_sha256"),
                    "run_config_match": base.get("run_config_sha256") == repeat.get("run_config_sha256"),
                    "final_fingerprint_match": base.get("final_fingerprint_sha256")
                    == repeat.get("final_fingerprint_sha256"),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-diagnostic", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    evidence_dir = Path(args.evidence_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    for scenario_id in SCENARIOS:
        cells.append(
            _run_cell(
                raw_root=raw_root,
                seed=424242,
                scenario_id=scenario_id,
                phase="smoke",
                generations=3,
                steps=40,
                population=32,
                grid=16,
                device=str(args.device),
            )
        )

    smoke_summary = _decision_summary(cells, [])
    if smoke_summary["decision"] in {"PATH2-ROZ1-SCHEDULES-C", "PATH2-ROZ1-SCHEDULES-D"}:
        diagnostic_cells: list[dict[str, Any]] = []
    elif args.run_diagnostic:
        diagnostic_cells = []
        for seed in (101, 202, 303):
            for scenario_id in SCENARIOS:
                diagnostic_cells.append(
                    _run_cell(
                        raw_root=raw_root,
                        seed=seed,
                        scenario_id=scenario_id,
                        phase="diagnostic",
                        generations=6,
                        steps=80,
                        population=64,
                        grid=16,
                        device=str(args.device),
                    )
                )
        for scenario_id, seed in (("disabled_control", 101), ("periodic_single_zone", 101)):
            diagnostic_cells.append(
                _run_cell(
                    raw_root=raw_root,
                    seed=seed,
                    scenario_id=scenario_id,
                    phase="diagnostic",
                    generations=6,
                    steps=80,
                    population=64,
                    grid=16,
                    device=str(args.device),
                    repeat_index=1,
                )
            )
        cells.extend(diagnostic_cells)
    else:
        diagnostic_cells = []

    parity = _parity(diagnostic_cells)
    decision = _decision_summary(cells, parity)
    summary = {
        "schema_version": 1,
        "raw_root": str(raw_root),
        "evidence_dir": str(evidence_dir),
        "device": str(args.device),
        "smoke_cell_count": 4,
        "diagnostic_cell_count": len([row for row in diagnostic_cells if int(row.get("repeat_index", 0)) == 0]),
        "repeat_cell_count": len([row for row in diagnostic_cells if int(row.get("repeat_index", 0)) > 0]),
        "campaigns_run": 0,
        **decision,
    }
    _write_csv(evidence_dir / "path2_roz1_cells.csv", cells)
    _write_csv(evidence_dir / "path2_roz1_parity.csv", parity)
    (evidence_dir / "path2_roz1_decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "path2_roz1_cells.json").write_text(
        json.dumps(cells, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if decision["decision"] in {"PATH2-ROZ1-SCHEDULES-A", "PATH2-ROZ1-SCHEDULES-B"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
