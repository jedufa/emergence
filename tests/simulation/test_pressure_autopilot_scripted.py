from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from simulation.autopilot_pkg.pressure_autopilot import PressureAutopilot
from simulation.control_plane_pkg.control_plane import ControlPlane


def _make_sim(tmp_path: Path, script: dict) -> SimpleNamespace:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = SimpleNamespace(
        branch_id="branch-a",
        maintenance_opportunity_cost_coeff=0.35,
        pressure_autopilot_enable=True,
        pressure_autopilot_hold=False,
        pressure_autopilot_reset_on_enable=False,
        pressure_autopilot_pressure_init=0.5,
        pressure_autopilot_tick_interval_gen=1,
        pressure_autopilot_start_gen=300,
        pressure_autopilot_profile="scripted",
        pressure_autopilot_script=script,
        pressure_autopilot_apply_delay_gen=1,
        pressure_autopilot_suggestion_only=False,
    )
    control_plane = ControlPlane(
        run_dir=str(run_dir),
        branch_id="branch-a",
        campaign_id="campaign-a",
        initial_generation=300,
    )
    logger = SimpleNamespace(
        last_row={
            "cohort_survival_rate": 0.5,
            "genome_diversity_norm_std_mean": 0.1,
            "diversity_collapse_streak": 0,
            "alive": 12,
        }
    )
    return SimpleNamespace(run_dir=str(run_dir), sim_cfg=sim_cfg, control_plane=control_plane, logger=logger)


def test_scripted_autopilot_schedules_and_applies_planned_update(tmp_path: Path) -> None:
    sim = _make_sim(
        tmp_path,
        {
            "schema_version": "pressure_autopilot_script_v1",
            "entries": [
                {
                    "decision_generation": 300,
                    "target_generation": 301,
                    "label": "apply_treatment",
                    "updates": [
                        {
                            "path": "sim_cfg.maintenance_opportunity_cost_coeff",
                            "value": 0.4375,
                        }
                    ],
                }
            ],
        },
    )
    autopilot = PressureAutopilot(sim.run_dir)

    autopilot.maybe_tick(sim, 300)

    decisions_path = Path(sim.run_dir) / "autopilot" / "pressure_autopilot_decisions.jsonl"
    records = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert records[0]["ack_status"] == "ACCEPTED"
    assert records[0]["target_generation"] == 301

    applied = sim.control_plane.process_pending(301, {"generation_index": 301}, sim=sim)
    assert len(applied) == 1
    assert sim.sim_cfg.maintenance_opportunity_cost_coeff == 0.4375

    events_path = Path(sim.run_dir) / "events" / "command_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert events[0]["generation_applied"] == 301
    updates = events[0]["command_payload"]["updates"]
    assert updates == [{"path": "sim_cfg.maintenance_opportunity_cost_coeff", "value": 0.4375}]


def test_scripted_autopilot_pulse_reverts_to_baseline(tmp_path: Path) -> None:
    sim = _make_sim(
        tmp_path,
        {
            "schema_version": "pressure_autopilot_script_v1",
            "entries": [
                {
                    "decision_generation": 300,
                    "target_generation": 301,
                    "label": "apply_treatment",
                    "updates": [
                        {
                            "path": "sim_cfg.maintenance_opportunity_cost_coeff",
                            "value": 0.175,
                        }
                    ],
                },
                {
                    "decision_generation": 310,
                    "target_generation": 311,
                    "label": "revert_to_baseline",
                    "updates": [
                        {
                            "path": "sim_cfg.maintenance_opportunity_cost_coeff",
                            "value": 0.35,
                        }
                    ],
                },
            ],
        },
    )
    autopilot = PressureAutopilot(sim.run_dir)

    autopilot.maybe_tick(sim, 300)
    sim.control_plane.process_pending(301, {"generation_index": 301}, sim=sim)
    assert sim.sim_cfg.maintenance_opportunity_cost_coeff == 0.175

    autopilot.maybe_tick(sim, 310)
    sim.control_plane.process_pending(311, {"generation_index": 311}, sim=sim)
    assert sim.sim_cfg.maintenance_opportunity_cost_coeff == 0.35

    state_path = Path(sim.run_dir) / "autopilot" / "pressure_autopilot_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["script_cursor"] == 2
