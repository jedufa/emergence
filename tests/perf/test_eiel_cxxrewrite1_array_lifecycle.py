from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from perf.cxxrewrite1_array_lifecycle import (
    begin_generation_lifecycle,
    build_lifecycle_state_from_runner,
    churn_tape_digest,
    churn_tape_schema_payload,
    finish_generation_lifecycle,
    lifecycle_digest,
    ownership_map_payload,
    replay_generation_lifecycle,
    sync_python_objects_from_lifecycle,
    validate_lifecycle_against_runner,
)
from simulation.runner_pkg.runner_config import SimConfig


class _Policy:
    pass


def _agent(agent_id: int, slot: int, *, alive: bool = True, parent_ids: tuple[int, ...] = ()) -> SimpleNamespace:
    stats = SimpleNamespace(
        survival_steps=agent_id + 2,
        lineage_id=f"L{agent_id:08d}",
        founder_id=parent_ids[0] if parent_ids else agent_id,
        parent_ids=parent_ids,
        exp_dE=np.full((7,), float(agent_id), dtype=np.float32),
        exp_dInj=np.full((7,), float(slot), dtype=np.float32),
    )
    return SimpleNamespace(
        agent_id=agent_id,
        slot=slot,
        slot_token=slot + 100,
        alive=alive,
        energy=10.0 + float(agent_id),
        health=20.0 + float(agent_id),
        injury=float(slot) / 10.0,
        drives=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32) + float(slot),
        stats=stats,
        policy=_Policy(),
    )


def _runner(tmp_path, agents: dict[int, SimpleNamespace], *, cfg: SimConfig | None = None) -> SimpleNamespace:
    cap = 8
    env_alive = np.zeros((cap,), dtype=bool)
    pos_y = np.full((cap,), -1, dtype=np.int32)
    pos_x = np.full((cap,), -1, dtype=np.int32)
    for aid, agent in agents.items():
        if bool(agent.alive):
            env_alive[aid] = True
            pos_y[aid] = 2 + int(agent.slot)
            pos_x[aid] = 4 + int(agent.slot)
    env = SimpleNamespace(agent_alive=env_alive, agent_pos_y=pos_y, agent_pos_x=pos_x)
    pop = SimpleNamespace(
        agents=agents,
        max_slots=6,
        next_id=max(agents) + 1,
        alive_agents=lambda: sorted([a for a in agents.values() if bool(a.alive)], key=lambda a: int(a.agent_id)),
    )
    return SimpleNamespace(
        env=env,
        pop=pop,
        sim_cfg=cfg or SimConfig(),
        run_dir=str(tmp_path),
        _Hmem=np.arange(6 * 4, dtype=np.float32).reshape(6, 4),
    )


def _start_runner(tmp_path) -> SimpleNamespace:
    return _runner(tmp_path, {1: _agent(1, 0), 2: _agent(2, 1)})


def _end_runner(tmp_path, *, cfg: SimConfig | None = None) -> SimpleNamespace:
    return _runner(
        tmp_path,
        {
            1: _agent(1, 0, alive=False),
            2: _agent(2, 1),
            3: _agent(3, 0, parent_ids=(1, 2)),
        },
        cfg=cfg,
    )


def test_cxxrewrite1_replay_records_death_birth_slot_and_sync_events(tmp_path) -> None:
    start = _start_runner(tmp_path / "start")
    start_state = build_lifecycle_state_from_runner(start, generation=0, phase="generation_start")
    start_state.record_event("generation_start_snapshot", agent_id=-1, reason="test")

    replay = replay_generation_lifecycle(_end_runner(tmp_path / "end"), start_state=start_state, generation=0)
    state = replay["state"]
    events = [row["event"] for row in state.churn_events]

    assert events[0] == "generation_start_snapshot"
    assert "death" in events
    assert "birth_success" in events
    assert "slot_freed" in events
    assert "slot_allocated" in events
    assert "hmem_row_allocated" in events
    assert "policy_row_attached" in events
    assert events[-1] == "python_sync_boundary"
    assert state.slot_to_agent_id.tolist()[:2] == [3, 2]
    assert state.alive_mask.tolist()[:2] == [True, True]
    assert churn_tape_digest(state.churn_events) == churn_tape_digest(state.churn_events)


def test_cxxrewrite1_sync_restores_python_agent_fields(tmp_path) -> None:
    runner = _end_runner(tmp_path)
    state = build_lifecycle_state_from_runner(runner, generation=1)
    state.energy[0] = np.float32(42.5)
    runner.pop.agents[3].energy = -1.0

    summary = sync_python_objects_from_lifecycle(runner, state)

    assert summary["synced_alive_agents"] == 2
    assert runner.pop.agents[3].energy == pytest.approx(42.5)
    assert validate_lifecycle_against_runner(state, runner)["ok"] is True


def test_cxxrewrite1_authoritative_mode_fails_closed_without_required_flags(tmp_path) -> None:
    cfg = SimConfig(
        cxxrewrite_array_backend_enable=True,
        cxxrewrite_lifecycle_authoritative_enable=True,
    )
    runner = _start_runner(tmp_path)
    runner.sim_cfg = cfg

    with pytest.raises(RuntimeError, match="required components"):
        begin_generation_lifecycle(runner, alive=runner.pop.alive_agents(), generation=0)


def test_cxxrewrite1_finish_authoritative_writes_summary_and_churn(tmp_path) -> None:
    cfg = SimConfig(
        cxxrewrite_array_backend_enable=True,
        cxxrewrite_lifecycle_authoritative_enable=True,
        cxxrewrite_lifecycle_fail_closed=True,
        cxxrewrite_lifecycle_sync_python_objects_enable=True,
        cxxrewrite_lifecycle_record_churn_tape_enable=True,
        cxxrewrite_lifecycle_validate_shadow_enable=True,
    )
    start = _start_runner(tmp_path / "start")
    start.sim_cfg = cfg
    start_state = begin_generation_lifecycle(start, alive=start.pop.alive_agents(), generation=0)
    end = _end_runner(tmp_path / "end", cfg=cfg)

    payload = finish_generation_lifecycle(end, start_state=start_state, generation=0)

    assert payload is not None
    assert payload["validation"]["ok"] is True
    assert payload["sync_summary"]["enabled"] is True
    assert payload["churn_event_count"] >= 7
    assert (tmp_path / "end" / "metrics" / "cxxrewrite1_lifecycle_summary.jsonl").exists()
    assert (tmp_path / "end" / "metrics" / "cxxrewrite1_churn_tape.jsonl").exists()


def test_cxxrewrite1_contract_payloads_are_stable(tmp_path) -> None:
    state = build_lifecycle_state_from_runner(_start_runner(tmp_path), generation=3)
    assert lifecycle_digest(state) == lifecycle_digest(state)
    assert "death" in churn_tape_schema_payload()["events"]
    assert ownership_map_payload()["selected_decision"] == "CXXREWRITE1-A-ARRAY-LIFECYCLE-AUTHORITATIVE-GO"
