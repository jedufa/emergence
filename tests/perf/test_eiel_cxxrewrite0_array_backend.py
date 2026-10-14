from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from perf.cxxrewrite0_array_backend import (
    build_array_backend_snapshot,
    churn_tape_contract_payload,
    ownership_map_payload,
    record_generation_shadow,
    snapshot_digest,
    validate_snapshot_against_runner,
)
from simulation.runner_pkg.runner_config import SimConfig


def _agent(agent_id: int, slot: int, *, alive: bool = True) -> SimpleNamespace:
    stats = SimpleNamespace(
        lineage_id=f"L{agent_id:08d}",
        founder_id=agent_id,
        exp_dE=np.full((7,), float(agent_id), dtype=np.float32),
        exp_dInj=np.full((7,), float(slot), dtype=np.float32),
    )
    return SimpleNamespace(
        agent_id=agent_id,
        slot=slot,
        slot_token=slot + 100,
        alive=alive,
        energy=10.0 + agent_id,
        health=20.0 + agent_id,
        injury=float(slot) / 10.0,
        age=3 + agent_id,
        drives=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32) + float(slot),
        stats=stats,
    )


def _runner(tmp_path, *, cfg: SimConfig | None = None) -> SimpleNamespace:
    agents = {
        1: _agent(1, 2),
        2: _agent(2, 0),
        3: _agent(3, 4, alive=False),
    }
    env = SimpleNamespace(
        agent_alive=np.asarray([False, True, True, False, False, False], dtype=bool),
        agent_pos_y=np.asarray([-1, 5, 6, -1, -1, -1], dtype=np.int32),
        agent_pos_x=np.asarray([-1, 7, 8, -1, -1, -1], dtype=np.int32),
    )
    pop = SimpleNamespace(
        agents=agents,
        max_slots=6,
        alive_agents=lambda: [agents[1], agents[2]],
    )
    return SimpleNamespace(
        env=env,
        pop=pop,
        sim_cfg=cfg or SimConfig(),
        run_dir=str(tmp_path),
        _Hmem=np.arange(6 * 4, dtype=np.float32).reshape(6, 4),
    )


def test_cxxrewrite0_snapshot_builds_slot_owned_arrays(tmp_path) -> None:
    runner = _runner(tmp_path)
    snapshot = build_array_backend_snapshot(runner, generation=5)

    assert snapshot.slot_to_agent_id.tolist() == [2, -1, 1, -1, -1, -1]
    assert snapshot.agent_id_to_slot_agent_ids.tolist() == [1, 2]
    assert snapshot.agent_id_to_slot_slots.tolist() == [2, 0]
    assert snapshot.alive_mask.tolist() == [True, False, True, False, False, False]
    assert snapshot.pos_y[2] == 5
    assert snapshot.pos_x[0] == 8
    assert snapshot.hmem.shape == (6, 4)
    assert snapshot.action_buffer.tolist() == [-1, -1, -1, -1, -1, -1]
    assert snapshot.observation_buffer.shape == (6, 0)


def test_cxxrewrite0_snapshot_validation_and_digest_are_stable(tmp_path) -> None:
    runner = _runner(tmp_path)
    first = build_array_backend_snapshot(runner, generation=1)
    second = build_array_backend_snapshot(runner, generation=1)

    assert validate_snapshot_against_runner(first, runner)["ok"] is True
    assert snapshot_digest(first) == snapshot_digest(second)


def test_cxxrewrite0_shadow_record_writes_debug_payload(tmp_path) -> None:
    cfg = SimConfig(
        runner_array_backend_enable=True,
        cxxrewrite_array_backend_shadow_enable=True,
        cxxrewrite_array_backend_debug_sync_enable=True,
        cxxrewrite_array_backend_fail_fast=True,
    )
    runner = _runner(tmp_path, cfg=cfg)
    payload = record_generation_shadow(runner, alive=runner.pop.alive_agents(), generation=2, buckets={})

    assert payload is not None
    assert payload["validation"]["ok"] is True
    assert getattr(runner, "_cxxrewrite0_array_backend_last_digest")
    assert (tmp_path / "metrics" / "cxxrewrite0_array_backend_shadow.jsonl").exists()


def test_cxxrewrite0_authoritative_mode_fails_closed(tmp_path) -> None:
    cfg = SimConfig(
        runner_array_backend_enable=True,
        cxxrewrite_array_backend_authoritative_enable=True,
    )
    runner = _runner(tmp_path, cfg=cfg)

    with pytest.raises(RuntimeError, match="authoritative array backend is not implemented"):
        record_generation_shadow(runner, alive=runner.pop.alive_agents(), generation=0)


def test_cxxrewrite0_contract_payloads_name_churn_and_owners() -> None:
    ownership = ownership_map_payload()
    churn = churn_tape_contract_payload()

    assert ownership["selected_decision"] == "CXXREWRITE0-A-ARRAYBACKEND-FIRST-GO"
    assert any(row["item"] == "births/deaths/refill" for row in ownership["state_items"])
    assert {event["event"] for event in churn["events"]} >= {"death", "birth", "slot_alloc", "hmem_alloc"}
