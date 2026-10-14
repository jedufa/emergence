from __future__ import annotations

import shutil

import numpy as np
import pytest

from perf.cxxrewrite4_upstream_resident import (
    assert_authoritative_not_supported,
    compile_native_backend,
    compare_packed,
    load_native_backend,
    prepare_policy_materialization,
    prepare_resident_generation_env_scalar_inputs,
    python_pack_policy_heads,
    record_action_array,
    record_post_env_scalar_arrays,
    sync_agents_scalar_state,
)
from simulation.runner_pkg.runner_config import SimConfig


class _Agent:
    def __init__(self, agent_id: int, slot: int, token: int):
        self.agent_id = int(agent_id)
        self.slot = int(slot)
        self.slot_token = int(token)
        self.alive = True
        self.energy = 10.0 + float(agent_id)
        self.health = 5.0 + float(agent_id)
        self.injury = 0.25 * float(agent_id)
        self.scar = 0.05 * float(agent_id)


class _Pop:
    max_slots = 4


class _Env:
    def __init__(self):
        self.agent_alive = np.zeros((8,), dtype=np.bool_)
        self.agent_alive[[1, 2]] = True
        self.agent_pos_y = np.full((8,), -1, dtype=np.int32)
        self.agent_pos_x = np.full((8,), -1, dtype=np.int32)
        self.agent_pos_y[1] = 1
        self.agent_pos_x[1] = 2
        self.agent_pos_y[2] = 3
        self.agent_pos_x[2] = 4


class _Runner:
    def __init__(self, cfg: SimConfig | None = None):
        self.sim_cfg = cfg if cfg is not None else SimConfig()
        self.env = _Env()
        self.pop = _Pop()
        self.run_dir = None
        self.agents = [_Agent(1, 0, 10), _Agent(2, 1, 11)]
        self._drive_hunger = np.asarray([0.1, 0.2, 0.0, 0.0], dtype=np.float32)
        self._drive_pain = np.asarray([0.3, 0.4, 0.0, 0.0], dtype=np.float32)
        self._drive_fatigue = np.asarray([0.5, 0.6, 0.0, 0.0], dtype=np.float32)
        self._drive_fear = np.asarray([0.7, 0.8, 0.0, 0.0], dtype=np.float32)

    def _dev_generation_index(self) -> int:
        return 0

    def _dev_tick_index(self) -> int:
        return 0

    def _ensure_drive_arrays(self, required_slots: int | None = None) -> None:
        return None


def _cfg(**updates: object) -> SimConfig:
    cfg = SimConfig()
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def test_cxxrewrite4_config_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.cxxrewrite_resident_observation_enable is False
    assert cfg.cxxrewrite_resident_policyframe_enable is False
    assert cfg.cxxrewrite_resident_post_env_scalar_enable is False
    assert cfg.cxxrewrite_resident_generation_scalar_env_inputs_enable is False
    assert cfg.cxxrewrite_resident_upstream_defer_summary_write_enable is False
    assert cfg.cxxrewrite_resident_upstream_summary_filename == "cxxrewrite4_upstream_summary.json"


def test_cxxrewrite4_authoritative_modes_fail_closed() -> None:
    cfg = _cfg(cxxrewrite_resident_policyframe_authoritative_enable=True)

    with pytest.raises(RuntimeError, match="not supported"):
        assert_authoritative_not_supported(cfg)


def test_cxxrewrite4_policy_materialization_uses_resident_heads() -> None:
    runner = _Runner(
        _cfg(
            cxxrewrite_resident_observation_enable=True,
            cxxrewrite_resident_observation_shadow_enable=True,
            cxxrewrite_resident_policyframe_enable=True,
            cxxrewrite_resident_policyframe_shadow_enable=True,
            cxxrewrite_resident_post_env_scalar_enable=True,
            cxxrewrite_resident_upstream_validate_shadow_enable=True,
        )
    )
    ids = np.asarray([1, 2], dtype=np.int32)
    result = prepare_policy_materialization(
        runner,
        alive=runner.agents,
        agent_ids_np=ids,
        e_norm=100.0,
        h_norm=50.0,
        stage="test",
    )

    assert result is not None
    assert result["use_resident_policy_heads"] is True
    assert np.allclose(result["energy_norm"], np.asarray([11.0, 12.0], dtype=np.float32) / 100.0)
    assert np.allclose(result["health_norm"], np.asarray([6.0, 7.0], dtype=np.float32) / 50.0)
    assert np.allclose(result["drives_np"], np.asarray([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8]], dtype=np.float32))
    summary = runner._cxxrewrite4_upstream_summary
    assert summary["python_agent_scalar_reads_avoided"] == 4
    assert summary["observation_digest_parity_ok"] is True


def test_cxxrewrite4_action_array_update() -> None:
    runner = _Runner(_cfg(cxxrewrite_resident_policyframe_enable=True))
    ids = np.asarray([1, 2], dtype=np.int32)

    record_action_array(runner, alive=runner.agents, agent_ids_np=ids, actions_np=np.asarray([4, 5], dtype=np.int32))

    frame = runner._cxxrewrite3_resident_frame
    assert frame.action_by_id[1] == 4
    assert frame.action_by_id[2] == 5
    assert runner._cxxrewrite4_upstream_summary["resident_action_updates"] == 2


def test_cxxrewrite4_post_env_arrays_update_resident_frame() -> None:
    runner = _Runner(_cfg(cxxrewrite_resident_post_env_scalar_enable=True))
    ids = np.asarray([1, 2], dtype=np.int32)

    record_post_env_scalar_arrays(
        runner,
        alive=runner.agents,
        agent_ids_np=ids,
        new_energy=np.asarray([20.0, 21.0], dtype=np.float32),
        new_health=np.asarray([8.0, 9.0], dtype=np.float32),
        new_injury=np.asarray([0.0, 0.1], dtype=np.float32),
        new_scar=np.asarray([0.2, 0.3], dtype=np.float32),
        died_mask=np.asarray([False, True], dtype=bool),
    )

    frame = runner._cxxrewrite3_resident_frame
    assert np.allclose(frame.energy_by_id[ids], np.asarray([20.0, 21.0], dtype=np.float32))
    assert bool(frame.alive_by_id[1]) is True
    assert bool(frame.alive_by_id[2]) is False
    assert runner._cxxrewrite4_upstream_summary["python_agent_scalar_writes_resident_first"] == 8


def test_resident_generation_scalar_env_inputs_read_resident_frame() -> None:
    runner = _Runner(_cfg(cxxrewrite_resident_generation_scalar_env_inputs_enable=True))
    ids = np.asarray([1, 2], dtype=np.int32)
    record_post_env_scalar_arrays(
        runner,
        alive=runner.agents,
        agent_ids_np=ids,
        new_energy=np.asarray([20.0, 21.0], dtype=np.float32),
        new_health=np.asarray([8.0, 9.0], dtype=np.float32),
        new_injury=np.asarray([0.0, 0.1], dtype=np.float32),
        new_scar=np.asarray([0.2, 0.3], dtype=np.float32),
        died_mask=np.asarray([False, False], dtype=bool),
    )

    inputs = prepare_resident_generation_env_scalar_inputs(
        runner,
        alive=runner.agents,
        agent_ids_np=ids,
    )

    assert inputs is not None
    assert np.allclose(inputs["energy"], np.asarray([20.0, 21.0], dtype=np.float32))
    assert np.allclose(inputs["health"], np.asarray([8.0, 9.0], dtype=np.float32))
    summary = runner._cxxrewrite4_upstream_summary
    assert summary["resident_generation_pre_env_records"] == 1
    assert summary["resident_generation_pre_env_scalar_input_reads"] == 8
    assert summary["python_agent_scalar_reads_avoided"] == 8


def test_cxxrewrite4_agent_scalar_sync_tracks_post_env_mutation() -> None:
    runner = _Runner(_cfg(cxxrewrite_resident_post_env_scalar_enable=True))
    agent = runner.agents[0]
    agent.energy = 3.0
    agent.health = 2.0
    agent.alive = False

    sync_agents_scalar_state(runner, [agent], stage="test_sync")

    frame = runner._cxxrewrite3_resident_frame
    assert float(frame.energy_by_id[1]) == 3.0
    assert float(frame.health_by_id[1]) == 2.0
    assert bool(frame.alive_by_id[1]) is False
    assert runner._cxxrewrite4_upstream_summary["python_post_env_agent_scalar_syncs"] == 1


def test_cxxrewrite4_native_policy_pack_matches_python_fixture(tmp_path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    build = compile_native_backend(tmp_path)
    assert build["ok"], build
    runner = _Runner(_cfg(cxxrewrite_resident_policyframe_enable=True))
    ids = np.asarray([1, 2], dtype=np.int32)
    prepare_policy_materialization(
        runner,
        alive=runner.agents,
        agent_ids_np=ids,
        e_norm=100.0,
        h_norm=50.0,
        stage="test",
    )
    frame = runner._cxxrewrite3_resident_frame
    expected = python_pack_policy_heads(runner, frame, ids, energy_norm_denom=100.0, health_norm_denom=50.0)
    actual = load_native_backend(build["library_path"])(
        runner,
        frame,
        ids,
        energy_norm_denom=100.0,
        health_norm_denom=50.0,
    )

    assert compare_packed(expected, actual)["ok"] is True
