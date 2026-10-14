from __future__ import annotations

import shutil

import numpy as np
import pytest

from perf.cxxrewrite3_resident_frame import (
    assert_authoritative_not_supported,
    build_resident_frame_from_runner,
    compare_packed,
    compile_native_backend,
    frame_digest,
    load_native_backend,
    python_pack_native_entry,
    stage_pre_env_native_entry,
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
        self.scar = 0.0


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
        self.agent_grid = np.full((6, 6), -1, dtype=np.int32)
        self.agent_grid[1, 2] = 1
        self.agent_grid[3, 4] = 2
        self.grid = np.zeros((6, 6), dtype=np.int8)


class _Runner:
    def __init__(self, cfg: SimConfig | None = None):
        self.sim_cfg = cfg if cfg is not None else SimConfig()
        self.env = _Env()
        self.pop = _Pop()
        self.run_dir = None
        self.agents = [_Agent(1, 0, 10), _Agent(2, 1, 11)]

    def _dev_generation_index(self) -> int:
        return 0

    def _dev_tick_index(self) -> int:
        return 0


def _cfg(**updates: object) -> SimConfig:
    cfg = SimConfig()
    cfg.cxxrewrite_resident_frame_enable = True
    cfg.cxxrewrite_resident_frame_shadow_enable = True
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def test_cxxrewrite3_config_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.cxxrewrite_resident_frame_enable is False
    assert cfg.cxxrewrite_resident_frame_native_packing_enable is False
    assert cfg.cxxrewrite_resident_frame_forbid_per_tick_agent_sync is False


def test_cxxrewrite3_authoritative_modes_fail_closed() -> None:
    cfg = _cfg(cxxrewrite_resident_frame_authoritative_enable=True)

    with pytest.raises(RuntimeError, match="not supported"):
        assert_authoritative_not_supported(cfg)


def test_cxxrewrite3_slot_token_digest_and_dirty_masks() -> None:
    runner = _Runner(_cfg())
    frame = build_resident_frame_from_runner(runner, alive=runner.agents, generation=0, tick=0)
    before = frame_digest(frame, agent_ids=np.asarray([1, 2], dtype=np.int32))

    assert frame.slot_to_agent_id.tolist() == [1, 2, -1, -1]
    assert frame.agent_id_to_slot[1] == 0
    assert frame.slot_tokens.tolist() == [10, 11, 0, 0]
    assert before["count"] == 2

    frame.energy_by_id[1] = np.float32(99.0)
    frame.scalar_dirty_by_id[1] = True
    after = frame_digest(frame, agent_ids=np.asarray([1, 2], dtype=np.int32))

    assert before["digest_sha256"] != after["digest_sha256"]
    assert bool(frame.scalar_dirty_by_id[1]) is True


def test_cxxrewrite3_pre_env_update_primes_cxxrewrite2_frame_without_agent_reads() -> None:
    runner = _Runner(_cfg())
    ids = np.asarray([1, 2], dtype=np.int32)
    record = stage_pre_env_native_entry(
        runner,
        alive=runner.agents,
        agent_ids_np=ids,
        actions_np=np.asarray([0, 4], dtype=np.int32),
        energies=np.asarray([20.0, 21.0], dtype=np.float32),
        healths=np.asarray([8.0, 9.0], dtype=np.float32),
        injuries=np.asarray([0.0, 0.0], dtype=np.float32),
        scars=np.asarray([0.0, 0.0], dtype=np.float32),
    )

    frame = runner._cxxrewrite3_resident_frame
    cxx2 = runner._cxxrewrite2_resident_tick_frame
    summary = runner._cxxrewrite3_resident_frame_summary

    assert record is not None
    assert record["native_input_agent_object_scalar_reads"] == 0
    assert frame.counters["pre_env_array_updates"] == 2
    assert frame.counters["pre_env_scalar_syncs_eliminated"] == 2
    assert np.allclose(cxx2.energy_by_id[ids], np.asarray([20.0, 21.0], dtype=np.float32))
    assert summary["pre_env_array_updates"] == 2
    assert summary["native_input_agent_object_scalar_reads"] == 0


def test_cxxrewrite3_native_pack_matches_python_fixture(tmp_path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    build = compile_native_backend(tmp_path)
    assert build["ok"], build
    runner = _Runner(_cfg())
    ids = np.asarray([1, 2], dtype=np.int32)
    stage_pre_env_native_entry(
        runner,
        alive=runner.agents,
        agent_ids_np=ids,
        actions_np=np.asarray([1, 5], dtype=np.int32),
        energies=np.asarray([11.0, 12.0], dtype=np.float32),
        healths=np.asarray([6.0, 7.0], dtype=np.float32),
        injuries=np.asarray([0.25, 0.50], dtype=np.float32),
        scars=np.asarray([0.0, 0.0], dtype=np.float32),
    )
    frame = runner._cxxrewrite3_resident_frame
    native = load_native_backend(build["library_path"])(frame, ids)
    expected = python_pack_native_entry(frame, ids)

    assert compare_packed(expected, native)["ok"] is True


def test_cxxrewrite3_native_pack_live_stage_records_compare(tmp_path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    build = compile_native_backend(tmp_path)
    assert build["ok"], build
    runner = _Runner(
        _cfg(
            cxxrewrite_resident_frame_native_packing_enable=True,
            cxxrewrite_resident_frame_native_packing_shadow_enable=True,
            cxxrewrite_resident_frame_native_packing_library_path=str(build["library_path"]),
        )
    )
    record = stage_pre_env_native_entry(
        runner,
        alive=runner.agents,
        agent_ids_np=np.asarray([1, 2], dtype=np.int32),
        actions_np=np.asarray([2, 3], dtype=np.int32),
        energies=np.asarray([11.0, 12.0], dtype=np.float32),
        healths=np.asarray([6.0, 7.0], dtype=np.float32),
        injuries=np.asarray([0.25, 0.50], dtype=np.float32),
        scars=np.asarray([0.0, 0.0], dtype=np.float32),
    )

    assert record is not None
    assert record["native_pack_used"] is True
    assert record["native_pack_compare"]["ok"] is True
    assert runner._cxxrewrite3_resident_frame_summary["native_pack_records"] == 1
