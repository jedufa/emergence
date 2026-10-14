from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.patch_world import (
    apply_wdl_delayed_hazard_step,
    patch_world_normalized_observation_maps,
)
from simulation.rng_manager import RNGManager


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WDL_ROOT = PROJECT_ROOT / "engine/native_runtime/wdl"
SOURCES = (
    WDL_ROOT / "wdl_native_contract.cpp",
    WDL_ROOT / "wdl_cpu_oracle.cpp",
    WDL_ROOT / "wdl_cpu_oracle_c_api.cpp",
)
SANITIZER_PROBE = Path(__file__).with_name("wdl_cpu_oracle_probe.cpp")

OK = 0
CANCELLED = 1
INVALID_STATE = 3
INSUFFICIENT_CAPACITY = 5


class OracleConfig(ctypes.Structure):
    _fields_ = [
        ("schema_version", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("delay_steps", ctypes.c_int32),
        ("cue_duration_steps", ctypes.c_int32),
        ("active_duration_steps", ctypes.c_int32),
        ("event_period_steps", ctypes.c_int32),
        ("radius", ctypes.c_int32),
        ("intensity", ctypes.c_float),
        ("observation_dropout_rate", ctypes.c_float),
        ("enabled", ctypes.c_uint8),
        ("cue_enabled", ctypes.c_uint8),
        ("observation_enabled", ctypes.c_uint8),
        ("damage_enabled", ctypes.c_uint8),
        ("deterministic_schedule_enabled", ctypes.c_uint8),
        ("view_radius", ctypes.c_int32),
        ("max_scheduled_events", ctypes.c_uint64),
        ("max_damage_events", ctypes.c_uint64),
    ]


class TickMetrics(ctypes.Structure):
    _fields_ = [
        ("cue_cell_count", ctypes.c_int64),
        ("active_cell_count", ctypes.c_int64),
        ("event_count", ctypes.c_int64),
        ("cue_without_active_count", ctypes.c_int64),
        ("active_without_prior_cue_count", ctypes.c_int64),
        ("cue_to_active_delay_sum", ctypes.c_double),
        ("cue_to_active_delay_count", ctypes.c_int64),
    ]


class DamageMetrics(ctypes.Structure):
    _fields_ = [
        ("active_exposure_count", ctypes.c_int64),
        ("agents_damaged", ctypes.c_int64),
        ("deaths", ctypes.c_int64),
        ("damage_total", ctypes.c_double),
        ("damage_max", ctypes.c_float),
    ]


class ScheduledEvent(ctypes.Structure):
    _fields_ = [
        ("event_id", ctypes.c_int64),
        ("abs_step", ctypes.c_int64),
        ("center_y", ctypes.c_int32),
        ("center_x", ctypes.c_int32),
        ("scheduled_cell_count", ctypes.c_int64),
        ("intensity", ctypes.c_float),
    ]


class TickReceipt(ctypes.Structure):
    _fields_ = [
        ("abs_step", ctypes.c_int64),
        ("tick", TickMetrics),
        ("damage", DamageMetrics),
        ("scheduled_event", ScheduledEvent),
        ("observation_rows", ctypes.c_uint64),
        ("observation_width", ctypes.c_uint64),
    ]


class GenerationSummary(ctypes.Structure):
    _fields_ = [
        ("event_count", ctypes.c_int64),
        ("cue_cell_count_mean", ctypes.c_double),
        ("active_cell_count_mean", ctypes.c_double),
        ("agent_cue_exposure_count", ctypes.c_int64),
        ("agent_active_exposure_count", ctypes.c_int64),
        ("damage_total", ctypes.c_double),
        ("damage_max", ctypes.c_float),
        ("agents_damaged", ctypes.c_int64),
        ("cue_to_active_delay_mean", ctypes.c_double),
        ("cue_without_active_count", ctypes.c_int64),
        ("active_without_prior_cue_count", ctypes.c_int64),
        ("enabled", ctypes.c_int64),
        ("tick_count", ctypes.c_int64),
    ]


FIELD_DTYPES = {
    0: np.dtype(np.float32),
    1: np.dtype(np.int32),
    2: np.dtype(np.float32),
    3: np.dtype(np.float32),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.int8),
}

PYTHON_FIELDS = {
    0: "patch_delayed_hazard_cue",
    1: "patch_delayed_hazard_timer",
    2: "patch_delayed_hazard_active",
    3: "patch_delayed_hazard_intensity",
    4: "patch_delayed_hazard_age",
    5: "patch_delayed_hazard_event_id",
    6: "patch_delayed_hazard_prior_cue",
}


def _configure_library(lib: ctypes.CDLL) -> None:
    handle = ctypes.c_void_p
    lib.eiel_wdl_cpu_oracle_abi_version_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_create_v1.argtypes = [
        ctypes.POINTER(OracleConfig), ctypes.c_int64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32)
    ]
    lib.eiel_wdl_cpu_oracle_create_v1.restype = handle
    lib.eiel_wdl_cpu_oracle_destroy_v1.argtypes = [handle]
    lib.eiel_wdl_cpu_oracle_set_agents_v1.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    lib.eiel_wdl_cpu_oracle_set_agents_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_step_v1.argtypes = [
        handle,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
        ctypes.c_uint8,
        ctypes.POINTER(TickReceipt),
    ]
    lib.eiel_wdl_cpu_oracle_step_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_begin_generation_v1.argtypes = [handle, ctypes.c_uint8]
    lib.eiel_wdl_cpu_oracle_begin_generation_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_observation_count_v1.argtypes = [handle]
    lib.eiel_wdl_cpu_oracle_observation_count_v1.restype = ctypes.c_uint64
    lib.eiel_wdl_cpu_oracle_observation_row_count_v1.argtypes = [handle]
    lib.eiel_wdl_cpu_oracle_observation_row_count_v1.restype = ctypes.c_uint64
    lib.eiel_wdl_cpu_oracle_copy_observation_agent_ids_v1.argtypes = [
        handle, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64
    ]
    lib.eiel_wdl_cpu_oracle_copy_observation_agent_ids_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_copy_observation_v1.argtypes = [
        handle, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64
    ]
    lib.eiel_wdl_cpu_oracle_copy_observation_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_state_field_bytes_v1.argtypes = [handle, ctypes.c_int32]
    lib.eiel_wdl_cpu_oracle_state_field_bytes_v1.restype = ctypes.c_uint64
    lib.eiel_wdl_cpu_oracle_copy_state_field_v1.argtypes = [
        handle, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint64
    ]
    lib.eiel_wdl_cpu_oracle_copy_state_field_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_copy_agents_v1.argtypes = [
        handle,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    lib.eiel_wdl_cpu_oracle_copy_agents_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_generation_summary_v1.argtypes = [
        handle, ctypes.POINTER(GenerationSummary)
    ]
    lib.eiel_wdl_cpu_oracle_generation_summary_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_scheduled_event_count_v1.argtypes = [handle]
    lib.eiel_wdl_cpu_oracle_scheduled_event_count_v1.restype = ctypes.c_uint64
    lib.eiel_wdl_cpu_oracle_damage_event_count_v1.argtypes = [handle]
    lib.eiel_wdl_cpu_oracle_damage_event_count_v1.restype = ctypes.c_uint64
    lib.eiel_wdl_cpu_oracle_checkpoint_size_v1.argtypes = [handle]
    lib.eiel_wdl_cpu_oracle_checkpoint_size_v1.restype = ctypes.c_uint64
    lib.eiel_wdl_cpu_oracle_write_checkpoint_v1.argtypes = [
        handle, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)
    ]
    lib.eiel_wdl_cpu_oracle_write_checkpoint_v1.restype = ctypes.c_int32
    lib.eiel_wdl_cpu_oracle_restore_checkpoint_v1.argtypes = [
        handle, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64
    ]
    lib.eiel_wdl_cpu_oracle_restore_checkpoint_v1.restype = ctypes.c_int32


@pytest.fixture(scope="module")
def oracle_lib(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    output = tmp_path_factory.mktemp("wdl_cpu_oracle") / "libwdl_cpu_oracle.so"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            *(str(source) for source in SOURCES),
            "-o",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    lib = ctypes.CDLL(str(output))
    _configure_library(lib)
    assert lib.eiel_wdl_cpu_oracle_abi_version_v1() == 1
    return lib


def _config(*, dropout: float = 0.0, max_scheduled_events: int = 32) -> OracleConfig:
    return OracleConfig(
        1, 6, 6, 4, 2, 3, 12, 0, 0.25, dropout,
        1, 1, 1, 1, 1, 1, max_scheduled_events, 64,
    )


def _python_env(*, seed: int, dropout: float) -> GridWorld:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.wdl_delayed_hazard_enable = True
    cfg.wdl_delayed_hazard_observation_enable = True
    cfg.wdl_delayed_hazard_delay_steps = 4
    cfg.wdl_delayed_hazard_cue_duration_steps = 2
    cfg.wdl_delayed_hazard_active_duration_steps = 3
    cfg.wdl_delayed_hazard_event_period_steps = 12
    cfg.wdl_delayed_hazard_radius = 0
    cfg.wdl_delayed_hazard_intensity = 0.25
    cfg.wdl_delayed_hazard_observation_dropout_rate = dropout
    rng = RNGManager(global_seed=seed)
    return GridWorld(cfg, seed=seed, run_dir=None, rng=rng.env_np)


class Oracle:
    def __init__(self, lib: ctypes.CDLL, cfg: OracleConfig, *, seed: int, agents: int = 1) -> None:
        self.lib = lib
        self.agent_count = agents
        status = ctypes.c_int32(-1)
        self.handle = lib.eiel_wdl_cpu_oracle_create_v1(
            ctypes.byref(cfg), seed, agents, ctypes.byref(status)
        )
        assert self.handle and status.value == OK

    def close(self) -> None:
        if self.handle:
            self.lib.eiel_wdl_cpu_oracle_destroy_v1(self.handle)
            self.handle = None

    def set_agents(
        self,
        positions: list[tuple[int, int]],
        *,
        health: list[float] | None = None,
        scales: list[float] | None = None,
        alive_values: list[int] | None = None,
    ) -> None:
        count = len(positions)
        alive = (ctypes.c_uint8 * count)(*(alive_values or [1] * count))
        ys = (ctypes.c_int32 * count)(*(p[0] for p in positions))
        xs = (ctypes.c_int32 * count)(*(p[1] for p in positions))
        hp = (ctypes.c_float * count)(*(health or [1.0] * count))
        scale = (ctypes.c_float * count)(*(scales or [1.0] * count))
        assert self.lib.eiel_wdl_cpu_oracle_set_agents_v1(
            self.handle, alive, ys, xs, hp, scale, count
        ) == OK

    def step(
        self,
        step: int,
        *,
        positions: list[tuple[int, int]] | None = None,
        cancel: bool = False,
        receipt: TickReceipt | None = None,
    ) -> tuple[int, TickReceipt]:
        out = TickReceipt() if receipt is None else receipt
        if positions is None:
            ys = xs = None
            count = 0
        else:
            count = len(positions)
            ys = (ctypes.c_int32 * count)(*(p[0] for p in positions))
            xs = (ctypes.c_int32 * count)(*(p[1] for p in positions))
        status = self.lib.eiel_wdl_cpu_oracle_step_v1(
            self.handle, step, ys, xs, count, int(cancel), ctypes.byref(out)
        )
        return status, out

    def state(self, field: int) -> np.ndarray:
        dtype = FIELD_DTYPES[field]
        size = self.lib.eiel_wdl_cpu_oracle_state_field_bytes_v1(self.handle, field)
        assert size == 36 * dtype.itemsize
        output = np.empty(36, dtype=dtype)
        assert self.lib.eiel_wdl_cpu_oracle_copy_state_field_v1(
            self.handle, field, output.ctypes.data_as(ctypes.c_void_p), size
        ) == OK
        return output.reshape(6, 6)

    def observation(self) -> np.ndarray:
        count = self.lib.eiel_wdl_cpu_oracle_observation_count_v1(self.handle)
        rows = self.lib.eiel_wdl_cpu_oracle_observation_row_count_v1(self.handle)
        output = np.empty(count, dtype=np.float32)
        assert self.lib.eiel_wdl_cpu_oracle_copy_observation_v1(
            self.handle, output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), count
        ) == OK
        return output.reshape(rows, -1)

    def observation_agent_ids(self) -> np.ndarray:
        rows = self.lib.eiel_wdl_cpu_oracle_observation_row_count_v1(self.handle)
        output = np.empty(rows, dtype=np.int32)
        assert self.lib.eiel_wdl_cpu_oracle_copy_observation_agent_ids_v1(
            self.handle, output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), rows
        ) == OK
        return output

    def agents(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        alive = np.empty(self.agent_count, dtype=np.uint8)
        y = np.empty(self.agent_count, dtype=np.int32)
        x = np.empty(self.agent_count, dtype=np.int32)
        health = np.empty(self.agent_count, dtype=np.float32)
        assert self.lib.eiel_wdl_cpu_oracle_copy_agents_v1(
            self.handle,
            alive.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            health.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.agent_count,
        ) == OK
        return alive, y, x, health

    def checkpoint(self) -> bytes:
        size = self.lib.eiel_wdl_cpu_oracle_checkpoint_size_v1(self.handle)
        output = (ctypes.c_uint8 * size)()
        written = ctypes.c_uint64(0)
        assert self.lib.eiel_wdl_cpu_oracle_write_checkpoint_v1(
            self.handle, output, size, ctypes.byref(written)
        ) == OK
        assert written.value == size
        return bytes(output)

    def restore(self, payload: bytes) -> int:
        source = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        return self.lib.eiel_wdl_cpu_oracle_restore_checkpoint_v1(
            self.handle, source, len(payload)
        )

    def summary(self) -> GenerationSummary:
        output = GenerationSummary()
        assert self.lib.eiel_wdl_cpu_oracle_generation_summary_v1(
            self.handle, ctypes.byref(output)
        ) == OK
        return output


def _edge_window(cue_map: np.ndarray, y: int, x: int, radius: int = 1) -> np.ndarray:
    side = 2 * radius + 1
    output = np.zeros((side, side), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = y + dy, x + dx
            if 0 <= yy < cue_map.shape[0] and 0 <= xx < cue_map.shape[1]:
                output[dy + radius, dx + radius] = cue_map[yy, xx]
    return output.reshape(-1)


@pytest.mark.parametrize("dropout", [0.0, 1.0])
def test_ctypes_oracle_matches_live_python_phase_state_and_edge_observation(
    oracle_lib: ctypes.CDLL,
    dropout: float,
) -> None:
    seed = 101
    env = _python_env(seed=seed, dropout=dropout)
    oracle = Oracle(oracle_lib, _config(dropout=dropout), seed=seed)
    oracle.set_agents([(0, 0)])
    try:
        for step in range(8):
            metrics = apply_wdl_delayed_hazard_step(env, step)
            status, receipt = oracle.step(step)
            assert status == OK
            for field, python_name in PYTHON_FIELDS.items():
                np.testing.assert_array_equal(oracle.state(field), getattr(env, python_name), err_msg=python_name)
            assert receipt.tick.cue_cell_count == int(metrics["wdl_delayed_hazard_cue_cell_count"])
            assert receipt.tick.active_cell_count == int(metrics["wdl_delayed_hazard_active_cell_count"])
            assert receipt.tick.event_count == int(metrics["wdl_delayed_hazard_event_count"])
            cue_map = patch_world_normalized_observation_maps(env)[-1]
            np.testing.assert_array_equal(oracle.observation()[0], _edge_window(cue_map, 0, 0))
    finally:
        oracle.close()


def test_ctypes_oracle_damage_is_post_action_and_not_early(oracle_lib: ctypes.CDLL) -> None:
    env = _python_env(seed=101, dropout=0.0)
    apply_wdl_delayed_hazard_step(env, 0)
    event_y, event_x = np.argwhere(env.patch_delayed_hazard_cue > 0.0)[0].tolist()
    oracle = Oracle(oracle_lib, _config(), seed=101)
    oracle.set_agents([(event_y, event_x)], health=[1.0])
    try:
        for step in range(4):
            status, receipt = oracle.step(step)
            assert status == OK
            assert receipt.damage.damage_total == 0.0
            assert oracle.agents()[3][0] == pytest.approx(1.0)
        status, receipt = oracle.step(4, positions=[(event_y, event_x)])
        assert status == OK
        assert receipt.damage.active_exposure_count == 1
        assert receipt.damage.damage_total == pytest.approx(0.25)
        assert oracle.agents()[3][0] == pytest.approx(0.75)
    finally:
        oracle.close()


def test_ctypes_oracle_failure_and_cancel_are_transactional(oracle_lib: ctypes.CDLL) -> None:
    oracle = Oracle(oracle_lib, _config(), seed=101)
    oracle.set_agents([(0, 0)])
    try:
        before = oracle.checkpoint()
        sentinel = TickReceipt()
        sentinel.abs_step = 777
        status, receipt = oracle.step(0, cancel=True, receipt=sentinel)
        assert status == CANCELLED
        assert receipt.abs_step == 777
        assert oracle.checkpoint() == before

        status, receipt = oracle.step(0, positions=[(6, 0)], receipt=sentinel)
        assert status == INVALID_STATE
        assert receipt.abs_step == 777
        assert oracle.checkpoint() == before
    finally:
        oracle.close()

    capacity_oracle = Oracle(oracle_lib, _config(max_scheduled_events=0), seed=101)
    capacity_oracle.set_agents([(0, 0)])
    try:
        before = capacity_oracle.checkpoint()
        status, _ = capacity_oracle.step(0)
        assert status == INSUFFICIENT_CAPACITY
        assert capacity_oracle.checkpoint() == before
    finally:
        capacity_oracle.close()


@pytest.mark.parametrize("checkpoint_step", [0, 2, 4])
def test_ctypes_oracle_full_checkpoint_continuation(
    oracle_lib: ctypes.CDLL,
    checkpoint_step: int,
) -> None:
    uninterrupted = Oracle(oracle_lib, _config(dropout=0.35), seed=7070, agents=2)
    prefix = Oracle(oracle_lib, _config(dropout=0.35), seed=7070, agents=2)
    restored = Oracle(oracle_lib, _config(dropout=0.35), seed=-1, agents=2)
    positions = [(0, 0), (5, 5)]
    for oracle in (uninterrupted, prefix, restored):
        oracle.set_agents(positions, health=[1.0, 0.5], scales=[1.0, 0.5])
    try:
        for step in range(8):
            assert uninterrupted.step(step)[0] == OK
        for step in range(checkpoint_step + 1):
            assert prefix.step(step)[0] == OK
        assert restored.restore(prefix.checkpoint()) == OK
        for step in range(checkpoint_step + 1, 8):
            assert restored.step(step)[0] == OK
        assert restored.checkpoint() == uninterrupted.checkpoint()
        np.testing.assert_array_equal(restored.observation(), uninterrupted.observation())
        for field in FIELD_DTYPES:
            np.testing.assert_array_equal(restored.state(field), uninterrupted.state(field))
    finally:
        uninterrupted.close()
        prefix.close()
        restored.close()


def test_cpu_oracle_asan_ubsan_probe(tmp_path: Path) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is unavailable")
    output = tmp_path / "wdl_cpu_oracle_sanitizer_probe"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(PROJECT_ROOT),
            str(SANITIZER_PROBE),
            *(str(source) for source in SOURCES),
            "-o",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    executed = subprocess.run(
        [str(output)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1", "UBSAN_OPTIONS": "halt_on_error=1"},
    )
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout == "wdl_cpu_oracle_sanitizer_probe=ok\n"


def test_begin_generation_preserves_authority_resets_summary_and_controls_journals(
    oracle_lib: ctypes.CDLL,
) -> None:
    oracle = Oracle(oracle_lib, _config(), seed=101, agents=2)
    oracle.set_agents([(0, 0), (-1, -1)], health=[1.0, 0.0], alive_values=[1, 0])
    try:
        assert oracle.step(0)[0] == OK
        event_cell = np.argwhere(oracle.state(0) > 0.0)[0]
        event_position = (int(event_cell[0]), int(event_cell[1]))
        oracle.set_agents([event_position, (-1, -1)], health=[0.1, 0.0], alive_values=[1, 0])

        np.testing.assert_array_equal(oracle.observation_agent_ids(), np.asarray([0], dtype=np.int32))
        assert oracle.observation().shape == (1, 9)
        for step in range(1, 5):
            assert oracle.step(step)[0] == OK
        assert oracle.lib.eiel_wdl_cpu_oracle_scheduled_event_count_v1(oracle.handle) == 1
        assert oracle.lib.eiel_wdl_cpu_oracle_damage_event_count_v1(oracle.handle) == 1
        assert oracle.summary().agents_damaged == 1

        state_before = {field: oracle.state(field).copy() for field in FIELD_DTYPES}
        agents_before = tuple(value.copy() for value in oracle.agents())
        assert oracle.lib.eiel_wdl_cpu_oracle_begin_generation_v1(oracle.handle, 0) == OK
        for field, expected in state_before.items():
            np.testing.assert_array_equal(oracle.state(field), expected)
        for actual, expected in zip(oracle.agents(), agents_before, strict=True):
            np.testing.assert_array_equal(actual, expected)
        summary = oracle.summary()
        assert summary.tick_count == 0
        assert summary.event_count == 0
        assert summary.damage_total == 0.0
        assert summary.agents_damaged == 0
        assert summary.enabled == 1
        assert oracle.lib.eiel_wdl_cpu_oracle_scheduled_event_count_v1(oracle.handle) == 1
        assert oracle.lib.eiel_wdl_cpu_oracle_damage_event_count_v1(oracle.handle) == 1

        before_invalid = oracle.checkpoint()
        assert oracle.lib.eiel_wdl_cpu_oracle_begin_generation_v1(oracle.handle, 2) == INVALID_STATE
        assert oracle.checkpoint() == before_invalid

        # A successful step at the next absolute index proves schedule ownership
        # was not rewound by begin_generation.
        assert oracle.step(5)[0] == OK
        assert oracle.lib.eiel_wdl_cpu_oracle_begin_generation_v1(oracle.handle, 1) == OK
        assert oracle.lib.eiel_wdl_cpu_oracle_scheduled_event_count_v1(oracle.handle) == 0
        assert oracle.lib.eiel_wdl_cpu_oracle_damage_event_count_v1(oracle.handle) == 0
    finally:
        oracle.close()
