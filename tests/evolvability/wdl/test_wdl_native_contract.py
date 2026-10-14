from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from engine.env.env_config import EnvConfig
from engine.env.gridworld_pkg.patch_world import (
    WDL_DELAYED_HAZARD_ARRAY_NAMES,
    _wdl_delayed_hazard_observation_cue_map,
    apply_wdl_delayed_hazard_step,
    initialize_wdl_delayed_hazard_arrays,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = PROJECT_ROOT
HEADER = PROJECT_ROOT / "engine/native_runtime/wdl/wdl_native_contract.hpp"
SOURCE = PROJECT_ROOT / "engine/native_runtime/wdl/wdl_native_contract.cpp"
PROBE = Path(__file__).with_name("wdl_native_contract_probe.cpp")
MANIFEST = PROJECT_ROOT / "engine/native_runtime/wdl/wdl_native_manifest_v1.json"
BUILD_MANIFEST = PROJECT_ROOT / "engine/native_runtime/wdl/wdl_cpu_oracle_build_manifest_v1.json"


@pytest.fixture(scope="module")
def compiled_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is not available")
    output = tmp_path_factory.mktemp("wdl_native_contract") / "probe"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(PROJECT_ROOT),
            str(PROBE),
            str(SOURCE),
            "-o",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return output


@pytest.fixture(scope="module")
def sanitized_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is not available")
    output = tmp_path_factory.mktemp("wdl_native_contract_sanitized") / "probe"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-I",
            str(PROJECT_ROOT),
            str(PROBE),
            str(SOURCE),
            "-o",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return output


def test_wdl_native_contract_strict_cpp_probe(compiled_probe: Path) -> None:
    completed = subprocess.run(
        [str(compiled_probe)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "wdl_native_contract_probe=ok cases=10\n"


@pytest.mark.parametrize(
    ("seed", "event_index", "height", "width"),
    [
        (101, 0, 6, 6),
        (101, 1, 6, 6),
        (202, 0, 9, 7),
        (-17, 12, 3, 11),
        (2**40 + 39, 2**20 + 7, 31, 29),
    ],
)
def test_stable_event_coordinate_matches_python_hashlib(
    compiled_probe: Path,
    seed: int,
    event_index: int,
    height: int,
    width: int,
) -> None:
    material = f"wdl1:{seed}:{event_index}:{height}:{width}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    expected = (
        int.from_bytes(digest[0:8], "big") % height,
        int.from_bytes(digest[8:16], "big") % width,
        int.from_bytes(digest[16:24], "big") & 0x7FFF_FFFF_FFFF_FFFF,
    )
    completed = subprocess.run(
        [
            str(compiled_probe),
            "coordinate",
            str(seed),
            str(event_index),
            str(height),
            str(width),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tuple(int(value) for value in completed.stdout.split()) == expected


def test_native_tick_trace_matches_live_python_schema_v1(compiled_probe: Path) -> None:
    cfg = EnvConfig()
    cfg.patch_world_enable = True
    cfg.wdl_delayed_hazard_enable = True
    cfg.wdl_delayed_hazard_delay_steps = 4
    cfg.wdl_delayed_hazard_cue_duration_steps = 2
    cfg.wdl_delayed_hazard_active_duration_steps = 3
    cfg.wdl_delayed_hazard_event_period_steps = 12
    cfg.wdl_delayed_hazard_radius = 0
    cfg.wdl_delayed_hazard_intensity = 0.25
    env = SimpleNamespace(cfg=cfg, grid=np.zeros((6, 6), dtype=np.int8), base_seed=101)
    arrays = initialize_wdl_delayed_hazard_arrays(cfg, shape=(6, 6))
    for name in WDL_DELAYED_HAZARD_ARRAY_NAMES:
        setattr(env, name, arrays[name])

    material = b"wdl1:101:0:6:6"
    digest = hashlib.sha256(material).digest()
    y = int.from_bytes(digest[0:8], "big") % 6
    x = int.from_bytes(digest[8:16], "big") % 6
    expected: list[str] = []
    for step in range(8):
        metrics = apply_wdl_delayed_hazard_step(env, step)
        expected.append(
            " ".join(
                str(value)
                for value in (
                    step,
                    int(env.patch_delayed_hazard_cue[y, x]),
                    int(env.patch_delayed_hazard_timer[y, x]),
                    int(env.patch_delayed_hazard_active[y, x]),
                    f"{float(env.patch_delayed_hazard_intensity[y, x]):.2f}",
                    int(env.patch_delayed_hazard_age[y, x]),
                    int(env.patch_delayed_hazard_event_id[y, x]),
                    int(env.patch_delayed_hazard_prior_cue[y, x]),
                    int(metrics["wdl_delayed_hazard_event_count"]),
                    int(metrics["wdl_delayed_hazard_cue_cell_count"]),
                    int(metrics["wdl_delayed_hazard_active_cell_count"]),
                )
            )
        )

    completed = subprocess.run(
        [str(compiled_probe), "trace"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == expected


def _event_seed_for_coordinate(height: int, width: int, y: int, x: int) -> int:
    for seed in range(100_000):
        digest = hashlib.sha256(f"wdl1:{seed}:0:{height}:{width}".encode("ascii")).digest()
        if int.from_bytes(digest[0:8], "big") % height == y and int.from_bytes(digest[8:16], "big") % width == x:
            return seed
    raise AssertionError("failed to find bounded edge seed")


def _float32_bits(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    return ",".join(str(int(value)) for value in array.view(np.uint32))


def _integer_values(values: np.ndarray, dtype: np.dtype) -> str:
    return ",".join(str(int(value)) for value in np.asarray(values, dtype=dtype).reshape(-1))


def _live_case_trace(
    *,
    seed: int,
    height: int,
    width: int,
    delay: int,
    cue_duration: int,
    active_duration: int,
    period: int,
    radius: int,
    intensity: float,
    dropout: float,
    steps: int,
) -> list[str]:
    cfg = EnvConfig()
    cfg.patch_world_enable = True
    cfg.wdl_delayed_hazard_enable = True
    cfg.wdl_delayed_hazard_delay_steps = delay
    cfg.wdl_delayed_hazard_cue_duration_steps = cue_duration
    cfg.wdl_delayed_hazard_active_duration_steps = active_duration
    cfg.wdl_delayed_hazard_event_period_steps = period
    cfg.wdl_delayed_hazard_radius = radius
    cfg.wdl_delayed_hazard_intensity = intensity
    cfg.wdl_delayed_hazard_observation_dropout_rate = dropout
    env = SimpleNamespace(cfg=cfg, grid=np.zeros((height, width), dtype=np.int8), base_seed=seed)
    arrays = initialize_wdl_delayed_hazard_arrays(cfg, shape=(height, width))
    for name in WDL_DELAYED_HAZARD_ARRAY_NAMES:
        setattr(env, name, arrays[name])

    lines: list[str] = []
    for step in range(steps):
        metrics = apply_wdl_delayed_hazard_step(env, step)
        observation = _wdl_delayed_hazard_observation_cue_map(env)
        delay_bits = struct.unpack("=Q", struct.pack("=d", float(env.wdl_delayed_hazard_cue_to_active_delay_sum_last)))[0]
        parts = [
            str(step),
            _float32_bits(env.patch_delayed_hazard_cue),
            _integer_values(env.patch_delayed_hazard_timer, np.int32),
            _float32_bits(env.patch_delayed_hazard_active),
            _float32_bits(env.patch_delayed_hazard_intensity),
            _integer_values(env.patch_delayed_hazard_age, np.int32),
            _integer_values(env.patch_delayed_hazard_event_id, np.int64),
            _integer_values(env.patch_delayed_hazard_prior_cue, np.int8),
            _float32_bits(observation),
            ",".join(
                str(value)
                for value in (
                    int(metrics["wdl_delayed_hazard_cue_cell_count"]),
                    int(metrics["wdl_delayed_hazard_active_cell_count"]),
                    int(metrics["wdl_delayed_hazard_event_count"]),
                    int(metrics["wdl_delayed_hazard_cue_without_active_count"]),
                    int(metrics["wdl_delayed_hazard_active_without_prior_cue_count"]),
                    delay_bits,
                    int(env.wdl_delayed_hazard_cue_to_active_delay_count_last),
                )
            ),
        ]
        lines.append("|".join(parts))
    return lines


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(seed=101, height=6, width=6, delay=4, cue_duration=2, active_duration=3, period=12, radius=0, intensity=0.25, dropout=0.0, steps=8),
            id="cue-gap-hazard-boundaries",
        ),
        pytest.param(
            dict(seed=_event_seed_for_coordinate(3, 4, 0, 0), height=3, width=4, delay=3, cue_duration=1, active_duration=5, period=9, radius=2, intensity=0.5, dropout=0.5, steps=7),
            id="top-left-clipped-radius-and-half-dropout",
        ),
        pytest.param(
            dict(seed=_event_seed_for_coordinate(4, 5, 3, 4), height=4, width=5, delay=2, cue_duration=1, active_duration=2, period=5, radius=1, intensity=1.0, dropout=1.0, steps=5),
            id="bottom-right-clipped-radius-and-full-dropout",
        ),
        pytest.param(
            dict(seed=17, height=1, width=1, delay=2, cue_duration=2, active_duration=1, period=4, radius=3, intensity=0.25, dropout=0.0, steps=5),
            id="single-cell-no-gap-cleanup-and-reschedule",
        ),
    ],
)
def test_focused_native_trace_matches_live_python(compiled_probe: Path, case: dict[str, int | float]) -> None:
    command = [
        str(compiled_probe),
        "trace-case",
        *(str(case[name]) for name in (
            "seed", "height", "width", "delay", "cue_duration", "active_duration",
            "period", "radius", "intensity", "dropout", "steps",
        )),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    assert completed.stdout.splitlines() == _live_case_trace(**case)


def test_sanitized_native_probe_is_clean(sanitized_probe: Path) -> None:
    env = {
        "ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1:strict_string_checks=1",
        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
    }
    commands = [
        [str(sanitized_probe)],
        [str(sanitized_probe), "trace-case", "7", "3", "4", "3", "1", "5", "9", "2", "0.5", "0.5", "7"],
    ]
    for index, command in enumerate(commands):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env={**os.environ, **env},
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == ""
        if index == 0:
            assert completed.stdout == "wdl_native_contract_probe=ok cases=10\n"


def test_contract_files_remain_device_neutral() -> None:
    combined = HEADER.read_text(encoding="utf-8") + SOURCE.read_text(encoding="utf-8")
    for forbidden in ("cuda", "torch", "pybind", "Python.h", "PyObject"):
        assert forbidden not in combined


def test_machine_manifest_matches_native_descriptor_contract() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    constants = manifest["header_constants"]
    assert constants == {
        "schema_version": 1,
        "state_descriptor_count": 7,
        "observation_channel_count": 1,
        "scheduled_event_field_count": 6,
        "tick_metric_field_count": 7,
        "generation_summary_field_count": 13,
    }
    assert len(manifest["state"]["fields"]) == constants["state_descriptor_count"]
    assert len(manifest["observation_schema"]["fields"]) == constants["observation_channel_count"]
    assert len(manifest["event_schema"]["native_scheduled_event_fields"]) == constants["scheduled_event_field_count"]
    assert len(manifest["tick_metric_schema"]["fields"]) == constants["tick_metric_field_count"]
    assert len(manifest["generation_summary_schema"]["header_fields"]) == constants["generation_summary_field_count"]
    for location in manifest["source_locations"]:
        path_text = location.split(":", 1)[0]
        assert (PROJECT_ROOT / path_text).is_file(), location


def test_cpu_oracle_build_manifest_is_reproducible(tmp_path: Path) -> None:
    manifest = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["runtime_jit"] is False
    assert manifest["artifact"]["kind"] == "deterministic_static_library"
    for source in manifest["sources"]:
        path = REPOSITORY_ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]

    compiler = shutil.which(manifest["toolchain"]["compiler"]["executable"])
    archiver = shutil.which(manifest["toolchain"]["archiver"]["executable"])
    assert compiler is not None and archiver is not None
    compiler_identity = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    archiver_identity = subprocess.run(
        [archiver, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    expected_compiler_identity = manifest["toolchain"]["compiler"]["identity"]
    expected_archiver_identity = manifest["toolchain"]["archiver"]["identity"]
    if (
        compiler_identity != expected_compiler_identity
        or archiver_identity != expected_archiver_identity
    ):
        pytest.skip(
            "pinned CPU-oracle build-manifest toolchain is not installed: "
            f"compiler={compiler_identity!r} expected={expected_compiler_identity!r}; "
            f"archiver={archiver_identity!r} expected={expected_archiver_identity!r}"
        )

    artifacts: list[bytes] = []
    build_env = {**os.environ, **manifest["build"]["environment"]}
    for pass_name in ("a", "b"):
        output_dir = tmp_path / pass_name
        output_dir.mkdir()
        replacements = {
            "{compiler}": compiler,
            "{archiver}": archiver,
            "{object}": str(output_dir / manifest["artifact"]["object_name"]),
            "{artifact}": str(output_dir / manifest["artifact"]["file_name"]),
        }
        for command in manifest["build"]["commands"]:
            expanded = [replacements.get(token, token) for token in command]
            completed = subprocess.run(
                expanded,
                cwd=REPOSITORY_ROOT,
                env=build_env,
                check=False,
                capture_output=True,
                text=True,
            )
            assert completed.returncode == 0, completed.stderr
        artifact = output_dir / manifest["artifact"]["file_name"]
        artifacts.append(artifact.read_bytes())

    assert artifacts[0] == artifacts[1]
    assert len(artifacts[0]) == manifest["artifact"]["size_bytes"]
    assert hashlib.sha256(artifacts[0]).hexdigest() == manifest["artifact"]["sha256"]
