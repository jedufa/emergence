from __future__ import annotations

import shutil

import numpy as np
import pytest

from evolution.reproduction import RunnerCoreC1TypedGenomeRow
from perf.runnercore14_native_c1_projection import (
    FLOAT_FIELDS,
    INT_FIELDS,
    build_projection_batch,
    compare_projection,
    compile_native_backend,
    load_native_backend,
    native_api_schema,
    native_backend_status,
    ownership_summary,
    python_project_batch,
)
from simulation.runner_pkg.runner_config import SimConfig


def _payload(i: int) -> dict[str, object]:
    use_gru = bool(i % 2)
    memory_dim = 16 if use_gru else 0
    return {
        "genes": {
            "hidden_size": 64 + int(i % 3),
            "exploration_bias": 0.01 * float(i + 1),
            "aggression_bias": -0.02 * float(i + 1),
            "reproduction_threshold": 30.0 + float(i),
            "mutation_rate": 0.05 + 0.001 * float(i),
            "action_temperature": 1.0 + 0.01 * float(i),
            "use_gru": use_gru,
            "memory_dim": memory_dim,
            "sensor_channel_mask": 255,
            "sensor_radius": 3,
            "introspection_channel_mask": 7,
            "self_regulation_control_mask": 5,
            "self_modification_target_mask": 3,
            "directed_inheritance_channel_mask": 1,
            "drive_inertia": 0.85,
            "drive_noise": 0.02,
            "drive_logit_gain": 1.0 + 0.05 * float(i),
            "pred_lr": 0.08,
            "drive_sp_hunger": 0.55,
            "drive_sp_pain": 0.05,
            "drive_sp_fatigue": 0.15,
            "drive_sp_fear": 0.10,
            "exp_prior_strength": 0.35,
            "exp_prior_noise": 0.05,
            "drive_birth_noise": 0.03,
            "c1_operator_noop_count": int(i),
            "c2_matched_modules": int(i + 2),
            "c2_matched_connections": int(i + 3),
        },
        "modules": [{"latent_pos": [float(i), float(i + 1)]}] if i % 2 else [],
        "connections": [{"from": 1, "to": 2}] if i % 3 == 0 else [],
        "b3_locality_cost_total": 0.1 * float(i),
        "b3_locality_mean_distance": 0.2 * float(i),
        "b3_locality_max_distance": 0.3 * float(i),
    }


def test_runnercore14_config_defaults_off_and_import_without_native() -> None:
    cfg = SimConfig()

    assert cfg.runnercore14_native_c1_projection_enable is False
    assert cfg.runnercore14_native_c1_projection_shadow_enable is False
    assert cfg.runnercore14_native_c1_projection_fail_closed is False
    assert cfg.runnercore14_native_c1_projection_library_path == ""
    assert native_api_schema()["normal_import_requires_native"] is False
    assert native_backend_status()["normal_import_requires_native"] is False
    assert ownership_summary()["native_rng_added"] is False


def test_runnercore14_python_projection_matches_typed_row_scalars() -> None:
    payloads = [_payload(i) for i in range(5)]
    batch = build_projection_batch(payloads)

    assert batch.float_cols.shape == (5, len(FLOAT_FIELDS))
    assert batch.int_cols.shape == (5, len(INT_FIELDS))
    for i, payload in enumerate(payloads):
        row = RunnerCoreC1TypedGenomeRow.from_canonical_payload(payload)
        assert batch.float_cols[i, FLOAT_FIELDS.index("exploration_bias")] == pytest.approx(row.exploration_bias)
        assert batch.float_cols[i, FLOAT_FIELDS.index("reproduction_threshold")] == pytest.approx(
            row.reproduction_threshold
        )
        assert batch.float_cols[i, FLOAT_FIELDS.index("b3_locality_max_distance")] == pytest.approx(
            row.b3_locality_max_distance
        )
        assert int(batch.int_cols[i, INT_FIELDS.index("hidden_size")]) == int(row.hidden_size)
        assert int(batch.int_cols[i, INT_FIELDS.index("use_gru")]) == int(bool(row.use_gru))
        assert int(batch.int_cols[i, INT_FIELDS.index("memory_dim")]) == int(row.memory_dim)
        assert int(batch.int_cols[i, INT_FIELDS.index("modules_count")]) == int(row.modules_count)
        assert int(batch.int_cols[i, INT_FIELDS.index("connections_count")]) == int(row.connections_count)


def test_runnercore14_native_projection_matches_python(tmp_path) -> None:
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    build = compile_native_backend(tmp_path)
    assert build["ok"], build
    batch = build_projection_batch([_payload(i) for i in range(16)])
    expected = python_project_batch(batch)
    actual = load_native_backend(build["library_path"])(batch)

    assert compare_projection(expected, actual)["ok"] is True


def test_runnercore14_projection_compare_tolerates_checksum_roundoff_only() -> None:
    batch = build_projection_batch([_payload(i) for i in range(8)])
    expected = python_project_batch(batch)
    actual = python_project_batch(batch)
    actual["checksum"] = float(actual["checksum"]) + 2.0e-5

    report = compare_projection(expected, actual)

    assert report["ok"] is True
    assert report["max_abs_float_diff"] == 0.0
    assert report["checksum_diff"] == pytest.approx(2.0e-5, abs=1.0e-10)
    assert report["checksum_tolerance"] == pytest.approx(1.0e-4)


def test_runnercore14_projection_compare_localizes_injected_divergence() -> None:
    batch = build_projection_batch([_payload(1), _payload(2)])
    expected = python_project_batch(batch)
    actual = python_project_batch(batch)
    actual["float_cols"] = np.asarray(actual["float_cols"], dtype=np.float64).copy()
    actual["float_cols"][1, 0] += 1.0

    report = compare_projection(expected, actual)

    assert report["ok"] is False
    assert report["first_divergence"]["field"] == "float_cols"
