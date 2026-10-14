from __future__ import annotations

from tools.scripts.eiel_b48_fixed_mlp_policy_birth_architecture_decision_v1 import (
    decide_architecture,
)


def _profile(*, m4: bool, wall_ns: int) -> dict:
    timing = {
        "host_wall_time_ns": {"count": 5, "mean_ns": wall_ns, "median_ns": wall_ns},
        "torch_random_generation_time_ns": {"mean_ns": 180_000_000},
        "torch_compare_time_ns": {"mean_ns": 20_000_000},
        "crossover_pack_time_ns": {"mean_ns": 250_000_000},
    }
    return {
        "schema_version": "eiel_b48_fixed_mlp_policy_birth_refill_profile_v1",
        "artifact_source_commit": "a" * 40,
        "fixture": {"cells": 48, "episodes": 124},
        "output_digest": "b" * 64,
        "timing_ns": timing,
        "build_identity": {
            "m0": {"option_d_m0_telemetry_enabled": True},
            "m4": {"option_d_m4_device_policy_refill_enabled": m4},
        },
    }


def test_architecture_decision_selects_exact_material_m4_win() -> None:
    anatomy = {
        "schema_version": "eiel_b48_fixed_mlp_policy_birth_callback_analysis_v1",
        "timing_ns": {"host_wall_time_ns": {"mean_ns": 481_800_000}},
    }
    decision = decide_architecture(
        anatomy,
        _profile(m4=False, wall_ns=480_000_000),
        _profile(m4=True, wall_ns=20_000_000),
    )
    assert decision["decision"] == (
        "SELECT_EXACT_M4_DEVICE_COMPACT_REFILL_AS_ARCHITECTURE_BASELINE"
    )
    assert decision["evidence"]["byte_exact_control_m4"] is True
    assert decision["evidence"]["mean_speedup"] == 24.0
