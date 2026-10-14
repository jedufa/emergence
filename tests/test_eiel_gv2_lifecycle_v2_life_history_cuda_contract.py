from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine" / "perf"


def _struct_body(source: str, name: str) -> str:
    match = re.search(rf"\bstruct\s+{re.escape(name)}\s*\{{(?P<body>.*?)\n\}};", source, re.DOTALL)
    assert match is not None, f"missing struct {name}"
    return match.group("body")


def test_life_history_additions_preserve_flat_mechanics_aggregate_tail() -> None:
    source = (ENGINE / "cuda_architecture_b_flattened_controller_v1.hpp").read_text(
        encoding="utf-8"
    )
    body = _struct_body(source, "FlatMechanicsConfig")

    assert body.index("int32_t developmental_maturity_age;") < body.index(
        "int32_t life_history_mode;"
    )
    assert body.index("int32_t life_history_max_age_ticks;") < body.index(
        "float drive_hunger_gain;"
    )
    assert body.index("float drive_hunger_gain;") < body.index("int64_t generation;")
    assert "int32_t life_history_max_age_ticks;" in body

    persistent_source = (
        ENGINE / "cuda_architecture_b_persistent_generation.hpp"
    ).read_text(encoding="utf-8")
    persistent_body = _struct_body(persistent_source, "FlatMechanicsConfig")
    assert persistent_body == body


def test_birth_plan_life_history_is_disabled_without_explicit_mode_and_age_pointer() -> None:
    source = (ENGINE / "cuda_architecture_b_persistent_birth_plan.cuh").read_text(
        encoding="utf-8"
    )
    state_body = _struct_body(source, "FlatOrdinaryBirthPlanState")
    config_body = _struct_body(source, "OrdinaryBirthPlanConfig")

    assert state_body.index("const float* reproduction_scale;") < state_body.index(
        "const int32_t* survival_steps;"
    )
    assert config_body.index("bool reproduction_scale_gate_active;") < config_body.index(
        "int32_t life_history_mode;"
    )
    assert "if (config.life_history_mode == 0) {\n        return 1.0f;" in source
    assert "age_ticks < config.maturity_start_ticks" in source
    assert "state.survival_steps == nullptr" in source
    assert "config.life_history_mode == 1" in source


def test_active_mechanics_keeps_legacy_overload_and_non_rng_life_history_costs() -> None:
    source = (ENGINE / "cuda_architecture_b_persistent_active_mechanics.cuh").read_text(
        encoding="utf-8"
    )

    assert "bounded_life_history_ramp_v1" in source
    assert "life_history_senescence_energy_cost" in source
    assert "life_history_senescence_health_cost" in source
    assert "life_history_max_age_ticks > 0" in source

    legacy_overload = source[source.rindex("__device__ __forceinline__ void run_flat_active_mechanics(") :]
    assert "const int32_t episode0_k" in legacy_overload
    assert "developmental_maturity_age,\n        0," in legacy_overload
    assert "pcg64" not in source.lower()


def test_persistent_generation_routes_life_history_to_active_mechanics() -> None:
    source = (ENGINE / "cuda_architecture_b_persistent_generation_kernel.cu").read_text(
        encoding="utf-8"
    )
    call = source[source.index("run_flat_active_mechanics(") :]
    call = call[: call.index(");") + 2]

    expected_order = (
        "mechanics_config.developmental_maturity_age",
        "mechanics_config.life_history_mode",
        "mechanics_config.life_history_maturity_start_ticks",
        "mechanics_config.life_history_maturity_rise_ticks",
        "mechanics_config.life_history_fertility_decline_start_ticks",
        "mechanics_config.life_history_fertility_decline_ticks",
        "mechanics_config.life_history_elder_fertility_floor",
        "mechanics_config.life_history_senescence_start_ticks",
        "mechanics_config.life_history_senescence_ramp_ticks",
        "mechanics_config.life_history_senescence_energy_cost",
        "mechanics_config.life_history_senescence_health_cost",
        "mechanics_config.life_history_max_age_ticks",
        "static_cast<int32_t>(episode0_k)",
    )
    position = -1
    for item in expected_order:
        next_position = call.index(item)
        assert next_position > position
        position = next_position
