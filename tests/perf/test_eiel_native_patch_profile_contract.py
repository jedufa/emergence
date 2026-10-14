from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from env.env_config import EnvConfig
from env.gridworld_pkg.constants import EMPTY
from native_runtime.patch_profile import (
    PATCH_PROFILE_P0,
    PATCH_PROFILE_P1,
    PATCH_PROFILE_P2,
    classify_patch_profile,
)
from perf.cxxbackbone10_resident_state_owner import (
    PATCH_REQUIRED_ENV_ARRAYS,
    _patch_import_arrays,
    _patch_profile_metadata,
)
from simulation.runner_pkg.runner_config import SimConfig


def _configs() -> tuple[SimConfig, EnvConfig]:
    sim_cfg = SimConfig(
        seed=17,
        generations=1,
        steps_per_generation=2,
        device="cpu",
        bank_device="cpu",
    )
    sim_cfg.native_runtime_fixed_policy_family = "mlp"
    sim_cfg.native_runtime_allow_runtime_jit_build = False
    sim_cfg.shock_start_prob = 0.0
    sim_cfg.shock_period = 0
    sim_cfg.shock_duration = 0
    sim_cfg.pressure_autopilot_allow_shock_enable = False

    env_cfg = EnvConfig(width=4, height=4, view_radius=1)
    env_cfg.patch_world_enable = True
    env_cfg.patch_world_schema_version = 1
    env_cfg.patch_world_observation_enable = False
    env_cfg.patch_world_observation_schema_version = 1
    env_cfg.patch_world_observation_stacked_gather_enable = False
    env_cfg.patch_world_observation_numba_gather_enable = False
    env_cfg.patch_world_diagnostics_enable = True
    env_cfg.patch_world_lineage_descriptors_enable = True
    env_cfg.patch_world_descriptor_schema_version = 1
    env_cfg.patch_world_waste_consequence_enable = False
    env_cfg.patch_movement_cost_apply_enable = False
    env_cfg.reproduce_requires_mate = False
    return sim_cfg, env_cfg


@pytest.mark.parametrize(
    ("profile", "observation", "consequence", "movement"),
    [
        (PATCH_PROFILE_P0, False, False, False),
        (PATCH_PROFILE_P1, True, False, False),
        (PATCH_PROFILE_P2, True, True, True),
    ],
)
def test_finite_patch_profiles_are_classified_exactly(
    profile: str,
    observation: bool,
    consequence: bool,
    movement: bool,
) -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.patch_world_observation_enable = observation
    env_cfg.patch_world_waste_consequence_enable = consequence
    env_cfg.patch_movement_cost_apply_enable = movement

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == profile
    assert decision.unsupported_reasons == ()
    assert decision.to_dict()["profile"] == profile


@pytest.mark.parametrize(
    ("target", "field", "value", "reason"),
    [
        ("env", "patch_world_enable", False, "patch_world_gate_required"),
        ("env", "patch_world_schema_version", 2, "patch_world_schema_1_required"),
        ("sim", "device", "cuda", "cuda_or_non_cpu_tensor_runtime_not_supported"),
        ("sim", "bank_device", "cuda", "cuda_or_non_cpu_tensor_runtime_not_supported"),
        ("sim", "native_runtime_fixed_policy_family", "gru", "unsupported_gru_or_recurrent_policy"),
        ("sim", "native_runtime_allow_runtime_jit_build", True, "runtime_jit_not_allowed"),
        ("env", "jungle_predator_pressure_enable", True, "unsupported_active_jungle"),
        ("env", "wdl_delayed_hazard_enable", True, "unsupported_wdl_or_delayed_world"),
        ("env", "tier4_juvenile_vulnerability_enable", True, "tier4_juvenile_vulnerability_not_supported"),
        ("env", "respec0b_resource_specialization_enable", True, "unsupported_respec"),
        ("env", "mosaic0a_depletion_recovery_enable", True, "unsupported_mosaic"),
        ("env", "predrisk_fixed1_enable", True, "unsupported_predrisk"),
        ("env", "corridor_cp0_enable", True, "unsupported_corridor"),
        ("env", "habacclim_enable", True, "unsupported_habacclim"),
        ("env", "roz2a_patch_niche_enable", True, "unsupported_roz2a"),
        ("env", "spatial_ecology_enable", True, "unsupported_spatial_ecology"),
        ("env", "spatial_droughts_enable", True, "unsupported_spatial_droughts"),
        ("sim", "shock_start_prob", 0.1, "native_shock_schedule_not_supported"),
        ("env", "patch_scent_birth_emit", 0.1, "unsupported_patch_lifecycle_emissions"),
        ("env", "patch_waste_density_emit_scale", 0.1, "unsupported_patch_density_waste_emission"),
        ("env", "patch_waste_move_emit", 0.1, "unsupported_patch_lifecycle_emissions"),
    ],
)
def test_named_unsupported_surfaces_fail_closed(
    target: str,
    field: str,
    value: object,
    reason: str,
) -> None:
    sim_cfg, env_cfg = _configs()
    setattr(sim_cfg if target == "sim" else env_cfg, field, value)

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is False
    assert decision.profile is None
    assert reason in decision.unsupported_reasons


def test_p1_requires_exact_observation_and_descriptor_contract() -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.patch_world_observation_enable = True
    env_cfg.patch_world_observation_schema_version = 2
    env_cfg.patch_world_observation_stacked_gather_enable = True
    env_cfg.patch_world_observation_numba_gather_enable = True
    env_cfg.patch_world_diagnostics_enable = False
    env_cfg.patch_world_lineage_descriptors_enable = False
    env_cfg.patch_world_descriptor_schema_version = 2

    decision = classify_patch_profile(sim_cfg, env_cfg)

    assert decision.profile is None
    assert "patch_observation_schema_1_required" in decision.unsupported_reasons
    assert "patch_observation_stacked_gather_not_supported" in decision.unsupported_reasons
    assert "patch_observation_numba_gather_not_supported" in decision.unsupported_reasons
    assert "patch_world_diagnostics_required" in decision.unsupported_reasons
    assert "patch_world_lineage_descriptors_required" in decision.unsupported_reasons
    assert "patch_descriptor_schema_1_required" in decision.unsupported_reasons


def test_partial_p2_toggle_combination_is_not_reclassified_as_p1() -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.patch_world_observation_enable = True
    env_cfg.patch_world_waste_consequence_enable = True

    decision = classify_patch_profile(sim_cfg, env_cfg)

    assert decision.profile is None
    assert decision.unsupported_reasons == ("unsupported_patch_profile_combination",)


def test_resident_recurrent_rows_are_rejected() -> None:
    sim_cfg, env_cfg = _configs()
    runner = SimpleNamespace(_policy_use_gru=[False, True])

    decision = classify_patch_profile(sim_cfg, env_cfg, runner)

    assert decision.profile is None
    assert "unsupported_gru_or_recurrent_policy" in decision.unsupported_reasons
    assert (
        "fixed_mlp_policy_family_gru_rows_not_supported"
        in decision.unsupported_reasons
    )


def test_live_population_policy_rows_are_inspected_without_legacy_row_alias() -> None:
    sim_cfg, env_cfg = _configs()
    runner = SimpleNamespace(
        pop=SimpleNamespace(
            alive_agents=lambda: [
                SimpleNamespace(genome=SimpleNamespace(use_gru=False)),
                SimpleNamespace(genome=SimpleNamespace(use_gru=True)),
            ]
        )
    )

    decision = classify_patch_profile(sim_cfg, env_cfg, runner)

    assert decision.profile is None
    assert "unsupported_gru_or_recurrent_policy" in decision.unsupported_reasons


def test_inactive_gru_math_flag_does_not_override_fixed_mlp_rows() -> None:
    sim_cfg, env_cfg = _configs()
    sim_cfg.cxxbackbone3_policy_aten_gru_authority_enable = True
    runner = SimpleNamespace(_policy_use_gru=[False, False])

    decision = classify_patch_profile(sim_cfg, env_cfg, runner)

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P0


def test_mate_required_reproduction_rng_is_outside_patch_envelope() -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.reproduce_requires_mate = True

    decision = classify_patch_profile(sim_cfg, env_cfg)

    assert decision.profile is None
    assert "unsupported_reproduction_mode_mate_required" in decision.unsupported_reasons
    assert (
        "mate_required_exact_recombination_rng_not_supported"
        in decision.unsupported_reasons
    )


def test_unknown_or_malformed_inspection_never_defaults_to_p0() -> None:
    sim_cfg, env_cfg = _configs()
    del sim_cfg.native_runtime_fixed_policy_family
    env_cfg.patch_world_enable = 1  # type: ignore[assignment]

    decision = classify_patch_profile(sim_cfg, env_cfg)

    assert decision.supported is False
    assert decision.profile is None
    assert (
        "missing_inspection_field:sim_cfg.native_runtime_fixed_policy_family"
        in decision.unsupported_reasons
    )
    assert (
        "malformed_inspection_field:env_cfg.patch_world_enable"
        in decision.unsupported_reasons
    )


@pytest.mark.parametrize(
    "field",
    ("g1_unknown_feature_enable", "patch_observations_enable"),
)
def test_undeclared_configuration_fields_fail_closed(field: str) -> None:
    sim_cfg, env_cfg = _configs()
    setattr(env_cfg, field, True)

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is False
    assert decision.profile is None
    assert (
        f"unknown_configuration_field:env_cfg.{field}"
        in decision.unsupported_reasons
    )


def test_tw_rngv2_campaign_fields_are_admitted_on_p2() -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.patch_world_observation_enable = True
    env_cfg.patch_world_waste_consequence_enable = True
    env_cfg.patch_movement_cost_apply_enable = True
    sim_cfg.rng_semantics_version = 2
    sim_cfg.world_uid = 1
    sim_cfg.rng_namespace_key64 = 0xCA05117E450A51C0

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P2
    assert decision.unsupported_reasons == ()

    sim_cfg.rng_semantics_version = 3
    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")
    assert decision.supported is False
    assert (
        "malformed_inspection_field:sim_cfg.rng_semantics_version"
        in decision.unsupported_reasons
    )

    sim_cfg.rng_semantics_version = 2
    sim_cfg.rng_namespace_key64 = 0
    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")
    assert decision.supported is False
    assert (
        "malformed_inspection_field:sim_cfg.rng_namespace_key64"
        in decision.unsupported_reasons
    )


def test_named_runtime_added_base_fields_are_admitted_only_when_inactive() -> None:
    sim_cfg, env_cfg = _configs()
    sim_cfg.shock_pulsed = False
    sim_cfg.memory_action_window_instrumentation_enable = False
    sim_cfg.memory_action_window_emit_jsonl = False
    sim_cfg.memory_action_window_include_lineage = False
    sim_cfg.memory_action_window_include_memory_state = False
    sim_cfg.memory_action_window_max_events_per_run = 0
    sim_cfg.native_flat_reporting_schema_version = "eiel_flat_native_reporting_v1"

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P0

    sim_cfg.memory_action_window_emit_jsonl = True
    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")
    assert decision.supported is False
    assert "memory_action_window_not_supported" in decision.unsupported_reasons

    sim_cfg.memory_action_window_emit_jsonl = False
    sim_cfg.native_flat_reporting_schema_version = "unknown"
    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")
    assert decision.supported is False
    assert "reporting_schema_version_mismatch" in decision.unsupported_reasons


def test_private_runtime_state_is_not_treated_as_user_configuration() -> None:
    sim_cfg, env_cfg = _configs()
    sim_cfg._mutation_weather_generation = 1

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P0


def _add_frozen_production_lifecycle_fields(sim_cfg: SimConfig) -> None:
    sim_cfg.ordinary_births_enabled = False
    sim_cfg.generation_turnover_enabled = True
    sim_cfg.population_size = 64
    sim_cfg.cull_fraction = 0.5
    sim_cfg.parent_fraction = 1.0
    sim_cfg.turnover_selection_mode = 1
    sim_cfg.turnover_selection_seed = 117


def _add_hybrid_production_lifecycle_fields(sim_cfg: SimConfig) -> None:
    sim_cfg.ordinary_births_enabled = True
    sim_cfg.generation_turnover_enabled = False
    sim_cfg.population_size = 64
    sim_cfg.cull_fraction = 0.0
    sim_cfg.parent_fraction = 1.0
    sim_cfg.lifecycle_mode = 1
    sim_cfg.turnover_selection_mode = 1
    sim_cfg.turnover_selection_seed = 117


def test_frozen_production_lifecycle_fields_are_admitted() -> None:
    sim_cfg, env_cfg = _configs()
    _add_frozen_production_lifecycle_fields(sim_cfg)

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P0
    assert decision.unsupported_reasons == ()


@pytest.mark.parametrize("population_size", (64, 128, 256, 512))
def test_frozen_turnover_lifecycle_admits_current_turnover_populations(
    population_size: int,
) -> None:
    sim_cfg, env_cfg = _configs()
    _add_frozen_production_lifecycle_fields(sim_cfg)
    sim_cfg.population_size = population_size

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P0
    assert decision.unsupported_reasons == ()


def test_hybrid_production_lifecycle_fields_are_admitted() -> None:
    sim_cfg, env_cfg = _configs()
    _add_hybrid_production_lifecycle_fields(sim_cfg)

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P0
    assert decision.unsupported_reasons == ()


def test_production_lifecycle_ordinary_births_remain_inside_profile() -> None:
    sim_cfg, env_cfg = _configs()
    _add_frozen_production_lifecycle_fields(sim_cfg)
    sim_cfg.ordinary_births_enabled = True

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P0
    assert decision.unsupported_reasons == ()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ordinary_births_enabled", 1),
        ("generation_turnover_enabled", False),
        ("population_size", 96),
        ("cull_fraction", 0.25),
        ("parent_fraction", 0.5),
        ("lifecycle_mode", 2),
        ("turnover_selection_mode", 2),
        ("turnover_selection_seed", -1),
    ),
)
def test_production_lifecycle_fields_fail_closed_when_profile_drifts(
    field: str,
    value: object,
) -> None:
    sim_cfg, env_cfg = _configs()
    _add_frozen_production_lifecycle_fields(sim_cfg)
    setattr(sim_cfg, field, value)

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is False
    assert decision.profile is None
    assert (
        "production_lifecycle_frozen_profile_required"
        in decision.unsupported_reasons
    )


@pytest.mark.parametrize("profile", (PATCH_PROFILE_P0, PATCH_PROFILE_P1))
def test_fertility_effect_is_rejected_outside_p2(profile: str) -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.patch_world_observation_enable = profile == PATCH_PROFILE_P1
    env_cfg.patch_waste_fertility_effect_enable = True

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is False
    assert decision.profile is None
    assert "p2_fertility_effect_requires_p2_profile" in decision.unsupported_reasons


def test_non_mapping_inspection_fails_closed_without_exception() -> None:
    _, env_cfg = _configs()

    decision = classify_patch_profile(object(), env_cfg)

    assert decision.supported is False
    assert decision.profile is None
    assert decision.unsupported_reasons == ("malformed_inspection:sim_cfg",)


def test_p2_optional_fertility_effect_remains_inside_frozen_profile() -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.patch_world_observation_enable = True
    env_cfg.patch_world_waste_consequence_enable = True
    env_cfg.patch_movement_cost_apply_enable = True
    env_cfg.patch_waste_fertility_effect_enable = True

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is True
    assert decision.profile == PATCH_PROFILE_P2

    metadata = _patch_profile_metadata(
        _owner_inspection_runner(),
        env_cfg,
        sim_cfg,
        {"flat": False, "patch": True, "churn": True},
    )
    assert metadata["native_runtime_patch_profile"] == PATCH_PROFILE_P2
    assert metadata["native_runtime_patch_profile_scope"] == "full_profile"
    assert metadata["patch_waste_fertility_effect_enable"] is True


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("patch_density_radius", -1, "invalid_patch_density_radius"),
        ("patch_visibility_min", 1.25, "invalid_patch_visibility_bounds"),
        ("patch_visibility_default", 2.0, "invalid_patch_visibility_default"),
    ),
)
def test_invalid_derived_map_configuration_fails_closed(
    field: str,
    value: object,
    reason: str,
) -> None:
    sim_cfg, env_cfg = _configs()
    if field == "patch_visibility_default":
        env_cfg.patch_visibility_from_cover_enable = False
    setattr(env_cfg, field, value)

    decision = classify_patch_profile(sim_cfg, env_cfg, world="patch")

    assert decision.supported is False
    assert reason in decision.unsupported_reasons


def _owner_inspection_runner() -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        _bank_device="cpu",
        _policy_use_gru=[False],
    )


@pytest.mark.parametrize(
    ("waste_consequence", "movement_apply"),
    ((True, False), (False, True)),
)
def test_owner_marks_required_partial_p2_fixtures_as_diagnostic_only(
    waste_consequence: bool,
    movement_apply: bool,
) -> None:
    sim_cfg, env_cfg = _configs()
    env_cfg.patch_world_observation_enable = True
    env_cfg.patch_world_waste_consequence_enable = waste_consequence
    env_cfg.patch_movement_cost_apply_enable = movement_apply

    metadata = _patch_profile_metadata(
        _owner_inspection_runner(),
        env_cfg,
        sim_cfg,
        {"flat": False, "patch": True, "churn": True},
    )

    assert metadata["native_runtime_patch_profile"] == PATCH_PROFILE_P2
    assert metadata["native_runtime_patch_profile_scope"] == "focused_p2_subcell"


def test_p2_array_import_requires_recovery_activity() -> None:
    _, env_cfg = _configs()
    shape = (env_cfg.height, env_cfg.width)
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name in PATCH_REQUIRED_ENV_ARRAYS
    }
    arrays["patch_fertility"].fill(np.float32(1.0))
    arrays["patch_movement_cost"].fill(np.float32(1.0))
    env = SimpleNamespace(
        agent_grid=np.full(shape, -1, dtype=np.int32),
        grid=np.full(shape, EMPTY, dtype=np.int32),
        **arrays,
    )

    with pytest.raises(RuntimeError, match="patch_recovery_activity"):
        _patch_import_arrays(env, env_cfg, profile=PATCH_PROFILE_P2)

    env.patch_recovery_activity = np.zeros(shape, dtype=np.float32)
    tensors = _patch_import_arrays(env, env_cfg, profile=PATCH_PROFILE_P2)

    assert "patch_recovery_activity" in tensors
