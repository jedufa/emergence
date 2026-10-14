from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import engine.campaigns.patch_runner_factory as patch_runner_factory
from engine.campaigns.patch_runner_factory import (
    PatchRunnerFactoryError,
    _eed_founder_resource_axis_v1,
    build_patch_eed_fixed_mlp_runner_v1,
    build_patch_fixed_runner_v1,
    build_patch_jungle_fixed_mlp_runner_v1,
    build_patch_wdl_fixed_mlp_runner_v1,
    build_patch_wdl_jungle_fixed_mlp_runner_v1,
    derive_patch_master_seed_v1,
    derive_production_turnover_seed_v1,
)
from engine.campaigns.production_engine_lifecycle_v1 import (
    LIFE_HISTORY_PROFILE_INERT_V1,
    LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1,
    LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1,
    PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1,
    PRODUCTION_DEFAULT_LIFE_HISTORY_PROFILE_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W16P128C256_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W16P64C128_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C768_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C1024_V1,
    production_lifecycle_life_history_profile_v1,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    resolve_genome_v2_founder_profile_v1,
)


class _CapturedRunner:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _cell(founder: str = "legacy", *, ecology: int = 101, task: int = 9101):
    return {
        "cell_id": f"cell-{founder}",
        "parameters": {
            "patch.grid_size": 16,
            "patch.initial_population": 64,
            "patch.capacity": 256,
            "patch.founder_placement": founder,
        },
        "seeds": {"ecology": ecology, "task": task},
    }


def _genome_v2_cell(founder_profile: str | None = None):
    cell = _cell()
    cell["parameters"].update(
        {
            "genome_upgrade_v2.profile": "genome_upgrade_v2_fixed_q10_v1",
            "genome_upgrade_v2.enable": True,
            "genome_upgrade_v2.mutation_rate_q10": 20,
            "genome_upgrade_v2.mutation_step_q10": 64,
        }
    )
    if founder_profile is not None:
        cell["parameters"]["genome_upgrade_v2.founder_profile"] = founder_profile
    return cell


def _checkpoint_authority_binding() -> dict[str, str]:
    return {
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": "0" * 40,
        "executable_package_core_sha256": "0" * 64,
        "package_id_sha256": "1" * 64,
        "package_receipt_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "state_schema_sha256": "5" * 64,
        "mechanic_contract_sha256": "6" * 64,
        "capability_contract_sha256": "b" * 64,
        "cancellation_abi_sha256": "7" * 64,
        "observability_schema_sha256": "8" * 64,
        "capacity_layout_contract_sha256": "9" * 64,
        "execution_route_sha256": "a" * 64,
    }


def _neutral_lifecycle_cell():
    cell = _genome_v2_cell("neutral")
    cell["parameters"].update(
        {
            "genome_upgrade_v2.mutation_rate_q10": 0,
            "genome_upgrade_v2.mutation_step_q10": 0,
            "production_lifecycle.turnover_enabled": True,
            "production_lifecycle.ordinary_births_enabled": False,
            "production_lifecycle.lifecycle_mode": "legacy_synchronized_truncation_v1",
            "production_lifecycle.life_history_profile": LIFE_HISTORY_PROFILE_INERT_V1,
            "production_lifecycle.turnover_cull_fraction": 0.5,
            "production_lifecycle.turnover_parent_fraction": 1.0,
        }
    )
    return cell


def _build(tmp_path: Path, founder: str = "legacy"):
    return build_patch_fixed_runner_v1(
        _cell(founder),
        tmp_path / founder,
        generations=14,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )


def _build_cpu_native(tmp_path: Path):
    return build_patch_fixed_runner_v1(
        _cell(),
        tmp_path / "cpu-native",
        generations=14,
        ticks_per_generation=160,
        artifact_path="/verified/resident-owner.so",
        qualification_receipt_path="",
        reporting_receipt_path="",
        execution_backend="cpu_native",
        runner_constructor=_CapturedRunner,
    )


def test_translates_compiled_cell_into_public_patch_runtime_configs(
    tmp_path: Path,
) -> None:
    built = _build(tmp_path)
    runner = built.runner

    assert (runner.env_cfg.width, runner.env_cfg.height) == (16, 16)
    assert runner.env_cfg.patch_world_enable is True
    assert runner.env_cfg.patch_world_observation_enable is True
    assert runner.env_cfg.patch_world_waste_consequence_enable is True
    assert runner.env_cfg.patch_movement_cost_apply_enable is True
    assert runner.env_cfg.spatial_ecology_seed == 101
    assert runner.env_cfg.spatial_ecology_enable is False
    assert runner.env_cfg.enable_biomes is False
    assert runner.env_cfg.enable_combat is False
    assert runner.env_cfg.enable_injury is False
    assert runner.env_cfg.wdl_delayed_hazard_enable is False
    assert runner.pop_cfg.population_size == 64
    assert runner.pop_cfg.max_slots_cap == 256

    cfg = runner.sim_cfg
    assert (cfg.generations, cfg.steps_per_generation) == (14, 80)
    assert (cfg.device, cfg.bank_device) == ("cpu", "cpu")
    assert cfg.cuda_architecture_b_request_mode == "explicit_cuda"
    assert cfg.native_runtime_allow_runtime_jit_build is False
    assert cfg.native_runtime_fixed_policy_family == "mlp"
    assert cfg.native_runtime_max_births_per_generation == 64
    assert cfg.gen1_f1_capture_enable is True
    assert cfg.campaign_output_profile == "compact_default"
    assert cfg.behavior_log_enable is False
    assert cfg.maintenance_enable is False
    assert cfg.chemistry_functional_coupling_enable is False

    assert built.receipt["seed_ownership"]["named_owners"] == {
        "ecology": 101,
        "task": 9101,
    }
    assert built.receipt["seed_ownership"]["engine_rng_root"] == cfg.seed
    assert built.receipt["founder_initialization"]["spatial_placement_changed"] is False


def test_cpu_native_runner_receipt_does_not_claim_cuda_residency(
    tmp_path: Path,
) -> None:
    built = _build_cpu_native(tmp_path)
    cfg = built.runner.sim_cfg

    assert cfg.cuda_architecture_b_production_enable is False
    assert cfg.cuda_architecture_b_request_mode == "cpu_native"
    assert cfg.native_runtime_allow_runtime_jit_build is False
    assert built.receipt["dimensions"]["ticks_per_generation"] == 160
    assert built.receipt["execution"]["resident_backend"] == "cpu_native"
    assert built.receipt["execution"]["request_mode"] == "cpu_native"
    assert built.receipt["execution"]["runtime_jit_allowed"] is False
    assert built.receipt["artifact_bindings"] == {
        "cpu_native_resident_owner_extension": "/verified/resident-owner.so"
    }


def test_cpu_native_runner_rejects_missing_extension_or_cuda_receipts(
    tmp_path: Path,
) -> None:
    arguments = {
        "generations": 14,
        "ticks_per_generation": 160,
        "execution_backend": "cpu_native",
        "runner_constructor": _CapturedRunner,
    }
    with pytest.raises(PatchRunnerFactoryError, match="requires a resident owner"):
        build_patch_fixed_runner_v1(
            _cell(),
            tmp_path / "cpu-missing-artifact",
            artifact_path="",
            qualification_receipt_path="",
            reporting_receipt_path="",
            **arguments,
        )
    with pytest.raises(PatchRunnerFactoryError, match="cannot bind CUDA"):
        build_patch_fixed_runner_v1(
            _cell(),
            tmp_path / "cpu-cuda-receipt",
            artifact_path="/verified/resident-owner.so",
            qualification_receipt_path="/cuda/qualification.json",
            reporting_receipt_path="",
            **arguments,
        )


def test_cuda_runner_still_requires_all_three_artifact_paths(tmp_path: Path) -> None:
    with pytest.raises(
        PatchRunnerFactoryError,
        match="explicit CUDA execution requires all artifact receipt paths",
    ):
        build_patch_fixed_runner_v1(
            _cell(),
            tmp_path / "cuda-missing-receipts",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/cuda/controller.so",
            qualification_receipt_path="",
            reporting_receipt_path="",
            execution_backend="cuda",
            runner_constructor=_CapturedRunner,
        )


def test_patch_wd1_carrying_capacity_profile_is_named_and_route_bound(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"]["patch.ecology_profile"] = "wd1_carrying_capacity_v1"

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "wd1",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    assert built.receipt["world_profile"] == "patch-p2-wd1_carrying_capacity_v1"
    assert built.receipt["patch_ecology_profile"]["profile_id"] == (
        "wd1_carrying_capacity_v1"
    )
    assert built.receipt["patch_ecology_profile"]["changes"] == {
        "start_energy": 36.0,
        "max_energy": 96.0,
        "move_cost": 0.35,
        "rest_cost": 0.08,
        "gather_cost": 0.12,
        "food_energy": 8.0,
        "patch_food_gather_amount": 0.35,
        "patch_food_regrow_per_step": 0.006,
        "patch_food_regrow_use_fertility": True,
        "patch_food_regrow_cap_by_fertility": True,
        "patch_waste_regrowth_penalty_scale": 0.75,
        "patch_waste_regrowth_penalty_max": 0.75,
        "patch_density_regrowth_penalty_scale": 0.75,
        "patch_density_regrowth_penalty_max": 0.75,
        "patch_regrowth_min_multiplier": 0.05,
        "reproduce_parent_energy_share": 0.25,
        "reproduce_min_energy_buffer": 18.0,
        "reproduce_cooldown_steps": 45,
        "reproduce_try_prob": 1.0,
    }
    assert env.start_energy == 36.0
    assert env.food_energy == 8.0
    assert env.patch_food_regrow_cap_by_fertility is True
    assert env.patch_waste_density_emit_scale == 0.0
    assert env.reproduce_parent_energy_share == 0.25


def test_patch_wd2_resource_deficit_survival_profile_adds_maintenance_budget(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"]["patch.ecology_profile"] = "wd2_resource_deficit_survival_v1"

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "wd2",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    cfg = built.runner.sim_cfg
    receipt = built.receipt["patch_ecology_profile"]
    assert built.receipt["world_profile"] == (
        "patch-p2-wd2_resource_deficit_survival_v1"
    )
    assert receipt["profile_id"] == "wd2_resource_deficit_survival_v1"
    assert receipt["sim_config_changes"] == {
        "resource_maintenance_energy_cost_per_tick": 0.035
    }
    assert receipt["maintenance_budget"] == {
        "resource_maintenance_energy_cost_per_tick": 0.035,
        "ticks_per_generation": 80,
        "energy_cost_per_generation": 2.8000000000000003,
        "energy_cost_per_g32_cell": 89.60000000000001,
        "nominal_food_energy_per_full_gather": 8.0,
    }
    assert cfg.resource_maintenance_energy_cost_per_tick == 0.035


def test_patch_wd3_resource_throughput_profile_is_conservation_bound(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"]["patch.ecology_profile"] = "wd3_resource_throughput_v1"

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "wd3",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    cfg = built.runner.sim_cfg
    receipt = built.receipt["patch_ecology_profile"]
    assert built.receipt["world_profile"] == "patch-p2-wd3_resource_throughput_v1"
    assert receipt["profile_id"] == "wd3_resource_throughput_v1"
    assert receipt["mechanic"] == (
        "authoritative_patch_resource_throughput_conservation_loop"
    )
    assert receipt["sim_config_changes"] == {
        "resource_maintenance_energy_cost_per_tick": 0.075
    }
    assert receipt["changes"]["food_energy"] == 2.0
    assert receipt["changes"]["patch_food_gather_amount"] == 0.25
    assert receipt["resource_throughput_budget"][
        "max_regrown_energy_per_tick_if_full_fertility"
    ] == pytest.approx(2.048)
    assert env.start_energy == 30.0
    assert env.max_energy == 64.0
    assert env.patch_food_regrow_per_step == 0.001
    assert env.reproduce_parent_energy_share == 0.35
    assert cfg.resource_maintenance_energy_cost_per_tick == 0.075


def test_initial_spatial_placement_profile_is_admitted_and_receipted(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"][
        "patch.initial_spatial_placement_profile"
    ] = "clustered_quadrant_v1"

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "placement",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert (
        built.runner.sim_cfg.initial_spatial_placement_profile
        == "clustered_quadrant_v1"
    )
    assert built.receipt["initial_spatial_placement"] == {
        "schema_version": "eiel_initial_spatial_placement_profile_v1",
        "profile_id": "clustered_quadrant_v1",
    }


def test_initial_spatial_placement_profile_fails_closed(tmp_path: Path) -> None:
    cell = _cell()
    cell["parameters"]["patch.initial_spatial_placement_profile"] = "surprise"

    with pytest.raises(
        PatchRunnerFactoryError,
        match="initial_spatial_placement_profile",
    ):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "placement",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_patch_ecology_profile_fails_closed_for_unknown_profile(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"]["patch.ecology_profile"] = "open_knob_override"

    with pytest.raises(PatchRunnerFactoryError, match="unsupported patch.ecology_profile"):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "bad-profile",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_translates_genome_v2_profile_into_global_engine_controls(
    tmp_path: Path,
) -> None:
    binding = _checkpoint_authority_binding()
    built = build_patch_fixed_runner_v1(
        _genome_v2_cell(),
        tmp_path / "genome-v2",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        genome_v2_checkpoint_authority_binding=binding,
        runner_constructor=_CapturedRunner,
    )

    cfg = built.runner.sim_cfg
    assert cfg.inheritance_profile == "diploid_linkage_v1"
    assert cfg.genome_upgrade_v2_enable is True
    assert cfg.genome_upgrade_v2_mutation_rate_q10 == 20
    assert cfg.genome_upgrade_v2_mutation_step_q10 == 64
    neutral = resolve_genome_v2_founder_profile_v1("neutral")
    assert cfg.genome_upgrade_v2_founder_profile_id == "neutral"
    assert cfg.genome_upgrade_v2_founder_profile_sha256 == neutral["payload_sha256"]
    assert cfg.diploid_mutation_enable is True
    assert cfg.recomb72_mutation_off is False
    assert built.runner.cuda_architecture_b_inheritance_sidecar_cell_id == (
        "cell-legacy"
    )
    assert built.runner.cuda_architecture_b_genome_v2_causal_physical_cell_ordinal == 0
    assert cfg.genome_v2_checkpoint_authority_binding == tuple(sorted(binding.items()))
    assert built.runner.env_cfg.reproduce_requires_mate is True
    assert built.receipt["genome"] == {
        "profile": "genome_upgrade_v2_fixed_q10_v1",
        "enabled": True,
        "inheritance_profile": "diploid_linkage_v1",
        "mutation_rate_q10": 20,
        "mutation_step_q10": 64,
        "founder_profile_id": "neutral",
        "founder_profile_sha256": neutral["payload_sha256"],
        "e1_arm_label": None,
    }


def test_translates_candidate_cuda_registry_paths_into_sim_config(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "cuda_architecture_b.qualification_registry_path": (
                "/qualified/candidate-registry.json"
            ),
            "cuda_architecture_b.promotion_registry_path": (
                "/qualified/candidate-promotion.json"
            ),
            "cuda_architecture_b.cpython_gauss_segment_capacity": 65_536,
            "cuda_architecture_b.policy_birth_segment_refill_enable": True,
            "cuda_architecture_b.policy_birth_segment_capacity": -1,
            "cuda_architecture_b.cpu_request_staging_enable": True,
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "candidate-registry",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.sim_cfg.cuda_architecture_b_qualification_registry_path == (
        "/qualified/candidate-registry.json"
    )
    assert built.runner.sim_cfg.cuda_architecture_b_promotion_registry_path == (
        "/qualified/candidate-promotion.json"
    )
    assert built.runner.sim_cfg.cuda_architecture_b_cpython_gauss_segment_capacity == (
        65_536
    )
    assert (
        built.runner.sim_cfg.cuda_architecture_b_policy_birth_segment_refill_enable
        is True
    )
    assert built.runner.sim_cfg.cuda_architecture_b_policy_birth_segment_capacity == -1
    assert built.runner.sim_cfg.cuda_architecture_b_cpu_request_staging_enable is True


def test_translates_tw_rngv2_semantics_into_sim_config(tmp_path: Path) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "tw_rngv2.semantics_version": 2,
            "tw_rngv2.world_uid": 7,
            "cuda_architecture_b.cpython_gauss_segment_capacity": 0,
            "cuda_architecture_b.policy_birth_segment_refill_enable": False,
        }
    )
    default_cell = _cell()
    default_built = build_patch_fixed_runner_v1(
        default_cell,
        tmp_path / "rngv2-default",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )
    assert getattr(default_built.runner.sim_cfg, "rng_semantics_version", 1) == 1
    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "rngv2-candidate",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )
    assert built.runner.sim_cfg.rng_semantics_version == 2
    assert built.runner.sim_cfg.world_uid == 7
    assert built.runner.sim_cfg.cuda_architecture_b_cpython_gauss_segment_capacity == 0


def test_translates_frozen_neutral_lifecycle_into_cuda_controls(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "neutral-lifecycle",
        generations=30,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    cfg = built.runner.sim_cfg
    expected_seed = derive_production_turnover_seed_v1(
        cell_id=cell["cell_id"],
        seeds=cell["seeds"],
    )
    assert cfg.ordinary_births_enabled is False
    assert cfg.generation_turnover_enabled is True
    assert cfg.population_size == 64
    assert cfg.cull_fraction == 0.5
    assert cfg.parent_fraction == 1.0
    assert built.runner.pop_cfg.parent_fraction == 1.0
    assert cfg.diploid_mutation_enable is False
    assert cfg.recomb72_mutation_off is True
    assert cfg.turnover_selection_mode == 1
    assert cfg.turnover_selection_seed == expected_seed
    assert cfg.genome_v2_causal_capture_enable is True
    assert cfg.life_history_profile_id == LIFE_HISTORY_PROFILE_INERT_V1
    assert cfg.life_history_mode == 0
    assert built.receipt["production_lifecycle"] == {
        "enabled": True,
        "profile": None,
        "lifecycle_mode": "legacy_synchronized_truncation_v1",
        "ordinary_births_enabled": False,
        "turnover_enabled": True,
        "turnover_cull_fraction": 0.5,
        "turnover_parent_fraction": 1.0,
        "lifecycle_mode_abi": 0,
        "diploid_mutation_enabled": False,
        "legacy_scalar_mutation_off": True,
        "genome_v2_mutation_enabled": False,
        "genome_v2_mutation_rate_q10": 0,
        "genome_v2_mutation_step_q10": 0,
        "genome_v2_mutation_regime": "fixed_q10_off_control",
        "turnover_selection_mode": "genotype_blind_neutral",
        "turnover_selection_mode_abi": 1,
        "turnover_selection_seed": expected_seed,
        "turnover_selection_seed_schema_version": (
            "eiel_production_turnover_selection_seed_v1"
        ),
        "genome_v2_causal_capture_enabled": True,
        "life_history": production_lifecycle_life_history_profile_v1(
            LIFE_HISTORY_PROFILE_INERT_V1
        ),
    }


def test_omitted_lifecycle_mode_uses_qualified_lifecycle_v2_default(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].pop("production_lifecycle.lifecycle_mode")
    cell["parameters"].pop("production_lifecycle.life_history_profile")
    cell["parameters"]["production_lifecycle.ordinary_births_enabled"] = True
    cell["parameters"]["production_lifecycle.turnover_cull_fraction"] = 0.0

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "default-lifecycle-v2",
        generations=30,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    cfg = built.runner.sim_cfg
    lifecycle = built.receipt["production_lifecycle"]
    assert cfg.ordinary_births_enabled is True
    assert cfg.generation_turnover_enabled is False
    assert cfg.lifecycle_mode == 1
    assert lifecycle["lifecycle_mode"] == PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1
    assert lifecycle["turnover_cull_fraction"] == 0.0
    assert lifecycle["life_history"] == production_lifecycle_life_history_profile_v1(
        PRODUCTION_DEFAULT_LIFE_HISTORY_PROFILE_V1
    )


def test_translates_hybrid_overlapping_lifecycle_into_cuda_controls(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
            "production_lifecycle.ordinary_births_enabled": True,
            "production_lifecycle.turnover_cull_fraction": 0.0,
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "hybrid-overlapping-lifecycle",
        generations=30,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    cfg = built.runner.sim_cfg
    lifecycle = built.receipt["production_lifecycle"]
    assert cfg.ordinary_births_enabled is True
    assert cfg.generation_turnover_enabled is False
    assert cfg.population_size == 64
    assert cfg.cull_fraction == 0.0
    assert cfg.parent_fraction == 1.0
    assert cfg.lifecycle_mode == 1
    assert cfg.turnover_selection_mode == 1
    assert lifecycle["lifecycle_mode"] == "hybrid_overlapping_v1"
    assert lifecycle["lifecycle_mode_abi"] == 1
    assert lifecycle["turnover_cull_fraction"] == 0.0
    assert lifecycle["turnover_selection_mode"] == "genotype_blind_neutral"
    assert lifecycle["turnover_selection_mode_abi"] == 1
    assert lifecycle["life_history"] == production_lifecycle_life_history_profile_v1(
        LIFE_HISTORY_PROFILE_INERT_V1
    )


def test_lifecycle_initial_live_count_p192_is_admitted_and_receipted(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "patch.initial_population": 192,
            "patch.initial_live_count": 192,
            "production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
            "production_lifecycle.ordinary_births_enabled": True,
            "production_lifecycle.turnover_cull_fraction": 0.0,
            "production_lifecycle.life_history_profile": (
                LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
            ),
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "hybrid-p192",
        generations=12,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.pop_cfg.population_size == 192
    assert built.runner.pop_cfg.max_slots_cap == 256
    assert built.runner.sim_cfg.population_size == 192
    assert built.receipt["dimensions"]["configured_initial_live_count"] == 192
    assert built.receipt["initial_occupancy"] == {
        "schema_version": "eiel_patch_initial_live_count_authority_v1",
        "configured_initial_live_count": 192,
        "physical_resident_capacity": 256,
        "admitted_initial_live_counts": [64, 192, 256, 512],
        "hard_cap_fuse": 256,
    }


def test_lifecycle_initial_live_count_fails_closed_for_mismatch(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"]["patch.initial_live_count"] = 192

    with pytest.raises(PatchRunnerFactoryError, match="must match"):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "mismatch-live-count",
            generations=12,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_lifecycle_initial_live_count_fails_closed_for_unadmitted_value(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "patch.initial_population": 128,
            "patch.initial_live_count": 128,
        }
    )

    with pytest.raises(PatchRunnerFactoryError, match="64, 192, 256, or 512"):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "unsupported-live-count",
            generations=12,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_translates_active_life_history_profile_into_cuda_controls(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
            "production_lifecycle.ordinary_births_enabled": True,
            "production_lifecycle.turnover_cull_fraction": 0.0,
            "production_lifecycle.life_history_profile": (
                LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1
            ),
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "hybrid-life-history",
        generations=30,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    expected = production_lifecycle_life_history_profile_v1(
        LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1
    )
    cfg = built.runner.sim_cfg
    assert cfg.life_history_profile_id == LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1
    for name, value in expected.items():
        if name in {"profile_id", "schema_version"}:
            continue
        assert getattr(cfg, name) == value
    assert built.receipt["production_lifecycle"]["life_history"] == expected


def test_translates_stage1s_overlap_life_history_profile_into_cuda_controls(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
            "production_lifecycle.ordinary_births_enabled": True,
            "production_lifecycle.turnover_cull_fraction": 0.0,
            "production_lifecycle.life_history_profile": (
                LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
            ),
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "hybrid-life-history-stage1s",
        generations=30,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    expected = production_lifecycle_life_history_profile_v1(
        LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
    )
    cfg = built.runner.sim_cfg
    assert cfg.life_history_profile_id == LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
    assert cfg.life_history_elder_fertility_floor == 0.35
    assert built.receipt["production_lifecycle"]["life_history"] == expected


def test_production_lifecycle_accepts_w32p256_patch_scale(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "patch.grid_size": 32,
            "patch.initial_population": 256,
            "patch.capacity": 256,
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "w32p256-lifecycle",
        generations=30,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.env_cfg.width == 32
    assert built.runner.env_cfg.height == 32
    assert built.runner.pop_cfg.population_size == 256


@pytest.mark.parametrize(
    ("profile_id", "grid_size", "initial_population", "capacity"),
    (
        (PATCH_FIXED_MLP_TURNOVER_ONLY_W16P64C128_V1, 16, 64, 128),
        (PATCH_FIXED_MLP_TURNOVER_ONLY_W16P128C256_V1, 16, 128, 256),
        (PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1, 32, 256, 512),
        (PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C768_V1, 32, 512, 768),
        (PATCH_FIXED_MLP_TURNOVER_ONLY_W32P512C1024_V1, 32, 512, 1024),
    ),
)
def test_production_lifecycle_accepts_explicit_turnover_only_shapes(
    tmp_path: Path,
    profile_id: str,
    grid_size: int,
    initial_population: int,
    capacity: int,
) -> None:
    cell = _neutral_lifecycle_cell()
    for name in list(cell["parameters"]):
        if name.startswith("genome_upgrade_v2."):
            cell["parameters"].pop(name)
    cell["parameters"].update(
        {
            "patch.grid_size": grid_size,
            "patch.initial_population": initial_population,
            "patch.capacity": capacity,
            "production_lifecycle.profile": profile_id,
            "production_lifecycle.lifecycle_mode": (
                "legacy_synchronized_truncation_v1"
            ),
            "production_lifecycle.life_history_profile": LIFE_HISTORY_PROFILE_INERT_V1,
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / profile_id,
        generations=30,
        ticks_per_generation=160,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.env_cfg.width == grid_size
    assert built.runner.env_cfg.height == grid_size
    assert built.runner.pop_cfg.population_size == initial_population
    assert built.runner.pop_cfg.max_slots_cap == capacity
    assert built.receipt["production_lifecycle"]["profile"] == profile_id


def test_legacy_lifecycle_rejects_active_life_history_profile(tmp_path: Path) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"][
        "production_lifecycle.life_history_profile"
    ] = LIFE_HISTORY_PROFILE_STAGE1R_QUALIFICATION_V1

    with pytest.raises(PatchRunnerFactoryError, match="inert life history"):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "legacy-active-life-history",
            generations=30,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_neutral_lifecycle_requires_complete_genome_v2_profile(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "production_lifecycle.turnover_enabled": True,
            "production_lifecycle.ordinary_births_enabled": False,
            "production_lifecycle.turnover_cull_fraction": 0.5,
            "production_lifecycle.turnover_parent_fraction": 1.0,
        }
    )

    with pytest.raises(
        PatchRunnerFactoryError,
        match="requires one complete Genome V2 profile",
    ):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "lifecycle-without-genome-v2",
            generations=30,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


@pytest.mark.parametrize("impact_applied", (False, True))
def test_eed_runner_binds_frozen_overlay_below_genome_v2(
    tmp_path: Path,
    impact_applied: bool,
) -> None:
    cell = _genome_v2_cell()
    cell["parameters"].update(
        {
            "eed.founder_resource_axis_profile": "symmetric_mix",
            "eed.resource_impact_apply_enable": impact_applied,
            "eed.ecology_frozen": False,
            "eed.phenotype_fixed": True,
        }
    )
    built = build_patch_eed_fixed_mlp_runner_v1(
        cell,
        tmp_path / f"eed-{impact_applied}",
        generations=8,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    sim = built.runner.sim_cfg
    assert env.eed_resource_impact_enable is True
    assert env.eed_resource_impact_apply_enable is impact_applied
    assert env.eed_ecology_frozen_enable is False
    assert env.eed_phenotype_fixed_enable is True
    assert env.eed_phenotype_fixed_axis_profile == "symmetric_mix"
    assert env.food_spawn_per_step == 0
    assert env.eed_resource_a_renewal_per_step == 0.25
    assert env.eed_resource_b_renewal_per_step == 0.25
    assert env.eed_resource_impact_per_success == 0.25
    assert env.eed_resource_impact_decay_per_step == 0.25
    assert env.eed_resource_impact_max_load == 1.0
    assert env.reproduce_try_prob == 1.0
    assert sim.genome_upgrade_v2_enable is True
    assert sim.inheritance_profile == "diploid_linkage_v1"
    assert sim.resource_handling_representation_enable is True
    assert sim.native_runtime_max_births_per_generation == 3
    assert sim.cuda_architecture_b_max_births_per_tick == 3
    assert built.receipt["overlay"]["component_id"] == (
        "world.overlay.eed_resource_impact.v1"
    )
    assert built.receipt["overlay"]["ecology_frozen"] is False
    assert built.receipt["overlay"]["phenotype_fixed"] is True
    assert built.receipt["overlay"]["phenotype_fixed_axis_profile"] == "symmetric_mix"
    assert built.receipt["overlay"]["reproduce_try_prob"] == 1.0
    assert built.receipt["overlay"]["new_rng_draws"] == 0


def test_eed_runner_accepts_production_lifecycle_controls(tmp_path: Path) -> None:
    cell = _genome_v2_cell()
    cell["parameters"].update(
        {
            "eed.founder_resource_axis_profile": "symmetric_mix",
            "eed.resource_impact_apply_enable": True,
            "eed.ecology_frozen": False,
            "eed.phenotype_fixed": False,
            "production_lifecycle.turnover_enabled": True,
            "production_lifecycle.ordinary_births_enabled": True,
            "production_lifecycle.lifecycle_mode": "hybrid_overlapping_v1",
            "production_lifecycle.life_history_profile": (
                LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
            ),
            "production_lifecycle.turnover_cull_fraction": 0.0,
            "production_lifecycle.turnover_parent_fraction": 1.0,
            "cuda_architecture_b.cpython_gauss_segment_capacity": 65_536,
            "cuda_architecture_b.policy_birth_segment_refill_enable": True,
            "cuda_architecture_b.policy_birth_segment_capacity": -1,
            "cuda_architecture_b.turnover_mlp_shadow_elide_enable": True,
            "cuda_architecture_b.cpu_request_staging_enable": True,
        }
    )

    built = build_patch_eed_fixed_mlp_runner_v1(
        cell,
        tmp_path / "eed-lifecycle",
        generations=8,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    sim = built.runner.sim_cfg
    lifecycle = built.receipt["production_lifecycle"]
    assert sim.ordinary_births_enabled is True
    assert sim.generation_turnover_enabled is False
    assert sim.cull_fraction == 0.0
    assert sim.lifecycle_mode == 1
    assert sim.life_history_profile_id == LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
    assert sim.cuda_architecture_b_cpython_gauss_segment_capacity == 65_536
    assert sim.cuda_architecture_b_policy_birth_segment_refill_enable is True
    assert sim.cuda_architecture_b_policy_birth_segment_capacity == -1
    assert sim.cuda_architecture_b_turnover_mlp_shadow_elide_enable is True
    assert sim.cuda_architecture_b_cpu_request_staging_enable is True
    assert lifecycle["lifecycle_mode"] == "hybrid_overlapping_v1"
    assert lifecycle["ordinary_births_enabled"] is True
    assert lifecycle["turnover_selection_mode"] == "genotype_blind_neutral"
    assert lifecycle["life_history"] == production_lifecycle_life_history_profile_v1(
        LIFE_HISTORY_PROFILE_STAGE1S_OVERLAP_V1
    )


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("neutral", [0.0, 0.0, 0.0, 0.0]),
        ("a_high", [1.0, 1.0, 1.0, 1.0]),
        ("b_high", [-1.0, -1.0, -1.0, -1.0]),
        ("symmetric_mix", [1.0, -1.0, 1.0, -1.0]),
    ),
)
def test_eed_founder_resource_axis_profiles_are_executable(
    profile: str,
    expected: list[float],
) -> None:
    observed = [_eed_founder_resource_axis_v1(profile, index) for index in range(4)]

    assert observed == expected


def test_eed_runner_projects_resource_food_grid_before_native_import(
    tmp_path: Path,
) -> None:
    from env.gridworld_pkg.constants import EMPTY, FOOD
    from env.gridworld_pkg.patch_world import patch_food_presence_threshold

    cell = _genome_v2_cell()
    cell["parameters"].update(
        {
            "eed.founder_resource_axis_profile": "symmetric_mix",
            "eed.resource_impact_apply_enable": False,
            "eed.ecology_frozen": False,
            "eed.phenotype_fixed": False,
            "patch.grid_size": 8,
            "patch.initial_population": 8,
            "patch.capacity": 32,
        }
    )
    built = build_patch_eed_fixed_mlp_runner_v1(
        cell,
        tmp_path / "eed-native-import-grid",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        genome_v2_checkpoint_authority_binding=_checkpoint_authority_binding(),
    )

    amount = np.asarray(built.runner.env.patch_food_amount, dtype=np.float32)
    expected = np.where(
        amount > np.float32(patch_food_presence_threshold(built.runner.env_cfg)),
        FOOD,
        EMPTY,
    ).astype(built.runner.env.grid.dtype, copy=False)
    np.testing.assert_array_equal(built.runner.env.grid, expected)


@pytest.mark.parametrize(
    "profile_id",
    [
        "neutral",
        "pleiotropic_tradeoffs",
        "regulatory_activation_inhibition",
    ],
)
def test_genome_v2_founder_profile_is_frozen_and_content_bound(
    tmp_path: Path, profile_id: str
) -> None:
    built = build_patch_fixed_runner_v1(
        _genome_v2_cell(profile_id),
        tmp_path / profile_id,
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )
    expected = resolve_genome_v2_founder_profile_v1(profile_id)
    assert built.runner.sim_cfg.genome_upgrade_v2_founder_profile_id == profile_id
    assert (
        built.runner.sim_cfg.genome_upgrade_v2_founder_profile_sha256
        == expected["payload_sha256"]
    )
    assert built.receipt["genome"]["founder_profile_id"] == profile_id
    assert (
        built.receipt["genome"]["founder_profile_sha256"] == expected["payload_sha256"]
    )


def test_e1_arm_labels_are_provenance_only_for_neutral_founders(
    tmp_path: Path,
) -> None:
    built = []
    for label in ("control", "treatment"):
        cell = _genome_v2_cell("neutral")
        cell["parameters"]["genome_upgrade_v2.e1_arm_label"] = label
        built.append(
            build_patch_fixed_runner_v1(
                cell,
                tmp_path / label,
                generations=3,
                ticks_per_generation=80,
                artifact_path="/qualified/controller.so",
                qualification_receipt_path="/qualified/qualification.json",
                reporting_receipt_path="/qualified/reporting.json",
                runner_constructor=_CapturedRunner,
            )
        )

    control, treatment = built
    assert control.receipt["genome"]["e1_arm_label"] == "control"
    assert treatment.receipt["genome"]["e1_arm_label"] == "treatment"
    for field in (
        "genome_upgrade_v2_founder_profile_id",
        "genome_upgrade_v2_founder_profile_sha256",
    ):
        assert getattr(control.runner.sim_cfg, field) == getattr(
            treatment.runner.sim_cfg, field
        )
    assert (
        control.receipt["genome"]["founder_profile_id"]
        == treatment.receipt["genome"]["founder_profile_id"]
        == "neutral"
    )
    assert (
        control.receipt["genome"]["founder_profile_sha256"]
        == treatment.receipt["genome"]["founder_profile_sha256"]
    )


def test_e1_arm_label_without_complete_genome_profile_fails_closed(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"]["genome_upgrade_v2.e1_arm_label"] = "control"
    with pytest.raises(PatchRunnerFactoryError, match="complete profile"):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "orphan-label",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_unsupported_e1_arm_label_fails_closed(tmp_path: Path) -> None:
    cell = _genome_v2_cell("neutral")
    cell["parameters"]["genome_upgrade_v2.e1_arm_label"] = "sham"
    with pytest.raises(PatchRunnerFactoryError, match="control or treatment"):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "unsupported-label",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_partial_or_unsupported_genome_v2_profile_fails_closed(
    tmp_path: Path,
) -> None:
    partial = _cell()
    partial["parameters"]["genome_upgrade_v2.enable"] = True
    with pytest.raises(PatchRunnerFactoryError, match="complete profile"):
        build_patch_fixed_runner_v1(
            partial,
            tmp_path / "partial-genome-v2",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )

    unsupported = _genome_v2_cell()
    unsupported["parameters"]["genome_upgrade_v2.profile"] = "future_v3"
    with pytest.raises(PatchRunnerFactoryError, match="unsupported Genome V2"):
        build_patch_fixed_runner_v1(
            unsupported,
            tmp_path / "unsupported-genome-v2",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


@pytest.mark.parametrize(
    ("label", "expected", "enabled", "groups"),
    [
        ("legacy", "legacy", False, 8),
        ("balanced", "balanced-4", True, 4),
        ("round_robin", "balanced-8", True, 8),
    ],
)
def test_founder_labels_bind_real_initialization_mechanics(
    tmp_path: Path,
    label: str,
    expected: str,
    enabled: bool,
    groups: int,
) -> None:
    built = _build(tmp_path, label)
    cfg = built.runner.sim_cfg

    assert cfg.lineage_founder_balance_enable is enabled
    assert cfg.lineage_founder_spatial_separation_enable is False
    assert built.receipt["founder_initialization"]["applied_treatment"] == expected
    if enabled:
        assert cfg.lineage_founder_group_count == groups
        assert cfg.lineage_founder_min_agents_per_group == 64 // groups
        assert cfg.lineage_founder_assignment_mode == "deterministic_round_robin"


def test_master_seed_is_stable_order_independent_and_owns_every_named_seed() -> None:
    base = derive_patch_master_seed_v1({"task": 9101, "ecology": 101, "founder": 7})
    assert base == derive_patch_master_seed_v1(
        {"founder": 7, "ecology": 101, "task": 9101}
    )
    assert base != derive_patch_master_seed_v1(
        {"founder": 8, "ecology": 101, "task": 9101}
    )
    assert base != derive_patch_master_seed_v1(
        {"founder": 7, "ecology": 102, "task": 9101}
    )


@pytest.mark.parametrize("founder", ["magic", "max_distance", "spatial"])
def test_unsupported_founder_placement_fails_closed(
    tmp_path: Path, founder: str
) -> None:
    with pytest.raises(
        PatchRunnerFactoryError, match="unsupported patch.founder_placement"
    ):
        _build(tmp_path, founder)


def test_missing_seed_or_artifact_binding_fails_before_runner_construction(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["seeds"] = {"task": 9101}
    with pytest.raises(PatchRunnerFactoryError, match="ecology seed"):
        build_patch_fixed_runner_v1(
            cell,
            tmp_path / "missing-seed",
            generations=14,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )

    with pytest.raises(PatchRunnerFactoryError, match="artifact receipt paths"):
        build_patch_fixed_runner_v1(
            _cell(),
            tmp_path / "missing-artifact",
            generations=14,
            ticks_per_generation=80,
            artifact_path="",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )


def test_wdl_fixed_mlp_composes_shared_overlay_with_fixed_policy(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        }
    )
    built = build_patch_wdl_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-fixed",
        generations=14,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.env_cfg.wdl_delayed_hazard_enable is True
    assert built.runner.env_cfg.wdl_delayed_hazard_delay_steps == 12
    assert built.runner.sim_cfg.native_runtime_fixed_policy_family == "mlp"
    assert built.runner.sim_cfg.native_runtime_max_births_per_generation == 1
    assert built.receipt["world_profile"] == "patch-p2-wdl-v2"
    assert built.receipt["policy_family"] == "fixed_mlp"
    assert built.receipt["wdl"]["task_seed"] == 9101


def test_wdl_fixed_mlp_preserves_genome_v2_founder_authority(
    tmp_path: Path,
) -> None:
    cell = _genome_v2_cell("pleiotropic_tradeoffs")
    cell["parameters"].update(
        {
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        }
    )
    built = build_patch_wdl_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-genome-v2",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )
    expected = resolve_genome_v2_founder_profile_v1("pleiotropic_tradeoffs")
    assert built.runner.env_cfg.wdl_delayed_hazard_enable is True
    assert built.runner.sim_cfg.genome_upgrade_v2_enable is True
    assert built.runner.sim_cfg.native_runtime_max_births_per_generation == 3
    assert (
        built.runner.sim_cfg.genome_upgrade_v2_founder_profile_sha256
        == expected["payload_sha256"]
    )
    explicit = build_patch_wdl_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-genome-v2-explicit-birth-limit",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        max_births_per_generation=5,
        runner_constructor=_CapturedRunner,
    )

    assert explicit.runner.sim_cfg.native_runtime_max_births_per_generation == 5
    assert explicit.runner.sim_cfg.cuda_architecture_b_max_births_per_tick == 5


def test_wdl_fixed_mlp_admits_frozen_lifecycle_turnover(tmp_path: Path) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        }
    )

    built = build_patch_wdl_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-lifecycle",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.env_cfg.wdl_delayed_hazard_enable is True
    assert built.runner.sim_cfg.ordinary_births_enabled is False
    assert built.runner.sim_cfg.generation_turnover_enabled is True
    assert built.receipt["production_lifecycle"]["ordinary_births_enabled"] is False
    assert built.receipt["production_lifecycle"]["lifecycle_mode"] == (
        "legacy_synchronized_truncation_v1"
    )


def test_wdl_fixed_mlp_admits_lifecycle_ordinary_births(tmp_path: Path) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "production_lifecycle.ordinary_births_enabled": True,
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        }
    )

    built = build_patch_wdl_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-lifecycle-births",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.env_cfg.wdl_delayed_hazard_enable is True
    assert built.runner.sim_cfg.ordinary_births_enabled is True
    assert built.runner.sim_cfg.generation_turnover_enabled is True
    assert built.receipt["production_lifecycle"]["ordinary_births_enabled"] is True
    assert built.receipt["production_lifecycle"]["lifecycle_mode"] == (
        "legacy_synchronized_truncation_v1"
    )


def test_wdl_fixed_mlp_admits_lifecycle_bounded_genome_v2_mutation(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "genome_upgrade_v2.mutation_rate_q10": 16,
            "genome_upgrade_v2.mutation_step_q10": 16,
            "production_lifecycle.ordinary_births_enabled": True,
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        }
    )

    built = build_patch_wdl_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-lifecycle-births-mutation",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    cfg = built.runner.sim_cfg
    lifecycle = built.receipt["production_lifecycle"]
    assert cfg.ordinary_births_enabled is True
    assert cfg.diploid_mutation_enable is True
    assert cfg.recomb72_mutation_off is False
    assert cfg.genome_upgrade_v2_mutation_rate_q10 == 16
    assert cfg.genome_upgrade_v2_mutation_step_q10 == 16
    assert lifecycle["genome_v2_mutation_enabled"] is True
    assert lifecycle["genome_v2_mutation_rate_q10"] == 16
    assert lifecycle["genome_v2_mutation_step_q10"] == 16
    assert lifecycle["genome_v2_mutation_regime"] == "fixed_q10_low_admitted"


def test_jungle_fixed_mlp_composes_the_single_canonical_v4_overlay(
    tmp_path: Path,
) -> None:
    built = build_patch_jungle_fixed_mlp_runner_v1(
        _cell(),
        tmp_path / "jungle-fixed",
        generations=7,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    assert (env.food_spawn_per_step, env.max_food) == (10, 64)
    assert env.jungle_predator_pressure_enable is True
    assert env.jungle_juvenile_predator_vulnerability_enable is True
    assert env.jungle_terrain_protection_enable is True
    assert env.jungle_reproduction_emissions_enable is True
    assert env.jungle_ecological_recovery_enable is True
    assert built.runner.sim_cfg.native_runtime_max_births_per_generation == 1
    assert built.runner.sim_cfg.native_runtime_fixed_policy_family == "mlp"
    assert built.runner.sim_cfg.native_runtime_allow_runtime_jit_build is False
    assert built.receipt["world_profile"] == "patch-p2-jungle-v4"
    assert built.receipt["overlay"]["configuration_id"] == "canonical-jungle-v4"


def test_wdl_jungle_fixed_mlp_composes_both_overlays_and_mutation(
    tmp_path: Path,
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "genome_upgrade_v2.founder_profile": (
                "recomb72_gv2_v1.single_locus_extreme_pair.all_heterozygous_ab"
            ),
            "genome_upgrade_v2.mutation_rate_q10": 16,
            "genome_upgrade_v2.mutation_step_q10": 16,
            "production_lifecycle.ordinary_births_enabled": True,
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        }
    )

    built = build_patch_wdl_jungle_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-jungle-lifecycle-births-mutation",
        generations=3,
        ticks_per_generation=160,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    cfg = built.runner.sim_cfg
    assert env.wdl_delayed_hazard_enable is True
    assert env.wdl_delayed_hazard_delay_steps == 12
    assert env.jungle_predator_pressure_enable is True
    assert env.jungle_reproduction_emissions_enable is True
    assert cfg.ordinary_births_enabled is True
    assert cfg.diploid_mutation_enable is True
    assert cfg.recomb72_mutation_off is False
    assert cfg.genome_upgrade_v2_mutation_rate_q10 == 16
    assert cfg.genome_upgrade_v2_mutation_step_q10 == 16
    assert built.receipt["world_profile"] == "patch-p2-wdl-v2-jungle-v4"
    assert built.receipt["wdl"]["task_seed"] == 9101
    assert built.receipt["jungle"]["configuration_id"] == "canonical-jungle-v4"
    assert (
        built.receipt["genome"]["founder_profile_id"]
        == "recomb72_gv2_v1.single_locus_extreme_pair.all_heterozygous_ab"
    )
    assert built.receipt["production_lifecycle"]["genome_v2_mutation_enabled"] is True


def test_wdl_jungle_authoritative_source_skips_fresh_world_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _neutral_lifecycle_cell()
    cell["parameters"].update(
        {
            "production_lifecycle.ordinary_births_enabled": True,
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 1,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
        }
    )
    initialized: list[object] = []
    monkeypatch.setattr(
        patch_runner_factory,
        "SimulationRunner",
        _CapturedRunner,
    )
    monkeypatch.setattr(
        patch_runner_factory,
        "prepare_patch_generation_zero_v1",
        initialized.append,
    )

    built = build_patch_wdl_jungle_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-jungle-authoritative-source",
        generations=3,
        ticks_per_generation=160,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        initialize_fresh_source=False,
        runner_constructor=_CapturedRunner,
    )

    assert initialized == []
    assert built.receipt["world_profile"] == "patch-p2-wdl-v2-jungle-v4"


def test_patch_existing_primitives_override_bundled_profile_after_translation(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "patch.ecology_profile": "wd2_resource_deficit_survival_v1",
            "env_cfg.patch_food_regrow_per_step": 0.125,
            "env_cfg.reproduce_try_prob": 0.35,
            "sim_cfg.resource_maintenance_energy_cost_per_tick": 0.02,
        }
    )

    built = build_patch_fixed_runner_v1(
        cell,
        tmp_path / "patch-primitives",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    assert built.runner.env_cfg.patch_food_regrow_per_step == 0.125
    assert built.runner.env_cfg.reproduce_try_prob == 0.35
    assert built.runner.sim_cfg.resource_maintenance_energy_cost_per_tick == 0.02
    assert built.receipt["primitive_controls"]["applied"] == {
        "env_cfg.patch_food_regrow_per_step": 0.125,
        "env_cfg.reproduce_try_prob": 0.35,
        "sim_cfg.resource_maintenance_energy_cost_per_tick": 0.02,
    }


def test_jungle_existing_primitives_override_canonical_overlay(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "env_cfg.food_spawn_per_step": 4,
            "env_cfg.reproduce_try_prob": 0.4,
            "env_cfg.jungle_predator_base_risk": 0.04,
            "env_cfg.jungle_predator_health_damage": 0.08,
        }
    )

    built = build_patch_jungle_fixed_mlp_runner_v1(
        cell,
        tmp_path / "jungle-primitives",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    assert env.food_spawn_per_step == 4
    assert env.reproduce_try_prob == 0.4
    assert env.jungle_predator_base_risk == 0.04
    assert env.jungle_predator_health_damage == 0.08
    assert built.receipt["overlay"]["primitive_controls"]["applied"] == {
        "env_cfg.food_spawn_per_step": 4,
        "env_cfg.jungle_predator_base_risk": 0.04,
        "env_cfg.jungle_predator_health_damage": 0.08,
        "env_cfg.reproduce_try_prob": 0.4,
    }


def test_eed_existing_primitives_override_companion_constants(
    tmp_path: Path,
) -> None:
    cell = _genome_v2_cell()
    cell["parameters"].update(
        {
            "eed.founder_resource_axis_profile": "symmetric_mix",
            "eed.resource_impact_apply_enable": True,
            "eed.ecology_frozen": False,
            "eed.phenotype_fixed": True,
            "env_cfg.eed_resource_a_capacity": 2.0,
            "env_cfg.eed_resource_a_renewal_per_step": 0.1,
            "env_cfg.reproduce_try_prob": 0.3,
        }
    )

    built = build_patch_eed_fixed_mlp_runner_v1(
        cell,
        tmp_path / "eed-primitives",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    assert env.eed_resource_a_capacity == 2.0
    assert env.eed_resource_a_renewal_per_step == 0.1
    assert env.reproduce_try_prob == 0.3
    assert built.receipt["overlay"]["resource_a_capacity"] == 2.0
    assert built.receipt["overlay"]["resource_a_renewal_per_step"] == 0.1
    assert built.receipt["overlay"]["reproduce_try_prob"] == 0.3


def test_wdl_existing_primitives_override_derived_internal_values(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 3,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": False,
            "env_cfg.wdl_delayed_hazard_event_period_steps": 40,
            "env_cfg.wdl_delayed_hazard_radius": 2,
            "env_cfg.wdl_delayed_hazard_observation_dropout_rate": 0.25,
            "env_cfg.reproduce_try_prob": 0.3,
        }
    )

    built = build_patch_wdl_fixed_mlp_runner_v1(
        cell,
        tmp_path / "wdl-primitives",
        generations=3,
        ticks_per_generation=80,
        artifact_path="/qualified/controller.so",
        qualification_receipt_path="/qualified/qualification.json",
        reporting_receipt_path="/qualified/reporting.json",
        runner_constructor=_CapturedRunner,
    )

    env = built.runner.env_cfg
    assert env.wdl_delayed_hazard_event_period_steps == 40
    assert env.wdl_delayed_hazard_radius == 2
    assert env.wdl_delayed_hazard_observation_dropout_rate == 0.25
    assert env.reproduce_try_prob == 0.3
    assert built.receipt["wdl"]["event_period_steps"] == 40
    assert built.receipt["wdl"]["radius"] == 2
    assert built.receipt["wdl"]["observation_dropout_rate"] == 0.25


def test_wdl_period_override_fails_closed_when_phases_overlap(
    tmp_path: Path,
) -> None:
    cell = _cell()
    cell["parameters"].update(
        {
            "wdl.delay_ticks": 12,
            "wdl.cue_duration_ticks": 1,
            "wdl.active_duration_ticks": 3,
            "wdl.intensity": 0.1,
            "wdl.cue_visible_at_decision": True,
            "env_cfg.wdl_delayed_hazard_event_period_steps": 15,
        }
    )

    with pytest.raises(PatchRunnerFactoryError, match=r"delay \+ active duration"):
        build_patch_wdl_fixed_mlp_runner_v1(
            cell,
            tmp_path / "wdl-invalid-period",
            generations=3,
            ticks_per_generation=80,
            artifact_path="/qualified/controller.so",
            qualification_receipt_path="/qualified/qualification.json",
            reporting_receipt_path="/qualified/reporting.json",
            runner_constructor=_CapturedRunner,
        )
