"""Public construction boundary for declarative fixed-policy Patch cells.

The factory owns configuration translation only.  Simulation and CUDA
execution remain owned by :class:`SimulationRunner` and the native resident
runtime respectively.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np
import torch

from engine.env.env_config import EnvConfig
from engine.evolution.population_pkg.population_config import PopConfig
from engine.evolution.diploid_inheritance_v1 import (
    DIPLOID_INHERITANCE_PROFILE,
)
from engine.simulation.metrics import MetricsLogger
from engine.simulation.runner import SimulationRunner
from engine.simulation.runner_pkg.runner_config import SimConfig
from engine.perf.cuda_genome_upgrade_v2_state_bank import (
    resolve_genome_v2_founder_profile_v1,
)

from .contracts import LogicalCellV1, Scalar
from .production_control_path_v1 import (
    EED_PRIMITIVE_PARAMETER_NAMES_V1,
    JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1,
    PATCH_PRIMITIVE_PARAMETER_NAMES_V1,
    WDL_PRIMITIVE_PARAMETER_NAMES_V1,
    apply_primitive_overrides_v1,
)
from .production_engine_lifecycle_v1 import (
    HYBRID_OVERLAPPING_V1,
    LEGACY_SYNCHRONIZED_TRUNCATION_V1,
    LIFECYCLE_PARAMETER_NAMES,
    OPTIONAL_LIFECYCLE_PARAMETER_NAMES,
    PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1,
    PRODUCTION_DEFAULT_LIFE_HISTORY_PROFILE_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1,
    production_lifecycle_life_history_profile_v1,
)


PATCH_FIXED_RUNNER_RECEIPT_SCHEMA_V1 = "eiel_patch_fixed_runner_factory_v1"
PATCH_ECOLOGY_PROFILE_DEFAULT_V1 = "patch_p2_default_v1"
PATCH_ECOLOGY_PROFILE_WD1_CARRYING_CAPACITY_V1 = "wd1_carrying_capacity_v1"
PATCH_ECOLOGY_PROFILE_WD2_RESOURCE_DEFICIT_SURVIVAL_V1 = (
    "wd2_resource_deficit_survival_v1"
)
PATCH_ECOLOGY_PROFILE_WD3_RESOURCE_THROUGHPUT_V1 = (
    "wd3_resource_throughput_v1"
)
PATCH_ECOLOGY_PROFILE_PARAMETER_NAMES_V1 = frozenset({"patch.ecology_profile"})
PATCH_INITIAL_SPATIAL_PLACEMENT_PROFILE_PARAMETER_NAMES_V1 = frozenset(
    {"patch.initial_spatial_placement_profile"}
)
PATCH_INITIAL_SPATIAL_PLACEMENT_PROFILES_V1 = frozenset(
    {
        "heritage_random_v1",
        "dispersed_lattice_v1",
        "dispersed_lattice_fill_v1",
        "clustered_quadrant_v1",
    }
)
PATCH_FIXED_PARAMETER_NAMES_V1 = frozenset(
    {
        "patch.grid_size",
        "patch.initial_population",
        "patch.capacity",
        "patch.founder_placement",
    }
)
PATCH_INITIAL_LIVE_COUNT_PARAMETER_NAMES_V1 = frozenset({"patch.initial_live_count"})
PATCH_LIFECYCLE_INITIAL_LIVE_COUNTS_V1 = frozenset({64, 192, 256, 512})
GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1 = frozenset(
    {
        "genome_upgrade_v2.profile",
        "genome_upgrade_v2.enable",
        "genome_upgrade_v2.mutation_rate_q10",
        "genome_upgrade_v2.mutation_step_q10",
    }
)
GENOME_UPGRADE_V2_PARAMETER_NAMES_V1 = frozenset(
    {
        *GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1,
        "genome_upgrade_v2.founder_profile",
        "genome_upgrade_v2.e1_arm_label",
    }
)
PRODUCTION_LIFECYCLE_REQUIRED_PARAMETER_NAMES_V1 = frozenset(
    LIFECYCLE_PARAMETER_NAMES
)
PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1 = (
    PRODUCTION_LIFECYCLE_REQUIRED_PARAMETER_NAMES_V1
    | frozenset(OPTIONAL_LIFECYCLE_PARAMETER_NAMES)
)
CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1 = frozenset(
    {
        "cuda_architecture_b.qualification_registry_path",
        "cuda_architecture_b.promotion_registry_path",
        "cuda_architecture_b.cpython_gauss_segment_capacity",
        "cuda_architecture_b.policy_birth_segment_refill_enable",
        "cuda_architecture_b.policy_birth_segment_capacity",
        "cuda_architecture_b.turnover_mlp_shadow_elide_enable",
        "cuda_architecture_b.cpu_request_staging_enable",
        "tw_rngv2.semantics_version",
        "tw_rngv2.world_uid",
        "tw_rngv2.namespace_key64",
    }
)


class PatchRunnerFactoryError(ValueError):
    """A compiled Patch cell cannot be translated without guessing."""


def _resolve_patch_runner_source_dimensions_v2(
    *,
    generations: int | None,
    ticks_per_generation: int | None,
    tick_native_control: Mapping[str, Any] | None,
) -> tuple[int, int, dict[str, Any]]:
    """Resolve legacy execution or native source-capacity dimensions.

    The SimulationRunner configuration still stores two integer dimensions,
    but active tick-native callers bind them only through explicitly named
    source-buffer coordinates. Mixing the two APIs fails closed.
    """

    legacy_supplied = generations is not None or ticks_per_generation is not None
    native_supplied = tick_native_control is not None
    if legacy_supplied == native_supplied:
        raise PatchRunnerFactoryError(
            "bind exactly one of legacy execution dimensions or native source "
            "capacity dimensions"
        )
    if legacy_supplied:
        if type(generations) is not int or generations <= 0:
            raise PatchRunnerFactoryError("generations must be a positive integer")
        if type(ticks_per_generation) is not int or ticks_per_generation <= 0:
            raise PatchRunnerFactoryError(
                "ticks_per_generation must be a positive integer"
            )
        return generations, ticks_per_generation, {
            "generations": generations,
            "ticks_per_generation": ticks_per_generation,
        }
    from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
        CudaArchitectureBDirectResidentSourceAllocatorError,
        _tick_native_allocator_projection_v1,
    )

    try:
        source_capacity_window_count, source_window_ticks = (
            _tick_native_allocator_projection_v1(tick_native_control)
        )
    except CudaArchitectureBDirectResidentSourceAllocatorError as exc:
        raise PatchRunnerFactoryError(
            "tick_native_control cannot derive bounded runner source capacity"
        ) from exc
    return source_capacity_window_count, source_window_ticks, {
        "time_authority": "absolute_tick_v1",
        "source_capacity_window_count": source_capacity_window_count,
        "source_window_ticks": source_window_ticks,
    }


@dataclass(frozen=True, slots=True)
class PatchFixedRunnerBuildV1:
    """Constructed runner plus the deterministic configuration receipt."""

    runner: Any
    receipt: Mapping[str, Any]


PATCH_STABLE_GRU_RUNNER_RECEIPT_SCHEMA_V1 = (
    "eiel_patch_stable_gru_runner_factory_v1"
)
PATCH_STABLE_GRU_PARAMETER_NAMES_V1 = frozenset(
    {
        *PATCH_FIXED_PARAMETER_NAMES_V1,
        "policy.hidden_width",
        "policy.reset_hidden_each_tick",
    }
)


@dataclass(frozen=True, slots=True)
class PatchStableGruRunnerBuildV1:
    """Constructed Patch/stable-GRU runner plus deterministic receipt."""

    runner: Any
    receipt: Mapping[str, Any]


WDL_STABLE_GRU_RUNNER_RECEIPT_SCHEMA_V1 = (
    "eiel_patch_wdl_stable_gru_runner_factory_v1"
)
WDL_STABLE_GRU_PARAMETER_NAMES_V1 = frozenset(
    {
        *PATCH_FIXED_PARAMETER_NAMES_V1,
        "wdl.delay_ticks",
        "wdl.cue_duration_ticks",
        "wdl.active_duration_ticks",
        "wdl.intensity",
        "wdl.cue_visible_at_decision",
        "policy.hidden_width",
        "policy.reset_hidden_each_tick",
    }
)


@dataclass(frozen=True, slots=True)
class PatchWdlStableGruRunnerBuildV1:
    """Constructed WDL/stable-GRU runner plus deterministic receipt."""

    runner: Any
    receipt: Mapping[str, Any]


WDL_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1 = "eiel_patch_wdl_fixed_mlp_runner_factory_v1"
WDL_FIXED_MLP_PARAMETER_NAMES_V1 = frozenset(
    {
        *PATCH_FIXED_PARAMETER_NAMES_V1,
        "wdl.delay_ticks",
        "wdl.cue_duration_ticks",
        "wdl.active_duration_ticks",
        "wdl.intensity",
        "wdl.cue_visible_at_decision",
    }
)


@dataclass(frozen=True, slots=True)
class PatchWdlFixedMlpRunnerBuildV1:
    """Constructed WDL/fixed-MLP runner plus deterministic receipt."""

    runner: Any
    receipt: Mapping[str, Any]


EED_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1 = (
    "eiel_patch_eed_fixed_mlp_runner_factory_v1"
)
EED_FIXED_MLP_PARAMETER_NAMES_V1 = frozenset(
    {
        *PATCH_FIXED_PARAMETER_NAMES_V1,
        "eed.founder_resource_axis_profile",
        "eed.resource_impact_apply_enable",
        "eed.ecology_frozen",
        "eed.phenotype_fixed",
    }
)


@dataclass(frozen=True, slots=True)
class PatchEedFixedMlpRunnerBuildV1:
    """Constructed EED/fixed-MLP runner plus deterministic receipt."""

    runner: Any
    receipt: Mapping[str, Any]


PATCH_JUNGLE_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1 = (
    "eiel_patch_jungle_fixed_mlp_runner_factory_v1"
)
WDL_JUNGLE_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1 = (
    "eiel_patch_wdl_jungle_fixed_mlp_runner_factory_v1"
)
WDL_JUNGLE_EED_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1 = (
    "eiel_patch_wdl_jungle_eed_fixed_mlp_runner_factory_v1"
)


@dataclass(frozen=True, slots=True)
class PatchJungleFixedMlpRunnerBuildV1:
    """Constructed canonical Jungle-v4 runner plus deterministic receipt."""

    runner: Any
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PatchWdlJungleFixedMlpRunnerBuildV1:
    """Constructed WDL+Jungle runner plus deterministic receipt."""

    runner: Any
    receipt: Mapping[str, Any]


def _materialize_stable_gru_generation_zero_v1(
    runner: Any, *, master_seed: int
) -> None:
    """Install deterministic H64 stable-GRU founders at generation zero."""

    from engine.agents.gru_policy import GRUPolicy

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(master_seed)
        np.random.seed(master_seed & 0xFFFF_FFFF)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(master_seed)
            runner.pop.init_random()
            agents = sorted(
                runner.pop.alive_agents(), key=lambda agent: int(agent.agent_id)
            )
            for index, agent in enumerate(agents):
                agent.genome.use_gru = True
                agent.genome.memory_dim = 64
                torch.manual_seed(master_seed + index + 1)
                agent.policy = GRUPolicy(
                    int(runner._obs_size), 64, int(runner._out)
                ).eval()
            runner._ensure_bank()
            runner._place_agents()
            runner._mark_bank_dirty()
            runner._current_gen = 0
            runner.pop._eiel_current_generation_index = 0
            runner.pop._current_generation = 0
            runner._phase3_pre_start_generation_sync(agents)
            runner._start_generation(0)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _restore_stable_gru_runtime_contract_v1(runner: Any) -> None:
    """Restore fields normalized by runner construction to the qualified lane."""

    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    runner.sim_cfg.native_runtime_cxx26_scope_lock_enable = True
    runner.sim_cfg.genome_c1_enable = False
    runner.sim_cfg.genome_c2_enable = False
    if hasattr(runner, "pop"):
        runner.pop.sim_cfg = runner.sim_cfg


def prepare_patch_generation_zero_v1(runner: Any) -> None:
    """Materialize the canonical fresh-run source boundary for resident import.

    This is the engine-owned equivalent of the historical gate preamble.  It
    stops immediately before tick dispatch and preserves process-global RNG
    state so constructing one declarative cell cannot perturb another.
    """

    required = ("pop", "_ensure_bank", "_place_agents", "_start_generation")
    if any(not hasattr(runner, name) for name in required):
        raise PatchRunnerFactoryError(
            "runner constructor did not provide the Patch generation-zero surface"
        )
    seed = int(getattr(runner.sim_cfg, "seed"))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed & 0xFFFF_FFFF)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            runner.pop.sim_cfg = runner.sim_cfg
            runner.pop.init_random()
            runner._ensure_bank()
            runner._place_agents()
            branch_observability = getattr(runner, "branch_observability", None)
            if branch_observability is not None:
                branch_observability.register_founders(
                    runner.pop.alive_agents(), generation=0, tick=0
                )
            mark_dirty = getattr(runner, "_mark_bank_dirty", None)
            if callable(mark_dirty):
                mark_dirty()
            runner._current_gen = 0
            runner.pop._eiel_current_generation_index = 0
            runner.pop._current_generation = 0
            sync = getattr(runner, "_phase3_pre_start_generation_sync", None)
            if callable(sync):
                sync(list(runner.pop.alive_agents()))
            runner._start_generation(0)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _cell_fields(
    cell: LogicalCellV1 | Mapping[str, Any],
) -> tuple[str, dict[str, Scalar], dict[str, int]]:
    if isinstance(cell, LogicalCellV1):
        cell_id = cell.cell_id
        parameters = dict(cell.parameters)
        seeds = dict(cell.seeds)
    elif isinstance(cell, Mapping):
        cell_id = cell.get("cell_id")
        parameters = cell.get("parameters")
        seeds = cell.get("seeds")
        if not isinstance(parameters, Mapping) or not isinstance(seeds, Mapping):
            raise PatchRunnerFactoryError("compiled cell must contain parameter and seed mappings")
        parameters = dict(parameters)
        seeds = dict(seeds)
    else:
        raise PatchRunnerFactoryError("cell must be a LogicalCellV1 or mapping")
    if not isinstance(cell_id, str) or not cell_id:
        raise PatchRunnerFactoryError("compiled cell_id must be a nonempty string")
    genome_upgrade_fields = (
        set(parameters) & GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
    )
    if genome_upgrade_fields and not (
        GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1 <= genome_upgrade_fields
        and genome_upgrade_fields
        <= GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
    ):
        raise PatchRunnerFactoryError(
            "Genome V2 parameters must be supplied as one complete profile"
        )
    lifecycle_fields = (
        set(parameters) & PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
    )
    lifecycle_profile = parameters.get("production_lifecycle.profile")
    turnover_only_profile = (
        lifecycle_profile in PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1
    )
    lifecycle_required_fields = (
        set(parameters) & PRODUCTION_LIFECYCLE_REQUIRED_PARAMETER_NAMES_V1
    )
    if lifecycle_fields and lifecycle_required_fields != (
        PRODUCTION_LIFECYCLE_REQUIRED_PARAMETER_NAMES_V1
    ):
        raise PatchRunnerFactoryError(
            "production lifecycle parameters must be supplied as one complete profile"
        )
    if lifecycle_fields and not turnover_only_profile and not (
        GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1 <= genome_upgrade_fields
    ):
        raise PatchRunnerFactoryError(
            "production lifecycle requires one complete Genome V2 profile"
        )
    if turnover_only_profile and genome_upgrade_fields:
        raise PatchRunnerFactoryError(
            "turnover-only W32 profile excludes Genome V2 parameters"
        )
    allowed_parameters = (
        PATCH_FIXED_PARAMETER_NAMES_V1
        | PATCH_ECOLOGY_PROFILE_PARAMETER_NAMES_V1
        | PATCH_INITIAL_SPATIAL_PLACEMENT_PROFILE_PARAMETER_NAMES_V1
        | PATCH_INITIAL_LIVE_COUNT_PARAMETER_NAMES_V1
        | GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
        | PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
        | CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1
        | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
    )
    required_parameters = PATCH_FIXED_PARAMETER_NAMES_V1 | (
        GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1
        if genome_upgrade_fields
        else frozenset()
    )
    if not (
        required_parameters <= set(parameters)
        and set(parameters) <= allowed_parameters
    ):
        missing = sorted(PATCH_FIXED_PARAMETER_NAMES_V1 - set(parameters))
        extra = sorted(
            set(parameters)
                - PATCH_FIXED_PARAMETER_NAMES_V1
                - PATCH_ECOLOGY_PROFILE_PARAMETER_NAMES_V1
                - PATCH_INITIAL_LIVE_COUNT_PARAMETER_NAMES_V1
                - GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
                - PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
                - CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1
                - PATCH_PRIMITIVE_PARAMETER_NAMES_V1
            )
        raise PatchRunnerFactoryError(
            f"fixed Patch parameters must match the public contract; missing={missing} extra={extra}"
        )
    if "ecology" not in seeds:
        raise PatchRunnerFactoryError("compiled cell is missing its ecology seed owner")
    if not seeds:
        raise PatchRunnerFactoryError("compiled cell must have named seed owners")
    for owner, value in seeds.items():
        if not isinstance(owner, str) or not owner:
            raise PatchRunnerFactoryError("seed owner names must be nonempty strings")
        if type(value) is not int or value < 0:
            raise PatchRunnerFactoryError(f"seed owner {owner!r} must hold a nonnegative integer")
    return cell_id, parameters, seeds


def derive_patch_master_seed_v1(seeds: Mapping[str, int]) -> int:
    """Derive the single engine RNG root from every explicitly named owner."""

    normalized: dict[str, int] = {}
    for owner, value in sorted(seeds.items()):
        if not isinstance(owner, str) or not owner or type(value) is not int or value < 0:
            raise PatchRunnerFactoryError("named seed owners must map to nonnegative integers")
        normalized[owner] = value
    if not normalized:
        raise PatchRunnerFactoryError("at least one named seed owner is required")
    payload = {
        "schema_version": "eiel_named_seed_owners_v1",
        "owners": normalized,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def derive_production_turnover_seed_v1(
    *,
    cell_id: str,
    seeds: Mapping[str, int],
) -> int:
    """Domain-separate neutral turnover selection from every named RNG owner."""

    payload = {
        "schema_version": "eiel_production_turnover_selection_seed_v1",
        "cell_id": cell_id,
        "owners": dict(sorted(seeds.items())),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def apply_hapb_frag_identity_runtime_controls_v1(
    parameters: Mapping[str, Any],
) -> None:
    """Pin force-vs-reuse env from explicit campaign parameters.

    Omitted keys leave the process environment unchanged. Explicit False
    clears a leftover opposite arm so Launchpad force/reuse specs cannot
    inherit the other campaign's env.
    """

    identity_reuse = parameters.get("hapb_frag.identity_reuse")
    force_physical = parameters.get("hapb_frag.force_physical_refill")
    if identity_reuse is True:
        os.environ["EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE"] = "1"
    elif identity_reuse is False:
        os.environ.pop("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE", None)
    if force_physical is True:
        os.environ["EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL"] = "1"
    elif force_physical is False:
        os.environ.pop("EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL", None)


def _positive_integer(parameters: Mapping[str, Scalar], name: str) -> int:
    value = parameters[name]
    if type(value) is not int or value <= 0:
        raise PatchRunnerFactoryError(f"{name} must be a positive integer")
    return value


def _apply_founder_treatment(
    sim_cfg: SimConfig,
    *,
    requested: Scalar,
    population: int,
) -> dict[str, Any]:
    if not isinstance(requested, str):
        raise PatchRunnerFactoryError("patch.founder_placement must be a string")
    # The two legacy declarative labels remain accepted, but the receipt names
    # the real initialization mechanic.  No spatial-placement claim is made.
    treatment_groups = {
        "legacy": None,
        "balanced": 4,
        "balanced-4": 4,
        "round_robin": 8,
        "balanced-8": 8,
    }
    if requested not in treatment_groups:
        raise PatchRunnerFactoryError(
            "unsupported patch.founder_placement; supported treatments are "
            "legacy, balanced/balanced-4, and round_robin/balanced-8"
        )
    group_count = treatment_groups[requested]
    sim_cfg.lineage_founder_spatial_separation_enable = False
    sim_cfg.lineage_founder_spatial_assignment_mode = "none"
    if group_count is None:
        sim_cfg.lineage_founder_balance_enable = False
        return {
            "requested_label": requested,
            "applied_treatment": "legacy",
            "mechanic": "one_founder_per_initial_agent",
            "spatial_placement_changed": False,
        }
    if group_count > population:
        raise PatchRunnerFactoryError(
            f"founder treatment requires {group_count} groups for population {population}"
        )
    sim_cfg.lineage_founder_balance_enable = True
    sim_cfg.lineage_founder_group_count = group_count
    sim_cfg.lineage_founder_min_agents_per_group = population // group_count
    sim_cfg.lineage_founder_assignment_mode = "deterministic_round_robin"
    sim_cfg.lineage_founder_balance_schema_version = 1
    return {
        "requested_label": requested,
        "applied_treatment": f"balanced-{group_count}",
        "mechanic": "deterministic_round_robin_founder_lineage_assignment",
        "group_count": group_count,
        "minimum_agents_per_group": population // group_count,
        "spatial_placement_changed": False,
    }


def _apply_patch_ecology_profile_v1(
    env_cfg: EnvConfig, profile_id: str
) -> dict[str, Any]:
    """Apply one named route-bound Patch ecology profile."""

    if profile_id == PATCH_ECOLOGY_PROFILE_DEFAULT_V1:
        return {
            "profile_id": profile_id,
            "mechanic": "patch_p2_default_resource_energy_profile",
            "resource_units": "patch_food_amount in [0, patch_food_amount_max]",
            "energy_units": "agent energy",
            "changes": {},
        }
    if profile_id not in {
        PATCH_ECOLOGY_PROFILE_WD1_CARRYING_CAPACITY_V1,
        PATCH_ECOLOGY_PROFILE_WD2_RESOURCE_DEFICIT_SURVIVAL_V1,
        PATCH_ECOLOGY_PROFILE_WD3_RESOURCE_THROUGHPUT_V1,
    }:
        raise PatchRunnerFactoryError(
            "unsupported patch.ecology_profile; supported profiles are "
            f"{PATCH_ECOLOGY_PROFILE_DEFAULT_V1} and "
            f"{PATCH_ECOLOGY_PROFILE_WD1_CARRYING_CAPACITY_V1} and "
            f"{PATCH_ECOLOGY_PROFILE_WD2_RESOURCE_DEFICIT_SURVIVAL_V1} and "
            f"{PATCH_ECOLOGY_PROFILE_WD3_RESOURCE_THROUGHPUT_V1}"
        )

    if profile_id == PATCH_ECOLOGY_PROFILE_WD3_RESOURCE_THROUGHPUT_V1:
        changes = {
            "start_energy": 30.0,
            "max_energy": 64.0,
            "move_cost": 0.45,
            "rest_cost": 0.10,
            "gather_cost": 0.16,
            "food_energy": 2.0,
            "patch_food_gather_amount": 0.25,
            "patch_food_regrow_per_step": 0.001,
            "patch_food_regrow_use_fertility": True,
            "patch_food_regrow_cap_by_fertility": True,
            "patch_waste_regrowth_penalty_scale": 0.85,
            "patch_waste_regrowth_penalty_max": 0.85,
            "patch_density_regrowth_penalty_scale": 0.85,
            "patch_density_regrowth_penalty_max": 0.85,
            "patch_regrowth_min_multiplier": 0.0,
            "reproduce_parent_energy_share": 0.35,
            "reproduce_min_energy_buffer": 22.0,
            "reproduce_cooldown_steps": 80,
            "reproduce_try_prob": 1.0,
        }
        sim_config_changes = {
            "resource_maintenance_energy_cost_per_tick": 0.075
        }
        mechanic = "authoritative_patch_resource_throughput_conservation_loop"
        causal_roles = (
            "finite food stock, harvest-proportional energy, bounded low "
            "regrowth, parent energy transfer, and deterministic per-living-tick "
            "maintenance make survival depend on resource throughput without "
            "population targeting"
        )
    else:
        changes = {
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
        sim_config_changes = {}
        mechanic = (
            "bounded_existing_patch_resource_energy_survival_loop"
            if profile_id == PATCH_ECOLOGY_PROFILE_WD2_RESOURCE_DEFICIT_SURVIVAL_V1
            else "bounded_existing_patch_resource_energy_carrying_capacity_loop"
        )
        causal_roles = (
            "finite food and bounded regrowth limit energy income; action costs "
            "and mate-birth energy transfer expose scarcity; the WD2 profile "
            "adds deterministic per-living-tick maintenance through the existing "
            "cost_charge_scalar authority; waste and density suppress regrowth "
            "without direct population targeting"
        )
    if profile_id == PATCH_ECOLOGY_PROFILE_WD2_RESOURCE_DEFICIT_SURVIVAL_V1:
        sim_config_changes["resource_maintenance_energy_cost_per_tick"] = 0.035
    for name, value in changes.items():
        setattr(env_cfg, name, value)
    return {
        "profile_id": profile_id,
        "mechanic": mechanic,
        "resource_units": "patch_food_amount in [0, patch_food_amount_max]",
        "regeneration_per_tick": {
            "patch_food_regrow_per_step": changes["patch_food_regrow_per_step"],
            "fertility_scaled": changes["patch_food_regrow_use_fertility"],
            "fertility_capped": changes["patch_food_regrow_cap_by_fertility"],
        },
        "energy_units": "agent energy",
        "maintenance_budget": {
            "resource_maintenance_energy_cost_per_tick": sim_config_changes.get(
                "resource_maintenance_energy_cost_per_tick", 0.0
            ),
            "ticks_per_generation": 80,
            "energy_cost_per_generation": 80
            * sim_config_changes.get("resource_maintenance_energy_cost_per_tick", 0.0),
            "energy_cost_per_g32_cell": 32
            * 80
            * sim_config_changes.get("resource_maintenance_energy_cost_per_tick", 0.0),
            "nominal_food_energy_per_full_gather": changes["food_energy"],
        },
        "resource_throughput_budget": {
            "grid_cells": 256,
            "food_amount_initial_if_full": 256.0,
            "food_energy_per_full_gather": changes["food_energy"],
            "food_amount_per_full_gather": changes["patch_food_gather_amount"],
            "energy_per_food_amount": (
                changes["food_energy"] / changes["patch_food_gather_amount"]
            ),
            "max_regrown_food_per_tick_if_full_fertility": (
                256 * changes["patch_food_regrow_per_step"]
            ),
            "max_regrown_energy_per_tick_if_full_fertility": (
                256
                * changes["patch_food_regrow_per_step"]
                * changes["food_energy"]
                / changes["patch_food_gather_amount"]
            ),
        },
        "causal_roles": causal_roles,
        "changes": changes,
        "sim_config_changes": sim_config_changes,
    }


def build_patch_fixed_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int | None = None,
    ticks_per_generation: int | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int = 64,
    genome_v2_checkpoint_authority_binding: Mapping[str, str] | None = None,
    execution_backend: str = "cuda",
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchFixedRunnerBuildV1:
    """Build one real fixed-MLP Patch runner from a compiled logical cell.

    ``runner_constructor`` is injectable so contract tests can inspect the
    translation without constructing a CUDA-backed runtime.
    """

    cell_id, parameters, seeds = _cell_fields(cell)
    grid_size = _positive_integer(parameters, "patch.grid_size")
    population = _positive_integer(parameters, "patch.initial_population")
    capacity = _positive_integer(parameters, "patch.capacity")
    initial_live_count = parameters.get("patch.initial_live_count", population)
    if type(initial_live_count) is not int or initial_live_count <= 0:
        raise PatchRunnerFactoryError(
            "patch.initial_live_count must be a positive integer"
        )
    if initial_live_count != population:
        raise PatchRunnerFactoryError(
            "patch.initial_live_count must match patch.initial_population"
        )
    if capacity < population:
        raise PatchRunnerFactoryError("patch.capacity must be >= patch.initial_population")
    (
        runner_capacity_windows,
        runner_source_window_ticks,
        runner_dimension_receipt,
    ) = _resolve_patch_runner_source_dimensions_v2(
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        tick_native_control=tick_native_control,
    )
    if type(max_births_per_generation) is not int or max_births_per_generation <= 0:
        raise PatchRunnerFactoryError("max_births_per_generation must be a positive integer")
    if execution_backend not in {"cuda", "cpu_native"}:
        raise PatchRunnerFactoryError("execution_backend must be cuda or cpu_native")
    if execution_backend == "cuda":
        artifact_bindings = {
            "cuda_extension": artifact_path,
            "qualification_receipt": qualification_receipt_path,
            "reporting_receipt": reporting_receipt_path,
        }
        if any(
            not isinstance(value, str) or not value
            for value in artifact_bindings.values()
        ):
            raise PatchRunnerFactoryError(
                "explicit CUDA execution requires all artifact receipt paths"
            )
    else:
        if not isinstance(artifact_path, str) or not artifact_path:
            raise PatchRunnerFactoryError(
                "CPU-native execution requires a resident owner extension path"
            )
        if qualification_receipt_path or reporting_receipt_path:
            raise PatchRunnerFactoryError(
                "CPU-native execution cannot bind CUDA qualification or reporting receipts"
            )
        artifact_bindings = {
            "cpu_native_resident_owner_extension": artifact_path,
        }

    master_seed = derive_patch_master_seed_v1(seeds)
    lifecycle_enabled = bool(
        set(parameters) & PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
    )
    lifecycle_profile = parameters.get("production_lifecycle.profile")
    turnover_only_profile = (
        lifecycle_profile in PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1
    )
    env_cfg = EnvConfig(width=grid_size, height=grid_size, view_radius=1)
    # Patch P2 qualification keeps the optional spatial-ecology overlay off.
    # The named ecology owner still contributes to the engine RNG root; turning
    # on another map mechanic here would silently change the qualified world.
    env_cfg.spatial_ecology_seed = seeds["ecology"]
    env_cfg.spatial_ecology_enable = False
    env_cfg.enable_biomes = False
    env_cfg.enable_injury = False
    env_cfg.enable_combat = False
    env_cfg.patch_world_enable = True
    env_cfg.patch_world_schema_version = 1
    env_cfg.patch_world_observation_enable = True
    env_cfg.patch_world_observation_schema_version = 1
    env_cfg.patch_world_observation_stacked_gather_enable = False
    env_cfg.patch_world_observation_numba_gather_enable = False
    env_cfg.patch_world_diagnostics_enable = True
    env_cfg.patch_world_lineage_descriptors_enable = True
    env_cfg.patch_world_descriptor_schema_version = 1
    env_cfg.patch_world_waste_consequence_enable = True
    env_cfg.patch_movement_cost_apply_enable = True
    env_cfg.reproduce_requires_mate = False
    env_cfg.tier4_juvenile_vulnerability_enable = False
    env_cfg.tier4_juvenile_steps = 0
    env_cfg.jungle_predator_pressure_enable = False
    env_cfg.jungle_reproduction_emissions_enable = False
    env_cfg.jungle_juvenile_predator_vulnerability_enable = False
    env_cfg.jungle_terrain_protection_enable = False
    env_cfg.jungle_ecological_recovery_enable = False
    env_cfg.wdl_delayed_hazard_enable = False
    env_cfg.patch_scent_birth_emit = 0.0
    env_cfg.patch_scent_death_emit = 0.0
    env_cfg.patch_waste_birth_emit = 0.0
    env_cfg.patch_waste_death_emit = 0.0
    env_cfg.patch_waste_density_emit_scale = 0.0
    ecology_profile_id = parameters.get(
        "patch.ecology_profile", PATCH_ECOLOGY_PROFILE_DEFAULT_V1
    )
    if not isinstance(ecology_profile_id, str) or not ecology_profile_id:
        raise PatchRunnerFactoryError("patch.ecology_profile must be a nonempty string")
    ecology_profile_receipt = _apply_patch_ecology_profile_v1(
        env_cfg, ecology_profile_id
    )
    genome_upgrade_v2_enabled = (
        GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1 <= set(parameters)
    )
    if genome_upgrade_v2_enabled:
        if (
            parameters["genome_upgrade_v2.profile"]
            != "genome_upgrade_v2_fixed_q10_v1"
            or parameters["genome_upgrade_v2.enable"] is not True
        ):
            raise PatchRunnerFactoryError(
                "unsupported Genome V2 profile or disabled admission"
            )
        for name in (
            "genome_upgrade_v2.mutation_rate_q10",
            "genome_upgrade_v2.mutation_step_q10",
        ):
            value = parameters[name]
            if type(value) is not int or not 0 <= value <= 1024:
                raise PatchRunnerFactoryError(
                    f"{name} must be an exact Q10 integer in [0, 1024]"
                )
        env_cfg.reproduce_requires_mate = True
        try:
            founder_profile = resolve_genome_v2_founder_profile_v1(
                str(
                    parameters.get(
                        "genome_upgrade_v2.founder_profile", "neutral"
                    )
                )
            )
        except ValueError as exc:
            raise PatchRunnerFactoryError(str(exc)) from exc
        e1_arm_label = parameters.get("genome_upgrade_v2.e1_arm_label")
        if e1_arm_label is not None and e1_arm_label not in {
            "control",
            "treatment",
        }:
            raise PatchRunnerFactoryError(
                "Genome V2 E1 arm label must be control or treatment"
            )
    else:
        founder_profile = None
        e1_arm_label = None

    sim_cfg = SimConfig(
        seed=master_seed,
        generations=runner_capacity_windows,
        steps_per_generation=runner_source_window_ticks,
        # Source construction is host-owned. Resident preparation performs the
        # explicit qualified transfer to CUDA.
        device="cpu",
        bank_device="cpu",
        campaign_output_profile="compact_default",
        behavior_log_enable=False,
        policy_deterministic_actions=False,
        policy_tick_stride=2,
    )
    sim_cfg.cuda_architecture_b_production_enable = execution_backend == "cuda"
    sim_cfg.cuda_architecture_b_request_mode = (
        "explicit_cuda" if execution_backend == "cuda" else "cpu_native"
    )
    sim_cfg.cuda_architecture_b_max_births_per_tick = max_births_per_generation
    sim_cfg.cuda_architecture_b_artifact_path = artifact_path
    sim_cfg.cuda_architecture_b_qualification_receipt_path = qualification_receipt_path
    sim_cfg.cuda_architecture_b_reporting_receipt_path = reporting_receipt_path
    qualification_registry_path = parameters.get(
        "cuda_architecture_b.qualification_registry_path"
    )
    promotion_registry_path = parameters.get(
        "cuda_architecture_b.promotion_registry_path"
    )
    if qualification_registry_path is not None:
        if not isinstance(qualification_registry_path, str) or not qualification_registry_path:
            raise PatchRunnerFactoryError(
                "cuda_architecture_b.qualification_registry_path must be a non-empty string"
            )
        sim_cfg.cuda_architecture_b_qualification_registry_path = (
            qualification_registry_path
        )
    if promotion_registry_path is not None:
        if not isinstance(promotion_registry_path, str) or not promotion_registry_path:
            raise PatchRunnerFactoryError(
                "cuda_architecture_b.promotion_registry_path must be a non-empty string"
            )
        sim_cfg.cuda_architecture_b_promotion_registry_path = promotion_registry_path
    cpython_gauss_segment_capacity = parameters.get(
        "cuda_architecture_b.cpython_gauss_segment_capacity"
    )
    if cpython_gauss_segment_capacity is not None:
        if (
            type(cpython_gauss_segment_capacity) is not int
            or cpython_gauss_segment_capacity < 0
        ):
            raise PatchRunnerFactoryError(
                "cuda_architecture_b.cpython_gauss_segment_capacity must be a "
                "nonnegative exact integer"
            )
        sim_cfg.cuda_architecture_b_cpython_gauss_segment_capacity = (
            cpython_gauss_segment_capacity
        )
    rng_semantics_version = parameters.get("tw_rngv2.semantics_version")
    if rng_semantics_version is not None:
        if type(rng_semantics_version) is not int or rng_semantics_version not in (1, 2):
            raise PatchRunnerFactoryError(
                "tw_rngv2.semantics_version must be exact integer 1 or 2"
            )
        sim_cfg.rng_semantics_version = rng_semantics_version
        if rng_semantics_version == 2:
            from engine.hapb_rngv2.route import apply_hapb_rng_route

            apply_hapb_rng_route(sim_cfg, hapb_rng_semantics_version=2)
    apply_hapb_frag_identity_runtime_controls_v1(parameters)
    world_uid = parameters.get("tw_rngv2.world_uid")
    if world_uid is not None:
        if type(world_uid) is not int or not 0 <= world_uid <= 0xFFFFFFFF:
            raise PatchRunnerFactoryError(
                "tw_rngv2.world_uid must be an exact uint32 integer"
            )
        sim_cfg.world_uid = world_uid
    namespace_key64 = parameters.get("tw_rngv2.namespace_key64")
    if namespace_key64 is not None:
        if (
            type(namespace_key64) is not int
            or namespace_key64 != 0xCA05117E450A51C0
        ):
            raise PatchRunnerFactoryError(
                "tw_rngv2.namespace_key64 must equal the canonical V2 key"
            )
        sim_cfg.rng_namespace_key64 = namespace_key64
    policy_birth_segment_refill_enable = parameters.get(
        "cuda_architecture_b.policy_birth_segment_refill_enable"
    )
    if policy_birth_segment_refill_enable is not None:
        if type(policy_birth_segment_refill_enable) is not bool:
            raise PatchRunnerFactoryError(
                "cuda_architecture_b.policy_birth_segment_refill_enable must be "
                "an exact bool"
            )
        sim_cfg.cuda_architecture_b_policy_birth_segment_refill_enable = (
            policy_birth_segment_refill_enable
        )
    policy_birth_segment_capacity = parameters.get(
        "cuda_architecture_b.policy_birth_segment_capacity"
    )
    if policy_birth_segment_capacity is not None:
        if (
            type(policy_birth_segment_capacity) is not int
            or (
                policy_birth_segment_capacity != -1
                and policy_birth_segment_capacity <= 0
            )
        ):
            raise PatchRunnerFactoryError(
                "cuda_architecture_b.policy_birth_segment_capacity must be -1 "
                "or a positive exact integer"
            )
        sim_cfg.cuda_architecture_b_policy_birth_segment_capacity = (
            policy_birth_segment_capacity
        )
    turnover_mlp_shadow_elide_enable = parameters.get(
        "cuda_architecture_b.turnover_mlp_shadow_elide_enable"
    )
    if turnover_mlp_shadow_elide_enable is not None:
        if type(turnover_mlp_shadow_elide_enable) is not bool:
            raise PatchRunnerFactoryError(
                "cuda_architecture_b.turnover_mlp_shadow_elide_enable must be "
                "an exact bool"
            )
        sim_cfg.cuda_architecture_b_turnover_mlp_shadow_elide_enable = (
            turnover_mlp_shadow_elide_enable
        )
    cpu_request_staging_enable = parameters.get(
        "cuda_architecture_b.cpu_request_staging_enable"
    )
    if cpu_request_staging_enable is not None:
        if type(cpu_request_staging_enable) is not bool:
            raise PatchRunnerFactoryError(
                "cuda_architecture_b.cpu_request_staging_enable must be "
                "an exact bool"
            )
        sim_cfg.cuda_architecture_b_cpu_request_staging_enable = (
            cpu_request_staging_enable
        )
    initial_placement_profile = parameters.get(
        "patch.initial_spatial_placement_profile", "heritage_random_v1"
    )
    if (
        not isinstance(initial_placement_profile, str)
        or initial_placement_profile
        not in PATCH_INITIAL_SPATIAL_PLACEMENT_PROFILES_V1
    ):
        raise PatchRunnerFactoryError(
            "patch.initial_spatial_placement_profile is not admitted"
        )
    sim_cfg.initial_spatial_placement_profile = initial_placement_profile
    sim_cfg.native_safe_default_enable = True
    sim_cfg.native_runtime_enable = True
    sim_cfg.native_runtime_fail_closed = True
    sim_cfg.native_runtime_allow_runtime_jit_build = False
    sim_cfg.native_runtime_fixed_policy_family = "mlp"
    for name, value in dict(
        ecology_profile_receipt.get("sim_config_changes", {})
    ).items():
        setattr(sim_cfg, name, value)
    primitive_control_receipt = apply_primitive_overrides_v1(
        env_cfg,
        sim_cfg,
        parameters,
        allowed_parameter_names=PATCH_PRIMITIVE_PARAMETER_NAMES_V1,
    )
    # RNG V2 is addressable resident state, not a future-value tape.  Keep the
    # legacy materialized source only for RNG V1.
    sim_cfg.native_runtime_use_precomputed_drive_noise = (
        int(getattr(sim_cfg, "rng_semantics_version", 1) or 1) != 2
    )
    sim_cfg.native_runtime_cxx26_scope_lock_enable = True
    sim_cfg.native_runtime_birth_commit_enable = True
    sim_cfg.native_runtime_max_births_per_generation = max_births_per_generation
    if lifecycle_enabled:
        lifecycle_mode = parameters.get(
            "production_lifecycle.lifecycle_mode",
            PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1,
        )
        if lifecycle_mode not in {
            LEGACY_SYNCHRONIZED_TRUNCATION_V1,
            HYBRID_OVERLAPPING_V1,
        }:
            raise PatchRunnerFactoryError(
                "production lifecycle mode is not admitted"
            )
        expected_cull_fraction = 0.0 if lifecycle_mode == HYBRID_OVERLAPPING_V1 else 0.5
        lifecycle_mode_abi = 1 if lifecycle_mode == HYBRID_OVERLAPPING_V1 else 0
        expected_lifecycle = {
            "production_lifecycle.turnover_enabled": True,
            "production_lifecycle.turnover_cull_fraction": expected_cull_fraction,
            "production_lifecycle.turnover_parent_fraction": 1.0,
        }
        ordinary_births_enabled = parameters.get(
            "production_lifecycle.ordinary_births_enabled"
        )
        if type(ordinary_births_enabled) is not bool:
            raise PatchRunnerFactoryError(
                "production lifecycle ordinary-birth switch must be boolean"
            )
        if any(
            type(parameters.get(name)) is not type(expected)
            or parameters.get(name) != expected
            for name, expected in expected_lifecycle.items()
        ):
            raise PatchRunnerFactoryError(
                "production lifecycle parameters differ from the frozen profile"
            )
        if turnover_only_profile:
            expected_grid_size, expected_population, expected_capacity = (
                PATCH_FIXED_MLP_TURNOVER_ONLY_PROFILE_SHAPES_V1[
                    lifecycle_profile
                ]
            )
            if (
                lifecycle_mode != LEGACY_SYNCHRONIZED_TRUNCATION_V1
                or ordinary_births_enabled is not False
                or grid_size != expected_grid_size
                or population != expected_population
                or capacity != expected_capacity
            ):
                raise PatchRunnerFactoryError(
                    "turnover-only W32 profile requires exact no-birth legacy turnover shape"
                )
        else:
            rate_q10 = parameters["genome_upgrade_v2.mutation_rate_q10"]
            step_q10 = parameters["genome_upgrade_v2.mutation_step_q10"]
            if (
                type(rate_q10) is not int
                or type(step_q10) is not int
                or not 0 <= rate_q10 <= 1024
                or not 0 <= step_q10 <= 1024
            ):
                raise PatchRunnerFactoryError(
                    "production lifecycle requires typed Genome V2 Q10 mutation bounds"
                )
            if (rate_q10 == 0) != (step_q10 == 0):
                raise PatchRunnerFactoryError(
                    "production lifecycle requires Genome V2 mutation rate/step to be both zero or both positive"
                )
            if population not in PATCH_LIFECYCLE_INITIAL_LIVE_COUNTS_V1:
                raise PatchRunnerFactoryError(
                    "production lifecycle requires initial live count "
                    "64, 192, 256, or 512"
                )
        life_history_profile = production_lifecycle_life_history_profile_v1(
            str(
                parameters.get(
                    "production_lifecycle.life_history_profile",
                    PRODUCTION_DEFAULT_LIFE_HISTORY_PROFILE_V1,
                )
            )
        )
        if (
            lifecycle_mode == LEGACY_SYNCHRONIZED_TRUNCATION_V1
            and life_history_profile["life_history_mode"] != 0
        ):
            raise PatchRunnerFactoryError(
                "legacy synchronized lifecycle requires inert life history"
            )
        sim_cfg.life_history_profile_id = str(life_history_profile["profile_id"])
        for name in (
            "developmental_maturity_age",
            "life_history_mode",
            "life_history_maturity_start_ticks",
            "life_history_maturity_rise_ticks",
            "life_history_fertility_decline_start_ticks",
            "life_history_fertility_decline_ticks",
            "life_history_elder_fertility_floor",
            "life_history_senescence_start_ticks",
            "life_history_senescence_ramp_ticks",
            "life_history_senescence_energy_cost",
            "life_history_senescence_health_cost",
            "life_history_max_age_ticks",
        ):
            setattr(sim_cfg, name, life_history_profile[name])
        sim_cfg.ordinary_births_enabled = ordinary_births_enabled
        sim_cfg.generation_turnover_enabled = lifecycle_mode_abi == 0
        sim_cfg.population_size = population
        sim_cfg.cull_fraction = expected_cull_fraction
        sim_cfg.parent_fraction = 1.0
        sim_cfg.lifecycle_mode = lifecycle_mode_abi
        sim_cfg.turnover_selection_mode = 1
        sim_cfg.turnover_selection_seed = derive_production_turnover_seed_v1(
            cell_id=cell_id,
            seeds=seeds,
        )
    # Production campaign runners always carry the passive GEN1-F1 recorder.
    # The recorder is observer-only and does not alter RNG ownership or results.
    sim_cfg.gen1_f1_capture_enable = True
    sim_cfg.native_runtime_write_summary = False
    sim_cfg.runnercore1_write_summary = False
    sim_cfg.developmental_trace_write_enable = False
    sim_cfg.checkpoint_bundle_v2_period_gen = 0
    sim_cfg.checkpoint_period_gen = 0
    sim_cfg.checkpoint_rotate = 0
    sim_cfg.phase_e_e1_enable = False
    sim_cfg.phase_e_e2_enable = False
    sim_cfg.epi_enable = False
    sim_cfg.enable_cognitive_budget_ledger = False
    sim_cfg.maintenance_enable = False
    sim_cfg.chemistry_functional_coupling_enable = False
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c2_enable = False
    sim_cfg.inheritance_profile = (
        DIPLOID_INHERITANCE_PROFILE
        if genome_upgrade_v2_enabled
        else sim_cfg.inheritance_profile
    )
    sim_cfg.genome_upgrade_v2_enable = genome_upgrade_v2_enabled
    if genome_upgrade_v2_enabled:
        sim_cfg.genome_v2_checkpoint_authority_binding = tuple(
            sorted(dict(genome_v2_checkpoint_authority_binding or {}).items())
        )
        mutation_enabled = bool(
            parameters["genome_upgrade_v2.mutation_rate_q10"] > 0
            and parameters["genome_upgrade_v2.mutation_step_q10"] > 0
        )
        sim_cfg.genome_v2_causal_capture_enable = lifecycle_enabled
        sim_cfg.diploid_mutation_enable = (
            mutation_enabled if lifecycle_enabled else True
        )
        sim_cfg.recomb72_mutation_off = (
            not mutation_enabled if lifecycle_enabled else False
        )
        sim_cfg.genome_upgrade_v2_mutation_rate_q10 = int(
            parameters["genome_upgrade_v2.mutation_rate_q10"]
        )
        sim_cfg.genome_upgrade_v2_mutation_step_q10 = int(
            parameters["genome_upgrade_v2.mutation_step_q10"]
        )
        sim_cfg.genome_upgrade_v2_founder_profile_id = str(
            founder_profile["profile_id"]
        )
        sim_cfg.genome_upgrade_v2_founder_profile_sha256 = str(
            founder_profile["payload_sha256"]
        )
    sim_cfg.pressure_autopilot_enable = False
    sim_cfg.pressure_autopilot_allow_shock_enable = False
    sim_cfg.shock_pulsed = False
    sim_cfg.shock_start_prob = 0.0
    sim_cfg.shock_period = 0
    sim_cfg.shock_duration = 0
    sim_cfg.force_prediction_disabled = True
    sim_cfg.ablate_prediction = True
    founder_receipt = _apply_founder_treatment(
        sim_cfg,
        requested=parameters["patch.founder_placement"],
        population=population,
    )

    run_path = Path(run_dir)
    logger = MetricsLogger(str(run_path))
    runner = runner_constructor(
        env_cfg=env_cfg,
        pop_cfg=PopConfig(
            population_size=population,
            max_slots_cap=capacity,
            parent_fraction=float(getattr(sim_cfg, "parent_fraction", 0.25)),
        ),
        sim_cfg=sim_cfg,
        logger=logger,
        run_dir=str(run_path),
        perf=False,
    )
    runner.cuda_architecture_b_inheritance_sidecar_cell_id = cell_id
    runner.cuda_architecture_b_genome_v2_causal_physical_cell_ordinal = 0
    if runner_constructor is SimulationRunner:
        prepare_patch_generation_zero_v1(runner)
    receipt = {
        "schema_version": PATCH_FIXED_RUNNER_RECEIPT_SCHEMA_V1,
        "cell_id": cell_id,
        "world_profile": (
            "patch-p2"
            if ecology_profile_id == PATCH_ECOLOGY_PROFILE_DEFAULT_V1
            else f"patch-p2-{ecology_profile_id}"
        ),
        "patch_ecology_profile": ecology_profile_receipt,
        "primitive_controls": primitive_control_receipt,
        "policy_family": "fixed_mlp",
        "dimensions": {
            "grid_size": grid_size,
            "initial_population": population,
            "configured_initial_live_count": initial_live_count,
            "resident_capacity": capacity,
            **runner_dimension_receipt,
        },
        "initial_occupancy": {
            "schema_version": "eiel_patch_initial_live_count_authority_v1",
            "configured_initial_live_count": initial_live_count,
            "physical_resident_capacity": capacity,
            "admitted_initial_live_counts": sorted(
                PATCH_LIFECYCLE_INITIAL_LIVE_COUNTS_V1
            )
            if lifecycle_enabled
            else [initial_live_count],
            "hard_cap_fuse": capacity,
        },
        "seed_ownership": {
            "named_owners": dict(sorted(seeds.items())),
            "engine_rng_root": master_seed,
            "engine_rng_root_derivation": "sha256(eiel_named_seed_owners_v1)[0:8]-signed63",
            "spatial_ecology_seed_owner": "ecology",
            "spatial_ecology_seed": seeds["ecology"],
        },
        "founder_initialization": founder_receipt,
        "initial_spatial_placement": {
            "schema_version": "eiel_initial_spatial_placement_profile_v1",
            "profile_id": sim_cfg.initial_spatial_placement_profile,
        },
        "genome": {
            "profile": (
                "genome_upgrade_v2_fixed_q10_v1"
                if genome_upgrade_v2_enabled
                else "patch_default"
            ),
            "enabled": genome_upgrade_v2_enabled,
            "inheritance_profile": sim_cfg.inheritance_profile,
            "mutation_rate_q10": (
                sim_cfg.genome_upgrade_v2_mutation_rate_q10
                if genome_upgrade_v2_enabled
                else None
            ),
            "mutation_step_q10": (
                sim_cfg.genome_upgrade_v2_mutation_step_q10
                if genome_upgrade_v2_enabled
                else None
            ),
            "founder_profile_id": (
                sim_cfg.genome_upgrade_v2_founder_profile_id
                if genome_upgrade_v2_enabled
                else None
            ),
            "founder_profile_sha256": (
                sim_cfg.genome_upgrade_v2_founder_profile_sha256
                if genome_upgrade_v2_enabled
                else None
            ),
            "e1_arm_label": (
                e1_arm_label if genome_upgrade_v2_enabled else None
            ),
        },
        "execution": {
            "runner_construction_device": "cpu",
            "resident_backend": execution_backend,
            "request_mode": sim_cfg.cuda_architecture_b_request_mode,
            "runtime_jit_allowed": False,
            "max_births_per_generation": max_births_per_generation,
            "output_profile": "compact_default",
        },
        "production_lifecycle": (
            {
                "enabled": True,
                "profile": lifecycle_profile,
                "lifecycle_mode": (
                    parameters.get(
                        "production_lifecycle.lifecycle_mode",
                        PRODUCTION_DEFAULT_LIFECYCLE_MODE_V1,
                    )
                ),
                "ordinary_births_enabled": sim_cfg.ordinary_births_enabled,
                "turnover_enabled": True,
                "turnover_cull_fraction": sim_cfg.cull_fraction,
                "turnover_parent_fraction": 1.0,
                "lifecycle_mode_abi": getattr(sim_cfg, "lifecycle_mode", 0),
                "diploid_mutation_enabled": getattr(
                    sim_cfg, "diploid_mutation_enable", False
                ),
                "legacy_scalar_mutation_off": getattr(
                    sim_cfg, "recomb72_mutation_off", False
                ),
                "genome_v2_mutation_enabled": getattr(
                    sim_cfg, "diploid_mutation_enable", False
                ),
                "genome_v2_mutation_rate_q10": (
                    getattr(sim_cfg, "genome_upgrade_v2_mutation_rate_q10", 0)
                ),
                "genome_v2_mutation_step_q10": (
                    getattr(sim_cfg, "genome_upgrade_v2_mutation_step_q10", 0)
                ),
                "genome_v2_mutation_regime": (
                    "fixed_q10_low_admitted"
                    if getattr(sim_cfg, "diploid_mutation_enable", False)
                    else "fixed_q10_off_control"
                ),
                "turnover_selection_mode": "genotype_blind_neutral",
                "turnover_selection_mode_abi": sim_cfg.turnover_selection_mode,
                "turnover_selection_seed": sim_cfg.turnover_selection_seed,
                "turnover_selection_seed_schema_version": (
                    "eiel_production_turnover_selection_seed_v1"
                ),
                "genome_v2_causal_capture_enabled": bool(genome_upgrade_v2_enabled),
                "life_history": {
                    name: life_history_profile[name]
                    for name in (
                        "profile_id",
                        "schema_version",
                        "developmental_maturity_age",
                        "life_history_mode",
                        "life_history_maturity_start_ticks",
                        "life_history_maturity_rise_ticks",
                        "life_history_fertility_decline_start_ticks",
                        "life_history_fertility_decline_ticks",
                        "life_history_elder_fertility_floor",
                        "life_history_senescence_start_ticks",
                        "life_history_senescence_ramp_ticks",
                        "life_history_senescence_energy_cost",
                        "life_history_senescence_health_cost",
                        "life_history_max_age_ticks",
                    )
                },
            }
            if lifecycle_enabled
            else None
        ),
        "artifact_bindings": artifact_bindings,
    }
    return PatchFixedRunnerBuildV1(runner=runner, receipt=receipt)


def build_patch_stable_gru_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int = 1,
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchStableGruRunnerBuildV1:
    """Build one Patch/stable-GRU runner with no WDL world mechanics."""

    if isinstance(cell, LogicalCellV1):
        raw = cell.payload()
    elif isinstance(cell, Mapping):
        raw = dict(cell)
    else:
        raise PatchRunnerFactoryError("cell must be a LogicalCellV1 or mapping")
    parameters = raw.get("parameters")
    seeds = raw.get("seeds")
    actual = set(parameters) if isinstance(parameters, Mapping) else set()
    allowed = PATCH_STABLE_GRU_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
    if not isinstance(parameters, Mapping) or not (
        PATCH_STABLE_GRU_PARAMETER_NAMES_V1 <= actual <= allowed
    ):
        raise PatchRunnerFactoryError(
            "Patch/stable-GRU parameters must include the public contract; "
            f"missing={sorted(PATCH_STABLE_GRU_PARAMETER_NAMES_V1 - actual)} "
            f"extra={sorted(actual - allowed)}"
        )
    if not isinstance(seeds, Mapping):
        raise PatchRunnerFactoryError("compiled Patch/GRU cell is missing named seeds")
    if parameters["policy.hidden_width"] != 64:
        raise PatchRunnerFactoryError("qualified stable-GRU policy.hidden_width must equal 64")
    if parameters["policy.reset_hidden_each_tick"] is not False:
        raise PatchRunnerFactoryError(
            "reset-hidden GRU is a distinct policy arm and is not the stable-GRU provider"
        )

    patch_cell = dict(raw)
    patch_cell["parameters"] = {
        name: parameters[name]
        for name in (PATCH_FIXED_PARAMETER_NAMES_V1 | (actual & PATCH_PRIMITIVE_PARAMETER_NAMES_V1))
    }

    def capture_constructor(**kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)

    base = build_patch_fixed_runner_v1(
        patch_cell,
        run_dir,
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        artifact_path=artifact_path,
        qualification_receipt_path=qualification_receipt_path,
        reporting_receipt_path=reporting_receipt_path,
        max_births_per_generation=max_births_per_generation,
        runner_constructor=capture_constructor,
    )
    captured = base.runner
    sim_cfg = captured.sim_cfg
    sim_cfg.native_runtime_fixed_policy_family = "gru"
    sim_cfg.native_runtime_use_precomputed_drive_noise = (
        int(getattr(sim_cfg, "rng_semantics_version", 1) or 1) != 2
    )
    sim_cfg.policy_weight_mutation_enable = False
    sim_cfg.behavior_log_enable = True
    sim_cfg.lineage_observability_enable = True
    runner = runner_constructor(
        env_cfg=captured.env_cfg,
        pop_cfg=captured.pop_cfg,
        sim_cfg=sim_cfg,
        logger=captured.logger,
        run_dir=captured.run_dir,
        perf=captured.perf,
    )
    _restore_stable_gru_runtime_contract_v1(runner)
    master_seed = derive_patch_master_seed_v1(dict(seeds))
    _materialize_stable_gru_generation_zero_v1(runner, master_seed=master_seed)
    receipt = {
        **dict(base.receipt),
        "schema_version": PATCH_STABLE_GRU_RUNNER_RECEIPT_SCHEMA_V1,
        "world_profile": "patch-p2-v1",
        "policy_family": "stable_gru",
        "policy": {"hidden_width": 64, "reset_hidden_each_tick": False},
    }
    return PatchStableGruRunnerBuildV1(runner=runner, receipt=receipt)


def build_patch_eed_fixed_mlp_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int = 3,
    genome_v2_checkpoint_authority_binding: Mapping[str, str] | None = None,
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchEedFixedMlpRunnerBuildV1:
    """Compose the frozen reciprocal-impact overlay with fixed-MLP Patch."""

    if isinstance(cell, LogicalCellV1):
        raw = cell.payload()
    elif isinstance(cell, Mapping):
        raw = dict(cell)
    else:
        raise PatchRunnerFactoryError(
            "cell must be a LogicalCellV1 or mapping"
        )
    parameters = raw.get("parameters")
    if not isinstance(parameters, Mapping):
        raise PatchRunnerFactoryError(
            "compiled EED cell is missing parameters"
        )
    actual = set(parameters)
    allowed = (
        EED_FIXED_MLP_PARAMETER_NAMES_V1
        | GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
        | PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
        | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        | EED_PRIMITIVE_PARAMETER_NAMES_V1
        | CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1
    )
    required = EED_FIXED_MLP_PARAMETER_NAMES_V1
    if not required <= actual or not actual <= allowed:
        raise PatchRunnerFactoryError(
            "EED/fixed-MLP parameters must match the public contract; "
            f"missing={sorted(required - actual)} "
            f"extra={sorted(actual - allowed)}"
        )
    impact_applied = parameters["eed.resource_impact_apply_enable"]
    if type(impact_applied) is not bool:
        raise PatchRunnerFactoryError(
            "eed.resource_impact_apply_enable must be an exact boolean"
        )
    ecology_frozen = parameters["eed.ecology_frozen"]
    if type(ecology_frozen) is not bool:
        raise PatchRunnerFactoryError(
            "eed.ecology_frozen must be an exact boolean"
        )
    phenotype_fixed = parameters["eed.phenotype_fixed"]
    if type(phenotype_fixed) is not bool:
        raise PatchRunnerFactoryError(
            "eed.phenotype_fixed must be an exact boolean"
        )
    founder_profile = parameters["eed.founder_resource_axis_profile"]
    if founder_profile not in {"neutral", "a_high", "b_high", "symmetric_mix"}:
        raise PatchRunnerFactoryError(
            "eed.founder_resource_axis_profile must be neutral, a_high, "
            "b_high, or symmetric_mix"
        )
    patch_cell = {
        **raw,
        "parameters": {
            name: value
            for name, value in parameters.items()
            if name not in (
                (EED_FIXED_MLP_PARAMETER_NAMES_V1 - PATCH_FIXED_PARAMETER_NAMES_V1)
                | (EED_PRIMITIVE_PARAMETER_NAMES_V1 - PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
            )
        },
    }

    def capture_constructor(**kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)

    base = build_patch_fixed_runner_v1(
        patch_cell,
        run_dir,
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        artifact_path=artifact_path,
        qualification_receipt_path=qualification_receipt_path,
        reporting_receipt_path=reporting_receipt_path,
        max_births_per_generation=max_births_per_generation,
        genome_v2_checkpoint_authority_binding=(
            genome_v2_checkpoint_authority_binding
        ),
        runner_constructor=capture_constructor,
    )
    captured = base.runner
    env_cfg = captured.env_cfg
    sim_cfg = captured.sim_cfg
    eed = _configure_eed_overlay_v1(
        env_cfg,
        sim_cfg,
        parameters,
        execution_backend="cuda",
    )
    runner = runner_constructor(
        env_cfg=env_cfg,
        pop_cfg=captured.pop_cfg,
        sim_cfg=sim_cfg,
        logger=captured.logger,
        run_dir=captured.run_dir,
        perf=captured.perf,
    )
    if runner_constructor is SimulationRunner:
        from engine.evolution.diploid_inheritance_v1 import (
            attach_diploid_genome,
            homozygous_genome_from_phenotype,
        )

        prepare_patch_generation_zero_v1(runner)
        agents = sorted(
            runner.pop.alive_agents(), key=lambda agent: int(agent.agent_id)
        )
        for index, agent in enumerate(agents):
            axis = _eed_founder_resource_axis_v1(founder_profile, index)
            agent.genome.resource_handling_axis = axis
            attach_diploid_genome(
                agent.genome,
                homozygous_genome_from_phenotype(agent.genome),
        )
        mark_dirty = getattr(runner, "_mark_bank_dirty", None)
        if callable(mark_dirty):
            mark_dirty()
        runner.env._reset_patch_world_arrays()
        from engine.env.gridworld_pkg.patch_world import (
            project_eed_resources_to_patch_food,
        )

        project_eed_resources_to_patch_food(runner.env)
    receipt = {
        **dict(base.receipt),
        "schema_version": EED_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1,
        "world_profile": "patch-p2-eed-resource-impact-v1",
        "overlay": eed,
    }
    return PatchEedFixedMlpRunnerBuildV1(runner=runner, receipt=receipt)


def _eed_founder_resource_axis_v1(profile: str, index: int) -> float:
    if profile == "neutral":
        return 0.0
    if profile == "a_high":
        return 1.0
    if profile == "b_high":
        return -1.0
    if profile == "symmetric_mix":
        return 1.0 if int(index) % 2 == 0 else -1.0
    raise PatchRunnerFactoryError(
        f"unsupported EED founder resource-axis profile: {profile!r}"
    )


def build_patch_jungle_fixed_mlp_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int = 1,
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchJungleFixedMlpRunnerBuildV1:
    """Compose fixed Patch with the single qualified Jungle-v4 configuration."""

    raw = asdict(cell) if isinstance(cell, LogicalCellV1) else dict(cell)
    parameters = raw.get("parameters")
    if not isinstance(parameters, Mapping):
        raise PatchRunnerFactoryError("compiled Jungle cell is missing parameters")
    patch_cell = dict(raw)
    patch_cell["parameters"] = {
        name: value
        for name, value in parameters.items()
        if name not in (JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1 - PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
    }

    def capture_constructor(**kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)

    base = build_patch_fixed_runner_v1(
        patch_cell,
        run_dir,
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        artifact_path=artifact_path,
        qualification_receipt_path=qualification_receipt_path,
        reporting_receipt_path=reporting_receipt_path,
        max_births_per_generation=max_births_per_generation,
        runner_constructor=capture_constructor,
    )
    captured = base.runner
    env_cfg = captured.env_cfg
    sim_cfg = captured.sim_cfg
    jungle = _configure_jungle_overlay_v1(env_cfg, sim_cfg, parameters)
    runner = runner_constructor(
        env_cfg=env_cfg,
        pop_cfg=captured.pop_cfg,
        sim_cfg=sim_cfg,
        logger=captured.logger,
        run_dir=captured.run_dir,
        perf=captured.perf,
    )
    if runner_constructor is SimulationRunner:
        prepare_patch_generation_zero_v1(runner)
    receipt = {
        **dict(base.receipt),
        "schema_version": PATCH_JUNGLE_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1,
        "world_profile": "patch-p2-jungle-v4",
        "policy_family": "fixed_mlp",
        "overlay": jungle,
    }
    return PatchJungleFixedMlpRunnerBuildV1(runner=runner, receipt=receipt)


def _configure_jungle_overlay_v1(
    env_cfg: EnvConfig,
    sim_cfg: SimConfig,
    parameters: Mapping[str, Scalar],
) -> dict[str, Any]:
    """Apply the single canonical Jungle-v4 configuration."""

    env_cfg.food_spawn_per_step = 10
    env_cfg.max_food = 64
    env_cfg.reproduce_try_prob = 1.0
    env_cfg.reproduce_parent_energy_share = 0.2
    env_cfg.reproduce_min_energy_buffer = 20.0
    env_cfg.reproduce_cooldown_steps = 35
    env_cfg.jungle_predator_pressure_enable = True
    env_cfg.jungle_predator_base_risk = 0.02
    env_cfg.jungle_predator_scent_weight = 0.01
    env_cfg.jungle_predator_waste_weight = 0.01
    env_cfg.jungle_predator_density_weight = 0.01
    env_cfg.jungle_predator_visibility_weight = 0.01
    env_cfg.jungle_predator_cover_protection_weight = 0.02
    env_cfg.jungle_predator_shelter_protection_weight = 0.02
    env_cfg.jungle_predator_risk_min = 0.0
    env_cfg.jungle_predator_risk_cap = 0.05
    env_cfg.jungle_predator_health_damage = 0.02
    env_cfg.jungle_juvenile_predator_vulnerability_enable = True
    env_cfg.jungle_juvenile_age_threshold = 10
    env_cfg.jungle_juvenile_predator_damage_multiplier = 1.5
    env_cfg.jungle_juvenile_vulnerability_cap = 3.0
    env_cfg.jungle_terrain_protection_enable = True
    env_cfg.jungle_terrain_protection_apply_to_damage_enable = True
    env_cfg.jungle_cover_damage_reduction_weight = 0.25
    env_cfg.jungle_shelter_damage_reduction_weight = 0.25
    env_cfg.jungle_terrain_damage_multiplier_min = 0.25
    env_cfg.jungle_reproduction_emissions_enable = True
    env_cfg.jungle_birth_emission_at_child_cell_enable = True
    env_cfg.jungle_birth_emission_clamp_enable = True
    env_cfg.jungle_birth_scent_emit = 0.05
    env_cfg.jungle_birth_waste_emit = 0.05
    env_cfg.jungle_ecological_recovery_enable = True
    env_cfg.jungle_recovery_activity_decay_per_step = 0.02
    env_cfg.jungle_recovery_activity_base_growth = 0.01
    env_cfg.jungle_recovery_activity_cover_weight = 0.02
    env_cfg.jungle_recovery_activity_fertility_weight = 0.02
    env_cfg.jungle_recovery_activity_waste_penalty_weight = 0.05
    env_cfg.jungle_recovery_activity_density_penalty_weight = 0.05
    env_cfg.jungle_recovery_activity_scent_penalty_weight = 0.02
    env_cfg.jungle_recovery_regrowth_boost_enable = True
    env_cfg.jungle_recovery_regrowth_boost_scale = 0.1
    env_cfg.jungle_recovery_regrowth_boost_max = 0.25
    sim_cfg.behavior_log_enable = True
    sim_cfg.lineage_observability_enable = True
    sim_cfg.native_runtime_fixed_policy_family = "mlp"
    sim_cfg.native_runtime_cxx26_scope_lock_enable = True
    sim_cfg.native_runtime_allow_runtime_jit_build = False
    sim_cfg.native_runtime_use_precomputed_drive_noise = (
        int(getattr(sim_cfg, "rng_semantics_version", 1) or 1) != 2
    )
    sim_cfg.pressure_autopilot_allow_shock_enable = False
    sim_cfg.genome_c1_enable = False
    sim_cfg.genome_c2_enable = False
    sim_cfg.policy_deterministic_actions = False
    primitive_controls = apply_primitive_overrides_v1(
        env_cfg,
        sim_cfg,
        {
            name: value
            for name, value in parameters.items()
            if name
            in (JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
        },
        allowed_parameter_names=(
            JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        ),
    )
    return {
        "configuration_id": "canonical-jungle-v4",
        "predator_pressure": True,
        "juvenile_vulnerability": True,
        "terrain_protection": True,
        "reproduction_emissions": True,
        "ecological_recovery": True,
        "primitive_controls": primitive_controls,
    }


def _configure_eed_overlay_v1(
    env_cfg: EnvConfig,
    sim_cfg: SimConfig,
    parameters: Mapping[str, Scalar],
    *,
    execution_backend: str = "cuda",
) -> dict[str, Any]:
    """Apply a deterministic EED overlay."""

    impact_applied = parameters["eed.resource_impact_apply_enable"]
    if type(impact_applied) is not bool:
        raise PatchRunnerFactoryError(
            "eed.resource_impact_apply_enable must be an exact boolean"
        )
    ecology_frozen = parameters["eed.ecology_frozen"]
    if type(ecology_frozen) is not bool:
        raise PatchRunnerFactoryError(
            "eed.ecology_frozen must be an exact boolean"
        )
    phenotype_fixed = parameters["eed.phenotype_fixed"]
    if type(phenotype_fixed) is not bool:
        raise PatchRunnerFactoryError(
            "eed.phenotype_fixed must be an exact boolean"
        )
    founder_profile = parameters["eed.founder_resource_axis_profile"]
    if founder_profile not in {"neutral", "a_high", "b_high", "symmetric_mix"}:
        raise PatchRunnerFactoryError(
            "eed.founder_resource_axis_profile must be neutral, a_high, "
            "b_high, or symmetric_mix"
        )
    env_cfg.food_spawn_per_step = 0
    env_cfg.max_food = 0
    env_cfg.patch_scent_gather_emit = 0.0
    env_cfg.patch_waste_consume_emit = 0.0
    if execution_backend == "cpu_native":
        env_cfg.respec0b_resource_specialization_enable = False
        env_cfg.respec0b_resource_class_mode = ""
        env_cfg.respec0b_resource_equalized_sham_enable = False
    else:
        env_cfg.respec0b_resource_specialization_enable = True
        env_cfg.respec0b_resource_class_mode = "vertical_split_ab"
        env_cfg.respec0b_resource_equalized_sham_enable = False
    env_cfg.eed_resource_impact_enable = True
    env_cfg.eed_resource_impact_apply_enable = impact_applied
    env_cfg.eed_ecology_frozen_enable = ecology_frozen
    env_cfg.eed_phenotype_fixed_enable = phenotype_fixed
    env_cfg.eed_phenotype_fixed_axis_profile = founder_profile
    env_cfg.eed_resource_a_initial_amount = 1.0
    env_cfg.eed_resource_b_initial_amount = 1.0
    env_cfg.eed_resource_a_capacity = 1.0
    env_cfg.eed_resource_b_capacity = 1.0
    env_cfg.eed_resource_a_renewal_per_step = 0.25
    env_cfg.eed_resource_b_renewal_per_step = 0.25
    env_cfg.eed_resource_impact_per_success = 0.25
    env_cfg.eed_resource_impact_decay_per_step = 0.25
    env_cfg.eed_resource_impact_max_load = 1.0
    # The resident CUDA mate-birth planner executes the admitted EED route at
    # probability one. Bind the source/checkpoint metadata to that exact
    # executed value rather than retaining EnvConfig's legacy 0.15 default.
    env_cfg.reproduce_try_prob = 1.0
    env_cfg.reproduce_requires_mate = True
    sim_cfg.inheritance_profile = DIPLOID_INHERITANCE_PROFILE
    sim_cfg.resource_handling_representation_enable = True
    sim_cfg.native_runtime_use_precomputed_drive_noise = (
        int(getattr(sim_cfg, "rng_semantics_version", 1) or 1) != 2
    )
    sim_cfg.native_runtime_cxx26_scope_lock_enable = True
    primitive_controls = apply_primitive_overrides_v1(
        env_cfg,
        sim_cfg,
        {
            name: value
            for name, value in parameters.items()
            if name
            in (EED_PRIMITIVE_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
        },
        allowed_parameter_names=(
            EED_PRIMITIVE_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        ),
    )
    return {
        "component_id": "world.overlay.eed_resource_impact.v1",
        "impact_applied": impact_applied,
        "ecology_frozen": ecology_frozen,
        "phenotype_fixed": phenotype_fixed,
        "phenotype_fixed_axis_profile": founder_profile,
        "founder_resource_axis_profile": founder_profile,
        "resource_a_initial_amount": env_cfg.eed_resource_a_initial_amount,
        "resource_b_initial_amount": env_cfg.eed_resource_b_initial_amount,
        "resource_a_capacity": env_cfg.eed_resource_a_capacity,
        "resource_b_capacity": env_cfg.eed_resource_b_capacity,
        "resource_a_renewal_per_step": env_cfg.eed_resource_a_renewal_per_step,
        "resource_b_renewal_per_step": env_cfg.eed_resource_b_renewal_per_step,
        "impact_per_success": env_cfg.eed_resource_impact_per_success,
        "impact_decay_per_step": env_cfg.eed_resource_impact_decay_per_step,
        "impact_max_load": env_cfg.eed_resource_impact_max_load,
        "reproduce_try_prob": env_cfg.reproduce_try_prob,
        "new_rng_draws": 0,
        "primitive_controls": primitive_controls,
    }


def _configure_wdl_overlay_v1(
    env_cfg: EnvConfig,
    sim_cfg: SimConfig,
    parameters: Mapping[str, Scalar],
    *,
    policy_family: str,
) -> dict[str, Any]:
    """Apply the single declarative WDL config translation for both policies."""

    intensity = parameters["wdl.intensity"]
    if type(intensity) is not float or not 0.0 <= intensity <= 1.0:
        raise PatchRunnerFactoryError("wdl.intensity must be a float in [0,1]")
    cue_visible = parameters["wdl.cue_visible_at_decision"]
    if type(cue_visible) is not bool:
        raise PatchRunnerFactoryError("wdl.cue_visible_at_decision must be boolean")
    env_cfg.wdl_delayed_hazard_enable = True
    env_cfg.wdl_delayed_hazard_cue_enable = cue_visible
    env_cfg.wdl_delayed_hazard_observation_enable = True
    env_cfg.wdl_delayed_hazard_observation_dropout_rate = 0.0
    env_cfg.wdl_delayed_hazard_damage_enable = True
    env_cfg.wdl_delayed_hazard_deterministic_schedule_enable = True
    env_cfg.wdl_delayed_hazard_delay_steps = _positive_integer(
        parameters, "wdl.delay_ticks"
    )
    env_cfg.wdl_delayed_hazard_cue_duration_steps = _positive_integer(
        parameters, "wdl.cue_duration_ticks"
    )
    env_cfg.wdl_delayed_hazard_active_duration_steps = _positive_integer(
        parameters, "wdl.active_duration_ticks"
    )
    if (
        env_cfg.wdl_delayed_hazard_cue_duration_steps
        > env_cfg.wdl_delayed_hazard_delay_steps
    ):
        raise PatchRunnerFactoryError(
            "wdl.cue_duration_ticks must not exceed wdl.delay_ticks"
        )
    env_cfg.wdl_delayed_hazard_event_period_steps = max(
        20,
        env_cfg.wdl_delayed_hazard_delay_steps
        + env_cfg.wdl_delayed_hazard_active_duration_steps
        + 1,
    )
    env_cfg.wdl_delayed_hazard_radius = 1
    env_cfg.wdl_delayed_hazard_intensity = intensity
    primitive_controls = apply_primitive_overrides_v1(
        env_cfg,
        sim_cfg,
        {
            name: value
            for name, value in parameters.items()
            if name
            in (WDL_PRIMITIVE_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
        },
        allowed_parameter_names=(
            WDL_PRIMITIVE_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        ),
    )
    minimum_period = (
        env_cfg.wdl_delayed_hazard_delay_steps
        + env_cfg.wdl_delayed_hazard_active_duration_steps
        + 1
    )
    if env_cfg.wdl_delayed_hazard_event_period_steps < minimum_period:
        raise PatchRunnerFactoryError(
            "env_cfg.wdl_delayed_hazard_event_period_steps must be at least "
            "delay + active duration + 1"
        )
    sim_cfg.native_runtime_fixed_policy_family = policy_family
    sim_cfg.native_runtime_use_precomputed_drive_noise = (
        int(getattr(sim_cfg, "rng_semantics_version", 1) or 1) != 2
    )
    sim_cfg.policy_weight_mutation_enable = False
    sim_cfg.behavior_log_enable = True
    sim_cfg.lineage_observability_enable = True
    return {
        "delay_ticks": env_cfg.wdl_delayed_hazard_delay_steps,
        "cue_duration_ticks": env_cfg.wdl_delayed_hazard_cue_duration_steps,
        "active_duration_ticks": env_cfg.wdl_delayed_hazard_active_duration_steps,
        "intensity": intensity,
        "cue_visible_at_decision": cue_visible,
        "event_period_steps": env_cfg.wdl_delayed_hazard_event_period_steps,
        "radius": env_cfg.wdl_delayed_hazard_radius,
        "observation_dropout_rate": env_cfg.wdl_delayed_hazard_observation_dropout_rate,
        "primitive_controls": primitive_controls,
    }


def build_patch_wdl_fixed_mlp_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int | None = None,
    genome_v2_checkpoint_authority_binding: Mapping[str, str] | None = None,
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchWdlFixedMlpRunnerBuildV1:
    """Compose the fixed-Patch runner with the shared WDL overlay translation."""

    raw = asdict(cell) if isinstance(cell, LogicalCellV1) else dict(cell)
    parameters = raw.get("parameters")
    seeds = raw.get("seeds")
    actual = set(parameters) if isinstance(parameters, Mapping) else set()
    allowed = (
        WDL_FIXED_MLP_PARAMETER_NAMES_V1
        | GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
        | PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
        | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        | WDL_PRIMITIVE_PARAMETER_NAMES_V1
    )
    valid_base = WDL_FIXED_MLP_PARAMETER_NAMES_V1 <= actual <= allowed
    valid_genome = not (actual & GENOME_UPGRADE_V2_PARAMETER_NAMES_V1) or (
        GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1 <= actual
    )
    if not (valid_base and valid_genome):
        raise PatchRunnerFactoryError(
            "WDL/fixed-MLP parameters must match the public contract; "
            f"missing={sorted(WDL_FIXED_MLP_PARAMETER_NAMES_V1 - actual)} "
            f"extra={sorted(actual - allowed)}"
        )
    if not isinstance(seeds, Mapping) or "task" not in seeds:
        raise PatchRunnerFactoryError("compiled WDL cell is missing its task seed owner")
    if max_births_per_generation is None:
        max_births_per_generation = (
            3 if parameters.get("genome_upgrade_v2.enable") is True else 1
        )
    patch_cell = dict(raw)
    patch_cell["parameters"] = {
        name: parameters[name]
        for name in (
            PATCH_FIXED_PARAMETER_NAMES_V1
            | (
                actual & GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
            )
            | (actual & PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1)
            | (actual & PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
        )
    }

    def capture_constructor(**kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)

    base = build_patch_fixed_runner_v1(
        patch_cell,
        run_dir,
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        artifact_path=artifact_path,
        qualification_receipt_path=qualification_receipt_path,
        reporting_receipt_path=reporting_receipt_path,
        max_births_per_generation=max_births_per_generation,
        genome_v2_checkpoint_authority_binding=(
            genome_v2_checkpoint_authority_binding
        ),
        runner_constructor=capture_constructor,
    )
    captured = base.runner
    wdl = _configure_wdl_overlay_v1(
        captured.env_cfg,
        captured.sim_cfg,
        parameters,
        policy_family="mlp",
    )
    runner = runner_constructor(
        env_cfg=captured.env_cfg,
        pop_cfg=captured.pop_cfg,
        sim_cfg=captured.sim_cfg,
        logger=captured.logger,
        run_dir=captured.run_dir,
        perf=captured.perf,
    )
    if hasattr(runner, "env"):
        runner.env.base_seed = int(seeds["task"])
    if runner_constructor is SimulationRunner:
        prepare_patch_generation_zero_v1(runner)
    receipt = {
        **dict(base.receipt),
        "schema_version": WDL_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1,
        "world_profile": "patch-p2-wdl-v2",
        "policy_family": "fixed_mlp",
        "wdl": {
            "task_seed_owner": "task",
            "task_seed": int(seeds["task"]),
            **wdl,
        },
    }
    return PatchWdlFixedMlpRunnerBuildV1(runner=runner, receipt=receipt)


def build_patch_wdl_jungle_fixed_mlp_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int | None = None,
    genome_v2_checkpoint_authority_binding: Mapping[str, str] | None = None,
    execution_backend: str = "cuda",
    initialize_fresh_source: bool = True,
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchWdlJungleFixedMlpRunnerBuildV1:
    """Compose the exact requested WDL+Jungle fixed-MLP configuration.

    This factory only owns deterministic runner configuration and receipt
    composition. Direct-resident execution admission remains a separate
    production binding and qualification step.
    """

    raw = asdict(cell) if isinstance(cell, LogicalCellV1) else dict(cell)
    parameters = raw.get("parameters")
    seeds = raw.get("seeds")
    actual = set(parameters) if isinstance(parameters, Mapping) else set()
    if not (
        WDL_FIXED_MLP_PARAMETER_NAMES_V1
        | GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1
        <= actual
        and actual
        <= WDL_FIXED_MLP_PARAMETER_NAMES_V1
        | GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
        | PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
        | CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1
        | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        | WDL_PRIMITIVE_PARAMETER_NAMES_V1
        | JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1
    ):
        raise PatchRunnerFactoryError(
            "WDL+Jungle/fixed-MLP parameters must include WDL and one complete "
            "Genome V2 profile; "
            f"missing={sorted((WDL_FIXED_MLP_PARAMETER_NAMES_V1 | GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1) - actual)} "
            f"extra={sorted(actual - (WDL_FIXED_MLP_PARAMETER_NAMES_V1 | GENOME_UPGRADE_V2_PARAMETER_NAMES_V1 | PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1 | CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1 | PATCH_PRIMITIVE_PARAMETER_NAMES_V1 | WDL_PRIMITIVE_PARAMETER_NAMES_V1 | JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1))}"
        )
    if not isinstance(seeds, Mapping) or "task" not in seeds:
        raise PatchRunnerFactoryError(
            "compiled WDL+Jungle cell is missing its task seed owner"
        )
    if max_births_per_generation is None:
        max_births_per_generation = 3
    patch_cell = dict(raw)
    patch_cell["parameters"] = {
        name: parameters[name]
        for name in (
            PATCH_FIXED_PARAMETER_NAMES_V1
            | (actual & GENOME_UPGRADE_V2_PARAMETER_NAMES_V1)
            | (actual & PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1)
            | (actual & CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1)
            | (actual & PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
        )
    }

    def capture_constructor(**kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)

    base = build_patch_fixed_runner_v1(
        patch_cell,
        run_dir,
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        artifact_path=artifact_path,
        qualification_receipt_path=qualification_receipt_path,
        reporting_receipt_path=reporting_receipt_path,
        max_births_per_generation=max_births_per_generation,
        genome_v2_checkpoint_authority_binding=(
            genome_v2_checkpoint_authority_binding
        ),
        execution_backend=execution_backend,
        runner_constructor=capture_constructor,
    )
    captured = base.runner
    wdl = _configure_wdl_overlay_v1(
        captured.env_cfg,
        captured.sim_cfg,
        parameters,
        policy_family="mlp",
    )
    jungle = _configure_jungle_overlay_v1(captured.env_cfg, captured.sim_cfg, parameters)
    runner = runner_constructor(
        env_cfg=captured.env_cfg,
        pop_cfg=captured.pop_cfg,
        sim_cfg=captured.sim_cfg,
        logger=captured.logger,
        run_dir=captured.run_dir,
        perf=captured.perf,
    )
    if hasattr(runner, "env"):
        runner.env.base_seed = int(seeds["task"])
    if type(initialize_fresh_source) is not bool:
        raise PatchRunnerFactoryError("initialize_fresh_source must be an exact bool")
    if runner_constructor is SimulationRunner and initialize_fresh_source:
        prepare_patch_generation_zero_v1(runner)
    receipt = {
        **dict(base.receipt),
        "schema_version": WDL_JUNGLE_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1,
        "world_profile": "patch-p2-wdl-v2-jungle-v4",
        "policy_family": "fixed_mlp",
        "wdl": {
            "task_seed_owner": "task",
            "task_seed": int(seeds["task"]),
            **wdl,
        },
        "jungle": jungle,
    }
    return PatchWdlJungleFixedMlpRunnerBuildV1(runner=runner, receipt=receipt)


def build_patch_wdl_jungle_eed_fixed_mlp_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int | None = None,
    ticks_per_generation: int | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int | None = None,
    genome_v2_checkpoint_authority_binding: Mapping[str, str] | None = None,
    execution_backend: str = "cuda",
    initialize_fresh_source: bool = True,
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchWdlJungleFixedMlpRunnerBuildV1:
    """Compose patch+WDL+Jungle+EED for the exact CPU/GPU direct-resident path."""

    raw = asdict(cell) if isinstance(cell, LogicalCellV1) else dict(cell)
    parameters = raw.get("parameters")
    seeds = raw.get("seeds")
    actual = set(parameters) if isinstance(parameters, Mapping) else set()
    required = (
        WDL_FIXED_MLP_PARAMETER_NAMES_V1
        | GENOME_UPGRADE_V2_CORE_PARAMETER_NAMES_V1
        | EED_FIXED_MLP_PARAMETER_NAMES_V1
    )
    allowed = (
        WDL_FIXED_MLP_PARAMETER_NAMES_V1
        | GENOME_UPGRADE_V2_PARAMETER_NAMES_V1
        | EED_FIXED_MLP_PARAMETER_NAMES_V1
        | PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1
        | CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1
        | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        | WDL_PRIMITIVE_PARAMETER_NAMES_V1
        | JUNGLE_PRIMITIVE_PARAMETER_NAMES_V1
        | EED_PRIMITIVE_PARAMETER_NAMES_V1
        | {
            "hapb_frag.identity_reuse",
            "hapb_frag.force_physical_refill",
            "tw_rngv2.semantics_version",
            "tw_rngv2.world_uid",
        }
    )
    if not (required <= actual and actual <= allowed):
        raise PatchRunnerFactoryError(
            "WDL+Jungle+EED/fixed-MLP parameters must include WDL+EED+"
            "Genome V2 core parameters; "
            f"missing={sorted(required - actual)} "
            f"extra={sorted(actual - allowed)}"
        )
    if not isinstance(seeds, Mapping) or "task" not in seeds:
        raise PatchRunnerFactoryError(
            "compiled WDL+Jungle+EED cell is missing its task seed owner"
        )
    if max_births_per_generation is None:
        max_births_per_generation = 3
    patch_cell = dict(raw)
    patch_cell["parameters"] = {
        name: parameters[name]
        for name in (
            PATCH_FIXED_PARAMETER_NAMES_V1
            | (actual & GENOME_UPGRADE_V2_PARAMETER_NAMES_V1)
            | (actual & PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1)
            | (actual & CUDA_ARCHITECTURE_B_PARAMETER_NAMES_V1)
            | (actual & PATCH_PRIMITIVE_PARAMETER_NAMES_V1)
        )
    }

    def capture_constructor(**kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)

    base = build_patch_fixed_runner_v1(
        patch_cell,
        run_dir,
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        tick_native_control=tick_native_control,
        artifact_path=artifact_path,
        qualification_receipt_path=qualification_receipt_path,
        reporting_receipt_path=reporting_receipt_path,
        max_births_per_generation=max_births_per_generation,
        genome_v2_checkpoint_authority_binding=(
            genome_v2_checkpoint_authority_binding
        ),
        execution_backend=execution_backend,
        runner_constructor=capture_constructor,
    )
    captured = base.runner
    wdl = _configure_wdl_overlay_v1(
        captured.env_cfg,
        captured.sim_cfg,
        parameters,
        policy_family="mlp",
    )
    jungle = _configure_jungle_overlay_v1(captured.env_cfg, captured.sim_cfg, parameters)
    eed = _configure_eed_overlay_v1(
        captured.env_cfg,
        captured.sim_cfg,
        parameters,
        execution_backend=execution_backend,
    )

    runner = runner_constructor(
        env_cfg=captured.env_cfg,
        pop_cfg=captured.pop_cfg,
        sim_cfg=captured.sim_cfg,
        logger=captured.logger,
        run_dir=captured.run_dir,
        perf=captured.perf,
    )
    if type(initialize_fresh_source) is not bool:
        raise PatchRunnerFactoryError("initialize_fresh_source must be an exact bool")
    if runner_constructor is SimulationRunner and initialize_fresh_source:
        from engine.evolution.diploid_inheritance_v1 import (
            attach_diploid_genome,
            homozygous_genome_from_phenotype,
        )

        prepare_patch_generation_zero_v1(runner)
        agents = sorted(
            runner.pop.alive_agents(), key=lambda agent: int(agent.agent_id)
        )
        founder_profile = str(parameters["eed.founder_resource_axis_profile"])
        for index, agent in enumerate(agents):
            axis = _eed_founder_resource_axis_v1(founder_profile, index)
            agent.genome.resource_handling_axis = axis
            attach_diploid_genome(
                agent.genome,
                homozygous_genome_from_phenotype(agent.genome),
            )
            mark_dirty = getattr(runner, "_mark_bank_dirty", None)
            if callable(mark_dirty):
                mark_dirty()
        runner.env._reset_patch_world_arrays()
        from engine.env.gridworld_pkg.patch_world import (
            project_eed_resources_to_patch_food,
        )

        project_eed_resources_to_patch_food(runner.env)
    if hasattr(runner, "env"):
        runner.env.base_seed = int(seeds["task"])

    receipt = {
        **dict(base.receipt),
        "schema_version": WDL_JUNGLE_EED_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1,
        "world_profile": "patch-p2-wdl-v2-jungle-v4-eed",
        "policy_family": "fixed_mlp",
        "wdl": {
            "task_seed_owner": "task",
            "task_seed": int(seeds["task"]),
            **wdl,
        },
        "jungle": jungle,
        "eed": eed,
    }
    return PatchWdlJungleFixedMlpRunnerBuildV1(runner=runner, receipt=receipt)


def build_patch_wdl_stable_gru_runner_v1(
    cell: LogicalCellV1 | Mapping[str, Any],
    run_dir: str | Path,
    *,
    generations: int,
    ticks_per_generation: int,
    artifact_path: str,
    qualification_receipt_path: str,
    reporting_receipt_path: str,
    max_births_per_generation: int = 1,
    runner_constructor: Callable[..., Any] = SimulationRunner,
) -> PatchWdlStableGruRunnerBuildV1:
    """Build one WDL/stable-GRU runner without relying on gate-script helpers."""

    if isinstance(cell, LogicalCellV1):
        raw = cell.payload()
    elif isinstance(cell, Mapping):
        raw = dict(cell)
    else:
        raise PatchRunnerFactoryError("cell must be a LogicalCellV1 or mapping")
    parameters = raw.get("parameters")
    seeds = raw.get("seeds")
    actual = set(parameters) if isinstance(parameters, Mapping) else set()
    allowed = (
        WDL_STABLE_GRU_PARAMETER_NAMES_V1
        | PATCH_PRIMITIVE_PARAMETER_NAMES_V1
        | WDL_PRIMITIVE_PARAMETER_NAMES_V1
    )
    if not isinstance(parameters, Mapping) or not (
        WDL_STABLE_GRU_PARAMETER_NAMES_V1 <= actual <= allowed
    ):
        raise PatchRunnerFactoryError(
            "WDL/stable-GRU parameters must include the public contract; "
            f"missing={sorted(WDL_STABLE_GRU_PARAMETER_NAMES_V1 - actual)} "
            f"extra={sorted(actual - allowed)}"
        )
    if not isinstance(seeds, Mapping) or "task" not in seeds:
        raise PatchRunnerFactoryError("compiled WDL cell is missing its task seed owner")
    if parameters["policy.hidden_width"] != 64:
        raise PatchRunnerFactoryError("qualified stable-GRU policy.hidden_width must equal 64")
    if parameters["policy.reset_hidden_each_tick"] is not False:
        raise PatchRunnerFactoryError(
            "reset-hidden GRU is a distinct policy arm and is not the stable-GRU provider"
        )
    intensity = parameters["wdl.intensity"]
    if type(intensity) is not float or not 0.0 <= intensity <= 1.0:
        raise PatchRunnerFactoryError("wdl.intensity must be a float in [0,1]")
    if type(parameters["wdl.cue_visible_at_decision"]) is not bool:
        raise PatchRunnerFactoryError("wdl.cue_visible_at_decision must be boolean")

    patch_cell = dict(raw)
    patch_cell["parameters"] = {
        name: parameters[name]
        for name in (PATCH_FIXED_PARAMETER_NAMES_V1 | (actual & PATCH_PRIMITIVE_PARAMETER_NAMES_V1))
    }

    def capture_constructor(**kwargs: Any) -> Any:
        return SimpleNamespace(**kwargs)

    base = build_patch_fixed_runner_v1(
        patch_cell,
        run_dir,
        generations=generations,
        ticks_per_generation=ticks_per_generation,
        artifact_path=artifact_path,
        qualification_receipt_path=qualification_receipt_path,
        reporting_receipt_path=reporting_receipt_path,
        max_births_per_generation=max_births_per_generation,
        runner_constructor=capture_constructor,
    )
    captured = base.runner
    env_cfg = captured.env_cfg
    sim_cfg = captured.sim_cfg
    wdl = _configure_wdl_overlay_v1(
        env_cfg,
        sim_cfg,
        parameters,
        policy_family="gru",
    )

    runner = runner_constructor(
        env_cfg=env_cfg,
        pop_cfg=captured.pop_cfg,
        sim_cfg=sim_cfg,
        logger=captured.logger,
        run_dir=captured.run_dir,
        perf=captured.perf,
    )
    _restore_stable_gru_runtime_contract_v1(runner)
    if hasattr(runner, "env"):
        runner.env.base_seed = int(seeds["task"])
    master_seed = derive_patch_master_seed_v1(dict(seeds))
    _materialize_stable_gru_generation_zero_v1(runner, master_seed=master_seed)
    receipt = {
        **dict(base.receipt),
        "schema_version": WDL_STABLE_GRU_RUNNER_RECEIPT_SCHEMA_V1,
        "world_profile": "patch-p2-wdl-v3",
        "policy_family": "stable_gru",
        "wdl": {
            "task_seed_owner": "task",
            "task_seed": int(seeds["task"]),
            **wdl,
        },
        "policy": {"hidden_width": 64, "reset_hidden_each_tick": False},
    }
    return PatchWdlStableGruRunnerBuildV1(runner=runner, receipt=receipt)


__all__ = [
    "EED_FIXED_MLP_PARAMETER_NAMES_V1",
    "EED_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1",
    "PATCH_FIXED_PARAMETER_NAMES_V1",
    "PATCH_FIXED_RUNNER_RECEIPT_SCHEMA_V1",
    "PATCH_JUNGLE_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1",
    "WDL_JUNGLE_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1",
    "WDL_JUNGLE_EED_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1",
    "PatchEedFixedMlpRunnerBuildV1",
    "PATCH_STABLE_GRU_PARAMETER_NAMES_V1",
    "PATCH_STABLE_GRU_RUNNER_RECEIPT_SCHEMA_V1",
    "PatchFixedRunnerBuildV1",
    "PatchJungleFixedMlpRunnerBuildV1",
    "PatchStableGruRunnerBuildV1",
    "PatchWdlFixedMlpRunnerBuildV1",
    "PatchWdlJungleFixedMlpRunnerBuildV1",
    "PatchWdlStableGruRunnerBuildV1",
    "PatchRunnerFactoryError",
    "build_patch_fixed_runner_v1",
    "build_patch_eed_fixed_mlp_runner_v1",
    "build_patch_jungle_fixed_mlp_runner_v1",
    "build_patch_wdl_jungle_eed_fixed_mlp_runner_v1",
    "build_patch_stable_gru_runner_v1",
    "build_patch_wdl_fixed_mlp_runner_v1",
    "build_patch_wdl_jungle_fixed_mlp_runner_v1",
    "build_patch_wdl_stable_gru_runner_v1",
    "derive_patch_master_seed_v1",
    "prepare_patch_generation_zero_v1",
    "WDL_STABLE_GRU_PARAMETER_NAMES_V1",
    "WDL_STABLE_GRU_RUNNER_RECEIPT_SCHEMA_V1",
    "WDL_FIXED_MLP_PARAMETER_NAMES_V1",
    "WDL_FIXED_MLP_RUNNER_RECEIPT_SCHEMA_V1",
]
