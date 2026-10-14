from __future__ import annotations

"""CXXBACKBONE-10 persistent native resident state owner."""

from contextlib import contextmanager
from dataclasses import dataclass
import copy
from functools import wraps
import hashlib
import importlib.util
import json
import math
import os
import random
import re
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.cpp_extension import load

from evolution.diploid_inheritance_v1 import (
    DIPLOID_INHERITANCE_PROFILE,
    DIPLOID_SCALAR_LOCI,
    attach_diploid_genome,
    diploid_genome_from_genome,
)
from simulation.runnercore.state import import_generation_start
from simulation.runnercore.action_cache_contract import (
    is_slot_keyed_policy_cache_entry,
    resolve_policy_action_cache_entry,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "engine" / "perf" / "native" / "cxxbackbone10_resident_state_owner.cpp"
CPU_COMPACT_SUMMARY_SOURCE = ROOT / "engine" / "perf" / "native" / "cpu_compact_summary_v1.cpp"
CPU_COMPACT_SUMMARY_DIGESTS_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "cpu_compact_summary_digests_v1.cpp"
)
CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "cpu_compact_summary_torch_adapter_v1.cpp"
)
CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_lifecycle_reduction_v1.cpp"
)
CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_identity_reduction_v1.cpp"
)
CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_reporting_ledger_v1.cpp"
)
CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_reporting_ledger_v1.hpp"
)
CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "native"
    / "cpu_compact_summary_source_composer_v1.cpp"
)
CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE = (
    ROOT
    / "engine"
    / "perf"
    / "cuda_architecture_b_compact_summary_host_canonical_v1.cpp"
)
WDL_NATIVE_CONTRACT_SOURCE = (
    ROOT / "engine" / "native_runtime" / "wdl" / "wdl_native_contract.cpp"
)
STABLE_GRU_NATIVE_SOURCE = (
    ROOT / "engine" / "perf" / "native" / "recurrent" / "stable_gru_v1.cpp"
)
ACS_NATIVE_STREAMED_CHECKPOINT_SOURCE = (
    ROOT
    / "engine"
    / "native_runtime"
    / "native"
    / "acs_native_streamed_checkpoint_v1.cpp"
)
ACS_NATIVE_STREAMED_CHECKPOINT_HEADER = (
    ROOT
    / "engine"
    / "native_runtime"
    / "native"
    / "acs_native_streamed_checkpoint_v1.hpp"
)
NATIVE_EXTENSION_SOURCES = (
    SOURCE,
    CPU_COMPACT_SUMMARY_SOURCE,
    CPU_COMPACT_SUMMARY_DIGESTS_SOURCE,
    CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE,
    CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE,
    CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE,
    CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE,
    CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE,
    ACS_NATIVE_STREAMED_CHECKPOINT_SOURCE,
    WDL_NATIVE_CONTRACT_SOURCE,
    STABLE_GRU_NATIVE_SOURCE,
)
NATIVE_EXTENSION_SOURCE_INVENTORY = (
    *NATIVE_EXTENSION_SOURCES,
    CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER,
    ACS_NATIVE_STREAMED_CHECKPOINT_HEADER,
)
NUMPY_INCLUDE = Path(np.get_include())
NUMPY_RANDOM_LIB = Path(np.__file__).resolve().parent / "random" / "lib" / "libnpyrandom.a"
SCHEMA_VERSION = "cxxbackbone10_resident_state_owner_v1"
CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1 = (
    "cxxbackbone10_checkpoint_projection_v1"
)
CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2 = (
    "cxxbackbone10_checkpoint_projection_v2"
)
OWNER_DESIGN_SCHEMA_VERSION = "cxxbackbone10_owner_design_decision_v1"
_MODULE: Any | None = None
RESIDENT_STATE_OWNER_EXTENSION_ENV = "EIEL_RESIDENT_STATE_OWNER_EXTENSION"
RESIDENT_STATE_OWNER_STABLE_MODULE_NAME = "cxxbackbone10_resident_state_owner_ext"
EVENT_TAPE_TENSOR_NAMES = (
    "event_tape_type",
    "event_tape_generation",
    "event_tape_tick",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
)
TYPED_LIFECYCLE_EVENT_TENSOR_NAMES = (
    "typed_lifecycle_event_event_type",
    "typed_lifecycle_event_generation",
    "typed_lifecycle_event_tick",
    "typed_lifecycle_event_sequence",
    "typed_lifecycle_event_actor_agent_id",
    "typed_lifecycle_event_target_agent_id",
    "typed_lifecycle_event_actor_slot",
    "typed_lifecycle_event_target_slot",
    "typed_lifecycle_event_parent_agent_id",
    "typed_lifecycle_event_child_agent_id",
    "typed_lifecycle_event_lineage_founder_id",
    "typed_lifecycle_event_lineage_depth",
    "typed_lifecycle_event_subtype_code",
    "typed_lifecycle_event_reason_code",
    "typed_lifecycle_event_success",
    "typed_lifecycle_event_value",
)
REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR = "_cxxbackbone10_reference_committed_death_contexts"
REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR = "_cxxbackbone10_reference_lifecycle_event_accumulator"
REFERENCE_LIFECYCLE_DEATH_CURSOR_ATTR = "_cxxbackbone10_reference_lifecycle_death_cursor"
REFERENCE_LIFECYCLE_REPRODUCTION_CURSOR_ATTR = "_cxxbackbone10_reference_lifecycle_reproduction_cursor"
RNG_MANAGER_STREAM_INDICES_METADATA = "native_runtime_rng_manager_stream_indices"
RNG_MANAGER_STREAM_NAMES = ("policy", "mutation", "metrics", "stress")
REFERENCE_GENOME_MATERIALIZATION_METADATA = "native_runtime_reference_genome_materialization"
REFERENCE_GENOME_MATERIALIZATION_SCHEMA = "native_runtime_reference_genome_materialization_v1"
REFERENCE_GENOME_SCALAR_MODE = "canonical_scalar_genes_v1"
REFERENCE_GENOME_MODULAR_MODE = "phase_e_modular_storage_exact_v1"
PASSIVE_MODULAR_ENABLE_METADATA = "native_runtime_modular_storage_enable"
PASSIVE_MODULAR_LIFECYCLE_ERROR = "PHASE_E_MODULAR_PASSIVE_LIFECYCLE_NOT_OWNED"
POLICY_ENERGY_OBSERVATION_CONTRACT_METADATA = "policy_energy_observation_contract_schema_version"
POLICY_ENERGY_OBSERVATION_CONTRACT_SCHEMA = "eiel_policy_energy_observation_shadow_v1"
PROJECTED_GENOME_GENE_FIELDS = frozenset(
    {
        "hidden_size",
        "exploration_bias",
        "aggression_bias",
        "reproduction_threshold",
        "mutation_rate",
        "action_temperature",
        "use_gru",
        "memory_dim",
        "sensor_channel_mask",
        "sensor_radius",
        "introspection_channel_mask",
        "self_regulation_control_mask",
        "self_modification_target_mask",
        "directed_inheritance_channel_mask",
        "drive_inertia",
        "drive_noise",
        "drive_logit_gain",
        "pred_lr",
        "drive_sp_hunger",
        "drive_sp_pain",
        "drive_sp_fatigue",
        "drive_sp_fear",
        "exp_prior_strength",
        "exp_prior_noise",
        "drive_birth_noise",
    }
)

PATCH_REQUIRED_ENV_ARRAYS = (
    "patch_food_amount",
    "patch_fertility",
    "patch_cover",
    "patch_shelter",
    "patch_movement_cost",
    "patch_scent",
    "patch_scent_emit_pending",
    "patch_waste",
    "patch_waste_emit_pending",
)
PATCH_P2_REQUIRED_ENV_ARRAYS = ("patch_recovery_activity",)
PATCH_DERIVED_ARRAYS = ("patch_visibility", "patch_density")
WDL_STORAGE_CAPABILITY = "wdl_storage"
WDL_STORAGE_SCHEMA_VERSION = "eiel_wdl_native_storage_v1"
WDL_STORAGE_ENV_ARRAYS = (
    ("patch_delayed_hazard_cue", np.dtype(np.float32)),
    ("patch_delayed_hazard_timer", np.dtype(np.int32)),
    ("patch_delayed_hazard_active", np.dtype(np.float32)),
    ("patch_delayed_hazard_intensity", np.dtype(np.float32)),
    ("patch_delayed_hazard_age", np.dtype(np.int32)),
    ("patch_delayed_hazard_event_id", np.dtype(np.int64)),
    ("patch_delayed_hazard_prior_cue", np.dtype(np.int8)),
)
EED_EXECUTION_CAPABILITY = "eed"
EED_STORAGE_SCHEMA_VERSION = "eiel_eed_resource_impact_native_storage_v1"
EED_GRID_ENV_ARRAYS = (
    ("eed_resource_a_amount", np.dtype(np.float32)),
    ("eed_resource_b_amount", np.dtype(np.float32)),
    ("eed_resource_a_capacity", np.dtype(np.float32)),
    ("eed_resource_b_capacity", np.dtype(np.float32)),
    ("eed_resource_a_renewal", np.dtype(np.float32)),
    ("eed_resource_b_renewal", np.dtype(np.float32)),
    ("eed_resource_a_pending_impact", np.dtype(np.float32)),
    ("eed_resource_b_pending_impact", np.dtype(np.float32)),
    ("eed_resource_a_impact_load", np.dtype(np.float32)),
    ("eed_resource_b_impact_load", np.dtype(np.float32)),
    ("respec0b_resource_class", np.dtype(np.float32)),
)
EED_CURSOR_ENV_ARRAYS = (
    ("respec0b_resource_active_y", np.dtype(np.int32)),
    ("respec0b_resource_active_x", np.dtype(np.int32)),
    ("respec0b_resource_active_class", np.dtype(np.int32)),
    ("respec0b_resource_handling_progress", np.dtype(np.float32)),
)
EED_BOOL_METADATA_FIELDS = (
    "respec0b_resource_specialization_enable",
    "respec0b_resource_equalized_sham_enable",
    "eed_resource_impact_enable",
    "eed_resource_impact_apply_enable",
    "eed_ecology_frozen_enable",
    "eed_phenotype_fixed_enable",
)
EED_FLOAT_METADATA_FIELDS = (
    "respec0b_resource_a_handling_threshold",
    "respec0b_resource_b_handling_threshold",
    "respec0b_resource_a_return",
    "respec0b_resource_b_return",
    "eed_resource_impact_per_success",
    "eed_resource_impact_decay_per_step",
    "eed_resource_impact_max_load",
)
JUNGLE_SUPPORTED_FLAGS = (
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
)
JUNGLE_BOOL_METADATA_FIELDS = (
    *JUNGLE_SUPPORTED_FLAGS,
    "jungle_birth_emission_at_child_cell_enable",
    "jungle_birth_emission_clamp_enable",
    "jungle_terrain_protection_apply_to_damage_enable",
    "jungle_recovery_regrowth_boost_enable",
)
JUNGLE_INT_METADATA_FIELDS = ("jungle_juvenile_age_threshold",)
JUNGLE_FLOAT_METADATA_FIELDS = (
    "jungle_predator_base_risk",
    "jungle_predator_scent_weight",
    "jungle_predator_waste_weight",
    "jungle_predator_density_weight",
    "jungle_predator_visibility_weight",
    "jungle_predator_cover_protection_weight",
    "jungle_predator_shelter_protection_weight",
    "jungle_predator_risk_min",
    "jungle_predator_risk_cap",
    "jungle_predator_health_damage",
    "jungle_birth_scent_emit",
    "jungle_birth_waste_emit",
    "jungle_juvenile_predator_damage_multiplier",
    "jungle_juvenile_vulnerability_cap",
    "jungle_cover_damage_reduction_weight",
    "jungle_shelter_damage_reduction_weight",
    "jungle_terrain_damage_multiplier_min",
    "jungle_recovery_activity_initial",
    "jungle_recovery_activity_min",
    "jungle_recovery_activity_max",
    "jungle_recovery_activity_decay_per_step",
    "jungle_recovery_activity_base_growth",
    "jungle_recovery_activity_cover_weight",
    "jungle_recovery_activity_fertility_weight",
    "jungle_recovery_activity_waste_penalty_weight",
    "jungle_recovery_activity_density_penalty_weight",
    "jungle_recovery_activity_scent_penalty_weight",
    "jungle_recovery_regrowth_boost_scale",
    "jungle_recovery_regrowth_boost_max",
)
PATCH_BOOL_METADATA_FIELDS = (
    "patch_world_enable",
    "patch_food_regrow_use_fertility",
    "patch_food_regrow_cap_by_fertility",
    "patch_shelter_from_cover_enable",
    "patch_visibility_from_cover_enable",
    "patch_movement_cost_apply_enable",
    "patch_density_include_self",
    "patch_density_normalize",
    "patch_world_observation_enable",
    "patch_world_observation_stacked_gather_enable",
    "patch_world_observation_numba_gather_enable",
    "patch_world_diagnostics_enable",
    "patch_world_lineage_descriptors_enable",
    "patch_world_waste_consequence_enable",
    "patch_waste_regrowth_penalty_enable",
    "patch_density_regrowth_penalty_enable",
    "patch_waste_fertility_effect_enable",
)
PATCH_INT_METADATA_FIELDS = (
    "patch_world_schema_version",
    "patch_density_radius",
    "patch_world_observation_schema_version",
    "patch_world_descriptor_schema_version",
)
PATCH_FLOAT_METADATA_FIELDS = (
    "patch_food_amount_max",
    "patch_food_amount_initial_food_cell_value",
    "patch_food_amount_initial_empty_cell_value",
    "patch_food_presence_threshold",
    "patch_food_gather_amount",
    "patch_food_regrow_per_step",
    "patch_fertility_default",
    "patch_fertility_min",
    "patch_fertility_max",
    "patch_cover_default",
    "patch_cover_min",
    "patch_cover_max",
    "patch_shelter_default",
    "patch_shelter_min",
    "patch_shelter_max",
    "patch_visibility_default",
    "patch_visibility_min",
    "patch_visibility_max",
    "patch_movement_cost_default",
    "patch_movement_cost_min",
    "patch_movement_cost_max",
    "patch_movement_cost_energy_scale",
    "patch_scent_initial",
    "patch_scent_max",
    "patch_scent_decay_per_step",
    "patch_scent_presence_emit",
    "patch_scent_move_emit",
    "patch_scent_gather_emit",
    "patch_scent_birth_emit",
    "patch_scent_death_emit",
    "patch_waste_initial",
    "patch_waste_max",
    "patch_waste_decay_per_step",
    "patch_waste_presence_emit",
    "patch_waste_consume_emit",
    "patch_waste_birth_emit",
    "patch_waste_death_emit",
    "patch_waste_density_emit_scale",
    "patch_waste_regrowth_penalty_scale",
    "patch_waste_regrowth_penalty_max",
    "patch_density_regrowth_penalty_scale",
    "patch_density_regrowth_penalty_max",
    "patch_regrowth_min_multiplier",
    "patch_waste_fertility_damage_rate",
    "patch_fertility_recovery_rate",
    "patch_fertility_recovery_target",
)
PATCH_UNSUPPORTED_ACTIVE_FLAGS = (
    "spatial_ecology_enable",
    "spatial_droughts_enable",
    "respec0b_resource_specialization_enable",
    "mosaic0a_depletion_recovery_enable",
    "predrisk_fixed1_enable",
    "corridor_cp0_enable",
    "corridor_hc1_homing_cue_enable",
    "habacclim_enable",
    "roz2a_patch_niche_enable",
    "jungle_predator_pressure_enable",
    "jungle_reproduction_emissions_enable",
    "jungle_juvenile_predator_vulnerability_enable",
    "jungle_terrain_protection_enable",
    "jungle_ecological_recovery_enable",
    "jungle_recovery_fertility_enable",
    "jungle_resource_spread_enable",
    "wdl_delayed_hazard_enable",
    "path1_drh0_consequence_enable",
    "prediction_demand_enable",
    "mem_max_delayed_choice_enable",
)
PATCH_UNSUPPORTED_NONZERO_EMISSIONS = (
    "patch_scent_birth_emit",
    "patch_scent_death_emit",
    "patch_waste_birth_emit",
    "patch_waste_death_emit",
    "patch_waste_density_emit_scale",
)


def native_extension_compile_source_manifest() -> dict[str, Any]:
    sources = []
    for path in NATIVE_EXTENSION_SOURCE_INVENTORY:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"path": str(path.resolve()), "sha256": digest})
    payload = json.dumps(sources, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "schema_version": "cxxbackbone10_compile_source_manifest_v1",
        "sources": sources,
        "compile_source_fingerprint_sha256": hashlib.sha256(payload).hexdigest(),
    }


@dataclass(frozen=True)
class ResidentStateOwnerExtensionStatus:
    schema_version: str = "cxxbackbone10_resident_owner_extension_status_v1"
    stable_module_name: str = RESIDENT_STATE_OWNER_STABLE_MODULE_NAME
    env_var: str = RESIDENT_STATE_OWNER_EXTENSION_ENV
    extension_path: str = ""
    manifest_path: str = ""
    extension_abi_version: str = ""
    extension_version: str = ""
    source_inventory_fingerprint_sha256: str = ""
    abi_compatible: bool = False
    output_sha256_verified: bool = False
    source_inventory_verified: bool = False
    available: bool = False
    runtime_jit: bool = False
    loaded: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stable_module_name": self.stable_module_name,
            "env_var": self.env_var,
            "extension_path": self.extension_path,
            "manifest_path": self.manifest_path,
            "extension_abi_version": self.extension_abi_version,
            "extension_version": self.extension_version,
            "source_inventory_fingerprint_sha256": (
                self.source_inventory_fingerprint_sha256
            ),
            "abi_compatible": self.abi_compatible,
            "output_sha256_verified": self.output_sha256_verified,
            "source_inventory_verified": self.source_inventory_verified,
            "available": self.available,
            "runtime_jit": self.runtime_jit,
            "loaded": self.loaded,
            "error": self.error,
        }


def _read_resident_owner_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"resident owner extension manifest is invalid: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RuntimeError("resident owner extension manifest must be an object")
    return dict(raw)


def resident_state_owner_extension_status(
    path: str | os.PathLike[str] | None = None,
) -> ResidentStateOwnerExtensionStatus:
    raw = str(path or os.environ.get(RESIDENT_STATE_OWNER_EXTENSION_ENV, "") or "")
    if not raw:
        return ResidentStateOwnerExtensionStatus(
            error=f"{RESIDENT_STATE_OWNER_EXTENSION_ENV} is not set"
        )
    extension = Path(raw)
    if not extension.exists():
        return ResidentStateOwnerExtensionStatus(
            extension_path=str(extension),
            error="configured resident owner extension path does not exist",
        )
    if not extension.is_file():
        return ResidentStateOwnerExtensionStatus(
            extension_path=str(extension),
            error="configured resident owner extension path is not a file",
        )
    extension = extension.resolve(strict=True)
    manifest_path = extension.with_suffix(extension.suffix + ".manifest.json")
    if not manifest_path.is_file():
        return ResidentStateOwnerExtensionStatus(
            extension_path=str(extension),
            manifest_path=str(manifest_path),
            error="configured resident owner extension manifest is missing",
        )
    try:
        manifest = _read_resident_owner_manifest(manifest_path)
    except RuntimeError as exc:
        return ResidentStateOwnerExtensionStatus(
            extension_path=str(extension),
            manifest_path=str(manifest_path),
            error=str(exc),
        )
    stable_module_name = str(
        manifest.get("stable_module_name", RESIDENT_STATE_OWNER_STABLE_MODULE_NAME)
        or ""
    )
    extension_abi_version = str(manifest.get("extension_abi_version", "") or "")
    extension_version = str(manifest.get("extension_version", "") or "")
    source_inventory_fingerprint = str(
        manifest.get("source_inventory_fingerprint_sha256", "") or ""
    )
    current_fingerprint = str(
        native_extension_compile_source_manifest()[
            "compile_source_fingerprint_sha256"
        ]
    )
    expected_sha256 = str(manifest.get("output_sha256", "") or "")
    actual_sha256 = hashlib.sha256(extension.read_bytes()).hexdigest()
    output_sha256_verified = bool(expected_sha256) and actual_sha256 == expected_sha256
    source_inventory_verified = (
        bool(source_inventory_fingerprint)
        and source_inventory_fingerprint == current_fingerprint
    )
    abi_compatible = (
        extension_abi_version == "eiel_native_runtime_abi_v1"
        and extension_version == "eiel_native_runtime_flat_v1"
    )
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stable_module_name) is None:
        error = "resident owner extension manifest has an invalid stable module name"
    elif not abi_compatible:
        error = (
            "resident owner extension ABI/version mismatch: "
            f"abi={extension_abi_version or '<missing>'} "
            f"version={extension_version or '<missing>'}"
        )
    elif not output_sha256_verified:
        error = "resident owner extension SHA-256 does not match its manifest"
    elif not source_inventory_verified:
        error = "resident owner extension source inventory does not match this checkout"
    else:
        error = ""
    return ResidentStateOwnerExtensionStatus(
        stable_module_name=stable_module_name,
        extension_path=str(extension),
        manifest_path=str(manifest_path),
        extension_abi_version=extension_abi_version,
        extension_version=extension_version,
        source_inventory_fingerprint_sha256=source_inventory_fingerprint,
        abi_compatible=abi_compatible,
        output_sha256_verified=output_sha256_verified,
        source_inventory_verified=source_inventory_verified,
        available=not error,
        loaded=_MODULE is not None,
        error=error,
    )


def _load_prebuilt_resident_state_owner_extension(
    path: str | os.PathLike[str] | None = None,
) -> Any:
    status = resident_state_owner_extension_status(path)
    if not status.available:
        raise RuntimeError(status.error)
    spec = importlib.util.spec_from_file_location(
        status.stable_module_name,
        status.extension_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not create module spec for {status.extension_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_abi = str(getattr(module, "EIEL_NATIVE_RUNTIME_ABI_VERSION", "") or "")
    module_version = str(
        getattr(module, "EIEL_NATIVE_RUNTIME_EXTENSION_VERSION", "") or ""
    )
    if (
        module_abi != "eiel_native_runtime_abi_v1"
        or module_version != "eiel_native_runtime_flat_v1"
    ):
        raise RuntimeError(
            "loaded resident owner extension ABI/version mismatch: "
            f"abi={module_abi or '<missing>'} "
            f"version={module_version or '<missing>'}"
        )
    return module


def _required_patch_config_value(env_cfg: Any, name: str) -> Any:
    if env_cfg is None or not hasattr(env_cfg, name):
        raise RuntimeError(f"Patch native import requires exact config field: {name}")
    return getattr(env_cfg, name)


def _required_patch_bool(env_cfg: Any, name: str) -> bool:
    value = _required_patch_config_value(env_cfg, name)
    if not isinstance(value, (bool, np.bool_)):
        raise RuntimeError(f"Patch native config field {name} must be boolean")
    return bool(value)


def _required_patch_int(env_cfg: Any, name: str) -> int:
    value = _required_patch_config_value(env_cfg, name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"Patch native config field {name} must be an integer")
    return int(value)


def _required_patch_float(env_cfg: Any, name: str) -> float:
    value = _required_patch_config_value(env_cfg, name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RuntimeError(f"Patch native config field {name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"Patch native config field {name} must be finite")
    return resolved


def _resolved_cpu_device(value: Any) -> str:
    try:
        return str(torch.device(value).type)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Patch native import has invalid device: {value!r}") from exc


def _patch_profile_metadata(
    runner: Any,
    env_cfg: Any,
    sim_cfg: Any,
    capability_mask: Mapping[str, bool],
) -> dict[str, Any]:
    from native_runtime.patch_profile import classify_patch_profile  # noqa: WPS433

    capabilities = {
        str(name): bool(value) for name, value in dict(capability_mask).items()
    }
    wdl_execution = capabilities.get("wdl", False)
    jungle_execution = capabilities.get("jungle", False)
    eed_execution = capabilities.get(EED_EXECUTION_CAPABILITY, False)
    cross_path_overlay = capabilities.get("cross_path_overlay", False)
    genome_v2 = capabilities.get("genome_v2", False)
    if cross_path_overlay and genome_v2:
        raise RuntimeError(
            "Genome V2 source import cannot claim the cross-path overlay"
        )
    classification_env_cfg = env_cfg
    classification_sim_cfg = sim_cfg
    if (
        wdl_execution
        or jungle_execution
        or eed_execution
        or cross_path_overlay
        or genome_v2
    ):
        classification_env_cfg = copy.copy(env_cfg)
        if wdl_execution:
            classification_env_cfg.wdl_delayed_hazard_enable = False
            classification_env_cfg.wdl_delayed_hazard_observation_enable = False
        if jungle_execution:
            for name in JUNGLE_SUPPORTED_FLAGS:
                setattr(classification_env_cfg, name, False)
        if eed_execution:
            classification_env_cfg.respec0b_resource_specialization_enable = False
            classification_env_cfg.eed_resource_impact_enable = False
            classification_env_cfg.eed_resource_impact_apply_enable = False
            classification_env_cfg.eed_ecology_frozen_enable = False
            classification_env_cfg.eed_phenotype_fixed_enable = False
        if cross_path_overlay or genome_v2:
            if getattr(env_cfg, "reproduce_requires_mate", None) is not True:
                raise RuntimeError(
                    "diploid source import requires mate-required "
                    "production configuration"
                )
            classification_env_cfg.reproduce_requires_mate = False
            classification_sim_cfg = copy.copy(sim_cfg)
            classification_sim_cfg.inheritance_profile = "legacy_midpoint_v1"
    classification = classify_patch_profile(
        classification_sim_cfg,
        classification_env_cfg,
        runner,
        world="patch",
        accepted_policy_families=("mlp", "gru", "mixed"),
    )
    profile_scope = "full_profile"
    if classification.supported and classification.profile is not None:
        profile = classification.profile
    elif classification.unsupported_reasons == (
        "unsupported_patch_profile_combination",
    ):
        observation = getattr(env_cfg, "patch_world_observation_enable", None)
        waste_consequence = getattr(env_cfg, "patch_world_waste_consequence_enable", None)
        movement_apply = getattr(env_cfg, "patch_movement_cost_apply_enable", None)
        if (
            type(observation) is bool
            and observation
            and type(waste_consequence) is bool
            and type(movement_apply) is bool
            and waste_consequence is not movement_apply
        ):
            profile = "P2"
            profile_scope = "focused_p2_subcell"
        else:
            reasons = ",".join(classification.unsupported_reasons)
            raise RuntimeError(f"Patch native import profile is unsupported: {reasons}")
    else:
        reasons = ",".join(classification.unsupported_reasons)
        raise RuntimeError(f"Patch native import profile is unsupported: {reasons}")

    metadata: dict[str, Any] = {}
    for name in PATCH_BOOL_METADATA_FIELDS:
        metadata[name] = _required_patch_bool(env_cfg, name)
    for name in PATCH_INT_METADATA_FIELDS:
        metadata[name] = _required_patch_int(env_cfg, name)
    for name in PATCH_FLOAT_METADATA_FIELDS:
        metadata[name] = _required_patch_float(env_cfg, name)

    if not metadata["patch_world_enable"]:
        raise RuntimeError("Patch native import requires patch_world_enable=true")
    if metadata["patch_world_schema_version"] != 1:
        raise RuntimeError("Patch native import supports only patch_world_schema_version=1")
    if metadata["patch_world_observation_schema_version"] != 1:
        raise RuntimeError("Patch native import supports only patch_world_observation_schema_version=1")
    if metadata["patch_world_descriptor_schema_version"] != 1:
        raise RuntimeError("Patch native import supports only patch_world_descriptor_schema_version=1")
    if metadata["patch_world_observation_stacked_gather_enable"]:
        raise RuntimeError("Patch native import does not support stacked Patch observations")
    if metadata["patch_world_observation_numba_gather_enable"]:
        raise RuntimeError("Patch native import does not support Numba Patch observations")

    p2_active = profile == "P2"
    if p2_active and not metadata["patch_world_observation_enable"]:
        raise RuntimeError("Patch P2 native import requires Patch observation schema v1")
    if metadata["patch_world_observation_enable"]:
        if not metadata["patch_world_diagnostics_enable"]:
            raise RuntimeError("Patch P1/P2 native import requires Patch diagnostics")
        if not metadata["patch_world_lineage_descriptors_enable"]:
            raise RuntimeError("Patch P1/P2 native import requires Patch lineage descriptors")

    for name in PATCH_UNSUPPORTED_ACTIVE_FLAGS:
        active = bool(getattr(env_cfg, name, False))
        metadata[name] = active
        if name == "wdl_delayed_hazard_enable" and wdl_execution:
            if not active:
                raise RuntimeError(
                    "WDL native execution requires wdl_delayed_hazard_enable=true"
                )
            continue
        if name in JUNGLE_SUPPORTED_FLAGS and jungle_execution:
            continue
        if name == "respec0b_resource_specialization_enable" and eed_execution:
            if not active:
                raise RuntimeError(
                    "EED native execution requires resource specialization"
                )
            continue
        if active:
            raise RuntimeError(f"Patch native import does not support active feature: {name}")
    for name in PATCH_UNSUPPORTED_NONZERO_EMISSIONS:
        if metadata[name] != 0.0:
            raise RuntimeError(f"Patch native import does not support nonzero emission: {name}")

    # Reference Patch movement emits no waste. This is a contract constant, not
    # a reconstructed configuration default.
    metadata["patch_waste_move_emit"] = 0.0

    configured_device = str(getattr(sim_cfg, "device", "") or "").strip().lower()
    resolved_device = _resolved_cpu_device(getattr(runner, "device", configured_device))
    resolved_bank_device = _resolved_cpu_device(
        getattr(runner, "_bank_device", getattr(runner, "bank_device", "cpu"))
    )
    if configured_device != "cpu" or resolved_device != "cpu" or resolved_bank_device != "cpu":
        raise RuntimeError("Patch native import supports only CPU simulation and tensor-bank devices")
    metadata["device"] = "cpu"
    metadata["bank_device"] = "cpu"

    fixed_policy_family = str(
        getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or ""
    ).strip().lower()
    if fixed_policy_family not in {"mlp", "gru", "mixed"}:
        raise RuntimeError(
            "Patch native import requires native_runtime_fixed_policy_family=mlp, gru, or mixed"
        )
    metadata["native_runtime_fixed_policy_family"] = fixed_policy_family
    allow_runtime_jit = bool(getattr(sim_cfg, "native_runtime_allow_runtime_jit_build", False))
    if allow_runtime_jit:
        raise RuntimeError("Patch native import forbids runtime JIT builds")
    metadata["native_runtime_allow_runtime_jit_build"] = False

    if capabilities.get("patch") is not True or capabilities.get("flat", False):
        raise RuntimeError("Patch native import requires patch=true and flat=false capabilities")
    for name in ("cuda",):
        if capabilities.get(name, False):
            raise RuntimeError(f"Patch native import does not support capability: {name}")
    if capabilities.get("gru", False) is not (fixed_policy_family in {"gru", "mixed"}):
        raise RuntimeError(
            "Patch native import GRU capability must match the policy-family envelope"
        )
    if wdl_execution:
        metadata.update(
            {
                "wdl_delayed_hazard_schema_version": _required_patch_int(
                    env_cfg, "wdl_delayed_hazard_schema_version"
                ),
                "wdl_delayed_hazard_cue_enable": _required_patch_bool(
                    env_cfg, "wdl_delayed_hazard_cue_enable"
                ),
                "wdl_delayed_hazard_observation_enable": _required_patch_bool(
                    env_cfg, "wdl_delayed_hazard_observation_enable"
                ),
                "wdl_delayed_hazard_damage_enable": _required_patch_bool(
                    env_cfg, "wdl_delayed_hazard_damage_enable"
                ),
                "wdl_delayed_hazard_deterministic_schedule_enable": (
                    _required_patch_bool(
                        env_cfg,
                        "wdl_delayed_hazard_deterministic_schedule_enable",
                    )
                ),
                "wdl_delayed_hazard_delay_steps": _required_patch_int(
                    env_cfg, "wdl_delayed_hazard_delay_steps"
                ),
                "wdl_delayed_hazard_cue_duration_steps": _required_patch_int(
                    env_cfg, "wdl_delayed_hazard_cue_duration_steps"
                ),
                "wdl_delayed_hazard_active_duration_steps": _required_patch_int(
                    env_cfg, "wdl_delayed_hazard_active_duration_steps"
                ),
                "wdl_delayed_hazard_event_period_steps": _required_patch_int(
                    env_cfg, "wdl_delayed_hazard_event_period_steps"
                ),
                "wdl_delayed_hazard_radius": _required_patch_int(
                    env_cfg, "wdl_delayed_hazard_radius"
                ),
                "wdl_delayed_hazard_intensity": _required_patch_float(
                    env_cfg, "wdl_delayed_hazard_intensity"
                ),
                "wdl_delayed_hazard_observation_dropout_rate": (
                    _required_patch_float(
                        env_cfg,
                        "wdl_delayed_hazard_observation_dropout_rate",
                    )
                ),
                "native_runtime_wdl_execution_enable": True,
            }
        )
    if jungle_execution:
        if not any(bool(getattr(env_cfg, name, False)) for name in JUNGLE_SUPPORTED_FLAGS):
            raise RuntimeError("Jungle native execution requires an active accepted Jungle mechanic")
        if bool(getattr(env_cfg, "jungle_recovery_fertility_enable", False)):
            raise RuntimeError("Jungle native execution does not support deferred fertility recovery")
        if bool(getattr(env_cfg, "jungle_resource_spread_enable", False)):
            raise RuntimeError("Jungle native execution does not support deferred resource spread")
        if bool(getattr(env_cfg, "jungle_juvenile_predator_vulnerability_enable", False)) and not bool(
            getattr(env_cfg, "jungle_predator_pressure_enable", False)
        ):
            raise RuntimeError("Jungle juvenile vulnerability requires predator pressure")
        if bool(getattr(env_cfg, "jungle_terrain_protection_enable", False)) and not bool(
            getattr(env_cfg, "jungle_predator_pressure_enable", False)
        ):
            raise RuntimeError("Jungle terrain protection requires predator pressure")
        metadata.update(
            {
                name: _required_patch_bool(env_cfg, name)
                for name in JUNGLE_BOOL_METADATA_FIELDS
            }
        )
        metadata.update(
            {
                name: _required_patch_int(env_cfg, name)
                for name in JUNGLE_INT_METADATA_FIELDS
            }
        )
        metadata.update(
            {
                name: _required_patch_float(env_cfg, name)
                for name in JUNGLE_FLOAT_METADATA_FIELDS
            }
        )
        metadata["native_runtime_jungle_execution_enable"] = True
    else:
        metadata["native_runtime_jungle_execution_enable"] = False
    if eed_execution:
        if not _required_patch_bool(env_cfg, "eed_resource_impact_enable"):
            raise RuntimeError(
                "EED native execution requires eed_resource_impact_enable=true"
            )
        if str(
            _required_patch_config_value(
                env_cfg, "respec0b_resource_class_mode"
            )
        ) not in {
            "vertical_split_ab",
            "all_a",
            "all_b",
            "checkerboard_ab",
            "none",
        }:
            raise RuntimeError(
                "EED native execution has an unsupported resource class mode"
            )
        metadata.update(
            {
                name: _required_patch_bool(env_cfg, name)
                for name in EED_BOOL_METADATA_FIELDS
            }
        )
        metadata.update(
            {
                name: _required_patch_float(env_cfg, name)
                for name in EED_FLOAT_METADATA_FIELDS
            }
        )
        metadata.update(
            {
                "respec0b_resource_class_mode": str(
                    _required_patch_config_value(
                        env_cfg, "respec0b_resource_class_mode"
                    )
                ),
                "eed_phenotype_fixed_axis_profile": str(
                    _required_patch_config_value(
                        env_cfg, "eed_phenotype_fixed_axis_profile"
                    )
                ),
                "native_runtime_eed_execution_enable": True,
                "native_runtime_eed_storage_schema_version": (
                    EED_STORAGE_SCHEMA_VERSION
                ),
            }
        )
        if (
            metadata["respec0b_resource_a_handling_threshold"] <= 0.0
            or metadata["respec0b_resource_b_handling_threshold"] <= 0.0
            or metadata["respec0b_resource_a_return"] < 0.0
            or metadata["respec0b_resource_b_return"] < 0.0
            or metadata["eed_resource_impact_per_success"] < 0.0
            or not 0.0
            <= metadata["eed_resource_impact_decay_per_step"]
            <= 1.0
            or metadata["eed_resource_impact_max_load"] < 0.0
        ):
            raise RuntimeError("EED native execution config is outside its domain")
        if metadata["eed_phenotype_fixed_axis_profile"] not in {
            "neutral",
            "a_high",
            "b_high",
            "symmetric_mix",
        }:
            raise RuntimeError("EED fixed phenotype profile is unsupported")
        if (
            metadata["patch_scent_gather_emit"] != 0.0
            or metadata["patch_waste_consume_emit"] != 0.0
        ):
            raise RuntimeError(
                "EED native execution requires zero gather emissions"
            )
        if float(getattr(env_cfg, "food_spawn_per_step", 0.0) or 0.0) != 0.0:
            raise RuntimeError(
                "EED native execution requires legacy food spawning disabled"
            )
    else:
        metadata["native_runtime_eed_execution_enable"] = False
    if capabilities.get(WDL_STORAGE_CAPABILITY, False) or wdl_execution:
        metadata.update(
            {
                "native_runtime_wdl_storage_enable": True,
                "native_runtime_wdl_storage_schema_version": (
                    WDL_STORAGE_SCHEMA_VERSION
                ),
                "native_runtime_wdl_execution_enable": wdl_execution,
            }
        )

    shock_start_prob = float(getattr(sim_cfg, "shock_start_prob", 0.0) or 0.0)
    if shock_start_prob != 0.0 or bool(getattr(sim_cfg, "shock_pulsed", False)):
        raise RuntimeError("Patch native import requires shock scheduling to be disabled")
    if bool(getattr(env_cfg, "tier4_juvenile_vulnerability_enable", False)):
        raise RuntimeError("Patch native import does not support Tier-4 juvenile vulnerability")
    if int(getattr(env_cfg, "tier4_juvenile_steps", 0) or 0) != 0:
        raise RuntimeError("Patch native import requires tier4_juvenile_steps=0")

    metadata.update(
        {
            "native_runtime_patch_profile": profile,
            "native_runtime_patch_profile_scope": profile_scope,
            "native_runtime_patch_contract_schema_version": "eiel_patch_native_import_contract_v1",
            "patch_array_import_phase": "post_reference_setup_once",
            "patch_visibility_authority": "derived_from_patch_cover",
            "patch_density_authority": "derived_pre_action_from_agent_grid",
            "patch_food_binary_projection": "patch_food_amount_strictly_greater_than_threshold",
            "patch_pending_emission_phase": "queued_until_next_tick",
        }
    )
    return metadata


def _patch_import_arrays(
    env: Any,
    env_cfg: Any,
    *,
    profile: str,
    include_wdl_storage: bool = False,
    include_eed_storage: bool = False,
    agent_id_capacity: int | None = None,
) -> dict[str, torch.Tensor]:
    if env is None:
        raise RuntimeError("Patch native import requires an initialized environment")
    agent_grid = getattr(env, "agent_grid", None)
    food_grid = getattr(env, "grid", None)
    if agent_grid is None or food_grid is None:
        raise RuntimeError("Patch native import requires initialized agent_grid and grid arrays")
    agent_grid_array = np.asarray(agent_grid)
    food_grid_array = np.asarray(food_grid)
    if agent_grid_array.ndim != 2 or tuple(food_grid_array.shape) != tuple(agent_grid_array.shape):
        raise RuntimeError("Patch native import requires matching two-dimensional world grids")
    shape = tuple(int(value) for value in agent_grid_array.shape)

    arrays: dict[str, np.ndarray] = {}
    required_arrays = PATCH_REQUIRED_ENV_ARRAYS + (
        PATCH_P2_REQUIRED_ENV_ARRAYS if profile == "P2" else ()
    )
    for name in required_arrays:
        if not hasattr(env, name):
            raise RuntimeError(f"Patch native import required array missing: {name}")
        value = getattr(env, name)
        if value is None:
            raise RuntimeError(f"Patch native import required array missing: {name}")
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float32):
            raise RuntimeError(f"Patch native import array {name} must be float32")
        if tuple(array.shape) != shape:
            raise RuntimeError(
                f"Patch native import array {name} shape mismatch: expected={shape} actual={tuple(array.shape)}"
            )
        if not bool(np.all(np.isfinite(array))):
            raise RuntimeError(f"Patch native import array {name} contains non-finite values")
        arrays[name] = np.ascontiguousarray(array, dtype=np.float32)

    if include_wdl_storage:
        present_wdl_arrays = {
            name
            for name, _dtype in WDL_STORAGE_ENV_ARRAYS
            if hasattr(env, name) and getattr(env, name) is not None
        }
        expected_wdl_arrays = {name for name, _dtype in WDL_STORAGE_ENV_ARRAYS}
        if present_wdl_arrays and present_wdl_arrays != expected_wdl_arrays:
            missing = sorted(expected_wdl_arrays - present_wdl_arrays)
            raise RuntimeError(
                f"WDL native storage arrays are partially initialized: missing={missing}"
            )
        for name, dtype in WDL_STORAGE_ENV_ARRAYS:
            if not present_wdl_arrays:
                fill_value = -1 if name == "patch_delayed_hazard_timer" else 0
                array = np.full(shape, fill_value, dtype=dtype)
            else:
                array = np.asarray(getattr(env, name))
            if array.dtype != dtype:
                raise RuntimeError(
                    f"WDL native storage array {name} must be {dtype.name}"
                )
            if tuple(array.shape) != shape:
                raise RuntimeError(
                    "WDL native storage array "
                    f"{name} shape mismatch: expected={shape} actual={tuple(array.shape)}"
                )
            if np.issubdtype(dtype, np.floating) and not bool(
                np.all(np.isfinite(array))
            ):
                raise RuntimeError(
                    f"WDL native storage array {name} contains non-finite values"
                )
            arrays[name] = np.ascontiguousarray(array, dtype=dtype)

    if include_eed_storage:
        for name, dtype in EED_GRID_ENV_ARRAYS:
            if not hasattr(env, name) or getattr(env, name) is None:
                raise RuntimeError(
                    f"EED native storage array missing: {name}"
                )
            array = np.asarray(getattr(env, name))
            if array.dtype != dtype:
                raise RuntimeError(
                    f"EED native storage array {name} must be {dtype.name}"
                )
            if tuple(array.shape) != shape:
                raise RuntimeError(
                    "EED native storage array "
                    f"{name} shape mismatch: expected={shape} "
                    f"actual={tuple(array.shape)}"
                )
            if not bool(np.all(np.isfinite(array))):
                raise RuntimeError(
                    f"EED native storage array {name} contains non-finite values"
                )
            arrays[name] = np.ascontiguousarray(array, dtype=dtype)
        environment_capacity = int(
            np.asarray(getattr(env, "agent_pos_y")).shape[0]
        )
        capacity = (
            environment_capacity
            if agent_id_capacity is None
            else int(agent_id_capacity)
        )
        if capacity < 1 or capacity > environment_capacity:
            raise RuntimeError(
                "EED native handling cursor capacity is outside the "
                "environment identity extent"
            )
        for name, dtype in EED_CURSOR_ENV_ARRAYS:
            if not hasattr(env, name) or getattr(env, name) is None:
                raise RuntimeError(
                    f"EED native handling cursor missing: {name}"
                )
            array = np.asarray(getattr(env, name))
            if array.dtype != dtype:
                raise RuntimeError(
                    f"EED native handling cursor {name} must be {dtype.name}"
                )
            if tuple(array.shape) != (capacity,):
                raise RuntimeError(
                    "EED native handling cursor "
                    f"{name} shape mismatch: expected={(capacity,)} "
                    f"actual={tuple(array.shape)}"
                )
            if np.issubdtype(dtype, np.floating) and not bool(
                np.all(np.isfinite(array))
            ):
                raise RuntimeError(
                    f"EED native handling cursor {name} contains non-finite values"
                )
            arrays[name] = np.ascontiguousarray(array, dtype=dtype)

    from env.gridworld_pkg.constants import EMPTY, FOOD  # noqa: WPS433
    from env.gridworld_pkg.patch_world import (  # noqa: WPS433
        compute_local_density_map,
        patch_food_presence_threshold,
        patch_visibility_from_cover,
    )

    threshold = float(patch_food_presence_threshold(env_cfg))
    configured_threshold = _required_patch_float(env_cfg, "patch_food_presence_threshold")
    if threshold != configured_threshold:
        raise RuntimeError("Patch native import rejects a food threshold requiring implicit normalization")
    expected_food_grid = np.where(
        arrays["patch_food_amount"] > np.float32(threshold),
        FOOD,
        EMPTY,
    ).astype(food_grid_array.dtype, copy=False)
    if not np.array_equal(food_grid_array, expected_food_grid):
        raise RuntimeError(
            "Patch native import food grid does not equal patch_food_amount > patch_food_presence_threshold"
        )

    arrays["patch_visibility"] = patch_visibility_from_cover(
        env_cfg,
        arrays["patch_cover"],
    )
    density_env = SimpleNamespace(
        agent_grid=np.ascontiguousarray(agent_grid_array, dtype=np.int32),
        cfg=env_cfg,
    )
    arrays["patch_density"] = compute_local_density_map(density_env)
    return {name: _tensor_from_numpy(value) for name, value in arrays.items()}


def _rebuild_patch_checkpoint_derived_arrays(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    surface: str,
) -> dict[str, Any]:
    agent_grid = tensors.get("agent_grid")
    patch_cover = tensors.get("patch_cover")
    if not hasattr(agent_grid, "detach") or not hasattr(patch_cover, "detach"):
        raise RuntimeError(
            f"{surface} Patch derived views require authoritative agent_grid and patch_cover"
        )
    if agent_grid.device.type != "cpu" or agent_grid.dtype != torch.int32:
        raise RuntimeError(f"{surface} Patch agent_grid must be contiguous CPU int32")
    if patch_cover.device.type != "cpu" or patch_cover.dtype != torch.float32:
        raise RuntimeError(f"{surface} Patch patch_cover must be contiguous CPU float32")
    if (
        int(agent_grid.dim()) != 2
        or tuple(patch_cover.shape) != tuple(agent_grid.shape)
        or not agent_grid.is_contiguous()
        or not patch_cover.is_contiguous()
    ):
        raise RuntimeError(f"{surface} Patch derived-view source shape/contiguity mismatch")
    if not bool(torch.all(torch.isfinite(patch_cover)).item()):
        raise RuntimeError(f"{surface} Patch patch_cover contains non-finite values")

    from env.gridworld_pkg.patch_world import (  # noqa: WPS433
        compute_local_density_map,
        patch_visibility_from_cover,
    )

    derived_cfg = SimpleNamespace(**dict(metadata))
    try:
        rebuilt_arrays = {
            "patch_visibility": patch_visibility_from_cover(
                derived_cfg,
                patch_cover.detach().numpy(),
            ),
            "patch_density": compute_local_density_map(
                SimpleNamespace(
                    agent_grid=agent_grid.detach().numpy(),
                    cfg=derived_cfg,
                )
            ),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{surface} Patch derived-view rebuild failed: {exc}") from exc

    rebuilt = dict(tensors)
    shape = tuple(int(value) for value in agent_grid.shape)
    for name, value in rebuilt_arrays.items():
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float32) or tuple(array.shape) != shape:
            raise RuntimeError(
                f"{surface} Patch rebuilt array {name} must be world-shaped float32"
            )
        if not bool(np.all(np.isfinite(array))):
            raise RuntimeError(f"{surface} Patch rebuilt array {name} contains non-finite values")
        rebuilt[name] = _tensor_from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    return rebuilt


def _prepare_patch_checkpoint_projection(
    checkpoint_projection: Mapping[str, Any],
    *,
    surface: str,
) -> dict[str, Any]:
    prepared = dict(checkpoint_projection)
    metadata = copy.deepcopy(dict(checkpoint_projection.get("metadata") or {}))
    if metadata.get("world") != "patch":
        raise RuntimeError(f"{surface} requires exact Patch world metadata")
    tensors = _rebuild_patch_checkpoint_derived_arrays(
        metadata,
        dict(checkpoint_projection.get("tensors") or {}),
        surface=surface,
    )
    _validate_patch_projection_contract(metadata, tensors, surface=surface)
    prepared["metadata"] = metadata
    prepared["tensors"] = tensors
    return prepared


def _validate_patch_projection_contract(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    surface: str,
    allow_cuda_state: bool = False,
) -> None:
    if type(allow_cuda_state) is not bool:
        raise RuntimeError(f"{surface} allow_cuda_state must be boolean")
    world = metadata.get("world")
    if type(world) is not str or world not in {"flat", "patch"}:
        raise RuntimeError(f"{surface} requires exact world metadata")
    if world != "patch":
        return
    wdl_execution = metadata.get("native_runtime_wdl_execution_enable", False)
    if type(wdl_execution) is not bool:
        raise RuntimeError(f"{surface} WDL execution marker must be boolean")
    jungle_execution = metadata.get("native_runtime_jungle_execution_enable", False)
    if type(jungle_execution) is not bool:
        raise RuntimeError(f"{surface} Jungle execution marker must be boolean")
    eed_execution = metadata.get("native_runtime_eed_execution_enable", False)
    if type(eed_execution) is not bool:
        raise RuntimeError(f"{surface} EED execution marker must be boolean")
    required_metadata = (
        *PATCH_BOOL_METADATA_FIELDS,
        *PATCH_INT_METADATA_FIELDS,
        *PATCH_FLOAT_METADATA_FIELDS,
        *PATCH_UNSUPPORTED_ACTIVE_FLAGS,
        "patch_waste_move_emit",
        "native_runtime_patch_profile",
        "native_runtime_patch_profile_scope",
        "native_runtime_patch_contract_schema_version",
        "patch_array_import_phase",
        "patch_visibility_authority",
        "patch_density_authority",
        "patch_food_binary_projection",
        "patch_pending_emission_phase",
        "bank_device",
        "native_runtime_allow_runtime_jit_build",
    )
    missing_metadata = sorted(name for name in required_metadata if name not in metadata)
    if missing_metadata:
        raise RuntimeError(f"{surface} Patch metadata missing required fields: {missing_metadata}")
    for name in (*PATCH_BOOL_METADATA_FIELDS, *PATCH_UNSUPPORTED_ACTIVE_FLAGS):
        if type(metadata[name]) is not bool:
            raise RuntimeError(f"{surface} Patch metadata field {name} must be boolean")
    if wdl_execution != bool(metadata["wdl_delayed_hazard_enable"]):
        raise RuntimeError(f"{surface} WDL execution marker contradicts feature state")
    for name in PATCH_INT_METADATA_FIELDS:
        if type(metadata[name]) is not int:
            raise RuntimeError(f"{surface} Patch metadata field {name} must be an integer")
    for name in (*PATCH_FLOAT_METADATA_FIELDS, "patch_waste_move_emit"):
        value = metadata[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"{surface} Patch metadata field {name} must be finite numeric")
    if metadata.get("patch_world_enable") is not True:
        raise RuntimeError(f"{surface} Patch projection requires patch_world_enable=true")
    if metadata.get("device") != "cpu" or metadata.get("bank_device") != "cpu":
        raise RuntimeError(f"{surface} Patch projection requires CPU devices")
    fixed_policy_family = metadata.get("native_runtime_fixed_policy_family")
    if fixed_policy_family not in {"mlp", "gru", "mixed"}:
        raise RuntimeError(f"{surface} Patch projection requires an MLP, GRU, or mixed policy envelope")
    if metadata.get("native_runtime_allow_runtime_jit_build") is not False:
        raise RuntimeError(f"{surface} Patch projection forbids runtime JIT")
    if metadata.get("native_runtime_patch_profile") not in {"P0", "P1", "P2"}:
        raise RuntimeError(f"{surface} Patch projection has unsupported profile")
    if metadata.get("patch_world_schema_version") != 1:
        raise RuntimeError(f"{surface} Patch projection has unsupported world schema")
    if metadata.get("patch_world_observation_schema_version") != 1:
        raise RuntimeError(f"{surface} Patch projection has unsupported observation schema")
    if metadata.get("patch_world_observation_stacked_gather_enable") is not False:
        raise RuntimeError(f"{surface} Patch projection has unsupported stacked observations")
    if metadata.get("patch_world_observation_numba_gather_enable") is not False:
        raise RuntimeError(f"{surface} Patch projection has unsupported Numba observations")
    if metadata["patch_density_radius"] < 0:
        raise RuntimeError(f"{surface} Patch density radius must be nonnegative")
    visibility_min = float(metadata["patch_visibility_min"])
    visibility_max = float(metadata["patch_visibility_max"])
    if visibility_min > visibility_max:
        raise RuntimeError(f"{surface} Patch visibility bounds are reversed")
    if (
        metadata["patch_visibility_from_cover_enable"] is False
        and not visibility_min
        <= float(metadata["patch_visibility_default"])
        <= visibility_max
    ):
        raise RuntimeError(f"{surface} Patch visibility default is outside its bounds")
    movement_apply = metadata["patch_movement_cost_apply_enable"]
    waste_consequence = metadata["patch_world_waste_consequence_enable"]
    profile_scope = metadata["native_runtime_patch_profile_scope"]
    partial_p2 = movement_apply is not waste_consequence
    if partial_p2:
        if (
            metadata["native_runtime_patch_profile"] != "P2"
            or profile_scope != "focused_p2_subcell"
        ):
            raise RuntimeError(f"{surface} Patch profile has an unmarked partial P2 combination")
        expected_profile = "P2"
    else:
        if profile_scope != "full_profile":
            raise RuntimeError(f"{surface} Patch full profile has a diagnostic-only scope")
        expected_profile = (
            "P2"
            if movement_apply and waste_consequence
            else ("P1" if metadata["patch_world_observation_enable"] else "P0")
        )
    p2_active = expected_profile == "P2"
    if metadata["native_runtime_patch_profile"] != expected_profile:
        raise RuntimeError(f"{surface} Patch profile metadata contradicts its feature flags")
    if p2_active and not metadata["patch_world_observation_enable"]:
        raise RuntimeError(f"{surface} Patch P2 projection requires observation schema v1")
    if metadata["patch_world_observation_enable"] and (
        not metadata["patch_world_diagnostics_enable"]
        or not metadata["patch_world_lineage_descriptors_enable"]
    ):
        raise RuntimeError(f"{surface} Patch P1/P2 diagnostics contract is incomplete")
    capabilities = metadata.get("capability_mask")
    if not isinstance(capabilities, Mapping):
        raise RuntimeError(f"{surface} Patch projection requires a capability mask")
    if capabilities.get("patch") is not True or bool(capabilities.get("flat", False)):
        raise RuntimeError(f"{surface} Patch projection capability mask is inconsistent")
    for name in ("cuda",):
        if bool(capabilities.get(name, False)):
            raise RuntimeError(f"{surface} Patch projection has unsupported capability: {name}")
    if bool(capabilities.get("jungle", False)) != jungle_execution:
        raise RuntimeError(f"{surface} Jungle capability contradicts execution state")
    if bool(capabilities.get("gru", False)) != (fixed_policy_family in {"gru", "mixed"}):
        raise RuntimeError(
            f"{surface} Patch GRU capability contradicts the policy-family envelope"
        )
    if bool(capabilities.get("wdl", False)) != wdl_execution:
        raise RuntimeError(f"{surface} WDL capability contradicts execution state")
    if bool(capabilities.get(EED_EXECUTION_CAPABILITY, False)) != eed_execution:
        raise RuntimeError(f"{surface} EED capability contradicts execution state")
    cross_path_overlay = bool(capabilities.get("cross_path_overlay", False))
    genome_v2 = bool(capabilities.get("genome_v2", False))
    if cross_path_overlay and genome_v2:
        raise RuntimeError(
            f"{surface} cannot combine Genome V2 and cross-path overlay capabilities"
        )
    if (
        cross_path_overlay or genome_v2
    ) != bool(metadata.get("reproduce_requires_mate", False)):
        raise RuntimeError(
            f"{surface} diploid capability contradicts "
            "mate-required source state"
        )
    if wdl_execution:
        wdl_bool_fields = (
            "wdl_delayed_hazard_cue_enable",
            "wdl_delayed_hazard_observation_enable",
            "wdl_delayed_hazard_damage_enable",
            "wdl_delayed_hazard_deterministic_schedule_enable",
        )
        wdl_int_fields = (
            "wdl_delayed_hazard_schema_version",
            "wdl_delayed_hazard_delay_steps",
            "wdl_delayed_hazard_cue_duration_steps",
            "wdl_delayed_hazard_active_duration_steps",
            "wdl_delayed_hazard_event_period_steps",
            "wdl_delayed_hazard_radius",
        )
        wdl_float_fields = (
            "wdl_delayed_hazard_intensity",
            "wdl_delayed_hazard_observation_dropout_rate",
        )
        missing_wdl = sorted(
            name
            for name in (*wdl_bool_fields, *wdl_int_fields, *wdl_float_fields)
            if name not in metadata
        )
        if missing_wdl:
            raise RuntimeError(
                f"{surface} WDL execution metadata missing fields: {missing_wdl}"
            )
        for name in wdl_bool_fields:
            if type(metadata[name]) is not bool:
                raise RuntimeError(f"{surface} WDL field {name} must be boolean")
        for name in wdl_int_fields:
            if type(metadata[name]) is not int:
                raise RuntimeError(f"{surface} WDL field {name} must be an integer")
        for name in wdl_float_fields:
            value = metadata[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(f"{surface} WDL field {name} must be finite numeric")
        if metadata["wdl_delayed_hazard_observation_enable"] and not metadata.get(
            "patch_world_observation_enable", False
        ):
            raise RuntimeError(
                f"{surface} WDL observations require Patch observations"
            )
    if jungle_execution:
        missing_jungle = sorted(
            name
            for name in (
                *JUNGLE_BOOL_METADATA_FIELDS,
                *JUNGLE_INT_METADATA_FIELDS,
                *JUNGLE_FLOAT_METADATA_FIELDS,
            )
            if name not in metadata
        )
        if missing_jungle:
            raise RuntimeError(
                f"{surface} Jungle execution metadata missing fields: {missing_jungle}"
            )
        for name in JUNGLE_BOOL_METADATA_FIELDS:
            if type(metadata[name]) is not bool:
                raise RuntimeError(f"{surface} Jungle field {name} must be boolean")
        for name in JUNGLE_INT_METADATA_FIELDS:
            if type(metadata[name]) is not int:
                raise RuntimeError(f"{surface} Jungle field {name} must be an integer")
        for name in JUNGLE_FLOAT_METADATA_FIELDS:
            value = metadata[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(f"{surface} Jungle field {name} must be finite numeric")
        if not any(bool(metadata[name]) for name in JUNGLE_SUPPORTED_FLAGS):
            raise RuntimeError(f"{surface} Jungle execution has no active accepted mechanic")
        if metadata["jungle_juvenile_predator_vulnerability_enable"] and not metadata[
            "jungle_predator_pressure_enable"
        ]:
            raise RuntimeError(f"{surface} Jungle juvenile vulnerability requires predator pressure")
        if metadata["jungle_terrain_protection_enable"] and not metadata[
            "jungle_predator_pressure_enable"
        ]:
            raise RuntimeError(f"{surface} Jungle terrain protection requires predator pressure")
    if eed_execution:
        missing_eed = sorted(
            name
            for name in (
                *EED_BOOL_METADATA_FIELDS,
                *EED_FLOAT_METADATA_FIELDS,
                "respec0b_resource_class_mode",
                "eed_phenotype_fixed_axis_profile",
                "native_runtime_eed_storage_schema_version",
            )
            if name not in metadata
        )
        if missing_eed:
            raise RuntimeError(
                f"{surface} EED execution metadata missing fields: {missing_eed}"
            )
        if metadata["native_runtime_eed_storage_schema_version"] != (
            EED_STORAGE_SCHEMA_VERSION
        ):
            raise RuntimeError(f"{surface} EED storage schema mismatch")
        for name in EED_BOOL_METADATA_FIELDS:
            if type(metadata[name]) is not bool:
                raise RuntimeError(f"{surface} EED field {name} must be boolean")
        for name in EED_FLOAT_METADATA_FIELDS:
            value = metadata[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(
                    f"{surface} EED field {name} must be finite numeric"
                )
        if (
            not metadata["respec0b_resource_specialization_enable"]
            or not metadata["eed_resource_impact_enable"]
        ):
            raise RuntimeError(
                f"{surface} EED execution requires both feature flags"
            )
    marker_contract = {
        "native_runtime_patch_contract_schema_version": "eiel_patch_native_import_contract_v1",
        "patch_array_import_phase": "post_reference_setup_once",
        "patch_visibility_authority": "derived_from_patch_cover",
        "patch_density_authority": "derived_pre_action_from_agent_grid",
        "patch_food_binary_projection": "patch_food_amount_strictly_greater_than_threshold",
        "patch_pending_emission_phase": "queued_until_next_tick",
    }
    for name, expected in marker_contract.items():
        if metadata.get(name) != expected:
            raise RuntimeError(f"{surface} Patch contract marker mismatch: {name}")
    for name in PATCH_UNSUPPORTED_ACTIVE_FLAGS:
        if name == "wdl_delayed_hazard_enable" and wdl_execution:
            continue
        if name in JUNGLE_SUPPORTED_FLAGS and jungle_execution:
            continue
        if name == "respec0b_resource_specialization_enable" and eed_execution:
            continue
        if bool(metadata.get(name, False)):
            raise RuntimeError(f"{surface} Patch projection has unsupported active feature: {name}")
    for name in (*PATCH_UNSUPPORTED_NONZERO_EMISSIONS, "patch_waste_move_emit"):
        if float(metadata.get(name, 0.0) or 0.0) != 0.0:
            raise RuntimeError(f"{surface} Patch projection has unsupported nonzero emission: {name}")

    food_grid = tensors.get("food_grid")
    agent_grid = tensors.get("agent_grid")
    if not hasattr(food_grid, "detach") or not hasattr(agent_grid, "detach"):
        raise RuntimeError(f"{surface} Patch projection requires food_grid and agent_grid")
    if food_grid.device.type != "cpu" or agent_grid.device.type != "cpu":
        raise RuntimeError(f"{surface} Patch projection grids must be CPU resident")
    if food_grid.dtype != torch.int32 or agent_grid.dtype != torch.int32:
        raise RuntimeError(f"{surface} Patch projection grids must be int32")
    if not food_grid.is_contiguous() or not agent_grid.is_contiguous():
        raise RuntimeError(f"{surface} Patch projection grids must be contiguous")
    if int(agent_grid.dim()) != 2 or tuple(food_grid.shape) != tuple(agent_grid.shape):
        raise RuntimeError(f"{surface} Patch projection world-grid shape mismatch")
    shape = tuple(int(value) for value in agent_grid.shape)
    patch_arrays: dict[str, torch.Tensor] = {}
    required_patch_arrays = PATCH_REQUIRED_ENV_ARRAYS + (
        PATCH_P2_REQUIRED_ENV_ARRAYS
        if metadata["native_runtime_patch_profile"] == "P2"
        else ()
    )
    for name in (*required_patch_arrays, *PATCH_DERIVED_ARRAYS):
        tensor = tensors.get(name)
        if not hasattr(tensor, "detach"):
            raise RuntimeError(f"{surface} Patch required array missing: {name}")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise RuntimeError(f"{surface} Patch array {name} must be contiguous CPU float32")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise RuntimeError(f"{surface} Patch array {name} shape/contiguity mismatch")
        if not bool(torch.all(torch.isfinite(tensor)).item()):
            raise RuntimeError(f"{surface} Patch array {name} contains non-finite values")
        patch_arrays[name] = tensor

    wdl_storage_enabled = metadata.get("native_runtime_wdl_storage_enable", False)
    if type(wdl_storage_enabled) is not bool:
        raise RuntimeError(f"{surface} WDL storage enable marker must be boolean")
    present_wdl_fields = {
        name for name, _dtype in WDL_STORAGE_ENV_ARRAYS if name in tensors
    }
    expected_wdl_fields = {name for name, _dtype in WDL_STORAGE_ENV_ARRAYS}
    if not wdl_storage_enabled:
        if present_wdl_fields:
            raise RuntimeError(
                f"{surface} WDL tensors require the explicit storage profile"
            )
    else:
        if metadata.get("native_runtime_wdl_storage_schema_version") != (
            WDL_STORAGE_SCHEMA_VERSION
        ):
            raise RuntimeError(f"{surface} WDL storage schema mismatch")
        if metadata.get("native_runtime_wdl_execution_enable") is not wdl_execution:
            raise RuntimeError(f"{surface} WDL storage execution marker mismatch")
        for name in ("wdl_base_seed", "wdl_next_absolute_step"):
            if type(metadata.get(name)) is not int:
                raise RuntimeError(
                    f"{surface} WDL schedule field {name} must be an exact integer"
                )
        if metadata["wdl_next_absolute_step"] < 0:
            raise RuntimeError(
                f"{surface} WDL next absolute step must be nonnegative"
            )
        if present_wdl_fields != expected_wdl_fields:
            missing = sorted(expected_wdl_fields - present_wdl_fields)
            raise RuntimeError(f"{surface} WDL storage arrays missing: {missing}")
        for name, numpy_dtype in WDL_STORAGE_ENV_ARRAYS:
            tensor = tensors[name]
            expected_dtype = torch.from_numpy(
                np.empty((), dtype=numpy_dtype)
            ).dtype
            if tensor.device.type != "cpu" or tensor.dtype != expected_dtype:
                raise RuntimeError(
                    f"{surface} WDL storage array {name} has wrong device or dtype"
                )
            if tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise RuntimeError(
                    f"{surface} WDL storage array {name} shape/contiguity mismatch"
                )
            if tensor.is_floating_point() and not bool(
                torch.all(torch.isfinite(tensor)).item()
            ):
                raise RuntimeError(
                    f"{surface} WDL storage array {name} contains non-finite values"
                )

    present_eed_fields = {
        name
        for name, _dtype in (*EED_GRID_ENV_ARRAYS, *EED_CURSOR_ENV_ARRAYS)
        if name in tensors
    }
    expected_eed_fields = {
        name
        for name, _dtype in (*EED_GRID_ENV_ARRAYS, *EED_CURSOR_ENV_ARRAYS)
    }
    if not eed_execution:
        if present_eed_fields:
            raise RuntimeError(
                f"{surface} EED tensors require the explicit execution profile"
            )
    else:
        if present_eed_fields != expected_eed_fields:
            missing = sorted(expected_eed_fields - present_eed_fields)
            raise RuntimeError(f"{surface} EED storage arrays missing: {missing}")
        agent_id_capacity = int(tensors["slot_to_agent_id"].numel())
        for name, numpy_dtype in EED_GRID_ENV_ARRAYS:
            tensor = tensors[name]
            expected_dtype = torch.from_numpy(
                np.empty((), dtype=numpy_dtype)
            ).dtype
            if tensor.device.type != "cpu" or tensor.dtype != expected_dtype:
                raise RuntimeError(
                    f"{surface} EED grid array {name} has wrong device or dtype"
                )
            if tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise RuntimeError(
                    f"{surface} EED grid array {name} shape/contiguity mismatch"
                )
            if not bool(torch.all(torch.isfinite(tensor)).item()):
                raise RuntimeError(
                    f"{surface} EED grid array {name} contains non-finite values"
                )
        for name, numpy_dtype in EED_CURSOR_ENV_ARRAYS:
            tensor = tensors[name]
            expected_dtype = torch.from_numpy(
                np.empty((), dtype=numpy_dtype)
            ).dtype
            if tensor.device.type != "cpu" or tensor.dtype != expected_dtype:
                raise RuntimeError(
                    f"{surface} EED cursor {name} has wrong device or dtype"
                )
            if (
                tuple(tensor.shape) != (agent_id_capacity,)
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(
                    f"{surface} EED cursor {name} shape/contiguity mismatch: "
                    f"expected={(agent_id_capacity,)} "
                    f"actual={tuple(tensor.shape)}"
                )
            if tensor.is_floating_point() and not bool(
                torch.all(torch.isfinite(tensor)).item()
            ):
                raise RuntimeError(
                    f"{surface} EED cursor {name} contains non-finite values"
                )

    threshold = np.float32(float(metadata["patch_food_presence_threshold"]))
    expected_food = torch.where(
        patch_arrays["patch_food_amount"] > threshold,
        torch.ones_like(food_grid),
        torch.zeros_like(food_grid),
    )
    if not torch.equal(food_grid, expected_food):
        raise RuntimeError(f"{surface} Patch binary food projection is inconsistent")

    from env.gridworld_pkg.patch_world import (  # noqa: WPS433
        compute_local_density_map,
        patch_visibility_from_cover,
    )

    derived_cfg = SimpleNamespace(**dict(metadata))
    expected_visibility = torch.from_numpy(
        patch_visibility_from_cover(
            derived_cfg,
            patch_arrays["patch_cover"].detach().numpy(),
        )
    )
    density_env = SimpleNamespace(
        agent_grid=agent_grid.detach().numpy().astype(np.int32, copy=False),
        cfg=derived_cfg,
    )
    expected_density = torch.from_numpy(compute_local_density_map(density_env))
    if not torch.equal(patch_arrays["patch_visibility"], expected_visibility):
        raise RuntimeError(f"{surface} Patch visibility is not the exact derived view")
    if not torch.equal(patch_arrays["patch_density"], expected_density):
        raise RuntimeError(f"{surface} Patch density is not the exact pre-action derived view")

    alive_mask = tensors.get("alive_mask")
    policy_use_gru = tensors.get("policy_use_gru")
    if hasattr(alive_mask, "detach") and hasattr(policy_use_gru, "detach"):
        live = alive_mask.detach().to(dtype=torch.bool)
        use_gru = policy_use_gru.detach().to(dtype=torch.bool)
        if fixed_policy_family != "mixed":
            live_wrong_family = live & (use_gru != (fixed_policy_family == "gru"))
            if bool(torch.any(live_wrong_family).item()):
                raise RuntimeError(
                    f"{surface} Patch projection contains live rows outside the fixed policy family"
                )


def load_resident_state_owner_extension(*, build_directory: str | Path | None = None, verbose: bool = False) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    prebuilt_path = os.environ.get(RESIDENT_STATE_OWNER_EXTENSION_ENV)
    if prebuilt_path:
        _MODULE = _load_prebuilt_resident_state_owner_extension(prebuilt_path)
        return _MODULE
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name="cxxbackbone10_resident_state_owner_ext",
        sources=[str(path) for path in NATIVE_EXTENSION_SOURCES],
        extra_include_paths=[str(NUMPY_INCLUDE)],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_ldflags=[str(NUMPY_RANDOM_LIB)] if NUMPY_RANDOM_LIB.exists() else [],
        with_cuda=False,
        verbose=bool(verbose),
    )
    return _MODULE


@dataclass(frozen=True)
class ImportProjectionV10:
    schema_version: str
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    pending_generation_free_agent_ids: tuple[int, ...] = ()
    pending_generation_free_slots: tuple[int, ...] = ()


def migrate_native_checkpoint_projection_v1_to_v2(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify legacy dense identity authority, then discard it."""

    schema_version = str(projection.get("schema_version", "") or "")
    if schema_version != CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1:
        raise RuntimeError(
            "native checkpoint migration requires projection v1"
        )
    tensors = dict(projection.get("tensors") or {})
    required = {
        "slot_to_agent_id",
        "alive_mask",
        "agent_ids_by_alive_order",
        "agent_id_to_slot",
        "agent_id_to_slot_agent_ids",
        "agent_id_to_slot_slots",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise RuntimeError(
            f"native checkpoint v1 migration is missing identity tensors: {missing}"
        )
    slot_to_agent_id = _checkpoint_cpu_tensor(
        tensors, "slot_to_agent_id"
    )
    alive_mask = _checkpoint_cpu_tensor(tensors, "alive_mask")
    dense = _checkpoint_cpu_tensor(tensors, "agent_id_to_slot")
    compact_ids = _checkpoint_cpu_tensor(
        tensors, "agent_id_to_slot_agent_ids"
    )
    compact_slots = _checkpoint_cpu_tensor(
        tensors, "agent_id_to_slot_slots"
    )
    alive_order = _checkpoint_cpu_tensor(
        tensors, "agent_ids_by_alive_order"
    )
    exact_layouts = (
        (slot_to_agent_id, torch.int64, "slot_to_agent_id"),
        (alive_mask, torch.bool, "alive_mask"),
        (dense, torch.int32, "agent_id_to_slot"),
        (compact_ids, torch.int64, "agent_id_to_slot_agent_ids"),
        (compact_slots, torch.int32, "agent_id_to_slot_slots"),
        (alive_order, torch.int64, "agent_ids_by_alive_order"),
    )
    for tensor, dtype, name in exact_layouts:
        if (
            tensor.dtype != dtype
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                f"native checkpoint v1 migration has invalid {name} layout"
            )
    if slot_to_agent_id.shape != alive_mask.shape:
        raise RuntimeError(
            "native checkpoint v1 migration slot authority dimensions disagree"
        )
    if (
        compact_ids.shape != compact_slots.shape
        or compact_ids.shape != alive_order.shape
    ):
        raise RuntimeError(
            "native checkpoint v1 migration compact identity dimensions disagree"
        )
    slot_ids = [int(value) for value in slot_to_agent_id.tolist()]
    alive_values = [bool(value) for value in alive_mask.tolist()]
    expected_pairs = sorted(
        (agent_id, slot)
        for slot, (agent_id, alive) in enumerate(
            zip(slot_ids, alive_values, strict=True)
        )
        if alive and agent_id >= 0
    )
    if any(alive != (agent_id >= 0) for agent_id, alive in zip(
        slot_ids, alive_values, strict=True
    )):
        raise RuntimeError(
            "native checkpoint v1 migration slot partition contradiction"
        )
    if len({agent_id for agent_id, _slot in expected_pairs}) != len(
        expected_pairs
    ):
        raise RuntimeError(
            "native checkpoint v1 migration has duplicate live agent IDs"
        )
    actual_pairs = list(
        zip(
            (int(value) for value in compact_ids.tolist()),
            (int(value) for value in compact_slots.tolist()),
            strict=True,
        )
    )
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            "native checkpoint v1 migration compact identity contradiction"
        )
    if [int(value) for value in alive_order.tolist()] != [
        agent_id for agent_id, _slot in expected_pairs
    ]:
        raise RuntimeError(
            "native checkpoint v1 migration alive-order contradiction"
        )
    dense_pairs: list[tuple[int, int]] = []
    for agent_id, raw_slot in enumerate(dense.tolist()):
        slot = int(raw_slot)
        if slot < -1:
            raise RuntimeError(
                "native checkpoint v1 migration dense identity has invalid negative slot"
            )
        if slot >= 0:
            dense_pairs.append((agent_id, slot))
    if dense_pairs != expected_pairs:
        raise RuntimeError(
            "native checkpoint v1 migration dense identity contradiction"
        )

    out = dict(projection)
    out["schema_version"] = CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2
    out["metadata"] = copy.deepcopy(dict(projection.get("metadata") or {}))
    out_tensors = dict(tensors)
    out_tensors.pop("agent_id_to_slot")
    out["tensors"] = out_tensors
    out["tensor_manifest"] = {
        name: _checkpoint_tensor_manifest(tensor)
        for name, tensor in sorted(out_tensors.items())
        if hasattr(tensor, "detach")
    }
    return out


def _compact_identity_v2_projection(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    schema_version = str(projection.get("schema_version", "") or "")
    if schema_version == CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1:
        return migrate_native_checkpoint_projection_v1_to_v2(projection)
    if schema_version != CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2:
        raise RuntimeError(
            "native checkpoint restore requires projection v1 or v2"
        )
    tensors = dict(projection.get("tensors") or {})
    if "agent_id_to_slot" in tensors:
        raise RuntimeError(
            "native checkpoint projection v2 rejects dense agent_id_to_slot"
        )
    return dict(projection)


class ResidentRuntimeState:
    def __init__(self, native_owner: Any, import_projection: ImportProjectionV10):
        self._native_owner = native_owner
        self.import_projection = import_projection

    @classmethod
    def from_runner(
        cls,
        runner: Any,
        capability_mask: Mapping[str, bool],
        *,
        generation: int = 0,
        tick: int = 0,
        build_directory: str | Path | None = None,
        owner_device: str | torch.device | None = None,
    ) -> "ResidentRuntimeState":
        projection = build_import_projection(
            runner,
            capability_mask,
            generation=int(generation),
            tick=int(tick),
        )
        _validate_passive_modular_storage_projection(
            projection.metadata,
            projection.tensors,
            surface="native owner construction",
        )
        if projection.metadata.get("world") == "patch" and owner_device is not None:
            if _resolved_cpu_device(owner_device) != "cpu":
                raise RuntimeError("Patch native owner supports only CPU-resident tensors")
        if owner_device is not None:
            projection = _projection_to_device(projection, owner_device)
        mod = load_resident_state_owner_extension(build_directory=build_directory)
        native = mod.ResidentRuntimeStateV10(projection.tensors, copy.deepcopy(projection.metadata))
        if projection.pending_generation_free_slots:
            native.restore_checkpoint_projection(
                {
                    "schema_version": CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2,
                    "metadata": copy.deepcopy(projection.metadata),
                    "tensors": projection.tensors,
                    "pending_generation_free_agent_ids": list(projection.pending_generation_free_agent_ids),
                    "pending_generation_free_slots": list(projection.pending_generation_free_slots),
                }
            )
        return cls(native, projection)

    @classmethod
    def from_checkpoint_projection(
        cls,
        checkpoint_projection: Mapping[str, Any],
        *,
        build_directory: str | Path | None = None,
        owner_device: str | torch.device | None = None,
        native_owner_module: Any | None = None,
    ) -> "ResidentRuntimeState":
        checkpoint_projection = _compact_identity_v2_projection(
            checkpoint_projection
        )
        metadata = copy.deepcopy(dict(checkpoint_projection.get("metadata") or {}))
        tensors = dict(checkpoint_projection.get("tensors") or {})
        if metadata.get("world") == "patch":
            checkpoint_projection = _prepare_patch_checkpoint_projection(
                checkpoint_projection,
                surface="native owner checkpoint restore",
            )
            metadata = copy.deepcopy(dict(checkpoint_projection["metadata"]))
            tensors = dict(checkpoint_projection["tensors"])
        if metadata.get("world") == "patch" and owner_device is not None:
            if _resolved_cpu_device(owner_device) != "cpu":
                raise RuntimeError("Patch native checkpoint restore supports only CPU-resident tensors")
        _validate_patch_projection_contract(
            metadata,
            tensors,
            surface="native owner checkpoint restore",
        )
        _validate_passive_modular_storage_projection(
            metadata,
            tensors,
            surface="native owner checkpoint restore",
        )
        mod = (
            native_owner_module
            if native_owner_module is not None
            else load_resident_state_owner_extension(build_directory=build_directory)
        )
        if owner_device is not None:
            tensors = _tensor_dict_to_device(tensors, owner_device)
            checkpoint_projection = dict(checkpoint_projection)
            checkpoint_projection["tensors"] = tensors
        _validate_policy_energy_observation_checkpoint_contract(
            metadata,
            tensors,
            surface="native owner checkpoint restore",
        )
        _validate_post_contract_checkpoint_source_projection(
            checkpoint_projection,
            metadata,
            surface="native owner checkpoint restore",
        )
        _validate_patch_checkpoint_source_projection(
            checkpoint_projection,
            metadata,
            surface="native owner checkpoint restore",
        )
        native = mod.ResidentRuntimeStateV10(tensors, copy.deepcopy(metadata))
        restore_projection = dict(checkpoint_projection)
        restore_projection["metadata"] = copy.deepcopy(metadata)
        native.restore_checkpoint_projection(restore_projection)
        projection = ImportProjectionV10(
            schema_version=SCHEMA_VERSION,
            metadata=metadata,
            tensors=tensors,
            pending_generation_free_agent_ids=tuple(
                int(value) for value in checkpoint_projection.get("pending_generation_free_agent_ids", ())
            ),
            pending_generation_free_slots=tuple(
                int(value) for value in checkpoint_projection.get("pending_generation_free_slots", ())
            ),
        )
        return cls(native, projection)

    def export_validation_projection(self) -> dict[str, Any]:
        projection = dict(self._native_owner.export_validation_projection())
        projection["metadata"] = copy.deepcopy(dict(projection.get("metadata") or {}))
        return projection

    def export_checkpoint_projection(self) -> dict[str, Any]:
        projection = _with_checkpoint_contract_metadata(
            self._native_owner.export_checkpoint_projection()
        )
        _validate_passive_modular_storage_projection(
            dict(projection.get("metadata") or {}),
            dict(projection.get("tensors") or {}),
            surface="native owner checkpoint export",
        )
        return projection

    def export_report_projection(self) -> dict[str, Any]:
        return dict(self._native_owner.export_report_projection())

    def restore_checkpoint_projection(self, checkpoint_projection: Mapping[str, Any]) -> None:
        checkpoint_projection = _compact_identity_v2_projection(
            checkpoint_projection
        )
        metadata = copy.deepcopy(dict(checkpoint_projection.get("metadata") or {}))
        if metadata.get("world") == "patch":
            restore_projection = _prepare_patch_checkpoint_projection(
                checkpoint_projection,
                surface="native owner checkpoint reimport",
            )
            metadata = dict(restore_projection["metadata"])
            tensors = dict(restore_projection["tensors"])
            _validate_policy_energy_observation_checkpoint_contract(
                metadata,
                tensors,
                surface="native owner checkpoint reimport",
            )
            _validate_patch_checkpoint_source_projection(
                restore_projection,
                metadata,
                surface="native owner checkpoint reimport",
            )
        else:
            restore_projection = dict(checkpoint_projection)
            restore_projection["metadata"] = metadata
            tensors = dict(checkpoint_projection.get("tensors") or {})
        _validate_patch_projection_contract(
            metadata,
            tensors,
            surface="native owner checkpoint reimport",
        )
        _validate_passive_modular_storage_projection(
            metadata,
            tensors,
            surface="native owner checkpoint reimport",
        )
        self._native_owner.restore_checkpoint_projection(restore_projection)

    def import_checkpoint_runtime_prefix(
        self,
        checkpoint_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        fields = (
            "restore_count",
            "generation_calls",
            "runtime_event_cursor",
            "last_applied_actions",
            "last_moved",
            "last_gathered",
            "last_policy_refill_calls",
            "last_policy_refill_rows",
            "pending_generation_free_agent_ids",
            "pending_generation_free_slots",
            "last_world",
        )
        source = dict(checkpoint_projection)
        self._native_owner.import_checkpoint_runtime_prefix(source)
        restored = dict(self._native_owner.export_checkpoint_projection())
        mismatches = {
            name: {"source": source.get(name), "restored": restored.get(name)}
            for name in fields
            if restored.get(name) != source.get(name)
        }
        if mismatches:
            raise RuntimeError(
                "native checkpoint runtime prefix verification failed: "
                f"{mismatches}"
            )
        return {
            "schema_version": "cxxbackbone10_checkpoint_runtime_prefix_import_v1",
            "status": "imported_and_verified",
            "fields": {name: copy.deepcopy(restored[name]) for name in fields},
        }

    def owner_manifest(self) -> dict[str, Any]:
        return self._native_owner.owner_manifest()

    def group_rows(self) -> dict[str, torch.Tensor]:
        return dict(self._native_owner.group_rows())

    def group_rows_device(self) -> dict[str, torch.Tensor]:
        return dict(self._native_owner.group_rows_device())

    def build_flat_observation_frame_cpu(self) -> torch.Tensor:
        return self._native_owner.build_flat_observation_frame_cpu()

    def build_flat_observation_frame_device(self) -> torch.Tensor:
        return self._native_owner.build_flat_observation_frame_device()

    def build_patch_observation_frame_cpu(self) -> torch.Tensor:
        return self._native_owner.build_patch_observation_frame_cpu()

    def refill_flat_policy_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.refill_flat_policy_cpu())

    def service_flat_action_cache_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.service_flat_action_cache_cpu())

    def refill_flat_policy_device(self) -> dict[str, Any]:
        return dict(self._native_owner.refill_flat_policy_device())

    def refill_patch_policy_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.refill_patch_policy_cpu())

    def advance_wdl_tick_cpu(
        self,
        abs_step: int,
        *,
        cancel_requested: bool = False,
    ) -> dict[str, Any]:
        return dict(
            self._native_owner.advance_wdl_tick_cpu(
                int(abs_step),
                bool(cancel_requested),
            )
        )

    def execute_generation_flat_cpu(self, generation_index: int, steps_per_generation: int) -> dict[str, Any]:
        return dict(self._native_owner.execute_generation_flat_cpu(int(generation_index), int(steps_per_generation)))

    def execute_generation_flat_scripted_cpu(self, scripted_actions_by_tick_slot: torch.Tensor) -> dict[str, Any]:
        return dict(self._native_owner.execute_generation_flat_scripted_cpu(scripted_actions_by_tick_slot.contiguous()))

    def execute_generation_patch_cpu(self, generation_index: int, steps_per_generation: int) -> dict[str, Any]:
        return dict(self._native_owner.execute_generation_patch_cpu(int(generation_index), int(steps_per_generation)))

    def execute_generation_patch_with_population_birth_commit_cpu(
        self,
        generation_index: int,
        steps_per_generation: int,
        max_births: int = 1,
    ) -> dict[str, Any]:
        execute = getattr(
            self._native_owner,
            "execute_generation_patch_with_population_birth_commit_cpu",
            None,
        )
        if not callable(execute):
            raise RuntimeError(
                "loaded native extension does not expose the controlled Patch Architecture B entrypoint"
            )
        return dict(
            execute(
                int(generation_index),
                int(steps_per_generation),
                int(max_births),
            )
        )

    def execute_tick_span_patch_cpu(
        self,
        span_ticks: int,
        max_births: int = 1,
    ) -> dict[str, Any]:
        """Advance the opt-in Patch owner by a cursor-addressed tick span.

        This is intentionally distinct from the legacy generation entrypoint:
        the native implementation owns the current checkpoint tick, does not
        schedule a generation boundary, and materializes no legacy behavior
        rows.  Provider promotion and WDL/B48 coverage remain separate gates.
        """

        execute = getattr(self._native_owner, "execute_tick_span_patch_cpu", None)
        if not callable(execute):
            raise RuntimeError(
                "loaded native extension does not expose the tick-native Patch span entrypoint"
            )
        return dict(execute(int(span_ticks), int(max_births)))

    def execute_generation_patch_scripted_cpu(self, scripted_actions_by_tick_slot: torch.Tensor) -> dict[str, Any]:
        return dict(self._native_owner.execute_generation_patch_scripted_cpu(scripted_actions_by_tick_slot.contiguous()))

    def execute_lifecycle_mutation_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.execute_lifecycle_mutation_cpu())

    def execute_wdl_jungle_context_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.execute_wdl_jungle_context_cpu())

    def execute_population_attempt_scheduler_cpu(self, decrement_cooldown: bool = True) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_attempt_scheduler_cpu(bool(decrement_cooldown)))

    def execute_population_mate_precheck_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_mate_precheck_cpu())

    def execute_population_asexual_spawn_space_precheck_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_asexual_spawn_space_precheck_cpu())

    def execute_population_asexual_energy_precheck_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_asexual_energy_precheck_cpu())

    def execute_population_birth_preflight_cpu(self, decrement_cooldown: bool = True) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_birth_preflight_cpu(bool(decrement_cooldown)))

    def execute_generation_flat_with_population_preflight_cpu(
        self,
        generation_index: int,
        steps_per_generation: int,
    ) -> dict[str, Any]:
        return dict(
            self._native_owner.execute_generation_flat_with_population_preflight_cpu(
                int(generation_index),
                int(steps_per_generation),
            )
        )

    def execute_population_asexual_resident_birth_commit_cpu(self, max_births: int = 1) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_asexual_resident_birth_commit_cpu(int(max_births)))

    def execute_population_death_cull_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_death_cull_cpu())

    def execute_population_generation_turnover_cpu(self) -> dict[str, Any]:
        return dict(self._native_owner.execute_population_generation_turnover_cpu())

    def execute_generation_flat_with_population_birth_commit_cpu(
        self,
        generation_index: int,
        steps_per_generation: int,
        max_births: int = 1,
    ) -> dict[str, Any]:
        return dict(
            self._native_owner.execute_generation_flat_with_population_birth_commit_cpu(
                int(generation_index),
                int(steps_per_generation),
                int(max_births),
            )
        )


def _with_checkpoint_contract_metadata(
    projection: Mapping[str, Any],
    *,
    allow_cuda_state: bool = False,
) -> dict[str, Any]:
    if type(allow_cuda_state) is not bool:
        raise RuntimeError("native owner checkpoint export allow_cuda_state must be boolean")
    out = dict(projection)
    metadata = copy.deepcopy(dict(out.get("metadata") or {}))
    tensors = dict(out.get("tensors") or {})
    raw_world = metadata.get("world")
    if type(raw_world) is not str or raw_world not in {"flat", "patch"}:
        raise RuntimeError("native owner checkpoint export requires exact world metadata")
    world = raw_world
    generation = int(metadata.get("generation", 0) or 0)
    tick = int(metadata.get("tick", 0) or 0)
    metadata["agent_grid_phase_contract_schema_version"] = "eiel_agent_grid_phase_contract_v1"
    metadata["agent_grid_world_mode"] = world
    metadata["agent_grid_runner_phase"] = "generation_boundary" if tick == 0 else "partial_generation"
    metadata["agent_grid_authority_epoch"] = f"generation:{generation}:tick:{tick}"
    metadata["agent_grid_synchronization_state"] = "stored_grid_authoritative"
    metadata["agent_grid_preserve_exact_stored"] = True
    metadata["agent_grid_reconstructible_without_phase_context"] = False
    _validate_patch_projection_contract(
        metadata,
        tensors,
        surface="native owner checkpoint export",
        allow_cuda_state=allow_cuda_state,
    )
    _attach_policy_energy_observation_contract_metadata(metadata, tensors)
    _refresh_event_tape_metadata(metadata, tensors)
    out["tensors"] = tensors
    out["metadata"] = metadata
    out["tensor_manifest"] = {
        name: _checkpoint_tensor_manifest(tensor)
        for name, tensor in sorted(tensors.items())
        if hasattr(tensor, "detach")
    }
    schema_version = str(out.get("schema_version", "") or "")
    if schema_version in {SCHEMA_VERSION, CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1}:
        out["schema_version"] = CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1
        schema_version = CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1
        if "agent_id_to_slot" in tensors:
            out = migrate_native_checkpoint_projection_v1_to_v2(out)
        else:
            out["schema_version"] = CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2
    elif schema_version == CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2:
        if "agent_id_to_slot" in tensors:
            raise RuntimeError(
                "native owner checkpoint projection v2 rejects dense agent_id_to_slot"
            )
    else:
        raise RuntimeError(
            "native owner checkpoint export requires projection v1 or v2 schema"
        )
    return out


_TORCH_MANIFEST_DTYPE_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _checkpoint_tensor_manifest(tensor: torch.Tensor) -> dict[str, Any]:
    dtype_name = _TORCH_MANIFEST_DTYPE_NAMES.get(tensor.dtype)
    if dtype_name is None:
        raise RuntimeError(f"unsupported checkpoint tensor manifest dtype: {tensor.dtype}")
    numel = int(tensor.numel())
    element_size = int(tensor.element_size())
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": dtype_name,
        "device": str(tensor.device),
        "numel": numel,
        "element_size": element_size,
        "bytes": numel * element_size,
        "is_cuda": bool(tensor.is_cuda),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _attach_policy_energy_observation_contract_metadata(
    metadata: dict[str, Any],
    tensors: Mapping[str, Any],
) -> None:
    policy_energy_observation = tensors.get("policy_energy_observation")
    alive_mask = tensors.get("alive_mask")
    metadata[POLICY_ENERGY_OBSERVATION_CONTRACT_METADATA] = POLICY_ENERGY_OBSERVATION_CONTRACT_SCHEMA
    metadata["policy_energy_observation_authority"] = "float64_python_visible_energy_at_flat_policy_boundary"
    metadata["policy_energy_observation_mechanics_energy_relation"] = (
        "mechanics_energy_is_float32_storage_not_authoritative_for_policy_observation_when_shadow_is_finite"
    )
    metadata["policy_energy_observation_live_finite_required"] = True
    if not hasattr(policy_energy_observation, "detach") or not hasattr(alive_mask, "detach"):
        return
    shadow = policy_energy_observation.detach().cpu()
    alive = alive_mask.detach().cpu().to(dtype=torch.bool)
    metadata["policy_energy_observation_dtype"] = str(shadow.dtype).removeprefix("torch.")
    metadata["policy_energy_observation_shape"] = [int(dim) for dim in shadow.shape]
    metadata["policy_energy_observation_tensor_sha256"] = _checkpoint_tensor_sha256(shadow)
    if int(shadow.ndim) == 1 and int(alive.ndim) == 1 and int(shadow.shape[0]) == int(alive.shape[0]):
        live = int(alive.sum().item())
        finite_live = int(torch.isfinite(shadow[alive]).sum().item()) if live else 0
        metadata["policy_energy_observation_live_count"] = live
        metadata["policy_energy_observation_finite_live_count"] = finite_live


def _checkpoint_tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    payload = {
        "dtype": str(cpu.dtype),
        "shape": [int(dim) for dim in cpu.shape],
    }
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(b"\0")
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _validate_post_contract_checkpoint_source_projection(
    checkpoint_projection: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    surface: str,
) -> None:
    schema = str(metadata.get(POLICY_ENERGY_OBSERVATION_CONTRACT_METADATA, "") or "")
    if not schema:
        return
    if str(metadata.get("world", "") or "").strip().lower() != "flat":
        return
    from native_runtime.checkpoint_digests import validate_checkpoint_source_projection  # noqa: WPS433

    try:
        validate_checkpoint_source_projection(checkpoint_projection)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{surface} source projection validation failed: {exc}") from exc


def _validate_patch_checkpoint_source_projection(
    checkpoint_projection: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    surface: str,
) -> None:
    if metadata.get("world") != "patch":
        return
    from native_runtime.checkpoint_digests import (  # noqa: WPS433
        checkpoint_source_projection_authority_manifest,
    )

    try:
        checkpoint_source_projection_authority_manifest(checkpoint_projection)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{surface} source projection validation failed: {exc}") from exc


def _validate_policy_energy_observation_checkpoint_contract(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    surface: str,
) -> None:
    schema = str(metadata.get(POLICY_ENERGY_OBSERVATION_CONTRACT_METADATA, "") or "")
    if not schema:
        return
    if schema != POLICY_ENERGY_OBSERVATION_CONTRACT_SCHEMA:
        raise RuntimeError(
            f"{surface} has unsupported {POLICY_ENERGY_OBSERVATION_CONTRACT_METADATA}: {schema}"
        )
    shadow = tensors.get("policy_energy_observation")
    alive = tensors.get("alive_mask")
    slot_to_agent_id = tensors.get("slot_to_agent_id")
    if not hasattr(shadow, "detach"):
        raise RuntimeError(f"{surface} is missing required policy_energy_observation tensor")
    if not hasattr(alive, "detach"):
        raise RuntimeError(f"{surface} is missing required alive_mask tensor for policy_energy_observation")
    if not hasattr(slot_to_agent_id, "detach"):
        raise RuntimeError(f"{surface} is missing required slot_to_agent_id tensor for policy_energy_observation")
    if shadow.dtype != torch.float64:
        raise RuntimeError(f"{surface} policy_energy_observation must be float64")
    if int(shadow.ndim) != 1:
        raise RuntimeError(f"{surface} policy_energy_observation must be rank-1")
    alive_bool = alive.detach().cpu().to(dtype=torch.bool)
    assigned = slot_to_agent_id.detach().cpu().to(dtype=torch.int64).ge(0)
    shadow_cpu = shadow.detach().cpu()
    if int(alive_bool.ndim) != 1 or int(alive_bool.shape[0]) != int(shadow_cpu.shape[0]):
        raise RuntimeError(
            f"{surface} policy_energy_observation shape must match alive_mask capacity"
        )
    if int(assigned.ndim) != 1 or int(assigned.shape[0]) != int(shadow_cpu.shape[0]):
        raise RuntimeError(
            f"{surface} policy_energy_observation shape must match slot_to_agent_id capacity"
        )
    if not bool(torch.equal(alive_bool, assigned)):
        raise RuntimeError(f"{surface} alive_mask and slot_to_agent_id partition contradiction")
    live_count = int(alive_bool.sum().item())
    if live_count and not bool(torch.isfinite(shadow_cpu[alive_bool]).all().item()):
        raise RuntimeError(f"{surface} policy_energy_observation has non-finite live-slot values")
    dead_or_free = ~alive_bool
    if bool(torch.isfinite(shadow_cpu[dead_or_free]).any().item()):
        raise RuntimeError(f"{surface} policy_energy_observation has finite dead/free-slot values")
    expected_digest = str(metadata.get("policy_energy_observation_tensor_sha256", "") or "")
    if expected_digest:
        actual_digest = _checkpoint_tensor_sha256(shadow_cpu)
        if actual_digest != expected_digest:
            raise RuntimeError(f"{surface} policy_energy_observation field digest mismatch")


def record_reference_committed_death_context(
    runner: Any,
    *,
    generation: int,
    tick: int,
    agent_id: int,
    reason_code: int,
    slot: int = -1,
    value: float | None = None,
) -> dict[str, Any]:
    generation_i = int(generation)
    tick_i = int(tick)
    agent_id_i = int(agent_id)
    reason_code_i = int(reason_code)
    value_f = float(reason_code_i if value is None else value)
    if generation_i < 0 or tick_i < 0:
        raise ValueError("reference committed death requires non-negative generation/tick")
    if agent_id_i < 0:
        raise ValueError("reference committed death requires actor agent identity")
    if reason_code_i not in (1, 2) or value_f != float(reason_code_i):
        raise ValueError("reference committed death requires matching terminal reason_code/value 1 or 2")

    contexts = getattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, None)
    if contexts is None:
        contexts = []
        setattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, contexts)
    if not isinstance(contexts, list):
        raise TypeError(f"{REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR} must be a list")
    row = {
        "event_type": "death_committed",
        "generation": generation_i,
        "tick": tick_i,
        "order": len(contexts),
        "agent_id": agent_id_i,
        "actor_agent_id": agent_id_i,
        "target_agent_id": -1,
        "slot": int(slot),
        "reason_code": reason_code_i,
        "value": value_f,
        "source": "reference_env_remove_commit_oracle",
    }
    from native_runtime.lifecycle_events import typed_lifecycle_events_from_reference_contexts  # noqa: WPS433

    typed_lifecycle_events_from_reference_contexts([row])
    contexts.append(row)
    return dict(row)


def _required_reference_lifecycle_coordinate(runner: Any, *, coordinate: str) -> int:
    names = (
        ("_eiel_fdp_cur_gen", "_eiel_dbg_gen", "gen", "_current_gen")
        if coordinate == "generation"
        else ("_eiel_fdp_cur_tick", "_eiel_dbg_tick", "_tick", "tick")
    )
    for name in names:
        if not hasattr(runner, name):
            continue
        try:
            value = int(getattr(runner, name))
        except Exception:
            continue
        if value >= 0:
            return value
    raise RuntimeError(f"reference committed death oracle has no authoritative {coordinate}")


@contextmanager
def reference_committed_death_oracle(runner: Any) -> Iterator[list[dict[str, Any]]]:
    env = getattr(runner, "env", None)
    remove_agent = getattr(env, "remove_agent", None)
    if env is None or not callable(remove_agent):
        raise TypeError("reference committed death oracle requires runner.env.remove_agent")
    env_alive = getattr(env, "agent_alive", None)
    if not hasattr(env_alive, "shape"):
        raise TypeError("reference committed death oracle requires runner.env.agent_alive")
    agents = getattr(getattr(runner, "pop", None), "agents", None)
    if not isinstance(agents, Mapping):
        raise TypeError("reference committed death oracle requires runner.pop.agents")
    if bool(getattr(runner, "_cxxbackbone10_reference_death_oracle_active", False)):
        raise RuntimeError("reference committed death oracle is already active")

    contexts = getattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, None)
    if contexts is None:
        contexts = []
        setattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, contexts)
    if not isinstance(contexts, list):
        raise TypeError(f"{REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR} must be a list")
    env_dict = getattr(env, "__dict__", {})
    had_instance_remove_agent = "remove_agent" in env_dict
    instance_remove_agent = env_dict.get("remove_agent")

    def remove_agent_with_death_oracle(agent_id: int) -> Any:
        aid = int(agent_id)
        was_present = False
        try:
            was_present = 0 <= aid < int(env_alive.shape[0]) and bool(env_alive[aid])
        except Exception:
            was_present = False
        agent = agents.get(aid)
        committed_death = was_present and agent is not None and not bool(getattr(agent, "alive", True))
        reason_code = 0
        slot = -1
        generation = -1
        tick = -1
        if committed_death:
            health = float(getattr(agent, "health"))
            energy = float(getattr(agent, "energy"))
            reason_code = 2 if health <= 0.0 else 1 if energy <= 0.0 else 0
            slot = int(getattr(agent, "slot", -1))
            if reason_code == 0:
                raise RuntimeError(
                    "reference committed death oracle observed a dead removal without terminal energy/health"
                )
            generation = _required_reference_lifecycle_coordinate(runner, coordinate="generation")
            tick = _required_reference_lifecycle_coordinate(runner, coordinate="tick")
        result = remove_agent(aid)
        if committed_death:
            record_reference_committed_death_context(
                runner,
                generation=generation,
                tick=tick,
                agent_id=aid,
                slot=slot,
                reason_code=reason_code,
                value=float(reason_code),
            )
        return result

    setattr(runner, "_cxxbackbone10_reference_death_oracle_active", True)
    env.remove_agent = remove_agent_with_death_oracle
    try:
        yield contexts
    finally:
        if had_instance_remove_agent:
            env.remove_agent = instance_remove_agent
        else:
            delattr(env, "remove_agent")
        setattr(runner, "_cxxbackbone10_reference_death_oracle_active", False)


def _reference_committed_death_oracle_requested(runner: Any) -> bool:
    sim_cfg = getattr(runner, "sim_cfg", None)
    if sim_cfg is None or not bool(getattr(sim_cfg, "native_runtime_cxx26_scope_lock_enable", False)):
        return False
    native_route_enabled = any(
        bool(getattr(sim_cfg, name, False))
        for name in (
            "native_safe_default_enable",
            "native_runtime_enable",
            "native_runtime_diagnostic_enable",
        )
    )
    return not native_route_enabled


def _reference_reproduction_context_source(runner: Any) -> list[Any] | tuple[Any, ...]:
    contexts = getattr(runner, "_lineage_reproduction_funnel_event_contexts_gen", None)
    if contexts is None:
        return ()
    if not isinstance(contexts, (list, tuple)):
        raise TypeError("reference reproduction lifecycle contexts must be a list or tuple")
    return contexts


def _reference_reproduction_contexts(
    runner: Any,
    *,
    start: int = 0,
) -> list[Mapping[str, Any]]:
    source = _reference_reproduction_context_source(runner)
    cursor = int(start)
    if cursor < 0 or cursor > len(source):
        raise RuntimeError("reference reproduction context cursor is outside the generation context list")
    return [
        context
        for context in source[cursor:]
        if int(context.get("birth_attempted", 0) or 0) > 0
        or int(context.get("birth_success", 0) or 0) > 0
    ]


def _empty_prefixed_typed_lifecycle_events() -> dict[str, torch.Tensor]:
    from native_runtime.lifecycle_events import empty_typed_lifecycle_event_tensors  # noqa: WPS433

    return {
        f"typed_lifecycle_event_{name}": tensor
        for name, tensor in empty_typed_lifecycle_event_tensors().items()
    }


def _reference_contexts_to_prefixed_typed_events(
    contexts: list[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    if not contexts:
        return _empty_prefixed_typed_lifecycle_events()
    from native_runtime.lifecycle_events import (  # noqa: WPS433
        typed_lifecycle_events_from_legacy_tape,
        typed_lifecycle_events_from_reference_contexts,
    )

    events = typed_lifecycle_events_from_reference_contexts(contexts)
    prefixed = {f"typed_lifecycle_event_{name}": tensor for name, tensor in events.items()}
    legacy = _legacy_event_tape_from_prefixed_typed_events(prefixed)
    normalized = typed_lifecycle_events_from_legacy_tape(
        {
            "metadata": {"event_tape_active_count": len(contexts)},
            "tensors": legacy,
        },
        active_count=len(contexts),
    )
    return {f"typed_lifecycle_event_{name}": tensor for name, tensor in normalized.items()}


def _pending_reference_lifecycle_contexts(
    runner: Any,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Any] | tuple[Any, ...]]:
    reproduction_source = _reference_reproduction_context_source(runner)
    reproduction_marker = getattr(runner, REFERENCE_LIFECYCLE_REPRODUCTION_CURSOR_ATTR, None)
    reproduction_start = 0
    if (
        isinstance(reproduction_marker, tuple)
        and len(reproduction_marker) == 2
        and reproduction_marker[0] is reproduction_source
    ):
        reproduction_start = int(reproduction_marker[1])
        if reproduction_start > len(reproduction_source):
            reproduction_start = 0
    reproduction_contexts = _reference_reproduction_contexts(
        runner,
        start=reproduction_start,
    )
    death_contexts = list(getattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, []) or [])
    death_start = int(getattr(runner, REFERENCE_LIFECYCLE_DEATH_CURSOR_ATTR, 0) or 0)
    if death_start < 0 or death_start > len(death_contexts):
        raise RuntimeError("reference committed death context cursor is outside the recorded context list")
    return reproduction_contexts, death_contexts[death_start:], reproduction_source


def _append_reference_lifecycle_segment(runner: Any) -> dict[str, int]:
    reproduction_contexts, new_death_contexts, reproduction_source = _pending_reference_lifecycle_contexts(runner)
    segment = _reference_contexts_to_prefixed_typed_events(
        [*reproduction_contexts, *new_death_contexts]
    )
    accumulated = getattr(runner, REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR, None)
    if accumulated is None:
        merged = segment
    elif isinstance(accumulated, Mapping):
        merged = _merge_prefixed_typed_lifecycle_events(accumulated, segment)
    else:
        raise TypeError(f"{REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR} must be a typed event mapping")
    setattr(
        runner,
        REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR,
        {name: tensor.detach().cpu().clone() for name, tensor in merged.items()},
    )
    death_contexts = list(getattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, []) or [])
    setattr(runner, REFERENCE_LIFECYCLE_DEATH_CURSOR_ATTR, len(death_contexts))
    setattr(
        runner,
        REFERENCE_LIFECYCLE_REPRODUCTION_CURSOR_ATTR,
        (reproduction_source, len(reproduction_source)),
    )
    return {
        "reproduction_context_count": len(reproduction_contexts),
        "death_context_count": len(new_death_contexts),
        "accumulated_event_count": int(merged["typed_lifecycle_event_event_type"].shape[0]),
    }


@contextmanager
def _reference_reproduction_context_capture(runner: Any) -> Iterator[None]:
    sim_cfg = getattr(runner, "sim_cfg", None)
    if sim_cfg is None:
        raise TypeError("reference lifecycle context capture requires runner.sim_cfg")
    names = (
        "lineage_reproduction_funnel_reporting_enable",
        "lineage_reproduction_funnel_write_events",
    )
    previous = {
        name: (hasattr(sim_cfg, name), getattr(sim_cfg, name, None))
        for name in names
    }
    for name in names:
        setattr(sim_cfg, name, True)
    try:
        yield
    finally:
        for name, (was_present, value) in previous.items():
            if was_present:
                setattr(sim_cfg, name, value)
            else:
                delattr(sim_cfg, name)


def install_reference_committed_death_oracle_hook() -> bool:
    from simulation.runner_pkg.runner import SimulationRunner  # noqa: WPS433

    installed = False
    current_ticks = getattr(SimulationRunner, "_run_ticks")
    if not bool(getattr(current_ticks, "_eiel_reference_committed_death_oracle_hook", False)):

        @wraps(current_ticks)
        def run_ticks_with_reference_death_oracle(runner: Any, *args: Any, **kwargs: Any) -> Any:
            delegate = getattr(
                run_ticks_with_reference_death_oracle,
                "_eiel_reference_run_delegate",
            )
            if not _reference_committed_death_oracle_requested(runner) or bool(
                getattr(runner, "_cxxbackbone10_reference_death_oracle_active", False)
            ):
                return delegate(runner, *args, **kwargs)
            with _reference_reproduction_context_capture(runner):
                with reference_committed_death_oracle(runner):
                    result = delegate(runner, *args, **kwargs)
                    _append_reference_lifecycle_segment(runner)
                    return result

        setattr(run_ticks_with_reference_death_oracle, "_eiel_reference_run_delegate", current_ticks)
        setattr(run_ticks_with_reference_death_oracle, "_eiel_reference_committed_death_oracle_hook", True)
        SimulationRunner._run_ticks = run_ticks_with_reference_death_oracle
        installed = True

    current_start = getattr(SimulationRunner, "_start_generation")
    if not bool(getattr(current_start, "_eiel_reference_lifecycle_pre_reset_hook", False)):

        @wraps(current_start)
        def start_generation_with_reference_lifecycle_finalize(runner: Any, *args: Any, **kwargs: Any) -> Any:
            delegate = getattr(
                start_generation_with_reference_lifecycle_finalize,
                "_eiel_reference_start_generation_delegate",
            )
            if _reference_committed_death_oracle_requested(runner):
                _append_reference_lifecycle_segment(runner)
            return delegate(runner, *args, **kwargs)

        setattr(
            start_generation_with_reference_lifecycle_finalize,
            "_eiel_reference_start_generation_delegate",
            current_start,
        )
        setattr(start_generation_with_reference_lifecycle_finalize, "_eiel_reference_lifecycle_pre_reset_hook", True)
        SimulationRunner._start_generation = start_generation_with_reference_lifecycle_finalize
        installed = True
    return installed


def _typed_lifecycle_event_tensors_for_runner(runner: Any) -> dict[str, torch.Tensor]:
    accumulated = getattr(runner, REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR, None)
    if accumulated is not None:
        if not isinstance(accumulated, Mapping):
            raise TypeError(f"{REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR} must be a typed event mapping")
        reproduction_contexts, death_contexts, _source = _pending_reference_lifecycle_contexts(runner)
        pending = _reference_contexts_to_prefixed_typed_events(
            [*reproduction_contexts, *death_contexts]
        )
        merged = _merge_prefixed_typed_lifecycle_events(accumulated, pending)
        if int(merged["typed_lifecycle_event_event_type"].shape[0]) <= 0:
            return {}
        return merged
    reproduction_contexts = _reference_reproduction_contexts(runner)
    death_contexts = list(getattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, []) or [])
    contexts = [*reproduction_contexts, *death_contexts]
    if not contexts:
        return {}
    return _reference_contexts_to_prefixed_typed_events(contexts)


def _legacy_event_tape_from_prefixed_typed_events(
    typed_events: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    event_type = typed_events["typed_lifecycle_event_event_type"].detach().cpu().to(dtype=torch.int32).contiguous()
    actor = typed_events["typed_lifecycle_event_actor_agent_id"].detach().cpu().to(dtype=torch.int64).contiguous()
    target = typed_events["typed_lifecycle_event_target_agent_id"].detach().cpu().to(dtype=torch.int64).contiguous()
    value = typed_events["typed_lifecycle_event_value"].detach().cpu().to(dtype=torch.float32).contiguous()
    return {
        "event_tape_type": event_type,
        "event_tape_generation": typed_events["typed_lifecycle_event_generation"]
        .detach()
        .cpu()
        .to(dtype=torch.int64)
        .contiguous(),
        "event_tape_tick": typed_events["typed_lifecycle_event_tick"]
        .detach()
        .cpu()
        .to(dtype=torch.int64)
        .contiguous(),
        "event_tape_actor_id": actor,
        "event_tape_target_id": target,
        "event_tape_value": value,
    }


def _merge_prefixed_typed_lifecycle_events(
    boundary_events: Mapping[str, Any],
    local_events: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    from native_runtime.lifecycle_events import (  # noqa: WPS433
        FIELD_DTYPES,
        concatenate_typed_lifecycle_events,
    )

    prefix = "typed_lifecycle_event_"
    names = tuple(f"{prefix}{name}" for name in FIELD_DTYPES)
    boundary_present = [name for name in names if hasattr(boundary_events.get(name), "detach")]
    local_present = [name for name in names if hasattr(local_events.get(name), "detach")]
    if boundary_present and len(boundary_present) != len(names):
        raise RuntimeError("checkpoint boundary has an incomplete typed lifecycle event tensor set")
    if len(local_present) != len(names):
        raise RuntimeError("local reference context has an incomplete typed lifecycle event tensor set")
    if not boundary_present:
        return {
            name: local_events[name].detach().cpu().contiguous()
            for name in names
        }
    boundary = {
        name: boundary_events[f"{prefix}{name}"]
        for name in FIELD_DTYPES
    }
    local = {
        name: local_events[f"{prefix}{name}"]
        for name in FIELD_DTYPES
    }
    merged = concatenate_typed_lifecycle_events(boundary, local)
    return {f"{prefix}{name}": tensor for name, tensor in merged.items()}


def _cat_checkpoint_tensor_prefix(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.cat(
        (
            first.detach().cpu().to(dtype=dtype).contiguous(),
            second.detach().cpu().to(dtype=dtype).contiguous(),
        ),
        dim=0,
    ).contiguous()


def _refresh_event_tape_metadata(metadata: dict[str, Any], tensors: dict[str, torch.Tensor]) -> None:
    legacy_dtypes = {
        "event_tape_type": torch.int32,
        "event_tape_generation": torch.int64,
        "event_tape_tick": torch.int64,
        "event_tape_actor_id": torch.int64,
        "event_tape_target_id": torch.int64,
        "event_tape_value": torch.float32,
    }
    legacy_present = [name for name in EVENT_TAPE_TENSOR_NAMES if hasattr(tensors.get(name), "shape")]
    if legacy_present and len(legacy_present) != len(EVENT_TAPE_TENSOR_NAMES):
        raise RuntimeError("incomplete legacy lifecycle event tensor set")
    legacy_lengths = {int(tensors[name].shape[0]) for name in legacy_present}
    if len(legacy_lengths) > 1:
        raise RuntimeError("legacy lifecycle event tensors have inconsistent lengths")
    for name in legacy_present:
        if tensors[name].dtype != legacy_dtypes[name]:
            raise RuntimeError(f"legacy lifecycle event tensor has wrong dtype: {name}")
    active_events = next(iter(legacy_lengths), 0)
    external_history = metadata.get("event_history_storage_mode") == (
        "eiel_resident_event_sink_v1"
    )
    sequence_base = int(metadata.get("typed_lifecycle_event_sequence_base", 0) or 0)
    frontier = int(
        metadata.get("event_history_frontier", sequence_base + active_events)
        or 0
    )
    if external_history and (
        sequence_base < 0
        or frontier != sequence_base + active_events
    ):
        raise RuntimeError("external event-history frontier is not exact")
    metadata["event_tape_contract_schema_version"] = "eiel_event_tape_active_inactive_contract_v1"
    metadata["event_tape_active_count"] = int(active_events)
    metadata["event_tape_cursor"] = int(active_events)
    metadata["event_tape_inactive_tail_count"] = 0
    if external_history:
        metadata["typed_lifecycle_event_sequence_base"] = sequence_base
        metadata["event_history_frontier"] = frontier
    metadata["event_tape_active_semantics"] = (
        "typed_lifecycle_birth_death_mutation_rows" if active_events else "no_active_typed_rows"
    )
    metadata["event_tape_public_projection"] = "not_public_output_canonicalized"
    typed_present = [
        name for name in TYPED_LIFECYCLE_EVENT_TENSOR_NAMES if hasattr(tensors.get(name), "shape")
    ]
    if typed_present and len(typed_present) != len(TYPED_LIFECYCLE_EVENT_TENSOR_NAMES):
        raise RuntimeError("incomplete typed lifecycle event tensor set")
    if typed_present:
        from native_runtime.lifecycle_events import FIELD_DTYPES  # noqa: WPS433

        typed_lengths = {int(tensors[name].shape[0]) for name in typed_present}
        if len(typed_lengths) != 1:
            raise RuntimeError("typed lifecycle event tensors have inconsistent lengths")
        typed_rows = next(iter(typed_lengths))
        if typed_rows != active_events:
            raise RuntimeError("typed and legacy lifecycle event tensor lengths disagree")
        for field_name, dtype in FIELD_DTYPES.items():
            name = f"typed_lifecycle_event_{field_name}"
            if tensors[name].dtype != dtype:
                raise RuntimeError(f"typed lifecycle event tensor has wrong dtype: {name}")
        sequence = tensors["typed_lifecycle_event_sequence"].detach().cpu().to(dtype=torch.int64)
        expected_sequence = torch.arange(
            sequence_base,
            sequence_base + typed_rows,
            dtype=torch.int64,
        )
        if not torch.equal(sequence, expected_sequence):
            raise RuntimeError(
                "typed lifecycle event sequence is not contiguous from its frontier"
            )
        compatibility_pairs = (
            ("typed_lifecycle_event_event_type", "event_tape_type"),
            ("typed_lifecycle_event_generation", "event_tape_generation"),
            ("typed_lifecycle_event_tick", "event_tape_tick"),
            ("typed_lifecycle_event_actor_agent_id", "event_tape_actor_id"),
            ("typed_lifecycle_event_target_agent_id", "event_tape_target_id"),
            ("typed_lifecycle_event_value", "event_tape_value"),
        )
        for typed_name, legacy_name in compatibility_pairs:
            if not torch.equal(tensors[typed_name].detach().cpu(), tensors[legacy_name].detach().cpu()):
                raise RuntimeError(f"typed and legacy lifecycle event tensors disagree: {typed_name}")
        metadata["typed_lifecycle_event_schema_version"] = "eiel_typed_lifecycle_event_v1"
        metadata["typed_lifecycle_event_active_count"] = typed_rows
        metadata["typed_lifecycle_event_next_sequence"] = (
            frontier if external_history else typed_rows
        )
        if external_history:
            metadata["typed_lifecycle_event_sequence_base"] = sequence_base
            metadata["event_history_frontier"] = frontier
        metadata.setdefault("typed_lifecycle_event_source", "native_resident_complete_tensors")
    elif active_events:
        from native_runtime.lifecycle_events import typed_lifecycle_events_from_legacy_tape  # noqa: WPS433

        typed_events = typed_lifecycle_events_from_legacy_tape(
            {"metadata": metadata, "tensors": tensors},
            active_count=int(active_events),
        )
        for name, tensor in typed_events.items():
            tensors[f"typed_lifecycle_event_{name}"] = tensor
        metadata["typed_lifecycle_event_schema_version"] = "eiel_typed_lifecycle_event_v1"
        metadata["typed_lifecycle_event_active_count"] = int(active_events)
        metadata["typed_lifecycle_event_next_sequence"] = (
            frontier if external_history else int(active_events)
        )
        metadata["typed_lifecycle_event_source"] = "native_legacy_event_tape_adapter"
    else:
        from native_runtime.lifecycle_events import empty_typed_lifecycle_event_tensors  # noqa: WPS433

        for name, tensor in empty_typed_lifecycle_event_tensors().items():
            tensors[f"typed_lifecycle_event_{name}"] = tensor
        metadata["typed_lifecycle_event_schema_version"] = "eiel_typed_lifecycle_event_v1"
        metadata["typed_lifecycle_event_active_count"] = 0
        metadata["typed_lifecycle_event_next_sequence"] = (
            frontier if external_history else 0
        )
        metadata["typed_lifecycle_event_source"] = "native_resident_complete_tensors"


def _action_cache_actions_tensor_for_runner(runner: Any, state: Any, width: int) -> torch.Tensor:
    capacity = int(getattr(state, "capacity", 0) or 0)
    cache_width = max(1, int(width))
    out = np.full((capacity, cache_width), -1, dtype=np.int32)
    cache = getattr(runner, "_policy_action_cache", getattr(runner, "policy_action_cache", None))
    if not isinstance(cache, dict):
        return _tensor_from_numpy(out)
    slot_to_agent_id = np.asarray(getattr(state, "slot_to_agent_id", np.full((capacity,), -1)), dtype=np.int64)
    alive = np.asarray(
        getattr(state, "alive", getattr(state, "alive_mask", np.zeros((capacity,), dtype=np.bool_))),
        dtype=np.bool_,
    )
    for slot, aid in enumerate(slot_to_agent_id.tolist()):
        if int(aid) < 0 or int(slot) >= int(alive.shape[0]) or not bool(alive[int(slot)]):
            continue
        value = resolve_policy_action_cache_entry(
            cache,
            slot=int(slot),
            agent_id=int(aid),
        )
        if isinstance(value, Mapping):
            raw_actions = value.get("actions", value.get("action_cache_actions", ()))
        elif isinstance(value, (tuple, list)):
            actions_index = 2 if is_slot_keyed_policy_cache_entry(value) else 1
            raw_actions = value[actions_index] if len(value) > actions_index else ()
        else:
            raw_actions = ()
        try:
            actions = np.asarray(raw_actions, dtype=np.int32).reshape(-1)
        except Exception:
            continue
        if int(actions.shape[0]) <= 0:
            continue
        rows = min(cache_width, int(actions.shape[0]))
        out[int(slot), :rows] = actions[:rows]
    return _tensor_from_numpy(out)


def owner_design_decision_payload() -> dict[str, Any]:
    return {
        "schema_version": OWNER_DESIGN_SCHEMA_VERSION,
        "selected_design": "A/single_pybind_cpp_class_with_persistent_at_tensor_members",
        "native_owner": True,
        "persistent_across_python_calls": True,
        "device_support": {"cpu": True, "cuda": False},
        "state_authority": "owner_clones_import_projection_tensors_and_returns_checkpoint_validation_projections",
        "generation_status": "owner_roundtrip_first_generation_loop_not_yet_integrated",
        "fallback_designs": [
            "B/torch_custom_class_or_intrusive_pointer_owner",
            "C/opaque_native_handle_with_explicit_functions",
        ],
    }


def _rng_manager_stream_indices_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    streams = snapshot.get("streams")
    if not isinstance(streams, Mapping):
        raise RuntimeError("RNGManager snapshot is missing its stream registry")
    out: dict[str, dict[str, Any]] = {}
    for stream_name in RNG_MANAGER_STREAM_NAMES:
        stream = streams.get(stream_name)
        indices = stream.get("indices") if isinstance(stream, Mapping) else None
        if not isinstance(indices, Mapping):
            raise RuntimeError(f"RNGManager snapshot is missing indices for stream={stream_name}")
        torch_cuda = indices.get("torch_cuda", {})
        if not isinstance(torch_cuda, Mapping):
            raise RuntimeError(f"RNGManager snapshot has invalid CUDA indices for stream={stream_name}")
        out[stream_name] = {
            "py": int(indices.get("py", 0) or 0),
            "np": int(indices.get("np", 0) or 0),
            "torch_cpu": int(indices.get("torch_cpu", 0) or 0),
            "torch_cuda": {
                str(device): int(value or 0)
                for device, value in sorted(torch_cuda.items(), key=lambda item: str(item[0]))
            },
        }
    return out


def _scalar_reconstructed_genome(genome: Any, *, fixed_mlp_family: bool) -> Any:
    from agents.developmental_program_v1 import canonicalize_developmental_program
    from agents.genome import Genome

    memory_dim = 0 if fixed_mlp_family else int(getattr(genome, "memory_dim", 0) or 0)
    return Genome(
        hidden_size=int(getattr(genome, "hidden_size", 64) or 64),
        exploration_bias=float(getattr(genome, "exploration_bias", 0.0) or 0.0),
        aggression_bias=float(getattr(genome, "aggression_bias", 0.0) or 0.0),
        reproduction_threshold=float(getattr(genome, "reproduction_threshold", 30.0) or 0.0),
        mutation_rate=float(getattr(genome, "mutation_rate", 0.05) or 0.0),
        action_temperature=float(getattr(genome, "action_temperature", 1.0) or 0.0),
        use_gru=False if fixed_mlp_family else bool(getattr(genome, "use_gru", False) or memory_dim > 0),
        memory_dim=memory_dim,
        sensor_channel_mask=int(getattr(genome, "sensor_channel_mask", 1023) or 0),
        sensor_radius=int(getattr(genome, "sensor_radius", 2) or 0),
        introspection_channel_mask=int(getattr(genome, "introspection_channel_mask", 32767) or 0),
        self_regulation_control_mask=int(getattr(genome, "self_regulation_control_mask", 31) or 0),
        self_modification_target_mask=int(getattr(genome, "self_modification_target_mask", 31) or 0),
        directed_inheritance_channel_mask=int(
            getattr(genome, "directed_inheritance_channel_mask", 63) or 0
        ),
        drive_inertia=float(getattr(genome, "drive_inertia", 0.85) or 0.0),
        drive_noise=float(getattr(genome, "drive_noise", 0.02) or 0.0),
        drive_logit_gain=float(getattr(genome, "drive_logit_gain", 1.0) or 0.0),
        pred_lr=float(getattr(genome, "pred_lr", 0.08) or 0.0),
        drive_sp_hunger=float(getattr(genome, "drive_sp_hunger", 0.55) or 0.0),
        drive_sp_pain=float(getattr(genome, "drive_sp_pain", 0.05) or 0.0),
        drive_sp_fatigue=float(getattr(genome, "drive_sp_fatigue", 0.15) or 0.0),
        drive_sp_fear=float(getattr(genome, "drive_sp_fear", 0.10) or 0.0),
        exp_prior_strength=float(getattr(genome, "exp_prior_strength", 0.35) or 0.0),
        exp_prior_noise=float(getattr(genome, "exp_prior_noise", 0.05) or 0.0),
        drive_birth_noise=float(getattr(genome, "drive_birth_noise", 0.03) or 0.0),
        developmental_program=canonicalize_developmental_program(None),
    )


def _nonprojected_genome_payload(genome: Any) -> dict[str, Any]:
    payload = copy.deepcopy(genome.canonical_payload())
    genes = dict(payload.get("genes") or {})
    for name in PROJECTED_GENOME_GENE_FIELDS:
        genes.pop(name, None)
    payload["genes"] = genes
    return payload


def _passive_modular_enabled(sim_cfg: Any) -> bool:
    raw = getattr(sim_cfg, "native_runtime_modular_storage_enable", False)
    if type(raw) is not bool:
        raise RuntimeError("native_runtime_modular_storage_enable must be an exact bool")
    return bool(raw)


def _require_passive_modular_boundary(sim_cfg: Any, *, tick: int) -> None:
    if int(tick) != 0:
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: passive modular storage requires tick=0"
        )
    raw_births = getattr(sim_cfg, "native_runtime_max_births_per_generation", 1)
    if isinstance(raw_births, bool) or not isinstance(raw_births, (int, np.integer)):
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: max births must be an exact integer"
        )
    if int(raw_births) != 0:
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: passive modular storage requires max births=0"
        )
    c1_enabled = getattr(sim_cfg, "genome_c1_enable", True)
    if type(c1_enabled) is not bool or c1_enabled is not False:
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: passive modular storage requires "
            "genome_c1_enable to be exact false"
        )
    c2_enabled = getattr(sim_cfg, "genome_c2_enable", True)
    if type(c2_enabled) is not bool or c2_enabled is not False:
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: passive modular storage requires "
            "genome_c2_enable to be exact false"
        )


def _passive_modular_storage_projection_for_runner(
    runner: Any,
    state: Any,
    *,
    tick: int,
) -> tuple[dict[str, dict[str, Any]], str] | None:
    """Pack the exact alive-slot modular state without consuming RNG."""

    sim_cfg = getattr(runner, "sim_cfg", None)
    if not _passive_modular_enabled(sim_cfg):
        return None
    _require_passive_modular_boundary(sim_cfg, tick=int(tick))

    from agents.modular_genome_v1 import (  # noqa: WPS433
        B2_CAPACITY_PROFILES,
        b2_capacity_profile_id,
    )
    from perf.cuda_architecture_b_modular_state_abi import (  # noqa: WPS433
        maybe_modular_storage_projection_payload,
    )

    capacity = int(getattr(state, "capacity", 0) or 0)
    if capacity <= 0:
        raise RuntimeError("passive modular storage requires a positive owner capacity")
    authoritative_slot_tokens = _slot_tokens_tensor(
        runner,
        np.asarray(state.slot_tokens, dtype=np.int64),
    )
    records: list[dict[str, Any] | None] = [None] * capacity
    profiles: set[str] = set()
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if slot < 0 or slot >= capacity:
            raise RuntimeError(
                f"passive modular storage has out-of-range alive slot={slot} capacity={capacity}"
            )
        if records[slot] is not None:
            raise RuntimeError(f"passive modular storage has duplicate alive slot={slot}")
        genome = getattr(agent, "genome", None)
        genome_to_dict = getattr(genome, "to_dict", None)
        if not callable(genome_to_dict):
            raise RuntimeError(
                f"passive modular storage requires Genome.to_dict at slot={slot}"
            )
        profile_id = b2_capacity_profile_id(getattr(genome, "b2_caps", None))
        profiles.add(profile_id)
        telemetry = getattr(agent, "structural_live_module_telemetry", {})
        telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
        cost_summary = telemetry.get("structural_cost")
        counts = cost_summary.get("counts") if isinstance(cost_summary, Mapping) else None
        saturated = (
            int(counts.get("saturated_or_rejected_activity", 0) or 0)
            if isinstance(counts, Mapping)
            else 0
        )
        record: dict[str, Any] = {
            "agent_id": int(getattr(agent, "agent_id", -1)),
            "agent_slot_token": int(authoritative_slot_tokens[slot].item()),
            "genome": copy.deepcopy(genome_to_dict()),
            "structural_live_module_state": copy.deepcopy(
                dict(getattr(agent, "structural_live_module_state", {}) or {})
            ),
            "structural_live_module_outputs": copy.deepcopy(
                dict(getattr(agent, "structural_live_module_outputs", {}) or {})
            ),
            "structural_development_latch_by_instance_id": copy.deepcopy(
                dict(
                    getattr(
                        agent,
                        "structural_development_latch_by_instance_id",
                        {},
                    )
                    or {}
                )
            ),
            "structural_development_effective_gain_by_instance_id": copy.deepcopy(
                dict(
                    getattr(
                        agent,
                        "structural_development_effective_gain_by_instance_id",
                        {},
                    )
                    or {}
                )
            ),
            "gated_recurrent_enabled": bool(
                getattr(sim_cfg, "structural_gated_recurrent_enable", False)
            ),
            "saturated_or_rejected_activity_count": saturated,
        }
        if isinstance(cost_summary, Mapping):
            record["cost_summary"] = copy.deepcopy(dict(cost_summary))
        records[slot] = record

    if len(profiles) > 1:
        raise RuntimeError(
            "passive modular storage requires one exact B2 profile: "
            f"profiles={sorted(profiles)}"
        )
    if profiles:
        profile_id = next(iter(profiles))
    else:
        from agents.modular_genome_v1 import b2_capacity_profile_id  # noqa: WPS433

        profile_id = b2_capacity_profile_id(
            {
                "max_modules_per_agent": getattr(
                    sim_cfg, "genome_b2_max_modules_per_agent", 12
                ),
                "max_connections_per_agent": getattr(
                    sim_cfg, "genome_b2_max_connections_per_agent", 32
                ),
                "max_module_fanin": getattr(sim_cfg, "genome_b2_max_module_fanin", 6),
                "max_module_fanout": getattr(sim_cfg, "genome_b2_max_module_fanout", 6),
            }
        )
    if profile_id not in B2_CAPACITY_PROFILES:
        raise RuntimeError(f"passive modular storage profile is unsupported: {profile_id}")
    payload = maybe_modular_storage_projection_payload(
        True,
        records,
        agent_capacity=capacity,
        profile_id=profile_id,
    )
    return payload, profile_id


def _validate_passive_modular_storage_projection(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    surface: str,
) -> Any | None:
    """Validate the optional overlay and bind it to resident biological slots."""

    from perf.cuda_architecture_b_modular_state_abi import (  # noqa: WPS433
        MODULAR_STORAGE_METADATA_KEY,
        MODULAR_STORAGE_PROJECTION_PREFIX,
        modular_storage_from_projection,
    )

    declared = metadata.get(PASSIVE_MODULAR_ENABLE_METADATA)
    has_envelope = MODULAR_STORAGE_METADATA_KEY in metadata
    has_planes = any(
        str(name).startswith(MODULAR_STORAGE_PROJECTION_PREFIX) for name in tensors
    )
    if declared is None:
        if has_envelope or has_planes:
            raise RuntimeError(
                f"{surface}: stray passive modular storage without enable declaration"
            )
        # The decoder also rejects mixed-version modular-looking keys.
        decoded = modular_storage_from_projection(metadata, tensors)
        if decoded is not None:
            raise RuntimeError(f"{surface}: undeclared passive modular storage")
        return None
    if declared is not True:
        raise RuntimeError(
            f"{surface}: {PASSIVE_MODULAR_ENABLE_METADATA} must be absent or true"
        )
    if int(metadata.get("tick", -1)) != 0 or str(
        metadata.get("agent_grid_runner_phase", "") or ""
    ) != "generation_boundary":
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: {surface} is not a supported tick-0 boundary"
        )
    if int(metadata.get("native_runtime_max_births_per_generation", -1)) != 0:
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: {surface} permits births"
        )
    if metadata.get("genome_c1_enable") is not False or metadata.get("genome_c2_enable") is not False:
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: {surface} permits C1/C2"
        )
    packed = modular_storage_from_projection(metadata, tensors)
    if packed is None:
        raise RuntimeError(f"{surface}: enabled passive modular storage is missing")
    required = {
        "slot_to_agent_id": (torch.int64, 1),
        "slot_tokens": (torch.int64, 1),
        "alive_mask": (torch.bool, 1),
    }
    resident: dict[str, torch.Tensor] = {}
    for name, (dtype, rank) in required.items():
        value = tensors.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != dtype
            or value.dim() != rank
        ):
            raise RuntimeError(f"{surface}: invalid resident modular binding tensor={name}")
        resident[name] = value.detach().contiguous()
    capacity = packed.layout.agent_capacity
    if any(int(value.shape[0]) != capacity for value in resident.values()):
        raise RuntimeError(f"{surface}: passive modular binding capacity mismatch")
    agent_id = packed.tensors["agent_id"]
    occupied = packed.tensors["agent_occupied"]
    alive = resident["alive_mask"]
    if not torch.equal(agent_id, resident["slot_to_agent_id"]):
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: {surface} agent membership changed"
        )
    if not torch.equal(occupied, alive):
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: {surface} occupancy changed"
        )
    normalized_tokens = torch.where(
        alive,
        resident["slot_tokens"],
        torch.full_like(resident["slot_tokens"], -1),
    )
    if not torch.equal(packed.tensors["agent_slot_token"], normalized_tokens):
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: {surface} slot token changed"
        )
    return packed


def _reference_genome_materialization_contract(
    runner: Any,
    *,
    modular_profile_id: str | None = None,
) -> dict[str, Any]:
    sim_cfg = getattr(runner, "sim_cfg", None)
    if modular_profile_id is not None:
        return {
            "schema_version": REFERENCE_GENOME_MATERIALIZATION_SCHEMA,
            "mode": REFERENCE_GENOME_MODULAR_MODE,
            "supported": True,
            "profile_id": str(modular_profile_id),
            "genome_c1_enable": False,
            "genome_c1_max_ops_per_offspring": 0,
            "genome_c2_enable": False,
            "unsupported_agent_ids": [],
            "reasons": [],
        }
    fixed_mlp_family = (
        str(getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or "").strip().lower() == "mlp"
    )
    reasons: list[str] = []
    unsupported_agent_ids: list[int] = []
    for agent in _alive_agents_from_runner(runner):
        genome = getattr(agent, "genome", None)
        try:
            reconstructed = _scalar_reconstructed_genome(genome, fixed_mlp_family=fixed_mlp_family)
            exact = _nonprojected_genome_payload(genome) == _nonprojected_genome_payload(reconstructed)
        except Exception:
            exact = False
        if not exact:
            unsupported_agent_ids.append(int(getattr(agent, "agent_id", -1)))
    if unsupported_agent_ids:
        reasons.append("non_scalar_canonical_payload")
    return {
        "schema_version": REFERENCE_GENOME_MATERIALIZATION_SCHEMA,
        "mode": REFERENCE_GENOME_SCALAR_MODE,
        "supported": not reasons,
        "genome_c1_enable": bool(getattr(sim_cfg, "genome_c1_enable", True)),
        "genome_c1_max_ops_per_offspring": int(getattr(sim_cfg, "genome_c1_max_ops_per_offspring", 2) or 0),
        "genome_c2_enable": bool(getattr(sim_cfg, "genome_c2_enable", True)),
        "unsupported_agent_ids": sorted(unsupported_agent_ids),
        "reasons": reasons,
    }


def _pending_generation_free_pairs_for_runner(
    runner: Any,
    *,
    capacity: int,
    slot_to_agent_id: torch.Tensor,
) -> tuple[tuple[int, int], ...]:
    pop = getattr(runner, "pop", None)
    slots = getattr(pop, "_slots", None)
    agents = dict(getattr(pop, "agents", {}) or {})
    if slots is None:
        return ()
    free_slots = {int(slot) for slot in getattr(slots, "free_slots", ())}
    slot_ids = slot_to_agent_id.detach().cpu().to(dtype=torch.int64)
    pending: list[tuple[int, int]] = []
    for agent in sorted(agents.values(), key=lambda row: int(getattr(row, "agent_id", -1))):
        if bool(getattr(agent, "alive", False)):
            continue
        agent_id = int(getattr(agent, "agent_id", -1))
        slot = int(getattr(agent, "slot", -1))
        if agent_id < 0 or slot < 0 or slot >= int(capacity) or slot in free_slots:
            continue
        if int(slot_ids[slot].item()) >= 0:
            raise RuntimeError(
                "resident owner projection dead-agent slot contradicts live slot mapping: "
                f"agent_id={agent_id} slot={slot}"
            )
        pending.append((agent_id, slot))
    return tuple(pending)


def build_import_projection(
    runner: Any,
    capability_mask: Mapping[str, bool],
    *,
    generation: int,
    tick: int,
) -> ImportProjectionV10:
    state = import_generation_start(runner, generation=int(generation), tick=int(tick))
    runnercore_state = None
    try:
        from simulation.runnercore.profile import SUPPORTED_PROFILE as _SUPPORTED_PROFILE  # noqa: WPS433
        from simulation.runnercore.tick_state import import_generation_state  # noqa: WPS433

        runnercore_state = import_generation_state(runner, generation=int(generation), profile=_SUPPORTED_PROFILE)
    except Exception:
        runnercore_state = None
    env = getattr(runner, "env", None)
    env_cfg = getattr(runner, "env_cfg", None)
    sim_cfg = getattr(runner, "sim_cfg", None)
    pop = getattr(runner, "pop", None)
    modular_projection_result = _passive_modular_storage_projection_for_runner(
        runner,
        state,
        tick=int(tick),
    )
    modular_projection = (
        modular_projection_result[0] if modular_projection_result is not None else None
    )
    modular_profile_id = (
        modular_projection_result[1] if modular_projection_result is not None else None
    )
    patch_food_gather_amount = 1.0
    patch_food_presence_threshold = 0.5
    patch_movement_cost_energy_scale = 1.0
    patch_scent_move_emit = 0.02
    patch_waste_move_emit = 0.0
    patch_scent_gather_emit = 0.05
    patch_waste_consume_emit = 0.05
    if env_cfg is not None:
        from env.gridworld_pkg.patch_world import (  # local import keeps Flat projection light
            _emit_value,
            patch_food_gather_amount as _patch_food_gather_amount,
            patch_food_presence_threshold as _patch_food_presence_threshold,
            patch_movement_cost_energy_scale as _patch_movement_cost_energy_scale,
        )

        patch_food_gather_amount = float(_patch_food_gather_amount(env_cfg))
        patch_food_presence_threshold = float(_patch_food_presence_threshold(env_cfg))
        patch_movement_cost_energy_scale = float(_patch_movement_cost_energy_scale(env_cfg))
        patch_scent_move_emit = float(_emit_value(env_cfg, "patch_scent_move_emit", 0.02))
        patch_scent_gather_emit = float(_emit_value(env_cfg, "patch_scent_gather_emit", 0.05))
        patch_waste_consume_emit = float(_emit_value(env_cfg, "patch_waste_consume_emit", 0.05))
    rng_manager = getattr(runner, "rng_manager", None)
    capture_rng_snapshot = getattr(rng_manager, "capture_snapshot", None)
    if not callable(capture_rng_snapshot):
        raise RuntimeError("resident owner projection requires RNGManager snapshot support")
    rng_manager_snapshot = capture_rng_snapshot(generation_index=int(generation))
    rng_manager_stream_indices = _rng_manager_stream_indices_from_snapshot(rng_manager_snapshot)
    policy_np = getattr(rng_manager, "policy_np", None)
    env_rng = getattr(env, "rng", None)
    env_rng_state = copy.deepcopy(getattr(getattr(env_rng, "bit_generator", None), "state", None))
    runner_rng_state = copy.deepcopy(getattr(getattr(getattr(runner, "np_rng", None), "bit_generator", None), "state", None))
    policy_rng_state = None
    try:
        policy_rng_state = copy.deepcopy(getattr(policy_np, "_get_bitgen_state", lambda: None)())
    except Exception:
        policy_rng_state = None
    population_rng_state = _python_random_state_payload(getattr(getattr(pop, "rng", None), "getstate", lambda: None)())
    population_np_rng_state = copy.deepcopy(getattr(getattr(getattr(pop, "np_rng", None), "bit_generator", None), "state", None))
    env_rng_scalar = 1
    try:
        env_rng_scalar = int(env_rng_state["state"]["state"]) & 0x7FFFFFFFFFFFFFFF
    except Exception:
        env_rng_scalar = int(getattr(sim_cfg, "seed", 0) or 0) & 0x7FFFFFFFFFFFFFFF
    world = "patch" if bool(getattr(env_cfg, "patch_world_enable", False)) else "flat"
    patch_contract_metadata: dict[str, Any] = {}
    patch_contract_tensors: dict[str, torch.Tensor] = {}
    if world == "patch":
        patch_contract_metadata = _patch_profile_metadata(
            runner,
            env_cfg,
            sim_cfg,
            capability_mask,
        )
        patch_contract_tensors = _patch_import_arrays(
            env,
            env_cfg,
            profile=patch_contract_metadata["native_runtime_patch_profile"],
            include_wdl_storage=bool(
                capability_mask.get(WDL_STORAGE_CAPABILITY, False)
                or capability_mask.get("wdl", False)
            ),
            include_eed_storage=bool(
                capability_mask.get(EED_EXECUTION_CAPABILITY, False)
            ),
            agent_id_capacity=int(state.capacity),
        )
    agent_grid_runner_phase = "generation_boundary" if int(tick) == 0 else "partial_generation"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generation": int(generation),
        "tick": int(tick),
        "capacity": int(state.capacity),
        "alive_count": int(state.alive_count),
        "population_size": int(getattr(getattr(runner, "pop_cfg", None), "population_size", state.alive_count)),
        "parent_fraction": float(getattr(getattr(runner, "pop_cfg", None), "parent_fraction", 0.25) or 0.25),
        "cull_fraction": float(getattr(getattr(runner, "pop_cfg", None), "cull_fraction", 0.5) or 0.0),
        "episode0_fitness_weight": float(getattr(sim_cfg, "episode0_fitness_weight", 0.0) or 0.0),
        "episode0_pred_err_fitness_weight": float(getattr(sim_cfg, "episode0_pred_err_fitness_weight", 0.0) or 0.0),
        "episode0_thrashing_fitness_weight": float(getattr(sim_cfg, "episode0_thrashing_fitness_weight", 0.0) or 0.0),
        "episode0_drive_jitter_fitness_weight": float(getattr(sim_cfg, "episode0_drive_jitter_fitness_weight", 0.0) or 0.0),
        "episode0_selfharm_fitness_weight": float(getattr(sim_cfg, "episode0_selfharm_fitness_weight", 0.0) or 0.0),
        "coherence_thrashing_weight": float(getattr(sim_cfg, "coherence_thrashing_weight", 0.15) or 0.0),
        "coherence_drive_jitter_weight": float(getattr(sim_cfg, "coherence_drive_jitter_weight", 0.08) or 0.0),
        "coherence_drive_volatility_weight": float(
            getattr(sim_cfg, "coherence_drive_volatility_weight", 0.08) or 0.0
        ),
        "coherence_selfharm_weight": float(getattr(sim_cfg, "coherence_selfharm_weight", 3.0) or 0.0),
        "coherence_pred_err_weight": float(getattr(sim_cfg, "coherence_pred_err_weight", 1.25) or 0.0),
        "coherence_reset_shock_weight": float(getattr(sim_cfg, "coherence_reset_shock_weight", 0.35) or 0.0),
        "coherence_episode0_pred_err_weight": float(getattr(sim_cfg, "coherence_episode0_pred_err_weight", 0.75) or 0.0),
        "pred_horizon": max(0, min(8, int(getattr(sim_cfg, "pred_horizon", 2) or 0))),
        "pred_injury_scale": float(getattr(sim_cfg, "pred_injury_scale", 1.0) or 0.0),
        "prediction_tick": int(getattr(runner, "_pred_tick", 0) or 0),
        "force_memory_disabled": bool(
            getattr(sim_cfg, "force_memory_disabled", False)
        ),
        "reset_memory_each_tick": bool(
            getattr(sim_cfg, "reset_memory_each_tick", False)
        ),
        "ablate_prediction": bool(getattr(sim_cfg, "ablate_prediction", False)),
        "force_prediction_disabled": bool(getattr(sim_cfg, "force_prediction_disabled", False)),
        "force_prediction_frozen": bool(getattr(sim_cfg, "force_prediction_frozen", False)),
        "force_prediction_zero_lr": bool(getattr(sim_cfg, "force_prediction_zero_lr", False)),
        "force_prediction_default": bool(getattr(sim_cfg, "force_prediction_default", False)),
        "reset_prediction_each_tick": bool(getattr(sim_cfg, "reset_prediction_each_tick", False)),
        "episode0_k": int(getattr(sim_cfg, "episode0_k", 20) or 0),
        "adult_move_penalty_enable": bool(getattr(sim_cfg, "adult_move_penalty_enable", False)),
        "adult_move_penalty_start_step": int(
            getattr(sim_cfg, "adult_move_penalty_start_step", getattr(sim_cfg, "episode0_k", 20)) or 0
        ),
        "adult_pred_err_ema_alpha": float(getattr(sim_cfg, "adult_pred_err_ema_alpha", 0.1) or 0.0),
        "disable_prediction_error_selection_pressure": bool(getattr(sim_cfg, "disable_prediction_error_selection_pressure", False)),
        "late_pred_cost_enable": bool(getattr(sim_cfg, "late_pred_cost_enable", False)),
        "late_pred_cost_weight": float(getattr(sim_cfg, "late_pred_cost_weight", 0.0) or 0.0),
        "lineage_dominance_threshold": float(
            getattr(sim_cfg, "lineage_dominance_threshold", 0.80)
        ),
        "lineage_entropy_minimum": float(
            getattr(sim_cfg, "lineage_entropy_minimum", 0.10)
        ),
        "lineage_reproduction_dominance_threshold": float(
            getattr(
                sim_cfg,
                "lineage_reproduction_dominance_threshold",
                0.80,
            )
        ),
        "lineage_dominance_streak_required": int(
            getattr(sim_cfg, "lineage_dominance_streak_required", 5)
        ),
        "lineage_min_active_lineages": int(
            getattr(sim_cfg, "lineage_min_active_lineages", 2)
        ),
        "lineage_reproduction_dominance_streak_required": int(
            getattr(
                sim_cfg,
                "lineage_reproduction_dominance_streak_required",
                5,
            )
        ),
        "native_compact_lineage_collapse_streak": 0,
        "native_compact_lineage_population_dominance_streak": 0,
        "native_compact_lineage_reproduction_dominance_streak": 0,
        "device": str(getattr(sim_cfg, "device", "cpu") or "cpu"),
        "capability_mask": {str(k): bool(v) for k, v in dict(capability_mask).items()},
        "policy_tick_stride": int(getattr(sim_cfg, "policy_tick_stride", 1) or 1),
        "configured_steps_per_generation": int(getattr(sim_cfg, "steps_per_generation", 0) or 0),
        "world": world,
        "agent_grid_phase_contract_schema_version": "eiel_agent_grid_phase_contract_v1",
        "agent_grid_world_mode": world,
        "agent_grid_runner_phase": agent_grid_runner_phase,
        "agent_grid_authority_epoch": f"generation:{int(generation)}:tick:{int(tick)}",
        "agent_grid_synchronization_state": "stored_grid_authoritative",
        "agent_grid_preserve_exact_stored": True,
        "agent_grid_reconstructible_without_phase_context": False,
        "event_tape_contract_schema_version": "eiel_event_tape_active_inactive_contract_v1",
        "event_tape_active_count": 0,
        "event_tape_cursor": 0,
        "event_tape_inactive_tail_count": 0,
        "event_tape_active_semantics": "no_active_typed_rows",
        "event_tape_public_projection": "not_public_output_canonicalized",
        "view_radius": int(getattr(env_cfg, "view_radius", 1)),
        "move_cost": float(getattr(env_cfg, "move_cost", 1.0)),
        "gather_cost": float(getattr(env_cfg, "gather_cost", 1.0)),
        "rest_cost": float(getattr(env_cfg, "rest_cost", 0.25)),
        "enable_rest_charge": bool(getattr(env_cfg, "enable_rest_charge", False)),
        "rest_streak_free": int(getattr(sim_cfg, "rest_streak_free", 0) or 0),
        "rest_charge_energy": float(getattr(sim_cfg, "rest_charge_energy", 0.0) or 0.0),
        "rest_charge_growth": float(getattr(sim_cfg, "rest_charge_growth", 0.0) or 0.0),
        "rest_deficit_to_health": float(getattr(sim_cfg, "rest_deficit_to_health", 1.0) or 1.0),
        "resource_maintenance_energy_cost_per_tick": float(
            getattr(sim_cfg, "resource_maintenance_energy_cost_per_tick", 0.0)
            or 0.0
        ),
        "initial_spatial_placement_profile": str(
            getattr(sim_cfg, "initial_spatial_placement_profile", "heritage_random_v1")
            or "heritage_random_v1"
        ),
        "attack_cost": float(getattr(env_cfg, "attack_cost", 0.60)),
        "enable_combat": bool(getattr(env_cfg, "enable_combat", False)),
        "attack_damage": float(getattr(env_cfg, "attack_damage", 0.0) or 0.0),
        "injury_per_damage": float(getattr(env_cfg, "injury_per_damage", 0.0) or 0.0),
        "injury_max": float(getattr(env_cfg, "injury_max", 10.0) or 10.0),
        "scar_max": float(getattr(env_cfg, "scar_max", getattr(env_cfg, "injury_max", 10.0)) or 10.0),
        "scar_start_threshold": float(getattr(env_cfg, "scar_start_threshold", 0.0) or 0.0),
        "scar_gain_per_injury": float(getattr(env_cfg, "scar_gain_per_injury", 0.0) or 0.0),
        "scar_gain_base": float(getattr(env_cfg, "scar_gain_base", 0.0) or 0.0),
        "gru_cost_base": float(getattr(env_cfg, "gru_cost_base", 0.0)),
        "gru_cost_per_dim": float(getattr(env_cfg, "gru_cost_per_dim", 0.0)),
        "developmental_maturity_age": int(getattr(sim_cfg, "developmental_maturity_age", 20) or 0),
        "action_cost_mult": float(getattr(env, "action_cost_mult", 1.0)),
        "rest_cost_mult": float(getattr(env, "rest_cost_mult", 1.0)),
        "start_energy": float(getattr(env_cfg, "start_energy", 60.0)),
        "max_energy": float(getattr(env_cfg, "max_energy", 20.0)),
        "max_food": int(getattr(env_cfg, "max_food", 0) or 0),
        "food_spawn_per_step": float(getattr(env_cfg, "food_spawn_per_step", 0.0) or 0.0),
        "food_spawn_mult": float(getattr(env, "food_spawn_mult", 1.0)),
        "food_energy": float(getattr(env_cfg, "food_energy", 5.0)),
        "enable_biomes": bool(getattr(env_cfg, "enable_biomes", False)),
        "biome_safe_food_mult": float(getattr(env_cfg, "biome_safe_food_mult", 1.0) or 1.0),
        "biome_wild_food_mult": float(getattr(env_cfg, "biome_wild_food_mult", 1.0) or 1.0),
        "biome_wild_hazard_prob": float(getattr(env_cfg, "biome_wild_hazard_prob", 0.0) or 0.0),
        "biome_wild_hazard_damage": float(getattr(env_cfg, "biome_wild_hazard_damage", 0.0) or 0.0),
        "biome_wild_hazard_injury_per_damage": float(
            getattr(env_cfg, "biome_wild_hazard_injury_per_damage", 0.0) or 0.0
        ),
        "biome_wild_action_cost_mult": float(getattr(env_cfg, "biome_wild_action_cost_mult", 1.0)),
        "start_health": float(getattr(env_cfg, "start_health", 1.0) or 1.0),
        "reproduce_requires_mate": bool(getattr(env_cfg, "reproduce_requires_mate", False)),
        "inheritance_profile": str(
            getattr(sim_cfg, "inheritance_profile", "legacy_midpoint_v1")
            or "legacy_midpoint_v1"
        ),
        "diploid_mutation_enable": bool(
            getattr(sim_cfg, "diploid_mutation_enable", True)
        ),
        "recomb72_mutation_off": bool(
            getattr(sim_cfg, "recomb72_mutation_off", False)
        ),
        "resource_handling_representation_enable": bool(
            getattr(
                sim_cfg,
                "resource_handling_representation_enable",
                False,
            )
        ),
        "genome_upgrade_v2_enable": bool(
            getattr(sim_cfg, "genome_upgrade_v2_enable", False)
        ),
        "genome_upgrade_v2_mutation_rate_q10": int(
            getattr(sim_cfg, "genome_upgrade_v2_mutation_rate_q10", 20)
        ),
        "genome_upgrade_v2_mutation_step_q10": int(
            getattr(sim_cfg, "genome_upgrade_v2_mutation_step_q10", 64)
        ),
        "reproduce_try_prob": float(getattr(env_cfg, "reproduce_try_prob", 0.0) or 0.0),
        "reproduce_parent_energy_share": float(getattr(env_cfg, "reproduce_parent_energy_share", 0.5)),
        "reproduce_min_energy_buffer": float(getattr(env_cfg, "reproduce_min_energy_buffer", 1.0)),
        "reproduce_cooldown_steps": int(getattr(env_cfg, "reproduce_cooldown_steps", 0) or 0),
        "population_next_id": int(getattr(getattr(runner, "pop", None), "next_id", 0) or 0),
        "enable_evolvable_sensor_channel_mask": bool(getattr(sim_cfg, "enable_evolvable_sensor_channel_mask", False)),
        "enable_evolvable_sensor_radius": bool(getattr(sim_cfg, "enable_evolvable_sensor_radius", False)),
        "enable_evolvable_introspection_channels": bool(getattr(sim_cfg, "enable_evolvable_introspection_channels", False)),
        "enable_evolvable_self_regulation_controls": bool(getattr(sim_cfg, "enable_evolvable_self_regulation_controls", False)),
        "enable_evolvable_self_modification_targets": bool(getattr(sim_cfg, "enable_evolvable_self_modification_targets", False)),
        "enable_evolvable_directed_inheritance_channels": bool(getattr(sim_cfg, "enable_evolvable_directed_inheritance_channels", False)),
        "chemistry_functional_coupling_enable": bool(getattr(sim_cfg, "chemistry_functional_coupling_enable", False)),
        "genome_sensor_radius_max": int(getattr(getattr(sim_cfg, "env_cfg", None), "view_radius", 0) or 0),
        "native_runtime_rng_state": int(env_rng_scalar),
        "native_runtime_numpy_env_state": env_rng_state,
        "native_runtime_numpy_runner_state": runner_rng_state,
        "native_runtime_numpy_policy_state": policy_rng_state,
        "native_runtime_env_policy_numpy_alias": bool(env_rng is policy_np),
        "native_runtime_use_precomputed_drive_noise": bool(
            getattr(sim_cfg, "native_runtime_use_precomputed_drive_noise", False)
        ),
        "native_runtime_drive_noise_mode": (
            "rngv2_gpu_formula_v1"
            if int(getattr(sim_cfg, "rng_semantics_version", 1) or 1) == 2
            else "precomputed_numpy_pcg64_v1"
        ),
        "tw_rngv2_semantics_version": int(
            getattr(sim_cfg, "rng_semantics_version", 1) or 1
        ),
        "tw_rngv2_world_uid": int(getattr(sim_cfg, "world_uid", 0) or 0),
        "tw_rngv2_namespace_key64": int(
            getattr(
                sim_cfg,
                "rng_namespace_key64",
                0xCA05117E450A51C0,
            )
        ),
        "tw_rngv2_semantic_epoch": int(
            getattr(sim_cfg, "rng_semantic_epoch", 1)
        ),
        "native_runtime_numpy_population_state": population_np_rng_state,
        "native_runtime_python_random_state": population_rng_state,
        RNG_MANAGER_STREAM_INDICES_METADATA: rng_manager_stream_indices,
        "policy_deterministic_actions": bool(getattr(sim_cfg, "policy_deterministic_actions", False)),
        "native_runtime_birth_mutation_sigma": float(
            getattr(sim_cfg, "native_runtime_birth_mutation_sigma", 0.0) or 0.0
        ),
        "policy_weight_mutation_enable": bool(getattr(sim_cfg, "policy_weight_mutation_enable", False)),
        "policy_weight_mutation_mode": str(getattr(sim_cfg, "policy_weight_mutation_mode", "dense") or "dense"),
        "policy_weight_mutation_rate": float(getattr(sim_cfg, "policy_weight_mutation_rate", 1.0) or 0.0),
        "policy_weight_mutation_std": float(getattr(sim_cfg, "policy_weight_mutation_std", 0.0) or 0.0),
        "genome_c1_enable": bool(getattr(sim_cfg, "genome_c1_enable", True)),
        "genome_c2_enable": bool(getattr(sim_cfg, "genome_c2_enable", True)),
        REFERENCE_GENOME_MATERIALIZATION_METADATA: _reference_genome_materialization_contract(
            runner,
            modular_profile_id=modular_profile_id,
        ),
        "native_runtime_fixed_policy_family": str(getattr(sim_cfg, "native_runtime_fixed_policy_family", "") or ""),
        "patch_food_amount_max": float(getattr(env_cfg, "patch_food_amount_max", 1.0) or 1.0),
        "patch_scent_max": float(getattr(env_cfg, "patch_scent_max", 1.0) or 1.0),
        "patch_waste_max": float(getattr(env_cfg, "patch_waste_max", 1.0) or 1.0),
        "patch_density_radius": int(getattr(env_cfg, "patch_density_radius", 1)),
        "patch_density_include_self": bool(getattr(env_cfg, "patch_density_include_self", True)),
        "patch_density_normalize": bool(getattr(env_cfg, "patch_density_normalize", False)),
        "patch_movement_cost_min": float(getattr(env_cfg, "patch_movement_cost_min", 0.0)),
        "patch_movement_cost_max": float(getattr(env_cfg, "patch_movement_cost_max", 4.0)),
        "patch_food_gather_amount": patch_food_gather_amount,
        "patch_food_presence_threshold": patch_food_presence_threshold,
        "patch_scent_move_emit": patch_scent_move_emit,
        "patch_waste_move_emit": patch_waste_move_emit,
        "patch_scent_gather_emit": patch_scent_gather_emit,
        "patch_waste_consume_emit": patch_waste_consume_emit,
        "patch_scent_decay_per_step": float(getattr(env_cfg, "patch_scent_decay_per_step", 0.05)),
        "patch_waste_decay_per_step": float(getattr(env_cfg, "patch_waste_decay_per_step", 0.02)),
        "patch_scent_presence_emit": float(getattr(env_cfg, "patch_scent_presence_emit", 0.01)),
        "patch_waste_presence_emit": float(getattr(env_cfg, "patch_waste_presence_emit", 0.005)),
        "patch_waste_density_emit_scale": float(getattr(env_cfg, "patch_waste_density_emit_scale", 0.0)),
        "patch_food_regrow_per_step": float(getattr(env_cfg, "patch_food_regrow_per_step", 0.02)),
        "patch_food_regrow_use_fertility": bool(getattr(env_cfg, "patch_food_regrow_use_fertility", True)),
        "patch_food_regrow_cap_by_fertility": bool(getattr(env_cfg, "patch_food_regrow_cap_by_fertility", False)),
        "patch_movement_cost_apply_enable": bool(getattr(env_cfg, "patch_movement_cost_apply_enable", False)),
        "patch_movement_cost_energy_scale": patch_movement_cost_energy_scale,
        "enable_injury": bool(getattr(env_cfg, "enable_injury", False)),
        "enable_scar": bool(getattr(env_cfg, "enable_scar", False)),
        "injury_action_cost_penalty": float(getattr(env_cfg, "injury_action_cost_penalty", 0.0)),
        "injury_max_energy_penalty": float(getattr(env_cfg, "injury_max_energy_penalty", 0.0)),
        "injury_min_energy_frac": float(getattr(env_cfg, "injury_min_energy_frac", 0.20)),
        "injury_penalty_curve_power": float(getattr(env_cfg, "injury_penalty_curve_power", 1.0)),
        "scar_cost_weight": float(getattr(env_cfg, "scar_cost_weight", 1.0)),
        "injury_repro_block": None
        if getattr(env_cfg, "injury_repro_block", None) is None
        else float(getattr(env_cfg, "injury_repro_block")),
        "scar_repro_block": None
        if getattr(env_cfg, "scar_repro_block", None) is None
        else float(getattr(env_cfg, "scar_repro_block")),
        "injury_repro_k_injury": float(getattr(env_cfg, "injury_repro_k_injury", 0.0) or 0.0),
        "injury_repro_k_scar": float(getattr(env_cfg, "injury_repro_k_scar", 0.0) or 0.0),
        "injury_birth_fail_scar": float(getattr(env_cfg, "injury_birth_fail_scar", 0.0) or 0.0),
        "injury_birth_fail_injury": float(getattr(env_cfg, "injury_birth_fail_injury", 0.0) or 0.0),
        "obs_size": int(getattr(runner, "_obs_size", 0) or 0),
        "hidden": int(getattr(runner, "_hidden", 0) or 0),
        "gru_hidden": int(getattr(runner, "_gru_hidden", getattr(runner, "_hidden", 0)) or 0),
        "action_count": int(getattr(runner, "_out", 7) or 7),
        "drive_w_hunger_gather": float(getattr(sim_cfg, "drive_w_hunger_gather", 1.2)),
        "drive_w_pain_rest": float(getattr(sim_cfg, "drive_w_pain_rest", 0.9)),
        "drive_w_fatigue_rest": float(getattr(sim_cfg, "drive_w_fatigue_rest", 1.0)),
        "drive_w_pain_attack": float(getattr(sim_cfg, "drive_w_pain_attack", 1.2)),
        "drive_w_fear_attack": float(getattr(sim_cfg, "drive_w_fear_attack", 1.4)),
        "drive_hunger_gain": float(getattr(sim_cfg, "drive_hunger_gain", 0.08)),
        "drive_hunger_decay": float(getattr(sim_cfg, "drive_hunger_decay", 0.02)),
        "drive_pain_gain": float(getattr(sim_cfg, "drive_pain_gain", 0.18)),
        "drive_pain_decay": float(getattr(sim_cfg, "drive_pain_decay", 0.03)),
        "drive_fatigue_gain": float(getattr(sim_cfg, "drive_fatigue_gain", 0.05)),
        "drive_fatigue_rest_relief": float(getattr(sim_cfg, "drive_fatigue_rest_relief", 0.10)),
        "drive_fatigue_decay": float(getattr(sim_cfg, "drive_fatigue_decay", 0.01)),
        "drive_fear_gain": float(getattr(sim_cfg, "drive_fear_gain", 0.22)),
        "drive_fear_decay": float(getattr(sim_cfg, "drive_fear_decay", 0.03)),
    }
    if world == "patch":
        metadata.update(patch_contract_metadata)
        if metadata.get("native_runtime_wdl_storage_enable") is True:
            from env.gridworld_pkg.patch_world import _wdl_seed  # noqa: WPS433

            configured_steps = int(
                metadata.get("configured_steps_per_generation", 0) or 0
            )
            metadata["wdl_base_seed"] = int(_wdl_seed(env))
            metadata["wdl_next_absolute_step"] = (
                int(generation) * configured_steps + int(tick)
            )
    if modular_projection is not None:
        metadata[PASSIVE_MODULAR_ENABLE_METADATA] = True
        metadata["native_runtime_max_births_per_generation"] = 0
        metadata.update(copy.deepcopy(modular_projection["metadata"]))
    if world == "flat":
        metadata["native_runtime_numpy_birth_spawn_env_state"] = copy.deepcopy(env_rng_state)
    else:
        metadata["native_runtime_numpy_birth_spawn_env_state"] = copy.deepcopy(env_rng_state)
    metadata_overrides = getattr(runner, "_cxxbackbone10_checkpoint_metadata_overrides", None)
    if isinstance(metadata_overrides, Mapping):
        try:
            same_boundary = (
                int(metadata_overrides.get("generation", -1)) == int(generation)
                and int(metadata_overrides.get("tick", -1)) == int(tick)
            )
        except Exception:
            same_boundary = False
        if same_boundary:
            for name in (
                "native_runtime_numpy_env_state",
                "native_runtime_numpy_birth_spawn_env_state",
                "native_runtime_numpy_runner_state",
                "native_runtime_numpy_policy_state",
                "native_runtime_numpy_population_state",
                "native_runtime_python_random_state",
                "native_runtime_env_policy_numpy_alias",
                RNG_MANAGER_STREAM_INDICES_METADATA,
            ):
                if name in metadata_overrides:
                    metadata[name] = copy.deepcopy(metadata_overrides[name])
    tensors = {
        "agent_ids_by_alive_order": _tensor_from_numpy(state.agent_id_to_slot_agent_ids),
        "slot_to_agent_id": _tensor_from_numpy(state.slot_to_agent_id),
        "agent_id_to_slot_agent_ids": _tensor_from_numpy(state.agent_id_to_slot_agent_ids),
        "agent_id_to_slot_slots": _tensor_from_numpy(state.agent_id_to_slot_slots),
        "slot_tokens": _slot_tokens_tensor(runner, state.slot_tokens),
        "slot_free_queue": _slot_free_queue_tensor(runner),
        "alive_mask": _tensor_from_numpy(state.alive_mask),
        "pos_y": _tensor_from_numpy(state.pos_y),
        "pos_x": _tensor_from_numpy(state.pos_x),
        "energy": _tensor_from_numpy(state.energy),
        "policy_energy_observation": _agent_float64_by_slot(runner, int(state.capacity), "energy", float("nan")),
        "energy_collected_exact": _stat_float64_by_slot(
            runner,
            int(state.capacity),
            "energy_collected",
            0.0,
        ),
        "early_energy_collected_exact": _stat_float64_by_slot(
            runner,
            int(state.capacity),
            "early_energy_collected",
            0.0,
        ),
        "health": _tensor_from_numpy(state.health),
        "injury": _tensor_from_numpy(state.injury),
        "scar": _scar_by_slot(runner, int(state.capacity)),
        "age": _tensor_from_numpy(state.age),
        "lineage_founder_id": _lineage_founder_id_by_slot(
            runner, int(state.capacity)
        ),
        "lineage_depth": _stat_int_by_slot(runner, int(state.capacity), "lineage_depth", 0),
        "lineage_birth_generation": _stat_int_by_slot(runner, int(state.capacity), "birth_generation", 0),
        "lineage_birth_tick": _stat_int_by_slot(runner, int(state.capacity), "birth_tick", 0),
        "offspring_count": _stat_int_by_slot(runner, int(state.capacity), "offspring_count", 0),
        "first_repro_step": _first_repro_step_by_slot(runner, int(state.capacity)),
        "survival_steps": _stat_int_by_slot(runner, int(state.capacity), "survival_steps", 0),
        "early_survival_steps": _stat_int_by_slot(
            runner,
            int(state.capacity),
            "early_survival_steps",
            0,
        ),
        "early_life_weight": _stat_float_by_slot(runner, int(state.capacity), "early_life_weight", 1.0),
        "early_injury_gained": _stat_float_by_slot(
            runner,
            int(state.capacity),
            "early_injury_gained",
            0.0,
        ),
        "dev_stage_initialized": _dev_stage_initialized_by_slot(runner, int(state.capacity)),
        "attacks": _stat_int_by_slot(runner, int(state.capacity), "attacks", 0),
        "last_action": _zero_free_rows_i32(
            _stat_int_by_slot(runner, int(state.capacity), "last_action", -1),
            state.alive_mask,
        ),
        "last_last_action": _zero_free_rows_i32(
            _stat_int_by_slot(runner, int(state.capacity), "last_last_action", -1),
            state.alive_mask,
        ),
        "action_switches": _stat_int_by_slot(runner, int(state.capacity), "action_switches", 0),
        "action_oscillations": _stat_int_by_slot(runner, int(state.capacity), "action_oscillations", 0),
        "episode0_switches": _stat_int_by_slot(runner, int(state.capacity), "episode0_switches", 0),
        "episode0_oscillations": _stat_int_by_slot(runner, int(state.capacity), "episode0_oscillations", 0),
        "selfharm_events": _stat_int_by_slot(runner, int(state.capacity), "selfharm_events", 0),
        "episode0_selfharm_events": _stat_int_by_slot(runner, int(state.capacity), "episode0_selfharm_events", 0),
        "pred_err_count": _stat_int_by_slot(runner, int(state.capacity), "pred_err_count", 0),
        "episode0_pred_err_count": _stat_int_by_slot(runner, int(state.capacity), "episode0_pred_err_count", 0),
        "late_pred_err_count": _stat_int_by_slot(runner, int(state.capacity), "late_pred_err_count", 0),
        "birth_reset_shock": _stat_float_by_slot(runner, int(state.capacity), "birth_reset_shock", 0.0),
        "drive_jitter": _stat_float_by_slot(runner, int(state.capacity), "drive_jitter", 0.0),
        "pred_err_sum": _stat_float_by_slot(runner, int(state.capacity), "pred_err_sum", 0.0),
        "pred_lr_exact": _effective_pred_lr_exact_by_slot(runner, int(state.capacity)),
        "pred_err_sum_exact": _stat_float64_by_slot(runner, int(state.capacity), "pred_err_sum", 0.0),
        "episode0_pred_err_sum": _stat_float_by_slot(runner, int(state.capacity), "episode0_pred_err_sum", 0.0),
        "episode0_pred_err_sum_exact": _stat_float64_by_slot(
            runner, int(state.capacity), "episode0_pred_err_sum", 0.0
        ),
        "late_pred_err_sum": _stat_float_by_slot(runner, int(state.capacity), "late_pred_err_sum", 0.0),
        "late_pred_err_sum_exact": _stat_float64_by_slot(runner, int(state.capacity), "late_pred_err_sum", 0.0),
        "prediction_state_delta_abs_sum_exact": _stat_float64_by_slot(
            runner, int(state.capacity), "prediction_state_delta_abs_sum", 0.0
        ),
        "prediction_state_delta_abs_max_exact": _stat_float64_by_slot(
            runner, int(state.capacity), "prediction_state_delta_abs_max", 0.0
        ),
        "adult_pred_err_ema_exact": _stat_float64_by_slot(
            runner, int(state.capacity), "adult_pred_err_ema", 0.0
        ),
        "patch_habitat_lineage_id": _stat_int_by_slot(runner, int(state.capacity), "patch_habitat_lineage_id", 0),
        "genome_hidden_size": _genome_int_by_slot(runner, int(state.capacity), "hidden_size", 64),
        "genome_memory_dim": _genome_int_by_slot(runner, int(state.capacity), "memory_dim", 0),
        "genome_sensor_channel_mask": _genome_int_by_slot(runner, int(state.capacity), "sensor_channel_mask", 1023),
        "genome_sensor_radius": _genome_int_by_slot(runner, int(state.capacity), "sensor_radius", 2),
        "genome_introspection_channel_mask": _genome_int_by_slot(
            runner,
            int(state.capacity),
            "introspection_channel_mask",
            32767,
        ),
        "genome_self_regulation_control_mask": _genome_int_by_slot(
            runner,
            int(state.capacity),
            "self_regulation_control_mask",
            31,
        ),
        "genome_self_modification_target_mask": _genome_int_by_slot(
            runner,
            int(state.capacity),
            "self_modification_target_mask",
            31,
        ),
        "genome_directed_inheritance_channel_mask": _genome_int_by_slot(
            runner,
            int(state.capacity),
            "directed_inheritance_channel_mask",
            63,
        ),
        "genome_exploration_bias": _genome_float_by_slot(runner, int(state.capacity), "exploration_bias", 0.0),
        "genome_aggression_bias": _genome_float_by_slot(runner, int(state.capacity), "aggression_bias", 0.0),
        "genome_mutation_rate": _genome_float_by_slot(runner, int(state.capacity), "mutation_rate", 0.05),
        "genome_reproduction_threshold": _genome_float_by_slot(
            runner,
            int(state.capacity),
            "reproduction_threshold",
            float(getattr(env_cfg, "reproduction_threshold", 12.0) if env_cfg is not None else 12.0),
        ),
        "genome_action_temperature": _genome_float_by_slot(runner, int(state.capacity), "action_temperature", 1.0),
        "genome_drive_inertia": _genome_float_by_slot(runner, int(state.capacity), "drive_inertia", 0.85),
        "genome_drive_noise": _genome_float_by_slot(runner, int(state.capacity), "drive_noise", 0.02),
        "genome_drive_logit_gain": _genome_float_by_slot(runner, int(state.capacity), "drive_logit_gain", 1.0),
        "genome_pred_lr": _genome_float_by_slot(runner, int(state.capacity), "pred_lr", 0.08),
        "genome_drive_sp_hunger": _genome_float_by_slot(runner, int(state.capacity), "drive_sp_hunger", 0.55),
        "genome_drive_sp_pain": _genome_float_by_slot(runner, int(state.capacity), "drive_sp_pain", 0.05),
        "genome_drive_sp_fatigue": _genome_float_by_slot(runner, int(state.capacity), "drive_sp_fatigue", 0.15),
        "genome_drive_sp_fear": _genome_float_by_slot(runner, int(state.capacity), "drive_sp_fear", 0.10),
        "genome_exp_prior_strength": _genome_float_by_slot(runner, int(state.capacity), "exp_prior_strength", 0.35),
        "genome_exp_prior_noise": _genome_float_by_slot(runner, int(state.capacity), "exp_prior_noise", 0.05),
        "genome_drive_birth_noise": _genome_float_by_slot(runner, int(state.capacity), "drive_birth_noise", 0.03),
        "event_tape_type": torch.empty((0,), dtype=torch.int32),
        "event_tape_generation": torch.empty((0,), dtype=torch.int64),
        "event_tape_tick": torch.empty((0,), dtype=torch.int64),
        "event_tape_actor_id": torch.empty((0,), dtype=torch.int64),
        "event_tape_target_id": torch.empty((0,), dtype=torch.int64),
        "event_tape_value": torch.empty((0,), dtype=torch.float32),
        "drives": _tensor_from_numpy(state.drives),
        "policy_attack_bias": _tensor_from_numpy(state.policy_attack_bias),
        "policy_action_temperature": _tensor_from_numpy(state.policy_action_temperature),
        "policy_drive_logit_gain": _tensor_from_numpy(state.policy_drive_logit_gain),
        "policy_use_gru": _tensor_from_numpy(state.policy_use_gru),
        "policy_mem_dim": _policy_mem_dim_by_slot(runner, int(state.capacity)),
        "action_cache_token": _tensor_from_numpy(state.action_cache_token),
        "action_cache_cursor": _tensor_from_numpy(state.action_cache_cursor),
        "action_cache_next": _tensor_from_numpy(state.action_cache_next),
        "action_cache_actions": torch.full(
            (int(state.capacity), max(1, int(getattr(sim_cfg, "policy_tick_stride", 1) or 1))),
            -1,
            dtype=torch.int32,
        ),
        "hmem": _hmem_tensor_for_projection(runner, int(state.capacity)),
        "mlp_pack": _required_tensor(getattr(runner, "_mlp_pack", None), "mlp_pack"),
        "gru_pack": _gru_pack_tensor_for_projection(runner, int(state.capacity)),
        "agent_grid": _tensor_from_numpy(np.asarray(getattr(env, "agent_grid"), dtype=np.int32)),
        "food_grid": _tensor_from_numpy(np.asarray(getattr(env, "grid"), dtype=np.int32)),
    }
    if metadata["inheritance_profile"] == DIPLOID_INHERITANCE_PROFILE:
        from evolution.canonical_hereditary_digest_template_v1 import (  # noqa: WPS433
            common_canonical_hereditary_digest_template_v1,
        )
        from evolution.inheritance_registry_v1 import (  # noqa: WPS433
            canonical_hereditary_payload_sha256,
        )

        allele_storage = np.zeros(
            (2 * len(DIPLOID_SCALAR_LOCI), int(state.capacity)),
            dtype=np.float32,
        )
        resource_values = np.zeros((int(state.capacity),), dtype=np.float32)
        canonical_digests = np.zeros(
            (int(state.capacity), 32),
            dtype=np.uint8,
        )
        alive_agents = [
            agent
            for agent in _alive_agents_from_runner(runner)
            if bool(getattr(agent, "alive", True))
        ]
        for agent in alive_agents:
            genome = getattr(agent, "genome", None)
            if getattr(genome, "_diploid_scalar_genome_v1", None) is None:
                attach_diploid_genome(
                    genome,
                    diploid_genome_from_genome(genome),
                )
        digest_template = common_canonical_hereditary_digest_template_v1(
            [getattr(agent, "genome", None) for agent in alive_agents]
        )
        for agent in alive_agents:
            if not bool(getattr(agent, "alive", True)):
                continue
            slot = int(getattr(agent, "slot", -1))
            if slot < 0 or slot >= int(state.capacity):
                continue
            diploid = diploid_genome_from_genome(
                getattr(agent, "genome", None)
            )
            allele_storage[: len(DIPLOID_SCALAR_LOCI), slot] = np.asarray(
                diploid.allele_0,
                dtype=np.float32,
            )
            allele_storage[len(DIPLOID_SCALAR_LOCI) :, slot] = np.asarray(
                diploid.allele_1,
                dtype=np.float32,
            )
            resource_values[slot] = np.float32(
                diploid.phenotype()["resource_handling_axis"]
            )
            canonical_digests[slot] = np.frombuffer(
                bytes.fromhex(
                    canonical_hereditary_payload_sha256(
                        getattr(agent, "genome", None)
                    )
                ),
                dtype=np.uint8,
            )
        resource_axis = _tensor_from_numpy(resource_values)
        tensors["genome_resource_handling_axis"] = resource_axis
        tensors["diploid_allele_storage"] = _tensor_from_numpy(allele_storage)
        tensors["canonical_hereditary_digest"] = _tensor_from_numpy(
            canonical_digests
        )
        tensors["canonical_hereditary_template"] = torch.tensor(
            list(digest_template.canonical_template),
            dtype=torch.uint8,
        )
        tensors["diploid_payload_template"] = torch.tensor(
            list(digest_template.diploid_payload_template),
            dtype=torch.uint8,
        )
        tensors["diploid_genotype_template"] = torch.tensor(
            list(digest_template.genotype_hash_template),
            dtype=torch.uint8,
        )
    if hasattr(env, "biome_grid"):
        tensors["biome_grid"] = _tensor_from_numpy(np.asarray(getattr(env, "biome_grid"), dtype=np.int32))
    if hasattr(env, "hazard_rate_map"):
        tensors["hazard_rate_map"] = _tensor_from_numpy(np.asarray(getattr(env, "hazard_rate_map"), dtype=np.float32))
    if str(metadata.get("native_runtime_fixed_policy_family", "") or "").strip().lower() == "mlp":
        tensors["hmem"] = torch.zeros_like(tensors["hmem"])
        tensors["gru_pack"] = torch.zeros_like(tensors["gru_pack"])
    typed_lifecycle_events = _typed_lifecycle_event_tensors_for_runner(runner)
    if typed_lifecycle_events:
        tensors.update(typed_lifecycle_events)
        tensors.update(_legacy_event_tape_from_prefixed_typed_events(typed_lifecycle_events))
        metadata["typed_lifecycle_event_schema_version"] = "eiel_typed_lifecycle_event_v1"
        metadata["typed_lifecycle_event_active_count"] = int(
            typed_lifecycle_events["typed_lifecycle_event_event_type"].shape[0]
        )
        reference_death_count = len(
            list(getattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, []) or [])
        )
        reference_accumulator = getattr(runner, REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR, None)
        if isinstance(reference_accumulator, Mapping):
            metadata["typed_lifecycle_event_source"] = "reference_cumulative_lifecycle_accumulator"
        elif reference_death_count and bool(
            getattr(runner, "_lineage_reproduction_funnel_event_contexts_gen", []) or []
        ):
            metadata["typed_lifecycle_event_source"] = "reference_reproduction_and_committed_death_contexts"
        elif reference_death_count:
            metadata["typed_lifecycle_event_source"] = "reference_committed_death_oracle"
        else:
            metadata["typed_lifecycle_event_source"] = "reference_lineage_reproduction_funnel_contexts"
    if rng_manager is not None:
        try:
            streams = rng_manager_snapshot.get("streams", {}) if isinstance(rng_manager_snapshot, dict) else {}
            mutation_stream = streams.get("mutation", {}) if isinstance(streams, dict) else {}
            mutation_states = mutation_stream.get("states", {}) if isinstance(mutation_stream, dict) else {}
            mutation_torch_cpu_hex = str(mutation_states.get("torch_cpu_state", "") or "")
            if mutation_torch_cpu_hex:
                tensors["rng_manager_mutation_torch_cpu_state"] = torch.tensor(
                    list(bytes.fromhex(mutation_torch_cpu_hex)),
                    dtype=torch.uint8,
                )
        except Exception:
            pass
    if runnercore_state is not None:
        for name in (
            "chemistry",
            "gru_upkeep_cost",
            "cost_charge_scalar",
            "cost_charge_maturation_scalar",
            "repro_cooldown",
            "repro_threshold",
            "rest_streak",
            "survival_steps",
            "energy_collected",
            "early_survival_steps",
            "early_energy_collected",
            "last_energy_delta",
            "last_health_delta",
            "last_injury_delta",
            "drive_inertia",
            "drive_noise",
            "drive_volatility",
            "drive_jitter",
            "episode0_drive_jitter_sum",
            "episode0_selfharm_events",
            "prev_drive_delta",
            "pred_lr",
            "pred_err_sum",
            "pred_err_count",
            "prediction_update_count",
            "prediction_state_delta_abs_sum",
            "prediction_state_delta_abs_max",
            "prediction_reset_count",
            "episode0_pred_err_sum",
            "episode0_pred_err_count",
            "late_pred_err_sum",
            "late_pred_err_count",
            "adult_pred_err_ema",
            "exp_dE",
            "exp_dInj",
        ):
            value = getattr(runnercore_state, name, None)
            if value is not None:
                tensors[name] = _tensor_from_numpy(np.asarray(value))
        tensors.update(_prediction_queue_tensors(runnercore_state, int(state.capacity)))
        tensors["action_cache_actions"] = _action_cache_actions_tensor_for_runner(
            runner,
            runnercore_state,
            max(1, int(getattr(sim_cfg, "policy_tick_stride", 1) or 1)),
        )
        drive_noise = np.asarray(getattr(runnercore_state, "drive_noise", np.zeros((int(state.capacity),), dtype=np.float32)), dtype=np.float32)
        alive_drive_noise = drive_noise[np.asarray(state.alive_mask, dtype=np.bool_)]
        if bool(metadata["native_runtime_use_precomputed_drive_noise"]):
            configured_steps = int(getattr(sim_cfg, "steps_per_generation", 0) or 0)
            start_tick = max(0, int(tick))
            steps = max(configured_steps, start_tick + configured_steps, configured_steps * 2)
            by_slot = np.zeros((int(steps), int(state.capacity), 4), dtype=np.float32)
            valid_by_slot = np.zeros((int(steps), int(state.capacity)), dtype=np.bool_)
            ids0 = [
                int(aid)
                for aid in np.asarray(getattr(runnercore_state, "slot_to_agent_id"), dtype=np.int64).tolist()
                if int(aid) >= 0
            ]
            ids0.sort()
            slot_for_aid = {
                int(aid): int(slot)
                for slot, aid in enumerate(np.asarray(getattr(runnercore_state, "slot_to_agent_id"), dtype=np.int64).tolist())
                if int(aid) >= 0
            }
            if bool(np.any(alive_drive_noise > np.float32(0.0))):
                rng_state = None
                rng_indices_state = None
                try:
                    rng_state = copy.deepcopy(runner.np_rng.bit_generator.state)
                except Exception:
                    rng_state = None
                try:
                    indices_ref = getattr(rng_manager, "_indices", None)
                    if isinstance(indices_ref, dict):
                        rng_indices_state = dict(indices_ref)
                except Exception:
                    rng_indices_state = None
                try:
                    # Native drive updates index this table by absolute generation tick.
                    # On resume, the next sampled row must therefore land at start_tick.
                    for step in range(start_tick, steps):
                        ids_list = list(ids0)
                        runner.np_rng.shuffle(ids_list)
                        samples = runner.np_rng.normal(0.0, 1.0, size=(len(ids_list), 4)).astype(np.float32, copy=False)
                        for row, aid in enumerate(ids_list):
                            slot = slot_for_aid.get(int(aid), -1)
                            if slot >= 0:
                                by_slot[step, slot, :] = samples[row, :]
                                valid_by_slot[step, slot] = True
                finally:
                    if rng_state is not None:
                        try:
                            runner.np_rng.bit_generator.state = rng_state
                        except Exception:
                            pass
                    if rng_indices_state is not None:
                        indices_ref = getattr(rng_manager, "_indices", None)
                        if isinstance(indices_ref, dict):
                            indices_ref.clear()
                            indices_ref.update(rng_indices_state)
            tensors["drive_noise_samples_by_tick_slot"] = _tensor_from_numpy(by_slot)
            tensors["drive_noise_samples_valid_by_tick_slot"] = _tensor_from_numpy(valid_by_slot)
    if world == "patch":
        tensors.update(patch_contract_tensors)
    else:
        for name in (
            "patch_food_amount",
            "patch_fertility",
            "patch_cover",
            "patch_shelter",
            "patch_movement_cost",
            "patch_scent",
            "patch_scent_emit_pending",
            "patch_waste",
            "patch_waste_emit_pending",
            "patch_density",
            "patch_recovery_activity",
            "patch_recovery",
            "patch_delayed_hazard_cue",
            "patch_delayed_hazard_active",
            "patch_delayed_hazard_timer",
            "patch_delayed_hazard_age",
            "patch_delayed_hazard_event_id",
            "patch_delayed_hazard_intensity",
            "patch_delayed_hazard_prior_cue",
            "mem_max_delayed_choice_target_a",
            "mem_max_delayed_choice_target_b",
            "mem_max_delayed_choice_cue_a",
            "mem_max_delayed_choice_cue_b",
            "mem_max_delayed_choice_active",
            "mem_max_delayed_choice_prior_cue",
            "mem_max_delayed_choice_correct_target",
            "mem_max_delayed_choice_event_id",
        ):
            value = getattr(env, name, None)
            if value is not None:
                tensors[name] = _tensor_from_numpy(np.asarray(value))
    if modular_projection is not None:
        tensors.update(
            {
                str(name): value.detach().clone()
                for name, value in modular_projection["tensors"].items()
            }
        )
    overrides = getattr(runner, "_cxxbackbone10_checkpoint_tensor_overrides", None)
    if isinstance(overrides, Mapping):
        variable_length_override_names = set(EVENT_TAPE_TENSOR_NAMES)
        checkpoint_override_same_boundary = not isinstance(metadata_overrides, Mapping)
        if isinstance(metadata_overrides, Mapping):
            try:
                checkpoint_override_same_boundary = (
                    int(metadata_overrides.get("generation", -1)) == int(generation)
                    and int(metadata_overrides.get("tick", -1)) == int(tick)
                )
            except Exception:
                checkpoint_override_same_boundary = False
        boundary_legacy_rows = 0
        if all(hasattr(overrides.get(name), "detach") for name in variable_length_override_names):
            boundary_legacy_rows = min(int(overrides[name].shape[0]) for name in variable_length_override_names)
        for name, value in dict(overrides).items():
            if not hasattr(value, "detach"):
                continue
            lifecycle_override = name in variable_length_override_names or name.startswith(
                "typed_lifecycle_event_"
            )
            if not checkpoint_override_same_boundary and not lifecycle_override:
                continue
            if name not in tensors:
                if name.startswith("typed_lifecycle_event_"):
                    tensors[name] = value.detach().cpu().contiguous()
                continue
            candidate = value.detach().to(tensors[name].device).contiguous()
            if tuple(candidate.shape) == tuple(tensors[name].shape) and candidate.dtype == tensors[name].dtype:
                tensors[name] = candidate
            elif (
                name in variable_length_override_names
                and candidate.dtype == tensors[name].dtype
                and int(candidate.dim()) == int(tensors[name].dim())
            ):
                tensors[name] = candidate
        if typed_lifecycle_events:
            merged_typed = _merge_prefixed_typed_lifecycle_events(overrides, typed_lifecycle_events)
            tensors.update(merged_typed)
            boundary_event_type = overrides.get("typed_lifecycle_event_event_type")
            boundary_typed_rows = (
                int(boundary_event_type.shape[0])
                if hasattr(boundary_event_type, "shape")
                else 0
            )
            if boundary_typed_rows:
                metadata["typed_lifecycle_event_schema_version"] = "eiel_typed_lifecycle_event_v1"
                metadata["typed_lifecycle_event_active_count"] = int(
                    tensors["typed_lifecycle_event_event_type"].shape[0]
                )
                metadata["typed_lifecycle_event_source"] = "native_checkpoint_boundary_plus_reference_contexts"
            elif boundary_legacy_rows <= 0:
                metadata["typed_lifecycle_event_active_count"] = int(
                    tensors["typed_lifecycle_event_event_type"].shape[0]
                )
                metadata["typed_lifecycle_event_source"] = "reference_contexts_after_empty_checkpoint_boundary"
            if boundary_legacy_rows > 0:
                local_legacy = _legacy_event_tape_from_prefixed_typed_events(typed_lifecycle_events)
                legacy_dtypes = {
                    "event_tape_type": torch.int32,
                    "event_tape_generation": torch.int64,
                    "event_tape_tick": torch.int64,
                    "event_tape_actor_id": torch.int64,
                    "event_tape_target_id": torch.int64,
                    "event_tape_value": torch.float32,
                }
                for name, dtype in legacy_dtypes.items():
                    tensors[name] = _cat_checkpoint_tensor_prefix(
                        overrides[name],
                        local_legacy[name],
                        dtype=dtype,
                    )
            else:
                tensors.update(_legacy_event_tape_from_prefixed_typed_events(typed_lifecycle_events))
    alive_mask = tensors.get("alive_mask")
    slot_to_agent_id = tensors.get("slot_to_agent_id")
    slot_tokens = tensors.get("slot_tokens")
    lineage_depth = tensors.get("lineage_depth")
    if (
        hasattr(alive_mask, "detach")
        and hasattr(slot_to_agent_id, "detach")
        and hasattr(slot_tokens, "detach")
        and hasattr(lineage_depth, "detach")
    ):
        active_rows = alive_mask.detach().cpu().to(torch.bool) & slot_to_agent_id.detach().cpu().ge(0)
        born_rows = active_rows & lineage_depth.detach().cpu().to(torch.int64).gt(0)
        zero_tokens = slot_tokens.detach().cpu().to(torch.int64).le(0)
        token_fix_rows = born_rows & zero_tokens
        if bool(torch.any(token_fix_rows)):
            normalized = slot_tokens.clone()
            normalized[token_fix_rows.to(normalized.device)] = 1
            tensors["slot_tokens"] = normalized
    if hasattr(alive_mask, "detach") and hasattr(slot_to_agent_id, "detach"):
        active_rows = alive_mask.detach().cpu().to(torch.bool) & slot_to_agent_id.detach().cpu().ge(0)
        inactive_rows = ~active_rows
        if bool(torch.any(inactive_rows)):
            for name, fill_value in (
                ("hmem", 0.0),
                ("gru_pack", 0.0),
                ("policy_use_gru", False),
                ("policy_mem_dim", 0),
                ("genome_memory_dim", 0),
                ("drive_inertia", 0.85),
                ("genome_drive_inertia", 0.85),
                ("genome_directed_inheritance_channel_mask", 0),
                ("genome_introspection_channel_mask", 0),
                ("genome_mutation_rate", 0.0),
                ("genome_reproduction_threshold", 30.0),
                ("genome_self_modification_target_mask", 0),
                ("genome_self_regulation_control_mask", 0),
                ("genome_sensor_channel_mask", 0),
                ("genome_sensor_radius", 1),
            ):
                tensor = tensors.get(name)
                if hasattr(tensor, "detach") and tensor.dim() >= 1 and int(tensor.shape[0]) == int(inactive_rows.shape[0]):
                    normalized = tensor.clone()
                    normalized[inactive_rows.to(normalized.device)] = fill_value
                    tensors[name] = normalized
    _refresh_event_tape_metadata(metadata, tensors)
    # Projection-only precomputation must not become an RNG draw. Recapture at
    # the sealed boundary so serialized indices describe the live source state.
    rng_manager_snapshot = capture_rng_snapshot(generation_index=int(generation))
    metadata[RNG_MANAGER_STREAM_INDICES_METADATA] = _rng_manager_stream_indices_from_snapshot(
        rng_manager_snapshot
    )
    streams = rng_manager_snapshot.get("streams", {}) if isinstance(rng_manager_snapshot, Mapping) else {}
    mutation_stream = streams.get("mutation", {}) if isinstance(streams, Mapping) else {}
    mutation_states = mutation_stream.get("states", {}) if isinstance(mutation_stream, Mapping) else {}
    mutation_torch_cpu_hex = str(mutation_states.get("torch_cpu_state", "") or "")
    if mutation_torch_cpu_hex:
        tensors["rng_manager_mutation_torch_cpu_state"] = torch.tensor(
            list(bytes.fromhex(mutation_torch_cpu_hex)),
            dtype=torch.uint8,
        )
    _validate_patch_projection_contract(
        metadata,
        tensors,
        surface="resident owner import",
    )
    _attach_policy_energy_observation_contract_metadata(metadata, tensors)
    _validate_passive_modular_storage_projection(
        metadata,
        tensors,
        surface="resident owner import",
    )
    metadata["import_digest"] = projection_digest({"metadata": metadata, "tensors": tensors})
    pending_generation_frees = _pending_generation_free_pairs_for_runner(
        runner,
        capacity=int(state.capacity),
        slot_to_agent_id=tensors["slot_to_agent_id"],
    )
    return ImportProjectionV10(
        schema_version=SCHEMA_VERSION,
        metadata=metadata,
        tensors=tensors,
        pending_generation_free_agent_ids=tuple(agent_id for agent_id, _slot in pending_generation_frees),
        pending_generation_free_slots=tuple(slot for _agent_id, slot in pending_generation_frees),
    )


def execute_flat_owner_batch(
    runners: list[Any],
    capability_mask: Mapping[str, bool],
    *,
    generation_index: int,
    steps_per_generation: int,
    build_directory: str | Path | None = None,
) -> dict[str, Any]:
    owners = [
        ResidentRuntimeState.from_runner(
            runner,
            capability_mask,
            generation=int(generation_index),
            tick=0,
            build_directory=build_directory,
        )
        for runner in runners
    ]
    mod = load_resident_state_owner_extension(build_directory=build_directory)
    native_batch = dict(
        mod.execute_native_batch_flat_with_population_birth_commit_cpu(
            [owner._native_owner for owner in owners],
            int(generation_index),
            int(steps_per_generation),
            1,
        )
    )
    results = [dict(result) for result in native_batch["results"]]
    return {
        "schema_version": "cxxbackbone11_flat_owner_multicell_batch_v1",
        "cell_count": len(owners),
        "python_tick_iterations": 0,
        "python_cell_loop": bool(native_batch.get("python_cell_loop", True)),
        "native_batch_owner": bool(native_batch.get("native_batch_owner", False)),
        "native_generation_calls": len(results),
        "hidden_fallback": any(bool(result.get("hidden_fallback", True)) for result in results),
        "applied_actions": int(sum(int(result.get("applied_actions", 0)) for result in results)),
        "policy_loop_tick_calls": int(sum(int(result.get("policy_loop_tick_calls", 0)) for result in results)),
        "policy_refill_calls": int(sum(int(result.get("policy_refill_calls", 0)) for result in results)),
        "policy_cache_hit_calls": int(sum(int(result.get("policy_cache_hit_calls", 0)) for result in results)),
        "committed_birth_rows": int(native_batch.get("committed_birth_rows", 0)),
        "typed_event_rows": int(native_batch.get("typed_event_rows", 0)),
        "results": results,
    }


def _sync_checkpoint_lineage_identity(
    stats: Any,
    founder_id: int,
    *,
    metadata: Mapping[str, Any],
) -> bool:
    founder_id_i = int(founder_id)
    stats.founder_id = founder_id_i
    world = str(metadata.get("world", "patch" if bool(metadata.get("patch")) else "flat") or "flat")
    if world != "flat" or bool(metadata.get("reproduce_requires_mate", False)) or founder_id_i < 0:
        return False
    from agents.agent import format_lineage_id  # noqa: WPS433

    stats.lineage_id = format_lineage_id(founder_id_i)
    return True


@dataclass(frozen=True)
class _ReferenceMaterializationPreflight:
    capacity: int
    obs_size: int
    hidden: int
    gru_hidden: int
    action_count: int
    policy_stride: int
    live_slot_agent_ids: tuple[tuple[int, int], ...]
    pending_slot_agent_ids: tuple[tuple[int, int], ...]
    free_slots: tuple[int, ...]
    rng_manager: Any
    policy_np: Any
    env_policy_alias: bool
    env_rng_state: Mapping[str, Any]
    runner_rng_state: Mapping[str, Any]
    policy_rng_state: Mapping[str, Any]
    population_np_rng_state: Mapping[str, Any]
    population_random_state: tuple[int, tuple[int, ...], float | None]
    rng_indices: dict[str, int]
    mutation_torch_cpu_rng: Any | None
    mutation_torch_cpu_state: torch.Tensor | None
    modular_records_by_slot: tuple[dict[str, Any] | None, ...] | None


def _strict_positive_metadata_int(metadata: Mapping[str, Any], name: str) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"native-to-reference materialization requires integer metadata field={name}")
    value_i = int(value)
    if value_i <= 0:
        raise RuntimeError(f"native-to-reference materialization requires positive metadata field={name}")
    return value_i


def _checkpoint_cpu_tensor(tensors: Mapping[str, Any], name: str) -> torch.Tensor:
    value = tensors.get(name)
    if value is None or not hasattr(value, "detach"):
        raise RuntimeError(f"native-to-reference materialization requires tensor field={name}")
    return value.detach().cpu().contiguous()


def _require_checkpoint_tensor_shape(
    tensors: Mapping[str, Any],
    name: str,
    expected: tuple[int, ...],
) -> torch.Tensor:
    tensor = _checkpoint_cpu_tensor(tensors, name)
    actual = tuple(int(value) for value in tensor.shape)
    if actual != expected:
        raise RuntimeError(
            "native-to-reference materialization tensor shape mismatch: "
            f"field={name} expected={expected} actual={actual}"
        )
    return tensor


def _validate_numpy_rng_state(state: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise RuntimeError(f"native-to-reference materialization requires RNG metadata field={field}")
    bit_generator_name = str(state.get("bit_generator", "") or "")
    bit_generator_type = getattr(np.random, bit_generator_name, None)
    if not callable(bit_generator_type):
        raise RuntimeError(
            "native-to-reference materialization has unsupported NumPy RNG state: "
            f"field={field} bit_generator={bit_generator_name!r}"
        )
    try:
        bit_generator = bit_generator_type()
        bit_generator.state = copy.deepcopy(dict(state))
    except Exception as exc:
        raise RuntimeError(
            f"native-to-reference materialization has invalid NumPy RNG state: field={field}"
        ) from exc
    return state


def _require_numpy_rng_state_target(value: Any, *, label: str) -> None:
    if callable(getattr(value, "_set_bitgen_state", None)):
        return
    bit_generator = getattr(value, "bit_generator", None)
    if bit_generator is None or not hasattr(bit_generator, "state"):
        raise RuntimeError(f"native-to-reference materialization cannot restore RNG target={label}")


def _normalized_rng_manager_indices(
    metadata: Mapping[str, Any],
    rng_manager: Any,
) -> dict[str, int]:
    raw_streams = metadata.get(RNG_MANAGER_STREAM_INDICES_METADATA)
    if not isinstance(raw_streams, Mapping):
        raise RuntimeError(
            "native-to-reference materialization is missing exact RNGManager stream indices; "
            f"producer must serialize metadata field={RNG_MANAGER_STREAM_INDICES_METADATA}"
        )
    unexpected_streams = sorted(set(str(name) for name in raw_streams) - set(RNG_MANAGER_STREAM_NAMES))
    if unexpected_streams:
        raise RuntimeError(
            "native-to-reference materialization has unsupported RNGManager streams: "
            f"{unexpected_streams}"
        )
    capture_snapshot = getattr(rng_manager, "capture_snapshot", None)
    if not callable(capture_snapshot):
        raise RuntimeError("native-to-reference materialization requires RNGManager snapshot support")
    destination_indices = _rng_manager_stream_indices_from_snapshot(capture_snapshot(generation_index=None))
    flat: dict[str, int] = {}
    for stream_name in RNG_MANAGER_STREAM_NAMES:
        indices = raw_streams.get(stream_name)
        if not isinstance(indices, Mapping):
            raise RuntimeError(
                "native-to-reference materialization is missing exact RNGManager stream indices: "
                f"stream={stream_name}"
            )
        for domain in ("py", "np", "torch_cpu"):
            raw_value = indices.get(domain)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, np.integer)):
                raise RuntimeError(
                    "native-to-reference materialization has invalid RNGManager stream index: "
                    f"stream={stream_name} domain={domain}"
                )
            value = int(raw_value)
            if value < 0:
                raise RuntimeError(
                    "native-to-reference materialization has negative RNGManager stream index: "
                    f"stream={stream_name} domain={domain}"
                )
            flat[f"{stream_name}.{domain}"] = value
        torch_cuda = indices.get("torch_cuda")
        if not isinstance(torch_cuda, Mapping):
            raise RuntimeError(
                "native-to-reference materialization has invalid CUDA RNGManager indices: "
                f"stream={stream_name}"
            )
        expected_cuda = set(destination_indices[stream_name]["torch_cuda"])
        actual_cuda = set(str(device) for device in torch_cuda)
        if actual_cuda != expected_cuda:
            raise RuntimeError(
                "native-to-reference materialization RNGManager CUDA device mismatch: "
                f"stream={stream_name} expected={sorted(expected_cuda)} actual={sorted(actual_cuda)}"
            )
        for device, raw_value in torch_cuda.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, np.integer)):
                raise RuntimeError(
                    "native-to-reference materialization has invalid CUDA RNGManager stream index: "
                    f"stream={stream_name} device={device}"
                )
            value = int(raw_value)
            if value < 0:
                raise RuntimeError(
                    "native-to-reference materialization has negative CUDA RNGManager stream index: "
                    f"stream={stream_name} device={device}"
                )
            flat[f"{stream_name}.torch_cuda.{device}"] = value
    indices_ref = getattr(rng_manager, "_indices", None)
    if not isinstance(indices_ref, dict):
        raise RuntimeError("native-to-reference materialization cannot install RNGManager stream indices")
    return flat


def _preflight_reference_materialization(
    runner: Any,
    checkpoint_projection: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> _ReferenceMaterializationPreflight:
    env = getattr(runner, "env", None)
    pop = getattr(runner, "pop", None)
    slots = getattr(pop, "_slots", None)
    if env is None or pop is None or slots is None:
        raise RuntimeError("native-to-reference materialization requires runner env/pop/SlotPool")

    capacity = _strict_positive_metadata_int(metadata, "capacity")
    obs_size = _strict_positive_metadata_int(metadata, "obs_size")
    hidden = _strict_positive_metadata_int(metadata, "hidden")
    gru_hidden = _strict_positive_metadata_int(metadata, "gru_hidden")
    action_count = _strict_positive_metadata_int(metadata, "action_count")
    policy_stride = _strict_positive_metadata_int(metadata, "policy_tick_stride")
    destination_dimensions = {
        "capacity": int(getattr(pop, "max_slots", 0) or 0),
        "slot_capacity": int(getattr(slots, "max_slots", 0) or 0),
        "obs_size": int(getattr(runner, "_obs_size", 0) or 0),
        "population_obs_size": int(getattr(pop, "obs_size", 0) or 0),
        "hidden": int(getattr(runner, "_hidden", 0) or 0),
        "gru_hidden": int(getattr(runner, "_gru_hidden", getattr(runner, "_hidden", 0)) or 0),
        "action_count": int(getattr(runner, "_out", 0) or 0),
        "policy_tick_stride": int(getattr(getattr(runner, "sim_cfg", None), "policy_tick_stride", 0) or 0),
    }
    expected_dimensions = {
        "capacity": capacity,
        "slot_capacity": capacity,
        "obs_size": obs_size,
        "population_obs_size": obs_size,
        "hidden": hidden,
        "gru_hidden": gru_hidden,
        "action_count": action_count,
        "policy_tick_stride": policy_stride,
    }
    for name, expected in expected_dimensions.items():
        actual = destination_dimensions[name]
        if actual != expected:
            raise RuntimeError(
                "native-to-reference materialization destination dimension mismatch: "
                f"field={name} expected={expected} actual={actual}"
            )

    modular_storage = _validate_passive_modular_storage_projection(
        metadata,
        tensors,
        surface="native-to-reference materialization preflight",
    )
    genome_contract = metadata.get(REFERENCE_GENOME_MATERIALIZATION_METADATA)
    if not isinstance(genome_contract, Mapping):
        raise RuntimeError(
            "native-to-reference materialization is missing the total-genome producer attestation; "
            f"producer must serialize metadata field={REFERENCE_GENOME_MATERIALIZATION_METADATA}"
        )
    if str(genome_contract.get("schema_version", "") or "") != REFERENCE_GENOME_MATERIALIZATION_SCHEMA:
        raise RuntimeError("native-to-reference materialization has an unsupported total-genome contract")
    genome_mode = str(genome_contract.get("mode", "") or "")
    if genome_mode == REFERENCE_GENOME_SCALAR_MODE:
        if modular_storage is not None:
            raise RuntimeError(
                "native-to-reference materialization scalar contract contains modular authority"
            )
    elif genome_mode == REFERENCE_GENOME_MODULAR_MODE:
        if modular_storage is None:
            raise RuntimeError(
                "native-to-reference materialization modular contract is missing modular authority"
            )
        if str(genome_contract.get("profile_id", "") or "") != modular_storage.layout.profile_id:
            raise RuntimeError(
                "native-to-reference materialization modular profile contradicts its sidecar"
            )
    else:
        raise RuntimeError("native-to-reference materialization has an unsupported total-genome contract")
    if genome_contract.get("supported") is not True:
        raise RuntimeError(
            "native-to-reference materialization cannot reconstruct the complete canonical genome payload; "
            "producer must serialize per-slot modular/B2/B3/C1/C2/resource/modules/connections state: "
            f"reasons={list(genome_contract.get('reasons') or [])} "
            f"agent_ids={list(genome_contract.get('unsupported_agent_ids') or [])}"
        )

    vector_names = (
        "slot_to_agent_id",
        "slot_tokens",
        "alive_mask",
        "pos_y",
        "pos_x",
        "energy",
        "health",
        "injury",
        "policy_use_gru",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    )
    for name in vector_names:
        _require_checkpoint_tensor_shape(tensors, name, (capacity,))
    mlp_width = (hidden * obs_size) + hidden + (hidden * hidden) + hidden + (action_count * hidden) + action_count
    gru_width = (
        (gru_hidden * obs_size)
        + gru_hidden
        + (3 * gru_hidden * gru_hidden)
        + (3 * gru_hidden)
        + (3 * gru_hidden * gru_hidden)
        + (3 * gru_hidden)
        + (action_count * gru_hidden)
        + action_count
    )
    _require_checkpoint_tensor_shape(tensors, "mlp_pack", (capacity, mlp_width))
    _require_checkpoint_tensor_shape(tensors, "gru_pack", (capacity, gru_width))
    _require_checkpoint_tensor_shape(tensors, "hmem", (capacity, gru_hidden))
    _require_checkpoint_tensor_shape(tensors, "action_cache_actions", (capacity, policy_stride))

    slot_to_agent_id = _checkpoint_cpu_tensor(tensors, "slot_to_agent_id").to(dtype=torch.int64).numpy()
    alive = _checkpoint_cpu_tensor(tensors, "alive_mask").to(dtype=torch.bool).numpy()
    assigned = slot_to_agent_id >= 0
    if not np.array_equal(assigned, alive):
        contradictory_slots = np.flatnonzero(assigned != alive).astype(int).tolist()
        raise RuntimeError(
            "native-to-reference materialization slot partition contradicts alive_mask: "
            f"slots={contradictory_slots}"
        )
    live_slot_agent_ids = tuple(
        (int(slot), int(slot_to_agent_id[slot]))
        for slot in np.flatnonzero(alive).astype(int).tolist()
    )
    live_agent_ids = [agent_id for _slot, agent_id in live_slot_agent_ids]
    if len(live_agent_ids) != len(set(live_agent_ids)):
        raise RuntimeError("native-to-reference materialization has duplicate live agent IDs")
    alive_count = metadata.get("alive_count")
    if isinstance(alive_count, bool) or not isinstance(alive_count, (int, np.integer)):
        raise RuntimeError("native-to-reference materialization requires integer metadata field=alive_count")
    if int(alive_count) != len(live_slot_agent_ids):
        raise RuntimeError(
            "native-to-reference materialization alive_count mismatch: "
            f"metadata={int(alive_count)} slots={len(live_slot_agent_ids)}"
        )

    free_tensor = _checkpoint_cpu_tensor(tensors, "slot_free_queue")
    if int(free_tensor.dim()) != 1:
        raise RuntimeError("native-to-reference materialization slot_free_queue must be one-dimensional")
    free_slots = tuple(int(value) for value in free_tensor.to(dtype=torch.int64).tolist())
    if len(free_slots) != len(set(free_slots)):
        raise RuntimeError("native-to-reference materialization has duplicate free slots")
    out_of_range_free_slots = [slot for slot in free_slots if slot < 0 or slot >= capacity]
    if out_of_range_free_slots:
        raise RuntimeError(
            "native-to-reference materialization has out-of-range free slots: "
            f"slots={out_of_range_free_slots} capacity={capacity}"
        )
    live_in_free_slots = [slot for slot in free_slots if bool(alive[slot]) or int(slot_to_agent_id[slot]) >= 0]
    if live_in_free_slots:
        raise RuntimeError(
            "native-to-reference materialization has live-in-free slots: "
            f"slots={live_in_free_slots}"
        )
    pending_agent_ids = checkpoint_projection.get("pending_generation_free_agent_ids", ())
    pending_slots = checkpoint_projection.get("pending_generation_free_slots", ())
    if not isinstance(pending_agent_ids, (list, tuple)) or not isinstance(pending_slots, (list, tuple)):
        raise RuntimeError("native-to-reference materialization has malformed pending-free state")
    if len(pending_agent_ids) != len(pending_slots):
        raise RuntimeError("native-to-reference materialization has contradictory pending-free state")
    pending_slot_agent_ids_list: list[tuple[int, int]] = []
    for raw_agent_id, raw_slot in zip(pending_agent_ids, pending_slots, strict=True):
        if (
            isinstance(raw_agent_id, bool)
            or not isinstance(raw_agent_id, (int, np.integer))
            or isinstance(raw_slot, bool)
            or not isinstance(raw_slot, (int, np.integer))
        ):
            raise RuntimeError("native-to-reference materialization has malformed pending-free state")
        agent_id = int(raw_agent_id)
        slot = int(raw_slot)
        if agent_id < 0 or slot < 0 or slot >= capacity:
            raise RuntimeError(
                "native-to-reference materialization has out-of-range pending-free state: "
                f"agent_id={agent_id} slot={slot} capacity={capacity}"
            )
        pending_slot_agent_ids_list.append((slot, agent_id))
    pending_slot_agent_ids = tuple(pending_slot_agent_ids_list)
    if modular_storage is not None and pending_slot_agent_ids:
        raise RuntimeError(
            f"{PASSIVE_MODULAR_LIFECYCLE_ERROR}: modular checkpoint has pending-free agents"
        )
    pending_slot_values = [slot for slot, _agent_id in pending_slot_agent_ids]
    pending_agent_id_values = [agent_id for _slot, agent_id in pending_slot_agent_ids]
    if len(pending_slot_values) != len(set(pending_slot_values)):
        raise RuntimeError("native-to-reference materialization has duplicate pending-free slots")
    if len(pending_agent_id_values) != len(set(pending_agent_id_values)):
        raise RuntimeError("native-to-reference materialization has duplicate pending-free agent IDs")
    live_slots = {slot for slot, _agent_id in live_slot_agent_ids}
    live_agent_id_set = set(live_agent_ids)
    pending_live_slots = sorted(set(pending_slot_values) & live_slots)
    pending_live_agent_ids = sorted(set(pending_agent_id_values) & live_agent_id_set)
    if pending_live_slots or pending_live_agent_ids:
        raise RuntimeError(
            "native-to-reference materialization pending-free state overlaps live state: "
            f"agent_ids={pending_live_agent_ids} slots={pending_live_slots}"
        )
    actual_free_slots = set(free_slots)
    pending_in_free_slots = sorted(set(pending_slot_values) & actual_free_slots)
    if pending_in_free_slots:
        raise RuntimeError(
            "native-to-reference materialization has pending slots also present in free queue: "
            f"slots={pending_in_free_slots}"
        )
    expected_free_slots = set(range(capacity)) - live_slots - set(pending_slot_values)
    if actual_free_slots != expected_free_slots:
        raise RuntimeError(
            "native-to-reference materialization slot partition is incomplete: "
            f"missing_free={sorted(expected_free_slots - actual_free_slots)} "
            f"unexpected_free={sorted(actual_free_slots - expected_free_slots)}"
        )

    expected_by_agent_id = {agent_id: slot for slot, agent_id in live_slot_agent_ids}

    alive_order = _checkpoint_cpu_tensor(tensors, "agent_ids_by_alive_order")
    if int(alive_order.dim()) != 1 or int(alive_order.shape[0]) != len(live_agent_ids):
        raise RuntimeError("native-to-reference materialization has malformed alive-order agent IDs")
    alive_order_values = [int(value) for value in alive_order.to(dtype=torch.int64).tolist()]
    if len(alive_order_values) != len(set(alive_order_values)) or set(alive_order_values) != set(live_agent_ids):
        raise RuntimeError("native-to-reference materialization alive-order agent IDs contradict live slots")

    sparse_agent_ids = _checkpoint_cpu_tensor(tensors, "agent_id_to_slot_agent_ids")
    sparse_slots = _checkpoint_cpu_tensor(tensors, "agent_id_to_slot_slots")
    if (
        int(sparse_agent_ids.dim()) != 1
        or int(sparse_slots.dim()) != 1
        or int(sparse_agent_ids.shape[0]) != len(live_agent_ids)
        or tuple(sparse_agent_ids.shape) != tuple(sparse_slots.shape)
    ):
        raise RuntimeError("native-to-reference materialization has malformed sparse slot mapping")
    sparse_pairs = [
        (int(agent_id), int(slot))
        for agent_id, slot in zip(
            sparse_agent_ids.to(dtype=torch.int64).tolist(),
            sparse_slots.to(dtype=torch.int64).tolist(),
            strict=True,
        )
    ]
    if len({agent_id for agent_id, _slot in sparse_pairs}) != len(sparse_pairs):
        raise RuntimeError("native-to-reference materialization sparse slot mapping has duplicate agent IDs")
    if [agent_id for agent_id, _slot in sparse_pairs] != sorted(
        agent_id for agent_id, _slot in sparse_pairs
    ):
        raise RuntimeError(
            "native-to-reference materialization sparse slot mapping is not strictly ordered"
        )
    if dict(sparse_pairs) != expected_by_agent_id:
        raise RuntimeError("native-to-reference materialization sparse slot mapping contradicts live slots")

    slot_token_values = _checkpoint_cpu_tensor(tensors, "slot_tokens").to(dtype=torch.int64).numpy()
    cache_token_values = _checkpoint_cpu_tensor(tensors, "action_cache_token").to(dtype=torch.int64).numpy()
    cache_cursor_values = _checkpoint_cpu_tensor(tensors, "action_cache_cursor").to(dtype=torch.int64).numpy()
    cache_next_values = _checkpoint_cpu_tensor(tensors, "action_cache_next").to(dtype=torch.int64).numpy()
    cache_actions_values = _checkpoint_cpu_tensor(tensors, "action_cache_actions").to(dtype=torch.int64).numpy()
    for slot in range(capacity):
        token = int(cache_token_values[slot])
        cursor = int(cache_cursor_values[slot])
        next_action = int(cache_next_values[slot])
        cache_active = token >= 0 or cursor >= 0 or next_action >= 0
        if not cache_active:
            if (token, cursor, next_action) != (-1, -1, -1):
                raise RuntimeError(
                    "native-to-reference materialization has contradictory inactive action cache: "
                    f"slot={slot} token={token} cursor={cursor} next={next_action}"
                )
            continue
        if not bool(alive[slot]):
            raise RuntimeError(f"native-to-reference materialization has an action cache on free slot={slot}")
        if token != int(slot_token_values[slot]):
            raise RuntimeError(
                "native-to-reference materialization action-cache token contradicts slot token: "
                f"slot={slot} cache={token} slot_token={int(slot_token_values[slot])}"
            )
        if cursor < 0 or cursor > policy_stride:
            raise RuntimeError(
                "native-to-reference materialization action-cache cursor is out of range: "
                f"slot={slot} cursor={cursor} stride={policy_stride}"
            )
        actions = cache_actions_values[slot]
        if bool(np.any(actions < 0)) or bool(np.any(actions >= action_count)):
            raise RuntimeError(f"native-to-reference materialization has invalid cached actions at slot={slot}")
        expected_next = int(actions[min(cursor, policy_stride - 1)])
        if next_action != expected_next:
            raise RuntimeError(
                "native-to-reference materialization action-cache next value contradicts cached actions: "
                f"slot={slot} expected={expected_next} actual={next_action}"
            )

    materialized_slot_agent_ids = live_slot_agent_ids + pending_slot_agent_ids
    for field in (
        "slot_to_agent_id",
        "slot_tokens",
        "agent_pos_y_by_slot",
        "agent_pos_x_by_slot",
        "agent_world_alive_by_slot",
    ):
        destination = getattr(env, field, None)
        if destination is None or tuple(np.asarray(destination).shape) != (capacity,):
            raise RuntimeError(
                "native-to-reference materialization destination world-slot "
                f"authority mismatch: field={field} expected_shape=({capacity},)"
            )

    fixed_policy_family = str(metadata.get("native_runtime_fixed_policy_family", "") or "").strip().lower()
    policy_use_gru = _checkpoint_cpu_tensor(tensors, "policy_use_gru").to(dtype=torch.bool).numpy()
    if fixed_policy_family == "mlp" and any(bool(policy_use_gru[slot]) for slot, _agent_id in live_slot_agent_ids):
        raise RuntimeError("native-to-reference materialization fixed-MLP projection contains live GRU slots")
    genome_hidden = tensors.get("genome_hidden_size")
    if genome_hidden is not None:
        genome_hidden_tensor = _require_checkpoint_tensor_shape(tensors, "genome_hidden_size", (capacity,))
        bad_hidden_slots = [
            slot
            for slot, _agent_id in live_slot_agent_ids
            if not bool(policy_use_gru[slot]) and int(genome_hidden_tensor[slot].item()) != hidden
        ]
        if bad_hidden_slots:
            raise RuntimeError(
                "native-to-reference materialization MLP hidden-size mismatch: "
                f"slots={bad_hidden_slots} expected={hidden}"
            )

    grid_fields = {
        "agent_grid": "agent_grid",
        "food_grid": "grid",
        "biome_grid": "biome_grid",
        "hazard_rate_map": "hazard_rate_map",
        "patch_food_amount": "patch_food_amount",
        "patch_scent": "patch_scent",
        "patch_waste": "patch_waste",
        "patch_scent_emit_pending": "patch_scent_emit_pending",
        "patch_waste_emit_pending": "patch_waste_emit_pending",
        "patch_recovery_activity": "patch_recovery_activity",
        **{name: name for name, _dtype in WDL_STORAGE_ENV_ARRAYS},
        **{name: name for name, _dtype in EED_GRID_ENV_ARRAYS},
        **{name: name for name, _dtype in EED_CURSOR_ENV_ARRAYS},
    }
    for tensor_name, env_name in grid_fields.items():
        if tensor_name not in tensors:
            continue
        destination = getattr(env, env_name, None)
        if destination is None or not hasattr(destination, "shape"):
            raise RuntimeError(
                "native-to-reference materialization destination is missing environment field: "
                f"field={env_name}"
            )
        tensor = _checkpoint_cpu_tensor(tensors, tensor_name)
        if tuple(int(value) for value in tensor.shape) != tuple(int(value) for value in destination.shape):
            raise RuntimeError(
                "native-to-reference materialization environment shape mismatch: "
                f"field={tensor_name} source={tuple(tensor.shape)} destination={tuple(destination.shape)}"
            )

    rng_manager = getattr(runner, "rng_manager", None)
    if rng_manager is None:
        raise RuntimeError("native-to-reference materialization requires RNGManager")
    policy_np = getattr(rng_manager, "policy_np", None)
    env_rng = getattr(env, "rng", None)
    runner_rng = getattr(runner, "np_rng", None)
    population_np_rng = getattr(pop, "np_rng", None)
    _require_numpy_rng_state_target(policy_np, label="rng_manager.policy_np")
    _require_numpy_rng_state_target(env_rng, label="env.rng")
    _require_numpy_rng_state_target(runner_rng, label="runner.np_rng")
    _require_numpy_rng_state_target(population_np_rng, label="pop.np_rng")
    env_rng_state = _validate_numpy_rng_state(
        metadata.get("native_runtime_numpy_env_state"),
        field="native_runtime_numpy_env_state",
    )
    runner_rng_state = _validate_numpy_rng_state(
        metadata.get("native_runtime_numpy_runner_state"),
        field="native_runtime_numpy_runner_state",
    )
    policy_rng_state = _validate_numpy_rng_state(
        metadata.get("native_runtime_numpy_policy_state"),
        field="native_runtime_numpy_policy_state",
    )
    population_np_rng_state = _validate_numpy_rng_state(
        metadata.get("native_runtime_numpy_population_state"),
        field="native_runtime_numpy_population_state",
    )
    alias_value = metadata.get("native_runtime_env_policy_numpy_alias")
    if not isinstance(alias_value, bool):
        raise RuntimeError(
            "native-to-reference materialization requires boolean metadata field="
            "native_runtime_env_policy_numpy_alias"
        )
    env_policy_alias = bool(alias_value)
    if env_policy_alias and dict(env_rng_state) != dict(policy_rng_state):
        raise RuntimeError("native-to-reference materialization env/policy alias has divergent RNG states")
    if not env_policy_alias and env_rng is policy_np:
        raise RuntimeError(
            "native-to-reference materialization cannot split an env/policy RNG alias without an exact producer object"
        )
    population_random_state = _python_random_state_from_payload(
        metadata.get("native_runtime_python_random_state")
    )
    if population_random_state is None:
        raise RuntimeError(
            "native-to-reference materialization requires RNG metadata field=native_runtime_python_random_state"
        )
    try:
        random.Random().setstate(population_random_state)
    except Exception as exc:
        raise RuntimeError("native-to-reference materialization has invalid population Python RNG state") from exc
    population_rng = getattr(pop, "rng", None)
    if not callable(getattr(population_rng, "setstate", None)):
        raise RuntimeError("native-to-reference materialization cannot restore pop.rng")

    rng_indices = _normalized_rng_manager_indices(metadata, rng_manager)
    mutation_torch_cpu_rng = None
    mutation_torch_cpu_state = None
    if "rng_manager_mutation_torch_cpu_state" in tensors:
        mutation_torch_cpu_state = _checkpoint_cpu_tensor(tensors, "rng_manager_mutation_torch_cpu_state")
        if int(mutation_torch_cpu_state.dim()) != 1 or mutation_torch_cpu_state.dtype != torch.uint8:
            raise RuntimeError(
                "native-to-reference materialization mutation torch CPU RNG state must be a uint8 vector"
            )
        try:
            torch.Generator(device="cpu").set_state(mutation_torch_cpu_state.clone())
        except Exception as exc:
            raise RuntimeError("native-to-reference materialization has invalid mutation torch CPU RNG state") from exc
        mutation_torch_cpu_rng = getattr(rng_manager, "torch_rng", lambda *_args, **_kwargs: None)(
            "mutation",
            "cpu",
        )
        if not callable(getattr(mutation_torch_cpu_rng, "_set_state_hex", None)):
            raise RuntimeError("native-to-reference materialization cannot restore mutation torch CPU RNG")

    return _ReferenceMaterializationPreflight(
        capacity=capacity,
        obs_size=obs_size,
        hidden=hidden,
        gru_hidden=gru_hidden,
        action_count=action_count,
        policy_stride=policy_stride,
        live_slot_agent_ids=live_slot_agent_ids,
        pending_slot_agent_ids=pending_slot_agent_ids,
        free_slots=free_slots,
        rng_manager=rng_manager,
        policy_np=policy_np,
        env_policy_alias=env_policy_alias,
        env_rng_state=env_rng_state,
        runner_rng_state=runner_rng_state,
        policy_rng_state=policy_rng_state,
        population_np_rng_state=population_np_rng_state,
        population_random_state=population_random_state,
        rng_indices=rng_indices,
        mutation_torch_cpu_rng=mutation_torch_cpu_rng,
        mutation_torch_cpu_state=mutation_torch_cpu_state,
        modular_records_by_slot=(
            copy.deepcopy(modular_storage.records_sidecar)
            if modular_storage is not None
            else None
        ),
    )


def _set_numpy_rng_state(value: Any, state: Mapping[str, Any]) -> None:
    setter = getattr(value, "_set_bitgen_state", None)
    if callable(setter):
        setter(copy.deepcopy(dict(state)))
        return
    getattr(value, "bit_generator").state = copy.deepcopy(dict(state))


def _install_reference_materialization_rng_state(
    runner: Any,
    preflight: _ReferenceMaterializationPreflight,
) -> None:
    env = runner.env
    pop = runner.pop
    _set_numpy_rng_state(preflight.policy_np, preflight.policy_rng_state)
    if preflight.env_policy_alias:
        env.rng = preflight.policy_np
    else:
        _set_numpy_rng_state(env.rng, preflight.env_rng_state)
    _set_numpy_rng_state(runner.np_rng, preflight.runner_rng_state)
    _set_numpy_rng_state(pop.np_rng, preflight.population_np_rng_state)
    pop.rng.setstate(preflight.population_random_state)
    if preflight.mutation_torch_cpu_state is not None:
        state_hex = bytes(preflight.mutation_torch_cpu_state.numpy().tobytes()).hex()
        preflight.mutation_torch_cpu_rng._set_state_hex(state_hex)
        restored_hex = preflight.mutation_torch_cpu_rng._get_state_hex()
        if restored_hex != state_hex:
            raise RuntimeError("native-to-reference materialization failed to restore mutation torch CPU RNG")
    indices_ref = getattr(preflight.rng_manager, "_indices")
    indices_ref.clear()
    indices_ref.update(preflight.rng_indices)


def apply_owner_checkpoint_to_runner(runner: Any, checkpoint_projection: Mapping[str, Any]) -> dict[str, Any]:
    tensors = dict(checkpoint_projection.get("tensors") or {})
    metadata = dict(checkpoint_projection.get("metadata") or {})
    env = getattr(runner, "env", None)
    pop = getattr(runner, "pop", None)
    if env is None or pop is None:
        raise RuntimeError("runner env/pop required for owner checkpoint export")
    _validate_policy_energy_observation_checkpoint_contract(
        metadata,
        tensors,
        surface="native-to-reference checkpoint materialization",
    )
    _validate_post_contract_checkpoint_source_projection(
        checkpoint_projection,
        metadata,
        surface="native-to-reference checkpoint materialization",
    )
    preflight = _preflight_reference_materialization(
        runner,
        checkpoint_projection,
        metadata,
        tensors,
    )

    def arr(name: str) -> np.ndarray:
        tensor = tensors[name]
        return tensor.detach().cpu().numpy()

    slot_to_agent_id = arr("slot_to_agent_id").astype(np.int64, copy=False)
    slot_tokens = arr("slot_tokens").astype(np.int64, copy=False) if "slot_tokens" in tensors else None
    alive = arr("alive_mask").astype(np.bool_, copy=False)
    pos_y = arr("pos_y").astype(np.int32, copy=False)
    pos_x = arr("pos_x").astype(np.int32, copy=False)
    energy = arr("energy").astype(np.float32, copy=False)
    policy_energy_observation = (
        arr("policy_energy_observation").astype(np.float64, copy=False)
        if "policy_energy_observation" in tensors
        else None
    )
    health = arr("health").astype(np.float32, copy=False)
    injury = arr("injury").astype(np.float32, copy=False)
    scar = arr("scar").astype(np.float32, copy=False) if "scar" in tensors else np.zeros_like(energy)
    age = arr("age").astype(np.int32, copy=False) if "age" in tensors else np.zeros_like(pos_y)
    drives = arr("drives").astype(np.float32, copy=False) if "drives" in tensors else None
    chemistry = arr("chemistry").astype(np.float32, copy=False) if "chemistry" in tensors else None
    policy_use_gru = arr("policy_use_gru").astype(np.bool_, copy=False) if "policy_use_gru" in tensors else None
    fixed_policy_family = str(metadata.get("native_runtime_fixed_policy_family", "") or "").strip().lower()
    fixed_mlp_family = fixed_policy_family == "mlp"
    survival_steps = arr("survival_steps").astype(np.int32, copy=False) if "survival_steps" in tensors else None
    early_survival_steps = (
        arr("early_survival_steps").astype(np.int32, copy=False) if "early_survival_steps" in tensors else None
    )
    energy_collected = arr("energy_collected").astype(np.float32, copy=False) if "energy_collected" in tensors else None
    energy_collected_exact = (
        arr("energy_collected_exact").astype(np.float64, copy=False)
        if "energy_collected_exact" in tensors
        else None
    )
    early_energy_collected = (
        arr("early_energy_collected").astype(np.float32, copy=False) if "early_energy_collected" in tensors else None
    )
    early_energy_collected_exact = (
        arr("early_energy_collected_exact").astype(np.float64, copy=False)
        if "early_energy_collected_exact" in tensors
        else None
    )
    last_energy_delta = arr("last_energy_delta").astype(np.float32, copy=False) if "last_energy_delta" in tensors else None
    last_health_delta = arr("last_health_delta").astype(np.float32, copy=False) if "last_health_delta" in tensors else None
    last_injury_delta = arr("last_injury_delta").astype(np.float32, copy=False) if "last_injury_delta" in tensors else None
    repro_cooldown = arr("repro_cooldown").astype(np.int32, copy=False) if "repro_cooldown" in tensors else None
    rest_streak = arr("rest_streak").astype(np.int32, copy=False) if "rest_streak" in tensors else None
    lineage_founder_id = arr("lineage_founder_id").astype(np.int32, copy=False) if "lineage_founder_id" in tensors else None
    lineage_depth = arr("lineage_depth").astype(np.int32, copy=False) if "lineage_depth" in tensors else None
    lineage_birth_generation = (
        arr("lineage_birth_generation").astype(np.int32, copy=False) if "lineage_birth_generation" in tensors else None
    )
    lineage_birth_tick = arr("lineage_birth_tick").astype(np.int32, copy=False) if "lineage_birth_tick" in tensors else None
    offspring_count = arr("offspring_count").astype(np.int32, copy=False) if "offspring_count" in tensors else None
    first_repro_step = arr("first_repro_step").astype(np.int32, copy=False) if "first_repro_step" in tensors else None
    patch_habitat_lineage_id = (
        arr("patch_habitat_lineage_id").astype(np.int32, copy=False) if "patch_habitat_lineage_id" in tensors else None
    )
    drive_volatility = arr("drive_volatility").astype(np.float32, copy=False) if "drive_volatility" in tensors else None
    drive_jitter = arr("drive_jitter").astype(np.float32, copy=False) if "drive_jitter" in tensors else None
    episode0_drive_jitter_sum = (
        arr("episode0_drive_jitter_sum").astype(np.float32, copy=False)
        if "episode0_drive_jitter_sum" in tensors
        else None
    )
    attacks = arr("attacks").astype(np.int32, copy=False) if "attacks" in tensors else None
    action_switches = arr("action_switches").astype(np.int32, copy=False) if "action_switches" in tensors else None
    action_oscillations = (
        arr("action_oscillations").astype(np.int32, copy=False) if "action_oscillations" in tensors else None
    )
    last_action = arr("last_action").astype(np.int32, copy=False) if "last_action" in tensors else None
    last_last_action = arr("last_last_action").astype(np.int32, copy=False) if "last_last_action" in tensors else None
    episode0_switches = (
        arr("episode0_switches").astype(np.int32, copy=False) if "episode0_switches" in tensors else None
    )
    episode0_oscillations = (
        arr("episode0_oscillations").astype(np.int32, copy=False) if "episode0_oscillations" in tensors else None
    )
    selfharm_events = arr("selfharm_events").astype(np.int32, copy=False) if "selfharm_events" in tensors else None
    episode0_selfharm_events = (
        arr("episode0_selfharm_events").astype(np.int32, copy=False)
        if "episode0_selfharm_events" in tensors
        else None
    )
    pred_err_count = arr("pred_err_count").astype(np.int32, copy=False) if "pred_err_count" in tensors else None
    episode0_pred_err_count = (
        arr("episode0_pred_err_count").astype(np.int32, copy=False)
        if "episode0_pred_err_count" in tensors
        else None
    )
    late_pred_err_count = (
        arr("late_pred_err_count").astype(np.int32, copy=False) if "late_pred_err_count" in tensors else None
    )
    birth_reset_shock = (
        arr("birth_reset_shock").astype(np.float32, copy=False) if "birth_reset_shock" in tensors else None
    )
    dev_stage_initialized = (
        arr("dev_stage_initialized").astype(np.bool_, copy=False)
        if "dev_stage_initialized" in tensors
        else None
    )
    pred_err_sum = arr("pred_err_sum").astype(np.float32, copy=False) if "pred_err_sum" in tensors else None
    episode0_pred_err_sum = (
        arr("episode0_pred_err_sum").astype(np.float32, copy=False) if "episode0_pred_err_sum" in tensors else None
    )
    late_pred_err_sum = (
        arr("late_pred_err_sum").astype(np.float32, copy=False) if "late_pred_err_sum" in tensors else None
    )
    prev_drive_delta = arr("prev_drive_delta").astype(np.float32, copy=False) if "prev_drive_delta" in tensors else None
    genome_int_fields = {
        "hidden_size": arr("genome_hidden_size").astype(np.int32, copy=False) if "genome_hidden_size" in tensors else None,
        "memory_dim": arr("genome_memory_dim").astype(np.int32, copy=False) if "genome_memory_dim" in tensors else None,
        "sensor_channel_mask": arr("genome_sensor_channel_mask").astype(np.int32, copy=False)
        if "genome_sensor_channel_mask" in tensors
        else None,
        "sensor_radius": arr("genome_sensor_radius").astype(np.int32, copy=False)
        if "genome_sensor_radius" in tensors
        else None,
        "introspection_channel_mask": arr("genome_introspection_channel_mask").astype(np.int32, copy=False)
        if "genome_introspection_channel_mask" in tensors
        else None,
        "self_regulation_control_mask": arr("genome_self_regulation_control_mask").astype(np.int32, copy=False)
        if "genome_self_regulation_control_mask" in tensors
        else None,
        "self_modification_target_mask": arr("genome_self_modification_target_mask").astype(np.int32, copy=False)
        if "genome_self_modification_target_mask" in tensors
        else None,
        "directed_inheritance_channel_mask": arr("genome_directed_inheritance_channel_mask").astype(np.int32, copy=False)
        if "genome_directed_inheritance_channel_mask" in tensors
        else None,
    }
    genome_float_fields = {
        "exploration_bias": arr("genome_exploration_bias").astype(np.float32, copy=False)
        if "genome_exploration_bias" in tensors
        else None,
        "aggression_bias": arr("genome_aggression_bias").astype(np.float32, copy=False)
        if "genome_aggression_bias" in tensors
        else None,
        "mutation_rate": arr("genome_mutation_rate").astype(np.float32, copy=False)
        if "genome_mutation_rate" in tensors
        else None,
        "reproduction_threshold": arr("genome_reproduction_threshold").astype(np.float32, copy=False)
        if "genome_reproduction_threshold" in tensors
        else None,
        "action_temperature": arr("genome_action_temperature").astype(np.float32, copy=False)
        if "genome_action_temperature" in tensors
        else None,
        "drive_inertia": arr("genome_drive_inertia").astype(np.float32, copy=False)
        if "genome_drive_inertia" in tensors
        else None,
        "drive_noise": arr("genome_drive_noise").astype(np.float32, copy=False)
        if "genome_drive_noise" in tensors
        else None,
        "drive_logit_gain": arr("genome_drive_logit_gain").astype(np.float32, copy=False)
        if "genome_drive_logit_gain" in tensors
        else None,
        "pred_lr": arr("genome_pred_lr").astype(np.float32, copy=False) if "genome_pred_lr" in tensors else None,
        "drive_sp_hunger": arr("genome_drive_sp_hunger").astype(np.float32, copy=False)
        if "genome_drive_sp_hunger" in tensors
        else None,
        "drive_sp_pain": arr("genome_drive_sp_pain").astype(np.float32, copy=False)
        if "genome_drive_sp_pain" in tensors
        else None,
        "drive_sp_fatigue": arr("genome_drive_sp_fatigue").astype(np.float32, copy=False)
        if "genome_drive_sp_fatigue" in tensors
        else None,
        "drive_sp_fear": arr("genome_drive_sp_fear").astype(np.float32, copy=False)
        if "genome_drive_sp_fear" in tensors
        else None,
        "exp_prior_strength": arr("genome_exp_prior_strength").astype(np.float32, copy=False)
        if "genome_exp_prior_strength" in tensors
        else None,
        "exp_prior_noise": arr("genome_exp_prior_noise").astype(np.float32, copy=False)
        if "genome_exp_prior_noise" in tensors
        else None,
        "drive_birth_noise": arr("genome_drive_birth_noise").astype(np.float32, copy=False)
        if "genome_drive_birth_noise" in tensors
        else None,
    }
    # Reference interchange is a total replacement boundary. Existing Python
    # agents, policies, and genomes may be stale while the native owner runs;
    # replacements remain detached until every row has been constructed.
    existing_agent_ids = {int(agent_id) for agent_id in dict(getattr(pop, "agents", {}))}
    agents: dict[int, Any] = {}
    materialized_slot_agent_ids = preflight.live_slot_agent_ids + preflight.pending_slot_agent_ids

    def _modular_record_for_slot(slot: int) -> dict[str, Any] | None:
        records = preflight.modular_records_by_slot
        if records is None:
            return None
        record = records[int(slot)]
        if not isinstance(record, Mapping):
            raise RuntimeError(
                "native-to-reference modular materialization is missing an occupied sidecar "
                f"record: slot={int(slot)}"
            )
        return copy.deepcopy(dict(record))

    def _genome_from_slot(slot: int) -> Any:
        from agents.developmental_program_v1 import canonicalize_developmental_program
        from agents.genome import Genome
        from agents.modular_genome_v1 import canon_float

        modular_record = _modular_record_for_slot(slot)
        if modular_record is not None:
            raw_genome = modular_record.get("genome")
            if not isinstance(raw_genome, Mapping):
                raise RuntimeError(
                    "native-to-reference modular materialization has malformed Genome authority: "
                    f"slot={int(slot)}"
                )
            try:
                return Genome.from_dict(copy.deepcopy(dict(raw_genome)))
            except Exception as exc:
                raise RuntimeError(
                    "native-to-reference modular materialization cannot restore the total Genome: "
                    f"slot={int(slot)}"
                ) from exc

        def gi(name: str, default: int) -> int:
            values = genome_int_fields.get(name)
            return int(values[slot]) if values is not None else int(default)

        def gf(name: str, default: float) -> float:
            values = genome_float_fields.get(name)
            return canon_float(float(values[slot]) if values is not None else float(default), field=name)

        memory_dim = 0 if fixed_mlp_family else gi("memory_dim", 0)
        return Genome(
            hidden_size=gi("hidden_size", 64),
            exploration_bias=gf("exploration_bias", 0.0),
            aggression_bias=gf("aggression_bias", 0.0),
            reproduction_threshold=gf("reproduction_threshold", 30.0),
            mutation_rate=gf("mutation_rate", 0.05),
            action_temperature=gf("action_temperature", 1.0),
            use_gru=False
            if fixed_mlp_family
            else bool((policy_use_gru is not None and bool(policy_use_gru[slot])) or memory_dim > 0),
            memory_dim=memory_dim,
            drive_inertia=gf("drive_inertia", 0.85),
            drive_noise=gf("drive_noise", 0.02),
            drive_logit_gain=gf("drive_logit_gain", 1.0),
            pred_lr=gf("pred_lr", 0.08),
            sensor_channel_mask=gi("sensor_channel_mask", 1023),
            sensor_radius=gi("sensor_radius", 2),
            introspection_channel_mask=gi("introspection_channel_mask", 32767),
            self_regulation_control_mask=gi("self_regulation_control_mask", 31),
            self_modification_target_mask=gi("self_modification_target_mask", 31),
            directed_inheritance_channel_mask=gi("directed_inheritance_channel_mask", 63),
            drive_sp_hunger=gf("drive_sp_hunger", 0.55),
            drive_sp_pain=gf("drive_sp_pain", 0.05),
            drive_sp_fatigue=gf("drive_sp_fatigue", 0.15),
            drive_sp_fear=gf("drive_sp_fear", 0.10),
            exp_prior_strength=gf("exp_prior_strength", 0.35),
            exp_prior_noise=gf("exp_prior_noise", 0.05),
            drive_birth_noise=gf("drive_birth_noise", 0.03),
            developmental_program=canonicalize_developmental_program(None),
        )

    def _policy_visible_energy_for_slot(slot: int) -> float:
        if policy_energy_observation is not None:
            value = float(policy_energy_observation[slot])
            if np.isfinite(value):
                return value
        return float(energy[slot])

    def _policy_from_packed_row(slot: int, genome: Any) -> Any:
        use_gru = False if fixed_mlp_family else bool(
            (policy_use_gru is not None and bool(policy_use_gru[slot])) or bool(getattr(genome, "use_gru", False))
        )
        obs_size = preflight.obs_size
        n_actions = preflight.action_count
        if use_gru:
            pack_t = tensors["gru_pack"]
            from agents.gru_policy import GRUPolicy

            hidden = preflight.gru_hidden
            row = pack_t.detach().cpu()[slot].to(dtype=torch.float32).contiguous()
            expected = (hidden * obs_size) + hidden + (3 * hidden * hidden) + (3 * hidden) + (3 * hidden * hidden) + (3 * hidden) + (n_actions * hidden) + n_actions
            if int(row.numel()) != int(expected):
                raise RuntimeError(
                    "native-to-reference materialization GRU policy row is incomplete: "
                    f"slot={slot} expected={expected} actual={int(row.numel())}"
                )
            cursor = 0
            pol = GRUPolicy(obs_size=obs_size, hidden_dim=hidden, n_actions=n_actions)
            with torch.no_grad():
                n = hidden * obs_size
                pol.encoder.weight.copy_(row[cursor : cursor + n].view(hidden, obs_size))
                cursor += n
                pol.encoder.bias.copy_(row[cursor : cursor + hidden])
                cursor += hidden
                n = 3 * hidden * hidden
                pol.gru.weight_ih.copy_(row[cursor : cursor + n].view(3 * hidden, hidden))
                cursor += n
                pol.gru.bias_ih.copy_(row[cursor : cursor + (3 * hidden)])
                cursor += 3 * hidden
                n = 3 * hidden * hidden
                pol.gru.weight_hh.copy_(row[cursor : cursor + n].view(3 * hidden, hidden))
                cursor += n
                pol.gru.bias_hh.copy_(row[cursor : cursor + (3 * hidden)])
                cursor += 3 * hidden
                n = n_actions * hidden
                pol.head.weight.copy_(row[cursor : cursor + n].view(n_actions, hidden))
                cursor += n
                pol.head.bias.copy_(row[cursor : cursor + n_actions])
            return pol

        pack_t = tensors["mlp_pack"]
        from agents.agent_policy import MLPPolicy

        hidden = preflight.hidden
        row = pack_t.detach().cpu()[slot].to(dtype=torch.float32).contiguous()
        expected = (hidden * obs_size) + hidden + (hidden * hidden) + hidden + (n_actions * hidden) + n_actions
        if int(row.numel()) != int(expected):
            raise RuntimeError(
                "native-to-reference materialization MLP policy row is incomplete: "
                f"slot={slot} expected={expected} actual={int(row.numel())}"
            )
        cursor = 0
        global_cpu_rng_state = torch.get_rng_state()
        try:
            pol = MLPPolicy(obs_size=obs_size, hidden_size=hidden, n_actions=n_actions)
        finally:
            # Native child construction already owns the constructor RNG draw.
            torch.set_rng_state(global_cpu_rng_state)
        with torch.no_grad():
            n = hidden * obs_size
            pol.net[0].weight.copy_(row[cursor : cursor + n].view(hidden, obs_size))
            cursor += n
            pol.net[0].bias.copy_(row[cursor : cursor + hidden])
            cursor += hidden
            n = hidden * hidden
            pol.net[2].weight.copy_(row[cursor : cursor + n].view(hidden, hidden))
            cursor += n
            pol.net[2].bias.copy_(row[cursor : cursor + hidden])
            cursor += hidden
            n = n_actions * hidden
            pol.net[4].weight.copy_(row[cursor : cursor + n].view(n_actions, hidden))
            cursor += n
            pol.net[4].bias.copy_(row[cursor : cursor + n_actions])
        return pol

    def _materialize_missing_agent_impl(slot: int, aid: int) -> Any:
        from agents.agent import Agent, normalize_lineage_stats

        genome = _genome_from_slot(slot)
        policy = _policy_from_packed_row(slot, genome)
        if policy is None:
            raise RuntimeError(
                "native-to-reference checkpoint materialization cannot reconstruct an exact policy: "
                f"slot={int(slot)} agent_id={int(aid)}"
            )
        agent = Agent(
            agent_id=int(aid),
            genome=genome,
            policy=policy,
            energy=_policy_visible_energy_for_slot(slot),
            health=float(health[slot]),
            alive=bool(alive[slot]),
            max_energy_norm=float(metadata.get("max_energy", 1.0) or 1.0),
            start_health_norm=float(metadata.get("start_health", 1.0) or 1.0),
            injury=float(injury[slot]),
            slot=int(slot),
            slot_token=int(slot_tokens[slot]) if slot_tokens is not None else 0,
        )
        agent.scar = float(scar[slot])
        agent.age = int(age[slot])
        if repro_cooldown is not None:
            agent.repro_cooldown = int(repro_cooldown[slot])
        if rest_streak is not None:
            agent.rest_streak = int(rest_streak[slot])
        if drives is not None:
            agent.drives = np.asarray(drives[slot], dtype=np.float32).copy()
        modular_record = _modular_record_for_slot(slot)
        if modular_record is not None:
            agent.structural_live_module_state = copy.deepcopy(
                dict(modular_record["structural_live_module_state"])
            )
            agent.structural_live_module_outputs = copy.deepcopy(
                dict(modular_record["structural_live_module_outputs"])
            )
            agent.structural_development_latch_by_instance_id = copy.deepcopy(
                dict(
                    modular_record[
                        "structural_development_latch_by_instance_id"
                    ]
                )
            )
            agent.structural_development_effective_gain_by_instance_id = copy.deepcopy(
                dict(
                    modular_record[
                        "structural_development_effective_gain_by_instance_id"
                    ]
                )
            )
            cost_summary = copy.deepcopy(dict(modular_record["cost_summary"]))
            agent.structural_live_module_telemetry = {
                "structural_cost": cost_summary,
                "structural_cost_total": float(cost_summary["total"]),
            }
        normalize_lineage_stats(
            agent.stats,
            int(aid),
            generation=int(metadata.get("generation", 0) or 0),
            tick=int(metadata.get("tick", 0) or 0),
        )
        return agent

    def _materialize_missing_agent(slot: int, aid: int) -> Any:
        global_cpu_rng_state = torch.get_rng_state()
        try:
            return _materialize_missing_agent_impl(slot, aid)
        finally:
            torch.set_rng_state(global_cpu_rng_state)

    materialized = 0
    synced = 0
    for slot, aid in materialized_slot_agent_ids:
        agent = _materialize_missing_agent(int(slot), int(aid))
        agents[int(aid)] = agent
        materialized += int(aid not in existing_agent_ids)
        agent.slot = int(slot)
        if slot_tokens is not None:
            agent.slot_token = int(slot_tokens[slot])
        agent.alive = bool(alive[slot])
        agent.energy = _policy_visible_energy_for_slot(slot)
        agent.health = float(health[slot])
        agent.injury = float(injury[slot])
        agent.scar = float(scar[slot])
        agent.age = int(age[slot])
        if repro_cooldown is not None:
            agent.repro_cooldown = int(repro_cooldown[slot])
        if rest_streak is not None:
            agent.rest_streak = int(rest_streak[slot])
        if drives is not None and hasattr(agent, "drives"):
            agent.drives[:] = drives[slot]
        if chemistry is not None:
            from agents.chemistry import chemistry_from_values

            agent.chemistry = chemistry_from_values(chemistry[slot])
        stats = getattr(agent, "stats", None)
        if stats is not None:
            if survival_steps is not None:
                stats.survival_steps = int(survival_steps[slot])
            if early_survival_steps is not None:
                stats.early_survival_steps = int(early_survival_steps[slot])
            if energy_collected_exact is not None:
                stats.energy_collected = float(energy_collected_exact[slot])
            elif energy_collected is not None:
                stats.energy_collected = float(energy_collected[slot])
            if early_energy_collected_exact is not None:
                stats.early_energy_collected = float(early_energy_collected_exact[slot])
            elif early_energy_collected is not None:
                stats.early_energy_collected = float(early_energy_collected[slot])
            if last_energy_delta is not None:
                stats.last_energy_delta = float(last_energy_delta[slot])
            if last_health_delta is not None:
                stats.last_health_delta = float(last_health_delta[slot])
            if last_injury_delta is not None:
                stats.last_injury_delta = float(last_injury_delta[slot])
            if lineage_founder_id is not None:
                _sync_checkpoint_lineage_identity(
                    stats,
                    int(lineage_founder_id[slot]),
                    metadata=metadata,
                )
            if lineage_depth is not None:
                stats.lineage_depth = int(lineage_depth[slot])
            if lineage_birth_generation is not None:
                stats.birth_generation = int(lineage_birth_generation[slot])
            if lineage_birth_tick is not None:
                stats.birth_tick = int(lineage_birth_tick[slot])
            if offspring_count is not None:
                stats.offspring_count = int(offspring_count[slot])
            if first_repro_step is not None:
                first_repro_value = int(first_repro_step[slot])
                stats.first_repro_step = None if first_repro_value < 0 else first_repro_value
            if patch_habitat_lineage_id is not None:
                stats.patch_habitat_lineage_id = int(patch_habitat_lineage_id[slot])
            if drive_volatility is not None:
                stats.drive_volatility = float(drive_volatility[slot])
            if drive_jitter is not None:
                stats.drive_jitter = float(drive_jitter[slot])
            if episode0_drive_jitter_sum is not None:
                stats.episode0_drive_jitter_sum = float(episode0_drive_jitter_sum[slot])
            if attacks is not None:
                stats.attacks = int(attacks[slot])
            if last_action is not None:
                stats.last_action = int(last_action[slot])
            if last_last_action is not None:
                stats.last_last_action = int(last_last_action[slot])
            if action_switches is not None:
                stats.action_switches = int(action_switches[slot])
            if action_oscillations is not None:
                stats.action_oscillations = int(action_oscillations[slot])
            if episode0_switches is not None:
                stats.episode0_switches = int(episode0_switches[slot])
            if episode0_oscillations is not None:
                stats.episode0_oscillations = int(episode0_oscillations[slot])
            if selfharm_events is not None:
                stats.selfharm_events = int(selfharm_events[slot])
            if episode0_selfharm_events is not None:
                stats.episode0_selfharm_events = int(episode0_selfharm_events[slot])
            if pred_err_count is not None:
                stats.pred_err_count = int(pred_err_count[slot])
            if episode0_pred_err_count is not None:
                stats.episode0_pred_err_count = int(episode0_pred_err_count[slot])
            if late_pred_err_count is not None:
                stats.late_pred_err_count = int(late_pred_err_count[slot])
            if birth_reset_shock is not None:
                stats.birth_reset_shock = float(birth_reset_shock[slot])
            if dev_stage_initialized is not None:
                if bool(dev_stage_initialized[slot]):
                    stats._dev_stage_id = "default"
                    stats._dev_active_module_count = 0
                else:
                    for name in ("_dev_stage_id", "_dev_active_module_count"):
                        if hasattr(stats, name):
                            delattr(stats, name)
            if pred_err_sum is not None:
                stats.pred_err_sum = float(pred_err_sum[slot])
            if episode0_pred_err_sum is not None:
                stats.episode0_pred_err_sum = float(episode0_pred_err_sum[slot])
            if late_pred_err_sum is not None:
                stats.late_pred_err_sum = float(late_pred_err_sum[slot])
            if prev_drive_delta is not None:
                stats.prev_drive_delta = np.asarray(prev_drive_delta[slot], dtype=np.float32).copy()
        genome = getattr(agent, "genome", None)
        if genome is not None and preflight.modular_records_by_slot is None:
            from agents.modular_genome_v1 import canon_float

            for name, values in genome_int_fields.items():
                if values is not None:
                    setattr(genome, name, int(values[slot]))
            for name, values in genome_float_fields.items():
                if values is not None:
                    setattr(genome, name, canon_float(float(values[slot]), field=name))
            if fixed_mlp_family:
                if hasattr(genome, "use_gru"):
                    genome.use_gru = False
                if hasattr(genome, "memory_dim"):
                    genome.memory_dim = 0
            elif policy_use_gru is not None and hasattr(genome, "use_gru"):
                genome.use_gru = bool(policy_use_gru[slot])
            refresh = getattr(genome, "refresh_canonical_payload_cache", None)
            if callable(refresh):
                refresh()
        synced += 1

    staged_agents = dict(sorted(agents.items(), key=lambda item: int(item[0])))
    projected_next_id = metadata.get("population_next_id")
    if isinstance(projected_next_id, bool) or not isinstance(projected_next_id, (int, np.integer)):
        raise RuntimeError("native-to-reference materialization requires integer metadata field=population_next_id")
    staged_next_id = max(
        max((agent_id for _slot, agent_id in materialized_slot_agent_ids), default=0) + 1,
        int(projected_next_id),
    )
    staged_slot_tokens = [int(value) for value in slot_tokens.tolist()]
    staged_free_slots = deque(preflight.free_slots)

    cache_token = arr("action_cache_token").astype(np.int64, copy=False)
    cache_cursor = arr("action_cache_cursor").astype(np.int32, copy=False)
    cache_actions = arr("action_cache_actions").astype(np.int32, copy=False)
    restored_cache: dict[int, list[Any]] = {}
    for slot, aid in preflight.live_slot_agent_ids:
        token = int(cache_token[slot])
        cursor = int(cache_cursor[slot])
        if token < 0 or cursor < 0:
            continue
        if cursor > preflight.policy_stride:
            raise RuntimeError(
                "native-to-reference materialization action-cache cursor is out of range: "
                f"slot={slot} cursor={cursor} stride={preflight.policy_stride}"
            )
        actions = np.asarray(cache_actions[slot], dtype=np.int32).reshape(-1).copy()
        restored_cache[aid] = [token, actions, cursor]

    staged_policy_tensors = {
        "_Hmem": tensors["hmem"].detach().clone(),
        "_mlp_pack": tensors["mlp_pack"].detach().clone(),
        "_gru_pack": tensors["gru_pack"].detach().clone(),
    }
    staged_env_positions_y = np.full_like(np.asarray(env.agent_pos_y), -1)
    staged_env_positions_x = np.full_like(np.asarray(env.agent_pos_x), -1)
    staged_env_alive = np.zeros_like(np.asarray(env.agent_alive), dtype=np.bool_)
    for slot, aid in materialized_slot_agent_ids:
        if aid < int(staged_env_alive.size):
            staged_env_positions_y[aid] = np.int32(pos_y[slot])
            staged_env_positions_x[aid] = np.int32(pos_x[slot])
            staged_env_alive[aid] = bool(alive[slot])
    staged_world_slot_to_agent_id = slot_to_agent_id.copy()
    staged_world_slot_tokens = slot_tokens.copy()
    staged_world_positions_y = pos_y.copy()
    staged_world_positions_x = pos_x.copy()
    staged_world_alive = alive.copy()
    staged_world_reverse = {
        int(agent_id): (int(slot), int(staged_world_slot_tokens[slot]))
        for slot, agent_id in enumerate(staged_world_slot_to_agent_id.tolist())
        if int(agent_id) >= 0
    }

    grid_fields = {
        "agent_grid": "agent_grid",
        "food_grid": "grid",
        "biome_grid": "biome_grid",
        "hazard_rate_map": "hazard_rate_map",
        "patch_food_amount": "patch_food_amount",
        "patch_scent": "patch_scent",
        "patch_waste": "patch_waste",
        "patch_scent_emit_pending": "patch_scent_emit_pending",
        "patch_waste_emit_pending": "patch_waste_emit_pending",
        "patch_recovery_activity": "patch_recovery_activity",
        **{name: name for name, _dtype in WDL_STORAGE_ENV_ARRAYS},
        **{name: name for name, _dtype in EED_GRID_ENV_ARRAYS},
        **{name: name for name, _dtype in EED_CURSOR_ENV_ARRAYS},
    }
    staged_grids: list[tuple[Any, np.ndarray]] = []
    for tensor_name, env_name in grid_fields.items():
        if tensor_name not in tensors:
            continue
        destination = getattr(env, env_name)
        staged_grids.append(
            (
                destination,
                np.asarray(arr(tensor_name), dtype=getattr(destination, "dtype", None)).copy(),
            )
        )

    staged_drive_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    if drives is not None:
        if drives.ndim != 2 or int(drives.shape[0]) != preflight.capacity or int(drives.shape[1]) != 4:
            raise RuntimeError(
                "native-to-reference materialization drives must have shape "
                f"({preflight.capacity}, 4), got {tuple(drives.shape)}"
            )
        for attr, col in (("_drive_hunger", 0), ("_drive_pain", 1), ("_drive_fatigue", 2), ("_drive_fear", 3)):
            destination = getattr(runner, attr, None)
            if destination is None or int(getattr(destination, "shape", [0])[0]) != preflight.capacity:
                raise RuntimeError(
                    "native-to-reference materialization destination drive array mismatch: "
                    f"field={attr}"
                )
            staged_drive_arrays.append((destination, np.asarray(drives[:, col], dtype=destination.dtype).copy()))

    typed_lifecycle = {
        name: tensor.detach().cpu().clone()
        for name, tensor in tensors.items()
        if name.startswith("typed_lifecycle_event_") and hasattr(tensor, "detach")
    }
    if typed_lifecycle:
        expected_typed_names = set(_empty_prefixed_typed_lifecycle_events())
        if set(typed_lifecycle) != expected_typed_names:
            raise RuntimeError("native-to-reference materialization has an incomplete typed lifecycle tensor set")
        staged_lifecycle_accumulator = typed_lifecycle
    else:
        staged_lifecycle_accumulator = _empty_prefixed_typed_lifecycle_events()
    reproduction_source = _reference_reproduction_context_source(runner)
    staged_metadata_overrides = {
        name: copy.deepcopy(metadata[name])
        for name in (
            "generation",
            "tick",
            "native_runtime_numpy_birth_spawn_env_state",
        )
        if name in metadata
    }

    # All validation and object construction is complete. Install the detached
    # replacement state without source tensor overrides so verification rebuilds
    # from the materialized reference objects rather than certifying copied input.
    _install_reference_materialization_rng_state(runner, preflight)
    for destination, source in staged_grids:
        destination[...] = source
    env.agent_pos_y[...] = staged_env_positions_y
    env.agent_pos_x[...] = staged_env_positions_x
    env.agent_alive[...] = staged_env_alive
    env.slot_to_agent_id[...] = staged_world_slot_to_agent_id
    env.slot_tokens[...] = staged_world_slot_tokens
    env.agent_pos_y_by_slot[...] = staged_world_positions_y
    env.agent_pos_x_by_slot[...] = staged_world_positions_x
    env.agent_world_alive_by_slot[...] = staged_world_alive
    env._agent_id_to_world_slot.clear()
    env._agent_id_to_world_slot.update(staged_world_reverse)
    env._world_authority_migration_required = False
    pop.agents = staged_agents
    pop.next_id = staged_next_id
    pop._slots.slot_tokens = staged_slot_tokens
    pop._slots.free_slots = staged_free_slots
    runner._policy_action_cache = restored_cache
    for name, value in staged_policy_tensors.items():
        setattr(runner, name, value)
    if any(bool(policy_use_gru[slot]) for slot, _agent_id in materialized_slot_agent_ids):
        runner._eiel_skip_gru_memory_reset_once = True
    for destination, source in staged_drive_arrays:
        destination[...] = source
    runner._cxxbackbone10_checkpoint_tensor_overrides = {}
    setattr(runner, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR, [])
    setattr(runner, REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR, staged_lifecycle_accumulator)
    setattr(runner, REFERENCE_LIFECYCLE_DEATH_CURSOR_ATTR, 0)
    setattr(
        runner,
        REFERENCE_LIFECYCLE_REPRODUCTION_CURSOR_ATTR,
        (reproduction_source, len(reproduction_source)),
    )
    runner._cxxbackbone10_checkpoint_metadata_overrides = staged_metadata_overrides
    env._views_dirty = True
    return {
        "schema_version": "cxxbackbone13_owner_checkpoint_runner_export_v1",
        "synced_agents": synced,
        "materialized_agents": materialized,
        "pending_dead_agents": len(preflight.pending_slot_agent_ids),
    }


def execute_owner_generation_from_runner(
    runner: Any,
    *,
    generation_index: int,
    steps_per_generation: int,
    build_directory: str | Path | None = None,
) -> dict[str, Any]:
    patch = bool(getattr(getattr(runner, "env_cfg", None), "patch_world_enable", False))
    capability = {"flat": not patch, "patch": patch, "churn": False}
    owner = ResidentRuntimeState.from_runner(
        runner,
        capability,
        generation=int(generation_index),
        tick=0,
        build_directory=build_directory,
    )
    if patch:
        result = owner.execute_generation_patch_cpu(int(generation_index), int(steps_per_generation))
    else:
        result = owner.execute_generation_flat_cpu(int(generation_index), int(steps_per_generation))
    checkpoint = owner.export_checkpoint_projection()
    export = apply_owner_checkpoint_to_runner(runner, checkpoint)
    out = dict(result)
    out["schema_version"] = "cxxbackbone13_owner_generation_runner_result_v1"
    out["world"] = "patch" if patch else "flat"
    out["runner_export"] = export
    out["python_generation_calls"] = 1
    out["python_tick_iterations"] = 0
    out["python_policy_callbacks"] = 0
    out["python_environment_callbacks"] = 0
    return out


def projection_digest(projection: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    metadata = dict(projection.get("metadata") or {})
    metadata.pop("import_digest", None)
    # This is a transient service-control flag, not trajectory state. It may be
    # absent after checkpoint apply/reimport while all authoritative tensors match.
    metadata.pop("native_policy_cache_served", None)
    # Execution-segment delivery provenance remains in the checkpoint and its
    # emitted segment artifacts, but it must not change the resident trajectory
    # identity: one uninterrupted owner and a checkpoint/fresh-owner resume
    # legitimately have different segment counts and chain hashes.
    metadata.pop("native_next_segment_index", None)
    metadata.pop("native_chain_sha256", None)
    h.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    tensors = dict(projection.get("tensors") or {})
    for name in sorted(tensors):
        tensor = tensors[name]
        if not hasattr(tensor, "detach"):
            continue
        cpu = tensor.detach().to("cpu").contiguous()
        arr = cpu.numpy()
        h.update(str(name).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(json.dumps([int(x) for x in arr.shape], separators=(",", ":")).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def compare_projection_digest(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_digest = projection_digest(left)
    right_digest = projection_digest(right)
    from native_runtime.lifecycle_events import compare_committed_death_semantics  # noqa: WPS433

    return {
        "schema_version": "cxxbackbone10_projection_digest_compare_v1",
        "ok": left_digest == right_digest,
        "left_digest": left_digest,
        "right_digest": right_digest,
        "committed_death_semantics": compare_committed_death_semantics(left, right),
    }


def _tensor_from_numpy(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).contiguous()


def _tensor_dict_to_device(tensors: Mapping[str, Any], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {
        str(name): tensor.detach().to(device).contiguous() if hasattr(tensor, "detach") else tensor
        for name, tensor in dict(tensors).items()
    }


def _projection_to_device(projection: ImportProjectionV10, device: str | torch.device) -> ImportProjectionV10:
    metadata = dict(projection.metadata)
    metadata["owner_device"] = str(torch.device(device))
    return ImportProjectionV10(
        schema_version=projection.schema_version,
        metadata=metadata,
        tensors=_tensor_dict_to_device(projection.tensors, device),
        pending_generation_free_agent_ids=projection.pending_generation_free_agent_ids,
        pending_generation_free_slots=projection.pending_generation_free_slots,
    )


def _required_tensor(value: Any, name: str) -> torch.Tensor:
    if value is None or not hasattr(value, "detach"):
        raise RuntimeError(f"required runner tensor missing: {name}")
    return value.detach().contiguous()


def _hmem_tensor_for_projection(runner: Any, capacity: int) -> torch.Tensor:
    value = getattr(runner, "_Hmem", None)
    if value is not None and hasattr(value, "detach"):
        return value.detach().contiguous()
    fixed_family = str(
        getattr(getattr(runner, "sim_cfg", None), "native_runtime_fixed_policy_family", "") or ""
    ).strip().lower()
    if fixed_family != "mlp":
        raise RuntimeError("required runner tensor missing: hmem")
    hidden = int(getattr(runner, "_gru_hidden", 64) or 64)
    return torch.zeros((int(capacity), int(hidden)), dtype=torch.float32)


def _gru_pack_tensor_for_projection(runner: Any, capacity: int) -> torch.Tensor:
    value = getattr(runner, "_gru_pack", None)
    if value is not None and hasattr(value, "detach"):
        return value.detach().contiguous()
    fixed_family = str(
        getattr(getattr(runner, "sim_cfg", None), "native_runtime_fixed_policy_family", "") or ""
    ).strip().lower()
    if fixed_family != "mlp":
        raise RuntimeError("required runner tensor missing: gru_pack")
    obs_size = int(getattr(runner, "_obs_size", 0) or 0)
    hidden = int(getattr(runner, "_gru_hidden", 64) or 64)
    actions = int(getattr(runner, "_out", 7) or 7)
    pack_cols = (
        (hidden * obs_size)
        + hidden
        + (3 * hidden * hidden)
        + (3 * hidden)
        + (3 * hidden * hidden)
        + (3 * hidden)
        + (actions * hidden)
        + actions
    )
    return torch.zeros((int(capacity), int(pack_cols)), dtype=torch.float32)


def _slot_free_queue_tensor(runner: Any) -> torch.Tensor:
    slots = getattr(getattr(runner, "pop", None), "_slots", None)
    free_slots = getattr(slots, "free_slots", ()) if slots is not None else ()
    values = np.asarray([int(slot) for slot in list(free_slots)], dtype=np.int64)
    return _tensor_from_numpy(values)


def _slot_tokens_tensor(runner: Any, fallback: np.ndarray) -> torch.Tensor:
    fallback_values = np.asarray(fallback, dtype=np.int64).reshape(-1)
    slots = getattr(getattr(runner, "pop", None), "_slots", None)
    slot_tokens = getattr(slots, "slot_tokens", None) if slots is not None else None
    if slot_tokens is None:
        return _tensor_from_numpy(fallback_values)
    values = np.asarray([int(token) for token in list(slot_tokens)], dtype=np.int64)
    if values.shape != fallback_values.shape:
        raise RuntimeError(
            "SlotPool token history shape does not match the resident owner capacity: "
            f"slot_pool={values.shape} owner={fallback_values.shape}"
        )
    return _tensor_from_numpy(values)


def _python_random_state_payload(state: Any) -> dict[str, Any] | None:
    if state is None:
        return None
    try:
        version, inner, gauss_next = state
        values = list(inner)
        if len(values) != 625:
            return None
        return {
            "schema_version": "python_random_state_v1",
            "version": int(version),
            "state": [int(value) & 0xFFFFFFFF for value in values[:624]],
            "index": int(values[624]),
            "gauss_next": None if gauss_next is None else float(gauss_next),
            "raw_draws": 0,
        }
    except Exception:
        return None


def _python_random_state_from_payload(payload: Any) -> tuple[int, tuple[int, ...], float | None] | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        values = [int(value) & 0xFFFFFFFF for value in list(payload.get("state") or [])]
        if len(values) != 624:
            return None
        index = int(payload.get("index", 624))
        gauss_next_raw = payload.get("gauss_next")
        gauss_next = None if gauss_next_raw is None else float(gauss_next_raw)
        return int(payload.get("version", 3)), tuple(values + [index]), gauss_next
    except Exception:
        return None


def _population_birth_noise_projection(pop: Any, capacity: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    samples = np.zeros((max(0, int(capacity)), 4), dtype=np.float32)
    post_states: list[dict[str, Any]] = []
    rng = getattr(pop, "np_rng", None) if pop is not None else None
    bitgen = getattr(rng, "bit_generator", None)
    if rng is None or bitgen is None or int(capacity) <= 0:
        return _tensor_from_numpy(samples), post_states
    original_state = copy.deepcopy(getattr(bitgen, "state", None))
    if original_state is None:
        return _tensor_from_numpy(samples), post_states
    try:
        n64 = np.empty((4,), dtype=np.float64)
        n32 = np.empty((4,), dtype=np.float32)
        for row in range(int(capacity)):
            rng.standard_normal(size=(4,), dtype=np.float64, out=n64)
            n32[...] = n64
            samples[row, :] = n32
            post_states.append(copy.deepcopy(bitgen.state))
    finally:
        try:
            bitgen.state = original_state
        except Exception:
            pass
    return _tensor_from_numpy(samples), post_states


def _policy_mem_dim_by_slot(runner: Any, capacity: int) -> torch.Tensor:
    from agents.cognition import effective_memory_dim_for_genome  # local import keeps base import narrow

    out = np.zeros((int(capacity),), dtype=np.int64)
    pop = getattr(runner, "pop", None)
    if pop is not None and hasattr(pop, "alive_agents"):
        agents = list(pop.alive_agents())
    else:
        raw_agents = getattr(pop, "agents", {}) if pop is not None else {}
        agents = list(raw_agents.values()) if isinstance(raw_agents, dict) else list(raw_agents or [])
    for agent in agents:
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if slot < 0 or slot >= int(capacity):
            continue
        genome = getattr(agent, "genome", None)
        out[slot] = np.int64(effective_memory_dim_for_genome(genome, sim_cfg=getattr(runner, "sim_cfg", None)))
    return _tensor_from_numpy(out)


def _scar_by_slot(runner: Any, capacity: int) -> torch.Tensor:
    out = np.zeros((int(capacity),), dtype=np.float32)
    pop = getattr(runner, "pop", None)
    if pop is not None and hasattr(pop, "alive_agents"):
        agents = list(pop.alive_agents())
    else:
        raw_agents = getattr(pop, "agents", {}) if pop is not None else {}
        agents = list(raw_agents.values()) if isinstance(raw_agents, dict) else list(raw_agents or [])
    for agent in agents:
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            out[slot] = np.float32(float(getattr(agent, "scar", 0.0) or 0.0))
    return _tensor_from_numpy(out)


def _alive_agents_from_runner(runner: Any) -> list[Any]:
    pop = getattr(runner, "pop", None)
    if pop is not None and hasattr(pop, "alive_agents"):
        return list(pop.alive_agents())
    raw_agents = getattr(pop, "agents", {}) if pop is not None else {}
    return list(raw_agents.values()) if isinstance(raw_agents, dict) else list(raw_agents or [])


def _stat_int_by_slot(runner: Any, capacity: int, name: str, default: int) -> torch.Tensor:
    out = np.full((int(capacity),), np.int32(default), dtype=np.int32)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            stats = getattr(agent, "stats", None)
            out[slot] = np.int32(int(getattr(stats, name, default) if stats is not None else default))
    return _tensor_from_numpy(out)


def _lineage_founder_id_by_slot(runner: Any, capacity: int) -> torch.Tensor:
    out = np.full((int(capacity),), np.int32(-1), dtype=np.int32)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            stats = getattr(agent, "stats", None)
            value = int(
                getattr(stats, "founder_id", -1) if stats is not None else -1
            )
            if value < -1 or value > np.iinfo(np.int32).max:
                raise RuntimeError(
                    "resident import lineage_founder_id must be -1 or an "
                    "integer in [0, 2147483647]"
                )
            out[slot] = np.int32(value)
    return _tensor_from_numpy(out)


def _stat_float64_by_slot(runner: Any, capacity: int, name: str, default: float) -> torch.Tensor:
    out = np.full((int(capacity),), np.float64(default), dtype=np.float64)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            stats = getattr(agent, "stats", None)
            out[slot] = np.float64(float(getattr(stats, name, default) if stats is not None else default))
    return _tensor_from_numpy(out)


def _effective_pred_lr_exact_by_slot(runner: Any, capacity: int) -> torch.Tensor:
    out = np.zeros((int(capacity),), dtype=np.float64)
    resolver = getattr(runner, "resolve_developmental_effective_scalars_for_agent", None)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if not 0 <= slot < int(capacity):
            continue
        value: float | None = None
        if callable(resolver):
            scalars = resolver(agent)
            if scalars is not None:
                value = float(scalars[5])
        if value is None:
            value = float(getattr(getattr(agent, "genome", None), "pred_lr", 0.08))
        out[slot] = np.float64(min(1.0, max(0.0, value)))
    return _tensor_from_numpy(out)


def _prediction_queue_tensors(state: Any, capacity: int) -> dict[str, torch.Tensor]:
    queue_capacity = 8
    due = np.full((int(capacity), queue_capacity), -1, dtype=np.int64)
    action = np.full((int(capacity), queue_capacity), -1, dtype=np.int32)
    exp_de = np.zeros((int(capacity), queue_capacity), dtype=np.float32)
    exp_dinj = np.zeros((int(capacity), queue_capacity), dtype=np.float32)
    energy = np.zeros((int(capacity), queue_capacity), dtype=np.float64)
    injury = np.zeros((int(capacity), queue_capacity), dtype=np.float64)
    count = np.zeros((int(capacity),), dtype=np.int32)
    queues = list(getattr(state, "pred_queue", ()) or ())
    if len(queues) < int(capacity):
        queues.extend(() for _ in range(int(capacity) - len(queues)))
    for slot in range(int(capacity)):
        rows = list(queues[slot] or ())
        if len(rows) > queue_capacity:
            raise ValueError(
                f"prediction queue exceeds fixed native capacity: slot={slot} rows={len(rows)}"
            )
        count[slot] = np.int32(len(rows))
        for index, raw in enumerate(rows):
            if not isinstance(raw, (tuple, list)) or len(raw) != 6:
                raise ValueError(
                    f"prediction queue row must contain six values: slot={slot} index={index}"
                )
            due[slot, index] = np.int64(int(raw[0]))
            action[slot, index] = np.int32(int(raw[1]))
            exp_de[slot, index] = np.float32(float(raw[2]))
            exp_dinj[slot, index] = np.float32(float(raw[3]))
            energy[slot, index] = np.float64(float(raw[4]))
            injury[slot, index] = np.float64(float(raw[5]))
    return {
        "prediction_queue_due": _tensor_from_numpy(due),
        "prediction_queue_action": _tensor_from_numpy(action),
        "prediction_queue_exp_de": _tensor_from_numpy(exp_de),
        "prediction_queue_exp_dinj": _tensor_from_numpy(exp_dinj),
        "prediction_queue_energy": _tensor_from_numpy(energy),
        "prediction_queue_injury": _tensor_from_numpy(injury),
        "prediction_queue_count": _tensor_from_numpy(count),
    }


def _first_repro_step_by_slot(runner: Any, capacity: int) -> torch.Tensor:
    out = np.full((int(capacity),), np.int32(-1), dtype=np.int32)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if not 0 <= slot < int(capacity):
            continue
        stats = getattr(agent, "stats", None)
        value = getattr(stats, "first_repro_step", None) if stats is not None else None
        if value is not None:
            out[slot] = np.int32(int(value))
    return _tensor_from_numpy(out)


def _dev_stage_initialized_by_slot(runner: Any, capacity: int) -> torch.Tensor:
    out = np.zeros((int(capacity),), dtype=np.bool_)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if not 0 <= slot < int(capacity):
            continue
        stats = getattr(agent, "stats", None)
        out[slot] = bool(str(getattr(stats, "_dev_stage_id", "") or "")) if stats is not None else False
    return _tensor_from_numpy(out)


def _zero_free_rows_i32(tensor: torch.Tensor, alive_mask: np.ndarray) -> torch.Tensor:
    out = tensor.clone()
    mask = torch.from_numpy(np.asarray(alive_mask, dtype=np.bool_))
    if int(mask.numel()) == int(out.numel()):
        out[~mask] = 0
    return out


def _stat_float_by_slot(runner: Any, capacity: int, name: str, default: float) -> torch.Tensor:
    out = np.full((int(capacity),), np.float32(default), dtype=np.float32)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            stats = getattr(agent, "stats", None)
            out[slot] = np.float32(float(getattr(stats, name, default) if stats is not None else default))
    return _tensor_from_numpy(out)


def _stat_float64_by_slot(runner: Any, capacity: int, name: str, default: float) -> torch.Tensor:
    out = np.full((int(capacity),), np.float64(default), dtype=np.float64)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            stats = getattr(agent, "stats", None)
            value = getattr(stats, name, default) if stats is not None else default
            out[slot] = np.float64(float(value))
    return _tensor_from_numpy(out)


def _agent_float64_by_slot(runner: Any, capacity: int, name: str, default: float) -> torch.Tensor:
    out = np.full((int(capacity),), np.float64(default), dtype=np.float64)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            out[slot] = np.float64(float(getattr(agent, name, default)))
    return _tensor_from_numpy(out)


def _genome_float_by_slot(runner: Any, capacity: int, name: str, default: float) -> torch.Tensor:
    out = np.full((int(capacity),), np.float32(default), dtype=np.float32)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            genome = getattr(agent, "genome", None)
            out[slot] = np.float32(float(getattr(genome, name, default) if genome is not None else default))
    return _tensor_from_numpy(out)


def _genome_int_by_slot(runner: Any, capacity: int, name: str, default: int) -> torch.Tensor:
    out = np.full((int(capacity),), np.int32(default), dtype=np.int32)
    for agent in _alive_agents_from_runner(runner):
        if not bool(getattr(agent, "alive", True)):
            continue
        slot = int(getattr(agent, "slot", -1))
        if 0 <= slot < int(capacity):
            genome = getattr(agent, "genome", None)
            out[slot] = np.int32(int(getattr(genome, name, default) if genome is not None else default))
    return _tensor_from_numpy(out)


__all__ = [
    "CHECKPOINT_PROJECTION_SCHEMA_VERSION_V1",
    "CHECKPOINT_PROJECTION_SCHEMA_VERSION_V2",
    "OWNER_DESIGN_SCHEMA_VERSION",
    "REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR",
    "REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR",
    "RESIDENT_STATE_OWNER_EXTENSION_ENV",
    "ResidentStateOwnerExtensionStatus",
    "ResidentRuntimeState",
    "SCHEMA_VERSION",
    "build_import_projection",
    "compare_projection_digest",
    "execute_flat_owner_batch",
    "execute_owner_generation_from_runner",
    "install_reference_committed_death_oracle_hook",
    "load_resident_state_owner_extension",
    "migrate_native_checkpoint_projection_v1_to_v2",
    "owner_design_decision_payload",
    "projection_digest",
    "reference_committed_death_oracle",
    "resident_state_owner_extension_status",
]
