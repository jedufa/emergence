"""Fail-closed R2 adapter for an ordinary resident C4 birth transaction.

The Architecture-B transaction is reused only when its flat, single-world
state is the authoritative TensorWorld state.  The current C4 bundle does not
yet meet that condition: several continuation tensors are absent, some source
authorities have incompatible layouts/dtypes, and the native helper does not
publish typed events or reset the complete C3 newborn surface.  This module
therefore provides a caller-owned shadow/projection contract and refuses to
launch the native transaction until those differences are removed in source.

Inspection is structural only.  It does not materialize CUDA values, perform a
host synchronization, allocate a fallback tensor, or loop over births/worlds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import torch

from ..perf.cuda_architecture_b_birth_transaction import (
    DRIVE_STATE_FIELDS,
    GENOME_FLOAT32_FIELDS,
    GENOME_INT32_FIELDS,
    IDENTITY_STATE_FIELDS,
    SCALAR_RNG_STATE_FIELDS,
    execute_fixed_mlp_birth_transaction_cuda,
)
from .c4_population import TensorWorldC4BirthConfigV1
from .c4_state import TensorWorldStateBundleV4
from .contracts import TensorWorldContractError


C4_RESIDENT_BIRTH_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_birth_adapter_v1"
)
C4_RESIDENT_BIRTH_SHADOW_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_birth_shadow_v1"
)
C4_RESIDENT_BIRTH_DECISION_V1: Final[str] = "NO_GATE_DECISION"

# The ordinary (asexual) Architecture-B entrypoint excludes the crossover tape.
C4_RESIDENT_BIRTH_POLICY_FIELDS_V1: Final[tuple[str, ...]] = (
    "mlp_pack",
    "hmem",
    "rng_manager_mutation_torch_cpu_state",
    "policy_birth_delta_tape",
    "policy_birth_mutation_mask_tape",
    "policy_birth_post_rng_state_packed_tape",
    "policy_birth_semantic_draws_tape",
    "policy_birth_tape_cursor",
    "mutation_torch_cpu_semantic_draws",
)

# These fields have no authoritative owner in TensorWorldStateBundleV4.  A
# caller may not make them authoritative merely by allocating lookalike CUDA
# tensors; doing so would fabricate continuation semantics.
C4_RESIDENT_BIRTH_UNRESOLVED_SOURCE_FIELDS_V1: Final[tuple[str, ...]] = (
    "plan.spawn_positions_interleaved",
    "drive.birth_reset_shock",
    "drive.pcg64_state_words",
    "drive.pcg64_buffer_words",
    "drive.mutation_np_semantic_draws",
    "policy.hmem",
    "policy.rng_manager_mutation_torch_cpu_state",
    "policy.policy_birth_delta_tape",
    "policy.policy_birth_mutation_mask_tape",
    "policy.policy_birth_post_rng_state_packed_tape",
    "policy.policy_birth_semantic_draws_tape",
    "policy.policy_birth_tape_cursor",
    "policy.mutation_torch_cpu_semantic_draws",
)

C4_RESIDENT_BIRTH_STATIC_BLOCKERS_V1: Final[tuple[str, ...]] = (
    "native_missing_complete_newborn_reset",
    "native_missing_typed_birth_events",
    "tensorworld_mirrored_authorities_not_atomically_updated",
)

_IDENTITY_NATIVE_DTYPES: Final[dict[str, torch.dtype]] = {
    "slot_to_agent_id": torch.int64,
    "agent_id_to_slot": torch.int32,
    "slot_tokens": torch.int64,
    "alive_mask": torch.bool,
    "pos_y": torch.int32,
    "pos_x": torch.int32,
    "agent_grid": torch.int32,
    "energy": torch.float32,
    "health": torch.float32,
    "injury": torch.float32,
    "scar": torch.float32,
    "age": torch.int32,
    "repro_cooldown": torch.int32,
    "repro_threshold": torch.float32,
    "offspring_count": torch.int32,
    "lineage_founder_id": torch.int32,
    "lineage_depth": torch.int32,
    "lineage_birth_generation": torch.int32,
    "lineage_birth_tick": torch.int32,
    "patch_habitat_lineage_id": torch.int32,
    "policy_energy_observation": torch.float64,
    "survival_steps": torch.int32,
    "rest_streak": torch.int32,
    "action_cache_token": torch.int64,
    "action_cache_cursor": torch.int32,
    "action_cache_next": torch.int32,
    "slot_free_queue": torch.int64,
    "slot_free_queue_count": torch.int64,
    "population_next_id": torch.int64,
    "last_action": torch.int32,
    "last_last_action": torch.int32,
    "action_switches": torch.int32,
    "action_oscillations": torch.int32,
    "episode0_switches": torch.int32,
    "episode0_oscillations": torch.int32,
}
_SCALAR_RNG_NATIVE_DTYPES: Final[dict[str, torch.dtype]] = {
    "python_random_words": torch.uint32,
    "python_random_index": torch.int64,
    "python_random_has_gauss": torch.bool,
    "python_random_gauss_next": torch.float64,
    "python_random_gauss_primary_tape": torch.float64,
    "python_random_gauss_secondary_tape": torch.float64,
    "python_random_raw_draws": torch.int64,
    "mutation_py_semantic_draws": torch.int64,
}
_GENOME_NATIVE_DTYPES: Final[dict[str, torch.dtype]] = {
    **{name: torch.int32 for name in GENOME_INT32_FIELDS},
    **{name: torch.float32 for name in GENOME_FLOAT32_FIELDS},
}
_DRIVE_NATIVE_DTYPES: Final[dict[str, torch.dtype]] = {
    "drives": torch.float32,
    "birth_reset_shock": torch.float32,
    "pcg64_state_words": torch.int64,
    "pcg64_buffer_words": torch.int64,
    "mutation_np_semantic_draws": torch.int64,
    "chemistry": torch.float32,
}
_POLICY_NATIVE_DTYPES: Final[dict[str, torch.dtype]] = {
    "mlp_pack": torch.float32,
    "hmem": torch.float32,
    "rng_manager_mutation_torch_cpu_state": torch.uint8,
    "policy_birth_delta_tape": torch.float32,
    "policy_birth_mutation_mask_tape": torch.bool,
    "policy_birth_post_rng_state_packed_tape": torch.uint8,
    "policy_birth_semantic_draws_tape": torch.int64,
    "policy_birth_tape_cursor": torch.int64,
    "mutation_torch_cpu_semantic_draws": torch.int64,
}


@dataclass(slots=True)
class TensorWorldC4ResidentBirthShadowV1:
    """Caller-owned mapping onto the flat Architecture-B birth ABI.

    Tensors that already exist in C4 must be exact aliases, not copies.  Fields
    absent from C4 remain inadmissible even if supplied here: this object is a
    mapping interface, not permission to invent RNG or policy state.
    """

    identity_state: Mapping[str, torch.Tensor]
    scalar_rng_state: Mapping[str, torch.Tensor]
    genome_state: Mapping[str, torch.Tensor]
    drive_state: Mapping[str, torch.Tensor]
    policy_state: Mapping[str, torch.Tensor]
    policy_mem_dim: torch.Tensor
    policy_use_gru: torch.Tensor
    parent_slots: torch.Tensor
    spawn_positions: torch.Tensor | None
    planned_count: torch.Tensor
    schema_version: str = C4_RESIDENT_BIRTH_SHADOW_SCHEMA_VERSION_V1


@dataclass(frozen=True, slots=True)
class TensorWorldC4ResidentBirthCompatibilityV1:
    """Structural admission result; never a residency or gate receipt."""

    compatible: bool
    blockers: tuple[str, ...]
    mapped_native_fields: tuple[str, ...]
    missing_native_fields: tuple[str, ...]
    decision: str = C4_RESIDENT_BIRTH_DECISION_V1
    schema_version: str = C4_RESIDENT_BIRTH_SCHEMA_VERSION_V1


def _row(tensor: torch.Tensor) -> torch.Tensor:
    """Return the B1 native view without allocating or copying."""

    return tensor[0]


def project_c4_resident_birth_shadow_v1(
    bundle: TensorWorldStateBundleV4,
) -> TensorWorldC4ResidentBirthShadowV1:
    """Project every source-owned native field as a direct B1 tensor view.

    The projection is intentionally incomplete where C4 has no source state.
    Validation reports those omissions.  In particular, it does not stack the
    split spawn coordinates or cast parent slots, because both would create a
    per-transaction copy rather than an authoritative alias.
    """

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 resident birth requires a C4 state bundle")
    core = bundle.base.base.core
    c3 = bundle.base
    p = bundle.population

    identity = {
        "slot_to_agent_id": _row(core.agents.agent_id),
        "agent_id_to_slot": _row(p.agent_id_to_slot),
        "slot_tokens": _row(p.slot_tokens),
        "alive_mask": _row(core.agents.alive),
        # The native ABI is int32.  C3 owns int32 mirrors for these two fields,
        # while the C1 biological authority remains int64; compatibility keeps
        # the mirror mismatch explicit below.
        "pos_y": _row(core.agents.pos_y),
        "pos_x": _row(core.agents.pos_x),
        "agent_grid": _row(c3.base.patch.agent_grid_i32),
        "energy": _row(core.agents.energy),
        "health": _row(core.agents.health),
        "injury": _row(c3.agent.injury),
        "scar": _row(p.scar),
        "age": _row(c3.agent.age),
        "repro_cooldown": _row(p.repro_cooldown),
        "repro_threshold": _row(c3.agent.repro_threshold),
        "offspring_count": _row(p.offspring_count),
        "lineage_founder_id": _row(p.lineage_founder_id),
        "lineage_depth": _row(p.lineage_depth),
        "lineage_birth_generation": _row(p.lineage_birth_generation),
        "lineage_birth_tick": _row(p.lineage_birth_tick),
        "patch_habitat_lineage_id": _row(p.patch_habitat_lineage_id),
        "policy_energy_observation": _row(c3.agent.policy_energy_observation),
        "survival_steps": _row(c3.agent.survival_steps),
        "rest_streak": _row(c3.agent.rest_streak),
        "action_cache_token": _row(c3.policy_runtime.action_cache_token),
        "action_cache_cursor": _row(c3.policy_runtime.action_cache_cursor),
        "action_cache_next": _row(c3.policy_runtime.action_cache_next),
        "slot_free_queue": _row(p.free_queue),
        "slot_free_queue_count": p.free_queue_count,
        "population_next_id": p.next_agent_id,
        "last_action": _row(c3.agent.last_action),
        "last_last_action": _row(c3.agent.last_last_action),
        "action_switches": _row(c3.agent.action_switches),
        "action_oscillations": _row(c3.agent.action_oscillations),
        "episode0_switches": _row(c3.agent.episode0_switches),
        "episode0_oscillations": _row(c3.agent.episode0_oscillations),
    }
    scalar_rng: dict[str, torch.Tensor] = {}
    if bundle.scalar_rng is not None:
        scalar_rng = {
            name: (
                _row(getattr(bundle.scalar_rng, name))
                if getattr(bundle.scalar_rng, name).ndim == 2
                else getattr(bundle.scalar_rng, name)
            )
            for name in SCALAR_RNG_STATE_FIELDS
        }

    genome_aliases = {
        "genome_hidden_size": c3.genome.hidden_size,
        "genome_memory_dim": c3.genome.memory_dim,
        "genome_sensor_channel_mask": c3.genome.sensor_channel_mask,
        "genome_sensor_radius": c3.genome.sensor_radius,
        "genome_introspection_channel_mask": c3.genome.introspection_channel_mask,
        "genome_self_regulation_control_mask": c3.genome.self_regulation_control_mask,
        "genome_self_modification_target_mask": c3.genome.self_modification_target_mask,
        "genome_directed_inheritance_channel_mask": c3.genome.directed_inheritance_channel_mask,
        "genome_exploration_bias": c3.genome.exploration_bias,
        "genome_aggression_bias": c3.genome.aggression_bias,
        "genome_mutation_rate": c3.genome.mutation_rate,
        "genome_reproduction_threshold": c3.genome.reproduction_threshold,
        "genome_action_temperature": c3.genome.action_temperature,
        "genome_drive_inertia": c3.genome.drive_inertia,
        "genome_drive_noise": c3.genome.drive_noise,
        "genome_drive_logit_gain": c3.genome.drive_logit_gain,
        "genome_pred_lr": c3.genome.pred_lr,
        "genome_drive_sp_hunger": c3.genome.genome_drive_sp_hunger,
        "genome_drive_sp_pain": c3.genome.genome_drive_sp_pain,
        "genome_drive_sp_fatigue": c3.genome.genome_drive_sp_fatigue,
        "genome_drive_sp_fear": c3.genome.genome_drive_sp_fear,
        "genome_exp_prior_strength": c3.genome.exp_prior_strength,
        "genome_exp_prior_noise": c3.genome.exp_prior_noise,
        "genome_drive_birth_noise": c3.genome.drive_birth_noise,
        "repro_threshold": c3.agent.repro_threshold,
        "gru_upkeep_cost": c3.agent.gru_upkeep_cost,
        "policy_attack_bias": core.policy.attack_bias,
        "policy_action_temperature": core.policy.action_temperature,
        "policy_drive_logit_gain": core.policy.drive_logit_gain,
        "drive_inertia": c3.agent.drive_inertia,
        "drive_noise": c3.agent.drive_noise,
        "pred_lr": c3.genome.pred_lr,
    }
    genome = {name: _row(genome_aliases[name]) for name in genome_aliases}

    return TensorWorldC4ResidentBirthShadowV1(
        identity_state=identity,
        scalar_rng_state=scalar_rng,
        genome_state=genome,
        drive_state={
            "drives": _row(c3.agent.drives),
            "chemistry": _row(c3.adaptation.chemistry),
        },
        policy_state={"mlp_pack": _row(core.policy.packed_weights)},
        policy_mem_dim=_row(c3.genome.policy_mem_dim),
        policy_use_gru=_row(c3.genome.policy_use_gru),
        parent_slots=_row(p.parent_slots),
        spawn_positions=None,
        planned_count=p.planned_count,
    )


def _same_view(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.dtype == expected.dtype
        and actual.device == expected.device
        and tuple(actual.shape) == tuple(expected.shape)
        and tuple(actual.stride()) == tuple(expected.stride())
        and actual.storage_offset() == expected.storage_offset()
        and actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
    )


def _append_mapping_checks(
    blockers: list[str],
    *,
    group: str,
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    required: tuple[str, ...],
    native_dtypes: Mapping[str, torch.dtype],
    device: torch.device,
) -> None:
    actual_keys = set(actual)
    required_keys = set(required)
    for name in sorted(required_keys - actual_keys):
        blockers.append(f"missing_source_state:{group}.{name}")
    for name in sorted(actual_keys - required_keys):
        blockers.append(f"unexpected_shadow_field:{group}.{name}")
    for name in required:
        tensor = actual.get(name)
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            blockers.append(f"not_tensor:{group}.{name}")
            continue
        if tensor.device.type != "cuda" or tensor.device != device:
            blockers.append(f"non_cuda_state:{group}.{name}")
        if not tensor.is_contiguous() or tensor.requires_grad:
            blockers.append(f"invalid_tensor_layout:{group}.{name}")
        native_dtype = native_dtypes[name]
        if tensor.dtype != native_dtype:
            blockers.append(
                f"dtype_mismatch:{group}.{name}:native_requires_{native_dtype}"
            )
        source = expected.get(name)
        if source is None:
            blockers.append(f"missing_authoritative_source:{group}.{name}")
            continue
        if tensor.dtype != source.dtype:
            blockers.append(f"dtype_mismatch:{group}.{name}")
        elif not _same_view(tensor, source):
            blockers.append(f"aliasing_mismatch:{group}.{name}")


def inspect_c4_resident_birth_compatibility_v1(
    bundle: TensorWorldStateBundleV4,
    shadow: TensorWorldC4ResidentBirthShadowV1,
    config: TensorWorldC4BirthConfigV1,
    *,
    mate_slots: torch.Tensor | None = None,
    recombination: bool = False,
) -> TensorWorldC4ResidentBirthCompatibilityV1:
    """Inspect exact R2/native compatibility without executing or syncing."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 resident birth requires a C4 state bundle")
    if not isinstance(shadow, TensorWorldC4ResidentBirthShadowV1):
        raise TensorWorldContractError(
            "C4 resident birth requires a caller-owned shadow"
        )
    if not isinstance(config, TensorWorldC4BirthConfigV1):
        raise TensorWorldContractError("C4 resident birth requires a C4 birth config")
    blockers: list[str] = []
    if shadow.schema_version != C4_RESIDENT_BIRTH_SHADOW_SCHEMA_VERSION_V1:
        blockers.append("unsupported_shadow_schema")
    if bundle.batch != 1:
        blockers.append("native_multicell_unsupported")
    if bundle.device.type != "cuda":
        blockers.append("non_cuda_state:bundle")
    if mate_slots is not None or recombination:
        blockers.append("unsupported_mate_or_recombination")
    if not config.allow_mutation:
        blockers.append("native_scalar_rng_consumption_mismatch")
    blockers.extend(C4_RESIDENT_BIRTH_STATIC_BLOCKERS_V1)

    expected = project_c4_resident_birth_shadow_v1(bundle)
    _append_mapping_checks(
        blockers,
        group="identity",
        actual=shadow.identity_state,
        expected=expected.identity_state,
        required=IDENTITY_STATE_FIELDS,
        native_dtypes=_IDENTITY_NATIVE_DTYPES,
        device=bundle.device,
    )
    _append_mapping_checks(
        blockers,
        group="scalar_rng",
        actual=shadow.scalar_rng_state,
        expected=expected.scalar_rng_state,
        required=SCALAR_RNG_STATE_FIELDS,
        native_dtypes=_SCALAR_RNG_NATIVE_DTYPES,
        device=bundle.device,
    )
    _append_mapping_checks(
        blockers,
        group="genome",
        actual=shadow.genome_state,
        expected=expected.genome_state,
        required=(*GENOME_INT32_FIELDS, *GENOME_FLOAT32_FIELDS),
        native_dtypes=_GENOME_NATIVE_DTYPES,
        device=bundle.device,
    )
    _append_mapping_checks(
        blockers,
        group="drive",
        actual=shadow.drive_state,
        expected=expected.drive_state,
        required=DRIVE_STATE_FIELDS,
        native_dtypes=_DRIVE_NATIVE_DTYPES,
        device=bundle.device,
    )
    _append_mapping_checks(
        blockers,
        group="policy",
        actual=shadow.policy_state,
        expected=expected.policy_state,
        required=C4_RESIDENT_BIRTH_POLICY_FIELDS_V1,
        native_dtypes=_POLICY_NATIVE_DTYPES,
        device=bundle.device,
    )
    for name in ("policy_mem_dim", "policy_use_gru", "parent_slots", "planned_count"):
        tensor = getattr(shadow, name)
        source = getattr(expected, name)
        if not isinstance(tensor, torch.Tensor):
            blockers.append(f"not_tensor:{name}")
        elif tensor.device.type != "cuda" or tensor.device != bundle.device:
            blockers.append(f"non_cuda_state:{name}")
        elif tensor.dtype != source.dtype:
            blockers.append(f"dtype_mismatch:{name}")
        elif not _same_view(tensor, source):
            blockers.append(f"aliasing_mismatch:{name}")
    if shadow.parent_slots.dtype != torch.int64:
        blockers.append("dtype_mismatch:parent_slots_native_int64")
    if shadow.spawn_positions is None:
        blockers.append("missing_source_state:plan.spawn_positions_interleaved")
    else:
        if shadow.spawn_positions.device.type != "cuda":
            blockers.append("non_cuda_state:spawn_positions")
        if shadow.spawn_positions.dtype != torch.int32:
            blockers.append("dtype_mismatch:spawn_positions")
        blockers.append("missing_authoritative_source:plan.spawn_positions_interleaved")

    unique = tuple(dict.fromkeys(blockers))
    missing = tuple(
        blocker.split(":", 1)[1]
        for blocker in unique
        if blocker.startswith(
            ("missing_source_state:", "missing_authoritative_source:")
        )
    )
    mapped = tuple(
        f"{group}.{name}"
        for group, mapping in (
            ("identity", shadow.identity_state),
            ("scalar_rng", shadow.scalar_rng_state),
            ("genome", shadow.genome_state),
            ("drive", shadow.drive_state),
            ("policy", shadow.policy_state),
        )
        for name in mapping
    )
    return TensorWorldC4ResidentBirthCompatibilityV1(
        compatible=not unique,
        blockers=unique,
        mapped_native_fields=mapped,
        missing_native_fields=missing,
    )


def validate_c4_resident_birth_compatibility_v1(
    bundle: TensorWorldStateBundleV4,
    shadow: TensorWorldC4ResidentBirthShadowV1,
    config: TensorWorldC4BirthConfigV1,
    *,
    mate_slots: torch.Tensor | None = None,
    recombination: bool = False,
) -> TensorWorldC4ResidentBirthCompatibilityV1:
    """Require exact compatibility or raise before the native helper loads."""

    result = inspect_c4_resident_birth_compatibility_v1(
        bundle,
        shadow,
        config,
        mate_slots=mate_slots,
        recombination=recombination,
    )
    if not result.compatible:
        raise TensorWorldContractError(
            "C4 resident birth compatibility failed: " + "; ".join(result.blockers)
        )
    return result


def execute_c4_resident_birth_transaction_cuda_v1(
    bundle: TensorWorldStateBundleV4,
    shadow: TensorWorldC4ResidentBirthShadowV1,
    config: TensorWorldC4BirthConfigV1,
    *,
    sensor_radius_max: int,
    mate_slots: torch.Tensor | None = None,
    recombination: bool = False,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Execute only after exact structural admission; there is no fallback."""

    if type(sensor_radius_max) is not int or sensor_radius_max < 0:
        raise TensorWorldContractError("C4 resident birth sensor_radius_max is invalid")
    validate_c4_resident_birth_compatibility_v1(
        bundle,
        shadow,
        config,
        mate_slots=mate_slots,
        recombination=recombination,
    )
    assert shadow.spawn_positions is not None  # guaranteed by admission
    return execute_fixed_mlp_birth_transaction_cuda(
        identity_state=shadow.identity_state,
        parent_slots=shadow.parent_slots,
        spawn_positions=shadow.spawn_positions,
        planned_count=shadow.planned_count,
        scalar_rng_state=shadow.scalar_rng_state,
        genome_state=shadow.genome_state,
        policy_mem_dim=shadow.policy_mem_dim,
        policy_use_gru=shadow.policy_use_gru,
        drive_state=shadow.drive_state,
        policy_state=shadow.policy_state,
        parent_energy_share=config.parent_energy_share,
        minimum_energy_buffer=config.minimum_energy_buffer,
        start_health=config.start_health,
        cooldown_steps=config.cooldown_steps,
        generation=config.generation,
        tick=int(bundle.base.tick),
        sensor_radius_max=sensor_radius_max,
        max_energy=config.max_energy,
        build_directory=build_directory,
    )


__all__ = [
    "C4_RESIDENT_BIRTH_DECISION_V1",
    "C4_RESIDENT_BIRTH_POLICY_FIELDS_V1",
    "C4_RESIDENT_BIRTH_SCHEMA_VERSION_V1",
    "C4_RESIDENT_BIRTH_SHADOW_SCHEMA_VERSION_V1",
    "C4_RESIDENT_BIRTH_STATIC_BLOCKERS_V1",
    "C4_RESIDENT_BIRTH_UNRESOLVED_SOURCE_FIELDS_V1",
    "TensorWorldC4ResidentBirthCompatibilityV1",
    "TensorWorldC4ResidentBirthShadowV1",
    "execute_c4_resident_birth_transaction_cuda_v1",
    "inspect_c4_resident_birth_compatibility_v1",
    "project_c4_resident_birth_shadow_v1",
    "validate_c4_resident_birth_compatibility_v1",
]
