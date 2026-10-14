"""Allocate an exact Patch P2 configurable-birth direct-resident CUDA source."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
import dataclasses
import math
from types import MappingProxyType
from typing import Any, Final

import torch

from engine.campaigns.tick_native_control import (
    EVENT_ROW_STORAGE_BYTES_V1,
    PHYSICAL_B136_BATCH_CELLS_V1,
    PHYSICAL_B136_GRID_V1,
    PHYSICAL_B136_MAX_BIRTHS_PER_TICK_V1,
    PHYSICAL_B136_POPULATION_V1,
    PHYSICAL_B136_RESIDENT_CAPACITY_V1,
    PHYSICAL_B136_SOURCE_WINDOW_TICKS_V1,
    physical_b136_event_window_capacity_v1,
)
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT as TW_RNGV2_NAMESPACE_DEFAULT
from engine.tw_rngv2.world_identity_v1 import (
    LOGICAL_WORLD_UID_DERIVATION_V1,
    logical_world_uid_v1,
)
from evolution.diploid_inheritance_v1 import DIPLOID_INHERITANCE_PROFILE
from evolution.genome_upgrade_v2_package_v1 import (
    validate_checkpoint_authority_binding_v1,
)
from evolution.genome_upgrade_v2_causal_cuda_output_v1 import (
    RECORD_WIDTH as GENOME_V2_CAUSAL_RECORD_WIDTH_V1,
)
from perf.cuda_diploid_state_bank_v1 import (
    CudaDiploidStateBankV1,
    checkpoint_metadata_v1,
    migrate_homozygous_from_phenotype_v1,
    pack_canonical_digest_templates_v1,
    restore_cuda_diploid_state_bank_v1,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
    GENOME_V2_FOUNDER_PROFILE_SCHEMA,
    resolve_genome_v2_founder_profile_v1,
    restore_cuda_genome_upgrade_v2_state_bank,
)
from perf.cuda_architecture_b_boundary import (
    CHECKPOINT_PROJECTION_SCHEMA_VERSION,
    EVENT_COLUMN_NAMES,
)
from perf.cuda_architecture_b_flattened_birth_scratch_layout_v2 import (
    BirthScratchSegmentV2,
    FlatBirthScratchLayoutV2,
    make_legacy_tail_birth_scratch_layout_v2,
)
from perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_ENCODING_V1,
    TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
    torch_cpu_rng_state_abi_receipt_v1,
)
from perf.cxxbackbone10_resident_state_owner import (
    _with_checkpoint_contract_metadata,
    projection_digest,
)
from perf.inheritance_sidecar_identity_v1 import (
    INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
    inheritance_sidecar_payload_sha256_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    BASE_BATCHED_FIELDS,
    GENERATION_SUMMARY_WIDTH_V1,
    ORDINARY_BIRTH_DRIVE_FIELDS,
    ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
    ORDINARY_BIRTH_GENOME_INT32_FIELDS,
    ORDINARY_BIRTH_IDENTITY_FIELDS,
    ORDINARY_BIRTH_OUTPUT_COUNT_V1,
    ORDINARY_BIRTH_POLICY_FIELDS,
    ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
    PATCH_P2_CONFIG_ARGUMENTS_V1,
    PATCH_P2_STATE_FIELDS_V1,
    PATCH_RESIDENT_FIELDS,
    PATCH_STAT_COUNT_V1,
    TURNOVER_STATE_FIELDS,
    DirectResidentPreparationError,
    DirectResidentSourceV1,
    required_lifecycle_scratch_bytes_v1,
    validate_asexual_birth_bound_v1,
)


def _logical_world_uid_v1(*, configured_world_uid: int, cell_id: str) -> int:
    """Derive a partition-independent uint32 RNG world coordinate."""

    try:
        return logical_world_uid_v1(
            configured_world_uid=configured_world_uid,
            cell_id=cell_id,
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc

INT32_MAX: Final[int] = (1 << 31) - 1
INT64_MAX: Final[int] = (1 << 63) - 1


BATCH_CELLS_V1: Final[int] = 1
LOGICAL_GENERATIONS_V1: Final[int] = 4
PHYSICAL_GENERATIONS_V1: Final[int] = 2
TICKS_PER_GENERATION_V1: Final[int] = 80
ADMITTED_TICKS_PER_GENERATION_V1: Final[tuple[int, ...]] = (80, 160)
MAX_TICKS_PER_GENERATION_V1: Final[int] = 160
OUTPUT_TICKS_V1: Final[int] = 1
FIRST_FAILURE_RECORD_WIDTH_V1: Final[int] = 10
LIFECYCLE_SCRATCH_BYTES_V1: Final[int] = 163_840
PHYSICAL_WINDOW_GENERATIONS_V1: Final[int] = 2
# The direct source ABI still has a minimum width for its private identity
# shadow.  This is an allocation bound, not a lifecycle or time authority.
PRIVATE_SOURCE_CAPACITY_MIN_WINDOWS_V1: Final[int] = 2
PHYSICAL_WINDOW_TICKS_V1: Final[int] = (
    PHYSICAL_WINDOW_GENERATIONS_V1 * MAX_TICKS_PER_GENERATION_V1
)
TURNOVER_EPISODES_PER_GENERATION_V1: Final[int] = 62
PHYSICAL_WINDOW_TURNOVER_EPISODES_V1: Final[int] = (
    PHYSICAL_WINDOW_GENERATIONS_V1 * TURNOVER_EPISODES_PER_GENERATION_V1
)
QUALIFIED_PATCH_GEOMETRIES_V1: Final[frozenset[tuple[int, tuple[int, int]]]] = (
    frozenset(
        {
            (64, (16, 16)),
            (128, (16, 16)),
            (256, (32, 32)),
            (512, (32, 32)),
        }
    )
)
LEGACY_DENSE_IDENTITY_BYTES_PER_CELL_ID_V1: Final[int] = (
    4 + 8  # agent_id_to_slot int32  # initial_id_to_row int64
)
LEGACY_DENSE_IDENTITY_ALLOCATION_BUDGET_BYTES_V1: Final[int] = 256 * 1024 * 1024
RESIDENT_AGENT_ID_CAPACITY_METADATA_V1: Final[str] = (
    "native_runtime_resident_agent_id_capacity_v1"
)
RAGGED_PROJECTION_FIELDS_V1: Final[Mapping[str, torch.dtype]] = MappingProxyType(
    {
        "agent_id_to_slot": torch.int32,
        "agent_id_to_slot_agent_ids": torch.int64,
        "agent_id_to_slot_slots": torch.int32,
        "agent_ids_by_alive_order": torch.int64,
        "slot_free_queue": torch.int64,
    }
)


def _genome_v2_checkpoint_authority_binding_v1(
    config: Any,
) -> tuple[tuple[str, str], ...]:
    raw = tuple(getattr(config, "genome_v2_checkpoint_authority_binding", ()))
    if not raw:
        return ()
    return tuple(sorted(validate_checkpoint_authority_binding_v1(raw).items()))


def _disable_genome_v2_qualification_fault_v1(
    scratch: torch.Tensor,
    layout: FlatBirthScratchLayoutV2,
) -> None:
    fault_segment = layout.segment(
        BirthScratchSegmentV2.GENOME_UPGRADE_V2_QUALIFICATION_FAULT_STAGE
    )
    scratch[
        :,
        fault_segment.offset : fault_segment.end_offset,
    ].view(torch.int32).zero_()


TURNOVER_MUTABLE_STATE_FIELDS: Final[tuple[str, ...]] = (
    "active_agent_ids",
    "active_agent_slots",
    "active_count_state",
    "slot_to_agent_id",
    "agent_id_to_slot",
    "slot_tokens",
    "alive_mask",
    "pos_y",
    "pos_x",
    "agent_grid",
    "energy",
    "health",
    "injury",
    "policy_energy_observation",
    "rest_streak",
    "pending_free_agent_ids",
    "pending_free_slots",
    "pending_free_count",
    "action_cache_actions",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "last_action",
    "last_last_action",
    "action_switches",
    "action_oscillations",
    "episode0_switches",
    "episode0_oscillations",
    "survival_steps",
    "slot_free_queue",
    "slot_free_queue_count",
    "population_next_id",
    "scar",
    "age",
    "repro_cooldown",
    "repro_threshold",
    "offspring_count",
    "lineage_founder_id",
    "lineage_depth",
    "lineage_birth_generation",
    "lineage_birth_tick",
    "patch_habitat_lineage_id",
    "python_random_words",
    "python_random_index",
    "python_random_has_gauss",
    "python_random_gauss_next",
    "python_random_raw_draws",
    "mutation_py_semantic_draws",
    *ORDINARY_BIRTH_GENOME_INT32_FIELDS,
    "policy_mem_dim",
    "policy_use_gru",
    *ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS[:16],
    "gru_upkeep_cost",
    "policy_attack_bias",
    "policy_action_temperature",
    "policy_drive_logit_gain",
    "drive_inertia",
    "drive_noise",
    "pred_lr",
    *ORDINARY_BIRTH_DRIVE_FIELDS,
    "mlp_pack",
    "hmem",
    "rng_manager_mutation_torch_cpu_state",
    "policy_birth_tape_cursor",
    "mutation_torch_cpu_semantic_draws",
    *TURNOVER_STATE_FIELDS,
    "prev_drive_delta",
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "selfharm_events",
    "episode0_selfharm_events",
    "pcg64_state_words",
    "pcg64_buffer_words",
    "policy_np_semantic_draws",
)
TURNOVER_OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "turnover_ranked_agent_ids",
    "turnover_culled_agent_ids",
    "turnover_parent_pool_agent_ids",
    "turnover_refill_parent_agent_ids",
    "turnover_refill_mate_agent_ids",
    "turnover_child_agent_ids",
    "turnover_child_slots",
    "turnover_counts",
    "turnover_status",
    "turnover_next_generation",
    "turnover_next_tick",
    "turnover_genome_v2_causal_record_rows",
    "turnover_genome_v2_causal_record_valid",
    "turnover_genome_v2_causal_record_count",
    "turnover_genome_v2_causal_record_dropped_count",
    "turnover_genome_v2_causal_capture_enabled",
)
ORDINARY_BIRTH_OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "birth_slot_free_queue_head",
    "birth_planned_repro_cooldown",
    "birth_planned_parent_slots",
    "birth_planned_child_slots",
    "birth_planned_spawn_positions",
    "birth_planned_child_energy",
    "birth_decision_parent_slots",
    "birth_decision_reason_codes",
    "birth_decision_candidate_counts",
    "birth_decision_draw_starts",
    "birth_decision_draw_ends",
    "birth_decision_selected_positions",
    "birth_decision_selected_reserved",
    "birth_reason_counts",
    "birth_planned_count",
    "birth_decision_count",
    "birth_spawn_rng_draws",
    "birth_spawn_rng_semantic_draws",
    "birth_runner_rng_draws",
    "birth_runner_rng_semantic_draws",
    "birth_plan_status",
    "birth_committed_parent_agent_ids",
    "birth_committed_child_agent_ids",
    "birth_committed_child_slots",
    "birth_identity_committed_count",
    "birth_identity_status",
    "birth_scalar_committed_count",
    "birth_scalar_semantic_draws",
    "birth_scalar_status",
    "birth_drive_committed_count",
    "birth_noise_rows_consumed",
    "birth_exp_prior_rows_consumed",
    "birth_standard_normal_rows",
    "birth_drive_semantic_draws",
    "birth_drive_status",
    "birth_chemistry_status",
    "birth_policy_committed_count",
    "birth_policy_semantic_draws",
    "birth_policy_mutation_events",
    "birth_policy_mutated_elements",
    "birth_policy_mutation_delta_total",
    "birth_policy_mutation_occurred",
    "birth_policy_mutation_delta",
    "birth_policy_status",
    "birth_combined_status",
    "birth_planner_pcg64_state_before",
    "birth_planner_pcg64_buffer_before",
    "birth_planned_mate_slots",
    "birth_committed_mate_agent_ids",
    "birth_mate_planned_energy",
    "birth_mate_planned_child_energy_exact",
    "birth_mate_planned_parent_energy_after",
    "birth_mate_planned_mate_energy_after",
    "birth_decision_mate_slots",
    "birth_decision_adjacent_counts",
    "birth_decision_spawn_candidate_counts",
    "birth_mate_reason_counts",
    "birth_mate_decision_reason_codes",
    "birth_genome_v2_causal_record_rows",
    "birth_genome_v2_causal_record_valid",
    "birth_genome_v2_causal_record_count",
    "birth_genome_v2_causal_record_dropped_count",
    "birth_genome_v2_causal_capture_enabled",
)

GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1: Final[int] = 1024

CANONICAL_RUNTIME_PREFIX_FIELDS: Final[tuple[str, ...]] = (
    "restore_count",
    "generation_calls",
    "event_cursor",
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


class CudaArchitectureBDirectResidentSourceAllocatorError(RuntimeError):
    """The exact engine-owned device allocation cannot be constructed."""


def _genome_v2_observability_config_v1(
    config: Any,
    patch_config: Mapping[str, Any],
) -> dict[str, int | float | bool]:
    return {
        "max_energy": float(config.max_energy),
        "start_health": float(config.start_health),
        "developmental_maturity_age": int(config.developmental_maturity_age),
        "patch_density_radius": int(patch_config["patch_density_radius"]),
        "patch_density_include_self": bool(patch_config["patch_density_include_self"]),
        "patch_density_normalize": bool(patch_config["patch_density_normalize"]),
        "patch_food_amount_max": float(patch_config["patch_food_amount_max"]),
        "patch_waste_max": float(patch_config["patch_waste_max"]),
    }


def _fail(message: str) -> CudaArchitectureBDirectResidentSourceAllocatorError:
    return CudaArchitectureBDirectResidentSourceAllocatorError(message)


_INHERITANCE_SIDECAR_FIELDS_V1: Final[tuple[str, ...]] = (
    "diploid_checkpoint_projection",
    "genome_upgrade_v2_checkpoint_projection",
)


def _projections_in_cell_identity_order_v1(
    projections: Sequence[Mapping[str, Any]],
    *,
    field: str,
    cell_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...] | None:
    payloads = [item.get(field) for item in projections]
    if all(payload is None for payload in payloads):
        return None
    if any(payload is None for payload in payloads):
        raise _fail(f"{field} sidecars differ across cells")
    if (
        len(projections) != len(cell_ids)
        or len(set(cell_ids)) != len(cell_ids)
        or any(type(cell_id) is not str or not cell_id for cell_id in cell_ids)
    ):
        raise _fail(f"{field} expected cell identity set is invalid")
    by_cell_id: dict[str, Mapping[str, Any]] = {}
    for projection, payload in zip(projections, payloads, strict=True):
        if not isinstance(payload, Mapping):
            raise _fail(f"{field} sidecar must be a mapping")
        identity = payload.get("identity")
        if not isinstance(identity, Mapping):
            raise _fail(f"{field} sidecar identity is missing")
        cell_id = identity.get("cell_id")
        if type(cell_id) is not str or not cell_id:
            raise _fail(f"{field} sidecar cell identity is invalid")
        if cell_id in by_cell_id:
            raise _fail(f"{field} duplicate sidecar cell identity")
        by_cell_id[cell_id] = projection
    if set(by_cell_id) != set(cell_ids):
        raise _fail(f"{field} identity mismatch: sidecar cell identity set differs")
    return tuple(by_cell_id[cell_id] for cell_id in cell_ids)


def _is_streamed_bwide_seed_v1(
    *,
    cell_requests: Sequence[Any] | None,
    per_cell_ids: Sequence[str] | None,
    batch_cells: int,
    canonical_projection_count: int,
) -> bool:
    """Identify the bounded B-wide seed used by the tick-native streamer.

    The streamer deliberately supplies one B1 request at a time, so the first
    allocation has one canonical projection and an explicit ordered ID list
    for the eventual B-wide owner.  That shape is not a valid multicell
    checkpoint restore input: the owner is seeded only for allocation shape,
    then every B1 row is copied into it.  Ordinary multicell callers continue
    through the exact sidecar identity-set validation above.
    """

    return (
        cell_requests is None
        and per_cell_ids is not None
        and batch_cells > 1
        and canonical_projection_count == 1
    )


def _validated_inheritance_sidecar_v1(
    payload: Any,
    *,
    canonical_projection: Mapping[str, Any],
    expected_cell_id: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise _fail("inheritance checkpoint sidecar must be a mapping")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise _fail("inheritance checkpoint sidecar identity is missing")
    metadata = canonical_projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("inheritance checkpoint canonical metadata is missing")
    identity_metadata = dict(metadata)
    if identity_metadata.get("hapb_policy_restore_pending_cache_rehydrate") is True:
        # HAPB marks its restored source solely to authorize adoption of the
        # one shared host policy bank.  Those transport fields are removed
        # before the next terminal projection, so sidecar validation must use
        # the same canonical identity base used for rebind and projection.
        if not isinstance(
            identity_metadata.get("hapb_policy_checkpoint_sha256"), str
        ) or not isinstance(
            identity_metadata.get("hapb_policy_checkpoint_schema_version"), str
        ):
            raise _fail("HAPB inheritance transport metadata is malformed")
        for name in (
            "hapb_policy_checkpoint_sha256",
            "hapb_policy_checkpoint_schema_version",
            "hapb_policy_restore_pending_cache_rehydrate",
        ):
            identity_metadata.pop(name, None)
    base_projection = dict(canonical_projection)
    base_projection["metadata"] = identity_metadata
    for name in _INHERITANCE_SIDECAR_FIELDS_V1:
        base_projection.pop(name, None)
    raw_payload = dict(payload)
    raw_payload.pop("identity", None)
    expected = {
        "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
        "cell_id": expected_cell_id,
        "seed": identity_metadata.get("seed"),
        "generation": identity_metadata.get("generation"),
        "tick": identity_metadata.get("tick"),
        "base_projection_sha256": projection_digest(base_projection),
        "payload_sha256": inheritance_sidecar_payload_sha256_v1(raw_payload),
    }
    if dict(identity) != expected:
        raise _fail(
            f"inheritance checkpoint sidecar identity mismatch for "
            f"cell_id={expected_cell_id!r}"
        )
    return payload


def _diploid_bank_from_canonical_projections_v1(
    projections: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    cell_ids: Sequence[str],
) -> CudaDiploidStateBankV1 | None:
    ordered = _projections_in_cell_identity_order_v1(
        projections,
        field="diploid_checkpoint_projection",
        cell_ids=cell_ids,
    )
    if ordered is None:
        return None
    payloads = [item["diploid_checkpoint_projection"] for item in ordered]
    try:
        banks = [
            restore_cuda_diploid_state_bank_v1(
                _validated_inheritance_sidecar_v1(
                    payload,
                    canonical_projection=projection,
                    expected_cell_id=cell_id,
                ),
                device=device,
            )
            for payload, projection, cell_id in zip(
                payloads, ordered, cell_ids, strict=True
            )
        ]
        if any(bank.cells != 1 for bank in banks):
            raise _fail("each diploid checkpoint sidecar must own exactly one cell")
        template_widths = {
            (
                int(bank.canonical_hereditary_template.size(1)),
                int(bank.diploid_payload_template.size(1)),
                int(bank.diploid_genotype_template.size(1)),
            )
            for bank in banks
        }
        if len(template_widths) != 1:
            raise _fail("diploid checkpoint template widths differ across cells")
        capacity = banks[0].capacity
        if any(bank.capacity != capacity for bank in banks):
            raise _fail("diploid checkpoint capacities differ across cells")
        return CudaDiploidStateBankV1(
            storage=torch.cat([bank.storage for bank in banks], dim=0).contiguous(),
            genome_resource_handling_axis=torch.cat(
                [bank.genome_resource_handling_axis for bank in banks], dim=0
            ).contiguous(),
            canonical_hereditary_digest=torch.cat(
                [bank.canonical_hereditary_digest for bank in banks], dim=0
            ).contiguous(),
            canonical_hereditary_template=torch.cat(
                [bank.canonical_hereditary_template for bank in banks], dim=0
            ).contiguous(),
            diploid_payload_template=torch.cat(
                [bank.diploid_payload_template for bank in banks], dim=0
            ).contiguous(),
            diploid_genotype_template=torch.cat(
                [bank.diploid_genotype_template for bank in banks], dim=0
            ).contiguous(),
            metadata=checkpoint_metadata_v1(cells=len(banks), capacity=capacity),
        )
    except CudaArchitectureBDirectResidentSourceAllocatorError:
        raise
    except Exception as exc:
        raise _fail(f"diploid checkpoint sidecars are incompatible: {exc}") from exc


_FRESH_DIPLOID_SOURCE_FIELDS_V1: Final[tuple[str, ...]] = (
    "diploid_allele_storage",
    "genome_resource_handling_axis",
    "canonical_hereditary_digest",
    "canonical_hereditary_template",
    "diploid_payload_template",
    "diploid_genotype_template",
)


def _diploid_bank_from_fresh_sources_v1(
    source: Mapping[str, Any],
    *,
    sources: Sequence[Mapping[str, Any]] | None,
    device: torch.device,
    batch_cells: int,
    source_groups: Sequence[int] | None,
) -> CudaDiploidStateBankV1 | None:
    """Retain exact imported diploid state before a checkpoint sidecar exists."""

    source_rows = tuple(sources) if sources is not None else (source,)
    presence = tuple(
        tuple(name in row for name in _FRESH_DIPLOID_SOURCE_FIELDS_V1)
        for row in source_rows
    )
    if not any(any(row) for row in presence):
        return None
    if any(not all(row) for row in presence):
        raise _fail(
            "fresh diploid resident source must provide the complete allele, "
            "phenotype, digest, and template set"
        )

    def stack(name: str) -> torch.Tensor:
        if sources is None:
            return _stack(
                source[name],
                device=device,
                batch_cells=batch_cells,
            )
        return _stack_per_cell_values_v1(
            [row[name] for row in source_rows],
            name=name,
            device=device,
            source_groups=source_groups,
        )

    try:
        return CudaDiploidStateBankV1(
            storage=stack("diploid_allele_storage"),
            genome_resource_handling_axis=stack("genome_resource_handling_axis"),
            canonical_hereditary_digest=stack("canonical_hereditary_digest"),
            canonical_hereditary_template=stack("canonical_hereditary_template"),
            diploid_payload_template=stack("diploid_payload_template"),
            diploid_genotype_template=stack("diploid_genotype_template"),
            metadata=checkpoint_metadata_v1(
                cells=batch_cells,
                capacity=int(source["energy"].numel()),
            ),
        )
    except CudaArchitectureBDirectResidentSourceAllocatorError:
        raise
    except Exception as exc:
        raise _fail(f"fresh diploid resident state is incompatible: {exc}") from exc


def _genome_v2_bank_from_canonical_projections_v1(
    projections: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    cell_ids: Sequence[str],
    founder_profile: Mapping[str, Any] | None = None,
    founder_profiles: Sequence[Mapping[str, Any]] | None = None,
) -> CudaGenomeUpgradeV2StateBank | None:
    ordered = _projections_in_cell_identity_order_v1(
        projections,
        field="genome_upgrade_v2_checkpoint_projection",
        cell_ids=cell_ids,
    )
    if ordered is None:
        return None
    payloads = [item["genome_upgrade_v2_checkpoint_projection"] for item in ordered]
    if founder_profile is not None and founder_profiles is not None:
        raise _fail("GENOME-UPGRADE-V2 founder profile authority is ambiguous")
    expected_profiles: tuple[Mapping[str, Any] | None, ...]
    if founder_profiles is not None:
        expected_profiles = tuple(founder_profiles)
        if len(expected_profiles) != len(ordered):
            raise _fail("GENOME-UPGRADE-V2 founder profile count differs from cells")
    else:
        expected_profiles = (founder_profile,) * len(ordered)
    try:
        banks = [
            restore_cuda_genome_upgrade_v2_state_bank(
                _validated_inheritance_sidecar_v1(
                    payload,
                    canonical_projection=projection,
                    expected_cell_id=cell_id,
                ),
                device=device,
                expected_founder_profile=expected_profile,
            )
            for payload, projection, cell_id, expected_profile in zip(
                payloads,
                ordered,
                cell_ids,
                expected_profiles,
                strict=True,
            )
        ]
        if any(bank.cells != 1 for bank in banks):
            raise _fail(
                "each GENOME-UPGRADE-V2 checkpoint sidecar must own exactly one cell"
            )
        capacity = banks[0].capacity
        if any(bank.capacity != capacity for bank in banks):
            raise _fail("GENOME-UPGRADE-V2 checkpoint capacities differ across cells")
        tensors = {
            name: torch.cat(
                [bank.named_tensors()[name] for bank in banks], dim=0
            ).contiguous()
            for name in banks[0].named_tensors()
        }
        return CudaGenomeUpgradeV2StateBank(
            alleles_q10=tensors["genome_v2_alleles_q10"],
            modifiers_q10=tensors["genome_v2_modifiers_q10"],
            products_q10=tensors["genome_v2_products_q10"],
            phenotypes_q10=tensors["genome_v2_phenotypes_q10"],
            expression_cost_q10=tensors["genome_v2_expression_cost_q10"],
            active_product_count=tensors["genome_v2_active_product_count"],
            saturated_product_count=tensors["genome_v2_saturated_product_count"],
        )
    except CudaArchitectureBDirectResidentSourceAllocatorError:
        raise
    except Exception as exc:
        raise _fail(
            f"GENOME-UPGRADE-V2 checkpoint sidecars are incompatible: {exc}"
        ) from exc


def _initialize_genome_v2_fresh_founders_v1(
    bank: CudaGenomeUpgradeV2StateBank,
    founder_profiles: Sequence[Mapping[str, Any]],
    *,
    alive_mask: torch.Tensor,
) -> tuple[dict[str, Any], ...]:
    heterogeneous = tuple(
        "founder_population" in profile for profile in founder_profiles
    )
    if all(heterogeneous):
        return bank.initialize_founder_populations_by_cell_v1(
            tuple(
                str(profile["founder_population"]["profile_id"])
                for profile in founder_profiles
            ),
            alive_mask=alive_mask,
        )
    bank.initialize_founders_by_cell_v1(
        tuple(tuple(profile["genotype_q10"]) for profile in founder_profiles),
        alive_mask=alive_mask,
    )
    geometries: list[dict[str, Any]] = []
    for cell, profile in enumerate(founder_profiles):
        if "founder_population" not in profile:
            continue
        geometries.append(
            bank.initialize_founder_population_cell_v1(
                cell,
                str(profile["founder_population"]["profile_id"]),
                alive_mask=alive_mask,
            )
        )
    return tuple(geometries)


def _validated_birth_bound(max_births_per_tick: object, capacity: int) -> int:
    try:
        return validate_asexual_birth_bound_v1(max_births_per_tick, capacity)
    except DirectResidentPreparationError as exc:
        raise _fail(str(exc)) from exc


def _validated_ticks_per_generation_v1(value: object) -> int:
    if type(value) is not int or value not in ADMITTED_TICKS_PER_GENERATION_V1:
        raise _fail(
            "ticks_per_generation must be one of "
            f"{list(ADMITTED_TICKS_PER_GENERATION_V1)!r}"
        )
    return value


def _validated_physical_window_generations_v1(value: object) -> int:
    if type(value) is not int or value < 1 or value > 2:
        raise _fail("physical_window_generations must be an exact integer in [1,2]")
    return value


def physical_window_ticks_v1(physical_window_generations: object) -> int:
    return (
        _validated_physical_window_generations_v1(physical_window_generations)
        * MAX_TICKS_PER_GENERATION_V1
    )


def physical_window_turnover_episodes_v1(
    physical_window_generations: object,
) -> int:
    return (
        _validated_physical_window_generations_v1(physical_window_generations)
        * TURNOVER_EPISODES_PER_GENERATION_V1
    )


def required_physical_window_policy_episodes_v1(
    max_births_per_tick: object,
    *,
    physical_window_generations: object = PHYSICAL_WINDOW_GENERATIONS_V1,
) -> int:
    """Return exact ordinary plus turnover rows for one reusable native window."""

    if type(max_births_per_tick) is not int or max_births_per_tick < 1:
        raise _fail("max_births_per_tick must be an exact positive integer")
    window_generations = _validated_physical_window_generations_v1(
        physical_window_generations
    )
    return physical_window_ticks_v1(
        window_generations
    ) * max_births_per_tick + physical_window_turnover_episodes_v1(window_generations)


def required_physical_window_turnover_episodes_v1(
    slot_capacity: object,
    *,
    ordinary_births_enabled: bool,
    physical_window_generations: object = PHYSICAL_WINDOW_GENERATIONS_V1,
) -> int:
    """Return turnover rows required by the selected physical-window profile."""

    if type(ordinary_births_enabled) is not bool:
        raise _fail("ordinary_births_enabled must be an exact bool")
    if ordinary_births_enabled:
        return physical_window_turnover_episodes_v1(physical_window_generations)
    if type(slot_capacity) is not int or slot_capacity < 1:
        raise _fail("slot_capacity must be a positive exact integer")
    return slot_capacity


def _runtime_turnover_refill_capacity_v1(
    configuration: Any,
    *,
    population_size: object,
    slot_capacity: object,
    ordinary_births_enabled: bool,
    physical_window_generations: object = PHYSICAL_WINDOW_GENERATIONS_V1,
) -> int:
    """Bind the turnover refill rows to the live lifecycle envelope.

    The historical 62-row bound remains the compatibility default for callers
    that do not expose the production lifecycle ABI.  Current legacy turnover
    owners reserve the full population-size refill envelope.  Native turnover
    refills ``population_size - culled_count``; because the live active
    population may be below the configured maximum, a cull-fraction product
    is not a safe upper bound.  Population size is the fail-closed ceiling for
    one turnover transaction.
    """

    if type(ordinary_births_enabled) is not bool:
        raise _fail("ordinary_births_enabled must be an exact bool")
    window_generations = _validated_physical_window_generations_v1(
        physical_window_generations
    )
    lifecycle_mode = getattr(configuration, "lifecycle_mode", None)
    cull_fraction = getattr(configuration, "cull_fraction", None)
    if (
        not ordinary_births_enabled
        or lifecycle_mode != 0
        or type(cull_fraction) is not float
    ):
        return required_physical_window_turnover_episodes_v1(
            slot_capacity,
            ordinary_births_enabled=ordinary_births_enabled,
            physical_window_generations=window_generations,
        )
    if type(population_size) is not int or population_size < 1:
        raise _fail("population_size must be a positive exact integer")
    if type(slot_capacity) is not int or slot_capacity < population_size:
        raise _fail("slot_capacity must cover population_size")
    if not math.isfinite(cull_fraction) or not 0.0 <= cull_fraction <= 1.0:
        raise _fail("cull_fraction must be finite in [0, 1]")
    per_generation = population_size
    return window_generations * per_generation


def legacy_dense_identity_allocation_bytes_v1(
    batch_cells: object,
    agent_id_capacity: object,
) -> int:
    """Return exact bytes for the two legacy biological-ID-indexed tensors."""

    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 72:
        raise _fail("legacy dense identity batch_cells must be an integer in [1, 72]")
    if type(agent_id_capacity) is not int or agent_id_capacity < 1:
        raise _fail(
            "legacy dense identity agent_id_capacity must be a positive integer"
        )
    return batch_cells * agent_id_capacity * LEGACY_DENSE_IDENTITY_BYTES_PER_CELL_ID_V1


def _authorize_legacy_dense_identity_allocation_v1(
    *,
    batch_cells: object,
    agent_id_capacity: object,
) -> int:
    required_bytes = legacy_dense_identity_allocation_bytes_v1(
        batch_cells,
        agent_id_capacity,
    )
    if required_bytes > LEGACY_DENSE_IDENTITY_ALLOCATION_BUDGET_BYTES_V1:
        raise _fail(
            "legacy dense historical-ID CUDA allocation exceeds the authorized "
            "pre-allocation budget: "
            f"required_bytes={required_bytes} "
            f"budget_bytes={LEGACY_DENSE_IDENTITY_ALLOCATION_BUDGET_BYTES_V1} "
            f"batch_cells={batch_cells} agent_id_capacity={agent_id_capacity}"
        )
    return required_bytes


def required_agent_id_capacity_v1(
    current_capacity: object,
    population_next_id: object,
    max_births_per_tick: object,
    slot_capacity: object,
    logical_generation_count: object = LOGICAL_GENERATIONS_V1,
) -> int:
    """Reserve identity rows for all logical births and turnover children."""

    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 3
        or logical_generation_count > 128
    ):
        raise _fail("logical_generation_count must be an integer in [3, 128]")
    return required_resident_agent_id_capacity_v1(
        current_capacity,
        population_next_id,
        max_births_per_tick,
        slot_capacity,
        allocation_generation_count=logical_generation_count,
        retained_capacity=current_capacity,
    )


def required_resident_agent_id_capacity_v1(
    current_capacity: object,
    population_next_id: object,
    max_births_per_tick: object,
    slot_capacity: object,
    *,
    allocation_generation_count: object,
    retained_capacity: object,
) -> int:
    """Return the capacity-bounded compatibility-shadow width.

    ``population_next_id`` is identity state, not an allocation dimension.
    The retained width is allowed only to preserve an already bounded shadow.
    """

    if type(current_capacity) is not int or current_capacity < 1:
        raise _fail("current agent-id capacity must be an exact positive integer")
    if (
        type(population_next_id) is not int
        or population_next_id < 1
        or population_next_id > INT32_MAX + 1
    ):
        raise _fail("population_next_id must be an exact integer in [1, INT32_MAX + 1]")
    if type(slot_capacity) is not int or slot_capacity < 1:
        raise _fail("slot_capacity must be an exact positive integer")
    if (
        type(allocation_generation_count) is not int
        or allocation_generation_count < PHYSICAL_WINDOW_GENERATIONS_V1
        or allocation_generation_count > 128
    ):
        raise _fail(
            "agent_id_allocation_generation_count must be an integer in [2, 128]"
        )
    if type(retained_capacity) is not int or retained_capacity < current_capacity:
        raise _fail(
            f"{RESIDENT_AGENT_ID_CAPACITY_METADATA_V1} must be an exact integer "
            "covering the canonical identity map"
        )
    _validated_birth_bound(max_births_per_tick, slot_capacity)
    if current_capacity > slot_capacity or retained_capacity > slot_capacity:
        raise _fail(
            f"{RESIDENT_AGENT_ID_CAPACITY_METADATA_V1} must not exceed "
            "resident slot capacity"
        )
    return slot_capacity


def _required_resident_agent_id_capacity_v1(
    current_capacity: int,
    population_next_id: int,
    max_births_per_tick: int,
    slot_capacity: int,
    *,
    logical_generation_count: int,
    allocation_generation_count: object,
    retained_capacity: object,
) -> int:
    if allocation_generation_count is None:
        allocation_generation_count = logical_generation_count
    if retained_capacity is None:
        retained_capacity = current_capacity
    return required_resident_agent_id_capacity_v1(
        current_capacity,
        population_next_id,
        max_births_per_tick,
        slot_capacity,
        allocation_generation_count=allocation_generation_count,
        retained_capacity=retained_capacity,
    )


def _validate_birth_state_receipt_v1(
    birth_state: object,
    *,
    max_births_per_tick: int,
    slot_capacity: int,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS_V1,
    ordinary_births_enabled: bool = True,
    turnover_refill_capacity: int | None = None,
) -> None:
    receipt = getattr(birth_state, "receipt", None)
    if type(ordinary_births_enabled) is not bool:
        raise _fail("ordinary_births_enabled must be an exact bool")
    window_ticks = physical_window_ticks_v1(physical_window_generations)
    if turnover_refill_capacity is None:
        turnover_refill_capacity = (
            physical_window_turnover_episodes_v1(physical_window_generations)
            if ordinary_births_enabled
            else required_physical_window_turnover_episodes_v1(
                slot_capacity,
                ordinary_births_enabled=False,
                physical_window_generations=physical_window_generations,
            )
        )
    elif type(turnover_refill_capacity) is not int or turnover_refill_capacity < 0:
        raise _fail("turnover_refill_capacity must be a nonnegative exact integer")
    expected_episode_capacity = (
        window_ticks * max_births_per_tick + turnover_refill_capacity
        if ordinary_births_enabled
        else turnover_refill_capacity
    )
    expected = {
        "steps": window_ticks,
        "max_births_per_tick": max_births_per_tick,
        "slot_capacity": slot_capacity,
        "turnover_refill_capacity": turnover_refill_capacity,
        "fixed_mlp_policy_full_horizon_episodes": expected_episode_capacity,
        "torch_rng_post_state_encoding": TORCH_CPU_RNG_STATE_ENCODING_V1,
        "torch_rng_post_state_logical_row_bytes": (TORCH_CPU_RNG_STATE_FULL_BYTES_V1),
        "torch_rng_post_state_packed_row_bytes": (TORCH_CPU_RNG_STATE_PACKED_BYTES_V1),
        "torch_rng_post_state_dense_allocation_materialized": False,
        "torch_rng_post_state_abi": torch_cpu_rng_state_abi_receipt_v1(),
    }
    segment_refill_enabled = getattr(
        receipt, "policy_birth_segment_refill_enabled", False
    )
    if type(segment_refill_enabled) is not bool:
        raise _fail("policy_birth_segment_refill_enabled must be an exact bool")
    if segment_refill_enabled:
        actual_episode_capacity = getattr(receipt, "birth_episode_capacity", None)
        actual_fixed_mlp_episodes = getattr(
            receipt, "fixed_mlp_policy_tape_episodes", None
        )
        minimum_segment_capacity = max(max_births_per_tick, turnover_refill_capacity)
        if (
            type(actual_episode_capacity) is not int
            or actual_episode_capacity < minimum_segment_capacity
            or actual_episode_capacity > expected_episode_capacity
        ):
            raise _fail(
                "policy birth segment refill capacity is outside the admitted "
                "window bounds"
            )
        if actual_fixed_mlp_episodes != actual_episode_capacity:
            raise _fail(
                "segmented fixed-MLP policy tape episodes must equal the "
                "resident birth episode capacity"
            )
    else:
        expected["birth_episode_capacity"] = expected_episode_capacity
        expected["fixed_mlp_policy_tape_episodes"] = expected_episode_capacity
    tape_free_candidate = (
        getattr(receipt, "cpython_gauss_raw_capacity", None) == 0
        and segment_refill_enabled is False
        and getattr(receipt, "birth_episode_capacity", None) in (0, 1)
        and getattr(receipt, "fixed_mlp_policy_tape_episodes", None)
        == getattr(receipt, "birth_episode_capacity", None)
    )
    if tape_free_candidate:
        stub = int(receipt.birth_episode_capacity)
        expected["birth_episode_capacity"] = stub
        expected["fixed_mlp_policy_tape_episodes"] = stub
        expected["fixed_mlp_policy_full_horizon_episodes"] = stub
    mismatches: list[str] = []
    for name, expected_value in expected.items():
        actual = getattr(receipt, name, None)
        if type(actual) is not type(expected_value) or actual != expected_value:
            mismatches.append(f"{name}:expected={expected_value},actual={actual!r}")
    if mismatches:
        raise _fail(
            "generation birth-state physical-window capacity mismatch: "
            + "; ".join(mismatches)
        )


def _required_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a mapping")
    return dict(value)


_MISSING: Final[object] = object()


def _projection_member(projection: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(projection, Mapping):
        value = projection.get(name, _MISSING)
    else:
        value = getattr(projection, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise _fail(f"import projection omitted canonical field {name!r}")
        return default
    return value


def _validate_physical_b136_native_event_capacity_v1(
    request: Any,
    *,
    batch_cells: int,
    native_event_capacity: int,
) -> None:
    """Enforce the amended source-derived capacity at the allocation owner."""

    if batch_cells != PHYSICAL_B136_BATCH_CELLS_V1:
        return

    envelope = request.envelope
    control = dict(getattr(request, "tick_native_control", {}) or {})
    metadata = _required_mapping(
        _projection_member(request.import_projection, "metadata"),
        name="physical B136 projection metadata",
    )
    configuration = request.configuration
    observed = {
        "world": getattr(envelope, "world", None),
        "profile": getattr(envelope, "profile", None),
        "population": getattr(envelope, "population", None),
        "grid": getattr(envelope, "grid", None),
        "resident_capacity": metadata.get("capacity"),
        "max_births_per_tick": getattr(envelope, "max_births_per_tick", None),
        # The native owner uses the fixed source-buffer width for its private
        # CUDA output ABI.  It is not a generation interval.
        "source_buffer_steps": getattr(configuration, "steps", None),
        "resident_owner_span_ticks": control.get("resident_owner_span_ticks"),
        "native_dispatch_window_ticks": control.get(
            "native_dispatch_window_ticks"
        ),
        "source_window_ticks": control.get("source_window_ticks"),
        "ordinary_births_enabled": getattr(
            configuration, "ordinary_births_enabled", None
        ),
        "generation_turnover_enabled": getattr(
            configuration, "generation_turnover_enabled", None
        ),
    }
    expected = {
        "world": "patch",
        "profile": "P2",
        "population": PHYSICAL_B136_POPULATION_V1,
        "grid": PHYSICAL_B136_GRID_V1,
        "resident_capacity": PHYSICAL_B136_RESIDENT_CAPACITY_V1,
        "max_births_per_tick": PHYSICAL_B136_MAX_BIRTHS_PER_TICK_V1,
        "source_buffer_steps": PHYSICAL_B136_SOURCE_WINDOW_TICKS_V1,
        "resident_owner_span_ticks": 1_120,
        "native_dispatch_window_ticks": 1_120,
        "source_window_ticks": PHYSICAL_B136_SOURCE_WINDOW_TICKS_V1,
        "ordinary_births_enabled": True,
        "generation_turnover_enabled": False,
    }
    mismatches = [
        f"{name}:expected={expected[name]!r},actual={observed[name]!r}"
        for name in expected
        if type(observed[name]) is not type(expected[name])
        or observed[name] != expected[name]
    ]
    if mismatches:
        raise _fail(
            "physical B136 native profile is not the frozen amended profile: "
            + "; ".join(mismatches)
        )

    expected_capacity = physical_b136_event_window_capacity_v1()
    if native_event_capacity != expected_capacity:
        raise _fail(
            "physical B136 native event capacity is not the source-derived exact "
            f"bound: configured={native_event_capacity} expected={expected_capacity}"
        )
    request_event_capacity = getattr(request, "event_capacity", None)
    if request_event_capacity != expected_capacity:
        raise _fail(
            "physical B136 request event capacity does not match the resident "
            f"window: request={request_event_capacity!r} expected={expected_capacity}"
        )


def _common_retained_agent_id_capacity_v1(
    projections: Sequence[Any],
) -> object:
    if not projections:
        raise _fail("retained agent ID capacity requires at least one projection")
    capacities: list[object] = []
    for cell, projection in enumerate(projections):
        metadata = _required_mapping(
            _projection_member(projection, "metadata"),
            name=f"per-cell projection {cell} metadata",
        )
        capacities.append(metadata.get(RESIDENT_AGENT_ID_CAPACITY_METADATA_V1))
    primary = capacities[0]
    for cell, capacity in enumerate(capacities[1:], start=1):
        if capacity != primary:
            raise _fail(f"per-cell request {cell} retained agent ID capacity mismatch")
    return primary


def _validated_source_generation(projection: Any, configuration: Any) -> int:
    metadata = _required_mapping(
        _projection_member(projection, "metadata"),
        name="projection.metadata",
    )
    generation = metadata.get("generation")
    if type(generation) is not int or generation < 0:
        raise _fail(
            "import projection metadata generation must be a nonnegative exact integer"
        )
    tick = metadata.get("tick")
    if type(tick) is not int or tick < 0:
        raise _fail("import projection metadata tick must be nonnegative")
    expected_phase = "generation_boundary" if tick == 0 else "partial_generation"
    if metadata.get("agent_grid_runner_phase") != expected_phase:
        raise _fail(
            "import projection exact tick 0 generation boundary or partial phase "
            "contradicts its exact tick cursor"
        )
    if type(getattr(configuration, "generation", None)) is not int or (
        configuration.generation != generation
    ):
        raise _fail(
            "configuration.generation must equal the import projection source generation"
        )
    if getattr(configuration, "start_tick", 0) != tick:
        raise _fail(
            "configuration.start_tick must equal the import projection source tick"
        )
    return generation


def _source_cardinality_is_qualified_v1(
    *,
    source_generation: int,
    source_tick: int,
    expected_population: int,
    capacity: int,
    initial_agent_counts: Sequence[int],
) -> bool:
    """Validate fresh-source cardinality without rejecting partial generation 0.

    A generation-zero source at tick 0 is the fresh profile-matched founder
    state and must contain exactly the admitted population. Once that same
    generation has advanced into a partial-generation boundary, births/deaths
    may change live cardinality; resident capacity is then the invariant.
    """

    if capacity < expected_population:
        return False
    if any(count > capacity for count in initial_agent_counts):
        return False
    return not (
        source_generation == 0
        and source_tick == 0
        and any(count != expected_population for count in initial_agent_counts)
    )


_PER_CELL_PRIVATE_METADATA_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "event_tape_active_count",
        "event_tape_active_semantics",
        "event_tape_cursor",
        "event_tape_inactive_tail_count",
        "event_tape_public_projection",
        "event_history_frontier",
        "event_history_reference",
        "import_digest",
        "move_cost",
        "patch_food_gather_amount",
        "patch_food_regrow_per_step",
        "patch_movement_cost_energy_scale",
        "jungle_predator_base_risk",
        "jungle_predator_health_damage",
        "jungle_predator_risk_cap",
        "jungle_recovery_activity_base_growth",
        "jungle_recovery_activity_decay_per_step",
        "jungle_recovery_activity_fertility_weight",
        "jungle_recovery_activity_waste_penalty_weight",
        "reproduce_cooldown_steps",
        "reproduce_min_energy_buffer",
        "reproduce_parent_energy_share",
        "reproduce_try_prob",
        "alive_count",
        "legacy_turnover_final_live_value",
        "policy_energy_observation_finite_live_count",
        "policy_energy_observation_live_count",
        "policy_energy_observation_tensor_sha256",
        "population_next_id",
        "native_runtime_reference_genome_materialization",
        "native_runtime_fixed_policy_family",
        "diploid_mutation_enable",
        "native_next_segment_index",
        "native_chain_sha256",
        "native_cumulative_attempt_count",
        "native_cumulative_failure_count",
        "native_cumulative_success_count",
        "typed_lifecycle_event_active_count",
        "typed_lifecycle_event_sequence_base",
        "typed_lifecycle_event_next_sequence",
        "genome_upgrade_v2_mutation_rate_q10",
        "genome_upgrade_v2_mutation_step_q10",
        "recomb72_mutation_off",
        # WDL/fixed-MLP materializes this value in the resident
        # ``radius_by_cell`` CUDA vector.  Treating it as shared here rejects
        # a transport path the prepared WDL owner explicitly supports.
        "wdl_delayed_hazard_radius",
        "wdl_agents_cued_current_generation_agent_ids",
        "eed_phenotype_fixed_axis_profile",
    }
)


def _is_per_cell_private_metadata_field_v1(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in _PER_CELL_PRIVATE_METADATA_FIELDS_V1
        or lowered == "seed"
        or lowered.endswith("_seed")
        or "rng" in lowered
        or "random" in lowered
        or "numpy" in lowered
    )


def _shared_projection_metadata_v1(metadata: Mapping[str, Any]) -> dict[str, Any]:
    shared = {
        name: value
        for name, value in metadata.items()
        if not _is_per_cell_private_metadata_field_v1(str(name))
    }
    capabilities = shared.get("capability_mask")
    if isinstance(capabilities, Mapping):
        shared["capability_mask"] = {
            name: value for name, value in capabilities.items() if name != "gru"
        }
    return shared


def _validate_per_cell_policy_family_v1(
    metadata: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    cell: int,
) -> None:
    family = metadata.get("native_runtime_fixed_policy_family", _MISSING)
    if type(family) is not str or family not in {"mlp", "gru", "mixed"}:
        raise _fail(
            f"per-cell projection {cell} policy family must be exactly mlp, gru, or mixed"
        )
    capabilities = metadata.get("capability_mask", _MISSING)
    if not isinstance(capabilities, Mapping):
        raise _fail(f"per-cell projection {cell} capability_mask must be a mapping")
    expected_gru = family in {"gru", "mixed"}
    if capabilities.get("gru", _MISSING) is not expected_gru:
        raise _fail(
            f"per-cell projection {cell} GRU capability contradicts its policy family"
        )

    alive = tensors.get("alive_mask", _MISSING)
    policy_use_gru = tensors.get("policy_use_gru", _MISSING)
    if alive is _MISSING and policy_use_gru is _MISSING:
        return
    if alive is _MISSING or policy_use_gru is _MISSING:
        raise _fail(
            f"per-cell projection {cell} policy-family tensors must appear together"
        )
    if (
        not isinstance(alive, torch.Tensor)
        or alive.dtype != torch.bool
        or alive.device.type != "cpu"
        or not isinstance(policy_use_gru, torch.Tensor)
        or policy_use_gru.dtype != torch.bool
        or policy_use_gru.device.type != "cpu"
        or policy_use_gru.shape != alive.shape
    ):
        raise _fail(
            f"per-cell projection {cell} alive_mask and policy_use_gru must be "
            "same-shape CPU bool tensors"
        )
    live_policy = policy_use_gru[alive]
    if family == "mlp" and bool(torch.any(live_policy).item()):
        raise _fail(f"per-cell projection {cell} MLP family contains a live GRU row")
    if family == "gru" and not bool(torch.all(live_policy).item()):
        raise _fail(f"per-cell projection {cell} GRU family contains a live MLP row")


def _same_structured_value_v1(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if tuple(left) != tuple(right):
            return False
        return all(_same_structured_value_v1(left[name], right[name]) for name in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_structured_value_v1(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, torch.Tensor):
        return False
    try:
        result = left == right
    except Exception:
        return False
    return type(result) is bool and result


def _same_configuration_except_founder_profile_v1(left: Any, right: Any) -> bool:
    if not dataclasses.is_dataclass(left) or not dataclasses.is_dataclass(right):
        return False
    left_payload = dataclasses.asdict(left)
    right_payload = dataclasses.asdict(right)
    for field in (
        "genome_upgrade_v2_founder_profile_id",
        "genome_upgrade_v2_founder_profile_sha256",
        "move_cost",
        "patch_food_gather_amount",
        "patch_food_regrow_per_step",
        "patch_movement_cost_energy_scale",
        "jungle_predator_base_risk",
        "jungle_predator_health_damage",
        "jungle_predator_risk_cap",
        "jungle_recovery_activity_base_growth",
        "jungle_recovery_activity_decay_per_step",
        "jungle_recovery_activity_fertility_weight",
        "jungle_recovery_activity_waste_penalty_weight",
        "reproduce_cooldown_steps",
        "reproduce_min_energy_buffer",
        "reproduce_parent_energy_share",
        "reproduce_try_prob",
        "diploid_mutation_enable",
        "genome_upgrade_v2_mutation_rate_q10",
        "genome_upgrade_v2_mutation_step_q10",
        "recomb72_mutation_off",
        "turnover_selection_seed",
    ):
        left_payload.pop(field, None)
        right_payload.pop(field, None)
    return _same_structured_value_v1(left_payload, right_payload)


def _projection_schema_version_v1(projection: Any) -> str:
    schema_version = _projection_member(projection, "schema_version")
    if type(schema_version) is not str or not schema_version:
        raise _fail("per-cell projection schema_version must be a nonempty string")
    return schema_version


def _validated_per_cell_projection_tensors_v1(
    projections: Sequence[Any],
    *,
    batch_cells: int,
) -> tuple[dict[str, torch.Tensor], ...]:
    if isinstance(projections, (str, bytes)) or not isinstance(projections, Sequence):
        raise _fail("per_cell_import_projections must be a sequence")
    if len(projections) != batch_cells:
        raise _fail("per_cell_import_projections count must exactly equal batch_cells")
    if len({id(projection) for projection in projections}) != batch_cells:
        raise _fail("per-cell projections must be distinct objects")

    first_schema = _projection_schema_version_v1(projections[0])
    first_metadata_raw = _required_mapping(
        _projection_member(projections[0], "metadata"),
        name="per_cell_import_projections[0].metadata",
    )
    first_tensors = _required_mapping(
        _projection_member(projections[0], "tensors"),
        name="per_cell_import_projections[0].tensors",
    )
    _validate_per_cell_policy_family_v1(
        first_metadata_raw,
        first_tensors,
        cell=0,
    )
    first_metadata = _shared_projection_metadata_v1(first_metadata_raw)
    field_names = tuple(first_tensors)
    if not field_names:
        raise _fail("per-cell projections must expose tensor fields")
    hapb_restore_views = (
        all(
            _required_mapping(
                _projection_member(projection, "metadata"),
                name=f"per_cell_import_projections[{cell}].metadata",
            ).get("hapb_policy_restore_pending_cache_rehydrate")
            is True
            for cell, projection in enumerate(projections)
        )
        and all(
            isinstance(
                _required_mapping(
                    _projection_member(projection, "metadata"),
                    name=f"per_cell_import_projections[{cell}].metadata",
                ).get("hapb_policy_checkpoint_sha256"),
                str,
            )
            for cell, projection in enumerate(projections)
        )
        and _shared_hapb_canonical_bank_from_rows_v1(
            [
                _required_mapping(
                    _projection_member(projection, "tensors"),
                    name=f"per_cell_import_projections[{cell}].tensors",
                ).get("mlp_pack")
                for cell, projection in enumerate(projections)
            ]
        )
        is not None
    )

    validated: list[dict[str, torch.Tensor]] = []
    occupied_storages: dict[tuple[str, int], tuple[int, str]] = {}
    expected_layout: dict[str, tuple[Any, ...]] = {}
    for cell, projection in enumerate(projections):
        if _projection_schema_version_v1(projection) != first_schema:
            raise _fail(f"per-cell projection {cell} schema_version mismatch")
        raw_metadata = _required_mapping(
            _projection_member(projection, "metadata"),
            name=f"per_cell_import_projections[{cell}].metadata",
        )
        tensors = _required_mapping(
            _projection_member(projection, "tensors"),
            name=f"per_cell_import_projections[{cell}].tensors",
        )
        if cell:
            _validate_per_cell_policy_family_v1(
                raw_metadata,
                tensors,
                cell=cell,
            )
        metadata = _shared_projection_metadata_v1(raw_metadata)
        if not _same_structured_value_v1(metadata, first_metadata):
            differing_fields = sorted(
                name
                for name in set(first_metadata) | set(metadata)
                if name not in first_metadata
                or name not in metadata
                or not _same_structured_value_v1(first_metadata[name], metadata[name])
            )
            raise _fail(
                f"per-cell projection {cell} metadata/config mismatch: "
                f"fields={differing_fields}"
            )
        if tuple(tensors) != field_names:
            raise _fail(f"per-cell projection {cell} tensor schema/layout mismatch")
        typed: dict[str, torch.Tensor] = {}
        for name in field_names:
            value = tensors[name]
            if not isinstance(value, torch.Tensor):
                raise _fail(f"per-cell projection tensor {name!r} must be a tensor")
            if value.device.type != "cpu" or value.requires_grad:
                raise _fail(
                    f"per-cell projection tensor {name!r} must be detached CPU storage"
                )
            layout = (
                value.dtype,
                value.layout,
                tuple(value.shape),
                tuple(value.stride()),
            )
            ragged_dtype = RAGGED_PROJECTION_FIELDS_V1.get(name)
            if ragged_dtype is not None:
                if (
                    value.dtype != ragged_dtype
                    or value.layout != torch.strided
                    or value.ndim != 1
                    or not value.is_contiguous()
                ):
                    raise _fail(
                        f"per-cell ragged projection tensor {name!r} "
                        "dtype/layout mismatch"
                    )
                if cell == 0:
                    expected_layout[name] = layout
            elif name in EVENT_COLUMN_NAMES:
                if (
                    value.layout != torch.strided
                    or value.ndim != 1
                    or not value.is_contiguous()
                    or (cell and value.dtype != expected_layout[name][0])
                ):
                    raise _fail(
                        f"per-cell event projection tensor {name!r} "
                        "dtype/layout mismatch"
                    )
                if cell == 0:
                    expected_layout[name] = layout
            elif cell == 0:
                expected_layout[name] = layout
            elif layout != expected_layout[name]:
                raise _fail(
                    f"per-cell projection tensor {name!r} shape/dtype/layout mismatch"
                )
            if value.numel():
                storage_key = (
                    str(value.device),
                    int(value.untyped_storage().data_ptr()),
                )
                previous = occupied_storages.get(storage_key)
                if previous is not None:
                    previous_cell, previous_name = previous
                    if previous_cell != cell and not (
                        hapb_restore_views
                        and name == "mlp_pack"
                        and previous_name == "mlp_pack"
                    ):
                        raise _fail(
                            "per-cell projection tensor storage alias: "
                            f"cell={cell} field={name!r} aliases "
                            f"cell={previous_cell} field={previous_name!r}"
                        )
                else:
                    occupied_storages[storage_key] = (cell, name)
            typed[name] = value
        active_identity_names = (
            "agent_ids_by_alive_order",
            "agent_id_to_slot_agent_ids",
            "agent_id_to_slot_slots",
        )
        active_lengths = {
            int(typed[name].numel()) for name in active_identity_names if name in typed
        }
        if len(active_lengths) > 1:
            raise _fail(f"per-cell projection {cell} compact identity lengths mismatch")
        alive = typed.get("alive_mask")
        if (
            active_lengths
            and isinstance(alive, torch.Tensor)
            and int(alive.sum().item()) != next(iter(active_lengths))
        ):
            raise _fail(
                f"per-cell projection {cell} compact identity count "
                "contradicts alive_mask"
            )
        validated.append(typed)
    return tuple(validated)


def stack_validated_per_cell_checkpoint_projections_v1(
    projections: Sequence[Any],
    *,
    batch_cells: int,
    device: torch.device | str,
    field_names: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Stack private checkpoint tensor banks after exact compatibility checks."""

    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 256:
        raise _fail("batch_cells must be an exact integer in [1,256]")
    tensors = _validated_per_cell_projection_tensors_v1(
        projections,
        batch_cells=batch_cells,
    )
    selected = tuple(tensors[0]) if field_names is None else tuple(field_names)
    if len(set(selected)) != len(selected):
        raise _fail("per-cell projection field_names must be unique")
    missing = sorted(name for name in selected if name not in tensors[0])
    if missing:
        raise _fail(f"per-cell projection fields are missing: {missing}")
    target = torch.device(device)
    stacked: dict[str, torch.Tensor] = {}
    for name in selected:
        ragged_dtype = RAGGED_PROJECTION_FIELDS_V1.get(name)
        if ragged_dtype is not None:
            stacked[name] = _stack_ragged_vectors_v1(
                [cell[name] for cell in tensors],
                name=name,
                dtype=ragged_dtype,
                device=target,
            )
        else:
            stacked[name] = torch.stack(
                [cell[name].to(device=target) for cell in tensors],
                dim=0,
            ).contiguous()
    return stacked


def _stack(
    value: Any, *, device: torch.device, batch_cells: int = BATCH_CELLS_V1
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _fail("resident source fields must be tensors")
    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 256:
        raise _fail("batch_cells must be an exact integer in [1,256]")
    stacked = value.unsqueeze(0).to(device)
    return stacked.repeat(batch_cells, *([1] * value.dim())).contiguous()


def _validate_per_cell_values_v1(
    values: Sequence[Any],
    *,
    name: str,
    source_groups: Sequence[int] | None = None,
) -> None:
    if not values or any(not isinstance(value, torch.Tensor) for value in values):
        raise _fail(f"per-cell resident source field {name!r} must contain tensors")
    expected = (
        values[0].dtype,
        values[0].layout,
        tuple(values[0].shape),
        tuple(values[0].stride()),
    )
    occupied_storages: dict[tuple[str, int], int] = {}
    for cell, value in enumerate(values):
        actual = (value.dtype, value.layout, tuple(value.shape), tuple(value.stride()))
        if actual != expected:
            raise _fail(
                f"per-cell resident source field {name!r} shape/dtype/layout "
                f"mismatch at cell {cell}"
            )
        if value.numel():
            storage_key = (
                str(value.device),
                int(value.untyped_storage().data_ptr()),
            )
            previous = occupied_storages.get(storage_key)
            if previous is not None:
                if (
                    source_groups is None
                    or source_groups[cell] != source_groups[previous]
                ):
                    raise _fail(
                        f"per-cell resident source field {name!r} storage alias: "
                        f"cell={cell} aliases cell={previous}"
                    )
            occupied_storages[storage_key] = cell


def _stack_per_cell_values_v1(
    values: Sequence[Any],
    *,
    name: str,
    device: torch.device,
    source_groups: Sequence[int] | None = None,
) -> torch.Tensor:
    _validate_per_cell_values_v1(
        values,
        name=name,
        source_groups=source_groups,
    )
    first = values[0]
    try:
        stacked = torch.empty(
            (len(values), *first.shape),
            dtype=first.dtype,
            device=device,
        )
        for cell, value in enumerate(values):
            stacked[cell].copy_(value, non_blocking=False)
        return stacked
    except torch.OutOfMemoryError as exc:
        raise _fail(
            f"CUDA allocation failed while stacking per-cell resident "
            f"source field {name!r}: cells={len(values)} "
            f"shape={tuple(first.shape)} dtype={first.dtype} "
            f"bytes={len(values) * first.numel() * first.element_size()}: {exc}"
        ) from exc


def _stack_ragged_vectors_v1(
    values: Sequence[torch.Tensor],
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    width: int | None = None,
    source_groups: Sequence[int] | None = None,
) -> torch.Tensor:
    if not values or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != dtype
        or value.layout != torch.strided
        or value.ndim != 1
        or not value.is_contiguous()
        for value in values
    ):
        raise _fail(
            f"per-cell ragged field {name!r} must contain contiguous {dtype} vectors"
        )
    occupied_storages: dict[tuple[str, int], int] = {}
    for cell, value in enumerate(values):
        if not value.numel():
            continue
        storage_key = (str(value.device), int(value.untyped_storage().data_ptr()))
        previous = occupied_storages.get(storage_key)
        if previous is not None and (
            source_groups is None or source_groups[cell] != source_groups[previous]
        ):
            raise _fail(
                f"per-cell resident source field {name!r} storage alias: "
                f"cell={cell} aliases cell={previous}"
            )
        occupied_storages[storage_key] = cell
    required_width = max(int(value.numel()) for value in values)
    target_width = required_width if width is None else width
    if (
        type(target_width) is not int
        or target_width < 0
        or target_width < required_width
    ):
        raise _fail(f"ragged field {name!r} exceeds its admitted width")
    out = torch.full(
        (len(values), target_width),
        -1,
        dtype=dtype,
        device=device,
    )
    for cell, value in enumerate(values):
        out[cell, : value.numel()].copy_(value, non_blocking=False)
    return out.contiguous()


def _validated_python_gauss_tape_width_v1(
    sources: Sequence[Mapping[str, Any]],
) -> int:
    fields = (
        "python_random_gauss_primary_tape",
        "python_random_gauss_secondary_tape",
    )
    width = 0
    for cell, source in enumerate(sources):
        tapes = tuple(source.get(name) for name in fields)
        if any(
            not isinstance(tape, torch.Tensor)
            or tape.dtype != torch.float64
            or tape.layout != torch.strided
            or tape.ndim != 1
            or not tape.is_contiguous()
            for tape in tapes
        ):
            raise _fail(f"per-cell Python Gaussian tapes are invalid at cell {cell}")
        if tapes[0].numel() != tapes[1].numel():
            raise _fail(
                f"per-cell Python Gaussian tape pair length mismatch at cell {cell}"
            )
        width = max(width, int(tapes[0].numel()))
    return width


def _resource_maintenance_energy_cost_per_tick_v1(config: object) -> float:
    value = float(
        getattr(config, "resource_maintenance_energy_cost_per_tick", 0.0) or 0.0
    )
    if not math.isfinite(value) or value < 0.0:
        raise _fail(
            "resource_maintenance_energy_cost_per_tick must be finite and non-negative"
        )
    return value


def _apply_resource_maintenance_cost_charge_floor_v1(
    batched: dict[str, torch.Tensor],
    *,
    config: object,
) -> None:
    cost = _resource_maintenance_energy_cost_per_tick_v1(config)
    if cost <= 0.0:
        return
    charge = batched.get("cost_charge_scalar")
    if charge is None:
        raise _fail("resource maintenance requires cost_charge_scalar")
    if charge.dtype != torch.float32:
        raise _fail("resource maintenance requires float32 cost_charge_scalar")
    baseline = torch.full_like(charge, float(cost))
    batched["cost_charge_scalar"] = torch.maximum(charge, baseline).contiguous()


def _stack_ragged_agent_id_to_slot_v1(
    values: Sequence[torch.Tensor],
    *,
    device: torch.device,
    source_groups: Sequence[int] | None = None,
) -> torch.Tensor:
    return _stack_ragged_vectors_v1(
        values,
        name="agent_id_to_slot",
        dtype=torch.int32,
        device=device,
        source_groups=source_groups,
    )


def _stack_ragged_active_agent_ids_v1(
    values: Sequence[torch.Tensor],
    *,
    capacity: int,
    device: torch.device,
    source_groups: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_ids = _stack_ragged_vectors_v1(
        values,
        name="agent_ids_by_alive_order",
        dtype=torch.int64,
        device=device,
        width=capacity,
        source_groups=source_groups,
    )
    active_count = torch.tensor(
        [value.numel() for value in values],
        dtype=torch.int64,
        device=device,
    )
    return active_ids, active_count


def _compact_identity_values_v1(
    source: Mapping[str, torch.Tensor],
    sources: Sequence[Mapping[str, torch.Tensor]] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    identity_sources = [source] if sources is None else list(sources)
    return (
        [item["agent_id_to_slot_agent_ids"] for item in identity_sources],
        [item["agent_id_to_slot_slots"] for item in identity_sources],
    )


def _capacity_bounded_active_agent_slots_v1(
    active_agent_ids: torch.Tensor,
    active_count_state: torch.Tensor,
    compact_agent_ids: torch.Tensor,
    compact_agent_slots: torch.Tensor,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        active_agent_ids.dtype != torch.int64
        or active_agent_ids.ndim != 2
        or not active_agent_ids.is_contiguous()
        or active_count_state.dtype != torch.int64
        or active_count_state.ndim != 1
        or not active_count_state.is_contiguous()
        or compact_agent_ids.dtype != torch.int64
        or compact_agent_ids.shape != active_agent_ids.shape
        or not compact_agent_ids.is_contiguous()
        or compact_agent_slots.dtype != torch.int32
        or compact_agent_slots.shape != active_agent_ids.shape
        or not compact_agent_slots.is_contiguous()
        or slot_to_agent_id.dtype != torch.int64
        or slot_to_agent_id.shape != active_agent_ids.shape
        or not slot_to_agent_id.is_contiguous()
        or alive_mask.dtype != torch.bool
        or alive_mask.shape != active_agent_ids.shape
        or not alive_mask.is_contiguous()
        or active_count_state.shape[0] != active_agent_ids.shape[0]
    ):
        raise _fail("active identity rows have incompatible shape, dtype, or layout")
    cells, capacity = active_agent_ids.shape
    active_agent_slots = torch.full(
        (cells, capacity),
        -1,
        dtype=torch.int32,
        device=active_agent_ids.device,
    )
    for cell in range(cells):
        active_count = int(active_count_state[cell].item())
        if active_count < 0 or active_count > capacity:
            raise _fail(f"active identity count for cell {cell} is invalid")
        if active_count == 0:
            continue
        agent_ids = active_agent_ids[cell, :active_count]
        compact_ids = compact_agent_ids[cell, :active_count]
        slots = compact_agent_slots[cell, :active_count].to(torch.int64)
        if (
            bool(agent_ids.lt(0).any().item())
            or bool(agent_ids.gt(INT32_MAX).any().item())
            or (
                active_count > 1 and bool(agent_ids[1:].le(agent_ids[:-1]).any().item())
            )
            or not torch.equal(agent_ids, compact_ids)
        ):
            raise _fail(f"active identity IDs for cell {cell} are invalid or unordered")
        if (
            bool(slots.lt(0).any().item())
            or bool(slots.ge(capacity).any().item())
            or int(torch.unique(slots).numel()) != active_count
            or not bool(alive_mask[cell].index_select(0, slots).all().item())
            or not torch.equal(
                slot_to_agent_id[cell].index_select(0, slots),
                agent_ids,
            )
        ):
            raise _fail(
                f"active identity ID/slot pairs for cell {cell} contradict "
                "resident slot authority"
            )
        active_agent_slots[cell, :active_count].copy_(slots.to(torch.int32))
    return active_agent_slots


def _canonicalize_agent_id_grid_v1(
    agent_grid: torch.Tensor,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    pos_y: torch.Tensor,
    pos_x: torch.Tensor,
) -> torch.Tensor:
    if (
        agent_grid.dtype != torch.int32
        or agent_grid.ndim != 3
        or not agent_grid.is_contiguous()
        or slot_to_agent_id.dtype != torch.int64
        or slot_to_agent_id.ndim != 2
        or not slot_to_agent_id.is_contiguous()
        or alive_mask.dtype != torch.bool
        or alive_mask.shape != slot_to_agent_id.shape
        or not alive_mask.is_contiguous()
        or pos_y.dtype != torch.int32
        or pos_y.shape != slot_to_agent_id.shape
        or not pos_y.is_contiguous()
        or pos_x.dtype != torch.int32
        or pos_x.shape != slot_to_agent_id.shape
        or not pos_x.is_contiguous()
        or int(agent_grid.size(0)) != int(slot_to_agent_id.size(0))
    ):
        raise _fail("agent-grid identity canonicalization inputs are incompatible")
    cells, capacity = slot_to_agent_id.shape
    height = int(agent_grid.size(1))
    width = int(agent_grid.size(2))
    canonical = torch.full_like(agent_grid, -1)
    for cell in range(cells):
        slots = torch.nonzero(alive_mask[cell], as_tuple=False).flatten()
        if not int(slots.numel()):
            continue
        ids = slot_to_agent_id[cell].index_select(0, slots)
        ys = pos_y[cell].index_select(0, slots).to(torch.int64)
        xs = pos_x[cell].index_select(0, slots).to(torch.int64)
        if (
            bool(ids.lt(0).any().item())
            or bool(ids.gt(INT32_MAX).any().item())
            or bool(ys.lt(0).any().item())
            or bool(ys.ge(height).any().item())
            or bool(xs.lt(0).any().item())
            or bool(xs.ge(width).any().item())
        ):
            raise _fail(f"agent-grid identity rows for cell {cell} are invalid")
        linear = ys * width + xs
        if int(torch.unique(linear).numel()) != int(slots.numel()):
            raise _fail(f"agent-grid identity positions for cell {cell} collide")
        canonical[cell, ys, xs] = ids.to(torch.int32)
    return canonical.contiguous()


def _allocate_initial_id_reporting_state_v1(
    values: Sequence[torch.Tensor],
    *,
    reporting_capacity: int,
    device: torch.device,
    reporting_rows_override: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not values
        or type(reporting_capacity) is not int
        or reporting_capacity < 1
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.int64
            or value.layout != torch.strided
            or value.ndim != 1
            or not value.is_contiguous()
            for value in values
        )
    ):
        raise _fail(
            "initial reporting IDs require contiguous int64 vectors and "
            "a positive reporting capacity"
        )
    reporting_rows = max(int(value.numel()) for value in values)
    if reporting_rows_override is not None:
        if (
            type(reporting_rows_override) is not int
            or reporting_rows_override < reporting_rows
            or reporting_rows_override > reporting_capacity
        ):
            raise _fail(
                "initial reporting row override must cover the source rows and "
                "remain within resident capacity"
            )
        reporting_rows = reporting_rows_override
    if reporting_rows > reporting_capacity:
        raise _fail("initial reporting rows exceed resident capacity")
    initial_agent_ids_by_row = torch.full(
        (len(values), reporting_rows),
        INT64_MAX,
        dtype=torch.int64,
        device=device,
    )
    for cell, value in enumerate(values):
        ids = value.to(device=device)
        if ids.numel() and (
            bool(ids.lt(0).any().item())
            or bool(ids.gt(INT32_MAX).any().item())
            or (ids.numel() > 1 and bool(ids[1:].le(ids[:-1]).any().item()))
        ):
            raise _fail(
                f"initial reporting IDs for cell {cell} are invalid or unordered"
            )
        initial_agent_ids_by_row[cell, : ids.numel()].copy_(ids)
    terminal_survival = torch.zeros(
        (len(values), reporting_rows),
        dtype=torch.int32,
        device=device,
    )
    return (
        initial_agent_ids_by_row,
        terminal_survival,
        torch.zeros_like(terminal_survival),
    )


_stack_per_cell_agent_id_to_slot_v1 = _stack_ragged_agent_id_to_slot_v1


def _stack_fixed_mlp_final_layout_v1(
    value: torch.Tensor,
    *,
    device: torch.device,
    batch_cells: int,
    base_observation_size: int,
    observation_size: int,
    hidden_size: int,
    action_count: int,
) -> torch.Tensor:
    """Batch a fixed-MLP row directly into its final observation layout."""

    original_width = (
        hidden_size * base_observation_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + action_count * hidden_size
        + action_count
    )
    final_width = original_width + hidden_size * (
        observation_size - base_observation_size
    )
    if value.ndim < 1 or int(value.size(-1)) != original_width:
        raise _fail("fixed-MLP source tensor has an invalid packed width")
    source = value.to(device=device)
    final = torch.zeros(
        (batch_cells, *value.shape[:-1], final_width),
        dtype=value.dtype,
        device=device,
    )
    source_prefix = source[..., : hidden_size * base_observation_size].view(
        *value.shape[:-1], hidden_size, base_observation_size
    )
    final_prefix = final[..., : hidden_size * observation_size].view(
        batch_cells, *value.shape[:-1], hidden_size, observation_size
    )
    final_prefix[..., :base_observation_size].copy_(
        source_prefix.unsqueeze(0).expand(batch_cells, *source_prefix.shape)
    )
    final[..., hidden_size * observation_size :].copy_(
        source[..., hidden_size * base_observation_size :]
        .unsqueeze(0)
        .expand(batch_cells, *value.shape[:-1], -1)
    )
    return final.contiguous()


def _checkpoint_schema_template_v1(value: Any) -> Any:
    """Retain checkpoint field structure without a second resident tensor bank."""

    if isinstance(value, torch.Tensor):
        return torch.empty((0,), dtype=value.dtype, device="cpu")
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                field.name: _checkpoint_schema_template_v1(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {
            name: _checkpoint_schema_template_v1(child) for name, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_checkpoint_schema_template_v1(child) for child in value)
    return value


def _source_mapping(request: Any) -> dict[str, torch.Tensor]:
    boundary = getattr(request, "cuda_boundary", None)
    boundary_tensors = _required_mapping(
        getattr(boundary, "tensors", None),
        name="cuda_boundary.tensors",
    )
    birth = request.generation_birth_state
    result = dict(boundary_tensors)
    host_authoritative = _required_mapping(
        getattr(boundary, "host_authoritative_tensors", {}),
        name="cuda_boundary.host_authoritative_tensors",
    )
    if set(result).intersection(host_authoritative):
        raise _fail("CUDA boundary tensors and host-authoritative tensors overlap")
    result.update(host_authoritative)
    for field in (
        "identity_state",
        "scalar_rng_state",
        "genome_state",
        "policy_state",
        "turnover_policy_state",
    ):
        result.update(_required_mapping(getattr(birth, field, None), name=field))
    drive_state = _required_mapping(
        getattr(birth, "drive_state", None), name="drive_state"
    )
    result.update(
        {
            "policy_mem_dim": birth.policy_mem_dim,
            "policy_use_gru": birth.policy_use_gru,
            "birth_reset_shock": drive_state["birth_reset_shock"],
            "population_pcg64_state_words": drive_state["pcg64_state_words"],
            "population_pcg64_buffer_words": drive_state["pcg64_buffer_words"],
            "mutation_np_semantic_draws": drive_state["mutation_np_semantic_draws"],
        }
    )
    # A HAPB boundary owns the fixed-MLP rows on pageable host memory. Birth
    # preparation may carry those rows only to retain the exact tape width;
    # it must not replace the explicit host authority in the source map.
    if "mlp_pack" in host_authoritative:
        result["mlp_pack"] = host_authoritative["mlp_pack"]
    result.setdefault(
        "early_injury_gained", torch.zeros_like(boundary_tensors["energy"])
    )
    result.setdefault(
        "chemistry_observation_fidelity",
        torch.ones_like(boundary_tensors["energy"]),
    )
    result.setdefault(
        "chemistry_observation_bias",
        torch.zeros_like(boundary_tensors["energy"]),
    )
    return result


def _shared_hapb_canonical_bank_from_rows_v1(
    rows: Sequence[torch.Tensor],
) -> torch.Tensor | None:
    """Adopt restored B48 row views without stacking a second host bank."""

    if not 1 <= len(rows) <= 256:
        return None
    first = rows[0]
    expected_shape = (512, 9991)
    if any(
        not isinstance(row, torch.Tensor)
        or row.device.type != "cpu"
        or row.is_pinned()
        or row.dtype != torch.float32
        or not row.is_contiguous()
        or tuple(row.shape) != expected_shape
        for row in rows
    ):
        return None
    row_bytes = 512 * 9991 * first.element_size()
    storage = first.untyped_storage()
    base_offset = first.storage_offset()
    for cell, row in enumerate(rows):
        if (
            row.untyped_storage().data_ptr() != storage.data_ptr()
            or row.storage_offset() != base_offset + cell * 512 * 9991
            or row.data_ptr() != first.data_ptr() + cell * row_bytes
        ):
            return None
    return first.as_strided((len(rows), 512, 9991), (512 * 9991, 9991, 1))


def _validate_source_tensor_devices_v1(
    source: Mapping[str, Any],
    *,
    expected_device: torch.device,
    name: str,
) -> None:
    for field, value in source.items():
        if (
            field in ORDINARY_BIRTH_POLICY_FIELDS
            and isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and expected_device.type == "cuda"
        ):
            # These canonical policy fields are copied into the CUDA resident
            # source by the bounded stacking path below.
            continue
        if isinstance(value, torch.Tensor) and value.device != expected_device:
            raise _fail(
                f"{name} resident source field {field!r} must be on "
                f"{expected_device.type}"
            )


def _allocate_tick_outputs(
    *,
    capacity: int,
    observation_size: int,
    action_count: int,
    device: torch.device,
    output_ticks: int = OUTPUT_TICKS_V1,
    batch_cells: int = BATCH_CELLS_V1,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS_V1,
) -> dict[str, Any]:
    if type(output_ticks) is not int or output_ticks < 1:
        raise ValueError("output_ticks must be a positive exact integer")
    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 256:
        raise ValueError("batch_cells must be an exact integer in [1,256]")
    window_ticks = physical_window_ticks_v1(physical_window_generations)
    b, t = batch_cells, output_ticks
    outputs: dict[str, Any] = {
        "ordered_agent_ids": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "ordered_slots": torch.full(
            (b, t, capacity), -1, dtype=torch.int32, device=device
        ),
        "observations": torch.zeros(
            (b, t, capacity, observation_size), dtype=torch.float32, device=device
        ),
        "logits": torch.zeros(
            (b, t, capacity, action_count), dtype=torch.float32, device=device
        ),
        "row_actions": torch.full(
            (b, t, capacity), -1, dtype=torch.int32, device=device
        ),
        "actions_by_slot": torch.full(
            (b, t, capacity), 5, dtype=torch.int32, device=device
        ),
        "pre_energy": torch.empty((b, t, capacity), dtype=torch.float32, device=device),
        "pre_health": torch.empty((b, t, capacity), dtype=torch.float32, device=device),
        "pre_injury": torch.empty((b, t, capacity), dtype=torch.float32, device=device),
        "pre_survival_steps": torch.empty(
            (b, t, capacity), dtype=torch.int32, device=device
        ),
        "active_count": torch.zeros((b, t), dtype=torch.int64, device=device),
        "policy_refill_rows": torch.zeros((b, t), dtype=torch.int64, device=device),
        "policy_cache_hit_rows": torch.zeros((b, t), dtype=torch.int64, device=device),
        "runtime_bookkeeping_by_tick": torch.zeros(
            (b, window_ticks, 5),
            dtype=torch.int64,
            device=device,
        ),
        "death_agent_ids": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "death_slots": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "death_reason_codes": torch.zeros(
            (b, t, capacity), dtype=torch.int32, device=device
        ),
        "death_count": torch.zeros((b, t), dtype=torch.int64, device=device),
        "survivor_agent_ids": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "survivor_count": torch.zeros((b, t), dtype=torch.int64, device=device),
        "lifecycle_status": torch.zeros((b, t), dtype=torch.int32, device=device),
        "chemistry_order_slots": torch.full(
            (b, t, capacity), -1, dtype=torch.int32, device=device
        ),
        "runner_shuffle_draws": torch.zeros((b, t), dtype=torch.int64, device=device),
        "runner_shuffle_status": torch.zeros((b, t), dtype=torch.int32, device=device),
        "drive_noise_draws": torch.zeros((b, t), dtype=torch.int64, device=device),
        "drive_noise_raw64_draws": torch.zeros(
            (b, t), dtype=torch.int64, device=device
        ),
        "drive_noise_status": torch.zeros((b, t), dtype=torch.int32, device=device),
        "chemistry_draws": torch.zeros((b, t), dtype=torch.int64, device=device),
        "chemistry_status": torch.zeros((b, t), dtype=torch.int32, device=device),
        "active_mechanics_status": torch.zeros(
            (b, t), dtype=torch.int32, device=device
        ),
        "active_mechanics_counts": torch.zeros(
            (b, t, 5), dtype=torch.int64, device=device
        ),
        "residual_death_agent_ids": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "residual_death_slots": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "residual_death_reason_codes": torch.zeros(
            (b, t, capacity), dtype=torch.int32, device=device
        ),
        "residual_death_count": torch.zeros((b, t), dtype=torch.int64, device=device),
        "residual_release_count": torch.zeros((b, t), dtype=torch.int64, device=device),
        "residual_release_status": torch.zeros(
            (b, t), dtype=torch.int32, device=device
        ),
        "policy_pcg64_state_after_action": torch.zeros(
            (b, t, 4), dtype=torch.uint64, device=device
        ),
        "policy_pcg64_buffer_after_action": torch.zeros(
            (b, t, 2), dtype=torch.int64, device=device
        ),
        "food_spawn_stats": torch.zeros((b, t, 3), dtype=torch.int64, device=device),
        "event_status_by_tick": torch.zeros((b, t), dtype=torch.int32, device=device),
        "food_gained_scratch": torch.zeros(
            (b, capacity), dtype=torch.float32, device=device
        ),
        "status": torch.zeros((b,), dtype=torch.int32, device=device),
        "first_failure_records": torch.full(
            (b, FIRST_FAILURE_RECORD_WIDTH_V1), -1, dtype=torch.int64, device=device
        ),
        "generation_summary_scalar": torch.full(
            (physical_window_generations, b, GENERATION_SUMMARY_WIDTH_V1),
            float("nan"),
            dtype=torch.float64,
            device=device,
        ),
    }
    return outputs


def _allocate_gmnw_qualification_trace_outputs(
    *,
    capacity: int,
    hidden_width: int,
    task_state_width: int,
    device: torch.device,
    output_ticks: int,
    batch_cells: int,
) -> dict[str, torch.Tensor]:
    if type(capacity) is not int or capacity < 1:
        raise ValueError("capacity must be a positive exact integer")
    if type(hidden_width) is not int or hidden_width < 1:
        raise ValueError("hidden_width must be a positive exact integer")
    if type(task_state_width) is not int or task_state_width < 1:
        raise ValueError("task_state_width must be a positive exact integer")
    if type(output_ticks) is not int or output_ticks < 1 or output_ticks > 160:
        raise ValueError("output_ticks must be an exact integer in [1,160]")
    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 256:
        raise ValueError("batch_cells must be an exact integer in [1,256]")
    hidden_shape = (batch_cells, output_ticks, capacity, hidden_width)
    task_shape = (batch_cells, output_ticks, capacity, task_state_width)
    return {
        "gmnw_trace_hidden_before": torch.full(
            hidden_shape, float("nan"), dtype=torch.float32, device=device
        ),
        "gmnw_trace_hidden_after": torch.full(
            hidden_shape, float("nan"), dtype=torch.float32, device=device
        ),
        "gmnw_trace_task_state_before": torch.full(
            task_shape, -1, dtype=torch.int64, device=device
        ),
        "gmnw_trace_task_state_after": torch.full(
            task_shape, -1, dtype=torch.int64, device=device
        ),
    }


def _allocate_birth_outputs(
    *,
    capacity: int,
    max_births_per_tick: int,
    device: torch.device,
    output_ticks: int = OUTPUT_TICKS_V1,
    batch_cells: int = BATCH_CELLS_V1,
    genome_v2_causal_record_capacity: int = 0,
) -> dict[str, Any]:
    births = _validated_birth_bound(max_births_per_tick, capacity)
    if type(output_ticks) is not int or output_ticks < 1:
        raise ValueError("output_ticks must be a positive exact integer")
    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 256:
        raise ValueError("batch_cells must be an exact integer in [1,256]")
    b, t = batch_cells, output_ticks
    if (
        type(genome_v2_causal_record_capacity) is not int
        or genome_v2_causal_record_capacity < 0
        or genome_v2_causal_record_capacity
        > GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1
    ):
        raise ValueError("genome_v2_causal_record_capacity is invalid")
    causal_capacity = genome_v2_causal_record_capacity
    outputs: dict[str, Any] = {
        "birth_slot_free_queue_head": torch.zeros(
            (b,), dtype=torch.int64, device=device
        ),
        "birth_planned_repro_cooldown": torch.zeros(
            (b, capacity), dtype=torch.int32, device=device
        ),
        "birth_planned_parent_slots": torch.full(
            (b, t, births), -1, dtype=torch.int64, device=device
        ),
        "birth_planned_child_slots": torch.full(
            (b, t, births), -1, dtype=torch.int64, device=device
        ),
        "birth_planned_spawn_positions": torch.full(
            (b, t, births, 2), -1, dtype=torch.int32, device=device
        ),
        "birth_planned_child_energy": torch.full(
            (b, t, births), float("nan"), dtype=torch.float32, device=device
        ),
        "birth_decision_parent_slots": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "birth_decision_reason_codes": torch.full(
            (b, t, capacity), -1, dtype=torch.int32, device=device
        ),
        "birth_decision_candidate_counts": torch.zeros(
            (b, t, capacity), dtype=torch.int32, device=device
        ),
        "birth_decision_draw_starts": torch.zeros(
            (b, t, capacity), dtype=torch.int64, device=device
        ),
        "birth_decision_draw_ends": torch.zeros(
            (b, t, capacity), dtype=torch.int64, device=device
        ),
        "birth_decision_selected_positions": torch.full(
            (b, t, capacity, 2), -1, dtype=torch.int32, device=device
        ),
        "birth_decision_selected_reserved": torch.zeros(
            (b, t, capacity), dtype=torch.bool, device=device
        ),
        "birth_reason_counts": torch.zeros(
            (b, t, 10), dtype=torch.int64, device=device
        ),
        "birth_committed_parent_agent_ids": torch.full(
            (b, t, births), -1, dtype=torch.int64, device=device
        ),
        "birth_committed_child_agent_ids": torch.full(
            (b, t, births), -1, dtype=torch.int64, device=device
        ),
        "birth_committed_child_slots": torch.full(
            (b, t, births), -1, dtype=torch.int64, device=device
        ),
        "birth_policy_mutation_delta_total": torch.zeros(
            (b, t), dtype=torch.float64, device=device
        ),
        "birth_policy_mutation_occurred": torch.zeros(
            (b, t, births), dtype=torch.bool, device=device
        ),
        "birth_policy_mutation_delta": torch.zeros(
            (b, t, births), dtype=torch.float32, device=device
        ),
        "birth_planner_pcg64_state_before": torch.zeros(
            (b, t, 4), dtype=torch.uint64, device=device
        ),
        "birth_planner_pcg64_buffer_before": torch.zeros(
            (b, t, 2), dtype=torch.int64, device=device
        ),
        "birth_planned_mate_slots": torch.full(
            (b, t, births), -1, dtype=torch.int64, device=device
        ),
        "birth_committed_mate_agent_ids": torch.full(
            (b, t, births), -1, dtype=torch.int64, device=device
        ),
        "birth_mate_planned_energy": torch.zeros(
            (b, capacity), dtype=torch.float64, device=device
        ),
        "birth_mate_planned_child_energy_exact": torch.full(
            (b, t, births), float("nan"), dtype=torch.float64, device=device
        ),
        "birth_mate_planned_parent_energy_after": torch.full(
            (b, t, births), float("nan"), dtype=torch.float64, device=device
        ),
        "birth_mate_planned_mate_energy_after": torch.full(
            (b, t, births), float("nan"), dtype=torch.float64, device=device
        ),
        "birth_decision_mate_slots": torch.full(
            (b, t, capacity), -1, dtype=torch.int64, device=device
        ),
        "birth_decision_adjacent_counts": torch.zeros(
            (b, t, capacity), dtype=torch.int32, device=device
        ),
        "birth_decision_spawn_candidate_counts": torch.zeros(
            (b, t, capacity), dtype=torch.int32, device=device
        ),
        "birth_mate_reason_counts": torch.zeros(
            (b, t, 12), dtype=torch.int64, device=device
        ),
        "birth_mate_decision_reason_codes": torch.full(
            (b, t, capacity), -1, dtype=torch.int32, device=device
        ),
        "birth_genome_v2_causal_record_rows": torch.full(
            (b, causal_capacity, GENOME_V2_CAUSAL_RECORD_WIDTH_V1),
            torch.iinfo(torch.int64).min,
            dtype=torch.int64,
            device=device,
        ),
        "birth_genome_v2_causal_record_valid": torch.zeros(
            (b, causal_capacity), dtype=torch.bool, device=device
        ),
        "birth_genome_v2_causal_record_count": torch.zeros(
            (b,), dtype=torch.int64, device=device
        ),
        "birth_genome_v2_causal_record_dropped_count": torch.zeros(
            (b,), dtype=torch.int64, device=device
        ),
        "birth_genome_v2_causal_capture_enabled": torch.zeros(
            (b,), dtype=torch.bool, device=device
        ),
    }
    int64_rows = (
        "birth_planned_count",
        "birth_decision_count",
        "birth_spawn_rng_draws",
        "birth_spawn_rng_semantic_draws",
        "birth_runner_rng_draws",
        "birth_runner_rng_semantic_draws",
        "birth_identity_committed_count",
        "birth_scalar_committed_count",
        "birth_scalar_semantic_draws",
        "birth_drive_committed_count",
        "birth_noise_rows_consumed",
        "birth_exp_prior_rows_consumed",
        "birth_standard_normal_rows",
        "birth_drive_semantic_draws",
        "birth_policy_committed_count",
        "birth_policy_semantic_draws",
        "birth_policy_mutation_events",
        "birth_policy_mutated_elements",
    )
    int32_rows = (
        "birth_plan_status",
        "birth_identity_status",
        "birth_scalar_status",
        "birth_drive_status",
        "birth_chemistry_status",
        "birth_policy_status",
        "birth_combined_status",
    )
    outputs.update(
        {
            name: torch.zeros((b, t), dtype=torch.int64, device=device)
            for name in int64_rows
        }
    )
    outputs.update(
        {
            name: torch.zeros((b, t), dtype=torch.int32, device=device)
            for name in int32_rows
        }
    )
    return outputs


def _allocate_turnover_outputs(
    *,
    capacity: int,
    generation: int,
    device: torch.device,
    batch_cells: int = BATCH_CELLS_V1,
    genome_v2_causal_record_capacity: int = 0,
) -> dict[str, Any]:
    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 256:
        raise ValueError("batch_cells must be an exact integer in [1,256]")
    if (
        type(genome_v2_causal_record_capacity) is not int
        or genome_v2_causal_record_capacity < 0
        or genome_v2_causal_record_capacity
        > GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1
    ):
        raise ValueError("genome_v2_causal_record_capacity is invalid")
    causal_capacity = genome_v2_causal_record_capacity
    outputs = {
        name: torch.full((batch_cells, capacity), -1, dtype=torch.int64, device=device)
        for name in TURNOVER_OUTPUT_FIELDS[:7]
    }
    outputs.update(
        {
            "turnover_counts": torch.zeros(
                (batch_cells, 8), dtype=torch.int64, device=device
            ),
            "turnover_status": torch.zeros(
                (batch_cells,), dtype=torch.int32, device=device
            ),
            "turnover_next_generation": torch.full(
                (batch_cells,), generation, dtype=torch.int64, device=device
            ),
            "turnover_next_tick": torch.full(
                (batch_cells,),
                TICKS_PER_GENERATION_V1,
                dtype=torch.int64,
                device=device,
            ),
            "turnover_genome_v2_causal_record_rows": torch.full(
                (
                    batch_cells,
                    causal_capacity,
                    GENOME_V2_CAUSAL_RECORD_WIDTH_V1,
                ),
                torch.iinfo(torch.int64).min,
                dtype=torch.int64,
                device=device,
            ),
            "turnover_genome_v2_causal_record_valid": torch.zeros(
                (batch_cells, causal_capacity),
                dtype=torch.bool,
                device=device,
            ),
            "turnover_genome_v2_causal_record_count": torch.zeros(
                (batch_cells,), dtype=torch.int64, device=device
            ),
            "turnover_genome_v2_causal_record_dropped_count": torch.zeros(
                (batch_cells,), dtype=torch.int64, device=device
            ),
            "turnover_genome_v2_causal_capture_enabled": torch.zeros(
                (batch_cells,), dtype=torch.bool, device=device
            ),
        }
    )
    return outputs


def _enable_authority_bound_genome_v2_causal_capture_v1(
    persistent_state: Mapping[str, Any],
    *,
    checkpoint_authority: tuple[tuple[str, str], ...],
) -> None:
    if not checkpoint_authority:
        return
    for producer in ("birth", "turnover"):
        name = f"{producer}_genome_v2_causal_capture_enabled"
        tensor = persistent_state.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.bool
            or tensor.layout != torch.strided
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise _fail(f"{name} must be exact resident bool cell storage")
        tensor.fill_(True)


def _canonical_projection(
    projection: Any,
    *,
    source_generation: int,
) -> dict[str, Any]:
    metadata = _required_mapping(
        _projection_member(projection, "metadata"), name="projection.metadata"
    )
    tensors = _required_mapping(
        _projection_member(projection, "tensors"), name="projection.tensors"
    )
    active_event_count = int(metadata.get("event_tape_active_count", 0) or 0)
    defaults = {
        "restore_count": 0,
        "generation_calls": 0,
        "event_cursor": active_event_count,
        "runtime_event_cursor": active_event_count,
        "last_applied_actions": 0,
        "last_moved": 0,
        "last_gathered": 0,
        "last_policy_refill_calls": 0,
        "last_policy_refill_rows": 0,
        "pending_generation_free_agent_ids": (),
        "pending_generation_free_slots": (),
        "last_world": "none",
    }
    runtime_prefix = {
        name: copy.deepcopy(
            _projection_member(
                projection,
                name,
                defaults[name] if source_generation == 0 else _MISSING,
            )
        )
        for name in CANONICAL_RUNTIME_PREFIX_FIELDS
    }
    for name in (
        "restore_count",
        "generation_calls",
        "event_cursor",
        "runtime_event_cursor",
        "last_applied_actions",
        "last_moved",
        "last_gathered",
        "last_policy_refill_calls",
        "last_policy_refill_rows",
    ):
        if type(runtime_prefix[name]) is not int or runtime_prefix[name] < 0:
            raise _fail(f"import projection {name} must be a nonnegative exact integer")
    for name in (
        "pending_generation_free_agent_ids",
        "pending_generation_free_slots",
    ):
        value = runtime_prefix[name]
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not int for item in value
        ):
            raise _fail(f"import projection {name} must be an integer list or tuple")
        runtime_prefix[name] = list(value)
    if type(runtime_prefix["last_world"]) is not str:
        raise _fail("import projection last_world must be an exact string")
    canonical = _with_checkpoint_contract_metadata(
        {
            "schema_version": CHECKPOINT_PROJECTION_SCHEMA_VERSION,
            "metadata": copy.deepcopy(metadata),
            # The import projection is an immutable source authority.  The
            # resident boundary owns all mutable CUDA state, so retaining
            # detached views here avoids cloning the entire B136 source bank
            # into anonymous host RAM before settlement.
            "tensors": {name: value.detach() for name, value in tensors.items()},
            **runtime_prefix,
        }
    )
    for name in _INHERITANCE_SIDECAR_FIELDS_V1:
        sidecar = _projection_member(projection, name, None)
        if sidecar is not None:
            canonical[name] = copy.deepcopy(sidecar)
    return canonical


def _tick_native_allocator_projection_v1(
    control: Mapping[str, Any],
) -> tuple[int, int]:
    """Derive private source-buffer allocation width from absolute-tick controls.

    The returned count is consumed only by the still-generation-shaped direct
    CUDA allocation ABI.  It is a bounded source-capacity width, never a
    simulation horizon, lifecycle count, checkpoint coordinate, or RNG axis.
    """

    forbidden = {
        "generations",
        "generation_count",
        "logical_generation_count",
        "physical_window_generations",
        "ticks_per_generation",
        "refill_chunk_ticks",
        "legacy_turnover_interval_ticks",
        "internal_lifecycle_interval_ticks",
        "internal_lifecycle_schedule",
    }
    contaminated = sorted(forbidden.intersection(control))
    if contaminated:
        raise _fail(
            f"tick-native allocator control contains retired authority: {contaminated}"
        )
    if control.get("time_authority") != "absolute_tick_v1":
        raise _fail("tick-native source control must use absolute_tick_v1")
    for name in (
        "total_ticks",
        "resident_owner_span_ticks",
        "native_dispatch_window_ticks",
        "source_window_ticks",
        "event_window_capacity",
    ):
        value = control.get(name)
        if type(value) is not int or value <= 0:
            raise _fail(f"tick-native source control {name} is invalid")
    if control["native_dispatch_window_ticks"] > control["resident_owner_span_ticks"]:
        raise _fail("tick-native dispatch window exceeds owner span")
    if control["source_window_ticks"] > control["resident_owner_span_ticks"]:
        raise _fail("tick-native source window exceeds owner span")
    source_window_ticks = int(control["source_window_ticks"])
    private_source_capacity_window_count = max(
        1,
        (
            int(control["native_dispatch_window_ticks"])
            + source_window_ticks
            - 1
        )
        // source_window_ticks,
    )
    return private_source_capacity_window_count, source_window_ticks


def _tick_native_private_source_capacity_window_count_v1(
    control: Mapping[str, Any],
) -> int:
    """Return the private identity-shadow width required by the source buffer."""

    private_source_capacity_window_count, _source_window_ticks = (
        _tick_native_allocator_projection_v1(control)
    )
    return max(
        PRIVATE_SOURCE_CAPACITY_MIN_WINDOWS_V1,
        private_source_capacity_window_count,
    )


def allocate_direct_resident_source_v1(
    request: Any,
    *,
    per_cell_requests: Sequence[Any] | None = None,
    per_cell_source_groups: Sequence[int] | None = None,
    per_cell_ids: Sequence[str] | None = None,
) -> DirectResidentSourceV1:
    """Allocate the qualified Patch P2 configurable-birth reusable-g2 envelope."""

    envelope = request.envelope
    native_control = dict(getattr(request, "tick_native_control", {}) or {})
    native_tick_route = bool(native_control)
    if native_tick_route:
        # The direct ABI still asks for generation-shaped allocation members;
        # keep them private and derive them only from the bounded source
        # buffer width.  They are not active time, lifecycle, checkpoint, or
        # RNG authorities.
        private_source_capacity_window_count, physical_window_ticks = (
            _tick_native_allocator_projection_v1(native_control)
        )
        physical_window_generations = 1
        private_source_capacity_window_count = (
            _tick_native_private_source_capacity_window_count_v1(native_control)
        )
        logical_generation_count = private_source_capacity_window_count
    else:
        logical_generation_count = getattr(
            request, "logical_generation_count", LOGICAL_GENERATIONS_V1
        )
        physical_window_generations = _validated_physical_window_generations_v1(
            getattr(
                request,
                "physical_window_generations",
                PHYSICAL_WINDOW_GENERATIONS_V1,
            )
        )
        physical_window_ticks = physical_window_ticks_v1(physical_window_generations)
        private_source_capacity_window_count = None
    native_event_capacity = getattr(request, "event_window_capacity", None)
    if native_tick_route:
        if type(native_event_capacity) is not int:
            native_event_capacity = native_control["event_window_capacity"]
        if native_event_capacity != native_control["event_window_capacity"]:
            raise _fail("tick-native event capacity is not plan-bound")
        # The active native owner is sized from the bounded source window.
        # Request generation fields are ignored so horizon or facade tampering
        # cannot change resident allocation.
    allocation_generation_count = (
        private_source_capacity_window_count
        if native_tick_route
        else getattr(request, "agent_id_allocation_generation_count", None)
    )
    minimum_generation_count = (
        1
        if native_tick_route
        else (
            1
            if getattr(
                request,
                "allow_short_technical_gate_generation_count",
                False,
            )
            is True
            else 3
        )
    )
    batch_cells = getattr(request, "batch_cells", BATCH_CELLS_V1)
    if (
        type(logical_generation_count) is not int
        or logical_generation_count < minimum_generation_count
        or logical_generation_count > 128
    ):
        raise _fail(
            "logical_generation_count must be an integer in "
            f"[{minimum_generation_count}, 128]"
        )
    if type(batch_cells) is not int or batch_cells < 1 or batch_cells > 256:
        raise _fail("batch_cells must be an exact integer in [1,256]")
    if native_tick_route:
        _validate_physical_b136_native_event_capacity_v1(
            request,
            batch_cells=batch_cells,
            native_event_capacity=native_event_capacity,
        )
    turnover_mlp_shadow_elide_enable = getattr(
        request,
        "turnover_mlp_shadow_elide_enable",
        False,
    )
    if type(turnover_mlp_shadow_elide_enable) is not bool:
        raise _fail("turnover_mlp_shadow_elide_enable must be an exact bool")
    inheritance_sidecar_cell_id = getattr(request, "inheritance_sidecar_cell_id", None)
    if inheritance_sidecar_cell_id is not None and (
        type(inheritance_sidecar_cell_id) is not str
        or not inheritance_sidecar_cell_id
        or inheritance_sidecar_cell_id.strip() != inheritance_sidecar_cell_id
    ):
        raise _fail("request inheritance sidecar cell_id must be stable text")
    genome_v2_causal_physical_cell_ordinal = getattr(
        request,
        "genome_v2_causal_physical_cell_ordinal",
        0,
    )
    if (
        type(genome_v2_causal_physical_cell_ordinal) is not int
        or genome_v2_causal_physical_cell_ordinal < 0
    ):
        raise _fail(
            "request Genome V2 causal physical cell ordinal must be nonnegative"
        )
    cell_requests: tuple[Any, ...] | None = None
    source_groups: tuple[int, ...] | None = None
    cell_ids: tuple[str, ...] | None = None
    if per_cell_ids is not None:
        if isinstance(per_cell_ids, (str, bytes)) or not isinstance(
            per_cell_ids, Sequence
        ):
            raise _fail("per_cell_ids must be a sequence")
        cell_ids = tuple(per_cell_ids)
        if (
            len(cell_ids) != batch_cells
            or any(
                type(cell_id) is not str or not cell_id or cell_id.strip() != cell_id
                for cell_id in cell_ids
            )
            or len(set(cell_ids)) != batch_cells
        ):
            raise _fail("per_cell_ids must contain one unique stable ID per cell")
    elif batch_cells == 1 and inheritance_sidecar_cell_id is not None:
        cell_ids = (inheritance_sidecar_cell_id,)
    if per_cell_requests is not None:
        if isinstance(per_cell_requests, (str, bytes)) or not isinstance(
            per_cell_requests, Sequence
        ):
            raise _fail("per_cell_requests must be a sequence")
        cell_requests = tuple(per_cell_requests)
        if len(cell_requests) != batch_cells:
            raise _fail("per_cell_requests count must exactly equal batch_cells")
        if not cell_requests or cell_requests[0] is not request:
            raise _fail("per_cell_requests[0] must be the primary request")
        if per_cell_source_groups is not None:
            if isinstance(per_cell_source_groups, (str, bytes)) or not isinstance(
                per_cell_source_groups, Sequence
            ):
                raise _fail("per_cell_source_groups must be a sequence")
            source_groups = tuple(per_cell_source_groups)
            if len(source_groups) != batch_cells or any(
                type(group) is not int or group < 0 for group in source_groups
            ):
                raise _fail(
                    "per_cell_source_groups must contain one nonnegative integer "
                    "per physical cell"
                )
            unique_groups = sorted(set(source_groups))
            if unique_groups != list(range(len(unique_groups))):
                raise _fail("per_cell_source_groups must be contiguous from zero")
        if len({id(owner) for owner in cell_requests}) != batch_cells:
            raise _fail("per-cell requests must be distinct objects")
        for owner_name in ("cuda_boundary", "generation_birth_state"):
            owners = tuple(getattr(item, owner_name, None) for item in cell_requests)
            if source_groups is None:
                if len({id(owner) for owner in owners}) != batch_cells:
                    raise _fail(f"per-cell {owner_name} must not alias")
                continue
            owner_by_group: dict[int, int] = {}
            group_by_owner: dict[int, int] = {}
            for cell, (group, owner) in enumerate(
                zip(source_groups, owners, strict=True)
            ):
                owner_id = id(owner)
                previous_owner = owner_by_group.setdefault(group, owner_id)
                if previous_owner != owner_id:
                    raise _fail(
                        f"per-cell {owner_name} group {group} changed owner at cell {cell}"
                    )
                previous_group = group_by_owner.setdefault(owner_id, group)
                if previous_group != group:
                    raise _fail(f"per-cell {owner_name} aliases across source groups")
        _validated_per_cell_projection_tensors_v1(
            tuple(item.import_projection for item in cell_requests),
            batch_cells=batch_cells,
        )
        shared_request_fields = (
            "cuda_device",
            "event_capacity",
            "logical_generation_count",
            "physical_window_generations",
            "agent_id_allocation_generation_count",
            "batch_cells",
            "observation_extra_columns",
            "output_ticks",
            "cpu_request_staging_enable",
            "direct_batch_host_import_enable",
            "tick_native_control",
            "event_window_capacity",
        )
        for cell, item in enumerate(cell_requests[1:], start=1):
            if not _same_configuration_except_founder_profile_v1(
                item.configuration, request.configuration
            ):
                raise _fail(
                    f"per-cell request {cell} shared field 'configuration' mismatch"
                )
            for name in shared_request_fields:
                if not _same_structured_value_v1(
                    getattr(item, name, None), getattr(request, name, None)
                ):
                    raise _fail(
                        f"per-cell request {cell} shared field {name!r} mismatch"
                    )
            if getattr(item, "immutable_shared_gru_pack", None) is not getattr(
                request, "immutable_shared_gru_pack", None
            ):
                raise _fail(
                    f"per-cell request {cell} immutable shared GRU pack mismatch"
                )
        primary_envelope = dataclasses.asdict(request.envelope)
        primary_envelope.pop("policy_family", None)
        for cell, item in enumerate(cell_requests):
            envelope_fields = dataclasses.asdict(item.envelope)
            envelope_fields.pop("policy_family", None)
            if not _same_structured_value_v1(envelope_fields, primary_envelope):
                raise _fail(
                    f"per-cell request {cell} non-policy envelope fields mismatch"
                )
            metadata = _required_mapping(
                _projection_member(item.import_projection, "metadata"),
                name=f"per_cell_requests[{cell}].projection.metadata",
            )
            projection_family = (
                str(metadata.get("native_runtime_fixed_policy_family", ""))
                .strip()
                .lower()
            )
            expected_family = {
                "mlp": "stochastic_fixed_mlp",
                "gru": "stochastic_stable_gru",
            }.get(projection_family)
            if (
                expected_family is None
                or item.envelope.policy_family != expected_family
            ):
                raise _fail(f"per-cell request {cell} envelope policy family mismatch")
    import_metadata = _required_mapping(
        _projection_member(request.import_projection, "metadata"),
        name="projection.metadata",
    )
    retained_agent_id_capacity = _common_retained_agent_id_capacity_v1(
        tuple(item.import_projection for item in cell_requests)
        if cell_requests is not None
        else (request.import_projection,)
    )
    projection_family = (
        str(import_metadata.get("native_runtime_fixed_policy_family", ""))
        .strip()
        .lower()
    )
    expected_policy_family = {
        "mlp": "stochastic_fixed_mlp",
        "gru": "stochastic_stable_gru",
    }.get(projection_family)
    if expected_policy_family is None:
        raise _fail("qualified source requires an explicit MLP or GRU family")
    expected_population = import_metadata.get("population_size")
    expected_grid = getattr(envelope, "grid", None)
    if (
        type(expected_population) is not int
        or type(expected_grid) is not tuple
        or (expected_population, expected_grid) not in QUALIFIED_PATCH_GEOMETRIES_V1
    ):
        raise _fail("qualified source requires an admitted Patch geometry")
    expected_envelope = {
        "world": "patch",
        "profile": "P2",
        "population": expected_population,
        "grid": expected_grid,
        "policy_family": expected_policy_family,
        # Each envelope describes one independently constructed logical cell.
        # Physical batch extent is carried by the allocation request and may
        # vary within the admitted B1..B72 range.
        "batch_cells": 1,
        "checkpoint_schedule": "terminal_only",
    }
    if not native_tick_route:
        # Legacy callers retain their old envelope contract.  The active
        # native route never reads generation-shaped request fields here.
        expected_envelope.update(
            {
                "generation_count": logical_generation_count,
                "ticks_per_generation": _validated_ticks_per_generation_v1(
                    getattr(envelope, "ticks_per_generation", None)
                ),
            }
        )
    for name, expected in expected_envelope.items():
        actual = getattr(envelope, name, None)
        if type(actual) is not type(expected) or actual != expected:
            raise _fail(f"qualified envelope {name} mismatch")
    config = request.configuration
    lifecycle_configurations = (
        tuple(item.configuration for item in cell_requests)
        if cell_requests is not None
        else (config,) * batch_cells
    )
    move_cost_by_cell: list[float] = []
    turnover_selection_seeds: list[int] = []
    ordinary_birth_authorities: list[bool] = []
    for cell, cell_configuration in enumerate(lifecycle_configurations):
        move_cost = getattr(cell_configuration, "move_cost", None)
        if (
            type(move_cost) is not float
            or not math.isfinite(move_cost)
            or move_cost < 0.0
        ):
            raise _fail(
                f"per-cell configuration {cell} move_cost must be a finite "
                "nonnegative float"
            )
        move_cost_by_cell.append(move_cost)
        ordinary_births_enabled = getattr(
            cell_configuration, "ordinary_births_enabled", None
        )
        if ordinary_births_enabled is None:
            configured_max_births = getattr(
                cell_configuration, "max_births_per_tick", None
            )
            if type(configured_max_births) is not int:
                raise _fail(
                    f"per-cell configuration {cell} max_births_per_tick "
                    "must be an exact integer"
                )
            ordinary_births_enabled = configured_max_births > 0
        elif type(ordinary_births_enabled) is not bool:
            raise _fail(
                f"per-cell configuration {cell} ordinary_births_enabled "
                "must be an exact bool"
            )
        ordinary_birth_authorities.append(ordinary_births_enabled)
        selection_mode = getattr(cell_configuration, "turnover_selection_mode", 0)
        if type(selection_mode) is not int or selection_mode not in {0, 1}:
            raise _fail(
                f"per-cell configuration {cell} turnover_selection_mode "
                "must be exact integer 0 or 1"
            )
        lifecycle_mode = getattr(cell_configuration, "lifecycle_mode", 0)
        if type(lifecycle_mode) is not int or lifecycle_mode not in {0, 1}:
            raise _fail(
                f"per-cell configuration {cell} lifecycle_mode "
                "must be exact integer 0 or 1"
            )
        expected_generation_turnover = lifecycle_mode == 0
        generation_turnover = getattr(
            cell_configuration, "generation_turnover_enabled", None
        )
        if (
            type(generation_turnover) is not bool
            or generation_turnover is not expected_generation_turnover
        ):
            raise _fail(
                f"per-cell configuration {cell} generation_turnover_enabled "
                "must match lifecycle_mode"
            )
        selection_seed = getattr(cell_configuration, "turnover_selection_seed", 0)
        if type(selection_seed) is not int or not 0 <= selection_seed <= (1 << 64) - 1:
            raise _fail(
                f"per-cell configuration {cell} turnover_selection_seed "
                "must be an exact uint64 integer"
            )
        turnover_selection_seeds.append(selection_seed)
    if len(set(ordinary_birth_authorities)) != 1:
        raise _fail("per-cell ordinary_births_enabled must be uniform within one batch")
    ordinary_births_enabled = ordinary_birth_authorities[0]
    inheritance_profile = str(
        getattr(config, "inheritance_profile", "legacy_midpoint_v1")
        or "legacy_midpoint_v1"
    )
    diploid_route = inheritance_profile == DIPLOID_INHERITANCE_PROFILE
    if bool(getattr(config, "genome_upgrade_v2_enable", False)) and not diploid_route:
        raise _fail(
            "GENOME-UPGRADE-V2 requires the qualified diploid inheritance profile"
        )
    source_generation = _validated_source_generation(
        request.import_projection,
        config,
    )
    if (
        type(config.start_tick) is not int
        or not 0 <= config.start_tick < config.steps
        or type(config.generation_turnover_enabled) is not bool
        or config.generation_turnover_enabled is not (config.lifecycle_mode == 0)
        or type(config.reproduce_requires_mate) is not bool
    ):
        raise _fail(
            "qualified configuration must start inside its internal lifecycle "
            "interval with an explicit birth mode and lifecycle-matched turnover"
        )
    if diploid_route and not config.reproduce_requires_mate:
        raise _fail(
            "qualified diploid configuration requires profile-matched mate mode"
        )

    cpu_request_staging_enable = getattr(
        request,
        "cpu_request_staging_enable",
        False,
    )
    if type(cpu_request_staging_enable) is not bool:
        raise _fail("cpu_request_staging_enable must be an exact bool")
    direct_batch_host_import_enable = getattr(
        request,
        "direct_batch_host_import_enable",
        False,
    )
    if type(direct_batch_host_import_enable) is not bool:
        raise _fail("direct_batch_host_import_enable must be an exact bool")
    if direct_batch_host_import_enable and cpu_request_staging_enable:
        raise _fail(
            "direct batch host import cannot be combined with CPU resident staging"
        )
    sources = (
        tuple(_source_mapping(item) for item in cell_requests)
        if cell_requests is not None
        else None
    )
    source = sources[0] if sources is not None else _source_mapping(request)
    source_device = source["energy"].device
    expected_source_device = (
        torch.device("cpu")
        if cpu_request_staging_enable or direct_batch_host_import_enable
        else source_device
    )
    _validate_source_tensor_devices_v1(
        source,
        expected_device=expected_source_device,
        name="primary",
    )
    if cpu_request_staging_enable or direct_batch_host_import_enable:
        device = torch.device(getattr(request, "cuda_device", ""))
        if device.type != "cuda":
            if cpu_request_staging_enable:
                raise _fail("CPU-staged request allocation requires a CUDA target device")
            raise _fail(
                "direct batch host-import request allocation requires a CUDA target device"
            )
    else:
        device = source_device
    if direct_batch_host_import_enable and (
        cell_requests is None or batch_cells < 2 or source_device.type != "cpu"
    ):
        raise _fail(
            "direct batch host import requires a multicell CPU owner source"
        )
    if not cpu_request_staging_enable and device.type != "cuda":
        raise _fail(
            "the production source allocator requires CUDA-resident boundary state"
        )
    turnover_selection_seed_by_cell = torch.tensor(
        turnover_selection_seeds,
        dtype=torch.uint64,
        device=device,
    ).contiguous()
    capacity = int(source["energy"].numel())
    hapb_host_policy_enable = "mlp_pack" in getattr(
        request.cuda_boundary, "host_authoritative_tensors", {}
    )
    hapb_policy_tile_rows = 512
    if hapb_host_policy_enable:
        if (
            batch_cells < 1
            or batch_cells > 256
            or capacity != 512
            or sources is None
            or len(sources) != batch_cells
            or any(
                "mlp_pack"
                not in getattr(item.cuda_boundary, "host_authoritative_tensors", {})
                for item in cell_requests
            )
        ):
            raise _fail("HAPB host-policy import requires one physical B/C512 batch")
        if any(
            not isinstance(item["mlp_pack"], torch.Tensor)
            or item["mlp_pack"].device.type != "cpu"
            or item["mlp_pack"].dtype != torch.float32
            or tuple(item["mlp_pack"].shape) != (capacity, 9991)
            for item in sources
        ):
            raise _fail("HAPB host-policy source rows must be CPU float32[C512,9991]")
        primary_metadata = getattr(request.cuda_boundary, "metadata", {})
        if not isinstance(primary_metadata, Mapping):
            raise _fail("HAPB boundary metadata is malformed")
        requested_tile_rows = primary_metadata.get("hapb_policy_tile_rows", 512)
        if type(requested_tile_rows) is not int or requested_tile_rows not in {
            512,
            4096,
        }:
            raise _fail("HAPB policy tile rows must be 512 or 4096")
        hapb_policy_tile_rows = requested_tile_rows
        for cell, item in enumerate(cell_requests):
            metadata = getattr(item.cuda_boundary, "metadata", {})
            if not isinstance(metadata, Mapping):
                raise _fail(f"HAPB boundary metadata is malformed for cell {cell}")
            cell_tile_rows = metadata.get("hapb_policy_tile_rows", 512)
            if cell_tile_rows != hapb_policy_tile_rows:
                raise _fail("HAPB policy tile rows must be identical across B48 cells")
    max_births_per_tick = _validated_birth_bound(config.max_births_per_tick, capacity)
    if config.reproduce_requires_mate and max_births_per_tick <= 2:
        raise _fail(
            "mate-required direct resident births require runtime-sized "
            "three-role scratch (max_births_per_tick > 2)"
        )
    envelope_births = getattr(envelope, "max_births_per_tick", None)
    if type(envelope_births) is not int or envelope_births != max_births_per_tick:
        raise _fail(
            "qualified envelope max_births_per_tick must exactly match configuration"
        )
    _validate_birth_state_receipt_v1(
        request.generation_birth_state,
        max_births_per_tick=max_births_per_tick,
        slot_capacity=capacity,
        physical_window_generations=physical_window_generations,
        ordinary_births_enabled=ordinary_births_enabled,
        turnover_refill_capacity=_runtime_turnover_refill_capacity_v1(
            config,
            population_size=expected_population,
            slot_capacity=capacity,
            ordinary_births_enabled=ordinary_births_enabled,
            physical_window_generations=physical_window_generations,
        ),
    )
    if cell_requests is not None:
        for cell, item in enumerate(cell_requests[1:], start=1):
            if (
                _validated_source_generation(item.import_projection, item.configuration)
                != source_generation
            ):
                raise _fail(f"per-cell request {cell} source generation mismatch")
            cell_source = sources[cell]
            if cell_source["energy"].device != source_device:
                raise _fail(f"per-cell request {cell} source device mismatch")
            _validate_source_tensor_devices_v1(
                cell_source,
                expected_device=expected_source_device,
                name=f"per-cell request {cell}",
            )
            if int(cell_source["energy"].numel()) != capacity:
                raise _fail(f"per-cell request {cell} source capacity mismatch")
            _validate_birth_state_receipt_v1(
                item.generation_birth_state,
                max_births_per_tick=max_births_per_tick,
                slot_capacity=capacity,
                physical_window_generations=physical_window_generations,
                ordinary_births_enabled=ordinary_births_enabled,
                turnover_refill_capacity=_runtime_turnover_refill_capacity_v1(
                    item.configuration,
                    population_size=expected_population,
                    slot_capacity=capacity,
                    ordinary_births_enabled=ordinary_births_enabled,
                    physical_window_generations=physical_window_generations,
                ),
            )
    initial_ids = source["agent_ids_by_alive_order"]
    per_cell_initial_ids = (
        [item["agent_ids_by_alive_order"] for item in sources]
        if sources is not None
        else [initial_ids]
    )
    initial_agent_counts = tuple(int(value.numel()) for value in per_cell_initial_ids)
    identity_states = (
        tuple(item.generation_birth_state.identity_state for item in cell_requests)
        if cell_requests is not None
        else (request.generation_birth_state.identity_state,)
    )
    current_id_capacity = capacity
    population_next_id = max(
        int(identity_state["population_next_id"].reshape(-1)[0].item())
        for identity_state in identity_states
    )
    _required_resident_agent_id_capacity_v1(
        current_id_capacity,
        population_next_id,
        max_births_per_tick,
        capacity,
        logical_generation_count=logical_generation_count,
        allocation_generation_count=allocation_generation_count,
        retained_capacity=retained_agent_id_capacity,
    )
    if not _source_cardinality_is_qualified_v1(
        source_generation=source_generation,
        source_tick=int(config.start_tick),
        expected_population=expected_population,
        capacity=capacity,
        initial_agent_counts=initial_agent_counts,
    ):
        raise _fail(
            "the qualified source requires profile-matched generation-zero agents or a "
            "resumed boundary within resident capacity"
        )
    view_radius = int(config.view_radius)
    base_observation_size = 8 * (2 * view_radius + 1) ** 2 + 2
    if base_observation_size != 74:
        raise _fail("the qualified Patch P2 observation width must equal 74")
    observation_extra_columns = getattr(request, "observation_extra_columns", 0)
    if (
        type(observation_extra_columns) is not int
        or observation_extra_columns < 0
        or observation_extra_columns > 64
    ):
        raise _fail("observation_extra_columns must be an exact integer in [0,64]")
    observation_size = base_observation_size + observation_extra_columns
    output_ticks = getattr(request, "output_ticks", OUTPUT_TICKS_V1)
    if type(output_ticks) is not int or output_ticks < 1 or output_ticks > 160:
        raise _fail("output_ticks must be an exact integer in [1,160]")
    shared_gru_pack = getattr(request, "immutable_shared_gru_pack", None)
    if shared_gru_pack is not None:
        expected_shared_width = int(source["gru_pack"].size(1)) + (
            64 * observation_extra_columns
        )
        if (
            projection_family != "gru"
            or import_metadata.get("policy_weight_mutation_enable") is not False
            or not isinstance(shared_gru_pack, torch.Tensor)
            or shared_gru_pack.device.type != "cpu"
            or shared_gru_pack.dtype != torch.float32
            or shared_gru_pack.ndim != 1
            or shared_gru_pack.numel() != expected_shared_width
            or shared_gru_pack.requires_grad
            or not shared_gru_pack.is_contiguous()
        ):
            raise _fail("immutable shared GRU pack contract is invalid")

    canonical = _canonical_projection(
        request.import_projection,
        source_generation=source_generation,
    )
    canonical_by_cell = (
        tuple(
            _canonical_projection(
                item.import_projection,
                source_generation=_validated_source_generation(
                    item.import_projection, item.configuration
                ),
            )
            for item in cell_requests
        )
        if cell_requests is not None
        else (canonical,)
    )
    streamed_bwide_seed = _is_streamed_bwide_seed_v1(
        cell_requests=cell_requests,
        per_cell_ids=cell_ids,
        batch_cells=batch_cells,
        canonical_projection_count=len(canonical_by_cell),
    )
    batched_names = tuple(
        dict.fromkeys(
            (
                *BASE_BATCHED_FIELDS,
                *ORDINARY_BIRTH_IDENTITY_FIELDS,
                *ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
                *ORDINARY_BIRTH_GENOME_INT32_FIELDS,
                *ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
                *ORDINARY_BIRTH_DRIVE_FIELDS,
                *ORDINARY_BIRTH_POLICY_FIELDS,
                "policy_mem_dim",
                "policy_use_gru",
                *TURNOVER_STATE_FIELDS,
                *PATCH_RESIDENT_FIELDS,
            )
        )
    )
    missing = sorted(name for name in batched_names if name not in source)
    if missing:
        raise _fail(f"resident source fields are missing: {missing}")
    if sources is not None:
        for cell, cell_source in enumerate(sources[1:], start=1):
            missing = sorted(name for name in batched_names if name not in cell_source)
            if missing:
                raise _fail(
                    f"per-cell resident source {cell} fields are missing: {missing}"
                )
    final_width_fields = {
        "mlp_pack",
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
    }
    gauss_tape_fields = {
        "python_random_gauss_primary_tape",
        "python_random_gauss_secondary_tape",
    }
    gauss_tape_width: int | None = None
    if sources is not None:
        gauss_tape_width = _validated_python_gauss_tape_width_v1(sources)
    compact_mlp_gru_pack = projection_family == "mlp"
    batched = {}
    for name in batched_names:
        if hapb_host_policy_enable and name == "mlp_pack":
            continue
        # Identity maps are private, variable-length rows at a continuation
        # boundary. They are padded once by the dedicated stacker below.
        if name == "agent_id_to_slot":
            continue
        if name == "gru_pack" and (shared_gru_pack is not None or compact_mlp_gru_pack):
            continue
        value = source[name]
        compact_mutation_tape = (
            name in {"policy_birth_delta_tape", "policy_birth_mutation_mask_tape"}
            and isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and int(value.size(-1)) == 1
        )
        if (
            observation_extra_columns
            and name in final_width_fields
            and not compact_mutation_tape
        ):
            if sources is None:
                batched[name] = _stack_fixed_mlp_final_layout_v1(
                    value,
                    device=device,
                    batch_cells=batch_cells,
                    base_observation_size=base_observation_size,
                    observation_size=observation_size,
                    hidden_size=int(config.hidden_size),
                    action_count=int(config.action_count),
                )
            else:
                _validate_per_cell_values_v1(
                    [item[name] for item in sources],
                    name=name,
                    source_groups=source_groups,
                )
                batched[name] = torch.cat(
                    [
                        _stack_fixed_mlp_final_layout_v1(
                            item[name],
                            device=device,
                            batch_cells=1,
                            base_observation_size=base_observation_size,
                            observation_size=observation_size,
                            hidden_size=int(config.hidden_size),
                            action_count=int(config.action_count),
                        )
                        for item in sources
                    ],
                    dim=0,
                ).contiguous()
        elif sources is not None and name in gauss_tape_fields:
            batched[name] = _stack_ragged_vectors_v1(
                [item[name] for item in sources],
                name=name,
                dtype=torch.float64,
                device=device,
                width=gauss_tape_width,
                source_groups=source_groups,
            )
        else:
            batched[name] = (
                _stack(value, device=device, batch_cells=batch_cells)
                if sources is None
                else _stack_per_cell_values_v1(
                    [item[name] for item in sources],
                    name=name,
                    device=device,
                    source_groups=source_groups,
                )
            )
    if hapb_host_policy_enable:
        # The normal direct ABI retains this argument for now, but its tiny
        # sentinel has no B48 policy rows.  The guarded HAPB callback is the
        # only admitted policy reader once direct dispatch is connected.
        batched["mlp_pack"] = torch.empty(
            (batch_cells, 1, 9991), dtype=torch.float32, device=device
        )
        from engine.perf.cuda_architecture_b_hapb_v1 import (
            initialize_selected_hapb_owner_v1,
        )

        source_rows = [item["mlp_pack"] for item in sources]
        canonical_host_bank = _shared_hapb_canonical_bank_from_rows_v1(source_rows)
        if canonical_host_bank is None:
            # Fresh per-cell sources legitimately arrive as independent CPU
            # tensors.  A restored HAPB checkpoint takes the no-copy branch
            # above, preserving its one canonical bank.
            canonical_host_bank = torch.stack(source_rows, dim=0).contiguous()
        restore_pending_cache_rehydrate = all(
            bool(
                getattr(item.cuda_boundary, "metadata", {}).get(
                    "hapb_policy_restore_pending_cache_rehydrate", False
                )
            )
            for item in cell_requests
        )
        from engine.hapb_rngv2.route import hapb_rngv2_env_enabled

        hapb_policy_owner = initialize_selected_hapb_owner_v1(
            canonical_host_bank,
            device=device,
            populate_hot_prefix=not restore_pending_cache_rehydrate,
            tile_rows=hapb_policy_tile_rows,
            hapb_rng_semantics_version=2 if hapb_rngv2_env_enabled() else 1,
        )
    else:
        hapb_policy_owner = None
    if shared_gru_pack is not None:
        batched["gru_pack"] = shared_gru_pack.to(device=device).contiguous()
    elif compact_mlp_gru_pack:
        policy_use_gru_sources = (
            [item["policy_use_gru"] for item in sources]
            if sources is not None
            else [source["policy_use_gru"]]
        )
        if any(bool(torch.any(item).item()) for item in policy_use_gru_sources):
            raise _fail("MLP projection cannot compact a live GRU policy pack")
        batched["gru_pack"] = torch.zeros(
            (batch_cells, 1), dtype=torch.float32, device=device
        )
    for name in ("pending_free_count", "slot_free_queue_count"):
        batched[name] = (
            _stack(
                source[name].reshape(()),
                device=device,
                batch_cells=batch_cells,
            )
            if sources is None
            else _stack_per_cell_values_v1(
                [item[name].reshape(()) for item in sources],
                name=name,
                device=device,
                source_groups=source_groups,
            )
        )
    _apply_resource_maintenance_cost_charge_floor_v1(batched, config=config)
    compact_id_values, compact_slot_values = _compact_identity_values_v1(
        source,
        sources,
    )
    if any(
        int(ids.numel()) != int(slots.numel())
        for ids, slots in zip(compact_id_values, compact_slot_values, strict=True)
    ):
        raise _fail("compact agent-ID/slot authorities are misaligned")
    if cell_requests is None:
        compact_count = int(compact_id_values[0].numel())
        if (
            compact_count > capacity
            or int(compact_slot_values[0].numel()) != compact_count
        ):
            raise _fail("broadcast compact identity pair count is invalid")
        compact_ids = torch.full(
            (batch_cells, capacity), -1, dtype=torch.int64, device=device
        )
        compact_slots = torch.full(
            (batch_cells, capacity), -1, dtype=torch.int32, device=device
        )
        if compact_count:
            compact_ids[:, :compact_count].copy_(
                compact_id_values[0]
                .to(device=device)
                .unsqueeze(0)
                .expand(batch_cells, -1)
            )
            compact_slots[:, :compact_count].copy_(
                compact_slot_values[0]
                .to(device=device)
                .unsqueeze(0)
                .expand(batch_cells, -1)
            )
        compact_counts = (compact_count,) * batch_cells
    else:
        compact_ids = _stack_ragged_vectors_v1(
            compact_id_values,
            name="agent_id_to_slot_agent_ids",
            dtype=torch.int64,
            device=device,
            width=capacity,
            source_groups=source_groups,
        )
        compact_slots = _stack_ragged_vectors_v1(
            compact_slot_values,
            name="agent_id_to_slot_slots",
            dtype=torch.int32,
            device=device,
            width=capacity,
            source_groups=source_groups,
        )
        compact_counts = tuple(int(value.numel()) for value in compact_id_values)
    if int(batched["population_next_id"].max().item()) != population_next_id:
        raise _fail("population identity watermark changed during stacking")
    # ABI compatibility only. Correctness is owned by compact pairs and slot
    # authority; high biological IDs are intentionally absent from this shadow.
    batched["agent_id_to_slot"] = torch.full(
        (batch_cells, capacity),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for cell, count in enumerate(compact_counts):
        if count > capacity:
            raise _fail(f"compact identity pair count for cell {cell} is invalid")
        ids = compact_ids[cell, :count]
        slots = compact_slots[cell, :count]
        low = ids.lt(capacity)
        if bool(low.any().item()):
            batched["agent_id_to_slot"][cell, ids[low]] = slots[low]
    for alias, canonical_name in {
        "attack_bias": "policy_attack_bias",
        "action_temperature": "policy_action_temperature",
        "drive_logit_gain": "policy_drive_logit_gain",
    }.items():
        batched[alias] = batched[canonical_name]
    # Generation-boundary preparation stages a window-local CPython Gaussian
    # tape from the checkpoint RNG state. Kernels index it through this base.
    batched["python_random_raw_draw_base"] = batched["python_random_raw_draws"][
        :, 0
    ].clone()
    batched["policy_birth_tape_cursor_base"] = batched["policy_birth_tape_cursor"][
        :, 0
    ].clone()

    active_ids = torch.full(
        (batch_cells, capacity), -1, dtype=torch.int64, device=device
    )
    if sources is None:
        active_ids[:, : initial_ids.numel()].copy_(
            initial_ids.to(device).unsqueeze(0).expand(batch_cells, -1)
        )
        active_count = torch.full(
            (batch_cells,), initial_ids.numel(), dtype=torch.int64, device=device
        )
    else:
        active_ids, active_count = _stack_ragged_active_agent_ids_v1(
            per_cell_initial_ids,
            capacity=capacity,
            device=device,
            source_groups=source_groups,
        )
    active_slots = _capacity_bounded_active_agent_slots_v1(
        active_ids,
        active_count,
        compact_ids,
        compact_slots,
        batched["slot_to_agent_id"],
        batched["alive_mask"],
    )
    batched["agent_grid"] = _canonicalize_agent_id_grid_v1(
        batched["agent_grid"],
        batched["slot_to_agent_id"],
        batched["alive_mask"],
        batched["pos_y"],
        batched["pos_x"],
    )
    outputs = _allocate_tick_outputs(
        capacity=capacity,
        observation_size=observation_size,
        action_count=int(config.action_count),
        device=device,
        output_ticks=output_ticks,
        batch_cells=batch_cells,
        physical_window_generations=physical_window_generations,
    )
    genome_v2_causal_capture_enabled = bool(
        getattr(config, "genome_v2_causal_capture_enable", False)
    )
    genome_v2_causal_record_capacity = GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1
    birth_outputs = _allocate_birth_outputs(
        capacity=capacity,
        max_births_per_tick=max_births_per_tick,
        device=device,
        output_ticks=output_ticks,
        batch_cells=batch_cells,
        genome_v2_causal_record_capacity=genome_v2_causal_record_capacity,
    )
    outputs.update(birth_outputs)
    outputs.update(
        _allocate_turnover_outputs(
            capacity=capacity,
            generation=source_generation,
            device=device,
            batch_cells=batch_cells,
            genome_v2_causal_record_capacity=genome_v2_causal_record_capacity,
        )
    )

    birth_state_tensors = [
        *(batched[name] for name in ORDINARY_BIRTH_IDENTITY_FIELDS),
        *(batched[name] for name in ORDINARY_BIRTH_SCALAR_RNG_FIELDS),
        *(batched[name] for name in ORDINARY_BIRTH_GENOME_INT32_FIELDS),
        batched["policy_mem_dim"],
        batched["policy_use_gru"],
        *(batched[name] for name in ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS),
        *(batched[name] for name in ORDINARY_BIRTH_DRIVE_FIELDS),
        *(batched[name] for name in ORDINARY_BIRTH_POLICY_FIELDS),
        *(batched[name] for name in TURNOVER_STATE_FIELDS),
        batched["policy_birth_tape_cursor_base"],
        batched["python_random_raw_draw_base"],
    ]
    diploid_bank = None
    genome_upgrade_v2_bank = None
    if (
        str(
            getattr(
                config,
                "inheritance_profile",
                "legacy_midpoint_v1",
            )
            or "legacy_midpoint_v1"
        )
        == DIPLOID_INHERITANCE_PROFILE
    ):
        digest_sources = (
            [item["canonical_hereditary_digest"] for item in sources]
            if sources is not None
            else None
        )
        if "canonical_hereditary_digest" not in source or (
            sources is not None
            and any("canonical_hereditary_digest" not in item for item in sources)
        ):
            raise _fail(
                "diploid resident source is missing canonical hereditary digest"
            )
        batched["canonical_hereditary_digest"] = (
            _stack(
                source["canonical_hereditary_digest"],
                device=device,
                batch_cells=batch_cells,
            )
            if digest_sources is None
            else _stack_per_cell_values_v1(
                digest_sources,
                name="canonical_hereditary_digest",
                device=device,
                source_groups=source_groups,
            )
        )
        for template_name in (
            "canonical_hereditary_template",
            "diploid_payload_template",
            "diploid_genotype_template",
        ):
            if template_name not in source or (
                sources is not None
                and any(template_name not in item for item in sources)
            ):
                raise _fail(f"diploid resident source is missing {template_name}")
            batched[template_name] = (
                _stack(
                    source[template_name],
                    device=device,
                    batch_cells=batch_cells,
                )
                if sources is None
                else _stack_per_cell_values_v1(
                    [item[template_name] for item in sources],
                    name=template_name,
                    device=device,
                    source_groups=source_groups,
                )
            )
        diploid_bank = (
            None
            if streamed_bwide_seed
            else _diploid_bank_from_canonical_projections_v1(
                canonical_by_cell,
                device=device,
                cell_ids=(
                    cell_ids
                    if cell_ids is not None
                    else tuple(
                        f"unbound-cell-{index}" for index in range(batch_cells)
                    )
                ),
            )
        )
        if diploid_bank is None:
            diploid_bank = _diploid_bank_from_fresh_sources_v1(
                source,
                sources=sources,
                device=device,
                batch_cells=batch_cells,
                source_groups=source_groups,
            )
        if diploid_bank is None:
            diploid_bank = migrate_homozygous_from_phenotype_v1(
                batched,
                cells=batch_cells,
                capacity=capacity,
                device=device,
            )
        if diploid_bank.cells != batch_cells or diploid_bank.capacity != capacity:
            raise _fail("diploid checkpoint dimensions disagree with resident source")
        cell_configurations = (
            tuple(item.configuration for item in cell_requests)
            if cell_requests is not None
            else (config,) * batch_cells
        )
        mutation_enabled_by_cell = tuple(
            not bool(getattr(item, "recomb72_mutation_off", False))
            and bool(getattr(item, "diploid_mutation_enable", True))
            for item in cell_configurations
        )
        birth_state_tensors.extend(
            (
                torch.full(
                    (batch_cells,),
                    2,
                    dtype=torch.int32,
                    device=device,
                ),
                torch.tensor(
                    mutation_enabled_by_cell,
                    dtype=torch.bool,
                    device=device,
                ),
                torch.full(
                    (batch_cells,),
                    bool(
                        getattr(
                            config,
                            "resource_handling_representation_enable",
                            False,
                        )
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    (batch_cells,),
                    dtype=torch.float64,
                    device=device,
                ),
                diploid_bank.storage,
                diploid_bank.genome_resource_handling_axis,
                torch.empty_like(diploid_bank.storage),
                torch.empty_like(diploid_bank.genome_resource_handling_axis),
                diploid_bank.canonical_hereditary_digest,
            )
        )
        if bool(getattr(config, "genome_upgrade_v2_enable", False)):
            mutation_rates_q10: list[int] = []
            mutation_steps_q10: list[int] = []
            founder_profiles_list: list[Mapping[str, Any]] = []
            for cell, cell_configuration in enumerate(cell_configurations):
                mutation_rate_q10 = getattr(
                    cell_configuration,
                    "genome_upgrade_v2_mutation_rate_q10",
                    None,
                )
                mutation_step_q10 = getattr(
                    cell_configuration,
                    "genome_upgrade_v2_mutation_step_q10",
                    None,
                )
                if (
                    type(mutation_rate_q10) is not int
                    or not 0 <= mutation_rate_q10 <= 1024
                ):
                    raise _fail(
                        "GENOME-UPGRADE-V2 mutation rate must be an exact "
                        f"integer in [0, 1024] for cell {cell}"
                    )
                if (
                    type(mutation_step_q10) is not int
                    or not 0 <= mutation_step_q10 <= 1024
                ):
                    raise _fail(
                        "GENOME-UPGRADE-V2 mutation step must be an exact "
                        f"integer in [0, 1024] for cell {cell}"
                    )
                mutation_rates_q10.append(mutation_rate_q10)
                mutation_steps_q10.append(mutation_step_q10)
                try:
                    profile = resolve_genome_v2_founder_profile_v1(
                        str(cell_configuration.genome_upgrade_v2_founder_profile_id)
                    )
                except Exception as exc:
                    raise _fail(
                        "GENOME-UPGRADE-V2 founder profile is invalid "
                        f"for cell {cell}: {exc}"
                    ) from exc
                if (
                    str(cell_configuration.genome_upgrade_v2_founder_profile_sha256)
                    != profile["payload_sha256"]
                ):
                    raise _fail(
                        "GENOME-UPGRADE-V2 founder profile payload hash "
                        f"mismatch for cell {cell}"
                    )
                founder_profiles_list.append(profile)
            founder_profiles = tuple(founder_profiles_list)
            genome_upgrade_v2_bank = (
                None
                if streamed_bwide_seed
                else _genome_v2_bank_from_canonical_projections_v1(
                    canonical_by_cell,
                    device=device,
                    cell_ids=(
                        cell_ids
                        if cell_ids is not None
                        else tuple(
                            f"unbound-cell-{index}" for index in range(batch_cells)
                        )
                    ),
                    founder_profiles=founder_profiles,
                )
            )
            if genome_upgrade_v2_bank is None:
                genome_upgrade_v2_bank = CudaGenomeUpgradeV2StateBank.allocate(
                    cells=batch_cells,
                    capacity=capacity,
                    device=device,
                )
                _initialize_genome_v2_fresh_founders_v1(
                    genome_upgrade_v2_bank,
                    founder_profiles,
                    alive_mask=batched["alive_mask"],
                )
            if (
                genome_upgrade_v2_bank.cells != batch_cells
                or genome_upgrade_v2_bank.capacity != capacity
            ):
                raise _fail(
                    "GENOME-UPGRADE-V2 checkpoint dimensions disagree "
                    "with resident source"
                )
            genome_v2_shadows = {
                name: torch.empty_like(tensor)
                for name, tensor in genome_upgrade_v2_bank.named_tensors().items()
            }
            birth_state_tensors.extend(
                (
                    torch.ones(
                        (batch_cells,),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.tensor(
                        tuple(mutation_rates_q10),
                        dtype=torch.int64,
                        device=device,
                    ),
                    torch.tensor(
                        tuple(mutation_steps_q10),
                        dtype=torch.int64,
                        device=device,
                    ),
                    *genome_upgrade_v2_bank.named_tensors().values(),
                    *genome_v2_shadows.values(),
                )
            )
    if genome_v2_causal_capture_enabled:
        if genome_upgrade_v2_bank is None:
            raise _fail(
                "Genome V2 causal capture requires an active Genome V2 resident bank"
            )
        outputs["birth_genome_v2_causal_capture_enabled"].fill_(True)
        outputs["turnover_genome_v2_causal_capture_enabled"].fill_(True)
    # The native birth ABI owns the expanded policy-tape tensors captured above.
    # Retain those CUDA tensors in the resident live source.  Canonical
    # checkpoint tapes remain in the imported canonical projections rather than
    # replacing a dispatchable resident tensor with a CPU copy.
    birth_output_tensors = (
        [birth_outputs[name] for name in ORDINARY_BIRTH_OUTPUT_FIELDS]
        if ordinary_births_enabled
        else []
    )
    try:
        birth_scratch_bytes = required_lifecycle_scratch_bytes_v1(
            max_births_per_tick,
            capacity,
            int(batched["mlp_pack"].size(-1)),
            int(batched["hmem"].size(-1)),
            int(batched["rng_manager_mutation_torch_cpu_state"].size(-1)),
        )
    except DirectResidentPreparationError as exc:
        raise _fail(str(exc)) from exc
    birth_scratch_layout = make_legacy_tail_birth_scratch_layout_v2(
        max_births_per_tick,
        capacity,
        int(batched["mlp_pack"].size(-1)),
        int(batched["hmem"].size(-1)),
        int(batched["rng_manager_mutation_torch_cpu_state"].size(-1)),
    )
    if (
        not birth_scratch_layout.ok()
        or birth_scratch_layout.bytes_per_cell != birth_scratch_bytes
    ):
        raise _fail("runtime birth scratch layout changed during allocation")
    digest_template_pack = (
        pack_canonical_digest_templates_v1(diploid_bank)
        if diploid_bank is not None
        else None
    )
    template_pack_bytes = (
        int(digest_template_pack.size(1)) if digest_template_pack is not None else 0
    )
    birth_scratch = [
        torch.empty(
            (batch_cells, birth_scratch_bytes + template_pack_bytes),
            dtype=torch.uint8,
            device=device,
        )
    ]
    _disable_genome_v2_qualification_fault_v1(birth_scratch[0], birth_scratch_layout)
    if digest_template_pack is not None:
        birth_scratch[0][:, birth_scratch_bytes:].copy_(digest_template_pack)

    if cell_requests is None:
        event_columns = [
            _stack(column, device=device, batch_cells=batch_cells)
            for column in request.generation_birth_state.event_tape_columns
        ]
    else:
        event_columns = [
            _stack_per_cell_values_v1(
                [
                    item.generation_birth_state.event_tape_columns[index]
                    for item in cell_requests
                ],
                name=f"event_tape_columns[{index}]",
                device=device,
                source_groups=source_groups,
            )
            for index in range(len(request.generation_birth_state.event_tape_columns))
        ]
    if len(event_columns) != len(EVENT_COLUMN_NAMES):
        raise _fail("resident event-column count changed")

    def stack_birth_scalar(name: str) -> torch.Tensor:
        value = getattr(request.generation_birth_state, name).reshape(())
        if cell_requests is None:
            return _stack(value, device=device, batch_cells=batch_cells)
        return _stack_per_cell_values_v1(
            [
                getattr(item.generation_birth_state, name).reshape(())
                for item in cell_requests
            ],
            name=name,
            device=device,
            source_groups=source_groups,
        )

    event_active_count = stack_birth_scalar("event_tape_active_count")
    event_status = stack_birth_scalar("event_tape_status")
    boundary_metadata = getattr(request.cuda_boundary, "metadata", {})
    boundary_sequence_base = getattr(request.cuda_boundary, "tensors", {}).get(
        "event_sequence_base"
    )
    if not isinstance(boundary_sequence_base, torch.Tensor):
        raise _fail("CUDA boundary omitted the resident event sequence frontier")
    if cell_requests is None:
        if boundary_sequence_base.numel() != batch_cells:
            raise _fail(
                "CUDA boundary event sequence frontier must contain one value "
                f"per cell: expected {batch_cells}, got {boundary_sequence_base.numel()}"
            )
        event_sequence_base = boundary_sequence_base.reshape(batch_cells).to(
            device=device
        ).contiguous()
    else:
        boundary_sequence_bases: list[torch.Tensor] = []
        for cell, item in enumerate(cell_requests):
            value = getattr(item.cuda_boundary, "tensors", {}).get(
                "event_sequence_base"
            )
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise _fail(
                    "per-cell CUDA boundary event sequence frontier must be a "
                    f"scalar tensor at cell {cell}"
                )
            boundary_sequence_bases.append(value.reshape(()))
        event_sequence_base = _stack_per_cell_values_v1(
            boundary_sequence_bases,
            name="event_sequence_base",
            device=device,
            source_groups=source_groups,
        )
    event_history_drained = (
        isinstance(boundary_metadata, Mapping)
        and boundary_metadata.get("cuda_boundary_event_history_drained") is True
    )
    reporting_initial_ids = (
        [initial_ids] * batch_cells if sources is None else per_cell_initial_ids
    )
    initial_agent_ids_by_row, terminal_survival, terminal_rows = (
        _allocate_initial_id_reporting_state_v1(
            reporting_initial_ids,
            reporting_capacity=capacity,
            device=device,
            reporting_rows_override=(capacity if streamed_bwide_seed else None),
        )
    )
    event_state_tensors = [
        *event_columns,
        event_active_count,
        event_status,
        batched["lineage_founder_id"],
        batched["lineage_depth"],
        initial_agent_ids_by_row,
        terminal_survival,
        terminal_rows,
        outputs["event_status_by_tick"],
        event_sequence_base,
    ]

    residual_ids = torch.full(
        (batch_cells, capacity), -1, dtype=torch.int64, device=device
    )
    residual_slots = torch.full_like(residual_ids, -1)
    residual_count = torch.zeros((batch_cells,), dtype=torch.int64, device=device)
    policy_draws = stack_birth_scalar("policy_np_semantic_draws")
    metrics_draws = stack_birth_scalar("metrics_np_semantic_draws")
    full_tick_state = [
        batched["age"],
        batched["gru_upkeep_cost"],
        batched["cost_charge_scalar"],
        batched["cost_charge_maturation_scalar"],
        batched["drive_inertia"],
        batched["drive_noise"],
        batched["prev_drive_delta"],
        batched["drive_volatility"],
        batched["drive_jitter"],
        batched["episode0_drive_jitter_sum"],
        batched["last_energy_delta"],
        batched["last_health_delta"],
        batched["last_injury_delta"],
        batched["selfharm_events"],
        batched["episode0_selfharm_events"],
        batched["chemistry"],
        batched["runner_pcg64_state_words"],
        batched["runner_pcg64_buffer_words"],
        batched["pcg64_state_words"],
        batched["pcg64_buffer_words"],
        batched["drive_noise_samples_by_tick_slot"],
        batched["drive_noise_samples_valid_by_tick_slot"],
        batched["slot_free_queue"],
        batched["slot_free_queue_count"],
        residual_ids,
        residual_slots,
        residual_count,
        policy_draws,
        metrics_draws,
    ]
    full_tick_outputs = [
        outputs["chemistry_order_slots"],
        outputs["runner_shuffle_draws"],
        outputs["runner_shuffle_status"],
        outputs["drive_noise_draws"],
        outputs["drive_noise_raw64_draws"],
        outputs["drive_noise_status"],
        outputs["chemistry_draws"],
        outputs["chemistry_status"],
        outputs["active_mechanics_status"],
        outputs["active_mechanics_counts"],
        outputs["residual_death_agent_ids"],
        outputs["residual_death_slots"],
        outputs["residual_death_reason_codes"],
        outputs["residual_death_count"],
        outputs["residual_release_count"],
        outputs["residual_release_status"],
        outputs["policy_pcg64_state_after_action"],
        outputs["policy_pcg64_buffer_after_action"],
        outputs["food_spawn_stats"],
        outputs["generation_summary_scalar"],
    ]

    persistent_state: dict[str, Any] = {
        **batched,
        # Bounded diagnostics are named for consumers but are not resident
        # simulation state and must not enter checkpoint/state inventories.
        **{
            name: outputs[name]
            for name in (
                "birth_genome_v2_causal_record_rows",
                "birth_genome_v2_causal_record_valid",
                "birth_genome_v2_causal_record_count",
                "birth_genome_v2_causal_record_dropped_count",
                "birth_genome_v2_causal_capture_enabled",
                "turnover_genome_v2_causal_record_rows",
                "turnover_genome_v2_causal_record_valid",
                "turnover_genome_v2_causal_record_count",
                "turnover_genome_v2_causal_record_dropped_count",
                "turnover_genome_v2_causal_capture_enabled",
            )
        },
        "active_agent_ids": active_ids,
        "active_agent_slots": active_slots,
        "active_count_state": active_count,
        "birth_state_tensors": birth_state_tensors,
        "ordinary_birth_output_tensors": birth_output_tensors,
        "ordinary_birth_scratch_tensors": birth_scratch,
        "turnover_selection_seed_by_cell": turnover_selection_seed_by_cell,
        "policy_np_semantic_draws": policy_draws,
        "metrics_np_semantic_draws": metrics_draws,
        "event_columns": event_columns,
        "event_active_count": event_active_count,
        "event_status": event_status,
        "event_sequence_base": event_sequence_base,
        "event_history_drained": event_history_drained,
        "direct_batch_host_import_enable": direct_batch_host_import_enable,
        "event_history_storage_mode": boundary_metadata.get(
            "event_history_storage_mode"
        ),
        "event_history_reference": boundary_metadata.get("event_history_reference"),
        "event_history_frontier": (
            boundary_metadata.get("event_history_frontier")
            if boundary_metadata.get("event_history_storage_mode")
            == "eiel_resident_event_sink_v1"
            else None
        ),
        "terminal_survival": terminal_survival,
        "terminal_rows": terminal_rows,
    }
    if hapb_policy_owner is not None:
        persistent_state["hapb_policy_owner"] = hapb_policy_owner
        persistent_state["hapb_full_cuda_mlp_pack_absent"] = True
    turnover_state: list[torch.Tensor] = []
    turnover_shadows: list[torch.Tensor] = []
    if config.generation_turnover_enabled:
        turnover_state = [
            (
                active_ids
                if name == "active_agent_ids"
                else (
                    active_count
                    if name == "active_count_state"
                    else persistent_state[name]
                )
            )
            for name in TURNOVER_MUTABLE_STATE_FIELDS
        ]
        shadows_by_address: dict[int, torch.Tensor] = {}
        for name, tensor in zip(TURNOVER_MUTABLE_STATE_FIELDS, turnover_state):
            shadow = shadows_by_address.get(int(tensor.data_ptr()))
            if shadow is None:
                if name == "mlp_pack":
                    row_count = (
                        0
                        if turnover_mlp_shadow_elide_enable or hapb_host_policy_enable
                        else int(config.population_size)
                    )
                    shadow = torch.empty(
                        (batch_cells, row_count, int(tensor.size(2))),
                        dtype=tensor.dtype,
                        device=device,
                    )
                else:
                    shadow = torch.empty_like(tensor)
                shadows_by_address[int(tensor.data_ptr())] = shadow
            turnover_shadows.append(shadow)
    turnover_outputs = [outputs[name] for name in TURNOVER_OUTPUT_FIELDS]
    turnover_pointer_table = (
        [
            torch.empty(
                (len(TURNOVER_MUTABLE_STATE_FIELDS) * 4,),
                dtype=torch.int64,
                device=device,
            )
        ]
        if config.generation_turnover_enabled
        else []
    )
    persistent_state.update(
        {
            "turnover_state_tensors": turnover_state,
            "turnover_shadow_tensors": turnover_shadows,
            "turnover_output_tensors": turnover_outputs,
            "turnover_pointer_table_tensors": turnover_pointer_table,
        }
    )
    if diploid_bank is not None:
        persistent_state.update(diploid_bank.named_tensors())
        persistent_state["diploid_allele_storage"] = diploid_bank.storage
        persistent_state["genome_resource_handling_axis"] = (
            diploid_bank.genome_resource_handling_axis
        )
    if genome_upgrade_v2_bank is not None:
        persistent_state.update(genome_upgrade_v2_bank.named_tensors())
        founder_profile_identities = tuple(
            {
                "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
                "profile_id": profile["profile_id"],
                "payload_sha256": profile["payload_sha256"],
            }
            for profile in founder_profiles
        )
        persistent_state["genome_v2_founder_profiles"] = founder_profile_identities
        if batch_cells == 1:
            persistent_state["genome_v2_founder_profile"] = founder_profile_identities[
                0
            ]
        checkpoint_authority = _genome_v2_checkpoint_authority_binding_v1(config)
        persistent_state["genome_v2_checkpoint_authority_binding"] = (
            checkpoint_authority
        )
        _enable_authority_bound_genome_v2_causal_capture_v1(
            persistent_state,
            checkpoint_authority=checkpoint_authority,
        )
    patch_state = [batched[name] for name in PATCH_P2_STATE_FIELDS_V1]
    height, width = int(patch_state[0].size(1)), int(patch_state[0].size(2))
    patch_scratch = [
        torch.empty(
            (batch_cells, 4, height, width),
            dtype=torch.float32,
            device=device,
        )
    ]
    patch_stats = [
        torch.empty(
            (
                batch_cells,
                physical_window_ticks,
                PATCH_STAT_COUNT_V1,
            ),
            dtype=torch.float64,
            device=device,
        )
    ]
    patch_config = {
        name: value_type(getattr(config, name))
        for name, value_type in PATCH_P2_CONFIG_ARGUMENTS_V1
    }
    raw_rng_semantics_versions = [
        getattr(cell_configuration, "rng_semantics_version", 1)
        for cell_configuration in lifecycle_configurations
    ]
    raw_world_uids = [
        getattr(cell_configuration, "world_uid", 0)
        for cell_configuration in lifecycle_configurations
    ]
    raw_namespace_keys = [
        getattr(
            cell_configuration,
            "rng_namespace_key64",
            TW_RNGV2_NAMESPACE_DEFAULT,
        )
        for cell_configuration in lifecycle_configurations
    ]
    raw_semantic_epochs = [
        getattr(cell_configuration, "rng_semantic_epoch", 1)
        for cell_configuration in lifecycle_configurations
    ]
    for values, name in (
        (raw_rng_semantics_versions, "semantic version"),
        (raw_world_uids, "world UID"),
        (raw_namespace_keys, "namespace key"),
        (raw_semantic_epochs, "semantic epoch"),
    ):
        if any(type(value) is not int for value in values):
            raise _fail(f"TW RNG {name} must be an exact integer")
    rng_semantics_versions = set(raw_rng_semantics_versions)
    world_uids = set(raw_world_uids)
    namespace_keys = set(raw_namespace_keys)
    semantic_epochs = set(raw_semantic_epochs)
    if len(rng_semantics_versions) != 1:
        raise _fail("physical batch mixes TW RNG semantic versions")
    if len(world_uids) != 1:
        raise _fail("physical batch mixes TW RNG world UIDs")
    if len(namespace_keys) != 1:
        raise _fail("physical batch mixes TW RNG namespace keys")
    if len(semantic_epochs) != 1:
        raise _fail("physical batch mixes TW RNG semantic epochs")
    rng_semantics_version = next(iter(rng_semantics_versions))
    world_uid = next(iter(world_uids))
    namespace_key64 = next(iter(namespace_keys))
    semantic_epoch = next(iter(semantic_epochs))
    if rng_semantics_version not in {1, 2}:
        raise _fail("TW RNG semantic version must be 1 or 2")
    if not 0 <= world_uid <= 0xFFFFFFFF:
        raise _fail("TW RNG world UID must fit uint32")
    if namespace_key64 != TW_RNGV2_NAMESPACE_DEFAULT:
        raise _fail("TW RNG namespace key must equal the canonical V2 key")
    if semantic_epoch != 1:
        raise _fail("TW RNG semantic epoch must equal the V1 contract epoch")
    if rng_semantics_version == 2 and batch_cells > 1 and cell_ids is None:
        raise _fail(
            "multi-cell TW RNG V2 requires stable logical cell identities"
        )
    logical_world_uids = (
        tuple(
            _logical_world_uid_v1(
                configured_world_uid=world_uid,
                cell_id=cell_id,
            )
            for cell_id in cell_ids
        )
        if rng_semantics_version == 2 and cell_ids is not None
        else tuple(world_uid for _ in range(batch_cells))
    )
    if rng_semantics_version == 2 and len(set(logical_world_uids)) != batch_cells:
        raise _fail("TW RNG V2 logical world UID derivation collided")
    logical_world_uid_by_cell = torch.tensor(
        logical_world_uids,
        dtype=torch.int64,
        device=active_ids.device,
    )
    patch_config["tw_rngv2_semantics_version"] = rng_semantics_version
    patch_config["tw_rngv2_world_uid"] = world_uid
    patch_config["tw_rngv2_world_uid_by_cell"] = logical_world_uid_by_cell
    patch_config["tw_rngv2_namespace_key64"] = namespace_key64
    patch_config["tw_rngv2_semantic_epoch"] = semantic_epoch
    event_history_sink_path = getattr(request, "event_history_sink_path", None)
    if event_history_sink_path is not None:
        if (
            type(event_history_sink_path) is not str
            or not event_history_sink_path.strip()
        ):
            raise _fail("request event_history_sink_path must be non-empty text")
        patch_config["event_history_sink_path"] = event_history_sink_path
    # Patch P2 normally retains scalar configuration.  This axis is explicitly
    # admitted to vary by physical cell, so preserve its actual dispatch input
    # rather than silently reusing the primary cell's scalar in a B2+ batch.
    for field in (
        "patch_food_gather_amount",
        "patch_food_regrow_per_step",
        "patch_movement_cost_energy_scale",
        "reproduce_min_energy_buffer",
        "reproduce_parent_energy_share",
        "reproduce_try_prob",
    ):
        patch_config[f"{field}_by_cell"] = torch.tensor(
            [
                float(getattr(cell_configuration, field))
                for cell_configuration in lifecycle_configurations
            ],
            dtype=torch.float64,
            device=active_ids.device,
        )
    patch_config["reproduce_cooldown_steps_by_cell"] = torch.tensor(
        [
            int(getattr(cell_configuration, "reproduce_cooldown_steps"))
            for cell_configuration in lifecycle_configurations
        ],
        dtype=torch.int32,
        device=active_ids.device,
    )
    if genome_upgrade_v2_bank is not None:
        persistent_state["genome_v2_observability_config"] = (
            _genome_v2_observability_config_v1(config, patch_config)
        )
    checkpoint_canonical_by_cell = canonical_by_cell
    if hapb_host_policy_enable:
        # These fields authenticate transport of the one HAPB policy bank at
        # restore time.  They are not generation authority and must not leak
        # into the next terminal checkpoint, whose digest must match an
        # uninterrupted resident trajectory.
        def without_hapb_checkpoint_transport(
            projection: Mapping[str, Any],
        ) -> dict[str, Any]:
            normalized = dict(projection)
            metadata = dict(normalized.get("metadata") or {})
            for name in (
                "hapb_policy_checkpoint_sha256",
                "hapb_policy_checkpoint_schema_version",
                "hapb_policy_restore_pending_cache_rehydrate",
            ):
                metadata.pop(name, None)
            normalized["metadata"] = metadata
            return normalized

        checkpoint_canonical_by_cell = tuple(
            without_hapb_checkpoint_transport(projection)
            for projection in canonical_by_cell
        )
        canonical = checkpoint_canonical_by_cell[0]
    descriptor_specs = torch.empty(
        (1, 13), dtype=torch.int64, device="cpu", pin_memory=True
    )[:0]
    return DirectResidentSourceV1(
        configuration=config,
        move_cost_by_cell=torch.tensor(
            move_cost_by_cell,
            dtype=torch.float64,
            device=active_ids.device,
        ),
        active_agent_ids=active_ids,
        active_agent_slots=active_slots,
        active_count_state=active_count,
        live_state=batched,
        outputs=outputs,
        full_tick_state_tensors=full_tick_state,
        full_tick_output_tensors=full_tick_outputs,
        compact_summary_reporting_tensors=(),
        compact_summary_checkpoint_source_tensors=(),
        compact_summary_checkpoint_descriptor_specs=descriptor_specs,
        compact_summary_checkpoint_output_tensors=(),
        compact_summary_settlement_tensors=(),
        event_state_tensors=event_state_tensors,
        ordinary_birth_state_tensors=birth_state_tensors,
        ordinary_birth_output_tensors=birth_output_tensors,
        ordinary_birth_scratch_tensors=birth_scratch,
        turnover_selection_seed_by_cell=turnover_selection_seed_by_cell,
        turnover_state_tensors=turnover_state,
        turnover_shadow_tensors=turnover_shadows,
        turnover_output_tensors=turnover_outputs,
        turnover_pointer_table_tensors=turnover_pointer_table,
        patch_p2_state_tensors=patch_state,
        patch_p2_scratch_tensors=patch_scratch,
        patch_p2_stats_tensors=patch_stats,
        patch_p2_config=patch_config,
        live_banks=persistent_state,
        checkpoint_projection_inputs={
            "cuda_capability": dict(request.cuda_capability),
            "tw_rngv2_semantics_version": rng_semantics_version,
            "tw_rngv2_configured_world_uid": world_uid,
            "tw_rngv2_namespace_key64": namespace_key64,
            "tw_rngv2_semantic_epoch": semantic_epoch,
            "tw_rngv2_logical_world_uids": logical_world_uids,
            "tw_rngv2_logical_world_uid_derivation": (
                LOGICAL_WORLD_UID_DERIVATION_V1
            ),
            "pre_generation_canonical_projection": canonical,
            "pre_generation_canonical_projections": checkpoint_canonical_by_cell,
            "resident_birth_state": _checkpoint_schema_template_v1(
                request.generation_birth_state
            ),
            "diploid_checkpoint_projection": (
                None
                if streamed_bwide_seed
                else diploid_bank.checkpoint_projection()
                if diploid_bank is not None
                else None
            ),
            "genome_upgrade_v2_checkpoint_projection": (
                None
                if streamed_bwide_seed
                else {
                    **genome_upgrade_v2_bank.checkpoint_projection(),
                    **(
                        {"founder_profile": founder_profile_identities[0]}
                        if batch_cells == 1
                        else {
                            "founder_profiles": [
                                dict(identity)
                                for identity in founder_profile_identities
                            ]
                        }
                    ),
                }
                if genome_upgrade_v2_bank is not None
                else None
            ),
            "genome_v2_founder_profile": (
                founder_profile_identities[0]
                if genome_upgrade_v2_bank is not None and batch_cells == 1
                else None
            ),
            "genome_v2_founder_profiles": (
                founder_profile_identities if genome_upgrade_v2_bank is not None else ()
            ),
            "genome_v2_checkpoint_authority_binding": (
                checkpoint_authority if genome_upgrade_v2_bank is not None else ()
            ),
            "inheritance_sidecar_cell_id": (
                inheritance_sidecar_cell_id if batch_cells == 1 else None
            ),
            "genome_v2_causal_physical_cell_ordinal": (
                genome_v2_causal_physical_cell_ordinal
            ),
            "tick_native_residency_contract": (
                {
                    "time_authority": "absolute_tick_v1",
                    "total_ticks": native_control.get("total_ticks"),
                    "resident_owner_span_ticks": native_control.get(
                        "resident_owner_span_ticks"
                    ),
                    "native_dispatch_window_ticks": native_control.get(
                        "native_dispatch_window_ticks"
                    ),
                    "resident_event_capacity": native_event_capacity,
                    "resident_event_capacity_horizon_invariant": True,
                    "resident_event_capacity_source": (
                        "source_derived_current_window_v1"
                    ),
                    "resident_event_row_bytes": EVENT_ROW_STORAGE_BYTES_V1,
                    "resident_event_bytes_per_cell": (
                        native_event_capacity * EVENT_ROW_STORAGE_BYTES_V1
                    ),
                    "resident_event_bytes": (
                        native_event_capacity
                        * EVENT_ROW_STORAGE_BYTES_V1
                        * batch_cells
                    ),
                    "historical_event_restack": False,
                }
                if native_tick_route
                else None
            ),
        },
        ticks_per_generation=config.steps,
        deterministic=False,
        generation_base=source_generation,
        logical_generation_count=logical_generation_count,
        physical_window_generations=physical_window_generations,
        batch_cells=batch_cells,
        allow_short_technical_gate_generation_count=(
            getattr(
                request,
                "allow_short_technical_gate_generation_count",
                False,
            )
            is True
        ),
        allow_partial_generation_source_cursor=(
            getattr(
                request,
                "allow_partial_generation_source_cursor",
                False,
            )
            is True
        ),
        time_authority=(
            "absolute_tick_v1" if native_tick_route else "legacy_generation_v1"
        ),
        resident_owner_span_ticks=(
            native_control.get("resident_owner_span_ticks")
            if native_tick_route
            else None
        ),
        native_dispatch_window_ticks=(
            native_control.get("native_dispatch_window_ticks")
            if native_tick_route
            else None
        ),
        source_window_ticks=(
            native_control.get("source_window_ticks") if native_tick_route else None
        ),
        resident_event_capacity=native_event_capacity if native_tick_route else None,
    )


if len(TURNOVER_MUTABLE_STATE_FIELDS) != 122:
    raise RuntimeError("turnover mutable tensor ABI must contain exactly 122 fields")
if len(ORDINARY_BIRTH_OUTPUT_FIELDS) != ORDINARY_BIRTH_OUTPUT_COUNT_V1:
    raise RuntimeError(
        "ordinary birth output ABI must contain exactly "
        f"{ORDINARY_BIRTH_OUTPUT_COUNT_V1} fields"
    )


__all__ = [
    "CudaArchitectureBDirectResidentSourceAllocatorError",
    "LEGACY_DENSE_IDENTITY_ALLOCATION_BUDGET_BYTES_V1",
    "LEGACY_DENSE_IDENTITY_BYTES_PER_CELL_ID_V1",
    "ORDINARY_BIRTH_OUTPUT_FIELDS",
    "PHYSICAL_WINDOW_TICKS_V1",
    "PHYSICAL_WINDOW_TURNOVER_EPISODES_V1",
    "RESIDENT_AGENT_ID_CAPACITY_METADATA_V1",
    "TURNOVER_MUTABLE_STATE_FIELDS",
    "allocate_direct_resident_source_v1",
    "legacy_dense_identity_allocation_bytes_v1",
    "required_agent_id_capacity_v1",
    "required_resident_agent_id_capacity_v1",
    "required_physical_window_policy_episodes_v1",
    "required_physical_window_turnover_episodes_v1",
    "stack_validated_per_cell_checkpoint_projections_v1",
]
