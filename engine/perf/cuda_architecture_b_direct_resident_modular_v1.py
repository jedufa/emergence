"""Execute the bounded modular-recurrence production CUDA slice."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import torch

from agents.modular_genome_v1 import canonical_hash_from_genome_dict
from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    create_run_control_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    _project_terminal_checkpoint,
    _with_materialized_event_history,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    allocate_direct_resident_source_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    QUALIFIED_CAPABILITY_MASK_V1,
    build_direct_resident_source_allocation_request_v1,
)
from engine.perf.cuda_architecture_b_disabled_prediction_state_v1 import (
    materialize_disabled_prediction_state_v1,
)
from engine.perf.cxxbackbone10_resident_state_owner import (
    build_import_projection,
    projection_digest,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_RUNTIME_MUTABLE_FIELD_NAMES,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    PackedModularStorageV1,
    modular_storage_from_projection,
    modular_storage_projection_payload,
    pack_modular_storage,
    validate_modular_storage,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    resident_window_count_v1,
)
from perf.cuda_architecture_b_modular_state_bank import (
    MODULAR_STATE_BANK_FIELD_COUNT,
    build_modular_state_bank_manifest,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptV1,
    build_modular_recurrent_mutation_receipt_v1,
    mutation_ancestry_sha256_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_replay_v1 import (
    ModularRecurrentMutationReplayError,
    replay_modular_recurrent_mutation_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    ModularRecurrentMutationSequenceV1,
    build_modular_recurrent_mutation_sequence_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_replay_v1 import (
    ModularRecurrentMutationSequenceReplayError,
    replay_modular_recurrent_mutation_sequence_v1,
)


MODULAR_ABI_VERSION_V1: Final[int] = 1
MODULAR_LIFECYCLE_ABI_VERSION_V2: Final[int] = 2
MODULAR_CAPABILITY_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_flattened_modular_capability_v1"
)
MODULAR_DIRECT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_modular_v1"
)
MODULAR_PHYSICAL_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_modular_physical_v1"
)
MODULAR_LIFECYCLE_CAPABILITY_ENTRYPOINT_V2: Final[str] = (
    "eiel_cuda_architecture_b_flattened_modular_lifecycle_capability_v2"
)
MODULAR_LIFECYCLE_DIRECT_ENTRYPOINT_V2: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_v2"
)
MODULAR_LIFECYCLE_PHYSICAL_ENTRYPOINT_V2: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_physical_v2"
)
MODULAR_TURNOVER_MUTATION_RECEIPT_WIDTH_V1: Final[int] = 21
MODULAR_TURNOVER_MUTATION_RAW_LAYOUT_VERSION_V1: Final[int] = 2
MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1: Final[int] = 3
MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_ABI_VERSION_V3: Final[int] = 3
MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_WIDTH_V3: Final[int] = 64
MODULAR_ORDINARY_BIRTH_ROWV1_DISABLED_REASON_MIN_V3: Final[int] = 0
MODULAR_ORDINARY_BIRTH_ROWV1_DISABLED_REASON_MAX_V3: Final[int] = 6
PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_runtime_bookkeeping")
)
PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_full_tick_output_tensors")
)
PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_patch_stats_tensors")
)
TICKS_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "ticks_per_generation"
)
CACHE_WIDTH_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "config.cache_width"
)
DETERMINISTIC_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "deterministic"
)
MAX_BIRTHS_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "config.max_births_per_tick"
)
TURNOVER_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "config.generation_turnover_enabled"
)
CULL_FRACTION_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "config.cull_fraction"
)
ORDINARY_BIRTH_OUTPUTS_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("ordinary_birth_output_tensors")
)
TURNOVER_TENSOR_ARGUMENT_INDICES_V1: Final[tuple[int, ...]] = tuple(
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(name)
    for name in (
        "turnover_state_tensors",
        "turnover_shadow_tensors",
        "turnover_output_tensors",
        "turnover_pointer_table_tensors",
    )
)
TURNOVER_MUTABLE_ARGUMENT_INDICES_V1: Final[tuple[int, ...]] = tuple(
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(name)
    for name in (
        "turnover_state_tensors",
        "turnover_shadow_tensors",
        "turnover_pointer_table_tensors",
    )
)
TURNOVER_OUTPUT_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "turnover_output_tensors"
)


class CudaArchitectureBDirectResidentModularV1Error(RuntimeError):
    """The bounded modular resident trajectory is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentModularV1:
    patch: PreparedDirectResidentTrajectoryV1
    modular_tensors: tuple[torch.Tensor, ...]
    profile_id: str
    modular_shadows: tuple[torch.Tensor, ...] = ()
    lifecycle_pointer_table: torch.Tensor | None = None
    cold_origin_slots: torch.Tensor | None = None
    cold_origin_slots_shadow: torch.Tensor | None = None
    modular_authorities: tuple[PackedModularStorageV1, ...] = ()
    action_residual_trace: torch.Tensor | None = None
    turnover_mutation_draws: torch.Tensor | None = None
    duplicate_descendant_mask: torch.Tensor | None = None
    duplicate_descendant_mask_shadow: torch.Tensor | None = None
    turnover_mutation_receipt_rows: torch.Tensor | None = None
    turnover_mutation_receipt_counts: torch.Tensor | None = None
    turnover_mutation_receipt_overflow: torch.Tensor | None = None
    ordinary_birth_mutation_draws: torch.Tensor | None = None
    ordinary_birth_mutation_receipt_rows: torch.Tensor | None = None
    ordinary_birth_mutation_receipt_counts: torch.Tensor | None = None
    ordinary_birth_mutation_receipt_overflow: torch.Tensor | None = None
    ordinary_birth_rowv1_receipt_rows: torch.Tensor | None = None
    ordinary_birth_rowv1_receipt_counts: torch.Tensor | None = None
    ordinary_birth_rowv1_receipt_overflow: torch.Tensor | None = None

    @property
    def batch_cells(self) -> int:
        return int(self.patch.source.batch_cells)


def _fail(message: str) -> CudaArchitectureBDirectResidentModularV1Error:
    return CudaArchitectureBDirectResidentModularV1Error(message)


def _validated_capability(module: Any) -> dict[str, Any]:
    entrypoint = getattr(module, MODULAR_CAPABILITY_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {MODULAR_CAPABILITY_ENTRYPOINT_V1}"
        )
    raw = entrypoint()
    if not isinstance(raw, Mapping):
        raise _fail("modular CUDA capability must be a mapping")
    capability = dict(raw)
    expected = {
        "abi_version": MODULAR_ABI_VERSION_V1,
        "resident_state": True,
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
        "action_width": 7,
        "deterministic_fixed_mlp": True,
        "cache_width": 1,
        "births": False,
        "turnover": False,
        "native_tick_ownership": True,
        "native_generation_ownership": True,
        "runtime_jit_required": False,
        "hidden_fallback": False,
        "production_qualified": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"modular CUDA capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return capability


def _validated_lifecycle_capability(module: Any) -> dict[str, Any]:
    entrypoint = getattr(module, MODULAR_LIFECYCLE_CAPABILITY_ENTRYPOINT_V2, None)
    if not callable(entrypoint):
        raise _fail(
            "CUDA extension is missing callable "
            f"{MODULAR_LIFECYCLE_CAPABILITY_ENTRYPOINT_V2}"
        )
    raw = entrypoint()
    if not isinstance(raw, Mapping):
        raise _fail("modular lifecycle CUDA capability must be a mapping")
    capability = dict(raw)
    expected = {
        "abi_version": MODULAR_LIFECYCLE_ABI_VERSION_V2,
        "resident_state": True,
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
        "action_width": 7,
        "deterministic_fixed_mlp": True,
        "cache_width": 1,
        "births": True,
        "turnover": True,
        "no_mutation_inheritance": True,
        "heterogeneous_parent_structure": True,
        "native_tick_ownership": True,
        "native_generation_ownership": True,
        "runtime_jit_required": False,
        "hidden_fallback": False,
        "production_qualified": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"modular lifecycle CUDA capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return capability


def _require_action_trace_capability(
    capability: Mapping[str, Any],
    prepared: PreparedDirectResidentModularV1,
) -> None:
    if (
        prepared.action_residual_trace is not None
        and capability.get("optional_action_residual_trace") is not True
    ):
        raise _fail("CUDA extension does not support modular action residual tracing")


def _require_turnover_mutation_capability(
    capability: Mapping[str, Any],
    prepared: PreparedDirectResidentModularV1,
) -> None:
    if prepared.turnover_mutation_draws is None:
        return
    if capability.get("turnover_refine_descendant_parameter_qualification") is not True:
        raise _fail("CUDA extension does not support bounded modular turnover mutation")
    expected = {
        "turnover_mutation_receipt_raw_layout_version": (
            MODULAR_TURNOVER_MUTATION_RAW_LAYOUT_VERSION_V1
        ),
        "turnover_mutation_receipt_sequence_capacity": (
            MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1
        ),
        "mutation_receipt_sequence_v1": True,
        "turnover_mutation_qualified_generation_horizon": 3,
        "turnover_mutation_overflow_test_generation": 4,
        "mutated_checkpoint_fresh_owner": True,
        "ordinary_birth_mutation": True,
        "turnover_mutation_g8_g32": False,
        "turnover_mutation_default_production": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"bounded modular turnover mutation capability {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )


def _require_ordinary_birth_mutation_capability(
    capability: Mapping[str, Any],
    prepared: PreparedDirectResidentModularV1,
) -> None:
    if prepared.ordinary_birth_mutation_draws is None:
        return
    expected = {
        "ordinary_birth_mutation": True,
        "ordinary_birth_gated_parameter_refinement": True,
        "ordinary_birth_mutation_receipt_raw_layout_version": (
            MODULAR_TURNOVER_MUTATION_RAW_LAYOUT_VERSION_V1
        ),
        "ordinary_birth_mutation_receipt_capacity": (
            MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1
        ),
        "ordinary_birth_rowv1_receipt_abi_version": (
            MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_ABI_VERSION_V3
        ),
        "ordinary_birth_rowv1_receipt_width": (
            MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_WIDTH_V3
        ),
        "ordinary_birth_rowv1_receipt_capacity": (
            MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1
        ),
        "ordinary_birth_rowv1_receipt_default_production": False,
        "ordinary_birth_rowv1_receipt_kernel_mirror_present": True,
        "ordinary_birth_rowv1_receipt_kernel_write_qualified": True,
        "ordinary_birth_rowv1_receipt_disabled_event_writer": True,
        "ordinary_birth_rowv1_receipt_device_writer_sanitized": True,
        "ordinary_birth_mutation_default_production": False,
    }
    for name, expected_value in expected.items():
        actual = capability.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"bounded modular ordinary-birth mutation capability {name} "
                f"mismatch: expected={expected_value!r} actual={actual!r}"
            )


def _require_no_native_failure(result: Mapping[str, Any]) -> None:
    failures = result.get("first_failure_by_cell", [])
    no_failure_sentinel = result.get("first_failure_no_failure_sentinel", -1)
    if type(no_failure_sentinel) is not int:
        raise _fail("modular CUDA result has malformed no-failure sentinel")
    if not isinstance(failures, list):
        raise _fail("modular CUDA result has malformed native failure records")
    for failure in failures:
        if not isinstance(failure, Mapping):
            raise _fail("modular CUDA result has malformed native failure records")
        status = failure.get("status")
        if type(status) is not int:
            raise _fail("modular CUDA result has malformed native failure status")
        if status != 0 and status != no_failure_sentinel:
            raise _fail(
                "modular CUDA native execution failed: "
                f"cell={failure.get('cell')!r} generation={failure.get('generation')!r} "
                f"tick={failure.get('tick')!r} status={status}"
            )


def prepare_direct_resident_modular_v1(
    patch: PreparedDirectResidentTrajectoryV1,
    cell_tensors: Sequence[Mapping[str, torch.Tensor]],
    *,
    profile_id: str = "b2-12x32-v1",
) -> PreparedDirectResidentModularV1:
    """Stack exact single-cell modular planes into one physical CUDA batch."""

    if not isinstance(patch, PreparedDirectResidentTrajectoryV1):
        raise _fail("patch must be PreparedDirectResidentTrajectoryV1")
    cells = int(patch.source.batch_cells)
    capacity = int(patch.source.live_state["alive_mask"].shape[1])
    if len(cell_tensors) != cells:
        raise _fail("one modular tensor mapping is required per physical cell")
    manifest = build_modular_state_bank_manifest(capacity, profile_id)
    if len(manifest.fields) != MODULAR_STATE_BANK_FIELD_COUNT:
        raise _fail("modular state-bank field count changed")
    device = patch.source.live_state["alive_mask"].device
    ordered: list[torch.Tensor] = []
    for field in manifest.fields:
        rows: list[torch.Tensor] = []
        for cell, mapping in enumerate(cell_tensors):
            value = mapping.get(field.projected_name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != field.torch_dtype
                or tuple(value.shape) != field.shape
                or value.requires_grad
            ):
                raise _fail(
                    f"invalid modular tensor {field.projected_name!r} for cell {cell}"
                )
            rows.append(value.to(device=device).contiguous())
        ordered.append(torch.stack(rows, dim=0).contiguous())
    return PreparedDirectResidentModularV1(
        patch=patch,
        modular_tensors=tuple(ordered),
        profile_id=profile_id,
    )


def prepare_direct_resident_modular_lifecycle_v2(
    patch: PreparedDirectResidentTrajectoryV1,
    cell_tensors: Sequence[Mapping[str, torch.Tensor]],
    *,
    profile_id: str = "b2-12x32-v1",
) -> PreparedDirectResidentModularV1:
    """Prepare the fixed 57-plane source/shadow lifecycle transaction."""

    prepared = prepare_direct_resident_modular_v1(
        patch, cell_tensors, profile_id=profile_id
    )
    shadows = tuple(torch.empty_like(tensor) for tensor in prepared.modular_tensors)
    pointer_table = torch.empty(
        (MODULAR_STATE_BANK_FIELD_COUNT, 5),
        dtype=torch.int64,
        device=prepared.modular_tensors[0].device,
    )
    occupied = prepared.modular_tensors[0].bool()
    capacity = int(occupied.size(1))
    cold_origin_slots = (
        torch.arange(
            capacity,
            dtype=torch.int64,
            device=occupied.device,
        )
        .expand(prepared.batch_cells, capacity)
        .clone()
    )
    cold_origin_slots.masked_fill_(~occupied, -1)
    cold_origin_slots_shadow = torch.empty_like(cold_origin_slots)
    return PreparedDirectResidentModularV1(
        patch=prepared.patch,
        modular_tensors=prepared.modular_tensors,
        profile_id=prepared.profile_id,
        modular_shadows=shadows,
        lifecycle_pointer_table=pointer_table,
        cold_origin_slots=cold_origin_slots,
        cold_origin_slots_shadow=cold_origin_slots_shadow,
    )


def build_prepared_direct_resident_modular_lifecycle_v2(
    runner: Any,
    authority: PackedModularStorageV1,
    *,
    qualified_cuda_module: Any,
    logical_generation_count: int = 2,
    initial_generation: int = 0,
    source_checkpoint_projection: Mapping[str, Any] | None = None,
    cuda_device: str = "cuda",
    output_ticks: int = 80,
) -> PreparedDirectResidentModularV1:
    """Build one B1 lifecycle owner from a canonical modular authority."""

    if type(output_ticks) is not int or output_ticks < 1 or output_ticks > 80:
        raise _fail("modular lifecycle output_ticks must be in [1,80]")
    validate_modular_storage(authority)
    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 1
        or logical_generation_count > 4
    ):
        raise _fail("modular lifecycle build requires generations in [1,4]")
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("modular lifecycle initial_generation must be nonnegative")
    if source_checkpoint_projection is None:
        raw_projection: Any = build_import_projection(
            runner,
            dict(QUALIFIED_CAPABILITY_MASK_V1),
            generation=initial_generation,
            tick=0,
        )
        projection = dict(vars(raw_projection))
        metadata = projection.get("metadata")
        if not isinstance(metadata, Mapping):
            raise _fail("fresh modular lifecycle projection omitted metadata")
        event_cursor = metadata.get(
            "typed_lifecycle_event_next_sequence",
            metadata.get(
                "event_tape_cursor",
                metadata.get("typed_lifecycle_event_active_count", 0),
            ),
        )
        if type(event_cursor) is not int or event_cursor < 0:
            raise _fail("fresh modular lifecycle projection has an invalid event cursor")
        projection["event_cursor"] = event_cursor
        projection["runtime_event_cursor"] = event_cursor
    else:
        projection = copy.deepcopy(dict(source_checkpoint_projection))
        metadata = projection.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("generation") != initial_generation
            or metadata.get("tick") != 0
        ):
            raise _fail("modular lifecycle checkpoint is not the requested boundary")

    metadata = projection.get("metadata")
    tensors = projection.get("tensors")
    if not isinstance(metadata, dict) or not isinstance(tensors, dict):
        raise _fail("modular lifecycle source projection is not mutable canonical data")
    capacity = authority.layout.agent_capacity
    alive_mask = tensors.get("alive_mask")
    slot_to_agent_id = tensors.get("slot_to_agent_id")
    slot_tokens = tensors.get("slot_tokens")
    if (
        not isinstance(alive_mask, torch.Tensor)
        or not isinstance(slot_to_agent_id, torch.Tensor)
        or not isinstance(slot_tokens, torch.Tensor)
        or tuple(alive_mask.shape) != (capacity,)
        or not torch.equal(authority.tensors["agent_occupied"], alive_mask)
        or not torch.equal(authority.tensors["agent_id"], slot_to_agent_id)
        or not torch.equal(
            authority.tensors["agent_slot_token"],
            torch.where(alive_mask, slot_tokens, torch.full_like(slot_tokens, -1)),
        )
    ):
        raise _fail("modular authority ownership contradicts the source projection")
    payload = modular_storage_projection_payload(authority)
    metadata.update(payload["metadata"])
    tensors.update(payload["tensors"])
    metadata["import_digest"] = projection_digest(
        {"metadata": metadata, "tensors": tensors}
    )

    request = build_direct_resident_source_allocation_request_v1(
        runner,
        qualified_cuda_module=qualified_cuda_module,
        cuda_device=cuda_device,
        logical_generation_count=max(3, logical_generation_count),
        source_checkpoint_projection=projection,
    )
    request = replace(request, output_ticks=output_ticks)
    source = allocate_direct_resident_source_v1(request)
    patch = prepare_direct_resident_trajectory_v1(source)
    prepared = prepare_direct_resident_modular_lifecycle_v2(
        patch,
        [
            {
                f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}": (
                    authority.tensors[field.name]
                )
                for field in authority.layout.fields
            }
        ],
        profile_id=authority.layout.profile_id,
    )
    return PreparedDirectResidentModularV1(
        patch=prepared.patch,
        modular_tensors=prepared.modular_tensors,
        profile_id=prepared.profile_id,
        modular_shadows=prepared.modular_shadows,
        lifecycle_pointer_table=prepared.lifecycle_pointer_table,
        cold_origin_slots=prepared.cold_origin_slots,
        cold_origin_slots_shadow=prepared.cold_origin_slots_shadow,
        modular_authorities=(authority,),
    )


def project_direct_resident_modular_lifecycle_checkpoint_v2(
    prepared: PreparedDirectResidentModularV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    selected_parent_slot_by_child_slot: Mapping[int, int],
) -> Mapping[str, Any]:
    """Project one B1 lifecycle boundary through the canonical checkpoint path."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if len(prepared.modular_authorities) != 1:
        raise _fail("B1 modular lifecycle checkpoint requires one cold authority")
    if (
        type(completed_generation_count) is not int
        or completed_generation_count < 1
        or completed_generation_count > 4
    ):
        raise _fail("modular lifecycle checkpoint requires generations in [1,4]")
    if not isinstance(direct_result, Mapping):
        raise _fail("direct_result must be a mapping")
    if not isinstance(selected_parent_slot_by_child_slot, Mapping):
        raise _fail("selected parent slots must be a mapping")

    result = dict(direct_result)
    outputs = dict(prepared.patch.source.outputs)
    generation_summary = result.get("generation_summary_scalar")
    if not isinstance(generation_summary, torch.Tensor):
        generation_summary = prepared.patch.arguments[
            PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1
        ][-1][:completed_generation_count]
    runtime_bookkeeping = result.get("runtime_bookkeeping_by_tick")
    if not isinstance(runtime_bookkeeping, torch.Tensor):
        runtime_bookkeeping = prepared.patch.arguments[
            PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1
        ][:, : completed_generation_count * 80]
    outputs["generation_summary_scalar"] = generation_summary
    outputs["runtime_bookkeeping_by_tick"] = runtime_bookkeeping
    persistent_state = dict(
        _with_materialized_event_history(prepared.patch.live_banks, result)
    )
    initial_projection = prepared.patch.checkpoint_projection_inputs.get(
        "pre_generation_canonical_projection"
    )
    if not isinstance(initial_projection, Mapping):
        raise _fail("prepared owner omitted its canonical source projection")
    persistent_state = materialize_disabled_prediction_state_v1(
        initial_projection,
        persistent_state,
        reset_count_max=80,
    )
    for field, tensor in zip(
        prepared.modular_authorities[0].layout.fields,
        prepared.modular_tensors,
        strict=True,
    ):
        persistent_state[f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}"] = tensor

    checkpoint_inputs = prepared.patch.checkpoint_projection_inputs
    mutation_receipt: (
        ModularRecurrentMutationReceiptV1 | ModularRecurrentMutationSequenceV1 | None
    )
    if prepared.ordinary_birth_mutation_draws is not None:
        mutation_receipt = project_modular_ordinary_birth_mutation_receipt_v1(
            prepared
        )
    elif (
        prepared.turnover_mutation_draws is not None
        and isinstance(prepared.turnover_mutation_receipt_counts, torch.Tensor)
        and int(prepared.turnover_mutation_receipt_counts[0].item()) != 1
    ):
        mutation_receipt = project_modular_turnover_mutation_sequence_v1(prepared)
    else:
        mutation_receipt = project_modular_turnover_mutation_receipt_v1(prepared)
    checkpoint = _project_terminal_checkpoint(
        initial_projection,
        checkpoint_inputs.get("resident_birth_state"),
        outputs,
        persistent_state,
        completed_generation_count=completed_generation_count,
        source_generation=prepared.patch.source.generation_base,
        phase_e_active_modular_authority=prepared.modular_authorities[0],
        phase_e_active_modular_lifecycle_parent_slots=(
            selected_parent_slot_by_child_slot
        ),
        phase_e_active_modular_lifecycle_mutation_receipt=mutation_receipt,
    )
    projection = checkpoint.projection
    packed = modular_storage_from_projection(
        projection["metadata"], projection["tensors"]
    )
    if packed is None:
        raise _fail("lifecycle checkpoint omitted modular storage")
    return projection


def project_modular_turnover_mutation_receipt_v1(
    prepared: PreparedDirectResidentModularV1,
) -> ModularRecurrentMutationReceiptV1 | None:
    """Decode the bounded G1 device receipt into its canonical CPU contract."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if prepared.turnover_mutation_draws is None:
        return None
    rows = _materialize_modular_turnover_mutation_rows_v1(prepared)
    if len(rows) != 1:
        raise _fail("bounded modular turnover mutation requires exactly one receipt")
    return _canonical_modular_turnover_mutation_receipt_from_row_v1(
        prepared.modular_authorities[0],
        rows[0],
    )


def project_modular_ordinary_birth_mutation_receipt_v1(
    prepared: PreparedDirectResidentModularV1,
) -> ModularRecurrentMutationReceiptV1 | None:
    """Decode one real CUDA RowV1 mirrored ordinary-birth mutation receipt."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if prepared.ordinary_birth_mutation_draws is None:
        return None
    rows = _materialize_modular_ordinary_birth_mutation_rows_v1(prepared)
    rowv1_rows = _materialize_modular_ordinary_birth_rowv1_rows_v1(prepared)
    if len(rows) != len(rowv1_rows):
        raise _fail("ordinary-birth RowV1 mirror count diverged from raw receipt")
    for raw_row, rowv1 in zip(rows, rowv1_rows, strict=True):
        if rowv1[2:22] != raw_row[1:21]:
            raise _fail("ordinary-birth RowV1 mirror diverged from raw receipt prefix")
    if len(rows) != 1:
        raise _fail(
            "bounded modular ordinary-birth mutation requires exactly one receipt"
        )
    return _canonical_modular_turnover_mutation_receipt_from_row_v1(
        prepared.modular_authorities[0],
        rows[0],
        transaction_kind="ordinary_birth",
    )


def _materialize_modular_turnover_mutation_rows_v1(
    prepared: PreparedDirectResidentModularV1,
    *,
    allow_overflow: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Validate and copy the fixed-capacity device receipt prefix."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if prepared.turnover_mutation_draws is None:
        return ()
    if len(prepared.modular_authorities) != 1:
        raise _fail("modular turnover mutation receipt requires one cold authority")
    rows = prepared.turnover_mutation_receipt_rows
    counts = prepared.turnover_mutation_receipt_counts
    overflow = prepared.turnover_mutation_receipt_overflow
    if (
        not isinstance(rows, torch.Tensor)
        or not isinstance(counts, torch.Tensor)
        or not isinstance(overflow, torch.Tensor)
    ):
        raise _fail("modular turnover mutation receipt tensors are incomplete")
    if tuple(overflow.shape) != (1,):
        raise _fail("modular turnover mutation overflow tensor shape changed")
    overflow_value = int(overflow[0].item())
    if overflow_value not in {0, 1} or (overflow_value != 0 and not allow_overflow):
        raise _fail("modular turnover mutation receipt sequence overflowed")
    if tuple(counts.shape) != (1,):
        raise _fail("modular turnover mutation count tensor shape changed")
    count = int(counts[0].item())
    if not 0 <= count <= MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1:
        raise _fail("modular turnover mutation receipt count is outside capacity")
    if tuple(rows.shape) != (
        1,
        MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
        MODULAR_TURNOVER_MUTATION_RECEIPT_WIDTH_V1,
    ):
        raise _fail("modular turnover mutation receipt tensor shape changed")
    materialized = rows[0].detach().cpu().contiguous()
    if bool(torch.any(materialized[count:] != -1).item()):
        raise _fail("inactive modular turnover mutation receipt rows are not empty")
    expected_first_generation = prepared.patch.source.generation_base + 1
    active: list[tuple[int, ...]] = []
    for sequence_index in range(count):
        row = tuple(int(value) for value in materialized[sequence_index].tolist())
        if (
            row[0] != 1
            or row[1] != sequence_index
            or row[2] != expected_first_generation + sequence_index
            or row[3] != 0
            or row[10] != 0
            or row[20] != 1
        ):
            raise _fail(
                "modular turnover mutation receipt order or commit coordinates changed"
            )
        if any(value < 0 for value in (*row[4:10],)):
            raise _fail(
                "modular turnover mutation receipt owner coordinates are invalid"
            )
        active.append(row)
    return tuple(active)


def _materialize_modular_ordinary_birth_mutation_rows_v1(
    prepared: PreparedDirectResidentModularV1,
) -> tuple[tuple[int, ...], ...]:
    if prepared.ordinary_birth_mutation_draws is None:
        return ()
    rows = prepared.ordinary_birth_mutation_receipt_rows
    counts = prepared.ordinary_birth_mutation_receipt_counts
    overflow = prepared.ordinary_birth_mutation_receipt_overflow
    if (
        not isinstance(rows, torch.Tensor)
        or not isinstance(counts, torch.Tensor)
        or not isinstance(overflow, torch.Tensor)
    ):
        raise _fail("modular ordinary-birth mutation receipt tensors are incomplete")
    if tuple(overflow.shape) != (prepared.batch_cells,):
        raise _fail("modular ordinary-birth mutation overflow tensor shape changed")
    if bool(overflow.detach().cpu().ne(0).any().item()):
        raise _fail("modular ordinary-birth mutation receipt sequence overflowed")
    if tuple(counts.shape) != (prepared.batch_cells,):
        raise _fail("modular ordinary-birth mutation count tensor shape changed")
    count_values = counts.detach().cpu().tolist()
    if any(
        type(value) is not int
        or value < 0
        or value > MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1
        for value in count_values
    ):
        raise _fail("modular ordinary-birth mutation receipt count is outside capacity")
    cpu_rows = rows.detach().cpu()
    if tuple(cpu_rows.shape) != (
        prepared.batch_cells,
        MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
        MODULAR_TURNOVER_MUTATION_RECEIPT_WIDTH_V1,
    ):
        raise _fail("modular ordinary-birth mutation receipt tensor shape changed")
    active: list[tuple[int, ...]] = []
    for cell, count in enumerate(count_values):
        if bool(cpu_rows[cell, count:].ne(-1).any().item()):
            raise _fail("inactive ordinary-birth mutation receipt rows are not empty")
        for sequence_index in range(count):
            row = tuple(int(value) for value in cpu_rows[cell, sequence_index].tolist())
            if row[0] != 1 or row[1] != sequence_index:
                raise _fail(
                    "ordinary-birth mutation receipt order or commit coordinates changed"
                )
            if row[4] < 0 or row[5] < 0 or row[7] < 0 or row[8] < 0:
                raise _fail("ordinary-birth mutation receipt owner coordinates are invalid")
            active.append(row)
    return tuple(active)


def _materialize_modular_ordinary_birth_rowv1_rows_v1(
    prepared: PreparedDirectResidentModularV1,
) -> tuple[tuple[int, ...], ...]:
    if prepared.ordinary_birth_mutation_draws is None:
        return ()
    rows = prepared.ordinary_birth_rowv1_receipt_rows
    counts = prepared.ordinary_birth_rowv1_receipt_counts
    overflow = prepared.ordinary_birth_rowv1_receipt_overflow
    if (
        not isinstance(rows, torch.Tensor)
        or not isinstance(counts, torch.Tensor)
        or not isinstance(overflow, torch.Tensor)
    ):
        raise _fail("modular ordinary-birth RowV1 receipt tensors are incomplete")
    if rows.dtype != torch.int64 or counts.dtype != torch.int64:
        raise _fail("modular ordinary-birth RowV1 receipt tensors must be int64")
    if overflow.dtype != torch.int32:
        raise _fail("modular ordinary-birth RowV1 overflow tensor must be int32")
    if tuple(overflow.shape) != (prepared.batch_cells,):
        raise _fail("modular ordinary-birth RowV1 overflow tensor shape changed")
    if bool(overflow.detach().cpu().ne(0).any().item()):
        raise _fail("modular ordinary-birth RowV1 receipt sequence overflowed")
    if tuple(counts.shape) != (prepared.batch_cells,):
        raise _fail("modular ordinary-birth RowV1 count tensor shape changed")
    count_values = counts.detach().cpu().tolist()
    if any(
        type(value) is not int
        or value < 0
        or value > MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1
        for value in count_values
    ):
        raise _fail("modular ordinary-birth RowV1 receipt count is outside capacity")
    cpu_rows = rows.detach().cpu().contiguous()
    if tuple(cpu_rows.shape) != (
        prepared.batch_cells,
        MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
        MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_WIDTH_V3,
    ):
        raise _fail("modular ordinary-birth RowV1 receipt tensor shape changed")
    active: list[tuple[int, ...]] = []
    for cell, count in enumerate(count_values):
        if bool(cpu_rows[cell, count:].ne(-1).any().item()):
            raise _fail("inactive ordinary-birth RowV1 receipt rows are not empty")
        for sequence_index in range(count):
            row = tuple(int(value) for value in cpu_rows[cell, sequence_index].tolist())
            if (
                row[0] != MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_ABI_VERSION_V3
                or row[1] != MODULAR_TURNOVER_MUTATION_RAW_LAYOUT_VERSION_V1
                or row[2] != sequence_index
            ):
                raise _fail(
                    "ordinary-birth RowV1 receipt order or ABI coordinates changed"
                )
            if not (
                MODULAR_ORDINARY_BIRTH_ROWV1_DISABLED_REASON_MIN_V3
                <= row[22]
                <= MODULAR_ORDINARY_BIRTH_ROWV1_DISABLED_REASON_MAX_V3
            ):
                raise _fail("ordinary-birth RowV1 disabled-reason code is invalid")
            if any(value != -1 for value in row[23:]):
                raise _fail("ordinary-birth RowV1 receipt tail is not empty")
            active.append(row)
    return tuple(active)


def _canonical_modular_turnover_mutation_receipt_from_row_v1(
    authority: PackedModularStorageV1,
    row: Sequence[int],
    *,
    transaction_kind: str = "turnover",
) -> ModularRecurrentMutationReceiptV1:
    """Convert one validated raw row to the accepted canonical G1 receipt."""

    if len(row) != MODULAR_TURNOVER_MUTATION_RECEIPT_WIDTH_V1:
        raise _fail("modular turnover mutation receipt row width changed")

    operation = {2: "refine_descendant_parameter"}.get(row[13])
    status = {1: "accepted", 3: "ineligible", 4: "noop"}.get(row[14])
    reason = {
        1: "proposal_valid",
        2: "no_duplicate_descendant_parameter",
        4: "refined_value_unchanged",
    }.get(row[15])
    dimension = {
        0: "none",
        1: "expression_scale",
        2: "live_retention",
        3: "live_input_gain",
        4: "live_output_gain",
        5: "reset_bias",
        6: "reset_input_gain",
        7: "reset_state_gain",
        8: "update_bias",
        9: "update_input_gain",
        10: "update_state_gain",
        11: "candidate_bias",
        12: "candidate_input_gain",
        13: "candidate_state_gain",
        14: "output_gain",
    }.get(row[19])
    if None in (operation, status, reason, dimension):
        raise _fail("CUDA returned unsupported modular mutation receipt enums")

    ancestry_sha256 = mutation_ancestry_sha256_v1(
        transaction_kind=transaction_kind,
        parent_slot=row[5],
        child_slot=row[8],
        parent_agent_id=row[4],
        child_agent_id=row[7],
        child_slot_token=row[9],
    )
    return build_modular_recurrent_mutation_receipt_v1(
        profile_id=authority.layout.profile_id,
        source_packed_sha256=authority.packed_sha256,
        ancestry_tape_sha256=ancestry_sha256,
        transactions=[
            {
                "transaction_kind": transaction_kind,
                "transaction_ordinal": 0,
                "parent_agent_id": row[4],
                "parent_slot": row[5],
                "parent_slot_token": row[6],
                "child_agent_id": row[7],
                "child_slot": row[8],
                "child_slot_token": row[9],
                "attempt_ordinal": row[10],
                "draw0_float64_bits": f"{row[11] & ((1 << 64) - 1):016x}",
                "draw1_float64_bits": f"{row[12] & ((1 << 64) - 1):016x}",
                "operation": operation,
                "status": status,
                "reason": reason,
                "selected_instance_id_index": row[17],
                "selected_homology_id_index": row[18],
                "changed_dimension": dimension,
                "changed": status == "accepted",
            }
        ],
    )


def project_modular_turnover_mutation_sequence_v1(
    prepared: PreparedDirectResidentModularV1,
) -> ModularRecurrentMutationSequenceV1 | None:
    """Replay and canonicalize the complete device mutation receipt prefix."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if prepared.turnover_mutation_draws is None:
        return None
    rows = _materialize_modular_turnover_mutation_rows_v1(prepared)
    authority = prepared.modular_authorities[0]
    records = copy.deepcopy(list(authority.records_sidecar))
    known_agent_ids = {
        int(record["agent_id"]) for record in records if record is not None
    }
    entries: list[dict[str, object]] = []
    for sequence_index, row in enumerate(rows):
        parent_slot = row[5]
        child_slot = row[8]
        parent = records[parent_slot]
        if (
            parent is None
            or int(parent["agent_id"]) != row[4]
            or int(parent["agent_slot_token"]) != row[6]
        ):
            raise _fail("modular turnover mutation sequence parent ownership mismatch")
        if row[7] in known_agent_ids:
            raise _fail("modular turnover mutation sequence reused a child ID")
        receipt = _canonical_modular_turnover_mutation_receipt_from_row_v1(
            authority,
            row,
        )
        transaction = receipt.transactions[0]
        pre_genome_sha256 = canonical_hash_from_genome_dict(
            copy.deepcopy(dict(parent["genome"]))
        )
        try:
            child_genome = replay_modular_recurrent_mutation_v1(
                parent["genome"],
                receipt,
                identity_table=authority.identity_table,
            )
        except ModularRecurrentMutationReplayError as exc:
            raise _fail(
                f"modular turnover mutation sequence CPU replay failed: {exc}"
            ) from exc
        post_genome_sha256 = canonical_hash_from_genome_dict(child_genome)
        child = copy.deepcopy(parent)
        child["agent_id"] = row[7]
        child["agent_slot_token"] = row[9]
        child["genome"] = child_genome
        records[child_slot] = child
        known_agent_ids.add(row[7])
        entries.append(
            {
                "sequence_index": sequence_index,
                "generation": row[2],
                "tick": row[3],
                "parent_agent_id": transaction.parent_agent_id,
                "parent_slot": transaction.parent_slot,
                "parent_slot_token": transaction.parent_slot_token,
                "child_agent_id": transaction.child_agent_id,
                "child_slot": transaction.child_slot,
                "child_slot_token": transaction.child_slot_token,
                "draw0_float64_bits": transaction.draw0_float64_bits,
                "draw1_float64_bits": transaction.draw1_float64_bits,
                "operation": transaction.operation,
                "status": transaction.status,
                "reason": transaction.reason,
                "selected_instance_id_index": (transaction.selected_instance_id_index),
                "selected_homology_id_index": (transaction.selected_homology_id_index),
                "changed_dimension": transaction.changed_dimension,
                "changed": transaction.changed,
                "pre_genome_sha256": pre_genome_sha256,
                "post_genome_sha256": post_genome_sha256,
                "committed": True,
            }
        )
    return build_modular_recurrent_mutation_sequence_v1(
        profile_id=authority.layout.profile_id,
        source_packed_sha256=authority.packed_sha256,
        entries=entries,
    )


def derive_modular_lifecycle_parent_slots_v2(
    prepared: PreparedDirectResidentModularV1,
) -> dict[int, int]:
    """Collapse ordinary-birth and turnover ancestry to pre-generation slots."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if len(prepared.modular_authorities) != 1 or prepared.batch_cells != 1:
        raise _fail("parent-slot derivation requires exactly one lifecycle cell")
    authority = prepared.modular_authorities[0]
    outputs = prepared.patch.source.outputs
    live = prepared.patch.live_banks

    def cpu(name: str, source: Mapping[str, Any]) -> torch.Tensor:
        value = source.get(name)
        if not isinstance(value, torch.Tensor):
            raise _fail(f"lifecycle ancestry omitted tensor {name}")
        return value.detach().to(device="cpu").contiguous()

    parent_ids = cpu("birth_committed_parent_agent_ids", outputs)
    child_ids = cpu("birth_committed_child_agent_ids", outputs)
    child_slots = cpu("birth_committed_child_slots", outputs)
    birth_counts = cpu("birth_identity_committed_count", outputs)
    if (
        parent_ids.dtype != torch.int64
        or child_ids.dtype != torch.int64
        or child_slots.dtype != torch.int64
        or birth_counts.dtype != torch.int64
        or parent_ids.ndim != 3
        or tuple(parent_ids.shape) != tuple(child_ids.shape)
        or tuple(parent_ids.shape) != tuple(child_slots.shape)
        or int(parent_ids.size(0)) != 1
        or int(parent_ids.size(1)) != 80
        or tuple(birth_counts.shape) != (1, 80)
    ):
        raise _fail("ordinary-birth ancestry must retain exact B1T80 output rows")

    origin_by_id: dict[int, int] = {}
    known_ids: set[int] = set()
    latest_child_by_slot: dict[int, int] = {}
    for slot, record in enumerate(authority.records_sidecar):
        if record is None:
            continue
        agent_id = int(record["agent_id"])
        if agent_id in known_ids:
            raise _fail("pre-generation modular authority contains duplicate agent IDs")
        known_ids.add(agent_id)
        origin_by_id[agent_id] = slot

    births_per_tick = int(parent_ids.size(2))
    capacity = authority.layout.agent_capacity
    for tick in range(80):
        count = int(birth_counts[0, tick].item())
        if count < 0 or count > births_per_tick:
            raise _fail("ordinary-birth committed count is out of range")
        staged: list[tuple[int, int, int, int]] = []
        tick_start_ids = set(known_ids)
        for row in range(count):
            parent_id = int(parent_ids[0, tick, row].item())
            child_id = int(child_ids[0, tick, row].item())
            child_slot = int(child_slots[0, tick, row].item())
            if (
                parent_id not in tick_start_ids
                or child_id < 0
                or child_id in known_ids
                or not 0 <= child_slot < capacity
            ):
                raise _fail("ordinary-birth ancestry row is inconsistent")
            staged.append((child_id, child_slot, origin_by_id[parent_id], parent_id))
        if len({row[0] for row in staged}) != len(staged) or len(
            {row[1] for row in staged}
        ) != len(staged):
            raise _fail(
                "ordinary-birth transaction contains duplicate children or slots"
            )
        if (
            bool(parent_ids[0, tick, count:].ne(-1).any().item())
            or bool(child_ids[0, tick, count:].ne(-1).any().item())
            or bool(child_slots[0, tick, count:].ne(-1).any().item())
        ):
            raise _fail("ordinary-birth ancestry tail is not canonical")
        for child_id, child_slot, origin_slot, _parent_id in staged:
            known_ids.add(child_id)
            origin_by_id[child_id] = origin_slot
            latest_child_by_slot[child_slot] = child_id

    turnover_counts = cpu("turnover_counts", outputs)
    turnover_parents = cpu("turnover_refill_parent_agent_ids", outputs)
    turnover_children = cpu("turnover_child_agent_ids", outputs)
    turnover_slots = cpu("turnover_child_slots", outputs)
    turnover_pool = cpu("turnover_parent_pool_agent_ids", outputs)
    if (
        turnover_counts.dtype != torch.int64
        or tuple(turnover_counts.shape) != (1, 8)
        or any(
            tensor.dtype != torch.int64 or tuple(tensor.shape) != (1, capacity)
            for tensor in (
                turnover_parents,
                turnover_children,
                turnover_slots,
                turnover_pool,
            )
        )
    ):
        raise _fail("turnover ancestry tensors have invalid shape or dtype")
    parent_pool_count = int(turnover_counts[0, 2].item())
    committed = int(turnover_counts[0, 6].item())
    if not 0 <= parent_pool_count <= capacity or not 0 <= committed <= capacity:
        raise _fail("turnover ancestry count is out of range")
    parent_pool = {
        int(value) for value in turnover_pool[0, :parent_pool_count].tolist()
    }
    staged_turnover: list[tuple[int, int, int]] = []
    turnover_start_ids = set(known_ids)
    for row in range(committed):
        parent_id = int(turnover_parents[0, row].item())
        child_id = int(turnover_children[0, row].item())
        child_slot = int(turnover_slots[0, row].item())
        if (
            parent_id not in turnover_start_ids
            or parent_id not in parent_pool
            or child_id < 0
            or child_id in known_ids
            or not 0 <= child_slot < capacity
        ):
            raise _fail("turnover ancestry row is inconsistent")
        staged_turnover.append((child_id, child_slot, origin_by_id[parent_id]))
    if len({row[0] for row in staged_turnover}) != len(staged_turnover) or len(
        {row[1] for row in staged_turnover}
    ) != len(staged_turnover):
        raise _fail("turnover transaction contains duplicate children or slots")
    for tensor in (turnover_parents, turnover_children, turnover_slots):
        if bool(tensor[0, committed:].ne(-1).any().item()):
            raise _fail("turnover ancestry tail is not canonical")
    for child_id, child_slot, origin_slot in staged_turnover:
        known_ids.add(child_id)
        origin_by_id[child_id] = origin_slot
        latest_child_by_slot[child_slot] = child_id

    alive_mask = cpu("alive_mask", live)[0].bool()
    terminal_ids = cpu("slot_to_agent_id", live)[0]
    terminal_tokens = cpu("slot_tokens", live)[0]
    mapping: dict[int, int] = {}
    for slot in range(capacity):
        if not bool(alive_mask[slot].item()):
            continue
        record = authority.records_sidecar[slot]
        terminal_id = int(terminal_ids[slot].item())
        terminal_token = int(terminal_tokens[slot].item())
        changed = (
            record is None
            or int(record["agent_id"]) != terminal_id
            or int(record["agent_slot_token"]) != terminal_token
        )
        if not changed:
            continue
        if (
            latest_child_by_slot.get(slot) != terminal_id
            or terminal_id not in origin_by_id
        ):
            raise _fail("terminal modular owner has no exact lifecycle ancestry")
        mapping[slot] = origin_by_id[terminal_id]
    return mapping


def derive_modular_lifecycle_structural_origins_v2(
    prepared: PreparedDirectResidentModularV1,
) -> dict[int, int]:
    """Read exact resident cold-record ancestry and validate hereditary state."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if len(prepared.modular_authorities) != 1 or prepared.batch_cells != 1:
        raise _fail("structural-origin derivation requires exactly one lifecycle cell")
    authority = prepared.modular_authorities[0]
    capacity = authority.layout.agent_capacity
    live_planes = {
        field.name: tensor[0].detach().to(device="cpu").contiguous()
        for field, tensor in zip(
            authority.layout.fields, prepared.modular_tensors, strict=True
        )
    }
    alive_mask = prepared.patch.live_banks["alive_mask"][0].detach().cpu().bool()
    terminal_ids = prepared.patch.live_banks["slot_to_agent_id"][0].detach().cpu()
    terminal_tokens = prepared.patch.live_banks["slot_tokens"][0].detach().cpu()
    if (
        not isinstance(prepared.cold_origin_slots, torch.Tensor)
        or prepared.cold_origin_slots.dtype != torch.int64
        or tuple(prepared.cold_origin_slots.shape) != (1, capacity)
    ):
        raise _fail("structural-origin derivation omitted exact resident ancestry")
    cold_origins = prepared.cold_origin_slots[0].detach().cpu()
    hereditary = tuple(
        field.name
        for field in authority.layout.fields
        if field.name not in MODULAR_RUNTIME_MUTABLE_FIELD_NAMES
        and field.name not in {"agent_occupied", "agent_id", "agent_slot_token"}
    )
    if bool(cold_origins[~alive_mask].ne(-1).any().item()):
        raise _fail("dead modular rows retain cold-record ancestry")
    sequence_replay = None
    expected_mutated_structure = None
    if prepared.turnover_mutation_draws is not None:
        sequence = project_modular_turnover_mutation_sequence_v1(prepared)
        if sequence is None:
            raise _fail("mutation qualification omitted its canonical sequence")
        try:
            sequence_replay = replay_modular_recurrent_mutation_sequence_v1(
                sequence,
                authority.records_sidecar,
                identity_table=authority.identity_table,
            )
        except ModularRecurrentMutationSequenceReplayError as exc:
            raise _fail(
                f"resident mutation sequence ancestry replay failed: {exc}"
            ) from exc
        terminal_records = tuple(
            record if bool(alive_mask[slot].item()) else None
            for slot, record in enumerate(sequence_replay.final_records)
        )
        expected_mutated_structure = pack_modular_storage(
            terminal_records,
            agent_capacity=capacity,
            profile_id=authority.layout.profile_id,
        )
        if expected_mutated_structure.identity_table != authority.identity_table:
            raise _fail("resident mutation sequence changed the identity table")
    mapping: dict[int, int] = {}
    for slot in range(capacity):
        if not bool(alive_mask[slot].item()):
            continue
        source_slot = int(cold_origins[slot].item())
        if (
            source_slot < 0
            or source_slot >= capacity
            or authority.records_sidecar[source_slot] is None
        ):
            raise _fail(
                "live modular owner has no exact cold-record ancestry: "
                f"slot={slot} source_slot={source_slot}"
            )
        if sequence_replay is not None:
            expected_record = sequence_replay.final_records[slot]
            expected_origin = sequence_replay.final_origin_slot_by_record_slot[slot]
            if (
                expected_record is None
                or expected_origin != source_slot
                or int(expected_record["agent_id"]) != int(terminal_ids[slot].item())
                or int(expected_record["agent_slot_token"])
                != int(terminal_tokens[slot].item())
            ):
                raise _fail(
                    "resident mutation sequence terminal ownership or origin mismatch"
                )
        if not all(
            torch.equal(
                live_planes[name][slot],
                (
                    authority.tensors[name][source_slot]
                    if expected_mutated_structure is None
                    else expected_mutated_structure.tensors[name][slot]
                ),
            )
            for name in hereditary
        ):
            raise _fail(
                "resident cold-record ancestry contradicts hereditary planes: "
                f"slot={slot} source_slot={source_slot}"
            )
        record = authority.records_sidecar[slot]
        changed = (
            record is None
            or int(record["agent_id"]) != int(terminal_ids[slot].item())
            or int(record["agent_slot_token"]) != int(terminal_tokens[slot].item())
        )
        if not changed:
            if source_slot != slot:
                raise _fail("unchanged modular owner changed cold-record ancestry")
            continue
        mapping[slot] = source_slot
    return mapping


def _config(
    prepared: PreparedDirectResidentModularV1,
    *,
    lifecycle_enabled: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "enabled": True,
        "lifecycle_enabled": lifecycle_enabled,
        "cells": prepared.batch_cells,
        "capacity": int(prepared.patch.source.live_state["alive_mask"].shape[1]),
        "mature_start_step": 20,
        "late_start_step": 40,
    }
    if lifecycle_enabled:
        if (
            len(prepared.modular_shadows) != MODULAR_STATE_BANK_FIELD_COUNT
            or not isinstance(prepared.lifecycle_pointer_table, torch.Tensor)
            or not isinstance(prepared.cold_origin_slots, torch.Tensor)
            or not isinstance(prepared.cold_origin_slots_shadow, torch.Tensor)
        ):
            raise _fail("modular lifecycle source/shadow table is incomplete")
        config["lifecycle_pointer_table"] = prepared.lifecycle_pointer_table
        config["cold_origin_slots"] = prepared.cold_origin_slots
        config["cold_origin_slots_shadow"] = prepared.cold_origin_slots_shadow
    if prepared.action_residual_trace is not None:
        config["action_residual_trace"] = prepared.action_residual_trace
    if prepared.turnover_mutation_draws is not None:
        mutation_tensors = {
            "turnover_mutation_draws": prepared.turnover_mutation_draws,
            "duplicate_descendant_mask": prepared.duplicate_descendant_mask,
            "duplicate_descendant_mask_shadow": (
                prepared.duplicate_descendant_mask_shadow
            ),
            "turnover_mutation_receipt_rows": (prepared.turnover_mutation_receipt_rows),
            "turnover_mutation_receipt_counts": (
                prepared.turnover_mutation_receipt_counts
            ),
            "turnover_mutation_receipt_overflow": (
                prepared.turnover_mutation_receipt_overflow
            ),
        }
        if any(
            not isinstance(value, torch.Tensor) for value in mutation_tensors.values()
        ):
            raise _fail("modular turnover mutation tensors are incomplete")
        config.update(mutation_tensors)
        config["turnover_mutation_first_generation"] = (
            prepared.patch.source.generation_base + 1
        )
    if prepared.ordinary_birth_mutation_draws is not None:
        if prepared.turnover_mutation_draws is not None:
            raise _fail(
                "ordinary-birth and turnover mutation modes are mutually exclusive"
            )
        mutation_tensors = {
            "ordinary_birth_mutation_draws": (
                prepared.ordinary_birth_mutation_draws
            ),
            "duplicate_descendant_mask": prepared.duplicate_descendant_mask,
            "duplicate_descendant_mask_shadow": (
                prepared.duplicate_descendant_mask_shadow
            ),
            "ordinary_birth_mutation_receipt_rows": (
                prepared.ordinary_birth_mutation_receipt_rows
            ),
            "ordinary_birth_mutation_receipt_counts": (
                prepared.ordinary_birth_mutation_receipt_counts
            ),
            "ordinary_birth_mutation_receipt_overflow": (
                prepared.ordinary_birth_mutation_receipt_overflow
            ),
            "ordinary_birth_rowv1_receipt_rows": (
                prepared.ordinary_birth_rowv1_receipt_rows
            ),
            "ordinary_birth_rowv1_receipt_counts": (
                prepared.ordinary_birth_rowv1_receipt_counts
            ),
            "ordinary_birth_rowv1_receipt_overflow": (
                prepared.ordinary_birth_rowv1_receipt_overflow
            ),
        }
        if any(
            not isinstance(value, torch.Tensor) for value in mutation_tensors.values()
        ):
            raise _fail("modular ordinary-birth mutation tensors are incomplete")
        config.update(mutation_tensors)
        config["ordinary_birth_mutation_first_generation"] = (
            prepared.patch.source.generation_base
        )
    return config


def with_modular_action_residual_trace_v1(
    prepared: PreparedDirectResidentModularV1,
    *,
    output_ticks_per_cell: int,
) -> PreparedDirectResidentModularV1:
    """Attach the optional validation-only modular action trace."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if type(output_ticks_per_cell) is not int or output_ticks_per_cell < 1:
        raise _fail("modular action trace ticks must be a positive integer")
    capacity = int(prepared.patch.source.live_state["alive_mask"].shape[1])
    trace = torch.full(
        (prepared.batch_cells, output_ticks_per_cell, capacity, 7),
        float("nan"),
        dtype=torch.float32,
        device=prepared.modular_tensors[0].device,
    )
    return replace(prepared, action_residual_trace=trace)


def with_modular_turnover_mutation_qualification_v1(
    prepared: PreparedDirectResidentModularV1,
    *,
    mutation_draws: torch.Tensor,
    duplicate_descendant_mask: torch.Tensor,
) -> PreparedDirectResidentModularV1:
    """Attach the explicit two-draw turnover mutation qualification tape."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if len(prepared.modular_shadows) != MODULAR_STATE_BANK_FIELD_COUNT:
        raise _fail("modular turnover mutation requires lifecycle shadow planes")
    capacity = int(prepared.patch.source.live_state["alive_mask"].shape[1])
    expected_draw_shape = (
        prepared.batch_cells,
        MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
        2,
    )
    expected_mask_shape = (prepared.batch_cells, capacity, 16)
    if (
        not isinstance(mutation_draws, torch.Tensor)
        or mutation_draws.dtype != torch.float64
        or tuple(mutation_draws.shape) != expected_draw_shape
        or mutation_draws.requires_grad
        or not bool(torch.isfinite(mutation_draws).all())
        or bool((mutation_draws < 0.0).any())
        or bool((mutation_draws >= 1.0).any())
    ):
        raise _fail(
            "modular turnover mutation draws must be finite float64"
            f"{list(expected_draw_shape)} values in [0,1)"
        )
    if (
        not isinstance(duplicate_descendant_mask, torch.Tensor)
        or duplicate_descendant_mask.dtype != torch.bool
        or tuple(duplicate_descendant_mask.shape) != expected_mask_shape
        or duplicate_descendant_mask.requires_grad
    ):
        raise _fail(
            f"modular duplicate-descendant mask must be bool{list(expected_mask_shape)}"
        )
    device = prepared.modular_tensors[0].device
    draws = mutation_draws.to(device=device).contiguous()
    mask = duplicate_descendant_mask.to(device=device).contiguous()
    live_count = int(
        prepared.patch.source.live_state["alive_mask"].sum().detach().cpu().item()
    )
    if live_count < 1:
        raise _fail("modular turnover mutation requires a nonempty population")
    arguments = list(prepared.patch.arguments)
    arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1] = 0
    arguments[ORDINARY_BIRTH_OUTPUTS_ARGUMENT_INDEX_V1] = []
    arguments[CULL_FRACTION_ARGUMENT_INDEX_V1] = 1.0 / float(live_count)
    patch = replace(prepared.patch, arguments=tuple(arguments))
    return replace(
        prepared,
        patch=patch,
        turnover_mutation_draws=draws,
        duplicate_descendant_mask=mask,
        duplicate_descendant_mask_shadow=torch.empty_like(mask),
        turnover_mutation_receipt_rows=torch.full(
            (
                prepared.batch_cells,
                MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
                MODULAR_TURNOVER_MUTATION_RECEIPT_WIDTH_V1,
            ),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        turnover_mutation_receipt_counts=torch.zeros(
            (prepared.batch_cells,), dtype=torch.int64, device=device
        ),
        turnover_mutation_receipt_overflow=torch.zeros(
            (prepared.batch_cells,), dtype=torch.int32, device=device
        ),
    )


def with_modular_ordinary_birth_mutation_qualification_v1(
    prepared: PreparedDirectResidentModularV1,
    *,
    mutation_draws: torch.Tensor,
    duplicate_descendant_mask: torch.Tensor,
) -> PreparedDirectResidentModularV1:
    """Attach a bounded two-draw mutation to real ordinary births."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if prepared.turnover_mutation_draws is not None:
        raise _fail("ordinary-birth and turnover mutation modes are mutually exclusive")
    if len(prepared.modular_shadows) != MODULAR_STATE_BANK_FIELD_COUNT:
        raise _fail("modular ordinary-birth mutation requires lifecycle shadow planes")
    capacity = int(prepared.patch.source.live_state["alive_mask"].shape[1])
    expected_draw_shape = (
        prepared.batch_cells,
        MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
        2,
    )
    expected_mask_shape = (prepared.batch_cells, capacity, 16)
    if (
        not isinstance(mutation_draws, torch.Tensor)
        or mutation_draws.dtype != torch.float64
        or tuple(mutation_draws.shape) != expected_draw_shape
        or mutation_draws.requires_grad
        or not bool(torch.isfinite(mutation_draws).all())
        or bool((mutation_draws < 0.0).any())
        or bool((mutation_draws >= 1.0).any())
    ):
        raise _fail(
            "modular ordinary-birth mutation draws must be finite float64"
            f"{list(expected_draw_shape)} values in [0,1)"
        )
    if (
        not isinstance(duplicate_descendant_mask, torch.Tensor)
        or duplicate_descendant_mask.dtype != torch.bool
        or tuple(duplicate_descendant_mask.shape) != expected_mask_shape
        or duplicate_descendant_mask.requires_grad
    ):
        raise _fail(
            f"modular duplicate-descendant mask must be bool{list(expected_mask_shape)}"
        )
    arguments = list(prepared.patch.arguments)
    if (
        type(arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1]) is not int
        or arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1] <= 0
    ):
        raise _fail("modular ordinary-birth mutation requires live births")
    arguments[TURNOVER_ARGUMENT_INDEX_V1] = False
    for index in TURNOVER_MUTABLE_ARGUMENT_INDICES_V1:
        arguments[index] = []
    patch = replace(prepared.patch, arguments=tuple(arguments))
    device = prepared.modular_tensors[0].device
    draws = mutation_draws.to(device=device).contiguous()
    mask = duplicate_descendant_mask.to(device=device).contiguous()
    return replace(
        prepared,
        patch=patch,
        ordinary_birth_mutation_draws=draws,
        duplicate_descendant_mask=mask,
        duplicate_descendant_mask_shadow=torch.empty_like(mask),
        ordinary_birth_mutation_receipt_rows=torch.full(
            (
                prepared.batch_cells,
                MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
                MODULAR_TURNOVER_MUTATION_RECEIPT_WIDTH_V1,
            ),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        ordinary_birth_mutation_receipt_counts=torch.zeros(
            (prepared.batch_cells,), dtype=torch.int64, device=device
        ),
        ordinary_birth_mutation_receipt_overflow=torch.zeros(
            (prepared.batch_cells,), dtype=torch.int32, device=device
        ),
        ordinary_birth_rowv1_receipt_rows=torch.full(
            (
                prepared.batch_cells,
                MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1,
                MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_WIDTH_V3,
            ),
            -1,
            dtype=torch.int64,
            device=device,
        ),
        ordinary_birth_rowv1_receipt_counts=torch.zeros(
            (prepared.batch_cells,), dtype=torch.int64, device=device
        ),
        ordinary_birth_rowv1_receipt_overflow=torch.zeros(
            (prepared.batch_cells,), dtype=torch.int32, device=device
        ),
    )


def _policy_mutation_contract(
    prepared: PreparedDirectResidentModularV1,
) -> tuple[Any, Any, Any, Any]:
    projection = prepared.patch.checkpoint_projection_inputs.get(
        "pre_generation_canonical_projection"
    )
    metadata = projection.get("metadata") if isinstance(projection, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise _fail("modular source omitted canonical policy-mutation metadata")
    return (
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )


def _lifecycle_tensors(
    prepared: PreparedDirectResidentModularV1,
) -> list[torch.Tensor]:
    if len(prepared.modular_shadows) != MODULAR_STATE_BANK_FIELD_COUNT:
        raise _fail("modular lifecycle requires exactly 57 shadow planes")
    return [*prepared.modular_tensors, *prepared.modular_shadows]


def _disable_lifecycle_arguments(
    arguments: list[Any],
    *,
    retain_reporting_outputs: bool,
) -> None:
    arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1] = 0
    arguments[TURNOVER_ARGUMENT_INDEX_V1] = False
    arguments[ORDINARY_BIRTH_OUTPUTS_ARGUMENT_INDEX_V1] = []
    indices = (
        TURNOVER_MUTABLE_ARGUMENT_INDICES_V1
        if retain_reporting_outputs
        else TURNOVER_TENSOR_ARGUMENT_INDICES_V1
    )
    for index in indices:
        arguments[index] = []


def _bounded_arguments(
    prepared: PreparedDirectResidentModularV1,
    *,
    generation_count: int,
    ticks_per_generation: int,
) -> list[Any]:
    if generation_count not in {1, 2}:
        raise _fail("physical modular qualification requires generation_count 1 or 2")
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation > 80
    ):
        raise _fail("physical modular ticks_per_generation must be in [1,80]")
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1
    ][:, :physical_ticks]
    full_tick_outputs = list(arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1])
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1])
    patch_stats[0] = patch_stats[0][:, :physical_ticks]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V1] = ticks_per_generation
    arguments[CACHE_WIDTH_ARGUMENT_INDEX_V1] = 1
    arguments[DETERMINISTIC_ARGUMENT_INDEX_V1] = True
    _disable_lifecycle_arguments(arguments, retain_reporting_outputs=False)
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    return arguments


def _lifecycle_physical_arguments(
    prepared: PreparedDirectResidentModularV1,
    *,
    generation_count: int,
    ticks_per_generation: int,
) -> list[Any]:
    if generation_count not in {1, 2}:
        raise _fail("physical modular lifecycle requires generation_count 1 or 2")
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation > 80
    ):
        raise _fail("physical modular lifecycle ticks_per_generation must be in [1,80]")
    arguments = list(prepared.patch.arguments)
    physical_ticks = generation_count * ticks_per_generation
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1] = arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V1
    ][:, :physical_ticks]
    full_tick_outputs = list(arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1])
    full_tick_outputs[-1] = full_tick_outputs[-1][:generation_count]
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V1] = full_tick_outputs
    patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1])
    patch_stats[0] = patch_stats[0][:, :physical_ticks]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V1] = patch_stats
    arguments[TICKS_ARGUMENT_INDEX_V1] = ticks_per_generation
    arguments[CACHE_WIDTH_ARGUMENT_INDEX_V1] = 1
    arguments[DETERMINISTIC_ARGUMENT_INDEX_V1] = True
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = generation_count
    turnover_enabled = prepared.ordinary_birth_mutation_draws is None
    if (
        type(arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1]) is not int
        or (
            prepared.turnover_mutation_draws is None
            and arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1] <= 0
        )
        or (
            prepared.turnover_mutation_draws is not None
            and arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1] != 0
        )
        or arguments[TURNOVER_ARGUMENT_INDEX_V1] is not turnover_enabled
        or (
            turnover_enabled
            and any(
                not arguments[index]
                for index in TURNOVER_TENSOR_ARGUMENT_INDICES_V1
            )
        )
        or (
            not turnover_enabled
            and (
                any(
                    arguments[index]
                    for index in TURNOVER_MUTABLE_ARGUMENT_INDICES_V1
                )
                or not arguments[TURNOVER_OUTPUT_ARGUMENT_INDEX_V1]
            )
        )
        or (
            prepared.turnover_mutation_draws is None
            and not arguments[ORDINARY_BIRTH_OUTPUTS_ARGUMENT_INDEX_V1]
        )
        or (
            prepared.turnover_mutation_draws is not None
            and arguments[ORDINARY_BIRTH_OUTPUTS_ARGUMENT_INDEX_V1]
        )
    ):
        raise _fail(
            "modular lifecycle requires prepared births, turnover, and output tensors"
        )
    return arguments


def execute_direct_resident_modular_physical_v1(
    module: Any,
    prepared: PreparedDirectResidentModularV1,
    *,
    generation_count: int,
    ticks_per_generation: int,
) -> dict[str, Any]:
    """Execute one bounded native-owned physical modular segment."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    capability = _validated_capability(module)
    _require_action_trace_capability(capability, prepared)
    entrypoint = getattr(module, MODULAR_PHYSICAL_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {MODULAR_PHYSICAL_ENTRYPOINT_V1}"
        )
    raw = entrypoint(
        list(prepared.modular_tensors),
        _config(prepared),
        *_bounded_arguments(
            prepared,
            generation_count=generation_count,
            ticks_per_generation=ticks_per_generation,
        ),
    )
    if not isinstance(raw, Mapping):
        raise _fail("modular physical CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    _require_no_native_failure(result)
    expected = {
        "modular_native_owner": True,
        "modular_native_controller_abi_version": MODULAR_ABI_VERSION_V1,
        "modular_physical_gate_entrypoint": True,
        "modular_python_tick_or_generation_ownership": False,
        "modular_lifecycle_supported": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"modular physical result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


def execute_direct_resident_modular_lifecycle_physical_v2(
    module: Any,
    prepared: PreparedDirectResidentModularV1,
    *,
    generation_count: int,
    ticks_per_generation: int,
) -> dict[str, Any]:
    """Execute the bounded lifecycle-capable physical modular route."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if prepared.ordinary_birth_mutation_draws is not None and (
        generation_count != 1 or ticks_per_generation != 1
    ):
        raise _fail(
            "bounded modular ordinary-birth mutation qualification requires b1g1t1"
        )
    capability = _validated_lifecycle_capability(module)
    _require_action_trace_capability(capability, prepared)
    _require_turnover_mutation_capability(capability, prepared)
    _require_ordinary_birth_mutation_capability(capability, prepared)
    entrypoint = getattr(module, MODULAR_LIFECYCLE_PHYSICAL_ENTRYPOINT_V2, None)
    if not callable(entrypoint):
        raise _fail(
            "CUDA extension is missing callable "
            f"{MODULAR_LIFECYCLE_PHYSICAL_ENTRYPOINT_V2}"
        )
    raw = entrypoint(
        _lifecycle_tensors(prepared),
        _config(prepared, lifecycle_enabled=True),
        *_lifecycle_physical_arguments(
            prepared,
            generation_count=generation_count,
            ticks_per_generation=ticks_per_generation,
        ),
    )
    if not isinstance(raw, Mapping):
        raise _fail("modular lifecycle physical entrypoint returned a non-mapping")
    result = dict(raw)
    _require_no_native_failure(result)
    expected = {
        "modular_native_owner": True,
        "modular_native_controller_abi_version": MODULAR_LIFECYCLE_ABI_VERSION_V2,
        "modular_physical_gate_entrypoint": True,
        "modular_python_tick_or_generation_ownership": False,
        "modular_lifecycle_supported": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"modular lifecycle physical result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


def execute_direct_resident_modular_v1(
    module: Any,
    prepared: PreparedDirectResidentModularV1,
    *,
    logical_generation_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Execute the bounded modular path through the native window controller."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if logical_generation_count < 3 or logical_generation_count > 128:
        raise _fail("resident modular qualification requires generations in [3,128]")
    capability = _validated_capability(module)
    _require_action_trace_capability(capability, prepared)
    entrypoint = getattr(module, MODULAR_DIRECT_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {MODULAR_DIRECT_ENTRYPOINT_V1}"
        )
    arguments = list(prepared.patch.arguments)
    arguments[CACHE_WIDTH_ARGUMENT_INDEX_V1] = 1
    arguments[DETERMINISTIC_ARGUMENT_INDEX_V1] = True
    _disable_lifecycle_arguments(arguments, retain_reporting_outputs=True)
    control = create_run_control_v2(module, run_id=run_id)
    mutation_contract = _policy_mutation_contract(prepared)
    raw = entrypoint(
        control,
        list(prepared.modular_tensors),
        _config(prepared),
        *arguments,
        logical_generation_count,
        *mutation_contract,
    )
    if not isinstance(raw, Mapping):
        raise _fail("modular resident CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    _require_no_native_failure(result)
    expected = {
        "modular_native_owner": True,
        "modular_native_controller_abi_version": MODULAR_ABI_VERSION_V1,
        "modular_python_tick_or_generation_ownership": False,
        "modular_lifecycle_supported": False,
        "completed_generations": logical_generation_count,
        "logical_generation_count": logical_generation_count,
        "window_count": resident_window_count_v1(
            initial_generation=prepared.patch.source.generation_base,
            logical_generation_count=logical_generation_count,
        ),
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"modular resident result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


def execute_direct_resident_modular_lifecycle_v2(
    module: Any,
    prepared: PreparedDirectResidentModularV1,
    *,
    logical_generation_count: int,
    run_id: str,
    run_control: Any | None = None,
    expected_completed_generation_count: int | None = None,
    overflow_probe: bool = False,
) -> dict[str, Any]:
    """Execute a bounded resident modular lifecycle G1-G4 segment."""

    if not isinstance(prepared, PreparedDirectResidentModularV1):
        raise _fail("prepared must be PreparedDirectResidentModularV1")
    if prepared.ordinary_birth_mutation_draws is not None:
        raise _fail(
            "bounded modular ordinary-birth mutation qualification requires "
            "the physical b1g1t1 entrypoint"
        )
    if logical_generation_count not in {1, 2, 3, 4}:
        raise _fail("resident modular lifecycle qualification requires G1-G4")
    if type(overflow_probe) is not bool:
        raise _fail("modular mutation overflow_probe must be a bool")
    mutation_qualified = prepared.turnover_mutation_draws is not None
    if overflow_probe:
        if (
            not mutation_qualified
            or logical_generation_count
            != MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1 + 1
        ):
            raise _fail(
                "modular mutation overflow probe requires the qualified G4 fixture"
            )
    elif (
        mutation_qualified
        and logical_generation_count
        > MODULAR_TURNOVER_MUTATION_SEQUENCE_CAPACITY_V1
    ):
        raise _fail(
            "qualified modular mutation supports successful execution only through G3; "
            "G4 requires explicit overflow_probe=True"
        )
    expected_completed = (
        logical_generation_count
        if expected_completed_generation_count is None
        else expected_completed_generation_count
    )
    if (
        type(expected_completed) is not int
        or expected_completed < 1
        or expected_completed > logical_generation_count
    ):
        raise _fail(
            "expected completed modular lifecycle generations must be in "
            "[1, logical_generation_count]"
        )
    if expected_completed != logical_generation_count and run_control is None:
        raise _fail("partial modular lifecycle execution requires external run control")
    capability = _validated_lifecycle_capability(module)
    _require_action_trace_capability(capability, prepared)
    _require_turnover_mutation_capability(capability, prepared)
    _require_ordinary_birth_mutation_capability(capability, prepared)
    entrypoint = getattr(module, MODULAR_LIFECYCLE_DIRECT_ENTRYPOINT_V2, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {MODULAR_LIFECYCLE_DIRECT_ENTRYPOINT_V2}"
        )
    arguments = list(prepared.patch.arguments)
    arguments[CACHE_WIDTH_ARGUMENT_INDEX_V1] = 1
    arguments[DETERMINISTIC_ARGUMENT_INDEX_V1] = True
    if (
        type(arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1]) is not int
        or (
            prepared.turnover_mutation_draws is None
            and arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1] <= 0
        )
        or (
            prepared.turnover_mutation_draws is not None
            and arguments[MAX_BIRTHS_ARGUMENT_INDEX_V1] != 0
        )
        or arguments[TURNOVER_ARGUMENT_INDEX_V1] is not True
    ):
        raise _fail("resident modular lifecycle requires births and turnover")
    control = (
        create_run_control_v2(module, run_id=run_id)
        if run_control is None
        else run_control
    )
    if run_control is not None:
        manifest = run_control.manifest()
        if not isinstance(manifest, Mapping) or manifest.get("run_id") != run_id:
            raise _fail("external modular lifecycle run control identity mismatch")
    mutation_contract = _policy_mutation_contract(prepared)
    raw = entrypoint(
        control,
        _lifecycle_tensors(prepared),
        _config(prepared, lifecycle_enabled=True),
        *arguments,
        logical_generation_count,
        *mutation_contract,
    )
    if not isinstance(raw, Mapping):
        raise _fail("modular lifecycle resident entrypoint returned a non-mapping")
    result = dict(raw)
    _require_no_native_failure(result)
    expected = {
        "modular_native_owner": True,
        "modular_native_controller_abi_version": MODULAR_LIFECYCLE_ABI_VERSION_V2,
        "modular_python_tick_or_generation_ownership": False,
        "modular_lifecycle_supported": True,
        "completed_generations": expected_completed,
        "logical_generation_count": logical_generation_count,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"modular lifecycle resident result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    return result


__all__ = [
    "MODULAR_CAPABILITY_ENTRYPOINT_V1",
    "MODULAR_DIRECT_ENTRYPOINT_V1",
    "MODULAR_PHYSICAL_ENTRYPOINT_V1",
    "MODULAR_LIFECYCLE_CAPABILITY_ENTRYPOINT_V2",
    "MODULAR_LIFECYCLE_DIRECT_ENTRYPOINT_V2",
    "MODULAR_LIFECYCLE_PHYSICAL_ENTRYPOINT_V2",
    "MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_ABI_VERSION_V3",
    "MODULAR_ORDINARY_BIRTH_ROWV1_RECEIPT_WIDTH_V3",
    "CudaArchitectureBDirectResidentModularV1Error",
    "PreparedDirectResidentModularV1",
    "build_prepared_direct_resident_modular_lifecycle_v2",
    "derive_modular_lifecycle_parent_slots_v2",
    "derive_modular_lifecycle_structural_origins_v2",
    "execute_direct_resident_modular_physical_v1",
    "execute_direct_resident_modular_lifecycle_physical_v2",
    "execute_direct_resident_modular_lifecycle_v2",
    "execute_direct_resident_modular_v1",
    "prepare_direct_resident_modular_lifecycle_v2",
    "prepare_direct_resident_modular_v1",
    "project_direct_resident_modular_lifecycle_checkpoint_v2",
    "project_modular_ordinary_birth_mutation_receipt_v1",
    "project_modular_turnover_mutation_receipt_v1",
    "project_modular_turnover_mutation_sequence_v1",
    "with_modular_action_residual_trace_v1",
    "with_modular_ordinary_birth_mutation_qualification_v1",
    "with_modular_turnover_mutation_qualification_v1",
]
