from __future__ import annotations

"""Decision-ordered CUDA lifecycle publication for ordinary asexual births."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Sequence

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine" / "perf" / "native"
LIFECYCLE_EVENTS_SOURCE = NATIVE / "cuda_architecture_b_birth_lifecycle_events.cu"
COMPILE_SOURCES: Final[tuple[Path, ...]] = (
    LIFECYCLE_EVENTS_SOURCE,
)

EVENT_TAPE_STATUS_OK = 0
EVENT_TAPE_STATUS_CAPACITY_EXCEEDED = 1
EVENT_TAPE_STATUS_INVALID_SEMANTICS = 2
EVENT_TAPE_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "event_tape_type",
    "event_tape_generation",
    "event_tape_tick",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
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
EVENT_TAPE_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.int8,
    torch.float32,
)

CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_birth_lifecycle_event_capabilities_v2",
    "decision": "ARCHB-CUDA/EXACT-FIXED-CAPACITY-ASEXUAL-LIFECYCLE-EVENTS",
    "planner_decision_parent_order": True,
    "public_failure_reasons_1_to_3": True,
    "prefilter_reasons_7_to_9_omitted": True,
    "success_mutation_then_birth_order": True,
    "success_ordinal_alignment_validation": True,
    "legacy_and_16_typed_columns": True,
    "resident_cursor_derived_sequence": True,
    "single_atomic_event_tape_append": True,
    "capacity_failure": "atomic_fail_closed",
    "invalid_semantics": "asynchronous_atomic_fail_closed",
    "fixed_capacity_inputs": True,
    "device_resident_active_counts": True,
    "device_side_preflight": True,
    "host_synchronization_during_publication": False,
    "dynamic_mask_sized_output": False,
    "composable_pybind_guard": True,
    "patch_context_sidecars": False,
    "generation_turnover": False,
    "generation_integration": False,
    "gru": False,
    "jungle": False,
    "wdl": False,
    "promotion_qualified": False,
}

_MODULES: dict[str, Any] = {}


@dataclass(frozen=True)
class CudaBirthLifecycleEventAppendResult:
    active_count: torch.Tensor
    status: torch.Tensor


def _source_digest() -> str:
    digest = sha256()
    for source in COMPILE_SOURCES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_cuda_birth_lifecycle_events_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Architecture B birth lifecycle events require an available CUDA device"
        )
    source_digest = _source_digest()
    module = _MODULES.get(source_digest)
    if module is not None:
        return module
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor>
    eiel_cuda_architecture_b_birth_lifecycle_events_append_v1(
        const std::vector<torch::Tensor>& event_tape_columns,
        torch::Tensor event_tape_active_count,
        torch::Tensor event_tape_status,
        torch::Tensor slot_to_agent_id,
        torch::Tensor lineage_founder_id,
        torch::Tensor lineage_depth,
        torch::Tensor decision_parent_slots,
        torch::Tensor decision_reason_codes,
        torch::Tensor decision_count,
        torch::Tensor parent_agent_ids,
        torch::Tensor child_agent_ids,
        torch::Tensor parent_slots,
        torch::Tensor child_slots,
        torch::Tensor mutation_occurred,
        torch::Tensor mutation_delta_total,
        torch::Tensor child_energy,
        torch::Tensor successful_birth_count,
        torch::Tensor generation,
        torch::Tensor tick
    );
    """
    module = load_inline(
        name=f"eiel_cuda_architecture_b_birth_lifecycle_events_{source_digest[:16]}",
        cpp_sources=[cpp_source],
        cuda_sources=[LIFECYCLE_EVENTS_SOURCE.read_text(encoding="utf-8")],
        functions=["eiel_cuda_architecture_b_birth_lifecycle_events_append_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=[
            "-O2",
            "--fmad=false",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
        ],
        with_cuda=True,
        verbose=bool(verbose),
    )
    _MODULES[source_digest] = module
    return module


def _validate_vector(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    minimum_rows: int,
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must share the lifecycle-event CUDA device")
    if not tensor.is_contiguous() or tensor.ndim != 1:
        raise ValueError(f"{name} must be a contiguous rank-1 tensor")
    if tensor.dtype != dtype or tensor.numel() < minimum_rows:
        raise ValueError(
            f"{name} must be {dtype} with at least {minimum_rows} rows"
        )


def append_birth_lifecycle_events_cuda(
    *,
    tape_columns: Sequence[torch.Tensor],
    active_count: torch.Tensor,
    status: torch.Tensor,
    slot_to_agent_id: torch.Tensor,
    lineage_founder_id: torch.Tensor,
    lineage_depth: torch.Tensor,
    decision_parent_slots: torch.Tensor,
    decision_reason_codes: torch.Tensor,
    decision_count: torch.Tensor,
    parent_agent_ids: torch.Tensor,
    child_agent_ids: torch.Tensor,
    parent_slots: torch.Tensor,
    child_slots: torch.Tensor,
    mutation_occurred: torch.Tensor,
    mutation_delta_total: torch.Tensor,
    child_energy: torch.Tensor,
    successful_birth_count: torch.Tensor,
    generation: torch.Tensor,
    tick: torch.Tensor,
    build_directory: str | Path | None = None,
) -> CudaBirthLifecycleEventAppendResult:
    """Publish decision-ordered failure/mutation/birth rows in one append."""
    if not slot_to_agent_id.is_cuda:
        raise ValueError("birth lifecycle semantic inputs must be CUDA resident")
    device = slot_to_agent_id.device
    slot_count = slot_to_agent_id.numel()
    _validate_vector(
        "slot_to_agent_id",
        slot_to_agent_id,
        dtype=torch.int64,
        device=device,
        minimum_rows=slot_count,
    )
    for name, tensor in (
        ("lineage_founder_id", lineage_founder_id),
        ("lineage_depth", lineage_depth),
    ):
        _validate_vector(
            name,
            tensor,
            dtype=torch.int32,
            device=device,
            minimum_rows=slot_count,
        )
        if tensor.numel() != slot_count:
            raise ValueError(f"{name} must contain one row per resident slot")
    for name, tensor, dtype in (
        ("decision_count", decision_count, torch.int64),
        ("successful_birth_count", successful_birth_count, torch.int64),
        ("generation", generation, torch.int64),
        ("tick", tick, torch.int64),
        ("active_count", active_count, torch.int64),
        ("status", status, torch.int32),
    ):
        _validate_vector(
            name,
            tensor,
            dtype=dtype,
            device=device,
            minimum_rows=1,
        )
        if tensor.numel() != 1:
            raise ValueError(f"{name} must contain exactly one resident value")

    decision_rows = decision_parent_slots.numel()
    _validate_vector(
        "decision_parent_slots",
        decision_parent_slots,
        dtype=torch.int64,
        device=device,
        minimum_rows=decision_rows,
    )
    _validate_vector(
        "decision_reason_codes",
        decision_reason_codes,
        dtype=torch.int32,
        device=device,
        minimum_rows=decision_rows,
    )
    if decision_reason_codes.numel() != decision_rows:
        raise ValueError("planner decision tensors must share one capacity")

    success_capacity = parent_agent_ids.numel()
    for name, tensor, dtype in (
        ("parent_agent_ids", parent_agent_ids, torch.int64),
        ("child_agent_ids", child_agent_ids, torch.int64),
        ("parent_slots", parent_slots, torch.int64),
        ("child_slots", child_slots, torch.int64),
        ("mutation_occurred", mutation_occurred, torch.bool),
        ("mutation_delta_total", mutation_delta_total, torch.float32),
        ("child_energy", child_energy, torch.float32),
    ):
        _validate_vector(
            name,
            tensor,
            dtype=dtype,
            device=device,
            minimum_rows=success_capacity,
        )
        if tensor.numel() != success_capacity:
            raise ValueError("successful birth tensors must share one capacity")

    columns = list(tape_columns)
    if len(columns) != len(EVENT_TAPE_COLUMN_NAMES):
        raise ValueError("birth lifecycle tape requires 6 legacy and 16 typed columns")
    capacity = columns[0].numel()
    for name, tensor, dtype in zip(
        EVENT_TAPE_COLUMN_NAMES,
        columns,
        EVENT_TAPE_DTYPES,
        strict=True,
    ):
        _validate_vector(
            name,
            tensor,
            dtype=dtype,
            device=device,
            minimum_rows=capacity,
        )
        if tensor.numel() != capacity:
            raise ValueError("event tape columns must share one capacity")

    module = load_cuda_birth_lifecycle_events_extension(
        build_directory=build_directory
    )
    outputs = module.eiel_cuda_architecture_b_birth_lifecycle_events_append_v1(
        columns,
        active_count,
        status,
        slot_to_agent_id,
        lineage_founder_id,
        lineage_depth,
        decision_parent_slots,
        decision_reason_codes,
        decision_count,
        parent_agent_ids,
        child_agent_ids,
        parent_slots,
        child_slots,
        mutation_occurred,
        mutation_delta_total,
        child_energy,
        successful_birth_count,
        generation,
        tick,
    )
    return CudaBirthLifecycleEventAppendResult(*outputs)


__all__ = [
    "CUDA_BIRTH_LIFECYCLE_EVENT_CAPABILITIES",
    "CudaBirthLifecycleEventAppendResult",
    "EVENT_TAPE_COLUMN_NAMES",
    "EVENT_TAPE_DTYPES",
    "EVENT_TAPE_STATUS_CAPACITY_EXCEEDED",
    "EVENT_TAPE_STATUS_INVALID_SEMANTICS",
    "EVENT_TAPE_STATUS_OK",
    "append_birth_lifecycle_events_cuda",
    "load_cuda_birth_lifecycle_events_extension",
]
