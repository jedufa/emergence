from __future__ import annotations

"""Synchronization-free fixed-capacity CUDA death lifecycle publication."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Sequence

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine" / "perf" / "native"
LIFECYCLE_EVENTS_SOURCE = NATIVE / "cuda_architecture_b_death_lifecycle_events.cu"
COMPILE_SOURCES: Final[tuple[Path, ...]] = (LIFECYCLE_EVENTS_SOURCE,)

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

CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_death_lifecycle_event_capabilities_v1",
    "decision": "ARCHB-CUDA/FIXED-CAPACITY-DEATH-LIFECYCLE-SIDECAR",
    "dense_owner_order_preserved": True,
    "energy_and_health_death_reasons": True,
    "legacy_and_16_typed_columns": True,
    "resident_cursor_derived_sequence": True,
    "single_atomic_event_tape_append": True,
    "capacity_failure": "atomic_fail_closed",
    "invalid_semantics": "asynchronous_atomic_fail_closed",
    "fixed_capacity_inputs": True,
    "device_resident_active_counts": True,
    "device_side_preflight": True,
    "host_synchronization_during_publication": False,
    "data_dependent_host_shapes": False,
    "composable_pybind_guard": True,
    "generation_integration": False,
    "birth_lifecycle": False,
    "transaction_integration": False,
    "promotion_qualified": False,
}

_MODULES: dict[str, Any] = {}


@dataclass(frozen=True)
class CudaDeathLifecycleEventAppendResult:
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


def load_cuda_death_lifecycle_events_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Architecture B death lifecycle events require an available CUDA device"
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
    eiel_cuda_architecture_b_death_lifecycle_events_append_v1(
        const std::vector<torch::Tensor>& event_tape_columns,
        torch::Tensor event_tape_active_count,
        torch::Tensor event_tape_status,
        torch::Tensor death_agent_ids,
        torch::Tensor death_slots,
        torch::Tensor reason_codes,
        torch::Tensor death_count,
        torch::Tensor generation,
        torch::Tensor tick,
        torch::Tensor lineage_founder_id,
        torch::Tensor lineage_depth
    );
    """
    module = load_inline(
        name=f"eiel_cuda_architecture_b_death_lifecycle_events_{source_digest[:16]}",
        cpp_sources=[cpp_source],
        cuda_sources=[LIFECYCLE_EVENTS_SOURCE.read_text(encoding="utf-8")],
        functions=["eiel_cuda_architecture_b_death_lifecycle_events_append_v1"],
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
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must share the death lifecycle CUDA device")
    if not tensor.is_contiguous() or tensor.ndim != 1:
        raise ValueError(f"{name} must be a contiguous rank-1 tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def append_death_lifecycle_events_cuda(
    *,
    tape_columns: Sequence[torch.Tensor],
    active_count: torch.Tensor,
    status: torch.Tensor,
    death_agent_ids: torch.Tensor,
    death_slots: torch.Tensor,
    reason_codes: torch.Tensor,
    death_count: torch.Tensor,
    generation: torch.Tensor,
    tick: torch.Tensor,
    lineage_founder_id: torch.Tensor,
    lineage_depth: torch.Tensor,
    build_directory: str | Path | None = None,
) -> CudaDeathLifecycleEventAppendResult:
    """Publish the active death prefix to the resident event tape atomically."""
    if not death_agent_ids.is_cuda:
        raise ValueError("death lifecycle semantic inputs must be CUDA resident")
    device = death_agent_ids.device
    death_capacity = death_agent_ids.numel()
    for name, tensor, dtype in (
        ("death_agent_ids", death_agent_ids, torch.int64),
        ("death_slots", death_slots, torch.int64),
        ("reason_codes", reason_codes, torch.int32),
    ):
        _validate_vector(name, tensor, dtype=dtype, device=device)
        if tensor.numel() != death_capacity:
            raise ValueError("death lifecycle inputs must share one fixed capacity")

    for name, tensor, dtype in (
        ("death_count", death_count, torch.int64),
        ("generation", generation, torch.int64),
        ("tick", tick, torch.int64),
        ("active_count", active_count, torch.int64),
        ("status", status, torch.int32),
    ):
        _validate_vector(name, tensor, dtype=dtype, device=device)
        if tensor.numel() != 1:
            raise ValueError(f"{name} must contain exactly one resident value")

    slot_count = lineage_founder_id.numel()
    if slot_count == 0:
        raise ValueError("lineage state must have positive fixed capacity")
    for name, tensor in (
        ("lineage_founder_id", lineage_founder_id),
        ("lineage_depth", lineage_depth),
    ):
        _validate_vector(name, tensor, dtype=torch.int32, device=device)
        if tensor.numel() != slot_count:
            raise ValueError("lineage state must contain one row per resident slot")

    columns = list(tape_columns)
    if len(columns) != len(EVENT_TAPE_COLUMN_NAMES):
        raise ValueError("death lifecycle tape requires 6 legacy and 16 typed columns")
    tape_capacity = columns[0].numel()
    for name, tensor, dtype in zip(
        EVENT_TAPE_COLUMN_NAMES,
        columns,
        EVENT_TAPE_DTYPES,
        strict=True,
    ):
        _validate_vector(name, tensor, dtype=dtype, device=device)
        if tensor.numel() != tape_capacity:
            raise ValueError("event tape columns must share one capacity")

    module = load_cuda_death_lifecycle_events_extension(
        build_directory=build_directory
    )
    outputs = module.eiel_cuda_architecture_b_death_lifecycle_events_append_v1(
        columns,
        active_count,
        status,
        death_agent_ids,
        death_slots,
        reason_codes,
        death_count,
        generation,
        tick,
        lineage_founder_id,
        lineage_depth,
    )
    return CudaDeathLifecycleEventAppendResult(*outputs)


__all__ = [
    "CUDA_DEATH_LIFECYCLE_EVENT_CAPABILITIES",
    "CudaDeathLifecycleEventAppendResult",
    "EVENT_TAPE_COLUMN_NAMES",
    "EVENT_TAPE_DTYPES",
    "EVENT_TAPE_STATUS_CAPACITY_EXCEEDED",
    "EVENT_TAPE_STATUS_INVALID_SEMANTICS",
    "EVENT_TAPE_STATUS_OK",
    "LIFECYCLE_EVENTS_SOURCE",
    "append_death_lifecycle_events_cuda",
    "load_cuda_death_lifecycle_events_extension",
]
