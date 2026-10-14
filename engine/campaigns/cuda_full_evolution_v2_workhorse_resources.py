"""Fail-closed disk accounting for Workhorse V2 ACS checkpoint routes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from . import cuda_full_evolution_v2_perf_surface as _perf_surface


_workhorse = _perf_surface.workhorse_v2
WORKHORSE_CHECKPOINT_STORAGE_ACCOUNTING_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_checkpoint_storage_accounting_v2"
)
WORKHORSE_SIGNED_PLANNING_ACCOUNTING_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_signed_planning_storage_accounting_v2"
)
WORKHORSE_INDEX_BYTES_PER_WORLD_UPPER_BOUND_V2: Final[int] = 1024
WORKHORSE_CHECKPOINT_METADATA_FIXED_BYTES_V2: Final[int] = 4 * 1024**2
WORKHORSE_EVENT_BYTES_PER_DISPATCH_UPPER_BOUND_V2: Final[int] = 256 * 1024
WORKHORSE_FIXED_OUTPUT_BYTES_UPPER_BOUND_V2: Final[int] = 64 * 1024**2


class WorkhorseStorageAccountingError(RuntimeError):
    """Raised when checkpoint storage cannot be bounded exactly enough."""


def workhorse_checkpoint_storage_accounting_v2(
    *,
    contract: Any,
    checkpoint_ticks: Sequence[int],
    acs_checkpoint_slots: int,
    dispatch_count: int,
    compact_output_bytes_upper_bound: int = 0,
) -> dict[str, Any]:
    """Account physical payload, CRCA projection, ACS scratch, and metadata.

    The current canonical occurrence contract retains one immutable physical
    checkpoint and one immutable absolute-tick projection.  The per-world
    index contains range/hash metadata only and adds no state payload copy.
    """

    if type(contract) is not _workhorse.FullEvolutionCudaWorkhorseContractV2:
        raise WorkhorseStorageAccountingError(
            "storage accounting requires an exact Workhorse contract"
        )
    ticks = tuple(checkpoint_ticks)
    target = contract.start_next_tick + contract.total_ticks
    if (
        isinstance(checkpoint_ticks, (str, bytes, bytearray))
        or not ticks
        or any(type(tick) is not int for tick in ticks)
        or tuple(sorted(set(ticks))) != ticks
        or any(not contract.start_next_tick < tick <= target for tick in ticks)
        or ticks[-1] != target
        or type(acs_checkpoint_slots) is not int
        or acs_checkpoint_slots not in {0, 1, 2}
        or type(dispatch_count) is not int
        or dispatch_count < 0
        or type(compact_output_bytes_upper_bound) is not int
        or compact_output_bytes_upper_bound < 0
    ):
        raise WorkhorseStorageAccountingError(
            "storage accounting requires exact absolute checkpoints, ACS slots, and outputs"
        )
    layout = _workhorse.workhorse_layout_v2(contract.shape)
    payload = contract.B * layout.state_bytes_per_world
    checkpoint_count = len(ticks) if acs_checkpoint_slots else 0
    physical_payload_copies = checkpoint_count
    absolute_projection_payload_copies = checkpoint_count
    durable_payload = payload * (
        physical_payload_copies + absolute_projection_payload_copies
    )
    indexed_metadata = checkpoint_count * (
        WORKHORSE_CHECKPOINT_METADATA_FIXED_BYTES_V2
        + contract.B * WORKHORSE_INDEX_BYTES_PER_WORLD_UPPER_BOUND_V2
    )
    scratch_capture_count = min(acs_checkpoint_slots, checkpoint_count)
    scratch_bytes = scratch_capture_count * payload
    event_bytes = dispatch_count * WORKHORSE_EVENT_BYTES_PER_DISPATCH_UPPER_BOUND_V2
    fixed_output = WORKHORSE_FIXED_OUTPUT_BYTES_UPPER_BOUND_V2
    checkpoint_bytes = durable_payload + indexed_metadata
    durable_noncheckpoint = event_bytes + fixed_output + compact_output_bytes_upper_bound
    disk_bytes = checkpoint_bytes + scratch_bytes + durable_noncheckpoint
    return {
        "schema_version": WORKHORSE_CHECKPOINT_STORAGE_ACCOUNTING_SCHEMA_V2,
        "time_authority": "absolute_tick_v1",
        "shape_id": contract.shape_id,
        "physical_B": contract.B,
        "state_bytes_per_world": layout.state_bytes_per_world,
        "checkpoint_payload_bytes": payload,
        "checkpoint_ticks": list(ticks),
        "checkpoint_count": checkpoint_count,
        "physical_checkpoint_payload_copies": physical_payload_copies,
        "absolute_tick_projection_payload_copies": (
            absolute_projection_payload_copies
        ),
        "indexed_world_payload_copies": 0,
        "checkpoint_durable_payload_bytes_upper_bound": durable_payload,
        "checkpoint_index_and_occurrence_metadata_bytes_upper_bound": (
            indexed_metadata
        ),
        "checkpoint_bytes_upper_bound": checkpoint_bytes,
        "acs_checkpoint_slots": acs_checkpoint_slots,
        "acs_pending_capture_files_upper_bound": scratch_capture_count,
        "acs_pending_capture_file_bytes_upper_bound": scratch_bytes,
        "event_dispatch_count_upper_bound": dispatch_count,
        "event_bytes_upper_bound": event_bytes,
        "fixed_output_bytes_upper_bound": fixed_output,
        "compact_output_bytes_upper_bound": compact_output_bytes_upper_bound,
        "durable_noncheckpoint_bytes_upper_bound": durable_noncheckpoint,
        "disk_bytes_upper_bound": disk_bytes,
        "horizon_dependent_resident_bytes": 0,
        "storage_model": (
            "one_physical_payload_plus_one_crca_projection_per_checkpoint_"
            "with_metadata_only_world_index_v2"
        ),
    }


__all__ = [
    "WORKHORSE_CHECKPOINT_STORAGE_ACCOUNTING_SCHEMA_V2",
    "WORKHORSE_CHECKPOINT_METADATA_FIXED_BYTES_V2",
    "WORKHORSE_EVENT_BYTES_PER_DISPATCH_UPPER_BOUND_V2",
    "WORKHORSE_FIXED_OUTPUT_BYTES_UPPER_BOUND_V2",
    "WORKHORSE_INDEX_BYTES_PER_WORLD_UPPER_BOUND_V2",
    "WORKHORSE_SIGNED_PLANNING_ACCOUNTING_SCHEMA_V2",
    "WorkhorseStorageAccountingError",
    "workhorse_checkpoint_storage_accounting_v2",
]
