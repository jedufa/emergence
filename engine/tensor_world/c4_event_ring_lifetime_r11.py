"""CPU evidence driver for R11 event-ring control-boundary lifetime.

Each driver step represents one already-completed captured five-tick graph
window.  Native execution owns event production inside that window.  This
module performs no per-tick drain: it stages the window's typed rows and, when
requested, advances acknowledgement once at the following control boundary.

The driver is evidence-only and CPU-only.  It does not replace the resident
native append route or add host work to normal graph replay.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final

from .c4_event_ring_r11 import (
    TensorWorldC4EventRingR11V1,
    TensorWorldC4EventRingRowR11V1,
    _event_ring_tensors,
)
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
EVENT_RING_SOURCE = ROOT / "engine/tensor_world/c4_event_ring_r11.py"
GRAPH_CHECKPOINT_SOURCE = ROOT / "engine/tensor_world/c5_graph_checkpoint_r11.py"
GRAPH_OWNER_SOURCE = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"

C4_EVENT_RING_LIFETIME_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_event_ring_lifetime_r11_v1"
)
C4_EVENT_RING_LIFETIME_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_EVENT_RING_LIFETIME_R11_WINDOW_TICKS_V1: Final[int] = 5
C4_EVENT_RING_LIFETIME_R11_BOUNDARY_V1: Final[str] = (
    "after_complete_captured_five_tick_window"
)


@dataclass(frozen=True, slots=True)
class TensorWorldC4EventRingWindowBoundaryR11V1:
    """Rows and acknowledgement policy for one completed graph window."""

    generation: int
    tick_start: int
    rows_by_world: tuple[tuple[TensorWorldC4EventRingRowR11V1, ...], ...]
    acknowledge_worlds: tuple[bool, ...]
    window_ticks: int = C4_EVENT_RING_LIFETIME_R11_WINDOW_TICKS_V1
    boundary: str = C4_EVENT_RING_LIFETIME_R11_BOUNDARY_V1

    @property
    def tick_end(self) -> int:
        return self.tick_start + self.window_ticks - 1

    def validate(self, *, batch: int) -> None:
        if (
            type(self.generation) is not int
            or self.generation < 0
            or type(self.tick_start) is not int
            or self.tick_start < 0
            or self.window_ticks != C4_EVENT_RING_LIFETIME_R11_WINDOW_TICKS_V1
            or self.boundary != C4_EVENT_RING_LIFETIME_R11_BOUNDARY_V1
        ):
            raise TensorWorldContractError("R11 event lifetime boundary is invalid")
        if (
            len(self.rows_by_world) != batch
            or len(self.acknowledge_worlds) != batch
            or any(type(value) is not bool for value in self.acknowledge_worlds)
        ):
            raise TensorWorldContractError(
                "R11 event lifetime boundary does not match the batch"
            )
        for rows in self.rows_by_world:
            if not isinstance(rows, tuple):
                raise TensorWorldContractError(
                    "R11 event lifetime rows must be immutable tuples"
                )
            for row in rows:
                if not isinstance(row, TensorWorldC4EventRingRowR11V1):
                    raise TensorWorldContractError(
                        "R11 event lifetime row type is unsupported"
                    )
                if (
                    row.generation != self.generation
                    or row.tick < self.tick_start
                    or row.tick > self.tick_end
                ):
                    raise TensorWorldContractError(
                        "R11 event lifetime row is outside its graph window"
                    )


@dataclass(frozen=True, slots=True)
class TensorWorldC4EventRingWindowTraceR11V1:
    generation: int
    tick_start: int
    tick_end: int
    emitted_sequences_by_world: tuple[tuple[int, ...], ...]
    producer_cursors: tuple[int, ...]
    consumer_cursors: tuple[int, ...]
    bank_counts: tuple[tuple[int, int], ...]
    bank_start_sequences: tuple[tuple[int, int], ...]
    active_banks: tuple[int, ...]


def _trace_after_boundary_v1(
    ring: TensorWorldC4EventRingR11V1,
    boundary: TensorWorldC4EventRingWindowBoundaryR11V1,
    emitted_sequences: tuple[tuple[int, ...], ...],
) -> TensorWorldC4EventRingWindowTraceR11V1:
    return TensorWorldC4EventRingWindowTraceR11V1(
        generation=boundary.generation,
        tick_start=boundary.tick_start,
        tick_end=boundary.tick_end,
        emitted_sequences_by_world=emitted_sequences,
        producer_cursors=tuple(int(value) for value in ring.producer_cursor),
        consumer_cursors=tuple(int(value) for value in ring.consumer_cursor),
        bank_counts=tuple(
            (int(ring.bank_counts[world, 0]), int(ring.bank_counts[world, 1]))
            for world in range(ring.batch)
        ),
        bank_start_sequences=tuple(
            (
                int(ring.bank_start_sequences[world, 0]),
                int(ring.bank_start_sequences[world, 1]),
            )
            for world in range(ring.batch)
        ),
        active_banks=tuple(int(value) for value in ring.active_bank),
    )


def execute_c4_event_ring_window_boundary_r11_v1(
    ring: TensorWorldC4EventRingR11V1,
    boundary: TensorWorldC4EventRingWindowBoundaryR11V1,
) -> TensorWorldC4EventRingWindowTraceR11V1:
    """Atomically publish and optionally acknowledge one complete window."""

    ring.validate(semantic=True)
    if ring.device.type != "cpu":
        raise TensorWorldContractError("R11 event lifetime evidence requires CPU")
    boundary.validate(batch=ring.batch)
    shadow = copy.deepcopy(ring)
    emitted = tuple(
        shadow.append(world=world, rows=rows)
        for world, rows in enumerate(boundary.rows_by_world)
    )
    for world, acknowledge in enumerate(boundary.acknowledge_worlds):
        if acknowledge:
            shadow.acknowledge(
                world=world,
                upto_sequence=int(shadow.producer_cursor[world]),
            )
    shadow.validate(semantic=True)
    destination_tensors = _event_ring_tensors(ring)
    source_tensors = _event_ring_tensors(shadow)
    for destination, source in zip(
        destination_tensors, source_tensors, strict=True
    ):
        destination.copy_(source)
    ring.validate(semantic=True)
    return _trace_after_boundary_v1(ring, boundary, emitted)


def run_c4_event_ring_lifetime_r11_v1(
    ring: TensorWorldC4EventRingR11V1,
    boundaries: tuple[TensorWorldC4EventRingWindowBoundaryR11V1, ...],
) -> tuple[TensorWorldC4EventRingWindowTraceR11V1, ...]:
    """Run a contiguous sequence of complete-window boundary transactions."""

    if not isinstance(boundaries, tuple) or not boundaries:
        raise TensorWorldContractError(
            "R11 event lifetime schedule must be a nonempty tuple"
        )
    previous: TensorWorldC4EventRingWindowBoundaryR11V1 | None = None
    for boundary in boundaries:
        boundary.validate(batch=ring.batch)
        if previous is not None and (
            boundary.tick_start != previous.tick_end + 1
            or boundary.generation < previous.generation
        ):
            raise TensorWorldContractError(
                "R11 event lifetime graph windows are not contiguous"
            )
        previous = boundary
    return tuple(
        execute_c4_event_ring_window_boundary_r11_v1(ring, boundary)
        for boundary in boundaries
    )


def c4_event_ring_lifetime_r11_source_receipt_v1() -> dict[str, object]:
    sources = {
        "driver": Path(__file__),
        "event_ring": EVENT_RING_SOURCE,
        "graph_checkpoint": GRAPH_CHECKPOINT_SOURCE,
        "graph_owner": GRAPH_OWNER_SOURCE,
    }
    return {
        "schema_version": C4_EVENT_RING_LIFETIME_R11_SCHEMA_VERSION_V1,
        "decision": C4_EVENT_RING_LIFETIME_R11_DECISION_V1,
        "graph_window_ticks": C4_EVENT_RING_LIFETIME_R11_WINDOW_TICKS_V1,
        "acknowledgement_boundary": C4_EVENT_RING_LIFETIME_R11_BOUNDARY_V1,
        "normal_tick_cpu_drain": False,
        "between_tick_cpu_drain": False,
        "native_event_production": "unchanged_resident_owner",
        "cpu_role": "control_boundary_lifetime_evidence_only",
        "boundary_atomicity": "full_ring_shadow_then_commit",
        "checkpoint_restart": "existing_C5_graph_checkpoint_owner",
        "sequence_policy": "monotonic_absolute_never_reset_on_bank_reuse",
        "gate_claim": False,
        "qualification_impact": {
            "resident_native_tick_route": "PRESERVED_UNCHANGED",
            "event_ring_control_boundary": "DELTA_QUALIFICATION_REQUIRED",
            "broad_requalification_required": False,
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_EVENT_RING_LIFETIME_R11_BOUNDARY_V1",
    "C4_EVENT_RING_LIFETIME_R11_DECISION_V1",
    "C4_EVENT_RING_LIFETIME_R11_SCHEMA_VERSION_V1",
    "C4_EVENT_RING_LIFETIME_R11_WINDOW_TICKS_V1",
    "TensorWorldC4EventRingWindowBoundaryR11V1",
    "TensorWorldC4EventRingWindowTraceR11V1",
    "c4_event_ring_lifetime_r11_source_receipt_v1",
    "execute_c4_event_ring_window_boundary_r11_v1",
    "run_c4_event_ring_lifetime_r11_v1",
]
