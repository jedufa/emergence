"""Reusable typed event storage for the R11 continuation owner.

The historical C4 event tape is intentionally append-only and bounded.  R11
uses a device-friendly double buffer with an absolute producer sequence and a
checkpoint/report consumer cursor.  Reclamation is explicit: a bank is reused
only after every row in it is below the consumer cursor.  Sequence numbers are
never reset when physical storage is recycled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

import torch

from .c4_state import C4EventTapeTensorStateV1
from .contracts import TensorWorldContractError


C4_EVENT_RING_R11_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c4_event_ring_r11_v1"
C4_EVENT_RING_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_EVENT_RING_R11_BANK_COUNT_V1: Final[int] = 2


@dataclass(frozen=True, slots=True)
class TensorWorldC4EventRingRowR11V1:
    event_type: int
    generation: int
    tick: int
    actor_id: int
    target_id: int
    value: float
    actor_slot: int = -1
    target_slot: int = -1
    parent_id: int = -1
    child_id: int = -1
    founder_id: int = -1
    lineage_depth: int = 0
    subtype: int = 0
    reason: int = 0
    success: int = 1


@dataclass(slots=True)
class TensorWorldC4EventRingR11V1:
    """Caller-owned fixed-capacity double-buffer event owner."""

    banks: tuple[C4EventTapeTensorStateV1, C4EventTapeTensorStateV1]
    bank_counts: torch.Tensor
    bank_start_sequences: torch.Tensor
    producer_cursor: torch.Tensor
    consumer_cursor: torch.Tensor
    active_bank: torch.Tensor
    schema_version: str = C4_EVENT_RING_R11_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.bank_counts.device

    @property
    def batch(self) -> int:
        return int(self.bank_counts.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.banks[0].capacity)

    def validate(self, *, semantic: bool = True) -> None:
        if self.schema_version != C4_EVENT_RING_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("R11 event ring schema is unsupported")
        if len(self.banks) != C4_EVENT_RING_R11_BANK_COUNT_V1:
            raise TensorWorldContractError("R11 event ring requires exactly two banks")
        if self.bank_counts.ndim != 2:
            raise TensorWorldContractError("R11 event ring bank_counts must be rank two")
        batch, banks = self.bank_counts.shape
        if banks != C4_EVENT_RING_R11_BANK_COUNT_V1 or self.capacity <= 0:
            raise TensorWorldContractError("R11 event ring dimensions are invalid")
        expected_scalars = (
            ("bank_start_sequences", self.bank_start_sequences, (batch, banks), torch.int64),
            ("producer_cursor", self.producer_cursor, (batch,), torch.int64),
            ("consumer_cursor", self.consumer_cursor, (batch,), torch.int64),
            ("active_bank", self.active_bank, (batch,), torch.int32),
        )
        for name, tensor, shape, dtype in expected_scalars:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.device != self.device
            ):
                raise TensorWorldContractError(f"R11 event ring {name} ABI is invalid")
        for bank in self.banks:
            bank.validate(expected_batch=batch)
            if bank.device != self.device:
                raise TensorWorldContractError("R11 event ring banks share one device")
            if bank.capacity != self.capacity:
                raise TensorWorldContractError("R11 event ring bank capacities differ")
        if not semantic or self.device.type != "cpu":
            return
        for world in range(batch):
            producer = int(self.producer_cursor[world])
            consumer = int(self.consumer_cursor[world])
            active = int(self.active_bank[world])
            if producer < consumer or active not in (0, 1):
                raise TensorWorldContractError("R11 event ring cursor state is invalid")
            occupied: list[tuple[int, int, int]] = []
            for bank_index in range(2):
                count = int(self.bank_counts[world, bank_index])
                start = int(self.bank_start_sequences[world, bank_index])
                if count < 0 or count > self.capacity:
                    raise TensorWorldContractError("R11 event ring bank count is invalid")
                if count == 0:
                    continue
                if start < 0 or start + count > producer:
                    raise TensorWorldContractError("R11 event ring bank sequence range is invalid")
                occupied.append((start, start + count, bank_index))
                tape = self.banks[bank_index]
                for row in range(count):
                    if int(tape.typed_sequence[world, row]) != start + row:
                        raise TensorWorldContractError(
                            "R11 event ring physical row has a non-monotonic sequence"
                        )
            occupied.sort()
            for previous, current in zip(occupied, occupied[1:]):
                if previous[1] > current[0]:
                    raise TensorWorldContractError("R11 event ring banks overlap")

    def _clear_bank(self, world: int, bank_index: int) -> None:
        self.bank_counts[world, bank_index] = 0
        self.bank_start_sequences[world, bank_index] = -1
        self.banks[bank_index].active_count[world] = 0
        self.banks[bank_index].status[world] = 0

    def acknowledge(self, *, world: int, upto_sequence: int) -> None:
        """Advance the report/checkpoint cursor and reclaim fully-read banks."""

        self.validate(semantic=True)
        if self.device.type != "cpu":
            raise TensorWorldContractError("R11 event-ring CPU oracle requires CPU state")
        if type(world) is not int or not 0 <= world < self.batch:
            raise TensorWorldContractError("R11 event-ring world is outside the batch")
        current = int(self.consumer_cursor[world])
        producer = int(self.producer_cursor[world])
        if type(upto_sequence) is not int or upto_sequence < current or upto_sequence > producer:
            raise TensorWorldContractError("R11 event-ring consumer cursor is invalid")
        self.consumer_cursor[world] = upto_sequence
        for bank_index in range(2):
            count = int(self.bank_counts[world, bank_index])
            start = int(self.bank_start_sequences[world, bank_index])
            if count and start + count <= upto_sequence:
                self._clear_bank(world, bank_index)

    def append(
        self, *, world: int, rows: Iterable[TensorWorldC4EventRingRowR11V1]
    ) -> tuple[int, ...]:
        """Preflight then append rows, returning their monotonic sequences."""

        self.validate(semantic=True)
        if self.device.type != "cpu":
            raise TensorWorldContractError("R11 event-ring CPU oracle requires CPU state")
        if type(world) is not int or not 0 <= world < self.batch:
            raise TensorWorldContractError("R11 event-ring world is outside the batch")
        materialized = tuple(rows)
        if any(not isinstance(row, TensorWorldC4EventRingRowR11V1) for row in materialized):
            raise TensorWorldContractError("R11 event-ring rows have an unsupported type")
        # Plan against local counters first: a failure cannot partially append
        # rows or advance the producer cursor.
        bank_counts = [int(self.bank_counts[world, index]) for index in range(2)]
        bank_starts = [int(self.bank_start_sequences[world, index]) for index in range(2)]
        active = int(self.active_bank[world])
        producer = int(self.producer_cursor[world])
        consumer = int(self.consumer_cursor[world])
        placements: list[tuple[int, int, int]] = []
        for _row in materialized:
            if bank_counts[active] >= self.capacity:
                other = 1 - active
                other_end = bank_starts[other] + bank_counts[other]
                if bank_counts[other] and other_end > consumer:
                    raise TensorWorldContractError("R11 event ring capacity is exhausted")
                bank_counts[other] = 0
                bank_starts[other] = producer
                active = other
            if bank_counts[active] == 0:
                bank_starts[active] = producer
            physical_row = bank_counts[active]
            placements.append((active, physical_row, producer))
            bank_counts[active] += 1
            producer += 1
        # Commit the planned rows only after the full capacity check passes.
        for row, (bank_index, physical_row, sequence) in zip(materialized, placements, strict=True):
            tape = self.banks[bank_index]
            _write_ring_row(tape, world, physical_row, row, sequence)
        self.bank_counts[world] = torch.tensor(bank_counts, dtype=torch.int64)
        self.bank_start_sequences[world] = torch.tensor(bank_starts, dtype=torch.int64)
        self.active_bank[world] = active
        self.producer_cursor[world] = producer
        self.validate(semantic=True)
        return tuple(sequence for _bank, _row, sequence in placements)

    def rows(self, *, world: int) -> tuple[TensorWorldC4EventRingRowR11V1, ...]:
        """Return all unconsumed rows in deterministic sequence order."""

        self.validate(semantic=True)
        if self.device.type != "cpu":
            raise TensorWorldContractError("R11 event-ring CPU oracle requires CPU state")
        if type(world) is not int or not 0 <= world < self.batch:
            raise TensorWorldContractError("R11 event-ring world is outside the batch")
        rows: list[tuple[int, TensorWorldC4EventRingRowR11V1]] = []
        for bank_index in range(2):
            count = int(self.bank_counts[world, bank_index])
            tape = self.banks[bank_index]
            for physical_row in range(count):
                rows.append(
                    (
                        int(tape.typed_sequence[world, physical_row]),
                        TensorWorldC4EventRingRowR11V1(
                        event_type=int(tape.typed_event_type[world, physical_row]),
                        generation=int(tape.typed_generation[world, physical_row]),
                        tick=int(tape.typed_tick[world, physical_row]),
                        actor_id=int(tape.typed_actor_id[world, physical_row]),
                        target_id=int(tape.typed_target_id[world, physical_row]),
                        value=float(tape.typed_value[world, physical_row]),
                        actor_slot=int(tape.typed_actor_slot[world, physical_row]),
                        target_slot=int(tape.typed_target_slot[world, physical_row]),
                        parent_id=int(tape.typed_parent_id[world, physical_row]),
                        child_id=int(tape.typed_child_id[world, physical_row]),
                        founder_id=int(tape.typed_founder_id[world, physical_row]),
                        lineage_depth=int(tape.typed_lineage_depth[world, physical_row]),
                        subtype=int(tape.typed_subtype[world, physical_row]),
                        reason=int(tape.typed_reason[world, physical_row]),
                        success=int(tape.typed_success[world, physical_row]),
                        ),
                    )
                )
        rows.sort(key=lambda item: item[0])
        return tuple(row for _sequence, row in rows)


def _write_ring_row(
    tape: C4EventTapeTensorStateV1,
    world: int,
    row_index: int,
    row: TensorWorldC4EventRingRowR11V1,
    sequence: int,
) -> None:
    tape.event_type[world, row_index] = row.event_type
    tape.generation[world, row_index] = row.generation
    tape.tick[world, row_index] = row.tick
    tape.actor_id[world, row_index] = row.actor_id
    tape.target_id[world, row_index] = row.target_id
    tape.value[world, row_index] = row.value
    tape.typed_event_type[world, row_index] = row.event_type
    tape.typed_generation[world, row_index] = row.generation
    tape.typed_tick[world, row_index] = row.tick
    tape.typed_sequence[world, row_index] = sequence
    tape.typed_actor_id[world, row_index] = row.actor_id
    tape.typed_target_id[world, row_index] = row.target_id
    tape.typed_actor_slot[world, row_index] = row.actor_slot
    tape.typed_target_slot[world, row_index] = row.target_slot
    tape.typed_parent_id[world, row_index] = row.parent_id
    tape.typed_child_id[world, row_index] = row.child_id
    tape.typed_founder_id[world, row_index] = row.founder_id
    tape.typed_lineage_depth[world, row_index] = row.lineage_depth
    tape.typed_subtype[world, row_index] = row.subtype
    tape.typed_reason[world, row_index] = row.reason
    tape.typed_success[world, row_index] = row.success
    tape.typed_value[world, row_index] = row.value
    tape.active_count[world] = row_index + 1
    tape.status[world] = 0


def _event_tape_tensors(tape: C4EventTapeTensorStateV1) -> list[torch.Tensor]:
    """Return the stable native ABI order for one ring bank."""

    return [
        tape.event_type, tape.generation, tape.tick, tape.actor_id, tape.target_id,
        tape.value, tape.typed_event_type, tape.typed_generation, tape.typed_tick,
        tape.typed_sequence, tape.typed_actor_id, tape.typed_target_id,
        tape.typed_actor_slot, tape.typed_target_slot, tape.typed_parent_id,
        tape.typed_child_id, tape.typed_founder_id, tape.typed_lineage_depth,
        tape.typed_subtype, tape.typed_reason, tape.typed_success, tape.typed_value,
        tape.active_count, tape.status,
    ]


def _event_ring_tensors(ring: TensorWorldC4EventRingR11V1) -> list[torch.Tensor]:
    """Return metadata followed by both fixed-capacity typed banks."""

    return [
        ring.bank_counts, ring.bank_start_sequences, ring.producer_cursor,
        ring.consumer_cursor, ring.active_bank,
        *_event_tape_tensors(ring.banks[0]),
        *_event_tape_tensors(ring.banks[1]),
    ]


def _make_event_tape(*, batch: int, capacity: int, device: torch.device) -> C4EventTapeTensorStateV1:
    def i32() -> torch.Tensor:
        return torch.zeros((batch, capacity), dtype=torch.int32, device=device)

    def i64() -> torch.Tensor:
        return torch.zeros((batch, capacity), dtype=torch.int64, device=device)

    def f32() -> torch.Tensor:
        return torch.zeros((batch, capacity), dtype=torch.float32, device=device)

    return C4EventTapeTensorStateV1(
        event_type=i32(), generation=i64(), tick=i64(), actor_id=i64(), target_id=i64(),
        value=f32(), typed_event_type=i32(), typed_generation=i64(), typed_tick=i64(),
        typed_sequence=i64(), typed_actor_id=i64(), typed_target_id=i64(),
        typed_actor_slot=i64(), typed_target_slot=i64(), typed_parent_id=i64(),
        typed_child_id=i64(), typed_founder_id=i64(), typed_lineage_depth=i32(),
        typed_subtype=i32(), typed_reason=i32(), typed_success=torch.zeros(
            (batch, capacity), dtype=torch.int8, device=device
        ), typed_value=f32(), active_count=torch.zeros(batch, dtype=torch.int64, device=device),
        status=torch.zeros(batch, dtype=torch.int32, device=device),
    )


def make_c4_event_ring_r11_v1(
    *, batch: int, capacity: int, event_capacity: int, device: torch.device | str = "cpu"
) -> TensorWorldC4EventRingR11V1:
    if type(batch) is not int or batch <= 0 or type(capacity) is not int or capacity <= 0:
        raise TensorWorldContractError("R11 event-ring dimensions must be positive ints")
    if type(event_capacity) is not int or event_capacity <= 0:
        raise TensorWorldContractError("R11 event-ring capacity must be a positive int")
    target = torch.device(device)
    ring = TensorWorldC4EventRingR11V1(
        banks=(_make_event_tape(batch=batch, capacity=event_capacity, device=target),
               _make_event_tape(batch=batch, capacity=event_capacity, device=target)),
        bank_counts=torch.zeros((batch, 2), dtype=torch.int64, device=target),
        bank_start_sequences=torch.full((batch, 2), -1, dtype=torch.int64, device=target),
        producer_cursor=torch.zeros(batch, dtype=torch.int64, device=target),
        consumer_cursor=torch.zeros(batch, dtype=torch.int64, device=target),
        active_bank=torch.zeros(batch, dtype=torch.int32, device=target),
    )
    ring.validate(semantic=target.type == "cpu")
    return ring


def c4_event_ring_r11_source_receipt_v1() -> dict[str, object]:
    return {
        "schema_version": C4_EVENT_RING_R11_SCHEMA_VERSION_V1,
        "decision": C4_EVENT_RING_R11_DECISION_V1,
        "storage": "two fixed-capacity typed banks",
        "producer_cursor": "monotonic absolute event sequence",
        "consumer_cursor": "checkpoint/report acknowledgement boundary",
        "reclamation": "bank reused only after end_sequence <= consumer_cursor",
        "sequence_reset_on_reuse": False,
        "cpu_fallback": False,
        "gate_claim": False,
    }


__all__ = [
    "C4_EVENT_RING_R11_BANK_COUNT_V1",
    "C4_EVENT_RING_R11_DECISION_V1",
    "C4_EVENT_RING_R11_SCHEMA_VERSION_V1",
    "TensorWorldC4EventRingR11V1",
    "TensorWorldC4EventRingRowR11V1",
    "_event_ring_tensors",
    "_event_tape_tensors",
    "c4_event_ring_r11_source_receipt_v1",
    "make_c4_event_ring_r11_v1",
]
