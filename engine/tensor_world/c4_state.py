"""Resident C4 population transaction state layered over the C3 bundle."""

from __future__ import annotations

from dataclasses import dataclass, fields
import random
from typing import Final

import torch

from .c3_state import TensorWorldStateBundleV3
from .contracts import TensorWorldContractError


C4_STATE_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c4_state_schema_v1"
C4_EVENT_TAPE_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c4_event_tape_schema_v1"
C4_CPYTHON_RANDOM_WORD_COUNT_V1: Final[int] = 624


@dataclass(slots=True)
class C4PythonRandomTensorStateV1:
    """Explicit CPython MT19937/Gaussian-tape state for scalar genome births."""

    python_random_words: torch.Tensor
    python_random_index: torch.Tensor
    python_random_has_gauss: torch.Tensor
    python_random_gauss_next: torch.Tensor
    python_random_gauss_primary_tape: torch.Tensor
    python_random_gauss_secondary_tape: torch.Tensor
    python_random_raw_draws: torch.Tensor
    mutation_py_semantic_draws: torch.Tensor
    schema_version: str = "tensor_world_c4_cpython_random_state_v1"

    @property
    def device(self) -> torch.device:
        return self.python_random_words.device

    @property
    def batch(self) -> int:
        return int(self.python_random_words.shape[0])

    @property
    def tape_capacity(self) -> int:
        return int(self.python_random_gauss_primary_tape.shape[1])

    def validate(self, *, expected_batch: int | None = None) -> None:
        if self.schema_version != "tensor_world_c4_cpython_random_state_v1":
            raise TensorWorldContractError("C4 CPython RNG schema version is unsupported")
        if self.python_random_words.ndim != 2:
            raise TensorWorldContractError("C4 CPython RNG words must be rank two")
        batch, words = self.python_random_words.shape
        if words != C4_CPYTHON_RANDOM_WORD_COUNT_V1:
            raise TensorWorldContractError("C4 CPython RNG requires 624 MT words")
        if expected_batch is not None and batch != expected_batch:
            raise TensorWorldContractError("C4 CPython RNG batch does not match state")
        tape_shape = tuple(self.python_random_gauss_primary_tape.shape)
        if len(tape_shape) != 2 or tape_shape[0] != batch or tape_shape != tuple(
            self.python_random_gauss_secondary_tape.shape
        ):
            raise TensorWorldContractError("C4 CPython Gaussian tapes have incompatible shapes")
        expected: tuple[tuple[str, torch.Tensor, tuple[int, ...], torch.dtype], ...] = (
            ("python_random_words", self.python_random_words, (batch, words), torch.uint32),
            ("python_random_index", self.python_random_index, (batch,), torch.int64),
            ("python_random_has_gauss", self.python_random_has_gauss, (batch,), torch.bool),
            ("python_random_gauss_next", self.python_random_gauss_next, (batch,), torch.float64),
            ("python_random_gauss_primary_tape", self.python_random_gauss_primary_tape, tape_shape, torch.float64),
            ("python_random_gauss_secondary_tape", self.python_random_gauss_secondary_tape, tape_shape, torch.float64),
            ("python_random_raw_draws", self.python_random_raw_draws, (batch,), torch.int64),
            ("mutation_py_semantic_draws", self.mutation_py_semantic_draws, (batch,), torch.int64),
        )
        devices = {self.device}
        for name, tensor, shape, dtype in expected:
            if not isinstance(tensor, torch.Tensor):
                raise TensorWorldContractError(f"C4 CPython RNG {name} is not a tensor")
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise TensorWorldContractError(
                    f"C4 CPython RNG {name} requires shape={shape!r} dtype={dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"C4 CPython RNG {name} must be contiguous")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("C4 CPython RNG state must share one device")
        if bool((self.python_random_index < 0).any().item()) or bool(
            (self.python_random_index > C4_CPYTHON_RANDOM_WORD_COUNT_V1).any().item()
        ):
            raise TensorWorldContractError("C4 CPython RNG index is outside 624-word state")
        if bool((self.python_random_raw_draws < 0).any().item()) or bool(
            (self.mutation_py_semantic_draws < 0).any().item()
        ):
            raise TensorWorldContractError("C4 CPython RNG counters must be nonnegative")
        if bool((~torch.isfinite(self.python_random_gauss_next)).any().item()) or bool(
            (~torch.isfinite(self.python_random_gauss_primary_tape)).any().item()
        ) or bool((~torch.isfinite(self.python_random_gauss_secondary_tape)).any().item()):
            raise TensorWorldContractError("C4 CPython Gaussian state/tapes must be finite")


def make_c4_cpython_random_state_v1(
    *, batch: int, tape_capacity: int = 2048, seed: int = 0
) -> C4PythonRandomTensorStateV1:
    """Materialize deterministic MT19937 words for a CPU C4 fixture."""

    if type(batch) is not int or batch <= 0:
        raise TensorWorldContractError("C4 CPython RNG batch must be positive int")
    if type(tape_capacity) is not int or tape_capacity < 0:
        raise TensorWorldContractError("C4 Gaussian tape capacity must be nonnegative int")
    if type(seed) is not int:
        raise TensorWorldContractError("C4 CPython RNG seed must be int")
    words = torch.empty((batch, C4_CPYTHON_RANDOM_WORD_COUNT_V1), dtype=torch.uint32)
    indices = torch.empty(batch, dtype=torch.int64)
    for world in range(batch):
        state = random.Random(int(seed) + world).getstate()
        internal = state[1]
        words[world].copy_(torch.tensor(internal[:C4_CPYTHON_RANDOM_WORD_COUNT_V1], dtype=torch.uint32))
        indices[world] = int(internal[C4_CPYTHON_RANDOM_WORD_COUNT_V1])
    return C4PythonRandomTensorStateV1(
        python_random_words=words,
        python_random_index=indices,
        python_random_has_gauss=torch.zeros(batch, dtype=torch.bool),
        python_random_gauss_next=torch.zeros(batch, dtype=torch.float64),
        python_random_gauss_primary_tape=torch.zeros((batch, tape_capacity), dtype=torch.float64),
        python_random_gauss_secondary_tape=torch.zeros((batch, tape_capacity), dtype=torch.float64),
        python_random_raw_draws=torch.zeros(batch, dtype=torch.int64),
        mutation_py_semantic_draws=torch.zeros(batch, dtype=torch.int64),
    )


@dataclass(slots=True)
class C4EventTapeTensorStateV1:
    """Fixed-capacity typed lifecycle event tape.

    The column layout is intentionally isomorphic to the source-current
    ``FlatEventTapeColumns`` contract.  Rows are world-local and append-only
    at a transaction boundary; ``active_count`` is the next row/sequence.
    """

    event_type: torch.Tensor
    generation: torch.Tensor
    tick: torch.Tensor
    actor_id: torch.Tensor
    target_id: torch.Tensor
    value: torch.Tensor
    typed_event_type: torch.Tensor
    typed_generation: torch.Tensor
    typed_tick: torch.Tensor
    typed_sequence: torch.Tensor
    typed_actor_id: torch.Tensor
    typed_target_id: torch.Tensor
    typed_actor_slot: torch.Tensor
    typed_target_slot: torch.Tensor
    typed_parent_id: torch.Tensor
    typed_child_id: torch.Tensor
    typed_founder_id: torch.Tensor
    typed_lineage_depth: torch.Tensor
    typed_subtype: torch.Tensor
    typed_reason: torch.Tensor
    typed_success: torch.Tensor
    typed_value: torch.Tensor
    active_count: torch.Tensor
    status: torch.Tensor
    schema_version: str = C4_EVENT_TAPE_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.event_type.device

    @property
    def batch(self) -> int:
        return int(self.event_type.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.event_type.shape[1])

    def validate(self, *, expected_batch: int | None = None) -> None:
        if self.schema_version != C4_EVENT_TAPE_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 event tape schema version is unsupported")
        if self.event_type.ndim != 2:
            raise TensorWorldContractError("C4 event tape columns must be rank two")
        batch, capacity = self.event_type.shape
        if capacity < 0:
            raise TensorWorldContractError("C4 event tape capacity must be nonnegative")
        if expected_batch is not None and batch != expected_batch:
            raise TensorWorldContractError("C4 event tape batch does not match state")
        columns: tuple[tuple[str, torch.Tensor, torch.dtype], ...] = (
            ("event_type", self.event_type, torch.int32),
            ("generation", self.generation, torch.int64),
            ("tick", self.tick, torch.int64),
            ("actor_id", self.actor_id, torch.int64),
            ("target_id", self.target_id, torch.int64),
            ("value", self.value, torch.float32),
            ("typed_event_type", self.typed_event_type, torch.int32),
            ("typed_generation", self.typed_generation, torch.int64),
            ("typed_tick", self.typed_tick, torch.int64),
            ("typed_sequence", self.typed_sequence, torch.int64),
            ("typed_actor_id", self.typed_actor_id, torch.int64),
            ("typed_target_id", self.typed_target_id, torch.int64),
            ("typed_actor_slot", self.typed_actor_slot, torch.int64),
            ("typed_target_slot", self.typed_target_slot, torch.int64),
            ("typed_parent_id", self.typed_parent_id, torch.int64),
            ("typed_child_id", self.typed_child_id, torch.int64),
            ("typed_founder_id", self.typed_founder_id, torch.int64),
            ("typed_lineage_depth", self.typed_lineage_depth, torch.int32),
            ("typed_subtype", self.typed_subtype, torch.int32),
            ("typed_reason", self.typed_reason, torch.int32),
            ("typed_success", self.typed_success, torch.int8),
            ("typed_value", self.typed_value, torch.float32),
        )
        devices = {self.device}
        for name, tensor, dtype in columns:
            if not isinstance(tensor, torch.Tensor):
                raise TensorWorldContractError(f"C4 event column {name} is not a tensor")
            if tuple(tensor.shape) != (batch, capacity) or tensor.dtype != dtype:
                raise TensorWorldContractError(
                    f"C4 event column {name} requires shape={(batch, capacity)!r} "
                    f"dtype={dtype}, got shape={tuple(tensor.shape)!r} dtype={tensor.dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"C4 event column {name} must be contiguous")
            devices.add(tensor.device)
        for name, tensor, shape, dtype in (
            ("active_count", self.active_count, (batch,), torch.int64),
            ("status", self.status, (batch,), torch.int32),
        ):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise TensorWorldContractError(
                    f"C4 event {name} requires shape={shape!r} dtype={dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"C4 event {name} must be contiguous")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("all C4 event state must share one device")
        if bool((self.active_count < 0).any().item()) or bool(
            (self.active_count > capacity).any().item()
        ):
            raise TensorWorldContractError("C4 event active_count is outside capacity")


@dataclass(slots=True)
class C4PopulationTensorStateV1:
    """Population-owned transaction and lineage state.

    The C3 bundle remains authoritative for the live agent tensors.  This
    owner contains only the population transaction projections and fields
    needed to settle death/free-slot state before a later birth commit.
    """

    active_agent_ids: torch.Tensor
    active_agent_slots: torch.Tensor
    active_count: torch.Tensor
    agent_id_to_slot: torch.Tensor
    slot_tokens: torch.Tensor
    next_agent_id: torch.Tensor
    pending_free_agent_ids: torch.Tensor
    pending_free_slots: torch.Tensor
    pending_free_count: torch.Tensor
    death_agent_ids: torch.Tensor
    death_slots: torch.Tensor
    death_reason_codes: torch.Tensor
    death_count: torch.Tensor
    survivor_agent_ids: torch.Tensor
    free_queue: torch.Tensor
    free_queue_count: torch.Tensor
    parent_slots: torch.Tensor
    child_slots: torch.Tensor
    child_pos_y: torch.Tensor
    child_pos_x: torch.Tensor
    child_energy: torch.Tensor
    mate_slots: torch.Tensor
    planned_count: torch.Tensor
    plan_status: torch.Tensor
    scar: torch.Tensor
    repro_cooldown: torch.Tensor
    offspring_count: torch.Tensor
    first_repro_step: torch.Tensor
    lineage_birth_generation: torch.Tensor
    lineage_birth_tick: torch.Tensor
    lineage_founder_id: torch.Tensor
    lineage_depth: torch.Tensor
    lineage_parent_agent_id: torch.Tensor
    patch_habitat_lineage_id: torch.Tensor
    decision_parent_slots: torch.Tensor
    decision_mate_slots: torch.Tensor
    decision_adjacent_counts: torch.Tensor
    decision_reason_codes: torch.Tensor
    decision_count: torch.Tensor
    planned_repro_cooldown: torch.Tensor
    birth_parent_ids: torch.Tensor
    birth_child_ids: torch.Tensor
    birth_count: torch.Tensor
    mutation_occurred: torch.Tensor
    mutation_delta_total: torch.Tensor
    spawn_rng_draws_added: torch.Tensor
    spawn_rng_semantic_draws_added: torch.Tensor
    runner_rng_draws_added: torch.Tensor
    runner_rng_semantic_draws_added: torch.Tensor


@dataclass(slots=True)
class TensorWorldStateBundleV4:
    """C4 owner; population mutations remain outside the C3 tick graph."""

    base: TensorWorldStateBundleV3
    population: C4PopulationTensorStateV1
    schema_version: str = C4_STATE_SCHEMA_VERSION_V1
    events: C4EventTapeTensorStateV1 | None = None
    scalar_rng: C4PythonRandomTensorStateV1 | None = None
    # Optional R11 continuation-owned event ring.  The concrete type lives in
    # ``c4_event_ring_r11`` (which imports this module), so this field is kept
    # structurally typed here to avoid a module cycle.  Binding/capture makes
    # the concrete validation and aliases explicit at the control boundary.
    event_ring: object | None = None

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def batch(self) -> int:
        return self.base.batch

    @property
    def capacity(self) -> int:
        return self.base.capacity

    @property
    def grid_size(self) -> int:
        return self.base.grid_size

    def validate(self) -> None:
        if self.schema_version != C4_STATE_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 state schema version is unsupported")
        self.base.validate()
        if self.events is not None:
            self.events.validate(expected_batch=self.base.batch)
        if self.scalar_rng is not None:
            self.scalar_rng.validate(expected_batch=self.base.batch)
            if self.scalar_rng.device != self.base.device:
                raise TensorWorldContractError("C4 CPython RNG state must share the bundle device")
        if self.event_ring is not None:
            validate = getattr(self.event_ring, "validate", None)
            if not callable(validate):
                raise TensorWorldContractError("C4 event ring does not expose validation")
            validate(semantic=self.base.device.type == "cpu")
            ring_device = getattr(self.event_ring, "device", None)
            ring_batch = getattr(self.event_ring, "batch", None)
            if ring_device != self.base.device or ring_batch != self.base.batch:
                raise TensorWorldContractError(
                    "C4 event ring must share the bundle device and batch"
                )
        batch, capacity = self.batch, self.capacity
        slots = (batch, capacity)
        expected: tuple[tuple[str, torch.Tensor, tuple[int, ...], torch.dtype], ...] = (
            ("active_agent_ids", self.population.active_agent_ids, slots, torch.int64),
            ("active_agent_slots", self.population.active_agent_slots, slots, torch.int32),
            ("active_count", self.population.active_count, (batch,), torch.int64),
            ("agent_id_to_slot", self.population.agent_id_to_slot, slots, torch.int32),
            ("slot_tokens", self.population.slot_tokens, slots, torch.int64),
            ("next_agent_id", self.population.next_agent_id, (batch,), torch.int64),
            ("pending_free_agent_ids", self.population.pending_free_agent_ids, slots, torch.int64),
            ("pending_free_slots", self.population.pending_free_slots, slots, torch.int64),
            ("pending_free_count", self.population.pending_free_count, (batch,), torch.int64),
            ("death_agent_ids", self.population.death_agent_ids, slots, torch.int64),
            ("death_slots", self.population.death_slots, slots, torch.int64),
            ("death_reason_codes", self.population.death_reason_codes, slots, torch.int32),
            ("death_count", self.population.death_count, (batch,), torch.int64),
            ("survivor_agent_ids", self.population.survivor_agent_ids, slots, torch.int64),
            ("free_queue", self.population.free_queue, slots, torch.int64),
            ("free_queue_count", self.population.free_queue_count, (batch,), torch.int64),
            ("parent_slots", self.population.parent_slots, slots, torch.int32),
            ("child_slots", self.population.child_slots, slots, torch.int32),
            ("child_pos_y", self.population.child_pos_y, slots, torch.int32),
            ("child_pos_x", self.population.child_pos_x, slots, torch.int32),
            ("child_energy", self.population.child_energy, slots, torch.float32),
            ("mate_slots", self.population.mate_slots, slots, torch.int32),
            ("planned_count", self.population.planned_count, (batch,), torch.int64),
            ("plan_status", self.population.plan_status, (batch,), torch.int32),
            ("scar", self.population.scar, slots, torch.float32),
            ("repro_cooldown", self.population.repro_cooldown, slots, torch.int32),
            ("offspring_count", self.population.offspring_count, slots, torch.int32),
            ("first_repro_step", self.population.first_repro_step, slots, torch.int32),
            ("lineage_birth_generation", self.population.lineage_birth_generation, slots, torch.int64),
            ("lineage_birth_tick", self.population.lineage_birth_tick, slots, torch.int64),
            ("lineage_founder_id", self.population.lineage_founder_id, slots, torch.int64),
            ("lineage_depth", self.population.lineage_depth, slots, torch.int32),
            ("lineage_parent_agent_id", self.population.lineage_parent_agent_id, slots, torch.int64),
            ("patch_habitat_lineage_id", self.population.patch_habitat_lineage_id, slots, torch.int32),
            ("decision_parent_slots", self.population.decision_parent_slots, slots, torch.int64),
            ("decision_mate_slots", self.population.decision_mate_slots, slots, torch.int64),
            ("decision_adjacent_counts", self.population.decision_adjacent_counts, slots, torch.int32),
            ("decision_reason_codes", self.population.decision_reason_codes, slots, torch.int32),
            ("decision_count", self.population.decision_count, (batch,), torch.int64),
            ("planned_repro_cooldown", self.population.planned_repro_cooldown, slots, torch.int32),
            ("birth_parent_ids", self.population.birth_parent_ids, slots, torch.int64),
            ("birth_child_ids", self.population.birth_child_ids, slots, torch.int64),
            ("birth_count", self.population.birth_count, (batch,), torch.int64),
            ("mutation_occurred", self.population.mutation_occurred, slots, torch.bool),
            ("mutation_delta_total", self.population.mutation_delta_total, slots, torch.float32),
            ("spawn_rng_draws_added", self.population.spawn_rng_draws_added, (batch,), torch.int64),
            ("spawn_rng_semantic_draws_added", self.population.spawn_rng_semantic_draws_added, (batch,), torch.int64),
            ("runner_rng_draws_added", self.population.runner_rng_draws_added, (batch,), torch.int64),
            ("runner_rng_semantic_draws_added", self.population.runner_rng_semantic_draws_added, (batch,), torch.int64),
        )
        devices = {self.base.device}
        for name, tensor, shape, dtype in expected:
            if not isinstance(tensor, torch.Tensor):
                raise TensorWorldContractError(f"C4 {name} is not a tensor")
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise TensorWorldContractError(
                    f"C4 {name} requires shape={shape!r} dtype={dtype}, got "
                    f"shape={tuple(tensor.shape)!r} dtype={tensor.dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"C4 {name} must be contiguous")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("all C4 resident state must share one device")


def c4_state_tensor_names_v1() -> tuple[str, ...]:
    population_names = tuple(
        f"{C4PopulationTensorStateV1.__name__}.{field.name}"
        for field in fields(C4PopulationTensorStateV1)
    )
    event_names = tuple(
        f"{C4EventTapeTensorStateV1.__name__}.{field.name}"
        for field in fields(C4EventTapeTensorStateV1)
        if field.name != "schema_version"
    )
    rng_names = tuple(
        f"{C4PythonRandomTensorStateV1.__name__}.{field.name}"
        for field in fields(C4PythonRandomTensorStateV1)
        if field.name != "schema_version"
    )
    return population_names + event_names + rng_names


__all__ = [
    "C4EventTapeTensorStateV1",
    "C4PythonRandomTensorStateV1",
    "C4_CPYTHON_RANDOM_WORD_COUNT_V1",
    "C4_EVENT_TAPE_SCHEMA_VERSION_V1",
    "C4PopulationTensorStateV1",
    "C4_STATE_SCHEMA_VERSION_V1",
    "TensorWorldStateBundleV4",
    "c4_state_tensor_names_v1",
    "make_c4_cpython_random_state_v1",
]
