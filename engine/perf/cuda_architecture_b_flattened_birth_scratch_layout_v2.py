"""Checked Python mirror of the flattened-birth scratch v2 layout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Final, Mapping


SCRATCH_SEGMENT_ALIGNMENT_V2: Final[int] = 16
SCRATCH_ARENA_ALIGNMENT_V2: Final[int] = 256
LEGACY_SCRATCH_PREFIX_BYTES_V2: Final[int] = 163_840
PYTHON_RANDOM_WORD_COUNT_V2: Final[int] = 624
MINI_STATE_ROLE_COUNT_V2: Final[int] = 3
GENOME_INT32_PLANE_COUNT_V2: Final[int] = 8
GENOME_FLOAT_PLANE_COUNT_V2: Final[int] = 24
DRIVE_WIDTH_V2: Final[int] = 4
CHEMISTRY_WIDTH_V2: Final[int] = 6
INT64_MAX_V2: Final[int] = (1 << 63) - 1


class BirthScratchLayoutStatusV2(IntEnum):
    OK = 0
    NEGATIVE_CONFIGURATION = 1
    BIRTH_COUNT_EXCEEDS_CAPACITY = 2
    INT64_OVERFLOW = 3


class BirthScratchSegmentV2(str, Enum):
    MINI_SLOT_ROWS = "mini_slot_rows"
    MINI_ALIVE_ROWS = "mini_alive_rows"
    MINI_GENOME_INT32_PLANES = "mini_genome_int32_planes"
    MINI_POLICY_MEMORY_DIMENSION_ROWS = "mini_policy_memory_dimension_rows"
    MINI_POLICY_USE_GRU_ROWS = "mini_policy_use_gru_rows"
    MINI_GENOME_FLOAT_PLANES = "mini_genome_float_planes"
    MINI_ENERGY_ROWS = "mini_energy_rows"
    PARENT_PLAN_ROWS = "parent_plan_rows"
    MATE_PLAN_ROWS = "mate_plan_rows"
    CHILD_PLAN_ROWS = "child_plan_rows"
    IDENTITY_PARENT_SLOTS = "identity_parent_slots"
    IDENTITY_CHILD_SLOTS = "identity_child_slots"
    IDENTITY_CHILD_IDS = "identity_child_ids"
    IDENTITY_CHILD_ENERGY = "identity_child_energy"
    IDENTITY_COUNT = "identity_count"
    QUEUE_HEAD = "queue_head"
    QUEUE_COUNT = "queue_count"
    PARENT_AGENT_IDS = "parent_agent_ids"
    CHILD_LINEAGE_FOUNDER_IDS = "child_lineage_founder_ids"
    CHILD_LINEAGE_DEPTHS = "child_lineage_depths"
    GENOME_INT32_PLANES = "genome_int32_planes"
    POLICY_MEMORY_DIMENSION_ROWS = "policy_memory_dimension_rows"
    POLICY_USE_GRU_ROWS = "policy_use_gru_rows"
    GENOME_FLOAT_PLANES = "genome_float_planes"
    DRIVE_ROWS = "drive_rows"
    CHEMISTRY_ROWS = "chemistry_rows"
    BIRTH_RESET_SHOCK_ROWS = "birth_reset_shock_rows"
    POLICY_MLP_ROWS = "policy_mlp_rows"
    HMEM_ROWS = "hmem_rows"
    PYTHON_RANDOM_POST_WORDS = "python_random_post_words"
    PYTHON_RANDOM_TRIAL_WORDS = "python_random_trial_words"
    PYTHON_RANDOM_SNAPSHOTS = "python_random_snapshots"
    PLANNER_PCG64_SNAPSHOT = "planner_pcg64_snapshot"
    RUNNER_PCG64_SNAPSHOT = "runner_pcg64_snapshot"
    POPULATION_PCG64_SNAPSHOT = "population_pcg64_snapshot"
    TORCH_STATE = "torch_state"
    MUTATION_FLAGS = "mutation_flags"
    MUTATION_DELTAS = "mutation_deltas"
    LINKED_DONOR_ORDINALS = "linked_donor_ordinals"
    DIPLOID_STAGED_CHILDREN = "diploid_staged_children"
    GENOME_UPGRADE_V2_STAGED_CHILDREN = "genome_upgrade_v2_staged_children"
    EVENT_STAGING_ROWS = "event_staging_rows"
    GENOME_UPGRADE_V2_QUALIFICATION_FAULT_STAGE = (
        "genome_upgrade_v2_qualification_fault_stage"
    )
    SCALAR_STATUSES = "scalar_statuses"


BIRTH_SCRATCH_SEGMENTS_V2: Final[tuple[BirthScratchSegmentV2, ...]] = tuple(
    BirthScratchSegmentV2
)


class BirthScratchLayoutArithmeticError(OverflowError):
    """A layout operation cannot be represented in nonnegative int64."""


@dataclass(frozen=True)
class BirthScratchSegmentLayoutV2:
    offset: int
    bytes: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.bytes


@dataclass(frozen=True)
class FlatBirthScratchLayoutV2:
    birth_count: int
    capacity: int
    mini_state_rows: int
    packed_width: int
    hmem_width: int
    torch_state_bytes: int
    prefix_bytes: int
    tail_offset: int
    status: BirthScratchLayoutStatusV2
    segments: Mapping[BirthScratchSegmentV2, BirthScratchSegmentLayoutV2]
    required_bytes: int
    bytes_per_cell: int

    def ok(self) -> bool:
        return self.status is BirthScratchLayoutStatusV2.OK

    def segment(
        self, value: BirthScratchSegmentV2 | str
    ) -> BirthScratchSegmentLayoutV2:
        return self.segments[BirthScratchSegmentV2(value)]

    @property
    def offsets(self) -> Mapping[str, int]:
        """Absolute per-cell offsets, named after the C++ segment enums."""

        return MappingProxyType(
            {segment.value: layout.offset for segment, layout in self.segments.items()}
        )

    @property
    def relative_offsets(self) -> Mapping[str, int]:
        """Header-relative offsets before an optional prefix shift."""

        return MappingProxyType(
            {
                segment.value: layout.offset - self.tail_offset
                for segment, layout in self.segments.items()
            }
        )

    @property
    def segment_bytes(self) -> Mapping[str, int]:
        return MappingProxyType(
            {segment.value: layout.bytes for segment, layout in self.segments.items()}
        )


def _require_exact_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


def checked_multiply_nonnegative_v2(left: int, right: int) -> int:
    left = _require_exact_int("left", left)
    right = _require_exact_int("right", right)
    if left < 0 or right < 0:
        raise BirthScratchLayoutArithmeticError(
            "multiplication operands must be nonnegative"
        )
    if left > INT64_MAX_V2 or right > INT64_MAX_V2:
        raise BirthScratchLayoutArithmeticError(
            "multiplication operands exceed int64"
        )
    if right != 0 and left > INT64_MAX_V2 // right:
        raise BirthScratchLayoutArithmeticError("int64 multiplication overflow")
    return left * right


def checked_add_nonnegative_v2(left: int, right: int) -> int:
    left = _require_exact_int("left", left)
    right = _require_exact_int("right", right)
    if left < 0 or right < 0:
        raise BirthScratchLayoutArithmeticError(
            "addition operands must be nonnegative"
        )
    if left > INT64_MAX_V2 - right:
        raise BirthScratchLayoutArithmeticError("int64 addition overflow")
    return left + right


def checked_align_up_v2(value: int, alignment: int) -> int:
    value = _require_exact_int("value", value)
    alignment = _require_exact_int("alignment", alignment)
    if value < 0 or alignment <= 0:
        raise BirthScratchLayoutArithmeticError(
            "alignment requires a nonnegative value and positive alignment"
        )
    if value > INT64_MAX_V2 or alignment > INT64_MAX_V2:
        raise BirthScratchLayoutArithmeticError(
            "alignment operands exceed int64"
        )
    if value > INT64_MAX_V2 - (alignment - 1):
        raise BirthScratchLayoutArithmeticError("int64 alignment overflow")
    return (value + alignment - 1) // alignment * alignment


def _empty_layout(
    *,
    birth_count: int,
    capacity: int,
    packed_width: int,
    hmem_width: int,
    torch_state_bytes: int,
    prefix_bytes: int,
    status: BirthScratchLayoutStatusV2,
    mini_state_rows: int = 0,
) -> FlatBirthScratchLayoutV2:
    return FlatBirthScratchLayoutV2(
        birth_count=birth_count,
        capacity=capacity,
        mini_state_rows=mini_state_rows,
        packed_width=packed_width,
        hmem_width=hmem_width,
        torch_state_bytes=torch_state_bytes,
        prefix_bytes=prefix_bytes,
        tail_offset=0,
        status=status,
        segments=MappingProxyType({}),
        required_bytes=0,
        bytes_per_cell=0,
    )


def make_flattened_birth_scratch_layout_v2(
    birth_count: int,
    capacity: int,
    packed_width: int,
    hmem_width: int,
    torch_state_bytes: int,
    *,
    prefix_bytes: int = 0,
) -> FlatBirthScratchLayoutV2:
    """Calculate one cell's checked layout, optionally after an existing prefix.

    ``required_bytes`` is the C++ header's birth-scratch extent. ``bytes_per_cell``
    includes the aligned prefix and is the total allocation required by a caller.
    Segment offsets are absolute within that combined per-cell allocation.
    """

    values = {
        "birth_count": birth_count,
        "capacity": capacity,
        "packed_width": packed_width,
        "hmem_width": hmem_width,
        "torch_state_bytes": torch_state_bytes,
        "prefix_bytes": prefix_bytes,
    }
    checked_values = {
        name: _require_exact_int(name, value) for name, value in values.items()
    }
    birth_count = checked_values["birth_count"]
    capacity = checked_values["capacity"]
    packed_width = checked_values["packed_width"]
    hmem_width = checked_values["hmem_width"]
    torch_state_bytes = checked_values["torch_state_bytes"]
    prefix_bytes = checked_values["prefix_bytes"]

    if any(value < 0 for value in checked_values.values()):
        return _empty_layout(
            birth_count=birth_count,
            capacity=capacity,
            packed_width=packed_width,
            hmem_width=hmem_width,
            torch_state_bytes=torch_state_bytes,
            prefix_bytes=prefix_bytes,
            status=BirthScratchLayoutStatusV2.NEGATIVE_CONFIGURATION,
        )
    if any(value > INT64_MAX_V2 for value in checked_values.values()):
        return _empty_layout(
            birth_count=birth_count,
            capacity=capacity,
            packed_width=packed_width,
            hmem_width=hmem_width,
            torch_state_bytes=torch_state_bytes,
            prefix_bytes=prefix_bytes,
            status=BirthScratchLayoutStatusV2.INT64_OVERFLOW,
        )
    if birth_count > capacity:
        return _empty_layout(
            birth_count=birth_count,
            capacity=capacity,
            packed_width=packed_width,
            hmem_width=hmem_width,
            torch_state_bytes=torch_state_bytes,
            prefix_bytes=prefix_bytes,
            status=BirthScratchLayoutStatusV2.BIRTH_COUNT_EXCEEDS_CAPACITY,
        )

    try:
        mini_state_rows = checked_multiply_nonnegative_v2(
            birth_count, MINI_STATE_ROLE_COUNT_V2
        )
        # Every slot can decide once; every successful birth can mutate once.
        event_staging_rows = checked_add_nonnegative_v2(capacity, birth_count)
    except BirthScratchLayoutArithmeticError:
        return _empty_layout(
            birth_count=birth_count,
            capacity=capacity,
            packed_width=packed_width,
            hmem_width=hmem_width,
            torch_state_bytes=torch_state_bytes,
            prefix_bytes=prefix_bytes,
            status=BirthScratchLayoutStatusV2.INT64_OVERFLOW,
        )

    specs = (
        (BirthScratchSegmentV2.MINI_SLOT_ROWS, mini_state_rows, 1, 8),
        (BirthScratchSegmentV2.MINI_ALIVE_ROWS, mini_state_rows, 1, 1),
        (
            BirthScratchSegmentV2.MINI_GENOME_INT32_PLANES,
            GENOME_INT32_PLANE_COUNT_V2,
            mini_state_rows,
            4,
        ),
        (
            BirthScratchSegmentV2.MINI_POLICY_MEMORY_DIMENSION_ROWS,
            mini_state_rows,
            1,
            8,
        ),
        (BirthScratchSegmentV2.MINI_POLICY_USE_GRU_ROWS, mini_state_rows, 1, 1),
        (
            BirthScratchSegmentV2.MINI_GENOME_FLOAT_PLANES,
            GENOME_FLOAT_PLANE_COUNT_V2,
            mini_state_rows,
            4,
        ),
        (BirthScratchSegmentV2.MINI_ENERGY_ROWS, mini_state_rows, 1, 8),
        (BirthScratchSegmentV2.PARENT_PLAN_ROWS, birth_count, 1, 8),
        (BirthScratchSegmentV2.MATE_PLAN_ROWS, birth_count, 1, 8),
        (BirthScratchSegmentV2.CHILD_PLAN_ROWS, birth_count, 1, 8),
        (BirthScratchSegmentV2.IDENTITY_PARENT_SLOTS, birth_count, 1, 8),
        (BirthScratchSegmentV2.IDENTITY_CHILD_SLOTS, birth_count, 1, 8),
        (BirthScratchSegmentV2.IDENTITY_CHILD_IDS, birth_count, 1, 8),
        (BirthScratchSegmentV2.IDENTITY_CHILD_ENERGY, birth_count, 1, 8),
        (BirthScratchSegmentV2.IDENTITY_COUNT, 1, 1, 8),
        (BirthScratchSegmentV2.QUEUE_HEAD, 1, 1, 8),
        (BirthScratchSegmentV2.QUEUE_COUNT, 1, 1, 8),
        (BirthScratchSegmentV2.PARENT_AGENT_IDS, birth_count, 1, 8),
        (BirthScratchSegmentV2.CHILD_LINEAGE_FOUNDER_IDS, birth_count, 1, 4),
        (BirthScratchSegmentV2.CHILD_LINEAGE_DEPTHS, birth_count, 1, 4),
        (
            BirthScratchSegmentV2.GENOME_INT32_PLANES,
            GENOME_INT32_PLANE_COUNT_V2,
            birth_count,
            4,
        ),
        (BirthScratchSegmentV2.POLICY_MEMORY_DIMENSION_ROWS, birth_count, 1, 8),
        (BirthScratchSegmentV2.POLICY_USE_GRU_ROWS, birth_count, 1, 1),
        (
            BirthScratchSegmentV2.GENOME_FLOAT_PLANES,
            GENOME_FLOAT_PLANE_COUNT_V2,
            birth_count,
            4,
        ),
        (BirthScratchSegmentV2.DRIVE_ROWS, birth_count, DRIVE_WIDTH_V2, 4),
        (
            BirthScratchSegmentV2.CHEMISTRY_ROWS,
            birth_count,
            CHEMISTRY_WIDTH_V2,
            4,
        ),
        (BirthScratchSegmentV2.BIRTH_RESET_SHOCK_ROWS, birth_count, 1, 4),
        (BirthScratchSegmentV2.POLICY_MLP_ROWS, birth_count, packed_width, 4),
        (BirthScratchSegmentV2.HMEM_ROWS, birth_count, hmem_width, 4),
        (
            BirthScratchSegmentV2.PYTHON_RANDOM_POST_WORDS,
            1,
            PYTHON_RANDOM_WORD_COUNT_V2,
            4,
        ),
        (
            BirthScratchSegmentV2.PYTHON_RANDOM_TRIAL_WORDS,
            1,
            PYTHON_RANDOM_WORD_COUNT_V2,
            4,
        ),
        (BirthScratchSegmentV2.PYTHON_RANDOM_SNAPSHOTS, 2, 1, 40),
        (BirthScratchSegmentV2.PLANNER_PCG64_SNAPSHOT, 1, 1, 48),
        (BirthScratchSegmentV2.RUNNER_PCG64_SNAPSHOT, 1, 1, 48),
        (BirthScratchSegmentV2.POPULATION_PCG64_SNAPSHOT, 1, 1, 48),
        (BirthScratchSegmentV2.TORCH_STATE, 1, torch_state_bytes, 1),
        (BirthScratchSegmentV2.MUTATION_FLAGS, birth_count, 1, 1),
        (BirthScratchSegmentV2.MUTATION_DELTAS, birth_count, 1, 4),
        (BirthScratchSegmentV2.LINKED_DONOR_ORDINALS, birth_count, 7, 1),
        (BirthScratchSegmentV2.DIPLOID_STAGED_CHILDREN, birth_count, 1, 256),
        (
            BirthScratchSegmentV2.GENOME_UPGRADE_V2_STAGED_CHILDREN,
            birth_count,
            1,
            976,
        ),
        (
            BirthScratchSegmentV2.EVENT_STAGING_ROWS,
            event_staging_rows,
            1,
            104,
        ),
        (
            BirthScratchSegmentV2.GENOME_UPGRADE_V2_QUALIFICATION_FAULT_STAGE,
            1,
            1,
            4,
        ),
        (BirthScratchSegmentV2.SCALAR_STATUSES, 1, 1, 176),
    )

    try:
        tail_offset = checked_align_up_v2(
            prefix_bytes, SCRATCH_ARENA_ALIGNMENT_V2
        )
        cursor = 0
        segments: dict[
            BirthScratchSegmentV2, BirthScratchSegmentLayoutV2
        ] = {}
        for segment, rows, row_width, element_bytes in specs:
            cursor = checked_align_up_v2(cursor, SCRATCH_SEGMENT_ALIGNMENT_V2)
            elements = checked_multiply_nonnegative_v2(rows, row_width)
            byte_count = checked_multiply_nonnegative_v2(elements, element_bytes)
            absolute_offset = checked_add_nonnegative_v2(tail_offset, cursor)
            checked_add_nonnegative_v2(absolute_offset, byte_count)
            segments[segment] = BirthScratchSegmentLayoutV2(
                offset=absolute_offset,
                bytes=byte_count,
            )
            cursor = checked_add_nonnegative_v2(cursor, byte_count)
        required_bytes = checked_align_up_v2(cursor, SCRATCH_ARENA_ALIGNMENT_V2)
        bytes_per_cell = checked_add_nonnegative_v2(tail_offset, required_bytes)
    except BirthScratchLayoutArithmeticError:
        return _empty_layout(
            birth_count=birth_count,
            capacity=capacity,
            packed_width=packed_width,
            hmem_width=hmem_width,
            torch_state_bytes=torch_state_bytes,
            prefix_bytes=prefix_bytes,
            status=BirthScratchLayoutStatusV2.INT64_OVERFLOW,
            mini_state_rows=mini_state_rows,
        )

    return FlatBirthScratchLayoutV2(
        birth_count=birth_count,
        capacity=capacity,
        mini_state_rows=mini_state_rows,
        packed_width=packed_width,
        hmem_width=hmem_width,
        torch_state_bytes=torch_state_bytes,
        prefix_bytes=prefix_bytes,
        tail_offset=tail_offset,
        status=BirthScratchLayoutStatusV2.OK,
        segments=MappingProxyType(segments),
        required_bytes=required_bytes,
        bytes_per_cell=bytes_per_cell,
    )


def make_legacy_tail_birth_scratch_layout_v2(
    birth_count: int,
    capacity: int,
    packed_width: int,
    hmem_width: int,
    torch_state_bytes: int,
) -> FlatBirthScratchLayoutV2:
    """Place the v2 birth scratch after the legacy C768-capable allocation."""

    return make_flattened_birth_scratch_layout_v2(
        birth_count,
        capacity,
        packed_width,
        hmem_width,
        torch_state_bytes,
        prefix_bytes=LEGACY_SCRATCH_PREFIX_BYTES_V2,
    )


make_layout = make_flattened_birth_scratch_layout_v2


__all__ = [
    "BIRTH_SCRATCH_SEGMENTS_V2",
    "BirthScratchLayoutArithmeticError",
    "BirthScratchLayoutStatusV2",
    "BirthScratchSegmentLayoutV2",
    "BirthScratchSegmentV2",
    "FlatBirthScratchLayoutV2",
    "INT64_MAX_V2",
    "LEGACY_SCRATCH_PREFIX_BYTES_V2",
    "MINI_STATE_ROLE_COUNT_V2",
    "SCRATCH_ARENA_ALIGNMENT_V2",
    "SCRATCH_SEGMENT_ALIGNMENT_V2",
    "checked_add_nonnegative_v2",
    "checked_align_up_v2",
    "checked_multiply_nonnegative_v2",
    "make_flattened_birth_scratch_layout_v2",
    "make_layout",
    "make_legacy_tail_birth_scratch_layout_v2",
]
