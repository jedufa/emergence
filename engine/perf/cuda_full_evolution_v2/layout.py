"""Deterministic resident allocation ledger for CUDA-native Full Evolution V2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from .contract import (
    AGENT_CAPACITY_V2,
    FULL_EVOLUTION_CAPABILITIES_V2,
    MAX_SIGNED_INT64,
    POLICY_ACTIONS_V2,
    POLICY_ALLELE_BYTES_V2,
    RESIDENT_CAPACITY_C_V2,
    TRAIT_ALLELES_V2,
    WORLD_WIDTH_V2,
    CudaFullEvolutionContractError,
    CudaFullEvolutionNativeContractV2,
    validate_cuda_full_evolution_contract_v2,
)

CUDA_NATIVE_ALIGNMENT_BYTES: Final[int] = 64
CUDA_NATIVE_B136_SENTINEL: Final[int] = 136
CUDA_NATIVE_KERNEL_GLOBAL_SCRATCH_BYTES: Final[int] = 0
CUDA_NATIVE_REFERENCE_WINNER_PLANES: Final[int] = 2
CUDA_NATIVE_CELLS_PER_WORLD: Final[int] = WORLD_WIDTH_V2 * WORLD_WIDTH_V2
CUDA_NATIVE_B136_STATIC_LIMIT_BYTES: Final[int] = 1 << 30
CUDA_NATIVE_B136_PREFERRED_LIMIT_BYTES: Final[int] = 256 << 20
CUDA_NATIVE_OWNER_ALLOWANCE_BYTES_V2: Final[int] = 64 << 10
CUDA_NATIVE_OWNER_ALLOWANCE_PURPOSE_V2: Final[tuple[str, ...]] = (
    "checkpoint manifest and schema descriptor",
    "checkpoint digest and seal",
    "native ABI and artifact identity descriptor",
    "bounded device-owner metadata and allocator margin",
)
CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2: Final[dict[str, int]] = {
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
    "u64": 8,
}
CUDA_NATIVE_HEADER_NAMES_V2: Final[tuple[str, ...]] = (
    "world_uid",
    "next_absolute_tick",
    "next_agent_id",
    "active_count",
    "status",
    "ticks_completed",
    *(f"capability_{name}_counter" for name in FULL_EVOLUTION_CAPABILITIES_V2),
    "rng_key",
    "rng_semantic_epoch",
    "birth_count",
    "death_count",
    "mutation_count",
    "recombination_count",
    "wdl_due_count",
    "jungle_active_count",
    "eed_pending_count",
    "reserved_0",
    "reserved_1",
)
CUDA_NATIVE_LIFECYCLE_EVENT_ROW_FIELDS_V2: Final[tuple[tuple[str, str], ...]] = (
    ("absolute_tick", "u64"),
    ("event_id", "u64"),
    ("agent_id", "u64"),
    ("parent_a_id", "u64"),
    ("parent_b_id", "u64"),
    ("event_kind", "u16"),
    ("site_index", "u16"),
    ("status", "u16"),
    ("reserved", "u16"),
)
CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2: Final[int] = 48
CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2: Final[int] = (
    2 * RESIDENT_CAPACITY_C_V2
)
CUDA_NATIVE_LIFECYCLE_EVENT_KIND_NONE_V2: Final[int] = 0
CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2: Final[int] = 1
CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2: Final[int] = 2
CUDA_NATIVE_LIFECYCLE_EVENT_RESERVED_VALUE_V2: Final[int] = 0
CUDA_NATIVE_LIFECYCLE_EVENT_ID_KIND_BITS_V2: Final[int] = 16
CUDA_NATIVE_LIFECYCLE_EVENT_ID_ORDINAL_BITS_V2: Final[int] = 48
CUDA_NATIVE_LIFECYCLE_EVENT_ID_ORDINAL_MASK_V2: Final[int] = (
    1 << CUDA_NATIVE_LIFECYCLE_EVENT_ID_ORDINAL_BITS_V2
) - 1
if (
    sum(
        CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2[dtype]
        for _, dtype in CUDA_NATIVE_LIFECYCLE_EVENT_ROW_FIELDS_V2
    )
    != CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2
):
    raise RuntimeError("CUDA Native V2 lifecycle event row must be exactly 48 bytes")

# Production event transport is an additive sparse wire.  The V2 dense rows
# remain the fixed-position CUDA differential oracle; they are never the
# production transfer or persistence representation.
CUDA_NATIVE_SPARSE_EVENT_WIRE_ABI_V3: Final[str] = (
    "eiel_full_evolution_cuda_sparse_lifecycle_event_v3"
)
CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3: Final[int] = 32
CUDA_NATIVE_SPARSE_EVENT_WORLD_OFFSET_BYTES_V3: Final[int] = 8
CUDA_NATIVE_SPARSE_EVENT_KEY_TICK_BITS_V3: Final[int] = 6
CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_BITS_V3: Final[int] = 2
CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_BITS_V3: Final[int] = 9
CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_BITS_V3: Final[int] = 10
CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_SHIFT_V3: Final[int] = (
    CUDA_NATIVE_SPARSE_EVENT_KEY_TICK_BITS_V3
)
CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_SHIFT_V3: Final[int] = (
    CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_SHIFT_V3
    + CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_BITS_V3
)
CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_SHIFT_V3: Final[int] = (
    CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_SHIFT_V3
    + CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_BITS_V3
)
CUDA_NATIVE_SPARSE_EVENT_KEY_USED_BITS_V3: Final[int] = (
    CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_SHIFT_V3
    + CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_BITS_V3
)
CUDA_NATIVE_SPARSE_EVENT_KEY_RESERVED_MASK_V3: Final[int] = ((1 << 32) - 1) ^ (
    (1 << CUDA_NATIVE_SPARSE_EVENT_KEY_USED_BITS_V3) - 1
)
if CUDA_NATIVE_SPARSE_EVENT_KEY_USED_BITS_V3 > 32:
    raise RuntimeError("CUDA Native sparse event key exceeds 32 bits")

CUDA_NATIVE_STATE_FIELD_SCHEMA_V2: Final[
    tuple[tuple[str, str, tuple[int, ...], int], ...]
] = (
    ("scalar_header_counters", "u64", (len(CUDA_NATIVE_HEADER_NAMES_V2),), 8),
    ("occupancy", "u16", (CUDA_NATIVE_CELLS_PER_WORLD,), 2),
    ("resource", "u16", (CUDA_NATIVE_CELLS_PER_WORLD,), 2),
    ("scent", "u16", (CUDA_NATIVE_CELLS_PER_WORLD,), 2),
    ("waste", "u16", (CUDA_NATIVE_CELLS_PER_WORLD,), 2),
    ("eed_impact", "u16", (CUDA_NATIVE_CELLS_PER_WORLD,), 2),
    ("eed_pending", "u16", (CUDA_NATIVE_CELLS_PER_WORLD,), 2),
    ("wdl_due", "u64", (CUDA_NATIVE_CELLS_PER_WORLD,), 8),
    ("wdl_active", "u8", (CUDA_NATIVE_CELLS_PER_WORLD,), 1),
    ("terrain", "u8", (CUDA_NATIVE_CELLS_PER_WORLD,), 1),
    ("jungle", "u8", (CUDA_NATIVE_CELLS_PER_WORLD,), 1),
    ("position", "u16", (AGENT_CAPACITY_V2,), 2),
    ("energy", "i16", (AGENT_CAPACITY_V2,), 2),
    ("health", "i16", (AGENT_CAPACITY_V2,), 2),
    ("age", "u16", (AGENT_CAPACITY_V2,), 2),
    ("absolute_birth_tick", "u64", (AGENT_CAPACITY_V2,), 8),
    ("alive_words", "u32", (AGENT_CAPACITY_V2 // 32,), 4),
    ("agent_id", "u64", (AGENT_CAPACITY_V2,), 8),
    ("parent_a_id", "u64", (AGENT_CAPACITY_V2,), 8),
    ("parent_b_id", "u64", (AGENT_CAPACITY_V2,), 8),
    ("policy_alleles", "i8", (2, POLICY_ALLELE_BYTES_V2, AGENT_CAPACITY_V2), 1),
    ("trait_alleles", "i16", (2, TRAIT_ALLELES_V2, AGENT_CAPACITY_V2), 2),
    ("expressed_policy_cache", "i32", (POLICY_ACTIONS_V2, 4, AGENT_CAPACITY_V2), 4),
    ("expressed_trait_cache", "i16", (TRAIT_ALLELES_V2, AGENT_CAPACITY_V2), 2),
)
CUDA_NATIVE_DERIVED_CACHE_FIELDS_V2: Final[tuple[str, ...]] = (
    "expressed_policy_cache",
    "expressed_trait_cache",
)


def _checked_add(left: int, right: int, label: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise CudaFullEvolutionContractError(
            f"{label} operands must be nonnegative integers"
        )
    if left > MAX_SIGNED_INT64 - right:
        raise CudaFullEvolutionContractError(f"{label} exceeds signed int64")
    return left + right


def _checked_multiply(left: int, right: int, label: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise CudaFullEvolutionContractError(
            f"{label} operands must be nonnegative integers"
        )
    if left and right > MAX_SIGNED_INT64 // left:
        raise CudaFullEvolutionContractError(f"{label} exceeds signed int64")
    return left * right


def _align_64(value: int) -> int:
    if value > MAX_SIGNED_INT64 - (CUDA_NATIVE_ALIGNMENT_BYTES - 1):
        raise CudaFullEvolutionContractError("aligned layout exceeds signed int64")
    return (
        (value + CUDA_NATIVE_ALIGNMENT_BYTES - 1) // CUDA_NATIVE_ALIGNMENT_BYTES
    ) * CUDA_NATIVE_ALIGNMENT_BYTES


def lifecycle_dispatch_event_bytes_per_world_v2(temporal_k: int) -> int:
    """Return the bounded 2*C*temporal_k lifecycle dispatch allocation."""
    if type(temporal_k) is not int or not 1 <= temporal_k <= 64:
        raise CudaFullEvolutionContractError("temporal_k must be an integer in [1, 64]")
    return _checked_multiply(
        _checked_multiply(
            CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2,
            temporal_k,
            "lifecycle dispatch event rows",
        ),
        CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
        "lifecycle dispatch event bytes",
    )


def encode_lifecycle_event_id_v2(event_kind: int, subject_ordinal: int) -> int:
    """Encode a stable per-world/per-tick event kind and canonical row ordinal.

    The complete event identity is ``(world_uid, absolute_tick, event_id)``.
    Full biological IDs and the site index remain explicit row fields; this
    codec is deliberately unrelated to the transition RNG address.
    """
    if type(event_kind) is not int or not 0 < event_kind < (1 << 16):
        raise CudaFullEvolutionContractError(
            "event_kind must be an integer in [1, 65535]"
        )
    if (
        type(subject_ordinal) is not int
        or not 0 <= subject_ordinal <= CUDA_NATIVE_LIFECYCLE_EVENT_ID_ORDINAL_MASK_V2
    ):
        raise CudaFullEvolutionContractError(
            "subject_ordinal must be an integer in [0, 2^48 - 1]"
        )
    return (
        event_kind << CUDA_NATIVE_LIFECYCLE_EVENT_ID_ORDINAL_BITS_V2
    ) | subject_ordinal


def decode_lifecycle_event_id_v2(event_id: int) -> tuple[int, int]:
    """Invert :func:`encode_lifecycle_event_id_v2` without loss."""
    if type(event_id) is not int or not 0 < event_id <= ((1 << 64) - 1):
        raise CudaFullEvolutionContractError(
            "event_id must be an integer in [1, 2^64 - 1]"
        )
    event_kind = event_id >> CUDA_NATIVE_LIFECYCLE_EVENT_ID_ORDINAL_BITS_V2
    if event_kind == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_NONE_V2:
        raise CudaFullEvolutionContractError(
            "event_id encodes the reserved NONE event kind"
        )
    return event_kind, event_id & CUDA_NATIVE_LIFECYCLE_EVENT_ID_ORDINAL_MASK_V2


def encode_sparse_lifecycle_event_key_v3(
    *, tick_offset: int, event_kind: int, subject_slot: int, site_index: int
) -> int:
    """Pack the reconstructible canonical fields of one sparse event row."""
    limits = (
        (tick_offset, 1 << CUDA_NATIVE_SPARSE_EVENT_KEY_TICK_BITS_V3, "tick_offset"),
        (event_kind, 1 << CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_BITS_V3, "event_kind"),
        (
            subject_slot,
            1 << CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_BITS_V3,
            "subject_slot",
        ),
        (site_index, 1 << CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_BITS_V3, "site_index"),
    )
    for value, limit, label in limits:
        if type(value) is not int or not 0 <= value < limit:
            raise CudaFullEvolutionContractError(
                f"{label} must be an integer in [0, {limit - 1}]"
            )
    if event_kind not in (
        CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2,
        CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2,
    ):
        raise CudaFullEvolutionContractError("event_kind must identify death or birth")
    return (
        tick_offset
        | (event_kind << CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_SHIFT_V3)
        | (subject_slot << CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_SHIFT_V3)
        | (site_index << CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_SHIFT_V3)
    )


def decode_sparse_lifecycle_event_key_v3(
    packed_key: int,
) -> tuple[int, int, int, int]:
    """Decode a V3 key and fail closed on every reserved high bit."""
    if type(packed_key) is not int or not 0 <= packed_key < (1 << 32):
        raise CudaFullEvolutionContractError(
            "packed sparse event key must be an unsigned 32-bit integer"
        )
    if packed_key & CUDA_NATIVE_SPARSE_EVENT_KEY_RESERVED_MASK_V3:
        raise CudaFullEvolutionContractError(
            "packed sparse event key has nonzero reserved bits"
        )
    tick_mask = (1 << CUDA_NATIVE_SPARSE_EVENT_KEY_TICK_BITS_V3) - 1
    kind_mask = (1 << CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_BITS_V3) - 1
    subject_mask = (1 << CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_BITS_V3) - 1
    site_mask = (1 << CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_BITS_V3) - 1
    tick_offset = packed_key & tick_mask
    event_kind = (packed_key >> CUDA_NATIVE_SPARSE_EVENT_KEY_KIND_SHIFT_V3) & kind_mask
    subject_slot = (
        packed_key >> CUDA_NATIVE_SPARSE_EVENT_KEY_SUBJECT_SHIFT_V3
    ) & subject_mask
    site_index = (packed_key >> CUDA_NATIVE_SPARSE_EVENT_KEY_SITE_SHIFT_V3) & site_mask
    if event_kind not in (
        CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2,
        CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2,
    ):
        raise CudaFullEvolutionContractError(
            "packed sparse event key does not identify death or birth"
        )
    return tick_offset, event_kind, subject_slot, site_index


def sparse_lifecycle_event_envelope_bytes_v3(
    *, batch_size: int, event_count: int
) -> int:
    """Return exact offset-plus-record wire bytes with signed-int64 guards."""
    if type(batch_size) is not int or batch_size <= 0:
        raise CudaFullEvolutionContractError("batch_size must be positive")
    if type(event_count) is not int or event_count < 0:
        raise CudaFullEvolutionContractError("event_count must be nonnegative")
    offset_bytes = _checked_multiply(
        _checked_add(batch_size, 1, "sparse world offset count"),
        CUDA_NATIVE_SPARSE_EVENT_WORLD_OFFSET_BYTES_V3,
        "sparse world offset bytes",
    )
    row_bytes = _checked_multiply(
        event_count,
        CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
        "sparse lifecycle event bytes",
    )
    return _checked_add(offset_bytes, row_bytes, "sparse event envelope bytes")


@dataclass(frozen=True, slots=True)
class CudaNativeFieldLayoutV2:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise CudaFullEvolutionContractError("field names must be nonempty strings")
        if (
            not isinstance(self.dtype, str)
            or self.dtype not in CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2
        ):
            raise CudaFullEvolutionContractError(
                f"unsupported CUDA Native V2 dtype: {self.dtype}"
            )
        if type(self.shape) is not tuple or not self.shape:
            raise CudaFullEvolutionContractError("field shapes must be nonempty tuples")
        if (
            type(self.offset_bytes) is not int
            or self.offset_bytes < 0
            or self.offset_bytes % CUDA_NATIVE_ALIGNMENT_BYTES
        ):
            raise CudaFullEvolutionContractError(
                "field offsets must be 64-byte aligned"
            )
        if type(self.bytes) is not int or self.bytes <= 0:
            raise CudaFullEvolutionContractError("fields must have positive byte sizes")
        elements = 1
        for dimension in self.shape:
            if type(dimension) is not int or dimension <= 0:
                raise CudaFullEvolutionContractError(
                    "field dimensions must be positive integers"
                )
            elements = _checked_multiply(elements, dimension, f"{self.name} elements")
        expected_bytes = _checked_multiply(
            elements, CUDA_NATIVE_DTYPE_WIDTH_BYTES_V2[self.dtype], f"{self.name} bytes"
        )
        if self.bytes != expected_bytes:
            raise CudaFullEvolutionContractError(
                f"{self.name} bytes must equal its dtype-width shape product"
            )


@dataclass(frozen=True, slots=True)
class CudaNativeStateLayoutV2:
    fields: tuple[CudaNativeFieldLayoutV2, ...]
    state_bytes_per_world: int
    reference_scratch_bytes_per_world: int
    optimized_kernel_global_scratch_bytes: int = CUDA_NATIVE_KERNEL_GLOBAL_SCRATCH_BYTES

    def __post_init__(self) -> None:
        if type(self.fields) is not tuple or not self.fields:
            raise CudaFullEvolutionContractError("state layout requires ordered fields")
        if (
            type(self.state_bytes_per_world) is not int
            or self.state_bytes_per_world <= 0
            or self.state_bytes_per_world % CUDA_NATIVE_ALIGNMENT_BYTES
        ):
            raise CudaFullEvolutionContractError(
                "state allocation must be 64-byte aligned"
            )
        names: set[str] = set()
        previous_end = 0
        for field in self.fields:
            if not isinstance(field, CudaNativeFieldLayoutV2):
                raise CudaFullEvolutionContractError(
                    "state fields must be CudaNativeFieldLayoutV2"
                )
            if field.name in names:
                raise CudaFullEvolutionContractError(
                    "state fields must have unique names"
                )
            if field.offset_bytes < previous_end:
                raise CudaFullEvolutionContractError(
                    "state fields must be ordered and non-overlapping"
                )
            names.add(field.name)
            previous_end = _checked_add(
                field.offset_bytes, field.bytes, f"{field.name} extent"
            )
        expected_signature: list[tuple[str, str, tuple[int, ...], int, int]] = []
        expected_offset = 0
        for name, dtype, shape, item_bytes in CUDA_NATIVE_STATE_FIELD_SCHEMA_V2:
            expected_offset = _align_64(expected_offset)
            elements = 1
            for dimension in shape:
                elements = _checked_multiply(elements, dimension, f"{name} elements")
            expected_bytes = _checked_multiply(elements, item_bytes, f"{name} bytes")
            expected_signature.append(
                (name, dtype, shape, expected_offset, expected_bytes)
            )
            expected_offset = _checked_add(
                expected_offset, expected_bytes, f"{name} end"
            )
        actual_signature = [
            (field.name, field.dtype, field.shape, field.offset_bytes, field.bytes)
            for field in self.fields
        ]
        if actual_signature != expected_signature:
            raise CudaFullEvolutionContractError(
                "state fields and offsets must exactly match the frozen V2 schema"
            )
        if self.state_bytes_per_world != _align_64(previous_end):
            raise CudaFullEvolutionContractError(
                "state bytes must equal the aligned final field extent"
            )
        if (
            type(self.reference_scratch_bytes_per_world) is not int
            or not 0 <= self.reference_scratch_bytes_per_world <= MAX_SIGNED_INT64
        ):
            raise CudaFullEvolutionContractError(
                "reference scratch bytes must be nonnegative"
            )
        if (
            type(self.optimized_kernel_global_scratch_bytes) is not int
            or self.optimized_kernel_global_scratch_bytes != 0
        ):
            raise CudaFullEvolutionContractError(
                "optimized kernel global scratch must remain zero"
            )

    @property
    def bytes_per_world(self) -> int:
        return self.state_bytes_per_world + self.reference_scratch_bytes_per_world

    @property
    def canonical_allele_bytes_per_slot(self) -> int:
        return 2 * POLICY_ALLELE_BYTES_V2 + 2 * TRAIT_ALLELES_V2 * 2

    @property
    def expressed_cache_bytes_per_slot(self) -> int:
        return POLICY_ALLELE_BYTES_V2 + TRAIT_ALLELES_V2 * 2

    @property
    def genome_planes_bytes_per_slot(self) -> int:
        """Canonical alleles plus reconstructible expressed cache (432 bytes)."""
        return (
            self.canonical_allele_bytes_per_slot + self.expressed_cache_bytes_per_slot
        )

    @property
    def derived_cache_bytes_per_world(self) -> int:
        return sum(
            self.field(name).bytes for name in CUDA_NATIVE_DERIVED_CACHE_FIELDS_V2
        )

    @property
    def canonical_checkpoint_bytes_per_world(self) -> int:
        """Checkpoint bytes before the reconstructible expressed caches."""
        return self.state_bytes_per_world - self.derived_cache_bytes_per_world

    def field(self, name: str) -> CudaNativeFieldLayoutV2:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)


def _state_fields_v2() -> tuple[CudaNativeFieldLayoutV2, ...]:
    # Policy and trait alleles are canonical. Expressed planes are derived cache.
    offset = 0
    fields: list[CudaNativeFieldLayoutV2] = []
    for name, dtype, shape, item_bytes in CUDA_NATIVE_STATE_FIELD_SCHEMA_V2:
        offset = _align_64(offset)
        elements = 1
        for dimension in shape:
            elements = _checked_multiply(elements, dimension, f"{name} elements")
        size = _checked_multiply(elements, item_bytes, f"{name} bytes")
        fields.append(CudaNativeFieldLayoutV2(name, dtype, shape, offset, size))
        offset = _checked_add(offset, size, f"{name} end")
    return tuple(fields)


def state_layout_v2() -> CudaNativeStateLayoutV2:
    """Return the fixed, absolute-tick-independent one-world state layout."""
    fields = _state_fields_v2()
    state_bytes = _align_64(fields[-1].offset_bytes + fields[-1].bytes)
    reference_scratch = _checked_multiply(
        CUDA_NATIVE_REFERENCE_WINNER_PLANES * CUDA_NATIVE_CELLS_PER_WORLD,
        4,
        "reference winner scratch",
    )
    return CudaNativeStateLayoutV2(fields, state_bytes, reference_scratch)


@dataclass(frozen=True, slots=True)
class CudaNativeResourceLedgerV2:
    batch_size: int
    state_bytes_per_world: int
    reference_scratch_bytes_per_world: int
    resident_state_bytes: int
    bounded_dispatch_event_bytes: int
    canonical_checkpoint_staging_slab_bytes: int
    canonical_checkpoint_staging_bytes: int
    reference_scratch_bytes: int
    owner_allowance_bytes: int
    optimized_kernel_global_scratch_bytes: int
    total_bytes: int

    @property
    def bytes_per_world(self) -> int:
        return (
            self.state_bytes_per_world
            + lifecycle_dispatch_event_bytes_per_world_v2(self.temporal_k)
            + 2 * self.canonical_checkpoint_staging_slab_bytes // self.batch_size
            + self.reference_scratch_bytes // self.batch_size
        )

    @property
    def canonical_state_bytes(self) -> int:
        """Backward-compatible name for the resident compact state allocation."""
        return self.resident_state_bytes

    temporal_k: int = 8
    reference_scratch_included: bool = True


def estimate_cuda_native_resources_v2(
    contract: CudaFullEvolutionNativeContractV2,
    *,
    include_reference_scratch: bool = True,
) -> CudaNativeResourceLedgerV2:
    """Account for B resident worlds; total_ticks never affects this result."""
    contract = validate_cuda_full_evolution_contract_v2(contract)
    if type(include_reference_scratch) is not bool:
        raise CudaFullEvolutionContractError(
            "include_reference_scratch must be boolean"
        )
    layout = state_layout_v2()
    state = _checked_multiply(contract.B, layout.state_bytes_per_world, "state bytes")
    event_per_world = lifecycle_dispatch_event_bytes_per_world_v2(contract.temporal_k)
    events = _checked_multiply(contract.B, event_per_world, "dispatch event bytes")
    checkpoint_slab = _checked_multiply(
        contract.B,
        layout.canonical_checkpoint_bytes_per_world,
        "checkpoint staging slab bytes",
    )
    checkpoint_staging = _checked_multiply(
        2, checkpoint_slab, "checkpoint staging bytes"
    )
    scratch = (
        _checked_multiply(
            contract.B,
            layout.reference_scratch_bytes_per_world,
            "reference scratch bytes",
        )
        if include_reference_scratch
        else 0
    )
    total = CUDA_NATIVE_OWNER_ALLOWANCE_BYTES_V2
    for value, label in (
        (state, "resident state total"),
        (events, "dispatch event total"),
        (checkpoint_staging, "checkpoint staging total"),
        (scratch, "reference scratch total"),
    ):
        total = _checked_add(total, value, label)
    return CudaNativeResourceLedgerV2(
        batch_size=contract.B,
        state_bytes_per_world=layout.state_bytes_per_world,
        reference_scratch_bytes_per_world=layout.reference_scratch_bytes_per_world,
        resident_state_bytes=state,
        bounded_dispatch_event_bytes=events,
        canonical_checkpoint_staging_slab_bytes=checkpoint_slab,
        canonical_checkpoint_staging_bytes=checkpoint_staging,
        reference_scratch_bytes=scratch,
        owner_allowance_bytes=CUDA_NATIVE_OWNER_ALLOWANCE_BYTES_V2,
        optimized_kernel_global_scratch_bytes=CUDA_NATIVE_KERNEL_GLOBAL_SCRATCH_BYTES,
        total_bytes=total,
        temporal_k=contract.temporal_k,
        reference_scratch_included=include_reference_scratch,
    )


@dataclass(frozen=True, slots=True)
class CudaNativeBatchAdmissionV2:
    requested_batch_size: int
    selected_batch_size: int
    available_vram_bytes: int
    reserve_bytes: int
    maximum_working_bytes: int
    required_bytes: int
    requested_required_bytes: int
    headroom_bytes: int
    admitted: bool
    partial_diagnostic_available: bool


def select_cuda_native_batch_v2(
    contract: CudaFullEvolutionNativeContractV2,
    *,
    available_vram_bytes: int,
    reserve_bytes: int = 0,
    include_reference_scratch: bool = True,
) -> CudaNativeBatchAdmissionV2:
    """Select the greatest requested B that leaves reserve plus 25% headroom."""
    contract = validate_cuda_full_evolution_contract_v2(contract)
    if (
        type(available_vram_bytes) is not int
        or not 0 <= available_vram_bytes <= MAX_SIGNED_INT64
    ):
        raise CudaFullEvolutionContractError(
            "available_vram_bytes must be an integer in [0, signed-int64 max]"
        )
    if type(reserve_bytes) is not int or not 0 <= reserve_bytes <= MAX_SIGNED_INT64:
        raise CudaFullEvolutionContractError(
            "reserve_bytes must be an integer in [0, signed-int64 max]"
        )
    usable = max(0, available_vram_bytes - reserve_bytes)
    maximum_working = (usable // 4) * 3 + ((usable % 4) * 3) // 4
    requested_ledger = estimate_cuda_native_resources_v2(
        contract, include_reference_scratch=include_reference_scratch
    )
    per_world = (
        requested_ledger.total_bytes - requested_ledger.owner_allowance_bytes
    ) // contract.B
    selected = (
        min(
            contract.B,
            (maximum_working - requested_ledger.owner_allowance_bytes) // per_world,
        )
        if maximum_working >= requested_ledger.owner_allowance_bytes
        else 0
    )
    selected_ledger = (
        estimate_cuda_native_resources_v2(
            replace(contract, B=selected),
            include_reference_scratch=include_reference_scratch,
        )
        if selected
        else None
    )
    required = 0 if selected_ledger is None else selected_ledger.total_bytes
    admitted = requested_ledger.total_bytes <= maximum_working
    return CudaNativeBatchAdmissionV2(
        requested_batch_size=contract.B,
        selected_batch_size=selected,
        available_vram_bytes=available_vram_bytes,
        reserve_bytes=reserve_bytes,
        maximum_working_bytes=maximum_working,
        required_bytes=required,
        requested_required_bytes=requested_ledger.total_bytes,
        headroom_bytes=usable - required,
        admitted=admitted,
        partial_diagnostic_available=selected > 0 and not admitted,
    )


_B136_LEDGER = estimate_cuda_native_resources_v2(
    CudaFullEvolutionNativeContractV2(B=CUDA_NATIVE_B136_SENTINEL, total_ticks=1)
)
CUDA_NATIVE_B136_STATE_AND_SCRATCH_BYTES: Final[int] = _B136_LEDGER.total_bytes
CUDA_NATIVE_B136_DESIGNED_BUFFER_BYTES: Final[int] = _B136_LEDGER.total_bytes
if CUDA_NATIVE_B136_DESIGNED_BUFFER_BYTES >= CUDA_NATIVE_B136_STATIC_LIMIT_BYTES:
    raise RuntimeError(
        "B136 CUDA Native V2 resident ledger exceeds the 1 GiB hard limit"
    )
if CUDA_NATIVE_B136_STATE_AND_SCRATCH_BYTES >= CUDA_NATIVE_B136_PREFERRED_LIMIT_BYTES:
    raise RuntimeError(
        "B136 CUDA Native V2 resident ledger exceeds the 256 MiB preference"
    )
