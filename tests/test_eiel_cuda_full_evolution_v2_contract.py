from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

import pytest

from engine.perf.cuda_full_evolution_v2 import (
    CUDA_NATIVE_B136_SENTINEL,
    CUDA_NATIVE_B136_COMPARISON_PACKED_TICK_MS_100X,
    CUDA_NATIVE_B136_COMPARISON_PACKED_TICKS_PER_SECOND_100X,
    CUDA_NATIVE_HEADER_NAMES_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_RESERVED_VALUE_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROW_FIELDS_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2,
    CUDA_NATIVE_OWNER_ALLOWANCE_PURPOSE_V2,
    CUDA_NATIVE_PACKED_TPS_BASELINE,
    CUDA_NATIVE_STATE_FIELD_SCHEMA_V2,
    CUDA_NATIVE_TARGET_PACKED_TICK_MS_100X,
    CUDA_NATIVE_TARGET_PACKED_TICKS_PER_SECOND_100X,
    CUDA_NATIVE_TARGET_WTPS_100X,
    CUDA_NATIVE_WTPS_BASELINE,
    FULL_EVOLUTION_CAPABILITIES_V2,
    LINKED_DIPLOID_LOCI_V2,
    MAX_TOTAL_TICKS_COUNTER_SAFE_V2,
    CudaFullEvolutionContractError,
    CudaFullEvolutionNativeContractV2,
    CudaNativeFieldLayoutV2,
    CudaNativeStateLayoutV2,
    cuda_native_required_packed_tick_ms_100x_v2,
    cuda_native_required_packed_ticks_per_second_100x_v2,
    decode_lifecycle_event_id_v2,
    decode_sparse_lifecycle_event_key_v3,
    encode_lifecycle_event_id_v2,
    encode_sparse_lifecycle_event_key_v3,
    estimate_cuda_native_resources_v2,
    sparse_lifecycle_event_envelope_bytes_v3,
    lifecycle_dispatch_event_bytes_per_world_v2,
    select_cuda_native_batch_v2,
    state_layout_v2,
)


def test_cuda_native_v2_contract_is_exact_and_absolute_tick_only() -> None:
    contract = CudaFullEvolutionNativeContractV2(
        B=CUDA_NATIVE_B136_SENTINEL, total_ticks=5_000, start_next_tick=17
    )
    assert contract.capabilities == FULL_EVOLUTION_CAPABILITIES_V2
    assert contract.transition_oracle == "cuda_golden_only"
    assert contract.to_dict()["total_ticks"] == 5_000
    assert contract.to_dict()["start_next_tick"] == 17
    assert CUDA_NATIVE_TARGET_WTPS_100X == CUDA_NATIVE_WTPS_BASELINE * 100
    assert CUDA_NATIVE_TARGET_PACKED_TICKS_PER_SECOND_100X == (
        CUDA_NATIVE_PACKED_TPS_BASELINE * 100
    )
    assert CUDA_NATIVE_TARGET_PACKED_TICKS_PER_SECOND_100X == Decimal(
        "3284.700819970491600"
    )
    assert CUDA_NATIVE_TARGET_PACKED_TICK_MS_100X == Decimal("0.30444172995")
    assert CUDA_NATIVE_B136_COMPARISON_PACKED_TICKS_PER_SECOND_100X == (
        CUDA_NATIVE_TARGET_PACKED_TICKS_PER_SECOND_100X
    )
    assert CUDA_NATIVE_B136_COMPARISON_PACKED_TICK_MS_100X == (
        CUDA_NATIVE_TARGET_PACKED_TICK_MS_100X
    )
    for batch in (1, 68, 136, 512):
        packed_tps = cuda_native_required_packed_ticks_per_second_100x_v2(batch)
        packed_tick_ms = cuda_native_required_packed_tick_ms_100x_v2(batch)
        assert packed_tps * batch == CUDA_NATIVE_TARGET_WTPS_100X
        assert packed_tick_ms == (
            Decimal(1000) * Decimal(batch) / CUDA_NATIVE_TARGET_WTPS_100X
        ).quantize(Decimal("0.00000000001"))


@pytest.mark.parametrize(
    "payload",
    (
        {
            "B": 136,
            "total_ticks": 1,
            "capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2[:-1]),
        },
        {
            "B": 136,
            "total_ticks": 1,
            "capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2[:-1]) + ["patch"],
        },
        {"B": 136, "total_ticks": 1, "time_authority": "generation_v1"},
        {"B": 136, "total_ticks": 1, "generation_count": 2},
        {"B": 0, "total_ticks": 1},
        {"B": 1, "total_ticks": 0},
        {"B": 1, "total_ticks": 1, "K": 127},
        {"B": 1, "total_ticks": 1, "temporal_k": 65},
        {"B": 1, "total_ticks": 2, "start_next_tick": (1 << 63) - 1},
        {"B": 1, "total_ticks": 1, "world_width": 32.0},
        {"B": 1, "total_ticks": 1, "initial_population": 256.0},
        {"B": 1, "total_ticks": 1, "agent_capacity": 512.0},
        {"B": 1, "total_ticks": MAX_TOTAL_TICKS_COUNTER_SAFE_V2 + 1},
    ),
)
def test_cuda_native_v2_contract_rejects_inexact_inputs(payload: dict) -> None:
    with pytest.raises(CudaFullEvolutionContractError):
        CudaFullEvolutionNativeContractV2.from_mapping(payload)


def test_layout_is_aligned_compact_and_horizon_independent() -> None:
    layout = state_layout_v2()
    sentinel = CudaFullEvolutionNativeContractV2(B=136, total_ticks=5_000)
    longer = CudaFullEvolutionNativeContractV2(B=136, total_ticks=500_000)
    assert all(field.offset_bytes % 64 == 0 for field in layout.fields)
    assert layout.field("policy_alleles").shape == (2, 112, 512)
    assert layout.field("trait_alleles").shape == (2, 16, 512)
    assert layout.field("scent").dtype == layout.field("waste").dtype == "u16"
    assert layout.field("scent").shape == layout.field("waste").shape == (1024,)
    assert layout.field("terrain").shape == (1024,)
    assert layout.field("absolute_birth_tick").shape == (512,)
    assert layout.field("expressed_policy_cache").dtype == "i32"
    assert layout.field("expressed_policy_cache").shape == (7, 4, 512)
    assert layout.field("expressed_trait_cache").shape == (16, 512)
    assert layout.field("scalar_header_counters").shape == (24,)
    assert layout.field("scalar_header_counters").offset_bytes == 0
    assert len(CUDA_NATIVE_HEADER_NAMES_V2) == 24
    assert CUDA_NATIVE_HEADER_NAMES_V2[:6] == (
        "world_uid",
        "next_absolute_tick",
        "next_agent_id",
        "active_count",
        "status",
        "ticks_completed",
    )
    assert (
        tuple(
            name.removeprefix("capability_").removesuffix("_counter")
            for name in CUDA_NATIVE_HEADER_NAMES_V2[6:13]
        )
        == FULL_EVOLUTION_CAPABILITIES_V2
    )
    assert "rng_counter" not in CUDA_NATIVE_HEADER_NAMES_V2
    assert not any("cursor" in name for name in CUDA_NATIVE_HEADER_NAMES_V2)
    assert layout.canonical_allele_bytes_per_slot == 288
    assert layout.expressed_cache_bytes_per_slot == 144
    assert layout.genome_planes_bytes_per_slot == 432
    assert layout.derived_cache_bytes_per_world == 73_728
    assert layout.canonical_checkpoint_bytes_per_world == (
        layout.state_bytes_per_world - layout.derived_cache_bytes_per_world
    )
    assert layout.optimized_kernel_global_scratch_bytes == 0
    assert tuple(
        (field.name, field.dtype, field.shape) for field in layout.fields
    ) == tuple(
        (name, dtype, shape)
        for name, dtype, shape, _ in CUDA_NATIVE_STATE_FIELD_SCHEMA_V2
    )
    ledger = estimate_cuda_native_resources_v2(sentinel)
    assert ledger.total_bytes < 256 << 20
    assert ledger.total_bytes < 1 << 30
    assert ledger.total_bytes == (
        ledger.resident_state_bytes
        + ledger.bounded_dispatch_event_bytes
        + ledger.canonical_checkpoint_staging_bytes
        + ledger.reference_scratch_bytes
        + ledger.owner_allowance_bytes
    )
    assert (
        ledger.canonical_checkpoint_staging_bytes
        == 2 * ledger.canonical_checkpoint_staging_slab_bytes
    )
    assert len(CUDA_NATIVE_OWNER_ALLOWANCE_PURPOSE_V2) == 4
    assert ledger == estimate_cuda_native_resources_v2(longer)


def test_every_canonical_field_is_covered_by_checkpoint_digest() -> None:
    layout = state_layout_v2()
    canonical = bytearray(layout.canonical_checkpoint_bytes_per_world)
    baseline = sha256(canonical).digest()
    for field in layout.fields:
        if field.name in {"expressed_policy_cache", "expressed_trait_cache"}:
            assert field.offset_bytes >= layout.canonical_checkpoint_bytes_per_world
            continue
        assert (
            field.offset_bytes + field.bytes
            <= layout.canonical_checkpoint_bytes_per_world
        )
        mutated = canonical.copy()
        mutated[field.offset_bytes] ^= 1
        assert sha256(mutated).digest() != baseline


def test_batch_selection_preserves_reserve_and_quarter_headroom() -> None:
    contract = CudaFullEvolutionNativeContractV2(B=136, total_ticks=5_000)
    ledger = estimate_cuda_native_resources_v2(contract)
    available = ledger.total_bytes * 2
    admission = select_cuda_native_batch_v2(contract, available_vram_bytes=available)
    assert admission.admitted
    assert not admission.partial_diagnostic_available
    assert admission.selected_batch_size == 136
    assert admission.headroom_bytes * 4 >= available


def test_dispatch_events_are_bounded_and_temporal_k_sensitive() -> None:
    contract = CudaFullEvolutionNativeContractV2(B=136, total_ticks=5_000, temporal_k=8)
    ledger = estimate_cuda_native_resources_v2(contract)
    no_reference = estimate_cuda_native_resources_v2(
        contract, include_reference_scratch=False
    )
    assert CUDA_NATIVE_LIFECYCLE_EVENT_ROW_BYTES_V2 == 48
    assert CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2 == 2 * 512
    assert CUDA_NATIVE_LIFECYCLE_EVENT_ROW_FIELDS_V2 == (
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
    assert CUDA_NATIVE_LIFECYCLE_EVENT_RESERVED_VALUE_V2 == 0
    assert lifecycle_dispatch_event_bytes_per_world_v2(8) == 2 * 512 * 8 * 48
    assert ledger.bounded_dispatch_event_bytes == 136 * 2 * 512 * 8 * 48
    assert no_reference.reference_scratch_bytes == 0
    assert no_reference.total_bytes < ledger.total_bytes
    assert (
        estimate_cuda_native_resources_v2(
            CudaFullEvolutionNativeContractV2(B=136, total_ticks=5_000, temporal_k=1)
        ).bounded_dispatch_event_bytes
        < ledger.bounded_dispatch_event_bytes
    )
    with pytest.raises(CudaFullEvolutionContractError):
        lifecycle_dispatch_event_bytes_per_world_v2(65)


@pytest.mark.parametrize(
    ("kind", "ordinal"),
    (
        (CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2, 0),
        (CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2, 511),
        (65535, (1 << 48) - 1),
    ),
)
def test_lifecycle_event_id_codec_is_exact(kind: int, ordinal: int) -> None:
    event_id = encode_lifecycle_event_id_v2(kind, ordinal)
    assert decode_lifecycle_event_id_v2(event_id) == (kind, ordinal)


@pytest.mark.parametrize(
    ("kind", "ordinal"),
    ((0, 0), (65536, 0), (1, -1), (1, 1 << 48), (True, 0), (1, False)),
)
def test_lifecycle_event_id_codec_rejects_out_of_range_values(
    kind: int, ordinal: int
) -> None:
    with pytest.raises(CudaFullEvolutionContractError):
        encode_lifecycle_event_id_v2(kind, ordinal)
    with pytest.raises(CudaFullEvolutionContractError):
        decode_lifecycle_event_id_v2(1)


@pytest.mark.parametrize(
    ("tick_offset", "event_kind", "subject_slot", "site_index"),
    [(0, 1, 0, 0), (63, 2, 511, 1023), (17, 1, 255, 513)],
)
def test_sparse_lifecycle_event_key_v3_round_trips(
    tick_offset: int, event_kind: int, subject_slot: int, site_index: int
) -> None:
    packed = encode_sparse_lifecycle_event_key_v3(
        tick_offset=tick_offset,
        event_kind=event_kind,
        subject_slot=subject_slot,
        site_index=site_index,
    )
    assert decode_sparse_lifecycle_event_key_v3(packed) == (
        tick_offset,
        event_kind,
        subject_slot,
        site_index,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tick_offset": 64, "event_kind": 1, "subject_slot": 0, "site_index": 0},
        {"tick_offset": 0, "event_kind": 0, "subject_slot": 0, "site_index": 0},
        {"tick_offset": 0, "event_kind": 3, "subject_slot": 0, "site_index": 0},
        {"tick_offset": 0, "event_kind": 1, "subject_slot": 512, "site_index": 0},
        {"tick_offset": 0, "event_kind": 1, "subject_slot": 0, "site_index": 1024},
    ],
)
def test_sparse_lifecycle_event_key_v3_rejects_out_of_range(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(CudaFullEvolutionContractError):
        encode_sparse_lifecycle_event_key_v3(**kwargs)
    with pytest.raises(CudaFullEvolutionContractError):
        decode_sparse_lifecycle_event_key_v3(1 << 31)


def test_sparse_lifecycle_event_envelope_v3_is_exact_and_checked() -> None:
    assert (
        sparse_lifecycle_event_envelope_bytes_v3(batch_size=896, event_count=307_825)
        == 9_857_576
    )
    with pytest.raises(CudaFullEvolutionContractError):
        sparse_lifecycle_event_envelope_bytes_v3(batch_size=0, event_count=1)


def test_batch_admission_never_silently_accepts_a_smaller_batch() -> None:
    contract = CudaFullEvolutionNativeContractV2(B=136, total_ticks=5_000)
    required = estimate_cuda_native_resources_v2(contract).total_bytes
    admission = select_cuda_native_batch_v2(contract, available_vram_bytes=required)
    assert not admission.admitted
    assert admission.partial_diagnostic_available
    assert 0 < admission.selected_batch_size < contract.B
    assert admission.requested_required_bytes == required


def test_batch_admission_boundaries_reserve_zero_and_production_owner() -> None:
    contract = CudaFullEvolutionNativeContractV2(B=136, total_ticks=5_000)
    reserve = 37_123
    required = estimate_cuda_native_resources_v2(contract).total_bytes
    minimum_usable = (required * 4 + 2) // 3
    exact = select_cuda_native_batch_v2(
        contract, available_vram_bytes=reserve + minimum_usable, reserve_bytes=reserve
    )
    assert exact.admitted and exact.selected_batch_size == contract.B
    assert exact.headroom_bytes * 4 >= minimum_usable
    below = select_cuda_native_batch_v2(
        contract,
        available_vram_bytes=reserve + minimum_usable - 1,
        reserve_bytes=reserve,
    )
    assert not below.admitted and below.partial_diagnostic_available
    zero = select_cuda_native_batch_v2(
        contract, available_vram_bytes=reserve, reserve_bytes=reserve
    )
    assert not zero.admitted and not zero.partial_diagnostic_available
    assert zero.selected_batch_size == zero.required_bytes == 0

    production_required = estimate_cuda_native_resources_v2(
        contract, include_reference_scratch=False
    ).total_bytes
    production_usable = (production_required * 4 + 2) // 3
    production = select_cuda_native_batch_v2(
        contract,
        available_vram_bytes=production_usable,
        include_reference_scratch=False,
    )
    assert production.admitted and production.selected_batch_size == contract.B
    assert production.requested_required_bytes == production_required
    with pytest.raises(CudaFullEvolutionContractError):
        select_cuda_native_batch_v2(contract, available_vram_bytes=1 << 63)
    with pytest.raises(CudaFullEvolutionContractError):
        select_cuda_native_batch_v2(
            contract, available_vram_bytes=0, reserve_bytes=1 << 63
        )


@pytest.mark.parametrize("temporal_k", (1, 8, 64))
def test_temporal_k_is_tunable_and_loci_remain_fixed(temporal_k: int) -> None:
    short = CudaFullEvolutionNativeContractV2(
        B=136, total_ticks=5_000, temporal_k=temporal_k
    )
    long = CudaFullEvolutionNativeContractV2(
        B=136, total_ticks=500_000, temporal_k=temporal_k
    )
    assert short.temporal_k == temporal_k
    assert short.linked_diploid_loci == LINKED_DIPLOID_LOCI_V2 == 128
    assert estimate_cuda_native_resources_v2(
        short
    ) == estimate_cuda_native_resources_v2(long)


def test_field_and_layout_validation_rejects_inexact_or_overflowed_shapes() -> None:
    with pytest.raises(CudaFullEvolutionContractError):
        CudaNativeFieldLayoutV2("bad", "f32", (1,), 0, 4)
    with pytest.raises(CudaFullEvolutionContractError):
        CudaNativeFieldLayoutV2("bad", "u8", (0,), 0, 1)
    with pytest.raises(CudaFullEvolutionContractError):
        CudaNativeFieldLayoutV2("bad", "u16", (2,), 0, 2)
    first = CudaNativeFieldLayoutV2("first", "u8", (1,), 0, 1)
    duplicate = CudaNativeFieldLayoutV2("first", "u8", (1,), 64, 1)
    with pytest.raises(CudaFullEvolutionContractError):
        CudaNativeStateLayoutV2((first, duplicate), 128, 0)
    with pytest.raises(CudaFullEvolutionContractError):
        CudaNativeStateLayoutV2((first,), 128, 0)
    layout = state_layout_v2()
    with pytest.raises(CudaFullEvolutionContractError):
        replace(layout, reference_scratch_bytes_per_world=(1 << 63))
    with pytest.raises(CudaFullEvolutionContractError):
        replace(layout, optimized_kernel_global_scratch_bytes=False)
    with pytest.raises(CudaFullEvolutionContractError):
        replace(layout, optimized_kernel_global_scratch_bytes=0.0)
    shifted = tuple(
        replace(field, offset_bytes=field.offset_bytes + 64) for field in layout.fields
    )
    with pytest.raises(CudaFullEvolutionContractError):
        replace(
            layout,
            fields=shifted,
            state_bytes_per_world=layout.state_bytes_per_world + 64,
        )
    with pytest.raises(CudaFullEvolutionContractError):
        estimate_cuda_native_resources_v2(
            CudaFullEvolutionNativeContractV2(B=(1 << 63) - 1, total_ticks=1)
        )
