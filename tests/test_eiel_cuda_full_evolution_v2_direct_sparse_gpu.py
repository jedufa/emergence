"""Hostile tiny-GPU gates for the direct-sparse Full Evolution V3 owner."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import pytest

_PREBUILT = os.environ.get("EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT")
_REQUIRE_GPU_GATE = os.environ.get("EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU") == "1"
try:
    import torch
except Exception:  # pragma: no cover - permits collection outside the CUDA env
    torch = None

pytestmark = pytest.mark.skipif(
    not _PREBUILT or torch is None or not torch.cuda.is_available(),
    reason="requires EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT and a physical CUDA device",
)

from engine.perf.cuda_full_evolution_v2.contract import (  # noqa: E402
    CudaFullEvolutionNativeContractV2,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (  # noqa: E402
    DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3,
    DIRECT_SPARSE_OVERFLOW_BIT_V3,
    CudaFullEvolutionDirectSparseOwnerV3,
    seal_direct_sparse_durable_sink_receipt_v3,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_sustained_benchmark import (  # noqa: E402
    _publish_prepared_dispatch,
)
from engine.perf.cuda_full_evolution_v2.layout import (  # noqa: E402
    CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2,
    CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
    decode_sparse_lifecycle_event_key_v3,
    encode_lifecycle_event_id_v2,
    encode_sparse_lifecycle_event_key_v3,
    state_layout_v2,
)
from engine.perf.cuda_full_evolution_v2.native_loader import (  # noqa: E402
    load_prebuilt_cuda_full_evolution_v2,
)
from engine.perf.cuda_full_evolution_v2.runtime import (  # noqa: E402
    FULL_CAPABILITY_MASK_V2,
    CudaFullEvolutionRuntimeV2,
    _canonical_sha256,
    _canonical_state_layout_sha256,
    _checkpoint_metadata_body,
    seal_event_sink_receipt_v2,
)

ARTIFACT = Path(_PREBUILT or "")
U64_MASK = (1 << 64) - 1
INVALID_OFFSETS_BIT = 1 << 0
INVALID_TICK_BIT = 1 << 3
INVALID_KIND_OR_ID_BIT = 1 << 4
INVALID_AGENT_ID_BIT = 1 << 5
INVALID_BIRTH_PARENTS_BIT = 1 << 7
INVALID_SITE_OR_RESERVED_BIT = 1 << 8
INVALID_ORDER_BIT = 1 << 12


if _REQUIRE_GPU_GATE:
    if not _PREBUILT:
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires "
            "EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT"
        )
    if not ARTIFACT.is_file():
        raise RuntimeError(f"required prebuilt artifact does not exist: {ARTIFACT}")
    manifest = ARTIFACT.with_suffix(ARTIFACT.suffix + ".manifest.json")
    if not manifest.is_file():
        raise RuntimeError(f"required prebuilt manifest does not exist: {manifest}")
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("required physical CUDA device is unavailable")


CAUSAL_WITNESS_FIELDS: dict[int, tuple[str, ...]] = {
    0: ("resource",),
    1: ("wdl_due", "wdl_active"),
    2: ("jungle",),
    3: ("eed_pending", "eed_impact"),
    4: ("occupancy", "agent_id", "parent_a_id", "parent_b_id"),
    5: ("policy_alleles", "trait_alleles"),
    6: ("policy_alleles", "trait_alleles"),
}


def _contract(
    total_ticks: int,
    *,
    temporal_k: int,
    start_tick: int = 0,
    batch: int = 1,
) -> CudaFullEvolutionNativeContractV2:
    return CudaFullEvolutionNativeContractV2(
        B=batch,
        total_ticks=total_ticks,
        temporal_k=temporal_k,
        start_next_tick=start_tick,
    )


def _ack_dense(owner: CudaFullEvolutionRuntimeV2, dispatch):
    receipt = seal_event_sink_receipt_v2(
        dispatch,
        sink_id="pytest-direct-sparse-dense-oracle",
        sink_receipt_id=(
            f"{dispatch.start_next_absolute_tick}-{dispatch.end_next_absolute_tick}-"
            f"{dispatch.content_sha256[:16]}"
        ),
    )
    owner.acknowledge_event_drain(receipt)
    return receipt


def _ack_direct(owner: CudaFullEvolutionDirectSparseOwnerV3, dispatch):
    sample_id = (
        f"pytest-{dispatch.start_next_absolute_tick}-"
        f"{dispatch.end_next_absolute_tick}-{dispatch.content_sha256[:16]}"
    )
    with TemporaryDirectory(prefix="eiel-direct-sparse-test-") as temporary:
        prepared = owner.prepare_sparse_event_durable_envelope_v3(
            dispatch, sample_id=sample_id
        )
        _, durable_envelope = _publish_prepared_dispatch(
            prepared,
            output_directory=Path(temporary),
            sample_id=sample_id,
        )
        try:
            receipt = seal_direct_sparse_durable_sink_receipt_v3(
                dispatch,
                sink_id="pytest-direct-sparse-exact-local-sink",
                sink_receipt_id=sample_id,
                sample_id=sample_id,
                durable_envelope=durable_envelope,
            )
            owner.acknowledge_sparse_event_drain_v3(
                receipt, durable_envelope=durable_envelope
            )
        finally:
            durable_envelope.close()
    return receipt


def _dense_records(dispatch) -> list[dict[str, int]]:
    rows = dispatch.event_rows.view(torch.int64).cpu().tolist()
    records: list[dict[str, int]] = []
    for world_index, world_rows in enumerate(rows):
        for row_index, row in enumerate(world_rows):
            tail = row[5] & U64_MASK
            status = (tail >> 32) & 0xFFFF
            if status == 0:
                continue
            event_kind = tail & 0xFFFF
            subject_slot = row[1] & ((1 << 48) - 1)
            records.append(
                {
                    "world_uid": int(dispatch.world_uids[world_index]),
                    "absolute_tick": row[0] & U64_MASK,
                    "event_id": row[1] & U64_MASK,
                    "agent_id": row[2] & U64_MASK,
                    "parent_a_id": row[3] & U64_MASK,
                    "parent_b_id": row[4] & U64_MASK,
                    "event_kind": event_kind,
                    "subject_slot": subject_slot,
                    "site_index": (tail >> 16) & 0xFFFF,
                    "status": status,
                    "reserved": (tail >> 48) & 0xFFFF,
                    "row_index": row_index,
                }
            )
    return records


def _sparse_records(dispatch) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    row_bytes = CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3
    for world_index, world_uid in enumerate(dispatch.world_uids):
        begin = dispatch.world_offsets[world_index]
        end = dispatch.world_offsets[world_index + 1]
        for packed_index in range(begin, end):
            offset = packed_index * row_bytes
            words = tuple(
                int.from_bytes(
                    dispatch.packed_events[offset + word * 8 : offset + (word + 1) * 8],
                    byteorder="little",
                    signed=False,
                )
                for word in range(4)
            )
            tick_offset, event_kind, subject_slot, site_index = (
                decode_sparse_lifecycle_event_key_v3(words[3])
            )
            records.append(
                {
                    "world_uid": world_uid,
                    "absolute_tick": dispatch.start_next_absolute_tick + tick_offset,
                    "event_id": encode_lifecycle_event_id_v2(event_kind, subject_slot),
                    "agent_id": words[0],
                    "parent_a_id": words[1],
                    "parent_b_id": words[2],
                    "event_kind": event_kind,
                    "subject_slot": subject_slot,
                    "site_index": site_index,
                    "status": 1,
                    "reserved": 0,
                    "row_index": (
                        tick_offset * 1024 + (event_kind - 1) * 512 + subject_slot
                    ),
                }
            )
    return records


def _assert_canonical(records: list[dict[str, int]]) -> None:
    keys = [
        (
            record["world_uid"],
            record["absolute_tick"],
            record["event_kind"],
            record["subject_slot"],
        )
        for record in records
    ]
    assert keys == sorted(keys)
    for key in {(row["world_uid"], row["absolute_tick"]) for row in records}:
        tick_rows = [
            row for row in records if (row["world_uid"], row["absolute_tick"]) == key
        ]
        kinds = [row["event_kind"] for row in tick_rows]
        assert kinds == sorted(kinds)
        for kind in (
            CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2,
            CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2,
        ):
            slots = [
                row["subject_slot"] for row in tick_rows if row["event_kind"] == kind
            ]
            assert slots == sorted(slots)
            assert len(slots) == len(set(slots))


def _reseal_checkpoint(checkpoint, canonical_state: bytes):
    content_sha256 = hashlib.sha256(canonical_state).hexdigest()
    shape = checkpoint.canonical_state_shape
    provisional = replace(
        checkpoint,
        canonical_state=canonical_state,
        canonical_state_nbytes=len(canonical_state),
        canonical_state_sha256=content_sha256,
        canonical_state_layout_sha256=_canonical_state_layout_sha256(
            shape=shape,
            nbytes=len(canonical_state),
            content_sha256=content_sha256,
        ),
        metadata_seal_sha256="",
    )
    return replace(
        provisional,
        metadata_seal_sha256=_canonical_sha256(_checkpoint_metadata_body(provisional)),
    )


def _add_tick_63_death_sentinel(checkpoint):
    """Force slot 511 to die by lifespan on dispatch tick offset 63."""
    payload = bytearray(checkpoint.canonical_state)
    layout = state_layout_v2()

    def pack_field(field_name: str, slot: int, fmt: str, value: int) -> None:
        field = layout.field(field_name)
        struct.pack_into(
            fmt, payload, field.offset_bytes + struct.calcsize(fmt) * slot, value
        )

    # Slot zero is an unconditional tick-zero death sentinel. Slot 511 has
    # lifespan 384 and reaches age 385 on the 64th tick.
    pack_field("energy", 0, "<h", 30_000)
    pack_field("health", 0, "<h", 30_000)
    pack_field("age", 0, "<H", 65_535)
    pack_field("energy", 511, "<h", 30_000)
    pack_field("health", 511, "<h", 30_000)
    pack_field("age", 511, "<H", 321)
    traits = layout.field("trait_alleles")
    for allele in range(2):
        index = (allele * 16 + 6) * 512 + 511
        struct.pack_into("<h", payload, traits.offset_bytes + index * 2, 0)
    return _reseal_checkpoint(checkpoint, bytes(payload))


def _forced_c512_checkpoint(*, total_ticks: int, temporal_k: int, start_tick: int = 0):
    source = CudaFullEvolutionRuntimeV2.initialize(
        _contract(total_ticks, temporal_k=temporal_k, start_tick=start_tick),
        seed=0x5A17,
        artifact_path=ARTIFACT,
        first_world_uid=71_001,
        execution_mode="eager_monolith",
    )
    source.force_full_capacity_benchmark_fixture_v2()
    source.force_test_conditions()
    checkpoint = source.checkpoint()
    source.close()
    if temporal_k == 64:
        checkpoint = _add_tick_63_death_sentinel(checkpoint)
    return checkpoint


def _direct_owner(
    checkpoint,
    *,
    capacity: int,
    maximum_capacity: int | None = None,
) -> CudaFullEvolutionDirectSparseOwnerV3:
    return CudaFullEvolutionDirectSparseOwnerV3.from_checkpoint(
        checkpoint,
        artifact_path=ARTIFACT,
        initial_capacity_rows=capacity,
        maximum_capacity_rows=(
            capacity if maximum_capacity is None else maximum_capacity
        ),
    )


@pytest.mark.parametrize("ticks", (1, 8, 64))
def test_direct_sparse_c512_exact_dense_oracle_and_canonical_order(ticks: int) -> None:
    start_tick = 91_000
    checkpoint = _forced_c512_checkpoint(
        total_ticks=ticks, temporal_k=ticks, start_tick=start_tick
    )
    dense = CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint, artifact_path=ARTIFACT, execution_mode="eager_monolith"
    )
    direct = _direct_owner(checkpoint, capacity=1024 * ticks)

    dense.advance(ticks)
    dense_dispatch = dense.drain_events(to_cpu=True)
    direct_dispatch = direct.advance_and_drain_v3(ticks)
    dense_records = _dense_records(dense_dispatch)
    direct_records = _sparse_records(direct_dispatch)

    assert direct_dispatch.event_counts == tuple(
        int(value) for value in dense_dispatch.event_counts.tolist()
    )
    assert direct_records == dense_records
    _assert_canonical(direct_records)
    assert all(row["reserved"] == 0 and row["status"] == 1 for row in direct_records)
    assert any(
        row["absolute_tick"] == start_tick
        and row["event_kind"] == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2
        for row in direct_records
    )
    assert any(
        row["absolute_tick"] == start_tick
        and row["event_kind"] == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2
        for row in direct_records
    )
    if ticks == 64:
        assert any(
            row["absolute_tick"] == start_tick + 63
            and row["event_kind"] == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2
            and row["subject_slot"] == 511
            for row in direct_records
        )

    _ack_dense(dense, dense_dispatch)
    _ack_direct(direct, direct_dispatch)
    assert torch.equal(dense.state, direct.state_projection())
    assert dense.counters() == direct.counters()
    assert all(value > 0 for value in direct.counters()[0].values())


def test_direct_sparse_c512_is_byte_deterministic_and_receipt_is_immutable() -> None:
    checkpoint = _forced_c512_checkpoint(total_ticks=8, temporal_k=8)
    first = _direct_owner(checkpoint, capacity=8192)
    second = _direct_owner(checkpoint, capacity=8192)

    first_dispatch = first.advance_and_drain_v3(8)
    second_dispatch = second.advance_and_drain_v3(8)
    assert first_dispatch.packed_events == second_dispatch.packed_events
    assert first_dispatch.world_offsets == second_dispatch.world_offsets
    assert first_dispatch.event_counts == second_dispatch.event_counts
    assert first_dispatch.final_next_agent_ids == second_dispatch.final_next_agent_ids
    assert first_dispatch.content_sha256 == second_dispatch.content_sha256
    _ack_direct(first, first_dispatch)
    _ack_direct(second, second_dispatch)
    assert torch.equal(first.state_projection(), second.state_projection())

    projected = first.execution_receipt_v3()
    assert projected is not None
    original_pointer = projected["pointers"]["state"]
    original_required = projected["resource_admission"]["required_free_bytes"]
    projected["pointers"]["state"] = 0
    projected["resource_admission"]["required_free_bytes"] = 0
    projected_again = first.execution_receipt_v3()
    assert projected_again is not None
    assert projected_again["pointers"]["state"] == original_pointer
    assert (
        projected_again["resource_admission"]["required_free_bytes"]
        == original_required
    )


def test_direct_sparse_overflow_rolls_back_grows_and_retries_exactly() -> None:
    checkpoint = _forced_c512_checkpoint(total_ticks=1, temporal_k=1)
    retrying = _direct_owner(
        checkpoint,
        capacity=1,
        maximum_capacity=DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3,
    )
    sufficient = _direct_owner(checkpoint, capacity=1024)

    retry_dispatch = retrying.advance_and_drain_v3(1)
    sufficient_dispatch = sufficient.advance_and_drain_v3(1)
    assert retry_dispatch.packed_events == sufficient_dispatch.packed_events
    assert retry_dispatch.world_offsets == sufficient_dispatch.world_offsets
    assert retry_dispatch.event_counts == sufficient_dispatch.event_counts
    assert (
        retry_dispatch.final_next_agent_ids == sufficient_dispatch.final_next_agent_ids
    )
    assert retry_dispatch.content_sha256 == sufficient_dispatch.content_sha256
    metrics = retrying.dispatch_metrics_v3
    assert metrics is not None
    assert metrics.retry_count >= 1
    assert metrics.rollback_elapsed_seconds >= 0.0
    assert metrics.selected_capacity_rows_per_world >= max(retry_dispatch.event_counts)
    receipt = retrying.execution_receipt_v3()
    assert receipt is not None
    assert receipt["retry_count"] == metrics.retry_count
    assert (
        receipt["rollback_snapshot_full_state_bytes"]
        == state_layout_v2().state_bytes_per_world
    )

    _ack_direct(retrying, retry_dispatch)
    _ack_direct(sufficient, sufficient_dispatch)
    assert torch.equal(retrying.state_projection(), sufficient.state_projection())
    assert retrying.counters() == sufficient.counters()


def test_direct_sparse_checkpoint_restore_continuation_is_exact() -> None:
    checkpoint = _forced_c512_checkpoint(total_ticks=16, temporal_k=8)
    uninterrupted = _direct_owner(checkpoint, capacity=8192)
    resumable = _direct_owner(checkpoint, capacity=8192)

    uninterrupted_records: list[dict[str, int]] = []
    uninterrupted_dispatches = []
    for _ in range(2):
        dispatch = uninterrupted.advance_and_drain_v3(8)
        uninterrupted_records.extend(_sparse_records(dispatch))
        uninterrupted_dispatches.append(dispatch)
        _ack_direct(uninterrupted, dispatch)

    first = resumable.advance_and_drain_v3(8)
    resumed_records = _sparse_records(first)
    _ack_direct(resumable, first)
    resumed_checkpoint = resumable.checkpoint()
    restored = CudaFullEvolutionDirectSparseOwnerV3.from_checkpoint(
        resumed_checkpoint,
        artifact_path=ARTIFACT,
        initial_capacity_rows=8192,
        maximum_capacity_rows=8192,
    )
    second = restored.advance_and_drain_v3(8)
    resumed_records.extend(_sparse_records(second))
    _ack_direct(restored, second)

    assert resumed_records == uninterrupted_records
    assert second.packed_events == uninterrupted_dispatches[1].packed_events
    assert second.content_sha256 == uninterrupted_dispatches[1].content_sha256
    assert torch.equal(restored.state_projection(), uninterrupted.state_projection())
    assert restored.counters() == uninterrupted.counters()
    assert restored.committed_next_absolute_tick == 16


def _raw_fixture(*, capacity: int = 4):
    loaded = load_prebuilt_cuda_full_evolution_v2(ARTIFACT)
    layout = state_layout_v2()
    state = torch.empty(
        (1, layout.state_bytes_per_world), dtype=torch.uint8, device="cuda"
    )
    loaded.module.initialize(
        state, torch.tensor([81_001], dtype=torch.int64, device="cuda"), 7331, 0
    )
    keys = (
        encode_sparse_lifecycle_event_key_v3(
            tick_offset=0, event_kind=1, subject_slot=1, site_index=1
        ),
        encode_sparse_lifecycle_event_key_v3(
            tick_offset=0, event_kind=1, subject_slot=2, site_index=2
        ),
    )
    host_words = [[[0, 0, 0, 0] for _ in range(capacity)]]
    host_words[0][0] = [1, 0, 0, keys[0]]
    host_words[0][1] = [2, 0, 0, keys[1]]
    direct_events = (
        torch.tensor(host_words, dtype=torch.int64, device="cuda")
        .view(torch.uint8)
        .reshape(1, capacity, CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3)
    )
    event_counts = torch.tensor([2], dtype=torch.int32, device="cuda")
    advance_validation = torch.zeros(1, dtype=torch.int32, device="cuda")
    return loaded, state, direct_events, event_counts, advance_validation


def _known_good_cuda_probe() -> None:
    probe = torch.arange(17, dtype=torch.int32, device="cuda")
    assert int(probe.sum().item()) == sum(range(17))


def _pack_with_guards(
    loaded,
    state,
    direct_events,
    event_counts,
    advance_validation,
    *,
    offsets: tuple[int, int],
    output_rows: int,
    tick_count: int = 1,
    capacity: int = 4,
):
    offsets_device = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    backing = torch.full(
        (output_rows + 2, CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    packed = backing[1 : output_rows + 1]
    validation = torch.zeros(1, dtype=torch.int32, device="cuda")
    loaded.module.pack_direct_sparse_events_v3(
        state,
        direct_events,
        event_counts,
        advance_validation,
        offsets_device,
        packed,
        validation,
        tick_count,
        capacity,
    )
    bits = int(validation.cpu().item())
    assert bool(torch.all(backing[0] == 0xA5).item())
    assert bool(torch.all(backing[-1] == 0xA5).item())
    _known_good_cuda_probe()
    return bits, packed.cpu()


@pytest.mark.parametrize(
    ("offsets", "output_rows"),
    (
        ((1, 3), 3),
        ((0, 2), 3),
        ((0, 3), 2),
        ((2, 1), 2),
    ),
)
def test_native_pack_rejects_hostile_offset_endpoints_without_oob(
    offsets: tuple[int, int], output_rows: int
) -> None:
    fixture = _raw_fixture()
    bits, _ = _pack_with_guards(*fixture, offsets=offsets, output_rows=output_rows)
    assert bits & INVALID_OFFSETS_BIT


@pytest.mark.parametrize(
    ("declared_count", "offsets", "output_rows"),
    ((-1, (0, 0), 1), (5, (0, 5), 5)),
)
def test_native_pack_rejects_hostile_count_capacity_without_oob(
    declared_count: int, offsets: tuple[int, int], output_rows: int
) -> None:
    loaded, state, direct_events, event_counts, advance_validation = _raw_fixture()
    event_counts.fill_(declared_count)
    bits, _ = _pack_with_guards(
        loaded,
        state,
        direct_events,
        event_counts,
        advance_validation,
        offsets=offsets,
        output_rows=output_rows,
    )
    assert bits & INVALID_OFFSETS_BIT

    with pytest.raises(RuntimeError, match="capacity"):
        loaded.module.pack_direct_sparse_events_v3(
            state,
            direct_events,
            event_counts,
            advance_validation,
            torch.tensor([0, 0], dtype=torch.int64, device="cuda"),
            torch.empty((0, 32), dtype=torch.uint8, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            1,
            3,
        )
    _known_good_cuda_probe()


@pytest.mark.parametrize(
    ("corruption", "expected_bit"),
    (
        ("reserved", INVALID_SITE_OR_RESERVED_BIT),
        ("tick", INVALID_TICK_BIT),
        ("kind", INVALID_KIND_OR_ID_BIT),
        ("agent_zero", INVALID_AGENT_ID_BIT),
        ("birth_without_parents", INVALID_BIRTH_PARENTS_BIT),
        ("order", INVALID_ORDER_BIT),
    ),
)
def test_native_pack_rejects_reserved_key_identity_and_order_without_publication(
    corruption: str, expected_bit: int
) -> None:
    loaded, state, direct_events, event_counts, advance_validation = _raw_fixture()
    words = direct_events.view(torch.int64)
    if corruption == "reserved":
        words[0, 0, 3] |= 1 << 27
    elif corruption == "tick":
        words[0, 0, 3] = encode_sparse_lifecycle_event_key_v3(
            tick_offset=1, event_kind=1, subject_slot=1, site_index=1
        )
    elif corruption == "kind":
        words[0, 0, 3] = (3 << 6) | (1 << 8) | (1 << 17)
    elif corruption == "agent_zero":
        words[0, 0, 0] = 0
    elif corruption == "birth_without_parents":
        words[0, 0, 3] = encode_sparse_lifecycle_event_key_v3(
            tick_offset=0, event_kind=2, subject_slot=1, site_index=1
        )
    elif corruption == "order":
        first = words[0, 0].clone()
        words[0, 0].copy_(words[0, 1])
        words[0, 1].copy_(first)
    else:  # pragma: no cover - protects the parametrized test table
        raise AssertionError(corruption)

    bits, _ = _pack_with_guards(
        loaded,
        state,
        direct_events,
        event_counts,
        advance_validation,
        offsets=(0, 2),
        output_rows=2,
    )
    assert bits & expected_bit


def test_native_pack_propagates_failed_advance_and_never_authorizes_publication() -> (
    None
):
    loaded, state, direct_events, event_counts, advance_validation = _raw_fixture()
    advance_validation.fill_(DIRECT_SPARSE_OVERFLOW_BIT_V3)
    bits, _ = _pack_with_guards(
        loaded,
        state,
        direct_events,
        event_counts,
        advance_validation,
        offsets=(0, 2),
        output_rows=2,
    )
    assert bits & DIRECT_SPARSE_OVERFLOW_BIT_V3


def test_native_direct_sparse_rejects_unsafe_aliases_before_launch() -> None:
    loaded, state, direct_events, event_counts, advance_validation = _raw_fixture()
    state_before = state.clone()
    with pytest.raises(RuntimeError, match="must not overlap"):
        loaded.module.advance_direct_sparse_v3(
            state,
            direct_events,
            event_counts,
            event_counts,
            1,
            4,
            FULL_CAPABILITY_MASK_V2,
        )
    assert torch.equal(state, state_before)

    offsets = torch.tensor([0, 4], dtype=torch.int64, device="cuda")
    event_counts.fill_(4)
    packed_alias = direct_events.reshape(4, 32)
    validation = torch.zeros_like(advance_validation)
    with pytest.raises(RuntimeError, match="must not overlap"):
        loaded.module.pack_direct_sparse_events_v3(
            state,
            direct_events,
            event_counts,
            advance_validation,
            offsets,
            packed_alias,
            validation,
            1,
            4,
        )
    with pytest.raises(RuntimeError, match="must not overlap"):
        loaded.module.pack_direct_sparse_events_v3(
            state,
            direct_events,
            event_counts,
            advance_validation,
            offsets,
            torch.empty((4, 32), dtype=torch.uint8, device="cuda"),
            advance_validation,
            1,
            4,
        )
    assert torch.equal(state, state_before)
    _known_good_cuda_probe()


@pytest.mark.parametrize("capability_bit", range(7))
def test_native_direct_sparse_all_seven_masks_have_positive_causal_witnesses(
    capability_bit: int,
) -> None:
    loaded = load_prebuilt_cuda_full_evolution_v2(ARTIFACT)
    layout = state_layout_v2()
    base = torch.empty(
        (1, layout.state_bytes_per_world), dtype=torch.uint8, device="cuda"
    )
    loaded.module.initialize(
        base, torch.tensor([91_001], dtype=torch.int64, device="cuda"), 991, 0
    )
    loaded.module.force_test_conditions(base)
    full = base.clone()
    masked = base.clone()

    def advance(state, mask: int) -> None:
        direct_events = torch.empty((1, 1024, 32), dtype=torch.uint8, device="cuda")
        counts = torch.zeros(1, dtype=torch.int32, device="cuda")
        validation = torch.zeros_like(counts)
        loaded.module.advance_direct_sparse_v3(
            state, direct_events, counts, validation, 1, 1024, mask
        )
        assert int(validation.cpu().item()) == 0
        assert 0 <= int(counts.cpu().item()) <= 1024

    advance(full, FULL_CAPABILITY_MASK_V2)
    advance(masked, FULL_CAPABILITY_MASK_V2 & ~(1 << capability_bit))
    full_header = full[:, :192].view(torch.int64).cpu()
    masked_header = masked[:, :192].view(torch.int64).cpu()
    assert int(full_header[0, 6 + capability_bit]) > 0
    assert int(masked_header[0, 6 + capability_bit]) == 0
    assert any(
        not torch.equal(
            full[
                :,
                layout.field(field_name).offset_bytes : (
                    layout.field(field_name).offset_bytes
                    + layout.field(field_name).bytes
                ),
            ],
            masked[
                :,
                layout.field(field_name).offset_bytes : (
                    layout.field(field_name).offset_bytes
                    + layout.field(field_name).bytes
                ),
            ],
        )
        for field_name in CAUSAL_WITNESS_FIELDS[capability_bit]
    )
    _known_good_cuda_probe()
