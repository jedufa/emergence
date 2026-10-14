"""Tiny physical-GPU gates for the prebuilt CUDA Native Full Evolution V2 slice."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

_PREBUILT = os.environ.get("EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT")
_REQUIRE_GPU_GATE = os.environ.get("EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU") == "1"
try:
    import torch
except Exception:  # pragma: no cover - clean collection without the CUDA environment
    torch = None

pytestmark = pytest.mark.skipif(
    not _PREBUILT or torch is None or not torch.cuda.is_available(),
    reason="requires EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT and a physical CUDA device",
)


from engine.perf.cuda_full_evolution_v2.contract import (  # noqa: E402
    MAX_SIGNED_INT64,
    MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2,
    MAX_TOTAL_TICKS_COUNTER_SAFE_V2,
    CudaFullEvolutionNativeContractV2,
)
from engine.perf.cuda_full_evolution_v2.layout import (  # noqa: E402
    CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2,
    CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2,
    CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3,
    decode_sparse_lifecycle_event_key_v3,
    encode_lifecycle_event_id_v2,
    state_layout_v2,
)
from engine.perf.cuda_full_evolution_v2.native_loader import (  # noqa: E402
    CudaFullEvolutionNativeLoadError,
    expected_layout_manifest_v2,
    load_prebuilt_cuda_full_evolution_v2,
)
from engine.perf.cuda_full_evolution_v2.runtime import (  # noqa: E402
    FULL_CAPABILITY_MASK_V2,
    FULL_CAPACITY_BENCHMARK_FIXTURE_V2,
    CudaFullEvolutionRuntimeError,
    CudaFullEvolutionRuntimeV2,
    _tensor_bytes,
    seal_event_sink_receipt_v2,
    seal_sparse_event_sink_receipt_v3,
)
from engine.campaigns.acs_native_detached_slots_v1 import (  # noqa: E402
    ACSNativeDetachedSlotsV1,
)
from engine.campaigns.cuda_full_evolution_v2_acs_adapter import (  # noqa: E402
    CudaFullEvolutionV2CapturePoolV1,
    capture_native_resident_checkpoint_to_file_v1,
    reopen_native_checkpoint_bundle_v1,
)

ARTIFACT = Path(_PREBUILT or "")
U64_MASK = (1 << 64) - 1


if _REQUIRE_GPU_GATE:
    if not _PREBUILT:
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires "
            "EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT"
        )
    if not ARTIFACT.is_file():
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires an existing "
            f"prebuilt artifact: {ARTIFACT}"
        )
    manifest = ARTIFACT.with_suffix(ARTIFACT.suffix + ".manifest.json")
    if not manifest.is_file():
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires the artifact "
            f"manifest: {manifest}"
        )
    if torch is None:
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires importable torch"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires a physical CUDA device"
        )


CAUSAL_WITNESS_FIELDS_V2: dict[str, tuple[str, ...]] = {
    "patch": ("resource",),
    "wdl_delayed_hazard": ("wdl_due", "wdl_active"),
    "jungle": ("jungle",),
    "eed": ("eed_pending", "eed_impact"),
    "genome_v2_ordinary_birth": (
        "occupancy",
        "agent_id",
        "parent_a_id",
        "parent_b_id",
        "absolute_birth_tick",
    ),
    "genome_v2_mutation": (
        "policy_alleles",
        "trait_alleles",
        "expressed_policy_cache",
        "expressed_trait_cache",
    ),
    "recombination_profile": (
        "policy_alleles",
        "trait_alleles",
        "expressed_policy_cache",
        "expressed_trait_cache",
    ),
}


def _contract(
    total_ticks: int,
    *,
    batch: int = 1,
    temporal_k: int = 8,
    start_tick: int = 0,
) -> CudaFullEvolutionNativeContractV2:
    return CudaFullEvolutionNativeContractV2(
        B=batch,
        total_ticks=total_ticks,
        temporal_k=temporal_k,
        start_next_tick=start_tick,
    )


def _owner(
    total_ticks: int,
    *,
    batch: int = 1,
    temporal_k: int = 8,
    start_tick: int = 0,
    seed: int = 71,
    reference: bool = False,
    world_uids: tuple[int, ...] | None = None,
    execution_mode: str = "eager_monolith",
) -> CudaFullEvolutionRuntimeV2:
    return CudaFullEvolutionRuntimeV2.initialize(
        _contract(
            total_ticks,
            batch=batch,
            temporal_k=temporal_k,
            start_tick=start_tick,
        ),
        seed=seed,
        artifact_path=ARTIFACT,
        enable_reference=reference,
        world_uids=world_uids,
        first_world_uid=9001,
        execution_mode=execution_mode,
    )


def _assert_owner_exact(
    first: CudaFullEvolutionRuntimeV2,
    second: CudaFullEvolutionRuntimeV2,
) -> None:
    assert torch.equal(first.state, second.state)
    assert torch.equal(first.event_rows, second.event_rows)
    assert torch.equal(first.event_counts, second.event_counts)


def _drain_and_ack(owner: CudaFullEvolutionRuntimeV2, *, to_cpu: bool = True):
    dispatch = owner.drain_events(to_cpu=to_cpu)
    assert dispatch.drain_elapsed_seconds >= 0.0
    assert dispatch.event_rows.is_cuda is (not to_cpu)
    assert dispatch.event_counts.is_cuda is (not to_cpu)
    receipt = seal_event_sink_receipt_v2(
        dispatch,
        sink_id="pytest-exact-content-sink",
        sink_receipt_id=(
            f"{dispatch.start_next_absolute_tick}-{dispatch.end_next_absolute_tick}-"
            f"{dispatch.content_sha256[:16]}"
        ),
    )
    owner.acknowledge_event_drain(receipt)
    assert owner.last_sink_receipt_sha256 == receipt.receipt_sha256
    return dispatch


def _drain_sparse_and_ack(owner: CudaFullEvolutionRuntimeV2):
    dispatch = owner.drain_events_sparse_v3()
    assert dispatch.drain_elapsed_seconds >= 0.0
    assert dispatch.event_count == sum(dispatch.event_counts)
    assert dispatch.wire_nbytes == len(dispatch.packed_events) + 8 * (
        len(dispatch.event_counts) + 1
    )
    receipt = seal_sparse_event_sink_receipt_v3(
        dispatch,
        sink_id="pytest-exact-sparse-content-sink",
        sink_receipt_id=(
            f"{dispatch.start_next_absolute_tick}-{dispatch.end_next_absolute_tick}-"
            f"{dispatch.content_sha256[:16]}"
        ),
    )
    owner.acknowledge_sparse_event_drain_v3(receipt)
    assert owner.last_sink_receipt_sha256 == receipt.receipt_sha256
    return dispatch


def _event_words(dispatch) -> list[list[int]]:
    return dispatch.event_rows[0].view(torch.int64).cpu().tolist()


def _active_event_rows(dispatch):
    rows = dispatch.tick_count * CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2
    return dispatch.event_rows[:, :rows]


def _field_bytes(owner: CudaFullEvolutionRuntimeV2, field_name: str, *, world: int = 0):
    field = state_layout_v2().field(field_name)
    return (
        owner.state[world, field.offset_bytes : field.offset_bytes + field.bytes]
        .detach()
        .clone()
    )


def _committed_event_records(dispatch) -> list[dict[str, int]]:
    """Decode every committed row with its complete dispatch identity/payload."""
    words = dispatch.event_rows.view(torch.int64).cpu().tolist()
    records: list[dict[str, int]] = []
    for world_index, world_rows in enumerate(words):
        for row_index, row in enumerate(world_rows):
            tail = row[5] & U64_MASK
            status = (tail >> 32) & 0xFFFF
            if status == 0:
                continue
            records.append(
                {
                    "world_uid": int(dispatch.world_uids[world_index]),
                    "absolute_tick": row[0] & U64_MASK,
                    "event_id": row[1] & U64_MASK,
                    "agent_id": row[2] & U64_MASK,
                    "parent_a_id": row[3] & U64_MASK,
                    "parent_b_id": row[4] & U64_MASK,
                    "event_kind": tail & 0xFFFF,
                    "site_index": (tail >> 16) & 0xFFFF,
                    "status": status,
                    "reserved": (tail >> 48) & 0xFFFF,
                    "row_index": row_index,
                }
            )
    return records


def _sparse_event_records(dispatch) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    row_bytes = CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3
    for world_index, world_uid in enumerate(dispatch.world_uids):
        for packed_index in range(
            dispatch.world_offsets[world_index],
            dispatch.world_offsets[world_index + 1],
        ):
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
                    "site_index": site_index,
                    "status": 1,
                    "reserved": 0,
                    "row_index": (
                        tick_offset
                        * CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2
                        + (event_kind - 1) * 512
                        + subject_slot
                    ),
                }
            )
    return records


def test_sparse_v3_compaction_is_exactly_equivalent_to_dense_gpu_oracle() -> None:
    dense_owner = _owner(3, batch=2, temporal_k=3)
    dense_owner.force_test_conditions()
    sparse_owner = dense_owner.fork()
    dense_owner.advance(3)
    sparse_owner.advance(3)

    dense_dispatch = _drain_and_ack(dense_owner)
    sparse_dispatch = _drain_sparse_and_ack(sparse_owner)

    assert sparse_dispatch.event_counts == tuple(
        int(value) for value in dense_dispatch.event_counts.tolist()
    )
    assert _sparse_event_records(sparse_dispatch) == _committed_event_records(
        dense_dispatch
    )


def test_sparse_v3_repeat_is_byte_deterministic() -> None:
    first = _owner(4, batch=2, temporal_k=4, seed=303)
    second = _owner(4, batch=2, temporal_k=4, seed=303)
    first.force_test_conditions()
    second.force_test_conditions()
    first.advance(4)
    second.advance(4)

    first_dispatch = _drain_sparse_and_ack(first)
    second_dispatch = _drain_sparse_and_ack(second)

    assert first_dispatch.packed_events == second_dispatch.packed_events
    assert first_dispatch.world_offsets == second_dispatch.world_offsets
    assert first_dispatch.event_counts == second_dispatch.event_counts
    assert first_dispatch.content_sha256 == second_dispatch.content_sha256
    assert torch.equal(first.state, second.state)


def test_sparse_v3_ack_checkpoint_restore_continuation_is_exact() -> None:
    uninterrupted = _owner(4, batch=2, temporal_k=2, seed=404)
    resumable = _owner(4, batch=2, temporal_k=2, seed=404)
    uninterrupted.force_test_conditions()
    resumable.force_test_conditions()

    uninterrupted_records: list[dict[str, int]] = []
    for _ in range(2):
        uninterrupted.advance(2)
        uninterrupted_records.extend(
            _sparse_event_records(_drain_sparse_and_ack(uninterrupted))
        )

    resumable.advance(2)
    resumed_records = _sparse_event_records(_drain_sparse_and_ack(resumable))
    checkpoint = resumable.checkpoint()
    restored = CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint,
        artifact_path=ARTIFACT,
    )
    restored.advance(2)
    resumed_records.extend(_sparse_event_records(_drain_sparse_and_ack(restored)))

    assert resumed_records == uninterrupted_records
    assert torch.equal(restored.state, uninterrupted.state)
    assert restored.committed_next_absolute_tick == 4


def test_native_layout_manifest_equals_final_host_layout() -> None:
    loaded = load_prebuilt_cuda_full_evolution_v2(ARTIFACT)
    expected = expected_layout_manifest_v2()
    assert loaded.module.layout_manifest() == expected
    assert expected["state_bytes_per_world"] == 265_472
    assert expected["canonical_checkpoint_bytes_per_world"] == 191_744
    assert expected["derived_cache_bytes_per_world"] == 73_728
    assert expected["reference_scratch_bytes_per_world"] == 8_192
    assert expected["optimized_kernel_global_scratch_bytes"] == 0
    assert expected["counter_safety"] == {
        "max_capability_counter_increment_per_tick": (
            MAX_CAPABILITY_COUNTER_INCREMENT_PER_TICK_V2
        ),
        "max_total_ticks": MAX_TOTAL_TICKS_COUNTER_SAFE_V2,
        "counter_dtype": "signed_int64_projection_of_u64_header",
    }
    assert loaded.manifest["compiled_from_sealed_snapshot"] is True
    assert loaded.manifest["sealed_snapshot"] == {
        "files_mode": "0444",
        "directories_mode": "0555",
        "post_compile_inventory_verified": True,
    }
    assert expected["rng_address"] == {
        "fields": [
            "rng_key",
            "rng_semantic_epoch",
            "world_uid",
            "absolute_tick",
            "domain",
            "subject",
            "operation",
            "draw",
        ],
        "semantic_epoch": 3,
        "generator": "philox4x32_10",
        "composition": "arx_folded_full_u64_counter_address_v3",
        "arithmetic": "integer_only",
        "output_width_bits": 32,
        "site_bundle_domain": "site_environment",
        "site_bundle_subject": "canonical_site_index",
        "site_bundle_outputs": [
            "patch_carrying",
            "patch_growth",
            "wdl_schedule",
            "jungle_transition",
        ],
        "child_genome_bundle": {
            "persistent_subject": "child_biological_uid",
            "recombination": (
                "one_philox_block_per_linked_block_parent_uids_in_operation_draw"
            ),
            "mutation": "one_philox_block_per_linked_locus_two_allele_outputs",
            "forbidden_subject": "canonical_agent_slot",
        },
        "forbidden_inputs": [
            "agent_slot",
            "batch_index",
            "cta",
            "warp",
            "lane",
            "launch_order",
        ],
    }


def test_acs2_direct_resident_two_slot_fresh_owner_continuation(
    tmp_path: Path,
) -> None:
    """Tiny real-CUDA proof of direct capture, FIFO publication, and resume."""

    pool = CudaFullEvolutionV2CapturePoolV1(
        chunk_bytes=64 * 1024,
        has_cuda_fields=True,
        require_native_pool=True,
    )
    slots = ACSNativeDetachedSlotsV1(slots=2)
    owner = _owner(4, batch=2, temporal_k=2, seed=808)

    owner.advance(2)
    _drain_and_ack(owner)
    synchronous_two = owner.checkpoint()
    token_two = slots.begin_capture(compute_frontier=2)
    capture_two = capture_native_resident_checkpoint_to_file_v1(
        owner.checkpoint_capture_source_v1(),
        tmp_path / "checkpoint-tick-000000000002",
        artifact_path=ARTIFACT,
        checkpoint_at_tick=2,
        chunk_bytes=64 * 1024,
        require_native_pool=True,
        capture_pool=pool,
    )
    slots.submit_capture(token_two, capture_two)

    owner.advance(2)
    original_continuation = _drain_and_ack(owner)
    synchronous_four = owner.checkpoint()
    token_four = slots.begin_capture(compute_frontier=4)
    capture_four = capture_native_resident_checkpoint_to_file_v1(
        owner.checkpoint_capture_source_v1(),
        tmp_path / "checkpoint-tick-000000000004",
        artifact_path=ARTIFACT,
        checkpoint_at_tick=4,
        chunk_bytes=64 * 1024,
        require_native_pool=True,
        capture_pool=pool,
    )
    slots.submit_capture(token_four, capture_four)
    assert slots.telemetry()["queue_occupancy_high_water"] == 2
    owner.close()

    result_two = dict(slots.result(token_two))
    slots.acknowledge_durable(token_two)
    result_four = dict(slots.result(token_four))
    slots.acknowledge_durable(token_four)
    slots.close()
    for result in (result_two, result_four):
        assert result["source_device_type"] == "cuda_resident_canonical_rows_v1"
        assert result["producer_full_state_host_bytes_materialized"] == 0
        assert result["producer_compacted_cuda_bytes_materialized"] == 0
        assert result["worker_full_state_bytes_materialized"] == 0
        assert result["worker_cuda_references"] == 0

    checkpoint_two = reopen_native_checkpoint_bundle_v1(result_two["path"])
    checkpoint_four = reopen_native_checkpoint_bundle_v1(result_four["path"])
    assert checkpoint_two == synchronous_two
    assert checkpoint_four == synchronous_four
    restored = CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint_two,
        artifact_path=ARTIFACT,
    )
    restored.advance(2)
    restored_continuation = _drain_and_ack(restored)
    restored_terminal = restored.checkpoint()
    restored.close()

    assert restored_continuation.content_sha256 == original_continuation.content_sha256
    assert restored_terminal.canonical_state == checkpoint_four.canonical_state
    assert restored_terminal.world_uids == checkpoint_four.world_uids
    assert restored_terminal.rng_keys == checkpoint_four.rng_keys
    assert restored_terminal.acknowledged_event_next_tick == 4
    telemetry = pool.telemetry()
    assert telemetry["native_pool"] is True
    assert telemetry["pinned"] is True
    assert telemetry["route_capture_count"] == 4
    assert telemetry["final_cuda_fence_count"] == 4


def test_loader_works_in_clean_process_without_prior_torch_import() -> None:
    command = (
        "from engine.perf.cuda_full_evolution_v2.native_loader import "
        "load_prebuilt_cuda_full_evolution_v2; "
        "loaded = load_prebuilt_cuda_full_evolution_v2(); "
        "assert loaded.target_sm == loaded.device_identity['sm']"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT": str(ARTIFACT),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    with pytest.raises(
        CudaFullEvolutionNativeLoadError, match="requires a CUDA device"
    ):
        load_prebuilt_cuda_full_evolution_v2(ARTIFACT, device="cpu")


@pytest.mark.parametrize(
    ("batch", "start_tick", "ticks", "temporal_k", "world_uids", "partitions"),
    (
        (1, 0, 1, 1, (0x12345,), (1,)),
        (1, 1, 2, 1, (0x12346,), (1, 1)),
        (1, 3, 7, 3, (0x12347,), (3, 3, 1)),
        (2, 17, 7, 2, (0x12348, 0x12349), (2, 2, 2, 1)),
        (4, 0, 7, 7, (0x1234A, 0x1234B, 0x1234C, 0x1234D), (7,)),
        (4, 17, 31, 8, (0x1234E, 0x1234F, 0x12350, 0x12351), (8, 8, 8, 7)),
    ),
)
def test_reference_and_optimized_are_integer_state_exact(
    batch: int,
    start_tick: int,
    ticks: int,
    temporal_k: int,
    world_uids: tuple[int, ...],
    partitions: tuple[int, ...],
) -> None:
    base = _owner(
        ticks,
        batch=batch,
        temporal_k=temporal_k,
        start_tick=start_tick,
        reference=True,
        world_uids=world_uids,
    )
    reference = base.fork()
    optimized = base.fork()
    for span in partitions:
        reference.advance_reference(span)
        optimized.advance(span)
        reference_dispatch = _drain_and_ack(reference)
        optimized_dispatch = _drain_and_ack(optimized)
        assert torch.equal(reference_dispatch.event_rows, optimized_dispatch.event_rows)
        assert torch.equal(
            reference_dispatch.event_counts, optimized_dispatch.event_counts
        )
        assert reference_dispatch.world_uids == optimized_dispatch.world_uids
        assert reference_dispatch.start_next_absolute_tick == (
            optimized_dispatch.start_next_absolute_tick
        )
        assert reference_dispatch.end_next_absolute_tick == (
            optimized_dispatch.end_next_absolute_tick
        )
    _assert_owner_exact(reference, optimized)


def test_temporal_k_decomposition_is_exact() -> None:
    one_tick_dispatches = _owner(7, temporal_k=1, world_uids=(41,))
    three_tick_dispatches = _owner(7, temporal_k=3, world_uids=(41,))
    seven_tick_dispatch = _owner(7, temporal_k=7, world_uids=(41,))
    one_tick_events = []
    three_tick_events = []
    one_tick_counts = torch.zeros((1,), dtype=torch.int64)
    three_tick_counts = torch.zeros((1,), dtype=torch.int64)
    for _ in range(7):
        one_tick_dispatches.advance(1)
        dispatch = _drain_and_ack(one_tick_dispatches)
        one_tick_events.append(dispatch.event_rows)
        one_tick_counts += dispatch.event_counts.to(torch.int64)
    for span in (3, 3, 1):
        three_tick_dispatches.advance(span)
        dispatch = _drain_and_ack(three_tick_dispatches)
        three_tick_events.append(_active_event_rows(dispatch))
        three_tick_counts += dispatch.event_counts.to(torch.int64)
    seven_tick_dispatch.advance(7)
    seven_dispatch = _drain_and_ack(seven_tick_dispatch)
    assert torch.equal(one_tick_dispatches.state, seven_tick_dispatch.state)
    assert torch.equal(three_tick_dispatches.state, seven_tick_dispatch.state)
    assert torch.equal(torch.cat(one_tick_events, dim=1), seven_dispatch.event_rows)
    assert torch.equal(torch.cat(three_tick_events, dim=1), seven_dispatch.event_rows)
    assert torch.equal(one_tick_counts, seven_dispatch.event_counts.to(torch.int64))
    assert torch.equal(three_tick_counts, seven_dispatch.event_counts.to(torch.int64))


def test_deterministic_repeat_is_exact() -> None:
    first = _owner(7, temporal_k=7, seed=8181, world_uids=(51,))
    second = _owner(7, temporal_k=7, seed=8181, world_uids=(51,))
    first.advance(7)
    second.advance(7)
    first_dispatch = _drain_and_ack(first)
    second_dispatch = _drain_and_ack(second)
    _assert_owner_exact(first, second)
    assert torch.equal(first_dispatch.event_rows, second_dispatch.event_rows)

    low_uid = 0x1234
    low_tick = 37
    low_seed = 71
    variants = (
        _owner(
            1, temporal_k=1, start_tick=low_tick, seed=low_seed, world_uids=(low_uid,)
        ),
        _owner(
            1,
            temporal_k=1,
            start_tick=low_tick,
            seed=low_seed,
            world_uids=((1 << 32) | low_uid,),
        ),
        _owner(
            1,
            temporal_k=1,
            start_tick=(1 << 32) | low_tick,
            seed=low_seed,
            world_uids=(low_uid,),
        ),
        _owner(
            1,
            temporal_k=1,
            start_tick=low_tick,
            seed=(1 << 32) | low_seed,
            world_uids=(low_uid,),
        ),
    )
    repeated_high = _owner(
        1,
        temporal_k=1,
        start_tick=(1 << 32) | low_tick,
        seed=(1 << 32) | low_seed,
        world_uids=((1 << 32) | low_uid,),
    )
    repeated_high_again = _owner(
        1,
        temporal_k=1,
        start_tick=(1 << 32) | low_tick,
        seed=(1 << 32) | low_seed,
        world_uids=((1 << 32) | low_uid,),
    )
    for owner in (*variants, repeated_high, repeated_high_again):
        owner.advance(1)
        _drain_and_ack(owner)
    _assert_owner_exact(repeated_high, repeated_high_again)

    layout = state_layout_v2()
    witness_names = ("resource", "wdl_due", "jungle", "policy_alleles", "trait_alleles")
    witnesses = []
    for owner in variants:
        state = owner.state[0]
        witnesses.append(
            torch.cat(
                [
                    state[
                        layout.field(name)
                        .offset_bytes : layout.field(name)
                        .offset_bytes
                        + layout.field(name).bytes
                    ]
                    for name in witness_names
                ]
            )
        )
    assert all(not torch.equal(witnesses[0], witness) for witness in witnesses[1:])


def test_world_uid_partition_and_reorder_invariance() -> None:
    batched = _owner(2, batch=2, temporal_k=2, world_uids=(202, 101))
    single_101 = _owner(2, temporal_k=2, world_uids=(101,))
    single_202 = _owner(2, temporal_k=2, world_uids=(202,))
    batched.advance(2)
    single_101.advance(2)
    single_202.advance(2)
    batched_events = _drain_and_ack(batched)
    events_101 = _drain_and_ack(single_101)
    events_202 = _drain_and_ack(single_202)
    assert torch.equal(batched.state[0], single_202.state[0])
    assert torch.equal(batched.state[1], single_101.state[0])
    assert torch.equal(batched_events.event_rows[0], events_202.event_rows[0])
    assert torch.equal(batched_events.event_rows[1], events_101.event_rows[0])


def test_b4_partition_reassembles_four_b1_owners_by_world_uid() -> None:
    uids = (404, 101, 303, 202)
    batched = _owner(7, batch=4, temporal_k=3, start_tick=17, world_uids=uids)
    singles = {
        uid: _owner(7, temporal_k=3, start_tick=17, world_uids=(uid,)) for uid in uids
    }
    batched_events = []
    single_events = {uid: [] for uid in uids}
    for span in (3, 3, 1):
        batched.advance(span)
        batched_events.append(_active_event_rows(_drain_and_ack(batched)))
        for uid, owner in singles.items():
            owner.advance(span)
            single_events[uid].append(_active_event_rows(_drain_and_ack(owner)))

    batched_state = batched.state
    batched_event_rows = torch.cat(batched_events, dim=1)
    for world_index, uid in enumerate(uids):
        assert torch.equal(batched_state[world_index], singles[uid].state[0])
        assert torch.equal(
            batched_event_rows[world_index],
            torch.cat(single_events[uid], dim=1)[0],
        )


def test_absolute_tick_and_native_overflow_guard() -> None:
    awkward = _owner(7, temporal_k=3, start_tick=37, world_uids=(60,))
    for span in (3, 3, 1):
        awkward.advance(span)
        _drain_and_ack(awkward)
    assert awkward.next_absolute_ticks() == [44]

    owner = _owner(
        1,
        temporal_k=1,
        start_tick=MAX_SIGNED_INT64 - 1,
        world_uids=(61,),
    )
    owner.advance(1)
    _drain_and_ack(owner)
    assert owner.next_absolute_ticks() == [MAX_SIGNED_INT64]
    with pytest.raises(CudaFullEvolutionRuntimeError, match="horizon"):
        owner.advance(1)

    assert owner.next_absolute_ticks() == [MAX_SIGNED_INT64]


def test_forced_birth_mutation_recombination_and_event_codec() -> None:
    base = _owner(1, temporal_k=1, seed=71, world_uids=(71,), reference=True)
    layout = state_layout_v2()
    occupancy_field = layout.field("occupancy")
    position_field = layout.field("position")
    occupancy = base.state[
        0,
        occupancy_field.offset_bytes : occupancy_field.offset_bytes
        + occupancy_field.bytes,
    ].view(torch.int16)
    position = base.state[
        0,
        position_field.offset_bytes : position_field.offset_bytes
        + position_field.bytes,
    ].view(torch.int16)
    assert int(occupancy[34].cpu()) == -1
    assert int(position[34].cpu()) == 256
    assert int(occupancy[256].cpu()) == 34
    base.force_test_conditions()
    initial_positions = (
        base.state[
            0,
            position_field.offset_bytes : position_field.offset_bytes
            + position_field.bytes,
        ]
        .view(torch.int16)
        .clone()
    )
    expressed_policy_field = layout.field("expressed_policy_cache")
    forced_policy = (
        base.state[
            0,
            expressed_policy_field.offset_bytes : expressed_policy_field.offset_bytes
            + expressed_policy_field.bytes,
        ]
        .view(torch.int32)
        .reshape(7, 4, 512)[:, :, :256]
        .cpu()
        .clone()
    )
    assert torch.all(forced_policy[0, 0] == 100)
    forced_policy[0, 0] = 0
    assert torch.count_nonzero(forced_policy).item() == 0
    death_site = int(initial_positions[0].cpu())
    collision_parent_sites = tuple(
        int(initial_positions[slot].cpu()) for slot in (2, 3, 33, 65)
    )
    for parent_site in collision_parent_sites:
        dx = abs((death_site & 31) - (parent_site & 31))
        dy = abs((death_site >> 5) - (parent_site >> 5))
        assert min(dx, 32 - dx) + min(dy, 32 - dy) != 1
    reference = base.fork()
    owner = base.fork()
    reference.advance_reference(1)
    owner.advance(1)
    reference_dispatch = _drain_and_ack(reference)
    dispatch = _drain_and_ack(owner)
    assert torch.equal(reference_dispatch.event_rows, dispatch.event_rows)
    assert torch.equal(reference_dispatch.event_counts, dispatch.event_counts)
    _assert_owner_exact(reference, owner)
    final_positions = owner.state[
        0,
        position_field.offset_bytes : position_field.offset_bytes
        + position_field.bytes,
    ].view(torch.int16)
    assert torch.equal(final_positions[1:256].cpu(), initial_positions[1:256].cpu())
    counters = owner.counters()[0]
    assert all(value > 0 for value in counters.values())
    assert counters["genome_v2_ordinary_birth"] == 9
    assert counters["recombination_profile"] == 9 * 16

    words = _event_words(dispatch)
    committed = 0
    death_seen = False
    birth_seen = False
    birth_slots: list[int] = []
    for row_index, row in enumerate(words):
        tail = row[5] & U64_MASK
        kind = tail & 0xFFFF
        status = (tail >> 32) & 0xFFFF
        reserved = (tail >> 48) & 0xFFFF
        if status == 0:
            assert all((word & U64_MASK) == 0 for word in row)
            continue
        committed += 1
        assert status == 1
        assert reserved == 0
        if row_index < 512:
            assert kind == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_DEATH_V2
            ordinal = row_index
            death_seen = True
        else:
            assert kind == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2
            ordinal = row_index - 512
            birth_seen = True
            birth_slots.append(ordinal)
        assert (row[1] & U64_MASK) == encode_lifecycle_event_id_v2(kind, ordinal)
    assert committed == int(dispatch.event_counts[0])
    assert death_seen and birth_seen
    assert birth_slots == [0, *range(256, 264)]

    records = _committed_event_records(dispatch)
    assert len(records) == committed
    identities = [
        (record["world_uid"], record["absolute_tick"], record["event_id"])
        for record in records
    ]
    assert len(identities) == len(set(identities))
    assert all(record["world_uid"] == 71 for record in records)
    assert all(
        record["absolute_tick"] == dispatch.start_next_absolute_tick
        for record in records
    )
    assert all(record["status"] == 1 and record["reserved"] == 0 for record in records)
    assert all(0 <= record["site_index"] < 1024 for record in records)
    for record in records:
        assert record["agent_id"] > 0
        if record["event_kind"] == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2:
            assert record["parent_a_id"] > 0
            assert record["parent_b_id"] > 0
            assert record["parent_a_id"] != record["parent_b_id"]

    birth_records = [
        record
        for record in records
        if record["event_kind"] == CUDA_NATIVE_LIFECYCLE_EVENT_KIND_BIRTH_V2
    ]
    assert [record["agent_id"] for record in birth_records] == list(range(257, 266))
    assert [record["parent_a_id"] for record in birth_records] == [
        3,
        7,
        11,
        15,
        19,
        23,
        27,
        243,
        247,
    ]
    assert len({record["site_index"] for record in birth_records}) == 9
    assert birth_records[0]["site_index"] == 34

    energy_field = layout.field("energy")
    energy = owner.state[
        0,
        energy_field.offset_bytes : energy_field.offset_bytes + energy_field.bytes,
    ].view(torch.int16)
    assert int(energy[2].cpu()) == 270
    assert int(energy[3].cpu()) == 270
    assert int(energy[33].cpu()) == 300
    assert int(energy[65].cpu()) == 300

    birth_ticks = owner.state[
        0,
        layout.field("absolute_birth_tick")
        .offset_bytes : layout.field("absolute_birth_tick")
        .offset_bytes
        + layout.field("absolute_birth_tick").bytes,
    ].view(torch.int64)
    assert all(int(birth_ticks[slot].cpu()) == 0 for slot in birth_slots)


def test_event_identity_is_unique_across_worlds_and_absolute_ticks() -> None:
    owner = _owner(3, batch=2, temporal_k=2, world_uids=(171, 172))
    owner.force_test_conditions()
    records: list[dict[str, int]] = []
    for span in (2, 1):
        owner.advance(span)
        dispatch = _drain_and_ack(owner)
        records.extend(_committed_event_records(dispatch))

    assert records
    identities = [
        (record["world_uid"], record["absolute_tick"], record["event_id"])
        for record in records
    ]
    assert len(identities) == len(set(identities))
    assert {record["world_uid"] for record in records} == {171, 172}
    assert all(0 <= record["absolute_tick"] < 3 for record in records)
    assert all(record["status"] == 1 and record["reserved"] == 0 for record in records)
    assert all(0 <= record["site_index"] < 1024 for record in records)


def test_public_projections_are_detached_from_runtime_authority() -> None:
    owner = _owner(1, temporal_k=1, reference=True, world_uids=(701,))
    state_projection = owner.state
    events_projection = owner.event_rows
    counts_projection = owner.event_counts
    scratch_projection = owner.reference_scratch
    state_projection.zero_()
    events_projection.fill_(1)
    counts_projection.fill_(1)
    assert scratch_projection is not None
    scratch_projection.fill_(1)
    assert torch.count_nonzero(owner.state).item() > 0
    assert torch.count_nonzero(owner.event_rows).item() == 0
    assert torch.count_nonzero(owner.event_counts).item() == 0
    assert torch.count_nonzero(owner.reference_scratch).item() == 0


def test_event_sink_receipt_binds_exact_unchanged_content_and_identity() -> None:
    owner = _owner(1, temporal_k=1, world_uids=(702,))
    owner.force_test_conditions()
    owner.advance(1)
    dispatch = owner.drain_events()
    receipt = seal_event_sink_receipt_v2(
        dispatch,
        sink_id="pytest-exact-content-sink",
        sink_receipt_id="negative-mutation-case",
    )
    dispatch.event_rows[0, 0, 0] ^= 1
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned|changed"):
        owner.acknowledge_event_drain(receipt)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned"):
        owner.advance(1)
    owner.close()
    owner.close()
    assert owner.closed is True


def test_cpu_event_drain_does_not_allocate_an_event_sized_gpu_temporary() -> None:
    owner = _owner(16, batch=32, temporal_k=16, world_uids=tuple(range(8001, 8033)))
    owner.force_test_conditions()
    owner.advance(16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before_drain = int(torch.cuda.memory_allocated())

    dispatch = owner.drain_events(to_cpu=True)

    peak_during_drain = int(torch.cuda.max_memory_allocated())
    assert dispatch.event_rows.is_cuda is False
    assert dispatch.event_counts.is_cuda is False
    assert peak_during_drain - allocated_before_drain <= 4 << 20
    receipt = seal_event_sink_receipt_v2(
        dispatch,
        sink_id="pytest-exact-content-sink",
        sink_receipt_id="cpu-projection-vram-boundary",
    )
    owner.acknowledge_event_drain(receipt)


def test_event_sink_receipt_rejects_wrong_receipt_and_double_ack() -> None:
    wrong_owner = _owner(1, temporal_k=1, world_uids=(703,))
    wrong_owner.advance(1)
    wrong_dispatch = wrong_owner.drain_events()
    wrong_receipt = seal_event_sink_receipt_v2(
        wrong_dispatch,
        sink_id="pytest-exact-content-sink",
        sink_receipt_id="wrong-seal-case",
    )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned|seal"):
        wrong_owner.acknowledge_event_drain(
            replace(wrong_receipt, receipt_sha256="0" * 64)
        )

    owner = _owner(1, temporal_k=1, world_uids=(704,))
    owner.advance(1)
    dispatch = owner.drain_events(to_cpu=False)
    receipt = seal_event_sink_receipt_v2(
        dispatch,
        sink_id="pytest-exact-content-sink",
        sink_receipt_id="gpu-drain-case",
    )
    owner.acknowledge_event_drain(receipt)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="no drained dispatch"):
        owner.acknowledge_event_drain(receipt)


def test_canonical_clone_excludes_caches_events_and_reference_scratch() -> None:
    owner = _owner(2, temporal_k=2, reference=False, world_uids=(81,))
    assert owner.reference_scratch is None
    owner.advance(1)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        owner.checkpoint()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        owner.advance(1)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        owner.fork()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        owner.counters()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        owner.world_uids()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        _ = owner.active_counts
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        _ = owner.state
    with pytest.raises(CudaFullEvolutionRuntimeError, match="drain"):
        owner.force_test_conditions()
    dispatch = owner.drain_events()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="receipt"):
        owner.checkpoint()
    receipt = seal_event_sink_receipt_v2(
        dispatch,
        sink_id="pytest-exact-content-sink",
        sink_receipt_id="checkpoint-frontier-1",
    )
    owner.acknowledge_event_drain(receipt)
    assert owner.active_counts == (257,)
    checkpoint = owner.checkpoint()
    assert checkpoint.canonical_state_shape == (1, 191_744)
    assert checkpoint.canonical_state_nbytes == 191_744
    assert isinstance(checkpoint.canonical_state, bytes)
    assert len(checkpoint.canonical_state_sha256) == 64
    assert len(checkpoint.canonical_state_layout_sha256) == 64
    assert len(checkpoint.metadata_seal_sha256) == 64
    assert checkpoint.last_sink_receipt_sha256 == receipt.receipt_sha256
    restored = CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint,
        artifact_path=ARTIFACT,
        enable_reference=True,
    )
    assert torch.equal(owner.state, restored.state)
    assert restored.reference_scratch is not None
    assert torch.count_nonzero(restored.reference_scratch).item() == 0
    assert torch.count_nonzero(restored.event_rows).item() == 0
    assert torch.count_nonzero(restored.event_counts).item() == 0


def test_checkpoint_cpu_projection_does_not_allocate_a_surface_sized_gpu_copy() -> None:
    batch = 544
    owner = _owner(1, batch=batch, temporal_k=1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before_checkpoint = int(torch.cuda.memory_allocated())

    checkpoint = owner.checkpoint()

    peak_during_checkpoint = int(torch.cuda.max_memory_allocated())
    expected_nbytes = batch * state_layout_v2().canonical_checkpoint_bytes_per_world
    assert checkpoint.canonical_state_shape == (
        batch,
        state_layout_v2().canonical_checkpoint_bytes_per_world,
    )
    assert checkpoint.canonical_state_nbytes == expected_nbytes
    assert len(checkpoint.canonical_state) == expected_nbytes
    assert peak_during_checkpoint - allocated_before_checkpoint <= 4 << 20

    strided_cuda_view = owner.state[
        :, : state_layout_v2().canonical_checkpoint_bytes_per_world
    ]
    assert strided_cuda_view.is_contiguous() is False
    torch.cuda.reset_peak_memory_stats()
    allocated_before_rejection = int(torch.cuda.memory_allocated())
    with pytest.raises(CudaFullEvolutionRuntimeError, match="non-contiguous CUDA"):
        _tensor_bytes(strided_cuda_view)
    peak_during_rejection = int(torch.cuda.max_memory_allocated())
    assert peak_during_rejection - allocated_before_rejection <= 4 << 20


def test_checkpoint_restore_rejects_frontier_tampering() -> None:
    owner = _owner(2, temporal_k=1, world_uids=(811,))
    owner.advance(1)
    _drain_and_ack(owner)
    checkpoint = owner.checkpoint()

    with pytest.raises(CudaFullEvolutionRuntimeError, match="sink receipt"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, last_sink_receipt_sha256=None),
            artifact_path=ARTIFACT,
        )
    with pytest.raises(
        CudaFullEvolutionRuntimeError, match="acknowledged event frontier"
    ):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(
                checkpoint,
                acknowledged_event_next_tick=(
                    checkpoint.acknowledged_event_next_tick + 1
                ),
            ),
            artifact_path=ARTIFACT,
        )

    bad_state = bytearray(checkpoint.canonical_state)
    bad_state[4 * 8] ^= 0x01
    with pytest.raises(CudaFullEvolutionRuntimeError, match="canonical state hash"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, canonical_state=bytes(bad_state)),
            artifact_path=ARTIFACT,
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="metadata seal"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, metadata_seal_sha256="0" * 64),
            artifact_path=ARTIFACT,
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="schema"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, schema_version="wrong"), artifact_path=ARTIFACT
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="time authority"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, time_authority="wrong"), artifact_path=ARTIFACT
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="byte layout"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(
                checkpoint,
                canonical_state=checkpoint.canonical_state[:-1],
                canonical_state_nbytes=checkpoint.canonical_state_nbytes - 1,
            ),
            artifact_path=ARTIFACT,
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="shape/size seal"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, canonical_state_layout_sha256="0" * 64),
            artifact_path=ARTIFACT,
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="initial tick"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(
                checkpoint,
                initial_next_absolute_tick=checkpoint.initial_next_absolute_tick + 1,
            ),
            artifact_path=ARTIFACT,
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="world UID"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, world_uids=(0,)), artifact_path=ARTIFACT
        )
    for bad_rng in (
        replace(checkpoint, rng_keys=(-1,)),
        replace(checkpoint, rng_semantic_epoch=2),
        replace(checkpoint, rng_semantic_epoch=checkpoint.rng_semantic_epoch + 1),
    ):
        with pytest.raises(CudaFullEvolutionRuntimeError, match="RNG identity"):
            CudaFullEvolutionRuntimeV2.from_checkpoint(bad_rng, artifact_path=ARTIFACT)
    for field_name in ("artifact_sha256", "manifest_sha256"):
        with pytest.raises(CudaFullEvolutionRuntimeError, match="artifact identity"):
            CudaFullEvolutionRuntimeV2.from_checkpoint(
                replace(checkpoint, **{field_name: "bad"}), artifact_path=ARTIFACT
            )


def test_uninterrupted_matches_drain_checkpoint_restore_continuation() -> None:
    base = _owner(7, temporal_k=4, start_tick=17, world_uids=(82,))
    uninterrupted = base.fork()
    resumable = base.fork()

    uninterrupted_events = []
    uninterrupted_counts = []
    uninterrupted_hashes = []
    for span in (3, 4):
        uninterrupted.advance(span)
        dispatch = _drain_and_ack(uninterrupted)
        uninterrupted_events.append(_active_event_rows(dispatch))
        uninterrupted_counts.append(dispatch.event_counts)
        uninterrupted_hashes.append(dispatch.content_sha256)

    resumable.advance(3)
    first_dispatch = _drain_and_ack(resumable)
    checkpoint = resumable.checkpoint()
    restored = CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint, artifact_path=ARTIFACT
    )
    restored.advance(4)
    second_dispatch = _drain_and_ack(restored)

    assert torch.equal(uninterrupted.state, restored.state)
    assert torch.equal(
        torch.cat(uninterrupted_events, dim=1),
        torch.cat(
            (_active_event_rows(first_dispatch), _active_event_rows(second_dispatch)),
            dim=1,
        ),
    )
    assert torch.equal(uninterrupted_counts[0], first_dispatch.event_counts)
    assert torch.equal(uninterrupted_counts[1], second_dispatch.event_counts)
    assert uninterrupted_hashes == [
        first_dispatch.content_sha256,
        second_dispatch.content_sha256,
    ]


@pytest.mark.parametrize(
    ("capability_bit", "counter_name"),
    tuple(
        enumerate(
            (
                "patch",
                "wdl_delayed_hazard",
                "jungle",
                "eed",
                "genome_v2_ordinary_birth",
                "genome_v2_mutation",
                "recombination_profile",
            )
        )
    ),
)
def test_each_diagnostic_capability_mask_is_causal(
    capability_bit: int,
    counter_name: str,
) -> None:
    base = _owner(1, temporal_k=1, world_uids=(91,))
    base.force_test_conditions()
    full = base.fork()
    without_capability = base.fork()
    full.advance(1)
    without_capability.advance_diagnostic(
        1,
        capability_mask=FULL_CAPABILITY_MASK_V2 & ~(1 << capability_bit),
    )
    _drain_and_ack(full)
    _drain_and_ack(without_capability)
    assert not torch.equal(full.state, without_capability.state)
    assert without_capability.counters()[0][counter_name] == 0
    assert full.counters()[0][counter_name] > 0
    witness_differences = {
        field_name: not torch.equal(
            _field_bytes(full, field_name),
            _field_bytes(without_capability, field_name),
        )
        for field_name in CAUSAL_WITNESS_FIELDS_V2[counter_name]
    }
    assert any(witness_differences.values()), (
        f"{counter_name} changed only a header/counter; named canonical witness "
        f"did not change: {witness_differences}"
    )


def test_event_staging_is_exactly_bounded_by_temporal_k() -> None:
    owner = _owner(1, temporal_k=3, world_uids=(101,))
    assert tuple(owner.event_rows.shape) == (
        1,
        3 * CUDA_NATIVE_LIFECYCLE_EVENT_ROWS_PER_WORLD_PER_TICK_V2,
        48,
    )


def test_eager_and_single_node_graph_are_byte_event_counter_exact() -> None:
    owners = {
        mode: _owner(
            4,
            batch=2,
            temporal_k=2,
            world_uids=(9101, 9102),
            execution_mode=mode,
        )
        for mode in ("eager_monolith", "cuda_graph_monolith")
    }
    replay = _owner(
        4,
        batch=2,
        temporal_k=2,
        world_uids=(9101, 9102),
        execution_mode="cuda_graph_monolith",
    )
    for owner in (*owners.values(), replay):
        owner.force_test_conditions()

    evidence: dict[str, tuple[object, object, list[dict[str, int]]]] = {}
    for mode, owner in (*owners.items(), ("graph_replay", replay)):
        event_rows = []
        event_counts = torch.zeros((2,), dtype=torch.int64)
        for _ in range(2):
            owner.advance(2)
            dispatch = _drain_and_ack(owner)
            event_rows.append(_active_event_rows(dispatch))
            event_counts += dispatch.event_counts.to(torch.int64)
        evidence[mode] = (
            torch.cat(event_rows, dim=1),
            event_counts,
            owner.counters(),
        )

    eager = owners["eager_monolith"]
    graph = owners["cuda_graph_monolith"]
    assert torch.equal(eager.state, graph.state)
    assert torch.equal(graph.state, replay.state)
    assert torch.equal(
        evidence["eager_monolith"][0], evidence["cuda_graph_monolith"][0]
    )
    assert torch.equal(evidence["cuda_graph_monolith"][0], evidence["graph_replay"][0])
    assert torch.equal(
        evidence["eager_monolith"][1], evidence["cuda_graph_monolith"][1]
    )
    assert evidence["eager_monolith"][2] == evidence["cuda_graph_monolith"][2]

    eager_receipt = eager.execution_plan_receipt_v2()
    graph_receipt = graph.execution_plan_receipt_v2()
    assert eager_receipt["topology"]["graph_node_count"] == 0
    assert eager_receipt["topology"]["node_types"] == []
    topology = graph_receipt["topology"]
    assert topology["graph_node_count"] == 1
    assert topology["execution_kernel_count"] == 1
    assert topology["node_types"] == ["kernel"]
    assert topology["dependency_edges"] == []
    assert [node["name"] for node in topology["kernel_nodes"]] == [
        "advance_optimized_kernel"
    ]
    assert topology["kernel_nodes"][0]["parameters"]["dispatch_temporal_k"] == 2
    assert topology["kernel_nodes"][0]["parameters"]["capability_mask"] == 127
    assert topology["pointer_substitution_allowed"] is False
    assert topology["strong_tensor_references"] is True
    assert topology["runtime_inspected"] is True
    assert topology["launch_stream_policy"] == (
        "immutable_construction_stream_including_default_stream_0"
    )
    receipt = graph_receipt["last_execution"]
    assert receipt["schema_version"] == (
        "eiel_full_evolution_cuda_native_execution_receipt_v2"
    )
    assert receipt["profile_id"] == graph.contract.profile_id
    assert receipt["artifact_sha256"] == graph.artifact_sha256
    assert receipt["manifest_sha256"] == graph.manifest_sha256
    assert receipt["device_index"] == graph_receipt["topology"]["device_index"]
    assert receipt["graph_launched"] is True
    assert receipt["topology_graph_node_count"] == 1
    assert receipt["topology_execution_kernel_count"] == 1
    assert receipt["topology_node_types"] == ["kernel"]
    assert receipt["topology_kernel_nodes"] == topology["kernel_nodes"]
    assert receipt["launch_ordinal"] == 2
    assert receipt["start_next_absolute_tick"] == 2
    assert receipt["end_next_absolute_tick"] == 4
    assert receipt["committed_world_ticks"] == 4
    assert receipt["runtime_status"] == "committed"
    assert receipt["event_status"] == "exact_content_sink_acknowledged"
    assert receipt["topology_dependency_edges"] == []
    assert receipt["per_node_timing"] is None
    assert receipt["inter_node_gap_timing"] is None
    assert receipt["timing_fabrication_allowed"] is False
    graph_receipt["topology"]["graph_node_count"] = 99
    assert graph.execution_plan_receipt_v2()["topology"]["graph_node_count"] == 1


def test_graph_k_partition_and_fresh_restore_plan_are_exact() -> None:
    k1 = _owner(
        4,
        temporal_k=1,
        start_tick=17,
        world_uids=(9201,),
        execution_mode="cuda_graph_monolith",
    )
    k2 = _owner(
        4,
        temporal_k=2,
        start_tick=17,
        world_uids=(9201,),
        execution_mode="cuda_graph_monolith",
    )
    partition_events: dict[int, list[object]] = {1: [], 2: []}
    for span, owner in ((1, k1), (2, k2)):
        for _ in range(4 // span):
            owner.advance(span)
            partition_events[span].append(_active_event_rows(_drain_and_ack(owner)))
    assert torch.equal(k1.state, k2.state)
    assert torch.equal(
        torch.cat(partition_events[1], dim=1),
        torch.cat(partition_events[2], dim=1),
    )
    assert k1.counters() == k2.counters()

    remainder = _owner(
        3,
        temporal_k=2,
        start_tick=31,
        world_uids=(9203,),
        execution_mode="cuda_graph_monolith",
    )
    unit = _owner(
        3,
        temporal_k=1,
        start_tick=31,
        world_uids=(9203,),
        execution_mode="cuda_graph_monolith",
    )
    for span in (2, 1):
        remainder.advance(span)
        _drain_and_ack(remainder)
    for _ in range(3):
        unit.advance(1)
        _drain_and_ack(unit)
    assert torch.equal(remainder.state, unit.state)
    remainder_receipt = remainder.execution_plan_receipt_v2()
    assert remainder_receipt["prepared_dispatch_temporal_k"] == [1, 2]
    assert remainder_receipt["topology"]["dispatch_temporal_k"] == 1
    assert remainder_receipt["last_execution"]["tick_count"] == 1
    assert remainder_receipt["last_execution"]["start_next_absolute_tick"] == 33
    assert remainder_receipt["last_execution"]["end_next_absolute_tick"] == 34

    source = _owner(
        4,
        temporal_k=2,
        start_tick=17,
        world_uids=(9202,),
        execution_mode="cuda_graph_monolith",
    )
    uninterrupted = source.fork()
    resumable = source.fork()
    uninterrupted_events = []
    for _ in range(2):
        uninterrupted.advance(2)
        uninterrupted_events.append(_active_event_rows(_drain_and_ack(uninterrupted)))
    resumable.advance(2)
    first = _active_event_rows(_drain_and_ack(resumable))
    checkpoint = resumable.checkpoint()
    prior_plan_identity = resumable.execution_plan_receipt_v2()["topology"][
        "plan_identity_sha256"
    ]
    restored = CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint,
        artifact_path=ARTIFACT,
        execution_mode="cuda_graph_monolith",
    )
    restored_plan_identity = restored.execution_plan_receipt_v2()["topology"][
        "plan_identity_sha256"
    ]
    assert restored_plan_identity != prior_plan_identity
    assert restored.execution_plan_receipt_v2()["last_execution"] is None
    restored.advance(2)
    second = _active_event_rows(_drain_and_ack(restored))
    assert torch.equal(uninterrupted.state, restored.state)
    assert torch.equal(
        torch.cat(uninterrupted_events, dim=1), torch.cat((first, second), dim=1)
    )
    assert uninterrupted.counters() == restored.counters()


def test_native_plan_rejects_binding_tamper_and_closed_launch() -> None:
    owner = _owner(2, temporal_k=2, world_uids=(9301,))
    loaded = load_prebuilt_cuda_full_evolution_v2(ARTIFACT)
    state = owner.state
    events = owner.event_rows
    event_counts = owner.event_counts
    with pytest.raises(RuntimeError, match="artifact identity"):
        loaded.module.create_execution_plan(
            "eager_monolith",
            state,
            events,
            event_counts,
            2,
            2,
            127,
            owner.contract.profile_id,
            "bad",
            owner.manifest_sha256,
        )

    plan = loaded.module.create_execution_plan(
        "eager_monolith",
        state,
        events,
        event_counts,
        2,
        2,
        127,
        owner.contract.profile_id,
        owner.artifact_sha256,
        owner.manifest_sha256,
    )
    events.resize_((1, 2048, 47))
    with pytest.raises(RuntimeError, match="events must|binding changed"):
        plan.launch(0, 2)
    plan.close()
    with pytest.raises(RuntimeError, match="closed"):
        plan.launch(0, 2)

    poisoned = _owner(
        2,
        temporal_k=2,
        world_uids=(9302,),
        execution_mode="cuda_graph_monolith",
    )
    poisoned._CudaFullEvolutionRuntimeV2__execution_plans[2].close()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned|launch failed"):
        poisoned.advance(2)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned"):
        poisoned.advance(2)
    poisoned.close()
    poisoned.close()
    assert poisoned.closed is True

    owner.close()
    assert owner.closed is True
    assert owner.execution_plan_receipt_v2()["closed"] is True
    with pytest.raises(CudaFullEvolutionRuntimeError, match="closed"):
        owner.advance(2)


def test_graph_topology_binding_rejects_every_fabricated_authority_field() -> None:
    owner = _owner(
        1,
        temporal_k=1,
        world_uids=(9401,),
        execution_mode="cuda_graph_monolith",
    )
    plan = owner._CudaFullEvolutionRuntimeV2__execution_plans[1]
    exact = dict(plan.topology_manifest())

    class _FabricatedTopologyPlan:
        def topology_manifest(self):
            return exact

    with pytest.raises(CudaFullEvolutionRuntimeError, match="exact loaded native"):
        owner._bind_execution_plan_topology(
            _FabricatedTopologyPlan(), dispatch_temporal_k=1
        )

    mutations = {
        "artifact_sha256": "0" * 64,
        "device_index": exact["device_index"] + 1,
        "graph_node_count": 2,
        "dependency_edges": [[0, 1]],
        "strong_tensor_references": False,
        "runtime_inspected": False,
        "pointer_substitution_allowed": True,
    }
    for field, fabricated in mutations.items():
        candidate = deepcopy(exact)
        candidate[field] = fabricated
        with pytest.raises(CudaFullEvolutionRuntimeError, match="mismatch"):
            owner._bind_execution_plan_topology_payload(
                candidate, dispatch_temporal_k=1
            )

    candidate = deepcopy(exact)
    candidate["kernel_nodes"][0]["name"] = "fabricated_kernel"
    with pytest.raises(CudaFullEvolutionRuntimeError, match="mismatch"):
        owner._bind_execution_plan_topology_payload(candidate, dispatch_temporal_k=1)

    candidate = deepcopy(exact)
    candidate["pointer_bindings"]["state"] += 256
    candidate["kernel_nodes"][0]["parameters"]["state_pointer"] += 256
    with pytest.raises(CudaFullEvolutionRuntimeError, match="pointer binding"):
        owner._bind_execution_plan_topology_payload(candidate, dispatch_temporal_k=1)

    candidate = deepcopy(exact)
    candidate["fabricated_field"] = True
    with pytest.raises(CudaFullEvolutionRuntimeError, match="binding mismatch"):
        owner._bind_execution_plan_topology_payload(candidate, dispatch_temporal_k=1)


@pytest.mark.parametrize(
    ("case", "field", "fabricated"),
    (
        ("wrong-schema", "schema_version", "fabricated_schema"),
        ("wrong-kernel", "kernel", "fabricated_kernel"),
        ("wrong-device", "device_index", -1),
        ("wrong-graph-flag", "graph_launched", False),
        ("fabricated-commit", "committed_world_ticks", 1),
        ("fabricated-node-timing", "per_node_timing", {"milliseconds": 0.1}),
        ("fabricated-gap-timing", "inter_node_gap_timing", {"milliseconds": 0.1}),
        ("missing-source-binding", "artifact_sha256", None),
        ("unexpected-field", "fabricated_field", True),
    ),
)
def test_launch_receipt_binding_rejects_fabrication_and_cleanup_remains_safe(
    case: str, field: str, fabricated: object
) -> None:
    owner = _owner(
        1,
        temporal_k=1,
        world_uids=(9501,),
        execution_mode="cuda_graph_monolith",
    )
    native_plan = owner._CudaFullEvolutionRuntimeV2__execution_plans[1]
    topology = owner._CudaFullEvolutionRuntimeV2__execution_plan_topologies[1]
    start_tick = owner.committed_next_absolute_tick
    native_plan.launch(start_tick, 1)
    receipt = dict(native_plan.last_execution_receipt())
    if case == "missing-source-binding":
        receipt.pop(field)
    else:
        receipt[field] = fabricated
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned|binding"):
        owner._bind_native_execution_receipt(
            receipt,
            topology=topology,
            start_next_absolute_tick=start_tick,
            tick_count=1,
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned"):
        owner.advance(1)
    owner.close()
    owner.close()
    assert owner.closed is True


def test_default_stream_zero_is_a_bound_stream_and_stream_change_fails_closed() -> None:
    default_stream = torch.cuda.default_stream()
    with torch.cuda.stream(default_stream):
        owner = _owner(
            2,
            temporal_k=1,
            world_uids=(9601,),
            execution_mode="cuda_graph_monolith",
        )
        topology = owner.execution_plan_receipt_v2()["topology"]
        assert topology["construction_stream_handle"] == int(default_stream.cuda_stream)
        owner.advance(1)
        _drain_and_ack(owner)

    alternate_stream = torch.cuda.Stream()
    with torch.cuda.stream(alternate_stream):
        with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned|stream"):
            owner.advance(1)
    owner.close()
    owner.close()
    assert owner.closed is True


def test_full_capacity_benchmark_fixture_is_exact_sealed_and_restorable() -> None:
    owner = _owner(
        2,
        batch=2,
        temporal_k=1,
        start_tick=41,
        world_uids=(9701, 9702),
        execution_mode="cuda_graph_monolith",
    )
    checkpoint = owner.force_full_capacity_benchmark_fixture_v2()
    assert owner.population_fixture_id == FULL_CAPACITY_BENCHMARK_FIXTURE_V2
    assert owner.active_counts == (512, 512)
    assert all(all(value == 0 for value in row.values()) for row in owner.counters())
    assert checkpoint.population_fixture_id == FULL_CAPACITY_BENCHMARK_FIXTURE_V2
    assert checkpoint.advanced_ticks == 0
    assert checkpoint.acknowledged_event_next_tick == 41
    assert checkpoint.artifact_sha256 == owner.artifact_sha256
    assert checkpoint.manifest_sha256 == owner.manifest_sha256
    assert (
        dict(checkpoint.device_identity)
        == owner.execution_plan_receipt_v2()["topology"]["device_identity"]
    )
    assert owner.checkpoint().canonical_state_sha256 == (
        checkpoint.canonical_state_sha256
    )

    restored = CudaFullEvolutionRuntimeV2.from_checkpoint(
        checkpoint,
        artifact_path=ARTIFACT,
        execution_mode="cuda_graph_monolith",
    )
    assert restored.population_fixture_id == FULL_CAPACITY_BENCHMARK_FIXTURE_V2
    assert restored.active_counts == (512, 512)
    assert torch.equal(owner.state, restored.state)
    owner.advance(1)
    owner_dispatch = _drain_and_ack(owner)
    restored.advance(1)
    restored_dispatch = _drain_and_ack(restored)
    assert torch.equal(owner.state, restored.state)
    assert torch.equal(owner_dispatch.event_rows, restored_dispatch.event_rows)
    assert torch.equal(owner_dispatch.event_counts, restored_dispatch.event_counts)
    assert owner.counters() == restored.counters()


def test_full_capacity_benchmark_fixture_fails_closed_outside_fresh_start() -> None:
    owner = _owner(2, temporal_k=1, world_uids=(9801,))
    checkpoint = owner.force_full_capacity_benchmark_fixture_v2()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="fresh canonical P256"):
        owner.force_full_capacity_benchmark_fixture_v2()

    advanced = _owner(2, temporal_k=1, world_uids=(9802,))
    advanced.advance(1)
    _drain_and_ack(advanced)
    with pytest.raises(CudaFullEvolutionRuntimeError, match="fresh canonical P256"):
        advanced.force_full_capacity_benchmark_fixture_v2()

    diagnostic = _owner(2, temporal_k=1, world_uids=(9803,))
    diagnostic.force_test_conditions()
    with pytest.raises(CudaFullEvolutionRuntimeError, match="poisoned|status"):
        diagnostic.force_full_capacity_benchmark_fixture_v2()
    diagnostic.close()
    assert diagnostic.closed is True

    tampered_device = dict(checkpoint.device_identity)
    tampered_device["index"] += 1
    with pytest.raises(CudaFullEvolutionRuntimeError, match="seal|device"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, device_identity=tampered_device),
            artifact_path=ARTIFACT,
        )
    with pytest.raises(CudaFullEvolutionRuntimeError, match="fixture|seal"):
        CudaFullEvolutionRuntimeV2.from_checkpoint(
            replace(checkpoint, population_fixture_id="fabricated_fixture"),
            artifact_path=ARTIFACT,
        )
