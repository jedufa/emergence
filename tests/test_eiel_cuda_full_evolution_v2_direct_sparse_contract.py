from __future__ import annotations

import pytest

from engine.perf.cuda_full_evolution_v2.direct_sparse_autotune import (
    DirectSparseAutotuneError,
    plan_direct_sparse_autotune_v3,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3,
)
from engine.perf.cuda_full_evolution_v2.native_loader import (
    expected_layout_manifest_v2,
)


def _b_values(plan: dict[str, object]) -> list[int]:
    return [
        int(row["batch_size"])
        for row in plan["candidates"]  # type: ignore[index,union-attr]
    ]


def test_direct_sparse_plan_contains_measured_gtx1660ti_neighborhood() -> None:
    plan = plan_direct_sparse_autotune_v3(
        multiprocessor_count=24,
        current_free_bytes=5_900_000_000,
        total_memory_bytes=6_027_608_064,
        temporal_k_candidates=(64,),
    )
    assert _b_values(plan) == [576, 704, 768, 896, 960, 1024, 1088, 1152, 1344]
    assert {row["initial_capacity_rows_per_world"] for row in plan["candidates"]} == {
        512
    }
    assert plan["scaling_law"]["horizon_dependent_resident_bytes"] == 0


def test_direct_sparse_plan_can_bind_one_explicit_native_carrier_size() -> None:
    plan = plan_direct_sparse_autotune_v3(
        multiprocessor_count=24,
        current_free_bytes=5_900_000_000,
        total_memory_bytes=6_027_608_064,
        temporal_k_candidates=(64,),
        batch_size_candidates=(1024,),
    )

    assert [(row["batch_size"], row["temporal_k"]) for row in plan["candidates"]] == [
        (1024, 64)
    ]
    assert plan["scaling_law"]["batch_source"] == (
        "explicit_aligned_batch_candidates"
    )


def test_direct_sparse_aligned_b64_plan_identity_is_unchanged() -> None:
    plan = plan_direct_sparse_autotune_v3(
        multiprocessor_count=24,
        current_free_bytes=5_900_000_000,
        total_memory_bytes=6_027_608_064,
        temporal_k_candidates=(64,),
        batch_size_candidates=(64,),
    )

    assert plan["plan_sha256"] == (
        "d111e4c707861de85d7e8c54ee7832ad6a9cb463ec7b97dfc8819b952a2633d0"
    )
    assert plan["scaling_law"]["batch_alignment"] == 64
    assert plan["scaling_law"]["batch_source"] == (
        "explicit_aligned_batch_candidates"
    )


def test_direct_sparse_plan_can_bind_one_exact_b16_tail() -> None:
    plan = plan_direct_sparse_autotune_v3(
        multiprocessor_count=24,
        current_free_bytes=5_900_000_000,
        total_memory_bytes=6_027_608_064,
        temporal_k_candidates=(64,),
        exact_tail_batch_size=16,
    )

    assert [(row["batch_size"], row["temporal_k"]) for row in plan["candidates"]] == [
        (16, 64)
    ]
    assert plan["candidates"][0]["blocks_per_sm_wave_target"] == 1
    assert plan["scaling_law"]["batch_alignment"] == 1
    assert plan["scaling_law"]["block_waves_per_sm"] == [1]
    assert plan["scaling_law"]["batch_source"] == (
        "explicit_exact_tail_candidate"
    )


@pytest.mark.parametrize("tail_batch_size", [True, 0, -1, 64, 80])
def test_direct_sparse_exact_tail_rejects_non_tail_sizes(
    tail_batch_size: object,
) -> None:
    with pytest.raises(
        DirectSparseAutotuneError,
        match=r"exact tail batch size must be an integer in \[1,63\]",
    ):
        plan_direct_sparse_autotune_v3(
            multiprocessor_count=24,
            current_free_bytes=5_900_000_000,
            total_memory_bytes=6_027_608_064,
            temporal_k_candidates=(64,),
            exact_tail_batch_size=tail_batch_size,  # type: ignore[arg-type]
        )


def test_direct_sparse_exact_tail_does_not_relax_aligned_candidates() -> None:
    common = {
        "multiprocessor_count": 24,
        "current_free_bytes": 5_900_000_000,
        "total_memory_bytes": 6_027_608_064,
        "temporal_k_candidates": (64,),
    }
    with pytest.raises(
        DirectSparseAutotuneError,
        match="positive multiples of 64",
    ):
        plan_direct_sparse_autotune_v3(
            **common,
            batch_size_candidates=(16,),
        )
    with pytest.raises(
        DirectSparseAutotuneError,
        match="mutually exclusive",
    ):
        plan_direct_sparse_autotune_v3(
            **common,
            batch_size_candidates=(64,),
            exact_tail_batch_size=16,
        )


def test_direct_sparse_plan_scales_with_a_larger_gpu() -> None:
    current = plan_direct_sparse_autotune_v3(
        multiprocessor_count=24,
        current_free_bytes=5_900_000_000,
        total_memory_bytes=6_027_608_064,
    )
    larger = plan_direct_sparse_autotune_v3(
        multiprocessor_count=128,
        current_free_bytes=22_000_000_000,
        total_memory_bytes=24_000_000_000,
    )
    assert min(_b_values(larger)) > max(_b_values(current))
    assert max(_b_values(larger)) == 7168
    assert larger["plan_sha256"] != current["plan_sha256"]


def test_direct_sparse_plan_applies_exact_vram_gate() -> None:
    constrained = plan_direct_sparse_autotune_v3(
        multiprocessor_count=24,
        current_free_bytes=700_000_000,
        total_memory_bytes=6_000_000_000,
    )
    assert _b_values(constrained) == [576, 704, 768, 896, 960, 1024, 1088]
    assert all(
        row["estimated_peak_bytes"] <= 700_000_000 for row in constrained["candidates"]
    )


def test_direct_sparse_plan_scales_capacity_from_temporal_k() -> None:
    plan = plan_direct_sparse_autotune_v3(
        multiprocessor_count=24,
        current_free_bytes=5_900_000_000,
        total_memory_bytes=6_027_608_064,
        temporal_k_candidates=(8, 64),
    )
    capacities = {
        (row["temporal_k"], row["initial_capacity_rows_per_world"])
        for row in plan["candidates"]
    }
    assert capacities == {(8, 64), (64, 512)}
    assert DIRECT_SPARSE_MAX_CAPACITY_ROWS_V3 == 65_536


@pytest.mark.parametrize(
    ("multiprocessors", "free_bytes", "total_bytes", "temporal"),
    [
        (0, 1, 1, (64,)),
        (1, 0, 1, (64,)),
        (1, 2, 1, (64,)),
        (1, 1, 1, ()),
        (1, 1_000_000_000, 1_000_000_000, (0,)),
        (1, 1_000_000_000, 1_000_000_000, (65,)),
    ],
)
def test_direct_sparse_plan_rejects_invalid_device_or_tick_envelopes(
    multiprocessors: int,
    free_bytes: int,
    total_bytes: int,
    temporal: tuple[int, ...],
) -> None:
    with pytest.raises(DirectSparseAutotuneError):
        plan_direct_sparse_autotune_v3(
            multiprocessor_count=multiprocessors,
            current_free_bytes=free_bytes,
            total_memory_bytes=total_bytes,
            temporal_k_candidates=temporal,
        )


def test_sparse_wire_abi_is_host_manifest_ratcheted() -> None:
    sparse = expected_layout_manifest_v2()["sparse_lifecycle_event_v3"]
    assert sparse == {
        "wire_abi": "eiel_full_evolution_cuda_sparse_lifecycle_event_v3",
        "bytes": 32,
        "fields": [
            {"name": "agent_id", "dtype": "u64"},
            {"name": "parent_a_id", "dtype": "u64"},
            {"name": "parent_b_id", "dtype": "u64"},
            {"name": "packed_key_and_reserved", "dtype": "u64"},
        ],
        "packed_key": {
            "tick_offset": {"bits": 6, "shift": 0},
            "event_kind": {"bits": 2, "shift": 6},
            "subject_slot": {"bits": 9, "shift": 8},
            "site_index": {"bits": 10, "shift": 17},
            "used_bits": 27,
            "high_reserved_value": 0,
        },
        "canonical_order": [
            "world_uid",
            "tick_offset",
            "death_before_birth",
            "subject_slot",
        ],
        "world_offsets": "u64[B+1]_zero_origin_exact_terminal_N",
    }
