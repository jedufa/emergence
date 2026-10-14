from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from engine.campaigns.providers import (
    CampaignProviderError,
    ResidentBatchRequestV1,
    _qualification_tensor_row_v1,
    _qualification_transition_sentinels_v1,
)


def test_fixed_patch_qualification_sentinels_cover_first_risky_boundary() -> None:
    batch = 2
    ticks = 6
    capacity = 4
    bookkeeping = torch.zeros((batch, ticks, 5), dtype=torch.int64)
    bookkeeping[:, 2, -2] = 1
    bookkeeping[:, 2, -1] = capacity
    result = SimpleNamespace(
        runtime_bookkeeping_by_tick=bookkeeping,
        drive_noise_rng_semantic_draws_by_tick=torch.arange(
            batch * ticks, dtype=torch.int64
        ).reshape(batch, ticks),
        event_tape_active_count=torch.arange(
            batch * ticks, dtype=torch.int64
        ).reshape(batch, ticks),
        actions_by_tick_slot=torch.arange(
            batch * ticks * capacity, dtype=torch.int64
        ).reshape(batch, ticks, capacity),
        lifecycle_alive_counts_by_tick=torch.full(
            (batch, ticks), capacity, dtype=torch.int64
        ),
        patch_stats_by_tick=torch.arange(
            batch * ticks * 2, dtype=torch.int64
        ).reshape(batch, ticks, 2),
        policy_pcg64_state_after_action_by_tick=torch.arange(
            batch * ticks * 2, dtype=torch.int64
        ).reshape(batch, ticks, 2),
    )
    request = ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id="batch-0000",
        cell_ids=("cell-a", "cell-b"),
        generations=3,
        ticks_per_generation=2,
        checkpoint_milestones=(3,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="a" * 64,
    )

    sentinels = _qualification_transition_sentinels_v1(
        result,
        request,
        start_generation=0,
        segment_generations=3,
    )

    assert set(sentinels) == {"cell-a", "cell-b"}
    names = [row["transition_name"] for row in sentinels["cell-a"]]
    assert names[:5] == [
        "last_tick_before_generation_boundary",
        "generation_boundary",
        "first_tick_after_boundary",
        "first_policy_cache_refill_after_boundary",
        "second_tick_after_boundary",
    ]
    first_post = sentinels["cell-a"][2]
    assert first_post["absolute_generation"] == 2
    assert first_post["absolute_tick"] == 2
    assert first_post["physical_window_position"] == 2
    refill = sentinels["cell-a"][3]
    assert refill["refill_calls"] == 1
    assert "refill_row_digest" not in refill
    assert "cache_epoch" not in refill
    assert "tape_cursor" not in refill
    assert refill["action_digest"]
    assert "alive_identity_digest" not in refill
    assert names[-2:] == ["terminal_settlement", "checkpoint_export_boundary"]


def test_sentinels_remain_optional_outside_qualification_mode() -> None:
    request = ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id="batch-0000",
        cell_ids=("cell-a",),
        generations=1,
        ticks_per_generation=1,
        checkpoint_milestones=(1,),
        parameter_arrays=(),
        seed_arrays=(),
    )
    result = SimpleNamespace(
        runtime_bookkeeping_by_tick=torch.zeros((1, 1, 5), dtype=torch.int64)
    )

    sentinels = _qualification_transition_sentinels_v1(
        result,
        request,
        start_generation=0,
        segment_generations=1,
    )

    assert [row["transition_name"] for row in sentinels["cell-a"]] == [
        "terminal_settlement",
        "checkpoint_export_boundary",
    ]


def test_resumed_segment_emits_honest_restore_entry_sentinels() -> None:
    ticks_per_generation = 2
    segment_generations = 2
    total_ticks = ticks_per_generation * segment_generations
    bookkeeping = torch.zeros((1, total_ticks, 5), dtype=torch.int64)
    bookkeeping[0, 0, -2] = 7
    bookkeeping[0, 1, -2] = 8
    actions = torch.arange(
        total_ticks * 3, dtype=torch.int64
    ).reshape(1, total_ticks, 3)
    result = SimpleNamespace(
        runtime_bookkeeping_by_tick=bookkeeping,
        drive_noise_rng_semantic_draws_by_tick=torch.tensor(
            [[101, 102, 103, 104]], dtype=torch.int64
        ),
        event_tape_active_count=torch.tensor(
            [[11, 12, 13, 14]], dtype=torch.int64
        ),
        actions_by_tick_slot=actions,
        patch_stats_by_tick=torch.arange(
            total_ticks * 2, dtype=torch.int64
        ).reshape(1, total_ticks, 2),
    )
    request = ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id="resume-cell-a",
        cell_ids=("cell-a",),
        generations=5,
        ticks_per_generation=ticks_per_generation,
        checkpoint_milestones=(5,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="a" * 64,
    )

    sentinels = _qualification_transition_sentinels_v1(
        result,
        request,
        start_generation=3,
        segment_generations=segment_generations,
    )["cell-a"]

    assert [row["transition_name"] for row in sentinels[:3]] == [
        "first_tick_after_restore",
        "first_policy_cache_refill_after_restore",
        "second_tick_after_restore",
    ]
    assert [
        (
            row["absolute_generation"],
            row["absolute_tick"],
            row["physical_window_position"],
        )
        for row in sentinels[:3]
    ] == [(4, 6, 0), (4, 6, 0), (4, 7, 1)]
    assert sentinels[0]["rng_cursor"] == 101
    assert sentinels[1]["refill_calls"] == 7
    assert sentinels[2]["rng_cursor"] == 102
    assert sentinels[2]["refill_calls"] == 8
    assert sentinels[0]["action_digest"] == sentinels[1]["action_digest"]
    assert sentinels[0]["action_digest"] != sentinels[2]["action_digest"]
    assert sentinels[0]["profile_fields"] == {
        "event_tape_active_count": 11
    }
    assert sentinels[2]["profile_fields"] == {
        "event_tape_active_count": 12
    }


def test_single_tick_resumed_segment_omits_unavailable_second_restore_tick() -> None:
    request = ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id="resume-cell-a",
        cell_ids=("cell-a",),
        generations=2,
        ticks_per_generation=1,
        checkpoint_milestones=(2,),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="a" * 64,
    )
    result = SimpleNamespace(
        runtime_bookkeeping_by_tick=torch.zeros((1, 1, 5), dtype=torch.int64)
    )

    sentinels = _qualification_transition_sentinels_v1(
        result,
        request,
        start_generation=1,
        segment_generations=1,
    )["cell-a"]

    names = [row["transition_name"] for row in sentinels]
    assert names[:2] == [
        "first_tick_after_restore",
        "first_policy_cache_refill_after_restore",
    ]
    assert "second_tick_after_restore" not in names


def test_tick_native_sentinels_use_exact_dispatch_ticks_not_generation_aliases() -> None:
    request = ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id="tick-native-cell-a",
        cell_ids=("cell-a",),
        generations=1,
        ticks_per_generation=4,
        checkpoint_milestones=(1,),
        parameter_arrays=(),
        seed_arrays=(),
        total_ticks=10,
        source_window_ticks=4,
        execution_route_sha256="a" * 64,
    )
    result = SimpleNamespace(
        runtime_bookkeeping_by_tick=torch.zeros((1, 4, 5), dtype=torch.int64)
    )

    sentinels = _qualification_transition_sentinels_v1(
        result,
        request,
        start_generation=0,
        segment_generations=1,
        start_next_tick=3,
        end_next_tick=6,
    )["cell-a"]

    assert [row["transition_name"] for row in sentinels] == [
        "first_tick_after_boundary",
        "second_tick_after_boundary",
        "terminal_settlement",
        "checkpoint_export_boundary",
    ]
    assert [row["absolute_tick"] for row in sentinels] == [3, 4, 5, 6]
    assert sentinels[-2]["absolute_generation"] == sentinels[-1][
        "absolute_generation"
    ]
    assert "first_policy_cache_refill_after_boundary" not in {
        row["transition_name"] for row in sentinels
    }


def test_tick_native_single_tick_boundary_omits_refill_and_second_tick() -> None:
    request = ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id="tick-native-cell-a",
        cell_ids=("cell-a",),
        generations=1,
        ticks_per_generation=4,
        checkpoint_milestones=(1,),
        parameter_arrays=(),
        seed_arrays=(),
        total_ticks=10,
        source_window_ticks=4,
        execution_route_sha256="a" * 64,
    )
    bookkeeping = torch.zeros((1, 2, 5), dtype=torch.int64)
    bookkeeping[0, 0, -2] = 1
    result = SimpleNamespace(runtime_bookkeeping_by_tick=bookkeeping)

    sentinels = _qualification_transition_sentinels_v1(
        result,
        request,
        start_generation=1,
        segment_generations=1,
        start_next_tick=4,
        end_next_tick=5,
    )["cell-a"]

    names = [row["transition_name"] for row in sentinels]
    assert names == [
        "first_tick_after_boundary",
        "terminal_settlement",
        "checkpoint_export_boundary",
    ]
    assert "second_tick_after_boundary" not in names
    assert "first_policy_cache_refill_after_boundary" not in names
    assert sentinels[0]["absolute_tick"] == 4


def test_tick_native_fresh_restore_is_not_mislabeled_as_dispatch_boundary() -> None:
    request = ResidentBatchRequestV1(
        profile_id="eiel-production-runtime-v1",
        batch_id="tick-native-cell-a",
        cell_ids=("cell-a",),
        generations=1,
        ticks_per_generation=4,
        checkpoint_milestones=(1,),
        parameter_arrays=(),
        seed_arrays=(),
        total_ticks=10,
        source_window_ticks=4,
        execution_route_sha256="a" * 64,
    )
    result = SimpleNamespace(
        runtime_bookkeeping_by_tick=torch.zeros((1, 4, 5), dtype=torch.int64)
    )

    sentinels = _qualification_transition_sentinels_v1(
        result,
        request,
        start_generation=0,
        segment_generations=1,
        start_next_tick=3,
        end_next_tick=6,
        restored_entry=True,
    )["cell-a"]

    names = [row["transition_name"] for row in sentinels]
    assert names[:3] == [
        "first_tick_after_restore",
        "first_policy_cache_refill_after_restore",
        "second_tick_after_restore",
    ]
    assert not any(name.endswith("after_boundary") for name in names)


def test_qualification_rows_use_declared_axes_at_b1_and_b2() -> None:
    b1 = torch.arange(12, dtype=torch.int64).reshape(1, 3, 4)
    b2 = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4)

    assert torch.equal(
        _qualification_tensor_row_v1(
            b1,
            name="actions_by_tick_slot",
            cell_index=0,
            tick_index=2,
            batch_cells=1,
        ),
        b1[0, 2],
    )
    assert torch.equal(
        _qualification_tensor_row_v1(
            b2,
            name="actions_by_tick_slot",
            cell_index=1,
            tick_index=2,
            batch_cells=2,
        ),
        b2[1, 2],
    )


def test_qualification_rows_reject_shape_coincidence_and_unknown_fields() -> None:
    with pytest.raises(CampaignProviderError, match="axis contract"):
        _qualification_tensor_row_v1(
            torch.zeros((2, 4), dtype=torch.int64),
            name="actions_by_tick_slot",
            cell_index=0,
            tick_index=0,
            batch_cells=2,
        )
    with pytest.raises(CampaignProviderError, match="no declared axis"):
        _qualification_tensor_row_v1(
            torch.zeros((2, 4), dtype=torch.int64),
            name="undeclared_tensor",
            cell_index=0,
            tick_index=0,
            batch_cells=2,
        )
