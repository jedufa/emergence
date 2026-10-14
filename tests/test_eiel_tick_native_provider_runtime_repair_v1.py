from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from engine.campaigns.providers import (
    CampaignProviderError,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
    _LazyProjectionBundleSequence,
    _checkpoint_identity_contract_v1,
    _checkpoint_identity_sha256_v1,
    _canonical_sha256,
    _checkpoint_snapshot_sequence_v1,
    _require_tick_native_event_sink_builder_v1,
    _reconcile_tick_native_resume_event_sink_v1,
    _validate_tick_native_direct_event_storage_v1,
    _validate_tick_native_checkpoint_identity_v1,
)
from engine.campaigns.production_engine_native_bindings import (
    _validated_tick_native_abi_span_v1,
)
from engine.campaigns.tick_native_control import TickNativeControlV1
from engine.native_runtime.run_controller import (
    NativeRunControllerError,
    _validated_observer_continuation,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
    _bind_tick_native_event_sink_arguments_v1,
    _require_tick_native_event_sink_result_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    CudaArchitectureBDirectResidentSourceAllocatorError,
    _tick_native_private_source_capacity_window_count_v1,
    _tick_native_allocator_projection_v1,
)


def _native_request(
    total_ticks: int,
    *,
    checkpoint_ticks: tuple[int, ...] = (),
    report_every_ticks: int | None = None,
    science_sample_interval_ticks: int | None = None,
    reducer_every_ticks: int | None = None,
    archive_every_ticks: int | None = None,
) -> ResidentBatchRequestV1:
    control = TickNativeControlV1(
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        source_window_ticks=160,
        event_window_capacity=65536,
        report_every_ticks=report_every_ticks,
        science_sample_interval_ticks=science_sample_interval_ticks,
        reducer_every_ticks=reducer_every_ticks,
        archive_every_ticks=archive_every_ticks,
    )
    return ResidentBatchRequestV1(
        profile_id="test-profile",
        batch_id="tick-native-repair",
        cell_ids=("cell-0",),
        generations=None,
        ticks_per_generation=None,
        checkpoint_milestones=(),
        parameter_arrays=(),
        seed_arrays=(),
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        internal_lifecycle_interval_ticks=None,
        max_internal_lifecycle_segments_per_dispatch=None,
        tick_native_control=control.payload(),
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        source_window_ticks=160,
        event_window_capacity=65536,
        report_every_ticks=report_every_ticks,
        science_sample_interval_ticks=science_sample_interval_ticks,
        reducer_every_ticks=reducer_every_ticks,
        archive_every_ticks=archive_every_ticks,
    )


@pytest.mark.parametrize(
    ("total_ticks", "checkpoints"),
    [
        (161, (161,)),
        (1121, (500, 1121)),
        (5000, (137, 500, 1121, 5000)),
    ],
)
def test_native_segments_are_executable_contiguous_and_owner_confined(
    total_ticks: int, checkpoints: tuple[int, ...]
) -> None:
    segments = ProductionEngineLifecycleV1._absolute_tick_segments_v1(
        _native_request(total_ticks, checkpoint_ticks=checkpoints),
        start_next_tick=0,
    )

    assert segments[0]["start_next_tick"] == 0
    assert segments[-1]["end_next_tick"] == total_ticks
    assert all(
        left["end_next_tick"] == right["start_next_tick"]
        for left, right in zip(segments, segments[1:])
    )
    assert all(1 <= row["absolute_tick_span"] <= 1120 for row in segments)
    assert all(
        row["owner_span_start_tick"] <= row["start_next_tick"]
        and row["end_next_tick"] <= row["owner_span_end_tick"]
        for row in segments
    )
    assert all(row["dispatch_window_ticks"] == 1120 for row in segments)
    assert [
        row["end_next_tick"] for row in segments if row["publish_checkpoint"]
    ] == list(checkpoints)


def _checkpoint(next_tick: int) -> dict[str, object]:
    route = "a" * 64
    state = {"cell-0": "b" * 64}
    storage = {"cell-0": "c" * 64}
    contract = _checkpoint_identity_contract_v1(
        execution_route_sha256=route,
        next_absolute_tick=next_tick,
        source_import_projection_digest_sha256s=state,
        parent_checkpoint_projection_digest_sha256s=None,
        terminal_projection_digest_sha256s=state,
        canonical_terminal_state_digest_sha256s=state,
        canonical_checkpoint_storage_digest_sha256s=storage,
    )
    return {
        "execution_route_sha256": route,
        "time_authority": "absolute_tick_v1",
        "absolute_tick": next_tick,
        "next_absolute_tick": next_tick,
        "source_import_projection_digest_sha256s": state,
        "parent_checkpoint_projection_digest_sha256s": None,
        "terminal_projection_digest_sha256s": state,
        "canonical_terminal_state_digest_sha256s": state,
        "canonical_checkpoint_storage_digest_sha256s": storage,
        "checkpoint_identity_contract": contract,
        "checkpoint_identity_sha256": _checkpoint_identity_sha256_v1(contract),
    }


def test_absolute_checkpoint_identity_recomputes_and_rejects_all_facade_tamper() -> (
    None
):
    checkpoint = _checkpoint(500)
    assert _validate_tick_native_checkpoint_identity_v1(
        checkpoint,
        execution_route_sha256="a" * 64,
    ) == 500

    mutations = []
    for field, value in (
        ("absolute_tick", 501),
        ("next_absolute_tick", 501),
        ("time_authority", "generation_v1"),
    ):
        mutated = deepcopy(checkpoint)
        mutated[field] = value
        mutations.append(mutated)
    mutated_contract = deepcopy(checkpoint)
    mutated_contract["checkpoint_identity_contract"]["legacy_compatibility"] = {
        "generation": 3,
        "tick": 20,
        "interval_ticks": 160,
    }
    mutations.append(mutated_contract)
    mutated_digest = deepcopy(checkpoint)
    mutated_digest["checkpoint_identity_sha256"] = "d" * 64
    mutations.append(mutated_digest)

    for mutated in mutations:
        with pytest.raises(CampaignProviderError):
                _validate_tick_native_checkpoint_identity_v1(
                    mutated,
                    execution_route_sha256="a" * 64,
                )


@pytest.mark.parametrize("next_tick", (161, 1120, 1121, 5000))
def test_checkpoint_identity_and_restore_ignore_removed_compatibility_metadata(
    next_tick: int,
) -> None:
    checkpoint = _checkpoint(next_tick)
    without_compatibility = deepcopy(checkpoint)
    without_compatibility["checkpoint_identity_contract"].pop(
        "legacy_compatibility", None
    )

    assert checkpoint["checkpoint_identity_sha256"] == without_compatibility[
        "checkpoint_identity_sha256"
    ]
    assert _checkpoint_identity_sha256_v1(
        checkpoint["checkpoint_identity_contract"]
    ) == _checkpoint_identity_sha256_v1(
        without_compatibility["checkpoint_identity_contract"]
    )
    assert _validate_tick_native_checkpoint_identity_v1(
        checkpoint,
        execution_route_sha256="a" * 64,
    ) == next_tick
    assert _validate_tick_native_checkpoint_identity_v1(
        without_compatibility,
        execution_route_sha256="a" * 64,
    ) == next_tick


def test_pre_g1_checkpoint_identity_remains_readable_after_compatibility_removal() -> (
    None
):
    checkpoint = _checkpoint(500)
    state = checkpoint["source_import_projection_digest_sha256s"]
    storage = checkpoint["canonical_checkpoint_storage_digest_sha256s"]
    historical = _checkpoint_identity_contract_v1(
        execution_route_sha256="a" * 64,
        generation=3,
        tick=20,
        next_absolute_tick=500,
        compatibility_interval_ticks=160,
        source_import_projection_digest_sha256s=state,
        parent_checkpoint_projection_digest_sha256s=None,
        terminal_projection_digest_sha256s=state,
        canonical_terminal_state_digest_sha256s=state,
        canonical_checkpoint_storage_digest_sha256s=storage,
        _legacy_compatibility_in_provenance=True,
    )
    checkpoint["checkpoint_identity_contract"] = historical
    checkpoint["checkpoint_identity_sha256"] = _canonical_sha256(historical)

    with pytest.raises(
        CampaignProviderError,
        match="compatibility requires an explicit interval",
    ):
        _validate_tick_native_checkpoint_identity_v1(
            checkpoint,
            execution_route_sha256="a" * 64,
        )
    assert _validate_tick_native_checkpoint_identity_v1(
        checkpoint,
        execution_route_sha256="a" * 64,
        compatibility_interval_ticks=160,
    ) == 500
    checkpoint["checkpoint_identity_contract"].pop("legacy_compatibility")
    assert _validate_tick_native_checkpoint_identity_v1(
        checkpoint,
        execution_route_sha256="a" * 64,
        compatibility_interval_ticks=160,
    ) == 500


def test_resume_reader_derives_historical_compatibility_from_source_window() -> None:
    provider = ProductionEngineLifecycleV1()
    request = replace(
        _native_request(501, checkpoint_ticks=(500,)),
        execution_route_sha256="a" * 64,
    )
    plan = {
        "campaign_id": "historical-checkpoint-reader",
        "plan_sha256": "1" * 64,
        "qualification_key_sha256": "2" * 64,
        "artifacts": {},
    }
    checkpoint = _checkpoint(500)
    state = checkpoint["source_import_projection_digest_sha256s"]
    storage = checkpoint["canonical_checkpoint_storage_digest_sha256s"]
    historical = _checkpoint_identity_contract_v1(
        execution_route_sha256="a" * 64,
        generation=3,
        tick=20,
        next_absolute_tick=500,
        compatibility_interval_ticks=160,
        source_import_projection_digest_sha256s=state,
        parent_checkpoint_projection_digest_sha256s=None,
        terminal_projection_digest_sha256s=state,
        canonical_terminal_state_digest_sha256s=state,
        canonical_checkpoint_storage_digest_sha256s=storage,
        _legacy_compatibility_in_provenance=True,
    )
    checkpoint.update(
        {
            "schema_version": provider.checkpoint_schema_version,
            "checkpoint_binding": provider._checkpoint_binding(plan, request),
            "continuation_storage_mode": request.checkpoint_storage_mode,
            "checkpoint_identity_contract": historical,
            "checkpoint_identity_sha256": _canonical_sha256(historical),
            "cell_bundles": [],
        }
    )

    # Reaching the bundle-order check proves the resume reader accepted the
    # historical identity using the request's explicit positive source window.
    with pytest.raises(CampaignProviderError, match="checkpoint cell order"):
        provider.resume_batch(plan, request, checkpoint, SimpleNamespace())


@pytest.mark.parametrize("interval", (0, -1, True))
def test_historical_checkpoint_identity_rejects_invalid_compatibility_interval(
    interval: int,
) -> None:
    checkpoint = _checkpoint(500)

    with pytest.raises(CampaignProviderError, match="compatibility interval is invalid"):
        _validate_tick_native_checkpoint_identity_v1(
            checkpoint,
            execution_route_sha256="a" * 64,
            compatibility_interval_ticks=interval,
        )


def test_historical_checkpoint_identity_rejects_conflicting_compatibility() -> None:
    checkpoint = _checkpoint(500)
    checkpoint["checkpoint_identity_contract"]["legacy_compatibility"] = {
        "generation": 3,
        "tick": 19,
        "interval_ticks": 160,
    }

    with pytest.raises(
        CampaignProviderError,
        match="compatibility projection is inconsistent",
    ):
        _validate_tick_native_checkpoint_identity_v1(
            checkpoint,
            execution_route_sha256="a" * 64,
            compatibility_interval_ticks=160,
        )


def test_awkward_tick_schedules_form_distinct_receipted_boundaries() -> None:
    request = _native_request(
        1121,
        checkpoint_ticks=(500, 1121),
        report_every_ticks=100,
        science_sample_interval_ticks=137,
        reducer_every_ticks=211,
        archive_every_ticks=251,
    )
    segments = ProductionEngineLifecycleV1._absolute_tick_segments_v1(
        request, start_next_tick=0
    )
    firings = {
        row["end_next_tick"]: tuple(row["fired_schedules"])
        for row in segments
        if row["fired_schedules"]
    }

    assert "report" in firings[100]
    assert "science_sample" in firings[137]
    assert "reducer" in firings[211]
    assert "archive" in firings[251]
    assert set(firings[500]) == {"report", "checkpoint"}
    assert firings[1120] == ("owner",)
    assert set(firings[1121]) == {"checkpoint", "terminal"}
    assert 160 not in {row["end_next_tick"] for row in segments}
    assert 320 not in {row["end_next_tick"] for row in segments}
    assert [row["end_next_tick"] for row in segments if row["publish_checkpoint"]] == [
        500,
        1121,
    ]


def test_checkpoint_at_lifecycle_boundary_keeps_absolute_snapshot_sequence() -> None:
    assert _checkpoint_snapshot_sequence_v1(generation=1, absolute_next_tick=160) == 160
    assert _checkpoint_snapshot_sequence_v1(generation=1, absolute_next_tick=None) == 1


def test_absolute_tick_observer_continuation_is_accepted_by_fresh_resume_validator() -> None:
    generation = 7
    snapshot_sequence = 1120
    continuation = ProductionEngineLifecycleV1._observer_continuation(
        generation,
        tick=0,
        event_cursor=0,
        snapshot_sequence=snapshot_sequence,
        absolute_tick_authority=True,
    )

    restored = _validated_observer_continuation(
        continuation,
        generation=generation,
        event_cursor=0,
        snapshot_sequence=snapshot_sequence,
    )
    assert restored["schedule_phase"] == {
        "origin_generation": generation,
        "next_generation": generation,
    }

    invalid = deepcopy(continuation)
    invalid["schedule_phase"]["origin_generation"] = generation - 1
    with pytest.raises(
        NativeRunControllerError, match="absolute snapshot schedule phase"
    ):
        _validated_observer_continuation(
            invalid,
            generation=generation,
            event_cursor=0,
            snapshot_sequence=snapshot_sequence,
        )


def test_lazy_projection_sequence_loads_each_bundle_once() -> None:
    loads: list[str] = []

    def load(path: str) -> dict[str, str]:
        loads.append(path)
        return {"path": path}

    projections = _LazyProjectionBundleSequence(("one", "two"), load)
    assert projections[0] is projections[0]
    assert projections[-1] is projections[1]
    assert loads == ["one", "two"]


def test_resident_tensor_measurement_counts_shared_storage_once() -> None:
    owner = torch.zeros((8,), dtype=torch.float32)
    view = owner.view(2, 4)
    other = torch.zeros((3,), dtype=torch.int64)

    measured = ProductionEngineLifecycleV1._resident_tensor_storage_bytes_v1(
        (owner, {"view": view, "other": other})
    )

    assert measured["all_tensor_bytes"] == owner.untyped_storage().nbytes() + (
        other.untyped_storage().nbytes()
    )
    assert measured["cuda_tensor_bytes"] == 0


def test_resume_reconciles_all_authenticated_event_prefixes_once(monkeypatch) -> None:
    calls: list[tuple[str, dict[int, object]]] = []

    def reconcile(path: str, *, references_by_cell):
        calls.append((path, dict(references_by_cell)))
        return {
            "schema_version": "eiel_event_sink_reconciliation_v1",
            "path": path,
            "checkpoint_frontiers": {0: 3, 1: 5},
            "suffix_records_removed": 1,
            "ahead_image_preserved": True,
        }

    monkeypatch.setattr(
        "engine.perf.cuda_architecture_b_event_sink_v1."
        "reconcile_event_sink_to_checkpoint_prefixes_v1",
        reconcile,
    )
    projections = tuple(
        {
            "metadata": {
                "event_history_storage_mode": "eiel_resident_event_sink_v1",
                "event_history_reference": {
                    "schema_version": "eiel_resident_event_sink_v1",
                    "path": "/tmp/eiel-events.bin",
                    "cell_index": cell,
                    "row_count": rows,
                    "prefix_sha256": "a" * 64,
                },
            }
        }
        for cell, rows in ((0, 3), (1, 5))
    )

    receipt = _reconcile_tick_native_resume_event_sink_v1(projections, expected_cells=2)
    assert receipt["suffix_records_removed"] == 1
    assert len(calls) == 1
    assert calls[0][0] == "/tmp/eiel-events.bin"
    assert set(calls[0][1]) == {0, 1}


def test_binding_firewall_requires_explicit_absolute_cursor() -> None:
    prepared = SimpleNamespace(
        wdl=SimpleNamespace(initial_projection={"metadata": {"tick": 159}})
    )
    with pytest.raises(RuntimeError, match="authoritative absolute cursor"):
        _validated_tick_native_abi_span_v1(prepared, 1)
    with pytest.raises(RuntimeError, match="positive span"):
        _validated_tick_native_abi_span_v1(
            prepared,
            0,
            absolute_start_next_tick=159,
            authoritative_next_absolute_tick=159,
        )


def test_binding_firewall_accepts_explicit_cursor_for_reused_owner() -> None:
    prepared = SimpleNamespace(
        wdl=SimpleNamespace(initial_projection={"metadata": {"tick": 0}})
    )
    assert (
        _validated_tick_native_abi_span_v1(
            prepared,
            1,
            absolute_start_next_tick=160,
            authoritative_next_absolute_tick=160,
        )
        == 1
    )
    with pytest.raises(RuntimeError, match="authoritative absolute cursor"):
        _validated_tick_native_abi_span_v1(
            prepared,
            1,
            absolute_start_next_tick=160,
        )
    with pytest.raises(RuntimeError, match="authoritative absolute cursor"):
        _validated_tick_native_abi_span_v1(
            prepared,
            1,
            absolute_start_next_tick=160,
            authoritative_next_absolute_tick=159,
        )
    assert (
        _validated_tick_native_abi_span_v1(
            prepared,
            1120,
            absolute_start_next_tick=319,
            authoritative_next_absolute_tick=319,
        )
        == 1120
    )


def test_tick_native_owner_rebinds_missing_abi_sink_from_authenticated_projection(
    tmp_path,
) -> None:
    sink_path = tmp_path / "events.bin"
    patch_config_index = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("patch_p2_config")
    arguments = [None] * len(DIRECT_RESIDENT_ARGUMENT_NAMES_V1)
    arguments[patch_config_index] = {"existing": "authority"}
    prepared = SimpleNamespace(
        patch=SimpleNamespace(
            source=SimpleNamespace(patch_p2_config={}),
        ),
        source_projections=(
            {
                "metadata": {
                    "event_history_storage_mode": "eiel_resident_event_sink_v1",
                    "event_history_reference": {"path": str(sink_path)},
                }
            },
        ),
    )

    bound_path, repaired = _bind_tick_native_event_sink_arguments_v1(
        prepared, arguments
    )

    assert bound_path == str(sink_path.resolve())
    assert repaired is True
    assert arguments[patch_config_index]["existing"] == "authority"
    assert arguments[patch_config_index]["event_history_sink_path"] == bound_path


def test_tick_native_owner_rejects_conflicting_abi_sink() -> None:
    patch_config_index = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("patch_p2_config")
    arguments = [None] * len(DIRECT_RESIDENT_ARGUMENT_NAMES_V1)
    arguments[patch_config_index] = {
        "event_history_sink_path": "/tmp/wrong-events.bin"
    }
    prepared = SimpleNamespace(
        patch=SimpleNamespace(
            source=SimpleNamespace(
                patch_p2_config={"event_history_sink_path": "/tmp/events.bin"}
            ),
        ),
        source_projections=(),
    )

    with pytest.raises(RuntimeError, match="different event sinks"):
        _bind_tick_native_event_sink_arguments_v1(prepared, arguments)


def test_tick_native_owner_accepts_only_bounded_native_sink_result(tmp_path) -> None:
    sink_path = tmp_path / "events.bin"
    result = {
        "event_history_storage_mode": "eiel_resident_event_sink_v1",
        "event_history_reference": {"path": str(sink_path)},
        "event_drain_receipts": [{"sequence": 0}],
        "event_history_frontier_by_cell": [0],
        "event_pending_rows_by_cell": [0],
        "host_event_materialized_bytes": 0,
        "native_event_sink_attestation_schema_version": (
            "eiel_native_event_sink_attestation_v1"
        ),
        "native_event_sink_last_attested_stage": "resident_window_controller_v1",
        "native_event_sink_abi_argument_index": 79,
        "native_event_sink_path": str(sink_path),
    }

    _require_tick_native_event_sink_result_v1(
        result,
        sink_path=str(sink_path.resolve()),
        require_native_attestation=True,
    )

    with pytest.raises(RuntimeError, match="inline cumulative"):
        _require_tick_native_event_sink_result_v1(
            {**result, "event_histories": []},
            sink_path=str(sink_path.resolve()),
            require_native_attestation=True,
        )
    with pytest.raises(RuntimeError, match="materialized event history"):
        _require_tick_native_event_sink_result_v1(
            {**result, "host_event_materialized_bytes": 141},
            sink_path=str(sink_path.resolve()),
            require_native_attestation=True,
        )


def test_tick_native_event_history_requires_resident_sink_at_both_seams() -> None:
    with pytest.raises(CampaignProviderError, match="event_history_sink_path"):
        _require_tick_native_event_sink_builder_v1(
            tick_native=True, accepts_event_sink=False
        )
    _require_tick_native_event_sink_builder_v1(
        tick_native=False, accepts_event_sink=False
    )

    _validate_tick_native_direct_event_storage_v1(
        {"event_history_storage_mode": "eiel_resident_event_sink_v1"},
        tick_native=True,
    )
    with pytest.raises(CampaignProviderError, match="resident event-history sink"):
        _validate_tick_native_direct_event_storage_v1(
            {"event_history_storage_mode": "EventArchiveV1"},
            tick_native=True,
        )
    _validate_tick_native_direct_event_storage_v1(
        {"event_history_storage_mode": "EventArchiveV1"},
        tick_native=False,
    )


def test_allocator_projection_is_horizon_and_generation_facade_invariant() -> None:
    base = {
        "time_authority": "absolute_tick_v1",
        "total_ticks": 5000,
        "resident_owner_span_ticks": 1120,
        "native_dispatch_window_ticks": 1120,
        "source_window_ticks": 160,
        "event_window_capacity": 65536,
    }
    long = {**base, "total_ticks": 500000}
    assert _tick_native_allocator_projection_v1(base) == (7, 160)
    assert _tick_native_allocator_projection_v1(long) == (7, 160)

    for field in (
        "generations",
        "generation_count",
        "ticks_per_generation",
        "physical_window_generations",
        "internal_lifecycle_schedule",
        "internal_lifecycle_interval_ticks",
        "refill_chunk_ticks",
        "legacy_turnover_interval_ticks",
    ):
        with pytest.raises(
            CudaArchitectureBDirectResidentSourceAllocatorError,
            match="retired authority",
        ):
            _tick_native_allocator_projection_v1({**base, field: 7})


def test_tick_native_private_source_capacity_is_horizon_invariant() -> None:
    base = {
        "time_authority": "absolute_tick_v1",
        "total_ticks": 5000,
        "resident_owner_span_ticks": 1120,
        "native_dispatch_window_ticks": 1120,
        "source_window_ticks": 160,
        "event_window_capacity": 65536,
    }
    assert _tick_native_private_source_capacity_window_count_v1(base) == 7
    assert _tick_native_private_source_capacity_window_count_v1(
        {**base, "total_ticks": 500000}
    ) == 7
