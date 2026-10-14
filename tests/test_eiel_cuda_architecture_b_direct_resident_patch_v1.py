from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_patch_v1 as patch
from engine.perf.cuda_architecture_b_checkpoint import CudaBoundaryCheckpoint
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    DirectResidentTerminalSettlementV1,
    GENERATION_SUMMARY_COLUMNS,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    DirectResidentSourceAllocationRequestV1,
)
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EVENT_SINK_ROW_BYTES,
    EVENT_SINK_SCHEMA_VERSION,
    append_event_sink_columns_v1,
)
from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)


def _request(
    projection: Any, *, physical_window_generations: int = 2
) -> DirectResidentSourceAllocationRequestV1:
    return DirectResidentSourceAllocationRequestV1(
        envelope=SimpleNamespace(),
        qualified_cuda_module=object(),
        cuda_capability={},
        import_projection=projection,
        cuda_boundary=object(),
        generation_birth_state=object(),
        configuration=SimpleNamespace(),
        cuda_device="cuda:0",
        physical_window_generations=physical_window_generations,
    )


def _m0_event_input(
    *,
    generation_base: int = 8,
    cells: int = 2,
    generations: int = 2,
    prior_prefix_rows: int = 3,
) -> dict[str, Any]:
    summary = torch.zeros(
        (generations, cells, len(GENERATION_SUMMARY_COLUMNS)),
        dtype=torch.float64,
    )
    summary_index = {
        name: offset for offset, name in enumerate(GENERATION_SUMMARY_COLUMNS)
    }
    histories: list[tuple[torch.Tensor, ...]] = []
    rows_by_cell: list[int] = []
    for cell in range(cells):
        event_types = [2] * prior_prefix_rows
        event_generations = [generation_base - 1] * prior_prefix_rows
        cursor = prior_prefix_rows
        for offset in range(generations):
            generation = generation_base + offset
            ordinary_types = [1, 2, 3, 4]
            turnover_types = [1, 2]
            cursor_start = cursor
            cursor_pre_turnover = cursor_start + len(ordinary_types)
            cursor = cursor_pre_turnover + len(turnover_types)
            event_types.extend((*ordinary_types, *turnover_types))
            event_generations.extend(
                [generation] * (len(ordinary_types) + len(turnover_types))
            )
            summary[offset, cell, summary_index["generation"]] = generation
            summary[offset, cell, summary_index["event_cursor_start"]] = (
                cursor_start
            )
            summary[offset, cell, summary_index["event_cursor_pre_turnover"]] = (
                cursor_pre_turnover
            )
            summary[offset, cell, summary_index["event_cursor_end"]] = cursor
            summary[offset, cell, summary_index["births"]] = 1
            summary[offset, cell, summary_index["deaths"]] = 1
            summary[offset, cell, summary_index["failed_births"]] = 1

        row_count = len(event_types)
        event_values: dict[str, list[int | float]] = {
            name: [] for name in EVENT_COLUMN_NAMES
        }
        for row, (event_type, generation) in enumerate(
            zip(event_types, event_generations, strict=True)
        ):
            tick = row
            actor = 1_000 + row
            actor_slot = row
            target = 2_000 + row if event_type == 1 else -1
            target_slot = 100 + row if event_type == 1 else -1
            parent = -1
            child = -1
            reason = -1
            success = 1
            value = 0.5
            if event_type == 1:
                parent = actor
                child = target
            elif event_type == 2:
                target = actor
                target_slot = actor_slot
                parent = 3_000 + row
                child = actor
            elif event_type == 3:
                reason = 1
                success = 0
                value = float(reason)
            elif event_type == 4:
                reason = 1
                value = float(reason)
            event_values["event_tape_type"].append(event_type)
            event_values["event_tape_generation"].append(generation)
            event_values["event_tape_tick"].append(tick)
            event_values["event_tape_actor_id"].append(actor)
            event_values["event_tape_target_id"].append(target)
            event_values["event_tape_value"].append(value)
            event_values["typed_lifecycle_event_event_type"].append(event_type)
            event_values["typed_lifecycle_event_generation"].append(generation)
            event_values["typed_lifecycle_event_tick"].append(tick)
            event_values["typed_lifecycle_event_sequence"].append(row)
            event_values["typed_lifecycle_event_actor_agent_id"].append(actor)
            event_values["typed_lifecycle_event_target_agent_id"].append(target)
            event_values["typed_lifecycle_event_actor_slot"].append(actor_slot)
            event_values["typed_lifecycle_event_target_slot"].append(target_slot)
            event_values["typed_lifecycle_event_parent_agent_id"].append(parent)
            event_values["typed_lifecycle_event_child_agent_id"].append(child)
            event_values["typed_lifecycle_event_lineage_founder_id"].append(actor)
            event_values["typed_lifecycle_event_lineage_depth"].append(1)
            event_values["typed_lifecycle_event_subtype_code"].append(event_type)
            event_values["typed_lifecycle_event_reason_code"].append(reason)
            event_values["typed_lifecycle_event_success"].append(success)
            event_values["typed_lifecycle_event_value"].append(value)
        columns: list[torch.Tensor] = []
        for name in EVENT_COLUMN_NAMES:
            value = torch.tensor(event_values[name], dtype=EVENT_COLUMN_DTYPES[name])
            columns.append(value.contiguous())
        histories.append(tuple(columns))
        rows_by_cell.append(row_count)
    return {
        "generation_summary_scalar": summary,
        "event_histories": tuple(histories),
        "event_rows_by_cell": tuple(rows_by_cell),
    }


def test_runner_factory_builds_one_normalized_request_per_ordered_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projections = ({"cell": "a"}, {"cell": "b"})
    calls: list[tuple[Any, dict[str, Any]]] = []
    allocated: list[tuple[Any, tuple[Any, ...]]] = []
    prepared = SimpleNamespace(physical_window_arguments=("abi",))

    def factory(runner: Any, **kwargs: Any) -> DirectResidentSourceAllocationRequestV1:
        calls.append((runner, kwargs))
        return _request(projections[kwargs["cell_index"]])

    def allocate(
        primary: Any, *, per_cell_requests: Any, per_cell_ids: Any
    ) -> object:
        assert tuple(per_cell_ids) == ("seed-11", "seed-29")
        allocated.append((primary, tuple(per_cell_requests)))
        return object()

    monkeypatch.setattr(
        patch,
        "_validated_source_projections",
        lambda values, **_kwargs: tuple(values),
    )
    monkeypatch.setattr(patch, "allocate_direct_resident_source_v1", allocate)
    monkeypatch.setattr(
        patch, "prepare_direct_resident_trajectory_v1", lambda source: prepared
    )

    result = patch.prepare_direct_resident_patch_multicell_v1(
        cell_ids=("seed-11", "seed-29"),
        runners=("runner-a", "runner-b"),
        request_factory=factory,
        qualified_cuda_module="module",
        cuda_device="cuda:0",
        logical_generation_count=14,
        agent_id_allocation_generation_count=16,
    )

    assert result.cell_ids == ("seed-11", "seed-29")
    assert result.source_projections == projections
    assert result.physical_window_arguments == ("abi",)
    assert result.resumed is False
    assert [entry[1]["cell_id"] for entry in calls] == ["seed-11", "seed-29"]
    assert [entry[1]["physical_window_generations"] for entry in calls] == [2, 2]
    normalized = allocated[0][1]
    assert [request.batch_cells for request in normalized] == [2, 2]
    assert [request.logical_generation_count for request in normalized] == [14, 14]
    assert [
        request.agent_id_allocation_generation_count for request in normalized
    ] == [16, 16]
    assert [request.physical_window_generations for request in normalized] == [2, 2]


def test_multicell_g1_route_binding_reaches_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projections = ({"cell": "a"}, {"cell": "b"})
    allocated: list[tuple[Any, tuple[Any, ...]]] = []
    prepared = SimpleNamespace(physical_window_arguments=("abi",))

    def allocate(
        primary: Any, *, per_cell_requests: Any, per_cell_ids: Any
    ) -> object:
        allocated.append((primary, tuple(per_cell_requests)))
        return object()

    monkeypatch.setattr(
        patch,
        "_validated_source_projections",
        lambda values, **_kwargs: tuple(values),
    )
    monkeypatch.setattr(patch, "allocate_direct_resident_source_v1", allocate)
    monkeypatch.setattr(
        patch, "prepare_direct_resident_trajectory_v1", lambda source: prepared
    )

    result = patch.prepare_direct_resident_patch_multicell_v1(
        cell_ids=("seed-11", "seed-29"),
        per_cell_requests=(
            _request(projections[0], physical_window_generations=1),
            _request(projections[1], physical_window_generations=1),
        ),
        logical_generation_count=14,
        physical_window_generations=1,
    )

    assert result.physical_window_arguments == ("abi",)
    normalized = allocated[0][1]
    assert [request.physical_window_generations for request in normalized] == [1, 1]


def test_multicell_rejects_mixed_physical_window_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        patch,
        "_validated_source_projections",
        lambda values, **_kwargs: tuple(values),
    )

    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="physical_window_generations",
    ):
        patch.prepare_direct_resident_patch_multicell_v1(
            cell_ids=("seed-11", "seed-29"),
            per_cell_requests=(
                _request({"cell": "a"}, physical_window_generations=1),
                _request({"cell": "b"}, physical_window_generations=2),
            ),
            physical_window_generations=1,
        )


def test_resume_factory_must_bind_the_validated_projection_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resume = {"checkpoint": 1}
    substitute = {"checkpoint": 1}
    monkeypatch.setattr(
        patch,
        "_validated_source_projections",
        lambda values, **_kwargs: tuple(values),
    )

    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="without substitution",
    ):
        patch.prepare_direct_resident_patch_multicell_v1(
            cell_ids=("cell",),
            runners=(object(),),
            request_factory=lambda runner, **kwargs: _request(substitute),
            resume_projections=(resume,),
        )


def test_resume_projections_reorder_only_by_complete_cell_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = {
        "genome_upgrade_v2_checkpoint_projection": {
            "identity": {"cell_id": "first"}
        }
    }
    second = {
        "genome_upgrade_v2_checkpoint_projection": {
            "identity": {"cell_id": "second"}
        }
    }
    monkeypatch.setattr(
        patch,
        "_projection_mapping",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        patch,
        "stack_validated_per_cell_checkpoint_projections_v1",
        lambda *_args, **_kwargs: None,
    )

    ordered = patch._validated_source_projections(
        (second, first),
        batch_cells=2,
        cell_ids=("first", "second"),
        allow_identity_reorder=True,
    )

    assert ordered == (first, second)


def test_resume_projections_reject_partial_cell_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identified = {
        "genome_upgrade_v2_checkpoint_projection": {
            "identity": {"cell_id": "first"}
        }
    }
    monkeypatch.setattr(
        patch,
        "_projection_mapping",
        lambda value, **_kwargs: value,
    )

    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="identity set is partial",
    ):
        patch._validated_source_projections(
            (identified, {}),
            batch_cells=2,
            cell_ids=("first", "second"),
            allow_identity_reorder=True,
        )


def test_multicell_projection_preserves_cell_order_and_flattens_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = 2
    source = SimpleNamespace(
        outputs={
            "generation_summary_scalar": torch.zeros((4, cells, 28)),
            "runtime_bookkeeping_by_tick": torch.zeros((cells, 320, 5)),
            "turnover_status": torch.zeros((cells, 1)),
        },
        generation_base=8,
        logical_generation_count=4,
        ticks_per_generation=160,
    )
    owner = SimpleNamespace(
        source=source,
        live_banks={
            "energy": torch.tensor([[11.0], [29.0]]),
            "genome_v2_alleles_q10": torch.tensor(
                [[[[[0]]]], [[[[1024]]]]], dtype=torch.int64
            ),
            "genome_v2_founder_profiles": (
                {
                    "schema_version": "genome_upgrade_v2_founder_profile_v1",
                    "profile_id": "neutral",
                    "payload_sha256": "1" * 64,
                },
                {
                    "schema_version": "genome_upgrade_v2_founder_profile_v1",
                    "profile_id": "pleiotropic_tradeoffs",
                    "payload_sha256": "2" * 64,
                },
            ),
            "event_columns": [torch.arange(6).reshape(cells, 3)],
        },
        checkpoint_projection_inputs={
            "resident_birth_state": object(),
            "pre_generation_canonical_projections": (
                {"source": 11},
                {"source": 29},
            ),
        },
        physical_window_arguments=(),
    )
    prepared = patch.PreparedDirectResidentPatchMulticellV1(
        schema_version=patch.PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        patch=owner,
        cell_ids=("seed-11", "seed-29"),
        source_projections=({"source": 11}, {"source": 29}),
        resumed=True,
    )
    seen: list[tuple[float, Any, int, bool, str, int]] = []

    def settle(
        context: Any,
        *,
        direct_result: Any,
        capability: Any,
        reporting_enabled: bool,
    ) -> Any:
        energy = float(context.live_banks["energy"].item())
        source_projection = context.checkpoint_projection_inputs[
            "pre_generation_canonical_projection"
        ]
        assert context.source.ticks_per_generation == 160
        seen.append(
            (
                energy,
                source_projection,
                len(direct_result["event_histories"]),
                reporting_enabled,
                context.live_banks["genome_v2_founder_profile"][
                    "payload_sha256"
                ],
                int(
                    context.live_banks["genome_v2_alleles_q10"]
                    .flatten()[0]
                    .item()
                ),
            )
        )
        projection = {"terminal": int(energy)}
        checkpoint = CudaBoundaryCheckpoint(
            projection=projection,
            projection_digest="digest",
            device="cpu",
            device_to_host_operations=0,
            device_to_host_bytes=0,
            device_synchronizations=0,
        )
        return DirectResidentTerminalSettlementV1(
            schema_version="settlement",
            compact_summary_rows=(
                ({"generation": 9}, {"generation": 10})
                if reporting_enabled
                else ()
            ),
            terminal_checkpoint=checkpoint,
            terminal_generation=12,
        )

    monkeypatch.setattr(
        patch, "settle_patch_p2_b1_direct_resident_prefix_v2", settle
    )
    reset_count_maxes: list[int] = []
    monkeypatch.setattr(
        patch,
        "materialize_disabled_prediction_state_v1",
        lambda _projection, state, **kwargs: (
            reset_count_maxes.append(kwargs["reset_count_max"]) or state
        ),
    )
    monkeypatch.setattr(patch, "validate_native_checkpoint_projection", lambda value: None)
    monkeypatch.setattr(
        patch,
        "_with_checkpoint_contract_metadata",
        lambda value: {
            **value,
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        },
    )

    result = patch.project_direct_resident_patch_multicell_checkpoints_v1(
        prepared,
        direct_result={
            "completed_generations": 4,
            "generation_summary_scalar": torch.zeros((4, cells, 28)),
            "runtime_bookkeeping_by_tick": torch.zeros((cells, 320, 5)),
            "event_histories": ((torch.tensor([1]),), (torch.tensor([2]),)),
            "event_rows_by_cell": (1, 1),
            "source_projection_cell_ids": ("seed-11", "seed-29"),
        },
        capability={},
        reporting_suppressed_cell_indices=(1,),
    )

    assert seen == [
        (11.0, {"source": 11}, 1, True, "1" * 64, 0),
        (29.0, {"source": 29}, 1, False, "2" * 64, 1024),
    ]
    assert result.cell_ids == ("seed-11", "seed-29")
    assert result.checkpoint_projections == (
        {
            "terminal": 11,
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        },
        {
            "terminal": 29,
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        },
    )
    assert [row["cell_id"] for row in result.compact_result_rows] == [
        "seed-11",
        "seed-11",
    ]
    assert reset_count_maxes == [160, 160]
    assert result.settlement_event_rows == ()
    assert result.terminal_generation == 12


def test_m0_settlement_event_rows_preserve_resume_prefix_and_partitions() -> None:
    direct_result = _m0_event_input(prior_prefix_rows=3)

    rows = patch._settlement_event_rows_v1(
        direct_result=direct_result,
        cell_ids=("cell-0", "cell-1"),
        generation_base=8,
        completed_generations=2,
    )

    assert [(row["cell_id"], row["generation"]) for row in rows] == [
        ("cell-0", 8),
        ("cell-0", 9),
        ("cell-1", 8),
        ("cell-1", 9),
    ]
    first_events = rows[0]["events"]
    assert first_events["cursor_start"] == 3
    assert first_events["cursor_pre_turnover"] == 7
    assert first_events["cursor_end"] == 9
    assert first_events["generation_counts_by_type"] == {
        "birth_committed": 1,
        "mutation": 1,
        "birth_failed": 1,
        "death_committed": 1,
    }
    assert first_events["turnover_counts_by_type"] == {
        "birth_committed": 1,
        "mutation": 1,
        "birth_failed": 0,
        "death_committed": 0,
    }


def test_m0_settlement_event_rows_stream_external_sink(
    tmp_path: Any,
) -> None:
    direct_result = _m0_event_input(prior_prefix_rows=3)
    sink_path = tmp_path / "events.bin"
    for cell, history in enumerate(direct_result["event_histories"]):
        append_event_sink_columns_v1(
            sink_path,
            cell_index=cell,
            columns=history,
            sequence_base=0,
            expected_cells=2,
        )
    external_result = dict(direct_result)
    external_result.pop("event_histories")
    external_result.update(
        {
            "event_history_storage_mode": EVENT_SINK_SCHEMA_VERSION,
            "event_history_reference": {
                "schema_version": EVENT_SINK_SCHEMA_VERSION,
                "path": str(sink_path),
                "row_bytes": EVENT_SINK_ROW_BYTES,
                "row_count_by_cell": list(direct_result["event_rows_by_cell"]),
                "cell_count": 2,
            },
            "event_history_frontier_by_cell": list(
                direct_result["event_rows_by_cell"]
            ),
        }
    )

    rows = patch._settlement_event_rows_v1(
        direct_result=external_result,
        cell_ids=("cell-0", "cell-1"),
        generation_base=8,
        completed_generations=2,
    )

    assert [(row["cell_id"], row["generation"]) for row in rows] == [
        ("cell-0", 8),
        ("cell-0", 9),
        ("cell-1", 8),
        ("cell-1", 9),
    ]
    assert rows[0]["events"]["cursor_start"] == 3


@pytest.mark.parametrize(
    "case",
    (
        "cursor_gap",
        "sequence_gap",
        "wrong_event_type",
        "wrong_event_generation",
        "legacy_type_mismatch",
        "legacy_generation_mismatch",
        "legacy_tick_mismatch",
        "wrong_subtype",
        "wrong_reason",
        "wrong_success",
        "wrong_identity",
        "wrong_value",
        "invalid_turnover_type",
        "wrong_summary_count",
        "nonfinite_summary",
        "fractional_summary",
        "wrong_column_dtype",
        "reported_rows_mismatch",
    ),
)
def test_m0_settlement_event_rows_fail_closed_on_malformed_authority(
    case: str,
) -> None:
    direct_result = deepcopy(_m0_event_input(prior_prefix_rows=3))
    summary_index = {
        name: offset for offset, name in enumerate(GENERATION_SUMMARY_COLUMNS)
    }
    sequence_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_sequence"
    )
    event_type_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_event_type"
    )
    event_generation_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_generation"
    )
    legacy_type_index = EVENT_COLUMN_NAMES.index("event_tape_type")
    legacy_generation_index = EVENT_COLUMN_NAMES.index("event_tape_generation")
    legacy_tick_index = EVENT_COLUMN_NAMES.index("event_tape_tick")
    subtype_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_subtype_code")
    reason_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_reason_code")
    success_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_success")
    child_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_child_agent_id")
    typed_value_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_value")
    histories = [list(history) for history in direct_result["event_histories"]]
    if case == "cursor_gap":
        direct_result["generation_summary_scalar"][
            1,
            0,
            summary_index["event_cursor_start"],
        ] += 1
    elif case == "sequence_gap":
        histories[0][sequence_index][4] += 1
    elif case == "wrong_event_type":
        histories[0][event_type_index][4] = 9
    elif case == "wrong_event_generation":
        histories[0][event_generation_index][4] = 99
    elif case == "legacy_type_mismatch":
        histories[0][legacy_type_index][4] = 4
    elif case == "legacy_generation_mismatch":
        histories[0][legacy_generation_index][4] = 99
    elif case == "legacy_tick_mismatch":
        histories[0][legacy_tick_index][4] = 99
    elif case == "wrong_subtype":
        histories[0][subtype_index][4] = 4
    elif case == "wrong_reason":
        histories[0][reason_index][4] = 2
    elif case == "wrong_success":
        histories[0][success_index][4] = 0
    elif case == "wrong_identity":
        histories[0][child_index][4] = -1
    elif case == "wrong_value":
        histories[0][typed_value_index][4] = float("nan")
    elif case == "invalid_turnover_type":
        histories[0][event_type_index][7] = 3
        histories[0][legacy_type_index][7] = 3
        histories[0][subtype_index][7] = 3
        histories[0][reason_index][7] = 1
        histories[0][success_index][7] = 0
    elif case == "wrong_summary_count":
        direct_result["generation_summary_scalar"][
            0,
            0,
            summary_index["births"],
        ] = 2
    elif case == "nonfinite_summary":
        direct_result["generation_summary_scalar"][
            0,
            0,
            summary_index["births"],
        ] = float("nan")
    elif case == "fractional_summary":
        direct_result["generation_summary_scalar"][
            0,
            0,
            summary_index["births"],
        ] = 1.5
    elif case == "wrong_column_dtype":
        histories[0][event_type_index] = histories[0][event_type_index].to(
            torch.int64
        )
    elif case == "reported_rows_mismatch":
        rows = list(direct_result["event_rows_by_cell"])
        rows[0] += 1
        direct_result["event_rows_by_cell"] = tuple(rows)
    direct_result["event_histories"] = tuple(
        tuple(history) for history in histories
    )

    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="M0 (settlement|generation summary)",
    ):
        patch._settlement_event_rows_v1(
            direct_result=direct_result,
            cell_ids=("cell-0", "cell-1"),
            generation_base=8,
            completed_generations=2,
        )


def test_projection_rejects_completed_count_beyond_prepared_horizon() -> None:
    prepared = patch.PreparedDirectResidentPatchMulticellV1(
        schema_version=patch.PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        patch=SimpleNamespace(source=SimpleNamespace(logical_generation_count=4)),
        cell_ids=("cell",),
        source_projections=({},),
        resumed=False,
    )
    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="prepared owner horizon",
    ):
        patch.project_direct_resident_patch_multicell_checkpoints_v1(
            prepared,
            direct_result={"completed_generations": 14},
            capability={},
        )


def test_projection_forwards_disabled_turnover_lifecycle_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = SimpleNamespace(
        source=SimpleNamespace(
            outputs={},
            configuration={
                "generation_turnover_enabled": True,
                "lifecycle_mode": 0,
                "sentinel": "preserved",
            },
            generation_base=3,
            logical_generation_count=4,
        ),
        live_banks={},
        checkpoint_projection_inputs={
            "pre_generation_canonical_projections": (
                {"metadata": {}},
            ),
        },
    )
    source_projection = {
        "metadata": {
            "generation_turnover_enabled": False,
            "lifecycle_mode_abi": 1,
        }
    }
    prepared = patch.PreparedDirectResidentPatchMulticellV1(
        schema_version=patch.PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        patch=owner,
        cell_ids=("cell-a",),
        source_projections=(source_projection,),
        resumed=True,
    )
    captured: dict[str, Any] = {}

    def settle(
        context: Any,
        *,
        direct_result: Any,
        capability: Any,
        reporting_enabled: bool,
    ) -> Any:
        captured.update(context.source.configuration)
        checkpoint = CudaBoundaryCheckpoint(
            projection={"metadata": {}},
            projection_digest="digest",
            device="cpu",
            device_to_host_operations=0,
            device_to_host_bytes=0,
            device_synchronizations=0,
        )
        return DirectResidentTerminalSettlementV1(
            schema_version="settlement",
            compact_summary_rows=(),
            terminal_checkpoint=checkpoint,
            terminal_generation=7,
        )

    monkeypatch.setattr(
        patch, "settle_patch_p2_b1_direct_resident_prefix_v2", settle
    )
    monkeypatch.setattr(
        patch,
        "materialize_disabled_prediction_state_v1",
        lambda _projection, state, **_kwargs: state,
    )
    monkeypatch.setattr(patch, "validate_native_checkpoint_projection", lambda value: None)
    monkeypatch.setattr(patch, "_with_checkpoint_contract_metadata", lambda value: value)
    patch.project_direct_resident_patch_multicell_checkpoints_v1(
        prepared,
        direct_result={
            "completed_generations": 4,
            "generation_summary_scalar": torch.zeros((4, 1, 28)),
            "runtime_bookkeeping_by_tick": torch.zeros((1, 320, 5)),
            "event_histories": ((),),
            "event_rows_by_cell": (0,),
        },
        capability={},
    )

    assert captured["generation_turnover_enabled"] is False
    assert captured["lifecycle_mode"] == 1
    assert captured["sentinel"] == "preserved"


def test_lifecycle_overrides_accept_checkpoint_object_metadata() -> None:
    projection = CudaBoundaryCheckpoint(
        projection={
            "metadata": {
                "generation_turnover_enabled": False,
                "lifecycle_mode_abi": 1,
            }
        },
        projection_digest="digest",
        device="cpu",
        device_to_host_operations=0,
        device_to_host_bytes=0,
        device_synchronizations=0,
    )

    assert patch._projection_lifecycle_configuration_overrides_v1(
        projection
    ) == {
        "generation_turnover_enabled": False,
        "lifecycle_mode": 1,
    }


def test_settlement_output_slicing_uses_only_declared_cell_axes() -> None:
    outputs = {
        "generation_summary_scalar": torch.arange(4 * 2 * 3).reshape(4, 2, 3),
        "runtime_bookkeeping_by_tick": torch.arange(2 * 5 * 2).reshape(2, 5, 2),
        "birth_reason_counts": torch.arange(2 * 5 * 10).reshape(2, 5, 10),
        "turnover_status": torch.tensor([11, 29]),
        # This has a coincidental leading b2 extent but is not a settlement
        # field and must not silently cross the boundary.
        "unregistered_shared_tensor": torch.tensor([[1, 2], [3, 4]]),
    }

    selected = patch._slice_outputs(outputs, cell=1, cells=2)

    assert set(selected) == {
        "generation_summary_scalar",
        "runtime_bookkeeping_by_tick",
        "birth_reason_counts",
        "turnover_status",
    }
    assert torch.equal(selected["generation_summary_scalar"], outputs[
        "generation_summary_scalar"
    ][:, 1:2])
    assert torch.equal(
        selected["runtime_bookkeeping_by_tick"],
        outputs["runtime_bookkeeping_by_tick"][1:2],
    )
    assert torch.equal(
        selected["birth_reason_counts"],
        outputs["birth_reason_counts"][1:2],
    )
    assert torch.equal(selected["turnover_status"], torch.tensor([29]))


def test_settlement_output_slicing_rejects_wrong_declared_axis() -> None:
    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="wrong cell axis",
    ):
        patch._slice_outputs(
            {"generation_summary_scalar": torch.zeros((4, 3, 28))},
            cell=0,
            cells=2,
        )


def test_partial_tick_direct_result_does_not_require_generation_summary() -> None:
    result = {
        "absolute_tick_span_native_owner": True,
        "runtime_bookkeeping_by_tick": torch.zeros((2, 17, 5), dtype=torch.int64),
    }

    selected = patch._slice_direct_result(result, cell=1, cells=2)

    assert "generation_summary_scalar" not in selected
    assert selected["runtime_bookkeeping_by_tick"].shape == (1, 17, 5)
    assert "event_histories" not in selected
    assert "event_rows_by_cell" not in selected


def test_provider_native_segment_artifact_splits_mixed_success_failure_rows() -> None:
    artifact, _metadata = patch._provider_native_segment_artifact_v1(
        cell_id="cell-0",
        cell_index=0,
        source_projection={
            "metadata": {
                "native_next_segment_index": 0,
                "native_chain_sha256": "0" * 64,
            }
        },
        compact_rows=(
            {"generation": 0, "alive": 122, "births": 58, "failed_births": 0},
            {"generation": 1, "alive": 238, "births": 116, "failed_births": 475},
            {"generation": 2, "alive": 256, "births": 18, "failed_births": 14673},
            {"generation": 3, "alive": 256, "births": 0, "failed_births": 20480},
        ),
    )

    cell = artifact["cell"]
    assert cell["attempt_count"] == 6
    assert cell["success_count"] == 3
    assert cell["failure_count"] == 3
    assert cell["success_count"] + cell["failure_count"] == cell["attempt_count"]
    assert [attempt["success"] for attempt in cell["attempts"]] == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_provider_native_segment_artifact_preserves_cumulative_counts() -> None:
    artifact, updated_projection = patch._provider_native_segment_artifact_v1(
        cell_id="cell-0",
        cell_index=0,
        source_projection={
            "metadata": {
                "native_next_segment_index": 1,
                "native_chain_sha256": "a" * 64,
                "native_cumulative_attempt_count": 5,
                "native_cumulative_success_count": 3,
                "native_cumulative_failure_count": 2,
                "native_cumulative_snapshot_count": 0,
            }
        },
        compact_rows=(
            {"generation": 3, "alive": 43, "births": 0, "failed_births": 0},
            {"generation": 4, "alive": 43, "births": 1, "failed_births": 0},
        ),
    )

    metadata = updated_projection["metadata"]
    assert artifact["segment_index"] == 1
    assert artifact["cell"]["cumulative_counts_after_segment"] == {
        "attempts": 7,
        "successes": 4,
        "failures": 3,
        "eligible_snapshots": 0,
    }
    assert metadata["native_next_segment_index"] == 2
    assert metadata["native_cumulative_attempt_count"] == 7
    assert metadata["native_cumulative_success_count"] == 4
    assert metadata["native_cumulative_failure_count"] == 3
    assert metadata["native_cumulative_snapshot_count"] == 0


def test_live_bank_slicing_preserves_genome_v2_observability_config() -> None:
    config = {
        "max_energy": 100.0,
        "patch_density_include_self": True,
    }

    selected = patch._slice_live_banks(
        {
            "energy": torch.tensor([[11.0], [29.0]]),
            "genome_v2_observability_config": config,
            "unregistered_mapping": {"must": "not cross"},
        },
        cell=1,
        cells=2,
    )

    assert torch.equal(selected["energy"], torch.tensor([[29.0]]))
    assert selected["genome_v2_observability_config"] == config
    assert selected["genome_v2_observability_config"] is not config
    assert "unregistered_mapping" not in selected


def test_live_bank_slicing_accepts_compact_mlp_gru_sentinel_cell_axis() -> None:
    selected = patch._slice_live_banks(
        {"gru_pack": torch.zeros((2, 1), dtype=torch.float32)},
        cell=1,
        cells=2,
    )

    assert torch.equal(
        selected["gru_pack"],
        torch.zeros((1, 1), dtype=torch.float32),
    )


def test_generation_zero_projection_attaches_initialized_resident_sidecars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projections = (
        {"metadata": {"generation": 0}, "tensors": {}},
        {"metadata": {"generation": 0}, "tensors": {}},
    )
    owner = SimpleNamespace(
        live_banks={
            "energy": torch.tensor([[11.0], [29.0]]),
            "genome_v2_founder_profiles": (
                {
                    "schema_version": "genome_upgrade_v2_founder_profile_v1",
                    "profile_id": "neutral",
                    "payload_sha256": "1" * 64,
                },
                {
                    "schema_version": "genome_upgrade_v2_founder_profile_v1",
                    "profile_id": "pleiotropic_tradeoffs",
                    "payload_sha256": "2" * 64,
                },
            ),
        },
        checkpoint_projection_inputs={
            "genome_v2_checkpoint_authority_binding": {"package": "bound"}
        },
    )
    prepared = patch.PreparedDirectResidentPatchMulticellV1(
        schema_version=patch.PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        patch=owner,
        cell_ids=("cell-a", "cell-b"),
        source_projections=projections,
        resumed=False,
    )
    calls: list[dict[str, Any]] = []
    promoted: list[Mapping[str, Any]] = []

    def attach(checkpoint, persistent_state, **kwargs):
        calls.append(
            {
                "energy": float(persistent_state["energy"].item()),
                **kwargs,
            }
        )
        return CudaBoundaryCheckpoint(
            **{
                **checkpoint.__dict__,
                "projection": {
                    **checkpoint.projection,
                    "genome_upgrade_v2_checkpoint_projection": {
                        "identity": {"cell_id": kwargs["cell_id"]}
                    },
                },
            }
        )

    def promote(projection):
        promoted.append(projection)
        return {
            **projection,
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
        }

    monkeypatch.setattr(patch, "_projection_mapping", lambda value, **_: value)
    monkeypatch.setattr(patch, "projection_digest", lambda _value: "d" * 64)
    monkeypatch.setattr(
        patch, "validate_native_checkpoint_projection", lambda _value: None
    )
    monkeypatch.setattr(
        patch, "with_resident_inheritance_sidecars_v1", attach
    )
    monkeypatch.setattr(patch, "_with_checkpoint_contract_metadata", promote)

    result = patch.project_direct_resident_patch_multicell_generation_zero_v1(
        prepared
    )

    assert len(promoted) == 2
    assert [
        row["genome_upgrade_v2_checkpoint_projection"]["identity"]["cell_id"]
        for row in result
    ] == ["cell-a", "cell-b"]
    assert calls == [
        {
            "energy": 11.0,
            "cell_id": "cell-a",
            "genome_v2_checkpoint_authority_binding": {"package": "bound"},
            "genome_v2_causal_physical_cell_ordinal": 0,
            "include_genome_v2_causal_projection": False,
        },
        {
            "energy": 29.0,
            "cell_id": "cell-b",
            "genome_v2_checkpoint_authority_binding": {"package": "bound"},
            "genome_v2_causal_physical_cell_ordinal": 1,
            "include_genome_v2_causal_projection": False,
        },
    ]


def test_live_bank_slicing_rejects_stale_founder_profile_collection() -> None:
    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="must match cells",
    ):
        patch._slice_live_banks(
            {
                "genome_v2_founder_profiles": (
                    {
                        "schema_version": "genome_upgrade_v2_founder_profile_v1",
                        "profile_id": "neutral",
                        "payload_sha256": "1" * 64,
                    },
                )
            },
            cell=0,
            cells=2,
        )


def test_live_bank_slicing_rejects_invalid_genome_v2_observability_config() -> None:
    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match="genome_v2_observability_config must be a mapping",
    ):
        patch._slice_live_banks(
            {"genome_v2_observability_config": None},
            cell=0,
            cells=2,
        )


def test_batched_pcg64_buffers_are_validated_before_ordered_settlement() -> None:
    valid = torch.tensor([[0, 0], [1, 0xFFFFFFFF]], dtype=torch.int64)
    patch._validate_live_pcg64_buffers_before_settlement_v1(
        {
            "runner_pcg64_buffer_words": valid,
            "pcg64_buffer_words": valid.clone(),
            "population_pcg64_buffer_words": valid.clone(),
        },
        cell_ids=("refill-logical", "overflow-fault"),
    )

    invalid = valid.clone()
    invalid[0] = torch.tensor([2, -1], dtype=torch.int64)
    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match=(
            r"field=pcg64_buffer_words cell_index=0 "
            r"cell_id=refill-logical values=\[2, -1\]"
        ),
    ):
        patch._validate_live_pcg64_buffers_before_settlement_v1(
            {"pcg64_buffer_words": invalid},
            cell_ids=("refill-logical", "overflow-fault"),
        )


def test_fault_suppressed_pcg64_state_restores_canonical_source() -> None:
    invalid_buffer = torch.tensor([[-1, -1]], dtype=torch.int64)
    invalid_state = torch.full(
        (1, 4), torch.iinfo(torch.uint64).max, dtype=torch.uint64
    )
    source_state = {
        "bit_generator": "PCG64",
        "state": {
            "state": 263079081272045986684532785722382679295,
            "inc": 171730982959198748047597532076080605511,
        },
        "has_uint32": 0,
        "uinteger": 0,
    }
    restored = patch._restore_suppressed_pcg64_state_v1(
        {
            "runner_pcg64_state_words": invalid_state.clone(),
            "runner_pcg64_buffer_words": invalid_buffer.clone(),
            "pcg64_state_words": invalid_state.clone(),
            "pcg64_buffer_words": invalid_buffer.clone(),
            "population_pcg64_state_words": invalid_state.clone(),
            "population_pcg64_buffer_words": invalid_buffer.clone(),
            "policy_np_semantic_draws": torch.tensor(
                [-1], dtype=torch.int64
            ),
            "metrics_np_semantic_draws": torch.tensor(
                [-1], dtype=torch.int64
            ),
        },
        canonical_projection={
            "metadata": {
                "native_runtime_numpy_runner_state": source_state,
                "native_runtime_numpy_policy_state": source_state,
                "native_runtime_numpy_population_state": source_state,
                "native_runtime_rng_manager_stream_indices": {
                    "policy": {"np": 144},
                    "metrics": {"np": 0},
                },
            }
        },
        cell_id="fault-target",
    )

    for prefix in ("runner_", "", "population_"):
        assert restored[f"{prefix}pcg64_buffer_words"].tolist() == [[0, 0]]
        assert not torch.equal(
            restored[f"{prefix}pcg64_state_words"], invalid_state
        )
    assert restored["policy_np_semantic_draws"].tolist() == [144]
    assert restored["metrics_np_semantic_draws"].tolist() == [0]


def test_only_explicit_fault_cell_may_skip_live_pcg64_validation() -> None:
    buffers = torch.tensor([[0, 0], [-1, -1]], dtype=torch.int64)

    patch._validate_live_pcg64_buffers_before_settlement_v1(
        {"runner_pcg64_buffer_words": buffers},
        cell_ids=("peer", "fault-target"),
        excluded_cell_indices=frozenset({1}),
    )
    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match=r"cell_index=1 cell_id=fault-target",
    ):
        patch._validate_live_pcg64_buffers_before_settlement_v1(
            {"runner_pcg64_buffer_words": buffers},
            cell_ids=("peer", "fault-target"),
        )


def test_source_import_cursor_is_ordered_and_keeps_one_payload_live() -> None:
    cell_ids = ("cell-0", "cell-1", "cell-2")
    descriptors = tuple(f"bundle-{index}" for index in range(len(cell_ids)))
    loaded: list[str] = []
    imported: list[tuple[str, str]] = []

    def loader(descriptor: str, index: int, cell_id: str) -> patch.DetachedSourcePayloadV1:
        assert descriptor == f"bundle-{index}"
        loaded.append(cell_id)
        return patch.DetachedSourcePayloadV1(
            schema_version=patch.SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            payload={"cell_id": cell_id},
            source_projection_sha256="a" * 64,
            live_payload_bytes=17,
        )

    def importer(
        payload: Mapping[str, Any], storage: str, index: int, cell_id: str
    ) -> str:
        assert payload["cell_id"] == cell_id
        imported.append((cell_id, storage))
        return f"cuda-{index}"

    cursor = patch.SourceImportCursorV1.begin(
        descriptors,
        cell_ids,
        loader=loader,
        importer=importer,
    )
    assert not isinstance(cursor, Sequence)
    for index, cell_id in enumerate(cell_ids):
        detached = cursor.next_cell()
        assert detached is not None
        assert detached.cell_id == cell_id
        with pytest.raises(
            patch.CudaArchitectureBDirectResidentPatchV1Error,
            match="requires import_and_release",
        ):
            cursor.next_cell()
        assert cursor.import_and_release(cell_id, f"slot-{index}") == f"cuda-{index}"
        assert detached.payload is None
    assert cursor.next_cell() is None
    receipt = cursor.close().payload()
    assert loaded == list(cell_ids)
    assert imported == [(cell_id, f"slot-{index}") for index, cell_id in enumerate(cell_ids)]
    assert receipt["state"] == "SEALED"
    assert receipt["covered_cell_ids"] == list(cell_ids)
    assert receipt["max_live_tensor_bearing_payloads"] == 1
    assert receipt["max_live_payload_bytes"] == 17
    assert receipt["importer_concurrency"] == 1


def test_source_import_cursor_failure_is_unpublished_and_retains_prefix_receipt() -> None:
    cell_ids = ("cell-0", "cell-1")

    def loader(_descriptor: str, index: int, cell_id: str) -> patch.DetachedSourcePayloadV1:
        return patch.DetachedSourcePayloadV1(
            schema_version=patch.SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            payload={"cell_id": cell_id},
            source_projection_sha256="b" * 64,
            live_payload_bytes=23,
        )

    def importer(
        _payload: Mapping[str, Any], _storage: object, index: int, _cell_id: str
    ) -> None:
        if index == 1:
            raise RuntimeError("import failed after cell-0")

    cursor = patch.SourceImportCursorV1.begin(
        ("bundle-0", "bundle-1"),
        cell_ids,
        loader=loader,
        importer=importer,
    )
    first = cursor.next_cell()
    assert first is not None
    cursor.import_and_release(first.cell_id, object())
    second = cursor.next_cell()
    assert second is not None
    with pytest.raises(RuntimeError, match="import failed after cell-0"):
        cursor.import_and_release(second.cell_id, object())
    receipt = cursor.close().payload()
    assert receipt["state"] == "FAILED_UNPUBLISHED"
    assert receipt["covered_cell_ids"] == ["cell-0"]
    assert receipt["failure_reason"] == "import failed after cell-0"


def test_checkpoint_projection_cursor_is_b136_ordered_and_one_cell_bounded() -> None:
    cell_ids = tuple(f"cell-{index:03d}" for index in range(136))
    factory_calls: list[str] = []

    def factory(index: int, cell_id: str) -> patch.DetachedCellProjectionV1:
        assert index == len(factory_calls)
        factory_calls.append(cell_id)
        return patch.DetachedCellProjectionV1(
            schema_version=patch.DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            projection={"cell_id": cell_id, "source_digest": "d" * 64},
            compact_result_rows=({"cell_id": cell_id},),
            segment_artifacts=({"cell_id": cell_id},),
            terminal_generation=15,
            terminal_tick=2_500,
            live_projection_bytes=29,
        )

    cursor = patch.CheckpointProjectionCursorV1.begin(
        owner=object(),
        coordinate=2_500,
        ordered_cell_ids=cell_ids,
        factory=factory,
    )
    for index, cell_id in enumerate(cell_ids):
        detached = cursor.next_cell()
        assert detached is not None
        assert detached.cell_id == cell_id
        assert detached.cell_index == index
        with pytest.raises(
            patch.CudaArchitectureBDirectResidentPatchV1Error,
            match="requires release_cell",
        ):
            cursor.next_cell()
        cursor.release_cell(cell_id)
        assert detached.projection is None

    assert cursor.next_cell() is None
    receipt = cursor.close().payload()
    aggregate = cursor.aggregate_payload()

    assert factory_calls == list(cell_ids)
    assert receipt["state"] == "SEALED"
    assert receipt["ordered_cell_ids"] == list(cell_ids)
    assert receipt["covered_cell_ids"] == list(cell_ids)
    assert receipt["max_live_tensor_bearing_projections"] == 1
    assert receipt["max_live_projection_bytes"] == 29
    assert receipt["writer_concurrency"] == 1
    assert len(aggregate["compact_result_rows"]) == 136
    assert len(aggregate["segment_artifacts"]) == 136


def test_checkpoint_projection_cursor_failure_does_not_seal_or_advance_frontier() -> None:
    cell_ids = tuple(f"cell-{index}" for index in range(4))
    failures: list[str] = []

    def factory(index: int, cell_id: str) -> patch.DetachedCellProjectionV1:
        if index == 2:
            raise RuntimeError("write failed after cell-1")
        return patch.DetachedCellProjectionV1(
            schema_version=patch.DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            projection={"cell_id": cell_id},
            terminal_generation=15,
            terminal_tick=2_500,
            live_projection_bytes=11,
        )

    cursor = patch.CheckpointProjectionCursorV1(
        owner=object(),
        coordinate=2_500,
        ordered_cell_ids=cell_ids,
        factory=factory,
        on_fail=failures.append,
    )
    for cell_id in cell_ids[:2]:
        detached = cursor.next_cell()
        assert detached is not None
        cursor.release_cell(cell_id)
    with pytest.raises(RuntimeError, match="write failed after cell-1"):
        cursor.next_cell()

    receipt = cursor.close().payload()
    assert receipt["state"] == "FAILED_UNPUBLISHED"
    assert receipt["covered_cell_ids"] == ["cell-0", "cell-1"]
    assert receipt["failure_reason"] == "write failed after cell-1"
    assert failures == ["write failed after cell-1"]


@pytest.mark.parametrize(
    "value",
    (
        torch.zeros((2, 2), dtype=torch.int32),
        torch.zeros((1, 2), dtype=torch.int64),
        torch.zeros((2, 3), dtype=torch.int64),
    ),
)
def test_batched_pcg64_buffer_shape_and_dtype_fail_closed(value: torch.Tensor) -> None:
    with pytest.raises(
        patch.CudaArchitectureBDirectResidentPatchV1Error,
        match=r"live_banks\.pcg64_buffer_words must be int64\[2,2\]",
    ):
        patch._validate_live_pcg64_buffers_before_settlement_v1(
            {"pcg64_buffer_words": value},
            cell_ids=("cell-0", "cell-1"),
        )
