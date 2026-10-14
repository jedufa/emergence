"""CPU contract tests for direct-resident physical capture."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import engine.campaigns.acs_direct_resident_physical_capture_v1 as capture
from engine.campaigns.acs_direct_resident_field_manifest_v1 import (
    CANONICAL_FIELD_NAMES_V1,
)
from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
    EED_STATE_FIELDS_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
    JUNGLE_STATE_FIELDS_V4,
)
from engine.perf.cuda_architecture_b_generation_birth_state import (
    CudaGenerationBirthState,
    CudaGenerationBirthStateReceipt,
)


CELLS = 136
EVENT_CAPACITY = 41_804
EVENT_BYTES = 801_633_504


def _birth_template() -> CudaGenerationBirthState:
    def empty() -> torch.Tensor:
        return torch.empty(0, dtype=torch.int64)

    receipt = CudaGenerationBirthStateReceipt(
        schema_version="receipt-v1",
        source_projection_digest="a" * 64,
        target_device="cpu",
        world="fixture",
        steps=160,
        max_births_per_tick=1,
        slot_capacity=1,
        birth_episode_capacity=1,
        turnover_refill_capacity=1,
        agent_id_to_slot_capacity=1,
        cpython_gauss_raw_capacity=1,
        fixed_mlp_policy_tape_episodes=1,
        fixed_mlp_policy_full_horizon_episodes=1,
        policy_birth_segment_refill_enabled=False,
        cpu_torch_rng_source_reads=0,
        generation_boundary_device_validation_syncs=0,
        torch_rng_post_state_encoding="fixture",
        torch_rng_post_state_logical_row_bytes=0,
        torch_rng_post_state_packed_row_bytes=0,
        torch_rng_post_state_dense_allocation_materialized=False,
        torch_rng_post_state_resident_rows_materialized=False,
        torch_rng_post_state_abi={"version": 1},
    )
    return CudaGenerationBirthState(
        schema_version="birth-v1",
        identity_state={},
        scalar_rng_state={},
        genome_state={},
        policy_mem_dim=empty(),
        policy_use_gru=empty(),
        drive_state={},
        policy_state={},
        turnover_policy_state={},
        planner_pcg64_state_words=empty(),
        planner_pcg64_buffer_words=empty(),
        policy_np_semantic_draws=empty(),
        runner_pcg64_state_words=empty(),
        runner_pcg64_buffer_words=empty(),
        metrics_np_semantic_draws=empty(),
        event_tape_columns=(empty(),),
        event_tape_active_count=empty(),
        event_tape_status=empty(),
        receipt=receipt,
    )


def _capture_owner(*, event_shape: tuple[int, ...] = (CELLS, EVENT_CAPACITY)):
    cell_ids = tuple(f"cell-{index}" for index in range(CELLS))
    manifest_names = (
        set(CANONICAL_FIELD_NAMES_V1)
        | set(JUNGLE_STATE_FIELDS_V4)
        | set(EED_STATE_FIELDS_V1)
    )
    live_banks: dict[str, object] = {
        name: torch.zeros((CELLS,), dtype=torch.float32)
        for name in manifest_names - set(EVENT_COLUMN_NAMES)
    }
    event_columns = [
        torch.empty(
            (*event_shape, *(1 for _ in range(index))),
            dtype=EVENT_COLUMN_DTYPES[name],
            device="meta",
        )
        for index, name in enumerate(EVENT_COLUMN_NAMES)
    ]
    live_banks["event_columns"] = event_columns
    source = SimpleNamespace(
        outputs={},
        configuration={"fixture": True, "nested": {"seed": 7}},
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        refill_chunk_ticks=160,
        source_window_ticks=160,
        resident_event_capacity=EVENT_CAPACITY,
        generation_base=0,
        logical_generation_count=1,
        physical_window_generations=1,
        ticks_per_generation=160,
    )
    trajectory = SimpleNamespace(
        source=source,
        live_banks=live_banks,
        checkpoint_projection_inputs={
            "resident_birth_state": _birth_template(),
            "cuda_capability": {"major": 8, "minor": 0},
            "scalar_authority": {"route": "tick-native"},
        },
    )
    descriptor_recipe = {
        "schema_version": capture.SOURCE_DESCRIPTOR_RECIPE_SCHEMA_VERSION,
        "cell_ids": list(cell_ids),
        "descriptors": [
            {"kind": "component", "path": f"/tmp/source-{index}.pt"}
            for index in range(CELLS)
        ],
    }
    factory = SimpleNamespace(
        detached_descriptor_recipe_v1=lambda: descriptor_recipe
    )
    wdl = SimpleNamespace(
        patch=trajectory,
        cell_ids=cell_ids,
        source_projection_cursor_factory=factory,
        schedule_tensors=(
            torch.zeros((CELLS,), dtype=torch.int64),
            torch.full((CELLS,), 1120, dtype=torch.int64),
        ),
        reporting_tensors=(
            *(torch.empty((CELLS, 1)) for _ in range(6)),
            torch.ones((CELLS, 1), dtype=torch.int64),
            torch.zeros((CELLS, 1), dtype=torch.int8),
        ),
    )
    composite = SimpleNamespace(
        wdl=wdl,
        jungle_state_tensors=tuple(live_banks[name] for name in JUNGLE_STATE_FIELDS_V4),
        eed_state_tensors=tuple(live_banks[name] for name in EED_STATE_FIELDS_V1),
    )
    prepared = SimpleNamespace(physical_window_arguments=composite)
    direct_result = {
        "schema_version": "eiel_cuda_architecture_b_native_resident_window_controller_v1",
        "owner": "native_extension",
        "physical_window_generations": 1,
        "logical_generation_count": 1,
        "window_count": 1,
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "event_history_storage_mode": "eiel_resident_event_sink_v1",
        "event_history_reference": {"path": "/tmp/events.bin"},
        "event_history_frontier_by_cell": [0] * CELLS,
        "event_history_prefix_references_by_cell": [
            {
                "path": "/tmp/events.bin",
                "cell_index": cell,
                "row_count": 0,
                "row_bytes": 141,
                "prefix_sha256": "a" * 64,
            }
            for cell in range(CELLS)
        ],
        "completed_generations": 1,
        "source_projection_cell_ids": list(cell_ids),
        "tick_span_start": 0,
    }
    return prepared, direct_result, cell_ids, live_banks


def test_birth_template_round_trip_preserves_admitted_dataclass_and_empty_cpu_tensors() -> None:
    original = _birth_template()
    encoded = capture._encode_birth_template(original, path="birth")
    decoded = capture.decode_birth_template_v1(encoded)

    assert isinstance(decoded, CudaGenerationBirthState)
    assert decoded.schema_version == original.schema_version
    assert decoded.receipt == original.receipt
    assert decoded.event_tape_columns[0].device.type == "cpu"
    assert decoded.event_tape_columns[0].numel() == 0
    assert torch.equal(decoded.policy_mem_dim, original.policy_mem_dim)


def test_strided_live_bank_view_is_described_from_one_contiguous_storage_owner() -> None:
    storage = torch.arange(CELLS * 2 * 3, dtype=torch.float32).reshape(
        CELLS, 2, 3
    )
    view = storage[:, :, 1]
    assert storage.is_contiguous()
    assert not view.is_contiguous()
    assert capture._tensor_storage_identity(storage) == (
        capture._tensor_storage_identity(view)
    )

    descriptor = capture._live_bank_alias_descriptor(
        view,
        base_name="diploid_allele_storage",
        base=storage,
    )

    assert descriptor == {
        "alias_of": "diploid_allele_storage",
        "view": {
            "storage_offset": 1,
            "shape": [2],
            "stride": [3],
        },
    }


def test_valid_b136_capture_records_exact_route_and_tensor_free_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, direct_result, cell_ids, _ = _capture_owner()
    source = prepared.physical_window_arguments.wdl.patch.source
    source_runtime = torch.zeros((CELLS, 160, 5), dtype=torch.int64)
    direct_runtime = torch.ones((CELLS, 1, 5), dtype=torch.int64)
    source.outputs["runtime_bookkeeping_by_tick"] = source_runtime
    direct_result["runtime_bookkeeping_by_tick"] = direct_runtime
    captured: dict[str, object] = {}

    def fake_writer(
        fields,
        destination,
        *,
        cell_ids,
        batch_id,
        scalar_metadata,
        require_native_pool,
    ):
        captured["fields"] = fields
        captured["cell_ids"] = tuple(cell_ids)
        captured["metadata"] = scalar_metadata
        captured["require_native_pool"] = require_native_pool
        return {
            "capture_path": str(tmp_path / "spool.acspb"),
            "requested_publication_path": str(destination),
            "payload_bytes": 123,
            "zero_fill_logical_bytes": 456,
            "slab_pool": {
                "native_pool": True,
                "slab_count": 2,
                "pinned": True,
                "preallocated": True,
            },
        }

    monkeypatch.setattr(capture, "capture_physical_batch_packed_checkpoint_v1", fake_writer)
    envelope = capture.build_direct_resident_physical_capture_v1(
        prepared,
        direct_result=direct_result,
        raw_destination=tmp_path / "raw.acspb",
        aggregate_destination=tmp_path / "final.acsaggregate",
        provider_id="fixture-provider",
        batch_id="batch-136",
        segment_ordinal=2,
        generation=1,
        tick=160,
        absolute_next_tick=1120,
        physical_owner_identity="fixture-owner",
        reporting_enabled=False,
        cpu_request_staging_enable=False,
    )

    recipe = envelope["settlement_recipe"]
    assert recipe["cell_ids"] == list(cell_ids)
    assert captured["require_native_pool"] is True
    assert recipe["route"] == {
        "execution_route_identity": "tick_native_execution_route_v1",
        "physical_batch_cells": CELLS,
        "event_window_capacity": EVENT_CAPACITY,
        "event_window_physical_bytes": EVENT_BYTES,
        "event_row_bytes": 141,
        "resident_owner_span_ticks": 1120,
        "native_dispatch_window_ticks": 1120,
        "source_window_ticks": 160,
        "cpu_request_staging_enable": False,
        "physical_owner_identity": "fixture-owner",
    }
    assert captured["cell_ids"] == cell_ids
    assert captured["fields"]["output/runtime_bookkeeping_by_tick"] is direct_runtime
    assert recipe["source"]["physical_window_generations"] == 1
    assert all(not isinstance(value, torch.Tensor) for value in captured["metadata"].values())
    assert json.dumps(recipe, sort_keys=True, allow_nan=False)
    assert recipe["source_descriptor_recipe"]["schema_version"] == (
        capture.SOURCE_DESCRIPTOR_RECIPE_SCHEMA_VERSION
    )


@pytest.mark.parametrize(
    (
        "start_next_tick",
        "end_next_tick",
        "settlement_generation",
        "private_span_start",
        "private_span_end",
        "boundary",
        "source_window_count",
    ),
    [
        (0, 160, 0, 0, 160, True, 1),
        (3359, 3360, 20, 159, 160, True, 1),
        (3358, 3359, 20, 158, 159, False, 1),
        (3200, 3520, 21, 0, 160, True, 2),
        (3199, 3519, 21, 0, 159, False, 3),
    ],
)
def test_tick_native_capture_derives_only_private_legacy_coordinates_from_ticks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_next_tick: int,
    end_next_tick: int,
    settlement_generation: int,
    private_span_start: int,
    private_span_end: int,
    boundary: bool,
    source_window_count: int,
) -> None:
    prepared, direct_result, cell_ids, _ = _capture_owner()
    source = prepared.physical_window_arguments.wdl.patch.source
    source.logical_generation_count = 7
    span_ticks = end_next_tick - start_next_tick
    terminal_generation, terminal_tick = divmod(end_next_tick, 160)
    direct_result.update(
        {
            "absolute_tick_span_native_owner": True,
            "physical_window_alignment": "absolute_tick_v1",
            "native_tick_iterations": span_ticks,
            "source_window_ticks": 160,
            "source_window_count": source_window_count,
            "tick_native_start_next_tick": start_next_tick,
            "tick_native_end_next_tick": end_next_tick,
            "tick_span_start": start_next_tick,
            "tick_span_end": end_next_tick,
            "tick_span_count": span_ticks,
            "terminal_generation_boundary": False,
            "tick_native_source_prefix_by_cell": [
                {
                    "metadata": {
                        "generation": terminal_generation,
                        "tick": terminal_tick,
                        "prediction_tick": terminal_tick,
                        "agent_grid_runner_phase": (
                            "generation_boundary"
                            if terminal_tick == 0
                            else "partial_generation"
                        ),
                    },
                    "runtime": {},
                }
                for _ in cell_ids
            ],
            "runtime_bookkeeping_by_tick": torch.ones(
                (CELLS, private_span_end - private_span_start, 5),
                dtype=torch.int64,
            ),
        }
    )
    raw_generation_facade = {
        name: direct_result.pop(name)
        for name in (
            "physical_window_generations",
            "logical_generation_count",
            "window_count",
            "completed_generations",
        )
    }

    def fake_writer(_fields, destination, **_kwargs):
        return {
            "capture_path": str(tmp_path / "spool.acspb"),
            "requested_publication_path": str(destination),
            "payload_bytes": 123,
            "zero_fill_logical_bytes": 456,
            "slab_pool": {
                "native_pool": True,
                "slab_count": 2,
                "pinned": True,
                "preallocated": True,
            },
        }

    monkeypatch.setattr(capture, "capture_physical_batch_packed_checkpoint_v1", fake_writer)

    def build(result: dict[str, object], suffix: str):
        return capture.build_direct_resident_physical_capture_v1(
            prepared,
            direct_result=result,
            raw_destination=tmp_path / f"raw-{suffix}.acspb",
            aggregate_destination=tmp_path / f"final-{suffix}.acsaggregate",
            provider_id="fixture-provider",
            batch_id="batch-136",
            segment_ordinal=2,
            generation=terminal_generation,
            tick=terminal_tick,
            absolute_next_tick=end_next_tick,
            physical_owner_identity="fixture-owner",
            reporting_enabled=False,
            cpu_request_staging_enable=False,
        )

    without_generation_facade = build(direct_result, "without")
    with_generation_facade_result = copy.deepcopy(direct_result)
    with_generation_facade_result.update(raw_generation_facade)
    with_generation_facade = build(with_generation_facade_result, "with")

    recipe = without_generation_facade["settlement_recipe"]
    assert recipe["time_authority"] == "absolute_tick_v1"
    assert recipe["absolute_tick_contract"] == {
        "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
        "time_authority": "absolute_tick_v1",
        "segment_start_next_tick": start_next_tick,
        "segment_end_next_tick": end_next_tick,
        "span_ticks": span_ticks,
        "native_tick_iterations": span_ticks,
        "source_window_ticks": 160,
        "source_window_count": source_window_count,
    }
    adapter = recipe["legacy_compatibility"]
    assert {
        name: value
        for name, value in adapter.items()
        if name != "private_settlement_direct_result_scalars"
    } == {
        "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
        "authority": "private_legacy_compatibility_only",
        "derived_from_absolute_next_tick": end_next_tick,
        "interval_ticks": 160,
        "source_generation": settlement_generation,
        "source_tick": private_span_start,
        "tick_span_start": private_span_start,
        "tick_span_end": private_span_end,
        "tick_span_count": private_span_end - private_span_start,
        "terminal_generation_boundary": boundary,
        "provider_generation": terminal_generation,
        "provider_tick": terminal_tick,
        "generation_base_override": settlement_generation,
        "private_settlement_source": {
            "generation_base": 0,
            "logical_generation_count": 7,
            "physical_window_generations": 1,
            "ticks_per_generation": 160,
        },
    }
    public_scalars = recipe["direct_result_scalars"]
    assert public_scalars["physical_window_alignment"] == "absolute_tick_v1"
    assert public_scalars["tick_span_start"] == start_next_tick
    assert public_scalars["tick_span_end"] == end_next_tick
    assert public_scalars["tick_span_count"] == span_ticks
    assert not {
        "completed_generations",
        "logical_generation_count",
        "physical_window_generations",
        "settled_window_count",
        "terminal_generation_boundary",
        "tick_native_source_prefix_by_cell",
        "window_count",
    } & set(public_scalars)
    scalars = adapter["private_settlement_direct_result_scalars"]
    assert scalars["logical_generation_count"] == 7
    assert scalars["window_count"] == 7
    assert scalars["completed_generations"] == 1
    assert scalars["physical_window_alignment"] == "absolute_generation_lattice_v1"
    assert scalars["tick_span_start"] == private_span_start
    assert scalars["tick_span_end"] == private_span_end
    assert scalars["tick_span_count"] == private_span_end - private_span_start
    assert scalars["terminal_generation_boundary"] is boundary
    assert all(
        row["metadata"]["generation"] == settlement_generation
        and row["metadata"]["tick"] == private_span_start
        and row["metadata"]["prediction_tick"] == private_span_start
        and row["metadata"]["agent_grid_runner_phase"]
        == ("generation_boundary" if private_span_start == 0 else "partial_generation")
        for row in scalars["tick_native_source_prefix_by_cell"]
    )
    assert recipe["source"] == {
        "configuration": {"fixture": True, "nested": {"seed": 7}},
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": end_next_tick,
    }
    assert "generation" not in recipe and "tick" not in recipe
    assert "generation" not in without_generation_facade
    assert "tick" not in without_generation_facade
    assert recipe["recipe_sha256"] == with_generation_facade["settlement_recipe"][
        "recipe_sha256"
    ]


@pytest.mark.parametrize(
    ("event_shape", "message"),
    [
        ((CELLS - 1, EVENT_CAPACITY), "B136x41804"),
        ((CELLS, EVENT_CAPACITY - 1), "B136x41804"),
    ],
)
def test_capture_rejects_exact_event_geometry_before_writer(
    tmp_path: Path,
    event_shape: tuple[int, ...],
    message: str,
) -> None:
    prepared, direct_result, _, _ = _capture_owner(event_shape=event_shape)
    with pytest.raises(capture.DirectResidentPhysicalCaptureError, match=message):
        capture.build_direct_resident_physical_capture_v1(
            prepared,
            direct_result=direct_result,
            raw_destination=tmp_path / "raw.acspb",
            aggregate_destination=tmp_path / "final.acsaggregate",
            provider_id="fixture-provider",
            batch_id="bad-geometry",
            segment_ordinal=0,
            generation=1,
            tick=160,
            absolute_next_tick=1120,
            physical_owner_identity="fixture-owner",
            reporting_enabled=False,
            cpu_request_staging_enable=False,
        )


def test_capture_rejects_event_byte_mismatch_and_owner_geometry(
    tmp_path: Path,
) -> None:
    prepared, direct_result, _, live_banks = _capture_owner()
    first = EVENT_COLUMN_NAMES[0]
    live_banks["event_columns"][0] = torch.empty(
        (CELLS, EVENT_CAPACITY), dtype=torch.float64, device="meta"
    )
    with pytest.raises(capture.DirectResidentPhysicalCaptureError, match="event-column bytes"):
        capture.build_direct_resident_physical_capture_v1(
            prepared,
            direct_result=direct_result,
            raw_destination=tmp_path / "raw.acspb",
            aggregate_destination=tmp_path / "final.acsaggregate",
            provider_id="fixture-provider",
            batch_id="bad-bytes",
            segment_ordinal=0,
            generation=1,
            tick=160,
            absolute_next_tick=1120,
            physical_owner_identity="fixture-owner",
            reporting_enabled=False,
            cpu_request_staging_enable=False,
        )
    assert first == EVENT_COLUMN_NAMES[0]

    prepared, direct_result, _, _ = _capture_owner()
    prepared.physical_window_arguments.wdl.patch.source.resident_owner_span_ticks = 1119
    with pytest.raises(capture.DirectResidentPhysicalCaptureError, match="frozen route"):
        capture.build_direct_resident_physical_capture_v1(
            prepared,
            direct_result=direct_result,
            raw_destination=tmp_path / "raw.acspb",
            aggregate_destination=tmp_path / "final.acsaggregate",
            provider_id="fixture-provider",
            batch_id="bad-geometry",
            segment_ordinal=0,
            generation=1,
            tick=160,
            absolute_next_tick=1120,
            physical_owner_identity="fixture-owner",
            reporting_enabled=False,
            cpu_request_staging_enable=False,
        )


@pytest.mark.parametrize(
    ("live_bank_change", "result_mode", "message"),
    [
        ({"hapb_policy_owner": object()}, None, "HAPB host-policy"),
        ({}, "legacy_event_tape_v1", "resident event-history"),
    ],
)
def test_capture_rejects_hapb_and_legacy_event_routes(
    tmp_path: Path,
    live_bank_change: dict[str, object],
    result_mode: str | None,
    message: str,
) -> None:
    prepared, direct_result, _, live_banks = _capture_owner()
    live_banks.update(live_bank_change)
    if result_mode is not None:
        direct_result["event_history_storage_mode"] = result_mode
    with pytest.raises(capture.DirectResidentPhysicalCaptureError, match=message):
        capture.build_direct_resident_physical_capture_v1(
            prepared,
            direct_result=direct_result,
            raw_destination=tmp_path / "raw.acspb",
            aggregate_destination=tmp_path / "final.acsaggregate",
            provider_id="fixture-provider",
            batch_id="wrong-route",
            segment_ordinal=0,
            generation=1,
            tick=160,
            absolute_next_tick=1120,
            physical_owner_identity="fixture-owner",
            reporting_enabled=False,
            cpu_request_staging_enable=False,
        )
