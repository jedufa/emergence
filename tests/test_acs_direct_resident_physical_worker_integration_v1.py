"""Bounded host integration for the physical detached-worker envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import engine.campaigns.acs_direct_resident_physical_settlement_v1 as settlement
import engine.campaigns.acs_native_detached_slots_v1 as detached
import engine.campaigns.acs_native_aggregate_checkpoint_v1 as aggregate
import engine.campaigns.acs_physical_batch_packed_checkpoint_v1 as packed
from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
    SCHEMA_VERSION as PHYSICAL_CAPTURE_SCHEMA_VERSION,
    SETTLEMENT_RECIPE_SCHEMA_VERSION,
)
from engine.campaigns.acs_physical_batch_packed_checkpoint_v1 import (
    capture_physical_batch_packed_checkpoint_v1,
)
from engine.campaigns.providers import (
    ProductionEngineLifecycleV1,
    _BoundedSourceProjectionStoreV1,
    _lazy_projection_bundle_sequence_v1,
)
from engine.native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from engine.perf.cuda_architecture_b_boundary import EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
    JUNGLE_STATE_FIELDS_V4,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    WDL_STATE_FIELDS_V2,
)
from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
    _projection as _authority_projection,
)


def _source_projection(cell_index: int) -> dict[str, object]:
    """Small valid source projection for the bounded host-only worker test."""

    projection = _authority_projection(generation=4, tick=0)
    projection["metadata"] = dict(projection["metadata"])
    projection["metadata"]["event_history_reference"] = {
        "path": f"/tmp/tw-acs-worker-test-cell-{cell_index}.jsonl",
        "frontier": 17 + cell_index,
    }
    return projection


def _component_recipe(
    tmp_path: Path, source_projections: tuple[dict[str, object], ...]
) -> dict[str, object]:
    descriptors: list[dict[str, object]] = []
    digests: dict[str, str] = {}
    cell_ids = tuple(f"cell-{index}" for index in range(len(source_projections)))
    for index, (cell_id, projection) in enumerate(
        zip(cell_ids, source_projections, strict=True)
    ):
        path = tmp_path / f"source-{index}.pt"
        torch.save(projection, path)
        digest = str(
            checkpoint_source_projection_authority_manifest(projection)[
                "source_projection_sha256"
            ]
        )
        descriptors.append(
            {
                "kind": "component",
                "path": str(path),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_projection_sha256": digest,
            }
        )
        digests[cell_id] = digest
    return {
        "schema_version": "eiel_bounded_source_descriptor_recipe_v1",
        "cell_ids": list(cell_ids),
        "descriptors": descriptors,
        "source_projection_sha256s": digests,
    }


def _physical_envelope(
    tmp_path: Path,
    *,
    cells: int = 2,
    source_recipe: dict[str, object] | None = None,
) -> tuple[dict[str, object], Path, Path]:
    cell_ids = tuple(f"cell-{index}" for index in range(cells))
    if source_recipe is None:
        source_projections = tuple(
            _source_projection(index) for index in range(len(cell_ids))
        )
        source_recipe = _component_recipe(tmp_path, source_projections)
    fields: dict[str, torch.Tensor] = {
        name: torch.full((len(cell_ids), 1), index, dtype=torch.float32)
        for index, name in enumerate(
            tuple(WDL_STATE_FIELDS_V2) + tuple(JUNGLE_STATE_FIELDS_V4)
        )
    }
    fields.update(
        {
            name: torch.ones((len(cell_ids), 1), dtype=torch.int64)
            for name in EVENT_COLUMN_NAMES
        }
    )
    fields.update(
        {
            "alive_mask": torch.tensor([[True, False, True]] * cells, dtype=torch.bool),
            "next-step": torch.arange(1120, 1120 + cells, dtype=torch.int64),
            "agent-keys": torch.arange(3, 3 + cells, dtype=torch.int64),
            "agent-flags": torch.arange(cells, dtype=torch.int8).remainder(2),
            "summary": torch.arange(5, 5 + cells, dtype=torch.int64).reshape(
                cells, 1, 1
            ),
        }
    )
    layout = {
        "live_bank_tensors": {name: {"field": name} for name in WDL_STATE_FIELDS_V2},
        "event_columns": {name: {"field": name} for name in EVENT_COLUMN_NAMES},
        "settlement_outputs": {
            "generation_summary_scalar": {"field": "summary"},
        },
        "wdl": {
            "next_absolute_step": {"field": "next-step"},
            "agent_keys": {"field": "agent-keys"},
            "agent_flags": {"field": "agent-flags"},
        },
        "jungle": {name: {"field": name} for name in JUNGLE_STATE_FIELDS_V4},
        "eed": {},
    }
    recipe = {
        "schema_version": SETTLEMENT_RECIPE_SCHEMA_VERSION,
        "provider_id": "worker-integration-test",
        "batch_id": "physical-worker-integration",
        "cell_ids": list(cell_ids),
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 1120,
        "route": {
            "schema_version": "tick_native_execution_route_v1",
            "time_authority": "absolute_tick_v1",
            "checkpoint_at_tick": 1120,
        },
        "absolute_tick_contract": {
            "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
            "time_authority": "absolute_tick_v1",
            "segment_end_next_tick": 1120,
            "source_window_ticks": 160,
        },
        "legacy_compatibility": {
            "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
            "authority": "private_legacy_compatibility_only",
            "derived_from_absolute_next_tick": 1120,
            "generation_base_override": 0,
        },
        "reporting_enabled": False,
        "source_descriptor_recipe": source_recipe,
        "source": {
            "configuration": {"fixture": True, "mode": "bounded-host"},
            "generation_base": 0,
            "logical_generation_count": 1,
            "ticks_per_generation": 160,
        },
        "checkpoint_projection_inputs": {
            "resident_birth_state": {"kind": "mapping", "items": {}},
            "cuda_capability": {"major": 8},
        },
        "live_bank_scalars": {},
        "direct_result_scalars": {
            "completed_generations": 1,
            "source_projection_cell_ids": list(cell_ids),
        },
        "field_layout": layout,
    }
    recipe["recipe_sha256"] = hashlib.sha256(
        json.dumps(
            recipe, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    raw_path = tmp_path / "checkpoint.acspb"
    aggregate_path = tmp_path / "checkpoint.acsaggregate"
    packed_descriptor = capture_physical_batch_packed_checkpoint_v1(
        fields,
        raw_path,
        cell_ids=cell_ids,
        batch_id=str(recipe["batch_id"]),
        scalar_metadata={"settlement_recipe": recipe},
        chunk_bytes=7,
    )
    envelope = {
        "schema_version": PHYSICAL_CAPTURE_SCHEMA_VERSION,
        "source_consumer_required": False,
        "require_native_extension": False,
        "capture_path": packed_descriptor["capture_path"],
        "spool_path": packed_descriptor["capture_path"],
        "destination": str(aggregate_path),
        "raw_publication_path": str(raw_path),
        "packed_descriptor": packed_descriptor,
        "packed_schema_version": packed_descriptor["schema_version"],
        "batch_id": recipe["batch_id"],
        "cell_ids": list(cell_ids),
        "segment_ordinal": 1,
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 1120,
        "logical_bytes": packed_descriptor["payload_bytes"],
        "zero_fill_logical_bytes": packed_descriptor["zero_fill_logical_bytes"],
        "native_slab_pool": dict(packed_descriptor["slab_pool"]),
        "settlement_recipe": recipe,
    }
    return envelope, raw_path, aggregate_path


def test_physical_worker_seals_reads_settles_and_publishes_authoritative_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, raw_path, aggregate_path = _physical_envelope(tmp_path)
    packed_descriptor = envelope["packed_descriptor"]
    assert isinstance(packed_descriptor, dict)
    unsealed_capture = Path(str(packed_descriptor["capture_path"]))
    assert packed_descriptor["sealed"] is False
    assert unsealed_capture.is_file()

    def bounded_canonical_settlement(combined, *, direct_result, **kwargs):
        del direct_result, kwargs
        projection = combined.wdl.source_projections[0]
        cell_id = combined.wdl.cell_ids[0]
        return SimpleNamespace(
            checkpoint_projections=(projection,),
            compact_result_rows=(
                {"cell_id": cell_id, "metrics": {"population.alive": 1}},
            ),
            segment_artifacts=(),
            lifecycle_qacct_receipts=(),
            terminal_generation=1,
            terminal_tick=1120,
        )

    monkeypatch.setattr(
        settlement,
        "project_direct_resident_wdl_jungle_multicell_settlement_v1",
        bounded_canonical_settlement,
    )

    materialized = detached._finalize_capture_v1(envelope)

    assert not unsealed_capture.exists()
    assert not raw_path.exists()
    assert aggregate_path.is_file()
    assert materialized["schema_version"] == aggregate.SCHEMA_VERSION
    assert materialized["status"] == "complete"
    assert materialized["path"] == str(aggregate_path)
    assert materialized["cell_count"] == 2
    assert set(materialized["cells"]) == {"cell-0", "cell-1"}
    assert len(materialized["artifact_sha256"]) == 64
    assert materialized["detached_authorities"] is not None
    assert set(materialized["detached_authorities"]["source_projection_sha256s"]) == {
        "cell-0",
        "cell-1",
    }

    # The worker's strict reader was exercised before canonical materialization;
    # the final artifact must independently be loadable and sealed.
    from engine.campaigns.acs_native_aggregate_checkpoint_v1 import (
        load_acs_native_aggregate_checkpoint,
    )

    restored = load_acs_native_aggregate_checkpoint(
        aggregate_path, cell_id="cell-1", verify_payload=True
    )
    assert restored["footer"]["status"] == "complete"
    assert restored["projection"]["metadata"]["generation"] == 4
    assert restored["projection"]["tensors"]
    assert restored["architecture_b_identity"]["run_id"] == (
        "physical-worker-integration"
    )
    assert restored["observer_continuation"]["checkpoint_next_generation"] == 7


def test_physical_worker_can_publish_world_shards_without_b1_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, raw_path, _aggregate_path = _physical_envelope(tmp_path)
    set_destination = tmp_path / "world-sets" / "checkpoint_tick000000001120"
    envelope["world_sharded_checkpoint"] = True
    envelope["world_sharded_destination_root"] = str(set_destination)
    envelope["world_sharded_worker_count"] = 2
    original_settlement_builder = settlement.build_physical_batch_settlement_cursor_v1
    capture_path = Path(str(envelope["packed_descriptor"]["capture_path"]))

    def forbidden_settlement(*args: object, **kwargs: object) -> object:
        raise AssertionError("world-sharded writer reconstructed a B1 runtime object")

    monkeypatch.setattr(
        settlement,
        "build_physical_batch_settlement_cursor_v1",
        forbidden_settlement,
    )

    def forbidden_raw_finalize(*args: object, **kwargs: object) -> object:
        raise AssertionError("world-sharded writer finalized a transient raw artifact")

    monkeypatch.setattr(
        packed,
        "finalize_physical_batch_packed_checkpoint_v1",
        forbidden_raw_finalize,
    )
    materialized = detached._finalize_capture_v1(envelope)
    monkeypatch.setattr(
        settlement,
        "build_physical_batch_settlement_cursor_v1",
        original_settlement_builder,
    )

    assert not raw_path.exists()
    assert not capture_path.exists()
    assert materialized["schema_version"] == (
        "eiel_acs_world_checkpoint_set_manifest_v1"
    )
    assert materialized["status"] == "complete"
    assert materialized["world_count"] == 2
    assert materialized["physical_batch_raw_finalize_bypassed"] is True
    assert materialized["source_physical_authority_kind"] == (
        "captured_unsealed_fingerprint_and_complete_range_copy_v1"
    )
    assert [row["cell_id"] for row in materialized["physical_compact_result_rows"]] == [
        "cell-0",
        "cell-1",
    ]
    assert all(
        row["metrics"]["runtime.terminal_absolute_tick"] == 1120
        for row in materialized["physical_compact_result_rows"]
    )
    from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
        load_full_world_checkpoint_set_v1,
    )

    counter: dict[str, int] = {}
    worlds = load_full_world_checkpoint_set_v1(
        materialized["manifest_path"], byte_counter=counter
    )
    assert [world["cell_id"] for world in worlds] == ["cell-0", "cell-1"]
    assert counter["world_objects_read"] == 2
    assert counter["set_manifests_read"] == 1

    # A complete set is self-contained: lifecycle release of producer source
    # components cannot invalidate later restore.
    for source_component in tmp_path.glob("source-*.pt"):
        source_component.unlink()

    def bounded_canonical_settlement(combined, *, direct_result, **kwargs):
        del direct_result, kwargs
        projection = combined.wdl.source_projections[0]
        cell_id = combined.wdl.cell_ids[0]
        return SimpleNamespace(
            checkpoint_projections=(projection,),
            compact_result_rows=(
                {"cell_id": cell_id, "metrics": {"population.alive": 1}},
            ),
            segment_artifacts=(),
            lifecycle_qacct_receipts=(),
            terminal_generation=1,
            terminal_tick=1120,
        )

    monkeypatch.setattr(
        settlement,
        "project_direct_resident_wdl_jungle_multicell_settlement_v1",
        bounded_canonical_settlement,
    )
    from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
        restore_full_world_checkpoint_set_v1,
    )

    restore_counter: dict[str, int] = {}
    restored = restore_full_world_checkpoint_set_v1(
        materialized["manifest_path"], byte_counter=restore_counter
    )
    assert [row["cell_id"] for row in restored] == ["cell-0", "cell-1"]
    assert restore_counter["world_objects_read"] == 2
    assert restore_counter["set_manifests_read"] == 1
    assert "source_dependencies_read" not in restore_counter
    for index, row in enumerate(restored):
        expected = _source_projection(index)
        assert checkpoint_source_projection_authority_manifest(
            row["projection"]
        ) == checkpoint_source_projection_authority_manifest(expected)


@pytest.mark.parametrize("cells", [2, 136])
def test_restored_world_set_flattens_before_next_world_sharded_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cells: int
) -> None:
    initial_root = tmp_path / "initial"
    initial_root.mkdir()
    cell_ids = tuple(f"cell-{index}" for index in range(cells))
    envelope, _raw_path, _aggregate_path = _physical_envelope(initial_root, cells=cells)
    initial_destination = initial_root / "world-sets" / "checkpoint_tick000000001120"
    envelope["world_sharded_checkpoint"] = True
    envelope["world_sharded_destination_root"] = str(initial_destination)
    envelope["world_sharded_worker_count"] = min(8, cells)
    first = detached._finalize_capture_v1(envelope)

    def bounded_canonical_settlement(combined, *, direct_result, **kwargs):
        del direct_result, kwargs
        projection = combined.wdl.source_projections[0]
        cell_id = combined.wdl.cell_ids[0]
        return SimpleNamespace(
            checkpoint_projections=(projection,),
            compact_result_rows=(
                {"cell_id": cell_id, "metrics": {"population.alive": 1}},
            ),
            segment_artifacts=(),
            lifecycle_qacct_receipts=(),
            terminal_generation=1,
            terminal_tick=1120,
        )

    monkeypatch.setattr(
        settlement,
        "project_direct_resident_wdl_jungle_multicell_settlement_v1",
        bounded_canonical_settlement,
    )
    from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
        load_world_checkpoint_set_manifest_v1,
        restore_world_checkpoint_projection_v1,
    )

    first_manifest = load_world_checkpoint_set_manifest_v1(first["manifest_path"])
    first_root = Path(first["set_root"])
    first_paths = tuple(
        str(first_root / row["relative_path"]) for row in first_manifest["objects"]
    )
    loader_calls: list[str] = []

    def load_world(path: str) -> dict[str, object]:
        loader_calls.append(path)
        return dict(restore_world_checkpoint_projection_v1(path)["projection"])

    sources = _lazy_projection_bundle_sequence_v1(
        first_paths,
        load_world,
        cell_ids=cell_ids,
    )

    class SourceImportProvider:
        terminal_projection_digest = staticmethod(
            lambda projection: str(
                checkpoint_source_projection_authority_manifest(projection)[
                    "source_projection_sha256"
                ]
            )
        )
        _write_component_checkpoint = staticmethod(
            ProductionEngineLifecycleV1._write_component_checkpoint
        )

    import_root = tmp_path / "continued-source"
    store = _BoundedSourceProjectionStoreV1(
        SourceImportProvider(),
        root=import_root,
        cell_ids=cell_ids,
        source_projections=sources,
    )
    flattened_recipe = store.detached_descriptor_recipe_v1()
    flattened = flattened_recipe["descriptors"]
    assert loader_calls == list(first_paths)
    assert all(row["kind"] == "component" for row in flattened)
    assert all(str(import_root) in row["path"] for row in flattened)
    assert not any(
        path.endswith(".acsworld") for path in (row["path"] for row in flattened)
    )

    # The next durability pass must depend only on the flattened components.
    # Removing the old set proves that it neither nests nor rereads a prior
    # world object on the persistence critical path.
    for path in first_paths:
        Path(path).unlink()
    Path(first["manifest_path"]).unlink()

    continued_root = tmp_path / "continued"
    continued_root.mkdir()
    continued, _continued_raw, _continued_aggregate = _physical_envelope(
        continued_root,
        cells=cells,
        source_recipe=flattened_recipe,
    )
    continued_destination = (
        continued_root / "world-sets" / "checkpoint_tick000000001120"
    )
    continued["world_sharded_checkpoint"] = True
    continued["world_sharded_destination_root"] = str(continued_destination)
    continued["world_sharded_worker_count"] = min(8, cells)

    def forbidden_settlement(*args: object, **kwargs: object) -> object:
        raise AssertionError("continued durability reconstructed a B1 runtime object")

    monkeypatch.setattr(
        settlement,
        "build_physical_batch_settlement_cursor_v1",
        forbidden_settlement,
    )
    second = detached._finalize_capture_v1(continued)

    assert second["status"] == "complete"
    assert second["world_count"] == cells
    assert len(tuple(Path(second["set_root"]).glob("objects/*.acsworld"))) == cells


def test_physical_b136_schema_emits_exact_independent_world_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope, raw_path, _aggregate_path = _physical_envelope(tmp_path, cells=136)
    destination = tmp_path / "world-sets" / "checkpoint_tick000000001120"
    envelope["world_sharded_checkpoint"] = True
    envelope["world_sharded_destination_root"] = str(destination)
    envelope["world_sharded_worker_count"] = 8

    def forbidden_settlement(*args: object, **kwargs: object) -> object:
        raise AssertionError("B136 sharding reconstructed a B1 runtime object")

    monkeypatch.setattr(
        settlement,
        "build_physical_batch_settlement_cursor_v1",
        forbidden_settlement,
    )
    materialized = detached._finalize_capture_v1(envelope)

    assert not raw_path.exists()
    assert materialized["world_count"] == 136
    assert materialized["worker_count"] == 8
    assert materialized["bounded_in_flight_world_jobs"] == 8
    assert materialized["queued_cuda_references"] == 0
    assert materialized["queued_cuda_callbacks"] == 0
    assert len(tuple(Path(materialized["set_root"]).glob("objects/*.acsworld"))) == 136

    from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
        load_selected_world_from_set_v1,
    )

    for index in (0, 67, 135):
        counter: dict[str, int] = {}
        loaded = load_selected_world_from_set_v1(
            materialized["manifest_path"],
            world_id=f"cell-{index}",
            byte_counter=counter,
        )
        assert loaded["physical_cell_index"] == index
        assert counter["world_objects_read"] == 1
        assert counter["set_manifests_read"] == 1


def test_absolute_tick_worker_coordinates_do_not_require_legacy_metadata() -> None:
    capture = {
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 321,
    }
    recipe = {
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 321,
        "absolute_tick_contract": {
            "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
            "time_authority": "absolute_tick_v1",
            "segment_end_next_tick": 321,
            "source_window_ticks": 160,
        },
        "legacy_compatibility": {
            "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
            "authority": "private_legacy_compatibility_only",
        },
    }

    assert detached._physical_batch_checkpoint_coordinates_v1(capture, recipe) == (
        2,
        1,
        321,
    )


@pytest.mark.parametrize(
    ("capture_delta", "compatibility", "message"),
    [
        (
            {"generation": 2, "tick": 1},
            {
                "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
                "authority": "private_legacy_compatibility_only",
            },
            "public authority leaked legacy coordinates",
        ),
        (
            {},
            {
                "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
                "authority": "private_legacy_compatibility_only",
                "provider_generation": 1,
                "provider_tick": 1,
            },
            "compatibility coordinates contradict absolute ticks",
        ),
    ],
)
def test_absolute_tick_worker_rejects_public_or_contradictory_legacy_coordinates(
    capture_delta: dict[str, int],
    compatibility: dict[str, object],
    message: str,
) -> None:
    capture = {
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 321,
        **capture_delta,
    }
    recipe = {
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 321,
        "absolute_tick_contract": {
            "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
            "time_authority": "absolute_tick_v1",
            "segment_end_next_tick": 321,
            "source_window_ticks": 160,
        },
        "legacy_compatibility": compatibility,
    }

    with pytest.raises(detached.ACSNativeDetachedSlotError, match=message):
        detached._physical_batch_checkpoint_coordinates_v1(capture, recipe)


@pytest.mark.parametrize(
    ("capture_delta", "recipe_delta", "message"),
    [
        (
            {},
            {
                "legacy_compatibility": {
                    "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
                    "authority": "private_legacy_compatibility_only",
                    "provider_generation": 2,
                }
            },
            "compatibility coordinates are incomplete",
        ),
        (
            {"time_authority": "absolute_tick_v2"},
            {"time_authority": "absolute_tick_v2"},
            "time authority is invalid",
        ),
    ],
)
def test_worker_coordinates_reject_incomplete_or_unknown_authority(
    capture_delta: dict[str, object],
    recipe_delta: dict[str, object],
    message: str,
) -> None:
    capture = {
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 321,
        **capture_delta,
    }
    recipe = {
        "time_authority": "absolute_tick_v1",
        "absolute_next_tick": 321,
        "absolute_tick_contract": {
            "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
            "time_authority": "absolute_tick_v1",
            "segment_end_next_tick": 321,
            "source_window_ticks": 160,
        },
        "legacy_compatibility": {
            "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
            "authority": "private_legacy_compatibility_only",
        },
        **recipe_delta,
    }

    with pytest.raises(detached.ACSNativeDetachedSlotError, match=message):
        detached._physical_batch_checkpoint_coordinates_v1(capture, recipe)


def test_legacy_worker_coordinates_remain_available_without_time_authority() -> None:
    capture = {"generation": 4, "tick": 17, "absolute_next_tick": None}
    recipe = {"generation": 4, "tick": 17}

    assert detached._physical_batch_checkpoint_coordinates_v1(capture, recipe) == (
        4,
        17,
        None,
    )
