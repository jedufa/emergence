from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

from tools.visualization.living_lineage_canopy.python.artifact_readers.checkpoint_bundle_v2 import (
    projection_paths_for_cell,
    snapshot_boundary,
    snapshot_from_projection,
    tensor_values,
)
from tools.visualization.living_lineage_canopy.python.backfill import backfill_sources
from infra.webui import canopy_library
from tools.visualization.living_lineage_canopy.python.production_exporter import cross_check_gv2, export_gv2_checkpoint_run
from tools.visualization.living_lineage_canopy.python.provenance import output_hash, validate_provenance_binding
from tools.visualization.living_lineage_canopy.python.schema import SchemaError
from tools.visualization.living_lineage_canopy.python.schema import validate_playback
from tools.visualization.living_lineage_canopy.python.source_audit import audit_source


def _tensor(values: list[int | bool], dtype: str) -> dict[str, object]:
    fmts = {"int64": "<q", "int32": "<i", "int8": "<b", "bool": "<?"}
    payload = b"".join(struct.pack(fmts[dtype], value) for value in values)
    return {
        "kind": "tensor",
        "dtype": dtype,
        "shape": [len(values)],
        "order": "C",
        "device": "cpu",
        "data_hex": payload.hex(),
    }


def _projection(
    tensors: dict[str, dict[str, object]],
    generation: int,
    *,
    source_tick: int = 0,
    ticks_per_generation: int | None = None,
) -> dict[str, object]:
    projection: dict[str, object] = {
        "schema_version": "native_runtime_checkpoint_projection_v1",
        "source_generation": generation,
        "source_tick": source_tick,
        "projection": {
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
            "tensors": tensors,
        },
    }
    if ticks_per_generation is not None:
        projection["projection"]["metadata"] = {
            "configured_steps_per_generation": ticks_per_generation,
        }
    return projection


def _write_projection(
    path: Path,
    tensors: dict[str, dict[str, object]],
    generation: int,
    *,
    source_tick: int = 0,
    ticks_per_generation: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            _projection(
                tensors,
                generation,
                source_tick=source_tick,
                ticks_per_generation=ticks_per_generation,
            ),
            handle,
            sort_keys=True,
        )


def _write_loose_projection(path: Path, tensors: dict[str, dict[str, object]], generation: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_projection(tensors, generation), sort_keys=True), encoding="utf-8")


def _base_tensors(
    *,
    alive: list[int],
    slot_to_agent: list[int],
    founders: list[int],
    depths: list[int],
    events: list[dict[str, int]],
) -> dict[str, dict[str, object]]:
    event_field = lambda key: [event[key] for event in events]
    return {
        "agent_ids_by_alive_order": _tensor(alive, "int64"),
        "slot_to_agent_id": _tensor(slot_to_agent, "int64"),
        "lineage_founder_id": _tensor(founders, "int64"),
        "lineage_depth": _tensor(depths, "int64"),
        "lineage_birth_generation": _tensor([0 for _ in slot_to_agent], "int64"),
        "lineage_birth_tick": _tensor([0 for _ in slot_to_agent], "int64"),
        "alive_mask": _tensor([agent >= 0 for agent in slot_to_agent], "bool"),
        "typed_lifecycle_event_event_type": _tensor(event_field("event_type"), "int8"),
        "typed_lifecycle_event_success": _tensor(event_field("success"), "int8"),
        "typed_lifecycle_event_sequence": _tensor(event_field("sequence"), "int64"),
        "typed_lifecycle_event_generation": _tensor(event_field("generation"), "int64"),
        "typed_lifecycle_event_tick": _tensor(event_field("tick"), "int64"),
        "typed_lifecycle_event_parent_agent_id": _tensor(event_field("parent_agent_id"), "int64"),
        "typed_lifecycle_event_child_agent_id": _tensor(event_field("child_agent_id"), "int64"),
        "typed_lifecycle_event_target_agent_id": _tensor(event_field("target_agent_id"), "int64"),
        "typed_lifecycle_event_lineage_founder_id": _tensor(event_field("lineage_founder_id"), "int64"),
        "typed_lifecycle_event_lineage_depth": _tensor(event_field("lineage_depth"), "int64"),
    }


def _make_gv2_source(root: Path, *, extra_events: list[dict[str, int]] | None = None) -> Path:
    source = root / "run"
    events = [
        {
            "event_type": 1,
            "success": 1,
            "sequence": 0,
            "generation": 0,
            "tick": 0,
            "parent_agent_id": 1,
            "child_agent_id": 3,
            "target_agent_id": 3,
            "lineage_founder_id": 1,
            "lineage_depth": 1,
        },
        {
            "event_type": 1,
            "success": 1,
            "sequence": 1,
            "generation": 0,
            "tick": 1,
            "parent_agent_id": 2,
            "child_agent_id": 4,
            "target_agent_id": 4,
            "lineage_founder_id": 2,
            "lineage_depth": 1,
        },
    ]
    events.extend(extra_events or [])
    base = source / "batches/batch-0000/checkpoint_bundles_v2/cell-test"
    _write_projection(
        base / "bundle_gen000000/native_runtime_checkpoint_projection.json.gz",
        _base_tensors(alive=[1, 2], slot_to_agent=[1, 2, -1, -1], founders=[1, 2, -1, -1], depths=[0, 0, 0, 0], events=[]),
        0,
    )
    _write_projection(
        base / "bundle_gen000005/native_runtime_checkpoint_projection.json.gz",
        _base_tensors(alive=[3], slot_to_agent=[-1, -1, 3, -1], founders=[-1, -1, 1, -1], depths=[0, 0, 1, 0], events=events),
        5,
    )
    return source


def test_tensor_reader_decodes_hex_values(tmp_path: Path) -> None:
    source = _make_gv2_source(tmp_path)
    projection_path = next(source.rglob("bundle_gen000005/native_runtime_checkpoint_projection.json.gz"))
    with gzip.open(projection_path, "rt", encoding="utf-8") as handle:
        projection = json.load(handle)
    assert tensor_values(projection, "agent_ids_by_alive_order") == [3]


def test_gv2_tick_named_checkpoints_preserve_exact_absolute_boundaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "tick-native-run"
    base = source / "checkpoint_bundles_v2/cell-test"
    tensors = _base_tensors(
        alive=[1],
        slot_to_agent=[1, -1],
        founders=[1, -1],
        depths=[0, 0],
        events=[],
    )
    for absolute_tick, generation, source_tick in (
        (17, 0, 17),
        (160, 1, 0),
        (161, 1, 1),
    ):
        _write_projection(
            base
            / f"bundle_tick{absolute_tick:012d}"
            / "native_runtime_checkpoint_projection.json.gz",
            tensors,
            generation,
            source_tick=source_tick,
            ticks_per_generation=160,
        )

    paths = projection_paths_for_cell(source, "cell-test")
    snapshots = [snapshot_from_projection(path) for path in paths]
    assert [snapshot_boundary(snapshot) for snapshot in snapshots] == [17, 160, 161]
    assert [snapshot.generation for snapshot in snapshots] == [0, 1, 1]

    playback, provenance, summary = export_gv2_checkpoint_run(
        source,
        cell_id="cell-test",
        generated_at="2026-08-11T00:00:00Z",
    )
    assert playback["metadata"]["time_unit"] == "absolute_tick"
    assert playback["metadata"]["first_tick"] == 17
    assert playback["metadata"]["last_tick"] == 161
    assert summary["first_tick"] == 17
    assert summary["last_tick"] == 161
    assert validate_provenance_binding(playback, provenance)["ok"]


def test_gv2_single_late_tick_checkpoint_excludes_pre_window_dead_births(
    tmp_path: Path,
) -> None:
    source = tmp_path / "late-terminal-checkpoint"
    base = source / "checkpoint_bundles_v2/cell-test"
    events = [
        {
            "event_type": 1,
            "success": 1,
            "sequence": 0,
            "generation": 0,
            "tick": 100,
            "parent_agent_id": 1,
            "child_agent_id": 2,
            "target_agent_id": 2,
            "lineage_founder_id": 1,
            "lineage_depth": 1,
        }
    ]
    _write_projection(
        base
        / "bundle_tick000000000480"
        / "native_runtime_checkpoint_projection.json.gz",
        _base_tensors(
            alive=[1],
            slot_to_agent=[1, -1],
            founders=[1, -1],
            depths=[0, 0],
            events=events,
        ),
        3,
        source_tick=0,
        ticks_per_generation=160,
    )

    playback, provenance, summary = export_gv2_checkpoint_run(
        source,
        cell_id="cell-test",
        generated_at="2026-08-12T00:00:00Z",
    )
    cross_check = cross_check_gv2(source, playback, cell_id="cell-test")

    assert [agent["source_agent_id"] for agent in playback["agents"]] == [1]
    assert playback["metadata"]["first_tick"] == 480
    assert playback["metadata"]["last_tick"] == 480
    assert summary["pre_window_birth_event_count"] == 1
    assert summary["exported_typed_birth_event_count"] == 0
    assert cross_check["ok"]
    assert validate_provenance_binding(playback, provenance)["ok"]


def test_gv2_source_audit_classifies_checkpoint_projection(tmp_path: Path) -> None:
    source = _make_gv2_source(tmp_path)
    result = audit_source(source)
    assert result.classification == "EXP-B"
    assert result.branch_semantics == "lineage_root_projection"
    assert "authoritative exact death tick for every checkpoint-disappeared agent" in result.missing_fields


def test_gv2_export_accepts_legacy_loose_projection_json(tmp_path: Path) -> None:
    source = tmp_path / "run"
    base = source / "batches/batch-0000/checkpoint_bundles_v2/cell-test"
    _write_loose_projection(
        base / "bundle_gen000000/native_runtime_checkpoint_projection.json",
        _base_tensors(alive=[1], slot_to_agent=[1, -1], founders=[1, -1], depths=[0, 0], events=[]),
        0,
    )
    _write_loose_projection(
        base / "bundle_gen000003/native_runtime_checkpoint_projection.json",
        _base_tensors(
            alive=[2],
            slot_to_agent=[-1, 2],
            founders=[-1, 1],
            depths=[0, 1],
            events=[
                {
                    "event_type": 1,
                    "success": 1,
                    "sequence": 0,
                    "generation": 0,
                    "tick": 0,
                    "parent_agent_id": 1,
                    "child_agent_id": 2,
                    "target_agent_id": 2,
                    "lineage_founder_id": 1,
                    "lineage_depth": 1,
                }
            ],
        ),
        3,
    )
    playback, _, summary = export_gv2_checkpoint_run(source, generated_at="2026-07-30T00:00:00Z")
    assert summary["agent_count"] == 2
    assert playback["metadata"]["last_tick"] == 2


def test_gv2_mixed_exact_and_projected_deaths_remain_exp_b(tmp_path: Path) -> None:
    source = _make_gv2_source(
        tmp_path,
        extra_events=[
            {
                "event_type": 3,
                "success": 1,
                "sequence": 2,
                "generation": 0,
                "tick": 2,
                "parent_agent_id": -1,
                "child_agent_id": -1,
                "target_agent_id": 1,
                "lineage_founder_id": 1,
                "lineage_depth": 0,
            }
        ],
    )
    result = audit_source(source)
    playback, provenance, summary = export_gv2_checkpoint_run(source, generated_at="2026-07-30T00:00:00Z")
    assert result.classification == "EXP-B"
    assert summary["classification"] == "EXP-B"
    assert playback["metadata"]["semantics_mode"] == "partial_agents"
    assert provenance["exporter_git_commit"]


def test_gv2_complete_exact_death_coverage_is_exp_a(tmp_path: Path) -> None:
    source = _make_gv2_source(
        tmp_path,
        extra_events=[
            {
                "event_type": 3,
                "success": 1,
                "sequence": 2,
                "generation": 0,
                "tick": 2,
                "parent_agent_id": -1,
                "child_agent_id": -1,
                "target_agent_id": 1,
                "lineage_founder_id": 1,
                "lineage_depth": 0,
            },
            {
                "event_type": 3,
                "success": 1,
                "sequence": 3,
                "generation": 0,
                "tick": 3,
                "parent_agent_id": -1,
                "child_agent_id": -1,
                "target_agent_id": 2,
                "lineage_founder_id": 2,
                "lineage_depth": 0,
            },
            {
                "event_type": 3,
                "success": 1,
                "sequence": 4,
                "generation": 0,
                "tick": 4,
                "parent_agent_id": -1,
                "child_agent_id": -1,
                "target_agent_id": 4,
                "lineage_founder_id": 2,
                "lineage_depth": 1,
            },
        ],
    )
    result = audit_source(source)
    playback, _, summary = export_gv2_checkpoint_run(source, generated_at="2026-07-30T00:00:00Z")
    assert result.classification == "EXP-A"
    assert summary["classification"] == "EXP-A"
    assert playback["metadata"]["semantics_mode"] == "actual_agents"


def test_gv2_exact_death_identity_conflict_is_exp_d(tmp_path: Path) -> None:
    source = _make_gv2_source(
        tmp_path,
        extra_events=[
            {
                "event_type": 3,
                "success": 1,
                "sequence": 2,
                "generation": 0,
                "tick": 2,
                "parent_agent_id": -1,
                "child_agent_id": -1,
                "target_agent_id": 99,
                "lineage_founder_id": 99,
                "lineage_depth": 0,
            }
        ],
    )
    result = audit_source(source)
    assert result.classification == "EXP-D"


def test_gv2_export_is_deterministic_and_cross_checked(tmp_path: Path) -> None:
    source = _make_gv2_source(tmp_path)
    first, first_provenance, first_summary = export_gv2_checkpoint_run(source, generated_at="2026-07-30T00:00:00Z")
    second, second_provenance, second_summary = export_gv2_checkpoint_run(source, generated_at="2026-07-30T00:00:00Z")
    validate_playback(first)
    assert output_hash(first) == output_hash(second)
    assert first_provenance["output_sha256"] == second_provenance["output_sha256"]
    assert validate_provenance_binding(first, first_provenance)["ok"]
    tampered_provenance = {**first_provenance, "output_sha256": "0" * 64}
    try:
        validate_provenance_binding(first, tampered_provenance)
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("tampered provenance passed validation")
    assert first_summary == second_summary
    assert first["metadata"]["source_classification"] == "EXP-B"
    cross_check = cross_check_gv2(source, first)
    assert cross_check["ok"]
    assert len(cross_check["per_boundary_checks"]) == 2
    final_alive = [
        agent
        for agent in first["agents"]
        if agent["birth_tick"] <= first["metadata"]["last_tick"]
        and (agent.get("death_tick") is None or agent["death_tick"] > first["metadata"]["last_tick"])
    ]
    assert [agent["agent_id"] for agent in final_alive] == ["gv2_agent_3"]


def test_gv2_export_and_cross_check_reuse_preserves_exact_output(tmp_path: Path) -> None:
    source = _make_gv2_source(tmp_path)
    generated_at = "2026-08-12T00:00:00Z"
    regular = export_gv2_checkpoint_run(source, generated_at=generated_at)
    snapshots = tuple(
        snapshot_from_projection(path) for path in projection_paths_for_cell(source)
    )

    reused = export_gv2_checkpoint_run(
        source,
        generated_at=generated_at,
        snapshots=snapshots,
    )

    assert reused == regular
    assert cross_check_gv2(source, reused[0], snapshots=snapshots) == cross_check_gv2(
        source, regular[0]
    )


def test_gv2_checkpoint_only_identity_appearances_are_partial_not_inferred_births(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint-only"
    base = source / "checkpoint_bundles_v2/cell-test"
    _write_projection(
        base / "bundle_gen000000/native_runtime_checkpoint_projection.json.gz",
        _base_tensors(alive=[1], slot_to_agent=[1, -1], founders=[1, -1], depths=[0, 0], events=[]),
        0,
    )
    _write_projection(
        base / "bundle_gen000005/native_runtime_checkpoint_projection.json.gz",
        _base_tensors(alive=[2], slot_to_agent=[-1, 2], founders=[-1, 1], depths=[0, 1], events=[]),
        5,
    )
    playback, provenance, summary = export_gv2_checkpoint_run(source, generated_at="2026-08-10T00:00:00Z")
    cross_check = cross_check_gv2(source, playback)

    observed = next(agent for agent in playback["agents"] if agent["source_agent_id"] == 2)
    assert observed["source_birth_kind"] == "checkpoint_first_observed_identity"
    assert observed["parent_agent_ids"] == []
    assert playback["metadata"]["source_classification"] == "EXP-B"
    assert summary["checkpoint_first_observed_identity_count"] == 1
    assert cross_check["ok"]
    assert validate_provenance_binding(playback, provenance)["ok"]


def test_checkpoint_backfill_is_idempotent_and_leaves_source_untouched(tmp_path: Path) -> None:
    source = _make_gv2_source(tmp_path)
    derived = tmp_path / "derived"
    first = backfill_sources([source], derived_root=derived, generated_at="2026-08-10T00:00:00Z")
    second = backfill_sources([source], derived_root=derived, generated_at="2026-08-10T00:00:00Z")
    assert first["rows"][0]["status"] == "generated"
    assert second["rows"][0]["status"] == "existing"
    assert (derived / "catalog.json").is_file()
    assert canopy_library.discover((("derived", derived),))[0]["label"] == source.name
    assert not list(source.rglob("lineage-playback-relay"))


def test_projected_death_cannot_claim_exp_a(tmp_path: Path) -> None:
    source = _make_gv2_source(tmp_path)
    playback, _, _ = export_gv2_checkpoint_run(source, generated_at="2026-07-30T00:00:00Z")
    playback["metadata"]["source_classification"] = "EXP-A"
    playback["metadata"]["semantics_mode"] = "actual_agents"
    try:
        validate_playback(playback)
    except SchemaError as exc:
        assert "projected deaths require" in str(exc)
    else:
        raise AssertionError("invalid projected-death EXP-A playback passed validation")
