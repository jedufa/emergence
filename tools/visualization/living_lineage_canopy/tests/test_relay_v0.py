from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.visualization.living_lineage_canopy.python.generator import balanced
from tools.visualization.living_lineage_canopy.python.provenance import (
    output_hash,
    write_canonical_json,
)
from tools.visualization.living_lineage_canopy.python.relay import (
    FileFingerprint,
    Gv2CheckpointSource,
    LineagePlaybackRelay,
    RelayIntegrityError,
    SourceNotReady,
    build_staged_playback_source,
    validate_relay_manifest,
)


def _terminal_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _write_source(tmp_path: Path, playback: dict | None = None) -> tuple[Path, Path]:
    data = playback or balanced()
    data["metadata"]["dataset_id"] = "relay_unit_balanced"
    data["metadata"]["source_kind"] = "real_export"
    data["metadata"]["source_classification"] = "EXP-B"
    data["metadata"]["semantics_mode"] = "partial_agents"
    data["metadata"]["parent_reference_policy"] = "allow_missing_external"
    provenance = {
        "schema_version": "eiel_lineage_playback_export_provenance_v1",
        "exporter_schema_version": "eiel_lineage_playback_exporter_v0",
        "playback_schema_version": data["metadata"]["schema_version"],
        "dataset_id": data["metadata"]["dataset_id"],
        "source_root": str(tmp_path / "historical_raw_absent"),
        "source_root_supplied": str(tmp_path / "historical_raw_absent"),
        "source_root_resolved": str(tmp_path / "historical_raw_absent"),
        "source_files": [],
        "source_run_id": "relay-unit",
        "source_cell_id": data["metadata"].get("source_cell_id"),
        "source_classification": data["metadata"]["source_classification"],
        "semantics_mode": data["metadata"]["semantics_mode"],
        "exporter_git_commit": "unit-test",
        "output_sha256": output_hash(data),
        "warnings": data["metadata"].get("warnings", []),
    }
    playback_path = tmp_path / "accepted_playback.json"
    provenance_path = tmp_path / "accepted_provenance.json"
    write_canonical_json(playback_path, data, overwrite=True)
    write_canonical_json(provenance_path, provenance, overwrite=True)
    return playback_path, provenance_path


def test_staged_source_publishes_successive_atomic_updates(tmp_path: Path) -> None:
    playback_path, provenance_path = _write_source(tmp_path)
    source_root = tmp_path / "staged_source"
    build_staged_playback_source(
        playback_path=playback_path,
        provenance_path=provenance_path,
        out=source_root,
        boundaries=[50, 100, 150],
    )
    relay = LineagePlaybackRelay(
        source_root=source_root,
        output_root=tmp_path / "relay",
        source_kind="staged_normalized_playback",
    )

    first_probe = relay.update_once()
    assert first_probe["updated"] is False
    assert first_probe["status"]["state"] == "initializing"

    first = relay.update_once()
    assert first["updated"] is True
    assert first["manifest"]["sequence_number"] == 1
    assert first["manifest"]["latest_exported_playback_boundary"] == 50
    validate_relay_manifest(tmp_path / "relay" / "relay_manifest.json")

    second = relay.update_once()
    assert second["updated"] is True
    assert second["manifest"]["sequence_number"] == 2
    assert second["manifest"]["latest_exported_playback_boundary"] == 100
    assert (tmp_path / "relay" / "playback_snapshot.json").is_file()
    assert (tmp_path / "relay" / "provenance.json").is_file()
    assert (tmp_path / "relay" / "relay_status.json").is_file()


def test_idempotent_scan_after_final_boundary_does_not_increment(
    tmp_path: Path,
) -> None:
    playback_path, provenance_path = _write_source(tmp_path)
    source_root = tmp_path / "staged_source"
    build_staged_playback_source(
        playback_path=playback_path,
        provenance_path=provenance_path,
        out=source_root,
        boundaries=[80],
    )
    relay = LineagePlaybackRelay(
        source_root=source_root,
        output_root=tmp_path / "relay",
        source_kind="staged_normalized_playback",
    )
    relay.update_once()
    result = relay.update_once()
    assert result["updated"] is True
    playback_before = (tmp_path / "relay" / "playback_snapshot.json").read_bytes()
    idle = relay.update_once()
    assert idle["updated"] is False
    assert idle["status"]["state"] == "source_interrupted"
    assert (
        tmp_path / "relay" / "playback_snapshot.json"
    ).read_bytes() == playback_before
    assert (
        validate_relay_manifest(tmp_path / "relay" / "relay_manifest.json")[
            "sequence_number"
        ]
        == 1
    )


def test_restart_resumes_without_duplicate_sequence(tmp_path: Path) -> None:
    playback_path, provenance_path = _write_source(tmp_path)
    source_root = tmp_path / "staged_source"
    build_staged_playback_source(
        playback_path=playback_path,
        provenance_path=provenance_path,
        out=source_root,
        boundaries=[50, 100],
    )
    output_root = tmp_path / "relay"
    relay = LineagePlaybackRelay(
        source_root=source_root,
        output_root=output_root,
        source_kind="staged_normalized_playback",
    )
    relay.update_once()
    relay.update_once()

    restarted = LineagePlaybackRelay(
        source_root=source_root,
        output_root=output_root,
        source_kind="staged_normalized_playback",
    )
    probe = restarted.update_once()
    assert probe["updated"] is False
    result = restarted.update_once()
    assert result["updated"] is True
    assert result["manifest"]["sequence_number"] == 2
    assert result["manifest"]["latest_exported_playback_boundary"] == 100


def test_accepted_source_hash_change_fails_closed_and_keeps_last_playback(
    tmp_path: Path,
) -> None:
    playback_path, provenance_path = _write_source(tmp_path)
    source_root = tmp_path / "staged_source"
    build_staged_playback_source(
        playback_path=playback_path,
        provenance_path=provenance_path,
        out=source_root,
        boundaries=[50, 100],
    )
    output_root = tmp_path / "relay"
    relay = LineagePlaybackRelay(
        source_root=source_root,
        output_root=output_root,
        source_kind="staged_normalized_playback",
    )
    relay.update_once()
    relay.update_once()
    playback_before = (output_root / "playback_snapshot.json").read_bytes()

    accepted_stage_playback = (
        source_root / "stages" / "stage_000001" / "playback_snapshot.json"
    )
    tampered = json.loads(accepted_stage_playback.read_text())
    tampered["metadata"]["notes"] = "tampered after acceptance"
    accepted_stage_playback.write_text(json.dumps(tampered))

    failed = relay.update_once()
    assert failed["updated"] is False
    assert failed["status"]["state"] == "integrity_error"
    assert "accepted source hash changed" in failed["status"]["last_error"]
    assert (output_root / "playback_snapshot.json").read_bytes() == playback_before


def test_missing_boundary_gap_is_detected(tmp_path: Path) -> None:
    playback_path, provenance_path = _write_source(tmp_path)
    source_root = tmp_path / "staged_source"
    manifest = build_staged_playback_source(
        playback_path=playback_path,
        provenance_path=provenance_path,
        out=source_root,
        boundaries=[50, 100],
    )
    manifest["stages"][1]["expected_previous_boundary"] = 25
    write_canonical_json(source_root / "staged_manifest.json", manifest, overwrite=True)
    relay = LineagePlaybackRelay(
        source_root=source_root,
        output_root=tmp_path / "relay",
        source_kind="staged_normalized_playback",
    )
    relay.update_once()
    relay.update_once()
    failed = relay.update_once()
    assert failed["updated"] is False
    assert failed["status"]["state"] == "integrity_error"
    assert "staged boundary gap" in failed["status"]["last_error"]


def test_validate_manifest_rejects_wrong_playback_hash(tmp_path: Path) -> None:
    playback_path, provenance_path = _write_source(tmp_path)
    source_root = tmp_path / "staged_source"
    build_staged_playback_source(
        playback_path=playback_path,
        provenance_path=provenance_path,
        out=source_root,
        boundaries=[50],
    )
    relay = LineagePlaybackRelay(
        source_root=source_root,
        output_root=tmp_path / "relay",
        source_kind="staged_normalized_playback",
    )
    relay.update_once()
    relay.update_once()
    manifest_path = tmp_path / "relay" / "relay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["playback_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RelayIntegrityError, match="playback hash mismatch"):
        validate_relay_manifest(manifest_path)


def test_poll_interval_must_not_be_too_fast(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one second"):
        LineagePlaybackRelay(
            source_root=tmp_path, output_root=tmp_path / "relay", poll_seconds=0.5
        )


def test_gv2_source_terminal_state_ignores_deliberate_intermediate_interruptions(
    tmp_path: Path,
) -> None:
    source = Gv2CheckpointSource(
        tmp_path / "run", cell_id="cell-a", work_root=tmp_path / "work"
    )
    source.source_root.mkdir(parents=True)
    (source.source_root / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_execution_manifest_v1",
                "status": "interrupted",
            }
        ),
        encoding="utf-8",
    )
    assert source.source_state() == "watching"

    (source.source_root / "qrr_run_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_qrr_run_receipt_v1",
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    assert source.source_state() == "source_complete"

    terminal_core = {
        "schema_version": "eiel_canopy_source_terminal_v1",
        "state": "source_interrupted",
    }
    (source.source_root / "canopy_source_terminal.json").write_text(
        json.dumps(
            {
                **terminal_core,
                "record_sha256": _terminal_digest(terminal_core),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RelayIntegrityError, match="contradicts a completed"):
        source.source_state()


def test_gv2_scan_reuses_one_decoded_checkpoint_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Gv2CheckpointSource(
        tmp_path / "run", cell_id="cell-a", work_root=tmp_path / "work"
    )
    first = tmp_path / "run" / "bundle_tick000000000017" / "projection.json"
    second = tmp_path / "run" / "bundle_tick000000000033" / "projection.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    snapshots = {
        first: SimpleNamespace(
            path=first,
            absolute_tick=17,
            event_boundary_tick=0,
            source_generation=0,
            source_tick=17,
        ),
        second: SimpleNamespace(
            path=second,
            absolute_tick=33,
            event_boundary_tick=0,
            source_generation=0,
            source_tick=33,
        ),
    }
    decode_calls: list[Path] = []
    shared_snapshot_ids: list[tuple[int, ...]] = []

    monkeypatch.setattr(
        "tools.visualization.living_lineage_canopy.python.relay.projection_paths_for_cell",
        lambda _root, _cell: [first, second],
    )
    monkeypatch.setattr(source, "source_state", lambda: "watching")
    monkeypatch.setattr(source, "_stable_path", lambda _path: True)
    monkeypatch.setattr(source, "_build_source_view", lambda _paths: tmp_path / "view")

    def decode(path: Path):
        decode_calls.append(path)
        return snapshots[path]

    def export(_root: Path, **kwargs):
        shared = kwargs["snapshots"]
        shared_snapshot_ids.append(tuple(id(snapshot) for snapshot in shared))
        return (
            {"metadata": {"branch_semantics": "lineage_root_projection"}},
            {},
            {"classification": "EXP-B", "semantics_mode": "actual_agents"},
        )

    def cross_check(_root: Path, _playback: dict, **kwargs):
        shared = kwargs["snapshots"]
        shared_snapshot_ids.append(tuple(id(snapshot) for snapshot in shared))
        return {"ok": True, "mismatches": []}

    monkeypatch.setattr(
        "tools.visualization.living_lineage_canopy.python.relay.snapshot_from_projection",
        decode,
    )
    monkeypatch.setattr(
        "tools.visualization.living_lineage_canopy.python.relay.export_gv2_checkpoint_run",
        export,
    )
    monkeypatch.setattr(
        "tools.visualization.living_lineage_canopy.python.relay.cross_check_gv2",
        cross_check,
    )

    update = source.scan(accepted_boundary=None)

    assert update.boundary == 33
    assert decode_calls == [first, second]
    assert shared_snapshot_ids == [
        (id(snapshots[first]), id(snapshots[second])),
        (id(snapshots[first]), id(snapshots[second])),
    ]

    decode_calls.clear()
    with pytest.raises(SourceNotReady, match="no new stable checkpoint boundary"):
        source.scan(accepted_boundary=33)
    assert decode_calls == []

    source._published_stable_paths = ()
    source._published_boundary = None
    with pytest.raises(SourceNotReady, match="no new stable checkpoint boundary"):
        source.scan(accepted_boundary=33)
    assert decode_calls == [first, second]
    decode_calls.clear()
    with pytest.raises(SourceNotReady, match="no new stable checkpoint boundary"):
        source.scan(accepted_boundary=33)
    assert decode_calls == []


def test_relay_scan_uses_one_run_group_checkpoint_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relay = object.__new__(LineagePlaybackRelay)
    relay.output_root = tmp_path / "group" / "cell-a"
    relay.output_root.mkdir(parents=True)
    expected = SimpleNamespace(boundary=33)
    accepted: list[int | None] = []
    relay.source = SimpleNamespace(
        scan=lambda boundary: accepted.append(boundary) or expected
    )
    lock_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "tools.visualization.living_lineage_canopy.python.relay.fcntl.flock",
        lambda fd, operation: lock_calls.append((fd, operation)),
    )

    result = relay._scan_source(17)

    assert result is expected
    assert accepted == [17]
    assert (tmp_path / "group" / ".checkpoint_scan.lock").is_file()
    assert [operation for _fd, operation in lock_calls] == [
        fcntl.LOCK_EX,
        fcntl.LOCK_UN,
    ]
    assert lock_calls[0][0] == lock_calls[1][0]


def test_gv2_terminal_without_checkpoint_stops_watcher_and_checks_digest(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "run"
    source_root.mkdir()
    terminal_core = {
        "schema_version": "eiel_canopy_source_terminal_v1",
        "state": "source_interrupted",
        "reason": "launch failed before first checkpoint",
    }
    terminal_path = source_root / "canopy_source_terminal.json"
    terminal_path.write_text(
        json.dumps({**terminal_core, "record_sha256": _terminal_digest(terminal_core)}),
        encoding="utf-8",
    )
    relay = LineagePlaybackRelay(
        source_root=source_root,
        output_root=tmp_path / "relay",
        source_kind="gv2_checkpoint_bundle_v2",
        cell_id="cell-a",
    )

    result = relay.update_once()

    assert result["updated"] is False
    assert result["status"]["state"] == "source_interrupted"

    terminal_path.write_text(
        json.dumps({**terminal_core, "record_sha256": "0" * 64}),
        encoding="utf-8",
    )
    tampered = LineagePlaybackRelay(
        source_root=source_root,
        output_root=tmp_path / "tampered-relay",
        source_kind="gv2_checkpoint_bundle_v2",
        cell_id="cell-a",
    ).update_once()
    assert tampered["status"]["state"] == "integrity_error"
    assert "digest-mismatched" in tampered["error"]


def test_gv2_terminal_source_waits_for_its_latest_checkpoint_to_stabilize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "run"
    first = source_root / "checkpoint-33.json"
    final = source_root / "checkpoint-49.json"
    first.parent.mkdir(parents=True)
    first.write_text("{}", encoding="utf-8")
    final.write_text("{}", encoding="utf-8")
    source = Gv2CheckpointSource(
        source_root, cell_id="cell-a", work_root=tmp_path / "work"
    )
    source._last_seen[first] = FileFingerprint.from_path(first)
    monkeypatch.setattr(
        "tools.visualization.living_lineage_canopy.python.relay.projection_paths_for_cell",
        lambda _root, _cell: [first, final],
    )
    monkeypatch.setattr(source, "source_state", lambda: "source_complete")

    with pytest.raises(SourceNotReady, match="awaiting stability") as pending:
        source.scan(accepted_boundary=33)

    assert pending.value.source_state == "watching"
    assert final in source._last_seen


def test_relay_allows_terminal_state_update_at_same_playback_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relay = LineagePlaybackRelay(
        source_root=tmp_path / "source",
        output_root=tmp_path / "relay",
        source_kind="gv2_checkpoint_bundle_v2",
    )
    manifest = {
        "latest_exported_playback_boundary": 33,
        "sequence_number": 1,
        "relay_id": "relay-test",
        "source_state": "watching",
    }
    update = SimpleNamespace(boundary=33, source_state="source_complete")
    monkeypatch.setattr(relay, "_load_manifest_if_present", lambda: manifest)
    monkeypatch.setattr(relay, "_verify_accepted_source_hashes", lambda: None)
    monkeypatch.setattr(relay.source, "scan", lambda _boundary: update)
    monkeypatch.setattr(
        relay,
        "_publish",
        lambda value, **_kwargs: {
            "updated": True,
            "manifest": {"sequence_number": 2},
            "status": {"state": value.source_state},
        },
    )

    result = relay.update_once()

    assert result["updated"] is True
    assert result["status"]["state"] == "source_complete"

    manifest["source_state"] = "source_complete"
    idle = relay.update_once()
    assert idle["updated"] is False
    assert idle["status"]["state"] == "source_complete"
