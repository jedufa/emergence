from __future__ import annotations

import json

import pytest

from engine.wrc.reducer import (
    RetentionError,
    retain_all,
    retain_compact,
    retain_selected,
)


def _raw_tree(root):
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "metrics.csv").write_text("generation,signal\n0,1\n")
    (root / "raw" / "events.jsonl").write_text('{"tick":0}\n')
    (root / "compact_summary.json").write_text('{"complete":true}\n')
    (root / "provenance.json").write_text('{"source":"fixture"}\n')


def test_retain_all_is_non_pruning_and_hash_verified(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "all"
    _raw_tree(source)
    manifest = retain_all(source, destination)
    assert manifest["source_count"] == 4
    assert manifest["retained_count"] == 4
    assert manifest["raw_prune_complete"] is False
    assert (source / "raw" / "metrics.csv").exists()
    assert json.loads((destination / "retention_manifest.json").read_text())["complete"]


def test_retain_compact_keeps_science_and_provenance_and_can_prune_explicit_raw(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "compact"
    _raw_tree(source)
    (source / "reduction_manifest.json").write_text(
        json.dumps({"complete_world_count": 1, "incomplete_world_count": 0})
    )
    manifest = retain_compact(
        source,
        destination,
        prune_raw=True,
        raw_paths=["raw/metrics.csv", "raw/events.jsonl"],
    )
    assert manifest["mode"] == "retain_compact"
    assert (destination / "compact_summary.json").exists()
    assert (destination / "provenance.json").exists()
    assert not (source / "raw" / "metrics.csv").exists()
    assert not (source / "raw" / "events.jsonl").exists()


def test_expected_world_count_is_verified_from_a_reduction_manifest(tmp_path) -> None:
    source = tmp_path / "source"
    _raw_tree(source)
    (source / "reduction_manifest.json").write_text(
        json.dumps({"complete_world_count": 1, "incomplete_world_count": 0})
    )
    with pytest.raises(RetentionError, match="world count"):
        retain_all(source, tmp_path / "wrong-count", expected_world_count=2)


def test_incomplete_worlds_keep_raw_evidence_on_prune_request(tmp_path) -> None:
    source = tmp_path / "source"
    _raw_tree(source)
    (source / "reduction_manifest.json").write_text(
        json.dumps({"complete_world_count": 0, "incomplete_world_count": 1})
    )
    with pytest.raises(RetentionError, match="unsafe"):
        retain_compact(
            source,
            tmp_path / "incomplete",
            prune_raw=True,
            raw_paths=["raw/metrics.csv"],
        )
    assert (source / "raw" / "metrics.csv").exists()


def test_failed_transaction_keeps_raw_and_does_not_publish_manifest(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "selected"
    _raw_tree(source)
    with pytest.raises(RetentionError):
        retain_selected(
            source,
            destination,
            selected_paths=["raw/missing.json"],
            prune_raw=True,
            raw_paths=["raw/metrics.csv"],
        )
    assert (source / "raw" / "metrics.csv").exists()
    assert not (destination / "retention_manifest.json").exists()


def test_selected_retention_is_deterministic(tmp_path) -> None:
    source = tmp_path / "source"
    _raw_tree(source)
    first = retain_selected(source, tmp_path / "one", selected_paths=["provenance.json"])
    second = retain_selected(source, tmp_path / "two", selected_paths=["provenance.json"])
    assert first["manifest_sha256"] == second["manifest_sha256"]
