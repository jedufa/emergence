from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from engine.campaigns.source_cache import (
    SourceCacheError,
    cache_source_projection,
    materialize_cached_source,
    source_cache_key,
)


HASHES = {"engine": "a" * 64, "source_schema": "b" * 64}
CONTRACT = {"schema_version": "ecology_source_v1", "world": "patch"}
INPUTS = {
    "seeds": {"ecology": 17, "task": 31},
    "parameters": {"patch.founder_placement": "balanced"},
}


def _cache(source: Path, root: Path, *, source_inputs=INPUTS):
    return cache_source_projection(
        source,
        root,
        source_contract=CONTRACT,
        source_inputs=source_inputs,
        relevant_hashes=HASHES,
    )


def test_key_is_canonical_and_changes_with_provenance() -> None:
    left = source_cache_key(CONTRACT, INPUTS, HASHES)
    reordered = source_cache_key(
        {"world": "patch", "schema_version": "ecology_source_v1"},
        {
            "parameters": {"patch.founder_placement": "balanced"},
            "seeds": {"task": 31, "ecology": 17},
        },
        {"source_schema": "b" * 64, "engine": "a" * 64},
    )

    assert left == reordered
    changed_task = {**INPUTS, "seeds": {"ecology": 17, "task": 32}}
    changed_founder = {
        **INPUTS,
        "parameters": {"patch.founder_placement": "legacy"},
    }
    assert source_cache_key(CONTRACT, changed_task, HASHES) != left
    assert source_cache_key(CONTRACT, changed_founder, HASHES) != left


def test_cache_copy_manifest_reuse_and_materialization_are_isolated(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"accepted ecology projection\x00")
    root = (tmp_path / "cache").resolve()

    created = _cache(source, root)
    reused = _cache(source, root)
    materialized = materialize_cached_source(created, tmp_path / "run" / "source.bin")

    manifest = json.loads(created.manifest_path.read_text(encoding="utf-8"))
    assert created.reused is False
    assert reused.reused is True
    assert reused.payload_path == created.payload_path
    assert manifest["key"] == created.key
    assert manifest["sha256"] == created.sha256
    assert manifest["size_bytes"] == source.stat().st_size
    assert manifest["provenance"]["source_inputs"] == INPUTS
    assert materialized.read_bytes() == source.read_bytes()
    assert os.path.samestat(source.stat(), created.payload_path.stat()) is False
    assert os.path.samestat(created.payload_path.stat(), materialized.stat()) is False


def test_stale_manifest_and_changed_cached_content_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    entry = _cache(source, (tmp_path / "cache").resolve())

    manifest = json.loads(entry.manifest_path.read_text(encoding="utf-8"))
    manifest["provenance"]["source_inputs"]["seeds"]["task"] = 99
    entry.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SourceCacheError, match="provenance is stale"):
        _cache(source, (tmp_path / "cache").resolve())

    manifest["provenance"]["source_inputs"] = INPUTS
    entry.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    entry.payload_path.write_bytes(b"tampered")
    with pytest.raises(SourceCacheError, match="content does not match"):
        _cache(source, (tmp_path / "cache").resolve())


def test_symlink_input_and_cached_payload_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    symlink = tmp_path / "source-link.bin"
    symlink.symlink_to(source)

    with pytest.raises(SourceCacheError, match="regular file"):
        _cache(symlink, (tmp_path / "cache").resolve())

    entry = _cache(source, (tmp_path / "cache").resolve())
    entry.payload_path.unlink()
    entry.payload_path.symlink_to(source)
    with pytest.raises(SourceCacheError, match="regular file"):
        _cache(source, (tmp_path / "cache").resolve())


def test_incomplete_entry_and_existing_destination_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    root = (tmp_path / "cache").resolve()
    key = source_cache_key(CONTRACT, INPUTS, HASHES)
    incomplete = root / key[:2] / key
    incomplete.mkdir(parents=True)
    (incomplete / "projection.bin").write_bytes(b"source")

    with pytest.raises(SourceCacheError, match="incomplete"):
        _cache(source, root)

    incomplete.joinpath("projection.bin").unlink()
    incomplete.rmdir()
    entry = _cache(source, root)
    destination = tmp_path / "destination.bin"
    destination.write_bytes(b"keep")
    with pytest.raises(SourceCacheError, match="already exists"):
        materialize_cached_source(entry, destination)
    assert destination.read_bytes() == b"keep"


def test_exact_existing_materialization_is_idempotently_reused(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    entry = _cache(source, (tmp_path / "cache").resolve())
    destination = tmp_path / "destination.bin"
    first = materialize_cached_source(entry, destination)

    second = materialize_cached_source(entry, destination, reuse_existing=True)

    assert second == first
    destination.write_bytes(b"changed")
    with pytest.raises(SourceCacheError, match="content mismatch"):
        materialize_cached_source(entry, destination, reuse_existing=True)
