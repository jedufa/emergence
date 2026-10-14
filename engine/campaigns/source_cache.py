"""Content-addressed storage for immutable exact source projections."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping

from .canonical import canonical_sha256, pretty_json


SOURCE_CACHE_SCHEMA_V1 = "eiel_exact_source_cache_v2"


class SourceCacheError(RuntimeError):
    """A source cache entry is unsafe, stale, or malformed."""


@dataclass(frozen=True, slots=True)
class SourceCacheEntry:
    key: str
    payload_path: Path
    manifest_path: Path
    sha256: str
    size_bytes: int
    reused: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, *, name: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise SourceCacheError(f"{name} is unavailable: {path}") from exc
    if not stat.S_ISREG(details.st_mode):
        raise SourceCacheError(f"{name} must be a regular file: {path}")
    return details


def _provenance(
    source_contract: Any,
    source_inputs: Mapping[str, Any],
    relevant_hashes: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(source_inputs, Mapping) or not source_inputs:
        raise SourceCacheError("source_inputs must be a nonempty mapping")
    if not isinstance(relevant_hashes, Mapping) or not relevant_hashes:
        raise SourceCacheError("relevant_hashes must be a nonempty mapping")
    hashes: dict[str, str] = {}
    for name, value in relevant_hashes.items():
        if not isinstance(name, str) or not name or name.strip() != name:
            raise SourceCacheError("relevant hash names must be nonempty strings")
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SourceCacheError(f"relevant hash {name!r} must be lowercase SHA-256")
        hashes[name] = value
    provenance = {
        "source_contract": source_contract,
        "source_inputs": dict(source_inputs),
        "relevant_hashes": hashes,
    }
    try:
        canonical_sha256(provenance)
    except ValueError as exc:
        raise SourceCacheError("source provenance must be canonical JSON data") from exc
    return provenance


def source_cache_key(
    source_contract: Any,
    source_inputs: Mapping[str, Any],
    relevant_hashes: Mapping[str, str],
) -> str:
    """Return the stable key for one exact source-generation contract."""

    return canonical_sha256(
        _provenance(source_contract, source_inputs, relevant_hashes)
    )


def _entry_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    directory = cache_root / key[:2] / key
    return directory / "projection.bin", directory / "manifest.json"


def _expected_manifest(
    *, key: str, provenance: Mapping[str, Any], sha256: str, size_bytes: int
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_CACHE_SCHEMA_V1,
        "key": key,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "provenance": dict(provenance),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    _regular_file(path, name="source cache manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceCacheError("source cache manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise SourceCacheError("source cache manifest must be an object")
    return value


def _verified_existing(
    payload_path: Path,
    manifest_path: Path,
    expected: Mapping[str, Any],
) -> SourceCacheEntry:
    payload_stat = _regular_file(payload_path, name="cached source projection")
    manifest = _read_manifest(manifest_path)
    if manifest != expected:
        raise SourceCacheError("cached source provenance is stale or mismatched")
    if payload_stat.st_size != expected["size_bytes"] or _sha256(payload_path) != expected["sha256"]:
        raise SourceCacheError("cached source content does not match its manifest")
    return SourceCacheEntry(
        key=str(expected["key"]),
        payload_path=payload_path,
        manifest_path=manifest_path,
        sha256=str(expected["sha256"]),
        size_bytes=int(expected["size_bytes"]),
        reused=True,
    )


def cache_source_projection(
    source_path: str | Path,
    cache_root: str | Path,
    *,
    source_contract: Any,
    source_inputs: Mapping[str, Any],
    relevant_hashes: Mapping[str, str],
) -> SourceCacheEntry:
    """Copy an accepted source into the cache or verify exact prior reuse."""

    source = Path(source_path)
    source_stat = _regular_file(source, name="source projection")
    root = Path(cache_root)
    if not root.is_absolute():
        raise SourceCacheError("cache_root must be absolute")
    provenance = _provenance(source_contract, source_inputs, relevant_hashes)
    key = canonical_sha256(provenance)
    source_sha256 = _sha256(source)
    expected = _expected_manifest(
        key=key,
        provenance=provenance,
        sha256=source_sha256,
        size_bytes=source_stat.st_size,
    )
    payload_path, manifest_path = _entry_paths(root, key)
    if payload_path.exists() or manifest_path.exists():
        if not payload_path.exists() or not manifest_path.exists():
            raise SourceCacheError("source cache entry is incomplete")
        return _verified_existing(payload_path, manifest_path, expected)

    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".projection-", dir=payload_path.parent
        )
        os.close(descriptor)
        payload_temp = Path(raw_path)
        shutil.copyfile(source, payload_temp)
        with payload_temp.open("rb") as handle:
            os.fsync(handle.fileno())
        copied_stat = _regular_file(payload_temp, name="temporary cached source")
        if (
            copied_stat.st_dev == source_stat.st_dev
            and copied_stat.st_ino == source_stat.st_ino
        ):
            raise SourceCacheError("cached source must not hardlink the input")
        if copied_stat.st_size != source_stat.st_size or _sha256(payload_temp) != source_sha256:
            raise SourceCacheError("source changed while it was copied")
        os.replace(payload_temp, payload_path)
        payload_temp = None

        descriptor, raw_path = tempfile.mkstemp(
            prefix=".manifest-", dir=manifest_path.parent
        )
        os.close(descriptor)
        manifest_temp = Path(raw_path)
        manifest_temp.write_text(pretty_json(expected), encoding="utf-8")
        with manifest_temp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
    finally:
        for temporary in (payload_temp, manifest_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    created = _verified_existing(payload_path, manifest_path, expected)
    return SourceCacheEntry(
        key=created.key,
        payload_path=created.payload_path,
        manifest_path=created.manifest_path,
        sha256=created.sha256,
        size_bytes=created.size_bytes,
        reused=False,
    )


def load_cached_source_projection(
    cache_root: str | Path,
    *,
    source_contract: Any,
    source_inputs: Mapping[str, Any],
    relevant_hashes: Mapping[str, str],
) -> SourceCacheEntry | None:
    """Load an exact accepted entry without regenerating its source payload."""

    root = Path(cache_root)
    if not root.is_absolute():
        raise SourceCacheError("cache_root must be absolute")
    provenance = _provenance(source_contract, source_inputs, relevant_hashes)
    key = canonical_sha256(provenance)
    payload_path, manifest_path = _entry_paths(root, key)
    if not payload_path.exists() and not manifest_path.exists():
        return None
    if not payload_path.exists() or not manifest_path.exists():
        raise SourceCacheError("source cache entry is incomplete")
    manifest = _read_manifest(manifest_path)
    if set(manifest) != {"schema_version", "key", "sha256", "size_bytes", "provenance"}:
        raise SourceCacheError("cached source manifest has unsupported fields")
    if (
        manifest.get("schema_version") != SOURCE_CACHE_SCHEMA_V1
        or manifest.get("key") != key
        or manifest.get("provenance") != provenance
    ):
        raise SourceCacheError("cached source provenance is stale or mismatched")
    digest = manifest.get("sha256")
    size = manifest.get("size_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise SourceCacheError("cached source manifest content identity is invalid")
    entry = _verified_existing(
        payload_path,
        manifest_path,
        _expected_manifest(
            key=key,
            provenance=provenance,
            sha256=digest,
            size_bytes=size,
        ),
    )
    return entry


def materialize_cached_source(
    entry: SourceCacheEntry,
    destination: str | Path,
    *,
    reuse_existing: bool = False,
) -> Path:
    """Copy a verified cached payload to a distinct destination inode."""

    if not isinstance(entry, SourceCacheEntry):
        raise SourceCacheError("entry must be a SourceCacheEntry")
    payload_stat = _regular_file(entry.payload_path, name="cached source projection")
    if payload_stat.st_size != entry.size_bytes or _sha256(entry.payload_path) != entry.sha256:
        raise SourceCacheError("cached source content changed after acceptance")
    target = Path(destination)
    if target.exists():
        if not reuse_existing:
            raise SourceCacheError("materialization destination already exists")
        target_stat = _regular_file(target, name="materialized source projection")
        if target_stat.st_size != entry.size_bytes or _sha256(target) != entry.sha256:
            raise SourceCacheError("existing materialized source content mismatch")
        if target_stat.st_dev == payload_stat.st_dev and target_stat.st_ino == payload_stat.st_ino:
            raise SourceCacheError("materialized source must use distinct storage")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        shutil.copyfile(entry.payload_path, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    target_stat = _regular_file(target, name="materialized source projection")
    if target_stat.st_dev == payload_stat.st_dev and target_stat.st_ino == payload_stat.st_ino:
        raise SourceCacheError("materialized source must use distinct storage")
    if target_stat.st_size != entry.size_bytes or _sha256(target) != entry.sha256:
        raise SourceCacheError("materialized source content mismatch")
    return target


__all__ = [
    "SOURCE_CACHE_SCHEMA_V1",
    "SourceCacheEntry",
    "SourceCacheError",
    "cache_source_projection",
    "load_cached_source_projection",
    "materialize_cached_source",
    "source_cache_key",
]
