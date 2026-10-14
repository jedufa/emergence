"""Read-only discovery and delivery of Living Lineage Canopy relay playbacks."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlencode


RELAY_MANIFEST_NAME = "relay_manifest.json"
RELAY_SCHEMA = "llc_near_live_playback_relay_v0_manifest"
MAX_SCAN_DIRS = 25_000
MAX_SCAN_DEPTH = 14
MAX_MANIFEST_BYTES = 512 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_ARTIFACT_NAMES = {"manifest", "playback", "provenance", "status"}
_MANIFEST_ARTIFACT_KEYS = {
    "playback": "playback_file_name",
    "provenance": "provenance_file_name",
    "status": "status_file_name",
}


class CanopyLibraryError(ValueError):
    """A requested library item is unavailable or does not meet its contract."""


def _safe_relative(value: object) -> PurePosixPath:
    raw = str(value or "").replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts or str(candidate) == ".":
        raise CanopyLibraryError("relay artifact path is not a safe relative path")
    return candidate


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != RELAY_SCHEMA:
        return None
    if not isinstance(value.get("source_run_id"), str) or not value["source_run_id"].strip():
        return None
    try:
        for key in ("playback_file_name", "provenance_file_name"):
            artifact = (path.parent / _safe_relative(value.get(key))).resolve()
            artifact.relative_to(path.parent.resolve())
            if not artifact.is_file() or artifact.stat().st_size > MAX_ARTIFACT_BYTES:
                return None
    except (CanopyLibraryError, OSError, ValueError):
        return None
    return value


def _walk_manifests(root: Path, *, scan_depth: int) -> Iterable[Path]:
    """Yield bounded candidate manifests without following links outside *root*."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        return
    scanned = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False, onerror=lambda _error: None):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError:
            directories[:] = []
            continue
        if depth >= scan_depth:
            directories[:] = []
        directories[:] = [name for name in directories if not (current_path / name).is_symlink()]
        scanned += 1
        if scanned > MAX_SCAN_DIRS:
            return
        if RELAY_MANIFEST_NAME in files:
            candidate = current_path / RELAY_MANIFEST_NAME
            if candidate.is_file() and not candidate.is_symlink():
                yield candidate


def _entry(*, root_index: int, root_label: str, root: Path, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    relay_root = manifest_path.parent
    relative = relay_root.relative_to(root).as_posix()
    key = f"root{root_index}/{relative}"
    return {
        "key": key,
        "label": str(manifest["source_run_id"]),
        "location": relative,
        "storage": root_label,
        "modified_at_ns": manifest_path.stat().st_mtime_ns,
        "source_state": str(manifest.get("source_state") or "saved"),
        "semantics_mode": str(manifest.get("semantics_mode") or "partial_agents"),
        "manifest_url": "/api/canopy/playback?" + urlencode({"key": key, "artifact": "manifest"}),
    }


def discover(roots: Iterable[tuple[str, Path]], *, scan_depth: int = MAX_SCAN_DEPTH) -> list[dict[str, Any]]:
    """Return valid, local relay libraries from explicit trusted roots only."""

    rows: list[dict[str, Any]] = []
    for index, (label, raw_root) in enumerate(roots):
        root = raw_root.expanduser().resolve()
        for manifest_path in _walk_manifests(root, scan_depth=max(1, min(MAX_SCAN_DEPTH, int(scan_depth)) )):
            manifest = _read_manifest(manifest_path)
            if manifest is None:
                continue
            try:
                rows.append(_entry(root_index=index, root_label=label, root=root, manifest_path=manifest_path, manifest=manifest))
            except (OSError, ValueError):
                continue
    return sorted(rows, key=lambda row: (-int(row["modified_at_ns"]), str(row["label"]).lower(), str(row["key"])))


def _lookup(roots: Iterable[tuple[str, Path]], key: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(key, str) or not key.startswith("root") or "/" not in key:
        raise CanopyLibraryError("invalid playback key")
    prefix, relative = key.split("/", 1)
    try:
        index = int(prefix.removeprefix("root"))
        root = list(roots)[index][1].expanduser().resolve()
        relay_root = (root / _safe_relative(relative)).resolve()
    except (IndexError, ValueError, CanopyLibraryError) as exc:
        raise CanopyLibraryError("invalid playback key") from exc
    try:
        relay_root.relative_to(root)
    except ValueError as exc:
        raise CanopyLibraryError("playback key escapes its storage root") from exc
    manifest_path = relay_root / RELAY_MANIFEST_NAME
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        raise CanopyLibraryError("playback is no longer available or valid")
    return relay_root, manifest


def artifact(roots: Iterable[tuple[str, Path]], *, key: str, name: str) -> tuple[bytes, str]:
    """Read one declared relay artifact; never expose arbitrary storage files."""

    if name not in _ARTIFACT_NAMES:
        raise CanopyLibraryError("unsupported playback artifact")
    relay_root, manifest = _lookup(roots, key)
    if name == "manifest":
        path = relay_root / RELAY_MANIFEST_NAME
    else:
        declared = manifest.get(_MANIFEST_ARTIFACT_KEYS[name])
        if not declared:
            raise CanopyLibraryError("playback artifact is not declared")
        path = (relay_root / _safe_relative(declared)).resolve()
        try:
            path.relative_to(relay_root)
        except ValueError as exc:
            raise CanopyLibraryError("playback artifact escapes relay directory") from exc
    try:
        if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise CanopyLibraryError("playback artifact is unavailable or exceeds its size limit")
        if name == "manifest":
            response_manifest = dict(manifest)
            for artifact_name, manifest_key in _MANIFEST_ARTIFACT_KEYS.items():
                if response_manifest.get(manifest_key):
                    response_manifest[manifest_key] = "/api/canopy/playback?" + urlencode(
                        {"key": key, "artifact": artifact_name}
                    )
            return (
                json.dumps(response_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                "application/json; charset=utf-8",
            )
        return path.read_bytes(), "application/json; charset=utf-8"
    except OSError as exc:
        raise CanopyLibraryError("playback artifact cannot be read") from exc
