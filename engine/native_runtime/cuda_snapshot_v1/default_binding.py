"""Resolve the repository's source-bound native snapshot artifact.

The native snapshot loader intentionally remains strict and accepts an explicit
artifact path.  This module supplies the canonical repository-local path for
operator-facing routes, while preserving the environment variable as an
explicit override for independently provisioned deployments.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

from .native_loader import (
    BOUND_SOURCE_RELATIVE_PATHS_V1,
    NativeSnapshotLoadError,
    PROJECT_ROOT,
    SNAPSHOT_EXTENSION_ENV_V1,
    validate_snapshot_extension_manifest_v1,
)


DEFAULT_SNAPSHOT_EXTENSION_RELATIVE_DIRECTORY_V1: Final[Path] = Path(
    "_artifacts/eiel-native-snapshot-100x-20260828/native-build-host-v1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bound_source_identity_v1() -> str:
    source_inventory: list[dict[str, str]] = []
    for relative in BOUND_SOURCE_RELATIVE_PATHS_V1:
        source = PROJECT_ROOT / relative
        if source.is_symlink() or not source.is_file():
            raise NativeSnapshotLoadError(
                f"canonical native snapshot source is absent or a symlink: {relative}"
            )
        source_inventory.append({"path": relative, "sha256": _sha256_file(source)})
    return _canonical_sha256(source_inventory)


def default_snapshot_extension_path_v1() -> Path:
    """Return the current-checkout artifact bound to the live source identity."""

    source_identity = _bound_source_identity_v1()
    return (
        PROJECT_ROOT
        / DEFAULT_SNAPSHOT_EXTENSION_RELATIVE_DIRECTORY_V1
        / f"eiel_native_snapshot_v1_{source_identity[:16]}.so"
    )


def resolve_snapshot_extension_path_v1() -> Path:
    """Resolve an explicit deployment override or the canonical local artifact."""

    configured = os.environ.get(SNAPSHOT_EXTENSION_ENV_V1, "").strip()
    if configured:
        return Path(configured)
    return default_snapshot_extension_path_v1()


def validate_resolved_snapshot_extension_v1() -> tuple[Path, Path, dict[str, object]]:
    """Validate the resolved artifact before a native execution is attempted."""

    return validate_snapshot_extension_manifest_v1(resolve_snapshot_extension_path_v1())


__all__ = [
    "DEFAULT_SNAPSHOT_EXTENSION_RELATIVE_DIRECTORY_V1",
    "default_snapshot_extension_path_v1",
    "resolve_snapshot_extension_path_v1",
    "validate_resolved_snapshot_extension_v1",
]
