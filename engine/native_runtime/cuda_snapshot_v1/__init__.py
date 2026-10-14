"""Coarse Python control boundary for native snapshot capsule V1.

The current packer is an explicitly dense parity oracle: it allocates native
per-section vectors, a final capsule, and one returned Python ``bytes`` copy.
It is suitable for exact reconstruction tests, not publication.  The compact
stage writer below is the first-class data-plane artifact: it projects active
columns natively and returns only a bounded completion descriptor.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from engine.native_runtime.cuda_snapshot_plan_v1 import (
    CompiledCudaSnapshotPlanV1,
    validate_cuda_snapshot_plan_v1,
)

from .format import (
    SNAPSHOT_CAPSULE_ABI_V1,
    SNAPSHOT_EXTENSION_VERSION_V1,
    SNAPSHOT_COMPACT_STAGE_ENCODING_POLICY_V1,
    SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1,
    SnapshotPlanContractError,
    canonical_plan_bytes_v1,
    capsule_format_manifest_v1,
    normalize_snapshot_plan_v1,
    plan_identities_v1,
)
from .native_loader import (
    LoadedNativeSnapshotExtensionV1,
    NativeSnapshotLoadError,
    load_prebuilt_snapshot_extension_v1,
)

_NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1 = (
    "eiel_native_snapshot_v1_stage_file_receipt_v1"
)


def _decode_plan(
    value: CompiledCudaSnapshotPlanV1 | Mapping[str, Any] | bytes,
) -> tuple[dict[str, Any], bytes]:
    """Resolve only an authoritative, exact V1 semantic plan."""

    compiled = validate_cuda_snapshot_plan_v1(value)
    return compiled.to_mapping(), compiled.canonical_plan_bytes


def compile_snapshot_plan_v1(
    plan: CompiledCudaSnapshotPlanV1 | Mapping[str, Any] | bytes,
    *,
    extension_path: str | os.PathLike[str] | None = None,
    loaded_extension: LoadedNativeSnapshotExtensionV1 | None = None,
) -> Any:
    """Compile one semantic plan into a persistent native host plan.

    Large state/metric inputs cross Python only later in one native
    ``pack_capsule`` call.  Plan compilation is intentionally outside the
    snapshot boundary.
    """

    decoded, canonical = _decode_plan(plan)
    normalized = normalize_snapshot_plan_v1(decoded)
    loaded = loaded_extension or load_prebuilt_snapshot_extension_v1(extension_path)
    native_plan = loaded.module.compile_snapshot_plan_v1(canonical, normalized)
    expected = plan_identities_v1(normalized, canonical)
    if native_plan.identities() != expected:
        raise NativeSnapshotLoadError("compiled native snapshot plan identities differ")
    return native_plan


def inspect_snapshot_capsule_v1(
    capsule: bytes,
    *,
    expected_plan: Any | None = None,
    extension_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate and inspect a capsule without creating snapshot row objects."""

    if expected_plan is not None:
        return dict(expected_plan.inspect_capsule(capsule))
    loaded = load_prebuilt_snapshot_extension_v1(extension_path)
    return dict(loaded.module.inspect_snapshot_capsule_v1(capsule))


def extract_snapshot_capsule_section_v1(
    capsule: bytes,
    name: str,
    *,
    expected_plan: Any | None = None,
    extension_path: str | os.PathLike[str] | None = None,
) -> bytes:
    """Return one verified section for parity/converter tooling."""

    if expected_plan is not None:
        return bytes(expected_plan.extract_section(capsule, name))
    loaded = load_prebuilt_snapshot_extension_v1(extension_path)
    return bytes(loaded.module.extract_snapshot_capsule_section_v1(capsule, name))


def write_owned_compact_snapshot_capsule_stage_file_v1(
    native_plan: Any,
    *,
    state: Any,
    metrics: Any,
    absolute_tick: int,
    stage_path: str | os.PathLike[str],
    encoding_policy: str = SNAPSHOT_COMPACT_STAGE_ENCODING_POLICY_V1,
) -> dict[str, Any]:
    """Project and content-fsync one exclusive unpublished stage file.

    ``state`` may contain either the canonical checkpoint prefix or the full
    resident state row.  No capsule or section payload bytes cross back into
    Python on this hot path.  The parent directory is intentionally not fsynced,
    so this API does not claim crash durability for the directory entry.
    """

    if type(absolute_tick) is not int:
        raise SnapshotPlanContractError("snapshot absolute tick must be an integer")
    raw_path = os.fspath(stage_path)
    if not isinstance(raw_path, str) or not os.path.isabs(raw_path) or "\x00" in raw_path:
        raise SnapshotPlanContractError(
            "compact snapshot stage_path must be one absolute text path"
        )
    return dict(
        native_plan.write_owned_compact_capsule_stage_file(
            state,
            metrics,
            absolute_tick,
            raw_path,
            encoding_policy,
        )
    )


def convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1(
    native_plan: Any,
    *,
    capsule_stage_path: str | os.PathLike[str],
    gzip_stage_path: str | os.PathLike[str],
    encoding_policy: str = SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1,
) -> dict[str, Any]:
    """Convert one authenticated compact capsule into exact canonical V1 gzip.

    Both paths are supplied unchanged to the native no-symlink, identity-bound
    stage-file implementation.  The output is exclusive-created and content
    fsynced, but its directory entry is intentionally not fsynced or published.
    Only a bounded completion descriptor crosses back into Python.
    """

    raw_capsule_path = os.fspath(capsule_stage_path)
    raw_gzip_path = os.fspath(gzip_stage_path)
    for label, raw_path in (
        ("capsule_stage_path", raw_capsule_path),
        ("gzip_stage_path", raw_gzip_path),
    ):
        if (
            not isinstance(raw_path, str)
            or not os.path.isabs(raw_path)
            or "\x00" in raw_path
        ):
            raise SnapshotPlanContractError(
                f"snapshot compatibility {label} must be one absolute text path"
            )
    if raw_capsule_path == raw_gzip_path:
        raise SnapshotPlanContractError(
            "snapshot compatibility input and output paths must be distinct"
        )
    return dict(
        native_plan.convert_compact_capsule_stage_file_to_v1_gzip_stage_file(
            raw_capsule_path,
            raw_gzip_path,
            encoding_policy,
        )
    )


def write_owned_compact_snapshot_to_v1_gzip_stage_file_v1(
    native_plan: Any,
    *,
    state: Any,
    metrics: Any,
    absolute_tick: int,
    stage_path: str | os.PathLike[str],
    encoding_policy: str = SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1,
) -> dict[str, Any]:
    """Project, convert, and fsync one exact V1 gzip stage in native code.

    This is the fused compatibility path: compact projection, capsule
    authentication, canonical JSONL reconstruction, gzip, hashing, and stage
    publication happen in one native call.  No capsule stage is written or
    reopened, and no snapshot payload bytes cross into Python.
    """

    if type(absolute_tick) is not int:
        raise SnapshotPlanContractError("snapshot absolute tick must be an integer")
    raw_path = os.fspath(stage_path)
    if (
        not isinstance(raw_path, str)
        or not os.path.isabs(raw_path)
        or "\x00" in raw_path
    ):
        raise SnapshotPlanContractError(
            "fused compatibility stage_path must be one absolute text path"
        )
    if encoding_policy != SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1:
        raise SnapshotPlanContractError(
            "fused compatibility stage encoding policy is unsupported"
        )
    result = dict(
        native_plan.write_owned_compact_capsule_to_v1_gzip_stage_file(
            state,
            metrics,
            absolute_tick,
            raw_path,
            encoding_policy,
        )
    )
    if result.get("schema_version") != _NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1:
        raise NativeSnapshotLoadError(
            "fused compatibility stage receipt schema differs"
        )
    if (
        result.get("stage_path") != raw_path
        or result.get("stage_encoding_policy") != encoding_policy
        or result.get("stage_file_state") != "FSYNCED_STAGE_ONLY_NOT_PUBLISHED"
        or result.get("stage_file_fsync_completed") is not True
        or result.get("stage_file_close_completed") is not True
        or result.get("atomic_publication_performed") is not False
        or result.get("directory_fsync_performed") is not False
        or result.get("input_ownership_contract")
        != "native_service_exclusive_tensor_lease_until_completion_v1"
        or result.get("gil_held_for_full_synchronous_call") is not False
        or result.get("python_row_materialization") is not False
        or result.get("capsule_stage_written") is not False
    ):
        raise NativeSnapshotLoadError("fused compatibility stage contract differs")
    timing = result.get("timing")
    if not isinstance(timing, Mapping) or not isinstance(
        timing.get("native_encode_write_fsync_close_ns"), int
    ):
        raise NativeSnapshotLoadError(
            "fused compatibility stage timing is absent"
        )
    return result


def inspect_compact_snapshot_capsule_v1(
    capsule: bytes,
    *,
    expected_plan: Any,
) -> dict[str, Any]:
    """Validate the compact profile and its plan-bound columnar invariants."""

    return dict(expected_plan.inspect_compact_capsule(capsule))


def extract_compact_snapshot_capsule_section_v1(
    capsule: bytes,
    name: str,
    *,
    expected_plan: Any,
) -> bytes:
    """Return one verified compact section for offline converters/tests."""

    return bytes(expected_plan.extract_compact_section(capsule, name))


__all__ = [
    "LoadedNativeSnapshotExtensionV1",
    "NativeSnapshotLoadError",
    "SNAPSHOT_CAPSULE_ABI_V1",
    "SNAPSHOT_EXTENSION_VERSION_V1",
    "SNAPSHOT_COMPACT_STAGE_ENCODING_POLICY_V1",
    "SNAPSHOT_COMPATIBILITY_STAGE_ENCODING_POLICY_V1",
    "SnapshotPlanContractError",
    "canonical_plan_bytes_v1",
    "capsule_format_manifest_v1",
    "compile_snapshot_plan_v1",
    "convert_compact_snapshot_capsule_stage_file_to_v1_gzip_stage_file_v1",
    "extract_snapshot_capsule_section_v1",
    "extract_compact_snapshot_capsule_section_v1",
    "inspect_compact_snapshot_capsule_v1",
    "inspect_snapshot_capsule_v1",
    "load_prebuilt_snapshot_extension_v1",
    "normalize_snapshot_plan_v1",
    "write_owned_compact_snapshot_capsule_stage_file_v1",
    "write_owned_compact_snapshot_to_v1_gzip_stage_file_v1",
]
