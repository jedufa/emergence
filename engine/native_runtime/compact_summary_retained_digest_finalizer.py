"""Finalize compact wire digests from one detached retained-authority row."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .checkpoint_digests import (
    SCHEMA_VERSION as CHECKPOINT_LAYER_SCHEMA_VERSION,
    SOURCE_AUTHORITY_CURSOR_METADATA_FIELDS,
    SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS,
    SOURCE_RNG_REQUIRED_METADATA_FIELDS,
    SOURCE_RNG_SCHEMA_VERSION,
    SOURCE_RNG_TENSOR_FIELDS,
    DigestLayer,
)
from .compact_summary_cpu import WIRE_SCHEMA_VERSION
from .compact_summary_digests import (
    ACTIVE_AGENT_IDS_SCHEMA_VERSION,
    POLICY_MUTATION_TAPE_SCHEMA_VERSION,
    RECOVERABLE_ERROR_SCHEMA_VERSION,
    active_agent_ids_digest_from_tensor_sha256,
    behavior_rows_digest_sha256,
    checkpoint_layer_digest_from_tensor_manifest,
    checkpoint_rng_digest_from_tensor_manifest,
    checkpoint_source_projection_digest_from_tensor_manifest,
    lineage_records_digest_sha256,
    policy_mutation_tape_digest_from_tensor_sha256,
    recoverable_error_digest_sha256,
)
from .compact_summary_planes import DIGEST_FIELDS
from .compact_summary_reporting_projection import (
    REPORTING_AUTHORITY_SCHEMA_VERSION,
    REPORTING_RETAINED_LAYOUT_VERSION,
)
from .lifecycle_events import SCHEMA_VERSION as TYPED_LIFECYCLE_SCHEMA_VERSION


FINALIZER_SOURCE_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_retained_digest_finalizer_source_v1"
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "wire_schema_version",
    "generation",
    "cell",
    "world",
    "checkpoint",
    "identity",
    "policy_mutation_tape",
    "reporting",
    "lifecycle",
    "terminal",
)
_CHECKPOINT_FIELDS: Final[tuple[str, ...]] = (
    "source_version",
    "rng_source_version",
    "projection_schema_version",
    "projection_fields",
    "tensor_manifest",
    "trajectory_state_fields",
    "canonical_checkpoint_storage_fields",
)
_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "source_version",
    "active_agent_count",
    "tensor_sha256",
)
_POLICY_FIELDS: Final[tuple[str, ...]] = (
    "source_version",
    "cursor",
    "consumed_shapes",
    "tensor_sha256",
)
_REPORTING_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "retained_layout_version",
    "generation",
    "cell",
    "start_id_ceiling",
    "behavior_rows",
    "lineage_records",
    "behavior_rows_digest_sha256",
    "lineage_records_digest_sha256",
)
_LIFECYCLE_FIELDS: Final[tuple[str, ...]] = (
    "source_version",
    "generation",
    "cell",
    "event_cursor_start",
    "event_cursor_pre_turnover",
    "event_cursor_end",
    "authoritative_event_cursor",
    "event_row_count",
    "active_prefix_digest_sha256",
)
_TERMINAL_FIELDS: Final[tuple[str, ...]] = (
    "source_version",
    "generation",
    "cell",
    "recoverable_error_code",
    "recoverable_error_details",
)
_SOURCE_TENSOR_DESCRIPTOR_FIELDS: Final[tuple[str, ...]] = (
    "device",
    "dtype",
    "shape",
    "sha256",
)


class CompactSummaryRetainedDigestFinalizerError(ValueError):
    """A retained digest authority is missing, malformed, or contradictory."""


def _fail(path: str, message: str) -> None:
    raise CompactSummaryRetainedDigestFinalizerError(f"{path}: {message}")


def _exact_mapping(
    value: Any,
    fields: Sequence[str],
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if any(type(name) is not str for name in value):
        raise TypeError(f"{path} keys must be strings")
    missing = sorted(set(fields) - set(value))
    extra = sorted(set(value) - set(fields))
    if missing or extra:
        _fail(path, f"fields mismatch: missing={missing} extra={extra}")
    return value


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if any(type(name) is not str for name in value):
        raise TypeError(f"{path} keys must be strings")
    return value


def _strict_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an int")
    if value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _strict_nonempty_string(value: Any, *, path: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _strict_sha256(value: Any, *, path: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(path, "must be a lowercase SHA-256 digest")
    return value


def _require_version(value: Any, expected: str, *, path: str) -> None:
    if type(value) is not str or value != expected:
        _fail(path, f"must equal {expected!r}")


def _require_detached_cpu_metadata(value: Any, *, path: str = "authority") -> None:
    if hasattr(value, "detach"):
        _fail(path, "must not contain tensors; transfer and prehash before finalization")
    value_type = type(value)
    if value_type.__module__.startswith("numpy") and value_type.__name__ == "ndarray":
        _fail(path, "must not contain arrays; transfer and prehash before finalization")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_detached_cpu_metadata(key, path=f"{path}.<key>")
            _require_detached_cpu_metadata(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_detached_cpu_metadata(child, path=f"{path}[{index}]")


def _shape(value: Any, *, path: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a list")
    return [
        _strict_int(dimension, path=f"{path}[{index}]")
        for index, dimension in enumerate(value)
    ]


def _source_tensor_manifest(
    value: Any,
    *,
    path: str,
) -> Mapping[str, Mapping[str, Any]]:
    manifest = _mapping(value, path=path)
    normalized: dict[str, Mapping[str, Any]] = {}
    for name, raw_entry in manifest.items():
        _strict_nonempty_string(name, path=f"{path}.<field>")
        entry = _exact_mapping(
            raw_entry,
            _SOURCE_TENSOR_DESCRIPTOR_FIELDS,
            path=f"{path}.{name}",
        )
        _strict_nonempty_string(entry["device"], path=f"{path}.{name}.device")
        _strict_nonempty_string(entry["dtype"], path=f"{path}.{name}.dtype")
        _shape(entry["shape"], path=f"{path}.{name}.shape")
        _strict_sha256(entry["sha256"], path=f"{path}.{name}.sha256")
        normalized[name] = entry
    return normalized


def _checkpoint_authority(
    raw: Any,
    *,
    generation: int,
    world: str,
    identity: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    checkpoint = _exact_mapping(raw, _CHECKPOINT_FIELDS, path="authority.checkpoint")
    _require_version(
        checkpoint["source_version"],
        CHECKPOINT_LAYER_SCHEMA_VERSION,
        path="authority.checkpoint.source_version",
    )
    _require_version(
        checkpoint["rng_source_version"],
        SOURCE_RNG_SCHEMA_VERSION,
        path="authority.checkpoint.rng_source_version",
    )
    projection_schema = _strict_nonempty_string(
        checkpoint["projection_schema_version"],
        path="authority.checkpoint.projection_schema_version",
    )
    projection_fields = _exact_mapping(
        checkpoint["projection_fields"],
        tuple(
            field
            for field in SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS
            if field != "tensors"
        ),
        path="authority.checkpoint.projection_fields",
    )
    if projection_fields["schema_version"] != projection_schema:
        _fail(
            "authority.checkpoint.projection_fields.schema_version",
            "contradicts projection_schema_version",
        )

    metadata = _mapping(
        projection_fields["metadata"],
        path="authority.checkpoint.projection_fields.metadata",
    )
    required_metadata = {
        "world",
        "alive_count",
        *SOURCE_AUTHORITY_CURSOR_METADATA_FIELDS,
        *SOURCE_RNG_REQUIRED_METADATA_FIELDS,
    }
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        _fail(
            "authority.checkpoint.projection_fields.metadata",
            f"missing required fields {missing_metadata}",
        )
    if type(metadata["world"]) is not str or metadata["world"] != world:
        _fail("authority.checkpoint.projection_fields.metadata.world", "contradicts world")
    projection_generation = _strict_int(
        metadata["generation"],
        path="authority.checkpoint.projection_fields.metadata.generation",
    )
    if projection_generation != generation + 1:
        _fail(
            "authority.checkpoint.projection_fields.metadata.generation",
            "must be the next generation",
        )
    tick = _strict_int(
        metadata["tick"], path="authority.checkpoint.projection_fields.metadata.tick"
    )
    if tick != 0:
        _fail("authority.checkpoint.projection_fields.metadata.tick", "must be zero")
    configured_steps = _strict_int(
        metadata["configured_steps_per_generation"],
        path=(
            "authority.checkpoint.projection_fields.metadata."
            "configured_steps_per_generation"
        ),
        minimum=1,
    )
    if tick > configured_steps:
        _fail("authority.checkpoint.projection_fields.metadata.tick", "exceeds configured steps")
    if metadata["agent_grid_runner_phase"] != "generation_boundary":
        _fail(
            "authority.checkpoint.projection_fields.metadata.agent_grid_runner_phase",
            "must be 'generation_boundary'",
        )
    expected_epoch = f"generation:{projection_generation}:tick:{tick}"
    if metadata["agent_grid_authority_epoch"] != expected_epoch:
        _fail(
            "authority.checkpoint.projection_fields.metadata.agent_grid_authority_epoch",
            f"must equal {expected_epoch!r}",
        )
    if type(metadata["native_runtime_env_policy_numpy_alias"]) is not bool:
        raise TypeError(
            "authority.checkpoint.projection_fields.metadata."
            "native_runtime_env_policy_numpy_alias must be a bool"
        )
    if projection_fields["last_world"] != world:
        _fail("authority.checkpoint.projection_fields.last_world", "contradicts world")
    event_cursor = _strict_int(
        projection_fields["event_cursor"],
        path="authority.checkpoint.projection_fields.event_cursor",
    )
    if event_cursor != lifecycle["authoritative_event_cursor"]:
        _fail(
            "authority.checkpoint.projection_fields.event_cursor",
            "contradicts lifecycle authority",
        )
    _strict_int(
        projection_fields["runtime_event_cursor"],
        path="authority.checkpoint.projection_fields.runtime_event_cursor",
    )

    tensor_manifest = _source_tensor_manifest(
        checkpoint["tensor_manifest"], path="authority.checkpoint.tensor_manifest"
    )
    original_manifest = _mapping(
        projection_fields["tensor_manifest"],
        path="authority.checkpoint.projection_fields.tensor_manifest",
    )
    expected_original_fields = set(tensor_manifest)
    if world == "patch":
        expected_original_fields -= set(SOURCE_PATCH_DERIVED_TENSOR_FIELDS)
    if set(original_manifest) != expected_original_fields:
        _fail(
            "authority.checkpoint.tensor_manifest",
            "fields contradict projection tensor_manifest",
        )
    for name in sorted(expected_original_fields):
        descriptor = tensor_manifest[name]
        original = _mapping(
            original_manifest[name],
            path=f"authority.checkpoint.projection_fields.tensor_manifest.{name}",
        )
        if "shape" not in original:
            _fail(
                f"authority.checkpoint.projection_fields.tensor_manifest.{name}",
                "is missing shape",
            )
        if _shape(
            original["shape"],
            path=(
                "authority.checkpoint.projection_fields.tensor_manifest."
                f"{name}.shape"
            ),
        ) != list(descriptor["shape"]):
            _fail(f"authority.checkpoint.tensor_manifest.{name}.shape", "contradicts projection")

    for required_tensor in (
        "agent_ids_by_alive_order",
        *SOURCE_RNG_TENSOR_FIELDS,
    ):
        if required_tensor not in tensor_manifest:
            _fail(
                "authority.checkpoint.tensor_manifest",
                f"is missing required descriptor {required_tensor!r}",
            )
    active_descriptor = tensor_manifest["agent_ids_by_alive_order"]
    if active_descriptor["dtype"] != "int64":
        _fail(
            "authority.checkpoint.tensor_manifest.agent_ids_by_alive_order.dtype",
            "must equal 'int64'",
        )
    if list(active_descriptor["shape"]) != [identity["active_agent_count"]]:
        _fail(
            "authority.checkpoint.tensor_manifest.agent_ids_by_alive_order.shape",
            "contradicts active_agent_count",
        )
    if active_descriptor["sha256"] != identity["tensor_sha256"]:
        _fail(
            "authority.checkpoint.tensor_manifest.agent_ids_by_alive_order.sha256",
            "contradicts identity authority",
        )
    alive_count = _strict_int(
        metadata["alive_count"],
        path="authority.checkpoint.projection_fields.metadata.alive_count",
    )
    if alive_count != identity["active_agent_count"]:
        _fail(
            "authority.checkpoint.projection_fields.metadata.alive_count",
            "contradicts identity authority",
        )

    next_generation_digest = checkpoint_source_projection_digest_from_tensor_manifest(
        projection_fields=projection_fields,
        tensor_manifest=tensor_manifest,
    )
    trajectory_digest = checkpoint_layer_digest_from_tensor_manifest(
        layer=DigestLayer.TRAJECTORY_STATE,
        projection_schema_version=projection_schema,
        world=world,
        generation=projection_generation,
        tick=tick,
        fields=checkpoint["trajectory_state_fields"],
    )
    storage_digest = checkpoint_layer_digest_from_tensor_manifest(
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
        projection_schema_version=projection_schema,
        world=world,
        generation=projection_generation,
        tick=tick,
        fields=checkpoint["canonical_checkpoint_storage_fields"],
    )
    rng_digest = checkpoint_rng_digest_from_tensor_manifest(
        metadata=metadata,
        tensor_manifest=tensor_manifest,
    )
    return trajectory_digest, storage_digest, next_generation_digest, rng_digest


def _identity_authority(raw: Any) -> tuple[Mapping[str, Any], str]:
    identity = _exact_mapping(raw, _IDENTITY_FIELDS, path="authority.identity")
    _require_version(
        identity["source_version"],
        ACTIVE_AGENT_IDS_SCHEMA_VERSION,
        path="authority.identity.source_version",
    )
    active_count = _strict_int(
        identity["active_agent_count"], path="authority.identity.active_agent_count"
    )
    tensor_sha256 = _strict_sha256(
        identity["tensor_sha256"], path="authority.identity.tensor_sha256"
    )
    return identity, active_agent_ids_digest_from_tensor_sha256(
        active_count, tensor_sha256
    )


def _policy_authority(raw: Any) -> str:
    policy = _exact_mapping(
        raw, _POLICY_FIELDS, path="authority.policy_mutation_tape"
    )
    _require_version(
        policy["source_version"],
        POLICY_MUTATION_TAPE_SCHEMA_VERSION,
        path="authority.policy_mutation_tape.source_version",
    )
    return policy_mutation_tape_digest_from_tensor_sha256(
        policy["cursor"], policy["consumed_shapes"], policy["tensor_sha256"]
    )


def _reporting_authority(
    raw: Any, *, generation: int, cell: int
) -> tuple[str, str]:
    reporting = _exact_mapping(raw, _REPORTING_FIELDS, path="authority.reporting")
    _require_version(
        reporting["schema_version"],
        REPORTING_AUTHORITY_SCHEMA_VERSION,
        path="authority.reporting.schema_version",
    )
    _require_version(
        reporting["retained_layout_version"],
        REPORTING_RETAINED_LAYOUT_VERSION,
        path="authority.reporting.retained_layout_version",
    )
    reporting_generation = _strict_int(
        reporting["generation"], path="authority.reporting.generation"
    )
    reporting_cell = _strict_int(reporting["cell"], path="authority.reporting.cell")
    if reporting_generation != generation or reporting_cell != cell:
        _fail("authority.reporting", "generation/cell contradict outer authority")
    _strict_int(reporting["start_id_ceiling"], path="authority.reporting.start_id_ceiling")
    behavior_digest = _strict_sha256(
        reporting["behavior_rows_digest_sha256"],
        path="authority.reporting.behavior_rows_digest_sha256",
    )
    lineage_digest = _strict_sha256(
        reporting["lineage_records_digest_sha256"],
        path="authority.reporting.lineage_records_digest_sha256",
    )
    if behavior_rows_digest_sha256(reporting["behavior_rows"]) != behavior_digest:
        _fail(
            "authority.reporting.behavior_rows_digest_sha256",
            "contradicts behavior rows",
        )
    if lineage_records_digest_sha256(
        generation, reporting["lineage_records"]
    ) != lineage_digest:
        _fail(
            "authority.reporting.lineage_records_digest_sha256",
            "contradicts lineage records",
        )
    return behavior_digest, lineage_digest


def _lifecycle_authority(
    raw: Any, *, generation: int, cell: int
) -> tuple[Mapping[str, Any], str]:
    lifecycle = _exact_mapping(raw, _LIFECYCLE_FIELDS, path="authority.lifecycle")
    _require_version(
        lifecycle["source_version"],
        TYPED_LIFECYCLE_SCHEMA_VERSION,
        path="authority.lifecycle.source_version",
    )
    lifecycle_generation = _strict_int(
        lifecycle["generation"], path="authority.lifecycle.generation"
    )
    lifecycle_cell = _strict_int(lifecycle["cell"], path="authority.lifecycle.cell")
    if lifecycle_generation != generation or lifecycle_cell != cell:
        _fail("authority.lifecycle", "generation/cell contradict outer authority")
    start = _strict_int(
        lifecycle["event_cursor_start"], path="authority.lifecycle.event_cursor_start"
    )
    pre_turnover = _strict_int(
        lifecycle["event_cursor_pre_turnover"],
        path="authority.lifecycle.event_cursor_pre_turnover",
    )
    end = _strict_int(
        lifecycle["event_cursor_end"], path="authority.lifecycle.event_cursor_end"
    )
    authoritative = _strict_int(
        lifecycle["authoritative_event_cursor"],
        path="authority.lifecycle.authoritative_event_cursor",
    )
    event_rows = _strict_int(
        lifecycle["event_row_count"], path="authority.lifecycle.event_row_count"
    )
    if not start <= pre_turnover <= end:
        _fail("authority.lifecycle", "event cursors are not monotonic")
    if authoritative != end or event_rows != end - start:
        _fail("authority.lifecycle", "event cursor/count authorities contradict")
    return lifecycle, _strict_sha256(
        lifecycle["active_prefix_digest_sha256"],
        path="authority.lifecycle.active_prefix_digest_sha256",
    )


def _terminal_authority(raw: Any, *, generation: int, cell: int) -> str | None:
    terminal = _exact_mapping(raw, _TERMINAL_FIELDS, path="authority.terminal")
    _require_version(
        terminal["source_version"],
        RECOVERABLE_ERROR_SCHEMA_VERSION,
        path="authority.terminal.source_version",
    )
    terminal_generation = _strict_int(
        terminal["generation"], path="authority.terminal.generation"
    )
    terminal_cell = _strict_int(terminal["cell"], path="authority.terminal.cell")
    if terminal_generation != generation or terminal_cell != cell:
        _fail("authority.terminal", "generation/cell contradict outer authority")
    code = terminal["recoverable_error_code"]
    details = terminal["recoverable_error_details"]
    if code is None and details is None:
        return None
    if code is None or details is None:
        _fail("authority.terminal", "recoverable error code/details must both be set")
    return recoverable_error_digest_sha256(code, details)


def finalize_compact_summary_retained_digests(
    authority: Mapping[str, Any],
) -> dict[str, str | None]:
    """Return exactly the ten frozen wire digests for one generation and cell."""

    _require_detached_cpu_metadata(authority)
    source = _exact_mapping(authority, _TOP_LEVEL_FIELDS, path="authority")
    _require_version(
        source["schema_version"],
        FINALIZER_SOURCE_SCHEMA_VERSION,
        path="authority.schema_version",
    )
    _require_version(
        source["wire_schema_version"],
        WIRE_SCHEMA_VERSION,
        path="authority.wire_schema_version",
    )
    generation = _strict_int(source["generation"], path="authority.generation")
    cell = _strict_int(source["cell"], path="authority.cell")
    world = source["world"]
    if type(world) is not str or world not in ("flat", "patch"):
        _fail("authority.world", "must be 'flat' or 'patch'")

    identity, active_ids_digest = _identity_authority(source["identity"])
    lifecycle, event_digest = _lifecycle_authority(
        source["lifecycle"], generation=generation, cell=cell
    )
    behavior_digest, lineage_digest = _reporting_authority(
        source["reporting"], generation=generation, cell=cell
    )
    policy_digest = _policy_authority(source["policy_mutation_tape"])
    terminal_digest = _terminal_authority(
        source["terminal"], generation=generation, cell=cell
    )
    trajectory, storage, next_generation, rng = _checkpoint_authority(
        source["checkpoint"],
        generation=generation,
        world=world,
        identity=identity,
        lifecycle=lifecycle,
    )

    result: dict[str, str | None] = {
        "events.active_prefix_digest_sha256": event_digest,
        "identity.active_agent_ids_digest_sha256": active_ids_digest,
        "rng.authority_digest_sha256": rng,
        "rng.policy_mutation_tape_digest_sha256": policy_digest,
        "terminal.recoverable_error_digest_sha256": terminal_digest,
        "state_digests.authoritative_state_digest_sha256": trajectory,
        "state_digests.canonical_checkpoint_storage_digest_sha256": storage,
        "state_digests.next_generation_start_state_digest_sha256": next_generation,
        "behavior_metrics.rows_digest_sha256": behavior_digest,
        "lineage_metrics.records_digest_sha256": lineage_digest,
    }
    if tuple(result) != DIGEST_FIELDS:
        raise RuntimeError("compact digest finalizer output registry drift")
    return result


__all__ = [
    "FINALIZER_SOURCE_SCHEMA_VERSION",
    "CompactSummaryRetainedDigestFinalizerError",
    "finalize_compact_summary_retained_digests",
]
