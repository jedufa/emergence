"""Authenticate retained Science Request V1 output for policy observation.

This module is a read-only projection boundary.  It authenticates the exact
terminal Science Request receipt, aggregate files, and lifecycle closeout, then
copies reducer-owned values into generic ``ObservationFrameV1`` payloads.  It
does not interpret Adaptive Illumination policy, call an execution front door,
or promote compact snapshots into continuation authority.

The retained campaign atlas does not expose the complete semantic world record
or full control history.  Current rows expose exact semantic resident capacity;
the versioned mapping below copies it to the generic ``capacity`` policy fact.
Historical rows remain readable but leave that fact absent, following the
policy's declared missing-fact rule.  No value is inferred from a phenotype,
active count, profile name, or ambient source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Final

from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
    ObservationFrameV1,
    StateClassV1,
    StateRefV1,
)
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.canonical import canonical_json_bytes, canonical_sha256
from emergence_contracts.ids import ArtifactId, ContentDigest, SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1
from emergence_evidence.science_campaign_atlas_v1 import (
    SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1,
    SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_CONTRACT_VERSION,
    COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
    COMPACT_WORLD_ATLAS_REDUCER_ID,
    compact_world_atlas_reducer_descriptor_v1,
)
from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    validate_artifact_disposition_manifest_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import (
    FullEvolutionCudaWorkhorseContractV2,
    WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
    validate_workhorse_indexed_continuation_result_v2,
    validate_workhorse_scientific_control_contract_v1,
    validate_workhorse_selected_world_observation_v1,
)

from .lifecycle import expected_child_lifecycle_profile_v1


SCIENCE_REQUEST_RUN_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_science_request_v1_run_receipt_v1"
)
AUTHENTICATED_OBSERVATION_ADAPTER_ID_V1: Final[str] = (
    "eiel_science_request_authenticated_observation_adapter_v1"
)
_RUN_RECEIPT_NAME: Final[str] = "science_campaign_run_receipt.json"
_RESULT_NAME: Final[str] = "campaign_atlas_result.json"
_VERIFICATION_NAME: Final[str] = "campaign_atlas_verification_receipt.json"
_MANIFEST_NAME: Final[str] = "artifact_disposition_manifest.json"
_STORAGE_RECEIPT_NAME: Final[str] = "storage_disposition_receipt.json"
_PASSPORT_NAME: Final[str] = "run_passport.json"
_MAX_CONTROL_JSON_BYTES: Final[int] = 64 * 1024 * 1024
_MAX_AGGREGATE_JSON_BYTES: Final[int] = 2 * 1024 * 1024 * 1024
_SHA256_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")
_POLICY_FACT_MAPPING_V1: Final[tuple[Mapping[str, object], ...]] = (
    {
        "source_field": "resident_capacity",
        "target_fact": "capacity",
        "value_contract": "positive_integer_exact_copy",
        "missing_source_rule": "omit_target_fact",
    },
)
_SELECTED_WORLD_HOST_CLOSEOUT_MODE_V1: Final[str] = (
    "AUTHENTICATED_SELECTED_WORLD_HOST_BOUNDARY_V1"
)
_HOST_OBSERVATION_ARTIFACT_KIND_V1: Final[str] = (
    "science.workhorse_selected_world_host_observation_v1"
)


class AuthenticatedObservationError(ValueError):
    """A retained output is absent, mutable, or identity-inconsistent."""


@dataclass(frozen=True, slots=True)
class AuthenticatedObservationProjectionV1:
    """Internal typed result; the durable public objects are its frames/refs."""

    request_sha256: ContentDigest
    run_receipt_sha256: ContentDigest
    run_receipt_file_sha256: ContentDigest
    aggregate_result_file_sha256: ContentDigest | None
    verification_receipt_file_sha256: ContentDigest | None
    host_observation_file_sha256s: tuple[ContentDigest, ...]
    lifecycle_manifest_sha256: ContentDigest
    frames: tuple[ObservationFrameV1, ...]
    state_refs: tuple[StateRefV1, ...]

    def __post_init__(self) -> None:
        if tuple(frame.source_state_ref for frame in self.frames) != self.state_refs:
            raise AuthenticatedObservationError(
                "projection frames and observation-only state references differ"
            )
        has_aggregate = self.aggregate_result_file_sha256 is not None
        if has_aggregate != (self.verification_receipt_file_sha256 is not None):
            raise AuthenticatedObservationError(
                "aggregate result and verification identities must be paired"
            )
        if has_aggregate == bool(self.host_observation_file_sha256s):
            raise AuthenticatedObservationError(
                "projection must bind exactly one aggregate or host-observation source"
            )

    def projection_sha256(self) -> ContentDigest:
        return canonical_sha256(
            {
                "adapter_id": AUTHENTICATED_OBSERVATION_ADAPTER_ID_V1,
                "request_sha256": self.request_sha256.to_dict(),
                "run_receipt_sha256": self.run_receipt_sha256.to_dict(),
                "run_receipt_file_sha256": self.run_receipt_file_sha256.to_dict(),
                "aggregate_result_file_sha256": (
                    None
                    if self.aggregate_result_file_sha256 is None
                    else self.aggregate_result_file_sha256.to_dict()
                ),
                "verification_receipt_file_sha256": (
                    None
                    if self.verification_receipt_file_sha256 is None
                    else self.verification_receipt_file_sha256.to_dict()
                ),
                "host_observation_file_sha256s": [
                    item.to_dict() for item in self.host_observation_file_sha256s
                ],
                "lifecycle_manifest_sha256": (
                    self.lifecycle_manifest_sha256.to_dict()
                ),
                "frame_sha256s": [
                    frame.observation_frame_sha256().to_dict()
                    for frame in self.frames
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class _LoadedJsonV1:
    path: Path
    value: dict[str, object]
    file_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _AuthenticatedSelectedWorldHostObservationV1:
    batch_index: int
    world_id: str
    file: _LoadedJsonV1
    observation: dict[str, object]
    continuation_result: dict[str, object]


def _sha256_hex(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise AuthenticatedObservationError(
            f"{field} is not a lowercase SHA-256 digest"
        )
    return value


def _digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise AuthenticatedObservationError(f"{field} is not a digest object")
    try:
        return ContentDigest(str(value["algorithm"]), str(value["hex"]))
    except (TypeError, ValueError) as error:
        raise AuthenticatedObservationError(f"{field} is invalid") from error


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise AuthenticatedObservationError(f"{field} must be an object")
    return dict(value)


def _sequence(value: object, *, field: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AuthenticatedObservationError(f"{field} must be an array")
    return list(value)


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
        before.st_mode,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
        after.st_mode,
    )


def _load_json_file_v1(
    path: Path,
    *,
    field: str,
    maximum_bytes: int,
) -> _LoadedJsonV1:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise AuthenticatedObservationError(f"{field} must not be a symbolic link")
    try:
        resolved = unresolved.resolve(strict=True)
        before = os.lstat(resolved)
    except OSError as error:
        raise AuthenticatedObservationError(f"{field} is unavailable: {unresolved}") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 2
        or before.st_size > maximum_bytes
    ):
        raise AuthenticatedObservationError(
            f"{field} is not a bounded single-link regular file"
        )
    try:
        encoded = resolved.read_bytes()
        after = os.lstat(resolved)
    except OSError as error:
        raise AuthenticatedObservationError(f"{field} could not be read") from error
    if not _same_file_state(before, after) or len(encoded) != before.st_size:
        raise AuthenticatedObservationError(f"{field} changed while being read")
    try:
        raw = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AuthenticatedObservationError(f"{field} is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise AuthenticatedObservationError(f"{field} must contain a JSON object")
    return _LoadedJsonV1(
        path=resolved,
        value=dict(raw),
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        size_bytes=len(encoded),
    )


def _runtime_and_receipt_path_v1(source: str | Path) -> tuple[Path, Path]:
    unresolved = Path(source).expanduser()
    if unresolved.is_symlink():
        raise AuthenticatedObservationError(
            "Science Request runtime root/receipt must not be a symbolic link"
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise AuthenticatedObservationError(
            f"Science Request runtime root/receipt is unavailable: {unresolved}"
        ) from error
    if resolved.is_dir():
        root = resolved
        receipt_path = root / _RUN_RECEIPT_NAME
    elif resolved.is_file() and resolved.name == _RUN_RECEIPT_NAME:
        root = resolved.parent
        receipt_path = resolved
    else:
        raise AuthenticatedObservationError(
            "source must be a runtime directory or science_campaign_run_receipt.json"
        )
    if root.is_symlink():
        raise AuthenticatedObservationError("runtime root must not be a symbolic link")
    return root, receipt_path


def _expected_child_path_v1(
    root: Path,
    raw: object,
    *,
    expected_name: str,
    field: str,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AuthenticatedObservationError(f"{field} path is invalid")
    expected = root / expected_name
    if expected.is_symlink():
        raise AuthenticatedObservationError(f"{field} must not be a symbolic link")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.resolve(strict=False) != expected.resolve(strict=False):
        raise AuthenticatedObservationError(
            f"{field} does not name the canonical retained runtime child"
        )
    return expected


def _verify_canonical_digest_object_v1(
    value: Mapping[str, object], *, digest_field: str, label: str
) -> ContentDigest:
    supplied = _digest(value.get(digest_field), field=f"{label}.{digest_field}")
    core = {key: item for key, item in value.items() if key != digest_field}
    if canonical_sha256(core) != supplied:
        raise AuthenticatedObservationError(f"{label} canonical digest differs")
    return supplied


def _verify_run_receipt_v1(
    receipt: _LoadedJsonV1, *, root: Path
) -> tuple[ContentDigest, ContentDigest]:
    value = receipt.value
    if value.get("schema_version") != SCIENCE_REQUEST_RUN_RECEIPT_SCHEMA_V1:
        raise AuthenticatedObservationError("Science Request run receipt schema differs")
    supplied = _verify_canonical_digest_object_v1(
        value, digest_field="receipt_sha256", label="Science Request run receipt"
    )
    request_sha256 = _digest(
        value.get("request_sha256"), field="Science Request run receipt.request_sha256"
    )
    atlas_closeout = (
        value.get("reduction_completed") is True
        and value.get("aggregation_completed") is True
        and value.get("aggregate") is not None
        and value.get("host_observation_closeout") is None
    )
    host_closeout = (
        value.get("reduction_completed") is False
        and value.get("aggregation_completed") is False
        and value.get("observation_closeout_completed") is True
        and value.get("aggregate") is None
        and isinstance(value.get("host_observation_closeout"), Mapping)
    )
    if (
        value.get("ok") is not True
        or value.get("status") != "COMPLETED"
        or value.get("result_class") != "SCIENCE_CAMPAIGN_COMPLETED"
        or atlas_closeout == host_closeout
        or value.get("closeout_completed") is not True
        or value.get("execution_started") is not True
        or value.get("prequal_started") is not False
        or value.get("qualification_started") is not False
        or value.get("qualification_authority_effect") is not False
        or Path(str(value.get("runtime_root"))).expanduser().resolve(strict=False)
        != root
    ):
        raise AuthenticatedObservationError(
            "Science Request run receipt is not a completed no-qualification occurrence"
        )
    return request_sha256, supplied


def _verify_aggregate_files_v1(
    receipt: Mapping[str, object],
    *,
    root: Path,
    request_sha256: ContentDigest,
) -> tuple[_LoadedJsonV1, _LoadedJsonV1, dict[str, object], dict[str, object]]:
    aggregate = _mapping(receipt.get("aggregate"), field="run receipt.aggregate")
    result_ref = _mapping(aggregate.get("result"), field="aggregate.result")
    verification_ref = _mapping(
        aggregate.get("verification_receipt"),
        field="aggregate.verification_receipt",
    )
    result_path = _expected_child_path_v1(
        root,
        result_ref.get("path"),
        expected_name=_RESULT_NAME,
        field="aggregate result",
    )
    verification_path = _expected_child_path_v1(
        root,
        verification_ref.get("path"),
        expected_name=_VERIFICATION_NAME,
        field="aggregate verification receipt",
    )
    result_file = _load_json_file_v1(
        result_path,
        field="campaign atlas result",
        maximum_bytes=_MAX_AGGREGATE_JSON_BYTES,
    )
    verification_file = _load_json_file_v1(
        verification_path,
        field="campaign atlas verification receipt",
        maximum_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    if (
        result_ref.get("file_sha256") != result_file.file_sha256
        or verification_ref.get("file_sha256") != verification_file.file_sha256
    ):
        raise AuthenticatedObservationError("campaign aggregate file SHA-256 differs")

    result = result_file.value
    verification = verification_file.value
    result_sha256 = _verify_canonical_digest_object_v1(
        result, digest_field="result_sha256", label="campaign atlas result"
    )
    verification_sha256 = _verify_canonical_digest_object_v1(
        verification,
        digest_field="receipt_sha256",
        label="campaign atlas verification receipt",
    )
    if (
        aggregate.get("result_sha256") != result_sha256.to_dict()
        or aggregate.get("verification_receipt_sha256")
        != verification_sha256.to_dict()
        or result.get("schema_version") != SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1
        or verification.get("schema_version")
        != SCIENCE_CAMPAIGN_ATLAS_VERIFICATION_RECEIPT_SCHEMA_V1
        or _digest(result.get("request_sha256"), field="result.request_sha256")
        != request_sha256
        or _digest(
            verification.get("request_sha256"),
            field="verification.request_sha256",
        )
        != request_sha256
        or verification.get("aggregate_result_sha256") != result_sha256.to_dict()
    ):
        raise AuthenticatedObservationError(
            "campaign aggregate internal identity differs from its run receipt"
        )

    descriptor = compact_world_atlas_reducer_descriptor_v1()
    descriptor_sha256 = descriptor.descriptor_sha256()
    common_identity_fields = (
        "bundle_content_sha256",
        "reducer_id",
        "reducer_contract_version",
        "reducer_descriptor_sha256",
        "output_contract_id",
        "logical_world_count",
        "physical_batch_count",
    )
    if any(result.get(field) != verification.get(field) for field in common_identity_fields):
        raise AuthenticatedObservationError("campaign result/verification identity differs")
    if (
        result.get("reducer_id") != COMPACT_WORLD_ATLAS_REDUCER_ID
        or result.get("reducer_contract_version")
        != str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION)
        or result.get("output_contract_id") != COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID
        or result.get("reducer_descriptor_sha256") != descriptor_sha256.to_dict()
        or verification.get("verification")
        != {
            "batch_result_digests_authenticated": True,
            "world_result_digests_authenticated": True,
            "complete_disjoint_world_coverage": True,
            "semantic_world_order_canonical": True,
        }
        or verification.get("trajectory_effects_performed") is not False
        or verification.get("authority_effect") != "none"
    ):
        raise AuthenticatedObservationError(
            "campaign aggregate lacks the expected authenticated pure reducer binding"
        )
    return result_file, verification_file, result, verification


def _verify_world_rows_v1(
    result: Mapping[str, object], verification: Mapping[str, object]
) -> list[dict[str, object]]:
    rows_raw = _sequence(result.get("world_results"), field="result.world_results")
    logical_world_count = result.get("logical_world_count")
    if (
        type(logical_world_count) is not int
        or logical_world_count < 1
        or len(rows_raw) != logical_world_count
    ):
        raise AuthenticatedObservationError("campaign logical world count differs")
    rows: list[dict[str, object]] = []
    world_ids: list[str] = []
    semantic_order: list[dict[str, object]] = []
    for index, raw in enumerate(rows_raw):
        row = _mapping(raw, field=f"result.world_results[{index}]")
        _verify_canonical_digest_object_v1(
            row,
            digest_field="result_sha256",
            label=f"world result {index}",
        )
        required = {
            "world_id",
            "science",
            "dimension_bindings",
            "semantic_world_sha256",
            "acquisition_sha256",
            "milestones",
            "integrity",
            "result_sha256",
        }
        if not required.issubset(row):
            raise AuthenticatedObservationError(
                f"world result {index} lacks authenticated projection fields"
            )
        world_id = row.get("world_id")
        if not isinstance(world_id, str) or not world_id or world_id in world_ids:
            raise AuthenticatedObservationError("campaign world identity is invalid")
        _digest(row.get("semantic_world_sha256"), field="semantic_world_sha256")
        _digest(row.get("acquisition_sha256"), field="acquisition_sha256")
        _mapping(row.get("science"), field="world result.science")
        _mapping(row.get("dimension_bindings"), field="world result.dimension_bindings")
        _mapping(row.get("integrity"), field="world result.integrity")
        _terminal_milestone_v1(row)
        world_ids.append(world_id)
        semantic_order.append(
            {
                "world_id": world_id,
                "semantic_world_sha256": row["semantic_world_sha256"],
                "acquisition_sha256": row["acquisition_sha256"],
            }
        )
        rows.append(row)
    if verification.get("semantic_world_order_sha256") != canonical_sha256(
        semantic_order
    ).to_dict():
        raise AuthenticatedObservationError("semantic world order digest differs")

    batch_rows = _sequence(
        verification.get("batch_results"), field="verification.batch_results"
    )
    physical_batch_count = result.get("physical_batch_count")
    if (
        type(physical_batch_count) is not int
        or physical_batch_count < 1
        or len(batch_rows) != physical_batch_count
    ):
        raise AuthenticatedObservationError("campaign physical batch count differs")
    batch_digests: list[object] = []
    owned_world_ids: list[str] = []
    for expected_index, raw in enumerate(batch_rows):
        batch = _mapping(raw, field=f"verification.batch_results[{expected_index}]")
        ids = _sequence(batch.get("world_ids"), field="batch.world_ids")
        if (
            batch.get("physical_batch_index") != expected_index
            or batch.get("logical_world_count") != len(ids)
            or any(not isinstance(item, str) or not item for item in ids)
        ):
            raise AuthenticatedObservationError("campaign batch ownership differs")
        _digest(batch.get("batch_result_sha256"), field="batch_result_sha256")
        _digest(
            batch.get("batch_bundle_content_sha256"),
            field="batch_bundle_content_sha256",
        )
        _digest(batch.get("capture_index_sha256"), field="capture_index_sha256")
        batch_digests.append(batch["batch_result_sha256"])
        owned_world_ids.extend(str(item) for item in ids)
    if (
        result.get("source_batch_result_sha256s") != batch_digests
        or len(owned_world_ids) != len(set(owned_world_ids))
        or set(owned_world_ids) != set(world_ids)
    ):
        raise AuthenticatedObservationError("campaign batch/world coverage differs")
    return rows


def _terminal_milestone_v1(row: Mapping[str, object]) -> dict[str, object]:
    milestones = _sequence(row.get("milestones"), field="world result.milestones")
    if not milestones:
        raise AuthenticatedObservationError("world result has no absolute-tick milestone")
    checked: list[dict[str, object]] = []
    ticks: list[int] = []
    for raw in milestones:
        milestone = _mapping(raw, field="world result.milestone")
        tick = milestone.get("absolute_tick")
        if type(tick) is not int or tick < 0:
            raise AuthenticatedObservationError("world milestone tick is invalid")
        ticks.append(tick)
        checked.append(milestone)
    if ticks != sorted(set(ticks)):
        raise AuthenticatedObservationError(
            "world milestones are not unique ascending absolute ticks"
        )
    return checked[-1]


def _verify_completed_batches_v1(
    receipt: Mapping[str, object],
    *,
    root: Path,
    expected_physical_batch_count: int,
) -> tuple[_LoadedJsonV1, ...]:
    raw_rows = _sequence(receipt.get("completed_batches"), field="completed_batches")
    if len(raw_rows) != expected_physical_batch_count:
        raise AuthenticatedObservationError("completed batch count differs")
    loaded: list[_LoadedJsonV1] = []
    for expected_index, raw in enumerate(raw_rows):
        row = _mapping(raw, field=f"completed_batches[{expected_index}]")
        expected_name = f"batch_result_{expected_index:04d}.json"
        path = _expected_child_path_v1(
            root,
            row.get("path"),
            expected_name=expected_name,
            field=f"completed batch {expected_index}",
        )
        artifact = _load_json_file_v1(
            path,
            field=f"completed batch {expected_index}",
            maximum_bytes=_MAX_AGGREGATE_JSON_BYTES,
        )
        world_ids = _sequence(
            row.get("materialized_world_ids"),
            field=f"completed_batches[{expected_index}].materialized_world_ids",
        )
        if (
            row.get("batch_index") != expected_index
            or row.get("file_sha256") != artifact.file_sha256
            or row.get("ok") is not True
            or str(row.get("status")).lower() != "completed"
            or not world_ids
            or any(not isinstance(item, str) or not item for item in world_ids)
        ):
            raise AuthenticatedObservationError(
                f"completed batch {expected_index} identity differs"
            )
        loaded.append(artifact)
    return tuple(loaded)


def _verify_selected_world_host_closeout_v1(
    receipt: Mapping[str, object],
    *,
    root: Path,
    completed_batch_files: Sequence[_LoadedJsonV1],
) -> tuple[_AuthenticatedSelectedWorldHostObservationV1, ...]:
    closeout = _mapping(
        receipt.get("host_observation_closeout"),
        field="run receipt.host_observation_closeout",
    )
    expected_closeout_fields = {
        "mode",
        "world_count",
        "physical_batch_count",
        "observations",
        "runtime_observation_mode",
        "host_adapter_observation_only",
        "atlas_generated",
        "causal_event_rows_available",
        "individual_lifetime_histories_available",
    }
    rows = _sequence(closeout.get("observations"), field="host observations")
    completed_rows = _sequence(
        receipt.get("completed_batches"), field="completed_batches"
    )
    if (
        set(closeout) != expected_closeout_fields
        or closeout.get("mode") != _SELECTED_WORLD_HOST_CLOSEOUT_MODE_V1
        or closeout.get("world_count") != 1
        or closeout.get("physical_batch_count") != 1
        or closeout.get("runtime_observation_mode") != "OFF"
        or closeout.get("host_adapter_observation_only") is not True
        or closeout.get("atlas_generated") is not False
        or closeout.get("causal_event_rows_available") is not False
        or closeout.get("individual_lifetime_histories_available") is not False
        or len(rows) != 1
        or len(completed_rows) != 1
        or len(completed_batch_files) != 1
    ):
        raise AuthenticatedObservationError(
            "selected-world host-observation closeout is incomplete or overclaims"
        )
    row = _mapping(rows[0], field="host observations[0]")
    completed = _mapping(completed_rows[0], field="completed_batches[0]")
    batch_file = completed_batch_files[0]
    batch_result = batch_file.value
    result_sha256 = batch_result.get("result_sha256")
    result_core = {
        key: item for key, item in batch_result.items() if key != "result_sha256"
    }
    binding = _mapping(
        row.get("host_boundary_observation"),
        field="host observation binding",
    )
    expected_binding_fields = {
        "observation_mode",
        "path",
        "file_sha256",
        "content_sha256",
        "file_bytes",
        "checkpoint_occurrence_sha256",
        "checkpoint_occurrence_projection_sha256",
        "terminally_reopened",
        "runtime_observation_mode",
        "host_adapter_observation_only",
    }
    front_door = batch_result.get("front_door_result")
    terminal = (
        front_door.get("workhorse_terminalized_indexed_continuation")
        if isinstance(front_door, Mapping)
        else None
    )
    world_id = row.get("world_id")
    materialized_world_ids = completed.get("materialized_world_ids")
    if (
        set(row)
        != {"physical_batch_index", "world_id", "host_boundary_observation"}
        or row.get("physical_batch_index") != 0
        or completed.get("batch_index") != 0
        or not isinstance(world_id, str)
        or not world_id
        or materialized_world_ids != [world_id]
        or type(result_sha256) is not str
        or result_sha256 != canonical_sha256(result_core).hex
        or completed.get("result_sha256") != result_sha256
        or batch_result.get("selected_world_host_observation") != binding
        or not isinstance(front_door, Mapping)
        or not isinstance(terminal, Mapping)
        or front_door.get("workhorse_terminalized_operation") != terminal
        or terminal.get("host_boundary_observation") != binding
        or set(binding) != expected_binding_fields
        or binding.get("observation_mode")
        != WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
        or binding.get("runtime_observation_mode") != "OFF"
        or binding.get("terminally_reopened") is not True
        or binding.get("host_adapter_observation_only") is not True
    ):
        raise AuthenticatedObservationError(
            "selected-world host observation differs across terminal bindings"
        )
    raw_attempt_root = completed.get("runtime_root")
    if not isinstance(raw_attempt_root, str) or not raw_attempt_root:
        raise AuthenticatedObservationError(
            "selected-world completed batch lacks its attempt root"
        )
    attempt_root_unresolved = Path(raw_attempt_root).expanduser()
    if attempt_root_unresolved.is_symlink():
        raise AuthenticatedObservationError(
            "selected-world attempt root must not be a symbolic link"
        )
    try:
        attempt_root = attempt_root_unresolved.resolve(strict=True)
        attempt_root.relative_to(root)
    except (OSError, ValueError) as error:
        raise AuthenticatedObservationError(
            "selected-world attempt root leaves its Science Request root"
        ) from error
    raw_observation_path = binding.get("path")
    if not isinstance(raw_observation_path, str) or not raw_observation_path:
        raise AuthenticatedObservationError(
            "selected-world host observation path is absent"
        )
    observation_file = _load_json_file_v1(
        Path(raw_observation_path),
        field="selected-world host observation",
        maximum_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    try:
        observation_file.path.relative_to(attempt_root)
        observation = validate_workhorse_selected_world_observation_v1(
            observation_file.path,
            expected_checkpoint_occurrence_sha256=str(
                binding["checkpoint_occurrence_sha256"]
            ),
            expected_checkpoint_occurrence_projection_sha256=str(
                binding["checkpoint_occurrence_projection_sha256"]
            ),
        )
        source = _mapping(
            observation.get("source_continuation"),
            field="selected-world observation source continuation",
        )
        continuation_result = validate_workhorse_indexed_continuation_result_v2(
            str(source.get("execution_result_path"))
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise AuthenticatedObservationError(
            f"selected-world host observation does not reopen exactly: {error}"
        ) from error
    integrity = observation.get("integrity")
    event_coverage = observation.get("event_observation_coverage")
    world_observations = observation.get("world_observations")
    if (
        observation != observation_file.value
        and {
            key: value
            for key, value in observation.items()
            if key
            not in {
                "observation_path",
                "observation_file_sha256",
                "observation_file_bytes",
            }
        }
        != observation_file.value
    ):
        raise AuthenticatedObservationError(
            "selected-world validator returned another observation body"
        )
    if (
        observation.get("cell_id") != world_id
        or observation.get("execution_authority_granted") is not False
        or observation.get("qualification_authority_granted") is not False
        or observation.get("named_phenotype_reward_authority_granted") is not False
        or not isinstance(integrity, Mapping)
        or integrity.get("world_count") != 1
        or integrity.get("dropped_record_count") != 0
        or integrity.get("runtime_observation_mode_remains_off") is not True
        or integrity.get("host_boundary_adapter_is_observation_only") is not True
        or not isinstance(event_coverage, Mapping)
        or event_coverage.get("event_kind_rows_available") is not False
        or event_coverage.get("birth_death_causal_rows_claimed") is not False
        or not isinstance(world_observations, list)
        or len(world_observations) != 1
        or binding.get("file_sha256") != observation_file.file_sha256
        or binding.get("file_sha256")
        != observation.get("observation_file_sha256")
        or binding.get("content_sha256")
        != observation.get("observation_content_sha256")
        or binding.get("file_bytes") != observation_file.size_bytes
        or binding.get("file_bytes") != observation.get("observation_file_bytes")
    ):
        raise AuthenticatedObservationError(
            "selected-world host observation identity or authority changed"
        )
    return (
        _AuthenticatedSelectedWorldHostObservationV1(
            batch_index=0,
            world_id=world_id,
            file=observation_file,
            observation=dict(observation),
            continuation_result=dict(continuation_result),
        ),
    )


def _verify_lifecycle_v1(
    receipt: Mapping[str, object],
    *,
    root: Path,
    retained_files: Sequence[_LoadedJsonV1],
    expected_lifecycle_profile: str,
) -> ContentDigest:
    lifecycle = _mapping(receipt.get("lifecycle"), field="run receipt.lifecycle")
    plan_sha256 = _sha256_hex(
        lifecycle.get("plan_sha256"), field="run receipt.lifecycle.plan_sha256"
    )
    manifest_path = _expected_child_path_v1(
        root,
        lifecycle.get("artifact_disposition_manifest_path"),
        expected_name=_MANIFEST_NAME,
        field="artifact disposition manifest",
    )
    storage_path = _expected_child_path_v1(
        root,
        lifecycle.get("storage_disposition_receipt_path"),
        expected_name=_STORAGE_RECEIPT_NAME,
        field="storage disposition receipt",
    )
    passport_path = _expected_child_path_v1(
        root,
        lifecycle.get("run_passport_path"),
        expected_name=_PASSPORT_NAME,
        field="run passport",
    )
    manifest_file = _load_json_file_v1(
        manifest_path,
        field="artifact disposition manifest",
        maximum_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    try:
        manifest = validate_artifact_disposition_manifest_v1(manifest_file.value)
    except ArtifactLifecycleError as error:
        raise AuthenticatedObservationError(
            f"artifact disposition manifest is invalid: {error}"
        ) from error
    storage_file = _load_json_file_v1(
        storage_path,
        field="storage disposition receipt",
        maximum_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    passport_file = _load_json_file_v1(
        passport_path,
        field="run passport",
        maximum_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    storage = storage_file.value
    passport = passport_file.value
    storage_sha256 = _sha256_hex(
        storage.get("receipt_sha256"), field="storage receipt.receipt_sha256"
    )
    passport_sha256 = _sha256_hex(
        passport.get("passport_sha256"), field="run passport.passport_sha256"
    )
    if hashlib.sha256(canonical_json_bytes({
        key: item for key, item in storage.items() if key != "receipt_sha256"
    })).hexdigest() != storage_sha256:
        raise AuthenticatedObservationError("storage disposition receipt digest differs")
    if hashlib.sha256(canonical_json_bytes({
        key: item for key, item in passport.items() if key != "passport_sha256"
    })).hexdigest() != passport_sha256:
        raise AuthenticatedObservationError("run passport digest differs")

    manifest_sha256 = _sha256_hex(
        manifest.get("manifest_sha256"), field="manifest.manifest_sha256"
    )
    lifecycle_sha256 = _sha256_hex(
        manifest.get("lifecycle_contract_sha256"),
        field="manifest.lifecycle_contract_sha256",
    )
    campaign_id = receipt.get("campaign_id")
    if (
        manifest.get("runtime_root") != str(root)
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("campaign_id") != campaign_id
        or manifest.get("run_id") != root.name
        or manifest.get("terminal_status") != "completed"
        or manifest.get("lifecycle_profile") != expected_lifecycle_profile
        or not isinstance(manifest.get("inventory"), Mapping)
        or manifest["inventory"].get("complete") is not True  # type: ignore[index]
        or not isinstance(manifest.get("summary"), Mapping)
        or manifest["summary"].get("unresolved_rx_bytes") != 0  # type: ignore[index]
        or storage.get("schema_version") != "eiel_storage_disposition_receipt_v1"
        or storage.get("manifest_sha256") != manifest_sha256
        or storage.get("plan_sha256") != plan_sha256
        or storage.get("campaign_id") != campaign_id
        or storage.get("run_id") != root.name
        or storage.get("lifecycle_contract_sha256") != lifecycle_sha256
        or passport.get("schema_version") != "eiel_run_passport_v1"
        or passport.get("plan_sha256") != plan_sha256
        or passport.get("campaign_id") != campaign_id
        or passport.get("run_id") != root.name
        or passport.get("terminal_status") != "completed"
        or passport.get("lifecycle_profile") != expected_lifecycle_profile
        or passport.get("lifecycle_contract_sha256") != lifecycle_sha256
    ):
        raise AuthenticatedObservationError(
            "Science Request terminal lifecycle closeout is identity-inconsistent"
        )
    _expected_child_path_v1(
        root,
        passport.get("disposition_manifest_path"),
        expected_name=_MANIFEST_NAME,
        field="run passport disposition manifest",
    )
    _expected_child_path_v1(
        root,
        passport.get("disposition_receipt_path"),
        expected_name=_STORAGE_RECEIPT_NAME,
        field="run passport disposition receipt",
    )

    raw_inventory = _sequence(manifest.get("artifacts"), field="manifest.artifacts")
    inventory: dict[str, dict[str, object]] = {}
    for raw in raw_inventory:
        row = _mapping(raw, field="manifest artifact")
        path = row.get("path")
        if not isinstance(path, str) or not path or path in inventory:
            raise AuthenticatedObservationError("lifecycle inventory path is invalid")
        inventory[path] = row
    for retained in retained_files:
        relative = str(retained.path.relative_to(root)).replace("\\", "/")
        row = inventory.get(relative)
        if (
            row is None
            or row.get("artifact_type") != "file"
            or row.get("sha256") != retained.file_sha256
            or row.get("bytes") != retained.size_bytes
            or row.get("retention_class") not in {"R3", "R4", "R5"}
            or row.get("release_eligible") is True
        ):
            raise AuthenticatedObservationError(
                f"retained lifecycle inventory does not bind {relative}"
            )
    return ContentDigest("sha256", manifest_sha256)


def _generic_policy_facts_v1(row: Mapping[str, object]) -> dict[str, object]:
    terminal = _terminal_milestone_v1(row)
    integrity = _mapping(row.get("integrity"), field="world result.integrity")
    facts: dict[str, object] = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "world_id",
            "science",
            "dimension_bindings",
            "semantic_world_sha256",
            "acquisition_sha256",
            "milestones",
            "integrity",
            "result_sha256",
        }
        and (value is None or type(value) in {bool, int, str})
    }
    active_count = terminal.get("active_agent_count")
    if type(active_count) is int and active_count >= 0:
        facts["active_count"] = active_count
    terminal_absolute_tick = terminal.get("absolute_tick")
    if type(terminal_absolute_tick) is int and terminal_absolute_tick >= 0:
        # Keep the terminal cursor explicit in the generic policy fact view.
        # Profile-defined rate descriptors consume this authenticated scalar;
        # omitting it would turn otherwise valid aggregate rows into an
        # ``INVALID_OBSERVATION`` solely because the source stores the cursor
        # inside its terminal milestone.
        facts["terminal_absolute_tick"] = terminal_absolute_tick
    for mapping in _POLICY_FACT_MAPPING_V1:
        source_field = str(mapping["source_field"])
        if source_field not in row:
            continue
        value = row[source_field]
        if type(value) is not int or value <= 0:
            raise AuthenticatedObservationError(
                f"world result.{source_field} violates its policy fact mapping"
            )
        facts[str(mapping["target_fact"])] = value
    facts.update(
        {
            "integrity_ok": integrity.get("dropped_record_count") == 0,
            "integrity": integrity,
            "terminal_milestone_scalars": {
                key: value
                for key, value in terminal.items()
                if value is None or type(value) in {bool, int, str}
            },
            "dimension_bindings": row["dimension_bindings"],
            "science": row["science"],
            "source_world_result_sha256": row["result_sha256"],
        }
    )
    return facts


def _project_frames_v1(
    rows: Sequence[Mapping[str, object]],
    *,
    manifest: ExpeditionManifestV1,
    request_sha256: ContentDigest,
    run_receipt_sha256: ContentDigest,
    result_file: _LoadedJsonV1,
) -> tuple[ObservationFrameV1, ...]:
    descriptor = compact_world_atlas_reducer_descriptor_v1()
    profile_descriptor = {
        "schema_version": "eiel_authenticated_science_request_observation_profile_v1",
        "adapter_id": AUTHENTICATED_OBSERVATION_ADAPTER_ID_V1,
        "observation_profile_id": manifest.observation_profile_id,
        "observation_descriptor_set_sha256": (
            manifest.observation_descriptor_set_sha256.to_dict()
        ),
        "source_reducer_descriptor_sha256": descriptor.descriptor_sha256().to_dict(),
        "projection": {
            "world_result": "opaque_authenticated_copy",
            "control_readback": "authenticated_dimension_bindings_only",
            "policy_facts": "opaque_scalars_plus_terminal_and_integrity",
            "policy_fact_mapping": [dict(row) for row in _POLICY_FACT_MAPPING_V1],
            "descriptor_interpretation": "none",
            "state_authority": "observation_only",
        },
    }
    observation_profile_sha256 = canonical_sha256(profile_descriptor)
    result_artifact_sha256 = ContentDigest("sha256", result_file.file_sha256)
    artifact_ref = RunArtifactRefV1(
        artifact_id=ArtifactId(f"science-atlas-{result_file.file_sha256}"),
        artifact_kind="science_campaign_atlas_result",
        artifact_schema_version=SchemaVersion(SCIENCE_CAMPAIGN_ATLAS_RESULT_SCHEMA_V1),
        locator=str(result_file.path),
        content_digest=result_artifact_sha256,
        size_bytes=result_file.size_bytes,
    )
    frames: list[ObservationFrameV1] = []
    for logical_world_id, row in enumerate(rows):
        world_id = str(row["world_id"])
        semantic_world_sha256 = _digest(
            row["semantic_world_sha256"], field="semantic_world_sha256"
        )
        terminal = _terminal_milestone_v1(row)
        next_absolute_tick = int(terminal["absolute_tick"])
        control_readback_value = {
            "coverage": "authenticated_dimension_bindings_only",
            "dimension_bindings": row["dimension_bindings"],
            "semantic_world_sha256": semantic_world_sha256.to_dict(),
            "branch_authority": False,
        }
        control_history_sha256 = canonical_sha256(
            {
                "schema_version": "eiel_science_request_control_projection_v1",
                "request_sha256": request_sha256.to_dict(),
                "world_id": world_id,
                "control_readback": control_readback_value,
            }
        )
        trajectory_sha256 = canonical_sha256(
            {
                "schema_version": "eiel_science_request_world_occurrence_v1",
                "run_receipt_sha256": run_receipt_sha256.to_dict(),
                "world_id": world_id,
                "semantic_world_sha256": semantic_world_sha256.to_dict(),
            }
        )
        trajectory_id = f"science-trajectory-{trajectory_sha256.hex}"
        state = StateRefV1(
            expedition_id=manifest.expedition_id,
            world_trajectory_id=trajectory_id,
            logical_world_id=logical_world_id,
            next_absolute_tick=next_absolute_tick,
            state_class=StateClassV1.OBSERVATION_ONLY,
            semantic_world_sha256=semantic_world_sha256,
            control_history_sha256=control_history_sha256,
            source_object_sha256=result_artifact_sha256,
            source_artifact_ref=artifact_ref,
            exact_state_sha256=None,
            checkpoint_occurrence_sha256=None,
            rng_state_sha256=None,
            event_prefix_sha256=None,
            parent_occurrence_sha256=None,
            replay_recipe_sha256=None,
        )
        frame = ObservationFrameV1(
            expedition_id=manifest.expedition_id,
            world_trajectory_id=trajectory_id,
            logical_world_id=logical_world_id,
            next_absolute_tick=next_absolute_tick,
            semantic_world_sha256=semantic_world_sha256,
            control_history_sha256=control_history_sha256,
            observation_profile_id=manifest.observation_profile_id,
            observation_profile_sha256=observation_profile_sha256,
            observation_descriptor_set_sha256=(
                manifest.observation_descriptor_set_sha256
            ),
            reducer_implementation_sha256=descriptor.implementation_sha256,
            observations=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_authenticated_science_world_summary_v1"),
                {
                    "world_id": world_id,
                    "world_result_sha256": row["result_sha256"],
                    "terminal_absolute_tick": next_absolute_tick,
                    "top_level_scalar_observations": {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "world_id",
                            "semantic_world_sha256",
                            "acquisition_sha256",
                            "result_sha256",
                        }
                        and (value is None or type(value) in {bool, int, str})
                    },
                    "terminal_scalar_observations": {
                        key: value
                        for key, value in terminal.items()
                        if value is None or type(value) in {bool, int, str}
                    },
                    "integrity": row["integrity"],
                    "exact_detail_source": {
                        "aggregate_file_sha256": result_artifact_sha256.to_dict(),
                        "world_result_sha256": row["result_sha256"],
                    },
                },
            ),
            control_readback=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_request_control_readback_projection_v1"),
                control_readback_value,
            ),
            policy_facing_facts=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_authenticated_science_policy_facts_v1"),
                _generic_policy_facts_v1(row),
            ),
            descriptor_values=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_uninterpreted_descriptor_values_v1"),
                {
                    "status": "NOT_INTERPRETED_BY_GENERIC_ADAPTER",
                    "descriptor_set_sha256": (
                        manifest.observation_descriptor_set_sha256.to_dict()
                    ),
                },
            ),
            calibration=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_manifest_bound_calibration_reference_v1"),
                {
                    "status": "OWNED_BY_FROZEN_POLICY_PROFILE",
                    "descriptor_set_sha256": (
                        manifest.observation_descriptor_set_sha256.to_dict()
                    ),
                    "adapter_applied_calibration": False,
                },
            ),
            source_state_ref=state,
            previous_frame_sha256=None,
        )
        frames.append(frame)
    return tuple(frames)


def _project_selected_world_host_frames_v1(
    rows: Sequence[_AuthenticatedSelectedWorldHostObservationV1],
    *,
    manifest: ExpeditionManifestV1,
    request_sha256: ContentDigest,
    run_receipt_sha256: ContentDigest,
) -> tuple[ObservationFrameV1, ...]:
    profile_descriptor = {
        "schema_version": "eiel_authenticated_selected_world_host_profile_v1",
        "adapter_id": AUTHENTICATED_OBSERVATION_ADAPTER_ID_V1,
        "observation_profile_id": manifest.observation_profile_id,
        "observation_descriptor_set_sha256": (
            manifest.observation_descriptor_set_sha256.to_dict()
        ),
        "source_observation_mode": WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
        "projection": {
            "facts": "authenticated_host_boundary_exact_copy_only",
            "descriptor_interpretation": "none",
            "atlas_generation": "none",
            "state_authority": "observation_only",
            "runtime_observation_mode": "OFF",
        },
    }
    profile_sha256 = canonical_sha256(profile_descriptor)
    frames: list[ObservationFrameV1] = []
    for row in rows:
        observation = row.observation
        continuation = _mapping(
            row.continuation_result.get("continuation_contract"),
            field="selected-world continuation contract",
        )
        logical = _mapping(
            continuation.get("science_continuation_identity"),
            field="selected-world science continuation identity",
        )
        physical = _mapping(
            continuation.get("physical_continuation_plan"),
            field="selected-world physical continuation plan",
        )
        try:
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                _mapping(
                    physical.get("target_contract"),
                    field="selected-world target contract",
                )
            )
        except (TypeError, ValueError) as error:
            raise AuthenticatedObservationError(
                f"selected-world target capacity contract is invalid: {error}"
            ) from error
        world_rows = _sequence(
            observation.get("world_observations"),
            field="selected-world world observations",
        )
        world = _mapping(world_rows[0], field="selected-world world observation")
        terminal = _mapping(
            observation.get("terminal_checkpoint"),
            field="selected-world terminal checkpoint",
        )
        control = _mapping(
            observation.get("control_readback"),
            field="selected-world control readback",
        )
        scientific_control = validate_workhorse_scientific_control_contract_v1(
            _mapping(
                continuation.get("scientific_control_contract"),
                field="selected-world scientific control contract",
            )
        )
        event_coverage = _mapping(
            observation.get("event_observation_coverage"),
            field="selected-world event coverage",
        )
        lineage = _mapping(
            world.get("active_lineage_summary"),
            field="selected-world active lineage summary",
        )
        genotype = _mapping(
            world.get("active_genotype_summary"),
            field="selected-world active genotype summary",
        )
        agent_aggregates = _mapping(
            world.get("active_agent_aggregates"),
            field="selected-world active agent aggregates",
        )
        counters = _mapping(
            world.get("cumulative_counters"),
            field="selected-world cumulative counters",
        )
        child_semantic_sha256 = _sha256_hex(
            logical.get("science_continuation_contract_sha256"),
            field="science_continuation_contract_sha256",
        )
        parent_control_history_sha256 = _sha256_hex(
            logical.get("source_control_history_sha256"),
            field="source_control_history_sha256",
        )
        control_application_sha256 = _sha256_hex(
            control.get("control_application_sha256"),
            field="control_application_sha256",
        )
        world_state_sha256 = _sha256_hex(
            world.get("canonical_world_state_sha256"),
            field="canonical_world_state_sha256",
        )
        world_state_set_sha256 = _sha256_hex(
            terminal.get("world_state_set_sha256"),
            field="world_state_set_sha256",
        )
        checkpoint_occurrence_sha256 = _sha256_hex(
            terminal.get("checkpoint_occurrence_sha256"),
            field="checkpoint_occurrence_sha256",
        )
        checkpoint_projection_sha256 = _sha256_hex(
            terminal.get("checkpoint_occurrence_projection_sha256"),
            field="checkpoint_occurrence_projection_sha256",
        )
        next_absolute_tick = world.get("next_absolute_tick")
        active_population = world.get("active_population")
        alive_count = world.get("alive_count")
        capacity = contract.shape.capacity
        if (
            contract.B != 1
            or type(next_absolute_tick) is not int
            or next_absolute_tick
            != contract.start_next_tick + contract.total_ticks
            or terminal.get("checkpoint_at_tick") != next_absolute_tick
            or logical.get("target_next_tick") != next_absolute_tick
            or type(active_population) is not int
            or not 0 <= active_population <= capacity
            or alive_count != active_population
            or type(counters.get("birth_count")) is not int
            or any(type(value) is not int or value < 0 for value in counters.values())
            or control.get("control_contract_sha256")
            != scientific_control.get("control_contract_sha256")
            or control.get("ordered_native_fields")
            != scientific_control["native_mapping"]["ordered_fields"]
            or control.get("requested_native_rows")
            != [scientific_control["native_mapping"]["ordered_values"]]
            or control.get("readback_native_rows")
            != [scientific_control["native_mapping"]["ordered_values"]]
        ):
            raise AuthenticatedObservationError(
                "selected-world host facts differ from their exact target contract"
            )
        source_file_sha256 = ContentDigest("sha256", row.file.file_sha256)
        artifact_ref = RunArtifactRefV1(
            artifact_id=ArtifactId(f"workhorse-host-{row.file.file_sha256}"),
            artifact_kind=_HOST_OBSERVATION_ARTIFACT_KIND_V1,
            artifact_schema_version=SchemaVersion(
                "eiel_workhorse_selected_world_host_observation_v1"
            ),
            locator=str(row.file.path),
            content_digest=source_file_sha256,
            size_bytes=row.file.size_bytes,
        )
        control_history_sha256 = canonical_sha256(
            {
                "schema_version": (
                    "eiel_selected_world_control_history_projection_v1"
                ),
                "parent_control_history_sha256": parent_control_history_sha256,
                "control_application_sha256": control_application_sha256,
                "applied_next_absolute_tick": control.get(
                    "applied_next_absolute_tick"
                ),
                "terminal_next_absolute_tick": next_absolute_tick,
            }
        )
        trajectory_sha256 = canonical_sha256(
            {
                "schema_version": "eiel_selected_world_observed_occurrence_v1",
                "run_receipt_sha256": run_receipt_sha256.to_dict(),
                "request_sha256": request_sha256.to_dict(),
                "world_id": row.world_id,
                "world_uid": world.get("world_uid"),
                "checkpoint_occurrence_sha256": checkpoint_occurrence_sha256,
                "canonical_world_state_sha256": world_state_sha256,
            }
        )
        unavailable_facts = {
            "event_kind_rows_available": False,
            "birth_death_causal_rows_available": False,
            "individual_lifetime_histories_available": False,
            "complete_per_agent_lifetime_rows_available": False,
            "reason": event_coverage.get("event_kind_rows_unavailable_reason"),
        }
        state_identity = {
            "canonical_world_state_sha256": world_state_sha256,
            "world_state_set_sha256": world_state_set_sha256,
            "checkpoint_occurrence_sha256": checkpoint_occurrence_sha256,
            "checkpoint_occurrence_projection_sha256": (
                checkpoint_projection_sha256
            ),
            "shape_id": contract.shape_id,
            "semantic_transition_sha256": contract.semantic_transition_sha256,
            "rng_key": world.get("rng_key"),
            "rng_epoch": world.get("rng_epoch"),
            "branch_authority_granted": False,
        }
        control_readback_value = {
            "coverage": "authenticated_workhorse_selected_world_readback",
            "controls": dict(scientific_control["controls"]),
            "control_contract_sha256": control.get("control_contract_sha256"),
            "control_application_sha256": control_application_sha256,
            "applied_next_absolute_tick": control.get(
                "applied_next_absolute_tick"
            ),
            "first_permitted_divergence_tick": control.get(
                "first_permitted_divergence_tick"
            ),
            "ordered_native_fields": control.get("ordered_native_fields"),
            "requested_native_rows": control.get("requested_native_rows"),
            "readback_native_rows": control.get("readback_native_rows"),
            "runtime_observation_mode": "OFF",
            "host_adapter_observation_only": True,
            "branch_authority": False,
        }
        observation_value = {
            "world_id": row.world_id,
            "world_uid": world.get("world_uid"),
            "terminal_absolute_tick": next_absolute_tick,
            "population": {
                "active": active_population,
                "alive": alive_count,
                "occupied_sites": world.get("occupied_site_count"),
                "capacity": capacity,
            },
            "cumulative_counters": counters,
            "site_field_totals": world.get("site_field_totals"),
            "current_state_agent_aggregates": agent_aggregates,
            "active_lineage_summary": lineage,
            "active_genotype_summary": genotype,
            "control_readback": control_readback_value,
            "state_identity": state_identity,
            "unavailable_facts": unavailable_facts,
            "exact_detail_source": {
                "host_observation_file_sha256": source_file_sha256.to_dict(),
                "host_observation_content_sha256": observation.get(
                    "observation_content_sha256"
                ),
            },
        }
        policy_facts = {
            "active_count": active_population,
            "capacity": capacity,
            "birth_count": counters.get("birth_count"),
            "death_count": counters.get("death_count"),
            "cumulative_counters": counters,
            "active_lineage_summary": lineage,
            "active_genotype_summary": genotype,
            "current_state_agent_aggregates": agent_aggregates,
            "terminal_absolute_tick": next_absolute_tick,
            "control_readback": control_readback_value,
            "state_identity": state_identity,
            "unavailable_facts": unavailable_facts,
            "integrity_ok": True,
            "source_host_observation_content_sha256": observation.get(
                "observation_content_sha256"
            ),
        }
        logical_world_id = world.get("world_uid")
        if type(logical_world_id) is not int or logical_world_id <= 0:
            raise AuthenticatedObservationError(
                "selected-world observation has an invalid logical world UID"
            )
        state = StateRefV1(
            expedition_id=manifest.expedition_id,
            world_trajectory_id=f"selected-world-{trajectory_sha256.hex}",
            logical_world_id=logical_world_id,
            next_absolute_tick=next_absolute_tick,
            state_class=StateClassV1.OBSERVATION_ONLY,
            semantic_world_sha256=ContentDigest(
                "sha256", child_semantic_sha256
            ),
            control_history_sha256=control_history_sha256,
            source_object_sha256=source_file_sha256,
            source_artifact_ref=artifact_ref,
            exact_state_sha256=None,
            checkpoint_occurrence_sha256=None,
            rng_state_sha256=None,
            event_prefix_sha256=None,
            parent_occurrence_sha256=None,
            replay_recipe_sha256=None,
        )
        frames.append(
            ObservationFrameV1(
                expedition_id=manifest.expedition_id,
                world_trajectory_id=state.world_trajectory_id,
                logical_world_id=logical_world_id,
                next_absolute_tick=next_absolute_tick,
                semantic_world_sha256=state.semantic_world_sha256,
                control_history_sha256=control_history_sha256,
                observation_profile_id=manifest.observation_profile_id,
                observation_profile_sha256=profile_sha256,
                observation_descriptor_set_sha256=(
                    manifest.observation_descriptor_set_sha256
                ),
                reducer_implementation_sha256=profile_sha256,
                observations=CanonicalPayloadV1.from_value(
                    SchemaVersion(
                        "eiel_authenticated_workhorse_selected_world_summary_v1"
                    ),
                    observation_value,
                ),
                control_readback=CanonicalPayloadV1.from_value(
                    SchemaVersion(
                        "eiel_selected_world_control_readback_projection_v1"
                    ),
                    control_readback_value,
                ),
                policy_facing_facts=CanonicalPayloadV1.from_value(
                    SchemaVersion(
                        "eiel_authenticated_selected_world_policy_facts_v1"
                    ),
                    policy_facts,
                ),
                descriptor_values=CanonicalPayloadV1.from_value(
                    SchemaVersion("eiel_uninterpreted_descriptor_values_v1"),
                    {
                        "status": "NOT_INTERPRETED_BY_GENERIC_ADAPTER",
                        "descriptor_set_sha256": (
                            manifest.observation_descriptor_set_sha256.to_dict()
                        ),
                        "missing_profile_facts": ["maximum_lineage_depth"],
                    },
                ),
                calibration=CanonicalPayloadV1.from_value(
                    SchemaVersion("eiel_manifest_bound_calibration_reference_v1"),
                    {
                        "status": "OWNED_BY_FROZEN_POLICY_PROFILE",
                        "descriptor_set_sha256": (
                            manifest.observation_descriptor_set_sha256.to_dict()
                        ),
                        "adapter_applied_calibration": False,
                    },
                ),
                source_state_ref=state,
                previous_frame_sha256=None,
            )
        )
    return tuple(frames)


def project_authenticated_science_request_observations_v1(
    source: str | Path,
    *,
    manifest: ExpeditionManifestV1 | Mapping[str, object],
) -> AuthenticatedObservationProjectionV1:
    """Authenticate one retained terminal Science Request root and project it.

    ``source`` is the exact runtime directory or its canonical terminal run
    receipt.  The function performs bounded reads of named files only.  It
    never scans a storage parent and never writes, launches, retains, releases,
    or grants branch/qualification authority.
    """

    checked_manifest = (
        manifest
        if isinstance(manifest, ExpeditionManifestV1)
        else ExpeditionManifestV1.from_dict(dict(manifest))
    )
    root, receipt_path = _runtime_and_receipt_path_v1(source)
    run_receipt_file = _load_json_file_v1(
        receipt_path,
        field="Science Request run receipt",
        maximum_bytes=_MAX_CONTROL_JSON_BYTES,
    )
    request_sha256, run_receipt_sha256 = _verify_run_receipt_v1(
        run_receipt_file, root=root
    )
    host_closeout = run_receipt_file.value.get("host_observation_closeout")
    if isinstance(host_closeout, Mapping):
        completed_batches = _verify_completed_batches_v1(
            run_receipt_file.value,
            root=root,
            expected_physical_batch_count=1,
        )
        host_rows = _verify_selected_world_host_closeout_v1(
            run_receipt_file.value,
            root=root,
            completed_batch_files=completed_batches,
        )
        lifecycle_manifest_sha256 = _verify_lifecycle_v1(
            run_receipt_file.value,
            root=root,
            retained_files=(
                run_receipt_file,
                *completed_batches,
                *(row.file for row in host_rows),
            ),
            expected_lifecycle_profile=expected_child_lifecycle_profile_v1(
                checked_manifest
            ),
        )
        frames = _project_selected_world_host_frames_v1(
            host_rows,
            manifest=checked_manifest,
            request_sha256=request_sha256,
            run_receipt_sha256=run_receipt_sha256,
        )
        return AuthenticatedObservationProjectionV1(
            request_sha256=request_sha256,
            run_receipt_sha256=run_receipt_sha256,
            run_receipt_file_sha256=ContentDigest(
                "sha256", run_receipt_file.file_sha256
            ),
            aggregate_result_file_sha256=None,
            verification_receipt_file_sha256=None,
            host_observation_file_sha256s=tuple(
                ContentDigest("sha256", row.file.file_sha256) for row in host_rows
            ),
            lifecycle_manifest_sha256=lifecycle_manifest_sha256,
            frames=frames,
            state_refs=tuple(frame.source_state_ref for frame in frames),
        )
    result_file, verification_file, result, verification = (
        _verify_aggregate_files_v1(
            run_receipt_file.value,
            root=root,
            request_sha256=request_sha256,
        )
    )
    world_rows = _verify_world_rows_v1(result, verification)
    completed_batches = _verify_completed_batches_v1(
        run_receipt_file.value,
        root=root,
        expected_physical_batch_count=int(result["physical_batch_count"]),
    )
    lifecycle_manifest_sha256 = _verify_lifecycle_v1(
        run_receipt_file.value,
        root=root,
        retained_files=(
            result_file,
            verification_file,
            run_receipt_file,
            *completed_batches,
        ),
        expected_lifecycle_profile=expected_child_lifecycle_profile_v1(
            checked_manifest
        ),
    )
    frames = _project_frames_v1(
        world_rows,
        manifest=checked_manifest,
        request_sha256=request_sha256,
        run_receipt_sha256=run_receipt_sha256,
        result_file=result_file,
    )
    return AuthenticatedObservationProjectionV1(
        request_sha256=request_sha256,
        run_receipt_sha256=run_receipt_sha256,
        run_receipt_file_sha256=ContentDigest(
            "sha256", run_receipt_file.file_sha256
        ),
        aggregate_result_file_sha256=ContentDigest(
            "sha256", result_file.file_sha256
        ),
        verification_receipt_file_sha256=ContentDigest(
            "sha256", verification_file.file_sha256
        ),
        host_observation_file_sha256s=(),
        lifecycle_manifest_sha256=lifecycle_manifest_sha256,
        frames=frames,
        state_refs=tuple(frame.source_state_ref for frame in frames),
    )


__all__ = (
    "AUTHENTICATED_OBSERVATION_ADAPTER_ID_V1",
    "AuthenticatedObservationError",
    "AuthenticatedObservationProjectionV1",
    "project_authenticated_science_request_observations_v1",
)
