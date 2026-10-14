"""Fail-closed selected-world branch preparation for Adaptive Expedition V1.

This module is deliberately an adapter, not an execution authority.  It joins
an exact :class:`StateRefV1` to canonical CRCA checkpoint occurrence material
and to the bounded retained Workhorse row consumed by indexed continuation.
The Science Request compiler/carrier can now lower that exact row as a fresh,
technical-only ExecutionRequestV2 child.  Scientific closeout remains
fail-closed until Workhorse publishes authenticated observation material.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    StateClassV1,
    StateRefV1,
)
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.checkpoint_occurrence import (
    ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION,
    CHECKPOINT_OCCURRENCE_SCHEMA_VERSION,
    WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION,
    AbsoluteTickCheckpointOccurrenceV1,
    AbsoluteTickCheckpointOccurrenceCommitV1,
    CheckpointOccurrenceInspectionStatusV1,
    CheckpointOccurrenceV1,
    WorkhorseSemanticCheckpointOccurrenceV2,
)
from emergence_contracts.ids import ArtifactId, ContentDigest, RunId, SchemaVersion
from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
    SET_MANIFEST_SCHEMA_VERSION,
    WORLD_SCHEMA_VERSION,
    load_selected_world_from_set_v1,
    restore_world_checkpoint_projection_v1,
)
from engine.campaigns.cuda_full_evolution_v2_workhorse_provider import (
    materialize_workhorse_exact_founder_plan_v1,
)
from engine.campaigns.science_cuda_v2_native_world_v1 import (
    ScienceCudaV2NativeWorldV1,
)
from engine.native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EventSinkError,
    validate_event_sink_prefix_reference_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse.indexed_checkpoint import (
    WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1,
    WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1,
    WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1,
    WORKHORSE_SELECTED_WORLD_STATE_FILE_V1,
    inspect_workhorse_selected_world_checkpoint_v1,
    inspect_workhorse_indexed_checkpoint_set_v1,
    materialize_workhorse_selected_world_checkpoint_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse.canonical_rows import (
    validate_workhorse_canonical_row_execution_result_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.continuation import (
    validate_workhorse_indexed_continuation_result_v2,
    validate_workhorse_terminalized_indexed_continuation_receipt_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    FullEvolutionCudaWorkhorseContractV2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.scientific_control import (
    validate_workhorse_active_scientific_control_state_v1,
    workhorse_active_scientific_control_state_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse.selected_world_observation import (
    validate_workhorse_selected_world_observation_v1,
)
from eiel_composition.checkpoint_occurrence import (
    ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME,
    ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME,
    inspect_absolute_tick_checkpoint_occurrence_v1,
)


WORLD_ARTIFACT_KIND = "checkpoint.acs_world_sharded_v1"
WORKHORSE_SELECTED_WORLD_ARTIFACT_KIND = (
    "checkpoint.workhorse_selected_world_v1"
)
WORKHORSE_OCCURRENCE_PROJECTION_ARTIFACT_KIND = (
    "checkpoint.workhorse_occurrence_projection_v2"
)
CONTROL_HISTORY_PREFIX_SCHEMA = (
    "eiel_adaptive_science_control_history_prefix_internal_v1"
)
CONTROL_SUFFIX_SCHEMA = "eiel_adaptive_science_control_suffix_internal_v1"
WORKHORSE_FOUNDER_RNG_PREFIX_SCHEMA = (
    "eiel_workhorse_founder_selected_rng_prefix_internal_v1"
)
WORKHORSE_FOUNDER_EVENT_PREFIX_SCHEMA = (
    "eiel_workhorse_founder_selected_event_prefix_internal_v1"
)
WORKHORSE_FOUNDER_CONTROL_PREFIX_SCHEMA = (
    "eiel_workhorse_founder_selected_control_prefix_internal_v1"
)
WORKHORSE_CHILD_RNG_PREFIX_SCHEMA = (
    "eiel_workhorse_child_selected_rng_prefix_internal_v1"
)
WORKHORSE_CHILD_EVENT_PREFIX_SCHEMA = (
    "eiel_workhorse_child_selected_event_prefix_internal_v1"
)
WORKHORSE_CHILD_CONTROL_PREFIX_SCHEMA = (
    "eiel_workhorse_child_selected_control_prefix_internal_v1"
)


class SelectedWorldBranchError(ValueError):
    """Selected-world state cannot support an exact branch preparation."""


class SelectedWorldBranchCarrierUnavailable(SelectedWorldBranchError):
    """The canonical Science Request front door cannot yet carry this branch."""


@dataclass(frozen=True, slots=True)
class ResolvedSelectedWorld:
    """Internal exact-state resolution; not a new durable/public contract."""

    state_ref: StateRefV1
    parent_run_id: RunId
    selected_world_id: str
    manifest_path: str
    checkpoint_set_id: str
    source_object_path: str
    physical_cell_index: int
    runtime_logical_world_uid: int
    next_absolute_tick: int
    source_object_sha256: ContentDigest
    exact_state_sha256: ContentDigest
    rng_state_sha256: ContentDigest
    event_prefix_sha256: ContentDigest
    checkpoint_occurrence_sha256: ContentDigest
    parent_occurrence_sha256: ContentDigest
    control_history_sha256: ContentDigest
    world_mechanics_sha256: ContentDigest
    active_scientific_control_state: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SelectedWorldBranchPreparation:
    """Internal child-prefix preparation; it grants no dispatch authority."""

    parent: ResolvedSelectedWorld
    child_run_id: RunId
    child_trajectory_id: str
    effective_next_absolute_tick: int
    first_permitted_divergence_tick: int
    control_suffix_sha256: ContentDigest
    retry_of_run_id: None = None


def _fail(message: str) -> None:
    raise SelectedWorldBranchError(message)


def _digest(value: str, *, field: str) -> ContentDigest:
    try:
        return ContentDigest("sha256", value)
    except (TypeError, ValueError) as exc:
        raise SelectedWorldBranchError(f"{field} is not a lowercase SHA-256") from exc


def _bounded_path(path: str | Path, *, root: Path, field: str) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        _fail(f"{field} must not be a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise SelectedWorldBranchError(f"{field} is unavailable") from exc
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{field} must be a regular file")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SelectedWorldBranchError(f"{field} escapes the parent run root") from exc
    return resolved


def _regular_path(path: str | Path, *, field: str) -> Path:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        _fail(f"{field} must not be a symbolic link")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise SelectedWorldBranchError(f"{field} is unavailable") from exc
    if resolved.is_symlink() or not resolved.is_file():
        _fail(f"{field} must be a regular file")
    return resolved


def _projection_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    projection = dict(value)
    expected_fields = {
        "schema_version",
        "claim_class",
        "science_authority_granted",
        "qualification_authority_granted",
        "time_authority",
        "checkpoint_at_tick",
        "checkpoint_manifest_path",
        "checkpoint_manifest_file_sha256",
        "checkpoint_bundle_body_sha256",
        "checkpoint_identity_sha256",
        "canonical_state_sha256",
        "canonical_parent_binding_sha256",
        "operation_contract_sha256",
        "absolute_tick_bundle_path",
        "absolute_tick_bundle_digest",
        "occurrence_root",
        "occurrence_digest",
        "occurrence_descriptor_path",
        "occurrence_descriptor_file_sha256",
        "occurrence_commit_path",
        "occurrence_commit_file_sha256",
        "parent_checkpoint_digest",
        "canonical_visible",
        "restore_admitted_exact",
        "projection_content_sha256",
    }
    core = {
        key: item
        for key, item in projection.items()
        if key != "projection_content_sha256"
    }
    if (
        set(projection) != expected_fields
        or projection.get("schema_version")
        not in {
            "eiel_workhorse_row_checkpoint_occurrence_projection_v2",
            "eiel_workhorse_indexed_continuation_checkpoint_occurrence_projection_v2",
        }
        or projection.get("claim_class") != "TECHNICAL_ONLY"
        or projection.get("science_authority_granted") is not False
        or projection.get("qualification_authority_granted") is not False
        or projection.get("time_authority") != "absolute_tick_v1"
        or projection.get("canonical_visible") is not True
        or projection.get("restore_admitted_exact") is not False
        or projection.get("projection_content_sha256")
        != canonical_sha256(core).hex
    ):
        _fail("Workhorse checkpoint occurrence projection is not exact")
    return projection


def _occurrence(
    value: Mapping[str, Any],
) -> CheckpointOccurrenceV1 | AbsoluteTickCheckpointOccurrenceV1 | WorkhorseSemanticCheckpointOccurrenceV2:
    schema = value.get("schema_version")
    try:
        if schema == str(CHECKPOINT_OCCURRENCE_SCHEMA_VERSION):
            return CheckpointOccurrenceV1.from_dict(dict(value))
        if schema == str(ABSOLUTE_TICK_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION):
            return AbsoluteTickCheckpointOccurrenceV1.from_dict(dict(value))
        if schema == str(WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_SCHEMA_VERSION):
            return WorkhorseSemanticCheckpointOccurrenceV2.from_dict(dict(value))
    except Exception as exc:
        raise SelectedWorldBranchError(
            "checkpoint occurrence is not a valid canonical CRCA occurrence"
        ) from exc
    _fail("checkpoint occurrence schema is not an exact restorable CRCA occurrence")


def _one_mapping(
    value: object,
    *,
    field: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 1
        or not isinstance(value[0], Mapping)
    ):
        _fail(f"{field} must contain exactly one selected-world row")
    return dict(value[0])


def _control_history_digest(
    value: Mapping[str, Any],
    *,
    state_ref: StateRefV1,
    selected_world_id: str,
    world_mechanics_sha256: str,
    source_configuration_sha256: str,
) -> ContentDigest:
    required = {
        "schema_version",
        "selected_world_id",
        "logical_world_id",
        "end_next_absolute_tick",
        "semantic_world_sha256",
        "world_mechanics_sha256",
        "source_configuration_sha256",
        "applied_control_prefix",
    }
    if set(value) != required:
        _fail("control-history prefix has unknown or missing fields")
    if value.get("schema_version") != CONTROL_HISTORY_PREFIX_SCHEMA:
        _fail("control-history prefix schema is unsupported")
    if (
        value.get("selected_world_id") != selected_world_id
        or value.get("logical_world_id") != state_ref.logical_world_id
        or value.get("end_next_absolute_tick") != state_ref.next_absolute_tick
        or value.get("semantic_world_sha256")
        != state_ref.semantic_world_sha256.to_dict()
        or value.get("world_mechanics_sha256")
        != _digest(world_mechanics_sha256, field="world mechanics").to_dict()
        or value.get("source_configuration_sha256")
        != _digest(
            source_configuration_sha256,
            field="source configuration",
        ).to_dict()
        or not isinstance(value.get("applied_control_prefix"), Mapping)
    ):
        _fail("control-history prefix differs from the selected exact state")
    digest = canonical_sha256(dict(value))
    if digest != state_ref.control_history_sha256:
        _fail("StateRef control-history digest differs from its exact prefix")
    return digest


def _inspect_selected_world_material_v1(
    *,
    state_ref: StateRefV1,
    checkpoint_set_manifest_path: str | Path,
    selected_world_id: str,
    checkpoint_occurrence: Mapping[str, Any],
    control_history_prefix: Mapping[str, Any],
    parent_run_root: str | Path,
) -> ResolvedSelectedWorld:
    """Inspect exact material without granting selected-world branch authority."""

    if not isinstance(state_ref, StateRefV1):
        _fail("selected parent must be a canonical StateRefV1")
    if state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT:
        _fail("selected parent is not a direct exact branchable state")
    if not isinstance(selected_world_id, str) or not selected_world_id:
        _fail("selected world ID is invalid")

    root = Path(parent_run_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        _fail("parent run root must be a stable directory")
    manifest_path = _bounded_path(
        checkpoint_set_manifest_path,
        root=root,
        field="checkpoint-set manifest",
    )
    if manifest_path.name != "checkpoint_set_manifest.json":
        _fail("selected source is not a complete world-sharded checkpoint set")

    occurrence = _occurrence(checkpoint_occurrence)
    occurrence_digest = occurrence.occurrence_digest()
    if occurrence_digest != state_ref.checkpoint_occurrence_sha256:
        _fail("StateRef checkpoint occurrence digest differs")
    if occurrence.absolute_next_tick != state_ref.next_absolute_tick:
        _fail("checkpoint occurrence tick differs from StateRef")
    if occurrence.parent_checkpoint_digest != state_ref.parent_occurrence_sha256:
        _fail("StateRef parent occurrence chain differs")

    loaded = load_selected_world_from_set_v1(
        manifest_path,
        world_id=selected_world_id,
    )
    manifest = loaded["set_manifest"]
    if manifest.get("schema_version") != SET_MANIFEST_SCHEMA_VERSION:
        _fail("selected source is not a complete world checkpoint set")
    if loaded.get("schema_version") != WORLD_SCHEMA_VERSION:
        _fail("selected source object is not an exact world checkpoint")
    if loaded.get("absolute_next_tick") != state_ref.next_absolute_tick:
        _fail("selected world tick differs from StateRef")
    object_path = _bounded_path(
        loaded["path"],
        root=root,
        field="selected world object",
    )

    row = next(
        (
            dict(item)
            for item in manifest["objects"]
            if isinstance(item, Mapping) and item.get("cell_id") == selected_world_id
        ),
        None,
    )
    if row is None:
        _fail("selected world is absent from the sealed set")
    artifact_ref = state_ref.source_artifact_ref
    if (
        artifact_ref.artifact_kind != WORLD_ARTIFACT_KIND
        or str(artifact_ref.artifact_schema_version) != WORLD_SCHEMA_VERSION
        or Path(artifact_ref.locator).expanduser().resolve() != object_path
        or artifact_ref.size_bytes != object_path.stat().st_size
    ):
        _fail("StateRef source artifact does not identify the selected world object")
    source_object_digest = _digest(
        str(row["artifact_sha256"]),
        field="selected world artifact",
    )
    exact_state_digest = _digest(
        str(row["world_state_sha256"]),
        field="selected exact state",
    )
    if (
        source_object_digest != state_ref.source_object_sha256
        or source_object_digest != artifact_ref.content_digest
        or exact_state_digest != state_ref.exact_state_sha256
    ):
        _fail("StateRef source/exact-state identity differs from the selected world")

    scalar = loaded.get("scalar_metadata")
    recipe = scalar.get("settlement_recipe") if isinstance(scalar, Mapping) else None
    checkpoint_inputs = (
        recipe.get("checkpoint_projection_inputs")
        if isinstance(recipe, Mapping)
        else None
    )
    runtime_uids = (
        checkpoint_inputs.get("tw_rngv2_logical_world_uids")
        if isinstance(checkpoint_inputs, Mapping)
        else None
    )
    if (
        not isinstance(runtime_uids, list)
        or len(runtime_uids) != 1
        or type(runtime_uids[0]) is not int
        or runtime_uids[0] != state_ref.logical_world_id
    ):
        _fail("selected checkpoint logical-world identity differs from StateRef")

    direct = recipe.get("direct_result_scalars") if isinstance(recipe, Mapping) else None
    event_reference = _one_mapping(
        direct.get("event_history_prefix_references_by_cell")
        if isinstance(direct, Mapping)
        else None,
        field="event-history prefix",
    )
    event_frontiers = (
        direct.get("event_history_frontier_by_cell")
        if isinstance(direct, Mapping)
        else None
    )
    if (
        not isinstance(event_frontiers, list)
        or len(event_frontiers) != 1
        or event_reference.get("row_count") != event_frontiers[0]
    ):
        _fail("event-history prefix/frontier binding differs")
    _bounded_path(event_reference.get("path", ""), root=root, field="event history")
    try:
        authenticated_event = validate_event_sink_prefix_reference_v1(event_reference)
    except EventSinkError as exc:
        raise SelectedWorldBranchError(
            "selected checkpoint event prefix is not authentic"
        ) from exc
    if authenticated_event.get("cell_index") != loaded["physical_cell_index"]:
        _fail("authenticated event prefix belongs to a different physical cell")
    event_digest = _digest(
        str(authenticated_event["prefix_sha256"]),
        field="event prefix",
    )
    if event_digest != state_ref.event_prefix_sha256:
        _fail("StateRef event prefix differs from the authenticated event trajectory")

    restored = restore_world_checkpoint_projection_v1(
        object_path,
        expected_artifact_sha256=source_object_digest.hex,
    )
    try:
        rng_authority = checkpoint_source_projection_authority_manifest(
            restored["projection"]
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            "restored selected world has no complete RNG authority"
        ) from exc
    rng_digest = _digest(
        str(rng_authority["source_rng_sha256"]),
        field="selected-world RNG state",
    )
    if rng_digest != state_ref.rng_state_sha256:
        _fail("StateRef RNG identity differs from the restored exact state")

    source_configuration_sha256 = (
        recipe.get("source_configuration_sha256")
        if isinstance(recipe, Mapping)
        else None
    )
    world_mechanics_sha256 = row.get("world_mechanics_sha256")
    if not isinstance(source_configuration_sha256, str) or not isinstance(
        world_mechanics_sha256, str
    ):
        _fail("selected world lacks configuration/mechanics identity")
    control_digest = _control_history_digest(
        control_history_prefix,
        state_ref=state_ref,
        selected_world_id=selected_world_id,
        world_mechanics_sha256=world_mechanics_sha256,
        source_configuration_sha256=source_configuration_sha256,
    )

    return ResolvedSelectedWorld(
        state_ref=state_ref,
        parent_run_id=occurrence.run_id,
        selected_world_id=selected_world_id,
        manifest_path=str(manifest_path),
        checkpoint_set_id=str(manifest["checkpoint_set_id"]),
        source_object_path=str(object_path),
        physical_cell_index=int(loaded["physical_cell_index"]),
        runtime_logical_world_uid=int(runtime_uids[0]),
        next_absolute_tick=state_ref.next_absolute_tick,
        source_object_sha256=source_object_digest,
        exact_state_sha256=exact_state_digest,
        rng_state_sha256=rng_digest,
        event_prefix_sha256=event_digest,
        checkpoint_occurrence_sha256=occurrence_digest,
        parent_occurrence_sha256=state_ref.parent_occurrence_sha256,
        control_history_sha256=control_digest,
        world_mechanics_sha256=_digest(
            world_mechanics_sha256,
            field="world mechanics",
        ),
    )


def resolve_selected_world_state_v1(
    *,
    state_ref: StateRefV1,
    checkpoint_set_manifest_path: str | Path,
    selected_world_id: str,
    checkpoint_occurrence: Mapping[str, Any],
    control_history_prefix: Mapping[str, Any],
    parent_run_root: str | Path,
) -> ResolvedSelectedWorld:
    """Fail closed after bounded inspection until producer authority is complete."""

    _inspect_selected_world_material_v1(
        state_ref=state_ref,
        checkpoint_set_manifest_path=checkpoint_set_manifest_path,
        selected_world_id=selected_world_id,
        checkpoint_occurrence=checkpoint_occurrence,
        control_history_prefix=control_history_prefix,
        parent_run_root=parent_run_root,
    )
    raise SelectedWorldBranchCarrierUnavailable(
        "selected-world material is exact and restorable, but branch authority is "
        "incomplete: no committed producer occurrence binds the checkpoint-set "
        "manifest and selected object, and no authenticated applied-control receipt "
        "chain is bound to that occurrence"
    )


def resolve_workhorse_selected_world_state_v1(
    *,
    state_ref: StateRefV1,
    selected_checkpoint_manifest_path: str | Path,
    checkpoint_occurrence_projection: Mapping[str, Any],
    parent_run_root: str | Path | None = None,
) -> ResolvedSelectedWorld:
    """Resolve one retained Workhorse row through its committed CRCA occurrence."""

    if not isinstance(state_ref, StateRefV1):
        _fail("selected parent must be a canonical StateRefV1")
    if state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT:
        _fail("selected parent is not a direct exact branchable state")
    root: Path | None = None
    if parent_run_root is not None:
        root = Path(parent_run_root).expanduser().resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            _fail("parent run root must be a stable directory")
    manifest_path = (
        _regular_path(
            selected_checkpoint_manifest_path,
            field="selected Workhorse checkpoint manifest",
        )
        if root is None
        else _bounded_path(
            selected_checkpoint_manifest_path,
            root=root,
            field="selected Workhorse checkpoint manifest",
        )
    )
    try:
        manifest, state_path = inspect_workhorse_selected_world_checkpoint_v1(
            manifest_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"selected Workhorse checkpoint cannot reopen exactly: {exc}"
        ) from exc
    selected = manifest["selected_world"]
    source = manifest["source_indexed_checkpoint"]
    authority = manifest["authority"]
    artifact_ref = state_ref.source_artifact_ref
    package_bytes = sum(
        path.stat().st_size
        for path in (
            manifest_path,
            manifest_path.parent / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1,
            manifest_path.parent / WORKHORSE_SELECTED_WORLD_STATE_FILE_V1,
        )
    )
    if (
        artifact_ref.artifact_kind != WORKHORSE_SELECTED_WORLD_ARTIFACT_KIND
        or str(artifact_ref.artifact_schema_version)
        != WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1
        or Path(artifact_ref.locator).expanduser().resolve() != manifest_path
        or artifact_ref.size_bytes
        not in {manifest_path.stat().st_size, package_bytes}
        or artifact_ref.content_digest.hex
        != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        or state_ref.source_object_sha256 != artifact_ref.content_digest
    ):
        _fail("StateRef does not identify the retained Workhorse parent object")
    if (
        selected["world_uid"] != state_ref.logical_world_id
        or selected["checkpoint_at_tick"] != state_ref.next_absolute_tick
        or selected["state_sha256"] != state_ref.exact_state_sha256.hex
        or authority["semantic_world_sha256"]
        != state_ref.semantic_world_sha256.hex
        or authority["control_history_sha256"]
        != state_ref.control_history_sha256.hex
        or authority["rng_state_sha256"] != state_ref.rng_state_sha256.hex
        or authority["event_prefix_sha256"] != state_ref.event_prefix_sha256.hex
        or authority["checkpoint_occurrence_sha256"]
        != state_ref.checkpoint_occurrence_sha256.hex
        or authority["parent_occurrence_sha256"]
        != state_ref.parent_occurrence_sha256.hex
    ):
        _fail("StateRef differs from the retained Workhorse parent identities")

    projection = _projection_v2(checkpoint_occurrence_projection)
    descriptor_path = (
        _regular_path(
            str(projection.get("occurrence_descriptor_path", "")),
            field="checkpoint occurrence descriptor",
        )
        if root is None
        else _bounded_path(
            str(projection.get("occurrence_descriptor_path", "")),
            root=root,
            field="checkpoint occurrence descriptor",
        )
    )
    commit_path = (
        _regular_path(
            str(projection.get("occurrence_commit_path", "")),
            field="checkpoint occurrence commit",
        )
        if root is None
        else _bounded_path(
            str(projection.get("occurrence_commit_path", "")),
            root=root,
            field="checkpoint occurrence commit",
        )
    )
    try:
        descriptor = AbsoluteTickCheckpointOccurrenceV1.from_dict(
            json.loads(descriptor_path.read_text(encoding="utf-8"))
        )
        commit = AbsoluteTickCheckpointOccurrenceCommitV1.from_dict(
            json.loads(commit_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            "selected Workhorse parent lacks a valid canonical occurrence"
        ) from exc
    occurrence_digest = descriptor.occurrence_digest()
    projection_digest = projection.get("occurrence_digest")
    retained_projection = authority["checkpoint_occurrence_projection"]
    if (
        projection.get("projection_content_sha256")
        != retained_projection["projection_content_sha256"]
        or projection.get("canonical_visible") is not True
        or projection.get("checkpoint_at_tick") != state_ref.next_absolute_tick
        or projection.get("checkpoint_identity_sha256")
        != source["source_checkpoint_identity_sha256"]
        or projection.get("occurrence_descriptor_file_sha256")
        != hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
        or projection.get("occurrence_commit_file_sha256")
        != hashlib.sha256(commit_path.read_bytes()).hexdigest()
        or projection_digest != occurrence_digest.to_dict()
        or descriptor.absolute_next_tick != state_ref.next_absolute_tick
        or occurrence_digest != state_ref.checkpoint_occurrence_sha256
        or descriptor.parent_checkpoint_digest
        != state_ref.parent_occurrence_sha256
        or commit.occurrence_digest != occurrence_digest
        or commit.checkpoint_bundle_digest != descriptor.checkpoint_bundle_digest
    ):
        _fail("selected Workhorse parent occurrence/commit identity differs")
    return ResolvedSelectedWorld(
        state_ref=state_ref,
        parent_run_id=descriptor.run_id,
        selected_world_id=str(selected["world_uid"]),
        manifest_path=str(manifest_path),
        checkpoint_set_id=str(manifest["manifest_body_sha256"]),
        source_object_path=str(state_path),
        physical_cell_index=int(source["source_physical_slot"]),
        runtime_logical_world_uid=int(selected["world_uid"]),
        next_absolute_tick=state_ref.next_absolute_tick,
        source_object_sha256=state_ref.source_object_sha256,
        exact_state_sha256=state_ref.exact_state_sha256,
        rng_state_sha256=state_ref.rng_state_sha256,
        event_prefix_sha256=state_ref.event_prefix_sha256,
        checkpoint_occurrence_sha256=occurrence_digest,
        parent_occurrence_sha256=state_ref.parent_occurrence_sha256,
        control_history_sha256=state_ref.control_history_sha256,
        world_mechanics_sha256=_digest(
            str(selected["semantic_transition_sha256"]),
            field="Workhorse semantic transition",
        ),
        active_scientific_control_state=dict(
            manifest["active_scientific_control_state"]
        ),
    )


def resolve_workhorse_selected_world_continuation_v1(
    *,
    state_ref: StateRefV1,
    authority_receipt_ref: RunArtifactRefV1,
) -> ResolvedSelectedWorld:
    """Authenticate the public continuation refs and reopen the retained row."""

    if not isinstance(state_ref, StateRefV1) or not isinstance(
        authority_receipt_ref, RunArtifactRefV1
    ):
        _fail("selected-world continuation requires canonical artifact references")
    manifest_path = _regular_path(
        state_ref.source_artifact_ref.locator,
        field="selected Workhorse checkpoint manifest",
    )
    try:
        manifest, _state_path = inspect_workhorse_selected_world_checkpoint_v1(
            manifest_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"selected Workhorse checkpoint cannot reopen exactly: {exc}"
        ) from exc
    authority_path = _regular_path(
        authority_receipt_ref.locator,
        field="selected Workhorse occurrence authority receipt",
    )
    expected_authority = manifest["authority"][
        "checkpoint_occurrence_projection"
    ]
    if (
        authority_receipt_ref.artifact_kind
        != WORKHORSE_OCCURRENCE_PROJECTION_ARTIFACT_KIND
        or authority_path
        != manifest_path.parent / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
        or authority_receipt_ref.size_bytes != authority_path.stat().st_size
        or authority_receipt_ref.content_digest.hex
        != hashlib.sha256(authority_path.read_bytes()).hexdigest()
        or authority_receipt_ref.content_digest.hex != expected_authority["sha256"]
        or authority_receipt_ref.size_bytes != expected_authority["bytes"]
    ):
        _fail("selected Workhorse occurrence authority reference differs")
    try:
        projection_raw = json.loads(authority_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SelectedWorldBranchError(
            "selected Workhorse occurrence authority receipt is invalid"
        ) from exc
    projection = _projection_v2(projection_raw)
    if str(authority_receipt_ref.artifact_schema_version) != projection[
        "schema_version"
    ]:
        _fail("selected Workhorse occurrence authority schema differs")
    return resolve_workhorse_selected_world_state_v1(
        state_ref=state_ref,
        selected_checkpoint_manifest_path=manifest_path,
        checkpoint_occurrence_projection=projection,
    )


def workhorse_selected_world_authority_ref_v1(
    state_ref: StateRefV1,
) -> RunArtifactRefV1:
    """Reconstruct and authenticate the neighboring occurrence authority ref.

    ``StateRefV1`` intentionally keeps the selected checkpoint object as its
    source artifact.  The Science Request continuation contract also needs the
    compact occurrence-projection reference stored beside that object.  This
    adapter derives no new authority: it reopens the retained manifest, binds
    the exact neighboring file, and resolves the pair before returning it.
    """

    if not isinstance(state_ref, StateRefV1):
        _fail("selected parent must be a canonical StateRefV1")
    manifest_path = _regular_path(
        state_ref.source_artifact_ref.locator,
        field="selected Workhorse checkpoint manifest",
    )
    try:
        manifest, _state_path = inspect_workhorse_selected_world_checkpoint_v1(
            manifest_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"selected Workhorse checkpoint cannot reopen exactly: {exc}"
        ) from exc
    authority_path = _regular_path(
        manifest_path.parent / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1,
        field="selected Workhorse occurrence authority receipt",
    )
    retained = manifest["authority"]["checkpoint_occurrence_projection"]
    try:
        projection = _projection_v2(
            json.loads(authority_path.read_text(encoding="utf-8"))
        )
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise SelectedWorldBranchError(
            "selected Workhorse occurrence authority receipt is invalid"
        ) from exc
    authority_digest = ContentDigest(
        "sha256", hashlib.sha256(authority_path.read_bytes()).hexdigest()
    )
    if (
        retained.get("sha256") != authority_digest.hex
        or retained.get("bytes") != authority_path.stat().st_size
    ):
        _fail("selected Workhorse occurrence authority reference differs")
    authority_ref = RunArtifactRefV1(
        artifact_id=ArtifactId(
            "workhorse-occurrence-projection-"
            + str(projection["projection_content_sha256"])[:24]
        ),
        artifact_kind=WORKHORSE_OCCURRENCE_PROJECTION_ARTIFACT_KIND,
        artifact_schema_version=SchemaVersion(str(projection["schema_version"])),
        locator=str(authority_path),
        content_digest=authority_digest,
        size_bytes=authority_path.stat().st_size,
    )
    resolve_workhorse_selected_world_continuation_v1(
        state_ref=state_ref,
        authority_receipt_ref=authority_ref,
    )
    return authority_ref


def _founder_json_object_v1(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{field} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectedWorldBranchError(f"{field} cannot be reopened") from exc
    if type(value) is not dict:
        _fail(f"{field} must be an object")
    return value


def _founder_file_sha256_v1(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _founder_occurrence_chain_v1(
    projection: Mapping[str, Any],
    *,
    genesis_tick: int,
    terminal_tick: int,
    run_root: Path,
) -> tuple[AbsoluteTickCheckpointOccurrenceV1, AbsoluteTickCheckpointOccurrenceV1]:
    terminal_occurrence_root = Path(str(projection["occurrence_root"]))
    terminal_bundle_root = Path(str(projection["absolute_tick_bundle_path"]))
    for path, field in (
        (terminal_occurrence_root, "terminal occurrence root"),
        (terminal_bundle_root, "terminal absolute checkpoint bundle"),
    ):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(run_root)
        except (OSError, ValueError) as exc:
            raise SelectedWorldBranchError(f"{field} escapes the founder run") from exc
        if resolved.is_symlink() or not resolved.is_dir():
            _fail(f"{field} must be a stable directory")
        if field.startswith("terminal occurrence"):
            terminal_occurrence_root = resolved
        else:
            terminal_bundle_root = resolved
    expected_terminal_name = f"checkpoint-tick-{terminal_tick:012d}"
    expected_genesis_name = f"checkpoint-tick-{genesis_tick:012d}"
    if (
        terminal_occurrence_root.name != expected_terminal_name
        or terminal_bundle_root.name != expected_terminal_name
    ):
        _fail("terminal Workhorse occurrence path differs from its absolute tick")
    genesis_occurrence_root = (
        terminal_occurrence_root.parent / expected_genesis_name
    )
    genesis_bundle_root = terminal_bundle_root.parent / expected_genesis_name
    inspections = (
        inspect_absolute_tick_checkpoint_occurrence_v1(
            genesis_bundle_root, genesis_occurrence_root
        ),
        inspect_absolute_tick_checkpoint_occurrence_v1(
            terminal_bundle_root, terminal_occurrence_root
        ),
    )
    if any(
        row.status is not CheckpointOccurrenceInspectionStatusV1.VISIBLE
        for row in inspections
    ):
        _fail("Workhorse founder occurrence chain is not canonically visible")

    def reopen(root: Path) -> tuple[
        AbsoluteTickCheckpointOccurrenceV1,
        AbsoluteTickCheckpointOccurrenceCommitV1,
    ]:
        descriptor_path = root / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
        commit_path = root / ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME
        try:
            descriptor = AbsoluteTickCheckpointOccurrenceV1.from_dict(
                _founder_json_object_v1(
                    descriptor_path, field="founder occurrence descriptor"
                )
            )
            commit = AbsoluteTickCheckpointOccurrenceCommitV1.from_dict(
                _founder_json_object_v1(
                    commit_path, field="founder occurrence commit"
                )
            )
        except (TypeError, ValueError) as exc:
            raise SelectedWorldBranchError(
                "Workhorse founder occurrence is not canonical"
            ) from exc
        if commit.occurrence_digest != descriptor.occurrence_digest():
            _fail("Workhorse founder occurrence commit binds another descriptor")
        return descriptor, commit

    genesis, _genesis_commit = reopen(genesis_occurrence_root)
    terminal, _terminal_commit = reopen(terminal_occurrence_root)
    terminal_descriptor_path = (
        terminal_occurrence_root / ABSOLUTE_TICK_OCCURRENCE_DESCRIPTOR_NAME
    )
    terminal_commit_path = (
        terminal_occurrence_root / ABSOLUTE_TICK_OCCURRENCE_COMMIT_NAME
    )
    projected_parent = projection.get("parent_checkpoint_digest")
    if (
        genesis.absolute_next_tick != genesis_tick
        or terminal.absolute_next_tick != terminal_tick
        or genesis.parent_checkpoint_digest is not None
        or terminal.parent_checkpoint_digest != genesis.occurrence_digest()
        or projected_parent != genesis.occurrence_digest().to_dict()
        or projection.get("occurrence_digest")
        != terminal.occurrence_digest().to_dict()
        or projection.get("absolute_tick_bundle_digest")
        != terminal.checkpoint_bundle_digest.to_dict()
        or projection.get("occurrence_descriptor_path")
        != str(terminal_descriptor_path)
        or projection.get("occurrence_commit_path") != str(terminal_commit_path)
        or projection.get("occurrence_descriptor_file_sha256")
        != _founder_file_sha256_v1(terminal_descriptor_path)
        or projection.get("occurrence_commit_file_sha256")
        != _founder_file_sha256_v1(terminal_commit_path)
        or genesis.run_id != terminal.run_id
        or genesis.attempt_id != terminal.attempt_id
        or genesis.semantic_spec_digest != terminal.semantic_spec_digest
        or genesis.implementation_digest != terminal.implementation_digest
    ):
        _fail("Workhorse founder genesis/terminal occurrence chain differs")
    return genesis, terminal


def materialize_workhorse_founder_exact_state_ref_v1(
    batch_manifest_path: str | Path,
    destination: str | Path,
    *,
    world_uid: int,
    expedition_id: str,
    world_trajectory_id: str,
) -> tuple[StateRefV1, RunArtifactRefV1, ResolvedSelectedWorld]:
    """Bootstrap one exact parent from an authenticated ACS2 founder campaign.

    The caller selects only an existing runtime UID and destination.  Semantic,
    state, RNG, event-prefix, control, Run/Attempt, and occurrence identities
    are all derived from reopened provider material; no digest input is
    accepted from the caller.
    """

    requested_manifest = Path(batch_manifest_path).expanduser()
    if requested_manifest.is_symlink():
        _fail("founder batch manifest path is invalid")
    manifest_path = requested_manifest.resolve(strict=True)
    if manifest_path.name != "manifest.json":
        _fail("founder batch manifest path is invalid")
    batch_root = manifest_path.parent
    if batch_root.parent.name != "batches":
        _fail("founder batch manifest is outside a canonical run")
    run_root = batch_root.parent.parent.resolve(strict=True)
    manifest = _founder_json_object_v1(
        manifest_path, field="founder batch manifest"
    )
    plan_path = _bounded_path(
        run_root / "execution_plan.json",
        root=run_root,
        field="founder execution plan",
    )
    plan = _founder_json_object_v1(plan_path, field="founder execution plan")
    plan_body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if (
        manifest.get("schema_version") != "eiel_campaign_batch_manifest_v1"
        or manifest.get("status") != "completed"
        or manifest.get("provider_id")
        != "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        or manifest.get("batch_id") != batch_root.name
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or plan.get("plan_sha256") != canonical_sha256(plan_body).hex
    ):
        _fail("founder batch manifest does not bind its completed Workhorse plan")
    try:
        provider_bound_plan = materialize_workhorse_exact_founder_plan_v1(plan)
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"founder plan lacks its producer exact-bootstrap identity: {exc}"
        ) from exc
    if provider_bound_plan != plan:
        _fail("founder plan was not producer-bound before execution")
    analysis = plan.get("analysis")
    extension = analysis.get("extension") if isinstance(analysis, Mapping) else None
    operation = (
        extension.get("contract") if isinstance(extension, Mapping) else None
    )
    cells = plan.get("cells")
    execution = plan.get("execution")
    lifecycle = plan.get("artifact_lifecycle")
    if (
        not isinstance(operation, Mapping)
        or operation.get("operation") != "EXECUTION_ROW"
        or not isinstance(cells, list)
        or len(cells) != 1
        or not isinstance(cells[0], Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(lifecycle, Mapping)
    ):
        _fail("founder plan is not one regular Workhorse Science Request row")
    try:
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            operation["contract"]
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SelectedWorldBranchError("founder Workhorse contract is invalid") from exc
    checkpoint_ticks = execution.get("checkpoint_ticks")
    target_tick = contract.start_next_tick + contract.total_ticks
    genesis_tick = contract.start_next_tick + contract.temporal_k
    roles = (
        lifecycle.get("checkpoints", {}).get("roles")
        if isinstance(lifecycle.get("checkpoints"), Mapping)
        else None
    )
    terminal_roles = (
        [
            row
            for row in roles
            if isinstance(row, Mapping)
            and row.get("checkpoint_tick") == target_tick
        ]
        if isinstance(roles, list)
        else []
    )
    if (
        contract.start_next_tick != 0
        or execution.get("acs_checkpoint_slots") != 2
        or checkpoint_ticks != [genesis_tick, target_tick]
        or len(terminal_roles) != 1
        or terminal_roles[0].get("role") != "terminal"
        or terminal_roles[0].get("retention_class") != "R4"
    ):
        _fail(
            "founder exact-state bootstrap requires absolute tick zero, ACS2, "
            "first-dispatch genesis, and terminal R4"
        )
    try:
        native_world = ScienceCudaV2NativeWorldV1.from_dict(
            dict(cells[0]["native_world_control"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectedWorldBranchError(
            "founder native-world control projection is invalid"
        ) from exc
    controls_by_id = {
        str(row["control_id"]): row
        for row in native_world.payload["control_bindings"]
        if row["control_id"]
        in {"environment.resource.multiplier", "mutation.rate"}
    }
    expected_controls = {
        "environment.resource.multiplier": 1.0,
        "mutation.rate": 0.125,
    }
    if set(controls_by_id) != set(expected_controls) or any(
        controls_by_id[control_id].get("resolved_value") != expected
        or controls_by_id[control_id].get("schedule") != {"kind": "static"}
        for control_id, expected in expected_controls.items()
    ):
        _fail(
            "founder exact-state bootstrap supports only explicit static native "
            "defaults resource=1.0 and mutation=0.125"
        )

    checkpoint = manifest.get("checkpoint")
    result_row = manifest.get("result")
    receipt = (
        checkpoint.get("acs_checkpoint_receipt")
        if isinstance(checkpoint, Mapping)
        else None
    )
    projection = (
        receipt.get("checkpoint_occurrence_projection")
        if isinstance(receipt, Mapping)
        else None
    )
    indexed = receipt.get("indexed_checkpoint") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(result_row, Mapping)
        or not isinstance(projection, Mapping)
        or not isinstance(indexed, Mapping)
        or receipt.get("checkpoint_at_tick") != target_tick
    ):
        _fail("founder terminal ACS2 publication is incomplete")
    projection = _projection_v2(projection)
    _genesis_occurrence, terminal_occurrence = _founder_occurrence_chain_v1(
        projection,
        genesis_tick=genesis_tick,
        terminal_tick=target_tick,
        run_root=run_root,
    )
    execution_result_path = _bounded_path(
        result_row.get("execution_result_path", ""),
        root=run_root,
        field="founder Workhorse execution result",
    )
    if (
        result_row.get("execution_result_file_sha256")
        != _founder_file_sha256_v1(execution_result_path)
    ):
        _fail("founder Workhorse execution result file identity differs")
    try:
        execution_result = validate_workhorse_canonical_row_execution_result_v2(
            execution_result_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"founder Workhorse execution result cannot reopen: {exc}"
        ) from exc
    indexed_manifest_path = _bounded_path(
        indexed.get("manifest_path", ""),
        root=run_root,
        field="founder indexed checkpoint",
    )
    try:
        indexed_manifest, _state_path = inspect_workhorse_indexed_checkpoint_set_v1(
            indexed_manifest_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"founder indexed checkpoint cannot reopen: {exc}"
        ) from exc
    matches = [
        row
        for row in indexed_manifest["world_index_by_uid"]
        if row["world_uid"] == world_uid
    ]
    event_receipts = execution_result.get("event_sink_receipts")
    checkpoint_binding = execution_result.get("checkpoint_binding")
    source_projection = indexed_manifest.get("source_physical_checkpoint")
    if (
        len(matches) != 1
        or not isinstance(event_receipts, list)
        or not event_receipts
        or not isinstance(checkpoint_binding, Mapping)
        or checkpoint_binding.get("checkpoint_ticks") != checkpoint_ticks
        or not isinstance(source_projection, Mapping)
        or source_projection.get("checkpoint_identity_sha256")
        != projection.get("checkpoint_identity_sha256")
        or source_projection.get("event_sink_receipt_path")
        != event_receipts[-1].get("receipt_path")
        or any(world_uid not in row.get("world_uids", ()) for row in event_receipts)
        or any(
            row.get("start_next_absolute_tick")
            != contract.start_next_tick + index * contract.temporal_k
            or row.get("end_next_absolute_tick")
            != contract.start_next_tick + (index + 1) * contract.temporal_k
            for index, row in enumerate(event_receipts)
        )
        or event_receipts[-1].get("end_next_absolute_tick") != target_tick
    ):
        _fail("founder indexed state, event prefix, or checkpoint schedule differs")
    terminal_bundle_root = Path(str(projection["absolute_tick_bundle_path"]))
    genesis_bundle_root = (
        terminal_bundle_root.parent / f"checkpoint-tick-{genesis_tick:012d}"
    )
    genesis_metadata = _founder_json_object_v1(
        genesis_bundle_root / "metadata.json",
        field="founder genesis absolute checkpoint metadata",
    )
    terminal_metadata = _founder_json_object_v1(
        terminal_bundle_root / "metadata.json",
        field="founder terminal absolute checkpoint metadata",
    )
    if (
        genesis_metadata.get("runtime_checkpoint", {}).get(
            "last_sink_receipt_sha256"
        )
        != event_receipts[0].get("receipt_body_sha256")
        or terminal_metadata.get("runtime_checkpoint", {}).get(
            "last_sink_receipt_sha256"
        )
        != event_receipts[-1].get("receipt_body_sha256")
    ):
        _fail("founder occurrence chain does not bind its complete event prefix")
    selected_row = matches[0]
    world_state_set_identity = indexed_manifest.get("world_state_set_identity")
    rng_epoch = (
        world_state_set_identity.get("rng_epoch")
        if isinstance(world_state_set_identity, Mapping)
        else None
    )
    if type(rng_epoch) is not int or rng_epoch < 0:
        _fail("founder indexed state lacks its authenticated RNG epoch")
    selected_rng_keys: list[int] = []
    for event_receipt in event_receipts:
        rng_identity = event_receipt.get("rng_identity")
        if not isinstance(rng_identity, Mapping):
            _fail("founder event receipt lacks its RNG identity")
        position = event_receipt["world_uids"].index(world_uid)
        selected_rng_keys.append(rng_identity["rng_keys"][position])
    if (
        len(set(selected_rng_keys)) != 1
        or selected_rng_keys[0] != selected_row["rng_key"]
    ):
        _fail("founder selected RNG key changed across its event prefix")
    semantic_world = cells[0].get("science_world_sha256")
    if (
        not isinstance(semantic_world, Mapping)
        or set(semantic_world) != {"algorithm", "hex"}
        or semantic_world.get("algorithm") != "sha256"
    ):
        _fail("founder semantic-world identity is absent")
    active_controls = workhorse_active_scientific_control_state_v1(
        captured_next_absolute_tick=target_tick,
        resource_multiplier=1.0,
        mutation_rate=0.125,
    )
    rng_state_sha256 = canonical_sha256(
        {
            "schema_version": WORKHORSE_FOUNDER_RNG_PREFIX_SCHEMA,
            "world_uid": world_uid,
            "rng_key": selected_row["rng_key"],
            "rng_epoch": rng_epoch,
            "checkpoint_at_tick": target_tick,
            "attempt_lineage_sha256": source_projection["attempt_lineage_sha256"],
            "ordered_rng_identity_sha256s": [
                row["rng_identity_sha256"] for row in event_receipts
            ],
        }
    )
    event_prefix_sha256 = canonical_sha256(
        {
            "schema_version": WORKHORSE_FOUNDER_EVENT_PREFIX_SCHEMA,
            "world_uid": world_uid,
            "end_next_absolute_tick": target_tick,
            "ordered_dispatches": [
                {
                    "start_next_absolute_tick": row["start_next_absolute_tick"],
                    "end_next_absolute_tick": row["end_next_absolute_tick"],
                    "receipt_body_sha256": row["receipt_body_sha256"],
                    "dispatch_identity_sha256": row["dispatch_identity_sha256"],
                }
                for row in event_receipts
            ],
        }
    )
    control_history_sha256 = canonical_sha256(
        {
            "schema_version": WORKHORSE_FOUNDER_CONTROL_PREFIX_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "semantic_world_sha256": dict(semantic_world),
            "native_world_control_sha256": native_world.payload["control_digest"],
            "active_scientific_control_state": active_controls,
            "genesis_occurrence_sha256": terminal_occurrence.parent_checkpoint_digest.to_dict(),
            "terminal_occurrence_sha256": terminal_occurrence.occurrence_digest().to_dict(),
        }
    )
    return materialize_workhorse_selected_world_state_ref_v1(
        indexed_manifest_path,
        destination,
        world_uid=world_uid,
        checkpoint_occurrence_projection=projection,
        expedition_id=expedition_id,
        world_trajectory_id=world_trajectory_id,
        semantic_world_sha256=str(semantic_world["hex"]),
        control_history_sha256=control_history_sha256.hex,
        rng_state_sha256=rng_state_sha256.hex,
        event_prefix_sha256=event_prefix_sha256.hex,
        active_scientific_control_state=active_controls,
    )


def materialize_workhorse_child_exact_state_ref_v1(
    host_observation_path: str | Path,
    destination: str | Path,
    *,
    parent_state_ref: StateRefV1,
    expedition_id: str,
    world_trajectory_id: str,
) -> tuple[StateRefV1, RunArtifactRefV1, ResolvedSelectedWorld]:
    """Promote one completed selected-world child terminal to an exact parent.

    Every new digest is derived from the validated parent, continuation result,
    terminalized receipt, committed occurrence projections, acknowledged event
    receipts, control readback, and terminal indexed bytes.  The caller cannot
    supply semantic, control, RNG, event, occurrence, Run, or Attempt identity.
    """

    if (
        not isinstance(parent_state_ref, StateRefV1)
        or parent_state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT
        or parent_state_ref.expedition_id != expedition_id
    ):
        _fail("child exact-state promotion requires its branchable parent")
    try:
        parent_authority = workhorse_selected_world_authority_ref_v1(
            parent_state_ref
        )
        resolve_workhorse_selected_world_continuation_v1(
            state_ref=parent_state_ref,
            authority_receipt_ref=parent_authority,
        )
        observation = validate_workhorse_selected_world_observation_v1(
            host_observation_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"child parent or host observation cannot reopen exactly: {exc}"
        ) from exc
    source = observation.get("source_continuation")
    terminal = observation.get("terminal_checkpoint")
    worlds = observation.get("world_observations")
    if (
        not isinstance(source, Mapping)
        or not isinstance(terminal, Mapping)
        or not isinstance(worlds, list)
        or len(worlds) != 1
        or not isinstance(worlds[0], Mapping)
    ):
        _fail("child host observation lacks one exact terminal world")
    result_path = _regular_path(
        str(source.get("execution_result_path", "")),
        field="child indexed continuation result",
    )
    try:
        result = validate_workhorse_indexed_continuation_result_v2(result_path)
        terminalized_path = (
            result_path.parent
            / "workhorse-terminalized-indexed-continuation-receipt.json"
        )
        terminalized = validate_workhorse_terminalized_indexed_continuation_receipt_v2(
            terminalized_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"child continuation terminal receipt cannot reopen: {exc}"
        ) from exc
    continuation = result.get("continuation_contract")
    logical = (
        continuation.get("science_continuation_identity")
        if isinstance(continuation, Mapping)
        else None
    )
    control_contract = (
        continuation.get("scientific_control_contract")
        if isinstance(continuation, Mapping)
        else None
    )
    control_application = result.get("control_application")
    if (
        not isinstance(logical, Mapping)
        or not isinstance(control_contract, Mapping)
        or not isinstance(control_application, Mapping)
        or logical.get("source_checkpoint_occurrence_sha256")
        != parent_state_ref.checkpoint_occurrence_sha256.hex
        or logical.get("source_parent_occurrence_sha256")
        != parent_state_ref.parent_occurrence_sha256.hex
        or logical.get("source_semantic_world_sha256")
        != parent_state_ref.semantic_world_sha256.hex
        or logical.get("source_control_history_sha256")
        != parent_state_ref.control_history_sha256.hex
        or logical.get("source_rng_state_sha256")
        != parent_state_ref.rng_state_sha256.hex
        or logical.get("source_event_prefix_sha256")
        != parent_state_ref.event_prefix_sha256.hex
        or logical.get("start_next_tick") != parent_state_ref.next_absolute_tick
        or logical.get("world_count") != 1
        or result.get("restored_world_uids")
        != [parent_state_ref.logical_world_id]
        or control_application.get("control_contract_sha256")
        != control_contract.get("control_contract_sha256")
    ):
        _fail("child continuation differs from its selected parent or controls")
    try:
        controls = dict(control_contract["controls"])
        target_tick = int(logical["target_next_tick"])
        active_controls = workhorse_active_scientific_control_state_v1(
            captured_next_absolute_tick=target_tick,
            resource_multiplier=controls["environment.resource.multiplier"],
            mutation_rate=controls["mutation.rate"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectedWorldBranchError(
            "child active scientific control readback is invalid"
        ) from exc

    provider_manifest_path = result_path.parent / "manifest.json"
    provider_manifest = _founder_json_object_v1(
        provider_manifest_path,
        field="child Workhorse provider manifest",
    )
    provider_result = provider_manifest.get("result")
    raw_receipts = (
        provider_result.get("indexed_checkpoint_receipts")
        if isinstance(provider_result, Mapping)
        else None
    )
    host_pointer = (
        provider_result.get("host_boundary_observation")
        if isinstance(provider_result, Mapping)
        else None
    )
    checkpoint_ticks = continuation.get("checkpoint_ticks")
    publications = terminalized.get("indexed_checkpoint_publications")
    if (
        provider_manifest.get("schema_version")
        != "eiel_campaign_batch_manifest_v1"
        or provider_manifest.get("status") != "completed"
        or provider_manifest.get("provider_id")
        != "eiel-full-evolution-cuda-workhorse-v2-provider-v1"
        or not isinstance(raw_receipts, list)
        or not isinstance(checkpoint_ticks, list)
        or not isinstance(publications, list)
        or len(raw_receipts) != len(checkpoint_ticks)
        or len(publications) != len(checkpoint_ticks)
        or not isinstance(host_pointer, Mapping)
        or Path(str(provider_result.get("execution_result_path", ""))).resolve()
        != result_path
        or provider_result.get("execution_result_file_sha256")
        != result.get("receipt_file_sha256")
        or provider_result.get("execution_result_content_sha256")
        != result.get("receipt_content_sha256")
        or source.get("execution_result_file_sha256")
        != result.get("receipt_file_sha256")
        or source.get("execution_result_content_sha256")
        != result.get("receipt_content_sha256")
        or Path(str(host_pointer.get("path", ""))).resolve()
        != Path(host_observation_path).expanduser().resolve()
        or host_pointer.get("file_sha256")
        != observation.get("observation_file_sha256")
        or host_pointer.get("content_sha256")
        != observation.get("observation_content_sha256")
    ):
        _fail("child provider publication chain is incomplete")

    expected_parent = parent_state_ref.checkpoint_occurrence_sha256.to_dict()
    terminal_projection: dict[str, Any] | None = None
    terminal_indexed_path: Path | None = None
    for raw, publication, checkpoint_tick in zip(
        raw_receipts, publications, checkpoint_ticks, strict=True
    ):
        if not isinstance(raw, Mapping) or not isinstance(publication, Mapping):
            _fail("child checkpoint publication row is malformed")
        projection_raw = raw.get("checkpoint_occurrence_projection")
        indexed_raw = raw.get("indexed_checkpoint")
        if not isinstance(projection_raw, Mapping) or not isinstance(indexed_raw, Mapping):
            _fail("child checkpoint publication lacks occurrence or indexed state")
        projection = _projection_v2(projection_raw)
        occurrence = projection.get("occurrence_digest")
        indexed_path = _regular_path(
            str(indexed_raw.get("manifest_path", "")),
            field="child indexed checkpoint manifest",
        )
        if (
            projection.get("checkpoint_at_tick") != checkpoint_tick
            or projection.get("parent_checkpoint_digest") != expected_parent
            or not isinstance(occurrence, Mapping)
            or publication.get("checkpoint_at_tick") != checkpoint_tick
            or publication.get("occurrence_digest") != occurrence.get("hex")
            or publication.get(
                "checkpoint_occurrence_projection_content_sha256"
            )
            != projection.get("projection_content_sha256")
            or publication.get("indexed_manifest_file_sha256")
            != _founder_file_sha256_v1(indexed_path)
        ):
            _fail("child occurrence/indexed publication chain differs")
        expected_parent = dict(occurrence)
        terminal_projection = projection
        terminal_indexed_path = indexed_path
    assert terminal_projection is not None and terminal_indexed_path is not None
    if (
        terminal_projection["checkpoint_at_tick"] != target_tick
        or terminal.get("checkpoint_at_tick") != target_tick
        or Path(str(terminal.get("indexed_manifest_path", ""))).resolve()
        != terminal_indexed_path
        or terminal.get("checkpoint_occurrence_sha256")
        != terminal_projection["occurrence_digest"]["hex"]
        or terminal.get("checkpoint_occurrence_projection_sha256")
        != terminal_projection["projection_content_sha256"]
    ):
        _fail("child host observation names another terminal checkpoint")
    try:
        indexed, _state_path = inspect_workhorse_indexed_checkpoint_set_v1(
            terminal_indexed_path
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"child terminal indexed state cannot reopen: {exc}"
        ) from exc
    selected_rows = [
        row
        for row in indexed["world_index_by_uid"]
        if row["world_uid"] == parent_state_ref.logical_world_id
    ]
    measurements = result.get("dispatch_measurements")
    if (
        len(selected_rows) != 1
        or not isinstance(measurements, list)
        or not measurements
        or any(not isinstance(row, Mapping) for row in measurements)
    ):
        _fail("child terminal selected row or event prefix is absent")
    ordered_acks = [dict(row["event_sink_receipt"]) for row in measurements]
    if (
        any(
            ack.get("world_uids") != [parent_state_ref.logical_world_id]
            for ack in ordered_acks
        )
        or ordered_acks[0].get("start_next_absolute_tick")
        != parent_state_ref.next_absolute_tick
        or ordered_acks[-1].get("end_next_absolute_tick") != target_tick
        or any(
            left.get("end_next_absolute_tick")
            != right.get("start_next_absolute_tick")
            for left, right in zip(ordered_acks, ordered_acks[1:])
        )
    ):
        _fail("child acknowledged event prefix is not contiguous")
    selected_row = selected_rows[0]
    semantic_world_sha256 = _digest(
        str(logical.get("science_continuation_contract_sha256")),
        field="child semantic world",
    )
    rng_state_sha256 = canonical_sha256(
        {
            "schema_version": WORKHORSE_CHILD_RNG_PREFIX_SCHEMA,
            "parent_rng_state_sha256": parent_state_ref.rng_state_sha256.to_dict(),
            "world_uid": parent_state_ref.logical_world_id,
            "rng_key": selected_row["rng_key"],
            "rng_epoch": indexed["world_state_set_identity"]["rng_epoch"],
            "checkpoint_at_tick": target_tick,
            "terminal_occurrence_sha256": terminal_projection[
                "occurrence_digest"
            ],
            "ordered_rng_identity_sha256s": [
                ack["rng_identity_sha256"] for ack in ordered_acks
            ],
        }
    )
    event_prefix_sha256 = canonical_sha256(
        {
            "schema_version": WORKHORSE_CHILD_EVENT_PREFIX_SCHEMA,
            "parent_event_prefix_sha256": parent_state_ref.event_prefix_sha256.to_dict(),
            "world_uid": parent_state_ref.logical_world_id,
            "start_next_absolute_tick": parent_state_ref.next_absolute_tick,
            "end_next_absolute_tick": target_tick,
            "ordered_dispatches": [
                {
                    "start_next_absolute_tick": ack[
                        "start_next_absolute_tick"
                    ],
                    "end_next_absolute_tick": ack["end_next_absolute_tick"],
                    "receipt_body_sha256": ack["receipt_body_sha256"],
                    "dispatch_identity_sha256": ack[
                        "dispatch_identity_sha256"
                    ],
                }
                for ack in ordered_acks
            ],
        }
    )
    control_history_sha256 = canonical_sha256(
        {
            "schema_version": WORKHORSE_CHILD_CONTROL_PREFIX_SCHEMA,
            "parent_control_history_sha256": (
                parent_state_ref.control_history_sha256.to_dict()
            ),
            "scientific_control_contract_sha256": control_contract[
                "control_contract_sha256"
            ],
            "control_application_sha256": control_application[
                "control_application_sha256"
            ],
            "active_scientific_control_state": active_controls,
            "source_occurrence_sha256": (
                parent_state_ref.checkpoint_occurrence_sha256.to_dict()
            ),
            "terminal_occurrence_sha256": terminal_projection[
                "occurrence_digest"
            ],
        }
    )
    return materialize_workhorse_selected_world_state_ref_v1(
        terminal_indexed_path,
        destination,
        world_uid=parent_state_ref.logical_world_id,
        checkpoint_occurrence_projection=terminal_projection,
        expedition_id=expedition_id,
        world_trajectory_id=world_trajectory_id,
        semantic_world_sha256=semantic_world_sha256.hex,
        control_history_sha256=control_history_sha256.hex,
        rng_state_sha256=rng_state_sha256.hex,
        event_prefix_sha256=event_prefix_sha256.hex,
        active_scientific_control_state=active_controls,
    )


def materialize_workhorse_selected_world_state_ref_v1(
    indexed_checkpoint_manifest_path: str | Path,
    destination: str | Path,
    *,
    world_uid: int,
    checkpoint_occurrence_projection: Mapping[str, Any],
    expedition_id: str,
    world_trajectory_id: str,
    semantic_world_sha256: str,
    control_history_sha256: str,
    rng_state_sha256: str,
    event_prefix_sha256: str,
    active_scientific_control_state: Mapping[str, Any],
) -> tuple[StateRefV1, RunArtifactRefV1, ResolvedSelectedWorld]:
    """Materialize one exact parent and return its existing public references."""

    projection = _projection_v2(checkpoint_occurrence_projection)
    occurrence = projection["occurrence_digest"]
    parent_occurrence = projection["parent_checkpoint_digest"]
    if not isinstance(parent_occurrence, Mapping):
        _fail("selected parent occurrence lacks a historical parent binding")
    try:
        active_controls = validate_workhorse_active_scientific_control_state_v1(
            active_scientific_control_state
        )
        semantic_digest = _digest(semantic_world_sha256, field="semantic world")
        control_digest = _digest(control_history_sha256, field="control history")
        rng_digest = _digest(rng_state_sha256, field="RNG state")
        event_digest = _digest(event_prefix_sha256, field="event prefix")
        output = Path(destination).expanduser().resolve()
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir():
                _fail("recovered selected Workhorse destination is invalid")
            manifest_path = output / WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
            manifest, _state_path = inspect_workhorse_selected_world_checkpoint_v1(
                manifest_path
            )
            authority_path = output / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
            retained_projection = _projection_v2(
                _founder_json_object_v1(
                    authority_path,
                    field="recovered selected Workhorse occurrence authority",
                )
            )
            source = manifest["source_indexed_checkpoint"]
            authority = manifest["authority"]
            if (
                not _equivalent_retained_projection_v1(retained_projection, projection)
                or manifest["selected_world"]["world_uid"] != world_uid
                or Path(str(source["manifest_path"])).resolve()
                != Path(indexed_checkpoint_manifest_path).expanduser().resolve()
                or authority["semantic_world_sha256"] != semantic_digest.hex
                or authority["control_history_sha256"] != control_digest.hex
                or authority["rng_state_sha256"] != rng_digest.hex
                or authority["event_prefix_sha256"] != event_digest.hex
                or manifest["active_scientific_control_state"] != active_controls
            ):
                _fail("recovered selected Workhorse materialization differs")
            selected = {
                "manifest_path": str(manifest_path.resolve()),
                "manifest_file_sha256": _founder_file_sha256_v1(manifest_path),
                "authority_receipt_path": str(authority_path.resolve()),
                "authority_receipt_file_sha256": _founder_file_sha256_v1(
                    authority_path
                ),
            }
        else:
            selected = _materialize_self_contained_selected_world_v1(
                indexed_checkpoint_manifest_path,
                output,
                world_uid=world_uid,
                checkpoint_occurrence_projection=projection,
                semantic_world_sha256=semantic_digest.hex,
                control_history_sha256=control_digest.hex,
                rng_state_sha256=rng_digest.hex,
                event_prefix_sha256=event_digest.hex,
                active_scientific_control_state=active_controls,
            )
        manifest, _state_path = inspect_workhorse_selected_world_checkpoint_v1(
            selected["manifest_path"]
        )
    except Exception as exc:
        raise SelectedWorldBranchError(
            f"selected Workhorse parent materialization failed: {exc}"
        ) from exc
    manifest_path = Path(selected["manifest_path"])
    authority_path = Path(selected["authority_receipt_path"])
    manifest_digest = _digest(
        str(selected["manifest_file_sha256"]),
        field="selected Workhorse manifest file",
    )
    authority_digest = _digest(
        str(selected["authority_receipt_file_sha256"]),
        field="selected Workhorse occurrence authority file",
    )
    source_ref = RunArtifactRefV1(
        artifact_id=ArtifactId(
            "workhorse-selected-world-" + manifest["manifest_body_sha256"][:24]
        ),
        artifact_kind=WORKHORSE_SELECTED_WORLD_ARTIFACT_KIND,
        artifact_schema_version=SchemaVersion(
            WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1
        ),
        locator=str(manifest_path),
        content_digest=manifest_digest,
        size_bytes=sum(
            path.stat().st_size
            for path in (
                manifest_path,
                authority_path,
                manifest_path.parent / WORKHORSE_SELECTED_WORLD_STATE_FILE_V1,
            )
        ),
    )
    effective_projection = _projection_v2(
        _founder_json_object_v1(
            authority_path, field="selected Workhorse occurrence authority"
        )
    )
    authority_ref = RunArtifactRefV1(
        artifact_id=ArtifactId(
            "workhorse-occurrence-projection-"
            + str(effective_projection["projection_content_sha256"])[:24]
        ),
        artifact_kind=WORKHORSE_OCCURRENCE_PROJECTION_ARTIFACT_KIND,
        artifact_schema_version=SchemaVersion(str(effective_projection["schema_version"])),
        locator=str(authority_path),
        content_digest=authority_digest,
        size_bytes=authority_path.stat().st_size,
    )
    state_ref = StateRefV1(
        expedition_id=expedition_id,
        world_trajectory_id=world_trajectory_id,
        logical_world_id=world_uid,
        next_absolute_tick=int(manifest["selected_world"]["checkpoint_at_tick"]),
        state_class=StateClassV1.BRANCHABLE_EXACT,
        semantic_world_sha256=_digest(
            semantic_world_sha256, field="semantic world"
        ),
        control_history_sha256=_digest(
            control_history_sha256, field="control history"
        ),
        source_object_sha256=manifest_digest,
        source_artifact_ref=source_ref,
        exact_state_sha256=_digest(
            str(manifest["selected_world"]["state_sha256"]),
            field="selected exact state",
        ),
        checkpoint_occurrence_sha256=_digest(
            str(occurrence["hex"]), field="checkpoint occurrence"
        ),
        rng_state_sha256=_digest(rng_state_sha256, field="RNG state"),
        event_prefix_sha256=_digest(event_prefix_sha256, field="event prefix"),
        parent_occurrence_sha256=_digest(
            str(parent_occurrence["hex"]), field="parent occurrence"
        ),
        replay_recipe_sha256=None,
    )
    resolved = resolve_workhorse_selected_world_continuation_v1(
        state_ref=state_ref,
        authority_receipt_ref=authority_ref,
    )
    return state_ref, authority_ref, resolved


_RELOCATED_PROJECTION_PATH_FIELDS_V1 = frozenset(
    {
        "occurrence_root",
        "occurrence_descriptor_path",
        "occurrence_commit_path",
        "projection_content_sha256",
    }
)


def _equivalent_retained_projection_v1(
    retained: Mapping[str, Any], source: Mapping[str, Any]
) -> bool:
    """Compare a relocated projection without treating package paths as science."""

    try:
        retained_projection = _projection_v2(retained)
        source_projection = _projection_v2(source)
    except SelectedWorldBranchError:
        return False
    keys = set(retained_projection) | set(source_projection)
    for key in keys - _RELOCATED_PROJECTION_PATH_FIELDS_V1:
        if retained_projection.get(key) != source_projection.get(key):
            return False
    return True


def _materialize_self_contained_selected_world_v1(
    indexed_checkpoint_manifest_path: str | Path,
    destination: Path,
    *,
    world_uid: int,
    checkpoint_occurrence_projection: Mapping[str, Any],
    semantic_world_sha256: str,
    control_history_sha256: str,
    rng_state_sha256: str,
    event_prefix_sha256: str,
    active_scientific_control_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize a selected row with package-local occurrence evidence.

    The Workhorse indexer intentionally preserves the source occurrence paths.
    An Expedition-retained parent must survive release of that raw child, so the
    adaptive adapter wraps the indexer's exact bytes in a second bounded package
    and rebases only the occurrence file paths.  Occurrence and state digests are
    unchanged; the relocated projection receives a new path-bound content hash.
    """

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_parent = Path(tempfile.mkdtemp(prefix=".expedition-selected.", dir=str(destination.parent)))
    indexed_output = scratch_parent / "indexed"
    try:
        generated = materialize_workhorse_selected_world_checkpoint_v1(
            indexed_checkpoint_manifest_path,
            indexed_output,
            world_uid=world_uid,
            checkpoint_occurrence_projection=checkpoint_occurrence_projection,
            semantic_world_sha256=semantic_world_sha256,
            control_history_sha256=control_history_sha256,
            rng_state_sha256=rng_state_sha256,
            event_prefix_sha256=event_prefix_sha256,
            active_scientific_control_state=active_scientific_control_state,
        )
        source_manifest = Path(generated["manifest_path"])
        source_authority = Path(generated["authority_receipt_path"])
        source_state = Path(generated["state_path"])
        generated_manifest = _founder_json_object_v1(
            source_manifest, field="generated selected Workhorse manifest"
        )
        source_projection = _projection_v2(
            _founder_json_object_v1(
                source_authority, field="generated selected Workhorse occurrence"
            )
        )
        descriptor = _regular_path(
            str(source_projection["occurrence_descriptor_path"]),
            field="source selected occurrence descriptor",
        )
        commit = _regular_path(
            str(source_projection["occurrence_commit_path"]),
            field="source selected occurrence commit",
        )
        source_files = (source_state, descriptor, commit)
        if any(path.is_symlink() or not path.is_file() for path in source_files):
            _fail("selected occurrence material is not stable")

        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".selected-self-contained.tmp",
                dir=str(destination.parent),
            )
        )
        try:
            occurrence_root = stage / "occurrence"
            occurrence_root.mkdir()
            copied_descriptor = occurrence_root / descriptor.name
            copied_commit = occurrence_root / commit.name
            shutil.copyfile(descriptor, copied_descriptor)
            shutil.copyfile(commit, copied_commit)
            copied_state = stage / WORKHORSE_SELECTED_WORLD_STATE_FILE_V1
            shutil.copyfile(source_state, copied_state)
            for path in (copied_descriptor, copied_commit, copied_state):
                path.chmod(0o444)
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())

            local_projection = dict(source_projection)
            final_occurrence_root = destination / "occurrence"
            local_projection.update(
                {
                    "occurrence_root": str(final_occurrence_root.resolve()),
                    "occurrence_descriptor_path": str(
                        (final_occurrence_root / descriptor.name).resolve()
                    ),
                    "occurrence_commit_path": str(
                        (final_occurrence_root / commit.name).resolve()
                    ),
                }
            )
            local_projection_core = {
                key: item
                for key, item in local_projection.items()
                if key != "projection_content_sha256"
            }
            local_projection["projection_content_sha256"] = canonical_sha256(
                local_projection_core
            ).hex
            authority_bytes = canonical_json_bytes(local_projection)
            authority_path = stage / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
            authority_path.write_bytes(authority_bytes)
            authority_path.chmod(0o444)
            with authority_path.open("rb") as handle:
                os.fsync(handle.fileno())

            manifest_body = {
                key: item
                for key, item in generated_manifest.items()
                if key != "manifest_body_sha256"
            }
            authority = dict(manifest_body["authority"])
            authority["checkpoint_occurrence_projection"] = {
                "file_name": WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1,
                "bytes": len(authority_bytes),
                "sha256": hashlib.sha256(authority_bytes).hexdigest(),
                "projection_content_sha256": local_projection[
                    "projection_content_sha256"
                ],
            }
            manifest_body["authority"] = authority
            relocated_manifest = {
                **manifest_body,
                "manifest_body_sha256": canonical_sha256(manifest_body).hex,
            }
            manifest_path = stage / WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
            manifest_path.write_bytes(canonical_json_bytes(relocated_manifest))
            manifest_path.chmod(0o444)
            with manifest_path.open("rb") as handle:
                os.fsync(handle.fileno())
            for path in (occurrence_root, stage):
                path.chmod(0o555)
            descriptor_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor_fd)
            finally:
                os.close(descriptor_fd)
            if destination.exists() or destination.is_symlink():
                _fail("selected Workhorse destination appeared during publication")
            os.rename(stage, destination)
            descriptor_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor_fd)
            finally:
                os.close(descriptor_fd)
        except BaseException:
            if stage.exists() and not stage.is_symlink():
                shutil.rmtree(stage)
            raise
        final_manifest = destination / WORKHORSE_SELECTED_WORLD_CHECKPOINT_FILE_V1
        final_authority = destination / WORKHORSE_SELECTED_WORLD_AUTHORITY_FILE_V1
        final_state = destination / WORKHORSE_SELECTED_WORLD_STATE_FILE_V1
        return {
            **relocated_manifest,
            "manifest_path": str(final_manifest.resolve()),
            "manifest_file_sha256": hashlib.sha256(final_manifest.read_bytes()).hexdigest(),
            "state_path": str(final_state.resolve()),
            "state_file_sha256": hashlib.sha256(final_state.read_bytes()).hexdigest(),
            "authority_receipt_path": str(final_authority.resolve()),
            "authority_receipt_file_sha256": hashlib.sha256(final_authority.read_bytes()).hexdigest(),
        }
    finally:
        if scratch_parent.exists() and not scratch_parent.is_symlink():
            for path in sorted(scratch_parent.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                try:
                    path.chmod(path.stat().st_mode | 0o700)
                except OSError:
                    pass
            try:
                scratch_parent.chmod(scratch_parent.stat().st_mode | 0o700)
            except OSError:
                pass
            shutil.rmtree(scratch_parent)


def _reject_generation_keys(value: object, *, path: str = "control_suffix") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(f"{path} contains a non-string key")
            if "generation" in key.casefold():
                _fail("control suffix contains generation-shaped authority")
            _reject_generation_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_generation_keys(child, path=f"{path}[{index}]")


def _reject_hidden_temporal_authority(value: object, *, path: str) -> None:
    forbidden_fragments = (
        "tick",
        "time",
        "generation",
        "schedule",
        "onset",
        "offset",
        "delay",
        "duration",
        "start",
        "end",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _fail(f"{path} contains a non-string key")
            if any(fragment in key.casefold() for fragment in forbidden_fragments):
                _fail("control payload contains hidden temporal authority")
            _reject_hidden_temporal_authority(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_hidden_temporal_authority(child, path=f"{path}[{index}]")


def _validate_control_suffix(
    value: Mapping[str, Any],
    *,
    parent_tick: int,
    effective_tick: int,
) -> ContentDigest:
    if set(value) != {
        "schema_version",
        "effective_next_absolute_tick",
        "controls",
    }:
        _fail("control suffix has unknown or missing fields")
    if (
        value.get("schema_version") != CONTROL_SUFFIX_SCHEMA
        or value.get("effective_next_absolute_tick") != effective_tick
    ):
        _fail("control suffix does not bind its declared effective tick")
    controls = value.get("controls")
    if not isinstance(controls, list) or not controls:
        _fail("branch-local control suffix must contain at least one control")
    for index, control in enumerate(controls):
        if not isinstance(control, Mapping):
            _fail(f"control suffix row {index} must be an object")
        if set(control) != {"control_id", "onset_tick", "payload"}:
            _fail("control suffix row has unknown or missing fields")
        control_id = control.get("control_id")
        if not isinstance(control_id, str) or not control_id:
            _fail(f"control suffix row {index} has an invalid control_id")
        onset_tick = control.get("onset_tick")
        if type(onset_tick) is not int:
            _fail(f"control suffix row {index} lacks an exact onset_tick")
        if onset_tick != effective_tick:
            _fail("changed control onset must equal the declared branch tick")
        payload = control.get("payload")
        if not isinstance(payload, Mapping):
            _fail(f"control suffix row {index} payload must be an object")
        _reject_hidden_temporal_authority(payload, path=f"controls[{index}].payload")
    _reject_generation_keys(value)
    observed_ticks: list[int] = []

    def collect_ticks(node: object) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if key.endswith("_tick"):
                    if type(child) is not int:
                        _fail("control suffix tick coordinates must be exact integers")
                    observed_ticks.append(child)
                collect_ticks(child)
        elif isinstance(node, list):
            for child in node:
                collect_ticks(child)

    collect_ticks(controls)
    if any(tick < effective_tick for tick in observed_ticks):
        _fail("changed controls would diverge before the declared branch tick")
    if effective_tick != parent_tick:
        _fail("branch effective tick must equal the selected parent occurrence")
    return canonical_sha256(dict(value))


def validate_control_suffix_v1(
    value: Mapping[str, Any],
    *,
    parent_next_absolute_tick: int,
    effective_next_absolute_tick: int,
) -> ContentDigest:
    """Validate the strict non-scientific timing envelope for changed controls."""

    return _validate_control_suffix(
        value,
        parent_tick=parent_next_absolute_tick,
        effective_tick=effective_next_absolute_tick,
    )


def prepare_selected_world_branch_v1(
    resolved: ResolvedSelectedWorld,
    *,
    child_run_id: RunId,
    child_trajectory_id: str,
    effective_next_absolute_tick: int,
    control_suffix: Mapping[str, Any],
    retry_of_run_id: RunId | None = None,
) -> SelectedWorldBranchPreparation:
    """Bind a new child identity and suffix without dispatching or repacking it."""

    if not isinstance(resolved, ResolvedSelectedWorld):
        _fail("branch preparation requires a resolved selected world")
    if not isinstance(child_run_id, RunId):
        _fail("child run identity must be a canonical RunId")
    if child_run_id == resolved.parent_run_id:
        _fail("a changed branch must be a new child run, not its parent")
    if retry_of_run_id is not None:
        _fail("a changed branch is never a retry")
    if (
        not isinstance(child_trajectory_id, str)
        or not child_trajectory_id
        or child_trajectory_id == resolved.state_ref.world_trajectory_id
    ):
        _fail("changed branch requires a distinct child trajectory identity")
    if type(effective_next_absolute_tick) is not int:
        _fail("branch effective tick must be an exact integer")
    suffix_digest = _validate_control_suffix(
        control_suffix,
        parent_tick=resolved.next_absolute_tick,
        effective_tick=effective_next_absolute_tick,
    )
    if (
        resolved.state_ref.source_artifact_ref.artifact_kind
        != WORKHORSE_SELECTED_WORLD_ARTIFACT_KIND
    ):
        raise SelectedWorldBranchCarrierUnavailable(
            "child preparation requires a retained Workhorse parent bound to a "
            "committed producer occurrence and authenticated control/event prefix"
        )
    return SelectedWorldBranchPreparation(
        parent=resolved,
        child_run_id=child_run_id,
        child_trajectory_id=child_trajectory_id,
        effective_next_absolute_tick=effective_next_absolute_tick,
        first_permitted_divergence_tick=effective_next_absolute_tick,
        control_suffix_sha256=suffix_digest,
        retry_of_run_id=None,
    )


def assert_canonical_selected_world_branch_execution_supported_v1() -> None:
    """Assert the installed OFF-runtime host observation seam is present."""

    try:
        from engine.perf.cuda_full_evolution_v2.workhorse.selected_world_observation import (
            WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
            seal_workhorse_selected_world_observation_v1,
            validate_workhorse_selected_world_observation_v1,
        )
    except ImportError as exc:
        raise SelectedWorldBranchCarrierUnavailable(
            "selected-world Workhorse host observation adapter is unavailable"
        ) from exc
    if (
        WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
        != "WORKHORSE_SELECTED_WORLD_HOST_BOUNDARY_COMPACT_V1"
        or not callable(seal_workhorse_selected_world_observation_v1)
        or not callable(validate_workhorse_selected_world_observation_v1)
    ):
        raise SelectedWorldBranchCarrierUnavailable(
            "selected-world Workhorse host observation adapter is incompatible"
        )


__all__ = (
    "CONTROL_HISTORY_PREFIX_SCHEMA",
    "CONTROL_SUFFIX_SCHEMA",
    "ResolvedSelectedWorld",
    "SelectedWorldBranchCarrierUnavailable",
    "SelectedWorldBranchError",
    "SelectedWorldBranchPreparation",
    "WORLD_ARTIFACT_KIND",
    "WORKHORSE_OCCURRENCE_PROJECTION_ARTIFACT_KIND",
    "WORKHORSE_SELECTED_WORLD_ARTIFACT_KIND",
    "assert_canonical_selected_world_branch_execution_supported_v1",
    "materialize_workhorse_child_exact_state_ref_v1",
    "materialize_workhorse_founder_exact_state_ref_v1",
    "materialize_workhorse_selected_world_state_ref_v1",
    "prepare_selected_world_branch_v1",
    "resolve_selected_world_state_v1",
    "resolve_workhorse_selected_world_continuation_v1",
    "resolve_workhorse_selected_world_state_v1",
    "validate_control_suffix_v1",
    "workhorse_selected_world_authority_ref_v1",
)
