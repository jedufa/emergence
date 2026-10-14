"""Bounded exact-state Adaptive Expedition loop above the Science front door.

This module adds action realization to the existing crash-safe founder loop. It
does not schedule providers or reinterpret policy science. ``EXPLORE_GLOBAL``
uses the immutable founder basis; ``BRANCH_LOCAL`` and historical
``REPLICATE`` use only directly branchable exact ``StateRefV1`` material;
``PARK`` and ``RETIRE`` are durable bookkeeping with no execution effect.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

from emergence_contracts import canonical_json_bytes
from emergence_contracts.adaptive_science_expedition_v1 import (
    ChildRealizationKindV1,
    DecisionV1,
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    ExperimentActionKindV1,
    GeneratedChildContinuationV1,
    ObservationFrameV1,
    RetainedParentBankV1,
    StateClassV1,
    StateRefV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import ContentDigest, RunId, SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1
from emergence_contracts.science_request_v1 import (
    EIEL_SCIENCE_REQUEST_SCHEMA_VERSION,
    EielScienceRequestV1,
    validate_eiel_science_request_v1,
)

from .continuation_adapter import (
    CONTROL_SUFFIX_SCHEMA,
    SelectedWorldBranchError,
    materialize_workhorse_child_exact_state_ref_v1,
    prepare_selected_world_branch_v1,
    resolve_workhorse_selected_world_continuation_v1,
    workhorse_selected_world_authority_ref_v1,
)
from .coordinator import PreparedDecisionV1, prepare_decision_v1
from .founder_closed_loop import (
    ChildSelectorV1,
    ClosedLoopActionRealizationV1,
    FounderClosedLoopDependenciesV1,
    FounderClosedLoopResultV1,
    InterruptionHookV1,
    ObservationProjectorV1,
    RetentionLifecycleBoundaryV1,
    SafeFrontierStopV1,
    _seeds_for_action_v1,
    run_founder_closed_loop_v1,
)
from .founder_loop import (
    FounderLoopError,
    build_founder_child_continuation_v1,
    founder_world_count_for_action_v1,
    run_exact_state_child_v1,
    run_founder_child_v1,
)
from .observation_adapter import project_authenticated_science_request_observations_v1
from .parent_package_lifecycle import (
    PARENT_CANDIDATE_SETTLEMENT_FIELD_V1,
    ParentPackageLifecycleError,
    build_parent_package_decision_receipt_v1,
    build_parent_package_terminal_receipt_v1,
    build_staged_parent_candidate_receipt_v1,
    expedition_root_for_ledger_v1,
    parent_candidate_root_for_ledger_v1,
)
from .policy_protocol import ExpeditionPolicyV1, canonical_plain_value_v1

GENERAL_LOOP_PROGRAMME_STAGE_V1: Final[str] = "G4_BOUNDED_TRUE_BRANCHING"
EXACT_ACTION_REQUEST_TEST_V1: Final[str] = "EIEL_ADAPTIVE_EXPEDITION_EXACT_STATE_V1"
_EXACT_ACTION_KINDS_V1: Final[frozenset[ExperimentActionKindV1]] = frozenset(
    {ExperimentActionKindV1.BRANCH_LOCAL, ExperimentActionKindV1.REPLICATE}
)
_BOOKKEEPING_ACTION_KINDS_V1: Final[frozenset[ExperimentActionKindV1]] = frozenset(
    {ExperimentActionKindV1.PARK, ExperimentActionKindV1.RETIRE}
)
# This tuple describes the currently reusable Workhorse exact-state adapter;
# it is not Expedition scientific truth.  A future exact-state adapter may
# supply a different control surface without changing the coordinator.
_WORKHORSE_CONTROL_IDS_V1: Final[tuple[str, str]] = (
    "environment.resource.multiplier",
    "mutation.rate",
)
EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1: Final[str] = (
    "eiel_adaptive_science_observation_frame_v1"
)
EXACT_OBSERVATION_REQUEST_BINDING_KIND_V1: Final[str] = (
    "eiel_adaptive_exact_child_observation_binding_internal_v1"
)
_MAXIMUM_EXACT_OBSERVATION_FRAME_BYTES_V1: Final[int] = 16 * 1024 * 1024


class GeneralClosedLoopError(ValueError):
    """A policy action cannot be realized without weakening its authority."""


ExactChildBuilderV1 = Callable[..., GeneratedChildContinuationV1]
GeneralChildRunnerV1 = Callable[..., Mapping[str, Any]]
ExactChildPromoterV1 = Callable[..., str]


@dataclass(frozen=True, slots=True)
class _ExactObservationFrameProjectionV1:
    """Internal direct projection of one already-public ObservationFrameV1."""

    source: str
    source_file_sha256: str
    frames: tuple[ObservationFrameV1, ...]
    request_sha256: ContentDigest | None

    @property
    def state_refs(self) -> tuple[StateRefV1, ...]:
        return tuple(frame.source_state_ref for frame in self.frames)

    def projection_sha256(self):
        return canonical_sha256(
            {
                "schema_version": (
                    "eiel_adaptive_exact_observation_frame_projection_internal_v1"
                ),
                "source": self.source,
                "source_file_sha256": self.source_file_sha256,
                "request_sha256": (
                    None
                    if self.request_sha256 is None
                    else self.request_sha256.to_dict()
                ),
                "frame_sha256s": [
                    frame.observation_frame_sha256().to_dict() for frame in self.frames
                ],
            }
        )


def _direct_exact_observation_projection_v1(
    source: str | Path,
    *,
    manifest: ExpeditionManifestV1,
) -> _ExactObservationFrameProjectionV1 | None:
    requested = Path(source).expanduser()
    if requested.is_symlink() or not requested.is_file():
        return None
    if requested.stat().st_size > _MAXIMUM_EXACT_OBSERVATION_FRAME_BYTES_V1:
        return None
    try:
        payload = requested.read_bytes()
        raw = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != (
        EXACT_OBSERVATION_FRAME_ARTIFACT_KIND_V1
    ):
        return None
    try:
        frame = ObservationFrameV1.from_dict(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise GeneralClosedLoopError(
            f"exact ObservationFrame artifact is invalid: {error}"
        ) from error
    if payload != canonical_json_bytes(frame.to_dict()):
        raise GeneralClosedLoopError(
            "exact ObservationFrame artifact is not canonical JSON"
        )
    state = frame.source_state_ref
    if (
        frame.expedition_id != manifest.expedition_id
        or frame.observation_profile_id != manifest.observation_profile_id
        or frame.observation_descriptor_set_sha256
        != manifest.observation_descriptor_set_sha256
        or state.state_class is not StateClassV1.BRANCHABLE_EXACT
    ):
        raise GeneralClosedLoopError(
            "exact ObservationFrame differs from the frozen Expedition"
        )
    try:
        authority = workhorse_selected_world_authority_ref_v1(state)
        resolve_workhorse_selected_world_continuation_v1(
            state_ref=state,
            authority_receipt_ref=authority,
        )
    except SelectedWorldBranchError as error:
        raise GeneralClosedLoopError(
            f"exact ObservationFrame StateRef cannot reopen: {error}"
        ) from error
    path = requested.resolve()
    request_binding_path = path.with_name("observation-frame.request-binding.json")
    request_sha256: ContentDigest | None = None
    if request_binding_path.exists() or request_binding_path.is_symlink():
        if request_binding_path.is_symlink() or not request_binding_path.is_file():
            raise GeneralClosedLoopError(
                "exact ObservationFrame request binding is not a regular file"
            )
        try:
            binding = json.loads(request_binding_path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise GeneralClosedLoopError(
                "exact ObservationFrame request binding is unreadable"
            ) from error
        if not isinstance(binding, dict):
            raise GeneralClosedLoopError(
                "exact ObservationFrame request binding is invalid"
            )
        binding_core = {
            key: value for key, value in binding.items() if key != "binding_sha256"
        }
        expected_fields = {
            "schema_version",
            "manifest_sha256",
            "continuation_sha256",
            "child_request_sha256",
            "observation_frame_sha256",
            "binding_sha256",
        }
        try:
            request_sha256 = ContentDigest(
                binding["child_request_sha256"]["algorithm"],
                binding["child_request_sha256"]["hex"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GeneralClosedLoopError(
                "exact ObservationFrame request binding digest is invalid"
            ) from error
        if (
            set(binding) != expected_fields
            or binding.get("schema_version")
            != EXACT_OBSERVATION_REQUEST_BINDING_KIND_V1
            or binding.get("manifest_sha256") != manifest.manifest_sha256().to_dict()
            or binding.get("observation_frame_sha256")
            != frame.observation_frame_sha256().to_dict()
            or binding.get("binding_sha256") != canonical_sha256(binding_core).to_dict()
            or request_binding_path.read_bytes() != canonical_json_bytes(binding)
        ):
            raise GeneralClosedLoopError(
                "exact ObservationFrame request binding changed"
            )
    return _ExactObservationFrameProjectionV1(
        source=str(path),
        source_file_sha256=hashlib.sha256(payload).hexdigest(),
        frames=(frame,),
        request_sha256=request_sha256,
    )


def _publish_exact_observation_frame_v1(
    path: Path,
    frame: ObservationFrameV1,
) -> str:
    payload = canonical_json_bytes(frame.to_dict())
    if len(payload) > _MAXIMUM_EXACT_OBSERVATION_FRAME_BYTES_V1:
        raise GeneralClosedLoopError("exact ObservationFrame exceeds its bounded size")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise GeneralClosedLoopError(
            "exact ObservationFrame destination contains a symbolic link"
        )
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise GeneralClosedLoopError(
                "recovered exact ObservationFrame differs from its child"
            )
        return str(path.resolve())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return str(path.resolve())


def _publish_exact_observation_request_binding_v1(
    path: Path,
    *,
    manifest: ExpeditionManifestV1,
    continuation: GeneratedChildContinuationV1,
    frame: ObservationFrameV1,
) -> None:
    core = {
        "schema_version": EXACT_OBSERVATION_REQUEST_BINDING_KIND_V1,
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "continuation_sha256": canonical_sha256(continuation.to_dict()).to_dict(),
        "child_request_sha256": continuation.child_request_sha256.to_dict(),
        "observation_frame_sha256": frame.observation_frame_sha256().to_dict(),
    }
    value = {**core, "binding_sha256": canonical_sha256(core).to_dict()}
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise GeneralClosedLoopError(
            "exact ObservationFrame request binding destination is invalid"
        )
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise GeneralClosedLoopError(
                "recovered exact ObservationFrame request binding differs"
            )
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def promote_exact_child_observation_v1(
    *,
    continuation: GeneratedChildContinuationV1,
    observation_frame: ObservationFrameV1,
    manifest: ExpeditionManifestV1,
    destination_root: str | Path,
) -> str:
    """Bind the host observation to its newly branchable terminal state."""

    parent = continuation.parent_state_ref
    if (
        continuation.realization_kind is not ChildRealizationKindV1.EXACT_STATE_REQUEST
        or parent is None
        or parent.state_class is not StateClassV1.BRANCHABLE_EXACT
        or observation_frame.source_state_ref.state_class
        is not StateClassV1.OBSERVATION_ONLY
        or observation_frame.source_state_ref.source_artifact_ref.artifact_kind
        != "science.workhorse_selected_world_host_observation_v1"
    ):
        raise GeneralClosedLoopError(
            "exact child terminal promotion lacks its host observation or parent"
        )
    root = Path(destination_root).expanduser().absolute()
    identity = canonical_sha256(
        {
            "schema_version": "eiel_adaptive_exact_child_promotion_identity_v1",
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "continuation_sha256": canonical_sha256(continuation.to_dict()).to_dict(),
            "host_observation_frame_sha256": (
                observation_frame.observation_frame_sha256().to_dict()
            ),
        }
    ).hex
    selected_root = root / f"exact-child-{identity[:32]}" / "selected-world"
    try:
        state, _authority, _resolved = materialize_workhorse_child_exact_state_ref_v1(
            observation_frame.source_state_ref.source_artifact_ref.locator,
            selected_root,
            parent_state_ref=parent,
            expedition_id=manifest.expedition_id,
            world_trajectory_id=continuation.child_trajectory_id,
        )
    except SelectedWorldBranchError as error:
        raise GeneralClosedLoopError(
            f"exact child terminal cannot become a retained parent: {error}"
        ) from error
    promoted = ObservationFrameV1(
        expedition_id=manifest.expedition_id,
        world_trajectory_id=continuation.child_trajectory_id,
        logical_world_id=state.logical_world_id,
        next_absolute_tick=state.next_absolute_tick,
        semantic_world_sha256=state.semantic_world_sha256,
        control_history_sha256=state.control_history_sha256,
        observation_profile_id=observation_frame.observation_profile_id,
        observation_profile_sha256=observation_frame.observation_profile_sha256,
        observation_descriptor_set_sha256=(
            observation_frame.observation_descriptor_set_sha256
        ),
        reducer_implementation_sha256=(observation_frame.reducer_implementation_sha256),
        observations=observation_frame.observations,
        control_readback=observation_frame.control_readback,
        policy_facing_facts=observation_frame.policy_facing_facts,
        descriptor_values=observation_frame.descriptor_values,
        calibration=observation_frame.calibration,
        source_state_ref=state,
        previous_frame_sha256=None,
    )
    promoted_path = selected_root.parent / "observation-frame.json"
    published = _publish_exact_observation_frame_v1(
        promoted_path,
        promoted,
    )
    _publish_exact_observation_request_binding_v1(
        promoted_path.with_name("observation-frame.request-binding.json"),
        manifest=manifest,
        continuation=continuation,
        frame=promoted,
    )
    return published


@dataclass(frozen=True, slots=True)
class GeneralClosedLoopDependenciesV1:
    """Narrow injection seam for focused recovery and action-path tests."""

    observation_projector: ObservationProjectorV1 = (
        project_authenticated_science_request_observations_v1
    )
    decision_preparer: Callable[..., PreparedDecisionV1] = prepare_decision_v1
    founder_child_builder: Callable[..., GeneratedChildContinuationV1] = (
        build_founder_child_continuation_v1
    )
    exact_child_builder: ExactChildBuilderV1 = (
        lambda **kwargs: build_exact_state_child_continuation_v1(**kwargs)
    )
    child_runner: GeneralChildRunnerV1 = (
        lambda continuation, **options: run_expedition_child_v1(continuation, **options)
    )
    exact_child_promoter: ExactChildPromoterV1 = promote_exact_child_observation_v1
    retention_lifecycle_boundary: RetentionLifecycleBoundaryV1 | None = None
    programme_stage: str = GENERAL_LOOP_PROGRAMME_STAGE_V1
    historical_continuation_authority: bool = True


GeneralClosedLoopResultV1 = FounderClosedLoopResultV1


def _mapping(value: object, *, field: str) -> dict[str, object]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not dict:
        raise GeneralClosedLoopError(f"{field} must be a canonical mapping")
    return plain


def _sequence(value: object, *, field: str) -> list[object]:
    plain = canonical_plain_value_v1(value)
    if type(plain) is not list:
        raise GeneralClosedLoopError(f"{field} must be a canonical sequence")
    return plain


def _decision_action_v1(
    decision: DecisionV1,
    *,
    action_id: str,
) -> tuple[object, dict[str, object]]:
    rows = [row for row in decision.action_rows if row.action_id == action_id]
    if len(rows) != 1:
        raise GeneralClosedLoopError(
            "Decision does not contain exactly one requested action row"
        )
    proposal = _mapping(decision.policy_proposal.to_value(), field="policy proposal")
    actions = [
        _mapping(raw, field="policy semantic action")
        for raw in _sequence(
            proposal.get("semantic_actions"), field="policy semantic actions"
        )
        if isinstance(raw, Mapping) and raw.get("action_id") == action_id
    ]
    if len(actions) != 1:
        raise GeneralClosedLoopError(
            "policy proposal does not contain exactly one requested action"
        )
    return rows[0], actions[0]


def _manifest_control_domains_v1(
    manifest: ExpeditionManifestV1,
) -> dict[str, list[object]]:
    search = _mapping(manifest.search_space.to_value(), field="manifest search space")
    domains: dict[str, list[object]] = {}
    for raw in _sequence(search.get("controls"), field="manifest controls"):
        row = _mapping(raw, field="manifest control")
        control_id = row.get("control_id")
        values = _sequence(row.get("allowed_values"), field="allowed control values")
        if type(control_id) is not str or not control_id or not values:
            raise GeneralClosedLoopError(
                "manifest control domain lacks a bounded identity or value set"
            )
        if control_id in domains:
            raise GeneralClosedLoopError("manifest control identity is duplicated")
        domains[control_id] = values
    return domains


def _active_controls_v1(resolved: object) -> dict[str, object]:
    value = getattr(resolved, "active_scientific_control_state", None)
    state = _mapping(value, field="selected parent active control state")
    controls = _mapping(state.get("controls"), field="selected parent active controls")
    if set(controls) != set(_WORKHORSE_CONTROL_IDS_V1):
        raise GeneralClosedLoopError(
            "selected parent does not bind the two supported Workhorse controls"
        )
    return controls


def _exact_action_controls_v1(
    *,
    action_kind: ExperimentActionKindV1,
    action: Mapping[str, object],
    active_controls: Mapping[str, object],
    manifest: ExpeditionManifestV1,
) -> dict[str, object]:
    raw_bindings = action.get("control_bindings")
    if raw_bindings is None and action_kind is ExperimentActionKindV1.REPLICATE:
        bindings = [
            {"control_id": control_id, "value": active_controls[control_id]}
            for control_id in _WORKHORSE_CONTROL_IDS_V1
        ]
    else:
        bindings = [
            _mapping(raw, field="exact action control binding")
            for raw in _sequence(raw_bindings, field="exact action control bindings")
        ]
    if any(set(row) != {"control_id", "value"} for row in bindings):
        raise GeneralClosedLoopError("exact action control fields must be exact")
    by_id = {str(row.get("control_id")): row.get("value") for row in bindings}
    if len(by_id) != len(bindings) or set(by_id) != set(_WORKHORSE_CONTROL_IDS_V1):
        raise GeneralClosedLoopError(
            "exact action requires exactly resource multiplier and mutation rate"
        )
    domains = _manifest_control_domains_v1(manifest)
    for control_id, value in by_id.items():
        if control_id not in domains or value not in domains[control_id]:
            raise GeneralClosedLoopError(
                "exact action left the manifest-authorized control domain"
            )
    if action_kind is ExperimentActionKindV1.BRANCH_LOCAL and all(
        by_id[control_id] == active_controls[control_id]
        for control_id in _WORKHORSE_CONTROL_IDS_V1
    ):
        raise GeneralClosedLoopError(
            "BRANCH_LOCAL must change at least one policy-selected control"
        )
    if action_kind is ExperimentActionKindV1.REPLICATE and any(
        by_id[control_id] != active_controls[control_id]
        for control_id in _WORKHORSE_CONTROL_IDS_V1
    ):
        raise GeneralClosedLoopError(
            "historical REPLICATE must preserve the selected parent's controls"
        )
    return by_id


def _founder_request_v1(
    manifest: ExpeditionManifestV1,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
) -> EielScienceRequestV1:
    request = validate_eiel_science_request_v1(founder_request)
    if request.expect_worlds != 1 or not any(
        ref.artifact_kind == "eiel_science_request_v1"
        and ref.content_digest == request.request_sha256()
        for ref in manifest.founder_basis_refs
    ):
        raise GeneralClosedLoopError(
            "exact child requires the manifest-bound singleton founder declaration"
        )
    return request


def build_exact_state_child_continuation_v1(
    *,
    manifest: ExpeditionManifestV1,
    decision: DecisionV1,
    action_id: str,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
    parent_state_ref: StateRefV1,
) -> GeneratedChildContinuationV1:
    """Build one fresh, singleton Science Request from an exact retained parent."""

    if (
        decision.expedition_id != manifest.expedition_id
        or decision.manifest_sha256 != manifest.manifest_sha256()
    ):
        raise GeneralClosedLoopError(
            "Decision does not bind the supplied Expedition manifest"
        )
    if (
        not isinstance(parent_state_ref, StateRefV1)
        or parent_state_ref.expedition_id != manifest.expedition_id
        or parent_state_ref.state_class is not StateClassV1.BRANCHABLE_EXACT
    ):
        raise GeneralClosedLoopError(
            "historical action requires a directly branchable exact StateRef"
        )
    row, action = _decision_action_v1(decision, action_id=action_id)
    action_kind = row.action_kind
    state_id = f"SR-{parent_state_ref.state_ref_sha256().hex}"
    if (
        action_kind not in _EXACT_ACTION_KINDS_V1
        or row.parent_state_ref_sha256 != parent_state_ref.state_ref_sha256()
        or row.effective_next_absolute_tick != parent_state_ref.next_absolute_tick
        or action.get("parent_state_ref_id") != state_id
        or action.get("parent_next_absolute_tick")
        != parent_state_ref.next_absolute_tick
    ):
        raise GeneralClosedLoopError(
            "historical action differs from its exact selected parent"
        )
    base = _founder_request_v1(manifest, founder_request)
    try:
        authority_ref = workhorse_selected_world_authority_ref_v1(parent_state_ref)
        resolved = resolve_workhorse_selected_world_continuation_v1(
            state_ref=parent_state_ref,
            authority_receipt_ref=authority_ref,
        )
    except SelectedWorldBranchError as error:
        raise GeneralClosedLoopError(
            f"selected parent authority cannot reopen exactly: {error}"
        ) from error
    active_controls = _active_controls_v1(resolved)
    controls = _exact_action_controls_v1(
        action_kind=action_kind,
        action=action,
        active_controls=active_controls,
        manifest=manifest,
    )
    branch_tick = parent_state_ref.next_absolute_tick
    target_tick = branch_tick + manifest.epoch_ticks
    if target_tick > manifest.maximum_absolute_tick:
        raise GeneralClosedLoopError(
            "exact child horizon exceeds the manifest absolute-tick bound"
        )
    control_suffix = {
        "schema_version": CONTROL_SUFFIX_SCHEMA,
        "effective_next_absolute_tick": branch_tick,
        "controls": [
            {
                "control_id": control_id,
                "onset_tick": branch_tick,
                "payload": {"value": controls[control_id]},
            }
            for control_id in _WORKHORSE_CONTROL_IDS_V1
        ],
    }
    identity_core = {
        "schema_version": "eiel_adaptive_exact_child_identity_v1",
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "decision_sha256": decision.decision_sha256().to_dict(),
        "action_id": action_id,
        "parent_state_ref_sha256": parent_state_ref.state_ref_sha256().to_dict(),
        "effective_next_absolute_tick": branch_tick,
        "target_next_absolute_tick": target_tick,
        "control_suffix": control_suffix,
    }
    identity = canonical_sha256(identity_core).hex
    child_run_id = RunId(f"run-exact-{identity[:24]}")
    child_trajectory_id = f"trajectory-exact-{identity[:24]}"
    try:
        prepared = prepare_selected_world_branch_v1(
            resolved,
            child_run_id=child_run_id,
            child_trajectory_id=child_trajectory_id,
            effective_next_absolute_tick=branch_tick,
            control_suffix=control_suffix,
        )
    except SelectedWorldBranchError as error:
        raise GeneralClosedLoopError(
            f"exact child prefix cannot be prepared: {error}"
        ) from error

    source = base.to_dict()
    source_observe = _mapping(source["observe"], field="founder observe")
    source_reduce = _mapping(source["reduce"], field="founder reducer")
    source_configuration = _mapping(
        source_reduce["configuration"], field="founder reducer configuration"
    )
    source_test = source.get("test")
    if type(source_test) is not str or not source_test:
        raise GeneralClosedLoopError("founder Science Request test identity is invalid")
    decision_family = source_configuration.get(
        "decision_family", f"EXPEDITION_{manifest.expedition_id}"
    )
    reporting_group = source_configuration.get(
        "reporting_group", f"EXPEDITION_{manifest.expedition_id}"
    )
    if (
        type(decision_family) is not str
        or not decision_family
        or type(reporting_group) is not str
        or not reporting_group
    ):
        raise GeneralClosedLoopError("founder reducer reporting identities are invalid")
    dimension_values = {
        "expedition_action": [action_id],
        "expedition_parent": [parent_state_ref.state_ref_sha256().hex],
        "expedition_replicate": [0],
    }
    request_value: dict[str, object] = {
        "schema_version": str(EIEL_SCIENCE_REQUEST_SCHEMA_VERSION),
        "campaign_id": f"EIEL-EXP-EXACT-{identity[:24]}",
        # Preserve the declared Science Request test identity.  The exact
        # continuation mechanism must not mint a permanent scientific label.
        "test": source_test,
        "question": {
            "description": manifest.question_description,
            "no_claims": True,
        },
        "world": deepcopy(source["world"]),
        "run": {
            "ticks": target_tick,
            "batch": {"policy": "exact", "worlds": 1},
            "checkpoint_ticks": [target_tick],
        },
        "dimensions": dimension_values,
        "world_template": {
            "controls": {
                control_id: {
                    "value": controls[control_id],
                    "schedule": {
                        "kind": "step_absolute_tick",
                        "onset_tick": branch_tick,
                        "baseline": active_controls[control_id],
                    },
                }
                for control_id in _WORKHORSE_CONTROL_IDS_V1
            }
        },
        "replicate_dimension": "expedition_replicate",
        "observe": {
            "profile": source_observe["profile"],
            # Exact Workhorse children execute with runtime observation OFF.
            # The terminal host adapter authenticates checkpoint state,
            # cumulative counters, and control readback, but deliberately has
            # no event-kind or birth/death causal rows.
            "events": ["total_event_count"],
            "state": ["genome_identity", "population_count"],
            "runtime_counters": [
                "birth_count",
                "death_count",
                "mutation_count",
            ],
            "snapshot_ticks": [branch_tick, target_tick],
        },
        "reduce": {
            "reducer": source_reduce["reducer"],
            "configuration": {
                "analysis_dimensions": list(dimension_values),
                "axis_a": "expedition_action",
                "axis_b": "expedition_parent",
                "decision_family": decision_family,
                "reporting_group": reporting_group,
                "tracked_policy_loci": source_configuration["tracked_policy_loci"],
                "tracked_trait_loci": source_configuration["tracked_trait_loci"],
            },
        },
        "expect_worlds": 1,
        "continuation": {
            "schema_version": "eiel_science_selected_world_continuation_v1",
            "state_ref": parent_state_ref.to_dict(),
            "authority_receipt_ref": authority_ref.to_dict(),
        },
    }
    child_request = validate_eiel_science_request_v1(request_value)
    payload = CanonicalPayloadV1.from_value(
        SchemaVersion(str(EIEL_SCIENCE_REQUEST_SCHEMA_VERSION)),
        child_request.to_dict(),
    )
    child = GeneratedChildContinuationV1(
        expedition_id=manifest.expedition_id,
        manifest_sha256=manifest.manifest_sha256(),
        decision_sha256=decision.decision_sha256(),
        action_id=action_id,
        action_kind=action_kind,
        realization_kind=ChildRealizationKindV1.EXACT_STATE_REQUEST,
        parent_state_ref=parent_state_ref,
        founder_basis_sha256=manifest.founder_basis_sha256(),
        child_trajectory_id=prepared.child_trajectory_id,
        child_run_id=prepared.child_run_id,
        retry_of_run_id=None,
        effective_next_absolute_tick=prepared.effective_next_absolute_tick,
        control_schedule_sha256=prepared.control_suffix_sha256,
        child_request=payload,
        child_request_sha256=payload.content_digest(),
    )
    return GeneratedChildContinuationV1.from_dict(child.to_dict())


def _available_exact_states_v1(
    observation_frames: Sequence[ObservationFrameV1],
    retained_parent_bank: RetainedParentBankV1,
) -> dict[str, StateRefV1]:
    result: dict[str, StateRefV1] = {}
    for state in (
        *(frame.source_state_ref for frame in observation_frames),
        *(entry.state_ref for entry in retained_parent_bank.entries),
    ):
        state_id = f"SR-{state.state_ref_sha256().hex}"
        prior = result.get(state_id)
        if prior is not None and prior != state:
            raise GeneralClosedLoopError("StateRef identity collision")
        result[state_id] = state
    return result


def _realize_general_actions_v1(
    *,
    manifest: ExpeditionManifestV1,
    prepared: PreparedDecisionV1,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
    observation_frames: Sequence[ObservationFrameV1],
    retained_parent_bank: RetainedParentBankV1,
    physical_batch_worlds: int,
    founder_builder: Callable[..., GeneratedChildContinuationV1],
    exact_builder: ExactChildBuilderV1,
) -> ClosedLoopActionRealizationV1:
    states = _available_exact_states_v1(observation_frames, retained_parent_bank)
    children: list[GeneratedChildContinuationV1] = []
    bookkeeping: list[dict[str, object]] = []
    founder_ids: list[str] = []
    exact_ids: list[str] = []
    for row in prepared.decision.action_rows:
        if row.action_kind is ExperimentActionKindV1.EXPLORE_GLOBAL:
            if (
                row.parent_state_ref_sha256 is not None
                or row.effective_next_absolute_tick != 0
            ):
                raise GeneralClosedLoopError(
                    "EXPLORE_GLOBAL cannot carry historical state authority"
                )
            action = _mapping(
                row.action_payload.to_value(), field="global action payload"
            )
            try:
                world_count = founder_world_count_for_action_v1(action)
                child = founder_builder(
                    manifest=manifest,
                    decision=prepared.decision,
                    action_id=row.action_id,
                    founder_request=founder_request,
                    replicate_seeds=_seeds_for_action_v1(
                        manifest,
                        frontier_ordinal=prepared.decision.frontier_ordinal,
                        action_id=row.action_id,
                        world_count=world_count,
                    ),
                    physical_batch_worlds=physical_batch_worlds,
                )
            except FounderLoopError as error:
                raise GeneralClosedLoopError(
                    f"global founder action cannot be represented: {error}"
                ) from error
            founder_ids.append(row.action_id)
            children.append(child)
        elif row.action_kind in _EXACT_ACTION_KINDS_V1:
            parent_digest = row.parent_state_ref_sha256
            if parent_digest is None:
                raise GeneralClosedLoopError(
                    "historical action lacks a selected exact parent"
                )
            parent = states.get(f"SR-{parent_digest.hex}")
            if (
                parent is None
                or parent.state_class is not StateClassV1.BRANCHABLE_EXACT
            ):
                raise GeneralClosedLoopError(
                    "historical action parent is unavailable or not directly exact"
                )
            child = exact_builder(
                manifest=manifest,
                decision=prepared.decision,
                action_id=row.action_id,
                founder_request=founder_request,
                parent_state_ref=parent,
            )
            if (
                child.realization_kind is not ChildRealizationKindV1.EXACT_STATE_REQUEST
                or child.parent_state_ref != parent
                or child.effective_next_absolute_tick != parent.next_absolute_tick
                or child.retry_of_run_id is not None
            ):
                raise GeneralClosedLoopError(
                    "exact child builder changed its selected-parent semantics"
                )
            exact_ids.append(row.action_id)
            children.append(child)
        elif row.action_kind in _BOOKKEEPING_ACTION_KINDS_V1:
            bookkeeping.append(
                {
                    "action_id": row.action_id,
                    "action_kind": row.action_kind.value,
                    "effective_next_absolute_tick": row.effective_next_absolute_tick,
                    "parent_state_ref_sha256": (
                        None
                        if row.parent_state_ref_sha256 is None
                        else row.parent_state_ref_sha256.to_dict()
                    ),
                    "reason_codes": list(row.reason_codes),
                    "action_payload_sha256": row.action_payload.content_digest().to_dict(),
                    "outcome": "BOOKKEEPING_ONLY_NO_EXECUTION",
                }
            )
        else:
            raise GeneralClosedLoopError(
                f"action kind {row.action_kind.value} has no V1 general-loop realization"
            )
    ordered_children = tuple(sorted(children, key=lambda child: child.action_id))
    return ClosedLoopActionRealizationV1(
        children=ordered_children,
        decision_metadata={
            "action_realization_schema_version": (
                "eiel_adaptive_expedition_general_action_realization_v1"
            ),
            "proposed_founder_action_ids": sorted(founder_ids),
            "proposed_exact_action_ids": sorted(exact_ids),
            "bookkeeping_actions": sorted(
                bookkeeping, key=lambda item: str(item["action_id"])
            ),
            "execution_authority": "EXISTING_SCIENCE_REQUEST_FRONT_DOOR_ONLY",
            "no_hidden_reward_authority": True,
        },
    )


def run_expedition_child_v1(
    continuation: GeneratedChildContinuationV1,
    *,
    repo_root: str | None = None,
    materialization_parent: str | None = None,
    runtime_parent: str | None = None,
    resume_root: str | None = None,
    **front_door_options: Any,
) -> dict[str, Any]:
    """Dispatch one sealed child only through the canonical Science front door."""

    if continuation.realization_kind is ChildRealizationKindV1.FOUNDER_BASIS_REQUEST:
        return run_founder_child_v1(
            continuation,
            repo_root=repo_root,
            materialization_parent=materialization_parent,
            runtime_parent=runtime_parent,
            resume_root=resume_root,
            **front_door_options,
        )
    return run_exact_state_child_v1(
        continuation,
        repo_root=repo_root,
        materialization_parent=materialization_parent,
        runtime_parent=runtime_parent,
        resume_root=resume_root,
        **front_door_options,
    )


def run_general_closed_loop_v1(
    *,
    manifest: ExpeditionManifestV1,
    policy: ExpeditionPolicyV1,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
    ledger_root: str | Path,
    atlas: EvolutionaryAtlasV1,
    retained_parent_bank: RetainedParentBankV1,
    logical_capacity: int,
    maximum_frontiers: int,
    initial_observation_sources: Sequence[str | Path] = (),
    shadow_observation_frontiers: Sequence[Sequence[str | Path]] | None = None,
    shadow_mode: bool = False,
    physical_batch_worlds: int | None = None,
    repo_root: str | None = None,
    materialization_parent: str | None = None,
    runtime_parent: str | Path | None = None,
    dependencies: GeneralClosedLoopDependenciesV1 = GeneralClosedLoopDependenciesV1(),
    child_selector: ChildSelectorV1 | None = None,
    safe_frontier_stop: SafeFrontierStopV1 | None = None,
    interruption_hook: InterruptionHookV1 | None = None,
) -> GeneralClosedLoopResultV1:
    """Run or recover the bounded founder/exact-state action loop."""

    expedition_root = expedition_root_for_ledger_v1(ledger_root)
    parent_candidate_root = parent_candidate_root_for_ledger_v1(ledger_root)

    def project(source: str | Path, *, manifest: ExpeditionManifestV1):
        direct = _direct_exact_observation_projection_v1(
            source,
            manifest=manifest,
        )
        if direct is not None:
            return direct
        return dependencies.observation_projector(source, manifest=manifest)

    def run_child(
        continuation: GeneratedChildContinuationV1,
        **options: Any,
    ) -> Mapping[str, Any]:
        result = dependencies.child_runner(continuation, **options)
        if (
            continuation.realization_kind
            is not ChildRealizationKindV1.EXACT_STATE_REQUEST
        ):
            return result
        raw_source = result.get("runtime_root")
        if not isinstance(raw_source, (str, Path)) or not str(raw_source):
            raise GeneralClosedLoopError(
                "exact child result lacks its canonical Science Request root"
            )
        projected = project(raw_source, manifest=manifest)
        projected_request_sha256 = getattr(projected, "request_sha256", None)
        try:
            normalized_request_sha256 = validate_eiel_science_request_v1(
                continuation.child_request.to_value()
            ).request_sha256()
        except (TypeError, ValueError):
            normalized_request_sha256 = None
        expected_request_digests = {continuation.child_request_sha256}
        if normalized_request_sha256 is not None:
            expected_request_digests.add(normalized_request_sha256)
        if projected_request_sha256 not in expected_request_digests:
            raise GeneralClosedLoopError(
                "authenticated exact child result names a different Science Request"
            )
        frames = tuple(projected.frames)
        if frames and all(
            frame.source_state_ref.state_class is StateClassV1.BRANCHABLE_EXACT
            for frame in frames
        ):
            # Synthetic/injected runners and an already promoted crash recovery
            # source may supply the exact public frame directly.
            return result
        if (
            len(frames) != 1
            or frames[0].source_state_ref.state_class
            is not StateClassV1.OBSERVATION_ONLY
        ):
            raise GeneralClosedLoopError(
                "exact child did not yield one authenticated host observation"
            )
        promoted_source = dependencies.exact_child_promoter(
            continuation=continuation,
            observation_frame=frames[0],
            manifest=manifest,
            destination_root=parent_candidate_root,
        )
        promoted = project(promoted_source, manifest=manifest)
        if (
            len(promoted.frames) != 1
            or promoted.frames[0].source_state_ref.state_class
            is not StateClassV1.BRANCHABLE_EXACT
        ):
            raise GeneralClosedLoopError(
                "exact child promoter did not publish one branchable StateRef"
            )
        try:
            candidate_receipt = build_staged_parent_candidate_receipt_v1(
                manifest=manifest,
                observation_frame=promoted.frames[0],
                run_root=expedition_root,
                raw_child_runtime_root=raw_source,
            )
        except ParentPackageLifecycleError as error:
            raise GeneralClosedLoopError(
                f"exact child parent package cannot be staged safely: {error}"
            ) from error
        return {
            **dict(result),
            "runtime_root": str(promoted_source),
            "science_request_runtime_root": str(Path(raw_source).absolute()),
            "terminal_exact_state_ref_sha256": (
                promoted.frames[0].source_state_ref.state_ref_sha256().hex
            ),
            PARENT_CANDIDATE_SETTLEMENT_FIELD_V1: candidate_receipt,
        }

    def realize(**options: Any) -> ClosedLoopActionRealizationV1:
        return _realize_general_actions_v1(
            **options,
            exact_builder=dependencies.exact_child_builder,
        )

    def record_retention_lifecycle(
        *,
        phase: str,
        manifest: ExpeditionManifestV1,
        ledger: Mapping[str, object],
        frontier_ordinal: int,
        retained_parent_bank: RetainedParentBankV1,
        observation_frames: Sequence[ObservationFrameV1],
        retention_outcomes: Mapping[str, object] | None,
        terminal_status: str | None,
    ) -> Mapping[str, object]:
        try:
            if phase == "DECISION":
                if retention_outcomes is None or terminal_status is not None:
                    raise ParentPackageLifecycleError(
                        "Decision package classification inputs are invalid"
                    )
                return build_parent_package_decision_receipt_v1(
                    manifest=manifest,
                    ledger=ledger,
                    frontier_ordinal=frontier_ordinal,
                    observation_frames=observation_frames,
                    retained_parent_bank=retained_parent_bank,
                    retention_outcomes=retention_outcomes,
                    run_root=expedition_root,
                )
            if phase == "TERMINAL":
                if retention_outcomes is not None or terminal_status is None:
                    raise ParentPackageLifecycleError(
                        "terminal package classification inputs are invalid"
                    )
                return build_parent_package_terminal_receipt_v1(
                    manifest=manifest,
                    ledger=ledger,
                    frontier_ordinal=frontier_ordinal,
                    retained_parent_bank=retained_parent_bank,
                    terminal_status=terminal_status,
                    run_root=expedition_root,
                )
            raise ParentPackageLifecycleError(
                "parent package lifecycle phase is unsupported"
            )
        except ParentPackageLifecycleError as error:
            raise GeneralClosedLoopError(
                f"physical Retained Parent Bank classification failed: {error}"
            ) from error

    return run_founder_closed_loop_v1(
        manifest=manifest,
        policy=policy,
        founder_request=founder_request,
        ledger_root=ledger_root,
        atlas=atlas,
        retained_parent_bank=retained_parent_bank,
        logical_capacity=logical_capacity,
        maximum_frontiers=maximum_frontiers,
        initial_observation_sources=initial_observation_sources,
        shadow_observation_frontiers=shadow_observation_frontiers,
        shadow_mode=shadow_mode,
        physical_batch_worlds=physical_batch_worlds,
        repo_root=repo_root,
        materialization_parent=materialization_parent,
        runtime_parent=runtime_parent,
        dependencies=FounderClosedLoopDependenciesV1(
            observation_projector=project,
            decision_preparer=dependencies.decision_preparer,
            child_builder=dependencies.founder_child_builder,
            child_runner=run_child,
            action_realizer=realize,
            retention_lifecycle_recorder=record_retention_lifecycle,
            retention_lifecycle_boundary=dependencies.retention_lifecycle_boundary,
            programme_stage=dependencies.programme_stage,
            historical_continuation_authority=(
                dependencies.historical_continuation_authority
            ),
        ),
        child_selector=child_selector,
        safe_frontier_stop=safe_frontier_stop,
        interruption_hook=interruption_hook,
    )


__all__ = (
    "EXACT_ACTION_REQUEST_TEST_V1",
    "GENERAL_LOOP_PROGRAMME_STAGE_V1",
    "GeneralClosedLoopDependenciesV1",
    "GeneralClosedLoopError",
    "GeneralClosedLoopResultV1",
    "build_exact_state_child_continuation_v1",
    "run_expedition_child_v1",
    "run_general_closed_loop_v1",
)
