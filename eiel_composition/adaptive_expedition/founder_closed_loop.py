"""Crash-safe founder-only Adaptive Expedition closed loop.

This module composes the generic coordinator with the existing Science Request
front door.  It is deliberately a G2.5 mechanism: every generated child starts
from the immutable founder basis at absolute tick zero.  Observation-only
StateRefs are never promoted into historical continuation authority.

The append-only Expedition ledger is the recovery boundary.  A child and its
dedicated runtime parent are sealed before dispatch.  If the coordinator dies
after the front door creates a runtime occurrence but before settlement, the
single occurrence beneath that exact action parent is resumed.  A terminal
failed occurrence is preserved and a later recovery may create a fresh
occurrence of the same sealed child request; this is not a changed branch or a
retry of the parent.  No scheduler, provider, database, or branch
approximation is introduced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Final, Protocol

from emergence_contracts.adaptive_science_expedition_v1 import (
    ChildRealizationKindV1,
    DecisionV1,
    EvolutionaryAtlasV1,
    ExpeditionManifestV1,
    ExperimentActionKindV1,
    GeneratedChildContinuationV1,
    ObservationFrameV1,
    RetainedParentBankV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.science_request_v1 import (
    EielScienceRequestV1,
    validate_eiel_science_request_v1,
)

from .coordinator import PreparedDecisionV1, prepare_decision_v1
from .founder_loop import (
    FounderLoopError,
    build_founder_child_continuation_v1,
    founder_world_count_for_action_v1,
    run_founder_child_v1,
)
from .ledger import (
    ACTION_DISPATCH_BOUND,
    ACTION_SETTLED,
    DECISION_PREPARED,
    EXPEDITION_TERMINAL,
    FRONTIER_OBSERVED,
    STOP_REQUESTED,
    append_ledger_store_event_v1,
    initialize_ledger_store_v1,
    new_expedition_ledger_v1,
    read_ledger_store_v1,
    reconstruct_expedition_frontier_v1,
)
from .observation_adapter import (
    AuthenticatedObservationProjectionV1,
    project_authenticated_science_request_observations_v1,
)
from .genealogy import (
    GenealogyIntegrityError,
    associate_observation_state_ref_v1,
    build_parent_retention_bindings_v1,
)
from .parent_package_lifecycle import (
    PARENT_CANDIDATE_SETTLEMENT_FIELD_V1,
    PARENT_PACKAGE_LEDGER_FIELD_V1,
)
from .policy_protocol import ExpeditionPolicyV1, canonical_plain_value_v1
from .state_materialization import apply_policy_state_deltas_v1

_FOUNDER_KINDS: Final[frozenset[ExperimentActionKindV1]] = frozenset(
    {
        ExperimentActionKindV1.EXPLORE_GLOBAL,
        ExperimentActionKindV1.REPLICATE,
        ExperimentActionKindV1.CAUSAL_PROBE,
    }
)
_SAFE_PATH_COMPONENT: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")


class FounderClosedLoopError(ValueError):
    """The founder-only loop cannot progress without weakening its boundary."""


class ObservationProjectorV1(Protocol):
    def __call__(
        self,
        source: str | Path,
        *,
        manifest: ExpeditionManifestV1 | Mapping[str, object],
    ) -> AuthenticatedObservationProjectionV1: ...


class FounderChildRunnerV1(Protocol):
    def __call__(
        self,
        continuation: GeneratedChildContinuationV1,
        **options: Any,
    ) -> Mapping[str, Any]: ...


class InterruptionHookV1(Protocol):
    def __call__(
        self,
        stage: str,
        frontier_ordinal: int,
        action_id: str | None,
    ) -> None: ...


class ChildSelectorV1(Protocol):
    def __call__(
        self,
        children: tuple[GeneratedChildContinuationV1, ...],
    ) -> Sequence[str]: ...


class SafeFrontierStopV1(Protocol):
    """Observe a safe unbound/settled boundary and request durable stop or None."""

    def __call__(
        self,
        frontier_ordinal: int,
        ledger: Mapping[str, object],
    ) -> Mapping[str, object] | None: ...


class RetentionLifecycleRecorderV1(Protocol):
    """Fold internal physical-retention facts into the existing ledger."""

    def __call__(
        self,
        *,
        phase: str,
        manifest: ExpeditionManifestV1,
        ledger: Mapping[str, object],
        frontier_ordinal: int,
        retained_parent_bank: RetainedParentBankV1,
        observation_frames: Sequence[ObservationFrameV1],
        retention_outcomes: Mapping[str, object] | None,
        terminal_status: str | None,
    ) -> Mapping[str, object]: ...


class RetentionLifecycleBoundaryV1(Protocol):
    """Finalize exact active retention only after its Decision is durable."""

    def __call__(
        self,
        *,
        manifest: ExpeditionManifestV1,
        ledger: Mapping[str, object],
        frontier_ordinal: int,
        decision_payload: Mapping[str, object],
        selected_children: Sequence[GeneratedChildContinuationV1],
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class ClosedLoopActionRealizationV1:
    """Internal deterministic action lowering folded into the Decision event."""

    children: tuple[GeneratedChildContinuationV1, ...]
    decision_metadata: Mapping[str, object]


class ClosedLoopActionRealizerV1(Protocol):
    def __call__(
        self,
        *,
        manifest: ExpeditionManifestV1,
        prepared: PreparedDecisionV1,
        founder_request: Mapping[str, object] | EielScienceRequestV1,
        observation_frames: Sequence[ObservationFrameV1],
        retained_parent_bank: RetainedParentBankV1,
        physical_batch_worlds: int,
        founder_builder: Callable[..., GeneratedChildContinuationV1],
    ) -> ClosedLoopActionRealizationV1: ...


@dataclass(frozen=True, slots=True)
class FounderClosedLoopDependenciesV1:
    """Narrow dependency-injection seam for recovery and focused tests."""

    observation_projector: ObservationProjectorV1 = (
        project_authenticated_science_request_observations_v1
    )
    decision_preparer: Callable[..., PreparedDecisionV1] = prepare_decision_v1
    child_builder: Callable[..., GeneratedChildContinuationV1] = (
        build_founder_child_continuation_v1
    )
    child_runner: FounderChildRunnerV1 = run_founder_child_v1
    action_realizer: ClosedLoopActionRealizerV1 | None = None
    retention_lifecycle_recorder: RetentionLifecycleRecorderV1 | None = None
    retention_lifecycle_boundary: RetentionLifecycleBoundaryV1 | None = None
    programme_stage: str = "G2.5_FOUNDER_ONLY"
    historical_continuation_authority: bool = False


@dataclass(frozen=True, slots=True)
class FounderClosedLoopResultV1:
    """Internal run result; durable truth remains the validated ledger."""

    ledger: Mapping[str, object]
    decisions: tuple[DecisionV1, ...]
    children: tuple[GeneratedChildContinuationV1, ...]
    atlas: EvolutionaryAtlasV1
    retained_parent_bank: RetainedParentBankV1
    shadow_mode: bool
    terminal: bool


def _safe_component(value: str) -> str:
    result = _SAFE_PATH_COMPONENT.sub("-", value).strip("-.")
    if not result:
        raise FounderClosedLoopError("expedition identity has no safe path projection")
    return result[:96]


def _retention_lifecycle_metadata_v1(
    *,
    dependencies: FounderClosedLoopDependenciesV1,
    phase: str,
    manifest: ExpeditionManifestV1,
    ledger: Mapping[str, object],
    frontier_ordinal: int,
    retained_parent_bank: RetainedParentBankV1,
    observation_frames: Sequence[ObservationFrameV1] = (),
    retention_outcomes: Mapping[str, object] | None = None,
    terminal_status: str | None = None,
) -> dict[str, object]:
    recorder = dependencies.retention_lifecycle_recorder
    if recorder is None:
        return {}
    value = canonical_plain_value_v1(
        recorder(
            phase=phase,
            manifest=manifest,
            ledger=ledger,
            frontier_ordinal=frontier_ordinal,
            retained_parent_bank=retained_parent_bank,
            observation_frames=observation_frames,
            retention_outcomes=retention_outcomes,
            terminal_status=terminal_status,
        )
    )
    if type(value) is not dict or not value:
        raise FounderClosedLoopError(
            "retention lifecycle recorder returned an invalid receipt"
        )
    return {PARENT_PACKAGE_LEDGER_FIELD_V1: value}


def _parent_retention_bindings_v1(
    *,
    manifest: ExpeditionManifestV1,
    bank: RetainedParentBankV1,
    ledger: Mapping[str, object],
    observation_frames: Sequence[ObservationFrameV1],
    frontier_ordinal: int,
) -> list[dict[str, object]]:
    """Build the explicit policy/provenance/hold projection for a Bank revision."""

    try:
        return build_parent_retention_bindings_v1(
            manifest=manifest,
            bank=bank,
            ledger=ledger,
            observation_frames=observation_frames,
            frontier_ordinal=frontier_ordinal,
        )
    except GenealogyIntegrityError as error:
        raise FounderClosedLoopError(
            f"Parent Bank genealogy binding cannot be authenticated: {error}"
        ) from error


def _initialize_or_read_ledger_v1(
    root: Path,
    *,
    manifest: ExpeditionManifestV1,
    policy: ExpeditionPolicyV1,
) -> dict[str, Any]:
    manifest_sha256 = manifest.manifest_sha256().hex
    policy_sha256 = canonical_sha256(canonical_plain_value_v1(policy.binding)).hex
    header = root / "ledger-header.json"
    if root.exists() and not header.exists():
        raise FounderClosedLoopError(
            "existing ledger root lacks its immutable Expedition header"
        )
    if not header.exists():
        initialize_ledger_store_v1(
            root,
            new_expedition_ledger_v1(
                expedition_id=manifest.expedition_id,
                manifest_sha256=manifest_sha256,
                policy_sha256=policy_sha256,
            ),
        )
    ledger = read_ledger_store_v1(root)
    if (
        ledger["expedition_id"] != manifest.expedition_id
        or ledger["manifest_sha256"] != manifest_sha256
        or ledger["policy_sha256"] != policy_sha256
    ):
        raise FounderClosedLoopError(
            "ledger identity differs from the frozen manifest or policy"
        )
    return ledger


def _projection_bundle_v1(
    sources: Sequence[str | Path],
    *,
    manifest: ExpeditionManifestV1,
    projector: ObservationProjectorV1,
    frontier_ordinal: int,
) -> tuple[tuple[ObservationFrameV1, ...], dict[str, object]]:
    source_text = tuple(str(Path(source).expanduser().absolute()) for source in sources)
    if len(source_text) != len(set(source_text)):
        raise FounderClosedLoopError("observation source was duplicated")
    frames: list[ObservationFrameV1] = []
    projection_sha256s: list[str] = []
    for source in source_text:
        projection = projector(source, manifest=manifest)
        projection_sha256s.append(projection.projection_sha256().hex)
        frames.extend(projection.frames)
    frame_sha256s = [frame.observation_frame_sha256().hex for frame in frames]
    if len(frame_sha256s) != len(set(frame_sha256s)):
        raise FounderClosedLoopError(
            "authenticated observation frontier contains a duplicate frame"
        )
    bundle: dict[str, object] = {
        "observation_sources": list(source_text),
        "projection_sha256s": projection_sha256s,
        "observation_frame_sha256s": sorted(frame_sha256s),
        "state_ref_associations": [
            associate_observation_state_ref_v1(
                frame=frame,
                # A frontier has no child result until its action settles.  The
                # authenticated frame is therefore the explicit source result
                # identity for this observation projection.
                source_result_sha256=frame.observation_frame_sha256(),
                source_frontier_ordinal=frontier_ordinal,
                lifecycle_hold_state="OBSERVED_FRONTIER",
            ).to_dict()
            for frame in sorted(frames, key=lambda item: item.observation_frame_sha256().hex)
        ],
    }
    return tuple(frames), bundle


def _seed_for_action_v1(
    manifest: ExpeditionManifestV1,
    *,
    frontier_ordinal: int,
    action_id: str,
    replicate_ordinal: int,
) -> int:
    # Seeds are counter-addressed from immutable Expedition and action identity,
    # never ambient RNG.  The policy-bound action chooses the world count.
    digest = canonical_sha256(
        {
            "schema_version": "eiel_adaptive_expedition_founder_seed_v1",
            "manifest_sha256": manifest.manifest_sha256().to_dict(),
            "outer_search_seed": manifest.outer_search_seed,
            "frontier_ordinal": frontier_ordinal,
            "action_id": action_id,
            "replicate_ordinal": replicate_ordinal,
        }
    )
    return int(digest.hex[:16], 16) & ((1 << 63) - 1)


def _seeds_for_action_v1(
    manifest: ExpeditionManifestV1,
    *,
    frontier_ordinal: int,
    action_id: str,
    world_count: int,
) -> list[int]:
    seeds: list[int] = []
    used: set[int] = set()
    modulus = 1 << 63
    for replicate_ordinal in range(world_count):
        seed = _seed_for_action_v1(
            manifest,
            frontier_ordinal=frontier_ordinal,
            action_id=action_id,
            replicate_ordinal=replicate_ordinal,
        )
        while seed in used:
            seed = (seed + 1) % modulus
        used.add(seed)
        seeds.append(seed)
    return seeds


def _candidate_children_v1(
    *,
    manifest: ExpeditionManifestV1,
    prepared: PreparedDecisionV1,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
    physical_batch_worlds: int,
    builder: Callable[..., GeneratedChildContinuationV1],
) -> tuple[GeneratedChildContinuationV1, ...]:
    children: list[GeneratedChildContinuationV1] = []
    for row in prepared.decision.action_rows:
        if (
            row.action_kind not in _FOUNDER_KINDS
            or row.parent_state_ref_sha256 is not None
            or row.effective_next_absolute_tick != 0
        ):
            continue
        try:
            action_payload = row.action_payload.to_value()
            if not isinstance(action_payload, Mapping):
                raise FounderLoopError("founder action payload is not a mapping")
            world_count = founder_world_count_for_action_v1(action_payload)
            child = builder(
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
        except FounderLoopError:
            # The lower adapter is authoritative for founder legality.  An
            # action that needs history or leaves the frozen search space is
            # not approximated and remains visible in the sealed Decision.
            continue
        if (
            child.parent_state_ref is not None
            or child.effective_next_absolute_tick != 0
        ):
            raise FounderClosedLoopError(
                "founder child builder crossed the no-history boundary"
            )
        children.append(child)
    return tuple(sorted(children, key=lambda item: item.action_id))


def _action_realization_v1(
    *,
    manifest: ExpeditionManifestV1,
    prepared: PreparedDecisionV1,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
    observation_frames: Sequence[ObservationFrameV1],
    retained_parent_bank: RetainedParentBankV1,
    physical_batch_worlds: int,
    dependencies: FounderClosedLoopDependenciesV1,
) -> ClosedLoopActionRealizationV1:
    if dependencies.action_realizer is None:
        children = _candidate_children_v1(
            manifest=manifest,
            prepared=prepared,
            founder_request=founder_request,
            physical_batch_worlds=physical_batch_worlds,
            builder=dependencies.child_builder,
        )
        realization = ClosedLoopActionRealizationV1(
            children=children,
            decision_metadata={
                "proposed_founder_action_ids": [child.action_id for child in children]
            },
        )
    else:
        realization = dependencies.action_realizer(
            manifest=manifest,
            prepared=prepared,
            founder_request=founder_request,
            observation_frames=observation_frames,
            retained_parent_bank=retained_parent_bank,
            physical_batch_worlds=physical_batch_worlds,
            founder_builder=dependencies.child_builder,
        )
    if not isinstance(realization, ClosedLoopActionRealizationV1):
        raise FounderClosedLoopError("action realizer returned an invalid result")
    children = realization.children
    if (
        not isinstance(children, tuple)
        or any(
            not isinstance(child, GeneratedChildContinuationV1) for child in children
        )
        or children != tuple(sorted(children, key=lambda child: child.action_id))
        or len({child.action_id for child in children}) != len(children)
    ):
        raise FounderClosedLoopError(
            "action realizer children must be unique and canonically ordered"
        )
    metadata = canonical_plain_value_v1(realization.decision_metadata)
    if type(metadata) is not dict:
        raise FounderClosedLoopError("action realization metadata must be canonical")
    reserved = {
        "decision_id",
        "decision_sha256",
        "decision",
        "action_ids",
        "rng_cursor_before_sha256",
        "rng_cursor_after_sha256",
        "next_outer_rng_cursor",
        "policy_input_sha256",
        "shadow_mode",
        "atlas_before_sha256",
        "atlas_after_sha256",
        "atlas_after",
        "retained_parent_bank_before_sha256",
        "retained_parent_bank_after_sha256",
        "retained_parent_bank_after",
        "retention_outcomes",
    }
    if reserved & set(metadata):
        raise FounderClosedLoopError(
            "action realization metadata overlaps the durable Decision fields"
        )
    return ClosedLoopActionRealizationV1(
        children=children,
        decision_metadata=metadata,
    )


def _selected_children_v1(
    children: tuple[GeneratedChildContinuationV1, ...],
    selector: ChildSelectorV1 | None,
) -> tuple[GeneratedChildContinuationV1, ...]:
    if selector is None:
        return children
    selected_ids = list(selector(children))
    if any(type(value) is not str or not value for value in selected_ids) or len(
        selected_ids
    ) != len(set(selected_ids)):
        raise FounderClosedLoopError("child selector returned invalid identities")
    by_id = {child.action_id: child for child in children}
    if any(action_id not in by_id for action_id in selected_ids):
        raise FounderClosedLoopError("child selector chose a non-founder action")
    return tuple(by_id[action_id] for action_id in selected_ids)


def _occurrence_classification_v1(
    path: Path,
    *,
    expected_request_sha256: str,
    expected_campaign_id: str,
) -> str:
    """Classify one occurrence under an already sealed action parent.

    A terminal failed attempt cannot be resumed by the Science Request front
    door.  It remains preserved for provenance, while the same immutable
    Expedition action may receive one fresh run occurrence on the next
    recovery pass.  Unknown or incomplete debris is deliberately not treated
    as a retryable failure and is returned as ``UNKNOWN`` for fail-closed
    handling by the parent classifier.
    """

    def _read_object(path: Path) -> Mapping[str, Any] | None:
        if path.is_symlink() or not path.is_file():
            return None
        try:
            value = json.loads(path.read_bytes())
        except (OSError, TypeError, ValueError):
            return None
        return value if isinstance(value, Mapping) else None

    def _authenticated(value: Mapping[str, Any], field: str) -> bool:
        core = dict(value)
        supplied = core.pop(field, None)
        digest = canonical_sha256(core)
        return supplied == digest.hex or supplied == digest.to_dict()

    def _retryable_admission_failure(journal: Mapping[str, Any]) -> bool:
        """Recognize only a pre-execution scheduler admission rejection.

        The occurrence passport is terminal because the outer Science Request
        recorded a failure, but a CRCA admission rejection proves that no
        child execution authority was granted.  Such an occurrence may resume
        its existing journal at the incomplete batch.  Every other failure
        remains terminal and must not be retried by the Expedition loop.
        """

        if journal.get("status") != "FAILED":
            return False
        rows = journal.get("batches")
        current = journal.get("current_batch_index")
        if not isinstance(rows, list) or type(current) is not int:
            return False
        failed = [
            (index, row)
            for index, row in enumerate(rows)
            if isinstance(row, Mapping) and row.get("state") == "FAILED"
        ]
        if len(failed) != 1 or failed[0][0] != current:
            return False
        row = failed[0][1]
        attempt_root_raw = row.get("attempt_root")
        if not isinstance(attempt_root_raw, str) or not attempt_root_raw:
            return False
        attempt_root = path / attempt_root_raw
        try:
            attempt_root = attempt_root.resolve()
            attempt_root.relative_to(path.resolve())
        except (OSError, ValueError):
            return False
        result_path = attempt_root / "execution_result.json"
        if result_path.is_symlink() or not result_path.is_file():
            return False
        try:
            result = _read_object(result_path)
        except Exception:
            return False
        warnings = result.get("warnings") if isinstance(result, Mapping) else None
        authority = (
            result.get("crca_attempt_authority")
            if isinstance(result, Mapping)
            else None
        )
        reason_codes = (
            authority.get("reason_codes")
            if isinstance(authority, Mapping)
            else None
        )
        prefix = "CRCA_SCHEDULER_ATTEMPT_REJECTED:"
        resource_markers = (
            "single-worker execution lock",
            "scheduler process",
            "exclusive_gpu_conflict",
            "resource manager",
            "resource admission",
            "device capacity",
            "start lease",
        )
        non_retryable_markers = (
            "canonical continuation lineage",
            "published terminal",
            "request identity",
            "semantic change",
        )

        def retryable_warning(value: object) -> bool:
            if not isinstance(value, str):
                return False
            lowered = value.lower()
            return (
                prefix in value
                and not any(marker in lowered for marker in non_retryable_markers)
                and any(marker in lowered for marker in resource_markers)
            )

        return bool(
            isinstance(result, Mapping)
            and result.get("schema_version") == "eiel_execution_result_v2"
            and result.get("ok") is False
            and result.get("status") == "precheck_failed"
            and result.get("result_class") == "ATTEMPT_ADMISSION_UNAVAILABLE"
            and isinstance(warnings, list)
            and any(retryable_warning(value) for value in warnings)
            and isinstance(authority, Mapping)
            and authority.get("execution_authority_granted") is False
            and isinstance(reason_codes, list)
            and any(retryable_warning(value) for value in reason_codes)
        )

    passport = _read_object(path / "run_passport.json")
    if passport is None or not _authenticated(passport, "passport_sha256"):
        return "UNKNOWN"
    source_identity = passport.get("source_identity")
    expected_request = {"algorithm": "sha256", "hex": expected_request_sha256}
    if (
        passport.get("run_id") != path.name
        or passport.get("campaign_id") != expected_campaign_id
        or not isinstance(source_identity, Mapping)
        or source_identity.get("request_sha256") != expected_request
    ):
        return "UNKNOWN"
    if passport.get("terminal_status") == "failed":
        failure = _read_object(path / "run_artifact_failure_receipt.json")
        science_failure = _read_object(path / "science_campaign_failure_receipt.json")
        if (
            isinstance(failure, Mapping)
            and _authenticated(failure, "receipt_sha256")
            and failure.get("schema_version") == "eiel_run_artifact_failure_receipt_v1"
            and failure.get("status") == "failed"
            and failure.get("run_id") == path.name
            and failure.get("campaign_id") == expected_campaign_id
            and failure.get("runtime_root") == str(path)
            and isinstance(science_failure, Mapping)
            and _authenticated(science_failure, "receipt_sha256")
            and science_failure.get("schema_version")
            == "eiel_science_request_v1_run_failure_receipt_v1"
            and science_failure.get("status") == "SCIENCE_BATCH_EXECUTION_FAILED"
            and science_failure.get("request_sha256") == expected_request
            and science_failure.get("runtime_root") == str(path)
        ):
            journal = _read_object(path / "run_journal.json")
            if (
                isinstance(journal, Mapping)
                and _authenticated(journal, "journal_sha256")
                and journal.get("schema_version")
                == "eiel_science_request_v1_run_journal_v1"
                and _retryable_admission_failure(journal)
            ):
                return "RESUMABLE"
            return "FAILED"
        return "UNKNOWN"
    if passport.get("terminal_status") in {"active", "completed"}:
        journal = _read_object(path / "run_journal.json")
        if journal is None or not _authenticated(journal, "journal_sha256"):
            return "UNKNOWN"
        journal_identity = journal.get("identity")
        if (
            not isinstance(journal_identity, Mapping)
            or journal_identity.get("campaign_id") != expected_campaign_id
            or journal_identity.get("request_sha256") != expected_request
        ):
            return "UNKNOWN"
        journal_status = journal.get("status")
        if passport.get("terminal_status") == "completed":
            return "RESUMABLE" if journal_status == "COMPLETED" else "UNKNOWN"
        if journal_status not in {
            "RUNNING",
            "WAIT_RESOURCE",
            "WAIT_STORAGE",
            "RESOURCE_WAIT",
            "STORAGE_WAIT",
        }:
            return "UNKNOWN"
        return "RESUMABLE"
    return "UNKNOWN"


def _resume_root_v1(
    action_runtime_parent: Path,
    *,
    expected_request_sha256: str | None = None,
    expected_campaign_id: str | None = None,
) -> str | None:
    if not action_runtime_parent.exists():
        return None
    if action_runtime_parent.is_symlink() or not action_runtime_parent.is_dir():
        raise FounderClosedLoopError("sealed action runtime parent is invalid")
    children = sorted(action_runtime_parent.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_dir() for path in children):
        raise FounderClosedLoopError(
            "sealed action runtime parent contains an unexpected object"
        )
    if expected_request_sha256 is None or expected_campaign_id is None:
        # Non-Science injected child runners used by the generic loop tests do
        # not have a Science Request passport/journal to authenticate.  Keep
        # their historical one-occurrence crash-resume seam bounded; the real
        # Science Request path always supplies both expected identities.
        if len(children) > 1:
            raise FounderClosedLoopError(
                "sealed action runtime parent contains multiple run occurrences"
            )
        return None if not children else str(children[0].resolve())
    classifications = {
        path: _occurrence_classification_v1(
            path,
            expected_request_sha256=expected_request_sha256,
            expected_campaign_id=expected_campaign_id,
        )
        for path in children
    }
    if any(value == "UNKNOWN" for value in classifications.values()):
        raise FounderClosedLoopError(
            "sealed action runtime parent contains an unknown or unauthenticated occurrence"
        )
    resumable = [
        path for path, value in classifications.items() if value == "RESUMABLE"
    ]
    if resumable:
        if len(resumable) == 1:
            return str(resumable[0].resolve())
        # Resource contention can leave more than one terminal admission
        # occurrence under the same action parent.  Prefer the one with the
        # greatest authenticated completed-batch prefix; a tie is ambiguous
        # and remains fail-closed rather than guessing between histories.
        progress: list[tuple[int, str, Path]] = []
        for candidate in resumable:
            try:
                journal_value = json.loads(
                    (candidate / "run_journal.json").read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                journal_value = None
            journal = (
                journal_value if isinstance(journal_value, Mapping) else {}
            )
            rows = journal.get("batches") if isinstance(journal, Mapping) else None
            completed = (
                sum(
                    1
                    for row in rows
                    if isinstance(row, Mapping) and row.get("state") == "COMPLETED"
                )
                if isinstance(rows, list)
                else -1
            )
            progress.append((completed, candidate.name, candidate))
        progress.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if len(progress) > 1 and progress[0][0] == progress[1][0]:
            raise FounderClosedLoopError(
                "sealed action runtime parent contains multiple equally progressed run occurrences"
            )
        return str(progress[0][2].resolve())
    # All known occurrences are terminal failures.  Preserve them.  A
    # scheduler-admission occurrence is returned above as RESUMABLE and is
    # resumed in-place; semantic/terminal failures stay non-retryable.
    return None


def _result_source_v1(result: Mapping[str, Any]) -> str:
    source = result.get("runtime_root")
    if not isinstance(source, (str, Path)) or not str(source):
        raise FounderClosedLoopError(
            "canonical Science Request result lacks its runtime root"
        )
    return str(Path(source).expanduser().absolute())


def _require_child_request_projection_v1(
    projection: object,
    *,
    child: GeneratedChildContinuationV1,
) -> None:
    """Bind one authenticated result projection to its sealed child request."""

    request_sha256 = getattr(projection, "request_sha256", None)
    try:
        expected_request_sha256 = validate_eiel_science_request_v1(
            child.child_request.to_value()
        ).request_sha256()
    except (TypeError, ValueError):
        expected_request_sha256 = None
    # The canonical Science Request adapter reports the normalized request
    # digest.  A direct exact ObservationFrame projection may instead expose
    # the authenticated CanonicalPayload wrapper digest; both are bound to
    # this immutable child and neither permits a different request.
    expected_request_digests = {child.child_request_sha256}
    if expected_request_sha256 is not None:
        expected_request_digests.add(expected_request_sha256)
    if request_sha256 not in expected_request_digests:
        raise FounderClosedLoopError(
            "authenticated child result names a different Science Request"
        )


def _ledger_decisions_and_children_v1(
    ledger: Mapping[str, Any],
) -> tuple[tuple[DecisionV1, ...], tuple[GeneratedChildContinuationV1, ...]]:
    decisions: list[DecisionV1] = []
    children: list[GeneratedChildContinuationV1] = []
    for event in ledger["events"]:
        payload = event["payload"]
        if event["event_type"] == DECISION_PREPARED:
            decisions.append(DecisionV1.from_dict(payload["decision"]))
        elif event["event_type"] == ACTION_DISPATCH_BOUND:
            children.append(
                GeneratedChildContinuationV1.from_dict(payload["continuation"])
            )
    return tuple(decisions), tuple(children)


def _scientific_state_from_ledger_v1(
    ledger: Mapping[str, Any],
    *,
    initial_atlas: EvolutionaryAtlasV1,
    initial_parent_bank: RetainedParentBankV1,
    before_frontier_ordinal: int | None,
) -> tuple[EvolutionaryAtlasV1, RetainedParentBankV1]:
    """Recover and verify state revisions folded into Decision events."""

    atlas = initial_atlas
    bank = initial_parent_bank
    for event in ledger["events"]:
        if event["event_type"] != DECISION_PREPARED:
            continue
        frontier_ordinal = int(event["frontier_ordinal"])
        if (
            before_frontier_ordinal is not None
            and frontier_ordinal >= before_frontier_ordinal
        ):
            break
        payload = event["payload"]
        required = {
            "atlas_before_sha256",
            "atlas_after_sha256",
            "atlas_after",
            "retained_parent_bank_before_sha256",
            "retained_parent_bank_after_sha256",
            "retained_parent_bank_after",
            "retention_outcomes",
        }
        if not required <= set(payload):
            raise FounderClosedLoopError(
                "committed Decision lacks recoverable Atlas/Parent Bank state"
            )
        if (
            payload["atlas_before_sha256"] != atlas.atlas_sha256().hex
            or payload["retained_parent_bank_before_sha256"] != bank.bank_sha256().hex
        ):
            raise FounderClosedLoopError(
                "Decision scientific-state predecessor differs during recovery"
            )
        recovered_atlas = EvolutionaryAtlasV1.from_dict(payload["atlas_after"])
        recovered_bank = RetainedParentBankV1.from_dict(
            payload["retained_parent_bank_after"]
        )
        if (
            payload["atlas_after_sha256"] != recovered_atlas.atlas_sha256().hex
            or payload["retained_parent_bank_after_sha256"]
            != recovered_bank.bank_sha256().hex
            or recovered_atlas.previous_atlas_sha256 != atlas.atlas_sha256()
            or recovered_bank.previous_bank_sha256 != bank.bank_sha256()
        ):
            raise FounderClosedLoopError(
                "Decision scientific-state revision chain differs during recovery"
            )
        atlas = recovered_atlas
        bank = recovered_bank
    return atlas, bank


def _retained_parent_divergence_proofs_v1(
    ledger: Mapping[str, Any],
    *,
    retained_parent_bank: RetainedParentBankV1,
    before_frontier_ordinal: int | None,
) -> dict[str, object]:
    """Recover opaque policy retention proofs for the active Parent Bank.

    The canonical Parent Bank deliberately stores generic StateRef and human
    reason data, not scientific policy semantics.  The policy proposal that
    admitted each entry is already sealed inside the append-only Decision
    event, so replay can recover the policy-owned divergence signature without
    adding another public contract.  Missing, conflicting, or orphaned proofs
    fail closed rather than allowing an unsafe same-niche deduplication.
    """

    proofs: dict[str, object] = {}
    proof_reasons: dict[str, str] = {}
    for event in ledger["events"]:
        if event["event_type"] != DECISION_PREPARED:
            continue
        frontier_ordinal = int(event["frontier_ordinal"])
        if (
            before_frontier_ordinal is not None
            and frontier_ordinal >= before_frontier_ordinal
        ):
            break
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or "decision" not in payload:
            raise FounderClosedLoopError(
                "Decision event lacks a sealed policy proposal for Parent Bank replay"
            )
        try:
            decision = DecisionV1.from_dict(payload["decision"])
            proposal = canonical_plain_value_v1(decision.policy_proposal.to_value())
        except (KeyError, TypeError, ValueError) as error:
            raise FounderClosedLoopError(
                "Decision policy proposal is not recoverable for Parent Bank replay"
            ) from error
        if type(proposal) is not dict:
            raise FounderClosedLoopError("Decision policy proposal is not a mapping")
        delta = proposal.get("retained_parent_bank_delta")
        if type(delta) is not dict:
            raise FounderClosedLoopError(
                "Decision policy proposal lacks a Parent Bank delta"
            )
        retain_entries = delta.get("retain_entries")
        release_ids = delta.get("release_state_ref_ids")
        if not isinstance(retain_entries, Sequence) or isinstance(
            retain_entries, (str, bytes)
        ) or not isinstance(release_ids, Sequence) or isinstance(
            release_ids, (str, bytes)
        ):
            raise FounderClosedLoopError(
                "Decision Parent Bank delta has invalid retention proof fields"
            )
        for raw_state_ref_id in release_ids:
            if type(raw_state_ref_id) is not str or not raw_state_ref_id:
                raise FounderClosedLoopError(
                    "Decision Parent Bank release proof has an invalid StateRef ID"
                )
            if raw_state_ref_id not in proofs:
                raise FounderClosedLoopError(
                    "Decision releases a Parent Bank proof that is not active"
                )
            del proofs[raw_state_ref_id]
            proof_reasons.pop(raw_state_ref_id, None)
        for raw_entry in retain_entries:
            if not isinstance(raw_entry, Mapping):
                raise FounderClosedLoopError(
                    "Decision Parent Bank retain proof is not a mapping"
                )
            state_ref_id = raw_entry.get("state_ref_id")
            signature = raw_entry.get("divergence_signature")
            if type(state_ref_id) is not str or not state_ref_id:
                raise FounderClosedLoopError(
                    "Decision Parent Bank retain proof has an invalid StateRef ID"
                )
            if signature is None:
                raise FounderClosedLoopError(
                    "Decision Parent Bank retain proof lacks divergence signature"
                )
            survival_reason = raw_entry.get("survival_reason")
            if type(survival_reason) is not str or not survival_reason:
                raise FounderClosedLoopError(
                    "Decision Parent Bank retain proof lacks its survival reason"
                )
            normalized_signature = canonical_plain_value_v1(signature)
            prior = proofs.get(state_ref_id)
            if prior is not None and prior != normalized_signature:
                raise FounderClosedLoopError(
                    "Parent Bank StateRef has conflicting divergence proofs"
                )
            proofs[state_ref_id] = normalized_signature
            proof_reasons[state_ref_id] = survival_reason

    active_ids = {
        f"SR-{entry.state_ref.state_ref_sha256().hex}"
        for entry in retained_parent_bank.entries
    }
    if set(proofs) != active_ids:
        raise FounderClosedLoopError(
            "active Retained Parent Bank does not have an exact policy divergence proof"
        )
    bank_reasons = {
        f"SR-{entry.state_ref.state_ref_sha256().hex}": set(entry.reason_codes)
        for entry in retained_parent_bank.entries
    }
    if any(
        proof_reasons.get(state_ref_id) not in bank_reasons[state_ref_id]
        for state_ref_id in active_ids
    ):
        raise FounderClosedLoopError(
            "active Retained Parent Bank reason differs from its policy retention proof"
        )
    return proofs


def _policy_requires_parent_divergence_proofs_v1(
    policy: ExpeditionPolicyV1,
) -> bool:
    profile = canonical_plain_value_v1(policy.profile)
    if type(profile) is not dict:
        return False
    authority = profile.get("policy_input_authority")
    if type(authority) is not dict:
        return False
    paths = authority.get("allowed_paths")
    return isinstance(paths, Sequence) and (
        "/retained_parent_bank/entries/*/divergence_signature" in paths
    )


def _append_terminal_frontier_finalization_v1(
    *,
    root: Path,
    ledger: Mapping[str, Any],
    manifest: ExpeditionManifestV1,
    policy: ExpeditionPolicyV1,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
    atlas: EvolutionaryAtlasV1,
    retained_parent_bank: RetainedParentBankV1,
    logical_capacity: int,
    physical_batch_worlds: int,
    dependencies: FounderClosedLoopDependenciesV1,
    terminal_status: str,
    interruption_hook: InterruptionHookV1 | None = None,
) -> tuple[dict[str, Any], EvolutionaryAtlasV1, RetainedParentBankV1] | None:
    """Fold the last settled child into a terminal, non-dispatching frontier.

    A completed child is itself an authenticated observation.  Historically the
    loop went straight from ``ACTION_SETTLED`` to ``EXPEDITION_TERMINAL`` and
    therefore dropped that observation from the final Atlas/Parent Bank.  This
    helper appends the same deterministic observation/Decision pair used during
    normal operation, but marks the Decision as terminal finalization and binds
    no action IDs.  It never calls a runner and has no execution authority.
    """

    if terminal_status not in {"STOPPED", "COMPLETED"}:
        raise FounderClosedLoopError("terminal finalization status is invalid")
    projection = reconstruct_expedition_frontier_v1(ledger)
    current = projection["current_frontier"]
    if current is None or current["decision"] is None:
        return None
    actions = current["actions"]
    if not actions or any(action["settlement"] is None for action in actions.values()):
        return None
    if current["decision"].get("terminal_frontier_finalization") is True:
        final_atlas, final_bank = _scientific_state_from_ledger_v1(
            ledger,
            initial_atlas=atlas,
            initial_parent_bank=retained_parent_bank,
            before_frontier_ordinal=None,
        )
        return dict(ledger), final_atlas, final_bank

    source_values = tuple(
        str(action["settlement"]["observation_source"])
        for action in actions.values()
    )
    ordinal = int(current["frontier_ordinal"]) + 1
    frames, observation_bundle = _projection_bundle_v1(
        source_values,
        manifest=manifest,
        projector=dependencies.observation_projector,
        frontier_ordinal=ordinal,
    )
    observation_sha256 = canonical_sha256(observation_bundle).hex
    frontier_id = f"frontier-founder-{ordinal:04d}-{observation_sha256[:16]}"
    ledger = append_ledger_store_event_v1(
        root,
        event_type=FRONTIER_OBSERVED,
        frontier_ordinal=ordinal,
        payload={
            "frontier_id": frontier_id,
            "observation_sha256": observation_sha256,
            **observation_bundle,
            "authenticated_only": True,
            "historical_continuation_authority": (
                dependencies.historical_continuation_authority
            ),
            "terminal_frontier_finalization": True,
        },
    )
    previous_frontiers = reconstruct_expedition_frontier_v1(ledger)["frontiers"]
    previous = previous_frontiers[ordinal - 1]
    bounded_history = [
        {
            "frontier_ordinal": row["frontier_ordinal"],
            "decision_sha256": row["decision"]["decision_sha256"],
            "settlement_sha256s": sorted(
                action["settlement"]["child_result_sha256"]
                for action in row["actions"].values()
            ),
        }
        for row in previous_frontiers[:ordinal]
    ]
    current_atlas, current_parent_bank = _scientific_state_from_ledger_v1(
        ledger,
        initial_atlas=atlas,
        initial_parent_bank=retained_parent_bank,
        before_frontier_ordinal=ordinal,
    )
    retained_parent_divergence_signatures = (
        _retained_parent_divergence_proofs_v1(
            ledger,
            retained_parent_bank=current_parent_bank,
            before_frontier_ordinal=ordinal,
        )
        if _policy_requires_parent_divergence_proofs_v1(policy)
        else None
    )
    prepared = dependencies.decision_preparer(
        manifest=manifest,
        policy=policy,
        frontier_id=frontier_id,
        frontier_ordinal=ordinal,
        observation_frames=frames,
        bounded_history=bounded_history,
        atlas=current_atlas,
        retained_parent_bank=current_parent_bank,
        logical_capacity=logical_capacity,
        outer_rng_cursor=int(previous["decision"]["next_outer_rng_cursor"]),
        retained_parent_divergence_signatures=retained_parent_divergence_signatures,
    )
    applied_state = apply_policy_state_deltas_v1(
        manifest=manifest,
        decision=prepared.decision,
        observation_frames=frames,
        atlas=current_atlas,
        retained_parent_bank=current_parent_bank,
    )
    retained_parent_bindings = _parent_retention_bindings_v1(
        manifest=manifest,
        bank=applied_state.retained_parent_bank,
        ledger=ledger,
        observation_frames=frames,
        frontier_ordinal=ordinal,
    )
    retention_lifecycle_metadata = _retention_lifecycle_metadata_v1(
        dependencies=dependencies,
        phase="DECISION",
        manifest=manifest,
        ledger=ledger,
        frontier_ordinal=ordinal,
        retained_parent_bank=applied_state.retained_parent_bank,
        observation_frames=frames,
        retention_outcomes=applied_state.retention_outcomes,
    )
    # Terminal finalization is deliberately non-dispatching.  The policy is
    # still allowed to classify the settled observation and update Atlas/Bank,
    # but its next frontier actions must not be realized: doing so would
    # require constructing a child that can never run after the frozen tick or
    # frontier bound.  Record the policy's suppressed action rows compactly so
    # review/replay can see what was intentionally not dispatched without
    # invoking a child builder or granting execution authority.
    suppressed_actions: list[dict[str, object]] = []
    proposed_founder_action_ids: list[str] = []
    proposed_exact_action_ids: list[str] = []
    bookkeeping_actions: list[dict[str, object]] = []
    for row in prepared.decision.action_rows:
        action_kind = row.action_kind.value
        action_row = {
            "action_id": row.action_id,
            "action_kind": action_kind,
            "effective_next_absolute_tick": row.effective_next_absolute_tick,
            "parent_state_ref_sha256": (
                None
                if row.parent_state_ref_sha256 is None
                else row.parent_state_ref_sha256.to_dict()
            ),
            "reason_codes": list(row.reason_codes),
            "action_payload_sha256": row.action_payload.content_digest().to_dict(),
        }
        suppressed_actions.append(action_row)
        if action_kind in {
            ExperimentActionKindV1.EXPLORE_GLOBAL.value,
            ExperimentActionKindV1.CAUSAL_PROBE.value,
        }:
            proposed_founder_action_ids.append(row.action_id)
        elif action_kind in {
            ExperimentActionKindV1.BRANCH_LOCAL.value,
            ExperimentActionKindV1.REPLICATE.value,
        }:
            proposed_exact_action_ids.append(row.action_id)
        elif action_kind in {
            ExperimentActionKindV1.PARK.value,
            ExperimentActionKindV1.RETIRE.value,
            ExperimentActionKindV1.QUARANTINE.value,
            ExperimentActionKindV1.REQUEST_DETAILED_OBSERVATION.value,
        }:
            bookkeeping_actions.append(
                {
                    **action_row,
                    "outcome": "BOOKKEEPING_ONLY_NO_EXECUTION",
                }
            )
    durable_action_metadata = {
        "action_realization_schema_version": (
            "eiel_adaptive_expedition_terminal_action_suppression_v1"
        ),
        "proposed_founder_action_ids": sorted(proposed_founder_action_ids),
        "proposed_exact_action_ids": sorted(proposed_exact_action_ids),
        "bookkeeping_actions": sorted(
            bookkeeping_actions, key=lambda item: str(item["action_id"])
        ),
        "suppressed_actions": sorted(
            suppressed_actions, key=lambda item: str(item["action_id"])
        ),
        "execution_authority": "NONE_TERMINAL_FINALIZATION_ONLY",
        "no_hidden_reward_authority": True,
    }
    durable_action_metadata["action_realization_sha256"] = canonical_sha256(
        durable_action_metadata
    ).hex
    decision_sha256 = prepared.decision.decision_sha256().hex
    ledger = append_ledger_store_event_v1(
        root,
        event_type=DECISION_PREPARED,
        frontier_ordinal=ordinal,
        payload={
            "decision_id": f"D-{decision_sha256}",
            "decision_sha256": decision_sha256,
            "decision": prepared.decision.to_dict(),
            "action_ids": [],
            **durable_action_metadata,
            "rng_cursor_before_sha256": prepared.decision.outer_rng_before_sha256.hex,
            "rng_cursor_after_sha256": prepared.decision.outer_rng_after_sha256.hex,
            "next_outer_rng_cursor": prepared.next_outer_rng_cursor,
            "policy_input_sha256": prepared.policy_input_sha256.hex,
            "shadow_mode": False,
            "atlas_before_sha256": current_atlas.atlas_sha256().hex,
            "atlas_after_sha256": applied_state.atlas.atlas_sha256().hex,
            "atlas_after": applied_state.atlas.to_dict(),
            "retained_parent_bank_before_sha256": current_parent_bank.bank_sha256().hex,
            "retained_parent_bank_after_sha256": (
                applied_state.retained_parent_bank.bank_sha256().hex
            ),
            "retained_parent_bank_after": applied_state.retained_parent_bank.to_dict(),
            "retention_outcomes": canonical_plain_value_v1(
                applied_state.retention_outcomes
            ),
            "retained_parent_bindings": retained_parent_bindings,
            "retained_parent_bindings_sha256": canonical_sha256(
                retained_parent_bindings
            ).hex,
            "terminal_frontier_finalization": True,
            "terminal_status": terminal_status,
            "action_dispatch_suppressed": True,
            "proposed_action_ids": sorted(
                item["action_id"] for item in suppressed_actions
            ),
            **retention_lifecycle_metadata,
        },
    )
    if interruption_hook is not None:
        interruption_hook("AFTER_TERMINAL_FINALIZATION_DECISION", ordinal, None)
    final_ledger = read_ledger_store_v1(root)
    return final_ledger, applied_state.atlas, applied_state.retained_parent_bank


def run_founder_closed_loop_v1(
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
    dependencies: FounderClosedLoopDependenciesV1 = FounderClosedLoopDependenciesV1(),
    child_selector: ChildSelectorV1 | None = None,
    safe_frontier_stop: SafeFrontierStopV1 | None = None,
    interruption_hook: InterruptionHookV1 | None = None,
) -> FounderClosedLoopResultV1:
    """Run or recover a bounded founder-only loop through canonical execution.

    In shadow mode, every observation frontier is explicit and no child is
    bound or dispatched.  In live mode, later frontiers consume only the
    authenticated terminal outputs of the prior frontier's founder children.
    """

    if physical_batch_worlds is None:
        try:
            physical_batch_worlds = validate_eiel_science_request_v1(
                founder_request
            ).batch_plan().worlds
        except (TypeError, ValueError) as error:
            raise FounderClosedLoopError(
                "founder-basis request has no valid nominal batch projection"
            ) from error
    if type(logical_capacity) is not int or logical_capacity < 0:
        raise FounderClosedLoopError("logical_capacity must be non-negative")
    if logical_capacity > manifest.maximum_actions_per_frontier:
        raise FounderClosedLoopError(
            "logical_capacity exceeds the immutable manifest frontier budget"
        )
    if (
        type(maximum_frontiers) is not int
        or maximum_frontiers <= 0
        or maximum_frontiers > manifest.maximum_frontiers
    ):
        raise FounderClosedLoopError(
            "maximum_frontiers must be positive and within the frozen manifest"
        )
    if type(physical_batch_worlds) is not int or physical_batch_worlds < 1:
        raise FounderClosedLoopError("nominal batch size must be a positive integer")
    if (
        type(dependencies.programme_stage) is not str
        or not dependencies.programme_stage
        or type(dependencies.historical_continuation_authority) is not bool
    ):
        raise FounderClosedLoopError("closed-loop programme identity is invalid")
    if not shadow_mode and runtime_parent is None:
        raise FounderClosedLoopError(
            "live founder recovery requires an explicit bounded runtime parent"
        )
    if not shadow_mode and shadow_observation_frontiers is not None:
        raise FounderClosedLoopError(
            "shadow observation schedule is valid only in explicit shadow mode"
        )

    root = Path(ledger_root).expanduser().absolute()
    ledger = _initialize_or_read_ledger_v1(root, manifest=manifest, policy=policy)
    retained_parent_bank.validate_against_manifest(manifest)
    if atlas.maximum_entries != manifest.maximum_atlas_entries:
        raise FounderClosedLoopError(
            "initial Atlas capacity differs from the frozen manifest budget"
        )
    if (
        atlas.expedition_id != manifest.expedition_id
        or atlas.manifest_sha256 != manifest.manifest_sha256()
        or atlas.descriptor_set_sha256 != manifest.observation_descriptor_set_sha256
    ):
        raise FounderClosedLoopError(
            "initial Atlas does not bind the frozen Expedition manifest"
        )
    runtime_base = (
        None if runtime_parent is None else Path(runtime_parent).expanduser().absolute()
    )
    shadow_schedule = (
        tuple(tuple(frontier) for frontier in shadow_observation_frontiers)
        if shadow_observation_frontiers is not None
        else (tuple(initial_observation_sources),)
    )

    while True:
        projection = reconstruct_expedition_frontier_v1(ledger)
        if projection["terminal"] is not None:
            decisions, children = _ledger_decisions_and_children_v1(ledger)
            final_atlas, final_bank = _scientific_state_from_ledger_v1(
                ledger,
                initial_atlas=atlas,
                initial_parent_bank=retained_parent_bank,
                before_frontier_ordinal=None,
            )
            return FounderClosedLoopResultV1(
                ledger=ledger,
                decisions=decisions,
                children=children,
                atlas=final_atlas,
                retained_parent_bank=final_bank,
                shadow_mode=shadow_mode,
                terminal=True,
            )

        if projection["stop"] is not None:
            current_for_stop = projection["current_frontier"]
            if current_for_stop is not None and any(
                action["settlement"] is None
                for action in current_for_stop["actions"].values()
            ):
                raise FounderClosedLoopError(
                    "safe-frontier stop was committed before all actions settled"
                )
            final_atlas, final_bank = _scientific_state_from_ledger_v1(
                ledger,
                initial_atlas=atlas,
                initial_parent_bank=retained_parent_bank,
                before_frontier_ordinal=None,
            )
            ledger = append_ledger_store_event_v1(
                root,
                event_type=EXPEDITION_TERMINAL,
                frontier_ordinal=(
                    0
                    if current_for_stop is None
                    else int(current_for_stop["frontier_ordinal"])
                ),
                payload={
                    "terminal_status": "STOPPED",
                    "programme_stage": dependencies.programme_stage,
                    "shadow_mode": shadow_mode,
                    "no_claims": True,
                    "historical_continuation_authority": (
                        dependencies.historical_continuation_authority
                    ),
                    "stop_event_sha256": projection["stop"]["event_sha256"],
                    "atlas_sha256": final_atlas.atlas_sha256().hex,
                    "retained_parent_bank_sha256": final_bank.bank_sha256().hex,
                    **_retention_lifecycle_metadata_v1(
                        dependencies=dependencies,
                        phase="TERMINAL",
                        manifest=manifest,
                        ledger=ledger,
                        frontier_ordinal=(
                            0
                            if current_for_stop is None
                            else int(current_for_stop["frontier_ordinal"])
                        ),
                        retained_parent_bank=final_bank,
                        terminal_status="STOPPED",
                    ),
                },
            )
            continue

        current = projection["current_frontier"]
        # A crash may occur after a terminal-finalization Decision is sealed but
        # before its terminal event.  Complete that append-only boundary without
        # dispatching the proposal recorded in the finalization Decision.
        if (
            current is not None
            and current["decision"] is not None
            and current["decision"].get("terminal_frontier_finalization") is True
            and not current["actions"]
        ):
            final_atlas, final_bank = _scientific_state_from_ledger_v1(
                ledger,
                initial_atlas=atlas,
                initial_parent_bank=retained_parent_bank,
                before_frontier_ordinal=None,
            )
            finalization_status = current["decision"].get("terminal_status")
            if finalization_status == "STOPPED":
                ledger = append_ledger_store_event_v1(
                    root,
                    event_type=STOP_REQUESTED,
                    frontier_ordinal=int(current["frontier_ordinal"]),
                    payload={
                        "reason": "TERMINAL_FINALIZATION_RECOVERY",
                        "safe_frontier": True,
                        "terminal_finalization_recovery": True,
                        "atlas_sha256": final_atlas.atlas_sha256().hex,
                        "retained_parent_bank_sha256": final_bank.bank_sha256().hex,
                    },
                )
                continue
            if finalization_status != "COMPLETED":
                raise FounderClosedLoopError(
                    "terminal finalization Decision has an invalid terminal status"
                )
            ledger = append_ledger_store_event_v1(
                root,
                event_type=EXPEDITION_TERMINAL,
                frontier_ordinal=int(current["frontier_ordinal"]),
                payload={
                    "terminal_status": "COMPLETED",
                    "programme_stage": dependencies.programme_stage,
                    "shadow_mode": shadow_mode,
                    "no_claims": True,
                    "historical_continuation_authority": (
                        dependencies.historical_continuation_authority
                    ),
                    "atlas_sha256": final_atlas.atlas_sha256().hex,
                    "retained_parent_bank_sha256": final_bank.bank_sha256().hex,
                    **_retention_lifecycle_metadata_v1(
                        dependencies=dependencies,
                        phase="TERMINAL",
                        manifest=manifest,
                        ledger=ledger,
                        frontier_ordinal=int(current["frontier_ordinal"]),
                        retained_parent_bank=final_bank,
                        terminal_status="COMPLETED",
                    ),
                },
            )
            continue
        decision_before_dispatch = None if current is None else current["decision"]
        at_unbound_safe_boundary = (
            current is None
            or decision_before_dispatch is None
            or (bool(decision_before_dispatch["action_ids"]) and not current["actions"])
        )
        if safe_frontier_stop is not None and at_unbound_safe_boundary:
            safe_ordinal = 0 if current is None else int(current["frontier_ordinal"])
            stop_payload = safe_frontier_stop(safe_ordinal, ledger)
            if stop_payload is not None:
                if (
                    not isinstance(stop_payload, Mapping)
                    or type(stop_payload.get("reason")) is not str
                    or not stop_payload["reason"]
                ):
                    raise FounderClosedLoopError(
                        "safe-frontier stop payload requires a non-empty reason"
                    )
                final_atlas, final_bank = _scientific_state_from_ledger_v1(
                    ledger,
                    initial_atlas=atlas,
                    initial_parent_bank=retained_parent_bank,
                    before_frontier_ordinal=None,
                )
                ledger = append_ledger_store_event_v1(
                    root,
                    event_type=STOP_REQUESTED,
                    frontier_ordinal=safe_ordinal,
                    payload={
                        **dict(stop_payload),
                        "safe_frontier": True,
                        "before_dispatch": True,
                        "atlas_sha256": final_atlas.atlas_sha256().hex,
                        "retained_parent_bank_sha256": final_bank.bank_sha256().hex,
                    },
                )
                continue
        if current is None:
            ordinal = 0
            sources: tuple[str | Path, ...] = (
                shadow_schedule[0]
                if shadow_mode
                else tuple(initial_observation_sources)
            )
        elif current["decision"] is None:
            ordinal = int(current["frontier_ordinal"])
            sources = tuple(current["observation"]["observation_sources"])
        elif any(
            action_id not in current["actions"]
            or current["actions"][action_id]["settlement"] is None
            for action_id in current["decision"]["action_ids"]
        ):
            ordinal = int(current["frontier_ordinal"])
            sources = tuple(current["observation"]["observation_sources"])
        else:
            completed_ordinal = int(current["frontier_ordinal"])
            if safe_frontier_stop is not None:
                stop_payload = safe_frontier_stop(completed_ordinal, ledger)
                if stop_payload is not None:
                    if (
                        not isinstance(stop_payload, Mapping)
                        or type(stop_payload.get("reason")) is not str
                        or not stop_payload["reason"]
                    ):
                        raise FounderClosedLoopError(
                            "safe-frontier stop payload requires a non-empty reason"
                        )
                    finalized = _append_terminal_frontier_finalization_v1(
                        root=root,
                        ledger=ledger,
                        manifest=manifest,
                        policy=policy,
                        founder_request=founder_request,
                        atlas=atlas,
                        retained_parent_bank=retained_parent_bank,
                        logical_capacity=logical_capacity,
                        physical_batch_worlds=physical_batch_worlds,
                        dependencies=dependencies,
                        terminal_status="STOPPED",
                        interruption_hook=interruption_hook,
                    )
                    if finalized is not None:
                        ledger, final_atlas, final_bank = finalized
                        finalized_frontier = reconstruct_expedition_frontier_v1(
                            ledger
                        )["current_frontier"]
                        assert finalized_frontier is not None
                        completed_ordinal = int(
                            finalized_frontier["frontier_ordinal"]
                        )
                    else:
                        final_atlas, final_bank = _scientific_state_from_ledger_v1(
                            ledger,
                            initial_atlas=atlas,
                            initial_parent_bank=retained_parent_bank,
                            before_frontier_ordinal=None,
                        )
                    ledger = append_ledger_store_event_v1(
                        root,
                        event_type=STOP_REQUESTED,
                        frontier_ordinal=completed_ordinal,
                        payload={
                            **dict(stop_payload),
                            "safe_frontier": True,
                            "atlas_sha256": final_atlas.atlas_sha256().hex,
                            "retained_parent_bank_sha256": final_bank.bank_sha256().hex,
                        },
                    )
                    continue
            if completed_ordinal + 1 >= maximum_frontiers:
                finalized = _append_terminal_frontier_finalization_v1(
                    root=root,
                    ledger=ledger,
                    manifest=manifest,
                    policy=policy,
                    founder_request=founder_request,
                    atlas=atlas,
                    retained_parent_bank=retained_parent_bank,
                    logical_capacity=logical_capacity,
                    physical_batch_worlds=physical_batch_worlds,
                    dependencies=dependencies,
                    terminal_status="COMPLETED",
                    interruption_hook=interruption_hook,
                )
                if finalized is not None:
                    ledger, final_atlas, final_bank = finalized
                    terminal_frontier = reconstruct_expedition_frontier_v1(ledger)[
                        "current_frontier"
                    ]
                    assert terminal_frontier is not None
                    completed_ordinal = int(terminal_frontier["frontier_ordinal"])
                else:
                    final_atlas, final_bank = _scientific_state_from_ledger_v1(
                        ledger,
                        initial_atlas=atlas,
                        initial_parent_bank=retained_parent_bank,
                        before_frontier_ordinal=None,
                    )
                ledger = append_ledger_store_event_v1(
                    root,
                    event_type=EXPEDITION_TERMINAL,
                    frontier_ordinal=completed_ordinal,
                    payload={
                        "terminal_status": "COMPLETED",
                        "programme_stage": dependencies.programme_stage,
                        "shadow_mode": shadow_mode,
                        "no_claims": True,
                        "historical_continuation_authority": (
                            dependencies.historical_continuation_authority
                        ),
                        "atlas_sha256": final_atlas.atlas_sha256().hex,
                        "retained_parent_bank_sha256": final_bank.bank_sha256().hex,
                        **_retention_lifecycle_metadata_v1(
                            dependencies=dependencies,
                            phase="TERMINAL",
                            manifest=manifest,
                            ledger=ledger,
                            frontier_ordinal=completed_ordinal,
                            retained_parent_bank=final_bank,
                            terminal_status="COMPLETED",
                        ),
                    },
                )
                continue
            ordinal = completed_ordinal + 1
            if shadow_mode:
                if ordinal >= len(shadow_schedule):
                    raise FounderClosedLoopError(
                        "shadow mode lacks the next explicit observation frontier"
                    )
                sources = shadow_schedule[ordinal]
            else:
                sources = tuple(
                    action["settlement"]["observation_source"]
                    for action in current["actions"].values()
                )

        frames, observation_bundle = _projection_bundle_v1(
            sources,
            manifest=manifest,
            projector=dependencies.observation_projector,
            frontier_ordinal=ordinal,
        )
        observation_sha256 = canonical_sha256(observation_bundle).hex
        frontier_id = f"frontier-founder-{ordinal:04d}-{observation_sha256[:16]}"

        if current is None or int(current["frontier_ordinal"]) != ordinal:
            ledger = append_ledger_store_event_v1(
                root,
                event_type=FRONTIER_OBSERVED,
                frontier_ordinal=ordinal,
                payload={
                    "frontier_id": frontier_id,
                    "observation_sha256": observation_sha256,
                    **observation_bundle,
                    "authenticated_only": True,
                    "historical_continuation_authority": (
                        dependencies.historical_continuation_authority
                    ),
                },
            )
            if interruption_hook is not None:
                interruption_hook("AFTER_FRONTIER_OBSERVED", ordinal, None)
            current = reconstruct_expedition_frontier_v1(ledger)["current_frontier"]
        else:
            observed = current["observation"]
            if (
                observed["frontier_id"] != frontier_id
                or observed["observation_sha256"] != observation_sha256
                or observed["observation_sources"]
                != observation_bundle["observation_sources"]
                or observed["projection_sha256s"]
                != observation_bundle["projection_sha256s"]
                or observed["observation_frame_sha256s"]
                != observation_bundle["observation_frame_sha256s"]
                or observed.get("state_ref_associations", [])
                != observation_bundle["state_ref_associations"]
            ):
                raise FounderClosedLoopError(
                    "recovered observation frontier differs from its ledger binding"
                )

        assert current is not None
        decision_payload = current["decision"]
        if ordinal == 0:
            rng_cursor = 0
            bounded_history: list[Mapping[str, object]] = []
        else:
            previous = reconstruct_expedition_frontier_v1(ledger)["frontiers"][
                ordinal - 1
            ]
            rng_cursor = int(previous["decision"]["next_outer_rng_cursor"])
            bounded_history = [
                {
                    "frontier_ordinal": row["frontier_ordinal"],
                    "decision_sha256": row["decision"]["decision_sha256"],
                    "settlement_sha256s": sorted(
                        action["settlement"]["child_result_sha256"]
                        for action in row["actions"].values()
                    ),
                }
                for row in reconstruct_expedition_frontier_v1(ledger)["frontiers"][
                    :ordinal
                ]
            ]
        current_atlas, current_parent_bank = _scientific_state_from_ledger_v1(
            ledger,
            initial_atlas=atlas,
            initial_parent_bank=retained_parent_bank,
            before_frontier_ordinal=ordinal,
        )
        retained_parent_divergence_signatures = (
            _retained_parent_divergence_proofs_v1(
                ledger,
                retained_parent_bank=current_parent_bank,
                before_frontier_ordinal=ordinal,
            )
            if _policy_requires_parent_divergence_proofs_v1(policy)
            else None
        )
        prepared = dependencies.decision_preparer(
            manifest=manifest,
            policy=policy,
            frontier_id=frontier_id,
            frontier_ordinal=ordinal,
            observation_frames=frames,
            bounded_history=bounded_history,
            atlas=current_atlas,
            retained_parent_bank=current_parent_bank,
            logical_capacity=logical_capacity,
            outer_rng_cursor=rng_cursor,
            retained_parent_divergence_signatures=retained_parent_divergence_signatures,
        )
        applied_state = apply_policy_state_deltas_v1(
            manifest=manifest,
            decision=prepared.decision,
            observation_frames=frames,
            atlas=current_atlas,
            retained_parent_bank=current_parent_bank,
        )
        retained_parent_bindings = _parent_retention_bindings_v1(
            manifest=manifest,
            bank=applied_state.retained_parent_bank,
            ledger=ledger,
            observation_frames=frames,
            frontier_ordinal=ordinal,
        )
        retention_lifecycle_metadata = _retention_lifecycle_metadata_v1(
            dependencies=dependencies,
            phase="DECISION",
            manifest=manifest,
            ledger=ledger,
            frontier_ordinal=ordinal,
            retained_parent_bank=applied_state.retained_parent_bank,
            observation_frames=frames,
            retention_outcomes=applied_state.retention_outcomes,
        )
        realization = _action_realization_v1(
            manifest=manifest,
            prepared=prepared,
            founder_request=founder_request,
            observation_frames=frames,
            retained_parent_bank=current_parent_bank,
            physical_batch_worlds=physical_batch_worlds,
            dependencies=dependencies,
        )
        candidates = realization.children
        durable_action_metadata = dict(realization.decision_metadata)
        if dependencies.action_realizer is not None:
            durable_action_metadata["action_realization_sha256"] = canonical_sha256(
                {
                    "children": [child.to_dict() for child in candidates],
                    "decision_metadata": realization.decision_metadata,
                }
            ).hex
        selected = (
            ()
            if shadow_mode
            or (
                decision_payload is not None
                and decision_payload.get("terminal_frontier_finalization") is True
            )
            else _selected_children_v1(candidates, child_selector)
        )
        declared_action_ids = [child.action_id for child in selected]

        if decision_payload is None:
            decision_sha256 = prepared.decision.decision_sha256().hex
            ledger = append_ledger_store_event_v1(
                root,
                event_type=DECISION_PREPARED,
                frontier_ordinal=ordinal,
                payload={
                    "decision_id": f"D-{decision_sha256}",
                    "decision_sha256": decision_sha256,
                    "decision": prepared.decision.to_dict(),
                    "action_ids": declared_action_ids,
                    **durable_action_metadata,
                    "rng_cursor_before_sha256": prepared.decision.outer_rng_before_sha256.hex,
                    "rng_cursor_after_sha256": prepared.decision.outer_rng_after_sha256.hex,
                    "next_outer_rng_cursor": prepared.next_outer_rng_cursor,
                    "policy_input_sha256": prepared.policy_input_sha256.hex,
                    "shadow_mode": shadow_mode,
                    "atlas_before_sha256": current_atlas.atlas_sha256().hex,
                    "atlas_after_sha256": applied_state.atlas.atlas_sha256().hex,
                    "atlas_after": applied_state.atlas.to_dict(),
                    "retained_parent_bank_before_sha256": (
                        current_parent_bank.bank_sha256().hex
                    ),
                    "retained_parent_bank_after_sha256": (
                        applied_state.retained_parent_bank.bank_sha256().hex
                    ),
                    "retained_parent_bank_after": (
                        applied_state.retained_parent_bank.to_dict()
                    ),
                    "retention_outcomes": canonical_plain_value_v1(
                        applied_state.retention_outcomes
                    ),
                    "retained_parent_bindings": retained_parent_bindings,
                    "retained_parent_bindings_sha256": canonical_sha256(
                        retained_parent_bindings
                    ).hex,
                    **retention_lifecycle_metadata,
                },
            )
            if interruption_hook is not None:
                interruption_hook("AFTER_DECISION_PREPARED", ordinal, None)
            current = reconstruct_expedition_frontier_v1(ledger)["current_frontier"]
            assert current is not None
            decision_payload = current["decision"]
        else:
            if (
                decision_payload["decision"] != prepared.decision.to_dict()
                or decision_payload["decision_sha256"]
                != prepared.decision.decision_sha256().hex
                or decision_payload["policy_input_sha256"]
                != prepared.policy_input_sha256.hex
                or decision_payload["next_outer_rng_cursor"]
                != prepared.next_outer_rng_cursor
                or decision_payload["shadow_mode"] is not shadow_mode
                or decision_payload["atlas_before_sha256"]
                != current_atlas.atlas_sha256().hex
                or decision_payload["atlas_after_sha256"]
                != applied_state.atlas.atlas_sha256().hex
                or decision_payload["atlas_after"] != applied_state.atlas.to_dict()
                or decision_payload["retained_parent_bank_before_sha256"]
                != current_parent_bank.bank_sha256().hex
                or decision_payload["retained_parent_bank_after_sha256"]
                != applied_state.retained_parent_bank.bank_sha256().hex
                or decision_payload["retained_parent_bank_after"]
                != applied_state.retained_parent_bank.to_dict()
                or decision_payload["retention_outcomes"]
                != canonical_plain_value_v1(applied_state.retention_outcomes)
                or decision_payload.get("retained_parent_bindings", [])
                != retained_parent_bindings
                or decision_payload.get("retained_parent_bindings_sha256")
                != canonical_sha256(retained_parent_bindings).hex
                or any(
                    decision_payload.get(key) != value
                    for key, value in retention_lifecycle_metadata.items()
                )
                or any(
                    decision_payload.get(key) != value
                    for key, value in durable_action_metadata.items()
                )
            ):
                raise FounderClosedLoopError(
                    "replayed decision differs from the committed Decision"
                )
            selected_by_id = {child.action_id: child for child in selected}
            if decision_payload["action_ids"] != declared_action_ids:
                raise FounderClosedLoopError(
                    "replayed founder child selection differs from the ledger"
                )
            selected = tuple(
                selected_by_id[action_id]
                for action_id in decision_payload["action_ids"]
            )

        if dependencies.retention_lifecycle_boundary is not None:
            boundary_value = canonical_plain_value_v1(
                dependencies.retention_lifecycle_boundary(
                    manifest=manifest,
                    ledger=ledger,
                    frontier_ordinal=ordinal,
                    decision_payload=decision_payload,
                    selected_children=selected,
                )
            )
            if type(boundary_value) is not dict:
                raise FounderClosedLoopError(
                    "retention lifecycle boundary returned an invalid receipt"
                )

        if shadow_mode:
            # Empty declared action IDs make a shadow frontier settled without
            # fabricating execution or settlement receipts.
            continue

        current = reconstruct_expedition_frontier_v1(ledger)["current_frontier"]
        assert current is not None
        action_states = current["actions"]
        for child in selected:
            action = action_states.get(child.action_id)
            child_binding_sha256 = canonical_sha256(child.to_dict()).hex
            assert runtime_base is not None
            action_runtime_parent = (
                runtime_base
                / _safe_component(manifest.expedition_id)
                / f"frontier-{ordinal:04d}"
                / f"action-{child_binding_sha256[:24]}"
            )
            if action is None:
                historical_child = (
                    child.realization_kind is ChildRealizationKindV1.EXACT_STATE_REQUEST
                )
                ledger = append_ledger_store_event_v1(
                    root,
                    event_type=ACTION_DISPATCH_BOUND,
                    frontier_ordinal=ordinal,
                    payload={
                        "decision_id": decision_payload["decision_id"],
                        "action_id": child.action_id,
                        "child_request_sha256": child.child_request_sha256.hex,
                        "child_binding_sha256": child_binding_sha256,
                        "continuation": child.to_dict(),
                        "action_runtime_parent": str(action_runtime_parent),
                        "child_next_absolute_tick": (
                            child.effective_next_absolute_tick
                        ),
                        "historical_continuation_authority": historical_child,
                    },
                )
                if interruption_hook is not None:
                    interruption_hook(
                        "AFTER_ACTION_DISPATCH_BOUND", ordinal, child.action_id
                    )
                action = reconstruct_expedition_frontier_v1(ledger)["current_frontier"][
                    "actions"
                ][child.action_id]
            else:
                binding = action["binding"]
                if (
                    binding["continuation"] != child.to_dict()
                    or binding["child_binding_sha256"] != child_binding_sha256
                    or binding["action_runtime_parent"] != str(action_runtime_parent)
                ):
                    raise FounderClosedLoopError(
                        "recovered child differs from its dispatch binding"
                    )
            if action["settlement"] is not None:
                continue

            try:
                child_request = validate_eiel_science_request_v1(
                    child.child_request.to_value()
                )
            except (TypeError, ValueError):
                # The generic loop's injected child runners may use a private
                # CanonicalPayload test contract.  They do not create Science
                # Request lifecycle occurrences; retain only the bounded
                # legacy one-occurrence resume behavior for that seam.
                resume_root = _resume_root_v1(action_runtime_parent)
            else:
                child_request_value = child_request.to_dict()
                resume_root = _resume_root_v1(
                    action_runtime_parent,
                    expected_request_sha256=child_request.request_sha256().hex,
                    expected_campaign_id=str(child_request_value["campaign_id"]),
                )
            result = dependencies.child_runner(
                child,
                repo_root=repo_root,
                materialization_parent=materialization_parent,
                runtime_parent=str(action_runtime_parent),
                resume_root=resume_root,
            )
            if interruption_hook is not None:
                interruption_hook("AFTER_ACTION_EXECUTION", ordinal, child.action_id)
            source = _result_source_v1(result)
            settled_projection = dependencies.observation_projector(
                source, manifest=manifest
            )
            _require_child_request_projection_v1(
                settled_projection,
                child=child,
            )
            candidate_receipt = result.get(PARENT_CANDIDATE_SETTLEMENT_FIELD_V1)
            settlement_metadata: dict[str, object] = {}
            if candidate_receipt is not None:
                plain_candidate = canonical_plain_value_v1(candidate_receipt)
                if type(plain_candidate) is not dict:
                    raise FounderClosedLoopError(
                        "exact parent candidate settlement receipt is invalid"
                    )
                settlement_metadata[PARENT_CANDIDATE_SETTLEMENT_FIELD_V1] = (
                    plain_candidate
                )
            parent_state_ref_sha256 = (
                None
                if child.parent_state_ref is None
                else child.parent_state_ref.state_ref_sha256()
            )
            try:
                state_ref_associations = [
                    associate_observation_state_ref_v1(
                        frame=frame,
                        source_result_sha256=settled_projection.projection_sha256(),
                        source_frontier_ordinal=ordinal,
                        child_run_id=str(child.child_run_id),
                        child_request_sha256=child.child_request_sha256,
                        parent_state_ref_sha256=parent_state_ref_sha256,
                        lifecycle_hold_state=(
                            "R1_ACTIVE_RECOVERY"
                            if frame.source_state_ref.supports_exact_continuation
                            else "OBSERVED_SETTLEMENT"
                        ),
                    ).to_dict()
                    for frame in settled_projection.frames
                ]
            except GenealogyIntegrityError as error:
                raise FounderClosedLoopError(
                    f"settled child StateRef association is invalid: {error}"
                ) from error
            ledger = append_ledger_store_event_v1(
                root,
                event_type=ACTION_SETTLED,
                frontier_ordinal=ordinal,
                payload={
                    "action_id": child.action_id,
                    "child_binding_sha256": child_binding_sha256,
                    "authenticated_child_request_sha256": (
                        child.child_request_sha256.hex
                    ),
                    "outcome": "COMPLETED_AUTHENTICATED",
                    "child_result_sha256": settled_projection.projection_sha256().hex,
                    "observation_source": source,
                    "observation_frame_sha256s": [
                        frame.observation_frame_sha256().hex
                        for frame in settled_projection.frames
                    ],
                    "state_ref_associations": state_ref_associations,
                    "historical_continuation_authority": (
                        child.realization_kind
                        is ChildRealizationKindV1.EXACT_STATE_REQUEST
                    ),
                    **settlement_metadata,
                },
            )
            if interruption_hook is not None:
                interruption_hook("AFTER_ACTION_SETTLED", ordinal, child.action_id)


__all__ = (
    "ChildSelectorV1",
    "ClosedLoopActionRealizationV1",
    "ClosedLoopActionRealizerV1",
    "FounderChildRunnerV1",
    "FounderClosedLoopDependenciesV1",
    "FounderClosedLoopError",
    "FounderClosedLoopResultV1",
    "InterruptionHookV1",
    "ObservationProjectorV1",
    "SafeFrontierStopV1",
    "run_founder_closed_loop_v1",
)
