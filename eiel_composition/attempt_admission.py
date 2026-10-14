"""CRCA-G3 Attempt admission, continuation fencing, and append-only receipts.

This is an opt-in occurrence shell.  It validates frozen CRCA contracts and
writes content-addressed receipts, but it grants no scheduler/resource lease,
launches no process, and is not called by the current Launchpad or QRR paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from emergence_contracts import (
    AttemptAdmissionDispositionV1,
    AttemptAdmissionRequestV1,
    AttemptFrontiersV1,
    AttemptRetryBasisCodeV1,
    AttemptRetryBasisV1,
    AttemptStartV1,
    AttemptTerminalStatusV1,
    AttemptTerminalV1,
    CanonicalContinuationLeaseV1,
    ConcretizedExecutionV1,
    ContentDigest,
    ContinuationLeaseEventCodeV1,
    ContinuationLeaseEventV1,
    ExactImplementationConformanceV1,
    ExecutionImplementationV1,
    RestoreCheckpointAuthorizationV1,
    RunOccurrenceV1,
    SemanticRunSpecV1,
    canonical_json_bytes,
    canonical_sha256,
)


_RECEIPT_KINDS = frozenset(
    {
        "run",
        "admission-disposition",
        "retry-basis",
        "implementation-conformance",
        "restore-authorization",
        "continuation-lease",
        "lease-event",
        "attempt-start",
        "attempt-terminal",
    }
)
_LEASE_TRANSITION_EVENTS = frozenset(
    {
        ContinuationLeaseEventCodeV1.RENEWED,
        ContinuationLeaseEventCodeV1.REASSIGNED,
        ContinuationLeaseEventCodeV1.RECLAIMED,
    }
)


class AttemptAdmissionError(ValueError):
    """A proposed occurrence violates frozen Run/Attempt or fencing law."""


def _fail(message: str) -> None:
    raise AttemptAdmissionError(message)


def _utc(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        _fail(f"{field} must use canonical second-resolution UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise AttemptAdmissionError(f"{field} is invalid") from exc


@dataclass(frozen=True, slots=True)
class AttemptRecordV1:
    start: AttemptStartV1
    concretized_execution: ConcretizedExecutionV1
    terminal: AttemptTerminalV1 | None

    def __post_init__(self) -> None:
        if self.start.concretized_execution_digest != (
            self.concretized_execution.concretized_execution_digest()
        ):
            _fail("prior Attempt start differs from its concretized execution")
        if self.terminal is not None:
            self.terminal.validate_transition(self.start)


def validate_frontier_progression_v1(
    previous: AttemptFrontiersV1, candidate: AttemptFrontiersV1
) -> None:
    """Require every absolute frontier to be monotonic."""

    if not isinstance(previous, AttemptFrontiersV1) or not isinstance(
        candidate, AttemptFrontiersV1
    ):
        _fail("frontier progression requires AttemptFrontiersV1 values")
    for field_name in (
        "last_dispatched_next_tick",
        "last_computed_next_tick",
        "durable_checkpoint_next_tick",
        "published_artifact_next_tick",
    ):
        if getattr(candidate, field_name) < getattr(previous, field_name):
            _fail(f"absolute frontier moved backwards: {field_name}")


def validate_terminal_once_v1(
    start: AttemptStartV1,
    existing: AttemptTerminalV1 | None,
    candidate: AttemptTerminalV1,
    run: RunOccurrenceV1,
    semantic_spec: SemanticRunSpecV1,
) -> AttemptTerminalV1:
    """Accept one terminal transition and idempotent byte-identical replay."""

    if not isinstance(semantic_spec, SemanticRunSpecV1):
        _fail("terminal validation requires a SemanticRunSpecV1")
    if start.run_id != run.run_id:
        _fail("terminal validation Run differs from the Attempt")
    if semantic_spec.semantic_spec_digest() != run.semantic_spec_digest:
        _fail("terminal validation semantic specification differs from the Run")
    candidate.validate_transition(start)
    target = semantic_spec.absolute_target_next_tick
    if any(
        frontier > target
        for frontier in (
            candidate.final_frontiers.last_dispatched_next_tick,
            candidate.final_frontiers.last_computed_next_tick,
            candidate.final_frontiers.durable_checkpoint_next_tick,
            candidate.final_frontiers.published_artifact_next_tick,
        )
    ):
        _fail("terminal frontier exceeds the absolute Run target")
    if candidate.status is AttemptTerminalStatusV1.COMPLETED and (
        candidate.final_frontiers.last_dispatched_next_tick != target
        or candidate.final_frontiers.last_computed_next_tick != target
    ):
        _fail("completed Attempt has not reached the absolute Run target")
    if existing is None:
        return candidate
    existing.validate_transition(start)
    if existing.terminal_digest() != candidate.terminal_digest():
        _fail("Attempt already has a different immutable terminal transition")
    return existing


def _validate_retry_basis_names(basis: AttemptRetryBasisV1) -> None:
    names = {item.name for item in basis.basis_evidence_digests}
    required = {
        AttemptRetryBasisCodeV1.CHANGED_CONDITION: "changed_condition",
        AttemptRetryBasisCodeV1.NEW_INFORMATION: "new_information",
        AttemptRetryBasisCodeV1.EXACT_RESTORE_AFTER_FAILURE: "restore_checkpoint",
        AttemptRetryBasisCodeV1.QUALIFICATION_INTEGRATION_ESCAPE: (
            "qualification_integration_escape"
        ),
    }.get(basis.code)
    if required is not None and required not in names:
        _fail(f"{basis.code.value} lacks required {required!r} evidence")


def _validate_prior_history(
    run: RunOccurrenceV1,
    semantic_spec: SemanticRunSpecV1,
    records: Sequence[AttemptRecordV1],
) -> tuple[AttemptRecordV1, ...]:
    prior = tuple(records)
    ordinals: list[int] = []
    ids: set[str] = set()
    for record in prior:
        if not isinstance(record, AttemptRecordV1):
            _fail("prior Attempt history contains an invalid record")
        if record.start.run_id != run.run_id:
            _fail("prior Attempt history crosses Run identity")
        if record.terminal is not None:
            validate_terminal_once_v1(
                record.start,
                None,
                record.terminal,
                run,
                semantic_spec,
            )
        ordinal = record.start.monotonic_ordinal
        if ordinal in ordinals or str(record.start.attempt_id) in ids:
            _fail("prior Attempt history repeats an identity")
        ordinals.append(ordinal)
        ids.add(str(record.start.attempt_id))
    if ordinals != list(range(1, len(prior) + 1)):
        _fail("prior Attempt ordinals are not a contiguous append-only sequence")
    return prior


def _validate_restore(
    *,
    run: RunOccurrenceV1,
    semantic_spec: SemanticRunSpecV1,
    start: AttemptStartV1,
    authorization: RestoreCheckpointAuthorizationV1 | None,
    prior: tuple[AttemptRecordV1, ...],
    external_source_record: AttemptRecordV1 | None,
) -> None:
    if start.restore_checkpoint_digest is None:
        if authorization is not None:
            _fail("restore authorization exists for a non-restored Attempt")
        return
    if authorization is None:
        _fail("restored Attempt lacks sealed checkpoint authorization")
    if start.restore_checkpoint_digest != authorization.checkpoint_digest:
        _fail("Attempt restore checkpoint differs from sealed authorization")
    same_run_restore = authorization.run_id == run.run_id
    child_branch_restore = (
        run.parent_checkpoint_digest == authorization.checkpoint_digest
        and authorization.run_id != run.run_id
    )
    if not same_run_restore and not child_branch_restore:
        _fail("restore authorization is neither same-Run nor declared child branch")
    if same_run_restore:
        source = next(
            (
                record
                for record in prior
                if record.start.attempt_id == authorization.source_attempt_id
            ),
            None,
        )
    else:
        source = external_source_record
        if source is None or source.start.run_id != authorization.run_id:
            _fail("child-Run restore lacks its parent source Attempt record")
    if source is None or source.terminal is None:
        _fail("restore source has no immutable terminal receipt")
    if source.start.attempt_id != authorization.source_attempt_id:
        _fail("restore source Attempt identity differs")
    if source.terminal.terminal_digest() != (
        authorization.source_attempt_terminal_digest
    ):
        _fail("restore source terminal digest differs")
    if (
        authorization.absolute_next_tick
        > source.terminal.final_frontiers.durable_checkpoint_next_tick
    ):
        _fail("restore exceeds the source durable checkpoint frontier")
    tick = authorization.absolute_next_tick
    if tick not in semantic_spec.trajectory_checkpoint_ticks:
        _fail("restore tick is not an authorized trajectory checkpoint")
    initial = start.initial_frontiers
    if (
        initial.last_dispatched_next_tick != tick
        or initial.last_computed_next_tick != tick
        or initial.durable_checkpoint_next_tick != tick
        or initial.published_artifact_next_tick > tick
    ):
        _fail("restored Attempt frontiers differ from the absolute checkpoint")


def validate_attempt_start_v1(
    *,
    run: RunOccurrenceV1,
    semantic_spec: SemanticRunSpecV1,
    concretized_execution: ConcretizedExecutionV1,
    implementation: ExecutionImplementationV1,
    admission_request: AttemptAdmissionRequestV1,
    disposition: AttemptAdmissionDispositionV1,
    retry_basis: AttemptRetryBasisV1,
    start: AttemptStartV1,
    continuation_lease: CanonicalContinuationLeaseV1,
    continuation_lease_ledger: ContinuationLeaseLedgerV1,
    execution_side_effect_at_utc: str,
    prior_records: Sequence[AttemptRecordV1] = (),
    restore_authorization: RestoreCheckpointAuthorizationV1 | None = None,
    restore_source_record: AttemptRecordV1 | None = None,
    implementation_conformance: ExactImplementationConformanceV1 | None = None,
) -> dict[str, Any]:
    """Validate one exact Attempt start without granting or launching it."""

    prior = _validate_prior_history(run, semantic_spec, prior_records)
    if semantic_spec.semantic_spec_digest() != run.semantic_spec_digest:
        _fail("semantic specification differs from the Run occurrence")
    if concretized_execution.semantic_spec_digest != run.semantic_spec_digest:
        _fail("semantic change cannot be admitted as another Attempt of one Run")
    if (
        concretized_execution.implementation_digest
        != implementation.implementation_digest()
    ):
        _fail("concretized execution implementation layer differs")
    if concretized_execution.attempt_admission_request_digest != (
        admission_request.attempt_admission_request_digest()
    ):
        _fail("concretized execution admission-request layer differs")
    if start.run_id != run.run_id:
        _fail("Attempt start differs from the Run occurrence")
    if start.concretized_execution_digest != (
        concretized_execution.concretized_execution_digest()
    ):
        _fail("Attempt start concretized execution differs")
    if start.implementation_digest != concretized_execution.implementation_digest:
        _fail("Attempt start implementation binding differs")
    if start.attempt_admission_request_digest != (
        concretized_execution.attempt_admission_request_digest
    ):
        _fail("Attempt start admission-request binding differs")
    start.validate_authorized_disposition(disposition)

    if retry_basis.run_id != run.run_id or (
        retry_basis.candidate_attempt_id != start.attempt_id
    ):
        _fail("retry basis differs from the candidate Attempt")
    if retry_basis.semantic_spec_digest != run.semantic_spec_digest:
        _fail("retry basis changes semantic Run identity")
    if retry_basis.candidate_concretized_execution_digest != (
        concretized_execution.concretized_execution_digest()
    ):
        _fail("retry basis candidate concretized execution differs")
    _validate_retry_basis_names(retry_basis)

    expected_ordinal = len(prior) + 1
    if start.monotonic_ordinal != expected_ordinal:
        _fail("Attempt ordinal is not the next append-only ordinal")
    if not prior:
        if retry_basis.code is not AttemptRetryBasisCodeV1.INITIAL_EXECUTION:
            _fail("first Attempt requires INITIAL_EXECUTION basis")
        if start.parent_attempt_id is not None:
            _fail("first Attempt cannot have a parent Attempt")
    else:
        latest = prior[-1]
        if retry_basis.code is AttemptRetryBasisCodeV1.INITIAL_EXECUTION:
            _fail("later Attempt cannot use INITIAL_EXECUTION basis")
        if retry_basis.code is AttemptRetryBasisCodeV1.QUALIFICATION_INTEGRATION_ESCAPE:
            _fail("Qualification integration escape requires a successor Run")
        if start.parent_attempt_id != latest.start.attempt_id:
            _fail("retry parent is not the latest Attempt")
        if retry_basis.prior_attempt_id != latest.start.attempt_id:
            _fail("retry basis is not bound to the latest Attempt")
        if retry_basis.prior_concretized_execution_digest != (
            latest.concretized_execution.concretized_execution_digest()
        ):
            _fail("retry basis prior concretized execution differs")
        if retry_basis.code is AttemptRetryBasisCodeV1.EXACT_RESTORE_AFTER_FAILURE:
            if latest.terminal is None or latest.terminal.status not in {
                AttemptTerminalStatusV1.INTERRUPTED,
                AttemptTerminalStatusV1.FAILED,
                AttemptTerminalStatusV1.QUARANTINED,
            }:
                _fail("exact-restore retry lacks a failed terminal predecessor")
            if restore_authorization is None:
                _fail("exact-restore retry lacks checkpoint authorization")
        implementation_changed = (
            latest.concretized_execution.implementation_digest
            != concretized_execution.implementation_digest
        )
        if implementation_changed:
            if implementation_conformance is None:
                _fail("implementation change lacks exact conformance evidence")
            if retry_basis.implementation_conformance_digest != (
                implementation_conformance.conformance_digest()
            ):
                _fail("retry basis exact conformance digest differs")
            if (
                implementation_conformance.run_id != run.run_id
                or implementation_conformance.candidate_attempt_id != start.attempt_id
                or implementation_conformance.prior_implementation_digest
                != latest.concretized_execution.implementation_digest
                or implementation_conformance.candidate_implementation_digest
                != implementation.implementation_digest()
                or implementation_conformance.candidate_abi_digest
                != implementation.abi_digest
                or implementation_conformance.candidate_device_conformance_envelope_digest
                != implementation.device_conformance_envelope_digest
            ):
                _fail("exact implementation conformance binding differs")
        elif (
            implementation_conformance is not None
            or retry_basis.implementation_conformance_digest is not None
        ):
            _fail("unchanged implementation cannot claim a conformance delta")

    if admission_request.requested_restore_checkpoint_digest != (
        start.restore_checkpoint_digest
    ):
        _fail("admission request restore checkpoint differs from the Attempt")

    if start.restore_checkpoint_digest is None:
        semantic_start = semantic_spec.absolute_start_next_tick
        if any(
            value != semantic_start
            for value in (
                start.initial_frontiers.last_dispatched_next_tick,
                start.initial_frontiers.last_computed_next_tick,
                start.initial_frontiers.durable_checkpoint_next_tick,
                start.initial_frontiers.published_artifact_next_tick,
            )
        ):
            _fail(
                "non-restored Attempt frontiers differ from the semantic "
                "absolute start"
            )

    _validate_restore(
        run=run,
        semantic_spec=semantic_spec,
        start=start,
        authorization=restore_authorization,
        prior=prior,
        external_source_record=restore_source_record,
    )
    if continuation_lease.run_id != run.run_id or (
        continuation_lease.attempt_id != start.attempt_id
    ):
        _fail("continuation lease differs from Attempt identity")
    if continuation_lease.restore_checkpoint_digest != (
        start.restore_checkpoint_digest
    ):
        _fail("continuation lease restore binding differs")
    if start.canonical_continuation_lease_digest != (continuation_lease.lease_digest()):
        _fail("Attempt start continuation lease digest differs")
    if not isinstance(continuation_lease_ledger, ContinuationLeaseLedgerV1):
        _fail("Attempt start requires a canonical continuation lease ledger")
    if continuation_lease_ledger.current_lease.lease_digest() != (
        continuation_lease.lease_digest()
    ):
        _fail("Attempt start does not hold the canonical continuation lease")
    if not continuation_lease_ledger.is_active:
        _fail("Attempt start continuation lease is not active")
    lease_digests = {lease.lease_digest() for lease in continuation_lease_ledger.leases}
    if any(
        record.start.canonical_continuation_lease_digest not in lease_digests
        for record in prior
    ):
        _fail("Attempt history is not covered by one continuation lease lineage")
    side_effect_at = _utc(
        execution_side_effect_at_utc, field="execution_side_effect_at_utc"
    )
    if not (
        _utc(continuation_lease.granted_at_utc, field="lease.granted_at_utc")
        <= side_effect_at
        < _utc(continuation_lease.expires_at_utc, field="lease.expires_at_utc")
    ):
        _fail("execution side effect lies outside the continuation lease window")
    return {
        "schema_version": "crca_attempt_start_validation_v1",
        "status": "START_VALID",
        "run_id": str(run.run_id),
        "attempt_id": str(start.attempt_id),
        "attempt_ordinal": start.monotonic_ordinal,
        "retry_basis_digest": retry_basis.basis_digest().to_dict(),
        "restore_authorization_digest": (
            None
            if restore_authorization is None
            else restore_authorization.authorization_digest().to_dict()
        ),
        "implementation_conformance_digest": (
            None
            if implementation_conformance is None
            else implementation_conformance.conformance_digest().to_dict()
        ),
        "execution_authority_granted": False,
    }


@dataclass(frozen=True, slots=True)
class ContinuationLeaseLedgerV1:
    """Immutable sequence validator for one canonical lineage scope."""

    leases: tuple[CanonicalContinuationLeaseV1, ...]
    events: tuple[ContinuationLeaseEventV1, ...] = ()

    def __post_init__(self) -> None:
        _validate_lease_ledger(self.leases, self.events)

    @classmethod
    def initial(
        cls, lease: CanonicalContinuationLeaseV1
    ) -> "ContinuationLeaseLedgerV1":
        return cls((lease,), ())

    def append_event(
        self, event: ContinuationLeaseEventV1
    ) -> "ContinuationLeaseLedgerV1":
        if event.code in _LEASE_TRANSITION_EVENTS:
            _fail("lease transition must append its successor atomically")
        return replace(self, events=(*self.events, event))

    def transition(
        self,
        event: ContinuationLeaseEventV1,
        successor: CanonicalContinuationLeaseV1,
    ) -> "ContinuationLeaseLedgerV1":
        if event.code not in _LEASE_TRANSITION_EVENTS:
            _fail("lease transition requires a successor-bearing event")
        if event.successor_lease_digest != successor.lease_digest():
            _fail("lease transition successor digest differs")
        return replace(
            self,
            leases=(*self.leases, successor),
            events=(*self.events, event),
        )

    @property
    def current_lease(self) -> CanonicalContinuationLeaseV1:
        return self.leases[-1]

    @property
    def canonical_publication_digest(self) -> ContentDigest | None:
        publications = [
            event.publication_digest
            for event in self.events
            if event.code is ContinuationLeaseEventCodeV1.PUBLISH_COMMITTED
        ]
        return publications[0] if publications else None

    @property
    def is_active(self) -> bool:
        if not self.events:
            return True
        return self.events[-1].code in {
            ContinuationLeaseEventCodeV1.HEARTBEAT,
            *_LEASE_TRANSITION_EVENTS,
        }


def _validate_event_binding(
    event: ContinuationLeaseEventV1,
    lease: CanonicalContinuationLeaseV1,
) -> None:
    if (
        event.lease_digest != lease.lease_digest()
        or event.run_id != lease.run_id
        or event.attempt_id != lease.attempt_id
        or event.owner_id != lease.owner_id
        or event.lease_epoch != lease.lease_epoch
        or event.fencing_token_digest != lease.fencing_token_digest
    ):
        _fail("continuation lease event differs from its fenced lease")
    at = _utc(event.event_at_utc, field="lease_event.event_at_utc")
    granted = _utc(lease.granted_at_utc, field="lease.granted_at_utc")
    expires = _utc(lease.expires_at_utc, field="lease.expires_at_utc")
    stale_event = event.code is ContinuationLeaseEventCodeV1.EXPIRED
    if stale_event and at < expires:
        _fail("lease expiry/reclaim occurred before the fencing deadline")
    if event.code is ContinuationLeaseEventCodeV1.REASSIGNED and at < granted:
        _fail("lease reassignment occurred before the predecessor grant")
    if event.code is ContinuationLeaseEventCodeV1.OWNER_DEAD and at < granted:
        _fail("dead-owner lease fence occurred before the predecessor grant")
    if (
        not stale_event
        and event.code
        not in {
            ContinuationLeaseEventCodeV1.REASSIGNED,
            ContinuationLeaseEventCodeV1.RECLAIMED,
            ContinuationLeaseEventCodeV1.OWNER_DEAD,
        }
        and not (granted <= at < expires)
    ):
        _fail("live lease event lies outside its fencing window")


def _validate_lease_ledger(
    leases: tuple[CanonicalContinuationLeaseV1, ...],
    events: tuple[ContinuationLeaseEventV1, ...],
) -> None:
    if not leases:
        _fail("continuation lease ledger requires an initial lease")
    first = leases[0]
    if first.lease_epoch != 1 or first.prior_lease_digest is not None:
        _fail("initial continuation lease must start epoch 1 without a predecessor")
    by_digest = {lease.lease_digest(): lease for lease in leases}
    if len(by_digest) != len(leases):
        _fail("continuation lease ledger repeats a lease")
    current = first
    next_lease_index = 1
    prior_event_digest: ContentDigest | None = None
    current_state = "ACTIVE"
    publication_count = 0
    prior_event_at: datetime | None = None
    for event in events:
        if event.prior_event_digest != prior_event_digest:
            _fail("continuation lease events are not one append-only digest chain")
        if current_state == "PUBLISHED":
            _fail("continuation lease event follows canonical publication")
        if current_state == "RELEASED" and (
            event.code is not ContinuationLeaseEventCodeV1.REASSIGNED
        ):
            _fail("released continuation lease permits only reassignment")
        if current_state == "EXPIRED" and (
            event.code is not ContinuationLeaseEventCodeV1.RECLAIMED
        ):
            _fail("expired continuation lease permits only reclaim")
        if current_state == "OWNER_DEAD" and (
            event.code is not ContinuationLeaseEventCodeV1.RECLAIMED
        ):
            _fail("dead-owner continuation lease permits only reclaim")
        if event.code is ContinuationLeaseEventCodeV1.REASSIGNED and (
            current_state != "RELEASED"
        ):
            _fail("continuation lease reassignment lacks a release receipt")
        if event.code is ContinuationLeaseEventCodeV1.RECLAIMED and (
            current_state not in {"EXPIRED", "OWNER_DEAD"}
        ):
            if current_state == "ACTIVE":
                if _utc(event.event_at_utc, field="lease_event.event_at_utc") < _utc(
                    current.expires_at_utc, field="lease.expires_at_utc"
                ):
                    _fail("lease expiry/reclaim occurred before the fencing deadline")
                # Preserve the V1 direct-at/after-expiry reclaim form.  A V2
                # proof-bound OWNER_DEAD event is the only pre-expiry fence.
            else:
                _fail("continuation lease reclaim lacks an expiry or dead-owner fence")
        if event.code is ContinuationLeaseEventCodeV1.RENEWED and (
            current_state != "ACTIVE"
        ):
            _fail("continuation lease renewal requires an active predecessor")
        _validate_event_binding(event, current)
        event_at = _utc(event.event_at_utc, field="lease_event.event_at_utc")
        if (
            event.code is ContinuationLeaseEventCodeV1.RECLAIMED
            and current_state == "EXPIRED"
            and event_at
            < _utc(current.expires_at_utc, field="lease.expires_at_utc")
        ):
            _fail("expired continuation lease reclaim occurred before its fence")
        if prior_event_at is not None and event_at < prior_event_at:
            _fail("continuation lease event time moved backwards")
        if event.code is ContinuationLeaseEventCodeV1.PUBLISH_COMMITTED:
            publication_count += 1
            if publication_count > 1:
                _fail("canonical lineage already has a publisher")
        if event.code in _LEASE_TRANSITION_EVENTS:
            if next_lease_index >= len(leases):
                _fail("lease transition has no appended successor")
            successor = leases[next_lease_index]
            if event.successor_lease_digest != successor.lease_digest():
                _fail("lease transition points to another successor")
            if (
                successor.prior_lease_digest != current.lease_digest()
                or successor.lease_epoch != current.lease_epoch + 1
                or successor.lease_id == current.lease_id
                or successor.fencing_token_digest == current.fencing_token_digest
                or successor.run_id != current.run_id
                or successor.lineage_scope_digest != current.lineage_scope_digest
                or successor.granted_at_utc != event.event_at_utc
            ):
                _fail(
                    "successor lease breaks identity, fencing, lineage, epoch, "
                    "or grant binding"
                )
            if event.code is ContinuationLeaseEventCodeV1.RENEWED and (
                successor.attempt_id != current.attempt_id
                or successor.owner_id != current.owner_id
                or successor.restore_checkpoint_digest
                != current.restore_checkpoint_digest
            ):
                _fail("lease renewal changes Attempt or owner identity")
            current = successor
            next_lease_index += 1
            current_state = "ACTIVE"
        elif event.code is ContinuationLeaseEventCodeV1.RELEASED:
            current_state = "RELEASED"
        elif event.code is ContinuationLeaseEventCodeV1.EXPIRED:
            current_state = "EXPIRED"
        elif event.code is ContinuationLeaseEventCodeV1.OWNER_DEAD:
            current_state = "OWNER_DEAD"
        elif event.code is ContinuationLeaseEventCodeV1.PUBLISH_COMMITTED:
            current_state = "PUBLISHED"
        prior_event_digest = event.event_digest()
        prior_event_at = event_at
    if next_lease_index != len(leases):
        _fail("continuation lease was appended without a transition event")


def require_canonical_publish_authority_v1(
    ledger: ContinuationLeaseLedgerV1,
    lease: CanonicalContinuationLeaseV1,
    *,
    publication_at_utc: str,
) -> None:
    """Fail with QUARANTINE_REQUIRED for a stale/racing publisher."""

    if ledger.canonical_publication_digest is not None:
        _fail("QUARANTINE_REQUIRED: canonical lineage is already published")
    if lease.lease_digest() != ledger.current_lease.lease_digest():
        _fail("QUARANTINE_REQUIRED: publisher holds a stale fencing token")
    at = _utc(publication_at_utc, field="publication_at_utc")
    if not (
        _utc(lease.granted_at_utc, field="lease.granted_at_utc")
        <= at
        < _utc(lease.expires_at_utc, field="lease.expires_at_utc")
    ):
        _fail("QUARANTINE_REQUIRED: publisher lease is expired")


def non_runtime_occurrence_decision_v1(action_kind: str) -> dict[str, Any]:
    """Keep derivation/evaluation work outside physical Attempt identity."""

    if action_kind not in {
        "ARTIFACT_ONLY",
        "EVIDENCE_REEVALUATION",
        "REPORTING_ONLY",
    }:
        _fail(f"unsupported non-runtime occurrence action: {action_kind!r}")
    return {
        "schema_version": "crca_non_runtime_occurrence_decision_v1",
        "action_kind": action_kind,
        "runtime_attempt_created": False,
        "execution_side_effect_authorized": False,
        "decision": "SEPARATE_DERIVATION_OR_EVALUATION_OCCURRENCE",
    }


def _append_exact_file_v1(
    target: Path,
    encoded: bytes,
    *,
    conflict_message: str,
) -> Path:
    try:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o640,
        )
    except FileExistsError:
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != encoded
        ):
            _fail(conflict_message)
        return target
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # A partial O_EXCL file is fail-closed evidence and is never overwritten.
        raise
    return target


def _receipt_directory_v1(root: str | Path, kind: str) -> Path:
    requested = Path(root).expanduser()
    if requested.is_symlink():
        _fail("occurrence receipt root must be a real directory")
    requested.mkdir(parents=True, exist_ok=True)
    base = requested.resolve()
    if not base.is_dir():
        _fail("occurrence receipt root must be a real directory")
    directory = base / kind
    directory.mkdir(mode=0o750, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        _fail("occurrence receipt kind path must be a real directory")
    return directory


def append_occurrence_receipt_v1(
    root: str | Path,
    *,
    kind: str,
    payload: Mapping[str, Any],
    digest: ContentDigest,
) -> Path:
    """Append one canonical content-addressed receipt without an overwrite path."""

    if kind not in _RECEIPT_KINDS:
        _fail(f"unsupported occurrence receipt kind: {kind!r}")
    value = dict(payload)
    if canonical_sha256(value) != digest:
        _fail("occurrence receipt digest differs from canonical payload")
    directory = _receipt_directory_v1(root, kind)
    target = directory / f"{digest.hex}.json"
    encoded = canonical_json_bytes(value) + b"\n"
    return _append_exact_file_v1(
        target,
        encoded,
        conflict_message="append-only occurrence receipt path already differs",
    )


def append_attempt_terminal_receipt_v1(
    root: str | Path,
    *,
    start: AttemptStartV1,
    terminal: AttemptTerminalV1,
    run: RunOccurrenceV1,
    semantic_spec: SemanticRunSpecV1,
) -> Path:
    """Persist exactly one immutable terminal binding per Attempt."""

    validate_terminal_once_v1(start, None, terminal, run, semantic_spec)
    terminal_digest = terminal.terminal_digest()
    binding = {
        "schema_version": "crca_attempt_terminal_pointer_v1",
        "attempt_id": str(start.attempt_id),
        "attempt_start_digest": start.start_digest().to_dict(),
        "terminal_digest": terminal_digest.to_dict(),
    }
    index = _receipt_directory_v1(root, "attempt-terminal-index")
    index_name = canonical_sha256(
        {
            "schema_version": "crca_attempt_terminal_pointer_v1",
            "attempt_id": str(start.attempt_id),
        }
    )
    _append_exact_file_v1(
        index / f"{index_name.hex}.json",
        canonical_json_bytes(binding) + b"\n",
        conflict_message="Attempt already has a different terminal receipt",
    )
    return append_occurrence_receipt_v1(
        root,
        kind="attempt-terminal",
        payload=terminal.to_dict(),
        digest=terminal_digest,
    )


def commit_canonical_publication_v1(
    root: str | Path,
    *,
    ledger: ContinuationLeaseLedgerV1,
    lease: CanonicalContinuationLeaseV1,
    event: ContinuationLeaseEventV1,
) -> tuple[ContinuationLeaseLedgerV1, Path]:
    """Atomically claim a lineage's single canonical publication pointer."""

    if event.code is not ContinuationLeaseEventCodeV1.PUBLISH_COMMITTED:
        _fail("canonical publication requires a PUBLISH_COMMITTED event")
    require_canonical_publish_authority_v1(
        ledger,
        lease,
        publication_at_utc=event.event_at_utc,
    )
    committed = ledger.append_event(event)
    assert event.publication_digest is not None
    binding = {
        "schema_version": "crca_canonical_publication_pointer_v1",
        "lineage_scope_digest": lease.lineage_scope_digest.to_dict(),
        "lease_digest": lease.lease_digest().to_dict(),
        "event_digest": event.event_digest().to_dict(),
        "publication_digest": event.publication_digest.to_dict(),
    }
    directory = _receipt_directory_v1(root, "canonical-publication")
    pointer = directory / f"{lease.lineage_scope_digest.hex}.json"
    _append_exact_file_v1(
        pointer,
        canonical_json_bytes(binding) + b"\n",
        conflict_message=(
            "QUARANTINE_REQUIRED: canonical lineage publication pointer already differs"
        ),
    )
    append_occurrence_receipt_v1(
        root,
        kind="lease-event",
        payload=event.to_dict(),
        digest=event.event_digest(),
    )
    return committed, pointer


__all__ = [
    "AttemptAdmissionError",
    "AttemptRecordV1",
    "ContinuationLeaseLedgerV1",
    "append_occurrence_receipt_v1",
    "append_attempt_terminal_receipt_v1",
    "commit_canonical_publication_v1",
    "non_runtime_occurrence_decision_v1",
    "require_canonical_publish_authority_v1",
    "validate_attempt_start_v1",
    "validate_frontier_progression_v1",
    "validate_terminal_once_v1",
]
