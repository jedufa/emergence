"""Append-only, reentrant CRCA programme driver owned by the Scheduler.

This module is deliberately a journal and reconciler, not a new authority
database.  Before an external effect it writes one immutable action intent.
After a crash it adopts a live owner, settles an already-complete effect, or
reclaims only an absent/stale fenced action.  Work, Session, Resource Manager,
runtime, Qualification, lifecycle, Bridge, and Project Manager records remain
the owners of their respective facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tools.scheduler.contracts import (
    ActorError,
    advisory_lock,
    atomic_write_json,
    canonical_bytes,
    canonical_host_actor_state_root,
    read_json,
    scheduler_root,
)
from tools.scheduler.programme_contracts import (
    ContentDigest,
    NotificationDeduplicationProofV1,
    OwnedStatusFactV1,
    ProgrammeActionClaimV1,
    ProgrammeActionIntentV1,
    ProgrammeActionPlanV1,
    ProgrammeActionSettlementV1,
    ProgrammeAuthorizationV1,
    ProgrammeContractError,
    ProgrammeDriverLeaseV1,
    ProgrammeTerminalStateV1,
    ProgrammeTransitionReceiptV1,
    LaunchpadResourceStartLeaseV1,
    ResourceStartLeaseEventCodeV1,
    ResourceStartLeaseEventV1,
    ResourceStartLeaseEventV2,
    ResourceStartLeaseEventV3,
    ResourceStartLeaseV1,
    TerminalNotificationDeliveryV1,
    TerminalNotificationIntentV1,
    canonical_sha256,
    project_unified_status_v1,
    resource_ids_conflict_v1,
    require_exclusive_resource_start_lease_v1,
)


PROGRAMME_JOURNAL_SCHEMA_VERSION = "crca_programme_journal_v1"
PROGRAMME_DRIVER_STATUS_SCHEMA_VERSION = "crca_programme_driver_status_v1"
ACTION_OBSERVATION_STATES = {"ABSENT", "LIVE", "COMPLETED", "STALE_FENCED"}
LEGACY_GPU_MIGRATION_AUTHORIZATION_SCHEMA_VERSION = (
    "crca_legacy_gpu_migration_authorization_v1"
)
CANONICAL_LEGACY_GPU_MIGRATION_MANIFEST_SHA256 = (
    "99a852711807c063f44a382507ee72891d1f820706c9029e20253dbad91fda98"
)
CANONICAL_LEGACY_GPU_MIGRATION_ROWS = {
    "LPRSTART-0821AC1B00E620DC015602D2": (
        "9639b1dd603d3340c67540e5012178ad1633b7d9be652857c0e9872cad94cb73"
    ),
    "LPRSTART-ADC91A35045464D12164591B": (
        "1a98d7cdc1061ce89d5a890d1c8c3f10ba1f6a48eae847e6601a989a4bee2fe9"
    ),
    "LPRSTART-6D9EDFA9DD8AF11F7C5FAEAC": (
        "d9ebaed2103cb9990ad90380726df4d5ee04361c01a50370655f977268fa053b"
    ),
    "LPRSTART-698230340528484EC507F181": (
        "5efc7af70062d3db54cbe946f368b176bdc70493d8c42259606ce6c928451353"
    ),
    "LPRSTART-2EB05148123296D1190A16FB": (
        "eaad0ed4696dcb7860e4794ffe7f0c0fa22586ed5739f9c0a5b5dff9a8584188"
    ),
    "LPRSTART-6D52F3BABC922E17D8236F3A": (
        "fc2e9be10821b7940c66da969aa97d23a642cea896eb9250a9f88d30cde9fc46"
    ),
    "LPRSTART-5BB1DB292DA2C0641A8DC6F1": (
        "4aef9d491e533320219eafc26d7e2d2af249bf1cc6539ee6fe2aa0d668791160"
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_utc(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError) as error:
        raise ActorError(
            "programme timestamp must use canonical second-resolution UTC",
            code="programme_timestamp_invalid",
        ) from error


def _add_seconds(value: str, seconds: int | float) -> str:
    return (
        _parse_utc(value) + timedelta(seconds=max(0.0, float(seconds)))
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _local_process_start_identity(process_id: int) -> str | None:
    try:
        text = (Path("/proc") / str(int(process_id)) / "stat").read_text(
            encoding="utf-8"
        )
        remainder = text[text.rfind(")") + 2 :].split()
        return remainder[19]
    except (OSError, ValueError, IndexError):
        return None


def _content_digest(raw: object, *, field: str) -> ContentDigest:
    if not isinstance(raw, Mapping):
        raise ActorError(f"{field} must be a content digest", code="programme_record_invalid")
    try:
        return ContentDigest(
            algorithm=str(raw.get("algorithm") or ""),
            hex=str(raw.get("hex") or ""),
        )
    except Exception as error:
        raise ActorError(
            f"{field} is not a valid content digest",
            code="programme_record_invalid",
        ) from error


def _same_payload(path: Path, payload: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        existing = read_json(path, name=path.name)
    except ActorError:
        return False
    return canonical_bytes(existing) == canonical_bytes(dict(payload))


def _write_immutable(path: Path, payload: Mapping[str, Any], *, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _same_payload(path, payload):
            return path
        raise ActorError(
            f"conflicting immutable {label}",
            code="programme_immutable_conflict",
            problems=[str(path)],
        )
    atomic_write_json(path, dict(payload))
    return path


def _terminal_notification_binding_v1(
    authorization: Mapping[str, Any],
    authorization_digest: ContentDigest,
    transition: Mapping[str, Any],
) -> dict[str, object]:
    terminal = ProgrammeTerminalStateV1(str(transition.get("terminal_state") or ""))
    transition_digest = canonical_sha256(transition)
    programme_id = str(authorization.get("programme_id") or "")
    terminal_programme_digest = canonical_sha256(
        {
            "programme_id": programme_id,
            "authorization_digest": authorization_digest.to_dict(),
            "terminal_transition_digest": transition_digest.to_dict(),
            "terminal_state": terminal.value,
        }
    )
    return {
        "programme_id": programme_id,
        "terminal_state": terminal,
        "terminal_transition_digest": transition_digest,
        "terminal_programme_digest": terminal_programme_digest,
        "recipient_digest": _content_digest(
            authorization.get("notification_recipient_digest"),
            field="notification_recipient_digest",
        ),
        "subject_digest": canonical_sha256(
            {
                "programme_id": programme_id,
                "terminal_state": terminal.value,
                "notification_kind": "EIEL_PROGRAMME_TERMINAL",
            }
        ),
        "body_digest": canonical_sha256(
            {
                "programme_id": programme_id,
                "terminal_state": terminal.value,
                "terminal_transition_digest": transition_digest.to_dict(),
                "delivery_contract": "PROJECT_MANAGER_TERMINAL_ONLY",
            }
        ),
    }


class ProgrammeJournalV1:
    """One bounded filesystem journal for an immutable programme authorization."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.authorization_path = self.root / "authorization.json"
        self.transitions_dir = self.root / "transitions"
        self.action_plans_dir = self.root / "action-plans"
        self.action_intents_dir = self.root / "action-intents"
        self.action_claims_dir = self.root / "action-claims"
        self.action_claim_heads_dir = self.root / "action-claim-heads"
        self.action_settlements_dir = self.root / "action-settlements"
        self.driver_leases_dir = self.root / "driver-leases"
        self.driver_head_path = self.root / "driver-lease-head.json"
        self.notification_intents_dir = self.root / "notification-intents"
        self.notification_deliveries_dir = self.root / "notification-deliveries"
        self.lock_path = self.root / ".programme.lock"

    def initialize(self, authorization: ProgrammeAuthorizationV1) -> Path:
        if not isinstance(authorization, ProgrammeAuthorizationV1):
            raise ActorError(
                "programme authorization contract is invalid",
                code="programme_authorization_invalid",
            )
        with advisory_lock(self.lock_path):
            return _write_immutable(
                self.authorization_path,
                authorization.to_dict(),
                label="ProgrammeAuthorization",
            )

    def authorization(self) -> dict[str, Any]:
        if not self.authorization_path.is_file():
            raise ActorError(
                "programme authorization is missing",
                code="programme_authorization_missing",
            )
        value = read_json(self.authorization_path, name="ProgrammeAuthorization")
        if value.get("schema_version") != "crca_programme_authorization_v1_1":
            raise ActorError(
                "programme authorization schema is unsupported",
                code="programme_authorization_invalid",
            )
        try:
            normalized = ProgrammeAuthorizationV1.from_dict(value).to_dict()
        except (ProgrammeContractError, TypeError, ValueError) as error:
            raise ActorError(
                "programme authorization content is invalid",
                code="programme_authorization_invalid",
            ) from error
        if canonical_bytes(normalized) != canonical_bytes(value):
            raise ActorError(
                "programme authorization is not canonical",
                code="programme_authorization_invalid",
            )
        return value

    def authorization_digest(self) -> ContentDigest:
        return canonical_sha256(self.authorization())

    def transitions(self) -> list[dict[str, Any]]:
        if not self.transitions_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        paths = sorted(self.transitions_dir.glob("*.json"))
        for path in paths:
            row = read_json(path, name="ProgrammeTransitionReceipt")
            rows.append(row)
        expected_prior: ContentDigest | None = None
        authorization = self.authorization()
        authorization_digest = self.authorization_digest()
        programme_id = str(authorization.get("programme_id") or "")
        plans = self.action_plans()
        if len(rows) > len(plans):
            raise ActorError(
                "programme transition has no authorized action plan",
                code="programme_transition_chain_invalid",
            )
        for ordinal, row in enumerate(rows, start=1):
            try:
                transition = ProgrammeTransitionReceiptV1.from_dict(row)
            except (ProgrammeContractError, TypeError, ValueError) as error:
                raise ActorError(
                    "programme transition contract is invalid",
                    code="programme_transition_chain_invalid",
                ) from error
            if transition.to_dict() != row:
                raise ActorError(
                    "programme transition is not canonical",
                    code="programme_transition_chain_invalid",
                )
            if transition.ordinal != ordinal:
                raise ActorError(
                    "programme transition ordinal is not contiguous",
                    code="programme_transition_chain_invalid",
                )
            if (
                transition.programme_id != programme_id
                or transition.authorization_digest != authorization_digest
            ):
                raise ActorError(
                    "programme transition binds another authorization",
                    code="programme_transition_chain_invalid",
                )
            if transition.prior_transition_digest != expected_prior:
                raise ActorError(
                    "programme transition prior digest is not canonical",
                    code="programme_transition_chain_invalid",
                )
            self._validate_transition_plan_binding(
                transition, ProgrammeActionPlanV1.from_dict(plans[ordinal - 1])
            )
            expected_prior = canonical_sha256(row)
            expected_name = f"{ordinal:06d}-{expected_prior.hex}.json"
            actual_name = paths[ordinal - 1].name
            if actual_name != expected_name:
                raise ActorError(
                    "programme transition filename does not match its content digest",
                    code="programme_transition_chain_invalid",
                )
        terminal_rows = [row for row in rows if row.get("terminal_state") is not None]
        if terminal_rows and terminal_rows[-1] is not rows[-1]:
            raise ActorError(
                "programme transition exists after terminal state",
                code="programme_transition_chain_invalid",
            )
        if len(terminal_rows) > 1:
            raise ActorError(
                "programme has more than one terminal transition",
                code="programme_transition_chain_invalid",
            )
        return rows

    def _validate_intent_plan_binding(
        self,
        intent: ProgrammeActionIntentV1,
        plan: ProgrammeActionPlanV1,
    ) -> None:
        if (
            intent.programme_id != plan.programme_id
            or intent.authorization_digest != plan.authorization_digest
            or intent.action_plan_digest != plan.plan_digest()
            or intent.gate_id != plan.gate_id
            or intent.action_code != plan.action_code
            or intent.action_payload_digest != plan.action_payload_digest
            or intent.exact_owner_id != plan.exact_owner_id
            or intent.exact_owner_digest != plan.exact_owner_digest
        ):
            raise ActorError(
                "action intent differs from its immutable action plan",
                code="programme_action_invalid",
            )

    def _validate_transition_plan_binding(
        self,
        transition: ProgrammeTransitionReceiptV1,
        plan: ProgrammeActionPlanV1,
    ) -> None:
        exact_pairs = (
            (transition.programme_id, plan.programme_id),
            (transition.authorization_digest, plan.authorization_digest),
            (transition.ordinal, plan.ordinal),
            (transition.gate_id, plan.gate_id),
            (transition.gate_state, plan.success_gate_state),
            (transition.work_digest, plan.work_digest),
            (transition.session_digest, plan.session_digest),
            (transition.assignment_digest, plan.assignment_digest),
            (transition.dispatch_digest, plan.dispatch_digest),
            (
                transition.scheduler_assignment_lease_digest,
                plan.scheduler_assignment_lease_digest,
            ),
            (transition.run_digest, plan.run_digest),
            (transition.attempt_digest, plan.attempt_digest),
            (transition.checkpoint_digest, plan.checkpoint_digest),
            (transition.next_safe_action, plan.next_safe_action),
            (transition.wake_condition, plan.wake_condition),
            (transition.terminal_state, plan.terminal_state),
        )
        if any(observed != expected for observed, expected in exact_pairs):
            raise ActorError(
                "programme transition differs from its authorized action plan",
                code="programme_transition_chain_invalid",
            )
        authorization = self.authorization()
        if (
            transition.recovery_state is not None
            and transition.recovery_state
            not in set(authorization.get("allowed_recovery_codes") or [])
        ):
            raise ActorError(
                "programme transition uses an unauthorized recovery code",
                code="programme_transition_chain_invalid",
            )
        if transition.action_intent_id is None:
            if any(
                value is not None
                for value in (
                    transition.effect_receipt_digest,
                    transition.result_receipt_digest,
                    transition.settlement_receipt_digest,
                )
            ):
                raise ActorError(
                    "programme transition has effects without an action intent",
                    code="programme_transition_chain_invalid",
                )
            return
        raw_intent = self.action_intent(transition.action_intent_id)
        try:
            intent = ProgrammeActionIntentV1.from_dict(raw_intent)
        except (ProgrammeContractError, TypeError, ValueError) as error:
            raise ActorError(
                "programme transition action intent is invalid",
                code="programme_transition_chain_invalid",
            ) from error
        if (
            intent.to_dict() != raw_intent
            or intent.action_digest() != transition.action_intent_digest
        ):
            raise ActorError(
                "programme transition binds another action intent",
                code="programme_transition_chain_invalid",
            )
        self._validate_intent_plan_binding(intent, plan)
        settlement = self.action_settlement(intent.action_intent_id())
        if transition.settlement_receipt_digest is None:
            if any(
                value is not None
                for value in (
                    transition.effect_receipt_digest,
                    transition.result_receipt_digest,
                )
            ):
                raise ActorError(
                    "programme transition effect has no settlement",
                    code="programme_transition_chain_invalid",
                )
            return
        if settlement is None or (
            canonical_sha256(settlement) != transition.settlement_receipt_digest
            or _content_digest(
                settlement.get("effect_receipt_digest"),
                field="effect_receipt_digest",
            )
            != transition.effect_receipt_digest
            or _content_digest(
                settlement.get("result_receipt_digest"),
                field="result_receipt_digest",
            )
            != transition.result_receipt_digest
        ):
            raise ActorError(
                "programme transition settlement binding is invalid",
                code="programme_transition_chain_invalid",
            )

    def latest_transition(self) -> dict[str, Any] | None:
        rows = self.transitions()
        return rows[-1] if rows else None

    def append_transition(self, transition: ProgrammeTransitionReceiptV1) -> Path:
        if not isinstance(transition, ProgrammeTransitionReceiptV1):
            raise ActorError(
                "programme transition contract is invalid",
                code="programme_transition_invalid",
            )
        with advisory_lock(self.lock_path):
            rows = self.transitions()
            expected_ordinal = len(rows) + 1
            expected_prior = None if not rows else canonical_sha256(rows[-1])
            if transition.authorization_digest != self.authorization_digest():
                raise ActorError(
                    "transition binds another authorization",
                    code="programme_transition_invalid",
                )
            if transition.ordinal != expected_ordinal:
                raise ActorError(
                    "transition ordinal is not next",
                    code="programme_transition_invalid",
                )
            if transition.prior_transition_digest != expected_prior:
                raise ActorError(
                    "transition prior digest is not the current head",
                    code="programme_transition_invalid",
                )
            if rows and rows[-1].get("terminal_state") is not None:
                raise ActorError(
                    "terminal programme cannot transition again",
                    code="programme_already_terminal",
                )
            plans = self.action_plans()
            if len(plans) < expected_ordinal:
                raise ActorError(
                    "transition has no authorized action plan",
                    code="programme_transition_invalid",
                )
            try:
                self._validate_transition_plan_binding(
                    transition,
                    ProgrammeActionPlanV1.from_dict(plans[expected_ordinal - 1]),
                )
            except ActorError as error:
                raise ActorError(
                    str(error), code="programme_transition_invalid"
                ) from error
            path = self.transitions_dir / (
                f"{transition.ordinal:06d}-{transition.transition_digest().hex}.json"
            )
            return _write_immutable(
                path, transition.to_dict(), label="ProgrammeTransitionReceipt"
            )

    def action_plans(self) -> list[dict[str, Any]]:
        if not self.action_plans_dir.is_dir():
            return []
        paths = sorted(self.action_plans_dir.glob("*.json"))
        rows: list[dict[str, Any]] = []
        authorization = self.authorization()
        authorization_digest = self.authorization_digest()
        gates = {
            str(gate["gate_id"]): dict(gate)
            for gate in authorization.get("gate_graph") or []
        }
        prior_gate: str | None = None
        for ordinal, path in enumerate(paths, start=1):
            raw = read_json(path, name="ProgrammeActionPlan")
            try:
                plan = ProgrammeActionPlanV1.from_dict(raw)
            except (ProgrammeContractError, TypeError, ValueError) as error:
                raise ActorError(
                    "programme action plan is invalid",
                    code="programme_action_plan_invalid",
                    problems=[str(path)],
                ) from error
            if plan.to_dict() != raw:
                raise ActorError(
                    "programme action plan is not canonical",
                    code="programme_action_plan_invalid",
                )
            if plan.ordinal != ordinal:
                raise ActorError(
                    "programme action plan ordinal is not contiguous",
                    code="programme_action_plan_invalid",
                )
            if (
                plan.programme_id != authorization.get("programme_id")
                or plan.authorization_digest != authorization_digest
                or plan.gate_id not in gates
            ):
                raise ActorError(
                    "programme action plan binds another authority",
                    code="programme_action_plan_invalid",
                )
            if prior_gate is not None:
                allowed = set(gates[prior_gate].get("allowed_successors") or [])
                if plan.gate_id != prior_gate and plan.gate_id not in allowed:
                    raise ActorError(
                        "programme action plan violates the authorized gate graph",
                        code="programme_action_plan_invalid",
                    )
            if plan.terminal_state is not None and gates[plan.gate_id].get(
                "allowed_successors"
            ):
                raise ActorError(
                    "programme action plan reaches terminal before the final gate",
                    code="programme_action_plan_invalid",
                )
            expected_name = f"{ordinal:06d}-{plan.plan_digest().hex}.json"
            if path.name != expected_name:
                raise ActorError(
                    "programme action plan filename does not match its digest",
                    code="programme_action_plan_invalid",
                )
            rows.append(raw)
            prior_gate = plan.gate_id
        return rows

    def persist_action_plan(self, plan: ProgrammeActionPlanV1) -> Path:
        if (
            plan.programme_id != self.authorization().get("programme_id")
            or plan.authorization_digest != self.authorization_digest()
        ):
            raise ActorError(
                "action plan binds another authorization",
                code="programme_action_plan_invalid",
            )
        with advisory_lock(self.lock_path):
            rows = self.action_plans()
            if plan.ordinal != len(rows) + 1:
                existing = self.action_plans_dir / (
                    f"{plan.ordinal:06d}-{plan.plan_digest().hex}.json"
                )
                if _same_payload(existing, plan.to_dict()):
                    return existing
                raise ActorError(
                    "action plan ordinal is not next",
                    code="programme_action_plan_invalid",
                )
            gates = {
                str(gate["gate_id"]): dict(gate)
                for gate in self.authorization().get("gate_graph") or []
            }
            if plan.gate_id not in gates:
                raise ActorError(
                    "action plan gate is not authorized",
                    code="programme_action_plan_invalid",
                )
            if rows:
                prior_plan = ProgrammeActionPlanV1.from_dict(rows[-1])
                allowed = set(
                    gates[prior_plan.gate_id].get("allowed_successors") or []
                )
                if plan.gate_id != prior_plan.gate_id and plan.gate_id not in allowed:
                    raise ActorError(
                        "action plan violates the authorized gate graph",
                        code="programme_action_plan_invalid",
                    )
            if plan.terminal_state is not None and gates[plan.gate_id].get(
                "allowed_successors"
            ):
                raise ActorError(
                    "action plan reaches terminal before the final gate",
                    code="programme_action_plan_invalid",
                )
            path = self.action_plans_dir / (
                f"{plan.ordinal:06d}-{plan.plan_digest().hex}.json"
            )
            written = _write_immutable(
                path, plan.to_dict(), label="ProgrammeActionPlan"
            )
            return written

    def action_intent_for_plan(
        self, plan: ProgrammeActionPlanV1
    ) -> dict[str, Any] | None:
        if not self.action_intents_dir.is_dir():
            return None
        matches: list[dict[str, Any]] = []
        for path in sorted(self.action_intents_dir.glob("*.json")):
            raw = read_json(path, name="ProgrammeActionIntent")
            try:
                intent = ProgrammeActionIntentV1.from_dict(raw)
            except (ProgrammeContractError, TypeError, ValueError) as error:
                raise ActorError(
                    "programme action intent is invalid",
                    code="programme_action_invalid",
                ) from error
            if intent.to_dict() != raw or path.stem != intent.action_intent_id():
                raise ActorError(
                    "programme action intent is not canonical",
                    code="programme_action_invalid",
                )
            if intent.action_plan_digest == plan.plan_digest():
                self._validate_intent_plan_binding(intent, plan)
                matches.append(raw)
        if len(matches) > 1:
            raise ActorError(
                "action plan has multiple immutable intents",
                code="programme_action_invalid",
            )
        return matches[0] if matches else None

    def persist_action_intent(self, intent: ProgrammeActionIntentV1) -> Path:
        if intent.authorization_digest != self.authorization_digest():
            raise ActorError(
                "action intent binds another authorization",
                code="programme_action_invalid",
            )
        with advisory_lock(self.lock_path):
            return _write_immutable(
                self.action_intents_dir / f"{intent.action_intent_id()}.json",
                intent.to_dict(),
                label="ProgrammeActionIntent",
            )

    def action_claims(self, action_intent_id: str) -> list[dict[str, Any]]:
        directory = self.action_claims_dir / action_intent_id
        if not directory.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        prior: ContentDigest | None = None
        for generation, path in enumerate(sorted(directory.glob("*.json")), start=1):
            raw = read_json(path, name="ProgrammeActionClaim")
            claim = ProgrammeActionClaimV1(
                schema_version=str(raw.get("schema_version") or ""),
                action_intent_id=str(raw.get("action_intent_id") or ""),
                action_intent_digest=_content_digest(
                    raw.get("action_intent_digest"), field="action_intent_digest"
                ),
                programme_id=str(raw.get("programme_id") or ""),
                authorization_digest=_content_digest(
                    raw.get("authorization_digest"), field="authorization_digest"
                ),
                driver_lease_digest=_content_digest(
                    raw.get("driver_lease_digest"), field="driver_lease_digest"
                ),
                driver_generation=int(raw.get("driver_generation") or 0),
                execution_fencing_token_digest=_content_digest(
                    raw.get("execution_fencing_token_digest"),
                    field="execution_fencing_token_digest",
                ),
                claim_generation=int(raw.get("claim_generation") or 0),
                prior_claim_digest=(
                    None
                    if raw.get("prior_claim_digest") is None
                    else _content_digest(
                        raw.get("prior_claim_digest"),
                        field="prior_claim_digest",
                    )
                ),
                claimed_at_utc=str(raw.get("claimed_at_utc") or ""),
                expires_at_utc=str(raw.get("expires_at_utc") or ""),
            )
            if (
                claim.action_intent_id != action_intent_id
                or claim.claim_generation != generation
                or claim.prior_claim_digest != prior
                or claim.to_dict() != raw
                or path.name
                != f"{generation:06d}-{claim.claim_digest().hex}.json"
            ):
                raise ActorError(
                    "programme action claim chain is invalid",
                    code="programme_action_claim_invalid",
                )
            rows.append(raw)
            prior = claim.claim_digest()
        return rows

    def action_claim_head(self, action_intent_id: str) -> dict[str, Any] | None:
        path = self.action_claim_heads_dir / f"{action_intent_id}.json"
        if not path.is_file():
            return None
        head = read_json(path, name="ProgrammeActionClaimHead")
        claims = self.action_claims(action_intent_id)
        if not claims or canonical_sha256(claims[-1]).hex != head.get("claim_digest"):
            raise ActorError(
                "programme action claim head is invalid",
                code="programme_action_claim_invalid",
            )
        return claims[-1]

    def claim_action(
        self,
        intent: ProgrammeActionIntentV1,
        *,
        driver_lease: ProgrammeDriverLeaseV1,
        at_utc: str,
        ttl_seconds: int = 900,
    ) -> ProgrammeActionClaimV1:
        self.persist_action_intent(intent)
        with advisory_lock(self.lock_path):
            settlement = self.action_settlement(intent.action_intent_id())
            if settlement is not None:
                raise ActorError(
                    "action intent is already settled",
                    code="programme_action_already_settled",
                    related_records=[intent.action_intent_id()],
                )
            current_driver = self.driver_lease_head()
            if (
                current_driver is None
                or canonical_sha256(current_driver) != driver_lease.lease_digest()
                or _parse_utc(at_utc) >= _parse_utc(driver_lease.expires_at_utc)
            ):
                raise ActorError(
                    "action claim is fenced by another Programme Driver",
                    code="programme_action_fenced",
                )
            claims = self.action_claims(intent.action_intent_id())
            prior = claims[-1] if claims else None
            if prior is not None and _parse_utc(str(prior["expires_at_utc"])) > _parse_utc(
                at_utc
            ):
                raise ActorError(
                    "another fresh action claim owns the effect boundary",
                    code="programme_action_in_flight",
                    related_records=[intent.action_intent_id()],
                )
            generation = len(claims) + 1
            prior_digest = None if prior is None else canonical_sha256(prior)
            execution_fence = canonical_sha256(
                {
                    "action_intent_digest": intent.action_digest().to_dict(),
                    "driver_lease_digest": driver_lease.lease_digest().to_dict(),
                    "claim_generation": generation,
                    "prior_claim_digest": (
                        None if prior_digest is None else prior_digest.to_dict()
                    ),
                    "claimed_at_utc": at_utc,
                }
            )
            claim_expires = min(
                _parse_utc(_add_seconds(at_utc, max(1, int(ttl_seconds)))),
                _parse_utc(driver_lease.expires_at_utc),
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            claim = ProgrammeActionClaimV1(
                action_intent_id=intent.action_intent_id(),
                action_intent_digest=intent.action_digest(),
                programme_id=intent.programme_id,
                authorization_digest=intent.authorization_digest,
                driver_lease_digest=driver_lease.lease_digest(),
                driver_generation=driver_lease.generation,
                execution_fencing_token_digest=execution_fence,
                claim_generation=generation,
                prior_claim_digest=prior_digest,
                claimed_at_utc=at_utc,
                expires_at_utc=claim_expires,
            )
            _write_immutable(
                self.action_claims_dir
                / intent.action_intent_id()
                / f"{generation:06d}-{claim.claim_digest().hex}.json",
                claim.to_dict(),
                label="ProgrammeActionClaim",
            )
            atomic_write_json(
                self.action_claim_heads_dir / f"{intent.action_intent_id()}.json",
                {
                    "schema_version": "crca_programme_action_claim_head_v1",
                    "claim_digest": claim.claim_digest().hex,
                },
            )
            return claim

    def action_intent(self, action_intent_id: str) -> dict[str, Any]:
        return read_json(
            self.action_intents_dir / f"{action_intent_id}.json",
            name="ProgrammeActionIntent",
        )

    def action_settlement(self, action_intent_id: str) -> dict[str, Any] | None:
        path = self.action_settlements_dir / f"{action_intent_id}.json"
        return read_json(path, name="ProgrammeActionSettlement") if path.is_file() else None

    def settle_action(self, settlement: ProgrammeActionSettlementV1) -> Path:
        intent = self.action_intent(settlement.action_intent_id)
        if canonical_sha256(intent) != settlement.action_intent_digest:
            raise ActorError(
                "action settlement binds another intent",
                code="programme_action_settlement_invalid",
            )
        claim = self.action_claim_head(settlement.action_intent_id)
        if (
            claim is None
            or canonical_sha256(claim) != settlement.action_claim_digest
            or _content_digest(
                claim.get("execution_fencing_token_digest"),
                field="execution_fencing_token_digest",
            )
            != settlement.execution_fencing_token_digest
            or _parse_utc(settlement.settled_at_utc)
            >= _parse_utc(str(claim.get("expires_at_utc") or ""))
        ):
            raise ActorError(
                "action settlement has no fresh exact action claim",
                code="programme_action_settlement_invalid",
            )
        with advisory_lock(self.lock_path):
            current_claim = self.action_claim_head(settlement.action_intent_id)
            if (
                current_claim is None
                or canonical_sha256(current_claim) != settlement.action_claim_digest
            ):
                raise ActorError(
                    "action settlement was fenced by a successor claim",
                    code="programme_action_settlement_invalid",
                )
            return _write_immutable(
                self.action_settlements_dir
                / f"{settlement.action_intent_id}.json",
                settlement.to_dict(),
                label="ProgrammeActionSettlement",
            )

    def persist_notification_intent(
        self, intent: TerminalNotificationIntentV1
    ) -> Path:
        with advisory_lock(self.lock_path):
            latest = self.latest_transition()
            if latest is None or latest.get("terminal_state") != intent.terminal_state.value:
                raise ActorError(
                    "terminal notification has no matching terminal transition",
                    code="programme_notification_invalid",
                )
            if canonical_sha256(latest) != intent.terminal_transition_digest:
                raise ActorError(
                    "terminal notification binds another terminal transition",
                    code="programme_notification_invalid",
                )
            expected = _terminal_notification_binding_v1(
                self.authorization(), self.authorization_digest(), latest
            )
            for field in (
                "programme_id",
                "terminal_state",
                "terminal_transition_digest",
                "terminal_programme_digest",
                "recipient_digest",
                "subject_digest",
                "body_digest",
            ):
                if getattr(intent, field) != expected[field]:
                    raise ActorError(
                        "terminal notification differs from its canonical binding",
                        code="programme_notification_invalid",
                    )
            return _write_immutable(
                self.notification_intents_dir
                / f"{intent.notification_intent_id()}.json",
                intent.to_dict(),
                label="TerminalNotificationIntent",
            )

    def notification_intents(self) -> list[dict[str, Any]]:
        if not self.notification_intents_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        latest = self.latest_transition()
        if latest is None or latest.get("terminal_state") is None:
            raise ActorError(
                "notification intent exists before a terminal transition",
                code="programme_notification_invalid",
            )
        expected = _terminal_notification_binding_v1(
            self.authorization(), self.authorization_digest(), latest
        )
        for path in sorted(self.notification_intents_dir.glob("*.json")):
            raw = read_json(path, name="TerminalNotificationIntent")
            try:
                intent = TerminalNotificationIntentV1.from_dict(raw)
            except (ProgrammeContractError, TypeError, ValueError) as error:
                raise ActorError(
                    "terminal notification intent is invalid",
                    code="programme_notification_invalid",
                ) from error
            if intent.to_dict() != raw or path.stem != intent.notification_intent_id():
                raise ActorError(
                    "terminal notification intent is not canonical",
                    code="programme_notification_invalid",
                )
            for field in (
                "programme_id",
                "terminal_state",
                "terminal_transition_digest",
                "terminal_programme_digest",
                "recipient_digest",
                "subject_digest",
                "body_digest",
            ):
                if getattr(intent, field) != expected[field]:
                    raise ActorError(
                        "terminal notification differs from its canonical binding",
                        code="programme_notification_invalid",
                    )
            rows.append(raw)
        return rows

    def notification_deliveries(self, notification_intent_id: str) -> list[dict[str, Any]]:
        directory = self.notification_deliveries_dir / notification_intent_id
        if not directory.is_dir():
            return []
        return [
            read_json(path, name="TerminalNotificationDelivery")
            for path in sorted(directory.glob("*.json"))
        ]

    def record_notification_delivery(
        self, delivery: TerminalNotificationDeliveryV1
    ) -> Path:
        intent_path = (
            self.notification_intents_dir
            / f"{delivery.notification_intent_id}.json"
        )
        if not intent_path.is_file():
            raise ActorError(
                "notification delivery has no durable intent",
                code="programme_notification_invalid",
            )
        intent = read_json(intent_path, name="TerminalNotificationIntent")
        if canonical_sha256(intent) != delivery.notification_intent_digest:
            raise ActorError(
                "notification delivery binds another intent",
                code="programme_notification_invalid",
            )
        with advisory_lock(self.lock_path):
            expected_attempt = len(
                self.notification_deliveries(delivery.notification_intent_id)
            ) + 1
            if delivery.delivery_attempt != expected_attempt:
                raise ActorError(
                    "notification delivery attempt is not monotonic",
                    code="programme_notification_invalid",
                )
            path = (
                self.notification_deliveries_dir
                / delivery.notification_intent_id
                / f"{delivery.delivery_attempt:06d}-{delivery.delivery_digest().hex}.json"
            )
            return _write_immutable(
                path, delivery.to_dict(), label="TerminalNotificationDelivery"
            )

    def driver_lease_head(self) -> dict[str, Any] | None:
        if not self.driver_head_path.is_file():
            return None
        head = read_json(self.driver_head_path, name="ProgrammeDriverLeaseHead")
        path = self.driver_leases_dir / f"{str(head.get('lease_digest') or '')}.json"
        if not path.is_file():
            raise ActorError(
                "driver lease head points to missing lease",
                code="programme_driver_lease_invalid",
            )
        lease = read_json(path, name="ProgrammeDriverLease")
        if canonical_sha256(lease).hex != head.get("lease_digest"):
            raise ActorError(
                "driver lease head digest mismatch",
                code="programme_driver_lease_invalid",
            )
        return lease

    def claim_driver_lease(
        self,
        *,
        driver_owner_id: str,
        driver_owner_digest: ContentDigest,
        process_id: int,
        process_start_identity: str,
        at_utc: str,
        ttl_seconds: int = 900,
    ) -> ProgrammeDriverLeaseV1:
        with advisory_lock(self.lock_path):
            current = self.driver_lease_head()
            prior_digest = None if current is None else canonical_sha256(current)
            generation = 1
            if current is not None:
                active = _parse_utc(str(current["expires_at_utc"])) > _parse_utc(at_utc)
                current_process_live = _local_process_start_identity(
                    int(current.get("process_id") or 0)
                ) == str(current.get("process_start_identity") or "")
                same_owner = (
                    current.get("driver_owner_id") == driver_owner_id
                    and current.get("driver_owner_digest")
                    == driver_owner_digest.to_dict()
                    and int(current.get("process_id") or 0) == process_id
                    and current.get("process_start_identity")
                    == process_start_identity
                )
                if active and not same_owner and current_process_live:
                    raise ActorError(
                        "another fresh Programme Driver lease is active",
                        code="programme_driver_lease_conflict",
                    )
                generation = int(current.get("generation") or 0) + (
                    0 if active and same_owner and current_process_live else 1
                )
            fence = canonical_sha256(
                {
                    "programme_id": self.authorization().get("programme_id"),
                    "authorization_digest": self.authorization_digest().to_dict(),
                    "driver_owner_id": driver_owner_id,
                    "driver_owner_digest": driver_owner_digest.to_dict(),
                    "process_id": process_id,
                    "process_start_identity": process_start_identity,
                    "generation": generation,
                    "prior_lease_digest": (
                        None if prior_digest is None else prior_digest.to_dict()
                    ),
                    "granted_at_utc": at_utc,
                }
            )
            lease = ProgrammeDriverLeaseV1(
                programme_id=str(self.authorization()["programme_id"]),
                authorization_digest=self.authorization_digest(),
                driver_owner_id=driver_owner_id,
                driver_owner_digest=driver_owner_digest,
                process_id=process_id,
                process_start_identity=process_start_identity,
                generation=generation,
                fencing_token_digest=fence,
                prior_lease_digest=prior_digest,
                granted_at_utc=at_utc,
                heartbeat_at_utc=at_utc,
                expires_at_utc=_add_seconds(at_utc, ttl_seconds),
            )
            digest = lease.lease_digest().hex
            _write_immutable(
                self.driver_leases_dir / f"{digest}.json",
                lease.to_dict(),
                label="ProgrammeDriverLease",
            )
            atomic_write_json(
                self.driver_head_path,
                {
                    "schema_version": "crca_programme_driver_lease_head_v1",
                    "lease_digest": digest,
                },
            )
            return lease

    def heartbeat_driver_lease(
        self,
        lease: ProgrammeDriverLeaseV1,
        *,
        at_utc: str,
        ttl_seconds: int = 900,
    ) -> ProgrammeDriverLeaseV1:
        with advisory_lock(self.lock_path):
            current = self.driver_lease_head()
            if current is None or canonical_sha256(current) != lease.lease_digest():
                raise ActorError(
                    "driver heartbeat is fenced by another lease",
                    code="programme_driver_lease_fenced",
                )
            if _parse_utc(at_utc) >= _parse_utc(lease.expires_at_utc):
                raise ActorError(
                    "driver lease expired before heartbeat",
                    code="programme_driver_lease_expired",
                )
            successor = ProgrammeDriverLeaseV1(
                programme_id=lease.programme_id,
                authorization_digest=lease.authorization_digest,
                driver_owner_id=lease.driver_owner_id,
                driver_owner_digest=lease.driver_owner_digest,
                process_id=lease.process_id,
                process_start_identity=lease.process_start_identity,
                generation=lease.generation,
                fencing_token_digest=lease.fencing_token_digest,
                prior_lease_digest=lease.lease_digest(),
                granted_at_utc=lease.granted_at_utc,
                heartbeat_at_utc=at_utc,
                expires_at_utc=_add_seconds(at_utc, ttl_seconds),
            )
            digest = successor.lease_digest().hex
            _write_immutable(
                self.driver_leases_dir / f"{digest}.json",
                successor.to_dict(),
                label="ProgrammeDriverLease",
            )
            atomic_write_json(
                self.driver_head_path,
                {
                    "schema_version": "crca_programme_driver_lease_head_v1",
                    "lease_digest": digest,
                },
            )
        return successor


class ResourceStartLeaseLedgerV1:
    """Scheduler-owned immutable grants and release/reclaim event chains."""

    def __init__(
        self,
        root: str | Path,
        *,
        allow_test_legacy_gpu_migration: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if (
            allow_test_legacy_gpu_migration
            and self.root
            == scheduler_root(canonical_host_actor_state_root()).resolve()
        ):
            raise ActorError(
                "legacy GPU migration test authority cannot target canonical state",
                code="resource_start_lease_invalid",
            )
        self._allow_test_legacy_gpu_migration = bool(
            allow_test_legacy_gpu_migration
        )
        self.leases_dir = self.root / "resource-start-leases"
        self.events_dir = self.root / "resource-start-lease-events"
        self.lock_path = self.root / "resource-start-leases.lock"
        self.legacy_gpu_migration_root = (
            self.root / "launchpad-attempts" / "legacy-gpu-migrations"
        )
        self.legacy_gpu_migration_proofs_dir = (
            self.legacy_gpu_migration_root / "proofs"
        )
        self.legacy_gpu_migration_authorizations_dir = (
            self.legacy_gpu_migration_root / "authorizations"
        )

    @staticmethod
    def _legacy_migration_key(value: str) -> str:
        return canonical_sha256({"value": value}).hex

    @staticmethod
    def _event_v3_from_dict(row: Mapping[str, Any]) -> ResourceStartLeaseEventV3:
        try:
            return ResourceStartLeaseEventV3(
                migration_id=str(row["migration_id"]),
                attempt_id=str(row["attempt_id"]),
                resource_lease_id=str(row["resource_lease_id"]),
                lease_digest=_content_digest(
                    row["lease_digest"], field="lease_digest"
                ),
                code=ResourceStartLeaseEventCodeV1(str(row["code"])),
                scheduler_owner_id=str(row["scheduler_owner_id"]),
                fencing_token_digest=_content_digest(
                    row["fencing_token_digest"], field="fencing_token_digest"
                ),
                event_at_utc=str(row["event_at_utc"]),
                reason=str(row["reason"]),
                prior_event_digest=_content_digest(
                    row["prior_event_digest"], field="prior_event_digest"
                ),
                migration_manifest_digest=_content_digest(
                    row["migration_manifest_digest"],
                    field="migration_manifest_digest",
                ),
                migration_row_digest=_content_digest(
                    row["migration_row_digest"], field="migration_row_digest"
                ),
                reclaim_proof_digest=_content_digest(
                    row["reclaim_proof_digest"], field="reclaim_proof_digest"
                ),
                migration_authorization_digest=_content_digest(
                    row["migration_authorization_digest"],
                    field="migration_authorization_digest",
                ),
                successor_lease_digest=None,
                schema_version=str(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, ProgrammeContractError) as error:
            raise ActorError(
                "legacy GPU migration resource event is invalid",
                code="resource_start_lease_invalid",
            ) from error

    def _validate_legacy_gpu_migration_authorization(
        self,
        *,
        lease_id: str,
        lease: Mapping[str, Any],
        event: ResourceStartLeaseEventV3,
    ) -> Path:
        if event.resource_lease_id != lease_id:
            raise ActorError(
                "legacy GPU migration event binds another resource lease",
                code="resource_start_lease_invalid",
            )
        lease_contract = self._lease_from_dict(lease)
        if (
            not isinstance(lease_contract, LaunchpadResourceStartLeaseV1)
            or event.attempt_id != lease_contract.attempt_id
            or event.lease_digest != lease_contract.lease_digest()
        ):
            raise ActorError(
                "legacy GPU migration event differs from its Launchpad lease",
                code="resource_start_lease_invalid",
            )
        if not self._allow_test_legacy_gpu_migration and (
            event.migration_manifest_digest.hex
            != CANONICAL_LEGACY_GPU_MIGRATION_MANIFEST_SHA256
            or CANONICAL_LEGACY_GPU_MIGRATION_ROWS.get(lease_id)
            != event.migration_row_digest.hex
        ):
            raise ActorError(
                "legacy GPU migration event is absent from the pinned cutover",
                code="resource_start_lease_invalid",
            )
        authorization_path = (
            self.legacy_gpu_migration_authorizations_dir
            / self._legacy_migration_key(event.attempt_id)
            / f"{event.migration_authorization_digest.hex}.json"
        )
        if not authorization_path.is_file() or authorization_path.is_symlink():
            raise ActorError(
                "legacy GPU migration authorization is unavailable",
                code="resource_start_lease_invalid",
            )
        authorization = read_json(
            authorization_path, name="LegacyGpuMigrationAuthorization"
        )
        expected_fields = {
            "schema_version",
            "migration_id",
            "migration_manifest_digest",
            "migration_row_digest",
            "attempt_id",
            "resource_lease_id",
            "lease_digest",
            "resource_started_event_digest",
            "reclaim_proof_digest",
            "continuation_owner_dead_event_digest",
            "resource_event_intent_digest",
            "authorized_at_utc",
            "authorization_sha256",
        }
        core = {
            key: value
            for key, value in authorization.items()
            if key != "authorization_sha256"
        }
        authorization_digest = canonical_sha256(core)
        proof_path = (
            self.legacy_gpu_migration_proofs_dir
            / self._legacy_migration_key(event.attempt_id)
            / f"{event.reclaim_proof_digest.hex}.json"
        )
        if (
            set(authorization) != expected_fields
            or authorization.get("schema_version")
            != LEGACY_GPU_MIGRATION_AUTHORIZATION_SCHEMA_VERSION
            or authorization.get("authorization_sha256")
            != authorization_digest.hex
            or authorization_digest != event.migration_authorization_digest
            or authorization.get("migration_id") != event.migration_id
            or authorization.get("migration_manifest_digest")
            != event.migration_manifest_digest.to_dict()
            or authorization.get("migration_row_digest")
            != event.migration_row_digest.to_dict()
            or authorization.get("attempt_id") != event.attempt_id
            or authorization.get("resource_lease_id") != lease_id
            or authorization.get("lease_digest") != event.lease_digest.to_dict()
            or authorization.get("resource_started_event_digest")
            != event.prior_event_digest.to_dict()
            or authorization.get("reclaim_proof_digest")
            != event.reclaim_proof_digest.to_dict()
            or authorization.get("resource_event_intent_digest")
            != canonical_sha256(event.authorization_intent()).to_dict()
            or authorization.get("authorized_at_utc") != event.event_at_utc
            or not proof_path.is_file()
            or proof_path.is_symlink()
        ):
            raise ActorError(
                "legacy GPU migration authorization is invalid",
                code="resource_start_lease_invalid",
            )
        proof = read_json(proof_path, name="LegacyGpuMigrationReclaimProof")
        proof_core = {
            key: value for key, value in proof.items() if key != "proof_sha256"
        }
        if (
            proof.get("proof_sha256") != event.reclaim_proof_digest.hex
            or canonical_sha256(proof_core) != event.reclaim_proof_digest
            or proof.get("attempt_id") != event.attempt_id
            or proof.get("migration_manifest_sha256")
            != event.migration_manifest_digest.hex
            or proof.get("migration_row_sha256")
            != event.migration_row_digest.hex
        ):
            raise ActorError(
                "legacy GPU migration proof identity is invalid",
                code="resource_start_lease_invalid",
            )
        return authorization_path

    def leases(self) -> list[dict[str, Any]]:
        if not self.leases_dir.is_dir():
            return []
        return [
            read_json(path, name="ResourceStartLease")
            for path in sorted(self.leases_dir.glob("*.json"))
        ]

    def events(self, lease_id: str) -> list[dict[str, Any]]:
        directory = self.events_dir / lease_id
        if not directory.is_dir():
            return []
        rows = [
            read_json(path, name="ResourceStartLeaseEvent")
            for path in sorted(directory.glob("*.json"))
        ]
        expected: ContentDigest | None = None
        for row in rows:
            if row.get("schema_version") == "crca_resource_start_lease_event_v3":
                event = self._event_v3_from_dict(row)
                if row != event.to_dict():
                    raise ActorError(
                        "legacy GPU migration resource event is not canonical",
                        code="resource_start_lease_invalid",
                    )
                lease_path = self.leases_dir / f"{lease_id}.json"
                if not lease_path.is_file() or lease_path.is_symlink():
                    raise ActorError(
                        "legacy GPU migration resource lease is unavailable",
                        code="resource_start_lease_invalid",
                    )
                self._validate_legacy_gpu_migration_authorization(
                    lease_id=lease_id,
                    lease=read_json(lease_path, name="ResourceStartLease"),
                    event=event,
                )
            prior = row.get("prior_event_digest")
            observed = None if prior is None else _content_digest(
                prior, field="prior_event_digest"
            )
            if observed != expected:
                raise ActorError(
                    "resource-start event chain is invalid",
                    code="resource_start_lease_invalid",
                )
            expected = canonical_sha256(row)
        return rows

    @staticmethod
    def _lease_from_dict(
        row: Mapping[str, Any],
    ) -> ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1:
        if row.get("schema_version") == "crca_launchpad_resource_start_lease_v1":
            try:
                return LaunchpadResourceStartLeaseV1.from_dict(row)
            except (ProgrammeContractError, TypeError, ValueError) as error:
                raise ActorError(
                    "Launchpad resource-start lease is invalid",
                    code="resource_start_lease_invalid",
                ) from error
        return ResourceStartLeaseV1(
            lease_id=str(row["lease_id"]),
            scheduler_owner_id=str(row["scheduler_owner_id"]),
            scheduler_owner_digest=_content_digest(
                row["scheduler_owner_digest"], field="scheduler_owner_digest"
            ),
            assignment_lease_id=str(row["assignment_lease_id"]),
            assignment_lease_digest=_content_digest(
                row["assignment_lease_digest"], field="assignment_lease_digest"
            ),
            attempt_admission_request_digest=_content_digest(
                row["attempt_admission_request_digest"],
                field="attempt_admission_request_digest",
            ),
            resource_ids=tuple(row["resource_ids"]),
            resource_scope_digest=_content_digest(
                row["resource_scope_digest"], field="resource_scope_digest"
            ),
            resource_observation_id=str(row["resource_observation_id"]),
            resource_observation_digest=_content_digest(
                row["resource_observation_digest"],
                field="resource_observation_digest",
            ),
            resource_recommendation_id=str(row["resource_recommendation_id"]),
            resource_recommendation_digest=_content_digest(
                row["resource_recommendation_digest"],
                field="resource_recommendation_digest",
            ),
            live_provider_check_digest=_content_digest(
                row["live_provider_check_digest"],
                field="live_provider_check_digest",
            ),
            generation=int(row["generation"]),
            fencing_token_digest=_content_digest(
                row["fencing_token_digest"], field="fencing_token_digest"
            ),
            prior_lease_digest=(
                None
                if row.get("prior_lease_digest") is None
                else _content_digest(
                    row["prior_lease_digest"], field="prior_lease_digest"
                )
            ),
            issued_at_utc=str(row["issued_at_utc"]),
            start_before_utc=str(row["start_before_utc"]),
            expires_at_utc=str(row["expires_at_utc"]),
        )

    @staticmethod
    def _lease_epoch(
        lease: ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1,
    ) -> int:
        if isinstance(lease, LaunchpadResourceStartLeaseV1):
            return lease.lease_epoch
        return lease.generation

    def active_leases(
        self, *, at_utc: str
    ) -> list[ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1]:
        active: list[ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1] = []
        observed = _parse_utc(at_utc)
        for row in self.leases():
            lease = self._lease_from_dict(row)
            events = self.events(lease.lease_id)
            terminal = events[-1].get("code") if events else None
            if terminal in {
                "RELEASED",
                "EXPIRED",
                "RECLAIMED",
                "OWNER_DEAD_RECLAIMED",
                "LEGACY_GPU_MIGRATION_RECLAIMED",
            }:
                continue
            if terminal == "STARTED":
                if isinstance(lease, LaunchpadResourceStartLeaseV1):
                    active.append(lease)
                continue
            if observed >= _parse_utc(lease.expires_at_utc):
                raise ActorError(
                    "expired resource-start lease lacks a durable EXPIRED event",
                    code="resource_start_lease_expiry_unreconciled",
                    related_records=[lease.lease_id],
                )
            active.append(lease)
        return active

    def _latest_overlapping_lease(
        self, resource_ids: tuple[str, ...]
    ) -> ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1 | None:
        resources = set(resource_ids)
        candidates = [
            self._lease_from_dict(row)
            for row in self.leases()
            if resource_ids_conflict_v1(
                resources, row.get("resource_ids") or []
            )
        ]
        return max(candidates, key=self._lease_epoch) if candidates else None

    def _append_event_unlocked(
        self,
        lease: Mapping[str, Any],
        event: (
            ResourceStartLeaseEventV1
            | ResourceStartLeaseEventV2
            | ResourceStartLeaseEventV3
        ),
    ) -> Path:
        lease_id = str(lease.get("lease_id") or "")
        lease_contract = self._lease_from_dict(lease)
        if lease_contract.lease_digest() != event.lease_digest:
            raise ActorError(
                "resource-start event binds another lease",
                code="resource_start_lease_invalid",
            )
        if isinstance(event, ResourceStartLeaseEventV3):
            self._validate_legacy_gpu_migration_authorization(
                lease_id=lease_id,
                lease=lease,
                event=event,
            )
        rows = self.events(lease_id)
        if (
            isinstance(lease_contract, LaunchpadResourceStartLeaseV1)
            and rows
            and rows[-1] == event.to_dict()
        ):
            return (
                self.events_dir
                / lease_id
                / f"{len(rows):06d}-{event.event_digest().hex}.json"
            )
        prior_code = None if not rows else str(rows[-1].get("code") or "")
        launchpad_release = (
            isinstance(lease_contract, LaunchpadResourceStartLeaseV1)
            and prior_code == ResourceStartLeaseEventCodeV1.STARTED.value
            and event.code is ResourceStartLeaseEventCodeV1.RELEASED
        )
        expiry_reclaim = (
            prior_code == ResourceStartLeaseEventCodeV1.EXPIRED.value
            and event.code is ResourceStartLeaseEventCodeV1.RECLAIMED
        )
        dead_owner_reclaim = (
            isinstance(lease_contract, LaunchpadResourceStartLeaseV1)
            and prior_code == ResourceStartLeaseEventCodeV1.STARTED.value
            and event.code is ResourceStartLeaseEventCodeV1.OWNER_DEAD_RECLAIMED
            and isinstance(event, ResourceStartLeaseEventV2)
        )
        legacy_gpu_migration_reclaim = (
            isinstance(lease_contract, LaunchpadResourceStartLeaseV1)
            and prior_code == ResourceStartLeaseEventCodeV1.STARTED.value
            and event.code
            is ResourceStartLeaseEventCodeV1.LEGACY_GPU_MIGRATION_RECLAIMED
            and isinstance(event, ResourceStartLeaseEventV3)
        )
        if prior_code and not (
            launchpad_release
            or expiry_reclaim
            or dead_owner_reclaim
            or legacy_gpu_migration_reclaim
        ):
            raise ActorError(
                "resource-start lease already has a terminal event",
                code="resource_start_lease_terminal",
            )
        expected = None if not rows else canonical_sha256(rows[-1])
        if event.prior_event_digest != expected:
            raise ActorError(
                "resource-start event prior digest is not canonical",
                code="resource_start_lease_invalid",
            )
        if event.code is ResourceStartLeaseEventCodeV1.RECLAIMED:
            if event.successor_lease_digest is None:
                raise ActorError(
                    "resource-start reclaim lacks a successor",
                    code="resource_start_lease_invalid",
                )
            successors = [
                self._lease_from_dict(row)
                for row in self.leases()
                if canonical_sha256(row) == event.successor_lease_digest
            ]
            if len(successors) != 1:
                raise ActorError(
                    "resource-start reclaim successor is unavailable",
                    code="resource_start_lease_invalid",
                )
            successor = successors[0]
            if (
                successor.prior_lease_digest != lease_contract.lease_digest()
                or self._lease_epoch(successor) <= self._lease_epoch(lease_contract)
                or not set(successor.resource_ids).intersection(
                    lease_contract.resource_ids
                )
                or event.scheduler_owner_id != successor.scheduler_owner_id
                or event.fencing_token_digest != successor.fencing_token_digest
            ):
                raise ActorError(
                    "resource-start reclaim successor binding is invalid",
                    code="resource_start_lease_invalid",
                )
        else:
            if (
                event.scheduler_owner_id != lease_contract.scheduler_owner_id
                or event.fencing_token_digest
                != lease_contract.fencing_token_digest
            ):
                raise ActorError(
                    "resource-start event owner or fence differs from its lease",
                    code="resource_start_lease_invalid",
                )
            if event.code is ResourceStartLeaseEventCodeV1.EXPIRED and _parse_utc(
                event.event_at_utc
            ) < _parse_utc(lease_contract.expires_at_utc):
                raise ActorError(
                    "resource-start lease cannot expire before its TTL",
                    code="resource_start_lease_invalid",
                )
            if (
                launchpad_release
                or dead_owner_reclaim
                or legacy_gpu_migration_reclaim
            ) and _parse_utc(
                event.event_at_utc
            ) < _parse_utc(
                str(rows[-1]["event_at_utc"])
            ):
                raise ActorError(
                    "resource occupancy release precedes its STARTED event",
                    code="resource_start_lease_invalid",
                )
        directory = self.events_dir / lease_id
        return _write_immutable(
            directory / f"{len(rows) + 1:06d}-{event.event_digest().hex}.json",
            event.to_dict(),
            label="ResourceStartLeaseEvent",
        )

    def _expire_stale_unlocked(self, *, at_utc: str) -> list[Path]:
        observed = _parse_utc(at_utc)
        written: list[Path] = []
        for row in self.leases():
            lease = self._lease_from_dict(row)
            events = self.events(lease.lease_id)
            if events or observed < _parse_utc(lease.expires_at_utc):
                continue
            event = ResourceStartLeaseEventV1(
                lease_digest=lease.lease_digest(),
                code=ResourceStartLeaseEventCodeV1.EXPIRED,
                scheduler_owner_id=lease.scheduler_owner_id,
                fencing_token_digest=lease.fencing_token_digest,
                event_at_utc=at_utc,
                reason="Scheduler reconciled an unused start lease after TTL",
                prior_event_digest=None,
                successor_lease_digest=None,
            )
            written.append(self._append_event_unlocked(row, event))
        return written

    def reconcile_expired(self, *, at_utc: str) -> list[Path]:
        with advisory_lock(self.lock_path):
            return self._expire_stale_unlocked(at_utc=at_utc)

    def next_grant_fields(
        self, *, resource_ids: tuple[str, ...], at_utc: str
    ) -> tuple[int, ContentDigest | None]:
        with advisory_lock(self.lock_path):
            self._expire_stale_unlocked(at_utc=at_utc)
            prior = self._latest_overlapping_lease(resource_ids)
            return (
                1 if prior is None else self._lease_epoch(prior) + 1,
                None if prior is None else prior.lease_digest(),
            )

    def _grant_unlocked(
        self, lease: ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1
    ) -> Path:
        exact_path = self.leases_dir / f"{lease.lease_id}.json"
        if (
            isinstance(lease, LaunchpadResourceStartLeaseV1)
            and exact_path.is_file()
            and not exact_path.is_symlink()
        ):
            existing = read_json(exact_path, name="ResourceStartLease")
            if existing == lease.to_dict():
                return exact_path
            raise ActorError(
                "Launchpad resource-start lease ID has conflicting bytes",
                code="resource_start_lease_invalid",
            )
        self._expire_stale_unlocked(at_utc=lease.issued_at_utc)
        prior = self._latest_overlapping_lease(lease.resource_ids)
        expected_generation = 1 if prior is None else self._lease_epoch(prior) + 1
        expected_prior = None if prior is None else prior.lease_digest()
        if (
            self._lease_epoch(lease) != expected_generation
            or lease.prior_lease_digest != expected_prior
        ):
            raise ActorError(
                "resource-start successor generation or prior digest is stale",
                code="resource_start_lease_stale_grant",
            )
        require_exclusive_resource_start_lease_v1(
            lease, self.active_leases(at_utc=lease.issued_at_utc)
        )
        written = _write_immutable(
            exact_path,
            lease.to_dict(),
            label="ResourceStartLease",
        )
        if prior is not None:
            prior_events = self.events(prior.lease_id)
            if (
                prior_events
                and prior_events[-1].get("code")
                == ResourceStartLeaseEventCodeV1.EXPIRED.value
            ):
                reclaim = ResourceStartLeaseEventV1(
                    lease_digest=prior.lease_digest(),
                    code=ResourceStartLeaseEventCodeV1.RECLAIMED,
                    scheduler_owner_id=lease.scheduler_owner_id,
                    fencing_token_digest=lease.fencing_token_digest,
                    event_at_utc=lease.issued_at_utc,
                    reason="Expired start lease reclaimed by exact successor",
                    prior_event_digest=canonical_sha256(prior_events[-1]),
                    successor_lease_digest=lease.lease_digest(),
                )
                prior_row = read_json(
                    self.leases_dir / f"{prior.lease_id}.json",
                    name="ResourceStartLease",
                )
                self._append_event_unlocked(prior_row, reclaim)
        return written

    def grant(
        self, lease: ResourceStartLeaseV1 | LaunchpadResourceStartLeaseV1
    ) -> Path:
        with advisory_lock(self.lock_path):
            return self._grant_unlocked(lease)

    def grant_first_available_launchpad(
        self,
        *,
        candidate_resource_ids: tuple[tuple[str, ...], ...],
        at_utc: str,
        lease_factory: Callable[
            [tuple[str, ...], int, ContentDigest | None],
            LaunchpadResourceStartLeaseV1,
        ],
    ) -> tuple[LaunchpadResourceStartLeaseV1, Path]:
        """Atomically choose and grant the first nonconflicting candidate set."""

        if (
            not candidate_resource_ids
            or len(candidate_resource_ids) != len(set(candidate_resource_ids))
            or any(
                not candidate
                or candidate != tuple(sorted(set(candidate)))
                for candidate in candidate_resource_ids
            )
        ):
            raise ActorError(
                "resource candidate sets must be unique, sorted, and non-empty",
                code="resource_start_lease_invalid",
            )
        with advisory_lock(self.lock_path):
            self._expire_stale_unlocked(at_utc=at_utc)
            active = self.active_leases(at_utc=at_utc)
            for resource_ids in candidate_resource_ids:
                if any(
                    resource_ids_conflict_v1(resource_ids, lease.resource_ids)
                    for lease in active
                ):
                    continue
                prior = self._latest_overlapping_lease(resource_ids)
                epoch = 1 if prior is None else self._lease_epoch(prior) + 1
                prior_digest = None if prior is None else prior.lease_digest()
                lease = lease_factory(resource_ids, epoch, prior_digest)
                if (
                    not isinstance(lease, LaunchpadResourceStartLeaseV1)
                    or lease.resource_ids != resource_ids
                    or lease.issued_at_utc != at_utc
                ):
                    raise ActorError(
                        "resource candidate lease factory changed its allocation",
                        code="resource_start_lease_invalid",
                    )
                return lease, self._grant_unlocked(lease)
        raise ActorError(
            "all Scheduler resource candidates are occupied",
            code="launchpad_attempt_resource_wait",
        )

    def append_event(
        self,
        lease_id: str,
        event: (
            ResourceStartLeaseEventV1
            | ResourceStartLeaseEventV2
            | ResourceStartLeaseEventV3
        ),
    ) -> Path:
        if isinstance(event, ResourceStartLeaseEventV3):
            raise ActorError(
                "legacy GPU migration events require the bound authorization path",
                code="resource_start_lease_invalid",
            )
        path = self.leases_dir / f"{lease_id}.json"
        if not path.is_file():
            raise ActorError(
                "resource-start lease is missing",
                code="resource_start_lease_missing",
            )
        lease = read_json(path, name="ResourceStartLease")
        with advisory_lock(self.lock_path):
            return self._append_event_unlocked(lease, event)

    def _append_authorized_legacy_gpu_migration_event(
        self,
        lease_id: str,
        event: ResourceStartLeaseEventV3,
    ) -> Path:
        """Append one receipt-bound event through the sole migration seam.

        The generic event API deliberately rejects V3.  This narrow method
        independently validates the durable authorization and proof referents
        before the resource can become allocatable.
        """

        if not isinstance(event, ResourceStartLeaseEventV3):
            raise ActorError(
                "legacy GPU migration requires a V3 resource event",
                code="resource_start_lease_invalid",
            )
        path = self.leases_dir / f"{lease_id}.json"
        if not path.is_file() or path.is_symlink():
            raise ActorError(
                "resource-start lease is missing",
                code="resource_start_lease_missing",
            )
        with advisory_lock(self.lock_path):
            lease = read_json(path, name="ResourceStartLease")
            return self._append_event_unlocked(lease, event)


ActionObserver = Callable[
    [ProgrammeActionIntentV1, ProgrammeActionClaimV1], Mapping[str, Any]
]
ActionExecutor = Callable[
    [ProgrammeActionIntentV1, ProgrammeActionClaimV1], Mapping[str, Any]
]
NotificationSender = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def reconcile_action_v1(
    journal: ProgrammeJournalV1,
    intent: ProgrammeActionIntentV1,
    *,
    driver_lease: ProgrammeDriverLeaseV1,
    observe: ActionObserver,
    execute: ActionExecutor,
    at_utc: str,
    action_claim_ttl_seconds: int = 900,
) -> dict[str, Any]:
    """Reconcile one exact action without transcript or model state."""

    current_driver = journal.driver_lease_head()
    if (
        current_driver is None
        or canonical_sha256(current_driver) != driver_lease.lease_digest()
        or driver_lease.authorization_digest != journal.authorization_digest()
        or driver_lease.programme_id != intent.programme_id
        or _parse_utc(at_utc) >= _parse_utc(driver_lease.expires_at_utc)
    ):
        raise ActorError(
            "action reconciliation is fenced by another Programme Driver",
            code="programme_action_fenced",
        )
    known_creation_fence = False
    if journal.driver_leases_dir.is_dir():
        for path in journal.driver_leases_dir.glob("*.json"):
            row = read_json(path, name="ProgrammeDriverLease")
            if (
                row.get("programme_id") == intent.programme_id
                and _content_digest(
                    row.get("authorization_digest"), field="authorization_digest"
                )
                == intent.authorization_digest
                and _content_digest(
                    row.get("fencing_token_digest"), field="fencing_token_digest"
                )
                == intent.fencing_token_digest
            ):
                known_creation_fence = True
                break
    if not known_creation_fence:
        raise ActorError(
            "action intent was not created by a known driver fence",
            code="programme_action_fenced",
        )
    journal.persist_action_intent(intent)
    existing = journal.action_settlement(intent.action_intent_id())
    if existing is not None:
        return {"status": "SETTLED", "settlement": existing, "effect_executed": False}
    try:
        claim = journal.claim_action(
            intent,
            driver_lease=driver_lease,
            at_utc=at_utc,
            ttl_seconds=action_claim_ttl_seconds,
        )
    except ActorError as error:
        if error.code == "programme_action_already_settled":
            settlement = journal.action_settlement(intent.action_intent_id())
            if settlement is None:
                raise ActorError(
                    "settlement disappeared while acquiring an action claim",
                    code="programme_action_settlement_invalid",
                ) from error
            return {
                "status": "SETTLED",
                "settlement": settlement,
                "effect_executed": False,
            }
        if error.code == "programme_action_in_flight":
            return {
                "status": "ACTION_IN_FLIGHT",
                "effect_executed": False,
                "action_intent_id": intent.action_intent_id(),
            }
        raise
    observation = dict(observe(intent, claim))
    state = str(observation.get("state") or "")
    if state not in ACTION_OBSERVATION_STATES:
        raise ActorError(
            "action owner returned an unsupported observation",
            code="programme_action_observation_invalid",
        )
    if state == "LIVE":
        observed_fence = _content_digest(
            observation.get("fencing_token_digest"),
            field="fencing_token_digest",
        )
        if observed_fence != claim.execution_fencing_token_digest:
            raise ActorError(
                "live action owner is fenced by another claim",
                code="programme_action_fenced",
            )
        return {
            "status": "ADOPTED_LIVE_OWNER",
            "owner_receipt_digest": observation.get("owner_receipt_digest"),
            "action_claim": claim.to_dict(),
            "effect_executed": False,
        }
    if state == "COMPLETED":
        result = observation
        effect_executed = False
        settlement_code = "SETTLED_FROM_OWNER_OBSERVATION"
    else:
        if state == "STALE_FENCED":
            claims = journal.action_claims(intent.action_intent_id())
            if claim.prior_claim_digest is None or len(claims) < 2:
                raise ActorError(
                    "stale action has no predecessor claim",
                    code="programme_action_fenced",
                )
            prior_claim = claims[-2]
            observed_fence = _content_digest(
                observation.get("fencing_token_digest"),
                field="fencing_token_digest",
            )
            if (
                canonical_sha256(prior_claim) != claim.prior_claim_digest
                or observed_fence
                != _content_digest(
                    prior_claim.get("execution_fencing_token_digest"),
                    field="execution_fencing_token_digest",
                )
                or _parse_utc(str(prior_claim.get("expires_at_utc") or ""))
                > _parse_utc(at_utc)
            ):
                raise ActorError(
                    "stale action is fenced by another owner generation",
                    code="programme_action_fenced",
                )
        result = dict(execute(intent, claim))
        effect_executed = True
        settlement_code = (
            "SETTLED_AFTER_FENCED_RECLAIM"
            if state == "STALE_FENCED"
            else "SETTLED_AFTER_EXECUTION"
        )
    effect_digest = _content_digest(
        result.get("effect_receipt_digest"), field="effect_receipt_digest"
    )
    result_digest = _content_digest(
        result.get("result_receipt_digest"), field="result_receipt_digest"
    )
    settlement = ProgrammeActionSettlementV1(
        action_intent_id=intent.action_intent_id(),
        action_intent_digest=intent.action_digest(),
        action_claim_digest=claim.claim_digest(),
        execution_fencing_token_digest=claim.execution_fencing_token_digest,
        effect_receipt_digest=effect_digest,
        result_receipt_digest=result_digest,
        settlement_code=settlement_code,
        settled_at_utc=at_utc,
    )
    journal.settle_action(settlement)
    return {
        "status": "SETTLED",
        "settlement": settlement.to_dict(),
        "action_claim": claim.to_dict(),
        "effect_executed": effect_executed,
    }


def deliver_terminal_notifications_v1(
    journal: ProgrammeJournalV1,
    *,
    send: NotificationSender,
    at_utc: str,
    provider: str,
) -> list[dict[str, Any]]:
    """Deliver each pending terminal intent with truthful delivery semantics."""

    outcomes: list[dict[str, Any]] = []
    for intent in journal.notification_intents():
        intent_id = (
            f"CRCA-NOTIFY-{str(dict(intent['terminal_programme_digest'])['hex']).upper()}"
        )
        prior = journal.notification_deliveries(intent_id)
        if prior:
            outcomes.append(
                {
                    "notification_intent_id": intent_id,
                    "status": "ALREADY_DELIVERED",
                    "delivery_semantics": prior[-1].get("delivery_semantics"),
                }
            )
            continue
        # The send may succeed externally and then raise before a receipt can
        # be written.  The next pass retries, making delivery at-least-once.
        result = dict(send(intent))
        provider_receipt = _content_digest(
            result.get("provider_receipt_digest"),
            field="provider_receipt_digest",
        )
        deduplication_raw = result.get("deduplication_proof")
        deduplication_proof: NotificationDeduplicationProofV1 | None = None
        if deduplication_raw is not None:
            if not isinstance(deduplication_raw, Mapping):
                raise ActorError(
                    "notification deduplication proof must be an object",
                    code="programme_notification_deduplication_invalid",
                )
            try:
                deduplication_proof = NotificationDeduplicationProofV1.from_dict(
                    deduplication_raw
                )
            except (ProgrammeContractError, TypeError, ValueError) as error:
                raise ActorError(
                    "notification deduplication proof is invalid",
                    code="programme_notification_deduplication_invalid",
                ) from error
            if (
                deduplication_proof.provider != provider
                or deduplication_proof.notification_intent_id != intent_id
            ):
                raise ActorError(
                    "notification deduplication proof binds another delivery",
                    code="programme_notification_deduplication_invalid",
                )
        delivery = TerminalNotificationDeliveryV1(
            notification_intent_id=intent_id,
            notification_intent_digest=canonical_sha256(intent),
            delivery_attempt=1,
            provider=provider,
            provider_receipt_digest=provider_receipt,
            delivered_at_utc=at_utc,
            deduplication_proof_digest=(
                None
                if deduplication_proof is None
                else deduplication_proof.proof_digest()
            ),
        )
        journal.record_notification_delivery(delivery)
        outcomes.append(
            {
                "notification_intent_id": intent_id,
                "status": "DELIVERED",
                "delivery": delivery.to_dict(),
            }
        )
    return outcomes


def programme_status_v1(
    journal: ProgrammeJournalV1,
    *,
    owner_facts: Iterable[OwnedStatusFactV1] = (),
    as_of_utc: str | None = None,
) -> dict[str, Any]:
    as_of = as_of_utc or _utc_now()
    authorization = journal.authorization()
    transitions = journal.transitions()
    action_plans = journal.action_plans()
    latest = transitions[-1] if transitions else None
    projection = project_unified_status_v1(owner_facts, as_of_utc=as_of)
    pending_actions: list[str] = []
    if journal.action_intents_dir.is_dir():
        for path in sorted(journal.action_intents_dir.glob("*.json")):
            action_id = path.stem
            if journal.action_settlement(action_id) is None:
                pending_actions.append(action_id)
    notification_intents = journal.notification_intents()
    pending_notifications = []
    for intent in notification_intents:
        intent_id = f"CRCA-NOTIFY-{str(dict(intent['terminal_programme_digest'])['hex']).upper()}"
        if not journal.notification_deliveries(intent_id):
            pending_notifications.append(intent_id)
    return {
        "schema_version": PROGRAMME_DRIVER_STATUS_SCHEMA_VERSION,
        "programme_id": authorization.get("programme_id"),
        "authorization_digest": journal.authorization_digest().to_dict(),
        "programme_state": (
            "NOT_STARTED"
            if latest is None
            else latest.get("terminal_state") or latest.get("gate_state")
        ),
        "exact_gate": None if latest is None else latest.get("gate_id"),
        "transition_count": len(transitions),
        "action_plan_count": len(action_plans),
        "next_action_plan_digest": (
            None
            if len(transitions) >= len(action_plans)
            else canonical_sha256(action_plans[len(transitions)]).to_dict()
        ),
        "latest_transition_digest": (
            None if latest is None else canonical_sha256(latest).to_dict()
        ),
        "pending_action_intents": pending_actions,
        "pending_notification_intents": pending_notifications,
        "driver_lease": journal.driver_lease_head(),
        "owner_projection": projection,
        "status": (
            "INCONSISTENT"
            if projection["status"] == "INCONSISTENT"
            else "READY"
        ),
        "as_of_utc": as_of,
        "watch_policy": {
            "model_polling": False,
            "owner_heartbeat_or_event_driven": True,
            "default_sparse_backoff_seconds": 300,
            "one_waiter_per_run_required": True,
        },
    }


def _ensure_terminal_notification_intent_v1(
    journal: ProgrammeJournalV1, *, at_utc: str
) -> TerminalNotificationIntentV1 | None:
    latest = journal.latest_transition()
    if latest is None or latest.get("terminal_state") is None:
        return None
    authorization = journal.authorization()
    binding = _terminal_notification_binding_v1(
        authorization, journal.authorization_digest(), latest
    )
    terminal_transition_digest = binding["terminal_transition_digest"]
    existing_intents = journal.notification_intents()
    matching = [
        row
        for row in existing_intents
        if row.get("terminal_transition_digest")
        == terminal_transition_digest.to_dict()
    ]
    if len(matching) == 1:
        return None
    if matching or existing_intents:
        raise ActorError(
            "terminal transition has conflicting notification intent state",
            code="programme_notification_invalid",
        )
    intent = TerminalNotificationIntentV1(
        programme_id=str(binding["programme_id"]),
        terminal_state=binding["terminal_state"],
        terminal_transition_digest=terminal_transition_digest,
        terminal_programme_digest=binding["terminal_programme_digest"],
        recipient_digest=binding["recipient_digest"],
        subject_digest=binding["subject_digest"],
        body_digest=binding["body_digest"],
        created_at_utc=at_utc,
    )
    journal.persist_notification_intent(intent)
    return intent


def reconcile_driver_once_v1(
    journal: ProgrammeJournalV1,
    *,
    driver_lease: ProgrammeDriverLeaseV1 | None = None,
    observe: ActionObserver | None = None,
    execute: ActionExecutor | None = None,
    at_utc: str | None = None,
    action_claim_ttl_seconds: int = 900,
) -> dict[str, Any]:
    """Advance one durable plan action or report the exact wait boundary."""

    observed_at = at_utc or _utc_now()
    status = programme_status_v1(journal, as_of_utc=observed_at)
    if status["programme_state"] in {
        state.value for state in ProgrammeTerminalStateV1
    }:
        _ensure_terminal_notification_intent_v1(journal, at_utc=observed_at)
        status = programme_status_v1(journal, as_of_utc=observed_at)
        return {
            "status": "PROGRAMME_TERMINAL",
            "programme": status,
            "effect_executed": False,
            "next": "terminal_notifier",
        }
    plans = journal.action_plans()
    transitions = journal.transitions()
    if len(transitions) >= len(plans):
        return {
            "status": "WAIT_ACTION_PLAN",
            "programme": status,
            "effect_executed": False,
            "next": "durable_action_plan",
        }
    if driver_lease is None or observe is None or execute is None:
        return {
            "status": "WAIT_ACTION_ADAPTER",
            "programme": status,
            "effect_executed": False,
            "next": "authorized_owner_adapter",
        }
    plan = ProgrammeActionPlanV1.from_dict(plans[len(transitions)])
    intent_raw = journal.action_intent_for_plan(plan)
    if intent_raw is None:
        intent = ProgrammeActionIntentV1(
            programme_id=plan.programme_id,
            authorization_digest=plan.authorization_digest,
            action_plan_digest=plan.plan_digest(),
            gate_id=plan.gate_id,
            action_code=plan.action_code,
            action_payload_digest=plan.action_payload_digest,
            exact_owner_id=plan.exact_owner_id,
            exact_owner_digest=plan.exact_owner_digest,
            fencing_token_digest=driver_lease.fencing_token_digest,
            created_at_utc=observed_at,
        )
        journal.persist_action_intent(intent)
    else:
        intent = ProgrammeActionIntentV1.from_dict(intent_raw)
    action = reconcile_action_v1(
        journal,
        intent,
        driver_lease=driver_lease,
        observe=observe,
        execute=execute,
        at_utc=observed_at,
        action_claim_ttl_seconds=action_claim_ttl_seconds,
    )
    if action["status"] != "SETTLED":
        return {
            "status": action["status"],
            "programme": programme_status_v1(journal, as_of_utc=observed_at),
            "effect_executed": bool(action.get("effect_executed", False)),
            "action": action,
            "next": "owner_event_or_action_claim_expiry",
        }
    settlement = dict(action["settlement"])
    latest = journal.latest_transition()
    transition = ProgrammeTransitionReceiptV1(
        programme_id=plan.programme_id,
        authorization_digest=plan.authorization_digest,
        ordinal=len(transitions) + 1,
        prior_transition_digest=(
            None if latest is None else canonical_sha256(latest)
        ),
        gate_id=plan.gate_id,
        gate_state=plan.success_gate_state,
        recovery_state=None,
        work_digest=plan.work_digest,
        session_digest=plan.session_digest,
        assignment_digest=plan.assignment_digest,
        dispatch_digest=plan.dispatch_digest,
        scheduler_assignment_lease_digest=plan.scheduler_assignment_lease_digest,
        run_digest=plan.run_digest,
        attempt_digest=plan.attempt_digest,
        checkpoint_digest=plan.checkpoint_digest,
        action_intent_id=intent.action_intent_id(),
        action_intent_digest=intent.action_digest(),
        effect_receipt_digest=_content_digest(
            settlement.get("effect_receipt_digest"),
            field="effect_receipt_digest",
        ),
        result_receipt_digest=_content_digest(
            settlement.get("result_receipt_digest"),
            field="result_receipt_digest",
        ),
        settlement_receipt_digest=canonical_sha256(settlement),
        next_safe_action=plan.next_safe_action,
        wake_condition=plan.wake_condition,
        terminal_state=plan.terminal_state,
        recorded_at_utc=observed_at,
    )
    journal.append_transition(transition)
    if plan.terminal_state is not None:
        _ensure_terminal_notification_intent_v1(journal, at_utc=observed_at)
    programme = programme_status_v1(journal, as_of_utc=observed_at)
    return {
        "status": (
            "PROGRAMME_TERMINAL"
            if plan.terminal_state is not None
            else "PROGRESSED"
        ),
        "programme": programme,
        "effect_executed": bool(action.get("effect_executed", False)),
        "action": action,
        "transition": transition.to_dict(),
        "next": (
            "terminal_notifier"
            if plan.terminal_state is not None
            else "next_durable_action_plan"
        ),
    }


def claim_local_driver_lease_v1(
    journal: ProgrammeJournalV1,
    *,
    at_utc: str | None = None,
    ttl_seconds: int = 900,
) -> ProgrammeDriverLeaseV1:
    """Bind the supervised local process to one fenced driver generation."""

    process_id = os.getpid()
    process_start_identity = _local_process_start_identity(process_id)
    if process_start_identity is None:
        raise ActorError(
            "local process start identity is unavailable",
            code="programme_driver_process_identity_unavailable",
        )
    module_path = Path(__file__).resolve()
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    owner_digest = canonical_sha256(
        {
            "owner": "tools.scheduler.programme_driver",
            "module_path": str(module_path),
            "module_sha256": module_sha256,
        }
    )
    return journal.claim_driver_lease(
        driver_owner_id="tools.scheduler.programme_driver",
        driver_owner_digest=owner_digest,
        process_id=process_id,
        process_start_identity=process_start_identity,
        at_utc=at_utc or _utc_now(),
        ttl_seconds=ttl_seconds,
    )


def executable_action_adapter_v1(
    executable: str | Path,
) -> tuple[ActionObserver, ActionExecutor]:
    """Bind an explicit owner adapter to the durable observe/execute protocol."""

    target = Path(executable).expanduser().resolve()
    if not target.is_file() or not os.access(target, os.X_OK):
        raise ActorError(
            "programme action adapter is not executable",
            code="programme_action_adapter_unavailable",
            problems=[str(target)],
        )
    adapter_digest = canonical_sha256(
        {
            "adapter_path": str(target),
            "adapter_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    )

    def invoke(
        operation: str,
        intent: ProgrammeActionIntentV1,
        claim: ProgrammeActionClaimV1,
    ) -> Mapping[str, Any]:
        if intent.exact_owner_digest != adapter_digest:
            raise ActorError(
                "programme action adapter differs from the authorized owner",
                code="programme_action_adapter_owner_mismatch",
            )
        completed = subprocess.run(
            [str(target)],
            input=json.dumps(
                {
                    "schema_version": "crca_programme_action_adapter_request_v1",
                    "operation": operation,
                    "action_intent": intent.to_dict(),
                    "action_claim": claim.to_dict(),
                },
                sort_keys=True,
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ActorError(
                "programme action adapter failed",
                code="programme_action_adapter_failed",
                problems=[completed.stderr[-2000:]],
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ActorError(
                "programme action adapter returned invalid JSON",
                code="programme_action_adapter_failed",
            ) from error
        if not isinstance(result, Mapping):
            raise ActorError(
                "programme action adapter returned a non-object",
                code="programme_action_adapter_failed",
            )
        return result

    def observe(
        intent: ProgrammeActionIntentV1, claim: ProgrammeActionClaimV1
    ) -> Mapping[str, Any]:
        return invoke("OBSERVE", intent, claim)

    def execute(
        intent: ProgrammeActionIntentV1, claim: ProgrammeActionClaimV1
    ) -> Mapping[str, Any]:
        return invoke("EXECUTE", intent, claim)

    return observe, execute


def executable_notification_sender_v1(
    executable: str | Path,
) -> NotificationSender:
    """Return an explicit JSON-stdin adapter for the Project Manager endpoint."""

    target = Path(executable).expanduser().resolve()
    if not target.is_file() or not os.access(target, os.X_OK):
        raise ActorError(
            "terminal notification adapter is not executable",
            code="programme_notification_adapter_unavailable",
            problems=[str(target)],
        )

    def send(intent: Mapping[str, Any]) -> Mapping[str, Any]:
        import json

        completed = subprocess.run(
            [str(target)],
            input=json.dumps(dict(intent), sort_keys=True),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
        if completed.returncode != 0:
            raise ActorError(
                "terminal notification adapter failed",
                code="programme_notification_delivery_failed",
                problems=[completed.stderr[-2000:]],
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ActorError(
                "terminal notification adapter returned invalid JSON",
                code="programme_notification_delivery_failed",
            ) from error
        if not isinstance(result, Mapping):
            raise ActorError(
                "terminal notification adapter returned a non-object",
                code="programme_notification_delivery_failed",
            )
        return result

    return send


def _human_status(value: Mapping[str, Any]) -> str:
    programme = dict(value.get("programme") or value)
    return "\n".join(
        (
            f"Programme: {programme.get('programme_id')}",
            f"State: {programme.get('programme_state')}",
            f"Gate: {programme.get('exact_gate') or 'not started'}",
            f"Pending actions: {len(programme.get('pending_action_intents') or [])}",
            f"Pending notifications: {len(programme.get('pending_notification_intents') or [])}",
        )
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.scheduler.programme_driver")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "reconcile", "watch", "notify"):
        command = sub.add_parser(name)
        command.add_argument("--programme-root", required=True)
        command.add_argument("--human", action="store_true")
        if name == "watch":
            command.add_argument("--poll-seconds", type=float, default=300.0)
            command.add_argument("--max-cycles", type=int)
            command.add_argument("--action-adapter-executable", required=True)
            command.add_argument(
                "--action-claim-ttl-seconds", type=int, default=900
            )
        if name == "notify":
            command.add_argument("--adapter-executable", required=True)
            command.add_argument("--provider", default="project-manager")
    args = parser.parse_args(list(argv) if argv is not None else None)
    journal = ProgrammeJournalV1(args.programme_root)
    try:
        if args.command == "status":
            result: dict[str, Any] = programme_status_v1(journal)
        elif args.command == "reconcile":
            result = reconcile_driver_once_v1(journal)
        elif args.command == "watch":
            cycles = 0
            lease = claim_local_driver_lease_v1(
                journal,
                ttl_seconds=max(30, int(float(args.poll_seconds) * 3)),
            )
            observe, execute = executable_action_adapter_v1(
                args.action_adapter_executable
            )
            result = reconcile_driver_once_v1(
                journal,
                driver_lease=lease,
                observe=observe,
                execute=execute,
                action_claim_ttl_seconds=max(
                    1, int(args.action_claim_ttl_seconds)
                ),
            )
            while result["status"] != "PROGRAMME_TERMINAL":
                cycles += 1
                if args.max_cycles is not None and cycles >= args.max_cycles:
                    break
                if result["status"] != "PROGRESSED":
                    time.sleep(max(5.0, float(args.poll_seconds)))
                lease = journal.heartbeat_driver_lease(
                    lease,
                    at_utc=_utc_now(),
                    ttl_seconds=max(30, int(float(args.poll_seconds) * 3)),
                )
                result = reconcile_driver_once_v1(
                    journal,
                    driver_lease=lease,
                    observe=observe,
                    execute=execute,
                    action_claim_ttl_seconds=max(
                        1, int(args.action_claim_ttl_seconds)
                    ),
                )
        else:
            outcomes = deliver_terminal_notifications_v1(
                journal,
                send=executable_notification_sender_v1(args.adapter_executable),
                at_utc=_utc_now(),
                provider=args.provider,
            )
            result = {
                "status": "NOTIFICATIONS_RECONCILED",
                "deliveries": outcomes,
                "programme": programme_status_v1(journal),
            }
    except (ActorError, ProgrammeContractError) as error:
        import json

        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "FAILED",
                    "code": getattr(error, "code", "programme_contract_invalid"),
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    if args.human:
        print(_human_status(result), end="")
    else:
        import json

        print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProgrammeJournalV1",
    "ResourceStartLeaseLedgerV1",
    "deliver_terminal_notifications_v1",
    "claim_local_driver_lease_v1",
    "executable_action_adapter_v1",
    "executable_notification_sender_v1",
    "programme_status_v1",
    "reconcile_action_v1",
    "reconcile_driver_once_v1",
]
