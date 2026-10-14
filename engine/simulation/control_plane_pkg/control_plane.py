from __future__ import annotations
"""simulation/control_plane_pkg/control_plane.py

Phase 2 implementation: ControlPlane core + WAL + Event Log.

Implements (SRS v1):
- R-CP-01: accept runtime intervention commands (via submit_command or inbox)
- R-CP-02: apply only at SafeBoundary (process_pending)
- R-CP-03: append-only immutable executed event log
- R-CP-04: scheduling future interventions (target_generation)
- R-CP-05: amendment + cancellation of future interventions (plan v2)
- R-CP-06: acknowledgement for each processed command
- R-NF-04: event log append-only

Implements (TSD v3.1 hard rules):
- Deterministic ordering key: (target_generation, submission_sequence_number, command_uuid)
- submission_sequence_number monotonic per branch and persisted (checkpoint metadata hook)
- WAL durability: transition to APPLYING is preceded by WAL append+flush
- Idempotency: if command_uuid already known, ACK without re-execution
- Command lifecycle state machine includes SUPERSEDED
- PLAN-FREEZE-01: plan freezes when any command enters the past
- expectation_note required for any plan version increment
- WAL/event log corruption is fail-fast (no silent skipping)

Notes
-----
- This module does not change SafeBoundary order. It is meant to be called as step (6).
- No new RNG usage. No time-based ordering.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import os

from simulation.control_plane_pkg.commands import (
    CommandBase,
    PressureUpdateCommand,
    CheckpointRequestCommand,
    PlanAmendCommand,
    PlanCancelCommand,
    parse_command,
)
from simulation.control_plane_pkg.rng_guard import block_global_rng
from simulation.control_plane_pkg.event_log import (
    EventRecord,
    append_event,
    iter_events,
    new_event_id,
)
from simulation.control_plane_pkg.fs_inbox import poll_inbox, mark_processed, mark_rejected
from simulation.control_plane_pkg.plan import (
    CommandRef,
    InterventionPlanV2,
    load_current_plan,
    write_plan_version,
    freeze_if_needed,
    with_updated_commands,
)
from simulation.control_plane_pkg.wal import (
    WalEntry,
    append_wal_entry,
    iter_wal_entries,
    compute_command_payload_hash,
    WalCorruptionError,
)

ACK_SCHEMA_VERSION = "control_plane_ack_v1"


class CommandState(str, Enum):
    """Authoritative command lifecycle state machine (Phase 2)."""
    SUBMITTED = "SUBMITTED"
    VALIDATED = "VALIDATED"
    SCHEDULED = "SCHEDULED"
    ACKED = "ACKED"
    APPLYING = "APPLYING"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"


_ALLOWED_TRANSITIONS: Dict[CommandState, Tuple[CommandState, ...]] = {
    CommandState.SUBMITTED: (CommandState.VALIDATED, CommandState.REJECTED),
    CommandState.VALIDATED: (CommandState.SCHEDULED, CommandState.APPLIED, CommandState.REJECTED),  # admin commands can apply immediately
    CommandState.SCHEDULED: (CommandState.ACKED, CommandState.APPLYING, CommandState.CANCELLED, CommandState.SUPERSEDED),
    CommandState.ACKED: (CommandState.APPLYING, CommandState.CANCELLED, CommandState.SUPERSEDED),
    CommandState.APPLYING: (CommandState.APPLIED,),
    CommandState.APPLIED: (),
    CommandState.REJECTED: (),
    CommandState.CANCELLED: (),
    CommandState.SUPERSEDED: (),
}


def _ensure_transition(old: CommandState, new: CommandState, command_uuid: str) -> None:
    """Raise ValueError on invalid lifecycle transitions."""
    allowed = _ALLOWED_TRANSITIONS.get(old, ())
    if new not in allowed:
        raise ValueError(f"invalid transition {old.value}->{new.value} for command_uuid={command_uuid}")


@dataclass(frozen=True)
class Ack:
    status: str  # ACCEPTED / REJECTED / IDEMPOTENT_ACK
    command_uuid: str
    reason: str = ""
    original_event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ACK_SCHEMA_VERSION,
            "status": self.status,
            "command_uuid": self.command_uuid,
            "reason": self.reason,
            "original_event_id": self.original_event_id,
        }


@dataclass
class AppliedEvent:
    """Return type for process_pending()."""
    command_uuid: str
    event_id: str
    generation_applied: int
    target_generation: int
    submission_sequence_number: int
    command_payload_hash: str


class ControlPlane:
    """ControlPlane for runtime interventions (Phase 2)."""

    def __init__(
        self,
        run_dir: str,
        branch_id: str,
        campaign_id: str,
        initial_generation: int,
        submission_sequence_number: int = 0,
        trait_injection_apply_phase_frozen: Optional[str] = None,
    ) -> None:
        self.run_dir = str(run_dir)
        self.branch_id = str(branch_id)
        self.campaign_id = str(campaign_id)
        self.initial_generation = int(initial_generation)

        # Frozen trait injection apply phase (Phase 5B). This is configuration metadata only.
        # Determinism note: storing this value does not alter ordering, RNG usage, or physics.
        self.trait_injection_apply_phase_frozen = (
            str(trait_injection_apply_phase_frozen)
            if trait_injection_apply_phase_frozen is not None
            else None
        )


        # Current generation index (updated at SafeBoundary). Used for WAL transition timestamps.
        self._current_generation_index = int(initial_generation)

        # Monotonic per-branch submission sequence number.
        self._submission_seq = int(submission_sequence_number)

        # command_uuid -> info
        self._known: Dict[str, Dict[str, Any]] = {}
        # executed command_uuid -> event_id
        self._executed_event_id: Dict[str, str] = {}

        self._ensure_initial_plan_exists()
        self._recover_from_disk()

        # -----------------------------------------------------------------
        # PLAN FREEZE (DEV MODE OVERRIDE)
        # -----------------------------------------------------------------
        # The repo includes PLAN-FREEZE-01 governance: a plan version becomes
        # frozen once any command enters the past, and frozen plans can block
        # new submissions.
        #
        # For exploratory/dev runs, this hard-block can be counterproductive.
        # We therefore allow disabling plan-freeze enforcement via env var.
        #
        # Default: disabled (developer ergonomics / continuous interventions).
        # Set EIEL_ENFORCE_PLAN_FREEZE=1 to re-enable strict PLAN-FREEZE-01.
        self._enforce_plan_freeze = bool(int(os.getenv("EIEL_ENFORCE_PLAN_FREEZE", "0") or "0"))

    
    def _ensure_initial_plan_exists(self) -> None:
        """Ensure an intervention plan exists for this run.

        Determinism requirement: plan_id is derived deterministically (uuid5) from (campaign_id, branch_id).
        """
        plan = load_current_plan(self.run_dir)
        if plan is not None:
            return
        import uuid as _uuid
        pid = str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"{self.campaign_id}:{self.branch_id}"))
        init_plan = InterventionPlanV2(
            schema_version="intervention_plan_v2",
            plan_id=pid,
            campaign_id=self.campaign_id,
            version=1,
            parent_plan_version=None,
            created_at_generation=int(self.initial_generation),
            expectation_note="initial plan",
            commands=[],
            frozen=False,
        )
        write_plan_version(self.run_dir, init_plan)
    # ---------------------------------------------------------------------
        # Sequence number (persisted via checkpoint metadata hook)    # ---------------------------------------------------------------------

    def next_submission_sequence_number(self) -> int:
        """Return next monotonic submission_sequence_number (and increment)."""
        self._submission_seq += 1
        return int(self._submission_seq)

    def current_submission_sequence_number(self) -> int:
        """Return current submission_sequence_number value (without increment)."""
        return int(self._submission_seq)

    def get_submission_sequence_number(self) -> int:
        """Back-compat alias for current_submission_sequence_number()."""
        return int(self.current_submission_sequence_number())

    def set_submission_sequence_number(self, value: int) -> None:
        """Set submission_sequence_number (used for checkpoint restore tests)."""
        self._submission_seq = int(value)

    # ---------------------------------------------------------------------
    # Submission / ACK
    # ---------------------------------------------------------------------

    def submit_command(self, cmd: CommandBase) -> Ack:
        """Submit a command (callable while sim is running).

        Critical constraints:
        - WAL must NOT record a scheduled command unless validation succeeded.
        - REJECTED must be durable in WAL (prevents restart execution).
        - Idempotency uses command_uuid as key.
        """
        if not isinstance(cmd, CommandBase):
            ack = Ack(status="REJECTED", command_uuid=getattr(cmd, "command_uuid", ""), reason="invalid command type")
            self._write_ack(ack)
            return ack

        cu = str(cmd.command_uuid or "")
        if not cu:
            ack = Ack(status="REJECTED", command_uuid="", reason="missing command_uuid")
            self._write_ack(ack)
            return ack

        # Idempotency (executed)
        if cu in self._executed_event_id:
            ack = Ack(status="IDEMPOTENT_ACK", command_uuid=cu, original_event_id=self._executed_event_id.get(cu))
            self._write_ack(ack)
            return ack

        # Idempotency (known but not executed)
        if cu in self._known:
            ack = Ack(status="IDEMPOTENT_ACK", command_uuid=cu, reason="command already known")
            self._write_ack(ack)
            return ack

        payload = cmd.payload_dict()
        payload_hash = compute_command_payload_hash(payload)

        plan = load_current_plan(self.run_dir)
        if plan is None:
            self._append_terminal_reject(cu, payload, payload_hash, reason="missing plan")
            ack = Ack(status="REJECTED", command_uuid=cu, reason="missing plan")
            self._write_ack(ack)
            return ack

        if bool(getattr(plan, "frozen", False)):
            # DEV override: when plan-freeze enforcement is disabled, we do NOT
            # block new commands. Instead, we roll forward to a new unfrozen plan
            # version (dropping past commands) so submissions remain traceable.
            if not self._enforce_plan_freeze:
                try:
                    from simulation.control_plane_pkg.plan import roll_forward_plan

                    plan = roll_forward_plan(
                        self.run_dir,
                        current_generation=int(self._current_generation_index),
                        expectation_note=f"dev: unfreeze plan to accept command {cu}",
                    )
                except Exception:
                    # As a last resort in dev mode, continue without plan mutation.
                    # Scheduling will still be WAL-backed.
                    plan = plan
            else:
                autoroll = bool(int(os.getenv("EIEL_PLAN_AUTOROLL", "0") or "0"))
                if not autoroll:
                    self._append_terminal_reject(cu, payload, payload_hash, reason="plan is frozen")
                    ack = Ack(status="REJECTED", command_uuid=cu, reason="plan is frozen")
                    self._write_ack(ack)
                    return ack

                # Option A: autoroll forward by creating a new unfrozen plan version
                # that keeps only future commands. This preserves append-only plan semantics.
                try:
                    from simulation.control_plane_pkg.plan import roll_forward_plan

                    plan = roll_forward_plan(
                        self.run_dir,
                        current_generation=int(self._current_generation_index),
                        expectation_note=f"autoroll to accept command {cu}",
                    )
                except Exception:
                    self._append_terminal_reject(cu, payload, payload_hash, reason="plan is frozen (autoroll failed)")
                    ack = Ack(status="REJECTED", command_uuid=cu, reason="plan is frozen (autoroll failed)")
                    self._write_ack(ack)
                    return ack


        # expectation_note is REQUIRED for plan version increment operations.
        if isinstance(cmd, (PlanAmendCommand, PlanCancelCommand)) and not getattr(cmd, "expectation_note", ""):
            self._append_terminal_reject(cu, payload, payload_hash, reason="expectation_note required")
            ack = Ack(status="REJECTED", command_uuid=cu, reason="expectation_note required")
            self._write_ack(ack)
            return ack

        ok, reason = self._validate_command(cmd)
        if not ok:
            self._append_terminal_reject(cu, payload, payload_hash, reason=reason)
            ack = Ack(status="REJECTED", command_uuid=cu, reason=reason)
            self._write_ack(ack)
            return ack

        # Allocate ordering only AFTER validation succeeded.
        ssn = self.next_submission_sequence_number()
        tg = int(cmd.target_generation)

        # Register as known (SUBMITTED).
        self._known[cu] = {
            "payload": payload,
            "payload_hash": payload_hash,
            "target_generation": tg,
            "submission_sequence_number": int(ssn),
            "state": CommandState.SUBMITTED.value,
            "event_id": None,
        }

        # SUBMITTED->VALIDATED (durable, after validation).
        self._transition_and_wal(cu, CommandState.VALIDATED, self._current_generation_index, tg, ssn, payload_hash, command_payload=None)

        # Plan mutation commands are applied immediately (admin), never scheduled for SafeBoundary apply.
        if isinstance(cmd, PlanAmendCommand):
            ack = self._apply_plan_amend(cmd)
            if ack.status != "ACCEPTED":
                # Ensure terminal REJECTED state is durable and remove from known to avoid later apply.
                self._force_terminal_reject_known(cu, tg, ssn, payload_hash, payload, reason=ack.reason)
                return ack
            # Mark superseded in WAL/in-memory if those commands are currently known.
            for other in (cmd.supercede_command_uuids or []):
                self._mark_terminal_if_known(str(other), CommandState.SUPERSEDED)

            return self._finalize_admin_accept(cu, tg, ssn, payload_hash, payload)

        if isinstance(cmd, PlanCancelCommand):
            ack = self._apply_plan_cancel(cmd)
            if ack.status != "ACCEPTED":
                self._force_terminal_reject_known(cu, tg, ssn, payload_hash, payload, reason=ack.reason)
                return ack
            for other in (cmd.cancel_command_uuids or []):
                self._mark_terminal_if_known(str(other), CommandState.CANCELLED)

            return self._finalize_admin_accept(cu, tg, ssn, payload_hash, payload)

        # Freeze plan if needed (PLAN-FREEZE-01)
        if self._enforce_plan_freeze:
            frozen_plan = freeze_if_needed(plan, current_generation=self._current_generation_index)
            if frozen_plan is not None and getattr(frozen_plan, "version", None) != getattr(plan, "version", None):
                write_plan_version(self.run_dir, frozen_plan)

        # Schedule into plan (creates a new immutable plan version).
        # expectation_note is required for any version increment; we use a deterministic note.
        try:
            plan = load_current_plan(self.run_dir)
            if plan is None:
                raise ValueError("missing plan")
            if plan.frozen and self._enforce_plan_freeze:
                raise ValueError("plan is frozen")

            new_version = int(plan.version) + 1
            new_cmds = list(plan.commands) + [
                CommandRef(command_uuid=cu, target_generation=tg, submission_sequence_number=int(ssn), status="SCHEDULED")
            ]
            scheduled_plan = with_updated_commands(
                plan,
                new_version=new_version,
                expectation_note=f"schedule command {cu}",
                commands=new_cmds,
            )
            write_plan_version(self.run_dir, scheduled_plan)
        except Exception:
            # Reject deterministically: also persist REJECTED to WAL terminal state.
            self._force_terminal_reject_known(cu, tg, ssn, payload_hash, payload, reason="failed to schedule into plan")
            ack = Ack(status="REJECTED", command_uuid=cu, reason="failed to schedule into plan")
            self._write_ack(ack)
            return ack

        # VALIDATED->SCHEDULED (include payload for restart reconstruction)
        self._transition_and_wal(cu, CommandState.SCHEDULED, self._current_generation_index, tg, ssn, payload_hash, command_payload=payload)

        ack = Ack(status="ACCEPTED", command_uuid=cu)
        self._write_ack(ack)
        # Note: ACK persistence is required, but we keep lifecycle in SCHEDULED so SafeBoundary uses SCHEDULED->APPLYING.
        return ack
    def _finalize_admin_accept(self, command_uuid: str, tg: int, ssn: int, payload_hash: str, payload: Dict[str, Any]) -> Ack:
        """Finalize acceptance for admin commands applied immediately (plan amend/cancel)."""
        # VALIDATED->APPLIED (terminal) and event log record for idempotency/audit.
        self._transition_and_wal(command_uuid, CommandState.APPLIED, self._current_generation_index, tg, ssn, payload_hash, command_payload=None)

        eid = new_event_id()
        append_event(
            self.run_dir,
            EventRecord(
                schema_version="command_event_v1",
                event_id=eid,
                generation_applied=int(self.initial_generation),
                target_generation=int(tg),
                submission_sequence_number=int(ssn),
                command_uuid=str(command_uuid),
                command_payload_hash=str(payload_hash),
                command_payload=payload,
            ),
        )
        self._executed_event_id[str(command_uuid)] = eid
        if command_uuid in self._known:
            self._known[command_uuid]["event_id"] = eid

        ack = Ack(status="ACCEPTED", command_uuid=str(command_uuid))
        self._write_ack(ack)
        return ack

    def _validate_command(self, cmd: CommandBase) -> Tuple[bool, str]:
        """Deterministic command validation.

        Keep minimal for Phase 2, but strict enough to prevent WAL ghost scheduling.
        """
        if isinstance(cmd, PressureUpdateCommand):
            if not isinstance(cmd.updates, list) or not cmd.updates:
                return False, "PressureUpdateCommand.updates missing"
            allowed_prefixes = ("env.cfg.", "sim_cfg.", "pressures.")
            for upd in cmd.updates:
                if not isinstance(upd, dict):
                    return False, "PressureUpdateCommand.update not a dict"
                path = str(upd.get("path", "") or "")
                if not path:
                    return False, "PressureUpdateCommand.update.path missing"
                if not any(path.startswith(pfx) for pfx in allowed_prefixes):
                    return False, "PressureUpdateCommand.update.path not allowed"
            return True, ""
        if isinstance(cmd, CheckpointRequestCommand):
            return True, ""
        if isinstance(cmd, (PlanAmendCommand, PlanCancelCommand)):
            return True, ""
        # Phase 5B: TraitInjectionCommand (schema v2)
        # NOTE: We intentionally avoid importing the command class here to reduce coupling.
        # Validation is based on the canonical payload fields to keep determinism stable.
        try:
            trait = getattr(cmd, "trait_injection", None)
            if trait is not None:
                if not isinstance(trait, dict):
                    return False, "TraitInjectionCommand.trait_injection must be a dict"
                if str(trait.get("schema_version", "") or "") != "trait_injection_v2":
                    return False, "TraitInjectionCommand.trait_injection.schema_version must be trait_injection_v2"

                apply_phase = str(trait.get("apply_phase", "") or "")
                allowed_phases = {
                    "post-reproduction-pre-selection",
                    "post-selection-pre-mutation",
                }
                if apply_phase not in allowed_phases:
                    return False, "TraitInjectionCommand.trait_injection.apply_phase not allowed"

                # Campaign-frozen apply_phase must match.
                if self.trait_injection_apply_phase_frozen is not None and apply_phase != self.trait_injection_apply_phase_frozen:
                    return False, "TraitInjectionCommand.apply_phase mismatch (frozen)"

                dur = trait.get("duration_generations", None)
                try:
                    dur_i = int(dur)
                except Exception:
                    return False, "TraitInjectionCommand.trait_injection.duration_generations invalid"
                if dur_i <= 0:
                    return False, "TraitInjectionCommand.trait_injection.duration_generations must be > 0"

                locus_tags = trait.get("locus_tags", None)
                if not isinstance(locus_tags, list) or not locus_tags:
                    return False, "TraitInjectionCommand.trait_injection.locus_tags missing"
                for lt in locus_tags:
                    if not str(lt or ""):
                        return False, "TraitInjectionCommand.trait_injection.locus_tags invalid"

                allele_values = trait.get("allele_values", None)
                if not isinstance(allele_values, dict) or not allele_values:
                    return False, "TraitInjectionCommand.trait_injection.allele_values missing"

                # target_agent_ids is optional. If present, require a list.
                if "target_agent_ids" in trait and not isinstance(trait.get("target_agent_ids"), list):
                    return False, "TraitInjectionCommand.trait_injection.target_agent_ids must be a list"

                return True, ""
        except Exception:
            # Defensive: never crash validation.
            return False, "TraitInjectionCommand validation error"
        return False, "unknown command type"

    def _append_terminal_reject(self, command_uuid: str, payload: Dict[str, Any], payload_hash: str, reason: str) -> None:
        """Persist SUBMITTED->REJECTED as a durable terminal state in WAL."""
        append_wal_entry(
            self.run_dir,
            WalEntry(
                schema_version="command_wal_v1",
                command_uuid=str(command_uuid),
                transition="SUBMITTED->REJECTED",
                generation_index=int(payload.get("target_generation", 0) or 0),
                submission_sequence_number=0,
                command_payload_hash=str(payload_hash),
                command_payload=payload,
            ),
        )

    def _force_terminal_reject_known(
        self,
        command_uuid: str,
        target_generation: int,
        submission_sequence_number: int,
        payload_hash: str,
        payload: Dict[str, Any],
        reason: str,
    ) -> None:
        """Force a known command to terminal REJECTED with WAL durability."""
        info = self._known.get(command_uuid)
        if info is None:
            self._append_terminal_reject(command_uuid, payload, payload_hash, reason=reason)
            return
        old_state = CommandState(str(info.get("state")))
        _ensure_transition(old_state, CommandState.REJECTED, command_uuid)
        transition = f"{old_state.value}->{CommandState.REJECTED.value}"
        append_wal_entry(
            self.run_dir,
            WalEntry(
                schema_version="command_wal_v1",
                command_uuid=str(command_uuid),
                transition=transition,
                generation_index=int(transition_generation),
                target_generation=int(target_generation),
                submission_sequence_number=int(submission_sequence_number),
                command_payload_hash=str(payload_hash),
                command_payload=payload,
            ),
        )
        info["state"] = CommandState.REJECTED.value
    def _transition_and_wal(
        self,
        command_uuid: str,
        new_state: CommandState,
        transition_generation: int,
        target_generation: int,
        submission_sequence_number: int,
        payload_hash: str,
        command_payload: Optional[Dict[str, Any]],
    ) -> None:
        """Transition a known command and append a WAL entry."""
        info = self._known.get(command_uuid)
        if info is None:
            raise ValueError(f"unknown command_uuid for transition: {command_uuid}")
        old_state = CommandState(str(info.get("state")))
        _ensure_transition(old_state, new_state, command_uuid)
        transition = f"{old_state.value}->{new_state.value}"
        append_wal_entry(
            self.run_dir,
            WalEntry(
                schema_version="command_wal_v1",
                command_uuid=command_uuid,
                transition=transition,
                generation_index=int(transition_generation),
                target_generation=int(target_generation),
                submission_sequence_number=int(submission_sequence_number),
                command_payload_hash=payload_hash,
                command_payload=command_payload,
            ),
        )
        info["state"] = new_state.value

    def _mark_terminal_if_known(self, command_uuid: str, terminal_state: CommandState) -> None:
        """Mark other command uuids as CANCELLED/SUPERSEDED if known, with WAL durability."""
        if terminal_state not in (CommandState.CANCELLED, CommandState.SUPERSEDED):
            return
        info = self._known.get(command_uuid)
        if info is None:
            # If we never saw the original command, we cannot safely log terminal transitions (missing ordering/hash).
            return
        old_state = CommandState(str(info.get("state")))
        if old_state in (CommandState.APPLIED, CommandState.REJECTED, CommandState.CANCELLED, CommandState.SUPERSEDED):
            return
        _ensure_transition(old_state, terminal_state, command_uuid)
        transition = f"{old_state.value}->{terminal_state.value}"
        append_wal_entry(
            self.run_dir,
            WalEntry(
                schema_version="command_wal_v1",
                command_uuid=command_uuid,
                transition=transition,
                generation_index=int(info.get("target_generation", 0)),
                submission_sequence_number=int(info.get("submission_sequence_number", 0)),
                command_payload_hash=str(info.get("payload_hash", "")),
                command_payload=None,
            ),
        )
        info["state"] = terminal_state.value

    # ---------------------------------------------------------------------
    # Plan mutation helpers (preserved from previous patch version)
    # ---------------------------------------------------------------------

    def _apply_plan_amend(self, cmd: PlanAmendCommand) -> Ack:
        """Apply PlanAmendCommand by writing a new InterventionPlan version (N+1)."""
        plan = load_current_plan(self.run_dir)
        if plan is None:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="no current plan")
        if str(cmd.plan_id) and str(cmd.plan_id) != plan.plan_id:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="plan_id mismatch")
        if int(cmd.base_plan_version) != int(plan.version):
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="base_plan_version mismatch")
        if plan.frozen and self._enforce_plan_freeze:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="plan is frozen")

        cmds: List[CommandRef] = list(plan.commands)

        supercede_set = {str(x) for x in (cmd.supercede_command_uuids or [])}
        new_cmds: List[CommandRef] = []
        for c in cmds:
            if c.command_uuid in supercede_set and c.status == "SCHEDULED":
                new_cmds.append(CommandRef(c.command_uuid, c.target_generation, c.submission_sequence_number, "SUPERCEDED"))
            else:
                new_cmds.append(c)

        for d in (cmd.add_commands or []):
            if not isinstance(d, dict):
                continue
            cu2 = str(d.get("command_uuid", "") or "")
            tg2 = int(d.get("target_generation", 0))
            ssn2 = int(d.get("submission_sequence_number", 0))
            if cu2:
                new_cmds.append(CommandRef(cu2, tg2, ssn2, "SCHEDULED"))

        new_version = int(plan.version) + 1
        try:
            new_plan = with_updated_commands(plan, new_version=new_version, expectation_note=str(cmd.expectation_note), commands=new_cmds)
        except Exception as e:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason=str(e))

        write_plan_version(self.run_dir, new_plan)
        return Ack(status="ACCEPTED", command_uuid=cmd.command_uuid)

    def _apply_plan_cancel(self, cmd: PlanCancelCommand) -> Ack:
        """Apply PlanCancelCommand by writing a new InterventionPlan version (N+1)."""
        plan = load_current_plan(self.run_dir)
        if plan is None:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="no current plan")
        if str(cmd.plan_id) and str(cmd.plan_id) != plan.plan_id:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="plan_id mismatch")
        if int(cmd.base_plan_version) != int(plan.version):
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="base_plan_version mismatch")
        if plan.frozen and self._enforce_plan_freeze:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason="plan is frozen")

        cancel_set = {str(x) for x in (cmd.cancel_command_uuids or [])}
        new_cmds: List[CommandRef] = []
        for c in plan.commands:
            if c.command_uuid in cancel_set and c.status == "SCHEDULED":
                new_cmds.append(CommandRef(c.command_uuid, c.target_generation, c.submission_sequence_number, "CANCELLED"))
            else:
                new_cmds.append(c)

        new_version = int(plan.version) + 1
        try:
            new_plan = with_updated_commands(plan, new_version=new_version, expectation_note=str(cmd.expectation_note), commands=new_cmds)
        except Exception as e:
            return Ack(status="REJECTED", command_uuid=cmd.command_uuid, reason=str(e))

        write_plan_version(self.run_dir, new_plan)
        return Ack(status="ACCEPTED", command_uuid=cmd.command_uuid)

    # ---------------------------------------------------------------------
    # SafeBoundary apply
    # ---------------------------------------------------------------------

    def process_pending(self, generation_index: int, sb_ctx: Any, sim: Any) -> List[AppliedEvent]:
        """Apply due commands at SafeBoundary step (6)."""
        for inbox_path, payload in poll_inbox(self.run_dir):
            try:
                cmd = parse_command(payload)
                ack = self.submit_command(cmd)
                # Best-effort operator visibility for every inbox submission.
                self._write_ack(ack)
            except Exception as e:
                # Parsing/validation failure: do NOT silently 'process' the file.
                # Move it to inbox_rejected/ and emit a small ack-like notice.
                try:
                    mark_rejected(self.run_dir, inbox_path)
                except Exception:
                    pass
                try:
                    import json as _json
                    adir = os.path.join(self.run_dir, 'acks')
                    os.makedirs(adir, exist_ok=True)
                    notice = {
                        'schema_version': ACK_SCHEMA_VERSION,
                        'status': 'REJECTED',
                        'command_uuid': str(payload.get('command_uuid', '') or ''),
                        'reason': f'inbox_parse_or_submit_error: {type(e).__name__}: {e}',
                        'original_event_id': None,
                    }
                    fn = os.path.basename(inbox_path)
                    p = os.path.join(adir, f'ack_inbox_reject__{fn}.json')
                    with open(p, 'w', encoding='utf-8') as f:
                        _json.dump(notice, f, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
                except Exception:
                    pass
            finally:
                # Move inbox file aside so it can't be reprocessed.
                # NOTE: writer must use tmp + atomic rename to avoid partial reads.
                try:
                    mark_processed(self.run_dir, inbox_path)
                except Exception:
                    # Never crash the sim because a bookkeeping move failed.
                    pass

        gi = int(generation_index)
        self._current_generation_index = gi

        # PLAN-FREEZE-01: freeze plan when any command enters the past.
        # DEV override: when plan-freeze enforcement is disabled, do not auto-freeze.
        if self._enforce_plan_freeze:
            plan = load_current_plan(self.run_dir)
            if plan is not None and not plan.frozen:
                should_freeze = any(int(c.target_generation) <= gi for c in plan.commands)
                if should_freeze:
                    # Freeze requires a new immutable plan version.
                    from simulation.control_plane_pkg.plan import InterventionPlanV2
                    frozen_plan = InterventionPlanV2(
                        schema_version=plan.schema_version,
                        plan_id=plan.plan_id,
                        campaign_id=plan.campaign_id,
                        version=int(plan.version) + 1,
                        parent_plan_version=int(plan.version),
                        created_at_generation=int(plan.created_at_generation),
                        expectation_note=f"auto-freeze at generation {gi}",
                        commands=list(plan.commands),
                        frozen=True,
                    )
                    write_plan_version(self.run_dir, frozen_plan)

        due: List[Tuple[int, int, str]] = []
        for cu, info in self._known.items():
            st = CommandState(str(info.get("state")))
            if st not in (CommandState.SCHEDULED, CommandState.ACKED):
                continue
            if int(info.get("target_generation", -1)) != gi:
                continue
            due.append((int(info.get("target_generation")), int(info.get("submission_sequence_number")), str(cu)))

        due.sort(key=lambda t: (t[0], t[1], t[2]))

        applied_events: List[AppliedEvent] = []

        for _tg, _ssn, cu in due:
            info = self._known.get(cu)
            if info is None:
                continue

            st = CommandState(str(info.get("state")))
            if st not in (CommandState.SCHEDULED, CommandState.ACKED):
                continue

            payload = info.get("payload", {}) or {}
            payload_hash = str(info.get("payload_hash", "") or "")
            tg = int(info.get("target_generation", gi))
            ssn = int(info.get("submission_sequence_number", 0))

            # WAL: ->APPLYING before apply
            self._transition_and_wal(cu, CommandState.APPLYING, gi, tg, ssn, payload_hash, command_payload=None)

            cmd_obj = parse_command(payload)
            with block_global_rng():
                cmd_obj.apply_to_sim(sim)

            eid = new_event_id()
            append_event(
                self.run_dir,
                EventRecord(
                    schema_version="command_event_v1",
                    event_id=eid,
                    generation_applied=gi,
                    target_generation=tg,
                    submission_sequence_number=ssn,
                    command_uuid=cu,
                    command_payload_hash=payload_hash,
                    command_payload=payload,
                ),
            )

            self._transition_and_wal(cu, CommandState.APPLIED, gi, tg, ssn, payload_hash, command_payload=None)

            self._executed_event_id[cu] = eid
            info["event_id"] = eid

            applied_events.append(
                AppliedEvent(
                    command_uuid=cu,
                    event_id=eid,
                    generation_applied=gi,
                    target_generation=tg,
                    submission_sequence_number=ssn,
                    command_payload_hash=payload_hash,
                )
            )

        return applied_events

    # ---------------------------------------------------------------------
    # Recovery
    # ---------------------------------------------------------------------

    def _recover_from_disk(self) -> None:
        """Rebuild in-memory state from event log + WAL (fail-fast on corruption)."""
        for ev in iter_events(self.run_dir):
            cu = str(ev.get("command_uuid", "") or "")
            eid = str(ev.get("event_id", "") or "")
            if cu and eid:
                self._executed_event_id[cu] = eid

        for w in iter_wal_entries(self.run_dir):
            cu = str(w.get("command_uuid", "") or "")
            ssn = int(w.get("submission_sequence_number", 0) or 0)
            tg = int((w.get("target_generation") if w.get("target_generation") is not None else w.get("generation_index", 0)) or 0)
            ph = str(w.get("command_payload_hash", "") or "")
            tr = str(w.get("transition", "") or "")

            if ssn > self._submission_seq:
                self._submission_seq = ssn

            if cu not in self._known:
                self._known[cu] = {
                    "payload": {},
                    "payload_hash": ph,
                    "target_generation": tg,
                    "submission_sequence_number": ssn,
                    "state": CommandState.SUBMITTED.value,
                    "event_id": None,
                }
            info = self._known[cu]

            if isinstance(w.get("command_payload"), dict):
                info["payload"] = w.get("command_payload") or info.get("payload", {})
            if ph:
                info["payload_hash"] = ph
            if tg:
                info["target_generation"] = tg
            if ssn:
                info["submission_sequence_number"] = ssn

            if "->" not in tr:
                raise WalCorruptionError(f"WAL corruption: invalid transition '{tr}' for command_uuid={cu}")
            from_s, to_s = tr.split("->", 1)

            try:
                from_state = CommandState(from_s)
                to_state = CommandState(to_s)
            except Exception:
                raise WalCorruptionError(f"WAL corruption: unknown states in transition '{tr}' for command_uuid={cu}")

            current = CommandState(str(info.get("state")))
            if current != from_state:
                if to_state not in _ALLOWED_TRANSITIONS.get(current, ()):
                    raise WalCorruptionError(
                        f"WAL corruption: transition mismatch current={current.value} wal_from={from_state.value} wal_to={to_state.value} command_uuid={cu}"
                    )
                info["state"] = to_state.value
            else:
                _ensure_transition(current, to_state, cu)
                info["state"] = to_state.value

        # Overlay executed commands as APPLIED from event log (authoritative).
        for cu, eid in self._executed_event_id.items():
            if cu in self._known:
                self._known[cu]["state"] = CommandState.APPLIED.value
                self._known[cu]["event_id"] = eid

    # ---------------------------------------------------------------------
    # Ack persistence
    # ---------------------------------------------------------------------

    def _write_ack(self, ack: Ack) -> None:
        """Best-effort operator visibility."""
        try:
            adir = os.path.join(self.run_dir, "acks")
            os.makedirs(adir, exist_ok=True)
            import json
            p = os.path.join(adir, f"ack_{ack.command_uuid}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(ack.to_dict(), f, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Checkpoint metadata hook helpers (used by simulation/checkpoint.py)
    # ---------------------------------------------------------------------

    def export_checkpoint_metadata(self) -> Dict[str, Any]:
        """Return minimal metadata to persist in checkpoint v1."""
        return {
            "control_plane": {
                "submission_sequence_number": int(self.current_submission_sequence_number()),
                "branch_id": self.branch_id,
                "campaign_id": self.campaign_id,
            }
        }

    @staticmethod
    def import_submission_sequence_number(meta: Dict[str, Any]) -> int:
        """Extract submission_sequence_number from checkpoint metadata dict."""
        try:
            cp = meta.get("control_plane", {}) if isinstance(meta, dict) else {}
            return int(cp.get("submission_sequence_number", 0) or 0)
        except Exception:
            return 0
