from __future__ import annotations
"""simulation/control_plane_pkg/plan.py

InterventionPlan v2 (schema 'intervention_plan_v2').

Phase 2 requirements:
- Version chain: amendments create N+1 with parent_plan_version=N
- expectation_note REQUIRED for any version increment
- PLAN-FREEZE-01: plan version freezes permanently if ANY command has target_generation <= current_generation
- Immutable on disk: plans are stored as JSON files; existing versions are never modified (append-only semantics at the file level)

This module keeps the data model simple so the ControlPlane core remains small.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import os
import uuid as _uuid

from simulation.control_plane_pkg.util_json import canonical_json_bytes


@dataclass(frozen=True)
class CommandRef:
    command_uuid: str
    target_generation: int
    submission_sequence_number: int
    status: str  # SCHEDULED / CANCELLED / SUPERCEDED / APPLIED


@dataclass(frozen=True)
class InterventionPlanV2:
    schema_version: str
    plan_id: str
    campaign_id: str
    version: int
    parent_plan_version: Optional[int]
    created_at_generation: int
    expectation_note: str
    commands: List[CommandRef] = field(default_factory=list)
    frozen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "campaign_id": self.campaign_id,
            "version": int(self.version),
            "parent_plan_version": self.parent_plan_version if self.parent_plan_version is None else int(self.parent_plan_version),
            "created_at_generation": int(self.created_at_generation),
            "expectation_note": str(self.expectation_note or ""),
            "commands": [
                {
                    "command_uuid": c.command_uuid,
                    "target_generation": int(c.target_generation),
                    "submission_sequence_number": int(c.submission_sequence_number),
                    "status": str(c.status),
                }
                for c in self.commands
            ],
            "frozen": bool(self.frozen),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "InterventionPlanV2":
        cmds = []
        for c in (d.get("commands") or []):
            if not isinstance(c, dict):
                continue
            cmds.append(
                CommandRef(
                    command_uuid=str(c.get("command_uuid", "")),
                    target_generation=int(c.get("target_generation", 0)),
                    submission_sequence_number=int(c.get("submission_sequence_number", 0)),
                    status=str(c.get("status", "SCHEDULED")),
                )
            )
        return InterventionPlanV2(
            schema_version=str(d.get("schema_version", "intervention_plan_v2")),
            plan_id=str(d.get("plan_id", "")),
            campaign_id=str(d.get("campaign_id", "")),
            version=int(d.get("version", 1)),
            parent_plan_version=d.get("parent_plan_version", None),
            created_at_generation=int(d.get("created_at_generation", 0)),
            expectation_note=str(d.get("expectation_note", "")),
            commands=cmds,
            frozen=bool(d.get("frozen", False)),
        )

    def payload_hash(self) -> str:
        b = canonical_json_bytes(self.to_dict())
        import hashlib
        return "sha256:" + hashlib.sha256(b).hexdigest()


def plan_dir(run_dir: str) -> str:
    return os.path.join(str(run_dir), "plans")


def plan_path(run_dir: str, plan_id: str, version: int) -> str:
    return os.path.join(plan_dir(run_dir), f"plan_{plan_id}_v{int(version):06d}.json")


def current_plan_pointer_path(run_dir: str) -> str:
    return os.path.join(plan_dir(run_dir), "current_plan.json")


def ensure_plan_dir(run_dir: str) -> None:
    os.makedirs(plan_dir(run_dir), exist_ok=True)


def load_current_plan(run_dir: str) -> Optional[InterventionPlanV2]:
    p = current_plan_pointer_path(run_dir)
    if not os.path.exists(p):
        return None
    import json
    with open(p, "r", encoding="utf-8") as f:
        ptr = json.load(f)
    if not isinstance(ptr, dict):
        return None
    plan_id = str(ptr.get("plan_id", "") or "")
    version = int(ptr.get("version", 0) or 0)
    if not plan_id or version <= 0:
        return None
    pp = plan_path(run_dir, plan_id, version)
    if not os.path.exists(pp):
        return None
    with open(pp, "r", encoding="utf-8") as f:
        d = json.load(f)
    return InterventionPlanV2.from_dict(d)


def write_plan_version(run_dir: str, plan: InterventionPlanV2) -> None:
    ensure_plan_dir(run_dir)
    import json
    p = plan_path(run_dir, plan.plan_id, plan.version)
    # Immutability: do not overwrite if exists.
    if os.path.exists(p):
        raise FileExistsError(f"plan version already exists: {p}")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2, sort_keys=True)
    # Update pointer (allowed to overwrite).
    with open(current_plan_pointer_path(run_dir), "w", encoding="utf-8") as f:
        json.dump({"plan_id": plan.plan_id, "version": int(plan.version)}, f, indent=2, sort_keys=True)


def create_initial_plan(run_dir: str, *, campaign_id: str, created_at_generation: int, expectation_note: str) -> InterventionPlanV2:
    if not expectation_note:
        raise ValueError("expectation_note is required")
    ensure_plan_dir(run_dir)
    pid = str(_uuid.uuid4())
    plan = InterventionPlanV2(
        schema_version="intervention_plan_v2",
        plan_id=pid,
        campaign_id=str(campaign_id),
        version=1,
        parent_plan_version=None,
        created_at_generation=int(created_at_generation),
        expectation_note=str(expectation_note),
        commands=[],
        frozen=False,
    )
    write_plan_version(run_dir, plan)
    return plan


def freeze_if_needed(plan: InterventionPlanV2, *, current_generation: int) -> InterventionPlanV2:
    """Return a new plan instance with frozen=True if PLAN-FREEZE-01 triggers."""
    G = int(current_generation)
    if plan.frozen:
        return plan
    for c in plan.commands:
        if int(c.target_generation) <= G:
            return InterventionPlanV2(
                schema_version=plan.schema_version,
                plan_id=plan.plan_id,
                campaign_id=plan.campaign_id,
                version=plan.version,
                parent_plan_version=plan.parent_plan_version,
                created_at_generation=plan.created_at_generation,
                expectation_note=plan.expectation_note,
                commands=plan.commands,
                frozen=True,
            )
    return plan


def with_updated_commands(plan: InterventionPlanV2, *, new_version: int, expectation_note: str, commands: List[CommandRef]) -> InterventionPlanV2:
    if not expectation_note:
        raise ValueError("expectation_note is required for plan version increment")
    return InterventionPlanV2(
        schema_version="intervention_plan_v2",
        plan_id=plan.plan_id,
        campaign_id=plan.campaign_id,
        version=int(new_version),
        parent_plan_version=int(plan.version),
        created_at_generation=plan.created_at_generation,
        expectation_note=str(expectation_note),
        commands=list(commands),
        frozen=False,
    )


def roll_forward_plan(
    run_dir: str,
    *,
    current_generation: int,
    expectation_note: str,
) -> InterventionPlanV2:
    """Create a new active (unfrozen) plan version that keeps only future commands.

    This implements Option A (autoroll) while preserving Phase 2 invariants:
    - existing plan versions are immutable (append-only)
    - expectation_note is required for any version increment
    - PLAN-FREEZE-01 remains intact (old frozen versions stay frozen)
    """
    if not expectation_note:
        raise ValueError("expectation_note is required")

    plan = load_current_plan(run_dir)
    if plan is None:
        raise ValueError("missing current plan")

    G = int(current_generation)

    future_cmds: List[CommandRef] = []
    for c in (plan.commands or []):
        try:
            if int(c.target_generation) > G:
                future_cmds.append(c)
        except Exception:
            continue

    new_version = int(plan.version) + 1
    new_plan = with_updated_commands(
        plan,
        new_version=new_version,
        expectation_note=str(expectation_note),
        commands=future_cmds,
    )
    write_plan_version(run_dir, new_plan)
    return new_plan
