from __future__ import annotations

"""CXXGEN-0 native generation-island contracts.

Default-off prototype metadata and validation helpers. This module deliberately
does not integrate a native path into the live runner. It records the smallest
native-generation boundary that can be exercised safely today, plus the state
ownership still required before a production science workload can run without
fallback.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from perf.cxxcore15_resident_soa_tick_island import SUMMARY_NAMES
from perf.cxxcore17_runner_generation_island import dependency_audit_payload


SCHEMA_VERSION = "cxxgen0_native_generation_island_v1"
CAPABILITY_SCHEMA_VERSION = "cxxgen0_resident_capability_set_v1"
EVENT_TAPE_SCHEMA_VERSION = "cxxgen0_typed_event_tape_v1"
CHECKPOINT_CURSOR_SCHEMA_VERSION = "cxxgen0_checkpoint_cursor_contract_v1"

DECISION_LABELS = (
    "CXXGEN0-A/DECISIVE-ARCHITECTURE-WIN",
    "CXXGEN0-B/PROMISING-MAJOR-PROTOTYPE",
    "CXXGEN0-C/HOST-REDUCTION-WIN-GPU-KERNEL-LIMIT",
    "CXXGEN0-D/BOUNDARY-CONVERSION-STILL-DOMINATES",
    "CXXGEN0-E/PARITY-OR-CHECKPOINT-BLOCKED",
    "CXXGEN0-F/IMPLEMENTATION-WIDTH-TOO-LARGE",
    "CXXGEN0-G/PROTOTYPE-SLOWER",
    "CXXGEN0-H/PARTIAL-PROTOTYPE-INCOMPLETE",
)

EVENT_TYPE_SUMMARY = 1
EVENT_TYPE_MOVE = 2
EVENT_TYPE_GATHER = 3
EVENT_TYPE_DEATH = 4


@dataclass(frozen=True)
class CapabilityRow:
    surface: str
    current_authority: str
    cxxgen0_authority: str
    support_status: str
    checkpoint_status: str
    fallback_required: bool
    blocker: str


@dataclass(frozen=True)
class CheckpointCursor:
    generation: int
    tick: int
    event_cursor: int
    sparse_summary_cursor: int
    slot_epoch: int
    rng_stream_digest: str
    action_cache_digest: str
    hmem_digest: str
    population_transaction_digest: str
    resident_state_digest: str


@dataclass(frozen=True)
class ResidentRuntimeStateV2:
    generation: int
    tick: int
    slot_epoch: int
    agent_ids: np.ndarray
    slot_tokens: np.ndarray
    alive_mask: np.ndarray
    energy: np.ndarray
    health: np.ndarray
    pos_y: np.ndarray
    pos_x: np.ndarray
    hmem: np.ndarray
    action_cache_cursor: np.ndarray
    event_cursor: int


@dataclass(frozen=True)
class PolicyExecutionPlanV1:
    generation: int
    membership_epoch: int
    policy_group_ids: np.ndarray
    slots: np.ndarray
    hmem_rows: np.ndarray
    output_scatter_slots: np.ndarray
    deterministic_backend: str
    invalidation_reasons: tuple[str, ...]


@dataclass(frozen=True)
class PolicyExecutionInputV1:
    plan: PolicyExecutionPlanV1
    observations: np.ndarray
    drives: np.ndarray
    gain: np.ndarray
    hmem: np.ndarray
    action_cache_cursor: np.ndarray
    cache_stride: int


@dataclass(frozen=True)
class PolicyExecutionResultV1:
    actions: np.ndarray
    logits: np.ndarray
    hmem_after: np.ndarray
    action_cache_cursor_after: np.ndarray


@dataclass(frozen=True)
class PopulationTransactionTapeV1:
    generation: int
    transaction_type: np.ndarray
    parent_slot: np.ndarray
    child_slot: np.ndarray
    agent_id: np.ndarray
    slot_token_before: np.ndarray
    slot_token_after: np.ndarray
    event_cursor_after: np.ndarray


def capability_rows() -> list[CapabilityRow]:
    return [
        CapabilityRow(
            "agent scalar arrays and grid positions",
            "GridWorld arrays plus Python Agent mirrors",
            "resident primitive arrays inside native candidate",
            "supported_in_captured_no_rng_window",
            "materialize_before_checkpoint",
            False,
            "",
        ),
        CapabilityRow(
            "precomputed actions and scalar costs",
            "Python policy/cost path before env apply",
            "read-only action/cost tapes",
            "supported_when_precomputed",
            "action tape must be replayable or checkpointed",
            False,
            "",
        ),
        CapabilityRow(
            "typed lineage/report events",
            "Python dictionaries/CSV/JSONL rows",
            "typed event tape contract with compatibility export boundary",
            "prototype_contract_only",
            "cursor and summary state required",
            True,
            "current lineage writer still owns public compatibility bytes",
        ),
        CapabilityRow(
            "GRU policy Hmem and action cache",
            "Python/Torch policy execution plus Python cache dictionaries",
            "deterministic resident policy contract owns logits/actions/Hmem/cache cursor; production Torch/GRU remains unowned",
            "contract_slice_supported_production_unsupported",
            "contract payload round-trips; live runner restore/rebuild still required",
            True,
            "production policy groups, Torch/GRU kernels, and checkpoint rebuild are outside the CXXGEN0 native contract slice",
        ),
        CapabilityRow(
            "WDL delayed hazard state",
            "GridWorld/Patch Python-owned arrays and delayed queues",
            "not owned by CXXGEN0 candidate",
            "unsupported",
            "delayed queues and RNG continuation not represented",
            True,
            "WDL state is not yet primitive-authoritative at the native boundary",
        ),
        CapabilityRow(
            "birth/death/cull/refill lifecycle",
            "Python reproduction and population objects",
            "death mask only; no birth transaction ownership",
            "death_summary_supported_births_unsupported",
            "transaction tape required",
            True,
            "accepted science campaigns need checkpoint-safe population transactions",
        ),
        CapabilityRow(
            "checkpoint bundle v2 continuation",
            "Python runner/pop/env object graph",
            "cursor contract only",
            "prototype_contract_only",
            "resident state must be authoritative or materialized exactly",
            True,
            "no native resident checkpoint schema exists for full generation ownership",
        ),
        CapabilityRow(
            "Patch/Jungle/migration/juvenile/developmental optional state",
            "mixed Python objects and arrays",
            "not owned by CXXGEN0 candidate",
            "unsupported",
            "feature-specific state schemas required",
            True,
            "current support envelope excludes real scientific workload families",
        ),
    ]


def capability_matrix_payload() -> dict[str, Any]:
    rows = [row.__dict__ for row in capability_rows()]
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "fallback_required_count": sum(1 for row in rows if row["fallback_required"]),
            "supported_count": sum(1 for row in rows if row["support_status"].startswith("supported")),
            "prototype_contract_count": sum(1 for row in rows if row["support_status"] == "prototype_contract_only"),
            "unsupported_count": sum(1 for row in rows if row["support_status"] == "unsupported"),
        },
    }


def typed_event_tape_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": EVENT_TAPE_SCHEMA_VERSION,
        "event_types": {
            str(EVENT_TYPE_SUMMARY): "generation_summary",
            str(EVENT_TYPE_MOVE): "agent_moved",
            str(EVENT_TYPE_GATHER): "agent_gathered_food",
            str(EVENT_TYPE_DEATH): "agent_died",
        },
        "columns": [
            {"name": "event_type", "dtype": "int32", "required": True},
            {"name": "tick", "dtype": "int32", "required": True},
            {"name": "agent_index", "dtype": "int32", "required": True, "missing": -1},
            {"name": "value0", "dtype": "float32", "required": True},
            {"name": "value1", "dtype": "float32", "required": True},
        ],
        "summary_fields": list(SUMMARY_NAMES)
        + ["action_count_0", "action_count_1", "action_count_2", "action_count_3", "action_count_4", "action_count_5", "action_count_6"],
        "compatibility_mode": "export JSONL/CSV at a controlled boundary after exact parity checks",
        "native_evidence_mode": "retain primitive typed rows and canonical manifests; export public formats only on demand",
        "checkpoint_required_state": [
            "event_cursor",
            "sparse_summary_cursor",
            "summary accumulators",
            "schema version",
            "row order",
        ],
    }


def checkpoint_cursor_payload(cursor: CheckpointCursor | None = None) -> dict[str, Any]:
    if cursor is None:
        cursor = CheckpointCursor(
            generation=0,
            tick=0,
            event_cursor=0,
            sparse_summary_cursor=0,
            slot_epoch=0,
            rng_stream_digest="unowned",
            action_cache_digest="unowned",
            hmem_digest="unowned",
            population_transaction_digest="unowned",
            resident_state_digest="unmaterialized",
        )
    return {
        "schema_version": CHECKPOINT_CURSOR_SCHEMA_VERSION,
        "cursor": cursor.__dict__,
        "strict_resume_requirements": [
            "same final scalar state after save/load/resume",
            "same RNG continuation for policy, environment, mutation, and reproduction streams",
            "same Hmem and action cache after first resumed tick",
            "same population transaction cursor and slot-token mapping",
            "same event cursor with no lost final-buffer events",
        ],
        "cxxgen0_status": "contract_only",
    }


def resident_runtime_state_payload(state: ResidentRuntimeStateV2) -> dict[str, Any]:
    return {
        "schema_version": "cxxgen0_resident_runtime_state_v2",
        "generation": int(state.generation),
        "tick": int(state.tick),
        "slot_epoch": int(state.slot_epoch),
        "agent_ids": _array_payload(state.agent_ids),
        "slot_tokens": _array_payload(state.slot_tokens),
        "alive_mask": _array_payload(np.asarray(state.alive_mask, dtype=np.uint8)),
        "energy": _array_payload(state.energy),
        "health": _array_payload(state.health),
        "pos_y": _array_payload(state.pos_y),
        "pos_x": _array_payload(state.pos_x),
        "hmem": _array_payload(state.hmem),
        "action_cache_cursor": _array_payload(state.action_cache_cursor),
        "event_cursor": int(state.event_cursor),
    }


def policy_execution_plan_payload(plan: PolicyExecutionPlanV1) -> dict[str, Any]:
    return {
        "schema_version": "cxxgen0_policy_execution_plan_v1",
        "generation": int(plan.generation),
        "membership_epoch": int(plan.membership_epoch),
        "policy_group_ids": _array_payload(plan.policy_group_ids),
        "slots": _array_payload(plan.slots),
        "hmem_rows": _array_payload(plan.hmem_rows),
        "output_scatter_slots": _array_payload(plan.output_scatter_slots),
        "deterministic_backend": str(plan.deterministic_backend),
        "invalidation_reasons": list(plan.invalidation_reasons),
    }


def policy_execution_input_payload(policy_input: PolicyExecutionInputV1) -> dict[str, Any]:
    return {
        "schema_version": "cxxgen0_policy_execution_input_v1",
        "plan": policy_execution_plan_payload(policy_input.plan),
        "observations": _array_payload(policy_input.observations),
        "drives": _array_payload(policy_input.drives),
        "gain": _array_payload(policy_input.gain),
        "hmem": _array_payload(policy_input.hmem),
        "action_cache_cursor": _array_payload(policy_input.action_cache_cursor),
        "cache_stride": int(policy_input.cache_stride),
    }


def policy_execution_result_payload(result: PolicyExecutionResultV1) -> dict[str, Any]:
    return {
        "schema_version": "cxxgen0_policy_execution_result_v1",
        "actions": _array_payload(result.actions),
        "logits": _array_payload(result.logits),
        "hmem_after": _array_payload(result.hmem_after),
        "action_cache_cursor_after": _array_payload(result.action_cache_cursor_after),
    }


def population_transaction_tape_payload(tape: PopulationTransactionTapeV1) -> dict[str, Any]:
    return {
        "schema_version": "cxxgen0_population_transaction_tape_v1",
        "generation": int(tape.generation),
        "transaction_type": _array_payload(tape.transaction_type),
        "parent_slot": _array_payload(tape.parent_slot),
        "child_slot": _array_payload(tape.child_slot),
        "agent_id": _array_payload(tape.agent_id),
        "slot_token_before": _array_payload(tape.slot_token_before),
        "slot_token_after": _array_payload(tape.slot_token_after),
        "event_cursor_after": _array_payload(tape.event_cursor_after),
        "transaction_count": int(np.asarray(tape.transaction_type).size),
    }


def checkpoint_roundtrip_contract_payload(
    *,
    state: ResidentRuntimeStateV2,
    policy_plan: PolicyExecutionPlanV1,
    transaction_tape: PopulationTransactionTapeV1,
    event_tape: Mapping[str, Any],
) -> dict[str, Any]:
    state_payload = resident_runtime_state_payload(state)
    plan_payload = policy_execution_plan_payload(policy_plan)
    transaction_payload = population_transaction_tape_payload(transaction_tape)
    event_payload = {key: _array_payload(value) for key, value in event_tape.items()}
    payload = {
        "schema_version": "cxxgen0_checkpoint_roundtrip_contract_v1",
        "resident_state": state_payload,
        "policy_execution_plan": plan_payload,
        "population_transaction_tape": transaction_payload,
        "event_tape": event_payload,
    }
    payload["digest"] = stable_payload_digest(payload)
    return payload


def validate_resident_state(state: ResidentRuntimeStateV2) -> dict[str, Any]:
    n = int(np.asarray(state.agent_ids).shape[0])
    checks = {
        "slot_tokens": int(np.asarray(state.slot_tokens).shape[0]) == n,
        "alive_mask": int(np.asarray(state.alive_mask).shape[0]) == n,
        "energy": int(np.asarray(state.energy).shape[0]) == n,
        "health": int(np.asarray(state.health).shape[0]) == n,
        "pos_y": int(np.asarray(state.pos_y).shape[0]) == n,
        "pos_x": int(np.asarray(state.pos_x).shape[0]) == n,
        "action_cache_cursor": int(np.asarray(state.action_cache_cursor).shape[0]) == n,
        "hmem_rows": np.asarray(state.hmem).ndim == 2 and int(np.asarray(state.hmem).shape[0]) == n,
        "event_cursor_nonnegative": int(state.event_cursor) >= 0,
    }
    return {"ok": all(checks.values()), "checks": checks, "agents": n}


def validate_policy_plan(state: ResidentRuntimeStateV2, plan: PolicyExecutionPlanV1) -> dict[str, Any]:
    slots = np.asarray(plan.slots, dtype=np.int32)
    hmem_rows = np.asarray(plan.hmem_rows, dtype=np.int32)
    scatter = np.asarray(plan.output_scatter_slots, dtype=np.int32)
    n = int(np.asarray(state.agent_ids).shape[0])
    checks = {
        "membership_epoch_matches": int(plan.membership_epoch) == int(state.slot_epoch),
        "slot_shapes": slots.shape == hmem_rows.shape == scatter.shape == np.asarray(plan.policy_group_ids).shape,
        "slots_in_range": bool(np.all((slots >= 0) & (slots < n))) if slots.size else True,
        "scatter_in_range": bool(np.all((scatter >= 0) & (scatter < n))) if scatter.size else True,
        "hmem_rows_valid": bool(np.all((hmem_rows >= -1) & (hmem_rows < n))) if hmem_rows.size else True,
        "backend_named": bool(plan.deterministic_backend),
    }
    return {"ok": all(checks.values()), "checks": checks, "rows": int(slots.size)}


def run_policy_execution_plan_reference(policy_input: PolicyExecutionInputV1) -> PolicyExecutionResultV1:
    obs = np.asarray(policy_input.observations, dtype=np.float32)
    drives = np.asarray(policy_input.drives, dtype=np.float32)
    gain = np.asarray(policy_input.gain, dtype=np.float32).reshape(-1)
    hmem = np.asarray(policy_input.hmem, dtype=np.float32)
    cursors = np.asarray(policy_input.action_cache_cursor, dtype=np.int32).copy()
    if obs.ndim != 2 or drives.ndim != 2 or hmem.ndim != 2:
        raise ValueError("policy input arrays must be 2D except gain/cursors")
    n = int(obs.shape[0])
    if drives.shape[0] != n or hmem.shape[0] != n or gain.shape[0] != n or cursors.shape[0] != n:
        raise ValueError("policy input row mismatch")
    logits = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        o0 = float(obs[i, 0]) if obs.shape[1] > 0 else 0.0
        o1 = float(obs[i, 1]) if obs.shape[1] > 1 else 0.0
        o2 = float(obs[i, 2]) if obs.shape[1] > 2 else 0.0
        o3 = float(obs[i, 3]) if obs.shape[1] > 3 else 0.0
        d0 = float(drives[i, 0]) if drives.shape[1] > 0 else 0.0
        d1 = float(drives[i, 1]) if drives.shape[1] > 1 else 0.0
        d2 = float(drives[i, 2]) if drives.shape[1] > 2 else 0.0
        d3 = float(drives[i, 3]) if drives.shape[1] > 3 else 0.0
        h0 = float(hmem[i, 0]) if hmem.shape[1] > 0 else 0.0
        h1 = float(hmem[i, 1]) if hmem.shape[1] > 1 else 0.0
        g = float(gain[i])
        logits[i, 0] = np.float32(o0 - d2)
        logits[i, 1] = np.float32(o1 - d2)
        logits[i, 2] = np.float32(o2 - d1)
        logits[i, 3] = np.float32(o3 - d1)
        logits[i, 4] = np.float32((o0 + o1 + d0) * g)
        logits[i, 5] = np.float32(d2 + h0)
        logits[i, 6] = np.float32(d3 + h1)
    actions = np.argmax(logits, axis=1).astype(np.int32)
    hmem_after = np.asarray(hmem, dtype=np.float32).copy()
    for i in range(n):
        max_logit = float(logits[i, int(actions[i])])
        if hmem_after.shape[1] > 0:
            hmem_after[i, 0] = np.float32(0.75 * float(hmem_after[i, 0]) + 0.05 * float(obs[i, 0]) + 0.01 * int(actions[i]))
        if hmem_after.shape[1] > 1:
            hmem_after[i, 1] = np.float32(0.75 * float(hmem_after[i, 1]) + 0.05 * max_logit)
        for j in range(2, hmem_after.shape[1]):
            hmem_after[i, j] = np.float32(0.75 * float(hmem_after[i, j]))
    stride = int(policy_input.cache_stride)
    if stride > 0:
        cursors = ((cursors.astype(np.int64) + 1) % stride).astype(np.int32)
    else:
        cursors = (cursors + np.int32(1)).astype(np.int32)
    return PolicyExecutionResultV1(
        actions=np.ascontiguousarray(actions),
        logits=np.ascontiguousarray(logits),
        hmem_after=np.ascontiguousarray(hmem_after),
        action_cache_cursor_after=np.ascontiguousarray(cursors),
    )


def sample_policy_execution_input() -> PolicyExecutionInputV1:
    plan = PolicyExecutionPlanV1(
        generation=3,
        membership_epoch=11,
        policy_group_ids=np.asarray([1, 1, 2], dtype=np.int32),
        slots=np.asarray([0, 1, 2], dtype=np.int32),
        hmem_rows=np.asarray([0, 1, 2], dtype=np.int32),
        output_scatter_slots=np.asarray([0, 1, 2], dtype=np.int32),
        deterministic_backend="cxxgen0_resident_reference",
        invalidation_reasons=("birth", "death", "slot_token_change", "policy_shape_change"),
    )
    return PolicyExecutionInputV1(
        plan=plan,
        observations=np.asarray(
            [
                [0.2, 0.1, 0.0, 0.4],
                [0.0, 0.5, 0.2, 0.1],
                [0.4, 0.0, 0.3, 0.2],
            ],
            dtype=np.float32,
        ),
        drives=np.asarray(
            [
                [0.8, 0.1, 0.2, 0.0],
                [0.1, 0.2, 0.9, 0.1],
                [0.0, 0.8, 0.1, 0.7],
            ],
            dtype=np.float32,
        ),
        gain=np.asarray([1.0, 1.2, 0.8], dtype=np.float32),
        hmem=np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        action_cache_cursor=np.asarray([0, 1, 2], dtype=np.int32),
        cache_stride=4,
    )


def sample_policy_execution_payload() -> dict[str, Any]:
    policy_input = sample_policy_execution_input()
    result = run_policy_execution_plan_reference(policy_input)
    payload = {
        "schema_version": "cxxgen0_policy_execution_reference_v1",
        "input": policy_execution_input_payload(policy_input),
        "result": policy_execution_result_payload(result),
    }
    payload["digest"] = stable_payload_digest(payload)
    return payload


def validate_population_transaction_tape(
    state: ResidentRuntimeStateV2,
    tape: PopulationTransactionTapeV1,
) -> dict[str, Any]:
    tx = np.asarray(tape.transaction_type, dtype=np.int32)
    fields = [
        np.asarray(tape.parent_slot),
        np.asarray(tape.child_slot),
        np.asarray(tape.agent_id),
        np.asarray(tape.slot_token_before),
        np.asarray(tape.slot_token_after),
        np.asarray(tape.event_cursor_after),
    ]
    n = int(np.asarray(state.agent_ids).shape[0])
    checks = {
        "field_shapes": all(field.shape == tx.shape for field in fields),
        "known_transaction_types": bool(np.all((tx == 1) | (tx == 2) | (tx == 3))) if tx.size else True,
        "parent_slots_valid": _slots_valid(np.asarray(tape.parent_slot, dtype=np.int32), n, allow_missing=True),
        "child_slots_valid": _slots_valid(np.asarray(tape.child_slot, dtype=np.int32), n, allow_missing=True),
        "event_cursors_monotonic": bool(np.all(np.diff(np.asarray(tape.event_cursor_after, dtype=np.int64)) >= 0)) if tx.size > 1 else True,
    }
    return {"ok": all(checks.values()), "checks": checks, "transactions": int(tx.size)}


def apply_population_transaction_tape(
    state: ResidentRuntimeStateV2,
    tape: PopulationTransactionTapeV1,
) -> ResidentRuntimeStateV2:
    validation = validate_population_transaction_tape(state, tape)
    if not validation["ok"]:
        raise ValueError("invalid population transaction tape")
    agent_ids = np.asarray(state.agent_ids, dtype=np.int32).copy()
    slot_tokens = np.asarray(state.slot_tokens, dtype=np.int64).copy()
    alive = np.asarray(state.alive_mask, dtype=np.uint8).copy()
    energy = np.asarray(state.energy, dtype=np.float32).copy()
    health = np.asarray(state.health, dtype=np.float32).copy()
    hmem = np.asarray(state.hmem, dtype=np.float32).copy()
    action_cache_cursor = np.asarray(state.action_cache_cursor, dtype=np.int32).copy()
    event_cursor = int(state.event_cursor)

    tx = np.asarray(tape.transaction_type, dtype=np.int32)
    parent_slot = np.asarray(tape.parent_slot, dtype=np.int32)
    child_slot = np.asarray(tape.child_slot, dtype=np.int32)
    agent_id = np.asarray(tape.agent_id, dtype=np.int32)
    token_after = np.asarray(tape.slot_token_after, dtype=np.int64)
    event_after = np.asarray(tape.event_cursor_after, dtype=np.int64)
    for i, kind in enumerate(tx.tolist()):
        if kind == 1:
            slot = int(child_slot[i])
            if slot < 0:
                raise ValueError("birth transaction requires child slot")
            agent_ids[slot] = np.int32(agent_id[i])
            slot_tokens[slot] = np.int64(token_after[i])
            alive[slot] = np.uint8(1)
            energy[slot] = np.float32(max(float(energy[slot]), 1.0))
            health[slot] = np.float32(max(float(health[slot]), 1.0))
            hmem[slot, :] = np.float32(0.0)
            action_cache_cursor[slot] = np.int32(0)
        elif kind == 2:
            slot = int(parent_slot[i])
            if slot < 0:
                raise ValueError("death transaction requires parent slot")
            slot_tokens[slot] = np.int64(token_after[i])
            alive[slot] = np.uint8(0)
            energy[slot] = np.float32(0.0)
            health[slot] = np.float32(0.0)
            hmem[slot, :] = np.float32(0.0)
            action_cache_cursor[slot] = np.int32(0)
        elif kind == 3:
            slot = int(parent_slot[i])
            if slot < 0:
                raise ValueError("cull transaction requires parent slot")
            slot_tokens[slot] = np.int64(token_after[i])
            alive[slot] = np.uint8(0)
            energy[slot] = np.float32(0.0)
            health[slot] = np.float32(0.0)
            action_cache_cursor[slot] = np.int32(0)
        event_cursor = int(event_after[i])
    return ResidentRuntimeStateV2(
        generation=state.generation,
        tick=state.tick,
        slot_epoch=int(state.slot_epoch) + int(tx.size),
        agent_ids=agent_ids,
        slot_tokens=slot_tokens,
        alive_mask=alive,
        energy=energy,
        health=health,
        pos_y=np.asarray(state.pos_y, dtype=np.int32).copy(),
        pos_x=np.asarray(state.pos_x, dtype=np.int32).copy(),
        hmem=hmem,
        action_cache_cursor=action_cache_cursor,
        event_cursor=event_cursor,
    )


def sample_checkpoint_contract_payload() -> dict[str, Any]:
    state = ResidentRuntimeStateV2(
        generation=3,
        tick=8,
        slot_epoch=11,
        agent_ids=np.asarray([101, 102, 103], dtype=np.int32),
        slot_tokens=np.asarray([5, 7, 9], dtype=np.int64),
        alive_mask=np.asarray([1, 1, 0], dtype=np.uint8),
        energy=np.asarray([9.5, 2.0, 0.0], dtype=np.float32),
        health=np.asarray([10.0, 4.0, 0.0], dtype=np.float32),
        pos_y=np.asarray([1, 2, 3], dtype=np.int32),
        pos_x=np.asarray([4, 5, 6], dtype=np.int32),
        hmem=np.asarray([[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]], dtype=np.float32),
        action_cache_cursor=np.asarray([0, 2, 0], dtype=np.int32),
        event_cursor=4,
    )
    plan = PolicyExecutionPlanV1(
        generation=3,
        membership_epoch=11,
        policy_group_ids=np.asarray([1, 1, 2], dtype=np.int32),
        slots=np.asarray([0, 1, 2], dtype=np.int32),
        hmem_rows=np.asarray([0, 1, -1], dtype=np.int32),
        output_scatter_slots=np.asarray([0, 1, 2], dtype=np.int32),
        deterministic_backend="numpy_reference",
        invalidation_reasons=("birth", "death", "policy_shape_change"),
    )
    transactions = PopulationTransactionTapeV1(
        generation=3,
        transaction_type=np.asarray([1, 2], dtype=np.int32),
        parent_slot=np.asarray([0, 2], dtype=np.int32),
        child_slot=np.asarray([2, -1], dtype=np.int32),
        agent_id=np.asarray([104, 103], dtype=np.int32),
        slot_token_before=np.asarray([0, 9], dtype=np.int64),
        slot_token_after=np.asarray([13, 10], dtype=np.int64),
        event_cursor_after=np.asarray([5, 6], dtype=np.int64),
    )
    event_tape = event_tape_from_native_arrays(
        actions_by_tick=np.asarray([[0, 4, 5], [1, 4, 2]], dtype=np.int32),
        pre_energy=np.asarray([10.0, 8.0, 4.0], dtype=np.float32),
        post_energy=np.asarray([8.0, 10.0, 0.0], dtype=np.float32),
        died_mask=np.asarray([0, 0, 1], dtype=np.uint8),
        moved_mask=np.asarray([1, 0, 1], dtype=np.uint8),
        food_gained=np.asarray([0.0, 4.0, 0.0], dtype=np.float32),
    )
    return checkpoint_roundtrip_contract_payload(
        state=state,
        policy_plan=plan,
        transaction_tape=transactions,
        event_tape=event_tape,
    )


def sample_transaction_replay_payload() -> dict[str, Any]:
    payload = sample_checkpoint_contract_payload()
    state = _state_from_payload(payload["resident_state"])
    tape = _transaction_tape_from_payload(payload["population_transaction_tape"])
    result_state = apply_population_transaction_tape(state, tape)
    result_payload = resident_runtime_state_payload(result_state)
    return {
        "schema_version": "cxxgen0_population_transaction_replay_v1",
        "input_digest": payload["digest"],
        "input_state_validation": validate_resident_state(state),
        "transaction_validation": validate_population_transaction_tape(state, tape),
        "result_state_validation": validate_resident_state(result_state),
        "result_state": result_payload,
        "result_digest": stable_payload_digest(result_payload),
    }


def support_blocker_payload() -> dict[str, Any]:
    dependency = dependency_audit_payload()
    blockers = []
    for row in dependency.get("rows", []):
        if bool(row.get("blocks_resident_island")):
            blockers.append(
                {
                    "surface": row.get("dependency"),
                    "classification": row.get("classification"),
                    "current_owner": row.get("current_owner"),
                    "reason": row.get("notes"),
                }
            )
    for row in capability_rows():
        if row.fallback_required:
            blockers.append(
                {
                    "surface": row.surface,
                    "classification": row.support_status,
                    "current_owner": row.current_authority,
                    "reason": row.blocker,
                }
            )
    return {
        "schema_version": "cxxgen0_support_blocker_matrix_v1",
        "source_dependency_schema": dependency.get("schema_version"),
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def event_tape_from_native_arrays(
    *,
    actions_by_tick: Any,
    pre_energy: Any,
    post_energy: Any,
    died_mask: Any,
    moved_mask: Any,
    food_gained: Any,
) -> dict[str, np.ndarray]:
    actions = np.asarray(actions_by_tick, dtype=np.int32)
    pre = np.asarray(pre_energy, dtype=np.float32).reshape(-1)
    post = np.asarray(post_energy, dtype=np.float32).reshape(-1)
    died = np.asarray(died_mask, dtype=bool).reshape(-1)
    moved = np.asarray(moved_mask, dtype=bool).reshape(-1)
    food = np.asarray(food_gained, dtype=np.float32).reshape(-1)
    rows: list[tuple[int, int, int, float, float]] = [
        (EVENT_TYPE_SUMMARY, int(actions.shape[0]), -1, float(np.sum(post - pre, dtype=np.float64)), float(np.count_nonzero(died)))
    ]
    for agent in range(int(post.shape[0])):
        if bool(moved[agent]):
            rows.append((EVENT_TYPE_MOVE, -1, agent, 1.0, 0.0))
        if float(food[agent]) != 0.0:
            rows.append((EVENT_TYPE_GATHER, -1, agent, float(food[agent]), 0.0))
        if bool(died[agent]):
            rows.append((EVENT_TYPE_DEATH, -1, agent, float(post[agent]), 0.0))
    return {
        "event_type": np.asarray([row[0] for row in rows], dtype=np.int32),
        "tick": np.asarray([row[1] for row in rows], dtype=np.int32),
        "agent_index": np.asarray([row[2] for row in rows], dtype=np.int32),
        "value0": np.asarray([row[3] for row in rows], dtype=np.float32),
        "value1": np.asarray([row[4] for row in rows], dtype=np.float32),
    }


def event_tape_digest(tape: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ("event_type", "tick", "agent_index", "value0", "value1"):
        arr = np.ascontiguousarray(tape[key])
        h.update(key.encode("ascii"))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def compare_event_tapes(expected: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("event_type", "tick", "agent_index", "value0", "value1"):
        a = np.asarray(expected[key])
        b = np.asarray(candidate[key])
        if a.shape != b.shape:
            return {"ok": False, "first_divergence": {"field": key, "reason": "shape", "expected": list(a.shape), "candidate": list(b.shape)}}
        if a.dtype.kind == "f":
            same = np.allclose(a, b, atol=1.0e-6, rtol=0.0)
        else:
            same = bool(np.array_equal(a, b))
        if not same:
            idx = int(np.flatnonzero(np.ravel(a != b))[0])
            return {
                "ok": False,
                "first_divergence": {
                    "field": key,
                    "reason": "value",
                    "index": idx,
                    "expected": _json_scalar(np.ravel(a)[idx]),
                    "candidate": _json_scalar(np.ravel(b)[idx]),
                },
            }
    return {"ok": True, "first_divergence": None, "digest": event_tape_digest(candidate)}


def decide_cxxgen0(payload: Mapping[str, Any]) -> str:
    if not bool(payload.get("native_available")):
        return "CXXGEN0-H/PARTIAL-PROTOTYPE-INCOMPLETE"
    if not bool(payload.get("parity_clean")):
        return "CXXGEN0-E/PARITY-OR-CHECKPOINT-BLOCKED"
    if int(payload.get("full_generation_blocker_count", 0)) > 0:
        return "CXXGEN0-E/PARITY-OR-CHECKPOINT-BLOCKED"
    if float(payload.get("median_speedup_vs_baseline", 0.0)) >= 1.30:
        return "CXXGEN0-B/PROMISING-MAJOR-PROTOTYPE"
    if bool(payload.get("boundary_conversion_dominates")):
        return "CXXGEN0-D/BOUNDARY-CONVERSION-STILL-DOMINATES"
    return "CXXGEN0-G/PROTOTYPE-SLOWER"


def stable_payload_digest(payload: Mapping[str, Any]) -> str:
    without_digest = dict(payload)
    without_digest.pop("digest", None)
    data = json.dumps(without_digest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_scalar)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verify_checkpoint_roundtrip_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = str(payload.get("digest", ""))
    actual = stable_payload_digest(payload)
    return {"ok": bool(expected and expected == actual), "expected": expected, "actual": actual}


def _slots_valid(slots: np.ndarray, n: int, *, allow_missing: bool) -> bool:
    if slots.size == 0:
        return True
    lower_ok = slots >= (-1 if allow_missing else 0)
    upper_ok = slots < n
    return bool(np.all(lower_ok & upper_ok))


def _state_from_payload(payload: Mapping[str, Any]) -> ResidentRuntimeStateV2:
    return ResidentRuntimeStateV2(
        generation=int(payload["generation"]),
        tick=int(payload["tick"]),
        slot_epoch=int(payload["slot_epoch"]),
        agent_ids=_array_from_payload(payload["agent_ids"], np.int32),
        slot_tokens=_array_from_payload(payload["slot_tokens"], np.int64),
        alive_mask=_array_from_payload(payload["alive_mask"], np.uint8),
        energy=_array_from_payload(payload["energy"], np.float32),
        health=_array_from_payload(payload["health"], np.float32),
        pos_y=_array_from_payload(payload["pos_y"], np.int32),
        pos_x=_array_from_payload(payload["pos_x"], np.int32),
        hmem=_array_from_payload(payload["hmem"], np.float32),
        action_cache_cursor=_array_from_payload(payload["action_cache_cursor"], np.int32),
        event_cursor=int(payload["event_cursor"]),
    )


def _transaction_tape_from_payload(payload: Mapping[str, Any]) -> PopulationTransactionTapeV1:
    return PopulationTransactionTapeV1(
        generation=int(payload["generation"]),
        transaction_type=_array_from_payload(payload["transaction_type"], np.int32),
        parent_slot=_array_from_payload(payload["parent_slot"], np.int32),
        child_slot=_array_from_payload(payload["child_slot"], np.int32),
        agent_id=_array_from_payload(payload["agent_id"], np.int32),
        slot_token_before=_array_from_payload(payload["slot_token_before"], np.int64),
        slot_token_after=_array_from_payload(payload["slot_token_after"], np.int64),
        event_cursor_after=_array_from_payload(payload["event_cursor_after"], np.int64),
    )


def _array_from_payload(payload: Mapping[str, Any], dtype: Any) -> np.ndarray:
    arr = np.asarray(payload["values"], dtype=dtype)
    return np.ascontiguousarray(arr.reshape(tuple(int(x) for x in payload["shape"])))


def _array_payload(value: Any) -> dict[str, Any]:
    arr = np.ascontiguousarray(value)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "sha256": _array_digest(arr),
        "values": arr.tolist(),
    }


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _json_scalar(value: Any) -> int | float | bool | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


__all__ = [
    "CHECKPOINT_CURSOR_SCHEMA_VERSION",
    "DECISION_LABELS",
    "EVENT_TAPE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "CapabilityRow",
    "CheckpointCursor",
    "PolicyExecutionPlanV1",
    "PolicyExecutionInputV1",
    "PolicyExecutionResultV1",
    "PopulationTransactionTapeV1",
    "ResidentRuntimeStateV2",
    "capability_matrix_payload",
    "capability_rows",
    "checkpoint_cursor_payload",
    "checkpoint_roundtrip_contract_payload",
    "compare_event_tapes",
    "decide_cxxgen0",
    "event_tape_digest",
    "event_tape_from_native_arrays",
    "policy_execution_plan_payload",
    "policy_execution_input_payload",
    "policy_execution_result_payload",
    "population_transaction_tape_payload",
    "resident_runtime_state_payload",
    "sample_checkpoint_contract_payload",
    "sample_policy_execution_input",
    "sample_policy_execution_payload",
    "sample_transaction_replay_payload",
    "support_blocker_payload",
    "typed_event_tape_contract_payload",
    "validate_policy_plan",
    "validate_population_transaction_tape",
    "validate_resident_state",
    "verify_checkpoint_roundtrip_payload",
]
