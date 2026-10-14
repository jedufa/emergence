from __future__ import annotations

"""CXXBACKBONE-8 resident flat/Patch generation-island contract.

The CXXBACKBONE-8 acceptance rule requires a production-wired generation island
with no Python grouping, row maps, action application, state update rows, or
per-refill action D2H inside the claimed lane. This module captures the real
RunnerCore resident state and fails closed until those counters can be zero.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from simulation.runnercore.tick_state import RunnerCore1TickState, alive_agent_ids


SCHEMA_VERSION = "cxxbackbone8_generation_island_v1"
STATE_SCHEMA_VERSION = "resident_runtime_state_v8"
RESULT_SCHEMA_VERSION = "cxxbackbone8_generation_island_result_v1"
SUMMARY_SCHEMA_VERSION = "cxxbackbone8_generation_island_summary_v1"
DECISION_INVALID = "CXXBACKBONE8-G/INVALID-WRAPPER-OR-DIAGNOSTIC-CLOSEOUT"
DECISION_READY = "CXXBACKBONE8-A/RESIDENT-FLAT-PATCH-GENERATION-ISLAND-CLEAN"

MANDATORY_ZERO_COUNTERS = (
    "python_grouping_calls",
    "python_row_map_calls",
    "python_action_applications",
    "python_action_state_update_rows",
    "per_refill_cuda_action_d2h_copies",
    "hidden_fallback_count",
)


@dataclass(frozen=True)
class ResidentRuntimeStateV8:
    schema_version: str
    generation: int
    tick: int
    cell_id: str
    device: str
    width: int
    height: int
    steps_per_generation: int
    support_envelope: str
    alive_agent_ids: np.ndarray
    alive_slots: np.ndarray
    slot_tokens: np.ndarray
    agent_id_to_slot: np.ndarray
    slot_to_agent_id: np.ndarray
    use_gru: np.ndarray
    mlp_rows: np.ndarray
    gru_rows: np.ndarray
    cache_agent_id: np.ndarray
    cache_token: np.ndarray
    cache_cursor: np.ndarray
    cache_actions: np.ndarray
    pos_y_by_id: np.ndarray
    pos_x_by_id: np.ndarray
    occupancy: np.ndarray
    energy_by_slot: np.ndarray
    health_by_slot: np.ndarray
    injury_by_slot: np.ndarray
    hmem_shape: tuple[int, ...]
    patch_arrays: Mapping[str, np.ndarray]
    capability_mask: Mapping[str, bool]
    mandatory_counters: Mapping[str, int]
    blockers: tuple[str, ...]
    state_digest: str


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _array_digest(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
    }


def _digest_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _patch_arrays(runner: Any) -> dict[str, np.ndarray]:
    env = getattr(runner, "env", None)
    out: dict[str, np.ndarray] = {}
    for name in (
        "patch_food_amount",
        "patch_fertility",
        "patch_cover",
        "patch_shelter",
        "patch_movement_cost",
        "patch_scent",
        "patch_waste",
        "patch_density",
        "patch_recovery",
    ):
        value = getattr(env, name, None)
        if value is not None:
            out[name] = np.asarray(value).copy()
    return out


def _hmem_shape(runner: Any) -> tuple[int, ...]:
    hmem = getattr(runner, "_Hmem", None)
    shape = getattr(hmem, "shape", ())
    return tuple(int(x) for x in shape)


def _mandatory_counters_from_buckets(buckets: Mapping[str, Any] | None) -> dict[str, int]:
    src = buckets if isinstance(buckets, Mapping) else {}
    return {
        "python_grouping_calls": int(float(src.get("cxxbackbone5_policy_grouping_python_calls", 0.0) or 0.0)),
        "python_row_map_calls": int(float(src.get("cxxbackbone5_policy_row_map_python_calls", 0.0) or 0.0)),
        "python_action_applications": int(float(src.get("cxxbackbone7_env_batch_action_application_calls", 0.0) or 0.0)),
        "python_action_state_update_rows": int(
            float(src.get("cxxbackbone7_python_action_application_state_update_rows", 0.0) or 0.0)
        ),
        "per_refill_cuda_action_d2h_copies": int(float(src.get("cxxbackbone5_cuda_to_cpu_action_copies", 0.0) or 0.0)),
        "hidden_fallback_count": int(float(src.get("cxxbackbone8_hidden_fallback_count", 0.0) or 0.0)),
    }


def _capability_mask(runner: Any) -> dict[str, bool]:
    cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    patch = _as_bool(getattr(env_cfg, "patch_world_enable", False))
    return {
        "production_run_ticks_gate": True,
        "flat_observation_exact_from_cxxbackbone7": True,
        "patch_observation_exact_from_cxxbackbone7": bool(patch or _as_bool(getattr(env_cfg, "patch_world_observation_enable", False))),
        "native_grouping_and_row_maps": False,
        "native_policy_hmem_cache_full_generation": False,
        "native_action_application": False,
        "native_position_occupancy_energy_patch_updates": False,
        "resident_cuda_generation": False,
        "checkpoint_continuation": False,
        "controlled_membership": False,
        "cxxbackbone3_gru_aten_enabled": _as_bool(getattr(cfg, "cxxbackbone3_policy_aten_gru_authority_enable", False)),
        "cxxbackbone4_mlp_aten_enabled": _as_bool(getattr(cfg, "cxxbackbone4_policy_aten_mlp_authority_enable", False)),
        "cxxbackbone7_service_enabled": _as_bool(
            getattr(cfg, "cxxbackbone7_resident_patch_policy_action_loop_enable", False)
        ),
    }


def support_blockers(runner: Any, counters: Mapping[str, int] | None = None) -> tuple[str, ...]:
    cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    blockers: list[str] = []
    if not _as_bool(getattr(cfg, "policy_deterministic_actions", False)):
        blockers.append("stochastic_policy_rng_not_native_for_v8")
    if _as_int(getattr(cfg, "policy_tick_stride", 0), 0) not in (1, 2):
        blockers.append("policy_tick_stride_must_be_1_or_2")
    if _as_bool(getattr(env_cfg, "wdl_delayed_hazard_enable", False)):
        blockers.append("wdl_not_in_cxxbackbone8_mandatory_lane")
    if _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", False)):
        blockers.append("chemistry_observation_coupling_not_native_for_v8")
    blockers.extend(
        [
            "native_grouping_and_row_maps_missing",
            "native_action_application_missing",
            "resident_cuda_generation_loop_missing",
            "checkpoint_continuation_unproven",
            "controlled_membership_unproven",
        ]
    )
    if counters is not None:
        for name in MANDATORY_ZERO_COUNTERS:
            value = int(counters.get(name, 0))
            if value != 0:
                blockers.append(f"mandatory_counter_nonzero:{name}={value}")
    return tuple(dict.fromkeys(blockers))


def capture_resident_runtime_state_v8(
    runner: Any,
    state: RunnerCore1TickState,
    *,
    generation: int,
    tick: int = 0,
    buckets: Mapping[str, Any] | None = None,
) -> ResidentRuntimeStateV8:
    ids = alive_agent_ids(state, runner).astype(np.int32, copy=False)
    slots = np.asarray([state.slot_for_agent_id(int(aid)) for aid in ids.tolist()], dtype=np.int32)
    use_gru = np.asarray(state.use_gru[slots], dtype=np.bool_) if int(slots.shape[0]) else np.zeros((0,), dtype=np.bool_)
    env = getattr(runner, "env", None)
    counters = _mandatory_counters_from_buckets(buckets)
    capability = _capability_mask(runner)
    patch_arrays = _patch_arrays(runner)
    digest_payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "generation": int(generation),
        "tick": int(tick),
        "alive_agent_ids": _array_digest(ids),
        "alive_slots": _array_digest(slots),
        "slot_tokens": _array_digest(state.slot_tokens),
        "cache_agent_id": _array_digest(state.cache_agent_id),
        "cache_token": _array_digest(state.cache_token),
        "cache_cursor": _array_digest(state.cache_cursor),
        "cache_actions": _array_digest(state.cache_actions),
        "pos_y_by_id": _array_digest(np.asarray(getattr(env, "agent_pos_y", np.zeros((0,), dtype=np.int32)))),
        "pos_x_by_id": _array_digest(np.asarray(getattr(env, "agent_pos_x", np.zeros((0,), dtype=np.int32)))),
        "occupancy": _array_digest(np.asarray(getattr(env, "agent_grid", np.zeros((0, 0), dtype=np.int32)))),
        "patch_arrays": {name: _array_digest(value) for name, value in sorted(patch_arrays.items())},
        "capability_mask": capability,
        "mandatory_counters": counters,
    }
    blockers = support_blockers(runner, counters)
    return ResidentRuntimeStateV8(
        schema_version=STATE_SCHEMA_VERSION,
        generation=int(generation),
        tick=int(tick),
        cell_id=str(getattr(getattr(runner, "sim_cfg", None), "campaign_cell_id", "") or f"generation_{int(generation)}"),
        device=str(getattr(getattr(runner, "sim_cfg", None), "device", "cpu") or "cpu"),
        width=_as_int(getattr(getattr(runner, "env_cfg", None), "width", 0), 0),
        height=_as_int(getattr(getattr(runner, "env_cfg", None), "height", 0), 0),
        steps_per_generation=_as_int(getattr(getattr(runner, "sim_cfg", None), "steps_per_generation", 0), 0),
        support_envelope="flat_or_patch_deterministic_no_churn_mixed_mlp_gru",
        alive_agent_ids=ids.copy(),
        alive_slots=slots.copy(),
        slot_tokens=np.asarray(state.slot_tokens, dtype=np.int64).copy(),
        agent_id_to_slot=np.asarray(state.agent_id_to_slot, dtype=np.int32).copy(),
        slot_to_agent_id=np.asarray(state.slot_to_agent_id, dtype=np.int64).copy(),
        use_gru=use_gru.copy(),
        mlp_rows=np.nonzero(~use_gru)[0].astype(np.int32),
        gru_rows=np.nonzero(use_gru)[0].astype(np.int32),
        cache_agent_id=np.asarray(state.cache_agent_id, dtype=np.int64).copy(),
        cache_token=np.asarray(state.cache_token, dtype=np.int64).copy(),
        cache_cursor=np.asarray(state.cache_cursor, dtype=np.int32).copy(),
        cache_actions=np.asarray(state.cache_actions, dtype=np.int32).copy(),
        pos_y_by_id=np.asarray(getattr(env, "agent_pos_y", np.zeros((0,), dtype=np.int32)), dtype=np.int32).copy(),
        pos_x_by_id=np.asarray(getattr(env, "agent_pos_x", np.zeros((0,), dtype=np.int32)), dtype=np.int32).copy(),
        occupancy=np.asarray(getattr(env, "agent_grid", np.zeros((0, 0), dtype=np.int32)), dtype=np.int32).copy(),
        energy_by_slot=np.asarray(state.energy, dtype=np.float32).copy(),
        health_by_slot=np.asarray(state.health, dtype=np.float32).copy(),
        injury_by_slot=np.asarray(state.injury, dtype=np.float32).copy(),
        hmem_shape=_hmem_shape(runner),
        patch_arrays=patch_arrays,
        capability_mask=capability,
        mandatory_counters=counters,
        blockers=blockers,
        state_digest=_digest_json(digest_payload),
    )


def resident_runtime_state_v8_payload(snapshot: ResidentRuntimeStateV8) -> dict[str, Any]:
    payload = asdict(snapshot)
    for key, value in list(payload.items()):
        if isinstance(value, np.ndarray):
            payload[key] = _array_digest(value)
        elif isinstance(value, Mapping) and key == "patch_arrays":
            payload[key] = {str(name): _array_digest(np.asarray(arr)) for name, arr in sorted(value.items())}
    return payload


def implementation_attempts_payload() -> list[dict[str, Any]]:
    return [
        {
            "attempt": "A",
            "name": "extend_cxxbackbone7_service_to_generation_boundary",
            "result": "rejected_before_claim",
            "reason": "CXXBACKBONE7 still delegates policy grouping/row maps to _compute_actions_k and applies environment state through Python-owned RunnerCore helpers.",
            "preserved_component": "exact flat/Patch observation construction and production service boundary",
        },
        {
            "attempt": "B",
            "name": "reuse_existing_cxxrewrite_resident_generation_fast_loop",
            "result": "rejected_before_claim",
            "reason": "the legacy resident-generation loop iterates Python Agent lists, invokes Python batched action paths, and does not provide V8 checkpoint authority.",
            "preserved_component": "array-resident scalar state inventory and patch accumulator list",
        },
    ]


def execute_generation(
    runner: Any,
    state: RunnerCore1TickState,
    *,
    generation_index: int,
    steps_per_generation: int,
    buckets: dict[str, float] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    before = capture_resident_runtime_state_v8(runner, state, generation=int(generation_index), tick=0, buckets=buckets)
    counters = dict(before.mandatory_counters)
    blockers = support_blockers(runner, counters)
    executed = not blockers and all(int(counters.get(name, 0)) == 0 for name in MANDATORY_ZERO_COUNTERS)
    decision = DECISION_READY if executed else DECISION_INVALID
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "decision": decision,
        "generation": int(generation_index),
        "steps_per_generation": int(steps_per_generation),
        "executed": False,
        "hidden_fallback": False,
        "native_generation_entry_calls": 0,
        "python_callback_count": 0,
        "mandatory_counters": counters,
        "mandatory_counter_targets_met": bool(executed),
        "blockers": list(blockers),
        "implementation_attempts": implementation_attempts_payload(),
        "resident_state_before": resident_runtime_state_v8_payload(before),
        "resident_state_after_digest": before.state_digest,
        "science_state_mutated": False,
        "elapsed_s": float(time.perf_counter() - started),
    }
    if buckets is not None:
        buckets["cxxbackbone8_generation_island_gate_s"] = float(
            buckets.get("cxxbackbone8_generation_island_gate_s", 0.0) + float(result["elapsed_s"])
        )
        buckets["cxxbackbone8_hidden_fallback_count"] = float(
            buckets.get("cxxbackbone8_hidden_fallback_count", 0.0)
        )
    return result


def summary_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": result.get("decision", DECISION_INVALID),
        "executed": bool(result.get("executed", False)),
        "hidden_fallback": bool(result.get("hidden_fallback", False)),
        "science_state_mutated": bool(result.get("science_state_mutated", True)),
        "mandatory_counters": dict(result.get("mandatory_counters", {}) or {}),
        "mandatory_counter_targets_met": bool(result.get("mandatory_counter_targets_met", False)),
        "blockers": list(result.get("blockers", []) or []),
        "implementation_attempts": list(result.get("implementation_attempts", []) or []),
        "resident_state_before": dict(result.get("resident_state_before", {}) or {}),
        "elapsed_s": float(result.get("elapsed_s", 0.0) or 0.0),
    }


def write_summary(runner: Any, payload: Mapping[str, Any]) -> Path | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not _as_bool(getattr(cfg, "cxxbackbone8_resident_flat_patch_generation_island_write_summary", True)):
        return None
    run_dir = str(getattr(runner, "run_dir", "") or "")
    if not run_dir:
        return None
    filename = str(
        getattr(
            cfg,
            "cxxbackbone8_resident_flat_patch_generation_island_summary_filename",
            "cxxbackbone8_generation_island_summary.json",
        )
        or "cxxbackbone8_generation_island_summary.json"
    )
    out = Path(run_dir) / "metrics" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out
