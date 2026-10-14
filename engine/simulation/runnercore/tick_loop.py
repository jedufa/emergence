from __future__ import annotations

"""RUNNERCORE-1 array-owned tick loop."""

from collections import Counter, deque
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from agents.chemistry import (
    CHEM_SLOT_ACUTE_DAMAGE,
    CHEM_SLOT_CHRONIC_DAMAGE,
    CHEM_SLOT_ENERGY_STORE,
    CHEM_SLOT_REGULATOR_1,
    CHEM_SLOT_REGULATOR_2,
    CHEM_SLOT_REPAIR_RESERVE,
    clamp01 as chem_clamp01,
    chemistry_from_values,
)
from agents.chemistry_coupling import reproduction_viability_scale_from_chem_row
from agents.maintenance import (
    MAINT_SLOT_DUE_TICK,
    MAINT_SLOT_LAST_SCHEDULE_TICK,
    MAINT_SLOT_PENDING_STRENGTH,
    MAINT_SLOT_RECENT_LOAD,
    clamp01 as maint_clamp01,
)
from agents.cognition import prediction_ablation_flags
from agents.developmental_program_v1 import default_program, module_structure_signature
from env.gridworld_pkg.constants import FOOD
from simulation.runner_pkg.runner_ticks_common import ACT_ATTACK, ACT_GATHER, ACT_REST
from simulation.runnercore.action_cache import runnercore1_actions_strided
from simulation.runnercore.tick_state import (
    RunnerCore1TickState,
    alive_agent_ids,
    export_to_agents,
    import_generation_state,
    invalidate_action_cache_slot,
    record_churn_event,
    runnercore3_profile_add,
    runnercore3_profile_count,
    sync_agent_row_from_mirror,
    sync_newborn_row_from_contract,
    state_summary_payload,
)


SUMMARY_SCHEMA_VERSION = "runnercore1_array_tick_authority_summary_v1"
DECISION_BLOCKED = "RUNNERCORE1-C/BLOCKED-BY-OWNERSHIP"
DECISION_BLOCKED_V2 = "RUNNERCORE2-C/BLOCKED-BY-OWNERSHIP"
DECISION_INVALID = "RUNNERCORE1-E/INVALID-PROFILE-CONTRACT"
SUPPORTED_PROFILE_V1 = "supported_fast_runner_v1"
SUPPORTED_PROFILE_V2 = "supported_fast_runner_v2"
SUPPORTED_PROFILE = SUPPORTED_PROFILE_V1
RUNNERCORE3_PROFILE_SCHEMA_VERSION = "runnercore3_profile_guided_native_churn_profile_v1"
RUNNERCORE4_STRUCTURAL_PAYLOAD_SCHEMA_VERSION = "runnercore4_structural_payload_fastpath_v1"
RUNNERCORE5_CHILD_CONSTRUCTION_SCHEMA_VERSION = "runnercore5_child_construction_ownership_v1"
RUNNERCORE6_RESIDENT_CHILD_ROW_SCHEMA_VERSION = "runnercore6_resident_child_row_v1"
RUNNERCORE9_C1_PAYLOAD_OR_LAZY_NEWBORN_SCHEMA_VERSION = "runnercore9_c1_payload_or_lazy_newborn_v1"
RUNNERCORE10_TYPED_CHILD_SCHEMA_VERSION = "runnercore10_typed_child_genome_policy_row_v1"
RUNNERCORE11_C1_TYPED_GENOME_ROW_SCHEMA_VERSION = "runnercore11_c1_typed_genome_row_v1"
RUNNERCORE34_DEFAULT_DEV_PROGRAM = default_program()
RUNNERCORE34_EMPTY_MODULE_SIGNATURE = module_structure_signature([])
RUNNERCORE34_EMPTY_LOCK_STATE_SHA256 = hashlib.sha256(
    json.dumps({}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
).hexdigest()
RUNNERCORE3_PROFILE_BUCKETS = (
    "alive_order_and_mask_build_ms",
    "observation_buffer_update_ms",
    "policy_input_frame_ms",
    "policy_forward_ms",
    "action_cache_serve_ms",
    "action_cache_refill_ms",
    "env_scalar_apply_ms",
    "birth_precheck_ms",
    "event_boundary_export_ms",
    "child_genome_policy_create_ms",
    "make_child_alloc_slot_ms",
    "make_child_genome_ms",
    "make_child_policy_ms",
    "make_child_row_spec_setup_ms",
    "runnercore4_child_payload_build_ms",
    "runnercore4_c2_self_payload_ms",
    "runnercore4_c1_operator_pipeline_ms",
    "runnercore4_genome_from_c1_payload_ms",
    "runnercore5_self_policy_flat_rng_ms",
    "runnercore7_mlp_policy_init_rng_replay_ms",
    "runnercore7_self_policy_source_alias_ms",
    "runnercore7_gru_policy_init_rng_replay_ms",
    "runnercore5_direct_mlp_bank_attach_ms",
    "runnercore7_direct_gru_bank_attach_ms",
    "runnercore6_resident_newborn_attach_ms",
    "runnercore24_jungle_predator_native_ms",
    "event_boundary_import_ms",
    "slot_attach_ms",
    "policy_row_attach_ms",
    "hmem_gru_row_init_ms",
    "action_cache_invalidate_ms",
    "metrics_fingerprint_ms",
    "total_candidate_tick_ms",
)


class RunnerCore1UnsupportedError(RuntimeError):
    """Raised when RUNNERCORE-1 cannot safely own the requested live tick."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _runnercore2_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore2_churn_slot_lifecycle_enable", False))
        or _as_bool(getattr(cfg, "runnercore2_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore2_authoritative_enable", False))
        or _as_bool(getattr(cfg, "runnercore2_native_enable", False))
        or _runnercore3_requested(runner)
    )


def _runnercore3_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore3_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore3_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore3_native_enable", False))
        or _as_bool(getattr(cfg, "runnercore3_vectorized_birth_precheck_enable", False))
        or _as_bool(getattr(cfg, "runnercore3_fast_birth_event_boundary_enable", False))
        or _as_bool(getattr(cfg, "runnercore3_self_recombination_fastpath_enable", False))
        or _as_bool(getattr(cfg, "runnercore3_vectorized_action_cache_enable", False))
        or _runnercore4_requested(runner)
    )


def _runnercore9_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore9_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore9_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore9_native_enable", False))
        or _as_bool(getattr(cfg, "runnercore9_c1_trace_enable", False))
        or _as_bool(getattr(cfg, "runnercore9_c1_scalar_op_normalize_skip_enable", False))
        or _as_bool(getattr(cfg, "runnercore9_lazy_newborn_materialization_enable", False))
    )


def _runnercore10_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore10_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore10_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore10_native_enable", False))
        or _as_bool(getattr(cfg, "runnercore10_typed_child_contract_enable", False))
        or _as_bool(getattr(cfg, "runnercore10_defer_direct_attached_compat_enable", False))
    )


def _runnercore11_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore11_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore11_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore11_native_enable", False))
        or _as_bool(getattr(cfg, "runnercore11_c1_typed_rows_enable", False))
    )


def _runnercore6_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore6_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore6_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore6_native_enable", False))
        or _as_bool(getattr(cfg, "runnercore6_resident_newborn_row_attach_enable", False))
        or _as_bool(getattr(cfg, "runnercore6_trusted_canonical_genome_materialization_enable", False))
        or _as_bool(getattr(cfg, "runnercore7_self_mlp_policy_source_alias_enable", False))
        or _as_bool(getattr(cfg, "runnercore7_self_gru_policy_source_alias_enable", False))
        or _as_bool(getattr(cfg, "runnercore7_direct_newborn_gru_bank_attach_enable", False))
        or _as_bool(getattr(cfg, "runnercore7_static_reproduction_threshold_fastpath_enable", False))
        or _runnercore9_requested(runner)
        or _runnercore10_requested(runner)
        or _runnercore11_requested(runner)
    )


def _runnercore5_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore5_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore5_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore5_native_enable", False))
        or _as_bool(getattr(cfg, "runnercore5_flat_self_policy_rng_enable", False))
        or _as_bool(getattr(cfg, "runnercore5_direct_newborn_mlp_bank_attach_enable", False))
        or _as_bool(getattr(cfg, "runnercore5_c1_canonical_boundary_fastpath_enable", False))
        or _runnercore6_requested(runner)
    )


def _runnercore4_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "runnercore4_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore4_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore4_native_enable", False))
        or _as_bool(getattr(cfg, "runnercore4_structural_payload_cache_enable", False))
        or _as_bool(getattr(cfg, "runnercore4_changed_parent_import_enable", False))
        or _runnercore5_requested(runner)
    )


def _active_profile(runner: Any) -> str:
    cfg = getattr(runner, "sim_cfg", None)
    if _runnercore2_requested(runner):
        return str(getattr(cfg, "runnercore2_supported_profile", SUPPORTED_PROFILE_V2) or "")
    return str(getattr(cfg, "runnercore1_supported_profile", SUPPORTED_PROFILE_V1) or "")


def _active_blocked_decision(runner: Any) -> str:
    return DECISION_BLOCKED_V2 if _runnercore2_requested(runner) else DECISION_BLOCKED


def _summary_attr(runner: Any) -> str:
    if _runnercore11_requested(runner):
        return "_runnercore11_c1_typed_genome_row_summary"
    if _runnercore10_requested(runner):
        return "_runnercore10_typed_child_genome_policy_row_summary"
    if _runnercore9_requested(runner):
        return "_runnercore9_c1_payload_or_lazy_newborn_summary"
    if _runnercore6_requested(runner):
        return "_runnercore6_resident_child_row_summary"
    if _runnercore5_requested(runner):
        return "_runnercore5_child_construction_ownership_summary"
    if _runnercore4_requested(runner):
        return "_runnercore4_structural_payload_fastpath_summary"
    if _runnercore3_requested(runner):
        return "_runnercore3_profile_guided_native_churn_summary"
    return "_runnercore2_churn_slot_lifecycle_summary" if _runnercore2_requested(runner) else "_runnercore1_array_tick_authority_summary"


def _summary(runner: Any) -> dict[str, Any]:
    attr = _summary_attr(runner)
    cur = getattr(runner, attr, None)
    if isinstance(cur, dict):
        return cur
    decision = _active_blocked_decision(runner)
    cur = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "requested": True,
        "profile": _active_profile(runner),
        "status": "initialized",
        "decision": decision,
        "support_blockers": [],
        "candidate_ran": False,
        "ticks_ran": 0,
        "fallback_counters": {},
        "state_summary": None,
        "boundary_export": None,
        "timing_s": {},
        "first_divergence": None,
    }
    setattr(runner, attr, cur)
    if attr != "_runnercore6_resident_child_row_summary" and _runnercore6_requested(runner):
        setattr(runner, "_runnercore6_resident_child_row_summary", cur)
    if attr != "_runnercore2_churn_slot_lifecycle_summary" and _runnercore2_requested(runner):
        setattr(runner, "_runnercore2_churn_slot_lifecycle_summary", cur)
    if attr != "_runnercore1_array_tick_authority_summary":
        setattr(runner, "_runnercore1_array_tick_authority_summary", cur)
    return cur


def write_summary_if_requested(runner: Any, summary: dict[str, Any]) -> Path | None:
    cfg = getattr(runner, "sim_cfg", None)
    if _runnercore11_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore11_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore11_summary_filename", "runnercore11_c1_typed_genome_row_summary.json") or "")
    elif _runnercore10_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore10_write_summary", True)):
            return None
        name = str(
            getattr(cfg, "runnercore10_summary_filename", "runnercore10_typed_child_genome_policy_row_summary.json")
            or ""
        )
    elif _runnercore9_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore9_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore9_summary_filename", "runnercore9_c1_payload_or_lazy_newborn_summary.json") or "")
    elif _runnercore6_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore6_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore6_summary_filename", "runnercore6_resident_child_row_summary.json") or "")
    elif _runnercore5_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore5_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore5_summary_filename", "runnercore5_child_construction_ownership_summary.json") or "")
    elif _runnercore4_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore4_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore4_summary_filename", "runnercore4_structural_payload_fastpath_summary.json") or "")
    elif _runnercore3_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore3_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore3_summary_filename", "runnercore3_profile_guided_native_churn_summary.json") or "")
    elif _runnercore2_requested(runner):
        if not _as_bool(getattr(cfg, "runnercore2_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore2_summary_filename", "runnercore2_churn_slot_lifecycle_summary.json") or "")
    else:
        if not _as_bool(getattr(cfg, "runnercore1_write_summary", True)):
            return None
        name = str(getattr(cfg, "runnercore1_summary_filename", "runnercore1_array_tick_authority_summary.json") or "")
    if not name:
        return None
    run_dir = str(getattr(runner, "run_dir", "") or "")
    if not run_dir:
        return None
    out = Path(run_dir) / "metrics" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out


def _support_blockers(runner: Any) -> list[str]:
    cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    blockers: list[str] = []
    is_v2 = _runnercore2_requested(runner)
    profile = _active_profile(runner)
    if is_v2:
        churn_enable = bool(
            _as_bool(getattr(cfg, "runnercore2_churn_slot_lifecycle_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_fast_birth_event_boundary_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_self_recombination_fastpath_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_vectorized_action_cache_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_structural_payload_cache_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_changed_parent_import_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_flat_self_policy_rng_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_direct_newborn_mlp_bank_attach_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_c1_canonical_boundary_fastpath_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_resident_newborn_row_attach_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_trusted_canonical_genome_materialization_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_c1_trace_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_c1_scalar_op_normalize_skip_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_lazy_newborn_materialization_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_typed_child_contract_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_defer_direct_attached_compat_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_c1_typed_rows_enable", False))
        )
        candidate_enable = bool(
            _as_bool(getattr(cfg, "runnercore2_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore2_authoritative_enable", False))
            or _as_bool(getattr(cfg, "runnercore2_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_fast_birth_event_boundary_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_self_recombination_fastpath_enable", False))
            or _as_bool(getattr(cfg, "runnercore3_vectorized_action_cache_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_structural_payload_cache_enable", False))
            or _as_bool(getattr(cfg, "runnercore4_changed_parent_import_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_flat_self_policy_rng_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_direct_newborn_mlp_bank_attach_enable", False))
            or _as_bool(getattr(cfg, "runnercore5_c1_canonical_boundary_fastpath_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_resident_newborn_row_attach_enable", False))
            or _as_bool(getattr(cfg, "runnercore6_trusted_canonical_genome_materialization_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_c1_trace_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_c1_scalar_op_normalize_skip_enable", False))
            or _as_bool(getattr(cfg, "runnercore9_lazy_newborn_materialization_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_typed_child_contract_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_defer_direct_attached_compat_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_profile_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_candidate_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_native_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_c1_typed_rows_enable", False))
        )
        if profile != SUPPORTED_PROFILE_V2:
            blockers.append("runnercore2_supported_profile_must_be_supported_fast_runner_v2")
        if not (churn_enable and candidate_enable):
            blockers.append("runnercore2_candidate_or_authoritative_flag_required")
        if not (
            _as_bool(getattr(cfg, "runnercore2_allow_birth_boundary_agent_materialization", False))
            or _as_bool(getattr(cfg, "runnercore3_allow_birth_boundary_agent_materialization", False))
            or _as_bool(getattr(cfg, "runnercore4_allow_birth_boundary_agent_materialization", False))
            or _as_bool(getattr(cfg, "runnercore5_allow_birth_boundary_agent_materialization", False))
            or _as_bool(getattr(cfg, "runnercore6_allow_birth_boundary_agent_materialization", False))
            or _as_bool(getattr(cfg, "runnercore9_lazy_newborn_materialization_enable", False))
            or _as_bool(getattr(cfg, "runnercore10_defer_direct_attached_compat_enable", False))
            or _as_bool(getattr(cfg, "runnercore11_c1_typed_rows_enable", False))
        ):
            blockers.append("runnercore2_birth_boundary_agent_materialization_must_be_explicit")
    else:
        if profile != SUPPORTED_PROFILE_V1:
            blockers.append("runnercore1_supported_profile_must_be_supported_fast_runner_v1")
        if not (
            _as_bool(getattr(cfg, "runnercore1_array_tick_authority_enable", False))
            and (
                _as_bool(getattr(cfg, "runnercore1_candidate_enable", False))
                or _as_bool(getattr(cfg, "runnercore1_authoritative_enable", False))
                or _as_bool(getattr(cfg, "runnercore1_native_enable", False))
            )
        ):
            blockers.append("runnercore1_candidate_or_authoritative_flag_required")
    if _as_int(getattr(cfg, "policy_tick_stride", 0), 0) not in (1, 2):
        blockers.append("policy_tick_stride_must_be_1_or_2")
    if not _as_bool(getattr(cfg, "policy_deterministic_actions", False)):
        blockers.append("policy_deterministic_actions_required")
    if _as_bool(getattr(cfg, "perf_policy_stride_slot_cache_enable", False)):
        blockers.append("legacy_slot_stride_cache_must_be_disabled")
    runnercore20_policy_obs = _as_bool(getattr(cfg, "runnercore20_policy_observation_arrays_enable", False))
    if _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", True)):
        if not runnercore20_policy_obs:
            blockers.append("runnercore20_policy_observation_arrays_required_for_chemistry")
        if _as_bool(getattr(cfg, "chemistry_action_cost_coupling_enable", False)):
            blockers.append("chemistry_action_cost_coupling_not_array_authoritative")
    if _as_int(getattr(cfg, "checkpoint_bundle_v2_period_gen", 0), 0) != 0:
        blockers.append("checkpoint_bundle_v2_period_gen_must_be_0")
    if _as_bool(getattr(cfg, "maintenance_enable", False)) and not _as_bool(
        getattr(cfg, "runnercore19_maintenance_arrays_enable", False)
    ):
        blockers.append("runnercore19_maintenance_arrays_required")
    runnercore22_birth_bridge = _as_bool(getattr(cfg, "runnercore22_full_birth_boundary_bridge_enable", False))
    if (not is_v2) and _as_float(getattr(env_cfg, "reproduce_try_prob", 0.0), 0.0) > 0.0:
        blockers.append("birth_refill_churn_not_array_authoritative")
    if is_v2 and _as_bool(getattr(env_cfg, "reproduce_requires_mate", False)) and not runnercore22_birth_bridge:
        blockers.append("runnercore2_mate_reproduction_not_array_authoritative")
    if is_v2 and _as_bool(getattr(cfg, "lineage_reproduction_funnel_reporting_enable", False)):
        blockers.append("runnercore2_lineage_reproduction_funnel_reporting_not_array_authoritative")
    if (
        is_v2
        and _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", True))
        and _as_bool(getattr(cfg, "chemistry_repro_viability_enable", False))
        and not runnercore22_birth_bridge
    ):
        blockers.append("runnercore2_chemistry_repro_viability_not_array_authoritative")
    if is_v2 and not runnercore22_birth_bridge and (
        _as_bool(getattr(cfg, "runnercore3_fast_birth_event_boundary_enable", False))
        or _as_bool(getattr(cfg, "runnercore4_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore5_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore6_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore9_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore10_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore11_candidate_enable", False))
    ):
        if _as_bool(getattr(env_cfg, "reproduce_requires_mate", False)):
            blockers.append("runnercore3_fast_birth_boundary_requires_no_mate")
        if _as_bool(getattr(cfg, "lineage_reproduction_funnel_reporting_enable", False)):
            blockers.append("runnercore3_fast_birth_boundary_lineage_funnel_not_supported")
        if _as_bool(getattr(getattr(runner, "env", None), "patch_world_enable", False)):
            blockers.append("runnercore3_fast_birth_boundary_patch_world_not_supported")
    jungle_owned = _as_bool(getattr(cfg, "runnercore23_jungle_predator_bridge_enable", False)) or _as_bool(
        getattr(cfg, "runnercore24_native_jungle_predator_arrays_enable", False)
    )
    if _as_bool(getattr(env_cfg, "jungle_predator_pressure_enable", False)) and not jungle_owned:
        blockers.append("jungle_predator_pressure_not_array_authoritative")
    if _as_bool(getattr(env_cfg, "wdl_delayed_hazard_enable", False)):
        blockers.append("wdl_delayed_hazard_damage_not_array_authoritative")
    perception_flags = (
        "enable_evolvable_sensor_channel_mask",
        "ablate_sensor_channel_mask",
        "force_all_sensor_channels_enabled",
        "force_minimal_sensor_channels_enabled",
        "enable_evolvable_sensor_radius",
        "ablate_sensor_radius",
        "force_min_sensor_radius",
        "force_max_sensor_radius",
        "force_default_sensor_radius",
    )
    if not runnercore20_policy_obs:
        for name in perception_flags:
            if _as_bool(getattr(cfg, name, False)):
                blockers.append(f"{name}_not_array_authoritative")

    cost_flags = (
        "enable_experimental_perception_cost_charging",
        "enable_experimental_sensor_radius_cost_charging",
        "enable_experimental_prediction_cost_charging",
        "enable_experimental_developmental_cost_charging",
        "enable_experimental_introspection_cost_charging",
        "enable_experimental_self_regulation_cost_charging",
        "enable_experimental_self_modification_cost_charging",
    )
    cost_requested = any(_as_bool(getattr(cfg, name, False)) for name in cost_flags)
    if cost_requested:
        if not _as_bool(getattr(cfg, "runnercore21_cost_charge_surface_arrays_enable", False)):
            for name in cost_flags:
                if _as_bool(getattr(cfg, name, False)):
                    blockers.append(f"{name}_not_array_authoritative")
        elif not _as_bool(getattr(cfg, "env_hotpath_cost_charge_surface_enable", False)):
            blockers.append("runnercore21_cost_charge_surface_arrays_require_cost_surface")

    unsupported_flags = (
        "identity_drift_enable",
        "adult_move_penalty_enable",
        "late_coherence_eff_enable",
        "late_pred_energy_tax_enable",
        "dev_plasticity_decay_enable",
        "enable_developmental_cognition",
        "force_zero_maturation_cost",
        "force_max_maturation_cost",
    )
    for name in unsupported_flags:
        if _as_bool(getattr(cfg, name, False)):
            blockers.append(f"{name}_not_array_authoritative")
    if _as_bool(getattr(env_cfg, "migration_pressure_enable", False)):
        blockers.append("migration_pressure_not_array_authoritative")
    if _as_bool(getattr(env_cfg, "tier4_juvenile_vulnerability_enable", False)):
        blockers.append("tier4_juvenile_vulnerability_not_array_authoritative")
    return blockers


def classify_support(runner: Any) -> dict[str, Any]:
    blockers = _support_blockers(runner)
    is_v2 = _runnercore2_requested(runner)
    profile = _active_profile(runner)
    decision_supported = "RUNNERCORE2-SUPPORTED" if is_v2 else "RUNNERCORE1-SUPPORTED"
    return {
        "schema_version": "runnercore2_support_report_v1" if is_v2 else "runnercore1_support_report_v1",
        "profile": profile,
        "supported": not blockers,
        "decision": decision_supported if not blockers else _active_blocked_decision(runner),
        "blockers": blockers,
        "config": {
            "policy_tick_stride": _as_int(getattr(runner.sim_cfg, "policy_tick_stride", 0), 0),
            "policy_deterministic_actions": _as_bool(getattr(runner.sim_cfg, "policy_deterministic_actions", False)),
            "chemistry_functional_coupling_enable": _as_bool(
                getattr(runner.sim_cfg, "chemistry_functional_coupling_enable", True)
            ),
            "checkpoint_bundle_v2_period_gen": _as_int(
                getattr(runner.sim_cfg, "checkpoint_bundle_v2_period_gen", 0),
                0,
            ),
            "behavior_log_enable": _as_bool(getattr(runner.sim_cfg, "behavior_log_enable", True)),
            "maintenance_enable": _as_bool(getattr(runner.sim_cfg, "maintenance_enable", False)),
            "reproduce_try_prob": _as_float(getattr(runner.env_cfg, "reproduce_try_prob", 0.0), 0.0),
            "reproduce_requires_mate": _as_bool(getattr(runner.env_cfg, "reproduce_requires_mate", False)),
        },
    }


def _runnercore3_profile_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return (
        _as_bool(getattr(cfg, "runnercore3_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore4_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore5_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore6_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore9_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore10_profile_enable", False))
        or _as_bool(getattr(cfg, "runnercore11_profile_enable", False))
    )


def _cxx26b_tick_in_filter(tick: int) -> bool:
    ticks_s = (os.environ.get("EIEL_STATE_DUMP_AGENT_DETAIL_TICKS", "") or "").strip()
    if not ticks_s:
        try:
            return int(tick) <= int(os.environ.get("EIEL_STATE_DUMP_AGENT_DETAIL_TMAX", "30"))
        except Exception:
            return int(tick) <= 30
    try:
        for part in ticks_s.split(","):
            item = part.strip()
            if not item:
                continue
            if "-" in item:
                left_s, right_s = item.split("-", 1)
                left = int(left_s.strip())
                right = int(right_s.strip())
                lo, hi = (left, right) if left <= right else (right, left)
                if lo <= int(tick) <= hi:
                    return True
            elif int(item) == int(tick):
                return True
    except Exception:
        return False
    return False


def _cxx26b_stage_requested(stage: str) -> bool:
    stages_s = (os.environ.get("EIEL_STATE_DUMP_AGENT_DETAIL_STAGES", "") or "").strip()
    if not stages_s:
        return str(stage) == "post_birth"
    return str(stage) in {item.strip() for item in stages_s.split(",") if item.strip()}


def _cxx26b_maybe_dump_tick_detail(runner: Any, state: RunnerCore1TickState, *, gen: int, tick: int, stage: str) -> None:
    if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
        return
    if not _cxx26b_tick_in_filter(int(tick)) or not _cxx26b_stage_requested(str(stage)):
        return
    try:
        from engine.simulation.state_dump import maybe_dump_state  # noqa: WPS433

        rows = []
        agents_by_id = getattr(getattr(runner, "pop", None), "agents", {}) or {}
        for slot, aid in sorted(
            (
                (int(slot), int(aid))
                for slot, aid in enumerate(np.asarray(state.slot_to_agent_id, dtype=np.int64).tolist())
                if bool(state.alive[int(slot)]) and int(aid) >= 0
            ),
            key=lambda item: item[1],
        ):
            agent = agents_by_id.get(int(aid)) if isinstance(agents_by_id, dict) else None
            stats = getattr(agent, "stats", None)
            rows.append(
                (
                    int(aid),
                    int(round(float(state.energy[int(slot)]) * 1000.0)),
                    int(round(float(state.health[int(slot)]) * 1000.0)),
                    int(round(float(state.injury[int(slot)]) * 1000.0)),
                    int(np.asarray(runner.env.agent_pos_x, dtype=np.int32)[int(aid)]),
                    int(np.asarray(runner.env.agent_pos_y, dtype=np.int32)[int(aid)]),
                    int(np.asarray(state.last_action, dtype=np.int32)[int(slot)])
                    if hasattr(state, "last_action")
                    else -1,
                    int(round(float(state.drives[int(slot), 0]) * 1000000.0)),
                    int(round(float(state.drives[int(slot), 1]) * 1000000.0)),
                    int(round(float(state.drives[int(slot), 2]) * 1000000.0)),
                    int(round(float(state.drives[int(slot), 3]) * 1000000.0)),
                    int(round(float(state.aggr[int(slot)]) * 1000000.0)),
                    int(round(float(state.temp[int(slot)]) * 1000000.0)),
                    int(round(float(state.gain[int(slot)]) * 1000000.0)),
                    int(slot),
                    int(np.asarray(state.repro_cooldown, dtype=np.int32)[int(slot)])
                    if hasattr(state, "repro_cooldown")
                    else 0,
                    int(np.asarray(state.survival_steps, dtype=np.int32)[int(slot)])
                    if hasattr(state, "survival_steps")
                    else 0,
                    int(np.asarray(state.early_survival_steps, dtype=np.int32)[int(slot)])
                    if hasattr(state, "early_survival_steps")
                    else 0,
                    int(getattr(stats, "birth_generation", 0)) if stats is not None else 0,
                    int(getattr(stats, "birth_tick", 0)) if stats is not None else 0,
                )
            )
        digest = hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), sort_keys=False).encode("utf-8")
        ).hexdigest()
        maybe_dump_state(
            runner,
            generation_index=int(gen),
            tag=f"tick_agent_state_detail_{stage}_t{int(tick):03d}",
            extra={
                "tick": int(tick),
                "stage": str(stage),
                "agent_state_detail_sha256": digest,
                "agent_state_detail": rows,
            },
        )
    except Exception:
        return


def _cxxwave1f_scalar_backbone_requested(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "cxxwave1f_scalar_backbone_enable", False))


def _cxxbackbone7_policy_action_loop_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "cxxbackbone7_resident_patch_policy_action_loop_enable", False)
    )


def _cxxbackbone7_fail_closed(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "cxxbackbone7_resident_patch_policy_action_loop_fail_closed", True)
    )


def _cxxbackbone8_generation_island_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(
            getattr(runner, "sim_cfg", None),
            "cxxbackbone8_resident_flat_patch_generation_island_enable",
            False,
        )
    )


def _cxxbackbone8_generation_island_fail_closed(runner: Any) -> bool:
    return _as_bool(
        getattr(
            getattr(runner, "sim_cfg", None),
            "cxxbackbone8_resident_flat_patch_generation_island_fail_closed",
            True,
        )
    )


def _cxxbackbone13_owner_generation_runner_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "cxxbackbone13_owner_generation_runner_enable", False)
    )


def _cxxbackbone13_owner_generation_runner_fail_closed(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "cxxbackbone13_owner_generation_runner_fail_closed", True)
    )


def _cxxbackbone13_write_summary(runner: Any, payload: dict[str, Any]) -> None:
    cfg = getattr(runner, "sim_cfg", None)
    if not _as_bool(getattr(cfg, "cxxbackbone13_owner_generation_runner_write_summary", True)):
        return
    run_dir = Path(str(getattr(runner, "run_dir", "") or "."))
    name = str(
        getattr(
            cfg,
            "cxxbackbone13_owner_generation_runner_summary_filename",
            "cxxbackbone13_owner_generation_runner_summary.json",
        )
        or "cxxbackbone13_owner_generation_runner_summary.json"
    )
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _cxxbackbone8r_flat_cpu_action_apply_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "cxxbackbone8r_flat_cpu_native_action_apply_enable", False)
    )


def _cxxbackbone8r_patch_cpu_action_apply_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "cxxbackbone8r_patch_cpu_native_action_apply_enable", False)
    )


def _cxxbackbone8r_flat_cpu_action_apply_supported(runner: Any) -> tuple[bool, str]:
    env_cfg = getattr(runner, "env_cfg", None)
    sim_cfg = getattr(runner, "sim_cfg", None)
    if _as_bool(getattr(env_cfg, "patch_world_enable", False)):
        return False, "patch_world_not_flat_cpu_stage"
    if _as_bool(getattr(env_cfg, "enable_biomes", False)):
        return False, "biomes_not_flat_cpu_stage"
    if _as_bool(getattr(env_cfg, "enable_injury", False)):
        return False, "injury_not_flat_cpu_stage"
    if _as_bool(getattr(env_cfg, "enable_combat", False)):
        return False, "combat_not_flat_cpu_stage"
    if str(getattr(sim_cfg, "device", "cpu") or "cpu").startswith("cuda"):
        return False, "cuda_not_flat_cpu_stage"
    return True, ""


def _cxxbackbone8r_patch_cpu_action_apply_supported(runner: Any) -> tuple[bool, str]:
    env_cfg = getattr(runner, "env_cfg", None)
    sim_cfg = getattr(runner, "sim_cfg", None)
    if not _as_bool(getattr(env_cfg, "patch_world_enable", False)):
        return False, "patch_world_required"
    if _as_bool(getattr(env_cfg, "enable_biomes", False)):
        return False, "biomes_not_patch_cpu_stage"
    if _as_bool(getattr(env_cfg, "enable_injury", False)):
        return False, "injury_not_patch_cpu_stage"
    if _as_bool(getattr(env_cfg, "enable_combat", False)):
        return False, "combat_not_patch_cpu_stage"
    if _as_bool(getattr(env_cfg, "wdl_delayed_hazard_enable", False)):
        return False, "wdl_delayed_hazard_not_patch_cpu_stage"
    if _as_bool(getattr(env_cfg, "prediction_demand_enable", False)):
        return False, "prediction_demand_not_patch_cpu_stage"
    if _as_bool(getattr(env_cfg, "delayed_resource_depletion_enable", False)):
        return False, "delayed_resource_depletion_not_patch_cpu_stage"
    if _as_bool(getattr(env_cfg, "delayed_crowding_waste_enable", False)):
        return False, "delayed_crowding_waste_not_patch_cpu_stage"
    if _as_bool(getattr(env_cfg, "mem_max_delayed_choice_enable", False)):
        return False, "mem_max_delayed_choice_not_patch_cpu_stage"
    if str(getattr(sim_cfg, "device", "cpu") or "cpu").startswith("cuda"):
        return False, "cuda_not_patch_cpu_stage"
    return True, ""


def _cxxwave1f_summary(runner: Any) -> dict[str, Any]:
    cur = getattr(runner, "_cxxwave1f_cpp_backbone_summary", None)
    if isinstance(cur, dict):
        return cur
    cur = {
        "schema_version": "cxxwave1f_cpp_backbone_summary_v1",
        "requested": bool(_cxxwave1f_scalar_backbone_requested(runner)),
        "native_ticks": 0,
        "native_rows": 0,
        "fallback_ticks": 0,
        "unsupported_ticks": 0,
        "support_blockers": {},
        "summary_totals": {},
        "last_support_report": None,
        "last_native_summary": None,
        "decision": "CXXWAVE1F-SCALAR-LIVE-UNREQUESTED",
    }
    setattr(runner, "_cxxwave1f_cpp_backbone_summary", cur)
    return cur


def _cxxwave1f_summary_add(summary: dict[str, Any], name: str, value: int | float = 1) -> None:
    totals = summary.setdefault("summary_totals", {})
    totals[str(name)] = float(totals.get(str(name), 0.0) or 0.0) + float(value)


def _write_cxxwave1f_summary_if_requested(runner: Any, summary: dict[str, Any]) -> Path | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not _as_bool(getattr(cfg, "cxxwave1f_scalar_backbone_write_summary", True)):
        return None
    run_dir = str(getattr(runner, "run_dir", "") or "")
    if not run_dir:
        return None
    name = str(
        getattr(
            cfg,
            "cxxwave1f_scalar_backbone_summary_filename",
            "cxxwave1f_cpp_backbone_summary.json",
        )
        or ""
    )
    if not name:
        return None
    out = Path(run_dir) / "metrics" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out


def _cxxwave1f_apply_scalar_backbone(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: np.ndarray,
    actions: np.ndarray,
    *,
    tick_index: int,
    buckets: dict[str, float],
) -> bool:
    if not _cxxwave1f_scalar_backbone_requested(runner):
        return False
    t0 = time.perf_counter()
    cfg_obj = getattr(runner, "sim_cfg", None)
    summary = _cxxwave1f_summary(runner)
    fail_closed = _as_bool(getattr(cfg_obj, "cxxwave1f_scalar_backbone_fail_closed", True))
    library_path = str(getattr(cfg_obj, "cxxwave1f_scalar_backbone_library_path", "") or "")
    flags = prediction_ablation_flags(cfg_obj)
    prediction_disabled = bool(flags.get("ablate", False) or flags.get("force_disabled", False))
    prediction_horizon = 0 if prediction_disabled else _as_int(getattr(cfg_obj, "pred_horizon", 2), 2)
    maintenance_opp = _as_float(getattr(cfg_obj, "maintenance_opportunity_cost_coeff", 0.0), 0.0)
    try:
        from perf.cxxwave1f_cpp_backbone import (  # noqa: WPS433
            Cxxwave1fScalarConfig,
            Cxxwave1fScalarState,
            live_support_report,
            load_scalar_backbone_kernel,
        )
    except Exception as exc:
        report = {
            "schema_version": "cxxwave1f_support_gate_v1",
            "supported": False,
            "blockers": [f"import_failed:{exc.__class__.__name__}"],
            "decision": "CXXWAVE1F-SCALAR-LIVE-FAIL-CLOSED",
        }
        summary["last_support_report"] = report
        summary["unsupported_ticks"] = int(summary.get("unsupported_ticks", 0)) + 1
        _write_cxxwave1f_summary_if_requested(runner, summary)
        if fail_closed:
            raise RunnerCore1UnsupportedError(report["decision"] + ": " + ",".join(report["blockers"])) from exc
        return False

    report = live_support_report(
        prediction_horizon=int(prediction_horizon),
        maintenance_opportunity_cost_coeff=float(maintenance_opp),
        library_path=library_path,
    )
    summary["last_support_report"] = report
    if not bool(report.get("supported", False)):
        summary["unsupported_ticks"] = int(summary.get("unsupported_ticks", 0)) + 1
        blockers = summary.setdefault("support_blockers", {})
        for reason in report.get("blockers", []):
            blockers[str(reason)] = int(blockers.get(str(reason), 0) or 0) + 1
        summary["decision"] = "CXXWAVE1F-SCALAR-LIVE-FAIL-CLOSED"
        _write_cxxwave1f_summary_if_requested(runner, summary)
        if fail_closed:
            raise RunnerCore1UnsupportedError(report["decision"] + ": " + ",".join(report.get("blockers", [])))
        summary["fallback_ticks"] = int(summary.get("fallback_ticks", 0)) + 1
        return False

    slots = np.asarray([state.slot_for_agent_id(int(aid)) for aid in agent_ids.tolist()], dtype=np.int32)
    if int(slots.shape[0]) <= 0:
        return True
    valid = slots >= 0
    if bool(np.any(valid)):
        valid[valid] = state.alive[slots[valid]]
    valid_idx = np.nonzero(valid)[0]
    chem_noise = runner.rng_manager.policy_np.normal(0.0, 1.0, size=(int(slots.shape[0]), 2)).astype(np.float32)
    drive_noise_samples = np.zeros((int(slots.shape[0]), 4), dtype=np.float32)
    if int(valid_idx.shape[0]) > 0:
        valid_slots = slots[valid_idx]
        if bool(np.any(state.drive_noise[valid_slots] > np.float32(0.0))):
            drive_noise_samples[valid_idx, :] = runner.np_rng.normal(0.0, 1.0, size=(int(valid_idx.shape[0]), 4)).astype(
                np.float32,
                copy=False,
            )

    backend = getattr(runner, "_cxxwave1f_scalar_backbone_backend", None)
    if backend is None or getattr(runner, "_cxxwave1f_scalar_backbone_backend_path", "") != library_path:
        backend = load_scalar_backbone_kernel(library_path)
        setattr(runner, "_cxxwave1f_scalar_backbone_backend", backend)
        setattr(runner, "_cxxwave1f_scalar_backbone_backend_path", library_path)

    native_state = Cxxwave1fScalarState(
        slots=np.ascontiguousarray(slots, dtype=np.int32),
        actions=np.ascontiguousarray(actions, dtype=np.int32),
        alive=np.ascontiguousarray(state.alive, dtype=np.uint8),
        energy=np.ascontiguousarray(state.energy, dtype=np.float32),
        health=np.ascontiguousarray(state.health, dtype=np.float32),
        injury=np.ascontiguousarray(state.injury, dtype=np.float32),
        last_energy_delta=np.ascontiguousarray(state.last_energy_delta, dtype=np.float32),
        last_health_delta=np.ascontiguousarray(state.last_health_delta, dtype=np.float32),
        last_injury_delta=np.ascontiguousarray(state.last_injury_delta, dtype=np.float32),
        drives=np.ascontiguousarray(state.drives, dtype=np.float32),
        prev_drive_delta=np.ascontiguousarray(state.prev_drive_delta, dtype=np.float32),
        drive_inertia=np.ascontiguousarray(state.drive_inertia, dtype=np.float32),
        drive_noise_scale=np.ascontiguousarray(state.drive_noise, dtype=np.float32),
        drive_noise_samples=np.ascontiguousarray(drive_noise_samples, dtype=np.float32),
        survival_steps=np.ascontiguousarray(state.survival_steps, dtype=np.int32),
        drive_volatility=np.ascontiguousarray(state.drive_volatility, dtype=np.float32),
        drive_jitter=np.ascontiguousarray(state.drive_jitter, dtype=np.float32),
        episode0_drive_jitter_sum=np.ascontiguousarray(state.episode0_drive_jitter_sum, dtype=np.float32),
        episode0_selfharm_events=np.ascontiguousarray(state.episode0_selfharm_events, dtype=np.int32),
        chemistry=np.ascontiguousarray(state.chemistry, dtype=np.float32),
        chemistry_noise=np.ascontiguousarray(chem_noise, dtype=np.float32),
        maintenance=np.ascontiguousarray(state.maintenance, dtype=np.float32),
        maintenance_opportunity=np.zeros((int(slots.shape[0]),), dtype=np.float32),
        maintenance_requests=np.ascontiguousarray(state.maintenance_requests, dtype=np.int32),
        maintenance_applied=np.ascontiguousarray(state.maintenance_applied, dtype=np.int32),
        maintenance_energy_spent_total=np.ascontiguousarray(state.maintenance_energy_spent_total, dtype=np.float32),
        maintenance_opportunity_cost_total=np.ascontiguousarray(state.maintenance_opportunity_cost_total, dtype=np.float32),
        maintenance_applied_effect_total=np.ascontiguousarray(state.maintenance_applied_effect_total, dtype=np.float32),
    )
    native_cfg = Cxxwave1fScalarConfig(
        max_energy=float(getattr(runner.env_cfg, "max_energy", 1.0) or 1.0),
        start_health=float(getattr(runner.env_cfg, "start_health", 1.0) or 1.0),
        hunger_gain=_as_float(getattr(cfg_obj, "drive_hunger_gain", 0.08), 0.08),
        hunger_decay=_as_float(getattr(cfg_obj, "drive_hunger_decay", 0.02), 0.02),
        pain_gain=_as_float(getattr(cfg_obj, "drive_pain_gain", 0.18), 0.18),
        pain_decay=_as_float(getattr(cfg_obj, "drive_pain_decay", 0.03), 0.03),
        fatigue_gain=_as_float(getattr(cfg_obj, "drive_fatigue_gain", 0.05), 0.05),
        fatigue_rest_relief=_as_float(getattr(cfg_obj, "drive_fatigue_rest_relief", 0.10), 0.10),
        fatigue_decay=_as_float(getattr(cfg_obj, "drive_fatigue_decay", 0.01), 0.01),
        fear_gain=_as_float(getattr(cfg_obj, "drive_fear_gain", 0.22), 0.22),
        fear_decay=_as_float(getattr(cfg_obj, "drive_fear_decay", 0.03), 0.03),
        episode0_k=_as_int(getattr(cfg_obj, "episode0_k", 20), 20),
        maintenance_enable=_as_bool(getattr(cfg_obj, "maintenance_enable", False)),
        maintenance_delay_ticks=max(0, _as_int(getattr(cfg_obj, "maintenance_delay_ticks", 0), 0)),
        maintenance_base_effect=_as_float(getattr(cfg_obj, "maintenance_base_effect", 0.0), 0.0),
        maintenance_diminishing_factor=max(0.0, _as_float(getattr(cfg_obj, "maintenance_diminishing_factor", 0.0), 0.0)),
        maintenance_recent_decay=_as_float(getattr(cfg_obj, "maintenance_recent_decay", 1.0), 1.0),
        maintenance_energy_cost=max(0.0, _as_float(getattr(cfg_obj, "maintenance_energy_cost", 0.0), 0.0)),
        maintenance_opportunity_cost_coeff=0.0,
        tick_index=int(tick_index),
    )
    result = backend(native_state, native_cfg)
    native_summary = dict(result.get("summary", {}))
    summary["last_native_summary"] = native_summary
    summary["native_ticks"] = int(summary.get("native_ticks", 0)) + 1
    summary["native_rows"] = int(summary.get("native_rows", 0)) + int(native_summary.get("drive_rows", 0))
    summary["decision"] = "CXXWAVE1F-SCALAR-LIVE-NATIVE"
    for key, value in native_summary.items():
        _cxxwave1f_summary_add(summary, key, float(value))

    # ctypes writes into the same contiguous arrays today. Assign explicitly so
    # future non-contiguous state layouts cannot silently drop native output.
    state.alive[...] = native_state.alive
    state.energy[...] = native_state.energy
    state.health[...] = native_state.health
    state.injury[...] = native_state.injury
    state.drives[...] = native_state.drives
    state.prev_drive_delta[...] = native_state.prev_drive_delta
    state.drive_volatility[...] = native_state.drive_volatility
    state.drive_jitter[...] = native_state.drive_jitter
    state.episode0_drive_jitter_sum[...] = native_state.episode0_drive_jitter_sum
    state.episode0_selfharm_events[...] = native_state.episode0_selfharm_events
    state.chemistry[...] = native_state.chemistry
    state.maintenance[...] = native_state.maintenance
    state.maintenance_requests[...] = native_state.maintenance_requests
    state.maintenance_applied[...] = native_state.maintenance_applied
    state.maintenance_energy_spent_total[...] = native_state.maintenance_energy_spent_total
    state.maintenance_opportunity_cost_total[...] = native_state.maintenance_opportunity_cost_total
    state.maintenance_applied_effect_total[...] = native_state.maintenance_applied_effect_total

    if int(valid_idx.shape[0]) > 0 and not prediction_disabled:
        valid_slots = slots[valid_idx]
        _update_prediction_bookkeeping(
            state,
            runner,
            valid_slots,
            np.asarray(actions[valid_idx], dtype=np.int32),
            state.last_energy_delta[valid_slots].astype(np.float32, copy=False),
            state.last_injury_delta[valid_slots].astype(np.float32, copy=False),
            tick_index=int(tick_index),
        )

    if _as_bool(getattr(cfg_obj, "maintenance_enable", False)):
        runner._maintenance_tick_count_gen = int(getattr(runner, "_maintenance_tick_count_gen", 0)) + 1
        runner._maintenance_action_opportunities_gen = int(getattr(runner, "_maintenance_action_opportunities_gen", 0)) + int(
            agent_ids.shape[0]
        )
        runner._maintenance_schedule_requests_gen = int(
            getattr(runner, "_maintenance_schedule_requests_gen", 0)
        ) + int(native_summary.get("maintenance_schedule_requests", 0))
        runner._maintenance_apply_count_gen = int(getattr(runner, "_maintenance_apply_count_gen", 0)) + int(
            native_summary.get("maintenance_apply_count", 0)
        )
        runner._maintenance_energy_cost_total_gen = float(
            getattr(runner, "_maintenance_energy_cost_total_gen", 0.0)
        ) + float(native_summary.get("maintenance_energy_cost_sum", 0.0))
        runner._maintenance_opportunity_cost_total_gen = float(
            getattr(runner, "_maintenance_opportunity_cost_total_gen", 0.0)
        ) + float(native_summary.get("maintenance_opportunity_cost_sum", 0.0))
        runner._maintenance_effect_total_gen = float(getattr(runner, "_maintenance_effect_total_gen", 0.0)) + float(
            native_summary.get("maintenance_applied_effect_sum", 0.0)
        )
        opportunities = float(max(1, int(getattr(runner, "_maintenance_action_opportunities_gen", 0))))
        ratio = float(getattr(runner, "_maintenance_schedule_requests_gen", 0)) / opportunities
        threshold = _as_float(getattr(cfg_obj, "maintenance_domination_threshold", 0.75), 0.75)
        min_ticks = _as_int(getattr(cfg_obj, "maintenance_domination_min_ticks", 8), 8)
        triggered_now = int(getattr(runner, "_maintenance_tick_count_gen", 0)) >= int(min_ticks) and ratio >= float(threshold)
        prev = dict(getattr(runner, "_maintenance_domination_annotation", {}) or {})
        was_triggered = bool(prev.get("triggered", False))
        runner._maintenance_domination_annotation = {
            "triggered": bool(was_triggered or triggered_now),
            "tick": int(tick_index),
            "ratio": float(ratio),
            "threshold": float(threshold),
            "requests": int(getattr(runner, "_maintenance_schedule_requests_gen", 0)),
            "opportunities": int(getattr(runner, "_maintenance_action_opportunities_gen", 0)),
        }
        buckets["maintenance_schedule_requests"] = buckets.get("maintenance_schedule_requests", 0.0) + float(
            native_summary.get("maintenance_schedule_requests", 0.0)
        )
        buckets["maintenance_apply_count"] = buckets.get("maintenance_apply_count", 0.0) + float(
            native_summary.get("maintenance_apply_count", 0.0)
        )

    elapsed = time.perf_counter() - t0
    buckets["cxxwave1f_scalar_backbone_last"] = float(elapsed)
    buckets["cxxwave1f_scalar_backbone"] = float(buckets.get("cxxwave1f_scalar_backbone", 0.0) + elapsed)
    _write_cxxwave1f_summary_if_requested(runner, summary)
    return True


def _runnercore4_structural_payload_payload(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore4_structural_payload_stats", None)
    if not isinstance(stats, dict):
        return None
    out: dict[str, Any] = {"schema_version": RUNNERCORE4_STRUCTURAL_PAYLOAD_SCHEMA_VERSION}
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, np.integer)):
            out[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = value
    return out


def _runnercore5_child_construction_payload(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore5_child_construction_stats", None)
    if not isinstance(stats, dict):
        return None
    out: dict[str, Any] = {"schema_version": RUNNERCORE5_CHILD_CONSTRUCTION_SCHEMA_VERSION}
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, np.integer)):
            out[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = value
    return out


def _runnercore6_resident_child_row_payload(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore6_resident_child_row_stats", None)
    if not isinstance(stats, dict):
        return None
    out: dict[str, Any] = {"schema_version": RUNNERCORE6_RESIDENT_CHILD_ROW_SCHEMA_VERSION}
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, np.integer)):
            out[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = value
    return out


def _runnercore9_c1_payload_or_lazy_newborn_payload(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore9_c1_payload_or_lazy_newborn_stats", None)
    if not isinstance(stats, dict):
        return None
    out: dict[str, Any] = {"schema_version": RUNNERCORE9_C1_PAYLOAD_OR_LAZY_NEWBORN_SCHEMA_VERSION}
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, np.integer)):
            out[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = value
    return out


def _runnercore10_typed_child_payload(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore10_typed_child_stats", None)
    if not isinstance(stats, dict):
        return None
    out: dict[str, Any] = {"schema_version": RUNNERCORE10_TYPED_CHILD_SCHEMA_VERSION}
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, np.integer)):
            out[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = value
    return out


def _runnercore11_c1_typed_genome_row_payload(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    stats = getattr(cfg, "_runnercore11_c1_typed_genome_row_stats", None)
    if not isinstance(stats, dict):
        return None
    out: dict[str, Any] = {"schema_version": RUNNERCORE11_C1_TYPED_GENOME_ROW_SCHEMA_VERSION}
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, np.integer)):
            out[str(key)] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[str(key)] = float(value)
        else:
            out[str(key)] = value
    return out


def _runnercore9_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None or not _runnercore9_requested(runner):
        return None
    stats = getattr(cfg, "_runnercore9_c1_payload_or_lazy_newborn_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": RUNNERCORE9_C1_PAYLOAD_OR_LAZY_NEWBORN_SCHEMA_VERSION,
            "c1_payload_contract": "trace_only_scalar_normalize_skip_parity_blocked",
            "lazy_newborn_contract": (
                "resident_rows_defer_compat_agent_until_export"
                if _as_bool(getattr(cfg, "runnercore9_lazy_newborn_materialization_enable", False))
                else "not_enabled"
            ),
            "native_cpp_status": "not_attached_python_c1_and_materialization_boundary",
            "newborn_agent_materializations_deferred": 0,
            "newborn_agent_materializations_on_demand": 0,
            "policy_source_forced_materializations": 0,
            "policy_bank_sync_forced_materializations": 0,
            "generation_end_materializations": 0,
            "checkpoint_forced_materializations": 0,
            "unsupported_reader_materializations": 0,
            "resident_rows_without_agent_objects": 0,
            "event_boundary_export_rows_avoided": 0,
            "hidden_agent_sync_rows": 0,
            "unsupported_fallback_count": 0,
        }
        try:
            setattr(cfg, "_runnercore9_c1_payload_or_lazy_newborn_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore10_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None or not _runnercore10_requested(runner):
        return None
    stats = getattr(cfg, "_runnercore10_typed_child_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": RUNNERCORE10_TYPED_CHILD_SCHEMA_VERSION,
            "typed_child_contract": "resident_child_genome_policy_row_v1",
            "native_cpp_status": "not_attached_python_typed_child_contract",
            "typed_child_specs_created": 0,
            "typed_child_specs_attached": 0,
            "typed_child_policy_rows_attached": 0,
            "typed_child_hmem_rows_attached": 0,
            "typed_child_direct_mlp_rows": 0,
            "typed_child_direct_gru_rows": 0,
            "typed_child_unsupported_rows": 0,
            "genome_from_dict_avoided": 0,
            "policy_object_creation_avoided": 0,
            "agent_materialization_avoided": 0,
            "compatibility_materializations_forced": 0,
            "newborn_agent_materializations_deferred": 0,
            "newborn_agent_materializations_on_demand": 0,
            "generation_end_materializations": 0,
            "checkpoint_forced_materializations": 0,
            "unsupported_reader_materializations": 0,
            "resident_rows_without_agent_objects": 0,
            "event_boundary_export_rows_avoided": 0,
            "hidden_agent_sync_rows": 0,
            "per_tick_agent_sync_rows": 0,
            "unsupported_typed_child_fallback_count": 0,
            "unsupported_fallback_count": 0,
            "forced_materialization_reasons": {},
        }
        try:
            setattr(cfg, "_runnercore10_typed_child_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore9_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore9_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore10_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore10_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore10_reason_add(runner: Any, reason: str, value: int = 1) -> None:
    stats = _runnercore10_stats(runner)
    if stats is None:
        return
    reasons = stats.get("forced_materialization_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    key = str(reason)
    reasons[key] = int(reasons.get(key, 0) or 0) + int(value)
    stats["forced_materialization_reasons"] = reasons


def _runnercore11_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None or not _runnercore11_requested(runner):
        return None
    stats = getattr(cfg, "_runnercore11_c1_typed_genome_row_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": RUNNERCORE11_C1_TYPED_GENOME_ROW_SCHEMA_VERSION,
            "typed_c1_contract": "canonical_c1_payload_projected_to_resident_genome_row",
            "native_cpp_status": "not_attached_python_projection",
            "c1_operator_pipeline_calls": 0,
            "c1_typed_rows_created": 0,
            "c1_typed_rows_attached": 0,
            "policy_rows_attached": 0,
            "hmem_rows_attached": 0,
            "c1_typed_rows_missing_field_fail_closed": 0,
            "c1_typed_rows_unsupported_kind": 0,
            "c1_payload_dict_roundtrips_avoided": 0,
            "genome_from_dict_avoided": 0,
            "genome_from_dict_forced": 0,
            "genome_materialization_avoided": 0,
            "genome_materialization_forced": 0,
            "generation_end_proxy_agents_created": 0,
            "generation_end_proxy_genome_materialization_avoided": 0,
            "policy_object_creation_avoided": 0,
            "policy_object_creation_forced": 0,
            "forced_materialization_policy_source": 0,
            "forced_materialization_policy_bank_sync": 0,
            "forced_materialization_behavior_log": 0,
            "compatibility_export_forced": 0,
            "compatibility_export_avoided": 0,
            "birth_success_count": 0,
            "birth_failure_count": 0,
            "hidden_agent_sync_rows": 0,
            "per_tick_agent_sync_rows": 0,
            "unsupported_fallback_count": 0,
            "fast_weighted_food_spawn_enabled": 0,
            "fast_weighted_food_spawn_calls": 0,
            "fast_weighted_food_spawn_draws": 0,
            "fast_weighted_food_spawn_placements": 0,
            "fast_weighted_food_spawn_incomplete": 0,
            "runnercore12_c1_payload_digest_reuse_enabled": 0,
            "runnercore12_c1_payload_digest_reuse_rows": 0,
            "runnercore12_c1_payload_digest_reuse_fallbacks": 0,
            "runnercore13_c1_projection_digest_sample_only_enabled": 0,
            "runnercore13_c1_projection_digest_full_rows": 0,
            "runnercore13_c1_projection_digest_skipped_rows": 0,
            "runnercore15_c1_noop_normalize_skip_enabled": 0,
            "runnercore15_c1_noop_normalize_skipped": 0,
            "runnercore16_c1_empty_structure_fastpath_enabled": 0,
            "runnercore16_c1_empty_structure_fastpath_hits": 0,
            "runnercore16_c1_empty_structure_fastpath_misses": 0,
            "runnercore16_c1_empty_structure_fastpath_noop_draws": 0,
            "runnercore17_resident_birth_parent_candidates": 0,
            "runnercore17_resident_birth_parent_specs_used": 0,
            "runnercore17_resident_birth_parent_state_refreshes": 0,
            "runnercore17_resident_birth_parent_materializations_avoided": 0,
            "runnercore17_resident_birth_parent_parent_import_syncs": 0,
            "runnercore17_resident_birth_parent_missing_specs": 0,
            "runnercore17_resident_birth_parent_unsupported_fallback_count": 0,
            "runnercore18_resident_parent_minimal_refresh_enabled": 0,
            "runnercore18_resident_parent_minimal_refreshes": 0,
            "runnercore18_resident_parent_full_refreshes": 0,
            "runnercore18_resident_parent_full_refresh_skipped_no_slot": 0,
            "runnercore18_resident_parent_minimal_import_syncs": 0,
            "runnercore18_resident_parent_unsupported_fallback_count": 0,
            "forced_materialization_reasons": {},
            "c1_projection_trace_rows": 0,
            "c1_projection_trace_sample": [],
            "required_field_failures": {},
        }
        try:
            setattr(cfg, "_runnercore11_c1_typed_genome_row_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore11_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore11_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore11_reason_add(runner: Any, reason: str, value: int = 1) -> None:
    stats = _runnercore11_stats(runner)
    if stats is None:
        return
    reasons = stats.get("forced_materialization_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
    key = str(reason)
    reasons[key] = int(reasons.get(key, 0) or 0) + int(value)
    stats["forced_materialization_reasons"] = reasons


def _runnercore17_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    _runnercore11_stat_add(runner, key, value)


def _runnercore17_resident_birth_parent_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _runnercore6_resident_attach_enabled(runner)
        and _runnercore11_requested(runner)
        and _as_bool(getattr(cfg, "runnercore11_c1_typed_rows_enable", False))
        and _as_bool(getattr(cfg, "runnercore17_resident_birth_parent_enable", False))
    )


def _runnercore17_resident_birth_parent_fail_closed(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore17_resident_birth_parent_fail_closed", False))


def _runnercore18_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    _runnercore11_stat_add(runner, key, value)


def _runnercore18_resident_parent_minimal_refresh_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _runnercore17_resident_birth_parent_enabled(runner)
        and _as_bool(getattr(cfg, "runnercore18_resident_parent_minimal_refresh_enable", False))
    )


def _runnercore18_population_has_free_slot(runner: Any) -> bool:
    slots = getattr(getattr(runner, "pop", None), "_slots", None)
    free_slots = getattr(slots, "free_slots", None)
    try:
        return bool(free_slots)
    except Exception:
        return False


def _runnercore17_typed_child_specs(runner: Any) -> dict[int, Any]:
    specs = getattr(getattr(runner, "pop", None), "_runnercore10_typed_child_specs", {})
    return specs if isinstance(specs, dict) else {}


def _runnercore17_resident_parent_spec(
    state: RunnerCore1TickState,
    runner: Any,
    aid: int,
) -> Any | None:
    spec = _runnercore17_typed_child_specs(runner).get(int(aid))
    if spec is None or getattr(spec, "c1_typed_genome_row", None) is None:
        return None
    slot = state.slot_for_agent_id(int(aid))
    if slot < 0 or not bool(state.alive[int(slot)]):
        return None
    if _as_int(getattr(spec, "slot", -1), -1) != int(slot):
        return None
    if _as_int(getattr(spec, "slot_token", -1), -1) != int(state.slot_tokens[int(slot)]):
        return None
    return spec


def _runnercore17_refresh_resident_parent_spec_from_state(
    state: RunnerCore1TickState,
    runner: Any,
    spec: Any,
    *,
    full: bool = True,
    count_r17_refresh: bool = True,
) -> bool:
    aid = _as_int(getattr(spec, "agent_id", -1), -1)
    slot = state.slot_for_agent_id(int(aid))
    if aid < 0 or slot < 0 or not bool(state.alive[int(slot)]):
        return False
    if _as_int(getattr(spec, "slot", -1), -1) != int(slot):
        return False
    if _as_int(getattr(spec, "slot_token", -1), -1) != int(state.slot_tokens[int(slot)]):
        return False

    spec.alive = bool(state.alive[int(slot)])
    spec.energy = float(state.energy[int(slot)])
    spec.health = float(state.health[int(slot)])
    spec.injury = float(state.injury[int(slot)])
    spec.scar = float(state.scar[int(slot)])
    spec.rest_streak = int(state.rest_streak[int(slot)])
    spec.repro_cooldown = int(state.repro_cooldown[int(slot)])
    if not full:
        try:
            spec._runnercore18_minimal_parent_refreshed = True
            spec._runnercore18_full_parent_refreshed = False
        except Exception:
            pass
        if count_r17_refresh:
            _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_state_refreshes", 1)
        _runnercore18_stat_add(runner, "runnercore18_resident_parent_minimal_refreshes", 1)
        return True

    d = getattr(spec, "drives", None)
    if d is not None and getattr(d, "shape", None) is not None and int(d.shape[0]) >= 4:
        d[:4] = state.drives[int(slot), :4]
    else:
        spec.drives = np.asarray(state.drives[int(slot)], dtype=np.float32).copy()
    spec.chemistry = np.asarray(state.chemistry[int(slot)], dtype=np.float32).copy()
    spec.maintenance = np.asarray(state.maintenance[int(slot)], dtype=np.float32).copy()
    if hasattr(runner, "_ensure_drive_arrays"):
        runner._ensure_drive_arrays(required_slots=int(state.capacity))
    if hasattr(runner, "_drive_hunger") and getattr(runner, "_drive_hunger", None) is not None:
        if int(slot) < int(runner._drive_hunger.shape[0]):
            runner._drive_hunger[int(slot)] = np.float32(state.drives[int(slot), 0])
            runner._drive_pain[int(slot)] = np.float32(state.drives[int(slot), 1])
            runner._drive_fatigue[int(slot)] = np.float32(state.drives[int(slot), 2])
            runner._drive_fear[int(slot)] = np.float32(state.drives[int(slot), 3])
            if getattr(runner, "_drive_slot_token", None) is not None and int(slot) < int(
                runner._drive_slot_token.shape[0]
            ):
                runner._drive_slot_token[int(slot)] = np.int32(state.slot_tokens[int(slot)])

    st = getattr(spec, "stats", None)
    if st is not None:
        st.survival_steps = int(state.survival_steps[int(slot)])
        st.energy_collected = float(state.energy_collected[int(slot)])
        st.early_survival_steps = int(state.early_survival_steps[int(slot)])
        st.early_energy_collected = float(state.early_energy_collected[int(slot)])
        st.early_injury_gained = float(state.early_injury_gained[int(slot)])
        st.last_energy_delta = float(state.last_energy_delta[int(slot)])
        st.last_health_delta = float(state.last_health_delta[int(slot)])
        st.last_injury_delta = float(state.last_injury_delta[int(slot)])
        st.last_action = int(state.last_action[int(slot)])
        st.last_last_action = int(state.last_last_action[int(slot)])
        st.action_switches = int(state.action_switches[int(slot)])
        st.action_oscillations = int(state.action_oscillations[int(slot)])
        st.episode0_switches = int(state.episode0_switches[int(slot)])
        st.episode0_oscillations = int(state.episode0_oscillations[int(slot)])
        st.attacks = int(state.attacks[int(slot)])
        st.idle_charge_paid = float(state.idle_charge_paid[int(slot)])
        st.selfharm_events = int(state.selfharm_events[int(slot)])
        st.episode0_selfharm_events = int(state.episode0_selfharm_events[int(slot)])
        st.drive_volatility = float(state.drive_volatility[int(slot)])
        st.drive_jitter = float(state.drive_jitter[int(slot)])
        st.prev_drive_delta = np.asarray(state.prev_drive_delta[int(slot)], dtype=np.float32).copy()
        st.pred_err_sum = float(state.pred_err_sum[int(slot)])
        st.pred_err_count = int(state.pred_err_count[int(slot)])
        st.prediction_update_count = int(state.prediction_update_count[int(slot)])
        st.prediction_state_delta_abs_sum = float(state.prediction_state_delta_abs_sum[int(slot)])
        st.prediction_state_delta_abs_max = float(state.prediction_state_delta_abs_max[int(slot)])
        st.prediction_reset_count = int(state.prediction_reset_count[int(slot)])
        st.episode0_pred_err_sum = float(state.episode0_pred_err_sum[int(slot)])
        st.episode0_pred_err_count = int(state.episode0_pred_err_count[int(slot)])
        st.late_pred_err_sum = float(state.late_pred_err_sum[int(slot)])
        st.late_pred_err_count = int(state.late_pred_err_count[int(slot)])
        st.adult_pred_err_ema = float(state.adult_pred_err_ema[int(slot)])
        st.exp_dE = np.asarray(state.exp_dE[int(slot)], dtype=np.float32).copy()
        st.exp_dInj = np.asarray(state.exp_dInj[int(slot)], dtype=np.float32).copy()
        st.pred_queue = deque(state.pred_queue[int(slot)])
        st.first_repro_step = state.first_repro_step[int(slot)]
        st.offspring_count = int(state.offspring_count[int(slot)])
        st.biome_id = int(state.biome_id[int(slot)])
        st.early_life_weight = float(state.early_life_weight[int(slot)])

    try:
        spec._runnercore18_full_parent_refreshed = True
    except Exception:
        pass
    if count_r17_refresh:
        _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_state_refreshes", 1)
    if _runnercore18_resident_parent_minimal_refresh_enabled(runner):
        _runnercore18_stat_add(runner, "runnercore18_resident_parent_full_refreshes", 1)
    return True


def _runnercore18_sync_minimal_resident_parent_to_state(
    state: RunnerCore1TickState,
    runner: Any,
    spec: Any,
) -> int:
    aid = _as_int(getattr(spec, "agent_id", -1), -1)
    slot = state.slot_for_agent_id(int(aid))
    if aid < 0 or slot < 0:
        return -1
    if _as_int(getattr(spec, "slot", -1), -1) != int(slot):
        return -1
    if _as_int(getattr(spec, "slot_token", -1), -1) != int(state.slot_tokens[int(slot)]):
        return -1
    state.energy[int(slot)] = np.float32(_as_float(getattr(spec, "energy", 0.0), 0.0))
    state.repro_cooldown[int(slot)] = np.int32(_as_int(getattr(spec, "repro_cooldown", 0), 0))
    state.alive[int(slot)] = bool(getattr(spec, "alive", True))
    _runnercore18_stat_add(runner, "runnercore18_resident_parent_minimal_import_syncs", 1)
    return int(slot)


def _runnercore10_store_typed_child_spec(runner: Any, child_spec: Any, *, reason: str) -> None:
    pop = getattr(runner, "pop", None)
    if pop is None:
        _runnercore10_stat_add(runner, "unsupported_typed_child_fallback_count", 1)
        _runnercore10_stat_add(runner, "unsupported_fallback_count", 1)
        return
    specs = getattr(pop, "_runnercore10_typed_child_specs", None)
    if not isinstance(specs, dict):
        specs = {}
        try:
            setattr(pop, "_runnercore10_typed_child_specs", specs)
        except Exception:
            _runnercore10_stat_add(runner, "unsupported_typed_child_fallback_count", 1)
            _runnercore10_stat_add(runner, "unsupported_fallback_count", 1)
            return
    try:
        setattr(child_spec, "typed_materialization_state", "resident_only")
    except Exception:
        pass
    specs[int(getattr(child_spec, "agent_id"))] = child_spec
    _runnercore10_stat_add(runner, "newborn_agent_materializations_deferred", 1)
    _runnercore10_stat_add(runner, "resident_rows_without_agent_objects", 1)
    _runnercore10_stat_add(runner, "event_boundary_export_rows_avoided", 1)
    _runnercore10_stat_add(runner, "agent_materialization_avoided", 1)
    _runnercore10_reason_add(runner, f"deferred_{reason}", 1)


def _runnercore5_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None:
        return None
    if not _runnercore5_requested(runner):
        return None
    stats = getattr(cfg, "_runnercore5_child_construction_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": RUNNERCORE5_CHILD_CONSTRUCTION_SCHEMA_VERSION,
            "policy_contract": "independent_child_policy_objects_flat_self_rng",
        }
        try:
            setattr(cfg, "_runnercore5_child_construction_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore5_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore5_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore6_stats(runner: Any) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None:
        return None
    if not _runnercore6_requested(runner):
        return None
    stats = getattr(cfg, "_runnercore6_resident_child_row_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "schema_version": RUNNERCORE6_RESIDENT_CHILD_ROW_SCHEMA_VERSION,
            "row_contract": "resident_child_row_spec_pre_agent_attach",
            "runnercore6_child_row_specs_created": 0,
            "runnercore6_resident_first_attaches": 0,
            "runnercore6_pre_attach_agent_materializations": 0,
            "runnercore6_pre_attach_genome_materializations": 0,
            "runnercore6_pre_attach_policy_materializations": 0,
            "runnercore6_post_attach_export_materializations": 0,
            "runnercore6_agent_materializations_deferred": 0,
            "runnercore6_genome_from_dict_deferred_or_avoided": 0,
            "runnercore6_policy_object_creation_deferred_or_avoided": 0,
            "runnercore6_event_import_rows_avoided": 0,
            "runnercore6_event_export_rows_avoided": 0,
            "runnercore25_full_birth_row_specs_created": 0,
            "runnercore25_full_birth_row_specs_applied": 0,
            "runnercore25_full_birth_newborn_attach_rows": 0,
            "runnercore25_generation_end_materializations": 0,
            "runnercore25_pre_attach_agent_materializations_avoided": 0,
            "pre_attach_full_genome_materialization_blocker": "c1_c2_payload_still_returns_python_genome",
            "pre_attach_policy_materialization_blocker": "policy_bank_attach_still_requires_child_policy_or_parent_copy",
        }
        try:
            setattr(cfg, "_runnercore6_resident_child_row_stats", stats)
        except Exception:
            return None
    return stats


def _runnercore6_stat_add(runner: Any, key: str, value: int | float = 1) -> None:
    stats = _runnercore6_stats(runner)
    if stats is None:
        return
    old = stats.get(str(key), 0)
    if isinstance(value, float):
        stats[str(key)] = float(old or 0.0) + float(value)
    else:
        stats[str(key)] = int(old or 0) + int(value)


def _runnercore6_resident_attach_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _runnercore6_requested(runner)
        and _as_bool(getattr(cfg, "runnercore6_candidate_enable", False))
        and _as_bool(getattr(cfg, "runnercore6_resident_newborn_row_attach_enable", False))
    )


def _runnercore5_direct_attach_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _runnercore5_requested(runner)
        and _as_bool(getattr(cfg, "runnercore5_candidate_enable", False))
        and _as_bool(getattr(cfg, "runnercore5_direct_newborn_mlp_bank_attach_enable", False))
    )


def _runnercore7_direct_gru_attach_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _runnercore5_requested(runner)
        and _as_bool(getattr(cfg, "runnercore5_candidate_enable", False))
        and _as_bool(getattr(cfg, "runnercore7_direct_newborn_gru_bank_attach_enable", False))
    )


def _runnercore5_direct_attach_newborn_mlp_rows(
    runner: Any,
    births: list[tuple[Any, tuple[int, int], Any]],
    buckets: dict[str, float],
) -> set[int]:
    if not _runnercore5_direct_attach_enabled(runner):
        return set()

    t0 = time.perf_counter()
    attached_ids: set[int] = set()
    rows: list[tuple[Any, Any, int, int]] = []
    unsupported = 0
    skipped = 0
    unsupported_reasons: Counter[str] = Counter()
    _runnercore5_stat_add(runner, "direct_mlp_bank_attach_attempts", 1)
    _runnercore5_stat_add(runner, "direct_mlp_bank_attach_candidate_birth_rows", len(births))

    def _unsupported(reason: str) -> None:
        nonlocal unsupported
        unsupported += 1
        unsupported_reasons[str(reason)] += 1

    if _as_bool(getattr(runner, "_bank_dirty_mlp", True)):
        _unsupported("mlp_bank_dirty")
    elif getattr(runner, "_mlp_pack", None) is None:
        _unsupported("mlp_pack_missing")
    else:
        cap = int(getattr(getattr(runner, "_mlp_pack", None), "shape", [0])[0] or 0)
        for child, _pos, parent in births:
            child_policy = getattr(child, "policy", None)
            parent_policy = getattr(parent, "policy", None)
            if not (hasattr(child_policy, "net") and hasattr(parent_policy, "net")):
                skipped += 1
                continue
            try:
                src_slot = int(getattr(parent, "slot", -1))
                dst_slot = int(getattr(child, "slot", -1))
                parent_token = int(getattr(parent, "slot_token", 0))
                parent_bank_token = int(getattr(parent, "_bank_token", -1))
            except Exception:
                _unsupported("slot_or_token_read_failed")
                continue
            if src_slot < 0 or dst_slot < 0 or src_slot >= cap or dst_slot >= cap:
                _unsupported("slot_out_of_bank_capacity")
                continue
            if parent_token != parent_bank_token:
                _unsupported("parent_bank_token_mismatch")
                continue
            rows.append((child, parent, src_slot, dst_slot))

    if unsupported:
        _runnercore5_stat_add(runner, "direct_mlp_bank_attach_unsupported_rows", int(unsupported))
        stats = _runnercore5_stats(runner)
        if stats is not None:
            existing = stats.get("direct_mlp_bank_attach_unsupported_reasons", {})
            if not isinstance(existing, dict):
                existing = {}
            for reason, count in unsupported_reasons.items():
                existing[str(reason)] = int(existing.get(str(reason), 0) or 0) + int(count)
            stats["direct_mlp_bank_attach_unsupported_reasons"] = existing
        if _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore5_fail_closed_on_unsupported", False)):
            _runnercore5_stat_add(runner, "direct_mlp_bank_attach_fail_closed", 1)
            raise RunnerCore1UnsupportedError(
                "RUNNERCORE5-UNSUPPORTED: direct newborn MLP bank attach encountered unsupported rows"
            )
    if skipped:
        _runnercore5_stat_add(runner, "direct_mlp_bank_attach_skipped_non_mlp_rows", int(skipped))

    if rows:
        src = torch.as_tensor([src_slot for _child, _parent, src_slot, _dst_slot in rows], dtype=torch.long, device=runner._bank_device)
        dst = torch.as_tensor([dst_slot for _child, _parent, _src_slot, dst_slot in rows], dtype=torch.long, device=runner._bank_device)
        for name in ("_mlp_pack", "_W1", "_b1", "_W2", "_b2", "_W3", "_b3"):
            tensor = getattr(runner, name, None)
            if tensor is None:
                continue
            tensor.index_copy_(0, dst, tensor.index_select(0, src))
        for child, _parent, _src_slot, _dst_slot in rows:
            try:
                child._bank_token = int(getattr(child, "slot_token", 0))
                child._eiel_force_full_mlp_bank_sync_once = False
                attached_ids.add(int(getattr(child, "agent_id", -1)))
            except Exception:
                continue
        _runnercore5_stat_add(runner, "direct_mlp_bank_attach_calls", 1)
        _runnercore5_stat_add(runner, "direct_mlp_bank_attach_rows", len(rows))
        _runnercore5_stat_add(runner, "direct_mlp_bank_attach_sync_rows_avoided", len(rows))

    elapsed = time.perf_counter() - t0
    buckets["runnercore5_direct_mlp_bank_attach"] = float(
        buckets.get("runnercore5_direct_mlp_bank_attach", 0.0) or 0.0
    ) + float(elapsed)
    return attached_ids


def _runnercore7_direct_attach_newborn_gru_rows(
    runner: Any,
    births: list[tuple[Any, tuple[int, int], Any]],
    buckets: dict[str, float],
) -> set[int]:
    if not _runnercore7_direct_gru_attach_enabled(runner):
        return set()

    t0 = time.perf_counter()
    attached_ids: set[int] = set()
    rows: list[tuple[Any, Any, int, int]] = []
    unsupported = 0
    skipped = 0
    unsupported_reasons: Counter[str] = Counter()
    _runnercore5_stat_add(runner, "direct_gru_bank_attach_attempts", 1)
    _runnercore5_stat_add(runner, "direct_gru_bank_attach_candidate_birth_rows", len(births))

    def _unsupported(reason: str) -> None:
        nonlocal unsupported
        unsupported += 1
        unsupported_reasons[str(reason)] += 1

    if _as_bool(getattr(runner, "_bank_dirty_gru", True)):
        _unsupported("gru_bank_dirty")
    elif getattr(runner, "_gru_pack", None) is None:
        _unsupported("gru_pack_missing")
    else:
        cap = int(getattr(getattr(runner, "_gru_pack", None), "shape", [0])[0] or 0)
        for child, _pos, parent in births:
            child_policy = getattr(child, "policy", None)
            parent_policy = getattr(parent, "policy", None)
            if not (hasattr(child_policy, "gru") and hasattr(parent_policy, "gru")):
                skipped += 1
                continue
            try:
                src_slot = int(getattr(parent, "slot", -1))
                dst_slot = int(getattr(child, "slot", -1))
                parent_token = int(getattr(parent, "slot_token", 0))
                parent_bank_token = int(getattr(parent, "_gru_bank_token", -1))
            except Exception:
                _unsupported("slot_or_token_read_failed")
                continue
            if src_slot < 0 or dst_slot < 0 or src_slot >= cap or dst_slot >= cap:
                _unsupported("slot_out_of_bank_capacity")
                continue
            if parent_token != parent_bank_token:
                _unsupported("parent_gru_bank_token_mismatch")
                continue
            rows.append((child, parent, src_slot, dst_slot))

    if unsupported:
        _runnercore5_stat_add(runner, "direct_gru_bank_attach_unsupported_rows", int(unsupported))
        stats = _runnercore5_stats(runner)
        if stats is not None:
            existing = stats.get("direct_gru_bank_attach_unsupported_reasons", {})
            if not isinstance(existing, dict):
                existing = {}
            for reason, count in unsupported_reasons.items():
                existing[str(reason)] = int(existing.get(str(reason), 0) or 0) + int(count)
            stats["direct_gru_bank_attach_unsupported_reasons"] = existing
        if _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore5_fail_closed_on_unsupported", False)):
            _runnercore5_stat_add(runner, "direct_gru_bank_attach_fail_closed", 1)
            raise RunnerCore1UnsupportedError(
                "RUNNERCORE7-UNSUPPORTED: direct newborn GRU bank attach encountered unsupported rows"
            )
    if skipped:
        _runnercore5_stat_add(runner, "direct_gru_bank_attach_skipped_non_gru_rows", int(skipped))

    if rows:
        src = torch.as_tensor([src_slot for _child, _parent, src_slot, _dst_slot in rows], dtype=torch.long, device=runner._bank_device)
        dst = torch.as_tensor([dst_slot for _child, _parent, _src_slot, dst_slot in rows], dtype=torch.long, device=runner._bank_device)
        for name in ("_gru_pack", "_encW", "_encb", "_Wih", "_Whh", "_bih", "_bhh", "_headW", "_headb"):
            tensor = getattr(runner, name, None)
            if tensor is None:
                continue
            tensor.index_copy_(0, dst, tensor.index_select(0, src))
        reset_fn = getattr(runner, "_reset_memory_slots", None)
        if callable(reset_fn):
            reset_fn(dst)
        for child, _parent, _src_slot, _dst_slot in rows:
            try:
                child._gru_bank_token = int(getattr(child, "slot_token", 0))
                child._eiel_force_full_gru_bank_sync_once = False
                attached_ids.add(int(getattr(child, "agent_id", -1)))
            except Exception:
                continue
        _runnercore5_stat_add(runner, "direct_gru_bank_attach_calls", 1)
        _runnercore5_stat_add(runner, "direct_gru_bank_attach_rows", len(rows))
        _runnercore5_stat_add(runner, "direct_gru_bank_attach_sync_rows_avoided", len(rows))

    elapsed = time.perf_counter() - t0
    buckets["runnercore7_direct_gru_bank_attach"] = float(
        buckets.get("runnercore7_direct_gru_bank_attach", 0.0) or 0.0
    ) + float(elapsed)
    return attached_ids


def _runnercore7_static_reproduction_threshold(agent: Any) -> float | None:
    genome = getattr(agent, "genome", None)
    if genome is None:
        return None
    modules = list(getattr(genome, "modules", []) or [])
    for module in modules:
        if not isinstance(module, dict):
            return None
        if str(module.get("module_type", "")) != "life_history":
            continue
        params = module.get("params", {}) if isinstance(module.get("params", {}), dict) else {}
        if "reproduction_threshold" in params:
            return None
    try:
        return float(getattr(genome, "reproduction_threshold", 0.0))
    except Exception:
        return None


def _runnercore3_profile_payload(state: RunnerCore1TickState) -> dict[str, Any]:
    buckets = {name: float(state.runnercore3_profile_buckets_ms.get(name, 0.0)) for name in RUNNERCORE3_PROFILE_BUCKETS}
    c = state.counters
    counts = {
        "births_applied": int(c.birth_events_applied),
        "birth_candidates_considered": int(state.runnercore3_profile_counts.get("birth_candidates_considered", 0)),
        "birth_candidates_exported": int(state.runnercore3_profile_counts.get("birth_candidates_exported", 0)),
        "runnercore22_full_birth_boundary_bridge_export_rows": int(
            state.runnercore3_profile_counts.get("runnercore22_full_birth_boundary_bridge_export_rows", 0)
        ),
        "runnercore22_full_birth_boundary_bridge_import_rows": int(
            state.runnercore3_profile_counts.get("runnercore22_full_birth_boundary_bridge_import_rows", 0)
        ),
        "runnercore22_full_birth_boundary_bridge_import_rows_considered": int(
            state.runnercore3_profile_counts.get("runnercore22_full_birth_boundary_bridge_import_rows_considered", 0)
        ),
        "runnercore22_full_birth_boundary_bridge_import_rows_avoided": int(
            state.runnercore3_profile_counts.get("runnercore22_full_birth_boundary_bridge_import_rows_avoided", 0)
        ),
        "runnercore22_full_birth_boundary_bridge_direct_cooldown_rows": int(
            state.runnercore3_profile_counts.get("runnercore22_full_birth_boundary_bridge_direct_cooldown_rows", 0)
        ),
        "runnercore27_full_birth_cooldown_prefilter_rows": int(
            state.runnercore3_profile_counts.get("runnercore27_full_birth_cooldown_prefilter_rows", 0)
        ),
        "runnercore27_full_birth_cooldown_prefilter_considered_rows": int(
            state.runnercore3_profile_counts.get("runnercore27_full_birth_cooldown_prefilter_considered_rows", 0)
        ),
        "runnercore30_full_birth_branch_prefilter_rows": int(
            state.runnercore3_profile_counts.get("runnercore30_full_birth_branch_prefilter_rows", 0)
        ),
        "runnercore30_full_birth_branch_prefilter_self_gate_rows": int(
            state.runnercore3_profile_counts.get("runnercore30_full_birth_branch_prefilter_self_gate_rows", 0)
        ),
        "runnercore30_full_birth_branch_prefilter_isolated_no_mate_rows": int(
            state.runnercore3_profile_counts.get("runnercore30_full_birth_branch_prefilter_isolated_no_mate_rows", 0)
        ),
        "runnercore30_full_birth_branch_prefilter_parent_refresh_rows": int(
            state.runnercore3_profile_counts.get("runnercore30_full_birth_branch_prefilter_parent_refresh_rows", 0)
        ),
        "runnercore34_full_birth_resident_decision_rows": int(
            state.runnercore3_profile_counts.get("runnercore34_full_birth_resident_decision_rows", 0)
        ),
        "runnercore34_full_birth_mate_selected_rows": int(
            state.runnercore3_profile_counts.get("runnercore34_full_birth_mate_selected_rows", 0)
        ),
        "runnercore34_full_birth_spawn_selected_rows": int(
            state.runnercore3_profile_counts.get("runnercore34_full_birth_spawn_selected_rows", 0)
        ),
        "runnercore34_full_birth_parent_export_rows": int(
            state.runnercore3_profile_counts.get("runnercore34_full_birth_parent_export_rows", 0)
        ),
        "runnercore34_full_birth_candidate_exports_avoided": int(
            state.runnercore3_profile_counts.get("runnercore34_full_birth_candidate_exports_avoided", 0)
        ),
        "runnercore25_full_birth_row_specs_created": int(
            state.runnercore3_profile_counts.get("runnercore25_full_birth_row_specs_created", 0)
        ),
        "runnercore25_full_birth_row_specs_applied": int(
            state.runnercore3_profile_counts.get("runnercore25_full_birth_row_specs_applied", 0)
        ),
        "runnercore25_full_birth_newborn_attach_rows": int(
            state.runnercore3_profile_counts.get("runnercore25_full_birth_newborn_attach_rows", 0)
        ),
        "runnercore25_generation_end_materializations": int(
            state.runnercore3_profile_counts.get("runnercore25_generation_end_materializations", 0)
        ),
        "runnercore23_jungle_predator_bridge_export_rows": int(
            state.runnercore3_profile_counts.get("runnercore23_jungle_predator_bridge_export_rows", 0)
        ),
        "runnercore23_jungle_predator_bridge_import_rows": int(
            state.runnercore3_profile_counts.get("runnercore23_jungle_predator_bridge_import_rows", 0)
        ),
        "runnercore24_jungle_predator_native_considered_rows": int(
            state.runnercore3_profile_counts.get("runnercore24_jungle_predator_native_considered_rows", 0)
        ),
        "runnercore24_jungle_predator_native_event_rows": int(
            state.runnercore3_profile_counts.get("runnercore24_jungle_predator_native_event_rows", 0)
        ),
        "runnercore24_jungle_predator_native_death_rows": int(
            state.runnercore3_profile_counts.get("runnercore24_jungle_predator_native_death_rows", 0)
        ),
        "runnercore24_jungle_predator_native_mirror_sync_rows": int(
            state.runnercore3_profile_counts.get("runnercore24_jungle_predator_native_mirror_sync_rows", 0)
        ),
        "runnercore45_jungle_predator_context_cache_rows": int(
            state.runnercore3_profile_counts.get("runnercore45_jungle_predator_context_cache_rows", 0)
        ),
        "runnercore48_chemistry_observation_vectorized_rows": int(
            state.runnercore3_profile_counts.get("runnercore48_chemistry_observation_vectorized_rows", 0)
        ),
        "runnercore49_repro_viability_tick_cache_hits": int(
            state.runnercore3_profile_counts.get("runnercore49_repro_viability_tick_cache_hits", 0)
        ),
        "runnercore49_repro_viability_tick_cache_misses": int(
            state.runnercore3_profile_counts.get("runnercore49_repro_viability_tick_cache_misses", 0)
        ),
        "runnercore50_spawn_first_valid_scan_rows": int(
            state.runnercore3_profile_counts.get("runnercore50_spawn_first_valid_scan_rows", 0)
        ),
        "runnercore50_spawn_first_valid_selected_rows": int(
            state.runnercore3_profile_counts.get("runnercore50_spawn_first_valid_selected_rows", 0)
        ),
        "runnercore26_jungle_predator_vectorized_sample_rows": int(
            state.runnercore3_profile_counts.get("runnercore26_jungle_predator_vectorized_sample_rows", 0)
        ),
        "runnercore26_jungle_predator_vectorized_event_rows": int(
            state.runnercore3_profile_counts.get("runnercore26_jungle_predator_vectorized_event_rows", 0)
        ),
        "boundary_agent_export_rows": int(c.boundary_agent_export_rows),
        "boundary_churn_materialization_rows": int(c.boundary_churn_materialization_rows),
        "event_boundary_agent_create_rows": int(c.event_boundary_agent_create_rows),
        "policy_row_attach_events": int(c.policy_row_attach_events),
        "hmem_row_init_events": int(c.hmem_row_init_events),
        "action_cache_invalidate_events": int(c.action_cache_invalidate_events),
        "action_cache_hits": int(c.action_cache_hits),
        "action_cache_misses": int(c.action_cache_misses),
        "action_cache_refill_rows": int(c.action_cache_refill_rows),
        "active_slots": int(np.count_nonzero(state.alive)),
        "max_capacity": int(state.capacity),
    }
    total_ms = float(sum(buckets.values()))
    ranked = [
        {"bucket": name, "ms": float(value), "share_of_profiled_ms": float(value / total_ms) if total_ms > 0.0 else 0.0}
        for name, value in sorted(buckets.items(), key=lambda item: (-float(item[1]), item[0]))
    ]
    return {
        "schema_version": RUNNERCORE3_PROFILE_SCHEMA_VERSION,
        "buckets_ms": buckets,
        "counts": counts,
        "ranked_buckets": ranked,
        "profiled_ms_total": total_ms,
    }


def _update_patch_context(runner: Any, *, gen: int, tick: int) -> None:
    if hasattr(runner, "_run_spatial_droughts_step_with_budget"):
        runner._run_spatial_droughts_step_with_budget(
            gen=int(gen),
            tick=int(tick),
            guard_call=lambda abs_step: runner.env.spatial_droughts_step(abs_step),
        )
    if not _as_bool(getattr(runner.env, "patch_world_enable", False)):
        return
    abs_step = int(gen) * int(runner.sim_cfg.steps_per_generation) + int(tick)
    runner.env.patch_world_step(abs_step)
    if _as_bool(getattr(runner.env_cfg, "prediction_demand_enable", False)):
        runner._prediction_demand_tick_count_gen = int(getattr(runner, "_prediction_demand_tick_count_gen", 0)) + 1
        runner._prediction_demand_event_count_gen = int(getattr(runner, "_prediction_demand_event_count_gen", 0)) + int(
            getattr(runner.env, "prediction_demand_event_count_last", 0)
        )
        runner._prediction_demand_transition_count_gen = int(
            getattr(runner, "_prediction_demand_transition_count_gen", 0)
        ) + int(getattr(runner.env, "prediction_demand_transition_count_last", 0))
        runner._prediction_demand_resource_consequence_total_gen = float(
            getattr(runner, "_prediction_demand_resource_consequence_total_gen", 0.0)
        ) + float(getattr(runner.env, "prediction_demand_resource_consequence_total_last", 0.0))
    for attr, env_attr, default in (
        ("_patch_food_regrowth_total_gen", "patch_food_regrowth_total_last", 0.0),
        ("_patch_regrowth_base_total_gen", "patch_regrowth_base_total_last", 0.0),
        ("_patch_regrowth_effective_total_gen", "patch_regrowth_effective_total_last", 0.0),
        ("_patch_regrowth_suppressed_total_gen", "patch_regrowth_suppressed_total_last", 0.0),
        ("_patch_regrowth_multiplier_mean_sum_gen", "patch_regrowth_multiplier_mean_last", 1.0),
        ("_patch_regrowth_multiplier_p10_sum_gen", "patch_regrowth_multiplier_p10_last", 1.0),
        ("_patch_regrowth_multiplier_p90_sum_gen", "patch_regrowth_multiplier_p90_last", 1.0),
        ("_patch_waste_regrowth_penalty_mean_sum_gen", "patch_waste_regrowth_penalty_mean_last", 0.0),
        ("_patch_density_regrowth_penalty_mean_sum_gen", "patch_density_regrowth_penalty_mean_last", 0.0),
        ("_patch_high_waste_regrowth_suppression_frac_sum_gen", "patch_high_waste_regrowth_suppression_frac_last", 0.0),
        (
            "_patch_high_density_regrowth_suppression_frac_sum_gen",
            "patch_high_density_regrowth_suppression_frac_last",
            0.0,
        ),
        ("_patch_recovery_positive_total_gen", "patch_recovery_positive_total_last", 0.0),
        ("_patch_recovery_negative_total_gen", "patch_recovery_negative_total_last", 0.0),
        ("_patch_recovery_decay_total_gen", "patch_recovery_decay_total_last", 0.0),
        ("_jungle_recovery_regrowth_boost_total_gen", "jungle_recovery_regrowth_boost_total_last", 0.0),
        ("_jungle_recovery_regrowth_boost_mean_sum_gen", "jungle_recovery_regrowth_boost_mean_last", 0.0),
        ("_jungle_recovery_regrowth_boost_p90_sum_gen", "jungle_recovery_regrowth_boost_p90_last", 0.0),
        ("_patch_regrowth_after_waste_total_gen", "patch_regrowth_after_waste_total_last", 0.0),
        ("_patch_regrowth_after_recovery_total_gen", "patch_regrowth_after_recovery_total_last", 0.0),
        ("_patch_recovery_added_regrowth_total_gen", "patch_recovery_added_regrowth_total_last", 0.0),
        ("_patch_scent_emitted_total_gen", "patch_scent_emitted_total_last", 0.0),
        ("_patch_scent_decayed_total_gen", "patch_scent_decayed_total_last", 0.0),
        ("_patch_waste_emitted_total_gen", "patch_waste_emitted_total_last", 0.0),
        ("_patch_waste_decayed_total_gen", "patch_waste_decayed_total_last", 0.0),
    ):
        setattr(runner, attr, float(getattr(runner, attr, 0.0)) + float(getattr(runner.env, env_attr, default)))
    runner._patch_regrowth_consequence_steps_gen = int(getattr(runner, "_patch_regrowth_consequence_steps_gen", 0)) + 1
    runner._patch_regrowth_multiplier_min_gen = min(
        float(getattr(runner, "_patch_regrowth_multiplier_min_gen", 1.0)),
        float(getattr(runner.env, "patch_regrowth_multiplier_min_last", 1.0)),
    )
    runner._patch_waste_regrowth_penalty_max_gen = max(
        float(getattr(runner, "_patch_waste_regrowth_penalty_max_gen", 0.0)),
        float(getattr(runner.env, "patch_waste_regrowth_penalty_max_last", 0.0)),
    )
    runner._patch_density_regrowth_penalty_max_gen = max(
        float(getattr(runner, "_patch_density_regrowth_penalty_max_gen", 0.0)),
        float(getattr(runner.env, "patch_density_regrowth_penalty_max_last", 0.0)),
    )
    runner._jungle_recovery_regrowth_boost_max_gen = max(
        float(getattr(runner, "_jungle_recovery_regrowth_boost_max_gen", 0.0)),
        float(getattr(runner.env, "jungle_recovery_regrowth_boost_max_last", 0.0)),
    )


def _update_action_coherence(state: RunnerCore1TickState, runner: Any, agent_ids: np.ndarray, actions: np.ndarray) -> None:
    k0 = _as_int(getattr(runner.sim_cfg, "episode0_k", 20), 20)
    for i, aid_raw in enumerate(agent_ids.tolist()):
        slot = state.slot_for_agent_id(int(aid_raw))
        if slot < 0 or not bool(state.alive[slot]):
            continue
        act = int(actions[i])
        last = int(state.last_action[slot])
        last_last = int(state.last_last_action[slot])
        if last != -1 and act != last:
            state.action_switches[slot] += np.int32(1)
            if int(state.survival_steps[slot]) < k0:
                state.episode0_switches[slot] += np.int32(1)
        if last_last != -1 and act == last_last and act != last:
            state.action_oscillations[slot] += np.int32(1)
            if int(state.survival_steps[slot]) < k0:
                state.episode0_oscillations[slot] += np.int32(1)
        state.last_last_action[slot] = np.int32(last)
        state.last_action[slot] = np.int32(act)


def _apply_env_tick_arrays(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: np.ndarray,
    actions: np.ndarray,
    deaths_this_gen: int,
    gen_peak_injury: float,
    *,
    tick: int,
    buckets: dict[str, float],
) -> tuple[int, float]:
    slots = np.asarray([state.slot_for_agent_id(int(aid)) for aid in agent_ids.tolist()], dtype=np.int32)
    pre_energy = np.asarray(state.energy[slots], dtype=np.float32).copy()
    pre_health = np.asarray(state.health[slots], dtype=np.float32).copy()
    pre_injury = np.asarray(state.injury[slots], dtype=np.float32).copy()
    scars = np.asarray(state.scar[slots], dtype=np.float32).copy()

    t0 = time.perf_counter()
    if _cxxbackbone8r_patch_cpu_action_apply_enabled(runner):
        supported, reason = _cxxbackbone8r_patch_cpu_action_apply_supported(runner)
        if not supported:
            raise RunnerCore1UnsupportedError(f"CXXBACKBONE8R Patch CPU native action apply unsupported: {reason}")
        from env.gridworld_pkg.patch_world import (  # noqa: WPS433
            _emit_value,
            patch_food_gather_amount,
            patch_food_presence_threshold,
            patch_movement_cost_energy_scale,
        )
        from perf.cxxbackbone8r_generation_island import patch_apply_actions_cpu  # noqa: WPS433

        native = patch_apply_actions_cpu(
            agent_ids=np.asarray(agent_ids, dtype=np.int32),
            actions=np.asarray(actions, dtype=np.int32),
            agent_pos_y=np.asarray(getattr(runner.env, "agent_pos_y"), dtype=np.int32),
            agent_pos_x=np.asarray(getattr(runner.env, "agent_pos_x"), dtype=np.int32),
            agent_grid=np.asarray(getattr(runner.env, "agent_grid"), dtype=np.int32),
            food_grid=np.asarray(getattr(runner.env, "grid"), dtype=np.int32),
            patch_food_amount=np.asarray(getattr(runner.env, "patch_food_amount"), dtype=np.float32),
            patch_scent_emit_pending=np.asarray(getattr(runner.env, "patch_scent_emit_pending"), dtype=np.float32),
            patch_waste_emit_pending=np.asarray(getattr(runner.env, "patch_waste_emit_pending"), dtype=np.float32),
            patch_movement_cost=np.asarray(getattr(runner.env, "patch_movement_cost"), dtype=np.float32),
            energies=pre_energy.copy(),
            healths=pre_health.copy(),
            injuries=pre_injury.copy(),
            scars=scars,
            move_cost=float(getattr(runner.env_cfg, "move_cost", 0.0)),
            gather_cost=float(getattr(runner.env_cfg, "gather_cost", 0.0)),
            rest_cost=float(getattr(runner.env_cfg, "rest_cost", 0.0)),
            attack_cost=float(getattr(runner.env_cfg, "attack_cost", 0.0)),
            action_cost_mult=float(getattr(runner.env, "action_cost_mult", 1.0)),
            rest_cost_mult=float(getattr(runner.env, "rest_cost_mult", 1.0)),
            max_energy=float(getattr(runner.env_cfg, "max_energy", 1.0)),
            food_energy=float(getattr(runner.env_cfg, "food_energy", 0.0)),
            patch_food_gather_amount=float(patch_food_gather_amount(runner.env_cfg)),
            patch_food_presence_threshold=float(patch_food_presence_threshold(runner.env_cfg)),
            patch_scent_move_emit=float(_emit_value(runner.env_cfg, "patch_scent_move_emit", 0.02)),
            patch_waste_move_emit=0.0,
            patch_scent_gather_emit=float(_emit_value(runner.env_cfg, "patch_scent_gather_emit", 0.05)),
            patch_waste_consume_emit=float(_emit_value(runner.env_cfg, "patch_waste_consume_emit", 0.05)),
            patch_movement_cost_apply_enable=_as_bool(
                getattr(runner.env_cfg, "patch_movement_cost_apply_enable", False)
            ),
            patch_movement_cost_energy_scale=float(patch_movement_cost_energy_scale(runner.env_cfg)),
            build_directory=str(getattr(runner.sim_cfg, "cxxbackbone3_policy_aten_build_dir", "") or "") or None,
        )
        runner.env.agent_pos_y[:] = np.asarray(native["agent_pos_y"], dtype=np.int32)
        runner.env.agent_pos_x[:] = np.asarray(native["agent_pos_x"], dtype=np.int32)
        runner.env.agent_grid[:, :] = np.asarray(native["agent_grid"], dtype=np.int32)
        runner.env.grid[:, :] = np.asarray(native["food_grid"], dtype=getattr(runner.env.grid, "dtype", np.int8))
        runner.env.patch_food_amount[:, :] = np.asarray(native["patch_food_amount"], dtype=np.float32)
        runner.env.patch_scent_emit_pending[:, :] = np.asarray(
            native["patch_scent_emit_pending"], dtype=np.float32
        )
        runner.env.patch_waste_emit_pending[:, :] = np.asarray(
            native["patch_waste_emit_pending"], dtype=np.float32
        )
        runner.env.patch_food_harvest_total_last = float(np.asarray(native["patch_food_harvest_total"])[0])
        runner.env.patch_movement_energy_extra_last = float(np.asarray(native["patch_movement_energy_extra"])[0])
        runner.env.patch_movement_energy_event_count_last = int(
            np.asarray(native["patch_movement_energy_event_count"])[0]
        )
        try:
            runner.env._views_dirty = True
        except Exception:
            pass
        out = (
            np.asarray(native["died_mask"], dtype=np.bool_),
            np.asarray(native["new_energy"], dtype=np.float32),
            np.asarray(native["new_health"], dtype=np.float32),
            np.asarray(native["food_gained"], dtype=np.float32),
            np.asarray(native["new_injury"], dtype=np.float32),
            np.asarray(native["new_scar"], dtype=np.float32),
        )
        buckets["cxxbackbone8r_native_action_application_calls"] = float(
            buckets.get("cxxbackbone8r_native_action_application_calls", 0.0) + 1.0
        )
        buckets["cxxbackbone8r_native_patch_action_application_calls"] = float(
            buckets.get("cxxbackbone8r_native_patch_action_application_calls", 0.0) + 1.0
        )
        buckets["cxxbackbone8r_native_action_application_rows"] = float(
            buckets.get("cxxbackbone8r_native_action_application_rows", 0.0) + float(np.asarray(agent_ids).shape[0])
        )
        buckets["cxxbackbone8r_science_state_mutated"] = 1.0
    elif _cxxbackbone8r_flat_cpu_action_apply_enabled(runner):
        supported, reason = _cxxbackbone8r_flat_cpu_action_apply_supported(runner)
        if not supported:
            raise RunnerCore1UnsupportedError(f"CXXBACKBONE8R flat CPU native action apply unsupported: {reason}")
        from perf.cxxbackbone8r_generation_island import flat_apply_actions_cpu  # noqa: WPS433

        native = flat_apply_actions_cpu(
            agent_ids=np.asarray(agent_ids, dtype=np.int32),
            actions=np.asarray(actions, dtype=np.int32),
            agent_pos_y=np.asarray(getattr(runner.env, "agent_pos_y"), dtype=np.int32),
            agent_pos_x=np.asarray(getattr(runner.env, "agent_pos_x"), dtype=np.int32),
            agent_grid=np.asarray(getattr(runner.env, "agent_grid"), dtype=np.int32),
            food_grid=np.asarray(getattr(runner.env, "grid"), dtype=np.int32),
            energies=pre_energy.copy(),
            healths=pre_health.copy(),
            injuries=pre_injury.copy(),
            scars=scars,
            move_cost=float(getattr(runner.env_cfg, "move_cost", 0.0)),
            gather_cost=float(getattr(runner.env_cfg, "gather_cost", 0.0)),
            rest_cost=float(getattr(runner.env_cfg, "rest_cost", 0.0)),
            attack_cost=float(getattr(runner.env_cfg, "attack_cost", 0.0)),
            action_cost_mult=float(getattr(runner.env, "action_cost_mult", 1.0)),
            rest_cost_mult=float(getattr(runner.env, "rest_cost_mult", 1.0)),
            max_energy=float(getattr(runner.env_cfg, "max_energy", 1.0)),
            food_energy=float(getattr(runner.env_cfg, "food_energy", 0.0)),
            build_directory=str(getattr(runner.sim_cfg, "cxxbackbone3_policy_aten_build_dir", "") or "") or None,
        )
        runner.env.agent_pos_y[:] = np.asarray(native["agent_pos_y"], dtype=np.int32)
        runner.env.agent_pos_x[:] = np.asarray(native["agent_pos_x"], dtype=np.int32)
        runner.env.agent_grid[:, :] = np.asarray(native["agent_grid"], dtype=np.int32)
        runner.env.grid[:, :] = np.asarray(native["food_grid"], dtype=getattr(runner.env.grid, "dtype", np.int8))
        try:
            runner.env._views_dirty = True
        except Exception:
            pass
        out = (
            np.asarray(native["died_mask"], dtype=np.bool_),
            np.asarray(native["new_energy"], dtype=np.float32),
            np.asarray(native["new_health"], dtype=np.float32),
            np.asarray(native["food_gained"], dtype=np.float32),
            np.asarray(native["new_injury"], dtype=np.float32),
            np.asarray(native["new_scar"], dtype=np.float32),
        )
        buckets["cxxbackbone8r_native_action_application_calls"] = float(
            buckets.get("cxxbackbone8r_native_action_application_calls", 0.0) + 1.0
        )
        buckets["cxxbackbone8r_native_action_application_rows"] = float(
            buckets.get("cxxbackbone8r_native_action_application_rows", 0.0) + float(np.asarray(agent_ids).shape[0])
        )
        buckets["cxxbackbone8r_science_state_mutated"] = 1.0
    else:
        out = runner.env.step_agents_batch(
            agent_ids=np.asarray(agent_ids, dtype=np.int32),
            actions=np.asarray(actions, dtype=np.int32),
            energies=pre_energy.copy(),
            healths=pre_health.copy(),
            injuries=pre_injury.copy(),
            scars=scars,
            move_cost_mults=None,
            juvenile_hazard_mults=None,
            juvenile_harvest_mults=None,
        )
    if isinstance(out, tuple) and len(out) == 6:
        died_mask, new_energy, new_health, food_gained, new_injury, new_scar = out
    else:
        died_mask, new_energy, new_health, food_gained = out
        new_injury, new_scar = pre_injury, scars

    gru_cost = np.asarray(state.gru_upkeep_cost[slots], dtype=np.float32)
    if bool(np.any(gru_cost > np.float32(0.0))):
        new_energy = np.maximum(np.asarray(new_energy, dtype=np.float32) - gru_cost, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
    cost_charge = np.asarray(getattr(state, "cost_charge_scalar", np.zeros((state.capacity,), dtype=np.float32))[slots], dtype=np.float32)
    maturation_charge = np.asarray(
        getattr(state, "cost_charge_maturation_scalar", np.zeros((state.capacity,), dtype=np.float32))[slots],
        dtype=np.float32,
    )
    if bool(np.any(maturation_charge > np.float32(0.0))):
        maturity_age = max(0, _as_int(getattr(runner.sim_cfg, "developmental_maturity_age", 20), 20))
        if int(maturity_age) > 0:
            age_mask = np.asarray(state.survival_steps[slots], dtype=np.int32) < np.int32(maturity_age)
        else:
            age_mask = np.zeros_like(maturation_charge, dtype=np.bool_)
        if bool(np.any(age_mask)):
            cost_charge = cost_charge + np.where(age_mask, maturation_charge, np.float32(0.0)).astype(np.float32, copy=False)
    if bool(np.any(cost_charge > np.float32(0.0))):
        new_energy = np.maximum(np.asarray(new_energy, dtype=np.float32) - cost_charge, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
        buckets["runnercore21_cost_charge_total"] = float(
            buckets.get("runnercore21_cost_charge_total", 0.0) + float(np.sum(cost_charge, dtype=np.float64))
        )

    for attr, env_attr in (
        ("_patch_food_harvest_total_gen", "patch_food_harvest_total_last"),
        ("_patch_movement_energy_extra_total_gen", "patch_movement_energy_extra_last"),
    ):
        if _as_bool(getattr(runner.env, "patch_world_enable", False)):
            setattr(runner, attr, float(getattr(runner, attr, 0.0)) + float(getattr(runner.env, env_attr, 0.0)))
    if _as_bool(getattr(runner.env, "patch_world_enable", False)):
        runner._patch_movement_energy_event_count_gen = int(getattr(runner, "_patch_movement_energy_event_count_gen", 0)) + int(
            getattr(runner.env, "patch_movement_energy_event_count_last", 0)
        )

    early_k = _as_int(getattr(runner.sim_cfg, "early_life_steps", 0), 0)
    early_w = _as_float(getattr(runner.sim_cfg, "early_life_weight", 1.0), 1.0)
    patch_world_enabled = _as_bool(getattr(runner.env, "patch_world_enable", False))
    agents = getattr(getattr(runner, "pop", None), "agents", {})
    env_agent_alive = getattr(runner.env, "agent_alive", None)
    env_pos_y = getattr(runner.env, "agent_pos_y", None)
    env_pos_x = getattr(runner.env, "agent_pos_x", None)
    biome_grid = getattr(runner.env, "biome_grid", None)
    for i, slot_raw in enumerate(slots.tolist()):
        slot = int(slot_raw)
        aid = int(agent_ids[i])
        e1 = float(new_energy[i])
        h1 = float(new_health[i])
        inj1 = float(new_injury[i])
        state.energy[slot] = np.float32(e1)
        state.health[slot] = np.float32(h1)
        state.injury[slot] = np.float32(inj1)
        state.scar[slot] = np.float32(float(new_scar[i]))
        state.last_energy_delta[slot] = np.float32(e1 - float(pre_energy[i]))
        state.last_health_delta[slot] = np.float32(h1 - float(pre_health[i]))
        state.last_injury_delta[slot] = np.float32(inj1 - float(pre_injury[i]))
        if inj1 > gen_peak_injury:
            gen_peak_injury = float(inj1)
        pre_age = int(state.survival_steps[slot])
        state.survival_steps[slot] = np.int32(pre_age + 1)
        fg = float(food_gained[i])
        if fg > 0.0:
            state.energy_collected[slot] += np.float32(fg)
        if early_k > 0 and pre_age < early_k:
            state.early_survival_steps[slot] += np.int32(1)
            if fg > 0.0:
                state.early_energy_collected[slot] += np.float32(fg)
            dinj = max(0.0, inj1 - float(pre_injury[i]))
            if dinj > 0.0:
                state.early_injury_gained[slot] += np.float32(dinj)
        state.early_life_weight[slot] = np.float32(early_w)
        biome_id = 0
        try:
            if env_agent_alive is not None and bool(env_agent_alive[aid]) and biome_grid is not None:
                by = int(env_pos_y[aid])
                bx = int(env_pos_x[aid])
                biome_id = int(biome_grid[by, bx])
        except Exception:
            biome_id = 0
        state.biome_id[slot] = np.int32(int(biome_id))
        if bool(died_mask[i]) and bool(state.alive[slot]):
            agent = agents.get(aid) if isinstance(agents, dict) else None
            if agent is not None:
                try:
                    agent.energy = float(e1)
                    agent.health = float(h1)
                    agent.injury = float(inj1)
                    agent.scar = float(new_scar[i])
                    st = getattr(agent, "stats", None)
                    if st is not None:
                        st.last_energy_delta = float(e1 - float(pre_energy[i]))
                        st.last_health_delta = float(h1 - float(pre_health[i]))
                        st.last_injury_delta = float(inj1 - float(pre_injury[i]))
                        st.biome_id = int(biome_id)
                except Exception:
                    pass
            if patch_world_enabled:
                try:
                    from env.gridworld_pkg.patch_world import sample_patch_context  # noqa: WPS433

                    y = int(env_pos_y[aid])
                    x = int(env_pos_x[aid])
                    if not hasattr(runner, "_patch_death_contexts_gen"):
                        runner._patch_death_contexts_gen = []
                    runner._patch_death_contexts_gen.append(
                        sample_patch_context(
                            runner.env,
                            y,
                            x,
                            agent=agent,
                            lineage_id=(aid if agent is None else None),
                        )
                    )
                except Exception:
                    pass
            state.alive[slot] = False
            if agent is not None:
                try:
                    agent.alive = False
                except Exception:
                    pass
            deaths_this_gen += 1
            state.counters.env_deaths += 1
            state.counters.death_events_applied += 1
            invalidate_action_cache_slot(state, slot, reason="env_death", agent_id=int(agent_ids[i]))
            record_churn_event(
                state,
                event_type="death",
                generation=int(getattr(runner, "gen", 0)),
                tick=int(tick),
                slot=int(slot),
                agent_id=int(agent_ids[i]),
                reason="env_step",
            )
            try:
                runner.env.remove_agent(aid)
            except Exception:
                pass
    buckets["runnercore1_env_apply"] = float(buckets.get("runnercore1_env_apply", 0.0) + (time.perf_counter() - t0))
    return int(deaths_this_gen), float(gen_peak_injury)


def _update_drives(state: RunnerCore1TickState, runner: Any, agent_ids: np.ndarray, actions: np.ndarray, *, tick_index: int) -> None:
    if int(agent_ids.shape[0]) <= 0:
        return
    slots = np.asarray([state.slot_for_agent_id(int(aid)) for aid in agent_ids.tolist()], dtype=np.int32)
    valid = slots >= 0
    if bool(np.any(valid)):
        valid[valid] = state.alive[slots[valid]]
    if not bool(np.any(valid)):
        return
    idx = np.nonzero(valid)[0]
    slots_v = slots[idx]
    e_norm = np.float32(float(getattr(runner.env_cfg, "max_energy", 1.0)) or 1.0)
    h_norm = np.float32(float(getattr(runner.env_cfg, "start_health", 1.0)) or 1.0)
    e_v = np.clip(state.energy[slots_v] / e_norm, 0.0, 1.0).astype(np.float32, copy=False)
    h_v = np.clip(state.health[slots_v] / h_norm, 0.0, 1.0).astype(np.float32, copy=False)
    de_v = state.last_energy_delta[slots_v].astype(np.float32, copy=False)
    dh_v = state.last_health_delta[slots_v].astype(np.float32, copy=False)
    dinj_v = state.last_injury_delta[slots_v].astype(np.float32, copy=False)
    act_v = np.asarray(actions[idx], dtype=np.int32)
    inertia = state.drive_inertia[slots_v].astype(np.float32, copy=False)
    noise = state.drive_noise[slots_v].astype(np.float32, copy=False)
    prev = state.drives[slots_v, :].astype(np.float32, copy=True)

    hunger_gain = np.float32(_as_float(getattr(runner.sim_cfg, "drive_hunger_gain", 0.08), 0.08))
    hunger_decay = np.float32(_as_float(getattr(runner.sim_cfg, "drive_hunger_decay", 0.02), 0.02))
    pain_gain = np.float32(_as_float(getattr(runner.sim_cfg, "drive_pain_gain", 0.18), 0.18))
    pain_decay = np.float32(_as_float(getattr(runner.sim_cfg, "drive_pain_decay", 0.03), 0.03))
    fatigue_gain = np.float32(_as_float(getattr(runner.sim_cfg, "drive_fatigue_gain", 0.05), 0.05))
    fatigue_rest_relief = np.float32(_as_float(getattr(runner.sim_cfg, "drive_fatigue_rest_relief", 0.10), 0.10))
    fatigue_decay = np.float32(_as_float(getattr(runner.sim_cfg, "drive_fatigue_decay", 0.01), 0.01))
    fear_gain = np.float32(_as_float(getattr(runner.sim_cfg, "drive_fear_gain", 0.22), 0.22))
    fear_decay = np.float32(_as_float(getattr(runner.sim_cfg, "drive_fear_decay", 0.03), 0.03))

    tgt_h = prev[:, 0] + hunger_gain * (np.float32(1.0) - e_v)
    tgt_h = tgt_h + np.float32(0.04) * (-de_v / np.float32(max(1e-6, float(e_norm)))) - hunger_decay
    gather_mask = (act_v == np.int32(ACT_GATHER)) & (de_v > np.float32(0.0))
    tgt_h[gather_mask] = tgt_h[gather_mask] - np.float32(0.20) * (de_v[gather_mask] / np.float32(max(1e-6, float(e_norm))))
    np.clip(tgt_h, 0.0, 1.0, out=tgt_h)

    inj_mask = dinj_v > np.float32(0.0)
    tgt_p = prev[:, 1].copy()
    tgt_p[inj_mask] = tgt_p[inj_mask] + pain_gain * np.minimum(np.float32(1.0), dinj_v[inj_mask])
    dh_mask = dh_v < np.float32(0.0)
    tgt_p[dh_mask] = tgt_p[dh_mask] + np.float32(0.10) * np.minimum(
        np.float32(1.0), (-dh_v[dh_mask]) / np.float32(max(1e-6, float(h_norm)))
    )
    tgt_p = tgt_p - pain_decay
    np.clip(tgt_p, 0.0, 1.0, out=tgt_p)

    tgt_f = prev[:, 2].copy()
    rest_mask = act_v == np.int32(ACT_REST)
    tgt_f[rest_mask] = tgt_f[rest_mask] - fatigue_rest_relief
    tgt_f[~rest_mask] = tgt_f[~rest_mask] + fatigue_gain
    tgt_f = tgt_f - fatigue_decay
    np.clip(tgt_f, 0.0, 1.0, out=tgt_f)

    tgt_fr = prev[:, 3].copy()
    tgt_fr[inj_mask] = tgt_fr[inj_mask] + fear_gain * np.minimum(np.float32(1.0), dinj_v[inj_mask])
    tgt_fr = tgt_fr + np.float32(0.05) * (np.float32(1.0) - h_v) - fear_decay
    np.clip(tgt_fr, 0.0, 1.0, out=tgt_fr)

    inv = np.float32(1.0) - inertia
    new = np.empty_like(prev)
    new[:, 0] = inertia * prev[:, 0] + inv * tgt_h
    new[:, 1] = inertia * prev[:, 1] + inv * tgt_p
    new[:, 2] = inertia * prev[:, 2] + inv * tgt_f
    new[:, 3] = inertia * prev[:, 3] + inv * tgt_fr
    if bool(np.any(noise > np.float32(0.0))):
        nvec = None
        advanced_noise = None
        overrides = getattr(runner, "_cxxbackbone10_checkpoint_tensor_overrides", None)
        if isinstance(overrides, dict):
            samples = overrides.get("drive_noise_samples_by_tick_slot")
            if hasattr(samples, "detach"):
                samples = samples.detach().cpu().numpy()
            if samples is not None:
                samples = np.asarray(samples, dtype=np.float32)
                if (
                    samples.ndim == 3
                    and 0 <= int(tick_index) < int(samples.shape[0])
                    and int(samples.shape[2]) >= 4
                    and int(np.max(slots_v, initial=-1)) < int(samples.shape[1])
                ):
                    nvec = np.asarray(samples[int(tick_index), slots_v, :4], dtype=np.float32)
                    valid = overrides.get("drive_noise_samples_valid_by_tick_slot")
                    if hasattr(valid, "detach"):
                        valid = valid.detach().cpu().numpy()
                    if valid is not None:
                        valid_arr = np.asarray(valid, dtype=np.bool_)
                        if (
                            valid_arr.ndim == 2
                            and 0 <= int(tick_index) < int(valid_arr.shape[0])
                            and int(np.max(slots_v, initial=-1)) < int(valid_arr.shape[1])
                        ):
                            advanced_noise = runner.np_rng.normal(0.0, 1.0, size=(int(new.shape[0]), 4)).astype(
                                np.float32,
                                copy=False,
                            )
                            valid_rows = valid_arr[int(tick_index), slots_v].reshape((-1, 1))
                            nvec = np.where(valid_rows, nvec, advanced_noise).astype(np.float32, copy=False)
        if nvec is None:
            nvec = runner.np_rng.normal(0.0, 1.0, size=(int(new.shape[0]), 4)).astype(np.float32, copy=False)
        elif advanced_noise is None:
            try:
                runner.np_rng.normal(0.0, 1.0, size=(int(new.shape[0]), 4))
            except Exception:
                pass
        new += noise.reshape(-1, 1) * nvec
    np.clip(new, 0.0, 1.0, out=new)

    delta = new - prev
    state.drives[slots_v, :] = new
    state.drive_volatility[slots_v] += np.sum(np.abs(delta), axis=1).astype(np.float32)
    jitter = np.sum(np.abs(delta - state.prev_drive_delta[slots_v, :]), axis=1).astype(np.float32)
    state.drive_jitter[slots_v] += jitter
    state.prev_drive_delta[slots_v, :] = delta
    k0 = _as_int(getattr(runner.sim_cfg, "episode0_k", 20), 20)
    early = state.survival_steps[slots_v] <= np.int32(k0)
    if bool(np.any(early)):
        state.episode0_drive_jitter_sum[slots_v[early]] += jitter[early]
    for local_i, slot in enumerate(slots_v.tolist()):
        if bool(early[local_i]):
            if int(act_v[local_i]) == int(ACT_ATTACK) and float(dinj_v[local_i]) > 0.0:
                state.episode0_selfharm_events[int(slot)] += np.int32(1)

    _update_prediction_bookkeeping(state, runner, slots_v, act_v, de_v, dinj_v, tick_index=tick_index)


def _update_prediction_bookkeeping(
    state: RunnerCore1TickState,
    runner: Any,
    slots_v: np.ndarray,
    act_v: np.ndarray,
    de_v: np.ndarray,
    dinj_v: np.ndarray,
    *,
    tick_index: int,
) -> None:
    flags = prediction_ablation_flags(getattr(runner, "sim_cfg", None))
    pred_state_available = not bool(flags.get("ablate", False) or flags.get("force_disabled", False))
    pred_update_allowed = bool(pred_state_available and not flags.get("force_frozen", False))
    pred_horizon = max(0, min(8, _as_int(getattr(runner.sim_cfg, "pred_horizon", 2), 2)))
    inj_scale = _as_float(getattr(runner.sim_cfg, "pred_injury_scale", 1.0), 1.0)
    k0 = _as_int(getattr(runner.sim_cfg, "episode0_k", 20), 20)
    for j, slot_raw in enumerate(slots_v.tolist()):
        slot = int(slot_raw)
        act = int(act_v[j])
        lr = float(state.pred_lr[slot])
        if bool(flags.get("force_zero_lr", False)):
            lr = 0.0
        lr = max(0.0, min(1.0, lr))
        if bool(flags.get("reset_each_tick", False) or not pred_state_available):
            state.exp_dE[slot, :] = 0.0
            state.exp_dInj[slot, :] = 0.0
            state.prediction_reset_count[slot] += np.int32(1)
        pe = 0.0
        if pred_horizon <= 0:
            exp_de = float(state.exp_dE[slot, act])
            exp_dinj = float(state.exp_dInj[slot, act])
            pe = abs(float(de_v[j]) - exp_de) + inj_scale * abs(float(dinj_v[j]) - exp_dinj)
            if pred_state_available:
                state.pred_err_sum[slot] += np.float32(pe)
                state.pred_err_count[slot] += np.int32(1)
                if int(state.survival_steps[slot]) <= k0:
                    state.episode0_pred_err_sum[slot] += np.float32(pe)
                    state.episode0_pred_err_count[slot] += np.int32(1)
            if pred_update_allowed and lr > 0.0:
                new_de = exp_de + lr * (float(de_v[j]) - exp_de)
                new_dinj = exp_dinj + lr * (float(dinj_v[j]) - exp_dinj)
                delta_abs = abs(new_de - exp_de) + abs(new_dinj - exp_dinj)
                state.exp_dE[slot, act] = np.float32(new_de)
                state.exp_dInj[slot, act] = np.float32(new_dinj)
                state.prediction_update_count[slot] += np.int32(1)
                state.prediction_state_delta_abs_sum[slot] += np.float32(delta_abs)
                state.prediction_state_delta_abs_max[slot] = np.float32(
                    max(float(state.prediction_state_delta_abs_max[slot]), delta_abs)
                )
        else:
            pq = state.pred_queue[slot]
            while pq and int(pq[0][0]) <= int(tick_index):
                _due, prev_act, exp_de, exp_dinj, e0, inj0 = pq.popleft()
                actual_de = float(state.energy[slot]) - float(e0)
                actual_dinj = float(state.injury[slot]) - float(inj0)
                pe = abs(actual_de - float(exp_de)) + inj_scale * abs(actual_dinj - float(exp_dinj))
                if pred_state_available:
                    state.pred_err_sum[slot] += np.float32(pe)
                    state.pred_err_count[slot] += np.int32(1)
                    if int(state.survival_steps[slot]) <= k0:
                        state.episode0_pred_err_sum[slot] += np.float32(pe)
                        state.episode0_pred_err_count[slot] += np.int32(1)
                if pred_update_allowed and lr > 0.0:
                    old_de = float(state.exp_dE[slot, int(prev_act)])
                    old_dinj = float(state.exp_dInj[slot, int(prev_act)])
                    new_de = old_de + lr * (actual_de - old_de)
                    new_dinj = old_dinj + lr * (actual_dinj - old_dinj)
                    delta_abs = abs(new_de - old_de) + abs(new_dinj - old_dinj)
                    state.exp_dE[slot, int(prev_act)] = np.float32(new_de)
                    state.exp_dInj[slot, int(prev_act)] = np.float32(new_dinj)
                    state.prediction_update_count[slot] += np.int32(1)
                    state.prediction_state_delta_abs_sum[slot] += np.float32(delta_abs)
                    state.prediction_state_delta_abs_max[slot] = np.float32(
                        max(float(state.prediction_state_delta_abs_max[slot]), delta_abs)
                    )
            if pred_state_available:
                pq.append(
                    (
                        int(tick_index) + int(pred_horizon),
                        int(act),
                        float(state.exp_dE[slot, act]),
                        float(state.exp_dInj[slot, act]),
                        float(state.energy[slot]),
                        float(state.injury[slot]),
                    )
                )
        if int(act) == int(ACT_ATTACK) and float(dinj_v[j]) > 0.0:
            state.selfharm_events[slot] += np.int32(1)
        fr = state.first_repro_step[slot]
        if pred_state_available and fr is not None:
            try:
                if int(state.survival_steps[slot]) > int(fr):
                    state.late_pred_err_sum[slot] += np.float32(pe)
                    state.late_pred_err_count[slot] += np.int32(1)
            except Exception:
                pass


def _update_chemistry(state: RunnerCore1TickState, runner: Any, agent_ids: np.ndarray) -> None:
    n = int(agent_ids.shape[0])
    if n <= 0:
        return
    slots = np.asarray([state.slot_for_agent_id(int(aid)) for aid in agent_ids.tolist()], dtype=np.int32)
    noise = runner.rng_manager.policy_np.normal(0.0, 1.0, size=(n, 2)).astype(np.float32)
    emax = np.float32(float(getattr(runner.env_cfg, "max_energy", 1.0)) or 1.0)
    hmax = np.float32(float(getattr(runner.env_cfg, "start_health", 1.0)) or 1.0)
    for i, slot_raw in enumerate(slots.tolist()):
        slot = int(slot_raw)
        if slot < 0 or not bool(state.alive[slot]):
            continue
        c = state.chemistry[slot]
        energy_n = chem_clamp01(float(state.energy[slot]) / float(emax))
        health_n = chem_clamp01(float(state.health[slot]) / float(hmax))
        injury_n = chem_clamp01(float(state.injury[slot]) / float(hmax))
        stress = chem_clamp01(0.55 * injury_n + 0.30 * float(state.drives[slot, 1]) + 0.15 * (1.0 - energy_n))
        energy_store = float(c[CHEM_SLOT_ENERGY_STORE])
        repair = float(c[CHEM_SLOT_REPAIR_RESERVE])
        acute = float(c[CHEM_SLOT_ACUTE_DAMAGE])
        chronic = float(c[CHEM_SLOT_CHRONIC_DAMAGE])
        reg1 = float(c[CHEM_SLOT_REGULATOR_1])
        reg2 = float(c[CHEM_SLOT_REGULATOR_2])
        acute_target = chem_clamp01(1.25 * injury_n + 0.45 * stress + 0.20 * (1.0 - health_n))
        acute = chem_clamp01(acute + 0.50 * (acute_target - acute) + 0.03 * reg1)
        chronic_target = chem_clamp01(0.75 * chronic + 0.25 * acute)
        chronic = chem_clamp01(chronic + 0.07 * (chronic_target - chronic))
        energy_store = chem_clamp01(energy_store + 0.03 * energy_n - 0.02 * stress - 0.01 * acute)
        repair = chem_clamp01(repair + 0.04 * energy_n - 0.03 * acute - 0.01 * chronic)
        reg1 = chem_clamp01(0.92 * reg1 + 0.08 * (1.0 - acute) + 0.01 * float(noise[i, 0]))
        reg2 = chem_clamp01(0.97 * reg2 + 0.03 * chronic + 0.01 * float(noise[i, 1]))
        acute_resolve = min(acute, 0.10 * repair + 0.04 * energy_store)
        acute = chem_clamp01(acute - acute_resolve)
        chronic = chem_clamp01(chronic + 0.25 * acute - 0.002 * repair)
        c[:] = np.asarray((energy_store, repair, acute, chronic, reg1, reg2), dtype=np.float32)


def _maintenance_opportunity_score(runner: Any, agent_id: int) -> float:
    env = getattr(runner, "env", None)
    if env is None:
        return 0.0
    grid = getattr(env, "grid", None)
    ys = getattr(env, "agent_pos_y", None)
    xs = getattr(env, "agent_pos_x", None)
    if grid is None or ys is None or xs is None:
        return 0.0
    try:
        aid = int(agent_id)
        y = int(ys[aid])
        x = int(xs[aid])
        h, w = int(grid.shape[0]), int(grid.shape[1])
        if y < 0 or x < 0 or y >= h or x >= w:
            return 0.0
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)
        x0 = max(0, x - 1)
        x1 = min(w, x + 2)
        patch = np.asarray(grid[y0:y1, x0:x1])
        if patch.size <= 0:
            return 0.0
        return float(np.count_nonzero(patch == FOOD)) / float(patch.size)
    except Exception:
        return 0.0


def _update_maintenance(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: np.ndarray,
    actions: np.ndarray,
    *,
    tick_index: int,
    buckets: dict[str, float],
) -> None:
    if not _as_bool(getattr(runner.sim_cfg, "maintenance_enable", False)):
        return
    n = int(agent_ids.shape[0])
    if n <= 0:
        return
    valid: list[tuple[int, int, int]] = []
    for i, aid_raw in enumerate(agent_ids.tolist()):
        aid = int(aid_raw)
        slot = state.slot_for_agent_id(aid)
        if slot >= 0 and bool(state.alive[slot]):
            valid.append((int(i), int(aid), int(slot)))
    if not valid:
        return

    delay_ticks = max(0, _as_int(getattr(runner.sim_cfg, "maintenance_delay_ticks", 0), 0))
    base_effect = maint_clamp01(_as_float(getattr(runner.sim_cfg, "maintenance_base_effect", 0.0), 0.0))
    diminishing = max(0.0, _as_float(getattr(runner.sim_cfg, "maintenance_diminishing_factor", 0.0), 0.0))
    decay = maint_clamp01(_as_float(getattr(runner.sim_cfg, "maintenance_recent_decay", 1.0), 1.0))
    energy_cost = max(0.0, _as_float(getattr(runner.sim_cfg, "maintenance_energy_cost", 0.0), 0.0))
    opp_coeff = max(0.0, _as_float(getattr(runner.sim_cfg, "maintenance_opportunity_cost_coeff", 0.0), 0.0))

    rm = getattr(runner, "rng_manager", None)
    schedule_requests = 0
    schedule_energy_cost_sum = 0.0
    schedule_opp_cost_sum = 0.0
    schedule_budget = nullcontext()
    if rm is not None:
        expected = rm.expected_delta_for_boundary(
            "maintenance.tick.schedule",
            ctx={"alive_count": int(n), "maintenance_requests": int(0)},
        )
        schedule_budget = rm.budget_scope("maintenance.tick.schedule", expected_delta=expected)

    with schedule_budget:
        for i, aid, slot in valid:
            m = state.maintenance[slot]
            m[MAINT_SLOT_RECENT_LOAD] = np.float32(maint_clamp01(float(m[MAINT_SLOT_RECENT_LOAD]) * decay))
            if int(actions[i]) != int(ACT_REST):
                continue
            schedule_requests += 1
            opportunity = _maintenance_opportunity_score(runner, aid)
            opp_cost = float(opp_coeff) * float(opportunity)
            total_cost = float(energy_cost) + float(opp_cost)
            if total_cost > 0.0:
                state.energy[slot] = np.float32(max(0.0, float(state.energy[slot]) - float(total_cost)))
                schedule_energy_cost_sum += float(total_cost)
                schedule_opp_cost_sum += float(opp_cost)
                state.maintenance_energy_spent_total[slot] += np.float32(total_cost)
                state.maintenance_opportunity_cost_total[slot] += np.float32(opp_cost)
            m[MAINT_SLOT_PENDING_STRENGTH] = np.float32(
                maint_clamp01(float(m[MAINT_SLOT_PENDING_STRENGTH]) + float(base_effect))
            )
            m[MAINT_SLOT_DUE_TICK] = np.float32(float(int(tick_index) + int(delay_ticks)))
            m[MAINT_SLOT_LAST_SCHEDULE_TICK] = np.float32(float(int(tick_index)))
            state.maintenance_requests[slot] += np.int32(1)

    apply_count = 0
    applied_effect_sum = 0.0
    apply_budget = nullcontext()
    if rm is not None:
        expected = rm.expected_delta_for_boundary(
            "maintenance.tick.apply_delayed",
            ctx={"alive_count": int(n), "maintenance_applications": int(0)},
        )
        apply_budget = rm.budget_scope("maintenance.tick.apply_delayed", expected_delta=expected)

    with apply_budget:
        for _, _, slot in valid:
            m = state.maintenance[slot]
            pending = float(m[MAINT_SLOT_PENDING_STRENGTH])
            due_tick = int(round(float(m[MAINT_SLOT_DUE_TICK])))
            if pending <= 0.0 or due_tick < 0 or int(tick_index) < due_tick:
                continue
            recent_load = float(m[MAINT_SLOT_RECENT_LOAD])
            eff_mult = 1.0 / (1.0 + float(diminishing) * recent_load)
            applied = maint_clamp01(pending * eff_mult)
            c = state.chemistry[slot]
            c[CHEM_SLOT_REPAIR_RESERVE] = np.float32(
                chem_clamp01(float(c[CHEM_SLOT_REPAIR_RESERVE]) + 0.65 * applied)
            )
            c[CHEM_SLOT_ACUTE_DAMAGE] = np.float32(
                chem_clamp01(float(c[CHEM_SLOT_ACUTE_DAMAGE]) - 0.55 * applied)
            )
            c[CHEM_SLOT_CHRONIC_DAMAGE] = np.float32(
                chem_clamp01(float(c[CHEM_SLOT_CHRONIC_DAMAGE]) - 0.25 * applied)
            )
            m[MAINT_SLOT_PENDING_STRENGTH] = np.float32(0.0)
            m[MAINT_SLOT_DUE_TICK] = np.float32(-1.0)
            m[MAINT_SLOT_RECENT_LOAD] = np.float32(maint_clamp01(recent_load + applied))
            apply_count += 1
            applied_effect_sum += float(applied)
            state.maintenance_applied[slot] += np.int32(1)
            state.maintenance_applied_effect_total[slot] += np.float32(applied)

    runner._maintenance_tick_count_gen = int(getattr(runner, "_maintenance_tick_count_gen", 0)) + 1
    runner._maintenance_action_opportunities_gen = int(
        getattr(runner, "_maintenance_action_opportunities_gen", 0)
    ) + int(n)
    runner._maintenance_schedule_requests_gen = int(
        getattr(runner, "_maintenance_schedule_requests_gen", 0)
    ) + int(schedule_requests)
    runner._maintenance_apply_count_gen = int(getattr(runner, "_maintenance_apply_count_gen", 0)) + int(apply_count)
    runner._maintenance_energy_cost_total_gen = float(
        getattr(runner, "_maintenance_energy_cost_total_gen", 0.0)
    ) + float(schedule_energy_cost_sum)
    runner._maintenance_opportunity_cost_total_gen = float(
        getattr(runner, "_maintenance_opportunity_cost_total_gen", 0.0)
    ) + float(schedule_opp_cost_sum)
    runner._maintenance_effect_total_gen = float(
        getattr(runner, "_maintenance_effect_total_gen", 0.0)
    ) + float(applied_effect_sum)

    opportunities = float(max(1, int(getattr(runner, "_maintenance_action_opportunities_gen", 0))))
    ratio = float(getattr(runner, "_maintenance_schedule_requests_gen", 0)) / opportunities
    threshold = _as_float(getattr(runner.sim_cfg, "maintenance_domination_threshold", 0.75), 0.75)
    min_ticks = _as_int(getattr(runner.sim_cfg, "maintenance_domination_min_ticks", 8), 8)
    triggered_now = int(getattr(runner, "_maintenance_tick_count_gen", 0)) >= int(min_ticks) and ratio >= float(threshold)
    prev = dict(getattr(runner, "_maintenance_domination_annotation", {}) or {})
    was_triggered = bool(prev.get("triggered", False))
    runner._maintenance_domination_annotation = {
        "triggered": bool(was_triggered or triggered_now),
        "tick": int(tick_index),
        "ratio": float(ratio),
        "threshold": float(threshold),
        "requests": int(getattr(runner, "_maintenance_schedule_requests_gen", 0)),
        "opportunities": int(getattr(runner, "_maintenance_action_opportunities_gen", 0)),
    }

    buckets["maintenance_schedule_requests"] = buckets.get("maintenance_schedule_requests", 0.0) + float(schedule_requests)
    buckets["maintenance_apply_count"] = buckets.get("maintenance_apply_count", 0.0) + float(apply_count)


def _apply_rest_and_combat(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: np.ndarray,
    actions: np.ndarray,
    deaths_this_gen: int,
    *,
    rest_streaks: dict[int, int],
    injury_recover_mult: float,
) -> int:
    if _as_bool(getattr(runner.env_cfg, "enable_rest_charge", False)):
        free = _as_int(getattr(runner.sim_cfg, "rest_streak_free", 0), 0)
        base = _as_float(getattr(runner.sim_cfg, "rest_charge_energy", 0.0), 0.0)
        grow = _as_float(getattr(runner.sim_cfg, "rest_charge_growth", 0.0), 0.0)
        deficit_to_health = _as_float(getattr(runner.sim_cfg, "rest_deficit_to_health", 1.0), 1.0)
        for i, aid_raw in enumerate(agent_ids.tolist()):
            slot = state.slot_for_agent_id(int(aid_raw))
            if slot < 0 or not bool(state.alive[slot]):
                continue
            if int(actions[i]) == int(ACT_REST):
                state.rest_streak[slot] += np.int32(1)
            else:
                state.rest_streak[slot] = np.int32(0)
            rest_streaks[int(aid_raw)] = int(state.rest_streak[slot])
            if int(state.rest_streak[slot]) > free:
                over = int(state.rest_streak[slot]) - free
                charge = base + grow * float(over - 1)
                state.energy[slot] -= np.float32(charge)
                state.idle_charge_paid[slot] += np.float32(charge)
                if float(state.energy[slot]) < 0.0:
                    state.health[slot] -= np.float32(-float(state.energy[slot]) * deficit_to_health)
                    state.energy[slot] = np.float32(0.0)
                    if float(state.health[slot]) <= 0.0:
                        state.alive[slot] = False
                        deaths_this_gen += 1
                        state.counters.death_events_applied += 1
                        invalidate_action_cache_slot(state, slot, reason="rest_deficit_death", agent_id=int(aid_raw))
                        record_churn_event(
                            state,
                            event_type="death",
                            generation=int(getattr(runner, "gen", 0)),
                            tick=int(getattr(runner, "tick", 0)),
                            slot=int(slot),
                            agent_id=int(aid_raw),
                            reason="rest_deficit",
                        )
                        try:
                            runner.env.remove_agent(int(aid_raw))
                        except Exception:
                            pass
        if _as_bool(getattr(runner.env_cfg, "enable_injury", False)):
            recover = _as_float(
                getattr(runner.env_cfg, "injury_recover_per_rest_step", getattr(runner.env_cfg, "injury_recover_per_rest", 0.0)),
                0.0,
            ) * float(injury_recover_mult)
            if recover > 0.0:
                rest_slots = [
                    state.slot_for_agent_id(int(aid))
                    for aid in agent_ids.tolist()
                    if int(actions[list(agent_ids).index(aid)]) == int(ACT_REST)
                ]
                for slot in rest_slots:
                    if slot >= 0 and bool(state.alive[slot]) and float(state.injury[slot]) > 0.0:
                        state.injury[slot] = np.float32(max(0.0, float(state.injury[slot]) - recover))
    if _as_bool(getattr(runner.env_cfg, "enable_combat", False)):
        for i, aid_raw in enumerate(agent_ids.tolist()):
            slot = state.slot_for_agent_id(int(aid_raw))
            if slot >= 0 and bool(state.alive[slot]) and int(actions[i]) == int(ACT_ATTACK):
                state.attacks[slot] += np.int32(1)
    return int(deaths_this_gen)


def _decrement_repro_cooldowns(state: RunnerCore1TickState, agent_ids: np.ndarray) -> None:
    for aid_raw in agent_ids.tolist():
        slot = state.slot_for_agent_id(int(aid_raw))
        if slot >= 0 and bool(state.alive[slot]) and int(state.repro_cooldown[slot]) > 0:
            state.repro_cooldown[slot] -= np.int32(1)


def _runnercore2_birth_boundary_needed(state: RunnerCore1TickState, runner: Any, agent_ids: np.ndarray) -> bool:
    if _as_float(getattr(runner.env_cfg, "reproduce_try_prob", 0.0), 0.0) <= 0.0:
        return False
    for aid_raw in sorted(int(aid) for aid in agent_ids.tolist()):
        slot = state.slot_for_agent_id(int(aid_raw))
        if slot < 0 or not bool(state.alive[slot]):
            continue
        cooldown_after_legacy_decrement = max(0, int(state.repro_cooldown[slot]) - 1)
        if cooldown_after_legacy_decrement != 0:
            continue
        if float(state.energy[slot]) >= float(state.repro_threshold[slot]):
            return True
    return False


def _runnercore2_ensure_birth_counters(runner: Any) -> None:
    for name, value in (
        ("_injury_blocked_births_gen", 0),
        ("_birth_attempts_gen", 0),
        ("_birth_pre_cooldown_gen", 0),
        ("_birth_pre_no_energy_gen", 0),
        ("_birth_pre_injury_gate_gen", 0),
        ("_birth_pre_no_space_gen", 0),
        ("_birth_pre_no_mate_gen", 0),
        ("_birth_attempt_conception_fail_gen", 0),
        ("_birth_gestation_fail_gen", 0),
        ("_birth_make_child_fail_gen", 0),
        ("_birth_fail_no_space_gen", 0),
        ("_birth_fail_no_mate_gen", 0),
        ("_birth_fail_buffer_gen", 0),
        ("_birth_fail_injury_gate_gen", 0),
        ("_birth_fail_scar_gate_gen", 0),
        ("_birth_fail_conception_gen", 0),
        ("_birth_fail_gestation_gen", 0),
    ):
        if not hasattr(runner, name):
            setattr(runner, name, value)


def _runnercore22_parent_import_snapshot(agent: Any) -> tuple[Any, ...]:
    st = getattr(agent, "stats", None)
    return (
        bool(getattr(agent, "alive", False)),
        float(getattr(agent, "energy", 0.0)),
        float(getattr(agent, "health", 0.0)),
        float(getattr(agent, "injury", 0.0)),
        float(getattr(agent, "scar", 0.0)),
        int(getattr(agent, "rest_streak", 0)),
        int(getattr(agent, "repro_cooldown", 0)),
        int(getattr(st, "survival_steps", 0) if st is not None else 0),
        float(getattr(st, "energy_collected", 0.0) if st is not None else 0.0),
        int(getattr(st, "early_survival_steps", 0) if st is not None else 0),
        float(getattr(st, "early_energy_collected", 0.0) if st is not None else 0.0),
        float(getattr(st, "early_injury_gained", 0.0) if st is not None else 0.0),
        float(getattr(st, "last_energy_delta", 0.0) if st is not None else 0.0),
        float(getattr(st, "last_health_delta", 0.0) if st is not None else 0.0),
        float(getattr(st, "last_injury_delta", 0.0) if st is not None else 0.0),
        int(getattr(st, "last_action", -1) if st is not None else -1),
        int(getattr(st, "last_last_action", -1) if st is not None else -1),
        int(getattr(st, "action_switches", 0) if st is not None else 0),
        int(getattr(st, "action_oscillations", 0) if st is not None else 0),
        int(getattr(st, "episode0_switches", 0) if st is not None else 0),
        int(getattr(st, "episode0_oscillations", 0) if st is not None else 0),
        int(getattr(st, "attacks", 0) if st is not None else 0),
        float(getattr(st, "idle_charge_paid", 0.0) if st is not None else 0.0),
        int(getattr(st, "maintenance_requests", 0) if st is not None else 0),
        int(getattr(st, "maintenance_applied", 0) if st is not None else 0),
        float(getattr(st, "maintenance_energy_spent_total", 0.0) if st is not None else 0.0),
        float(getattr(st, "maintenance_opportunity_cost_total", 0.0) if st is not None else 0.0),
        float(getattr(st, "maintenance_applied_effect_total", 0.0) if st is not None else 0.0),
        getattr(st, "first_repro_step", None) if st is not None else None,
        int(getattr(st, "offspring_count", 0) if st is not None else 0),
    )


def _runnercore22_only_repro_cooldown_changed(before: tuple[Any, ...], after: tuple[Any, ...]) -> bool:
    if len(before) != len(after) or len(before) <= 6:
        return False
    if before[6] == after[6]:
        return False
    return before[:6] == after[:6] and before[7:] == after[7:]


def _runnercore2_apply_no_boundary_birth_prechecks(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: np.ndarray,
) -> None:
    _runnercore2_ensure_birth_counters(runner)
    considered = 0
    for aid_raw in sorted(int(aid) for aid in agent_ids.tolist()):
        slot = state.slot_for_agent_id(int(aid_raw))
        if slot < 0 or not bool(state.alive[slot]):
            continue
        considered += 1
        if int(state.repro_cooldown[slot]) > 0:
            state.repro_cooldown[slot] -= np.int32(1)
        if int(state.repro_cooldown[slot]) != 0:
            runner._birth_pre_cooldown_gen = int(getattr(runner, "_birth_pre_cooldown_gen", 0)) + 1
            continue
        if float(state.energy[slot]) < float(state.repro_threshold[slot]):
            runner._birth_pre_no_energy_gen = int(getattr(runner, "_birth_pre_no_energy_gen", 0)) + 1
    if _runnercore3_profile_enabled(runner):
        runnercore3_profile_count(state, "birth_candidates_considered", considered)


def _runnercore2_birth_candidate_ids(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: np.ndarray,
) -> list[int]:
    _runnercore2_ensure_birth_counters(runner)
    if _as_float(getattr(runner.env_cfg, "reproduce_try_prob", 0.0), 0.0) <= 0.0:
        _runnercore2_apply_no_boundary_birth_prechecks(state, runner, agent_ids)
        return []
    candidates: list[int] = []
    considered = 0
    for aid_raw in sorted(int(aid) for aid in agent_ids.tolist()):
        slot = state.slot_for_agent_id(int(aid_raw))
        if slot < 0 or not bool(state.alive[slot]):
            continue
        considered += 1
        cooldown = int(state.repro_cooldown[slot])
        cooldown_after_legacy_decrement = max(0, cooldown - 1)
        if cooldown_after_legacy_decrement != 0:
            state.repro_cooldown[slot] = np.int32(cooldown_after_legacy_decrement)
            runner._birth_pre_cooldown_gen = int(getattr(runner, "_birth_pre_cooldown_gen", 0)) + 1
            continue
        if float(state.energy[slot]) < float(state.repro_threshold[slot]):
            state.repro_cooldown[slot] = np.int32(cooldown_after_legacy_decrement)
            runner._birth_pre_no_energy_gen = int(getattr(runner, "_birth_pre_no_energy_gen", 0)) + 1
            continue
        candidates.append(int(aid_raw))
    if _runnercore3_profile_enabled(runner):
        runnercore3_profile_count(state, "birth_candidates_considered", considered)
    return candidates


def _runnercore30_refresh_birth_parent_from_state(
    state: RunnerCore1TickState,
    runner: Any,
    aid: int,
) -> tuple[Any, int] | None:
    slot = state.slot_for_agent_id(int(aid))
    if slot < 0 or not bool(state.alive[slot]):
        return None
    agents = getattr(getattr(runner, "pop", None), "agents", {})
    agent = agents.get(int(aid)) if isinstance(agents, dict) else None
    if agent is None:
        return None
    try:
        agent.alive = bool(state.alive[slot])
        agent.energy = float(state.energy[slot])
        agent.health = float(state.health[slot])
        agent.injury = float(state.injury[slot])
        agent.scar = float(state.scar[slot])
        agent.repro_cooldown = int(state.repro_cooldown[slot])
        agent.rest_streak = int(state.rest_streak[slot])
        agent.drives[:] = state.drives[slot]
        agent.chemistry = chemistry_from_values(state.chemistry[slot])
    except Exception:
        return None
    st = getattr(agent, "stats", None)
    if st is not None:
        try:
            st.survival_steps = int(state.survival_steps[slot])
            st.energy_collected = float(state.energy_collected[slot])
            st.first_repro_step = state.first_repro_step[slot]
            st.offspring_count = int(state.offspring_count[slot])
        except Exception:
            return None
    return agent, int(slot)


def _runnercore30_reproduction_threshold(runner: Any, agent: Any) -> float:
    scalars_fn = getattr(runner, "resolve_developmental_effective_scalars_for_agent", None)
    scalars = scalars_fn(agent) if callable(scalars_fn) else None
    if scalars is not None:
        try:
            return float(scalars[6])
        except Exception:
            pass
    expr_fn = getattr(runner, "resolve_developmental_expression_for_agent", None)
    dev = expr_fn(agent) if callable(expr_fn) else {}
    eff = dev.get("effective", {}) if isinstance(dev, dict) else {}
    genome = getattr(agent, "genome", None)
    return float(eff.get("reproduction_threshold", getattr(genome, "reproduction_threshold", 50.0)))


def _runnercore30_full_birth_branch_prefilter(
    state: RunnerCore1TickState,
    runner: Any,
    candidate_ids: list[int],
) -> list[int]:
    """Own no-RNG full-birth failure branches before the legacy W6 birth scan."""

    if not candidate_ids:
        return []
    if not _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore30_full_birth_branch_prefilter_enable", False)):
        return list(candidate_ids)
    if _as_bool(getattr(getattr(runner, "sim_cfg", None), "lineage_reproduction_funnel_write_events", False)):
        raise RunnerCore1UnsupportedError(
            "RUNNERCORE30-UNSUPPORTED: branch prefilter cannot skip funnel event contexts"
        )
    if not _as_bool(getattr(getattr(runner, "env_cfg", None), "reproduce_requires_mate", False)):
        return list(candidate_ids)

    injury_gate = None
    if _as_bool(getattr(runner.env_cfg, "enable_injury", False)):
        ig = getattr(runner.env_cfg, "injury_repro_block", None)
        if ig is not None:
            injury_gate = float(ig)
    scar_gate = None
    if _as_bool(getattr(runner.env_cfg, "enable_scar", False)):
        sg = getattr(runner.env_cfg, "scar_repro_block", None)
        if sg is not None:
            scar_gate = float(sg)

    chem_repro_enable = bool(getattr(runner.sim_cfg, "chemistry_functional_coupling_enable", False)) and bool(
        getattr(runner.sim_cfg, "chemistry_repro_viability_enable", False)
    )
    chem_floor = float(getattr(runner.sim_cfg, "chemistry_repro_viability_floor", 0.2))
    chem_strength = float(getattr(runner.sim_cfg, "chemistry_repro_viability_strength", 0.8))
    chem_chronic_block = float(getattr(runner.sim_cfg, "chemistry_repro_chronic_block_threshold", 0.95))

    filtered: list[int] = []
    skipped = 0
    self_gate_rows = 0
    isolated_no_mate_rows = 0
    refresh_rows = 0
    for aid in sorted(int(aid) for aid in candidate_ids):
        try:
            adj = runner.env.adjacent_agent_ids(int(aid))
        except Exception:
            filtered.append(int(aid))
            continue
        if adj:
            filtered.append(int(aid))
            continue

        refreshed = _runnercore30_refresh_birth_parent_from_state(state, runner, int(aid))
        if refreshed is None:
            filtered.append(int(aid))
            continue
        agent, slot = refreshed
        refresh_rows += 1

        cooldown_after = max(0, int(state.repro_cooldown[slot]) - 1)
        if int(state.repro_cooldown[slot]) > 0:
            state.repro_cooldown[slot] = np.int32(int(cooldown_after))
            agent.repro_cooldown = int(cooldown_after)
        if int(cooldown_after) != 0:
            runner._birth_pre_cooldown_gen = int(getattr(runner, "_birth_pre_cooldown_gen", 0)) + 1
            skipped += 1
            self_gate_rows += 1
            continue

        threshold = _runnercore30_reproduction_threshold(runner, agent)
        if float(getattr(agent, "energy", 0.0)) < float(threshold):
            runner._birth_pre_no_energy_gen = int(getattr(runner, "_birth_pre_no_energy_gen", 0)) + 1
            skipped += 1
            self_gate_rows += 1
            continue

        inj = float(getattr(agent, "injury", 0.0))
        scar = float(getattr(agent, "scar", 0.0))
        if injury_gate is not None and inj > injury_gate:
            runner._injury_blocked_births_gen = int(getattr(runner, "_injury_blocked_births_gen", 0)) + 1
            runner._birth_pre_injury_gate_gen = int(getattr(runner, "_birth_pre_injury_gate_gen", 0)) + 1
            runner._birth_precheck_injury_gate_gen = int(getattr(runner, "_birth_precheck_injury_gate_gen", 0)) + 1
            runner._birth_fail_injury_gate_gen = int(getattr(runner, "_birth_fail_injury_gate_gen", 0)) + 1
            skipped += 1
            self_gate_rows += 1
            continue
        if scar_gate is not None and scar > scar_gate:
            runner._birth_precheck_scar_gate_gen = int(getattr(runner, "_birth_precheck_scar_gate_gen", 0)) + 1
            runner._birth_fail_scar_gate_gen = int(getattr(runner, "_birth_fail_scar_gate_gen", 0)) + 1
            skipped += 1
            self_gate_rows += 1
            continue

        if chem_repro_enable:
            cself_raw = getattr(agent, "chemistry", None)
            cself = chemistry_from_values(cself_raw if cself_raw is not None else (0.0,) * 6)
            viability_self = reproduction_viability_scale_from_chem_row(cself, floor=chem_floor, strength=chem_strength)
            if float(cself[CHEM_SLOT_CHRONIC_DAMAGE]) >= chem_chronic_block:
                runner._birth_fail_injury_gate_gen = int(getattr(runner, "_birth_fail_injury_gate_gen", 0)) + 1
                skipped += 1
                self_gate_rows += 1
                continue
            try:
                agent.stats.chemistry_repro_viability = float(viability_self)
            except Exception:
                pass

        runner._birth_pre_no_mate_gen = int(getattr(runner, "_birth_pre_no_mate_gen", 0)) + 1
        runner._birth_precheck_no_mate_gen = int(getattr(runner, "_birth_precheck_no_mate_gen", 0)) + 1
        runner._birth_fail_no_mate_gen = int(getattr(runner, "_birth_fail_no_mate_gen", 0)) + 1
        skipped += 1
        isolated_no_mate_rows += 1

    if _runnercore3_profile_enabled(runner):
        runnercore3_profile_count(state, "runnercore30_full_birth_branch_prefilter_rows", skipped)
        runnercore3_profile_count(state, "runnercore30_full_birth_branch_prefilter_self_gate_rows", self_gate_rows)
        runnercore3_profile_count(
            state,
            "runnercore30_full_birth_branch_prefilter_isolated_no_mate_rows",
            isolated_no_mate_rows,
        )
        runnercore3_profile_count(state, "runnercore30_full_birth_branch_prefilter_parent_refresh_rows", refresh_rows)
    return filtered


def _runnercore3_handle_no_mate_births_fast_boundary(
    runner: Any,
    state: RunnerCore1TickState | None,
    alive: list[Any],
    successful_births: int,
    failed_births: int,
    buckets: dict[str, float],
) -> tuple[int, int]:
    """Supported RUNNERCORE-3 no-mate birth boundary with canonical child creation."""

    if _as_bool(getattr(runner.env_cfg, "reproduce_requires_mate", False)):
        raise RunnerCore1UnsupportedError("RUNNERCORE3-C/FAST-BIRTH-BOUNDARY-UNSUPPORTED: mate reproduction requested")
    if _as_bool(getattr(getattr(runner, "sim_cfg", None), "lineage_reproduction_funnel_reporting_enable", False)):
        raise RunnerCore1UnsupportedError("RUNNERCORE3-C/FAST-BIRTH-BOUNDARY-UNSUPPORTED: lineage funnel requested")
    if _as_bool(getattr(getattr(runner, "env", None), "patch_world_enable", False)):
        raise RunnerCore1UnsupportedError("RUNNERCORE3-C/FAST-BIRTH-BOUNDARY-UNSUPPORTED: patch world requested")

    t_repro0 = time.perf_counter()
    t_birth_scan0 = time.perf_counter()
    t_make_child_sum = 0.0
    births: list[tuple[Any, tuple[int, int], Any]] = []
    reserved_tiles: set[tuple[int, int]] = set()
    changed_parent_ids: set[int] = set()
    r18_full_refresh_skipped_no_slot = 0
    r6_resident_attach = _runnercore6_resident_attach_enabled(runner)
    r9_lazy_newborn = bool(
        r6_resident_attach
        and _runnercore9_requested(runner)
        and _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore9_lazy_newborn_materialization_enable", False))
    )
    r11_typed_rows = bool(
        r6_resident_attach
        and _runnercore11_requested(runner)
        and _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore11_c1_typed_rows_enable", False))
    )
    r18_minimal_parent_refresh = bool(_runnercore18_resident_parent_minimal_refresh_enabled(runner))
    if r18_minimal_parent_refresh:
        _runnercore18_stat_add(runner, "runnercore18_resident_parent_minimal_refresh_enabled", 1)
    r10_typed_child = bool(
        r6_resident_attach
        and (
            (
                _runnercore10_requested(runner)
                and _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore10_typed_child_contract_enable", False))
                and _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore10_defer_direct_attached_compat_enable", False))
            )
            or r11_typed_rows
        )
    )
    if r6_resident_attach and state is None:
        raise RunnerCore1UnsupportedError("RUNNERCORE6-UNSUPPORTED: resident child-row attach requires runner state")
    if r6_resident_attach:
        try:
            runner._runnercore6_resident_attached_newborn_ids = []
        except Exception:
            pass

    def _record_birth_event(
        *,
        parent: Any,
        event_type: str,
        child: Any | None = None,
        failure_reason: str = "",
    ) -> None:
        contexts = getattr(runner, "_lineage_reproduction_funnel_event_contexts_gen", None)
        if not isinstance(contexts, list):
            contexts = []
            runner._lineage_reproduction_funnel_event_contexts_gen = contexts
        parent_id = int(getattr(parent, "agent_id", -1))
        child_id = int(getattr(child, "agent_id", -1)) if child is not None else -1
        stats = getattr(child, "stats", None) if child is not None else getattr(parent, "stats", None)
        contexts.append(
            {
                "generation": int(getattr(getattr(runner, "pop", None), "_eiel_current_generation_index", 0) or 0),
                "tick": int(getattr(runner, "_tick", getattr(getattr(runner, "pop", None), "_eiel_current_tick_index", 0)) or 0),
                "event_type": str(event_type),
                "parent_id": parent_id,
                "child_id": child_id,
                "lineage_id": int(getattr(stats, "founder_id", parent_id) if stats is not None else parent_id),
                "lineage_depth": int(getattr(stats, "lineage_depth", -1) if stats is not None else -1),
                "birth_success": int(str(event_type) in {"birth_success", "successful_birth"}),
                "failure_reason": str(failure_reason),
            }
        )

    injury_gate = None
    if _as_bool(getattr(runner.env_cfg, "enable_injury", False)):
        ig = getattr(runner.env_cfg, "injury_repro_block", None)
        if ig is not None:
            injury_gate = float(ig)
    scar_gate = None
    if _as_bool(getattr(runner.env_cfg, "enable_scar", False)):
        sg = getattr(runner.env_cfg, "scar_repro_block", None)
        if sg is not None:
            scar_gate = float(sg)

    k_inj = float(getattr(runner.env_cfg, "injury_repro_k_injury", 0.0))
    k_scar = float(getattr(runner.env_cfg, "injury_repro_k_scar", 0.0))
    p_fail_scar = float(getattr(runner.env_cfg, "injury_birth_fail_scar", 0.0))
    p_fail_inj = float(getattr(runner.env_cfg, "injury_birth_fail_injury", 0.0))
    share = float(runner.env_cfg.reproduce_parent_energy_share)
    buffer = float(runner.env_cfg.reproduce_min_energy_buffer)
    cd = int(runner.env_cfg.reproduce_cooldown_steps)
    try_prob = max(0.0, min(1.0, float(getattr(runner.env_cfg, "reproduce_try_prob", 1.0))))

    for name in (
        "_birth_pre_no_energy_gen",
        "_birth_pre_no_space_gen",
        "_birth_attempt_conception_fail_gen",
        "_birth_make_child_fail_gen",
        "_injury_blocked_births_gen",
        "_birth_pre_injury_gate_gen",
        "_birth_fail_injury_gate_gen",
        "_birth_fail_scar_gate_gen",
        "_birth_fail_conception_gen",
        "_birth_precheck_no_space_gen",
        "_birth_precheck_no_mate_gen",
        "_birth_precheck_scar_gate_gen",
        "_birth_precheck_injury_gate_gen",
    ):
        setattr(runner, name, int(getattr(runner, name, 0)))

    for agent in alive:
        if not _as_bool(getattr(agent, "alive", False)):
            continue
        if _as_int(getattr(agent, "repro_cooldown", 0), 0) > 0:
            agent.repro_cooldown -= 1
            changed_parent_ids.add(int(agent.agent_id))
        if _as_int(getattr(agent, "repro_cooldown", 0), 0) != 0:
            runner._birth_pre_cooldown_gen = int(getattr(runner, "_birth_pre_cooldown_gen", 0)) + 1
            continue

        threshold_fastpath = _as_bool(
            getattr(getattr(runner, "sim_cfg", None), "runnercore7_static_reproduction_threshold_fastpath_enable", False)
        )
        threshold = None
        if threshold_fastpath:
            threshold = _runnercore7_static_reproduction_threshold(agent)
            if threshold is not None:
                _runnercore5_stat_add(runner, "runnercore7_static_reproduction_threshold_fastpath_calls", 1)
        if threshold is None and hasattr(runner, "resolve_developmental_expression_for_agent"):
            dev = runner.resolve_developmental_expression_for_agent(agent)
            eff = dev.get("effective", {}) if isinstance(dev, dict) else {}
            threshold = float(eff.get("reproduction_threshold", getattr(agent.genome, "reproduction_threshold", 0.0)))
        elif threshold is None:
            dev = {}
            threshold = float(getattr(agent.genome, "reproduction_threshold", 0.0))
        if float(getattr(agent, "energy", 0.0)) < threshold:
            runner._birth_pre_no_energy_gen = int(getattr(runner, "_birth_pre_no_energy_gen", 0)) + 1
            continue

        inj = float(getattr(agent, "injury", 0.0))
        scar = float(getattr(agent, "scar", 0.0))
        if injury_gate is not None and inj > injury_gate:
            runner._injury_blocked_births_gen = int(getattr(runner, "_injury_blocked_births_gen", 0)) + 1
            runner._birth_pre_injury_gate_gen = int(getattr(runner, "_birth_pre_injury_gate_gen", 0)) + 1
            runner._birth_precheck_injury_gate_gen = int(getattr(runner, "_birth_precheck_injury_gate_gen", 0)) + 1
            runner._birth_fail_injury_gate_gen = int(getattr(runner, "_birth_fail_injury_gate_gen", 0)) + 1
            _record_birth_event(parent=agent, event_type="failed_birth", failure_reason="birth_fail_injury_gate")
            continue
        if scar_gate is not None and scar > scar_gate:
            runner._birth_precheck_scar_gate_gen = int(getattr(runner, "_birth_precheck_scar_gate_gen", 0)) + 1
            runner._birth_fail_scar_gate_gen = int(getattr(runner, "_birth_fail_scar_gate_gen", 0)) + 1
            _record_birth_event(parent=agent, event_type="failed_birth", failure_reason="birth_fail_scar_gate")
            continue

        pos = runner.env.find_adjacent_empty(int(agent.agent_id))
        candidates = [pos] if pos is not None and pos not in reserved_tiles else []
        spawn_pos = None
        for p in candidates:
            if p not in reserved_tiles:
                spawn_pos = p
                break
        if spawn_pos is None:
            runner._birth_pre_no_space_gen = int(getattr(runner, "_birth_pre_no_space_gen", 0)) + 1
            runner._birth_precheck_no_space_gen = int(getattr(runner, "_birth_precheck_no_space_gen", 0)) + 1
            runner._birth_fail_no_space_gen = int(getattr(runner, "_birth_fail_no_space_gen", 0)) + 1
            _record_birth_event(parent=agent, event_type="failed_birth", failure_reason="birth_fail_no_space")
            continue

        if try_prob < 1.0 and float(runner.np_rng.random()) > try_prob:
            continue

        if r18_minimal_parent_refresh and bool(getattr(agent, "_runnercore18_minimal_parent_refreshed", False)):
            if _runnercore18_population_has_free_slot(runner):
                if not _runnercore17_refresh_resident_parent_spec_from_state(
                    state,
                    runner,
                    agent,
                    full=True,
                    count_r17_refresh=False,
                ):
                    _runnercore18_stat_add(runner, "runnercore18_resident_parent_unsupported_fallback_count", 1)
                    if _runnercore17_resident_birth_parent_fail_closed(runner):
                        raise RunnerCore1UnsupportedError(
                            "RUNNERCORE18-UNSUPPORTED: full resident parent refresh failed before child creation"
                        )
            else:
                r18_full_refresh_skipped_no_slot += 1

        give1 = float(agent.energy) * share
        if float(agent.energy) - give1 < buffer:
            failed_births += 1
            runner._birth_fail_buffer_gen = int(getattr(runner, "_birth_fail_buffer_gen", 0)) + 1
            _record_birth_event(parent=agent, event_type="failed_birth", failure_reason="birth_fail_buffer")
            continue

        runner._birth_attempts_gen = int(getattr(runner, "_birth_attempts_gen", 0)) + 1
        if (k_inj > 0.0) or (k_scar > 0.0):
            p1 = float(np.exp(-(k_inj * inj + k_scar * scar)))
            if float(runner.np_rng.random()) > p1:
                failed_births += 1
                runner._birth_attempt_conception_fail_gen = int(getattr(runner, "_birth_attempt_conception_fail_gen", 0)) + 1
                runner._birth_fail_conception_gen = int(getattr(runner, "_birth_fail_conception_gen", 0)) + 1
                _record_birth_event(parent=agent, event_type="failed_birth", failure_reason="birth_fail_conception")
                continue

        reserved_tiles.add(spawn_pos)
        agent.energy -= give1
        child_energy = give1
        if (p_fail_scar > 0.0) or (p_fail_inj > 0.0):
            p_fail = min(0.95, p_fail_scar * scar + p_fail_inj * inj)
            if float(runner.np_rng.random()) < p_fail:
                agent.energy += give1
                failed_births += 1
                runner._birth_gestation_fail_gen = int(getattr(runner, "_birth_gestation_fail_gen", 0)) + 1
                runner._birth_fail_gestation_gen = int(getattr(runner, "_birth_fail_gestation_gen", 0)) + 1
                _record_birth_event(parent=agent, event_type="failed_birth", failure_reason="birth_fail_gestation")
                reserved_tiles.discard(spawn_pos)
                continue

        t_mc = time.perf_counter()
        if r6_resident_attach:
            _runnercore6_stat_add(runner, "runnercore6_pre_attach_agent_materializations", 0)
            _runnercore6_stat_add(runner, "runnercore6_pre_attach_genome_materializations", 1)
            _runnercore6_stat_add(runner, "runnercore6_pre_attach_policy_materializations", 1)
            child = runner.pop.make_child_row_spec(agent, agent, child_energy, buckets=buckets)
            _runnercore6_stat_add(runner, "runnercore6_child_row_specs_created", 1)
            _runnercore6_stat_add(runner, "runnercore6_agent_materializations_deferred", 1)
            if child is not None and r10_typed_child:
                _runnercore10_stat_add(runner, "typed_child_specs_created", 1)
        else:
            child = runner.pop.make_child(agent, agent, child_energy, buckets=buckets)
        t_make_child_sum += time.perf_counter() - t_mc
        if child is None:
            agent.energy += give1
            failed_births += 1
            if r11_typed_rows:
                _runnercore11_stat_add(runner, "birth_failure_count", 1)
            runner._birth_make_child_fail_gen = int(getattr(runner, "_birth_make_child_fail_gen", 0)) + 1
            _record_birth_event(parent=agent, event_type="failed_birth", failure_reason="birth_fail_buffer")
            reserved_tiles.discard(spawn_pos)
            continue

        child.injury = 0.0
        child.scar = 0.0
        child.rest_streak = 0
        births.append((child, spawn_pos, agent))
        agent.stats.offspring_count += 1
        try:
            st = agent.stats
            if float(getattr(st, "offspring_count", 0.0)) > 0.0 and getattr(st, "first_repro_step", None) is None:
                st.first_repro_step = int(getattr(st, "survival_steps", 0))
        except Exception:
            pass
        agent.repro_cooldown = cd
        changed_parent_ids.add(int(agent.agent_id))

    if r18_full_refresh_skipped_no_slot:
        _runnercore18_stat_add(
            runner,
            "runnercore18_resident_parent_full_refresh_skipped_no_slot",
            r18_full_refresh_skipped_no_slot,
        )

    if isinstance(buckets, dict):
        loop_dt = time.perf_counter() - t_birth_scan0
        buckets["birth_scan_ms"] = buckets.get("birth_scan_ms", 0.0) + loop_dt
        buckets["birth_make_child_ms"] = buckets.get("birth_make_child_ms", 0.0) + t_make_child_sum
        buckets["birth_scan_excl_make_child_ms"] = buckets.get("birth_scan_excl_make_child_ms", 0.0) + max(
            0.0,
            loop_dt - t_make_child_sum,
        )

    t_birth0 = time.perf_counter()
    newborns: list[Any] = []
    newborn_birth_rows: list[tuple[Any, tuple[int, int], Any]] = []
    r6_successful_birth_rows: list[tuple[Any, tuple[int, int], Any]] = []
    r6_attached_mlp_ids: set[int] = set()
    r7_attached_gru_ids: set[int] = set()
    for child, pos, parent in births:
        try:
            runner.env.add_agent(int(child.agent_id), pos=pos)
            successful_births += 1
            _record_birth_event(parent=parent, event_type="birth_success", child=child)
            if r11_typed_rows:
                _runnercore11_stat_add(runner, "birth_success_count", 1)
            if r6_resident_attach:
                r6_successful_birth_rows.append((child, pos, parent))
                continue
            newborns.append(child)
            if not r6_resident_attach:
                newborn_birth_rows.append((child, pos, parent))
            try:
                runner.behavior_logger.register_agent(
                    int(child.agent_id),
                    cohort="born",
                    injuries={int(child.agent_id): float(getattr(child, "injury", 0.0))},
                )
            except Exception:
                pass
        except ValueError:
            failed_births += 1
            runner._birth_fail_no_space_gen = int(getattr(runner, "_birth_fail_no_space_gen", 0)) + 1
            _record_birth_event(parent=parent, event_type="failed_birth", failure_reason="birth_fail_no_space")
            child.alive = False
            runner.pop.agents.pop(int(child.agent_id), None)

    if r6_successful_birth_rows:
        assert state is not None
        t_r6_attach = time.perf_counter()
        _runnercore6_stat_add(runner, "resident_newborn_attach_attempts", 1)
        _runnercore6_stat_add(runner, "resident_newborn_attach_candidate_rows", len(r6_successful_birth_rows))
        r6_attached_mlp_ids = _runnercore5_direct_attach_newborn_mlp_rows(runner, r6_successful_birth_rows, buckets)
        r7_attached_gru_ids = _runnercore7_direct_attach_newborn_gru_rows(runner, r6_successful_birth_rows, buckets)
        r6_attached_newborn_ids_to_add: list[int] = []
        for child_spec, pos, parent in r6_successful_birth_rows:
            slot = sync_newborn_row_from_contract(state, runner, child_spec)
            if slot < 0:
                _runnercore6_stat_add(runner, "resident_newborn_attach_unsupported_rows", 1)
                if _as_bool(getattr(runner.sim_cfg, "runnercore6_fail_closed_on_unsupported", False)):
                    raise RunnerCore1UnsupportedError("RUNNERCORE6-UNSUPPORTED: newborn resident row attach failed")
                continue
            _runnercore6_stat_add(runner, "resident_newborn_attach_rows", 1)
            _runnercore6_stat_add(runner, "runnercore6_resident_first_attaches", 1)
            _runnercore6_stat_add(runner, "generic_newborn_import_rows_avoided", 1)
            _runnercore6_stat_add(runner, "runnercore6_event_import_rows_avoided", 1)
            if r10_typed_child:
                _runnercore10_stat_add(runner, "typed_child_specs_attached", 1)
                _runnercore10_stat_add(runner, "typed_child_policy_rows_attached", 1)
            if r11_typed_rows and getattr(child_spec, "c1_typed_genome_row", None) is not None:
                _runnercore11_stat_add(runner, "c1_typed_rows_attached", 1)
                _runnercore11_stat_add(runner, "policy_rows_attached", 1)
            state.counters.birth_events_applied += 1
            state.counters.event_boundary_agent_create_rows += 1
            state.counters.event_boundary_policy_bank_sync_rows += 1
            state.counters.policy_row_attach_events += 1
            if bool(state.use_gru[int(slot)]):
                state.counters.hmem_row_init_events += 1
                if r10_typed_child:
                    _runnercore10_stat_add(runner, "typed_child_hmem_rows_attached", 1)
                if r11_typed_rows and getattr(child_spec, "c1_typed_genome_row", None) is not None:
                    _runnercore11_stat_add(runner, "hmem_rows_attached", 1)
            record_churn_event(
                state,
                event_type="birth",
                generation=int(getattr(runner.pop, "_eiel_current_generation_index", 0) or 0),
                tick=int(getattr(runner.pop, "_eiel_current_tick_index", 0) or 0),
                slot=int(slot),
                agent_id=int(child_spec.agent_id),
                reason="resident_child_row_attach",
                slot_token=int(state.slot_tokens[int(slot)]),
            )
            r6_attached_newborn_ids_to_add.append(int(child_spec.agent_id))
            if r10_typed_child:
                child_id = int(child_spec.agent_id)
                direct_mlp = child_id in r6_attached_mlp_ids
                direct_gru = child_id in r7_attached_gru_ids
                if direct_mlp or direct_gru:
                    if direct_mlp:
                        _runnercore10_stat_add(runner, "typed_child_direct_mlp_rows", 1)
                    if direct_gru:
                        _runnercore10_stat_add(runner, "typed_child_direct_gru_rows", 1)
                    if str(getattr(child_spec, "policy_source_kind", "")) in {"parent_a_alias", "parent_b_alias"}:
                        _runnercore10_stat_add(runner, "policy_object_creation_avoided", 1)
                        if r11_typed_rows:
                            _runnercore11_stat_add(runner, "policy_object_creation_avoided", 1)
                    if r11_typed_rows:
                        _runnercore11_stat_add(runner, "compatibility_export_avoided", 1)
                    _runnercore10_store_typed_child_spec(
                        runner,
                        child_spec,
                        reason=("direct_mlp" if direct_mlp else "direct_gru"),
                    )
                    continue
                if (
                    r11_typed_rows
                    and getattr(child_spec, "c1_typed_genome_row", None) is not None
                    and str(getattr(child_spec, "policy_source_kind", "")) == "materialized_policy"
                ):
                    policy_obj = getattr(child_spec, "policy", None)
                    if bool(state.use_gru[int(slot)]):
                        if hasattr(runner, "_sync_gru_bank_for_agents"):
                            runner._sync_gru_bank_for_agents([child_spec], buckets=buckets)
                    elif hasattr(policy_obj, "net") and hasattr(runner, "_sync_bank_for_agents"):
                        runner._sync_bank_for_agents([child_spec], buckets=buckets)
                    _runnercore11_stat_add(runner, "policy_object_creation_forced", 1)
                    _runnercore11_stat_add(runner, "compatibility_export_avoided", 1)
                    _runnercore10_store_typed_child_spec(
                        runner,
                        child_spec,
                        reason="materialized_policy",
                    )
                    continue
                _runnercore10_stat_add(runner, "typed_child_unsupported_rows", 1)
                _runnercore10_stat_add(runner, "compatibility_materializations_forced", 1)
                _runnercore10_reason_add(runner, "policy_source", 1)
                if r11_typed_rows:
                    _runnercore11_stat_add(runner, "c1_typed_rows_unsupported_kind", 1)
                    _runnercore11_stat_add(runner, "forced_materialization_policy_source", 1)
                    _runnercore11_reason_add(runner, "policy_source", 1)
                child = runner.pop.materialize_child_row_spec(child_spec)
                _runnercore6_stat_add(runner, "runnercore6_post_attach_export_materializations", 1)
                if child_id not in r7_attached_gru_ids:
                    _runnercore10_reason_add(runner, "policy_bank_sync", 1)
                    if r11_typed_rows:
                        _runnercore11_stat_add(runner, "forced_materialization_policy_bank_sync", 1)
                        _runnercore11_reason_add(runner, "policy_bank_sync", 1)
                newborns.append(child)
                newborn_birth_rows.append((child, pos, parent))
                if _as_bool(getattr(runner.sim_cfg, "behavior_log_enable", True)):
                    if r11_typed_rows:
                        _runnercore11_stat_add(runner, "forced_materialization_behavior_log", 1)
                        _runnercore11_reason_add(runner, "behavior_log", 1)
                    try:
                        runner.behavior_logger.register_agent(
                            int(child.agent_id),
                            cohort="born",
                            injuries={int(child.agent_id): float(getattr(child, "injury", 0.0))},
                        )
                    except Exception:
                        pass
                continue
            if r9_lazy_newborn:
                child_id = int(child_spec.agent_id)
                if child_id not in r6_attached_mlp_ids:
                    child = runner.pop.materialize_child_row_spec(child_spec)
                    _runnercore9_stat_add(runner, "policy_source_forced_materializations", 1)
                    if child_id not in r7_attached_gru_ids:
                        _runnercore9_stat_add(runner, "policy_bank_sync_forced_materializations", 1)
                    _runnercore6_stat_add(runner, "runnercore6_post_attach_export_materializations", 1)
                    newborns.append(child)
                    newborn_birth_rows.append((child, pos, parent))
                    if _as_bool(getattr(runner.sim_cfg, "behavior_log_enable", True)):
                        try:
                            runner.behavior_logger.register_agent(
                                int(child.agent_id),
                                cohort="born",
                                injuries={int(child.agent_id): float(getattr(child, "injury", 0.0))},
                            )
                        except Exception:
                            pass
                    continue
                specs = getattr(runner.pop, "_runnercore9_lazy_child_specs", None)
                if not isinstance(specs, dict):
                    specs = {}
                    try:
                        setattr(runner.pop, "_runnercore9_lazy_child_specs", specs)
                    except Exception:
                        specs = {}
                specs[child_id] = child_spec
                _runnercore9_stat_add(runner, "newborn_agent_materializations_deferred", 1)
                _runnercore9_stat_add(runner, "resident_rows_without_agent_objects", 1)
                _runnercore9_stat_add(runner, "event_boundary_export_rows_avoided", 1)
                continue
            child = runner.pop.materialize_child_row_spec(child_spec)
            _runnercore6_stat_add(runner, "runnercore6_post_attach_export_materializations", 1)
            newborns.append(child)
            if int(getattr(child, "agent_id", -1)) not in r6_attached_mlp_ids:
                newborn_birth_rows.append((child, pos, parent))
            if _as_bool(getattr(runner.sim_cfg, "behavior_log_enable", True)):
                try:
                    runner.behavior_logger.register_agent(
                        int(child.agent_id),
                        cohort="born",
                        injuries={int(child.agent_id): float(getattr(child, "injury", 0.0))},
                    )
                except Exception:
                    pass
        if r6_attached_newborn_ids_to_add:
            try:
                attached = set(int(aid) for aid in list(getattr(runner, "_runnercore6_resident_attached_newborn_ids", []) or []))
                attached.update(int(aid) for aid in r6_attached_newborn_ids_to_add)
                runner._runnercore6_resident_attached_newborn_ids = sorted(attached)
            except Exception:
                pass
        buckets["runnercore6_resident_newborn_attach"] = float(
            buckets.get("runnercore6_resident_newborn_attach", 0.0) or 0.0
        ) + float(time.perf_counter() - t_r6_attach)

    if newborns:
        attached_mlp_ids = (
            set(r6_attached_mlp_ids)
            if r6_resident_attach
            else _runnercore5_direct_attach_newborn_mlp_rows(runner, newborn_birth_rows, buckets)
        )
        sync_newborns = [
            child
            for child in newborns
            if int(getattr(child, "agent_id", -1)) not in attached_mlp_ids
        ]
        if sync_newborns:
            runner._sync_bank_for_agents(sync_newborns, buckets=buckets)
        if hasattr(runner, "_sync_gru_bank_for_agents"):
            sync_gru_newborns = [
                child
                for child in newborns
                if int(getattr(child, "agent_id", -1)) not in r7_attached_gru_ids
            ]
            if sync_gru_newborns:
                runner._sync_gru_bank_for_agents(sync_gru_newborns, buckets=buckets)

    buckets["births_apply"] = buckets.get("births_apply", 0.0) + (time.perf_counter() - t_birth0)
    buckets["reproduction_ms"] = buckets.get("reproduction_ms", 0.0) + (time.perf_counter() - t_repro0)
    try:
        runner._runnercore4_birth_changed_parent_ids = sorted(int(aid) for aid in changed_parent_ids)
    except Exception:
        pass
    return int(successful_births), int(failed_births)


def _runnercore34_full_birth_resident_decision_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(
            getattr(runner, "sim_cfg", None),
            "runnercore34_full_birth_resident_decision_enable",
            False,
        )
    )


def _runnercore42_patch_birth_context_cache_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return _as_bool(getattr(cfg, "runnercore42_full_birth_patch_context_cache_enable", False))


def _runnercore43_threshold_tick_cache_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return _as_bool(getattr(cfg, "runnercore43_full_birth_threshold_tick_cache_enable", False))


def _runnercore45_jungle_predator_context_cache_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return _as_bool(getattr(cfg, "runnercore45_jungle_predator_context_cache_enable", False))


def _runnercore42_density_denom(cfg: Any) -> np.float32:
    radius = int(max(0, getattr(cfg, "patch_density_radius", 1)))
    denom = float((2 * radius + 1) * (2 * radius + 1))
    if not bool(getattr(cfg, "patch_density_include_self", True)):
        denom -= 1.0
    return np.float32(denom if denom > 0.0 else 1.0)


def _runnercore42_compute_density_counts(env: Any) -> np.ndarray:
    agent_grid = np.asarray(getattr(env, "agent_grid"), dtype=np.int32)
    cfg = getattr(env, "cfg", None)
    radius = int(max(0, getattr(cfg, "patch_density_radius", 1)))
    include_self = bool(getattr(cfg, "patch_density_include_self", True))
    density = np.zeros(tuple(agent_grid.shape), dtype=np.float32)
    ys, xs = np.where(agent_grid != -1)
    h, w = tuple(agent_grid.shape)
    for y_raw, x_raw in zip(ys.tolist(), xs.tolist()):
        y = int(y_raw)
        x = int(x_raw)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        density[y0:y1, x0:x1] += np.float32(1.0)
        if not include_self:
            density[y, x] -= np.float32(1.0)
    return np.ascontiguousarray(density, dtype=np.float32)


def _runnercore42_update_density_counts_for_birth(cache: dict[str, Any], y: int, x: int) -> None:
    density = np.asarray(cache["density_counts"], dtype=np.float32)
    radius = int(cache["density_radius"])
    include_self = bool(cache["density_include_self"])
    h, w = tuple(density.shape)
    yy = max(0, min(h - 1, int(y)))
    xx = max(0, min(w - 1, int(x)))
    y0 = max(0, yy - radius)
    y1 = min(h, yy + radius + 1)
    x0 = max(0, xx - radius)
    x1 = min(w, xx + radius + 1)
    density[y0:y1, x0:x1] += np.float32(1.0)
    if not include_self:
        density[yy, xx] -= np.float32(1.0)


def _runnercore45_update_density_counts_for_death(cache: dict[str, Any], y: int, x: int) -> None:
    density = np.asarray(cache["density_counts"], dtype=np.float32)
    radius = int(cache["density_radius"])
    include_self = bool(cache["density_include_self"])
    h, w = tuple(density.shape)
    yy = max(0, min(h - 1, int(y)))
    xx = max(0, min(w - 1, int(x)))
    y0 = max(0, yy - radius)
    y1 = min(h, yy + radius + 1)
    x0 = max(0, xx - radius)
    x1 = min(w, xx + radius + 1)
    density[y0:y1, x0:x1] -= np.float32(1.0)
    if not include_self:
        density[yy, xx] += np.float32(1.0)


def _runnercore42_prepare_patch_context_cache(env: Any) -> dict[str, Any]:
    from env.gridworld_pkg.jungle import queue_jungle_birth_emission  # noqa: WPS433
    from env.gridworld_pkg.patch_world import (  # noqa: WPS433
        PATCH_WORLD_CONTEXT_FIELDS,
        patch_habitat_lineage_id,
        patch_visibility_from_cover,
    )

    cfg = getattr(env, "cfg", None)
    cover = np.asarray(getattr(env, "patch_cover"), dtype=np.float32)
    maps = {
        "food_amount": np.asarray(getattr(env, "patch_food_amount"), dtype=np.float32),
        "fertility": np.asarray(getattr(env, "patch_fertility"), dtype=np.float32),
        "cover": cover,
        "shelter": np.asarray(getattr(env, "patch_shelter"), dtype=np.float32),
        "visibility": patch_visibility_from_cover(cfg, cover),
        "movement_cost": np.asarray(getattr(env, "patch_movement_cost"), dtype=np.float32),
        "scent": np.asarray(getattr(env, "patch_scent"), dtype=np.float32),
        "waste": np.asarray(getattr(env, "patch_waste"), dtype=np.float32),
        "recovery_activity": np.asarray(getattr(env, "patch_recovery_activity"), dtype=np.float32),
    }
    density_counts = _runnercore42_compute_density_counts(env)
    grid_shape = tuple(np.asarray(getattr(env, "grid")).shape)
    return {
        "fields": tuple(PATCH_WORLD_CONTEXT_FIELDS),
        "maps": maps,
        "density_counts": density_counts,
        "density_normalize": bool(getattr(cfg, "patch_density_normalize", False)),
        "density_denom": _runnercore42_density_denom(cfg),
        "density_radius": int(max(0, getattr(cfg, "patch_density_radius", 1))),
        "density_include_self": bool(getattr(cfg, "patch_density_include_self", True)),
        "grid_shape": grid_shape,
        "patch_habitat_lineage_id": patch_habitat_lineage_id,
        "queue_jungle_birth_emission": queue_jungle_birth_emission,
    }


def _runnercore42_context_from_cache(
    cache: dict[str, Any],
    y: int,
    x: int,
    *,
    agent: Any = None,
    lineage_id: int | None = None,
) -> dict[str, float]:
    h, w = tuple(cache["grid_shape"])
    yy = max(0, min(int(h) - 1, int(y)))
    xx = max(0, min(int(w) - 1, int(x)))
    maps = cache["maps"]
    density = np.asarray(cache["density_counts"], dtype=np.float32)
    out: dict[str, float] = {}
    for name in cache["fields"]:
        if name == "density":
            value = density[yy, xx]
            if bool(cache["density_normalize"]):
                value = np.float32(value / np.float32(cache["density_denom"]))
            out[name] = float(value)
        else:
            out[name] = float(np.asarray(maps[name], dtype=np.float32)[yy, xx])
    out["y"] = float(yy)
    out["x"] = float(xx)
    if lineage_id is None and agent is not None:
        lineage_id = cache["patch_habitat_lineage_id"](agent)
    out["lineage_id"] = float(int(lineage_id or 0))
    return out


def _runnercore42_finalize_patch_context_cache(cache: dict[str, Any], env: Any) -> None:
    density = np.asarray(cache["density_counts"], dtype=np.float32)
    if bool(cache["density_normalize"]):
        density = np.ascontiguousarray(density / np.float32(cache["density_denom"]), dtype=np.float32)
    else:
        density = np.ascontiguousarray(density, dtype=np.float32)
    env.local_density_map = density


def _runnercore34_stat_add(runner: Any, key: str, value: int = 1) -> None:
    stats = getattr(getattr(runner, "sim_cfg", None), "_runnercore6_resident_child_row_stats", None)
    if not isinstance(stats, dict):
        stats = {}
        try:
            setattr(runner.sim_cfg, "_runnercore6_resident_child_row_stats", stats)
        except Exception:
            return
    stats[str(key)] = int(stats.get(str(key), 0) or 0) + int(value)


def _runnercore49_repro_viability_tick_cache_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "runnercore49_repro_viability_tick_cache_enable", False)
    )


def _runnercore50_spawn_first_valid_enabled(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore50_spawn_first_valid_enable", False))


def _runnercore34_dev_diag_active(runner: Any) -> bool:
    diag = getattr(runner, "env_hotpath_expression_diagnostics", None)
    if diag is None:
        return False
    return bool(
        getattr(diag, "developmental_cache_equality_enabled", False)
        or getattr(diag, "developmental_effective_projection_enabled", False)
        or getattr(diag, "developmental_residual_enabled", False)
    )


def _runnercore34_static_reproduction_threshold_or_none(
    runner: Any,
    agent: Any,
    *,
    survival_steps: int,
) -> float | None:
    if not _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore31_full_birth_static_dev_threshold_enable", False)):
        return None
    if _runnercore34_dev_diag_active(runner):
        return None
    key_fn = getattr(runner, "_dev_expression_cache_key_for_agent", None)
    cache = getattr(runner, "_dev_expr_cache", None)
    if not callable(key_fn) or not isinstance(cache, dict):
        return None
    st = getattr(agent, "stats", None)
    old_survival_steps = None
    if st is not None:
        old_survival_steps = getattr(st, "survival_steps", None)
        try:
            st.survival_steps = int(survival_steps)
        except Exception:
            pass
    try:
        key = key_fn(agent)
    except Exception:
        if st is not None and old_survival_steps is not None:
            try:
                st.survival_steps = old_survival_steps
            except Exception:
                pass
        return None
    genome = getattr(agent, "genome", None)
    cached = cache.get(key)
    if isinstance(cached, dict):
        if bool(getattr(runner, "_dev_effective_scalar_cache_enabled", lambda: False)()):
            scalar_cache = getattr(runner, "_dev_effective_scalar_cache", None)
            if not isinstance(scalar_cache, dict):
                scalar_cache = {}
                runner._dev_effective_scalar_cache = scalar_cache
            if key not in scalar_cache:
                scalar_fn = getattr(runner, "_dev_effective_scalars_from_expression", None)
                if callable(scalar_fn):
                    try:
                        scalar_cache[key] = scalar_fn(agent, cached)
                    except Exception:
                        pass
        eff = cached.get("effective", {}) if isinstance(cached.get("effective", {}), dict) else {}
        _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_cache_hits", 1)
        _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_hits", 1)
        return float(eff.get("reproduction_threshold", getattr(genome, "reproduction_threshold", 50.0)))

    if genome is None:
        return None
    modules = list(getattr(genome, "modules", []) or [])
    if modules:
        _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_module_fallbacks", 1)
        return None
    if getattr(genome, "developmental_program", None) != RUNNERCORE34_DEFAULT_DEV_PROGRAM:
        _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_program_fallbacks", 1)
        return None
    empty_module_sig = RUNNERCORE34_EMPTY_MODULE_SIGNATURE
    if st is not None:
        baseline_sig = str(getattr(st, "_dev_module_structure_signature", "") or "")
        if baseline_sig and baseline_sig != empty_module_sig:
            _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_signature_fallbacks", 1)
            return None

    try:
        aid, gidx, tidx, age = key
        eff = {
            "aggression_bias": float(getattr(genome, "aggression_bias", 0.0)),
            "action_temperature": float(getattr(genome, "action_temperature", 1.0)),
            "drive_logit_gain": float(getattr(genome, "drive_logit_gain", 1.0)),
            "drive_inertia": float(getattr(genome, "drive_inertia", 0.85)),
            "drive_noise": float(getattr(genome, "drive_noise", 0.02)),
            "pred_lr": float(getattr(genome, "pred_lr", 0.08)),
            "reproduction_threshold": float(getattr(genome, "reproduction_threshold", 50.0)),
        }
        expr = {
            "stage_id": "default",
            "active_module_count": 0,
            "module_count": 0,
            "active_by_module_id": {},
            "inactive_reason_by_module_id": {},
            "module_structure_signature": str(empty_module_sig),
            "effective": eff,
            "structural_change_ignored": False,
            "structural_change_reason": "",
            "lock_state_by_module_id": {},
            "lock_events": [],
            "lock_event_count": 0,
        }
        lock_state_sha = RUNNERCORE34_EMPTY_LOCK_STATE_SHA256
        if st is not None:
            if not str(getattr(st, "_dev_module_structure_signature", "") or ""):
                setattr(st, "_dev_module_structure_signature", str(empty_module_sig))
            prev_stage = str(getattr(st, "_dev_stage_id", "") or "")
            prev_active = runner._dev_previous_active_module_count(st)
            if (str(expr["stage_id"]) != prev_stage) or (int(expr["active_module_count"]) != prev_active):
                queue_trace = getattr(runner, "_dev_queue_trace", None)
                reason_counts = getattr(runner, "_dev_reason_counts", None)
                if callable(queue_trace):
                    queue_trace(
                        {
                            "schema_version": "developmental_stage_trace_v1",
                            "generation": int(gidx),
                            "tick": int(tidx),
                            "agent_id": int(aid),
                            "age_steps": int(age),
                            "stage_id": "default",
                            "active_module_count": 0,
                            "module_count": 0,
                            "locked_module_count": 0,
                            "lock_state_sha256": str(lock_state_sha),
                            "structural_change_ignored": False,
                            "structural_change_reason": "",
                            "inactive_reason_counts": reason_counts(expr) if callable(reason_counts) else {},
                        }
                    )
                runner._dev_stage_transition_count_gen = int(
                    getattr(runner, "_dev_stage_transition_count_gen", 0) or 0
                ) + 1
                counts = getattr(runner, "_dev_stage_eval_stage_counts_gen", None)
                if not isinstance(counts, dict):
                    counts = {}
                counts["default"] = int(counts.get("default", 0) or 0) + 1
                runner._dev_stage_eval_stage_counts_gen = counts
            setattr(st, "_dev_stage_id", "default")
            setattr(st, "_dev_active_module_count", 0)
            setattr(st, "_dev_last_effective", dict(eff))
            setattr(st, "_dev_d2_lock_state_by_module_id", {})
            setattr(st, "_dev_d2_lock_state_sha256", str(lock_state_sha))
            setattr(
                st,
                "_dev_last_expr_snapshot",
                {
                    "stage_id": "default",
                    "active_module_count": 0,
                    "module_count": 0,
                    "active_by_module_id": {},
                    "inactive_reason_by_module_id": {},
                    "effective": dict(eff),
                    "lock_state_by_module_id": {},
                },
            )
            setattr(st, "_dev_last_trace_gen", int(gidx))
            setattr(st, "_dev_last_trace_tick", int(tidx))
        cache[key] = expr
        stage_store = getattr(runner, "_dev_stage_cache_store", None)
        if callable(stage_store):
            stage_store(agent, genome, expr, age_steps=int(age))
        if bool(getattr(runner, "_dev_effective_scalar_cache_enabled", lambda: False)()):
            scalar_fn = getattr(runner, "_dev_effective_scalars_from_expression", None)
            if callable(scalar_fn):
                try:
                    scalar_cache = getattr(runner, "_dev_effective_scalar_cache", None)
                    if not isinstance(scalar_cache, dict):
                        scalar_cache = {}
                        runner._dev_effective_scalar_cache = scalar_cache
                    scalar_cache[key] = scalar_fn(agent, expr)
                except Exception:
                    pass
        _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_synthetic_lookups", 1)
        _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_hits", 1)
        return float(eff["reproduction_threshold"])
    except Exception:
        _runnercore34_stat_add(runner, "runnercore31_full_birth_static_dev_threshold_exception_fallbacks", 1)
        return None


def _runnercore34_apply_dev_expression_side_effect_for_row(
    state: RunnerCore1TickState,
    runner: Any,
    aid: int,
    *,
    source: str,
) -> bool:
    slot = state.slot_for_agent_id(int(aid))
    if slot < 0 or not bool(state.alive[slot]):
        return False
    agents = getattr(getattr(runner, "pop", None), "agents", {})
    agent = agents.get(int(aid)) if isinstance(agents, dict) else None
    if agent is None:
        runnercore3_profile_count(state, f"runnercore34_dev_expression_{source}_missing_agent_rows", 1)
        return False
    survival_steps = int(state.survival_steps[slot])
    threshold = _runnercore34_static_reproduction_threshold_or_none(
        runner,
        agent,
        survival_steps=int(survival_steps),
    )
    if threshold is not None:
        runnercore3_profile_count(state, f"runnercore34_dev_expression_{source}_static_rows", 1)
        return True
    expr_fn = getattr(runner, "resolve_developmental_expression_for_agent", None)
    if not callable(expr_fn):
        return False
    st = getattr(agent, "stats", None)
    old_survival_steps = None
    if st is not None:
        old_survival_steps = getattr(st, "survival_steps", None)
        try:
            st.survival_steps = int(survival_steps)
        except Exception:
            pass
    try:
        expr_fn(agent)
        runnercore3_profile_count(state, f"runnercore34_dev_expression_{source}_canonical_rows", 1)
        return True
    except Exception:
        if st is not None and old_survival_steps is not None:
            try:
                st.survival_steps = old_survival_steps
            except Exception:
                pass
        runnercore3_profile_count(state, f"runnercore34_dev_expression_{source}_exception_rows", 1)
        return False


def _runnercore34_apply_dev_expression_side_effects(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: Any,
    *,
    source: str,
) -> int:
    if bool(getattr(getattr(runner, "sim_cfg", None), "perf_skip_dev_expression_prewarm_enable", False)):
        return 0
    rows = 0
    for aid_raw in sorted(int(aid) for aid in np.asarray(agent_ids, dtype=np.int64).reshape(-1).tolist()):
        if _runnercore34_apply_dev_expression_side_effect_for_row(
            state,
            runner,
            int(aid_raw),
            source=str(source),
        ):
            rows += 1
    runnercore3_profile_count(state, f"runnercore34_dev_expression_{source}_rows", rows)
    return int(rows)


def _runnercore34_handle_mate_births_resident_decision(
    runner: Any,
    state: RunnerCore1TickState,
    candidate_ids: list[int],
    successful_births: int,
    failed_births: int,
    buckets: dict[str, float],
    *,
    gen: int,
    tick: int,
) -> tuple[int, int]:
    """Own W6 mate/spawn birth decisions while preserving canonical child creation."""

    if not _as_bool(getattr(getattr(runner, "env_cfg", None), "reproduce_requires_mate", False)):
        raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: only mate-required full birth is supported")
    if _as_bool(getattr(getattr(runner, "sim_cfg", None), "lineage_reproduction_funnel_reporting_enable", False)):
        raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: lineage funnel reporting is not supported")
    if _as_bool(getattr(getattr(runner, "sim_cfg", None), "lineage_reproduction_funnel_write_events", False)):
        raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: lineage funnel event writes are not supported")
    env = getattr(runner, "env", None)
    pop = getattr(runner, "pop", None)
    agents = getattr(pop, "agents", {}) if pop is not None else {}
    if env is None or pop is None or not isinstance(agents, dict) or not hasattr(pop, "make_child"):
        raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: population/env child creation contract missing")
    runnercore25_requested = _as_bool(
        getattr(
            getattr(runner, "sim_cfg", None),
            "runnercore25_full_birth_resident_newborn_attach_enable",
            False,
        )
    )
    runnercore25_birth_row_spec_bridge = bool(runnercore25_requested)
    runnercore25_successful_ids: list[int] = []
    if runnercore25_birth_row_spec_bridge:
        if not _runnercore6_resident_attach_enabled(runner) or not hasattr(pop, "make_child_row_spec"):
            raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: R25 row-spec newborn bridge contract missing")
        _runnercore6_stats(runner)
        try:
            runner._runnercore25_full_birth_row_spec_newborn_ids = []
        except Exception:
            pass
    env_alive = getattr(env, "agent_alive", None)
    pos_y = getattr(env, "agent_pos_y", None)
    pos_x = getattr(env, "agent_pos_x", None)
    agent_grid = getattr(env, "agent_grid", None)
    if env_alive is None or pos_y is None or pos_x is None or agent_grid is None:
        raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: array env position/grid state missing")

    cap = int(np.asarray(env_alive).shape[0])
    candidate_set = {int(aid) for aid in candidate_ids}
    missing_agents = [aid for aid in sorted(candidate_set) if agents.get(int(aid)) is None]
    if missing_agents:
        raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: candidate parent agent mirrors missing")

    t_repro0 = time.perf_counter()
    t_birth_scan0 = time.perf_counter()
    t_make_child_sum = 0.0
    births: list[tuple[Any, tuple[int, int], Any]] = []
    newborn_birth_rows: list[tuple[Any, tuple[int, int], Any]] = []
    mated_this_tick: set[int] = set()
    reserved_tiles: set[tuple[int, int]] = set()
    changed_parent_ids: set[int] = set()
    parent_export_rows = 0
    considered_rows = 0
    mate_selected_rows = 0
    spawn_selected_rows = 0

    injury_gate = None
    if _as_bool(getattr(runner.env_cfg, "enable_injury", False)):
        ig = getattr(runner.env_cfg, "injury_repro_block", None)
        if ig is not None:
            injury_gate = float(ig)
    scar_gate = None
    if _as_bool(getattr(runner.env_cfg, "enable_scar", False)):
        sg = getattr(runner.env_cfg, "scar_repro_block", None)
        if sg is not None:
            scar_gate = float(sg)

    k_inj = float(getattr(runner.env_cfg, "injury_repro_k_injury", 0.0))
    k_scar = float(getattr(runner.env_cfg, "injury_repro_k_scar", 0.0))
    p_fail_scar = float(getattr(runner.env_cfg, "injury_birth_fail_scar", 0.0))
    p_fail_inj = float(getattr(runner.env_cfg, "injury_birth_fail_injury", 0.0))
    chem_repro_enable = bool(getattr(runner.sim_cfg, "chemistry_functional_coupling_enable", False)) and bool(
        getattr(runner.sim_cfg, "chemistry_repro_viability_enable", False)
    )
    chem_viability_tick_cache: dict[int, tuple[float, bool]] | None = (
        {} if chem_repro_enable and _runnercore49_repro_viability_tick_cache_enabled(runner) else None
    )
    chem_floor = float(getattr(runner.sim_cfg, "chemistry_repro_viability_floor", 0.2))
    chem_strength = float(getattr(runner.sim_cfg, "chemistry_repro_viability_strength", 0.8))
    chem_chronic_block = float(getattr(runner.sim_cfg, "chemistry_repro_chronic_block_threshold", 0.95))
    share = float(runner.env_cfg.reproduce_parent_energy_share)
    buffer = float(runner.env_cfg.reproduce_min_energy_buffer)
    cd = int(runner.env_cfg.reproduce_cooldown_steps)
    try_prob = max(0.0, min(1.0, float(getattr(runner.env_cfg, "reproduce_try_prob", 1.0))))
    spawn_first_valid = _runnercore50_spawn_first_valid_enabled(runner)

    for name in (
        "_birth_pre_no_energy_gen",
        "_birth_pre_no_space_gen",
        "_birth_pre_no_mate_gen",
        "_birth_attempt_conception_fail_gen",
        "_birth_make_child_fail_gen",
        "_injury_blocked_births_gen",
        "_birth_pre_injury_gate_gen",
        "_birth_fail_injury_gate_gen",
        "_birth_fail_scar_gate_gen",
        "_birth_fail_conception_gen",
        "_birth_precheck_no_space_gen",
        "_birth_precheck_no_mate_gen",
        "_birth_precheck_scar_gate_gen",
        "_birth_precheck_injury_gate_gen",
        "_birth_fail_buffer_gen",
        "_birth_fail_gestation_gen",
        "_birth_gestation_fail_gen",
        "_birth_attempts_gen",
    ):
        setattr(runner, name, int(getattr(runner, name, 0)))

    def _slot(aid: int) -> int:
        slot = state.slot_for_agent_id(int(aid))
        if slot < 0 or not bool(state.alive[slot]):
            return -1
        if not (0 <= int(aid) < cap and bool(env_alive[int(aid)])):
            return -1
        return int(slot)

    threshold_tick_cache: dict[int, float] | None = (
        {} if _runnercore43_threshold_tick_cache_enabled(runner) else None
    )

    def _threshold(aid: int, slot: int) -> float:
        if threshold_tick_cache is not None and int(aid) in threshold_tick_cache:
            _runnercore34_stat_add(runner, "runnercore43_full_birth_threshold_tick_cache_hits", 1)
            return float(threshold_tick_cache[int(aid)])
        if _as_bool(
            getattr(getattr(runner, "sim_cfg", None), "runnercore31_full_birth_static_dev_threshold_enable", False)
        ):
            agent = agents.get(int(aid))
            if agent is None:
                raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: threshold agent mirror missing")
            threshold = _runnercore34_static_reproduction_threshold_or_none(
                runner,
                agent,
                survival_steps=int(state.survival_steps[int(slot)]),
            )
            if threshold is None:
                raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: dynamic developmental threshold required")
            value = float(threshold)
        else:
            value = float(state.repro_threshold[int(slot)])
        if threshold_tick_cache is not None:
            threshold_tick_cache[int(aid)] = float(value)
            _runnercore34_stat_add(runner, "runnercore43_full_birth_threshold_tick_cache_misses", 1)
        return float(value)

    def _set_agent_cooldown(aid: int, slot: int, value: int) -> None:
        state.repro_cooldown[int(slot)] = np.int32(int(value))
        agent = agents.get(int(aid))
        if agent is not None:
            agent.repro_cooldown = int(value)

    def _set_agent_energy(aid: int, slot: int, value: float) -> None:
        state.energy[int(slot)] = np.float32(float(value))
        agent = agents.get(int(aid))
        if agent is not None:
            agent.energy = float(state.energy[int(slot)])

    def _chem_viability(slot: int, aid: int) -> tuple[float, bool]:
        if not chem_repro_enable:
            return 1.0, False
        cache_key = int(aid)
        if chem_viability_tick_cache is not None and cache_key in chem_viability_tick_cache:
            if _runnercore3_profile_enabled(runner):
                runnercore3_profile_count(state, "runnercore49_repro_viability_tick_cache_hits", 1)
            return chem_viability_tick_cache[cache_key]
        chem = chemistry_from_values(state.chemistry[int(slot)])
        viability = reproduction_viability_scale_from_chem_row(chem, floor=chem_floor, strength=chem_strength)
        blocked = float(chem[CHEM_SLOT_CHRONIC_DAMAGE]) >= chem_chronic_block
        agent = agents.get(int(aid))
        try:
            if agent is not None:
                agent.stats.chemistry_repro_viability = float(viability)
        except Exception:
            pass
        result = (float(viability), bool(blocked))
        if chem_viability_tick_cache is not None:
            chem_viability_tick_cache[cache_key] = result
            if _runnercore3_profile_enabled(runner):
                runnercore3_profile_count(state, "runnercore49_repro_viability_tick_cache_misses", 1)
        return result

    def _in_bounds(y: int, x: int) -> bool:
        return bool(env.in_bounds(int(y), int(x)))

    def _make_child(agent: Any, mate: Any, child_energy: float) -> Any | None:
        if not runnercore25_birth_row_spec_bridge:
            return pop.make_child(agent, mate, child_energy, buckets=buckets)
        child_spec = pop.make_child_row_spec(agent, mate, child_energy, buckets=buckets)
        if child_spec is None:
            return None
        agents[int(child_spec.agent_id)] = child_spec
        _runnercore6_stat_add(runner, "runnercore25_full_birth_row_specs_created", 1)
        _runnercore6_stat_add(runner, "runnercore25_pre_attach_agent_materializations_avoided", 1)
        return child_spec

    old_typed_rows = getattr(runner.sim_cfg, "runnercore11_c1_typed_rows_enable", None)
    setattr(runner.sim_cfg, "runnercore11_c1_typed_rows_enable", False)
    try:
        for aid_raw in sorted(int(aid) for aid in candidate_ids):
            aid = int(aid_raw)
            slot = _slot(aid)
            if slot < 0:
                continue
            considered_rows += 1

            cooldown = int(state.repro_cooldown[slot])
            if cooldown > 0:
                cooldown = max(0, int(cooldown) - 1)
                _set_agent_cooldown(aid, slot, cooldown)
                changed_parent_ids.add(aid)
            if cooldown != 0:
                runner._birth_pre_cooldown_gen = int(getattr(runner, "_birth_pre_cooldown_gen", 0)) + 1
                continue

            if aid in mated_this_tick:
                continue

            if float(state.energy[slot]) < _threshold(aid, slot):
                runner._birth_pre_no_energy_gen = int(getattr(runner, "_birth_pre_no_energy_gen", 0)) + 1
                continue

            inj = float(state.injury[slot])
            scar = float(state.scar[slot])
            if injury_gate is not None and inj > injury_gate:
                runner._injury_blocked_births_gen = int(getattr(runner, "_injury_blocked_births_gen", 0)) + 1
                runner._birth_pre_injury_gate_gen = int(getattr(runner, "_birth_pre_injury_gate_gen", 0)) + 1
                runner._birth_precheck_injury_gate_gen = int(getattr(runner, "_birth_precheck_injury_gate_gen", 0)) + 1
                runner._birth_fail_injury_gate_gen = int(getattr(runner, "_birth_fail_injury_gate_gen", 0)) + 1
                continue
            if scar_gate is not None and scar > scar_gate:
                runner._birth_precheck_scar_gate_gen = int(getattr(runner, "_birth_precheck_scar_gate_gen", 0)) + 1
                runner._birth_fail_scar_gate_gen = int(getattr(runner, "_birth_fail_scar_gate_gen", 0)) + 1
                continue

            viability_self, self_blocked = _chem_viability(slot, aid)
            if self_blocked:
                runner._birth_fail_injury_gate_gen = int(getattr(runner, "_birth_fail_injury_gate_gen", 0)) + 1
                continue

            adj = env.adjacent_agent_ids(aid)
            if not adj:
                runner._birth_pre_no_mate_gen = int(getattr(runner, "_birth_pre_no_mate_gen", 0)) + 1
                runner._birth_precheck_no_mate_gen = int(getattr(runner, "_birth_precheck_no_mate_gen", 0)) + 1
                runner._birth_fail_no_mate_gen = int(getattr(runner, "_birth_fail_no_mate_gen", 0)) + 1
                continue

            runner.np_rng.shuffle(adj)
            mate_id = -1
            mate_slot = -1
            mate_inj = 0.0
            mate_scar = 0.0
            viability_mate = 1.0
            for cand_raw in adj:
                cand_id = int(cand_raw)
                cand_slot = _slot(cand_id)
                if cand_slot < 0:
                    continue
                if agents.get(cand_id) is None:
                    continue
                if int(state.repro_cooldown[cand_slot]) != 0:
                    continue
                if cand_id in mated_this_tick:
                    continue
                if float(state.energy[cand_slot]) < _threshold(cand_id, cand_slot):
                    continue
                cand_inj = float(state.injury[cand_slot])
                cand_scar = float(state.scar[cand_slot])
                if injury_gate is not None and cand_inj > injury_gate:
                    continue
                if scar_gate is not None and cand_scar > scar_gate:
                    continue
                viability_cand, cand_blocked = _chem_viability(cand_slot, cand_id)
                if cand_blocked:
                    continue
                mate_id = cand_id
                mate_slot = cand_slot
                mate_inj = cand_inj
                mate_scar = cand_scar
                viability_mate = float(viability_cand)
                break

            if mate_id < 0 or mate_slot < 0:
                runner._birth_pre_no_mate_gen = int(getattr(runner, "_birth_pre_no_mate_gen", 0)) + 1
                runner._birth_precheck_no_mate_gen = int(getattr(runner, "_birth_precheck_no_mate_gen", 0)) + 1
                runner._birth_fail_no_mate_gen = int(getattr(runner, "_birth_fail_no_mate_gen", 0)) + 1
                continue
            mate_selected_rows += 1

            ay = int(pos_y[aid])
            ax = int(pos_x[aid])
            my = int(pos_y[mate_id])
            mx = int(pos_x[mate_id])
            cand = [
                (ay - 1, ax),
                (ay + 1, ax),
                (ay, ax - 1),
                (ay, ax + 1),
                (my - 1, mx),
                (my + 1, mx),
                (my, mx - 1),
                (my, mx + 1),
            ]
            env.rng.shuffle(cand)
            spawn_pos = None
            if spawn_first_valid:
                if _runnercore3_profile_enabled(runner):
                    runnercore3_profile_count(state, "runnercore50_spawn_first_valid_scan_rows", 1)
                for ny, nx in cand:
                    iy = int(ny)
                    ix = int(nx)
                    if not _in_bounds(iy, ix):
                        continue
                    if int(agent_grid[iy, ix]) != -1:
                        continue
                    if (iy, ix) in reserved_tiles:
                        continue
                    spawn_pos = (iy, ix)
                    break
                if spawn_pos is not None and _runnercore3_profile_enabled(runner):
                    runnercore3_profile_count(state, "runnercore50_spawn_first_valid_selected_rows", 1)
            else:
                candidates: list[tuple[int, int]] = []
                for ny, nx in cand:
                    if not _in_bounds(int(ny), int(nx)):
                        continue
                    if int(agent_grid[int(ny), int(nx)]) != -1:
                        continue
                    if (int(ny), int(nx)) in reserved_tiles:
                        continue
                    candidates.append((int(ny), int(nx)))
                if candidates:
                    seen: set[tuple[int, int]] = set()
                    uniq: list[tuple[int, int]] = []
                    for p in candidates:
                        if p not in seen:
                            uniq.append(p)
                            seen.add(p)
                    candidates = uniq

                for p in candidates:
                    if p not in reserved_tiles:
                        spawn_pos = p
                        break
            if spawn_pos is None:
                runner._birth_pre_no_space_gen = int(getattr(runner, "_birth_pre_no_space_gen", 0)) + 1
                runner._birth_precheck_no_space_gen = int(getattr(runner, "_birth_precheck_no_space_gen", 0)) + 1
                runner._birth_fail_no_space_gen = int(getattr(runner, "_birth_fail_no_space_gen", 0)) + 1
                continue
            spawn_selected_rows += 1

            if try_prob < 1.0 and float(runner.np_rng.random()) > try_prob:
                continue

            e1 = float(state.energy[slot])
            e2 = float(state.energy[mate_slot])
            give1 = e1 * share
            give2 = e2 * share
            if (e1 - give1 < buffer) or (e2 - give2 < buffer):
                failed_births += 1
                runner._birth_fail_buffer_gen = int(getattr(runner, "_birth_fail_buffer_gen", 0)) + 1
                continue

            runner._birth_attempts_gen = int(getattr(runner, "_birth_attempts_gen", 0)) + 1
            if (k_inj > 0.0) or (k_scar > 0.0):
                p1 = float(np.exp(-(k_inj * inj + k_scar * scar))) * float(viability_self)
                if float(runner.np_rng.random()) > p1:
                    failed_births += 1
                    runner._birth_attempt_conception_fail_gen = (
                        int(getattr(runner, "_birth_attempt_conception_fail_gen", 0)) + 1
                    )
                    runner._birth_fail_conception_gen = int(getattr(runner, "_birth_fail_conception_gen", 0)) + 1
                    continue
                p2 = float(np.exp(-(k_inj * mate_inj + k_scar * mate_scar))) * float(viability_mate)
                if float(runner.np_rng.random()) > p2:
                    failed_births += 1
                    runner._birth_attempt_conception_fail_gen = (
                        int(getattr(runner, "_birth_attempt_conception_fail_gen", 0)) + 1
                    )
                    runner._birth_fail_conception_gen = int(getattr(runner, "_birth_fail_conception_gen", 0)) + 1
                    continue

            reserved_tiles.add(spawn_pos)
            export_to_agents(
                state,
                runner,
                boundary_reason="runnercore34_successful_birth_parent_export",
                agent_ids=[aid, mate_id],
            )
            parent_export_rows += 2
            agent = agents.get(aid)
            mate = agents.get(mate_id)
            if agent is None or mate is None:
                reserved_tiles.discard(spawn_pos)
                raise RunnerCore1UnsupportedError("RUNNERCORE34-UNSUPPORTED: selected parent mirror missing")

            _set_agent_energy(aid, slot, e1 - give1)
            _set_agent_energy(mate_id, mate_slot, e2 - give2)
            child_energy = give1 + give2
            if (p_fail_scar > 0.0) or (p_fail_inj > 0.0):
                p_fail = min(0.95, p_fail_scar * (scar + mate_scar) + p_fail_inj * (inj + mate_inj))
                if float(runner.np_rng.random()) < p_fail:
                    _set_agent_energy(aid, slot, e1)
                    _set_agent_energy(mate_id, mate_slot, e2)
                    failed_births += 1
                    runner._birth_gestation_fail_gen = int(getattr(runner, "_birth_gestation_fail_gen", 0)) + 1
                    runner._birth_fail_gestation_gen = int(getattr(runner, "_birth_fail_gestation_gen", 0)) + 1
                    reserved_tiles.discard(spawn_pos)
                    continue

            t_mc = time.perf_counter()
            child = _make_child(agent, mate, child_energy)
            t_make_child_sum += time.perf_counter() - t_mc
            if child is None:
                _set_agent_energy(aid, slot, e1)
                _set_agent_energy(mate_id, mate_slot, e2)
                failed_births += 1
                runner._birth_make_child_fail_gen = int(getattr(runner, "_birth_make_child_fail_gen", 0)) + 1
                reserved_tiles.discard(spawn_pos)
                continue

            child.injury = 0.0
            child.scar = 0.0
            child.rest_streak = 0
            births.append((child, spawn_pos, agent))

            try:
                agent.stats.offspring_count += 1
                state.offspring_count[slot] = np.int32(int(agent.stats.offspring_count))
                if float(getattr(agent.stats, "offspring_count", 0.0)) > 0.0 and getattr(
                    agent.stats,
                    "first_repro_step",
                    None,
                ) is None:
                    agent.stats.first_repro_step = int(getattr(agent.stats, "survival_steps", 0))
                    state.first_repro_step[slot] = agent.stats.first_repro_step
            except Exception:
                pass
            try:
                mate.stats.offspring_count += 1
                state.offspring_count[mate_slot] = np.int32(int(mate.stats.offspring_count))
                if float(getattr(mate.stats, "offspring_count", 0.0)) > 0.0 and getattr(
                    mate.stats,
                    "first_repro_step",
                    None,
                ) is None:
                    mate.stats.first_repro_step = int(getattr(mate.stats, "survival_steps", 0))
                    state.first_repro_step[mate_slot] = mate.stats.first_repro_step
            except Exception:
                pass

            _set_agent_cooldown(aid, slot, cd)
            _set_agent_cooldown(mate_id, mate_slot, cd)
            mated_this_tick.add(aid)
            mated_this_tick.add(mate_id)
            changed_parent_ids.add(aid)
            changed_parent_ids.add(mate_id)
    finally:
        if old_typed_rows is not None:
            setattr(runner.sim_cfg, "runnercore11_c1_typed_rows_enable", old_typed_rows)

    if isinstance(buckets, dict):
        loop_dt = time.perf_counter() - t_birth_scan0
        buckets["birth_scan_ms"] = buckets.get("birth_scan_ms", 0.0) + loop_dt
        buckets["birth_make_child_ms"] = buckets.get("birth_make_child_ms", 0.0) + t_make_child_sum
        buckets["birth_scan_excl_make_child_ms"] = buckets.get("birth_scan_excl_make_child_ms", 0.0) + max(
            0.0,
            loop_dt - t_make_child_sum,
        )

    t_birth0 = time.perf_counter()
    newborns: list[Any] = []
    runnercore42_patch_context_cache: dict[str, Any] | None = None
    if (
        births
        and bool(getattr(env, "patch_world_enable", False))
        and _runnercore42_patch_birth_context_cache_enabled(runner)
    ):
        try:
            runnercore42_patch_context_cache = _runnercore42_prepare_patch_context_cache(env)
        except Exception as exc:
            raise RunnerCore1UnsupportedError(
                f"RUNNERCORE42-UNSUPPORTED: patch birth context cache unavailable: {exc}"
            ) from exc
    for child, pos, parent in births:
        try:
            env.add_agent(int(child.agent_id), pos=pos)
            successful_births += 1
            newborns.append(child)
            newborn_birth_rows.append((child, pos, parent))
            if runnercore25_birth_row_spec_bridge:
                try:
                    runnercore25_successful_ids.append(int(child.agent_id))
                except Exception:
                    pass
            if bool(getattr(env, "patch_world_enable", False)):
                runnercore42_birth_context = None
                if runnercore42_patch_context_cache is not None:
                    try:
                        _runnercore42_update_density_counts_for_birth(
                            runnercore42_patch_context_cache,
                            int(pos[0]),
                            int(pos[1]),
                        )
                        if not hasattr(runner, "_patch_birth_contexts_gen"):
                            runner._patch_birth_contexts_gen = []
                        runnercore42_birth_context = _runnercore42_context_from_cache(
                            runnercore42_patch_context_cache,
                            int(pos[0]),
                            int(pos[1]),
                            agent=child,
                        )
                        runner._patch_birth_contexts_gen.append(runnercore42_birth_context)
                        _runnercore34_stat_add(runner, "runnercore42_patch_birth_context_cache_rows", 1)
                    except Exception as exc:
                        raise RunnerCore1UnsupportedError(
                            f"RUNNERCORE42-UNSUPPORTED: patch birth context cache failed: {exc}"
                        ) from exc
                else:
                    try:
                        from env.gridworld_pkg.patch_world import sample_patch_context  # noqa: WPS433

                        if not hasattr(runner, "_patch_birth_contexts_gen"):
                            runner._patch_birth_contexts_gen = []
                        runner._patch_birth_contexts_gen.append(
                            sample_patch_context(
                                env,
                                int(pos[0]),
                                int(pos[1]),
                                agent=child,
                            )
                        )
                    except Exception:
                        pass
                try:
                    if runnercore42_patch_context_cache is not None:
                        queue_jungle_birth_emission = runnercore42_patch_context_cache[
                            "queue_jungle_birth_emission"
                        ]
                    else:
                        from env.gridworld_pkg.jungle import queue_jungle_birth_emission  # noqa: WPS433

                    scent_emit, waste_emit = queue_jungle_birth_emission(env, int(pos[0]), int(pos[1]))
                    if (scent_emit > 0.0) or (waste_emit > 0.0):
                        runner._jungle_birth_emission_count_gen = int(
                            getattr(runner, "_jungle_birth_emission_count_gen", 0)
                        ) + 1
                        runner._jungle_birth_scent_emitted_total_gen = float(
                            getattr(runner, "_jungle_birth_scent_emitted_total_gen", 0.0)
                        ) + float(scent_emit)
                        runner._jungle_birth_waste_emitted_total_gen = float(
                            getattr(runner, "_jungle_birth_waste_emitted_total_gen", 0.0)
                        ) + float(waste_emit)
                        if not hasattr(runner, "_jungle_birth_emission_contexts_gen"):
                            runner._jungle_birth_emission_contexts_gen = []
                        if runnercore42_patch_context_cache is not None:
                            if runnercore42_birth_context is None:
                                runnercore42_birth_context = _runnercore42_context_from_cache(
                                    runnercore42_patch_context_cache,
                                    int(pos[0]),
                                    int(pos[1]),
                                    agent=child,
                                )
                            runner._jungle_birth_emission_contexts_gen.append(dict(runnercore42_birth_context))
                            _runnercore34_stat_add(runner, "runnercore42_jungle_birth_context_cache_rows", 1)
                        else:
                            runner._jungle_birth_emission_contexts_gen.append(
                                sample_patch_context(env, int(pos[0]), int(pos[1]), agent=child)
                            )
                except Exception:
                    pass
            try:
                runner.behavior_logger.register_agent(
                    int(child.agent_id),
                    cohort="born",
                    injuries={int(child.agent_id): float(getattr(child, "injury", 0.0))},
                )
            except Exception:
                pass
        except ValueError:
            failed_births += 1
            runner._birth_fail_no_space_gen = int(getattr(runner, "_birth_fail_no_space_gen", 0)) + 1
            child.alive = False
            pop.agents.pop(int(child.agent_id), None)

    if runnercore42_patch_context_cache is not None:
        _runnercore42_finalize_patch_context_cache(runnercore42_patch_context_cache, env)

    if newborns:
        runner._sync_bank_for_agents(newborns, buckets=buckets)
        if hasattr(runner, "_sync_gru_bank_for_agents"):
            runner._sync_gru_bank_for_agents(newborns, buckets=buckets)

    if runnercore25_birth_row_spec_bridge:
        try:
            runner._runnercore25_full_birth_row_spec_newborn_ids = sorted(
                int(aid) for aid in runnercore25_successful_ids
            )
        except Exception:
            runner._runnercore25_full_birth_row_spec_newborn_ids = []
        _runnercore6_stat_add(runner, "runnercore25_full_birth_row_specs_applied", len(runnercore25_successful_ids))

    buckets["births_apply"] = buckets.get("births_apply", 0.0) + (time.perf_counter() - t_birth0)
    buckets["reproduction_ms"] = buckets.get("reproduction_ms", 0.0) + (time.perf_counter() - t_repro0)
    try:
        runner._runnercore4_birth_changed_parent_ids = sorted(int(aid) for aid in changed_parent_ids)
    except Exception:
        pass
    if _runnercore3_profile_enabled(runner):
        runnercore3_profile_count(state, "runnercore34_full_birth_resident_decision_rows", considered_rows)
        runnercore3_profile_count(state, "runnercore34_full_birth_mate_selected_rows", mate_selected_rows)
        runnercore3_profile_count(state, "runnercore34_full_birth_spawn_selected_rows", spawn_selected_rows)
        runnercore3_profile_count(state, "runnercore34_full_birth_parent_export_rows", parent_export_rows)
        runnercore3_profile_count(
            state,
            "runnercore34_full_birth_candidate_exports_avoided",
            max(0, int(len(candidate_ids)) - int(parent_export_rows)),
        )
    return int(successful_births), int(failed_births)


def _runnercore2_handle_births(
    state: RunnerCore1TickState,
    runner: Any,
    agent_ids: np.ndarray,
    successful_births: int,
    failed_births: int,
    buckets: dict[str, float],
    *,
    gen: int,
    tick: int,
) -> tuple[int, int]:
    _runnercore2_ensure_birth_counters(runner)
    profile_enabled = _runnercore3_profile_enabled(runner)
    full_birth_bridge = _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "runnercore22_full_birth_boundary_bridge_enable", False)
    )
    t_precheck = time.perf_counter() if profile_enabled else 0.0
    if full_birth_bridge:
        if _as_float(getattr(runner.env_cfg, "reproduce_try_prob", 0.0), 0.0) <= 0.0:
            _runnercore2_apply_no_boundary_birth_prechecks(state, runner, agent_ids)
            candidate_ids = []
        else:
            cooldown_prefilter = _as_bool(
                getattr(
                    getattr(runner, "sim_cfg", None),
                    "runnercore27_full_birth_cooldown_prefilter_enable",
                    False,
                )
            )
            if cooldown_prefilter and _as_bool(
                getattr(getattr(runner, "sim_cfg", None), "lineage_reproduction_funnel_write_events", False)
            ):
                raise RunnerCore1UnsupportedError(
                    "RUNNERCORE27-UNSUPPORTED: cooldown prefilter cannot skip funnel event contexts"
                )
            candidate_ids = []
            prefilter_rows = 0
            considered_rows = 0
            for aid_raw in sorted(int(aid) for aid in agent_ids.tolist()):
                aid = int(aid_raw)
                slot = state.slot_for_agent_id(aid)
                if slot < 0 or not bool(state.alive[slot]):
                    continue
                considered_rows += 1
                cooldown = int(state.repro_cooldown[slot])
                cooldown_after = max(0, int(cooldown) - 1)
                if cooldown_prefilter and cooldown_after != 0:
                    state.repro_cooldown[slot] = np.int32(int(cooldown_after))
                    runner._birth_pre_cooldown_gen = int(getattr(runner, "_birth_pre_cooldown_gen", 0)) + 1
                    _runnercore34_apply_dev_expression_side_effect_for_row(
                        state,
                        runner,
                        aid,
                        source="cooldown_prefilter",
                    )
                    prefilter_rows += 1
                    continue
                candidate_ids.append(aid)
            if cooldown_prefilter and profile_enabled:
                runnercore3_profile_count(
                    state,
                    "runnercore27_full_birth_cooldown_prefilter_considered_rows",
                    considered_rows,
                )
                runnercore3_profile_count(state, "runnercore27_full_birth_cooldown_prefilter_rows", prefilter_rows)
            candidate_ids = _runnercore30_full_birth_branch_prefilter(state, runner, candidate_ids)
    else:
        candidate_ids = _runnercore2_birth_candidate_ids(state, runner, agent_ids)
    if profile_enabled:
        runnercore3_profile_add(state, "birth_precheck_ms", time.perf_counter() - t_precheck)
        runnercore3_profile_count(state, "birth_candidates_considered_for_export", len(candidate_ids))
    if not candidate_ids:
        return int(successful_births), int(failed_births)

    runnercore34_resident_decision = bool(
        full_birth_bridge
        and _runnercore34_full_birth_resident_decision_enabled(runner)
        and _as_bool(getattr(getattr(runner, "env_cfg", None), "reproduce_requires_mate", False))
    )
    before_all_ids = set(int(aid) for aid in getattr(getattr(runner, "pop", None), "agents", {}).keys())
    before_candidate_ids = set(int(aid) for aid in candidate_ids)
    resident_parent_specs: dict[int, Any] = {}
    export_candidate_ids = list(candidate_ids)
    if full_birth_bridge:
        before_candidate_ids = set(int(aid) for aid in candidate_ids)
        export_candidate_ids = [] if runnercore34_resident_decision else sorted(before_candidate_ids)
        if profile_enabled:
            runnercore3_profile_count(state, "runnercore22_full_birth_boundary_bridge_export_rows", len(export_candidate_ids))
    if (not runnercore34_resident_decision) and _runnercore17_resident_birth_parent_enabled(runner):
        r18_minimal_parent_refresh = _runnercore18_resident_parent_minimal_refresh_enabled(runner)
        _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_candidates", len(candidate_ids))
        export_candidate_ids = []
        agents_for_split = getattr(getattr(runner, "pop", None), "agents", {})
        for aid in sorted(before_candidate_ids):
            agent = agents_for_split.get(int(aid)) if isinstance(agents_for_split, dict) else None
            if agent is not None:
                export_candidate_ids.append(int(aid))
                continue
            spec = _runnercore17_resident_parent_spec(state, runner, int(aid))
            if spec is not None and _runnercore17_refresh_resident_parent_spec_from_state(
                state,
                runner,
                spec,
                full=not r18_minimal_parent_refresh,
            ):
                resident_parent_specs[int(aid)] = spec
                continue
            _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_missing_specs", 1)
            _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_unsupported_fallback_count", 1)
            state.counters.unsupported_fallback_count += 1
            if _runnercore17_resident_birth_parent_fail_closed(runner):
                raise RunnerCore1UnsupportedError(
                    "RUNNERCORE17-UNSUPPORTED: resident typed birth parent missing or stale"
                )
            export_candidate_ids.append(int(aid))
        if resident_parent_specs:
            ids = sorted(int(aid) for aid in resident_parent_specs)
            try:
                runner._runnercore17_resident_birth_parent_ids = ids
            except Exception:
                pass
            _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_specs_used", len(ids))
            _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_materializations_avoided", len(ids))
            _runnercore10_stat_add(runner, "event_boundary_export_rows_avoided", len(ids))
            _runnercore11_stat_add(runner, "compatibility_export_avoided", len(ids))
        else:
            try:
                runner._runnercore17_resident_birth_parent_ids = []
            except Exception:
                pass
    else:
        try:
            runner._runnercore17_resident_birth_parent_ids = []
        except Exception:
            pass
    t_export = time.perf_counter() if profile_enabled else 0.0
    if export_candidate_ids:
        export_to_agents(state, runner, boundary_reason="runnercore2_birth_event_boundary", agent_ids=export_candidate_ids)
    if profile_enabled:
        runnercore3_profile_add(state, "event_boundary_export_ms", time.perf_counter() - t_export)
        runnercore3_profile_count(state, "birth_candidates_exported", len(export_candidate_ids))
        runnercore3_profile_count(state, "birth_candidates_resident_parent_specs", len(resident_parent_specs))

    alive_for_births = []
    agents = getattr(getattr(runner, "pop", None), "agents", {})
    if isinstance(agents, dict):
        for aid in sorted(before_candidate_ids):
            agent = agents.get(int(aid))
            if agent is not None and bool(getattr(agent, "alive", False)):
                alive_for_births.append(agent)
                continue
            spec = resident_parent_specs.get(int(aid))
            if spec is not None and bool(getattr(spec, "alive", False)):
                alive_for_births.append(spec)
    runnercore22_parent_before: dict[int, tuple[Any, ...]] = {}
    if full_birth_bridge and isinstance(agents, dict):
        for aid in sorted(before_candidate_ids):
            agent = agents.get(int(aid))
            if agent is not None and bool(getattr(agent, "alive", False)):
                runnercore22_parent_before[int(aid)] = _runnercore22_parent_import_snapshot(agent)

    old_tick = getattr(runner, "_tick", None)
    runner._tick = int(tick)
    t_child = time.perf_counter() if profile_enabled else 0.0
    bank_sync_before = float(buckets.get("bank_sync", 0.0)) + float(buckets.get("gru_bank_sync", 0.0)) if profile_enabled else 0.0
    runnercore4_bucket_keys = (
        "make_child_alloc_slot",
        "make_child_genome",
        "make_child_policy",
        "make_child_row_spec_setup",
        "runnercore4_child_payload_build",
        "runnercore4_c2_self_payload",
        "runnercore4_c1_operator_pipeline",
        "runnercore4_genome_from_c1_payload",
        "runnercore5_self_policy_flat_rng",
        "runnercore7_mlp_policy_init_rng_replay",
        "runnercore7_self_policy_source_alias",
        "runnercore7_gru_policy_init_rng_replay",
        "runnercore5_direct_mlp_bank_attach",
        "runnercore7_direct_gru_bank_attach",
        "runnercore6_resident_newborn_attach",
    )
    runnercore4_before = (
        {name: float(buckets.get(name, 0.0) or 0.0) for name in runnercore4_bucket_keys}
        if profile_enabled
        else {}
    )
    try:
        try:
            runner._runnercore4_birth_changed_parent_ids = []
        except Exception:
            pass
        fast_birth_boundary_requested = _as_bool(
            getattr(runner.sim_cfg, "runnercore3_fast_birth_event_boundary_enable", False)
        ) or _as_bool(
            getattr(runner.sim_cfg, "runnercore4_candidate_enable", False)
        ) or _as_bool(
            getattr(runner.sim_cfg, "runnercore5_candidate_enable", False)
        ) or _as_bool(
            getattr(runner.sim_cfg, "runnercore6_candidate_enable", False)
        ) or _as_bool(
            getattr(runner.sim_cfg, "runnercore9_candidate_enable", False)
        ) or _as_bool(
            getattr(runner.sim_cfg, "runnercore10_candidate_enable", False)
        ) or _as_bool(
            getattr(runner.sim_cfg, "runnercore11_candidate_enable", False)
        )
        if fast_birth_boundary_requested and not full_birth_bridge:
            successful_births, failed_births = _runnercore3_handle_no_mate_births_fast_boundary(
                runner,
                state,
                alive_for_births,
                int(successful_births),
                int(failed_births),
                buckets,
            )
        else:
            if runnercore34_resident_decision:
                successful_births, failed_births = _runnercore34_handle_mate_births_resident_decision(
                    runner,
                    state,
                    sorted(int(aid) for aid in before_candidate_ids),
                    int(successful_births),
                    int(failed_births),
                    buckets,
                    gen=int(gen),
                    tick=int(tick),
                )
            else:
                old_typed_rows = getattr(runner.sim_cfg, "runnercore11_c1_typed_rows_enable", None)
                old_r25_active = getattr(runner, "_runnercore25_full_birth_row_spec_bridge_active", None)
                runnercore25_birth_row_spec_bridge = bool(
                    full_birth_bridge
                    and _runnercore6_resident_attach_enabled(runner)
                    and _as_bool(
                        getattr(
                            getattr(runner, "sim_cfg", None),
                            "runnercore25_full_birth_resident_newborn_attach_enable",
                            False,
                        )
                    )
                )
                if full_birth_bridge:
                    setattr(runner.sim_cfg, "runnercore11_c1_typed_rows_enable", False)
                if runnercore25_birth_row_spec_bridge:
                    _runnercore6_stats(runner)
                    try:
                        runner._runnercore25_full_birth_row_spec_newborn_ids = []
                        runner._runnercore25_full_birth_row_spec_bridge_active = True
                    except Exception:
                        pass
                try:
                    successful_births, failed_births = runner._handle_births(
                        alive_for_births,
                        int(successful_births),
                        int(failed_births),
                        buckets,
                    )
                finally:
                    if full_birth_bridge and old_typed_rows is not None:
                        setattr(runner.sim_cfg, "runnercore11_c1_typed_rows_enable", old_typed_rows)
                    if runnercore25_birth_row_spec_bridge:
                        if old_r25_active is None:
                            try:
                                delattr(runner, "_runnercore25_full_birth_row_spec_bridge_active")
                            except Exception:
                                pass
                        else:
                            try:
                                runner._runnercore25_full_birth_row_spec_bridge_active = old_r25_active
                            except Exception:
                                pass
    finally:
        if old_tick is None:
            try:
                delattr(runner, "_tick")
            except Exception:
                pass
        else:
            runner._tick = old_tick
    if profile_enabled:
        runnercore3_profile_add(state, "child_genome_policy_create_ms", time.perf_counter() - t_child)
        for name in runnercore4_bucket_keys:
            delta = float(buckets.get(name, 0.0) or 0.0) - float(runnercore4_before.get(name, 0.0) or 0.0)
            if delta > 0.0:
                runnercore3_profile_add(state, f"{name}_ms", delta)
        bank_sync_after = float(buckets.get("bank_sync", 0.0)) + float(buckets.get("gru_bank_sync", 0.0))
        runnercore3_profile_add(state, "policy_row_attach_ms", max(0.0, bank_sync_after - bank_sync_before))

    t_import = time.perf_counter() if profile_enabled else 0.0
    agents = getattr(getattr(runner, "pop", None), "agents", {})
    after_alive_ids: set[int] = set()
    if isinstance(agents, dict):
        for aid, agent in agents.items():
            if bool(getattr(agent, "alive", False)):
                after_alive_ids.add(int(aid))

    on_demand_ids = {
        int(aid) for aid in list(getattr(runner, "_runnercore9_on_demand_materialized_ids", []) or [])
    }
    try:
        runner._runnercore9_on_demand_materialized_ids = []
    except Exception:
        pass
    new_ids = sorted((after_alive_ids - before_all_ids) - on_demand_ids)
    runnercore25_newborn_ids = {
        int(aid) for aid in list(getattr(runner, "_runnercore25_full_birth_row_spec_newborn_ids", []) or [])
    }
    if runnercore25_newborn_ids and profile_enabled:
        runnercore3_profile_count(state, "runnercore25_full_birth_row_specs_created", len(runnercore25_newborn_ids))
        runnercore3_profile_count(state, "runnercore25_full_birth_row_specs_applied", len(runnercore25_newborn_ids))
    existing_alive_ids = sorted(after_alive_ids & before_candidate_ids)
    sync_existing_ids = existing_alive_ids
    changed_parent_set: set[int] | None = None
    runnercore22_direct_cooldown_rows = 0
    if full_birth_bridge and runnercore34_resident_decision:
        changed_parent_set = {
            int(aid) for aid in list(getattr(runner, "_runnercore4_birth_changed_parent_ids", []) or [])
        }
        sync_existing_ids = []
        avoided = int(len(existing_alive_ids))
        if profile_enabled:
            runnercore3_profile_count(state, "runnercore22_full_birth_boundary_bridge_import_rows", 0)
            runnercore3_profile_count(
                state,
                "runnercore22_full_birth_boundary_bridge_import_rows_considered",
                len(existing_alive_ids),
            )
            runnercore3_profile_count(
                state,
                "runnercore22_full_birth_boundary_bridge_import_rows_avoided",
                avoided,
            )
            runnercore3_profile_count(state, "runnercore22_full_birth_boundary_bridge_direct_cooldown_rows", 0)
        stats = getattr(runner.sim_cfg, "_runnercore4_structural_payload_stats", None)
        if isinstance(stats, dict):
            stats["event_boundary_parent_import_rows_considered"] = int(
                stats.get("event_boundary_parent_import_rows_considered", 0) or 0
            ) + int(len(existing_alive_ids))
            stats["event_boundary_parent_import_rows_synced"] = int(
                stats.get("event_boundary_parent_import_rows_synced", 0) or 0
            )
            stats["event_boundary_parent_import_rows_avoided"] = int(
                stats.get("event_boundary_parent_import_rows_avoided", 0) or 0
            ) + int(avoided)
    elif full_birth_bridge:
        changed_parent_set = set()
        if isinstance(agents, dict):
            for aid in existing_alive_ids:
                agent = agents.get(int(aid))
                if agent is None:
                    changed_parent_set.add(int(aid))
                    continue
                before_snapshot = runnercore22_parent_before.get(int(aid))
                after_snapshot = _runnercore22_parent_import_snapshot(agent)
                if before_snapshot is None or after_snapshot != before_snapshot:
                    if (
                        before_snapshot is not None
                        and _runnercore22_only_repro_cooldown_changed(before_snapshot, after_snapshot)
                    ):
                        slot = state.slot_for_agent_id(int(aid))
                        if slot >= 0 and bool(state.alive[slot]):
                            state.repro_cooldown[slot] = np.int32(int(after_snapshot[6]))
                            runnercore22_direct_cooldown_rows += 1
                            continue
                    changed_parent_set.add(int(aid))
        else:
            changed_parent_set = set(existing_alive_ids)
        sync_existing_ids = sorted(int(aid) for aid in changed_parent_set)
        avoided = max(0, int(len(existing_alive_ids)) - int(len(sync_existing_ids)))
        if profile_enabled:
            runnercore3_profile_count(state, "runnercore22_full_birth_boundary_bridge_import_rows", len(sync_existing_ids))
            runnercore3_profile_count(
                state,
                "runnercore22_full_birth_boundary_bridge_import_rows_considered",
                len(existing_alive_ids),
            )
            runnercore3_profile_count(
                state,
                "runnercore22_full_birth_boundary_bridge_import_rows_avoided",
                avoided,
            )
            runnercore3_profile_count(
                state,
                "runnercore22_full_birth_boundary_bridge_direct_cooldown_rows",
                runnercore22_direct_cooldown_rows,
            )
        stats = getattr(runner.sim_cfg, "_runnercore4_structural_payload_stats", None)
        if isinstance(stats, dict):
            stats["event_boundary_parent_import_rows_considered"] = int(
                stats.get("event_boundary_parent_import_rows_considered", 0) or 0
            ) + int(len(existing_alive_ids))
            stats["event_boundary_parent_import_rows_synced"] = int(
                stats.get("event_boundary_parent_import_rows_synced", 0) or 0
            ) + int(len(sync_existing_ids))
            stats["event_boundary_parent_import_rows_avoided"] = int(
                stats.get("event_boundary_parent_import_rows_avoided", 0) or 0
            ) + int(avoided)
    elif _as_bool(getattr(runner.sim_cfg, "runnercore4_changed_parent_import_enable", False)):
        changed_parent_ids = getattr(runner, "_runnercore4_birth_changed_parent_ids", [])
        changed_parent_set = {int(aid) for aid in list(changed_parent_ids or [])}
        sync_existing_ids = sorted(after_alive_ids & before_candidate_ids & changed_parent_set)
        avoided = max(0, int(len(existing_alive_ids)) - int(len(sync_existing_ids)))
        stats = getattr(runner.sim_cfg, "_runnercore4_structural_payload_stats", None)
        if isinstance(stats, dict):
            stats["event_boundary_parent_import_rows_considered"] = int(
                stats.get("event_boundary_parent_import_rows_considered", 0) or 0
            ) + int(len(existing_alive_ids))
            stats["event_boundary_parent_import_rows_synced"] = int(
                stats.get("event_boundary_parent_import_rows_synced", 0) or 0
            ) + int(len(sync_existing_ids))
            stats["event_boundary_parent_import_rows_avoided"] = int(
                stats.get("event_boundary_parent_import_rows_avoided", 0) or 0
            ) + int(avoided)
    for aid in sync_existing_ids:
        agent = agents.get(int(aid)) if isinstance(agents, dict) else None
        if agent is not None:
            sync_agent_row_from_mirror(state, runner, agent, is_new=False)

    r17_parent_ids = {
        int(aid) for aid in list(getattr(runner, "_runnercore17_resident_birth_parent_ids", []) or [])
    }
    if r17_parent_ids:
        specs = _runnercore17_typed_child_specs(runner)
        if changed_parent_set is None:
            sync_resident_parent_ids = sorted(r17_parent_ids & before_candidate_ids)
        else:
            sync_resident_parent_ids = sorted(r17_parent_ids & before_candidate_ids & changed_parent_set)
        for aid in sync_resident_parent_ids:
            spec = specs.get(int(aid))
            if spec is None:
                _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_unsupported_fallback_count", 1)
                state.counters.unsupported_fallback_count += 1
                if _runnercore17_resident_birth_parent_fail_closed(runner):
                    raise RunnerCore1UnsupportedError("RUNNERCORE17-UNSUPPORTED: resident parent spec vanished before sync")
                continue
            if (
                _runnercore18_resident_parent_minimal_refresh_enabled(runner)
                and bool(getattr(spec, "_runnercore18_minimal_parent_refreshed", False))
                and not bool(getattr(spec, "_runnercore18_full_parent_refreshed", False))
            ):
                synced_slot = _runnercore18_sync_minimal_resident_parent_to_state(state, runner, spec)
            else:
                synced_slot = sync_agent_row_from_mirror(state, runner, spec, is_new=False)
            if synced_slot < 0:
                _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_unsupported_fallback_count", 1)
                _runnercore18_stat_add(runner, "runnercore18_resident_parent_unsupported_fallback_count", 1)
                state.counters.unsupported_fallback_count += 1
                if _runnercore17_resident_birth_parent_fail_closed(runner):
                    raise RunnerCore1UnsupportedError("RUNNERCORE17-UNSUPPORTED: resident parent sync failed")
                continue
            _runnercore17_stat_add(runner, "runnercore17_resident_birth_parent_parent_import_syncs", 1)
        try:
            runner._runnercore17_resident_birth_parent_ids = []
        except Exception:
            pass

    runnercore6_attached_ids: set[int] = {
        int(aid) for aid in list(getattr(runner, "_runnercore6_resident_attached_newborn_ids", []) or [])
    }
    if _runnercore6_resident_attach_enabled(runner) and not runnercore6_attached_ids:
        t_r6_attach = time.perf_counter()
        _runnercore6_stat_add(runner, "resident_newborn_attach_attempts", 1)
        _runnercore6_stat_add(runner, "resident_newborn_attach_candidate_rows", len(new_ids))
        for aid in new_ids:
            agent = agents.get(int(aid)) if isinstance(agents, dict) else None
            if agent is None:
                _runnercore6_stat_add(runner, "resident_newborn_attach_missing_agent_rows", 1)
                if _as_bool(getattr(runner.sim_cfg, "runnercore6_fail_closed_on_unsupported", False)):
                    raise RunnerCore1UnsupportedError("RUNNERCORE6-UNSUPPORTED: missing newborn agent for resident attach")
                continue
            slot = sync_newborn_row_from_contract(state, runner, agent)
            if slot < 0:
                _runnercore6_stat_add(runner, "resident_newborn_attach_unsupported_rows", 1)
                if _as_bool(getattr(runner.sim_cfg, "runnercore6_fail_closed_on_unsupported", False)):
                    raise RunnerCore1UnsupportedError("RUNNERCORE6-UNSUPPORTED: newborn resident row attach failed")
                continue
            runnercore6_attached_ids.add(int(aid))
            _runnercore6_stat_add(runner, "resident_newborn_attach_rows", 1)
            _runnercore6_stat_add(runner, "generic_newborn_import_rows_avoided", 1)
            if int(aid) in runnercore25_newborn_ids:
                _runnercore6_stat_add(runner, "runnercore25_full_birth_newborn_attach_rows", 1)
                if profile_enabled:
                    runnercore3_profile_count(state, "runnercore25_full_birth_newborn_attach_rows", 1)
            state.counters.birth_events_applied += 1
            state.counters.event_boundary_agent_create_rows += 1
            state.counters.event_boundary_policy_bank_sync_rows += 1
            state.counters.policy_row_attach_events += 1
            if bool(state.use_gru[int(slot)]):
                state.counters.hmem_row_init_events += 1
            record_churn_event(
                state,
                event_type="birth",
                generation=int(gen),
                tick=int(tick),
                slot=int(slot),
                agent_id=int(aid),
                reason="resident_newborn_attach",
                slot_token=int(state.slot_tokens[int(slot)]),
            )
        elapsed = time.perf_counter() - t_r6_attach
        buckets["runnercore6_resident_newborn_attach"] = float(
            buckets.get("runnercore6_resident_newborn_attach", 0.0) or 0.0
        ) + float(elapsed)

    for aid in new_ids:
        if int(aid) in runnercore6_attached_ids:
            continue
        agent = agents.get(int(aid)) if isinstance(agents, dict) else None
        if agent is None:
            continue
        slot = sync_agent_row_from_mirror(state, runner, agent, is_new=True)
        if slot < 0:
            continue
        state.counters.birth_events_applied += 1
        state.counters.event_boundary_agent_create_rows += 1
        state.counters.event_boundary_policy_bank_sync_rows += 1
        state.counters.policy_row_attach_events += 1
        if bool(state.use_gru[int(slot)]):
            state.counters.hmem_row_init_events += 1
        record_churn_event(
            state,
            event_type="birth",
            generation=int(gen),
            tick=int(tick),
            slot=int(slot),
            agent_id=int(aid),
            reason="birth_event_boundary",
            slot_token=int(state.slot_tokens[int(slot)]),
        )

    if profile_enabled:
        runnercore3_profile_add(state, "event_boundary_import_ms", time.perf_counter() - t_import)
    return int(successful_births), int(failed_births)


def _runnercore24_apply_native_jungle_predator_pressure(
    state: RunnerCore1TickState,
    runner: Any,
    deaths_this_gen: int,
) -> int:
    """Apply Jungle predator pressure directly to resident arrays.

    The loop intentionally preserves legacy draw order: sorted alive agent id,
    one env RNG draw for each occupied positive-risk cell.
    """

    try:
        from env.gridworld_pkg.jungle import (
            compute_predator_risk_maps,
            compute_terrain_damage_multiplier_map,
            jungle_predator_pressure_enabled,
        )
        from env.gridworld_pkg.patch_world import _bounded, sample_patch_context
    except Exception:
        return int(deaths_this_gen)

    cfg = getattr(runner, "env_cfg", None)
    if not jungle_predator_pressure_enabled(cfg):
        return int(deaths_this_gen)

    alive_ids = alive_agent_ids(state, runner)
    if int(alive_ids.size) <= 0:
        return int(deaths_this_gen)

    env = getattr(runner, "env", None)
    rng = getattr(env, "rng", None)
    if env is None:
        return int(deaths_this_gen)

    risk = np.asarray(compute_predator_risk_maps(env)["risk"], dtype=np.float32)
    terrain_mult_map = np.asarray(compute_terrain_damage_multiplier_map(env), dtype=np.float32)
    pos_y = np.asarray(getattr(env, "agent_pos_y"), dtype=np.int32)
    pos_x = np.asarray(getattr(env, "agent_pos_x"), dtype=np.int32)
    damage = max(0.0, _as_float(getattr(cfg, "jungle_predator_health_damage", 0.25), 0.25))
    profile_enabled = _runnercore3_profile_enabled(runner)

    runner._jungle_predator_exposure_count_gen = int(
        getattr(runner, "_jungle_predator_exposure_count_gen", 0)
    ) + int(alive_ids.size)
    if profile_enabled:
        runnercore3_profile_count(
            state,
            "runnercore24_jungle_predator_native_considered_rows",
            int(alive_ids.size),
        )
    if rng is None:
        return int(deaths_this_gen)

    agents = getattr(getattr(runner, "pop", None), "agents", {})
    juvenile_threshold = max(0, _as_int(getattr(cfg, "jungle_juvenile_age_threshold", 10), 10))
    juvenile_vulnerability = bool(
        _as_bool(getattr(cfg, "patch_world_enable", False))
        and _as_bool(getattr(cfg, "jungle_predator_pressure_enable", False))
        and _as_bool(getattr(cfg, "jungle_juvenile_predator_vulnerability_enable", False))
    )
    juvenile_raw_mult = _bounded(
        getattr(cfg, "jungle_juvenile_predator_damage_multiplier", 1.5),
        default=1.5,
        lo=1.0,
    )
    juvenile_cap = _bounded(getattr(cfg, "jungle_juvenile_vulnerability_cap", 3.0), default=3.0, lo=1.0)
    juvenile_damage_mult = float(min(float(juvenile_raw_mult), float(juvenile_cap)))
    event_rows = 0
    death_rows = 0
    mirror_sync_rows = 0
    context_cache_rows = 0
    vectorized_sampling = bool(
        _as_bool(getattr(runner.sim_cfg, "runnercore26_vectorized_jungle_predator_sampling_enable", False))
    )
    context_cache: dict[str, Any] | None = None
    if _runnercore45_jungle_predator_context_cache_enabled(runner) and bool(getattr(env, "patch_world_enable", False)):
        try:
            context_cache = _runnercore42_prepare_patch_context_cache(env)
        except Exception as exc:
            raise RunnerCore1UnsupportedError(
                f"RUNNERCORE45-UNSUPPORTED: jungle predator context cache unavailable: {exc}"
            ) from exc

    def _sample_predator_context(y: int, x: int, agent: Any, aid: int) -> dict[str, float]:
        nonlocal context_cache_rows
        if context_cache is None:
            return sample_patch_context(env, y, x, agent=agent, lineage_id=(aid if agent is None else None))
        context_cache_rows += 1
        return _runnercore42_context_from_cache(
            context_cache,
            int(y),
            int(x),
            agent=agent,
            lineage_id=(aid if agent is None else None),
        )

    def _apply_predator_event(aid: int, slot: int, y: int, x: int) -> None:
        nonlocal deaths_this_gen, event_rows, death_rows, mirror_sync_rows

        age = int(state.survival_steps[slot])
        is_juvenile = bool(age <= int(juvenile_threshold))
        juvenile_mult = float(juvenile_damage_mult if (is_juvenile and juvenile_vulnerability) else 1.0)
        terrain_mult = float(terrain_mult_map[y, x])
        raw_event_damage = float(damage) * float(juvenile_mult)
        event_damage = float(raw_event_damage) * float(terrain_mult)
        damage_prevented = max(0.0, float(raw_event_damage) - float(event_damage))
        event_rows += 1

        runner._jungle_predator_events_gen = int(getattr(runner, "_jungle_predator_events_gen", 0)) + 1
        runner._jungle_predator_damage_total_gen = float(
            getattr(runner, "_jungle_predator_damage_total_gen", 0.0)
        ) + float(event_damage)
        runner._jungle_predator_damage_raw_total_gen = float(
            getattr(runner, "_jungle_predator_damage_raw_total_gen", 0.0)
        ) + float(raw_event_damage)
        runner._jungle_predator_damage_effective_total_gen = float(
            getattr(runner, "_jungle_predator_damage_effective_total_gen", 0.0)
        ) + float(event_damage)
        runner._jungle_predator_damage_prevented_total_gen = float(
            getattr(runner, "_jungle_predator_damage_prevented_total_gen", 0.0)
        ) + float(damage_prevented)
        if not hasattr(runner, "_jungle_predator_damage_multipliers_gen"):
            runner._jungle_predator_damage_multipliers_gen = []
        runner._jungle_predator_damage_multipliers_gen.append(float(terrain_mult))
        if is_juvenile:
            runner._jungle_juvenile_predator_events_gen = int(
                getattr(runner, "_jungle_juvenile_predator_events_gen", 0)
            ) + 1
            runner._jungle_juvenile_predator_damage_total_gen = float(
                getattr(runner, "_jungle_juvenile_predator_damage_total_gen", 0.0)
            ) + float(event_damage)
        else:
            runner._jungle_adult_predator_events_gen = int(
                getattr(runner, "_jungle_adult_predator_events_gen", 0)
            ) + 1
            runner._jungle_adult_predator_damage_total_gen = float(
                getattr(runner, "_jungle_adult_predator_damage_total_gen", 0.0)
            ) + float(event_damage)

        agent = agents.get(aid) if isinstance(agents, dict) else None
        try:
            if not hasattr(runner, "_jungle_predator_event_contexts_gen"):
                runner._jungle_predator_event_contexts_gen = []
            ctx = _sample_predator_context(y, x, agent, aid)
            ctx["damage_multiplier"] = float(terrain_mult)
            ctx["damage_prevented"] = float(damage_prevented)
            ctx["raw_damage"] = float(raw_event_damage)
            ctx["effective_damage"] = float(event_damage)
            runner._jungle_predator_event_contexts_gen.append(ctx)
        except Exception:
            pass

        h0 = float(state.health[slot])
        h1 = max(0.0, h0 - float(event_damage))
        state.health[slot] = np.float32(h1)
        state.last_health_delta[slot] = np.float32(float(state.last_health_delta[slot]) - float(event_damage))
        if agent is not None:
            agent.health = float(h1)
            try:
                st = getattr(agent, "stats", None)
                if st is not None:
                    st.last_health_delta = float(getattr(st, "last_health_delta", 0.0)) - float(event_damage)
            except Exception:
                pass
            mirror_sync_rows += 1

        if h1 <= 0.0 and bool(state.alive[slot]):
            if bool(getattr(env, "patch_world_enable", False)):
                try:
                    if not hasattr(runner, "_patch_death_contexts_gen"):
                        runner._patch_death_contexts_gen = []
                    runner._patch_death_contexts_gen.append(
                        _sample_predator_context(y, x, agent, aid)
                    )
                except Exception:
                    pass
            state.alive[slot] = False
            if agent is not None:
                agent.alive = False
            runner._jungle_predator_deaths_gen = int(getattr(runner, "_jungle_predator_deaths_gen", 0)) + 1
            if is_juvenile:
                runner._jungle_juvenile_predator_deaths_gen = int(
                    getattr(runner, "_jungle_juvenile_predator_deaths_gen", 0)
                ) + 1
            deaths_this_gen = int(deaths_this_gen) + 1
            death_rows += 1
            state.counters.death_events_applied += 1
            invalidate_action_cache_slot(state, slot, reason="jungle_predator_death", agent_id=aid)
            record_churn_event(
                state,
                event_type="death",
                generation=int(getattr(runner, "gen", 0)),
                tick=int(getattr(runner, "tick", 0)),
                slot=int(slot),
                agent_id=int(aid),
                reason="jungle_predator",
            )
            try:
                env.remove_agent(aid)
            except Exception:
                pass
            if context_cache is not None:
                _runnercore45_update_density_counts_for_death(context_cache, y, x)

    if vectorized_sampling:
        sample_aids: list[int] = []
        sample_slots: list[int] = []
        sample_y: list[int] = []
        sample_x: list[int] = []
        sample_prob: list[float] = []
        for aid_raw in alive_ids.tolist():
            aid = int(aid_raw)
            slot = state.slot_for_agent_id(aid)
            if slot < 0 or not bool(state.alive[slot]):
                continue
            if aid < 0 or aid >= int(pos_y.shape[0]) or aid >= int(pos_x.shape[0]):
                continue
            y = int(pos_y[aid])
            x = int(pos_x[aid])
            if y < 0 or x < 0:
                continue
            p = float(risk[y, x])
            if p <= 0.0:
                continue
            sample_aids.append(aid)
            sample_slots.append(int(slot))
            sample_y.append(y)
            sample_x.append(x)
            sample_prob.append(p)

        event_indices: np.ndarray = np.empty(0, dtype=np.int64)
        if sample_prob:
            try:
                draws = np.asarray(rng.random(len(sample_prob)), dtype=np.float64)
            except Exception as exc:
                raise RunnerCore1UnsupportedError(
                    "RUNNERCORE26-UNSUPPORTED: env rng does not support vectorized float64 draws"
                ) from exc
            if int(draws.size) != int(len(sample_prob)):
                raise RunnerCore1UnsupportedError("RUNNERCORE26-UNSUPPORTED: vectorized rng draw count mismatch")
            event_indices = np.nonzero(draws < np.asarray(sample_prob, dtype=np.float64))[0]
            for idx_raw in event_indices.tolist():
                idx = int(idx_raw)
                _apply_predator_event(
                    int(sample_aids[idx]),
                    int(sample_slots[idx]),
                    int(sample_y[idx]),
                    int(sample_x[idx]),
                )

        if profile_enabled:
            runnercore3_profile_count(
                state,
                "runnercore26_jungle_predator_vectorized_sample_rows",
                int(len(sample_prob)),
            )
            runnercore3_profile_count(
                state,
                "runnercore26_jungle_predator_vectorized_event_rows",
                int(event_indices.size),
            )
            runnercore3_profile_count(state, "runnercore24_jungle_predator_native_event_rows", int(event_rows))
            runnercore3_profile_count(state, "runnercore24_jungle_predator_native_death_rows", int(death_rows))
            runnercore3_profile_count(state, "runnercore45_jungle_predator_context_cache_rows", int(context_cache_rows))
            runnercore3_profile_count(
                state,
                "runnercore24_jungle_predator_native_mirror_sync_rows",
                int(mirror_sync_rows),
            )
        return int(deaths_this_gen)

    for aid_raw in alive_ids.tolist():
        aid = int(aid_raw)
        slot = state.slot_for_agent_id(aid)
        if slot < 0 or not bool(state.alive[slot]):
            continue
        if aid < 0 or aid >= int(pos_y.shape[0]) or aid >= int(pos_x.shape[0]):
            continue
        y = int(pos_y[aid])
        x = int(pos_x[aid])
        if y < 0 or x < 0:
            continue
        p = float(risk[y, x])
        if p <= 0.0:
            continue
        if float(rng.random()) >= p:
            continue
        _apply_predator_event(aid, int(slot), y, x)

    if profile_enabled:
        runnercore3_profile_count(state, "runnercore24_jungle_predator_native_event_rows", int(event_rows))
        runnercore3_profile_count(state, "runnercore24_jungle_predator_native_death_rows", int(death_rows))
        runnercore3_profile_count(state, "runnercore45_jungle_predator_context_cache_rows", int(context_cache_rows))
        runnercore3_profile_count(
            state,
            "runnercore24_jungle_predator_native_mirror_sync_rows",
            int(mirror_sync_rows),
        )
    return int(deaths_this_gen)


def _fallback_payload(state: RunnerCore1TickState) -> dict[str, Any]:
    c = state.counters
    return {
        "legacy_batched_actions_calls": int(c.legacy_batched_actions_calls),
        "legacy_batched_actions_k_calls": int(c.legacy_batched_actions_k_calls),
        "legacy_batched_actions_strided_calls": int(c.legacy_batched_actions_strided_calls),
        "legacy_action_cache_refill_calls": int(c.legacy_action_cache_refill_calls),
        "per_tick_agent_sync_rows": int(c.per_tick_agent_sync_rows),
        "boundary_agent_import_rows": int(c.boundary_agent_import_rows),
        "boundary_agent_export_rows": int(c.boundary_agent_export_rows),
        "boundary_policy_bank_sync_rows": int(c.boundary_policy_bank_sync_rows),
        "event_boundary_agent_export_rows": int(c.event_boundary_agent_export_rows),
        "event_boundary_agent_create_rows": int(c.event_boundary_agent_create_rows),
        "event_boundary_policy_bank_sync_rows": int(c.event_boundary_policy_bank_sync_rows),
        "boundary_churn_materialization_rows": int(c.boundary_churn_materialization_rows),
        "action_cache_invalidate_events": int(c.action_cache_invalidate_events),
        "birth_events_applied": int(c.birth_events_applied),
        "death_events_applied": int(c.death_events_applied),
        "refill_events_applied": int(c.refill_events_applied),
        "policy_row_attach_events": int(c.policy_row_attach_events),
        "hmem_row_init_events": int(c.hmem_row_init_events),
        "slot_token_attach_events": int(c.slot_token_attach_events),
        "churn_events_total": int(c.churn_events_total),
        "unsupported_fallback_count": int(c.unsupported_fallback_count),
    }


def _native_policy_cache_public_counts(
    raw_result: Any,
    compact_result: Mapping[str, Any],
) -> dict[str, int]:
    from native_runtime.public_metrics_overlay import native_policy_cache_metrics  # noqa: WPS433

    return native_policy_cache_metrics(raw_result, compact_result)


def run_array_tick_authority(
    runner: Any,
    *,
    gen: int,
    buckets: dict[str, float],
    injury_recover_mult: float,
    deaths_this_gen: int,
    successful_births: int,
    failed_births: int,
    gen_peak_injury: float,
    rest_streaks: dict[int, int],
) -> tuple[int, int, int, float, int]:
    started = time.perf_counter()
    summary = _summary(runner)
    support = classify_support(runner)
    summary["support_report"] = support
    summary["support_blockers"] = list(support.get("blockers", []))
    is_v2 = _runnercore2_requested(runner)
    is_v3 = _runnercore3_requested(runner)
    profile_enabled = _runnercore3_profile_enabled(runner)
    fail_closed_attr = (
        "runnercore11_fail_closed_on_unsupported"
        if _runnercore11_requested(runner)
        else "runnercore10_fail_closed_on_unsupported"
        if _runnercore10_requested(runner)
        else "runnercore9_fail_closed_on_unsupported"
        if _runnercore9_requested(runner)
        else "runnercore6_fail_closed_on_unsupported"
        if _runnercore6_requested(runner)
        else "runnercore5_fail_closed_on_unsupported"
        if _runnercore5_requested(runner)
        else "runnercore3_fail_closed_on_unsupported"
        if is_v3
        else ("runnercore2_fail_closed_on_unsupported" if is_v2 else "runnercore1_fail_closed_on_unsupported")
    )
    fail_closed = _as_bool(getattr(runner.sim_cfg, fail_closed_attr, True))
    profile = _active_profile(runner)
    from native_runtime.safe_default import native_generation_requested  # noqa: WPS433

    native_requested = bool(native_generation_requested(runner))
    if native_requested:
        try:
            from native_runtime.runner_integration import (  # noqa: WPS433
                emit_native_typed_lifecycle_public_outputs,
                execute_native_generation_from_runner,
                write_native_behavior_public_outputs,
            )

            native_result = execute_native_generation_from_runner(
                runner,
                generation_index=int(gen),
                steps_per_generation=int(getattr(runner.sim_cfg, "steps_per_generation", 0) or 0),
            )
            native_summary = native_result.to_diagnostic_dict()
            native_engine = getattr(runner, "_native_eiel_engine")
            if _as_bool(
                getattr(
                    runner.sim_cfg,
                    "typed_lifecycle_public_output_route_neutral_enable",
                    False,
                )
            ):
                native_typed_event_public_output = {
                    "schema_version": "eiel_route_neutral_typed_lifecycle_public_output_v1",
                    "source": "post_turnover_generation_boundary",
                    "deferred": True,
                }
            else:
                native_typed_event_public_output = emit_native_typed_lifecycle_public_outputs(runner, native_engine)
            native_behavior_public_output = write_native_behavior_public_outputs(runner, native_result.raw_result)
            if not bool(native_behavior_public_output.get("flush_complete", False)):
                raise RuntimeError("native behavior public output did not flush successfully")
            if not bool(native_typed_event_public_output.get("deferred", False)) and not bool(
                native_typed_event_public_output.get("flush_complete", False)
            ):
                raise RuntimeError("native typed lifecycle public output did not flush successfully")
            native_summary["typed_lifecycle_public_output"] = dict(native_typed_event_public_output)
            native_summary["behavior_public_output"] = dict(native_behavior_public_output)
            native_validation = dict(native_result.raw_result.get("native_result_validation") or {})
            native_validation["behavior_output_flush_complete"] = True
            native_validation["typed_lifecycle_output_deferred"] = bool(
                native_typed_event_public_output.get("deferred", False)
            )
            native_summary["native_result_validation"] = native_validation
            compact_result = dict(native_summary.get("compact_result") or {})
            policy_cache_metrics = _native_policy_cache_public_counts(native_result.raw_result, compact_result)
            native_summary["policy_cache_metrics"] = dict(policy_cache_metrics)
            try:
                setattr(runner, "_native_runtime_summary", dict(native_summary))
            except Exception:
                pass
            native_deaths_this_gen = _as_int(compact_result.get("death_rows", deaths_this_gen), deaths_this_gen)
            native_successful_births = _as_int(
                compact_result.get("committed_birth_rows", successful_births),
                successful_births,
            )
            native_failed_births = _as_int(compact_result.get("failed_birth_rows", failed_births), failed_births)
            native_public_failed_births = _as_int(
                compact_result.get("public_failed_birth_rows", native_failed_births),
                native_failed_births,
            )
            native_birth_attempts = _as_int(
                compact_result.get("birth_attempt_rows", native_successful_births + native_public_failed_births),
                native_successful_births + native_public_failed_births,
            )
            native_alive_count = _as_int(compact_result.get("alive_count", 0), 0)
            native_capacity = _as_int(compact_result.get("capacity", 0), 0)
            native_free_slots = _as_int(compact_result.get("free_slots", 0), 0)
            summary["native_runtime"] = native_summary
            summary["candidate_ran"] = True
            summary["status"] = "completed"
            summary["decision"] = "EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION"
            summary["ticks_ran"] = int(native_result.native_tick_loop_count)
            native_public_output_counts = {
                "schema_version": "eiel_native_runtime_public_output_counts_v1",
                "source": "native_compact_result",
                "deaths": int(native_deaths_this_gen),
                "successful_births": int(native_successful_births),
                "failed_births": int(native_public_failed_births),
                "native_failed_birth_event_rows": int(native_failed_births),
                "birth_attempts": int(native_birth_attempts),
                "injury_blocked_births": _as_int(compact_result.get("injury_blocked_births", 0), 0),
                "behavior_action_total": _as_int(compact_result.get("applied_actions", 0), 0),
                "behavior_move_total": _as_int(compact_result.get("action_move_total", 0), 0),
                "behavior_move_n_total": _as_int(compact_result.get("action_move_n", 0), 0),
                "behavior_move_s_total": _as_int(compact_result.get("action_move_s", 0), 0),
                "behavior_move_e_total": _as_int(compact_result.get("action_move_e", 0), 0),
                "behavior_move_w_total": _as_int(compact_result.get("action_move_w", 0), 0),
                "behavior_gather_total": _as_int(compact_result.get("action_gather", 0), 0),
                "behavior_rest_total": _as_int(compact_result.get("action_rest", 0), 0),
                "behavior_attack_total": _as_int(compact_result.get("action_attack", 0), 0),
                "native_moved_successfully": _as_int(compact_result.get("moved", 0), 0),
                "native_gathered_successfully": _as_int(compact_result.get("gathered", 0), 0),
                "native_movement_attempts": _as_int(compact_result.get("movement_attempts", 0), 0),
                "native_movement_blocked_occupied": _as_int(compact_result.get("movement_blocked_occupied", 0), 0),
                "native_movement_blocked_bounds": _as_int(compact_result.get("movement_blocked_bounds", 0), 0),
                "birth_pre_cooldown": _as_int(compact_result.get("birth_pre_cooldown", 0), 0),
                "birth_pre_no_energy": _as_int(compact_result.get("birth_pre_no_energy", 0), 0),
                "birth_pre_injury_gate": _as_int(compact_result.get("birth_pre_injury_gate", 0), 0),
                "birth_pre_no_space": _as_int(compact_result.get("birth_pre_no_space", 0), 0),
                "birth_pre_no_mate": _as_int(compact_result.get("birth_pre_no_mate", 0), 0),
                "birth_fail_no_space": _as_int(compact_result.get("birth_fail_no_space", 0), 0),
                "birth_fail_no_mate": _as_int(compact_result.get("birth_fail_no_mate", 0), 0),
                "birth_fail_buffer": _as_int(compact_result.get("birth_fail_buffer", 0), 0),
                "birth_fail_injury_gate": _as_int(compact_result.get("birth_fail_injury_gate", 0), 0),
                "birth_fail_scar_gate": _as_int(compact_result.get("birth_fail_scar_gate", 0), 0),
                "birth_fail_conception": _as_int(compact_result.get("birth_fail_conception", 0), 0),
                "birth_fail_gestation": _as_int(compact_result.get("birth_fail_gestation", 0), 0),
                "alive": int(native_alive_count),
                "capacity": int(native_capacity),
                "free_slots": int(native_free_slots),
            }
            native_public_output_counts.update(policy_cache_metrics)
            summary["native_public_output_counts"] = native_public_output_counts
            try:
                setattr(runner, "_native_runtime_public_output_counts", dict(summary["native_public_output_counts"]))
            except Exception:
                pass
            summary["state_summary"] = {
                "schema_version": "native_runtime_state_summary_v1",
                "source_of_truth": "native_runtime",
                "python_agent_sync": False,
                "per_generation_import_projection_calls": int(
                    native_summary["counters"]["import_projection_calls"]
                ),
                "normal_apply_owner_checkpoint_to_runner_calls": int(
                    native_summary["counters"]["normal_apply_owner_checkpoint_to_runner_calls"]
                ),
            }
            write_summary_if_requested(runner, summary)
            return (
                int(native_deaths_this_gen),
                int(native_successful_births),
                int(native_public_failed_births),
                float(compact_result.get("peak_injury", gen_peak_injury)),
                int(native_result.native_tick_loop_count),
            )
        except Exception as exc:
            from native_runtime.runner_integration import record_native_runtime_failure  # noqa: WPS433

            record_native_runtime_failure(runner, exc, stage="runnercore_native_generation")
            summary["status"] = "native_runtime_fail_closed_exception"
            summary["decision"] = "EIEL-NATIVE-RUNTIME-D/FAIL-CLOSED"
            summary["native_runtime_exception"] = repr(exc)
            write_summary_if_requested(runner, summary)
            raise
    state = import_generation_state(runner, generation=int(gen), profile=profile)
    if _cxxbackbone13_owner_generation_runner_enabled(runner):
        try:
            from perf.cxxbackbone10_resident_state_owner import execute_owner_generation_from_runner  # noqa: WPS433

            cxxbackbone13_result = execute_owner_generation_from_runner(
                runner,
                generation_index=int(gen),
                steps_per_generation=int(getattr(runner.sim_cfg, "steps_per_generation", 0) or 0),
                build_directory=str(getattr(runner.sim_cfg, "cxxbackbone3_policy_aten_build_dir", "") or "") or None,
            )
            cxxbackbone13_summary = {
                "schema_version": "cxxbackbone13_owner_generation_runner_summary_v1",
                "executed": bool(cxxbackbone13_result.get("executed", False)),
                "decision": str(cxxbackbone13_result.get("decision", "CXXBACKBONE13-OWNER-GENERATION-RUNNER")),
                "world": str(cxxbackbone13_result.get("world", "")),
                "generation_index": int(gen),
                "steps_per_generation": int(getattr(runner.sim_cfg, "steps_per_generation", 0) or 0),
                "result": cxxbackbone13_result,
            }
            setattr(runner, "_cxxbackbone13_owner_generation_runner_summary", cxxbackbone13_summary)
            _cxxbackbone13_write_summary(runner, cxxbackbone13_summary)
            summary["cxxbackbone13_owner_generation_runner"] = cxxbackbone13_summary
            summary["candidate_ran"] = True
            summary["status"] = "completed"
            summary["decision"] = "CXXBACKBONE13-OWNER-POLICY-GENERATION-RUNNER"
            summary["ticks_ran"] = int(getattr(runner.sim_cfg, "steps_per_generation", 0) or 0)
            summary["state_summary"] = state_summary_payload(import_generation_state(runner, generation=int(gen), profile=profile))
            write_summary_if_requested(runner, summary)
            return (
                int(cxxbackbone13_result.get("died", deaths_this_gen)),
                int(successful_births),
                int(failed_births),
                float(gen_peak_injury),
                int(getattr(runner.sim_cfg, "steps_per_generation", 0) or 0),
            )
        except Exception as exc:
            summary["status"] = "cxxbackbone13_fail_closed_exception"
            summary["decision"] = "CXXBACKBONE13-D/PRODUCTION-POLICY-GENERATION-PARITY-BLOCKED-AFTER-IMPLEMENTATION"
            summary["cxxbackbone13_exception"] = repr(exc)
            write_summary_if_requested(runner, summary)
            if _cxxbackbone13_owner_generation_runner_fail_closed(runner):
                raise
            return int(deaths_this_gen), int(successful_births), int(failed_births), float(gen_peak_injury), 0
    if _cxxbackbone8_generation_island_enabled(runner):
        from perf.cxxbackbone8_generation_island import (  # noqa: WPS433
            execute_generation as cxxbackbone8_execute_generation,
            summary_payload as cxxbackbone8_summary_payload,
            write_summary as cxxbackbone8_write_summary,
        )

        cxxbackbone8_result = cxxbackbone8_execute_generation(
            runner,
            state,
            generation_index=int(gen),
            steps_per_generation=int(getattr(runner.sim_cfg, "steps_per_generation", 0) or 0),
            buckets=buckets,
        )
        cxxbackbone8_summary = cxxbackbone8_summary_payload(cxxbackbone8_result)
        setattr(runner, "_cxxbackbone8_generation_island_summary", cxxbackbone8_summary)
        cxxbackbone8_write_summary(runner, cxxbackbone8_summary)
        summary["cxxbackbone8_generation_island"] = cxxbackbone8_summary
        if not bool(cxxbackbone8_result.get("executed", False)):
            summary["status"] = "cxxbackbone8_fail_closed_unsupported"
            summary["decision"] = str(cxxbackbone8_result.get("decision", "CXXBACKBONE8-G/INVALID-WRAPPER-OR-DIAGNOSTIC-CLOSEOUT"))
            write_summary_if_requested(runner, summary)
            if _cxxbackbone8_generation_island_fail_closed(runner):
                raise RunnerCore1UnsupportedError(
                    str(cxxbackbone8_result.get("decision", "CXXBACKBONE8-G/INVALID-WRAPPER-OR-DIAGNOSTIC-CLOSEOUT"))
                    + ":"
                    + ",".join(str(reason) for reason in cxxbackbone8_result.get("blockers", []))
                )
            return int(deaths_this_gen), int(successful_births), int(failed_births), float(gen_peak_injury), 0
    if not bool(support.get("supported", False)):
        summary["status"] = "fail_closed_unsupported"
        summary["decision"] = _active_blocked_decision(runner)
        write_summary_if_requested(runner, summary)
        if fail_closed:
            raise RunnerCore1UnsupportedError(f"{_active_blocked_decision(runner)}: {','.join(summary['support_blockers'])}")
        return int(deaths_this_gen), int(successful_births), int(failed_births), float(gen_peak_injury), 0

    ticks_ran = 0
    runnercore11_fast_food_spawn = bool(
        _runnercore11_requested(runner)
        and _as_bool(getattr(runner.sim_cfg, "runnercore11_fast_weighted_food_spawn_enable", False))
    )
    try:
        setattr(runner.env, "_runnercore11_fast_weighted_food_spawn_enable", bool(runnercore11_fast_food_spawn))
    except Exception:
        pass
    if runnercore11_fast_food_spawn:
        _runnercore11_stat_add(runner, "fast_weighted_food_spawn_enabled", 1)
    try:
        resume_start_tick = int(getattr(runner, "_eiel_resume_start_tick", 0) or 0)
        resume_end_tick = resume_start_tick + int(runner.sim_cfg.steps_per_generation)
        for tick in range(resume_start_tick, resume_end_tick):
            t_tick_total = time.perf_counter() if profile_enabled else 0.0
            if _as_bool(getattr(runner, "_stop_requested", False)):
                break
            runner._eiel_fdp_cur_tick = int(tick)
            runner._eiel_fdp_cur_gen = int(gen)
            runner.tick = int(tick)
            runner.gen = int(gen)
            runner._eiel_dbg_tick = int(tick)
            runner._eiel_dbg_gen = int(gen)

            _update_patch_context(runner, gen=int(gen), tick=int(tick))
            t_spawn = time.perf_counter()
            runner.env.spawn_food()
            buckets["env_spawn_food"] = float(buckets.get("env_spawn_food", 0.0) + (time.perf_counter() - t_spawn))

            if _as_bool(getattr(runner.sim_cfg, "runnercore24_native_jungle_predator_arrays_enable", False)) and _as_bool(
                getattr(runner.env_cfg, "jungle_predator_pressure_enable", False)
            ):
                t_jungle = time.perf_counter()
                deaths_this_gen = _runnercore24_apply_native_jungle_predator_pressure(state, runner, int(deaths_this_gen))
                if profile_enabled:
                    runnercore3_profile_add(state, "runnercore24_jungle_predator_native_ms", time.perf_counter() - t_jungle)
                buckets["jungle_predator_pressure"] = float(
                    buckets.get("jungle_predator_pressure", 0.0) + (time.perf_counter() - t_jungle)
                )
            elif _as_bool(getattr(runner.sim_cfg, "runnercore23_jungle_predator_bridge_enable", False)) and _as_bool(
                getattr(runner.env_cfg, "jungle_predator_pressure_enable", False)
            ):
                t_jungle = time.perf_counter()
                jungle_ids = alive_agent_ids(state, runner)
                if int(jungle_ids.shape[0]) > 0:
                    export_to_agents(
                        state,
                        runner,
                        boundary_reason="runnercore23_jungle_predator_bridge",
                        agent_ids=jungle_ids.tolist(),
                    )
                    if profile_enabled:
                        runnercore3_profile_count(
                            state,
                            "runnercore23_jungle_predator_bridge_export_rows",
                            int(jungle_ids.shape[0]),
                        )
                    deaths_this_gen = runner._apply_jungle_predator_pressure(int(deaths_this_gen))
                    agents = getattr(getattr(runner, "pop", None), "agents", {})
                    synced = 0
                    if isinstance(agents, dict):
                        for aid in sorted(int(aid) for aid in jungle_ids.tolist()):
                            agent = agents.get(int(aid))
                            if agent is None:
                                continue
                            if sync_agent_row_from_mirror(state, runner, agent, is_new=False) >= 0:
                                synced += 1
                    if profile_enabled:
                        runnercore3_profile_count(state, "runnercore23_jungle_predator_bridge_import_rows", synced)
                buckets["jungle_predator_pressure"] = float(
                    buckets.get("jungle_predator_pressure", 0.0) + (time.perf_counter() - t_jungle)
                )

            t_alive = time.perf_counter() if profile_enabled else 0.0
            ids = alive_agent_ids(state, runner)
            if int(ids.shape[0]) <= 0:
                break
            ids_list = [int(x) for x in ids.tolist()]
            if (
                hasattr(runner, "resolve_developmental_expression_for_agent")
                and not bool(getattr(runner.sim_cfg, "perf_skip_dev_expression_prewarm_enable", False))
            ):
                t_dev = time.perf_counter()
                _runnercore34_apply_dev_expression_side_effects(
                    state,
                    runner,
                    ids,
                    source="prewarm",
                )
                buckets["aggperf3_dev_expression_prewarm"] = float(
                    buckets.get("aggperf3_dev_expression_prewarm", 0.0) + (time.perf_counter() - t_dev)
                )
            runner.np_rng.shuffle(ids_list)
            ids = np.asarray(ids_list, dtype=np.int32)
            if profile_enabled:
                runnercore3_profile_add(state, "alive_order_and_mask_build_ms", time.perf_counter() - t_alive)

            cxxbackbone7_result = None
            cxxbackbone7_enabled = _cxxbackbone7_policy_action_loop_enabled(runner)
            t_policy = time.perf_counter()
            if cxxbackbone7_enabled:
                from perf.cxxbackbone7_policy_action_service import select_actions_policy_service_v1  # noqa: WPS433

                actions = select_actions_policy_service_v1(runner, state, ids, buckets=buckets)
                if actions is None:
                    if _cxxbackbone7_fail_closed(runner):
                        raise RunnerCore1UnsupportedError(
                            "CXXBACKBONE7 resident patch policy-action loop unsupported for current run"
                        )
                    actions = runnercore1_actions_strided(runner, state, ids, buckets=buckets)
                    cxxbackbone7_enabled = False
            else:
                actions = runnercore1_actions_strided(runner, state, ids, buckets=buckets)
            buckets["runnercore1_policy_action_selection"] = float(
                buckets.get("runnercore1_policy_action_selection", 0.0) + (time.perf_counter() - t_policy)
            )
            runner._eiel_fdp_last_actions_tick = int(tick)
            runner._eiel_fdp_last_actions_agent_ids = np.asarray(ids, dtype=np.int64).copy()
            runner._eiel_fdp_last_actions_actions = np.asarray(actions, dtype=np.int64).copy()

            _update_action_coherence(state, runner, ids, actions)
            _cxx26b_maybe_dump_tick_detail(runner, state, gen=int(gen), tick=int(tick), stage="pre_env")
            runner.behavior_logger.record_actions(
                np.asarray(ids, dtype=np.int32),
                np.asarray(actions, dtype=np.int32),
            )
            t_env = time.perf_counter() if profile_enabled else 0.0
            if cxxbackbone7_enabled:
                from perf.cxxbackbone7_policy_action_service import apply_actions_environment_service_v1  # noqa: WPS433

                cxxbackbone7_result = apply_actions_environment_service_v1(
                    runner,
                    state,
                    ids,
                    actions,
                    int(deaths_this_gen),
                    float(gen_peak_injury),
                    tick=int(tick),
                    buckets=buckets,
                )
                deaths_this_gen = int(cxxbackbone7_result.deaths_this_gen)
                gen_peak_injury = float(cxxbackbone7_result.gen_peak_injury)
            else:
                deaths_this_gen, gen_peak_injury = _apply_env_tick_arrays(
                    state,
                    runner,
                    ids,
                    actions,
                    int(deaths_this_gen),
                    float(gen_peak_injury),
                    tick=int(tick),
                    buckets=buckets,
                )
            if profile_enabled:
                runnercore3_profile_add(state, "env_scalar_apply_ms", time.perf_counter() - t_env)
            cxxwave1f_applied = _cxxwave1f_apply_scalar_backbone(
                state,
                runner,
                ids,
                actions,
                tick_index=int(getattr(runner, "_pred_tick", 0)),
                buckets=buckets,
            )
            if not cxxwave1f_applied:
                _update_drives(state, runner, ids, actions, tick_index=int(getattr(runner, "_pred_tick", 0)))
                _update_chemistry(state, runner, ids)
                t_maint = time.perf_counter() if profile_enabled else 0.0
                _update_maintenance(
                    state,
                    runner,
                    ids,
                    actions,
                    tick_index=int(getattr(runner, "_pred_tick", 0)),
                    buckets=buckets,
                )
                if profile_enabled:
                    runnercore3_profile_add(state, "maintenance_update_ms", time.perf_counter() - t_maint)
            elif profile_enabled:
                runnercore3_profile_add(state, "maintenance_update_ms", float(buckets.get("cxxwave1f_scalar_backbone_last", 0.0)))
            _cxx26b_maybe_dump_tick_detail(runner, state, gen=int(gen), tick=int(tick), stage="post_drives")
            deaths_this_gen = _apply_rest_and_combat(
                state,
                runner,
                ids,
                actions,
                int(deaths_this_gen),
                rest_streaks=rest_streaks,
                injury_recover_mult=float(injury_recover_mult),
            )
            if is_v2:
                successful_births, failed_births = _runnercore2_handle_births(
                    state,
                    runner,
                    ids,
                    int(successful_births),
                    int(failed_births),
                    buckets,
                    gen=int(gen),
                    tick=int(tick),
                )
            else:
                _decrement_repro_cooldowns(state, ids)
            _cxx26b_maybe_dump_tick_detail(runner, state, gen=int(gen), tick=int(tick), stage="post_birth")
            runner._pred_tick = int(getattr(runner, "_pred_tick", 0)) + 1
            ticks_ran += 1
            state.counters.ticks_ran = int(ticks_ran)
            if profile_enabled:
                runnercore3_profile_add(state, "total_candidate_tick_ms", time.perf_counter() - t_tick_total)
    except Exception:
        export_to_agents(state, runner, boundary_reason="fail_closed_restore")
        summary["status"] = "exception_restored"
        summary["decision"] = _active_blocked_decision(runner)
        summary["state_summary"] = state_summary_payload(state)
        summary["fallback_counters"] = _fallback_payload(state)
        if profile_enabled:
            summary["runnercore3_profile"] = _runnercore3_profile_payload(state)
        runnercore4_payload = _runnercore4_structural_payload_payload(runner)
        if runnercore4_payload is not None:
            summary["runnercore4_structural_payload"] = runnercore4_payload
        runnercore5_payload = _runnercore5_child_construction_payload(runner)
        if runnercore5_payload is not None:
            summary["runnercore5_child_construction"] = runnercore5_payload
        runnercore6_payload = _runnercore6_resident_child_row_payload(runner)
        if runnercore6_payload is not None:
            summary["runnercore6_resident_child_row"] = runnercore6_payload
        runnercore9_payload = _runnercore9_c1_payload_or_lazy_newborn_payload(runner)
        if runnercore9_payload is not None:
            summary["runnercore9_c1_payload_or_lazy_newborn"] = runnercore9_payload
        runnercore10_payload = _runnercore10_typed_child_payload(runner)
        if runnercore10_payload is not None:
            summary["runnercore10_typed_child"] = runnercore10_payload
        runnercore11_payload = _runnercore11_c1_typed_genome_row_payload(runner)
        if runnercore11_payload is not None:
            summary["runnercore11_c1_typed_genome_row"] = runnercore11_payload
        write_summary_if_requested(runner, summary)
        raise

    boundary_export = export_to_agents(state, runner, boundary_reason="generation_end")
    summary["candidate_ran"] = True
    summary["status"] = "completed"
    summary["decision"] = (
        "RUNNERCORE11-CANDIDATE-C1-TYPED-GENOME-ROW"
        if _runnercore11_requested(runner)
        else "RUNNERCORE9-CANDIDATE-C1-PAYLOAD-OR-LAZY-NEWBORN"
        if _runnercore9_requested(runner)
        else "RUNNERCORE6-CANDIDATE-RESIDENT-CHILD-ROW"
        if _runnercore6_requested(runner)
        else "RUNNERCORE5-CANDIDATE-CHILD-CONSTRUCTION-OWNERSHIP"
        if _runnercore5_requested(runner)
        else "RUNNERCORE4-CANDIDATE-STRUCTURAL-PAYLOAD-CHURN"
        if _runnercore4_requested(runner)
        else (
            "RUNNERCORE3-CANDIDATE-AUTHORITATIVE-CHURN"
            if is_v3
            else (
                "RUNNERCORE2-CANDIDATE-AUTHORITATIVE-CHURN"
                if is_v2
                else "RUNNERCORE1-CANDIDATE-AUTHORITATIVE-NO-CHURN"
            )
        )
    )
    summary["ticks_ran"] = int(ticks_ran)
    summary["state_summary"] = state_summary_payload(state)
    summary["fallback_counters"] = _fallback_payload(state)
    summary["boundary_export"] = boundary_export
    if profile_enabled:
        summary["runnercore3_profile"] = _runnercore3_profile_payload(state)
    runnercore4_payload = _runnercore4_structural_payload_payload(runner)
    if runnercore4_payload is not None:
        summary["runnercore4_structural_payload"] = runnercore4_payload
    runnercore5_payload = _runnercore5_child_construction_payload(runner)
    if runnercore5_payload is not None:
        summary["runnercore5_child_construction"] = runnercore5_payload
    runnercore6_payload = _runnercore6_resident_child_row_payload(runner)
    if runnercore6_payload is not None:
        summary["runnercore6_resident_child_row"] = runnercore6_payload
    runnercore9_payload = _runnercore9_c1_payload_or_lazy_newborn_payload(runner)
    if runnercore9_payload is not None:
        summary["runnercore9_c1_payload_or_lazy_newborn"] = runnercore9_payload
    runnercore10_payload = _runnercore10_typed_child_payload(runner)
    if runnercore10_payload is not None:
        summary["runnercore10_typed_child"] = runnercore10_payload
    runnercore11_payload = _runnercore11_c1_typed_genome_row_payload(runner)
    if runnercore11_payload is not None:
        env = getattr(runner, "env", None)
        if env is not None:
            stats = _runnercore11_stats(runner)
            if stats is not None:
                stats["fast_weighted_food_spawn_calls"] = int(
                    getattr(env, "_runnercore11_fast_weighted_food_spawn_calls", 0) or 0
                )
                stats["fast_weighted_food_spawn_draws"] = int(
                    getattr(env, "_runnercore11_fast_weighted_food_spawn_draws", 0) or 0
                )
                stats["fast_weighted_food_spawn_placements"] = int(
                    getattr(env, "_runnercore11_fast_weighted_food_spawn_placements", 0) or 0
                )
                stats["fast_weighted_food_spawn_incomplete"] = int(
                    getattr(env, "_runnercore11_fast_weighted_food_spawn_incomplete", 0) or 0
                )
            runnercore11_payload = _runnercore11_c1_typed_genome_row_payload(runner)
        summary["runnercore11_c1_typed_genome_row"] = runnercore11_payload
    summary["timing_s"]["runnercore1_total"] = float(time.perf_counter() - started)
    write_summary_if_requested(runner, summary)
    return int(deaths_this_gen), int(successful_births), int(failed_births), float(gen_peak_injury), int(ticks_ran)


def support_audit_payload(runner: Any) -> dict[str, Any]:
    return classify_support(runner)
