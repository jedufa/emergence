from __future__ import annotations

"""Supported-profile contract for RUNNERCORE-0.

This module answers a narrow question: can the requested live runner profile be
owned by a resident array runner without consulting Python Agent objects every
tick?  The current answer is "shadow contract only"; candidate/native use is
blocked by explicit lifecycle ownership gaps.
"""

from typing import Any


SCHEMA_VERSION = "runnercore0_supported_profile_contract_v1"
OWNERSHIP_SCHEMA_VERSION = "runnercore0_lifecycle_ownership_map_v1"
SUPPORTED_PROFILE = "fast_exploratory_metrics_only_v2"
RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY = "RUNNERCORE0-C/BLOCKED-BY-BOUNDARY"
RUNNERCORE_DECISION_INVALID_PROFILE = "RUNNERCORE0-D/PROFILE-CONTRACT-MISMATCH"

_REQUEST_FLAGS = (
    "runnercore_supported_profile_enable",
    "runnercore_shadow_contract_enable",
    "runnercore_candidate_enable",
    "runnercore_authoritative_enable",
    "runnercore_native_enable",
    "runnercore_legacy_compare_enable",
)

_BOUNDARY_BLOCKERS = (
    "per_tick_python_agent_sync_required_for_policy_inputs",
    "alive_order_shuffle_and_action_cache_rng_order_not_owned",
    "policy_bank_rows_and_gru_hmem_rows_not_resident_authoritative",
    "env_apply_side_effects_still_write_python_agent_stats",
    "death_birth_refill_churn_lifecycle_not_owned",
    "checkpoint_metrics_and_behavior_logs_depend_on_python_objects",
    "chemistry_drive_and_recovery_side_effects_not_owned_by_runnercore",
)

_OWNED_ROWS = (
    {
        "item": "supported profile declaration",
        "owner": "runnercore contract",
        "status": "owned",
        "notes": "Default-off gate for fast_exploratory_metrics_only_v2.",
    },
    {
        "item": "slot token resident snapshot",
        "owner": "runnercore shadow state",
        "status": "owned_at_import_boundary",
        "notes": "Imports arrays from the legacy runner for parity/debug only.",
    },
    {
        "item": "candidate/native fail-closed decision",
        "owner": "runnercore gate",
        "status": "owned",
        "notes": "Prevents unsupported C++ or authoritative paths from silently falling back to Python Agent sync.",
    },
)

_NON_OWNED_ROWS = (
    {
        "item": "population slot lifecycle",
        "current_owner": "Population/Python Agent objects",
        "required_for_candidate": True,
        "blocker": "death/birth/refill tape is not resident-authoritative through the full generation.",
    },
    {
        "item": "alive ordering and shuffle",
        "current_owner": "runner tick loop Python list plus metrics_np RNG",
        "required_for_candidate": True,
        "blocker": "candidate lane would need to reproduce list ordering and RNG consumption exactly.",
    },
    {
        "item": "policy inputs and action cache",
        "current_owner": "runner_policy Python staging from Agent/genome/stat objects",
        "required_for_candidate": True,
        "blocker": "resident arrays are not yet the source of truth for every policy scalar, drive, GRU row, and cache token.",
    },
    {
        "item": "environment apply side effects",
        "current_owner": "runner_ticks_env_apply Python/env arrays with Agent writeback",
        "required_for_candidate": True,
        "blocker": "stats, injury, costs, patch effects, combat, rest/recovery, and diagnostics still mutate object state.",
    },
    {
        "item": "metrics/checkpoint/behavior logging",
        "current_owner": "Python loggers and checkpoint serializers",
        "required_for_candidate": True,
        "blocker": "generation metrics and checkpoint compatibility still read Python objects.",
    },
    {
        "item": "native policy-frame packing",
        "current_owner": "CXXREWRITE3 lower-bound resident frame helper",
        "required_for_candidate": False,
        "blocker": "native packing can consume selected arrays, but cannot make the full runner authoritative.",
    },
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def runnercore_requested(cfg: Any) -> bool:
    return any(_as_bool(getattr(cfg, name, False)) for name in _REQUEST_FLAGS)


def candidate_requested(cfg: Any) -> bool:
    return (
        _as_bool(getattr(cfg, "runnercore_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore_authoritative_enable", False))
        or _as_bool(getattr(cfg, "runnercore_native_enable", False))
    )


def profile_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "supported_profile": SUPPORTED_PROFILE,
        "default_off": True,
        "required_config": {
            "runnercore_supported_profile": SUPPORTED_PROFILE,
            "policy_tick_stride": 2,
            "policy_deterministic_actions": True,
            "chemistry_functional_coupling_enable": False,
            "checkpoint_bundle_v2_period_gen": 0,
        },
        "supported_modes": {
            "shadow_contract": True,
            "legacy_compare_oracle": True,
            "candidate_authoritative": False,
            "native_cpp": False,
        },
        "native_cpp_status": (
            "deferred: native attachment is blocked until resident arrays own the live tick "
            "without per-tick Python Agent synchronization"
        ),
        "candidate_blockers": list(_BOUNDARY_BLOCKERS),
        "ownership_map": ownership_map_payload(),
    }


def ownership_map_payload() -> dict[str, Any]:
    return {
        "schema_version": OWNERSHIP_SCHEMA_VERSION,
        "owned_lifecycle": list(_OWNED_ROWS),
        "non_owned_lifecycle": list(_NON_OWNED_ROWS),
        "boundary_conclusion": (
            "The RUNNERCORE-0 contract can import and audit slot arrays, but the live action/env/reproduction "
            "loop remains Python-object authoritative. Candidate/native modes must fail closed."
        ),
    }


def classify_supported_profile(cfg: Any) -> dict[str, Any]:
    requested_profile = str(getattr(cfg, "runnercore_supported_profile", SUPPORTED_PROFILE) or "").strip()
    blockers: list[str] = []
    warnings: list[str] = []

    if requested_profile != SUPPORTED_PROFILE:
        blockers.append("unsupported_runnercore_profile")
    if _as_int(getattr(cfg, "policy_tick_stride", 0), 0) != 2:
        blockers.append("policy_tick_stride_must_equal_2")
    if not _as_bool(getattr(cfg, "policy_deterministic_actions", False)):
        blockers.append("policy_deterministic_actions_required")
    if _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", True)):
        blockers.append("chemistry_functional_coupling_must_be_disabled")
    if _as_int(getattr(cfg, "checkpoint_bundle_v2_period_gen", 0), 0) != 0:
        blockers.append("checkpoint_bundle_v2_period_gen_must_be_0")

    if _as_bool(getattr(cfg, "behavior_log_enable", True)):
        warnings.append("behavior_log_enable_adds_output_cost_but_does_not_change_contract")
    if not _as_bool(getattr(cfg, "perf_aggperf3_timing_buckets_enable", False)):
        warnings.append("perf_aggperf3_timing_buckets_enable_off_limits_timing_attribution")

    candidate = candidate_requested(cfg)
    shadow_supported = not blockers
    candidate_blockers = list(_BOUNDARY_BLOCKERS) if shadow_supported else []
    decision = (
        RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY
        if shadow_supported and candidate
        else (RUNNERCORE_DECISION_INVALID_PROFILE if blockers else "RUNNERCORE0-SHADOW-CONTRACT-ONLY")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "requested": runnercore_requested(cfg),
        "requested_profile": requested_profile,
        "supported_profile": SUPPORTED_PROFILE,
        "shadow_contract_supported": bool(shadow_supported),
        "candidate_requested": bool(candidate),
        "native_requested": _as_bool(getattr(cfg, "runnercore_native_enable", False)),
        "authoritative_requested": _as_bool(getattr(cfg, "runnercore_authoritative_enable", False)),
        "candidate_supported": False,
        "decision": decision,
        "profile_blockers": blockers,
        "candidate_blockers": candidate_blockers,
        "warnings": warnings,
        "config": {
            "policy_tick_stride": _as_int(getattr(cfg, "policy_tick_stride", 0), 0),
            "policy_deterministic_actions": _as_bool(getattr(cfg, "policy_deterministic_actions", False)),
            "chemistry_functional_coupling_enable": _as_bool(
                getattr(cfg, "chemistry_functional_coupling_enable", True)
            ),
            "checkpoint_bundle_v2_period_gen": _as_int(getattr(cfg, "checkpoint_bundle_v2_period_gen", 0), 0),
            "runnercore_fail_closed_on_unsupported": _as_bool(
                getattr(cfg, "runnercore_fail_closed_on_unsupported", True)
            ),
        },
        "ownership_map": ownership_map_payload(),
    }
