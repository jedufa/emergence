from __future__ import annotations

"""RUNNERCORE-1 array-tick authority audit and fail-closed decision helpers."""

from typing import Any

from simulation.runnercore.action_cache import action_cache_contract_payload
from simulation.runnercore.churn_tape import churn_tape_contract_payload
from simulation.runnercore.diagnostics import divergence_localizer_payload, fallback_counters_payload
from simulation.runnercore.exporters import exporter_contract_payload
from simulation.runnercore.tick_state import tick_state_schema_payload


SCHEMA_VERSION = "runnercore1_array_tick_authority_audit_v1"
OWNERSHIP_SCHEMA_VERSION = "runnercore1_array_tick_authority_ownership_map_v1"
SUPPORTED_PROFILE = "supported_fast_runner_v1"
RUNNERCORE1_DECISION_BLOCKED_BY_OWNERSHIP = "RUNNERCORE1-C/BLOCKED-BY-OWNERSHIP"
RUNNERCORE1_DECISION_INVALID_PROFILE = "RUNNERCORE1-E/INVALID-PROFILE-CONTRACT"

_REQUEST_FLAGS = (
    "runnercore1_array_tick_authority_enable",
    "runnercore1_candidate_enable",
    "runnercore1_authoritative_enable",
    "runnercore1_native_enable",
)

_CANDIDATE_BLOCKERS = (
    "birth_refill_churn_creates_agent_genome_policy_objects",
    "slotpool_population_dict_and_envsync_not_churn_tape_authoritative",
    "maintenance_state_update_not_array_authoritative",
    "checkpoint_and_rich_metrics_exporters_need_boundary_object_views",
)

_OWNED_ROWS = (
    {
        "item": "supported-profile audit contract",
        "owner": "RUNNERCORE-1 audit gate",
        "status": "owned",
        "notes": "Default-off RUNNERCORE-1 flags classify candidate feasibility before action materialization.",
    },
    {
        "item": "array tick state schema",
        "owner": "RUNNERCORE-1 tick_state contract",
        "status": "owned_for_no_churn_tick_profile",
        "notes": "Imports alive Agent mirrors once at generation boundary and exports once at generation end.",
    },
    {
        "item": "churn tape contract",
        "owner": "RUNNERCORE-1 churn_tape contract",
        "status": "schema_only",
        "notes": "Names death/birth/refill/slot/Hmem/policy-row events; churn remains fail-closed.",
    },
    {
        "item": "fallback counters",
        "owner": "RUNNERCORE-1 diagnostics",
        "status": "owned",
        "notes": "No-churn candidate records zero legacy batched-action fallback and zero per-tick Agent sync.",
    },
)

_NON_OWNED_ROWS = (
    {
        "item": "policy input source of truth outside supported profile",
        "current_owner": "PolicyMixin._batched_actions from live Agent/genome/stats objects",
        "required_for_candidate": False,
        "blocker": "The no-churn candidate owns policy input arrays for the supported profile; broader developmental/perception knobs remain out of contract.",
    },
    {
        "item": "action cache and final action authority outside supported profile",
        "current_owner": "PolicyMixin stride cache plus legacy _batched_actions/_batched_actions_k",
        "required_for_candidate": False,
        "blocker": "The candidate owns a slot-token stride cache for supported no-churn ticks; unsupported diagnostics and churn invalidate the cache contract.",
    },
    {
        "item": "GRU Hmem and policy bank row lifecycle under churn",
        "current_owner": "runner policy sync helpers and Hmem tensor writes keyed by Agent slot/token",
        "required_for_candidate": True,
        "blocker": "Boundary import owns current alive rows, but birth/refill/cull updates do not yet atomically update resident Hmem, policy rows, cache tokens, and population identity.",
    },
    {
        "item": "env scalar/stat side effects outside supported profile",
        "current_owner": "TickEnvApplyMixin._apply_env_tick writing Agent and AgentStats objects every tick",
        "required_for_candidate": False,
        "blocker": "The candidate owns supported no-churn scalar/stat arrays and exports at generation end; richer side effects remain out of contract.",
    },
    {
        "item": "death/birth/refill churn",
        "current_owner": "TickReproductionMixin, Population, SlotPool, SpawnMixin",
        "required_for_candidate": True,
        "blocker": "Churn creates/mutates Agent, Genome, Policy, lineage, chemistry, drive, SlotPool, env, Hmem, and bank-dirty state.",
    },
    {
        "item": "metrics/checkpoint/export readers outside boundary export",
        "current_owner": "MetricsLogger, checkpoint_bundle_v2, lineage/funnel summaries",
        "required_for_candidate": True,
        "blocker": "The candidate can export Agent/AgentStats mirrors at generation end, but checkpoints and rich per-tick readers remain default-off/fail-closed.",
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


def runnercore1_requested(cfg: Any) -> bool:
    return any(_as_bool(getattr(cfg, name, False)) for name in _REQUEST_FLAGS)


def runnercore1_candidate_requested(cfg: Any) -> bool:
    return (
        _as_bool(getattr(cfg, "runnercore1_candidate_enable", False))
        or _as_bool(getattr(cfg, "runnercore1_authoritative_enable", False))
        or _as_bool(getattr(cfg, "runnercore1_native_enable", False))
    )


def profile_blockers(cfg: Any) -> list[str]:
    blockers: list[str] = []
    requested_profile = str(getattr(cfg, "runnercore1_supported_profile", SUPPORTED_PROFILE) or "").strip()
    if requested_profile != SUPPORTED_PROFILE:
        blockers.append("unsupported_runnercore1_profile")
    if _as_int(getattr(cfg, "policy_tick_stride", 0), 0) != 2:
        blockers.append("policy_tick_stride_must_equal_2")
    if not _as_bool(getattr(cfg, "policy_deterministic_actions", False)):
        blockers.append("policy_deterministic_actions_required")
    if _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", True)):
        blockers.append("chemistry_functional_coupling_must_be_disabled")
    if _as_int(getattr(cfg, "checkpoint_bundle_v2_period_gen", 0), 0) != 0:
        blockers.append("checkpoint_bundle_v2_period_gen_must_be_0")
    return blockers


def ownership_map_payload() -> dict[str, Any]:
    return {
        "schema_version": OWNERSHIP_SCHEMA_VERSION,
        "owned_lifecycle": list(_OWNED_ROWS),
        "non_owned_lifecycle": list(_NON_OWNED_ROWS),
        "boundary_conclusion": (
            "RUNNERCORE-1 cannot safely make the supported-profile tick array-authoritative "
            "until policy/action, env side effects, churn, and metrics/export readers stop using "
            "Python Agent objects as per-tick truth."
        ),
    }


def support_audit_payload(cfg: Any) -> dict[str, Any]:
    blockers = profile_blockers(cfg)
    profile_ok = not blockers
    candidate = runnercore1_candidate_requested(cfg)
    decision = (
        RUNNERCORE1_DECISION_INVALID_PROFILE
        if blockers
        else RUNNERCORE1_DECISION_BLOCKED_BY_OWNERSHIP
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "requested": runnercore1_requested(cfg),
        "candidate_requested": bool(candidate),
        "supported_profile": SUPPORTED_PROFILE,
        "requested_profile": str(getattr(cfg, "runnercore_supported_profile", SUPPORTED_PROFILE) or ""),
        "profile_supported": bool(profile_ok),
        "candidate_supported": False,
        "authoritative_candidate_available": False,
        "decision": decision,
        "profile_blockers": blockers,
        "candidate_blockers": list(_CANDIDATE_BLOCKERS) if profile_ok else [],
        "config": {
            "policy_tick_stride": _as_int(getattr(cfg, "policy_tick_stride", 0), 0),
            "policy_deterministic_actions": _as_bool(getattr(cfg, "policy_deterministic_actions", False)),
            "chemistry_functional_coupling_enable": _as_bool(
                getattr(cfg, "chemistry_functional_coupling_enable", True)
            ),
            "checkpoint_bundle_v2_period_gen": _as_int(getattr(cfg, "checkpoint_bundle_v2_period_gen", 0), 0),
            "runnercore1_fail_closed_on_unsupported": _as_bool(getattr(cfg, "runnercore1_fail_closed_on_unsupported", True)),
        },
        "ownership_map": ownership_map_payload(),
        "tick_state_schema": tick_state_schema_payload(),
        "action_cache_contract": action_cache_contract_payload(),
        "churn_tape_contract": churn_tape_contract_payload(),
        "exporter_contract": exporter_contract_payload(),
        "fallback_counters": fallback_counters_payload(candidate_attempted=bool(candidate), candidate_accepted=bool(profile_ok and candidate)),
        "divergence_localizer": divergence_localizer_payload(status="available"),
    }


def decision_payload(cfg: Any) -> dict[str, Any]:
    audit = support_audit_payload(cfg)
    return {
        "schema_version": "runnercore1_decision_v1",
        "decision": audit["decision"],
        "parity_ok": None,
        "authoritative_candidate_ran": None,
        "candidate_supported": True,
        "candidate_blockers": list(audit.get("candidate_blockers", [])),
        "timing_claim": "requires live harness result; no static speedup claim",
        "native_cpp_status": "not compiled by audit; native attachment requires parity and live timing evidence first",
        "recommended_next_step": (
            "Implement resident-array policy/action authority, env scalar/stat side-effect arrays, "
            "and a death/birth/refill churn tape before native attachment."
        ),
        "no_claims_statement": (
            "RUNNERCORE-1 is a bounded performance-engineering and runner architecture task; "
            "it makes no scientific or cognition claims."
        ),
    }
