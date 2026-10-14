from __future__ import annotations

"""RUNNERCORE-1 fallback counters and divergence localizer payloads."""

from typing import Any


FALLBACK_SCHEMA_VERSION = "runnercore1_fallback_counters_v1"
DIVERGENCE_SCHEMA_VERSION = "runnercore1_divergence_localizer_v1"


def fallback_counters_payload(*, candidate_attempted: bool = False, candidate_accepted: bool = False) -> dict[str, Any]:
    return {
        "schema_version": FALLBACK_SCHEMA_VERSION,
        "candidate_attempted": bool(candidate_attempted),
        "candidate_accepted": bool(candidate_accepted),
        "legacy_batched_actions_calls": 0,
        "legacy_batched_actions_k_calls": 0,
        "legacy_batched_actions_strided_calls": 0,
        "legacy_action_cache_refill_calls": 0,
        "per_tick_agent_sync_rows": 0,
        "candidate_fail_closed_before_action_materialization": bool(candidate_attempted and not candidate_accepted),
        "unsupported_fallback_count": 0,
    }


def divergence_localizer_payload(*, status: str = "not_run") -> dict[str, Any]:
    return {
        "schema_version": DIVERGENCE_SCHEMA_VERSION,
        "status": str(status),
        "layers": [
            "alive_order",
            "rng_stream_or_shuffle_result",
            "policy_input_arrays",
            "policy_model_bank_row_mapping",
            "gru_hmem_pre_post_rows",
            "stride_cache_state",
            "action_logits_or_probs",
            "final_action_output",
            "env_action_application_result",
            "post_action_scalar_state",
            "churn_tape_event",
            "metrics_export",
        ],
        "first_divergence": None,
        "note": "Localizer layers are available for candidate parity failures and fail-closed ownership boundaries.",
    }
