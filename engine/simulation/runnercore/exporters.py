from __future__ import annotations

"""RUNNERCORE-1 compatibility exporter contract."""

from typing import Any


SCHEMA_VERSION = "runnercore1_exporter_contract_v1"


def exporter_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_export_boundaries": [
            "end_of_tick_for_legacy_debug_views_if_enabled",
            "end_of_generation_for_metrics",
            "pre_checkpoint_if_checkpoint_period_nonzero",
            "post_churn_for_population_and_env_views",
        ],
        "required_object_views": [
            "Agent scalar fields",
            "AgentStats counters",
            "Population.agents membership",
            "SlotPool free/allocated state",
            "env agent arrays",
        ],
        "candidate_fast_profile_checkpoint_support": "unsupported_requires_checkpoint_bundle_v2_period_gen_0",
        "status": "generation_boundary_agent_export_owned_for_supported_no_churn_profile",
    }
