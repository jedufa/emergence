from __future__ import annotations

"""RUNNERCORE-1 churn tape contract."""

from typing import Any


SCHEMA_VERSION = "runnercore1_churn_tape_contract_v1"


def churn_tape_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "events": [
            "env_death",
            "birth_attempt",
            "birth_success",
            "birth_failure",
            "generation_turnover_death",
            "hardcap_cull",
            "low_population_reinitialize",
            "refill_child",
            "slot_free",
            "slot_alloc",
            "hmem_reset",
            "policy_bank_row_alloc",
            "policy_bank_row_dirty",
            "action_cache_invalidate",
            "env_capacity_grow",
            "env_population_sync",
        ],
        "atomic_update_targets": [
            "resident identity arrays",
            "SlotPool state",
            "Population membership",
            "env agent arrays",
            "policy banks",
            "GRU Hmem",
            "action cache",
            "lineage/founder metric arrays",
        ],
        "status": "schema_only_fail_closed_for_candidate",
        "blocker": (
            "The supported RUNNERCORE-1 candidate requires reproduce_try_prob=0. "
            "Population.make_child and reproduce_and_cull still create/mutate Agent, Genome, Policy, SlotPool, and env state directly."
        ),
    }
