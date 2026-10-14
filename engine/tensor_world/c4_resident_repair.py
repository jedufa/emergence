"""Source-level C4 resident repair contract and reset coverage inventory.

The current C4 population transaction remains a CPU preparation oracle.  This
module describes the bounded resident repair that must replace that boundary;
it does not implement a CUDA transaction and it cannot authorize a C4 gate.
The inventory is generated from the TensorWorld V4 owner dataclasses so a new
resident field cannot silently escape newborn/dead-slot classification.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Final

from .c2_state import (
    AgentEffectBufferV1,
    C2HotMetricStateV1,
    C2PolicyTensorStateV1,
    C2RngTensorStateV1,
    C2TransactionTensorStateV1,
    EedTensorStateV1,
    PatchTensorStateV1,
    WdlTensorStateV1,
    WorldEventBufferV1,
)
from .c3_state import (
    C3AdaptationTensorStateV1,
    C3AgentRuntimeTensorStateV1,
    C3GenomeExecutionTensorStateV1,
    C3LifecycleEligibilityTensorStateV1,
    C3PolicyRuntimeTensorStateV1,
    C3TransactionTensorStateV1,
)
from .c4_state import (
    C4EventTapeTensorStateV1,
    C4PopulationTensorStateV1,
    C4PythonRandomTensorStateV1,
)
from .state import (
    AgentTensorStateV1,
    HotEventAccumulatorStateV1,
    IdentityTensorStateV1,
    OccupancyTensorStateV1,
    PolicyTensorStateV1,
    RngTensorStateV1,
    TransactionTensorStateV1,
    WorldTensorStateV1,
)


C4_RESIDENT_REPAIR_CONTRACT_SCHEMA_V1: Final[str] = (
    "tensor_world_c4_resident_repair_contract_v1"
)
C4_RESIDENT_RESET_MANIFEST_SCHEMA_V1: Final[str] = (
    "tensor_world_c4_newborn_dead_slot_reset_coverage_v1"
)
C4_RESIDENT_REPAIR_STATUS_V1: Final[str] = "REPAIR_CONTRACT_ONLY"
C4_RESIDENT_REPAIR_GATE_CLAIM_V1: Final[bool] = False

C4_RESIDENT_REPAIR_PHASE_ORDER_V1: Final[tuple[str, ...]] = (
    "eligibility_snapshot",
    "ordered_death_candidate_materialization",
    "death_settlement",
    "occupancy_release",
    "free_slot_queue_update",
    "parent_mate_candidate_construction",
    "deterministic_parent_mate_selection",
    "birth_plan_construction",
    "slot_reservation",
    "identity_lineage_allocation",
    "mutation_recombination",
    "newborn_genome_policy_adaptation_reset",
    "newborn_placement_occupancy_commit",
    "typed_lifecycle_event_publication",
    "transaction_receipt_and_counters",
    "atomic_authoritative_commit",
)

C4_RESIDENT_REPAIR_INVARIANTS_V1: Final[tuple[str, ...]] = (
    "fixed_capacity_buffers_are_allocated_before_steady_state_execution",
    "semantic_order_is_explicit_and_never_gpu_completion_order",
    "slot_reuse_never_reuses_biological_agent_identity",
    "death_event_is_sealed_before_slot_reuse",
    "occupancy_agent_id_and_slot_maps_agree_after_commit",
    "failed_transaction_preserves_authoritative_state_rng_and_event_cursor",
    "no_cpu_parent_selection_identity_allocation_or_per_birth_fallback",
    "newborn_reset_covers_every_slot_owned_v4_field",
)

C4_RESIDENT_REPAIR_BUFFER_CONTRACT_V1: Final[dict[str, dict[str, object]]] = {
    "death_candidates": {
        "capacity": "[batch,capacity]",
        "owner": "base.lifecycle.death_candidate",
        "order": "ascending_biological_agent_id",
    },
    "free_slots": {
        "capacity": "[batch,capacity]",
        "owner": "population.free_queue_or_free_mask",
        "order": "source_cpu_oracle_allocation_order",
    },
    "birth_plan": {
        "capacity": "[batch,max_births]",
        "owner": "population.parent_slots/child_slots/mate_slots",
        "order": "world_local_attempt_order",
    },
    "event_rows": {
        "capacity": "[batch,event_capacity]",
        "owner": "events.typed_*",
        "order": "transaction_phase_order",
    },
    "device_shadows": {
        "capacity": "all_authoritative_v4_tensors",
        "owner": "caller_owned_reusable_scratch",
        "order": "all_phase_status_zero_before_commit",
    },
}


# Each tuple is (runtime path prefix, owner dataclass).  The prefixes are the
# actual V4 attribute paths, rather than a second abbreviated state schema.
_V4_OWNER_TYPES: Final[tuple[tuple[str, type], ...]] = (
    ("base.base.core.world", WorldTensorStateV1),
    ("base.base.core.agents", AgentTensorStateV1),
    ("base.base.core.occupancy", OccupancyTensorStateV1),
    ("base.base.core.policy", PolicyTensorStateV1),
    ("base.base.core.identity", IdentityTensorStateV1),
    ("base.base.core.rng", RngTensorStateV1),
    ("base.base.core.transactions", TransactionTensorStateV1),
    ("base.base.core.hot_events", HotEventAccumulatorStateV1),
    ("base.base.patch", PatchTensorStateV1),
    ("base.base.wdl", WdlTensorStateV1),
    ("base.base.eed", EedTensorStateV1),
    ("base.base.policy", C2PolicyTensorStateV1),
    ("base.base.rng", C2RngTensorStateV1),
    ("base.base.effects", AgentEffectBufferV1),
    ("base.base.world_events", WorldEventBufferV1),
    ("base.base.transactions", C2TransactionTensorStateV1),
    ("base.base.hot_metrics", C2HotMetricStateV1),
    ("base.agent", C3AgentRuntimeTensorStateV1),
    ("base.genome", C3GenomeExecutionTensorStateV1),
    ("base.adaptation", C3AdaptationTensorStateV1),
    ("base.policy_runtime", C3PolicyRuntimeTensorStateV1),
    ("base.lifecycle", C3LifecycleEligibilityTensorStateV1),
    ("base.transactions", C3TransactionTensorStateV1),
    ("population", C4PopulationTensorStateV1),
    ("events", C4EventTapeTensorStateV1),
    ("scalar_rng", C4PythonRandomTensorStateV1),
)

_C2_SLOT_EFFECT_FIELDS: Final[frozenset[str]] = frozenset(
    {"active_y", "active_x", "active_class", "handling_progress", "handling_axis"}
)
_C2_SLOT_EVENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"gather_amount", "gather_resource_class"}
)


def _field_classification(owner_path: str, field_name: str) -> str:
    """Return the one reset/ownership class for a V4 tensor field."""

    if owner_path == "base.base.core.occupancy" or (
        owner_path == "base.base.patch" and field_name == "agent_grid_i32"
    ):
        return "occupancy_projection"
    if owner_path == "base.base.core.world" or owner_path in {
        "base.base.patch",
        "base.base.wdl",
    }:
        return "world_state_preserved"
    if owner_path == "base.base.core.agents":
        return "agent_slot_state"
    if owner_path == "base.base.core.policy" or owner_path in {
        "base.base.policy",
    }:
        return "policy_slot_state"
    if owner_path == "base.base.core.identity":
        return "identity_projection"
    if owner_path == "base.base.core.rng" or owner_path == "base.base.rng":
        return "rng_continuation"
    if owner_path in {"base.base.core.transactions", "base.base.transactions", "base.transactions"}:
        return "transaction_slot_state"
    if owner_path == "base.base.core.hot_events" or owner_path == "base.base.hot_metrics":
        return "metrics_state"
    if owner_path == "base.base.eed" and field_name in _C2_SLOT_EFFECT_FIELDS:
        return "slot_resource_state"
    if owner_path == "base.base.eed":
        return "world_state_preserved"
    if owner_path == "base.base.effects":
        return "slot_effect_state"
    if owner_path == "base.base.world_events" and field_name in _C2_SLOT_EVENT_FIELDS:
        return "slot_effect_state"
    if owner_path == "base.base.world_events":
        return "world_state_preserved"
    if owner_path == "base.base.core.world" or owner_path == "base.base.patch":
        return "world_state_preserved"
    if owner_path == "base.agent":
        return "agent_slot_state"
    if owner_path == "base.genome":
        return "genome_slot_state"
    if owner_path == "base.adaptation":
        return "adaptation_slot_state"
    if owner_path == "base.policy_runtime":
        return "policy_runtime_slot_state"
    if owner_path == "base.lifecycle":
        return "lifecycle_slot_state"
    if owner_path == "population":
        return "population_transaction_state"
    if owner_path == "events":
        return "event_tape"
    if owner_path == "scalar_rng":
        return "scalar_rng_continuation"
    # Keep an unexpected owner fail-closed rather than silently treating it as
    # outside C4 scope.  The validator rejects this class as unsupported.
    return "UNCLASSIFIED"


_CLASS_POLICY: Final[dict[str, dict[str, str]]] = {
    "agent_slot_state": {
        "scope": "slot",
        "death_action": "invalidate_and_scrub",
        "newborn_action": "initialize_or_inherit",
        "authority": "c4_cpu_oracle_newborn_and_dead_slot_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "genome_slot_state": {
        "scope": "slot",
        "death_action": "invalidate_and_scrub",
        "newborn_action": "mutate_or_recombine_then_compile",
        "authority": "c4_cpu_genome_oracle",
        "resident_status": "REPAIR_REQUIRED",
    },
    "policy_slot_state": {
        "scope": "slot",
        "death_action": "invalidate_and_scrub",
        "newborn_action": "inherit_or_compile",
        "authority": "c3_policy_projection_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "adaptation_slot_state": {
        "scope": "slot",
        "death_action": "invalidate_and_scrub",
        "newborn_action": "reset_to_source_defaults",
        "authority": "c4_cpu_newborn_reset_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "policy_runtime_slot_state": {
        "scope": "slot",
        "death_action": "invalidate_and_scrub",
        "newborn_action": "clear_cache_and_initialize",
        "authority": "c4_cpu_newborn_reset_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "lifecycle_slot_state": {
        "scope": "slot",
        "death_action": "clear_candidate_and_reason",
        "newborn_action": "reset_eligibility_state",
        "authority": "c4_cpu_population_oracle",
        "resident_status": "REPAIR_REQUIRED",
    },
    "transaction_slot_state": {
        "scope": "slot_or_transaction",
        "death_action": "clear_stale_slot_rows",
        "newborn_action": "clear_and_seed_pre_state",
        "authority": "c4_cpu_transaction_boundary",
        "resident_status": "REPAIR_REQUIRED",
    },
    "occupancy_projection": {
        "scope": "world_projection",
        "death_action": "release_before_slot_reuse",
        "newborn_action": "place_and_commit_both_maps",
        "authority": "c4_cpu_population_oracle",
        "resident_status": "REPAIR_REQUIRED",
    },
    "identity_projection": {
        "scope": "identity_projection",
        "death_action": "preserve_world_identity_domain",
        "newborn_action": "allocate_world_local_slot_projection",
        "authority": "c4_identity_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "population_transaction_state": {
        "scope": "population_transaction",
        "death_action": "settle_append_or_clear_by_field",
        "newborn_action": "reserve_allocate_or_record_by_field",
        "authority": "c4_cpu_population_oracle",
        "resident_status": "REPAIR_REQUIRED",
    },
    "slot_resource_state": {
        "scope": "slot",
        "death_action": "invalidate_and_scrub",
        "newborn_action": "reset_to_source_defaults",
        "authority": "c2_resource_owner_and_c4_reset_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "slot_effect_state": {
        "scope": "slot_or_world_effect",
        "death_action": "clear_stale_slot_effects",
        "newborn_action": "clear_and_initialize",
        "authority": "c2_effect_owner_and_c4_reset_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "event_tape": {
        "scope": "append_only_event_tape",
        "death_action": "append_before_slot_reuse",
        "newborn_action": "append_birth_rows",
        "authority": "c4_typed_event_contract",
        "resident_status": "REPAIR_REQUIRED",
    },
    "rng_continuation": {
        "scope": "rng_continuation",
        "death_action": "preserve_stream_and_cursor",
        "newborn_action": "consume_declared_draw_budget",
        "authority": "c5_rng_handoff",
        "resident_status": "C5_HANDOFF_REQUIRED",
    },
    "scalar_rng_continuation": {
        "scope": "rng_continuation",
        "death_action": "preserve_stream_and_cursor",
        "newborn_action": "consume_declared_draw_budget",
        "authority": "c4_cpu_cpython_rng_oracle",
        "resident_status": "C5_HANDOFF_REQUIRED",
    },
    "metrics_state": {
        "scope": "world_metrics",
        "death_action": "preserve_or_advance_outside_c4",
        "newborn_action": "not_applicable",
        "authority": "c3_c2_metric_owner",
        "resident_status": "OUTSIDE_C4_SCOPE",
    },
    "world_state_preserved": {
        "scope": "world_state",
        "death_action": "preserve",
        "newborn_action": "not_applicable",
        "authority": "c2_world_owner",
        "resident_status": "OUTSIDE_C4_SCOPE",
    },
}


def c4_newborn_dead_slot_reset_coverage_manifest_v1() -> dict[str, object]:
    """Return the complete generated V4 field classification manifest."""

    records: list[dict[str, object]] = []
    for owner_path, owner_type in _V4_OWNER_TYPES:
        for field in fields(owner_type):
            if field.name == "schema_version":
                continue
            classification = _field_classification(owner_path, field.name)
            policy = _CLASS_POLICY.get(classification)
            if policy is None:
                raise ValueError(
                    f"C4 resident field has no reset policy: {owner_path}.{field.name}"
                )
            records.append(
                {
                    "path": f"{owner_path}.{field.name}",
                    "owner": owner_path,
                    "field": field.name,
                    "classification": classification,
                    **policy,
                }
            )
    return {
        "schema_version": C4_RESIDENT_RESET_MANIFEST_SCHEMA_V1,
        "status": C4_RESIDENT_REPAIR_STATUS_V1,
        "gate_claim": C4_RESIDENT_REPAIR_GATE_CLAIM_V1,
        "source_of_truth": "engine/tensor_world/c4_resident_repair.py",
        "field_count": len(records),
        "fields": records,
        "control_fields": {
            "base.tick": {
                "classification": "transaction_metadata",
                "death_action": "advance_or_preserve",
                "newborn_action": "inherit_transaction_tick",
            },
            "base.schema_version": {
                "classification": "schema_metadata",
                "death_action": "preserve",
                "newborn_action": "not_applicable",
            },
        },
    }


def c4_resident_repair_contract_v1() -> dict[str, object]:
    """Return the bounded repair contract without granting gate authority."""

    coverage = c4_newborn_dead_slot_reset_coverage_manifest_v1()
    by_classification: dict[str, int] = {}
    for entry in coverage["fields"]:
        classification = str(entry["classification"])
        by_classification[classification] = by_classification.get(classification, 0) + 1
    return {
        "schema_version": C4_RESIDENT_REPAIR_CONTRACT_SCHEMA_V1,
        "phase": "TW-C4",
        "status": C4_RESIDENT_REPAIR_STATUS_V1,
        "gate_claim": C4_RESIDENT_REPAIR_GATE_CLAIM_V1,
        "historical_decision": "TW-C4-RESIDENCY-BLOCKED_PRESERVED",
        "resident_transaction_phase_order": list(C4_RESIDENT_REPAIR_PHASE_ORDER_V1),
        "invariants": list(C4_RESIDENT_REPAIR_INVARIANTS_V1),
        "buffer_contract": {
            name: dict(spec)
            for name, spec in C4_RESIDENT_REPAIR_BUFFER_CONTRACT_V1.items()
        },
        "coverage_schema_version": coverage["schema_version"],
        "field_count": coverage["field_count"],
        "classification_counts": by_classification,
        "resident_transaction_status": "NOT_BOUND_FAIL_CLOSED",
        "cpu_oracle_role": "semantic_authority_only",
        "cuda_identity_preflight_role": "non_mutating_staging_only",
        "gpu_execution": "NOT_RUN_BY_THIS_CONTRACT_LANE",
    }


def validate_c4_resident_repair_contract_v1() -> dict[str, object]:
    """Validate uniqueness and total coverage of the V4 tensor inventory."""

    coverage = c4_newborn_dead_slot_reset_coverage_manifest_v1()
    records = coverage["fields"]
    paths = [str(entry["path"]) for entry in records]
    if len(paths) != len(set(paths)):
        raise ValueError("C4 resident reset manifest contains duplicate paths")
    if any(entry["classification"] == "UNCLASSIFIED" for entry in records):
        raise ValueError("C4 resident reset manifest contains an unclassified field")
    if coverage["gate_claim"] is not False:
        raise ValueError("C4 resident repair manifest cannot claim a gate")
    return {
        "schema_version": C4_RESIDENT_REPAIR_CONTRACT_SCHEMA_V1,
        "status": C4_RESIDENT_REPAIR_STATUS_V1,
        "field_count": len(paths),
        "classification_counts": c4_resident_repair_contract_v1()[
            "classification_counts"
        ],
        "gate_claim": False,
    }


__all__ = [
    "C4_RESIDENT_REPAIR_BUFFER_CONTRACT_V1",
    "C4_RESIDENT_REPAIR_CONTRACT_SCHEMA_V1",
    "C4_RESIDENT_REPAIR_GATE_CLAIM_V1",
    "C4_RESIDENT_REPAIR_INVARIANTS_V1",
    "C4_RESIDENT_REPAIR_PHASE_ORDER_V1",
    "C4_RESIDENT_REPAIR_STATUS_V1",
    "C4_RESIDENT_RESET_MANIFEST_SCHEMA_V1",
    "c4_newborn_dead_slot_reset_coverage_manifest_v1",
    "c4_resident_repair_contract_v1",
    "validate_c4_resident_repair_contract_v1",
]
