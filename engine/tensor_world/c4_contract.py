"""Source-bound C4 population-transaction contract.

This module is a preparation boundary only.  It inventories the current
death/birth/identity source seams and their ownership rules without enabling a
population mutation inside the C3 backend.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Final

from .c3_contract import C3_SOURCE_FIELD_GROUPS
from .contracts import TensorWorldContractError


C4_SOURCE_AUDIT_SCHEMA_V1: Final[str] = "tensor_world_c4_population_contract_v1"
C4_GATE_AUTHORIZATION_V1: Final[str] = (
    "C4_ENGINEERING_PREPARATION_AFTER_TW-C3-GO-C4"
)

C4_SOURCE_REFERENCES: Final[dict[str, dict[str, str]]] = {
    "controller_kernel": {
        "path": "engine/perf/cuda_architecture_b_flattened_controller_v1_kernels.cu",
        "symbols": "is_death_candidate_v3, lifecycle_death_cull_cell_v3, execute_ordinary_birth_phase_v1",
        "sha256": "5f1e76615c1631b29dd9402a4251304d183ab4217f030e63d0ac0b0d6ca94851",
    },
    "birth_plan": {
        "path": "engine/perf/cuda_architecture_b_persistent_birth_plan.cuh",
        "symbols": "plan_ordinary_births_serial, life_history_fertility_scale_v1",
        "sha256": "a8c84055d1e1a9a2d7f26933ce765264ff80a1d24dd76fe8ea0284348470b783",
    },
    "mate_birth_plan": {
        "path": "engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh",
        "symbols": "plan_births_serial_v1, life_history_fertility_scale_v1",
        "sha256": "0db57397c20e0b5966b287dc2ba1fcbcb582dc55fccf65d02cbcb4f0e6951994",
    },
    "birth_identity": {
        "path": "engine/perf/cuda_architecture_b_persistent_birth_identity.cuh",
        "symbols": "identity allocation and slot reuse boundary",
        "sha256": "71d68a85e34777bc78e9d034bcb58513af19aa987ae9dbd8a658e8c778b2841a",
    },
    "birth_policy": {
        "path": "engine/perf/cuda_architecture_b_persistent_birth_policy.cuh",
        "symbols": "resident policy birth commit and reset",
        "sha256": "f7182ccce6c4234080c9d52f374b50874384a214bd33efa1346ee97b5835c689",
    },
    "birth_genome": {
        "path": "engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh",
        "symbols": "resident genome birth commit and execution-view reset",
        "sha256": "b7bf58be767fce7dcb1b21d85331d2b8d227044495fed77da0e9dac261ba2ecb",
    },
    "birth_events": {
        "path": "engine/perf/cuda_architecture_b_persistent_birth_events.cuh",
        "symbols": "typed lifecycle event append and sequence ownership",
        "sha256": "72e4522393790f31f5e8ce7e8375c1807075b54a0ed4b7513ed5d96e6bf9b27e",
    },
    "birth_action_history": {
        "path": "engine/perf/cuda_architecture_b_persistent_birth_action_history.cuh",
        "symbols": "newborn action-history and counter reset",
        "sha256": "02444c6f6bca12413734b7a7308b030ae153101aaff9d07084b6af6ccf14b29d",
    },
}

C4_PHASE_ORDER: Final[tuple[str, ...]] = (
    "c3_death_and_reproduction_eligibility_snapshot",
    "world_local_death_settlement_in_agent_id_order",
    "free_slot_queue_append_and_duplicate_rejection",
    "parent_and_mate_selection_with_declared_rng_boundary",
    "birth_plan_and_reserved_slot_commit",
    "identity_lineage_and_generation_metadata_commit",
    "resident_genome_policy_and_adaptation_reset",
    "newborn_placement_and_occupancy_commit",
    "typed_lifecycle_event_commit",
)

C4_IDENTITY_INVARIANTS: Final[tuple[str, ...]] = (
    "slot_reuse_does_not_reuse_biological_agent_id",
    "death_event_is_sealed_before_slot_can_be_reused",
    "lineage_founder_and_parent_ids_are_external_identity_values",
    "free_slot_queue_order_is_world_local_and_deterministic",
    "birth_plan_is_read_only_until_commit_boundary",
    "c3_death_candidate_does_not_mutate_alive_or_identity_masks",
    "newborn_state_has_no_stale_parent_or_dead_slot_residue",
)

# The production program names these fixtures explicitly.  Keep the names in
# the source contract so a future parity campaign cannot silently omit a
# population edge case while still producing a plausible aggregate result.
C4_STRESS_FIXTURES_V1: Final[tuple[str, ...]] = (
    "all_agents_survive_no_births",
    "high_death_count",
    "free_slot_fragmentation",
    "maximum_admitted_births",
    "placement_conflicts",
    "failed_births",
    "parent_ties",
    "mutation_heavy_path",
    "recombination_path",
    "lineage_split_merge_identity",
    "repeated_slot_reuse",
    "mixed_world_transaction_counts",
)

# These are observation surfaces at a transaction boundary, not an alternate
# state owner.  A resident implementation must expose equivalent device-side
# staging/receipt fields before C4 can claim parity or residency.
C4_PARITY_BOUNDARY_SURFACES_V1: Final[dict[str, tuple[str, ...]]] = {
    "pre_transaction_state": (
        "base.base.core.agents.alive",
        "base.base.core.agents.agent_id",
        "base.base.core.agents.pos_y",
        "base.base.core.agents.pos_x",
        "base.base.core.occupancy.agent_id_grid",
        "base.base.core.occupancy.slot_grid",
        "population.free_queue",
        "population.free_queue_count",
        "population.next_agent_id",
    ),
    "ordered_candidate_sets": (
        "population.decision_parent_slots",
        "population.decision_mate_slots",
        "population.decision_reason_codes",
        "population.decision_count",
    ),
    "selected_parents_and_rng": (
        "population.parent_slots",
        "population.mate_slots",
        "population.spawn_rng_draws_added",
        "population.spawn_rng_semantic_draws_added",
        "population.runner_rng_draws_added",
        "population.runner_rng_semantic_draws_added",
    ),
    "event_order": (
        "events.typed_event_type",
        "events.typed_sequence",
        "events.typed_actor_id",
        "events.typed_target_id",
        "events.active_count",
    ),
    "identity_lineage": (
        "population.birth_parent_ids",
        "population.birth_child_ids",
        "population.active_agent_ids",
        "population.active_agent_slots",
        "population.agent_id_to_slot",
        "population.slot_tokens",
        "population.lineage_founder_id",
        "population.lineage_parent_agent_id",
        "population.lineage_depth",
    ),
    "post_commit_masks_and_effects": (
        "base.base.core.agents.alive",
        "base.base.core.agents.agent_id",
        "base.base.core.occupancy.agent_id_grid",
        "base.base.core.occupancy.slot_grid",
        "base.base.core.world.food_amount",
        "population.free_queue",
        "population.free_queue_count",
    ),
}

C4_RESIDENCY_BOUNDARY_INVARIANTS_V1: Final[tuple[str, ...]] = (
    "steady_state_population_transaction_has_no_cpu_round_trip",
    "candidate_and_commit_buffers_are_fixed_capacity_and_device_local",
    "identity_preflight_failure_does_not_trigger_cpu_fallback",
    "event_and_rng_preflights_complete_before_authoritative_device_mutation",
    "lineage_reporting_is_external_to_the_hot_transaction",
)

C4_FIELD_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "C4_TRANSACTIONAL": C3_SOURCE_FIELD_GROUPS["C4_TRANSACTIONAL"],
    "C3_INPUTS": (
        "alive_mask",
        "slot_to_agent_id",
        "energy",
        "health",
        "injury",
        "survival_steps",
        "repro_threshold",
        "death_eligibility_outputs",
    ),
    "C5_HANDOFF": (
        "environment_pcg64_state",
        "policy_pcg64_state",
        "birth_spawn_pcg64_state",
        "population_pcg64_state",
        "policy_birth_semantic_draws_tape",
        "policy_birth_tape_cursor",
    ),
    "C6_HANDOFF": (
        "typed_lifecycle_event_actor_agent_id",
        "typed_lifecycle_event_actor_slot",
        "typed_lifecycle_event_child_agent_id",
        "typed_lifecycle_event_event_type",
        "typed_lifecycle_event_sequence",
    ),
}


def validate_c4_population_contract(repo_root: Path | str = ".") -> dict[str, object]:
    """Validate the source files and the explicit C4 ownership inventory."""

    root = Path(repo_root)
    source_hashes: dict[str, str] = {}
    for name, reference in C4_SOURCE_REFERENCES.items():
        path = root / reference["path"]
        if not path.is_file():
            raise TensorWorldContractError(f"C4 source reference is missing: {path}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != reference["sha256"]:
            raise TensorWorldContractError(
                f"C4 source drift for {name}: expected {reference['sha256']}, got {digest}"
            )
        source_hashes[name] = digest

    flattened = [field for fields in C4_FIELD_GROUPS.values() for field in fields]
    duplicate_fields = sorted({field for field in flattened if flattened.count(field) > 1})
    if duplicate_fields:
        raise TensorWorldContractError(
            f"C4 field appears in multiple handoff groups: {duplicate_fields}"
        )
    if len(C4_PHASE_ORDER) != len(set(C4_PHASE_ORDER)):
        raise TensorWorldContractError("C4 phase order contains duplicates")
    if len(C4_STRESS_FIXTURES_V1) != len(set(C4_STRESS_FIXTURES_V1)):
        raise TensorWorldContractError("C4 stress fixture names contain duplicates")
    boundary_fields = [
        field
        for fields in C4_PARITY_BOUNDARY_SURFACES_V1.values()
        for field in fields
    ]
    if not all(field.startswith(("base.", "population.", "events.")) for field in boundary_fields):
        raise TensorWorldContractError("C4 parity surface is outside resident state owners")
    return {
        "schema_version": C4_SOURCE_AUDIT_SCHEMA_V1,
        "gate_authorization": C4_GATE_AUTHORIZATION_V1,
        "source_hashes": source_hashes,
        "phase_order": list(C4_PHASE_ORDER),
        "identity_invariants": list(C4_IDENTITY_INVARIANTS),
        "stress_fixtures": list(C4_STRESS_FIXTURES_V1),
        "parity_boundary_surfaces": {
            name: list(fields) for name, fields in C4_PARITY_BOUNDARY_SURFACES_V1.items()
        },
        "residency_boundary_invariants": list(C4_RESIDENCY_BOUNDARY_INVARIANTS_V1),
        "field_groups": {name: list(fields) for name, fields in C4_FIELD_GROUPS.items()},
        "status": "PREPARATION_ONLY",
    }


__all__ = [
    "C4_FIELD_GROUPS",
    "C4_GATE_AUTHORIZATION_V1",
    "C4_IDENTITY_INVARIANTS",
    "C4_PARITY_BOUNDARY_SURFACES_V1",
    "C4_PHASE_ORDER",
    "C4_RESIDENCY_BOUNDARY_INVARIANTS_V1",
    "C4_SOURCE_AUDIT_SCHEMA_V1",
    "C4_SOURCE_REFERENCES",
    "C4_STRESS_FIXTURES_V1",
    "validate_c4_population_contract",
]
