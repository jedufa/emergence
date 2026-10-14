"""Machine-checkable C4 R11 newborn initialization registry.

The registry enumerates every tensor whose first two axes are the bundle's
``[batch, capacity]`` axes.  It distinguishes genuine slot state from
same-shaped transaction rows, identity projections, and static/C5-owned
surfaces.  There is deliberately no catch-all initializer: a newly introduced
surface makes registry construction fail until its semantics are classified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import torch

from .c4_fixture import make_c4_fixture_v1
from .c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _zero_tensors,
)
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
CPU_ORACLE = ROOT / "engine/tensor_world/c4_population.py"
SCALAR_ORACLE = ROOT / "engine/tensor_world/c4_genome.py"
R2_RESET_OWNER = ROOT / "engine/tensor_world/c4_resident_birth_r2.py"
R5_COMPOSED_OWNER = ROOT / "engine/tensor_world/c4_resident_population_r5.py"
R11_OWNER = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
STATE_OWNER = ROOT / "engine/tensor_world/c4_state.py"

C4_NEWBORN_REGISTRY_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_newborn_initialization_registry_r11_v1"
)
C4_NEWBORN_REGISTRY_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"

_STATIC_SLOT_ACTIONS: Final[dict[str, str]] = {
    "base.base.core.identity.slot_ids": "preserve_physical_slot_identity",
    "base.base.rng.jungle_uniforms": "preserve_existing_slot_value_C5_owned",
}
_WORLD_METRIC_ACTIONS: Final[dict[str, str]] = {
    "base.base.hot_metrics.jungle_integers": "not_applicable_world_metric_axis",
}

_DIRECT_NEWBORN_ACTIONS: Final[dict[str, str]] = {
    "base.base.core.agents.alive": "set_true",
    "base.base.core.agents.agent_id": "allocate_monotonic_biological_id",
    "base.base.core.agents.pos_y": "copy_planned_child_y",
    "base.base.core.agents.pos_x": "copy_planned_child_x",
    "base.base.core.agents.energy": "copy_planned_child_energy",
    "base.base.core.agents.health": "set_config_start_health",
    "base.base.core.agents.age": "reset_zero",
    "base.base.core.agents.survival_steps": "reset_zero",
    "base.base.core.agents.energy_collected": "reset_zero",
    "base.base.core.policy.drives": "derive_from_child_genome_and_energy",
    "base.base.core.policy.attack_bias": "project_child_genome_aggression_bias",
    "base.base.core.policy.action_temperature": (
        "project_child_genome_action_temperature"
    ),
    "base.base.core.policy.drive_logit_gain": "project_child_genome_drive_logit_gain",
    "base.agent.age": "reset_zero",
    "base.agent.survival_steps": "reset_zero",
    "base.agent.policy_energy_observation": "copy_planned_child_energy",
    "base.agent.drives": "derive_from_child_genome_and_energy",
    "base.agent.drive_inertia": "project_child_genome_drive_inertia",
    "base.agent.drive_noise": "project_child_genome_drive_noise",
    "base.agent.repro_threshold": "project_child_genome_reproduction_threshold",
    "base.adaptation.chemistry": "derive_from_child_energy",
    "base.transactions.pre_energy": "copy_planned_child_energy",
    "base.transactions.pre_health": "set_config_start_health",
    "base.transactions.pre_injury": "reset_zero",
    "base.transactions.pre_survival_steps": "reset_zero",
}

_MUTATING_FIXED_MLP_GENOME_ACTIONS: Final[dict[str, str]] = {
    "base.genome.hidden_size": "inherit_primary_parent_fixed_mlp",
    "base.genome.memory_dim": "set_zero_fixed_mlp_after_memory_draw",
    "base.genome.policy_mem_dim": "set_zero_fixed_mlp_after_memory_draw",
    "base.genome.policy_use_gru": "set_false_after_parent_mate_fail_closed_preflight",
    "base.genome.sensor_channel_mask": "normalize_primary_parent_sensor_mask",
    "base.genome.sensor_radius": "normalize_primary_parent_sensor_radius",
    "base.genome.introspection_channel_mask": "mask_primary_parent_value",
    "base.genome.self_regulation_control_mask": "mask_primary_parent_value",
    "base.genome.self_modification_target_mask": "mask_primary_parent_value",
    "base.genome.directed_inheritance_channel_mask": "mask_primary_parent_value",
    "base.genome.exploration_bias": "midpoint_parent_mate_then_scalar_mutation",
    "base.genome.aggression_bias": "midpoint_parent_mate_then_scalar_mutation",
    "base.genome.mutation_rate": "midpoint_parent_mate_then_rate_mutation",
    "base.genome.reproduction_threshold": (
        "midpoint_parent_mate_then_scalar_mutation"
    ),
    "base.genome.action_temperature": "midpoint_parent_mate_then_scalar_mutation",
    "base.genome.drive_inertia": "midpoint_parent_mate_then_scalar_mutation",
    "base.genome.drive_noise": "midpoint_parent_mate_then_scalar_mutation",
    "base.genome.drive_logit_gain": "midpoint_parent_mate_then_scalar_mutation",
    "base.genome.pred_lr": "midpoint_parent_mate_then_scalar_mutation",
    "base.genome.drive_setpoints": "mean_parent_mate_then_clamp",
    "base.genome.exp_prior_strength": "midpoint_parent_mate_then_clamp",
    "base.genome.exp_prior_noise": "midpoint_parent_mate_then_clamp",
    "base.genome.drive_birth_noise": "midpoint_parent_mate_then_clamp",
}

_POPULATION_SLOT_ACTIONS: Final[dict[str, str]] = {
    "population.slot_tokens": "increment_reused_slot_token",
    "population.scar": "reset_zero",
    "population.repro_cooldown": "reset_zero",
    "population.offspring_count": "reset_zero",
    "population.first_repro_step": "reset_minus_one",
    "population.lineage_birth_generation": "set_config_generation",
    "population.lineage_birth_tick": "set_bundle_tick",
    "population.lineage_founder_id": "inherit_or_create_primary_founder_id",
    "population.lineage_depth": "primary_parent_depth_plus_one",
    "population.lineage_parent_agent_id": "set_primary_parent_agent_id",
    "population.patch_habitat_lineage_id": "inherit_or_create_primary_patch_lineage",
    "population.planned_repro_cooldown": "reset_zero",
}

_POPULATION_IDENTITY_PROJECTIONS: Final[dict[str, str]] = {
    "population.agent_id_to_slot": (
        "write_only_if_child_id_is_within_legacy_projection_capacity"
    ),
}

_POPULATION_ROW_ACTIONS: Final[dict[str, str]] = {
    "population.active_agent_ids": "rebuild_sorted_active_projection_rows",
    "population.active_agent_slots": "rebuild_sorted_active_projection_rows",
    "population.pending_free_agent_ids": "death_transaction_rows_not_child_slots",
    "population.pending_free_slots": "death_transaction_rows_not_child_slots",
    "population.death_agent_ids": "death_transaction_rows_not_child_slots",
    "population.death_slots": "death_transaction_rows_not_child_slots",
    "population.death_reason_codes": "death_transaction_rows_not_child_slots",
    "population.survivor_agent_ids": "active_projection_rows_not_child_slots",
    "population.free_queue": "compact_consumed_free_queue_rows",
    "population.parent_slots": "birth_plan_rows_not_child_slots",
    "population.child_slots": "birth_plan_rows_not_child_slots",
    "population.child_pos_y": "birth_plan_rows_not_child_slots",
    "population.child_pos_x": "birth_plan_rows_not_child_slots",
    "population.child_energy": "birth_plan_rows_not_child_slots",
    "population.mate_slots": "birth_plan_rows_not_child_slots",
    "population.decision_parent_slots": "birth_decision_rows_not_child_slots",
    "population.decision_mate_slots": "birth_decision_rows_not_child_slots",
    "population.decision_adjacent_counts": "birth_decision_rows_not_child_slots",
    "population.decision_reason_codes": "birth_decision_rows_not_child_slots",
    "population.birth_parent_ids": "typed_birth_event_rows_not_child_slots",
    "population.birth_child_ids": "typed_birth_event_rows_not_child_slots",
    "population.mutation_occurred": "mutation_result_rows_not_child_slots",
    "population.mutation_delta_total": "mutation_result_rows_not_child_slots",
}

_UNSUPPORTED_ROUTES: Final[tuple[dict[str, str], ...]] = (
    {
        "route": "GRU_parent_or_mate",
        "disposition": "fail_closed_before_any_authoritative_write",
    },
    {
        "route": "policy_vector_crossover_or_mate_policy_blend",
        "disposition": "unsupported; inherit_primary_parent_fixed_MLP_weights_only",
    },
    {
        "route": "newborn_jungle_uniform_generation_or_reinterpretation",
        "disposition": "unsupported; preserve_existing_slot_value_pending_C5_owner",
    },
    {
        "route": "non_fixed_MLP_memory_or_recurrent_state",
        "disposition": "unsupported; fixed_MLP route stores zero memory dimensions",
    },
    {
        "route": "unclassified_new_per_capacity_tensor",
        "disposition": "registry construction fails closed",
    },
)


@dataclass(frozen=True, slots=True)
class C4NewbornInitializationRecordR11V1:
    path: str
    dtype: str
    trailing_shape: tuple[int, ...]
    dimension_role: str
    initializer: str
    authoritative_owner: str


def _walk_slot_shaped_tensors(
    owner: object, *, prefix: str, batch: int, capacity: int
) -> dict[str, torch.Tensor]:
    found: dict[str, torch.Tensor] = {}

    def walk(value: object, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2 and tuple(value.shape[:2]) == (batch, capacity):
                found[path] = value
            return
        if is_dataclass(value):
            for field in fields(value):
                if field.name == "schema_version":
                    continue
                child = getattr(value, field.name)
                walk(child, f"{path}.{field.name}" if path else field.name)

    walk(owner, prefix)
    return found


def _helper_action_by_tensor_id(bundle: object) -> dict[int, str]:
    actions: dict[int, str] = {}
    groups = (
        (_inherit_tensors, "inherit_primary_parent"),
        (_zero_tensors, "reset_zero"),
        (_minus_one_tensors, "reset_minus_one"),
        (_half_tensors, "fill_neutral_half"),
        (_one_tensors, "fill_one"),
    )
    for getter, action in groups:
        for tensor in getter(bundle):
            prior = actions.setdefault(id(tensor), action)
            if prior != action:
                raise TensorWorldContractError(
                    "C4 newborn tensor appears in conflicting reset groups"
                )
    return actions


def c4_newborn_initialization_registry_r11_v1() -> dict[str, object]:
    """Build the complete source-current fixed-MLP newborn registry."""

    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=8, active=3, width=4, seed=0
    )
    candidates = {
        **_walk_slot_shaped_tensors(
            bundle.base, prefix="base", batch=bundle.batch, capacity=bundle.capacity
        ),
        **_walk_slot_shaped_tensors(
            bundle.population,
            prefix="population",
            batch=bundle.batch,
            capacity=bundle.capacity,
        ),
    }
    helper_actions = _helper_action_by_tensor_id(bundle)
    records: list[C4NewbornInitializationRecordR11V1] = []
    unclassified: list[str] = []

    for path, tensor in sorted(candidates.items()):
        role = "newborn_slot"
        owner = "ordinary_birth_oracle_and_R5_composed_commit"
        initializer = helper_actions.get(id(tensor))

        if path in _DIRECT_NEWBORN_ACTIONS:
            initializer = _DIRECT_NEWBORN_ACTIONS[path]
        if path in _MUTATING_FIXED_MLP_GENOME_ACTIONS:
            initializer = _MUTATING_FIXED_MLP_GENOME_ACTIONS[path]
            owner = "scalar_mutation_oracle_then_composed_commit"
        if path in _POPULATION_SLOT_ACTIONS:
            initializer = _POPULATION_SLOT_ACTIONS[path]
            owner = "population_commit"
        if path in _STATIC_SLOT_ACTIONS:
            role = "static_slot_identity_or_continuation"
            initializer = _STATIC_SLOT_ACTIONS[path]
            owner = "preserved_owner"
        if path in _WORLD_METRIC_ACTIONS:
            role = "world_metric_capacity_axis"
            initializer = _WORLD_METRIC_ACTIONS[path]
            owner = "outside_newborn_transaction"
        if path in _POPULATION_IDENTITY_PROJECTIONS:
            role = "identity_indexed_projection"
            initializer = _POPULATION_IDENTITY_PROJECTIONS[path]
            owner = "population_projection"
        if path in _POPULATION_ROW_ACTIONS:
            role = "transaction_row_not_child_slot"
            initializer = _POPULATION_ROW_ACTIONS[path]
            owner = "population_transaction_projection"

        if initializer is None:
            unclassified.append(path)
            continue
        records.append(
            C4NewbornInitializationRecordR11V1(
                path=path,
                dtype=str(tensor.dtype),
                trailing_shape=tuple(int(value) for value in tensor.shape[2:]),
                dimension_role=role,
                initializer=initializer,
                authoritative_owner=owner,
            )
        )

    if unclassified:
        raise TensorWorldContractError(
            "C4 newborn registry has unclassified per-capacity tensors: "
            + ", ".join(unclassified)
        )
    if len(records) != len(candidates):
        raise TensorWorldContractError("C4 newborn registry completeness mismatch")

    sources = {
        "state_owner": STATE_OWNER,
        "cpu_birth_oracle": CPU_ORACLE,
        "scalar_mutation_oracle": SCALAR_ORACLE,
        "R2_reset_owner": R2_RESET_OWNER,
        "R5_composed_owner": R5_COMPOSED_OWNER,
        "R11_owner": R11_OWNER,
    }
    return {
        "schema_version": C4_NEWBORN_REGISTRY_R11_SCHEMA_VERSION_V1,
        "decision": C4_NEWBORN_REGISTRY_R11_DECISION_V1,
        "supported_route": (
            "R11_fixed_MLP_mated_scalar_mutation_with_primary_policy_inheritance"
        ),
        "candidate_rule": (
            "all_base_and_population_tensors_with_leading_axes_[batch,capacity]"
        ),
        "excluded_owner_scopes": (
            "events_are_event_rows_not_slots",
            "scalar_rng_is_world_continuation_not_slots",
            "event_ring_is_event_rows_not_slots",
            "occupancy_and_patch_maps_are_spatial_not_slots",
        ),
        "records": tuple(asdict(record) for record in records),
        "record_count": len(records),
        "newborn_slot_count": sum(
            record.dimension_role == "newborn_slot" for record in records
        ),
        "unsupported_routes": _UNSUPPORTED_ROUTES,
        "fail_closed_on_unclassified": True,
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_NEWBORN_REGISTRY_R11_DECISION_V1",
    "C4_NEWBORN_REGISTRY_R11_SCHEMA_VERSION_V1",
    "C4NewbornInitializationRecordR11V1",
    "c4_newborn_initialization_registry_r11_v1",
]
