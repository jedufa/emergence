"""Completeness tests for the source-current C4 newborn registry."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_newborn_registry_r11 import (
    C4_NEWBORN_REGISTRY_R11_DECISION_V1,
    C4_NEWBORN_REGISTRY_R11_SCHEMA_VERSION_V1,
    c4_newborn_initialization_registry_r11_v1,
)


def _runtime_per_capacity_paths() -> set[str]:
    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=8, active=3, width=4, seed=0
    )
    found: set[str] = set()

    def walk(value: object, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.ndim >= 2 and tuple(value.shape[:2]) == (1, 8):
                found.add(path)
            return
        if is_dataclass(value):
            for field in fields(value):
                if field.name != "schema_version":
                    child = getattr(value, field.name)
                    walk(child, f"{path}.{field.name}" if path else field.name)

    walk(bundle.base, "base")
    walk(bundle.population, "population")
    return found


def test_r11_registry_covers_every_source_current_per_capacity_tensor_once() -> None:
    registry = c4_newborn_initialization_registry_r11_v1()
    records = registry["records"]
    paths = [record["path"] for record in records]

    assert registry["schema_version"] == C4_NEWBORN_REGISTRY_R11_SCHEMA_VERSION_V1
    assert registry["decision"] == C4_NEWBORN_REGISTRY_R11_DECISION_V1
    assert registry["record_count"] == 161
    assert registry["newborn_slot_count"] == 134
    assert len(paths) == len(set(paths))
    assert set(paths) == _runtime_per_capacity_paths()
    assert all(record["initializer"] for record in records)
    assert all(record["authoritative_owner"] for record in records)
    assert registry["fail_closed_on_unclassified"] is True
    assert registry["gate_claim"] is False


def test_r11_registry_has_exact_required_reset_and_derivation_actions() -> None:
    records = {
        row["path"]: row for row in c4_newborn_initialization_registry_r11_v1()["records"]
    }
    expected = {
        "base.base.core.transactions.actions": "reset_minus_one",
        "base.base.core.transactions.moved": "reset_zero",
        "base.base.core.transactions.gathered": "reset_zero",
        "base.base.core.transactions.food_gained": "reset_zero",
        "base.base.core.transactions.observations": "reset_zero",
        "base.base.core.transactions.logits": "reset_zero",
        "base.base.eed.active_y": "reset_minus_one",
        "base.base.eed.active_x": "reset_minus_one",
        "base.base.eed.active_class": "reset_zero",
        "base.base.eed.handling_progress": "reset_zero",
        "base.base.eed.handling_axis": "reset_zero",
        "base.base.policy.packed_weights": "inherit_primary_parent",
        "base.base.effects.pre_policy_health_delta": "reset_zero",
        "base.base.effects.post_action_health_delta": "reset_zero",
        "base.base.effects.hazard_flags": "reset_zero",
        "base.base.world_events.gather_amount": "reset_zero",
        "base.base.world_events.gather_resource_class": "reset_zero",
        "base.base.transactions.observations": "reset_zero",
        "base.base.transactions.logits": "reset_zero",
        "base.agent.age": "reset_zero",
        "base.agent.survival_steps": "reset_zero",
        "base.agent.energy_collected": "reset_zero",
        "base.agent.energy_collected_exact": "reset_zero",
        "base.transactions.pre_energy": "copy_planned_child_energy",
        "base.transactions.pre_health": "set_config_start_health",
        "base.transactions.pre_injury": "reset_zero",
        "base.transactions.pre_survival_steps": "reset_zero",
    }
    assert {path: records[path]["initializer"] for path in expected} == expected
    assert all(records[path]["dimension_role"] == "newborn_slot" for path in expected)


def test_r11_registry_preserves_static_owners_and_rejects_invented_routes() -> None:
    registry = c4_newborn_initialization_registry_r11_v1()
    records = {row["path"]: row for row in registry["records"]}

    assert records["base.base.core.identity.slot_ids"] == {
        "path": "base.base.core.identity.slot_ids",
        "dtype": "torch.int64",
        "trailing_shape": (),
        "dimension_role": "static_slot_identity_or_continuation",
        "initializer": "preserve_physical_slot_identity",
        "authoritative_owner": "preserved_owner",
    }
    assert records["base.base.rng.jungle_uniforms"]["initializer"] == (
        "preserve_existing_slot_value_C5_owned"
    )
    assert records["base.base.hot_metrics.jungle_integers"]["dimension_role"] == (
        "world_metric_capacity_axis"
    )
    assert records["population.agent_id_to_slot"]["dimension_role"] == (
        "identity_indexed_projection"
    )
    assert {
        row["route"] for row in registry["unsupported_routes"]
    } == {
        "GRU_parent_or_mate",
        "policy_vector_crossover_or_mate_policy_blend",
        "newborn_jungle_uniform_generation_or_reinterpretation",
        "non_fixed_MLP_memory_or_recurrent_state",
        "unclassified_new_per_capacity_tensor",
    }


def test_r11_mutating_fixed_mlp_registry_keeps_two_parent_scalar_seam_exact() -> None:
    registry = c4_newborn_initialization_registry_r11_v1()
    records = {row["path"]: row for row in registry["records"]}

    for name in (
        "exploration_bias",
        "aggression_bias",
        "reproduction_threshold",
        "action_temperature",
        "drive_inertia",
        "drive_noise",
        "drive_logit_gain",
        "pred_lr",
    ):
        assert records[f"base.genome.{name}"]["initializer"] == (
            "midpoint_parent_mate_then_scalar_mutation"
        )
    assert records["base.genome.mutation_rate"]["initializer"] == (
        "midpoint_parent_mate_then_rate_mutation"
    )
    assert records["base.genome.drive_setpoints"]["initializer"] == (
        "mean_parent_mate_then_clamp"
    )
    assert records["base.genome.memory_dim"]["initializer"] == (
        "set_zero_fixed_mlp_after_memory_draw"
    )
    assert records["base.genome.policy_mem_dim"]["initializer"] == (
        "set_zero_fixed_mlp_after_memory_draw"
    )
    assert records["base.base.core.policy.packed_weights"]["initializer"] == (
        "inherit_primary_parent"
    )
    assert "crossover" not in records[
        "base.base.core.policy.packed_weights"
    ]["initializer"]


def test_r11_population_capacity_axes_are_not_silently_treated_as_child_slots() -> None:
    registry = c4_newborn_initialization_registry_r11_v1()
    records = {row["path"]: row for row in registry["records"]}

    for path in (
        "population.free_queue",
        "population.parent_slots",
        "population.child_slots",
        "population.decision_parent_slots",
        "population.birth_child_ids",
        "population.mutation_occurred",
    ):
        assert records[path]["dimension_role"] == "transaction_row_not_child_slot"
    for path in (
        "population.slot_tokens",
        "population.scar",
        "population.repro_cooldown",
        "population.offspring_count",
        "population.lineage_depth",
        "population.planned_repro_cooldown",
    ):
        assert records[path]["dimension_role"] == "newborn_slot"


def test_r11_registry_source_receipt_is_hash_bound() -> None:
    sources = c4_newborn_initialization_registry_r11_v1()["sources"]
    assert set(sources) == {
        "state_owner",
        "cpu_birth_oracle",
        "scalar_mutation_oracle",
        "R2_reset_owner",
        "R5_composed_owner",
        "R11_owner",
    }
    assert all(len(row["sha256"]) == 64 for row in sources.values())
