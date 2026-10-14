"""Poison-slot regression for the C4 ordinary no-mutation birth oracle."""

from __future__ import annotations

from dataclasses import fields

import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_population import (
    TensorWorldC4BirthConfigV1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.c4_resident_repair import (
    c4_newborn_dead_slot_reset_coverage_manifest_v1,
)


_NEWBORN_SLOT_CLASSES = {
    "agent_slot_state",
    "policy_slot_state",
    "transaction_slot_state",
    "slot_resource_state",
    "slot_effect_state",
    "genome_slot_state",
    "adaptation_slot_state",
    "policy_runtime_slot_state",
    "lifecycle_slot_state",
}
_ADMISSION_IDENTITY_PATHS = {
    "base.base.core.agents.alive",
    "base.base.core.agents.agent_id",
}
_REVIEW_REQUIRED_PATHS = {
    "base.base.core.transactions.actions",
    "base.base.core.transactions.moved",
    "base.base.core.transactions.gathered",
    "base.base.core.transactions.food_gained",
    "base.base.core.transactions.observations",
    "base.base.core.transactions.logits",
    "base.base.eed.active_y",
    "base.base.eed.active_x",
    "base.base.eed.active_class",
    "base.base.eed.handling_progress",
    "base.base.eed.handling_axis",
    "base.base.policy.packed_weights",
    "base.base.effects.pre_policy_health_delta",
    "base.base.effects.post_action_health_delta",
    "base.base.effects.hazard_flags",
    "base.base.world_events.gather_amount",
    "base.base.world_events.gather_resource_class",
    "base.base.transactions.observations",
    "base.base.transactions.logits",
    "base.agent.age",
    "base.agent.survival_steps",
    "base.agent.energy_collected",
    "base.agent.energy_collected_exact",
}


def _resolve(bundle, path: str) -> torch.Tensor:
    value = bundle
    for component in path.split("."):
        value = getattr(value, component)
    assert isinstance(value, torch.Tensor)
    return value


def _poison_row(tensor: torch.Tensor, *, world: int, slot: int) -> None:
    row = tensor[world, slot]
    if tensor.dtype == torch.bool:
        row.fill_(True)
    elif tensor.is_floating_point():
        row.fill_(73.25)
    else:
        row.fill_(73)


def _poison_all_newborn_slot_state(bundle, *, world: int, slot: int) -> set[str]:
    manifest = c4_newborn_dead_slot_reset_coverage_manifest_v1()
    poisoned: set[str] = set()
    for record in manifest["fields"]:
        path = str(record["path"])
        if (
            record["classification"] not in _NEWBORN_SLOT_CLASSES
            or path in _ADMISSION_IDENTITY_PATHS
        ):
            continue
        tensor = _resolve(bundle, path)
        # Some transaction-owner fields (notably C3 hot_counters) are
        # transaction-global despite sharing the generated classification.
        if tensor.ndim < 2 or tensor.shape[1] != bundle.capacity:
            continue
        _poison_row(tensor, world=world, slot=slot)
        poisoned.add(path)
    return poisoned


def test_c4_no_mutation_birth_scrubs_every_poisoned_newborn_slot_surface() -> None:
    poisoned, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    expected, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    config = TensorWorldC4BirthConfigV1(max_births=1, allow_mutation=False)
    spawn_uniforms = torch.zeros((1, 8, 3), dtype=torch.float64)

    for bundle in (poisoned, expected):
        plan_c4_ordinary_births_oracle_v1(
            bundle,
            config,
            spawn_uniforms=spawn_uniforms,
        )
        assert bundle.population.planned_count.tolist() == [1]

    child_slot = int(poisoned.population.child_slots[0, 0])
    assert child_slot == int(expected.population.child_slots[0, 0])
    poisoned.population.planned_repro_cooldown[0, child_slot] = 17
    poisoned_paths = _poison_all_newborn_slot_state(
        poisoned,
        world=0,
        slot=child_slot,
    )
    assert _REVIEW_REQUIRED_PATHS.issubset(poisoned_paths)

    # These are continuation/static identity surfaces, not newborn reset rows.
    poisoned.base.base.rng.jungle_uniforms[0, child_slot] = 0.314159
    jungle_before = poisoned.base.base.rng.jungle_uniforms.clone()
    domain_world_ids_before = poisoned.base.base.core.identity.domain_world_ids.clone()
    slot_ids_before = poisoned.base.base.core.identity.slot_ids.clone()
    assert poisoned.scalar_rng is not None
    scalar_rng_before = {
        field.name: getattr(poisoned.scalar_rng, field.name).clone()
        for field in fields(type(poisoned.scalar_rng))
        if field.name != "schema_version"
    }

    commit_c4_ordinary_births_oracle_v1(poisoned, config)
    commit_c4_ordinary_births_oracle_v1(expected, config)

    for path in sorted(poisoned_paths):
        torch.testing.assert_close(
            _resolve(poisoned, path)[0, child_slot],
            _resolve(expected, path)[0, child_slot],
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda message, field_path=path: f"{field_path}: {message}",
        )

    # The C2 fixed-MLP projection is inherited from the selected parent.
    parent_slot = int(poisoned.population.parent_slots[0, 0])
    torch.testing.assert_close(
        poisoned.base.base.policy.packed_weights[0, child_slot],
        poisoned.base.base.policy.packed_weights[0, parent_slot],
        rtol=0,
        atol=0,
    )
    assert float(poisoned.base.base.eed.handling_axis[0, child_slot]) == 0.0
    assert int(poisoned.population.planned_repro_cooldown[0, child_slot]) == 0

    # C4 does not reinterpret the C2 Jungle draw surface.  It is preserved for
    # the named C5 continuation owner instead of assigning newborn RNG values.
    assert torch.equal(poisoned.base.base.rng.jungle_uniforms, jungle_before)
    assert torch.equal(
        poisoned.base.base.core.identity.domain_world_ids,
        domain_world_ids_before,
    )
    assert torch.equal(poisoned.base.base.core.identity.slot_ids, slot_ids_before)
    assert poisoned.scalar_rng is not None
    for name, before in scalar_rng_before.items():
        assert torch.equal(getattr(poisoned.scalar_rng, name), before), name
    poisoned.validate()
