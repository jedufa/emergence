"""CPU/static R6 adversarial coverage mapped to the resident R5 ABI."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_mating import (
    commit_c4_mate_births_oracle_v1,
    plan_c4_mate_births_oracle_v1,
)
from engine.tensor_world.c4_population import apply_c4_death_settlement_oracle_v1
from engine.tensor_world.c4_resident_collision_r6 import (
    C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1,
    C4_RESIDENT_COLLISION_R6_CASES_V1,
    C4_RESIDENT_COLLISION_R6_DECISION_V1,
    C4_RESIDENT_COLLISION_R6_SCHEMA_VERSION_V1,
    c4_resident_collision_r6_source_receipt_v1,
    make_c4_resident_collision_r6_boundary_v1,
    validate_c4_resident_collision_r6_boundary_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
R5_KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu"


def _tensor_leaves(*values: object) -> list[torch.Tensor]:
    leaves: list[torch.Tensor] = []

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            leaves.append(value)
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))

    for value in values:
        visit(value)
    return leaves


def test_c4_r6_static_receipt_binds_r5_abi_and_requires_semantic_wrapper() -> None:
    receipt = c4_resident_collision_r6_source_receipt_v1()

    assert receipt["schema_version"] == C4_RESIDENT_COLLISION_R6_SCHEMA_VERSION_V1
    assert receipt["route"] == "R6_CPU_STATIC_ADVERSARIAL_MAP_TO_R5_ABI"
    assert receipt["decision"] == C4_RESIDENT_COLLISION_R6_DECISION_V1
    assert receipt["batch"] == 4
    assert receipt["authoritative_abi_counts"] == C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1
    assert receipt["native_staging"] == "not_materialized_cpu_fallback_forbidden"
    assert receipt["native_wrapper_required"] is True
    assert receipt["gate_claim"] is False

    source = R5_KERNEL.read_text(encoding="utf-8")
    death_loop = source.index("for (int64_t row = 0; row < candidates; ++row)")
    identity_check = source.index("live.agent_id_to_slot[offset + selected_id]")
    occupancy_check = source.index("turn.occupancy_agent[cell] != selected_id")
    assert death_loop < identity_check < occupancy_check
    assert "turn.virtual_queue[offset + prior] == slot" in source


def test_c4_r6_positive_boundary_maps_every_r5_authoritative_group_and_oracle() -> None:
    boundary = make_c4_resident_collision_r6_boundary_v1(
        "b4_fragmentation_lineage_reuse"
    )
    groups = validate_c4_resident_collision_r6_boundary_v1(boundary)
    assert {name: len(tensors) for name, tensors in groups.items()} == (
        C4_RESIDENT_COLLISION_R6_ABI_COUNTS_V1
    )
    assert all(
        tensor.device.type == "cpu" and tensor.is_contiguous()
        for tensors in groups.values()
        for tensor in tensors
    )

    bundle = boundary.bundle
    population = bundle.population
    apply_c4_death_settlement_oracle_v1(
        bundle, event_sequence=boundary.rng_context.event_sequence
    )
    plan_c4_mate_births_oracle_v1(
        bundle, boundary.config, rng_context=boundary.rng_context
    )
    commit_c4_mate_births_oracle_v1(
        bundle, boundary.config, rng_context=boundary.rng_context
    )

    assert population.death_count.tolist() == [1, 1, 0, 0]
    assert population.birth_count.tolist() == [1, 1, 1, 1]
    assert population.child_slots[:, 0].tolist() == [6, 1, 3, 3]
    assert population.birth_child_ids[:, 0].tolist() == [4, 4, 4, 4]
    assert population.next_agent_id.tolist() == [5, 5, 5, 5]
    assert population.free_queue[0, :5].tolist() == [3, 7, 4, 5, 1]
    assert population.slot_tokens[0, 6].item() == 7
    assert population.slot_tokens[1, 1].item() == 2

    assert bundle.events is not None
    assert bundle.events.typed_event_type[1, :3].tolist() == [4, 2, 1]
    assert bundle.events.typed_generation[1, :3].tolist() == [0, 6, 6]
    assert bundle.events.typed_founder_id[1, :3].tolist() == [777, 111, 111]
    assert bundle.events.typed_lineage_depth[1, :3].tolist() == [4, 1, 1]


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("duplicate_free_queue", "duplicate slot"),
        ("conflicting_occupancy", "occupancy, patch, and live identity"),
        ("conflicting_identity_map", "identity map disagrees"),
    ),
)
def test_c4_r6_invalid_world_fails_before_native_boundary_and_preserves_batch(
    case: str,
    message: str,
) -> None:
    boundary = make_c4_resident_collision_r6_boundary_v1(case)
    assert int(boundary.metadata["invalid_world"]) > 0
    before = [
        tensor.clone()
        for tensor in _tensor_leaves(boundary.bundle, boundary.rng_context)
    ]

    with pytest.raises(TensorWorldContractError, match=message):
        validate_c4_resident_collision_r6_boundary_v1(boundary)

    for expected, actual in zip(
        before,
        _tensor_leaves(boundary.bundle, boundary.rng_context),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, equal_nan=True)


def test_c4_r6_case_inventory_is_exact_and_unknown_case_fails_closed() -> None:
    assert C4_RESIDENT_COLLISION_R6_CASES_V1 == (
        "b4_fragmentation_lineage_reuse",
        "duplicate_free_queue",
        "conflicting_occupancy",
        "conflicting_identity_map",
    )
    with pytest.raises(ValueError, match="unknown C4 resident R6 case"):
        make_c4_resident_collision_r6_boundary_v1("not_a_case")
