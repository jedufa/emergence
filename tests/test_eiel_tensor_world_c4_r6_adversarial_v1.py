"""CPU-only contract tests for C4 R6 identity/slot adversarial inputs."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest
import torch

from engine.tensor_world.c4_r6_adversarial import (
    C4_R6_ADVERSARIAL_CASES_V1,
    C4_R6_ADVERSARIAL_DECISION_V1,
    C4_R6_ADVERSARIAL_SCHEMA_VERSION_V1,
    c4_r6_adversarial_contract_v1,
    make_c4_r6_adversarial_fixture_v1,
    validate_c4_r6_identity_slot_preflight_v1,
)
from engine.tensor_world.c4_transaction import (
    execute_c4_population_transaction_oracle_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _tensor_leaves(value: object) -> list[torch.Tensor]:
    leaves: list[torch.Tensor] = []

    def visit(owner: object) -> None:
        if isinstance(owner, torch.Tensor):
            leaves.append(owner)
        elif is_dataclass(owner):
            for field in fields(owner):
                visit(getattr(owner, field.name))

    visit(value)
    return leaves


def test_c4_r6_contract_is_b4_cpu_only_and_has_no_gate_claim() -> None:
    contract = c4_r6_adversarial_contract_v1()

    assert contract["schema_version"] == C4_R6_ADVERSARIAL_SCHEMA_VERSION_V1
    assert contract["route"] == "CPU_ORACLE_PREPARATION_ONLY"
    assert contract["batch"] == 4
    assert tuple(contract["cases"]) == C4_R6_ADVERSARIAL_CASES_V1
    assert contract["decision"] == C4_R6_ADVERSARIAL_DECISION_V1
    assert contract["gate_claim"] is False


def test_c4_r6_b4_fragmentation_cross_world_ids_and_lineage_before_reuse() -> None:
    bundle, config, spawn_uniforms, metadata = (
        make_c4_r6_adversarial_fixture_v1(
            "b4_fragmentation_lineage_reuse"
        )
    )
    core = bundle.base.base.core
    population = bundle.population
    assert bundle.batch == 4
    assert core.agents.agent_id[:, :3].tolist() == [[1, 2, 3]] * 4

    validate_c4_r6_identity_slot_preflight_v1(bundle)
    receipt = execute_c4_population_transaction_oracle_v1(
        bundle,
        birth_config=config,
        spawn_uniforms=spawn_uniforms,
    )

    assert receipt.status == "PREPARATION_PASS"
    assert receipt.counts_after["death_count"] == (1, 1, 0, 0)
    assert receipt.counts_after["birth_count"] == (1, 1, 1, 1)
    assert receipt.counts_after["active_count"] == (3, 3, 4, 4)
    assert population.child_slots[:, 0].tolist() == [6, 1, 3, 3]
    assert population.birth_child_ids[:, 0].tolist() == [4, 4, 4, 4]
    assert population.next_agent_id.tolist() == [5, 5, 5, 5]

    # The old fragmented prefix stays ahead of the newly released death slot;
    # consuming slot 6 leaves the exact source-order remainder.
    assert population.free_queue_count[0].item() == 5
    assert population.free_queue[0, :5].tolist() == [3, 7, 4, 5, 1]
    assert population.slot_tokens[0, 6].item() == 7

    # World 1 has no older free entry, so slot 1 is reused immediately.  The
    # death row must retain the old occupant's lineage before newborn reset.
    assert population.free_queue_count[1].item() == 0
    assert population.slot_tokens[1, 1].item() == 2
    assert core.agents.agent_id[1, 1].item() == 4
    assert bundle.events is not None
    assert bundle.events.typed_event_type[1, :2].tolist() == [4, 1]
    assert bundle.events.typed_generation[1, :2].tolist() == [0, 6]
    assert bundle.events.typed_actor_id[1, :2].tolist() == [2, 1]
    assert bundle.events.typed_target_id[1, :2].tolist() == [-1, 4]
    assert bundle.events.typed_founder_id[1, :2].tolist() == [
        metadata["dead_founder_id"],
        metadata["newborn_founder_id"],
    ]
    assert bundle.events.typed_lineage_depth[1, :2].tolist() == [4, 1]
    assert population.lineage_founder_id[1, 1].item() == 111
    assert population.lineage_parent_agent_id[1, 1].item() == 1

    # All identity/occupancy mirrors remain valid after the mixed B4 commit.
    validate_c4_r6_identity_slot_preflight_v1(bundle)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("duplicate_free_queue", "duplicate slot"),
        ("conflicting_occupancy", "occupancy, patch, and live identity"),
        ("conflicting_identity_map", "identity map disagrees"),
    ),
)
def test_c4_r6_invalid_later_world_preserves_every_batch_tensor(
    case: str,
    message: str,
) -> None:
    bundle, _config, _spawn_uniforms, metadata = (
        make_c4_r6_adversarial_fixture_v1(case)
    )
    assert int(metadata["invalid_world"]) > 0
    before = [tensor.clone() for tensor in _tensor_leaves(bundle)]

    with pytest.raises(TensorWorldContractError, match=message):
        validate_c4_r6_identity_slot_preflight_v1(bundle)

    for expected, actual in zip(before, _tensor_leaves(bundle), strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)


def test_c4_r6_unknown_case_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown C4 R6 adversarial case"):
        make_c4_r6_adversarial_fixture_v1("not_a_case")
