from __future__ import annotations

from copy import deepcopy

import pytest

from agents.structural_identity_v1 import new_native_identity
from engine.evolution.p12_stage2_mutation_oracles_v1 import (
    OrdinaryBirthOwnerV1,
    P12MutationOracleError,
    recurrent_ordinary_birth_oracle_v1,
    structural_ordinary_birth_oracle_v1,
)


def _identity(index: int, *, parent: dict | None = None, operator: str = "founder") -> dict:
    return new_native_identity(
        context={"lineage_token": f"p12-oracle-{index}", "generation": index},
        operator_id=operator,
        attempt_index=0,
        operator_role=f"role-{index}",
        parent_module=parent,
        inherited_homology=False,
    )


def _module(index: int, *, parent: dict | None = None, noop: bool = False) -> dict:
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": {
            "expression_scale": 1.0,
            "live_operator": "bounded_scalar_integrator_v1",
            "live_retention": 0.5,
            "live_input_gain": 0.875 if noop else 1.0,
            "live_output_gain": 0.0 if parent is not None else 1.0,
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **_identity(
            index,
            parent=parent,
            operator="duplicate_with_decay" if parent is not None else "founder",
        ),
    }


def _payload(*, noop: bool = False) -> dict:
    source = _module(1)
    descendant = _module(2, parent=source, noop=noop)
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [source, descendant],
        "connections": [],
    }


PARENT = OrdinaryBirthOwnerV1(agent_id=100, slot=2, slot_token=12)
CHILD = OrdinaryBirthOwnerV1(agent_id=(1 << 40) + 7, slot=4, slot_token=18)


@pytest.mark.parametrize(
    ("draw0", "operation"),
    [
        (0.10, "toggle_dormancy"),
        (0.30, "prune_eligible_module"),
        (0.60, "refine_descendant_parameter"),
        (0.80, "refine_descendant_routing_or_expression"),
    ],
)
def test_structural_oracle_covers_operator_bins_and_preserves_parent(
    draw0: float,
    operation: str,
) -> None:
    parent = _payload()
    before = deepcopy(parent)

    child, receipt = structural_ordinary_birth_oracle_v1(
        parent,
        parent_owner=PARENT,
        child_owner=CHILD,
        draw0=draw0,
        draw1=0.6796875,
    )

    assert parent == before
    assert child is not parent
    assert receipt["contract_id"] == "p12-structural-ordinary-birth-reference-oracle-v1"
    assert receipt["rng_draw_budget"] == 2
    assert receipt["committed"] is True
    assert receipt["operation"] == operation
    assert receipt["parent_preserved"] is True
    assert receipt["oracle_sha256"]
    assert receipt["native_cuda_promotion_authorized"] is False
    assert operation in {"toggle_dormancy", "prune_eligible_module", "refine_descendant_parameter", "refine_descendant_routing_or_expression"}


def test_structural_oracle_rejects_stale_slot_reuse_before_mutation() -> None:
    with pytest.raises(P12MutationOracleError, match="newer slot token"):
        structural_ordinary_birth_oracle_v1(
            _payload(),
            parent_owner=PARENT,
            child_owner=OrdinaryBirthOwnerV1(agent_id=101, slot=2, slot_token=12),
            draw0=0.6,
            draw1=0.6796875,
        )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"capacity_available": False}, "capacity_unavailable"),
        ({"publish_success": False}, "child_publication_failed"),
    ],
)
def test_structural_oracle_rolls_back_rejected_publication(
    kwargs: dict[str, bool],
    reason: str,
) -> None:
    parent = _payload()

    child, receipt = structural_ordinary_birth_oracle_v1(
        parent,
        parent_owner=PARENT,
        child_owner=CHILD,
        draw0=0.6,
        draw1=0.6796875,
        **kwargs,
    )

    assert child == parent
    assert child is not parent
    assert receipt["committed"] is False
    assert receipt["status"] == "rejected"
    assert receipt["reason"] == reason
    assert receipt["changed"] is False


def test_recurrent_oracle_replays_receipt_for_high_id_child() -> None:
    parent = _payload()
    before = deepcopy(parent)

    child, receipt = recurrent_ordinary_birth_oracle_v1(
        parent,
        parent_owner=PARENT,
        child_owner=CHILD,
        draw0=0.6,
        draw1=0.6796875,
    )

    assert parent == before
    assert child != parent
    assert receipt["contract_id"] == "p12-recurrent-ordinary-birth-reference-oracle-v1"
    assert receipt["child_owner"]["agent_id"] == (1 << 40) + 7
    assert receipt["replay_receipt_sha256"]
    assert receipt["identity_table_sha256"]
    assert receipt["native_cuda_promotion_authorized"] is False


def test_recurrent_oracle_flags_off_returns_valid_independent_clone() -> None:
    parent = _payload(noop=True)
    before = deepcopy(parent)

    child, receipt = recurrent_ordinary_birth_oracle_v1(
        parent,
        parent_owner=PARENT,
        child_owner=CHILD,
        draw0=0.6,
        draw1=0.6796875,
        mutation_enabled=False,
    )

    assert parent == before
    assert child is not parent
    assert receipt["status"] == "disabled"
    assert receipt["changed"] is False
