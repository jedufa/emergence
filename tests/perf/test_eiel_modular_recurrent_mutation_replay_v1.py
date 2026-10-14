from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from agents.structural_identity_v1 import new_native_identity
from evolution.structural_sustainable_growth_v1 import mutate_sustainable_growth
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    build_modular_recurrent_mutation_receipt_v1,
    float64_to_bits,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_replay_v1 import (
    ModularRecurrentMutationReplayError,
    replay_modular_recurrent_mutation_v1,
)


SOURCE_SHA = "11" * 32
ANCESTRY_SHA = "22" * 32


def _identity(
    index: int, *, parent: dict | None = None, operator: str = "founder"
) -> dict:
    return new_native_identity(
        context={"lineage_token": f"mutation-replay-{index}", "generation": index},
        operator_id=operator,
        attempt_index=0,
        operator_role=f"role-{index}",
        parent_module=parent,
        inherited_homology=False,
    )


def _module(index: int, *, parent: dict | None = None) -> dict:
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": {
            "expression_scale": 1.0,
            "live_operator": "bounded_scalar_integrator_v1",
            "live_retention": 0.5,
            "live_input_gain": 1.0,
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


def _payload(*, duplicate: bool = True, noop: bool = False) -> dict:
    source = _module(1)
    modules = [source]
    if duplicate:
        descendant = _module(2, parent=source)
        if noop:
            descendant["params"]["live_input_gain"] = 0.875
        modules.append(descendant)
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": modules,
        "connections": [],
    }


def _identity_table(parent: dict) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(module[field])
                for module in parent["modules"]
                for field in ("module_instance_id", "homology_id")
            }
        )
    )


def _transaction(
    parent: dict,
    table: tuple[str, ...],
    *,
    status: str = "accepted",
    reason: str = "proposal_valid",
    changed_dimension: str = "live_input_gain",
    changed: bool = True,
) -> dict[str, object]:
    selected = parent["modules"][1] if status == "accepted" else None
    return {
        "transaction_kind": "turnover",
        "transaction_ordinal": 0,
        "parent_agent_id": 100,
        "parent_slot": 2,
        "parent_slot_token": 12,
        "child_agent_id": 200,
        "child_slot": 4,
        "child_slot_token": 18,
        "attempt_ordinal": 0,
        "draw0_float64_bits": float64_to_bits(0.6),
        "draw1_float64_bits": float64_to_bits(0.6796875),
        "operation": "refine_descendant_parameter",
        "status": status,
        "reason": reason,
        "selected_instance_id_index": (
            table.index(selected["module_instance_id"]) if selected else -1
        ),
        "selected_homology_id_index": (
            table.index(selected["homology_id"]) if selected else -1
        ),
        "changed_dimension": changed_dimension,
        "changed": changed,
    }


def _receipt(transaction: dict[str, object]):
    return build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256=SOURCE_SHA,
        ancestry_tape_sha256=ANCESTRY_SHA,
        transactions=[transaction],
    )


def _redigest(wire: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in wire.items() if key != "canonical_sha256"}
    wire["canonical_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return wire


def test_exact_float64_replay_accepts_integrator_input_gain_and_clones() -> None:
    parent = _payload()
    before = deepcopy(parent)
    table = _identity_table(parent)
    receipt = _receipt(_transaction(parent, table))

    child = replay_modular_recurrent_mutation_v1(parent, receipt, identity_table=table)

    assert parent == before
    assert child is not parent
    assert child["modules"] is not parent["modules"]
    selected_id = parent["modules"][1]["module_instance_id"]
    changed = next(
        module
        for module in child["modules"]
        if module["module_instance_id"] == selected_id
    )
    assert changed["params"]["live_input_gain"] == 0.875
    assert parent["modules"][1]["params"]["live_input_gain"] == 1.0


def test_ineligible_replay_returns_validated_independent_parent_clone() -> None:
    parent = _payload(duplicate=False)
    before = deepcopy(parent)
    table = _identity_table(parent)
    transaction = _transaction(
        {"modules": [parent["modules"][0], parent["modules"][0]]},
        table,
        status="ineligible",
        reason="no_duplicate_descendant_parameter",
        changed_dimension="none",
        changed=False,
    )
    transaction["selected_instance_id_index"] = -1
    transaction["selected_homology_id_index"] = -1

    child = replay_modular_recurrent_mutation_v1(
        parent, _receipt(transaction), identity_table=table
    )

    assert parent == before
    expected = mutate_sustainable_growth(parent, 0.6, 0.6796875).payload
    assert child == expected
    assert child is not parent
    assert child["modules"] is not parent["modules"]


def test_noop_replay_matches_no_selection_and_preserves_input() -> None:
    parent = _payload(noop=True)
    before = deepcopy(parent)
    table = _identity_table(parent)
    transaction = _transaction(
        parent,
        table,
        status="noop",
        reason="refined_value_unchanged",
        changed_dimension="none",
        changed=False,
    )
    transaction["selected_instance_id_index"] = -1
    transaction["selected_homology_id_index"] = -1

    child = replay_modular_recurrent_mutation_v1(
        parent, _receipt(transaction), identity_table=table
    )

    assert parent == before
    expected = mutate_sustainable_growth(parent, 0.6, 0.6796875).payload
    assert child == expected
    assert child is not parent


def test_disabled_replay_uses_authoritative_disabled_path() -> None:
    parent = _payload()
    table = _identity_table(parent)
    transaction = _transaction(
        parent,
        table,
        status="disabled",
        reason="operator_disabled",
        changed_dimension="none",
        changed=False,
    )
    transaction["selected_instance_id_index"] = -1
    transaction["selected_homology_id_index"] = -1

    child = replay_modular_recurrent_mutation_v1(
        parent, _receipt(transaction), identity_table=table
    )
    expected = mutate_sustainable_growth(parent, 0.6, 0.6796875, enabled=False).payload
    assert child == expected
    assert child is not parent


def test_receipt_digest_tamper_fails_closed() -> None:
    parent = _payload()
    table = _identity_table(parent)
    wire = _receipt(_transaction(parent, table)).to_dict()
    wire["transactions"][0]["reason"] = "tampered"  # type: ignore[index]

    with pytest.raises(ModularRecurrentMutationReplayError, match="receipt is invalid"):
        replay_modular_recurrent_mutation_v1(parent, wire, identity_table=table)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reason", "wrong_but_canonical", "reason"),
        ("changed_dimension", "live_output_gain", "changed_dimension"),
        ("selected_instance_id_index", 0, "selected_instance_id_index"),
    ],
)
def test_redigested_outcome_tamper_fails_closed(
    field: str, value: object, match: str
) -> None:
    parent = _payload()
    table = _identity_table(parent)
    wire = _receipt(_transaction(parent, table)).to_dict()
    wire["transactions"][0][field] = value  # type: ignore[index]

    with pytest.raises(ModularRecurrentMutationReplayError, match=match):
        replay_modular_recurrent_mutation_v1(
            parent, _redigest(wire), identity_table=table
        )


@pytest.mark.parametrize("damage", ["unsorted", "duplicate", "missing", "empty"])
def test_identity_table_errors_fail_closed(damage: str) -> None:
    parent = _payload()
    table = _identity_table(parent)
    receipt = _receipt(_transaction(parent, table))
    if damage == "unsorted":
        damaged = tuple(reversed(table))
    elif damage == "duplicate":
        damaged = tuple(sorted((*table, table[0])))
    elif damage == "missing":
        damaged = table[1:]
    else:
        damaged = (*table, "")

    with pytest.raises(ModularRecurrentMutationReplayError, match="identity_table"):
        replay_modular_recurrent_mutation_v1(parent, receipt, identity_table=damaged)


def test_receipt_identity_index_out_of_range_fails_closed() -> None:
    parent = _payload()
    table = _identity_table(parent)
    wire = _receipt(_transaction(parent, table)).to_dict()
    wire["transactions"][0]["selected_instance_id_index"] = len(table)  # type: ignore[index]

    with pytest.raises(ModularRecurrentMutationReplayError, match="outside"):
        replay_modular_recurrent_mutation_v1(
            parent, _redigest(wire), identity_table=table
        )


def test_duplicate_parent_instance_identity_fails_closed() -> None:
    parent = _payload()
    parent["modules"][1]["module_instance_id"] = parent["modules"][0][
        "module_instance_id"
    ]
    table = _identity_table(parent)
    transaction = _transaction(parent, table)

    with pytest.raises(ModularRecurrentMutationReplayError, match="must be unique"):
        replay_modular_recurrent_mutation_v1(
            parent, _receipt(transaction), identity_table=table
        )
