from __future__ import annotations

from copy import deepcopy

import pytest

from agents.modular_genome_v1 import (
    canonical_hash_from_genome_dict,
    canonical_payload_from_genome_dict,
)
from agents.structural_identity_v1 import new_native_identity
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    build_modular_recurrent_mutation_receipt_v1,
    float64_to_bits,
    turnover_mutation_ancestry_sha256_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_replay_v1 import (
    replay_modular_recurrent_mutation_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_replay_v1 import (
    ModularRecurrentMutationSequenceReplayError,
    replay_modular_recurrent_mutation_sequence_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    build_modular_recurrent_mutation_sequence_v1,
)
from evolution.structural_observation_routing_v1 import ROUTING_BLOCK_KEY
from evolution.structural_output_routing_v1 import OUTPUT_ROUTING_BLOCK_KEY


SOURCE_SHA = "11" * 32
PROFILE = "b2-12x32-v1"


def _identity(
    index: int, *, parent: dict | None = None, operator: str = "founder"
) -> dict:
    return new_native_identity(
        context={"lineage_token": f"sequence-replay-{index}", "generation": index},
        operator_id=operator,
        attempt_index=0,
        operator_role=f"role-{index}",
        parent_module=parent,
        inherited_homology=False,
    )


def _module(index: int, *, parent: dict | None = None) -> dict:
    return {
        "module_id": f"m{index - 1:03d}",
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


def _payload(first_module_index: int = 1) -> dict:
    source = _module(first_module_index)
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [source, _module(first_module_index + 1, parent=source)],
        "connections": [],
    }


def _identity_table(records: list[dict | None]) -> tuple[str, ...]:
    strings: set[str] = set()
    for record in records:
        if record is None:
            continue
        genome = canonical_payload_from_genome_dict(record["genome"])
        for module in genome["modules"]:
            for field in (
                "module_id",
                "module_instance_id",
                "homology_id",
                "module_type",
            ):
                strings.add(str(module.get(field, "")))
            params = module.get("params", {})
            for block_name in (ROUTING_BLOCK_KEY, OUTPUT_ROUTING_BLOCK_KEY):
                block = params.get(block_name)
                if isinstance(block, dict):
                    for binding in block.get("bindings", []):
                        strings.add(str(binding.get("semantic_id", "")))
        for connection in genome["connections"]:
            for field in (
                "connection_id",
                "src_slot",
                "dst_slot",
                "value_type",
                "channel",
            ):
                strings.add(str(connection.get(field, "")))
    strings.discard("")
    return tuple(sorted(strings))


def _transaction(
    *,
    parent: dict,
    table: tuple[str, ...],
    parent_agent_id: int,
    parent_slot: int,
    parent_slot_token: int,
    child_agent_id: int,
    child_slot: int,
    child_slot_token: int,
    draw1: float,
    draw0: float = 0.6,
) -> tuple[dict[str, object], dict]:
    selected = parent["modules"][1]
    transaction = {
        "transaction_kind": "turnover",
        "transaction_ordinal": 0,
        "parent_agent_id": parent_agent_id,
        "parent_slot": parent_slot,
        "parent_slot_token": parent_slot_token,
        "child_agent_id": child_agent_id,
        "child_slot": child_slot,
        "child_slot_token": child_slot_token,
        "attempt_ordinal": 0,
        "draw0_float64_bits": float64_to_bits(draw0),
        "draw1_float64_bits": float64_to_bits(draw1),
        "operation": "refine_descendant_parameter",
        "status": "accepted",
        "reason": "proposal_valid",
        "selected_instance_id_index": table.index(selected["module_instance_id"]),
        "selected_homology_id_index": table.index(selected["homology_id"]),
        "changed_dimension": "live_input_gain",
        "changed": True,
    }
    receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id=PROFILE,
        source_packed_sha256=SOURCE_SHA,
        ancestry_tape_sha256=turnover_mutation_ancestry_sha256_v1(
            parent_slot=parent_slot,
            child_slot=child_slot,
            parent_agent_id=parent_agent_id,
            child_agent_id=child_agent_id,
            child_slot_token=child_slot_token,
        ),
        transactions=[transaction],
    )
    child = replay_modular_recurrent_mutation_v1(parent, receipt, identity_table=table)
    return transaction, child


def _entry(
    transaction: dict[str, object],
    *,
    sequence_index: int,
    generation: int,
    parent: dict,
    child: dict,
) -> dict[str, object]:
    return {
        "sequence_index": sequence_index,
        "generation": generation,
        "tick": 0,
        **{
            key: value
            for key, value in transaction.items()
            if key
            not in {
                "transaction_kind",
                "transaction_ordinal",
                "attempt_ordinal",
            }
        },
        "pre_genome_sha256": canonical_hash_from_genome_dict(parent),
        "post_genome_sha256": canonical_hash_from_genome_dict(child),
        "committed": True,
    }


def _chain_fixture() -> tuple[object, list[dict | None], tuple[str, ...], list[dict]]:
    initial = [
        {"agent_id": 100, "agent_slot_token": 10, "genome": _payload()},
        None,
        None,
    ]
    table = _identity_table(initial)
    genomes = [initial[0]["genome"]]
    coordinates = (
        (100, 0, 10, 200, 1, 20, 0.65),
        (200, 1, 20, 300, 2, 30, 0.68),
        (300, 2, 30, 400, 1, 40, 0.72),
    )
    entries: list[dict[str, object]] = []
    for index, coordinate in enumerate(coordinates):
        transaction, child = _transaction(
            parent=genomes[-1],
            table=table,
            parent_agent_id=coordinate[0],
            parent_slot=coordinate[1],
            parent_slot_token=coordinate[2],
            child_agent_id=coordinate[3],
            child_slot=coordinate[4],
            child_slot_token=coordinate[5],
            draw1=coordinate[6],
        )
        entries.append(
            _entry(
                transaction,
                sequence_index=index,
                generation=index + 2,
                parent=genomes[-1],
                child=child,
            )
        )
        genomes.append(child)
    sequence = build_modular_recurrent_mutation_sequence_v1(
        profile_id=PROFILE,
        source_packed_sha256=SOURCE_SHA,
        entries=entries,
    )
    return sequence, initial, table, genomes


def _rebuilt_sequence(sequence: object, entries: list[dict[str, object]]):
    return build_modular_recurrent_mutation_sequence_v1(
        profile_id=sequence.profile_id,
        source_packed_sha256=sequence.source_packed_sha256,
        entries=[
            {key: value for key, value in entry.items() if key != "receipt_sha256"}
            for entry in entries
        ],
    )


def test_replays_child_of_child_chain_and_returns_independent_exact_records() -> None:
    sequence, initial, table, genomes = _chain_fixture()
    before = deepcopy(initial)

    result = replay_modular_recurrent_mutation_sequence_v1(
        sequence, initial, identity_table=table
    )

    assert initial == before
    assert [record["agent_id"] for record in result.final_records if record] == [
        100,
        400,
        300,
    ]
    assert result.final_records[0]["genome"] == genomes[0]
    assert result.final_records[1]["genome"] == genomes[3]
    assert result.final_records[2]["genome"] == genomes[2]
    assert result.final_origin_slot_by_record_slot == (0, 0, 0)
    assert result.origin_slot_by_agent_id == {100: 0, 200: 0, 300: 0, 400: 0}
    assert [entry.parent_agent_id for entry in result.entry_results] == [
        100,
        200,
        300,
    ]
    assert all(entry.origin_slot == 0 for entry in result.entry_results)
    assert result.records is result.final_records
    assert result.replay_results is result.entry_results
    assert result.final_records is not initial
    assert result.final_records[0] is not initial[0]
    assert result.entry_results[0].child_record is not result.final_records[2]


def test_empty_sequence_returns_an_independent_validated_sidecar() -> None:
    _, initial, table, _ = _chain_fixture()
    sequence = build_modular_recurrent_mutation_sequence_v1(
        profile_id=PROFILE,
        source_packed_sha256=SOURCE_SHA,
        entries=[],
    )

    result = replay_modular_recurrent_mutation_sequence_v1(
        sequence, initial, identity_table=table
    )

    assert result.final_records == tuple(initial)
    assert result.final_records[0] is not initial[0]
    assert result.entry_results == ()
    assert result.final_origin_slot_by_record_slot == (0, None, None)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("parent_agent_id", 999, "stale parent_agent_id"),
        ("parent_slot_token", 999, "stale parent_slot_token"),
    ],
)
def test_stale_parent_ownership_fails_closed(
    field: str, value: int, match: str
) -> None:
    sequence, initial, table, _ = _chain_fixture()
    entries = [
        entry.to_dict()
        for entry in sequence.entries[: sequence.count]
        if entry is not None
    ]
    entries[1][field] = value

    with pytest.raises(ModularRecurrentMutationSequenceReplayError, match=match):
        replay_modular_recurrent_mutation_sequence_v1(
            _rebuilt_sequence(sequence, entries),
            initial,
            identity_table=table,
        )


def test_missing_current_parent_record_fails_closed() -> None:
    sequence, initial, table, _ = _chain_fixture()
    entry = sequence.entries[0].to_dict()
    entry["parent_slot"] = 2

    with pytest.raises(
        ModularRecurrentMutationSequenceReplayError, match="no current record"
    ):
        replay_modular_recurrent_mutation_sequence_v1(
            _rebuilt_sequence(sequence, [entry]),
            initial,
            identity_table=table,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pre_genome_sha256", "aa" * 32, "pre_genome_sha256 mismatch"),
        ("post_genome_sha256", "bb" * 32, "post_genome_sha256 mismatch"),
    ],
)
def test_wrong_canonical_genome_digest_fails_closed(
    field: str, value: str, match: str
) -> None:
    sequence, initial, table, _ = _chain_fixture()
    entry = sequence.entries[0].to_dict()
    entry[field] = value

    with pytest.raises(ModularRecurrentMutationSequenceReplayError, match=match):
        replay_modular_recurrent_mutation_sequence_v1(
            _rebuilt_sequence(sequence, [entry]),
            initial,
            identity_table=table,
        )


def test_duplicate_initial_live_owner_fails_closed() -> None:
    sequence, initial, table, _ = _chain_fixture()
    duplicate = deepcopy(initial[0])
    records = [initial[0], duplicate, None]

    with pytest.raises(
        ModularRecurrentMutationSequenceReplayError, match="duplicate live owner"
    ):
        replay_modular_recurrent_mutation_sequence_v1(
            sequence, records, identity_table=table
        )


def test_existing_child_owner_id_fails_closed_before_replacement() -> None:
    sequence, initial, table, _ = _chain_fixture()
    existing_child = deepcopy(initial[0])
    existing_child["agent_id"] = 200
    existing_child["agent_slot_token"] = 90
    records = [initial[0], None, existing_child]

    with pytest.raises(
        ModularRecurrentMutationSequenceReplayError,
        match="existing or historical child_agent_id",
    ):
        replay_modular_recurrent_mutation_sequence_v1(
            sequence, records, identity_table=table
        )


def test_identity_table_must_exactly_describe_initial_sidecar() -> None:
    sequence, initial, table, _ = _chain_fixture()

    with pytest.raises(
        ModularRecurrentMutationSequenceReplayError,
        match="does not exactly describe",
    ):
        replay_modular_recurrent_mutation_sequence_v1(
            sequence,
            initial,
            identity_table=tuple(sorted((*table, "unused-identity"))),
        )


def test_replacement_fails_when_it_changes_the_identity_table() -> None:
    initial = [
        {"agent_id": 100, "agent_slot_token": 10, "genome": _payload()},
        {"agent_id": 900, "agent_slot_token": 90, "genome": _payload(10)},
    ]
    table = _identity_table(initial)
    parent = initial[0]["genome"]
    transaction, child = _transaction(
        parent=parent,
        table=table,
        parent_agent_id=100,
        parent_slot=0,
        parent_slot_token=10,
        child_agent_id=200,
        child_slot=1,
        child_slot_token=20,
        draw1=0.65,
    )
    sequence = build_modular_recurrent_mutation_sequence_v1(
        profile_id=PROFILE,
        source_packed_sha256=SOURCE_SHA,
        entries=[
            _entry(
                transaction,
                sequence_index=0,
                generation=2,
                parent=parent,
                child=child,
            )
        ],
    )

    with pytest.raises(
        ModularRecurrentMutationSequenceReplayError,
        match="changed the fixed record identity_table",
    ):
        replay_modular_recurrent_mutation_sequence_v1(
            sequence, initial, identity_table=table
        )
