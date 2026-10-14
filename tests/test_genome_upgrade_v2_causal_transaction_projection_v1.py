from __future__ import annotations

from dataclasses import replace

import pytest

from engine.evolution.genome_upgrade_v2_causal_transaction_projection_v1 import (
    SCHEMA_VERSION,
    GenomeV2CausalTransactionIdentityV1,
    GenomeV2CausalTransactionProjectionError,
    GenomeV2CausalTransactionRecordV1,
    join_causal_transaction_records_v1,
)


def _identity(
    *,
    child_id: int = 17,
    package: str = "b" * 64,
    mate_parent_id: int = 5,
    parent_slot: int = 1,
    mate_parent_slot: int = 2,
) -> GenomeV2CausalTransactionIdentityV1:
    return GenomeV2CausalTransactionIdentityV1.create(
        event_kind="birth_or_turnover_child",
        event_sequence=9,
        child_id=child_id,
        cell_id="cell-a",
        generation=2,
        tick=31,
        destination_slot=4,
        parent_slot=parent_slot,
        mate_parent_slot=mate_parent_slot,
        parent_id=3,
        mate_parent_id=mate_parent_id,
        lineage_founder_id=1,
        lineage_depth=2,
        capability_contract_sha256="a" * 64,
        package_id_sha256=package,
        artifact_sha256="c" * 64,
        execution_route_sha256="d" * 64,
    )


def _record(
    phase: str,
    *,
    identity: GenomeV2CausalTransactionIdentityV1 | None = None,
) -> GenomeV2CausalTransactionRecordV1:
    return GenomeV2CausalTransactionRecordV1.create(
        record_phase=phase,
        identity=identity or _identity(),
        genotype_summary={"sha256": "e" * 64},
        product_summary={"energy_flux": 120},
        regulatory_summary={"channels_q10": [1, -1, 2, -2]},
        context_summary={"energy_deficit": 512},
        phenotype_summary={"metabolic_efficiency": 80},
        cost_summary={
            "expression_cost_q10": 7,
            "regulatory_cost_q10": 3,
        },
        applied_consequence={"energy_debit": 0.25},
        resulting_state={"energy": 8.75},
    )


def test_identity_is_content_addressed_and_round_trips() -> None:
    identity = _identity()

    assert SCHEMA_VERSION == "genome_v2_causal_transaction_projection_v3"
    assert len(identity.transaction_id) == 64
    assert identity.parent_slot == 1
    assert identity.mate_parent_slot == 2
    assert identity.mate_parent_id == 5
    assert (
        GenomeV2CausalTransactionIdentityV1.from_mapping(
            identity.as_dict()
        )
        == identity
    )
    changed = _identity(package="f" * 64)
    assert changed.transaction_id != identity.transaction_id
    changed_mate = _identity(mate_parent_id=7)
    assert changed_mate.transaction_id != identity.transaction_id
    changed_parent_slot = _identity(parent_slot=3)
    assert changed_parent_slot.transaction_id != identity.transaction_id


def test_identity_requires_distinct_parent_identities() -> None:
    with pytest.raises(
        GenomeV2CausalTransactionProjectionError,
        match="parent",
    ):
        _identity(mate_parent_id=3)


def test_record_requires_explicit_regulatory_context_and_cost_dimensions() -> None:
    record = _record("event_time")
    payload = record.as_dict()

    assert payload["regulatory_summary"]["channels_q10"] == [1, -1, 2, -2]
    assert payload["context_summary"] == {"energy_deficit": 512}
    assert payload["cost_summary"] == {
        "expression_cost_q10": 7,
        "regulatory_cost_q10": 3,
    }
    with pytest.raises(
        GenomeV2CausalTransactionProjectionError,
        match="record fields differ",
    ):
        GenomeV2CausalTransactionRecordV1.from_mapping(
            {
                key: value
                for key, value in payload.items()
                if key != "regulatory_summary"
            }
        )
    with pytest.raises(
        GenomeV2CausalTransactionProjectionError,
        match="non-finite",
    ):
        GenomeV2CausalTransactionRecordV1.create(
            record_phase="terminal",
            identity=_identity(),
            genotype_summary={"sha256": "e" * 64},
            product_summary={"energy_flux": 0},
            regulatory_summary={"channels_q10": [0]},
            context_summary={"energy_deficit": 0},
            phenotype_summary={"metabolic_efficiency": 0},
            cost_summary={"expression_cost_q10": float("nan")},
            applied_consequence={"energy_debit": 0.0},
            resulting_state={"energy": 0.0},
        )


def test_strict_join_accepts_exact_one_to_one_pair() -> None:
    event_time = _record("event_time")
    terminal = _record("terminal")

    joined = join_causal_transaction_records_v1([event_time], [terminal])

    assert len(joined) == 1
    assert joined[0].transaction_id == event_time.identity.transaction_id
    assert joined[0].event_time.record_phase == "event_time"
    assert joined[0].terminal.record_phase == "terminal"


@pytest.mark.parametrize(
    ("events", "terminals", "message"),
    (
        ([], [], "no evaluable records"),
        ([_record("event_time")], [], "join is incomplete"),
        (
            [_record("event_time"), _record("event_time")],
            [_record("terminal")],
            "duplicate event_time transaction_id",
        ),
    ),
)
def test_strict_join_rejects_missing_and_duplicate_records(
    events: list[GenomeV2CausalTransactionRecordV1],
    terminals: list[GenomeV2CausalTransactionRecordV1],
    message: str,
) -> None:
    with pytest.raises(
        GenomeV2CausalTransactionProjectionError, match=message
    ):
        join_causal_transaction_records_v1(events, terminals)


def test_strict_join_rejects_ambiguous_natural_identity() -> None:
    first = _record("event_time")
    second = _record(
        "event_time",
        identity=_identity(package="f" * 64),
    )

    with pytest.raises(
        GenomeV2CausalTransactionProjectionError,
        match="ambiguous event_time natural transaction identity",
    ):
        join_causal_transaction_records_v1([first, second], [])


def test_strict_join_revalidates_typed_records() -> None:
    event_time = _record("event_time")
    invalid_identity = replace(
        event_time.identity,
        transaction_id="0" * 64,
    )
    invalid_record = replace(event_time, identity=invalid_identity)

    with pytest.raises(
        GenomeV2CausalTransactionProjectionError,
        match="transaction_id contradicts",
    ):
        join_causal_transaction_records_v1([invalid_record], [])
