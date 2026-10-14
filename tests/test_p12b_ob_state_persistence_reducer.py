from __future__ import annotations

import copy
import struct
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.campaigns import p12b_ob_state_persistence_reducer as reducer
from engine.campaigns.canonical import pretty_json
from engine.evolution.genome_upgrade_v2_causal_transaction_projection_v1 import (
    GenomeV2CausalTransactionIdentityV1,
    GenomeV2CausalTransactionRecordV1,
)


AUTHORITY = dict(reducer.EXPECTED_AUTHORITY)


def _tensor(values: list[object], dtype: str, shape: list[int]) -> dict[str, object]:
    if dtype == "bool":
        raw = bytes(1 if item else 0 for item in values)
    elif dtype == "uint8":
        raw = bytes(int(item) for item in values)
    elif dtype == "int32":
        raw = struct.pack("<" + "i" * len(values), *[int(item) for item in values])
    elif dtype == "int64":
        raw = struct.pack("<" + "q" * len(values), *[int(item) for item in values])
    else:  # pragma: no cover
        raise AssertionError(dtype)
    return {
        "data_hex": raw.hex(),
        "device": "cpu",
        "dtype": dtype,
        "kind": "tensor",
        "order": "C",
        "sha256": "0" * 64,
        "shape": shape,
    }


def _digest(value: int) -> bytes:
    return bytes([value]) * 32


def _digest_rows(rows: list[bytes]) -> list[int]:
    return [item for row in rows for item in row]


def _receipt(record: GenomeV2CausalTransactionRecordV1, *, schema: str) -> dict[str, object]:
    payload = {"schema_version": schema, "transaction_record": record.as_dict()}
    return {**payload, "receipt_sha256": content_sha256(payload)}


def _causal_projection(
    *,
    event_kind: str = "ordinary_birth",
    retention_relation: str = "direct_child_survival",
    event_child_id: int = 2,
    terminal_agent_id: int = 2,
) -> dict[str, object]:
    identity = GenomeV2CausalTransactionIdentityV1.create(
        event_kind=event_kind,
        event_sequence=1,
        child_id=event_child_id,
        cell_id="cell-test",
        generation=5,
        tick=1,
        destination_slot=1,
        parent_slot=0,
        mate_parent_slot=2,
        parent_id=1,
        mate_parent_id=9,
        lineage_founder_id=1,
        lineage_depth=1,
        capability_contract_sha256="f8f3b1fb40770c77a66bf2bbd34e1bb2a6050fa8770a7c1d17743100ae0d09ff",
        package_id_sha256=AUTHORITY["package_id_sha256"],
        artifact_sha256=AUTHORITY["cuda_artifact_sha256"],
        execution_route_sha256=AUTHORITY["execution_route_sha256"],
    )
    common = {
        "genotype_summary": {"alleles_q10": [[101]], "modifiers_q10": [[0]]},
        "product_summary": {"products_q10": [201], "active_product_count": 1, "saturated_product_count": 0},
        "regulatory_summary": {"available": False},
        "context_summary": {"available": True, "context_q10": [0]},
        "phenotype_summary": {"phenotypes_q10": [201]},
        "cost_summary": {"expression_cost_q10": 1},
    }
    event = GenomeV2CausalTransactionRecordV1.create(
        record_phase="event_time",
        identity=identity,
        applied_consequence={"source": "fixture"},
        resulting_state={"available": False},
        **common,
    )
    terminal = GenomeV2CausalTransactionRecordV1.create(
        record_phase="terminal",
        identity=identity,
        applied_consequence={"available": False},
        resulting_state={
            "survived_to_terminal": retention_relation != "terminal_loss",
            "retention_relation": retention_relation,
            "terminal_slot": None if retention_relation == "terminal_loss" else 1,
            "agent_id": terminal_agent_id,
            "event_child_id": event_child_id,
            "located_by": "fixture",
        },
        **common,
    )
    event_receipt = _receipt(event, schema="genome_v2_causal_event_receipt_v2")
    terminal_receipt = _receipt(terminal, schema="genome_v2_causal_terminal_receipt_v2")
    join_payload = {
        "schema_version": "genome_v2_causal_join_receipt_v2",
        "transaction_id": identity.transaction_id,
        "event_receipt_sha256": event_receipt["receipt_sha256"],
        "terminal_receipt_sha256": terminal_receipt["receipt_sha256"],
        "retention_relation": retention_relation,
        "terminal_agent_id": terminal_agent_id,
    }
    return {
        "schema_version": "genome_v2_causal_settlement_projection_v2",
        "capture_status": "capture_enabled_joined",
        "captured_counts": {"birth": 1 if event_kind == "ordinary_birth" else 0, "turnover": 0 if event_kind == "ordinary_birth" else 1, "total": 1},
        "dropped_counts": {"birth": 0, "turnover": 0, "total": 0},
        "event_receipts": [event_receipt],
        "terminal_receipts": [terminal_receipt],
        "join_receipts": [{**join_payload, "receipt_sha256": content_sha256(join_payload)}],
        "evaluable": True,
        "record_counts": {"event": 1, "terminal": 1, "joined": 1},
    }


def _projection(
    generation: int,
    *,
    carrier_agent: int = 2,
    carrier_depth: int = 1,
    carrier_founder: int = 1,
    carrier_slot: int = 1,
    causal: dict[str, object] | None = None,
) -> dict[str, object]:
    capacity = 3
    alive = [True, False, False]
    agent_ids = [1, -1, -1]
    depths = [0, 0, 0]
    founder_ids = [1, -1, -1]
    digests = [_digest(7), _digest(8), _digest(9)]
    allele = [101, 102, 103]
    package = [201, 202, 203]
    if generation:
        alive[0] = False
        alive[carrier_slot] = True
        agent_ids[0] = -1
        agent_ids[carrier_slot] = carrier_agent
        depths[carrier_slot] = carrier_depth
        founder_ids[carrier_slot] = carrier_founder
        digests[carrier_slot] = _digest(7)
        allele[carrier_slot] = 101
        package[carrier_slot] = 201
    tensors = {
        "alive_mask": _tensor(alive, "bool", [capacity]),
        "slot_to_agent_id": _tensor(agent_ids, "int64", [capacity]),
        "slot_tokens": _tensor([10, 11, 12], "int64", [capacity]),
        "lineage_depth": _tensor(depths, "int32", [capacity]),
        "lineage_founder_id": _tensor(founder_ids, "int32", [capacity]),
        "canonical_hereditary_digest": _tensor(
            _digest_rows(digests), "uint8", [capacity, 32]
        ),
    }
    gv2 = {
        "schema_version": "genome_upgrade_v2_checkpoint_projection_v1",
        "causal_transaction_projection": causal or _causal_projection(),
        "tensors": {
            "genome_v2_alleles_q10": [allele],
            "genome_v2_modifiers_q10": [allele],
            "genome_v2_products_q10": [allele],
            "genome_v2_phenotypes_q10": [package],
            "genome_v2_active_product_count": [package],
            "genome_v2_saturated_product_count": [package],
        },
    }
    return {
        "projection": {
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
            "metadata": {"generation": generation, "alive_count": sum(alive)},
            "tensors": tensors,
            "genome_upgrade_v2_checkpoint_projection": gv2,
        }
    }


def _write(tmp_path: Path, name: str, value: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(pretty_json(value), encoding="utf-8")
    return path


def _run(tmp_path: Path, target: dict[str, object], *, milestone: int = 5) -> dict[str, object]:
    return reducer.reduce_ob_state_persistence(
        g0_path=_write(tmp_path, "g0.json", _projection(0)),
        target_path=_write(tmp_path, "target.json", target),
        target_milestone=milestone,
        authority=AUTHORITY,
        source_commit="5286fad91",
        campaign_id="p12b-test",
        cell_id="cell-test",
        seed=101,
        evidence_root=tmp_path,
    )


def _exact_row(result: dict[str, object]) -> dict[str, object]:
    rows = [
        row
        for row in result["reducer_rows"]
        if row["identity_type"] == "exact_hereditary_payload_identity"
    ]
    assert len(rows) == 1
    return rows[0]


def test_direct_ordinary_birth_carrier_counts_as_positive(tmp_path: Path) -> None:
    result = _run(tmp_path, _projection(5))

    row = _exact_row(result)
    assert row["positive_ordinary_birth_carrier_count"] == 1
    assert row["ordinary_birth_direct_child_carrier_count"] == 1
    assert row["classification"]["positive_ordinary_birth_persistence"] is True


def test_mixed_descendant_path_counts_as_positive_ordinary_birth_ancestry(tmp_path: Path) -> None:
    target = _projection(
        5,
        carrier_agent=3,
        carrier_depth=2,
        carrier_slot=2,
        causal=_causal_projection(
            retention_relation="event_ledger_descendant_retention",
            event_child_id=2,
            terminal_agent_id=3,
        ),
    )

    row = _exact_row(_run(tmp_path, target))

    assert row["positive_ordinary_birth_carrier_count"] == 1
    assert row["mixed_descendant_path_carrier_count"] == 1
    assert row["classification"]["ordinary_birth_descendant_persistence"] is True


def test_turnover_only_carrier_does_not_count_as_positive(tmp_path: Path) -> None:
    target = _projection(5, causal=_causal_projection(event_kind="turnover_refill"))

    row = _exact_row(_run(tmp_path, target))

    assert row["carrier_count"] == 1
    assert row["positive_ordinary_birth_carrier_count"] == 0
    assert row["turnover_only_carrier_count"] == 1


def test_join_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    target = _projection(5)
    causal = target["projection"]["genome_upgrade_v2_checkpoint_projection"][
        "causal_transaction_projection"
    ]
    tampered = copy.deepcopy(causal["join_receipts"][0])
    tampered["event_receipt_sha256"] = "1" * 64
    causal["join_receipts"][0] = tampered

    with pytest.raises(
        reducer.P12BOrdinaryBirthStatePersistenceReducerError,
        match="event SHA-256",
    ):
        _run(tmp_path, target)


def test_authority_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        reducer.P12BOrdinaryBirthStatePersistenceReducerError,
        match="authority binding mismatch",
    ):
        reducer.reduce_ob_state_persistence(
            g0_path=_write(tmp_path, "g0.json", _projection(0)),
            target_path=_write(tmp_path, "target.json", _projection(5)),
            target_milestone=5,
            authority={**AUTHORITY, "package_id_sha256": "0" * 64},
            source_commit="5286fad91",
            campaign_id="p12b-test",
            cell_id="cell-test",
            seed=101,
            evidence_root=tmp_path,
        )
