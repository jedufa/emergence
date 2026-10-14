from __future__ import annotations

import copy
import gzip
import struct
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.campaigns import p12b_compact_descendant_oracle as oracle
from engine.campaigns import p12b_ob_state_persistence_reducer as full_reducer
from engine.campaigns.canonical import pretty_json
from engine.evolution.genome_upgrade_v2_causal_transaction_projection_v1 import (
    GenomeV2CausalTransactionIdentityV1,
    GenomeV2CausalTransactionRecordV1,
)


AUTHORITY = dict(full_reducer.EXPECTED_AUTHORITY)


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


def _event_receipt(
    *,
    child_id: int,
    parent_id: int,
    mate_parent_id: int,
    generation: int,
    event_sequence: int,
    parent_slot: int = 0,
    mate_parent_slot: int = 3,
    lineage_founder_id: int = 1,
    lineage_depth: int = 1,
) -> dict[str, object]:
    identity = GenomeV2CausalTransactionIdentityV1.create(
        event_kind="ordinary_birth",
        event_sequence=event_sequence,
        child_id=child_id,
        cell_id="cell-test",
        generation=generation,
        tick=1,
        destination_slot=1,
        parent_slot=parent_slot,
        mate_parent_slot=mate_parent_slot,
        parent_id=parent_id,
        mate_parent_id=mate_parent_id,
        lineage_founder_id=lineage_founder_id,
        lineage_depth=lineage_depth,
        capability_contract_sha256="f8f3b1fb40770c77a66bf2bbd34e1bb2a6050fa8770a7c1d17743100ae0d09ff",
        package_id_sha256=AUTHORITY["package_id_sha256"],
        artifact_sha256=AUTHORITY["cuda_artifact_sha256"],
        execution_route_sha256=AUTHORITY["execution_route_sha256"],
    )
    common = {
        "genotype_summary": {"alleles_q10": [[101]], "modifiers_q10": [[0]]},
        "product_summary": {
            "products_q10": [201],
            "active_product_count": 1,
            "saturated_product_count": 0,
        },
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
    return _receipt(event, schema="genome_v2_causal_event_receipt_v2")


def _causal_projection(
    events: list[dict[str, object]],
    *,
    joined: bool = False,
    terminal_agent_id: int = 2,
) -> dict[str, object]:
    event_receipts = copy.deepcopy(events)
    terminal_receipts: list[dict[str, object]] = []
    join_receipts: list[dict[str, object]] = []
    if joined:
        for event_receipt in event_receipts:
            event_record = event_receipt["transaction_record"]
            identity = event_record["identity"]
            terminal = GenomeV2CausalTransactionRecordV1.create(
                record_phase="terminal",
                identity=GenomeV2CausalTransactionIdentityV1.from_mapping(identity),
                applied_consequence={"source": "fixture"},
                resulting_state={
                    "survived_to_terminal": True,
                    "retention_relation": "direct_child_survival",
                    "terminal_slot": 1,
                    "agent_id": terminal_agent_id,
                    "event_child_id": identity["child_id"],
                    "located_by": "fixture",
                },
                genotype_summary=event_record["genotype_summary"],
                product_summary=event_record["product_summary"],
                regulatory_summary=event_record["regulatory_summary"],
                context_summary=event_record["context_summary"],
                phenotype_summary=event_record["phenotype_summary"],
                cost_summary=event_record["cost_summary"],
            )
            terminal_receipt = _receipt(
                terminal, schema="genome_v2_causal_terminal_receipt_v2"
            )
            terminal_receipts.append(terminal_receipt)
            join_payload = {
                "schema_version": "genome_v2_causal_join_receipt_v2",
                "transaction_id": identity["transaction_id"],
                "event_receipt_sha256": event_receipt["receipt_sha256"],
                "terminal_receipt_sha256": terminal_receipt["receipt_sha256"],
                "retention_relation": "direct_child_survival",
                "terminal_agent_id": terminal_agent_id,
            }
            join_receipts.append(
                {**join_payload, "receipt_sha256": content_sha256(join_payload)}
            )
    return {
        "schema_version": "genome_v2_causal_settlement_projection_v2",
        "capture_status": "capture_enabled_joined",
        "captured_counts": {
            "birth": len(event_receipts),
            "turnover": 0,
            "total": len(event_receipts),
        },
        "dropped_counts": {"birth": 0, "turnover": 0, "total": 0},
        "event_receipts": event_receipts,
        "terminal_receipts": terminal_receipts,
        "join_receipts": join_receipts,
        "evaluable": True,
        "record_counts": {
            "event": len(event_receipts),
            "terminal": len(terminal_receipts),
            "joined": len(join_receipts),
        },
    }


def _non_evaluable_causal() -> dict[str, object]:
    return {
        "schema_version": "genome_v2_causal_settlement_projection_v2",
        "capture_status": "capture_enabled_not_evaluable",
        "captured_counts": {"birth": 1024, "turnover": 0, "total": 1024},
        "dropped_counts": {"birth": 1, "turnover": 0, "total": 1},
        "event_receipts": [],
        "terminal_receipts": [],
        "join_receipts": [],
        "evaluable": False,
        "record_counts": {"event": 0, "terminal": 0, "joined": 0},
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
    capacity = 4
    alive = [True, False, False, False]
    agent_ids = [1, -1, -1, 9]
    depths = [0, 0, 0, 0]
    founder_ids = [1, -1, -1, 9]
    digests = [_digest(7), _digest(8), _digest(9), _digest(4)]
    allele = [101, 102, 103, 104]
    package = [201, 202, 203, 204]
    if generation:
        alive = [False, False, False, True]
        agent_ids = [-1, -1, -1, carrier_agent]
        depths = [0, 0, 0, carrier_depth]
        founder_ids = [-1, -1, -1, carrier_founder]
        digests[carrier_slot] = _digest(7)
        allele[carrier_slot] = 101
        package[carrier_slot] = 201
        if carrier_slot != 3:
            alive[3] = False
            agent_ids[3] = -1
            depths[3] = 0
            founder_ids[3] = -1
            alive[carrier_slot] = True
            agent_ids[carrier_slot] = carrier_agent
            depths[carrier_slot] = carrier_depth
            founder_ids[carrier_slot] = carrier_founder
    tensors = {
        "alive_mask": _tensor(alive, "bool", [capacity]),
        "slot_to_agent_id": _tensor(agent_ids, "int64", [capacity]),
        "slot_tokens": _tensor([10, 11, 12, 13], "int64", [capacity]),
        "lineage_depth": _tensor(depths, "int32", [capacity]),
        "lineage_founder_id": _tensor(founder_ids, "int32", [capacity]),
        "canonical_hereditary_digest": _tensor(
            _digest_rows(digests), "uint8", [capacity, 32]
        ),
    }
    return {
        "projection": {
            "schema_version": "cxxbackbone10_checkpoint_projection_v2",
            "metadata": {"generation": generation, "alive_count": sum(alive)},
            "tensors": tensors,
            "genome_upgrade_v2_checkpoint_projection": {
                "schema_version": "genome_upgrade_v2_checkpoint_projection_v1",
                "causal_transaction_projection": causal or _causal_projection([]),
                "tensors": {
                    "genome_v2_alleles_q10": [allele],
                    "genome_v2_modifiers_q10": [allele],
                    "genome_v2_products_q10": [allele],
                    "genome_v2_phenotypes_q10": [package],
                    "genome_v2_active_product_count": [package],
                    "genome_v2_saturated_product_count": [package],
                },
            },
        }
    }


def _write(tmp_path: Path, name: str, value: dict[str, object]) -> Path:
    path = tmp_path / name
    payload = pretty_json(value).encode("utf-8")
    if name.endswith(".gz"):
        path.write_bytes(gzip.compress(payload, mtime=0))
    else:
        path.write_bytes(payload)
    return path


def _run(
    tmp_path: Path,
    *,
    target: dict[str, object],
    segments: list[dict[str, object]],
    milestone: int,
) -> dict[str, object]:
    return oracle.reduce_compact_descendant_oracle(
        g0_path=_write(tmp_path, "g0.json", _projection(0)),
        target_path=_write(tmp_path, "target.json.gz", target),
        segment_paths=tuple(
            _write(tmp_path, f"segment_{index}.json", segment)
            for index, segment in enumerate(segments)
        ),
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


def test_compact_oracle_matches_full_ledger_direct_birth_counts(tmp_path: Path) -> None:
    event = _event_receipt(
        child_id=2,
        parent_id=1,
        mate_parent_id=9,
        generation=4,
        event_sequence=1,
    )
    segment = _projection(5, causal=_causal_projection([event], joined=True))
    compact = _run(tmp_path, target=segment, segments=[segment], milestone=5)
    full = full_reducer.reduce_ob_state_persistence(
        g0_path=_write(tmp_path, "full_g0.json", _projection(0)),
        target_path=_write(tmp_path, "full_target.json", segment),
        target_milestone=5,
        authority=AUTHORITY,
        source_commit="5286fad91",
        campaign_id="p12b-test",
        cell_id="cell-test",
        seed=101,
        evidence_root=tmp_path,
    )

    assert _exact_row(compact)["positive_ordinary_birth_carrier_count"] == _exact_row(full)[
        "positive_ordinary_birth_carrier_count"
    ]
    assert compact["summary"]["compact_event_row_count"] == 1
    source = compact["source_event_rows"][0]
    assert source["parent_slot"] == 0
    assert source["mate_slot"] == 3


def test_compact_oracle_accepts_mate_side_founder_ancestry(tmp_path: Path) -> None:
    event = _event_receipt(
        child_id=2,
        parent_id=9,
        mate_parent_id=1,
        generation=4,
        event_sequence=1,
    )
    segment = _projection(5, causal=_causal_projection([event]))

    row = _exact_row(_run(tmp_path, target=segment, segments=[segment], milestone=5))

    assert row["positive_ordinary_birth_carrier_count"] == 1
    carrier = [
        item
        for item in _run(tmp_path, target=segment, segments=[segment], milestone=5)[
            "target_carrier_table"
        ]
        if item["identity_type"] == "exact_hereditary_payload_identity"
    ][0]
    assert carrier["ordinary_birth_chain"][0]["mate_stable_id"] == 1
    assert carrier["ordinary_birth_chain"][0]["mate_slot"] == 3


def test_compact_oracle_reduces_non_evaluable_terminal_from_complete_segments(
    tmp_path: Path,
) -> None:
    first = _event_receipt(
        child_id=2,
        parent_id=1,
        mate_parent_id=9,
        generation=4,
        event_sequence=1,
        lineage_depth=1,
    )
    second = _event_receipt(
        child_id=3,
        parent_id=2,
        mate_parent_id=9,
        generation=7,
        event_sequence=2,
        lineage_depth=2,
    )
    segment_g5 = _projection(5, causal=_causal_projection([first]))
    segment_g10 = _projection(
        10,
        carrier_agent=3,
        carrier_depth=2,
        carrier_slot=3,
        causal=_causal_projection([second]),
    )
    terminal = _projection(
        10,
        carrier_agent=3,
        carrier_depth=2,
        carrier_slot=3,
        causal=_non_evaluable_causal(),
    )

    result = _run(
        tmp_path,
        target=terminal,
        segments=[segment_g5, segment_g10],
        milestone=10,
    )

    row = _exact_row(result)
    assert row["positive_ordinary_birth_carrier_count"] == 1
    assert row["mixed_descendant_path_carrier_count"] == 1
    assert result["summary"]["source_event_row_count"] == 2
    assert result["summary"]["compact_event_row_count"] == 2


def test_unresolved_non_positive_carrier_is_reported_without_integrity_failure(
    tmp_path: Path,
) -> None:
    event = _event_receipt(
        child_id=2,
        parent_id=8,
        mate_parent_id=9,
        generation=4,
        event_sequence=1,
        lineage_founder_id=1,
        lineage_depth=1,
    )
    segment = _projection(5, causal=_causal_projection([event]))

    result = _run(tmp_path, target=segment, segments=[segment], milestone=5)

    row = _exact_row(result)
    assert row["positive_ordinary_birth_carrier_count"] == 0
    assert row["unresolved_carrier_count"] == 1
    assert result["summary"]["lineage_candidate_unresolved_chain_count"] == 3
    assert result["summary"]["evaluable"] is True


def test_segment_with_dropped_rows_fails_closed(tmp_path: Path) -> None:
    terminal = _projection(5)

    with pytest.raises(
        oracle.P12BCompactDescendantOracleError,
        match="segment causal projection is not capture_enabled_joined",
    ):
        _run(
            tmp_path,
            target=terminal,
            segments=[_projection(5, causal=_non_evaluable_causal())],
            milestone=5,
        )
