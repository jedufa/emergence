from __future__ import annotations

import copy
import struct
from pathlib import Path

import pytest

from engine.campaigns import p12a_gv2_persistence_reducer as reducer
from engine.campaigns.canonical import pretty_json


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
    else:  # pragma: no cover - fixture helper guard
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


def _digest_rows(rows: list[bytes]) -> list[int]:
    return [item for row in rows for item in row]


def _digest(value: int) -> bytes:
    return bytes([value]) * 32


def _projection(
    generation: int,
    *,
    alive: list[bool],
    agent_ids: list[int],
    tokens: list[int] | None = None,
    depths: list[int] | None = None,
    founder_ids: list[int] | None = None,
    digests: list[bytes] | None = None,
    allele: list[int] | None = None,
    package: list[int] | None = None,
) -> dict[str, object]:
    capacity = len(alive)
    tokens = tokens or list(range(100, 100 + capacity))
    depths = depths or [0] * capacity
    founder_ids = founder_ids or list(agent_ids)
    digests = digests or [_digest(index + 1) for index in range(capacity)]
    allele = allele or list(range(10, 10 + capacity))
    package = package or list(range(30, 30 + capacity))
    tensors = {
        "alive_mask": _tensor(alive, "bool", [capacity]),
        "slot_to_agent_id": _tensor(agent_ids, "int64", [capacity]),
        "slot_tokens": _tensor(tokens, "int64", [capacity]),
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
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(pretty_json(value), encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    g5: dict[str, object],
    *,
    g0: dict[str, object] | None = None,
    authority: dict[str, object] | None = None,
    target_milestone: int = 5,
) -> dict[str, object]:
    g0_path = _write(
        tmp_path,
        "g0.json",
        g0
        or _projection(
            0,
            alive=[True, False, False],
            agent_ids=[1, -1, -1],
            depths=[0, 0, 0],
            founder_ids=[1, -1, -1],
            digests=[_digest(7), _digest(8), _digest(9)],
            allele=[101, 102, 103],
            package=[201, 202, 203],
        ),
    )
    g5_path = _write(tmp_path, "g5.json", g5)
    return reducer.reduce_persistence(
        g0_path=g0_path,
        g5_path=g5_path,
        target_milestone=target_milestone,
        authority=authority or AUTHORITY,
        source_commit="f6f05db4b",
        campaign_id="p12a-test",
        cell_id="cell-test",
        seed=9202,
        evidence_root=tmp_path,
    )


def _row(
    result: dict[str, object], identity_type: str, milestone: int
) -> dict[str, object]:
    rows = [
        row
        for row in result["reducer_rows"]
        if row["identity_type"] == identity_type and row["milestone"] == milestone
    ]
    assert len(rows) == 1
    return rows[0]


def test_founder_remains_without_nonfounder_is_founder_only(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            5,
            alive=[True, False, False],
            agent_ids=[1, -1, -1],
            depths=[0, 0, 0],
            founder_ids=[1, -1, -1],
            digests=[_digest(7), _digest(8), _digest(9)],
            allele=[101, 102, 103],
            package=[201, 202, 203],
        ),
    )

    exact = _row(result, "exact_hereditary_payload_identity", 5)
    assert exact["classification"]["founder_only_persistence"] is True
    assert exact["classification"]["nonfounder_carrier_present"] is False


def test_founder_lost_exact_payload_nonfounder_carrier(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            5,
            alive=[False, True, False],
            agent_ids=[-1, 2, -1],
            depths=[0, 1, 0],
            founder_ids=[-1, 1, -1],
            digests=[_digest(8), _digest(7), _digest(9)],
            allele=[102, 101, 103],
            package=[202, 201, 203],
        ),
    )

    exact = _row(result, "exact_hereditary_payload_identity", 5)
    assert exact["nonfounder_carrier_count"] == 1
    assert exact["classification"]["state_present_in_direct_child"] is True


def test_allele_state_can_persist_when_exact_digest_changes(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            5,
            alive=[False, True, False],
            agent_ids=[-1, 2, -1],
            depths=[0, 2, 0],
            founder_ids=[-1, 1, -1],
            digests=[_digest(8), _digest(99), _digest(9)],
            allele=[102, 101, 103],
            package=[202, 201, 203],
        ),
    )

    exact = _row(result, "exact_hereditary_payload_identity", 5)
    allele = _row(result, "allele_genotype_state_identity", 5)
    assert exact["carrier_count"] == 0
    assert allele["nonfounder_carrier_count"] == 1
    assert allele["classification"]["state_present_after_2_generations"] is True


def test_package_can_persist_when_whole_genome_and_alleles_change(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            5,
            alive=[False, True, False],
            agent_ids=[-1, 2, -1],
            depths=[0, 5, 0],
            founder_ids=[-1, 1, -1],
            digests=[_digest(8), _digest(99), _digest(9)],
            allele=[102, 999, 103],
            package=[202, 201, 203],
        ),
    )

    allele = _row(result, "allele_genotype_state_identity", 5)
    package = _row(result, "preregistered_package_identity", 5)
    assert allele["carrier_count"] == 0
    assert package["nonfounder_carrier_count"] == 1
    assert package["classification"]["state_present_after_5_generations"] is True


def test_direct_child_only_does_not_require_founder_survival(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            5,
            alive=[False, True, False],
            agent_ids=[-1, 2, -1],
            depths=[0, 1, 0],
            founder_ids=[-1, 1, -1],
            digests=[_digest(8), _digest(7), _digest(9)],
            allele=[102, 101, 103],
            package=[202, 201, 203],
        ),
    )

    exact = _row(result, "exact_hereditary_payload_identity", 5)
    assert exact["founder_carrier_count"] == 0
    assert exact["classification"]["state_present_in_direct_child"] is True


def test_state_lost_fixed_and_coexisting_classifications(tmp_path: Path) -> None:
    lost = _run(
        tmp_path / "lost",
        _projection(
            5,
            alive=[False, True, False],
            agent_ids=[-1, 2, -1],
            depths=[0, 1, 0],
            founder_ids=[-1, 2, -1],
            digests=[_digest(8), _digest(8), _digest(9)],
            allele=[102, 102, 103],
            package=[202, 202, 203],
        ),
    )
    fixed = _run(
        tmp_path / "fixed",
        _projection(
            5,
            alive=[True, True, True],
            agent_ids=[1, 2, 3],
            depths=[0, 1, 2],
            founder_ids=[1, 1, 1],
            digests=[_digest(7), _digest(7), _digest(7)],
            allele=[101, 101, 101],
            package=[201, 201, 201],
        ),
    )
    coexisting = _run(
        tmp_path / "coexisting",
        _projection(
            5,
            alive=[True, True, True],
            agent_ids=[1, 2, 3],
            depths=[0, 1, 1],
            founder_ids=[1, 1, 3],
            digests=[_digest(7), _digest(7), _digest(8)],
            allele=[101, 101, 102],
            package=[201, 201, 202],
        ),
    )

    assert _row(lost, "exact_hereditary_payload_identity", 5)["classification"]["state_lost"] is True
    assert _row(fixed, "exact_hereditary_payload_identity", 5)["classification"]["state_fixed"] is True
    assert _row(coexisting, "exact_hereditary_payload_identity", 5)["classification"]["state_coexisting"] is True


def test_unrelated_partial_match_is_not_nonfounder_persistence(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            5,
            alive=[False, True, False],
            agent_ids=[-1, 2, -1],
            depths=[0, 1, 0],
            founder_ids=[-1, 99, -1],
            digests=[_digest(8), _digest(99), _digest(9)],
            allele=[102, 999, 103],
            package=[202, 201, 203],
        ),
    )

    package = _row(result, "preregistered_package_identity", 5)
    assert package["carrier_count"] == 1
    assert package["nonfounder_carrier_count"] == 0
    assert result["g5_carrier_table"][0]["carrier_classification"] == "unrelated_carrier"


def test_reducer_outputs_are_byte_identical_across_repeats(tmp_path: Path) -> None:
    g5 = _projection(
        5,
        alive=[True, True, False],
        agent_ids=[1, 2, -1],
        depths=[0, 1, 0],
        founder_ids=[1, 1, -1],
        digests=[_digest(7), _digest(7), _digest(9)],
        allele=[101, 101, 103],
        package=[201, 201, 203],
    )

    first = _run(tmp_path / "a", copy.deepcopy(g5))
    second = _run(tmp_path / "b", copy.deepcopy(g5))

    assert first["result_sha256"] == second["result_sha256"]


def test_target_milestone_can_be_g6(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            6,
            alive=[False, True, False],
            agent_ids=[-1, 2, -1],
            depths=[0, 3, 0],
            founder_ids=[-1, 1, -1],
            digests=[_digest(8), _digest(7), _digest(9)],
            allele=[102, 101, 103],
            package=[202, 201, 203],
        ),
        target_milestone=6,
    )

    exact = _row(result, "exact_hereditary_payload_identity", 6)
    assert result["target_milestone"] == 6
    assert result["summary"]["g6_alive_denominator"] == 1
    assert exact["nonfounder_carrier_count"] == 1
    assert exact["classification"]["state_present_after_2_generations"] is True


def test_duplicate_slot_tokens_do_not_merge_stable_identities(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _projection(
            5,
            alive=[True, True, False],
            agent_ids=[1, 2, -1],
            tokens=[1, 1, 1],
            depths=[0, 1, 0],
            founder_ids=[1, 1, -1],
            digests=[_digest(7), _digest(7), _digest(9)],
            allele=[101, 101, 103],
            package=[201, 201, 203],
        ),
    )

    exact = _row(result, "exact_hereditary_payload_identity", 5)
    carriers = [
        row
        for row in result["g5_carrier_table"]
        if row["identity_type"] == "exact_hereditary_payload_identity"
    ]
    assert exact["carrier_count"] == 2
    assert exact["founder_carrier_count"] == 1
    assert exact["nonfounder_carrier_count"] == 1
    assert {row["carrier_stable_id"] for row in carriers} == {1, 2}
    assert {row["carrier_slot_token"] for row in carriers} == {1}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda g0, g5, auth: g5["projection"]["tensors"]["canonical_hereditary_digest"].__setitem__("shape", [3, 31]),
            "canonical_hereditary_digest",
        ),
        (
            lambda g0, g5, auth: auth.__setitem__("execution_route_sha256", "1" * 64),
            "authority binding mismatch",
        ),
        (
            lambda g0, g5, auth: auth.__setitem__("handoff_decision", "HANDOFF-A"),
            "stale handoff",
        ),
        (
            lambda g0, g5, auth: g5.__setitem__("classification", "caller"),
            "reporting-derived",
        ),
        (
            lambda g0, g5, auth: g5["projection"]["tensors"]["slot_to_agent_id"].__setitem__(
                "data_hex", _tensor([1, 1, -1], "int64", [3])["data_hex"]
            ),
            "duplicate stable agent ID",
        ),
        (
            lambda g0, g5, auth: g5.clear() or g5.update({"reducer_rows": []}),
            "reporting-derived",
        ),
        (
            lambda g0, g5, auth: g5["projection"]["metadata"].__setitem__("generation", 6),
            "checkpoint milestone mismatch",
        ),
    ],
)
def test_fail_closed_integrity_cases(tmp_path: Path, mutate, message: str) -> None:
    g0 = _projection(
        0,
        alive=[True, False, False],
        agent_ids=[1, -1, -1],
        depths=[0, 0, 0],
        founder_ids=[1, -1, -1],
        digests=[_digest(7), _digest(8), _digest(9)],
        allele=[101, 102, 103],
        package=[201, 202, 203],
    )
    g5 = _projection(
        5,
        alive=[True, True, False],
        agent_ids=[1, 2, -1],
        depths=[0, 1, 0],
        founder_ids=[1, 1, -1],
        digests=[_digest(7), _digest(7), _digest(9)],
        allele=[101, 101, 103],
        package=[201, 201, 203],
    )
    auth = dict(AUTHORITY)
    mutate(g0, g5, auth)

    with pytest.raises(reducer.P12APersistenceReducerError, match=message):
        _run(tmp_path, g5, g0=g0, authority=auth)
