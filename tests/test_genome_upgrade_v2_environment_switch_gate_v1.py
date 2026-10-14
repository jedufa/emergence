from __future__ import annotations

from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (
    BACKENDS,
    CONDITIONS,
    DECISION_A,
    DECISION_D,
    EVIDENCE_SCHEMA_VERSION,
    PREREGISTRATION_SHA256,
    REPEAT_INDICES,
    SNAPSHOT_SCHEMA_VERSION,
    evaluate_environment_switch_v1,
)
from tools.scripts.genome_upgrade_v2_environment_switch_gate_v1 import (
    materialize,
)


def _authority() -> dict[str, str]:
    return {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "source_commit": "1" * 40,
        "package_id_sha256": "2" * 64,
        "package_receipt_sha256": "3" * 64,
        "artifact_sha256": "4" * 64,
        "execution_route_sha256": "5" * 64,
    }


def _snapshot(environment: str) -> dict[str, object]:
    offset = 0 if environment == "A" else 128
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "context_q10": [offset, 1024 - offset],
        "products_q10": [100 + offset, 200 - offset],
        "regulatory_q10": [20 + offset],
        "phenotypes_q10": [300 + offset, 400 - offset],
        "runtime_consequences_q10": [500 + offset],
        "expression_cost_q10": 10 + offset,
        "regulatory_cost_q10": 4 + offset,
        "runtime_cost_q10": 6 + offset,
    }


def _matrix() -> list[dict[str, object]]:
    trajectory = {
        "stable_a": ("A", "A"),
        "stable_b": ("B", "B"),
        "a_to_b": ("A", "B"),
        "b_to_a": ("B", "A"),
    }
    rows = []
    for condition in CONDITIONS:
        for backend in BACKENDS:
            for repeat in REPEAT_INDICES:
                initial, terminal = trajectory[condition]
                value = {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "preregistration_sha256": PREREGISTRATION_SHA256,
                    "authority": _authority(),
                    "condition": condition,
                    "backend": backend,
                    "repeat_index": repeat,
                    "genotype_sha256": "a" * 64,
                    "initial": _snapshot(initial),
                    "terminal": _snapshot(terminal),
                    "no_fallback": True,
                    "no_jit": True,
                }
                rows.append(
                    {**value, "evidence_sha256": content_sha256(value)}
                )
    return rows


def _rehash(row: dict[str, object]) -> None:
    unsigned = dict(row)
    unsigned.pop("evidence_sha256")
    row["evidence_sha256"] = content_sha256(unsigned)


def test_environment_switch_exact_matrix_and_fail_closed_parity() -> None:
    rows = _matrix()
    result = evaluate_environment_switch_v1(rows, authority=_authority())
    assert result["decision"] == DECISION_A
    assert result["matrix"]["expected_rows"] == 24

    rows[0]["terminal"] = _snapshot("B")
    _rehash(rows[0])
    assert (
        evaluate_environment_switch_v1(rows, authority=_authority())["decision"]
        == DECISION_D
    )


def test_environment_switch_rejects_genotype_change_and_writes_outputs(
    tmp_path: Path,
) -> None:
    rows = _matrix()
    rows[-1]["genotype_sha256"] = "b" * 64
    _rehash(rows[-1])
    assert (
        evaluate_environment_switch_v1(rows, authority=_authority())["decision"]
        == DECISION_D
    )

    output = tmp_path / "result"
    manifest = materialize(
        authority=_authority(),
        evidence=_matrix(),
        output_directory=output,
    )
    assert manifest["decision"] == DECISION_A
    assert len(list(output.iterdir())) == 3
    with pytest.raises(FileExistsError):
        materialize(
            authority=_authority(),
            evidence=_matrix(),
            output_directory=output,
        )
