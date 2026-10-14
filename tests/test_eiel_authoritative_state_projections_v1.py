from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from engine.authoritative_state.projections import (
    ProjectionSchemaError,
    validate_projection_contract,
    validate_projection_envelope,
)


SPEC = Path(__file__).parents[1] / "_spec" / "authoritative_state" / "v1"


def _contract(name: str) -> dict:
    return json.loads((SPEC / name).read_text(encoding="utf-8"))


def _bindings(contract: dict) -> dict[str, str]:
    return {key: f"{index + 1:064x}" for index, key in enumerate(contract["binding_keys"])}


def _envelope(contract: dict, bindings: dict[str, str]) -> dict:
    fields = deepcopy(contract["fields"])
    provenance = {}
    for row in fields:
        provenance[row["field_id"]] = {
            "transform": row["transform"],
            "source_field_ids": list(row["source_field_ids"]),
            "source_record_hashes": [
                f"{index + 10:064x}" for index, _ in enumerate(row["source_field_ids"])
            ],
        }
        if row["transform"] == "constant":
            provenance[row["field_id"]]["constant_value"] = row["constant_value"]
    return {
        "projection_schema_version": contract["schema_version"],
        "projection_schema_hash": contract["contract_hash"],
        "bindings": dict(bindings),
        "output_manifest": fields,
        "values": {row["field_id"]: index for index, row in enumerate(fields)},
        "provenance": provenance,
    }


@pytest.mark.parametrize(
    "name",
    ["provider_projection_schema.json", "analysis_projection_schema.json"],
)
def test_projection_contract_and_envelope_admit_exact_provenance(name: str) -> None:
    contract = _contract(name)
    bindings = _bindings(contract)
    assert validate_projection_contract(contract) == contract["contract_hash"]
    assert validate_projection_envelope(
        _envelope(contract, bindings),
        contract=contract,
        expected_bindings=bindings,
    ) == tuple(row["field_id"] for row in contract["fields"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda envelope: envelope.__setitem__(
                "projection_schema_version", "stale_v1"
            ),
            "version",
        ),
        (
            lambda envelope: envelope.__setitem__(
                "projection_schema_hash", "f" * 64
            ),
            "schema hash",
        ),
        (
            lambda envelope: envelope["bindings"].__setitem__(
                "binary_schema_hash", "f" * 64
            ),
            "binary_schema_hash",
        ),
        (
            lambda envelope: envelope["output_manifest"][0].__setitem__(
                "dtype", "int32"
            ),
            "dtype",
        ),
        (
            lambda envelope: envelope["output_manifest"][0].__setitem__(
                "shape", ["other"]
            ),
            "shape",
        ),
        (
            lambda envelope: envelope["provenance"][
                next(iter(envelope["provenance"]))
            ].__setitem__("transform", "derived"),
            "transform",
        ),
        (
            lambda envelope: envelope["provenance"][
                next(iter(envelope["provenance"]))
            ].__setitem__("source_record_hashes", []),
            "source hash count",
        ),
    ],
)
def test_provider_projection_fails_closed_for_manifest_and_binding_drift(
    mutation, message: str
) -> None:
    contract = _contract("provider_projection_schema.json")
    bindings = _bindings(contract)
    envelope = _envelope(contract, bindings)
    mutation(envelope)
    with pytest.raises(ProjectionSchemaError, match=message):
        validate_projection_envelope(
            envelope, contract=contract, expected_bindings=bindings
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda envelope: envelope.__setitem__(
                "projection_schema_version", "stale_v1"
            ),
            "version",
        ),
        (
            lambda envelope: envelope.__setitem__(
                "projection_schema_hash", "f" * 64
            ),
            "schema hash",
        ),
        (
            lambda envelope: envelope["output_manifest"][0].__setitem__(
                "dtype", "int32"
            ),
            "dtype",
        ),
        (
            lambda envelope: envelope["output_manifest"][0].__setitem__(
                "shape", ["other"]
            ),
            "shape",
        ),
    ],
)
def test_analysis_projection_fails_closed_for_schema_and_manifest_drift(
    mutation, message: str
) -> None:
    contract = _contract("analysis_projection_schema.json")
    bindings = _bindings(contract)
    envelope = _envelope(contract, bindings)
    mutation(envelope)

    with pytest.raises(ProjectionSchemaError, match=message):
        validate_projection_envelope(
            envelope, contract=contract, expected_bindings=bindings
        )


def test_provider_contract_cannot_claim_derived_transform() -> None:
    contract = _contract("provider_projection_schema.json")
    contract["fields"][0]["transform"] = "derived"
    contract.pop("contract_hash")
    from engine.authoritative_state.admission import canonical_sha256

    contract["contract_hash"] = canonical_sha256(contract)
    with pytest.raises(ProjectionSchemaError, match="unsupported"):
        validate_projection_contract(contract)
