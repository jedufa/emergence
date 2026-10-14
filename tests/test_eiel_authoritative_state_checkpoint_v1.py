from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.authoritative_state.schema import load_schema_v1
from engine.authoritative_state.checkpoint import (
    CheckpointSchemaError,
    checkpoint_manifest_hash,
    validate_checkpoint_contract,
    validate_checkpoint_envelope,
)
from simulation.checkpoint_bundle_v2 import (
    AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_FIELD,
    AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD,
    CheckpointStrictError,
    _authoritative_state_checkpoint_identity_v1,
    _build_metadata,
    _validate_authoritative_state_checkpoint_metadata,
    checkpoint_bundle_v2_apply,
)


SPEC = Path(__file__).parents[1] / "_spec" / "authoritative_state" / "v1"


def _contract() -> dict:
    return json.loads((SPEC / "checkpoint_schema.json").read_text(encoding="utf-8"))


def _bindings(contract: dict) -> dict[str, str]:
    return {key: f"{index + 1:064x}" for index, key in enumerate(contract["binding_keys"])}


def _envelope(contract: dict, bindings: dict[str, str]) -> dict:
    fields = deepcopy(contract["fields"])
    return {
        "checkpoint_schema_version": contract["schema_version"],
        "checkpoint_schema_hash": contract["contract_hash"],
        "bindings": dict(bindings),
        "field_manifest": fields,
        "field_manifest_hash": checkpoint_manifest_hash(fields),
        "values": {row["field_id"]: index for index, row in enumerate(fields)},
    }


def test_checkpoint_contract_and_envelope_admit_exact_manifest() -> None:
    contract = _contract()
    bindings = _bindings(contract)
    assert validate_checkpoint_contract(contract) == contract["contract_hash"]
    assert validate_checkpoint_envelope(
        _envelope(contract, bindings),
        contract=contract,
        expected_bindings=bindings,
    ) == tuple(row["field_id"] for row in contract["fields"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda envelope: envelope.__setitem__(
                "checkpoint_schema_version", "stale_v1"
            ),
            "version",
        ),
        (
            lambda envelope: envelope.__setitem__(
                "checkpoint_schema_hash", "f" * 64
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
            lambda envelope: envelope["field_manifest"][0].__setitem__(
                "dtype", "int32"
            ),
            "dtype",
        ),
        (
            lambda envelope: envelope["field_manifest"][2].__setitem__(
                "shape", ["cells"]
            ),
            "shape",
        ),
        (
            lambda envelope: envelope["field_manifest"].pop(),
            "missing",
        ),
    ],
)
def test_checkpoint_fails_closed_for_contract_drift(mutation, message: str) -> None:
    contract = _contract()
    bindings = _bindings(contract)
    envelope = _envelope(contract, bindings)
    mutation(envelope)
    with pytest.raises(CheckpointSchemaError, match=message):
        validate_checkpoint_envelope(
            envelope, contract=contract, expected_bindings=bindings
        )


def test_checkpoint_rejects_manifest_hash_and_value_order_mismatch() -> None:
    contract = _contract()
    bindings = _bindings(contract)
    envelope = _envelope(contract, bindings)
    envelope["field_manifest_hash"] = "0" * 64
    with pytest.raises(CheckpointSchemaError, match="manifest hash"):
        validate_checkpoint_envelope(
            envelope, contract=contract, expected_bindings=bindings
        )
    envelope = _envelope(contract, bindings)
    values = list(envelope["values"].items())
    envelope["values"] = dict(reversed(values))
    with pytest.raises(CheckpointSchemaError, match="values"):
        validate_checkpoint_envelope(
            envelope, contract=contract, expected_bindings=bindings
        )


def test_new_bundle_metadata_carries_exact_authoritative_state_identity() -> None:
    schema = load_schema_v1()
    metadata = _build_metadata(
        SimpleNamespace(sim_cfg=SimpleNamespace()),
        3,
        True,
        *(["0" * 64] * 9),
    )
    identity = metadata[AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD]
    assert identity == _authoritative_state_checkpoint_identity_v1()
    assert identity["state_schema_version"] == schema.root["schema_version"]
    assert identity["state_schema_sha256"] == schema.schema_hash
    assert identity["component_hashes"] == schema.root["component_hashes"]
    assert (
        identity["checkpoint_component_file_sha256"]
        == schema.root["component_hashes"]["checkpoint_schema.json"]
    )
    assert (
        identity["checkpoint_schema_contract_sha256"]
        == schema.components["checkpoint_schema.json"]["contract_hash"]
    )


def test_checkpoint_identity_mismatch_fails_closed_even_outside_strict_mode() -> None:
    identity = _authoritative_state_checkpoint_identity_v1()
    metadata = {AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD: deepcopy(identity)}
    metadata[AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD][
        "state_schema_sha256"
    ] = "f" * 64
    with pytest.raises(CheckpointStrictError, match="self-hash mismatch"):
        _validate_authoritative_state_checkpoint_metadata(metadata)

    metadata[AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD] = deepcopy(identity)
    metadata[AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD][
        "state_schema_sha256"
    ] = "f" * 64
    from engine.authoritative_state.schema import content_sha256

    material = dict(metadata[AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD])
    material.pop("identity_sha256")
    metadata[AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD][
        "identity_sha256"
    ] = content_sha256(material)
    with pytest.raises(CheckpointStrictError, match="identity mismatch"):
        _validate_authoritative_state_checkpoint_metadata(metadata)


def test_legacy_checkpoint_is_explicitly_classified_not_admitted_as_v1() -> None:
    admission = _validate_authoritative_state_checkpoint_metadata(
        {"schema_version": "checkpoint_bundle_v2"}
    )
    assert admission == {
        "admission_schema_version": (
            "eiel_checkpoint_bundle_v2_authoritative_state_admission_v1"
        ),
        "classification": "legacy_checkpoint_bundle_v2_unbound",
        "migration": "retain_legacy_semantics_without_schema_v1_admission",
        "admitted_as_authoritative_state_v1": False,
        "state_schema_sha256": None,
        "identity_sha256": None,
    }


def test_restore_revalidates_identity_before_applying_state() -> None:
    identity = _authoritative_state_checkpoint_identity_v1()
    metadata = {AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD: deepcopy(identity)}
    admission = _validate_authoritative_state_checkpoint_metadata(metadata)
    payload = {
        "metadata": metadata,
        AUTHORITATIVE_STATE_CHECKPOINT_ADMISSION_FIELD: admission,
    }
    payload["metadata"][AUTHORITATIVE_STATE_CHECKPOINT_METADATA_FIELD][
        "component_hashes"
    ]["checkpoint_schema.json"] = "f" * 64
    destination = SimpleNamespace()
    with pytest.raises(CheckpointStrictError, match="self-hash mismatch"):
        checkpoint_bundle_v2_apply(destination, payload)
    assert vars(destination) == {}
