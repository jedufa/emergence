"""Checkpoint-schema validation for authoritative-state admission."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .admission import (
    SchemaAdmissionError,
    canonical_sha256,
    require_sha256,
    validate_contract_identity,
    validate_exact_bindings,
    validate_field_manifest,
)


CHECKPOINT_CONTRACT_KIND = "authoritative_checkpoint_schema"


class CheckpointSchemaError(SchemaAdmissionError):
    """A checkpoint manifest is not admitted by its component contract."""


def validate_checkpoint_contract(contract: Mapping[str, Any]) -> str:
    try:
        digest = validate_contract_identity(contract)
        if contract.get("kind") != CHECKPOINT_CONTRACT_KIND:
            raise CheckpointSchemaError("checkpoint contract kind mismatch")
        validate_field_manifest(contract.get("fields"), contract.get("fields"))
        binding_keys = contract.get("binding_keys")
        if (
            not isinstance(binding_keys, list)
            or not binding_keys
            or any(not isinstance(key, str) or not key for key in binding_keys)
            or len(set(binding_keys)) != len(binding_keys)
        ):
            raise CheckpointSchemaError(
                "checkpoint binding_keys must be unique nonempty names"
            )
        return digest
    except SchemaAdmissionError as exc:
        if isinstance(exc, CheckpointSchemaError):
            raise
        raise CheckpointSchemaError(str(exc)) from exc


def checkpoint_manifest_hash(field_manifest: Any) -> str:
    return canonical_sha256({"fields": field_manifest})


def validate_checkpoint_envelope(
    envelope: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate one checkpoint envelope without interpreting tensor values."""

    try:
        wanted_hash = validate_checkpoint_contract(contract)
        if not isinstance(envelope, Mapping):
            raise CheckpointSchemaError("checkpoint envelope must be an object")
        if envelope.get("checkpoint_schema_version") != contract.get(
            "schema_version"
        ):
            raise CheckpointSchemaError("checkpoint schema version mismatch")
        actual_hash = require_sha256(
            envelope.get("checkpoint_schema_hash"),
            path="checkpoint.checkpoint_schema_hash",
        )
        if actual_hash != wanted_hash:
            raise CheckpointSchemaError("checkpoint schema hash mismatch")
        if set(expected_bindings) != set(contract["binding_keys"]):
            raise CheckpointSchemaError(
                "caller binding keys disagree with checkpoint contract"
            )
        validate_exact_bindings(
            envelope.get("bindings"),
            expected_bindings,
            path="checkpoint.bindings",
        )
        field_ids = validate_field_manifest(
            contract.get("fields"), envelope.get("field_manifest")
        )
        declared_manifest_hash = require_sha256(
            envelope.get("field_manifest_hash"),
            path="checkpoint.field_manifest_hash",
        )
        observed_manifest_hash = checkpoint_manifest_hash(
            envelope.get("field_manifest")
        )
        if declared_manifest_hash != observed_manifest_hash:
            raise CheckpointSchemaError("checkpoint field manifest hash mismatch")
        values = envelope.get("values")
        if not isinstance(values, Mapping):
            raise CheckpointSchemaError("checkpoint values must be an object")
        if tuple(values) != field_ids:
            raise CheckpointSchemaError(
                "checkpoint values must exactly follow field manifest order"
            )
        return field_ids
    except SchemaAdmissionError as exc:
        if isinstance(exc, CheckpointSchemaError):
            raise
        raise CheckpointSchemaError(str(exc)) from exc
