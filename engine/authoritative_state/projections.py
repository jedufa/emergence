"""Provider and analysis projection-schema validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .admission import (
    SchemaAdmissionError,
    require_field_id,
    require_sha256,
    validate_contract_identity,
    validate_exact_bindings,
    validate_field_manifest,
)


PROJECTION_KINDS = frozenset(
    {"authoritative_provider_projection", "authoritative_analysis_projection"}
)
PROVIDER_TRANSFORMS = frozenset({"copied", "renamed", "converted", "constant"})
ANALYSIS_TRANSFORMS = PROVIDER_TRANSFORMS | {"derived"}


class ProjectionSchemaError(SchemaAdmissionError):
    """A provider or analysis projection is incomplete or unbound."""


def validate_projection_contract(contract: Mapping[str, Any]) -> str:
    try:
        digest = validate_contract_identity(contract)
        kind = contract.get("kind")
        if kind not in PROJECTION_KINDS:
            raise ProjectionSchemaError("projection contract kind is unsupported")
        validate_field_manifest(contract.get("fields"), contract.get("fields"))
        allowed_transforms = (
            PROVIDER_TRANSFORMS
            if kind == "authoritative_provider_projection"
            else ANALYSIS_TRANSFORMS
        )
        for index, row in enumerate(contract["fields"]):
            transform = row.get("transform")
            if transform not in allowed_transforms:
                raise ProjectionSchemaError(
                    f"fields[{index}].transform is unsupported for {kind}"
                )
            sources = row.get("source_field_ids")
            if not isinstance(sources, list):
                raise ProjectionSchemaError(
                    f"fields[{index}].source_field_ids must be an array"
                )
            for source_index, field_id in enumerate(sources):
                require_field_id(
                    field_id,
                    path=f"fields[{index}].source_field_ids[{source_index}]",
                )
            if transform == "constant":
                if sources or "constant_value" not in row:
                    raise ProjectionSchemaError(
                        f"fields[{index}] constant transform is not deterministic"
                    )
            elif not sources:
                raise ProjectionSchemaError(
                    f"fields[{index}] transform requires source field IDs"
                )
        binding_keys = contract.get("binding_keys")
        if (
            not isinstance(binding_keys, list)
            or not binding_keys
            or len(set(binding_keys)) != len(binding_keys)
        ):
            raise ProjectionSchemaError(
                "projection binding_keys must be a unique nonempty array"
            )
        return digest
    except SchemaAdmissionError as exc:
        if isinstance(exc, ProjectionSchemaError):
            raise
        raise ProjectionSchemaError(str(exc)) from exc


def validate_projection_envelope(
    envelope: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> tuple[str, ...]:
    """Validate exact outputs and per-field provenance against one contract."""

    try:
        wanted_hash = validate_projection_contract(contract)
        if envelope.get("projection_schema_version") != contract.get(
            "schema_version"
        ):
            raise ProjectionSchemaError("projection schema version mismatch")
        if require_sha256(
            envelope.get("projection_schema_hash"),
            path="projection.projection_schema_hash",
        ) != wanted_hash:
            raise ProjectionSchemaError("projection schema hash mismatch")
        if set(expected_bindings) != set(contract["binding_keys"]):
            raise ProjectionSchemaError(
                "caller binding keys disagree with projection contract"
            )
        validate_exact_bindings(
            envelope.get("bindings"),
            expected_bindings,
            path="projection.bindings",
        )
        field_ids = validate_field_manifest(
            contract.get("fields"), envelope.get("output_manifest")
        )
        values = envelope.get("values")
        provenance = envelope.get("provenance")
        if not isinstance(values, Mapping) or tuple(values) != field_ids:
            raise ProjectionSchemaError(
                "projection values must exactly follow output manifest order"
            )
        if not isinstance(provenance, Mapping) or tuple(provenance) != field_ids:
            raise ProjectionSchemaError(
                "projection provenance must exactly follow output manifest order"
            )
        expected_rows = {row["field_id"]: row for row in contract["fields"]}
        for field_id in field_ids:
            actual = provenance[field_id]
            if not isinstance(actual, Mapping):
                raise ProjectionSchemaError(f"{field_id} provenance must be an object")
            expected = expected_rows[field_id]
            if actual.get("transform") != expected.get("transform"):
                raise ProjectionSchemaError(f"{field_id} transform mismatch")
            if actual.get("source_field_ids") != expected.get("source_field_ids"):
                raise ProjectionSchemaError(f"{field_id} source fields mismatch")
            source_hashes = actual.get("source_record_hashes")
            if not isinstance(source_hashes, list) or len(source_hashes) != len(
                expected.get("source_field_ids", [])
            ):
                raise ProjectionSchemaError(f"{field_id} source hash count mismatch")
            for index, digest in enumerate(source_hashes):
                require_sha256(
                    digest, path=f"provenance.{field_id}.source_record_hashes[{index}]"
                )
            if expected.get("transform") == "constant":
                if actual.get("constant_value") != expected.get("constant_value"):
                    raise ProjectionSchemaError(f"{field_id} constant value mismatch")
        return field_ids
    except SchemaAdmissionError as exc:
        if isinstance(exc, ProjectionSchemaError):
            raise
        raise ProjectionSchemaError(str(exc)) from exc
