"""Scientific Chronicle schema binding for authoritative-state projections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .admission import (
    SchemaAdmissionError,
    canonical_sha256,
    require_field_id,
    require_sha256,
    validate_contract_identity,
    validate_exact_bindings,
    validate_field_manifest,
)


CHRONICLE_CONTRACT_KIND = "authoritative_chronicle_schema"


class ChronicleSchemaBindingError(SchemaAdmissionError):
    """A Chronicle event is not bound to its declared source schemas."""


def validate_chronicle_contract(contract: Mapping[str, Any]) -> str:
    try:
        digest = validate_contract_identity(contract)
        if contract.get("kind") != CHRONICLE_CONTRACT_KIND:
            raise ChronicleSchemaBindingError("Chronicle contract kind mismatch")
        validate_field_manifest(contract.get("fields"), contract.get("fields"))
        identity_fields = contract.get("identity_field_ids")
        if not isinstance(identity_fields, list) or not identity_fields:
            raise ChronicleSchemaBindingError(
                "Chronicle identity_field_ids must be a nonempty array"
            )
        declared_ids = {row["field_id"] for row in contract["fields"]}
        for index, field_id in enumerate(identity_fields):
            require_field_id(field_id, path=f"identity_field_ids[{index}]")
            if field_id not in declared_ids:
                raise ChronicleSchemaBindingError(
                    f"Chronicle identity field is undeclared: {field_id}"
                )
        if len(set(identity_fields)) != len(identity_fields):
            raise ChronicleSchemaBindingError("Chronicle identity fields repeat")
        open_namespaces = contract.get("open_namespaces")
        if not isinstance(open_namespaces, Mapping):
            raise ChronicleSchemaBindingError(
                "Chronicle open_namespaces must be an object"
            )
        for namespace, policy in open_namespaces.items():
            if (
                not isinstance(namespace, str)
                or not namespace
                or policy not in {"provenance_required", "forbidden"}
            ):
                raise ChronicleSchemaBindingError(
                    "Chronicle open namespace policy is invalid"
                )
        binding_keys = contract.get("binding_keys")
        if (
            not isinstance(binding_keys, list)
            or not binding_keys
            or len(set(binding_keys)) != len(binding_keys)
        ):
            raise ChronicleSchemaBindingError(
                "Chronicle binding_keys must be a unique nonempty array"
            )
        return digest
    except SchemaAdmissionError as exc:
        if isinstance(exc, ChronicleSchemaBindingError):
            raise
        raise ChronicleSchemaBindingError(str(exc)) from exc


def chronicle_event_id(
    *,
    contract: Mapping[str, Any],
    values: Mapping[str, Any],
) -> str:
    identity = []
    for field_id in contract["identity_field_ids"]:
        if field_id not in values:
            raise ChronicleSchemaBindingError(
                f"Chronicle identity value is missing: {field_id}"
            )
        identity.append([field_id, values[field_id]])
    return "evt_" + canonical_sha256(
        {
            "namespace": contract["schema_version"],
            "identity": identity,
        }
    )[:32]


def validate_chronicle_event(
    event: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> str:
    """Validate formal fields, source bindings, provenance, and stable identity."""

    try:
        wanted_hash = validate_chronicle_contract(contract)
        if event.get("chronicle_schema_version") != contract.get("schema_version"):
            raise ChronicleSchemaBindingError("Chronicle schema version mismatch")
        if require_sha256(
            event.get("chronicle_schema_hash"), path="event.chronicle_schema_hash"
        ) != wanted_hash:
            raise ChronicleSchemaBindingError("Chronicle schema hash mismatch")
        if set(expected_bindings) != set(contract["binding_keys"]):
            raise ChronicleSchemaBindingError(
                "caller binding keys disagree with Chronicle contract"
            )
        validate_exact_bindings(
            event.get("bindings"), expected_bindings, path="event.bindings"
        )
        field_ids = validate_field_manifest(
            contract.get("fields"), event.get("field_manifest")
        )
        values = event.get("values")
        if not isinstance(values, Mapping) or tuple(values) != field_ids:
            raise ChronicleSchemaBindingError(
                "Chronicle values must exactly follow field manifest order"
            )
        open_values = event.get("open_values", {})
        provenance = event.get("open_value_provenance", {})
        if not isinstance(open_values, Mapping) or not isinstance(
            provenance, Mapping
        ):
            raise ChronicleSchemaBindingError(
                "Chronicle open values and provenance must be objects"
            )
        allowed = {
            name
            for name, policy in contract["open_namespaces"].items()
            if policy == "provenance_required"
        }
        if set(open_values) - allowed:
            raise ChronicleSchemaBindingError("undeclared Chronicle open namespace")
        if set(provenance) != set(open_values):
            raise ChronicleSchemaBindingError(
                "Chronicle open namespace provenance mismatch"
            )
        for namespace, namespace_values in open_values.items():
            if not isinstance(namespace_values, Mapping):
                raise ChronicleSchemaBindingError(
                    f"Chronicle open namespace {namespace} must be an object"
                )
            rows = provenance[namespace]
            if not isinstance(rows, Mapping) or set(rows) != set(namespace_values):
                raise ChronicleSchemaBindingError(
                    f"Chronicle provenance keys mismatch in {namespace}"
                )
            for key, source in rows.items():
                if not isinstance(source, Mapping):
                    raise ChronicleSchemaBindingError(
                        f"Chronicle provenance for {namespace}.{key} is invalid"
                    )
                require_field_id(
                    source.get("source_field_id"),
                    path=f"provenance.{namespace}.{key}.source_field_id",
                )
                require_sha256(
                    source.get("source_record_hash"),
                    path=f"provenance.{namespace}.{key}.source_record_hash",
                )
                if source.get("transform") not in {
                    "copied",
                    "renamed",
                    "converted",
                    "derived",
                    "constant",
                }:
                    raise ChronicleSchemaBindingError(
                        f"Chronicle provenance transform is invalid: {namespace}.{key}"
                    )
        expected_event_id = chronicle_event_id(contract=contract, values=values)
        if event.get("event_id") != expected_event_id:
            raise ChronicleSchemaBindingError("Chronicle event ID mismatch")
        return expected_event_id
    except SchemaAdmissionError as exc:
        if isinstance(exc, ChronicleSchemaBindingError):
            raise
        raise ChronicleSchemaBindingError(str(exc)) from exc
