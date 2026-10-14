"""Independent supported-route context inventory reconciliation.

The canonical schema proves that registered fields are internally resolved.
This module provides the separate proof that an explicit inventory of the
admitted production route discovers those fields across every governed
boundary.  It deliberately consumes a machine-readable inventory rather than
guessing field identities from source text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, NoReturn

from .schema import (
    AuthoritativeStateSchemaV1,
    canonical_json_bytes,
    content_sha256,
    file_sha256,
    load_schema_v1,
)


INVENTORY_SCHEMA_VERSION_V1 = "eiel_supported_route_context_inventory_v1"
RECONCILIATION_RECEIPT_SCHEMA_VERSION_V1 = (
    "eiel_supported_route_context_inventory_reconciliation_receipt_v1"
)
REQUIRED_BOUNDARY_KINDS_V1 = (
    "analysis_reporting",
    "artifact_manifest",
    "campaign",
    "checkpoint_bundle_v2",
    "chronicle",
    "cpp_native",
    "cuda_resident",
    "execution_route_receipt",
    "provider",
    "pybind",
    "python_orchestration",
)
_SCHEMA_PACKAGE_RELATIVE = PurePosixPath("_spec/authoritative_state/v1")
_COMPATIBILITY_BINDINGS_V1 = {
    "analysis_projection": "analysis_projection_schema.json",
    "checkpoint": "checkpoint_schema.json",
    "chronicle": "chronicle_schema.json",
    "provider_projection": "provider_projection_schema.json",
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "route_id",
        "state_schema_version",
        "state_schema_sha256",
        "contexts",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "context_id",
        "boundary_kind",
        "source_path",
        "source_sha256",
        "field_mappings",
    }
)
_MAPPING_FIELDS = frozenset({"context_field_id", "canonical_field_id"})
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "route_id",
        "state_schema_version",
        "state_schema_sha256",
        "inventory_file_name",
        "inventory_sha256",
        "boundary_kinds",
        "context_count",
        "discovered_field_occurrence_count",
        "distinct_canonical_field_count",
        "admitted_canonical_field_count",
        "contexts",
        "receipt_sha256",
    }
)


class RouteInventoryReconciliationError(ValueError):
    """Stable fail-closed error for route-inventory reconciliation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise RouteInventoryReconciliationError(code, message)


def _exact_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("EIEL_ROUTE_INVENTORY_TYPE", f"{path} must be an object")
    actual = set(value)
    if actual != fields:
        _fail(
            "EIEL_ROUTE_INVENTORY_FIELDS",
            f"{path} fields differ: missing={sorted(fields - actual)} "
            f"extra={sorted(actual - fields)}",
        )
    return value


def _nonempty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("EIEL_ROUTE_INVENTORY_IDENTITY", f"{path} must be a nonempty string")
    return value


def _token(value: Any, *, path: str) -> str:
    text = _nonempty_string(value, path=path)
    if _TOKEN.fullmatch(text) is None:
        _fail("EIEL_ROUTE_INVENTORY_IDENTITY", f"{path} has invalid token format")
    return text


def _sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail("EIEL_ROUTE_INVENTORY_IDENTITY", f"{path} is not a SHA-256 digest")
    return value


def _sequence(value: Any, *, path: str, nonempty: bool = True) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("EIEL_ROUTE_INVENTORY_TYPE", f"{path} must be an array")
    if nonempty and not value:
        _fail("EIEL_ROUTE_INVENTORY_MISSING", f"{path} must not be empty")
    return value


def _resolve_source_path(source_root: Path, raw_path: Any, *, path: str) -> Path:
    text = _nonempty_string(raw_path, path=path)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != text:
        _fail(
            "EIEL_ROUTE_INVENTORY_IDENTITY",
            f"{path} must be a normalized relative POSIX path",
        )
    root = source_root.resolve(strict=True)
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        _fail("EIEL_ROUTE_INVENTORY_IDENTITY", f"{path} escapes source root")
    if not candidate.is_file():
        _fail(
            "EIEL_ROUTE_INVENTORY_SOURCE_MISSING",
            f"{path} does not identify a file: {text}",
        )
    return candidate


def _load_inventory(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("EIEL_ROUTE_INVENTORY_UNREADABLE", f"cannot read {path}: {exc}")
    return _exact_mapping(value, fields=_ROOT_FIELDS, path="inventory")


def _component_field_ids(component: Mapping[str, Any]) -> set[str]:
    field_ids: set[str] = set()
    for raw_row in component.get("fields", []):
        if not isinstance(raw_row, Mapping):
            _fail(
                "EIEL_ROUTE_INVENTORY_DERIVATION",
                "canonical component field row is not an object",
            )
        field_id = raw_row.get("field_id")
        if not isinstance(field_id, str):
            _fail(
                "EIEL_ROUTE_INVENTORY_DERIVATION",
                "canonical component field identity is absent",
            )
        field_ids.add(field_id)
        for source_id in raw_row.get("source_field_ids", []):
            if not isinstance(source_id, str):
                _fail(
                    "EIEL_ROUTE_INVENTORY_DERIVATION",
                    "canonical projection source identity is invalid",
                )
            field_ids.add(source_id)
    for identity_id in component.get("identity_field_ids", []):
        if not isinstance(identity_id, str):
            _fail(
                "EIEL_ROUTE_INVENTORY_DERIVATION",
                "canonical component identity field is invalid",
            )
        field_ids.add(identity_id)
    return field_ids


def _registry_ids_for_source_suffix(
    fields: Sequence[Mapping[str, Any]],
    *,
    suffixes: frozenset[str],
) -> set[str]:
    """Select identities using declared source-anchor file types only."""

    selected: set[str] = set()
    for field in fields:
        anchors = field.get("source_anchors")
        if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
            _fail(
                "EIEL_ROUTE_INVENTORY_DERIVATION",
                f"{field.get('field_id')} source anchors are invalid",
            )
        if any(
            isinstance(anchor, Mapping)
            and isinstance(anchor.get("path"), str)
            and PurePosixPath(anchor["path"]).suffix in suffixes
            for anchor in anchors
        ):
            selected.add(str(field["field_id"]))
    return selected


def _registry_ids_for_consumer(
    fields: Sequence[Mapping[str, Any]],
    *,
    consumer: str,
) -> set[str]:
    selected: set[str] = set()
    for field in fields:
        consumers = field.get("consumers")
        if not isinstance(consumers, Sequence) or isinstance(consumers, (str, bytes)):
            _fail(
                "EIEL_ROUTE_INVENTORY_DERIVATION",
                f"{field.get('field_id')} consumer declaration is invalid",
            )
        if consumer in consumers:
            selected.add(str(field["field_id"]))
    return selected


def _context_payload(
    *,
    context_id: str,
    boundary_kind: str,
    source_path: PurePosixPath,
    source_root: Path,
    field_ids: set[str],
) -> dict[str, Any]:
    if not field_ids:
        _fail(
            "EIEL_ROUTE_INVENTORY_DERIVATION",
            f"{boundary_kind} has no explicitly declared field identities",
        )
    resolved_source = _resolve_source_path(
        source_root,
        source_path.as_posix(),
        path=f"derived.{boundary_kind}.source_path",
    )
    return {
        "context_id": context_id,
        "boundary_kind": boundary_kind,
        "source_path": source_path.as_posix(),
        "source_sha256": file_sha256(resolved_source),
        "field_mappings": [
            {
                "context_field_id": f"{boundary_kind}:{field_id}",
                "canonical_field_id": field_id,
            }
            for field_id in sorted(field_ids)
        ],
    }


def derive_production_route_inventory_v1(
    *,
    source_root: str | Path,
    schema: AuthoritativeStateSchemaV1 | None = None,
) -> dict[str, Any]:
    """Derive the admitted-route inventory from exact canonical declarations.

    Language contexts use field-registry source anchors and consumer enums.
    Persistence/projection contexts use their canonical component field lists.
    Campaign and receipt envelopes bind the complete registry through
    ``schema.json`` and the compatibility matrix; artifact participation is an
    explicit per-field registry flag.  No source-text or name-token scan is
    performed.
    """

    root = Path(source_root).resolve(strict=True)
    package = root / Path(*_SCHEMA_PACKAGE_RELATIVE.parts)
    resolved_schema = schema or load_schema_v1(package)
    admitted_fields = tuple(
        row
        for row in resolved_schema.fields
        if row.get("production_admission") == "admitted"
    )
    admitted_ids = {str(row["field_id"]) for row in admitted_fields}
    if not admitted_ids:
        _fail(
            "EIEL_ROUTE_INVENTORY_DERIVATION",
            "canonical registry has no admitted production fields",
        )
    supported_surfaces = resolved_schema.root.get("supported_surfaces")
    if (
        not isinstance(supported_surfaces, Sequence)
        or isinstance(supported_surfaces, (str, bytes))
        or len(supported_surfaces) != 1
        or not isinstance(supported_surfaces[0], Mapping)
    ):
        _fail(
            "EIEL_ROUTE_INVENTORY_DERIVATION",
            "exactly one admitted production surface is required",
        )
    provider_id = _token(
        supported_surfaces[0].get("provider_id"),
        path="schema.supported_surfaces[0].provider_id",
    )
    profile_id = _token(
        supported_surfaces[0].get("profile_id"),
        path="schema.supported_surfaces[0].profile_id",
    )

    compatibility = resolved_schema.components.get("compatibility_matrix.json")
    if compatibility is None:
        _fail(
            "EIEL_ROUTE_INVENTORY_DERIVATION",
            "canonical compatibility matrix is absent",
        )
    compatible_sets = compatibility.get("compatible_sets")
    if (
        not isinstance(compatible_sets, Sequence)
        or isinstance(compatible_sets, (str, bytes))
        or len(compatible_sets) != 1
        or not isinstance(compatible_sets[0], Mapping)
    ):
        _fail(
            "EIEL_ROUTE_INVENTORY_DERIVATION",
            "exactly one canonical compatible component set is required",
        )
    bindings = compatible_sets[0].get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        _COMPATIBILITY_BINDINGS_V1
    ):
        _fail(
            "EIEL_ROUTE_INVENTORY_DERIVATION",
            "canonical compatibility bindings differ from the production set",
        )
    for binding_name, component_name in _COMPATIBILITY_BINDINGS_V1.items():
        component = resolved_schema.components.get(component_name)
        if component is None or bindings[binding_name] != component.get(
            "contract_hash"
        ):
            _fail(
                "EIEL_ROUTE_INVENTORY_DERIVATION",
                f"{binding_name} compatibility binding is stale",
            )

    component_ids: dict[str, set[str]] = {}
    for component_name in (
        "analysis_projection_schema.json",
        "checkpoint_schema.json",
        "chronicle_schema.json",
        "provider_projection_schema.json",
    ):
        ids = _component_field_ids(resolved_schema.components[component_name])
        unknown = ids - admitted_ids
        if unknown:
            _fail(
                "EIEL_ROUTE_INVENTORY_DERIVATION",
                f"{component_name} contains non-admitted identities: "
                f"{sorted(unknown)[:5]}",
            )
        component_ids[component_name] = ids

    artifact_ids = {
        str(row["field_id"])
        for row in admitted_fields
        if row.get("artifact_participation") is True
    }
    context_specs = (
        (
            "production.python-orchestration.v1",
            "python_orchestration",
            "field_registry.json",
            _registry_ids_for_source_suffix(
                admitted_fields,
                suffixes=frozenset({".py"}),
            ),
        ),
        (
            "production.cpp-native.v1",
            "cpp_native",
            "field_registry.json",
            _registry_ids_for_consumer(admitted_fields, consumer="native CPU"),
        ),
        (
            "production.cuda-resident.v1",
            "cuda_resident",
            "field_registry.json",
            _registry_ids_for_consumer(admitted_fields, consumer="CUDA"),
        ),
        (
            "production.pybind.v1",
            "pybind",
            "abi_variants.json",
            _registry_ids_for_consumer(admitted_fields, consumer="pybind"),
        ),
        (
            "production.checkpoint-bundle-v2.v1",
            "checkpoint_bundle_v2",
            "checkpoint_schema.json",
            component_ids["checkpoint_schema.json"],
        ),
        (
            "production.chronicle.v1",
            "chronicle",
            "chronicle_schema.json",
            component_ids["chronicle_schema.json"],
        ),
        (
            "production.provider.v1",
            "provider",
            "provider_projection_schema.json",
            component_ids["provider_projection_schema.json"],
        ),
        (
            "production.campaign.v1",
            "campaign",
            "schema.json",
            admitted_ids,
        ),
        (
            "production.execution-route-receipt.v1",
            "execution_route_receipt",
            "compatibility_matrix.json",
            admitted_ids,
        ),
        (
            "production.analysis-reporting.v1",
            "analysis_reporting",
            "analysis_projection_schema.json",
            component_ids["analysis_projection_schema.json"],
        ),
        (
            "production.artifact-manifest.v1",
            "artifact_manifest",
            "field_registry.json",
            artifact_ids,
        ),
    )
    contexts = [
        _context_payload(
            context_id=context_id,
            boundary_kind=boundary_kind,
            source_path=_SCHEMA_PACKAGE_RELATIVE / component_name,
            source_root=root,
            field_ids=set(field_ids),
        )
        for context_id, boundary_kind, component_name, field_ids in context_specs
    ]
    contexts.sort(key=lambda row: row["context_id"])
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION_V1,
        "route_id": f"{provider_id}:{profile_id}",
        "state_schema_version": resolved_schema.root["schema_version"],
        "state_schema_sha256": resolved_schema.schema_hash,
        "contexts": contexts,
    }


def reconcile_supported_route_inventory_v1(
    inventory_path: str | Path,
    *,
    source_root: str | Path,
    schema: AuthoritativeStateSchemaV1 | None = None,
) -> dict[str, Any]:
    """Reconcile one explicit route inventory and return a self-hashed receipt."""

    inventory_file = Path(inventory_path)
    inventory = _load_inventory(inventory_file)
    resolved_schema = schema or load_schema_v1()

    if inventory["schema_version"] != INVENTORY_SCHEMA_VERSION_V1:
        _fail(
            "EIEL_ROUTE_INVENTORY_VERSION",
            f"expected {INVENTORY_SCHEMA_VERSION_V1}, "
            f"got {inventory['schema_version']!r}",
        )
    route_id = _token(inventory["route_id"], path="inventory.route_id")
    if inventory["state_schema_version"] != resolved_schema.root["schema_version"]:
        _fail(
            "EIEL_ROUTE_INVENTORY_STALE_SCHEMA",
            "inventory state schema version differs from the loaded schema",
        )
    if inventory["state_schema_sha256"] != resolved_schema.schema_hash:
        _fail(
            "EIEL_ROUTE_INVENTORY_STALE_SCHEMA",
            "inventory state schema hash differs from the loaded schema",
        )

    admitted_ids = {
        str(row["field_id"])
        for row in resolved_schema.fields
        if row.get("production_admission") == "admitted"
    }
    contexts = _sequence(inventory["contexts"], path="inventory.contexts")
    context_ids: set[str] = set()
    observed_kinds: list[str] = []
    discovered_ids: set[str] = set()
    receipt_contexts: list[dict[str, Any]] = []
    occurrence_count = 0

    for context_index, raw_context in enumerate(contexts):
        context_path = f"inventory.contexts[{context_index}]"
        context = _exact_mapping(
            raw_context,
            fields=_CONTEXT_FIELDS,
            path=context_path,
        )
        context_id = _token(context["context_id"], path=f"{context_path}.context_id")
        if context_id in context_ids:
            _fail(
                "EIEL_ROUTE_INVENTORY_DUPLICATE",
                f"duplicate context_id {context_id!r}",
            )
        context_ids.add(context_id)
        boundary_kind = _token(
            context["boundary_kind"],
            path=f"{context_path}.boundary_kind",
        )
        if boundary_kind not in REQUIRED_BOUNDARY_KINDS_V1:
            _fail(
                "EIEL_ROUTE_INVENTORY_EXTRA",
                f"{context_id} has unsupported boundary kind {boundary_kind!r}",
            )
        observed_kinds.append(boundary_kind)

        source_path = _nonempty_string(
            context["source_path"],
            path=f"{context_path}.source_path",
        )
        resolved_source = _resolve_source_path(
            Path(source_root),
            source_path,
            path=f"{context_path}.source_path",
        )
        expected_source_sha = _sha256(
            context["source_sha256"],
            path=f"{context_path}.source_sha256",
        )
        actual_source_sha = file_sha256(resolved_source)
        if actual_source_sha != expected_source_sha:
            _fail(
                "EIEL_ROUTE_INVENTORY_STALE_SOURCE",
                f"{context_id} source hash differs: expected "
                f"{expected_source_sha}, got {actual_source_sha}",
            )

        raw_mappings = _sequence(
            context["field_mappings"],
            path=f"{context_path}.field_mappings",
        )
        local_ids: set[str] = set()
        normalized_mappings: list[dict[str, str]] = []
        for mapping_index, raw_mapping in enumerate(raw_mappings):
            mapping_path = f"{context_path}.field_mappings[{mapping_index}]"
            mapping = _exact_mapping(
                raw_mapping,
                fields=_MAPPING_FIELDS,
                path=mapping_path,
            )
            local_id = _nonempty_string(
                mapping["context_field_id"],
                path=f"{mapping_path}.context_field_id",
            )
            if local_id in local_ids:
                _fail(
                    "EIEL_ROUTE_INVENTORY_DUPLICATE",
                    f"{context_id} repeats context field identity {local_id!r}",
                )
            local_ids.add(local_id)
            canonical_id = _nonempty_string(
                mapping["canonical_field_id"],
                path=f"{mapping_path}.canonical_field_id",
            )
            if canonical_id not in admitted_ids:
                _fail(
                    "EIEL_ROUTE_INVENTORY_EXTRA",
                    f"{context_id}.{local_id} maps to unknown or non-admitted "
                    f"canonical field {canonical_id!r}",
                )
            discovered_ids.add(canonical_id)
            occurrence_count += 1
            normalized_mappings.append(
                {
                    "context_field_id": local_id,
                    "canonical_field_id": canonical_id,
                }
            )
        normalized_mappings.sort(
            key=lambda row: (row["context_field_id"], row["canonical_field_id"])
        )
        receipt_contexts.append(
            {
                "context_id": context_id,
                "boundary_kind": boundary_kind,
                "source_path": source_path,
                "source_sha256": actual_source_sha,
                "field_mapping_count": len(normalized_mappings),
                "field_mappings_sha256": content_sha256(normalized_mappings),
                "field_mappings": normalized_mappings,
            }
        )

    actual_kinds = set(observed_kinds)
    required_kinds = set(REQUIRED_BOUNDARY_KINDS_V1)
    if actual_kinds != required_kinds:
        _fail(
            "EIEL_ROUTE_INVENTORY_MISSING",
            f"boundary kinds differ: missing={sorted(required_kinds - actual_kinds)} "
            f"extra={sorted(actual_kinds - required_kinds)}",
        )
    missing_ids = admitted_ids - discovered_ids
    if missing_ids:
        _fail(
            "EIEL_ROUTE_INVENTORY_MISSING",
            f"{len(missing_ids)} admitted canonical field identities are absent; "
            f"first={sorted(missing_ids)[:5]}",
        )
    extra_ids = discovered_ids - admitted_ids
    if extra_ids:
        _fail(
            "EIEL_ROUTE_INVENTORY_EXTRA",
            f"non-admitted canonical identities were discovered: {sorted(extra_ids)[:5]}",
        )

    receipt_contexts.sort(key=lambda row: row["context_id"])
    payload: dict[str, Any] = {
        "schema_version": RECONCILIATION_RECEIPT_SCHEMA_VERSION_V1,
        "status": "accepted",
        "route_id": route_id,
        "state_schema_version": resolved_schema.root["schema_version"],
        "state_schema_sha256": resolved_schema.schema_hash,
        "inventory_file_name": inventory_file.name,
        "inventory_sha256": file_sha256(inventory_file),
        "boundary_kinds": list(REQUIRED_BOUNDARY_KINDS_V1),
        "context_count": len(receipt_contexts),
        "discovered_field_occurrence_count": occurrence_count,
        "distinct_canonical_field_count": len(discovered_ids),
        "admitted_canonical_field_count": len(admitted_ids),
        "contexts": receipt_contexts,
    }
    payload["receipt_sha256"] = content_sha256(payload)
    validate_reconciliation_receipt_v1(payload)
    return payload


def validate_reconciliation_receipt_v1(receipt: Any) -> None:
    """Validate exact shape and self-hash of a published receipt."""

    row = _exact_mapping(receipt, fields=_RECEIPT_FIELDS, path="receipt")
    if row["schema_version"] != RECONCILIATION_RECEIPT_SCHEMA_VERSION_V1:
        _fail("EIEL_ROUTE_INVENTORY_RECEIPT", "receipt schema version differs")
    if row["status"] != "accepted":
        _fail("EIEL_ROUTE_INVENTORY_RECEIPT", "receipt status is not accepted")
    supplied = _sha256(row["receipt_sha256"], path="receipt.receipt_sha256")
    material = dict(row)
    material.pop("receipt_sha256")
    if supplied != content_sha256(material):
        _fail("EIEL_ROUTE_INVENTORY_RECEIPT", "receipt self-hash differs")


def _publish_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        _fail(
            "EIEL_ROUTE_INVENTORY_PUBLISH",
            f"cannot atomically publish {target}: {exc}",
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def publish_production_inventory_atomic_v1(
    path: str | Path,
    inventory: Mapping[str, Any],
) -> Path:
    """Atomically publish an exact-shape derived production inventory."""

    _exact_mapping(inventory, fields=_ROOT_FIELDS, path="inventory")
    if inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION_V1:
        _fail(
            "EIEL_ROUTE_INVENTORY_VERSION",
            "cannot publish a differently versioned inventory",
        )
    return _publish_json_atomic(path, inventory)


def publish_reconciliation_receipt_atomic_v1(
    path: str | Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Validate and atomically replace one receipt at ``path``."""

    validate_reconciliation_receipt_v1(receipt)
    return _publish_json_atomic(path, receipt)


__all__ = [
    "INVENTORY_SCHEMA_VERSION_V1",
    "RECONCILIATION_RECEIPT_SCHEMA_VERSION_V1",
    "REQUIRED_BOUNDARY_KINDS_V1",
    "RouteInventoryReconciliationError",
    "derive_production_route_inventory_v1",
    "publish_production_inventory_atomic_v1",
    "publish_reconciliation_receipt_atomic_v1",
    "reconcile_supported_route_inventory_v1",
    "validate_reconciliation_receipt_v1",
]
