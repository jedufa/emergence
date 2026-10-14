from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from engine.authoritative_state.route_inventory import (
    INVENTORY_SCHEMA_VERSION_V1,
    REQUIRED_BOUNDARY_KINDS_V1,
    RouteInventoryReconciliationError,
    derive_production_route_inventory_v1,
    publish_production_inventory_atomic_v1,
    publish_reconciliation_receipt_atomic_v1,
    reconcile_supported_route_inventory_v1,
    validate_reconciliation_receipt_v1,
)
from engine.authoritative_state.schema import load_schema_v1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inventory(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _inventory_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    schema = load_schema_v1()
    admitted = sorted(
        str(row["field_id"])
        for row in schema.fields
        if row.get("production_admission") == "admitted"
    )
    source_root = tmp_path / "sources"
    source_root.mkdir()
    contexts: list[dict[str, object]] = []
    for index, boundary_kind in enumerate(REQUIRED_BOUNDARY_KINDS_V1):
        source = source_root / f"{boundary_kind}.json"
        source.write_text(
            json.dumps({"boundary_kind": boundary_kind}),
            encoding="utf-8",
        )
        mappings = [
            {
                "context_field_id": f"{boundary_kind}.sentinel",
                "canonical_field_id": admitted[index],
            }
        ]
        if index == 0:
            mappings.extend(
                {
                    "context_field_id": f"analysis.authority.{field_id}",
                    "canonical_field_id": field_id,
                }
                for field_id in admitted[len(REQUIRED_BOUNDARY_KINDS_V1) :]
            )
        contexts.append(
            {
                "context_id": f"{boundary_kind}.v1",
                "boundary_kind": boundary_kind,
                "source_path": source.name,
                "source_sha256": _file_sha(source),
                "field_mappings": mappings,
            }
        )
    inventory: dict[str, object] = {
        "schema_version": INVENTORY_SCHEMA_VERSION_V1,
        "route_id": "patch_fixed_execution_provider_v1",
        "state_schema_version": schema.root["schema_version"],
        "state_schema_sha256": schema.schema_hash,
        "contexts": contexts,
    }
    inventory_path = tmp_path / "inventory.json"
    _write_inventory(inventory_path, inventory)
    return inventory_path, source_root, inventory


def test_reconciles_complete_explicit_inventory_and_atomically_publishes(
    tmp_path: Path,
) -> None:
    inventory_path, source_root, _ = _inventory_fixture(tmp_path)

    receipt = reconcile_supported_route_inventory_v1(
        inventory_path,
        source_root=source_root,
    )

    assert receipt["status"] == "accepted"
    assert receipt["boundary_kinds"] == list(REQUIRED_BOUNDARY_KINDS_V1)
    assert receipt["distinct_canonical_field_count"] == 648
    assert receipt["admitted_canonical_field_count"] == 648
    validate_reconciliation_receipt_v1(receipt)
    output = publish_reconciliation_receipt_atomic_v1(
        tmp_path / "publication" / "receipt.json",
        receipt,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_derives_exact_production_inventory_from_canonical_artifacts(
    tmp_path: Path,
) -> None:
    first = derive_production_route_inventory_v1(source_root=PROJECT_ROOT)
    second = derive_production_route_inventory_v1(source_root=PROJECT_ROOT)

    assert first == second
    assert first["route_id"] == (
        "eiel-production-provider-v1:"
        "eiel-production-runtime-v1"
    )
    contexts = {
        row["boundary_kind"]: row
        for row in first["contexts"]
    }
    assert tuple(sorted(contexts)) == REQUIRED_BOUNDARY_KINDS_V1
    assert {
        kind: len(row["field_mappings"])
        for kind, row in contexts.items()
    } == {
        "analysis_reporting": 7,
        "artifact_manifest": 648,
        "campaign": 648,
        "checkpoint_bundle_v2": 6,
        "chronicle": 8,
        "cpp_native": 606,
        "cuda_resident": 620,
        "execution_route_receipt": 648,
        "provider": 8,
        "pybind": 528,
        "python_orchestration": 648,
    }
    discovered = {
        mapping["canonical_field_id"]
        for row in contexts.values()
        for mapping in row["field_mappings"]
    }
    assert len(discovered) == 648
    for row in contexts.values():
        source = PROJECT_ROOT / row["source_path"]
        assert source.is_file()
        assert row["source_sha256"] == _file_sha(source)

    inventory_path = publish_production_inventory_atomic_v1(
        tmp_path / "production-inventory.json",
        first,
    )
    receipt = reconcile_supported_route_inventory_v1(
        inventory_path,
        source_root=PROJECT_ROOT,
    )
    assert receipt["context_count"] == 11
    assert receipt["distinct_canonical_field_count"] == 648
    assert receipt["discovered_field_occurrence_count"] == 4375


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_boundary", "EIEL_ROUTE_INVENTORY_MISSING"),
        ("extra_boundary", "EIEL_ROUTE_INVENTORY_EXTRA"),
        ("missing_canonical", "EIEL_ROUTE_INVENTORY_MISSING"),
        ("unknown_canonical", "EIEL_ROUTE_INVENTORY_EXTRA"),
        ("duplicate_context", "EIEL_ROUTE_INVENTORY_DUPLICATE"),
        ("duplicate_local_identity", "EIEL_ROUTE_INVENTORY_DUPLICATE"),
        ("stale_source", "EIEL_ROUTE_INVENTORY_STALE_SOURCE"),
        ("stale_schema", "EIEL_ROUTE_INVENTORY_STALE_SCHEMA"),
    ],
)
def test_rejects_incomplete_extra_duplicate_and_stale_identities(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    inventory_path, source_root, original = _inventory_fixture(tmp_path)
    inventory = deepcopy(original)
    contexts = inventory["contexts"]
    assert isinstance(contexts, list)
    first = contexts[0]
    assert isinstance(first, dict)
    mappings = first["field_mappings"]
    assert isinstance(mappings, list)

    if mutation == "missing_boundary":
        contexts.pop()
    elif mutation == "extra_boundary":
        first["boundary_kind"] = "unadmitted_boundary"
    elif mutation == "missing_canonical":
        mappings.pop()
    elif mutation == "unknown_canonical":
        mappings[0]["canonical_field_id"] = "unknown.field.v1"
    elif mutation == "duplicate_context":
        contexts[1]["context_id"] = first["context_id"]
    elif mutation == "duplicate_local_identity":
        mappings.append(deepcopy(mappings[0]))
    elif mutation == "stale_source":
        first["source_sha256"] = "0" * 64
    elif mutation == "stale_schema":
        inventory["state_schema_sha256"] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _write_inventory(inventory_path, inventory)

    with pytest.raises(RouteInventoryReconciliationError) as exc_info:
        reconcile_supported_route_inventory_v1(
            inventory_path,
            source_root=source_root,
        )

    assert exc_info.value.code == expected_code


def test_rejects_receipt_tampering(tmp_path: Path) -> None:
    inventory_path, source_root, _ = _inventory_fixture(tmp_path)
    receipt = reconcile_supported_route_inventory_v1(
        inventory_path,
        source_root=source_root,
    )
    receipt["context_count"] = 0

    with pytest.raises(RouteInventoryReconciliationError) as exc_info:
        validate_reconciliation_receipt_v1(receipt)

    assert exc_info.value.code == "EIEL_ROUTE_INVENTORY_RECEIPT"
