from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from engine.artifact_catalog import parse_artifact_catalog_v2
from engine.planning_receipts import (
    PlanningReceiptError,
    create_planning_receipt,
    verify_planning_receipt,
)
from engine.qualification_plans import validate_plan


RECEIPT_SHA = "b" * 64


def _plan(plan_id: str = "batch-plan-v1") -> dict[str, object]:
    return {
        "schema_version": "eiel_qualification_plan_v1",
        "plan_id": plan_id,
        "description": "data only",
        "dimensions": [{"id": "batch_cells", "values": [2, 8, 32]}],
        "capability_requirements": ["resident_window"],
        "cases": [
            {
                "id": "batch-two",
                "assignments": {"batch_cells": 2},
                "capability_requirements": ["resident_window"],
            }
        ],
    }


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _redigest(receipt: dict[str, object]) -> dict[str, object]:
    updated = copy.deepcopy(receipt)
    updated.pop("receipt_sha256", None)
    updated["receipt_sha256"] = _digest(updated)
    return updated


def _binding(plan: dict[str, object] | None = None) -> dict[str, object]:
    plan = _plan() if plan is None else plan
    return {
        "plan_id": plan["plan_id"],
        "plan_sha256": _digest(validate_plan(plan)),
        "receipt_sha256": RECEIPT_SHA,
        "mechanics_envelope_id": "bounded-v1",
        "qualified_chunk_sizes": [32, 8, 2],
        "capabilities": ["resident_window"],
        "status": "qualified",
    }


def _request(logical: str = "B100", *, tail_policy: str = "allow") -> dict[str, object]:
    return {
        "schema_version": "eiel_planning_request_v1",
        "logical_request": logical,
        "dispatch": {
            "profile_id": "bounded-chunks-v1",
            "qualified_chunk_sizes": [32, 8, 2],
            "max_physical_dispatches": 100,
            "tail_policy": tail_policy,
        },
        "artifact_constraints": {
            "kind": "python_extension",
            "abi": "runtime_abi_v1",
            "mechanics_envelope_id": "bounded-v1",
            "required_capabilities": ["resident_window"],
        },
        "qualification_plan_ids": ["batch-plan-v1"],
    }


def _profile(root: Path, **benchmark: object) -> dict[str, object]:
    values: dict[str, object] = {
        "device_selector": "auto",
        "objective": "throughput",
        "memory_headroom_bytes": 1_000,
    }
    values.update(benchmark)
    return {
        "schema_version": "eiel_capacity_profile_v1",
        "profile_id": "planning-profile-v1",
        "paths": {"runtime_root": str(root.parent), "artifact_root": str(root)},
        "benchmark": values,
    }


def _inventory() -> list[dict[str, object]]:
    return [
        {
            "index": 0,
            "uuid": "GPU-fast-incompatible",
            "compute_capability": "sm_86",
            "available_device_memory_bytes": 10_000,
            "memory_headroom_bytes": 8_000,
            "estimated_latency_ms": 1.0,
            "estimated_throughput": 500.0,
        },
        {
            "index": 1,
            "uuid": "GPU-qualified",
            "compute_capability": "sm_75",
            "available_device_memory_bytes": 10_000,
            "memory_headroom_bytes": 8_000,
            "estimated_latency_ms": 2.0,
            "estimated_throughput": 200.0,
        },
    ]


def _catalog(root: Path, contents: bytes = b"artifact"):
    digest = hashlib.sha256(contents).hexdigest()
    raw = {
        "schema_version": "eiel_artifact_catalog_v2",
        "artifacts": [
            {
                "artifact_id": "qualified-sm75-v1",
                "kind": "python_extension",
                "abi": "runtime_abi_v1",
                "architecture": "sm_75",
                "build_identity": "build-v1",
                "status": "qualified",
                "relative_path": "sm75/artifact.so",
                "sha256": digest,
                "requirements": {
                    "compute_capability": "sm_75",
                    "minimum_device_memory_bytes": 5_000,
                },
                "qualification": {
                    "schema_version": "binding_v1",
                    "envelope_id": "bounded-v1",
                    "receipt_sha256": RECEIPT_SHA,
                    "capabilities": ["resident_window"],
                },
            }
        ],
    }
    catalog = parse_artifact_catalog_v2(raw, artifact_root=root)
    path = root / "sm75/artifact.so"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return catalog, path


def _create(tmp_path: Path, **updates: object) -> dict[str, object]:
    root = tmp_path / "artifacts"
    catalog, _ = _catalog(root)
    args: dict[str, object] = {
        "planning_request": _request(),
        "caller_inventory": _inventory(),
        "capacity_profile": _profile(root),
        "artifact_catalog": catalog,
        "qualification_plans": [_plan()],
        "qualification_bindings": [_binding()],
    }
    args.update(updates)
    return create_planning_receipt(**args)  # type: ignore[arg-type]


def test_end_to_end_receipt_selects_compatible_device_verifies_and_dispatches(
    tmp_path: Path,
) -> None:
    receipt = _create(tmp_path)

    assert receipt["status"] == "planned_verified"
    assert receipt["authority"] == "planning_only"
    assert receipt["selected_device"]["uuid"] == "GPU-qualified"
    assert receipt["selected_device"]["artifact_compatible_device_count"] == 1
    assert receipt["selected_artifact"]["hash_verification"]["verified"] is True
    assert [row["units"] for row in receipt["dispatch_plan"]["physical_dispatches"]] == [
        32,
        32,
        32,
        2,
        2,
    ]
    verify_planning_receipt(receipt)


def test_reordered_caller_inputs_produce_identical_receipt(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    catalog, _ = _catalog(root)
    common = {
        "capacity_profile": _profile(root),
        "artifact_catalog": catalog,
        "qualification_plans": [_plan()],
        "qualification_bindings": [_binding()],
    }
    first = create_planning_receipt(
        _request(),
        caller_inventory=_inventory(),
        **common,  # type: ignore[arg-type]
    )
    second = create_planning_receipt(
        _request(),
        caller_inventory=list(reversed(_inventory())),
        **common,  # type: ignore[arg-type]
    )

    assert first == second


def test_hash_mismatch_prevents_receipt(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    catalog, path = _catalog(root)
    path.write_bytes(b"tampered")

    with pytest.raises(PlanningReceiptError, match="sha256 mismatch"):
        create_planning_receipt(
            _request(),
            caller_inventory=_inventory(),
            capacity_profile=_profile(root),
            artifact_catalog=catalog,
            qualification_plans=[_plan()],
            qualification_bindings=[_binding()],
        )


def test_binding_digest_and_catalog_receipt_mismatches_fail(tmp_path: Path) -> None:
    bad_digest = _binding()
    bad_digest["plan_sha256"] = "0" * 64
    with pytest.raises(PlanningReceiptError, match="plan digest mismatch"):
        _create(tmp_path / "digest", qualification_bindings=[bad_digest])

    wrong_receipt = _binding()
    wrong_receipt["receipt_sha256"] = "c" * 64
    with pytest.raises(PlanningReceiptError, match="qualification receipt"):
        _create(tmp_path / "receipt", qualification_bindings=[wrong_receipt])


def test_binding_must_cover_plan_and_case_capabilities(tmp_path: Path) -> None:
    binding = _binding()
    binding["capabilities"] = []

    with pytest.raises(PlanningReceiptError, match="do not cover plan"):
        _create(tmp_path, qualification_bindings=[binding])


def test_fast_artifact_with_wrong_receipt_provenance_is_excluded(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    catalog, _ = _catalog(root)
    selected = catalog.artifacts[0]
    base = {
        "artifact_id": selected.artifact_id,
        "kind": selected.kind,
        "abi": selected.abi,
        "architecture": selected.architecture,
        "build_identity": selected.build_identity,
        "status": selected.status,
        "relative_path": selected.relative_path.as_posix(),
        "sha256": selected.sha256,
        "requirements": {
            "compute_capability": selected.requirements.compute_capability,
            "minimum_device_memory_bytes": selected.requirements.minimum_device_memory_bytes,
        },
        "qualification": {
            "schema_version": selected.qualification.schema_version,
            "envelope_id": selected.qualification.envelope_id,
            "receipt_sha256": selected.qualification.receipt_sha256,
            "capabilities": list(selected.qualification.capabilities),
        },
    }
    wrong = copy.deepcopy(base)
    wrong.update(
        artifact_id="wrong-provenance-sm86",
        architecture="sm_86",
        relative_path="sm86/wrong.so",
    )
    wrong["requirements"]["compute_capability"] = "sm_86"
    wrong["qualification"]["receipt_sha256"] = "c" * 64
    combined = parse_artifact_catalog_v2(
        {"schema_version": "eiel_artifact_catalog_v2", "artifacts": [base, wrong]},
        artifact_root=root,
    )

    receipt = _create(tmp_path, artifact_catalog=combined)

    assert receipt["selected_device"]["uuid"] == "GPU-qualified"
    assert receipt["selected_device"]["artifact_compatible_device_count"] == 1


def test_tail_reject_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(PlanningReceiptError, match="tail is rejected"):
        _create(tmp_path, planning_request=_request("B101", tail_policy="reject"))


@pytest.mark.parametrize(
    "benchmark",
    [
        {"device_selector": "cpu", "objective": "latency"},
        {"device_selector": "auto", "objective": "correctness"},
    ],
)
def test_gpu_receipt_rejects_cpu_and_correctness_profiles(
    tmp_path: Path, benchmark: dict[str, object]
) -> None:
    root = tmp_path / "artifacts"
    with pytest.raises(PlanningReceiptError):
        _create(tmp_path, capacity_profile=_profile(root, **benchmark))


def test_explicit_capacity_selector_is_applied(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    receipt = _create(
        tmp_path,
        capacity_profile=_profile(
            root, device_selector="uuid:GPU-qualified", objective="latency"
        ),
    )
    assert receipt["selected_device"]["selection_basis"] == "explicit_device"


def test_ambiguous_compatible_artifacts_are_fatal(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    catalog, _ = _catalog(root)
    raw = json.loads(json.dumps({
        "schema_version": "eiel_artifact_catalog_v2",
        "artifacts": [],
    }))
    base = json.loads(json.dumps({
        "artifact_id": catalog.artifacts[0].artifact_id,
        "kind": catalog.artifacts[0].kind,
        "abi": catalog.artifacts[0].abi,
        "architecture": catalog.artifacts[0].architecture,
        "build_identity": catalog.artifacts[0].build_identity,
        "status": catalog.artifacts[0].status,
        "relative_path": catalog.artifacts[0].relative_path.as_posix(),
        "sha256": catalog.artifacts[0].sha256,
        "requirements": {"compute_capability": "sm_75", "minimum_device_memory_bytes": 5000},
        "qualification": {"schema_version": "binding_v1", "envelope_id": "bounded-v1", "receipt_sha256": RECEIPT_SHA, "capabilities": ["resident_window"]},
    }))
    other = copy.deepcopy(base)
    other["artifact_id"] = "qualified-sm75-v2"
    other["relative_path"] = "sm75/artifact-v2.so"
    raw["artifacts"] = [base, other]
    ambiguous = parse_artifact_catalog_v2(raw, artifact_root=root)

    with pytest.raises(PlanningReceiptError, match="ambiguous"):
        _create(tmp_path, artifact_catalog=ambiguous)


def test_receipt_mutation_invalidates_digest(tmp_path: Path) -> None:
    receipt = _create(tmp_path)
    mutated = copy.deepcopy(receipt)
    mutated["selected_device"]["uuid"] = "GPU-mutated"

    with pytest.raises(PlanningReceiptError, match="self digest mismatch"):
        verify_planning_receipt(mutated)


def test_redigested_receipt_rejects_malformed_provenance(tmp_path: Path) -> None:
    receipt = _create(tmp_path)

    bad_catalog = copy.deepcopy(receipt)
    bad_catalog["inputs"]["artifact_catalog_schema_version"] = "unknown"  # type: ignore[index]
    with pytest.raises(PlanningReceiptError, match="catalog schema"):
        verify_planning_receipt(_redigest(bad_catalog))

    bad_path = copy.deepcopy(receipt)
    bad_path["selected_artifact"]["relative_path"] = "../escape.so"  # type: ignore[index]
    with pytest.raises(PlanningReceiptError, match="relative path"):
        verify_planning_receipt(_redigest(bad_path))

    bad_qualification = copy.deepcopy(receipt)
    bad_qualification["selected_artifact"]["qualification"][  # type: ignore[index]
        "receipt_sha256"
    ] = "invalid"
    with pytest.raises(PlanningReceiptError, match="receipt digest"):
        verify_planning_receipt(_redigest(bad_qualification))


def test_redigested_receipt_normalizes_malformed_binding(tmp_path: Path) -> None:
    receipt = _create(tmp_path)
    receipt["qualification_bindings"] = [7]

    with pytest.raises(PlanningReceiptError, match="bindings are invalid"):
        verify_planning_receipt(_redigest(receipt))

    receipt = _create(tmp_path / "missing-fields")
    receipt["qualification_bindings"] = [{}]
    with pytest.raises(PlanningReceiptError, match="fields are not exact"):
        verify_planning_receipt(_redigest(receipt))


def test_dispatch_count_limit_is_bounded_before_materialization(
    tmp_path: Path,
) -> None:
    request = _request()
    request["dispatch"]["max_physical_dispatches"] = 100_001  # type: ignore[index]

    with pytest.raises(PlanningReceiptError, match="receipt maximum"):
        _create(tmp_path, planning_request=request)


def test_redigested_receipt_rejects_noncanonical_dispatch_order(
    tmp_path: Path,
) -> None:
    receipt = _create(tmp_path)
    rows = receipt["dispatch_plan"]["physical_dispatches"]  # type: ignore[index]
    units = [2, 32, 32, 32, 2]
    offset = 0
    for ordinal, (row, unit) in enumerate(zip(rows, units)):
        row.update(ordinal=ordinal, logical_offset=offset, units=unit)
        offset += unit

    with pytest.raises(PlanningReceiptError, match="not canonical"):
        verify_planning_receipt(_redigest(receipt))


def test_offline_verifier_declares_self_consistency_scope(tmp_path: Path) -> None:
    receipt = _create(tmp_path)
    assert receipt["offline_verification_scope"] == "self_consistency_only"

    receipt["offline_verification_scope"] = "source_reverification"
    with pytest.raises(PlanningReceiptError, match="scope"):
        verify_planning_receipt(_redigest(receipt))


def test_verifier_cross_checks_plan_summary_against_binding(tmp_path: Path) -> None:
    receipt = _create(tmp_path)
    receipt["qualification_bindings"][0]["plan_sha256"] = "d" * 64
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = _digest(unsigned)

    with pytest.raises(PlanningReceiptError, match="qualification binding is invalid"):
        verify_planning_receipt(receipt)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt: receipt["selected_artifact"].update(abi="other_abi"),
            "identity violates",
        ),
        (
            lambda receipt: receipt["qualification_bindings"][0].update(
                mechanics_envelope_id="other-envelope"
            ),
            "qualification binding is invalid",
        ),
        (
            lambda receipt: receipt["dispatch_plan"]["physical_dispatches"][0].update(
                units=4
            ),
            "dispatch plan is not canonical",
        ),
    ],
)
def test_verifier_rejects_redigested_internal_inconsistency(
    tmp_path: Path, mutate, message: str
) -> None:
    receipt = _create(tmp_path)
    mutate(receipt)
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = _digest(unsigned)

    with pytest.raises(PlanningReceiptError, match=message):
        verify_planning_receipt(receipt)


def test_receipt_contains_no_artifact_execution_authority(tmp_path: Path) -> None:
    receipt = _create(tmp_path)
    rendered = json.dumps(receipt, sort_keys=True)

    assert '"authority": "planning_only"' in rendered
    assert "argv" not in rendered
    assert "environment" not in rendered
    assert "resolved_path" not in rendered
