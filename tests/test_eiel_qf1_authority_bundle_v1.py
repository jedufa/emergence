from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.campaigns.qualification_framework import (
    FULL_QUALIFICATION_REQUIRED,
    QUALIFIED,
    QualificationCatalogV1,
    validate_authority_bundle,
    write_catalog,
)
from tools.scripts import eiel_qualification


SOURCE_COMMIT = "1c7997d69f657263f2e754188fa0bd791690e66d"
PROVIDER = "eiel-production-provider-v1"
ARTIFACT_SHA = "96f8f23147453833c763884f143e1881ef0a003c17b07137682060cd7134ad66"
GPU_ROUTE = "0030497933fddd87f9789567d08759a5b8e0c50f4b3838238a625ddd05c1133b"
CPU_ROUTE = "b6fe7cf8ce287750571d232aa22f5c4dbd3f0983eea92c7cfedc3da0303533df"
GPU_OBLIGATIONS = tuple(f"gpu-obligation-{index:02d}" for index in range(20))
CPU_OBLIGATIONS = tuple(f"cpu-obligation-{index:02d}" for index in range(17))


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(*, backend: str) -> dict:
    runtime = "eiel-production-runtime-v1" if backend == "cuda_resident" else "eiel-production-runtime-v1-cpu-native"
    return {
        "backend": backend,
        "batch_size": 48 if backend == "cuda_resident" else 1,
        "controller_family": "fixed_mlp",
        "device": "cuda" if backend == "cuda_resident" else "cpu",
        "execution_route_sha256": GPU_ROUTE if backend == "cuda_resident" else CPU_ROUTE,
        "generations": 200,
        "grid_height": 32,
        "grid_width": 32,
        "initial_population": 256,
        "provider_id": PROVIDER,
        "resident_capacity": 256,
        "runtime_id": runtime,
        "source_commit": SOURCE_COMMIT,
        "ticks_per_generation": 80,
        "world_family": "patch",
    }


def _catalog() -> QualificationCatalogV1:
    anchors = []
    for backend in ("cuda_resident", "cpu_native"):
        target = _target(backend=backend)
        anchors.append(
            {
                "anchor_id": f"w32-c256-t80-{target['device']}",
                "config": target,
                "decision": QUALIFIED,
                "evidence": [],
                "execution_route_sha256": target["execution_route_sha256"],
                "provider_id": PROVIDER,
                "runtime_id": target["runtime_id"],
                "scope": "CONFIGURATION_OPERATING_ENVELOPE",
                "source_commit": SOURCE_COMMIT,
            }
        )
    return QualificationCatalogV1.from_mapping(
        {
            "anchors": anchors,
            "envelopes": [],
            "schema_version": "eiel_qualification_framework_v1",
        }
    )


def _route_receipt(*, backend: str) -> dict:
    target = _target(backend=backend)
    obligations = GPU_OBLIGATIONS if backend == "cuda_resident" else CPU_OBLIGATIONS
    return {
        "artifact_sha256": ARTIFACT_SHA,
        "authority_state": "route_qualified",
        "checkpoint_results": {"case_count": 8, "exact": True},
        "covered_obligations": list(obligations),
        "exact_comparisons": [
            {"case_id": "disk-continuation", "match": True},
            {"case_id": "reporting-enabled", "match": True},
        ],
        "execution_route_sha256": target["execution_route_sha256"],
        "profile_id": target["runtime_id"],
        "provider_id": PROVIDER,
        "reporting_results": {"case_count": 8, "exact": True},
        "route_contract": {
            "components": [
                {
                    "admitted_ticks_per_generation": [80, 160],
                    "component_id": "native_mechanics_binding",
                    "request_ticks_per_generation": 80,
                    "ticks_per_generation": 80,
                },
                {
                    "backend": backend,
                    "component_id": "runtime_execution_backend",
                    "device": target["device"],
                    "fallback_allowed": False,
                    "profile_id": target["runtime_id"],
                },
            ],
            "required_obligations": list(obligations),
        },
        "schema_version": "eiel_execution_route_qualification_v1",
    }


def _preflight(*, backend: str, route_receipt_sha256: str) -> dict:
    target = _target(backend=backend)
    obligations = GPU_OBLIGATIONS if backend == "cuda_resident" else CPU_OBLIGATIONS
    cuda_resource = (
        {
            "admission_required": True,
            "free_covers_admission_bound_and_headroom": True,
            "live_admission_bound_source": "provider_exact_upper_bound",
            "provider_estimate_within_budget": True,
            "runtime_environment_ok": True,
        }
        if backend == "cuda_resident"
        else {
            "admission_required": False,
            "skip_reason": "profile backend 'cpu_native'",
        }
    )
    return {
        "artifacts": [
            {
                "actual_sha256": ARTIFACT_SHA,
                "name": "cuda_extension",
                "regular_file": True,
                "sha256_exact": True,
            }
        ],
        "execution_provider": {
            "provider_id": PROVIDER,
            "resource_estimator_id": "w32-b48-estimator" if backend == "cuda_resident" else "cpu-estimator",
        },
        "execution_route": {
            "artifact_conformance_exact": True,
            "covered_obligations": list(obligations),
            "execution_route_sha256": target["execution_route_sha256"],
            "missing_obligations": [],
            "profile_binding_exact": True,
            "provider_binding_exact": True,
            "qualification_receipt_sha256": route_receipt_sha256,
            "required_obligations": list(obligations),
            "route_hash_exact": True,
        },
        "ok": True,
        "plan_sha256": f"{target['device']}-plan-sha256",
        "profile": {
            "profile_id": target["runtime_id"],
            "requested_physical_batch_cells": target["batch_size"],
        },
        "qualification_reuse": {
            "batch_admission": {
                "admitted_range": [1, 72] if backend == "cuda_resident" else [1, 1],
                "within_admitted_range": True,
            },
            "identity_binding": {
                "capacity": {"fixed_invariants": {"parameter.patch.capacity": 256}},
                "mechanics": {
                    "backend": backend,
                    "fixed_invariants": {
                        "device_owner": "cuda_resident_no_fallback"
                        if backend == "cuda_resident"
                        else "cpu_native_b1_no_cuda_fallback",
                        "parameter.patch.grid_size": 32,
                        "parameter.patch.initial_population": 256,
                    },
                },
            },
            "qualification_key_hash_exact": True,
            "reuses_bound_qualification": True,
        },
        "ready_for_execution": True,
        "resources": {
            "cuda": cuda_resource,
            "vram_bytes_upper_bound": 2453667872 if backend == "cuda_resident" else 1957303976,
        },
        "schema_version": "eiel_campaign_preflight_v1",
    }


def _comparison_summary() -> dict:
    return {
        "comparisons": [
            {"comparison_match": True, "kind": "fresh_repeat", "status": "completed"},
            {"comparison_match": True, "kind": "reporting_parity", "status": "completed"},
            {"comparison_match": True, "kind": "checkpoint_continuation", "status": "completed"},
        ],
        "schema_version": "w32_cpu_t80_g14_v3_comparison_summary_v1",
    }


def _bundle(tmp_path: Path, *, backend: str) -> dict:
    target = _target(backend=backend)
    root = tmp_path / target["device"]
    route_path = root / "route_receipt.json"
    route_sha = _write_json(route_path, _route_receipt(backend=backend))
    preflight_path = root / "preflight.json"
    preflight_sha = _write_json(
        preflight_path,
        _preflight(backend=backend, route_receipt_sha256=route_sha),
    )
    evidence_files = {
        "preflight": {"kind": "campaign_preflight", "path": str(preflight_path), "sha256": preflight_sha},
        "route_receipt": {"kind": "execution_route_qualification", "path": str(route_path), "sha256": route_sha},
    }
    expected = {
        "artifact_sha256": ARTIFACT_SHA,
        "backend": backend,
        "bounded_generations_proven": 14 if backend == "cpu_native" else 1,
        "device": target["device"],
        "device_owner": "cuda_resident_no_fallback" if backend == "cuda_resident" else "cpu_native_b1_no_cuda_fallback",
        "execution_route_sha256": target["execution_route_sha256"],
        "minimum_exact_comparisons": 2,
        "plan_sha256": f"{target['device']}-plan-sha256",
        "provider_id": PROVIDER,
        "qf_record_id": f"w32-c256-t80-{target['device']}",
        "resource_estimator_id": "w32-b48-estimator" if backend == "cuda_resident" else "cpu-estimator",
        "route_obligation_count": 20 if backend == "cuda_resident" else 17,
        "runtime_id": target["runtime_id"],
        "scope": "CONFIGURATION_OPERATING_ENVELOPE",
        "source_commit": SOURCE_COMMIT,
    }
    if backend == "cpu_native":
        comparison_path = root / "comparison_summary.json"
        comparison_sha = _write_json(comparison_path, _comparison_summary())
        evidence_files["comparison_summary"] = {
            "kind": "cpu_continuation_comparison",
            "path": str(comparison_path),
            "sha256": comparison_sha,
        }
        expected["cpu_continuation_required"] = True
    return {
        "authority_id": f"w32-c256-t80-{target['device']}-authority",
        "evidence_files": evidence_files,
        "expected": expected,
        "exclusions": ["no T160 inference", "no full uninterrupted G200 transcript claim"],
        "limitations": ["bounded mechanical QF1 authority only"],
        "schema_version": "eiel_qf1_authority_bundle_v1",
        "scope": "CONFIGURATION_OPERATING_ENVELOPE",
        "target": target,
    }


def _replace_evidence_payload(bundle: dict, evidence_id: str, payload: dict) -> None:
    path = Path(bundle["evidence_files"][evidence_id]["path"])
    bundle["evidence_files"][evidence_id]["sha256"] = _write_json(path, payload)
    if evidence_id == "route_receipt":
        preflight_path = Path(bundle["evidence_files"]["preflight"]["path"])
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["execution_route"]["qualification_receipt_sha256"] = bundle["evidence_files"][evidence_id]["sha256"]
        bundle["evidence_files"]["preflight"]["sha256"] = _write_json(preflight_path, preflight)


def _codes(result: dict) -> set[str]:
    return {str(row.get("code")) for row in result["reasons"]}


def test_authority_bundle_accepts_real_cpu_and_gpu_shapes(tmp_path: Path) -> None:
    catalog = _catalog()

    for backend in ("cuda_resident", "cpu_native"):
        result = validate_authority_bundle(_bundle(tmp_path, backend=backend), catalog=catalog)

        assert result["decision"] == QUALIFIED
        assert result["ok"] is True
        assert result["scope"] == "CONFIGURATION_OPERATING_ENVELOPE"


def test_validate_authority_cli_returns_success_for_scoped_gpu_bundle(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    bundle_path = tmp_path / "bundle.json"
    write_catalog(catalog_path, _catalog())
    _write_json(bundle_path, _bundle(tmp_path, backend="cuda_resident"))

    assert eiel_qualification.main(
        [
            "validate-authority",
            "--bundle",
            str(bundle_path),
            "--catalog",
            str(catalog_path),
            "--output",
            str(tmp_path / "decision.json"),
        ]
    ) == 0


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("stale_source", "source_commit_mismatch"),
        ("wrong_route_hash", "execution_route_mismatch"),
        ("wrong_plan_hash", "plan_sha256_mismatch"),
        ("t80_receipt_applied_to_t160", "target_ticks_not_bound"),
        ("cpu_receipt_applied_to_gpu", "backend_mismatch"),
        ("missing_obligation", "obligation_coverage_incomplete"),
        ("preflight_not_ready", "preflight_not_ready_for_execution"),
        ("missing_gpu_resource", "gpu_headroom_not_confirmed"),
        ("cpu_continuation_mismatch", "cpu_continuation_comparison_missing_or_failed"),
        ("wrong_artifact", "artifact_sha256_not_present"),
        ("full_g200_claim", "unsupported_full_uninterrupted_execution_claim"),
        ("missing_route_receipt", "missing_route_receipt_evidence"),
        ("tampered_preflight_hash", "evidence_sha256_mismatch"),
    ],
)
def test_authority_bundle_fails_closed_for_required_negative_cases(
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    backend = "cpu_native" if "cpu" in case and case != "missing_gpu_resource" else "cuda_resident"
    bundle = _bundle(tmp_path, backend=backend)

    if case == "stale_source":
        bundle["target"]["source_commit"] = "stale-source"
    elif case == "wrong_route_hash":
        bundle["target"]["execution_route_sha256"] = "wrong-route"
        bundle["expected"]["execution_route_sha256"] = "wrong-route"
    elif case == "wrong_plan_hash":
        bundle["expected"]["plan_sha256"] = "wrong-plan"
    elif case == "t80_receipt_applied_to_t160":
        bundle["target"]["ticks_per_generation"] = 160
    elif case == "cpu_receipt_applied_to_gpu":
        bundle = _bundle(tmp_path, backend="cpu_native")
        bundle["target"] = _target(backend="cuda_resident")
        bundle["expected"].update(
            {
                "backend": "cuda_resident",
                "device": "cuda",
                "device_owner": "cuda_resident_no_fallback",
                "execution_route_sha256": GPU_ROUTE,
                "qf_record_id": "w32-c256-t80-cuda",
                "runtime_id": "eiel-production-runtime-v1",
            }
        )
    elif case == "missing_obligation":
        preflight = json.loads(Path(bundle["evidence_files"]["preflight"]["path"]).read_text(encoding="utf-8"))
        preflight["execution_route"]["covered_obligations"] = preflight["execution_route"]["covered_obligations"][:-1]
        _replace_evidence_payload(bundle, "preflight", preflight)
    elif case == "preflight_not_ready":
        preflight = json.loads(Path(bundle["evidence_files"]["preflight"]["path"]).read_text(encoding="utf-8"))
        preflight["ready_for_execution"] = False
        _replace_evidence_payload(bundle, "preflight", preflight)
    elif case == "missing_gpu_resource":
        preflight = json.loads(Path(bundle["evidence_files"]["preflight"]["path"]).read_text(encoding="utf-8"))
        preflight["resources"]["cuda"]["free_covers_admission_bound_and_headroom"] = False
        _replace_evidence_payload(bundle, "preflight", preflight)
    elif case == "cpu_continuation_mismatch":
        comparison = json.loads(Path(bundle["evidence_files"]["comparison_summary"]["path"]).read_text(encoding="utf-8"))
        comparison["comparisons"][2]["comparison_match"] = False
        _replace_evidence_payload(bundle, "comparison_summary", comparison)
    elif case == "wrong_artifact":
        preflight = json.loads(Path(bundle["evidence_files"]["preflight"]["path"]).read_text(encoding="utf-8"))
        preflight["artifacts"][0]["actual_sha256"] = "wrong-artifact"
        _replace_evidence_payload(bundle, "preflight", preflight)
    elif case == "full_g200_claim":
        bundle["expected"]["claim_full_uninterrupted_execution"] = True
    elif case == "missing_route_receipt":
        del bundle["evidence_files"]["route_receipt"]
    elif case == "tampered_preflight_hash":
        preflight_path = Path(bundle["evidence_files"]["preflight"]["path"])
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["ready_for_execution"] = False
        _write_json(preflight_path, preflight)

    result = validate_authority_bundle(bundle, catalog=_catalog())

    assert result["decision"] == FULL_QUALIFICATION_REQUIRED
    assert expected_code in _codes(result)
