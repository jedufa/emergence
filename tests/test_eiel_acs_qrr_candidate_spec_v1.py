from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.scripts import eiel_acs_qrr_candidate_spec_v1 as candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _candidate_receipt(tmp_path: Path, *, m0_enabled: bool = True) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "candidate.so"
    manifest = tmp_path / "candidate.so.manifest.json"
    artifact.write_bytes(b"candidate native artifact")
    compile_definition = f"-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED={int(m0_enabled)}"
    resource_receipt = {"option_d_m0_telemetry_enabled": m0_enabled}
    identity_payload = {
        "schema_version": "eiel_option_d_m0_telemetry_build_identity_v1",
        "compile_definition": compile_definition,
        "compile_time_macro": "EIEL_OPTION_D_M0_TELEMETRY_ENABLED",
        "default_enabled": False,
        "identity_authority": "compiled_binary_resource_receipt",
        "option_d_m0_telemetry_enabled": m0_enabled,
        "receipt_field": "option_d_m0_telemetry_enabled",
        "runtime_resource_receipt_confirmed": True,
        "runtime_resource_receipt_sha256": candidate._canonical_sha256(
            resource_receipt
        ),
    }
    identity = {
        **identity_payload,
        "identity_sha256": candidate._canonical_sha256(identity_payload),
    }
    _write_json(
        manifest,
        {
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            "option_d_m0_telemetry_build": identity,
            "artifact_identity": {"option_d_m0_telemetry_build": identity},
            "resource_receipt": resource_receipt,
            "cxx_flags": [compile_definition],
            "nvcc_flags": [compile_definition],
        },
    )
    artifact_sha256 = _sha256(artifact)
    manifest_sha256 = _sha256(manifest)
    qualification = tmp_path / "qualification.json"
    reporting = tmp_path / "reporting.json"
    _write_json(
        qualification,
        {
            "production_authority": False,
            "artifact_abi": "test-abi",
            "artifact": {
                "path": str(artifact),
                "sha256": artifact_sha256,
                "manifest_path": str(manifest),
                "manifest_sha256": manifest_sha256,
            },
        },
    )
    _write_json(
        reporting,
        {
            "production_authority": False,
            "artifact_sha256": artifact_sha256,
        },
    )
    receipt = tmp_path / "publication.json"
    _write_json(
        receipt,
        {
            "ok": True,
            "production_authority": False,
            "artifact_abi": "test-abi",
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": manifest_sha256,
            "qualification": {
                "path": str(qualification),
                "sha256": _sha256(qualification),
            },
            "reporting": {"path": str(reporting), "sha256": _sha256(reporting)},
        },
    )
    return receipt


def _genome_v2_package(tmp_path: Path, candidate_receipt: Path) -> Path:
    receipt = json.loads(candidate_receipt.read_text(encoding="utf-8"))
    artifact_sha256 = receipt["artifact_sha256"]
    package = tmp_path / "genome-package"
    contracts = package / "contracts"
    contracts.mkdir(parents=True)
    for _artifact_id, relative_path in candidate.GENOME_V2_PACKAGE_ARTIFACTS:
        target = package / relative_path
        if target.name == "genome_v2_package_identity_receipt.json":
            _write_json(
                target,
                {
                    "schema_version": "genome_v2_package_identity_receipt_v1",
                    "ok": True,
                    "authority_state": "package_candidate_complete",
                    "identity": {
                        "cuda_artifact_sha256": artifact_sha256,
                        "loaded_artifact_sha256": artifact_sha256,
                    },
                },
            )
        else:
            _write_json(target, {"artifact_id": _artifact_id})
    return package


def _base_spec(tmp_path: Path) -> Path:
    source = Path(
        "/tmp/emergence-test/runtime/qualified_runtime_releases_v1/"
        "qrr1-d4464e8284a172a10e50ed16c86d89b0/payload/spec/base_spec.yaml"
    )
    target = tmp_path / "base_spec.json"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _source_environment_receipt(tmp_path: Path) -> Path:
    path = tmp_path / "option_d_m0_source_environment_receipt.json"
    _write_json(path, {"schema_version": candidate.OPTION_D_M0_SOURCE_ENVIRONMENT_ABI})
    return path


def test_materializes_default_off_acs0_candidate_spec(tmp_path: Path) -> None:
    candidate_receipt = _candidate_receipt(tmp_path)
    result = candidate.materialize_acs_candidate_spec_v1(
        base_spec_path=_base_spec(tmp_path),
        candidate_receipt_path=candidate_receipt,
        genome_v2_package_dir=_genome_v2_package(tmp_path, candidate_receipt),
        source_environment_receipt_path=_source_environment_receipt(tmp_path),
        runtime_root=tmp_path / "runtime",
        campaign_id="acs0-route-canary",
    )

    assert result["campaign_id"] == "acs0-route-canary"
    assert result["checkpoints"]["acs_slots"] == 0
    assert result["checkpoints"]["acs_capture_instrumentation"] is True
    assert result["checkpoints"]["native_streamed_checkpoint"] is True
    assert result["artifacts"]["cuda_extension"]["kind"] == "cuda_extension"
    assert result["artifacts"][candidate.OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT] == {
        "kind": candidate.OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT,
        "abi": candidate.OPTION_D_M0_SOURCE_ENVIRONMENT_ABI,
        "path": str(_source_environment_receipt(tmp_path).resolve()),
        "sha256": _sha256(_source_environment_receipt(tmp_path)),
    }
    assert (
        result["artifacts"]["qualification_receipt"]["kind"] == "qualification_receipt"
    )
    assert result["artifacts"]["genome_upgrade_v2_package_receipt"]["path"].endswith(
        "genome_v2_package_identity_receipt.json"
    )
    assert result["output"]["runtime_root"] == str((tmp_path / "runtime").resolve())


def test_materializes_m0_off_single_slot_candidate_without_m0_receipt(
    tmp_path: Path,
) -> None:
    candidate_receipt = _candidate_receipt(tmp_path, m0_enabled=False)
    result = candidate.materialize_acs_candidate_spec_v1(
        base_spec_path=_base_spec(tmp_path),
        candidate_receipt_path=candidate_receipt,
        genome_v2_package_dir=_genome_v2_package(tmp_path, candidate_receipt),
        source_environment_receipt_path=None,
        runtime_root=tmp_path / "runtime",
        campaign_id="acs2-route-canary",
        acs_slots=1,
    )

    assert result["checkpoints"]["acs_slots"] == 1
    assert result["checkpoints"]["acs_capture_instrumentation"] is True
    assert result["checkpoints"]["native_streamed_checkpoint"] is True
    assert candidate.OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT not in result["artifacts"]


def test_rejects_m0_mode_and_source_receipt_mismatch(tmp_path: Path) -> None:
    off_receipt = _candidate_receipt(tmp_path, m0_enabled=False)
    package = _genome_v2_package(tmp_path, off_receipt)
    with pytest.raises(candidate.ACSCandidateSpecError, match="M0-OFF artifact forbids"):
        candidate.materialize_acs_candidate_spec_v1(
            base_spec_path=_base_spec(tmp_path),
            candidate_receipt_path=off_receipt,
            genome_v2_package_dir=package,
            source_environment_receipt_path=_source_environment_receipt(tmp_path),
            runtime_root=tmp_path / "runtime",
            campaign_id="acs2-route-canary",
            acs_slots=1,
        )


@pytest.mark.parametrize("field", ["cxx_flags", "nvcc_flags", "resource_receipt"])
def test_rejects_m0_compiled_identity_mismatch(tmp_path: Path, field: str) -> None:
    receipt = _candidate_receipt(tmp_path, m0_enabled=False)
    package = _genome_v2_package(tmp_path, receipt)
    publication = json.loads(receipt.read_text(encoding="utf-8"))
    qualification_path = Path(publication["qualification"]["path"])
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    manifest_path = Path(qualification["artifact"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "resource_receipt":
        manifest[field]["option_d_m0_telemetry_enabled"] = True
    else:
        manifest[field] = []
    _write_json(manifest_path, manifest)
    qualification["artifact"]["manifest_sha256"] = _sha256(manifest_path)
    _write_json(qualification_path, qualification)
    publication["manifest_sha256"] = _sha256(manifest_path)
    publication["qualification"]["sha256"] = _sha256(qualification_path)
    _write_json(receipt, publication)

    with pytest.raises(candidate.ACSCandidateSpecError, match="M0"):
        candidate.materialize_acs_candidate_spec_v1(
            base_spec_path=_base_spec(tmp_path),
            candidate_receipt_path=receipt,
            genome_v2_package_dir=package,
            source_environment_receipt_path=None,
            runtime_root=tmp_path / "runtime",
            campaign_id="acs2-route-canary",
            acs_slots=1,
        )


def test_rejects_invalid_slots_and_m0_on_without_receipt(tmp_path: Path) -> None:
    receipt = _candidate_receipt(tmp_path)
    package = _genome_v2_package(tmp_path, receipt)
    with pytest.raises(candidate.ACSCandidateSpecError, match="requires"):
        candidate.materialize_acs_candidate_spec_v1(
            base_spec_path=_base_spec(tmp_path),
            candidate_receipt_path=receipt,
            genome_v2_package_dir=package,
            source_environment_receipt_path=None,
            runtime_root=tmp_path / "runtime",
            campaign_id="acs0-route-canary",
        )
    with pytest.raises(candidate.ACSCandidateSpecError, match="slots"):
        candidate.materialize_acs_candidate_spec_v1(
            base_spec_path=_base_spec(tmp_path),
            candidate_receipt_path=receipt,
            genome_v2_package_dir=package,
            source_environment_receipt_path=_source_environment_receipt(tmp_path),
            runtime_root=tmp_path / "runtime",
            campaign_id="acs0-route-canary",
            acs_slots=True,
        )

    on_receipt = _candidate_receipt(tmp_path / "on", m0_enabled=True)
    on_package = _genome_v2_package(tmp_path / "on", on_receipt)
    with pytest.raises(candidate.ACSCandidateSpecError, match="cannot materialize"):
        candidate.materialize_acs_candidate_spec_v1(
            base_spec_path=_base_spec(tmp_path),
            candidate_receipt_path=on_receipt,
            genome_v2_package_dir=on_package,
            source_environment_receipt_path=_source_environment_receipt(tmp_path),
            runtime_root=tmp_path / "runtime-on",
            campaign_id="acs2-route-canary-on",
            acs_slots=1,
        )


def test_rejects_production_candidate_receipt(tmp_path: Path) -> None:
    receipt = _candidate_receipt(tmp_path)
    package = _genome_v2_package(tmp_path, receipt)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["production_authority"] = True
    _write_json(receipt, value)

    with pytest.raises(candidate.ACSCandidateSpecError, match="non-production"):
        candidate.materialize_acs_candidate_spec_v1(
            base_spec_path=_base_spec(tmp_path),
            candidate_receipt_path=receipt,
            genome_v2_package_dir=package,
            source_environment_receipt_path=_source_environment_receipt(tmp_path),
            runtime_root=tmp_path / "runtime",
            campaign_id="acs0-route-canary",
        )


def test_rejects_package_for_different_cuda_artifact(tmp_path: Path) -> None:
    receipt = _candidate_receipt(tmp_path)
    package = _genome_v2_package(tmp_path, receipt)
    receipt_path = package / "genome_v2_package_identity_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["identity"]["cuda_artifact_sha256"] = "0" * 64
    _write_json(receipt_path, payload)

    with pytest.raises(
        candidate.ACSCandidateSpecError, match="candidate CUDA artifact"
    ):
        candidate.materialize_acs_candidate_spec_v1(
            base_spec_path=_base_spec(tmp_path),
            candidate_receipt_path=receipt,
            genome_v2_package_dir=package,
            source_environment_receipt_path=_source_environment_receipt(tmp_path),
            runtime_root=tmp_path / "runtime",
            campaign_id="acs0-route-canary",
        )


def test_refuses_to_overwrite_candidate_spec(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(candidate.ACSCandidateSpecError, match="overwrite"):
        candidate.write_new_candidate_spec_v1(output, {"ok": True})
