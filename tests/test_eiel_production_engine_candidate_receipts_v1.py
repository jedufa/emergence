import hashlib
import json
from pathlib import Path

import pytest

from engine.campaigns.production_engine_identity import (
    PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION,
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
    PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION,
)
from tools.scripts import eiel_production_engine_candidate_receipts_v1 as receipts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity() -> dict[str, object]:
    return {
        "schema_version": "eiel_repository_source_identity_v1",
        "git_head": "a" * 40,
        "worktree_clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }


def test_candidate_receipts_bind_one_engine_and_deny_production_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "engine.so"
    artifact.write_bytes(b"candidate")
    manifest = tmp_path / "engine.so.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": _sha256(artifact),
                "module_name": "eiel_test_engine",
                "entrypoints": [
                    "eiel_cuda_architecture_b_execute_direct_resident_windows_v1"
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(receipts, "_repository_identity", _source_identity)

    result = receipts.publish(
        artifact=artifact,
        manifest=manifest,
        binding_id="patch_fixed_mlp_native_binding_v1",
        artifact_abi="architecture_b_patch_p2_stage3_v2",
        output_dir=tmp_path / "receipts",
        evidence=[],
    )

    qualification = json.loads(
        Path(result["qualification"]["path"]).read_text(encoding="utf-8")
    )
    reporting = json.loads(
        Path(result["reporting"]["path"]).read_text(encoding="utf-8")
    )
    assert result["production_authority"] is False
    assert qualification["schema_version"] == (
        PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION
    )
    assert reporting["schema_version"] == (
        PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION
    )
    assert qualification["provider_id"] == reporting["provider_id"] == (
        PRODUCTION_PROVIDER_ID
    )
    assert qualification["profile_id"] == reporting["profile_id"] == (
        PRODUCTION_PROFILE_ID
    )
    assert qualification["production_authority"] is False
    assert reporting["production_authority"] is False


def test_candidate_receipts_reject_missing_binding_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "engine.so"
    artifact.write_bytes(b"candidate")
    manifest = tmp_path / "engine.so.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": _sha256(artifact),
                "module_name": "eiel_test_engine",
                "entrypoints": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(receipts, "_repository_identity", _source_identity)

    with pytest.raises(
        receipts.CandidateReceiptError,
        match="lacks the configured mechanics exports",
    ):
        receipts.publish(
            artifact=artifact,
            manifest=manifest,
            binding_id="patch_fixed_mlp_native_binding_v1",
            artifact_abi="architecture_b_patch_p2_stage3_v2",
            output_dir=tmp_path / "receipts",
            evidence=[],
        )


def test_eed_candidate_requires_execution_and_capability_exports() -> None:
    assert receipts.REQUIRED_EXPORTS["eed_fixed_mlp_native_binding_v1"] == (
        "eiel_cuda_architecture_b_execute_direct_resident_eed_v1",
        "eiel_cuda_architecture_b_flattened_eed_capability_v1",
    )


def test_wdl_candidate_requires_compact_history_v3_execution_export() -> None:
    assert receipts.REQUIRED_EXPORTS["wdl_fixed_mlp_native_binding_v1"] == (
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_v3",
    )


def test_integrated_b1_candidate_requires_direct_resident_windows_export() -> None:
    assert receipts.REQUIRED_EXPORTS[
        "wdl_jungle_eed_fixed_mlp_native_binding_v1"
    ] == ("eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_v1",)
