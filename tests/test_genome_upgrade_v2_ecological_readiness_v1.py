from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.scripts import genome_upgrade_v2_ecological_readiness_v1 as readiness


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "Readiness Test")
    _git(repository, "config", "user.email", "readiness@example.invalid")
    (repository / "source.txt").write_text("governed source\n", encoding="ascii")
    _git(repository, "add", "source.txt")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return repository, _git(repository, "rev-parse", "HEAD")


def _artifact_fixture(
    tmp_path: Path,
    source_commit: str,
) -> tuple[Path, Path, dict[str, object]]:
    artifact = tmp_path / "artifact.so"
    artifact.write_bytes(b"immutable sass artifact")
    manifest = {
        "artifact": str(artifact.resolve()),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "committed_source_snapshot": {
            "build_source_commit": source_commit,
            "schema_version": "eiel_authoritative_state_committed_source_snapshot_v1",
        },
        "cuda_arch": "sm_75",
        "immutable_output": True,
        "runtime_jit_compilation": False,
        "sass_only": True,
    }
    manifest_path = tmp_path / "artifact.so.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return artifact, manifest_path, manifest


def _receipt(tmp_path: Path) -> dict[str, object]:
    repository, commit = _clean_repository(tmp_path)
    artifact, manifest, _ = _artifact_fixture(tmp_path, commit)
    return readiness.build_ecological_readiness_receipt(
        repository_root=repository,
        artifact_path=artifact,
        manifest_path=manifest,
    )


def test_receipt_binds_governed_authorities_without_claiming_e1(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)

    assert receipt["decision"] == "TECHNICAL_ADMISSION_ONLY_PASS"
    assert receipt["production_identity"] == {
        "profile_id": readiness.EXPECTED_PROFILE_ID,
        "provider_id": readiness.EXPECTED_PROVIDER_ID,
        "runtime_profile_id": readiness.EXPECTED_RUNTIME_PROFILE_ID,
    }
    assert receipt["capability_contract"]["contract_sha256"] == (
        readiness.GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
    )
    assert receipt["causal_c1_preregistration"]["preregistration_sha256"] == (
        readiness.PREREGISTRATION_SHA256
    )
    assert receipt["package_boundary"] == {
        "package_identity_bound": False,
        "execution_route_bound": False,
        "causal_c1_decision_bound": False,
        "e1_authorized": False,
        "evidence_role": "pre_package_technical_readiness",
    }
    assert receipt["current_authority"]["scientific_result"] is False
    assert receipt["claims"]["technical_readiness_only"] is True
    assert all(
        value is False
        for claim, value in receipt["claims"].items()
        if claim != "technical_readiness_only"
    )


def test_receipt_is_deterministic_and_self_hashes_payload(tmp_path: Path) -> None:
    repository, commit = _clean_repository(tmp_path)
    artifact, manifest, _ = _artifact_fixture(tmp_path, commit)

    first = readiness.build_ecological_readiness_receipt(
        repository_root=repository,
        artifact_path=artifact,
        manifest_path=manifest,
    )
    second = readiness.build_ecological_readiness_receipt(
        repository_root=repository,
        artifact_path=artifact,
        manifest_path=manifest,
    )

    assert first == second
    payload = dict(first)
    claimed_hash = payload.pop("receipt_sha256")
    assert claimed_hash == readiness._content_sha256(payload)


def test_receipt_rejects_dirty_source_repository(tmp_path: Path) -> None:
    repository, commit = _clean_repository(tmp_path)
    artifact, manifest, _ = _artifact_fixture(tmp_path, commit)
    (repository / "source.txt").write_text("dirty source\n", encoding="ascii")

    with pytest.raises(readiness.EcologicalReadinessError, match="repository is dirty"):
        readiness.build_ecological_readiness_receipt(
            repository_root=repository,
            artifact_path=artifact,
            manifest_path=manifest,
        )


@pytest.mark.parametrize(
    ("field", "unsupported"),
    [
        ("artifact", "/tmp/different-artifact.so"),
        ("artifact_sha256", "0" * 64),
        ("immutable_output", False),
        ("sass_only", False),
        ("runtime_jit_compilation", True),
        ("cuda_arch", "sm_86"),
    ],
)
def test_receipt_rejects_stale_or_unsupported_manifest(
    tmp_path: Path,
    field: str,
    unsupported: object,
) -> None:
    repository, commit = _clean_repository(tmp_path)
    artifact, manifest_path, manifest = _artifact_fixture(tmp_path, commit)
    manifest[field] = unsupported
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    with pytest.raises(readiness.EcologicalReadinessError, match="artifact manifest"):
        readiness.build_ecological_readiness_receipt(
            repository_root=repository,
            artifact_path=artifact,
            manifest_path=manifest_path,
        )


def test_receipt_rejects_manifest_from_another_source_commit(
    tmp_path: Path,
) -> None:
    repository, commit = _clean_repository(tmp_path)
    artifact, manifest_path, manifest = _artifact_fixture(tmp_path, commit)
    snapshot = manifest["committed_source_snapshot"]
    assert isinstance(snapshot, dict)
    snapshot["build_source_commit"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")

    with pytest.raises(readiness.EcologicalReadinessError, match="build_source_commit"):
        readiness.build_ecological_readiness_receipt(
            repository_root=repository,
            artifact_path=artifact,
            manifest_path=manifest_path,
        )


def test_receipt_rejects_unsupported_production_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, commit = _clean_repository(tmp_path)
    artifact, manifest, _ = _artifact_fixture(tmp_path, commit)
    monkeypatch.setattr(readiness, "PROVIDER_ID", "legacy-provider")

    with pytest.raises(
        readiness.EcologicalReadinessError,
        match="unsupported production identity",
    ):
        readiness.build_ecological_readiness_receipt(
            repository_root=repository,
            artifact_path=artifact,
            manifest_path=manifest,
        )
