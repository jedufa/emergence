#!/usr/bin/env python3
"""Generate the governed Genome V2 ecological technical-readiness receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
    PROFILE_ID,
    PROVIDER_ID,
    RUNTIME_PROFILE_ID,
    SCHEMA_VERSION as CAPABILITY_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    PREREGISTRATION_SHA256,
    SCHEMA_VERSION as CAUSAL_GATE_SCHEMA_VERSION,
)


SCHEMA_VERSION: Final[str] = "genome_v2_ecological_probe_readiness_v2"
EXPECTED_PROFILE_ID: Final[str] = "genome_upgrade_v2_fixed_q10_v1"
EXPECTED_PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
EXPECTED_RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
EXPECTED_CUDA_ARCH: Final[str] = "sm_75"


class EcologicalReadinessError(ValueError):
    """Raised when technical-readiness authority cannot be established."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise EcologicalReadinessError(
            f"{label} must be an existing regular non-symlink file"
        )
    return path.resolve()


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EcologicalReadinessError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def _clean_source_commit(repository: Path) -> str:
    if not repository.is_dir():
        raise EcologicalReadinessError("repository_root must be an existing directory")
    top_level = Path(
        _run_git(repository, "rev-parse", "--show-toplevel")
    ).resolve()
    status = _run_git(
        top_level,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise EcologicalReadinessError(
            "source repository is dirty; readiness requires an exact clean commit"
        )
    commit = _run_git(top_level, "rev-parse", "--verify", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise EcologicalReadinessError("source HEAD is not a canonical 40-character commit")
    return commit


def _validate_canonical_identities() -> None:
    identities = {
        "profile_id": (PROFILE_ID, EXPECTED_PROFILE_ID),
        "provider_id": (PROVIDER_ID, EXPECTED_PROVIDER_ID),
        "runtime_profile_id": (
            RUNTIME_PROFILE_ID,
            EXPECTED_RUNTIME_PROFILE_ID,
        ),
    }
    unsupported = [
        f"{name}={actual!r}"
        for name, (actual, expected) in identities.items()
        if actual != expected
    ]
    if unsupported:
        raise EcologicalReadinessError(
            "unsupported production identity: " + ", ".join(unsupported)
        )


def _load_manifest(manifest_path: Path) -> tuple[dict[str, Any], bytes]:
    manifest_bytes = manifest_path.read_bytes()
    try:
        value = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EcologicalReadinessError("artifact manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise EcologicalReadinessError("artifact manifest must be a JSON object")
    return value, manifest_bytes


def _require_manifest_value(
    manifest: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    if manifest.get(key) != expected:
        raise EcologicalReadinessError(
            f"artifact manifest {key!r} does not match required value {expected!r}"
        )


def build_ecological_readiness_receipt(
    *,
    repository_root: Path,
    artifact_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Build a technical-only receipt after validating every bound authority."""

    _validate_canonical_identities()
    source_commit = _clean_source_commit(repository_root.resolve())
    artifact = _require_regular_file(artifact_path, "artifact")
    manifest_file = _require_regular_file(manifest_path, "artifact manifest")
    manifest, manifest_bytes = _load_manifest(manifest_file)

    artifact_sha256 = _sha256_bytes(artifact.read_bytes())
    _require_manifest_value(manifest, "artifact", str(artifact))
    _require_manifest_value(manifest, "artifact_sha256", artifact_sha256)
    _require_manifest_value(manifest, "immutable_output", True)
    _require_manifest_value(manifest, "sass_only", True)
    _require_manifest_value(manifest, "runtime_jit_compilation", False)
    _require_manifest_value(manifest, "cuda_arch", EXPECTED_CUDA_ARCH)

    snapshot = manifest.get("committed_source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise EcologicalReadinessError(
            "artifact manifest lacks committed_source_snapshot"
        )
    _require_manifest_value(snapshot, "build_source_commit", source_commit)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "TECHNICAL_ADMISSION_ONLY_PASS",
        "source_commit": source_commit,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "supported_campaign_class": "bounded_ecological",
        "production_identity": {
            "profile_id": PROFILE_ID,
            "provider_id": PROVIDER_ID,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
        },
        "capability_contract": {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "contract_sha256": GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
        },
        "causal_c1_preregistration": {
            "schema_version": CAUSAL_GATE_SCHEMA_VERSION,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        },
        "artifact_contract": {
            "cuda_arch": EXPECTED_CUDA_ARCH,
            "immutable_output": True,
            "runtime_jit_compilation": False,
            "sass_only": True,
        },
        "preconditions": {
            "package_qualification": "required",
            "route_qualification_T3": "required_before_decision_bearing_execution",
            "causal_gate_C1": "required_before_E1",
            "immutable_preregistration": "required",
            "matched_controls_and_multiple_seeds": "required",
        },
        "package_boundary": {
            "package_identity_bound": False,
            "execution_route_bound": False,
            "causal_c1_decision_bound": False,
            "e1_authorized": False,
            "evidence_role": "pre_package_technical_readiness",
        },
        "current_authority": {
            "decision_bearing_ecological_execution": False,
            "scientific_result": False,
        },
        "claims": {
            "technical_readiness_only": True,
            "ecological_usefulness": False,
            "intelligence": False,
            "cognition": False,
            "emergence": False,
            "consciousness": False,
        },
    }
    return {**payload, "receipt_sha256": _content_sha256(payload)}


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="ascii",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = build_ecological_readiness_receipt(
        repository_root=arguments.repository_root,
        artifact_path=arguments.artifact,
        manifest_path=arguments.manifest,
    )
    _write_receipt(arguments.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
