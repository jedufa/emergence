#!/usr/bin/env python3
"""Publish candidate receipts used only by unified-engine route qualification.

These receipts admit one source-bound binary to the bounded route canary. They
do not grant production authority. Production authority is established only by
the execution-route qualification sidecar published after the canary succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Final, Mapping

from engine.campaigns.production_engine_identity import (
    PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION,
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
    PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCHEMA_VERSION: Final[str] = (
    "eiel_production_engine_candidate_receipt_publication_v1"
)
REQUIRED_EXPORTS: Final[dict[str, tuple[str, ...]]] = {
    "eed_fixed_mlp_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_eed_v1",
        "eiel_cuda_architecture_b_flattened_eed_capability_v1",
    ),
    "patch_fixed_mlp_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_windows_v1",
    ),
    "wdl_fixed_mlp_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_v3",
    ),
    "jungle_fixed_mlp_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_jungle_v4",
    ),
    "wdl_jungle_eed_fixed_mlp_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_v1",
    ),
    "patch_stable_gru_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_gru_v3",
    ),
    "wdl_stable_gru_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_gru_v3",
    ),
    "gmnw_policy_arms_native_binding_v1": (
        "eiel_cuda_architecture_b_execute_direct_resident_gmnw_gru_v1",
    ),
}


class CandidateReceiptError(RuntimeError):
    """A candidate binary cannot be admitted to route qualification."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateReceiptError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateReceiptError(f"{label} must be a JSON object: {path}")
    return value


def _repository_identity() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CandidateReceiptError(
            f"cannot bind repository identity: {exc}"
        ) from exc
    if status:
        raise CandidateReceiptError(
            "candidate receipts require a clean committed worktree"
        )
    return {
        "schema_version": "eiel_repository_source_identity_v1",
        "git_head": head,
        "worktree_clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _evidence_rows(values: list[str]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if (
            not separator
            or not name
            or name != name.strip()
            or name in rows
        ):
            raise CandidateReceiptError(
                f"evidence must be a unique NAME=PATH binding: {value!r}"
            )
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise CandidateReceiptError(
                f"evidence is not a regular file: {path}"
            )
        rows[name] = {"path": str(path), "sha256": _sha256(path)}
    return dict(sorted(rows.items()))


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CandidateReceiptError(f"refusing to overwrite receipt: {path}")
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish(
    *,
    artifact: Path,
    manifest: Path,
    binding_id: str,
    artifact_abi: str,
    output_dir: Path,
    evidence: list[str],
) -> dict[str, Any]:
    artifact = artifact.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if binding_id not in REQUIRED_EXPORTS:
        raise CandidateReceiptError(f"unknown production binding: {binding_id}")
    if not artifact.is_file() or artifact.is_symlink():
        raise CandidateReceiptError(f"artifact is not a regular file: {artifact}")
    if not manifest.is_file() or manifest.is_symlink():
        raise CandidateReceiptError(f"manifest is not a regular file: {manifest}")
    manifest_payload = _object(manifest, label="artifact manifest")
    artifact_sha256 = _sha256(artifact)
    manifest_sha256 = _sha256(manifest)
    if (
        manifest_payload.get("artifact_sha256") != artifact_sha256
        or Path(str(manifest_payload.get("artifact", ""))).resolve() != artifact
    ):
        raise CandidateReceiptError(
            "artifact manifest does not bind the selected binary"
        )
    entrypoints = manifest_payload.get("entrypoints")
    required_exports = REQUIRED_EXPORTS[binding_id]
    if (
        not isinstance(entrypoints, list)
        or any(name not in entrypoints for name in required_exports)
    ):
        raise CandidateReceiptError(
            "artifact manifest lacks the configured mechanics exports"
        )
    module_name = manifest_payload.get("module_name")
    if not isinstance(module_name, str) or not module_name.isidentifier():
        raise CandidateReceiptError("artifact manifest module name is invalid")
    repository = _repository_identity()
    evidence_rows = _evidence_rows(evidence)
    output_dir.mkdir(parents=True, exist_ok=False)
    qualification_path = output_dir / "qualification.json"
    reporting_path = output_dir / "reporting.json"
    artifact_row = {
        "path": str(artifact),
        "sha256": artifact_sha256,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "module_name": module_name,
    }
    qualification = {
        "schema_version": PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION,
        "ok": True,
        "production_authority": False,
        "authority_scope": "bounded_execution_route_qualification_candidate_v1",
        "provider_id": PRODUCTION_PROVIDER_ID,
        "profile_id": PRODUCTION_PROFILE_ID,
        "mechanics_binding_id": binding_id,
        "artifact_abi": artifact_abi,
        "artifact": artifact_row,
        "required_exports": list(required_exports),
        "repository_source_identity": repository,
        "supporting_evidence": evidence_rows,
    }
    reporting = {
        "schema_version": PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION,
        "ok": True,
        "production_authority": False,
        "authority_scope": "bounded_execution_route_qualification_candidate_v1",
        "provider_id": PRODUCTION_PROVIDER_ID,
        "profile_id": PRODUCTION_PROFILE_ID,
        "mechanics_binding_id": binding_id,
        "artifact_sha256": artifact_sha256,
        "reporting_is_observational": True,
        "reporting_authority_source": (
            "execution_route_reporting_parity_obligation_v1"
        ),
        "repository_source_identity": repository,
        "supporting_evidence": evidence_rows,
    }
    _write_new(qualification_path, qualification)
    _write_new(reporting_path, reporting)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "production_authority": False,
        "provider_id": PRODUCTION_PROVIDER_ID,
        "profile_id": PRODUCTION_PROFILE_ID,
        "mechanics_binding_id": binding_id,
        "artifact_abi": artifact_abi,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "qualification": {
            "path": str(qualification_path),
            "sha256": _sha256(qualification_path),
        },
        "reporting": {
            "path": str(reporting_path),
            "sha256": _sha256(reporting_path),
        },
    }
    receipt_path = output_dir / "publication.json"
    _write_new(receipt_path, receipt)
    return {**receipt, "publication_path": str(receipt_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--binding-id", choices=tuple(sorted(REQUIRED_EXPORTS)), required=True
    )
    parser.add_argument("--artifact-abi", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence", action="append", default=[])
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        receipt = publish(
            artifact=arguments.artifact,
            manifest=arguments.manifest,
            binding_id=arguments.binding_id,
            artifact_abi=arguments.artifact_abi,
            output_dir=arguments.output_dir,
            evidence=arguments.evidence,
        )
    except CandidateReceiptError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
