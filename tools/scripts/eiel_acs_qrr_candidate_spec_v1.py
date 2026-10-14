#!/usr/bin/env python3
"""Materialize one ACS route-qualification candidate campaign specification.

The candidate is intentionally not a production release.  It binds a clean,
current-source native artifact and its bounded route-qualification candidate
receipts to the frozen QRR base specification, then selects the default-off
ACS capture/staging route.  A route canary must still succeed before
the resulting specification can contribute to a successor QRR.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine"
for candidate in (ROOT, ENGINE):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from engine.campaigns.compiler import load_campaign_spec, resolve_campaign_profile  # noqa: E402
from engine.campaigns.wrc_science import materialize_full_stack_wrc_science_spec_v1  # noqa: E402


SCHEMA_VERSION = "eiel_acs_qrr_candidate_spec_v1"
OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT = "option_d_m0_source_environment_receipt"
OPTION_D_M0_SOURCE_ENVIRONMENT_ABI = "eiel_option_d_m0_source_rebind_v3"
ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT = "acs_native_runtime_extension"
ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT = "acs_native_runtime_extension_manifest"
ACS_NATIVE_RUNTIME_ABI = "eiel_native_runtime_abi_v1"
ACS_NATIVE_RUNTIME_VERSION = "eiel_native_runtime_flat_v1"
ACS_NATIVE_RUNTIME_MODULE_NAME = "eiel_native_runtime_ext"

GENOME_V2_PACKAGE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (
        "genome_upgrade_v2_executable_contract",
        "contracts/genome_v2_executable_contract.json",
    ),
    (
        "genome_upgrade_v2_capability_contract",
        "contracts/genome_v2_p10_capability_contract.json",
    ),
    (
        "genome_upgrade_v2_package_contract",
        "contracts/genome_v2_executable_package_contract_v1.json",
    ),
    (
        "genome_upgrade_v2_obligation_dictionary",
        "contracts/genome_v2_package_obligation_dictionary.json",
    ),
    (
        "genome_upgrade_v2_invalidation_rules",
        "contracts/genome_v2_package_invalidation_rules.json",
    ),
    (
        "genome_upgrade_v2_chronicle_schema",
        "contracts/genome_v2_chronicle_schema.json",
    ),
    (
        "genome_upgrade_v2_reporting_schema",
        "contracts/genome_v2_reporting_schema.json",
    ),
    (
        "genome_upgrade_v2_observability_storage_budget",
        "contracts/genome_v2_observability_storage_budget.json",
    ),
    ("genome_upgrade_v2_package_receipt", "genome_v2_package_identity_receipt.json"),
)


class ACSCandidateSpecError(ValueError):
    """The requested ACS route-qualification candidate is not trustworthy."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ACSCandidateSpecError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ACSCandidateSpecError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _regular_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ACSCandidateSpecError(f"{label} is not a regular file: {resolved}")
    return resolved


def _artifact_row(*, kind: str, abi: str, path: Path, sha256: str) -> dict[str, str]:
    return {
        "kind": kind,
        "abi": abi,
        "path": str(path),
        "sha256": sha256,
    }


def _native_runtime_artifacts(
    extension_path: Path,
) -> dict[str, dict[str, str]]:
    """Bind the native ACS pool extension and its adjacent build manifest."""

    extension = _regular_file(
        extension_path, label="native ACS runtime extension"
    )
    manifest = _regular_file(
        extension.with_suffix(extension.suffix + ".manifest.json"),
        label="native ACS runtime build manifest",
    )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ACSCandidateSpecError(
            f"cannot read native ACS runtime build manifest: {exc}"
        ) from exc
    extension_sha256 = _sha256(extension)
    manifest_sha256 = _sha256(manifest)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != "eiel_native_runtime_build_manifest_v1"
        or payload.get("stable_module_name") != ACS_NATIVE_RUNTIME_MODULE_NAME
        or payload.get("extension_abi_version") != ACS_NATIVE_RUNTIME_ABI
        or payload.get("extension_version") != ACS_NATIVE_RUNTIME_VERSION
        or payload.get("output_sha256") != extension_sha256
        or payload.get("runtime_jit_in_measured_execution") is not False
    ):
        raise ACSCandidateSpecError(
            "native ACS runtime build manifest identity is invalid"
        )
    return {
        ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT: _artifact_row(
            kind="native_runtime_extension",
            abi=ACS_NATIVE_RUNTIME_ABI,
            path=extension,
            sha256=extension_sha256,
        ),
        ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT: _artifact_row(
            kind="native_runtime_extension_manifest",
            abi=ACS_NATIVE_RUNTIME_ABI,
            path=manifest,
            sha256=manifest_sha256,
        ),
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validated_m0_build_mode(manifest: Mapping[str, Any]) -> bool:
    identity = manifest.get("option_d_m0_telemetry_build")
    artifact_identity = manifest.get("artifact_identity")
    resource_receipt = manifest.get("resource_receipt")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(artifact_identity, Mapping)
        or not isinstance(resource_receipt, Mapping)
        or artifact_identity.get("option_d_m0_telemetry_build") != identity
    ):
        raise ACSCandidateSpecError("candidate artifact M0 build identity is incomplete")
    enabled = identity.get("option_d_m0_telemetry_enabled")
    expected_definition = (
        "-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED=1"
        if enabled is True
        else "-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED=0"
        if enabled is False
        else None
    )
    payload = dict(identity)
    identity_sha256 = payload.pop("identity_sha256", None)
    if (
        expected_definition is None
        or identity.get("schema_version")
        != "eiel_option_d_m0_telemetry_build_identity_v1"
        or identity.get("compile_time_macro") != "EIEL_OPTION_D_M0_TELEMETRY_ENABLED"
        or identity.get("compile_definition") != expected_definition
        or identity.get("receipt_field") != "option_d_m0_telemetry_enabled"
        or identity.get("default_enabled") is not False
        or identity.get("identity_authority") != "compiled_binary_resource_receipt"
        or identity.get("runtime_resource_receipt_confirmed") is not True
        or resource_receipt.get("option_d_m0_telemetry_enabled") is not enabled
        or identity.get("runtime_resource_receipt_sha256")
        != _canonical_sha256(resource_receipt)
        or identity_sha256 != _canonical_sha256(payload)
    ):
        raise ACSCandidateSpecError("candidate artifact M0 build identity differs")
    for field in ("cxx_flags", "nvcc_flags"):
        flags = manifest.get(field)
        if not isinstance(flags, list) or flags.count(expected_definition) != 1:
            raise ACSCandidateSpecError(
                f"candidate artifact {field} does not bind the exact M0 mode"
            )
    return enabled


def _candidate_artifacts(receipt_path: Path) -> dict[str, dict[str, str]]:
    receipt = _read_object(receipt_path, label="candidate-receipt publication")
    if (
        receipt.get("ok") is not True
        or receipt.get("production_authority") is not False
    ):
        raise ACSCandidateSpecError(
            "candidate receipt must be successful and explicitly non-production"
        )
    abi = receipt.get("artifact_abi")
    artifact_sha256 = receipt.get("artifact_sha256")
    manifest_sha256 = receipt.get("manifest_sha256")
    if not (
        isinstance(abi, str)
        and abi
        and isinstance(artifact_sha256, str)
        and len(artifact_sha256) == 64
        and isinstance(manifest_sha256, str)
        and len(manifest_sha256) == 64
    ):
        raise ACSCandidateSpecError("candidate receipt artifact identity is malformed")

    qualification = receipt.get("qualification")
    reporting = receipt.get("reporting")
    if not isinstance(qualification, Mapping) or not isinstance(reporting, Mapping):
        raise ACSCandidateSpecError(
            "candidate receipt lacks qualification/reporting rows"
        )
    qualification_path = _regular_file(
        Path(str(qualification.get("path", ""))), label="candidate qualification"
    )
    reporting_path = _regular_file(
        Path(str(reporting.get("path", ""))), label="candidate reporting"
    )
    if qualification.get("sha256") != _sha256(qualification_path):
        raise ACSCandidateSpecError("candidate qualification SHA-256 differs")
    if reporting.get("sha256") != _sha256(reporting_path):
        raise ACSCandidateSpecError("candidate reporting SHA-256 differs")

    qualification_payload = _read_object(
        qualification_path, label="candidate qualification"
    )
    reporting_payload = _read_object(reporting_path, label="candidate reporting")
    artifact = _regular_file(
        Path(str(qualification_payload.get("artifact", {}).get("path", ""))),
        label="candidate CUDA artifact",
    )
    manifest = _regular_file(
        Path(str(qualification_payload.get("artifact", {}).get("manifest_path", ""))),
        label="candidate artifact manifest",
    )
    if (
        qualification_payload.get("production_authority") is not False
        or qualification_payload.get("artifact_abi") != abi
        or qualification_payload.get("artifact", {}).get("sha256") != artifact_sha256
        or qualification_payload.get("artifact", {}).get("manifest_sha256")
        != manifest_sha256
        or _sha256(artifact) != artifact_sha256
        or _sha256(manifest) != manifest_sha256
        or reporting_payload.get("production_authority") is not False
        or reporting_payload.get("artifact_sha256") != artifact_sha256
    ):
        raise ACSCandidateSpecError("candidate receipt closure differs")
    return {
        "cuda_extension": _artifact_row(
            kind="cuda_extension", abi=abi, path=artifact, sha256=artifact_sha256
        ),
        "extension_manifest": _artifact_row(
            kind="extension_manifest", abi=abi, path=manifest, sha256=manifest_sha256
        ),
        "qualification_receipt": _artifact_row(
            kind="qualification_receipt",
            abi=abi,
            path=qualification_path,
            sha256=str(qualification["sha256"]),
        ),
        "reporting_receipt": _artifact_row(
            kind="reporting_receipt",
            abi=abi,
            path=reporting_path,
            sha256=str(reporting["sha256"]),
        ),
    }


def _genome_v2_package_artifacts(
    package_dir: Path,
    *,
    base_artifacts: Mapping[str, Any],
    artifact_sha256: str,
) -> dict[str, dict[str, str]]:
    """Return the exact Genome V2 package rows bound to the candidate binary."""

    root = package_dir.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ACSCandidateSpecError(
            f"Genome V2 package directory is not a real directory: {root}"
        )
    receipt_path = _regular_file(
        root / "genome_v2_package_identity_receipt.json",
        label="Genome V2 package receipt",
    )
    receipt = _read_object(receipt_path, label="Genome V2 package receipt")
    identity = receipt.get("identity")
    if (
        receipt.get("schema_version") != "genome_v2_package_identity_receipt_v1"
        or receipt.get("ok") is not True
        or receipt.get("authority_state") != "package_candidate_complete"
        or not isinstance(identity, Mapping)
        or identity.get("cuda_artifact_sha256") != artifact_sha256
        or identity.get("loaded_artifact_sha256") != artifact_sha256
    ):
        raise ACSCandidateSpecError(
            "Genome V2 package receipt is not bound to the candidate CUDA artifact"
        )

    rows: dict[str, dict[str, str]] = {}
    for artifact_id, relative_path in GENOME_V2_PACKAGE_ARTIFACTS:
        base_row = base_artifacts.get(artifact_id)
        if not isinstance(base_row, Mapping):
            raise ACSCandidateSpecError(
                f"base specification lacks Genome V2 artifact: {artifact_id}"
            )
        kind = base_row.get("kind")
        abi = base_row.get("abi")
        if not isinstance(kind, str) or not kind or not isinstance(abi, str) or not abi:
            raise ACSCandidateSpecError(
                f"base Genome V2 artifact metadata is malformed: {artifact_id}"
            )
        path = _regular_file(root / relative_path, label=artifact_id)
        rows[artifact_id] = _artifact_row(
            kind=kind, abi=abi, path=path, sha256=_sha256(path)
        )
    return rows


def materialize_acs_candidate_spec_v1(
    *,
    base_spec_path: Path,
    candidate_receipt_path: Path,
    genome_v2_package_dir: Path,
    source_environment_receipt_path: Path | None,
    runtime_root: Path,
    campaign_id: str,
    acs_slots: int = 0,
    production_science_request_path: Path | None = None,
    native_runtime_extension_path: Path | None = None,
) -> dict[str, Any]:
    """Bind one exact ACS route to a candidate artifact without promotion."""

    if not campaign_id or campaign_id != campaign_id.strip():
        raise ACSCandidateSpecError("campaign ID must be a nonempty trimmed string")
    if type(acs_slots) is not int or acs_slots not in {0, 1, 2, 3}:
        raise ACSCandidateSpecError("ACS slots must be one of 0, 1, 2, or 3")
    destination = runtime_root.expanduser().resolve()
    if not destination.is_absolute():
        raise ACSCandidateSpecError("runtime root must be absolute")
    base = deepcopy(load_campaign_spec(base_spec_path.expanduser().resolve()))
    artifacts = base.get("artifacts")
    checkpoints = base.get("checkpoints")
    output = base.get("output")
    if not all(
        isinstance(value, Mapping) for value in (artifacts, checkpoints, output)
    ):
        raise ACSCandidateSpecError(
            "base specification lacks artifacts/checkpoints/output"
        )
    candidate_artifacts = _candidate_artifacts(candidate_receipt_path)
    manifest = _read_object(
        Path(candidate_artifacts["extension_manifest"]["path"]),
        label="candidate artifact manifest",
    )
    m0_enabled = _validated_m0_build_mode(manifest)
    if m0_enabled and acs_slots != 0:
        raise ACSCandidateSpecError(
            "Option D M0 artifact cannot materialize asynchronous ACS slots"
        )
    if m0_enabled and source_environment_receipt_path is None:
        raise ACSCandidateSpecError(
            "Option D M0 artifact requires a source/environment receipt"
        )
    if not m0_enabled and source_environment_receipt_path is not None:
        raise ACSCandidateSpecError(
            "M0-OFF artifact forbids an Option D M0 source/environment receipt"
        )
    next_artifacts = dict(artifacts)
    next_artifacts.update(candidate_artifacts)
    if native_runtime_extension_path is not None:
        next_artifacts.update(
            _native_runtime_artifacts(native_runtime_extension_path)
        )
    next_artifacts.pop(OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT, None)
    if source_environment_receipt_path is not None:
        source_environment_receipt = _regular_file(
            source_environment_receipt_path,
            label="Option D M0 source/environment receipt",
        )
        next_artifacts[OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT] = _artifact_row(
            kind=OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT,
            abi=OPTION_D_M0_SOURCE_ENVIRONMENT_ABI,
            path=source_environment_receipt,
            sha256=_sha256(source_environment_receipt),
        )
    next_artifacts.update(
        _genome_v2_package_artifacts(
            genome_v2_package_dir,
            base_artifacts=artifacts,
            artifact_sha256=candidate_artifacts["cuda_extension"]["sha256"],
        )
    )
    next_checkpoints = dict(checkpoints)
    next_checkpoints.update(
        {
            "acs_slots": acs_slots,
            "acs_capture_instrumentation": True,
            # Slot zero is synchronous, but it still uses the native
            # aggregate format.  Detached slots additionally rely on this
            # flag to prevent accidental legacy-format route planning.
            "native_streamed_checkpoint": True,
        }
    )
    next_output = dict(output)
    next_output["runtime_root"] = str(destination)
    candidate = {
        **base,
        "campaign_id": campaign_id,
        "description": (
            "ACS0 bounded capture-instrumentation candidate."
            if acs_slots == 0
            else f"ACS bounded asynchronous persistence candidate with {acs_slots} slot(s)."
        ),
        "no_claims": (
            "Candidate route-canary input only; no production authority until exact "
            "route qualification and successor-QRR sealing succeed."
        ),
        "artifacts": next_artifacts,
        "checkpoints": next_checkpoints,
        "output": next_output,
    }
    if production_science_request_path is None:
        return candidate
    request = _read_object(production_science_request_path, label="production science request")
    configurations = request.get("configurations")
    seeds = request.get("seeds")
    total_ticks = request.get("total_ticks")
    batch_cells = request.get("batch_cells")
    checkpoint_ticks = request.get("checkpoint_ticks")
    if (
        not isinstance(configurations, list)
        or not isinstance(seeds, list)
        or not isinstance(total_ticks, int)
        or not isinstance(batch_cells, int)
        or not isinstance(checkpoint_ticks, list)
        or total_ticks <= 0
        or batch_cells <= 0
    ):
        raise ACSCandidateSpecError("production science request is malformed")
    profile = resolve_campaign_profile(candidate)
    materialized, _ = materialize_full_stack_wrc_science_spec_v1(
        candidate,
        profile=profile,
        axes={},
        configuration_rows=configurations,
        seeds={"task": seeds},
        fixed_context_axes={},
        generations=total_ticks // int(candidate["execution"]["ticks_per_generation"]),
    )
    execution = dict(materialized["execution"])
    execution["batch"] = {"mode": "automatic", "maximum": batch_cells}
    materialized["execution"] = execution
    checkpoints = dict(materialized["checkpoints"])
    ticks_per_generation = int(execution["ticks_per_generation"])
    checkpoints["milestones"] = [tick // ticks_per_generation for tick in checkpoint_ticks]
    materialized["checkpoints"] = checkpoints
    materialized["description"] = (
        "ACS0 B48 prepared-owner candidate; non-production."
        if acs_slots == 0
        else f"ACS B48 {acs_slots}-slot persistence candidate; non-production."
    )
    return materialized


def write_new_candidate_spec_v1(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise ACSCandidateSpecError(
            f"refusing to overwrite candidate specification: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-spec", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--genome-v2-package-dir", type=Path, required=True)
    parser.add_argument("--source-environment-receipt", type=Path)
    parser.add_argument("--acs-slots", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--production-science-request", type=Path)
    parser.add_argument("--native-runtime-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = materialize_acs_candidate_spec_v1(
            base_spec_path=args.base_spec,
            candidate_receipt_path=args.candidate_receipt,
            genome_v2_package_dir=args.genome_v2_package_dir,
            source_environment_receipt_path=args.source_environment_receipt,
            runtime_root=args.runtime_root,
            campaign_id=args.campaign_id,
            acs_slots=args.acs_slots,
            production_science_request_path=args.production_science_request,
            native_runtime_extension_path=args.native_runtime_extension,
        )
        output = write_new_candidate_spec_v1(args.output, payload)
    except ACSCandidateSpecError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "schema_version": SCHEMA_VERSION,
                "output": str(output),
                "output_sha256": _sha256(output),
                "acs_capture_instrumentation": True,
                "acs_slots": args.acs_slots,
                "production_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
