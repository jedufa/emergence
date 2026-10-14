#!/usr/bin/env python3
"""Materialize immutable, authority-bound Genome V2 E1 campaign specs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Final, Mapping

import yaml

from engine.authoritative_state.schema import content_sha256
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
    PREREGISTRATION_SHA256,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    package_identity_sha256_v1,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SCHEMA_VERSION: Final[str] = "genome_v2_e1_spec_materialization_v1"
PACKAGE_ABI: Final[str] = "genome_upgrade_v2_fixed_q10_v1"
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
DEFAULT_TEMPLATES: Final[tuple[Path, ...]] = (
    ROOT
    / "_spec/campaigns/v1/genome_v2_e1_neutral_no_influence_g32.yaml",
    ROOT / "_spec/campaigns/v1/genome_v2_e1_patch_pleiotropic_g32.yaml",
    ROOT / "_spec/campaigns/v1/genome_v2_e1_wdl_hazard_regulatory_g32.yaml",
)
PACKAGE_ARTIFACTS: Final[dict[str, tuple[str, str]]] = {
    "genome_upgrade_v2_executable_contract": (
        "genome_v2_executable_contract.json",
        "executable_contract",
    ),
    "genome_upgrade_v2_capability_contract": (
        "genome_v2_p10_capability_contract.json",
        "capability_contract",
    ),
    "genome_upgrade_v2_package_contract": (
        "genome_v2_executable_package_contract_v1.json",
        "package_contract",
    ),
    "genome_upgrade_v2_obligation_dictionary": (
        "genome_v2_package_obligation_dictionary.json",
        "obligation_dictionary",
    ),
    "genome_upgrade_v2_invalidation_rules": (
        "genome_v2_package_invalidation_rules.json",
        "invalidation_rules",
    ),
    "genome_upgrade_v2_chronicle_schema": (
        "genome_v2_chronicle_schema.json",
        "chronicle_schema",
    ),
    "genome_upgrade_v2_reporting_schema": (
        "genome_v2_reporting_schema.json",
        "reporting_schema",
    ),
    "genome_upgrade_v2_observability_storage_budget": (
        "genome_v2_observability_storage_budget.json",
        "observability_storage_budget",
    ),
}
RECEIPT_HASH_BINDINGS: Final[dict[str, str]] = {
    "genome_upgrade_v2_executable_contract": "executable_contract_sha256",
    "genome_upgrade_v2_package_contract": "package_contract_sha256",
    "genome_upgrade_v2_obligation_dictionary": (
        "obligation_dictionary_sha256"
    ),
    "genome_upgrade_v2_invalidation_rules": "invalidation_rules_sha256",
}


class MaterializationError(RuntimeError):
    """The E1 runtime specifications cannot be materialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path, *, label: str) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise MaterializationError(
            f"{label} must be an existing non-symlink file: {path}"
        )
    return path


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    path = _file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must contain a JSON object: {path}")
    return value


def _artifact_row(
    path: Path, *, kind: str, abi: str
) -> dict[str, str]:
    path = _file(path, label=kind)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "kind": kind,
        "abi": abi,
    }


def _validate_route(
    *,
    label: str,
    artifact: Path,
    manifest: Path,
    qualification: Path,
    reporting: Path,
    binding_id: str,
    abi: str,
) -> dict[str, dict[str, str]]:
    artifact = _file(artifact, label=f"{label} CUDA artifact")
    manifest_payload = _json_file(
        manifest, label=f"{label} artifact manifest"
    )
    qualification_payload = _json_file(
        qualification, label=f"{label} qualification receipt"
    )
    reporting_payload = _json_file(
        reporting, label=f"{label} reporting receipt"
    )
    artifact_sha256 = _sha256(artifact)
    manifest = manifest.expanduser().absolute()
    if (
        manifest_payload.get("artifact_sha256") != artifact_sha256
        or Path(str(manifest_payload.get("artifact", ""))).expanduser().absolute()
        != artifact
    ):
        raise MaterializationError(
            f"{label} manifest does not bind the selected artifact"
        )
    expected_common = {
        "ok": True,
        "provider_id": PROVIDER_ID,
        "profile_id": RUNTIME_PROFILE_ID,
        "mechanics_binding_id": binding_id,
    }
    if any(qualification_payload.get(key) != value for key, value in expected_common.items()):
        raise MaterializationError(
            f"{label} qualification receipt has the wrong production binding"
        )
    qualified_artifact = qualification_payload.get("artifact")
    if (
        qualification_payload.get("artifact_abi") != abi
        or not isinstance(qualified_artifact, Mapping)
        or qualified_artifact.get("sha256") != artifact_sha256
        or qualified_artifact.get("manifest_sha256") != _sha256(manifest)
    ):
        raise MaterializationError(
            f"{label} qualification receipt does not bind the artifact"
        )
    if (
        any(reporting_payload.get(key) != value for key, value in expected_common.items())
        or reporting_payload.get("artifact_sha256") != artifact_sha256
    ):
        raise MaterializationError(
            f"{label} reporting receipt does not bind the artifact"
        )
    return {
        "cuda_extension": _artifact_row(
            artifact, kind="cuda_extension", abi=abi
        ),
        "extension_manifest": _artifact_row(
            manifest, kind="extension_manifest", abi=abi
        ),
        "qualification_receipt": _artifact_row(
            qualification, kind="qualification_receipt", abi=abi
        ),
        "reporting_receipt": _artifact_row(
            reporting, kind="reporting_receipt", abi=abi
        ),
    }


def _validate_package(
    *,
    contracts_dir: Path,
    package_receipt: Path,
    artifact_sha256s: set[str],
) -> dict[str, dict[str, str]]:
    contracts_dir = contracts_dir.expanduser().absolute()
    if contracts_dir.is_symlink() or not contracts_dir.is_dir():
        raise MaterializationError(
            "package contracts directory must be an existing non-symlink "
            f"directory: {contracts_dir}"
        )
    rows: dict[str, dict[str, str]] = {}
    for artifact_id, (filename, kind) in PACKAGE_ARTIFACTS.items():
        rows[artifact_id] = _artifact_row(
            contracts_dir / filename, kind=kind, abi=PACKAGE_ABI
        )
    package_receipt = _file(
        package_receipt, label="Genome V2 package receipt"
    )
    receipt = _json_file(
        package_receipt, label="Genome V2 package receipt"
    )
    identity = receipt.get("identity")
    if (
        receipt.get("ok") is not True
        or receipt.get("authority_state") != "package_candidate_complete"
        or receipt.get("package_id_sha256")
        != package_identity_sha256_v1(receipt)
        or not isinstance(identity, Mapping)
        or identity.get("provider_id") != PROVIDER_ID
        or identity.get("runtime_profile_id") != RUNTIME_PROFILE_ID
        or identity.get("cuda_artifact_sha256") not in artifact_sha256s
        or identity.get("loaded_artifact_sha256")
        != identity.get("cuda_artifact_sha256")
        or any(
            receipt.get(receipt_field) != rows[artifact_id]["sha256"]
            for artifact_id, receipt_field in RECEIPT_HASH_BINDINGS.items()
        )
    ):
        raise MaterializationError(
            "Genome V2 package receipt is invalid or does not bind the "
            "selected production artifact"
        )
    if any(
        artifact_sha256 != identity["cuda_artifact_sha256"]
        for artifact_sha256 in artifact_sha256s
    ):
        raise MaterializationError(
            "every selected route artifact must be bound by the shared "
            "Genome V2 package receipt"
        )
    rows["genome_upgrade_v2_package_receipt"] = _artifact_row(
        package_receipt, kind="package_receipt", abi=PACKAGE_ABI
    )
    return rows


def _load_template(path: Path) -> dict[str, Any]:
    path = _file(path, label="E1 campaign template")
    document = load_campaign_spec(path)
    artifacts = document.get("artifacts")
    prereg = artifacts.get("e1_preregistration") if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(prereg, Mapping)
        or prereg.get("sha256") != PREREGISTRATION_SHA256
    ):
        raise MaterializationError(
            f"template E1 preregistration hash differs from code authority: {path}"
        )
    return document


def materialize(
    *,
    output_dir: Path,
    runtime_root: Path,
    templates: tuple[Path, Path, Path],
    patch_artifact: Path,
    patch_manifest: Path,
    patch_qualification: Path,
    patch_reporting: Path,
    wdl_artifact: Path,
    wdl_manifest: Path,
    wdl_qualification: Path,
    wdl_reporting: Path,
    package_contracts_dir: Path,
    package_receipt: Path,
    e1_preregistration: Path,
    e1_preregistration_sha256: str,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().absolute()
    runtime_root = runtime_root.expanduser().absolute()
    if not runtime_root.is_absolute():
        raise MaterializationError("runtime root must be absolute")
    if output_dir.exists() or output_dir.is_symlink():
        raise MaterializationError(
            f"refusing to overwrite materialized output: {output_dir}"
        )
    if len(set(templates)) != 3:
        raise MaterializationError("the three E1 template paths must be unique")
    documents = [_load_template(path) for path in templates]
    campaign_ids = [str(document["campaign_id"]) for document in documents]
    expected_campaign_ids = {
        "genome-v2-e1-neutral-no-influence-g32",
        "genome-v2-e1-patch-pleiotropic-g32",
        "genome-v2-e1-wdl-hazard-regulatory-g32",
    }
    if set(campaign_ids) != expected_campaign_ids:
        raise MaterializationError(
            "templates do not contain the exact frozen E1 campaign set"
        )
    prereg = _file(e1_preregistration, label="E1 preregistration")
    if (
        len(e1_preregistration_sha256) != 64
        or _sha256(prereg) != e1_preregistration_sha256
    ):
        raise MaterializationError("E1 preregistration file SHA-256 mismatch")
    prereg_payload = _json_file(prereg, label="E1 preregistration")
    if content_sha256(prereg_payload) != PREREGISTRATION_SHA256:
        raise MaterializationError(
            "E1 preregistration content differs from code authority"
        )
    patch_rows = _validate_route(
        label="Patch",
        artifact=patch_artifact,
        manifest=patch_manifest,
        qualification=patch_qualification,
        reporting=patch_reporting,
        binding_id="patch_fixed_mlp_native_binding_v1",
        abi="architecture_b_patch_p2_stage3_v2",
    )
    wdl_rows = _validate_route(
        label="WDL",
        artifact=wdl_artifact,
        manifest=wdl_manifest,
        qualification=wdl_qualification,
        reporting=wdl_reporting,
        binding_id="wdl_fixed_mlp_native_binding_v1",
        abi="architecture_b_wdl_fixed_mlp_v2",
    )
    package_rows = _validate_package(
        contracts_dir=package_contracts_dir,
        package_receipt=package_receipt,
        artifact_sha256s={
            patch_rows["cuda_extension"]["sha256"],
            wdl_rows["cuda_extension"]["sha256"],
        },
    )
    prereg_row = {
        "path": str(prereg),
        "sha256": e1_preregistration_sha256,
        "kind": "preregistration",
        "abi": "genome_v2_p10_e1_ecological_gate_v1",
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        plans: dict[str, dict[str, str]] = {}
        artifact_hashes: dict[str, dict[str, str]] = {}
        for document in documents:
            campaign_id = str(document["campaign_id"])
            route_rows = (
                wdl_rows
                if document["world"]["overlays"] == ["wdl_delayed_hazard"]
                else patch_rows
            )
            document["artifacts"] = {
                **route_rows,
                **package_rows,
                "e1_preregistration": prereg_row,
            }
            document["output"]["runtime_root"] = str(
                runtime_root / campaign_id
            )
            spec_path = temporary / f"{campaign_id}.yaml"
            spec_path.write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            loaded = load_campaign_spec(spec_path)
            plan = compile_campaign(loaded)
            plans[campaign_id] = {
                "spec_path": spec_path.name,
                "spec_sha256": _sha256(spec_path),
                "plan_sha256": plan.plan_sha256,
            }
            artifact_hashes[campaign_id] = {
                artifact_id: str(row["sha256"])
                for artifact_id, row in sorted(
                    document["artifacts"].items()
                )
            }
        manifest_core = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "provider_id": PROVIDER_ID,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
            "e1_preregistration_sha256": PREREGISTRATION_SHA256,
            "runtime_root": str(runtime_root),
            "plans": dict(sorted(plans.items())),
            "artifact_sha256s": dict(sorted(artifact_hashes.items())),
        }
        manifest_id = content_sha256(manifest_core)
        manifest = {
            **manifest_core,
            "materialization_sha256": manifest_id,
        }
        manifest_name = (
            f"genome_v2_e1_materialized_specs_manifest_{manifest_id}.json"
        )
        manifest_path = temporary / manifest_name
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        **manifest,
        "manifest_path": str(output_dir / manifest_name),
        "manifest_file_sha256": _sha256(output_dir / manifest_name),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--template", action="append", type=Path, default=[]
    )
    for route in ("patch", "wdl"):
        parser.add_argument(f"--{route}-artifact", type=Path, required=True)
        parser.add_argument(f"--{route}-manifest", type=Path, required=True)
        parser.add_argument(
            f"--{route}-qualification", type=Path, required=True
        )
        parser.add_argument(
            f"--{route}-reporting", type=Path, required=True
        )
    parser.add_argument("--package-contracts-dir", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--e1-preregistration", type=Path, required=True)
    parser.add_argument(
        "--e1-preregistration-sha256", required=True
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    templates = tuple(args.template or DEFAULT_TEMPLATES)
    if len(templates) != 3:
        print(
            json.dumps(
                {"ok": False, "error": "exactly three --template values required"},
                sort_keys=True,
            )
        )
        return 1
    try:
        result = materialize(
            output_dir=args.output_dir,
            runtime_root=args.runtime_root,
            templates=templates,  # type: ignore[arg-type]
            patch_artifact=args.patch_artifact,
            patch_manifest=args.patch_manifest,
            patch_qualification=args.patch_qualification,
            patch_reporting=args.patch_reporting,
            wdl_artifact=args.wdl_artifact,
            wdl_manifest=args.wdl_manifest,
            wdl_qualification=args.wdl_qualification,
            wdl_reporting=args.wdl_reporting,
            package_contracts_dir=args.package_contracts_dir,
            package_receipt=args.package_receipt,
            e1_preregistration=args.e1_preregistration,
            e1_preregistration_sha256=args.e1_preregistration_sha256,
        )
    except (MaterializationError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
