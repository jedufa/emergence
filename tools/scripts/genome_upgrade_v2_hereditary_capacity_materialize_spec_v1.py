#!/usr/bin/env python3
"""Materialize the authority-bound Genome V2 hereditary-capacity campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Final, Mapping, Sequence

import yaml

from engine.authoritative_state.schema import content_sha256
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
    validate_genome_v2_p10_capability_contract_v1,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    PREREGISTRATION_SHA256,
    SCHEMA_VERSION as PREREGISTRATION_SCHEMA_VERSION,
    preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION,
    package_identity_sha256_v1,
)


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE: Final[Path] = (
    ROOT
    / "_spec/campaigns/v1/genome_v2_p10_hereditary_capacity_b2g3.yaml"
)
SCHEMA_VERSION: Final[str] = (
    "genome_v2_p10_hereditary_capacity_spec_materialization_v1"
)
CAMPAIGN_ID: Final[str] = "genome-v2-p10-hereditary-capacity-b2g3"
NO_CLAIMS: Final[str] = (
    "Bounded hereditary capacity and cost evidence only; no ecological "
    "usefulness, cognition, intelligence, emergence, speciation, or "
    "open-ended evolution claim."
)
PROVIDER_ID: Final[str] = "eiel-production-provider-v1"
RUNTIME_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
ROUTE_ABI: Final[str] = "architecture_b_patch_p2_stage3_v2"
PACKAGE_ABI: Final[str] = "genome_upgrade_v2_fixed_q10_v1"
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
    """The frozen hereditary-capacity specification is not admissible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().absolute()
    if resolved.is_symlink() or not resolved.is_file():
        raise MaterializationError(
            f"{label} must be an existing non-symlink file: {resolved}"
        )
    return resolved


def _json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _file(path, label=label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must contain an object")
    return value


def _row(path: Path, *, kind: str, abi: str) -> dict[str, str]:
    resolved = _file(path, label=kind)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "kind": kind,
        "abi": abi,
    }


def _validate_route(
    *,
    artifact: Path,
    manifest: Path,
    qualification: Path,
    reporting: Path,
) -> dict[str, dict[str, str]]:
    artifact = _file(artifact, label="Patch CUDA artifact")
    manifest_payload = _json(manifest, label="Patch artifact manifest")
    qualification_payload = _json(
        qualification, label="Patch qualification receipt"
    )
    reporting_payload = _json(reporting, label="Patch reporting receipt")
    artifact_sha = _sha256(artifact)
    manifest = _file(manifest, label="Patch artifact manifest")
    expected = {
        "ok": True,
        "provider_id": PROVIDER_ID,
        "profile_id": RUNTIME_PROFILE_ID,
        "mechanics_binding_id": "patch_fixed_mlp_native_binding_v1",
    }
    qualified_artifact = qualification_payload.get("artifact")
    if (
        manifest_payload.get("artifact_sha256") != artifact_sha
        or Path(str(manifest_payload.get("artifact", ""))).expanduser().absolute()
        != artifact
        or any(
            qualification_payload.get(name) != value
            for name, value in expected.items()
        )
        or qualification_payload.get("artifact_abi") != ROUTE_ABI
        or not isinstance(qualified_artifact, Mapping)
        or qualified_artifact.get("sha256") != artifact_sha
        or qualified_artifact.get("manifest_sha256") != _sha256(manifest)
        or any(
            reporting_payload.get(name) != value
            for name, value in expected.items()
        )
        or reporting_payload.get("artifact_sha256") != artifact_sha
    ):
        raise MaterializationError(
            "Patch artifact, manifest, qualification, and reporting do not "
            "form one exact production binding"
        )
    return {
        "cuda_extension": _row(
            artifact, kind="cuda_extension", abi=ROUTE_ABI
        ),
        "extension_manifest": _row(
            manifest, kind="extension_manifest", abi=ROUTE_ABI
        ),
        "qualification_receipt": _row(
            qualification, kind="qualification_receipt", abi=ROUTE_ABI
        ),
        "reporting_receipt": _row(
            reporting, kind="reporting_receipt", abi=ROUTE_ABI
        ),
    }


def _validate_package(
    *,
    contracts_dir: Path,
    package_receipt: Path,
    artifact_sha256: str,
) -> dict[str, dict[str, str]]:
    contracts_dir = contracts_dir.expanduser().absolute()
    if contracts_dir.is_symlink() or not contracts_dir.is_dir():
        raise MaterializationError("package contracts directory is invalid")
    rows = {
        artifact_id: _row(
            contracts_dir / filename, kind=kind, abi=PACKAGE_ABI
        )
        for artifact_id, (filename, kind) in PACKAGE_ARTIFACTS.items()
    }
    capability = _json(
        contracts_dir / PACKAGE_ARTIFACTS[
            "genome_upgrade_v2_capability_contract"
        ][0],
        label="capability contract",
    )
    try:
        validated_capability = (
            validate_genome_v2_p10_capability_contract_v1(capability)
        )
    except ValueError as exc:
        raise MaterializationError(
            "capability contract differs from current P1.0 authority"
        ) from exc
    receipt_path = _file(
        package_receipt, label="Genome V2 package receipt"
    )
    receipt = _json(receipt_path, label="Genome V2 package receipt")
    identity = receipt.get("identity")
    if (
        receipt.get("schema_version")
        != PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION
        or receipt.get("ok") is not True
        or receipt.get("authority_state") != "package_candidate_complete"
        or receipt.get("package_id_sha256")
        != package_identity_sha256_v1(receipt)
        or not isinstance(identity, Mapping)
        or identity.get("provider_id") != PROVIDER_ID
        or identity.get("runtime_profile_id") != RUNTIME_PROFILE_ID
        or identity.get("cuda_artifact_sha256") != artifact_sha256
        or identity.get("loaded_artifact_sha256") != artifact_sha256
        or identity.get("capability_contract_sha256")
        != validated_capability.contract_sha256
        or validated_capability.contract_sha256
        != GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        or identity.get("hereditary_capacity_schema_version")
        != PREREGISTRATION_SCHEMA_VERSION
        or identity.get("hereditary_capacity_preregistration_sha256")
        != PREREGISTRATION_SHA256
        or any(
            receipt.get(receipt_field) != rows[artifact_id]["sha256"]
            for artifact_id, receipt_field in RECEIPT_HASH_BINDINGS.items()
        )
    ):
        raise MaterializationError(
            "package receipt or contracts differ from current package authority"
        )
    rows["genome_upgrade_v2_package_receipt"] = _row(
        receipt_path, kind="package_receipt", abi=PACKAGE_ABI
    )
    return rows


def _template_design(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "world": document["world"]["base"],
        "overlays": document["world"]["overlays"],
        "policy_family": document["policy"]["family"],
        "selection_mode": document["selection"]["mode"],
        "grid_size": document["parameters"]["patch.grid_size"],
        "initial_population": document["parameters"]["patch.initial_population"],
        "resident_capacity": document["parameters"]["patch.capacity"],
        "founder_placement": document["parameters"]["patch.founder_placement"],
        "genome_profile": document["parameters"]["genome_upgrade_v2.profile"],
        "founder_profile": document["parameters"][
            "genome_upgrade_v2.founder_profile"
        ],
        "mutation_rate_q10": document["parameters"][
            "genome_upgrade_v2.mutation_rate_q10"
        ],
        "mutation_step_q10": document["parameters"][
            "genome_upgrade_v2.mutation_step_q10"
        ],
        "ecology_seeds": document["matrix"]["dimensions"][0]["values"],
        "task_seeds": document["matrix"]["dimensions"][1]["values"],
        "backend": document["execution"]["backend"],
        "logical_cells": document["matrix"]["expected_cells"],
        "physical_batch_maximum": document["execution"]["batch"]["maximum"],
        "generations": document["execution"]["generations"],
        "ticks_per_generation": document["execution"]["ticks_per_generation"],
        "checkpoint_generations": document["checkpoints"]["milestones"],
        "digest_every_generation": document["checkpoints"][
            "digest_every_generation"
        ],
        "checkpoint_storage_mode": document["checkpoints"]["storage_mode"],
        "fresh_repeats": document["comparisons"]["fresh_repeats"],
        "continuation_cells": document["comparisons"]["continuation_cells"],
    }


def _template_structure_is_exact(document: Mapping[str, Any]) -> bool:
    parameters = document.get("parameters")
    matrix = document.get("matrix")
    dimensions = matrix.get("dimensions") if isinstance(matrix, Mapping) else None
    comparisons = document.get("comparisons")
    return bool(
        document.get("schema_version") == "eiel_campaign_v1"
        and document.get("no_claims") == NO_CLAIMS
        and isinstance(parameters, Mapping)
        and set(parameters)
        == {
            "patch.grid_size",
            "patch.initial_population",
            "patch.capacity",
            "patch.founder_placement",
            "genome_upgrade_v2.profile",
            "genome_upgrade_v2.enable",
            "genome_upgrade_v2.founder_profile",
            "genome_upgrade_v2.mutation_rate_q10",
            "genome_upgrade_v2.mutation_step_q10",
        }
        and isinstance(matrix, Mapping)
        and isinstance(dimensions, list)
        and [row.get("id") for row in dimensions if isinstance(row, Mapping)]
        == ["ecology_seed", "task_seed"]
        and matrix.get("parameter_bindings") == {}
        and matrix.get("seed_bindings")
        == {"ecology_seed": "ecology", "task_seed": "task"}
        and matrix.get("include") == []
        and matrix.get("exclude") == []
        and matrix.get("max_cells") == 2
        and isinstance(comparisons, Mapping)
        and dict(comparisons)
        == {
            "fresh_repeats": 0,
            "continuation_splits": [],
            "continuation_cells": 0,
        }
    )


def materialize(
    *,
    output_dir: Path,
    runtime_root: Path,
    template: Path,
    patch_artifact: Path,
    patch_manifest: Path,
    patch_qualification: Path,
    patch_reporting: Path,
    package_contracts_dir: Path,
    package_receipt: Path,
    hereditary_preregistration: Path,
    hereditary_preregistration_sha256: str,
) -> dict[str, Any]:
    if not runtime_root.expanduser().is_absolute():
        raise MaterializationError("runtime root must be absolute")
    output_dir = output_dir.expanduser().absolute()
    runtime_root = runtime_root.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise MaterializationError("refusing to overwrite materialized output")
    prereg_path = _file(
        hereditary_preregistration,
        label="hereditary-capacity preregistration",
    )
    prereg = _json(
        prereg_path, label="hereditary-capacity preregistration"
    )
    if (
        _sha256(prereg_path) != hereditary_preregistration_sha256
        or prereg != preregistration_payload_v1()
        or content_sha256(prereg) != PREREGISTRATION_SHA256
    ):
        raise MaterializationError(
            "hereditary-capacity preregistration content or hash is stale"
        )
    template = _file(template, label="hereditary-capacity template")
    try:
        document = yaml.safe_load(template.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MaterializationError("campaign template is invalid YAML") from exc
    if (
        not isinstance(document, dict)
        or document.get("campaign_id") != CAMPAIGN_ID
        or not _template_structure_is_exact(document)
        or _template_design(document) != prereg["campaign_design"]
        or document["parameters"].get("genome_upgrade_v2.enable") is not True
    ):
        raise MaterializationError(
            "campaign template differs from preregistered exact design"
        )
    prereg_template = document.get("artifacts", {}).get(
        "hereditary_capacity_preregistration"
    )
    if (
        not isinstance(prereg_template, Mapping)
        or prereg_template.get("sha256") != PREREGISTRATION_SHA256
    ):
        raise MaterializationError(
            "template preregistration hash differs from code authority"
        )
    route_rows = _validate_route(
        artifact=patch_artifact,
        manifest=patch_manifest,
        qualification=patch_qualification,
        reporting=patch_reporting,
    )
    package_rows = _validate_package(
        contracts_dir=package_contracts_dir,
        package_receipt=package_receipt,
        artifact_sha256=route_rows["cuda_extension"]["sha256"],
    )
    document["artifacts"] = {
        **route_rows,
        **package_rows,
        "hereditary_capacity_preregistration": {
            "path": str(prereg_path),
            "sha256": hereditary_preregistration_sha256,
            "kind": "preregistration",
            "abi": PREREGISTRATION_SCHEMA_VERSION,
        },
    }
    document["output"]["runtime_root"] = str(runtime_root / CAMPAIGN_ID)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        spec_path = temporary / f"{CAMPAIGN_ID}.yaml"
        spec_path.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        plan = compile_campaign(load_campaign_spec(spec_path))
        artifact_sha256s = {
            name: row["sha256"]
            for name, row in sorted(document["artifacts"].items())
        }
        core = {
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "provider_id": PROVIDER_ID,
            "runtime_profile_id": RUNTIME_PROFILE_ID,
            "campaign_id": CAMPAIGN_ID,
            "hereditary_capacity_preregistration_sha256": (
                PREREGISTRATION_SHA256
            ),
            "runtime_root": str(runtime_root),
            "spec_path": spec_path.name,
            "spec_sha256": _sha256(spec_path),
            "plan_sha256": plan.plan_sha256,
            "qualification_key_sha256": plan.qualification_key_sha256,
            "artifact_sha256s": artifact_sha256s,
        }
        materialization_sha = content_sha256(core)
        manifest = {
            **core,
            "materialization_sha256": materialization_sha,
        }
        manifest_name = (
            "genome_v2_p10_hereditary_capacity_materialized_spec_manifest_"
            f"{materialization_sha}.json"
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--patch-artifact", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--patch-qualification", type=Path, required=True)
    parser.add_argument("--patch-reporting", type=Path, required=True)
    parser.add_argument("--package-contracts-dir", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument(
        "--hereditary-preregistration", type=Path, required=True
    )
    parser.add_argument(
        "--hereditary-preregistration-sha256", required=True
    )
    args = parser.parse_args(argv)
    try:
        result = materialize(**vars(args))
    except (MaterializationError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
