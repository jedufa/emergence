from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.campaigns.production_engine_configuration import (
    production_engine_configuration_v1,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
    PREREGISTRATION_SHA256,
    preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    package_identity_sha256_v1,
)
from tools.scripts.genome_upgrade_v2_e1_materialize_specs_v1 import (
    DEFAULT_TEMPLATES,
    PACKAGE_ARTIFACTS,
    MaterializationError,
    materialize,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _route(
    root: Path, *, name: str, binding_id: str, abi: str
) -> dict[str, Path]:
    root.mkdir()
    artifact = root / "unified.so"
    artifact.write_bytes(b"ELF-" + name.encode("ascii"))
    manifest = root / "unified.so.manifest.json"
    _write_json(
        manifest,
        {
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
        },
    )
    common = {
        "ok": True,
        "provider_id": "eiel-production-provider-v1",
        "profile_id": "eiel-production-runtime-v1",
        "mechanics_binding_id": binding_id,
    }
    qualification = root / "qualification.json"
    _write_json(
        qualification,
        {
            **common,
            "artifact_abi": abi,
            "artifact": {
                "sha256": _sha256(artifact),
                "manifest_sha256": _sha256(manifest),
            },
        },
    )
    reporting = root / "reporting.json"
    _write_json(
        reporting,
        {**common, "artifact_sha256": _sha256(artifact)},
    )
    return {
        "artifact": artifact,
        "manifest": manifest,
        "qualification": qualification,
        "reporting": reporting,
    }


def _inputs(tmp_path: Path) -> dict[str, object]:
    # The shared package receipt intentionally binds one unified binary.
    patch = _route(
        tmp_path / "patch",
        name="shared",
        binding_id="patch_fixed_mlp_native_binding_v1",
        abi="architecture_b_patch_p2_stage3_v2",
    )
    wdl_root = tmp_path / "wdl"
    wdl_root.mkdir()
    wdl_artifact = wdl_root / "unified.so"
    wdl_artifact.write_bytes(patch["artifact"].read_bytes())
    wdl_manifest = wdl_root / "unified.so.manifest.json"
    _write_json(
        wdl_manifest,
        {
            "artifact": str(wdl_artifact),
            "artifact_sha256": _sha256(wdl_artifact),
        },
    )
    common = {
        "ok": True,
        "provider_id": "eiel-production-provider-v1",
        "profile_id": "eiel-production-runtime-v1",
        "mechanics_binding_id": "wdl_fixed_mlp_native_binding_v1",
    }
    wdl_qualification = wdl_root / "qualification.json"
    _write_json(
        wdl_qualification,
        {
            **common,
            "artifact_abi": "architecture_b_wdl_fixed_mlp_v2",
            "artifact": {
                "sha256": _sha256(wdl_artifact),
                "manifest_sha256": _sha256(wdl_manifest),
            },
        },
    )
    wdl_reporting = wdl_root / "reporting.json"
    _write_json(
        wdl_reporting,
        {**common, "artifact_sha256": _sha256(wdl_artifact)},
    )
    wdl = {
        "artifact": wdl_artifact,
        "manifest": wdl_manifest,
        "qualification": wdl_qualification,
        "reporting": wdl_reporting,
    }

    contracts = tmp_path / "contracts"
    contracts.mkdir()
    rows: dict[str, str] = {}
    for artifact_id, (filename, _kind) in PACKAGE_ARTIFACTS.items():
        path = contracts / filename
        _write_json(path, {"artifact_id": artifact_id})
        rows[artifact_id] = _sha256(path)
    receipt = {
        "schema_version": "genome_v2_executable_package_identity_receipt_v1",
        "authority_state": "package_candidate_complete",
        "ok": True,
        "package_contract_sha256": rows[
            "genome_upgrade_v2_package_contract"
        ],
        "obligation_dictionary_sha256": rows[
            "genome_upgrade_v2_obligation_dictionary"
        ],
        "invalidation_rules_sha256": rows[
            "genome_upgrade_v2_invalidation_rules"
        ],
        "executable_contract_sha256": rows[
            "genome_upgrade_v2_executable_contract"
        ],
        "identity": {
            "provider_id": "eiel-production-provider-v1",
            "runtime_profile_id": "eiel-production-runtime-v1",
            "cuda_artifact_sha256": _sha256(patch["artifact"]),
            "loaded_artifact_sha256": _sha256(patch["artifact"]),
        },
        "obligations": {},
    }
    receipt["package_id_sha256"] = package_identity_sha256_v1(receipt)
    receipt_path = tmp_path / "package_receipt.json"
    _write_json(receipt_path, receipt)
    prereg = tmp_path / "e1_preregistration.json"
    _write_json(prereg, preregistration_payload_v1())
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    templates: list[Path] = []
    for source in DEFAULT_TEMPLATES:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        document["artifacts"]["e1_preregistration"]["sha256"] = (
            PREREGISTRATION_SHA256
        )
        destination = templates_dir / source.name
        destination.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        templates.append(destination)
    return {
        "output_dir": tmp_path / "materialized",
        "runtime_root": tmp_path / "runtime",
        "templates": tuple(templates),
        "patch_artifact": patch["artifact"],
        "patch_manifest": patch["manifest"],
        "patch_qualification": patch["qualification"],
        "patch_reporting": patch["reporting"],
        "wdl_artifact": wdl["artifact"],
        "wdl_manifest": wdl["manifest"],
        "wdl_qualification": wdl["qualification"],
        "wdl_reporting": wdl["reporting"],
        "package_contracts_dir": contracts,
        "package_receipt": receipt_path,
        "e1_preregistration": prereg,
        "e1_preregistration_sha256": _sha256(prereg),
    }


def test_materializes_exact_production_specs_and_compiles(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    result = materialize(**inputs)  # type: ignore[arg-type]

    output = inputs["output_dir"]
    assert isinstance(output, Path)
    specs = sorted(output.glob("*.yaml"))
    assert len(specs) == 3
    expected_keys = {
        "cuda_extension",
        "extension_manifest",
        "qualification_receipt",
        "reporting_receipt",
        *(artifact_id for artifact_id in PACKAGE_ARTIFACTS),
        "genome_upgrade_v2_package_receipt",
        "e1_preregistration",
    }
    configuration_hashes: dict[str, str] = {}
    for path in specs:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(document["artifacts"]) == expected_keys
        assert document["execution"]["budgets"]["vram_bytes"] == 2_000_000_000
        assert document["output"]["runtime_root"] == str(
            inputs["runtime_root"] / document["campaign_id"]
        )
        plan = compile_campaign(load_campaign_spec(path))
        assert plan.plan_sha256 == (
            result["plans"][document["campaign_id"]]["plan_sha256"]
        )
        configuration_hashes[document["campaign_id"]] = (
            production_engine_configuration_v1(
                plan.payload()
            ).configuration_sha256
        )
    assert (
        configuration_hashes["genome-v2-e1-neutral-no-influence-g32"]
        == configuration_hashes["genome-v2-e1-patch-pleiotropic-g32"]
    )
    assert (
        configuration_hashes["genome-v2-e1-wdl-hazard-regulatory-g32"]
        != configuration_hashes["genome-v2-e1-patch-pleiotropic-g32"]
    )
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.name.endswith(
        f"{result['materialization_sha256']}.json"
    )


@pytest.mark.parametrize(
    "case",
    ("artifact_tamper", "wrong_package", "existing_output"),
)
def test_rejects_tamper_wrong_package_and_existing_output(
    tmp_path: Path, case: str
) -> None:
    inputs = _inputs(tmp_path)
    if case == "artifact_tamper":
        Path(inputs["patch_artifact"]).write_bytes(b"tampered")
    elif case == "wrong_package":
        receipt_path = Path(inputs["package_receipt"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["identity"]["cuda_artifact_sha256"] = "0" * 64
        receipt["identity"]["loaded_artifact_sha256"] = "0" * 64
        receipt["package_id_sha256"] = package_identity_sha256_v1(receipt)
        _write_json(receipt_path, receipt)
    else:
        Path(inputs["output_dir"]).mkdir()

    with pytest.raises(MaterializationError):
        materialize(**inputs)  # type: ignore[arg-type]
    assert case == "existing_output" or not Path(inputs["output_dir"]).exists()
