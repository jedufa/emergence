from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    GenomeV2P10CapabilityContract,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    PREREGISTRATION_SHA256,
    SCHEMA_VERSION as PREREGISTRATION_SCHEMA_VERSION,
    preregistration_payload_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    package_identity_sha256_v1,
)
from tools.scripts.genome_upgrade_v2_hereditary_capacity_materialize_spec_v1 import (
    CAMPAIGN_ID,
    DEFAULT_TEMPLATE,
    PACKAGE_ARTIFACTS,
    MaterializationError,
    materialize,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _inputs(tmp_path: Path) -> dict[str, object]:
    route = tmp_path / "route"
    route.mkdir()
    artifact = route / "unified.so"
    artifact.write_bytes(b"ELF-shared-genome-v2")
    manifest = route / "unified.so.manifest.json"
    _json(
        manifest,
        {"artifact": str(artifact), "artifact_sha256": _sha(artifact)},
    )
    common = {
        "ok": True,
        "provider_id": "eiel-production-provider-v1",
        "profile_id": "eiel-production-runtime-v1",
        "mechanics_binding_id": "patch_fixed_mlp_native_binding_v1",
    }
    qualification = route / "qualification.json"
    _json(
        qualification,
        {
            **common,
            "artifact_abi": "architecture_b_patch_p2_stage3_v2",
            "artifact": {
                "sha256": _sha(artifact),
                "manifest_sha256": _sha(manifest),
            },
        },
    )
    reporting = route / "reporting.json"
    _json(reporting, {**common, "artifact_sha256": _sha(artifact)})

    contracts = tmp_path / "contracts"
    contracts.mkdir()
    hashes: dict[str, str] = {}
    for artifact_id, (filename, _kind) in PACKAGE_ARTIFACTS.items():
        path = contracts / filename
        if artifact_id == "genome_upgrade_v2_capability_contract":
            _json(path, GenomeV2P10CapabilityContract.v1().to_dict())
        else:
            _json(path, {"artifact_id": artifact_id})
        hashes[artifact_id] = _sha(path)
    capability_sha = GenomeV2P10CapabilityContract.v1().contract_sha256
    receipt = {
        "schema_version": "genome_v2_package_identity_receipt_v1",
        "authority_state": "package_candidate_complete",
        "ok": True,
        "package_contract_sha256": hashes[
            "genome_upgrade_v2_package_contract"
        ],
        "obligation_dictionary_sha256": hashes[
            "genome_upgrade_v2_obligation_dictionary"
        ],
        "invalidation_rules_sha256": hashes[
            "genome_upgrade_v2_invalidation_rules"
        ],
        "executable_contract_sha256": hashes[
            "genome_upgrade_v2_executable_contract"
        ],
        "identity": {
            "provider_id": "eiel-production-provider-v1",
            "runtime_profile_id": "eiel-production-runtime-v1",
            "cuda_artifact_sha256": _sha(artifact),
            "loaded_artifact_sha256": _sha(artifact),
            "capability_contract_sha256": capability_sha,
            "hereditary_capacity_schema_version": (
                PREREGISTRATION_SCHEMA_VERSION
            ),
            "hereditary_capacity_preregistration_sha256": (
                PREREGISTRATION_SHA256
            ),
        },
        "obligations": {},
    }
    receipt["package_id_sha256"] = package_identity_sha256_v1(receipt)
    receipt_path = tmp_path / "package.json"
    _json(receipt_path, receipt)
    prereg = tmp_path / "prereg.json"
    _json(prereg, preregistration_payload_v1())
    return {
        "output_dir": tmp_path / "output",
        "runtime_root": tmp_path / "runtime",
        "template": DEFAULT_TEMPLATE,
        "patch_artifact": artifact,
        "patch_manifest": manifest,
        "patch_qualification": qualification,
        "patch_reporting": reporting,
        "package_contracts_dir": contracts,
        "package_receipt": receipt_path,
        "hereditary_preregistration": prereg,
        "hereditary_preregistration_sha256": _sha(prereg),
    }


def test_materializes_exact_preregistered_compilable_campaign(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    result = materialize(**inputs)  # type: ignore[arg-type]
    output = Path(inputs["output_dir"])
    spec = output / f"{CAMPAIGN_ID}.yaml"
    document = yaml.safe_load(spec.read_text(encoding="utf-8"))
    plan = compile_campaign(load_campaign_spec(spec))

    assert result["plan_sha256"] == plan.plan_sha256
    assert result["spec_sha256"] == _sha(spec)
    assert document["checkpoints"]["milestones"] == [0, 3]
    assert document["execution"]["batch"]["maximum"] == 2
    assert document["parameters"]["genome_upgrade_v2.mutation_rate_q10"] == 20
    assert document["parameters"]["genome_upgrade_v2.mutation_step_q10"] == 64
    assert document["output"]["runtime_root"] == str(
        Path(inputs["runtime_root"]) / CAMPAIGN_ID
    )
    assert result["manifest_path"].endswith(
        f"{result['materialization_sha256']}.json"
    )


@pytest.mark.parametrize(
    "case",
    ("stale_prereg", "stale_package", "artifact_tamper", "overwrite"),
)
def test_rejects_stale_authority_tamper_and_overwrite(
    tmp_path: Path, case: str
) -> None:
    inputs = _inputs(tmp_path)
    if case == "stale_prereg":
        prereg = Path(inputs["hereditary_preregistration"])
        value = json.loads(prereg.read_text(encoding="ascii"))
        value["campaign_design"]["mutation_rate_q10"] = 21
        _json(prereg, value)
        inputs["hereditary_preregistration_sha256"] = _sha(prereg)
    elif case == "stale_package":
        receipt = Path(inputs["package_receipt"])
        value = json.loads(receipt.read_text(encoding="ascii"))
        value["identity"]["hereditary_capacity_preregistration_sha256"] = (
            "0" * 64
        )
        value["package_id_sha256"] = package_identity_sha256_v1(value)
        _json(receipt, value)
    elif case == "artifact_tamper":
        Path(inputs["patch_artifact"]).write_bytes(b"tampered")
    else:
        Path(inputs["output_dir"]).mkdir()

    with pytest.raises(MaterializationError):
        materialize(**inputs)  # type: ignore[arg-type]
    assert case == "overwrite" or not Path(inputs["output_dir"]).exists()
