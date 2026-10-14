from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.campaigns.qualification import (
    EXECUTION_ROUTE_OBLIGATIONS_V1,
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    GenomeV2P10CapabilityContract,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    PREREGISTRATION_SHA256 as C1_PREREGISTRATION_SHA256,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
    ARMS,
    CAMPAIGNS,
    PREREGISTRATION_SHA256 as E1_PREREGISTRATION_SHA256,
    REPRESENTATIVE_SEED,
    SEEDS,
    founder_payload_sha256s_v1,
)
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (
    PREREGISTRATION_SHA256 as ENVIRONMENT_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as ENVIRONMENT_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    PREREGISTRATION_SHA256 as HEREDITARY_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as HEREDITARY_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_p10_completion_gate_v1 import (
    OUTPUT_NAMES,
    reduce_p10_completion_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    checkpoint_authority_binding_v1,
    package_identity_sha256_v1,
)
from tools.scripts.genome_upgrade_v2_p10_completion_collect_v1 import (
    CompletionCollectorError,
    SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
    canonical_json_bytes,
    collect_completion_evidence_v1,
    main,
)


COMMIT = "1" * 40
ARTIFACT = "2" * 64
PATCH_ROUTE = "3" * 64
WDL_ROUTE = "4" * 64


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="ascii")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    value["decision_sha256"] = content_sha256(value)
    _write(path, value)
    return value


def _route(
    path: Path,
    route: str,
    *,
    obligations: frozenset[str] = EXECUTION_ROUTE_OBLIGATIONS_V1,
) -> None:
    _write(
        path,
        {
            "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
            "authority_state": "route_qualified",
            "execution_route_sha256": route,
            "provider_id": "eiel-production-provider-v1",
            "profile_id": "eiel-production-runtime-v1",
            "covered_obligations": sorted(obligations),
            "qualification_plan_ids": ["genome-v2-terminal-canary"],
            "case_ids": ["exact-case"],
            "artifact_sha256": ARTIFACT,
            "checkpoint_results": {"exact": True, "case_count": 1},
            "reporting_results": {"exact": True, "case_count": 1},
            "exact_comparisons": [
                {
                    "comparison_id": "exact",
                    "match": True,
                    "first_difference": None,
                    "execution_route_sha256": route,
                }
            ],
        },
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    capability_path = tmp_path / "capability.json"
    capability = GenomeV2P10CapabilityContract.v1().to_dict()
    _write(capability_path, capability)
    identity = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": COMMIT,
        "source_clean": True,
        "diagnostic_only": False,
        "no_fallback": True,
        "no_jit": True,
        "capability_contract_sha256": capability["contract_sha256"],
        "causal_gate_preregistration_sha256": C1_PREREGISTRATION_SHA256,
        "ecological_e1_preregistration_sha256": E1_PREREGISTRATION_SHA256,
        "hereditary_capacity_schema_version": HEREDITARY_SCHEMA_VERSION,
        "hereditary_capacity_preregistration_sha256": (
            HEREDITARY_PREREGISTRATION_SHA256
        ),
        "environment_switch_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "environment_switch_preregistration_sha256": (
            ENVIRONMENT_PREREGISTRATION_SHA256
        ),
        "cuda_artifact_sha256": ARTIFACT,
        "loaded_artifact_sha256": ARTIFACT,
        "executable_package_core_sha256": "5" * 64,
        "cuda_manifest_sha256": "6" * 64,
        "authoritative_state_schema_sha256": "7" * 64,
        "mechanic_contract_sha256": "8" * 64,
        "cancellation_abi_sha256": "9" * 64,
        "observability_storage_budget_sha256": "a" * 64,
        "capacity_layout_contract_sha256": "b" * 64,
    }
    package = {
        "schema_version": "genome_v2_package_identity_receipt_v1",
        "authority_state": "package_candidate_complete",
        "ok": True,
        "identity": identity,
    }
    package["package_id_sha256"] = package_identity_sha256_v1(package)
    package_path = tmp_path / "package.json"
    _write(package_path, package)
    package_receipt_sha = _file_sha(package_path)

    patch_path = tmp_path / "patch-route.json"
    wdl_path = tmp_path / "wdl-route.json"
    _route(patch_path, PATCH_ROUTE)
    _route(
        wdl_path,
        WDL_ROUTE,
        obligations=SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
    )
    c1_path = tmp_path / "c1.json"
    c1 = _decision(
        c1_path,
        {
            "schema_version": "genome_v2_p10_c1_decision_v2",
            "preregistration_sha256": C1_PREREGISTRATION_SHA256,
            "decision": "C1-A",
            "reasons": [],
            "pairs": [],
            "family_support": [],
        },
    )
    campaign_routes = {
        CAMPAIGNS[0]: PATCH_ROUTE,
        CAMPAIGNS[1]: PATCH_ROUTE,
        CAMPAIGNS[2]: WDL_ROUTE,
    }
    bindings = {
        campaign: checkpoint_authority_binding_v1(
            identity,
            package_id_sha256=package["package_id_sha256"],
            package_receipt_sha256=package_receipt_sha,
            execution_route_sha256=campaign_routes[campaign],
        )
        for campaign in CAMPAIGNS
    }
    e1_authority = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "source_commit": COMMIT,
        "package_id_sha256": package["package_id_sha256"],
        "package_receipt_sha256": package_receipt_sha,
        "artifact_sha256": ARTIFACT,
        "execution_route_sha256s": campaign_routes,
        "checkpoint_authority_bindings": bindings,
        "c1_preregistration_sha256": C1_PREREGISTRATION_SHA256,
        "c1_decision_sha256": c1["decision_sha256"],
        "c1_decision": "C1-A",
        "campaign_plan_sha256s": {
            campaign: "c" * 64 for campaign in CAMPAIGNS
        },
        "founder_payload_sha256s": founder_payload_sha256s_v1(),
        "checkpoint_sha256s": {
            f"{campaign}:{arm}:{seed}": "d" * 64
            for campaign in CAMPAIGNS
            for arm in ARMS
            for seed in SEEDS
        },
        "parity_receipt_sha256s": {
            f"{campaign}:{arm}:{REPRESENTATIVE_SEED}": "e" * 64
            for campaign in CAMPAIGNS
            for arm in ARMS
        },
        "reporting_parity_receipt_sha256s": {
            campaign: "f" * 64 for campaign in CAMPAIGNS
        },
        "e1_preregistration_sha256": E1_PREREGISTRATION_SHA256,
    }
    e1_authority_path = tmp_path / "e1-authority.json"
    _write(e1_authority_path, e1_authority)
    e1_path = tmp_path / "e1.json"
    _decision(
        e1_path,
        {
            "schema_version": "genome_v2_p10_e1_ecological_gate_v1",
            "preregistration_sha256": E1_PREREGISTRATION_SHA256,
            "decision": "E1-A",
            "cell_count": 36,
            "pair_count": 18,
            "paired_deltas": [],
            "reasons": [],
            "claims_boundary": "bounded",
        },
    )
    common_authority = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "source_commit": COMMIT,
        "package_id_sha256": package["package_id_sha256"],
        "package_receipt_sha256": package_receipt_sha,
        "artifact_sha256": ARTIFACT,
    }
    hereditary_path = tmp_path / "hereditary.json"
    _decision(
        hereditary_path,
        {
            "schema_version": HEREDITARY_SCHEMA_VERSION,
            "preregistration_sha256": HEREDITARY_PREREGISTRATION_SHA256,
            "authority": {
                **common_authority,
                "execution_route_sha256": PATCH_ROUTE,
            },
            "decision": "HEREDITARY-CAPACITY-A",
            "reasons": [],
            "repeat_summaries": [],
            "claims": {"hereditary_capacity_supported": True},
        },
    )
    environment_path = tmp_path / "environment.json"
    _decision(
        environment_path,
        {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "preregistration_sha256": ENVIRONMENT_PREREGISTRATION_SHA256,
            "authority": {
                **common_authority,
                "execution_route_sha256": WDL_ROUTE,
            },
            "decision": "ENVIRONMENT-SWITCH-A",
            "reasons": [],
            "switch_summaries": [],
            "claims": {"environment_switch_supported": True},
        },
    )
    return {
        "package_receipt_path": package_path,
        "patch_route_receipt_path": patch_path,
        "wdl_route_receipt_path": wdl_path,
        "c1_decision_path": c1_path,
        "e1_authority_path": e1_authority_path,
        "e1_decision_path": e1_path,
        "hereditary_capacity_decision_path": hereditary_path,
        "environment_switch_decision_path": environment_path,
        "capability_contract_path": capability_path,
    }


def test_collects_deterministic_terminal_evidence(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    first = collect_completion_evidence_v1(**paths)
    second = collect_completion_evidence_v1(**paths)

    assert first == second
    artifacts = reduce_p10_completion_v1(first)
    assert artifacts[OUTPUT_NAMES["audit"]]["decision"] == "P10-A"
    assert len(artifacts) == 5


def test_cli_writes_one_canonical_non_overwriting_file(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    output = tmp_path / "completion-evidence.json"
    args: list[str] = []
    names = {
        "package_receipt_path": "--package-receipt",
        "patch_route_receipt_path": "--patch-route-receipt",
        "wdl_route_receipt_path": "--wdl-route-receipt",
        "c1_decision_path": "--c1-decision",
        "e1_authority_path": "--e1-authority",
        "e1_decision_path": "--e1-decision",
        "hereditary_capacity_decision_path": (
            "--hereditary-capacity-decision"
        ),
        "environment_switch_decision_path": "--environment-switch-decision",
        "capability_contract_path": "--capability-contract",
    }
    for key, flag in names.items():
        args.extend((flag, str(paths[key])))
    args.extend(("--output", str(output)))

    assert main(args) == 0
    value = json.loads(output.read_text(encoding="ascii"))
    assert output.read_bytes() == canonical_json_bytes(value)
    assert main(args) == 2


@pytest.mark.parametrize(
    ("path_key", "mutation"),
    [
        (
            "package_receipt_path",
            lambda value: value["identity"].__setitem__(
                "cuda_artifact_sha256", "0" * 64
            ),
        ),
        (
            "patch_route_receipt_path",
            lambda value: value.__setitem__(
                "authority_state", "qualification_only"
            ),
        ),
        (
            "e1_decision_path",
            lambda value: value.__setitem__("decision", "E1-C"),
        ),
    ],
)
def test_rejects_tampered_or_nonterminal_inputs(
    tmp_path: Path,
    path_key: str,
    mutation: Any,
) -> None:
    paths = _inputs(tmp_path)
    path = paths[path_key]
    value = json.loads(path.read_text(encoding="ascii"))
    mutation(value)
    _write(path, value)

    with pytest.raises(CompletionCollectorError):
        collect_completion_evidence_v1(**paths)


def test_rejects_content_addressed_mixed_terminal_authority(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    path = paths["environment_switch_decision_path"]
    value = json.loads(path.read_text(encoding="ascii"))
    value["authority"]["package_id_sha256"] = "0" * 64
    value.pop("decision_sha256")
    value["decision_sha256"] = content_sha256(value)
    _write(path, value)

    with pytest.raises(
        CompletionCollectorError, match="mixed with another package"
    ):
        collect_completion_evidence_v1(**paths)
