from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from engine.campaigns.qualification import (
    EXECUTION_ROUTE_OBLIGATIONS_V1,
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
)
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    PREREGISTRATION_SHA256 as C1_PREREGISTRATION_SHA256,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
    CAMPAIGNS,
    PREREGISTRATION_SHA256 as E1_PREREGISTRATION_SHA256,
)
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (
    PREREGISTRATION_SHA256 as ENVIRONMENT_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as ENVIRONMENT_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    PREREGISTRATION_SHA256 as HEREDITARY_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as HEREDITARY_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    package_identity_sha256_v1,
    validate_checkpoint_authority_binding_v1,
)
from tools.scripts.genome_upgrade_v2_p10_authority_issue_v1 import (
    AUTHORITY_FILES,
    PREREGISTRATION_FILES,
    GenomeV2P10AuthorityIssueError,
    canonical_json_bytes,
    issue_authority_bindings_v1,
    issue_preregistrations_v1,
)
from tools.scripts.genome_upgrade_v2_p10_completion_collect_v1 import (
    SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
)


COMMIT = "1" * 40
ARTIFACT = "2" * 64
PATCH_ROUTE = "3" * 64
WDL_ROUTE = "4" * 64


def _write(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _route(
    path: Path,
    route: str,
    *,
    artifact: str = ARTIFACT,
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
            "qualification_plan_ids": ["exact-plan"],
            "case_ids": ["exact-case"],
            "artifact_sha256": artifact,
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


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    identity = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": COMMIT,
        "source_clean": True,
        "diagnostic_only": False,
        "no_fallback": True,
        "no_jit": True,
        "capability_contract_sha256": (
            GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        ),
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
        "loaded_artifact_sha256": ARTIFACT,
        "cuda_artifact_sha256": ARTIFACT,
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
        "identity": identity,
        "ok": True,
    }
    package["package_id_sha256"] = package_identity_sha256_v1(package)
    package_path = tmp_path / "package.json"
    patch_path = tmp_path / "patch.json"
    wdl_path = tmp_path / "wdl.json"
    _write(package_path, package)
    _route(patch_path, PATCH_ROUTE)
    _route(
        wdl_path,
        WDL_ROUTE,
        obligations=SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
    )
    return package_path, patch_path, wdl_path


def test_preregistration_issuer_writes_current_content_addresses(
    tmp_path: Path,
) -> None:
    paths = issue_preregistrations_v1(tmp_path / "preregistrations")

    expected = {
        PREREGISTRATION_FILES["c1"]: C1_PREREGISTRATION_SHA256,
        PREREGISTRATION_FILES["e1"]: E1_PREREGISTRATION_SHA256,
        PREREGISTRATION_FILES[
            "hereditary_capacity"
        ]: HEREDITARY_PREREGISTRATION_SHA256,
        PREREGISTRATION_FILES[
            "environment_switch"
        ]: ENVIRONMENT_PREREGISTRATION_SHA256,
    }
    for filename, digest in expected.items():
        assert hashlib.sha256(paths[filename].read_bytes()).hexdigest() == digest


def test_authority_issuer_projects_exact_package_and_routes(
    tmp_path: Path,
) -> None:
    package, patch, wdl = _inputs(tmp_path)
    paths = issue_authority_bindings_v1(
        package_receipt_path=package,
        patch_route_receipt_path=patch,
        wdl_route_receipt_path=wdl,
        output_directory=tmp_path / "authority",
    )

    patch_binding = validate_checkpoint_authority_binding_v1(
        json.loads(paths[AUTHORITY_FILES["patch_checkpoint"]].read_text())
    )
    wdl_binding = validate_checkpoint_authority_binding_v1(
        json.loads(paths[AUTHORITY_FILES["wdl_checkpoint"]].read_text())
    )
    e1 = json.loads(
        paths[AUTHORITY_FILES["e1_campaign_routes"]].read_text()
    )
    hereditary = json.loads(
        paths[AUTHORITY_FILES["hereditary_capacity"]].read_text()
    )
    assert patch_binding["execution_route_sha256"] == PATCH_ROUTE
    assert wdl_binding["execution_route_sha256"] == WDL_ROUTE
    assert e1 == {
        CAMPAIGNS[0]: patch_binding,
        CAMPAIGNS[1]: patch_binding,
        CAMPAIGNS[2]: wdl_binding,
    }
    assert hereditary == {
        name: patch_binding[name]
        for name in (
            "provider_id",
            "runtime_profile_id",
            "source_commit",
            "package_id_sha256",
            "package_receipt_sha256",
            "artifact_sha256",
            "execution_route_sha256",
        )
    }


def test_authority_issuer_rejects_stale_package_identity(
    tmp_path: Path,
) -> None:
    package, patch, wdl = _inputs(tmp_path)
    value = json.loads(package.read_text())
    value["identity"]["no_jit"] = False
    value["package_id_sha256"] = package_identity_sha256_v1(value)
    _write(package, value)

    with pytest.raises(
        GenomeV2P10AuthorityIssueError,
        match="current Genome V2 P1.0 authority",
    ):
        issue_authority_bindings_v1(
            package_receipt_path=package,
            patch_route_receipt_path=patch,
            wdl_route_receipt_path=wdl,
            output_directory=tmp_path / "authority",
        )


def test_authority_issuer_rejects_inexact_or_aliased_routes(
    tmp_path: Path,
) -> None:
    package, patch, wdl = _inputs(tmp_path)
    inexact = json.loads(wdl.read_text())
    inexact["reporting_results"]["exact"] = False
    _write(wdl, inexact)
    with pytest.raises(GenomeV2P10AuthorityIssueError, match="not exactly"):
        issue_authority_bindings_v1(
            package_receipt_path=package,
            patch_route_receipt_path=patch,
            wdl_route_receipt_path=wdl,
            output_directory=tmp_path / "inexact",
        )

    _route(
        wdl,
        PATCH_ROUTE,
        obligations=SINGLE_CELL_EXECUTION_ROUTE_OBLIGATIONS_V1,
    )
    with pytest.raises(GenomeV2P10AuthorityIssueError, match="must be distinct"):
        issue_authority_bindings_v1(
            package_receipt_path=package,
            patch_route_receipt_path=patch,
            wdl_route_receipt_path=wdl,
            output_directory=tmp_path / "aliased",
        )


def test_authority_issuer_rejects_symlinks_and_overwrite(
    tmp_path: Path,
) -> None:
    package, patch, wdl = _inputs(tmp_path)
    linked = tmp_path / "linked-package.json"
    linked.symlink_to(package)
    with pytest.raises(
        GenomeV2P10AuthorityIssueError,
        match="non-symlink",
    ):
        issue_authority_bindings_v1(
            package_receipt_path=linked,
            patch_route_receipt_path=patch,
            wdl_route_receipt_path=wdl,
            output_directory=tmp_path / "linked",
        )

    output = tmp_path / "preregistrations"
    issue_preregistrations_v1(output)
    with pytest.raises(
        GenomeV2P10AuthorityIssueError,
        match="already exists",
    ):
        issue_preregistrations_v1(output)
