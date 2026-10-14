from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
    PREREGISTRATION_SHA256 as C1_PREREGISTRATION_SHA256,
)
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
    ARMS,
    CAMPAIGNS,
    CAMPAIGN_FOUNDER_PAYLOADS,
    REPRESENTATIVE_SEED,
    SEEDS,
    TASK_SEED,
    E1AuthorityBindingV1,
    founder_payload_sha256s_v1,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    checkpoint_authority_binding_v1,
    package_identity_sha256_v1,
)
from tools.scripts.genome_upgrade_v2_e1_collect_v1 import (
    GenomeV2E1CollectorError,
    collect_e1_cell_evidence_v1,
    issue_e1_authority_v1,
    materialize_e1_cell_evidence_v1,
)
from tools.scripts.genome_upgrade_v2_e1_gate_v1 import cells_from_payload_v1


COMMIT = "2" * 40
HASH = "1" * 64


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _route_binding(route: str) -> dict[str, str]:
    return {
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": COMMIT,
        "executable_package_core_sha256": "3" * 64,
        "package_id_sha256": HASH,
        "package_receipt_sha256": HASH,
        "artifact_sha256": HASH,
        "manifest_sha256": "4" * 64,
        "state_schema_sha256": "5" * 64,
        "mechanic_contract_sha256": "6" * 64,
        "capability_contract_sha256": "7" * 64,
        "cancellation_abi_sha256": "8" * 64,
        "observability_schema_sha256": "9" * 64,
        "capacity_layout_contract_sha256": "a" * 64,
        "execution_route_sha256": route,
    }


def _issuance_inputs(
    tmp_path: Path,
    routes: dict[str, str],
) -> tuple[Path, dict[str, dict[str, str]], dict[str, object]]:
    identity = {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": COMMIT,
        "source_clean": True,
        "no_fallback": True,
        "no_jit": True,
        "executable_package_core_sha256": "3" * 64,
        "cuda_artifact_sha256": HASH,
        "cuda_manifest_sha256": "4" * 64,
        "authoritative_state_schema_sha256": "5" * 64,
        "mechanic_contract_sha256": "6" * 64,
        "capability_contract_sha256": "7" * 64,
        "cancellation_abi_sha256": "8" * 64,
        "observability_storage_budget_sha256": "9" * 64,
        "capacity_layout_contract_sha256": "a" * 64,
    }
    receipt = {
        "schema_version": "genome_v2_package_identity_receipt_v1",
        "authority_state": "package_candidate_complete",
        "identity": identity,
        "ok": True,
    }
    receipt["package_id_sha256"] = package_identity_sha256_v1(receipt)
    path = tmp_path / "package-identity.json"
    _write(path, receipt)
    import hashlib

    receipt_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    bindings = {
        campaign: checkpoint_authority_binding_v1(
            identity,
            package_id_sha256=str(receipt["package_id_sha256"]),
            package_receipt_sha256=receipt_sha,
            execution_route_sha256=route,
        )
        for campaign, route in routes.items()
    }
    c1 = {
        "schema_version": "genome_v2_p10_c1_causal_gate_v2",
        "preregistration_sha256": C1_PREREGISTRATION_SHA256,
        "decision": "C1-A",
    }
    c1["decision_sha256"] = content_sha256(c1)
    return path, bindings, c1


def _comparison_receipt() -> dict[str, object]:
    surface = {"terminal_checkpoint_digest": "d" * 64}
    return {
        "schema_version": "eiel_campaign_comparison_receipt_v1",
        "source_cell_id": "replaced",
        "compared_surfaces": ["terminal_checkpoint_digest"],
        "baseline": surface,
        "candidate": surface,
        "evidence_status": "terminal_checkpoint_and_observations_match",
        "match": True,
        "missing_surfaces": [],
        "mismatches": [],
        "first_difference": None,
    }


def _reporting_comparison_receipt(
    source_cell_id: str,
    route: str,
) -> dict[str, object]:
    digest = "d" * 64
    return {
        "schema_version": "eiel_campaign_reporting_parity_comparison_v1",
        "source_cell_id": source_cell_id,
        "evidence_mode": "enabled_disabled_exact_terminal_state",
        "compared_surfaces": ["terminal_checkpoint_digest"],
        "reporting_enabled_terminal_digest": digest,
        "reporting_disabled_terminal_digest": digest,
        "state_mutation_sentinel": digest,
        "match": True,
        "evidence_status": "reporting_parity_proven",
        "first_difference": None,
        "mismatches": [],
        "missing_surfaces": [],
        "execution_route_sha256": route,
        "branch_role": "reporting_parity",
    }


def _materialize_campaign(
    root: Path,
    campaign: str,
    route: str,
    *,
    artifact_sha256: str = HASH,
    package_receipt_sha256: str = HASH,
) -> tuple[str, dict[str, str], dict[str, str], str]:
    cells = []
    cell_identity: dict[str, tuple[str, int, str]] = {}
    for arm in ARMS:
        profile = CAMPAIGN_FOUNDER_PAYLOADS[campaign][arm]
        for seed in SEEDS:
            cell_id = f"cell-{arm}-{seed}"
            dimensions = {
                "ecology_seed": seed,
                "task_seed": TASK_SEED,
            }
            parameters = {
                "genome_upgrade_v2.founder_profile": profile,
            }
            if campaign == CAMPAIGNS[0]:
                dimensions["e1_arm"] = arm
                parameters["genome_upgrade_v2.e1_arm_label"] = arm
            else:
                dimensions["founder_profile"] = profile
            cell = {
                "cell_id": cell_id,
                "ordinal": len(cells),
                "dimensions": dimensions,
                "parameters": parameters,
                "seeds": {"ecology": seed, "task": TASK_SEED},
            }
            cells.append(cell)
            cell_identity[cell_id] = (arm, seed, profile)
    representative = {
        arm: f"cell-{arm}-{REPRESENTATIVE_SEED}" for arm in ARMS
    }
    branches = []
    for arm in ARMS:
        source = representative[arm]
        branches.extend(
            [
                {
                    "ordinal": len(branches),
                    "branch_id": f"repeat-{arm}",
                    "kind": "fresh_repeat",
                    "source_cell_id": source,
                    "repeat_ordinal": 1,
                    "split_generation": None,
                },
                {
                    "ordinal": len(branches) + 1,
                    "branch_id": f"resume-{arm}-g16",
                    "kind": "checkpoint_continuation",
                    "source_cell_id": source,
                    "repeat_ordinal": None,
                    "split_generation": 16,
                },
            ]
        )
    reporting_source = f"cell-{ARMS[0]}-{SEEDS[-1]}"
    branches.append(
        {
            "ordinal": len(branches),
            "branch_id": "reporting-disabled",
            "kind": "reporting_parity",
            "source_cell_id": reporting_source,
            "repeat_ordinal": None,
            "split_generation": None,
        }
    )
    unsigned_plan = {
        "schema_version": "eiel_campaign_execution_plan_v1",
        "campaign_id": campaign,
        "resolved_profile": "eiel-production-runtime-v1",
        "execution": {
            "generations": 32,
            "ticks_per_generation": 80,
        },
        "qualification_key": {
            "schema_version": "eiel_runtime_qualification_key_v1",
            "profile_id": "eiel-production-runtime-v1",
            "backend": "cuda_resident",
            "artifact_hashes": {
                "cuda_extension": artifact_sha256,
                "genome_upgrade_v2_package_receipt": (
                    package_receipt_sha256
                ),
            },
        },
        "cells": cells,
        "physical_batches": [
            {
                "ordinal": 0,
                "batch_id": "batch-0000",
                "cell_ids": [cell["cell_id"] for cell in cells],
            }
        ],
        "comparisons": {
            "reporting_parity": {
                "required": True,
                "accepted_evidence_modes": [
                    "enabled_disabled_exact_terminal_state",
                    "provider_observational_contract_with_state_mutation_sentinel",
                ],
            },
            "branches": branches,
        },
    }
    plan_sha = content_sha256(unsigned_plan)
    plan = {**unsigned_plan, "plan_sha256": plan_sha}
    _write(root / "execution_plan.json", plan)

    checkpoint_hashes = {
        cell_id: content_sha256(
            {"campaign": campaign, "cell_id": cell_id, "checkpoint": 32}
        )
        for cell_id in cell_identity
    }
    native_metrics = []
    runner_receipts = []
    public_metrics = []
    for cell_id, (arm, _seed, profile) in cell_identity.items():
        metric_values = {
            "population.terminal_alive": 64 + (arm == "treatment"),
            "population.successful_births_cumulative": 32,
            "genome_v2.expression_observation_count": 64,
            "genome_v2.context_observation_count": 64,
            "genome_v2.cost_observation_count": 64,
            "genome_v2.cap_observation_count": 64,
        }
        generation_metrics = [
            {
                "generation": generation,
                "alive": 64,
                "successful_births": 1,
            }
            for generation in range(32)
        ]
        native_metrics.append(
            {
                "schema_version": "eiel_genome_v2_e1_native_cell_metrics_v1",
                "cell_id": cell_id,
                "metrics": metric_values,
                "generation_metrics": generation_metrics,
            }
        )
        runner_receipts.append(
            {
                "schema_version": "eiel_patch_fixed_runner_factory_v1",
                "cell_id": cell_id,
                "genome": {
                    "founder_profile_id": profile,
                    "founder_profile_sha256": (
                        founder_payload_sha256s_v1()[profile]
                    ),
                },
            }
        )
        public_metrics.append(
            {"cell_id": cell_id, "metrics": metric_values}
        )
    batch_path = root / "batches/batch-0000/manifest.json"
    batch = {
        "schema_version": "eiel_campaign_batch_manifest_v1",
        "campaign_id": campaign,
        "plan_sha256": plan_sha,
        "profile_id": "eiel-production-runtime-v1",
        "provider_id": "eiel-production-provider-v1",
        "execution_route_sha256": route,
        "batch_id": "batch-0000",
        "cell_ids": list(cell_identity),
        "status": "completed",
        "metrics_rows": native_metrics,
        "result": {
            "fallback_used": False,
            "native_execution_receipts": [
                {
                    "execution_device": "cuda",
                    "explicit_native_no_fallback": True,
                    "hidden_fallback": False,
                    "runtime_jit": False,
                    "start_generation": 0,
                    "end_generation": 32,
                }
            ],
            "runner_receipts": runner_receipts,
            "checkpoint_catalog": {
                "32": {
                    "generation": 32,
                    "execution_route_sha256": route,
                    "canonical_checkpoint_storage_digests": checkpoint_hashes,
                    "mechanics_checkpoint_digests": {
                        cell_id: content_sha256(
                            {
                                "campaign": campaign,
                                "cell_id": cell_id,
                                "mechanics": 32,
                            }
                        )
                        for cell_id in cell_identity
                    },
                }
            },
        },
    }
    _write(batch_path, batch)
    _write(
        root / "campaign_manifest.json",
        {
            "schema_version": "eiel_campaign_execution_manifest_v1",
            "campaign_id": campaign,
            "plan_sha256": plan_sha,
            "profile_id": "eiel-production-runtime-v1",
            "provider_id": "eiel-production-provider-v1",
            "execution_route_sha256": route,
            "status": "completed",
            "batches": [
                {
                    "batch_id": "batch-0000",
                    "status": "completed",
                    "manifest_path": str(batch_path),
                }
            ],
        },
    )
    receipt_rows = []
    for cell in cells:
        cell_id = cell["cell_id"]
        cell_root = root / "cells" / cell_id
        cell_manifest = cell_root / "cell_manifest.json"
        completion = cell_root / "completion.json"
        _write(
            cell_manifest,
            {
                "schema_version": "eiel_campaign_cell_manifest_v1",
                "campaign_id": campaign,
                "plan_sha256": plan_sha,
                "profile_id": "eiel-production-runtime-v1",
                "provider_id": "eiel-production-provider-v1",
                "execution_route_sha256": route,
                "batch_id": "batch-0000",
                "batch_manifest_path": str(batch_path),
                "cell": cell,
            },
        )
        _write(
            completion,
            {
                "schema_version": "eiel_campaign_cell_completion_v1",
                "campaign_id": campaign,
                "plan_sha256": plan_sha,
                "cell_id": cell_id,
                "status": "completed",
                "terminal_checkpoint_generation": 32,
                "execution_route_sha256": route,
            },
        )
        receipt_rows.append(
            {
                "cell_id": cell_id,
                "cell_manifest_path": str(cell_manifest),
                "completion_path": str(completion),
            }
        )
    _write(
        root / "cell_receipts.json",
        {
            "schema_version": "eiel_campaign_cell_receipt_index_v1",
            "campaign_id": campaign,
            "plan_sha256": plan_sha,
            "completed_cell_count": 12,
            "rows": receipt_rows,
        },
    )
    _write(
        root / "cell_metrics.json",
        {
            "schema_version": "eiel_campaign_cell_metrics_v1",
            "campaign_id": campaign,
            "plan_sha256": plan_sha,
            "rows": public_metrics,
        },
    )

    comparison_summaries = []
    comparison_manifests: dict[str, dict[str, object]] = {}
    for branch in branches:
        branch_id = branch["branch_id"]
        comparison = (
            _reporting_comparison_receipt(
                str(branch["source_cell_id"]),
                route,
            )
            if branch["kind"] == "reporting_parity"
            else _comparison_receipt()
        )
        comparison["source_cell_id"] = branch["source_cell_id"]
        manifest = {
            "schema_version": "eiel_campaign_comparison_manifest_v1",
            "campaign_id": campaign,
            "plan_sha256": plan_sha,
            "profile_id": "eiel-production-runtime-v1",
            "provider_id": "eiel-production-provider-v1",
            "execution_route_sha256": route,
            "branch_id": branch_id,
            "kind": branch["kind"],
            "source_cell_id": branch["source_cell_id"],
            "split_generation": branch["split_generation"],
            "repeat_ordinal": branch["repeat_ordinal"],
            "status": "completed",
            "comparison": comparison,
        }
        path = root / "comparisons" / str(branch_id) / "manifest.json"
        _write(path, manifest)
        comparison_manifests[str(branch_id)] = manifest
        comparison_summaries.append(
            {
                "branch_id": branch_id,
                "kind": branch["kind"],
                "source_cell_id": branch["source_cell_id"],
                "status": "completed",
                "manifest_path": str(path),
            }
        )
    _write(
        root / "comparisons_manifest.json",
        {
            "schema_version": "eiel_campaign_comparisons_manifest_v1",
            "campaign_id": campaign,
            "plan_sha256": plan_sha,
            "profile_id": "eiel-production-runtime-v1",
            "provider_id": "eiel-production-provider-v1",
            "execution_route_sha256": route,
            "status": "completed",
            "branches": comparison_summaries,
        },
    )
    reporting = comparison_manifests["reporting-disabled"]["comparison"]
    reporting_receipt = {
        "schema_version": "eiel_campaign_reporting_parity_receipt_v1",
        "campaign_id": campaign,
        "plan_sha256": plan_sha,
        "profile_id": "eiel-production-runtime-v1",
        "provider_id": "eiel-production-provider-v1",
        "execution_route_sha256": route,
        "comparison_branch_id": "reporting-disabled",
        "comparison_receipt_sha256": content_sha256(reporting),
        "comparison_first_difference": None,
        "qualification_evidence_status": "reporting_parity_proven",
        "status": "supported",
        "evidence_mode": "enabled_disabled_exact_terminal_state",
        "observational_contract": (
            "reporting_disabled_skips_compact_materialization_only_v1"
        ),
        "state_mutation_sentinel": "d" * 64,
        "reason_code": None,
        "reason": None,
    }
    _write(root / "reporting_parity_receipt.json", reporting_receipt)
    parity_hashes = {}
    for arm in ARMS:
        source = representative[arm]
        fresh = comparison_manifests[f"repeat-{arm}"]
        continuation = comparison_manifests[f"resume-{arm}-g16"]
        parity_hashes[
            f"{campaign}:{arm}:{REPRESENTATIVE_SEED}"
        ] = content_sha256(
            {
                "schema_version": (
                    "genome_v2_p10_e1_representative_parity_receipt_v1"
                ),
                "campaign_id": campaign,
                "cell_id": source,
                "arm": arm,
                "ecology_seed": REPRESENTATIVE_SEED,
                "execution_route_sha256": route,
                "fresh_repeat": {
                    "manifest_sha256": content_sha256(fresh),
                    "comparison_receipt_sha256": content_sha256(
                        fresh["comparison"]
                    ),
                },
                "continuation_g16": {
                    "manifest_sha256": content_sha256(continuation),
                    "comparison_receipt_sha256": content_sha256(
                        continuation["comparison"]
                    ),
                },
            }
        )
    authority_checkpoints = {
        f"{campaign}:{arm}:{seed}": checkpoint_hashes[f"cell-{arm}-{seed}"]
        for arm in ARMS
        for seed in SEEDS
    }
    return (
        plan_sha,
        authority_checkpoints,
        parity_hashes,
        content_sha256(reporting_receipt),
    )


def _issuance_matrix(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, dict[str, str]],
    dict[str, object],
    list[Path],
]:
    routes = {
        CAMPAIGNS[0]: "b" * 64,
        CAMPAIGNS[1]: "c" * 64,
        CAMPAIGNS[2]: "d" * 64,
    }
    package, bindings, c1 = _issuance_inputs(tmp_path, routes)
    import hashlib

    package_sha = hashlib.sha256(package.read_bytes()).hexdigest()
    roots = []
    for campaign in CAMPAIGNS:
        root = tmp_path / f"issued-{campaign}"
        roots.append(root)
        _materialize_campaign(
            root,
            campaign,
            routes[campaign],
            artifact_sha256=HASH,
            package_receipt_sha256=package_sha,
        )
    return package, bindings, c1, roots


@pytest.fixture
def matrix(
    tmp_path: Path,
) -> tuple[E1AuthorityBindingV1, list[Path]]:
    routes = {
        CAMPAIGNS[0]: "b" * 64,
        CAMPAIGNS[1]: "c" * 64,
        CAMPAIGNS[2]: "d" * 64,
    }
    roots = []
    plans = {}
    checkpoints = {}
    parity = {}
    reporting_parity = {}
    for campaign in CAMPAIGNS:
        root = tmp_path / campaign
        roots.append(root)
        (
            plan,
            campaign_checkpoints,
            campaign_parity,
            campaign_reporting_parity,
        ) = _materialize_campaign(root, campaign, routes[campaign])
        plans[campaign] = plan
        checkpoints.update(campaign_checkpoints)
        parity.update(campaign_parity)
        reporting_parity[campaign] = campaign_reporting_parity
    authority = E1AuthorityBindingV1(
        source_commit=COMMIT,
        package_id_sha256=HASH,
        package_receipt_sha256=HASH,
        artifact_sha256=HASH,
        execution_route_sha256s=routes,
        checkpoint_authority_bindings={
            campaign: _route_binding(route)
            for campaign, route in routes.items()
        },
        c1_preregistration_sha256=C1_PREREGISTRATION_SHA256,
        c1_decision_sha256="e" * 64,
        c1_decision="C1-A",
        campaign_plan_sha256s=plans,
        founder_payload_sha256s=founder_payload_sha256s_v1(),
        checkpoint_sha256s=checkpoints,
        parity_receipt_sha256s=parity,
        reporting_parity_receipt_sha256s=reporting_parity,
    )
    return authority, roots


def test_collects_exact_36_cell_gate_payload(
    matrix: tuple[E1AuthorityBindingV1, list[Path]],
    tmp_path: Path,
) -> None:
    authority, roots = matrix
    output = tmp_path / "cell-evidence.json"
    rows = materialize_e1_cell_evidence_v1(
        authority=authority,
        run_roots=roots,
        output_path=output,
    )

    parsed = cells_from_payload_v1(json.loads(output.read_text()))
    assert len(rows) == len(parsed) == 36
    assert all(row.no_fallback and row.no_jit for row in parsed)
    representative = [
        row for row in parsed if row.ecology_seed == REPRESENTATIVE_SEED
    ]
    assert len(representative) == 6
    assert all(
        row.fresh_repeat_exact and row.continuation_g16_exact
        for row in representative
    )
    reporting_path = roots[0] / "reporting_parity_receipt.json"
    reporting = json.loads(reporting_path.read_text(encoding="utf-8"))
    reporting["state_mutation_sentinel"] = "e" * 64
    _write(reporting_path, reporting)
    with pytest.raises(
        GenomeV2E1CollectorError,
        match="reporting parity receipt identity join mismatch",
    ):
        collect_e1_cell_evidence_v1(
            authority=authority,
            run_roots=roots,
        )


def test_collects_segmented_runner_receipts_and_flat_generation_rows(
    matrix: tuple[E1AuthorityBindingV1, list[Path]],
) -> None:
    authority, roots = matrix
    for root in roots:
        path = root / "batches/batch-0000/manifest.json"
        batch = json.loads(path.read_text(encoding="utf-8"))
        nested = batch["metrics_rows"]
        batch["metrics_rows"] = [
            {**row, "cell_id": metrics["cell_id"]}
            for metrics in nested
            for row in metrics["generation_metrics"]
        ]
        batch["result"]["runner_receipts"] *= 2
        _write(path, batch)

    rows = collect_e1_cell_evidence_v1(
        authority=authority,
        run_roots=roots,
    )

    assert len(rows) == 36


def test_issues_authority_from_package_routes_c1_and_runtime_only(
    tmp_path: Path,
) -> None:
    for spec_name in (
        "genome_v2_e1_neutral_no_influence_g32.yaml",
        "genome_v2_e1_patch_pleiotropic_g32.yaml",
        "genome_v2_e1_wdl_hazard_regulatory_g32.yaml",
    ):
        compiled = compile_campaign(
            load_campaign_spec(Path("_spec/campaigns/v1") / spec_name)
        ).payload()
        kinds = [
            branch["kind"]
            for branch in compiled["comparisons"]["branches"]
        ]
        assert len(kinds) == 5
        assert kinds.count("fresh_repeat") == 2
        assert kinds.count("checkpoint_continuation") == 2
        assert kinds.count("reporting_parity") == 1
    package, bindings, c1, roots = _issuance_matrix(tmp_path)

    issued = issue_e1_authority_v1(
        package_receipt_path=package,
        route_bindings=bindings,
        c1_decision=c1,
        run_roots=roots,
    )

    assert set(issued.campaign_plan_sha256s) == set(CAMPAIGNS)
    assert len(issued.checkpoint_sha256s) == 36
    assert len(issued.parity_receipt_sha256s) == 6
    assert set(issued.reporting_parity_receipt_sha256s) == set(CAMPAIGNS)
    assert len(
        collect_e1_cell_evidence_v1(authority=issued, run_roots=roots)
    ) == 36


def test_authority_issuance_rejects_operator_invented_route_binding(
    tmp_path: Path,
) -> None:
    package, bindings, c1, roots = _issuance_matrix(tmp_path)
    bindings[CAMPAIGNS[0]]["artifact_sha256"] = "f" * 64

    with pytest.raises(
        GenomeV2E1CollectorError,
        match="differs from package authority",
    ):
        issue_e1_authority_v1(
            package_receipt_path=package,
            route_bindings=bindings,
            c1_decision=c1,
            run_roots=roots,
        )


def test_fails_closed_on_campaign_route_identity_drift(
    matrix: tuple[E1AuthorityBindingV1, list[Path]],
) -> None:
    authority, roots = matrix
    path = roots[0] / "campaign_manifest.json"
    value = json.loads(path.read_text())
    value["execution_route_sha256"] = "f" * 64
    _write(path, value)

    with pytest.raises(GenomeV2E1CollectorError, match="identity join mismatch"):
        collect_e1_cell_evidence_v1(authority=authority, run_roots=roots)


def test_fails_closed_when_representative_parity_is_missing(
    matrix: tuple[E1AuthorityBindingV1, list[Path]],
) -> None:
    authority, roots = matrix
    path = roots[1] / "comparisons_manifest.json"
    value = json.loads(path.read_text())
    value["branches"].pop()
    _write(path, value)

    with pytest.raises(
        GenomeV2E1CollectorError,
        match="representative comparison matrix",
    ):
        collect_e1_cell_evidence_v1(authority=authority, run_roots=roots)


def test_fails_closed_on_execution_plan_content_tamper(
    matrix: tuple[E1AuthorityBindingV1, list[Path]],
) -> None:
    authority, roots = matrix
    path = roots[2] / "execution_plan.json"
    value = json.loads(path.read_text())
    value["execution"]["ticks_per_generation"] = 81
    _write(path, value)

    with pytest.raises(GenomeV2E1CollectorError, match="content hash mismatch"):
        collect_e1_cell_evidence_v1(authority=authority, run_roots=roots)
