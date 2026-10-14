from __future__ import annotations

from dataclasses import replace
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
    DECISION_A,
    DECISION_B,
    DECISION_C,
    DECISION_D,
    OBSERVATION_COUNTS,
    PREREGISTRATION_SHA256,
    PRIMARY_METRICS,
    REPRESENTATIVE_SEED,
    SEEDS,
    TASK_SEED,
    E1AuthorityBindingV1,
    E1CellEvidenceV1,
    evaluate_e1_v1,
    founder_payload_sha256s_v1,
    preregistration_payload_v1,
)
from engine.perf.cuda_genome_upgrade_v2_state_bank import (
    resolve_genome_v2_founder_profile_v1,
)
from tools.scripts.genome_upgrade_v2_e1_gate_v1 import (
    GenomeV2E1RunnerError,
    materialize_e1_outputs_v1,
)


HASH = "1" * 64
COMMIT = "2" * 40


def _route_binding(route_sha256: str = HASH) -> dict[str, str]:
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
        "execution_route_sha256": route_sha256,
    }


def _c1_decision() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "genome_v2_p10_c1_causal_gate_v2",
        "preregistration_sha256": C1_PREREGISTRATION_SHA256,
        "decision": "C1-A",
    }
    return {**value, "decision_sha256": content_sha256(value)}


def _checkpoint_hashes() -> dict[str, str]:
    return {
        f"{campaign}:{arm}:{seed}": content_sha256(
            {"campaign": campaign, "arm": arm, "seed": seed}
        )
        for campaign in CAMPAIGNS
        for arm in ARMS
        for seed in SEEDS
    }


def _authority() -> E1AuthorityBindingV1:
    c1 = _c1_decision()
    routes = {
        CAMPAIGNS[0]: "b" * 64,
        CAMPAIGNS[1]: "b" * 64,
        CAMPAIGNS[2]: "c" * 64,
    }
    return E1AuthorityBindingV1(
        source_commit=COMMIT,
        package_id_sha256=HASH,
        package_receipt_sha256=HASH,
        artifact_sha256=HASH,
        execution_route_sha256s=routes,
        checkpoint_authority_bindings={
            campaign: _route_binding(route_sha256)
            for campaign, route_sha256 in routes.items()
        },
        c1_preregistration_sha256=C1_PREREGISTRATION_SHA256,
        c1_decision_sha256=str(c1["decision_sha256"]),
        c1_decision="C1-A",
        campaign_plan_sha256s={
            campaign: content_sha256({"campaign": campaign})
            for campaign in CAMPAIGNS
        },
        founder_payload_sha256s=founder_payload_sha256s_v1(),
        checkpoint_sha256s=_checkpoint_hashes(),
        parity_receipt_sha256s={
            f"{campaign}:{arm}:{REPRESENTATIVE_SEED}": content_sha256(
                {
                    "parity": campaign,
                    "arm": arm,
                    "seed": REPRESENTATIVE_SEED,
                }
            )
            for campaign in CAMPAIGNS
            for arm in ARMS
        },
        reporting_parity_receipt_sha256s={
            campaign: content_sha256(
                {"reporting_parity": campaign}
            )
            for campaign in CAMPAIGNS
        },
    )


def _cells(authority: E1AuthorityBindingV1) -> list[E1CellEvidenceV1]:
    cells = []
    for campaign in CAMPAIGNS:
        for arm in ARMS:
            for ordinal, seed in enumerate(SEEDS):
                metrics = {
                    metric: 10 for metric in PRIMARY_METRICS
                } | {
                    observation: 1 for observation in OBSERVATION_COUNTS
                }
                if arm == "treatment" and ordinal < 4:
                    if campaign == CAMPAIGNS[1]:
                        metrics["population.terminal_alive"] += 1
                    elif campaign == CAMPAIGNS[2]:
                        metrics[
                            "population.successful_births_cumulative"
                        ] += 2
                trajectory = content_sha256(
                    {
                        "campaign": campaign,
                        "seed": seed,
                        "arm": (
                            "joined"
                            if campaign == CAMPAIGNS[0]
                            else arm
                        ),
                    }
                )
                payload_id = CAMPAIGN_FOUNDER_PAYLOADS[campaign][arm]
                cells.append(
                    E1CellEvidenceV1(
                        campaign_id=campaign,
                        arm=arm,
                        ecology_seed=seed,
                        task_seed=TASK_SEED,
                        authority_sha256=authority.authority_sha256(),
                        campaign_plan_sha256=(
                            authority.campaign_plan_sha256s[campaign]
                        ),
                        execution_route_sha256=(
                            authority.execution_route_sha256s[campaign]
                        ),
                        founder_payload_id=payload_id,
                        founder_payload_sha256=(
                            authority.founder_payload_sha256s[payload_id]
                        ),
                        checkpoint_sha256=authority.checkpoint_sha256s[
                            f"{campaign}:{arm}:{seed}"
                        ],
                        execution_receipt_sha256=content_sha256(
                            {"execution": campaign, "arm": arm, "seed": seed}
                        ),
                        trajectory_sha256=trajectory,
                        route_parity_receipt_sha256=content_sha256(
                            {"parity": campaign, "arm": arm, "seed": seed}
                        ),
                        fresh_repeat_exact=seed == REPRESENTATIVE_SEED,
                        continuation_g16_exact=seed == REPRESENTATIVE_SEED,
                        no_fallback=True,
                        no_jit=True,
                        metrics=metrics,
                    )
                )
    return cells


def test_preregistration_is_content_addressed_and_freezes_matrix() -> None:
    payload = preregistration_payload_v1()
    assert content_sha256(payload) == PREREGISTRATION_SHA256
    assert payload["extent"] == {
        "generations": 32,
        "ticks_per_generation": 80,
        "paired_seeds": list(SEEDS),
        "task_seed": TASK_SEED,
        "logical_cells": 36,
        "mutation_rate_q10": 0,
    }
    assert len(payload["campaigns"]) == 3
    assert payload["representative_parity"]["required_cells"] == 6
    assert founder_payload_sha256s_v1() == {
        profile_id: resolve_genome_v2_founder_profile_v1(profile_id)[
            "payload_sha256"
        ]
        for profile_id in (
            "neutral",
            "pleiotropic_tradeoffs",
            "regulatory_activation_inhibition",
        )
    }


def test_three_frozen_campaign_specs_compile_to_exact_36_cell_matrix() -> None:
    root = Path(__file__).parents[1] / "_spec" / "campaigns" / "v1"
    plans = [
        compile_campaign(
            load_campaign_spec(root / f"{campaign.replace('-', '_')}.yaml")
        )
        for campaign in CAMPAIGNS
    ]
    assert sum(len(plan.cells) for plan in plans) == 36
    assert all(
        plan.profile_id == "eiel-production-runtime-v1"
        and plan.generations == 32
        and plan.ticks_per_generation == 80
        and len(
            [
                branch
                for branch in plan.comparison_branches
                if branch.kind == "fresh_repeat"
            ]
        )
        == 2
        and len(
            [
                branch
                for branch in plan.comparison_branches
                if branch.kind == "checkpoint_continuation"
                and branch.split_generation == 16
            ]
        )
        == 2
        for plan in plans
    )


def test_complete_frozen_matrix_decides_a_with_raw_deltas() -> None:
    authority = _authority()
    result = evaluate_e1_v1(_cells(authority), authority=authority)
    assert result["decision"] == DECISION_A
    assert result["cell_count"] == 36
    assert result["pair_count"] == 18
    assert result["no_influence_exact"] is True
    assert result["observations_complete"] is True
    assert all(row["supports"] for row in result["treatment_summaries"])
    assert len(result["paired_deltas"]) == 18


def test_valid_complete_matrix_not_meeting_direction_rule_decides_b() -> None:
    authority = _authority()
    cells = [
        replace(
            cell,
            metrics={
                **cell.metrics,
                "population.terminal_alive": 10,
                "population.successful_births_cumulative": 10,
            },
        )
        for cell in _cells(authority)
    ]
    result = evaluate_e1_v1(cells, authority=authority)
    assert result["decision"] == DECISION_B
    assert "treatment_campaign_direction_rule_not_met" in result["reasons"]


def test_nonexact_neutral_no_influence_decides_b() -> None:
    authority = _authority()
    cells = _cells(authority)
    target = next(
        index
        for index, cell in enumerate(cells)
        if cell.campaign_id == CAMPAIGNS[0] and cell.arm == "treatment"
    )
    cells[target] = replace(
        cells[target],
        metrics={
            **cells[target].metrics,
            "population.terminal_alive": 11,
        },
    )
    result = evaluate_e1_v1(cells, authority=authority)
    assert result["decision"] == DECISION_B
    assert result["no_influence_exact"] is False


def test_missing_cell_or_metrics_decides_c() -> None:
    authority = _authority()
    cells = _cells(authority)
    assert evaluate_e1_v1(cells[:-1], authority=authority)["decision"] == (
        DECISION_C
    )
    cells[0] = replace(
        cells[0], metrics={"population.terminal_alive": 1}
    )
    assert evaluate_e1_v1(cells, authority=authority)["decision"] == (
        DECISION_C
    )


@pytest.mark.parametrize(
    "defect",
    [
        "authority",
        "c1",
        "founder",
        "checkpoint",
        "route",
        "parity",
        "fallback",
        "duplicate",
    ],
)
def test_identity_and_integrity_failures_decide_d(defect: str) -> None:
    authority = _authority()
    cells = _cells(authority)
    if defect == "authority":
        bindings = {
            campaign: dict(binding)
            for campaign, binding in (
                authority.checkpoint_authority_bindings.items()
            )
        }
        binding = bindings[CAMPAIGNS[0]]
        binding["artifact_sha256"] = "b" * 64
        bindings[CAMPAIGNS[0]] = binding
        authority = replace(
            authority,
            checkpoint_authority_bindings=bindings,
        )
    elif defect == "c1":
        authority = replace(authority, c1_decision="C1-B")
    elif defect == "founder":
        cells[0] = replace(cells[0], founder_payload_sha256="b" * 64)
    elif defect == "checkpoint":
        cells[0] = replace(cells[0], checkpoint_sha256="b" * 64)
    elif defect == "route":
        cells[0] = replace(cells[0], execution_route_sha256="d" * 64)
    elif defect == "parity":
        cells[0] = replace(cells[0], fresh_repeat_exact=False)
    elif defect == "fallback":
        cells[0] = replace(cells[0], no_fallback=False)
    elif defect == "duplicate":
        cells[-1] = cells[0]
    assert evaluate_e1_v1(cells, authority=authority)["decision"] == (
        DECISION_D
    )


def test_runner_writes_immutable_json_csv_and_manifest(tmp_path: Path) -> None:
    authority = _authority()
    output = tmp_path / "e1"
    manifest = materialize_e1_outputs_v1(
        authority=authority,
        cells=_cells(authority),
        c1_decision=_c1_decision(),
        output_directory=output,
    )
    assert manifest["decision"] == DECISION_A
    assert set(manifest["artifacts"]) == {
        "preregistration",
        "paired_json",
        "paired_csv",
        "decision",
    }
    decision = json.loads(
        (output / "genome_v2_e1_decision.json").read_text()
    )
    assert decision["authority_sha256"] == authority.authority_sha256()
    with pytest.raises(GenomeV2E1RunnerError, match="overwrite refused"):
        materialize_e1_outputs_v1(
            authority=authority,
            cells=_cells(authority),
            c1_decision=_c1_decision(),
            output_directory=output,
        )


def test_runner_rejects_non_content_addressed_c1_admission(
    tmp_path: Path,
) -> None:
    authority = _authority()
    c1 = _c1_decision()
    c1["decision"] = "C1-B"
    with pytest.raises(GenomeV2E1RunnerError, match="C1-A admission"):
        materialize_e1_outputs_v1(
            authority=authority,
            cells=_cells(authority),
            c1_decision=c1,
            output_directory=tmp_path / "rejected",
        )
