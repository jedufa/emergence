from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.contracts import ExecutionRouteContractV1
from engine.campaigns.providers import (
    ExecutionProviderRegistryV1,
    ResidentBatchRequestV1,
)
from engine.campaigns.orchestration import _compare_branch, _narrow_request
from engine.campaigns.qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION
from engine.campaigns.route_qualification import execute_route_qualification_v1
from engine.campaigns.route_sentinels import RouteTransitionSentinelV1
from engine.qualification_plans.v1 import generate_execution_route_canary_plan


def test_comparison_narrowing_preserves_declared_generation_zero_boundary() -> None:
    request = ResidentBatchRequestV1(
        profile_id="profile",
        batch_id="batch",
        cell_ids=("cell-a", "cell-b"),
        generations=9,
        ticks_per_generation=80,
        checkpoint_milestones=(0, 3, 9),
        parameter_arrays=(("parameter", (1, 2)),),
        seed_arrays=(("seed", (11, 12)),),
    )

    narrowed = _narrow_request(
        request,
        branch_id="comparison",
        source_index=1,
        source_cell_id="cell-b",
        generations=6,
        checkpoints=(6,),
    )

    assert narrowed.checkpoint_milestones == (0, 6)


def test_comparisons_prefer_mechanics_over_path_dependent_audit_history() -> None:
    baseline = {
        "checkpoint": {
            "terminal_cell_digests": {"cell": "a" * 64},
            "mechanics_checkpoint_digests": {"cell": "c" * 64},
        },
        "metrics_rows": [{"cell_id": "cell", "metrics": {"alive": 7}}],
    }
    continued = {
        "checkpoint": {
            "terminal_cell_digests": {"cell": "b" * 64},
            "mechanics_checkpoint_digests": {"cell": "c" * 64},
        },
        "metrics_rows": [{"cell_id": "cell", "metrics": {"alive": 7}}],
    }

    for continuation in (False, True):
        receipt = _compare_branch(
            baseline=baseline,
            candidate=continued,
            cell_id="cell",
            provider=object(),
            continuation=continuation,
        )

        assert receipt["match"] is True
        assert receipt["baseline"]["terminal_checkpoint_digest"] == "c" * 64
        assert receipt["candidate"]["terminal_checkpoint_digest"] == "c" * 64


class _Provider:
    provider_id = "orchestrated-route-provider-v1"
    profile_id = "orchestrated-route-profile-v1"
    reporting_parity_supported = True

    def __init__(self) -> None:
        self.contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                {
                    "component_id": "controller",
                    "contract_version": "test-v1",
                },
            ),
            required_obligations=(
                "generation_boundary",
                "multicell_execution",
                "checkpoint_export",
                "disk_checkpoint_reload",
                "fresh_owner_restore",
                "reporting_parity",
            ),
        )

    def execution_route_contract(self, plan, request=None):
        return self.contract

    def translate_batch(self, plan, batch_id):
        batch = next(
            row for row in plan["physical_batches"] if row["batch_id"] == batch_id
        )
        execution = plan["execution"]
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=tuple(batch["cell_ids"]),
            generations=execution["generations"],
            ticks_per_generation=execution["ticks_per_generation"],
            checkpoint_milestones=tuple(execution["checkpoints"]),
            parameter_arrays=(),
            seed_arrays=(),
            checkpoint_storage_mode="disk",
            reporting_enabled=True,
            execution_route_sha256=self.contract.execution_route_sha256,
        )

    def execute_batch(self, plan, request, context):
        return self._result(request, start=0)

    def resume_batch(self, plan, request, checkpoint, context):
        return self._result(request, start=int(checkpoint["generation"]))

    def _result(self, request, *, start: int):
        digests = {
            cell_id: hashlib.sha256(
                f"{cell_id}:{request.generations}".encode()
            ).hexdigest()
            for cell_id in request.cell_ids
        }
        sentinels = {
            cell_id: [
                RouteTransitionSentinelV1(
                    transition_name="generation_boundary",
                    absolute_generation=generation,
                    absolute_tick=generation * request.ticks_per_generation,
                    physical_window_position=generation,
                    action_digest=hashlib.sha256(
                        f"{cell_id}:{generation}".encode()
                    ).hexdigest(),
                ).payload()
                for generation in range(start + 1, request.generations + 1)
            ]
            + [
                RouteTransitionSentinelV1(
                    transition_name="checkpoint_export_boundary",
                    absolute_generation=request.generations,
                    absolute_tick=(
                        request.generations * request.ticks_per_generation
                    ),
                    physical_window_position=request.generations,
                ).payload()
            ]
            for cell_id in request.cell_ids
        }
        return {
            "status": "completed",
            "checkpoint": {
                "generation": request.generations,
                "terminal_cell_digests": digests,
            },
            "metrics_rows": [
                {"cell_id": cell_id, "metrics": {"population.alive": 1}}
                for cell_id in request.cell_ids
            ],
            "result": {
                "transition_sentinels": sentinels,
            },
        }


def _plan(tmp_path: Path, provider: _Provider) -> dict[str, Any]:
    return {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "route-orchestration-source",
        "resolved_profile": provider.profile_id,
        "qualification_key": {
            "artifact_hashes": {"cuda_extension": "a" * 64},
            "admitted_batch_range": [1, 2],
        },
        "execution": {
            "generations": 20,
            "ticks_per_generation": 80,
            "checkpoints": [20],
            "checkpoint_storage_mode": "disk",
        },
        "cells": [
            {
                "ordinal": 0,
                "cell_id": "source-cell",
                "branch_id": "source-branch",
                "rng_lineage_id": "source-rng",
                "dimensions": {},
                "parameters": {},
                "seeds": {},
            }
        ],
        "physical_batches": [
            {
                "ordinal": 0,
                "batch_id": "source-batch",
                "cell_ids": ["source-cell"],
            }
        ],
        "comparisons": {"branches": []},
        "output": {"runtime_root": str(tmp_path / "unused")},
        "plan_sha256": "source",
    }


def test_generated_canary_runs_existing_orchestration_and_publishes(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    plan = _plan(tmp_path, provider)
    canary = generate_execution_route_canary_plan(
        {
            **provider.contract.payload(),
            "execution_route_sha256": provider.contract.execution_route_sha256,
            "canary_requirements": {
                "qualified_ticks_per_generation": 80,
                "admitted_physical_batch_sizes": [1, 2],
                "boundary_generations": 3,
            },
        }
    )
    registry = ExecutionProviderRegistryV1()
    registry.register(provider)

    def preflight(case_plan: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "eiel_campaign_preflight_v1",
            "campaign_id": case_plan["campaign_id"],
            "plan_sha256": case_plan["plan_sha256"],
            "ready_for_execution": False,
            "ready_for_route_qualification": True,
            "reasons": [],
            "execution_authority_reasons": [
                {
                    "code": "profile_not_execution_qualified",
                    "path": "resolved_profile.qualification_status",
                    "message": "bounded canary must establish route authority",
                },
                {
                    "code": "provider_vram_bytes_upper_bound_unavailable",
                    "path": "execution.budgets.vram_bytes",
                    "message": "bounded canary uses a conservative live projection",
                }
            ],
            "execution_route": {
                "authority_state": "qualification_only",
                "execution_route_sha256": (
                    provider.contract.execution_route_sha256
                ),
                "artifact_conformance_exact": True,
            },
        }

    mechanic_receipt = tmp_path / "mechanic" / "qualification.json"
    result = execute_route_qualification_v1(
        campaign_plan=plan,
        canary=canary,
        run_root=tmp_path / "qualification",
        route_contract=provider.contract,
        qualification_receipt_path=mechanic_receipt,
        preflight_runner=preflight,
        provider_registry=registry,
    )

    published = json.loads(
        (mechanic_receipt.parent / "execution_route_qualification.json").read_text()
    )
    assert result["qualification_succeeded"] is True
    assert published["authority_state"] == "route_qualified"
    assert published["covered_obligations"] == sorted(
        provider.contract.required_obligations
    )
    assert list((tmp_path / "qualification/cases").glob("*/campaign_manifest.json"))
