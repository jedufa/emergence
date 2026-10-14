from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from engine.campaigns.contracts import ExecutionRouteContractV1
from engine.campaigns.orchestration import CampaignExecutionError, execute_campaign
from engine.campaigns.providers import (
    ExecutionProviderRegistryV1,
    ResidentBatchRequestV1,
)
from engine.campaigns.qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION


PROFILE = "route-checkpoint-test-profile-v1"


class _Provider:
    provider_id = "route-checkpoint-test-provider-v1"
    profile_id = PROFILE

    def __init__(self) -> None:
        self.resume_calls = 0
        self.contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                {
                    "component_id": "checkpoint_restore",
                    "contract_version": "test-v1",
                },
            ),
            required_obligations=("checkpoint_export", "fresh_owner_restore"),
        )

    def execution_route_contract(
        self,
        plan: Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        return self.contract

    def translate_batch(
        self, plan: Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=("cell-0",),
            generations=2,
            ticks_per_generation=2,
            checkpoint_milestones=(1, 2),
            parameter_arrays=(),
            seed_arrays=(),
            execution_route_sha256=self.contract.execution_route_sha256,
        )

    def execute_batch(self, plan, request, context):
        return {
            "status": "interrupted",
            "checkpoint": {
                "generation": 1,
                "next_tick": 2,
                "path": "checkpoint",
            },
            "metrics_rows": [],
            "result": {},
        }

    def resume_batch(self, plan, request, checkpoint, context):
        self.resume_calls += 1
        return {
            "status": "completed",
            "checkpoint": {"generation": 2, "path": "checkpoint"},
            "metrics_rows": [
                {"cell_id": "cell-0", "metrics": {"population.alive": 1}}
            ],
            "result": {},
        }


def _plan(root: Path, provider: _Provider) -> dict[str, Any]:
    plan = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "route-checkpoint-test",
        "plan_sha256": "a" * 64,
        "resolved_profile": PROFILE,
        "execution": {
            "generations": 2,
            "ticks_per_generation": 2,
            "checkpoints": [1, 2],
        },
        "cells": [{"cell_id": "cell-0", "parameters": {}, "seeds": {}}],
        "physical_batches": [
            {"ordinal": 0, "batch_id": "batch-0000", "cell_ids": ["cell-0"]}
        ],
        "comparisons": {"branches": []},
        "output": {"runtime_root": str(root)},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "preflight_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_preflight_v1",
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "ready_for_execution": True,
                "reasons": [],
                "execution_authority_reasons": [],
                "execution_route": {
                    "authority_state": "route_qualified",
                    "execution_route_sha256": (
                        provider.contract.execution_route_sha256
                    ),
                    "artifact_conformance_exact": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return plan


def _registry(provider: _Provider) -> ExecutionProviderRegistryV1:
    registry = ExecutionProviderRegistryV1()
    registry.register(provider)
    return registry


def test_checkpoint_route_hash_is_bound_before_resume(tmp_path: Path) -> None:
    provider = _Provider()
    plan = _plan(tmp_path, provider)
    registry = _registry(provider)

    interrupted = execute_campaign(
        plan, run_root=tmp_path, provider_registry=registry
    )
    checkpoint_path = tmp_path / "batches/batch-0000/manifest.json"
    checkpoint_manifest = json.loads(checkpoint_path.read_text())

    assert interrupted["status"] == "interrupted"
    assert checkpoint_manifest["checkpoint"]["execution_route_sha256"] == (
        provider.contract.execution_route_sha256
    )

    completed = execute_campaign(
        plan, run_root=tmp_path, provider_registry=registry
    )
    assert completed["status"] == "completed"
    assert provider.resume_calls == 1


def test_checkpoint_route_mismatch_rejects_before_provider_resume(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    plan = _plan(tmp_path, provider)
    registry = _registry(provider)
    execute_campaign(plan, run_root=tmp_path, provider_registry=registry)
    path = tmp_path / "batches/batch-0000/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint"]["execution_route_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        CampaignExecutionError,
        match="checkpoint execution-route binding mismatch",
    ):
        execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    assert provider.resume_calls == 0


def test_missing_checkpoint_route_rejects_before_provider_resume(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    plan = _plan(tmp_path, provider)
    registry = _registry(provider)
    execute_campaign(plan, run_root=tmp_path, provider_registry=registry)
    path = tmp_path / "batches/batch-0000/manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["checkpoint"]["execution_route_sha256"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        CampaignExecutionError,
        match="checkpoint execution-route binding is missing",
    ):
        execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    assert provider.resume_calls == 0


def test_preflight_route_mismatch_rejects_before_batch_dispatch(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    plan = _plan(tmp_path, provider)
    preflight_path = tmp_path / "preflight_receipt.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["execution_route"]["execution_route_sha256"] = "0" * 64
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    with pytest.raises(
        CampaignExecutionError,
        match="does not grant execution-route authority",
    ):
        execute_campaign(
            plan,
            run_root=tmp_path,
            provider_registry=_registry(provider),
        )

    assert not (tmp_path / "batches").exists()
