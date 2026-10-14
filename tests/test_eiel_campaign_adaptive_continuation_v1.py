from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.compiler import CampaignError
from engine.campaigns.orchestration import (
    CampaignExecutionError,
    atomic_write_json,
    prepare_adaptive_campaign_continuation_v1,
)
from engine.campaigns.providers import (
    ExecutionProviderRegistryV1,
    ResidentBatchRequestV1,
)
from tools.scripts.eiel_campaign import _adaptive_resume


ROUTE_SHA = "7" * 64
CELL_IDS = tuple(f"cell-{index:02d}" for index in range(48))


class _Provider:
    provider_id = "adaptive-test-provider"
    profile_id = "adaptive-test-profile"

    def execution_route_contract(self, _plan):
        return SimpleNamespace(execution_route_sha256=ROUTE_SHA)

    def translate_batch(self, plan, batch_id):
        values = tuple(cell["parameters"]["dial"] for cell in plan["cells"])
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=CELL_IDS,
            generations=plan["execution"]["generations"],
            ticks_per_generation=160,
            checkpoint_milestones=(plan["execution"]["generations"],),
            parameter_arrays=(("dial", values),),
            seed_arrays=(("ecology", tuple(range(48))),),
            execution_route_sha256=ROUTE_SHA,
        )

    def prepare_wrc_checkpoint_adaptive_rebinding_v1(
        self,
        _source_plan,
        restore_plan,
        _source_request,
        restore_request,
        checkpoint,
    ):
        rebound = dict(checkpoint)
        rebound["checkpoint_binding"] = {
            "plan_sha256": restore_plan["plan_sha256"]
        }
        rebound["wrc_checkpoint_adaptive_rebinding_v1"] = {
            "receipt_sha256": "a" * 64,
            "restore_generation": restore_request.generations,
        }
        return rebound


def _plan(*, generation: int, value: float, plan_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "adaptive-test",
        "plan_sha256": plan_sha,
        "resolved_profile": _Provider.profile_id,
        "qualification_key": {"fixed": "same"},
        "qualification_key_sha256": "2" * 64,
        "artifacts": {"runtime": {"sha256": "3" * 64}},
        "execution": {"generations": generation, "ticks_per_generation": 160},
        "cells": [
            {"cell_id": cell_id, "parameters": {"dial": value}}
            for cell_id in CELL_IDS
        ],
        "physical_batches": [
            {"batch_id": "batch-0000", "cell_ids": list(CELL_IDS)}
        ],
    }


def _stage_manifest(source, restore):
    body = {
        "schema_version": "eiel_wrc_b48_adaptive_stage_manifest_v1",
        "stage": "C2",
        "source_generation": 25,
        "restore_generation": 50,
        "tick_start": 4_000,
        "tick_end": 8_000,
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_sha256": restore["plan_sha256"],
        "qualification_key_sha256": source["qualification_key_sha256"],
        "production_artifacts_sha256": canonical_sha256(source["artifacts"]),
        "configuration_count": 16,
        "seed_replicates_per_configuration": 3,
        "physical_cell_count": 48,
        "changed_cell_count": 48,
        "cell_overrides": [
            {
                "cell_id": cell_id,
                "config_id": f"C{index // 3 + 1:02d}",
                "seed_tuple": f"S{index % 3 + 1}",
                "overrides": {"dial": 0.4},
            }
            for index, cell_id in enumerate(CELL_IDS)
        ],
        "qualification_receipt_action": "unchanged",
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _persist_stage_roots(
    tmp_path: Path,
    source: Mapping[str, Any],
    restore: Mapping[str, Any],
) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    restore_root = tmp_path / "restore"
    atomic_write_json(source_root / "execution_plan.json", source)
    atomic_write_json(restore_root / "execution_plan.json", restore)
    atomic_write_json(
        source_root / "batches" / "batch-0000" / "manifest.json",
        {
            "schema_version": "eiel_campaign_batch_manifest_v1",
            "campaign_id": source["campaign_id"],
            "plan_sha256": source["plan_sha256"],
            "profile_id": source["resolved_profile"],
            "provider_id": _Provider.provider_id,
            "execution_route_sha256": ROUTE_SHA,
            "batch_id": "batch-0000",
            "cell_ids": list(CELL_IDS),
            "status": "completed",
            "attempts": 1,
            "checkpoint": {
                "schema_version": "checkpoint-v1",
                "generation": source["execution"]["generations"],
                "execution_route_sha256": ROUTE_SHA,
            },
            "metrics_rows": [],
            "result": {},
            "error": None,
            "created_at": "2026-08-11T00:00:00Z",
            "updated_at": "2026-08-11T00:00:00Z",
        },
    )
    atomic_write_json(
        restore_root / "preflight_receipt.json",
        {
            "schema_version": "eiel_campaign_preflight_v1",
            "campaign_id": restore["campaign_id"],
            "plan_sha256": restore["plan_sha256"],
            "ready_for_execution": True,
            "reasons": [],
            "execution_authority_reasons": [],
            "execution_route": {
                "authority_state": "route_qualified",
                "execution_route_sha256": ROUTE_SHA,
                "artifact_conformance_exact": True,
            },
        },
    )
    return source_root, restore_root


def _registry() -> ExecutionProviderRegistryV1:
    registry = ExecutionProviderRegistryV1()
    registry.register(_Provider())
    return registry


def test_prepares_exact_b48_adaptive_stage_for_existing_executor(tmp_path):
    source = _plan(generation=25, value=0.2, plan_sha="4" * 64)
    restore = _plan(generation=50, value=0.4, plan_sha="5" * 64)
    source_root, restore_root = _persist_stage_roots(tmp_path, source, restore)

    receipt = prepare_adaptive_campaign_continuation_v1(
        source,
        restore,
        source_run_root=source_root,
        restore_run_root=restore_root,
        provider_registry=_registry(),
    )

    manifest = __import__("json").loads(
        (restore_root / "batches" / "batch-0000" / "manifest.json").read_text()
    )
    assert receipt["status"] == "READY_TO_RESUME"
    assert receipt["qualification_receipt_action"] == "unchanged"
    assert receipt["cell_ids"] == list(CELL_IDS)
    assert manifest["status"] == "interrupted"
    assert manifest["checkpoint"]["wrc_checkpoint_adaptive_rebinding_v1"][
        "receipt_sha256"
    ] == "a" * 64


def test_rejects_any_cell_mapping_or_qualification_change(tmp_path):
    source = _plan(generation=25, value=0.2, plan_sha="4" * 64)
    restore = _plan(generation=50, value=0.4, plan_sha="5" * 64)
    source_root, restore_root = _persist_stage_roots(tmp_path, source, restore)
    changed = dict(restore)
    changed["qualification_key_sha256"] = "9" * 64

    with pytest.raises(CampaignExecutionError, match="qualification or artifacts"):
        prepare_adaptive_campaign_continuation_v1(
            source,
            changed,
            source_run_root=source_root,
            restore_run_root=restore_root,
            provider_registry=_registry(),
        )

    changed = dict(restore)
    changed["physical_batches"] = [
        {
            "batch_id": "batch-0000",
            "cell_ids": list(reversed(CELL_IDS)),
        }
    ]
    atomic_write_json(restore_root / "execution_plan.json", changed)
    with pytest.raises(CampaignExecutionError, match="unchanged physical B48"):
        prepare_adaptive_campaign_continuation_v1(
            source,
            changed,
            source_run_root=source_root,
            restore_run_root=restore_root,
            provider_registry=_registry(),
        )


def test_adaptive_resume_cli_path_preflights_seeds_then_executes(tmp_path, monkeypatch):
    source = _plan(generation=25, value=0.2, plan_sha="4" * 64)
    restore = _plan(generation=50, value=0.4, plan_sha="5" * 64)
    source_root = tmp_path / "source"
    restore_root = tmp_path / "restore"
    restore_plan_path = tmp_path / "restore-plan.json"
    stage_manifest_path = tmp_path / "stage-manifest.json"
    atomic_write_json(source_root / "execution_plan.json", source)
    atomic_write_json(restore_plan_path, restore)
    atomic_write_json(stage_manifest_path, _stage_manifest(source, restore))
    calls: list[str] = []

    def fake_preflight(plan, **kwargs):
        calls.append("preflight")
        assert plan == restore
        assert kwargs["provider_registry"] is registry
        return {
            "schema_version": "eiel_campaign_preflight_v1",
            "campaign_id": restore["campaign_id"],
            "plan_sha256": restore["plan_sha256"],
            "ready_for_execution": True,
            "reasons": [],
            "execution_authority_reasons": [],
        }

    def fake_prepare(source_plan, restore_plan, **kwargs):
        calls.append("prepare")
        assert source_plan == source
        assert restore_plan == restore
        assert kwargs["source_run_root"] == source_root.resolve()
        assert kwargs["restore_run_root"] == restore_root.resolve()
        return {
            "receipt_path": str(restore_root / "adaptive_transition_receipt.json"),
            "receipt_sha256": "b" * 64,
            "adaptive_rebinding_receipt_sha256": "a" * 64,
            "qualification_receipt_action": "unchanged",
        }

    def fake_execute(plan, **kwargs):
        calls.append("execute")
        assert plan == restore
        assert kwargs["run_root"] == restore_root.resolve()
        assert (restore_root / "preflight_receipt.json").is_file()
        return {
            "ok": True,
            "schema_version": "eiel_campaign_execution_receipt_v1",
            "status": "completed",
        }

    registry = _registry()
    monkeypatch.setattr("engine.campaigns.preflight.preflight_campaign", fake_preflight)
    monkeypatch.setattr(
        "engine.campaigns.orchestration.prepare_adaptive_campaign_continuation_v1",
        fake_prepare,
    )
    monkeypatch.setattr("engine.campaigns.orchestration.execute_campaign", fake_execute)

    receipt = _adaptive_resume(
        source_root,
        restore_plan_path,
        stage_manifest_path,
        restore_root,
        provider_registry=registry,
    )

    assert calls == ["preflight", "prepare", "execute"]
    assert receipt["ok"] is True
    assert receipt["schema_version"] == ("eiel_wrc_adaptive_stage_execution_receipt_v1")
    assert receipt["adaptive_rebinding_receipt_sha256"] == "a" * 64
    assert receipt["qualification_receipt_action"] == "unchanged"
    assert (
        __import__("json").loads((restore_root / "execution_plan.json").read_text())
        == restore
    )


def test_adaptive_resume_rejects_same_source_and_restore_root(tmp_path):
    with pytest.raises(
        CampaignError,
        match="distinct source and restore run roots",
    ):
        _adaptive_resume(
            tmp_path,
            tmp_path / "restore-plan.json",
            tmp_path / "stage-manifest.json",
            tmp_path,
        )


def test_adaptive_resume_reuses_exact_persisted_transition(tmp_path, monkeypatch):
    source = _plan(generation=25, value=0.2, plan_sha="4" * 64)
    restore = _plan(generation=50, value=0.4, plan_sha="5" * 64)
    source_root = (tmp_path / "source").resolve()
    restore_root = (tmp_path / "restore").resolve()
    restore_plan_path = tmp_path / "restore-plan.json"
    stage_manifest_path = tmp_path / "stage-manifest.json"
    atomic_write_json(source_root / "execution_plan.json", source)
    atomic_write_json(restore_plan_path, restore)
    atomic_write_json(stage_manifest_path, _stage_manifest(source, restore))
    transition = {
        "schema_version": "eiel_wrc_adaptive_campaign_transition_v1",
        "source_campaign_id": source["campaign_id"],
        "restore_campaign_id": restore["campaign_id"],
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_sha256": restore["plan_sha256"],
        "execution_route_sha256": ROUTE_SHA,
        "qualification_key_sha256": restore["qualification_key_sha256"],
        "batch_id": "batch-0000",
        "cell_ids": list(CELL_IDS),
        "source_generation": 25,
        "restore_generation": 50,
        "adaptive_rebinding_receipt_sha256": "a" * 64,
        "source_batch_manifest_path": str(
            source_root / "batches" / "batch-0000" / "manifest.json"
        ),
        "restore_batch_manifest_path": str(
            restore_root / "batches" / "batch-0000" / "manifest.json"
        ),
        "qualification_receipt_action": "unchanged",
        "status": "READY_TO_RESUME",
    }
    atomic_write_json(
        restore_root / "adaptive_transition_receipt.json",
        {**transition, "receipt_sha256": canonical_sha256(transition)},
    )

    monkeypatch.setattr(
        "engine.campaigns.preflight.preflight_campaign",
        lambda *_args, **_kwargs: {
            "schema_version": "eiel_campaign_preflight_v1",
            "campaign_id": restore["campaign_id"],
            "plan_sha256": restore["plan_sha256"],
            "ready_for_execution": True,
            "reasons": [],
            "execution_authority_reasons": [],
        },
    )

    def should_not_prepare(*_args, **_kwargs):
        raise AssertionError("an exact persisted transition must be reused")

    monkeypatch.setattr(
        "engine.campaigns.orchestration.prepare_adaptive_campaign_continuation_v1",
        should_not_prepare,
    )
    monkeypatch.setattr(
        "engine.campaigns.orchestration.execute_campaign",
        lambda *_args, **_kwargs: {"ok": True, "status": "completed"},
    )

    receipt = _adaptive_resume(
        source_root,
        restore_plan_path,
        stage_manifest_path,
        restore_root,
        provider_registry=_registry(),
    )

    assert receipt["ok"] is True
    assert receipt["adaptive_rebinding_receipt_sha256"] == "a" * 64
