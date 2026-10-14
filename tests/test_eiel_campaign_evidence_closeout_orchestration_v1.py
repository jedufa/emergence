from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import engine.campaigns.chronicle_closeout as chronicle_closeout
import engine.campaigns.evidence_closeout as evidence_closeout
import simulation.chronicle.archive_history as archive_history
from engine.campaigns.orchestration import CampaignExecutionError, execute_campaign
from engine.campaigns.providers import (
    ExecutionProviderRegistryV1,
    ResidentBatchRequestV1,
)


PROFILE = "evidence-closeout-order-profile-v1"


class _Provider:
    provider_id = "evidence-closeout-order-provider-v1"
    profile_id = PROFILE

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def translate_batch(
        self, plan: Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        del plan
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=("cell-0",),
            generations=2,
            ticks_per_generation=1,
            checkpoint_milestones=(2,),
            parameter_arrays=(),
            seed_arrays=(),
        )

    def execute_batch(self, plan, request, context):
        del plan
        self.events.append(
            "primary" if request.batch_id == "batch-0000" else "comparison"
        )
        checkpoint = {
            "generation": request.generations,
            "terminal_cell_digests": {"cell-0": "terminal-cell-0"},
        }
        context.record_checkpoint(checkpoint)
        return {
            "status": "completed",
            "checkpoint": checkpoint,
            "metrics_rows": [
                {
                    "cell_id": "cell-0",
                    "metrics": {"population.alive": 1},
                }
            ],
            "result": {
                "terminal_generation": request.generations,
                "cell_results": {
                    "cell-0": {
                        "cell_id": "cell-0",
                        "evidence_projection": {"sentinel": "provider-owned"},
                    }
                },
            },
        }


def _plan(run_root: Path, *, qualification_mode: bool = False) -> dict[str, Any]:
    plan = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "evidence-closeout-order",
        "plan_sha256": "a" * 64,
        "resolved_profile": PROFILE,
        "execution": {
            "generations": 2,
            "ticks_per_generation": 1,
            "checkpoints": [2],
        },
        "cells": [{"cell_id": "cell-0", "parameters": {}, "seeds": {}}],
        "physical_batches": [
            {
                "ordinal": 0,
                "batch_id": "batch-0000",
                "cell_ids": ["cell-0"],
            }
        ],
        "comparisons": {
            "branches": [
                {
                    "ordinal": 0,
                    "branch_id": "repeat-cell-0",
                    "kind": "fresh_repeat",
                    "source_cell_id": "cell-0",
                    "repeat_ordinal": 1,
                    "split_generation": None,
                }
            ]
        },
        "output": {
            "runtime_root": str(run_root),
            "chronicle_profile": "scientific_full",
            "chronicle_history_root": str(run_root / "chronicle-history"),
            "chronicle_archive_enabled": True,
            "evidence_closeout_enabled": True,
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    ready_key = (
        "ready_for_route_qualification" if qualification_mode else "ready_for_execution"
    )
    (run_root / "preflight_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_preflight_v1",
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                ready_key: True,
                "reasons": [],
                "execution_authority_reasons": [],
            }
        ),
        encoding="utf-8",
    )
    return plan


def _registry(provider: _Provider) -> ExecutionProviderRegistryV1:
    registry = ExecutionProviderRegistryV1()
    registry.register(provider)
    return registry


def _patch_closeouts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_root: Path,
    events: list[str],
    fail_evidence: bool = False,
) -> dict[str, Any]:
    receipt = {
        "schema_version": "test_evidence_closeout_receipt_v1",
        "verified": True,
        "zip_path": str(run_root / "evidence.zip"),
    }

    def materialize(plan, *, run_root, batch_manifests):
        del plan, batch_manifests
        assert (
            json.loads(
                (run_root / "comparisons_manifest.json").read_text(encoding="utf-8")
            )["status"]
            == "completed"
        )
        events.append("chronicle_materialize")
        return {"cell-0": run_root / "authoritative-chronicle"}

    def archive(history_root, *, campaign_id, run_id, worlds, metadata):
        del history_root, worlds, metadata
        assert events[-1] == "chronicle_materialize"
        assert run_id == run_root.name
        events.append("chronicle_archive")
        return SimpleNamespace(
            campaign_id=campaign_id,
            run_id=run_id,
            archive_path=run_root / "chronicles.zip",
            sha256="b" * 64,
            size_bytes=1,
            world_ids=("cell-0",),
            member_count=1,
        )

    def closeout(plan, *, run_root, provider_cell_results):
        assert events == [
            "primary",
            "comparison",
            "chronicle_materialize",
            "chronicle_archive",
        ]
        assert (run_root / "chronicle_archive_receipt.json").is_file()
        assert set(provider_cell_results) == {"cell-0"}
        assert provider_cell_results["cell-0"]["cell_id"] == "cell-0"
        events.append("evidence_closeout")
        if fail_evidence:
            raise evidence_closeout.CampaignEvidenceCloseoutError(
                "synthetic closeout failure"
            )
        return receipt

    monkeypatch.setattr(
        chronicle_closeout,
        "materialize_completed_campaign_chronicles",
        materialize,
    )
    monkeypatch.setattr(archive_history, "create_completed_run_archive", archive)
    monkeypatch.setattr(evidence_closeout, "closeout_campaign_evidence", closeout)
    return receipt


def test_closeout_runs_after_primary_comparison_and_chronicle_and_returns_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    plan = _plan(tmp_path)
    expected = _patch_closeouts(monkeypatch, run_root=tmp_path, events=events)

    result = execute_campaign(
        plan,
        run_root=tmp_path,
        provider_registry=_registry(_Provider(events)),
    )

    assert result["ok"] is True
    assert result["evidence_closeout_receipt"] == expected
    assert events == [
        "primary",
        "comparison",
        "chronicle_materialize",
        "chronicle_archive",
        "evidence_closeout",
    ]


def test_closeout_failure_cannot_return_ok_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    plan = _plan(tmp_path)
    _patch_closeouts(
        monkeypatch,
        run_root=tmp_path,
        events=events,
        fail_evidence=True,
    )

    with pytest.raises(
        CampaignExecutionError,
        match="completed campaign evidence closeout failed: synthetic closeout failure",
    ):
        execute_campaign(
            plan,
            run_root=tmp_path,
            provider_registry=_registry(_Provider(events)),
        )

    assert events[-1] == "evidence_closeout"


def test_qualification_mode_skips_chronicle_and_evidence_closeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    plan = _plan(tmp_path, qualification_mode=True)

    def unexpected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("qualification mode must skip closeout")

    monkeypatch.setattr(
        chronicle_closeout,
        "materialize_completed_campaign_chronicles",
        unexpected,
    )
    monkeypatch.setattr(archive_history, "create_completed_run_archive", unexpected)
    monkeypatch.setattr(evidence_closeout, "closeout_campaign_evidence", unexpected)

    result = execute_campaign(
        plan,
        run_root=tmp_path,
        provider_registry=_registry(_Provider(events)),
        qualification_mode=True,
    )

    assert result["ok"] is True
    assert result["evidence_closeout_receipt"] is None
    assert result["chronicle_archive_receipt_path"] is None
    assert events == ["primary", "comparison"]
