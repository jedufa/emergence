from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.campaigns.orchestration import (
    CampaignExecutionError,
    _execute_pre_primary_technical_gate,
    _technical_gate_receipt,
    _write_generation_metrics,
)
from engine.campaigns.providers import ResidentBatchRequestV1


def _batch_manifest(root: Path, rows: list[dict]) -> list[dict]:
    path = root / "batches" / "batch-0000"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "result": {"generation_metrics_rows": rows},
            }
        ),
        encoding="utf-8",
    )
    return [{"batch_id": "batch-0000", "cell_ids": ["cell-a"]}]


def test_generation_series_is_plan_bound_sorted_and_schema_preserving(
    tmp_path: Path,
) -> None:
    batches = _batch_manifest(
        tmp_path,
        [
            {"schema_version": "provider_rows_v2", "cell_id": "cell-a", "generation": 2, "x": 3},
            {"schema_version": "provider_rows_v2", "cell_id": "cell-a", "generation": 0, "x": 1},
        ],
    )

    path = _write_generation_metrics(
        tmp_path,
        {"campaign_id": "campaign", "plan_sha256": "a" * 64},
        batches,
    )

    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "eiel_campaign_generation_series_v1"
    assert payload["provider_row_schema_versions"] == ["provider_rows_v2"]
    assert [row["generation"] for row in payload["rows"]] == [0, 2]


def test_generation_series_rejects_duplicate_cell_generation(tmp_path: Path) -> None:
    row = {
        "schema_version": "provider_rows_v1",
        "cell_id": "cell-a",
        "generation": 1,
    }
    batches = _batch_manifest(tmp_path, [row, dict(row)])

    with pytest.raises(CampaignExecutionError, match="duplicate generation"):
        _write_generation_metrics(
            tmp_path,
            {"campaign_id": "campaign", "plan_sha256": "b" * 64},
            batches,
        )


def test_provider_owned_technical_gate_is_plan_and_route_bound(tmp_path: Path) -> None:
    class Provider:
        provider_id = "provider-v1"

        @staticmethod
        def execution_route_contract(plan):
            class Contract:
                execution_route_sha256 = "c" * 64

            return Contract()

        @staticmethod
        def evaluate_technical_gate(**kwargs):
            assert kwargs["gate_cell_ids"] == ("cell-a",)
            assert len(kwargs["batch_manifests"]) == 1
            return {"passed": True, "checks": {"population_exact": True}}

    batches = _batch_manifest(tmp_path, [])
    plan = {
        "campaign_id": "campaign",
        "plan_sha256": "a" * 64,
        "resolved_profile": "profile-v1",
        "execution": {
            "technical_gate": {
                "required": True,
                "cell_ids": ["cell-a"],
                "batch_ids": ["batch-0000"],
            }
        },
    }

    receipt = _technical_gate_receipt(tmp_path, plan, Provider(), batches)

    assert receipt is not None
    assert receipt["passed"] is True
    assert receipt["execution_route_sha256"] == "c" * 64
    persisted = json.loads(
        (tmp_path / "technical_gate_receipt.json").read_text(encoding="utf-8")
    )
    assert persisted == receipt


def test_pre_primary_gate_runs_bounded_restore_once_and_rejects_stale_receipt(
    tmp_path: Path,
) -> None:
    route = "c" * 64
    genome_v2_binding = (("schema_version", "genome_v2_checkpoint_authority_binding_v1"),)
    cell_ids = [f"cell-{index}" for index in range(4)]
    calls = {"execute": 0, "resume": 0, "evaluate": 0, "rebind": 0}

    class Provider:
        provider_id = "provider-v1"

        @staticmethod
        def execution_route_contract(plan):
            del plan

            class Contract:
                execution_route_sha256 = route

            return Contract()

        @staticmethod
        def translate_batch(plan, batch_id):
            del plan
            assert batch_id == "batch-0000"
            return ResidentBatchRequestV1(
                profile_id="profile-v1",
                batch_id=batch_id,
                cell_ids=tuple(cell_ids),
                generations=30,
                ticks_per_generation=80,
                checkpoint_milestones=(0, 5, 10, 20, 30),
                parameter_arrays=(("mode", tuple(range(4))),),
                seed_arrays=(("ecology", (101, 101, 101, 101)),),
                execution_route_sha256=route,
                genome_v2_checkpoint_authority_binding=genome_v2_binding,
            )

        @staticmethod
        def evaluate_technical_gate(**kwargs):
            calls["evaluate"] += 1
            manifests = kwargs["batch_manifests"]
            assert [row["gate_role"] for row in manifests] == [
                "canary_execute",
                "fresh_owner_restore",
            ]
            return {"schema_version": "mock_gate_v1", "passed": True}

        @staticmethod
        def prepare_fresh_owner_restore_checkpoint(
            _plan, source_request, restore_request, checkpoint
        ):
            calls["rebind"] += 1
            assert source_request.batch_id == "technical-gate-canary"
            assert restore_request.batch_id == "technical-gate-fresh-owner-restore"
            return {**checkpoint, "rebound_for": restore_request.batch_id}

    plan = {
        "campaign_id": "campaign",
        "plan_sha256": "a" * 64,
        "resolved_profile": "profile-v1",
        "execution": {
            "technical_gate": {
                "required": True,
                "execution_mode": "pre_primary_canary",
                "cell_ids": cell_ids,
                "canary": {
                    "batch_id": "technical-gate-canary",
                    "restore_batch_id": "technical-gate-fresh-owner-restore",
                    "generations": 5,
                    "checkpoint_milestones": [0, 5],
                    "cell_ids": cell_ids,
                },
            }
        },
    }

    def result(request):
        checkpoint = {
            "generation": 5,
            "execution_route_sha256": route,
            "cell_ids": list(request.cell_ids),
        }
        return {
            "status": "completed",
            "checkpoint": checkpoint,
            "metrics_rows": [],
            "result": {"terminal_cell_digests": {cell_id: "d" for cell_id in cell_ids}},
        }

    def execute(_plan, request, _context):
        calls["execute"] += 1
        assert request.batch_id == "technical-gate-canary"
        assert request.generations == 5
        assert request.checkpoint_milestones == (0, 5)
        assert request.genome_v2_checkpoint_authority_binding == genome_v2_binding
        return result(request)

    def resume(_plan, request, checkpoint, _context):
        calls["resume"] += 1
        assert request.batch_id == "technical-gate-fresh-owner-restore"
        assert request.genome_v2_checkpoint_authority_binding == genome_v2_binding
        assert checkpoint["generation"] == 5
        assert checkpoint["rebound_for"] == request.batch_id
        return result(request)

    receipt = _execute_pre_primary_technical_gate(
        plan=plan,
        run_root=tmp_path,
        provider=Provider(),
        execute=execute,
        resume=resume,
        batches=[{"batch_id": "batch-0000", "cell_ids": cell_ids}],
    )
    assert receipt is not None and receipt["passed"] is True
    assert len(receipt["receipt_sha256"]) == 64
    assert calls == {"execute": 1, "resume": 1, "evaluate": 1, "rebind": 1}

    repeated = _execute_pre_primary_technical_gate(
        plan=plan,
        run_root=tmp_path,
        provider=Provider(),
        execute=execute,
        resume=resume,
        batches=[{"batch_id": "batch-0000", "cell_ids": cell_ids}],
    )
    assert repeated == receipt
    assert calls == {"execute": 1, "resume": 1, "evaluate": 1, "rebind": 1}

    stale = dict(plan)
    stale["plan_sha256"] = "b" * 64
    with pytest.raises(CampaignExecutionError, match="stale or binding-invalid"):
        _execute_pre_primary_technical_gate(
            plan=stale,
            run_root=tmp_path,
            provider=Provider(),
            execute=execute,
            resume=resume,
            batches=[{"batch_id": "batch-0000", "cell_ids": cell_ids}],
        )


def test_pre_primary_g10_gate_restores_from_declared_g5_checkpoint(
    tmp_path: Path,
) -> None:
    route = "d" * 64
    cell_ids = [f"cell-{index}" for index in range(4)]
    calls = {"execute": 0, "resume": 0, "evaluate": 0, "rebind": 0}

    class Provider:
        provider_id = "provider-v1"

        @staticmethod
        def execution_route_contract(plan):
            del plan

            class Contract:
                execution_route_sha256 = route

            return Contract()

        @staticmethod
        def translate_batch(plan, batch_id):
            del plan
            assert batch_id == "batch-0000"
            return ResidentBatchRequestV1(
                profile_id="profile-v1",
                batch_id=batch_id,
                cell_ids=tuple(cell_ids),
                generations=30,
                ticks_per_generation=80,
                checkpoint_milestones=(0, 5, 10, 20, 30),
                parameter_arrays=(("mode", tuple(range(4))),),
                seed_arrays=(("ecology", (101, 101, 101, 101)),),
                execution_route_sha256=route,
            )

        @staticmethod
        def evaluate_technical_gate(**kwargs):
            calls["evaluate"] += 1
            manifests = kwargs["batch_manifests"]
            assert [row["gate_role"] for row in manifests] == [
                "canary_execute",
                "fresh_owner_restore",
            ]
            assert [
                row["checkpoint"]["generation"] for row in manifests
            ] == [10, 10]
            return {"schema_version": "mock_gate_v1", "passed": True}

        @staticmethod
        def prepare_fresh_owner_restore_checkpoint(
            _plan, source_request, restore_request, checkpoint
        ):
            calls["rebind"] += 1
            assert source_request.batch_id == "technical-gate-canary"
            assert restore_request.batch_id == "technical-gate-fresh-owner-restore"
            return {**checkpoint, "rebound_for": restore_request.batch_id}

    plan = {
        "campaign_id": "campaign",
        "plan_sha256": "b" * 64,
        "resolved_profile": "profile-v1",
        "execution": {
            "technical_gate": {
                "required": True,
                "execution_mode": "pre_primary_canary",
                "cell_ids": cell_ids,
                "canary": {
                    "batch_id": "technical-gate-canary",
                    "restore_batch_id": "technical-gate-fresh-owner-restore",
                    "generations": 10,
                    "checkpoint_milestones": [0, 5, 10],
                    "cell_ids": cell_ids,
                },
            }
        },
    }

    def checkpoint(generation):
        return {
            "generation": generation,
            "execution_route_sha256": route,
            "cell_ids": list(cell_ids),
        }

    def execute(_plan, request, _context):
        calls["execute"] += 1
        assert request.batch_id == "technical-gate-canary"
        assert request.generations == 10
        assert request.checkpoint_milestones == (0, 5, 10)
        return {
            "status": "completed",
            "checkpoint": checkpoint(10),
            "metrics_rows": [],
            "result": {
                "terminal_cell_digests": {
                    cell_id: "d" for cell_id in cell_ids
                },
                "checkpoint_catalog": {
                    "0": checkpoint(0),
                    "5": checkpoint(5),
                    "10": checkpoint(10),
                },
            },
        }

    def resume(_plan, request, source_checkpoint, _context):
        calls["resume"] += 1
        assert request.batch_id == "technical-gate-fresh-owner-restore"
        assert request.generations == 10
        assert request.checkpoint_milestones == (0, 5, 10)
        assert source_checkpoint["generation"] == 5
        assert source_checkpoint["rebound_for"] == request.batch_id
        return {
            "status": "completed",
            "checkpoint": checkpoint(10),
            "metrics_rows": [],
            "result": {
                "terminal_cell_digests": {
                    cell_id: "d" for cell_id in cell_ids
                }
            },
        }

    receipt = _execute_pre_primary_technical_gate(
        plan=plan,
        run_root=tmp_path,
        provider=Provider(),
        execute=execute,
        resume=resume,
        batches=[{"batch_id": "batch-0000", "cell_ids": cell_ids}],
    )

    assert receipt is not None and receipt["passed"] is True
    assert calls == {"execute": 1, "resume": 1, "evaluate": 1, "rebind": 1}
