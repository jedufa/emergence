from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest

from engine.campaigns.compiler import CampaignError
from engine.campaigns.orchestration import (
    CampaignExecutionError,
    _batch_rows,
    _chronicle_archive_run_id,
    _compact_resident_checkpoint_settle_v1,
    _narrow_request,
    execute_campaign,
    reseal_terminal_lifecycle_inventory_v1,
)
from engine.campaigns import orchestration as orchestration_module
from engine.campaigns.artifact_lifecycle import (
    RunStorageGuard,
    StorageBudgetExceeded,
    build_artifact_disposition_manifest_v1,
)
from engine.campaigns.providers import (
    ExecutionProviderRegistryV1,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
)
from engine.campaigns.production_engine_identity import PRODUCTION_PROVIDER_ID
from tools.scripts import eiel_campaign as campaign_cli
from tools.scripts.eiel_campaign import _resume
from simulation.chronicle.archive_history import create_completed_run_archive

PROFILE = "fake-resident-profile-v1"


def _plan(run_root: Path) -> dict[str, Any]:
    plan = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "fake-campaign",
        "plan_sha256": "a" * 64,
        "resolved_profile": PROFILE,
        "execution": {
            "generations": 14,
            "ticks_per_generation": 80,
            "checkpoints": [7, 14],
        },
        "cells": [
            {"cell_id": "cell-0", "parameters": {}, "seeds": {}},
            {"cell_id": "cell-1", "parameters": {}, "seeds": {}},
        ],
        "physical_batches": [
            {"ordinal": 0, "batch_id": "batch-0000", "cell_ids": ["cell-0"]},
            {"ordinal": 1, "batch_id": "batch-0001", "cell_ids": ["cell-1"]},
        ],
        "comparisons": {"branches": []},
        "output": {"runtime_root": str(run_root)},
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "preflight_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_preflight_v1",
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "ready_for_execution": True,
                "reasons": [],
                "execution_authority_reasons": [],
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_batch_rows_accepts_compiler_style_tuple_batches(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["physical_batches"] = tuple(
        {**batch, "cell_ids": tuple(batch["cell_ids"])}
        for batch in plan["physical_batches"]
    )

    assert _batch_rows(plan) == [
        {"batch_id": "batch-0000", "cell_ids": ["cell-0"]},
        {"batch_id": "batch-0001", "cell_ids": ["cell-1"]},
    ]


def test_compact_resident_checkpoint_settlement_bounds_active_storage(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    plan["execution"]["execution_route_identity"] = "tick_native_execution_route_v1"
    plan["resource_estimate"] = {"physical_batch_cells": 1, "logical_cells": 2}
    plan["artifact_lifecycle"] = {"lifecycle_profile": "SCIENCE_COMPACT"}
    batch = plan["physical_batches"][0]
    manifest_path = tmp_path / "batches" / batch["batch_id"] / "manifest.json"
    bundle = (
        manifest_path.parent
        / "checkpoint_bundles_v2"
        / "cell-0"
        / "bundle_tick000000005000"
    )
    bundle.mkdir(parents=True)
    (bundle / "world.json").write_text("terminal", encoding="utf-8")
    manifest = {
        "status": "completed",
        "checkpoint": {
            "cell_bundles": [{"cell_id": "cell-0", "path": str(bundle)}],
        },
        "result": {
            "checkpoint_catalog": {"5000": {"cell_bundles": [{"path": str(bundle)}]}}
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    settled = _compact_resident_checkpoint_settle_v1(
        plan=plan,
        batch=batch,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    durable_bundle = (
        tmp_path
        / "durable"
        / "checkpoint_bundles_v2"
        / "cell-0"
        / "bundle_tick000000005000"
    )
    assert not (manifest_path.parent / "checkpoint_bundles_v2").exists()
    assert durable_bundle.is_dir()
    assert settled["checkpoint"]["cell_bundles"][0]["path"] == str(durable_bundle)
    assert settled["result"]["checkpoint_catalog"]["5000"]["cell_bundles"][0][
        "path"
    ] == str(durable_bundle)
    receipt = json.loads(
        (manifest_path.parent / "checkpoint_settlement_receipt.json").read_text()
    )
    assert receipt["status"] == "settled"
    assert receipt["bytes_moved"] > 0
    assert receipt["verified"] is True


def test_compact_resident_checkpoint_settlement_accepts_native_aggregate_file(
    tmp_path: Path,
) -> None:
    """Native ACS aggregate checkpoints use file authority, not Bundle-v2 dirs."""

    plan = _plan(tmp_path)
    plan["execution"]["execution_route_identity"] = "tick_native_execution_route_v1"
    plan["resource_estimate"] = {"physical_batch_cells": 1, "logical_cells": 2}
    plan["artifact_lifecycle"] = {"lifecycle_profile": "SCIENCE_COMPACT"}
    batch = plan["physical_batches"][0]
    manifest_path = tmp_path / "batches" / batch["batch_id"] / "manifest.json"
    aggregate = (
        manifest_path.parent
        / "checkpoint_native_aggregate_v1"
        / "checkpoint_tick000000000161.acsaggregate"
    )
    aggregate.parent.mkdir(parents=True)
    aggregate.write_bytes(b"native aggregate checkpoint")
    manifest = {
        "status": "completed",
        "checkpoint": {
            "cell_bundles": [
                {
                    "cell_id": "cell-0",
                    "checkpoint_tick": 161,
                    "physical_format": "acs100x_native_aggregate_checkpoint_v2",
                    "path": str(aggregate),
                }
            ],
        },
        "result": {
            "checkpoint_catalog": {
                "161": {
                    "cell_bundles": [
                        {
                            "cell_id": "cell-0",
                            "physical_format": "acs100x_native_aggregate_checkpoint_v2",
                            "path": str(aggregate),
                        }
                    ]
                }
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    settled = _compact_resident_checkpoint_settle_v1(
        plan=plan,
        batch=batch,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    durable_aggregate = (
        tmp_path
        / "durable"
        / "checkpoint_native_aggregate_v1"
        / batch["batch_id"]
        / aggregate.name
    )
    assert not aggregate.parent.exists()
    assert durable_aggregate.is_file()
    assert settled["checkpoint"]["cell_bundles"][0]["path"] == str(durable_aggregate)
    assert settled["result"]["checkpoint_catalog"]["161"]["cell_bundles"][0][
        "path"
    ] == str(durable_aggregate)
    receipt = json.loads(
        (manifest_path.parent / "checkpoint_settlement_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["checkpoint_storage_kind"] == "native_aggregate_file_v1"
    assert receipt["bytes_moved"] == len(b"native aggregate checkpoint")
    assert receipt["verified"] is True


class FakeProvider:
    provider_id = PRODUCTION_PROVIDER_ID
    profile_id = PROFILE

    def __init__(
        self, *, interrupt_once: str | None = None, crash_once: str | None = None
    ):
        self.interrupt_once = interrupt_once
        self.crash_once = crash_once
        self.executed: list[str] = []
        self.resumed: list[str] = []

    def translate_batch(
        self, plan: Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        row = next(
            item for item in plan["physical_batches"] if item["batch_id"] == batch_id
        )
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=tuple(row["cell_ids"]),
            generations=14,
            ticks_per_generation=80,
            checkpoint_milestones=(7, 14),
            parameter_arrays=(),
            seed_arrays=(),
        )

    def execute_batch(self, plan, request, context):
        del plan
        self.executed.append(request.batch_id)
        if self.crash_once == request.batch_id:
            self.crash_once = None
            context.record_checkpoint(
                {"generation": 7, "next_tick": 560, "path": "checkpoint.json"}
            )
            raise RuntimeError("simulated process loss")
        if self.interrupt_once == request.batch_id:
            self.interrupt_once = None
            return {
                "status": "interrupted",
                "checkpoint": {
                    "generation": 7,
                    "next_tick": 560,
                    "path": "checkpoint.json",
                },
                "result": {"reason": "bounded interruption"},
            }
        return self._completed(request)

    def resume_batch(self, plan, request, checkpoint, context):
        del plan, context
        assert checkpoint["generation"] == 7
        self.resumed.append(request.batch_id)
        return self._completed(request)

    @staticmethod
    def _completed(request):
        return {
            "status": "completed",
            "metrics_rows": [
                {"cell_id": request.cell_ids[0], "metrics": {"population.alive": 1}}
            ],
            "result": {"terminal_generation": 14},
        }


class RawMetricsProvider(FakeProvider):
    campaign_metric_rows = staticmethod(
        ProductionEngineLifecycleV1.campaign_metric_rows
    )

    @staticmethod
    def _completed(request):
        cell_id = request.cell_ids[0]
        return {
            "status": "completed",
            "metrics_rows": [
                {
                    "cell_id": cell_id,
                    "generation": generation,
                    "alive": 64,
                    "births": generation + 1,
                    "deaths": generation,
                }
                for generation in (0, 13)
            ],
            "result": {"terminal_generation": 14},
        }


def _registry(provider: Any) -> ExecutionProviderRegistryV1:
    registry = ExecutionProviderRegistryV1()
    registry.register(provider)
    return registry


def test_selected_batch_measurement_plan_preserves_plan_identity(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    narrowed = campaign_cli._selected_batch_measurement_plan(plan, "batch-0001")

    assert narrowed is not plan
    assert narrowed["plan_sha256"] == plan["plan_sha256"]
    assert narrowed["campaign_id"] == plan["campaign_id"]
    assert narrowed["physical_batches"] == [plan["physical_batches"][1]]
    assert plan["physical_batches"][0]["batch_id"] == "batch-0000"


def test_selected_batch_measurement_plan_rejects_unknown_batch(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)

    with pytest.raises(CampaignError, match="unknown physical batch"):
        campaign_cli._selected_batch_measurement_plan(plan, "missing-batch")


def test_multi_prepared_owner_receipts_promote_all_physical_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.campaigns import preflight as preflight_module

    plan = _plan(tmp_path)
    receipt = {
        "ok": True,
        "ready_for_execution": False,
        "execution_authority_reasons": [
            {"code": "provider_vram_bytes_upper_bound_unavailable"}
        ],
        "resources": {},
        "prepared_owner_resource_audit": {
            "batch_ids": ["batch-0000", "batch-0001"],
        },
    }

    def fake_preflight(
        narrowed_plan,
        *,
        prepared_owner_resource_receipt=None,
        **_kwargs,
    ):
        batch_id = narrowed_plan["physical_batches"][0]["batch_id"]
        request_sha = f"request:{batch_id}"
        if prepared_owner_resource_receipt is None:
            return {
                "prepared_owner_resource_audit": {
                    "measurement_authorization": {
                        "request_bindings": {
                            "batch_request_sha256": request_sha,
                        },
                    },
                },
            }
        assert (
            prepared_owner_resource_receipt["request"]["batch_request_sha256"]
            == request_sha
        )
        return {
            "ready_for_execution": True,
            "prepared_owner_resource_audit": {
                "receipt_promotion": {
                    "vram_bytes_upper_bound": (
                        1000 if batch_id == "batch-0000" else 1200
                    ),
                },
            },
        }

    monkeypatch.setattr(preflight_module, "preflight_campaign", fake_preflight)

    campaign_cli._promote_multi_prepared_owner_receipts(
        receipt,
        plan=plan,
        profile=None,
        provider_registry=None,
        prepared_owner_receipts=[
            {
                "receipt_sha256": "a" * 64,
                "request": {"batch_request_sha256": "request:batch-0001"},
            },
            {
                "receipt_sha256": "b" * 64,
                "request": {"batch_request_sha256": "request:batch-0000"},
            },
        ],
    )

    promotion = receipt["prepared_owner_resource_audit"]["multi_receipt_promotion"]
    assert receipt["ready_for_execution"] is True
    assert receipt["execution_authority_reasons"] == []
    assert receipt["resources"]["vram_bytes_upper_bound"] == 1200
    assert promotion["promoted"] is True
    assert [row["batch_id"] for row in promotion["batch_promotions"]] == [
        "batch-0000",
        "batch-0001",
    ]


def test_completed_batches_are_skipped_by_idempotent_redispatch(tmp_path: Path) -> None:
    provider = FakeProvider()
    plan = _plan(tmp_path)
    registry = _registry(provider)

    first = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)
    second = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    assert first["status"] == second["status"] == "completed"
    assert provider.executed == ["batch-0000", "batch-0001"]
    assert provider.resumed == []
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert manifest["counts"]["completed"] == 2
    metrics = json.loads((tmp_path / "cell_metrics.json").read_text())
    assert [row["cell_id"] for row in metrics["rows"]] == ["cell-0", "cell-1"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_campaign_passes_one_owned_progressive_storage_guard_to_every_batch(
    tmp_path: Path,
) -> None:
    class GuardCapturingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.guards: list[RunStorageGuard] = []

        def execute_batch(self, plan, request, context):
            assert isinstance(context.storage_guard, RunStorageGuard)
            self.guards.append(context.storage_guard)
            return super().execute_batch(plan, request, context)

    provider = GuardCapturingProvider()
    plan = _plan(tmp_path)
    plan["resource_estimate"] = {
        "disk_bytes_upper_bound": 4096,
        "storage_reservation_mode": "bounded_incremental_guarded_v1",
        "storage_reservation_guard_scope": (
            "provider_dispatch_and_checkpoint_boundary_v1"
        ),
        "storage_reservation_control_bytes_upper_bound": 128,
        "storage_reservation_dispatch_increment_bytes_upper_bound": 256,
        "storage_reservation_boundary_increment_bytes_upper_bound": 512,
        "storage_reservation_emergency_checkpoint_bytes_upper_bound": 1024,
        "storage_reservation_window_bytes_upper_bound": 1408,
        "storage_reservation_total_envelope_bytes_upper_bound": 4096,
    }

    execute_campaign(plan, run_root=tmp_path, provider_registry=_registry(provider))

    assert len(provider.guards) == 2
    assert provider.guards[0] is provider.guards[1]
    assert provider.guards[0].run_root == tmp_path.resolve()
    assert provider.guards[0].planned_bytes == 4096
    assert provider.guards[0].reservation_window_bytes == 1408


def test_terminal_lifecycle_reseal_refreshes_front_door_control_files(
    tmp_path: Path,
) -> None:
    """Final V2 control writes cannot leave stale manifest identities."""

    provider = FakeProvider()
    plan = _plan(tmp_path)
    execute_campaign(plan, run_root=tmp_path, provider_registry=_registry(provider))
    final_controls = {
        "baseline_conformance_receipt.json": '{"phase":"terminal-conformance"}\n',
        "execution_state.json": '{"phase":"terminal-state"}\n',
        "execution_result.json": '{"phase":"terminal-result"}\n',
        "run_passport.json": '{"phase":"terminal-passport"}\n',
    }
    for relative, payload in final_controls.items():
        (tmp_path / relative).write_text(payload, encoding="utf-8")
    row_relative = "batches/batch-0000/workhorse-technical-row-receipt.json"
    row_path = tmp_path / row_relative
    row_path.write_text('{"row":"terminalized"}\n', encoding="utf-8")

    resealed = reseal_terminal_lifecycle_inventory_v1(
        plan,
        run_root=tmp_path,
        additional_artifacts={
            row_relative: {
                "artifact_family": "qualification.workhorse_technical_row_v2",
                "retention_class": "R3",
                "owner_writer": "workhorse-v2-single-row-owner",
                "release_predicate": "retained for offline matrix aggregation",
            }
        },
    )
    manifest = resealed["manifest"]
    rows = {str(row["path"]): row for row in manifest["artifacts"]}
    for relative in final_controls:
        path = tmp_path / relative
        digest = sha256(path.read_bytes()).hexdigest()
        assert rows[relative]["bytes"] == path.stat().st_size
        assert rows[relative]["sha256"] == digest
        assert rows[relative]["artifact_family"] == "lifecycle.control"
        assert rows[relative]["retention_class"] == "R5"
    assert rows[row_relative]["sha256"] == sha256(row_path.read_bytes()).hexdigest()
    assert rows[row_relative]["retention_class"] == "R3"
    assert rows[row_relative]["artifact_family"] == (
        "qualification.workhorse_technical_row_v2"
    )
    receipt = json.loads(
        (tmp_path / "storage_disposition_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["manifest_sha256"] == manifest["manifest_sha256"]


def test_terminal_lifecycle_reseal_preserves_large_accepted_exact_inventory(
    tmp_path: Path,
) -> None:
    """A bounded metadata reseal must not shrink an admitted producer inventory."""

    plan = _plan(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    inventory = []
    for index in range(4097):
        relative = f"scratch/tick-{index:05d}.json"
        payload = bytes([index % 251])
        (tmp_path / relative).write_bytes(payload)
        inventory.append(
            {
                "path": relative,
                "artifact_family": "science.native_tick_receipt",
                "artifact_type": "file",
                "owner_writer": "science-provider",
                "bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
                "retention_class": "R0",
                "release_predicate": "terminal compact replacement available",
            }
        )
    disposition = build_artifact_disposition_manifest_v1(
        plan,
        run_root=tmp_path,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
        max_objects=len(inventory),
    )
    (tmp_path / "artifact_disposition_manifest.json").write_text(
        json.dumps(disposition), encoding="utf-8"
    )
    for relative in (
        "baseline_conformance_receipt.json",
        "execution_state.json",
        "execution_result.json",
        "run_passport.json",
    ):
        (tmp_path / relative).write_text("{}\n", encoding="utf-8")

    resealed = reseal_terminal_lifecycle_inventory_v1(plan, run_root=tmp_path)

    rows = resealed["manifest"]["artifacts"]
    assert len(rows) == len(inventory) + 4
    assert sum(
        row["artifact_family"] == "science.native_tick_receipt" for row in rows
    ) == len(inventory)


def test_completed_campaign_publishes_one_verified_chronicle_archive(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-001"
    plan = _plan(run_root)
    history_root = tmp_path / "chronicle-history"
    plan["output"].update(
        {
            "chronicle_profile": "notable",
            "chronicle_history_root": str(history_root),
            "chronicle_archive_enabled": True,
        }
    )
    registry = _registry(FakeProvider())

    first = execute_campaign(plan, run_root=run_root, provider_registry=registry)
    second = execute_campaign(plan, run_root=run_root, provider_registry=registry)

    archive = history_root / "fake-campaign" / "run-001" / "chronicles.zip"
    assert first["chronicle_archive_path"] == str(archive)
    assert second["chronicle_archive_path"] == str(archive)
    assert archive.is_file()
    receipt = json.loads((run_root / "chronicle_archive_receipt.json").read_text())
    assert receipt["verified"] is True
    assert receipt["world_ids"] == ["cell-0", "cell-1"]
    assert receipt["loose_materialized_sources_removed"] is True
    assert not (run_root / "chronicle_worlds").exists()


def test_chronicle_archive_collision_uses_bound_execution_identity(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run-001"
    plan = _plan(run_root)
    history_root = tmp_path / "chronicle-history"
    plan["output"].update(
        {
            "chronicle_profile": "notable",
            "chronicle_history_root": str(history_root),
            "chronicle_archive_enabled": True,
        }
    )
    foreign_world = tmp_path / "foreign-world"
    foreign_world.mkdir()
    (foreign_world / "foreign.json").write_text('{"foreign":true}\n')
    legacy = create_completed_run_archive(
        history_root,
        campaign_id="fake-campaign",
        run_id=run_root.name,
        worlds={"foreign-world": foreign_world},
    )
    legacy_bytes = legacy.archive_path.read_bytes()

    result = execute_campaign(
        plan,
        run_root=run_root,
        provider_registry=_registry(FakeProvider()),
    )

    expected_run_id = _chronicle_archive_run_id(
        run_root,
        campaign_id="fake-campaign",
        plan_sha256="a" * 64,
        profile_id=PROFILE,
        provider_id=FakeProvider.provider_id,
        execution_route_sha256=None,
    )
    expected_archive = (
        history_root / "fake-campaign" / expected_run_id / "chronicles.zip"
    )
    receipt = json.loads((run_root / "chronicle_archive_receipt.json").read_text())
    assert result["chronicle_archive_path"] == str(expected_archive)
    assert receipt["run_id"] == expected_run_id
    assert receipt["run_id_scheme"] == "bound_execution_v2"
    assert expected_archive.is_file()
    assert legacy.archive_path.read_bytes() == legacy_bytes


def test_provider_owned_metric_projection_repairs_public_output_without_redispatch(
    tmp_path: Path,
) -> None:
    provider = RawMetricsProvider()
    plan = _plan(tmp_path)
    registry = _registry(provider)

    execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    batch = json.loads((tmp_path / "batches/batch-0000/manifest.json").read_text())
    assert len(batch["metrics_rows"]) == 2
    assert "metrics" not in batch["metrics_rows"][0]
    metrics_path = tmp_path / "cell_metrics.json"
    public = json.loads(metrics_path.read_text())
    assert public["rows"][0] == {
        "cell_id": "cell-0",
        "metrics": {
            "population.alive": 64,
            "population.births": 14,
            "population.deaths": 13,
        },
    }

    metrics_path.write_text("{}\n", encoding="utf-8")
    execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    repaired = json.loads(metrics_path.read_text())
    assert repaired == public
    assert provider.executed == ["batch-0000", "batch-0001"]


def test_pre_lifecycle_closeout_hook_seals_consumer_artifacts_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async consumer output must be complete before lifecycle inventory sealing."""

    monkeypatch.setattr(
        orchestration_module,
        "register_pending_durable_archive_v1",
        lambda *_args, **_kwargs: None,
    )
    plan = _plan(tmp_path)
    events: list[str] = []

    def finish_consumer() -> None:
        events.append("consumer-finished")
        output = tmp_path / "reducer-artifacts" / "consumer-output.bin"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"sealed consumer output")

    result = execute_campaign(
        plan,
        run_root=tmp_path,
        provider_registry=_registry(FakeProvider()),
        before_lifecycle_closeout_fn=finish_consumer,
    )

    assert result["status"] == "completed"
    assert events == ["consumer-finished"]
    disposition = json.loads(
        (tmp_path / "artifact_disposition_manifest.json").read_text()
    )
    row = next(
        row
        for row in disposition["artifacts"]
        if row["path"] == "reducer-artifacts/consumer-output.bin"
    )
    assert row["bytes"] == len(b"sealed consumer output")
    assert row["sha256"] == sha256(b"sealed consumer output").hexdigest()


def test_interrupted_batch_resumes_while_completed_batch_is_skipped(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(interrupt_once="batch-0001")
    plan = _plan(tmp_path)
    registry = _registry(provider)

    interrupted = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)
    completed = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    assert interrupted["status"] == "interrupted"
    assert completed["status"] == "completed"
    assert provider.executed == ["batch-0000", "batch-0001"]
    assert provider.resumed == ["batch-0001"]
    batch = json.loads((tmp_path / "batches/batch-0001/manifest.json").read_text())
    assert batch["attempts"] == 2
    assert batch["status"] == "completed"


def test_storage_budget_wait_publishes_resumable_fail_closed_closeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    plan = _plan(tmp_path)
    blocked = {
        "schema_version": "eiel_storage_failure_receipt_v1",
        "status": "STORAGE_BUDGET_BLOCKED",
        "snapshot": {"free_bytes": 0, "required_free_bytes": 100},
    }

    def enforce(
        _guard: object,
        *,
        stage: str,
        pre_admission_reclaimer: object | None = None,
    ) -> dict[str, Any]:
        raise StorageBudgetExceeded(
            "synthetic storage pressure",
            receipt={**blocked, "stage": stage},
        )

    monkeypatch.setattr(orchestration_module.RunStorageGuard, "enforce", enforce)

    result = execute_campaign(
        plan, run_root=tmp_path, provider_registry=_registry(provider)
    )

    assert result["status"] == "interrupted"
    assert result["storage_wait"] is True
    assert provider.executed == []
    wait = json.loads((tmp_path / "storage_wait_receipt.json").read_text())
    assert wait["status"] == "WAIT_STORAGE"
    disposition = json.loads(
        (tmp_path / "artifact_disposition_manifest.json").read_text()
    )
    assert disposition["terminal_status"] == "interrupted"
    receipt = json.loads((tmp_path / "storage_disposition_receipt.json").read_text())
    assert receipt["result"] == "storage_wait"


def test_checkpoint_callback_makes_failed_attempt_resumable(tmp_path: Path) -> None:
    provider = FakeProvider(crash_once="batch-0000")
    plan = _plan(tmp_path)
    registry = _registry(provider)

    with pytest.raises(CampaignExecutionError, match="simulated process loss"):
        execute_campaign(plan, run_root=tmp_path, provider_registry=registry)
    recovered = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    assert recovered["status"] == "completed"
    assert provider.executed == ["batch-0000", "batch-0001"]
    assert provider.resumed == ["batch-0000"]


def test_batch_context_seals_exact_provider_artifact_idempotently(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batches" / "batch-0000"
    batch_root.mkdir(parents=True)
    manifest_path = batch_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_batch_manifest_v1",
                "status": "running",
                "artifact_inventory": [],
                "artifact_inventory_complete": True,
            }
        ),
        encoding="utf-8",
    )
    artifact = batch_root / "scratch" / "semantic.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"sealed-semantic-output")
    context = orchestration_module.BatchExecutionContextV1(
        run_root=tmp_path,
        batch_root=batch_root,
        manifest_path=manifest_path,
        provider_id="test-provider",
    )

    first = context.record_artifact(
        artifact,
        artifact_family="test.semantic_output",
        retention_class="R0",
        release_predicate="terminal acknowledgement sealed",
    )
    second = context.record_artifact(
        artifact,
        artifact_family="test.semantic_output",
        retention_class="R0",
        release_predicate="terminal acknowledgement sealed",
    )

    assert first == second
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_inventory_complete"] is True
    assert manifest["artifact_inventory"] == [first]
    assert first["path"] == "batches/batch-0000/scratch/semantic.bin"
    assert first["owner_writer"] == "test-provider"


@pytest.mark.parametrize(
    "relative",
    [
        "canopy_relay_supervision.json",
        "execution_plan.json",
        "execution_request.json",
        "cells/cell-a/completion.json",
    ],
)
def test_campaign_closeout_control_artifacts_are_retained_not_rx(
    relative: str,
) -> None:
    assert orchestration_module._lifecycle_inventory_retention_v1(relative) == (
        "R5",
        "lifecycle.control",
        "control/closeout authority is retained",
    )


@pytest.mark.parametrize(
    ("crash_stage", "exit_code", "expected_generation"),
    (
        ("before_catalog_replace", 91, 7),
        ("after_catalog_replace", 92, 14),
    ),
)
def test_checkpoint_catalog_process_crash_preserves_atomic_durable_authority(
    tmp_path: Path,
    crash_stage: str,
    exit_code: int,
    expected_generation: int,
) -> None:
    """A restart sees either the prior or successor catalog, never torn state."""

    manifest_path = tmp_path / "manifest.json"
    route_sha256 = "6" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_batch_manifest_v1",
                "status": "running",
                "checkpoint": {
                    "generation": 7,
                    "execution_route_sha256": route_sha256,
                    "cell_bundles": [{"cell_id": "cell-0", "path": "prior"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    crash_script = r"""
import os
from pathlib import Path
import sys

from engine.campaigns import orchestration

manifest_path = Path(sys.argv[1]).resolve()
stage = sys.argv[2]
exit_code = int(sys.argv[3])
original_replace = orchestration.os.replace

def crashing_replace(source, destination):
    if Path(destination).resolve() != manifest_path:
        return original_replace(source, destination)
    if stage == "before_catalog_replace":
        os._exit(exit_code)
    original_replace(source, destination)
    os._exit(exit_code)

orchestration.os.replace = crashing_replace
context = orchestration.BatchExecutionContextV1(
    run_root=manifest_path.parent,
    batch_root=manifest_path.parent,
    manifest_path=manifest_path,
    execution_route_sha256="6" * 64,
)
context.record_checkpoint(
    {
        "generation": 14,
        "cell_bundles": [{"cell_id": "cell-0", "path": "successor"}],
    }
)
os._exit(99)
"""

    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            crash_script,
            str(manifest_path),
            crash_stage,
            str(exit_code),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
    )

    assert crashed.returncode == exit_code
    restarted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert restarted["checkpoint"]["generation"] == expected_generation
    assert restarted["checkpoint"]["execution_route_sha256"] == route_sha256
    if expected_generation == 7:
        assert restarted["checkpoint"]["cell_bundles"][0]["path"] == "prior"
        assert len(list(tmp_path.glob(".manifest.json.*.tmp"))) == 1
    else:
        assert restarted["checkpoint"]["cell_bundles"][0]["path"] == "successor"
        assert list(tmp_path.glob(".manifest.json.*.tmp")) == []


def test_missing_execution_capability_fails_closed(tmp_path: Path) -> None:
    class TranslationOnlyProvider:
        provider_id = "translation-only"
        profile_id = PROFILE

    with pytest.raises(CampaignExecutionError, match="no batch execution capability"):
        execute_campaign(
            _plan(tmp_path),
            run_root=tmp_path,
            provider_registry=_registry(TranslationOnlyProvider()),
        )


def test_direct_execution_requires_bound_preflight_authority(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    (tmp_path / "preflight_receipt.json").unlink()
    with pytest.raises(CampaignExecutionError, match="requires a durable preflight"):
        execute_campaign(
            plan, run_root=tmp_path, provider_registry=_registry(FakeProvider())
        )

    _plan(tmp_path)
    receipt_path = tmp_path / "preflight_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["plan_sha256"] = "b" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(CampaignExecutionError, match="plan_sha256"):
        execute_campaign(
            plan, run_root=tmp_path, provider_registry=_registry(FakeProvider())
        )


def test_resume_cli_helper_dispatches_through_explicit_registry(tmp_path: Path) -> None:
    provider = FakeProvider(interrupt_once="batch-0000")
    plan = _plan(tmp_path)
    registry = _registry(provider)
    (tmp_path / "execution_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "preflight_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_preflight_v1",
                "campaign_id": plan["campaign_id"],
                "ready_for_execution": True,
                "plan_sha256": plan["plan_sha256"],
            }
        ),
        encoding="utf-8",
    )

    first = _resume(tmp_path, provider_registry=registry)
    second = _resume(tmp_path, provider_registry=registry)

    assert first["status"] == "interrupted"
    assert second["status"] == "completed"
    assert provider.resumed == ["batch-0000"]


def test_run_cli_helper_dispatches_preflighted_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider()
    plan = _plan(tmp_path)
    registry = _registry(provider)
    plan_path = tmp_path / "execution_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(
        campaign_cli,
        "_preflight",
        lambda spec: {
            "ready_for_execution": True,
            "execution_plan_path": str(plan_path),
        },
    )

    result = campaign_cli._run(tmp_path / "campaign.yaml", provider_registry=registry)

    assert result["status"] == "completed"
    assert provider.executed == ["batch-0000", "batch-0001"]


class ComparisonProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        self.request_shapes: list[tuple[str, tuple[int, ...], str]] = []

    def execute_batch(self, plan, request, context):
        del plan
        self.executed.append(request.batch_id)
        self.request_shapes.append(
            (request.batch_id, request.checkpoint_milestones, "execute")
        )
        return self._branch_completed(request, context)

    def resume_batch(self, plan, request, checkpoint, context):
        del plan
        assert checkpoint["generation"] <= request.generations
        self.resumed.append(request.batch_id)
        self.request_shapes.append(
            (request.batch_id, request.checkpoint_milestones, "resume")
        )
        return self._branch_completed(request, context)

    @staticmethod
    def _branch_completed(request, context):
        checkpoint = {
            "generation": request.generations,
            "terminal_cell_digests": {
                request.cell_ids[0]: f"digest-{request.cell_ids[0]}"
            },
        }
        context.record_checkpoint(checkpoint)
        return {
            "status": "completed",
            "checkpoint": checkpoint,
            "metrics_rows": [
                {"cell_id": request.cell_ids[0], "metrics": {"population.alive": 1}}
            ],
            "result": {"terminal_generation": request.generations},
        }


@pytest.mark.parametrize("storage_mode", ["disk", "disk_and_memory"])
def test_comparison_request_preserves_checkpoint_storage_mode(
    storage_mode: str,
) -> None:
    request = ResidentBatchRequestV1(
        profile_id=PROFILE,
        batch_id="batch-0000",
        cell_ids=("cell-0", "cell-1"),
        generations=14,
        ticks_per_generation=80,
        checkpoint_milestones=(7, 14),
        parameter_arrays=(("parameter", (1, 2)),),
        seed_arrays=(("ecology", (101, 202)),),
        agent_id_allocation_generation_count=14,
        checkpoint_storage_mode=storage_mode,
        genome_v2_checkpoint_authority_binding=(
            ("package_id_sha256", "a" * 64),
            ("execution_route_sha256", "b" * 64),
        ),
    )

    narrowed = _narrow_request(
        request,
        branch_id="repeat-cell-1",
        source_index=1,
        source_cell_id="cell-1",
        generations=14,
        checkpoints=(7, 14),
    )

    assert narrowed.checkpoint_storage_mode == storage_mode
    assert narrowed.cell_ids == ("cell-1",)
    assert narrowed.parameter_arrays == (("parameter", (2,)),)
    assert narrowed.seed_arrays == (("ecology", (202,)),)
    assert narrowed.agent_id_allocation_generation_count == 14
    assert (
        narrowed.genome_v2_checkpoint_authority_binding
        == request.genome_v2_checkpoint_authority_binding
    )


def test_tick_native_comparison_preserves_partial_terminal_horizon() -> None:
    request = ResidentBatchRequestV1(
        profile_id=PROFILE,
        batch_id="batch-0000",
        cell_ids=("cell-0",),
        generations=32,
        ticks_per_generation=160,
        checkpoint_milestones=(32,),
        parameter_arrays=(("parameter", (1,)),),
        seed_arrays=(("ecology", (101,)),),
        agent_id_allocation_generation_count=32,
        total_ticks=5000,
        checkpoint_ticks=(5000,),
        source_window_ticks=160,
    )

    terminal = _narrow_request(
        request,
        branch_id="repeat-cell-0",
        source_index=0,
        source_cell_id="cell-0",
        generations=32,
        checkpoints=(32,),
    )
    split = _narrow_request(
        request,
        branch_id="resume-cell-0-g7",
        source_index=0,
        source_cell_id="cell-0",
        generations=7,
        checkpoints=(7,),
    )

    assert terminal.total_ticks == 5000
    assert terminal.checkpoint_ticks == (5000,)
    assert split.total_ticks == 1120
    assert split.checkpoint_ticks == (1120,)


def _comparison_plan(run_root: Path) -> dict[str, Any]:
    plan = _plan(run_root)
    plan["comparisons"] = {
        "reporting_parity": {
            "required": True,
            "accepted_evidence_modes": [
                "enabled_disabled_exact_terminal_state",
                "provider_observational_contract_with_state_mutation_sentinel",
            ],
        },
        "branches": [
            {
                "ordinal": 0,
                "branch_id": "repeat-cell-0",
                "kind": "fresh_repeat",
                "source_cell_id": "cell-0",
                "repeat_ordinal": 1,
                "split_generation": None,
            },
            {
                "ordinal": 1,
                "branch_id": "resume-cell-0-g7",
                "kind": "checkpoint_continuation",
                "source_cell_id": "cell-0",
                "repeat_ordinal": None,
                "split_generation": 7,
            },
        ],
    }
    return plan


def test_comparison_branches_are_durable_exact_and_idempotent(tmp_path: Path) -> None:
    provider = ComparisonProvider()
    plan = _comparison_plan(tmp_path)
    registry = _registry(provider)

    first = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)
    second = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    assert first["status"] == second["status"] == "completed"
    assert first["comparison_counts"]["completed"] == 2
    assert provider.executed == [
        "batch-0000",
        "batch-0001",
        "repeat-cell-0",
        "resume-cell-0-g7-uninterrupted",
        "resume-cell-0-g7",
    ]
    assert provider.resumed == ["resume-cell-0-g7"]
    assert (
        "resume-cell-0-g7-uninterrupted",
        (14,),
        "execute",
    ) in provider.request_shapes
    assert ("resume-cell-0-g7", (7,), "execute") in provider.request_shapes
    assert ("resume-cell-0-g7", (7, 14), "resume") in provider.request_shapes
    continuation = json.loads(
        (tmp_path / "comparisons/resume-cell-0-g7/manifest.json").read_text()
    )
    assert continuation["comparison"]["match"] is True
    assert continuation["comparison"]["baseline_execution"] == (
        "fresh_terminal_only_uninterrupted_v1"
    )
    assert continuation["comparison"]["compared_surfaces"] == [
        "metrics_rows_sha256",
        "terminal_checkpoint_digest",
    ]
    assert continuation["attempts"] == 1
    reporting = json.loads((tmp_path / "reporting_parity_receipt.json").read_text())
    assert reporting["qualification_evidence_status"] == "reporting_parity_unsupported"
    assert reporting["reason_code"] == "provider_reporting_parity_contract_unavailable"


def test_comparison_uses_public_metrics_not_route_local_audit_fields(
    tmp_path: Path,
) -> None:
    class RichComparisonProvider(ComparisonProvider):
        @staticmethod
        def _branch_completed(request, context):
            result = ComparisonProvider._branch_completed(request, context)
            result["metrics_rows"][0] = {
                "cell_id": request.cell_ids[0],
                "metrics": {"population.alive": 1},
                "genome_v2_attribution": {
                    "summary_sha256": f"audit:{request.batch_id}",
                },
            }
            return result

        @staticmethod
        def campaign_metric_rows(rows, *, cell_ids):
            del cell_ids
            return [
                {"cell_id": row["cell_id"], "metrics": row["metrics"]} for row in rows
            ]

    provider = RichComparisonProvider()
    plan = _comparison_plan(tmp_path)
    plan["comparisons"]["branches"] = plan["comparisons"]["branches"][:1]

    result = execute_campaign(
        plan,
        run_root=tmp_path,
        provider_registry=_registry(provider),
    )

    assert result["status"] == "completed"
    manifest = json.loads(
        (tmp_path / "comparisons/repeat-cell-0/manifest.json").read_text()
    )
    assert manifest["comparison"]["match"] is True
    assert manifest["comparison"]["compared_surfaces"] == [
        "metrics_rows_sha256",
        "terminal_checkpoint_digest",
    ]


def test_comparison_mismatch_fails_closed_with_durable_error(tmp_path: Path) -> None:
    class MismatchProvider(ComparisonProvider):
        @staticmethod
        def _branch_completed(request, context):
            result = ComparisonProvider._branch_completed(request, context)
            if request.batch_id.startswith("repeat-"):
                result["metrics_rows"][0]["metrics"]["population.alive"] = 2
            return result

    provider = MismatchProvider()
    plan = _comparison_plan(tmp_path)
    plan["comparisons"]["branches"] = plan["comparisons"]["branches"][:1]

    with pytest.raises(CampaignExecutionError, match="metrics_rows_sha256"):
        execute_campaign(plan, run_root=tmp_path, provider_registry=_registry(provider))

    manifest = json.loads(
        (tmp_path / "comparisons/repeat-cell-0/manifest.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["checkpoint"]["generation"] == 14
    comparisons = json.loads((tmp_path / "comparisons_manifest.json").read_text())
    assert comparisons["status"] == "failed"


def test_comparison_manifest_plan_binding_is_enforced(tmp_path: Path) -> None:
    provider = ComparisonProvider()
    plan = _comparison_plan(tmp_path)
    execute_campaign(plan, run_root=tmp_path, provider_registry=_registry(provider))
    path = tmp_path / "comparisons/repeat-cell-0/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_cell_id"] = "cell-1"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CampaignExecutionError, match="binding mismatch"):
        execute_campaign(plan, run_root=tmp_path, provider_registry=_registry(provider))


def test_interrupted_comparison_resumes_without_repeating_dispatch(
    tmp_path: Path,
) -> None:
    class InterruptComparisonProvider(ComparisonProvider):
        interrupted = False

        def resume_batch(self, plan, request, checkpoint, context):
            if request.generations == 14 and not self.interrupted:
                self.interrupted = True
                self.resumed.append(request.batch_id)
                context.record_checkpoint(dict(checkpoint))
                return {
                    "status": "interrupted",
                    "checkpoint": dict(checkpoint),
                    "metrics_rows": checkpoint.get("metrics_rows", []),
                    "result": {"reason": "bounded interruption"},
                }
            return super().resume_batch(plan, request, checkpoint, context)

    provider = InterruptComparisonProvider()
    plan = _comparison_plan(tmp_path)
    plan["comparisons"]["branches"] = plan["comparisons"]["branches"][1:]
    registry = _registry(provider)

    interrupted = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)
    completed = execute_campaign(plan, run_root=tmp_path, provider_registry=registry)

    assert interrupted["status"] == "interrupted"
    assert completed["status"] == "completed"
    assert provider.executed.count("resume-cell-0-g7") == 1
    assert provider.resumed.count("resume-cell-0-g7") == 2


def test_reporting_disabled_branch_proves_exact_terminal_state_parity(
    tmp_path: Path,
) -> None:
    class ReportingProvider(ComparisonProvider):
        reporting_parity_supported = True
        reporting_flags: list[bool] = []

        def execute_batch(self, plan, request, context):
            self.reporting_flags.append(request.reporting_enabled)
            return super().execute_batch(plan, request, context)

    provider = ReportingProvider()
    plan = _comparison_plan(tmp_path)
    plan["comparisons"]["branches"] = [
        {
            "ordinal": 0,
            "branch_id": "reporting-cell-0",
            "kind": "reporting_parity",
            "source_cell_id": "cell-0",
            "repeat_ordinal": None,
            "split_generation": None,
        }
    ]

    result = execute_campaign(
        plan, run_root=tmp_path, provider_registry=_registry(provider)
    )

    assert result["status"] == "completed"
    assert provider.reporting_flags == [True, True, False]
    receipt = json.loads((tmp_path / "reporting_parity_receipt.json").read_text())
    assert receipt["qualification_evidence_status"] == "reporting_parity_proven"
    assert receipt["evidence_mode"] == "enabled_disabled_exact_terminal_state"
    assert receipt["state_mutation_sentinel"] == "digest-cell-0"


def test_adaptive_reporting_parity_resumes_the_same_parent_checkpoint(
    tmp_path: Path,
) -> None:
    class AdaptiveReportingProvider(ComparisonProvider):
        reporting_parity_supported = True
        reporting_requests: list[tuple[str, tuple[str, ...], bool, str]] = []

        def execute_batch(self, plan, request, context):
            if request.batch_id != "batch-0000":
                return super().execute_batch(plan, request, context)
            self.executed.append(request.batch_id)
            self.reporting_requests.append(
                (
                    request.batch_id,
                    request.cell_ids,
                    request.reporting_enabled,
                    "execute",
                )
            )
            adaptive = {
                "schema_version": "wrc_checkpoint_adaptive_rebinding_v1",
                "capability_id": "WRC_CHECKPOINT_ADAPTIVE_REBINDING_V1",
                "source_generation": 7,
                "restore_generation": 14,
                "cell_ids": list(request.cell_ids),
                "execution_route_sha256": request.execution_route_sha256,
            }
            parent = {
                "generation": 7,
                "wrc_checkpoint_adaptive_rebinding_v1": adaptive,
            }
            checkpoint = {
                "generation": 14,
                "terminal_cell_digests": {
                    cell_id: f"adaptive-digest-{cell_id}"
                    for cell_id in request.cell_ids
                },
            }
            context.record_checkpoint(checkpoint)
            return {
                "status": "completed",
                "checkpoint": checkpoint,
                "metrics_rows": [
                    {"cell_id": cell_id, "metrics": {"population.alive": 1}}
                    for cell_id in request.cell_ids
                ],
                "result": {
                    "terminal_generation": 14,
                    "checkpoint_catalog": {"7": parent, "14": checkpoint},
                    "fresh_owner_restore_receipt": {
                        "wrc_checkpoint_adaptive_rebinding": adaptive,
                    },
                },
            }

        def resume_batch(self, plan, request, checkpoint, context):
            del plan
            assert checkpoint["generation"] == 7
            self.resumed.append(request.batch_id)
            self.reporting_requests.append(
                (
                    request.batch_id,
                    request.cell_ids,
                    request.reporting_enabled,
                    "resume",
                )
            )
            terminal = {
                "generation": 14,
                "terminal_cell_digests": {
                    cell_id: f"adaptive-digest-{cell_id}"
                    for cell_id in request.cell_ids
                },
            }
            context.record_checkpoint(terminal)
            return {
                "status": "completed",
                "checkpoint": terminal,
                "metrics_rows": [
                    {"cell_id": cell_id, "metrics": {"population.alive": 1}}
                    for cell_id in request.cell_ids
                ],
                "result": {"terminal_generation": 14},
            }

    provider = AdaptiveReportingProvider()
    plan = _comparison_plan(tmp_path)
    plan["comparisons"]["branches"] = [
        {
            "ordinal": 0,
            "branch_id": "reporting-cell-0",
            "kind": "reporting_parity",
            "source_cell_id": "cell-0",
            "repeat_ordinal": None,
            "split_generation": None,
        }
    ]

    result = execute_campaign(
        plan, run_root=tmp_path, provider_registry=_registry(provider)
    )

    assert result["status"] == "completed"
    assert provider.reporting_requests == [
        ("batch-0000", ("cell-0",), True, "execute"),
        ("batch-0000", ("cell-0",), False, "resume"),
    ]
    manifest = json.loads(
        (tmp_path / "comparisons/reporting-cell-0/manifest.json").read_text()
    )
    assert manifest["comparison"]["match"] is True
    assert manifest["comparison"]["adaptive_parent_checkpoint_generation"] == 7
    assert manifest["comparison"]["candidate_execution"] == (
        "same_parent_adaptive_reporting_disabled_v1"
    )
