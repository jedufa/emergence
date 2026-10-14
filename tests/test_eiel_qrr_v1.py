from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
from jsonschema import validate as validate_json_schema

from engine.campaigns.canonical import canonical_sha256, pretty_json
from engine.campaigns.contracts import ExecutionRouteContractV1
from engine.campaigns.providers import (
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
)
from engine.campaigns.tick_native import TickNativeError, TickNativeExecutionRequestV1
from engine.campaigns.qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION
from engine.campaigns.orchestration import _subset_request
from engine.qrr.cli import (
    CANOPY_HOST_ACTIVE_SCAN_PER_CHECKPOINT_BYTES,
    CANOPY_HOST_RESERVE_PER_CELL_BYTES,
    QrrLauncherError,
    _apply_acs_checkpoint_selection_v1,
    _bind_science_materialization_v1,
    _complete_production_resource_admission_v1,
    _configure_checkpoint_resume_probe_v1,
    _rebind_technical_gate_selector_v1,
    _write_runability_technical_gate_receipt_v1,
    assert_release_module_origins_v1,
    main as qrr_main,
)
from engine.qrr.composite_runtime_state_v1 import (
    CompositeRuntimeStateError,
    compose_composite_runtime_state_v1,
    validate_composite_state_exact_equality_v1,
)
from engine.qrr.execution_request_v1 import (
    ExecutionRequestDerivationV1,
    build_execution_request_v1,
)
from engine.qrr.historical_escape_suite_v1 import (
    HISTORICAL_ESCAPE_CASES_V1,
    historical_escape_inventory_v1,
)
from engine.qrr.qualified_execution_envelope_v1 import (
    EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION,
    ExperimentRepresentabilityError,
    QualifiedExecutionEnvelopeError,
    QualifiedExecutionEnvelopeV1,
    load_qualified_execution_envelope_v1,
)
from engine.qrr.qualified_runtime_release_v1 import (
    QUALIFIED_RUNTIME_RELEASE_SCHEMA_VERSION,
    QualifiedRuntimeReleaseError,
    QualifiedRuntimeReleaseV1,
    _exact_route_qualified_preflight_v1,
    _launcher_text,
    _write_integrity_manifest,
    discover_python_payload_paths_v1,
    load_qualified_runtime_release_v1,
    runtime_environment_fingerprint_v1,
    stable_python_executable_v1,
    stable_runtime_environment_fingerprint_v1,
)
from engine.qrr.runability_closure_v1 import (
    assess_runability_closure_v1,
    classify_runability_failure_v1,
)


class _Provider:
    provider_id = "provider-v1"
    profile_id = "profile-v1"
    resource_estimator_id = "resource-v1"

    def __init__(self) -> None:
        self.contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=({"component_id": "native_mechanics_binding", "batch_max": 48},),
            required_obligations=("checkpoint_export", "fresh_owner_restore"),
        )

    def execution_route_contract(self, _plan: Any, request: Any = None):
        del request
        return self.contract

    def estimate_resources(self, _plan: Any, _requests: Any):
        return {
            "resource_estimator_id": "resource-v1",
            "vram_bytes_upper_bound": 2_000_000,
            "wall_seconds_upper_bound": 300,
            "disk_bytes_upper_bound": 1_000_000,
        }

    def translate_batch(
        self, plan: Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        batch = next(
            row for row in plan["physical_batches"] if row["batch_id"] == batch_id
        )
        ids = tuple(batch["cell_ids"])
        cells = {row["cell_id"]: row for row in plan["cells"]}
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=ids,
            generations=6,
            ticks_per_generation=160,
            checkpoint_milestones=(0, 3, 6),
            parameter_arrays=(
                (
                    "variable",
                    tuple(cells[cell]["parameters"]["variable"] for cell in ids),
                ),
                ("fixed", tuple(cells[cell]["parameters"]["fixed"] for cell in ids)),
            ),
            seed_arrays=(
                ("ecology", tuple(cells[cell]["seeds"]["ecology"] for cell in ids)),
            ),
            agent_id_allocation_generation_count=128,
            checkpoint_storage_mode="disk",
            reporting_enabled=False,
            execution_route_sha256=self.contract.execution_route_sha256,
            genome_v2_checkpoint_authority_binding=(("authority", "a" * 64),),
        )


class _TickProvider(_Provider):
    def translate_batch(
        self, plan: Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        base = super().translate_batch(plan, batch_id)
        execution = plan["execution"]
        request = TickNativeExecutionRequestV1.from_execution(execution)
        return replace(
            base,
            generations=None,
            ticks_per_generation=None,
            checkpoint_milestones=(),
            total_ticks=request.total_ticks,
            checkpoint_ticks=request.checkpoint_ticks,
            deliberate_restore_tick=request.deliberate_restore_tick,
            source_window_ticks=request.source_window_ticks,
            resident_owner_span_ticks=request.resident_owner_span_ticks,
            native_dispatch_window_ticks=request.native_dispatch_window_ticks,
            tick_native_control=dict(execution["tick_native_control"]),
        )


def test_tick_native_release_preflight_binds_materialized_reference_plan() -> None:
    tick_plan = deepcopy(_plan())
    tick_plan["plan_sha256"] = "e" * 64
    preflight = {
        "schema_version": "eiel_campaign_preflight_v1",
        "plan_sha256": tick_plan["plan_sha256"],
        "ready_for_execution": True,
        "execution_route": {
            "authority_state": "route_qualified",
            "artifact_conformance_exact": True,
        },
    }

    assert _exact_route_qualified_preflight_v1(preflight, tick_plan) is True
    preflight["plan_sha256"] = "b" * 64
    assert _exact_route_qualified_preflight_v1(preflight, tick_plan) is False


def test_qrr_experiment_binds_acs_route_without_changing_default_spec() -> None:
    spec = {"checkpoints": {"milestones": [3]}}

    assert _apply_acs_checkpoint_selection_v1(spec, experiment={}) == spec
    assert _apply_acs_checkpoint_selection_v1(
        spec,
        experiment={"acs_capture_instrumentation": True},
    )["checkpoints"] == {
        "milestones": [3],
        "acs_capture_instrumentation": True,
    }
    assert _apply_acs_checkpoint_selection_v1(
        spec,
        experiment={"acs_checkpoint_slots": 2},
    )["checkpoints"] == {
        "milestones": [3],
        "acs_capture_instrumentation": True,
        "acs_slots": 2,
    }
    assert spec == {"checkpoints": {"milestones": [3]}}


@pytest.mark.parametrize(
    "experiment",
    (
        {"acs_capture_instrumentation": 1},
        {"acs_checkpoint_slots": True},
        {"acs_checkpoint_slots": 4},
    ),
)
def test_qrr_experiment_rejects_malformed_acs_route(
    experiment: Mapping[str, Any],
) -> None:
    with pytest.raises(QrrLauncherError, match="acs_"):
        _apply_acs_checkpoint_selection_v1(
            {"checkpoints": {"milestones": [3]}},
            experiment=experiment,
        )


def _plan() -> dict[str, Any]:
    return {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "qrr-test",
        "plan_sha256": "b" * 64,
        "resolved_profile": "profile-v1",
        "qualification_key_sha256": "c" * 64,
        "qualification_key": {"admitted_batch_range": [1, 48]},
        "execution": {
            "generations": 6,
            "ticks_per_generation": 160,
            "checkpoints": [0, 3, 6],
            "checkpoint_storage_mode": "disk",
            "budgets": {
                "disk_bytes": 1_000_000,
                "vram_bytes": 2_000_000,
                "wall_seconds": 300,
            },
        },
        "cells": [
            {
                "cell_id": "cell-a",
                "parameters": {"variable": 1.0, "fixed": 7},
                "seeds": {"ecology": 11},
            },
            {
                "cell_id": "cell-b",
                "parameters": {"variable": 2.0, "fixed": 7},
                "seeds": {"ecology": 12},
            },
        ],
        "physical_batches": [
            {"batch_id": "batch-0000", "cell_ids": ["cell-a", "cell-b"]}
        ],
        "artifacts": {"native": {"sha256": "d" * 64, "path": "/qualified/native.so"}},
    }


def test_bound_request_derivation_preserves_every_nonderived_field() -> None:
    provider = _Provider()
    bound = build_execution_request_v1(provider, _plan(), "batch-0000")
    derived = bound.derive_v1(
        ExecutionRequestDerivationV1(
            batch_id="canary",
            cell_ids=("cell-b",),
            generations=3,
            checkpoint_milestones=(3,),
            reporting_enabled=True,
        )
    )
    changed = {
        "batch_id",
        "cell_ids",
        "generations",
        "checkpoint_milestones",
        "parameter_arrays",
        "seed_arrays",
        "reporting_enabled",
    }
    for field in fields(ResidentBatchRequestV1):
        if field.name not in changed:
            assert getattr(derived.resident_request, field.name) == getattr(
                bound.resident_request, field.name
            )
    assert dict(derived.resident_request.parameter_arrays) == {
        "variable": (2.0,),
        "fixed": (7,),
    }
    assert derived.resident_request.checkpoint_milestones == (0, 3)
    assert derived.bound_request_sha256 != bound.bound_request_sha256
    assert (
        build_execution_request_v1(provider, _plan(), "batch-0000").bound_request_sha256
        == bound.bound_request_sha256
    )


def test_qrr_binds_exact_requested_configuration_and_seed_identity_to_cells() -> None:
    plan = _plan()
    bound = _bind_science_materialization_v1(
        plan,
        experiment={
            "axes": {},
            "configurations": [
                {"id": "baseline", "controls": {"variable": 1.0}},
                {"id": "treatment", "controls": {"variable": 2.0}},
            ],
        },
    )

    closure = bound["science_materialization"]
    assert closure["requested_materialized_exact"] is True
    assert closure["configuration_row_counts"] == {
        "baseline": 1,
        "treatment": 1,
    }
    assert [cell["science_configuration_id"] for cell in bound["cells"]] == [
        "baseline",
        "treatment",
    ]
    assert all(cell["science_seed_identity"] for cell in bound["cells"])
    assert bound["plan_sha256"] == canonical_sha256(
        {key: value for key, value in bound.items() if key != "plan_sha256"}
    )


def test_qrr_science_closure_fails_when_a_requested_axis_disappears() -> None:
    with pytest.raises(QrrLauncherError, match="requested controls disappeared"):
        _bind_science_materialization_v1(
            _plan(),
            experiment={
                "axes": {"variable": [1.0, 2.0], "dropped": [3.0]},
                "configurations": [],
            },
        )


def test_qrr_automatic_production_resource_admission_covers_live_peaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "ok": True,
        "ready_for_execution": True,
        "resources": {
            "allocation_diagnostic_resource_promotion": {
                "measured_host_peak_bytes": 2_000_000,
                "restore_peak_host_bytes_upper_bound": 3_000_000,
            },
            "admission": {"status": "RESOURCE_ADMISSION_PASSED"},
        },
    }
    plan = {
        "cells": [{"cell_id": "cell-a"}, {"cell_id": "cell-b"}],
        "execution": {},
        "resource_estimate": {
            "disk_bytes_upper_bound": 10_000_000,
            "checkpoint_boundaries": 3,
        },
        "observability": {"living_lineage_canopy": True},
    }
    monkeypatch.setattr(
        "engine.qrr.cli._meminfo_bytes_v1", lambda: {"MemAvailable": 2**40}
    )
    monkeypatch.setattr(
        "engine.qrr.cli.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=2**41, used=0, free=2**41),
    )

    admitted = _complete_production_resource_admission_v1(
        receipt, plan=plan, run_root=tmp_path / "run"
    )["resources"]["production_admission"]

    assert admitted["status"] == "RESOURCE_ADMITTED"
    assert admitted["manual_resource_receipt_selection"] == 0
    assert admitted["host_ram"]["swap_relied_upon"] is False
    assert admitted["restore_peak"]["covered"] is True
    assert admitted["storage"]["checkpoint_count"] == 3
    assert admitted["host_ram"]["canopy_reserve_bytes"] > 0
    assert admitted["host_ram"]["canopy_reserve_bytes"] == (
        2 * CANOPY_HOST_RESERVE_PER_CELL_BYTES
        + 3 * CANOPY_HOST_ACTIVE_SCAN_PER_CHECKPOINT_BYTES
    )
    assert admitted["host_ram"]["canopy_active_scan_concurrency"] == 1
    assert (
        admitted["host_ram"]["canopy_active_scan_reserve_per_checkpoint_bytes"]
        == CANOPY_HOST_ACTIVE_SCAN_PER_CHECKPOINT_BYTES
    )
    assert admitted["host_ram"]["checkpoint_staging_reserve_bytes"] == (
        2 * 64 * 1024 * 1024
    )
    assert admitted["host_ram"]["checkpoint_peak_before_headroom_bytes"] == (
        2_000_000 + 2 * 64 * 1024 * 1024
    )
    assert admitted["host_ram"]["aggregate_checkpoint_peak_before_headroom_bytes"] == (
        admitted["host_ram"]["checkpoint_peak_before_headroom_bytes"]
        + admitted["host_ram"]["canopy_reserve_bytes"]
    )
    assert admitted["host_ram"]["safety_headroom_bytes"] > 0
    assert admitted["host_ram"]["required_bytes"] == (
        admitted["host_ram"]["aggregate_checkpoint_peak_before_headroom_bytes"]
        + admitted["host_ram"]["safety_headroom_bytes"]
    )
    assert admitted["restore_peak"]["required_bytes"] > 3_000_000
    assert admitted["storage"]["canopy_evidence_reserve_bytes"] > 0


def test_qrr_automatic_production_resource_admission_fails_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "ok": True,
        "ready_for_execution": True,
        "resources": {
            "allocation_diagnostic_resource_promotion": {
                "measured_host_peak_bytes": 2_000_000,
                "restore_peak_host_bytes_upper_bound": 3_000_000,
            },
            "admission": {"status": "RESOURCE_ADMISSION_PASSED"},
        },
    }
    plan = {
        "cells": [{"cell_id": "cell-a"}],
        "execution": {},
        "resource_estimate": {
            "disk_bytes_upper_bound": 10_000_000,
            "checkpoint_boundaries": 2,
        },
        "observability": {"living_lineage_canopy": False},
    }
    monkeypatch.setattr("engine.qrr.cli._meminfo_bytes_v1", lambda: {"MemAvailable": 1})
    monkeypatch.setattr(
        "engine.qrr.cli.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10, used=9, free=1),
    )

    blocked = _complete_production_resource_admission_v1(
        receipt, plan=plan, run_root=tmp_path / "run"
    )

    codes = {
        row["code"] for row in blocked["resources"]["production_admission"]["failures"]
    }
    assert codes == {
        "HOST_RAM_BLOCKED",
        "RESTORE_PEAK_BLOCKED",
        "STORAGE_BLOCKED",
    }
    assert blocked["ready_for_execution"] is False
    assert blocked["ok"] is False


def test_qrr_host_admission_does_not_reserve_checkpoint_staging_without_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "ok": True,
        "ready_for_execution": True,
        "resources": {
            "allocation_diagnostic_resource_promotion": {
                "measured_host_peak_bytes": 2_000_000,
                "restore_peak_host_bytes_upper_bound": 3_000_000,
            },
            "admission": {"status": "RESOURCE_ADMISSION_PASSED"},
        },
    }
    plan = {
        "cells": [{"cell_id": "cell-a"}],
        "execution": {},
        "resource_estimate": {
            "disk_bytes_upper_bound": 10_000_000,
            "checkpoint_boundaries": 0,
        },
        "observability": {"living_lineage_canopy": False},
    }
    monkeypatch.setattr(
        "engine.qrr.cli._meminfo_bytes_v1", lambda: {"MemAvailable": 2**40}
    )
    monkeypatch.setattr(
        "engine.qrr.cli.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=2**41, used=0, free=2**41),
    )

    admitted = _complete_production_resource_admission_v1(
        receipt, plan=plan, run_root=tmp_path / "run"
    )["resources"]["production_admission"]

    assert admitted["status"] == "RESOURCE_ADMITTED"
    assert admitted["host_ram"]["checkpoint_staging_reserve_bytes"] == 0
    assert admitted["host_ram"]["checkpoint_peak_before_headroom_bytes"] == 2_000_000
    assert admitted["host_ram"]["canopy_active_scan_concurrency"] == 0
    assert admitted["host_ram"]["canopy_reserve_bytes"] == 0


def test_qrr_canopy_host_reserve_scales_with_checkpoint_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "ok": True,
        "ready_for_execution": True,
        "resources": {
            "allocation_diagnostic_resource_promotion": {
                "measured_host_peak_bytes": 2_000_000,
                "restore_peak_host_bytes_upper_bound": 3_000_000,
            },
            "admission": {"status": "RESOURCE_ADMISSION_PASSED"},
        },
    }
    plan = {
        "cells": [{"cell_id": "cell-a"}, {"cell_id": "cell-b"}],
        "execution": {},
        "resource_estimate": {
            "disk_bytes_upper_bound": 10_000_000,
            "checkpoint_boundaries": 5,
        },
        "observability": {"living_lineage_canopy": True},
    }
    monkeypatch.setattr(
        "engine.qrr.cli._meminfo_bytes_v1", lambda: {"MemAvailable": 2**40}
    )
    monkeypatch.setattr(
        "engine.qrr.cli.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=2**41, used=0, free=2**41),
    )

    admitted = _complete_production_resource_admission_v1(
        receipt, plan=plan, run_root=tmp_path / "run"
    )["resources"]["production_admission"]

    assert admitted["host_ram"]["canopy_active_scan_concurrency"] == 1
    assert admitted["host_ram"]["canopy_reserve_bytes"] == (
        2 * CANOPY_HOST_RESERVE_PER_CELL_BYTES
        + 5 * CANOPY_HOST_ACTIVE_SCAN_PER_CHECKPOINT_BYTES
    )
    assert admitted["restore_peak"]["aggregate_peak_before_headroom_bytes"] == (
        3_000_000 + admitted["host_ram"]["canopy_reserve_bytes"]
    )


def test_qrr_canopy_host_reserve_keeps_waiting_watchers_without_active_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "ok": True,
        "ready_for_execution": True,
        "resources": {
            "allocation_diagnostic_resource_promotion": {
                "measured_host_peak_bytes": 2_000_000,
                "restore_peak_host_bytes_upper_bound": 3_000_000,
            },
            "admission": {"status": "RESOURCE_ADMISSION_PASSED"},
        },
    }
    plan = {
        "cells": [{"cell_id": "cell-a"}],
        "execution": {},
        "resource_estimate": {
            "disk_bytes_upper_bound": 10_000_000,
            "checkpoint_boundaries": 0,
        },
        "observability": {"living_lineage_canopy": True},
    }
    monkeypatch.setattr(
        "engine.qrr.cli._meminfo_bytes_v1", lambda: {"MemAvailable": 2**40}
    )
    monkeypatch.setattr(
        "engine.qrr.cli.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=2**41, used=0, free=2**41),
    )

    admitted = _complete_production_resource_admission_v1(
        receipt, plan=plan, run_root=tmp_path / "run"
    )["resources"]["production_admission"]

    assert admitted["host_ram"]["checkpoint_staging_reserve_bytes"] == 0
    assert admitted["host_ram"]["canopy_active_scan_concurrency"] == 1
    assert admitted["host_ram"]["canopy_active_scan_reserve_bytes"] == 0
    assert (
        admitted["host_ram"]["canopy_reserve_bytes"]
        == CANOPY_HOST_RESERVE_PER_CELL_BYTES
    )


def test_qrr_host_admission_blocks_when_checkpoint_staging_exceeds_available_ram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner_peak = 13_000_000_000
    available = 14_000_000_000
    receipt = {
        "ok": True,
        "ready_for_execution": True,
        "resources": {
            "allocation_diagnostic_resource_promotion": {
                "measured_host_peak_bytes": owner_peak,
                "restore_peak_host_bytes_upper_bound": 1_000_000,
            },
            "admission": {"status": "RESOURCE_ADMISSION_PASSED"},
        },
    }
    plan = {
        "cells": [{"cell_id": f"cell-{ordinal}"} for ordinal in range(48)],
        "execution": {},
        "resource_estimate": {
            "disk_bytes_upper_bound": 10_000_000,
            "checkpoint_boundaries": 3,
        },
        "observability": {"living_lineage_canopy": False},
    }
    monkeypatch.setattr(
        "engine.qrr.cli._meminfo_bytes_v1", lambda: {"MemAvailable": available}
    )
    monkeypatch.setattr(
        "engine.qrr.cli.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=2**41, used=0, free=2**41),
    )

    blocked = _complete_production_resource_admission_v1(
        receipt, plan=plan, run_root=tmp_path / "run"
    )
    production = blocked["resources"]["production_admission"]

    assert production["host_ram"]["checkpoint_staging_reserve_bytes"] == (
        48 * 64 * 1024 * 1024
    )
    assert production["host_ram"]["required_bytes"] > available
    assert {row["code"] for row in production["failures"]} == {"HOST_RAM_BLOCKED"}
    assert production["restore_peak"]["covered"] is True
    assert blocked["ready_for_execution"] is False


def _composite(*, markers: Mapping[str, Any] | None = None):
    eed = torch.tensor([3.0])
    base = {
        "metadata": {
            "cell_id": "cell-a",
            "capability_mask": {"wdl": True, "jungle": True, "eed": True},
        },
        "tensors": {"recovery": torch.tensor([1.0]), "eed": eed.clone()},
        "genome_upgrade_v2_checkpoint_projection": {
            "inheritance_sidecar_identity": {
                "schema_version": "eiel_inheritance_sidecar_identity_v1",
                "cell_id": "cell-a",
            }
        },
    }
    return compose_composite_runtime_state_v1(
        base,
        base_component_id="patch_wdl",
        component_tensors={
            "jungle": {"visibility": torch.tensor([2.0])},
            "eed": {"eed": eed},
        },
        shared_tensor_owners={"recovery": ("patch_wdl", "jungle")},
        expected_existing_tensors={"recovery": torch.tensor([1.0])},
        execution_markers=markers,
        expected_cell_id="cell-a",
    )


def test_composite_state_owns_final_digest_and_exact_restore() -> None:
    state = _composite(
        markers={"wdl_jungle_native_owner": True, "eed_native_owner": True}
    )
    assert state.tensor_owners["eed"] == ("eed",)
    assert state.tensor_owners["recovery"] == ("patch_wdl", "jungle")
    validate_composite_state_exact_equality_v1(state, state)


def test_composite_state_rejects_capability_marker_or_sidecar_conflict() -> None:
    with pytest.raises(CompositeRuntimeStateError, match="lacks execution marker"):
        _composite(markers={"wdl_jungle_native_owner": True})

    state = _composite()
    altered = deepcopy(state.wire_projection())
    altered["genome_upgrade_v2_checkpoint_projection"]["inheritance_sidecar_identity"][
        "cell_id"
    ] = "cell-b"
    with pytest.raises(CompositeRuntimeStateError, match="sidecar identity"):
        compose_composite_runtime_state_v1(
            altered,
            base_component_id="patch_wdl",
            component_tensors={"jungle": {"visibility": torch.tensor([2.0])}},
            expected_cell_id="cell-a",
        )


def _envelope() -> QualifiedExecutionEnvelopeV1:
    provider = _Provider()
    artifact_sha256s = {"native": "d" * 64}
    return QualifiedExecutionEnvelopeV1(
        envelope_id="qrr-test-envelope",
        provider_id=provider.provider_id,
        profile_id=provider.profile_id,
        execution_route_sha256=provider.contract.execution_route_sha256,
        qualification_key_sha256="c" * 64,
        artifact_sha256s=artifact_sha256s,
        experiment_variable_contracts=(
            {
                "name": "variable",
                "value_contract": {
                    "kind": "float",
                    "finite_only": True,
                    "minimum": 0.0,
                    "maximum": 2.0,
                },
            },
        ),
        fixed_parameter_values={"fixed": (7,)},
        admitted_batch_range=(1, 48),
        admitted_generation_range=(3, 128),
        admitted_ticks_per_generation=(160,),
        checkpoint_storage_modes=("disk",),
        resource_contract={
            "declared_budgets": {
                "disk_bytes": 1_000_000,
                "vram_bytes": 2_000_000,
                "wall_seconds": 300,
            },
            "provider_resource_estimator_id": "resource-v1",
            "reference_provider_estimate": {
                "resource_estimator_id": "resource-v1",
                "vram_bytes_upper_bound": 2_000_000,
                "wall_seconds_upper_bound": 300,
                "disk_bytes_upper_bound": 1_000_000,
            },
            "admission_policy": ("release_local_preflight_current_resource_state_v1"),
        },
        representation_contract={
            "schema_version": EXPERIMENT_REPRESENTATION_CONTRACT_SCHEMA_VERSION,
            "policy": "shared_scalar_configs_with_named_per_cell_exceptions_v1",
            "execution_route_sha256": provider.contract.execution_route_sha256,
            "artifact_sha256s": artifact_sha256s,
            "per_cell_mixed_parameter_names": ["variable"],
            "batch_uniform_parameter_names": [],
            "native_mechanics_binding_id": "test-shared-scalar-abi-v1",
            "constraint_sources": ["test-provider-v1"],
        },
    )


def test_qualified_envelope_roundtrips_and_validates_one_bound_request() -> None:
    envelope = _envelope()
    loaded = load_qualified_execution_envelope_v1(envelope.payload())
    validation = loaded.validate_plan(_plan(), provider=_Provider())
    assert validation["valid"] is True
    assert set(validation["bound_request_sha256s"]) == {"batch-0000"}

    invalid = _plan()
    invalid["cells"][1]["parameters"]["variable"] = 2.1
    with pytest.raises(QualifiedExecutionEnvelopeError, match="outside"):
        loaded.validate_plan(invalid, provider=_Provider())


def test_qualified_envelope_accepts_current_qualified_per_cell_policy() -> None:
    envelope = _envelope()
    current = replace(
        envelope,
        representation_contract={
            **dict(envelope.representation_contract),
            "policy": "shared_scalar_configs_with_qualified_per_cell_exceptions_v1",
        },
    )

    assert load_qualified_execution_envelope_v1(current.payload()).envelope_id == (
        current.envelope_id
    )


def test_historical_b48_w32_50000_tick_production_request_is_exactly_representable() -> (
    None
):
    plan = _plan()
    cells = []
    for index in range(48):
        cells.append(
            {
                "cell_id": f"cell-{index:02d}",
                "parameters": {"variable": float(index % 3), "fixed": 7},
                "seeds": {"ecology": 10_000 + index},
            }
        )
    plan["cells"] = cells
    plan["physical_batches"] = [
        {"batch_id": "batch-0000", "cell_ids": [row["cell_id"] for row in cells]}
    ]
    checkpoint_ticks = tuple(range(5_000, 50_001, 5_000))
    plan["execution"] = {
        "execution_route_identity": "tick_native_execution_route_v1",
        "total_ticks": 50_000,
        "checkpoint_ticks": list(checkpoint_ticks),
        "deliberate_restore_tick": 25_000,
        "tick_native_control": {
            "source_window_ticks": 160,
            "resident_owner_span_ticks": 1120,
            "native_dispatch_window_ticks": 1120,
            "event_window_capacity": 65536,
        },
        "checkpoint_storage_mode": "disk",
        "budgets": {
            "disk_bytes": 1_000_000,
            "vram_bytes": 2_000_000,
            "wall_seconds": 300,
        },
    }
    envelope = replace(
        _envelope(),
        time_contract={
            "authority": "absolute_tick_v1",
            "cursor_semantics": "committed_half_open_[0,next_tick)",
            "arbitrary_positive_terminal_tick": True,
            "arbitrary_checkpoint_ticks": True,
            "fresh_owner_partial_tick_restore": True,
            "tick_native_control": dict(plan["execution"]["tick_native_control"]),
        },
    )

    validation = envelope.validate_plan(plan, provider=_TickProvider())
    request = _TickProvider().translate_batch(plan, "batch-0000")
    segments = ProductionEngineLifecycleV1._absolute_tick_segments_v1(
        request, start_next_tick=0
    )

    assert validation["valid"] is True
    assert request.total_ticks == 50_000
    assert request.deliberate_restore_tick == 25_000
    assert (
        tuple(row["end_next_tick"] for row in segments if row["publish_checkpoint"])
        == checkpoint_ticks
    )
    assert segments[-1]["end_next_tick"] == 50_000
    assert segments[-1]["terminal_tick"] == 80
    assert any(row["absolute_tick_span"] is not None for row in segments)
    assert sum(row["absolute_tick_span"] for row in segments) == 50_000
    # The same request remains invalid if a historical alias regains authority.
    invalid = deepcopy(plan)
    invalid["execution"]["internal_lifecycle_schedule"] = {"interval_ticks": 160}
    with pytest.raises(QualifiedExecutionEnvelopeError, match="forbidden compatibility"):
        envelope.validate_plan(invalid, provider=_TickProvider())


@pytest.mark.parametrize("terminal_tick", (161, 320))
def test_absolute_tick_horizon_has_no_divisibility_requirement(
    terminal_tick: int,
) -> None:
    request = ResidentBatchRequestV1(
        profile_id="profile-v1",
        batch_id="batch-0000",
        cell_ids=("cell-a",),
        generations=None,
        ticks_per_generation=None,
        checkpoint_milestones=(),
        parameter_arrays=(),
        seed_arrays=(),
        total_ticks=terminal_tick,
        checkpoint_ticks=(terminal_tick,),
        source_window_ticks=160,
        resident_owner_span_ticks=160,
        native_dispatch_window_ticks=160,
    )
    segments = ProductionEngineLifecycleV1._absolute_tick_segments_v1(
        request, start_next_tick=0
    )
    assert segments[-1]["end_next_tick"] == terminal_tick
    assert segments[-1]["terminal_tick"] == terminal_tick % 160
    if terminal_tick == 161:
        assert [row["absolute_tick_span"] for row in segments] == [160, 1]
    else:
        assert [row["absolute_tick_span"] for row in segments] == [160, 160]
    assert all(row["logical_generation_count"] == 1 for row in segments)


@pytest.mark.parametrize(
    "checkpoint_ticks",
    ((160, 159), (0,), (161, 321)),
)
def test_malformed_absolute_checkpoint_schedules_fail_closed(
    checkpoint_ticks: tuple[int, ...],
) -> None:
    with pytest.raises(TickNativeError):
        TickNativeExecutionRequestV1(
            total_ticks=320,
            checkpoint_ticks=checkpoint_ticks,
        )


def test_representability_rejects_mixed_shared_jungle_eed_controls_before_request_or_resource() -> (
    None
):
    base = _envelope()
    mixed_supported = {
        "env_cfg.move_cost",
        "env_cfg.patch_movement_cost_energy_scale",
    }
    mixed_unsupported = {
        "env_cfg.jungle_predator_base_risk",
        "env_cfg.jungle_predator_health_damage",
        "env_cfg.jungle_predator_risk_cap",
        "env_cfg.jungle_recovery_activity_base_growth",
        "env_cfg.jungle_recovery_activity_decay_per_step",
        "env_cfg.jungle_recovery_activity_fertility_weight",
        "env_cfg.jungle_recovery_activity_waste_penalty_weight",
        "env_cfg.patch_food_gather_amount",
        "env_cfg.patch_food_regrow_per_step",
        "env_cfg.reproduce_cooldown_steps",
        "env_cfg.reproduce_min_energy_buffer",
        "env_cfg.reproduce_parent_energy_share",
        "env_cfg.reproduce_try_prob",
    }
    incident_parameters = sorted(mixed_supported | mixed_unsupported)
    envelope = replace(
        base,
        experiment_variable_contracts=tuple(
            {
                "name": name,
                "value_contract": {
                    "kind": (
                        "int" if name == "env_cfg.reproduce_cooldown_steps" else "float"
                    ),
                    "finite_only": (name != "env_cfg.reproduce_cooldown_steps"),
                    "minimum": 0,
                    "maximum": 100,
                },
            }
            for name in incident_parameters
        ),
        fixed_parameter_values={},
        representation_contract={
            **dict(base.representation_contract),
            "per_cell_mixed_parameter_names": sorted(mixed_supported),
            "batch_uniform_parameter_names": sorted(mixed_unsupported),
        },
    )
    incident_plan = _plan()
    for index, cell in enumerate(incident_plan["cells"], start=1):
        cell["parameters"] = {
            name: (
                index if name == "env_cfg.reproduce_cooldown_steps" else float(index)
            )
            for name in incident_parameters
        }

    class _MustNotExecuteProvider(_Provider):
        def translate_batch(self, plan: Mapping[str, Any], batch_id: str):
            raise AssertionError("request construction must follow representability")

        def estimate_resources(self, plan: Any, requests: Any):
            raise AssertionError("resource estimation must follow representability")

    with pytest.raises(
        ExperimentRepresentabilityError,
        match="QRR_REPRESENTABILITY_BLOCKED.*jungle_predator_base_risk",
    ) as caught:
        envelope.validate_plan(incident_plan, provider=_MustNotExecuteProvider())

    receipt = caught.value.receipt
    assert receipt["status"] == "FAIL"
    assert receipt["failure_class"] == "QRR_REPRESENTABILITY_BLOCKED"
    assert {row["parameter"] for row in receipt["violations"]} == mixed_unsupported
    assert {row["representation_family"] for row in receipt["violations"]} == {
        "jungle",
        "patch",
        "reproduction",
    }


def _release(root: Path) -> QualifiedRuntimeReleaseV1:
    envelope = _envelope().payload()
    environment_contract = stable_runtime_environment_fingerprint_v1()
    launcher_text = _launcher_text(environment_contract["python_executable"])
    launcher_sha256 = hashlib.sha256(launcher_text.encode("utf-8")).hexdigest()
    historical_inventory = historical_escape_inventory_v1()
    python_payload: list[dict[str, Any]] = []
    identity = {
        "schema_version": QUALIFIED_RUNTIME_RELEASE_SCHEMA_VERSION,
        "source_identity": {
            "git_commit": "e" * 40,
            "working_tree": "clean",
            "python_payload_sha256": canonical_sha256({"files": python_payload}),
        },
        "qualified_execution_envelope": envelope,
        "environment_contract": environment_contract,
        "qualified_artifacts": [],
        "python_payload": python_payload,
        "qualification_authority": {"decision": "qualified"},
        "template_plan_sha256": "b" * 64,
        "historical_escape_suite_id": "HistoricalQualificationEscapeSuiteV1",
        "historical_escape_suite_inventory_sha256": historical_inventory[
            "inventory_sha256"
        ],
        "launcher_sha256": launcher_sha256,
    }
    release = QualifiedRuntimeReleaseV1(
        release_id=f"qrr1-{canonical_sha256(identity)[:32]}",
        source_identity=identity["source_identity"],
        qualified_execution_envelope=envelope,
        environment_contract=identity["environment_contract"],
        qualified_artifacts=(),
        python_payload=(),
        qualification_authority=identity["qualification_authority"],
        template_plan_sha256="b" * 64,
        historical_escape_suite_id="HistoricalQualificationEscapeSuiteV1",
        historical_escape_suite_inventory_sha256=historical_inventory[
            "inventory_sha256"
        ],
        launcher_sha256=launcher_sha256,
        created_at_utc="2026-08-11T00:00:00Z",
    )
    root.mkdir(parents=True)
    (root / "payload").mkdir()
    (root / "payload/sentinel.txt").write_text("immutable\n", encoding="utf-8")
    (root / "bin").mkdir()
    (root / "bin/qrr").write_text(launcher_text, encoding="utf-8")
    (root / "manifest.json").write_text(
        pretty_json(release.payload()), encoding="utf-8"
    )
    _write_integrity_manifest(root)
    return release


def test_qrr_launcher_survives_builder_context_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable_python = Path(runtime_environment_fingerprint_v1()["python_executable"])
    builder = tmp_path / "ephemeral-builder/.venv/bin"
    builder.mkdir(parents=True)
    builder_python = builder / "python"
    builder_python.symlink_to(durable_python)
    monkeypatch.setattr(
        "engine.qrr.qualified_runtime_release_v1.sys.executable", str(builder_python)
    )

    stable_python = stable_python_executable_v1(durable_python)
    release = tmp_path / "release"
    (release / "bin").mkdir(parents=True)
    cli = release / "payload/python/engine/qrr/cli.py"
    cli.parent.mkdir(parents=True)
    (cli.parents[1] / "__init__.py").write_text("", encoding="utf-8")
    (cli.parent / "__init__.py").write_text("", encoding="utf-8")
    cli.write_text(
        "def main(*, release_root):\n    print(release_root.name)\n    return 0\n",
        encoding="utf-8",
    )
    launcher = release / "bin/qrr"
    launcher.write_text(_launcher_text(stable_python), encoding="utf-8")
    launcher.chmod(0o755)

    builder_python.unlink()
    builder.rmdir()
    builder.parent.rmdir()
    builder.parent.parent.rmdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "release"
    assert launcher.read_text(encoding="utf-8").splitlines()[0] == f"#!{stable_python}"


def test_qrr_runtime_python_rejects_worktree_local_environment() -> None:
    with pytest.raises(QualifiedRuntimeReleaseError, match="ephemeral worktree"):
        stable_python_executable_v1(
            Path("/tmp/emergence-test/user/emergence-worktrees/example/.venv/bin/python")
        )


def test_release_integrity_fails_closed_after_payload_tamper(tmp_path: Path) -> None:
    root = tmp_path / "release"
    release = _release(root)
    assert load_qualified_runtime_release_v1(root).release_id == release.release_id
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "tools/actors/schemas/eiel_qualified_runtime_release_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    validate_json_schema(release.payload(), schema)
    (root / "payload/sentinel.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(QualifiedRuntimeReleaseError, match="integrity differs"):
        load_qualified_runtime_release_v1(root)


def test_release_module_origin_guard_rejects_current_checkout(tmp_path: Path) -> None:
    (tmp_path / "payload/python").mkdir(parents=True)
    with pytest.raises(QrrLauncherError, match="outside the release payload"):
        assert_release_module_origins_v1(tmp_path)


def test_release_payload_includes_route_adjacent_runtime_manifests() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = set(discover_python_payload_paths_v1(root))

    assert (
        root / "engine/perf/flat_patch_device_neutral_state_manifest_v4.json"
    ).resolve() in paths


def test_runability_probe_uses_existing_fresh_owner_restore_gate() -> None:
    configured = _configure_checkpoint_resume_probe_v1(
        {
            "matrix": {
                "dimensions": [
                    {"id": "ecology_seed", "values": [101]},
                    {"id": "treatment", "values": [0.1, 1.0]},
                ]
            },
            "execution": {
                "execution_route_identity": "tick_native_execution_route_v1",
                "total_ticks": 960,
                "checkpoint_ticks": [480, 960],
                "deliberate_restore_ticks": [480],
                "technical_gate": {"selectors": []},
            },
            "checkpoints": {"milestones": [], "storage_mode": "disk"},
        }
    )

    assert configured["execution"]["technical_gate"] == {
        "total_ticks": 960,
        "checkpoint_ticks": [480, 960],
        "partition": False,
        "profile": "genome_v2_wdl_fixed_mutation_persistence_v1",
        "selectors": [{"ecology_seed": 101, "treatment": 0.1}],
    }
    assert configured["checkpoints"]["milestones"] == []


def test_absolute_tick_restore_gate_subset_does_not_self_interrupt() -> None:
    request = ResidentBatchRequestV1(
        profile_id="profile-v1",
        batch_id="batch-0000",
        cell_ids=("cell-0",),
        checkpoint_milestones=(),
        parameter_arrays=(),
        seed_arrays=(),
        generations=7,
        ticks_per_generation=160,
        total_ticks=1121,
        checkpoint_ticks=(500, 1121),
        deliberate_restore_tick=500,
        deliberate_restore_ticks=(500,),
        tick_native_control={
            "schema_version": "eiel_tick_native_control_v1",
            "time_authority": "absolute_tick_v1",
            "total_ticks": 1121,
            "start_next_tick": 0,
            "resident_owner_span_ticks": 1120,
            "native_dispatch_window_ticks": 1120,
            "refill_chunk_ticks": 160,
            "report_every_ticks": None,
            "science_sample_interval_ticks": None,
            "reducer_every_ticks": None,
            "archive_every_ticks": None,
            "checkpoint_ticks": [500, 1121],
            "checkpoint_every_ticks": None,
            "event_window_capacity": 65536,
            "source_window_ticks": 160,
            "legacy_turnover_interval_ticks": None,
        },
    )

    gate = _subset_request(
        request,
        batch_id="technical-gate-canary",
        cell_ids=("cell-0",),
        generations=None,
        checkpoint_milestones=(),
        total_ticks=1121,
        checkpoint_ticks=(500, 1121),
    )

    assert gate.deliberate_restore_tick is None
    assert gate.deliberate_restore_ticks == ()
    assert gate.checkpoint_ticks == (500, 1121)


def test_qrr_rebinds_inherited_gate_after_provenance_only_seed_change() -> None:
    spec = {
        "matrix": {
            "dimensions": [
                {"id": "ecology_seed", "values": [99173]},
                {"id": "task_seed", "values": [1001]},
                {"id": "wrc_axis_00", "values": [0.1]},
            ]
        },
        "execution": {
            "technical_gate": {
                "generations": 6,
                "selectors": [
                    {
                        "ecology_seed": 101,
                        "task_seed": 1001,
                        "wrc_axis_00": 0.1,
                    }
                ],
            }
        },
    }

    configured = _rebind_technical_gate_selector_v1(spec)

    assert configured["execution"]["technical_gate"]["selectors"] == [
        {
            "ecology_seed": 99173,
            "task_seed": 1001,
            "wrc_axis_00": 0.1,
        }
    ]


def test_runability_gate_accepts_exact_runtime_proofs_without_science_claim(
    tmp_path: Path,
) -> None:
    plan = _plan()
    plan["execution"]["technical_gate"] = {
        "required": True,
        "execution_mode": "pre_primary_canary",
        "cell_ids": ["cell-a"],
        "profile": "genome_v2_wdl_fixed_mutation_persistence_v1",
        "canary": {
            "batch_id": "technical-gate-canary",
            "restore_batch_id": "technical-gate-fresh-owner-restore",
            "cell_ids": ["cell-a"],
            "generations": 6,
            "checkpoint_milestones": [0, 3, 6],
        },
    }
    runtime_checks = {
        "selected_cells_exact": True,
        "selected_cells_present_once": True,
        "manifest_roles_exact": True,
        "plan_and_gate_binding_exact": True,
        "checkpoint_ticks_exact": True,
        "canary_execute_completed": True,
        "fresh_owner_restore_completed": True,
        "fresh_owner_terminal_checkpoint_bound": True,
        "cuda_without_fallback": True,
        "fresh_owner_bundle_reload_and_equality_proven": True,
    }

    class _RunabilityProvider(_Provider):
        def evaluate_technical_gate(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {
                "passed": False,
                "checks": runtime_checks,
                "reasons": [
                    "Genome V2 WDL technical gate profile conflicts with compiled mechanics",
                    "fresh-owner telemetry differs from the canary evidence",
                ],
            }

    gate_root = tmp_path / "technical_gate"
    gate_root.mkdir()
    semantic = {
        "cell_id": "cell-a",
        "generation_metrics": [{"generation": 0, "alive": 1}],
    }
    for name, readback_sha256 in (
        ("canary_manifest.json", "1" * 64),
        ("fresh_owner_restore_manifest.json", "2" * 64),
    ):
        (gate_root / name).write_text(
            json.dumps(
                {
                    "metrics_rows": [
                        {
                            **semantic,
                            "production_control_readback_sha256": readback_sha256,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    receipt = _write_runability_technical_gate_receipt_v1(
        plan=plan,
        run_root=tmp_path,
        provider=_RunabilityProvider(),
    )

    assert receipt["passed"] is True
    assert receipt["evidence"]["provider_science_gate_passed"] is False
    assert receipt["evidence"]["semantic_metric_rows_equal"] is True
    assert receipt["evidence"]["claim_scope"] == (
        "runability_only_no_science_or_qualification_claim_v1"
    )


def test_historical_escape_inventory_points_to_existing_regressions() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = historical_escape_inventory_v1()
    assert inventory["suite_id"] == "HistoricalQualificationEscapeSuiteV1"
    assert len(HISTORICAL_ESCAPE_CASES_V1) >= 15
    for case in HISTORICAL_ESCAPE_CASES_V1:
        relative, separator, test_name = case.regression_node_id.partition("::")
        assert separator and test_name.startswith("test_")
        source = root / relative
        assert source.is_file()
        assert f"def {test_name}(" in source.read_text(encoding="utf-8")


def test_runability_closure_requires_release_identity_resume_and_suite(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    release = _release(release_root)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "qrr_run_binding.json").write_text(
        json.dumps(
            {
                "release_id": release.release_id,
                "launcher": "qrr_v1",
                "route_reconstruction_steps": 0,
                "source_edits_after_qrr_creation": 0,
                "native_rebuilds": 0,
                "runtime_jit_builds": 0,
                "qualification_actions": 0,
                "representability_receipt_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    (run_root / "qrr_run_receipt.json").write_text(
        json.dumps(
            {
                "release_id": release.release_id,
                "status": "completed",
                "checkpoint_resume_exercised": True,
                "checkpoint_resume_exact": True,
                "representability_receipt_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    representation_core = {
        "schema_version": "eiel_qrr_experiment_representability_receipt_v1",
        "status": "PASS",
        "failure_class": None,
        "envelope_sha256": release.qualified_execution_envelope["envelope_sha256"],
    }
    representation_sha256 = canonical_sha256(representation_core)
    for filename in ("qrr_run_binding.json", "qrr_run_receipt.json"):
        path = run_root / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["representability_receipt_sha256"] = representation_sha256
        path.write_text(json.dumps(payload), encoding="utf-8")
    (run_root / "qrr_representability_receipt.json").write_text(
        json.dumps({**representation_core, "receipt_sha256": representation_sha256}),
        encoding="utf-8",
    )
    receipt = assess_runability_closure_v1(
        release_root=release_root,
        run_root=run_root,
        historical_suite_receipt={
            "suite_id": "HistoricalQualificationEscapeSuiteV1",
            "inventory_sha256": (release.historical_escape_suite_inventory_sha256),
            "passed": True,
        },
    )
    assert receipt.decision == "RUNABILITY_CLOSED"
    assert receipt.failure_class is None
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "tools/actors/schemas/eiel_runability_closure_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    validate_json_schema(receipt.payload(), schema)


def test_runability_failure_taxonomy_distinguishes_resource_and_science() -> None:
    assert (
        classify_runability_failure_v1("insufficient VRAM", stage="launch")
        == "RESOURCE_BLOCKED"
    )
    assert (
        classify_runability_failure_v1("SCIENCE_DESIGN conflict", stage="launch")
        == "SCIENCE_DESIGN_BLOCKED"
    )
    assert (
        classify_runability_failure_v1(
            "QRR_REPRESENTABILITY_BLOCKED: mixed Jungle control", stage="launch"
        )
        == "QRR_REPRESENTABILITY_BLOCKED"
    )


def test_qrr_run_reserves_stdout_for_one_json_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(
        "engine.qrr.cli._release_context",
        lambda _root: (SimpleNamespace(release_id="qrr1-test"), SimpleNamespace()),
    )
    monkeypatch.setattr(
        "engine.qrr.cli.assert_release_module_origins_v1", lambda _root: None
    )

    def _run(**_kwargs):
        print("runtime diagnostic")
        return {
            "ok": True,
            "schema_version": "eiel_qrr_launch_result_v1",
            "release_id": "qrr1-test",
        }

    monkeypatch.setattr("engine.qrr.cli._run", _run)

    exit_code = qrr_main(
        [
            "run",
            "--experiment",
            str(tmp_path / "experiment.json"),
            "--run-root",
            str(tmp_path / "run"),
        ],
        release_root=tmp_path / "release",
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["ok"] is True
    assert captured.out.count("{") == 1
    assert captured.err == "runtime diagnostic\n"
