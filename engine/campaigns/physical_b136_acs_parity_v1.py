"""Cross-route parity evidence for physical-B136 ACS QualPrep executions.

ACS0 and ACS2 intentionally have distinct execution-route identities.  This
module never merges those authorities.  It validates four independently
executed canonical roots (ACS0/ACS2, uninterrupted/restored) and materializes
only their shared logical-state, event-history, checkpoint, and restore-process
parity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from .canonical import canonical_sha256
from .execution_front_door_v1 import (
    ExecutionFrontDoorError,
    load_execution_request_v1,
)
from .orchestration import atomic_write_json
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EventSinkError,
    event_history_digest_from_references_v1,
    validate_event_sink_prefix_reference_v1,
)


PHYSICAL_B136_ACS_PARITY_SCHEMA_V1: Final[str] = (
    "eiel_physical_b136_acs_parity_evidence_v1"
)

# These are the source-bound physical-B136 values.  They are deliberately
# kept here as verifier constants: a result, report, or telemetry receipt must
# prove the values rather than redefine them.
PRODUCTION_PROFILE_ID: Final[str] = "eiel-production-runtime-v1"
REQUESTED_RUN_CONTRACT_PROFILE_ID: Final[str] = "eiel_full_evolution_v1"
EVENT_CAPACITY_ROWS_PER_CELL: Final[int] = 41_804
EVENT_ROW_STORAGE_BYTES: Final[int] = 141
EVENT_CAPACITY_BYTES_B136: Final[int] = 801_633_504
TOTAL_TICKS: Final[int] = 3_360
MINIMUM_FREE_VRAM_BYTES: Final[int] = 536_870_912
UNCHANGED_RESERVE_BYTES: Final[int] = 413_138_944
MAX_CAPTURE_RELEASE_SECONDS: Final[float] = 10.0
MAX_SUCCESSOR_GPU_WORK_SECONDS: Final[float] = 10.0
MIN_ACTIVE_GPU_DUTY_FRACTION: Final[float] = 0.8
MIN_MEAN_SM_FRACTION: Final[float] = 0.5
MAX_DISPATCH_OVERHEAD_FRACTION: Final[float] = 0.05
MAX_UNEXPLAINED_SIMULATION_IDLE_SECONDS: Final[float] = 2.0
MAX_TEMPORARY_CAPTURE_FILES: Final[int] = 2
EXPECTED_OWNER_TRANSITION_TICKS: Final[tuple[int, int]] = (1120, 2240)


class PhysicalB136ACSParityError(RuntimeError):
    """A canonical run or cross-route parity assertion is incomplete."""


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhysicalB136ACSParityError(f"{name} must be an object")
    return dict(value)


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PhysicalB136ACSParityError(
            f"{name} must be a regular non-symlink file: {path}"
        )
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), name=name)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalB136ACSParityError(f"{name} is unreadable: {exc}") from exc


def _file_sha256(path: Path, *, name: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise PhysicalB136ACSParityError(
            f"{name} must be a regular non-symlink file: {path}"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PhysicalB136ACSParityError(f"{name} is unreadable: {exc}") from exc
    return digest.hexdigest()


def _number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalB136ACSParityError(f"{name} must be numeric")
    rendered = float(value)
    if (
        not math.isfinite(rendered)
        or (minimum is not None and rendered < minimum)
        or (maximum is not None and rendered > maximum)
    ):
        raise PhysicalB136ACSParityError(f"{name} is outside its numeric domain")
    return rendered


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhysicalB136ACSParityError(f"{name} must be an integer >= {minimum}")
    return value


def _bound_path(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PhysicalB136ACSParityError(f"{name} path is absent")
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PhysicalB136ACSParityError(
            f"{name} escapes the canonical run root"
        ) from exc
    return path


def _validated_result(root: Path) -> dict[str, Any]:
    result = _read_object(root / "execution_result.json", name="execution result")
    supplied = result.get("result_sha256")
    core = dict(result)
    core.pop("result_sha256", None)
    if supplied != canonical_sha256(core):
        raise PhysicalB136ACSParityError("execution result SHA-256 differs")
    mechanical = result.get("mechanical_verification")
    if (
        result.get("schema_version") != "eiel_execution_result_v2"
        or result.get("status") != "completed"
        or result.get("execution_status") != "completed"
        or result.get("result_class") != "SUCCESS"
        or result.get("ok") is not True
        or not isinstance(mechanical, Mapping)
        or mechanical.get("status") != "PASS"
    ):
        raise PhysicalB136ACSParityError(
            f"canonical execution is not a verified success: {root}"
        )
    return result


def _cell_ids(plan: Mapping[str, Any]) -> list[str]:
    cells = plan.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise PhysicalB136ACSParityError("plan cells are absent")
    values: list[str] = []
    for raw in cells:
        row = _object(raw, name="logical cell")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise PhysicalB136ACSParityError("logical cell identity is invalid")
        values.append(cell_id)
    if len(values) != 136 or len(set(values)) != 136:
        raise PhysicalB136ACSParityError(
            "parity evidence requires exactly 136 unique logical cells"
        )
    return values


def _validated_run(root_value: str | Path) -> dict[str, Any]:
    root = Path(root_value).expanduser().resolve()
    result = _validated_result(root)
    plan = _read_object(root / "execution_plan.json", name="execution plan")
    plan_core = dict(plan)
    supplied_plan_sha = plan_core.pop("plan_sha256", None)
    if supplied_plan_sha != canonical_sha256(plan_core):
        raise PhysicalB136ACSParityError("execution plan SHA-256 differs")
    if result.get("plan_sha256") != supplied_plan_sha:
        raise PhysicalB136ACSParityError(
            "execution result does not bind its persisted plan"
        )
    request = _read_object(root / "execution_request.json", name="execution request")
    request_payload = _object(request.get("request"), name="request payload")
    request_payload_core = dict(request_payload)
    nested_request_sha = request_payload_core.pop("request_sha256", None)
    source_identity = _object(
        request.get("source_identity"), name="request source identity"
    )
    if (
        request.get("schema_version") != "eiel_execution_request_receipt_v2"
        or request_payload.get("schema_version") != "eiel_execution_request_v2"
        or nested_request_sha != canonical_sha256(request_payload_core)
        or request.get("request_sha256") != nested_request_sha
        or request.get("request_sha256") != result.get("request_sha256")
        or result.get("invocation_mode") != "execute"
        or result.get("purpose") != request_payload.get("purpose")
        or result.get("campaign_id") != plan.get("campaign_id")
        or result.get("compiled_plan_sha256")
        != source_identity.get("compiled_plan_sha256")
        or result.get("profile_id") != plan.get("resolved_profile")
        or result.get("run_root") != str(root)
        or request.get("bound_plan_sha256") != supplied_plan_sha
        or source_identity != result.get("source_identity")
        or source_identity.get("bound_plan_sha256") != supplied_plan_sha
        or source_identity.get("campaign_spec_path")
        != request_payload.get("campaign_spec")
        or request_payload.get("output_root") != str(root)
        or request.get("run_root") != str(root)
    ):
        raise PhysicalB136ACSParityError(
            "execution request/result/source identity binding differs"
        )
    preflight_path_value = result.get("preflight_receipt_path")
    preflight_path = _bound_path(
        root,
        preflight_path_value,
        name="preflight receipt",
    )
    preflight = _read_object(preflight_path, name="preflight receipt")
    if (
        preflight.get("schema_version") != "eiel_campaign_preflight_v1"
        or preflight.get("campaign_id") != plan.get("campaign_id")
        or preflight.get("plan_sha256") != supplied_plan_sha
        or preflight.get("ready_for_execution") is not True
        or preflight.get("execution_authority_scope") != "qualprep_development_only"
        or preflight.get("reasons") not in (None, [])
        or result.get("preflight_receipt_sha256") != canonical_sha256(preflight)
    ):
        raise PhysicalB136ACSParityError(
            "preflight receipt is not a ready, plan-bound development authority"
        )
    receipt = _object(result.get("execution_receipt"), name="execution receipt")
    manifest_path = _bound_path(
        root, receipt.get("campaign_manifest_path"), name="campaign manifest"
    )
    manifest = _read_object(manifest_path, name="campaign manifest")
    route = manifest.get("execution_route_sha256")
    if (
        manifest.get("status") != "completed"
        or manifest.get("plan_sha256") != supplied_plan_sha
        or receipt.get("execution_route_sha256") != route
        or not isinstance(route, str)
        or len(route) != 64
    ):
        raise PhysicalB136ACSParityError(
            "campaign manifest is not an exact completed-route binding"
        )
    physical = plan.get("physical_batches")
    summaries = manifest.get("batches")
    if (
        not isinstance(physical, list)
        or len(physical) != 1
        or not isinstance(summaries, list)
        or len(summaries) != 1
    ):
        raise PhysicalB136ACSParityError(
            "parity evidence requires one physical B136 batch"
        )
    cell_ids = _cell_ids(plan)
    physical_row = _object(physical[0], name="physical batch")
    summary = _object(summaries[0], name="campaign batch summary")
    if list(physical_row.get("cell_ids", ())) != cell_ids:
        raise PhysicalB136ACSParityError(
            "physical batch does not preserve ordered logical B136 identity"
        )
    batch_path = _bound_path(root, summary.get("manifest_path"), name="batch manifest")
    batch = _read_object(batch_path, name="batch manifest")
    if (
        summary.get("status") != "completed"
        or batch.get("status") != "completed"
        or batch.get("plan_sha256") != supplied_plan_sha
        or batch.get("execution_route_sha256") != route
        or list(batch.get("cell_ids", ())) != cell_ids
    ):
        raise PhysicalB136ACSParityError(
            "batch manifest is not an exact physical-B136 route binding"
        )
    return {
        "root": root,
        "result": result,
        "request": request,
        "receipt": receipt,
        "plan": plan,
        "manifest": manifest,
        "batch": batch,
        "preflight": preflight,
        "request_path": root / "execution_request.json",
        "batch_path": batch_path,
        "cell_ids": cell_ids,
        "route": route,
    }


def _logical_contract(run: Mapping[str, Any]) -> dict[str, Any]:
    plan = _object(run.get("plan"), name="plan")
    execution = _object(plan.get("execution"), name="execution")
    control = _object(execution.get("tick_native_control"), name="tick-native control")
    budgets = _object(execution.get("budgets"), name="execution budgets")
    run_contract = _object(plan.get("run_contract"), name="resolved run contract")
    requested_run_contract = _object(
        run_contract.get("requested"), name="requested run contract"
    )
    resource_estimate = _object(plan.get("resource_estimate"), name="resource estimate")
    qualification_key = _object(plan.get("qualification_key"), name="qualification key")
    qualification_key_sha = plan.get("qualification_key_sha256")
    artifacts = _object(plan.get("artifacts"), name="artifact bindings")
    lifecycle = _object(plan.get("artifact_lifecycle"), name="artifact lifecycle")
    required_artifacts = {
        "cuda_extension",
        "extension_manifest",
        "qualification_receipt",
    }
    if (
        not qualification_key
        or qualification_key_sha != canonical_sha256(qualification_key)
        or not required_artifacts.issubset(artifacts)
        or any(
            not isinstance(
                _object(artifacts[name], name=f"artifact {name}").get("sha256"), str
            )
            for name in required_artifacts
        )
        or not isinstance(lifecycle.get("contract_sha256"), str)
        or any(
            type(budgets.get(name)) is not int or budgets[name] <= 0
            for name in ("vram_bytes", "host_bytes", "minimum_free_gpu_bytes")
        )
    ):
        raise PhysicalB136ACSParityError(
            "run lacks frozen qualification, artifact, lifecycle, or resource authority"
        )
    if (
        plan.get("resolved_profile") != PRODUCTION_PROFILE_ID
        or run_contract.get("schema_version") != "eiel_resolved_run_contract_v1"
        or requested_run_contract
        != {
            "profile_id": REQUESTED_RUN_CONTRACT_PROFILE_ID,
            "mode": "strict",
        }
        or run_contract.get("admission_status") != "passed"
        or resource_estimate.get("resident_event_capacity")
        != EVENT_CAPACITY_ROWS_PER_CELL
        or resource_estimate.get("compiled_batch_count") != 1
        or resource_estimate.get("compiled_physical_batch_cells_total") != 136
        or resource_estimate.get("compiled_partition") is not False
    ):
        raise PhysicalB136ACSParityError(
            "run does not bind the production resolved profile and strict requested authority"
        )
    cells = plan.get("cells")
    if not isinstance(cells, list):
        raise PhysicalB136ACSParityError("logical cells are absent")
    for raw_cell in cells:
        cell = _object(raw_cell, name="logical cell")
        parameters = _object(cell.get("parameters"), name="logical cell parameters")
        if (
            parameters.get("patch.grid_size") != 32
            or parameters.get("patch.initial_population") != 256
            or parameters.get("patch.capacity") != 512
            or parameters.get("tw_rngv2.semantics_version") != 2
            or parameters.get("cuda_architecture_b.cpu_request_staging_enable")
            is not False
            or parameters.get("production_lifecycle.ordinary_births_enabled")
            is not True
        ):
            raise PhysicalB136ACSParityError(
                "run changes frozen geometry, RNG, birth, or staging semantics"
            )
    if (
        plan.get("resolved_profile") != PRODUCTION_PROFILE_ID
        or execution.get("execution_route_identity") != "tick_native_execution_route_v1"
        or execution.get("native_streamed_checkpoint") is not True
        or execution.get("acs_capture_instrumentation") is not True
        or execution.get("checkpoint_storage_mode") != "disk"
        or execution.get("total_ticks") != 3360
        or execution.get("checkpoint_ticks") != [3359, 3360]
        or any(
            name in execution
            for name in (
                "generations",
                "ticks_per_generation",
                "physical_window_generations",
                "internal_lifecycle_schedule",
                "refill_chunk_ticks",
            )
        )
        or any(
            name in control
            for name in (
                "refill_chunk_ticks",
                "legacy_turnover_interval_ticks",
            )
        )
        or control.get("event_window_capacity") != EVENT_CAPACITY_ROWS_PER_CELL
        or control.get("resident_owner_span_ticks") != 1120
        or control.get("native_dispatch_window_ticks") != 1120
        or control.get("source_window_ticks") != 160
    ):
        raise PhysicalB136ACSParityError(
            "run does not bind the exact physical-B136 tick-native control"
        )
    return {
        "campaign_id": plan.get("campaign_id"),
        "resolved_profile": plan.get("resolved_profile"),
        "qualification_key": qualification_key,
        "qualification_key_sha256": qualification_key_sha,
        "cells": plan.get("cells"),
        "physical_batches": plan.get("physical_batches"),
        "artifacts": artifacts,
        "artifact_lifecycle": lifecycle,
        "execution": {
            "execution_route_identity": execution.get("execution_route_identity"),
            "total_ticks": execution.get("total_ticks"),
            "checkpoint_ticks": execution.get("checkpoint_ticks"),
            "checkpoint_storage_mode": execution.get("checkpoint_storage_mode"),
            "native_streamed_checkpoint": execution.get("native_streamed_checkpoint"),
            "budgets": budgets,
            "tick_native_control": control,
        },
    }


def _request_contract(run: Mapping[str, Any], *, slots: int, restored: bool) -> None:
    request_receipt = _object(run.get("request"), name="execution request")
    request = _object(request_receipt.get("request"), name="request payload")
    baseline = _object(request.get("baseline"), name="Baseline declaration")
    deviations = baseline.get("deviations")
    execution = _object(
        _object(run.get("plan"), name="plan").get("execution"),
        name="execution",
    )
    if (
        request.get("purpose") != "qualprep"
        or request.get("qualification_profile") is not None
        or baseline.get("schema_version") != "eiel_baseline_declaration_v1"
        or baseline.get("profile_id") != "EIEL_PRODUCTION_BASELINE_V1"
        or set(baseline) != {"schema_version", "profile_id", "intent", "deviations"}
        or not isinstance(deviations, list)
        or execution.get("acs_checkpoint_slots") != slots
        or execution.get("deliberate_restore_ticks", []) != ([3359] if restored else [])
        or request.get("restore_process_isolation")
        != ("fresh_invocation_per_restore" if restored else None)
    ):
        raise PhysicalB136ACSParityError(
            "run request, Baseline, ACS slot, or restore schedule differs"
        )
    if slots == 0:
        if (
            baseline.get("intent") != "control"
            or len(deviations) != 1
            or not isinstance(deviations[0], Mapping)
            or set(deviations[0]) != {"requirement_id", "reason"}
            or deviations[0].get("requirement_id") != "checkpoint.acs_two_slot"
            or not isinstance(deviations[0].get("reason"), str)
            or not deviations[0]["reason"]
        ):
            raise PhysicalB136ACSParityError(
                "ACS0 request lacks its exact Baseline control deviation"
            )
        expected_baseline_status = "VALID_DECLARED_DEVIATION"
        expected_result_deviations = [
            {
                "requirement_id": "checkpoint.acs_two_slot",
                "reason": deviations[0]["reason"],
                "status": "DECLARED",
                "expected": {
                    "acs_checkpoint_slots": 2,
                    "persistence_mode": "bounded_async_single_writer_v1",
                },
                "observed": {
                    "acs_checkpoint_slots": 0,
                    "persistence_mode": "synchronous_v1",
                },
            }
        ]
    else:
        if baseline.get("intent") != "development" or deviations != []:
            raise PhysicalB136ACSParityError(
                "ACS2 request is not Baseline-conformant development intent"
            )
        expected_baseline_status = "BASELINE_CONFORMANT"
        expected_result_deviations = []
    result_baseline = _object(
        _object(run.get("result"), name="execution result").get("baseline"),
        name="Baseline result binding",
    )
    if (
        result_baseline.get("profile_id") != "EIEL_PRODUCTION_BASELINE_V1"
        or result_baseline.get("admitted") is not True
        or result_baseline.get("operational_status") != "CONFORMANT"
        or result_baseline.get("overall_status") != expected_baseline_status
        or result_baseline.get("deviations") != expected_result_deviations
    ):
        raise PhysicalB136ACSParityError(
            "run result lacks its exact admitted Baseline binding"
        )


def _path_value(value: Any, path: Sequence[str]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _path_present(value: Any, path: Sequence[str]) -> bool:
    current = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _authoritative_integer(
    values: Sequence[tuple[str, Any]], *, name: str, expected: int | None = None
) -> int:
    if not values or any(type(value) is not int or value < 0 for _, value in values):
        raise PhysicalB136ACSParityError(
            f"{name} is not measured by canonical authority"
        )
    observed = list(values)
    distinct = {value for _, value in observed}
    if len(distinct) != 1:
        raise PhysicalB136ACSParityError(f"{name} authorities disagree")
    result = observed[0][1]
    if expected is not None and result != expected:
        raise PhysicalB136ACSParityError(
            f"{name} differs from required value {expected}"
        )
    return result


def _physical_plan_authority(run: Mapping[str, Any]) -> dict[str, int]:
    """Cross-check plan/preflight capacity and frozen physical thresholds."""

    plan = _object(run.get("plan"), name="plan")
    preflight = _object(run.get("preflight"), name="preflight")
    resources = _object(preflight.get("resources"), name="preflight resources")
    shape = _object(resources.get("physical_b136_shape"), name="preflight B136 shape")
    if (
        shape.get("applicable") is not True
        or shape.get("status") != "PASS"
        or shape.get("logical_cells") != 136
        or shape.get("compiled_batch_count") != 1
        or shape.get("compiled_batch_sizes") != [136]
        or shape.get("technical_gate_partition") is not False
        or shape.get("event_window_capacity") != EVENT_CAPACITY_ROWS_PER_CELL
        or shape.get("expected_event_window_capacity") != EVENT_CAPACITY_ROWS_PER_CELL
    ):
        raise PhysicalB136ACSParityError(
            "preflight does not prove the canonical physical-B136 shape"
        )
    provider_estimate = _object(
        resources.get("provider_estimate"), name="preflight provider estimate"
    )
    estimate = _object(plan.get("resource_estimate"), name="resource estimate")
    row_values = [
        (
            "preflight provider resident_event_rows_per_cell",
            provider_estimate.get("resident_event_rows_per_cell"),
        ),
        ("plan resident_event_capacity", estimate.get("resident_event_capacity")),
        (
            "plan control event_window_capacity",
            _path_value(
                plan, ("execution", "tick_native_control", "event_window_capacity")
            ),
        ),
    ]
    row_count = _authoritative_integer(
        row_values,
        name="event capacity rows per cell",
        expected=EVENT_CAPACITY_ROWS_PER_CELL,
    )
    row_bytes_authorities = [
        (
            "preflight provider resident_event_row_bytes",
            provider_estimate.get("resident_event_row_bytes"),
        )
    ]
    plan_row_bytes = estimate.get("resident_event_row_bytes")
    if "resident_event_row_bytes" in estimate:
        row_bytes_authorities.append(("plan resident_event_row_bytes", plan_row_bytes))
    row_bytes_assumption = _path_value(
        estimate,
        ("resource_estimator_assumptions", "resident_event_row_bytes"),
    )
    if _path_present(
        estimate, ("resource_estimator_assumptions", "resident_event_row_bytes")
    ):
        row_bytes_authorities.append(
            ("resource-estimate resident_event_row_bytes", row_bytes_assumption)
        )
    row_bytes = _authoritative_integer(
        row_bytes_authorities,
        name="event row storage bytes",
        expected=EVENT_ROW_STORAGE_BYTES,
    )
    event_bytes_authorities = [
        (
            "preflight provider resident_event_bytes",
            provider_estimate.get("resident_event_bytes"),
        )
    ]
    plan_event_bytes = estimate.get("resident_event_bytes")
    if "resident_event_bytes" in estimate:
        event_bytes_authorities.append(("plan resident_event_bytes", plan_event_bytes))
    preflight_event_bytes = resources.get("resident_event_bytes_b136")
    if "resident_event_bytes_b136" in resources:
        event_bytes_authorities.append(
            ("preflight B136 resident event bytes", preflight_event_bytes)
        )
    event_bytes = _authoritative_integer(
        event_bytes_authorities,
        name="B136 resident event bytes",
        expected=EVENT_CAPACITY_BYTES_B136,
    )
    if row_count * row_bytes * 136 != event_bytes:
        raise PhysicalB136ACSParityError(
            "resident event capacity arithmetic is not exact for B136"
        )
    reserve_authorities = []
    for label, container in (
        ("preflight provider ACS reserve", provider_estimate),
        ("plan ACS reserve", estimate),
    ):
        assumptions = container.get("resource_estimator_assumptions")
        for field in (
            "acs_native_physical_b136_reserve_bytes",
            "frozen_reserve_bytes",
        ):
            value = _path_value(container, ("resource_estimator_assumptions", field))
            if isinstance(assumptions, Mapping) and field in assumptions:
                reserve_authorities.append((f"{label} {field}", value))
    provider_reserve_authorities = [
        value
        for source, value in reserve_authorities
        if source.startswith("preflight provider")
    ]
    if not provider_reserve_authorities:
        raise PhysicalB136ACSParityError(
            "preflight provider unchanged VRAM reserve is absent"
        )
    reserve = _authoritative_integer(
        reserve_authorities,
        name="unchanged VRAM reserve",
        expected=UNCHANGED_RESERVE_BYTES,
    )
    plan_minimum_free = _authoritative_integer(
        [
            (
                "plan minimum free GPU budget",
                _path_value(plan, ("execution", "budgets", "minimum_free_gpu_bytes")),
            )
        ],
        name="minimum free VRAM threshold",
        expected=MINIMUM_FREE_VRAM_BYTES,
    )
    preflight_cuda = resources.get("cuda")
    current_cuda_fields = {
        "admission_required",
        "free_bytes",
        "live_admission_bound_bytes",
        "required_headroom_bytes",
        "free_covers_admission_bound_and_headroom",
        "free_covers_measurement_bootstrap_floor",
    }
    has_current_cuda_shape = isinstance(preflight_cuda, Mapping) and any(
        field in preflight_cuda for field in current_cuda_fields
    )
    threshold_authorities: list[tuple[str, Any]] = [
        ("plan minimum free GPU budget", plan_minimum_free)
    ]
    if isinstance(preflight_cuda, Mapping) and (
        "minimum_free_gpu_bytes" in preflight_cuda
    ):
        threshold_authorities.append(
            (
                "preflight resources.cuda minimum free GPU budget",
                preflight_cuda.get("minimum_free_gpu_bytes"),
            )
        )
    legacy_cuda = preflight.get("cuda")
    if isinstance(legacy_cuda, Mapping) and (
        "minimum_free_gpu_bytes" in legacy_cuda
    ):
        threshold_authorities.append(
            (
                "legacy preflight minimum free GPU budget",
                legacy_cuda.get("minimum_free_gpu_bytes"),
            )
        )
    if has_current_cuda_shape:
        current_cuda = _object(preflight_cuda, name="preflight CUDA resources")
        if (
            current_cuda.get("admission_required") is not True
            or current_cuda.get("budget_within_90_percent_total") is not True
            or current_cuda.get("free_covers_admission_bound_and_headroom") is not True
            or current_cuda.get("free_covers_measurement_bootstrap_floor") is not True
            or current_cuda.get("provider_estimate_within_budget") is not True
            or current_cuda.get("runtime_environment_ok") is not True
        ):
            raise PhysicalB136ACSParityError(
                "preflight CUDA admission/free/bound evidence is incomplete"
            )
        if current_cuda.get("live_admission_bound_source") != (
            "provider_exact_upper_bound"
        ):
            raise PhysicalB136ACSParityError(
                "preflight CUDA live admission bound authority is not exact"
            )
        cuda_values = {
            field: _integer(
                current_cuda.get(field),
                name=f"preflight CUDA {field}",
                minimum=0,
            )
            for field in (
                "free_bytes",
                "total_bytes",
                "declared_vram_budget_bytes",
                "live_admission_bound_bytes",
                "required_headroom_bytes",
                "measurement_bootstrap_minimum_free_bytes",
            )
        }
        minimum_residual_headroom = _integer(
            current_cuda.get("minimum_residual_headroom_bytes"),
            name="preflight CUDA minimum residual headroom",
            minimum=0,
        )
        if (
            cuda_values["total_bytes"] <= 0
            or cuda_values["free_bytes"] > cuda_values["total_bytes"]
            or cuda_values["declared_vram_budget_bytes"]
            > cuda_values["total_bytes"]
            or cuda_values["live_admission_bound_bytes"]
            > cuda_values["total_bytes"]
            or cuda_values["free_bytes"]
            < cuda_values["live_admission_bound_bytes"]
            + cuda_values["required_headroom_bytes"]
            or cuda_values["free_bytes"]
            < cuda_values["measurement_bootstrap_minimum_free_bytes"]
            or cuda_values["measurement_bootstrap_minimum_free_bytes"]
            < cuda_values["required_headroom_bytes"] + minimum_residual_headroom
        ):
            raise PhysicalB136ACSParityError(
                "preflight CUDA free/admission bound arithmetic is invalid"
            )
    elif len(threshold_authorities) == 1:
        raise PhysicalB136ACSParityError(
            "minimum free VRAM threshold lacks canonical authority"
        )
    minimum_free = _authoritative_integer(
        threshold_authorities,
        name="minimum free VRAM threshold",
        expected=MINIMUM_FREE_VRAM_BYTES,
    )
    return {
        "event_capacity_rows_per_cell": row_count,
        "event_row_storage_bytes": row_bytes,
        "event_capacity_bytes_b136": event_bytes,
        "unchanged_reserve_bytes": reserve,
        "minimum_free_vram_bytes": minimum_free,
    }


def _process_identity(value: Any, *, name: str) -> dict[str, Any]:
    identity = _object(value, name=name)
    if (
        set(identity) != {"schema_version", "pid", "process_instance_id"}
        or identity.get("schema_version") != "eiel_execution_process_identity_v1"
        or type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or not isinstance(identity.get("process_instance_id"), str)
        or not identity["process_instance_id"]
    ):
        raise PhysicalB136ACSParityError(f"{name} is not an exact process identity")
    return identity


def _provider_result(run: Mapping[str, Any]) -> dict[str, Any]:
    batch = _object(run.get("batch"), name="batch")
    return _object(batch.get("result"), name="provider batch result")


def _digest_map(value: Any, *, cell_ids: Sequence[str], name: str) -> dict[str, str]:
    raw = _object(value, name=name)
    if (
        set(raw) != set(cell_ids)
        or len(raw) != len(cell_ids)
        or any(not isinstance(digest, str) or not digest for digest in raw.values())
    ):
        raise PhysicalB136ACSParityError(
            f"{name} does not exactly cover ordered physical B136"
        )
    return {str(cell_id): str(raw[cell_id]) for cell_id in cell_ids}


def _checkpoint_digest(
    checkpoint: Any, *, cell_ids: Sequence[str], next_tick: int, name: str
) -> tuple[str, dict[str, str]]:
    row = _object(checkpoint, name=name)
    if (
        row.get("time_authority") != "absolute_tick_v1"
        or row.get("absolute_tick") != next_tick
        or row.get("next_absolute_tick") != next_tick
    ):
        raise PhysicalB136ACSParityError(f"{name} frontier differs")
    _digest_map(
        row.get("terminal_cell_digests"), cell_ids=cell_ids, name=f"{name} digests"
    )
    values = _digest_map(
        row.get("canonical_terminal_state_digest_sha256s"),
        cell_ids=cell_ids,
        name=f"{name} canonical state digests",
    )
    return canonical_sha256(values), values


def _catalog_checkpoint_digest(
    run: Mapping[str, Any], *, tick: int
) -> tuple[str, dict[str, str]]:
    result = _provider_result(run)
    catalog = _object(result.get("checkpoint_catalog"), name="checkpoint catalog")
    checkpoint = catalog.get(str(tick), catalog.get(tick))
    return _checkpoint_digest(
        checkpoint,
        cell_ids=run["cell_ids"],
        next_tick=tick,
        name=f"checkpoint {tick}",
    )


def _catalog_projection_digest(run: Mapping[str, Any], *, tick: int) -> str:
    result = _provider_result(run)
    catalog = _object(result.get("checkpoint_catalog"), name="checkpoint catalog")
    checkpoint = _object(
        catalog.get(str(tick), catalog.get(tick)), name=f"checkpoint {tick}"
    )
    values = _digest_map(
        checkpoint.get("terminal_cell_digests"),
        cell_ids=run["cell_ids"],
        name=f"checkpoint {tick} projection digests",
    )
    return canonical_sha256(values)


def _terminal_evidence(run: Mapping[str, Any]) -> dict[str, Any]:
    batch = _object(run.get("batch"), name="batch")
    terminal_digest, terminal_map = _checkpoint_digest(
        batch.get("checkpoint"),
        cell_ids=run["cell_ids"],
        next_tick=3360,
        name="terminal checkpoint",
    )
    result = _provider_result(run)
    receipts = result.get("native_execution_receipts")
    if (
        not isinstance(receipts, Sequence)
        or isinstance(receipts, (str, bytes))
        or not receipts
    ):
        raise PhysicalB136ACSParityError("native execution receipts are absent")
    terminal_receipt = _object(receipts[-1], name="terminal native receipt")
    event_digest = terminal_receipt.get("external_event_history_digest")
    measurement = _object(
        terminal_receipt.get("resource_window_measurement"),
        name="terminal resource measurement",
    )
    if (
        not isinstance(event_digest, str)
        or not event_digest
        or measurement.get("end_tick") != 3360
        or type(measurement.get("external_event_history_count")) is not int
        or type(measurement.get("terminal_event_ring_occupancy")) is not int
    ):
        raise PhysicalB136ACSParityError(
            "terminal event-history evidence is incomplete"
        )
    frontier_candidates: list[list[int]] = []
    settlements = result.get("compact_segment_settlements")
    if isinstance(settlements, Sequence) and not isinstance(settlements, (str, bytes)):
        terminal_settlements = [
            _object(value, name="compact settlement")
            for value in settlements
            if isinstance(value, Mapping) and value.get("end_next_tick") == 3360
        ]
        if len(terminal_settlements) > 1:
            raise PhysicalB136ACSParityError(
                "terminal compact settlement frontier is ambiguous"
            )
        if terminal_settlements:
            value = terminal_settlements[0].get("event_frontier_by_cell")
            if (
                not isinstance(value, list)
                or len(value) != 136
                or any(type(item) is not int or item < 0 for item in value)
            ):
                raise PhysicalB136ACSParityError(
                    "terminal compact settlement frontier does not exactly cover B136"
                )
            frontier_candidates.append(value)
    references = terminal_receipt.get("event_history_prefix_references_by_cell")
    if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
        if len(references) != 136:
            raise PhysicalB136ACSParityError(
                "terminal native event frontier does not exactly cover B136"
            )
        native_frontier: list[int] = []
        authenticated_references: list[dict[str, Any]] = []
        for index, raw_reference in enumerate(references):
            reference = _object(
                raw_reference, name=f"terminal event prefix reference {index}"
            )
            try:
                authenticated = validate_event_sink_prefix_reference_v1(reference)
            except EventSinkError as exc:
                raise PhysicalB136ACSParityError(
                    "terminal native event prefix reference is invalid"
                ) from exc
            row_count = authenticated.get("row_count")
            if (
                authenticated != reference
                or authenticated.get("cell_index") != index
                or authenticated.get("row_bytes") != EVENT_ROW_STORAGE_BYTES
                or type(row_count) is not int
                or row_count < 0
            ):
                raise PhysicalB136ACSParityError(
                    "terminal native event prefix reference is invalid"
                )
            authenticated_references.append(authenticated)
            native_frontier.append(row_count)
        try:
            authenticated_history_digest = event_history_digest_from_references_v1(
                authenticated_references
            )
        except EventSinkError as exc:
            raise PhysicalB136ACSParityError(
                "terminal native event-history digest is invalid"
            ) from exc
        if authenticated_history_digest != event_digest:
            raise PhysicalB136ACSParityError(
                "terminal native event-history digest differs from its prefixes"
            )
        frontier_candidates.append(native_frontier)
    if not frontier_candidates or any(
        value != frontier_candidates[0] for value in frontier_candidates[1:]
    ):
        raise PhysicalB136ACSParityError(
            "terminal event frontier does not exactly cover B136"
        )
    frontier = frontier_candidates[0]
    if sum(frontier) != measurement["external_event_history_count"]:
        raise PhysicalB136ACSParityError(
            "terminal event frontier and event-history count differ"
        )
    return {
        "terminal_digest": terminal_digest,
        "terminal_cell_digests": terminal_map,
        "event_history_digest": event_digest,
        "event_history_count": measurement["external_event_history_count"],
        "terminal_event_ring_occupancy": measurement["terminal_event_ring_occupancy"],
        "event_frontier_sha256": canonical_sha256(frontier),
    }


def _native_resource_measurements(
    run: Mapping[str, Any], *, authority: Mapping[str, int]
) -> dict[str, Any]:
    """Validate provider-native resource windows and derive bounded peaks."""

    result = _provider_result(run)
    raw_receipts = result.get("native_execution_receipts")
    if (
        not isinstance(raw_receipts, Sequence)
        or isinstance(raw_receipts, (str, bytes))
        or not raw_receipts
    ):
        raise PhysicalB136ACSParityError("native resource windows are absent")
    windows: list[dict[str, int]] = []
    pinned_values: list[int] = []
    for index, raw_receipt in enumerate(raw_receipts):
        receipt = _object(raw_receipt, name=f"native receipt {index}")
        measurement = _object(
            receipt.get("resource_window_measurement"),
            name=f"native resource window {index}",
        )
        required = {
            field: measurement.get(field)
            for field in (
                "start_tick",
                "end_tick",
                "peak_gpu_allocated_bytes",
                "peak_gpu_reserved_bytes",
                "minimum_free_gpu_bytes",
                "host_peak_rss_bytes",
                "host_terminal_rss_bytes",
                "resident_event_bytes",
                "resident_source_bytes",
                "event_ring_capacity",
                "event_ring_peak_occupancy",
                "terminal_event_ring_occupancy",
            )
        }
        if any(type(value) is not int or value < 0 for value in required.values()):
            raise PhysicalB136ACSParityError(
                f"native resource window {index} lacks numeric resource measurements"
            )
        if required["end_tick"] <= required["start_tick"]:
            raise PhysicalB136ACSParityError(
                f"native resource window {index} has an invalid tick span"
            )
        if (
            required["resident_event_bytes"] != authority["event_capacity_bytes_b136"]
            or required["event_ring_capacity"]
            != authority["event_capacity_rows_per_cell"] * 136
            or required["event_ring_peak_occupancy"] > required["event_ring_capacity"]
            or required["terminal_event_ring_occupancy"]
            > required["event_ring_capacity"]
            or required["peak_gpu_reserved_bytes"]
            < required["peak_gpu_allocated_bytes"]
        ):
            raise PhysicalB136ACSParityError(
                f"native resource window {index} violates resident capacity arithmetic"
            )
        if windows and required["start_tick"] < windows[-1]["end_tick"]:
            raise PhysicalB136ACSParityError(
                f"native resource window {index} is not ordered with its predecessor"
            )
        windows.append(required)
        host_samples = measurement.get("host_samples")
        if (
            not isinstance(host_samples, Sequence)
            or isinstance(host_samples, (str, bytes))
            or not host_samples
        ):
            raise PhysicalB136ACSParityError(
                f"native resource window {index} lacks host samples"
            )
        for sample_index, raw_sample in enumerate(host_samples):
            sample = _object(
                raw_sample,
                name=f"native host sample {index}:{sample_index}",
            )
            for field in ("rss_bytes", "rss_high_water_bytes", "pinned_bytes"):
                value = sample.get(field)
                if type(value) is not int or value < 0:
                    raise PhysicalB136ACSParityError(
                        f"native host sample {index}:{sample_index} lacks {field}"
                    )
            pinned_values.append(sample["pinned_bytes"])
    captures = result.get("acs_capture_measurements")
    if not isinstance(captures, list) or not captures:
        raise PhysicalB136ACSParityError(
            "native/provider telemetry cannot decide pinned-memory peak"
        )
    for capture_index, raw_capture in enumerate(captures):
        capture = _object(raw_capture, name=f"ACS capture measurement {capture_index}")
        for field in (
            "host_resources_before",
            "host_resources_peak",
            "host_resources_after",
        ):
            sample = capture.get(field)
            if isinstance(sample, Mapping):
                pinned = sample.get("pinned_bytes")
                if type(pinned) is int and pinned >= 0:
                    pinned_values.append(pinned)
    if not pinned_values:
        raise PhysicalB136ACSParityError(
            "native/provider telemetry cannot decide pinned-memory peak"
        )
    budgets = _object(
        _object(run.get("plan"), name="plan").get("execution"),
        name="execution",
    ).get("budgets")
    budget = _object(budgets, name="execution budgets")
    gpu_budget = _integer(budget.get("vram_bytes"), name="VRAM budget", minimum=1)
    host_budget = _integer(budget.get("host_bytes"), name="host budget", minimum=1)
    host_slope_limit = _integer(
        budget.get("maximum_host_memory_slope_bytes_per_window"),
        name="host-memory slope limit",
        minimum=1,
    )
    gpu_slope_limit = _integer(
        budget.get("maximum_gpu_memory_slope_bytes_per_window"),
        name="GPU-memory slope limit",
        minimum=1,
    )

    def max_adjacent_slope(values: Sequence[int]) -> float:
        if len(values) < 2:
            return 0.0
        return max(
            abs(later - earlier)
            for earlier, later in zip(values, values[1:], strict=False)
        )

    def slope(field: str) -> float:
        return max_adjacent_slope([row[field] for row in windows])

    peak_gpu_reserved = max(row["peak_gpu_reserved_bytes"] for row in windows)
    peak_gpu_allocated = max(row["peak_gpu_allocated_bytes"] for row in windows)
    minimum_free_gpu = min(row["minimum_free_gpu_bytes"] for row in windows)
    peak_host_rss = max(row["host_peak_rss_bytes"] for row in windows)
    if (
        peak_gpu_reserved > gpu_budget
        or peak_gpu_allocated > gpu_budget
        or minimum_free_gpu < authority["minimum_free_vram_bytes"]
        or peak_host_rss > host_budget
        or slope("peak_gpu_reserved_bytes") > gpu_slope_limit
        or slope("peak_gpu_allocated_bytes") > gpu_slope_limit
        or slope("host_terminal_rss_bytes") > host_slope_limit
        or slope("resident_event_bytes") != 0.0
        or slope("resident_source_bytes") != 0.0
    ):
        raise PhysicalB136ACSParityError(
            "native resource windows exceed the canonical host/VRAM bounds"
        )
    captures = result.get("acs_capture_measurements")
    if not isinstance(captures, list) or not captures:
        raise PhysicalB136ACSParityError(
            "provider result lacks ACS capture measurements for two-slab bounds"
        )
    captured_bytes: list[int] = []
    for index, raw_capture in enumerate(captures):
        capture = _object(raw_capture, name=f"ACS capture measurement {index}")
        captured_bytes.append(
            _integer(
                capture.get("captured_bytes"),
                name=f"ACS capture {index} captured bytes",
            )
        )
    two_slab_bound = max(captured_bytes) * 2
    pinned_peak = max(pinned_values)
    pinned_slope = max_adjacent_slope(pinned_values)
    if pinned_peak > two_slab_bound or pinned_slope > host_slope_limit:
        raise PhysicalB136ACSParityError(
            "pinned-memory peak or slope exceeds the numeric two-slab/host bound"
        )
    return {
        "window_count": len(windows),
        "peak_gpu_reserved_bytes": peak_gpu_reserved,
        "peak_gpu_allocated_bytes": peak_gpu_allocated,
        "minimum_free_gpu_bytes": minimum_free_gpu,
        "peak_host_rss_bytes": peak_host_rss,
        "peak_pinned_bytes": pinned_peak,
        "gpu_reserved_slope_bytes_per_window": slope("peak_gpu_reserved_bytes"),
        "gpu_allocated_slope_bytes_per_window": slope("peak_gpu_allocated_bytes"),
        "host_rss_slope_bytes_per_window": slope("host_terminal_rss_bytes"),
        "pinned_slope_bytes_per_sample": pinned_slope,
        "two_slab_bound_bytes": two_slab_bound,
        "resident_peak_arithmetic": {
            "peak_gpu_reserved_bytes": peak_gpu_reserved,
            "peak_gpu_allocated_bytes": peak_gpu_allocated,
            "allocator_slack_bytes": peak_gpu_reserved - peak_gpu_allocated,
            "unchanged_reserve_bytes": authority["unchanged_reserve_bytes"],
            "declared_vram_budget_bytes": gpu_budget,
            "budget_margin_bytes": gpu_budget - peak_gpu_reserved,
        },
    }


def _acs_telemetry(run: Mapping[str, Any], *, slots: int) -> dict[str, Any]:
    telemetry = _object(
        _provider_result(run).get("acs_checkpoint_staging"),
        name="ACS checkpoint telemetry",
    )
    if telemetry.get("slots") != slots:
        raise PhysicalB136ACSParityError("ACS slot telemetry differs")
    if slots == 0:
        return telemetry
    if (
        telemetry.get("schema_version") != "eiel_acs_native_detached_slots_v1"
        or telemetry.get("capture_mode") != "detached_file_backed_native_v1"
        or type(telemetry.get("queue_occupancy_high_water")) is not int
        or telemetry.get("queue_occupancy") != 0
        or telemetry.get("failure_latched") is not False
        or telemetry.get("aborted") is not False
        or type(telemetry.get("abandoned_snapshot_count")) is not int
        or telemetry.get("abandoned_snapshot_count") != 0
        or telemetry.get("closed") is not True
        or telemetry.get("writer_alive") is not False
        or telemetry.get("worker_accepts_cuda_objects") is not False
        or telemetry.get("worker_accepts_callbacks") is not False
        or telemetry.get("worker_tensor_device_policy")
        != "forced_cpu_context_v1"
        or type(telemetry.get("queued_cuda_references")) is not int
        or telemetry.get("queued_cuda_references") != 0
        or type(telemetry.get("queued_cuda_callbacks")) is not int
        or telemetry.get("queued_cuda_callbacks") != 0
        or not isinstance(telemetry.get("slot_states"), list)
        or telemetry.get("slot_states") != ["FREE"] * slots
        or type(telemetry.get("temporary_capture_file_high_water")) is not int
        or telemetry.get("temporary_capture_file_high_water")
        > MAX_TEMPORARY_CAPTURE_FILES
        or telemetry.get("temporary_capture_files_active") != 0
        or telemetry.get("detached_source_lease_count") != 0
        or telemetry.get("source_consumer_leases_active") != 0
        or telemetry.get("source_lease_spool_bytes_active") != 0
        or telemetry.get("capture_pool_count") != 1
        or telemetry.get("capture_pool_model")
        != "one_shared_two_slab_pool_not_per_slot"
        or telemetry.get("durable_checkpoint_frontier") != TOTAL_TICKS
    ):
        raise PhysicalB136ACSParityError(
            "asynchronous ACS queue high-water or terminal host-only telemetry is invalid"
        )
    return telemetry


def _restore_evidence(run: Mapping[str, Any]) -> dict[str, Any]:
    request_receipt = _object(run.get("request"), name="execution request")
    request = _object(request_receipt.get("request"), name="request payload")
    if request.get("restore_process_isolation") != "fresh_invocation_per_restore":
        raise PhysicalB136ACSParityError(
            "restored run lacks fresh-process request authority"
        )
    receipt = _object(run.get("receipt"), name="execution receipt")
    transitions = receipt.get("restore_process_transitions_completed")
    expected_event = {"batch_id": "batch-0000", "restore_tick": 3359}
    if (
        receipt.get("deliberate_restore_ticks_completed") != [3359]
        or receipt.get("resume_invocations") != 1
        or not isinstance(transitions, list)
        or len(transitions) != 1
    ):
        raise PhysicalB136ACSParityError(
            "fresh-process restore completion frontier is incomplete"
        )
    transition = _object(transitions[0], name="restore process transition")
    producer = _process_identity(
        transition.get("producer_process"), name="checkpoint producer process"
    )
    restorer = _process_identity(
        transition.get("restore_process"), name="checkpoint restore process"
    )
    if (
        transition.get("schema_version")
        != "eiel_execution_restore_process_transition_v1"
        or transition.get("event") != expected_event
        or producer.get("process_instance_id") == restorer.get("process_instance_id")
    ):
        raise PhysicalB136ACSParityError(
            "producer and restore process identities are not distinct and exact"
        )
    fresh_owner = _object(
        _provider_result(run).get("fresh_owner_restore_receipt"),
        name="fresh-owner restore receipt",
    )
    if (
        fresh_owner.get("schema_version")
        != "eiel_patch_fixed_fresh_owner_restore_receipt_v1"
        or fresh_owner.get("projection_bundles_reloaded") is not True
        or fresh_owner.get("cell_ids") != list(run["cell_ids"])
        or fresh_owner.get("source_checkpoint_next_absolute_tick") != 3359
        or fresh_owner.get("restore_next_tick") != 3359
        or fresh_owner.get("first_executed_absolute_tick") != 3359
        or fresh_owner.get("first_committed_next_tick") != 3360
        or fresh_owner.get("off_by_one_check") != "PASS"
    ):
        raise PhysicalB136ACSParityError(
            "fresh-owner tick frontier or off-by-one evidence differs"
        )
    source_digests = _digest_map(
        fresh_owner.get("source_checkpoint_cell_digests"),
        cell_ids=run["cell_ids"],
        name="fresh-owner source checkpoint digests",
    )
    raw_reloads = fresh_owner.get("reload_receipts")
    if not isinstance(raw_reloads, list) or len(raw_reloads) != len(run["cell_ids"]):
        raise PhysicalB136ACSParityError(
            "fresh-owner reload receipts do not exactly cover physical B136"
        )
    loaded_digests: dict[str, str] = {}
    for raw_reload in raw_reloads:
        reload = _object(raw_reload, name="fresh-owner reload receipt")
        cell_id = reload.get("cell_id")
        source_digest = reload.get("source_projection_sha256")
        loaded_digest = reload.get("loaded_projection_sha256")
        if (
            not isinstance(cell_id, str)
            or cell_id not in source_digests
            or cell_id in loaded_digests
            or source_digest != source_digests[cell_id]
            or loaded_digest != source_digest
        ):
            raise PhysicalB136ACSParityError(
                "fresh-owner loaded projection digest differs"
            )
        loaded_digests[cell_id] = loaded_digest
    if list(loaded_digests) != list(run["cell_ids"]):
        raise PhysicalB136ACSParityError(
            "fresh-owner reload receipt order differs from physical B136"
        )
    return {
        "checkpoint_digest": canonical_sha256(source_digests),
        "loaded_checkpoint_digest": canonical_sha256(loaded_digests),
        "process_transition": transition,
        "fresh_owner_restore_receipt_sha256": canonical_sha256(fresh_owner),
    }


def _guarded_telemetry(
    path_value: str | Path,
    run: Mapping[str, Any],
    *,
    source_git: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the one source-bound guarded ACS2 telemetry authority."""

    path = Path(path_value).expanduser().resolve()
    telemetry = _read_object(path, name="ACS2 guarded telemetry receipt")
    batch_path = Path(run["batch_path"]).resolve()
    request_path_value = telemetry.get("execution_request_path")
    if not isinstance(request_path_value, str) or not request_path_value:
        raise PhysicalB136ACSParityError(
            "guarded ACS2 telemetry execution request path is absent"
        )
    request_path = Path(request_path_value).expanduser().resolve()
    try:
        guarded_request = load_execution_request_v1(request_path)
    except (ExecutionFrontDoorError, OSError, UnicodeError) as exc:
        raise PhysicalB136ACSParityError(
            "guarded ACS2 telemetry execution request is invalid"
        ) from exc
    canonical_request = _object(
        _object(run.get("request"), name="execution request receipt").get("request"),
        name="canonical execution request",
    )
    argv = _object(telemetry.get("command"), name="guarded command").get("argv")
    repository_root_value = source_git.get("repository_root")
    expected_argv = None
    if isinstance(repository_root_value, str) and repository_root_value:
        expected_argv = [
            str(
                Path(repository_root_value).expanduser().resolve()
                / "tools"
                / "bin"
                / "eiel"
            ),
            "execute",
            str(request_path),
            "--human",
        ]
    if (
        telemetry.get("schema_version") != "eiel_acs_gpu_telemetry_receipt_v1"
        or telemetry.get("status") != "passed"
        or telemetry.get("launched") is not True
        or telemetry.get("process_attributed") is not True
        or telemetry.get("exclusive_gpu_window") is not True
        or telemetry.get("no_competing_gpu_workloads") is not True
        or telemetry.get("require_async_checkpoint_overlap") is not True
        or telemetry.get("source_commit") != source_git.get("commit")
        or telemetry.get("manifest_path") != str(batch_path)
        or telemetry.get("manifest_sha256") != canonical_sha256(run["batch"])
        or telemetry.get("execution_request_sha256")
        != _file_sha256(request_path, name="execution request")
        or guarded_request.payload() != canonical_request
        or not isinstance(argv, list)
        or any(not isinstance(value, str) or not value for value in argv)
        or argv != expected_argv
    ):
        raise PhysicalB136ACSParityError(
            "guarded ACS2 telemetry is not bound to the exact batch, request, or source"
        )
    overlap = _object(
        telemetry.get("async_checkpoint_overlap"),
        name="guarded asynchronous overlap evidence",
    )
    launched_pid = telemetry.get("launched_pid")
    target_gpu_pids = telemetry.get("target_gpu_pids")
    runtime_samples = telemetry.get("runtime_samples")
    if (
        type(launched_pid) is not int
        or launched_pid <= 0
        or not isinstance(target_gpu_pids, list)
        or not target_gpu_pids
        or any(type(pid) is not int or pid <= 0 for pid in target_gpu_pids)
        or target_gpu_pids != sorted(set(target_gpu_pids))
        or not isinstance(runtime_samples, list)
        or not runtime_samples
    ):
        raise PhysicalB136ACSParityError(
            "guarded ACS2 telemetry target PID authority is invalid"
        )
    observed_target_gpu_pids: set[int] = set()
    launched_pid_observed_in_tree = False
    for raw_sample in runtime_samples:
        sample = _object(raw_sample, name="guarded runtime GPU sample")
        sample_target_pids = sample.get("target_gpu_pids")
        process_tree_pids = sample.get("target_process_tree_pids")
        if (
            not isinstance(sample_target_pids, list)
            or any(type(pid) is not int or pid <= 0 for pid in sample_target_pids)
            or not isinstance(process_tree_pids, list)
            or any(type(pid) is not int or pid <= 0 for pid in process_tree_pids)
        ):
            raise PhysicalB136ACSParityError(
                "guarded runtime GPU sample PID authority is invalid"
            )
        observed_target_gpu_pids.update(sample_target_pids)
        launched_pid_observed_in_tree = launched_pid_observed_in_tree or (
            launched_pid in process_tree_pids
        )
    if (
        sorted(observed_target_gpu_pids) != target_gpu_pids
        or not launched_pid_observed_in_tree
    ):
        raise PhysicalB136ACSParityError(
            "guarded ACS2 telemetry target PID set is not sample-bound"
        )
    if (
        overlap.get("schema_version") != "eiel_acs_async_overlap_evidence_v5"
        or overlap.get("passed") is not True
        or overlap.get("slots") != 2
        or overlap.get("queue_occupancy_high_water") != 2
        or overlap.get("worker_accepts_cuda_objects") is not False
        or overlap.get("worker_accepts_callbacks") is not False
        or overlap.get("worker_tensor_device_policy")
        != "forced_cpu_context_v1"
        or type(overlap.get("queued_cuda_references")) is not int
        or overlap.get("queued_cuda_references") != 0
        or type(overlap.get("queued_cuda_callbacks")) is not int
        or overlap.get("queued_cuda_callbacks") != 0
        or overlap.get("temporary_capture_file_high_water")
        > MAX_TEMPORARY_CAPTURE_FILES
        or overlap.get("source_file_lifecycle_bounded") is not True
        or overlap.get("physical_owner_continuity_measured") is not True
        or overlap.get("exact_owner_transition_coverage") is not True
        or overlap.get("expected_owner_transition_ticks")
        != list(EXPECTED_OWNER_TRANSITION_TICKS)
        or overlap.get("observed_owner_transition_ticks")
        != list(EXPECTED_OWNER_TRANSITION_TICKS)
        or overlap.get("owner_epoch_transfer_ticks")
        != list(EXPECTED_OWNER_TRANSITION_TICKS)
    ):
        raise PhysicalB136ACSParityError(
            "guarded ACS2 overlap evidence does not prove the exact two-slot owner contract"
        )
    return telemetry


def _timeline_physical_evidence(run: Mapping[str, Any]) -> dict[str, Any]:
    result = _provider_result(run)
    binding = _object(
        result.get("tick_native_performance_timeline_binding"),
        name="tick-native performance timeline binding",
    )
    timeline = _object(binding.get("timeline"), name="tick-native performance timeline")
    window = _object(
        timeline.get("gpu_performance_window"), name="GPU performance window"
    )
    tail = _object(timeline.get("post_simulation_acs_tail"), name="ACS tail timeline")
    if (
        binding.get("schema_version")
        != "eiel_tick_native_performance_timeline_binding_v1"
        or timeline.get("schema_version") != "eiel_tick_native_performance_timeline_v1"
        or timeline.get("strict") is not True
        or window.get("post_simulation_acs_tail_excluded") is not True
        or tail.get("excluded_from_gpu_performance_metrics") is not True
        or tail.get("metric_name") != "T_ACS_TAIL"
        or timeline.get("wall_time_reconciliation", {}).get("pass") is not True
    ):
        raise PhysicalB136ACSParityError(
            "timeline does not separate the strict GPU window from T_ACS_TAIL"
        )
    dispatch_authority = _object(
        timeline.get("dispatch_overhead"), name="timeline dispatch overhead"
    )
    dispatch = _number(
        timeline.get("dispatch_overhead_fraction"),
        name="dispatch overhead fraction",
        minimum=0.0,
        maximum=1.0,
    )
    dispatch_domain_start_ns = _integer(
        dispatch_authority.get("dispatch_domain_start_ns"),
        name="timeline dispatch domain start",
        minimum=0,
    )
    dispatch_domain_end_ns = _integer(
        dispatch_authority.get("dispatch_domain_end_ns"),
        name="timeline dispatch domain end",
        minimum=1,
    )
    dispatch_span_ns = _integer(
        dispatch_authority.get("dispatch_span_ns"),
        name="timeline dispatch span duration",
        minimum=1,
    )
    simulation_union_ns = _integer(
        dispatch_authority.get("simulation_dispatch_union_ns"),
        name="timeline simulation dispatch union",
        minimum=0,
    )
    classified_non_dispatch_ns = _integer(
        dispatch_authority.get("classified_non_dispatch_union_ns"),
        name="timeline classified non-dispatch union",
        minimum=0,
    )
    dispatch_gap_ns = _integer(
        dispatch_authority.get("dispatch_gap_ns"),
        name="timeline unclassified dispatch gap",
        minimum=0,
    )
    dispatch_domain_ns = _integer(
        dispatch_authority.get("dispatch_domain_ns"),
        name="timeline simulation dispatch domain duration",
        minimum=1,
    )
    authority_fraction = _number(
        dispatch_authority.get("fraction"),
        name="timeline dispatch authority fraction",
        minimum=0.0,
        maximum=1.0,
    )
    performance_start_ns = _integer(
        window.get("start_ns"), name="timeline GPU performance start", minimum=0
    )
    performance_end_ns = _integer(
        window.get("end_ns"), name="timeline GPU performance end", minimum=1
    )
    if (
        dispatch_authority.get("definition")
        != "unclassified_simulation_dispatch_domain_fraction_v1"
        or dispatch_domain_end_ns <= dispatch_domain_start_ns
        or dispatch_span_ns != dispatch_domain_end_ns - dispatch_domain_start_ns
        or performance_start_ns > dispatch_domain_start_ns
        or dispatch_domain_end_ns > performance_end_ns
        or simulation_union_ns + classified_non_dispatch_ns + dispatch_gap_ns
        != dispatch_span_ns
        or simulation_union_ns + dispatch_gap_ns != dispatch_domain_ns
        or dispatch_authority.get("acs_tail_excluded") is not True
        or not math.isclose(
            authority_fraction,
            dispatch_gap_ns / float(dispatch_domain_ns),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            dispatch,
            authority_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PhysicalB136ACSParityError(
            "timeline dispatch overhead arithmetic is not source-bound"
        )
    return {
        "timeline_sha256": canonical_sha256(timeline),
        "gpu_performance_window": window,
        "acs_tail": tail,
        "dispatch_overhead": timeline.get("dispatch_overhead"),
        "dispatch_overhead_fraction": timeline.get("dispatch_overhead_fraction"),
    }


def _guarded_threshold_evidence(
    telemetry: Mapping[str, Any],
    *,
    timeline: Mapping[str, Any],
    batch_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _object(
        telemetry.get("post_analysis_summary"),
        name="guarded post-analysis summary",
    )
    activity = _object(summary.get("gpu_activity"), name="guarded GPU activity")
    sample_count = _integer(
        activity.get("sample_count"), name="GPU activity sample count", minimum=1
    )
    duty = _number(
        activity.get("active_sample_fraction"),
        name="active GPU duty fraction",
        minimum=0.0,
        maximum=1.0,
    )
    sm = _number(
        _path_value(activity, ("gpu_sm_percent", "mean")),
        name="mean SM utilization percent",
        minimum=0.0,
        maximum=100.0,
    )
    interval = _number(
        _path_value(activity, ("observed_interval_seconds", "max")),
        name="GPU telemetry maximum sample interval",
        minimum=0.0,
    )
    idle_gap = _number(
        activity.get("longest_idle_gap_observed_span_seconds"),
        name="longest observed simulation idle span",
        minimum=0.0,
    )
    requested_interval = _number(
        activity.get("requested_sample_interval_seconds"),
        name="requested GPU telemetry sample interval",
        minimum=0.000001,
    )
    if sample_count < 2 or interval > MAX_UNEXPLAINED_SIMULATION_IDLE_SECONDS:
        raise PhysicalB136ACSParityError(
            "guarded GPU telemetry lacks sufficient ordered cadence evidence"
        )
    dispatch_authority = _object(
        timeline.get("dispatch_overhead"), name="timeline dispatch overhead"
    )
    dispatch = _number(
        timeline.get("dispatch_overhead_fraction"),
        name="dispatch overhead fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if dispatch_authority.get("definition") != (
        "unclassified_simulation_dispatch_domain_fraction_v1"
    ):
        raise PhysicalB136ACSParityError(
            "timeline dispatch overhead lacks its exact clock definition"
        )
    performance_window = _object(
        timeline.get("gpu_performance_window"),
        name="timeline GPU performance window",
    )
    dispatch_domain_start_ns = _integer(
        dispatch_authority.get("dispatch_domain_start_ns"),
        name="timeline dispatch domain start",
        minimum=0,
    )
    dispatch_domain_end_ns = _integer(
        dispatch_authority.get("dispatch_domain_end_ns"),
        name="timeline dispatch domain end",
        minimum=1,
    )
    dispatch_span_ns = _integer(
        dispatch_authority.get("dispatch_span_ns"),
        name="timeline dispatch span duration",
        minimum=1,
    )
    simulation_union_ns = _integer(
        dispatch_authority.get("simulation_dispatch_union_ns"),
        name="timeline simulation dispatch union",
        minimum=0,
    )
    classified_non_dispatch_ns = _integer(
        dispatch_authority.get("classified_non_dispatch_union_ns"),
        name="timeline classified non-dispatch union",
        minimum=0,
    )
    dispatch_gap_ns = _integer(
        dispatch_authority.get("dispatch_gap_ns"),
        name="timeline unclassified dispatch gap",
        minimum=0,
    )
    dispatch_domain_ns = _integer(
        dispatch_authority.get("dispatch_domain_ns"),
        name="timeline simulation dispatch domain duration",
        minimum=1,
    )
    for field in (
        "capture_detach_union_ns",
        "persistence_union_ns",
        "owner_preparation_union_ns",
    ):
        _integer(
            dispatch_authority.get(field),
            name=f"timeline {field}",
            minimum=0,
        )
    authority_fraction = _number(
        dispatch_authority.get("fraction"),
        name="timeline dispatch authority fraction",
        minimum=0.0,
        maximum=1.0,
    )
    performance_start_ns = _integer(
        performance_window.get("start_ns"),
        name="timeline GPU performance window start",
        minimum=0,
    )
    performance_end_ns = _integer(
        performance_window.get("end_ns"),
        name="timeline GPU performance window end",
        minimum=1,
    )
    release_boundary = _object(
        summary.get("terminal_gpu_release_boundary"),
        name="terminal GPU release boundary",
    )
    post_release_residency = _object(
        summary.get("post_release_target_gpu_residency"),
        name="post-release target GPU residency",
    )
    released_perf_ns = _integer(
        release_boundary.get("released_perf_ns"),
        name="terminal GPU release timestamp",
        minimum=1,
    )
    post_release_sample_count = _integer(
        post_release_residency.get("post_release_sample_count"),
        name="post-release GPU residency sample count",
        minimum=1,
    )
    resident_sample_count = _integer(
        post_release_residency.get("resident_sample_count"),
        name="post-release target resident sample count",
        minimum=0,
    )
    max_post_release_framebuffer_mib = _number(
        post_release_residency.get("max_framebuffer_memory_mib"),
        name="post-release target framebuffer memory",
        minimum=0.0,
    )
    resident_rows = post_release_residency.get("resident_rows")
    if (
        telemetry.get("require_post_release_zero_residency") is not True
        or release_boundary.get("schema_version")
        != "eiel_terminal_gpu_release_boundary_v1"
        or release_boundary.get("terminal_checkpoint_tick")
        != performance_window.get("terminal_checkpoint_tick")
        or release_boundary.get("cuda_device_reset_return_code") != 0
        or release_boundary.get("performance_window_end_bound") is not True
        or released_perf_ns != performance_end_ns
        or post_release_residency.get("schema_version")
        != "eiel_post_release_target_gpu_residency_v1"
        or post_release_residency.get("released_perf_ns") != released_perf_ns
        or post_release_residency.get("settlement_grace_seconds") != 0.0
        or post_release_residency.get("observation_not_before_perf_ns")
        != released_perf_ns
        or post_release_residency.get("sample_window_clock_bound") is not True
        or post_release_residency.get("launched_pid")
        != telemetry.get("launched_pid")
        or post_release_residency.get("attributed_target_gpu_pids")
        != telemetry.get("target_gpu_pids")
        or post_release_residency.get("passed") is not True
        or resident_sample_count != 0
        or max_post_release_framebuffer_mib != 0.0
        or resident_rows != []
    ):
        raise PhysicalB136ACSParityError(
            "guarded telemetry does not prove zero target GPU residency after terminal release"
        )
    if (
        dispatch_domain_end_ns <= dispatch_domain_start_ns
        or dispatch_span_ns != dispatch_domain_end_ns - dispatch_domain_start_ns
        or performance_start_ns > dispatch_domain_start_ns
        or dispatch_domain_end_ns > performance_end_ns
        or simulation_union_ns + classified_non_dispatch_ns + dispatch_gap_ns
        != dispatch_span_ns
        or simulation_union_ns + dispatch_gap_ns != dispatch_domain_ns
        or dispatch_authority.get("acs_tail_excluded") is not True
        or not math.isclose(
            authority_fraction,
            dispatch_gap_ns / float(dispatch_domain_ns),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            dispatch,
            authority_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PhysicalB136ACSParityError(
            "timeline dispatch overhead arithmetic is not source-bound"
        )
    if (
        duty < MIN_ACTIVE_GPU_DUTY_FRACTION
        or sm < MIN_MEAN_SM_FRACTION * 100.0
        or requested_interval > MAX_UNEXPLAINED_SIMULATION_IDLE_SECONDS
        or idle_gap > MAX_UNEXPLAINED_SIMULATION_IDLE_SECONDS
        or dispatch > MAX_DISPATCH_OVERHEAD_FRACTION
    ):
        raise PhysicalB136ACSParityError(
            "guarded GPU duty, SM, cadence/idle, or dispatch overhead exceeds threshold"
        )
    overlap = _object(
        telemetry.get("async_checkpoint_overlap"),
        name="guarded overlap evidence",
    )
    windows = overlap.get("checkpoint_windows")
    if not isinstance(windows, list) or not windows:
        raise PhysicalB136ACSParityError("guarded overlap lacks checkpoint windows")
    batch_result = _object(
        batch_manifest.get("result"), name="guarded batch result"
    )
    raw_captures = batch_result.get("acs_capture_measurements")
    if not isinstance(raw_captures, list) or not raw_captures:
        raise PhysicalB136ACSParityError(
            "guarded batch lacks authoritative capture measurements"
        )
    captures_by_coordinate: dict[int, dict[str, Any]] = {}
    for capture_index, raw_capture in enumerate(raw_captures):
        capture = _object(
            raw_capture, name=f"guarded batch capture {capture_index}"
        )
        coordinate = _integer(
            capture.get("coordinate"),
            name=f"guarded batch capture {capture_index} coordinate",
        )
        if coordinate in captures_by_coordinate:
            raise PhysicalB136ACSParityError(
                "guarded batch capture coordinates are not unique"
            )
        captures_by_coordinate[coordinate] = capture
    bounded_windows = 0
    for index, raw_window in enumerate(windows):
        window = _object(raw_window, name=f"guarded checkpoint window {index}")
        if not window.get("nonterminal"):
            continue
        coordinate = _integer(
            window.get("coordinate"),
            name=f"checkpoint window {index} coordinate",
        )
        capture_authority = captures_by_coordinate.get(coordinate)
        if capture_authority is None:
            raise PhysicalB136ACSParityError(
                f"checkpoint window {index} lacks a bound batch capture"
            )
        capture_started = _integer(
            window.get("capture_started_perf_ns"),
            name=f"checkpoint window {index} capture start",
        )
        submitted = _integer(
            window.get("capture_submitted_perf_ns"),
            name=f"checkpoint window {index} capture submission",
        )
        capture_completed = _integer(
            window.get("capture_completed_perf_ns"),
            name=f"checkpoint window {index} capture completion",
        )
        _integer(
            window.get("capture_barrier_time_ns"),
            name=f"checkpoint window {index} capture barrier",
        )
        released = _integer(
            window.get("source_released_perf_ns"),
            name=f"checkpoint window {index} source release",
        )
        release_authority = _integer(
            window.get("release_authority_perf_ns"),
            name=f"checkpoint window {index} release authority",
        )
        dispatch_started = _integer(
            window.get("successor_dispatch_started_perf_ns"),
            name=f"checkpoint window {index} successor dispatch start",
        )
        prior_useful = _integer(
            capture_authority.get(
                "last_useful_kernel_completion_observed_host_time_ns"
            ),
            name=f"checkpoint window {index} prior useful completion",
        )
        useful = _integer(
            window.get("successor_last_useful_kernel_completion_observed_host_time_ns"),
            name=f"checkpoint window {index} useful successor completion",
        )
        if (
            window.get("capture_clock_ordered") is not True
            or window.get("gpu_safe_release_ordered") is not True
            or window.get("capture_then_writer_ordered") is not True
            or window.get("source_lease_ordered") is not True
            or window.get("successor_dispatch_ordered") is not True
            or window.get("release_authority_kind") != "source"
            or release_authority != released
            or not (
                capture_started <= capture_completed <= submitted <= release_authority
            )
            or release_authority - capture_started
            > MAX_CAPTURE_RELEASE_SECONDS * 1_000_000_000
            or prior_useful > dispatch_started
            or dispatch_started - prior_useful
            > MAX_SUCCESSOR_GPU_WORK_SECONDS * 1_000_000_000
            or useful < dispatch_started
        ):
            raise PhysicalB136ACSParityError(
                f"guarded checkpoint timing window {index} exceeds the 10-second bound"
            )
        bounded_windows += 1
    if bounded_windows == 0:
        raise PhysicalB136ACSParityError(
            "guarded overlap lacks a nonterminal capture/release successor window"
        )
    return {
        "sample_count": sample_count,
        "active_gpu_duty_fraction": duty,
        "mean_sm_fraction": sm / 100.0,
        "maximum_sample_interval_seconds": interval,
        "longest_idle_gap_seconds": idle_gap,
        "dispatch_overhead_fraction": dispatch,
        "bounded_nonterminal_windows": bounded_windows,
        "terminal_gpu_released_perf_ns": released_perf_ns,
        "post_release_gpu_residency_sample_count": post_release_sample_count,
        "post_release_max_framebuffer_memory_mib": (
            max_post_release_framebuffer_mib
        ),
    }


def _owner_physical_evidence(
    run: Mapping[str, Any], *, require_transitions: bool
) -> dict[str, Any]:
    result = _provider_result(run)
    if (
        result.get("resident_owner_max_simultaneous") != 1
        or result.get("resident_owner_live_count_terminal") != 0
    ):
        raise PhysicalB136ACSParityError(
            "provider owner telemetry does not prove one owner and zero terminal overlap"
        )
    expected = result.get("expected_resident_owner_transition_ticks")
    observed = result.get("observed_resident_owner_transition_ticks")
    if not isinstance(expected, list) or not isinstance(observed, list):
        raise PhysicalB136ACSParityError("owner transition telemetry is absent")
    if require_transitions and (
        expected != list(EXPECTED_OWNER_TRANSITION_TICKS)
        or observed != list(EXPECTED_OWNER_TRANSITION_TICKS)
    ):
        raise PhysicalB136ACSParityError(
            "provider owner transition telemetry does not cover 1120/2240 exactly"
        )
    transfers = result.get("resident_owner_epoch_transfer_receipts")
    if not isinstance(transfers, list):
        raise PhysicalB136ACSParityError("owner transfer receipts are absent")
    if require_transitions:
        ticks: list[int] = []
        for index, raw_transfer in enumerate(transfers):
            transfer = _object(raw_transfer, name=f"owner transfer receipt {index}")
            if (
                transfer.get("schema_version")
                != "eiel_resident_owner_epoch_transfer_v1"
                or transfer.get("physical_owner_reused") is not True
                or transfer.get("cuda_allocation_recreated") is not False
                or transfer.get("simultaneous_physical_owner_count") != 1
            ):
                raise PhysicalB136ACSParityError(
                    "owner transfer receipt proves overlap or CUDA reallocation"
                )
            ticks.append(
                _integer(
                    transfer.get("boundary_next_absolute_tick"),
                    name="owner transfer tick",
                )
            )
        if ticks != list(EXPECTED_OWNER_TRANSITION_TICKS):
            raise PhysicalB136ACSParityError(
                "owner transfer ticks are not exactly 1120/2240"
            )
    return {
        "owner_max_simultaneous": 1,
        "owner_live_count_terminal": 0,
        "expected_transition_ticks": expected,
        "observed_transition_ticks": observed,
    }


def materialize_physical_b136_acs_parity_v1(
    sync_run_root: str | Path,
    sync_restored_run_root: str | Path,
    async_run_root: str | Path,
    async_restored_run_root: str | Path,
    output_path: str | Path,
    acs2_gpu_telemetry_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and materialize PREQUAL-4A/B/C cross-route parity evidence.

    The ACS2 guarded telemetry receipt is mandatory for production-profile
    roots.  It is intentionally a single authority bound to the uninterrupted
    ACS2 batch; the fresh-restore root is checked through its own provider and
    native receipts and is not assigned a fictitious process-local queue peak.
    """

    sync = _validated_run(sync_run_root)
    sync_restored = _validated_run(sync_restored_run_root)
    asynchronous = _validated_run(async_run_root)
    async_restored = _validated_run(async_restored_run_root)
    runs = (sync, sync_restored, asynchronous, async_restored)
    roots = [str(run["root"]) for run in runs]
    if len(set(roots)) != 4:
        raise PhysicalB136ACSParityError("parity inputs must use four distinct roots")
    logical_contracts = [_logical_contract(run) for run in runs]
    logical_sha = canonical_sha256(logical_contracts[0])
    if any(canonical_sha256(value) != logical_sha for value in logical_contracts[1:]):
        raise PhysicalB136ACSParityError(
            "ACS0/ACS2 uninterrupted/restored logical execution contracts differ"
        )
    physical_authorities = [_physical_plan_authority(run) for run in runs]
    physical_authority_sha = canonical_sha256(physical_authorities[0])
    if any(
        canonical_sha256(value) != physical_authority_sha
        for value in physical_authorities[1:]
    ):
        raise PhysicalB136ACSParityError(
            "physical plan/preflight authorities differ across the four roots"
        )
    git_identities = [
        _object(
            _object(run["result"].get("source_identity"), name="source identity").get(
                "git"
            ),
            name="Git source identity",
        )
        for run in runs
    ]
    if any(value != git_identities[0] for value in git_identities[1:]):
        raise PhysicalB136ACSParityError(
            "parity inputs do not share one exact Git working-state identity"
        )
    plans = [run["plan"] for run in runs]
    slots = [
        _object(plan.get("execution"), name="execution").get("acs_checkpoint_slots")
        for plan in plans
    ]
    if slots != [0, 0, 2, 2]:
        raise PhysicalB136ACSParityError(
            "parity inputs must be ordered ACS0 uninterrupted/restored then "
            "ACS2 uninterrupted/restored"
        )
    for run, slot, restored in zip(
        runs, slots, (False, True, False, True), strict=True
    ):
        _request_contract(run, slots=slot, restored=restored)
    if (
        sync["route"] != sync_restored["route"]
        or asynchronous["route"] != async_restored["route"]
        or sync["route"] == asynchronous["route"]
    ):
        raise PhysicalB136ACSParityError(
            "ACS0/ACS2 route separation or restored route identity differs"
        )
    terminals = [_terminal_evidence(run) for run in runs]
    parity_fields = (
        "terminal_digest",
        "event_history_digest",
        "event_history_count",
        "terminal_event_ring_occupancy",
        "event_frontier_sha256",
    )
    for field in parity_fields:
        if len({value[field] for value in terminals}) != 1:
            raise PhysicalB136ACSParityError(
                f"ACS0/ACS2 uninterrupted/restored {field} differ"
            )
    sync_checkpoint, _ = _catalog_checkpoint_digest(sync, tick=3359)
    sync_restored_checkpoint, _ = _catalog_checkpoint_digest(sync_restored, tick=3359)
    async_checkpoint, _ = _catalog_checkpoint_digest(asynchronous, tick=3359)
    async_restored_checkpoint, _ = _catalog_checkpoint_digest(async_restored, tick=3359)
    sync_restore = _restore_evidence(sync_restored)
    async_restore = _restore_evidence(async_restored)
    sync_projection = _catalog_projection_digest(sync, tick=3359)
    sync_restored_projection = _catalog_projection_digest(sync_restored, tick=3359)
    async_projection = _catalog_projection_digest(asynchronous, tick=3359)
    async_restored_projection = _catalog_projection_digest(async_restored, tick=3359)
    if not (
        sync_checkpoint
        == sync_restored_checkpoint
        == async_checkpoint
        == async_restored_checkpoint
        and sync_projection
        == sync_restored_projection
        == sync_restore["checkpoint_digest"]
        == sync_restore["loaded_checkpoint_digest"]
        and async_projection
        == async_restored_projection
        == async_restore["checkpoint_digest"]
        == async_restore["loaded_checkpoint_digest"]
    ):
        raise PhysicalB136ACSParityError(
            "tick-3359 canonical state or route-local loaded checkpoint digests differ"
        )
    sync_telemetry = _acs_telemetry(sync, slots=0)
    sync_restored_telemetry = _acs_telemetry(sync_restored, slots=0)
    async_telemetry = _acs_telemetry(asynchronous, slots=2)
    async_restored_telemetry = _acs_telemetry(async_restored, slots=2)
    if (
        type(async_telemetry.get("queue_occupancy_high_water")) is not int
        or async_telemetry["queue_occupancy_high_water"] != 2
        or type(async_restored_telemetry.get("queue_occupancy_high_water")) is not int
        or async_restored_telemetry["queue_occupancy_high_water"] != 1
    ):
        raise PhysicalB136ACSParityError(
            "ACS2 queue occupancy high-water is not exact for uninterrupted/fresh restore"
        )
    source_git = git_identities[0]
    if acs2_gpu_telemetry_receipt_path is None:
        raise PhysicalB136ACSParityError(
            "one mandatory ACS2 guarded telemetry receipt is required"
        )
    guarded = _guarded_telemetry(
        acs2_gpu_telemetry_receipt_path,
        asynchronous,
        source_git=source_git,
    )
    native_resources = [
        _native_resource_measurements(run, authority=physical_authorities[index])
        for index, run in enumerate(runs)
    ]
    owner_uninterrupted = _owner_physical_evidence(
        asynchronous, require_transitions=True
    )
    owner_restored = _owner_physical_evidence(async_restored, require_transitions=False)
    timelines = [
        _timeline_physical_evidence(asynchronous),
        _timeline_physical_evidence(async_restored),
    ]
    guarded_thresholds = _guarded_threshold_evidence(
        guarded,
        timeline=timelines[0],
        batch_manifest=asynchronous["batch"],
    )
    body: dict[str, Any] = {
        "schema_version": PHYSICAL_B136_ACS_PARITY_SCHEMA_V1,
        "status": "PASS",
        "gate": "PREQUAL-4C",
        "logical_contract_sha256": logical_sha,
        "physical_authority_sha256": physical_authority_sha,
        "ordered_cell_ids_sha256": canonical_sha256(sync["cell_ids"]),
        "source_git_identity": git_identities[0],
        "runs": {
            name: {
                "run_root": str(run["root"]),
                "request_sha256": run["result"].get("request_sha256"),
                "plan_sha256": run["plan"].get("plan_sha256"),
                "execution_result_sha256": run["result"].get("result_sha256"),
                "execution_route_sha256": run["route"],
                "acs_slots": slot,
            }
            for name, run, slot in zip(
                (
                    "acs0_uninterrupted",
                    "acs0_restored",
                    "acs2_uninterrupted",
                    "acs2_restored",
                ),
                runs,
                slots,
                strict=True,
            )
        },
        "terminal_parity": {field: terminals[0][field] for field in parity_fields},
        "checkpoint_3359_parity": {
            "canonical_state_digest": sync_checkpoint,
            "acs0_projection_digest": sync_projection,
            "acs0_restored_projection_digest": sync_restore["loaded_checkpoint_digest"],
            "acs2_projection_digest": async_projection,
            "acs2_restored_projection_digest": async_restore[
                "loaded_checkpoint_digest"
            ],
            "status": "EXACT",
        },
        "fresh_process_restore": {
            "acs0_process_transition": sync_restore["process_transition"],
            "acs0_fresh_owner_restore_receipt_sha256": sync_restore[
                "fresh_owner_restore_receipt_sha256"
            ],
            "acs2_process_transition": async_restore["process_transition"],
            "acs2_fresh_owner_restore_receipt_sha256": async_restore[
                "fresh_owner_restore_receipt_sha256"
            ],
            "status": "EXACT",
        },
        "acs_telemetry": {
            "acs0_uninterrupted_sha256": canonical_sha256(sync_telemetry),
            "acs0_restored_sha256": canonical_sha256(sync_restored_telemetry),
            "acs2_uninterrupted_sha256": canonical_sha256(async_telemetry),
            "acs2_restored_sha256": canonical_sha256(async_restored_telemetry),
            "asynchronous_queue_occupancy_high_water": async_telemetry[
                "queue_occupancy_high_water"
            ],
            "fresh_restore_queue_occupancy_high_water": async_restored_telemetry[
                "queue_occupancy_high_water"
            ],
        },
        "physical_thresholds": {
            "guarded_telemetry_receipt_sha256": canonical_sha256(guarded),
            "native_resource_windows": {
                name: value
                for name, value in zip(
                    (
                        "acs0_uninterrupted",
                        "acs0_restored",
                        "acs2_uninterrupted",
                        "acs2_restored",
                    ),
                    native_resources,
                    strict=True,
                )
            },
            "owner_uninterrupted": owner_uninterrupted,
            "owner_restored": owner_restored,
            "timelines": timelines,
            "guarded": guarded_thresholds,
            "thresholds": {
                "minimum_free_vram_bytes": MINIMUM_FREE_VRAM_BYTES,
                "unchanged_reserve_bytes": UNCHANGED_RESERVE_BYTES,
                "capture_release_seconds_max": MAX_CAPTURE_RELEASE_SECONDS,
                "successor_gpu_work_seconds_max": MAX_SUCCESSOR_GPU_WORK_SECONDS,
                "active_gpu_duty_fraction_min": MIN_ACTIVE_GPU_DUTY_FRACTION,
                "mean_sm_fraction_min": MIN_MEAN_SM_FRACTION,
                "dispatch_overhead_fraction_max": MAX_DISPATCH_OVERHEAD_FRACTION,
                "unexplained_simulation_idle_seconds_max": (
                    MAX_UNEXPLAINED_SIMULATION_IDLE_SECONDS
                ),
                "temporary_capture_files_max": MAX_TEMPORARY_CAPTURE_FILES,
            },
        },
        "claim_boundary": (
            "PREQUAL physical-B136 cross-route parity only; this receipt does not "
            "grant Qualification, production, performance, or science authority."
        ),
    }
    evidence = {**body, "receipt_sha256": canonical_sha256(body)}
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        if _read_object(output, name="existing parity evidence") != evidence:
            raise PhysicalB136ACSParityError(
                "output path contains different parity evidence"
            )
        return evidence
    atomic_write_json(output, evidence)
    return evidence


__all__ = [
    "EventSinkError",
    "PHYSICAL_B136_ACS_PARITY_SCHEMA_V1",
    "PhysicalB136ACSParityError",
    "event_history_digest_from_references_v1",
    "materialize_physical_b136_acs_parity_v1",
    "validate_event_sink_prefix_reference_v1",
]
