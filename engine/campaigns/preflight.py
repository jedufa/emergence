"""Read-only planning preflight for compiled declarative campaigns.

This module deliberately grants no execution authority.  It verifies that a
compiled plan still refers to the expected immutable artifacts and that its
declared resource envelope is plausible on the inspected CUDA device.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Final

from engine.qrr.execution_request_v1 import build_execution_request_v1

from .artifact_lifecycle import (
    ArtifactLifecycleError,
    RunStorageGuard,
    ensure_lifecycle_contract_v1,
    progressive_storage_reservation_window_bytes_v1,
)
from .canonical import canonical_sha256
from .contracts import CampaignPlanV1, ExecutionRouteContractV1, RuntimeProfileV1
from .execution_policy import production_execution_policy_reasons_v1
from .providers import (
    ExecutionProviderRegistryV1,
    default_execution_provider_registry,
)
from .production_engine_identity import PRODUCTION_PROFILE_ID
from .qualification import (
    QualificationReceiptError,
    execution_route_qualification_sidecar_path_v1,
    execution_route_qualification_status_v1,
    qualification_reuse_receipt_v1,
    read_execution_route_qualification_sidecar_v1,
)
from .registry import ProfileRegistryV1, default_profile_registry
from .resources import (
    minimum_residual_headroom_bytes_v1,
    required_runtime_environment_v1,
)
from .tick_native_control import physical_b136_event_window_capacity_v1
from .resident_prepared_owner_resource_audit_v1 import (
    MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION,
    detect_cuda_hardware_identity_v1,
    prepared_owner_configuration_sha256_v1,
    validate_cuda_hardware_identity_v1,
    validate_prepared_owner_resource_audit_receipt_v1,
)


PREFLIGHT_SCHEMA_VERSION: Final[str] = "eiel_campaign_preflight_v1"
CUDA_HEADROOM_FRACTION: Final[float] = 0.04
CudaMemoryDetectorV1 = Callable[[], tuple[int, int]]
CudaHardwareDetectorV1 = Callable[[], Mapping[str, Any]]
RuntimeEnvironmentDetectorV1 = Callable[[], Mapping[str, str | None]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_cuda_memory_v1() -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for the current CUDA device."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def detect_runtime_environment_v1() -> Mapping[str, str | None]:
    """Return runtime environment values that can affect CUDA allocation."""

    return {"PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}


def _member(plan: CampaignPlanV1 | Mapping[str, Any], name: str) -> Any:
    if isinstance(plan, Mapping):
        public_name = "resolved_profile" if name == "profile_id" else name
        if name == "batches":
            public_name = "physical_batches"
        return plan.get(public_name)
    return getattr(plan, name)


def _validate_allocation_diagnostic_resource_receipt_v1(
    receipt: Mapping[str, Any],
    *,
    plan: CampaignPlanV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact B48 allocation-only evidence for live admission.

    This admits hardware resources only.  It neither changes route
    qualification nor treats an allocation diagnostic as runtime evidence.
    """

    row = dict(receipt)
    receipt_sha256 = row.pop("receipt_sha256", None)
    if row.get("schema_version") != "eiel_b48_wdl_jungle_eed_allocation_diagnostic_v1":
        raise ValueError("allocation diagnostic schema is unsupported")
    if receipt_sha256 != canonical_sha256(row):
        raise ValueError("allocation diagnostic receipt SHA-256 mismatch")
    if row.get("authority") != "allocation_only_not_execution_admission_v1":
        raise ValueError("allocation diagnostic authority is invalid")
    payload = _plan_payload_for_authorization(plan)
    geometry = row.get("geometry")
    if not isinstance(geometry, Mapping) or {
        "physical_batch_cells": geometry.get("physical_batch_cells"),
        "generations": geometry.get("generations"),
        "ticks_per_generation": geometry.get("ticks_per_generation"),
    } != {"physical_batch_cells": 48, "generations": 200, "ticks_per_generation": 160}:
        raise ValueError("allocation diagnostic geometry is not exact B48/G200/T160")
    artifacts = _member(plan, "artifacts")
    artifact = row.get("artifact")
    if not isinstance(artifacts, Mapping) or not isinstance(artifact, Mapping):
        raise ValueError("allocation diagnostic artifact binding is incomplete")
    for receipt_name, plan_name in (
        ("sha256", "cuda_extension"),
        ("manifest_sha256", "extension_manifest"),
    ):
        expected = (
            artifacts.get(plan_name, {}).get("sha256")
            if isinstance(artifacts.get(plan_name), Mapping)
            else None
        )
        if artifact.get(receipt_name) != expected:
            raise ValueError(f"allocation diagnostic {plan_name} binding mismatch")
    outcome = row.get("outcome")
    decision = row.get("decision")
    if (
        row.get("eed_overlay") is not True
        or row.get("executor_call_count") != 0
        or row.get("scientific_tick_count") != 0
        or not isinstance(outcome, Mapping)
        or outcome.get("status") != "prepared"
        or outcome.get("cuda_oom") is not False
        or not isinstance(decision, Mapping)
        or decision.get("headroom_ok") is not True
        or decision.get("teardown_ok") is not True
    ):
        raise ValueError(
            "allocation diagnostic is not a successful zero-dispatch receipt"
        )
    peak = decision.get("conservative_peak_bytes")
    if type(peak) is not int or peak <= 0:
        raise ValueError("allocation diagnostic peak is invalid")
    # The resident diagnostic records total device residency at each stage.
    # Live preflight already observes free memory after establishing its own
    # CUDA context, so charging the diagnostic baseline again double-counts
    # desktop/context residency.  When the staged measurements are present,
    # admit against the exact incremental owner footprint instead.
    stages = row.get("stages")
    baseline_device_bytes: int | None = None
    incremental_peak = peak
    measured_host_peak_bytes: int | None = None
    measured_host_baseline_bytes: int | None = None
    if stages is not None and (
        not isinstance(stages, Sequence) or isinstance(stages, (str, bytes))
    ):
        raise ValueError("allocation diagnostic stages are malformed")
    if isinstance(stages, Sequence) and not isinstance(stages, (str, bytes)):
        stage_rows = [stage for stage in stages if isinstance(stage, Mapping)]
        baseline = next(
            (stage for stage in stage_rows if stage.get("stage") == "process_baseline"),
            None,
        )
        measured_peaks = [
            stage.get("process_gpu_bytes")
            for stage in stage_rows
            if type(stage.get("process_gpu_bytes")) is int
        ]
        if baseline is None or not measured_peaks:
            raise ValueError("allocation diagnostic staged residency is incomplete")
        baseline_device_bytes = baseline.get("process_gpu_bytes")
        if (
            type(baseline_device_bytes) is not int
            or baseline_device_bytes < 0
            or max(measured_peaks) != peak
            or peak <= baseline_device_bytes
        ):
            raise ValueError("allocation diagnostic staged residency differs")
        incremental_peak = peak - baseline_device_bytes
        host_rows = [
            (
                stage.get("host_memory", {}).get("hwm_bytes")
                if isinstance(stage.get("host_memory"), Mapping)
                else None
            )
            for stage in stage_rows
        ]
        host_peaks = [value for value in host_rows if type(value) is int and value >= 0]
        if host_peaks:
            baseline_host = (
                baseline.get("host_memory", {}).get("hwm_bytes")
                if isinstance(baseline.get("host_memory"), Mapping)
                else None
            )
            if type(baseline_host) is not int or baseline_host < 0:
                raise ValueError("allocation diagnostic host baseline is incomplete")
            measured_host_baseline_bytes = baseline_host
            measured_host_peak_bytes = max(host_peaks)
    exact_base_plan = row.get("base_b48_plan_sha256") == _member(plan, "plan_sha256")
    if exact_base_plan:
        if row.get("route_configuration_sha256") != canonical_sha256(
            {"cells": payload.get("cells"), "execution": payload.get("execution")}
        ):
            raise ValueError("allocation diagnostic route configuration mismatch")
        binding_scope = "exact_b48_plan_v1"
    else:
        execution = payload.get("execution")
        batches = payload.get("physical_batches")
        tick_native_control = (
            execution.get("tick_native_control")
            if isinstance(execution, Mapping)
            else None
        )
        legacy_time_shape = bool(
            isinstance(execution, Mapping)
            and execution.get("ticks_per_generation") == 160
            and type(execution.get("generations")) is int
            and 3 <= execution["generations"] <= 128
        )
        tick_native_time_shape = bool(
            isinstance(execution, Mapping)
            and execution.get("execution_route_identity")
            == "tick_native_execution_route_v1"
            and type(execution.get("total_ticks")) is int
            and execution["total_ticks"] > 0
            and isinstance(tick_native_control, Mapping)
            and tick_native_control.get("time_authority") == "absolute_tick_v1"
            and tick_native_control.get("source_window_ticks") == 160
            and "refill_chunk_ticks" not in tick_native_control
            and type(tick_native_control.get("resident_owner_span_ticks")) is int
            and type(tick_native_control.get("native_dispatch_window_ticks")) is int
            and tick_native_control["native_dispatch_window_ticks"] <= tick_native_control[
                "resident_owner_span_ticks"
            ]
            and "generations" not in execution
            and "ticks_per_generation" not in execution
            and "physical_window_generations" not in execution
        )
        if (
            not isinstance(execution, Mapping)
            or (
                execution.get("physical_window_generations") != 1
                if not tick_native_time_shape
                else False
            )
            or not (legacy_time_shape or tick_native_time_shape)
            or not isinstance(batches, Sequence)
            or isinstance(batches, (str, bytes))
            or not batches
            or any(
                not isinstance(batch, Mapping)
                or not isinstance(batch.get("cell_ids"), Sequence)
                or isinstance(batch["cell_ids"], (str, bytes))
                or not 1 <= len(batch["cell_ids"]) <= 48
                for batch in batches
            )
        ):
            raise ValueError("allocation diagnostic bounded resident shape mismatch")
        binding_scope = "bounded_resident_b48_shape_v1"
    return {
        "schema_version": "eiel_b48_allocation_diagnostic_resource_promotion_v1",
        "receipt_sha256": receipt_sha256,
        "vram_bytes_upper_bound": incremental_peak,
        "measured_total_device_peak_bytes": peak,
        "measured_baseline_device_residency_bytes": baseline_device_bytes,
        "measured_host_peak_bytes": measured_host_peak_bytes,
        "measured_host_baseline_bytes": measured_host_baseline_bytes,
        "restore_peak_host_bytes_upper_bound": measured_host_peak_bytes,
        "admission_bound_semantics": (
            "incremental_owner_footprint_after_observed_cuda_baseline_v1"
            if baseline_device_bytes is not None
            else "legacy_total_device_peak_v1"
        ),
        "authority": "allocation_only_not_execution_admission_v1",
        "binding_scope": binding_scope,
        "qualification_receipt_action": "unchanged",
    }


def _validate_sealed_allocation_resource_promotion_v1(
    promotion: Mapping[str, Any], *, plan: CampaignPlanV1 | Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse an integrity-sealed B48 promotion without receipt archaeology."""

    row = dict(promotion)
    if (
        row.get("schema_version")
        != "eiel_b48_allocation_diagnostic_resource_promotion_v1"
        or row.get("authority") != "allocation_only_not_execution_admission_v1"
        or row.get("binding_scope")
        not in {"exact_b48_plan_v1", "bounded_resident_b48_shape_v1"}
        or not isinstance(row.get("receipt_sha256"), str)
        or len(row["receipt_sha256"]) != 64
        or type(row.get("vram_bytes_upper_bound")) is not int
        or row["vram_bytes_upper_bound"] <= 0
    ):
        raise ValueError("sealed allocation promotion is malformed")
    payload = _plan_payload_for_authorization(plan)
    execution = payload.get("execution")
    batches = payload.get("physical_batches")
    control = (
        execution.get("tick_native_control") if isinstance(execution, Mapping) else None
    )
    if (
        not isinstance(execution, Mapping)
        or execution.get("execution_route_identity") != "tick_native_execution_route_v1"
        or not isinstance(control, Mapping)
        or control.get("time_authority") != "absolute_tick_v1"
        or control.get("source_window_ticks") != 160
        or "refill_chunk_ticks" in control
        or type(control.get("resident_owner_span_ticks")) is not int
        or type(control.get("native_dispatch_window_ticks")) is not int
        or control["native_dispatch_window_ticks"] > control["resident_owner_span_ticks"]
        or not isinstance(batches, Sequence)
        or isinstance(batches, (str, bytes))
        or not batches
        or any(
            not isinstance(batch, Mapping)
            or not isinstance(batch.get("cell_ids"), Sequence)
            or isinstance(batch["cell_ids"], (str, bytes))
            or not 1 <= len(batch["cell_ids"]) <= 48
            for batch in batches
        )
    ):
        raise ValueError("sealed allocation promotion does not cover this plan shape")
    return row


def _plan_payload_for_authorization(
    plan: CampaignPlanV1 | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(plan, Mapping):
        return plan
    return plan.payload()


def _profile_for_plan(
    plan: CampaignPlanV1 | Mapping[str, Any], registry: ProfileRegistryV1
) -> RuntimeProfileV1 | None:
    profile_id = _member(plan, "profile_id")
    return next(
        (
            profile
            for profile in registry.profiles()
            if profile.profile_id == profile_id
        ),
        None,
    )


def _positive_budget(
    name: str,
    explicit: Mapping[str, int] | None,
    estimate: Mapping[str, Any],
) -> int | None:
    if explicit is not None and name in explicit:
        value: Any = explicit[name]
    else:
        nested = estimate.get("declared_budgets", estimate.get("budgets", {}))
        nested_value = nested.get(name) if isinstance(nested, Mapping) else None
        value = estimate.get(
            f"declared_{name}", estimate.get(f"{name}_budget", nested_value)
        )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _route_qualification_status(
    plan: CampaignPlanV1 | Mapping[str, Any],
    contract: ExecutionRouteContractV1,
) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
    artifacts = _member(plan, "artifacts")
    if not isinstance(artifacts, Mapping):
        return execution_route_qualification_status_v1(contract, None), None, None
    candidates: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    seen: set[Path] = set()
    for _, raw in sorted(artifacts.items()):
        if not isinstance(raw, Mapping) or raw.get("kind") != "qualification_receipt":
            continue
        path = Path(str(raw.get("path", ""))).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        receipt = read_execution_route_qualification_sidecar_v1(
            path,
            contract.execution_route_sha256,
        )
        if receipt is None:
            continue
        status = execution_route_qualification_status_v1(contract, receipt)
        sidecar_path = execution_route_qualification_sidecar_path_v1(
            path,
            contract.execution_route_sha256,
        )
        if not sidecar_path.is_file():
            sidecar_path = execution_route_qualification_sidecar_path_v1(path)
        candidates.append((status, str(sidecar_path), receipt))
    if not candidates:
        return execution_route_qualification_status_v1(contract, None), None, None
    candidates.sort(
        key=lambda row: (
            row[0].get("authority_state") == "route_qualified",
            row[0].get("provider_binding_exact") is True,
            row[0].get("profile_binding_exact") is True,
            row[0].get("route_hash_exact") is True,
        ),
        reverse=True,
    )
    return candidates[0]


def preflight_campaign(
    plan: CampaignPlanV1 | Mapping[str, Any],
    *,
    declared_budgets: Mapping[str, int] | None = None,
    profile: RuntimeProfileV1 | None = None,
    registry: ProfileRegistryV1 | None = None,
    provider_registry: ExecutionProviderRegistryV1 | None = None,
    cuda_memory_detector: CudaMemoryDetectorV1 = detect_cuda_memory_v1,
    cuda_hardware_detector: CudaHardwareDetectorV1 = (detect_cuda_hardware_identity_v1),
    runtime_environment_detector: RuntimeEnvironmentDetectorV1 = (
        detect_runtime_environment_v1
    ),
    prepared_owner_resource_receipt: Mapping[str, Any] | None = None,
    allocation_diagnostic_resource_receipt: Mapping[str, Any] | None = None,
    sealed_allocation_resource_promotion: Mapping[str, Any] | None = None,
    enforce_storage_admission: bool = False,
    qualprep_mode: bool = False,
    engineering_mode: bool = False,
    nested_storage_parent_run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed, read-only preflight receipt for ``plan``.

    ``declared_budgets`` uses the campaign specification names
    ``disk_bytes`` and ``vram_bytes``.  A compiler may instead bind them into
    ``resource_estimate`` as ``declared_*`` or ``*_budget`` fields.
    """

    if type(qualprep_mode) is not bool or type(engineering_mode) is not bool:
        raise TypeError("preflight development modes must be exact bool values")
    if qualprep_mode and engineering_mode:
        raise ValueError("QualPrep and engineering preflight modes are exclusive")
    development_mode = qualprep_mode or engineering_mode
    authority_scope = (
        "qualprep_development_only"
        if qualprep_mode
        else (
            "engineering_unqualified_real_use"
            if engineering_mode
            else "qualified_execution"
        )
    )

    registry = default_profile_registry() if registry is None else registry
    provider_registry = (
        default_execution_provider_registry()
        if provider_registry is None
        else provider_registry
    )
    if declared_budgets is None and isinstance(plan, Mapping):
        execution = plan.get("execution")
        if isinstance(execution, Mapping) and isinstance(
            execution.get("budgets"), Mapping
        ):
            declared_budgets = execution["budgets"]
    reasons: list[dict[str, str]] = []
    artifact_rows: list[dict[str, Any]] = []

    artifacts = _member(plan, "artifacts")
    if not isinstance(artifacts, Mapping):
        artifacts = {}
        reasons.append(
            {
                "code": "artifacts_missing",
                "path": "artifacts",
                "message": "compiled plan has no artifact mapping",
            }
        )
    for name, raw in sorted(artifacts.items()):
        path = Path(str(raw.get("path", "")))
        expected = str(raw.get("sha256", ""))
        row: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "expected_sha256": expected,
            "regular_file": path.is_file(),
            "actual_sha256": None,
            "sha256_exact": False,
        }
        if row["regular_file"]:
            try:
                actual = _sha256(path)
                row["actual_sha256"] = actual
                row["sha256_exact"] = actual == expected
            except OSError as exc:
                row["error"] = str(exc)
        if not row["regular_file"]:
            reasons.append(
                {
                    "code": "artifact_not_regular_file",
                    "path": f"artifacts.{name}.path",
                    "message": f"artifact is not an existing regular file: {path}",
                }
            )
        elif not row["sha256_exact"]:
            reasons.append(
                {
                    "code": "artifact_sha256_mismatch",
                    "path": f"artifacts.{name}.sha256",
                    "message": f"artifact SHA-256 does not match: {name}",
                }
            )
        artifact_rows.append(row)

    estimate = _member(plan, "resource_estimate")
    if not isinstance(estimate, Mapping):
        estimate = {}
    estimate = dict(estimate)
    allocation_diagnostic_promotion: dict[str, Any] | None = None
    if (
        allocation_diagnostic_resource_receipt is not None
        and sealed_allocation_resource_promotion is not None
    ):
        reasons.append(
            {
                "code": "resource_authority_ambiguous",
                "path": "resource_authority",
                "message": "raw and sealed allocation authorities are alternatives",
            }
        )
    if allocation_diagnostic_resource_receipt is not None:
        try:
            allocation_diagnostic_promotion = (
                _validate_allocation_diagnostic_resource_receipt_v1(
                    allocation_diagnostic_resource_receipt,
                    plan=plan,
                )
            )
        except (TypeError, ValueError) as exc:
            reasons.append(
                {
                    "code": "allocation_diagnostic_resource_receipt_invalid",
                    "path": "allocation_diagnostic_resource_receipt",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    elif sealed_allocation_resource_promotion is not None:
        try:
            allocation_diagnostic_promotion = (
                _validate_sealed_allocation_resource_promotion_v1(
                    sealed_allocation_resource_promotion,
                    plan=plan,
                )
            )
        except (TypeError, ValueError) as exc:
            reasons.append(
                {
                    "code": "sealed_allocation_resource_promotion_invalid",
                    "path": "sealed_allocation_resource_promotion",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    profile_id = str(_member(plan, "profile_id"))
    resolved_profile = (
        profile if profile is not None else _profile_for_plan(plan, registry)
    )
    cuda_admission_required = (
        resolved_profile is None or resolved_profile.backend == "cuda_resident"
    )
    provider = provider_registry.get(profile_id)
    batches = _member(plan, "batches")
    if batches is None and isinstance(plan, Mapping):
        batches = plan.get("physical_batches")
    batch_ids = [
        getattr(batch, "batch_id", None)
        if not isinstance(batch, Mapping)
        else batch.get("batch_id")
        for batch in (batches or ())
    ]
    bound_requests = [
        build_execution_request_v1(provider, plan, batch_id)
        for batch_id in batch_ids
        if provider is not None and isinstance(batch_id, str)
    ]
    translated_requests = [bound.resident_request for bound in bound_requests]
    provider_estimate_error: str | None = None
    if provider is not None:
        try:
            estimator = getattr(provider, "estimate_resources", None)
            provider_estimate = (
                estimator(plan, tuple(translated_requests))
                if callable(estimator)
                else None
            )
            if provider_estimate is not None:
                estimate.update(provider_estimate)
        except (TypeError, ValueError) as exc:
            provider_estimate_error = f"{type(exc).__name__}: {exc}"
    if allocation_diagnostic_promotion is not None:
        # The provider projection remains in the receipt for provenance, but
        # an exact, plan-bound allocation diagnostic is the admission bound.
        estimate["vram_bytes_upper_bound"] = allocation_diagnostic_promotion[
            "vram_bytes_upper_bound"
        ]
        estimate["vram_bytes_unavailable_reason"] = None
    physical_b136_shape: dict[str, Any] = {
        "applicable": False,
        "status": "NOT_APPLICABLE",
    }
    plan_payload = _plan_payload_for_authorization(plan)
    execution_payload = plan_payload.get("execution")
    logical_cells = _member(plan, "cells")
    logical_cell_count = len(logical_cells) if isinstance(logical_cells, Sequence) else 0
    if (
        profile_id == PRODUCTION_PROFILE_ID
        and isinstance(execution_payload, Mapping)
        and execution_payload.get("execution_route_identity")
        == "tick_native_execution_route_v1"
        and logical_cell_count == 136
        and estimate.get("physical_batch_cells") == 136
    ):
        physical_b136_shape["applicable"] = True
        batch_rows = list(batches or ())
        batch_sizes = [
            len(row.cell_ids) if not isinstance(row, Mapping) else len(row.get("cell_ids", ()))
            for row in batch_rows
        ]
        control = execution_payload.get("tick_native_control")
        technical_gate = execution_payload.get("technical_gate")
        configured_capacity = (
            control.get("event_window_capacity")
            if isinstance(control, Mapping)
            else None
        )
        physical_b136_shape.update(
            {
                "logical_cells": logical_cell_count,
                "compiled_batch_count": len(batch_rows),
                "compiled_batch_sizes": batch_sizes,
                "technical_gate_partition": (
                    technical_gate.get("partition")
                    if isinstance(technical_gate, Mapping)
                    else None
                ),
                "event_window_capacity": configured_capacity,
                "expected_event_window_capacity": physical_b136_event_window_capacity_v1(),
            }
        )
        shape_ok = (
            len(batch_rows) == 1
            and batch_sizes == [136]
            and isinstance(technical_gate, Mapping)
            and technical_gate.get("partition") is False
            and configured_capacity == physical_b136_event_window_capacity_v1()
        )
        physical_b136_shape["status"] = "PASS" if shape_ok else "FAIL"
        if not shape_ok:
            reasons.append(
                {
                    "code": "physical_b136_shape_not_bound",
                    "path": "physical_batches",
                    "message": (
                        "tick-native B136 admission requires one compiled physical "
                        "batch of 136, partition=false, and exact source-derived "
                        "event capacity"
                    ),
                }
            )
    disk_estimate = estimate.get("disk_bytes_upper_bound")
    try:
        lifecycle = ensure_lifecycle_contract_v1(
            plan if isinstance(plan, Mapping) else plan.payload()
        )
    except (ArtifactLifecycleError, AttributeError, TypeError) as exc:
        lifecycle = None
        reasons.append(
            {
                "code": "artifact_lifecycle_invalid",
                "path": "artifact_lifecycle",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
    storage_admission: dict[str, Any] = {
        "enforced": bool(enforce_storage_admission),
        "status": "NOT_ENFORCED",
        "snapshot": None,
    }
    if lifecycle is not None:
        output = _member(plan, "output")
        runtime_root = (
            Path(str(output.get("runtime_root")))
            if isinstance(output, Mapping) and output.get("runtime_root")
            else Path.cwd()
        )
        try:
            reservation_window = progressive_storage_reservation_window_bytes_v1(
                estimate
            )
            guard = RunStorageGuard(
                lifecycle,
                run_root=runtime_root,
                planned_bytes=(
                    int(disk_estimate)
                    if type(disk_estimate) is int and disk_estimate >= 0
                    else None
                ),
                reservation_window_bytes=reservation_window,
                same_owner_nested_parent_run_root=nested_storage_parent_run_root,
            )
            storage_admission["snapshot"] = guard.preflight()
        except ArtifactLifecycleError as exc:
            storage_admission["snapshot"] = {
                "ok": False,
                "violations": [
                    {
                        "code": "progressive_storage_reservation_invalid",
                        "message": str(exc),
                    }
                ],
            }
        storage_admission["status"] = (
            "PASS"
            if storage_admission["snapshot"]["ok"]
            else "RESOURCE_BLOCKED"
        )
        if enforce_storage_admission and not storage_admission["snapshot"]["ok"]:
            reasons.append(
                {
                    "code": "lifecycle_storage_admission_failed",
                    "path": "artifact_lifecycle.storage",
                    "message": "filesystem state does not cover the declared lifecycle storage envelope",
                }
            )
    runtime_environment_requirements = required_runtime_environment_v1(estimate)
    observed_runtime_environment = {
        key: runtime_environment_detector().get(key)
        for key in runtime_environment_requirements
    }
    runtime_environment_ok = all(
        observed_runtime_environment.get(key) == expected
        for key, expected in runtime_environment_requirements.items()
    )
    minimum_residual_headroom_bytes = minimum_residual_headroom_bytes_v1(estimate)
    disk_budget = _positive_budget("disk_bytes", declared_budgets, estimate)
    disk_ok = (
        isinstance(disk_estimate, int)
        and not isinstance(disk_estimate, bool)
        and disk_estimate >= 0
        and disk_budget is not None
        and disk_estimate <= disk_budget
    )
    if not disk_ok:
        code = "disk_budget_missing" if disk_budget is None else "disk_budget_exceeded"
        reasons.append(
            {
                "code": code,
                "path": "execution.budgets.disk_bytes",
                "message": (
                    "a positive disk budget is required"
                    if disk_budget is None
                    else f"planned disk upper bound {disk_estimate} exceeds budget {disk_budget}"
                ),
            }
        )

    vram_budget = _positive_budget("vram_bytes", declared_budgets, estimate)
    vram_estimate = estimate.get("vram_bytes_upper_bound")
    vram_estimate_within_budget: bool | None = (
        (
            vram_estimate >= 0
            and vram_budget is not None
            and vram_estimate <= vram_budget
        )
        if type(vram_estimate) is int
        else None
    )
    if (
        cuda_admission_required
        and type(vram_estimate) is int
        and vram_budget is not None
        and not vram_estimate_within_budget
    ):
        reasons.append(
            {
                "code": "vram_budget_exceeded",
                "path": "execution.budgets.vram_bytes",
                "message": (
                    f"provider VRAM upper bound {vram_estimate} exceeds budget "
                    f"{vram_budget}"
                ),
            }
        )

    wall_budget = _positive_budget("wall_seconds", declared_budgets, estimate)
    wall_estimate = estimate.get("wall_seconds_upper_bound")
    wall_ok: bool | None = (
        0 <= wall_estimate <= wall_budget
        if type(wall_estimate) is int and wall_budget is not None
        else None
    )
    if wall_budget is None or wall_ok is False:
        reasons.append(
            {
                "code": (
                    "wall_budget_missing"
                    if wall_budget is None
                    else "wall_budget_exceeded_or_unavailable"
                ),
                "path": "execution.budgets.wall_seconds",
                "message": (
                    "a positive wall-time budget is required"
                    if wall_budget is None
                    else f"provider wall-time bound {wall_estimate} is not within "
                    f"budget {wall_budget}"
                ),
            }
        )
    exact_vram_bound = estimate.get("vram_bytes_upper_bound")
    conservative_vram_projection = estimate.get("vram_bytes_projection")
    if type(exact_vram_bound) is int and exact_vram_bound >= 0:
        live_vram_admission_bound = exact_vram_bound
        live_vram_admission_source = "provider_exact_upper_bound"
    elif type(conservative_vram_projection) is int and conservative_vram_projection > 0:
        live_vram_admission_bound = conservative_vram_projection
        live_vram_admission_source = "provider_conservative_projection"
    else:
        live_vram_admission_bound = vram_budget
        live_vram_admission_source = "declared_budget_fallback"
    cuda_row: dict[str, Any] = {
        "free_bytes": None,
        "total_bytes": None,
        "required_headroom_bytes": None,
        "declared_vram_budget_bytes": vram_budget,
        "provider_estimate_within_budget": vram_estimate_within_budget,
        "live_admission_bound_bytes": live_vram_admission_bound,
        "live_admission_bound_source": live_vram_admission_source,
        "budget_within_90_percent_total": False,
        "free_covers_admission_bound_and_headroom": False,
        "measurement_bootstrap_minimum_free_bytes": None,
        "projection_fits_total_memory_with_headroom": False,
        "free_covers_measurement_bootstrap_floor": False,
        "minimum_residual_headroom_bytes": minimum_residual_headroom_bytes,
        "runtime_environment_requirements": runtime_environment_requirements,
        "observed_runtime_environment": observed_runtime_environment,
        "runtime_environment_ok": runtime_environment_ok,
    }
    if not cuda_admission_required:
        cuda_row.update(
            {
                "admission_required": False,
                "skip_reason": f"profile backend {resolved_profile.backend!r}",
                "budget_within_90_percent_total": True,
                "free_covers_admission_bound_and_headroom": True,
            }
        )
    else:
        cuda_row["admission_required"] = True
        try:
            free_bytes, total_bytes = cuda_memory_detector()
            if (
                isinstance(free_bytes, bool)
                or isinstance(total_bytes, bool)
                or not isinstance(free_bytes, int)
                or not isinstance(total_bytes, int)
                or free_bytes < 0
                or total_bytes <= 0
                or free_bytes > total_bytes
            ):
                raise ValueError("CUDA detector returned invalid free/total bytes")
            headroom = math.ceil(total_bytes * CUDA_HEADROOM_FRACTION)
            measurement_bootstrap_floor = headroom + minimum_residual_headroom_bytes
            cuda_row.update(
                {
                    "free_bytes": free_bytes,
                    "total_bytes": total_bytes,
                    "required_headroom_bytes": headroom,
                    "budget_within_90_percent_total": (
                        vram_budget is not None
                        and vram_budget + headroom <= total_bytes
                    ),
                    "free_covers_admission_bound_and_headroom": (
                        live_vram_admission_bound is not None
                        and live_vram_admission_bound + headroom <= free_bytes
                        and live_vram_admission_bound
                        + headroom
                        + minimum_residual_headroom_bytes
                        <= free_bytes
                    ),
                    "measurement_bootstrap_minimum_free_bytes": (
                        measurement_bootstrap_floor
                    ),
                    "projection_fits_total_memory_with_headroom": (
                        live_vram_admission_source == "provider_conservative_projection"
                        and type(conservative_vram_projection) is int
                        and conservative_vram_projection + measurement_bootstrap_floor
                        <= total_bytes
                    ),
                    "free_covers_measurement_bootstrap_floor": (
                        measurement_bootstrap_floor <= free_bytes
                    ),
                }
            )
        except (
            Exception
        ) as exc:  # detector boundary must become a receipt, not authority
            cuda_row["error"] = f"{type(exc).__name__}: {exc}"
            reasons.append(
                {
                    "code": "cuda_memory_unavailable",
                    "path": "execution.backend",
                    "message": str(exc),
                }
            )
        else:
            if (
                development_mode
                and live_vram_admission_source == "provider_conservative_projection"
                and type(vram_budget) is int
                and vram_budget > 0
                and cuda_row.get("projection_fits_total_memory_with_headroom") is False
                and cuda_row.get("budget_within_90_percent_total") is True
            ):
                # A conservative combined projection that cannot fit this device
                # even when empty is not a transient occupancy signal. QualPrep
                # admits against the declared budget so Launchpad can WAIT on
                # real free VRAM instead of polling a never-satisfiable bound.
                live_vram_admission_bound = vram_budget
                live_vram_admission_source = "qualprep_declared_budget_fallback"
                cuda_row["live_admission_bound_bytes"] = live_vram_admission_bound
                cuda_row["live_admission_bound_source"] = live_vram_admission_source
                headroom = cuda_row.get("required_headroom_bytes")
                free_bytes = cuda_row.get("free_bytes")
                cuda_row["free_covers_admission_bound_and_headroom"] = (
                    type(headroom) is int
                    and type(free_bytes) is int
                    and live_vram_admission_bound + headroom <= free_bytes
                    and live_vram_admission_bound
                    + headroom
                    + minimum_residual_headroom_bytes
                    <= free_bytes
                )
            if vram_budget is None:
                reasons.append(
                    {
                        "code": "vram_budget_missing",
                        "path": "execution.budgets.vram_bytes",
                        "message": "a positive VRAM budget is required",
                    }
                )
            elif not cuda_row["budget_within_90_percent_total"]:
                reasons.append(
                    {
                        "code": "vram_budget_exceeds_device_envelope",
                        "path": "execution.budgets.vram_bytes",
                        "message": "declared VRAM budget leaves less than 4% device headroom",
                    }
                )
            elif not cuda_row["free_covers_admission_bound_and_headroom"]:
                reasons.append(
                    {
                        "code": "insufficient_free_vram",
                        "path": "execution.budgets.vram_bytes",
                        "message": (
                            "free CUDA memory does not cover the current admission "
                            "bound plus 4% total-device headroom"
                        ),
                    }
                )

    qualification_key = _member(plan, "qualification_key")
    if not isinstance(qualification_key, Mapping):
        qualification_key = {}
    profile_row: dict[str, Any] = {
        "profile_id": profile_id,
        "qualification_status": None,
        "execution_authority_scope": (
            authority_scope
        ),
        "admitted_batch_range": qualification_key.get("admitted_batch_range"),
        "requested_physical_batch_cells": estimate.get("physical_batch_cells"),
    }
    if resolved_profile is None or resolved_profile.profile_id != profile_id:
        reasons.append(
            {
                "code": "profile_not_registered",
                "path": "resolved_profile",
                "message": f"runtime profile is not registered: {profile_id}",
            }
        )
    else:
        profile_row["qualification_status"] = resolved_profile.qualification_status

    planning_ok = not reasons
    execution_reasons: list[dict[str, str]] = []
    ready = planning_ok
    if resolved_profile is not None:
        execution_reasons.extend(
            production_execution_policy_reasons_v1(plan, resolved_profile)
        )
    if (
        resolved_profile is not None
        and resolved_profile.qualification_status != "qualified"
        and not development_mode
    ):
        execution_reasons.append(
            {
                "code": "profile_not_execution_qualified",
                "path": "resolved_profile.qualification_status",
                "message": (
                    f"profile {resolved_profile.profile_id} has qualification status "
                    f"{resolved_profile.qualification_status!r}; planning succeeds but execution authority is withheld"
                ),
            }
        )
    provider_row: dict[str, Any] = {
        "provider_id": getattr(provider, "provider_id", None),
        "registered": provider is not None,
        "resource_estimator_id": estimate.get("resource_estimator_id"),
    }
    route_row: dict[str, Any] = {
        "evaluated": False,
        "authority_state": None,
        "execution_route_sha256": None,
        "required_obligations": [],
        "covered_obligations": [],
        "missing_obligations": [],
        "qualification_receipt_path": None,
        "artifact_conformance_exact": None,
    }
    if provider is None:
        execution_reasons.append(
            {
                "code": "execution_provider_not_registered",
                "path": "resolved_profile",
                "message": f"no execution provider is registered for {profile_id}",
            }
        )
    else:
        seen: set[tuple[str, str, str]] = set()
        for request in translated_requests:
            for reason in provider.readiness_reasons(plan, request):
                key = (reason.code, reason.path, reason.message)
                if key in seen:
                    continue
                seen.add(key)
                execution_reasons.append(
                    {
                        "code": reason.code,
                        "path": reason.path,
                        "message": reason.message,
                    }
                )
        if seen:
            ready = False
        route_contract_owner = getattr(provider, "execution_route_contract", None)
        if not callable(route_contract_owner):
            execution_reasons.append(
                {
                    "code": "execution_route_contract_unavailable",
                    "path": "resolved_profile",
                    "message": (
                        f"provider {provider.provider_id!r} has no execution-route contract"
                    ),
                }
            )
        elif not seen:
            try:
                route_contract = route_contract_owner(plan)
                if not isinstance(route_contract, ExecutionRouteContractV1):
                    raise TypeError(
                        "provider execution-route contract is not ExecutionRouteContractV1"
                    )
                route_hash = route_contract.execution_route_sha256
                request_hashes = {
                    request.execution_route_sha256 for request in translated_requests
                }
                if request_hashes != {route_hash}:
                    raise ValueError(
                        "physical requests do not bind the provider-derived route hash"
                    )
                route_status, route_receipt_path, route_receipt = (
                    _route_qualification_status(plan, route_contract)
                )
                qualification_key_row = _member(plan, "qualification_key")
                expected_variability_envelope = (
                    qualification_key_row.get("variability_envelope", {})
                    if isinstance(qualification_key_row, Mapping)
                    else {}
                )
                received_variability_envelope = (
                    route_receipt.get("variability_envelope", {})
                    if isinstance(route_receipt, Mapping)
                    else {}
                )
                variability_envelope_exact = (
                    expected_variability_envelope == received_variability_envelope
                )
                artifact_rows_by_name = (
                    artifacts if isinstance(artifacts, Mapping) else {}
                )
                artifact_row = artifact_rows_by_name.get("cuda_extension")
                expected_artifact_sha256 = (
                    artifact_row.get("sha256")
                    if isinstance(artifact_row, Mapping)
                    else None
                )
                receipt_artifact_sha256 = (
                    route_receipt.get("artifact_sha256")
                    if isinstance(route_receipt, Mapping)
                    else None
                )
                artifact_exact = (
                    isinstance(expected_artifact_sha256, str)
                    and receipt_artifact_sha256 == expected_artifact_sha256
                )
                route_receipt_file = (
                    Path(route_receipt_path)
                    if isinstance(route_receipt_path, str)
                    else None
                )
                route_receipt_file_exact = bool(
                    route_receipt_file is not None
                    and route_receipt_file.is_file()
                    and not route_receipt_file.is_symlink()
                )
                if (
                    route_status["authority_state"] == "route_qualified"
                    and not route_receipt_file_exact
                ):
                    raise ValueError(
                        "route qualification sidecar is not a regular file"
                    )
                route_row = {
                    "evaluated": True,
                    **route_status,
                    "qualification_receipt_path": route_receipt_path,
                    "qualification_receipt_sha256": (
                        _sha256(route_receipt_file)
                        if route_receipt_file_exact
                        else None
                    ),
                    "artifact_conformance_exact": artifact_exact,
                    "variability_envelope_exact": variability_envelope_exact,
                }
                if (
                    expected_variability_envelope
                    and route_status["authority_state"] == "route_qualified"
                    and not variability_envelope_exact
                ):
                    execution_reasons.append(
                        {
                            "code": "wrc_variability_envelope_receipt_mismatch",
                            "path": "execution_route.variability_envelope",
                            "message": (
                                "route qualification receipt does not bind the "
                                "requested WRC variability envelope"
                            ),
                        }
                    )
                if route_status["authority_state"] != "route_qualified":
                    from engine.qualification_plans.v1 import (
                        generate_execution_route_canary_plan,
                    )

                    qualification_key_row = _member(plan, "qualification_key")
                    admitted = (
                        qualification_key_row.get("admitted_batch_range")
                        if isinstance(qualification_key_row, Mapping)
                        else None
                    )
                    native_batch_max = next(
                        (
                            row.get("batch_max")
                            for row in route_contract.components
                            if row.get("component_id") == "native_mechanics_binding"
                            and type(row.get("batch_max")) is int
                            and row["batch_max"] > 0
                        ),
                        None,
                    )
                    admitted_batch_sizes = (
                        list(
                            range(
                                admitted[0],
                                min(admitted[1], native_batch_max) + 1,
                            )
                        )
                        if isinstance(admitted, (list, tuple))
                        and len(admitted) == 2
                        and all(type(value) is int for value in admitted)
                        and isinstance(native_batch_max, int)
                        and admitted[0] <= min(admitted[1], native_batch_max)
                        else [1]
                    )
                    plan_execution = (
                        dict(plan.get("execution", {}))
                        if isinstance(plan, Mapping)
                        else {}
                    )
                    if (
                        plan_execution.get("execution_route_identity")
                        == "tick_native_execution_route_v1"
                    ):
                        native_control = plan_execution.get("tick_native_control")
                        canary_requirements = {
                            "time_authority": "absolute_tick_v1",
                            "total_ticks": plan_execution.get("total_ticks"),
                            "checkpoint_ticks": list(
                                plan_execution.get("checkpoint_ticks", [])
                            ),
                            "resident_owner_span_ticks": (
                                native_control.get("resident_owner_span_ticks")
                                if isinstance(native_control, Mapping)
                                else None
                            ),
                            "native_dispatch_window_ticks": (
                                native_control.get("native_dispatch_window_ticks")
                                if isinstance(native_control, Mapping)
                                else None
                            ),
                            "event_window_capacity": (
                                native_control.get("event_window_capacity")
                                if isinstance(native_control, Mapping)
                                else None
                            ),
                            "admitted_physical_batch_sizes": admitted_batch_sizes,
                        }
                    else:
                        ticks = (
                            _member(plan, "ticks_per_generation")
                            if not isinstance(plan, Mapping)
                            else plan_execution.get("ticks_per_generation")
                        )
                        canary_requirements = {
                            "qualified_ticks_per_generation": ticks,
                            "admitted_physical_batch_sizes": admitted_batch_sizes,
                            "boundary_generations": max(
                                2,
                                int(
                                    getattr(
                                        provider,
                                        "resident_segment_generation_min",
                                        2,
                                    )
                                ),
                            ),
                            "resident_segment_generation_min": max(
                                1,
                                int(
                                    getattr(
                                        provider,
                                        "resident_segment_generation_min",
                                        1,
                                    )
                                ),
                            ),
                        }
                    canary_contract = {
                        **route_contract.payload(),
                        "execution_route_sha256": route_hash,
                        "canary_requirements": canary_requirements,
                    }
                    canary_obligations = (
                        route_status["missing_obligations"]
                        or route_status["required_obligations"]
                    )
                    route_row["qualification_canary"] = (
                        generate_execution_route_canary_plan(
                            canary_contract,
                            missing_obligations=canary_obligations,
                        )
                    )
                    for obligation in route_status["missing_obligations"]:
                        execution_reasons.append(
                            {
                                "code": "execution_route_obligation_uncovered",
                                "path": (
                                    f"execution_route.required_obligations.{obligation}"
                                ),
                                "message": (
                                    f"execution route {route_hash} lacks qualification "
                                    f"evidence for {obligation}"
                                ),
                            }
                        )
                    if not route_status["missing_obligations"]:
                        for reason in route_status["reasons"]:
                            execution_reasons.append(
                                {
                                    "code": str(reason["code"]),
                                    "path": "execution_route",
                                    "message": str(reason["message"]),
                                }
                            )
                elif not artifact_exact:
                    execution_reasons.append(
                        {
                            "code": "execution_route_artifact_conformance_mismatch",
                            "path": "execution_route.artifact_sha256",
                            "message": (
                                "route qualification does not bind the selected "
                                "artifact instance"
                            ),
                        }
                    )
            except (OSError, QualificationReceiptError, TypeError, ValueError) as exc:
                execution_reasons.append(
                    {
                        "code": "execution_route_admission_failed",
                        "path": "execution_route",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
    if provider_estimate_error is not None:
        execution_reasons.append(
            {
                "code": "provider_resource_estimate_failed",
                "path": "resource_estimate",
                "message": provider_estimate_error,
            }
        )
    for estimate_name, budget_path in (
        ("vram_bytes_upper_bound", "execution.budgets.vram_bytes"),
        ("wall_seconds_upper_bound", "execution.budgets.wall_seconds"),
    ):
        estimate_value = estimate.get(estimate_name)
        if (
            isinstance(estimate_value, bool)
            or not isinstance(estimate_value, int)
            or estimate_value < 0
        ):
            ready = False
            unavailable_reason = estimate.get(
                estimate_name.replace("_upper_bound", "_unavailable_reason")
            )
            execution_reasons.append(
                {
                    "code": f"provider_{estimate_name}_unavailable",
                    "path": budget_path,
                    "message": str(unavailable_reason)
                    if unavailable_reason
                    else (
                        f"{estimate_name} is not available from the current execution "
                        "provider; resource-based execution admission is withheld"
                    ),
                }
            )

    try:
        qualification_reuse = qualification_reuse_receipt_v1(plan)
    except QualificationReceiptError as exc:
        qualification_reuse = {
            "reuses_bound_qualification": False,
            "error": str(exc),
        }
    if (
        isinstance(qualification_reuse, Mapping)
        and qualification_reuse.get("reuses_bound_qualification") is not True
    ):
        batch_admission = qualification_reuse.get("batch_admission")
        if isinstance(batch_admission, Mapping):
            admitted = batch_admission.get("admitted_range")
            requested = batch_admission.get("requested_physical_batch_cells")
            execution_reasons.append(
                {
                    "code": "execution_qualification_reuse_not_admitted",
                    "path": "qualification_reuse.batch_admission",
                    "message": (
                        "requested physical batch "
                        f"{requested} is outside the bound admitted range "
                        f"{admitted}"
                    ),
                }
            )
        else:
            execution_reasons.append(
                {
                    "code": "execution_qualification_reuse_not_admitted",
                    "path": "qualification_reuse",
                    "message": "campaign does not reuse its bound qualification",
                }
            )

    prepared_owner_supported = bool(
        provider is not None
        and callable(
            getattr(
                provider,
                "measure_prepared_owner_resources_v1",
                None,
            )
        )
    )
    prepared_owner_support_reason: dict[str, str] | None = None
    support_probe = getattr(provider, "prepared_owner_measurement_support_v1", None)
    if prepared_owner_supported and callable(support_probe):
        support = support_probe(_plan_payload_for_authorization(plan))
        if not isinstance(support, Mapping) or support.get("supported") is not True:
            prepared_owner_supported = False
            prepared_owner_support_reason = {
                "code": str(
                    support.get("code")
                    if isinstance(support, Mapping)
                    else "prepared_owner_measurement_unsupported"
                ),
                "path": "execution_route.mechanics_binding",
                "message": str(
                    support.get("message")
                    if isinstance(support, Mapping)
                    else "provider declined prepared-owner measurement"
                ),
            }
    measurement_authorization: dict[str, Any] | None = None
    measurement_authority_reasons: list[dict[str, str]] = []
    receipt_promotion: dict[str, Any] | None = None
    unavailable_codes = {
        row.get("code") for row in execution_reasons if isinstance(row, Mapping)
    }
    exact_vram_unavailable_only = unavailable_codes == {
        "provider_vram_bytes_upper_bound_unavailable"
    }
    measurement_tolerated_reason_codes: set[str] = set()
    if cuda_row.get("free_covers_measurement_bootstrap_floor") is True:
        measurement_tolerated_reason_codes.update(
            {
                "insufficient_free_vram",
                "vram_budget_exceeds_device_envelope",
            }
        )
    measurement_planning_reasons = [
        row
        for row in reasons
        if not (
            isinstance(row, Mapping)
            and row.get("code") in measurement_tolerated_reason_codes
        )
    ]
    projection = estimate.get("vram_bytes_projection")
    estimator_id = estimate.get("resource_estimator_id")
    if not prepared_owner_supported:
        measurement_authority_reasons.append(
            prepared_owner_support_reason
            or {
                "code": "prepared_owner_measurement_unsupported",
                "path": "resolved_profile",
                "message": "selected provider has no prepared-owner measurement surface",
            }
        )
    elif measurement_planning_reasons or not exact_vram_unavailable_only:
        measurement_authority_reasons.append(
            {
                "code": "prepared_owner_measurement_prerequisites_failed",
                "path": "preflight",
                "message": (
                    "prepared-owner measurement tolerates only the exact "
                    "provider VRAM upper-bound-unavailable execution blocker"
                ),
            }
        )
    elif len(translated_requests) != 1:
        measurement_authority_reasons.append(
            {
                "code": "prepared_owner_measurement_batch_ambiguous",
                "path": "physical_batches",
                "message": "prepared-owner bootstrap requires exactly one physical batch",
            }
        )
    elif (
        type(projection) is not int
        or projection <= 0
        or not isinstance(estimator_id, str)
        or not estimator_id
        or vram_budget is None
    ):
        measurement_authority_reasons.append(
            {
                "code": "prepared_owner_conservative_projection_invalid",
                "path": "resource_estimate.vram_bytes_projection",
                "message": (
                    "a positive conservative projection and declared VRAM "
                    "budget are required"
                ),
            }
        )
    elif not runtime_environment_ok:
        measurement_authority_reasons.append(
            {
                "code": "prepared_owner_runtime_environment_mismatch",
                "path": "resource_estimate.runtime_environment_requirements",
                "message": (
                    "prepared-owner measurement requires exact runtime "
                    "environment assumptions"
                ),
            }
        )
    elif (
        type(cuda_row.get("free_bytes")) is not int
        or type(cuda_row.get("total_bytes")) is not int
        or type(cuda_row.get("required_headroom_bytes")) is not int
        or cuda_row.get("free_covers_measurement_bootstrap_floor") is not True
    ):
        measurement_authority_reasons.append(
            {
                "code": "prepared_owner_projection_lacks_cuda_headroom",
                "path": "execution.budgets.vram_bytes",
                "message": (
                    "live CUDA memory must cover the physical-headroom and route "
                    "residual measurement bootstrap floor"
                ),
            }
        )
    else:
        request = translated_requests[0]
        try:
            hardware_identity = validate_cuda_hardware_identity_v1(
                cuda_hardware_detector()
            )
            if hardware_identity.get("total_memory_bytes") != cuda_row["total_bytes"]:
                raise ValueError(
                    "CUDA hardware identity total differs from memory telemetry"
                )
            request_payload = request.payload()
            parameters = {name: values for name, values in request.parameter_arrays}

            def uniform_integer(name: str) -> int:
                values = parameters.get(name)
                if (
                    not isinstance(values, tuple)
                    or not values
                    or any(type(value) is not int or value < 1 for value in values)
                    or len(set(values)) != 1
                ):
                    raise ValueError(
                        f"prepared-owner measurement requires one exact {name}"
                    )
                return int(values[0])

            execution = _plan_payload_for_authorization(plan).get("execution")
            tick_native = request.total_ticks is not None
            native_segments = (
                provider._absolute_tick_segments_v1(request, start_next_tick=0)
                if tick_native
                else ()
            )
            segment_ends = (
                native_segments
                if tick_native
                else provider._segment_ends(request, start=0)
            )
            if not segment_ends:
                raise ValueError("prepared-owner measurement has no resident segment")
            allocation_capacity = provider._agent_id_allocation_generation_count(
                request, start_generation=0
            )
            artifact_sha256s = {
                str(name): str(row.get("sha256"))
                for name, row in artifacts.items()
                if isinstance(row, Mapping)
            }
            request_bindings = {
                "provider_id": str(provider.provider_id),
                "profile_id": profile_id,
                "campaign_id": str(_member(plan, "campaign_id")),
                "plan_sha256": str(_member(plan, "plan_sha256")),
                "qualification_key_sha256": str(
                    _member(plan, "qualification_key_sha256")
                ),
                "execution_route_sha256": str(route_row["execution_route_sha256"]),
                "route_qualification": {
                    "authority_state": route_row["authority_state"],
                    "receipt_path": route_row["qualification_receipt_path"],
                    "receipt_sha256": route_row["qualification_receipt_sha256"],
                    "artifact_conformance_exact": route_row[
                        "artifact_conformance_exact"
                    ],
                },
                "artifact_sha256s": artifact_sha256s,
                "configuration_sha256": prepared_owner_configuration_sha256_v1(
                    _plan_payload_for_authorization(plan),
                    request_payload,
                ),
                "resource_estimator_id": estimator_id,
                "conservative_vram_projection_bytes": projection,
                "runtime_assumptions": runtime_environment_requirements,
                "hardware_identity": hardware_identity,
                "ordered_cell_ids_sha256": canonical_sha256(list(request.cell_ids)),
                "batch_request_sha256": canonical_sha256(request_payload),
                "dimensions": (
                    {
                        "batch_cells": len(request.cell_ids),
                        "time_authority": "absolute_tick_v1",
                        "total_ticks": request.total_ticks,
                        "resident_owner_span_ticks": request.resident_owner_span_ticks,
                        "native_dispatch_window_ticks": request.native_dispatch_window_ticks,
                        "resident_segment_ticks": int(
                            segment_ends[0]["absolute_tick_span"]
                        ),
                        "event_window_capacity": request.event_window_capacity,
                        "source_window_ticks": request.source_window_ticks,
                        "resident_capacity": uniform_integer("patch.capacity"),
                        "initial_population": uniform_integer("patch.initial_population"),
                        "agent_id_allocation_capacity": allocation_capacity,
                    }
                    if tick_native
                    else {
                        "batch_cells": len(request.cell_ids),
                        "resident_segment_generations": int(segment_ends[0]),
                        "physical_window_generations": (
                            execution.get("physical_window_generations")
                            if isinstance(execution, Mapping)
                            and execution.get("physical_window_generations") is not None
                            else 2
                        ),
                        "ticks_per_generation": request.ticks_per_generation,
                        "resident_capacity": uniform_integer("patch.capacity"),
                        "initial_population": uniform_integer("patch.initial_population"),
                        "agent_id_allocation_generation_count": allocation_capacity,
                    }
                ),
                "policy": {
                    "declared_vram_budget_bytes": vram_budget,
                    "minimum_physical_headroom_fraction": CUDA_HEADROOM_FRACTION,
                    "minimum_physical_headroom_bytes": 0,
                    "minimum_residual_headroom_bytes": (
                        minimum_residual_headroom_bytes
                    ),
                    "measurement_live_free_policy": {
                        "mode": "exact_allocation_only_measurement_bootstrap_v2",
                        "minimum_free_bytes": cuda_row[
                            "measurement_bootstrap_minimum_free_bytes"
                        ],
                        "requires_projection_within_total_memory": False,
                        "allows_projection_over_declared_budget": True,
                        "zero_dispatch_only": True,
                        "production_execution_admission_unchanged": True,
                    },
                    "teardown_tolerance_bytes": int(
                        getattr(
                            provider,
                            "prepared_owner_teardown_tolerance_bytes",
                            0,
                        )
                    ),
                },
            }
            unsigned_authorization = {
                "schema_version": MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION,
                "request_bindings": request_bindings,
            }
            measurement_authorization = {
                **unsigned_authorization,
                "authorization_sha256": canonical_sha256(unsigned_authorization),
            }
        except Exception as exc:  # hardware and provider boundaries fail closed
            measurement_authority_reasons.append(
                {
                    "code": "prepared_owner_measurement_authorization_failed",
                    "path": "prepared_owner_resource_audit",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    if prepared_owner_resource_receipt is not None:
        if measurement_authorization is None:
            execution_reasons.append(
                {
                    "code": "prepared_owner_resource_receipt_not_admissible",
                    "path": "prepared_owner_resource_receipt",
                    "message": (
                        "receipt promotion requires a current exact measurement "
                        "authorization"
                    ),
                }
            )
        else:
            try:
                expected_bindings = {
                    **measurement_authorization["request_bindings"],
                    "measurement_authorization_sha256": (
                        measurement_authorization["authorization_sha256"]
                    ),
                }
                receipt_promotion = validate_prepared_owner_resource_audit_receipt_v1(
                    prepared_owner_resource_receipt,
                    expected_request_bindings=expected_bindings,
                    live_hardware_identity=measurement_authorization[
                        "request_bindings"
                    ]["hardware_identity"],
                )
            except (TypeError, ValueError) as exc:
                execution_reasons.append(
                    {
                        "code": "prepared_owner_resource_receipt_invalid",
                        "path": "prepared_owner_resource_receipt",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                execution_reasons = [
                    row
                    for row in execution_reasons
                    if row.get("code") != "provider_vram_bytes_upper_bound_unavailable"
                ]
                vram_estimate = receipt_promotion["vram_bytes_upper_bound"]
                vram_estimate_within_budget = (
                    vram_budget is not None and vram_estimate <= vram_budget
                )
                cuda_row["provider_estimate_within_budget"] = (
                    vram_estimate_within_budget
                )

    development_deferred_authority_reasons: list[dict[str, str]] = []
    if development_mode and route_row.get("authority_state") in {
        "legacy_unbound",
        "qualification_only",
    }:
        # A first real QualPrep rehearsal must be able to produce evidence for
        # an opt-in experimental or reminted-unqualified route without falsely
        # promoting that route to qualified execution.  Defer only uncovered
        # route obligations; stale, mismatched, malformed, or otherwise unsafe
        # route evidence still fails closed.
        deferred_codes = {
            "execution_route_obligation_uncovered",
            "provider_vram_bytes_upper_bound_unavailable",
        }
        development_deferred_authority_reasons = [
            row
            for row in execution_reasons
            if row.get("code") in deferred_codes
        ]
        execution_reasons = [
            row
            for row in execution_reasons
            if row.get("code") not in deferred_codes
        ]
        route_row[f"{authority_scope}_route_authorized"] = bool(
            development_deferred_authority_reasons
        )

    admission_rejection_codes = {
        "disk_budget_exceeded",
        "vram_budget_exceeds_device_envelope",
        "insufficient_free_vram",
    }
    reason_codes = {str(row.get("code")) for row in reasons}
    if "cuda_memory_unavailable" in reason_codes:
        admission_status = "RESOURCE_ADMISSION_UNAVAILABLE"
    elif reason_codes & admission_rejection_codes:
        admission_status = "RESOURCE_ADMISSION_REJECTED"
    else:
        admission_status = "RESOURCE_ADMISSION_NOT_REJECTED"

    ready = planning_ok and not execution_reasons

    from engine.perf.cuda_architecture_b_hapb_v1 import (
        hapb_production_route_admission_v1,
    )

    hapb_production_route = hapb_production_route_admission_v1()

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "campaign_id": _member(plan, "campaign_id"),
        "plan_sha256": _member(plan, "plan_sha256"),
        "artifact_lifecycle": lifecycle,
        "qualification_key_sha256": _member(plan, "qualification_key_sha256"),
        "qualification_reuse": qualification_reuse,
        "ok": planning_ok,
        "ready_for_execution": ready,
        "execution_authority_scope": (
            authority_scope
        ),
        "ready_for_prepared_owner_measurement": (
            measurement_authorization is not None and not measurement_authority_reasons
        ),
        "reasons": reasons,
        "execution_authority_reasons": execution_reasons,
        "qualprep_deferred_authority_reasons": (
            development_deferred_authority_reasons if qualprep_mode else []
        ),
        "engineering_deferred_authority_reasons": (
            development_deferred_authority_reasons if engineering_mode else []
        ),
        "artifacts": artifact_rows,
        "resources": {
            # Admission is an instantaneous resource decision.  Preflight is
            # read-only with respect to route qualification receipts, so this
            # status can never turn into REQUALIFICATION_REQUIRED by itself.
            "admission": {
                "status": admission_status,
                "qualification_receipt_action": "unchanged",
            },
            "disk_bytes_upper_bound": disk_estimate,
            "declared_disk_budget_bytes": disk_budget,
            "disk_budget_ok": disk_ok,
            "vram_bytes_upper_bound": vram_estimate,
            "wall_seconds_upper_bound": estimate.get("wall_seconds_upper_bound"),
            "declared_wall_budget_seconds": wall_budget,
            "wall_budget_ok": wall_ok,
            "provider_estimate": {
                key: estimate[key]
                for key in (
                    "resource_estimator_id",
                    "resource_estimator_provenance",
                    "resource_estimator_assumptions",
                    "resource_dimensions_schema_version",
                    "physical_window_route_identity",
                    "physical_storage_generations",
                    "per_cell_vram_bytes_upper_bound",
                    "vram_bytes_upper_bound",
                    "vram_bytes_projection",
                    "vram_bytes_unavailable_reason",
                    "wall_seconds_projection",
                    "wall_seconds_bound_authority",
                    "wall_seconds_unavailable_reason",
                    "peak_resident_gpu_owner_event_bytes",
                    "peak_host_checkpoint_import_scratch_bytes",
                    "total_durable_output_checkpoint_bytes_upper_bound",
                    "resident_gpu_owner_event_bytes_horizon_invariant",
                    "host_checkpoint_import_scratch_bytes_horizon_invariant",
                    "resident_event_rows_per_cell",
                    "resident_event_row_bytes",
                    "resident_event_bytes",
                    "amended_b136_capacity_profile",
                )
                if key in estimate
            },
            "physical_b136_shape": physical_b136_shape,
            "cuda": cuda_row,
            "allocation_diagnostic_resource_promotion": allocation_diagnostic_promotion,
            "storage": storage_admission,
            "hapb_production_route": hapb_production_route,
        },
        "profile": profile_row,
        "execution_provider": provider_row,
        "execution_route": route_row,
        "execution_requests": {
            bound.resident_request.batch_id: {
                "source_batch_id": bound.source_batch_id,
                "bound_request_sha256": bound.bound_request_sha256,
                "execution_route_sha256": bound.execution_route_sha256,
            }
            for bound in bound_requests
        },
        "prepared_owner_resource_audit": {
            "supported": prepared_owner_supported,
            "provider_id": (None if provider is None else str(provider.provider_id)),
            "profile_id": profile_id,
            "batch_ids": [value for value in batch_ids if isinstance(value, str)],
            "evidence_scope": (
                "structural_prepare_full_owner_zero_dispatch_counter_"
                "zero_tick_counter_v1"
            ),
            "measurement_authority_reasons": measurement_authority_reasons,
            "measurement_authorization": measurement_authorization,
            "receipt_promotion": receipt_promotion,
        },
    }


__all__ = [
    "CUDA_HEADROOM_FRACTION",
    "PREFLIGHT_SCHEMA_VERSION",
    "detect_cuda_memory_v1",
    "preflight_campaign",
]
