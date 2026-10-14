#!/usr/bin/env python3
"""Run one bounded, diagnostic-only production-route tick window.

The harness reuses a frozen production execution plan and CUDA artifact, but
binds them to explicit non-admission descriptors.  It prepares the same
resident WDL/Jungle/EED owner and executes an exact partial tick span without
checkpoint projection, scientific reporting, or qualification claims.

An NVTX range encloses only the native tick span.  B48 is authoritative for the
anatomy; B12/B24 are diagnostic scaling controls.  Source/owner construction
is excluded from per-tick measurements.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SOURCE_ROOT = Path(
    os.environ.get("EIEL_GPU_TICK_RUNTIME_SOURCE_ROOT", str(ROOT))
).resolve()
ENGINE_ROOT = RUNTIME_SOURCE_ROOT / "engine"
for import_root in (ENGINE_ROOT, RUNTIME_SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (  # noqa: E402
    _wdl_source_capabilities_v2,
)


def _load_diagnostic_helper() -> Any:
    """Load audit mechanics without letting the sealed payload shadow them."""

    path = (
        ROOT / "tools/scripts/eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1.py"
    )
    spec = importlib.util.spec_from_file_location("_eiel_gpu_tick_audit_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("GPU tick audit helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if RUNTIME_SOURCE_ROOT == ROOT:
    from tools.scripts import (  # noqa: E402
        eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1 as _DIAGNOSTIC_HELPER,
    )
else:
    _DIAGNOSTIC_HELPER = _load_diagnostic_helper()
AllocationDiagnosticError = _DIAGNOSTIC_HELPER.AllocationDiagnosticError
AllocationOnlyProvider = _DIAGNOSTIC_HELPER.AllocationOnlyProvider
_diagnostic_cells_plan = _DIAGNOSTIC_HELPER._diagnostic_cells_plan
_diagnostic_descriptor = _DIAGNOSTIC_HELPER._diagnostic_descriptor
_diagnostic_plan = _DIAGNOSTIC_HELPER._diagnostic_plan
_load_exact_module = _DIAGNOSTIC_HELPER._load_exact_module
_sha256 = _DIAGNOSTIC_HELPER._sha256


SCHEMA_VERSION = "eiel_gpu_tick_anatomy_v1"
AUTHORITY = "diagnostic_profiling_only_not_science_or_qualification_v1"
NVTX_RANGE = "gpu_tick_anatomy_measure"
EXPECTED_CELLS = 48
EXPECTED_CAPACITY = 512
EXPECTED_TICKS_PER_GENERATION = 160
PRODUCTION_HOST_FILES = (
    "engine/campaigns/providers.py",
    "engine/campaigns/production_engine_native_bindings.py",
    "engine/perf/cuda_architecture_b_direct_resident_wdl_jungle_v1.py",
)
M7_MANIFEST_FIELD = "option_d_m7_batched_cpu_policy_refill_build"
M7_ENABLED_FIELD = "option_d_m7_batched_cpu_policy_refill_enabled"
M7_USED_FIELD = "option_d_m7_batched_cpu_policy_refill_used"
M7_AUTHORITY_FIELD = "option_d_m7_batched_cpu_policy_refill_authority"
M7_SCHEMA_FIELD = "option_d_m7_batched_cpu_policy_refill_schema_version"
M7_BUILD_SCHEMA = "eiel_option_d_m7_batched_cpu_policy_refill_build_identity_v1"
M7_RUNTIME_SCHEMA = "eiel_option_d_m7_batched_cpu_policy_refill_v1"
M8_MANIFEST_FIELD = "option_d_m8_resident_policy_suffix_reuse_build"
M8_ENABLED_FIELD = "option_d_m8_resident_policy_suffix_reuse_enabled"
M8_USED_FIELD = "option_d_m8_resident_policy_suffix_reuse_used"
M8_AUTHORITY_FIELD = "option_d_m8_resident_policy_suffix_reuse_authority"
M8_SCHEMA_FIELD = "option_d_m8_resident_policy_suffix_reuse_schema_version"
M8_BUILD_SCHEMA = "eiel_option_d_m8_resident_policy_suffix_reuse_build_identity_v1"
M8_RUNTIME_SCHEMA = "eiel_option_d_m8_resident_policy_suffix_reuse_v1"
BENCHMARK_ARM_ORDER = ("legacy", "m7_physical_refill", "m8_suffix_reuse")
WARMUP_IDENTITY = "source_and_resident_owner_preparation_no_native_ticks_v1"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _process_identity() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "python": str(Path(sys.executable).resolve()),
        "cwd": str(Path.cwd().resolve()),
    }


def _gpu_compute_processes() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise AllocationDiagnosticError(
            "nvidia-smi compute-process preflight is unavailable"
        )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not fields[0]:
            continue
        try:
            pid = int(fields[0])
            used_mib = int(fields[2])
        except ValueError as exc:
            raise AllocationDiagnosticError(
                "nvidia-smi returned a malformed compute-process row"
            ) from exc
        rows.append({"pid": pid, "process_name": fields[1], "used_mib": used_mib})
    return rows


def _competing_gpu_processes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject only likely simulation/profiler writers, not desktop clients."""

    own_pid = os.getpid()
    competing: list[dict[str, Any]] = []
    for row in rows:
        if row["pid"] == own_pid:
            continue
        name = str(row["process_name"]).lower()
        if "python" in name or "qrr" in name or "nsys" in name or "ncu" in name:
            competing.append(row)
    return competing


def _validated_plan(plan_path: Path) -> dict[str, Any]:
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise AllocationDiagnosticError("execution plan is not a mapping")
    plan = dict(raw)
    batches = plan.get("physical_batches")
    execution = plan.get("execution")
    artifacts = plan.get("artifacts")
    if (
        not isinstance(batches, list)
        or len(batches) != 1
        or not isinstance(batches[0], Mapping)
        or len(batches[0].get("cell_ids", ())) != EXPECTED_CELLS
    ):
        raise AllocationDiagnosticError(
            "profiling plan must contain one exact B48 batch"
        )
    if not isinstance(execution, Mapping):
        raise AllocationDiagnosticError("profiling plan lacks execution identity")
    if (
        plan.get("resolved_profile") != "eiel-production-runtime-v1"
        or execution.get("execution_route_identity") != "tick_native_execution_route_v1"
        or execution.get("internal_lifecycle_schedule", {}).get("interval_ticks")
        != EXPECTED_TICKS_PER_GENERATION
    ):
        raise AllocationDiagnosticError(
            "profiling plan is not the production tick-native CUDA route"
        )
    if not isinstance(artifacts, Mapping):
        raise AllocationDiagnosticError("profiling plan lacks artifact bindings")
    cuda_row = artifacts.get("cuda_extension")
    manifest_row = artifacts.get("extension_manifest")
    if not isinstance(cuda_row, Mapping) or not isinstance(manifest_row, Mapping):
        raise AllocationDiagnosticError("profiling plan lacks CUDA artifact identity")
    if cuda_row.get("abi") != "architecture_b_wdl_jungle_eed_fixed_mlp_native_v1":
        raise AllocationDiagnosticError("profiling plan selects a different native ABI")
    observed_plan_sha256 = plan.get("plan_sha256")
    plan_payload = {
        name: value for name, value in plan.items() if name != "plan_sha256"
    }
    if observed_plan_sha256 != canonical_sha256(plan_payload):
        raise AllocationDiagnosticError("profiling plan identity digest mismatch")
    return plan


def _bind_artifacts(
    plan: Mapping[str, Any],
    *,
    artifact: Path,
    manifest: Path,
    allow_diagnostic_artifact_rebind: bool = False,
    allow_candidate_artifact_rebind: bool = False,
    expected_m7_enabled: bool,
    expected_m8_enabled: bool = False,
) -> dict[str, Any]:
    if type(expected_m7_enabled) is not bool:
        raise TypeError("expected_m7_enabled must be a bool")
    if type(expected_m8_enabled) is not bool:
        raise TypeError("expected_m8_enabled must be a bool")
    if expected_m8_enabled and not expected_m7_enabled:
        raise AllocationDiagnosticError("M8 suffix reuse requires the M7 fallback")
    if allow_diagnostic_artifact_rebind and allow_candidate_artifact_rebind:
        raise AllocationDiagnosticError("artifact rebind modes are mutually exclusive")
    artifacts = {
        name: dict(row)
        for name, row in dict(plan["artifacts"]).items()
        if isinstance(row, Mapping)
    }
    artifact_sha256 = _sha256(artifact) if artifact.is_file() else None
    manifest_sha256 = _sha256(manifest) if manifest.is_file() else None
    allow_artifact_rebind = (
        allow_diagnostic_artifact_rebind or allow_candidate_artifact_rebind
    )
    manifest_payload: Any = None
    if manifest.is_file():
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if allow_artifact_rebind:
        if artifact_sha256 is None or manifest_sha256 is None:
            raise AllocationDiagnosticError(
                "diagnostic artifact rebind inputs are absent"
            )
        if (
            not isinstance(manifest_payload, Mapping)
            or manifest_payload.get("module_name")
            != "eiel_cuda_architecture_b_wdl_jungle_eed_v1"
            or manifest_payload.get("artifact_sha256") != artifact_sha256
            or manifest_payload.get("artifact_identity", {}).get("sha256")
            != artifact_sha256
            or bool(
                manifest_payload.get("option_d_m0_telemetry_build", {}).get(
                    "option_d_m0_telemetry_enabled"
                )
            )
            is not allow_diagnostic_artifact_rebind
        ):
            raise AllocationDiagnosticError(
                "diagnostic artifact is not hash-bound by its manifest or has the "
                "wrong telemetry mode"
            )
    _m7_manifest_identity(manifest_payload, expected_enabled=expected_m7_enabled)
    _m8_manifest_identity(manifest_payload, expected_enabled=expected_m8_enabled)
    for name, path in (("cuda_extension", artifact), ("extension_manifest", manifest)):
        if not path.is_file():
            raise AllocationDiagnosticError(f"{name} is absent: {path}")
        expected = artifacts[name].get("sha256")
        observed = _sha256(path)
        if observed != expected and not allow_artifact_rebind:
            raise AllocationDiagnosticError(f"{name} hash differs from the frozen plan")
        artifacts[name]["path"] = str(path)
        artifacts[name]["sha256"] = observed
    result = dict(plan)
    result["artifacts"] = artifacts
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _m7_manifest_identity(
    manifest_payload: Any, *, expected_enabled: bool
) -> dict[str, Any]:
    """Validate the build-time M7 arm identity without fallback or inference."""

    if type(expected_enabled) is not bool:
        raise TypeError("expected_enabled must be a bool")
    if not isinstance(manifest_payload, Mapping):
        raise AllocationDiagnosticError("artifact manifest is not a mapping")
    raw = manifest_payload.get(M7_MANIFEST_FIELD)
    if not isinstance(raw, Mapping):
        raise AllocationDiagnosticError(
            "artifact manifest lacks exact M7 build identity"
        )
    row = dict(raw)
    expected_authority = (
        "candidate_batched_exact_cpu_lsb0_policy_refill"
        if expected_enabled
        else "disabled"
    )
    expected = {
        "schema_version": M7_BUILD_SCHEMA,
        "compile_time_macro": "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED",
        "receipt_field": M7_ENABLED_FIELD,
        "default_enabled": False,
        M7_ENABLED_FIELD: expected_enabled,
        "compile_definition": (
            "-DEIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED="
            f"{1 if expected_enabled else 0}"
        ),
        "authority": expected_authority,
        "identity_authority": "compiled_binary_resource_receipt",
        "runtime_resource_receipt_confirmed": True,
    }
    mismatches = {
        name: {"expected": value, "observed": row.get(name)}
        for name, value in expected.items()
        if row.get(name) != value
    }
    if mismatches:
        raise AllocationDiagnosticError(
            "artifact manifest M7 arm identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    identity_sha256 = row.get("identity_sha256")
    payload = {name: value for name, value in row.items() if name != "identity_sha256"}
    if identity_sha256 != canonical_sha256(payload):
        raise AllocationDiagnosticError("artifact manifest M7 identity digest mismatch")
    return row


def _m7_runtime_identity(
    result: Mapping[str, Any],
    *,
    expected_enabled: bool,
    expected_used: bool | None = None,
) -> dict[str, Any]:
    """Require the native result to prove that the selected M7 arm executed."""

    if expected_used is None:
        expected_used = expected_enabled
    if expected_used and not expected_enabled:
        raise AllocationDiagnosticError("disabled M7 cannot be used")
    expected = {
        M7_ENABLED_FIELD: expected_enabled,
        M7_USED_FIELD: expected_used,
        M7_AUTHORITY_FIELD: (
            "candidate_batched_exact_cpu_compact_refill"
            if expected_used
            else (
                "compiled_candidate_not_used_for_surface"
                if expected_enabled
                else "disabled"
            )
        ),
        M7_SCHEMA_FIELD: M7_RUNTIME_SCHEMA if expected_used else None,
    }
    mismatches = {
        name: {"expected": value, "observed": result.get(name)}
        for name, value in expected.items()
        if result.get(name) != value
    }
    if mismatches:
        raise AllocationDiagnosticError(
            "native result M7 arm identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {name: result.get(name) for name in expected}


def _m8_manifest_identity(
    manifest_payload: Any, *, expected_enabled: bool
) -> dict[str, Any]:
    """Validate the build-time M8 identity without inference."""

    if type(expected_enabled) is not bool:
        raise TypeError("expected_enabled must be a bool")
    if not isinstance(manifest_payload, Mapping):
        raise AllocationDiagnosticError("artifact manifest is not a mapping")
    raw = manifest_payload.get(M8_MANIFEST_FIELD)
    if not isinstance(raw, Mapping):
        raise AllocationDiagnosticError(
            "artifact manifest lacks exact M8 build identity"
        )
    row = dict(raw)
    expected = {
        "schema_version": M8_BUILD_SCHEMA,
        "compile_time_macro": "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED",
        "receipt_field": M8_ENABLED_FIELD,
        "default_enabled": False,
        M8_ENABLED_FIELD: expected_enabled,
        "compile_definition": (
            "-DEIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED="
            f"{1 if expected_enabled else 0}"
        ),
        "authority": "candidate_resident_policy_suffix_reuse"
        if expected_enabled
        else "disabled",
        "identity_authority": "compiled_binary_resource_receipt",
        "runtime_resource_receipt_confirmed": True,
    }
    mismatches = {
        name: {"expected": value, "observed": row.get(name)}
        for name, value in expected.items()
        if row.get(name) != value
    }
    if mismatches:
        raise AllocationDiagnosticError(
            "artifact manifest M8 arm identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    identity_sha256 = row.get("identity_sha256")
    payload = {name: value for name, value in row.items() if name != "identity_sha256"}
    if identity_sha256 != canonical_sha256(payload):
        raise AllocationDiagnosticError("artifact manifest M8 identity digest mismatch")
    return row


def _m8_runtime_identity(
    result: Mapping[str, Any], *, expected_enabled: bool
) -> dict[str, Any]:
    """Validate one M8 ensure receipt and its physical-refill fallback layering."""

    expected_disabled = {
        M8_ENABLED_FIELD: False,
        M8_USED_FIELD: False,
        M8_AUTHORITY_FIELD: "disabled",
        M8_SCHEMA_FIELD: None,
        "resident_window_reused": False,
        "physical_refill_performed": True,
    }
    if not expected_enabled:
        mismatches = {
            name: {"expected": value, "observed": result.get(name)}
            for name, value in expected_disabled.items()
            if result.get(name) != value
        }
        if mismatches:
            raise AllocationDiagnosticError(
                "native result M8 arm identity mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        return {name: result.get(name) for name in expected_disabled}

    reused = result.get("resident_window_reused")
    refilled = result.get("physical_refill_performed")
    if type(reused) is not bool or type(refilled) is not bool or reused == refilled:
        raise AllocationDiagnosticError(
            "native M8 receipt must prove exactly one of suffix reuse or physical refill"
        )
    expected = {
        M8_ENABLED_FIELD: True,
        M8_USED_FIELD: reused,
        M8_AUTHORITY_FIELD: (
            "candidate_resident_policy_suffix_reuse"
            if reused
            else "compiled_candidate_physical_refill_fallback"
        ),
        M8_SCHEMA_FIELD: M8_RUNTIME_SCHEMA,
    }
    mismatches = {
        name: {"expected": value, "observed": result.get(name)}
        for name, value in expected.items()
        if result.get(name) != value
    }
    if mismatches:
        raise AllocationDiagnosticError(
            "native result M8 arm identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        **{name: result.get(name) for name in expected},
        "resident_window_reused": reused,
        "physical_refill_performed": refilled,
    }


def _translation_compatible_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Restore compile-shape fields consumed by the production translator.

    QRR's sealed tick-native plan intentionally replaces these fields with an
    absolute-tick request.  The production translator still consumes the
    compile-shape fields before the tick-span adapter takes ownership.
    """

    result = dict(plan)
    execution = dict(result["execution"])
    execution.update(
        {
            "generations": 1,
            "ticks_per_generation": EXPECTED_TICKS_PER_GENERATION,
            "checkpoints": [1],
        }
    )
    result["execution"] = execution
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _release_host_identity(release_root: Path) -> list[dict[str, Any]]:
    payload = release_root.resolve() / "payload" / "python"
    rows: list[dict[str, Any]] = []
    for relative in PRODUCTION_HOST_FILES:
        working_path = RUNTIME_SOURCE_ROOT / relative
        release_path = payload / relative
        if not working_path.is_file() or not release_path.is_file():
            raise AllocationDiagnosticError(
                f"production release host source is absent: {relative}"
            )
        working_sha256 = _sha256(working_path)
        release_sha256 = _sha256(release_path)
        if working_sha256 != release_sha256:
            raise AllocationDiagnosticError(
                f"working host route differs from sealed release: {relative}"
            )
        rows.append(
            {
                "relative_path": relative,
                "working_path": str(working_path),
                "release_path": str(release_path),
                "sha256": working_sha256,
            }
        )
    return rows


def _benchmark_arm(*, m7_enabled: bool, m8_enabled: bool) -> str:
    arms = {
        (False, False): "legacy",
        (True, False): "m7_physical_refill",
        (True, True): "m8_suffix_reuse",
    }
    try:
        return arms[(m7_enabled, m8_enabled)]
    except KeyError as exc:
        raise AllocationDiagnosticError("M8 suffix reuse requires M7 fallback") from exc


def _hardware_identity(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if completed.returncode != 0 or len(fields) != 2 or not all(fields):
        raise AllocationDiagnosticError("exact GPU hardware identity is unavailable")
    identity = {
        "device_index": 0,
        "device_count": int(torch.cuda.device_count()),
        "device_name": str(properties.name),
        "device_capability": [int(properties.major), int(properties.minor)],
        "device_total_memory_bytes": int(properties.total_memory),
        "device_multiprocessor_count": int(properties.multi_processor_count),
        "gpu_uuid": fields[0],
        "driver_version": fields[1],
        "torch_version": str(torch.__version__),
        "torch_cuda_runtime_version": str(torch.version.cuda),
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _comparison_identity(
    *,
    source_plan: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    release_host_identity: list[dict[str, Any]],
    release_root: Path,
    hardware_identity: Mapping[str, Any],
    cells: int,
    ticks: int,
) -> dict[str, Any]:
    source_plan_sha256 = source_plan.get("plan_sha256")
    if not isinstance(source_plan_sha256, str) or len(source_plan_sha256) != 64:
        raise AllocationDiagnosticError("source plan lacks exact plan identity")
    configuration = {
        "cells": cells,
        "ticks": ticks,
        "resident_capacity": EXPECTED_CAPACITY,
        "internal_lifecycle_interval_ticks": EXPECTED_TICKS_PER_GENERATION,
        "execution": source_plan.get("execution"),
        "resolved_profile": source_plan.get("resolved_profile"),
    }
    committed_snapshot = artifact_manifest.get("committed_source_snapshot")
    if not isinstance(committed_snapshot, Mapping):
        raise AllocationDiagnosticError(
            "artifact manifest lacks committed source identity"
        )
    artifact_source_commit = committed_snapshot.get("build_source_commit")
    repository_head = _git_head()
    if (
        not isinstance(artifact_source_commit, str)
        or len(artifact_source_commit) != 40
        or artifact_source_commit != repository_head
    ):
        raise AllocationDiagnosticError(
            "artifact source commit differs from the frozen repository HEAD"
        )
    source = {
        "repository_head": repository_head,
        "artifact_source_commit": artifact_source_commit,
        "runtime_source_root": str(RUNTIME_SOURCE_ROOT),
        "production_release_root": str(release_root.resolve()),
        "production_release_host_identity": release_host_identity,
    }
    execution_contract = {
        "schema_version": f"{SCHEMA_VERSION}_execution_contract_v1",
        "authority": AUTHORITY,
        "execution_route_identity": source_plan["execution"][
            "execution_route_identity"
        ],
        "native_tick_span": True,
        "terminal_projection_outside_timed_window": True,
    }
    identity = {
        "source_plan_sha256": source_plan_sha256,
        "execution_contract_identity_sha256": canonical_sha256(execution_contract),
        "source_identity_sha256": canonical_sha256(source),
        "device_identity_sha256": hardware_identity.get("identity_sha256"),
        "configuration_identity_sha256": canonical_sha256(configuration),
        "warmup_identity_sha256": canonical_sha256(
            {"warmup_identity": WARMUP_IDENTITY}
        ),
        "warmup_identity": WARMUP_IDENTITY,
    }
    if not all(isinstance(value, str) and value for value in identity.values()):
        raise AllocationDiagnosticError("benchmark comparison identity is incomplete")
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def _result_summary(
    result: Mapping[str, Any],
    *,
    ticks: int,
    expected_m7_enabled: bool,
    expected_m8_enabled: bool = False,
) -> dict[str, Any]:
    scalar_fields = (
        "owner",
        "absolute_tick_span_native_owner",
        "native_tick_iterations",
        "tick_span_start",
        "tick_span_end",
        "tick_span_count",
        "terminal_generation_boundary",
        "wdl_jungle_native_owner",
        "eed_native_owner",
        "wdl_jungle_python_tick_or_generation_ownership",
        "direct_cpp_window_core_dispatch",
        "production_native_chunk_loop",
        "hidden_fallback",
        "runtime_jit",
        "total_kernel_launches",
        "phase_1_launches",
        "birth_prepare_launches",
        "birth_commit_launches",
        "birth_prepare_commit_fused_launches",
        "turnover_shadow_plan_launches",
        "turnover_cull_refill_plan_launches",
        "turnover_commit_launches",
        "turnover_publish_launches",
        "host_synchronizations_inside_loop",
        "final_host_synchronizations",
        "intermediate_device_to_host_operations",
        "intermediate_host_to_device_operations",
        "final_device_to_host_operations",
        "final_host_to_device_operations",
        "option_d_m0_telemetry_enabled",
        "policy_birth_segment_refill_calls_inside_tick_loop",
        "policy_birth_segment_refill_receipts",
        "policy_birth_segment_device_to_host_operations_lower_bound",
        "policy_birth_segment_host_to_device_operations_lower_bound",
        "policy_birth_segment_host_wall_time_ns",
        "policy_birth_segment_device_to_host_time_ns",
        "policy_birth_segment_host_to_device_time_ns",
        "policy_birth_segment_torch_random_generation_time_ns",
        "policy_birth_segment_torch_compare_pack_time_ns",
        "policy_birth_segment_allocation_stack_copy_time_ns",
        "gaussian_segment_refill_calls_inside_tick_loop",
        "gaussian_segment_refill_receipts",
        "gaussian_segment_boundary_receipts",
        "gaussian_segment_reuse_calls_inside_tick_loop",
        "gaussian_segment_device_to_host_operations_lower_bound",
        "gaussian_segment_host_to_device_operations_lower_bound",
        "gaussian_segment_host_wall_time_ns",
        "gaussian_segment_device_to_host_time_ns",
        "gaussian_segment_host_to_device_time_ns",
        "gaussian_segment_raw_generation_time_ns",
        "gaussian_segment_transcendental_conversion_time_ns",
        "gaussian_segment_allocation_copy_time_ns",
        "gaussian_segment_state_preparation_time_ns",
        "gaussian_segment_tape_allocation_time_ns",
        "gaussian_segment_stack_pack_time_ns",
        "gaussian_segment_primary_h2d_submission_time_ns",
        "gaussian_segment_secondary_h2d_submission_time_ns",
        "gaussian_segment_window_base_h2d_submission_time_ns",
        "gaussian_pre_refill_synchronization_time_ns",
        "gaussian_post_copy_synchronization_time_ns",
        "resident_window_outer_boundary_device_to_host_operations_lower_bound",
        "resident_window_outer_event_device_to_host_operations_lower_bound",
        "resident_window_outer_refill_device_to_host_operations_lower_bound",
        "resident_window_outer_refill_host_to_device_operations_lower_bound",
    )
    summary = {name: result.get(name) for name in scalar_fields}
    required = {
        "owner": "native_extension",
        "absolute_tick_span_native_owner": True,
        "native_tick_iterations": ticks,
        "tick_span_start": 0,
        "tick_span_end": ticks,
        "tick_span_count": ticks,
        "terminal_generation_boundary": ticks == EXPECTED_TICKS_PER_GENERATION,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "wdl_jungle_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    mismatches = {
        name: {"expected": expected, "observed": summary.get(name)}
        for name, expected in required.items()
        if summary.get(name) != expected
    }
    if mismatches:
        raise AllocationDiagnosticError(
            "native profiling result differs from its exact ownership contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    refill_receipts = summary.get("policy_birth_segment_refill_receipts")
    if not isinstance(refill_receipts, list) or not refill_receipts:
        raise AllocationDiagnosticError(
            "native result lacks policy-refill receipts for M7 arm validation"
        )
    m7_runtime_identities: list[dict[str, Any]] = []
    m8_runtime_identities: list[dict[str, Any]] = []
    turnover_boundary_receipt_count = 0
    for index, refill_receipt in enumerate(refill_receipts):
        if not isinstance(refill_receipt, Mapping):
            raise AllocationDiagnosticError(
                f"policy-refill receipt {index} is not a mapping"
            )
        m8_observed = _m8_runtime_identity(
            refill_receipt, expected_enabled=expected_m8_enabled
        )
        turnover_boundary_receipt_count += int(
            refill_receipt.get("turnover_boundary") is True
        )
        m7_observed = _m7_runtime_identity(
            refill_receipt,
            expected_enabled=expected_m7_enabled,
            expected_used=(
                bool(m8_observed["physical_refill_performed"])
                if expected_m8_enabled
                else expected_m7_enabled
            ),
        )
        m7_runtime_identities.append(m7_observed)
        m8_runtime_identities.append(m8_observed)
    physical_refill_count = sum(
        identity.get("physical_refill_performed") is True
        for identity in m8_runtime_identities
    )
    reuse_count = sum(
        identity.get("resident_window_reused") is True
        for identity in m8_runtime_identities
    )
    if not expected_m8_enabled:
        physical_refill_count = len(refill_receipts)
    summary["validated_policy_arm_identity"] = {
        "expected_m7_enabled": expected_m7_enabled,
        "expected_m8_enabled": expected_m8_enabled,
        "receipt_count": len(refill_receipts),
        "resident_suffix_reuse_receipt_count": reuse_count,
        "physical_refill_receipt_count": physical_refill_count,
        "turnover_boundary_receipt_count": turnover_boundary_receipt_count,
        "m7_used_on_every_physical_refill": (
            expected_m7_enabled
            if not expected_m8_enabled
            else all(
                identity[M7_USED_FIELD]
                for identity, m8_identity in zip(
                    m7_runtime_identities, m8_runtime_identities, strict=True
                )
                if m8_identity.get("physical_refill_performed") is True
            )
        ),
        "m7_unused_on_every_suffix_reuse": all(
            not identity[M7_USED_FIELD]
            for identity, m8_identity in zip(
                m7_runtime_identities, m8_runtime_identities, strict=True
            )
            if m8_identity.get("resident_window_reused") is True
        ),
    }
    if ticks == EXPECTED_TICKS_PER_GENERATION:
        turnover_mismatches = {
            name: summary.get(name)
            for name in (
                "turnover_shadow_plan_launches",
                "turnover_cull_refill_plan_launches",
                "turnover_commit_launches",
                "turnover_publish_launches",
            )
            if summary.get(name) != 0
        }
        if turnover_boundary_receipt_count != 0 or turnover_mismatches:
            raise AllocationDiagnosticError(
                "hybrid-overlapping B48/T160 route unexpectedly activated native "
                "terminal turnover: "
                + json.dumps(
                    {
                        "turnover_boundary_receipt_count": (
                            turnover_boundary_receipt_count
                        ),
                        "turnover_launches": turnover_mismatches,
                    },
                    sort_keys=True,
                )
            )
    return summary


def run_profile_window(
    *,
    plan_path: Path,
    artifact: Path,
    manifest: Path,
    output_root: Path,
    ticks: int,
    cells: int,
    release_root: Path,
    allow_diagnostic_artifact_rebind: bool = False,
    allow_candidate_artifact_rebind: bool = False,
    project_terminal_trajectory_digests: bool = False,
    expected_m7_enabled: bool,
    expected_m8_enabled: bool,
    benchmark_run_index: int,
    benchmark_sequence_index: int,
) -> dict[str, Any]:
    if not 3 <= ticks <= EXPECTED_TICKS_PER_GENERATION:
        raise AllocationDiagnosticError("profile tick window must be in [3,160]")
    if cells not in {12, 24, 48}:
        raise AllocationDiagnosticError("profile cell count must be B12, B24, or B48")
    if type(benchmark_run_index) is not int or benchmark_run_index < 0:
        raise AllocationDiagnosticError("benchmark run index must be nonnegative")
    if type(benchmark_sequence_index) is not int or benchmark_sequence_index < 0:
        raise AllocationDiagnosticError("benchmark sequence index must be nonnegative")
    benchmark_arm = _benchmark_arm(
        m7_enabled=expected_m7_enabled, m8_enabled=expected_m8_enabled
    )
    plan_path = plan_path.resolve()
    artifact = artifact.resolve()
    manifest = manifest.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise AllocationDiagnosticError(
            f"refusing to overwrite output root: {output_root}"
        )

    gpu_processes = _gpu_compute_processes()
    competing = _competing_gpu_processes(gpu_processes)
    if competing:
        raise AllocationDiagnosticError(
            "competing GPU Python/profiler process is active: "
            + json.dumps(competing, sort_keys=True)
        )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AllocationDiagnosticError(
            "profiling requires the single admitted CUDA device"
        )

    source_plan = _validated_plan(plan_path)
    bound_plan = _bind_artifacts(
        source_plan,
        artifact=artifact,
        manifest=manifest,
        allow_diagnostic_artifact_rebind=allow_diagnostic_artifact_rebind,
        allow_candidate_artifact_rebind=allow_candidate_artifact_rebind,
        expected_m7_enabled=expected_m7_enabled,
        expected_m8_enabled=expected_m8_enabled,
    )
    bound_plan = _diagnostic_cells_plan(bound_plan, cells=cells)
    bound_plan = _translation_compatible_plan(bound_plan)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    m7_manifest_identity = _m7_manifest_identity(
        manifest_payload, expected_enabled=expected_m7_enabled
    )
    m8_manifest_identity = _m8_manifest_identity(
        manifest_payload, expected_enabled=expected_m8_enabled
    )
    release_host_identity = _release_host_identity(release_root)
    hardware_identity = _hardware_identity(torch)
    comparison_identity = _comparison_identity(
        source_plan=source_plan,
        artifact_manifest=manifest_payload,
        release_host_identity=release_host_identity,
        release_root=release_root,
        hardware_identity=hardware_identity,
        cells=cells,
        ticks=ticks,
    )
    output_root.mkdir(parents=True)
    qualification, reporting = _diagnostic_descriptor(
        output_root,
        artifact_sha256=_sha256(artifact),
        manifest_sha256=_sha256(manifest),
        plan_sha256=str(bound_plan["plan_sha256"]),
    )
    diagnostic_plan = _diagnostic_plan(
        bound_plan, qualification=qualification, reporting=reporting
    )
    execution_contract = {
        "schema_version": f"{SCHEMA_VERSION}_execution_contract_v1",
        "authority": AUTHORITY,
        "purpose": f"bounded B{cells} post-warm-up tick anatomy",
        "repository_head": _git_head(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "audit_working_directory": str(ROOT),
        "runtime_source_root": str(RUNTIME_SOURCE_ROOT),
        "production_release_root": str(release_root.resolve()),
        "production_release_host_identity": release_host_identity,
        "process": _process_identity(),
        "plan_path": str(plan_path),
        "source_plan_sha256": source_plan.get("plan_sha256"),
        "diagnostic_plan_sha256": diagnostic_plan["plan_sha256"],
        "artifact_path": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "hardware_identity": hardware_identity,
        "cells": cells,
        "resident_capacity": EXPECTED_CAPACITY,
        "ticks": ticks,
        "internal_lifecycle_interval_ticks": EXPECTED_TICKS_PER_GENERATION,
        "nvtx_capture_range": NVTX_RANGE,
        "output_root": str(output_root),
        "gpu_processes_at_preflight": gpu_processes,
        "expected_terminal_artifact": str(output_root / "execution_receipt.json"),
        "qualification_claim": False,
        "scientific_evidence_claim": False,
        "optimization_performed": False,
        "diagnostic_artifact_rebind": allow_diagnostic_artifact_rebind,
        "candidate_artifact_rebind": allow_candidate_artifact_rebind,
        "benchmark_arm": benchmark_arm,
        "benchmark_run_index": benchmark_run_index,
        "benchmark_sequence_index": benchmark_sequence_index,
        "benchmark_arm_order": list(BENCHMARK_ARM_ORDER),
        "comparison_identity": comparison_identity,
        "warmup_completed": True,
        "m7_arm": "candidate" if expected_m7_enabled else "control",
        "expected_m7_enabled": expected_m7_enabled,
        "m7_manifest_identity": m7_manifest_identity,
        "expected_m8_enabled": expected_m8_enabled,
        "m8_manifest_identity": m8_manifest_identity,
    }
    execution_contract["contract_sha256"] = canonical_sha256(execution_contract)
    atomic_write_json(output_root / "execution_contract.json", execution_contract)

    provider: Any = None
    module: Any = None
    prepared: Any = None
    sources: Any = None
    started_at_ns = time.time_ns()
    try:
        module = _load_exact_module(artifact, eed_overlay=True)
        provider = AllocationOnlyProvider(bind_committed_defaults=False)
        batch = diagnostic_plan["physical_batches"][0]
        request = provider.translate_batch(diagnostic_plan, str(batch["batch_id"]))
        adapters = dict(provider._resolved_execution_adapters(diagnostic_plan))
        selected_cells = provider._batch_cells(diagnostic_plan, request)
        context = SimpleNamespace(
            run_root=output_root, batch_root=output_root, progress_events=[]
        )
        source_preparer = (
            adapters.get("source_preparer")
            or provider._prepare_generation_zero_sources_v2
        )
        sources = tuple(
            source_preparer(
                diagnostic_plan,
                request,
                context,
                runner_builder=adapters["runner_builder"],
                cells=selected_cells,
                artifact=str(artifact),
                qualification=str(qualification),
                reporting=str(reporting),
                segment_generations=1,
                source_capability_mask=_wdl_source_capabilities_v2(
                    {"genome_v2": True}, jungle_overlay=True, eed_overlay=True
                ),
            )
        )
        atomic_write_json(
            output_root / "progress.json",
            {"schema_version": SCHEMA_VERSION, "stage": "sources_prepared"},
        )
        _, prepared_requests, prepared = provider._prepare_resident_segment_owner_v1(
            plan=diagnostic_plan,
            runner_builder=adapters["runner_builder"],
            request_builder=adapters["request_builder"],
            preparer=adapters["preparer"],
            cells=selected_cells,
            cell_ids=request.cell_ids,
            batch_root=output_root,
            start_generation=0,
            end_generation=1,
            ticks_per_generation=EXPECTED_TICKS_PER_GENERATION,
            artifact=str(artifact),
            qualification=str(qualification),
            reporting=str(reporting),
            module=module,
            source_projections=sources,
            allocation_generations=provider._agent_id_allocation_generation_count(
                request, start_generation=0
            ),
            genome_v2_checkpoint_authority_binding=dict(
                request.genome_v2_checkpoint_authority_binding
            ),
            allow_short_technical_gate_generation_count=True,
            allow_partial_generation_source_cursor=True,
            physical_window_generations=1,
        )
        sources = None
        gc.collect()
        atomic_write_json(
            output_root / "progress.json",
            {"schema_version": SCHEMA_VERSION, "stage": "owner_prepared"},
        )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        nvtx_pushed = False
        try:
            torch.cuda.nvtx.range_push(NVTX_RANGE)
            nvtx_pushed = True
            window_started_at_unix_ns = time.time_ns()
            wall_start_ns = time.perf_counter_ns()
            direct_result = adapters["executor"](
                module,
                physical_window_arguments=prepared.physical_window_arguments,
                logical_generation_count=1,
                absolute_tick_span=ticks,
            )
            torch.cuda.synchronize()
            wall_end_ns = time.perf_counter_ns()
            window_completed_at_unix_ns = time.time_ns()
        finally:
            if nvtx_pushed:
                torch.cuda.nvtx.range_pop()

        if not isinstance(direct_result, Mapping):
            raise AllocationDiagnosticError("native profiler result is not a mapping")
        elapsed_ns = wall_end_ns - wall_start_ns
        result_summary = _result_summary(
            direct_result,
            ticks=ticks,
            expected_m7_enabled=expected_m7_enabled,
            expected_m8_enabled=expected_m8_enabled,
        )
        validated_policy_arm = result_summary["validated_policy_arm_identity"]
        terminal_trajectory_digests: dict[str, str] | None = None
        terminal_cursor: dict[str, int] | None = None
        if project_terminal_trajectory_digests:
            from engine.native_runtime.checkpoint_digests import (
                DigestLayer,
                checkpoint_digest_manifest,
            )

            settlement = adapters["projector"](
                prepared,
                direct_result=direct_result,
                capability=prepared_requests[0].cuda_capability,
                reporting_enabled=False,
            )
            terminal_trajectory_digests = {
                cell_id: str(
                    checkpoint_digest_manifest(
                        projection,
                        layer=DigestLayer.TRAJECTORY_STATE,
                    )["digest"]
                )
                for cell_id, projection in zip(
                    settlement.cell_ids,
                    settlement.checkpoint_projections,
                    strict=True,
                )
            }
            terminal_cursor = {
                "generation": int(settlement.terminal_generation),
                "tick": int(settlement.terminal_tick),
            }
        receipt = {
            "schema_version": f"{SCHEMA_VERSION}_execution_receipt_v1",
            "authority": AUTHORITY,
            "status": "success",
            "started_at_unix_ns": started_at_ns,
            "completed_at_unix_ns": time.time_ns(),
            "execution_contract_sha256": execution_contract["contract_sha256"],
            "artifact_source_commit": manifest_payload["committed_source_snapshot"][
                "build_source_commit"
            ],
            "artifact_sha256": _sha256(artifact),
            "manifest_sha256": _sha256(manifest),
            "native_window_started_at_unix_ns": window_started_at_unix_ns,
            "native_window_completed_at_unix_ns": window_completed_at_unix_ns,
            "native_window_wall_time_ns": elapsed_ns,
            "native_window_wall_time_ms": elapsed_ns / 1_000_000.0,
            "wall_ms_per_tick": elapsed_ns / ticks / 1_000_000.0,
            "ticks": ticks,
            "cells": cells,
            "benchmark_arm": benchmark_arm,
            "benchmark_run_index": benchmark_run_index,
            "benchmark_sequence_index": benchmark_sequence_index,
            "benchmark_arm_order": list(BENCHMARK_ARM_ORDER),
            "comparison_identity": comparison_identity,
            "warmup_completed": True,
            "torch_peak_allocated_bytes_during_window": int(
                torch.cuda.max_memory_allocated()
            ),
            "torch_peak_reserved_bytes_during_window": int(
                torch.cuda.max_memory_reserved()
            ),
            "native_result": result_summary,
            "qualification_claim": False,
            "scientific_evidence_claim": False,
            "optimization_performed": False,
            "terminal_trajectory_digest_sha256s": terminal_trajectory_digests,
            "terminal_cursor": terminal_cursor,
            "m7_arm_identity": {
                "arm": "candidate" if expected_m7_enabled else "control",
                "expected_enabled": expected_m7_enabled,
                "manifest": m7_manifest_identity,
                "runtime": validated_policy_arm,
            },
            "m8_arm_identity": {
                "arm": benchmark_arm,
                "expected_enabled": expected_m8_enabled,
                "manifest": m8_manifest_identity,
                "runtime": validated_policy_arm,
            },
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(output_root / "execution_receipt.json", receipt)
        atomic_write_json(
            output_root / "progress.json",
            {"schema_version": SCHEMA_VERSION, "stage": "complete"},
        )
        return receipt
    except BaseException as exc:
        atomic_write_json(
            output_root / "failure.json",
            {
                "schema_version": f"{SCHEMA_VERSION}_failure_v1",
                "authority": AUTHORITY,
                "error": f"{type(exc).__name__}: {exc}",
                "started_at_unix_ns": started_at_ns,
                "failed_at_unix_ns": time.time_ns(),
            },
        )
        raise
    finally:
        prepared = None
        sources = None
        module = None
        provider = None
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=24)
    parser.add_argument("--cells", type=int, choices=(12, 24, 48), default=48)
    parser.add_argument(
        "--expected-m7-enabled",
        choices=("false", "true"),
        required=True,
        help="fail closed unless manifest and native receipt match this M7 arm",
    )
    parser.add_argument(
        "--expected-m8-enabled",
        choices=("false", "true"),
        required=True,
        help="fail closed unless manifest and native receipts match this M8 arm",
    )
    parser.add_argument("--benchmark-run-index", type=int, required=True)
    parser.add_argument("--benchmark-sequence-index", type=int, required=True)
    parser.add_argument(
        "--allow-diagnostic-artifact-rebind",
        action="store_true",
        help=(
            "allow a manifest-hash-bound M0 telemetry artifact to replace the "
            "frozen plan artifact only in the non-admitting diagnostic plan"
        ),
    )
    parser.add_argument(
        "--allow-candidate-artifact-rebind",
        action="store_true",
        help=(
            "allow a manifest-hash-bound telemetry-off candidate artifact to "
            "replace the frozen plan artifact only in the non-admitting plan"
        ),
    )
    parser.add_argument(
        "--project-terminal-trajectory-digests",
        action="store_true",
        help=(
            "project canonical trajectory-state digests after the timed window; "
            "projection time is excluded from the benchmark"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = run_profile_window(
        plan_path=args.plan,
        artifact=args.artifact,
        manifest=args.manifest,
        output_root=args.output_root,
        ticks=args.ticks,
        cells=args.cells,
        release_root=args.release_root,
        allow_diagnostic_artifact_rebind=args.allow_diagnostic_artifact_rebind,
        allow_candidate_artifact_rebind=args.allow_candidate_artifact_rebind,
        project_terminal_trajectory_digests=(args.project_terminal_trajectory_digests),
        expected_m7_enabled=args.expected_m7_enabled == "true",
        expected_m8_enabled=args.expected_m8_enabled == "true",
        benchmark_run_index=args.benchmark_run_index,
        benchmark_sequence_index=args.benchmark_sequence_index,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
