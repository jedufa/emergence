#!/usr/bin/env python3
"""Bounded OFF/ON acceptance harness for Option D M0 instrumentation."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine"
for _path in (ROOT, ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine.campaigns.compiler import compile_campaign  # noqa: E402
from engine.campaigns.orchestration import (  # noqa: E402
    atomic_write_json,
    execute_campaign,
)
from engine.campaigns.providers import (  # noqa: E402
    default_execution_provider_registry,
)
from engine.campaigns.route_qualification import (  # noqa: E402
    bounded_route_qualification_preflight_v1,
)
from engine.campaigns.production_engine_lifecycle_v1 import (  # noqa: E402
    LEGACY_SYNCHRONIZED_TRUNCATION_V1,
    LIFE_HISTORY_PROFILE_INERT_V1,
    PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1,
)


SCHEMA_VERSION = "eiel_option_d_m0_acceptance_harness_v1"
WORKER_SCHEMA_VERSION = "eiel_option_d_m0_acceptance_worker_v1"
ACCEPTANCE_SCHEMA_VERSION = "eiel_option_d_m0_acceptance_receipt_v1"
OVERHEAD_SCHEMA_VERSION = "eiel_option_d_m0_instrumentation_overhead_receipt_v1"
COMPONENT_SCHEMA_VERSION = "eiel_option_d_m0_component_profile_v1"
PRIMARY_RECEIPT_SCHEMA_VERSION = "eiel_option_d_rng_instrumentation_receipt_v1"
SOURCE_ENVIRONMENT_SCHEMA_VERSION = "eiel_option_d_m0_source_rebind_v3"
ARTIFACT_ABI = "architecture_b_patch_p2_stage3_v2"
PROVIDER_ID = "eiel-production-provider-v1"
PROFILE_ID = "eiel-production-runtime-v1"
EXTENSION_NAME = "eiel_cuda_architecture_b_flattened_controller_v1.so"
MANIFEST_SCHEMA_VERSION = (
    "eiel_cuda_architecture_b_flattened_controller_extension_manifest_v1"
)
SOURCE_ENVIRONMENT_ARTIFACT = "option_d_m0_source_environment_receipt"
SOURCE_ENVIRONMENT_ABI = "eiel_option_d_m0_source_rebind_v3"
MODES = ("telemetry-off", "telemetry-on")
SURFACE_ORDINARY = "ordinary-surface"
SURFACE_TURNOVER = "turnover-surface"
TERMINAL_GENERATION = 3

_PARITY_CHECKPOINT_FIELDS = (
    "source_import_projection_digest_sha256s",
    "terminal_projection_digest_sha256s",
    "canonical_terminal_state_digest_sha256s",
    "canonical_checkpoint_storage_digest_sha256s",
)
_BUILD_TWIN_FIELDS = (
    "schema_version",
    "module_name",
    "cuda_arch",
    "compiler_binary_sha256",
    "compiler_version",
    "nvcc_binary_sha256",
    "nvcc_version",
    "torch_version",
    "torch_cuda_version",
    "qualified_cache_width",
    "source_hashes",
    "primary_sources",
    "relevant_linked_library_sha256",
    "direct_link_library_identities",
    "committed_source_snapshot.build_source_commit",
    "committed_source_snapshot.compiled_source_snapshot_sha256",
)
_ALLOWED_BUILD_TWIN_DIFFERENCES = frozenset(
    {
        "artifact",
        "artifact_environment_identity",
        "artifact_environment_identity_sha256",
        "artifact_identity",
        "artifact_sha256",
        "build_directory",
        "cache_width_resource_validation",
        "commands",
        "cxx_flags",
        "elf_build_id",
        "manifest",
        "nvcc_flags",
        "object_directory",
        "option_d_m0_telemetry_build",
        "resource_receipt",
    }
)
_IDLE_GPU_BACKGROUND_PROCESSES = frozenset(
    {
        "Xorg",
        "firefox",
        "gnome-remote-desktop-daemon",
        "gnome-shell",
        "nautilus",
        "ptyxis",
        "resources",
        "xdg-desktop-portal-gnome",
    }
)
_IDLE_GPU_UTILIZATION_THRESHOLD_PERCENT = 25
_NATIVE_COMPONENT_FIELDS = (
    "policy_birth_segment_refill_calls_inside_tick_loop",
    "policy_birth_segment_device_to_host_operations_lower_bound",
    "policy_birth_segment_host_to_device_operations_lower_bound",
    "policy_birth_segment_host_wall_time_ns",
    "policy_birth_segment_device_to_host_time_ns",
    "policy_birth_segment_host_to_device_time_ns",
    "policy_birth_segment_torch_random_generation_time_ns",
    "policy_birth_segment_torch_compare_pack_time_ns",
    "policy_birth_segment_allocation_stack_copy_time_ns",
    "gaussian_segment_refill_calls_inside_tick_loop",
    "gaussian_segment_device_to_host_operations_lower_bound",
    "gaussian_segment_host_to_device_operations_lower_bound",
    "gaussian_segment_host_wall_time_ns",
    "gaussian_segment_device_to_host_time_ns",
    "gaussian_segment_host_to_device_time_ns",
    "gaussian_segment_raw_generation_time_ns",
    "gaussian_segment_transcendental_conversion_time_ns",
    "gaussian_segment_allocation_copy_time_ns",
    "host_synchronizations_inside_loop",
    "final_host_synchronizations",
    "intermediate_device_to_host_operations",
    "final_device_to_host_operations",
    "intermediate_host_to_device_operations",
    "final_host_to_device_operations",
    "intermediate_host_synchronizations",
    "aten_operations_inside_tick_loop",
    "exact_cpu_policy_service_calls_inside_tick_loop",
    "resident_window_outer_boundary_device_to_host_operations_lower_bound",
    "resident_window_outer_event_device_to_host_operations_lower_bound",
    "resident_window_outer_refill_device_to_host_operations_lower_bound",
    "resident_window_outer_refill_host_to_device_operations_lower_bound",
    "checkpoint_projection_time_ns",
    "checkpoint_compression_time_ns",
)
_PRIMARY_COMPONENT_FIELDS = (
    "actual_ordinary_birth_count",
    "actual_turnover_birth_count",
    "torch_policy_raw_word_delta",
    "torch_policy_semantic_draw_delta",
    "process_rss_bytes",
    "process_hwm_bytes",
    "resident_vram_bytes",
    "peak_vram_bytes",
    "receipt_construction_time_ns",
    "checkpoint_projection_time_ns",
    "checkpoint_compression_time_ns",
)
_ROUTE_PUBLICATION_NAMES = frozenset(
    {
        "execution_route_qualification.json",
        "route_qualification.json",
        "route_qualification_publication.json",
    }
)


class M0AcceptanceError(RuntimeError):
    """The bounded M0 candidate failed a fail-closed acceptance condition."""


@dataclass(frozen=True, slots=True)
class HarnessOptions:
    off_build_dir: Path
    on_build_dir: Path
    off_receipt_dir: Path
    on_receipt_dir: Path
    on_source_environment: Path
    output_root: Path
    campaign_id: str = "option-d-m0-b1g3t160-parity-a86bf726a"
    ecology_seed: int = 303
    task_seed: int = 1001
    founder_mode: str = "legacy"
    python_executable: Path = Path(sys.executable)
    worker_timeout_seconds: float = 600.0
    require_idle_gpu: bool = False


@dataclass(frozen=True, slots=True)
class ModeBinding:
    mode: str
    artifacts: Mapping[str, Mapping[str, Any]]
    artifact_path: Path
    artifact_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerInvocation:
    mode: str
    command: tuple[str, ...]
    subprocess_wall_time_ns: int
    worker_execution_wall_time_ns: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ModeResult:
    mode: str
    run_root: Path
    plan: Mapping[str, Any]
    worker: Mapping[str, Any]
    preflight: Mapping[str, Any]
    batch_manifest: Mapping[str, Any]
    invocation: WorkerInvocation


class WorkerRunner(Protocol):
    def __call__(
        self,
        *,
        mode: str,
        plan_path: Path,
        run_root: Path,
        worker_result_path: Path,
        python_executable: Path,
        timeout_seconds: float,
    ) -> WorkerInvocation: ...


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise M0AcceptanceError(f"{name} is unreadable: {exc}") from exc
    if not isinstance(value, Mapping):
        raise M0AcceptanceError(f"{name} must be a JSON object")
    return dict(value)


def _existing_file(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise M0AcceptanceError(f"{name} is not an existing file: {resolved}")
    return resolved


def _path_value(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise M0AcceptanceError(f"artifact manifest lacks {dotted}")
        current = current[part]
    return current


def _artifact_row(path: Path, *, kind: str, abi: str) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "kind": kind,
        "abi": abi,
    }


def _validate_candidate_receipts(
    *,
    receipt_dir: Path,
    artifact_path: Path,
    artifact_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[Path, Path]:
    qualification_path = _existing_file(
        receipt_dir / "qualification.json", name="mechanic qualification receipt"
    )
    reporting_path = _existing_file(
        receipt_dir / "reporting.json", name="reporting qualification receipt"
    )
    qualification = _read_object(
        qualification_path, name="mechanic qualification receipt"
    )
    reporting = _read_object(reporting_path, name="reporting qualification receipt")
    qualified_artifact = qualification.get("artifact")
    if (
        qualification.get("schema_version")
        != "eiel_production_engine_mechanic_qualification_v1"
        or qualification.get("ok") is not True
        or qualification.get("provider_id") != PROVIDER_ID
        or qualification.get("profile_id") != PROFILE_ID
        or qualification.get("artifact_abi") != ARTIFACT_ABI
        or not isinstance(qualified_artifact, Mapping)
        or Path(str(qualified_artifact.get("path", ""))).resolve() != artifact_path
        or qualified_artifact.get("sha256") != artifact_sha256
        or Path(str(qualified_artifact.get("manifest_path", ""))).resolve()
        != manifest_path
        or qualified_artifact.get("manifest_sha256") != manifest_sha256
    ):
        raise M0AcceptanceError(
            "mechanic qualification receipt does not bind the selected artifact"
        )
    if (
        reporting.get("schema_version")
        != "eiel_production_engine_reporting_qualification_v1"
        or reporting.get("ok") is not True
        or reporting.get("provider_id") != PROVIDER_ID
        or reporting.get("profile_id") != PROFILE_ID
        or reporting.get("artifact_sha256") != artifact_sha256
    ):
        raise M0AcceptanceError(
            "reporting qualification receipt does not bind the selected artifact"
        )
    return qualification_path, reporting_path


def _mode_binding(
    *,
    mode: str,
    build_dir: Path,
    receipt_dir: Path,
    source_environment_path: Path | None,
) -> ModeBinding:
    expected_enabled = mode == "telemetry-on"
    resolved_build = build_dir.expanduser().resolve()
    resolved_receipts = receipt_dir.expanduser().resolve()
    artifact_path = _existing_file(
        resolved_build / EXTENSION_NAME, name=f"{mode} CUDA extension"
    )
    manifest_path = _existing_file(
        Path(f"{artifact_path}.manifest.json"), name=f"{mode} extension manifest"
    )
    manifest = _read_object(manifest_path, name=f"{mode} extension manifest")
    artifact_sha256 = _sha256_file(artifact_path)
    manifest_sha256 = _sha256_file(manifest_path)
    telemetry = manifest.get("option_d_m0_telemetry_build")
    qualified_cache_width = manifest.get("qualified_cache_width")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("module_name") != EXTENSION_NAME.removesuffix(".so")
        or Path(str(manifest.get("artifact", ""))).resolve() != artifact_path
        or manifest.get("artifact_sha256") != artifact_sha256
        or not isinstance(telemetry, Mapping)
        or telemetry.get("schema_version")
        != "eiel_option_d_m0_telemetry_build_identity_v1"
        or telemetry.get("option_d_m0_telemetry_enabled") is not expected_enabled
        or telemetry.get("compile_definition")
        != f"-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED={int(expected_enabled)}"
        or not isinstance(qualified_cache_width, Mapping)
        or qualified_cache_width.get("schema_version")
        != "eiel_cuda_architecture_b_qualified_cache_width_v1"
        or qualified_cache_width.get("value") != 2
        or qualified_cache_width.get("compile_definition")
        != "-DEIEL_CUDA_ARCHITECTURE_B_CACHE_WIDTH_VALID=(cache_width==2)"
    ):
        raise M0AcceptanceError(f"{mode} artifact telemetry identity is invalid")
    qualification_path, reporting_path = _validate_candidate_receipts(
        receipt_dir=resolved_receipts,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    artifacts: dict[str, Mapping[str, Any]] = {
        "cuda_extension": _artifact_row(
            artifact_path, kind="cuda_extension", abi=ARTIFACT_ABI
        ),
        "extension_manifest": _artifact_row(
            manifest_path, kind="extension_manifest", abi=ARTIFACT_ABI
        ),
        "qualification_receipt": _artifact_row(
            qualification_path, kind="qualification_receipt", abi=ARTIFACT_ABI
        ),
        "reporting_receipt": _artifact_row(
            reporting_path, kind="reporting_receipt", abi=ARTIFACT_ABI
        ),
    }
    if expected_enabled:
        if source_environment_path is None:
            raise M0AcceptanceError(
                "telemetry-on requires a source/environment receipt"
            )
        source_path = _existing_file(
            source_environment_path, name="Option D source/environment receipt"
        )
        source_environment = _read_object(
            source_path, name="Option D source/environment receipt"
        )
        route_scope = source_environment.get("route_scope_contract")
        snapshot = manifest.get("committed_source_snapshot")
        if (
            source_environment.get("schema_version")
            != SOURCE_ENVIRONMENT_SCHEMA_VERSION
            or source_environment.get("production_interpreter", {}).get("status")
            != "PASS"
            or not isinstance(route_scope, Mapping)
            or route_scope.get("passed_for_measurement_identity") is not True
            or route_scope.get("production_integration_authority") is not False
            or not isinstance(snapshot, Mapping)
            or route_scope.get("accepted_eiel_commit")
            != snapshot.get("build_source_commit")
            or route_scope.get("compiled_source_snapshot_sha256")
            != snapshot.get("compiled_source_snapshot_sha256")
        ):
            raise M0AcceptanceError(
                "Option D source/environment receipt does not bind the ON build"
            )
        artifacts[SOURCE_ENVIRONMENT_ARTIFACT] = _artifact_row(
            source_path,
            kind=SOURCE_ENVIRONMENT_ARTIFACT,
            abi=SOURCE_ENVIRONMENT_ABI,
        )
    elif source_environment_path is not None:
        raise M0AcceptanceError(
            "telemetry-off must not receive Option D source/environment authority"
        )
    return ModeBinding(
        mode=mode,
        artifacts=artifacts,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
    )


def _validate_build_twins(off: ModeBinding, on: ModeBinding) -> dict[str, Any]:
    if set(off.manifest) != set(on.manifest):
        raise M0AcceptanceError("OFF/ON build twin manifest fields differ")
    unexpected_differences = sorted(
        key
        for key in off.manifest
        if off.manifest[key] != on.manifest[key]
        and key not in _ALLOWED_BUILD_TWIN_DIFFERENCES
    )
    if unexpected_differences:
        raise M0AcceptanceError(
            "OFF/ON build twins differ outside telemetry-controlled fields: "
            + ", ".join(unexpected_differences)
        )

    normalized_flags: dict[str, list[str]] = {}
    for field in ("cxx_flags", "nvcc_flags"):
        normalized_by_mode: dict[str, list[str]] = {}
        for binding, expected_enabled in ((off, False), (on, True)):
            raw_flags = binding.manifest.get(field)
            expected_flag = (
                f"-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED={int(expected_enabled)}"
            )
            if (
                not isinstance(raw_flags, list)
                or any(not isinstance(flag, str) for flag in raw_flags)
                or raw_flags.count(expected_flag) != 1
            ):
                raise M0AcceptanceError(
                    f"{binding.mode} {field} lacks one exact telemetry definition"
                )
            normalized_by_mode[binding.mode] = [
                "-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED=<MODE>"
                if flag == expected_flag
                else flag
                for flag in raw_flags
            ]
        if normalized_by_mode[off.mode] != normalized_by_mode[on.mode]:
            raise M0AcceptanceError(
                f"OFF/ON {field} differ outside the telemetry definition"
            )
        normalized_flags[field] = normalized_by_mode[off.mode]

    equal_fields: dict[str, Any] = {}
    for field in _BUILD_TWIN_FIELDS:
        left = _path_value(off.manifest, field)
        right = _path_value(on.manifest, field)
        if left != right:
            raise M0AcceptanceError(
                f"OFF/ON build twins differ outside telemetry identity: {field}"
            )
        equal_fields[field] = left
    if off.artifact_sha256 == on.artifact_sha256:
        raise M0AcceptanceError("OFF/ON telemetry artifacts must have distinct hashes")
    return {
        "schema_version": "eiel_option_d_m0_build_twin_contract_v1",
        "equal_field_sha256s": {
            field: _canonical_sha256(value) for field, value in equal_fields.items()
        },
        "off_artifact_sha256": off.artifact_sha256,
        "on_artifact_sha256": on.artifact_sha256,
        "only_intended_build_toggle": "EIEL_OPTION_D_M0_TELEMETRY_ENABLED=0/1",
        "normalized_flag_sha256s": {
            field: _canonical_sha256(flags) for field, flags in normalized_flags.items()
        },
    }


def _canonical_spec(
    *,
    options: HarnessOptions,
    artifacts: Mapping[str, Mapping[str, Any]],
    runtime_root: Path,
    surface: str = SURFACE_ORDINARY,
) -> dict[str, Any]:
    spec = {
        "schema_version": "eiel_campaign_v1",
        "campaign_id": options.campaign_id,
        "description": (
            "Bounded Option D M0 OFF/ON instrumentation acceptance canary for "
            "W32 P256 C512 under one-generation resident windows."
        ),
        "no_claims": (
            "Measurement-only mechanics parity canary; no production route, "
            "performance, tape-removal, B48, or scientific claim."
        ),
        "world": {"base": "patch", "overlays": []},
        "policy": {"family": "fixed_mlp"},
        "selection": {"mode": "standard"},
        "parameters": {
            "patch.grid_size": 32,
            "patch.initial_population": 256,
            "patch.capacity": 512,
            "cuda_architecture_b.cpython_gauss_segment_capacity": 65_536,
            "cuda_architecture_b.policy_birth_segment_refill_enable": True,
            "cuda_architecture_b.policy_birth_segment_capacity": -1,
            "cuda_architecture_b.turnover_mlp_shadow_elide_enable": True,
            "cuda_architecture_b.cpu_request_staging_enable": True,
        },
        "matrix": {
            "dimensions": [
                {
                    "id": "ecology_seed",
                    "description": "One bounded ecology seed.",
                    "values": [options.ecology_seed],
                },
                {
                    "id": "task_seed",
                    "description": "One bounded task seed.",
                    "values": [options.task_seed],
                },
                {
                    "id": "founder_mode",
                    "description": "One fixed founder treatment.",
                    "values": [options.founder_mode],
                },
            ],
            "parameter_bindings": {"founder_mode": "patch.founder_placement"},
            "seed_bindings": {
                "ecology_seed": "ecology",
                "task_seed": "task",
            },
            "include": [],
            "exclude": [],
            "expected_cells": 1,
            "max_cells": 1,
        },
        "execution": {
            "backend": "cuda_resident",
            "generations": TERMINAL_GENERATION,
            "ticks_per_generation": 160,
            "physical_window_generations": 1,
            "batch": {"mode": "automatic", "maximum": 1},
            "budgets": {
                "vram_bytes": 2_000_000_000,
                "disk_bytes": 1_000_000_000,
                "wall_seconds": 600,
            },
            "cancellation": False,
            "technical_gate": {"selectors": [], "partition": True},
        },
        "checkpoints": {
            "milestones": [TERMINAL_GENERATION],
            "digest_every_generation": False,
            "storage_mode": "disk",
        },
        "comparisons": {
            "fresh_repeats": 0,
            "fresh_selectors": [],
            "continuation_splits": [],
            "continuation_cells": 0,
            "continuation_baseline": "fresh_branch",
            "continuation_selectors": [],
            "reporting_selector": None,
        },
        "artifacts": {name: dict(row) for name, row in artifacts.items()},
        "analysis": {
            "required_metrics": ["population.alive"],
            "optional_metrics": [],
            "rules": [
                {
                    "id": "m0-population-coverage",
                    "metric": "population.alive",
                    "aggregate": "mean",
                    "group_by": ["founder_mode"],
                    "pair_by": ["ecology_seed", "task_seed"],
                    "operator": ">=",
                    "threshold": 0,
                    "minimum_coverage": 1.0,
                }
            ],
        },
        "output": {
            "runtime_root": str(runtime_root),
            "reporting_profile": "compact",
            "chronicle_profile": "summary",
            "chronicle_history_root": str(runtime_root / "chronicle-history-disabled"),
            "chronicle_archive_enabled": False,
            "evidence_closeout_enabled": False,
        },
    }
    if surface == SURFACE_ORDINARY:
        return spec
    if surface != SURFACE_TURNOVER:
        raise M0AcceptanceError(f"unknown M0 measurement surface: {surface}")
    spec["campaign_id"] = f"{options.campaign_id}-turnover"
    spec["description"] += (
        " Separate turnover-only lifecycle surface used only to exercise "
        "generation-turnover refill accounting."
    )
    spec["selection"] = {"mode": "genotype_blind_neutral"}
    spec["parameters"].update(
        {
            "production_lifecycle.profile": (
                PATCH_FIXED_MLP_TURNOVER_ONLY_W32P256C512_V1
            ),
            "production_lifecycle.turnover_enabled": True,
            "production_lifecycle.ordinary_births_enabled": False,
            "production_lifecycle.lifecycle_mode": (
                LEGACY_SYNCHRONIZED_TRUNCATION_V1
            ),
            "production_lifecycle.life_history_profile": LIFE_HISTORY_PROFILE_INERT_V1,
            "production_lifecycle.turnover_cull_fraction": 0.5,
            "production_lifecycle.turnover_parent_fraction": 1.0,
        }
    )
    return spec


def _compile_mode_plan(
    *, options: HarnessOptions, binding: ModeBinding, run_root: Path, surface: str
) -> dict[str, Any]:
    spec = _canonical_spec(
        options=options,
        artifacts=binding.artifacts,
        runtime_root=run_root,
        surface=surface,
    )
    plan = compile_campaign(spec).payload()
    expected_campaign_id = (
        options.campaign_id
        if surface == SURFACE_ORDINARY
        else f"{options.campaign_id}-turnover"
    )
    if (
        plan.get("resolved_profile") != PROFILE_ID
        or plan.get("campaign_id") != expected_campaign_id
        or plan.get("execution", {}).get("physical_window_generations") != 1
        or plan.get("execution", {}).get("generations") != TERMINAL_GENERATION
        or plan.get("execution", {}).get("ticks_per_generation") != 160
        or len(plan.get("cells", [])) != 1
        or len(plan.get("physical_batches", [])) != 1
    ):
        raise M0AcceptanceError("compiled M0 plan left its bounded b1g3t160 g1 scope")
    return plan


def _plan_logical_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(plan))
    value.pop("plan_sha256", None)
    value.pop("qualification_key_sha256", None)
    qualification = value.get("qualification_key")
    if not isinstance(qualification, dict):
        raise M0AcceptanceError("compiled plan lacks a qualification key")
    qualification.pop("artifact_hashes", None)
    value.pop("artifacts", None)
    output = value.get("output")
    if not isinstance(output, dict):
        raise M0AcceptanceError("compiled plan lacks output bindings")
    output.pop("runtime_root", None)
    output.pop("chronicle_history_root", None)
    return value


def _validate_plan_pair(
    off_plan: Mapping[str, Any], on_plan: Mapping[str, Any]
) -> dict[str, Any]:
    off_projection = _plan_logical_projection(off_plan)
    on_projection = _plan_logical_projection(on_plan)
    if off_projection != on_projection:
        raise M0AcceptanceError(
            "OFF/ON plans differ outside artifacts, route identity, or output root"
        )
    off_cells = off_plan.get("cells")
    on_cells = on_plan.get("cells")
    if off_cells != on_cells or not isinstance(off_cells, list):
        raise M0AcceptanceError("OFF/ON cell, branch, seed, or RNG identities differ")
    cell = off_cells[0]
    if not isinstance(cell, Mapping) or any(
        not isinstance(cell.get(name), str) or not cell[name]
        for name in ("cell_id", "branch_id", "rng_lineage_id")
    ):
        raise M0AcceptanceError("canonical M0 cell identities are incomplete")
    return {
        "schema_version": "eiel_option_d_m0_logical_plan_pair_v1",
        "logical_projection_sha256": _canonical_sha256(off_projection),
        "cell_id": cell["cell_id"],
        "branch_id": cell["branch_id"],
        "rng_lineage_id": cell["rng_lineage_id"],
        "allowed_differences": [
            "artifacts",
            "qualification_key.artifact_hashes",
            "qualification_key_sha256",
            "plan_sha256",
            "output.runtime_root",
            "output.chronicle_history_root",
            "execution_route_sha256",
            "runtime provenance and timing",
            "Option D telemetry fields",
        ],
    }


def _idle_gpu_receipt() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        sample = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        rows = [row.strip() for row in sample.stdout.splitlines() if row.strip()]
        if len(rows) != 1:
            raise M0AcceptanceError("idle-GPU probe did not return exactly one GPU")
        parts = [part.strip() for part in rows[0].split(",")]
        if len(parts) != 2:
            raise ValueError("unexpected nvidia-smi row")
        utilization = int(parts[0])
        memory_used_mib = int(parts[1])
        compute = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise M0AcceptanceError(f"idle-GPU probe failed: {exc}") from exc
    active_compute_rows = [
        row.strip() for row in compute.stdout.splitlines() if row.strip()
    ]
    allowed_background_rows: list[str] = []
    blocking_compute_rows: list[str] = []
    for row in active_compute_rows:
        parts = [part.strip() for part in row.split(",")]
        process_name = Path(parts[1]).name if len(parts) >= 2 else ""
        if process_name in _IDLE_GPU_BACKGROUND_PROCESSES:
            allowed_background_rows.append(row)
        else:
            blocking_compute_rows.append(row)
    idle = (
        utilization <= _IDLE_GPU_UTILIZATION_THRESHOLD_PERCENT
        and not blocking_compute_rows
    )
    receipt = {
        "schema_version": "eiel_option_d_m0_idle_gpu_probe_v1",
        "utilization_percent": utilization,
        "memory_used_mib": memory_used_mib,
        "active_compute_rows": active_compute_rows,
        "allowed_background_processes": sorted(_IDLE_GPU_BACKGROUND_PROCESSES),
        "allowed_background_rows": allowed_background_rows,
        "blocking_compute_rows": blocking_compute_rows,
        "idle_threshold_percent": _IDLE_GPU_UTILIZATION_THRESHOLD_PERCENT,
        "idle": idle,
    }
    if not idle:
        raise M0AcceptanceError(
            f"GPU is not idle enough for sequential M0 overhead measurement: {receipt}"
        )
    return receipt


def _default_worker_runner(
    *,
    mode: str,
    plan_path: Path,
    run_root: Path,
    worker_result_path: Path,
    python_executable: Path,
    timeout_seconds: float,
) -> WorkerInvocation:
    command = (
        str(python_executable),
        "-m",
        "tools.scripts.eiel_option_d_m0_acceptance_v1",
        "--_worker-mode",
        mode,
        "--_worker-plan",
        str(plan_path),
        "--_worker-run-root",
        str(run_root),
        "--_worker-result",
        str(worker_result_path),
    )
    environment = dict(os.environ)
    if environment.get("EIEL_NATIVE_RUNTIME_EXTENSION"):
        raise M0AcceptanceError(
            "EIEL_NATIVE_RUNTIME_EXTENSION must be unset for artifact-bound M0 twins"
        )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise M0AcceptanceError(
            f"{mode} worker exceeded {timeout_seconds:g} seconds"
        ) from exc
    elapsed = time.perf_counter_ns() - started
    if completed.returncode != 0:
        raise M0AcceptanceError(
            f"{mode} worker failed with exit {completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    worker = _read_object(worker_result_path, name=f"{mode} worker result")
    worker_elapsed = worker.get("execution_wall_time_ns")
    if type(worker_elapsed) is not int or worker_elapsed <= 0:
        raise M0AcceptanceError(f"{mode} worker did not report positive execution time")
    return WorkerInvocation(
        mode=mode,
        command=command,
        subprocess_wall_time_ns=elapsed,
        worker_execution_wall_time_ns=worker_elapsed,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _route_publication_paths(run_root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(run_root))
        for path in run_root.rglob("*.json")
        if path.name in _ROUTE_PUBLICATION_NAMES
        or path.name.endswith(".execution_route_qualification.json")
    )


def _load_mode_result(
    *,
    mode: str,
    run_root: Path,
    plan: Mapping[str, Any],
    invocation: WorkerInvocation,
) -> ModeResult:
    worker_path = run_root / "worker_result.json"
    worker = _read_object(worker_path, name=f"{mode} worker result")
    preflight_path = run_root / "preflight_receipt.json"
    preflight = _read_object(preflight_path, name=f"{mode} preflight receipt")
    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or len(batches) != 1:
        raise M0AcceptanceError(f"{mode} plan is not one physical batch")
    batch = batches[0]
    batch_id = batch.get("batch_id") if isinstance(batch, Mapping) else None
    if not isinstance(batch_id, str) or not batch_id:
        raise M0AcceptanceError(f"{mode} plan batch identity is invalid")
    batch_path = run_root / "batches" / batch_id / "manifest.json"
    batch_manifest = _read_object(batch_path, name=f"{mode} batch manifest")
    execution = worker.get("execution_receipt")
    publication_paths = _route_publication_paths(run_root)
    if (
        worker.get("schema_version") != WORKER_SCHEMA_VERSION
        or worker.get("mode") != mode
        or worker.get("ok") is not True
        or worker.get("plan_sha256") != plan.get("plan_sha256")
        or not isinstance(execution, Mapping)
        or execution.get("ok") is not True
        or execution.get("status") != "completed"
        or preflight.get("qualification_mode") is not True
        or preflight.get("ready_for_route_qualification") is not True
        or preflight.get("ready_for_execution") is not False
        or preflight.get("campaign_id") != plan.get("campaign_id")
        or preflight.get("plan_sha256") != plan.get("plan_sha256")
        or batch_manifest.get("schema_version") != "eiel_campaign_batch_manifest_v1"
        or batch_manifest.get("plan_sha256") != plan.get("plan_sha256")
        or batch_manifest.get("campaign_id") != plan.get("campaign_id")
        or batch_manifest.get("batch_id") != batch_id
        or batch_manifest.get("status") != "completed"
        or publication_paths
    ):
        raise M0AcceptanceError(
            f"{mode} worker evidence is incomplete or published route authority: "
            f"{publication_paths}"
        )
    return ModeResult(
        mode=mode,
        run_root=run_root,
        plan=plan,
        worker=worker,
        preflight=preflight,
        batch_manifest=batch_manifest,
        invocation=invocation,
    )


def _digest_map(
    checkpoint: Mapping[str, Any], *, field: str, cell_ids: set[str]
) -> dict[str, str]:
    raw = checkpoint.get(field)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != cell_ids
        or any(
            not isinstance(key, str) or not _valid_sha256(value)
            for key, value in raw.items()
        )
    ):
        raise M0AcceptanceError(f"checkpoint field {field} is not an exact digest map")
    return dict(sorted(raw.items()))


def _mechanics_surface(result: ModeResult) -> dict[str, Any]:
    manifest = result.batch_manifest
    checkpoint = manifest.get("checkpoint")
    native_result = manifest.get("result")
    metrics_rows = manifest.get("metrics_rows")
    cells = result.plan.get("cells")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(native_result, Mapping)
        or not isinstance(metrics_rows, list)
        or not all(isinstance(row, Mapping) for row in metrics_rows)
        or not isinstance(cells, list)
    ):
        raise M0AcceptanceError(
            f"{result.mode} lacks checkpoint/result/metrics surfaces"
        )
    cell_ids = {str(row.get("cell_id")) for row in cells if isinstance(row, Mapping)}
    if len(cell_ids) != 1 or "None" in cell_ids:
        raise M0AcceptanceError(f"{result.mode} plan cell identities are invalid")
    if (
        checkpoint.get("generation") != TERMINAL_GENERATION
        or native_result.get("terminal_generation") != TERMINAL_GENERATION
    ):
        raise M0AcceptanceError(f"{result.mode} did not settle generation 3")
    return {
        "checkpoint": {
            field: _digest_map(checkpoint, field=field, cell_ids=cell_ids)
            for field in _PARITY_CHECKPOINT_FIELDS
        },
        "terminal_generation": native_result["terminal_generation"],
        "metrics_rows": [dict(row) for row in metrics_rows],
    }


def _native_receipts(result: ModeResult) -> list[dict[str, Any]]:
    native_result = result.batch_manifest.get("result")
    rows = (
        native_result.get("native_execution_receipts")
        if isinstance(native_result, Mapping)
        else None
    )
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise M0AcceptanceError(
            f"{result.mode} must contain one exact native g3 segment receipt"
        )
    return [dict(row) for row in rows]


def _validate_telemetry_pair(
    off: ModeResult,
    on: ModeResult,
    *,
    require_ordinary: bool,
    require_turnover: bool,
    surface: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    off_native = _native_receipts(off)[0]
    on_native = _native_receipts(on)[0]
    if (
        off_native.get("option_d_m0_telemetry_enabled") is not False
        or "option_d_rng_instrumentation_receipt" in off_native
        or on_native.get("option_d_m0_telemetry_enabled") is not True
    ):
        raise M0AcceptanceError("OFF/ON runtime telemetry enablement is not exact")
    primary = on_native.get("option_d_rng_instrumentation_receipt")
    if not isinstance(primary, Mapping):
        raise M0AcceptanceError("telemetry-on lacks its provider-finalized M0 receipt")
    binding = primary.get("runtime_plan_binding")
    ordinary = primary.get("actual_ordinary_birth_count")
    turnover = primary.get("actual_turnover_birth_count")
    if (
        primary.get("schema_version") != PRIMARY_RECEIPT_SCHEMA_VERSION
        or primary.get("option_d_m0_telemetry_enabled") is not True
        or type(ordinary) is not int
        or (require_ordinary and ordinary <= 0)
        or type(turnover) is not int
        or (require_turnover and turnover <= 0)
        or not isinstance(binding, Mapping)
        or binding.get("status") != "PASS_EXACT_PROVIDER_RUNTIME_PLAN_BINDING"
        or binding.get("physical_window_generations") != 1
        or binding.get("requested_generations") != TERMINAL_GENERATION
        or binding.get("ticks_per_generation") != 160
        or binding.get("world_grid_size") != 32
        or binding.get("production_integration_authority") is not False
    ):
        raise M0AcceptanceError(
            f"telemetry-on {surface} receipt lacks exact g1 scope or required "
            "birth class"
        )
    return on_native, dict(primary)


def _required_fields(
    value: Mapping[str, Any], names: Sequence[str], *, surface: str
) -> dict[str, Any]:
    missing = [name for name in names if name not in value]
    if missing:
        raise M0AcceptanceError(f"{surface} lacks fields: {', '.join(missing)}")
    selected = {name: value[name] for name in names}
    for name, item in selected.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise M0AcceptanceError(f"{surface}.{name} must be a nonnegative integer")
    return selected


def _receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["receipt_sha256"] = _canonical_sha256(value)
    return value


def _artifact_evidence(result: ModeResult, binding: ModeBinding) -> dict[str, Any]:
    return {
        "plan_path": str(result.run_root / "execution_plan.json"),
        "plan_sha256": result.plan["plan_sha256"],
        "preflight_path": str(result.run_root / "preflight_receipt.json"),
        "preflight_sha256": _sha256_file(result.run_root / "preflight_receipt.json"),
        "batch_manifest_path": str(
            result.run_root / "batches" / "batch-0000" / "manifest.json"
        ),
        "batch_manifest_sha256": _sha256_file(
            result.run_root / "batches" / "batch-0000" / "manifest.json"
        ),
        "worker_result_path": str(result.run_root / "worker_result.json"),
        "worker_result_sha256": _sha256_file(result.run_root / "worker_result.json"),
        "artifact_path": str(binding.artifact_path),
        "artifact_sha256": binding.artifact_sha256,
        "manifest_path": str(binding.manifest_path),
        "manifest_sha256": binding.manifest_sha256,
        "execution_route_sha256": result.worker["execution_receipt"].get(
            "execution_route_sha256"
        ),
    }


def _write_success_receipts(
    *,
    output_root: Path,
    off: ModeResult,
    on: ModeResult,
    turnover_off: ModeResult,
    turnover_on: ModeResult,
    off_binding: ModeBinding,
    on_binding: ModeBinding,
    build_twin_contract: Mapping[str, Any],
    plan_pair_contract: Mapping[str, Any],
    turnover_plan_pair_contract: Mapping[str, Any],
    idle_gpu: Mapping[str, Any] | None,
) -> dict[str, Any]:
    off_surface = _mechanics_surface(off)
    on_surface = _mechanics_surface(on)
    if off_surface != on_surface:
        raise M0AcceptanceError(
            "OFF/ON terminal, source-import, checkpoint, or metrics parity differs"
        )
    turnover_off_surface = _mechanics_surface(turnover_off)
    turnover_on_surface = _mechanics_surface(turnover_on)
    if turnover_off_surface != turnover_on_surface:
        raise M0AcceptanceError(
            "turnover OFF/ON terminal, source-import, checkpoint, or metrics "
            "parity differs"
        )
    on_native, primary = _validate_telemetry_pair(
        off,
        on,
        require_ordinary=True,
        require_turnover=False,
        surface=SURFACE_ORDINARY,
    )
    _turnover_native, turnover_primary = _validate_telemetry_pair(
        turnover_off,
        turnover_on,
        require_ordinary=False,
        require_turnover=True,
        surface=SURFACE_TURNOVER,
    )
    native_components = _required_fields(
        on_native, _NATIVE_COMPONENT_FIELDS, surface="telemetry-on native receipt"
    )
    primary_components = _required_fields(
        primary, _PRIMARY_COMPONENT_FIELDS, surface="telemetry-on primary receipt"
    )
    cpu_policy_summary = primary.get("cpu_policy_service_compact_summary")
    resident_accounting = primary.get("resident_window_transfer_accounting")
    if not isinstance(cpu_policy_summary, Mapping) or not isinstance(
        resident_accounting, Mapping
    ):
        raise M0AcceptanceError(
            "telemetry-on primary receipt lacks compact policy or resident accounting"
        )
    component = _receipt(
        {
            "schema_version": COMPONENT_SCHEMA_VERSION,
            "status": "PASS_M0_MEASUREMENT_ONLY_COMPONENT_PROFILE",
            "measurement_only": True,
            "optimization_performed": False,
            "production_integration_authority": False,
            "route_qualification_authority": False,
            "scope": "one physical b1g3t160 W32/P256/C512 g1 native segment",
            "native_component_counters_and_times": native_components,
            "provider_component_counters_and_memory": primary_components,
            "cpu_policy_service_compact_summary": dict(cpu_policy_summary),
            "resident_window_transfer_accounting": dict(resident_accounting),
            "policy_refill_receipt_count": len(
                on_native.get("policy_birth_segment_refill_receipts", [])
            ),
            "gaussian_refill_receipt_count": len(
                on_native.get("gaussian_segment_refill_receipts", [])
            ),
        }
    )
    atomic_write_json(output_root / "m0_component_profile.json", component)

    off_subprocess = off.invocation.subprocess_wall_time_ns
    on_subprocess = on.invocation.subprocess_wall_time_ns
    off_execution = off.invocation.worker_execution_wall_time_ns
    on_execution = on.invocation.worker_execution_wall_time_ns
    overhead = _receipt(
        {
            "schema_version": OVERHEAD_SCHEMA_VERSION,
            "status": "PASS_M0_SINGLE_PAIR_DESCRIPTIVE_OVERHEAD",
            "measurement_only": True,
            "optimization_performed": False,
            "production_performance_claim": False,
            "route_qualification_authority": False,
            "timing_scope": (
                "one sequential fresh-process OFF/ON b1g3t160 pair; descriptive "
                "only, not repeated performance qualification"
            ),
            "idle_gpu_required": idle_gpu is not None,
            "idle_gpu_receipt": None if idle_gpu is None else dict(idle_gpu),
            "telemetry_off_subprocess_wall_time_ns": off_subprocess,
            "telemetry_on_subprocess_wall_time_ns": on_subprocess,
            "subprocess_wall_time_delta_ns": on_subprocess - off_subprocess,
            "subprocess_wall_time_ratio": on_subprocess / off_subprocess,
            "telemetry_off_worker_execution_wall_time_ns": off_execution,
            "telemetry_on_worker_execution_wall_time_ns": on_execution,
            "worker_execution_wall_time_delta_ns": on_execution - off_execution,
            "worker_execution_wall_time_ratio": on_execution / off_execution,
            "component_profile_path": str(output_root / "m0_component_profile.json"),
            "component_profile_sha256": _sha256_file(
                output_root / "m0_component_profile.json"
            ),
        }
    )
    atomic_write_json(
        output_root / "m0_instrumentation_overhead_receipt.json", overhead
    )

    surface_sha256s = {
        SURFACE_ORDINARY: {
            "source_import_projection_digest_sha256s": _canonical_sha256(
                off_surface["checkpoint"]["source_import_projection_digest_sha256s"]
            ),
            "terminal_projection_digest_sha256s": _canonical_sha256(
                off_surface["checkpoint"]["terminal_projection_digest_sha256s"]
            ),
            "canonical_terminal_state_digest_sha256s": _canonical_sha256(
                off_surface["checkpoint"]["canonical_terminal_state_digest_sha256s"]
            ),
            "canonical_checkpoint_storage_digest_sha256s": _canonical_sha256(
                off_surface["checkpoint"]["canonical_checkpoint_storage_digest_sha256s"]
            ),
            "metrics_rows": _canonical_sha256(off_surface["metrics_rows"]),
        },
        SURFACE_TURNOVER: {
            "source_import_projection_digest_sha256s": _canonical_sha256(
                turnover_off_surface["checkpoint"][
                    "source_import_projection_digest_sha256s"
                ]
            ),
            "terminal_projection_digest_sha256s": _canonical_sha256(
                turnover_off_surface["checkpoint"][
                    "terminal_projection_digest_sha256s"
                ]
            ),
            "canonical_terminal_state_digest_sha256s": _canonical_sha256(
                turnover_off_surface["checkpoint"][
                    "canonical_terminal_state_digest_sha256s"
                ]
            ),
            "canonical_checkpoint_storage_digest_sha256s": _canonical_sha256(
                turnover_off_surface["checkpoint"][
                    "canonical_checkpoint_storage_digest_sha256s"
                ]
            ),
            "metrics_rows": _canonical_sha256(turnover_off_surface["metrics_rows"]),
        },
    }
    acceptance = _receipt(
        {
            "schema_version": ACCEPTANCE_SCHEMA_VERSION,
            "status": "PASS_M0_MEASUREMENT_ONLY_ACCEPTANCE",
            "decision": "M0-A",
            "passed": True,
            "measurement_authority": True,
            "production_integration_authority": False,
            "route_qualification_authority": False,
            "route_qualification_published": False,
            "optimization_performed": False,
            "performance_claim": False,
            "tape_elimination_claim": False,
            "b48_claim": False,
            "harness_schema_version": SCHEMA_VERSION,
            "harness_source_path": str(Path(__file__).resolve()),
            "harness_source_sha256": _sha256_file(Path(__file__).resolve()),
            "build_twin_contract": dict(build_twin_contract),
            "logical_plan_pair_contract": dict(plan_pair_contract),
            "turnover_logical_plan_pair_contract": dict(turnover_plan_pair_contract),
            "compared_surface_sha256s": surface_sha256s,
            "exact_parity": {
                "source_import_projection_digest_sha256s": True,
                "terminal_projection_digest_sha256s": True,
                "canonical_terminal_state_digest_sha256s": True,
                "canonical_checkpoint_storage_digest_sha256s": True,
                "terminal_generation": True,
                "metrics_rows": True,
            },
            "telemetry_on_birth_evidence": {
                "actual_ordinary_birth_count": primary["actual_ordinary_birth_count"],
                "ordinary_surface_turnover_birth_count": primary[
                    "actual_turnover_birth_count"
                ],
                "turnover_surface_ordinary_birth_count": turnover_primary[
                    "actual_ordinary_birth_count"
                ],
                "actual_turnover_birth_count": turnover_primary[
                    "actual_turnover_birth_count"
                ],
                "both_required_and_observed": True,
                "birth_surfaces": {
                    SURFACE_ORDINARY: {
                        "actual_ordinary_birth_count": primary[
                            "actual_ordinary_birth_count"
                        ],
                        "actual_turnover_birth_count": primary[
                            "actual_turnover_birth_count"
                        ],
                    },
                    SURFACE_TURNOVER: {
                        "actual_ordinary_birth_count": turnover_primary[
                            "actual_ordinary_birth_count"
                        ],
                        "actual_turnover_birth_count": turnover_primary[
                            "actual_turnover_birth_count"
                        ],
                    },
                },
            },
            "telemetry_off": _artifact_evidence(off, off_binding),
            "telemetry_on": _artifact_evidence(on, on_binding),
            "turnover_telemetry_off": _artifact_evidence(
                turnover_off, off_binding
            ),
            "turnover_telemetry_on": _artifact_evidence(
                turnover_on, on_binding
            ),
            "overhead_receipt_path": str(
                output_root / "m0_instrumentation_overhead_receipt.json"
            ),
            "overhead_receipt_sha256": _sha256_file(
                output_root / "m0_instrumentation_overhead_receipt.json"
            ),
            "component_profile_path": str(output_root / "m0_component_profile.json"),
            "component_profile_sha256": _sha256_file(
                output_root / "m0_component_profile.json"
            ),
        }
    )
    atomic_write_json(output_root / "m0_acceptance_receipt.json", acceptance)
    return acceptance


def _write_failure_receipts(output_root: Path, exc: BaseException) -> None:
    error = {"type": type(exc).__name__, "message": str(exc)}
    rows = (
        (
            "m0_component_profile.json",
            COMPONENT_SCHEMA_VERSION,
            "FAIL_M0_COMPONENT_PROFILE_WITHHELD",
        ),
        (
            "m0_instrumentation_overhead_receipt.json",
            OVERHEAD_SCHEMA_VERSION,
            "FAIL_M0_OVERHEAD_ACCEPTANCE_WITHHELD",
        ),
        (
            "m0_acceptance_receipt.json",
            ACCEPTANCE_SCHEMA_VERSION,
            "FAIL_M0_ACCEPTANCE_WITHHELD",
        ),
    )
    for filename, schema, status in rows:
        payload = _receipt(
            {
                "schema_version": schema,
                "status": status,
                "passed": False,
                "measurement_authority": False,
                "production_integration_authority": False,
                "route_qualification_authority": False,
                "route_qualification_published": False,
                "optimization_performed": False,
                "error": error,
            }
        )
        atomic_write_json(output_root / filename, payload)


def _run_surface_pair(
    *,
    validated: HarnessOptions,
    output_root: Path,
    surface: str,
    off_binding: ModeBinding,
    on_binding: ModeBinding,
    runner: WorkerRunner,
) -> tuple[dict[str, ModeResult], dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for mode, binding in (
        ("telemetry-off", off_binding),
        ("telemetry-on", on_binding),
    ):
        run_root = output_root / mode
        run_root.mkdir(parents=False, exist_ok=False)
        plan = _compile_mode_plan(
            options=validated,
            binding=binding,
            run_root=run_root,
            surface=surface,
        )
        atomic_write_json(run_root / "execution_plan.json", plan)
        plans[mode] = plan
    plan_pair_contract = _validate_plan_pair(
        plans["telemetry-off"], plans["telemetry-on"]
    )
    mode_results: dict[str, ModeResult] = {}
    for mode in MODES:
        run_root = output_root / mode
        invocation = runner(
            mode=mode,
            plan_path=run_root / "execution_plan.json",
            run_root=run_root,
            worker_result_path=run_root / "worker_result.json",
            python_executable=validated.python_executable,
            timeout_seconds=validated.worker_timeout_seconds,
        )
        mode_results[mode] = _load_mode_result(
            mode=mode,
            run_root=run_root,
            plan=plans[mode],
            invocation=invocation,
        )
    return mode_results, plan_pair_contract


def _validated_options(options: HarnessOptions) -> HarnessOptions:
    output_root = options.output_root.expanduser().resolve()
    if "evidences" in output_root.parts:
        raise M0AcceptanceError("raw M0 acceptance output must not be in evidences")
    if output_root.exists():
        raise M0AcceptanceError(f"output root already exists: {output_root}")
    if (
        not options.campaign_id
        or type(options.ecology_seed) is not int
        or options.ecology_seed < 0
        or type(options.task_seed) is not int
        or options.task_seed < 0
        or options.founder_mode != "legacy"
        or options.worker_timeout_seconds <= 0
    ):
        raise M0AcceptanceError(
            "M0 campaign identity, seeds, founder mode, or timeout is invalid"
        )
    python_executable = options.python_executable.expanduser().absolute()
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise M0AcceptanceError(
            f"worker Python executable is not executable: {python_executable}"
        )
    return HarnessOptions(
        off_build_dir=options.off_build_dir,
        on_build_dir=options.on_build_dir,
        off_receipt_dir=options.off_receipt_dir,
        on_receipt_dir=options.on_receipt_dir,
        on_source_environment=options.on_source_environment,
        output_root=output_root,
        campaign_id=options.campaign_id,
        ecology_seed=options.ecology_seed,
        task_seed=options.task_seed,
        founder_mode=options.founder_mode,
        python_executable=python_executable,
        worker_timeout_seconds=float(options.worker_timeout_seconds),
        require_idle_gpu=bool(options.require_idle_gpu),
    )


def run_harness(
    options: HarnessOptions,
    *,
    worker_runner: WorkerRunner | None = None,
) -> dict[str, Any]:
    validated = _validated_options(options)
    validated.output_root.mkdir(parents=True, exist_ok=False)
    runner = worker_runner or _default_worker_runner
    try:
        idle_gpu = _idle_gpu_receipt() if validated.require_idle_gpu else None
        off_binding = _mode_binding(
            mode="telemetry-off",
            build_dir=validated.off_build_dir,
            receipt_dir=validated.off_receipt_dir,
            source_environment_path=None,
        )
        on_binding = _mode_binding(
            mode="telemetry-on",
            build_dir=validated.on_build_dir,
            receipt_dir=validated.on_receipt_dir,
            source_environment_path=validated.on_source_environment,
        )
        build_twin_contract = _validate_build_twins(off_binding, on_binding)
        mode_results, plan_pair_contract = _run_surface_pair(
            validated=validated,
            output_root=validated.output_root,
            surface=SURFACE_ORDINARY,
            off_binding=off_binding,
            on_binding=on_binding,
            runner=runner,
        )
        turnover_output_root = validated.output_root / SURFACE_TURNOVER
        turnover_output_root.mkdir(parents=False, exist_ok=False)
        turnover_results, turnover_plan_pair_contract = _run_surface_pair(
            validated=validated,
            output_root=turnover_output_root,
            surface=SURFACE_TURNOVER,
            off_binding=off_binding,
            on_binding=on_binding,
            runner=runner,
        )
        return _write_success_receipts(
            output_root=validated.output_root,
            off=mode_results["telemetry-off"],
            on=mode_results["telemetry-on"],
            turnover_off=turnover_results["telemetry-off"],
            turnover_on=turnover_results["telemetry-on"],
            off_binding=off_binding,
            on_binding=on_binding,
            build_twin_contract=build_twin_contract,
            plan_pair_contract=plan_pair_contract,
            turnover_plan_pair_contract=turnover_plan_pair_contract,
            idle_gpu=idle_gpu,
        )
    except Exception as exc:
        _write_failure_receipts(validated.output_root, exc)
        raise


def _execute_worker(
    *,
    mode: str,
    plan_path: Path,
    run_root: Path,
    worker_result_path: Path,
    provider_registry_factory: Callable[[], Any] = default_execution_provider_registry,
    preflight_runner: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] = (
        bounded_route_qualification_preflight_v1
    ),
    campaign_executor: Callable[..., Mapping[str, Any]] = execute_campaign,
) -> dict[str, Any]:
    plan = _read_object(
        _existing_file(plan_path, name="worker plan"), name="worker plan"
    )
    unsigned = dict(plan)
    claimed_plan_sha256 = unsigned.pop("plan_sha256", None)
    if (
        not _valid_sha256(claimed_plan_sha256)
        or _canonical_sha256(unsigned) != claimed_plan_sha256
    ):
        raise M0AcceptanceError("worker plan SHA-256 is invalid")
    resolved_root = run_root.expanduser().resolve()
    if resolved_root != plan_path.expanduser().resolve().parent:
        raise M0AcceptanceError("worker run root differs from its plan directory")
    if (
        worker_result_path.expanduser().resolve()
        != resolved_root / "worker_result.json"
    ):
        raise M0AcceptanceError("worker result path is not canonical")
    if _route_publication_paths(resolved_root):
        raise M0AcceptanceError("worker run root already contains route publication")
    registry = provider_registry_factory()
    provider = registry.get(str(plan.get("resolved_profile", "")))
    if provider is None:
        raise M0AcceptanceError("worker has no selected production provider")
    preflight = dict(preflight_runner(provider, plan))
    if (
        preflight.get("qualification_mode") is not True
        or preflight.get("ready_for_route_qualification") is not True
        or preflight.get("ready_for_execution") is not False
    ):
        raise M0AcceptanceError("bounded preflight withheld canary dispatch")
    atomic_write_json(resolved_root / "preflight_receipt.json", preflight)
    started = time.perf_counter_ns()
    execution = dict(
        campaign_executor(
            plan,
            run_root=resolved_root,
            provider_registry=registry,
            qualification_mode=True,
        )
    )
    execution_wall_time_ns = time.perf_counter_ns() - started
    if execution.get("ok") is not True or execution.get("status") != "completed":
        raise M0AcceptanceError("qualification-mode campaign did not complete")
    publication_paths = _route_publication_paths(resolved_root)
    if publication_paths:
        raise M0AcceptanceError(
            f"qualification-mode worker published route authority: {publication_paths}"
        )
    result = _receipt(
        {
            "schema_version": WORKER_SCHEMA_VERSION,
            "mode": mode,
            "ok": True,
            "plan_sha256": claimed_plan_sha256,
            "preflight_path": str(resolved_root / "preflight_receipt.json"),
            "preflight_sha256": _sha256_file(resolved_root / "preflight_receipt.json"),
            "execution_wall_time_ns": execution_wall_time_ns,
            "execution_receipt": execution,
            "qualification_mode": True,
            "route_qualification_published": False,
            "production_integration_authority": False,
        }
    )
    atomic_write_json(worker_result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-build-dir", type=Path)
    parser.add_argument("--on-build-dir", type=Path)
    parser.add_argument("--off-receipt-dir", type=Path)
    parser.add_argument("--on-receipt-dir", type=Path)
    parser.add_argument("--on-source-environment", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--campaign-id", default="option-d-m0-b1g3t160-parity-a86bf726a"
    )
    parser.add_argument("--ecology-seed", type=int, default=303)
    parser.add_argument("--task-seed", type=int, default=1001)
    parser.add_argument("--founder-mode", default="legacy")
    parser.add_argument(
        "--python", dest="python_executable", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--require-idle-gpu", action="store_true")
    parser.add_argument(
        "--require-ordinary-births",
        action="store_true",
        help="accepted compatibility flag; ordinary births are always required",
    )
    parser.add_argument(
        "--require-turnover-births",
        action="store_true",
        help="accepted compatibility flag; turnover births are always required",
    )
    parser.add_argument("--_worker-mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-run-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def _required_parent_path(value: Path | None, *, option: str) -> Path:
    if value is None:
        raise M0AcceptanceError(f"{option} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args._worker_mode is not None:
            if any(
                value is None
                for value in (
                    args._worker_plan,
                    args._worker_run_root,
                    args._worker_result,
                )
            ):
                raise M0AcceptanceError("worker paths are required")
            _execute_worker(
                mode=args._worker_mode,
                plan_path=args._worker_plan,
                run_root=args._worker_run_root,
                worker_result_path=args._worker_result,
            )
            return 0
        options = HarnessOptions(
            off_build_dir=_required_parent_path(
                args.off_build_dir, option="--off-build-dir"
            ),
            on_build_dir=_required_parent_path(
                args.on_build_dir, option="--on-build-dir"
            ),
            off_receipt_dir=_required_parent_path(
                args.off_receipt_dir, option="--off-receipt-dir"
            ),
            on_receipt_dir=_required_parent_path(
                args.on_receipt_dir, option="--on-receipt-dir"
            ),
            on_source_environment=_required_parent_path(
                args.on_source_environment, option="--on-source-environment"
            ),
            output_root=_required_parent_path(args.output_root, option="--output-root"),
            campaign_id=args.campaign_id,
            ecology_seed=args.ecology_seed,
            task_seed=args.task_seed,
            founder_mode=args.founder_mode,
            python_executable=args.python_executable,
            worker_timeout_seconds=args.worker_timeout_seconds,
            require_idle_gpu=args.require_idle_gpu,
        )
        receipt = run_harness(options)
    except Exception as exc:
        print(
            f"M0 acceptance failed closed: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
