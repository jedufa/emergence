#!/usr/bin/env python3
"""Bounded real-runtime QualPrep probe for the non-EED WDL/Jungle route.

This executes B1, B2, or one physical B48 batch for three to 128 generations
using the exact B48 configuration and isolated artifact. It is development
evidence, never a production-admission or qualification receipt.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.compiler import compile_campaign, load_campaign_spec  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (  # noqa: E402
    _wdl_source_capabilities_v2,
)
from engine.perf.cxxbackbone10_resident_state_owner import (  # noqa: E402
    projection_digest,
)
from tools.scripts.eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1 import (  # noqa: E402
    AllocationDiagnosticError,
    AllocationOnlyProvider,
    _diagnostic_descriptor,
    _diagnostic_plan,
    _load_exact_module,
    _sha256,
)


SCHEMA_VERSION = "eiel_b48_non_eed_wdl_jungle_qualprep_runtime_v1"
HAPB_ARTIFACT_ABI = "architecture_b_wdl_jungle_eed_hapb_fixed_mlp_native_v1"


def _git_head() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _runtime_plan(
    plan: Mapping[str, Any],
    *,
    cells: int,
    cell_offset: int = 0,
    selected_cell_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    from engine.perf.hapb_physical_extent_v1 import flatten_plan_cell_ids_v1
    from engine.perf.cuda_architecture_b_hapb_v1 import hapb_physical_cell_count_v1

    try:
        available = flatten_plan_cell_ids_v1(plan)
        count = hapb_physical_cell_count_v1(cells)
    except ValueError as exc:
        raise AllocationDiagnosticError(str(exc)) from exc
    if selected_cell_ids is not None:
        chosen = tuple(str(cell_id) for cell_id in selected_cell_ids)
        if len(chosen) != count:
            raise AllocationDiagnosticError(
                "QualPrep selected cell identities do not match physical B"
            )
        available_set = set(available)
        if any(cell_id not in available_set for cell_id in chosen):
            raise AllocationDiagnosticError(
                "QualPrep selected cell identities are outside the compiled plan"
            )
    else:
        if type(cell_offset) is not int or cell_offset < 0:
            raise AllocationDiagnosticError("QualPrep cell offset is invalid")
        if cell_offset + count > len(available):
            raise AllocationDiagnosticError(
                "QualPrep cell offset is outside the compiled physical extent"
            )
        chosen = tuple(available[cell_offset : cell_offset + count])
    result = dict(plan)
    result["physical_batches"] = [
        {
            "batch_id": f"qualprep-b{count:02d}-o{cell_offset:02d}",
            "cell_ids": chosen,
        }
    ]
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _bind_diagnostic_artifact(
    plan: Mapping[str, Any],
    *,
    artifact_override: Path | None,
    manifest_override: Path | None,
) -> dict[str, Any]:
    """Bind an independently materialized non-EED artifact to a probe only."""

    if (artifact_override is None) != (manifest_override is None):
        raise AllocationDiagnosticError(
            "diagnostic artifact and manifest overrides must be supplied together"
        )
    result = dict(plan)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AllocationDiagnosticError("runtime plan lacks artifact bindings")
    bound = {name: dict(value) for name, value in artifacts.items()}
    if artifact_override is not None and manifest_override is not None:
        artifact = artifact_override.expanduser().resolve()
        manifest = manifest_override.expanduser().resolve()
        if not artifact.is_file() or not manifest.is_file():
            raise AllocationDiagnosticError("diagnostic artifact or manifest is absent")
        bound["cuda_extension"].update(
            {"path": str(artifact), "sha256": _sha256(artifact)}
        )
        bound["extension_manifest"].update(
            {"path": str(manifest), "sha256": _sha256(manifest)}
        )
    result["artifacts"] = bound
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _preflight_external_artifact_bindings_v1(
    plan: Mapping[str, Any],
    *,
    skipped_artifact_ids: frozenset[str] = frozenset({
        "cuda_extension",
        "extension_manifest",
        "qualification_receipt",
        "reporting_receipt",
    }),
) -> None:
    """Fail before CUDA/output creation when a declared external input is stale.

    Diagnostic probes replace the native and qualification/reporting bindings
    with their explicit overrides/descriptors.  Every other declared artifact
    is an execution dependency and must exist with its receipt-bound digest.
    Keeping this check before output-root creation prevents an avoidable native
    launch and avoids leaving a misleading partial root for a missing package.
    """

    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AllocationDiagnosticError("runtime plan lacks artifact bindings")
    failures: list[str] = []
    for artifact_id, raw_row in artifacts.items():
        if artifact_id in skipped_artifact_ids:
            continue
        if not isinstance(raw_row, Mapping):
            failures.append(f"{artifact_id}: binding is not an object")
            continue
        raw_path, expected_sha = raw_row.get("path"), raw_row.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            failures.append(f"{artifact_id}: path is absent")
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            failures.append(f"{artifact_id}: missing {path}")
            continue
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            failures.append(f"{artifact_id}: digest is absent")
            continue
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            failures.append(
                f"{artifact_id}: digest mismatch expected {expected_sha} got {actual_sha}"
            )
    if failures:
        raise AllocationDiagnosticError(
            "external artifact preflight failed before CUDA/output creation: "
            + "; ".join(failures)
        )


def _bind_hapb_artifact_abi(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the direct HAPB binding distinct from the resident CUDA ABI."""

    result = dict(plan)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AllocationDiagnosticError("runtime plan lacks artifact bindings")
    bound = {name: dict(value) for name, value in artifacts.items()}
    for artifact_id in ("cuda_extension", "extension_manifest"):
        row = bound.get(artifact_id)
        if not isinstance(row, dict):
            raise AllocationDiagnosticError("HAPB runtime lacks an artifact binding")
        row["abi"] = HAPB_ARTIFACT_ABI
    result["artifacts"] = bound
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _execution_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce native output to a JSON receipt without discarding its contract."""

    scalar_names = (
        "wdl_jungle_native_owner",
        "wdl_native_controller_abi_version",
        "jungle_native_controller_abi_version",
        "eed_native_owner",
        "eed_native_controller_abi_version",
        "wdl_jungle_python_tick_or_generation_ownership",
        "completed_generations",
        "logical_generation_count",
        "window_count",
        "physical_window_alignment",
        "external_python_window_boundaries",
        "direct_cpp_window_core_dispatch",
        "production_native_chunk_loop",
        "hidden_fallback",
        "runtime_jit",
    )
    summary: dict[str, Any] = {name: result.get(name) for name in scalar_names}
    run_control = result.get("native_run_control_v2")
    if not isinstance(run_control, Mapping):
        raise AllocationDiagnosticError("native result lacks serializable run control")
    try:
        # This round-trip is deliberately strict: a development receipt must
        # never silently serialize CUDA tensors or arbitrary Python objects.
        summary["native_run_control_v2"] = json.loads(json.dumps(dict(run_control)))
    except (TypeError, ValueError) as exc:
        raise AllocationDiagnosticError("native run control is not JSON-safe") from exc
    return summary


def _resident_initial_policy_refill_trace_v1(
    source: Any, *, source_commit: str, route_identity: str
) -> dict[str, Any]:
    """Materialize the actual B1 resident pre-policy all-active row set.

    The H1 canary needs the resident result, not a hand-written row count.
    This deliberately accepts B1 only: later H1 code expands the observed
    stable slot sequence across the physical B48 envelope as an explicitly
    labelled traffic scale, never as another execution result.
    """

    import torch

    if not source_commit or not route_identity:
        raise AllocationDiagnosticError("refill trace lacks source or route identity")
    live_state = getattr(source, "live_state", None)
    active_slots = getattr(source, "active_agent_slots", None)
    if not isinstance(live_state, Mapping) or not isinstance(active_slots, torch.Tensor):
        raise AllocationDiagnosticError("resident source lacks initial policy-row ownership")
    alive = live_state.get("alive_mask")
    if not isinstance(alive, torch.Tensor):
        raise AllocationDiagnosticError("resident source lacks initial alive-mask ownership")
    slots = [int(value) for value in active_slots.detach().cpu()[0].tolist() if int(value) >= 0]
    alive_cpu = alive.detach().cpu()
    if alive_cpu.ndim != 2 or alive_cpu.shape[0] != 1 or len(slots) != 256:
        raise AllocationDiagnosticError("resident source does not expose exact B1 P256 occupancy")
    if slots != list(range(256)) or not bool(torch.all(alive_cpu[0, slots]).item()):
        raise AllocationDiagnosticError("resident source lacks stable all-active low-slot ownership")
    rows_by_tick = [[[0, slot] for slot in slots] for _ in range(160)]
    return {
        "schema_version": "eiel_hapb_policy_refill_trace_v1",
        "authority": "resident_eed_b1_observed_policy_refill_trace_v1",
        "route_identity": route_identity,
        "source_commit": source_commit,
        "ticks_per_generation": 160,
        "rows_by_tick": rows_by_tick,
        "observed_cells": 1,
        "observed_tick_count": 0,
        "trace_capture_mode": "resident_prepolicy_initial_slot_deterministic_t160_replay",
        "observed_capacity": int(alive_cpu.shape[1]),
        "all_active_cache_invalidated": True,
    }


def _projection_digests(projections: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return the canonical ordered checkpoint digest vector."""

    if not projections:
        raise AllocationDiagnosticError("continuation probe has no projections")
    return tuple(projection_digest(projection) for projection in projections)


def _require_continuation_equivalence(
    uninterrupted: Sequence[Mapping[str, Any]],
    resumed: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Fail closed unless a serialized fresh-owner resume is digest-identical."""

    uninterrupted_digests = _projection_digests(uninterrupted)
    resumed_digests = _projection_digests(resumed)
    if uninterrupted_digests != resumed_digests:
        raise AllocationDiagnosticError(
            "uninterrupted and serialized fresh-owner continuation projections differ"
        )
    return uninterrupted_digests


def _continuation_difference_summary(
    uninterrupted: Sequence[Mapping[str, Any]],
    resumed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Serialize only the first-order cause of a failed exactness comparison."""

    import torch

    result: dict[str, Any] = {
        "uninterrupted_digests": list(_projection_digests(uninterrupted)),
        "resumed_digests": list(_projection_digests(resumed)),
        "cell_count_matches": len(uninterrupted) == len(resumed),
        "cells": [],
    }
    for cell, (left, right) in enumerate(zip(uninterrupted, resumed, strict=False)):
        left_metadata = dict(left.get("metadata") or {})
        right_metadata = dict(right.get("metadata") or {})
        metadata_differences = {
            name: {"uninterrupted": left_metadata.get(name), "resumed": right_metadata.get(name)}
            for name in sorted(set(left_metadata) | set(right_metadata))
            if left_metadata.get(name) != right_metadata.get(name)
        }
        left_tensors = dict(left.get("tensors") or {})
        right_tensors = dict(right.get("tensors") or {})
        tensor_differences = []
        for name in sorted(set(left_tensors) | set(right_tensors)):
            left_tensor, right_tensor = left_tensors.get(name), right_tensors.get(name)
            if not isinstance(left_tensor, torch.Tensor) or not isinstance(right_tensor, torch.Tensor):
                tensor_differences.append(name)
            elif left_tensor.dtype != right_tensor.dtype or left_tensor.shape != right_tensor.shape or not torch.equal(left_tensor, right_tensor):
                tensor_differences.append(name)
        result["cells"].append(
            {
                "cell": cell,
                "metadata_differences": metadata_differences,
                "tensor_difference_names": tensor_differences,
            }
        )
    return result


def _native_failure_snapshot(prepared: Any) -> dict[str, Any]:
    """Capture the small native failure planes without serializing the owner."""

    import torch

    from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    )

    physical = prepared.physical_window_arguments
    source = physical.wdl.patch.source
    outputs = physical.wdl.patch.source.outputs

    def _json_safe_config(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {
                "dtype": str(value.dtype),
                "device": str(value.device),
                "shape": list(value.shape),
            }
        if isinstance(value, Mapping):
            return {
                str(key): _json_safe_config(child)
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [_json_safe_config(child) for child in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return repr(value)

    result: dict[str, Any] = {
        "source": {
            name: getattr(source, name)
            for name in (
                "generation_base",
                "logical_generation_count",
                "physical_window_generations",
                "ticks_per_generation",
                "batch_cells",
                "deterministic",
            )
        },
        "scalar_arguments": {
            name: value
            for name, value in zip(
                DIRECT_RESIDENT_ARGUMENT_NAMES_V1, physical.wdl.patch.arguments, strict=True
            )
            if type(value) in {bool, int, float, str}
        },
        "tensor_argument_layout": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in zip(
                DIRECT_RESIDENT_ARGUMENT_NAMES_V1, physical.wdl.patch.arguments, strict=True
            )
            if isinstance(value, torch.Tensor)
        },
        "wdl_config": _json_safe_config(physical.wdl.config),
        "wdl_tensor_layout": {
            "state": [
                {"dtype": str(value.dtype), "shape": list(value.shape)}
                for value in physical.wdl.state_tensors
            ],
            "schedule": [
                {"dtype": str(value.dtype), "shape": list(value.shape)}
                for value in physical.wdl.schedule_tensors
            ],
            "reporting": [
                {"dtype": str(value.dtype), "shape": list(value.shape)}
                for value in physical.wdl.reporting_tensors
            ],
        },
        "jungle_config": _json_safe_config(physical.jungle_config),
        "resident_sequence_layout": {
            "ordinary_birth_scratch": [
                {"dtype": str(value.dtype), "shape": list(value.shape)}
                for value in source.ordinary_birth_scratch_tensors
            ],
            "patch_p2_stats": [
                {"dtype": str(value.dtype), "shape": list(value.shape)}
                for value in source.patch_p2_stats_tensors
            ],
        },
        # Turnover failures must expose the selected inheritance authority.
        # The native ABI receives this as an ordered tensor sidecar, so a
        # status such as scalar/26 is otherwise impossible to distinguish
        # from a stale native artifact after the owner has unwound.  Keep the
        # diagnostic bounded to the mode/control planes; never serialize the
        # large resident genome or policy banks.
        "ordinary_birth_rng": {
            "planes": {
                str(index): {
                    "dtype": str(source.ordinary_birth_state_tensors[index].dtype),
                    "shape": list(source.ordinary_birth_state_tensors[index].shape),
                    "values": source.ordinary_birth_state_tensors[index]
                    .detach()
                    .cpu()
                    .reshape(-1)[:64]
                    .tolist(),
                }
                for index in (35, 36, 37, 38, 39, 40, 41, 42, 111, 112)
                if index < len(source.ordinary_birth_state_tensors)
            },
        },
        "ordinary_birth_policy_cursors": {
            "planes": {
                str(index): {
                    "dtype": str(source.ordinary_birth_state_tensors[index].dtype),
                    "shape": list(source.ordinary_birth_state_tensors[index].shape),
                    "values": source.ordinary_birth_state_tensors[index]
                    .detach()
                    .cpu()
                    .reshape(-1)[:64]
                    .tolist(),
                }
                for index in (91, 92, 111)
                if index < len(source.ordinary_birth_state_tensors)
            },
        },
        "ordinary_birth_authority": {
            "tensor_count": len(source.ordinary_birth_state_tensors),
            "planes": {
                str(index): {
                    "dtype": str(source.ordinary_birth_state_tensors[index].dtype),
                    "shape": list(source.ordinary_birth_state_tensors[index].shape),
                    "values": source.ordinary_birth_state_tensors[index]
                    .detach()
                    .cpu()
                    .reshape(-1)[:64]
                    .tolist(),
                }
                for index in (113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124)
                if index < len(source.ordinary_birth_state_tensors)
            },
        },
    }
    for name in (
        "status",
        "first_failure_records",
        "lifecycle_status",
        "birth_combined_status",
        "birth_policy_status",
        "turnover_status",
        "turnover_counts",
        "turnover_refill_parent_agent_ids",
        "turnover_refill_mate_agent_ids",
        "turnover_child_slots",
        "active_count_state",
    ):
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            result[name] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": value.detach().cpu().reshape(-1)[:64].tolist(),
            }
    active_count = getattr(source, "active_count_state", None)
    if isinstance(active_count, torch.Tensor):
        result["active_count_state"] = {
            "dtype": str(active_count.dtype),
            "shape": list(active_count.shape),
            "values": active_count.detach().cpu().reshape(-1)[:64].tolist(),
        }
    return result


def _apply_jungle_feature_control(prepared: Any, *, control: str) -> None:
    """Apply a labeled non-qualifying overlay isolation control."""

    allowed = {
        "full",
        "no_predator",
        "no_recovery",
        "no_birth_emissions",
        "minimal",
    }
    if control not in allowed:
        raise AllocationDiagnosticError("QualPrep Jungle feature control is unsupported")
    if control == "full":
        return
    config = prepared.physical_window_arguments.jungle_config
    if not isinstance(config, dict):
        raise AllocationDiagnosticError("prepared Jungle configuration is not mutable")
    field = {
        "no_predator": "predator_pressure_enabled",
        "no_recovery": "ecological_recovery_enabled",
        "no_birth_emissions": "birth_emissions_enabled",
    }.get(control)
    if field is not None:
        config[field] = False
    else:
        config.update(
            {
                "predator_pressure_enabled": False,
                "ecological_recovery_enabled": False,
                "birth_emissions_enabled": False,
            }
        )


def run_qualprep_runtime_v1(
    spec_path: Path,
    *,
    output_root: Path,
    cells: int,
    generations: int,
    cell_offset: int = 0,
    artifact_override: Path | None = None,
    manifest_override: Path | None = None,
    native_mode: str = "wdl_jungle",
    jungle_feature_control: str = "full",
    eed_overlay: bool = False,
    capture_h1_refill_trace: bool = False,
    hapb_policy_only: bool = False,
    hapb_ordinary_birth: bool = False,
    generation_turnover: bool = False,
    selected_cell_ids: Sequence[str] | None = None,
    expand_cells: int | None = None,
) -> dict[str, Any]:
    """Run a real native B1/B2/B48 Q3 probe and serialize terminal projections."""

    from engine.perf.cuda_architecture_b_hapb_v1 import hapb_physical_cell_count_v1

    try:
        cells = hapb_physical_cell_count_v1(cells)
    except Exception as exc:
        raise AllocationDiagnosticError(
            "QualPrep runtime cells must be in [1, 256]"
        ) from exc
    if type(generations) is not int or not 3 <= generations <= 128:
        raise AllocationDiagnosticError("QualPrep runtime generations must be in [3,128]")
    if native_mode not in {"wdl_jungle", "wdl_only"}:
        raise AllocationDiagnosticError("QualPrep native mode is unsupported")
    if hapb_policy_only and hapb_ordinary_birth:
        raise AllocationDiagnosticError("HAPB runtime mode must select one direct route")
    hapb_direct_route = hapb_policy_only or hapb_ordinary_birth
    if hapb_direct_route and (
        not eed_overlay or native_mode != "wdl_jungle"
    ):
        raise AllocationDiagnosticError(
            "HAPB direct runtime requires WDL/Jungle/EED"
        )
    if jungle_feature_control not in {
        "full", "no_predator", "no_recovery", "no_birth_emissions"
        , "minimal"
    }:
        raise AllocationDiagnosticError("QualPrep Jungle feature control is unsupported")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") not in {None, "expandable_segments:True"}:
        raise AllocationDiagnosticError("unsupported PYTORCH_CUDA_ALLOC_CONF")
    spec_path = spec_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise AllocationDiagnosticError(f"QualPrep output root already exists: {output_root}")
    source_spec = load_campaign_spec(spec_path)
    if hapb_direct_route:
        route_name = "policy-only" if hapb_policy_only else "ordinary-birth"
        source_spec["campaign_id"] = (
            f"{source_spec['campaign_id']}-hapb-{route_name}"
        )
        source_spec["description"] = (
            f"Physical B48 HAPB {route_name} direct-route development probe."
        )
        source_spec["no_claims"] = (
            "HAPB bounded direct-route probe; checkpoint/resume, science, and "
            "qualification claims are excluded."
        )
        source_spec["parameters"]["production_lifecycle.ordinary_births_enabled"] = (
            hapb_ordinary_birth
        )
        if generation_turnover:
            source_spec["parameters"]["production_lifecycle.lifecycle_mode"] = (
                "legacy_synchronized_truncation_v1"
            )
            source_spec["parameters"]["production_lifecycle.life_history_profile"] = (
                "legacy_inert_v1"
            )
            source_spec["parameters"]["production_lifecycle.turnover_cull_fraction"] = 0.5
            source_spec["parameters"]["production_lifecycle.turnover_parent_fraction"] = 1.0
    expand_to = cells if expand_cells is None else hapb_physical_cell_count_v1(expand_cells)
    if expand_to > 48:
        from engine.perf.hapb_physical_extent_v1 import (
            expand_campaign_spec_physical_cells_v1,
        )

        source_spec = expand_campaign_spec_physical_cells_v1(source_spec, expand_to)
    source_plan = compile_campaign(source_spec).payload()
    runtime_plan = _runtime_plan(
        source_plan,
        cells=cells,
        cell_offset=cell_offset,
        selected_cell_ids=selected_cell_ids,
    )
    runtime_plan = _bind_diagnostic_artifact(
        runtime_plan,
        artifact_override=artifact_override,
        manifest_override=manifest_override,
    )
    if hapb_direct_route:
        runtime_plan = _bind_hapb_artifact_abi(runtime_plan)
    _preflight_external_artifact_bindings_v1(runtime_plan)
    import torch

    if not torch.cuda.is_available():
        raise AllocationDiagnosticError("CUDA is unavailable")
    artifact_row = runtime_plan["artifacts"]["cuda_extension"]
    manifest_row = runtime_plan["artifacts"]["extension_manifest"]
    artifact = Path(str(artifact_row["path"])).resolve()
    manifest = Path(str(manifest_row["path"])).resolve()
    if _sha256(artifact) != artifact_row["sha256"] or _sha256(manifest) != manifest_row["sha256"]:
        raise AllocationDiagnosticError("QualPrep artifact identity differs from exact B48 plan")
    output_root.mkdir(parents=True)
    descriptor_qualification, descriptor_reporting = _diagnostic_descriptor(
        output_root,
        artifact_sha256=str(artifact_row["sha256"]),
        manifest_sha256=str(manifest_row["sha256"]),
        plan_sha256=str(runtime_plan["plan_sha256"]),
    )
    diagnostic_plan = _diagnostic_plan(
        runtime_plan,
        qualification=descriptor_qualification,
        reporting=descriptor_reporting,
    )
    progress_path = output_root / "qualprep_runtime_progress.json"

    def record_progress(stage: str) -> None:
        atomic_write_json(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "authority": "qualprep_progress_not_admission_v1",
                "stage": stage,
                "cells": cells,
                "cell_offset": cell_offset,
                "logical_generations": generations,
            },
        )

    record_progress("descriptors_written")
    module: Any = None
    prepared: Any = None
    provider: Any = None
    sources: Any = None
    try:
        module = _load_exact_module(
            artifact,
            eed_overlay=eed_overlay,
            hapb_policy_only=hapb_direct_route,
        )
        provider = AllocationOnlyProvider(bind_committed_defaults=False)
        request = provider.translate_batch(
            diagnostic_plan,
            str(runtime_plan["physical_batches"][0]["batch_id"]),
        )
        adapters = dict(provider._resolved_execution_adapters(diagnostic_plan))
        if hapb_policy_only:
            from engine.campaigns.production_engine_native_bindings import (
                _wdl_jungle_eed_hapb_policy_execute_adapter,
                _wdl_jungle_eed_hapb_policy_prepare_adapter,
            )

            adapters["preparer"] = _wdl_jungle_eed_hapb_policy_prepare_adapter
            adapters["executor"] = _wdl_jungle_eed_hapb_policy_execute_adapter
        elif hapb_ordinary_birth:
            from engine.campaigns.production_engine_native_bindings import (
                _wdl_jungle_eed_hapb_birth_execute_adapter,
                _wdl_jungle_eed_hapb_birth_prepare_adapter,
            )

            adapters["preparer"] = _wdl_jungle_eed_hapb_birth_prepare_adapter
            adapters["executor"] = _wdl_jungle_eed_hapb_birth_execute_adapter
        selected_cells = provider._batch_cells(diagnostic_plan, request)
        context = SimpleNamespace(
            run_root=output_root, batch_root=output_root, progress_events=[]
        )
        sources = tuple(
            provider._prepare_generation_zero_sources_v2(
                diagnostic_plan,
                request,
                context,
                runner_builder=adapters["runner_builder"],
                cells=selected_cells,
                artifact=str(artifact),
                qualification=str(descriptor_qualification),
                reporting=str(descriptor_reporting),
                segment_generations=request.generations,
                source_capability_mask=_wdl_source_capabilities_v2(
                    {"genome_v2": True},
                    jungle_overlay=True,
                    eed_overlay=eed_overlay,
                ),
            )
        )
        record_progress("sources_prepared")
        _, _, prepared = provider._prepare_resident_segment_owner_v1(
            plan=diagnostic_plan,
            runner_builder=adapters["runner_builder"],
            request_builder=adapters["request_builder"],
            preparer=adapters["preparer"],
            cells=selected_cells,
            cell_ids=request.cell_ids,
            batch_root=output_root,
            start_generation=0,
            end_generation=generations,
            ticks_per_generation=request.ticks_per_generation,
            artifact=str(artifact),
            qualification=str(descriptor_qualification),
            reporting=str(descriptor_reporting),
            module=module,
            source_projections=sources,
            allocation_generations=provider._agent_id_allocation_generation_count(
                request, start_generation=0
            ),
            genome_v2_checkpoint_authority_binding=dict(
                request.genome_v2_checkpoint_authority_binding
            ),
            physical_window_generations=1,
        )
        record_progress("owner_prepared")
        refill_trace: dict[str, Any] | None = None
        if capture_h1_refill_trace:
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            provenance = manifest_payload.get("authoritative_state_build_provenance_v1")
            if not isinstance(provenance, Mapping) or not isinstance(provenance.get("build_source_commit"), str):
                raise AllocationDiagnosticError("artifact manifest lacks source provenance")
            refill_trace = _resident_initial_policy_refill_trace_v1(
                prepared.physical_window_arguments.wdl.patch.source,
                source_commit=provenance["build_source_commit"],
                route_identity=str(source_plan.get("campaign_id", "")),
            )
        _apply_jungle_feature_control(
            prepared, control=jungle_feature_control
        )
        # A native abort cannot be converted into a Python exception.  Persist
        # the dispatch boundary first so a bounded probe remains diagnostic.
        record_progress("native_execution_started")
        try:
            if native_mode == "wdl_jungle":
                direct_result = adapters["executor"](
                    module,
                    physical_window_arguments=prepared.physical_window_arguments,
                    logical_generation_count=generations,
                )
            else:
                from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
                    execute_direct_resident_wdl_v2,
                )

                direct_result = execute_direct_resident_wdl_v2(
                    module,
                    prepared.physical_window_arguments.wdl,
                    logical_generation_count=generations,
                    run_id=f"qualprep-b{cells:02d}-wdl-only",
                )
        except BaseException as exc:
            atomic_write_json(
                output_root / "native_failure_diagnostic.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "authority": "qualprep_failure_diagnostic_not_admission_v1",
                    "error": f"{type(exc).__name__}: {exc}",
                    "failure_planes": _native_failure_snapshot(prepared),
                },
            )
            raise
        record_progress("native_execution_complete")
        if hapb_policy_only:
            policy_result = _execution_summary(direct_result)
            policy_result["hapb_policy_only_route"] = (
                direct_result.get("hapb_policy_only_route")
            )
            if policy_result["hapb_policy_only_route"] is not True:
                raise AllocationDiagnosticError(
                    "HAPB direct execution did not retain its policy-only guard"
                )
            policy_path = output_root / "hapb_policy_only_direct_result.json"
            atomic_write_json(policy_path, policy_result)
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "authority": "hapb_policy_only_direct_development_probe_v1",
                "repository_head": _git_head(),
                "base_b48_plan_sha256": source_plan["plan_sha256"],
                "runtime_probe_plan_sha256": diagnostic_plan["plan_sha256"],
                "artifact_sha256": artifact_row["sha256"],
                "cells": cells,
                "logical_generations": generations,
                "completed_generations": int(policy_result["completed_generations"]),
                "direct_result_path": str(policy_path),
                "direct_result_sha256": _sha256(policy_path),
                "hapb_policy_only_route": True,
                "checkpoint_projection": "intentionally_fail_closed_pending_hapb_checkpoint",
                "production_execution_admission": False,
                "qualification_claim": False,
                "eed_overlay": eed_overlay,
                "eed_absent_from_artifact": not eed_overlay,
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(
                output_root / "hapb_policy_only_direct_receipt.json", receipt
            )
            record_progress("hapb_policy_only_direct_execution_complete")
            return receipt
        if hapb_ordinary_birth:
            birth_result = _execution_summary(direct_result)
            if direct_result.get("hapb_host_authoritative_ordinary_birth") is not True:
                raise AllocationDiagnosticError(
                    "HAPB direct execution did not retain its ordinary-birth bridge"
                )
            if direct_result.get("hapb_policy_only_route") is not False:
                raise AllocationDiagnosticError(
                    "HAPB ordinary-birth execution retained the policy-only guard"
                )
            birth_path = output_root / "hapb_ordinary_birth_direct_result.json"
            atomic_write_json(birth_path, birth_result)
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "authority": "hapb_ordinary_birth_direct_development_probe_v1",
                "repository_head": _git_head(),
                "base_b48_plan_sha256": source_plan["plan_sha256"],
                "runtime_probe_plan_sha256": diagnostic_plan["plan_sha256"],
                "artifact_sha256": artifact_row["sha256"],
                "cells": cells,
                "logical_generations": generations,
                "completed_generations": int(birth_result["completed_generations"]),
                "direct_result_path": str(birth_path),
                "direct_result_sha256": _sha256(birth_path),
                "hapb_host_authoritative_ordinary_birth": True,
                "hapb_policy_only_route": False,
                "hapb_host_authoritative_generation_turnover": bool(
                    direct_result.get("hapb_host_authoritative_generation_turnover")
                ),
                "generation_turnover": (
                    "host_authoritative_transaction"
                    if direct_result.get("hapb_host_authoritative_generation_turnover")
                    else "not_enabled_in_this_probe"
                ),
                "checkpoint_projection": "component route wired; split/resume unproven",
                "production_execution_admission": False,
                "qualification_claim": False,
                "eed_overlay": eed_overlay,
                "eed_absent_from_artifact": not eed_overlay,
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(
                output_root / "hapb_ordinary_birth_direct_receipt.json", receipt
            )
            record_progress("hapb_ordinary_birth_direct_execution_complete")
            return receipt
        # Q3 needs the canonical checkpoint boundary.  The provider's optional
        # compact metrics merger is exercised separately: it is deliberately
        # not allowed to erase an otherwise valid native terminal projection.
        record_progress("terminal_checkpoint_projection_started")
        if native_mode == "wdl_jungle":
            from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
                project_direct_resident_wdl_jungle_multicell_checkpoints_v1,
            )

            projections = project_direct_resident_wdl_jungle_multicell_checkpoints_v1(
                prepared.physical_window_arguments,
                direct_result=direct_result,
                completed_generation_count=generations,
                reporting_enabled=False,
            )
        else:
            from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
                project_direct_resident_wdl_multicell_checkpoints_v2,
            )

            projections = project_direct_resident_wdl_multicell_checkpoints_v2(
                prepared.physical_window_arguments.wdl,
                direct_result=direct_result,
                completed_generation_count=generations,
                reporting_enabled=False,
            )
        record_progress("terminal_checkpoint_projection_complete")
        if len(projections) != cells:
            raise AllocationDiagnosticError("runtime projection count differs from B1/B2 request")
        refill_trace_path: Path | None = None
        if refill_trace is not None:
            refill_trace["trace_sha256"] = canonical_sha256(refill_trace)
            refill_trace_path = output_root / "resident_policy_refill_trace.json"
            atomic_write_json(refill_trace_path, refill_trace)
        projection_path = output_root / "terminal_checkpoint_projections.pt"
        torch.save(projections, projection_path)
        record_progress("terminal_projection_serialized")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "authority": "qualprep_runtime_not_production_admission_v1",
            "repository_head": _git_head(),
            "base_b48_plan_sha256": source_plan["plan_sha256"],
            "runtime_probe_plan_sha256": diagnostic_plan["plan_sha256"],
            "artifact_sha256": artifact_row["sha256"],
            "cells": cells,
            "native_mode": native_mode,
            "jungle_feature_control": jungle_feature_control,
            "logical_generations": generations,
            "ticks_per_generation": request.ticks_per_generation,
            "execution_result": _execution_summary(direct_result),
            "completed_generations": int(direct_result["completed_generations"]),
            "terminal_generation": prepared.physical_window_arguments.wdl.patch.source.generation_base
            + int(direct_result["completed_generations"]),
            "checkpoint_projection_sha256": _sha256(projection_path),
            "checkpoint_projection_path": str(projection_path),
            "resident_policy_refill_trace_path": (
                str(refill_trace_path) if refill_trace_path is not None else None
            ),
            "resident_policy_refill_trace_sha256": (
                _sha256(refill_trace_path) if refill_trace_path is not None else None
            ),
            "production_execution_admission": False,
            "qualification_claim": False,
            "eed_overlay": eed_overlay,
            "eed_absent_from_artifact": not eed_overlay,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(output_root / "qualprep_runtime_receipt.json", receipt)
        return receipt
    finally:
        prepared = None
        sources = None
        provider = None
        module = None
        gc.collect()
        torch.cuda.empty_cache()


def run_qualprep_continuation_v1(
    spec_path: Path,
    *,
    output_root: Path,
    cells: int,
    generations: int,
    split_generations: int,
    artifact_override: Path | None = None,
    manifest_override: Path | None = None,
    eed_overlay: bool = False,
) -> dict[str, Any]:
    """Compare uninterrupted execution with a serialized fresh-owner resume.

    This is deliberately a small QualPrep proof: both legs use the production
    WDL/Jungle preparer and executor, the midpoint is written to disk, all
    CUDA owner references are released, and the resumed leg receives only the
    freshly deserialized checkpoint projections.
    """

    if cells not in {1, 2}:
        raise AllocationDiagnosticError("continuation probe cells must be B1 or B2")
    if (
        type(generations) is not int
        or type(split_generations) is not int
        or split_generations < 3
        or generations - split_generations < 3
        or generations > 128
    ):
        raise AllocationDiagnosticError(
            "continuation probe requires two native legs of at least three generations"
        )
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") not in {None, "expandable_segments:True"}:
        raise AllocationDiagnosticError("unsupported PYTORCH_CUDA_ALLOC_CONF")
    spec_path = spec_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise AllocationDiagnosticError(
            f"continuation probe output root already exists: {output_root}"
        )
    source_plan = compile_campaign(load_campaign_spec(spec_path)).payload()
    runtime_plan = _bind_diagnostic_artifact(
        _runtime_plan(source_plan, cells=cells),
        artifact_override=artifact_override,
        manifest_override=manifest_override,
    )
    _preflight_external_artifact_bindings_v1(runtime_plan)
    import torch

    if not torch.cuda.is_available():
        raise AllocationDiagnosticError("CUDA is unavailable")
    artifact_row = runtime_plan["artifacts"]["cuda_extension"]
    manifest_row = runtime_plan["artifacts"]["extension_manifest"]
    artifact = Path(str(artifact_row["path"])).resolve()
    manifest = Path(str(manifest_row["path"])).resolve()
    if _sha256(artifact) != artifact_row["sha256"] or _sha256(manifest) != manifest_row["sha256"]:
        raise AllocationDiagnosticError("continuation artifact identity differs from plan")
    output_root.mkdir(parents=True)
    qualification, reporting = _diagnostic_descriptor(
        output_root,
        artifact_sha256=str(artifact_row["sha256"]),
        manifest_sha256=str(manifest_row["sha256"]),
        plan_sha256=str(runtime_plan["plan_sha256"]),
    )
    diagnostic_plan = _diagnostic_plan(
        runtime_plan, qualification=qualification, reporting=reporting
    )
    progress_path = output_root / "qualprep_continuation_progress.json"

    def record(stage: str) -> None:
        atomic_write_json(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "authority": "qualprep_continuation_not_admission_v1",
                "stage": stage,
                "cells": cells,
                "generations": generations,
                "split_generations": split_generations,
            },
        )

    def project(prepared: Any, direct_result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
            project_direct_resident_wdl_jungle_multicell_checkpoints_v1,
        )

        return project_direct_resident_wdl_jungle_multicell_checkpoints_v1(
            prepared.physical_window_arguments,
            direct_result=direct_result,
            completed_generation_count=int(direct_result["completed_generations"]),
            reporting_enabled=True,
        )

    module: Any = None
    provider: Any = None
    sources: tuple[Mapping[str, Any], ...] | None = None
    try:
        record("descriptors_written")
        module = _load_exact_module(artifact, eed_overlay=eed_overlay)
        provider = AllocationOnlyProvider(bind_committed_defaults=False)
        request = provider.translate_batch(
            diagnostic_plan, str(runtime_plan["physical_batches"][0]["batch_id"])
        )
        adapters = dict(provider._resolved_execution_adapters(diagnostic_plan))
        selected_cells = provider._batch_cells(diagnostic_plan, request)
        context = SimpleNamespace(
            run_root=output_root, batch_root=output_root, progress_events=[]
        )
        sources = tuple(
            provider._prepare_generation_zero_sources_v2(
                diagnostic_plan,
                request,
                context,
                runner_builder=adapters["runner_builder"],
                cells=selected_cells,
                artifact=str(artifact),
                qualification=str(qualification),
                reporting=str(reporting),
                segment_generations=request.generations,
                source_capability_mask=_wdl_source_capabilities_v2(
                    {"genome_v2": True},
                    jungle_overlay=True,
                    eed_overlay=eed_overlay,
                ),
            )
        )

        def execute_segment(
            source_projections: Sequence[Mapping[str, Any]], *, start: int, end: int
        ) -> tuple[Mapping[str, Any], ...]:
            _, _, prepared = provider._prepare_resident_segment_owner_v1(
                plan=diagnostic_plan,
                runner_builder=adapters["runner_builder"],
                request_builder=adapters["request_builder"],
                preparer=adapters["preparer"],
                cells=selected_cells,
                cell_ids=request.cell_ids,
                batch_root=output_root,
                start_generation=start,
                end_generation=end,
                ticks_per_generation=request.ticks_per_generation,
                artifact=str(artifact),
                qualification=str(qualification),
                reporting=str(reporting),
                module=module,
                source_projections=source_projections,
                allocation_generations=provider._agent_id_allocation_generation_count(
                    request, start_generation=start
                ),
                genome_v2_checkpoint_authority_binding=dict(
                    request.genome_v2_checkpoint_authority_binding
                ),
                physical_window_generations=1,
            )
            try:
                direct_result = adapters["executor"](
                    module,
                    physical_window_arguments=prepared.physical_window_arguments,
                    logical_generation_count=end - start,
                )
                return project(prepared, direct_result)
            finally:
                prepared = None
                gc.collect()
                torch.cuda.empty_cache()

        record("uninterrupted_started")
        uninterrupted = execute_segment(sources, start=0, end=generations)
        uninterrupted_path = output_root / "uninterrupted_terminal_projections.pt"
        torch.save(uninterrupted, uninterrupted_path)
        record("uninterrupted_complete")
        split_checkpoint = execute_segment(sources, start=0, end=split_generations)
        midpoint_path = output_root / "serialized_midpoint_projections.pt"
        torch.save(split_checkpoint, midpoint_path)
        midpoint_sha256 = _sha256(midpoint_path)
        split_checkpoint = None
        gc.collect()
        torch.cuda.empty_cache()
        record("midpoint_serialized_owner_released")
        restored = tuple(torch.load(midpoint_path, map_location="cpu", weights_only=False))
        if len(restored) != cells:
            raise AllocationDiagnosticError("serialized midpoint lost physical cell identity")
        resumed = execute_segment(restored, start=split_generations, end=generations)
        resumed_path = output_root / "resumed_terminal_projections.pt"
        torch.save(resumed, resumed_path)
        record("fresh_owner_resume_complete")
        comparison = _continuation_difference_summary(uninterrupted, resumed)
        try:
            digests = _require_continuation_equivalence(uninterrupted, resumed)
        except AllocationDiagnosticError:
            atomic_write_json(
                output_root / "continuation_mismatch_diagnostic.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "authority": "qualprep_continuation_mismatch_diagnostic_v1",
                    "uninterrupted_terminal_path": str(uninterrupted_path),
                    "resumed_terminal_path": str(resumed_path),
                    "comparison": comparison,
                },
            )
            raise
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "authority": "qualprep_continuation_not_production_admission_v1",
            "repository_head": _git_head(),
            "artifact_sha256": artifact_row["sha256"],
            "cells": cells,
            "generations": generations,
            "split_generations": split_generations,
            "ticks_per_generation": request.ticks_per_generation,
            "serialized_midpoint_path": str(midpoint_path),
            "serialized_midpoint_sha256": midpoint_sha256,
            "uninterrupted_terminal_projection_sha256": _sha256(uninterrupted_path),
            "resumed_terminal_projection_sha256": _sha256(resumed_path),
            "terminal_projection_digests": list(digests),
            "fresh_owner_restore": True,
            "continuation_exact": True,
            "production_execution_admission": False,
            "qualification_claim": False,
            "eed_overlay": eed_overlay,
            "eed_absent_from_artifact": not eed_overlay,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(output_root / "qualprep_continuation_receipt.json", receipt)
        record("continuation_exact")
        return receipt
    finally:
        sources = None
        provider = None
        module = None
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--cell-offset", type=int, default=0)
    parser.add_argument("--artifact-override", type=Path)
    parser.add_argument("--manifest-override", type=Path)
    parser.add_argument(
        "--eed-overlay",
        action="store_true",
        help="require the combined WDL/Jungle/EED owner and entrypoint",
    )
    parser.add_argument(
        "--capture-h1-refill-trace",
        action="store_true",
        help="write the actual B1 resident all-active policy-refill observation",
    )
    parser.add_argument(
        "--hapb-policy-only",
        action="store_true",
        help="run only the guarded B48 no-birth/no-turnover HAPB policy route",
    )
    parser.add_argument(
        "--hapb-ordinary-birth",
        action="store_true",
        help="run the guarded B48 HAPB ordinary-birth/no-turnover direct route",
    )
    parser.add_argument(
        "--generation-turnover",
        action="store_true",
        help=(
            "run HAPB ordinary birth with legacy synchronized generation "
            "turnover and the host-authoritative transaction"
        ),
    )
    parser.add_argument("--native-mode", choices=("wdl_jungle", "wdl_only"), default="wdl_jungle")
    parser.add_argument(
        "--jungle-feature-control",
        choices=("full", "no_predator", "no_recovery", "no_birth_emissions", "minimal"),
        default="full",
    )
    parser.add_argument(
        "--continuation-split-generations",
        type=int,
        help="run the real serialized fresh-owner continuation comparison",
    )
    args = parser.parse_args(argv)
    if args.continuation_split_generations is None:
        receipt = run_qualprep_runtime_v1(
            args.spec,
            output_root=args.output_root,
            cells=args.cells,
            generations=args.generations,
            cell_offset=args.cell_offset,
            artifact_override=args.artifact_override,
            manifest_override=args.manifest_override,
            native_mode=args.native_mode,
             jungle_feature_control=args.jungle_feature_control,
             eed_overlay=args.eed_overlay,
             capture_h1_refill_trace=args.capture_h1_refill_trace,
             hapb_policy_only=args.hapb_policy_only,
             hapb_ordinary_birth=args.hapb_ordinary_birth,
             generation_turnover=args.generation_turnover,
        )
        summary = {"receipt_sha256": receipt["receipt_sha256"], "completed_generations": receipt["completed_generations"]}
    else:
        receipt = run_qualprep_continuation_v1(
            args.spec,
            output_root=args.output_root,
            cells=args.cells,
            generations=args.generations,
            split_generations=args.continuation_split_generations,
            artifact_override=args.artifact_override,
            manifest_override=args.manifest_override,
            eed_overlay=args.eed_overlay,
        )
        summary = {"receipt_sha256": receipt["receipt_sha256"], "continuation_exact": receipt["continuation_exact"]}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
