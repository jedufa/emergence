#!/usr/bin/env python3
"""Measure one exact non-EED B48 owner without granting execution admission.

This diagnostic deliberately has no execution callback.  It imports only the
isolated WDL/Jungle CUDA artifact, constructs the configured generation-zero
sources on CPU, and prepares one CUDA owner.  Its receipt is diagnostic-only:
it cannot be reused as a route qualification or production admission receipt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = ROOT / "engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if ENGINE_ROOT.is_dir() and str(ENGINE_ROOT) not in sys.path:
    # This diagnostic is intentionally launched with a clean environment.
    # Historical simulator imports still use top-level engine packages.
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.compiler import compile_campaign, load_campaign_spec  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from engine.campaigns.production_engine_provider import (  # noqa: E402
    EielProductionExecutionProviderV1,
)
from engine.campaigns.providers import ProductionEngineLifecycleV1  # noqa: E402
from engine.campaigns.resident_prepared_owner_resource_audit_v1 import (  # noqa: E402
    inventory_resident_prepared_owner_storages_v1,
)


SCHEMA_VERSION = "eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1"
DIAGNOSTIC_AUTHORITY = "allocation_only_not_execution_admission_v1"
REQUIRED_ALLOCATOR_VALUE = "expandable_segments:True"
MINIMUM_RESIDUAL_BYTES = 64 * 1024 * 1024


class AllocationDiagnosticError(RuntimeError):
    """The exact allocation-only diagnostic cannot safely run."""


class AllocationOnlyProvider(EielProductionExecutionProviderV1):
    """Expose mechanics preparation while explicitly bypassing production admission.

    The inherited production route contract validates a formal Genome V2 package
    against an admitted CUDA artifact.  That is correct for execution, but it is
    circular for this pre-admission allocation diagnostic.  This override binds
    the full diagnostic plan and artifacts to a distinct route hash and uses the
    generic request translator; it never supplies an execution method.
    """

    def execution_route_contract(self, plan: Mapping[str, Any], request: Any = None) -> Any:
        del request
        return SimpleNamespace(
            execution_route_sha256=canonical_sha256(
                {
                    "schema_version": SCHEMA_VERSION,
                    "authority": DIAGNOSTIC_AUTHORITY,
                    "plan_sha256": plan.get("plan_sha256"),
                    "artifact_sha256s": {
                        name: row.get("sha256")
                        for name, row in sorted(dict(plan.get("artifacts") or {}).items())
                        if isinstance(row, Mapping)
                    },
                }
            )
        )

    def translate_batch(self, plan: Mapping[str, Any], batch_id: str) -> Any:
        return ProductionEngineLifecycleV1.translate_batch(self, plan, batch_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_diagnostic_artifact_v1(
    plan: Mapping[str, Any],
    *,
    artifact_override: Path | None,
    manifest_override: Path | None,
    artifact_source_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a fresh immutable artifact to an allocation-only diagnostic plan."""

    if (artifact_override is None) != (manifest_override is None):
        raise AllocationDiagnosticError(
            "diagnostic artifact and manifest overrides must be supplied together"
        )
    result = dict(plan)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AllocationDiagnosticError("diagnostic plan lacks artifact bindings")
    bound = {name: dict(value) for name, value in artifacts.items()}
    if artifact_source_plan is not None:
        source_artifacts = artifact_source_plan.get("artifacts")
        if not isinstance(source_artifacts, Mapping):
            raise AllocationDiagnosticError(
                "artifact-source plan lacks artifact bindings"
            )
        bound.update(
            {
                str(name): dict(value)
                for name, value in source_artifacts.items()
                if isinstance(value, Mapping)
            }
        )
    if artifact_override is not None and manifest_override is not None:
        artifact = artifact_override.expanduser().resolve()
        manifest = manifest_override.expanduser().resolve()
        if not artifact.is_file() or not manifest.is_file():
            raise AllocationDiagnosticError("diagnostic artifact or manifest is absent")
        bound["cuda_extension"].update({"path": str(artifact), "sha256": _sha256(artifact)})
        bound["extension_manifest"].update(
            {"path": str(manifest), "sha256": _sha256(manifest)}
        )
    result["artifacts"] = bound
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _gpu_processes() -> list[dict[str, str]]:
    """Inspect compute users only; this diagnostic never touches other PIDs."""

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
        return []
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 3 and values[0]:
            rows.append(
                {"pid": values[0], "process_name": values[1], "used_mib": values[2]}
            )
    return rows


def _host_memory() -> dict[str, int | None]:
    hwm_kib: int | None = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                hwm_kib = int(line.split()[1])
                break
    except (OSError, ValueError, IndexError):
        pass
    return {
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "hwm_bytes": None if hwm_kib is None else hwm_kib * 1024,
    }


def _cuda_sample(torch: Any, stage: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    pinned = 0
    # PyTorch has no complete process-wide pinned-memory counter.  The
    # diagnostic states that limitation explicitly rather than inventing one.
    return {
        "stage": stage,
        "driver_free_bytes": int(free_bytes),
        "driver_total_bytes": int(total_bytes),
        "torch_allocated_bytes": int(torch.cuda.memory_allocated()),
        "torch_reserved_bytes": int(torch.cuda.memory_reserved()),
        "torch_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "allocator_reserve_gap_bytes": max(
            0, int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())
        ),
        "process_gpu_bytes": int(total_bytes) - int(free_bytes),
        "pinned_memory_bytes": pinned,
        "pinned_memory_scope": "unavailable_process_wide_v1",
        "host_memory": _host_memory(),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def _load_exact_module(
    path: Path,
    *,
    eed_overlay: bool = False,
    hapb_policy_only: bool = False,
) -> Any:
    if hapb_policy_only and not eed_overlay:
        raise AllocationDiagnosticError("HAPB artifact requires the EED overlay")
    module_name = (
        "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"
        if hapb_policy_only
        else (
            "eiel_cuda_architecture_b_wdl_jungle_eed_v1"
            if eed_overlay
            else "eiel_cuda_architecture_b_non_eed_wdl_jungle_v1"
        )
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AllocationDiagnosticError("isolated CUDA artifact cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    exports = tuple(sorted(name for name in dir(module) if not name.startswith("__")))
    if not eed_overlay and any("eed" in name.lower() for name in exports):
        raise AllocationDiagnosticError("isolated artifact exposes an EED symbol")
    entrypoint = (
        "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_v1"
        if hapb_policy_only
        else (
            "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_v1"
            if eed_overlay
            else "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_v1"
        )
    )
    if not callable(getattr(module, entrypoint, None)):
        raise AllocationDiagnosticError("isolated artifact lacks the WDL/Jungle owner entrypoint")
    return module


def _diagnostic_descriptor(
    root: Path, *, artifact_sha256: str, manifest_sha256: str, plan_sha256: str
) -> tuple[Path, Path]:
    payload = {
        "schema_version": "eiel_cuda_allocation_diagnostic_artifact_descriptor_v1",
        "authority": DIAGNOSTIC_AUTHORITY,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "plan_sha256": plan_sha256,
        "qualification_claim": False,
        "reporting_claim": False,
    }
    qualification = root / "allocation_only_not_qualification.json"
    reporting = root / "allocation_only_not_reporting_admission.json"
    atomic_write_json(qualification, {**payload, "kind": "not_a_qualification_receipt"})
    atomic_write_json(reporting, {**payload, "kind": "not_a_reporting_receipt"})
    return qualification, reporting


def _diagnostic_plan(
    plan: Mapping[str, Any], *, qualification: Path, reporting: Path
) -> dict[str, Any]:
    """Bind required engine artifact slots to explicit non-admission descriptors."""

    result = dict(plan)
    artifacts = dict(plan["artifacts"])
    abi = str(artifacts["cuda_extension"]["abi"])
    for artifact_id, path, kind in (
        ("qualification_receipt", qualification, "allocation_only_descriptor"),
        ("reporting_receipt", reporting, "allocation_only_descriptor"),
    ):
        artifacts[artifact_id] = {
            "path": str(path),
            "sha256": _sha256(path),
            "kind": kind,
            "abi": abi,
        }
    result["artifacts"] = artifacts
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _inventory(owner: Any) -> dict[str, Any]:
    inventory = inventory_resident_prepared_owner_storages_v1(owner).payload()
    rows = inventory["storage_rows"]
    pointers = [(row["device"], row["data_ptr"]) for row in rows]
    if len(pointers) != len(set(pointers)):
        raise AllocationDiagnosticError("storage inventory double-counted an alias")
    inventory["distinct_storage_pointer_count"] = len(pointers)
    return inventory


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _measurement_decision(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not stages or stages[0].get("stage") != "process_baseline":
        raise AllocationDiagnosticError("diagnostic samples lack a process baseline")
    baseline, teardown = stages[0], stages[-1]
    total = int(baseline["driver_total_bytes"])
    peak = max(
        max(int(sample["process_gpu_bytes"]) for sample in stages),
        max(int(sample["torch_peak_reserved_bytes"]) for sample in stages),
    )
    required_headroom = math.ceil(total * 0.10) + MINIMUM_RESIDUAL_BYTES
    teardown_allocated_delta = max(
        0, int(teardown["torch_allocated_bytes"]) - int(baseline["torch_allocated_bytes"])
    )
    return {
        "conservative_peak_bytes": peak,
        "required_headroom_bytes": required_headroom,
        "available_headroom_bytes": total - peak,
        "headroom_ok": total - peak >= required_headroom,
        "teardown_allocated_delta_bytes": teardown_allocated_delta,
        "teardown_tolerance_bytes": MINIMUM_RESIDUAL_BYTES,
        "teardown_ok": teardown_allocated_delta <= MINIMUM_RESIDUAL_BYTES,
    }


def _diagnostic_cells_plan(plan: Mapping[str, Any], *, cells: int) -> dict[str, Any]:
    """Derive a B12/B24 diagnostic control from the immutable B48 source plan."""

    if cells not in {12, 24, 48}:
        raise AllocationDiagnosticError("allocation diagnostic cells must be B12, B24, or B48")
    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or len(batches) != 1:
        raise AllocationDiagnosticError("allocation diagnostic source requires one physical B48 batch")
    source = batches[0]
    source_cell_ids = source.get("cell_ids") if isinstance(source, Mapping) else None
    if not isinstance(source_cell_ids, (list, tuple)) or len(source_cell_ids) != 48:
        raise AllocationDiagnosticError("allocation diagnostic source must be exact B48")
    result = dict(plan)
    result["physical_batches"] = [
        {
            "batch_id": f"allocation-diagnostic-b{cells:02d}",
            "cell_ids": tuple(source_cell_ids[:cells]),
        }
    ]
    result.pop("plan_sha256", None)
    result["plan_sha256"] = canonical_sha256(result)
    return result


def _selected_source_projections(prepared: Any, *, expected_cells: int) -> tuple[Any, ...]:
    """Return exactly the selected generation-zero projections from one owner."""

    projections = tuple(prepared.source_projections)
    if len(projections) != expected_cells:
        raise AllocationDiagnosticError("prepared owner does not retain all selected source projections")
    return projections


def _requires_full_projection_checkpoint(*, cells: int) -> bool:
    """Only the exact physical B48 artifact carries a restorable checkpoint."""

    return cells == 48


def _record_stage(output_root: Path, stages: list[dict[str, Any]], sample: dict[str, Any]) -> None:
    """Persist diagnostic progress so abrupt native failure remains attributable."""

    stages.append(sample)
    atomic_write_json(
        output_root / "allocation_progress.json",
        {
            "authority": "allocation_diagnostic_progress_not_admission_v1",
            "stages": stages,
        },
    )


def run_allocation_diagnostic_v1(
    spec_path: Path,
    *,
    output_root: Path,
    cells: int = 48,
    eed_overlay: bool = False,
    generation_turnover: bool = False,
    artifact_override: Path | None = None,
    manifest_override: Path | None = None,
    artifact_source_spec: Path | None = None,
) -> dict[str, Any]:
    """Run one cold-process, zero-dispatch B12/B24/B48 allocation diagnostic."""

    allocator_value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_value not in {None, REQUIRED_ALLOCATOR_VALUE}:
        raise AllocationDiagnosticError(
            "PYTORCH_CUDA_ALLOC_CONF must be unset or exactly " + REQUIRED_ALLOCATOR_VALUE
        )
    import torch

    if not torch.cuda.is_available():
        raise AllocationDiagnosticError("CUDA is unavailable")
    spec_path = spec_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise AllocationDiagnosticError(f"diagnostic output root already exists: {output_root}")
    artifact_source_plan = (
        None
        if artifact_source_spec is None
        else compile_campaign(
            load_campaign_spec(artifact_source_spec.expanduser().resolve())
        ).payload()
    )
    source_spec = load_campaign_spec(spec_path)
    if generation_turnover:
        parameters = source_spec.setdefault("parameters", {})
        parameters["production_lifecycle.lifecycle_mode"] = (
            "legacy_synchronized_truncation_v1"
        )
        parameters["production_lifecycle.life_history_profile"] = (
            "legacy_inert_v1"
        )
        parameters["production_lifecycle.turnover_cull_fraction"] = 0.5
        parameters["production_lifecycle.turnover_parent_fraction"] = 1.0
    source_plan = bind_diagnostic_artifact_v1(
        compile_campaign(source_spec).payload(),
        artifact_override=artifact_override,
        manifest_override=manifest_override,
        artifact_source_plan=artifact_source_plan,
    )
    plan = _diagnostic_cells_plan(source_plan, cells=cells)
    batches = plan.get("physical_batches")
    if not isinstance(batches, list) or len(batches) != 1:
        raise AllocationDiagnosticError("exact diagnostic requires exactly one physical batch")
    batch = batches[0]
    if not isinstance(batch, Mapping) or len(batch.get("cell_ids", ())) != cells:
        raise AllocationDiagnosticError("allocation diagnostic physical batch differs from selected cells")
    selected_cell_count = len(batch["cell_ids"])
    execution = plan.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("generations"), execution.get("ticks_per_generation")
    ) != (200, 160):
        raise AllocationDiagnosticError("allocation diagnostic geometry must remain G200/T160")
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AllocationDiagnosticError("diagnostic plan has no artifact bindings")
    artifact = Path(str(artifacts["cuda_extension"]["path"])).resolve()
    manifest = Path(str(artifacts["extension_manifest"]["path"])).resolve()
    if not artifact.is_file() or not manifest.is_file():
        raise AllocationDiagnosticError("exact isolated artifact or manifest is unavailable")
    artifact_sha256, manifest_sha256 = _sha256(artifact), _sha256(manifest)
    if artifact_sha256 != artifacts["cuda_extension"]["sha256"] or manifest_sha256 != artifacts["extension_manifest"]["sha256"]:
        raise AllocationDiagnosticError("exact isolated artifact hash differs from plan")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    bootstrap_floor = math.ceil(int(total_bytes) * 0.10) + MINIMUM_RESIDUAL_BYTES
    if int(free_bytes) < bootstrap_floor:
        raise AllocationDiagnosticError("live CUDA memory does not cover the diagnostic bootstrap floor")
    gpu_processes_before = _gpu_processes()
    output_root.mkdir(parents=True)
    base_plan_sha256 = str(plan["plan_sha256"])
    qualification_path, reporting_path = _diagnostic_descriptor(
        output_root,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        plan_sha256=base_plan_sha256,
    )
    diagnostic_plan = _diagnostic_plan(
        plan, qualification=qualification_path, reporting=reporting_path
    )
    stages: list[dict[str, Any]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _record_stage(output_root, stages, _cuda_sample(torch, "process_baseline"))
    module: Any = None
    prepared: Any = None
    provider: Any = None
    request: Any = None
    adapters: Any = None
    cells: Any = None
    context: Any = None
    sources: Any = None
    physical: Any = None
    projections: Any = None
    outcome: dict[str, Any] = {"status": "unknown", "cuda_oom": False}
    inventory: dict[str, Any] | None = None
    projection_path = output_root / "generation_zero_source_projections.pt"
    try:
        module = _load_exact_module(artifact, eed_overlay=eed_overlay)
        _record_stage(output_root, stages, _cuda_sample(torch, "module_load"))
        provider = AllocationOnlyProvider(bind_committed_defaults=False)
        request = provider.translate_batch(diagnostic_plan, str(batch["batch_id"]))
        adapters = dict(provider._resolved_execution_adapters(diagnostic_plan))
        cells = provider._batch_cells(diagnostic_plan, request)
        context = SimpleNamespace(run_root=output_root, batch_root=output_root, progress_events=[])
        source_preparer = adapters.get("source_preparer") or provider._prepare_generation_zero_sources_v2
        from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
            _wdl_source_capabilities_v2,
        )

        sources = tuple(
            source_preparer(
                diagnostic_plan,
                request,
                context,
                runner_builder=adapters["runner_builder"],
                cells=cells,
                artifact=str(artifact),
                qualification=str(qualification_path),
                reporting=str(reporting_path),
                segment_generations=request.generations,
                source_capability_mask=_wdl_source_capabilities_v2(
                    {"genome_v2": True},
                    jungle_overlay=True,
                    eed_overlay=eed_overlay,
                ),
            )
        )
        _record_stage(output_root, stages, _cuda_sample(torch, "request_construction"))
        first_end = provider._segment_ends(request, start=0)[0]
        _, _, prepared = provider._prepare_resident_segment_owner_v1(
            plan=diagnostic_plan,
            runner_builder=adapters["runner_builder"],
            request_builder=adapters["request_builder"],
            preparer=adapters["preparer"],
            cells=cells,
            cell_ids=request.cell_ids,
            batch_root=output_root,
            start_generation=0,
            end_generation=first_end,
            ticks_per_generation=request.ticks_per_generation,
            artifact=str(artifact),
            qualification=str(qualification_path),
            reporting=str(reporting_path),
            module=module,
            source_projections=sources,
            allocation_generations=provider._agent_id_allocation_generation_count(request, start_generation=0),
            genome_v2_checkpoint_authority_binding=dict(request.genome_v2_checkpoint_authority_binding),
            physical_window_generations=1,
        )
        _record_stage(output_root, stages, _cuda_sample(torch, "canonical_patch_wdl_owner"))
        physical = prepared.physical_window_arguments
        if not hasattr(physical, "jungle_state_tensors"):
            raise AllocationDiagnosticError("prepared owner lacks a Jungle-only overlay")
        _record_stage(output_root, stages, _cuda_sample(torch, "jungle_overlay"))
        if eed_overlay:
            if (
                getattr(physical, "eed_state_tensors", None) is None
                or getattr(physical, "eed_config", None) is None
            ):
                raise AllocationDiagnosticError("prepared owner lacks the EED overlay")
            _record_stage(output_root, stages, _cuda_sample(torch, "eed_overlay"))
        inventory = _inventory(physical)
        _record_stage(output_root, stages, _cuda_sample(torch, "finalized_combined_owner"))
        projections = _selected_source_projections(prepared, expected_cells=selected_cell_count)
        _record_stage(output_root, stages, _cuda_sample(torch, "checkpoint_projection"))
        if _requires_full_projection_checkpoint(cells=selected_cell_count):
            # The physical B48 artifact retains a restorable checkpoint.  The
            # B12/B24 controls measure resident allocation only; serializing
            # their multi-gigabyte CPU projections is neither a checkpoint
            # qualification nor a faithful part of that measurement.
            torch.save(projections, projection_path)
        else:
            atomic_write_json(
                output_root / "control_projection_summary.json",
                {
                    "authority": "allocation_control_projection_summary_v1",
                    "physical_batch_cells": selected_cell_count,
                    "projection_count": len(projections),
                    "full_checkpoint_serialized": False,
                },
            )
        _record_stage(output_root, stages, _cuda_sample(torch, "serialization"))
        outcome = {"status": "prepared", "cuda_oom": False}
    except BaseException as exc:
        if _is_cuda_oom(exc):
            outcome = {"status": "cuda_oom", "cuda_oom": True, "error": f"{type(exc).__name__}: {exc}"}
        else:
            atomic_write_json(
                output_root / "allocation_failure_diagnostic.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "authority": "allocation_failure_diagnostic_not_admission_v1",
                    "cells": cells,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stages": stages,
                },
            )
            raise
    finally:
        projections = None
        physical = None
        prepared = None
        sources = None
        context = None
        cells = None
        adapters = None
        request = None
        provider = None
        module = None
        gc.collect()
        torch.cuda.empty_cache()
        _record_stage(output_root, stages, _cuda_sample(torch, "teardown"))
    decision = _measurement_decision(stages)
    result = {
        "schema_version": (
            "eiel_b48_wdl_jungle_eed_allocation_diagnostic_v1"
            if eed_overlay
            else SCHEMA_VERSION
        ),
        "authority": DIAGNOSTIC_AUTHORITY,
        "repository_head": _git_head(),
        "spec_path": str(spec_path),
        "base_b48_plan_sha256": source_plan["plan_sha256"],
        "base_plan_sha256": base_plan_sha256,
        "diagnostic_plan_sha256": diagnostic_plan["plan_sha256"],
        "route_configuration_sha256": canonical_sha256({"cells": diagnostic_plan["cells"], "execution": diagnostic_plan["execution"]}),
        "artifact": {"path": str(artifact), "sha256": artifact_sha256, "manifest_sha256": manifest_sha256},
        "geometry": {"physical_batch_cells": selected_cell_count, "generations": 200, "ticks_per_generation": 160, "prepared_segment_generations": 50},
        "eed_overlay": eed_overlay,
        "allocator_environment": {"PYTORCH_CUDA_ALLOC_CONF": allocator_value},
        "gpu_processes_before": gpu_processes_before,
        "bootstrap_floor_bytes": bootstrap_floor,
        "stages": stages,
        "storage_inventory": inventory,
        "executor_call_count": 0,
        "scientific_tick_count": 0,
        "outcome": outcome,
        "decision": decision,
        "production_execution_admission": False,
        "qualification_claim": False,
    }
    result["receipt_sha256"] = canonical_sha256(result)
    atomic_write_json(output_root / "allocation_diagnostic_receipt.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cells", type=int, choices=(12, 24, 48), default=48)
    parser.add_argument("--artifact-override", type=Path)
    parser.add_argument("--manifest-override", type=Path)
    parser.add_argument(
        "--generation-turnover",
        action="store_true",
        help=(
            "prepare the legacy synchronized generation-turnover source "
            "(lifecycle ABI 0) instead of the default hybrid source"
        ),
    )
    parser.add_argument(
        "--artifact-source-spec",
        type=Path,
        help="frozen campaign spec supplying the complete current artifact bundle",
    )
    args = parser.parse_args(argv)
    result = run_allocation_diagnostic_v1(
        args.spec,
        output_root=args.output_root,
        cells=args.cells,
        generation_turnover=args.generation_turnover,
        artifact_override=args.artifact_override,
        manifest_override=args.manifest_override,
        artifact_source_spec=args.artifact_source_spec,
    )
    print(json.dumps({"outcome": result["outcome"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
