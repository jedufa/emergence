from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    append_event_sink_columns_v1,
    event_history_digest_from_references_v1,
    event_sink_prefix_identities_v1,
)


try:
    from engine.campaigns.physical_b136_acs_parity_v1 import (
        _checkpoint_digest,
        _physical_plan_authority,
        _validated_run,
        materialize_physical_b136_acs_parity_v1,
    )
except ModuleNotFoundError as exc:
    if exc.name == "engine.campaigns.physical_b136_acs_parity_v1":
        materialize_physical_b136_acs_parity_v1 = None  # type: ignore[assignment]
    else:
        raise


BATCH_ID = "batch-0000"
CELL_IDS = tuple(f"cell-{index:03d}" for index in range(136))
TOTAL_TICKS = 3_360
CHECKPOINT_TICK = 3_359
SYNC_ROUTE = "1" * 64
ASYNC_ROUTE = "2" * 64
TERMINAL_DIGEST = "terminal-seed"
CHECKPOINT_DIGEST = "checkpoint-seed"
EVENT_DIGEST = "c" * 64
EVENT_FRONTIER = TOTAL_TICKS
EVENT_COUNT = EVENT_FRONTIER * len(CELL_IDS)
TRANSITION_SCHEMA = "eiel_execution_restore_process_transition_v1"
PROCESS_SCHEMA = "eiel_execution_process_identity_v1"
SOURCE_GIT = {
    "repository_root": "/repo",
    "commit": "a" * 40,
    "working_tree_clean": False,
    "working_tree_status_sha256": "b" * 64,
    "working_tree_status_lines": [" M frozen-candidate"],
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _process(*, pid: int, instance: str) -> dict[str, Any]:
    return {
        "schema_version": PROCESS_SCHEMA,
        "pid": pid,
        "process_instance_id": instance,
    }


def _transition(*, same_process: bool) -> dict[str, Any]:
    producer = _process(pid=1001, instance="producer-instance")
    restorer = _process(
        pid=1001 if same_process else 2002,
        instance="producer-instance" if same_process else "restorer-instance",
    )
    return {
        "schema_version": TRANSITION_SCHEMA,
        "event": {"batch_id": BATCH_ID, "restore_tick": CHECKPOINT_TICK},
        "producer_process": producer,
        "restore_process": restorer,
    }


def _cell_digest_map(seed: str, cell_ids: tuple[str, ...]) -> dict[str, str]:
    return {
        cell_id: canonical_sha256({"cell_id": cell_id, "seed": seed})
        for cell_id in cell_ids
    }


def _checkpoint(*, tick: int, terminal_cell_digests: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "eiel_tick_native_checkpoint_v1",
        "time_authority": "absolute_tick_v1",
        "absolute_tick": tick,
        "next_absolute_tick": tick,
        "terminal_cell_digests": terminal_cell_digests,
        "canonical_terminal_state_digest_sha256s": terminal_cell_digests,
    }


def _make_run(
    root: Path,
    *,
    route: str,
    acs_slots: int,
    restored: bool,
    cell_ids: tuple[str, ...] = CELL_IDS,
    terminal_digest: str = TERMINAL_DIGEST,
    checkpoint_digest: str = CHECKPOINT_DIGEST,
    event_digest: str = EVENT_DIGEST,
    event_count: int = EVENT_COUNT,
    event_frontier: int = EVENT_FRONTIER,
    high_water: int | None = None,
    same_process: bool = False,
    loaded_digest_mismatch: bool = False,
    malformed_process: bool = False,
    extra_process_field: bool = False,
) -> None:
    high_water = (
        (1 if restored and acs_slots == 2 else acs_slots)
        if high_water is None
        else high_water
    )
    terminal_cell_digests = _cell_digest_map(terminal_digest, cell_ids)
    checkpoint_cell_digests = _cell_digest_map(checkpoint_digest, cell_ids)
    qualification_key = {"profile": "physical-b136-acs-parity"}
    plan_body: dict[str, Any] = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "tw-physical-b136-acs-parity-v1",
        "resolved_profile": "eiel-production-runtime-v1",
        "run_contract": {
            "schema_version": "eiel_resolved_run_contract_v1",
            "requested": {
                "profile_id": "eiel_full_evolution_v1",
                "mode": "strict",
            },
            "admission_status": "passed",
        },
        "resource_estimate": {
            "resident_event_capacity": 41804,
            "resident_event_row_bytes": 141,
            "resident_event_bytes": 801633504,
            "compiled_batch_count": 1,
            "compiled_physical_batch_cells_total": 136,
            "compiled_partition": False,
            "resource_estimator_assumptions": {
                "frozen_reserve_bytes": 413138944,
            },
        },
        "qualification_key": qualification_key,
        "qualification_key_sha256": canonical_sha256(qualification_key),
        "artifacts": {
            name: {"sha256": character * 64}
            for name, character in (
                ("cuda_extension", "1"),
                ("extension_manifest", "2"),
                ("qualification_receipt", "3"),
            )
        },
        "artifact_lifecycle": {
            "schema_version": "eiel_run_artifact_lifecycle_contract_v1",
            "contract_sha256": "4" * 64,
        },
        "execution_route_sha256": route,
        "execution": {
            "execution_route_identity": "tick_native_execution_route_v1",
            "total_ticks": TOTAL_TICKS,
            "checkpoint_ticks": [CHECKPOINT_TICK, TOTAL_TICKS],
            "acs_checkpoint_slots": acs_slots,
            "checkpoint_storage_mode": "disk",
            "native_streamed_checkpoint": True,
            "acs_capture_instrumentation": True,
            "budgets": {
                "vram_bytes": 5_786_503_741,
                "host_bytes": 34_359_738_368,
                "minimum_free_gpu_bytes": 536_870_912,
                "maximum_gpu_memory_slope_bytes_per_window": 67_108_864,
                "maximum_host_memory_slope_bytes_per_window": 67_108_864,
            },
            "tick_native_control": {
                "event_window_capacity": 41804,
                "resident_owner_span_ticks": 1120,
                "native_dispatch_window_ticks": 1120,
                "source_window_ticks": 160,
            },
            "technical_gate": {"partition": False},
            **({"deliberate_restore_ticks": [CHECKPOINT_TICK]} if restored else {}),
        },
        "output": {"runtime_root": str(root)},
        "cells": [
            {
                "cell_id": cell_id,
                "parameters": {
                    "patch.grid_size": 32,
                    "patch.initial_population": 256,
                    "patch.capacity": 512,
                    "tw_rngv2.semantics_version": 2,
                    "cuda_architecture_b.cpu_request_staging_enable": False,
                    "production_lifecycle.ordinary_births_enabled": True,
                },
            }
            for cell_id in cell_ids
        ],
        "physical_batches": [{"batch_id": BATCH_ID, "cell_ids": list(cell_ids)}],
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    _write_json(root / "execution_plan.json", plan)

    terminal_checkpoint = _checkpoint(
        tick=TOTAL_TICKS,
        terminal_cell_digests=terminal_cell_digests,
    )
    split_checkpoint = _checkpoint(
        tick=CHECKPOINT_TICK,
        terminal_cell_digests=checkpoint_cell_digests,
    )
    transition = _transition(same_process=same_process) if restored else None
    window_count = 1 if restored else 3
    native_receipts = []
    for index in range(window_count):
        start_tick = CHECKPOINT_TICK if restored else index * 1120
        end_tick = TOTAL_TICKS if restored else (index + 1) * 1120
        native_receipts.append(
            {
                "external_event_history_digest": event_digest,
                "resource_window_measurement": {
                    "start_tick": start_tick,
                    "end_tick": end_tick,
                    "peak_gpu_allocated_bytes": 5_000_000_000,
                    "peak_gpu_reserved_bytes": 5_100_000_000,
                    "minimum_free_gpu_bytes": 600_000_000,
                    "host_peak_rss_bytes": 1_000_000_000,
                    "host_terminal_rss_bytes": 900_000_000,
                    "resident_event_bytes": 801_633_504,
                    "resident_source_bytes": 10_000_000,
                    "event_ring_capacity": 41804 * 136,
                    "event_ring_peak_occupancy": 0,
                    "external_event_history_count": event_count,
                    "terminal_event_ring_occupancy": 0,
                    "host_samples": [
                        {
                            "rss_bytes": 1_000_000_000,
                            "rss_high_water_bytes": 1_000_000_000,
                            "pinned_bytes": 0,
                        }
                    ],
                },
            }
        )
    capture_count = 1 if restored else (2 if acs_slots == 2 else 1)
    captures = [
        {
            "coordinate": CHECKPOINT_TICK if restored else (index + 1) * 1120,
            "captured_bytes": 100_000_000,
            "last_useful_kernel_completion_observed_host_time_ns": (
                (CHECKPOINT_TICK if restored else (index + 1) * 1120) * 10 - 21
            ),
            "host_resources_before": {"pinned_bytes": 0},
            "host_resources_peak": {"pinned_bytes": 0},
            "host_resources_after": {"pinned_bytes": 0},
        }
        for index in range(capture_count)
    ]
    provider_result: dict[str, Any] = {
        "status": "completed",
        "checkpoint_catalog": {
            str(CHECKPOINT_TICK): split_checkpoint,
            str(TOTAL_TICKS): terminal_checkpoint,
        },
        "native_execution_receipts": native_receipts,
        "acs_capture_measurements": captures,
        "resident_owner_max_simultaneous": 1,
        "resident_owner_live_count_terminal": 0,
        "expected_resident_owner_transition_ticks": (
            [1120, 2240] if acs_slots == 2 and not restored else []
        ),
        "observed_resident_owner_transition_ticks": (
            [1120, 2240] if acs_slots == 2 and not restored else []
        ),
        "resident_owner_epoch_transfer_receipts": (
            [
                {
                    "schema_version": "eiel_resident_owner_epoch_transfer_v1",
                    "physical_owner_reused": True,
                    "cuda_allocation_recreated": False,
                    "simultaneous_physical_owner_count": 1,
                    "boundary_next_absolute_tick": tick,
                }
                for tick in (1120, 2240)
            ]
            if acs_slots == 2 and not restored
            else []
        ),
        "compact_segment_settlements": [
            {
                "end_next_tick": TOTAL_TICKS,
                "event_frontier_by_cell": [event_frontier] * len(cell_ids),
            }
        ],
        "acs_checkpoint_staging": {
            "schema_version": "eiel_acs_native_detached_slots_v1",
            "capture_mode": "detached_file_backed_native_v1",
            "slots": acs_slots,
            "queue_occupancy_high_water": high_water,
            "queue_occupancy": 0,
            "failure_latched": False,
            "aborted": False,
            "abandoned_snapshot_count": 0,
            "slot_states": ["FREE"] * acs_slots,
            "closed": True,
            "writer_alive": False,
            "durable_checkpoint_frontier": TOTAL_TICKS,
            "worker_accepts_cuda_objects": False if acs_slots else False,
            "worker_accepts_callbacks": False if acs_slots else False,
            "worker_tensor_device_policy": "forced_cpu_context_v1",
            "queued_cuda_references": 0,
            "queued_cuda_callbacks": 0,
            "temporary_capture_file_high_water": high_water,
            "temporary_capture_files_active": 0,
            "detached_source_lease_count": 0,
            "source_consumer_leases_active": 0,
            "source_lease_spool_bytes_active": 0,
            "capture_pool_count": 1,
            "capture_pool_model": "one_shared_two_slab_pool_not_per_slot",
        },
    }
    if acs_slots == 2:
        provider_result["tick_native_performance_timeline_binding"] = {
            "schema_version": "eiel_tick_native_performance_timeline_binding_v1",
            "timeline": {
                "schema_version": "eiel_tick_native_performance_timeline_v1",
                "strict": True,
                "gpu_performance_window": {
                    "post_simulation_acs_tail_excluded": True,
                    "start_ns": 1,
                    "end_ns": 101,
                    "terminal_checkpoint_tick": TOTAL_TICKS,
                },
                "dispatch_overhead": {
                    "definition": "unclassified_simulation_dispatch_domain_fraction_v1",
                    "dispatch_domain_start_ns": 1,
                    "dispatch_domain_end_ns": 101,
                    "dispatch_span_ns": 100,
                    "simulation_dispatch_union_ns": 99,
                    "classified_non_dispatch_union_ns": 0,
                    "capture_detach_union_ns": 0,
                    "persistence_union_ns": 0,
                    "owner_preparation_union_ns": 0,
                    "dispatch_gap_ns": 1,
                    "dispatch_domain_ns": 100,
                    "acs_tail_excluded": True,
                    "fraction": 0.01,
                },
                "dispatch_overhead_fraction": 0.01,
                "post_simulation_acs_tail": {
                    "excluded_from_gpu_performance_metrics": True,
                    "metric_name": "T_ACS_TAIL",
                },
                "wall_time_reconciliation": {"pass": True},
            },
        }
    if restored:
        if malformed_process:
            transition["restore_process"]["pid"] = "bad"
        if extra_process_field:
            transition["restore_process"]["unexpected"] = True
        reload_receipts = [
            {
                "cell_id": cell_id,
                "bundle_path": str(root / "bundles" / cell_id),
                "source_projection_sha256": checkpoint_cell_digests[cell_id],
                "loaded_projection_sha256": (
                    "bad-loaded-digest"
                    if loaded_digest_mismatch and index == 0
                    else checkpoint_cell_digests[cell_id]
                ),
            }
            for index, cell_id in enumerate(cell_ids)
        ]
        provider_result["fresh_owner_restore_receipt"] = {
            "schema_version": "eiel_patch_fixed_fresh_owner_restore_receipt_v1",
            "projection_bundles_reloaded": True,
            "cell_ids": list(cell_ids),
            "source_checkpoint_next_absolute_tick": CHECKPOINT_TICK,
            "restore_next_tick": CHECKPOINT_TICK,
            "first_executed_absolute_tick": CHECKPOINT_TICK,
            "first_committed_next_tick": CHECKPOINT_TICK + 1,
            "off_by_one_check": "PASS",
            "source_checkpoint_cell_digests": checkpoint_cell_digests,
            "reload_receipts": reload_receipts,
            "source_execution_route_sha256": route,
            "restore_execution_route_sha256": route,
            "process_transition": transition,
        }

    batch_path = root / "batches" / BATCH_ID / "manifest.json"
    _write_json(
        batch_path,
        {
            "schema_version": "eiel_campaign_batch_manifest_v1",
            "campaign_id": plan["campaign_id"],
            "batch_id": BATCH_ID,
            "cell_ids": list(cell_ids),
            "status": "completed",
            "plan_sha256": plan["plan_sha256"],
            "execution_route_sha256": route,
            "checkpoint": terminal_checkpoint,
            "result": provider_result,
        },
    )
    campaign_path = root / "campaign_manifest.json"
    _write_json(
        campaign_path,
        {
            "schema_version": "eiel_campaign_manifest_v1",
            "campaign_id": plan["campaign_id"],
            "status": "completed",
            "plan_sha256": plan["plan_sha256"],
            "execution_route_sha256": route,
            "batches": [
                {
                    "batch_id": BATCH_ID,
                    "status": "completed",
                    "manifest_path": str(batch_path),
                }
            ],
        },
    )
    campaign_spec = str(root / ("restored.yaml" if restored else "uninterrupted.yaml"))
    source_identity = {
        "campaign_spec_path": campaign_spec,
        "campaign_spec_sha256": canonical_sha256({"campaign_spec": campaign_spec}),
        "compiled_plan_sha256": canonical_sha256({"compiled": plan["campaign_id"]}),
        "bound_plan_sha256": plan["plan_sha256"],
        "git": SOURCE_GIT,
    }
    baseline = {
        "schema_version": "eiel_baseline_declaration_v1",
        "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
        "intent": "control" if acs_slots == 0 else "development",
        "deviations": (
            [
                {
                    "requirement_id": "checkpoint.acs_two_slot",
                    "reason": "ACS0 is the declared synchronous parity control.",
                }
            ]
            if acs_slots == 0
            else []
        ),
    }
    request_body = {
        "schema_version": "eiel_execution_request_v2",
        "purpose": "qualprep",
        "campaign_spec": campaign_spec,
        "output_root": str(root),
        "resource_wait": {
            "enabled": True,
            "poll_seconds": 300,
            "max_wait_seconds": None,
        },
        "qualification_profile": None,
        "baseline": baseline,
        **(
            {"restore_process_isolation": "fresh_invocation_per_restore"}
            if restored
            else {}
        ),
    }
    request_sha = canonical_sha256(request_body)
    request_path = root / "execution_request.json"
    _write_json(
        request_path,
        {
            "schema_version": "eiel_execution_request_receipt_v2",
            "request": {**request_body, "request_sha256": request_sha},
            "request_sha256": request_sha,
            "bound_plan_sha256": plan["plan_sha256"],
            "source_identity": source_identity,
            "run_root": str(root),
        },
    )
    execution_receipt: dict[str, Any] = {
        "campaign_manifest_path": str(campaign_path),
        "execution_route_sha256": route,
    }
    if transition is not None:
        execution_receipt["deliberate_restore_ticks_completed"] = [CHECKPOINT_TICK]
        execution_receipt["resume_invocations"] = 1
        execution_receipt["restore_process_transitions_completed"] = [transition]
    preflight = {
        "schema_version": "eiel_campaign_preflight_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "ready_for_execution": True,
        "execution_authority_scope": "qualprep_development_only",
        "reasons": [],
        "resources": {
            "physical_b136_shape": {
                "applicable": True,
                "status": "PASS",
                "logical_cells": 136,
                "compiled_batch_count": 1,
                "compiled_batch_sizes": [136],
                "technical_gate_partition": False,
                "event_window_capacity": 41804,
                "expected_event_window_capacity": 41804,
            },
            "provider_estimate": {
                "resident_event_rows_per_cell": 41804,
                "resident_event_row_bytes": 141,
                "resident_event_bytes": 801633504,
                "resource_estimator_assumptions": {
                    "frozen_reserve_bytes": 413138944,
                },
            },
            "cuda": {
                "admission_required": True,
                "budget_within_90_percent_total": True,
                "declared_vram_budget_bytes": 5_786_503_741,
                "free_bytes": 5_823_660_032,
                "free_covers_admission_bound_and_headroom": True,
                "free_covers_measurement_bootstrap_floor": True,
                "live_admission_bound_bytes": 5_441_799_168,
                "live_admission_bound_source": "provider_exact_upper_bound",
                "measurement_bootstrap_minimum_free_bytes": 241_104_323,
                "minimum_residual_headroom_bytes": 0,
                "observed_runtime_environment": {},
                "projection_fits_total_memory_with_headroom": False,
                "provider_estimate_within_budget": True,
                "required_headroom_bytes": 241_104_323,
                "runtime_environment_ok": True,
                "runtime_environment_requirements": {},
                "total_bytes": 6_027_608_064,
            },
        },
    }
    _write_json(root / "preflight_receipt.json", preflight)
    result_body = {
        "schema_version": "eiel_execution_result_v2",
        "invocation_mode": "execute",
        "purpose": request_body["purpose"],
        "campaign_id": plan["campaign_id"],
        "compiled_plan_sha256": source_identity["compiled_plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "run_root": str(root),
        "preflight_receipt_path": str(root / "preflight_receipt.json"),
        "preflight_receipt_sha256": canonical_sha256(preflight),
        "status": "completed",
        "result_class": "SUCCESS",
        "ok": True,
        "execution_status": "completed",
        "plan_sha256": plan["plan_sha256"],
        "request_sha256": request_sha,
        "source_identity": source_identity,
        "execution_route_sha256": route,
        "mechanical_verification": {"status": "PASS"},
        "baseline": {
            "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
            "admitted": True,
            "operational_status": "CONFORMANT",
            "overall_status": (
                "VALID_DECLARED_DEVIATION" if acs_slots == 0 else "BASELINE_CONFORMANT"
            ),
            "deviations": (
                [
                    {
                        "requirement_id": "checkpoint.acs_two_slot",
                        "reason": baseline["deviations"][0]["reason"],
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
                if acs_slots == 0
                else []
            ),
        },
        "execution_receipt": execution_receipt,
    }
    _write_json(
        root / "execution_result.json",
        {**result_body, "result_sha256": canonical_sha256(result_body)},
    )


def _make_guarded_receipt(root: Path, output: Path) -> None:
    batch_path = root / "batches" / BATCH_ID / "manifest.json"
    request_receipt = _read_json(root / "execution_request.json")
    request_path = output.parent / f"{root.name}-guarded-execution-request.json"
    _write_json(request_path, request_receipt["request"])
    windows = []
    for coordinate in (1120, 2240):
        windows.append(
            {
                "coordinate": coordinate,
                "nonterminal": True,
                "capture_then_writer_ordered": True,
                "source_lease_ordered": True,
                "successor_dispatch_ordered": True,
                "capture_clock_ordered": True,
                "gpu_safe_release_ordered": True,
                "capture_started_perf_ns": coordinate * 10 - 20,
                "capture_completed_perf_ns": coordinate * 10 - 1,
                "capture_barrier_time_ns": 1,
                "capture_submitted_perf_ns": coordinate * 10,
                "writer_started_perf_ns": coordinate * 10,
                "source_released_perf_ns": coordinate * 10 + 1,
                "release_authority_perf_ns": coordinate * 10 + 1,
                "release_authority_kind": "source",
                "successor_dispatch_started_perf_ns": coordinate * 10 + 1,
                "successor_last_useful_kernel_completion_observed_host_time_ns": (
                    coordinate * 10 + 2
                ),
            }
        )
    overlap = {
        "schema_version": "eiel_acs_async_overlap_evidence_v5",
        "passed": True,
        "slots": 2,
        "queue_occupancy_high_water": 2,
        "worker_accepts_cuda_objects": False,
        "worker_tensor_device_policy": "forced_cpu_context_v1",
        "worker_accepts_callbacks": False,
        "queued_cuda_references": 0,
        "queued_cuda_callbacks": 0,
        "temporary_capture_file_high_water": 2,
        "source_file_lifecycle_bounded": True,
        "physical_owner_continuity_measured": True,
        "exact_owner_transition_coverage": True,
        "expected_owner_transition_ticks": [1120, 2240],
        "observed_owner_transition_ticks": [1120, 2240],
        "owner_epoch_transfer_ticks": [1120, 2240],
        "checkpoint_windows": windows,
    }
    activity = {
        "sample_count": 10,
        "active_sample_fraction": 0.9,
        "gpu_sm_percent": {"mean": 60.0},
        "observed_interval_seconds": {"max": 1.0},
        "longest_idle_gap_observed_span_seconds": 1.0,
        "requested_sample_interval_seconds": 1.0,
    }
    batch = _read_json(batch_path)
    source_git = _read_json(root / "execution_result.json")["source_identity"]["git"]
    _write_json(
        output,
        {
            "schema_version": "eiel_acs_gpu_telemetry_receipt_v1",
            "source_commit": source_git["commit"],
            "status": "passed",
            "launched": True,
            "process_attributed": True,
            "exclusive_gpu_window": True,
            "no_competing_gpu_workloads": True,
            "require_async_checkpoint_overlap": True,
            "require_post_release_zero_residency": True,
            "launched_pid": 1234,
            "target_gpu_pids": [1234],
            "runtime_samples": [
                {
                    "target_gpu_pids": [1234],
                    "target_process_tree_pids": [1234],
                },
                {
                    "target_gpu_pids": [],
                    "target_process_tree_pids": [1234],
                },
            ],
            "manifest_path": str(batch_path),
            "manifest_sha256": canonical_sha256(batch),
            "execution_request_path": str(request_path),
            "execution_request_sha256": hashlib.sha256(
                request_path.read_bytes()
            ).hexdigest(),
            "command": {
                "argv": [
                    str(Path(source_git["repository_root"]) / "tools" / "bin" / "eiel"),
                    "execute",
                    str(request_path),
                    "--human",
                ]
            },
            "async_checkpoint_overlap": overlap,
            "post_analysis_summary": {
                "gpu_activity": activity,
                "terminal_gpu_release_boundary": {
                    "schema_version": "eiel_terminal_gpu_release_boundary_v1",
                    "terminal_checkpoint_tick": TOTAL_TICKS,
                    "released_perf_ns": 101,
                    "cuda_device_reset_return_code": 0,
                    "performance_window_end_bound": True,
                },
                "post_release_target_gpu_residency": {
                    "schema_version": "eiel_post_release_target_gpu_residency_v1",
                    "released_perf_ns": 101,
                    "settlement_grace_seconds": 0.0,
                    "observation_not_before_perf_ns": 101,
                    "sample_window_clock_bound": True,
                    "straddling_sample_count": 0,
                    "launched_pid": 1234,
                    "attributed_target_gpu_pids": [1234],
                    "post_release_sample_count": 2,
                    "resident_sample_count": 0,
                    "max_framebuffer_memory_mib": 0.0,
                    "resident_rows": [],
                    "passed": True,
                },
            },
        },
    )


def _make_quadruplet(
    tmp_path: Path,
    *,
    sync_kwargs: dict[str, Any] | None = None,
    sync_restored_kwargs: dict[str, Any] | None = None,
    async_kwargs: dict[str, Any] | None = None,
    async_restored_kwargs: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    sync_root = tmp_path / "sync"
    sync_restored_root = tmp_path / "sync-restored"
    async_root = tmp_path / "async"
    async_restored_root = tmp_path / "async-restored"
    sync_config = {"route": SYNC_ROUTE, "acs_slots": 0, "restored": False}
    sync_config.update(sync_kwargs or {})
    sync_restored_config = {
        "route": SYNC_ROUTE,
        "acs_slots": 0,
        "restored": True,
    }
    sync_restored_config.update(sync_restored_kwargs or {})
    async_config = {"route": ASYNC_ROUTE, "acs_slots": 2, "restored": False}
    async_config.update(async_kwargs or {})
    async_restored_config = {
        "route": ASYNC_ROUTE,
        "acs_slots": 2,
        "restored": True,
    }
    async_restored_config.update(async_restored_kwargs or {})
    _make_run(sync_root, **sync_config)
    _make_run(sync_restored_root, **sync_restored_config)
    _make_run(async_root, **async_config)
    _make_run(async_restored_root, **async_restored_config)
    telemetry = tmp_path / "acs2-guarded-telemetry.json"
    _make_guarded_receipt(async_root, telemetry)
    return (
        sync_root,
        sync_restored_root,
        async_root,
        async_restored_root,
        tmp_path / "parity.json",
        telemetry,
    )


def _materialize(
    sync_root: Path,
    sync_restored_root: Path,
    async_root: Path,
    async_restored_root: Path,
    output_path: Path,
    telemetry_path: Path,
) -> dict[str, Any]:
    if materialize_physical_b136_acs_parity_v1 is None:
        pytest.skip("physical B136 ACS parity implementation is not available")
    return materialize_physical_b136_acs_parity_v1(
        sync_root,
        sync_restored_root,
        async_root,
        async_restored_root,
        output_path,
        telemetry_path,
    )


def test_materializes_physical_b136_acs_parity_pass(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)

    receipt = _materialize(*roots)

    assert receipt["status"] == "PASS"
    output_path = roots[-2]
    assert output_path.is_file()
    assert _read_json(output_path) == receipt


def test_physical_thresholds_accept_legacy_preflight_shape(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)

    def use_legacy_threshold(preflight: dict[str, Any]) -> None:
        preflight["resources"].pop("cuda")
        preflight["cuda"] = {"minimum_free_gpu_bytes": 536_870_912}

    for root in roots[:4]:
        _rewrite_preflight(root, use_legacy_threshold)

    receipt = _materialize(*roots)

    assert receipt["status"] == "PASS"


def test_physical_thresholds_reject_plan_threshold_drift(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)
    run = _validated_run(roots[0])
    run["plan"]["execution"]["budgets"]["minimum_free_gpu_bytes"] = (
        536_870_911
    )

    with pytest.raises(Exception, match="minimum free VRAM threshold"):
        _physical_plan_authority(run)


def test_checkpoint_digest_prefers_canonical_state_over_route_projection() -> None:
    projection = _cell_digest_map("route-projection", CELL_IDS)
    canonical = _cell_digest_map("canonical-state", CELL_IDS)
    digest, values = _checkpoint_digest(
        {
            "time_authority": "absolute_tick_v1",
            "absolute_tick": TOTAL_TICKS,
            "next_absolute_tick": TOTAL_TICKS,
            "terminal_cell_digests": projection,
            "canonical_terminal_state_digest_sha256s": canonical,
        },
        cell_ids=CELL_IDS,
        next_tick=TOTAL_TICKS,
        name="test checkpoint",
    )

    assert values == canonical
    assert digest == canonical_sha256(canonical)

    with pytest.raises(Exception, match="canonical state digests"):
        _checkpoint_digest(
            {
                "time_authority": "absolute_tick_v1",
                "absolute_tick": TOTAL_TICKS,
                "next_absolute_tick": TOTAL_TICKS,
                "terminal_cell_digests": projection,
            },
            cell_ids=CELL_IDS,
            next_tick=TOTAL_TICKS,
            name="missing canonical checkpoint",
        )


def test_terminal_frontier_accepts_native_prefix_authority_without_settlement(
    tmp_path: Path,
) -> None:
    run_kwargs = {"event_count": len(CELL_IDS), "event_frontier": 1}
    roots = _make_quadruplet(
        tmp_path,
        sync_kwargs=run_kwargs,
        sync_restored_kwargs=run_kwargs,
        async_kwargs=run_kwargs,
        async_restored_kwargs=run_kwargs,
    )
    for root in roots[:4]:
        sink_path = root / "event_history_v1" / "events.bin"
        columns = [
            torch.zeros(1, dtype=EVENT_COLUMN_DTYPES[name])
            for name in EVENT_COLUMN_NAMES
        ]
        for cell_index in range(len(CELL_IDS)):
            append_event_sink_columns_v1(
                sink_path,
                cell_index=cell_index,
                columns=columns,
                sequence_base=0,
                expected_cells=len(CELL_IDS),
            )
        references_by_cell = event_sink_prefix_identities_v1(
            sink_path,
            expected_rows_by_cell={index: 1 for index in range(len(CELL_IDS))},
        )
        references = [references_by_cell[index] for index in range(len(CELL_IDS))]

        def remove_settlement_and_bind_prefixes(result: dict[str, Any]) -> None:
            result["compact_segment_settlements"] = []
            terminal_receipt = result["native_execution_receipts"][-1]
            terminal_receipt["event_history_prefix_references_by_cell"] = references
            terminal_receipt["external_event_history_digest"] = (
                event_history_digest_from_references_v1(references)
            )

        _rewrite_batch_result(
            root,
            remove_settlement_and_bind_prefixes,
        )
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    telemetry["manifest_sha256"] = canonical_sha256(
        _read_json(roots[2] / "batches" / BATCH_ID / "manifest.json")
    )
    _write_json(telemetry_path, telemetry)

    receipt = _materialize(*roots)

    assert receipt["status"] == "PASS"

    corrupted_sink = roots[0] / "event_history_v1" / "events.bin"
    with corrupted_sink.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(Exception, match="prefix reference is invalid"):
        _materialize(
            roots[0],
            roots[1],
            roots[2],
            roots[3],
            tmp_path / "tampered-parity.json",
            roots[5],
        )


def test_real_compiled_plan_binds_resolved_and_requested_profiles() -> None:
    plan = compile_campaign(
        load_campaign_spec(
            Path("_spec/campaigns/v1/tw_rngv2/")
            / "science_representative_3360tick_b136_acs2_v1.yaml"
        )
    ).payload()
    assert plan["resolved_profile"] == "eiel-production-runtime-v1"
    assert plan["run_contract"]["requested"] == {
        "profile_id": "eiel_full_evolution_v1",
        "mode": "strict",
    }


def test_fresh_restore_uses_structural_high_water_one(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)
    receipt = _materialize(*roots)
    assert receipt["acs_telemetry"]["fresh_restore_queue_occupancy_high_water"] == 1


def test_rejects_tampered_guarded_manifest_binding(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    telemetry["manifest_sha256"] = "f" * 64
    _write_json(telemetry_path, telemetry)
    with pytest.raises(Exception, match="bound|batch|source"):
        _materialize(*roots)


def test_rejects_guarded_receipt_without_required_front_door_command(
    tmp_path: Path,
) -> None:
    roots = _make_quadruplet(tmp_path)
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    telemetry["command"]["argv"] = ["python", "run.py"]
    _write_json(telemetry_path, telemetry)
    with pytest.raises(Exception, match="bound|batch|request|source"):
        _materialize(*roots)


@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "execution_request_path",
        "execution_request_sha256",
        "process_attributed",
        "exclusive_gpu_window",
    ],
)
def test_rejects_tampered_guarded_identity_fields(tmp_path: Path, field: str) -> None:
    roots = _make_quadruplet(tmp_path)
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    telemetry[field] = (
        False if field in {"process_attributed", "exclusive_gpu_window"} else "tampered"
    )
    _write_json(telemetry_path, telemetry)
    with pytest.raises(Exception, match="bound|batch|request|source"):
        _materialize(*roots)


def test_rejects_missing_dispatch_threshold_evidence(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)
    _rewrite_batch_result(
        roots[2],
        lambda result: result["tick_native_performance_timeline_binding"][
            "timeline"
        ].pop("dispatch_overhead_fraction"),
    )
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    telemetry["manifest_sha256"] = canonical_sha256(
        _read_json(roots[2] / "batches" / BATCH_ID / "manifest.json")
    )
    _write_json(telemetry_path, telemetry)
    with pytest.raises(Exception, match="dispatch overhead|decide"):
        _materialize(*roots)


def test_successor_dispatch_start_bound_allows_long_successor_dispatch(
    tmp_path: Path,
) -> None:
    roots = _make_quadruplet(tmp_path)
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    window = telemetry["async_checkpoint_overlap"]["checkpoint_windows"][0]
    prior_useful = window["coordinate"] * 10 - 21
    dispatch_started = prior_useful + 10_000_000_000
    window["successor_dispatch_started_perf_ns"] = dispatch_started
    window["successor_last_useful_kernel_completion_observed_host_time_ns"] = (
        dispatch_started + 10_000_000_001
    )
    _write_json(telemetry_path, telemetry)

    assert _materialize(*roots)["status"] == "PASS"


def test_rejects_successor_dispatch_start_delay_over_ten_seconds(
    tmp_path: Path,
) -> None:
    roots = _make_quadruplet(tmp_path)
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    window = telemetry["async_checkpoint_overlap"]["checkpoint_windows"][0]
    prior_useful = window["coordinate"] * 10 - 21
    window["successor_dispatch_started_perf_ns"] = (
        prior_useful + 10_000_000_001
    )
    _write_json(telemetry_path, telemetry)

    with pytest.raises(Exception, match="timing|10-second|successor"):
        _materialize(*roots)


def test_rejects_missing_bound_batch_prior_useful_completion(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)
    batch_path = roots[2] / "batches" / BATCH_ID / "manifest.json"
    batch = _read_json(batch_path)
    batch["result"]["acs_capture_measurements"][0].pop(
        "last_useful_kernel_completion_observed_host_time_ns"
    )
    _write_json(batch_path, batch)
    telemetry = _read_json(roots[-1])
    telemetry["manifest_sha256"] = canonical_sha256(batch)
    _write_json(roots[-1], telemetry)

    with pytest.raises(Exception, match="prior useful completion"):
        _materialize(*roots)


@pytest.mark.parametrize("field", ["queued_cuda_references", "queued_cuda_callbacks"])
def test_rejects_missing_guarded_queue_zero_authority(
    tmp_path: Path, field: str
) -> None:
    roots = _make_quadruplet(tmp_path)
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    telemetry["async_checkpoint_overlap"].pop(field)
    _write_json(telemetry_path, telemetry)
    with pytest.raises(Exception, match="exact two-slot owner contract"):
        _materialize(*roots)


@pytest.mark.parametrize(
    ("label", "mutate", "pattern"),
    [
        (
            "minimum free VRAM",
            lambda result: [
                row["resource_window_measurement"].__setitem__(
                    "minimum_free_gpu_bytes", 1
                )
                for row in result["native_execution_receipts"]
            ],
            "VRAM|resource|minimum",
        ),
        (
            "RSS peak",
            lambda result: [
                row["resource_window_measurement"].__setitem__(
                    "host_peak_rss_bytes", 40_000_000_000
                )
                for row in result["native_execution_receipts"]
            ],
            "host|resource",
        ),
        (
            "host RSS slope",
            lambda result: result["native_execution_receipts"][-1][
                "resource_window_measurement"
            ].__setitem__("host_terminal_rss_bytes", 2_000_000_000),
            "host|slope|resource",
        ),
        (
            "pinned two-slab bound",
            lambda result: result["native_execution_receipts"][0][
                "resource_window_measurement"
            ]["host_samples"][0].__setitem__("pinned_bytes", 300_000_000),
            "pinned|slab",
        ),
        (
            "owner overlap",
            lambda result: result.__setitem__("resident_owner_max_simultaneous", 2),
            "owner|overlap",
        ),
        (
            "wrong owner transitions",
            lambda result: result.__setitem__(
                "observed_resident_owner_transition_ticks", [1120, 2239]
            ),
            "owner|1120|2240|transition",
        ),
        (
            "queued CUDA callbacks",
            lambda result: result["acs_checkpoint_staging"].__setitem__(
                "worker_accepts_callbacks", True
            ),
            "queue|host-only|callback",
        ),
        (
            "temporary files",
            lambda result: result["acs_checkpoint_staging"].__setitem__(
                "temporary_capture_file_high_water", 3
            ),
            "queue|high-water|temporary",
        ),
        (
            "missing queued CUDA references",
            lambda result: result["acs_checkpoint_staging"].pop(
                "queued_cuda_references"
            ),
            "queue|host-only|CUDA|reference|zero",
        ),
        (
            "missing queued CUDA callbacks",
            lambda result: result["acs_checkpoint_staging"].pop(
                "queued_cuda_callbacks"
            ),
            "queue|host-only|callback|zero",
        ),
        (
            "intermediate host RSS slope",
            lambda result: result["native_execution_receipts"][1][
                "resource_window_measurement"
            ].__setitem__("host_terminal_rss_bytes", 2_000_000_000),
            "host|slope|resource",
        ),
        (
            "missing host pinned sample",
            lambda result: result["native_execution_receipts"][0][
                "resource_window_measurement"
            ]["host_samples"][0].pop("pinned_bytes"),
            "host sample|pinned",
        ),
        (
            "event ring occupancy",
            lambda result: result["native_execution_receipts"][0][
                "resource_window_measurement"
            ].__setitem__("terminal_event_ring_occupancy", 41804 * 136 + 1),
            "resident capacity arithmetic",
        ),
    ],
)
def test_physical_thresholds_fail_closed_for_provider_tampering(
    tmp_path: Path, label: str, mutate: Any, pattern: str
) -> None:
    del label
    roots = _make_quadruplet(tmp_path)
    _rewrite_batch_result(roots[2], mutate)
    telemetry = _read_json(roots[-1])
    telemetry["manifest_sha256"] = canonical_sha256(
        _read_json(roots[2] / "batches" / BATCH_ID / "manifest.json")
    )
    _write_json(roots[-1], telemetry)
    with pytest.raises(Exception, match=pattern):
        _materialize(*roots)


@pytest.mark.parametrize(
    ("label", "mutate", "pattern"),
    [
        (
            "reserve authority",
            lambda preflight: preflight["resources"]["provider_estimate"][
                "resource_estimator_assumptions"
            ].__setitem__("frozen_reserve_bytes", 1),
            "reserve|threshold|authority",
        ),
        (
            "event capacity arithmetic",
            lambda preflight: preflight["resources"]["provider_estimate"].__setitem__(
                "resident_event_bytes", 1
            ),
            "event|capacity|bytes|authority",
        ),
        (
            "missing provider event bytes authority",
            lambda preflight: preflight["resources"]["provider_estimate"].pop(
                "resident_event_bytes"
            ),
            "event bytes|canonical authority",
        ),
        (
            "missing provider reserve authority",
            lambda preflight: preflight["resources"]["provider_estimate"][
                "resource_estimator_assumptions"
            ].pop("frozen_reserve_bytes"),
            "unchanged VRAM reserve|canonical authority",
        ),
        (
            "missing CUDA admission evidence",
            lambda preflight: preflight["resources"]["cuda"].pop(
                "admission_required"
            ),
            "admission|free|bound|evidence",
        ),
        (
            "missing CUDA free evidence",
            lambda preflight: preflight["resources"]["cuda"].pop("free_bytes"),
            "preflight CUDA free_bytes|integer",
        ),
        (
            "missing CUDA bound evidence",
            lambda preflight: preflight["resources"]["cuda"].pop(
                "live_admission_bound_bytes"
            ),
            "preflight CUDA live_admission_bound_bytes|integer",
        ),
        (
            "invalid CUDA bound coverage",
            lambda preflight: preflight["resources"]["cuda"].__setitem__(
                "free_covers_admission_bound_and_headroom", False
            ),
            "admission|free|bound|evidence",
        ),
        (
            "invalid CUDA free/bound arithmetic",
            lambda preflight: preflight["resources"]["cuda"].__setitem__(
                "free_bytes", 5_441_799_168
            ),
            "free/admission bound arithmetic",
        ),
        (
            "mismatching optional current CUDA threshold",
            lambda preflight: preflight["resources"]["cuda"].__setitem__(
                "minimum_free_gpu_bytes", 1
            ),
            "minimum free VRAM threshold authorities disagree",
        ),
        (
            "mismatching legacy CUDA threshold",
            lambda preflight: preflight.__setitem__(
                "cuda", {"minimum_free_gpu_bytes": 1}
            ),
            "minimum free VRAM threshold authorities disagree",
        ),
        (
            "missing canonical CUDA authority",
            lambda preflight: preflight["resources"].pop("cuda"),
            "minimum free VRAM threshold|canonical authority",
        ),
    ],
)
def test_physical_thresholds_fail_closed_for_preflight_tampering(
    tmp_path: Path, label: str, mutate: Any, pattern: str
) -> None:
    del label
    roots = _make_quadruplet(tmp_path)
    _rewrite_preflight(roots[0], mutate)
    with pytest.raises(Exception, match=pattern):
        _materialize(*roots)


@pytest.mark.parametrize(
    ("label", "mutate", "pattern"),
    [
        (
            "capture release timing",
            lambda telemetry: telemetry["async_checkpoint_overlap"][
                "checkpoint_windows"
            ][0].__setitem__("source_released_perf_ns", 100_000_000_000),
            "timing|10-second|capture",
        ),
        (
            "successor dispatch start timing",
            lambda telemetry: telemetry["async_checkpoint_overlap"][
                "checkpoint_windows"
            ][0].update(
                {
                    "successor_dispatch_started_perf_ns": 100_000_000_000,
                    "successor_last_useful_kernel_completion_observed_host_time_ns": (
                        100_000_000_001
                    ),
                }
            ),
            "timing|10-second|successor",
        ),
        (
            "active duty",
            lambda telemetry: telemetry["post_analysis_summary"][
                "gpu_activity"
            ].__setitem__("active_sample_fraction", 0.5),
            "duty|SM|cadence|idle|dispatch",
        ),
        (
            "active duty out of domain",
            lambda telemetry: telemetry["post_analysis_summary"][
                "gpu_activity"
            ].__setitem__("active_sample_fraction", 1.1),
            "active GPU duty fraction|duty|SM|cadence|idle|dispatch",
        ),
        (
            "mean SM",
            lambda telemetry: telemetry["post_analysis_summary"]["gpu_activity"][
                "gpu_sm_percent"
            ].__setitem__("mean", 10.0),
            "duty|SM|cadence|idle|dispatch",
        ),
        (
            "mean SM out of domain",
            lambda telemetry: telemetry["post_analysis_summary"]["gpu_activity"][
                "gpu_sm_percent"
            ].__setitem__("mean", 100.1),
            "mean SM utilization percent|duty|SM|cadence|idle|dispatch",
        ),
        (
            "sample cadence",
            lambda telemetry: telemetry["post_analysis_summary"]["gpu_activity"][
                "observed_interval_seconds"
            ].__setitem__("max", 3.0),
            "duty|SM|cadence|idle|dispatch",
        ),
        (
            "idle gap",
            lambda telemetry: telemetry["post_analysis_summary"][
                "gpu_activity"
            ].__setitem__("longest_idle_gap_observed_span_seconds", 3.0),
            "duty|SM|cadence|idle|dispatch",
        ),
        (
            "capture timing starts at submission",
            lambda telemetry: [
                window.update(
                    {
                        "capture_completed_perf_ns": 0,
                        "capture_submitted_perf_ns": 1,
                        "source_released_perf_ns": 10_000_000_002,
                    }
                )
                for window in telemetry["async_checkpoint_overlap"][
                    "checkpoint_windows"
                ][:1]
            ],
            "timing|10-second|capture",
        ),
        (
            "capture barrier starts before submission",
            lambda telemetry: telemetry["async_checkpoint_overlap"][
                "checkpoint_windows"
            ][0].update(
                {
                    "capture_started_perf_ns": 0,
                    "capture_completed_perf_ns": 1,
                    "capture_submitted_perf_ns": 20_000_000_000,
                    "source_released_perf_ns": 20_000_000_001,
                    "release_authority_perf_ns": 20_000_000_001,
                }
            ),
            "timing|10-second|capture",
        ),
        (
            "GPU-safe release ordering",
            lambda telemetry: telemetry["async_checkpoint_overlap"][
                "checkpoint_windows"
            ][0].__setitem__("gpu_safe_release_ordered", False),
            "timing|10-second|capture",
        ),
        (
            "post-release launched PID binding",
            lambda telemetry: telemetry["post_analysis_summary"][
                "post_release_target_gpu_residency"
            ].__setitem__("launched_pid", 9999),
            "zero target GPU residency|post-release",
        ),
        (
            "post-release attributed PID binding",
            lambda telemetry: telemetry["post_analysis_summary"][
                "post_release_target_gpu_residency"
            ].__setitem__("attributed_target_gpu_pids", []),
            "zero target GPU residency|post-release",
        ),
        (
            "receipt target PID set binding",
            lambda telemetry: telemetry.__setitem__("target_gpu_pids", [9999]),
            "target PID set|target PID authority",
        ),
        (
            "overlap evidence schema",
            lambda telemetry: telemetry["async_checkpoint_overlap"].__setitem__(
                "schema_version", "eiel_acs_async_overlap_evidence_v4"
            ),
            "exact two-slot owner contract",
        ),
        (
            "post-release same-PID framebuffer residency",
            lambda telemetry: telemetry["post_analysis_summary"][
                "post_release_target_gpu_residency"
            ].update(
                {
                    "resident_sample_count": 1,
                    "max_framebuffer_memory_mib": 4096.0,
                    "resident_rows": [
                        {
                            "pid": 1234,
                            "framebuffer_memory_mib": 4096.0,
                        }
                    ],
                    "passed": False,
                }
            ),
            "zero target GPU residency|post-release",
        ),
    ],
)
def test_guarded_thresholds_fail_closed_for_telemetry_tampering(
    tmp_path: Path, label: str, mutate: Any, pattern: str
) -> None:
    del label
    roots = _make_quadruplet(tmp_path)
    telemetry_path = roots[-1]
    telemetry = _read_json(telemetry_path)
    mutate(telemetry)
    _write_json(telemetry_path, telemetry)
    with pytest.raises(Exception, match=pattern):
        _materialize(*roots)


def test_t_acs_tail_must_remain_separate(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)
    _rewrite_batch_result(
        roots[2],
        lambda result: result["tick_native_performance_timeline_binding"]["timeline"][
            "post_simulation_acs_tail"
        ].__setitem__("excluded_from_gpu_performance_metrics", False),
    )
    telemetry = _read_json(roots[-1])
    telemetry["manifest_sha256"] = canonical_sha256(
        _read_json(roots[2] / "batches" / BATCH_ID / "manifest.json")
    )
    _write_json(roots[-1], telemetry)
    with pytest.raises(Exception, match="timeline|T_ACS_TAIL|tail"):
        _materialize(*roots)


def test_restored_dispatch_clock_is_validated_in_its_own_clock_domain(
    tmp_path: Path,
) -> None:
    roots = _make_quadruplet(tmp_path)

    def offset_restored_timeline(result: dict[str, Any]) -> None:
        timeline = result["tick_native_performance_timeline_binding"]["timeline"]
        timeline["gpu_performance_window"]["start_ns"] += 10_000
        timeline["gpu_performance_window"]["end_ns"] += 10_000
        timeline["dispatch_overhead"]["dispatch_domain_start_ns"] += 10_000
        timeline["dispatch_overhead"]["dispatch_domain_end_ns"] += 10_000

    _rewrite_batch_result(roots[3], offset_restored_timeline)
    result = _materialize(*roots)
    assert result["status"] == "PASS"


def _rewrite_request_payload(root: Path, payload: dict[str, Any]) -> None:
    request_path = root / "execution_request.json"
    request_receipt = _read_json(request_path)
    request_sha = canonical_sha256(payload)
    request_receipt["request"] = {**payload, "request_sha256": request_sha}
    request_receipt["request_sha256"] = request_sha
    _write_json(request_path, request_receipt)
    result_path = root / "execution_result.json"
    result = _read_json(result_path)
    result.pop("result_sha256")
    result["request_sha256"] = request_sha
    _write_json(
        result_path,
        {**result, "result_sha256": canonical_sha256(result)},
    )


def _rewrite_result(root: Path, **changes: Any) -> None:
    result_path = root / "execution_result.json"
    result = _read_json(result_path)
    result.pop("result_sha256")
    result.update(changes)
    _write_json(
        result_path,
        {**result, "result_sha256": canonical_sha256(result)},
    )


def _rewrite_batch_result(root: Path, mutate: Any) -> None:
    path = root / "batches" / BATCH_ID / "manifest.json"
    batch = _read_json(path)
    result = batch["result"]
    mutate(result)
    _write_json(path, batch)


def _rewrite_preflight(root: Path, mutate: Any) -> None:
    path = root / "preflight_receipt.json"
    preflight = _read_json(path)
    mutate(preflight)
    _write_json(path, preflight)
    _rewrite_result(
        root,
        preflight_receipt_sha256=canonical_sha256(preflight),
    )


def test_rejects_non_v2_or_tampered_execution_request_receipt(
    tmp_path: Path,
) -> None:
    roots = _make_quadruplet(tmp_path)
    sync_root = roots[0]
    request_path = sync_root / "execution_request.json"
    request_receipt = _read_json(request_path)
    request_receipt["schema_version"] = "eiel_execution_request_v1"
    _write_json(request_path, request_receipt)
    with pytest.raises(Exception, match="request|binding"):
        _materialize(*roots)

    roots = _make_quadruplet(tmp_path / "tampered")
    sync_root = roots[0]
    request_path = sync_root / "execution_request.json"
    request_receipt = _read_json(request_path)
    request_receipt["request"]["purpose"] = "validation"
    _write_json(request_path, request_receipt)
    with pytest.raises(Exception, match="request|binding"):
        _materialize(*roots)


def test_rejects_non_v2_execution_result(tmp_path: Path) -> None:
    roots = _make_quadruplet(tmp_path)
    _rewrite_result(roots[0], schema_version="eiel_execution_result_v1")

    with pytest.raises(Exception, match="verified success"):
        _materialize(*roots)


@pytest.mark.parametrize(
    ("field", "foreign_value"),
    [
        ("invocation_mode", "validate"),
        ("purpose", "qualification"),
        ("campaign_id", "foreign-campaign"),
        ("compiled_plan_sha256", "9" * 64),
        ("plan_sha256", "8" * 64),
        ("profile_id", "foreign-profile"),
        ("source_identity", {"foreign": True}),
        ("run_root", "/foreign/canonical/run-root"),
        ("request_sha256", "7" * 64),
    ],
)
def test_rejects_hash_valid_foreign_execution_result_identity(
    tmp_path: Path,
    field: str,
    foreign_value: Any,
) -> None:
    roots = _make_quadruplet(tmp_path)
    _rewrite_result(roots[0], **{field: foreign_value})

    with pytest.raises(Exception, match="plan|binding"):
        _materialize(*roots)


@pytest.mark.parametrize("target_index", [0, 2])
def test_rejects_wrong_baseline_deviation_contract(
    tmp_path: Path, target_index: int
) -> None:
    roots = _make_quadruplet(tmp_path)
    target = roots[target_index]
    request_receipt = _read_json(target / "execution_request.json")
    payload = dict(request_receipt["request"])
    payload.pop("request_sha256")
    baseline = dict(payload["baseline"])
    baseline["deviations"] = (
        []
        if target_index == 0
        else [
            {
                "requirement_id": "checkpoint.acs_two_slot",
                "reason": "invalid ACS2 deviation",
            }
        ]
    )
    payload["baseline"] = baseline
    _rewrite_request_payload(target, payload)

    with pytest.raises(Exception, match="Baseline|deviation|conformant"):
        _materialize(*roots)


@pytest.mark.parametrize(
    ("name", "kwargs", "pattern"),
    [
        (
            "cell identity mismatch",
            {"async_restored_kwargs": {"cell_ids": (*CELL_IDS[:-1], "cell-bad")}},
            "cell|logical execution contracts",
        ),
        (
            "async route mismatch vs restored",
            {"async_restored_kwargs": {"route": "3" * 64}},
            "route",
        ),
        (
            "sync/async terminal digest mismatch",
            {"async_kwargs": {"terminal_digest": "d" * 64}},
            "terminal",
        ),
        (
            "ACS0 fresh-restore terminal digest mismatch",
            {"sync_restored_kwargs": {"terminal_digest": "d" * 64}},
            "terminal",
        ),
        (
            "event digest mismatch",
            {"async_kwargs": {"event_digest": "e" * 64}},
            "event",
        ),
        (
            "tick-3359 checkpoint mismatch",
            {"async_restored_kwargs": {"checkpoint_digest": "f" * 64}},
            "checkpoint",
        ),
        (
            "same producer/restorer process identity",
            {"async_restored_kwargs": {"same_process": True}},
            "process",
        ),
        (
            "async high-water below two",
            {"async_kwargs": {"high_water": 1}},
            "high.water|occupancy|slot",
        ),
        (
            "async high-water above two",
            {"async_kwargs": {"high_water": 3}},
            "high.water|occupancy|slot",
        ),
        (
            "restored async high-water above two",
            {"async_restored_kwargs": {"high_water": 3}},
            "high.water|occupancy|slot",
        ),
        (
            "restored async high-water has wrong type",
            {"async_restored_kwargs": {"high_water": "2"}},
            "high.water|occupancy|slot",
        ),
        (
            "loaded checkpoint projection mismatch",
            {"async_restored_kwargs": {"loaded_digest_mismatch": True}},
            "loaded projection",
        ),
        (
            "malformed restore process identity",
            {"async_restored_kwargs": {"malformed_process": True}},
            "process identity",
        ),
        (
            "restore process identity has an extra field",
            {"async_restored_kwargs": {"extra_process_field": True}},
            "process identity",
        ),
    ],
)
def test_fail_closed_for_invalid_b136_acs_parity(
    tmp_path: Path,
    name: str,
    kwargs: dict[str, Any],
    pattern: str,
) -> None:
    del name
    roots = _make_quadruplet(tmp_path, **kwargs)

    with pytest.raises(Exception, match=pattern):
        _materialize(*roots)
    output_path = roots[-2]
    assert not output_path.exists()
