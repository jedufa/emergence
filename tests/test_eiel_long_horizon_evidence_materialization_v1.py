from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.contracts import ExecutionRouteContractV1
from engine.campaigns.long_horizon_evidence_materialization import (
    LongHorizonEvidenceMaterializationError,
    materialize_long_horizon_evidence_v1,
)
from engine.campaigns.long_horizon_qualification import (
    BOUNDED_ROUTE_QUALIFIED,
    LONG_HORIZON_RUNABILITY_QUALIFIED,
    validate_long_horizon_evidence_v1,
)
from engine.campaigns.qualification import (
    EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
    execution_route_qualification_sidecar_path_v1,
)
from engine.campaigns.route_qualification import (
    RouteQualificationError,
    promote_execution_route_long_horizon_evidence_v1,
)
from engine.campaigns.tick_native_control import TickNativeControlV1
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EVENT_COLUMN_NAMES,
    EVENT_SINK_HEADER,
    EVENT_SINK_MAGIC,
    EVENT_SINK_RECORD,
    EVENT_SINK_ROW_BYTES,
    event_history_digest_from_references_v1,
)
from tools.scripts.eiel_campaign import _frozen_long_horizon_plan, build_parser


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _git(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture_source_repo(parent: Path) -> tuple[Path, Path, str]:
    repository_root = (parent / "source-repo").resolve()
    repository_root.mkdir(parents=True, exist_ok=True)
    spec_path = repository_root / "campaign_spec.yaml"
    if not (repository_root / ".git").exists():
        _git(repository_root, "init", "--quiet")
        _git(repository_root, "config", "user.email", "eiel-fixture@example.invalid")
        _git(repository_root, "config", "user.name", "EIEL fixture")
        spec_path.write_text("schema_version: eiel-fixture-v1\n", encoding="utf-8")
        (repository_root / "tracked_fixture.txt").write_text(
            "tracked fixture\n", encoding="utf-8"
        )
        science_log = (
            repository_root
            / "docs/200_strategy-science-programme/200_current-strategy/"
            "EIEL-DOC-002526--unregistered-eiel-canonical-science-log.md"
        )
        science_log.parent.mkdir(parents=True)
        science_log.write_text("science log\n", encoding="utf-8")
        _git(repository_root, "add", spec_path.name)
        _git(repository_root, "add", "tracked_fixture.txt")
        _git(repository_root, "add", str(science_log.relative_to(repository_root)))
        _git(repository_root, "commit", "--quiet", "-m", "fixture source")
    return repository_root, spec_path, _git(repository_root, "rev-parse", "HEAD")


def _make_run(
    root: Path,
    *,
    route: str,
    restored: bool,
    plan_sha256: str | None = None,
    status_lines: list[str] | None = None,
    baseline_profile_id: str = "EIEL_PRODUCTION_BASELINE_V1",
    baseline_qualification_status: str = "UNQUALIFIED",
) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository_root, spec_path, commit = _fixture_source_repo(root.parent)
    spec_sha256 = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    cells = [f"cell-{index:03d}" for index in range(136)]
    physical = [{"batch_id": "batch-0000", "cell_ids": cells}]
    if status_lines is None:
        status_lines = []
    plan = {
        "campaign_id": (
            "eiel-full-evolution-v1-qual-b136-5000tick-acs2-restore-w32p256c512-20260824"
            if restored
            else "eiel-full-evolution-v1-qual-b136-5000tick-acs2-uninterrupted-w32p256c512-20260824"
        ),
        "description": "restore twin" if restored else "uninterrupted twin",
        "no_claims": "technical qualification only",
        "resolved_profile": "profile-v1",
        "parameters": {
            "patch.grid_size": 32,
            "patch.initial_population": 256,
            "patch.capacity": 512,
        },
        "execution": {
            "execution_route_identity": "tick_native_execution_route_v1",
            "start_next_tick": 0,
            "total_ticks": 5000,
            "checkpoint_ticks": [2500, 5000],
            "acs_checkpoint_slots": 2,
            "acs_capture_instrumentation": True,
            "native_streamed_checkpoint": True,
            "checkpoint_storage_mode": "disk",
            "time_authority": "absolute_tick_v1",
            "batch": {"mode": "explicit", "cells": 136},
            "tick_native_control": {
                "total_ticks": 5000,
                "checkpoint_ticks": [2500, 5000],
                "resident_owner_span_ticks": 1120,
                "native_dispatch_window_ticks": 1120,
                "source_window_ticks": 160,
                "event_window_capacity": 41804,
                "time_authority": "absolute_tick_v1",
            },
            "budgets": {
                "vram_bytes": 10_000_000,
                "host_bytes": 10_000_000,
                "minimum_free_gpu_bytes": 100,
                "maximum_gpu_memory_slope_bytes_per_window": 10,
                "maximum_host_memory_slope_bytes_per_window": 10,
                "maximum_event_ring_occupancy_slope_per_window": 10,
                "disk_bytes": 1_000_000,
                "wall_seconds": 100,
            },
            **(
                {"deliberate_restore_tick": 2500, "deliberate_restore_ticks": [2500]}
                if restored
                else {"deliberate_restore_tick": None, "deliberate_restore_ticks": []}
            ),
        },
        "output": {"runtime_root": str(root)},
        "artifact_lifecycle": {},
        "artifacts": {"cuda_extension": {"sha256": "e" * 64}},
        "cells": [
            {
                "cell_id": cell_id,
                "parameters": {
                    "patch.grid_size": 32,
                    "patch.initial_population": 256,
                    "patch.capacity": 512,
                    "cuda_architecture_b.cpu_request_staging_enable": False,
                    "tw_rngv2.semantics_version": 2,
                },
            }
            for cell_id in cells
        ],
        "physical_batches": physical,
        "resource_estimate": {
            "compiled_batch_count": 1,
            "compiled_logical_cells": 136,
            "compiled_partition": False,
            "compiled_physical_batch_cells_max": 136,
            "compiled_physical_batch_cells_min": 136,
            "compiled_physical_batch_cells_total": 136,
            "physical_batch_cells": 136,
            "requested_total_ticks": 5000,
            "resident_event_capacity": 41804,
        },
    }
    lifecycle = {
        "schema_version": "eiel_run_artifact_lifecycle_contract_v1",
        "lifecycle_profile": "QUALIFICATION",
        "storage": {
            "total_max_bytes": 1_000_000,
            "scratch_max_bytes": 300_000,
            "checkpoint_max_bytes": 300_000,
            "durable_max_bytes": 350_000,
            "emergency_reserve_bytes": 50_000,
        },
        "observability": {
            "output_profile": "compact_default",
            "raw_tick_trace_policy": "forbidden",
            "debug_event_trace_policy": "forbidden",
            "ring_buffer": {"enabled": False, "max_bytes": 1},
        },
        "checkpoints": {
            "checkpoint_ticks": [2500, 5000],
            "authority": "absolute_tick",
            "roles": [
                {
                    "identity": "checkpoint-tick-000000002500",
                    "checkpoint_tick": 2500,
                    "role": "intermediate",
                    "retention_class": "R1",
                },
                {
                    "identity": "checkpoint-tick-000000005000",
                    "checkpoint_tick": 5000,
                    "role": "qualification_required",
                    "retention_class": "R4",
                },
            ],
        },
        "closeout": {
            "required_types": [
                "run_passport",
                "artifact_disposition_manifest",
                "storage_disposition_receipt",
            ],
            "compact_replacement_verification_required": False,
        },
        "retention": {
            "default_class": "R4",
            "decision_evidence_class": None,
            "audit_hold_binding": "not_required",
            "qualification_hold_binding": "required",
            "unknown_artifact_policy": "RX_BLOCKS_AUTO_DELETE",
        },
        "prune": {
            "automatic_prune_allowed": True,
            "required_predicates": [
                "valid_disposition_manifest",
                "unknown_artifacts_resolved",
                "no_live_references",
                "audit_hold_released_or_not_required",
                "qualification_hold_released_or_not_required",
                "replacement_verified_or_not_required",
                "explicit_checkpoint_roles_resolved",
                "storage_disposition_receipt",
            ],
        },
    }
    lifecycle["contract_sha256"] = canonical_sha256(lifecycle)
    plan["artifact_lifecycle"] = lifecycle
    plan["execution"]["tick_native_control"] = TickNativeControlV1.from_execution(
        plan["execution"]
    ).payload()
    computed_plan_sha256 = canonical_sha256(plan)
    plan["plan_sha256"] = computed_plan_sha256
    _write(root / "execution_plan.json", plan)
    _write(root / "artifact_lifecycle_contract.json", lifecycle)
    source_identity = {
        "bound_plan_sha256": computed_plan_sha256,
        "compiled_plan_sha256": canonical_sha256(
            {"compiled_plan_sha256": computed_plan_sha256}
        ),
        "campaign_spec_sha256": spec_sha256,
        "campaign_spec_path": str(spec_path),
        "git": {
            "commit": commit,
            "repository_root": str(repository_root),
            "working_tree_clean": not status_lines,
            "working_tree_status_lines": status_lines,
            "working_tree_status_sha256": canonical_sha256(status_lines),
        },
    }
    request_payload = {
        "schema_version": "eiel_execution_request_v2",
        "purpose": "qualification",
        "campaign_spec": str(spec_path),
        "output_root": str(root),
        "resource_wait": {
            "enabled": True,
            "max_wait_seconds": None,
            "poll_seconds": 30,
        },
        "qualification_profile": None,
        "baseline": {
            "deviations": [],
            "intent": "qualification",
            "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
            "schema_version": "eiel_baseline_declaration_v1",
        },
    }
    if restored:
        request_payload["restore_process_isolation"] = "fresh_invocation_per_restore"
    request_payload["request_sha256"] = canonical_sha256(request_payload)
    _write(
        root / "execution_request.json",
        {
            "bound_plan_sha256": computed_plan_sha256,
            "created_at_utc": "2026-08-24T00:00:00Z",
            "request": request_payload,
            "request_sha256": request_payload["request_sha256"],
            "run_root": str(root),
            "schema_version": "eiel_execution_request_receipt_v2",
            "source_identity": source_identity,
        },
    )
    terminal_mechanics = {
        cell_id: canonical_sha256({"cell_id": cell_id, "canonical": "mechanics"})
        for cell_id in cells
    }
    terminal_storage = {
        cell_id: canonical_sha256({"cell_id": cell_id, "canonical": "storage"})
        for cell_id in cells
    }
    checkpoint_mechanics = {
        cell_id: canonical_sha256(
            {"cell_id": cell_id, "canonical": "checkpoint-mechanics"}
        )
        for cell_id in cells
    }
    checkpoint_storage = {
        cell_id: canonical_sha256(
            {"cell_id": cell_id, "canonical": "checkpoint-storage"}
        )
        for cell_id in cells
    }

    def event_payload(index: int) -> bytes:
        return bytes([index % 251]) * (10 * EVENT_SINK_ROW_BYTES)

    def event_prefix_digest(index: int) -> str:
        digest = hashlib.sha256()
        digest.update(b"eiel_event_sink_cell_prefix_v1\0")
        digest.update(struct.pack("<QQ", index, 10))
        digest.update(struct.pack("<Q", 10))
        digest.update(event_payload(index))
        return digest.hexdigest()

    frontier = [
        {
            "cell_index": index,
            "path": str(
                root
                / "batches"
                / "batch-0000"
                / "event_history_v1"
                / f"events-{index:03d}.bin"
            ),
            "prefix_identity_schema": "eiel_event_sink_cell_prefix_v1",
            "prefix_sha256": event_prefix_digest(index),
            "row_bytes": EVENT_SINK_ROW_BYTES,
            "row_count": 10,
            "schema_version": "eiel_resident_event_sink_v1",
        }
        for index, cell_id in enumerate(cells)
    ]
    for index, row in enumerate(frontier):
        event_path = Path(row["path"])
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.write_bytes(
            EVENT_SINK_HEADER.pack(
                EVENT_SINK_MAGIC, 1, 136, len(EVENT_COLUMN_NAMES), EVENT_SINK_ROW_BYTES
            )
            + EVENT_SINK_RECORD.pack(index, 0, 10)
            + event_payload(index)
        )
    event_digest = event_history_digest_from_references_v1(frontier)
    sidecar_path = root.parent / "qualification-route.json"
    if not sidecar_path.exists():
        sidecar_path.write_text('{"qualification":"fresh-q6"}\n', encoding="utf-8")
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    required_obligations = [
        "acs_capture_barrier_measurement",
        "alternate_batch_partition",
        "async_checkpoint_backpressure",
        "async_checkpoint_failure_latch",
        "checkpoint_export",
        "cold_start",
        "disk_checkpoint_reload",
        "durable_frontier_publication",
        "exact_absolute_tick_checkpoint_export",
        "first_committed_tick_after_partial_restore",
        "first_post_boundary_tick",
        "first_post_restore_refill",
        "first_post_restore_tick",
        "first_tick",
        "fresh_owner_partial_tick_restore",
        "fresh_owner_restore",
        "multicell_execution",
        "non_lattice_divisible_checkpoint_tick",
        "non_lattice_divisible_terminal_tick",
        "odd_physical_tail",
        "ordinary_birth",
        "reporting_parity",
        "second_post_boundary_tick",
        "terminal_settlement",
    ]
    native = []
    spans = (
        [(2500, 3360), (3360, 4480), (4480, 5000)]
        if restored
        else [
            (0, 1120),
            (1120, 2240),
            (2240, 2500),
            (2500, 3360),
            (3360, 4480),
            (4480, 5000),
        ]
    )
    for index, (start, end) in enumerate(spans):
        owner_start = (start // 1120) * 1120
        owner_end = min(owner_start + 1120, 5000)
        owner_identity = canonical_sha256({"owner_start": owner_start})
        row = {
            "time_authority": "absolute_tick_v1",
            "start_next_tick": start,
            "end_next_tick": end,
            "execution_device": "cuda",
            "native_owner": "native_extension",
            "owner_identity": owner_identity,
            "owner_span_start_tick": owner_start,
            "owner_span_end_tick": owner_end,
            "explicit_native_no_fallback": True,
            "hidden_fallback": False,
            "runtime_jit": False,
            "native_host_gaussian_segment_refill_used": False,
            "gaussian_segment_refill_calls_inside_tick_loop": 0,
            "gaussian_segment_refill_receipts": [],
            "intermediate_device_to_host_operations": 0,
            "intermediate_host_to_device_operations": 0,
            "gaussian_segment_device_to_host_operations_lower_bound": 0,
            "gaussian_segment_host_to_device_operations_lower_bound": 0,
            "compact_settlement": {
                "schema_version": "eiel_compact_segment_settlement_v1",
                "start_next_tick": start,
                "end_next_tick": end,
                "owner_identity": owner_identity,
            },
            "resource_window_measurement": {
                "start_tick": start,
                "end_tick": end,
                "peak_gpu_allocated_bytes": 80,
                "peak_gpu_reserved_bytes": 100,
                "minimum_free_gpu_bytes": 500,
                "host_peak_rss_bytes": 200,
                "host_terminal_rss_bytes": 180,
                "resident_event_bytes": 801_633_504,
                "resident_source_bytes": 128,
                "event_ring_capacity": 5_685_344,
                "event_ring_peak_occupancy": 0,
                "events_emitted": 1360 if index == len(spans) - 1 else 0,
                "events_drained": 1360 if index == len(spans) - 1 else 0,
                "terminal_event_ring_occupancy": 0,
                "external_event_history_count": 1360 if index == len(spans) - 1 else 0,
            },
        }
        if index == len(spans) - 1:
            row["external_event_history_digest"] = event_digest
            row["event_history_prefix_references_by_cell"] = frontier
        native.append(row)

    def checkpoint_entry(
        tick: int, mechanics: dict[str, str], storage: dict[str, str]
    ) -> dict:
        source_projection = {
            cell_id: canonical_sha256({"cell_id": cell_id, "source": "q6"})
            for cell_id in cells
        }
        parent_projection = checkpoint_mechanics if tick == 5000 else None
        identity = {
            "schema_version": "eiel_provider_checkpoint_identity_contract_v1",
            "source_import_projection_digest_sha256s": source_projection,
            "parent_checkpoint_projection_digest_sha256s": parent_projection,
            "terminal_projection_digest_sha256s": mechanics,
            "canonical_terminal_state_digest_sha256s": mechanics,
            "canonical_checkpoint_storage_digest_sha256s": storage,
            "route_envelope_or_execution_provenance_digest_sha256": canonical_sha256(
                {"route": route, "tick": tick}
            ),
            "q5_terminal_equality_authority": "canonical_terminal_state_digest_sha256s",
            "route_provenance_may_differ_from_terminal_state": True,
            "legacy_import_digest_mapping": (
                "legacy_import_digest_maps_only_to_source_import_projection_digest_sha256"
            ),
            "time_authority": "absolute_tick_v1",
            "absolute_tick": tick,
            "next_absolute_tick": tick,
        }
        return {
            "schema_version": "eiel_production_engine_checkpoint_set_v1",
            "time_authority": "absolute_tick_v1",
            "absolute_tick": tick,
            "next_absolute_tick": tick,
            "execution_route_sha256": route,
            "source_import_projection_digest_sha256s": source_projection,
            "parent_checkpoint_projection_digest_sha256s": parent_projection,
            "terminal_projection_digest_sha256s": mechanics,
            "canonical_terminal_state_digest_sha256s": mechanics,
            "canonical_checkpoint_storage_digest_sha256s": storage,
            "mechanics_checkpoint_digests": mechanics,
            "canonical_checkpoint_storage_digests": storage,
            "cell_bundles": [
                {
                    "cell_id": cells[0],
                    "path": str(checkpoint_paths[tick]),
                }
            ],
            "checkpoint_identity_contract": identity,
            "checkpoint_identity_sha256": canonical_sha256(identity),
        }

    checkpoint_paths = {
        tick: root
        / "batches"
        / "batch-0000"
        / "checkpoint_native_aggregate_v1"
        / f"tick-{tick:012d}.acsaggregate"
        for tick in (2500, 5000)
    }
    for tick, checkpoint_path in checkpoint_paths.items():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(f"checkpoint-{tick}".encode("ascii"))

    checkpoint_catalog = {
        "2500": checkpoint_entry(2500, checkpoint_mechanics, checkpoint_storage),
        "5000": checkpoint_entry(5000, terminal_mechanics, terminal_storage),
    }
    checkpoint_inventory = []
    checkpoint_roles = {
        2500: ("intermediate", "R1", "checkpoint-tick-000000002500"),
        5000: (
            "qualification_required",
            "R4",
            "checkpoint-tick-000000005000",
        ),
    }
    for tick, checkpoint_path in checkpoint_paths.items():
        stat = checkpoint_path.stat()
        role, retention_class, role_identity = checkpoint_roles[tick]
        checkpoint_inventory.append(
            {
                "artifact_id": f"checkpoint-{tick}",
                "path": str(checkpoint_path.relative_to(root)),
                "artifact_family": "checkpoint.bundle_v2",
                "artifact_type": "file",
                "owner_writer": "provider-v1",
                "bytes": int(stat.st_size),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "retention_class": retention_class,
                "live_reference_status": "none",
                "audit_hold_status": "not_required",
                "qualification_hold_status": "active",
                "checkpoint_role": role,
                "checkpoint_identity": role_identity,
                "producer_checkpoint_identity_sha256": checkpoint_catalog[
                    str(tick)
                ]["checkpoint_identity_sha256"],
                "checkpoint_run_association": {
                    "campaign_id": plan["campaign_id"],
                    "batch_id": "batch-0000",
                    "checkpoint_tick": tick,
                },
                "producer_sealed_stat": {
                    "device": int(stat.st_dev),
                    "inode": int(stat.st_ino),
                    "bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "ctime_ns": int(stat.st_ctime_ns),
                },
                "exact_object_identity": True,
            }
        )
    owner_boundaries = [3360, 4480] if restored else [1120, 2240, 3360, 4480]
    owner_transfers = [
        {
            "schema_version": "eiel_resident_owner_epoch_transfer_v1",
            "boundary_next_absolute_tick": boundary,
            "prior_owner_identity": canonical_sha256({"owner_start": boundary - 1120}),
            "successor_owner_identity": canonical_sha256({"owner_start": boundary}),
            "successor_span_end_tick": min(boundary + 1120, 5000),
            "simultaneous_physical_owner_count": 1,
            "cuda_allocation_recreated": False,
            "physical_owner_reused": True,
        }
        for boundary in owner_boundaries
    ]
    provider_result = {
        "native_execution_receipts": native,
        "expected_resident_owner_transition_ticks": (
            [3360, 4480] if restored else [1120, 2240, 3360, 4480]
        ),
        "observed_resident_owner_transition_ticks": (
            [3360, 4480] if restored else [1120, 2240, 3360, 4480]
        ),
        "resident_owner_live_count_terminal": 0,
        "resident_owner_max_simultaneous": 1,
        "resident_owner_epoch_transfer_receipts": owner_transfers,
        "resident_owner_release_receipts": [
            {
                "schema_version": "eiel_owner_release_receipt_v1",
                "final_next_absolute_tick": 5000,
                "owner_identity": canonical_sha256({"owner_start": 4480}),
                "live_alias_count": 0,
                "release_synchronization_complete": True,
            }
        ],
        "terminal_next_absolute_tick": 5000,
        "time_authority": "absolute_tick_v1",
        "lifecycle_qacct_receipts": [],
        "checkpoint_catalog": checkpoint_catalog,
    }
    if restored:
        provider_result["fresh_owner_restore_receipt"] = {
            "time_authority": "absolute_tick_v1",
            "source_checkpoint_absolute_tick": 2500,
            "source_checkpoint_next_absolute_tick": 2500,
            "restore_next_tick": 2500,
            "first_executed_absolute_tick": 2500,
            "first_committed_next_tick": 2501,
            "off_by_one_check": "PASS",
            "source_checkpoint_cell_digests": dict(checkpoint_mechanics),
            "reload_receipts": [
                {
                    "cell_id": cell_id,
                    "source_projection_sha256": checkpoint_mechanics[cell_id],
                    "loaded_projection_sha256": checkpoint_mechanics[cell_id],
                }
                for cell_id in cells
            ],
        }
    manifest_path = root / "batches" / "batch-0000" / "manifest.json"
    _write(
        manifest_path,
        {
            "batch_id": "batch-0000",
            "cell_ids": cells,
            "status": "completed",
            "plan_sha256": computed_plan_sha256,
            "execution_route_sha256": route,
            "artifact_inventory": checkpoint_inventory,
            "checkpoint": {
                **checkpoint_catalog["5000"],
                "terminal_cell_digests": {
                    cell_id: canonical_sha256(
                        {
                            "cell_id": cell_id,
                            "projection": "restored" if restored else "uninterrupted",
                        }
                    )
                    for cell_id in cells
                },
            },
            "result": provider_result,
        },
    )
    summaries = [
        {
            "batch_id": "batch-0000",
            "status": "completed",
            "manifest_path": str(manifest_path),
        }
    ]
    campaign_manifest = root / "campaign_manifest.json"
    _write(
        campaign_manifest,
        {
            "status": "completed",
            "plan_sha256": computed_plan_sha256,
            "provider_id": "provider-v1",
            "profile_id": "profile-v1",
            "execution_route_sha256": route,
            "batches": summaries,
        },
    )
    disposition_path = root / "artifact_disposition_manifest.json"
    disposition_receipt_path = root / "storage_disposition_receipt.json"
    passport = {
        "schema_version": "eiel_run_passport_v1",
        "plan_sha256": computed_plan_sha256,
        "terminal_status": "completed",
        "lifecycle_profile": "QUALIFICATION",
        "lifecycle_contract_sha256": lifecycle["contract_sha256"],
        "disposition_manifest_path": str(disposition_path),
        "disposition_receipt_path": str(disposition_receipt_path),
        "provider_identity": {
            "execution_route_sha256": route,
            "profile_id": "profile-v1",
            "provider_id": "provider-v1",
        },
    }
    passport["passport_sha256"] = canonical_sha256(passport)
    passport_path = root / "run_passport.json"
    _write(passport_path, passport)
    disposition = {
        "schema_version": "eiel_artifact_disposition_manifest_v1",
        "plan_sha256": computed_plan_sha256,
        "lifecycle_contract_sha256": lifecycle["contract_sha256"],
        "lifecycle_profile": "QUALIFICATION",
        "terminal_status": "completed",
        "runtime_root": str(root),
        "inventory": {"mode": "producer_declared", "complete": True},
        "artifacts": [
            {
                **row,
                "decision_evidence_class": None,
                "replacement_artifact_identity": None,
                "release_predicate": "qualification checkpoint retained",
                "release_eligible": False,
                "disposition": "RETAIN_DURABLE",
                "reason": "qualification checkpoint is held",
                "exact_object_identity": True,
            }
            for row in checkpoint_inventory
        ],
        "retention": {
            "live_reference_status": "none",
            "audit_hold_status": "not_required",
            "qualification_hold_status": "active",
            "unknown_artifact_policy": "RX_BLOCKS_AUTO_DELETE",
        },
    }
    disposition["manifest_sha256"] = canonical_sha256(disposition)
    _write(disposition_path, disposition)
    storage_receipt = {
        "schema_version": "eiel_storage_disposition_receipt_v1",
        "plan_sha256": computed_plan_sha256,
        "lifecycle_contract_sha256": lifecycle["contract_sha256"],
        "manifest_sha256": disposition["manifest_sha256"],
        "result": "dry_run",
    }
    storage_receipt["receipt_sha256"] = canonical_sha256(storage_receipt)
    _write(disposition_receipt_path, storage_receipt)
    execution_receipt = {
        "campaign_manifest_path": str(campaign_manifest),
        "execution_route_sha256": route,
        "run_passport_path": str(passport_path),
    }
    if restored:
        execution_receipt["restore_process_transitions_completed"] = [
            {
                "schema_version": "eiel_execution_restore_process_transition_v1",
                "event": {"batch_id": "batch-0000", "restore_tick": 2500},
                "producer_process": {
                    "schema_version": "eiel_execution_process_identity_v1",
                    "process_instance_id": "1" * 32,
                },
                "restore_process": {
                    "schema_version": "eiel_execution_process_identity_v1",
                    "process_instance_id": "2" * 32,
                },
            }
        ]
    result = {
        "schema_version": "eiel_execution_result_v2",
        "run_root": str(root),
        "status": "completed",
        "result_class": "SUCCESS",
        "ok": True,
        "execution_status": "completed",
        "purpose": "qualification",
        "invocation_mode": "execute",
        "created_at_utc": "2026-08-24T00:00:01Z",
        "request_sha256": request_payload["request_sha256"],
        "plan_sha256": computed_plan_sha256,
        "preflight": {
            "execution_route": {
                "authority_state": "route_qualified",
                "execution_route_sha256": route,
                "qualification_receipt_path": str(sidecar_path),
                "qualification_receipt_sha256": sidecar_sha,
                "covered_obligations": required_obligations,
                "required_obligations": required_obligations,
                "missing_obligations": [],
                "qualification_dimensions": ["BOUNDED_ROUTE_QUALIFIED"],
            }
        },
        "baseline": {
            "admitted": True,
            "configuration_eligible": True,
            "configuration_status": "CONFORMANT",
            "deviations": [],
            "execution_admission_status": "ADMITTED",
            "operational_status": "CONFORMANT",
            "overall_status": "BASELINE_CONFORMANT",
            "profile_id": baseline_profile_id,
            "profile_version": 2,
            "qualification_status": baseline_qualification_status,
        },
        "source_identity": source_identity,
        "execution_receipt": execution_receipt,
    }
    result["result_sha256"] = canonical_sha256(result)
    _write(root / "execution_result.json", result)
    return plan


def _refresh_plan_binding(root: Path, *, mutate_plan=None) -> None:
    plan_path = root / "execution_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if mutate_plan is not None:
        mutate_plan(plan)
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = canonical_sha256(plan)
    _write(plan_path, plan)
    request_path = root / "execution_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["bound_plan_sha256"] = plan["plan_sha256"]
    request["source_identity"]["bound_plan_sha256"] = plan["plan_sha256"]
    _write(request_path, request)
    passport_path = root / "run_passport.json"
    passport = json.loads(passport_path.read_text(encoding="utf-8"))
    passport["plan_sha256"] = plan["plan_sha256"]
    passport.pop("passport_sha256", None)
    passport["passport_sha256"] = canonical_sha256(passport)
    _write(passport_path, passport)
    disposition_path = root / "artifact_disposition_manifest.json"
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["plan_sha256"] = plan["plan_sha256"]
    disposition.pop("manifest_sha256", None)
    disposition["manifest_sha256"] = canonical_sha256(disposition)
    _write(disposition_path, disposition)
    storage_path = root / "storage_disposition_receipt.json"
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    storage["plan_sha256"] = plan["plan_sha256"]
    storage["manifest_sha256"] = disposition["manifest_sha256"]
    storage.pop("receipt_sha256", None)
    storage["receipt_sha256"] = canonical_sha256(storage)
    _write(storage_path, storage)
    result_path = root / "execution_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["plan_sha256"] = plan["plan_sha256"]
    result["source_identity"]["bound_plan_sha256"] = plan["plan_sha256"]
    result.pop("result_sha256", None)
    result["result_sha256"] = canonical_sha256(result)
    _write(result_path, result)


def _rewrite_result(root: Path, mutate_result) -> None:
    path = root / "execution_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    mutate_result(result)
    result.pop("result_sha256", None)
    result["result_sha256"] = canonical_sha256(result)
    _write(path, result)


def _rewrite_request_receipt(root: Path, mutate_request) -> None:
    path = root / "execution_request.json"
    request = json.loads(path.read_text(encoding="utf-8"))
    mutate_request(request)
    _write(path, request)


def test_rejects_mutated_plan_bound_resource_limits(tmp_path: Path) -> None:
    route = "7" * 64
    _make_run(
        tmp_path / "uninterrupted",
        route=route,
        restored=False,
        plan_sha256="a" * 64,
    )
    _make_run(
        tmp_path / "restored",
        route=route,
        restored=True,
        plan_sha256="b" * 64,
    )
    path = tmp_path / "uninterrupted" / "execution_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["execution"]["budgets"]["host_bytes"] = 1_000_000
    _write(path, plan)

    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="persisted execution plan SHA-256 differs",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_materializes_and_promotes_paired_long_horizon_evidence(tmp_path: Path) -> None:
    contract = ExecutionRouteContractV1(
        schema_version="eiel_execution_route_contract_v1",
        provider_id="provider-v1",
        profile_id="profile-v1",
        components=(),
        required_obligations=(),
    )
    uninterrupted_plan = _make_run(
        tmp_path / "uninterrupted",
        route=contract.execution_route_sha256,
        restored=False,
        plan_sha256="a" * 64,
        baseline_qualification_status="PARTIAL",
    )
    _make_run(
        tmp_path / "restored",
        route=contract.execution_route_sha256,
        restored=True,
        plan_sha256="b" * 64,
        baseline_qualification_status="PARTIAL",
    )
    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence.json",
    )

    assert evidence["ticks_committed"] == 5000
    assert evidence["owner_boundaries_crossed"] == 4
    assert (
        evidence["evidence_identity"]["route_scope"]
        == "long_horizon_runability_route_v1"
    )
    assert validate_long_horizon_evidence_v1(evidence)["qualification_dimensions"] == [
        LONG_HORIZON_RUNABILITY_QUALIFIED
    ]
    assert evidence["checkpoint_restore"]["fresh_owner_restore_tick"] == 2500
    assert (tmp_path / "evidence.json").is_file()
    bindings = evidence["paired_run_bindings"]
    assert bindings["uninterrupted"]["run_root"] != bindings["restored"]["run_root"]
    assert (
        bindings["uninterrupted"]["git_status_sha256"]
        == bindings["restored"]["git_status_sha256"]
    )
    assert (
        bindings["uninterrupted"]["qualification_receipt_sha256"]
        == bindings["restored"]["qualification_receipt_sha256"]
    )
    assert (
        bindings["uninterrupted"]["event_frontier_digest"]
        == bindings["restored"]["event_frontier_digest"]
    )
    assert (
        bindings["uninterrupted"]["terminal_storage_digest"]
        == bindings["restored"]["terminal_storage_digest"]
    )
    assert evidence["paired_run_bindings_sha256"] == canonical_sha256(bindings)

    mechanic = tmp_path / "qualification" / "qualification.json"
    sidecar = execution_route_qualification_sidecar_path_v1(
        mechanic, contract.execution_route_sha256
    )
    receipt = {
        "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
        "authority_state": "route_qualified",
        "execution_route_sha256": contract.execution_route_sha256,
        "route_contract_sha256": contract.execution_route_sha256,
        "route_contract": contract.payload(),
        "provider_id": contract.provider_id,
        "profile_id": contract.profile_id,
        "covered_obligations": [],
        "missing_obligations": [],
        "artifact_sha256": "e" * 64,
        "qualification_dimensions": [BOUNDED_ROUTE_QUALIFIED],
    }
    _write(sidecar, receipt)
    source_receipt_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    for run_name in ("uninterrupted", "restored"):
        def bind_promotion_source(result: dict) -> None:
            route_binding = result["preflight"]["execution_route"]
            route_binding["qualification_receipt_path"] = str(sidecar)
            route_binding["qualification_receipt_sha256"] = source_receipt_sha256

        _rewrite_result(tmp_path / run_name, bind_promotion_source)
    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence-bound-to-promotion-source.json",
    )
    campaign_plan = {
        **uninterrupted_plan,
        "qualification_key": {"artifact_hashes": {"cuda_extension": "e" * 64}},
    }
    promoted = promote_execution_route_long_horizon_evidence_v1(
        campaign_plan=campaign_plan,
        route_contract=contract,
        qualification_receipt_path=mechanic,
        long_horizon_evidence=evidence,
    )

    assert promoted["bounded_canaries_rerun"] is False
    assert set(promoted["qualification_dimensions"]) == {
        BOUNDED_ROUTE_QUALIFIED,
        LONG_HORIZON_RUNABILITY_QUALIFIED,
    }
    promoted_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert (
        promoted_sidecar["promotion_source_receipt_sha256"]
        == source_receipt_sha256
    )
    promoted_sidecar.pop("promotion_source_receipt_sha256")
    _write(sidecar, promoted_sidecar)
    repeated = promote_execution_route_long_horizon_evidence_v1(
        campaign_plan=campaign_plan,
        route_contract=contract,
        qualification_receipt_path=mechanic,
        long_horizon_evidence=evidence,
    )
    assert repeated["qualification_succeeded"] is True
    assert (
        json.loads(sidecar.read_text(encoding="utf-8"))[
            "promotion_source_receipt_sha256"
        ]
        == source_receipt_sha256
    )
    generic = json.loads(json.dumps(evidence))
    generic.pop("materialization_profile")
    generic.pop("evidence_sha256")
    generic["evidence_sha256"] = canonical_sha256(generic)
    with pytest.raises(
        RouteQualificationError,
        match="existing long-horizon promotion differs|qualification-pair attestation",
    ):
        promote_execution_route_long_horizon_evidence_v1(
            campaign_plan=campaign_plan,
            route_contract=contract,
            qualification_receipt_path=mechanic,
            long_horizon_evidence=generic,
        )
    forged = json.loads(json.dumps(evidence))
    forged["paired_run_bindings"]["uninterrupted"]["result_sha256"] = "f" * 64
    forged["paired_run_bindings_sha256"] = canonical_sha256(
        forged["paired_run_bindings"]
    )
    forged.pop("evidence_sha256")
    forged["evidence_sha256"] = canonical_sha256(forged)
    with pytest.raises(
        RouteQualificationError,
        match="existing long-horizon promotion differs|canonical revalidation",
    ):
        promote_execution_route_long_horizon_evidence_v1(
            campaign_plan=campaign_plan,
            route_contract=contract,
            qualification_receipt_path=mechanic,
            long_horizon_evidence=forged,
        )


def test_accepts_bound_tracked_science_metadata_state(tmp_path: Path) -> None:
    route = "c" * 64
    status_lines = [
        "M docs/200_strategy-science-programme/200_current-strategy/"
        "EIEL-DOC-002526--unregistered-eiel-canonical-science-log.md"
    ]
    _make_run(
        tmp_path / "uninterrupted",
        route=route,
        restored=False,
        status_lines=status_lines,
        baseline_qualification_status="PARTIAL",
    )
    _make_run(
        tmp_path / "restored",
        route=route,
        restored=True,
        status_lines=status_lines,
        baseline_qualification_status="PARTIAL",
    )
    science_log = (
        tmp_path
        / "source-repo/docs/200_strategy-science-programme/200_current-strategy/"
        "EIEL-DOC-002526--unregistered-eiel-canonical-science-log.md"
    )
    science_log.write_text("updated science log\n", encoding="utf-8")

    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence.json",
    )

    assert evidence["ticks_committed"] == 5000


def test_accepts_detached_checkpoint_settlement_without_compact_row(
    tmp_path: Path,
) -> None:
    route = "d" * 64
    for name, restored in (("uninterrupted", False), ("restored", True)):
        root = tmp_path / name
        _make_run(root, route=route, restored=restored)
        manifest_path = root / "batches/batch-0000/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        terminal = manifest["result"]["native_execution_receipts"][-1]
        terminal.pop("compact_settlement")
        terminal["checkpoint_capture_mode"] = (
            "detached_physical_batch_host_settlement_v1"
        )
        manifest["result"]["checkpoint_catalog"]["5000"][
            "acs_checkpoint_staging"
        ] = {
            "detached_checkpoint_receipt": {
                "detached_authorities": {
                    "private_legacy_adapter": {"next_generation": 31}
                }
            }
        }
        _write(manifest_path, manifest)

    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence.json",
    )

    assert evidence["ticks_committed"] == 5000


def test_rejects_generation_authority_in_public_checkpoint_identity(
    tmp_path: Path,
) -> None:
    route = "e" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    manifest_path = tmp_path / "uninterrupted/batches/batch-0000/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["checkpoint_catalog"]["2500"][
        "checkpoint_identity_contract"
    ]["next_generation"] = 16
    _write(manifest_path, manifest)

    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="active generation authority"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_accepts_distinct_checkpoint_source_projection_domains(
    tmp_path: Path,
) -> None:
    route = "f" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    manifest_path = tmp_path / "restored/batches/batch-0000/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = manifest["result"]["checkpoint_catalog"]["2500"]
    projection = {
        cell_id: canonical_sha256({"cell_id": cell_id, "source": "restore-arm"})
        for cell_id in checkpoint["terminal_projection_digest_sha256s"]
    }
    checkpoint["terminal_projection_digest_sha256s"] = projection
    identity = checkpoint["checkpoint_identity_contract"]
    identity["terminal_projection_digest_sha256s"] = projection
    checkpoint["checkpoint_identity_sha256"] = canonical_sha256(identity)
    for row in manifest["artifact_inventory"]:
        if row["checkpoint_run_association"]["checkpoint_tick"] == 2500:
            row["producer_checkpoint_identity_sha256"] = checkpoint[
                "checkpoint_identity_sha256"
            ]
    restore = manifest["result"]["fresh_owner_restore_receipt"]
    restore["source_checkpoint_cell_digests"] = projection
    for row in restore["reload_receipts"]:
        row["source_projection_sha256"] = projection[row["cell_id"]]
        row["loaded_projection_sha256"] = projection[row["cell_id"]]
    _write(manifest_path, manifest)
    disposition_path = tmp_path / "restored/artifact_disposition_manifest.json"
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    for row in disposition["artifacts"]:
        if row["checkpoint_run_association"]["checkpoint_tick"] == 2500:
            row["producer_checkpoint_identity_sha256"] = checkpoint[
                "checkpoint_identity_sha256"
            ]
    disposition.pop("manifest_sha256")
    disposition["manifest_sha256"] = canonical_sha256(disposition)
    _write(disposition_path, disposition)
    storage_path = tmp_path / "restored/storage_disposition_receipt.json"
    storage = json.loads(storage_path.read_text(encoding="utf-8"))
    storage["manifest_sha256"] = disposition["manifest_sha256"]
    storage.pop("receipt_sha256")
    storage["receipt_sha256"] = canonical_sha256(storage)
    _write(storage_path, storage)

    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence.json",
    )

    paired = evidence["checkpoint_restore"]["paired_checkpoint"]
    assert paired["uninterrupted_identity_digest"] != paired[
        "restored_identity_digest"
    ]


def test_rejects_physical_b2_decomposition(tmp_path: Path) -> None:
    route = "8" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)

    def split_into_b2(plan: dict) -> None:
        cells = [row["cell_id"] for row in plan["cells"]]
        plan["physical_batches"] = [
            {
                "batch_id": f"batch-{index:04d}",
                "cell_ids": cells[index * 2 : index * 2 + 2],
            }
            for index in range(68)
        ]

    _refresh_plan_binding(tmp_path / "uninterrupted", mutate_plan=split_into_b2)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="one physical B136"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_source_status_mismatch(tmp_path: Path) -> None:
    route = "9" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(
        tmp_path / "restored",
        route=route,
        restored=True,
        status_lines=["?? docs/900_lifecycle/940_incoming/other.json"],
    )
    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="exact committed repository/status",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_semantic_untracked_source_state(tmp_path: Path) -> None:
    route = "a" * 64
    _make_run(
        tmp_path / "uninterrupted",
        route=route,
        restored=False,
        status_lines=["?? engine/foreign.py"],
    )
    _make_run(
        tmp_path / "restored",
        route=route,
        restored=True,
        status_lines=["?? engine/foreign.py"],
    )
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="metadata allowlist"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_nonqualification_baseline_and_wrong_event_capacity(
    tmp_path: Path,
) -> None:
    route = "b" * 64
    _make_run(
        tmp_path / "uninterrupted",
        route=route,
        restored=False,
        baseline_profile_id="EIEL_PRODUCTION_BASELINE_V2",
    )
    _make_run(tmp_path / "restored", route=route, restored=True)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="Production Baseline V2"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )

    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "uninterrupted" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["result"]["native_execution_receipts"][0]["resource_window_measurement"][
        "event_ring_capacity"
    ] -= 1
    _write(batch_path, batch)
    with pytest.raises(LongHorizonEvidenceMaterializationError, match="capacity/bytes"):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_canonical_storage_or_lifecycle_mismatch(tmp_path: Path) -> None:
    route = "c" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["checkpoint"]["canonical_checkpoint_storage_digests"]["cell-000"] = "d" * 64
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="checkpoint identity"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_lifecycle_qacct_mismatch(tmp_path: Path) -> None:
    route = "f" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["result"]["lifecycle_qacct_receipts"] = [{"qacct": "mismatch"}]
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="lifecycle/QACCT"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_mutated_lifecycle_closeout_receipt(tmp_path: Path) -> None:
    route = "4" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    receipt_path = tmp_path / "restored" / "storage_disposition_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["result"] = "forged"
    _write(receipt_path, receipt)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="lifecycle closeout artifacts",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_unpaired_tick_2500_checkpoint_authority(tmp_path: Path) -> None:
    route = "d" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["result"]["checkpoint_catalog"]["2500"]["mechanics_checkpoint_digests"][
        "cell-000"
    ] = ("e" * 64)
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="tick-2500 checkpoint"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_obsolete_or_non_absolute_restore_frontier(tmp_path: Path) -> None:
    route = "2" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    receipt = batch["result"]["fresh_owner_restore_receipt"]
    receipt.pop("source_checkpoint_absolute_tick")
    receipt.pop("source_checkpoint_next_absolute_tick")
    receipt["source_checkpoint_next_tick"] = 2500
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="restore frontier"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )

    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["result"]["fresh_owner_restore_receipt"]["time_authority"] = "generation_v1"
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="restore frontier"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_same_process_restore_transition(tmp_path: Path) -> None:
    route = "3" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)

    def reuse_producer_process(result: dict) -> None:
        transition = result["execution_receipt"][
            "restore_process_transitions_completed"
        ][0]
        transition["restore_process"]["process_instance_id"] = transition[
            "producer_process"
        ]["process_instance_id"]

    _rewrite_result(tmp_path / "restored", reuse_producer_process)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="distinct canonical process identities",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_route_sidecar_mismatch(tmp_path: Path) -> None:
    route = "1" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)

    def mutate_route_receipt(result: dict) -> None:
        result["preflight"]["execution_route"]["qualification_receipt_sha256"] = (
            "0" * 64
        )

    _rewrite_result(tmp_path / "restored", mutate_route_receipt)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="qualification sidecar SHA-256"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_accepts_exact_monotonic_long_horizon_sidecar_promotion(
    tmp_path: Path,
) -> None:
    route = "1" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence-before-promotion.json",
    )
    source_receipt_sha256 = evidence["paired_run_bindings"]["uninterrupted"][
        "qualification_receipt_sha256"
    ]
    sidecar = tmp_path / "qualification-route.json"
    _write(
        sidecar,
        {
            "schema_version": EXECUTION_ROUTE_QUALIFICATION_SCHEMA_VERSION,
            "authority_state": "route_qualified",
            "execution_route_sha256": route,
            "qualification_dimensions": [
                BOUNDED_ROUTE_QUALIFIED,
                LONG_HORIZON_RUNABILITY_QUALIFIED,
            ],
            "promotion_source_receipt_sha256": source_receipt_sha256,
            "long_horizon_scope": {"validated_evidence": evidence},
        },
    )

    rematerialized = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence-after-promotion.json",
    )

    assert rematerialized["evidence_sha256"] == evidence["evidence_sha256"]


def test_rejects_unqualified_route_sidecar_binding(tmp_path: Path) -> None:
    route = "e" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)

    def remove_obligation(result: dict) -> None:
        result["preflight"]["execution_route"]["covered_obligations"] = [
            "BOUNDED_ROUTE_QUALIFIED"
        ]

    _rewrite_result(tmp_path / "restored", remove_obligation)
    with pytest.raises(LongHorizonEvidenceMaterializationError, match="24 obligations"):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_unresolvable_git_commit_binding(tmp_path: Path) -> None:
    route = "5" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)

    _rewrite_result(
        tmp_path / "uninterrupted",
        lambda result: result["source_identity"]["git"].__setitem__(
            "commit", "f" * 40
        ),
    )
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="Git commit"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_accepts_reachable_ancestor_git_commit_binding(tmp_path: Path) -> None:
    route = "5" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    result = json.loads(
        (tmp_path / "uninterrupted" / "execution_result.json").read_text(
            encoding="utf-8"
        )
    )
    repository_root = Path(
        result["source_identity"]["git"]["repository_root"]
    )
    (repository_root / "second-commit.txt").write_text(
        "reachable non-head commit\n", encoding="utf-8"
    )
    _git(repository_root, "add", "second-commit.txt")
    _git(repository_root, "commit", "--quiet", "-m", "second fixture source")

    evidence = materialize_long_horizon_evidence_v1(
        uninterrupted_run_root=tmp_path / "uninterrupted",
        restored_run_root=tmp_path / "restored",
        output_path=tmp_path / "evidence.json",
    )

    assert evidence["ticks_committed"] == 5000


def test_rejects_live_tracked_dirty_git_status_mismatch(tmp_path: Path) -> None:
    route = "5" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    result = json.loads(
        (tmp_path / "uninterrupted" / "execution_result.json").read_text(
            encoding="utf-8"
        )
    )
    repository_root = Path(
        result["source_identity"]["git"]["repository_root"]
    )
    tracked_path = repository_root / "tracked_fixture.txt"
    tracked_path.write_text("tracked fixture changed\n", encoding="utf-8")

    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="exact committed repository/status",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_reused_paired_run_root(tmp_path: Path) -> None:
    route = "4" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="two distinct fresh run roots"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "uninterrupted",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_non_git_repository_root_and_spec_outside_worktree(
    tmp_path: Path,
) -> None:
    route = "6" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    non_git_root = tmp_path / "not-a-git-worktree"
    non_git_root.mkdir()
    _rewrite_result(
        tmp_path / "uninterrupted",
        lambda result: result["source_identity"]["git"].__setitem__(
            "repository_root", str(non_git_root)
        ),
    )
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="Git worktree"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )

    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    outside_spec = tmp_path / "outside-spec.yaml"
    outside_spec.write_text("outside: true\n", encoding="utf-8")

    def move_spec(result: dict) -> None:
        result["source_identity"]["campaign_spec_path"] = str(outside_spec)
        result["source_identity"]["campaign_spec_sha256"] = hashlib.sha256(
            outside_spec.read_bytes()
        ).hexdigest()

    _rewrite_result(tmp_path / "uninterrupted", move_spec)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="escapes the committed"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_spec_bytes_that_differ_from_committed_path(tmp_path: Path) -> None:
    route = "7" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    _rewrite_result(
        tmp_path / "uninterrupted",
        lambda result: result["source_identity"].__setitem__(
            "campaign_spec_sha256", "f" * 64
        ),
    )
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="bytes differ from the bound"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_unparseable_or_nonmonotonic_run_timestamps(tmp_path: Path) -> None:
    route = "8" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    _rewrite_request_receipt(
        tmp_path / "uninterrupted",
        lambda request: request.__setitem__("created_at_utc", "not-a-timestamp"),
    )
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="not parseable"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )

    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _rewrite_result(
        tmp_path / "uninterrupted",
        lambda result: result.__setitem__(
            "created_at_utc", "2026-08-23T23:59:59Z"
        ),
    )
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="precedes the execution request"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_fresh_owner_projection_map_tamper(tmp_path: Path) -> None:
    route = "9" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    restore = batch["result"]["fresh_owner_restore_receipt"]
    forged = "f" * 64
    restore["source_checkpoint_cell_digests"]["cell-000"] = forged
    restore["reload_receipts"][0]["source_projection_sha256"] = forged
    restore["reload_receipts"][0]["loaded_projection_sha256"] = forged
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="terminal projection authority",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_tampered_provider_checkpoint_inventory_seal(tmp_path: Path) -> None:
    route = "a" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = (
        tmp_path / "uninterrupted" / "batches" / "batch-0000" / "manifest.json"
    )
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["artifact_inventory"][0]["producer_sealed_stat"]["bytes"] += 1
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="producer-sealed stat"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_false_batch_artifact_inventory_complete_marker(
    tmp_path: Path,
) -> None:
    route = "a" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = (
        tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    )
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["artifact_inventory_complete"] = False
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="not complete producer_declared",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_tampered_self_hashed_lifecycle_checkpoint_row(
    tmp_path: Path,
) -> None:
    route = "e" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    disposition_path = tmp_path / "restored" / "artifact_disposition_manifest.json"
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    disposition["artifacts"][0]["sha256"] = "f" * 64
    _write(disposition_path, disposition)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="lifecycle closeout artifacts"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_missing_checkpoint_inventory_row(tmp_path: Path) -> None:
    route = "b" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = (
        tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    )
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["artifact_inventory"] = [
        row for row in batch["artifact_inventory"] if row["checkpoint_run_association"]["checkpoint_tick"] != 5000
    ]
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="exactly one row at tick 5000"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_terminal_event_occupancy_accounting_tamper(tmp_path: Path) -> None:
    route = "c" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = (
        tmp_path / "uninterrupted" / "batches" / "batch-0000" / "manifest.json"
    )
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    terminal = batch["result"]["native_execution_receipts"][-1]
    terminal["resource_window_measurement"]["terminal_event_ring_occupancy"] = 1
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError, match="event accounting bounds"
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_rejects_external_history_delta_that_is_not_events_drained(
    tmp_path: Path,
) -> None:
    route = "d" * 64
    _make_run(tmp_path / "uninterrupted", route=route, restored=False)
    _make_run(tmp_path / "restored", route=route, restored=True)
    batch_path = tmp_path / "restored" / "batches" / "batch-0000" / "manifest.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["result"]["native_execution_receipts"][1]["resource_window_measurement"][
        "external_event_history_count"
    ] = 1
    _write(batch_path, batch)
    with pytest.raises(
        LongHorizonEvidenceMaterializationError,
        match="external event history does not advance",
    ):
        materialize_long_horizon_evidence_v1(
            uninterrupted_run_root=tmp_path / "uninterrupted",
            restored_run_root=tmp_path / "restored",
            output_path=tmp_path / "evidence.json",
        )


def test_long_horizon_cli_commands_are_parseable() -> None:
    parser = build_parser()
    materialize = parser.parse_args(
        [
            "materialize-long-horizon",
            "/runs/uninterrupted",
            "/runs/restored",
            "--output",
            "/evidence/long.json",
        ]
    )
    promote = parser.parse_args(
        [
            "promote-long-horizon",
            "/spec/campaign.yaml",
            "/evidence/long.json",
            "--frozen-execution-plan",
            "/runs/uninterrupted/execution_plan.json",
        ]
    )
    parity = parser.parse_args(
        [
            "materialize-physical-b136-acs-parity",
            "/runs/sync",
            "/runs/sync-restored",
            "/runs/async",
            "/runs/async-restored",
            "--output",
            "/evidence/physical-b136-acs-parity.json",
            "--acs2-gpu-telemetry-receipt",
            "/evidence/acs2-gpu-telemetry.json",
        ]
    )

    assert materialize.command == "materialize-long-horizon"
    assert promote.command == "promote-long-horizon"
    assert promote.frozen_execution_plan == Path(
        "/runs/uninterrupted/execution_plan.json"
    )
    assert parity.command == "materialize-physical-b136-acs-parity"


def test_frozen_long_horizon_plan_requires_exact_evidence_bound_plan(
    tmp_path: Path,
) -> None:
    run_root = (tmp_path / "uninterrupted").resolve()
    plan_body = {
        "campaign_id": "campaign-v1",
        "output": {"runtime_root": str(run_root)},
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    plan_path = run_root / "execution_plan.json"
    _write(plan_path, plan)
    evidence = {
        "paired_run_bindings": {
            "uninterrupted": {
                "run_root": str(run_root),
                "plan_sha256": plan["plan_sha256"],
            }
        }
    }

    assert _frozen_long_horizon_plan(plan_path, evidence) == plan

    tampered = dict(plan)
    tampered["campaign_id"] = "different"
    _write(plan_path, tampered)
    with pytest.raises(Exception, match="invalid plan_sha256"):
        _frozen_long_horizon_plan(plan_path, evidence)


def test_frozen_long_horizon_plan_rejects_unbound_path(tmp_path: Path) -> None:
    run_root = (tmp_path / "uninterrupted").resolve()
    other_root = (tmp_path / "other").resolve()
    plan_body = {
        "campaign_id": "campaign-v1",
        "output": {"runtime_root": str(other_root)},
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    plan_path = other_root / "execution_plan.json"
    _write(plan_path, plan)
    evidence = {
        "paired_run_bindings": {
            "uninterrupted": {
                "run_root": str(run_root),
                "plan_sha256": plan["plan_sha256"],
            }
        }
    }

    with pytest.raises(Exception, match="not the uninterrupted plan"):
        _frozen_long_horizon_plan(plan_path, evidence)


def test_frozen_long_horizon_plan_rejects_symlink(tmp_path: Path) -> None:
    run_root = (tmp_path / "uninterrupted").resolve()
    plan_body = {
        "campaign_id": "campaign-v1",
        "output": {"runtime_root": str(run_root)},
    }
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    plan_path = run_root / "execution_plan.json"
    _write(plan_path, plan)
    link_path = tmp_path / "linked-plan.json"
    link_path.symlink_to(plan_path)
    evidence = {
        "paired_run_bindings": {
            "uninterrupted": {
                "run_root": str(run_root),
                "plan_sha256": plan["plan_sha256"],
            }
        }
    }

    with pytest.raises(Exception, match="regular file"):
        _frozen_long_horizon_plan(link_path, evidence)
