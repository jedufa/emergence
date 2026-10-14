from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import engine.campaigns.providers as providers_module
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.tick_native_control import TickNativeControlV1
from engine.campaigns.provider_native_chronicle_v1 import (
    ProviderNativeChronicleConsumerV1,
)
from engine.campaigns.providers import (
    _BoundedSourceProjectionStoreV1,
    CampaignProviderError,
    CHECKPOINT_IDENTITY_CONTRACT_SCHEMA_VERSION,
    FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION,
    FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION,
    PRODUCTION_CPU_PROFILE_ID,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
    _provider_native_exposure_receipt_v1,
    _tick_native_performance_timeline_binding_v1,
    _generation_zero_projection_payload_v1,
    _lazy_projection_bundle_sequence_v1,
    _trim_cuda_allocator_after_resident_owner_preparation_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    _provider_native_segment_artifact_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    CheckpointProjectionCursorV1,
    DetachedCellProjectionV1,
    DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
)
from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    append_event_sink_columns_v1,
    event_sink_prefix_identity_v1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CONTINUITY_SCHEMA_VERSION,
)


class _LifecycleHarness(ProductionEngineLifecycleV1):
    """Test-only lifecycle harness; not a registered execution provider."""

    provider_id = "test-production-engine-lifecycle"
    profile_id = "test-production-engine-lifecycle"


def test_tick_native_performance_binding_uses_exact_provider_phase_clocks() -> None:
    binding = _tick_native_performance_timeline_binding_v1(
        launch_started_perf_ns=0,
        launch_completed_perf_ns=1_000,
        native_execution_receipts=(
            {
                "segment_dispatch_started_perf_ns": 100,
                "segment_dispatch_finished_perf_ns": 300,
                "start_next_tick": 0,
                "end_next_tick": 160,
            },
            {
                "segment_dispatch_started_perf_ns": 600,
                "segment_dispatch_finished_perf_ns": 800,
                "start_next_tick": 160,
                "end_next_tick": 320,
            },
        ),
        acs_capture_measurements=(
            {
                "coordinate": 160,
                "segment_dispatch_finished_perf_ns": 300,
                "detached_capture_submitted_perf_ns": 350,
                "detached_writer_started_perf_ns": 350,
                "owner_span_start_tick": 0,
                "owner_span_end_tick": 1_120,
            },
            {
                "coordinate": 320,
                "segment_dispatch_finished_perf_ns": 800,
                "detached_capture_submitted_perf_ns": 850,
                "detached_writer_started_perf_ns": 850,
                "gpu_safely_releasable_perf_ns": 840,
                "terminal_cuda_primary_context_release": {
                    "released_perf_ns": 840,
                },
                "owner_span_start_tick": 0,
                "owner_span_end_tick": 1_120,
            },
        ),
        acs_catalog_publication_measurements=(
            {"coordinate": 160, "durable_acknowledged_perf_ns": 700},
            {"coordinate": 320, "durable_acknowledged_perf_ns": 960},
        ),
        owner_preparation_measurements=(
            {
                "start_perf_ns": 0,
                "end_perf_ns": 100,
                "owner_id": "owner-0",
                "owner_span": {"start_next_tick": 0, "end_next_tick": 1_120},
            },
        ),
        terminal_next_absolute_tick=320,
        acs_close_completed_perf_ns=980,
        restore_receipt=None,
    )

    timeline = binding["timeline"]
    assert timeline["durations_ns"] == {
        "T_SIM": 400,
        "T_DETACH": 90,
        "T_PERSIST": 460,
        "T_ACS_TAIL": 140,
        "T_RESTORE": 0,
        "OWNER_PREPARATION": 100,
    }
    assert timeline["administrative"]["duration_ns"] == 20
    assert timeline["wall_time_reconciliation"]["unclassified_ns"] == 0
    assert timeline["wall_time_reconciliation"]["pass"] is True
    assert timeline["gpu_performance_window"] == {
        "start_ns": 100,
        "end_ns": 840,
        "duration_ns": 740,
        "seconds": pytest.approx(0.00000074),
        "terminal_checkpoint_tick": 320,
        "post_simulation_acs_tail_excluded": True,
        "measurement_scope": (
            "first_simulation_dispatch_through_terminal_gpu_safe_release_v1"
        ),
    }
    assert timeline["dispatch_overhead"] == {
        "definition": "unclassified_simulation_dispatch_domain_fraction_v1",
        "dispatch_domain_start_ns": 100,
        "dispatch_domain_end_ns": 800,
        "dispatch_span_ns": 700,
        "simulation_dispatch_union_ns": 400,
        "classified_non_dispatch_union_ns": 300,
        "capture_detach_union_ns": 50,
        "persistence_union_ns": 350,
        "owner_preparation_union_ns": 0,
        "dispatch_gap_ns": 0,
        "dispatch_domain_ns": 400,
        "acs_tail_excluded": True,
        "fraction": pytest.approx(0.0),
    }
    assert timeline["dispatch_overhead_fraction"] == pytest.approx(0.0)
    assert binding["launch_bound_receipt"]["acs_tail_receipt"] == {
        "start_perf_ns": 840,
        "end_perf_ns": 980,
    }


def test_shared_aggregate_lazy_sources_bind_without_component_materialization(
    tmp_path: Path,
) -> None:
    aggregate_path = tmp_path / "checkpoint.acsaggregate"
    aggregate_path.write_bytes(b"synthetic aggregate descriptor")
    cell_ids = ("cell-0", "cell-1")
    digests = {"cell-0": "a" * 64, "cell-1": "b" * 64}
    lazy_loads: list[tuple[str, str, bool]] = []

    def lazy_loader(
        path: str, *, cell_id: str, verify_aggregate_payload: bool
    ) -> Mapping[str, Any]:
        lazy_loads.append((path, cell_id, verify_aggregate_payload))
        return {"cell_id": cell_id, "source_digest": digests[cell_id]}

    sources = _lazy_projection_bundle_sequence_v1(
        (str(aggregate_path), str(aggregate_path)),
        lazy_loader,
        cell_ids=cell_ids,
        source_projection_digests=digests,
    )

    # Direct sequence access remains cell-indexed and lazy.
    assert lazy_loads == []
    assert sources[1]["cell_id"] == "cell-1"
    assert lazy_loads == [(str(aggregate_path), "cell-1", True)]
    lazy_loads.clear()
    sources = _lazy_projection_bundle_sequence_v1(
        (str(aggregate_path), str(aggregate_path)),
        lazy_loader,
        cell_ids=cell_ids,
        source_projection_digests=digests,
    )

    class AggregateProvider:
        def terminal_projection_digest(self, projection: Mapping[str, Any]) -> str:
            return str(projection["source_digest"])

        def _write_component_checkpoint(
            self, _projection: Mapping[str, Any], _destination: Path
        ) -> Mapping[str, str]:
            raise AssertionError("aggregate restore must not create source_import_v1 copies")

    store = _BoundedSourceProjectionStoreV1(
        AggregateProvider(),
        root=tmp_path,
        cell_ids=cell_ids,
        source_projections=sources,
    )

    # Binding records durable indexed descriptors and exact per-cell digests;
    # it does not decode either cell or create a component source tree.
    assert lazy_loads == []
    assert store.source_projection_digests == digests
    assert not (tmp_path / "source_import_v1").exists()

    cursor = store.cursor_factory()()
    first = cursor.next_cell()
    assert first is not None
    assert first.cell_id == "cell-0"
    assert first.source_projection_sha256 == digests["cell-0"]
    assert lazy_loads == [(str(aggregate_path), "cell-0", True)]
    cursor.import_and_release("cell-0", object())

    second = cursor.next_cell()
    assert second is not None
    assert second.cell_id == "cell-1"
    assert second.source_projection_sha256 == digests["cell-1"]
    assert lazy_loads == [
        (str(aggregate_path), "cell-0", True),
        (str(aggregate_path), "cell-1", False),
    ]
    cursor.import_and_release("cell-1", object())
    receipt = cursor.close()
    assert receipt.covered_cell_ids == cell_ids
    assert receipt.max_live_tensor_bearing_payloads == 1


def _plan() -> dict[str, Any]:
    cells = [
        {
            "cell_id": f"cell-{index}",
            "parameters": {
                "patch.grid_size": 16,
                "patch.initial_population": 64,
                "patch.capacity": 256,
                "patch.founder_placement": "legacy",
            },
            "seeds": {"ecology": 9100 + index},
        }
        for index in range(3)
    ]
    return {
        "campaign_id": "patch-provider-test",
        "plan_sha256": "1" * 64,
        "qualification_key_sha256": "2" * 64,
        "resolved_profile": _LifecycleHarness.profile_id,
        "cells": cells,
        "artifacts": {
            "cuda_extension": {"path": "/qualified/controller.so", "sha256": "3" * 64},
            "qualification_receipt": {
                "path": "/qualified/qualification.json",
                "sha256": "4" * 64,
            },
            "reporting_receipt": {
                "path": "/qualified/reporting.json",
                "sha256": "5" * 64,
            },
        },
    }


def _scientific_plan() -> dict[str, Any]:
    plan = _plan()
    plan["output"] = {"chronicle_profile": "scientific_full"}
    return plan


def _acs_plan(slots: int = 2) -> dict[str, Any]:
    plan = _plan()
    plan["execution"] = {
        "acs_checkpoint_slots": slots,
        "acs_capture_instrumentation": True,
    }
    return plan


def _tick_native_acs_plan(slots: int = 2) -> dict[str, Any]:
    plan = _plan()
    control = TickNativeControlV1(
        total_ticks=161,
        checkpoint_ticks=(32, 64, 161),
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        source_window_ticks=160,
        event_window_capacity=65536,
    ).payload()
    plan["execution"] = {
        "execution_route_identity": "tick_native_execution_route_v1",
        "total_ticks": 161,
        "checkpoint_ticks": [32, 64, 161],
        "checkpoint_storage_mode": "disk",
        "tick_native_control": control,
        "native_streamed_checkpoint": True,
        "acs_checkpoint_slots": slots,
        "acs_capture_instrumentation": True,
    }
    return plan


def _request(
    *, generations: int = 37, milestones: tuple[int, ...] = (14, 37)
) -> ResidentBatchRequestV1:
    return ResidentBatchRequestV1(
        profile_id=_LifecycleHarness.profile_id,
        batch_id="batch-0000",
        cell_ids=("cell-0", "cell-1", "cell-2"),
        generations=generations,
        ticks_per_generation=80,
        checkpoint_milestones=milestones,
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="6" * 64,
    )


def _tick_request(
    *,
    total_ticks: int = 161,
    checkpoints: tuple[int, ...] = (17, 160, 161),
    restore_tick: int | None = 17,
    restore_ticks: tuple[int, ...] = (),
    acs_checkpoint_slots: int = 0,
    acs_capture_instrumentation: bool = False,
    native_streamed_checkpoint: bool = False,
) -> ResidentBatchRequestV1:
    control = TickNativeControlV1(
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoints,
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        source_window_ticks=160,
        event_window_capacity=65536,
    ).payload()
    return ResidentBatchRequestV1(
        profile_id=_LifecycleHarness.profile_id,
        batch_id="batch-0000",
        cell_ids=("cell-0", "cell-1", "cell-2"),
        checkpoint_milestones=(),
        parameter_arrays=(),
        seed_arrays=(),
        execution_route_sha256="6" * 64,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoints,
        deliberate_restore_tick=restore_tick,
        deliberate_restore_ticks=restore_ticks,
        resident_owner_span_ticks=1120,
        native_dispatch_window_ticks=1120,
        source_window_ticks=160,
        event_window_capacity=65536,
        tick_native_control=control,
        acs_checkpoint_slots=acs_checkpoint_slots,
        acs_capture_instrumentation=acs_capture_instrumentation,
        native_streamed_checkpoint=native_streamed_checkpoint,
    )


def _native_segment_artifact(
    *,
    cell_id: str,
    cell_index: int,
    segment_index: int,
    generation: int,
    previous_chain_sha256: str,
) -> dict[str, Any]:
    schema = "inheritance_cross_path_reproduction_exposure_segment_v1"
    cell = {
        "cell_index": cell_index,
        "enabled": True,
        "status": 0,
        "attempt_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "snapshot_count": 1,
        "attempts": [
            {
                "attempt_index": 0,
                "path": cell_index % 2,
                "generation": generation,
                "tick": 4,
                "eligible_parent_counts": [2, 2],
                "eligible_snapshot_refs": [0],
                "selected_parent_ids": [11, 13],
                "success": True,
                "failure_code": 0,
                "commit_ordinal": segment_index,
                "authoritative_commit": {
                    "child_agent_id": 17,
                    "commit_authority_sha256": "c" * 64,
                },
            }
        ],
        "eligible_parent_snapshots": [{"snapshot_index": 0}],
        "selected_parent_counts": [{"agent_id": 11, "count": 1}],
        "parent_contribution_counts": [{"agent_id": 11, "count": 1}],
        "primary_eligibility_opportunity_counts": [{"agent_id": 11, "count": 1}],
        "mate_eligibility_opportunity_counts": [{"agent_id": 13, "count": 1}],
        "cumulative_counts_after_segment": {
            "attempts": segment_index + 1,
            "successes": segment_index + 1,
            "failures": 0,
            "eligible_snapshots": segment_index + 1,
        },
    }
    cell["segment_sha256"] = canonical_sha256(
        {
            "schema_version": schema,
            "segment_index": segment_index,
            "cell": cell,
        }
    )
    cell["previous_chain_sha256"] = previous_chain_sha256
    cell["chain_sha256_after_segment"] = canonical_sha256(
        {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "segment_index": segment_index,
            "previous_chain_sha256": previous_chain_sha256,
            "segment_sha256": cell["segment_sha256"],
        }
    )
    return {
        "schema_version": schema,
        "segment_index": segment_index,
        "capacity_semantics": {"storage_mode": "exact_full"},
        "cell_id": cell_id,
        "cell": cell,
        "segment_output_sha256": canonical_sha256(
            {
                "cell_id": cell_id,
                "segment_index": segment_index,
                "cell_segment_sha256": cell["segment_sha256"],
            }
        ),
    }


def test_native_exposure_receipt_uses_exact_ticks_for_partial_terminal_segment() -> (
    None
):
    artifact, _metadata = _provider_native_segment_artifact_v1(
        cell_id="cell-0",
        cell_index=0,
        source_projection={
            "metadata": {
                "native_next_segment_index": 0,
                "native_chain_sha256": "0" * 64,
            }
        },
        compact_rows=(),
    )
    request = SimpleNamespace(
        batch_id="batch-0",
        profile_id="profile-0",
        execution_route_sha256="a" * 64,
        cell_ids=("cell-0",),
        ticks_per_generation=160,
    )
    receipt = _provider_native_exposure_receipt_v1(
        request=request,
        native_execution_receipts=[
            {
                "start_generation": 31,
                "end_generation": 31,
                "start_next_tick": 4960,
                "end_next_tick": 5000,
            }
        ],
        segment_artifacts=[artifact],
    )

    assert receipt is not None
    assert receipt["current_execution_generation_spans"] == [
        {"start_generation": 31, "end_generation": 31}
    ]
    assert receipt["current_execution_tick_spans"] == [
        {"start_next_tick": 4960, "end_next_tick": 5000}
    ]
    assert receipt["current_execution_scientific_tick_count"] == 40


def test_tick_native_does_not_force_generation_exposure_receipt() -> None:
    artifact, _metadata = _provider_native_segment_artifact_v1(
        cell_id="cell-0",
        cell_index=0,
        source_projection={
            "metadata": {
                "native_next_segment_index": 0,
                "native_chain_sha256": "0" * 64,
            }
        },
        compact_rows=(),
    )
    request = SimpleNamespace(cell_ids=("cell-0",))

    assert (
        _provider_native_exposure_receipt_v1(
            request=request,
            native_execution_receipts=[{"start_next_tick": 0, "end_next_tick": 1}],
            segment_artifacts=[artifact],
            tick_native=True,
        )
        is None
    )


class _Context:
    def __init__(self, root: Path) -> None:
        self.batch_root = root
        self.records: list[dict[str, Any]] = []

    def record_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.records.append(dict(checkpoint))


def _provider_harness(
    store: dict[str, Mapping[str, Any]],
    *,
    source_preparer=None,
    explicit_native_no_fallback: bool = False,
    explicit_cpu_native_no_fallback: bool = False,
    native_segment_artifacts: bool = False,
    direct_result_overrides: Mapping[str, Any] | None = None,
    track_owner_release: bool = False,
    bounded_projection_cursor_support: bool = False,
    cursor_tensor_fields: bool = False,
) -> tuple[_LifecycleHarness, dict[str, Any]]:
    state: dict[str, Any] = {
        "artifact_loads": [],
        "runner_ids": [],
        "runner_objects": [],
        "runner_generation_extents": [],
        "runner_source_capacities": [],
        "request_sources": [],
        "dispatch_generations": [],
        "checkpoint_identities": [],
        "checkpoint_observer_continuations": [],
        "checkpoint_loads": [],
        "allocation_generations": [],
        "owner_refs": [],
        "prior_owner_alive_before_prepare": [],
        "projected_batches": [],
        "event_sink_path": None,
    }
    module = object()

    def artifact_loader(*paths: str) -> object:
        state["artifact_loads"].append(paths)
        return SimpleNamespace(module=module)

    def runner_builder(cell: Mapping[str, Any], run_dir: Path, **kwargs: Any) -> Any:
        runner = object()
        state["runner_ids"].append(id(runner))
        state["runner_objects"].append(runner)
        native_control = kwargs.get("tick_native_control")
        if native_control is not None:
            assert isinstance(native_control, Mapping)
            assert "generations" not in kwargs
            assert "ticks_per_generation" not in kwargs
            capacity_windows, source_window_ticks = (
                providers_module._tick_native_runner_source_dimensions_v1(
                    native_control
                )
            )
            source_capacity = {
                "source_capacity_window_count": capacity_windows,
                "source_window_ticks": source_window_ticks,
            }
            state["runner_source_capacities"].append(source_capacity)
            dimension_receipt = {"source_capacity": source_capacity}
        else:
            assert "generations" in kwargs
            assert "ticks_per_generation" in kwargs
            state["runner_generation_extents"].append(kwargs["generations"])
            dimension_receipt = {"generations": kwargs["generations"]}
        return SimpleNamespace(
            runner=runner,
            receipt={
                "cell_id": cell["cell_id"],
                "run_dir": str(run_dir),
                **dimension_receipt,
            },
        )

    def request_builder(_runner: Any, **kwargs: Any) -> Any:
        import torch

        assert kwargs["qualified_cuda_module"] is module
        state["allocation_generations"].append(
            kwargs["agent_id_allocation_generation_count"]
        )
        source = kwargs["source_checkpoint_projection"]
        if source is None:
            source = {
                "metadata": {
                    "generation": 0,
                    "policy_weight_mutation_enable": False,
                    "policy_weight_mutation_mode": "dense",
                    "policy_weight_mutation_rate": 0.0,
                    "policy_weight_mutation_std": 0.0,
                }
            }
        sink_path = kwargs.get("event_history_sink_path")
        sink_cell = kwargs.get("event_history_sink_cell_index")
        if isinstance(sink_path, str):
            state["event_sink_path"] = sink_path
        if (
            isinstance(sink_path, str)
            and type(sink_cell) is int
            and bool(kwargs.get("tick_native_control"))
        ):
            if source["metadata"].get("event_history_storage_mode") != (
                "eiel_resident_event_sink_v1"
            ):
                empty_columns = tuple(
                    torch.empty((0,), dtype=EVENT_COLUMN_DTYPES[name])
                    for name in EVENT_COLUMN_NAMES
                )
                append_event_sink_columns_v1(
                    sink_path,
                    cell_index=sink_cell,
                    columns=empty_columns,
                    sequence_base=0,
                    expected_cells=3,
                )
                source = {
                    **source,
                    "metadata": {
                        **source["metadata"],
                        "event_history_storage_mode": ("eiel_resident_event_sink_v1"),
                        "event_history_reference": event_sink_prefix_identity_v1(
                            sink_path,
                            cell_index=sink_cell,
                            expected_rows=0,
                        ),
                        "event_history_frontier": 0,
                    },
                }
        state["request_sources"].append(source)
        return SimpleNamespace(
            import_projection=source,
            cuda_capability={"qualified": True},
            cuda_device="cpu" if explicit_cpu_native_no_fallback else "cuda:0",
            tick_native_control=kwargs.get("tick_native_control"),
        )

    def preparer(**kwargs: Any) -> Any:
        requests = tuple(kwargs["per_cell_requests"])
        resumes = kwargs["resume_projections"]
        if resumes is not None:
            assert all(
                request.import_projection is projection
                for request, projection in zip(requests, resumes, strict=True)
            )
        physical_window_arguments: Any = (object(),)
        if track_owner_release:
            gc.collect()
            state["prior_owner_alive_before_prepare"].append(
                [reference() is not None for reference in state["owner_refs"]]
            )

            class TrackedOwner:
                pass

            physical_window_arguments = TrackedOwner()
            state["owner_refs"].append(weakref.ref(physical_window_arguments))
        elif cursor_tensor_fields:
            class NativeCaptureOwner:
                checkpoint_projection_inputs: dict[str, Any]

            physical_window_arguments = NativeCaptureOwner()
            physical_window_arguments.checkpoint_projection_inputs = {}
        return SimpleNamespace(
            physical_window_arguments=physical_window_arguments,
            source_projections=tuple(request.import_projection for request in requests),
        )

    def executor(_module: Any, **kwargs: Any) -> dict[str, Any]:
        generations = kwargs["logical_generation_count"]
        state["dispatch_generations"].append(generations)
        result = {
            "completed_generations": generations,
            "final_stream_synchronization_time_ns": 41,
            "last_useful_kernel_completion_observed_host_time_ns": 1001,
            "final_host_synchronizations": 1,
            "final_device_to_host_operations": 12,
            "compact_settlement_d2h_operations": 11,
            "compact_summary_reporting_storage": {
                "final_transfer": {
                    "bytes": 1234,
                    "observed_host_transfer_count": 11,
                    "status": "native_stream_ordered_callback_wired",
                }
            },
        }
        if "absolute_tick_span" in kwargs:
            absolute_tick_span = kwargs["absolute_tick_span"]
            tick_native_control = dict(kwargs.get("tick_native_control") or {})
            source_window_ticks = int(tick_native_control["source_window_ticks"])
            source_window_count = (
                absolute_tick_span + source_window_ticks - 1
            ) // source_window_ticks
            result["absolute_tick_span"] = absolute_tick_span
            result["source_window_count"] = source_window_count
            result["final_host_synchronizations"] = source_window_count
            result["event_history_storage_mode"] = "eiel_resident_event_sink_v1"
            if isinstance(state["event_sink_path"], str):
                result["event_history_reference"] = {
                    "path": state["event_sink_path"]
                }
                result["event_history_frontier_by_cell"] = [0, 0, 0]
        if explicit_native_no_fallback:
            result.update(
                {
                    "owner": "native_extension",
                    "hidden_fallback": False,
                    "runtime_jit": False,
                }
            )
        if explicit_cpu_native_no_fallback:
            result.update(
                {
                    "owner": "cpu_native_extension",
                    "hidden_fallback": False,
                    "runtime_jit": False,
                    "cross_path_hidden_fallback": False,
                }
            )
        if direct_result_overrides is not None:
            result.update(direct_result_overrides)
        return result

    def projector(
        prepared: Any,
        *,
        direct_result: Mapping[str, Any],
        capability: Mapping[str, Any],
        reporting_enabled: bool,
    ) -> Any:
        assert capability == {"qualified": True}
        completed = direct_result["completed_generations"]
        projections = []
        rows = []
        artifacts = []
        for index, source in enumerate(prepared.source_projections):
            source_generation = int(source["metadata"]["generation"])
            source_tick = int(source["metadata"].get("tick", 0))
            span = direct_result.get("absolute_tick_span")
            absolute_start = direct_result.get("tick_native_start_next_tick")
            absolute_end = direct_result.get("tick_native_end_next_tick")
            if type(absolute_start) is int and type(absolute_end) is int:
                generation = absolute_end // 160
                terminal_tick = absolute_end % 160
            elif type(span) is int:
                absolute_end = source_tick + span
                generation = source_generation + absolute_end // 160
                terminal_tick = absolute_end % 160
            else:
                generation = source_generation + completed
                terminal_tick = 0
            segment_index = int(source["metadata"].get("native_next_segment_index", 0))
            previous_chain = str(
                source["metadata"].get("native_chain_sha256", "0" * 64)
            )
            artifact = _native_segment_artifact(
                cell_id=f"cell-{index}",
                cell_index=index,
                segment_index=segment_index,
                generation=generation - 1,
                previous_chain_sha256=previous_chain,
            )
            projected_projection = {
                "metadata": {
                    **source["metadata"],
                    "generation": generation,
                    "tick": terminal_tick,
                    "agent_grid_runner_phase": (
                        "generation_boundary"
                        if terminal_tick == 0
                        else "partial_generation"
                    ),
                    "native_next_segment_index": segment_index + 1,
                    "native_chain_sha256": artifact["cell"][
                        "chain_sha256_after_segment"
                    ],
                }
            }
            if cursor_tensor_fields:
                import torch

                projected_projection["tensors"] = {
                    "alive_mask": torch.ones(8, dtype=torch.bool),
                    "energy": torch.arange(8, dtype=torch.float32) + index,
                }
            projections.append(projected_projection)
            artifacts.append(artifact)
            rows.append(
                {
                    "cell_id": f"cell-{index}",
                    "generation": generation,
                    "alive": 64 - index,
                    "births": generation + index,
                    "deaths": index,
                }
            )
        state["projected_batches"].append(projections)
        return SimpleNamespace(
            checkpoint_projections=tuple(projections),
            compact_result_rows=tuple(rows) if reporting_enabled else (),
            terminal_generation=generation,
            terminal_tick=terminal_tick,
            segment_artifacts=(tuple(artifacts) if native_segment_artifacts else ()),
        )

    if bounded_projection_cursor_support:
        base_projector = projector

        def projector_with_cursor(
            prepared: Any,
            *,
            direct_result: Mapping[str, Any],
            capability: Mapping[str, Any],
            reporting_enabled: bool,
            bounded_projection_cursor: bool = False,
        ) -> Any:
            settlement = base_projector(
                prepared,
                direct_result=direct_result,
                capability=capability,
                reporting_enabled=reporting_enabled,
            )
            if not bounded_projection_cursor:
                return settlement
            projections = tuple(settlement.checkpoint_projections)
            terminal_generation = int(settlement.terminal_generation)
            terminal_tick = int(settlement.terminal_tick)
            coordinate = terminal_generation * 160 + terminal_tick

            def cursor_factory(
                index: int, cell_id: str
            ) -> DetachedCellProjectionV1:
                return DetachedCellProjectionV1(
                    schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
                    cell_id=cell_id,
                    cell_index=index,
                    projection=projections[index],
                    terminal_generation=terminal_generation,
                    terminal_tick=terminal_tick,
                    live_projection_bytes=1,
                )

            cursor = CheckpointProjectionCursorV1.begin(
                owner=prepared,
                coordinate=coordinate,
                ordered_cell_ids=tuple(
                    f"cell-{index}" for index in range(len(projections))
                ),
                factory=cursor_factory,
            )
            return SimpleNamespace(
                checkpoint_projections=(),
                compact_result_rows=(),
                terminal_generation=terminal_generation,
                terminal_tick=terminal_tick,
                segment_artifacts=(),
                checkpoint_projection_cursor=cursor,
            )

        projector = projector_with_cursor

    def checkpoint_writer(
        projection: Mapping[str, Any], path: str, **_kwargs: Any
    ) -> str:
        state["checkpoint_identities"].append(_kwargs["architecture_b_identity"])
        state["checkpoint_observer_continuations"].append(
            _kwargs["observer_continuation"]
        )
        store[path] = projection
        return path

    def checkpoint_loader(path: str, **_kwargs: Any) -> Mapping[str, Any]:
        state["checkpoint_loads"].append(path)
        return {"native_runtime_checkpoint_projection": {"path": path}}

    def checkpoint_decoder(sidecar: Mapping[str, Any]) -> Mapping[str, Any]:
        return store[str(sidecar["path"])]

    provider = _LifecycleHarness(
        artifact_loader=artifact_loader,
        runner_builder=runner_builder,
        request_builder=request_builder,
        preparer=preparer,
        executor=executor,
        projector=projector,
        checkpoint_writer=checkpoint_writer,
        checkpoint_loader=checkpoint_loader,
        checkpoint_decoder=checkpoint_decoder,
        checkpoint_digest=lambda projection: (
            f"{projection['metadata']['generation']:064x}"
        ),
        source_preparer=source_preparer,
    )
    return provider, state


def _strict_filesystem_provider_harness() -> tuple[_LifecycleHarness, dict[str, Any]]:
    """Use the provider harness with real Bundle-v2 persistence and strict load."""

    from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
        _projection,
    )
    from perf.cxxbackbone10_resident_state_owner import (
        migrate_native_checkpoint_projection_v1_to_v2,
    )

    provider, state = _provider_harness({})
    original_request_builder = provider._request_builder
    original_projector = provider._projector
    assert original_request_builder is not None
    assert original_projector is not None

    def full_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
        value = _projection(
            generation=int(projection["metadata"]["generation"]),
            tick=int(projection["metadata"].get("tick", 0)),
        )
        value["metadata"] = {
            **projection["metadata"],
            **value["metadata"],
        }
        return dict(migrate_native_checkpoint_projection_v1_to_v2(value))

    def filesystem_request_builder(*args: Any, **kwargs: Any) -> Any:
        request = original_request_builder(*args, **kwargs)
        projection = request.import_projection
        if "tensors" not in projection:
            request.import_projection = full_projection(projection)
            state["request_sources"][-1] = request.import_projection
        return request

    def filesystem_projector(*args: Any, **kwargs: Any) -> Any:
        settlement = original_projector(*args, **kwargs)
        values = vars(settlement).copy()
        values["checkpoint_projections"] = tuple(
            full_projection(projection)
            for projection in settlement.checkpoint_projections
        )
        return SimpleNamespace(**values)

    provider._request_builder = filesystem_request_builder
    provider._projector = filesystem_projector
    provider._checkpoint_writer = None
    provider._checkpoint_loader = None
    provider._checkpoint_decoder = None
    provider._checkpoint_digest = None
    return provider, state


def _filesystem_tree_identity(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_layered_checkpoint_digest_excludes_source_provenance_for_tensor_projection() -> None:
    from engine.native_runtime.checkpoint_digests import (
        DigestLayer,
        checkpoint_digest_manifest,
    )
    from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
        _projection,
    )

    left = _projection(generation=1, tick=17)
    right = _projection(generation=1, tick=17)
    left["metadata"]["event_history_reference"] = {
        "path": "/run/partition-default/events.bin"
    }
    right["metadata"]["event_history_reference"] = {
        "path": "/run/partition-alternate/events.bin"
    }
    left["metadata"]["import_digest"] = "a" * 64
    right["metadata"]["import_digest"] = "b" * 64

    provider = _LifecycleHarness(
        checkpoint_digest=lambda projection: canonical_sha256(projection["metadata"])
    )
    assert provider.terminal_projection_digest(left) != provider.terminal_projection_digest(
        right
    )

    for layer in DigestLayer:
        expected = checkpoint_digest_manifest(left, layer=layer)["digest"]
        assert provider._layered_checkpoint_digest_v1(left, layer) == expected
        assert provider._layered_checkpoint_digest_v1(left, layer) == (
            provider._layered_checkpoint_digest_v1(right, layer)
        )


def _install_production_control_readback_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    from engine.authoritative_state import runtime_abi
    from engine.campaigns import production_control_path_v1

    monkeypatch.setattr(
        runtime_abi,
        "resolve_prepared_direct_resident_trajectory_v1",
        lambda prepared: prepared,
    )
    monkeypatch.setattr(
        runtime_abi,
        "observe_applied_direct_resident_abi_v1",
        lambda _prepared: {
            "schema_version": "runtime_abi_receipt_v1",
            "phase": "post_dispatch",
            "receipt_sha256": "a" * 64,
            "outer_arguments": [],
        },
    )

    def build_readback(**kwargs: Any) -> dict[str, Any]:
        cells = []
        for cell, projection in zip(
            kwargs["cells"],
            kwargs["checkpoint_projections"],
            strict=True,
        ):
            cell_id = str(cell["cell_id"])
            bindings = {
                "cell_id": cell_id,
                "controls": [
                    {
                        "control": "patch.grid_size",
                        "requested": 16,
                        "compiled": 16,
                        "applied": 16,
                        "readback": 16,
                        "status": "READBACK_COMPLETE",
                    }
                ],
            }
            cells.append(
                {
                    **bindings,
                    "control_bindings_sha256": canonical_sha256(bindings),
                    "authoritative_state_projection_sha256": canonical_sha256(
                        projection
                    ),
                    "metric_row_sha256": "b" * 64,
                    "metric_binding_status": "BOUND",
                }
            )
        payload = {
            "schema_version": "eiel_production_control_readback_v1",
            "profile_id": kwargs["profile_id"],
            "segment_ordinal": kwargs["segment_ordinal"],
            "start_generation": kwargs["start_generation"],
            "end_generation": kwargs["end_generation"],
            "post_dispatch_abi_receipt_sha256": "a" * 64,
            "cells": cells,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload

    monkeypatch.setattr(
        production_control_path_v1,
        "build_segment_control_readback_v1",
        build_readback,
    )


def test_provider_dispatches_once_per_declared_segment_without_batch_whitelist(
    tmp_path: Path,
) -> None:
    provider, state = _provider_harness({})
    context = _Context(tmp_path)

    result = provider.execute_batch(_plan(), _request(), context)

    assert result["status"] == "completed"
    assert result["result"]["terminal_generation"] == 37
    assert result["result"]["native_dispatches"] == 2
    assert state["dispatch_generations"] == [14, 23]
    assert len(state["runner_ids"]) == 6
    assert len(set(state["runner_ids"])) == 6
    assert state["runner_generation_extents"] == [14] * 3 + [23] * 3
    assert [row["generation"] for row in context.records] == [14, 37]
    assert state["allocation_generations"] == [37, 37, 37, 23, 23, 23]
    assert context.records[0]["continuation_storage_mode"] == "disk"
    assert len(state["checkpoint_loads"]) == 3
    assert len(result["metrics_rows"]) == 6
    assert result["checkpoint"]["terminal_cell_digests"] == {
        "cell-0": f"{37:064x}",
        "cell-1": f"{37:064x}",
        "cell-2": f"{37:064x}",
    }
    identity = result["checkpoint"]["checkpoint_identity_contract"]
    assert identity["schema_version"] == CHECKPOINT_IDENTITY_CONTRACT_SCHEMA_VERSION
    assert identity["q5_terminal_equality_authority"] == (
        "canonical_terminal_state_digest_sha256s"
    )
    assert identity["route_provenance_may_differ_from_terminal_state"] is True
    assert result["checkpoint"]["terminal_projection_digest_sha256s"] == {
        "cell-0": f"{37:064x}",
        "cell-1": f"{37:064x}",
        "cell-2": f"{37:064x}",
    }
    assert result["checkpoint"]["canonical_terminal_state_digest_sha256s"] == {
        "cell-0": f"{37:064x}",
        "cell-1": f"{37:064x}",
        "cell-2": f"{37:064x}",
    }
    assert result["checkpoint"]["canonical_checkpoint_storage_digest_sha256s"] == {
        "cell-0": f"{37:064x}",
        "cell-1": f"{37:064x}",
        "cell-2": f"{37:064x}",
    }
    assert result["checkpoint"]["parent_checkpoint_projection_digest_sha256s"] == {
        "cell-0": f"{14:064x}",
        "cell-1": f"{14:064x}",
        "cell-2": f"{14:064x}",
    }
    route_digest = result["checkpoint"][
        "route_envelope_or_execution_provenance_digest_sha256"
    ]
    assert isinstance(route_digest, str)
    assert len(route_digest) == 64
    int(route_digest, 16)
    assert result["checkpoint"]["component_field_trace_status"] == (
        "exact_component_field_unavailable_only_terminal_projection_digest"
    )
    assert [
        row["source_projection_sha256"] for row in result["checkpoint"]["cell_bundles"]
    ] == [f"{37:064x}"] * 3
    identities = state["checkpoint_identities"]
    assert len({row["snapshot_id"] for row in identities}) == 6
    assert {row["owner_epoch"] for row in identities} == {14, 37}
    assert state["artifact_loads"] == [
        (
            "/qualified/controller.so",
            "/qualified/qualification.json",
            "/qualified/reporting.json",
        )
    ]


def test_two_slot_acs_overlaps_next_dispatch_and_publishes_only_after_seal(
    tmp_path: Path,
) -> None:
    provider, state = _provider_harness({})
    original_writer = provider._checkpoint_writer
    assert original_writer is not None
    first_writer_started = threading.Event()
    release_first_writer = threading.Event()
    second_dispatch_started = threading.Event()
    original_executor = provider._executor
    assert original_executor is not None

    def blocking_writer(projection: Mapping[str, Any], path: str, **kwargs: Any) -> str:
        if kwargs["generation_index"] == 14 and not first_writer_started.is_set():
            first_writer_started.set()
            assert release_first_writer.wait(5)
        return original_writer(projection, path, **kwargs)

    def observed_executor(module: Any, **kwargs: Any) -> Mapping[str, Any]:
        if len(state["dispatch_generations"]) == 1:
            second_dispatch_started.set()
        return original_executor(module, **kwargs)

    provider._checkpoint_writer = blocking_writer
    provider._executor = observed_executor
    context = _Context(tmp_path)
    request = replace(
        _request(),
        acs_checkpoint_slots=2,
        acs_capture_instrumentation=True,
    )
    result_box: list[dict[str, Any]] = []
    failure_box: list[BaseException] = []

    def execute() -> None:
        try:
            result_box.append(provider.execute_batch(_acs_plan(), request, context))
        except BaseException as exc:
            failure_box.append(exc)

    run_thread = threading.Thread(target=execute)
    run_thread.start()
    assert first_writer_started.wait(5)
    assert second_dispatch_started.wait(5)
    assert context.records == []
    release_first_writer.set()
    run_thread.join(10)

    assert not run_thread.is_alive()
    assert failure_box == []
    assert len(result_box) == 1
    result = result_box[0]
    assert [row["generation"] for row in context.records] == [14, 37]
    assert state["checkpoint_loads"] == []
    assert context.records[1]["parent_checkpoint_projection_digest_sha256s"] == {
        "cell-0": f"{14:064x}",
        "cell-1": f"{14:064x}",
        "cell-2": f"{14:064x}",
    }
    telemetry = result["result"]["acs_checkpoint_staging"]
    assert telemetry["slots"] == 2
    assert telemetry["queue_occupancy_high_water"] == 2
    assert telemetry["compute_frontier"] == 37
    assert telemetry["durable_checkpoint_frontier"] == 37
    assert telemetry["frontier_lag"] == 0
    assert telemetry["failure_latched"] is False
    assert telemetry["closed"] is True
    assert len(result["result"]["acs_capture_measurements"]) == 2
    first_measurement = result["result"]["acs_capture_measurements"][0]
    assert first_measurement["coordinate"] == 14
    assert first_measurement["captured_cell_count"] == 3
    assert first_measurement["capture_barrier_time_ns"] >= (
        first_measurement["checkpoint_projection_time_ns"]
    )
    assert first_measurement["capture_cpu_time_ns"] >= 0
    assert first_measurement["host_resources_before"]["rss_bytes"] > 0
    assert first_measurement["host_resources_after"]["pss_bytes"] > 0
    assert first_measurement["host_resources_after"]["pinned_bytes"] >= 0
    assert first_measurement["native_transfer_receipt"] == {
        "final_host_synchronizations": 1,
        "final_device_to_host_operations": 12,
        "compact_settlement_d2h_operations": 11,
        "final_transfer_bytes": 1234,
        "final_transfer_contract": {
            "bytes": 1234,
            "observed_host_transfer_count": 11,
            "status": "native_stream_ordered_callback_wired",
        },
    }
    assert set(first_measurement["immutable_snapshot_digest_sha256s"]) == {
        "cell-0",
        "cell-1",
        "cell-2",
    }
    assert len(result["result"]["acs_catalog_publication_measurements"]) == 2
    first_publication = result["result"]["acs_catalog_publication_measurements"][0]
    assert first_publication["coordinate"] == 14
    assert first_publication["total_finalization_time_ns"] >= (
        first_publication["context_publication_time_ns"]
    )
    for stages in context.records[0]["acs_checkpoint_staging"][
        "cell_stage_telemetry_seconds"
    ].values():
        assert stages["provider_post_digest_seconds"] >= 0.0
    assert context.records[0]["acs_checkpoint_staging"]["writer_cpu_time_ns"] > 0
    assert context.records[0]["acs_checkpoint_staging"]["checkpoint_bytes"] == 0
    assert (
        context.records[0]["acs_checkpoint_staging"][
            "checkpoint_bytes_measurement_status"
        ]
        == "unavailable_non_filesystem_writer"
    )


def test_acs_writer_failure_never_publishes_persisting_checkpoint(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({})

    def fail_writer(_projection: Mapping[str, Any], _path: str, **_kwargs: Any) -> str:
        raise OSError("injected ACS disk full")

    provider._checkpoint_writer = fail_writer
    context = _Context(tmp_path)

    with pytest.raises(Exception, match="injected ACS disk full") as caught:
        provider.execute_batch(
            _acs_plan(),
            replace(
                _request(),
                acs_checkpoint_slots=2,
                acs_capture_instrumentation=True,
            ),
            context,
        )

    assert isinstance(caught.value.__cause__, OSError)
    assert context.records == []
    assert provider._active_acs_pipeline is None


@pytest.mark.parametrize("acs_slots", (1, 2))
def test_acs_one_world_failure_never_publishes_partial_checkpoint_set(
    tmp_path: Path,
    acs_slots: int,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    original_writer = provider._checkpoint_writer
    assert original_writer is not None

    def fail_second_world(
        projection: Mapping[str, Any], path: str, **kwargs: Any
    ) -> str:
        if "/cell-1/" in path:
            raise OSError("injected ACS one-world failure")
        return original_writer(projection, path, **kwargs)

    provider._checkpoint_writer = fail_second_world
    context = _Context(tmp_path)

    with pytest.raises(Exception, match="one-world failure"):
        provider.execute_batch(
            _acs_plan(acs_slots),
            replace(
                _request(generations=14, milestones=(14,)),
                acs_checkpoint_slots=acs_slots,
                acs_capture_instrumentation=True,
            ),
            context,
        )

    assert context.records == []
    assert len(store) == 1
    assert "/cell-0/" in next(iter(store))
    assert provider._active_acs_pipeline is None


def test_acs_request_cannot_borrow_the_synchronous_route_binding(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({})

    with pytest.raises(CampaignProviderError, match="route binding mismatch"):
        provider.execute_batch(
            _plan(),
            replace(
                _request(),
                acs_checkpoint_slots=2,
                acs_capture_instrumentation=True,
            ),
            _Context(tmp_path),
        )


def test_acs_later_writer_failure_preserves_last_durable_checkpoint(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({})
    original_writer = provider._checkpoint_writer
    assert original_writer is not None

    def fail_terminal_writer(
        projection: Mapping[str, Any], path: str, **kwargs: Any
    ) -> str:
        if kwargs["generation_index"] == 37:
            raise OSError("injected terminal ACS write failure")
        return original_writer(projection, path, **kwargs)

    provider._checkpoint_writer = fail_terminal_writer
    context = _Context(tmp_path)

    with pytest.raises(Exception, match="terminal ACS write failure"):
        provider.execute_batch(
            _acs_plan(),
            replace(
                _request(),
                acs_checkpoint_slots=2,
                acs_capture_instrumentation=True,
            ),
            context,
        )

    assert [row["generation"] for row in context.records] == [14]
    assert context.records[0]["terminal_projection_digest_sha256s"] == {
        "cell-0": f"{14:064x}",
        "cell-1": f"{14:064x}",
        "cell-2": f"{14:064x}",
    }
    assert provider._active_acs_pipeline is None


def test_acs_catalog_publication_failure_leaves_successor_orphaned(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    observed_pipelines: list[Any] = []

    class FailingPublicationContext(_Context):
        def record_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
            pipeline = provider._active_acs_pipeline
            assert pipeline is not None
            observed_pipelines.append(pipeline)
            if checkpoint["generation"] == 20:
                raise OSError("injected ACS catalog publication failure")
            super().record_checkpoint(checkpoint)

    context = FailingPublicationContext(tmp_path)

    with pytest.raises(OSError, match="catalog publication failure"):
        provider.execute_batch(
            _acs_plan(1),
            replace(
                _request(generations=20, milestones=(10, 20)),
                acs_checkpoint_slots=1,
                acs_capture_instrumentation=True,
            ),
            context,
        )

    assert [row["generation"] for row in context.records] == [10]
    assert {projection["metadata"]["generation"] for projection in store.values()} == {
        10,
        20,
    }
    published_paths = {row["path"] for row in context.records[0]["cell_bundles"]}
    orphan_paths = {
        path
        for path, projection in store.items()
        if projection["metadata"]["generation"] == 20
    }
    assert orphan_paths
    assert orphan_paths.isdisjoint(published_paths)
    assert provider._active_acs_pipeline is None
    assert observed_pipelines[-1]._writer.is_alive() is False


def test_acs_durable_ack_failure_preserves_published_restart_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from engine.campaigns.acs_checkpoint_staging import ACSCheckpointStaging

    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    original_acknowledge = ACSCheckpointStaging.acknowledge_durable
    observed_pipelines: list[ACSCheckpointStaging[Any]] = []

    def fail_successor_acknowledgement(
        pipeline: ACSCheckpointStaging[Any], token: Any
    ) -> None:
        observed_pipelines.append(pipeline)
        if token.compute_frontier == 20:
            raise RuntimeError("injected ACS durable acknowledgement failure")
        original_acknowledge(pipeline, token)

    monkeypatch.setattr(
        ACSCheckpointStaging,
        "acknowledge_durable",
        fail_successor_acknowledgement,
    )
    context = _Context(tmp_path / "source")
    request = replace(
        _request(generations=30, milestones=(10, 20, 30)),
        acs_checkpoint_slots=1,
        acs_capture_instrumentation=True,
    )

    with pytest.raises(RuntimeError, match="durable acknowledgement failure"):
        provider.execute_batch(_acs_plan(1), request, context)

    assert [row["generation"] for row in context.records] == [10, 20]
    assert provider._active_acs_pipeline is None
    assert observed_pipelines[-1]._writer.is_alive() is False

    restore_provider, _restore_state = _provider_harness(store)
    restored = restore_provider.resume_batch(
        _acs_plan(1),
        request,
        context.records[-1],
        _Context(tmp_path / "restored"),
    )

    assert restored["status"] == "completed"
    assert restored["result"]["resumed_from_generation"] == 20
    assert restored["result"]["terminal_generation"] == 30


@pytest.mark.parametrize(
    (
        "crash_stage",
        "acs_slots",
        "exit_code",
        "restart_generation",
        "expected_dispatches",
        "precrash_generation_20_cells",
    ),
    (
        ("after_bundle_before_catalog", 1, 91, 10, [10, 10], (0, 1, 2)),
        ("after_catalog_before_ack", 1, 92, 20, [10], (0, 1, 2)),
        ("after_bundle_before_catalog", 2, 93, 10, [10, 10], (0, 1, 2)),
        ("after_catalog_before_ack", 2, 94, 20, [10], (0, 1, 2)),
        ("during_partial_set_before_checkpoint_complete", 2, 95, 10, [10, 10], (0,)),
    ),
)
def test_acs_actual_provider_process_crash_restores_last_durable_frontier(
    tmp_path: Path,
    crash_stage: str,
    acs_slots: int,
    exit_code: int,
    restart_generation: int,
    expected_dispatches: list[int],
    precrash_generation_20_cells: tuple[int, ...],
) -> None:
    """A fresh process resumes only the catalogued actual-provider frontier."""

    from engine.simulation.checkpoint_bundle_v2 import (
        ARCHITECTURE_B_BUNDLE_KIND,
        checkpoint_bundle_v2_load,
    )

    batch_root = tmp_path / "batch"
    batch_root.mkdir()
    manifest_path = batch_root / "manifest.json"
    route_sha256 = "6" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "eiel_campaign_batch_manifest_v1",
                "status": "running",
                "checkpoint": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    crash_pid_path = tmp_path / "crash.pid"
    crash_script = r"""
import os
from dataclasses import replace
from pathlib import Path
import sys

from engine.campaigns.acs_checkpoint_staging import ACSCheckpointStaging
from engine.campaigns.orchestration import BatchExecutionContextV1
from engine.simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_save_projection
from tests.test_eiel_campaign_patch_provider_execution_v1 import (
    _acs_plan,
    _request,
    _strict_filesystem_provider_harness,
)

batch_root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
stage = sys.argv[3]
exit_code = int(sys.argv[4])
Path(sys.argv[5]).write_text(str(os.getpid()), encoding="utf-8")
acs_slots = int(sys.argv[6])
provider, _state = _strict_filesystem_provider_harness()
request = replace(
    _request(generations=30, milestones=(10, 20, 30)),
    acs_checkpoint_slots=acs_slots,
    acs_capture_instrumentation=True,
)
context = BatchExecutionContextV1(
    run_root=batch_root.parent,
    batch_root=batch_root,
    manifest_path=manifest_path,
    execution_route_sha256="6" * 64,
)
if stage == "during_partial_set_before_checkpoint_complete":
    def crash_during_second_cell_write(projection, path, **kwargs):
        if kwargs["generation_index"] == 20 and "/cell-1/" in str(path):
            os._exit(exit_code)
        return checkpoint_bundle_v2_save_projection(projection, path, **kwargs)

    provider._checkpoint_writer = crash_during_second_cell_write
elif stage == "after_bundle_before_catalog":
    original_record = BatchExecutionContextV1.record_checkpoint

    def crash_before_catalog(self, checkpoint):
        if checkpoint.get("generation") == 20:
            os._exit(exit_code)
        original_record(self, checkpoint)

    BatchExecutionContextV1.record_checkpoint = crash_before_catalog
else:
    original_ack = ACSCheckpointStaging.acknowledge_durable

    def crash_before_ack(self, token):
        if token.compute_frontier == 20:
            os._exit(exit_code)
        original_ack(self, token)

    ACSCheckpointStaging.acknowledge_durable = crash_before_ack
provider.execute_batch(_acs_plan(acs_slots), request, context)
os._exit(99)
"""
    child_environment = dict(os.environ)
    child_environment["CUDA_VISIBLE_DEVICES"] = ""
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            crash_script,
            str(batch_root),
            str(manifest_path),
            crash_stage,
            str(exit_code),
            str(crash_pid_path),
            str(acs_slots),
        ],
        cwd=Path(__file__).parents[1],
        env=child_environment,
        check=False,
    )

    assert crashed.returncode == exit_code
    crash_pid = int(crash_pid_path.read_text(encoding="utf-8"))
    crashed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    durable_checkpoint = crashed_manifest["checkpoint"]
    assert durable_checkpoint["generation"] == restart_generation
    assert durable_checkpoint["execution_route_sha256"] == route_sha256
    assert [row["cell_id"] for row in durable_checkpoint["cell_bundles"]] == [
        "cell-0",
        "cell-1",
        "cell-2",
    ]
    durable_paths = [row["path"] for row in durable_checkpoint["cell_bundles"]]
    generation_20_roots = [
        batch_root / "checkpoint_bundles_v2" / f"cell-{index}" / "bundle_gen000020"
        for index in range(3)
    ]
    for generation in (10,):
        for index in range(3):
            bundle = (
                batch_root
                / "checkpoint_bundles_v2"
                / f"cell-{index}"
                / f"bundle_gen{generation:06d}"
            )
            loaded = checkpoint_bundle_v2_load(
                str(bundle),
                strict=True,
                required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
            )
            assert loaded["bundle_complete"]["status"] == "complete"
    assert {
        index for index, path in enumerate(generation_20_roots) if path.exists()
    } == set(precrash_generation_20_cells)
    for index in precrash_generation_20_cells:
        loaded = checkpoint_bundle_v2_load(
            str(generation_20_roots[index]),
            strict=True,
            required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
        )
        assert loaded["bundle_complete"]["status"] == "complete"
    if restart_generation == 10:
        assert all(str(path) not in durable_paths for path in generation_20_roots)
    else:
        assert durable_paths == [str(path) for path in generation_20_roots]
        assert set(durable_checkpoint["prior_checkpoint_catalog"]) == {"10"}
    generation_20_identity = {
        path.parent.name: _filesystem_tree_identity(path)
        for index, path in enumerate(generation_20_roots)
        if index in precrash_generation_20_cells
    }

    recovery_path = tmp_path / "recovery.json"
    recovery_script = r"""
import json
import os
from dataclasses import replace
from pathlib import Path
import sys

from engine.campaigns.orchestration import BatchExecutionContextV1
from tests.test_eiel_campaign_patch_provider_execution_v1 import (
    _acs_plan,
    _request,
    _strict_filesystem_provider_harness,
)

batch_root = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()
acs_slots = int(sys.argv[4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
source_checkpoint = manifest["checkpoint"]
provider, state = _strict_filesystem_provider_harness()
request = replace(
    _request(generations=30, milestones=(10, 20, 30)),
    acs_checkpoint_slots=acs_slots,
    acs_capture_instrumentation=True,
)
context = BatchExecutionContextV1(
    run_root=batch_root.parent,
    batch_root=batch_root,
    manifest_path=manifest_path,
    execution_route_sha256="6" * 64,
)
result = provider.resume_batch(
    _acs_plan(acs_slots), request, source_checkpoint, context
)
receipt = result["result"]["fresh_owner_restore_receipt"]
telemetry = result["result"]["acs_checkpoint_staging"]
terminal_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
output_path.write_text(
    json.dumps(
        {
            "pid": os.getpid(),
            "status": result["status"],
            "resumed_from_generation": result["result"][
                "resumed_from_generation"
            ],
            "terminal_generation": result["result"]["terminal_generation"],
            "native_dispatches": result["result"]["native_dispatches"],
            "dispatch_generations": state["dispatch_generations"],
            "fresh_owner_restore_receipt": receipt,
            "acs_checkpoint_staging": telemetry,
            "terminal_manifest_checkpoint_generation": terminal_manifest[
                "checkpoint"
            ]["generation"],
            "terminal_manifest_cell_ids": [
                row["cell_id"]
                for row in terminal_manifest["checkpoint"]["cell_bundles"]
            ],
            "terminal_manifest_digests": terminal_manifest["checkpoint"][
                "terminal_projection_digest_sha256s"
            ],
            "checkpoint_catalog_coordinates": sorted(
                result["result"]["checkpoint_catalog"], key=int
            ),
        },
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
"""
    recovered = subprocess.run(
        [
            sys.executable,
            "-c",
            recovery_script,
            str(batch_root),
            str(manifest_path),
            str(recovery_path),
            str(acs_slots),
        ],
        cwd=Path(__file__).parents[1],
        env=child_environment,
        check=False,
    )

    assert recovered.returncode == 0
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert recovery["pid"] != crash_pid
    assert recovery["status"] == "completed"
    assert recovery["resumed_from_generation"] == restart_generation
    assert recovery["terminal_generation"] == 30
    assert recovery["native_dispatches"] == len(expected_dispatches)
    assert recovery["dispatch_generations"] == expected_dispatches
    receipt = recovery["fresh_owner_restore_receipt"]
    assert receipt["schema_version"] == FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION
    assert receipt["projection_bundles_reloaded"] is True
    assert receipt["source_checkpoint_generation"] == restart_generation
    assert [row["cell_id"] for row in receipt["reload_receipts"]] == [
        "cell-0",
        "cell-1",
        "cell-2",
    ]
    assert [row["bundle_path"] for row in receipt["reload_receipts"]] == (durable_paths)
    assert all(
        row["source_projection_sha256"] == row["loaded_projection_sha256"]
        for row in receipt["reload_receipts"]
    )
    telemetry = recovery["acs_checkpoint_staging"]
    assert telemetry["slots"] == acs_slots
    assert telemetry["slot_states"] == ["FREE"] * acs_slots
    assert telemetry["closed"] is True
    assert telemetry["writer_alive"] is False
    assert telemetry["failure_latched"] is False
    assert telemetry["durable_checkpoint_frontier"] == 30
    assert telemetry["frontier_lag"] == 0
    assert recovery["terminal_manifest_checkpoint_generation"] == 30
    assert recovery["terminal_manifest_cell_ids"] == [
        "cell-0",
        "cell-1",
        "cell-2",
    ]
    assert set(recovery["terminal_manifest_digests"]) == {
        "cell-0",
        "cell-1",
        "cell-2",
    }
    assert recovery["checkpoint_catalog_coordinates"] == ["10", "20", "30"]
    for generation in (20, 30):
        for index in range(3):
            loaded = checkpoint_bundle_v2_load(
                str(
                    batch_root
                    / "checkpoint_bundles_v2"
                    / f"cell-{index}"
                    / f"bundle_gen{generation:06d}"
                ),
                strict=True,
                required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
            )
            assert loaded["bundle_complete"]["status"] == "complete"
    assert {
        path.parent.name: _filesystem_tree_identity(path)
        for index, path in enumerate(generation_20_roots)
        if index in precrash_generation_20_cells
    } == generation_20_identity


def test_acs_reserves_slots_before_capture_and_rejects_snapshot_mutation(
    tmp_path: Path,
) -> None:
    provider, state = _provider_harness({})
    original_writer = provider._checkpoint_writer
    original_projector = provider._projector
    assert original_writer is not None
    assert original_projector is not None
    first_writer_started = threading.Event()
    release_first_writer = threading.Event()
    second_snapshot_captured = threading.Event()

    def blocking_writer(projection: Mapping[str, Any], path: str, **kwargs: Any) -> str:
        if kwargs["generation_index"] == 10 and not first_writer_started.is_set():
            first_writer_started.set()
            assert release_first_writer.wait(5)
        return original_writer(projection, path, **kwargs)

    def observed_projector(*args: Any, **kwargs: Any) -> Any:
        settlement = original_projector(*args, **kwargs)
        if len(state["projected_batches"]) == 2:
            second_snapshot_captured.set()
        return settlement

    provider._checkpoint_writer = blocking_writer
    provider._projector = observed_projector
    context = _Context(tmp_path)
    failure_box: list[BaseException] = []

    def execute() -> None:
        try:
            provider.execute_batch(
                _acs_plan(),
                replace(
                    _request(generations=30, milestones=(10, 20, 30)),
                    acs_checkpoint_slots=2,
                    acs_capture_instrumentation=True,
                ),
                context,
            )
        except BaseException as exc:
            failure_box.append(exc)

    run_thread = threading.Thread(target=execute)
    run_thread.start()
    assert first_writer_started.wait(5)
    assert second_snapshot_captured.wait(5)
    assert state["dispatch_generations"] == [10, 10]
    state["projected_batches"][1][0]["metadata"]["generation"] = 999
    release_first_writer.set()
    run_thread.join(10)

    assert not run_thread.is_alive()
    assert len(failure_box) == 1
    assert "immutable checkpoint snapshot changed" in str(failure_box[0])
    assert state["dispatch_generations"] == [10, 10]
    assert [row["generation"] for row in context.records] == [10]
    assert provider._active_acs_pipeline is None


def test_acs_native_capture_receipt_binds_sync_and_transfer_authority(
    tmp_path: Path,
) -> None:
    provider, state = _provider_harness(
        {},
        direct_result_overrides={
            "final_stream_synchronization_time_ns": 41,
            "last_useful_kernel_completion_observed_host_time_ns": 1001,
            "final_host_synchronizations": 1,
            "final_device_to_host_operations": 12,
            "compact_settlement_d2h_operations": 11,
            "compact_summary_reporting_storage": {
                "final_transfer": {
                    "bytes": 1234,
                    "observed_host_transfer_count": 11,
                    "status": "native_stream_ordered_callback_wired",
                }
            },
        },
    )

    result = provider.execute_batch(
        _acs_plan(1),
        replace(
            _request(),
            acs_checkpoint_slots=1,
            acs_capture_instrumentation=True,
        ),
        _Context(tmp_path),
    )

    measurement = result["result"]["acs_capture_measurements"][0]
    assert measurement["native_final_stream_synchronization_time_ns"] == 41
    assert measurement["last_useful_kernel_completion_observed_host_time_ns"] == 1001
    assert measurement["capture_barrier_time_ns"] == (
        measurement["host_capture_freeze_time_ns"] + 41
    )
    assert measurement["native_transfer_receipt"] == {
        "final_host_synchronizations": 1,
        "final_device_to_host_operations": 12,
        "compact_settlement_d2h_operations": 11,
        "final_transfer_bytes": 1234,
        "final_transfer_contract": {
            "bytes": 1234,
            "observed_host_transfer_count": 11,
            "status": "native_stream_ordered_callback_wired",
        },
    }


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"final_stream_synchronization_time_ns": None},
            "final_stream_synchronization_time_ns",
        ),
        (
            {"last_useful_kernel_completion_observed_host_time_ns": True},
            "last_useful_kernel_completion_observed_host_time_ns",
        ),
        (
            {"last_useful_kernel_completion_observed_host_time_ns": 0},
            "last-useful-kernel boundary must be positive",
        ),
        (
            {"final_host_synchronizations": -1},
            "final_host_synchronizations",
        ),
        (
            {"final_device_to_host_operations": 1.0},
            "final_device_to_host_operations",
        ),
        (
            {"compact_settlement_d2h_operations": None},
            "compact_settlement_d2h_operations",
        ),
        (
            {"compact_summary_reporting_storage": None},
            "lacks compact reporting storage",
        ),
        (
            {"compact_summary_reporting_storage": {"final_transfer": None}},
            "lacks final transfer contract",
        ),
        (
            {
                "compact_summary_reporting_storage": {
                    "final_transfer": {
                        "bytes": True,
                        "observed_host_transfer_count": 11,
                        "status": "native_stream_ordered_callback_wired",
                    }
                }
            },
            "final transfer bytes",
        ),
        (
            {
                "compact_summary_reporting_storage": {
                    "final_transfer": {
                        "bytes": 1234,
                        "observed_host_transfer_count": 10,
                        "status": "native_stream_ordered_callback_wired",
                    }
                }
            },
            "final transfer count differs",
        ),
        (
            {
                "compact_summary_reporting_storage": {
                    "final_transfer": {
                        "bytes": 1234,
                        "observed_host_transfer_count": 11,
                        "status": "wired",
                    }
                }
            },
            "final transfer status is invalid",
        ),
    ],
)
def test_acs_native_capture_receipt_fails_closed_before_publication(
    tmp_path: Path,
    overrides: Mapping[str, Any],
    error: str,
) -> None:
    provider, _state = _provider_harness(
        {},
        direct_result_overrides=overrides,
    )
    context = _Context(tmp_path)

    with pytest.raises(CampaignProviderError, match=error):
        provider.execute_batch(
            _acs_plan(1),
            replace(
                _request(),
                acs_checkpoint_slots=1,
                acs_capture_instrumentation=True,
            ),
            context,
        )

    assert context.records == []


def test_acs0_instrumentation_only_route_remains_synchronous(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({})
    original_writer = provider._checkpoint_writer
    original_executor = provider._executor
    assert original_writer is not None
    assert original_executor is not None
    writer_active = False

    def observed_writer(projection: Mapping[str, Any], path: str, **kwargs: Any) -> str:
        nonlocal writer_active
        assert writer_active is False
        writer_active = True
        try:
            return original_writer(projection, path, **kwargs)
        finally:
            writer_active = False

    def observed_executor(module: Any, **kwargs: Any) -> Mapping[str, Any]:
        assert writer_active is False
        return original_executor(module, **kwargs)

    provider._checkpoint_writer = observed_writer
    provider._executor = observed_executor
    plan = _plan()
    plan["execution"] = {"acs_capture_instrumentation": True}
    context = _Context(tmp_path)

    result = provider.execute_batch(
        plan,
        replace(_request(), acs_capture_instrumentation=True),
        context,
    )

    assert provider._active_acs_pipeline is None
    assert "acs_checkpoint_staging" not in result["result"]
    assert len(result["result"]["acs_capture_measurements"]) == 2
    assert all(
        row["mode"] == "synchronous_instrumentation_only_v1"
        for row in result["result"]["acs_catalog_publication_measurements"]
    )
    assert all(
        record["acs_checkpoint_instrumentation"]["mode"]
        == "synchronous_instrumentation_only_v1"
        for record in context.records
    )
    assert all(
        record["acs_checkpoint_instrumentation"]["writer_service_time_ns"] > 0
        for record in context.records
    )


def test_acs_snapshot_logical_content_matches_synchronous_checkpoint_input(
    tmp_path: Path,
) -> None:
    synchronous_store: dict[str, Mapping[str, Any]] = {}
    synchronous_provider, _sync_state = _provider_harness(synchronous_store)
    synchronous_context = _Context(tmp_path / "synchronous")
    synchronous_provider.execute_batch(
        _plan(),
        _request(),
        synchronous_context,
    )

    acs_store: dict[str, Mapping[str, Any]] = {}
    acs_provider, _acs_state = _provider_harness(acs_store)
    acs_context = _Context(tmp_path / "acs")
    acs_provider.execute_batch(
        _acs_plan(1),
        replace(
            _request(),
            acs_checkpoint_slots=1,
            acs_capture_instrumentation=True,
        ),
        acs_context,
    )

    assert len(synchronous_context.records) == len(acs_context.records) == 2
    for synchronous, asynchronous in zip(
        synchronous_context.records,
        acs_context.records,
        strict=True,
    ):
        assert asynchronous["generation"] == synchronous["generation"]
        assert asynchronous["terminal_projection_digest_sha256s"] == (
            synchronous["terminal_projection_digest_sha256s"]
        )
        assert asynchronous["canonical_terminal_state_digest_sha256s"] == (
            synchronous["canonical_terminal_state_digest_sha256s"]
        )
        assert (
            asynchronous["canonical_checkpoint_storage_digest_sha256s"]
            == synchronous["canonical_checkpoint_storage_digest_sha256s"]
        )
        assert asynchronous["metrics_rows"] == synchronous["metrics_rows"]
        for sync_bundle, acs_bundle in zip(
            synchronous["cell_bundles"],
            asynchronous["cell_bundles"],
            strict=True,
        ):
            assert synchronous_store[sync_bundle["path"]] == (
                acs_store[acs_bundle["path"]]
            )


def test_acs_restore_uses_last_durable_frontier_while_successor_persists(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    original_writer = provider._checkpoint_writer
    assert original_writer is not None
    successor_persisting = threading.Event()
    release_successor = threading.Event()
    durable_frontier_published = threading.Event()

    def blocking_successor_writer(
        projection: Mapping[str, Any], path: str, **kwargs: Any
    ) -> str:
        if kwargs["generation_index"] == 20 and not successor_persisting.is_set():
            successor_persisting.set()
            assert release_successor.wait(8)
        return original_writer(projection, path, **kwargs)

    class SignalingContext(_Context):
        def record_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
            super().record_checkpoint(checkpoint)
            if checkpoint["generation"] == 10:
                durable_frontier_published.set()

    provider._checkpoint_writer = blocking_successor_writer
    source_context = SignalingContext(tmp_path / "source")
    request = replace(
        _request(generations=30, milestones=(10, 20, 30)),
        acs_checkpoint_slots=2,
        acs_capture_instrumentation=True,
    )
    source_failure: list[BaseException] = []

    def execute_source() -> None:
        try:
            provider.execute_batch(_acs_plan(), request, source_context)
        except BaseException as exc:
            source_failure.append(exc)

    source_thread = threading.Thread(target=execute_source)
    source_thread.start()
    assert successor_persisting.wait(5)
    assert durable_frontier_published.wait(5)
    assert [row["generation"] for row in source_context.records] == [10]

    restore_provider, _restore_state = _provider_harness(store)
    restored = restore_provider.resume_batch(
        _acs_plan(),
        request,
        source_context.records[0],
        _Context(tmp_path / "restored"),
    )
    assert restored["status"] == "completed"
    assert restored["result"]["resumed_from_generation"] == 10
    assert restored["result"]["terminal_generation"] == 30

    release_successor.set()
    source_thread.join(10)
    assert not source_thread.is_alive()
    assert source_failure == []


def test_tick_native_exact_partial_checkpoint_fresh_owner_restore_and_terminal(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, state = _provider_harness(store)
    request = _tick_request()
    source_context = _Context(tmp_path / "source")

    interrupted = provider.execute_batch(_plan(), request, source_context)

    assert interrupted["status"] == "interrupted"
    assert interrupted["checkpoint"]["next_absolute_tick"] == 17
    assert interrupted["checkpoint"]["time_authority"] == "absolute_tick_v1"
    assert [row["next_absolute_tick"] for row in source_context.records] == [17]

    restore_context = _Context(tmp_path / "restore")
    completed = provider.resume_batch(
        _plan(), request, interrupted["checkpoint"], restore_context
    )

    assert completed["status"] == "completed"
    assert completed["result"]["terminal_next_absolute_tick"] == 161
    assert completed["checkpoint"]["next_absolute_tick"] == 161
    assert completed["checkpoint"]["time_authority"] == "absolute_tick_v1"
    assert [row["next_absolute_tick"] for row in restore_context.records] == [160, 161]
    assert state["dispatch_generations"] == [1, 1, 1]
    receipt = completed["result"]["fresh_owner_restore_receipt"]
    assert receipt["restore_next_tick"] == 17
    assert receipt["first_executed_absolute_tick"] == 17
    assert receipt["first_committed_next_tick"] == 18
    assert receipt["off_by_one_check"] == "PASS"


@pytest.mark.parametrize(
    ("total_ticks", "checkpoints", "restore_tick"),
    (
        (161, (17, 160, 161), 17),
        (1121, (500, 1120, 1121), 500),
        (5000, (2500, 5000), 2500),
    ),
)
def test_tick_native_compatibility_removed_restore_keeps_first_dispatch_and_trajectory(
    tmp_path: Path,
    total_ticks: int,
    checkpoints: tuple[int, ...],
    restore_tick: int,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    source_provider, _source_state = _provider_harness(store)
    request = _tick_request(
        total_ticks=total_ticks,
        checkpoints=checkpoints,
        restore_tick=restore_tick,
    )
    interrupted = source_provider.execute_batch(
        _plan(), request, _Context(tmp_path / "source")
    )
    checkpoint = interrupted["checkpoint"]
    compatibility_removed = copy.deepcopy(checkpoint)
    compatibility_removed["checkpoint_identity_contract"].pop(
        "legacy_compatibility", None
    )

    ordinary_provider, ordinary_state = _provider_harness(store)
    ordinary = ordinary_provider.resume_batch(
        _plan(), request, checkpoint, _Context(tmp_path / "ordinary")
    )
    stripped_provider, stripped_state = _provider_harness(store)
    stripped = stripped_provider.resume_batch(
        _plan(), request, compatibility_removed, _Context(tmp_path / "stripped")
    )

    assert ordinary_state["dispatch_generations"] == stripped_state[
        "dispatch_generations"
    ]
    assert ordinary["result"]["fresh_owner_restore_receipt"][
        "first_executed_absolute_tick"
    ] == stripped["result"]["fresh_owner_restore_receipt"][
        "first_executed_absolute_tick"
    ] == restore_tick
    assert ordinary["metrics_rows"] == stripped["metrics_rows"]
    for field in (
        "terminal_cell_digests",
        "canonical_terminal_state_digest_sha256s",
        "canonical_checkpoint_storage_digest_sha256s",
    ):
        assert ordinary["checkpoint"][field] == stripped["checkpoint"][field]
    assert ordinary["checkpoint"]["next_absolute_tick"] == stripped[
        "checkpoint"
    ]["next_absolute_tick"] == total_ticks


def test_tick_native_two_independent_fresh_owner_restore_boundaries(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    request = _tick_request(
        total_ticks=49,
        checkpoints=(17, 33, 49),
        restore_tick=None,
        restore_ticks=(17, 33),
    )

    owner_a = provider.execute_batch(_plan(), request, _Context(tmp_path / "owner-a"))
    assert owner_a["status"] == "interrupted"
    assert owner_a["checkpoint"]["next_absolute_tick"] == 17

    owner_b = provider.resume_batch(
        _plan(), request, owner_a["checkpoint"], _Context(tmp_path / "owner-b")
    )
    assert owner_b["status"] == "interrupted"
    assert owner_b["checkpoint"]["next_absolute_tick"] == 33
    assert owner_b["result"]["fresh_owner_restore_receipt"]["restore_next_tick"] == 17

    owner_c = provider.resume_batch(
        _plan(), request, owner_b["checkpoint"], _Context(tmp_path / "owner-c")
    )
    assert owner_c["status"] == "completed"
    assert owner_c["checkpoint"]["next_absolute_tick"] == 49
    receipt = owner_c["result"]["fresh_owner_restore_receipt"]
    assert receipt["restore_next_tick"] == 33
    assert receipt["first_executed_absolute_tick"] == 33
    assert receipt["first_committed_next_tick"] == 34
    assert receipt["off_by_one_check"] == "PASS"


def test_tick_native_source_factory_receives_no_generation_extent(
    tmp_path: Path,
) -> None:
    observed: list[dict[str, Any]] = []

    def prepare(
        _plan: Mapping[str, Any],
        _request: ResidentBatchRequestV1,
        _context: Any,
        **kwargs: Any,
    ) -> tuple[Mapping[str, Any], ...]:
        observed.append(dict(kwargs))
        return tuple(
            {
                "metadata": {
                    "generation": 0,
                    "tick": 0,
                    "agent_grid_runner_phase": "generation_boundary",
                    "policy_weight_mutation_enable": False,
                    "policy_weight_mutation_mode": "dense",
                    "policy_weight_mutation_rate": 0.0,
                    "policy_weight_mutation_std": 0.0,
                }
            }
            for _cell_id in _request.cell_ids
        )

    provider, _state = _provider_harness({}, source_preparer=prepare)

    provider.execute_batch(_plan(), _tick_request(), _Context(tmp_path))

    assert len(observed) == 1
    assert "segment_generations" not in observed[0]


def test_trim_cuda_allocator_orders_collection_before_cuda_empty_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeCuda:
        def is_available(self) -> bool:
            events.append("cuda_available")
            return True

        def empty_cache(self) -> None:
            events.append("empty_cache")

    monkeypatch.setattr(
        providers_module.gc,
        "collect",
        lambda: events.append("gc") or 0,
    )

    _trim_cuda_allocator_after_resident_owner_preparation_v1(
        torch_module=SimpleNamespace(cuda=FakeCuda())
    )

    assert events == ["gc", "cuda_available", "empty_cache"]


def test_trim_cuda_allocator_is_noop_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeCuda:
        def is_available(self) -> bool:
            events.append("cuda_available")
            return False

        def empty_cache(self) -> None:
            raise AssertionError("empty_cache must not run without CUDA")

    monkeypatch.setattr(
        providers_module.gc,
        "collect",
        lambda: events.append("gc") or 0,
    )

    _trim_cuda_allocator_after_resident_owner_preparation_v1(
        torch_module=SimpleNamespace(cuda=FakeCuda())
    )

    assert events == ["gc", "cuda_available"]


def test_trim_cuda_allocator_fails_closed_when_empty_cache_fails() -> None:
    class FailingCuda:
        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            raise RuntimeError("allocator failure")

    with pytest.raises(
        CampaignProviderError,
        match="CUDA allocator trim failed after resident owner preparation",
    ):
        _trim_cuda_allocator_after_resident_owner_preparation_v1(
            torch_module=SimpleNamespace(cuda=FailingCuda())
        )


def test_tick_native_trims_once_when_logical_owner_epochs_transfer_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _state = _provider_harness({}, track_owner_release=True)
    events: list[str] = []
    original_prepare = provider._prepare_resident_segment_owner_v1
    original_resource_sample = provider._acs_process_resource_sample_v1

    def prepare_owner(**kwargs: Any) -> tuple[Any, ...]:
        result = original_prepare(**kwargs)
        events.append("prepare_return")
        return result

    def resource_sample() -> dict[str, int | str | None]:
        events.append("resource_sample")
        return original_resource_sample()

    monkeypatch.setattr(
        provider,
        "_prepare_resident_segment_owner_v1",
        prepare_owner,
    )
    monkeypatch.setattr(
        providers_module,
        "_trim_cuda_allocator_after_resident_owner_preparation_v1",
        lambda: events.append("trim"),
    )
    monkeypatch.setattr(
        provider,
        "_acs_process_resource_sample_v1",
        resource_sample,
    )

    result = provider.execute_batch(
        _plan(),
        _tick_request(
            total_ticks=2400,
            checkpoints=(2400,),
            restore_tick=None,
        ),
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    assert result["result"]["resident_owner_preparations"] == 1
    assert len(result["result"]["resident_owner_epoch_transfer_receipts"]) == 2
    assert events.count("prepare_return") == 1
    assert events.count("trim") == 1
    assert all(
        events[index + 1] == "trim"
        for index, event in enumerate(events[:-1])
        if event == "prepare_return"
    )
    first_trim = events.index("trim")
    first_resource_sample = events.index("resource_sample")
    assert first_trim < first_resource_sample


def test_tick_native_releases_completed_owner_before_successor_preparation(
    tmp_path: Path,
) -> None:
    provider, state = _provider_harness({}, track_owner_release=True)

    result = provider.execute_batch(
        _plan(),
        _tick_request(
            total_ticks=161,
            checkpoints=(17, 160, 161),
            restore_tick=None,
        ),
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    assert state["prior_owner_alive_before_prepare"] == [[]]
    assert result["result"]["resident_owner_preparations"] == 1


def test_tick_native_slot_zero_does_not_checkpoint_logical_owner_transfer(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({}, bounded_projection_cursor_support=True)
    result = provider.execute_batch(
        _acs_plan(slots=0),
        _tick_request(
            total_ticks=1200,
            checkpoints=(1200,),
            restore_tick=None,
            acs_checkpoint_slots=0,
            acs_capture_instrumentation=True,
        ),
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    captures = result["result"]["acs_capture_measurements"]
    assert [capture["coordinate"] for capture in captures] == [1200]
    assert all(
        type(capture["capture_completed_unix_ns"]) is int for capture in captures
    )
    assert len(result["result"]["bounded_terminal_capture_receipts"]) == 1
    transfers = result["result"]["resident_owner_epoch_transfer_receipts"]
    assert [row["boundary_next_absolute_tick"] for row in transfers] == [1120]
    assert transfers[0]["physical_owner_reused"] is True
    assert transfers[0]["cuda_allocation_recreated"] is False


def test_tick_native_bounded_projection_capture_owns_clock_without_acs_measurements(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness(
        {},
        bounded_projection_cursor_support=True,
    )
    result = provider.execute_batch(
        _plan(),
        _tick_request(
            total_ticks=161,
            checkpoints=(161,),
            restore_tick=None,
            acs_checkpoint_slots=0,
            acs_capture_instrumentation=False,
        ),
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    execution = result["result"]
    receipts = execution["bounded_terminal_capture_receipts"]
    assert receipts
    assert all(
        type(receipt["capture_started_unix_ns"]) is int
        and type(receipt["capture_completed_unix_ns"]) is int
        and receipt["capture_started_unix_ns"]
        <= receipt["capture_completed_unix_ns"]
        for receipt in receipts
    )
    assert [receipt["coordinate"] for receipt in receipts] == sorted(
        receipt["coordinate"] for receipt in receipts
    )
    assert execution["acs_capture_measurements"] == []
    assert execution["acs_catalog_publication_measurements"] == []


@pytest.mark.parametrize("acs_slots", (1, 2))
def test_tick_native_native_streamed_async_cursor_provider_host_route(
    tmp_path: Path,
    acs_slots: int,
) -> None:
    provider, state = _provider_harness(
        {},
        bounded_projection_cursor_support=True,
        cursor_tensor_fields=True,
    )
    request = _tick_request(
        total_ticks=1121,
        checkpoints=(1120, 1121),
        restore_tick=None,
        acs_checkpoint_slots=acs_slots,
        acs_capture_instrumentation=True,
        native_streamed_checkpoint=True,
    )
    result = provider.execute_batch(
        _tick_native_acs_plan(acs_slots),
        request,
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    execution = result["result"]
    telemetry = execution["acs_checkpoint_staging"]
    assert telemetry["slots"] == acs_slots
    assert telemetry["durable_checkpoint_frontier"] == 1121
    assert telemetry["frontier_lag"] == 0
    assert telemetry["failure_latched"] is False
    assert telemetry["closed"] is True
    assert len(state["runner_ids"]) == 3
    assert execution["resident_owner_preparations"] == 1
    assert len(execution["resident_owner_epoch_transfer_receipts"]) == 1
    assert len(execution["acs_capture_measurements"]) == 2
    assert all(
        isinstance(measurement["cursor_receipt"], Mapping)
        for measurement in execution["acs_capture_measurements"]
    )
    assert all(
        measurement["cursor_receipt"]["max_live_tensor_bearing_projections"] == 1
        for measurement in execution["acs_capture_measurements"]
    )
    assert all(
        measurement[
            "successor_last_useful_kernel_completion_observed_host_time_ns"
        ]
        == 1001
        and measurement["successor_final_stream_synchronization_time_ns"] == 41
        and measurement["successor_final_host_synchronizations"] == 1
        for measurement in execution["acs_capture_measurements"][:-1]
    )
    assert (
        "successor_last_useful_kernel_completion_observed_host_time_ns"
        not in execution["acs_capture_measurements"][-1]
    )


def test_tick_native_5000_ticks_transfers_five_logical_epochs_on_one_physical_owner(
    tmp_path: Path,
) -> None:
    provider, state = _provider_harness({}, track_owner_release=True)

    result = provider.execute_batch(
        _plan(),
        _tick_request(
            total_ticks=5000,
            checkpoints=(2500, 5000),
            restore_tick=None,
        ),
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    execution = result["result"]
    assert execution["terminal_next_absolute_tick"] == 5000
    assert execution["resident_owner_preparations"] == 1
    assert execution["resident_owner_max_simultaneous"] == 1
    assert len(execution["resident_owner_release_receipts"]) == 1
    assert all(
        receipt["live_alias_count"] == 0
        and receipt["release_synchronization_complete"] is True
        for receipt in execution["resident_owner_release_receipts"]
    )
    # Dispatch settlement follows four completed 1,120-tick owner epochs. The
    # bounded 160-tick source capacity is reused inside native dispatch and no
    # longer creates Python-visible settlement boundaries.
    assert len(execution["compact_segment_settlements"]) == 4
    assert len(state["projected_batches"]) == 2
    assert state["prior_owner_alive_before_prepare"] == [[]]
    transfers = execution["resident_owner_epoch_transfer_receipts"]
    assert [row["boundary_next_absolute_tick"] for row in transfers] == [
        1120,
        2240,
        3360,
        4480,
    ]
    assert all(
        row["physical_owner_reused"] is True
        and row["cuda_allocation_recreated"] is False
        and row["simultaneous_physical_owner_count"] == 1
        for row in transfers
    )
    assert len({row["physical_owner_identity"] for row in transfers}) == 1
    owner_ids = [row["prior_owner_identity"] for row in transfers] + [
        execution["resident_owner_release_receipts"][0]["owner_identity"]
    ]
    assert len(set(owner_ids)) == 5
    native_owner_ids = [
        receipt["owner_identity"] for receipt in execution["native_execution_receipts"]
    ]
    # The checkpoint at tick 2,500 splits only the third owner epoch; it does
    # not expose the internal 160-tick source-capacity windows.
    assert native_owner_ids == [
        owner_ids[0],
        owner_ids[1],
        owner_ids[2],
        owner_ids[2],
        owner_ids[3],
        owner_ids[4],
    ]


def test_tick_native_final_adjacent_checkpoints_reuse_the_same_owner(
    tmp_path: Path,
) -> None:
    """Cover the Gate 21 checkpoint/capture/re-entry seam before tick 3360."""

    provider, state = _provider_harness({}, track_owner_release=True)

    result = provider.execute_batch(
        _plan(),
        _tick_request(
            total_ticks=3360,
            checkpoints=(3358, 3359, 3360),
            restore_tick=None,
        ),
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    execution = result["result"]
    receipts = execution["native_execution_receipts"]
    assert [receipt["start_next_tick"] for receipt in receipts] == [
        0,
        1120,
        2240,
        3358,
        3359,
    ]
    assert [receipt["end_next_tick"] for receipt in receipts] == [
        1120,
        2240,
        3358,
        3359,
        3360,
    ]
    assert all(receipt["time_authority"] == "absolute_tick_v1" for receipt in receipts)
    assert [receipt["first_executed_absolute_tick"] for receipt in receipts[-2:]] == [
        3358,
        3359,
    ]
    assert [receipt["first_committed_next_tick"] for receipt in receipts[-2:]] == [
        3359,
        3360,
    ]
    assert len({receipt["owner_identity"] for receipt in receipts[-3:]}) == 1
    assert execution["resident_owner_preparations"] == 1
    assert execution["resident_owner_max_simultaneous"] == 1
    assert state["prior_owner_alive_before_prepare"] == [[]]


def test_terminal_snapshot_merger_runs_across_disk_segments(tmp_path: Path) -> None:
    provider, _state = _provider_harness({})
    calls: list[tuple[list[int], list[int]]] = []

    def merge(prior, projected):
        calls.append(
            (
                [int(row["generation"]) for row in prior],
                [int(row["generation"]) for row in projected],
            )
        )
        return projected

    provider.terminal_snapshot_metrics = True
    provider._resolved_execution_adapters = lambda _plan: {
        "terminal_snapshot_metrics_merger": merge
    }

    result = provider.execute_batch(_plan(), _request(), _Context(tmp_path))

    assert result["status"] == "completed"
    assert calls == [([14, 14, 14], [37, 37, 37])]
    assert [row["generation"] for row in result["metrics_rows"]] == [37, 37, 37]


def test_reporting_disabled_skips_required_generation_series_projection(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({})
    projected = False

    def project_generation_series(**_kwargs: Any) -> list[dict[str, Any]]:
        nonlocal projected
        projected = True
        raise AssertionError("reporting-disabled execution projected reporting rows")

    provider.generation_series_required = True
    provider.project_generation_series = project_generation_series  # type: ignore[attr-defined]

    result = provider.execute_batch(
        _plan(),
        replace(_request(), reporting_enabled=False),
        _Context(tmp_path),
    )

    assert projected is False
    assert result["metrics_rows"] == []


def test_reporting_disabled_scientific_full_does_not_require_chronicle_segments(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({})

    result = provider.execute_batch(
        _scientific_plan(),
        replace(_request(), reporting_enabled=False),
        _Context(tmp_path),
    )

    assert result["status"] == "completed"
    assert result["metrics_rows"] == []
    assert "provider_native_chronicle_required" not in result["result"]
    assert "provider_native_chronicle" not in result["result"]


def test_provider_publishes_only_explicit_native_cuda_no_fallback(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness(
        {},
        explicit_native_no_fallback=True,
    )

    result = provider.execute_batch(_plan(), _request(), _Context(tmp_path))

    assert result["result"]["execution_device"] == "cuda:0"
    assert result["result"]["fallback_used"] is False
    assert all(
        row["explicit_native_no_fallback"] is True
        for row in result["result"]["native_execution_receipts"]
    )


def test_provider_publishes_explicit_cpu_native_no_fallback_for_cpu_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_production_control_readback_stubs(monkeypatch)
    provider, _state = _provider_harness(
        {},
        explicit_cpu_native_no_fallback=True,
    )
    # The CPU-native owner is deliberately not a CUDA direct-resident ABI
    # wrapper.  Production control readbacks remain available, but the CUDA
    # authoritative-state ABI observation must stay disabled for this route.
    provider.authoritative_state_schema_required_v1 = False

    request = replace(
        _request(),
        profile_id=PRODUCTION_CPU_PROFILE_ID,
        cell_ids=("cell-0",),
    )
    result = provider.execute_batch(_plan(), request, _Context(tmp_path))

    assert result["result"]["execution_device"] == "cpu"
    assert result["result"]["fallback_used"] is False
    assert all(
        row["explicit_native_no_fallback"] is True
        for row in result["result"]["native_execution_receipts"]
    )
    # CPU-native owners are not CUDA direct-resident ABI wrappers.  Do not
    # manufacture CUDA-style control-readback authority from their checkpoint
    # projections; the CPU checkpoint route remains explicitly bounded to its
    # own persistence/continuation contract.
    assert result["result"]["production_control_readbacks"] == []
    assert all(
        "production_control_readback_status" not in row
        for row in result["metrics_rows"]
    )
    context = _Context(tmp_path / "second")
    second = provider.execute_batch(_plan(), request, context)
    assert "production_control_readback" not in context.records[-1]
    assert second["result"]["production_control_readbacks"] == []


def test_provider_preserves_bounded_gaussian_segment_receipts(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness(
        {},
        explicit_native_no_fallback=True,
        direct_result_overrides={
            "native_host_gaussian_segment_refill_used": True,
            "gaussian_segment_refill_calls_inside_tick_loop": 2,
            "gaussian_segment_refill_receipts": [
                {
                    "segment_schema": "host_prepared_exact_gaussian_segment_v1",
                    "segment_boundary": "native_tick_boundary_v1",
                    "generation": 0,
                    "completed_tick": 7,
                    "next_tick": 8,
                    "host_to_device_operations_lower_bound": 3,
                    "device_to_host_operations_lower_bound": 6,
                }
            ],
            "gaussian_segment_device_to_host_operations_lower_bound": 12,
            "gaussian_segment_host_to_device_operations_lower_bound": 6,
            "intermediate_device_to_host_operations": 12,
            "intermediate_host_to_device_operations": 6,
        },
    )

    result = provider.execute_batch(_plan(), _request(), _Context(tmp_path))

    receipt = result["result"]["native_execution_receipts"][0]
    assert receipt["native_host_gaussian_segment_refill_used"] is True
    assert receipt["gaussian_segment_refill_calls_inside_tick_loop"] == 2
    assert receipt["gaussian_segment_refill_receipts"] == [
        {
            "segment_schema": "host_prepared_exact_gaussian_segment_v1",
            "segment_boundary": "native_tick_boundary_v1",
            "generation": 0,
            "completed_tick": 7,
            "next_tick": 8,
            "host_to_device_operations_lower_bound": 3,
            "device_to_host_operations_lower_bound": 6,
        }
    ]
    assert receipt["gaussian_segment_device_to_host_operations_lower_bound"] == 12
    assert receipt["gaussian_segment_host_to_device_operations_lower_bound"] == 6
    assert receipt["intermediate_device_to_host_operations"] == 12
    assert receipt["intermediate_host_to_device_operations"] == 6


def test_provider_routes_generation_zero_through_source_factory_seam(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    source_generation_extents: list[int] = []

    def prepare(_plan, request, _context, **kwargs):
        calls.append(request.cell_ids)
        source_generation_extents.append(kwargs["segment_generations"])
        return tuple(
            {
                "metadata": {
                    "generation": 0,
                    "policy_weight_mutation_enable": False,
                    "policy_weight_mutation_mode": "dense",
                    "policy_weight_mutation_rate": 0.0,
                    "policy_weight_mutation_std": 0.0,
                }
            }
            for _cell_id in request.cell_ids
        )

    provider, state = _provider_harness({}, source_preparer=prepare)

    result = provider.execute_batch(_plan(), _request(), _Context(tmp_path))

    assert result["status"] == "completed"
    assert calls == [("cell-0", "cell-1", "cell-2")]
    assert source_generation_extents == [37]
    assert [row["metadata"]["generation"] for row in state["request_sources"][:3]] == [
        0,
        0,
        0,
    ]


def test_resumed_segment_promotes_ragged_event_capacity_to_batch_max(
    tmp_path: Path,
) -> None:
    calls: list[tuple[int | None, int]] = []
    prepared_requests: list[tuple[Any, ...]] = []
    module = object()
    source_projections = tuple(
        {
            "metadata": {
                "generation": 5,
                "event_tape_active_count": active_rows,
                "policy_weight_mutation_enable": False,
                "policy_weight_mutation_mode": "dense",
                "policy_weight_mutation_rate": 0.0,
                "policy_weight_mutation_std": 0.0,
            }
        }
        for active_rows in (900, 912, 906)
    )

    def runner_builder(_cell: Mapping[str, Any], _run_dir: Path, **_kwargs: Any) -> Any:
        return SimpleNamespace(runner=object(), receipt={})

    def request_builder(_runner: Any, **kwargs: Any) -> Any:
        source = kwargs["source_checkpoint_projection"]
        active_rows = int(source["metadata"]["event_tape_active_count"])
        required = active_rows + 1_338
        override = kwargs.get("event_capacity_override")
        event_capacity = required if override is None else override
        calls.append((override, event_capacity))
        return SimpleNamespace(
            # Integrated WDL/Jungle request adapters expose the source under
            # this wrapper field and construct the direct request inside the
            # preparer.
            source_checkpoint_projection=source,
            cuda_capability={"qualified": True},
            cuda_device="cuda:0",
            event_capacity=event_capacity,
        )

    def preparer(**kwargs: Any) -> Any:
        requests = tuple(kwargs["per_cell_requests"])
        prepared_requests.append(requests)
        return SimpleNamespace(source_projections=source_projections)

    builds, requests, prepared = _LifecycleHarness._prepare_resident_segment_owner_v1(
        plan=_plan(),
        runner_builder=runner_builder,
        request_builder=request_builder,
        preparer=preparer,
        cells=_plan()["cells"],
        cell_ids=("cell-0", "cell-1", "cell-2"),
        batch_root=tmp_path,
        start_generation=5,
        end_generation=8,
        ticks_per_generation=80,
        artifact="/qualified/controller.so",
        qualification="/qualified/qualification.json",
        reporting="/qualified/reporting.json",
        module=module,
        source_projections=source_projections,
        allocation_generations=30,
    )

    assert len(builds) == 3
    assert prepared.source_projections == source_projections
    assert tuple(request.event_capacity for request in requests) == (2_250,) * 3
    assert (
        tuple(request.event_capacity for request in prepared_requests[0])
        == (2_250,) * 3
    )
    assert calls == [
        (None, 2_238),
        (None, 2_250),
        (None, 2_244),
        (2_250, 2_250),
        (2_250, 2_250),
        (2_250, 2_250),
    ]


def test_generation_zero_typed_projection_materializes_event_cursors() -> None:
    projection = SimpleNamespace(
        schema_version="typed-g0-v1",
        metadata={"generation": 0, "event_tape_active_count": 3},
        tensors={"state": object()},
        pending_generation_free_agent_ids=(7,),
        pending_generation_free_slots=(2,),
    )

    payload = _generation_zero_projection_payload_v1(projection)

    assert payload["event_cursor"] == 3
    assert payload["runtime_event_cursor"] == 3
    assert payload["pending_generation_free_agent_ids"] == [7]
    assert payload["pending_generation_free_slots"] == [2]
    assert payload["restore_count"] == 0
    assert payload["generation_calls"] == 0
    assert payload["last_applied_actions"] == 0
    assert payload["last_world"] == "none"
    assert "tensor_manifest" in payload


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("event_cursor", True),
        ("event_cursor", -1),
        ("runtime_event_cursor", 2),
    ],
)
def test_generation_zero_typed_projection_rejects_invalid_event_cursors(
    attribute: str,
    value: object,
) -> None:
    projection = SimpleNamespace(
        schema_version="typed-g0-v1",
        metadata={"generation": 0, "event_tape_active_count": 3},
        tensors={},
        event_cursor=3,
        runtime_event_cursor=3,
    )
    setattr(projection, attribute, value)

    with pytest.raises(CampaignProviderError, match="event cursors"):
        _generation_zero_projection_payload_v1(projection)


def test_declared_g0_is_a_durable_bound_checkpoint_catalog_entry(
    tmp_path: Path,
) -> None:
    provider, state = _provider_harness({})
    request = _request(milestones=(0, 14, 37))

    result = provider.execute_batch(_plan(), request, _Context(tmp_path))

    catalog = result["result"]["checkpoint_catalog"]
    assert list(catalog) == ["0", "14", "37"]
    assert catalog["0"]["generation"] == 0
    assert catalog["0"]["checkpoint_binding"] == provider._checkpoint_binding(
        _plan(), request
    )
    assert [row["cell_id"] for row in catalog["0"]["cell_bundles"]] == list(
        request.cell_ids
    )
    assert {
        row["owner_epoch"]
        for row in state["checkpoint_identities"]
        if row["snapshot_sequence"] == 0
    } == {1}
    initial_observers = [
        row
        for row in state["checkpoint_observer_continuations"]
        if row["checkpoint_next_generation"] == 0
    ]
    assert len(initial_observers) == len(request.cell_ids)
    assert all(
        row["behavior_completed_generation_and_row_key"]
        == {"completed_generation": -1, "last_row_key": None}
        for row in initial_observers
    )


def test_declared_g0_is_offered_to_provider_native_chronicle_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    original = ProviderNativeChronicleConsumerV1.observe_generation_zero

    def observe_generation_zero(
        consumer: ProviderNativeChronicleConsumerV1,
        **kwargs: Any,
    ) -> None:
        calls.append(dict(kwargs))
        original(consumer, **kwargs)

    monkeypatch.setattr(
        ProviderNativeChronicleConsumerV1,
        "observe_generation_zero",
        observe_generation_zero,
    )
    provider, _state = _provider_harness(
        {},
        native_segment_artifacts=True,
    )

    provider.execute_batch(
        _scientific_plan(),
        _request(milestones=(0, 14, 37)),
        _Context(tmp_path),
    )

    assert len(calls) == 1
    assert calls[0]["cell_ids"] == ("cell-0", "cell-1", "cell-2")
    assert list(calls[0]["source_projection_sha256_by_cell"]) == [
        "cell-0",
        "cell-1",
        "cell-2",
    ]
    assert len(calls[0]["checkpoint_projections"]) == 3


def test_disk_and_memory_mode_keeps_in_process_projection(tmp_path: Path) -> None:
    provider, state = _provider_harness({})
    context = _Context(tmp_path)
    request = _request()
    request = ResidentBatchRequestV1(
        **{
            **request.payload(),
            "cell_ids": request.cell_ids,
            "checkpoint_milestones": request.checkpoint_milestones,
            "parameter_arrays": request.parameter_arrays,
            "seed_arrays": request.seed_arrays,
            "checkpoint_storage_mode": "disk_and_memory",
        }
    )

    result = provider.execute_batch(_plan(), request, context)

    assert result["status"] == "completed"
    assert state["checkpoint_loads"] == []
    assert context.records[-1]["continuation_storage_mode"] == "disk_and_memory"


def test_resume_loads_strict_bundles_into_fresh_owners_and_preserves_metrics(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, first_state = _provider_harness(store)
    first_context = _Context(tmp_path)
    provider.execute_batch(_plan(), _request(), first_context)
    checkpoint = first_context.records[0]

    resumed_provider, resumed_state = _provider_harness(store)
    resumed_context = _Context(tmp_path)
    result = resumed_provider.resume_batch(
        _plan(), _request(), checkpoint, resumed_context
    )

    assert resumed_state["dispatch_generations"] == [23]
    assert len(resumed_state["runner_ids"]) == 3
    assert set(resumed_state["runner_ids"]).isdisjoint(first_state["runner_ids"])
    assert all(
        source["metadata"]["generation"] == 14
        for source in resumed_state["request_sources"]
    )
    assert result["result"]["resumed_from_generation"] == 14
    assert result["result"]["terminal_generation"] == 37
    assert len(result["metrics_rows"]) == 6
    assert {row["owner_epoch"] for row in resumed_state["checkpoint_identities"]} == {
        37
    }
    assert set(
        row["snapshot_id"] for row in resumed_state["checkpoint_identities"]
    ).isdisjoint(
        row["snapshot_id"]
        for row in first_state["checkpoint_identities"]
        if row["owner_epoch"] == 14
    )


def test_resume_rejects_chronicle_checkpoint_core_substitution_before_load(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    context = _Context(tmp_path / "source")
    provider.execute_batch(_plan(), _request(), context)
    checkpoint = dict(context.records[0])
    checkpoint["provider_native_chronicle_receipt"] = {"placeholder": True}
    checkpoint["provider_native_reproduction_continuity"] = {"placeholder": True}
    checkpoint["provider_native_chronicle_checkpoint_authority_sha256"] = (
        canonical_sha256(context.records[0])
    )
    checkpoint["metrics_rows"] = [
        {**checkpoint["metrics_rows"][0], "births": 999_999},
        *checkpoint["metrics_rows"][1:],
    ]
    resumed, resumed_state = _provider_harness(store)

    with pytest.raises(
        CampaignProviderError,
        match="Chronicle checkpoint authority differs",
    ):
        resumed.resume_batch(
            _plan(),
            _request(),
            checkpoint,
            _Context(tmp_path / "restored"),
        )

    assert resumed_state["checkpoint_loads"] == []


def test_scientific_chronicle_checkpoint_continues_in_derived_fresh_owner(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(
        store,
        native_segment_artifacts=True,
    )
    source_context = _Context(tmp_path / "source")
    source_result = provider.execute_batch(
        _scientific_plan(), _request(), source_context
    )
    checkpoint = source_context.records[0]
    source_exposure = source_result["result"]["provider_native_exposure_receipt"]

    assert set(checkpoint) >= {
        "provider_native_chronicle_receipt",
        "provider_native_reproduction_continuity",
        "provider_native_chronicle_checkpoint_authority_sha256",
    }
    assert source_exposure["segment_indices"] == [0, 1]
    prefix = checkpoint["provider_native_chronicle_receipt"]
    assert prefix["batch_id"] == "batch-0000"
    assert all(row["segment_count"] == 1 for row in prefix["cell_receipts"].values())

    resumed, resumed_state = _provider_harness(
        store,
        native_segment_artifacts=True,
    )
    result = resumed.resume_batch(
        _scientific_plan(),
        _request(),
        checkpoint,
        _Context(tmp_path / "restored"),
    )
    receipt = result["result"]["provider_native_chronicle"]
    expected_owner = (
        "batch-0000-fresh-"
        f"{checkpoint['provider_native_chronicle_checkpoint_authority_sha256'][:12]}"
    )

    assert result["result"]["provider_native_chronicle_required"] is True
    assert receipt["batch_id"] == expected_owner
    assert receipt["batch_id"] != prefix["batch_id"]
    assert resumed_state["dispatch_generations"] == [23]
    exposure = result["result"]["provider_native_exposure_receipt"]
    assert exposure["segment_indices"] == [1]
    assert exposure["current_execution_generation_spans"] == [
        {"start_generation": 14, "end_generation": 37}
    ]
    assert exposure["current_execution_scientific_tick_count"] == 23 * 80 * 3
    assert all(row["segment_count"] == 1 for row in exposure["cell_receipts"].values())
    assert all(
        row["segment_count"] == 2
        and row["continuation_binding"]["source_batch_id"] == "batch-0000"
        and row["continuation_binding"]["restored_batch_id"] == expected_owner
        for row in receipt["cell_receipts"].values()
    )


def test_scientific_chronicle_fails_closed_without_native_segment_artifacts(
    tmp_path: Path,
) -> None:
    provider, _state = _provider_harness({})

    with pytest.raises(
        CampaignProviderError,
        match="scientific_full requires provider-native Chronicle segments",
    ):
        provider.execute_batch(
            _scientific_plan(),
            _request(),
            _Context(tmp_path),
        )


def test_terminal_recovery_retains_scientific_chronicle_receipt(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(
        store,
        native_segment_artifacts=True,
    )
    source_context = _Context(tmp_path / "source")
    provider.execute_batch(_scientific_plan(), _request(), source_context)
    terminal = source_context.records[-1]
    recovered, recovered_state = _provider_harness(
        store,
        native_segment_artifacts=True,
    )

    result = recovered.resume_batch(
        _scientific_plan(),
        _request(),
        terminal,
        _Context(tmp_path / "recovered"),
    )

    assert result["result"]["terminal_checkpoint_recovered"] is True
    assert result["result"]["provider_native_chronicle_required"] is True
    assert (
        result["result"]["provider_native_chronicle"]
        == terminal["provider_native_chronicle_receipt"]
    )
    assert recovered_state["dispatch_generations"] == []


def test_resume_rejects_checkpoint_cell_reordering(tmp_path: Path) -> None:
    provider, _state = _provider_harness({})
    checkpoint = {
        "schema_version": provider.checkpoint_schema_version,
        "checkpoint_binding": provider._checkpoint_binding(_plan(), _request()),
        "generation": 14,
        "cell_bundles": [
            {"cell_id": "cell-1", "path": "one"},
            {"cell_id": "cell-0", "path": "zero"},
            {"cell_id": "cell-2", "path": "two"},
        ],
        "metrics_rows": [],
    }

    with pytest.raises(CampaignProviderError, match="cell order"):
        provider.resume_batch(_plan(), _request(), checkpoint, _Context(tmp_path))


def test_terminal_checkpoint_recovery_never_redispatches(tmp_path: Path) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, state = _provider_harness(store)
    first_context = _Context(tmp_path)
    provider.execute_batch(_plan(), _request(), first_context)
    terminal = first_context.records[-1]

    recovered_provider, recovered_state = _provider_harness(store)
    result = recovered_provider.resume_batch(
        _plan(), _request(), terminal, _Context(tmp_path)
    )

    assert state["dispatch_generations"] == [14, 23]
    assert recovered_state["dispatch_generations"] == []
    assert len(recovered_state["checkpoint_loads"]) == 3
    assert result["status"] == "completed"
    assert result["result"]["terminal_checkpoint_recovered"] is True
    assert (
        result["result"]["fresh_owner_restore_receipt"]["projection_bundles_reloaded"]
        is True
    )
    assert len(result["metrics_rows"]) == 6


def test_fresh_owner_checkpoint_rebind_preserves_source_authority(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    source_request = _request()
    context = _Context(tmp_path / "source")
    provider.execute_batch(_plan(), source_request, context)
    source_checkpoint = dict(context.records[0])
    source_checkpoint["terminal_cell_digests"] = dict(
        reversed(list(source_checkpoint["terminal_cell_digests"].items()))
    )
    restore_request = replace(source_request, batch_id="fresh-owner-batch")

    rebound = provider.prepare_fresh_owner_restore_checkpoint(
        _plan(),
        source_request,
        restore_request,
        source_checkpoint,
    )
    resumed, resumed_state = _provider_harness(store)
    result = resumed.resume_batch(
        _plan(),
        restore_request,
        rebound,
        _Context(tmp_path / "restore"),
    )

    assert len(resumed_state["checkpoint_loads"]) == 3
    assert rebound["checkpoint_binding"] == provider._checkpoint_binding(
        _plan(), restore_request
    )
    rebind = rebound["fresh_owner_checkpoint_rebind"]
    assert rebind["schema_version"] == FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION
    assert rebind["source_checkpoint_binding"] == provider._checkpoint_binding(
        _plan(), source_request
    )
    receipt = result["result"]["fresh_owner_restore_receipt"]
    assert receipt["schema_version"] == FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION
    assert receipt["source_batch_id"] == source_request.batch_id
    assert receipt["restored_batch_id"] == restore_request.batch_id
    assert receipt["source_checkpoint_generation"] == 14
    assert all(
        row["source_projection_sha256"] == row["loaded_projection_sha256"]
        for row in receipt["reload_receipts"]
    )


@pytest.mark.parametrize("terminal", [False, True])
def test_resume_rejects_checkpoint_from_another_plan_before_bundle_load(
    tmp_path: Path, terminal: bool
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    context = _Context(tmp_path)
    provider.execute_batch(_plan(), _request(), context)
    checkpoint = context.records[-1 if terminal else 0]
    changed_plan = _plan()
    changed_plan["plan_sha256"] = "9" * 64

    resumed, resumed_state = _provider_harness(store)
    with pytest.raises(CampaignProviderError, match="checkpoint binding mismatch"):
        resumed.resume_batch(
            changed_plan, _request(), checkpoint, _Context(tmp_path / "resume")
        )

    assert resumed_state["checkpoint_loads"] == []
    assert resumed_state["dispatch_generations"] == []


def test_resume_refuses_missing_strict_run_contract_binding_before_bundle_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    context = _Context(tmp_path / "source")
    provider.execute_batch(_plan(), _request(), context)
    checkpoint = context.records[0]
    legacy_binding = provider._checkpoint_binding(_plan(), _request())
    strict_binding = {
        **legacy_binding,
        "run_contract": {
            "schema_version": "eiel_run_contract_checkpoint_binding_v1",
            "profile_id": "eiel_full_evolution_v1",
            "run_contract_hash": "sha256:" + "a" * 64,
            "required_capabilities": ["patch"],
            "capability_contracts": {"patch": {}},
        },
    }
    monkeypatch.setattr(provider, "_checkpoint_binding", lambda *_args: strict_binding)
    resumed, resumed_state = _provider_harness(store)
    monkeypatch.setattr(resumed, "_checkpoint_binding", lambda *_args: strict_binding)

    with pytest.raises(
        CampaignProviderError, match="run-contract checkpoint binding is absent"
    ):
        resumed.resume_batch(
            _plan(), _request(), checkpoint, _Context(tmp_path / "resume")
        )

    assert resumed_state["checkpoint_loads"] == []
    assert resumed_state["dispatch_generations"] == []


def test_resume_refuses_changed_strict_run_contract_binding_before_bundle_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    context = _Context(tmp_path / "source")
    provider.execute_batch(_plan(), _request(), context)
    checkpoint = dict(context.records[0])
    checkpoint["checkpoint_binding"] = {
        **checkpoint["checkpoint_binding"],
        "run_contract": {"run_contract_hash": "sha256:" + "b" * 64},
    }
    expected = {
        **provider._checkpoint_binding(_plan(), _request()),
        "run_contract": {"run_contract_hash": "sha256:" + "a" * 64},
    }
    resumed, resumed_state = _provider_harness(store)
    monkeypatch.setattr(resumed, "_checkpoint_binding", lambda *_args: expected)

    with pytest.raises(
        CampaignProviderError, match="run-contract checkpoint binding mismatch"
    ):
        resumed.resume_batch(
            _plan(), _request(), checkpoint, _Context(tmp_path / "resume")
        )

    assert resumed_state["checkpoint_loads"] == []
    assert resumed_state["dispatch_generations"] == []


def test_legacy_missing_storage_mode_is_disk_compatible(tmp_path: Path) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    context = _Context(tmp_path)
    provider.execute_batch(_plan(), _request(), context)
    legacy = dict(context.records[0])
    legacy.pop("continuation_storage_mode")

    resumed, resumed_state = _provider_harness(store)
    result = resumed.resume_batch(_plan(), _request(), legacy, _Context(tmp_path))

    assert resumed_state["dispatch_generations"] == [23]
    assert result["result"]["resumed_from_generation"] == 14


def test_resume_rejects_storage_mode_mismatch_before_bundle_load(
    tmp_path: Path,
) -> None:
    store: dict[str, Mapping[str, Any]] = {}
    provider, _state = _provider_harness(store)
    context = _Context(tmp_path)
    provider.execute_batch(_plan(), _request(), context)

    resumed, resumed_state = _provider_harness(store)
    with pytest.raises(CampaignProviderError, match="storage mode does not match"):
        resumed.resume_batch(
            _plan(),
            replace(_request(), checkpoint_storage_mode="disk_and_memory"),
            context.records[0],
            _Context(tmp_path),
        )

    assert resumed_state["checkpoint_loads"] == []


def test_checkpoint_observer_uses_nonzero_projection_lifecycle_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from engine.native_runtime import lifecycle_events

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        lifecycle_events,
        "typed_lifecycle_events_from_projection",
        lambda _projection: ("event",),
    )
    monkeypatch.setattr(
        lifecycle_events,
        "validated_typed_lifecycle_event_cursor",
        lambda _projection, _events: 9,
    )

    def writer(_projection: Mapping[str, Any], path: str, **kwargs: Any) -> str:
        captured.update(kwargs)
        return path

    provider = _LifecycleHarness(
        checkpoint_writer=writer,
        checkpoint_digest=lambda _projection: "a" * 64,
    )
    provider._write_projection_bundle(
        {"metadata": {"generation": 7}},
        tmp_path / "bundle_gen000007",
        generation=7,
        batch_id="batch-0000",
        cell_id="cell-0",
        segment_ordinal=1,
    )

    observer = captured["observer_continuation"]
    assert observer["typed_lifecycle_event_cursor"] == 9
    assert observer["acknowledged_channel_cursor_ranges"]["typed_lifecycle"] == [0, 9]
    identity = captured["architecture_b_identity"]
    assert identity["snapshot_id"] == (
        f"{provider.provider_id}-cell-0:owner:7:boundary:7"
    )
    assert observer["exact_reporting_boundary_ordinal"] == 7


def test_patch_reporting_parity_is_precisely_unsupported_without_sentinel() -> None:
    evidence = _LifecycleHarness().reporting_parity_evidence(_plan())

    assert evidence["status"] == "unsupported"
    assert evidence["observational_contract"] == (
        "exact_reporting_must_not_mutate_authoritative_state_v1"
    )
    assert evidence["state_mutation_sentinel"] is None
    assert evidence["reason_code"] == "reporting_state_mutation_sentinel_unavailable"
