from __future__ import annotations

from typing import Any

import pytest

from engine.campaigns.science_benchmark_clocks_v1 import (
    ScienceBenchmarkClockError,
    derive_science_benchmark_clocks_v1,
    extrapolate_horizon_seconds_v1,
    render_science_benchmark_report_v1,
    terminal_acs_restore_validity_v1,
)


def _capture(**overrides: Any) -> dict[str, Any]:
    row = {
        "schema_version": "eiel_acs_capture_barrier_measurement_v1",
        "measurement_scope": (
            "native_final_sync_plus_provider_projector_settlement_"
            "postprocessing_and_digest_bound_detached_host_snapshot"
        ),
        "limitations": [
            "post-sync host capture time is a GPU-idle proxy; external utilization telemetry remains required for ACS0 decision evidence"
        ],
        "host_capture_freeze_time_ns": 2_000_000,
        "native_final_stream_synchronization_time_ns": 7_000,
        "captured_bytes": 12_000,
        "captured_cell_count": 1,
        "capture_completed_unix_ns": 1_000_000_000_000 + 300_002_000_000,
        "gpu_resources_before": {
            "memory_allocated_bytes": 200_000_000,
            "memory_reserved_bytes": 220_000_000,
            "max_memory_allocated_bytes": 230_000_000,
        },
        "gpu_resources_after": {
            "memory_allocated_bytes": 200_000_000,
            "memory_reserved_bytes": 220_000_000,
            "max_memory_allocated_bytes": 230_000_000,
        },
        "host_resources_before": {"rss_bytes": 1_000_000_000},
        "host_resources_after": {"rss_bytes": 1_200_000_000},
        "immutable_snapshot_digest_sha256s": {"cell-0": "aa" * 32},
        "native_transfer_receipt": {
            "final_device_to_host_operations": 1,
            "final_transfer_bytes": 0,
            "final_host_synchronizations": 1,
            "compact_settlement_d2h_operations": 0,
        },
    }
    row.update(overrides)
    return row


def _manifest(
    *,
    decision_shape: str = "A",
    t_sim_ns: int = 300_000_000_000,
    t_acs_ns: int = 12_000_000_000,
    t_release_ns: int = 0,
) -> dict[str, Any]:
    start = 1_000_000_000_000
    stop = start + t_sim_ns
    capture_done = stop + (2_000_000 if decision_shape != "B" else t_release_ns)
    durable = (
        stop + t_acs_ns
        if decision_shape == "A"
        else capture_done + t_acs_ns
        if decision_shape == "B"
        else stop + t_release_ns + t_acs_ns
    )
    capture = _capture(capture_completed_unix_ns=capture_done)
    staging_slots = 1
    if decision_shape == "B":
        capture["gpu_resources_after"] = {
            "memory_allocated_bytes": 210_000_000,
            "memory_reserved_bytes": 230_000_000,
            "max_memory_allocated_bytes": 240_000_000,
        }
        capture["host_capture_freeze_time_ns"] = t_release_ns
        capture["measurement_scope"] = "gpu_capture_d2h"
        capture["limitations"] = []
    if decision_shape == "C":
        capture["measurement_scope"] = "unknown"
        capture["limitations"] = []
        capture["host_capture_freeze_time_ns"] = 0
        capture["native_transfer_receipt"]["final_device_to_host_operations"] = 0
        capture.pop("capture_completed_unix_ns", None)
        staging_slots = 0
        durable = stop + t_release_ns + t_acs_ns
    return {
        "result": {
            "native_execution_receipts": [
                {
                    "segment_ordinal": 0,
                    "start_next_tick": 0,
                    "end_next_tick": 4960,
                    "segment_dispatch_started_unix_ns": start,
                    "segment_dispatch_finished_unix_ns": start + t_sim_ns - 1_000_000_000,
                    "gaussian_segment_refill_calls_inside_tick_loop": 0,
                    "intermediate_host_to_device_operations": 0,
                    "intermediate_device_to_host_operations": 0,
                    "gaussian_segment_host_to_device_operations_lower_bound": 0,
                    "graph_replays": 0,
                    "graph_recaptures": 0,
                    "hot_loop_host_control_calls": 2,
                    "hot_loop_event_reclaim_copies": 0,
                    "tape_bytes": 0,
                },
                {
                    "segment_ordinal": 1,
                    "start_next_tick": 4960,
                    "end_next_tick": 5000,
                    "segment_dispatch_started_unix_ns": start + t_sim_ns - 1_000_000_000,
                    "segment_dispatch_finished_unix_ns": stop,
                    "gaussian_segment_refill_calls_inside_tick_loop": 0,
                    "intermediate_host_to_device_operations": 0,
                    "intermediate_device_to_host_operations": 0,
                    "gaussian_segment_host_to_device_operations_lower_bound": 0,
                    "graph_replays": 0,
                    "graph_recaptures": 0,
                    "hot_loop_host_control_calls": 1,
                    "hot_loop_event_reclaim_copies": 0,
                    "tape_bytes": 0,
                },
            ],
            "acs_capture_measurements": [capture],
            "acs_catalog_publication_measurements": [
                {"coordinate": 5000, "durable_completed_unix_ns": durable}
            ],
            "acs_checkpoint_staging": {
                "slots": staging_slots,
                "writer_busy_time_ns": t_acs_ns,
                "checkpoint_bytes": 4_000_000,
            },
        },
        "checkpoint": {
            "cell_bundles": [{"cell_id": "cell-0", "path": "/tmp/bundle"}],
            "terminal_projection_digest_sha256s": {"cell-0": "aa" * 32},
            "acs_checkpoint_staging": {
                "checkpoint_bytes": 4_000_000,
                "cell_stage_telemetry_seconds": {
                    "cell-0": {
                        "checkpoint_serialize": 1.5,
                        "checkpoint_projection_gzip": 0.8,
                        "checkpoint_hash": 0.3,
                        "checkpoint_write": 0.4,
                        "checkpoint_staged_tree_fsync": 0.1,
                    }
                },
            },
        },
    }


def test_three_clocks_are_non_overlapping_and_exclude_acs_from_t_sim() -> None:
    clocks = derive_science_benchmark_clocks_v1(
        packed_ticks=5000,
        packed_worlds=1,
        batch_manifest=_manifest(),
    )

    assert clocks["T_SIM_seconds"] == pytest.approx(300.0)
    assert clocks["T_RELEASE_seconds"] == pytest.approx(0.0)
    assert clocks["T_ACS_TAIL_seconds"] == pytest.approx(12.0)
    assert clocks["T_JOB_seconds"] == pytest.approx(312.0)
    assert clocks["T_JOB_seconds"] == pytest.approx(
        clocks["T_SIM_seconds"]
        + clocks["T_RELEASE_seconds"]
        + clocks["T_ACS_TAIL_seconds"]
    )
    assert clocks["SCIENCE_SECONDS_PER_1000_TICKS"] == pytest.approx(60.0)
    assert clocks["SCIENCE_TICKS_PER_SECOND"] == pytest.approx(5000 / 300.0)
    assert clocks["SCIENCE_WORLD_TICKS_PER_SECOND"] == pytest.approx(5000 / 300.0)
    assert clocks["GPU_RELEASE_DECISION"] == "A"
    assert clocks["t_job_is_not_tick_throughput"] is True
    assert clocks["non_overlapping"] is True
    assert clocks["hot_loop_host_control_calls"] == 3
    assert clocks["terminal_acs"]["serialization_seconds"] == pytest.approx(1.5)
    estimates = clocks["estimates_from_t_sim"]
    assert estimates["ticks_10000_seconds"] == pytest.approx(600.0)
    assert estimates["terminal_acs_separate_seconds"] == pytest.approx(12.0)
    assert extrapolate_horizon_seconds_v1(clocks, packed_ticks=100_000) == pytest.approx(
        6012.0
    )


def test_gpu_release_b_uses_only_gpu_dependent_tail() -> None:
    clocks = derive_science_benchmark_clocks_v1(
        packed_ticks=5000,
        packed_worlds=1,
        batch_manifest=_manifest(decision_shape="B", t_release_ns=50_000_000),
    )

    assert clocks["GPU_RELEASE_DECISION"] == "B"
    assert clocks["T_RELEASE_seconds"] == pytest.approx(0.05)
    assert clocks["T_SIM_seconds"] == pytest.approx(300.0)
    assert clocks["T_ACS_TAIL_seconds"] == pytest.approx(12.0)
    assert clocks["T_JOB_seconds"] == pytest.approx(312.05)


def test_bounded_capture_uses_explicit_gpu_safe_release_boundary() -> None:
    manifest = _manifest(decision_shape="B", t_release_ns=50_000_000)
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture["schema_version"] = "eiel_acs_capture_barrier_measurement_v1"
    capture["capture_completed_unix_ns"] -= 20_000_000
    capture["gpu_safely_releasable_unix_ns"] = (
        capture["capture_completed_unix_ns"] + 20_000_000
    )
    clocks = derive_science_benchmark_clocks_v1(
        packed_ticks=5000,
        packed_worlds=1,
        batch_manifest=manifest,
    )

    assert clocks["GPU_RELEASE_DECISION"] == "B"
    assert clocks["T_RELEASE_seconds"] == pytest.approx(0.05)
    assert clocks["gpu_safely_releasable_unix_ns"] == (
        capture["gpu_safely_releasable_unix_ns"]
    )


def test_gpu_utilization_excludes_post_simulation_acs_tail() -> None:
    manifest = _manifest(decision_shape="B", t_release_ns=2_000_000_000)
    start = manifest["result"]["native_execution_receipts"][0][
        "segment_dispatch_started_unix_ns"
    ]
    capture = manifest["result"]["acs_capture_measurements"][0]
    capture["gpu_safely_releasable_unix_ns"] = start + 302_000_000_000
    clocks = derive_science_benchmark_clocks_v1(
        packed_ticks=5000,
        packed_worlds=136,
        batch_manifest=manifest,
        gpu_samples=[
            {"observed_unix_ns": start + 1_000_000_000, "sm_percent": 90.0},
            {"observed_unix_ns": start + 200_000_000_000, "sm_percent": 70.0},
            {"observed_unix_ns": start + 303_000_000_000, "sm_percent": 0.0},
            {"observed_unix_ns": start + 310_000_000_000, "sm_percent": 0.0},
        ],
    )

    assert clocks["gpu_utilization"] == {
        "mean_percent": pytest.approx(80.0),
        "maximum_percent": pytest.approx(90.0),
        "sample_count": 2,
    }
    assert clocks["gpu_performance_window"][
        "post_simulation_acs_tail_excluded"
    ] is True
    assert clocks["post_simulation_acs_tail_gpu_activity"]["gpu_utilization"] == {
        "mean_percent": pytest.approx(0.0),
        "maximum_percent": pytest.approx(0.0),
        "sample_count": 2,
    }
    assert clocks["post_simulation_acs_tail_gpu_activity"][
        "excluded_from_gpu_performance"
    ] is True


def test_gpu_samples_without_authoritative_unix_time_fail_closed() -> None:
    with pytest.raises(ScienceBenchmarkClockError, match="observed_unix_ns"):
        derive_science_benchmark_clocks_v1(
            packed_ticks=5000,
            packed_worlds=136,
            batch_manifest=_manifest(),
            gpu_samples=[{"sm_percent": 5.0}],
        )


def test_gpu_release_c_retains_gpu_until_durable_when_detach_is_unproven() -> None:
    clocks = derive_science_benchmark_clocks_v1(
        packed_ticks=5000,
        packed_worlds=1,
        batch_manifest=_manifest(
            decision_shape="C", t_release_ns=12_000_000_000, t_acs_ns=0
        ),
    )

    assert clocks["GPU_RELEASE_DECISION"] == "C"
    assert clocks["T_RELEASE_seconds"] == pytest.approx(12.0)
    assert clocks["T_ACS_TAIL_seconds"] == pytest.approx(0.0)
    assert clocks["gpu_release_dependency"]


def test_terminal_acs_restore_validity_requires_digest_bound_bundles() -> None:
    validity = terminal_acs_restore_validity_v1(_manifest())
    assert validity["status"] == "PASS"
    assert validity["snapshot_cells"] == 1
    assert validity["bundle_count"] == 1


def test_report_starts_with_required_identity_blocks() -> None:
    clocks = derive_science_benchmark_clocks_v1(
        packed_ticks=5000,
        packed_worlds=1,
        batch_manifest=_manifest(),
    )
    report = render_science_benchmark_report_v1(
        status="VALID",
        qualified_science_route="EIEL_FULL_EVOLUTION_V1",
        b_w_p_c="B1/W32/P256/C512",
        source_sha="abc",
        artifact_sha="def",
        clocks=clocks,
        restore_validity="PASS",
        interpretation=["terminal ACS excluded from tick-throughput"],
        b140_unused_reason="B140 is not an un-revoked qualified science route",
    )
    assert report.startswith("BENCHMARK STATUS:")
    for token in (
        "QUALIFIED SCIENCE ROUTE:",
        "B/W/P/C:",
        "SOURCE SHA:",
        "ARTIFACT SHA:",
        "SCIENCE PERFORMANCE",
        "ticks: 5000",
        "T_SIM:",
        "TERMINAL GPU RELEASE",
        "GPU_RELEASE_DECISION:",
        "TERMINAL ACS",
        "TOTAL JOB",
        "T_JOB:",
        "INTERPRETATION",
    ):
        assert token in report


def test_missing_dispatch_timestamps_fail_closed() -> None:
    manifest = _manifest()
    del manifest["result"]["native_execution_receipts"][0][
        "segment_dispatch_started_unix_ns"
    ]
    with pytest.raises(ScienceBenchmarkClockError, match="segment dispatch"):
        derive_science_benchmark_clocks_v1(
            packed_ticks=5000, packed_worlds=1, batch_manifest=manifest
        )
