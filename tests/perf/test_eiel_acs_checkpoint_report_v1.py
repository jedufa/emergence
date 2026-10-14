from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from engine.perf.acs_checkpoint_report_v1 import (
    ACS_CADENCES_TICKS_V1,
    ACS_GPU_TELEMETRY_SCHEMA_VERSION,
    ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION,
    ACSCheckpointReportError,
    build_acs0_report_v1,
    build_acs4_report_v1,
    model_acs_cadence_v1,
)
from tools.scripts.eiel_acs_checkpoint_report import main


POST_CAPTURE_STAGES = {
    "source_digest_seconds": 0.01,
    "checkpoint_identity_seconds": 0.01,
    "checkpoint_clone_materialization": 0.5,
    "checkpoint_canonical_pack": 0.7,
    "checkpoint_sidecar_validation": 0.2,
    "checkpoint_serialize": 1.5,
    "checkpoint_hash": 0.3,
    "checkpoint_digest_coherence": 0.4,
    "checkpoint_projection_gzip": 1.0,
    "checkpoint_write": 2.0,
    "checkpoint_seal_write": 0.1,
    "checkpoint_strict_readback": 0.4,
    "checkpoint_staged_tree_fsync": 0.2,
    "checkpoint_atomic_replace": 0.1,
    "checkpoint_total": 4.9,
    "writer_call_seconds": 5.0,
    "outer_total_seconds": 5.02,
    "provider_post_digest_seconds": 0.01,
}


def _manifest(*, slots: int, wall_seconds: int, route: str) -> dict[str, Any]:
    instrumentation = {
        "writer_service_time_ns": 5_000_000_000,
        "checkpoint_bytes": 4_000_000,
        "checkpoint_bytes_measurement_status": "complete",
        "cell_checkpoint_bytes": {
            "cell-0": 2_000_000,
            "cell-1": 2_000_000,
        },
        "cell_stage_telemetry_seconds": {
            "cell-0": dict(POST_CAPTURE_STAGES),
            "cell-1": dict(POST_CAPTURE_STAGES),
        },
    }

    checkpoint = {
        "checkpoint_binding": {
            "artifact_sha256s": {
                "cuda_extension": "7" * 64,
                "qualification_receipt": "8" * 64,
            },
            "qualification_key_sha256": "9" * 64,
        },
        "terminal_projection_digest_sha256s": {
            "cell-0": "1" * 64,
            "cell-1": "2" * 64,
        },
        "canonical_terminal_state_digest_sha256s": {
            "cell-0": "3" * 64,
            "cell-1": "4" * 64,
        },
        "canonical_checkpoint_storage_digest_sha256s": {
            "cell-0": "5" * 64,
            "cell-1": "6" * 64,
        },
        "mechanics_checkpoint_digests": {
            "cell-0": "3" * 64,
            "cell-1": "4" * 64,
        },
        "checkpoint_identity_contract": {
            "q5_terminal_equality_authority": (
                "canonical_terminal_state_digest_sha256s"
            ),
            "route_provenance_may_differ_from_terminal_state": True,
        },
        (
            "acs_checkpoint_staging" if slots else "acs_checkpoint_instrumentation"
        ): instrumentation,
    }
    capture = {
        "checkpoint_projection_time_ns": 700_000_000,
        "snapshot_digest_time_ns": 200_000_000,
        "host_capture_freeze_time_ns": 900_000_000,
        "capture_cpu_time_ns": 800_000_000,
        "native_final_stream_synchronization_time_ns": 100_000_000,
        "last_useful_kernel_completion_observed_host_time_ns": 10_000_000_000,
        "capture_barrier_time_ns": 1_000_000_000,
        "post_sync_gpu_idle_proxy_time_ns": 900_000_000,
        "captured_bytes": 2_000_000,
        "host_resources_before": {
            "rss_bytes": 100,
            "pss_bytes": 90,
            "pinned_bytes": 10,
        },
        "host_resources_after": {
            "rss_bytes": 120,
            "pss_bytes": 110,
            "pinned_bytes": 12,
        },
        "gpu_resources_before": {
            "memory_allocated_bytes": 1_000,
            "memory_reserved_bytes": 2_000,
            "max_memory_allocated_bytes": 3_000,
        },
        "gpu_resources_after": {
            "memory_allocated_bytes": 900,
            "memory_reserved_bytes": 2_000,
            "max_memory_allocated_bytes": 3_000,
        },
        "native_transfer_receipt": {
            "final_host_synchronizations": 1,
            "final_device_to_host_operations": 10,
            "compact_settlement_d2h_operations": 9,
            "final_transfer_bytes": 2_000_000,
        },
        "limitations": ["external GPU utilization is required"],
    }
    return {
        "schema_version": "eiel_campaign_batch_manifest_v1",
        "campaign_id": "campaign-1",
        "batch_id": "batch-0000",
        "plan_sha256": "a" * 64,
        "execution_route_sha256": route,
        "provider_id": "provider",
        "profile_id": "profile",
        "cell_ids": ["cell-0", "cell-1"],
        "created_at": "2026-08-13T00:00:00Z",
        "updated_at": f"2026-08-13T00:00:{wall_seconds:02d}Z",
        "status": "completed",
        "result": {
            "terminal_next_tick": 100,
            "checkpoint_catalog": {"100": checkpoint},
            "acs_capture_measurements": [capture],
            "acs_catalog_publication_measurements": [
                {
                    "coordinate": 100,
                    "total_finalization_time_ns": 200_000_000,
                }
            ],
            **(
                {
                    "acs_checkpoint_staging": {
                        "slots": slots,
                        "queue_occupancy_high_water": slots,
                        "producer_blocked_time_ns": 0,
                        "backlog_age_ns": 0,
                        "compute_frontier": 100,
                        "durable_checkpoint_frontier": 100,
                    }
                }
                if slots
                else {}
            ),
        },
    }


def _gpu_telemetry(manifest: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": ACS_GPU_TELEMETRY_SCHEMA_VERSION,
        "status": "passed",
        "baseline_admitted": True,
        "launched": True,
        "command_exit_code": 0,
        "no_competing_gpu_workloads": True,
        "exclusive_gpu_window": True,
        "process_attributed": False,
        "sample_count": 2,
        "runtime_samples": [
            {"monotonic_time_ns": 10_500_000_000, "sm_percent": 0.0},
            {"monotonic_time_ns": 11_000_000_000, "sm_percent": 10.0},
        ],
        "manifest_sha256": digest,
        "post_sync_boundary_monotonic_ns": 10_000_000_000,
        "post_sync_sample_count": 2,
        "idle_sm_threshold_percent": 5.0,
        "gpu_idle_fraction": 0.5,
    }


def _generation_route(
    manifest: dict[str, Any],
    *,
    terminal_generation: int = 3,
    runner_ticks: tuple[Any, ...] = (160, 160),
) -> dict[str, Any]:
    result = manifest["result"]
    result.pop("terminal_next_tick", None)
    result["terminal_generation"] = terminal_generation
    result["runner_receipts"] = [
        {
            "cell_id": f"cell-{index}",
            "dimensions": {"ticks_per_generation": ticks},
        }
        for index, ticks in enumerate(runner_ticks)
    ]
    return manifest


def test_acs0_report_requires_complete_capture_and_post_capture_denominator() -> None:
    manifest = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    report = build_acs0_report_v1(
        manifest,
        gpu_telemetry_receipt=_gpu_telemetry(manifest),
    )

    assert report["measurement_status"] == "COMPLETE"
    assert report["decision"] == "UNSET_REQUIRES_EVIDENCE_REVIEW"
    assert report["admissible_decisions"] == ["ACS0-A", "ACS0-B", "ACS0-C"]
    assert (
        report["measurements"]["capture"]["capture_barrier_time_ns"]["median"]
        == 1_000_000_000
    )
    assert (
        report["measurements"]["post_capture_stages_seconds"][
            "checkpoint_clone_materialization"
        ]["median"]
        == 0.5
    )
    assert report["measurements"]["checkpoint_bytes"]["median"] == 4_000_000
    assert (
        report["measurements"]["effective_persistence_bytes_per_second"]["median"]
        == 800_000
    )
    assert len(report["report_sha256"]) == 64


def test_acs0_report_fails_closed_when_gpu_or_transfer_authority_is_missing() -> None:
    manifest = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    manifest["result"]["acs_capture_measurements"][0]["native_transfer_receipt"][
        "final_transfer_bytes"
    ] = None

    report = build_acs0_report_v1(
        manifest,
        gpu_telemetry_receipt=_gpu_telemetry(manifest),
    )

    assert report["measurement_status"] == "INCOMPLETE"
    assert report["decision"] == "ACS0-D"
    assert any(
        field.endswith("native_transfer_receipt.final_transfer_bytes")
        for field in report["measurements"]["missing_or_invalid_fields"]
    )


def test_acs0_report_rejects_gpu_receipt_bound_to_another_manifest() -> None:
    manifest = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    receipt = _gpu_telemetry(manifest)
    receipt["manifest_sha256"] = "f" * 64

    report = build_acs0_report_v1(manifest, gpu_telemetry_receipt=receipt)

    assert report["measurement_status"] == "INCOMPLETE"
    assert report["decision"] == "ACS0-D"
    assert report["external_gpu_telemetry"]["status"] == "INVALID"


def test_cadence_model_reports_stability_and_exact_first_backpressure() -> None:
    stable = model_acs_cadence_v1(
        cadence_ticks=1_000,
        horizon_ticks=10_000,
        compute_wall_seconds_per_tick=0.001,
        capture_barrier_seconds=0.0,
        persistence_service_seconds=0.5,
        catalog_publication_seconds=0.0,
        slots=2,
    )
    overloaded = model_acs_cadence_v1(
        cadence_ticks=1_000,
        horizon_ticks=10_000,
        compute_wall_seconds_per_tick=0.001,
        capture_barrier_seconds=0.0,
        persistence_service_seconds=3.0,
        catalog_publication_seconds=0.0,
        slots=2,
    )

    assert stable["service_rate_sustainable"] is True
    assert stable["time_to_backpressure_ticks"] is None
    # Reserve-before-compute briefly owns the next slot while the prior writer
    # finishes, even though the writer drains before a third reservation.
    assert stable["buffer_high_water"] == 2
    assert overloaded["service_rate_sustainable"] is False
    assert overloaded["first_backpressure_checkpoint"] == 3
    assert overloaded["time_to_backpressure_ticks"] == 2_000
    assert overloaded["producer_blocked_seconds"] > 0
    assert overloaded["buffer_high_water"] == 2


def test_acs4_report_compares_primary_metric_identity_and_all_cadences() -> None:
    synchronous = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    acs = _manifest(slots=2, wall_seconds=15, route="b" * 64)
    restore = {
        "schema_version": ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION,
        "status": "passed",
        "semantic_digest_identity": True,
        "restore_wall_seconds": 2.5,
    }

    incomplete = build_acs4_report_v1(synchronous, acs)
    report = build_acs4_report_v1(
        synchronous,
        acs,
        restore_receipt=restore,
        synchronous_gpu_telemetry_receipt=_gpu_telemetry(synchronous),
        acs_gpu_telemetry_receipt=_gpu_telemetry(acs),
    )

    assert incomplete["measurement_status"] == "INCOMPLETE"
    assert incomplete["restore_measurement"]["status"] == "MISSING"
    assert report["measurement_status"] == "COMPLETE"
    assert report["decision"] == "UNSET_REQUIRES_EVIDENCE_REVIEW"
    assert report["measured_comparison"]["semantic_digest_identity"] is True
    assert report["measured_comparison"]["terminal_projection_digest_identity"] is True
    assert report["measured_comparison"]["acs_to_synchronous_rate_ratio"] == (20 / 15)
    assert (
        tuple(row["cadence_ticks"] for row in report["cadence_models"])
        == ACS_CADENCES_TICKS_V1
    )
    assert all(
        "synchronous_tax_seconds" in row
        and "async_writer_utilization" in row
        and "time_to_backpressure_ticks" in row
        and "safe_sustainable_minimum_cadence_ticks" in row
        and "expected_total_wall_seconds" in row
        and "buffer_high_water" in row
        for row in report["cadence_models"]
    )


def test_acs4_report_separates_route_provenance_from_semantic_identity() -> None:
    synchronous = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    acs = _manifest(slots=2, wall_seconds=15, route="b" * 64)
    acs["result"]["checkpoint_catalog"]["100"]["terminal_projection_digest_sha256s"][
        "cell-0"
    ] = "a" * 64

    report = build_acs4_report_v1(
        synchronous,
        acs,
        restore_receipt={
            "schema_version": ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION,
            "status": "passed",
            "semantic_digest_identity": True,
            "restore_wall_seconds": 2.5,
        },
        synchronous_gpu_telemetry_receipt=_gpu_telemetry(synchronous),
        acs_gpu_telemetry_receipt=_gpu_telemetry(acs),
    )

    assert report["measurement_status"] == "COMPLETE"
    assert report["measured_comparison"]["semantic_digest_identity"] is True
    assert report["measured_comparison"]["terminal_projection_digest_identity"] is False


@pytest.mark.parametrize(
    "field",
    [
        "canonical_terminal_state_digest_sha256s",
        "canonical_checkpoint_storage_digest_sha256s",
    ],
)
def test_acs4_report_fails_semantic_identity_on_canonical_digest_drift(
    field: str,
) -> None:
    synchronous = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    acs = _manifest(slots=2, wall_seconds=15, route="b" * 64)
    acs["result"]["checkpoint_catalog"]["100"][field]["cell-0"] = "a" * 64
    if field == "canonical_terminal_state_digest_sha256s":
        acs["result"]["checkpoint_catalog"]["100"]["mechanics_checkpoint_digests"][
            "cell-0"
        ] = "a" * 64

    report = build_acs4_report_v1(
        synchronous,
        acs,
        restore_receipt={
            "schema_version": ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION,
            "status": "passed",
            "semantic_digest_identity": True,
            "restore_wall_seconds": 2.5,
        },
        synchronous_gpu_telemetry_receipt=_gpu_telemetry(synchronous),
        acs_gpu_telemetry_receipt=_gpu_telemetry(acs),
    )

    assert report["measurement_status"] == "INCOMPLETE"
    assert report["measured_comparison"]["semantic_digest_identity"] is False


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_projection", "exact manifest cell set"),
        ("mechanics_drift", "mechanics digests disagree"),
        ("wrong_authority", "lacks canonical terminal authority"),
    ],
)
def test_acs4_report_rejects_malformed_terminal_identity_contract(
    mutation: str,
    match: str,
) -> None:
    synchronous = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    acs = copy.deepcopy(synchronous)
    checkpoint = acs["result"]["checkpoint_catalog"]["100"]
    if mutation == "missing_projection":
        checkpoint.pop("terminal_projection_digest_sha256s")
    elif mutation == "mechanics_drift":
        checkpoint["mechanics_checkpoint_digests"]["cell-0"] = "a" * 64
    else:
        checkpoint["checkpoint_identity_contract"]["q5_terminal_equality_authority"] = (
            "terminal_projection_digest_sha256s"
        )

    with pytest.raises(ACSCheckpointReportError, match=match):
        build_acs4_report_v1(synchronous, acs)


def test_acs4_report_derives_generation_route_terminal_ticks() -> None:
    synchronous = _generation_route(_manifest(slots=0, wall_seconds=20, route="a" * 64))
    acs = _generation_route(_manifest(slots=2, wall_seconds=15, route="b" * 64))
    report = build_acs4_report_v1(
        synchronous,
        acs,
        restore_receipt={
            "schema_version": ACS_RESTORE_MEASUREMENT_SCHEMA_VERSION,
            "status": "passed",
            "semantic_digest_identity": True,
            "restore_wall_seconds": 2.5,
        },
        synchronous_gpu_telemetry_receipt=_gpu_telemetry(synchronous),
        acs_gpu_telemetry_receipt=_gpu_telemetry(acs),
    )

    assert report["measurement_status"] == "COMPLETE"
    assert report["measured_comparison"]["terminal_time_authority"] == (
        "terminal_generation"
    )
    assert report["measured_comparison"]["terminal_generation"] == 3
    assert report["measured_comparison"]["ticks_per_generation"] == 160
    assert report["measured_comparison"]["terminal_ticks"] == 480
    assert report["measured_comparison"]["useful_world_ticks"] == 960


@pytest.mark.parametrize("bad_ticks", [None, True, 160.0, "160", 0, -1])
def test_acs4_report_rejects_non_exact_positive_runner_ticks(
    bad_ticks: Any,
) -> None:
    synchronous = _generation_route(
        _manifest(slots=0, wall_seconds=20, route="a" * 64),
        runner_ticks=(160, bad_ticks),
    )
    acs = _generation_route(_manifest(slots=2, wall_seconds=15, route="b" * 64))

    with pytest.raises(ACSCheckpointReportError, match="ticks_per_generation"):
        build_acs4_report_v1(synchronous, acs)


@pytest.mark.parametrize("bad_receipts", [None, [], {}, [None], [{}]])
def test_acs4_report_rejects_missing_or_malformed_runner_receipts(
    bad_receipts: Any,
) -> None:
    synchronous = _generation_route(_manifest(slots=0, wall_seconds=20, route="a" * 64))
    synchronous["result"]["runner_receipts"] = bad_receipts
    acs = _generation_route(_manifest(slots=2, wall_seconds=15, route="b" * 64))

    with pytest.raises(ACSCheckpointReportError, match="runner_receipts"):
        build_acs4_report_v1(synchronous, acs)


def test_acs4_report_rejects_runner_tick_disagreement() -> None:
    synchronous = _generation_route(
        _manifest(slots=0, wall_seconds=20, route="a" * 64),
        runner_ticks=(160, 80),
    )
    acs = _generation_route(_manifest(slots=2, wall_seconds=15, route="b" * 64))

    with pytest.raises(ACSCheckpointReportError, match="disagree"):
        build_acs4_report_v1(synchronous, acs)


def test_acs4_report_requires_runner_receipts_to_cover_manifest_cells() -> None:
    synchronous = _generation_route(
        _manifest(slots=0, wall_seconds=20, route="a" * 64),
        runner_ticks=(160,),
    )
    acs = _generation_route(_manifest(slots=2, wall_seconds=15, route="b" * 64))

    with pytest.raises(ACSCheckpointReportError, match="exact manifest cell set"):
        build_acs4_report_v1(synchronous, acs)


@pytest.mark.parametrize("bad_generation", [None, True, 3.0, "3", 0, -1])
def test_acs4_report_rejects_invalid_terminal_generation(
    bad_generation: Any,
) -> None:
    synchronous = _generation_route(_manifest(slots=0, wall_seconds=20, route="a" * 64))
    synchronous["result"]["terminal_generation"] = bad_generation
    acs = _generation_route(_manifest(slots=2, wall_seconds=15, route="b" * 64))

    with pytest.raises(ACSCheckpointReportError, match="terminal_generation"):
        build_acs4_report_v1(synchronous, acs)


def test_acs4_report_rejects_mixed_terminal_time_authorities() -> None:
    synchronous = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    acs = _generation_route(_manifest(slots=2, wall_seconds=15, route="b" * 64))

    with pytest.raises(ACSCheckpointReportError, match="mixed"):
        build_acs4_report_v1(synchronous, acs)


def test_acs4_report_requires_exactly_matched_generation_arms() -> None:
    synchronous = _generation_route(
        _manifest(slots=0, wall_seconds=20, route="a" * 64),
        terminal_generation=3,
        runner_ticks=(160, 160),
    )
    acs = _generation_route(
        _manifest(slots=2, wall_seconds=15, route="b" * 64),
        terminal_generation=6,
        runner_ticks=(80, 80),
    )

    with pytest.raises(ACSCheckpointReportError, match="horizons disagree"):
        build_acs4_report_v1(synchronous, acs)


def test_acs4_report_requires_identical_ordered_cell_arms() -> None:
    synchronous = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    acs = _manifest(slots=2, wall_seconds=15, route="b" * 64)
    acs["cell_ids"].reverse()

    with pytest.raises(ACSCheckpointReportError, match="identical ordered cell arms"):
        build_acs4_report_v1(synchronous, acs)


def test_acs4_report_rejects_terminal_next_tick_horizon_disagreement() -> None:
    synchronous = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    acs = _manifest(slots=2, wall_seconds=15, route="b" * 64)
    acs["result"]["terminal_next_tick"] = 101

    with pytest.raises(ACSCheckpointReportError, match="same positive tick horizon"):
        build_acs4_report_v1(synchronous, acs)


def test_acs0_cli_writes_digest_bound_report(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    gpu_path = tmp_path / "gpu.json"
    output_path = tmp_path / "acs0_report.json"
    manifest = _manifest(slots=0, wall_seconds=20, route="a" * 64)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gpu_path.write_text(json.dumps(_gpu_telemetry(manifest)), encoding="utf-8")

    assert (
        main(
            [
                "acs0",
                "--manifest",
                str(manifest_path),
                "--gpu-telemetry-receipt",
                str(gpu_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["measurement_status"] == "COMPLETE"
    assert len(report["report_sha256"]) == 64
