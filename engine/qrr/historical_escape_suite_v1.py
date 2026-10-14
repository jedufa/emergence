"""Compact permanent regressions for costly historical qualification escapes."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Final

from engine.campaigns.canonical import canonical_sha256


HISTORICAL_ESCAPE_SUITE_ID: Final[str] = "HistoricalQualificationEscapeSuiteV1"


@dataclass(frozen=True, slots=True)
class HistoricalEscapeCaseV1:
    escape_id: str
    historical_commits: tuple[str, ...]
    regression_node_id: str
    qrr_surface: str
    historical_evidence: tuple[str, ...] = ()


HISTORICAL_ESCAPE_CASES_V1: Final[tuple[HistoricalEscapeCaseV1, ...]] = (
    HistoricalEscapeCaseV1(
        "genome_v2_request_authority",
        ("856cddfc", "a0efdb7b"),
        "tests/test_eiel_production_engine_native_bindings_v1.py::test_wdl_fixed_request_preserves_genome_v2_source_capability",
        "canonical_execution_request",
    ),
    HistoricalEscapeCaseV1(
        "genome_v2_checkpoint_authority",
        ("856cddfc", "a0efdb7b"),
        "tests/test_eiel_cuda_architecture_b_direct_resident_source_allocator_v1.py::test_genome_v2_checkpoint_authority_binding_is_validated_and_canonical",
        "composite_checkpoint_state",
    ),
    HistoricalEscapeCaseV1(
        "eed_capability_context",
        ("226c2f27", "a8a6cc21"),
        "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_jungle_v1.py::test_combined_eed_owner_propagates_its_capability_to_wdl_validation",
        "capability_envelope",
    ),
    HistoricalEscapeCaseV1(
        "eed_checkpoint_state_loss",
        ("b1f4bfaa",),
        "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_jungle_v1.py::test_combined_eed_checkpoint_injects_terminal_banks_before_patch_settlement",
        "composite_checkpoint_state",
    ),
    HistoricalEscapeCaseV1(
        "jungle_t160_reporting_drift",
        ("2cb0b60a",),
        "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_jungle_v1.py::test_combined_owner_propagates_canonical_wdl_t160_to_jungle",
        "qualified_execution_envelope",
    ),
    HistoricalEscapeCaseV1(
        "cpu_staging_final_batch_shape",
        ("d0f01f83",),
        "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_jungle_v1.py::test_cpu_staging_allocates_wdl_and_jungle_banks_at_final_batch_shape",
        "release_selftest",
    ),
    HistoricalEscapeCaseV1(
        "cumulative_metric_prefix",
        ("e1f751fd", "c43cc127"),
        "tests/test_eiel_eed_opportunity_metrics_v1.py::test_resume_segment_filters_accumulated_prefix_and_merges_once",
        "checkpoint_resume",
    ),
    HistoricalEscapeCaseV1(
        "resource_authority_duplication",
        ("a00f1dfe", "681f0157"),
        "tests/test_eiel_execution_route_preflight_v1.py::test_ordinary_birth_route_rejects_full_resident_capacity",
        "resource_envelope",
    ),
    HistoricalEscapeCaseV1(
        "resume_event_capacity",
        ("bc06efd7", "37bada8c"),
        "tests/test_eiel_campaign_patch_provider_execution_v1.py::test_resumed_segment_promotes_ragged_event_capacity_to_batch_max",
        "checkpoint_resume",
    ),
    HistoricalEscapeCaseV1(
        "stale_artifact_relocation",
        ("d660f76f",),
        "tests/test_eiel_execution_route_preflight_v1.py::test_route_hash_and_artifact_mismatch_fail_closed",
        "qrr_integrity",
    ),
    HistoricalEscapeCaseV1(
        "sidecar_identity_boundary",
        ("ae4dd999", "ec543f87"),
        "tests/test_eiel_cuda_architecture_b_direct_resident_source_allocator_v1.py::test_genome_v2_multicell_checkpoint_sidecars_restore_by_cell_identity",
        "composite_checkpoint_state",
    ),
    HistoricalEscapeCaseV1(
        "authoritative_checkpoint_identity",
        ("ae4dd999", "ec543f87"),
        "tests/test_eiel_authoritative_state_checkpoint_v1.py::test_checkpoint_identity_mismatch_fails_closed_even_outside_strict_mode",
        "checkpoint_resume",
    ),
    HistoricalEscapeCaseV1(
        "eed_metric_snapshot_continuation",
        ("e1f751fd", "c43cc127"),
        "tests/test_eiel_production_engine_native_bindings_v1.py::test_eed_metric_snapshot_merge_preserves_reproductive_prefix",
        "checkpoint_resume",
    ),
    HistoricalEscapeCaseV1(
        "canonical_request_field_preservation",
        (),
        "tests/test_eiel_qrr_v1.py::test_bound_request_derivation_preserves_every_nonderived_field",
        "canonical_execution_request",
    ),
    HistoricalEscapeCaseV1(
        "release_source_contamination",
        (),
        "tests/test_eiel_qrr_v1.py::test_release_module_origin_guard_rejects_current_checkout",
        "canonical_launcher",
    ),
    HistoricalEscapeCaseV1(
        "release_runtime_data_omission",
        (),
        "tests/test_eiel_qrr_v1.py::test_release_payload_includes_route_adjacent_runtime_manifests",
        "qrr_integrity",
    ),
    HistoricalEscapeCaseV1(
        "runability_checkpoint_resume_probe",
        (),
        "tests/test_eiel_qrr_v1.py::test_runability_probe_uses_existing_fresh_owner_restore_gate",
        "checkpoint_resume",
    ),
    HistoricalEscapeCaseV1(
        "mixed_per_cell_experiment_not_representable",
        (),
        "tests/test_eiel_qrr_v1.py::test_representability_rejects_mixed_shared_jungle_eed_controls_before_request_or_resource",
        "experiment_representability",
        (
            public_runtime_path('evidence_raw/b48_g125_adaptive_experiment_20260811/PRE_SCIENCE_QUALIFICATION_CONTRACT_BLOCKER.json'),
            "blocker_receipt_sha256:8d25782b1ec786d8e1dabe481607fb7b519370736a70a5c439a02626286c2fca",
            "experiment_plan_sha256:406bbda2f3f4785352e2989f0f79ce0622e47628e94475378e44e03ed93d4f70",
        ),
    ),
    HistoricalEscapeCaseV1(
        "runability_gate_science_evaluator_mismatch",
        (),
        "tests/test_eiel_qrr_v1.py::test_runability_gate_accepts_exact_runtime_proofs_without_science_claim",
        "checkpoint_resume",
    ),
    HistoricalEscapeCaseV1(
        "absolute_tick_horizon_not_representable",
        (),
        "tests/test_eiel_qrr_v1.py::test_historical_b48_w32_50000_tick_production_request_is_exactly_representable",
        "experiment_representability",
        (
            public_runtime_path('qrr_runs_v1/qrr1-3506018e34bd2ddef45cf3ceed1fb029/production-qrr-test-1-normal-long-run-20260811/production_qrr_test_1_receipt.json'),
            "failure_code:ABSOLUTE_TICK_HORIZON_NOT_REPRESENTABLE",
        ),
    ),
    HistoricalEscapeCaseV1(
        "tick_native_segment_owner_co_residency_oom",
        (),
        "tests/test_eiel_campaign_patch_provider_execution_v1.py::test_tick_native_releases_completed_owner_before_successor_preparation",
        "checkpoint_resume",
        (
            public_runtime_path('qrr_runs_v1/qrr1-0a294b58700020e14a459fefdf0c5b86/production-qrr-test-1-normal-long-run-20260812/batches/batch-0000/manifest.json'),
            "failure_field:mlp_pack",
            "failure_class:ENVIRONMENT_BLOCKED",
            "checkpoint_tick:5000",
        ),
    ),
    HistoricalEscapeCaseV1(
        "tick_native_partial_span_overlay_time_drift",
        (),
        "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_jungle_v1.py::test_partial_tick_span_rejects_sticky_native_failure_before_later_boundary",
        "checkpoint_resume",
        (
            public_runtime_path('qrr_runs_v1/qrr1-9c467d0f06cb4819467b56dcfb9fee9a/production-qrr-test-1-normal-long-run-20260812-final/batches/batch-0000/manifest.json'),
            "failure_status:113",
            "checkpoint_tick:5000",
            "next_lifecycle_boundary_tick:5120",
            "historical_shape:B48_W32_terminal50000_checkpoint_every5000_restore25000",
        ),
    ),
    HistoricalEscapeCaseV1(
        "tick_native_short_aligned_segment_legacy_minimum",
        (),
        "tests/test_eiel_qrr_v1.py::test_absolute_tick_horizon_has_no_divisibility_requirement",
        "canonical_execution_request",
        (
            "failure:shared WDL/Jungle logical generation count must be in [3,128]",
            "production_shape:B48_W32_terminal321_checkpoint161",
            "required_behavior:aligned_160_tick_segments_use_tick_span_owner",
        ),
    ),
    HistoricalEscapeCaseV1(
        "legacy_windowed_native_failure_surface_misclassified",
        (),
        "tests/test_eiel_cuda_architecture_b_direct_resident_wdl_jungle_v1.py::test_legacy_windowed_result_validates_nested_native_failure_records",
        "runtime_failure_authority",
        (
            "qualification_escape:WDL/Jungle/EED result has malformed native failure records",
            "route:legacy_G3_plus_windowed_owner",
            "required_behavior:validate_each_nested_native_window_failure_surface",
        ),
    ),
)


def historical_escape_inventory_v1() -> dict[str, Any]:
    payload = {
        "schema_version": "eiel_historical_qualification_escape_inventory_v1",
        "suite_id": HISTORICAL_ESCAPE_SUITE_ID,
        "cases": [asdict(case) for case in HISTORICAL_ESCAPE_CASES_V1],
    }
    return {**payload, "inventory_sha256": canonical_sha256(payload)}


def run_historical_escape_suite_v1(source_root: Path) -> dict[str, Any]:
    """Run only accumulated escape sentinels and return a durable receipt."""

    root = source_root.expanduser().resolve()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *(case.regression_node_id for case in HISTORICAL_ESCAPE_CASES_V1),
    ]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{root}:{root / 'engine'}" + (
        f":{existing}" if existing else ""
    )
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    duration = time.monotonic() - started
    core = {
        "schema_version": "eiel_historical_qualification_escape_suite_receipt_v1",
        "suite_id": HISTORICAL_ESCAPE_SUITE_ID,
        "inventory_sha256": historical_escape_inventory_v1()["inventory_sha256"],
        "case_count": len(HISTORICAL_ESCAPE_CASES_V1),
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 3),
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-4000:],
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


__all__ = [
    "HISTORICAL_ESCAPE_CASES_V1",
    "HISTORICAL_ESCAPE_SUITE_ID",
    "HistoricalEscapeCaseV1",
    "historical_escape_inventory_v1",
    "run_historical_escape_suite_v1",
]
