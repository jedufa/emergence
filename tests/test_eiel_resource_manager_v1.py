from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import resource_monitor as erm
from tools.actors import resource_manager

FIXED = "2026-08-03T12:00:00Z"
GIB = 1024**3


def _report(*, blocked: bool = False) -> dict:
    blockers = (
        [{"class": "GPU_BLOCKER", "kind": "active_gpu_workload", "pid": 123}]
        if blocked
        else []
    )
    return {
        "schema_version": erm.SCHEMA_VERSION,
        "generated_at_utc": FIXED,
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": ["no_scheduler"],
        "cpu": {"logical_threads": 8, "utilization_percent": 20.0},
        "memory": {"ram_available_bytes": 12 * GIB},
        "gpu": {
            "available": True,
            "gpus": [{"index": 0, "vram_free_bytes": 5 * GIB, "vram_total_bytes": 6 * GIB}],
            "processes": ([{"pid": 123}] if blocked else []),
        },
        "storage": {"roots": [{"path": "/tmp/emergence-test/storage", "free_bytes": 100 * GIB}]},
        "running_eiel_workloads": ([{"pid": 123, "device": "cuda"}] if blocked else []),
        "slot_heartbeats": [],
        "campaign_manifests": [],
        "ste": {"available": False},
        "blockers": blockers,
        "recommendation": {
            "value": "WAIT" if blocked else "SAFE",
            "confidence": "HIGH",
            "reason": "test",
        },
        "confidence": "HIGH",
        "blocker_classification": ["GPU_BLOCKER"] if blocked else [],
        "recommended_poll_interval_seconds": 30,
        "next_check_after_seconds": 30,
    }


def test_resource_manager_admits_safe_cuda_and_retains_history(tmp_path: Path, capsys) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    demand = tmp_path / "demand.json"
    demand.write_text(json.dumps({
        "device": "cuda",
        "exclusive_gpu": True,
        "gpu_admission_mode": "exclusive",
        "minimum_ram_available_gib": 4,
        "minimum_vram_available_gib": 4,
        "minimum_storage_available_gib": 10,
    }), encoding="utf-8")
    state = tmp_path / "state"

    assert resource_manager.main([
        "assess", "--demand", str(demand), "--report", str(report),
        "--state-root", str(state), "--generated-at", FIXED,
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["assessment"]["decision"] == "ADMIT"
    assert result["assessment"]["recommended_route"] == "cuda"

    assert resource_manager.main(["history", "--kind", "assessments", "--state-root", str(state)]) == 0
    history = json.loads(capsys.readouterr().out)
    assert history["count"] == 1


def _shared_report(*, existing_peak_gib: float = 1.7) -> dict[str, Any]:
    return {
        "schema_version": erm.SCHEMA_VERSION,
        "generated_at_utc": FIXED,
        "mode": "read_only_stateless_fresh_inspection",
        "non_goals": ["no_scheduler"],
        "cpu": {"logical_threads": 8, "utilization_percent": 20.0},
        "memory": {"ram_available_bytes": 12 * GIB},
        "gpu": {
            "available": True,
            "gpus": [{
                "index": 0,
                "vram_free_bytes": int(3.9 * GIB),
                "vram_total_bytes": int(5.6 * GIB),
            }],
            "processes": [{"pid": 456, "used_memory_bytes": int(existing_peak_gib * GIB)}],
        },
        "storage": {"roots": [{"path": "/tmp/emergence-test/storage", "free_bytes": 100 * GIB}]},
        "running_eiel_workloads": [{
            "pid": 456,
            "device": "cuda",
            "campaign_id": "hapb-functional-diagnostic",
            "gpu_admission_mode": "shared_functional",
            "peak_vram_gib": existing_peak_gib,
        }],
        "slot_heartbeats": [],
        "campaign_manifests": [],
        "ste": {"available": False},
        "blockers": [{"class": "GPU_BLOCKER", "kind": "active_gpu_workload", "pid": 456}],
        "recommendation": {"value": "WAIT", "confidence": "HIGH", "reason": "active CUDA workload"},
        "confidence": "HIGH",
        "blocker_classification": ["GPU_BLOCKER"],
        "recommended_poll_interval_seconds": 30,
        "next_check_after_seconds": 30,
    }


def _shared_demand(peak_vram_gib: float | None, *, exclusive: bool = False) -> dict[str, Any]:
    return {
        "schema_version": resource_manager.DEMAND_SCHEMA_VERSION,
        "device": "cuda",
        "exclusive_gpu": exclusive,
        "gpu_admission_mode": "exclusive" if exclusive else "shared_functional",
        "peak_vram_gib": peak_vram_gib,
        "minimum_ram_available_gib": 4.0,
        "minimum_vram_available_gib": 2.0,
        "minimum_storage_available_gib": 10.0,
        "expected_duration_minutes": 5,
        "workload_class": "qualprep_functional_diagnostic",
    }


def test_gpu_admission_mode_defaults_to_functional_share_but_critical_classes_are_exclusive() -> None:
    shared = resource_manager.normalize_demand({
        "device": "cuda",
        "workload_class": "qualprep_functional_diagnostic",
    })
    critical = resource_manager.normalize_demand({
        "device": "cuda",
        "workload_class": "formal_checkpoint_resume",
    })

    assert shared["gpu_admission_mode"] == "shared_functional"
    assert shared["exclusive_gpu"] is False
    assert critical["gpu_admission_mode"] == "exclusive"
    assert critical["exclusive_gpu"] is True


def test_shared_functional_cuda_is_admitted_when_declared_peaks_fit() -> None:
    assessment = resource_manager.assess_resources(
        _shared_demand(2.2), _shared_report(), now_utc=FIXED
    )

    assert assessment["decision"] == "ADMIT_WITH_LIMITS"
    assert assessment["recommended_route"] == "cuda"
    assert "shared_functional_gpu_fits" in assessment["reason_codes"]
    assert assessment["observed"]["gpu_state"] == "CONTENDED_SHARED"
    assert assessment["observed"]["projected_peak_vram_gib"] == pytest.approx(3.9)
    assert assessment["operating_policy"]["contended_measurements_non_gating"] is True
    assert assessment["operating_policy"]["formal_qualification_requires_exclusive_rerun"] is True


def test_shared_functional_cuda_requires_peak_declaration_while_idle() -> None:
    report = _report()
    assessment = resource_manager.assess_resources(
        _shared_demand(None), report, now_utc=FIXED
    )

    assert assessment["decision"] == "WAIT"
    assert "shared_candidate_peak_unknown" in assessment["reason_codes"]


def test_shared_functional_cuda_waits_when_combined_peak_exceeds_capacity() -> None:
    assessment = resource_manager.assess_resources(
        _shared_demand(4.964), _shared_report(), now_utc=FIXED
    )

    assert assessment["decision"] == "WAIT"
    assert "shared_peak_vram_exceeds_capacity" in assessment["reason_codes"]
    assert assessment["observed"]["projected_peak_vram_gib"] == pytest.approx(6.664)
    assert assessment["observed"]["usable_vram_gib"] == pytest.approx(5.6)


def test_exclusive_cuda_waits_even_when_current_owner_is_shareable() -> None:
    assessment = resource_manager.assess_resources(
        _shared_demand(2.2, exclusive=True), _shared_report(), now_utc=FIXED
    )

    assert assessment["decision"] == "WAIT"
    assert "exclusive_gpu_conflict" in assessment["reason_codes"]


def test_resource_manager_waits_for_cuda_but_allows_cpu_fallback(tmp_path: Path, capsys) -> None:
    report = tmp_path / "blocked.json"
    report.write_text(json.dumps(_report(blocked=True)), encoding="utf-8")
    state = tmp_path / "state"

    assert resource_manager.main([
        "assess", "--device", "cuda", "--report", str(report),
        "--state-root", str(state), "--generated-at", FIXED,
    ]) == 1
    cuda = json.loads(capsys.readouterr().out)
    assert cuda["assessment"]["decision"] == "WAIT"

    assert resource_manager.main([
        "recommend", "--device", "either", "--report", str(report),
        "--state-root", str(state), "--generated-at", FIXED,
    ]) == 0
    fallback = json.loads(capsys.readouterr().out)
    assert fallback["assessment"]["decision"] == "ADMIT_WITH_LIMITS"
    assert fallback["assessment"]["recommended_route"] == "cpu"


def test_resource_manager_inspect_and_campaign_views_are_read_only(tmp_path: Path, capsys) -> None:
    report = tmp_path / "report.json"
    payload = _report(blocked=True)
    payload["slot_heartbeats"] = [{"slot_id": "slot-1", "status": "healthy"}]
    report.write_text(json.dumps(payload), encoding="utf-8")
    state = tmp_path / "state"

    assert resource_manager.main([
        "inspect", "--report", str(report), "--state-root", str(state),
        "--generated-at", FIXED,
    ]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["report"]["mode"] == "read_only_stateless_fresh_inspection"
    assert inspected["record_path"]

    assert resource_manager.main(["campaigns", "--report", str(report), "--state-root", str(state)]) == 0
    campaigns = json.loads(capsys.readouterr().out)
    assert len(campaigns["running_eiel_workloads"]) == 1
    assert len(campaigns["slot_heartbeats"]) == 1

    assert resource_manager.main(["history", "--kind", "observations", "--state-root", str(state)]) == 0
    history = json.loads(capsys.readouterr().out)
    assert history["count"] == 1


def test_resource_manager_accepts_worktree_as_a_repo_root_compatibility_alias(
    tmp_path: Path, capsys
) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    assert resource_manager.main([
        "inspect", "--report", str(report), "--worktree", str(worktree),
    ]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["report"]["schema_version"] == erm.SCHEMA_VERSION


def test_dead_cuda_owner_reclaim_requires_dual_complete_absence_proof() -> None:
    identity = "linux-proc:boot-test:123"
    report = {
        "generated_at_utc": FIXED,
        "gpu": {
            "available": True,
            "process_query_complete": True,
            "gpus": [{"uuid": "GPU-TEST"}],
            "processes": [{"gpu_uuid": "GPU-TEST", "pid": 99}],
        },
    }
    kwargs = {
        "attempt_id": "ATTEMPT-TEST",
        "resource_lease_digest": {"algorithm": "sha256", "hex": "a" * 64},
        "fencing_token_digest": {"algorithm": "sha256", "hex": "b" * 64},
        "continuation_lease_digest": {
            "algorithm": "sha256",
            "hex": "c" * 64,
        },
        "continuation_fence_state": "EXPIRED",
        "continuation_terminal_event_digest": {
            "algorithm": "sha256",
            "hex": "d" * 64,
        },
        "resource_started_event_digest": {
            "algorithm": "sha256",
            "hex": "e" * 64,
        },
        "device_uuid": "GPU-TEST",
        "selected_resource_ids": ["gpu:GPU-TEST/lane/test/0"],
        "owner_pid": 42,
        "owner_process_start_identity": identity,
        "process_identity_observation": {
            "schema_version": "eiel_process_identity_observation_v1",
            "pid": 42,
            "expected_process_start_identity": identity,
            "observed_at_utc": FIXED,
            "state": "ABSENT",
            "observed_process_start_identity": None,
        },
        "report": report,
        "now_utc": FIXED,
    }
    accepted = resource_manager.assess_dead_cuda_owner_reclaim_v1(**kwargs)
    assert accepted["decision"] == "RECLAIM"
    assert accepted["reason_codes"] == []

    uuid_alias_report = json.loads(json.dumps(report))
    uuid_alias_report["gpu"]["gpus"][0]["uuid"] = "TEST"
    uuid_alias_report["gpu"]["processes"][0]["gpu_uuid"] = "TEST"
    uuid_alias = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{**kwargs, "report": uuid_alias_report}
    )
    assert uuid_alias["decision"] == "RECLAIM"

    invalid_fence = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{
            **kwargs,
            "continuation_fence_state": "OWNER_DEAD_REQUIRED",
        }
    )
    assert invalid_fence["decision"] == "WAIT"
    assert "CONTINUATION_FENCE_INVALID" in invalid_fence["reason_codes"]

    live = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{
            **kwargs,
            "process_identity_observation": {
                **kwargs["process_identity_observation"],
                "state": "LIVE",
                "observed_process_start_identity": identity,
            },
        }
    )
    assert live["decision"] == "WAIT"
    assert "OWNER_PROCESS_NOT_PROVEN_DEAD" in live["reason_codes"]

    cuda_live_report = json.loads(json.dumps(report))
    cuda_live_report["gpu"]["processes"].append(
        {"gpu_uuid": "GPU-TEST", "pid": 42}
    )
    cuda_live = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{**kwargs, "report": cuda_live_report}
    )
    assert cuda_live["decision"] == "WAIT"
    assert "OWNER_PID_STILL_HAS_TARGET_CUDA_CONTEXT" in cuda_live["reason_codes"]

    incomplete_report = json.loads(json.dumps(report))
    incomplete_report["gpu"]["process_query_complete"] = False
    incomplete = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{**kwargs, "report": incomplete_report}
    )
    assert incomplete["decision"] == "WAIT"
    assert "CUDA_PROCESS_TELEMETRY_INCOMPLETE" in incomplete["reason_codes"]

    stale = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{**kwargs, "now_utc": "2026-08-03T12:00:06Z"}
    )
    assert stale["decision"] == "WAIT"
    assert "STALE_RESOURCE_OBSERVATION" in stale["reason_codes"]

    stale_process = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{
            **kwargs,
            "process_identity_observation": {
                **kwargs["process_identity_observation"],
                "observed_at_utc": "2026-08-03T11:59:54Z",
            },
        }
    )
    assert stale_process["decision"] == "WAIT"
    assert (
        "STALE_PROCESS_IDENTITY_OBSERVATION" in stale_process["reason_codes"]
    )

    future_resource = json.loads(json.dumps(report))
    future_resource["generated_at_utc"] = "2026-08-03T12:00:01Z"
    future = resource_manager.assess_dead_cuda_owner_reclaim_v1(
        **{**kwargs, "report": future_resource}
    )
    assert future["decision"] == "WAIT"
    assert "CAUSAL_EVIDENCE_ORDER_INVALID" in future["reason_codes"]


def test_legacy_dead_cuda_owner_reclaim_requires_a_fully_idle_exact_device() -> None:
    report = {
        "generated_at_utc": FIXED,
        "gpu": {
            "available": True,
            "process_query_complete": True,
            "gpus": [{"uuid": "GPU-TEST"}],
            "processes": [{"gpu_uuid": "GPU-OTHER", "pid": 99}],
        },
    }
    kwargs = {
        "attempt_id": "ATTEMPT-LEGACY",
        "resource_lease_digest": {"algorithm": "sha256", "hex": "a" * 64},
        "fencing_token_digest": {"algorithm": "sha256", "hex": "b" * 64},
        "device_uuid": "GPU-TEST",
        "selected_resource_ids": ["gpu:GPU-TEST"],
        "start_effect_digest": {"algorithm": "sha256", "hex": "c" * 64},
        "resource_started_event_digest": {
            "algorithm": "sha256",
            "hex": "d" * 64,
        },
        "report": report,
        "now_utc": FIXED,
    }
    accepted = resource_manager.assess_legacy_dead_cuda_owner_reclaim_v1(
        **kwargs
    )
    assert accepted["decision"] == "RECLAIM"
    assert accepted["reason_codes"] == []

    occupied_report = json.loads(json.dumps(report))
    occupied_report["gpu"]["processes"].append(
        {"gpu_uuid": "GPU-TEST", "pid": 42}
    )
    occupied = resource_manager.assess_legacy_dead_cuda_owner_reclaim_v1(
        **{**kwargs, "report": occupied_report}
    )
    assert occupied["decision"] == "WAIT"
    assert "TARGET_CUDA_DEVICE_NOT_IDLE" in occupied["reason_codes"]

    unknown_report = json.loads(json.dumps(report))
    unknown_report["gpu"]["processes"].append({"gpu_uuid": None, "pid": 43})
    unknown = resource_manager.assess_legacy_dead_cuda_owner_reclaim_v1(
        **{**kwargs, "report": unknown_report}
    )
    assert unknown["decision"] == "WAIT"
    assert "CUDA_PROCESS_DEVICE_UNKNOWN" in unknown["reason_codes"]


def test_legacy_gpu_migration_uses_class_specific_absence_proofs() -> None:
    report = {
        "generated_at_utc": FIXED,
        "gpu": {
            "available": True,
            "process_query_complete": True,
            "gpus": [{"uuid": "GPU-TEST"}],
            "processes": [{"gpu_uuid": "GPU-TEST", "pid": 99}],
        },
    }
    common = {
        "migration_manifest_sha256": "1" * 64,
        "migration_row_sha256": "2" * 64,
        "attempt_id": "ATTEMPT-MIGRATION",
        "resource_lease_digest": {"algorithm": "sha256", "hex": "a" * 64},
        "fencing_token_digest": {"algorithm": "sha256", "hex": "b" * 64},
        "continuation_lease_digest": {
            "algorithm": "sha256",
            "hex": "c" * 64,
        },
        "continuation_terminal_event_digest": None,
        "continuation_successor_event_digest": None,
        "device_uuid": "GPU-TEST",
        "selected_resource_ids": ["gpu:GPU-TEST"],
        "start_effect_digest": {"algorithm": "sha256", "hex": "d" * 64},
        "resource_started_event_digest": {
            "algorithm": "sha256",
            "hex": "e" * 64,
        },
        "report": report,
        "now_utc": FIXED,
    }
    sealed = resource_manager.assess_legacy_gpu_migration_reclaim_v1(
        **common,
        classification="SEALED_PROCESS_CONTEXT",
        continuation_fence_code="ACTIVE",
        owner_pid=42,
        owner_process_start_ticks=123,
        process_start_ticks_observation={
            "schema_version": "eiel_legacy_process_start_ticks_observation_v1",
            "pid": 42,
            "expected_process_start_ticks": 123,
            "observed_at_utc": FIXED,
            "state": "ABSENT",
            "observed_process_start_ticks": None,
        },
    )
    assert sealed["decision"] == "RECLAIM"
    assert sealed["target_device_gpu_processes"] == [
        {"gpu_uuid": "GPU-TEST", "pid": 99}
    ]

    owner_live_report = json.loads(json.dumps(report))
    owner_live_report["gpu"]["processes"].append(
        {"gpu_uuid": "GPU-TEST", "pid": 42}
    )
    owner_live = resource_manager.assess_legacy_gpu_migration_reclaim_v1(
        **{**common, "report": owner_live_report},
        classification="SEALED_PROCESS_CONTEXT",
        continuation_fence_code="ACTIVE",
        owner_pid=42,
        owner_process_start_ticks=123,
        process_start_ticks_observation={
            "schema_version": "eiel_legacy_process_start_ticks_observation_v1",
            "pid": 42,
            "expected_process_start_ticks": 123,
            "observed_at_utc": FIXED,
            "state": "ABSENT",
            "observed_process_start_ticks": None,
        },
    )
    assert owner_live["decision"] == "WAIT"
    assert "OWNER_PID_STILL_HAS_TARGET_CUDA_CONTEXT" in owner_live["reason_codes"]

    expired = {"algorithm": "sha256", "hex": "f" * 64}
    pidless = resource_manager.assess_legacy_gpu_migration_reclaim_v1(
        **{
            **common,
            "report": {
                **report,
                "gpu": {**report["gpu"], "processes": []},
            },
            "continuation_terminal_event_digest": expired,
        },
        classification="TRUE_PIDLESS_DEVICE_IDLE",
        continuation_fence_code="EXPIRED",
        owner_pid=None,
        owner_process_start_ticks=None,
        process_start_ticks_observation=None,
    )
    assert pidless["decision"] == "RECLAIM"

    pidless_busy = resource_manager.assess_legacy_gpu_migration_reclaim_v1(
        **{**common, "continuation_terminal_event_digest": expired},
        classification="TRUE_PIDLESS_DEVICE_IDLE",
        continuation_fence_code="EXPIRED",
        owner_pid=None,
        owner_process_start_ticks=None,
        process_start_ticks_observation=None,
    )
    assert pidless_busy["decision"] == "WAIT"
    assert "TARGET_CUDA_DEVICE_NOT_IDLE" in pidless_busy["reason_codes"]
