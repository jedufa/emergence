from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from engine.campaigns.canonical import canonical_sha256
from tools.scripts.eiel_gpu_tick_ab_reducer_v1 import (
    ABReductionError,
    BENCHMARK_ARM_ORDER,
    M7_BUILD_SCHEMA,
    M8_BUILD_SCHEMA,
    reduce_ab_receipts,
)


def _hardware_identity() -> dict[str, Any]:
    payload = {
        "device_index": 0,
        "device_count": 1,
        "device_name": "test GPU",
        "device_capability": [7, 5],
        "device_total_memory_bytes": 6_000_000_000,
        "device_multiprocessor_count": 24,
        "gpu_uuid": "GPU-test",
        "driver_version": "999.1",
        "torch_version": "2.test",
        "torch_cuda_runtime_version": "12.test",
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _build_identity(arm: str, *, m8: bool) -> dict[str, Any]:
    enabled = arm != "legacy" if not m8 else arm == "m8_suffix_reuse"
    field = (
        "option_d_m8_resident_policy_suffix_reuse_enabled"
        if m8
        else "option_d_m7_batched_cpu_policy_refill_enabled"
    )
    macro = (
        "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED"
        if m8
        else "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED"
    )
    authority = (
        "candidate_resident_policy_suffix_reuse"
        if m8
        else "candidate_batched_exact_cpu_lsb0_policy_refill"
    )
    payload = {
        "schema_version": M8_BUILD_SCHEMA if m8 else M7_BUILD_SCHEMA,
        field: enabled,
        "compile_time_macro": macro,
        "receipt_field": field,
        "default_enabled": False,
        "compile_definition": f"-D{macro}={1 if enabled else 0}",
        "authority": authority if enabled else "disabled",
        "identity_authority": "compiled_binary_resource_receipt",
        "runtime_resource_receipt_confirmed": True,
        "runtime_resource_receipt_sha256": (
            str(BENCHMARK_ARM_ORDER.index(arm) + 1) if not m8 else "a"
        )
        * 64,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _comparison_identity() -> dict[str, str]:
    payload = {
        "source_plan_sha256": "1" * 64,
        "execution_contract_identity_sha256": "2" * 64,
        "source_identity_sha256": "3" * 64,
        "device_identity_sha256": _hardware_identity()["identity_sha256"],
        "configuration_identity_sha256": "5" * 64,
        "warmup_identity_sha256": "6" * 64,
        "warmup_identity": "source_and_resident_owner_preparation_no_native_ticks_v1",
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _receipt(
    path: Path,
    *,
    arm: str,
    run_index: int,
    wall_ms: float,
    digest: str = "a" * 64,
    peak_allocated: int | None = 100,
) -> Path:
    arm_index = BENCHMARK_ARM_ORDER.index(arm)
    m7_enabled = arm != "legacy"
    m8_enabled = arm == "m8_suffix_reuse"
    runtime = {
        "expected_m7_enabled": m7_enabled,
        "expected_m8_enabled": m8_enabled,
        "receipt_count": 159,
        "resident_suffix_reuse_receipt_count": 159 if m8_enabled else 0,
        "physical_refill_receipt_count": 0 if m8_enabled else 159,
        "turnover_boundary_receipt_count": 0,
        "m7_used_on_every_physical_refill": m7_enabled,
        "m7_unused_on_every_suffix_reuse": True,
    }
    payload: dict[str, Any] = {
        "schema_version": "eiel_gpu_tick_anatomy_v1_execution_receipt_v1",
        "status": "success",
        "cells": 48,
        "ticks": 160,
        "benchmark_arm": arm,
        "benchmark_run_index": run_index,
        "benchmark_sequence_index": run_index * 3 + arm_index,
        "benchmark_arm_order": list(BENCHMARK_ARM_ORDER),
        "comparison_identity": _comparison_identity(),
        "warmup_completed": True,
        "artifact_source_commit": "a" * 40,
        "artifact_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "native_window_wall_time_ms": wall_ms * 160,
        "wall_ms_per_tick": wall_ms,
        "native_result": {
            "validated_policy_arm_identity": dict(runtime),
            "turnover_shadow_plan_launches": 0,
            "turnover_cull_refill_plan_launches": 0,
            "turnover_commit_launches": 0,
            "turnover_publish_launches": 0,
        },
        "terminal_cursor": {"generation": 1, "tick": 160},
        "terminal_trajectory_digest_sha256s": {
            f"cell-{index:02d}": digest for index in range(48)
        },
        "m7_arm_identity": {
            "arm": "candidate" if m7_enabled else "control",
            "expected_enabled": m7_enabled,
            "manifest": _build_identity(arm, m8=False),
            "runtime": dict(runtime),
        },
        "m8_arm_identity": {
            "arm": arm,
            "expected_enabled": m8_enabled,
            "manifest": _build_identity(arm, m8=True),
            "runtime": dict(runtime),
        },
        "started_at_unix_ns": run_index * 10 + arm_index * 2,
        "completed_at_unix_ns": run_index * 10 + arm_index * 2 + 1,
    }
    if peak_allocated is not None:
        payload["torch_peak_allocated_bytes_during_window"] = peak_allocated
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _receipts(
    tmp_path: Path, *, legacy_ms: float = 10.0, m7_ms: float = 8.5, m8_ms: float = 5.0
) -> dict[str, list[Path]]:
    times = {
        "legacy": legacy_ms,
        "m7_physical_refill": m7_ms,
        "m8_suffix_reuse": m8_ms,
    }
    return {
        arm: [
            _receipt(
                tmp_path / f"{arm}-{run_index}.json",
                arm=arm,
                run_index=run_index,
                wall_ms=times[arm] + run_index * 0.01,
                peak_allocated=100 - BENCHMARK_ARM_ORDER.index(arm),
            )
            for run_index in range(5)
        ]
        for arm in BENCHMARK_ARM_ORDER
    }


def _component_gate(path: Path, *, passed: bool = True) -> Path:
    medians = {
        "legacy": 110.0,
        "m7_physical_refill": 100.0 if passed else 101.0,
        "m8_suffix_reuse": 90.0,
    }
    speedups = {
        arm: medians["legacy"] / medians[arm]
        for arm in ("m7_physical_refill", "m8_suffix_reuse")
    }
    payload: dict[str, Any] = {
        "schema_version": "eiel_b48_fixed_mlp_policy_callback_gate_v1",
        "authority": "fixed_mlp_component_callback_promotion_gate_v1",
        "status": "success",
        "component_surface": {
            "cells": 48,
            "episodes": 124,
            "policy_width": 9_991,
            "packed_row_bytes": 1_249,
            "mutation_enable": False,
            "mutation_mode": "dense",
            "compact_mutation_width": 1,
        },
        "required_repeats_per_arm": 5,
        "artifact_source_commit": "a" * 40,
        "host_profiler_source_commit": "a" * 40,
        "hardware_identity": _hardware_identity(),
        "control_output_digest": "b" * 64,
        "exact_output_digest_match_across_arms": True,
        "m8_build_binds_m7_exact_physical_refill_fallback": True,
        "metric": "ratio_of_arm_medians_callback_host_wall_time_ns",
        "minimum_median_speedup_inclusive": 1.10,
        "callback_median_gate_passed": passed,
        "arms": {
            arm: {
                "profile_path": str(path.parent / f"{arm}-profile.json"),
                "profile_receipt_sha256": "c" * 64,
                "artifact_sha256": "d" * 64,
                "manifest_sha256": "e" * 64,
                "compiled_source_snapshot_sha256": "f" * 64,
                "build_identity": {
                    "m7": _build_identity(arm, m8=False),
                    "m8": _build_identity(arm, m8=True),
                },
                "runtime_identity": {},
                "host_wall_time_ns": {
                    "count": 5,
                    "min_ns": medians[arm] - 1,
                    "median_ns": medians[arm],
                    "mean_ns": medians[arm],
                    "p95_ns": medians[arm] + 1,
                    "max_ns": medians[arm] + 1,
                },
            }
            for arm in BENCHMARK_ARM_ORDER
        },
        "median_speedup_over_legacy": speedups,
        "qualification_claim": False,
        "automatic_promotion_authorized_by_this_receipt": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reduce(
    rows: dict[str, list[Path]], *, component_gate: Path | None = None
) -> dict[str, Any]:
    if component_gate is None:
        component_gate = _component_gate(rows["legacy"][0].parent / "component.json")
    return reduce_ab_receipts(
        legacy_paths=rows["legacy"],
        m7_paths=rows["m7_physical_refill"],
        m8_paths=rows["m8_suffix_reuse"],
        component_gate_path=component_gate,
    )


@pytest.mark.parametrize(
    ("m8_ms", "band", "eligible"),
    [
        (10.0 / 1.03, "reject", False),
        (10.0 / 1.05, "marginal_no_automatic_promotion", False),
        (10.0 / 1.20, "eligible_subject_to_maintainability_and_qualification", True),
    ],
)
def test_reducer_encodes_three_arm_whole_tick_decision_bands(
    tmp_path: Path, m8_ms: float, band: str, eligible: bool
) -> None:
    result = _reduce(_receipts(tmp_path, legacy_ms=10.0, m8_ms=m8_ms))

    assert result["decision"]["band"] == band
    assert result["decision"]["performance_eligible"] is eligible
    assert result["decision"]["automatic_promotion_authorized_by_this_receipt"] is False
    assert result["metrics"]["wall_ms_per_tick"]["legacy"]["count"] == 5
    assert result["alternation_proven"] is True


def test_reducer_reports_three_arm_distributions_and_resources(tmp_path: Path) -> None:
    result = _reduce(_receipts(tmp_path))

    assert result["required_repeats_per_arm"] == 5
    assert result["receipt_count"] == 15
    assert len(result["triplets"]) == 5
    resource = result["resource_metrics"]["torch_peak_allocated_bytes_during_window"]
    assert resource["m8_minus_legacy_paired"]["median"] == -2.0


def _mutate(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    mutation(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(cells=24), "B48/T160"),
        (lambda row: row.update(ticks=159), "B48/T160"),
        (lambda row: row.update(benchmark_sequence_index=14), "alternating"),
        (lambda row: row.pop("comparison_identity"), "comparison identity"),
        (
            lambda row: row["comparison_identity"].update(
                device_identity_sha256="f" * 64
            ),
            "comparison identity digest",
        ),
        (
            lambda row: row["terminal_trajectory_digest_sha256s"].update(
                {"cell-00": "b" * 64}
            ),
            "trajectory",
        ),
        (
            lambda row: row["m8_arm_identity"]["runtime"].update(
                physical_refill_receipt_count=1
            ),
            "M7/M8 arm identity",
        ),
    ],
)
def test_reducer_fails_closed_on_identity_or_semantic_drift(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None], message: str
) -> None:
    rows = _receipts(tmp_path)
    _mutate(rows["m8_suffix_reuse"][0], mutation)

    with pytest.raises(ABReductionError, match=message):
        _reduce(rows)


def test_reducer_requires_exactly_five_runs_per_arm(tmp_path: Path) -> None:
    rows = _receipts(tmp_path)
    rows["m8_suffix_reuse"].pop()

    with pytest.raises(ABReductionError, match="exactly 5"):
        _reduce(rows)


def test_reducer_uses_component_gate_not_untrusted_whole_tick_booleans(
    tmp_path: Path,
) -> None:
    rows = _receipts(tmp_path)
    for arm in BENCHMARK_ARM_ORDER:
        for path in rows[arm]:
            _mutate(path, lambda row: row.update(callback_median_gate_passed=True))
    component_gate = _component_gate(tmp_path / "failed-component.json", passed=False)

    result = _reduce(rows, component_gate=component_gate)

    assert result["decision"]["performance_eligible"] is False
    assert result["decision"]["reason"] == "predeclared_callback_median_gate_not_proven"


def test_reducer_fails_closed_on_component_gate_hardware_or_identity_drift(
    tmp_path: Path,
) -> None:
    rows = _receipts(tmp_path)
    component_gate = _component_gate(tmp_path / "component.json")
    payload = json.loads(component_gate.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    payload["hardware_identity"]["gpu_uuid"] = "GPU-other"
    payload["hardware_identity"].pop("identity_sha256")
    payload["hardware_identity"]["identity_sha256"] = canonical_sha256(
        payload["hardware_identity"]
    )
    payload["receipt_sha256"] = canonical_sha256(payload)
    component_gate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ABReductionError, match="hardware identity differs"):
        _reduce(rows, component_gate=component_gate)


def test_reducer_rejects_receipt_digest_tampering(tmp_path: Path) -> None:
    rows = _receipts(tmp_path)
    payload = json.loads(rows["legacy"][0].read_text(encoding="utf-8"))
    payload["wall_ms_per_tick"] = 1.0
    rows["legacy"][0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ABReductionError, match="digest mismatch"):
        _reduce(rows)
