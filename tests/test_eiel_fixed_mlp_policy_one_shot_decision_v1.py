from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from engine.campaigns.canonical import canonical_sha256
from tools.scripts.eiel_fixed_mlp_policy_one_shot_decision_v1 import (
    BENCHMARK_ARM_ORDER,
    CONTINUE_DECISION,
    M8_CONTINUE_MIN_SPEEDUP,
    OneShotDecisionError,
    SCHEMA_VERSION,
    STOP_DECISION,
    reduce_one_shot_receipts,
)
from tools.scripts.eiel_gpu_tick_ab_reducer_v1 import M7_BUILD_SCHEMA, M8_BUILD_SCHEMA


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
        "device_identity_sha256": "4" * 64,
        "configuration_identity_sha256": "5" * 64,
        "warmup_identity_sha256": "6" * 64,
        "warmup_identity": "source_and_resident_owner_preparation_no_native_ticks_v1",
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def _receipt(
    path: Path,
    *,
    arm: str,
    wall_ms_per_tick: float,
    trajectory_digest: str = "b" * 64,
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
        "authority": "diagnostic_profiling_only_not_science_or_qualification_v1",
        "status": "success",
        "cells": 48,
        "ticks": 160,
        "benchmark_arm": arm,
        "benchmark_run_index": 0,
        "benchmark_sequence_index": arm_index,
        "benchmark_arm_order": list(BENCHMARK_ARM_ORDER),
        "comparison_identity": _comparison_identity(),
        "warmup_completed": True,
        "artifact_source_commit": "a" * 40,
        "artifact_sha256": f"{arm_index + 1}" * 64,
        "manifest_sha256": f"{arm_index + 4}" * 64,
        "native_window_wall_time_ms": wall_ms_per_tick * 160,
        "wall_ms_per_tick": wall_ms_per_tick,
        "native_result": {
            "validated_policy_arm_identity": dict(runtime),
            "turnover_shadow_plan_launches": 0,
            "turnover_cull_refill_plan_launches": 0,
            "turnover_commit_launches": 0,
            "turnover_publish_launches": 0,
        },
        "terminal_cursor": {"generation": 1, "tick": 0},
        "terminal_trajectory_digest_sha256s": {
            f"cell-{index:02d}": trajectory_digest for index in range(48)
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
        "started_at_unix_ns": arm_index * 10,
        "completed_at_unix_ns": arm_index * 10 + 1,
        "qualification_claim": False,
        "scientific_evidence_claim": False,
        "optimization_performed": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _triplet(
    tmp_path: Path,
    *,
    legacy_ms: float = 10.0,
    m7_ms: float = 9.0,
    m8_ms: float = 8.0,
) -> dict[str, Path]:
    return {
        "legacy": _receipt(
            tmp_path / "legacy.json", arm="legacy", wall_ms_per_tick=legacy_ms
        ),
        "m7_physical_refill": _receipt(
            tmp_path / "m7.json", arm="m7_physical_refill", wall_ms_per_tick=m7_ms
        ),
        "m8_suffix_reuse": _receipt(
            tmp_path / "m8.json", arm="m8_suffix_reuse", wall_ms_per_tick=m8_ms
        ),
    }


def _reduce(paths: dict[str, Path]) -> dict[str, Any]:
    return reduce_one_shot_receipts(
        legacy_path=paths["legacy"],
        m7_path=paths["m7_physical_refill"],
        m8_path=paths["m8_suffix_reuse"],
    )


def _rewrite(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    mutate(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_continues_at_inclusive_m8_whole_tick_threshold(tmp_path: Path) -> None:
    result = _reduce(_triplet(tmp_path))

    digest = result.pop("reduction_sha256")
    assert digest == canonical_sha256(result)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["decision"]["result"] == CONTINUE_DECISION
    assert result["decision"]["continue_full_campaign"] is True
    assert result["decision"]["m8_speedup_over_legacy"] == M8_CONTINUE_MIN_SPEEDUP
    assert result["speedup_over_legacy"] == {
        "m7_physical_refill": 10.0 / 9.0,
        "m8_suffix_reuse": 1.25,
    }
    assert result["arms"]["m8_suffix_reuse"]["wall_ms_per_tick"] == 8.0
    assert result["arms"]["m8_suffix_reuse"]["ticks_per_second"] == 125.0
    assert result["semantic_equality"]["exact_parity"] is True
    assert result["qualification_claim"] is False
    assert result["promotion_claim"] is False


def test_stops_when_m8_whole_tick_speedup_is_below_threshold(tmp_path: Path) -> None:
    result = _reduce(_triplet(tmp_path, m8_ms=8.01))
    assert result["decision"]["result"] == STOP_DECISION
    assert result["decision"]["continue_full_campaign"] is False
    assert result["decision"]["reason"] == "m8_whole_tick_speedup_below_1_25"


def test_trajectory_mismatch_is_retained_as_stop_decision(tmp_path: Path) -> None:
    paths = _triplet(tmp_path)

    def change_trajectory(payload: dict[str, Any]) -> None:
        payload["terminal_trajectory_digest_sha256s"]["cell-47"] = "c" * 64

    _rewrite(paths["m8_suffix_reuse"], change_trajectory)
    result = _reduce(paths)
    assert result["decision"]["result"] == STOP_DECISION
    assert result["decision"]["reason"] == "terminal_trajectory_parity_failed"
    assert result["semantic_equality"]["exact_parity"] is False


@pytest.mark.parametrize(
    ("arm", "mutate", "match"),
    [
        ("legacy", lambda row: row.update(cells=47), "exact B48/T160"),
        (
            "m7_physical_refill",
            lambda row: row.update(benchmark_run_index=1),
            "run index",
        ),
        (
            "m8_suffix_reuse",
            lambda row: row.update(benchmark_sequence_index=1),
            "sequence indexes",
        ),
        (
            "m8_suffix_reuse",
            lambda row: row["m8_arm_identity"]["runtime"].update(
                resident_suffix_reuse_receipt_count=158,
                physical_refill_receipt_count=1,
            ),
            "arm identity mismatch",
        ),
        (
            "m7_physical_refill",
            lambda row: row["native_result"].update(turnover_commit_launches=1),
            "arm identity mismatch",
        ),
        (
            "legacy",
            lambda row: row.update(native_window_wall_time_ms=1599.0),
            "inconsistent",
        ),
        (
            "m8_suffix_reuse",
            lambda row: row.update(qualification_claim=True),
            "diagnostic anatomy contract",
        ),
    ],
)
def test_malformed_or_noncomparable_receipt_fails_closed(
    tmp_path: Path,
    arm: str,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    paths = _triplet(tmp_path)
    _rewrite(paths[arm], mutate)
    with pytest.raises(OneShotDecisionError, match=match):
        _reduce(paths)


@pytest.mark.parametrize(
    "field",
    [
        "source_plan_sha256",
        "execution_contract_identity_sha256",
        "source_identity_sha256",
        "device_identity_sha256",
        "configuration_identity_sha256",
        "warmup_identity_sha256",
    ],
)
def test_each_comparison_axis_must_be_exact_across_arms(
    tmp_path: Path, field: str
) -> None:
    paths = _triplet(tmp_path)

    def drift(payload: dict[str, Any]) -> None:
        comparison = payload["comparison_identity"]
        comparison.pop("identity_sha256")
        comparison[field] = "f" * 64
        comparison["identity_sha256"] = canonical_sha256(comparison)

    _rewrite(paths["m8_suffix_reuse"], drift)
    with pytest.raises(OneShotDecisionError, match="identity drift"):
        _reduce(paths)


def test_receipt_self_digest_is_mandatory(tmp_path: Path) -> None:
    paths = _triplet(tmp_path)
    payload = json.loads(paths["legacy"].read_text(encoding="utf-8"))
    payload["wall_ms_per_tick"] = 1.0
    paths["legacy"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OneShotDecisionError, match="digest mismatch"):
        _reduce(paths)


def test_pre_rollover_t160_cursor_is_rejected(tmp_path: Path) -> None:
    paths = _triplet(tmp_path)
    _rewrite(
        paths["legacy"],
        lambda row: row.update(terminal_cursor={"generation": 1, "tick": 160}),
    )
    with pytest.raises(OneShotDecisionError, match="post-T160 G1:T0"):
        _reduce(paths)


def test_overlapping_intervals_fail_alternation_proof(tmp_path: Path) -> None:
    paths = _triplet(tmp_path)
    _rewrite(
        paths["legacy"],
        lambda row: row.update(started_at_unix_ns=0, completed_at_unix_ns=11),
    )
    with pytest.raises(OneShotDecisionError, match="non-overlapping"):
        _reduce(paths)
