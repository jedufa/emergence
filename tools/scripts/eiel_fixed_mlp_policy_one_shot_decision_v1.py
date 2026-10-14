#!/usr/bin/env python3
"""Reduce one exact legacy/M7/M8 B48/T160 diagnostic decision triplet.

This is a campaign-spend decision receipt, not qualification or promotion
evidence.  Receipt integrity, execution identity, policy-arm identity, and the
exact benchmark geometry fail closed.  A valid terminal trajectory mismatch is
retained as a negative result and forces ``STOP_AND_REASSESS``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402
from tools.scripts.eiel_gpu_tick_ab_reducer_v1 import (  # noqa: E402
    ABReductionError,
    BENCHMARK_ARM_ORDER,
    REQUIRED_CALLBACKS,
    REQUIRED_CELLS,
    REQUIRED_TICKS,
    _comparison_identity,
    _load_immutable_receipt,
    _positive_metric,
    _terminal_identity,
    _validated_arm_identity,
)


SCHEMA_VERSION = "eiel_fixed_mlp_policy_one_shot_decision_v1"
AUTHORITY = "diagnostic_campaign_spend_decision_only_not_qualification_v1"
INPUT_SCHEMA = "eiel_gpu_tick_anatomy_v1_execution_receipt_v1"
INPUT_AUTHORITY = "diagnostic_profiling_only_not_science_or_qualification_v1"
CONTINUE_DECISION = "CONTINUE_FULL_CAMPAIGN"
STOP_DECISION = "STOP_AND_REASSESS"
M8_CONTINUE_MIN_SPEEDUP = 1.25
REQUIRED_TERMINAL_CURSOR = {"generation": 1, "tick": 0}
SHA256_FIELDS = (
    "source_plan_sha256",
    "execution_contract_identity_sha256",
    "source_identity_sha256",
    "device_identity_sha256",
    "configuration_identity_sha256",
    "warmup_identity_sha256",
    "identity_sha256",
)


class OneShotDecisionError(ABReductionError):
    """Raised when the triplet cannot support the one-shot decision."""


def _fail_closed(callable_: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return callable_(*args, **kwargs)
    except ABReductionError as exc:
        raise OneShotDecisionError(str(exc)) from exc


def _load_receipt(path: Path) -> dict[str, Any]:
    receipt = _fail_closed(_load_immutable_receipt, path)
    if (
        receipt.get("schema_version") != INPUT_SCHEMA
        or receipt.get("authority") != INPUT_AUTHORITY
        or receipt.get("qualification_claim") is not False
        or receipt.get("scientific_evidence_claim") is not False
        or receipt.get("optimization_performed") is not False
    ):
        raise OneShotDecisionError(
            "execution receipt is not the exact diagnostic anatomy contract"
        )
    return receipt


def _validated_comparison_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    identity = _fail_closed(_comparison_identity, receipt)
    for field in SHA256_FIELDS:
        value = identity.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise OneShotDecisionError(
                f"comparison identity lacks exact lowercase SHA-256 field {field}"
            )
    warmup = identity.get("warmup_identity")
    if not isinstance(warmup, str) or not warmup:
        raise OneShotDecisionError("comparison identity lacks exact warmup identity")
    return identity


def _validated_window_metrics(
    receipt: Mapping[str, Any],
) -> tuple[float, float, float]:
    wall_ms_per_tick = _fail_closed(_positive_metric, receipt, "wall_ms_per_tick")
    window_wall_ms = _fail_closed(
        _positive_metric, receipt, "native_window_wall_time_ms"
    )
    if not math.isclose(
        window_wall_ms,
        wall_ms_per_tick * REQUIRED_TICKS,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise OneShotDecisionError(
            "native window wall time is inconsistent with exact T160 per-tick timing"
        )
    return window_wall_ms, wall_ms_per_tick, 1000.0 / wall_ms_per_tick


def _validated_timestamp(receipt: Mapping[str, Any]) -> tuple[int, int]:
    started = receipt.get("started_at_unix_ns")
    completed = receipt.get("completed_at_unix_ns")
    if type(started) is not int or type(completed) is not int or started >= completed:
        raise OneShotDecisionError("execution receipt has invalid wall-clock interval")
    return started, completed


def reduce_one_shot_receipts(
    *, legacy_path: Path, m7_path: Path, m8_path: Path
) -> dict[str, Any]:
    """Return the fail-closed one-triplet diagnostic campaign decision."""

    paths = {
        "legacy": legacy_path,
        "m7_physical_refill": m7_path,
        "m8_suffix_reuse": m8_path,
    }
    arms = {arm: _load_receipt(path) for arm, path in paths.items()}
    receipt_sha256s = [arms[arm]["receipt_sha256"] for arm in BENCHMARK_ARM_ORDER]
    if len(set(receipt_sha256s)) != len(BENCHMARK_ARM_ORDER):
        raise OneShotDecisionError("decision triplet does not contain three receipts")

    comparison_reference: dict[str, Any] | None = None
    source_commit_reference: str | None = None
    intervals: list[tuple[int, int]] = []
    metrics: dict[str, dict[str, float]] = {}
    terminals: dict[str, tuple[dict[str, int], dict[str, str]]] = {}

    for sequence_index, arm in enumerate(BENCHMARK_ARM_ORDER):
        receipt = arms[arm]
        _fail_closed(_validated_arm_identity, receipt, expected_arm=arm)
        if (
            receipt.get("cells") != REQUIRED_CELLS
            or receipt.get("ticks") != REQUIRED_TICKS
        ):
            raise OneShotDecisionError("decision triplet requires exact B48/T160")
        if receipt.get("warmup_completed") is not True:
            raise OneShotDecisionError("execution receipt lacks completed warmup")
        if receipt.get("benchmark_arm_order") != list(BENCHMARK_ARM_ORDER):
            raise OneShotDecisionError("execution receipt benchmark arm order mismatch")
        if (
            type(receipt.get("benchmark_run_index")) is not int
            or receipt.get("benchmark_run_index") != 0
        ):
            raise OneShotDecisionError(
                "one-shot benchmark run index must be exactly zero"
            )
        if (
            type(receipt.get("benchmark_sequence_index")) is not int
            or receipt.get("benchmark_sequence_index") != sequence_index
        ):
            raise OneShotDecisionError(
                "one-shot receipts are not exact legacy/M7/M8 sequence indexes 0/1/2"
            )

        comparison = _validated_comparison_identity(receipt)
        if comparison_reference is None:
            comparison_reference = comparison
        elif comparison != comparison_reference:
            raise OneShotDecisionError(
                "plan/source/hardware/configuration/route/warmup identity drift"
            )

        source_commit = receipt.get("artifact_source_commit")
        if (
            not isinstance(source_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        ):
            raise OneShotDecisionError("execution receipt lacks frozen source commit")
        if source_commit_reference is None:
            source_commit_reference = source_commit
        elif source_commit != source_commit_reference:
            raise OneShotDecisionError(
                "artifact source commit drift across decision triplet"
            )

        interval = _validated_timestamp(receipt)
        intervals.append(interval)
        window_wall_ms, wall_ms_per_tick, ticks_per_second = _validated_window_metrics(
            receipt
        )
        metrics[arm] = {
            "native_window_wall_time_ms": window_wall_ms,
            "wall_ms_per_tick": wall_ms_per_tick,
            "ticks_per_second": ticks_per_second,
        }

        terminal = _fail_closed(_terminal_identity, receipt)
        if terminal[0] != REQUIRED_TERMINAL_CURSOR:
            raise OneShotDecisionError(
                "terminal cursor is not exact post-T160 G1:T0 boundary"
            )
        if len(terminal[1]) != REQUIRED_CELLS:
            raise OneShotDecisionError(
                "terminal trajectory projection is not exact B48"
            )
        terminals[arm] = terminal

    for preceding, following in zip(intervals[:-1], intervals[1:], strict=True):
        if following[0] < preceding[1]:
            raise OneShotDecisionError(
                "decision receipts do not prove non-overlapping A/B/C alternation"
            )

    assert comparison_reference is not None
    assert source_commit_reference is not None
    terminal_reference = terminals["legacy"]
    cursor_parity = all(
        terminals[arm][0] == terminal_reference[0] for arm in BENCHMARK_ARM_ORDER[1:]
    )
    trajectory_parity = all(
        terminals[arm][1] == terminal_reference[1] for arm in BENCHMARK_ARM_ORDER[1:]
    )
    terminal_parity = cursor_parity and trajectory_parity
    m7_speedup = (
        metrics["legacy"]["wall_ms_per_tick"]
        / metrics["m7_physical_refill"]["wall_ms_per_tick"]
    )
    m8_speedup = (
        metrics["legacy"]["wall_ms_per_tick"]
        / metrics["m8_suffix_reuse"]["wall_ms_per_tick"]
    )
    continue_campaign = terminal_parity and m8_speedup >= M8_CONTINUE_MIN_SPEEDUP

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "diagnostic_only": True,
        "qualification_claim": False,
        "promotion_claim": False,
        "required_receipts_per_arm": 1,
        "receipt_count": len(BENCHMARK_ARM_ORDER),
        "cells_per_receipt": REQUIRED_CELLS,
        "ticks_per_receipt": REQUIRED_TICKS,
        "expected_policy_callback_count": REQUIRED_CALLBACKS,
        "benchmark_arm_order": list(BENCHMARK_ARM_ORDER),
        "benchmark_run_index": 0,
        "benchmark_sequence_indexes": [0, 1, 2],
        "comparison_identity": comparison_reference,
        "artifact_source_commit": source_commit_reference,
        "identity_equality": {
            "comparison_identity_exact_across_arms": True,
            "source_plan_exact_across_arms": True,
            "execution_route_exact_across_arms": True,
            "source_exact_across_arms": True,
            "hardware_exact_across_arms": True,
            "configuration_exact_across_arms": True,
            "warmup_exact_across_arms": True,
        },
        "arm_expectations": {
            "legacy": {
                "policy_callback_count": REQUIRED_CALLBACKS,
                "physical_refill_count": REQUIRED_CALLBACKS,
                "resident_suffix_reuse_count": 0,
                "turnover_boundary_count": 0,
            },
            "m7_physical_refill": {
                "policy_callback_count": REQUIRED_CALLBACKS,
                "physical_refill_count": REQUIRED_CALLBACKS,
                "resident_suffix_reuse_count": 0,
                "turnover_boundary_count": 0,
            },
            "m8_suffix_reuse": {
                "policy_callback_count": REQUIRED_CALLBACKS,
                "physical_refill_count": 0,
                "resident_suffix_reuse_count": REQUIRED_CALLBACKS,
                "turnover_boundary_count": 0,
            },
        },
        "arms": {
            arm: {
                "receipt_path": arms[arm]["_path"],
                "receipt_sha256": arms[arm]["receipt_sha256"],
                "validated_policy_arm_identity": arms[arm]["native_result"][
                    "validated_policy_arm_identity"
                ],
                **metrics[arm],
            }
            for arm in BENCHMARK_ARM_ORDER
        },
        "semantic_equality": {
            "terminal_cursor_exact_across_arms": cursor_parity,
            "terminal_trajectory_digest_sha256s_exact_across_arms": trajectory_parity,
            "exact_parity": terminal_parity,
            "legacy_terminal_cursor": terminal_reference[0],
            "legacy_terminal_trajectory_digest_sha256s": terminal_reference[1],
            "terminal_trajectory_digest_sha256s_by_arm": {
                arm: terminals[arm][1] for arm in BENCHMARK_ARM_ORDER
            },
        },
        "speedup_over_legacy": {
            "m7_physical_refill": m7_speedup,
            "m8_suffix_reuse": m8_speedup,
        },
        "decision": {
            "result": CONTINUE_DECISION if continue_campaign else STOP_DECISION,
            "primary_metric": "legacy_wall_ms_per_tick_over_m8_wall_ms_per_tick",
            "m8_speedup_over_legacy": m8_speedup,
            "minimum_speedup_inclusive": M8_CONTINUE_MIN_SPEEDUP,
            "exact_parity_required": True,
            "exact_parity_observed": terminal_parity,
            "continue_full_campaign": continue_campaign,
            "formal_qualification_authorized_by_this_receipt": False,
            "reason": (
                "exact_parity_and_m8_whole_tick_speedup_at_or_above_1_25"
                if continue_campaign
                else (
                    "terminal_trajectory_parity_failed"
                    if not terminal_parity
                    else "m8_whole_tick_speedup_below_1_25"
                )
            ),
        },
    }
    result["reduction_sha256"] = canonical_sha256(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-receipt", type=Path, required=True)
    parser.add_argument("--m7-receipt", type=Path, required=True)
    parser.add_argument("--m8-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise OneShotDecisionError(f"refusing to overwrite reducer output: {output}")
    result = reduce_one_shot_receipts(
        legacy_path=args.legacy_receipt,
        m7_path=args.m7_receipt,
        m8_path=args.m8_receipt,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
