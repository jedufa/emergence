#!/usr/bin/env python3
"""Reduce immutable alternating legacy/M7/M8 whole-tick execution receipts.

The reducer is diagnostic-only.  It proves arm identity and terminal-state
equality before reporting performance, and it never fills in unavailable
resource metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402


SCHEMA_VERSION = "eiel_gpu_tick_ab_reducer_v1"
AUTHORITY = "diagnostic_performance_comparison_only_not_qualification_v1"
REJECT_MAX_SPEEDUP = 1.03
ELIGIBLE_MIN_SPEEDUP = 1.10
M7_ENABLED_FIELD = "option_d_m7_batched_cpu_policy_refill_enabled"
M7_USED_FIELD = "option_d_m7_batched_cpu_policy_refill_used"
M7_AUTHORITY_FIELD = "option_d_m7_batched_cpu_policy_refill_authority"
M7_SCHEMA_FIELD = "option_d_m7_batched_cpu_policy_refill_schema_version"
M7_RUNTIME_SCHEMA = "eiel_option_d_m7_batched_cpu_policy_refill_v1"
M7_BUILD_SCHEMA = "eiel_option_d_m7_batched_cpu_policy_refill_build_identity_v1"
M8_ENABLED_FIELD = "option_d_m8_resident_policy_suffix_reuse_enabled"
M8_BUILD_SCHEMA = "eiel_option_d_m8_resident_policy_suffix_reuse_build_identity_v1"
COMPONENT_GATE_SCHEMA = "eiel_b48_fixed_mlp_policy_callback_gate_v1"
COMPONENT_GATE_AUTHORITY = "fixed_mlp_component_callback_promotion_gate_v1"
BENCHMARK_ARM_ORDER = ("legacy", "m7_physical_refill", "m8_suffix_reuse")
REQUIRED_REPEATS_PER_ARM = 5
REQUIRED_CELLS = 48
REQUIRED_TICKS = 160
REQUIRED_CALLBACKS = REQUIRED_TICKS - 1
COMPARISON_IDENTITY_FIELDS = {
    "source_plan_sha256",
    "execution_contract_identity_sha256",
    "source_identity_sha256",
    "device_identity_sha256",
    "configuration_identity_sha256",
    "warmup_identity_sha256",
    "warmup_identity",
    "identity_sha256",
}
RESOURCE_FIELDS = {
    "torch_peak_allocated_bytes_during_window": "bytes",
    "torch_peak_reserved_bytes_during_window": "bytes",
    "process_peak_rss_bytes": "bytes",
    "maximum_resident_set_size_kib": "KiB",
    "nvidia_smi_peak_used_gpu_memory_mib": "MiB",
}


class ABReductionError(RuntimeError):
    """Raised when an A/B receipt set cannot support an exact comparison."""


def _validated_build_identity(
    value: Any,
    *,
    schema: str,
    enabled_field: str,
    expected_enabled: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ABReductionError(f"benchmark receipt lacks {schema} build identity")
    identity = dict(value)
    observed_sha256 = identity.pop("identity_sha256", None)
    macro = {
        M7_ENABLED_FIELD: "EIEL_OPTION_D_M7_BATCHED_CPU_POLICY_REFILL_ENABLED",
        M8_ENABLED_FIELD: "EIEL_OPTION_D_M8_RESIDENT_POLICY_SUFFIX_REUSE_ENABLED",
    }[enabled_field]
    expected_authority = {
        M7_ENABLED_FIELD: "candidate_batched_exact_cpu_lsb0_policy_refill",
        M8_ENABLED_FIELD: "candidate_resident_policy_suffix_reuse",
    }[enabled_field]
    runtime_sha256 = identity.get("runtime_resource_receipt_sha256")
    if observed_sha256 != canonical_sha256(identity):
        raise ABReductionError(f"benchmark {schema} identity digest mismatch")
    if (
        identity.get("schema_version") != schema
        or identity.get(enabled_field) is not expected_enabled
        or identity.get("compile_time_macro") != macro
        or identity.get("receipt_field") != enabled_field
        or identity.get("default_enabled") is not False
        or identity.get("compile_definition")
        != f"-D{macro}={1 if expected_enabled else 0}"
        or identity.get("authority")
        != (expected_authority if expected_enabled else "disabled")
        or identity.get("identity_authority") != "compiled_binary_resource_receipt"
        or identity.get("runtime_resource_receipt_confirmed") is not True
        or not isinstance(runtime_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None
    ):
        raise ABReductionError(f"benchmark has wrong {schema} build identity")
    identity["identity_sha256"] = observed_sha256
    return identity


def _load_component_gate(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ABReductionError(f"component callback gate is absent: {resolved}")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ABReductionError("component callback gate is not a mapping")
    gate = dict(raw)
    observed_sha256 = gate.pop("receipt_sha256", None)
    if observed_sha256 != canonical_sha256(gate):
        raise ABReductionError("component callback gate digest mismatch")
    gate["receipt_sha256"] = observed_sha256
    source_commit = gate.get("artifact_source_commit")
    hardware = gate.get("hardware_identity")
    surface = gate.get("component_surface")
    arms = gate.get("arms")
    if (
        gate.get("schema_version") != COMPONENT_GATE_SCHEMA
        or gate.get("authority") != COMPONENT_GATE_AUTHORITY
        or gate.get("status") != "success"
        or gate.get("required_repeats_per_arm") != REQUIRED_REPEATS_PER_ARM
        or gate.get("minimum_median_speedup_inclusive") != ELIGIBLE_MIN_SPEEDUP
        or type(gate.get("callback_median_gate_passed")) is not bool
        or not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or not isinstance(surface, Mapping)
        or surface.get("cells") != REQUIRED_CELLS
        or surface.get("episodes") != 124
        or surface.get("policy_width") != 9_991
        or surface.get("packed_row_bytes") != 1_249
        or surface.get("mutation_enable") is not False
        or surface.get("mutation_mode") != "dense"
        or surface.get("compact_mutation_width") != 1
        or gate.get("exact_output_digest_match_across_arms") is not True
        or gate.get("m8_build_binds_m7_exact_physical_refill_fallback") is not True
        or not isinstance(hardware, Mapping)
        or not isinstance(arms, Mapping)
        or set(arms) != set(BENCHMARK_ARM_ORDER)
    ):
        raise ABReductionError("component callback gate contract is invalid")
    hardware_payload = dict(hardware)
    hardware_sha256 = hardware_payload.pop("identity_sha256", None)
    if hardware_sha256 != canonical_sha256(hardware_payload):
        raise ABReductionError("component callback gate hardware digest mismatch")
    if gate.get("host_profiler_source_commit") != source_commit:
        raise ABReductionError("component callback gate source binding mismatch")
    control_digest = gate.get("control_output_digest")
    if (
        not isinstance(control_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", control_digest) is None
    ):
        raise ABReductionError("component callback gate lacks control digest")
    medians: dict[str, float] = {}
    for arm in BENCHMARK_ARM_ORDER:
        arm_row = arms[arm]
        build = arm_row.get("build_identity") if isinstance(arm_row, Mapping) else None
        if not isinstance(build, Mapping):
            raise ABReductionError("component callback gate lacks arm build identity")
        for field in (
            "profile_receipt_sha256",
            "artifact_sha256",
            "manifest_sha256",
            "compiled_source_snapshot_sha256",
        ):
            value = arm_row.get(field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ABReductionError(
                    f"component callback gate arm lacks exact {field}"
                )
        _validated_build_identity(
            build.get("m7"),
            schema=M7_BUILD_SCHEMA,
            enabled_field=M7_ENABLED_FIELD,
            expected_enabled=arm != "legacy",
        )
        _validated_build_identity(
            build.get("m8"),
            schema=M8_BUILD_SCHEMA,
            enabled_field=M8_ENABLED_FIELD,
            expected_enabled=arm == "m8_suffix_reuse",
        )
        timing = arm_row.get("host_wall_time_ns")
        if (
            not isinstance(timing, Mapping)
            or timing.get("count") != REQUIRED_REPEATS_PER_ARM
            or isinstance(timing.get("median_ns"), bool)
            or not isinstance(timing.get("median_ns"), (int, float))
            or not math.isfinite(float(timing["median_ns"]))
            or float(timing["median_ns"]) <= 0.0
        ):
            raise ABReductionError("component callback gate arm timing is invalid")
        medians[arm] = float(timing["median_ns"])
    expected_speedups = {
        arm: medians["legacy"] / medians[arm]
        for arm in ("m7_physical_refill", "m8_suffix_reuse")
    }
    observed_speedups = gate.get("median_speedup_over_legacy")
    if (
        not isinstance(observed_speedups, Mapping)
        or set(observed_speedups) != set(expected_speedups)
        or any(
            isinstance(observed_speedups[arm], bool)
            or not isinstance(observed_speedups[arm], (int, float))
            or not math.isclose(
                float(observed_speedups[arm]),
                expected_speedups[arm],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for arm in expected_speedups
        )
        or gate["callback_median_gate_passed"]
        is not all(
            value >= ELIGIBLE_MIN_SPEEDUP for value in expected_speedups.values()
        )
    ):
        raise ABReductionError("component callback gate decision is inconsistent")
    gate["_path"] = str(resolved)
    return gate


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ABReductionError("cannot summarize an empty metric")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ABReductionError("metric contains a non-finite value")
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower]
    if upper != lower:
        p95 += (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": p95,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _load_immutable_receipt(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ABReductionError(f"execution receipt is absent: {resolved}")
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ABReductionError(f"execution receipt is not a mapping: {resolved}")
    receipt = dict(raw)
    observed_sha256 = receipt.pop("receipt_sha256", None)
    expected_sha256 = canonical_sha256(receipt)
    if observed_sha256 != expected_sha256:
        raise ABReductionError(f"execution receipt digest mismatch: {resolved}")
    receipt["receipt_sha256"] = observed_sha256
    if receipt.get("status") != "success":
        raise ABReductionError(f"execution receipt is not successful: {resolved}")
    if not str(receipt.get("schema_version", "")).startswith(
        "eiel_gpu_tick_anatomy_v1_execution_receipt_"
    ):
        raise ABReductionError(f"unexpected execution receipt schema: {resolved}")
    receipt["_path"] = str(resolved)
    return receipt


def _validated_arm_identity(
    receipt: Mapping[str, Any], *, expected_arm: str
) -> dict[str, Any]:
    expected_flags = {
        "legacy": (False, False),
        "m7_physical_refill": (True, False),
        "m8_suffix_reuse": (True, True),
    }
    if expected_arm not in expected_flags:
        raise ABReductionError(f"unknown benchmark arm: {expected_arm}")
    expected_m7, expected_m8 = expected_flags[expected_arm]
    m7 = receipt.get("m7_arm_identity")
    m8 = receipt.get("m8_arm_identity")
    native_result = receipt.get("native_result")
    if not all(isinstance(row, Mapping) for row in (m7, m8, native_result)):
        raise ABReductionError("execution receipt lacks complete M7/M8 arm identity")
    m7_manifest = m7.get("manifest")
    m8_manifest = m8.get("manifest")
    m7_runtime = m7.get("runtime")
    m8_runtime = m8.get("runtime")
    if not all(
        isinstance(row, Mapping)
        for row in (m7_manifest, m8_manifest, m7_runtime, m8_runtime)
    ):
        raise ABReductionError("execution receipt has incomplete M7/M8 arm identity")
    m7_manifest = _validated_build_identity(
        m7_manifest,
        schema=M7_BUILD_SCHEMA,
        enabled_field=M7_ENABLED_FIELD,
        expected_enabled=expected_m7,
    )
    m8_manifest = _validated_build_identity(
        m8_manifest,
        schema=M8_BUILD_SCHEMA,
        enabled_field=M8_ENABLED_FIELD,
        expected_enabled=expected_m8,
    )
    expected_runtime = {
        "expected_m7_enabled": expected_m7,
        "expected_m8_enabled": expected_m8,
        "receipt_count": REQUIRED_CALLBACKS,
        "resident_suffix_reuse_receipt_count": (
            REQUIRED_CALLBACKS if expected_m8 else 0
        ),
        "physical_refill_receipt_count": 0 if expected_m8 else REQUIRED_CALLBACKS,
        "turnover_boundary_receipt_count": 0,
        "m7_used_on_every_physical_refill": expected_m7,
        "m7_unused_on_every_suffix_reuse": True,
    }
    mismatches: dict[str, Any] = {}
    expected_identity = {
        "receipt.benchmark_arm": expected_arm,
        "m7.expected_enabled": expected_m7,
        "m8.arm": expected_arm,
        "m8.expected_enabled": expected_m8,
        f"m7.manifest.{M7_ENABLED_FIELD}": expected_m7,
        f"m8.manifest.{M8_ENABLED_FIELD}": expected_m8,
    }
    observed_identity = {
        "receipt.benchmark_arm": receipt.get("benchmark_arm"),
        "m7.expected_enabled": m7.get("expected_enabled"),
        "m8.arm": m8.get("arm"),
        "m8.expected_enabled": m8.get("expected_enabled"),
        f"m7.manifest.{M7_ENABLED_FIELD}": m7_manifest.get(M7_ENABLED_FIELD),
        f"m8.manifest.{M8_ENABLED_FIELD}": m8_manifest.get(M8_ENABLED_FIELD),
    }
    for name, expected in expected_identity.items():
        if observed_identity[name] != expected:
            mismatches[name] = {
                "expected": expected,
                "observed": observed_identity[name],
            }
    for runtime_name, runtime in (
        ("m7.runtime", m7_runtime),
        ("m8.runtime", m8_runtime),
    ):
        for name, expected in expected_runtime.items():
            if runtime.get(name) != expected:
                mismatches[f"{runtime_name}.{name}"] = {
                    "expected": expected,
                    "observed": runtime.get(name),
                }
    if dict(m7_runtime) != dict(m8_runtime):
        mismatches["arm_runtime_equality"] = {
            "expected": "exact",
            "observed": "different",
        }
    if native_result.get("validated_policy_arm_identity") != m7_runtime:
        mismatches["native_result.validated_policy_arm_identity"] = {
            "expected": dict(m7_runtime),
            "observed": native_result.get("validated_policy_arm_identity"),
        }
    for field in (
        "turnover_shadow_plan_launches",
        "turnover_cull_refill_plan_launches",
        "turnover_commit_launches",
        "turnover_publish_launches",
    ):
        if native_result.get(field) != 0:
            mismatches[f"native_result.{field}"] = {
                "expected": 0,
                "observed": native_result.get(field),
            }
    if mismatches:
        raise ABReductionError(
            "execution receipt M7/M8 arm identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "m7": {**dict(m7), "manifest": m7_manifest},
        "m8": {**dict(m8), "manifest": m8_manifest},
    }


def _comparison_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    raw = receipt.get("comparison_identity")
    if not isinstance(raw, Mapping) or set(raw) != COMPARISON_IDENTITY_FIELDS:
        raise ABReductionError("execution receipt lacks exact comparison identity")
    identity = dict(raw)
    observed_sha256 = identity.pop("identity_sha256", None)
    if observed_sha256 != canonical_sha256(identity):
        raise ABReductionError("comparison identity digest mismatch")
    identity["identity_sha256"] = observed_sha256
    return identity


def _positive_metric(receipt: Mapping[str, Any], name: str) -> float:
    value = receipt.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ABReductionError(f"execution receipt lacks numeric {name}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ABReductionError(f"execution receipt has invalid {name}")
    return result


def _terminal_identity(
    receipt: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, str]]:
    cursor = receipt.get("terminal_cursor")
    digests = receipt.get("terminal_trajectory_digest_sha256s")
    if (
        not isinstance(cursor, Mapping)
        or set(cursor) != {"generation", "tick"}
        or any(type(value) is not int for value in cursor.values())
    ):
        raise ABReductionError("execution receipt lacks exact terminal_cursor")
    if not isinstance(digests, Mapping) or not digests:
        raise ABReductionError(
            "execution receipt lacks terminal trajectory digest projection"
        )
    normalized: dict[str, str] = {}
    for cell_id, digest in digests.items():
        if (
            not isinstance(cell_id, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ABReductionError("terminal trajectory digest map is malformed")
        normalized[cell_id] = digest
    return ({name: int(value) for name, value in cursor.items()}, normalized)


def _resource_reduction(
    arms: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available: dict[str, Any] = {}
    unavailable: list[dict[str, Any]] = []
    for field, unit in RESOURCE_FIELDS.items():
        values_by_arm = {
            arm: [row.get(field) for row in rows] for arm, rows in arms.items()
        }
        all_values = [value for values in values_by_arm.values() for value in values]
        valid = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in all_values
        )
        if not valid:
            unavailable.append(
                {
                    "field": field,
                    "present_by_arm": {
                        arm: sum(value is not None for value in values)
                        for arm, values in values_by_arm.items()
                    },
                    "required_per_arm": len(next(iter(arms.values()))),
                }
            )
            continue
        floats_by_arm = {
            arm: [float(value) for value in values]
            for arm, values in values_by_arm.items()
        }
        available[field] = {
            "unit": unit,
            **{arm: _stats(values) for arm, values in floats_by_arm.items()},
            "m7_minus_legacy_paired": _stats(
                [
                    m7 - legacy
                    for legacy, m7 in zip(
                        floats_by_arm["legacy"],
                        floats_by_arm["m7_physical_refill"],
                        strict=True,
                    )
                ]
            ),
            "m8_minus_legacy_paired": _stats(
                [
                    m8 - legacy
                    for legacy, m8 in zip(
                        floats_by_arm["legacy"],
                        floats_by_arm["m8_suffix_reuse"],
                        strict=True,
                    )
                ]
            ),
        }
    return available, unavailable


def reduce_ab_receipts(
    *,
    legacy_paths: Sequence[Path],
    m7_paths: Sequence[Path],
    m8_paths: Sequence[Path],
    component_gate_path: Path,
) -> dict[str, Any]:
    component_gate = _load_component_gate(component_gate_path)
    paths_by_arm = {
        "legacy": legacy_paths,
        "m7_physical_refill": m7_paths,
        "m8_suffix_reuse": m8_paths,
    }
    counts = {arm: len(paths) for arm, paths in paths_by_arm.items()}
    if set(counts.values()) != {REQUIRED_REPEATS_PER_ARM}:
        raise ABReductionError(
            f"exactly {REQUIRED_REPEATS_PER_ARM} receipts per arm are required: "
            + json.dumps(counts, sort_keys=True)
        )
    arms = {
        arm: [_load_immutable_receipt(path) for path in paths]
        for arm, paths in paths_by_arm.items()
    }
    receipt_digests = [
        receipt["receipt_sha256"] for receipts in arms.values() for receipt in receipts
    ]
    if len(set(receipt_digests)) != len(receipt_digests):
        raise ABReductionError("benchmark executions are not independent receipts")

    reference_comparison: dict[str, Any] | None = None
    chronological_receipts: list[Mapping[str, Any]] = []
    for arm_index, arm in enumerate(BENCHMARK_ARM_ORDER):
        for run_index, receipt in enumerate(arms[arm]):
            arm_identity = _validated_arm_identity(receipt, expected_arm=arm)
            gate_arm = component_gate["arms"][arm]
            gate_build = gate_arm["build_identity"]
            if (
                arm_identity["m7"]["manifest"] != gate_build["m7"]
                or arm_identity["m8"]["manifest"] != gate_build["m8"]
            ):
                raise ABReductionError(
                    "whole-tick candidate identity differs from component gate"
                )
            if (
                receipt.get("artifact_source_commit")
                != component_gate["artifact_source_commit"]
                or receipt.get("artifact_sha256") != gate_arm["artifact_sha256"]
                or receipt.get("manifest_sha256") != gate_arm["manifest_sha256"]
            ):
                raise ABReductionError(
                    "whole-tick artifact/source binding differs from component gate"
                )
            if (
                receipt.get("cells") != REQUIRED_CELLS
                or receipt.get("ticks") != REQUIRED_TICKS
            ):
                raise ABReductionError("promotion comparison requires exact B48/T160")
            if receipt.get("warmup_completed") is not True:
                raise ABReductionError(
                    "execution receipt lacks completed warmup identity"
                )
            if receipt.get("benchmark_arm_order") != list(BENCHMARK_ARM_ORDER):
                raise ABReductionError("execution receipt benchmark arm order mismatch")
            if receipt.get("benchmark_run_index") != run_index:
                raise ABReductionError(
                    "benchmark run indices must be contiguous per arm"
                )
            expected_sequence_index = run_index * len(BENCHMARK_ARM_ORDER) + arm_index
            if receipt.get("benchmark_sequence_index") != expected_sequence_index:
                raise ABReductionError(
                    "benchmark receipts are not exact alternating legacy/M7/M8 runs"
                )
            chronological_receipts.append(receipt)
            comparison = _comparison_identity(receipt)
            if reference_comparison is None:
                reference_comparison = comparison
            elif comparison != reference_comparison:
                raise ABReductionError(
                    "plan/execution/source/device/configuration/warmup identity drift"
                )
    assert reference_comparison is not None
    if (
        component_gate["hardware_identity"].get("identity_sha256")
        != reference_comparison["device_identity_sha256"]
    ):
        raise ABReductionError(
            "whole-tick hardware identity differs from component callback gate"
        )
    chronological_receipts.sort(key=lambda row: int(row["benchmark_sequence_index"]))
    previous_completed: int | None = None
    for receipt in chronological_receipts:
        started = receipt.get("started_at_unix_ns")
        completed = receipt.get("completed_at_unix_ns")
        if (
            type(started) is not int
            or type(completed) is not int
            or started >= completed
            or (previous_completed is not None and started < previous_completed)
        ):
            raise ABReductionError(
                "benchmark receipts do not prove independent non-overlapping alternation"
            )
        previous_completed = completed

    triplets: list[dict[str, Any]] = []
    terminal_reference: tuple[dict[str, int], dict[str, str]] | None = None
    values: dict[str, dict[str, list[float]]] = {
        metric: {arm: [] for arm in BENCHMARK_ARM_ORDER}
        for metric in ("native_window_wall_time_ms", "wall_ms_per_tick")
    }
    for run_index in range(REQUIRED_REPEATS_PER_ARM):
        run_terminal: tuple[dict[str, int], dict[str, str]] | None = None
        rows: dict[str, Any] = {"run_index": run_index}
        for arm in BENCHMARK_ARM_ORDER:
            receipt = arms[arm][run_index]
            terminal = _terminal_identity(receipt)
            if terminal[0] != {"generation": 1, "tick": REQUIRED_TICKS}:
                raise ABReductionError("terminal cursor is not exact T160 boundary")
            if len(terminal[1]) != REQUIRED_CELLS:
                raise ABReductionError(
                    "terminal trajectory projection is not exact B48"
                )
            if run_terminal is None:
                run_terminal = terminal
            elif terminal != run_terminal:
                raise ABReductionError(
                    f"three-arm terminal cursor/trajectory mismatch at run {run_index}"
                )
            if terminal_reference is None:
                terminal_reference = terminal
            elif terminal != terminal_reference:
                raise ABReductionError("terminal trajectory differs between repeats")
            for metric in values:
                values[metric][arm].append(_positive_metric(receipt, metric))
            rows[arm] = {
                "receipt_path": receipt["_path"],
                "receipt_sha256": receipt["receipt_sha256"],
                "benchmark_sequence_index": receipt["benchmark_sequence_index"],
            }
        assert run_terminal is not None
        rows["terminal_cursor"] = run_terminal[0]
        rows["terminal_trajectory_digest_sha256s"] = run_terminal[1]
        triplets.append(rows)

    wall = values["wall_ms_per_tick"]
    world_tps = {
        arm: [1000.0 / value for value in wall[arm]] for arm in BENCHMARK_ARM_ORDER
    }
    m8_speedup = statistics.median(wall["legacy"]) / statistics.median(
        wall["m8_suffix_reuse"]
    )
    callback_gate_passed = component_gate["callback_median_gate_passed"] is True
    if m8_speedup <= REJECT_MAX_SPEEDUP:
        band, eligible, reason = "reject", False, "whole_tick_speedup_at_or_below_1_03"
    elif m8_speedup < ELIGIBLE_MIN_SPEEDUP:
        band, eligible, reason = (
            "marginal_no_automatic_promotion",
            False,
            "whole_tick_speedup_above_1_03_but_below_1_10",
        )
    elif not callback_gate_passed:
        band, eligible, reason = (
            "whole_tick_win_but_callback_gate_failed",
            False,
            "predeclared_callback_median_gate_not_proven",
        )
    else:
        band, eligible, reason = (
            "eligible_subject_to_maintainability_and_qualification",
            True,
            "whole_tick_and_callback_gates_passed",
        )

    def metric_projection(
        metric_values: Mapping[str, Sequence[float]],
    ) -> dict[str, Any]:
        projection = {arm: _stats(metric_values[arm]) for arm in BENCHMARK_ARM_ORDER}
        projection["paired_legacy_over_m7_speedup"] = _stats(
            [
                legacy / m7
                for legacy, m7 in zip(
                    metric_values["legacy"],
                    metric_values["m7_physical_refill"],
                    strict=True,
                )
            ]
        )
        projection["paired_legacy_over_m8_speedup"] = _stats(
            [
                legacy / m8
                for legacy, m8 in zip(
                    metric_values["legacy"],
                    metric_values["m8_suffix_reuse"],
                    strict=True,
                )
            ]
        )
        return projection

    resource_metrics, unavailable_resources = _resource_reduction(arms)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "required_repeats_per_arm": REQUIRED_REPEATS_PER_ARM,
        "receipt_count": len(receipt_digests),
        "ticks_per_receipt": REQUIRED_TICKS,
        "cells_per_receipt": REQUIRED_CELLS,
        "alternation_proven": True,
        "independent_receipts_proven": True,
        "comparison_identity": reference_comparison,
        "component_callback_gate": {
            "path": component_gate["_path"],
            "receipt_sha256": component_gate["receipt_sha256"],
            "artifact_source_commit": component_gate["artifact_source_commit"],
            "hardware_identity_sha256": component_gate["hardware_identity"][
                "identity_sha256"
            ],
            "callback_median_gate_passed": callback_gate_passed,
        },
        "semantic_equality": {
            "terminal_cursor_exact_across_all_arms_and_repeats": True,
            "terminal_trajectory_digest_sha256s_exact_across_all_arms_and_repeats": True,
        },
        "arms": {
            arm: {
                "expected_m7_enabled": arm != "legacy",
                "expected_m8_enabled": arm == "m8_suffix_reuse",
                "receipt_count": len(arms[arm]),
            }
            for arm in BENCHMARK_ARM_ORDER
        },
        "metrics": {
            "native_window_wall_time_ms": metric_projection(
                values["native_window_wall_time_ms"]
            ),
            "wall_ms_per_tick": metric_projection(wall),
            "world_ticks_per_second": metric_projection(world_tps),
        },
        "resource_metrics": resource_metrics,
        "unavailable_or_partial_resource_metrics": unavailable_resources,
        "decision": {
            "primary_metric": "ratio_of_arm_medians_legacy_over_m8_wall_ms_per_tick",
            "primary_speedup": m8_speedup,
            "callback_median_gate_passed": callback_gate_passed,
            "reject_max_speedup_inclusive": REJECT_MAX_SPEEDUP,
            "eligible_min_speedup_inclusive": ELIGIBLE_MIN_SPEEDUP,
            "band": band,
            "performance_eligible": eligible,
            "automatic_promotion_authorized_by_this_receipt": False,
            "reason": reason,
        },
        "triplets": triplets,
        "qualification_claim": False,
        "promotion_claim": False,
    }
    result["reduction_sha256"] = canonical_sha256(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-receipt", type=Path, action="append", required=True)
    parser.add_argument("--m7-receipt", type=Path, action="append", required=True)
    parser.add_argument("--m8-receipt", type=Path, action="append", required=True)
    parser.add_argument("--component-gate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ABReductionError(f"refusing to overwrite reducer output: {output}")
    result = reduce_ab_receipts(
        legacy_paths=args.legacy_receipt,
        m7_paths=args.m7_receipt,
        m8_paths=args.m8_receipt,
        component_gate_path=args.component_gate_receipt,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
