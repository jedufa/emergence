#!/usr/bin/env python3
"""Close the recorded B48 fixed-MLP policy-birth callback timing equation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


COMPONENT_FIELDS = (
    "device_to_host_time_ns",
    "torch_random_generation_time_ns",
    "torch_compare_pack_time_ns",
    "policy_allocation_stack_copy_time_ns",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Iterable[int]) -> dict[str, float | int]:
    samples = sorted(int(value) for value in values)
    if not samples:
        raise ValueError("at least one policy callback receipt is required")
    return {
        "count": len(samples),
        "sum_ns": sum(samples),
        "mean_ns": statistics.fmean(samples),
        "median_ns": statistics.median(samples),
        "p95_ns": (
            statistics.quantiles(samples, n=20, method="exclusive")[18]
            if len(samples) > 1
            else samples[0]
        ),
        "min_ns": samples[0],
        "max_ns": samples[-1],
    }


def _native_result(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    native = receipt.get("native_result")
    if not isinstance(native, Mapping):
        raise ValueError("execution receipt has no native_result")
    return native


def analyze_execution_receipt(path: Path) -> dict[str, Any]:
    execution = json.loads(path.read_text(encoding="utf-8"))
    native = _native_result(execution)
    rows = native.get("policy_birth_segment_refill_receipts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("execution receipt has no policy-birth callback rows")
    if int(native.get("policy_birth_segment_refill_calls_inside_tick_loop", -1)) != len(rows):
        raise ValueError("policy-birth callback count does not match its receipt rows")

    normalized: list[dict[str, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"policy-birth receipt {index} is not a mapping")
        if row.get("operation") != "refill_fixed_mlp_policy_window_v1":
            raise ValueError(f"policy-birth receipt {index} has the wrong operation")
        values = {"host_wall_time_ns": int(row["host_wall_time_ns"])}
        values.update({field: int(row[field]) for field in COMPONENT_FIELDS})
        values["host_to_device_time_ns"] = int(row["host_to_device_time_ns"])
        values["attributed_component_sum_ns"] = sum(
            values[field] for field in COMPONENT_FIELDS
        )
        values["unattributed_residual_ns"] = (
            values["host_wall_time_ns"] - values["attributed_component_sum_ns"]
        )
        if values["unattributed_residual_ns"] < 0:
            raise ValueError(f"policy-birth receipt {index} over-attributes host wall")
        normalized.append(values)

    aggregate_fields = (
        "host_wall_time_ns",
        *COMPONENT_FIELDS,
        "host_to_device_time_ns",
        "attributed_component_sum_ns",
        "unattributed_residual_ns",
    )
    distributions = {
        field: _summary(row[field] for row in normalized)
        for field in aggregate_fields
    }
    recorded_total = int(native["policy_birth_segment_host_wall_time_ns"])
    row_total = int(distributions["host_wall_time_ns"]["sum_ns"])
    if recorded_total != row_total:
        raise ValueError("policy-birth row wall time does not close to native aggregate")
    aggregate_bindings = {
        "device_to_host_time_ns": "policy_birth_segment_device_to_host_time_ns",
        "torch_random_generation_time_ns": "policy_birth_segment_torch_random_generation_time_ns",
        "torch_compare_pack_time_ns": "policy_birth_segment_torch_compare_pack_time_ns",
        "policy_allocation_stack_copy_time_ns": "policy_birth_segment_allocation_stack_copy_time_ns",
        "host_to_device_time_ns": "policy_birth_segment_host_to_device_time_ns",
    }
    for field, aggregate_field in aggregate_bindings.items():
        if int(distributions[field]["sum_ns"]) != int(native[aggregate_field]):
            raise ValueError(f"{field} rows do not close to {aggregate_field}")

    mean_host_ns = float(distributions["host_wall_time_ns"]["mean_ns"])
    shares = {
        field: float(distributions[field]["mean_ns"]) / mean_host_ns
        for field in (*COMPONENT_FIELDS, "unattributed_residual_ns")
    }
    first = rows[0]
    invariant_fields = (
        "batch_cells",
        "physical_capacity",
        "policy_row_width",
        "policy_crossover_mask_encoding",
        "policy_crossover_packed_row_bytes",
        "option_d_m4_device_policy_refill_enabled",
        "option_d_m4_device_policy_refill_authority",
    )
    invariants = {field: first[field] for field in invariant_fields}
    for field, expected in invariants.items():
        if any(row.get(field) != expected for row in rows):
            raise ValueError(f"policy-birth callback invariant changed: {field}")

    return {
        "schema_version": "eiel_b48_fixed_mlp_policy_birth_callback_analysis_v1",
        "purpose": "M0 decomposition of the recorded 481.8 ms fixed-MLP policy-birth callback before architecture selection",
        "source_execution_receipt": str(path.resolve()),
        "source_execution_receipt_sha256": _sha256(path),
        "callback_count": len(rows),
        "fixture": invariants,
        "timing_ns": distributions,
        "mean_wall_share": shares,
        "closure": {
            "recorded_host_wall_total_ns": recorded_total,
            "row_host_wall_total_ns": row_total,
            "exact_row_to_aggregate_closure": True,
            "attributed_components_are_disjoint": [
                "device_to_host_time_ns",
                "torch_random_generation_time_ns",
                "torch_compare_pack_time_ns",
                "policy_allocation_stack_copy_time_ns",
            ],
            "host_to_device_time_is_included_in_policy_allocation_stack_copy_time": True,
        },
        "architecture_gate": {
            "state": "DECOMPOSITION_REQUIRES_COMPARE_VS_PACK_SPLIT_AND_CURRENT_M4_MEASUREMENT",
            "observed_primary_bucket": "torch_compare_pack_time_ns",
            "candidate_routes": [
                "batched exact CPU generation with vectorized LSB0 packing",
                "existing exact CUDA MT19937 compact refill candidate M4",
            ],
            "selection_prohibited_by_this_receipt_alone": True,
        },
        "scope_limit": "recorded B48 T160 M0 callback rows; no candidate speedup or qualification claim",
    }


def _markdown(receipt: Mapping[str, Any]) -> str:
    timing = receipt["timing_ns"]
    shares = receipt["mean_wall_share"]
    rows = [
        ("Complete callback", "host_wall_time_ns"),
        ("Torch compare plus LSB0 pack", "torch_compare_pack_time_ns"),
        ("Torch random generation", "torch_random_generation_time_ns"),
        ("Allocation, stack, and H2D", "policy_allocation_stack_copy_time_ns"),
        ("D2H state and cursors", "device_to_host_time_ns"),
        ("Unattributed residual", "unattributed_residual_ns"),
    ]
    lines = [
        "# Fixed-MLP policy-birth callback M0 decomposition",
        "",
        f"Source callbacks: {receipt['callback_count']} physical-B48 boundaries.",
        "",
        "| Component | Mean ms | Median ms | p95 ms | Wall share |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, field in rows:
        summary = timing[field]
        share = 1.0 if field == "host_wall_time_ns" else shares[field]
        lines.append(
            f"| {label} | {summary['mean_ns'] / 1e6:.3f} | "
            f"{summary['median_ns'] / 1e6:.3f} | {summary['p95_ns'] / 1e6:.3f} | {share:.2%} |"
        )
    lines.extend(
        [
            "",
            "Architecture remains deliberately unselected: compare and pack must be split, then the current exact M4 candidate must be measured on the same B48 surface.",
            "",
            "The H2D time is already included in the allocation/stack/copy bucket and is not double-counted.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable callback analysis")
    receipt = analyze_execution_receipt(args.execution_receipt)
    args.output_root.mkdir(parents=True)
    output = args.output_root / "m0_policy_birth_callback_analysis.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.output_root / "m0_policy_birth_callback_analysis.md").write_text(
        _markdown(receipt)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
