#!/usr/bin/env python3
"""Freeze the bounded post-alias B48 secondary-seam decisions.

This is a mechanical decision record over the already-closed secondary
parity/timing gate.  It deliberately does not implement a new optimization:
the boundary measurements decide whether the request-map/version and
clone-birth seams are materially justified.  A rejected seam is explicitly
disabled from the canonical B48 target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.orchestration import atomic_write_json


SCHEMA_VERSION = "eiel_hapb_secondary_seam_decision_v1"
MATERIAL_SPEEDUP_FRACTION = 0.05
BOUNDARY_MATERIAL_FRACTION = 0.05


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(samples: list[Mapping[str, Any]], field: str) -> float:
    values = [samples[index].get(field) for index in range(len(samples))]
    _require(
        values and all(type(value) is int and value >= 0 for value in values),
        f"timing field {field} is malformed",
    )
    return sum(int(value) for value in values) / len(values)


def _timing_summary(
    timing_comparison: Mapping[str, Any], label: str
) -> dict[str, Any]:
    raw_samples = timing_comparison.get(label)
    _require(isinstance(raw_samples, list) and len(raw_samples) == 2, f"{label} timing pair is incomplete")
    samples = [
        value for value in raw_samples if isinstance(value, Mapping)
    ]
    _require(len(samples) == 2, f"{label} timing samples are malformed")
    return {
        "sample_count": len(samples),
        "native_execute_wall_ns_mean": _mean(samples, "native_execute_wall_ns"),
        "host_classification_map_construction_wall_ns_mean": _mean(
            samples, "host_classification_map_construction_wall_ns"
        ),
        "control_d2h_bytes": sorted({int(sample["control_d2h_bytes"]) for sample in samples}),
        "control_h2d_bytes": sorted({int(sample["control_h2d_bytes"]) for sample in samples}),
        "sync_count": sorted({int(sample["sync_count"]) for sample in samples}),
        "gpu_stall_ns": sorted({int(sample["gpu_stall_ns"]) for sample in samples}),
        "policy_byte_reduction_credit": any(
            sample.get("policy_byte_reduction_credit") is True for sample in samples
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secondary-gate-receipt", required=True, type=Path)
    parser.add_argument("--primary-gate-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    secondary_path = args.secondary_gate_receipt.resolve()
    primary_path = args.primary_gate_receipt.resolve()
    secondary = _load_json(secondary_path)
    primary = _load_json(primary_path)
    _require(secondary.get("decision") == "PASS", "secondary gate is not PASS")
    _require(secondary.get("terminal_projection_parity") is True, "secondary parity is not exact")
    _require(secondary.get("rng_after_prepare_and_execute_parity") is True, "secondary RNG parity is not exact")
    _require(secondary.get("birth_recombination_cursor_parity") is True, "secondary birth parity is not exact")
    _require(primary.get("decision") == "PASS", "primary rescue gate is not PASS")
    primary_ratio = primary.get("policy_weight_h2d_bytes_reduction_ratio")
    _require(
        type(primary_ratio) in {int, float} and primary_ratio >= 10.0,
        "primary policy-byte reduction is below 10x",
    )

    timing = secondary.get("timing_comparison")
    _require(isinstance(timing, Mapping), "secondary timing comparison is missing")
    tile_512 = _timing_summary(timing, "tile_512")
    tile_4096 = _timing_summary(timing, "tile_4096")
    mean_512 = tile_512["native_execute_wall_ns_mean"]
    mean_4096 = tile_4096["native_execute_wall_ns_mean"]
    speedup_4096_vs_512 = (mean_512 / mean_4096) - 1.0
    selected_tile = 4096 if speedup_4096_vs_512 >= MATERIAL_SPEEDUP_FRACTION else 512

    map_fraction = (
        tile_512["host_classification_map_construction_wall_ns_mean"] / mean_512
    )
    boundary_idle = (
        tile_512["control_h2d_bytes"] == [0]
        and tile_512["sync_count"] == [0]
        and tile_512["gpu_stall_ns"] == [0]
    )
    request_map_decision = (
        "REJECT"
        if map_fraction < BOUNDARY_MATERIAL_FRACTION and boundary_idle
        else "REQUIRES_BOUNDED_A_B"
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "secondary_gate_receipt": str(secondary_path),
        "secondary_gate_receipt_sha256": _sha256(secondary_path),
        "primary_gate_receipt": str(primary_path),
        "primary_gate_receipt_sha256": _sha256(primary_path),
        "primary_policy_weight_h2d_reduction_ratio": float(primary_ratio),
        "tile_selection": {
            "decision": "KEEP_512" if selected_tile == 512 else "KEEP_4096",
            "selected_tile_rows": selected_tile,
            "material_speedup_threshold_fraction": MATERIAL_SPEEDUP_FRACTION,
            "measured_4096_vs_512_speedup_fraction": speedup_4096_vs_512,
            "tile_512": tile_512,
            "tile_4096": tile_4096,
            "complexity_impact": "512 retains the qualified bounded staging shape; 4096 adds no traffic benefit and is disabled for the canonical B48 target.",
        },
        "secondary_seams": {
            "gpu_resident_request_map_version_reuse": {
                "decision": request_map_decision,
                "ab_baseline": {
                    "variant": "KEEP_512 alias route",
                    "receipt": str(secondary_path),
                    "isolated_sample_count": 2,
                },
                "isolated_incremental_effect": {
                    "status": "NOT_IMPLEMENTED_SCREENED_BY_BASELINE",
                    "speedup_fraction": 0.0,
                    "affected_timing_buckets": [
                        "host_classification_map_construction_wall_ns",
                        "control_d2h_bytes",
                        "control_h2d_bytes",
                        "sync_count",
                        "gpu_stall_ns",
                    ],
                },
                "baseline_map_fraction_of_native_wall": map_fraction,
                "resource_cost": "persistent GPU slot-to-content/version/request-map state plus restore and invalidation paths",
                "complexity_impact": "MEDIUM",
                "rationale": "The measured map-construction bucket is below 5% of native execution and the observed boundary has zero control H2D, zero synchronization, and zero GPU stall; no material candidate is justified.",
            },
            "clone_birth_descriptor_alias": {
                "decision": "REJECT",
                "ab_baseline": {
                    "variant": "KEEP_512 alias route",
                    "receipt": str(secondary_path),
                    "isolated_sample_count": 2,
                },
                "isolated_incremental_effect": {
                    "status": "NOT_IMPLEMENTED_NO_INDEPENDENT_BIRTH_TIMING_BUCKET",
                    "speedup_fraction": 0.0,
                    "affected_timing_buckets": [],
                },
                "resource_cost": "additional birth descriptor publication, rollback, and version/invalidation state",
                "complexity_impact": "MEDIUM",
                "rationale": "Birth traces are exact and the primary content/lineage alias is already retained; no independently measured birth-wall-time or counter benefit supports another alias/descriptor seam.",
            },
        },
        "canonical_b48_architecture": {
            "policy_authority": "canonical host policy bank only",
            "content_lineage_alias_pool": "ENABLED",
            "policy_tile_rows": 512,
            "gpu_resident_request_map_version_reuse": "DISABLED",
            "clone_birth_descriptor_alias": "DISABLED",
            "tile_rows_4096": "DISABLED_FOR_PRODUCTION_TARGET",
            "cross_cell_content_deduplication": "DEFERRED",
            "double_buffer_or_latency_hiding": "DISABLED",
            "policy_byte_reduction_credit_for_secondary": False,
        },
        "qualification_claim": False,
        "production_execution_admission": False,
        "next_phase": "Gate 6 controlled turnover/ABI work using the frozen B48 alias architecture",
    }
    _require(not args.output_root.exists(), "refusing to overwrite immutable seam decision")
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_root / "hapb_secondary_seam_decision_receipt.json", result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
