#!/usr/bin/env python3
"""Select the next policy-birth refill architecture from same-surface M0 data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"receipt is not a mapping: {path}")
    return value


def _mean(receipt: Mapping[str, Any], field: str) -> float:
    return float(receipt["timing_ns"][field]["mean_ns"])


def _median(receipt: Mapping[str, Any], field: str) -> float:
    return float(receipt["timing_ns"][field]["median_ns"])


def decide_architecture(
    anatomy: Mapping[str, Any],
    control: Mapping[str, Any],
    m4: Mapping[str, Any],
) -> dict[str, Any]:
    if anatomy.get("schema_version") != "eiel_b48_fixed_mlp_policy_birth_callback_analysis_v1":
        raise ValueError("anatomy receipt has the wrong schema")
    for name, receipt in (("control", control), ("m4", m4)):
        if receipt.get("schema_version") != "eiel_b48_fixed_mlp_policy_birth_refill_profile_v1":
            raise ValueError(f"{name} profile has the wrong schema")
        if int(receipt["timing_ns"]["host_wall_time_ns"]["count"]) < 5:
            raise ValueError(f"{name} profile requires at least five repeats")
    if control["artifact_source_commit"] != m4["artifact_source_commit"]:
        raise ValueError("control and M4 artifacts do not share one source commit")
    if control["fixture"] != m4["fixture"]:
        raise ValueError("control and M4 profiles do not share one physical fixture")
    if control["output_digest"] != m4["output_digest"]:
        raise ValueError("M4 output is not byte-exact to control")
    if control["build_identity"]["m0"]["option_d_m0_telemetry_enabled"] is not True:
        raise ValueError("control is not an M0 telemetry build")
    if m4["build_identity"]["m0"]["option_d_m0_telemetry_enabled"] is not True:
        raise ValueError("M4 candidate is not an M0 telemetry build")
    if control["build_identity"]["m4"]["option_d_m4_device_policy_refill_enabled"] is not False:
        raise ValueError("control unexpectedly enables M4")
    if m4["build_identity"]["m4"]["option_d_m4_device_policy_refill_enabled"] is not True:
        raise ValueError("candidate does not enable M4")

    anatomy_mean_ns = _mean(anatomy, "host_wall_time_ns")
    control_mean_ns = _mean(control, "host_wall_time_ns")
    surface_ratio = control_mean_ns / anatomy_mean_ns
    surface_matched = 0.75 <= surface_ratio <= 1.25
    control_median_ns = _median(control, "host_wall_time_ns")
    m4_mean_ns = _mean(m4, "host_wall_time_ns")
    m4_median_ns = _median(m4, "host_wall_time_ns")
    mean_speedup = control_mean_ns / m4_mean_ns
    median_speedup = control_median_ns / m4_median_ns
    exact = control["output_digest"] == m4["output_digest"]
    material_win = mean_speedup >= 1.10 and median_speedup >= 1.10
    if exact and surface_matched and material_win:
        decision = "SELECT_EXACT_M4_DEVICE_COMPACT_REFILL_AS_ARCHITECTURE_BASELINE"
        selected = "existing exact CUDA MT19937 compact policy-refill candidate M4"
        rationale = (
            "same-source M4 is byte-exact and materially faster on the physical-B48 "
            "compact disabled-mutation surface"
        )
    elif exact and surface_matched:
        pack_mean_ns = _mean(control, "crossover_pack_time_ns")
        compare_mean_ns = _mean(control, "torch_compare_time_ns")
        if pack_mean_ns > compare_mean_ns:
            decision = "SELECT_BATCHED_EXACT_CPU_LSB0_PACKING_PROTOTYPE"
            selected = "batched exact CPU crossover generation and vectorized LSB0 packing"
            rationale = "M4 lacks a material win and measured LSB0 packing exceeds comparison"
        else:
            decision = "NO_ARCHITECTURE_SELECTED"
            selected = None
            rationale = "M4 lacks a material win and no alternate dominant substage is proven"
    else:
        decision = "NO_ARCHITECTURE_SELECTED"
        selected = None
        rationale = "same-surface or byte-exact selection prerequisites failed"

    return {
        "schema_version": "eiel_b48_fixed_mlp_policy_birth_architecture_decision_v1",
        "decision": decision,
        "selected_architecture": selected,
        "rationale": rationale,
        "evidence": {
            "source_commit": control["artifact_source_commit"],
            "byte_exact_output_digest": control["output_digest"],
            "byte_exact_control_m4": exact,
            "historical_anatomy_mean_ms": anatomy_mean_ns / 1e6,
            "current_control_mean_ms": control_mean_ns / 1e6,
            "current_control_median_ms": control_median_ns / 1e6,
            "m4_mean_ms": m4_mean_ns / 1e6,
            "m4_median_ms": m4_median_ns / 1e6,
            "control_to_historical_surface_ratio": surface_ratio,
            "surface_matched_within_25_percent": surface_matched,
            "mean_speedup": mean_speedup,
            "median_speedup": median_speedup,
            "material_win_threshold": 1.10,
            "material_win": material_win,
            "control_random_mean_ms": _mean(control, "torch_random_generation_time_ns") / 1e6,
            "control_compare_mean_ms": _mean(control, "torch_compare_time_ns") / 1e6,
            "control_pack_mean_ms": _mean(control, "crossover_pack_time_ns") / 1e6,
        },
        "implementation_constraints": [
            "retain the compact disabled-mutation surface gate and fail closed elsewhere",
            "preserve exact Torch CPU MT19937 replay, semantic draw counts, stable B48 row order, and checkpoint state",
            "keep M4 default-off until targeted delta QualPrep and Q6 checkpoint continuity pass",
            "do not transfer qualification from this M0 component measurement",
        ],
        "scope_limit": "M0 architecture selection only; not implementation acceptance, promotion, or qualification",
    }


def _markdown(receipt: Mapping[str, Any]) -> str:
    evidence = receipt["evidence"]
    return "\n".join(
        [
            "# Fixed-MLP policy-birth architecture decision",
            "",
            f"Decision: `{receipt['decision']}`",
            "",
            str(receipt["rationale"]) + ".",
            "",
            "| Measurement | Value |",
            "|---|---:|",
            f"| Historical callback mean | {evidence['historical_anatomy_mean_ms']:.3f} ms |",
            f"| Current control mean | {evidence['current_control_mean_ms']:.3f} ms |",
            f"| M4 mean | {evidence['m4_mean_ms']:.3f} ms |",
            f"| Mean speedup | {evidence['mean_speedup']:.2f}x |",
            f"| Median speedup | {evidence['median_speedup']:.2f}x |",
            f"| Control random generation | {evidence['control_random_mean_ms']:.3f} ms |",
            f"| Control comparison | {evidence['control_compare_mean_ms']:.3f} ms |",
            f"| Control LSB0 packing | {evidence['control_pack_mean_ms']:.3f} ms |",
            "",
            "This is an M0 architecture decision, not production or qualification authority.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--m4", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite immutable architecture decision")
    receipt = decide_architecture(_load(args.anatomy), _load(args.control), _load(args.m4))
    receipt["inputs"] = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in (("anatomy", args.anatomy), ("control", args.control), ("m4", args.m4))
    }
    args.output_root.mkdir(parents=True)
    output = args.output_root / "m0_policy_birth_architecture_decision.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.output_root / "m0_policy_birth_architecture_decision.md").write_text(
        _markdown(receipt)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
