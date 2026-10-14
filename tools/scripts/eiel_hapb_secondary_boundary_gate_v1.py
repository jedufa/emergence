#!/usr/bin/env python3
"""Mechanically review the bounded post-alias HAPB tile-boundary timing pair.

This gate consumes two exact current-source H4 repeat-tick receipts: the
qualified 512-row owner and the existing native-ABI 4096-row candidate.  It
checks semantic parity and the newly exposed boundary counters, but it never
turns fewer tile turns or latency hiding into policy-byte reduction credit and
never authorizes Gate 6 or production admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.orchestration import atomic_write_json


SCHEMA_VERSION = "eiel_hapb_secondary_boundary_gate_v1"
H4_SCHEMA_PREFIX = (
    "eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1_"
    "repeat_tick_diagnostic_v1"
)
TIMING_SCHEMA_PREFIX = (
    "eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1_"
    "secondary_boundary_timing_v1"
)
SAMPLE_SCHEMA_PREFIX = (
    "eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1_"
    "secondary_boundary_timing_sample_v1"
)
POLICY_ROW_BYTES = 39_964


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a non-negative integer")
    return int(value)


def _check_timing(
    *, label: str, receipt: Mapping[str, Any], tile_rows: int
) -> list[dict[str, Any]]:
    timing = receipt.get("secondary_boundary_timing")
    _require(isinstance(timing, Mapping), f"{label} boundary timing is missing")
    _require(
        str(timing.get("schema_version", "")).startswith(TIMING_SCHEMA_PREFIX),
        f"{label} boundary timing schema differs",
    )
    _require(
        timing.get("policy_byte_reduction_credit") is False,
        f"{label} timing incorrectly claims policy-byte reduction credit",
    )
    samples = timing.get("samples")
    _require(isinstance(samples, list) and len(samples) == 2, f"{label} timing samples are incomplete")
    labels: list[str] = []
    checked: list[dict[str, Any]] = []
    for index, raw_sample in enumerate(samples):
        _require(isinstance(raw_sample, Mapping), f"{label} timing sample {index} is malformed")
        sample = dict(raw_sample)
        _require(
            str(sample.get("schema_version", "")).startswith(SAMPLE_SCHEMA_PREFIX),
            f"{label} timing sample {index} schema differs",
        )
        sample_label = sample.get("label")
        _require(sample_label in {"first", "second"}, f"{label} timing sample label is invalid")
        labels.append(str(sample_label))
        _require(sample.get("tile_rows") == tile_rows, f"{label} timing tile rows differ")
        for field in (
            "host_classification_map_construction_wall_ns",
            "native_execute_wall_ns",
            "control_d2h_bytes",
            "control_h2d_bytes",
            "sync_count",
            "gpu_stall_ns",
        ):
            _nonnegative_int(sample.get(field), f"{label} {field}")
        _require(
            sample.get("policy_byte_reduction_credit") is False,
            f"{label} timing sample claims policy-byte reduction credit",
        )
        _require(
            "not a policy-byte reduction measurement" in str(sample.get("scope", "")),
            f"{label} timing sample scope is not explicitly bounded",
        )
        counters = sample.get("counters")
        _require(isinstance(counters, Mapping), f"{label} timing counters are missing")
        for field in (
            "policy_h2d_rows",
            "policy_h2d_bytes",
            "policy_d2h_rows",
            "policy_d2h_bytes",
            "control_d2h_bytes",
            "control_h2d_bytes",
            "sync_count",
            "gpu_stall_ns",
        ):
            _nonnegative_int(counters.get(field), f"{label} counter {field}")
        _require(
            counters.get("policy_row_bytes") == POLICY_ROW_BYTES,
            f"{label} policy row byte width differs",
        )
        _require(
            int(counters["policy_h2d_bytes"])
            == int(counters["policy_h2d_rows"]) * POLICY_ROW_BYTES,
            f"{label} policy H2D row/byte accounting differs",
        )
        _require(
            int(counters["policy_d2h_rows"]) == 0
            and int(counters["policy_d2h_bytes"]) == 0,
            f"{label} unexpected policy D2H traffic",
        )
        _require(
            int(counters["control_d2h_bytes"]) == int(sample["control_d2h_bytes"])
            and int(counters["control_h2d_bytes"]) == int(sample["control_h2d_bytes"])
            and int(counters["sync_count"]) == int(sample["sync_count"])
            and int(counters["gpu_stall_ns"]) == int(sample["gpu_stall_ns"]),
            f"{label} timing counters do not match the raw observation",
        )
        checked.append(sample)
    _require(sorted(labels) == ["first", "second"], f"{label} timing labels are not one repeated pair")
    return checked


def _check_receipt(
    *, label: str, path: Path, tile_rows: int
) -> dict[str, Any]:
    receipt = _load_json(path)
    _require(
        str(receipt.get("schema_version", "")).startswith(H4_SCHEMA_PREFIX),
        f"{label} H4 receipt schema differs",
    )
    _require(receipt.get("tile_rows") == tile_rows, f"{label} tile rows differ")
    _require(receipt.get("content_alias_enabled") is True, f"{label} is not the alias route")
    _require(receipt.get("production_execution_admission") is False, f"{label} claims production admission")
    _require(receipt.get("qualification_claim") is False, f"{label} claims qualification")
    _require(receipt.get("repeat_count") == 2, f"{label} is not a two-run repeat")
    _require(receipt.get("terminal_projection_digests_exact") is True, f"{label} terminal parity is not exact")
    _require(type(receipt.get("mismatch_count")) is int and receipt["mismatch_count"] == 0, f"{label} has projection mismatches")
    digests = receipt.get("terminal_projection_digests")
    _require(isinstance(digests, Mapping), f"{label} terminal digests are missing")
    _require(digests.get("first") == digests.get("second"), f"{label} repeated terminal digests differ")

    observations = receipt.get("policy_traffic_observations")
    _require(isinstance(observations, Mapping), f"{label} policy traffic observations are missing")
    for trace_label in ("first", "second"):
        observation = observations.get(trace_label)
        _require(isinstance(observation, Mapping), f"{label} {trace_label} policy observation is missing")
        _require(observation.get("content_alias_enabled") is True, f"{label} {trace_label} is not alias-enabled")
        _require(observation.get("content_alias_relation_sha256") not in (None, "disabled"), f"{label} {trace_label} alias relation is missing")
        counters = observation.get("counters")
        _require(isinstance(counters, Mapping), f"{label} {trace_label} policy counters are missing")
        for field in ("policy_h2d_rows", "policy_h2d_bytes", "policy_d2h_rows", "policy_d2h_bytes"):
            _nonnegative_int(counters.get(field), f"{label} {trace_label} {field}")
        _require(counters.get("policy_row_bytes") == POLICY_ROW_BYTES, f"{label} {trace_label} policy row byte width differs")
        _require(int(counters["policy_h2d_bytes"]) == int(counters["policy_h2d_rows"]) * POLICY_ROW_BYTES, f"{label} {trace_label} policy H2D row/byte accounting differs")
        _require(int(counters["policy_d2h_rows"]) == 0 and int(counters["policy_d2h_bytes"]) == 0, f"{label} {trace_label} policy D2H traffic is nonzero")

    rng = receipt.get("process_rng_traces")
    _require(isinstance(rng, Mapping), f"{label} process RNG traces are missing")
    for trace_label in ("first", "second"):
        trace = rng.get(trace_label)
        _require(isinstance(trace, Mapping), f"{label} {trace_label} RNG trace is missing")
        for phase in ("after_prepare", "after_execute"):
            _require(isinstance(trace.get(phase), Mapping), f"{label} {trace_label} {phase} RNG trace is missing")

    birth_traces: dict[str, dict[str, Any]] = {}
    for trace_label in ("first", "second"):
        birth_path = path.parent / f"repeat_tick_{trace_label}_birth_transaction_trace.json"
        _require(birth_path.is_file(), f"{label} {trace_label} birth trace is missing")
        birth = _load_json(birth_path)
        _require(
            birth.get("trace_label") == trace_label,
            f"{label} {trace_label} birth trace label is invalid",
        )
        events = birth.get("events")
        _require(isinstance(events, list), f"{label} {trace_label} birth trace is malformed")
        for event in events:
            _require(isinstance(event, Mapping), f"{label} {trace_label} birth event is malformed")
            for item in event.get("active_cells", []):
                _require(isinstance(item, Mapping), f"{label} {trace_label} birth cell is malformed")
                _require(
                    item.get("policy_birth_tape_cursor_end")
                    == item.get("policy_birth_tape_cursor_start", -1) + item.get("birth_count", -1),
                    f"{label} {trace_label} birth cursor accounting differs",
                )
        # The label identifies the repeat and is expected to differ. Compare
        # only the semantic birth/cursor/recombination payload across repeats.
        birth_traces[trace_label] = {
            key: value for key, value in birth.items() if key != "trace_label"
        }
    _require(
        birth_traces["first"] == birth_traces["second"],
        f"{label} repeated birth traces differ",
    )
    timing = _check_timing(label=label, receipt=receipt, tile_rows=tile_rows)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "receipt": receipt,
        "timing": timing,
        "birth_traces": birth_traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-512-receipt", required=True, type=Path)
    parser.add_argument("--tile-4096-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError("refusing to overwrite an immutable secondary gate receipt")
    tile_512 = _check_receipt(label="tile-512", path=args.tile_512_receipt.resolve(), tile_rows=512)
    tile_4096 = _check_receipt(label="tile-4096", path=args.tile_4096_receipt.resolve(), tile_rows=4096)
    receipt_512 = tile_512["receipt"]
    receipt_4096 = tile_4096["receipt"]
    for field in ("artifact_sha256", "manifest_sha256", "reused_generation_zero_source_root", "host_harness_source_commit"):
        _require(receipt_512.get(field) == receipt_4096.get(field), f"tile comparison {field} binding differs")
    _require(
        receipt_512["terminal_projection_digests"] == receipt_4096["terminal_projection_digests"],
        "512-vs-4096 terminal projection digests differ",
    )
    for phase in ("after_prepare", "after_execute"):
        for trace_label in ("first", "second"):
            _require(
                receipt_512["process_rng_traces"][trace_label][phase]
                == receipt_4096["process_rng_traces"][trace_label][phase],
                f"512-vs-4096 {trace_label} {phase} RNG traces differ",
            )
    _require(tile_512["birth_traces"] == tile_4096["birth_traces"], "512-vs-4096 birth/recombination traces differ")
    result = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "tile_512_receipt": tile_512["path"],
        "tile_512_receipt_sha256": tile_512["sha256"],
        "tile_4096_receipt": tile_4096["path"],
        "tile_4096_receipt_sha256": tile_4096["sha256"],
        "artifact_sha256": receipt_512["artifact_sha256"],
        "manifest_sha256": receipt_512["manifest_sha256"],
        "source_root": receipt_512["reused_generation_zero_source_root"],
        "terminal_projection_parity": True,
        "rng_after_prepare_and_execute_parity": True,
        "birth_recombination_cursor_parity": True,
        "timing_comparison": {
            "tile_512": tile_512["timing"],
            "tile_4096": tile_4096["timing"],
        },
        "policy_byte_reduction_credit": False,
        "latency_hiding_or_tile_turn_reduction_is_not_policy_byte_reduction": True,
        "qualification_claim": False,
        "production_execution_admission": False,
        "gate6_authorization": False,
        "scope": "bounded post-alias 512-vs-4096 timing and exact-parity seam only",
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(args.output_root / "hapb_secondary_boundary_gate_receipt.json", result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
