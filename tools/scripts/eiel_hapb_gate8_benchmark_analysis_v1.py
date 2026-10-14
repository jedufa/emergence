#!/usr/bin/env python3
"""Analyze current-source Gate 8 measurements without executing CUDA.

The input is a receipt-bound JSON measurement table produced by clean runtime
processes.  This tool only computes the declared metric and applies the frozen
admission/promotion rules; it never invents unavailable telemetry, relaxes
semantic status, or makes a qualification/production claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.orchestration import atomic_write_json


SCHEMA_VERSION = "eiel_hapb_gate8_benchmark_analysis_v1"
INPUT_SCHEMA = "eiel_hapb_gate8_benchmark_measurements_v1"
REQUIRED_CANDIDATES = (48, 60, 72, 84, 96)
OPTIONAL_CANDIDATES = (108, 120)
UPPER_FRONTIER = (128, 130, 131)
AUTHORIZED_LADDER_CELLS = frozenset(
    REQUIRED_CANDIDATES + OPTIONAL_CANDIDATES + UPPER_FRONTIER
)
VALID_LADDER_STOP_REASONS = frozenset(
    {
        "next_candidate_cannot_beat_winner",
        "gpu_or_host_or_checkpoint_margin_unsafe",
        "frontier_exhausted",
    }
)
PROMOTION_THRESHOLD = 1.25
DEVICE_HEADROOM_FRACTION = 0.10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def _finite_positive(value: Any, label: str) -> float:
    _require(type(value) in {int, float} and not isinstance(value, bool), f"{label} is unavailable or malformed")
    result = float(value)
    _require(math.isfinite(result) and result > 0.0, f"{label} must be finite and positive")
    return result


def _bool(value: Any, label: str) -> bool:
    _require(type(value) is bool, f"{label} must be an explicit boolean")
    return bool(value)


def _memory_admission(row: Mapping[str, Any]) -> dict[str, Any]:
    memory = row.get("memory")
    if not isinstance(memory, Mapping):
        return {"status": "UNAVAILABLE", "physical_margin_bytes": None, "rule_pass": False}
    total = memory.get("device_total_bytes")
    minimum_free = memory.get("minimum_free_at_peak_bytes")
    if type(total) is not int or type(minimum_free) is not int or total <= 0 or minimum_free < 0:
        return {"status": "UNAVAILABLE", "physical_margin_bytes": None, "rule_pass": False}
    reserve = math.ceil(total * DEVICE_HEADROOM_FRACTION)
    margin = minimum_free - reserve
    declared = memory.get("rule_pass")
    return {
        "status": "MEASURED",
        "device_total_bytes": total,
        "minimum_free_at_peak_bytes": minimum_free,
        "ten_percent_reserve_bytes": reserve,
        "physical_margin_bytes": margin,
        "rule_pass": declared is True and margin >= 0,
        "declared_rule_pass": declared,
    }


def _analyze_row(row: Mapping[str, Any], *, baseline: bool = False) -> dict[str, Any]:
    candidate_id = row.get("candidate_id")
    _require(isinstance(candidate_id, str) and candidate_id, "candidate_id is missing")
    cells = row.get("batch_cells")
    _require(type(cells) is int and cells > 0, f"{candidate_id} batch_cells is invalid")
    semantic = _bool(row.get("semantically_exact"), f"{candidate_id} semantically_exact")
    sustained = _bool(row.get("sustained_rehearsal_pass"), f"{candidate_id} sustained_rehearsal_pass")
    production_admission = _bool(row.get("production_execution_admission"), f"{candidate_id} production_execution_admission")
    qualification_claim = _bool(row.get("qualification_claim"), f"{candidate_id} qualification_claim")
    _require(not production_admission, f"{candidate_id} is not an engineering-only measurement")
    _require(not qualification_claim, f"{candidate_id} incorrectly claims qualification")
    seconds = _finite_positive(
        row.get("amortized_production_seconds_per_generation"),
        f"{candidate_id} amortized_production_seconds_per_generation",
    )
    throughput = cells * 3600.0 / seconds
    memory = _memory_admission(row)
    timing = row.get("timing")
    _require(isinstance(timing, Mapping), f"{candidate_id} timing buckets are missing")
    unavailable = timing.get("unavailable")
    _require(unavailable is None or isinstance(unavailable, list), f"{candidate_id} unavailable telemetry is malformed")
    return {
        "candidate_id": candidate_id,
        "route": row.get("route"),
        "batch_cells": cells,
        "baseline": baseline,
        "semantically_exact": semantic,
        "sustained_rehearsal_pass": sustained,
        "production_execution_admission": production_admission,
        "qualification_claim": qualification_claim,
        "amortized_production_seconds_per_generation": seconds,
        "production_equivalent_cell_generations_per_hour": throughput,
        "memory": memory,
        "timing_unavailable": unavailable or [],
        "indispensable_capability": row.get("indispensable_capability") is True,
        "indispensable_capability_evidence": row.get("indispensable_capability_evidence"),
        "finalist_clean_process_count": row.get("finalist_clean_process_count"),
    }


def _decision(
    rows: list[dict[str, Any]],
    *,
    implemented_batch_cells: tuple[int, ...] | None = None,
    high_b_abi: str | None = None,
    ladder_complete: bool | None = None,
    ladder_stop_reason: str | None = None,
) -> dict[str, Any]:
    if ladder_stop_reason == "wait_resource":
        return {
            "hapb_perf": "HAPB-PERF-C",
            "terminal_hapb_e2e": "HAPB-E2E-C",
            "reason": "WAIT-RESOURCE is incomplete evidence and is not PERF-B",
            "resident_control": None,
            "best_hapb": None,
            "relative_gain": None,
            "required_candidate_set_complete": False,
            "high_b_abi": high_b_abi,
            "ladder_complete": False,
            "ladder_stop_reason": ladder_stop_reason,
        }
    resident = [row for row in rows if row["baseline"] and row["batch_cells"] == 48]
    if len(resident) != 1:
        return {
            "hapb_perf": "HAPB-PERF-C",
            "terminal_hapb_e2e": "HAPB-E2E-C",
            "reason": "exactly one current Resident B48 control measurement is required",
            "resident_control": None,
            "best_hapb": None,
            "relative_gain": None,
        }
    control = resident[0]
    hapb = [row for row in rows if not row["baseline"]]
    measured_hapb = {row["batch_cells"] for row in hapb}
    authorized_complete_climb = (
        high_b_abi == "frozen"
        and ladder_complete is True
        and ladder_stop_reason in VALID_LADDER_STOP_REASONS
        and 48 in measured_hapb
        and measured_hapb <= AUTHORIZED_LADDER_CELLS
    )
    if high_b_abi == "not_implemented" and implemented_batch_cells:
        required = set(implemented_batch_cells)
    elif authorized_complete_climb:
        required = measured_hapb
    else:
        required = set(REQUIRED_CANDIDATES)
    required_present = {row["batch_cells"] for row in hapb} >= required
    comparable = [
        row for row in hapb
        if row["semantically_exact"] and row["memory"]["rule_pass"]
    ]
    valid = [row for row in comparable if row["sustained_rehearsal_pass"]]
    best = max(valid, key=lambda row: row["production_equivalent_cell_generations_per_hour"], default=None)
    gain = None if best is None else best["production_equivalent_cell_generations_per_hour"] / control["production_equivalent_cell_generations_per_hour"]
    finalists_ready = all(
        type(row.get("finalist_clean_process_count")) is int and row["finalist_clean_process_count"] >= 3
        for row in (valid[:2] if len(valid) >= 2 else valid)
    )
    comparable_b48 = [row for row in comparable if row["batch_cells"] == 48]
    unimplemented_high_b = high_b_abi == "not_implemented" and required == {48}
    if (
        unimplemented_high_b
        and len(comparable_b48) == 1
        and not any(row["indispensable_capability"] for row in comparable_b48)
    ):
        implemented_gain = (
            comparable_b48[0]["production_equivalent_cell_generations_per_hour"]
            / control["production_equivalent_cell_generations_per_hour"]
        )
        if implemented_gain < PROMOTION_THRESHOLD:
            return {
                "hapb_perf": "HAPB-PERF-B",
                "terminal_hapb_e2e": "HAPB-E2E-B",
                "reason": (
                    "implemented HAPB B48 is below the 1.25x Resident-B48 threshold "
                    "and high-B ABI is not implemented"
                ),
                "resident_control": control["candidate_id"],
                "best_hapb": comparable_b48[0]["candidate_id"],
                "relative_gain": implemented_gain,
                "required_candidate_cells_present": sorted({row["batch_cells"] for row in hapb}),
                "required_candidate_set_complete": required_present,
                "finalists_ready": finalists_ready,
                "high_b_abi": high_b_abi,
            }
    clear_miss = (
        authorized_complete_climb
        and best is not None
        and gain is not None
        and gain < PROMOTION_THRESHOLD
        and not any(row["indispensable_capability"] for row in valid)
        and all(
            type(row.get("finalist_clean_process_count")) is int
            and row["finalist_clean_process_count"] >= 1
            for row in valid
        )
    )
    if not required_present or best is None:
        perf, terminal, reason = "HAPB-PERF-C", "HAPB-E2E-C", "required current-source measurements, finalists, or rehearsal evidence is incomplete"
    elif not finalists_ready and not clear_miss:
        perf, terminal, reason = "HAPB-PERF-C", "HAPB-E2E-C", "required current-source measurements, finalists, or rehearsal evidence is incomplete"
    elif gain is not None and gain >= PROMOTION_THRESHOLD:
        perf, terminal, reason = "HAPB-PERF-A", "HAPB-E2E-A", "best valid HAPB candidate meets the 1.25x production-equivalent threshold"
    elif any(row["indispensable_capability"] for row in valid):
        perf, terminal, reason = "HAPB-PERF-A", "HAPB-E2E-A", "a valid HAPB candidate has documented indispensable production capability"
    else:
        perf, terminal, reason = "HAPB-PERF-B", "HAPB-E2E-B", "no valid HAPB candidate meets the promotion threshold"
    return {
        "hapb_perf": perf,
        "terminal_hapb_e2e": terminal,
        "reason": reason,
        "resident_control": control["candidate_id"],
        "best_hapb": None if best is None else best["candidate_id"],
        "relative_gain": gain,
        "required_candidate_cells_present": sorted({row["batch_cells"] for row in hapb}),
        "required_candidate_set_complete": required_present,
        "finalists_ready": finalists_ready,
        "high_b_abi": high_b_abi,
        "ladder_complete": bool(authorized_complete_climb),
        "ladder_stop_reason": ladder_stop_reason,
    }


def _template() -> dict[str, Any]:
    return {
        "schema_version": INPUT_SCHEMA,
        "contract": {
            "required_candidate_cells": list(REQUIRED_CANDIDATES),
            "optional_candidate_cells": list(OPTIONAL_CANDIDATES),
            "upper_frontier_cells": list(UPPER_FRONTIER),
            "promotion_threshold": PROMOTION_THRESHOLD,
        },
        "measurements": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--write-template", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    _require(not output_root.exists(), f"refusing to overwrite immutable analysis root: {output_root}")
    if args.write_template:
        _require(args.input is None, "--write-template cannot be combined with --input")
        output_root.mkdir(parents=True, exist_ok=False)
        template_path = output_root / "gate8_measurements_template.json"
        atomic_write_json(template_path, _template())
        print(json.dumps({"schema_version": SCHEMA_VERSION, "template": str(template_path), "gpu_execution_performed": False}, indent=2))
        return 0
    _require(args.input is not None, "--input is required unless --write-template is used")
    input_path = args.input.expanduser().resolve()
    _require(input_path.is_file(), f"measurement input is missing: {input_path}")
    payload = _load_json(input_path)
    _require(payload.get("schema_version") == INPUT_SCHEMA, "measurement schema differs")
    raw_rows = payload.get("measurements")
    _require(isinstance(raw_rows, list) and raw_rows, "measurement table is empty")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        _require(isinstance(raw, Mapping), "measurement row is malformed")
        rows.append(_analyze_row(raw, baseline=raw.get("route") == "resident"))
    implemented_raw = payload.get("implemented_batch_cells")
    implemented_batch_cells = None
    if isinstance(implemented_raw, list) and implemented_raw:
        _require(all(type(item) is int and item > 0 for item in implemented_raw), "implemented_batch_cells is malformed")
        implemented_batch_cells = tuple(implemented_raw)
    high_b_abi = payload.get("high_b_abi")
    if high_b_abi is not None:
        _require(high_b_abi in {"not_implemented", "frozen"}, "high_b_abi must be not_implemented or frozen")
    ladder_complete_raw = payload.get("ladder_complete")
    ladder_complete = None
    if ladder_complete_raw is not None:
        _require(type(ladder_complete_raw) is bool, "ladder_complete must be an explicit boolean")
        ladder_complete = ladder_complete_raw
    ladder_stop_reason = payload.get("ladder_stop_reason")
    if ladder_stop_reason is not None:
        _require(isinstance(ladder_stop_reason, str) and ladder_stop_reason, "ladder_stop_reason is malformed")
    decision = _decision(
        rows,
        implemented_batch_cells=implemented_batch_cells,
        high_b_abi=high_b_abi,
        ladder_complete=ladder_complete,
        ladder_stop_reason=ladder_stop_reason,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "candidate_results": rows,
        "decision": decision,
        "promotion_threshold": PROMOTION_THRESHOLD,
        "qualification_claim": False,
        "production_execution_admission": False,
        "gpu_execution_performed": False,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    receipt = output_root / "hapb_gate8_benchmark_analysis_receipt.json"
    result["receipt_sha256"] = canonical_sha256(result)
    atomic_write_json(receipt, result)
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "receipt": str(receipt),
        "receipt_sha256": result["receipt_sha256"],
        "hapb_perf": decision["hapb_perf"],
        "terminal_hapb_e2e": decision["terminal_hapb_e2e"],
        "gpu_execution_performed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
