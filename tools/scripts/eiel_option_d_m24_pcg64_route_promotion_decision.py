#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "eiel_option_d_m24_pcg64_route_promotion_decision_v1"
DECISION_WITHHELD = (
    "M24-A-BOUNDED-PCG64-RECEIPTS-QUALIFIED-ROUTE-PROMOTION-WITHHELD"
)

BOUND_OPERATION_TO_STREAM: dict[str, str] = {
    "runner.drive_noise.standard_normal": "numpy.runner",
    "turnover.spawn_coordinate": "numpy.birth_spawn_environment",
    "population.newborn_drive.standard_normal": "numpy.population",
}
PCG64_REQUIRED_PHYSICAL_STREAMS: tuple[str, ...] = (
    "numpy.environment_policy",
    "numpy.runner",
    "numpy.population",
    "numpy.birth_spawn_environment",
)
KNOWN_BOUND_STREAMS: frozenset[str] = frozenset(BOUND_OPERATION_TO_STREAM.values())


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return data


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _m15_pcg64_streams(receipt: Mapping[str, Any]) -> list[str]:
    streams = []
    for row in list(receipt.get("active_physical_streams") or []):
        if not isinstance(row, Mapping):
            continue
        stream_id = str(row.get("physical_stream_id") or "")
        algorithm_id = str(row.get("algorithm_id") or "")
        if stream_id.startswith("numpy.") or "pcg64" in algorithm_id.lower():
            streams.append(stream_id)
    return sorted(set(streams))


def _qualified_m22_receipts(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    authority = receipt.get("pcg64_promotion_authority")
    if not isinstance(authority, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for group_key in (
        "runner_pcg64_operation_receipts",
        "turnover_spawn_coordinate_operation_receipts",
    ):
        group = authority.get(group_key)
        if not isinstance(group, Mapping):
            continue
        for row in list(group.get("receipts") or []):
            if not isinstance(row, Mapping):
                continue
            operator_id = str(row.get("operator_id") or "")
            if (
                operator_id in BOUND_OPERATION_TO_STREAM
                and row.get("promotion_qualified") is True
                and row.get("raw_pcg64_delta_available") is True
                and not list(row.get("promotion_blockers") or [])
            ):
                result[operator_id] = dict(row)
    return result


def _qualified_m23_receipt(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    operator_id = str(receipt.get("operator_id") or "")
    if (
        operator_id in BOUND_OPERATION_TO_STREAM
        and receipt.get("promotion_qualified") is True
        and receipt.get("raw_pcg64_delta_available") is True
        and not list(receipt.get("promotion_blockers") or [])
    ):
        return {operator_id: dict(receipt)}
    return {}


def decide(
    *,
    m15_receipt: Mapping[str, Any],
    m22_receipt: Mapping[str, Any],
    m23_receipt: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    m15_streams = _m15_pcg64_streams(m15_receipt)
    qualified_operations = _qualified_m22_receipts(m22_receipt)
    qualified_operations.update(_qualified_m23_receipt(m23_receipt))

    qualified_streams = sorted(
        {
            BOUND_OPERATION_TO_STREAM[operator_id]
            for operator_id in qualified_operations
        }
    )
    missing_required_streams = sorted(
        stream
        for stream in PCG64_REQUIRED_PHYSICAL_STREAMS
        if stream not in KNOWN_BOUND_STREAMS
    )
    unqualified_live_streams = sorted(
        stream for stream in m15_streams if stream not in qualified_streams
    )
    blockers = []
    if missing_required_streams:
        blockers.append("pcg64_physical_streams_without_operation_receipts")
    if unqualified_live_streams:
        blockers.append("m15_live_pcg64_streams_not_route_promoted")
    blockers.append("pcg64_checkpoint_projection_migration_not_bound")
    blockers.append("production_route_removal_and_performance_not_qualified")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION_WITHHELD,
        "passed": True,
        "pcg64_route_promotion_qualified": False,
        "bounded_operation_receipts_qualified": (
            sorted(qualified_operations) == sorted(BOUND_OPERATION_TO_STREAM)
        ),
        "qualified_operations": sorted(qualified_operations),
        "qualified_streams": qualified_streams,
        "m15_pcg64_physical_streams": m15_streams,
        "missing_required_streams": missing_required_streams,
        "unqualified_live_streams": unqualified_live_streams,
        "promotion_blockers": blockers,
        "source_receipts": {
            name: {
                "path": str(path),
                "sha256": _digest(path),
            }
            for name, path in paths.items()
        },
        "next_recommended_slice": (
            "bind or explicitly exclude numpy.environment_policy before "
            "removing route-level PCG64 preparation/checkpoint surfaces; "
            "otherwise continue with measured kernel-launch/controller "
            "overhead work from M8"
        ),
    }
    result["receipt_sha256"] = sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m15-receipt", type=Path, required=True)
    parser.add_argument("--m22-receipt", type=Path, required=True)
    parser.add_argument("--m23-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "m15_live_rng_inventory": args.m15_receipt,
        "m22_pcg64_device_failure_atomicity": args.m22_receipt,
        "m23_population_newborn_drive_raw64": args.m23_receipt,
    }
    result = decide(
        m15_receipt=_read_json(args.m15_receipt),
        m22_receipt=_read_json(args.m22_receipt),
        m23_receipt=_read_json(args.m23_receipt),
        paths=paths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
