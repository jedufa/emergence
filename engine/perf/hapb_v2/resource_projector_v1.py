"""Static HAPB B-scale resource projector.  Not an admission decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from engine.perf.hapb_v2.exactness_contract_v1 import POLICY_WIDTH_V1


PROJECTOR_SCHEMA_V1: Final[str] = "eiel_hapb_v2_resource_projector_v1"
DEFAULT_B_VALUES_V1: Final[tuple[int, ...]] = (
    48, 72, 96, 120, 128, 130, 131, 160, 192
)
SLOTS_PER_CELL_V1: Final[int] = 512
HOT_PREFIX_ROWS_V1: Final[int] = 256
POLICY_ROW_BYTES_V1: Final[int] = POLICY_WIDTH_V1 * 4
DEVICE_RESERVE_FRACTION_V1: Final[float] = 0.10


class HapbV2ResourceProjectorError(ValueError):
    """Resource projection inputs are malformed."""


@dataclass(frozen=True)
class ProjectorScenarioV1:
    name: str
    unique_per_cell: float
    unique_global: float | None
    snapshot_streaming: bool
    notes: str


def _require_ratio(name: str, value: float) -> float:
    if type(value) is not float and type(value) is not int:
        raise HapbV2ResourceProjectorError(f"{name} must be numeric")
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise HapbV2ResourceProjectorError(f"{name} must be in [0, 1]")
    return number


def project_bytes_v1(
    *,
    cells: int,
    unique_per_cell: float,
    unique_global: float | None,
    snapshot_streaming: bool,
    host_admission_bytes: int | None,
    device_capacity_bytes: int | None,
) -> dict[str, object]:
    if type(cells) is not int or cells <= 0:
        raise HapbV2ResourceProjectorError("cells must be a positive int")
    if unique_per_cell < 0.0:
        raise HapbV2ResourceProjectorError("unique_per_cell is negative")
    v1_per_cell_pool = cells * HOT_PREFIX_ROWS_V1 * POLICY_ROW_BYTES_V1
    v1_full_bank = cells * SLOTS_PER_CELL_V1 * POLICY_ROW_BYTES_V1
    v2_per_cell_map = cells * SLOTS_PER_CELL_V1 * (8 + 8 + 8)
    v2_content_pool = int(round(cells * unique_per_cell * POLICY_ROW_BYTES_V1))
    global_pool = (
        None
        if unique_global is None
        else int(round(unique_global * POLICY_ROW_BYTES_V1))
    )
    snapshot = v1_full_bank if not snapshot_streaming else v1_full_bank
    if snapshot_streaming:
        snapshot_note = "one canonical bank; reconstructible caches omitted"
    else:
        snapshot_note = "canonical bank plus possible overlapping pickle/cache"
        snapshot += v1_per_cell_pool
    device_keepout = None
    if device_capacity_bytes is not None:
        device_keepout = int(device_capacity_bytes * (1.0 - DEVICE_RESERVE_FRACTION_V1))
    return {
        "cells": cells,
        "v1_per_cell_prefix_pool_bytes": v1_per_cell_pool,
        "v1_canonical_bank_bytes": v1_full_bank,
        "v2_slot_map_bytes": v2_per_cell_map,
        "v2_content_pool_bytes": v2_content_pool,
        "v2_global_dedup_pool_bytes": global_pool,
        "snapshot_bytes": snapshot,
        "snapshot_note": snapshot_note,
        "device_reserve_fraction": DEVICE_RESERVE_FRACTION_V1,
        "device_keepout_bytes": device_keepout,
        "host_admission_bytes": host_admission_bytes,
        "admitted": None,
        "admission_claim": "not_made",
    }


def project_b_ladder_v1(
    *,
    measured_unique_per_cell: float | None,
    measured_unique_global: float | None,
    assumed_unique_ratio_optimistic: float,
    assumed_unique_ratio_pessimistic: float,
    host_admission_bytes: int | None = None,
    device_capacity_bytes: int | None = None,
    cells: tuple[int, ...] = DEFAULT_B_VALUES_V1,
) -> dict[str, object]:
    optimistic = _require_ratio(
        "assumed_unique_ratio_optimistic", assumed_unique_ratio_optimistic
    )
    pessimistic = _require_ratio(
        "assumed_unique_ratio_pessimistic", assumed_unique_ratio_pessimistic
    )
    if optimistic > pessimistic:
        # optimistic uniqueness means fewer unique rows
        pass
    scenarios = []
    for b in cells:
        measured_per_cell = (
            measured_unique_per_cell
            if measured_unique_per_cell is not None
            else float(HOT_PREFIX_ROWS_V1)
        )
        rows = [
            (
                "v1_per_cell_pool",
                float(HOT_PREFIX_ROWS_V1),
                None,
                False,
                "source-derived V1 P256 prefix pool",
            ),
            (
                "v2_per_cell_content_map",
                measured_per_cell,
                measured_unique_global,
                False,
                "measured or source-derived unique-per-cell contents",
            ),
            (
                "snapshot_streaming_release",
                measured_per_cell,
                measured_unique_global,
                True,
                "omit reconstructible caches from the snapshot",
            ),
            (
                "measured_global_dedup",
                measured_per_cell,
                measured_unique_global,
                True,
                "uses measured global unique count when available",
            ),
            (
                "optimistic_uniqueness_growth",
                float(SLOTS_PER_CELL_V1) * optimistic,
                None,
                True,
                "assumption, not a measurement",
            ),
            (
                "pessimistic_uniqueness_growth",
                float(SLOTS_PER_CELL_V1) * pessimistic,
                None,
                True,
                "assumption, not a measurement",
            ),
        ]
        projections = []
        for name, unique_per_cell, unique_global, streaming, note in rows:
            item = project_bytes_v1(
                cells=b,
                unique_per_cell=unique_per_cell,
                unique_global=unique_global,
                snapshot_streaming=streaming,
                host_admission_bytes=host_admission_bytes,
                device_capacity_bytes=device_capacity_bytes,
            )
            item["scenario"] = name
            item["input_class"] = (
                "measured"
                if "measured" in name and measured_unique_per_cell is not None
                else ("source_derived" if name.startswith("v1_") else "assumption")
            )
            item["notes"] = note
            projections.append(item)
        scenarios.append({"cells": b, "projections": projections})
    return {
        "schema_version": PROJECTOR_SCHEMA_V1,
        "device_reserve_fraction": DEVICE_RESERVE_FRACTION_V1,
        "policy_row_bytes": POLICY_ROW_BYTES_V1,
        "measured_unique_per_cell": measured_unique_per_cell,
        "measured_unique_global": measured_unique_global,
        "assumptions": {
            "optimistic_unique_ratio": optimistic,
            "pessimistic_unique_ratio": pessimistic,
        },
        "ladder": scenarios,
        "no_admission_claim": True,
    }
