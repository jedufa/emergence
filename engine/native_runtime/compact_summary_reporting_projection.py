"""Strict host projection for retained CUDA behavior and lineage authorities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .compact_summary_digests import (
    behavior_rows_digest_sha256,
    lineage_records_digest_sha256,
)


REPORTING_AUTHORITY_SCHEMA_VERSION = (
    "eiel_cuda_architecture_b_compact_summary_reporting_authority_v1"
)
REPORTING_RETAINED_LAYOUT_VERSION = (
    "eiel_cuda_architecture_b_compact_summary_reporting_retained_layout_v2"
)
REPORTING_HEADER_BYTES = 16
BEHAVIOR_ROW_BYTES = 56
LINEAGE_ROW_BYTES = 136
STRICT_ROW_CAPACITY = 64
DYNAMIC_ROW_CAPACITY = 224
MAXIMUM_DYNAMIC_ROW_CAPACITY = 31_104
LEDGER_SEALED = 2

_BEHAVIOR_ALIVE_END = 1 << 0
_BEHAVIOR_ELIGIBLE = 1 << 1
_BEHAVIOR_KEPT = 1 << 2
_BEHAVIOR_TERMINAL_SEALED = 1 << 3
_BEHAVIOR_KNOWN_FLAGS = (
    _BEHAVIOR_ALIVE_END
    | _BEHAVIOR_ELIGIBLE
    | _BEHAVIOR_KEPT
    | _BEHAVIOR_TERMINAL_SEALED
)

_HEADER_DTYPE = np.dtype(
    [
        ("start_id_ceiling", "<i8"),
        ("behavior_row_count", "<u2"),
        ("lineage_record_count", "<u2"),
        ("status", "<u4"),
    ],
    align=True,
)
_BEHAVIOR_DTYPE = np.dtype(
    [
        ("agent_id", "<i8"),
        ("energy_collected_exact", "<f8"),
        ("base_fitness", "<f8"),
        ("injury_start", "<f4"),
        ("injury_end", "<f4"),
        ("aggression_bias_f32", "<f4"),
        ("lifespan_steps", "<i4"),
        ("offspring", "<i4"),
        ("attacks_made", "<i4"),
        ("action_counts", "u1", (7,)),
        ("flags", "u1"),
    ],
    align=True,
)
_LINEAGE_DTYPE = np.dtype(
    [
        ("agent_id", "<i8"),
        ("energy", "<f8"),
        ("energy_collected_exact", "<f8"),
        ("episode0_pred_err_mean", "<f8"),
        ("identity_drift_ema", "<f8"),
        ("health", "<f4"),
        ("injury", "<f4"),
        ("drives", "<f4", (4,)),
        ("pred_err_sum", "<f4"),
        ("drive_jitter", "<f4"),
        ("aggression_bias_f32", "<f4"),
        ("exploration_bias_f32", "<f4"),
        ("y", "<i4"),
        ("x", "<i4"),
        ("founder_id", "<i4"),
        ("birth_generation", "<i4"),
        ("birth_tick", "<i4"),
        ("lineage_depth", "<i4"),
        ("offspring_count", "<i4"),
        ("survival_steps", "<i4"),
        ("first_repro_step", "<i4"),
        ("pred_err_count", "<i4"),
        ("patch_habitat_lineage_id", "<i4"),
        ("hidden_size", "<i4"),
        ("memory_dim", "<i4"),
        ("use_gru", "u1"),
        ("reserved", "u1", (3,)),
    ],
    align=True,
)

if _HEADER_DTYPE.itemsize != REPORTING_HEADER_BYTES:
    raise RuntimeError("reporting header ABI size drift")
if _BEHAVIOR_DTYPE.itemsize != BEHAVIOR_ROW_BYTES:
    raise RuntimeError("behavior authority ABI size drift")
if _LINEAGE_DTYPE.itemsize != LINEAGE_ROW_BYTES:
    raise RuntimeError("lineage authority ABI size drift")


def retained_bytes_per_generation_cell(row_capacity: int) -> int:
    if (
        type(row_capacity) is not int
        or not 0 < row_capacity <= MAXIMUM_DYNAMIC_ROW_CAPACITY
    ):
        raise ValueError(
            "row_capacity must be an integer in "
            f"[1, {MAXIMUM_DYNAMIC_ROW_CAPACITY}]"
        )
    return REPORTING_HEADER_BYTES + row_capacity * (
        BEHAVIOR_ROW_BYTES + LINEAGE_ROW_BYTES
    )


def _cpu_uint8_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif all(hasattr(value, name) for name in ("detach", "device", "dtype")):
        if value.device.type != "cpu":
            raise ValueError(
                "retained reporting authority must be copied to CPU exactly once "
                "before projection"
            )
        array = value.detach().contiguous().numpy()
    else:
        raise TypeError("retained reporting authority must be a CPU tensor or ndarray")
    if array.dtype != np.uint8:
        raise ValueError("retained reporting authority must use uint8 storage")
    if array.ndim == 2:
        array = array[:, np.newaxis, :]
    if array.ndim != 3 or not array.flags.c_contiguous:
        raise ValueError("retained reporting authority must be contiguous [G,C,bytes]")
    return array


def _row_view(raw: np.ndarray, dtype: np.dtype[Any], count: int) -> np.ndarray:
    return np.frombuffer(raw, dtype=dtype, count=count)


def _finite(value: Any, *, path: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _decode_behavior_rows(
    rows: np.ndarray,
    *,
    generation: int,
    start_id_ceiling: int,
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    previous_id = -1
    for index, source in enumerate(rows):
        agent_id = int(source["agent_id"])
        flags = int(source["flags"])
        if agent_id <= previous_id or agent_id < 0:
            raise ValueError("behavior authority agent IDs must be strictly ascending")
        if flags & ~_BEHAVIOR_KNOWN_FLAGS:
            raise ValueError(f"behavior authority row {index} has unknown flags")
        if flags & _BEHAVIOR_TERMINAL_SEALED == 0:
            raise ValueError(f"behavior authority row {index} is not terminally sealed")
        eligible = int(bool(flags & _BEHAVIOR_ELIGIBLE))
        kept = int(bool(flags & _BEHAVIOR_KEPT))
        if eligible and agent_id >= start_id_ceiling:
            raise ValueError("born-cohort behavior row cannot be selection eligible")
        if kept and not eligible:
            raise ValueError("kept behavior row must be selection eligible")
        actions = [int(value) for value in source["action_counts"]]
        nonnegative = {
            "lifespan_steps": int(source["lifespan_steps"]),
            "offspring": int(source["offspring"]),
            "attacks_made": int(source["attacks_made"]),
        }
        if any(value < 0 for value in nonnegative.values()):
            raise ValueError(f"behavior authority row {index} has negative counters")
        injury_start = _finite(source["injury_start"], path="injury_start")
        injury_end = _finite(source["injury_end"], path="injury_end")
        decoded.append(
            {
                "generation": generation,
                "agent_id": agent_id,
                "cohort": "start" if agent_id < start_id_ceiling else "born",
                "move_n": actions[0],
                "move_s": actions[1],
                "move_e": actions[2],
                "move_w": actions[3],
                "gather": actions[4],
                "rest": actions[5],
                "attack": actions[6],
                "lifespan_steps": nonnegative["lifespan_steps"],
                "energy_collected": _finite(
                    source["energy_collected_exact"], path="energy_collected"
                ),
                "offspring": nonnegative["offspring"],
                "attacks_made": nonnegative["attacks_made"],
                "fitness": _finite(source["base_fitness"], path="fitness"),
                "alive_end": int(bool(flags & _BEHAVIOR_ALIVE_END)),
                "eligible_for_selection": eligible,
                "kept_next": kept,
                "injury_end": injury_end,
                "injury_delta": injury_end - injury_start,
                "aggression_bias": float(
                    round(
                        _finite(
                            source["aggression_bias_f32"], path="aggression_bias"
                        ),
                        8,
                    )
                ),
            }
        )
        previous_id = agent_id
    return decoded


def _decode_lineage_records(rows: np.ndarray) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    previous_id = -1
    for index, source in enumerate(rows):
        agent_id = int(source["agent_id"])
        founder_id = int(source["founder_id"])
        first_repro = int(source["first_repro_step"])
        use_gru = int(source["use_gru"])
        if agent_id <= previous_id or agent_id < 0 or founder_id < 0:
            raise ValueError("lineage authority IDs must be nonnegative and ascending")
        if use_gru not in (0, 1) or np.any(source["reserved"] != 0):
            raise ValueError(f"lineage authority row {index} has invalid flag storage")
        nonnegative_fields = (
            "y",
            "x",
            "birth_generation",
            "birth_tick",
            "lineage_depth",
            "offspring_count",
            "survival_steps",
            "pred_err_count",
            "patch_habitat_lineage_id",
            "hidden_size",
            "memory_dim",
        )
        if any(int(source[field]) < 0 for field in nonnegative_fields):
            raise ValueError(f"lineage authority row {index} has negative fields")
        if first_repro < -1:
            raise ValueError(
                f"lineage authority row {index} has invalid first_repro_step"
            )
        record = {
            "agent_id": agent_id,
            "energy": _finite(source["energy"], path="lineage.energy"),
            "health": _finite(source["health"], path="lineage.health"),
            "injury": _finite(source["injury"], path="lineage.injury"),
            "drives": [
                _finite(value, path=f"lineage.drives[{drive}]")
                for drive, value in enumerate(source["drives"])
            ],
            "y": int(source["y"]),
            "x": int(source["x"]),
            "stats": {
                "lineage_id": f"L{founder_id:08d}",
                "founder_id": founder_id,
                "parent_ids": [],
                "parent_lineage_ids": [],
                "birth_generation": int(source["birth_generation"]),
                "birth_tick": int(source["birth_tick"]),
                "lineage_depth": int(source["lineage_depth"]),
                "offspring_count": int(source["offspring_count"]),
                "survival_steps": float(int(source["survival_steps"])),
                "energy_collected": _finite(
                    source["energy_collected_exact"],
                    path="lineage.energy_collected",
                ),
                "first_repro_step": None if first_repro < 0 else first_repro,
                "pred_err_count": int(source["pred_err_count"]),
                "pred_err_sum": _finite(
                    source["pred_err_sum"], path="lineage.pred_err_sum"
                ),
                "episode0_pred_err_mean": _finite(
                    source["episode0_pred_err_mean"],
                    path="lineage.episode0_pred_err_mean",
                ),
                "identity_drift_ema": _finite(
                    source["identity_drift_ema"],
                    path="lineage.identity_drift_ema",
                ),
                "drive_jitter": _finite(
                    source["drive_jitter"], path="lineage.drive_jitter"
                ),
                "ecological_role": "general_agent",
                "role_source": "default",
                "role_confidence": None,
                "patch_habitat_lineage_id": int(
                    source["patch_habitat_lineage_id"]
                ),
            },
            "genome": {
                "hidden_size": int(source["hidden_size"]),
                "memory_dim": int(source["memory_dim"]),
                "use_gru": bool(use_gru),
                "aggression_bias": float(
                    round(
                        _finite(
                            source["aggression_bias_f32"],
                            path="lineage.aggression_bias",
                        ),
                        8,
                    )
                ),
                "exploration_bias": float(
                    round(
                        _finite(
                            source["exploration_bias_f32"],
                            path="lineage.exploration_bias",
                        ),
                        8,
                    )
                ),
            },
        }
        decoded.append(record)
        previous_id = agent_id
    return decoded


def project_retained_reporting_authority(
    retained: Any,
    *,
    first_generation: int,
    row_capacity: int = DYNAMIC_ROW_CAPACITY,
) -> list[list[dict[str, Any]]]:
    """Decode one already-transferred `[G,C,bytes]` authority matrix."""

    if type(first_generation) is not int or first_generation < 0:
        raise ValueError("first_generation must be a nonnegative integer")
    array = _cpu_uint8_array(retained)
    expected_stride = retained_bytes_per_generation_cell(row_capacity)
    if array.shape[2] != expected_stride:
        raise ValueError(
            f"retained reporting stride must be {expected_stride}, got {array.shape[2]}"
        )
    out: list[list[dict[str, Any]]] = []
    behavior_offset = REPORTING_HEADER_BYTES
    lineage_offset = behavior_offset + row_capacity * BEHAVIOR_ROW_BYTES
    for generation_offset in range(array.shape[0]):
        generation_rows: list[dict[str, Any]] = []
        generation = first_generation + generation_offset
        for cell in range(array.shape[1]):
            raw = array[generation_offset, cell]
            header = _row_view(raw, _HEADER_DTYPE, 1)[0]
            behavior_count = int(header["behavior_row_count"])
            lineage_count = int(header["lineage_record_count"])
            start_id_ceiling = int(header["start_id_ceiling"])
            if int(header["status"]) != LEDGER_SEALED:
                raise ValueError(
                    f"reporting authority [{generation_offset},{cell}] is not sealed"
                )
            if (
                start_id_ceiling < 0
                or not 0 <= behavior_count <= row_capacity
                or not 0 <= lineage_count <= behavior_count
            ):
                raise ValueError("reporting authority header is inconsistent")
            behavior_storage = raw[
                behavior_offset : behavior_offset + row_capacity * BEHAVIOR_ROW_BYTES
            ]
            lineage_storage = raw[
                lineage_offset : lineage_offset + row_capacity * LINEAGE_ROW_BYTES
            ]
            behavior = _row_view(
                behavior_storage, _BEHAVIOR_DTYPE, row_capacity
            )
            lineage = _row_view(lineage_storage, _LINEAGE_DTYPE, row_capacity)
            if np.any(behavior_storage[behavior_count * BEHAVIOR_ROW_BYTES :]) or np.any(
                lineage_storage[lineage_count * LINEAGE_ROW_BYTES :]
            ):
                raise ValueError("inactive reporting authority capacity must be zero")
            behavior_rows = _decode_behavior_rows(
                behavior[:behavior_count],
                generation=generation,
                start_id_ceiling=start_id_ceiling,
            )
            lineage_records = _decode_lineage_records(lineage[:lineage_count])
            alive_ids = [
                int(row["agent_id"])
                for row in behavior_rows
                if int(row["alive_end"]) == 1
            ]
            lineage_ids = [int(row["agent_id"]) for row in lineage_records]
            if lineage_ids != alive_ids:
                raise ValueError(
                    "lineage authority IDs must exactly match pre-turnover alive rows"
                )
            generation_rows.append(
                {
                    "schema_version": REPORTING_AUTHORITY_SCHEMA_VERSION,
                    "retained_layout_version": REPORTING_RETAINED_LAYOUT_VERSION,
                    "generation": generation,
                    "cell": cell,
                    "start_id_ceiling": start_id_ceiling,
                    "behavior_rows": behavior_rows,
                    "lineage_records": lineage_records,
                    "behavior_rows_digest_sha256": behavior_rows_digest_sha256(
                        behavior_rows
                    ),
                    "lineage_records_digest_sha256": lineage_records_digest_sha256(
                        generation, lineage_records
                    ),
                }
            )
        out.append(generation_rows)
    return out


def reporting_memory_budget(
    *,
    generation_count: int,
    cells: int,
    slot_capacity: int,
    row_capacity: int = DYNAMIC_ROW_CAPACITY,
) -> Mapping[str, int]:
    if type(generation_count) is not int or generation_count <= 0:
        raise ValueError("generation_count must be positive")
    if type(cells) is not int or cells <= 0:
        raise ValueError("cells must be positive")
    if type(slot_capacity) is not int or slot_capacity <= 0:
        raise ValueError("slot_capacity must be positive")
    retained = retained_bytes_per_generation_cell(row_capacity)
    scratch_per_generation_cell = slot_capacity * 4 + row_capacity * 8
    return {
        "retained_bytes_per_generation_cell": retained,
        "retained_bytes": generation_count * cells * retained,
        "scratch_bytes_per_generation_cell": scratch_per_generation_cell,
        "scratch_bytes": generation_count * cells * scratch_per_generation_cell,
        "total_bytes": generation_count
        * cells
        * (retained + scratch_per_generation_cell),
    }
