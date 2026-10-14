"""Measure policy-content uniqueness from CPU-readable banks or artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Final, Mapping

import numpy as np

from engine.perf.hapb_v2.exactness_contract_v1 import POLICY_WIDTH_V1


UNIQUENESS_SCHEMA_V1: Final[str] = "eiel_hapb_v2_content_uniqueness_v1"
POLICY_ROW_BYTES_V1: Final[int] = POLICY_WIDTH_V1 * 4


class HapbV2UniquenessError(ValueError):
    """Content-uniqueness analysis rejected its input."""


def row_digest_v1(row: np.ndarray) -> str:
    array = np.ascontiguousarray(row, dtype=np.float32)
    if array.shape[-1] != POLICY_WIDTH_V1:
        raise HapbV2UniquenessError("policy row width is not 9991")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def analyze_policy_bank_v1(
    bank: np.ndarray,
    *,
    source: str,
    live_mask: np.ndarray | None = None,
) -> dict[str, object]:
    array = np.asarray(bank, dtype=np.float32)
    if array.ndim != 3 or int(array.shape[2]) != POLICY_WIDTH_V1:
        raise HapbV2UniquenessError("policy bank must be [cells, slots, 9991]")
    cells, slots, _ = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    if live_mask is None:
        mask = np.ones((cells, slots), dtype=bool)
    else:
        mask = np.asarray(live_mask, dtype=bool)
        if mask.shape != (cells, slots):
            raise HapbV2UniquenessError("live mask shape does not match the bank")

    per_cell: list[dict[str, object]] = []
    global_counts: Counter[str] = Counter()
    cells_by_digest: dict[str, set[int]] = defaultdict(set)
    for cell in range(cells):
        cell_counts: Counter[str] = Counter()
        for slot in range(slots):
            if not bool(mask[cell, slot]):
                continue
            digest = row_digest_v1(array[cell, slot])
            cell_counts[digest] += 1
            global_counts[digest] += 1
            cells_by_digest[digest].add(cell)
        live = int(sum(cell_counts.values()))
        unique = int(len(cell_counts))
        per_cell.append(
            {
                "cell": cell,
                "live_slots": live,
                "unique_contents": unique,
                "duplicate_ratio": (
                    0.0 if live == 0 else float(1.0 - (unique / live))
                ),
            }
        )
    live_total = int(sum(global_counts.values()))
    unique_global = int(len(global_counts))
    cross_cell_shared = int(
        sum(1 for owners in cells_by_digest.values() if len(owners) > 1)
    )
    top_repeated = [
        {"digest_sha256": digest, "count": int(count)}
        for digest, count in global_counts.most_common(8)
        if count > 1
    ]
    return {
        "schema_version": UNIQUENESS_SCHEMA_V1,
        "source": source,
        "cells": cells,
        "slots_per_cell": slots,
        "live_slots": live_total,
        "per_cell_unique_contents": [int(row["unique_contents"]) for row in per_cell],
        "per_cell": per_cell,
        "global_unique_contents": unique_global,
        "duplicate_ratio": (
            0.0 if live_total == 0 else float(1.0 - (unique_global / live_total))
        ),
        "cross_cell_shared_content_ids": cross_cell_shared,
        "cross_cell_sharing_ratio": (
            0.0 if unique_global == 0 else float(cross_cell_shared / unique_global)
        ),
        "top_repeated_contents": top_repeated,
        "notes": [
            "content lifetime is not inferred from a single bank snapshot",
            "do not extrapolate B160/B192 from one unexplained histogram",
        ],
    }


def project_pool_bytes_v1(
    *,
    cells: int,
    unique_per_cell: float,
    unique_global: float | None,
    include_slot_map: bool = True,
) -> dict[str, float]:
    per_cell_pool = float(cells) * float(unique_per_cell) * float(POLICY_ROW_BYTES_V1)
    global_pool = (
        None
        if unique_global is None
        else float(unique_global) * float(POLICY_ROW_BYTES_V1)
    )
    slot_map = float(cells * 512 * (8 + 8 + 8)) if include_slot_map else 0.0
    return {
        "per_cell_pool_bytes": per_cell_pool,
        "global_pool_bytes": global_pool if global_pool is not None else float("nan"),
        "slot_map_bytes": slot_map,
        "policy_row_bytes": float(POLICY_ROW_BYTES_V1),
    }


def extract_policy_bank_from_cell_projection_v1(path: Path | str) -> np.ndarray:
    """Read the [512, 9991] float32 member from one source-projection zip."""

    import zipfile

    artifact = Path(path)
    expected = 512 * POLICY_ROW_BYTES_V1
    if not zipfile.is_zipfile(artifact):
        raise HapbV2UniquenessError(f"source projection is not a zip: {artifact}")
    with zipfile.ZipFile(artifact) as archive:
        matches = [
            info for info in archive.infolist() if info.file_size == expected
        ]
        if len(matches) != 1:
            raise HapbV2UniquenessError(
                f"{artifact} has {len(matches)} members of policy-bank size {expected}"
            )
        payload = archive.read(matches[0].filename)
    bank = np.frombuffer(payload, dtype=np.float32).reshape(512, POLICY_WIDTH_V1).copy()
    return bank


def analyze_source_projection_root_v1(root: Path | str) -> dict[str, object]:
    directory = Path(root)
    cells = []
    for child in sorted(directory.iterdir()):
        projection = child / "projection.bin"
        if not projection.is_file():
            continue
        cells.append(extract_policy_bank_from_cell_projection_v1(projection))
    if not cells:
        raise HapbV2UniquenessError(f"no projection.bin files under {directory}")
    bank = np.stack(cells, axis=0)
    return analyze_policy_bank_v1(bank, source=str(directory))


def analyze_canonical_bytes_v1(
    payload: bytes,
    *,
    cells: int,
    slots: int,
    source: str,
) -> dict[str, object]:
    expected = cells * slots * POLICY_ROW_BYTES_V1
    if len(payload) != expected:
        raise HapbV2UniquenessError(
            f"canonical bank bytes {len(payload)} != {expected}"
        )
    bank = np.frombuffer(payload, dtype=np.float32).reshape(cells, slots, POLICY_WIDTH_V1)
    return analyze_policy_bank_v1(bank, source=source)


def summarize_uniqueness_limitations_v1(report: Mapping[str, object]) -> list[str]:
    limitations = list(report.get("notes") or [])
    if float(report.get("cross_cell_sharing_ratio") or 0.0) == 0.0:
        limitations.append(
            "no cross-cell sharing observed in this snapshot; V1 forbids it"
        )
    return limitations
