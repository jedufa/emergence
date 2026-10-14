"""Stable logical-world coordinates for TensorWorld RNG V2."""

from __future__ import annotations

import hashlib
from typing import Final


LOGICAL_WORLD_UID_DERIVATION_V1: Final[str] = (
    "sha256(eiel_tw_rngv2_logical_world_uid_v1,configured_world_uid,cell_id)"
    "[0:4]-little"
)


def logical_world_uid_v1(*, configured_world_uid: int, cell_id: str) -> int:
    """Derive one partition-independent uint32 coordinate from a stable cell ID."""

    if (
        type(configured_world_uid) is not int
        or not 0 <= configured_world_uid <= 0xFFFFFFFF
    ):
        raise ValueError("TW RNG configured world UID must fit uint32")
    if type(cell_id) is not str or not cell_id:
        raise ValueError("TW RNG logical world identity requires a stable cell ID")
    payload = (
        "eiel_tw_rngv2_logical_world_uid_v1\0"
        f"{configured_world_uid}\0{cell_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
