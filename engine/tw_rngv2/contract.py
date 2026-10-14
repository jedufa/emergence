"""Frozen TW-RNGV2-DEA1 prototype contract identity."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Final


CONTRACT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "_spec"
    / "campaigns"
    / "v1"
    / "tw_rngv2"
    / "standalone_contract_v1.json"
)


def _load() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def contract_hash(document: dict[str, Any] | None = None) -> str:
    payload = document if document is not None else _load()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


CONTRACT: Final[dict[str, Any]] = _load()
CONTRACT_HASH: Final[str] = contract_hash(CONTRACT)
CONTRACT_IDENTITY: Final[str] = (
    f"{CONTRACT['prototype_identity']}@{CONTRACT_HASH[:16]}"
)
TWDMATH_IDENTITY: Final[str] = str(CONTRACT["twdmath_1"]["identity"])
SEMANTIC_EPOCH: Final[int] = int(CONTRACT["semantic_epoch"])
NAMESPACE_DEFAULT: Final[int] = 0xCA0511_7E450A_51C0
MASK48: Final[int] = (1 << 48) - 1
MASK16: Final[int] = (1 << 16) - 1
MASK32: Final[int] = (1 << 32) - 1
MAX_DRAW_BLOCK: Final[int] = MASK16
MAX_WORD_ORDINAL: Final[int] = (MAX_DRAW_BLOCK << 2) | 3
