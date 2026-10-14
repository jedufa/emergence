"""HAPB-FRAG RNGV2 contract bound to the live TensorWorld V2 parent."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from engine.tensor_world.c5_rngv2_route import (
    V2_ROUTE_ID,
    production_contract_hash,
    production_contract_identity,
)
from engine.tw_rngv2.contract import (
    CONTRACT_HASH as PARENT_PROTOTYPE_HASH,
    CONTRACT_IDENTITY as PARENT_PROTOTYPE_IDENTITY,
    MASK16,
    MASK32,
    MASK48,
    MAX_WORD_ORDINAL,
    NAMESPACE_DEFAULT,
    SEMANTIC_EPOCH,
    TWDMATH_IDENTITY,
)


CONTRACT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "_spec"
    / "campaigns"
    / "v1"
    / "hapb_rngv2"
    / "hapb_frag_rngv2_v1.json"
)
INVENTORY_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "_spec"
    / "campaigns"
    / "v1"
    / "hapb_rngv2"
    / "domain_inventory_v1.json"
)

PARENT_TW_IDENTITY_DECLARED: Final[str] = "tensor_world_c0_c7_rngv2_v1@7a6614d2b7851d85"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def contract_hash(document: dict[str, Any] | None = None) -> str:
    payload = document if document is not None else _load(CONTRACT_PATH)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


CONTRACT: Final[dict[str, Any]] = _load(CONTRACT_PATH)
INVENTORY: Final[dict[str, Any]] = _load(INVENTORY_PATH)
CONTRACT_HASH: Final[str] = contract_hash(CONTRACT)
CONTRACT_IDENTITY: Final[str] = f"{CONTRACT['route_id']}@{CONTRACT_HASH[:16]}"

LIVE_PARENT_TW_IDENTITY: Final[str] = production_contract_identity()
LIVE_PARENT_TW_HASH: Final[str] = production_contract_hash()
if LIVE_PARENT_TW_IDENTITY != PARENT_TW_IDENTITY_DECLARED:
    # A newer accepted parent is allowed only when the live production identity
    # still uses the same route id. Record the successor; do not silently
    # change shared arithmetic.
    if not LIVE_PARENT_TW_IDENTITY.startswith(f"{V2_ROUTE_ID}@"):
        raise RuntimeError(
            "HAPB RNGV2 parent is not the TensorWorld V2 production route: "
            f"{LIVE_PARENT_TW_IDENTITY}"
        )

PARENT_TW_IDENTITY: Final[str] = LIVE_PARENT_TW_IDENTITY
PARENT_TW_HASH: Final[str] = LIVE_PARENT_TW_HASH

HAPB_ONLY_DOMAINS: Final[dict[int, dict[str, object]]] = {
    int(row["id"]): dict(row) for row in CONTRACT["hapb_only_domains"]
}
SHARED_PARENT_DOMAINS: Final[frozenset[int]] = frozenset(
    int(value) for value in CONTRACT["shared_parent_domains"]
)

DOMAIN_HAPB_POLICY_WINDOW_CROSSOVER: Final[int] = 0x0600
DOMAIN_HAPB_POLICY_WINDOW_MUTATION: Final[int] = 0x0601
DOMAIN_HAPB_POLICY_WINDOW_MASK: Final[int] = 0x0602

__all__ = [
    "CONTRACT",
    "CONTRACT_HASH",
    "CONTRACT_IDENTITY",
    "CONTRACT_PATH",
    "DOMAIN_HAPB_POLICY_WINDOW_CROSSOVER",
    "DOMAIN_HAPB_POLICY_WINDOW_MASK",
    "DOMAIN_HAPB_POLICY_WINDOW_MUTATION",
    "HAPB_ONLY_DOMAINS",
    "INVENTORY",
    "INVENTORY_PATH",
    "LIVE_PARENT_TW_HASH",
    "LIVE_PARENT_TW_IDENTITY",
    "MASK16",
    "MASK32",
    "MASK48",
    "MAX_WORD_ORDINAL",
    "NAMESPACE_DEFAULT",
    "PARENT_PROTOTYPE_HASH",
    "PARENT_PROTOTYPE_IDENTITY",
    "PARENT_TW_HASH",
    "PARENT_TW_IDENTITY",
    "PARENT_TW_IDENTITY_DECLARED",
    "SEMANTIC_EPOCH",
    "SHARED_PARENT_DOMAINS",
    "TWDMATH_IDENTITY",
]
