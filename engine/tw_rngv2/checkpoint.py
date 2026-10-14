"""Backend-neutral V2 semantic checkpoint. No MT/tape/cache state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

from .contract import (
    CONTRACT_HASH,
    CONTRACT_IDENTITY,
    NAMESPACE_DEFAULT,
    SEMANTIC_EPOCH,
    TWDMATH_IDENTITY,
)


def _canonical(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


@dataclass(slots=True)
class SemanticCheckpoint:
    contract_identity: str
    contract_hash: str
    twdmath_identity: str
    rng_namespace_key64: int
    semantic_epoch: int
    world_uids: tuple[int, ...]
    absolute_tick: int
    next_entity_uid: tuple[int, ...]
    pending_work: str = "none"
    config_digest: str = "0" * 64
    algorithm: str = "Philox4x32-10"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["world_uids"] = list(self.world_uids)
        payload["next_entity_uid"] = list(self.next_entity_uid)
        return payload

    def digest(self) -> str:
        return sha256(_canonical(self.to_dict())).hexdigest()


def capture_checkpoint(
    *,
    world_uids: tuple[int, ...],
    absolute_tick: int,
    next_entity_uid: tuple[int, ...],
    rng_namespace_key64: int = NAMESPACE_DEFAULT,
    semantic_epoch: int = SEMANTIC_EPOCH,
    config_digest: str = "0" * 64,
) -> SemanticCheckpoint:
    if len(world_uids) != len(next_entity_uid):
        raise ValueError("world_uids and next_entity_uid must align")
    return SemanticCheckpoint(
        contract_identity=CONTRACT_IDENTITY,
        contract_hash=CONTRACT_HASH,
        twdmath_identity=TWDMATH_IDENTITY,
        rng_namespace_key64=int(rng_namespace_key64),
        semantic_epoch=int(semantic_epoch),
        world_uids=tuple(int(value) for value in world_uids),
        absolute_tick=int(absolute_tick),
        next_entity_uid=tuple(int(value) for value in next_entity_uid),
        config_digest=config_digest,
    )


def restore_checkpoint(payload: dict[str, Any] | SemanticCheckpoint) -> SemanticCheckpoint:
    document = payload.to_dict() if isinstance(payload, SemanticCheckpoint) else payload
    if document.get("pending_work") != "none":
        raise ValueError("checkpoint pending_work must be none")
    if document.get("contract_hash") != CONTRACT_HASH:
        raise ValueError("checkpoint contract hash does not match this prototype")
    return SemanticCheckpoint(
        contract_identity=str(document["contract_identity"]),
        contract_hash=str(document["contract_hash"]),
        twdmath_identity=str(document["twdmath_identity"]),
        rng_namespace_key64=int(document["rng_namespace_key64"]),
        semantic_epoch=int(document["semantic_epoch"]),
        world_uids=tuple(int(value) for value in document["world_uids"]),
        absolute_tick=int(document["absolute_tick"]),
        next_entity_uid=tuple(int(value) for value in document["next_entity_uid"]),
        pending_work="none",
        config_digest=str(document.get("config_digest") or "0" * 64),
        algorithm=str(document.get("algorithm") or "Philox4x32-10"),
    )
