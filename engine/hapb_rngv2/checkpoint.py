"""Backend-neutral HAPB RNGV2 checkpoint. No future tapes or CPython cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

from engine.tw_rngv2.checkpoint import SemanticCheckpoint, capture_checkpoint, restore_checkpoint
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT

from .contract import CONTRACT_HASH, CONTRACT_IDENTITY, PARENT_TW_HASH, PARENT_TW_IDENTITY
from .identity import HapbIdentityMap
from .route import RNG_SEMANTICS_V2, V2_HAPB_FRAG_ROUTE_ID, HapbRngv2ContractError


CHECKPOINT_SCHEMA: str = "hapb_frag_rngv2_checkpoint_v1"


@dataclass(slots=True)
class HapbRngv2Checkpoint:
    schema_version: str
    hapb_rng_semantics_version: int
    route_id: str
    contract_identity: str
    contract_hash: str
    parent_tw_contract_identity: str
    parent_tw_contract_hash: str
    semantic: SemanticCheckpoint
    identity: dict[str, object]
    host_page_mapping: list[list[int]]
    preserved_compact_rng: dict[str, object]
    host_tape_bytes: int = 0
    host_gaussian_cache_bytes: int = 0
    pending_work: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["semantic"] = self.semantic.to_dict()
        return payload

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def capture_hapb_rngv2_checkpoint(
    *,
    identity: HapbIdentityMap,
    world_uids: tuple[int, ...],
    absolute_tick: int,
    rng_namespace_key64: int = NAMESPACE_DEFAULT,
    preserved_compact_rng: dict[str, object] | None = None,
) -> HapbRngv2Checkpoint:
    if identity.cells != len(world_uids):
        raise HapbRngv2ContractError("world_uids must align with identity cells")
    semantic = capture_checkpoint(
        world_uids=world_uids,
        absolute_tick=absolute_tick,
        next_entity_uid=tuple(identity.next_entity_uid),
        rng_namespace_key64=rng_namespace_key64,
    )
    snap = identity.snapshot()
    return HapbRngv2Checkpoint(
        schema_version=CHECKPOINT_SCHEMA,
        hapb_rng_semantics_version=RNG_SEMANTICS_V2,
        route_id=V2_HAPB_FRAG_ROUTE_ID,
        contract_identity=CONTRACT_IDENTITY,
        contract_hash=CONTRACT_HASH,
        parent_tw_contract_identity=PARENT_TW_IDENTITY,
        parent_tw_contract_hash=PARENT_TW_HASH,
        semantic=semantic,
        identity=snap,
        host_page_mapping=list(snap["page_of_slot"]),
        preserved_compact_rng=dict(preserved_compact_rng or {}),
        host_tape_bytes=0,
        host_gaussian_cache_bytes=0,
        pending_work="none",
    )


def restore_hapb_rngv2_checkpoint(payload: dict[str, Any]) -> HapbRngv2Checkpoint:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise HapbRngv2ContractError("HAPB RNGV2 checkpoint schema is unsupported")
    if int(payload.get("hapb_rng_semantics_version", 0)) != RNG_SEMANTICS_V2:
        raise HapbRngv2ContractError("legacy HAPB checkpoint cannot restore onto RNGV2")
    if int(payload.get("host_tape_bytes", 0)) != 0:
        raise HapbRngv2ContractError("HAPB RNGV2 checkpoint cannot carry host tape bytes")
    if int(payload.get("host_gaussian_cache_bytes", 0)) != 0:
        raise HapbRngv2ContractError("HAPB RNGV2 checkpoint cannot carry a CPython gauss cache")
    if "mt19937" in payload or "python_random_gauss_primary" in payload:
        raise HapbRngv2ContractError("HAPB RNGV2 checkpoint forbids MT/gauss-cache fields")
    if payload.get("contract_hash") != CONTRACT_HASH:
        raise HapbRngv2ContractError("HAPB RNGV2 checkpoint contract hash mismatch")
    if payload.get("parent_tw_contract_hash") != PARENT_TW_HASH:
        raise HapbRngv2ContractError("HAPB RNGV2 parent TW hash mismatch")
    semantic = restore_checkpoint(payload["semantic"])
    return HapbRngv2Checkpoint(
        schema_version=CHECKPOINT_SCHEMA,
        hapb_rng_semantics_version=RNG_SEMANTICS_V2,
        route_id=str(payload["route_id"]),
        contract_identity=str(payload["contract_identity"]),
        contract_hash=str(payload["contract_hash"]),
        parent_tw_contract_identity=str(payload["parent_tw_contract_identity"]),
        parent_tw_contract_hash=str(payload["parent_tw_contract_hash"]),
        semantic=semantic,
        identity=dict(payload["identity"]),
        host_page_mapping=[list(row) for row in payload["host_page_mapping"]],
        preserved_compact_rng=dict(payload.get("preserved_compact_rng") or {}),
        host_tape_bytes=0,
        host_gaussian_cache_bytes=0,
        pending_work="none",
    )


def identity_from_checkpoint(checkpoint: HapbRngv2Checkpoint) -> HapbIdentityMap:
    return HapbIdentityMap.restore(checkpoint.identity)
