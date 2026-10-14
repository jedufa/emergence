"""V2 production checkpoint: semantic frontier only, no MT/tapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

from engine.tw_rngv2.checkpoint import SemanticCheckpoint, capture_checkpoint, restore_checkpoint
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT

from .c5_rngv2_route import (
    RNG_SEMANTICS_V2,
    V2_ROUTE_ID,
    production_contract_hash,
    production_contract_identity,
)
from .contracts import TensorWorldContractError


CHECKPOINT_SCHEMA: str = "tensor_world_c6_checkpoint_rngv2_v1"


@dataclass(slots=True)
class TensorWorldV2CheckpointV1:
    schema_version: str
    rng_semantics_version: int
    route_id: str
    production_contract_identity: str
    production_contract_hash: str
    semantic: SemanticCheckpoint
    host_tape_bytes: int = 0
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


def capture_v2_checkpoint(
    *,
    world_uids: tuple[int, ...],
    absolute_tick: int,
    next_entity_uid: tuple[int, ...],
    rng_namespace_key64: int = NAMESPACE_DEFAULT,
) -> TensorWorldV2CheckpointV1:
    if any(value < 0 for value in world_uids + next_entity_uid) or absolute_tick < 0:
        raise TensorWorldContractError("V2 checkpoint identities must be nonnegative")
    semantic = capture_checkpoint(
        world_uids=world_uids,
        absolute_tick=absolute_tick,
        next_entity_uid=next_entity_uid,
        rng_namespace_key64=rng_namespace_key64,
    )
    return TensorWorldV2CheckpointV1(
        schema_version=CHECKPOINT_SCHEMA,
        rng_semantics_version=RNG_SEMANTICS_V2,
        route_id=V2_ROUTE_ID,
        production_contract_identity=production_contract_identity(),
        production_contract_hash=production_contract_hash(),
        semantic=semantic,
        host_tape_bytes=0,
        pending_work="none",
    )


def restore_v2_checkpoint(payload: dict[str, Any]) -> TensorWorldV2CheckpointV1:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise TensorWorldContractError("V2 checkpoint schema is unsupported")
    if int(payload.get("rng_semantics_version", 0)) != RNG_SEMANTICS_V2:
        raise TensorWorldContractError("V1 checkpoint cannot restore onto the V2 route")
    if payload.get("host_tape_bytes", 0) != 0:
        raise TensorWorldContractError("V2 checkpoint cannot carry host tape bytes")
    if payload.get("production_contract_hash") != production_contract_hash():
        raise TensorWorldContractError("V2 checkpoint contract hash mismatch")
    semantic = restore_checkpoint(payload["semantic"])
    return TensorWorldV2CheckpointV1(
        schema_version=CHECKPOINT_SCHEMA,
        rng_semantics_version=RNG_SEMANTICS_V2,
        route_id=str(payload["route_id"]),
        production_contract_identity=str(payload["production_contract_identity"]),
        production_contract_hash=str(payload["production_contract_hash"]),
        semantic=semantic,
        host_tape_bytes=0,
        pending_work="none",
    )
