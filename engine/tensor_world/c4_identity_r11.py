"""Monotonic identity authority for the repaired C4 resident route.

The original C4 population tensor still carries ``agent_id_to_slot`` for
compatibility with the frozen R2--R9 ABI.  That tensor is a bounded legacy
projection and must not be used as the biological identity authority: a
monotonic child ID is allowed to exceed slot capacity while a physical slot is
reused.  R11 resolves IDs through the sorted active ``(id, slot)`` pairs and
keeps slot tokens separate from both values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .contracts import TensorWorldContractError
from .c4_state import TensorWorldStateBundleV4


C4_IDENTITY_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_identity_authority_r11_v1"
)
C4_IDENTITY_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_IDENTITY_R11_STAGING_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_identity_staging_r11_v1"
)
C4_IDENTITY_R11_STATUS_V1: Final[dict[int, str]] = {
    60: "active_count_invalid",
    61: "active_order_or_duplicate",
    62: "active_slot_mismatch",
    63: "next_agent_id_invalid_or_overflow",
    64: "compatibility_projection_mismatch",
    65: "slot_token_mismatch",
    66: "slot_token_overflow",
    67: "int32_patch_projection_overflow",
}


@dataclass(slots=True)
class TensorWorldC4IdentityStagingR11V1:
    death_slot_tokens: torch.Tensor
    parent_slot_tokens: torch.Tensor
    mate_slot_tokens: torch.Tensor
    child_slot_tokens: torch.Tensor
    next_agent_id_before: torch.Tensor
    status: torch.Tensor
    schema_version: str = C4_IDENTITY_R11_STAGING_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.status.device

    def validate(self, *, batch: int, capacity: int) -> None:
        if self.schema_version != C4_IDENTITY_R11_STAGING_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("R11 identity staging schema is unsupported")
        expected = (
            (self.death_slot_tokens, (batch, capacity), torch.int64),
            (self.parent_slot_tokens, (batch, capacity), torch.int64),
            (self.mate_slot_tokens, (batch, capacity), torch.int64),
            (self.child_slot_tokens, (batch, capacity), torch.int64),
            (self.next_agent_id_before, (batch,), torch.int64),
            (self.status, (batch,), torch.int32),
        )
        devices = set()
        for tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise TensorWorldContractError("R11 identity staging ABI mismatch")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("R11 identity staging must share one device")


def make_c4_identity_staging_r11_v1(
    *, batch: int, capacity: int, device: torch.device | str
) -> TensorWorldC4IdentityStagingR11V1:
    target = torch.device(device)
    slots = (batch, capacity)
    staging = TensorWorldC4IdentityStagingR11V1(
        death_slot_tokens=torch.full(slots, -1, dtype=torch.int64, device=target),
        parent_slot_tokens=torch.full(slots, -1, dtype=torch.int64, device=target),
        mate_slot_tokens=torch.full(slots, -1, dtype=torch.int64, device=target),
        child_slot_tokens=torch.full(slots, -1, dtype=torch.int64, device=target),
        next_agent_id_before=torch.zeros(batch, dtype=torch.int64, device=target),
        status=torch.zeros(batch, dtype=torch.int32, device=target),
    )
    staging.validate(batch=batch, capacity=capacity)
    return staging


def _identity_staging_tensors_r11_v1(
    staging: TensorWorldC4IdentityStagingR11V1,
) -> list[torch.Tensor]:
    return [
        staging.death_slot_tokens, staging.parent_slot_tokens,
        staging.mate_slot_tokens, staging.child_slot_tokens,
        staging.next_agent_id_before, staging.status,
    ]


@dataclass(frozen=True, slots=True)
class TensorWorldC4IdentityAuthorityR11V1:
    """Views into the canonical V4 identity/slot projections.

    These are aliases, not a second mutable identity store.  The physical
    agent tensor is the slot-to-ID authority; the active arrays are its sorted
    lookup projection, and ``slot_tokens`` is the ABA/reuse authority.
    """

    slot_to_agent_id: torch.Tensor
    slot_tokens: torch.Tensor
    next_agent_id: torch.Tensor
    active_agent_ids: torch.Tensor
    active_agent_slots: torch.Tensor
    active_count: torch.Tensor
    schema_version: str = C4_IDENTITY_R11_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.slot_to_agent_id.device

    @property
    def batch(self) -> int:
        return int(self.slot_to_agent_id.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.slot_to_agent_id.shape[1])

    def validate(self, *, semantic: bool = True) -> None:
        if self.schema_version != C4_IDENTITY_R11_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("R11 identity schema is unsupported")
        if self.slot_to_agent_id.ndim != 2:
            raise TensorWorldContractError("R11 slot_to_agent_id must be rank two")
        batch, capacity = self.slot_to_agent_id.shape
        expected = (
            ("slot_to_agent_id", self.slot_to_agent_id, (batch, capacity), torch.int64),
            ("slot_tokens", self.slot_tokens, (batch, capacity), torch.int64),
            ("next_agent_id", self.next_agent_id, (batch,), torch.int64),
            ("active_agent_ids", self.active_agent_ids, (batch, capacity), torch.int64),
            ("active_agent_slots", self.active_agent_slots, (batch, capacity), torch.int32),
            ("active_count", self.active_count, (batch,), torch.int64),
        )
        devices = {self.device}
        for name, tensor, shape, dtype in expected:
            if not isinstance(tensor, torch.Tensor):
                raise TensorWorldContractError(f"R11 identity {name} is not a tensor")
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise TensorWorldContractError(
                    f"R11 identity {name} requires shape={shape!r} dtype={dtype}"
                )
            if not tensor.is_contiguous():
                raise TensorWorldContractError(f"R11 identity {name} must be contiguous")
            devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("R11 identity tensors must share one device")
        if not semantic:
            return
        if self.device.type != "cpu":
            # Semantic validation is deliberately a construction/boundary
            # operation.  The resident path uses device kernels instead of
            # synchronizing to validate every tick.
            return
        for world in range(batch):
            count = int(self.active_count[world])
            if count < 0 or count > capacity:
                raise TensorWorldContractError("R11 active_count is outside capacity")
            previous = -1
            seen_slots: set[int] = set()
            live_max = -1
            for row in range(capacity):
                slot_id = int(self.slot_to_agent_id[world, row])
                token = int(self.slot_tokens[world, row])
                if token < 0:
                    raise TensorWorldContractError("R11 slot token is negative")
                if slot_id >= 0:
                    live_max = max(live_max, slot_id)
            for row in range(count):
                agent_id = int(self.active_agent_ids[world, row])
                slot = int(self.active_agent_slots[world, row])
                if agent_id < 0 or slot < 0 or slot >= capacity:
                    raise TensorWorldContractError("R11 active identity row is invalid")
                if row and agent_id <= previous:
                    raise TensorWorldContractError(
                        "R11 active identities must be strictly ID ordered"
                    )
                if slot in seen_slots or int(self.slot_to_agent_id[world, slot]) != agent_id:
                    raise TensorWorldContractError(
                        "R11 active identity projection disagrees with slot authority"
                    )
                seen_slots.add(slot)
                previous = agent_id
            if int(self.next_agent_id[world]) <= live_max:
                raise TensorWorldContractError(
                    "R11 next_agent_id is not monotonic above live identities"
                )
            for row in range(count, capacity):
                if int(self.active_agent_ids[world, row]) != -1 or int(
                    self.active_agent_slots[world, row]
                ) != -1:
                    raise TensorWorldContractError(
                        "R11 inactive active-projection row is not cleared"
                    )


def bind_c4_identity_authority_r11_v1(
    bundle: TensorWorldStateBundleV4, *, semantic: bool = True
) -> TensorWorldC4IdentityAuthorityR11V1:
    """Bind the R11 identity views to one V4 bundle without allocating state."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("R11 identity binding requires a V4 bundle")
    authority = TensorWorldC4IdentityAuthorityR11V1(
        slot_to_agent_id=bundle.base.base.core.agents.agent_id,
        slot_tokens=bundle.population.slot_tokens,
        next_agent_id=bundle.population.next_agent_id,
        active_agent_ids=bundle.population.active_agent_ids,
        active_agent_slots=bundle.population.active_agent_slots,
        active_count=bundle.population.active_count,
    )
    authority.validate(semantic=semantic)
    return authority


def resolve_c4_identity_slot_r11_v1(
    authority: TensorWorldC4IdentityAuthorityR11V1,
    *,
    world: int,
    agent_id: int,
) -> int | None:
    """Resolve one biological ID using sorted active pairs, never dense indexing."""

    authority.validate(semantic=True)
    if type(world) is not int or not 0 <= world < authority.batch:
        raise TensorWorldContractError("R11 identity world is outside the batch")
    if type(agent_id) is not int or agent_id < 0:
        return None
    count = int(authority.active_count[world])
    ids = authority.active_agent_ids[world, :count]
    index = int(torch.searchsorted(ids, torch.tensor(agent_id, dtype=torch.int64)).item())
    if index >= count or int(ids[index]) != agent_id:
        return None
    slot = int(authority.active_agent_slots[world, index])
    return slot


def c4_identity_authority_r11_source_receipt_v1() -> dict[str, object]:
    return {
        "schema_version": C4_IDENTITY_R11_SCHEMA_VERSION_V1,
        "decision": C4_IDENTITY_R11_DECISION_V1,
        "authority": "core.agents.agent_id plus sorted active_agent_ids/slots",
        "slot_tokens": "population.slot_tokens; increment on committed reuse",
        "next_agent_id": "population.next_agent_id; monotonic int64 independent of capacity",
        "lookup": "sorted active ID/slot pairs with binary search",
        "legacy_dense_map": "compatibility projection only; not authoritative",
        "native_mode": "composed_R11_sorted_monotonic; R2_R4_R5_R6_R9_legacy_dense",
        "aba_preflight": "count-masked expected slot tokens before guarded commit",
        "status_codes": dict(C4_IDENTITY_R11_STATUS_V1),
        "int32_projection_limit": "fail_closed_above_INT32_MAX_pending_projection_widening",
        "gate_claim": False,
    }


__all__ = [
    "C4_IDENTITY_R11_DECISION_V1",
    "C4_IDENTITY_R11_SCHEMA_VERSION_V1",
    "C4_IDENTITY_R11_STAGING_SCHEMA_VERSION_V1",
    "C4_IDENTITY_R11_STATUS_V1",
    "TensorWorldC4IdentityStagingR11V1",
    "TensorWorldC4IdentityAuthorityR11V1",
    "bind_c4_identity_authority_r11_v1",
    "c4_identity_authority_r11_source_receipt_v1",
    "make_c4_identity_staging_r11_v1",
    "resolve_c4_identity_slot_r11_v1",
]
