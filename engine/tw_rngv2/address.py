"""Event-addressed key/counter codec for TW-RNGV2-DEA1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final, Iterable, Sequence

from .contract import CONTRACT, MASK16, MASK32, MASK48, SEMANTIC_EPOCH
from .status import Status


class AddressError(ValueError):
    """Fail-closed address/codec error."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status = status


_FE_DOMAIN_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "_spec"
    / "campaigns"
    / "v1"
    / "tw_rngv2"
    / "full_evolution_domains_v1.json"
)


def _merged_domain_registry() -> list[dict[str, object]]:
    rows = [dict(row) for row in CONTRACT["proposed_domain_registry"]]
    known = {int(row["id"]) for row in rows}
    extra = json.loads(_FE_DOMAIN_PATH.read_text(encoding="utf-8"))
    for row in extra.get("domains", ()):
        domain_id = int(row["id"])
        if domain_id in known:
            continue
        rows.append(
            {
                "id": domain_id,
                "id_hex": row["id_hex"],
                "name": row["name"],
                "subject": row.get("subject", ""),
            }
        )
        known.add(domain_id)
    return rows


DOMAIN_BY_ID: Final[dict[int, dict[str, object]]] = {
    int(row["id"]): row for row in _merged_domain_registry()
}
DOMAIN_BY_NAME: Final[dict[str, dict[str, object]]] = {
    str(row["name"]): row for row in DOMAIN_BY_ID.values()
}


def validate_fields(
    *,
    world_uid: int,
    domain_id: int,
    semantic_epoch: int,
    subject_uid: int,
    absolute_tick: int,
    operation_id: int,
    draw_block: int,
    require_known_domain: bool = True,
) -> Status:
    if not (0 <= world_uid <= MASK32):
        return Status.RNG_FIELD_OVERFLOW
    if not (0 <= domain_id <= MASK16):
        return Status.RNG_FIELD_OVERFLOW
    if not (0 <= semantic_epoch <= MASK16):
        return Status.RNG_FIELD_OVERFLOW
    if not (0 <= subject_uid <= MASK48):
        return Status.RNG_FIELD_OVERFLOW
    if not (0 <= absolute_tick <= MASK48):
        return Status.RNG_FIELD_OVERFLOW
    if not (0 <= operation_id <= MASK16):
        return Status.RNG_FIELD_OVERFLOW
    if not (0 <= draw_block <= MASK16):
        return Status.RNG_FIELD_OVERFLOW
    if require_known_domain and domain_id not in DOMAIN_BY_ID:
        return Status.RNG_UNKNOWN_DOMAIN
    return Status.RNG_OK


def encode_key(
    *,
    rng_namespace_key64: int,
    world_uid: int,
    domain_id: int,
    semantic_epoch: int = SEMANTIC_EPOCH,
) -> tuple[int, int]:
    packed = (
        (world_uid & MASK32)
        | ((domain_id & MASK16) << 32)
        | ((semantic_epoch & MASK16) << 48)
    )
    key = (rng_namespace_key64 & ((1 << 64) - 1)) ^ packed
    return key & MASK32, (key >> 32) & MASK32


def encode_counter(
    *,
    subject_uid: int,
    absolute_tick: int,
    operation_id: int,
    draw_block: int,
) -> tuple[int, int, int, int]:
    packed = (
        (subject_uid & MASK48)
        | ((absolute_tick & MASK48) << 48)
        | ((operation_id & MASK16) << 96)
        | ((draw_block & MASK16) << 112)
    )
    return (
        packed & MASK32,
        (packed >> 32) & MASK32,
        (packed >> 64) & MASK32,
        (packed >> 96) & MASK32,
    )


def decode_counter(ctr: Sequence[int]) -> dict[str, int]:
    packed = (
        int(ctr[0])
        | (int(ctr[1]) << 32)
        | (int(ctr[2]) << 64)
        | (int(ctr[3]) << 96)
    )
    return {
        "subject_uid": packed & MASK48,
        "absolute_tick": (packed >> 48) & MASK48,
        "operation_id": (packed >> 96) & MASK16,
        "draw_block": (packed >> 112) & MASK16,
    }


@dataclass(frozen=True, slots=True, order=True)
class EventAddress:
    world_uid: int
    domain_id: int
    subject_uid: int
    absolute_tick: int
    operation_id: int
    semantic_epoch: int = SEMANTIC_EPOCH

    def validate(self) -> Status:
        return validate_fields(
            world_uid=self.world_uid,
            domain_id=self.domain_id,
            semantic_epoch=self.semantic_epoch,
            subject_uid=self.subject_uid,
            absolute_tick=self.absolute_tick,
            operation_id=self.operation_id,
            draw_block=0,
        )

    def key_words(self, rng_namespace_key64: int) -> tuple[int, int]:
        status = self.validate()
        if status is not Status.RNG_OK:
            raise AddressError(status, f"invalid address: {status.name}")
        return encode_key(
            rng_namespace_key64=rng_namespace_key64,
            world_uid=self.world_uid,
            domain_id=self.domain_id,
            semantic_epoch=self.semantic_epoch,
        )

    def counter_words(self, draw_block: int) -> tuple[int, int, int, int]:
        status = validate_fields(
            world_uid=self.world_uid,
            domain_id=self.domain_id,
            semantic_epoch=self.semantic_epoch,
            subject_uid=self.subject_uid,
            absolute_tick=self.absolute_tick,
            operation_id=self.operation_id,
            draw_block=draw_block,
        )
        if status is not Status.RNG_OK:
            raise AddressError(status, f"invalid address: {status.name}")
        return encode_counter(
            subject_uid=self.subject_uid,
            absolute_tick=self.absolute_tick,
            operation_id=self.operation_id,
            draw_block=draw_block,
        )

    def identity_tuple(self, draw_block: int, lane: int) -> tuple[int, ...]:
        return (
            self.semantic_epoch,
            self.world_uid,
            self.domain_id,
            self.subject_uid,
            self.absolute_tick,
            self.operation_id,
            draw_block,
            lane,
        )


def pair_event_rank(
    parent_uids: Sequence[int],
    occurrence: int,
    peers: Sequence[tuple[tuple[int, ...], int]],
) -> int:
    """Rank of a canonical sorted-parent pair among peers.

    Physical slot, launch, and completion order are forbidden inputs.
    """
    canonical = (tuple(sorted(int(uid) for uid in parent_uids)), int(occurrence))
    ordered = sorted(
        (tuple(sorted(int(uid) for uid in parents)), int(occ))
        for parents, occ in peers
    )
    try:
        return ordered.index(canonical)
    except ValueError as exc:
        raise AddressError(
            Status.RNG_FIELD_OVERFLOW, "pair event is not in the peer set"
        ) from exc


def birth_proposal_rank(
    parent_uids: Sequence[int],
    proposal_ordinal: int,
    peers: Sequence[tuple[tuple[int, ...], int]],
) -> int:
    canonical = (tuple(int(uid) for uid in parent_uids), int(proposal_ordinal))
    ordered = sorted(
        (tuple(int(uid) for uid in parents), int(ord_))
        for parents, ord_ in peers
    )
    try:
        return ordered.index(canonical)
    except ValueError as exc:
        raise AddressError(
            Status.RNG_FIELD_OVERFLOW, "birth proposal is not in the peer set"
        ) from exc


def assign_accepted_child_uids(
    *,
    next_entity_uid: int,
    accepted_in_canonical_order: Iterable[bool],
) -> tuple[list[int | None], int]:
    assigned: list[int | None] = []
    cursor = int(next_entity_uid)
    if cursor < 1:
        raise AddressError(Status.RNG_FIELD_OVERFLOW, "next_entity_uid must be >= 1")
    for accepted in accepted_in_canonical_order:
        if accepted:
            if cursor > MASK48:
                raise AddressError(Status.RNG_FIELD_OVERFLOW, "entity uid overflow")
            assigned.append(cursor)
            cursor += 1
        else:
            assigned.append(None)
    return assigned, cursor
