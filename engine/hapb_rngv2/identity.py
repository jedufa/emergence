"""Semantic HAPB identity. Physical slots and host pages are never RNG inputs."""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.tw_rngv2.address import (
    DOMAIN_BY_ID,
    encode_counter,
    encode_key,
    validate_fields,
)
from engine.tw_rngv2.status import Status

from .contract import HAPB_ONLY_DOMAINS, MASK48, SEMANTIC_EPOCH


class HapbIdentityError(ValueError):
    """Fail-closed semantic/physical identity error."""


def hapb_known_domain(domain_id: int) -> bool:
    return int(domain_id) in DOMAIN_BY_ID or int(domain_id) in HAPB_ONLY_DOMAINS


def hapb_validate_fields(
    *,
    world_uid: int,
    domain_id: int,
    semantic_epoch: int,
    subject_uid: int,
    absolute_tick: int,
    operation_id: int,
    draw_block: int,
) -> Status:
    status = validate_fields(
        world_uid=world_uid,
        domain_id=domain_id,
        semantic_epoch=semantic_epoch,
        subject_uid=subject_uid,
        absolute_tick=absolute_tick,
        operation_id=operation_id,
        draw_block=draw_block,
        require_known_domain=False,
    )
    if status is not Status.RNG_OK:
        return status
    if not hapb_known_domain(domain_id):
        return Status.RNG_UNKNOWN_DOMAIN
    return Status.RNG_OK


def hapb_key_words(
    *,
    rng_namespace_key64: int,
    world_uid: int,
    domain_id: int,
    semantic_epoch: int = SEMANTIC_EPOCH,
) -> tuple[int, int]:
    return encode_key(
        rng_namespace_key64=rng_namespace_key64,
        world_uid=world_uid,
        domain_id=domain_id,
        semantic_epoch=semantic_epoch,
    )


def hapb_counter_words(
    *,
    subject_uid: int,
    absolute_tick: int,
    operation_id: int,
    draw_block: int,
) -> tuple[int, int, int, int]:
    return encode_counter(
        subject_uid=subject_uid,
        absolute_tick=absolute_tick,
        operation_id=operation_id,
        draw_block=draw_block,
    )


@dataclass(slots=True)
class HapbIdentityMap:
    """World-local semantic UIDs with a remappable physical slot table.

    RNG addresses use only ``uid_at``. Slot reuse, FRAG remapping, host-page
    movement, and compaction change the table, not the UID.
    """

    cells: int
    slots: int
    next_entity_uid: list[int] = field(default_factory=list)
    slot_to_uid: list[list[int | None]] = field(default_factory=list)
    page_of_slot: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cells < 1 or self.slots < 1:
            raise HapbIdentityError("identity map requires positive cells and slots")
        if not self.next_entity_uid:
            self.next_entity_uid = [1] * self.cells
        if not self.slot_to_uid:
            self.slot_to_uid = [[None] * self.slots for _ in range(self.cells)]
        if not self.page_of_slot:
            self.page_of_slot = [
                [slot for slot in range(self.slots)] for _ in range(self.cells)
            ]
        if (
            len(self.next_entity_uid) != self.cells
            or len(self.slot_to_uid) != self.cells
            or any(len(row) != self.slots for row in self.slot_to_uid)
        ):
            raise HapbIdentityError("identity map shape is inconsistent")

    def allocate(self, cell: int, slot: int) -> int:
        self._check(cell, slot)
        if self.slot_to_uid[cell][slot] is not None:
            raise HapbIdentityError("physical slot already holds a live identity")
        uid = int(self.next_entity_uid[cell])
        if uid < 1 or uid > MASK48:
            raise HapbIdentityError("entity uid overflow")
        self.slot_to_uid[cell][slot] = uid
        self.next_entity_uid[cell] = uid + 1
        return uid

    def release(self, cell: int, slot: int) -> int:
        uid = self.uid_at(cell, slot)
        if uid is None:
            raise HapbIdentityError("physical slot is empty")
        self.slot_to_uid[cell][slot] = None
        return uid

    def uid_at(self, cell: int, slot: int) -> int | None:
        self._check(cell, slot)
        value = self.slot_to_uid[cell][slot]
        return None if value is None else int(value)

    def remap_physical(self, cell: int, source_slot: int, dest_slot: int) -> int:
        """Move a live identity to another physical slot. UID is unchanged."""

        uid = self.release(cell, source_slot)
        if self.slot_to_uid[cell][dest_slot] is not None:
            raise HapbIdentityError("destination physical slot is occupied")
        self.slot_to_uid[cell][dest_slot] = uid
        self.page_of_slot[cell][dest_slot] = self.page_of_slot[cell][source_slot]
        return uid

    def move_host_page(self, cell: int, slot: int, page: int) -> None:
        self._check(cell, slot)
        if page < 0:
            raise HapbIdentityError("host page must be nonnegative")
        self.page_of_slot[cell][slot] = int(page)

    def compact(self, cell: int) -> None:
        """Pack live identities into the lowest physical slots. UIDs stay."""

        self._check(cell, 0)
        live_uids: list[int | None] = []
        live_pages: list[int] = []
        unused_pages: list[int] = []
        for slot in range(self.slots):
            uid = self.slot_to_uid[cell][slot]
            page = int(self.page_of_slot[cell][slot])
            if uid is None:
                unused_pages.append(page)
            else:
                live_uids.append(uid)
                live_pages.append(page)
        pad = self.slots - len(live_uids)
        self.slot_to_uid[cell] = live_uids + [None] * pad
        self.page_of_slot[cell] = live_pages + unused_pages

    def snapshot(self) -> dict[str, object]:
        return {
            "cells": self.cells,
            "slots": self.slots,
            "next_entity_uid": list(self.next_entity_uid),
            "slot_to_uid": [
                [(-1 if uid is None else int(uid)) for uid in row]
                for row in self.slot_to_uid
            ],
            "page_of_slot": [list(row) for row in self.page_of_slot],
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> "HapbIdentityMap":
        cells = int(payload["cells"])
        slots = int(payload["slots"])
        raw_slots = payload["slot_to_uid"]
        slot_to_uid = [
            [None if int(uid) < 0 else int(uid) for uid in row] for row in raw_slots
        ]
        return cls(
            cells=cells,
            slots=slots,
            next_entity_uid=[int(value) for value in payload["next_entity_uid"]],
            slot_to_uid=slot_to_uid,
            page_of_slot=[
                [int(page) for page in row] for row in payload["page_of_slot"]
            ],
        )

    def _check(self, cell: int, slot: int) -> None:
        if cell < 0 or cell >= self.cells or slot < 0 or slot >= self.slots:
            raise HapbIdentityError("cell/slot is outside the identity map")
