"""Bounded V2 contract for clone reuse and genuinely new unique content.

V1 alias-on binds overflow rows only when they exactly match an existing
per-cell P256 prefix digest.  Genuinely new unique overflow content stays
unbound and fails closed.  This experimental contract keeps host intern
authority and adds an explicit one-shot publication path for new content
without changing production mutation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Final

import numpy as np

from engine.perf.hapb_v2.device_map_v1 import HapbV2DeviceHotMapV1
from engine.perf.hapb_v2.exactness_contract_v1 import POLICY_WIDTH_V1


PUBLICATION_SCHEMA_V1: Final[str] = "eiel_hapb_v2_content_publication_v1"
HOST_CANONICAL_AUTHORITY_V1: Final[str] = "host_canonical_policy_bytes"


class HapbV2PublicationError(ValueError):
    """The experimental unique-content contract rejected an operation."""


def content_digest_v1(row: np.ndarray) -> bytes:
    array = np.asarray(row, dtype=np.float32)
    if array.shape != (POLICY_WIDTH_V1,):
        raise HapbV2PublicationError("policy content row must have width 9991")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).digest()


@dataclass
class HapbV2ContentRecordV1:
    content_id: int
    version: int
    digest: bytes
    row: np.ndarray
    published_to_device: bool


@dataclass
class HapbV2ContentPublicationLedgerV1:
    """Host-canonical intern table plus explicit device publication events."""

    cells: int
    slots_per_cell: int
    next_content_id: int = 1
    records: dict[int, HapbV2ContentRecordV1] = field(default_factory=dict)
    digest_index: dict[bytes, int] = field(default_factory=dict)
    device_map: HapbV2DeviceHotMapV1 = field(init=False)
    publication_events: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.device_map = HapbV2DeviceHotMapV1(
            cells=self.cells, slots_per_cell=self.slots_per_cell
        )

    def intern_host_content_v1(self, row: np.ndarray) -> HapbV2ContentRecordV1:
        digest = content_digest_v1(row)
        existing_id = self.digest_index.get(digest)
        if existing_id is not None:
            return self.records[existing_id]
        content_id = self.next_content_id
        self.next_content_id += 1
        record = HapbV2ContentRecordV1(
            content_id=content_id,
            version=1,
            digest=digest,
            row=np.ascontiguousarray(row, dtype=np.float32).copy(),
            published_to_device=False,
        )
        self.records[content_id] = record
        self.digest_index[digest] = content_id
        return record

    def publish_new_unique_content_v1(
        self,
        *,
        cell: int,
        slot: int,
        row: np.ndarray,
    ) -> dict[str, object]:
        record = self.intern_host_content_v1(row)
        already_published = record.published_to_device
        if not already_published:
            record.published_to_device = True
        self.device_map.publish_binding_v1(
            cell=cell,
            slot=slot,
            content_id=record.content_id,
            content_version=record.version,
        )
        event = {
            "schema_version": PUBLICATION_SCHEMA_V1,
            "kind": "reuse" if already_published else "new_unique_publish",
            "cell": cell,
            "slot": slot,
            "content_id": record.content_id,
            "content_version": record.version,
            "digest_sha256": record.digest.hex(),
            "h2d_row_published": not already_published,
            "authority": HOST_CANONICAL_AUTHORITY_V1,
        }
        self.publication_events.append(event)
        return event

    def alias_exact_clone_v1(
        self,
        *,
        cell: int,
        parent_slot: int,
        child_slot: int,
        parent_row: np.ndarray,
        child_row: np.ndarray,
    ) -> dict[str, object]:
        if content_digest_v1(parent_row) != content_digest_v1(child_row):
            raise HapbV2PublicationError(
                "clone alias requires exact parent/child policy bytes"
            )
        parent_record = self.intern_host_content_v1(parent_row)
        if not parent_record.published_to_device:
            raise HapbV2PublicationError(
                "clone alias requires the parent content to already be visible"
            )
        self.device_map.alias_clone_v1(
            cell=cell, parent_slot=parent_slot, child_slot=child_slot
        )
        event = {
            "schema_version": PUBLICATION_SCHEMA_V1,
            "kind": "exact_clone_alias",
            "cell": cell,
            "parent_slot": parent_slot,
            "child_slot": child_slot,
            "content_id": parent_record.content_id,
            "content_version": parent_record.version,
            "digest_sha256": parent_record.digest.hex(),
            "h2d_row_published": False,
            "authority": HOST_CANONICAL_AUTHORITY_V1,
        }
        self.publication_events.append(event)
        return event

    def bump_mutated_content_v1(
        self,
        *,
        cell: int,
        slot: int,
        new_row: np.ndarray,
    ) -> dict[str, object]:
        """Host intern of mutated bytes, then one bounded H2D publication."""

        digest = content_digest_v1(new_row)
        if digest in self.digest_index:
            return self.publish_new_unique_content_v1(
                cell=cell, slot=slot, row=new_row
            )
        record = self.intern_host_content_v1(new_row)
        record.published_to_device = True
        self.device_map.publish_binding_v1(
            cell=cell,
            slot=slot,
            content_id=record.content_id,
            content_version=record.version,
        )
        event = {
            "schema_version": PUBLICATION_SCHEMA_V1,
            "kind": "mutated_unique_publish",
            "cell": cell,
            "slot": slot,
            "content_id": record.content_id,
            "content_version": record.version,
            "digest_sha256": record.digest.hex(),
            "h2d_row_published": True,
            "authority": HOST_CANONICAL_AUTHORITY_V1,
        }
        self.publication_events.append(event)
        return event

    def invalidate_slot_v1(self, cell: int, slot: int) -> None:
        self.device_map.invalidate_slot_v1(cell, slot)

    def reconstruct_device_from_host_v1(self) -> None:
        bindings = []
        for event in self.publication_events:
            if event["kind"] == "exact_clone_alias":
                bindings.append(
                    {
                        "cell": int(event["cell"]),
                        "slot": int(event["child_slot"]),
                        "content_id": int(event["content_id"]),
                        "content_version": int(event["content_version"]),
                    }
                )
            elif event["kind"] in {"reuse", "new_unique_publish", "mutated_unique_publish"}:
                bindings.append(
                    {
                        "cell": int(event["cell"]),
                        "slot": int(event["slot"]),
                        "content_id": int(event["content_id"]),
                        "content_version": int(event["content_version"]),
                    }
                )
        self.device_map.reconstruct_from_host_v1(
            bindings, generation_epoch=self.device_map.generation_epoch
        )
