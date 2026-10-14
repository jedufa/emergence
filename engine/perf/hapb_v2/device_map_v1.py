"""CPU-tested HAPB V2 device-owned hot execution map.

Host remains the canonical authority.  The device map is reconstructible and
non-authoritative.  This module does not mutate the V1 production owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Final, Iterable, Mapping

import numpy as np


DEVICE_MAP_SCHEMA_V1: Final[str] = "eiel_hapb_v2_device_hot_map_v1"
UNBOUND_CONTENT_V1: Final[int] = -1
UNBOUND_VERSION_V1: Final[int] = 0


class HapbV2DeviceMapError(ValueError):
    """The experimental device map rejected an operation."""


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise HapbV2DeviceMapError(f"{name} must be an int >= {minimum}")
    return value


@dataclass
class HapbV2DeviceHotMapV1:
    cells: int
    slots_per_cell: int
    generation_epoch: int = 1
    content_ids: np.ndarray = field(init=False)
    content_versions: np.ndarray = field(init=False)
    slot_generations: np.ndarray = field(init=False)
    valid: np.ndarray = field(init=False)
    observation_gather: np.ndarray = field(init=False)
    logit_scatter: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        cells = _require_int("cells", self.cells, minimum=1)
        slots = _require_int("slots_per_cell", self.slots_per_cell, minimum=1)
        if type(self.generation_epoch) is not int or self.generation_epoch < 1:
            raise HapbV2DeviceMapError("generation epoch must be >= 1")
        shape = (cells, slots)
        self.content_ids = np.full(shape, UNBOUND_CONTENT_V1, dtype=np.int64)
        self.content_versions = np.zeros(shape, dtype=np.int64)
        self.slot_generations = np.zeros(shape, dtype=np.int64)
        self.valid = np.zeros(shape, dtype=np.bool_)
        self.observation_gather = np.arange(cells * slots, dtype=np.int64).reshape(shape)
        self.logit_scatter = np.arange(cells * slots, dtype=np.int64).reshape(shape)

    def _coords(self, cell: int, slot: int) -> tuple[int, int]:
        cell = _require_int("cell", cell)
        slot = _require_int("slot", slot)
        if cell >= self.cells or slot >= self.slots_per_cell:
            raise HapbV2DeviceMapError("slot coordinates are outside the map")
        return cell, slot

    def invalidate_slot_v1(self, cell: int, slot: int) -> None:
        cell, slot = self._coords(cell, slot)
        self.valid[cell, slot] = False
        self.content_ids[cell, slot] = UNBOUND_CONTENT_V1
        self.content_versions[cell, slot] = UNBOUND_VERSION_V1
        self.slot_generations[cell, slot] += 1

    def invalidate_all_v1(self) -> None:
        for cell in range(self.cells):
            for slot in range(self.slots_per_cell):
                self.invalidate_slot_v1(cell, slot)
        self.generation_epoch += 1

    def publish_binding_v1(
        self,
        *,
        cell: int,
        slot: int,
        content_id: int,
        content_version: int,
    ) -> None:
        cell, slot = self._coords(cell, slot)
        content_id = _require_int("content_id", content_id, minimum=0)
        content_version = _require_int("content_version", content_version, minimum=1)
        self.slot_generations[cell, slot] += 1
        self.content_ids[cell, slot] = content_id
        self.content_versions[cell, slot] = content_version
        self.valid[cell, slot] = True

    def require_live_binding_v1(
        self,
        *,
        cell: int,
        slot: int,
        expected_content_id: int,
        expected_version: int,
        expected_slot_generation: int | None = None,
    ) -> int:
        cell, slot = self._coords(cell, slot)
        if not bool(self.valid[cell, slot]):
            raise HapbV2DeviceMapError("device map slot is invalid")
        content_id = int(self.content_ids[cell, slot])
        version = int(self.content_versions[cell, slot])
        generation = int(self.slot_generations[cell, slot])
        if content_id != expected_content_id:
            raise HapbV2DeviceMapError("stale or mismatched content id")
        if version != expected_version:
            raise HapbV2DeviceMapError("stale content version")
        if (
            expected_slot_generation is not None
            and generation != expected_slot_generation
        ):
            raise HapbV2DeviceMapError("slot ABA generation mismatch")
        return content_id

    def alias_clone_v1(
        self,
        *,
        cell: int,
        parent_slot: int,
        child_slot: int,
    ) -> None:
        parent = self.require_live_binding_v1(
            cell=cell,
            slot=parent_slot,
            expected_content_id=int(self.content_ids[cell, parent_slot]),
            expected_version=int(self.content_versions[cell, parent_slot]),
        )
        self.publish_binding_v1(
            cell=cell,
            slot=child_slot,
            content_id=parent,
            content_version=int(self.content_versions[cell, parent_slot]),
        )

    def reconstruct_from_host_v1(
        self,
        bindings: Iterable[Mapping[str, int]],
        *,
        generation_epoch: int | None = None,
    ) -> None:
        self.invalidate_all_v1()
        if generation_epoch is not None:
            if type(generation_epoch) is not int or generation_epoch < 1:
                raise HapbV2DeviceMapError("reconstructed generation epoch is invalid")
            self.generation_epoch = generation_epoch
        for binding in bindings:
            self.publish_binding_v1(
                cell=int(binding["cell"]),
                slot=int(binding["slot"]),
                content_id=int(binding["content_id"]),
                content_version=int(binding["content_version"]),
            )

    def checkpoint_metadata_v1(self) -> dict[str, object]:
        payload = {
            "schema_version": DEVICE_MAP_SCHEMA_V1,
            "cells": self.cells,
            "slots_per_cell": self.slots_per_cell,
            "generation_epoch": self.generation_epoch,
            "content_ids": np.ascontiguousarray(self.content_ids).tobytes(order="C"),
            "content_versions": np.ascontiguousarray(
                self.content_versions
            ).tobytes(order="C"),
            "slot_generations": np.ascontiguousarray(
                self.slot_generations
            ).tobytes(order="C"),
            "valid": np.ascontiguousarray(self.valid.astype(np.uint8)).tobytes(
                order="C"
            ),
        }
        digest = hashlib.sha256()
        for key in (
            "cells",
            "slots_per_cell",
            "generation_epoch",
            "content_ids",
            "content_versions",
            "slot_generations",
            "valid",
        ):
            value = payload[key]
            if isinstance(value, bytes):
                digest.update(value)
            else:
                digest.update(str(value).encode("utf-8"))
        payload["sha256"] = digest.hexdigest()
        return payload

    @classmethod
    def restore_checkpoint_metadata_v1(
        cls,
        payload: Mapping[str, object],
    ) -> "HapbV2DeviceHotMapV1":
        if payload.get("schema_version") != DEVICE_MAP_SCHEMA_V1:
            raise HapbV2DeviceMapError("device-map checkpoint schema mismatch")
        restored = cls(
            cells=int(payload["cells"]),
            slots_per_cell=int(payload["slots_per_cell"]),
            generation_epoch=int(payload["generation_epoch"]),
        )
        shape = (restored.cells, restored.slots_per_cell)
        restored.content_ids = np.frombuffer(
            payload["content_ids"], dtype=np.int64
        ).reshape(shape).copy()
        restored.content_versions = np.frombuffer(
            payload["content_versions"], dtype=np.int64
        ).reshape(shape).copy()
        restored.slot_generations = np.frombuffer(
            payload["slot_generations"], dtype=np.int64
        ).reshape(shape).copy()
        restored.valid = np.frombuffer(
            payload["valid"], dtype=np.uint8
        ).reshape(shape).astype(bool).copy()
        if restored.checkpoint_metadata_v1()["sha256"] != payload.get("sha256"):
            raise HapbV2DeviceMapError("device-map checkpoint digest mismatch")
        return restored
