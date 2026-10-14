"""Prototype transaction: local results, fail-closed publication."""

from __future__ import annotations

from dataclasses import dataclass, field

from .address import EventAddress
from .checkpoint import SemanticCheckpoint, capture_checkpoint
from .oracle import DrawResult, PrototypeOracle


@dataclass(slots=True)
class Transaction:
    oracle: PrototypeOracle
    world_uids: tuple[int, ...]
    absolute_tick: int
    next_entity_uid: tuple[int, ...]
    _local: list[DrawResult] = field(default_factory=list)
    _published: tuple[DrawResult, ...] | None = None
    _aborted: bool = False
    _frontier_at_open: SemanticCheckpoint | None = None

    def __post_init__(self) -> None:
        self._frontier_at_open = capture_checkpoint(
            world_uids=self.world_uids,
            absolute_tick=self.absolute_tick,
            next_entity_uid=self.next_entity_uid,
            rng_namespace_key64=self.oracle.namespace,
        )

    def draw(self, address: EventAddress, *, n: int = 7) -> DrawResult:
        if self._aborted or self._published is not None:
            raise RuntimeError("transaction is closed")
        result = self.oracle.generate(address, n=n)
        self._local.append(result)
        return result

    def abort(self) -> SemanticCheckpoint:
        self._aborted = True
        self._local.clear()
        self.oracle.forget()
        assert self._frontier_at_open is not None
        return self._frontier_at_open

    def commit(self) -> tuple[tuple[DrawResult, ...], SemanticCheckpoint]:
        if self._aborted:
            raise RuntimeError("aborted transaction cannot commit")
        if any(result.status.value != 0 for result in self._local):
            self.abort()
            raise RuntimeError("transaction failed closed on non-OK status")
        self._published = tuple(self._local)
        frontier = capture_checkpoint(
            world_uids=self.world_uids,
            absolute_tick=self.absolute_tick,
            next_entity_uid=self.next_entity_uid,
            rng_namespace_key64=self.oracle.namespace,
        )
        return self._published, frontier

    @property
    def published(self) -> tuple[DrawResult, ...] | None:
        return self._published
