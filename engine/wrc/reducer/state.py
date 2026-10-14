"""Durable, reducer-only checkpoint and finalized-world state.

This module deliberately stores no simulator state.  A checkpoint contains a
``WorldReducer`` snapshot and the small amount of reducer lifecycle metadata
needed to make replay/resume idempotent.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_sha256, pretty_json
from .config import REDUCER_VERSION, ReducerConfig
from .core import WorldReducer
from .model import ReducerInputError, WorldIdentity


CHECKPOINT_SCHEMA_VERSION = "eiel_wrc_reducer_checkpoint_v1"
STATE_STORE_SCHEMA_VERSION = "eiel_wrc_reducer_state_store_v1"


class ReducerStateError(ReducerInputError):
    """A reducer checkpoint is malformed, stale, or unsafe to restore."""


class FinalizedWorldError(ReducerStateError):
    """A caller attempted to append to a finalized world."""


def _identity(value: Mapping[str, Any]) -> WorldIdentity:
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ReducerStateError("checkpoint identity metadata must be an object")
    try:
        return WorldIdentity(
            world_id=str(value["world_id"]),
            parameter_id=str(value["parameter_id"]),
            parameter_value=value.get("parameter_value"),
            seed=int(value["seed"]),
            ticks_per_observation=int(value.get("ticks_per_observation", 1)),
            metadata=dict(metadata),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReducerStateError("checkpoint identity is invalid") from exc


def _verified_payload(value: Mapping[str, Any], hash_name: str) -> dict[str, Any]:
    expected = value.get(hash_name)
    material = dict(value)
    material.pop(hash_name, None)
    if not isinstance(expected, str) or expected != canonical_sha256(material):
        raise ReducerStateError(f"checkpoint {hash_name} differs")
    return material


class ReducerCheckpoint:
    """An immutable-in-use representation of reducer progress for one world."""

    def __init__(
        self,
        *,
        reducer_state: Mapping[str, Any],
        finalized: bool = False,
        finalized_result: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(reducer_state, Mapping):
            raise ReducerStateError("reducer checkpoint state must be an object")
        if not isinstance(finalized, bool):
            raise ReducerStateError("checkpoint finalized flag must be boolean")
        if finalized_result is not None and not isinstance(finalized_result, Mapping):
            raise ReducerStateError("checkpoint finalized result must be an object")
        if finalized and finalized_result is None:
            raise ReducerStateError("finalized checkpoint requires finalized result")
        if finalized_result is not None and not finalized:
            raise ReducerStateError("unfinalized checkpoint cannot contain a result")
        self.reducer_state = deepcopy(dict(reducer_state))
        self.finalized = finalized
        self.finalized_result = (
            deepcopy(dict(finalized_result)) if finalized_result is not None else None
        )

    @property
    def identity(self) -> WorldIdentity:
        raw = self.reducer_state.get("identity")
        if not isinstance(raw, Mapping):
            raise ReducerStateError("checkpoint lacks reducer identity")
        return _identity(raw)

    def payload(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "reducer_version": REDUCER_VERSION,
            "reducer_state": self.reducer_state,
            "finalized": self.finalized,
            "finalized_result": self.finalized_result,
        }
        value["checkpoint_sha256"] = canonical_sha256(value)
        return value

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "ReducerCheckpoint":
        if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ReducerStateError("unsupported reducer checkpoint schema")
        if value.get("reducer_version") != REDUCER_VERSION:
            raise ReducerStateError("reducer checkpoint version differs")
        _verified_payload(value, "checkpoint_sha256")
        state = value.get("reducer_state")
        if not isinstance(state, Mapping):
            raise ReducerStateError("reducer checkpoint state is missing")
        checkpoint = cls(
            reducer_state=state,
            finalized=value.get("finalized", False),
            finalized_result=(
                value.get("finalized_result")
                if isinstance(value.get("finalized_result"), Mapping)
                else None
            ),
        )
        # WorldReducer performs the deeper state/config/integrity validation.
        restored = WorldReducer.from_snapshot(checkpoint.reducer_state)
        if restored.identity.sha256 != checkpoint.identity.sha256:
            raise ReducerStateError("checkpoint identity differs from reducer state")
        if checkpoint.finalized and checkpoint.finalized_result is None:
            raise ReducerStateError("finalized checkpoint lacks finalized result")
        return checkpoint

    @classmethod
    def from_reducer(
        cls,
        reducer: WorldReducer,
        *,
        finalized: bool | None = None,
        finalized_result: Mapping[str, Any] | None = None,
    ) -> "ReducerCheckpoint":
        return cls(
            reducer_state=reducer.snapshot(),
            finalized=reducer.finalized if finalized is None else finalized,
            finalized_result=(
                reducer.finalized_result
                if finalized_result is None and reducer.finalized
                else finalized_result
            ),
        )

    def restore(self, *, config: ReducerConfig | None = None) -> WorldReducer:
        reducer = WorldReducer.from_snapshot(self.reducer_state)
        if config is not None and reducer.config.sha256 != config.sha256:
            raise ReducerStateError("checkpoint configuration differs")
        if self.finalized:
            reducer.restore_finalized(self.finalized_result)
        return reducer


class ReducerStateStore:
    """Filesystem-backed per-world reducer checkpoints.

    The store uses hashed filenames, atomic replacement, and a store manifest.
    Existing finalized worlds are never reopened or silently overwritten.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(world_id: str) -> str:
        if not isinstance(world_id, str) or not world_id:
            raise ReducerStateError("world ID must be nonempty")
        return canonical_sha256({"world_id": world_id})

    def path_for(self, world_id: str) -> Path:
        return self.root / f"world-{self._key(world_id)}.json"

    def save(
        self,
        reducer: WorldReducer,
        *,
        finalized: bool | None = None,
        finalized_result: Mapping[str, Any] | None = None,
    ) -> ReducerCheckpoint:
        checkpoint = ReducerCheckpoint.from_reducer(
            reducer, finalized=finalized, finalized_result=finalized_result
        )
        existing = self.load(reducer.identity.world_id)
        if existing is not None and existing.finalized:
            if existing.finalized_result == checkpoint.finalized_result:
                return existing
            raise FinalizedWorldError(
                f"world {reducer.identity.world_id} is already finalized"
            )
        if existing is not None and existing.identity.sha256 != reducer.identity.sha256:
            raise ReducerStateError("checkpoint world identity differs")
        if existing is not None:
            existing_count = int(existing.reducer_state.get("observation_count", 0))
            if existing_count > reducer.observation_count:
                raise ReducerStateError("checkpoint would regress reducer progress")
        destination = self.path_for(reducer.identity.world_id)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(pretty_json(checkpoint.payload()), encoding="utf-8")
        temporary.replace(destination)
        return checkpoint

    def load(self, world_id: str) -> ReducerCheckpoint | None:
        path = self.path_for(world_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReducerStateError(f"cannot read reducer checkpoint {path}") from exc
        if not isinstance(value, Mapping):
            raise ReducerStateError("reducer checkpoint root must be an object")
        checkpoint = ReducerCheckpoint.from_payload(value)
        if checkpoint.identity.world_id != world_id:
            raise ReducerStateError("checkpoint filename/world identity differs")
        return checkpoint

    def restore(
        self,
        identity: WorldIdentity,
        *,
        config: ReducerConfig | None = None,
    ) -> WorldReducer | None:
        checkpoint = self.load(identity.world_id)
        if checkpoint is None:
            return None
        if checkpoint.identity.sha256 != identity.sha256:
            raise ReducerStateError("checkpoint identity differs from requested world")
        return checkpoint.restore(config=config)

    def is_finalized(self, world_id: str) -> bool:
        checkpoint = self.load(world_id)
        return bool(checkpoint and checkpoint.finalized)

    def finalize(
        self, reducer: WorldReducer, result: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if result is None:
            result = reducer.finalize()
        checkpoint = self.save(reducer, finalized=True, finalized_result=result)
        assert checkpoint.finalized_result is not None
        return deepcopy(checkpoint.finalized_result)


def save_checkpoint(
    path: str | Path,
    reducer: WorldReducer,
    *,
    finalized: bool | None = None,
    finalized_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one reducer checkpoint atomically and return its payload."""

    checkpoint = ReducerCheckpoint.from_reducer(
        reducer, finalized=finalized, finalized_result=finalized_result
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(pretty_json(checkpoint.payload()), encoding="utf-8")
    temporary.replace(destination)
    return checkpoint.payload()


def load_checkpoint(path: str | Path) -> ReducerCheckpoint:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReducerStateError(f"cannot read reducer checkpoint {source}") from exc
    if not isinstance(value, Mapping):
        raise ReducerStateError("reducer checkpoint root must be an object")
    return ReducerCheckpoint.from_payload(value)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "STATE_STORE_SCHEMA_VERSION",
    "FinalizedWorldError",
    "ReducerCheckpoint",
    "ReducerStateError",
    "ReducerStateStore",
    "load_checkpoint",
    "save_checkpoint",
]
