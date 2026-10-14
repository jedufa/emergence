"""Default-enabled, observational WRC live-reducer integration.

This module is the small adapter boundary for a WRC reporting callback.  It
does not know about simulator internals and it never owns the raw metrics
writer.  The callback may therefore send the same normalized measurements to
the raw writer and to this adapter without changing simulation state.

WP1 intentionally supports ``retain_all`` only.  The resume and transactional
retention modes are separate later work-package boundaries.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_sha256, pretty_json
from .config import REDUCER_VERSION, ReducerConfig
from .core import LiveReducerSink
from .model import ReducerInputError, WorldIdentity


LIVE_INTEGRATION_SCHEMA_VERSION = "eiel_wrc_live_integration_v1"
WRC_LIVE_REDUCER_DEFAULT_ENABLED = True
WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE = "retain_all"


def _identity_from_payload(value: Mapping[str, Any]) -> WorldIdentity:
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ReducerInputError("live reducer identity metadata must be an object")
    return WorldIdentity(
        world_id=str(value["world_id"]),
        parameter_id=str(value["parameter_id"]),
        parameter_value=value.get("parameter_value"),
        seed=int(value["seed"]),
        ticks_per_observation=int(value.get("ticks_per_observation", 1)),
        metadata=dict(metadata),
    )


class WrcLiveReducer:
    """WRC-scoped default-enabled adapter over the shared reducer core.

    ``ingest`` is intended to be called at the existing WRC reporting
    boundary.  When disabled, it is a strict no-op: no reducer object is
    created, no metrics are copied, and no RNG or simulation callback is
    touched.  When enabled, all reduction is delegated to ``LiveReducerSink``
    and therefore uses the same ``WorldReducer`` as offline replay.
    """

    def __init__(
        self,
        identity: WorldIdentity,
        *,
        live_reducer_enable: bool = WRC_LIVE_REDUCER_DEFAULT_ENABLED,
        config: ReducerConfig | None = None,
        source_uri: str | None = None,
        retention_mode: str = WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE,
    ) -> None:
        if not isinstance(live_reducer_enable, bool):
            raise ReducerInputError("live_reducer_enable must be a boolean")
        if retention_mode != WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE:
            raise ReducerInputError(
                "WP1 live integration supports only retain_all; retention modes "
                "belong to the later retention work package"
            )
        self.identity = identity
        self.enabled = live_reducer_enable
        self.retention_mode = retention_mode
        self.source_uri = source_uri or f"live://{identity.world_id}"
        self._sink = (
            LiveReducerSink(identity, config=config, source_uri=self.source_uri)
            if self.enabled
            else None
        )
        self._finalized = False
        self._finalized_result: dict[str, Any] | None = None

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def observation_count(self) -> int:
        if self._sink is None:
            return 0
        return self._sink.reducer.observation_count

    def ingest(
        self,
        *,
        sequence: int,
        metrics: Mapping[str, int | float],
        tick: int | None = None,
        event_id: str | None = None,
    ) -> None:
        """Consume one normalized WRC reporting-boundary observation."""

        if self._finalized:
            raise ReducerInputError("cannot ingest after live reducer finalization")
        if self._sink is not None:
            self._sink.ingest(
                sequence=sequence,
                tick=tick,
                metrics=metrics,
                event_id=event_id,
            )

    observe = ingest

    def finalize(self) -> dict[str, Any] | None:
        """Finalize the world, or return ``None`` for an explicit opt-out.

        Successful finalization is idempotent.  A failed finalization leaves
        the adapter unfinalized so the caller can retain raw evidence and
        recover according to the surrounding campaign policy.
        """

        if self._finalized_result is not None:
            return deepcopy(self._finalized_result)
        if self._sink is None:
            self._finalized = True
            return None
        result = self._sink.finalize()
        self._finalized = True
        self._finalized_result = deepcopy(result)
        return deepcopy(result)

    def snapshot(self) -> dict[str, Any]:
        """Return a canonical, versioned reducer-only snapshot."""

        reducer_state = self._sink.snapshot() if self._sink is not None else None
        payload: dict[str, Any] = {
            "schema_version": LIVE_INTEGRATION_SCHEMA_VERSION,
            "reducer_version": REDUCER_VERSION,
            "enabled": self.enabled,
            "retention_mode": self.retention_mode,
            "finalized": self._finalized,
            "source_uri": self.source_uri,
            "identity": self.identity.payload(),
            "identity_sha256": self.identity.sha256,
            "reducer_state": reducer_state,
        }
        payload["snapshot_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_snapshot(
        cls,
        value: Mapping[str, Any],
        *,
        source_uri: str | None = None,
    ) -> "WrcLiveReducer":
        if value.get("schema_version") != LIVE_INTEGRATION_SCHEMA_VERSION:
            raise ReducerInputError("unsupported live reducer integration schema")
        if value.get("reducer_version") != REDUCER_VERSION:
            raise ReducerInputError("live reducer version differs")
        expected_hash = value.get("snapshot_sha256")
        material = dict(value)
        material.pop("snapshot_sha256", None)
        if expected_hash != canonical_sha256(material):
            raise ReducerInputError("live reducer snapshot hash differs")
        enabled = value.get("enabled")
        if not isinstance(enabled, bool):
            raise ReducerInputError("live reducer snapshot enabled flag is invalid")
        retention_mode = value.get("retention_mode")
        if retention_mode != WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE:
            raise ReducerInputError("live reducer snapshot retention mode is invalid")
        finalized = value.get("finalized")
        if not isinstance(finalized, bool):
            raise ReducerInputError("live reducer snapshot finalized flag is invalid")
        identity_raw = value.get("identity")
        if not isinstance(identity_raw, Mapping):
            raise ReducerInputError("live reducer snapshot identity is missing")
        identity = _identity_from_payload(identity_raw)
        if value.get("identity_sha256") != identity.sha256:
            raise ReducerInputError("live reducer snapshot identity hash differs")
        resolved_source_uri = str(
            source_uri or value.get("source_uri") or f"live://{identity.world_id}"
        )
        result = cls(
            identity,
            live_reducer_enable=enabled,
            source_uri=resolved_source_uri,
            retention_mode=retention_mode,
        )
        reducer_state = value.get("reducer_state")
        if enabled:
            if not isinstance(reducer_state, Mapping):
                raise ReducerInputError("enabled live reducer snapshot lacks state")
            result._sink = LiveReducerSink.from_snapshot(
                reducer_state,
                source_uri=resolved_source_uri,
            )
            if result._sink.reducer.identity.sha256 != identity.sha256:
                raise ReducerInputError("live reducer state identity differs")
        elif reducer_state is not None:
            raise ReducerInputError("disabled live reducer snapshot contains state")
        result._finalized = finalized
        return result

    def write_snapshot(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(pretty_json(self.snapshot()), encoding="utf-8")

    @classmethod
    def from_snapshot_file(
        cls,
        path: str | Path,
        *,
        source_uri: str | None = None,
    ) -> "WrcLiveReducer":
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReducerInputError(
                f"cannot restore live reducer snapshot {source}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ReducerInputError("live reducer snapshot root must be an object")
        return cls.from_snapshot(value, source_uri=source_uri)

    def status(self) -> dict[str, Any]:
        """Return a bounded operator-facing integration status."""

        last_sequence = None
        if self._sink is not None:
            last_sequence = self._sink.reducer.last_sequence
        return {
            "schema_version": LIVE_INTEGRATION_SCHEMA_VERSION,
            "reducer_version": REDUCER_VERSION,
            "enabled": self.enabled,
            "retention_mode": self.retention_mode,
            "finalized": self._finalized,
            "observation_count": self.observation_count,
            "last_sequence": last_sequence,
            "world_id": self.identity.world_id,
        }


def create_wrc_live_reducer(
    identity: WorldIdentity,
    *,
    live_reducer_enable: bool = WRC_LIVE_REDUCER_DEFAULT_ENABLED,
    config: ReducerConfig | None = None,
    source_uri: str | None = None,
) -> WrcLiveReducer:
    """Create the default WRC live adapter at an existing reporting seam."""

    return WrcLiveReducer(
        identity,
        live_reducer_enable=live_reducer_enable,
        config=config,
        source_uri=source_uri,
    )


__all__ = [
    "LIVE_INTEGRATION_SCHEMA_VERSION",
    "WRC_LIVE_REDUCER_DEFAULT_ENABLED",
    "WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE",
    "WrcLiveReducer",
    "create_wrc_live_reducer",
]
