"""Deterministic live/replay reducer for WRC simulation trajectories."""

from .config import REDUCER_VERSION, ReducerConfig, reducer_schema
from .core import LiveReducerSink, WorldReducer
from .live import (
    LIVE_INTEGRATION_SCHEMA_VERSION,
    WRC_LIVE_REDUCER_DEFAULT_ENABLED,
    WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE,
    WrcLiveReducer,
    create_wrc_live_reducer,
)
from .model import Observation, ReducerInputError, SourceLocator, WorldIdentity
from .retention import (
    RETAIN_ALL,
    RETAIN_COMPACT,
    RETAIN_SELECTED,
    RetentionError,
    retain,
    retain_all,
    retain_compact,
    retain_selected,
)
from .state import (
    FinalizedWorldError,
    ReducerCheckpoint,
    ReducerStateError,
    ReducerStateStore,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "REDUCER_VERSION",
    "LiveReducerSink",
    "LIVE_INTEGRATION_SCHEMA_VERSION",
    "Observation",
    "ReducerConfig",
    "ReducerInputError",
    "SourceLocator",
    "WorldIdentity",
    "FinalizedWorldError",
    "ReducerCheckpoint",
    "ReducerStateError",
    "ReducerStateStore",
    "RETAIN_ALL",
    "RETAIN_COMPACT",
    "RETAIN_SELECTED",
    "RetentionError",
    "WorldReducer",
    "WRC_LIVE_REDUCER_DEFAULT_ENABLED",
    "WRC_LIVE_REDUCER_DEFAULT_RETENTION_MODE",
    "WrcLiveReducer",
    "create_wrc_live_reducer",
    "load_checkpoint",
    "retain",
    "retain_all",
    "retain_compact",
    "retain_selected",
    "reducer_schema",
    "save_checkpoint",
]
