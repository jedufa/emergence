"""Public multicell facade for the fixed-policy Patch resident owner.

This module composes the qualified source allocator, preparation boundary, and
canonical B1 checkpoint settlement.  It deliberately does not construct
``SimulationRunner`` instances: campaign code supplies either complete public
allocation requests or a public runner-to-request factory.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, Final, Protocol

import torch

from engine.perf.cuda_architecture_b_disabled_prediction_state_v1 import (
    DisabledPredictionSettlementError,
    materialize_disabled_prediction_state_v1,
)
from engine.perf.cuda_architecture_b_pcg64 import pcg64_state_tensors
from engine.perf.cxxbackbone10_resident_state_owner import (
    _with_checkpoint_contract_metadata,
    projection_digest,
)

from engine.native_runtime.interchange_projection import (
    validate_native_checkpoint_projection,
)
from engine.perf.cuda_architecture_b_checkpoint import CudaBoundaryCheckpoint
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    PreparedDirectResidentTrajectoryV1,
    prepare_direct_resident_trajectory_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    PHYSICAL_WINDOW_GENERATIONS,
    validate_physical_window_generations_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    GENERATION_SUMMARY_COLUMNS,
    settle_patch_p2_b1_direct_resident_prefix_v2,
    with_resident_inheritance_sidecars_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    allocate_direct_resident_source_v1,
    stack_validated_per_cell_checkpoint_projections_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    DirectResidentSourceAllocationRequestV1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CONTINUITY_SCHEMA_VERSION,
    PATH_GENERATION_BOUNDARY_REFILL,
    SEGMENT_OUTPUT_SCHEMA_VERSION,
    STORAGE_MODE_COMPACT_COMPLETE,
)
from perf.cuda_architecture_b_boundary import EVENT_COLUMN_DTYPES, EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EVENT_SINK_ROW_BYTES,
    EVENT_SINK_SCHEMA_VERSION,
    EventSinkError,
    event_sink_prefix_identities_v1,
    iter_event_sink_columns_v1,
)
from simulation.chronicle.schema import canonical_json


PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_patch_multicell_v1"
)
DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_detached_cell_projection_v1"
)
CHECKPOINT_PROJECTION_CURSOR_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_checkpoint_projection_cursor_v1"
)
CHECKPOINT_PROJECTION_CURSOR_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_checkpoint_projection_cursor_receipt_v1"
)
SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_source_import_cursor_v1"
)
SOURCE_IMPORT_CURSOR_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_source_import_cursor_receipt_v1"
)
_SETTLEMENT_OUTPUT_CELL_AXES_V1: Final[Mapping[str, int]] = {
    "generation_summary_scalar": 1,
    "runtime_bookkeeping_by_tick": 0,
    "birth_combined_status": 0,
    "lifecycle_status": 0,
    "turnover_next_generation": 0,
    "turnover_next_tick": 0,
    "event_status_by_tick": 0,
    "turnover_status": 0,
    "turnover_counts": 0,
    "birth_reason_counts": 0,
}
_TICK_NATIVE_RUNTIME_PREFIX_FIELDS_V1: Final[tuple[str, ...]] = (
    "restore_count",
    "generation_calls",
    "event_cursor",
    "runtime_event_cursor",
    "last_applied_actions",
    "last_moved",
    "last_gathered",
    "last_policy_refill_calls",
    "last_policy_refill_rows",
    "pending_generation_free_agent_ids",
    "pending_generation_free_slots",
    "last_world",
)
_LIVE_PCG64_BUFFER_FIELDS_V1: Final[tuple[str, ...]] = (
    "runner_pcg64_buffer_words",
    "pcg64_buffer_words",
    "population_pcg64_buffer_words",
)
_SHARED_LIVE_BANK_MAPPINGS_V1: Final[frozenset[str]] = frozenset(
    {"genome_v2_observability_config"}
)
_LIVE_PCG64_METADATA_FIELDS_V1: Final[Mapping[str, str]] = {
    "runner": "native_runtime_numpy_runner_state",
    "pcg64": "native_runtime_numpy_policy_state",
    "population": "native_runtime_numpy_population_state",
}
_LIVE_SEMANTIC_DRAW_METADATA_FIELDS_V1: Final[Mapping[str, tuple[str, str]]] = {
    "policy_np_semantic_draws": ("policy", "np"),
    "metrics_np_semantic_draws": ("metrics", "np"),
}


class CudaArchitectureBDirectResidentPatchV1Error(RuntimeError):
    """The requested fixed-Patch multicell owner is not structurally valid."""


def _rebase_tick_native_source_projection_v1(
    canonical_projection: Mapping[str, Any],
    prefix: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Overlay a tensor-free resident-owner prefix onto a source skeleton.

    A tick-native owner may execute many lifecycle intervals before a durable
    checkpoint is requested.  The original source projection remains the
    tensor schema authority, while this compact overlay advances only the
    source cursor and runtime prefix counters.  No tensor is copied or moved
    to host memory by this rebasing operation.
    """

    if not isinstance(canonical_projection, Mapping):
        raise _fail("tick-native source skeleton is not a mapping")
    if not isinstance(prefix, Mapping):
        raise _fail("tick-native source prefix is not a mapping")
    prefix_metadata = prefix.get("metadata")
    runtime = prefix.get("runtime")
    if not isinstance(prefix_metadata, Mapping) or not isinstance(runtime, Mapping):
        raise _fail("tick-native source prefix lacks metadata or runtime fields")
    metadata = canonical_projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("tick-native source skeleton lacks metadata")
    rebased = dict(canonical_projection)
    rebased_metadata = dict(metadata)
    for name, value in prefix_metadata.items():
        if hasattr(value, "device") or hasattr(value, "detach"):
            raise _fail(
                "tick-native source prefix metadata retained a tensor-bearing value"
            )
        rebased_metadata[str(name)] = copy.deepcopy(value)
    rebased["metadata"] = rebased_metadata
    for name in _TICK_NATIVE_RUNTIME_PREFIX_FIELDS_V1:
        if name not in runtime:
            continue
        value = runtime[name]
        if hasattr(value, "device") or hasattr(value, "detach"):
            raise _fail(
                "tick-native source prefix retained a tensor-bearing runtime field: "
                f"{name}"
            )
        rebased[name] = copy.deepcopy(value)
    return rebased


class DirectResidentPatchRequestFactoryV1(Protocol):
    """Public campaign-owned translation from one runner to one request."""

    def __call__(
        self,
        runner: Any,
        *,
        cell_index: int,
        cell_id: str,
        batch_cells: int,
        logical_generation_count: int,
        physical_window_generations: int,
        qualified_cuda_module: Any,
        cuda_device: str,
        resume_projection: Mapping[str, Any] | None,
    ) -> DirectResidentSourceAllocationRequestV1: ...


@dataclass(slots=True)
class DetachedCellProjectionV1:
    """One mutable cell payload owned by a checkpoint cursor.

    The projection is deliberately cleared by :meth:`release` so a caller
    cannot accidentally retain tensor-bearing state after the cursor advances.
    The cursor never exposes a batch-wide projection collection.
    """

    schema_version: str
    cell_id: str
    cell_index: int
    projection: Mapping[str, Any] | None
    compact_result_rows: tuple[dict[str, Any], ...] = ()
    segment_artifacts: tuple[Mapping[str, Any], ...] = ()
    lifecycle_qacct_receipt: Mapping[str, Any] | None = None
    terminal_generation: int = 0
    terminal_tick: int = 0
    live_projection_bytes: int = 0
    _released: bool = False

    def require_projection(self) -> Mapping[str, Any]:
        if self._released or not isinstance(self.projection, Mapping):
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "detached cell projection has already been released"
            )
        return self.projection

    def release(self) -> None:
        """Drop the tensor-bearing mapping and all per-cell payload aliases."""

        if self._released:
            return
        self.projection = None
        self.compact_result_rows = ()
        self.segment_artifacts = ()
        self.lifecycle_qacct_receipt = None
        self.live_projection_bytes = 0
        self._released = True


@dataclass(frozen=True, slots=True)
class CheckpointProjectionCursorReceiptV1:
    """Immutable proof emitted when a bounded projection cursor closes."""

    schema_version: str
    state: str
    coordinate: int
    ordered_cell_ids: tuple[str, ...]
    covered_cell_ids: tuple[str, ...]
    max_live_tensor_bearing_projections: int
    max_live_projection_bytes: int
    writer_concurrency: int
    failure_reason: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "coordinate": self.coordinate,
            "ordered_cell_ids": list(self.ordered_cell_ids),
            "covered_cell_ids": list(self.covered_cell_ids),
            "max_live_tensor_bearing_projections": (
                self.max_live_tensor_bearing_projections
            ),
            "max_live_projection_bytes": self.max_live_projection_bytes,
            "writer_concurrency": self.writer_concurrency,
            "failure_reason": self.failure_reason,
        }


class CheckpointProjectionCursorV1:
    """Strict one-cell-at-a-time projection cursor.

    This intentionally is not a ``Sequence`` and has no ``__iter__``,
    ``__len__``, or ``__getitem__`` implementation.  A second cell cannot be
    obtained until the caller explicitly releases the first one.
    """

    __slots__ = (
        "_owner",
        "_coordinate",
        "_cell_ids",
        "_factory",
        "_on_close",
        "_on_fail",
        "_next_index",
        "_current",
        "_covered",
        "_state",
        "_failure_reason",
        "_max_live_count",
        "_max_live_bytes",
        "_aggregate_compact_rows",
        "_aggregate_artifacts",
        "_aggregate_qacct",
        "_terminal_generations",
        "_terminal_ticks",
    )

    def __init__(
        self,
        *,
        owner: Any,
        coordinate: int,
        ordered_cell_ids: tuple[str, ...],
        factory: Callable[[int, str], DetachedCellProjectionV1],
        on_close: Callable[[], None] | None = None,
        on_fail: Callable[[str], None] | None = None,
    ) -> None:
        if type(coordinate) is not int or coordinate < 0:
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "checkpoint projection cursor coordinate must be nonnegative"
            )
        if not ordered_cell_ids or len(set(ordered_cell_ids)) != len(ordered_cell_ids):
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "checkpoint projection cursor cell IDs must be unique"
            )
        if not callable(factory):
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "checkpoint projection cursor factory must be callable"
            )
        self._owner = owner
        self._coordinate = coordinate
        self._cell_ids = ordered_cell_ids
        self._factory = factory
        self._on_close = on_close
        self._on_fail = on_fail
        self._next_index = 0
        self._current: DetachedCellProjectionV1 | None = None
        self._covered: list[str] = []
        self._state = "OPEN"
        self._failure_reason: str | None = None
        self._max_live_count = 0
        self._max_live_bytes = 0
        self._aggregate_compact_rows: list[dict[str, Any]] = []
        self._aggregate_artifacts: list[Mapping[str, Any]] = []
        self._aggregate_qacct: list[Mapping[str, Any]] = []
        self._terminal_generations: set[int] = set()
        self._terminal_ticks: set[int] = set()

    @classmethod
    def begin(
        cls,
        owner: Any,
        coordinate: int,
        ordered_cell_ids: tuple[str, ...],
        factory: Callable[[int, str], DetachedCellProjectionV1],
        on_fail: Callable[[str], None] | None = None,
    ) -> "CheckpointProjectionCursorV1":
        return cls(
            owner=owner,
            coordinate=coordinate,
            ordered_cell_ids=ordered_cell_ids,
            factory=factory,
            on_fail=on_fail,
        )

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self._cell_ids

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_cell_id(self) -> str | None:
        return self._current.cell_id if self._current is not None else None

    def next_cell(self) -> DetachedCellProjectionV1 | None:
        if self._state != "OPEN":
            raise CudaArchitectureBDirectResidentPatchV1Error(
                f"checkpoint projection cursor is not open: {self._state}"
            )
        if self._current is not None:
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "checkpoint projection cursor requires release_cell before next_cell"
            )
        if self._next_index == len(self._cell_ids):
            return None
        index = self._next_index
        cell_id = self._cell_ids[index]
        try:
            detached = self._factory(index, cell_id)
            if not isinstance(detached, DetachedCellProjectionV1):
                raise CudaArchitectureBDirectResidentPatchV1Error(
                    "checkpoint projection cursor factory returned the wrong type"
                )
            if (
                detached.cell_index != index
                or detached.cell_id != cell_id
                or detached.schema_version
                != DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1
            ):
                raise CudaArchitectureBDirectResidentPatchV1Error(
                    "checkpoint projection cursor factory returned an unordered cell"
                )
            detached.require_projection()
        except BaseException as exc:
            self.fail(str(exc))
            raise
        self._current = detached
        self._max_live_count = max(self._max_live_count, 1)
        self._max_live_bytes = max(
            self._max_live_bytes, int(detached.live_projection_bytes)
        )
        return detached

    def release_cell(self, cell_id: str) -> None:
        if self._state != "OPEN":
            raise CudaArchitectureBDirectResidentPatchV1Error(
                f"cannot release a cell on cursor state {self._state}"
            )
        current = self._current
        if current is None or current.cell_id != cell_id:
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "checkpoint projection cursor release does not match its live cell"
            )
        if current.terminal_generation < 0 or current.terminal_tick < 0:
            self.fail("detached cell terminal coordinates are invalid")
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "detached cell terminal coordinates are invalid"
            )
        self._aggregate_compact_rows.extend(
            dict(row) for row in current.compact_result_rows
        )
        self._aggregate_artifacts.extend(current.segment_artifacts)
        if current.lifecycle_qacct_receipt is not None:
            self._aggregate_qacct.append(dict(current.lifecycle_qacct_receipt))
        self._terminal_generations.add(int(current.terminal_generation))
        self._terminal_ticks.add(int(current.terminal_tick))
        current.release()
        self._current = None
        self._covered.append(cell_id)
        self._next_index += 1

    def fail(self, reason: str) -> None:
        if self._state in {"SEALED", "FAILED_UNPUBLISHED"}:
            return
        if self._current is not None:
            self._current.release()
            self._current = None
        self._state = "FAILED_UNPUBLISHED"
        self._failure_reason = str(reason)
        if self._on_fail is not None:
            callback, self._on_fail = self._on_fail, None
            callback(self._failure_reason)
        self._factory = lambda _index, _cell_id: (_ for _ in ()).throw(
            CudaArchitectureBDirectResidentPatchV1Error(
                "failed checkpoint projection cursor is unusable"
            )
        )
        self._owner = None

    def close(self) -> CheckpointProjectionCursorReceiptV1:
        if self._current is not None:
            self.fail("cursor closed while one cell payload remained live")
        if self._state == "OPEN":
            if self._next_index != len(self._cell_ids):
                self.fail("cursor closed before every ordered cell was released")
            else:
                self._state = "SEALED"
        if self._on_close is not None:
            callback, self._on_close = self._on_close, None
            callback()
        self._factory = lambda _index, _cell_id: (_ for _ in ()).throw(
            CudaArchitectureBDirectResidentPatchV1Error(
                "closed checkpoint projection cursor is unusable"
            )
        )
        self._owner = None
        return CheckpointProjectionCursorReceiptV1(
            schema_version=CHECKPOINT_PROJECTION_CURSOR_RECEIPT_SCHEMA_VERSION_V1,
            state=self._state,
            coordinate=self._coordinate,
            ordered_cell_ids=self._cell_ids,
            covered_cell_ids=tuple(self._covered),
            max_live_tensor_bearing_projections=self._max_live_count,
            max_live_projection_bytes=self._max_live_bytes,
            writer_concurrency=1,
            failure_reason=self._failure_reason,
        )

    def aggregate_payload(self) -> dict[str, Any]:
        if self._state != "SEALED":
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "cursor aggregate is available only after SEALED"
            )
        if len(self._terminal_generations) != 1 or len(self._terminal_ticks) != 1:
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "bounded checkpoint cells disagree on terminal coordinates"
            )
        return {
            "compact_result_rows": tuple(self._aggregate_compact_rows),
            "segment_artifacts": tuple(self._aggregate_artifacts),
            "lifecycle_qacct_receipts": tuple(self._aggregate_qacct),
            "terminal_generation": next(iter(self._terminal_generations)),
            "terminal_tick": next(iter(self._terminal_ticks)),
        }

    def map(
        self,
        transform: Callable[[DetachedCellProjectionV1], DetachedCellProjectionV1],
    ) -> "CheckpointProjectionCursorV1":
        """Map one cell before releasing the parent payload.

        The mapped cursor publishes only the transformed cell.  The parent
        payload is released before the mapped cursor exposes its result, so
        the externally observable live payload bound remains one cell.
        """

        parent = self

        def factory(index: int, cell_id: str) -> DetachedCellProjectionV1:
            source = parent.next_cell()
            if source is None:
                raise CudaArchitectureBDirectResidentPatchV1Error(
                    "parent checkpoint cursor ended before mapped cell"
                )
            try:
                mapped = transform(source)
                if not isinstance(mapped, DetachedCellProjectionV1):
                    raise CudaArchitectureBDirectResidentPatchV1Error(
                        "mapped checkpoint cursor transform returned the wrong type"
                    )
                return mapped
            finally:
                source.release()
                parent.release_cell(cell_id)

        def close_parent() -> None:
            if parent.state == "OPEN":
                parent.close()

        return CheckpointProjectionCursorV1(
            owner=self,
            coordinate=self._coordinate,
            ordered_cell_ids=self._cell_ids,
            factory=factory,
            on_close=close_parent,
        )


@dataclass(slots=True)
class DetachedSourcePayloadV1:
    """One verified source payload held by :class:`SourceImportCursorV1`."""

    schema_version: str
    cell_id: str
    cell_index: int
    payload: Mapping[str, Any] | None
    source_projection_sha256: str
    live_payload_bytes: int = 0
    _released: bool = False

    def require_payload(self) -> Mapping[str, Any]:
        if self._released or not isinstance(self.payload, Mapping):
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "detached source payload has already been released"
            )
        return self.payload

    def release(self) -> None:
        if self._released:
            return
        self.payload = None
        self.live_payload_bytes = 0
        self._released = True


@dataclass(frozen=True, slots=True)
class SourceImportCursorReceiptV1:
    """Immutable proof for one-cell-at-a-time source import."""

    schema_version: str
    state: str
    ordered_cell_ids: tuple[str, ...]
    covered_cell_ids: tuple[str, ...]
    max_live_tensor_bearing_payloads: int
    max_live_payload_bytes: int
    importer_concurrency: int
    failure_reason: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "ordered_cell_ids": list(self.ordered_cell_ids),
            "covered_cell_ids": list(self.covered_cell_ids),
            "max_live_tensor_bearing_payloads": (
                self.max_live_tensor_bearing_payloads
            ),
            "max_live_payload_bytes": self.max_live_payload_bytes,
            "importer_concurrency": self.importer_concurrency,
            "failure_reason": self.failure_reason,
        }


class SourceImportCursorV1:
    """Verified source descriptors with a strict one-payload live bound.

    Descriptors are retained, but decoded checkpoint payloads are never cached.
    The importer must consume the current payload into its prepared CUDA
    storage before ``import_and_release`` advances the cursor.
    """

    __slots__ = (
        "_descriptors",
        "_cell_ids",
        "_loader",
        "_importer",
        "_next_index",
        "_current",
        "_covered",
        "_state",
        "_failure_reason",
        "_max_live_count",
        "_max_live_bytes",
    )

    def __init__(
        self,
        *,
        checkpoint_descriptors: tuple[Any, ...],
        ordered_cell_ids: tuple[str, ...],
        loader: Callable[[Any, int, str], DetachedSourcePayloadV1],
        importer: Callable[[Mapping[str, Any], Any, int, str], Any],
    ) -> None:
        if not checkpoint_descriptors or len(checkpoint_descriptors) != len(
            ordered_cell_ids
        ):
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "source import cursor descriptors do not cover the cell order"
            )
        if not ordered_cell_ids or len(set(ordered_cell_ids)) != len(ordered_cell_ids):
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "source import cursor cell IDs must be unique"
            )
        if not callable(loader) or not callable(importer):
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "source import cursor loader and importer must be callable"
            )
        self._descriptors = checkpoint_descriptors
        self._cell_ids = ordered_cell_ids
        self._loader = loader
        self._importer = importer
        self._next_index = 0
        self._current: DetachedSourcePayloadV1 | None = None
        self._covered: list[str] = []
        self._state = "OPEN"
        self._failure_reason: str | None = None
        self._max_live_count = 0
        self._max_live_bytes = 0

    @classmethod
    def begin(
        cls,
        checkpoint_descriptors: Sequence[Any],
        ordered_cell_ids: Sequence[str],
        *,
        loader: Callable[[Any, int, str], DetachedSourcePayloadV1],
        importer: Callable[[Mapping[str, Any], Any, int, str], Any],
    ) -> "SourceImportCursorV1":
        return cls(
            checkpoint_descriptors=tuple(checkpoint_descriptors),
            ordered_cell_ids=tuple(ordered_cell_ids),
            loader=loader,
            importer=importer,
        )

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self._cell_ids

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_cell_id(self) -> str | None:
        return self._current.cell_id if self._current is not None else None

    def next_cell(self) -> DetachedSourcePayloadV1 | None:
        if self._state != "OPEN":
            raise CudaArchitectureBDirectResidentPatchV1Error(
                f"source import cursor is not open: {self._state}"
            )
        if self._current is not None:
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "source import cursor requires import_and_release before next_cell"
            )
        if self._next_index == len(self._cell_ids):
            return None
        index = self._next_index
        cell_id = self._cell_ids[index]
        try:
            detached = self._loader(self._descriptors[index], index, cell_id)
            if not isinstance(detached, DetachedSourcePayloadV1):
                raise CudaArchitectureBDirectResidentPatchV1Error(
                    "source import cursor loader returned the wrong type"
                )
            if (
                detached.cell_index != index
                or detached.cell_id != cell_id
                or detached.schema_version != SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1
                or len(detached.source_projection_sha256) != 64
            ):
                raise CudaArchitectureBDirectResidentPatchV1Error(
                    "source import cursor loader returned an unordered payload"
                )
            detached.require_payload()
        except BaseException as exc:
            self.fail(str(exc))
            raise
        self._current = detached
        self._max_live_count = max(self._max_live_count, 1)
        self._max_live_bytes = max(
            self._max_live_bytes, int(detached.live_payload_bytes)
        )
        return detached

    def import_and_release(self, cell_id: str, prepared_cuda_storage: Any) -> Any:
        if self._state != "OPEN":
            raise CudaArchitectureBDirectResidentPatchV1Error(
                f"cannot import on source cursor state {self._state}"
            )
        current = self._current
        if current is None or current.cell_id != cell_id:
            raise CudaArchitectureBDirectResidentPatchV1Error(
                "source import cursor import does not match its live cell"
            )
        try:
            result = self._importer(
                current.require_payload(), prepared_cuda_storage, current.cell_index, cell_id
            )
        except BaseException as exc:
            self.fail(str(exc))
            raise
        current.release()
        self._current = None
        self._covered.append(cell_id)
        self._next_index += 1
        return result

    def fail(self, reason: str) -> None:
        if self._state in {"SEALED", "FAILED_UNPUBLISHED"}:
            return
        if self._current is not None:
            self._current.release()
            self._current = None
        self._state = "FAILED_UNPUBLISHED"
        self._failure_reason = str(reason)
        self._loader = lambda _descriptor, _index, _cell_id: (_ for _ in ()).throw(
            CudaArchitectureBDirectResidentPatchV1Error(
                "failed source import cursor is unusable"
            )
        )
        self._importer = lambda _payload, _storage, _index, _cell_id: (_ for _ in ()).throw(
            CudaArchitectureBDirectResidentPatchV1Error(
                "failed source import cursor is unusable"
            )
        )

    def close(self) -> SourceImportCursorReceiptV1:
        if self._current is not None:
            self.fail("source import cursor closed while one payload remained live")
        if self._state == "OPEN":
            if self._next_index != len(self._cell_ids):
                self.fail("source import cursor closed before every cell was imported")
            else:
                self._state = "SEALED"
        return SourceImportCursorReceiptV1(
            schema_version=SOURCE_IMPORT_CURSOR_RECEIPT_SCHEMA_VERSION_V1,
            state=self._state,
            ordered_cell_ids=self._cell_ids,
            covered_cell_ids=tuple(self._covered),
            max_live_tensor_bearing_payloads=self._max_live_count,
            max_live_payload_bytes=self._max_live_bytes,
            importer_concurrency=1,
            failure_reason=self._failure_reason,
        )


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentPatchMulticellV1:
    """One dispatch-ready owner plus its ordered canonical source boundaries."""

    schema_version: str
    patch: PreparedDirectResidentTrajectoryV1
    cell_ids: tuple[str, ...]
    source_projections: tuple[Any, ...]
    resumed: bool
    source_projection_cursor_factory: Callable[[], Any] | None = None

    @property
    def batch_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def physical_window_arguments(self) -> tuple[Any, ...]:
        return self.patch.physical_window_arguments


@dataclass(frozen=True, slots=True)
class DirectResidentPatchMulticellSettlementV1:
    """Canonical per-cell checkpoints and flattened compact reporting rows."""

    schema_version: str
    cell_ids: tuple[str, ...]
    checkpoint_projections: tuple[Mapping[str, Any], ...]
    compact_result_rows: tuple[dict[str, Any], ...]
    completed_generation_count: int
    terminal_generation: int
    terminal_tick: int = 0
    segment_artifacts: tuple[Mapping[str, Any], ...] = ()
    analysis_source_rows: tuple[Mapping[str, Any], ...] = ()
    lifecycle_qacct_receipts: tuple[Mapping[str, Any], ...] = ()
    settlement_event_rows: tuple[Mapping[str, Any], ...] = ()
    checkpoint_projection_cursor: CheckpointProjectionCursorV1 | None = None


def _fail(message: str) -> CudaArchitectureBDirectResidentPatchV1Error:
    return CudaArchitectureBDirectResidentPatchV1Error(message)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_SETTLEMENT_EVENT_TYPE_NAMES_V1: Final[Mapping[int, str]] = {
    1: "birth_committed",
    2: "mutation",
    3: "birth_failed",
    4: "death_committed",
}


def _exact_generation_summary_integer_v1(
    row: torch.Tensor,
    *,
    index: Mapping[str, int],
    name: str,
) -> int:
    value = float(row[index[name]].item())
    if not math.isfinite(value):
        raise _fail(f"M0 generation summary {name} is not a finite exact integer")
    integer = int(value)
    if value != float(integer):
        raise _fail(f"M0 generation summary {name} is not a finite exact integer")
    return integer


def _event_type_counts_v1(values: torch.Tensor) -> dict[str, int]:
    return {
        name: int(torch.count_nonzero(values == event_type).item())
        for event_type, name in _SETTLEMENT_EVENT_TYPE_NAMES_V1.items()
    }


def _validate_settlement_event_semantics_v1(
    *,
    columns: Mapping[str, torch.Tensor],
    cell_id: str,
    reported_rows: int,
) -> None:
    active = slice(0, reported_rows)
    compatibility_pairs = (
        ("typed_lifecycle_event_event_type", "event_tape_type"),
        ("typed_lifecycle_event_generation", "event_tape_generation"),
        ("typed_lifecycle_event_tick", "event_tape_tick"),
        ("typed_lifecycle_event_actor_agent_id", "event_tape_actor_id"),
        ("typed_lifecycle_event_target_agent_id", "event_tape_target_id"),
        ("typed_lifecycle_event_value", "event_tape_value"),
    )
    if any(
        not torch.equal(columns[typed][active], columns[legacy][active])
        for typed, legacy in compatibility_pairs
    ):
        raise _fail(
            f"M0 settlement legacy and typed event columns differ for {cell_id}"
        )
    if reported_rows == 0:
        return

    event_type = columns["typed_lifecycle_event_event_type"][active]
    tick = columns["typed_lifecycle_event_tick"][active]
    actor = columns["typed_lifecycle_event_actor_agent_id"][active]
    target = columns["typed_lifecycle_event_target_agent_id"][active]
    actor_slot = columns["typed_lifecycle_event_actor_slot"][active]
    target_slot = columns["typed_lifecycle_event_target_slot"][active]
    parent = columns["typed_lifecycle_event_parent_agent_id"][active]
    child = columns["typed_lifecycle_event_child_agent_id"][active]
    founder = columns["typed_lifecycle_event_lineage_founder_id"][active]
    depth = columns["typed_lifecycle_event_lineage_depth"][active]
    subtype = columns["typed_lifecycle_event_subtype_code"][active]
    reason = columns["typed_lifecycle_event_reason_code"][active]
    success = columns["typed_lifecycle_event_success"][active]
    value = columns["typed_lifecycle_event_value"][active]

    birth = event_type == 1
    mutation = event_type == 2
    failed = event_type == 3
    death = event_type == 4
    successful = birth | mutation | death
    semantic_valid = (
        torch.all(tick >= 0)
        and torch.all(subtype == event_type)
        and torch.all(founder >= 0)
        and torch.all(depth >= 0)
        and torch.all(success[successful] == 1)
        and torch.all(success[failed] == 0)
        and torch.all(reason[birth | mutation] == -1)
        and torch.all((reason[failed] >= 1) & (reason[failed] <= 3))
        and torch.all((reason[death] >= 1) & (reason[death] <= 2))
        and torch.all(torch.isfinite(value))
        and torch.all(value[failed | death] == reason[failed | death].to(value.dtype))
        and torch.all(actor[birth | mutation | failed | death] >= 0)
        and torch.all(actor_slot[birth | mutation | failed | death] >= 0)
        and torch.all(target[birth] >= 0)
        and torch.all(target_slot[birth] >= 0)
        and torch.all(parent[birth] == actor[birth])
        and torch.all(child[birth] == target[birth])
        and torch.all(target[mutation] == actor[mutation])
        and torch.all(target_slot[mutation] == actor_slot[mutation])
        and torch.all(child[mutation] == actor[mutation])
        and torch.all(parent[mutation] >= 0)
        and torch.all(target[failed | death] == -1)
        and torch.all(target_slot[failed | death] == -1)
        and torch.all(parent[failed | death] == -1)
        and torch.all(child[failed | death] == -1)
    )
    if not bool(semantic_valid):
        raise _fail(f"M0 settlement typed event semantics are invalid for {cell_id}")


def _settlement_event_rows_from_sink_v1(
    *,
    direct_result: Mapping[str, Any],
    cell_ids: tuple[str, ...],
    generation_base: int,
    completed_generations: int,
) -> tuple[Mapping[str, Any], ...]:
    """Validate M0 lifecycle rows from bounded sink slices.

    The normal simulation path never reloads emitted history.  This optional
    reporting path is deliberately a streaming reader: it validates one
    bounded sink slice at a time and retains only per-generation counters.
    """

    summary = direct_result.get("generation_summary_scalar")
    event_rows_by_cell = direct_result.get("event_rows_by_cell")
    reference = direct_result.get("event_history_reference")
    cells = len(cell_ids)
    if (
        not isinstance(summary, torch.Tensor)
        or summary.device.type != "cpu"
        or summary.dtype != torch.float64
        or not summary.is_contiguous()
        or tuple(summary.shape)
        != (completed_generations, cells, len(GENERATION_SUMMARY_COLUMNS))
        or not isinstance(event_rows_by_cell, Sequence)
        or isinstance(event_rows_by_cell, (str, bytes, bytearray))
        or len(event_rows_by_cell) != cells
        or not isinstance(reference, Mapping)
        or reference.get("schema_version") != EVENT_SINK_SCHEMA_VERSION
        or not isinstance(reference.get("path"), str)
        or reference.get("row_bytes") != EVENT_SINK_ROW_BYTES
    ):
        raise _fail("M0 external event sink summary or reference is invalid")
    row_counts = reference.get("row_count_by_cell")
    if (
        not isinstance(row_counts, Sequence)
        or isinstance(row_counts, (str, bytes, bytearray))
        or len(row_counts) != cells
    ):
        raise _fail("M0 external event sink reference row counts are invalid")
    frontier_by_cell = direct_result.get("event_history_frontier_by_cell")
    if frontier_by_cell is not None and (
        not isinstance(frontier_by_cell, Sequence)
        or isinstance(frontier_by_cell, (str, bytes, bytearray))
        or len(frontier_by_cell) != cells
    ):
        raise _fail("M0 external event sink frontier vector is invalid")

    summary_index = {
        name: offset for offset, name in enumerate(GENERATION_SUMMARY_COLUMNS)
    }
    event_type_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_event_type"
    )
    event_generation_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_generation"
    )
    event_sequence_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_sequence"
    )
    rows: list[Mapping[str, Any]] = []
    for cell, cell_id in enumerate(cell_ids):
        reported_rows = event_rows_by_cell[cell]
        if (
            type(reported_rows) is not int
            or reported_rows < 0
            or type(row_counts[cell]) is not int
            or row_counts[cell] != reported_rows
            or (
                frontier_by_cell is not None
                and frontier_by_cell[cell] != reported_rows
            )
        ):
            raise _fail(f"M0 external event sink row count is invalid for {cell_id}")

        ranges: list[tuple[int, int, int, int]] = []
        previous_end: int | None = None
        for generation_offset in range(completed_generations):
            summary_row = summary[generation_offset, cell]
            generation = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="generation",
            )
            cursor_start = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="event_cursor_start",
            )
            cursor_pre_turnover = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="event_cursor_pre_turnover",
            )
            cursor_end = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="event_cursor_end",
            )
            if (
                generation != generation_base + generation_offset
                or cursor_start < 0
                or not cursor_start <= cursor_pre_turnover <= cursor_end
                or cursor_end > reported_rows
                or (previous_end is not None and cursor_start != previous_end)
            ):
                raise _fail(
                    f"M0 external event sink cursor lattice is invalid for {cell_id}"
                )
            ranges.append((cursor_start, cursor_pre_turnover, cursor_end, generation))
            previous_end = cursor_end
        if previous_end != reported_rows:
            raise _fail(
                f"M0 external event sink final cursor differs for {cell_id}"
            )

        ordinary_counts = [
            {name: 0 for name in _SETTLEMENT_EVENT_TYPE_NAMES_V1.values()}
            for _ in ranges
        ]
        turnover_counts = [
            {name: 0 for name in _SETTLEMENT_EVENT_TYPE_NAMES_V1.values()}
            for _ in ranges
        ]
        scan_start = ranges[0][0] if ranges else reported_rows
        seen = scan_start
        range_index = 0
        try:
            chunks = iter_event_sink_columns_v1(
                reference["path"],
                cell_index=cell,
                expected_rows=reported_rows,
                start_row=scan_start,
                end_row=reported_rows,
            )
            for chunk_start, columns in chunks:
                chunk_rows = int(columns[0].numel())
                if chunk_start != seen:
                    raise _fail(
                        f"M0 external event sink rows are not ordered for {cell_id}"
                    )
                sequence = columns[event_sequence_index]
                if not torch.equal(
                    sequence,
                    torch.arange(seen, seen + chunk_rows, dtype=torch.int64),
                ):
                    raise _fail(
                        f"M0 external event sink sequence is invalid for {cell_id}"
                    )
                event_types = columns[event_type_index]
                event_generations = columns[event_generation_index]
                if chunk_rows and not bool(
                    torch.all((event_types >= 1) & (event_types <= 4)).item()
                ):
                    raise _fail(
                        f"M0 external event sink event type is invalid for {cell_id}"
                    )
                _validate_settlement_event_semantics_v1(
                    columns=dict(zip(EVENT_COLUMN_NAMES, columns, strict=True)),
                    cell_id=cell_id,
                    reported_rows=chunk_rows,
                )
                position = chunk_start
                chunk_end = chunk_start + chunk_rows
                while position < chunk_end:
                    while (
                        range_index < len(ranges)
                        and position >= ranges[range_index][2]
                    ):
                        range_index += 1
                    if range_index >= len(ranges):
                        raise _fail(
                            "M0 external event sink has rows beyond the summary "
                            f"for {cell_id}"
                        )
                    range_start, range_pre, range_end, generation = ranges[
                        range_index
                    ]
                    if position < range_start or position >= range_end:
                        raise _fail(
                            "M0 external event sink row is outside its generation "
                            f"for {cell_id}"
                        )
                    take_end = min(chunk_end, range_end)
                    local_start = position - chunk_start
                    local_end = take_end - chunk_start
                    generation_values = event_generations[local_start:local_end]
                    if generation_values.numel() and not bool(
                        torch.all(generation_values == generation).item()
                    ):
                        raise _fail(
                            "M0 external event sink generation differs for "
                            f"{cell_id}"
                        )
                    if position < range_pre:
                        ordinary_end = min(take_end, range_pre)
                        ordinary_slice = event_types[
                            local_start : ordinary_end - chunk_start
                        ]
                        for name, value in _event_type_counts_v1(
                            ordinary_slice
                        ).items():
                            ordinary_counts[range_index][name] += value
                        if ordinary_end < take_end:
                            turnover_slice = event_types[
                                ordinary_end - chunk_start : local_end
                            ]
                            for name, value in _event_type_counts_v1(
                                turnover_slice
                            ).items():
                                turnover_counts[range_index][name] += value
                    else:
                        for name, value in _event_type_counts_v1(
                            event_types[local_start:local_end]
                        ).items():
                            turnover_counts[range_index][name] += value
                    position = take_end
                seen = chunk_end
        except EventSinkError as exc:
            raise _fail(f"M0 external event sink cannot be validated: {exc}") from exc
        if seen != reported_rows:
            raise _fail(f"M0 external event sink rows are incomplete for {cell_id}")

        for generation_offset, (
            cursor_start,
            cursor_pre_turnover,
            cursor_end,
            generation,
        ) in enumerate(ranges):
            summary_row = summary[generation_offset, cell]
            summary_counts = {
                "birth_committed": _exact_generation_summary_integer_v1(
                    summary_row, index=summary_index, name="births"
                ),
                "death_committed": _exact_generation_summary_integer_v1(
                    summary_row, index=summary_index, name="deaths"
                ),
                "birth_failed": _exact_generation_summary_integer_v1(
                    summary_row, index=summary_index, name="failed_births"
                ),
            }
            if (
                any(value < 0 for value in summary_counts.values())
                or any(
                    ordinary_counts[generation_offset][name] != expected
                    for name, expected in summary_counts.items()
                )
                or turnover_counts[generation_offset]["birth_failed"] != 0
                or turnover_counts[generation_offset]["death_committed"] != 0
            ):
                raise _fail(
                    f"M0 external event sink partition differs for {cell_id}"
                )
            rows.append(
                {
                    "schema_version": "eiel_direct_resident_settlement_event_row_v1",
                    "cell_id": cell_id,
                    "generation": generation,
                    "events": {
                        "cursor_start": cursor_start,
                        "cursor_pre_turnover": cursor_pre_turnover,
                        "cursor_end": cursor_end,
                        "authoritative_cursor": cursor_end,
                        "event_row_count": cursor_end - cursor_start,
                        "generation_event_count": cursor_pre_turnover - cursor_start,
                        "turnover_event_count": cursor_end - cursor_pre_turnover,
                        "generation_counts_by_type": ordinary_counts[generation_offset],
                        "turnover_counts_by_type": turnover_counts[generation_offset],
                    },
                }
            )
    return tuple(rows)


def _settlement_event_rows_v1(
    *,
    direct_result: Mapping[str, Any],
    cell_ids: tuple[str, ...],
    generation_base: int,
    completed_generations: int,
) -> tuple[Mapping[str, Any], ...]:
    """Project exact lifecycle counts without reusing reporting rows."""

    if direct_result.get("event_history_storage_mode") == EVENT_SINK_SCHEMA_VERSION:
        return _settlement_event_rows_from_sink_v1(
            direct_result=direct_result,
            cell_ids=cell_ids,
            generation_base=generation_base,
            completed_generations=completed_generations,
        )

    summary = direct_result.get("generation_summary_scalar")
    histories = direct_result.get("event_histories")
    event_rows_by_cell = direct_result.get("event_rows_by_cell")
    cells = len(cell_ids)
    if (
        not isinstance(summary, torch.Tensor)
        or summary.device.type != "cpu"
        or summary.dtype != torch.float64
        or not summary.is_contiguous()
        or tuple(summary.shape)
        != (completed_generations, cells, len(GENERATION_SUMMARY_COLUMNS))
        or not isinstance(histories, Sequence)
        or isinstance(histories, (str, bytes, bytearray))
        or len(histories) != cells
        or not isinstance(event_rows_by_cell, Sequence)
        or isinstance(event_rows_by_cell, (str, bytes, bytearray))
        or len(event_rows_by_cell) != cells
    ):
        raise _fail("M0 settlement summary or event-history dimensions are invalid")
    summary_index = {
        name: offset for offset, name in enumerate(GENERATION_SUMMARY_COLUMNS)
    }
    event_type_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_event_type"
    )
    event_generation_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_generation"
    )
    event_sequence_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_sequence"
    )
    rows: list[Mapping[str, Any]] = []
    for cell, cell_id in enumerate(cell_ids):
        history = histories[cell]
        reported_rows = event_rows_by_cell[cell]
        if (
            not isinstance(history, Sequence)
            or isinstance(history, (str, bytes, bytearray))
            or len(history) != len(EVENT_COLUMN_NAMES)
            or type(reported_rows) is not int
            or reported_rows < 0
        ):
            raise _fail(f"M0 settlement event history is invalid for {cell_id}")
        columns: list[torch.Tensor] = []
        for name, value in zip(EVENT_COLUMN_NAMES, history, strict=True):
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != EVENT_COLUMN_DTYPES[name]
                or value.ndim != 1
                or not value.is_contiguous()
                or value.numel() != reported_rows
            ):
                raise _fail(
                    f"M0 settlement event column is invalid: {cell_id}.{name}"
                )
            columns.append(value)
        columns_by_name = dict(zip(EVENT_COLUMN_NAMES, columns, strict=True))
        sequence = columns[event_sequence_index]
        if not torch.equal(
            sequence,
            torch.arange(reported_rows, dtype=torch.int64),
        ):
            raise _fail(f"M0 settlement event sequence is invalid for {cell_id}")
        event_types = columns[event_type_index]
        if reported_rows and not bool(
            torch.all((event_types >= 1) & (event_types <= 4)).item()
        ):
            raise _fail(f"M0 settlement event type is invalid for {cell_id}")
        event_generations = columns[event_generation_index]
        _validate_settlement_event_semantics_v1(
            columns=columns_by_name,
            cell_id=cell_id,
            reported_rows=reported_rows,
        )
        previous_end: int | None = None
        for generation_offset in range(completed_generations):
            summary_row = summary[generation_offset, cell]
            generation = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="generation",
            )
            cursor_start = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="event_cursor_start",
            )
            cursor_pre_turnover = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="event_cursor_pre_turnover",
            )
            cursor_end = _exact_generation_summary_integer_v1(
                summary_row,
                index=summary_index,
                name="event_cursor_end",
            )
            if (
                generation != generation_base + generation_offset
                or cursor_start < 0
                or not cursor_start <= cursor_pre_turnover <= cursor_end
                or cursor_end > reported_rows
                or (
                    previous_end is not None
                    and cursor_start != previous_end
                )
            ):
                raise _fail(
                    f"M0 settlement generation cursor lattice is invalid for {cell_id}"
                )
            generation_values = event_generations[cursor_start:cursor_end]
            if generation_values.numel() and not bool(
                torch.all(generation_values == generation).item()
            ):
                raise _fail(
                    f"M0 settlement event generation differs for {cell_id}"
                )
            ordinary_counts = _event_type_counts_v1(
                event_types[cursor_start:cursor_pre_turnover]
            )
            turnover_counts = _event_type_counts_v1(
                event_types[cursor_pre_turnover:cursor_end]
            )
            summary_counts = {
                "birth_committed": _exact_generation_summary_integer_v1(
                    summary_row,
                    index=summary_index,
                    name="births",
                ),
                "death_committed": _exact_generation_summary_integer_v1(
                    summary_row,
                    index=summary_index,
                    name="deaths",
                ),
                "birth_failed": _exact_generation_summary_integer_v1(
                    summary_row,
                    index=summary_index,
                    name="failed_births",
                ),
            }
            if (
                any(value < 0 for value in summary_counts.values())
                or any(
                    ordinary_counts[name] != expected
                    for name, expected in summary_counts.items()
                )
                or turnover_counts["birth_failed"] != 0
                or turnover_counts["death_committed"] != 0
            ):
                raise _fail(
                    f"M0 settlement event partition differs for {cell_id}"
                )
            rows.append(
                {
                    "schema_version": (
                        "eiel_direct_resident_settlement_event_row_v1"
                    ),
                    "cell_id": cell_id,
                    "generation": generation,
                    "events": {
                        "cursor_start": cursor_start,
                        "cursor_pre_turnover": cursor_pre_turnover,
                        "cursor_end": cursor_end,
                        "authoritative_cursor": cursor_end,
                        "event_row_count": cursor_end - cursor_start,
                        "generation_event_count": (
                            cursor_pre_turnover - cursor_start
                        ),
                        "turnover_event_count": cursor_end - cursor_pre_turnover,
                        "generation_counts_by_type": ordinary_counts,
                        "turnover_counts_by_type": turnover_counts,
                    },
                }
            )
            previous_end = cursor_end
        if previous_end != reported_rows:
            raise _fail(
                f"M0 settlement final event cursor differs for {cell_id}"
            )
    return tuple(rows)


def _segment_index_chain_and_counts_v1(
    projection: Mapping[str, Any],
) -> tuple[int, str, dict[str, int]]:
    metadata = projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("source projection metadata is malformed")
    segment_index = metadata.get("native_next_segment_index", 0)
    previous_chain = metadata.get("native_chain_sha256", "0" * 64)
    if type(segment_index) is not int or segment_index < 0:
        raise _fail("native segment index is invalid")
    if (
        not isinstance(previous_chain, str)
        or len(previous_chain) != 64
        or any(ch not in "0123456789abcdef" for ch in previous_chain)
    ):
        raise _fail("native segment chain is invalid")
    counts = {
        "attempts": metadata.get("native_cumulative_attempt_count", 0),
        "successes": metadata.get("native_cumulative_success_count", 0),
        "failures": metadata.get("native_cumulative_failure_count", 0),
        "eligible_snapshots": metadata.get("native_cumulative_snapshot_count", 0),
    }
    if (
        any(type(value) is not int or value < 0 for value in counts.values())
        or counts["attempts"] != counts["successes"] + counts["failures"]
    ):
        raise _fail("native cumulative segment counts are invalid")
    return segment_index, previous_chain, counts


def _projection_lifecycle_configuration_overrides_v1(
    projection: Any,
) -> dict[str, bool | int]:
    if isinstance(projection, Mapping):
        metadata = projection.get("metadata")
    else:
        checkpoint_projection = getattr(projection, "projection", None)
        if isinstance(checkpoint_projection, Mapping):
            metadata = checkpoint_projection.get("metadata")
        else:
            metadata = getattr(projection, "metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    turnover = metadata.get("generation_turnover_enabled")
    lifecycle_mode = metadata.get("lifecycle_mode_abi")
    if type(turnover) is not bool or isinstance(lifecycle_mode, bool):
        return {}
    if lifecycle_mode not in {0, 1} or lifecycle_mode != (0 if turnover else 1):
        return {}
    return {
        "generation_turnover_enabled": turnover,
        "lifecycle_mode": int(lifecycle_mode),
    }


def _configuration_with_overrides_v1(
    configuration: Any,
    overrides: Mapping[str, Any],
) -> Any:
    if not overrides:
        return configuration
    payload: dict[str, Any] = {}
    if isinstance(configuration, Mapping):
        payload.update(configuration)
    elif configuration is not None:
        for name in dir(configuration):
            if name.startswith("_"):
                continue
            try:
                value = getattr(configuration, name)
            except Exception:
                continue
            if type(value) in {bool, int, float, str, tuple}:
                payload[name] = value
    payload.update(overrides)
    return payload


def _provider_native_segment_artifact_v1(
    *,
    cell_id: str,
    cell_index: int,
    source_projection: Mapping[str, Any],
    compact_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    segment_index, previous_chain, prior_counts = (
        _segment_index_chain_and_counts_v1(source_projection)
    )
    attempts: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    for row in compact_rows:
        births = int(row.get("births", 0) or 0)
        failed_births = int(row.get("failed_births", 0) or 0)
        alive = max(0, int(row.get("alive", 0) or 0))
        if births > 0:
            success_count += 1
            attempts.append(
                {
                    "attempt_index": len(attempts),
                    "path": PATH_GENERATION_BOUNDARY_REFILL,
                    "generation": int(row["generation"]),
                    "tick": 0,
                    "eligible_parent_counts": [alive, alive],
                    "selected_parent_ids": [-1, -1],
                    "success": True,
                    "failure_code": 0,
                    "commit_ordinal": int(row["generation"]),
                    "authoritative_commit": None,
                    "compact_generation_summary": dict(row),
                }
            )
        if failed_births > 0 or births == 0:
            failure_count += 1
            attempts.append(
                {
                    "attempt_index": len(attempts),
                    "path": PATH_GENERATION_BOUNDARY_REFILL,
                    "generation": int(row["generation"]),
                    "tick": 0,
                    "eligible_parent_counts": [alive, alive],
                    "selected_parent_ids": [-1, -1],
                    "success": False,
                    "failure_code": 1,
                    "commit_ordinal": None,
                    "authoritative_commit": None,
                    "compact_generation_summary": dict(row),
                }
            )
    if not attempts:
        raise _fail("provider-native Chronicle segment has no compact rows")
    payload = {
        "cell_index": cell_index,
        "enabled": True,
        "status": 0,
        "attempt_count": len(attempts),
        "success_count": success_count,
        "failure_count": failure_count,
        "snapshot_count": 0,
        "attempts": attempts,
        "eligible_parent_snapshots": [],
        "selected_parent_counts": [],
        "parent_contribution_counts": [],
        "primary_eligibility_opportunity_counts": [],
        "mate_eligibility_opportunity_counts": [],
        "path_instrumentation": [],
        "cumulative_counts_after_segment": {
            "attempts": prior_counts["attempts"] + len(attempts),
            "successes": prior_counts["successes"] + success_count,
            "failures": prior_counts["failures"] + failure_count,
            "eligible_snapshots": prior_counts["eligible_snapshots"],
        },
    }
    segment_sha256 = _canonical_sha256(
        {
            "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
            "segment_index": segment_index,
            "cell": payload,
        }
    )
    chain_sha256 = _canonical_sha256(
        {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "segment_index": segment_index,
            "previous_chain_sha256": previous_chain,
            "segment_sha256": segment_sha256,
        }
    )
    cell_payload = {
        **payload,
        "segment_sha256": segment_sha256,
        "previous_chain_sha256": previous_chain,
        "chain_sha256_after_segment": chain_sha256,
    }
    segment_core = {
        "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
        "segment_index": segment_index,
        "capacity_semantics": {
            "storage_mode": STORAGE_MODE_COMPACT_COMPLETE,
            "scope": "per_execution_segment_compact_generation_summary_v1",
        },
        "cell_id": cell_id,
        "cell": cell_payload,
    }
    artifact = {
        **segment_core,
        "segment_output_sha256": _canonical_sha256(segment_core),
    }
    updated_projection = copy.deepcopy(dict(source_projection))
    metadata = dict(updated_projection.get("metadata") or {})
    metadata["native_next_segment_index"] = segment_index + 1
    metadata["native_chain_sha256"] = chain_sha256
    metadata["native_cumulative_attempt_count"] = payload[
        "cumulative_counts_after_segment"
    ]["attempts"]
    metadata["native_cumulative_success_count"] = payload[
        "cumulative_counts_after_segment"
    ]["successes"]
    metadata["native_cumulative_failure_count"] = payload[
        "cumulative_counts_after_segment"
    ]["failures"]
    metadata["native_cumulative_snapshot_count"] = payload[
        "cumulative_counts_after_segment"
    ]["eligible_snapshots"]
    updated_projection["metadata"] = metadata
    return artifact, updated_projection


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _fail(f"{name} must be a sequence")
    return tuple(value)


def _validated_cell_ids(value: Sequence[str], *, count: int) -> tuple[str, ...]:
    cell_ids = _sequence(value, name="cell_ids")
    if len(cell_ids) != count:
        raise _fail("cell_ids count must exactly match the resident batch")
    if any(
        type(cell_id) is not str
        or not cell_id
        or cell_id.strip() != cell_id
        for cell_id in cell_ids
    ):
        raise _fail("cell_ids must contain nonempty stable strings")
    if len(set(cell_ids)) != count:
        raise _fail("cell_ids must be unique")
    return cell_ids


def _projection_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        projection = value
    else:
        try:
            projection = dict(vars(value))
        except (TypeError, AttributeError) as exc:
            raise _fail(f"{name} must be a canonical checkpoint projection") from exc
    try:
        validate_native_checkpoint_projection(projection)
    except Exception as exc:
        raise _fail(f"{name} is not a valid canonical checkpoint: {exc}") from exc
    return projection


def _validated_source_projections(
    projections: Sequence[Any],
    *,
    batch_cells: int,
    cell_ids: tuple[str, ...] | None = None,
    allow_identity_reorder: bool = False,
) -> tuple[Any, ...]:
    values = _sequence(projections, name="source_projections")
    if len(values) != batch_cells:
        raise _fail("source projection count must exactly match batch_cells")
    canonical_views = tuple(
        _projection_mapping(value, name=f"source_projections[{index}]")
        for index, value in enumerate(values)
    )
    if cell_ids is not None:
        identified: list[str | None] = []
        for projection in canonical_views:
            identities = []
            for field in (
                "diploid_checkpoint_projection",
                "genome_upgrade_v2_checkpoint_projection",
            ):
                sidecar = projection.get(field)
                if sidecar is None:
                    continue
                identity = (
                    sidecar.get("identity")
                    if isinstance(sidecar, Mapping)
                    else None
                )
                identities.append(
                    identity.get("cell_id")
                    if isinstance(identity, Mapping)
                    else None
                )
            if identities and (
                any(type(value) is not str or not value for value in identities)
                or len(set(identities)) != 1
            ):
                raise _fail(
                    "resume inheritance sidecars have missing or mixed cell identities"
                )
            identified.append(identities[0] if identities else None)
        if any(value is not None for value in identified):
            if any(value is None for value in identified):
                raise _fail("resume sidecar cell identity set is partial")
            observed = tuple(identified)
            if len(set(observed)) != batch_cells or set(observed) != set(cell_ids):
                raise _fail("resume sidecar cell identities do not match cell_ids")
            if observed != cell_ids:
                if not allow_identity_reorder:
                    raise _fail(
                        "resume sidecars are reordered without identity mapping"
                    )
                by_id = dict(zip(observed, values, strict=True))
                values = tuple(by_id[cell_id] for cell_id in cell_ids)
                canonical_views = tuple(
                    _projection_mapping(
                        value, name=f"source_projections[{index}]"
                    )
                    for index, value in enumerate(values)
                )
    # The public allocator validator proves schema/config/layout compatibility,
    # private storage, and absence of cross-cell tensor aliases.  Selecting no
    # fields avoids allocating a redundant stacked checkpoint bank.
    stack_validated_per_cell_checkpoint_projections_v1(
        values,
        batch_cells=batch_cells,
        device="cpu",
        field_names=(),
    )
    # Retain exact caller objects, reordered only by explicit sidecar identity.
    return values


def _requests_from_runners(
    runners: Sequence[Any],
    *,
    request_factory: DirectResidentPatchRequestFactoryV1,
    cell_ids: tuple[str, ...],
    logical_generation_count: int,
    physical_window_generations: int,
    qualified_cuda_module: Any,
    cuda_device: str,
    resume_projections: tuple[Any, ...] | None,
) -> tuple[DirectResidentSourceAllocationRequestV1, ...]:
    if not callable(request_factory):
        raise _fail("request_factory must be callable when runners are supplied")
    batch_cells = len(runners)
    requests: list[DirectResidentSourceAllocationRequestV1] = []
    for index, runner in enumerate(runners):
        request = request_factory(
            runner,
            cell_index=index,
            cell_id=cell_ids[index],
            batch_cells=batch_cells,
            logical_generation_count=logical_generation_count,
            physical_window_generations=physical_window_generations,
            qualified_cuda_module=qualified_cuda_module,
            cuda_device=cuda_device,
            resume_projection=(
                None if resume_projections is None else resume_projections[index]
            ),
        )
        if not isinstance(request, DirectResidentSourceAllocationRequestV1):
            raise _fail(
                f"request_factory returned the wrong type for cell {cell_ids[index]!r}"
            )
        requests.append(request)
    return tuple(requests)


def prepare_direct_resident_patch_multicell_v1(
    *,
    cell_ids: Sequence[str],
    per_cell_requests: Sequence[DirectResidentSourceAllocationRequestV1] | None = None,
    runners: Sequence[Any] | None = None,
    request_factory: DirectResidentPatchRequestFactoryV1 | None = None,
    qualified_cuda_module: Any = None,
    cuda_device: str = "cuda",
    logical_generation_count: int = 4,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS,
    agent_id_allocation_generation_count: int | None = None,
    resume_projections: Sequence[Mapping[str, Any]] | None = None,
    tick_native_control: Mapping[str, Any] | None = None,
) -> PreparedDirectResidentPatchMulticellV1:
    """Prepare B1..B72 fixed-Patch cells as one resident physical batch.

    Exactly one input route is accepted.  Runner construction remains a
    campaign/provider concern exposed through ``request_factory``; this facade
    owns only request normalization and engine-boundary composition.
    """

    if type(logical_generation_count) is not int or logical_generation_count < 1:
        raise _fail("logical_generation_count must be a positive exact integer")
    physical_window_generations = validate_physical_window_generations_v1(
        physical_window_generations
    )
    if (per_cell_requests is None) == (runners is None):
        raise _fail("supply exactly one of per_cell_requests or runners")
    raw = per_cell_requests if per_cell_requests is not None else runners
    values = _sequence(raw, name="per_cell_requests" if runners is None else "runners")
    batch_cells = len(values)
    if not 1 <= batch_cells <= 72:
        raise _fail("the fixed-Patch physical batch must contain B1..B72 cells")
    ordered_ids = _validated_cell_ids(cell_ids, count=batch_cells)

    resumes: tuple[Any, ...] | None = None
    if resume_projections is not None:
        resumes = _validated_source_projections(
            resume_projections,
            batch_cells=batch_cells,
            cell_ids=ordered_ids,
            allow_identity_reorder=True,
        )

    if per_cell_requests is None:
        if request_factory is None:
            raise _fail("runner input requires an injected public request_factory")
        requests = _requests_from_runners(
            values,
            request_factory=request_factory,
            cell_ids=ordered_ids,
            logical_generation_count=logical_generation_count,
            physical_window_generations=physical_window_generations,
            qualified_cuda_module=qualified_cuda_module,
            cuda_device=cuda_device,
            resume_projections=resumes,
        )
    else:
        requests = values
        if any(
            not isinstance(request, DirectResidentSourceAllocationRequestV1)
            for request in requests
        ):
            raise _fail("per_cell_requests must contain only public allocation requests")
    request_windows = tuple(
        validate_physical_window_generations_v1(
            getattr(request, "physical_window_generations", PHYSICAL_WINDOW_GENERATIONS)
        )
        for request in requests
    )
    if any(window != physical_window_generations for window in request_windows):
        raise _fail(
            "per-cell physical_window_generations must match the route binding"
        )
    if tick_native_control is not None:
        if not isinstance(tick_native_control, Mapping):
            raise _fail("tick_native_control must be a mapping")
        event_capacity = tick_native_control.get("event_window_capacity")
        if type(event_capacity) is not int or event_capacity <= 0:
            raise _fail("tick-native event_window_capacity is invalid")
        for request in requests:
            if dict(getattr(request, "tick_native_control", {}) or {}) != dict(
                tick_native_control
            ):
                raise _fail("per-cell tick-native controls do not match")

    normalized = tuple(
        replace(
            request,
            batch_cells=batch_cells,
            logical_generation_count=logical_generation_count,
            physical_window_generations=physical_window_generations,
            agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        )
        for request in requests
    )
    source_projections = _validated_source_projections(
        tuple(request.import_projection for request in normalized),
        batch_cells=batch_cells,
        cell_ids=ordered_ids,
    )
    if resumes is not None and any(
        request.import_projection is not resume
        for request, resume in zip(normalized, resumes, strict=True)
    ):
        raise _fail(
            "resume request_factory must bind each validated resume projection "
            "without substitution"
        )

    source = allocate_direct_resident_source_v1(
        normalized[0],
        per_cell_requests=normalized,
        per_cell_ids=ordered_ids,
    )
    prepared = prepare_direct_resident_trajectory_v1(source)
    return PreparedDirectResidentPatchMulticellV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        patch=prepared,
        cell_ids=ordered_ids,
        source_projections=source_projections,
        resumed=resumes is not None,
    )


def _slice_tensor(
    value: Any, *, name: str, cell: int, cells: int, axis: int = 0
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim <= axis:
        raise _fail(f"multicell owner tensor is invalid: {name}")
    if int(value.size(axis)) != cells:
        raise _fail(f"multicell owner tensor has the wrong cell axis: {name}")
    narrowed = value.narrow(axis, cell, 1).detach()
    if value.device.type == "cpu":
        return narrowed.clone().contiguous()
    # Settlement consumes CUDA slices read-only and materializes the canonical
    # CPU checkpoint before advancing to the next cell. A narrow view avoids a
    # second resident CUDA bank at the peak-memory checkpoint boundary.
    return narrowed


def _slice_live_banks(
    banks: Mapping[str, Any], *, cell: int, cells: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    hapb_policy_owner = banks.get("hapb_policy_owner")
    for name, value in banks.items():
        if isinstance(value, torch.Tensor):
            if name == "mlp_pack" and hapb_policy_owner is not None:
                canonical = getattr(hapb_policy_owner, "canonical_host_bank", None)
                # HAPB deliberately keeps a one-row CUDA ABI sentinel at this
                # key.  Checkpoint settlement still needs the exact per-cell
                # CPU row view, which comes solely from the shared host bank.
                out[name] = _slice_tensor(
                    canonical,
                    name="hapb_policy_owner.canonical_host_bank",
                    cell=cell,
                    cells=cells,
                )
                continue
            out[name] = _slice_tensor(
                value, name=f"live_banks.{name}", cell=cell, cells=cells
            )
        elif name == "event_columns":
            columns = _sequence(value, name="live_banks.event_columns")
            out[name] = [
                _slice_tensor(
                    column,
                    name=f"live_banks.event_columns[{index}]",
                    cell=cell,
                    cells=cells,
                )
                for index, column in enumerate(columns)
            ]
        elif name in {
            "hapb_policy_owner",
            "hapb_full_cuda_mlp_pack_absent",
            "event_history_drained",
        }:
            # The HAPB checkpoint seam needs this explicit authority marker to
            # admit its one pageable policy row while retaining the normal
            # one-device rule for every resident bank.
            out[name] = value
        elif name in _SHARED_LIVE_BANK_MAPPINGS_V1:
            if not isinstance(value, Mapping):
                raise _fail(f"live_banks.{name} must be a mapping")
            out[name] = copy.deepcopy(dict(value))
        elif name == "genome_v2_founder_profiles":
            profiles = _sequence(
                value, name="live_banks.genome_v2_founder_profiles"
            )
            if len(profiles) != cells:
                raise _fail(
                    "live_banks.genome_v2_founder_profiles must match cells"
                )
            selected = profiles[cell]
            if (
                not isinstance(selected, Mapping)
                or set(selected)
                != {"schema_version", "profile_id", "payload_sha256"}
            ):
                raise _fail(
                    "live_banks Genome V2 founder profile identity is invalid"
                )
            out["genome_v2_founder_profile"] = copy.deepcopy(dict(selected))
    return out


def project_direct_resident_patch_multicell_generation_zero_v1(
    prepared: PreparedDirectResidentPatchMulticellV1,
) -> tuple[Mapping[str, Any], ...]:
    """Project the initialized resident G0 authority for durable checkpoints."""

    if not isinstance(prepared, PreparedDirectResidentPatchMulticellV1):
        raise _fail(
            "prepared must be PreparedDirectResidentPatchMulticellV1"
        )
    cells = prepared.batch_cells
    if len(prepared.source_projections) != cells:
        raise _fail("prepared owner omitted ordered G0 source projections")
    checkpoint_inputs = prepared.patch.checkpoint_projection_inputs
    if not isinstance(checkpoint_inputs, Mapping):
        raise _fail("prepared owner omitted checkpoint projection inputs")
    authority_binding = checkpoint_inputs.get(
        "genome_v2_checkpoint_authority_binding", ()
    )
    projected: list[Mapping[str, Any]] = []
    for cell, (cell_id, source_projection) in enumerate(
        zip(
            prepared.cell_ids,
            prepared.source_projections,
            strict=True,
        )
    ):
        canonical = _projection_mapping(
            source_projection,
            name=f"source_projections[{cell}]",
        )
        checkpoint = CudaBoundaryCheckpoint(
            projection=dict(canonical),
            projection_digest=projection_digest(canonical),
            device="cpu",
            device_to_host_operations=0,
            device_to_host_bytes=0,
            device_synchronizations=0,
        )
        checkpoint = with_resident_inheritance_sidecars_v1(
            checkpoint,
            _slice_live_banks(
                prepared.patch.live_banks,
                cell=cell,
                cells=cells,
            ),
            cell_id=cell_id,
            genome_v2_checkpoint_authority_binding=authority_binding,
            genome_v2_causal_physical_cell_ordinal=cell,
            include_genome_v2_causal_projection=False,
        )
        checkpoint = CudaBoundaryCheckpoint(
            **{
                **checkpoint.__dict__,
                "projection": _with_checkpoint_contract_metadata(
                    checkpoint.projection
                ),
            }
        )
        validate_native_checkpoint_projection(checkpoint.projection)
        projected.append(checkpoint.projection)
    return tuple(projected)


def _validate_live_pcg64_buffers_before_settlement_v1(
    banks: Mapping[str, Any],
    *,
    cell_ids: tuple[str, ...],
    excluded_cell_indices: frozenset[int] = frozenset(),
) -> None:
    """Reject batched native RNG corruption before ordered cell settlement."""

    cells = len(cell_ids)
    for name in _LIVE_PCG64_BUFFER_FIELDS_V1:
        if name not in banks:
            continue
        value = banks[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.int64
            or tuple(value.shape) != (cells, 2)
        ):
            raise _fail(f"live_banks.{name} must be int64[{cells},2]")
        host = value.detach().to(device="cpu")
        for cell, buffered in enumerate(host.tolist()):
            if cell in excluded_cell_indices:
                continue
            flag, uinteger = (int(buffered[0]), int(buffered[1]))
            if flag not in {0, 1} or not 0 <= uinteger <= 0xFFFFFFFF:
                raise _fail(
                    "native PCG64 buffer is invalid before settlement: "
                    f"field={name} cell_index={cell} "
                    f"cell_id={cell_ids[cell]} values={[flag, uinteger]}"
                )


def _restore_suppressed_pcg64_state_v1(
    banks: Mapping[str, Any],
    *,
    canonical_projection: Mapping[str, Any],
    cell_id: str,
) -> dict[str, Any]:
    """Restore qualification-fault RNG from the validated source boundary."""

    restored = dict(banks)
    metadata: Mapping[str, Any] | None = None
    for prefix, metadata_name in _LIVE_PCG64_METADATA_FIELDS_V1.items():
        state_name = (
            "pcg64_state_words"
            if prefix == "pcg64"
            else f"{prefix}_pcg64_state_words"
        )
        buffer_name = (
            "pcg64_buffer_words"
            if prefix == "pcg64"
            else f"{prefix}_pcg64_buffer_words"
        )
        current_buffer = restored.get(buffer_name)
        if not isinstance(current_buffer, torch.Tensor):
            continue
        current = current_buffer.detach().to(device="cpu").reshape(-1).tolist()
        if (
            len(current) == 2
            and int(current[0]) in {0, 1}
            and 0 <= int(current[1]) <= 0xFFFFFFFF
        ):
            continue
        if metadata is None:
            candidate_metadata = canonical_projection.get("metadata")
            if not isinstance(candidate_metadata, Mapping):
                raise _fail(
                    f"fault-suppressed cell {cell_id!r} lacks canonical RNG "
                    "metadata"
                )
            metadata = candidate_metadata
        source_state = metadata.get(metadata_name)
        if not isinstance(source_state, Mapping):
            raise _fail(
                f"fault-suppressed cell {cell_id!r} lacks {metadata_name!r}"
            )
        try:
            state_words, buffer_words = pcg64_state_tensors(
                source_state, device="cpu"
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise _fail(
                f"fault-suppressed cell {cell_id!r} has invalid "
                f"{metadata_name!r}"
            ) from exc
        target_state = restored.get(state_name)
        if (
            not isinstance(target_state, torch.Tensor)
            or target_state.dtype != torch.uint64
            or tuple(target_state.shape) != (1, 4)
            or current_buffer.dtype != torch.int64
            or tuple(current_buffer.shape) != (1, 2)
        ):
            raise _fail(
                f"fault-suppressed cell {cell_id!r} RNG bank shape drifted"
            )
        restored[state_name] = state_words.to(
            device=target_state.device
        ).reshape(1, 4)
        restored[buffer_name] = buffer_words.to(
            device=current_buffer.device
        ).reshape(1, 2)
    for field_name, (stream_name, counter_name) in (
        _LIVE_SEMANTIC_DRAW_METADATA_FIELDS_V1.items()
    ):
        current = restored.get(field_name)
        if not isinstance(current, torch.Tensor):
            continue
        if (
            current.dtype != torch.int64
            or tuple(current.shape) != (1,)
        ):
            raise _fail(
                f"fault-suppressed cell {cell_id!r} semantic-draw bank "
                "shape drifted"
            )
        if int(current.item()) >= 0:
            continue
        if metadata is None:
            candidate_metadata = canonical_projection.get("metadata")
            if not isinstance(candidate_metadata, Mapping):
                raise _fail(
                    f"fault-suppressed cell {cell_id!r} lacks canonical RNG "
                    "metadata"
                )
            metadata = candidate_metadata
        stream_indices = metadata.get(
            "native_runtime_rng_manager_stream_indices"
        )
        stream = (
            stream_indices.get(stream_name)
            if isinstance(stream_indices, Mapping)
            else None
        )
        value = (
            stream.get(counter_name)
            if isinstance(stream, Mapping)
            else None
        )
        if type(value) is not int or value < 0:
            raise _fail(
                f"fault-suppressed cell {cell_id!r} lacks canonical "
                f"{field_name!r}"
            )
        restored[field_name] = torch.tensor(
            [value], dtype=torch.int64, device=current.device
        )
    return restored


def _slice_outputs(
    outputs: Mapping[str, Any], *, cell: int, cells: int
) -> dict[str, Any]:
    """Select only outputs consumed by canonical settlement.

    The selected tensor and its cell axis are explicit.  Extra owner scratch
    outputs do not cross the settlement boundary and are not guessed from
    coincidental dimensions.
    """

    out: dict[str, Any] = {}
    for name, axis in _SETTLEMENT_OUTPUT_CELL_AXES_V1.items():
        if name not in outputs:
            continue
        out[name] = _slice_tensor(
            outputs[name],
            name=f"outputs.{name}",
            cell=cell,
            cells=cells,
            axis=axis,
        )
    return out


def _slice_direct_result(
    result: Mapping[str, Any], *, cell: int, cells: int
) -> dict[str, Any]:
    out = dict(result)
    summary = result.get("generation_summary_scalar")
    if summary is not None:
        out["generation_summary_scalar"] = _slice_tensor(
            summary,
            name="direct_result.generation_summary_scalar",
            cell=cell,
            cells=cells,
            axis=1,
        )
    elif result.get("absolute_tick_span_native_owner") is not True:
        raise _fail(
            "multicell owner tensor is invalid: "
            "direct_result.generation_summary_scalar"
        )
    out["runtime_bookkeeping_by_tick"] = _slice_tensor(
        result.get("runtime_bookkeeping_by_tick"),
        name="direct_result.runtime_bookkeeping_by_tick",
        cell=cell,
        cells=cells,
        axis=0,
    )
    if result.get("event_history_storage_mode") == "eiel_resident_event_sink_v1":
        rows_value = result.get("event_rows_by_cell")
        if not isinstance(rows_value, Sequence) or isinstance(
            rows_value, (str, bytes, bytearray)
        ) or len(rows_value) != cells:
            raise _fail("external event sink row counts do not match the physical batch")
        out["event_rows_by_cell"] = [copy.deepcopy(rows_value[cell])]
        frontier_value = result.get("event_history_frontier_by_cell")
        if isinstance(frontier_value, Sequence) and not isinstance(
            frontier_value, (str, bytes, bytearray)
        ):
            if len(frontier_value) != cells:
                raise _fail("external event sink frontiers do not match the batch")
            out["event_history_frontier"] = copy.deepcopy(frontier_value[cell])
        else:
            out["event_history_frontier"] = copy.deepcopy(
                result.get("event_history_frontier")
            )
        sequence_base = result.get("event_sequence_base")
        if isinstance(sequence_base, torch.Tensor):
            out["event_sequence_base"] = _slice_tensor(
                sequence_base,
                name="direct_result.event_sequence_base",
                cell=cell,
                cells=cells,
                axis=0,
            )
        reference = result.get("event_history_reference")
        prefix_references = result.get("event_history_prefix_references_by_cell")
        if (
            not isinstance(prefix_references, Sequence)
            or isinstance(prefix_references, (str, bytes, bytearray))
            or len(prefix_references) != cells
            or not isinstance(prefix_references[cell], Mapping)
        ):
            raise _fail("external event sink lacks checkpoint-bound prefix identities")
        if isinstance(reference, Mapping):
            out["event_history_reference"] = copy.deepcopy(
                dict(prefix_references[cell])
            )
        if "source_projection_cell_ids" in out:
            source_ids = out["source_projection_cell_ids"]
            if (
                not isinstance(source_ids, Sequence)
                or isinstance(source_ids, (str, bytes, bytearray))
                or len(source_ids) != cells
            ):
                raise _fail(
                    "direct result source projection cell ids do not match "
                    "the physical batch"
                )
            out["source_projection_cell_ids"] = [source_ids[cell]]
        return out
    histories_value = result.get("event_histories")
    rows_value = result.get("event_rows_by_cell")
    if histories_value is None and rows_value is None:
        if result.get("absolute_tick_span_native_owner") is not True:
            raise _fail("event_histories must be a sequence")
    else:
        histories = _sequence(histories_value, name="event_histories")
        rows = _sequence(rows_value, name="event_rows_by_cell")
        if len(histories) != cells or len(rows) != cells:
            raise _fail("direct result event histories do not match the physical batch")
        out["event_histories"] = [copy.deepcopy(histories[cell])]
        out["event_rows_by_cell"] = [copy.deepcopy(rows[cell])]
    if "source_projection_cell_ids" in out:
        out["source_projection_cell_ids"] = [out["source_projection_cell_ids"][cell]]
    return out


def project_direct_resident_patch_multicell_checkpoints_v1(
    prepared: PreparedDirectResidentPatchMulticellV1,
    *,
    direct_result: Mapping[str, Any],
    capability: Mapping[str, Any],
    reporting_enabled: bool = True,
    reporting_suppressed_cell_indices: tuple[int, ...] = (),
    generation_base_override: int | None = None,
    bounded_projection_cursor: bool = False,
) -> DirectResidentPatchMulticellSettlementV1:
    """Project ordered canonical checkpoints through the accepted B1 contract.

    Each cell is projected independently through the public B1 checkpoint
    contract while the native owner and physical dispatch remain multicell.
    """

    if not isinstance(prepared, PreparedDirectResidentPatchMulticellV1):
        raise _fail("prepared must be PreparedDirectResidentPatchMulticellV1")
    if not isinstance(direct_result, Mapping):
        raise _fail("direct_result must be a mapping")
    if not isinstance(capability, Mapping):
        raise _fail("capability must be a mapping")
    if type(reporting_enabled) is not bool:
        raise _fail("reporting_enabled must be boolean")
    if type(bounded_projection_cursor) is not bool:
        raise _fail("bounded_projection_cursor must be boolean")
    if (
        not isinstance(reporting_suppressed_cell_indices, tuple)
        or any(
            type(cell) is not int or cell < 0 or cell >= prepared.batch_cells
            for cell in reporting_suppressed_cell_indices
        )
        or len(set(reporting_suppressed_cell_indices))
        != len(reporting_suppressed_cell_indices)
    ):
        raise _fail("reporting-suppressed cell indices are invalid")
    suppressed = frozenset(reporting_suppressed_cell_indices)
    completed = direct_result.get("completed_generations")
    logical_generation_count = getattr(
        prepared.patch.source, "logical_generation_count", None
    )
    if (
        type(logical_generation_count) is not int
        or not 1 <= logical_generation_count <= 128
    ):
        raise _fail("prepared owner has no admitted logical generation horizon")
    if (
        type(completed) is not int
        or not 1 <= completed <= logical_generation_count
    ):
        raise _fail("completed generations exceed the prepared owner horizon")
    cells = prepared.batch_cells
    if direct_result.get("event_history_storage_mode") == EVENT_SINK_SCHEMA_VERSION:
        reference = direct_result.get("event_history_reference")
        frontiers = direct_result.get("event_history_frontier_by_cell")
        provided_prefixes = direct_result.get(
            "event_history_prefix_references_by_cell"
        )
        if (
            not isinstance(reference, Mapping)
            or type(reference.get("path")) is not str
            or not isinstance(frontiers, Sequence)
            or isinstance(frontiers, (str, bytes, bytearray))
            or len(frontiers) != cells
            or any(type(value) is not int or value < 0 for value in frontiers)
        ):
            raise _fail("external event sink cannot bind terminal checkpoint prefixes")
        if (
            isinstance(provided_prefixes, Sequence)
            and not isinstance(provided_prefixes, (str, bytes, bytearray))
            and len(provided_prefixes) == cells
            and all(isinstance(row, Mapping) for row in provided_prefixes)
        ):
            prefix_rows = [dict(row) for row in provided_prefixes]
        else:
            try:
                identities = event_sink_prefix_identities_v1(
                    reference["path"],
                    expected_rows_by_cell={
                        cell: int(frontiers[cell]) for cell in range(cells)
                    },
                )
            except EventSinkError as exc:
                raise _fail(f"cannot authenticate external event sink: {exc}") from exc
            prefix_rows = [dict(identities[cell]) for cell in range(cells)]
        direct_result = {
            **dict(direct_result),
            "event_history_prefix_references_by_cell": prefix_rows,
        }
    terminal_generations: set[int] = set()
    terminal_ticks: set[int] = set()
    projections: list[Mapping[str, Any]] = []
    segment_artifacts: list[Mapping[str, Any]] | None = []
    lifecycle_qacct_receipts: list[Mapping[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    source = prepared.patch.source
    generation_base = source.generation_base
    if generation_base_override is not None:
        if type(generation_base_override) is not int or generation_base_override < 0:
            raise _fail("generation_base_override must be a nonnegative integer")
        generation_base = generation_base_override
    canonical_projections = prepared.patch.checkpoint_projection_inputs.get(
        "pre_generation_canonical_projections"
    )
    source_projection_cursor_factory = (
        prepared.source_projection_cursor_factory
        or prepared.patch.checkpoint_projection_inputs.get(
            "source_projection_cursor_factory"
        )
    )
    if (
        not isinstance(canonical_projections, (list, tuple))
        or len(canonical_projections) != cells
    ):
        if not bounded_projection_cursor or not callable(source_projection_cursor_factory):
            raise _fail("prepared owner omitted per-cell canonical checkpoint skeletons")
        canonical_projections = None
    _validate_live_pcg64_buffers_before_settlement_v1(
        prepared.patch.live_banks,
        cell_ids=prepared.cell_ids,
        excluded_cell_indices=suppressed,
    )

    if bounded_projection_cursor:
        source_projection_cursor = (
            source_projection_cursor_factory()
            if canonical_projections is None
            else None
        )
        if canonical_projections is None and (
            source_projection_cursor is None
            or not callable(getattr(source_projection_cursor, "next_cell", None))
            or not callable(
                getattr(source_projection_cursor, "import_and_release", None)
            )
        ):
            raise _fail("bounded checkpoint source cursor is incomplete")

        def projection_tensor_bytes(projection: Mapping[str, Any]) -> int:
            manifest = projection.get("tensor_manifest")
            if not isinstance(manifest, Mapping):
                return 0
            total = 0
            for row in manifest.values():
                if isinstance(row, Mapping) and type(row.get("bytes")) is int:
                    total += max(0, int(row["bytes"]))
            return total

        def project_one_cell(
            cell: int, cell_id: str
        ) -> DetachedCellProjectionV1:
            if canonical_projections is None:
                detached_source = source_projection_cursor.next_cell()
                if detached_source is None:
                    raise _fail(
                        "bounded source cursor ended before checkpoint projection"
                    )
                if (
                    detached_source.cell_id != cell_id
                    or detached_source.cell_index != cell
                ):
                    raise _fail("bounded source cursor emitted unordered projection")
                canonical_projection = detached_source.require_payload()
            else:
                canonical_projection = canonical_projections[cell]
            prefix_rows = direct_result.get(
                "tick_native_source_prefix_by_cell"
            )
            if prefix_rows is not None:
                if (
                    not isinstance(prefix_rows, Sequence)
                    or isinstance(prefix_rows, (str, bytes, bytearray))
                    or len(prefix_rows) != cells
                    or not isinstance(prefix_rows[cell], Mapping)
                ):
                    raise _fail(
                        "tick-native source prefix rows do not cover the physical batch"
                    )
                canonical_projection = _rebase_tick_native_source_projection_v1(
                    _projection_mapping(
                        canonical_projection,
                        name=f"canonical_projections[{cell}]",
                    ),
                    prefix_rows[cell],
                )
            live_banks = _slice_live_banks(
                prepared.patch.live_banks, cell=cell, cells=cells
            )
            if cell in suppressed:
                live_banks = _restore_suppressed_pcg64_state_v1(
                    live_banks,
                    canonical_projection=canonical_projection,
                    cell_id=cell_id,
                )
            try:
                live_banks = materialize_disabled_prediction_state_v1(
                    canonical_projection,
                    live_banks,
                    reset_count_max=getattr(source, "ticks_per_generation", 80),
                )
            except DisabledPredictionSettlementError as exc:
                raise _fail(f"cannot settle disabled prediction state: {exc}") from exc
            outputs = _slice_outputs(source.outputs, cell=cell, cells=cells)
            checkpoint_inputs = dict(prepared.patch.checkpoint_projection_inputs)
            checkpoint_inputs["pre_generation_canonical_projection"] = (
                _projection_mapping(
                    canonical_projection,
                    name=f"canonical_projections[{cell}]",
                )
            )
            lifecycle_overrides = _projection_lifecycle_configuration_overrides_v1(
                checkpoint_inputs["pre_generation_canonical_projection"]
            )
            if not lifecycle_overrides and cell < len(prepared.source_projections):
                source_projection = prepared.source_projections[cell]
                if isinstance(source_projection, Mapping):
                    lifecycle_overrides = (
                        _projection_lifecycle_configuration_overrides_v1(
                            source_projection
                        )
                    )
            checkpoint_inputs["inheritance_sidecar_cell_id"] = cell_id
            physical_cell_ordinals = checkpoint_inputs.get(
                "genome_v2_causal_physical_cell_ordinals"
            )
            checkpoint_inputs["genome_v2_causal_physical_cell_ordinal"] = (
                int(physical_cell_ordinals[cell])
                if isinstance(physical_cell_ordinals, Sequence)
                and not isinstance(
                    physical_cell_ordinals, (str, bytes, bytearray)
                )
                and len(physical_cell_ordinals) == cells
                and type(physical_cell_ordinals[cell]) is int
                and physical_cell_ordinals[cell] >= 0
                else cell
            )
            context = SimpleNamespace(
                source=SimpleNamespace(
                    outputs=outputs,
                    configuration=_configuration_with_overrides_v1(
                        getattr(source, "configuration", None),
                        lifecycle_overrides,
                    ),
                    **lifecycle_overrides,
                    generation_base=generation_base,
                    logical_generation_count=logical_generation_count,
                    ticks_per_generation=getattr(source, "ticks_per_generation", 80),
                ),
                live_banks=live_banks,
                checkpoint_projection_inputs=checkpoint_inputs,
            )
            cell_settlement = settle_patch_p2_b1_direct_resident_prefix_v2(
                context,
                direct_result=_slice_direct_result(
                    direct_result, cell=cell, cells=cells
                ),
                capability=capability,
                reporting_enabled=reporting_enabled and cell not in suppressed,
            )
            if not isinstance(cell_settlement.terminal_checkpoint, CudaBoundaryCheckpoint):
                raise _fail("public Patch settlement returned no canonical checkpoint")
            projection = _with_checkpoint_contract_metadata(
                cell_settlement.terminal_checkpoint.projection
            )
            validate_native_checkpoint_projection(projection)
            compact_rows = tuple(
                {"cell_id": cell_id, **dict(row)}
                for row in cell_settlement.compact_summary_rows
            )
            qacct = None
            if isinstance(cell_settlement.lifecycle_qacct_receipt, Mapping):
                qacct = {
                    "cell_id": cell_id,
                    "start_generation": int(generation_base),
                    "terminal_generation": int(cell_settlement.terminal_generation),
                    **dict(cell_settlement.lifecycle_qacct_receipt),
                }
            detached_projection = DetachedCellProjectionV1(
                schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
                cell_id=cell_id,
                cell_index=cell,
                projection=projection,
                compact_result_rows=compact_rows,
                segment_artifacts=tuple(cell_settlement.segment_artifacts),
                lifecycle_qacct_receipt=qacct,
                terminal_generation=int(cell_settlement.terminal_generation),
                terminal_tick=int(cell_settlement.terminal_tick),
                live_projection_bytes=projection_tensor_bytes(projection),
            )
            if canonical_projections is None:
                source_projection_cursor.import_and_release(cell_id, None)
                if cell == cells - 1:
                    source_receipt = source_projection_cursor.close()
                    if getattr(source_receipt, "state", None) != "SEALED":
                        raise _fail("bounded source cursor did not seal")
            return detached_projection

        cursor = CheckpointProjectionCursorV1.begin(
            owner=prepared,
            coordinate=(
                int(generation_base)
                if direct_result.get("absolute_tick_span_native_owner") is not True
                else int(
                    direct_result.get(
                        "tick_span_end",
                        direct_result.get("tick_native_end_next_tick", 0),
                    )
                )
            ),
            ordered_cell_ids=prepared.cell_ids,
            factory=project_one_cell,
            on_fail=(
                (lambda reason: source_projection_cursor.fail(reason))
                if source_projection_cursor is not None
                else None
            ),
        )
        if direct_result.get("absolute_tick_span_native_owner") is True:
            span_end = direct_result.get("tick_span_end")
            boundary = direct_result.get("terminal_generation_boundary")
            if type(span_end) is not int or type(boundary) is not bool:
                raise _fail("native absolute-tick result lacks cursor terminal metadata")
            cursor_terminal_tick = 0 if boundary else span_end
            cursor_terminal_generation = (
                generation_base + completed
                if cursor_terminal_tick == 0
                else generation_base
            )
        else:
            cursor_terminal_tick = 0
            cursor_terminal_generation = generation_base + completed
        option_d_m0_telemetry_enabled = direct_result.get(
            "option_d_m0_telemetry_enabled", False
        )
        if type(option_d_m0_telemetry_enabled) is not bool:
            raise _fail("Option D M0 telemetry marker must be an exact boolean")
        settlement_event_rows = (
            _settlement_event_rows_v1(
                direct_result=direct_result,
                cell_ids=prepared.cell_ids,
                generation_base=int(generation_base),
                completed_generations=completed,
            )
            if option_d_m0_telemetry_enabled
            else ()
        )
        return DirectResidentPatchMulticellSettlementV1(
            schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
            cell_ids=prepared.cell_ids,
            checkpoint_projections=(),
            compact_result_rows=(),
            completed_generation_count=completed,
            terminal_generation=cursor_terminal_generation,
            terminal_tick=cursor_terminal_tick,
            settlement_event_rows=settlement_event_rows,
            checkpoint_projection_cursor=cursor,
        )

    for cell, cell_id in enumerate(prepared.cell_ids):
        canonical_projection = canonical_projections[cell]
        prefix_rows = direct_result.get("tick_native_source_prefix_by_cell")
        if prefix_rows is not None:
            if (
                not isinstance(prefix_rows, Sequence)
                or isinstance(prefix_rows, (str, bytes, bytearray))
                or len(prefix_rows) != cells
                or not isinstance(prefix_rows[cell], Mapping)
            ):
                raise _fail(
                    "tick-native source prefix rows do not cover the physical batch"
                )
            canonical_projection = _rebase_tick_native_source_projection_v1(
                _projection_mapping(
                    canonical_projection,
                    name=f"canonical_projections[{cell}]",
                ),
                prefix_rows[cell],
            )
        live_banks = _slice_live_banks(
            prepared.patch.live_banks, cell=cell, cells=cells
        )
        if cell in suppressed:
            live_banks = _restore_suppressed_pcg64_state_v1(
                live_banks,
                canonical_projection=canonical_projection,
                cell_id=cell_id,
            )
        try:
            live_banks = materialize_disabled_prediction_state_v1(
                canonical_projection,
                live_banks,
                # Preserve the explicit owner geometry at the checkpoint
                # boundary; 80 is only the legacy source fallback.
                reset_count_max=getattr(source, "ticks_per_generation", 80),
            )
        except DisabledPredictionSettlementError as exc:
            raise _fail(f"cannot settle disabled prediction state: {exc}") from exc
        outputs = _slice_outputs(source.outputs, cell=cell, cells=cells)
        checkpoint_inputs = dict(prepared.patch.checkpoint_projection_inputs)
        checkpoint_inputs["pre_generation_canonical_projection"] = (
            _projection_mapping(
                canonical_projection,
                name=f"canonical_projections[{cell}]",
            )
        )
        lifecycle_overrides = _projection_lifecycle_configuration_overrides_v1(
            checkpoint_inputs["pre_generation_canonical_projection"]
        )
        if not lifecycle_overrides and cell < len(prepared.source_projections):
            source_projection = prepared.source_projections[cell]
            if isinstance(source_projection, Mapping):
                lifecycle_overrides = (
                    _projection_lifecycle_configuration_overrides_v1(
                        source_projection
                    )
                )
        checkpoint_inputs["inheritance_sidecar_cell_id"] = cell_id
        physical_cell_ordinals = checkpoint_inputs.get(
            "genome_v2_causal_physical_cell_ordinals"
        )
        checkpoint_inputs["genome_v2_causal_physical_cell_ordinal"] = (
            int(physical_cell_ordinals[cell])
            if isinstance(physical_cell_ordinals, Sequence)
            and not isinstance(physical_cell_ordinals, (str, bytes, bytearray))
            and len(physical_cell_ordinals) == cells
            and type(physical_cell_ordinals[cell]) is int
            and physical_cell_ordinals[cell] >= 0
            else cell
        )
        context = SimpleNamespace(
            source=SimpleNamespace(
                outputs=outputs,
                configuration=_configuration_with_overrides_v1(
                    getattr(source, "configuration", None),
                    lifecycle_overrides,
                ),
                **lifecycle_overrides,
                generation_base=generation_base,
                logical_generation_count=logical_generation_count,
                ticks_per_generation=getattr(source, "ticks_per_generation", 80),
            ),
            live_banks=live_banks,
            checkpoint_projection_inputs=checkpoint_inputs,
        )
        settlement = settle_patch_p2_b1_direct_resident_prefix_v2(
            context,
            direct_result=_slice_direct_result(
                direct_result, cell=cell, cells=cells
            ),
            capability=capability,
            reporting_enabled=reporting_enabled and cell not in suppressed,
        )
        if not isinstance(settlement.terminal_checkpoint, CudaBoundaryCheckpoint):
            raise _fail("public Patch settlement returned no canonical checkpoint")
        projection = _with_checkpoint_contract_metadata(
            settlement.terminal_checkpoint.projection
        )
        validate_native_checkpoint_projection(projection)
        cell_compact_rows = tuple(
            {"cell_id": cell_id, **dict(row)}
            for row in settlement.compact_summary_rows
        )
        if segment_artifacts is not None and settlement.segment_artifacts:
            segment_artifacts.extend(settlement.segment_artifacts)
        else:
            segment_artifacts = None
        if isinstance(settlement.lifecycle_qacct_receipt, Mapping):
            lifecycle_qacct_receipts.append(
                {
                    "cell_id": cell_id,
                    "start_generation": int(generation_base),
                    "terminal_generation": int(settlement.terminal_generation),
                    **dict(settlement.lifecycle_qacct_receipt),
                }
            )
        terminal_generations.add(int(settlement.terminal_generation))
        terminal_ticks.add(int(settlement.terminal_tick))
        projections.append(projection)
        compact_rows.extend(cell_compact_rows)
    if len(terminal_generations) != 1:
        raise _fail("per-cell terminal generations disagree")
    if len(terminal_ticks) != 1:
        raise _fail("per-cell terminal ticks disagree")
    option_d_m0_telemetry_enabled = direct_result.get(
        "option_d_m0_telemetry_enabled",
        False,
    )
    if type(option_d_m0_telemetry_enabled) is not bool:
        raise _fail("Option D M0 telemetry marker must be an exact boolean")
    generation_base = int(generation_base)
    if option_d_m0_telemetry_enabled and (
        type(generation_base) is not int or generation_base < 0
    ):
        raise _fail("Option D M0 settlement generation base is invalid")
    settlement_event_rows = (
        _settlement_event_rows_v1(
            direct_result=direct_result,
            cell_ids=prepared.cell_ids,
            generation_base=generation_base,
            completed_generations=completed,
        )
        if option_d_m0_telemetry_enabled
        else ()
    )
    return DirectResidentPatchMulticellSettlementV1(
        schema_version=PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1,
        cell_ids=prepared.cell_ids,
        checkpoint_projections=tuple(projections),
        compact_result_rows=tuple(compact_rows),
        completed_generation_count=completed,
        terminal_generation=terminal_generations.pop(),
        terminal_tick=terminal_ticks.pop(),
        segment_artifacts=tuple(segment_artifacts or ()),
        lifecycle_qacct_receipts=tuple(lifecycle_qacct_receipts),
        settlement_event_rows=settlement_event_rows,
    )


__all__ = [
    "CHECKPOINT_PROJECTION_CURSOR_RECEIPT_SCHEMA_VERSION_V1",
    "CHECKPOINT_PROJECTION_CURSOR_SCHEMA_VERSION_V1",
    "PATCH_MULTICELL_FACADE_SCHEMA_VERSION_V1",
    "SOURCE_IMPORT_CURSOR_RECEIPT_SCHEMA_VERSION_V1",
    "SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1",
    "CudaArchitectureBDirectResidentPatchV1Error",
    "CheckpointProjectionCursorReceiptV1",
    "CheckpointProjectionCursorV1",
    "DetachedCellProjectionV1",
    "DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1",
    "DetachedSourcePayloadV1",
    "DirectResidentPatchMulticellSettlementV1",
    "DirectResidentPatchRequestFactoryV1",
    "PreparedDirectResidentPatchMulticellV1",
    "SourceImportCursorReceiptV1",
    "SourceImportCursorV1",
    "prepare_direct_resident_patch_multicell_v1",
    "project_direct_resident_patch_multicell_generation_zero_v1",
    "project_direct_resident_patch_multicell_checkpoints_v1",
]
