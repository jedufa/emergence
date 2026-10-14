"""Localized execution-provider contracts for compiled campaign plans.

Providers translate a generic plan into a runtime request. Registration and
historical qualification do not confer current production authority.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from dataclasses import dataclass, field
import copy
import gc
import hashlib
import inspect
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Final, Mapping, Protocol, Sequence

from .contracts import (
    CampaignPlanV1,
    ExecutionRouteContractV1,
    UnsupportedReasonV1,
)
from .qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION
from .production_engine_identity import (
    PRODUCTION_CHECKPOINT_SCHEMA_VERSION,
    PRODUCTION_CPU_PROFILE_ID,
    PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION,
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
    PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION,
)
from .route_sentinels import RouteTransitionSentinelV1
from .tick_native_control import TickNativeControlError, TickNativeControlV1

# The simulation package still exposes legacy top-level imports such as
# ``evolution`` and ``simulation``.  Campaign providers are a public entrypoint,
# so make that existing engine root available before lazy runner imports occur.
_LEGACY_ENGINE_IMPORT_ROOT_V1 = str(Path(__file__).resolve().parents[1])
if _LEGACY_ENGINE_IMPORT_ROOT_V1 not in sys.path:
    sys.path.insert(0, _LEGACY_ENGINE_IMPORT_ROOT_V1)


class CampaignProviderError(RuntimeError):
    """A provider cannot translate or own the requested campaign."""


def _tick_native_runner_source_dimensions_v1(
    control: Mapping[str, Any],
) -> tuple[int, int]:
    """Derive the runner's private source-buffer dimensions from native control.

    The patch runner still has a private generation-shaped storage ABI, but an
    active tick-native caller may reach it only through the canonical bounded
    source allocator projection.  Requested horizon and segment boundaries do
    not participate in this derivation.
    """

    from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
        CudaArchitectureBDirectResidentSourceAllocatorError,
        _tick_native_allocator_projection_v1,
    )

    if not isinstance(control, Mapping):
        raise CampaignProviderError(
            "tick-native runner source dimensions require native control"
        )
    try:
        return _tick_native_allocator_projection_v1(control)
    except CudaArchitectureBDirectResidentSourceAllocatorError as exc:
        raise CampaignProviderError(
            "tick-native runner source dimensions are invalid"
        ) from exc


@dataclass(frozen=True, slots=True)
class _TickNativePrivateOwnerAdapterV1:
    """Private legacy-ABI coordinates derived from one absolute-tick owner."""

    owner_start_tick: int
    owner_end_tick: int
    source_capacity_window_count: int
    source_window_ticks: int
    private_source_start_index: int
    private_source_end_index: int

    @property
    def private_source_window_count(self) -> int:
        return self.private_source_end_index - self.private_source_start_index

    @property
    def private_physical_window_generation_count(self) -> int:
        """Return the frozen one-window persistence ABI, never a time model."""

        return 1

    @property
    def private_agent_id_allocation_generation_count(self) -> int:
        """Return the bounded identity envelope derived from source capacity."""

        return max(2, self.source_capacity_window_count)

    @property
    def runner_directory_name(self) -> str:
        return (
            f"owner-tick{self.owner_start_tick:012d}-"
            f"tick{self.owner_end_tick:012d}"
        )


def _tick_native_private_owner_adapter_v1(
    control: Mapping[str, Any],
    *,
    owner_start_tick: int,
    owner_end_tick: int,
) -> _TickNativePrivateOwnerAdapterV1:
    """Translate absolute owner ticks at the one private legacy ABI boundary."""

    source_capacity_window_count, source_window_ticks = (
        _tick_native_runner_source_dimensions_v1(control)
    )
    total_ticks = control.get("total_ticks")
    if (
        type(owner_start_tick) is not int
        or type(owner_end_tick) is not int
        or owner_start_tick < 0
        or owner_end_tick <= owner_start_tick
        or type(total_ticks) is not int
        or owner_end_tick > total_ticks
    ):
        raise CampaignProviderError(
            "tick-native resident owner requires one valid absolute-tick span"
        )
    private_source_start_index = owner_start_tick // source_window_ticks
    private_source_end_index = (
        owner_end_tick + source_window_ticks - 1
    ) // source_window_ticks
    if private_source_end_index <= private_source_start_index:
        raise CampaignProviderError(
            "tick-native resident owner has an empty private source envelope"
        )
    return _TickNativePrivateOwnerAdapterV1(
        owner_start_tick=owner_start_tick,
        owner_end_tick=owner_end_tick,
        source_capacity_window_count=source_capacity_window_count,
        source_window_ticks=source_window_ticks,
        private_source_start_index=private_source_start_index,
        private_source_end_index=private_source_end_index,
    )


def _trim_cuda_allocator_after_resident_owner_preparation_v1(
    *, torch_module: Any | None = None
) -> None:
    """Trim only unoccupied CUDA cache after a new resident owner is prepared.

    The call site is deliberately after owner preparation has returned, so
    temporary preparation frames are no longer live.  ``empty_cache`` cannot
    release live tensors, and this helper issues no explicit device
    synchronization.  An allocator failure is nevertheless a
    capacity-integrity failure and must remain visible.
    """

    gc.collect()
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return
    try:
        if not torch_module.cuda.is_available():
            return
        torch_module.cuda.empty_cache()
    except Exception as exc:
        raise CampaignProviderError(
            "CUDA allocator trim failed after resident owner preparation"
        ) from exc


def hapb_host_authoritative_policy_binding_v1(mechanics_binding: Any) -> bool:
    """Return whether policy state is host-authoritative for this binding."""

    return getattr(mechanics_binding, "binding_id", None) == (
        "wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1"
    )


RESIDENT_OWNER_SPAN_SCHEMA_VERSION_V1: Final[str] = "eiel_resident_owner_span_v1"
COMPACT_SEGMENT_SETTLEMENT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_compact_segment_settlement_v1"
)
OWNER_RELEASE_RECEIPT_SCHEMA_VERSION_V1: Final[str] = "eiel_owner_release_receipt_v1"
OWNER_EPOCH_TRANSFER_RECEIPT_SCHEMA_VERSION_V1: Final[str] = (
    "eiel_resident_owner_epoch_transfer_v1"
)
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


def _tick_native_retained_prior_checkpoint_catalog_v1(
    checkpoint_catalog: Mapping[str, Mapping[str, Any]],
    *,
    requested_checkpoint_ticks: Sequence[int],
    maximum_tick: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Retain only declared tick checkpoints in a flat predecessor catalog.

    Owner-boundary checkpoints are rolling R1 continuation state.  They remain
    available until a successor checkpoint is published, but they must not be
    recursively embedded into every later checkpoint or become durable merely
    because the owner span ended.
    """

    requested = {int(value) for value in requested_checkpoint_ticks}
    retained: dict[str, dict[str, Any]] = {}
    for coordinate, raw in checkpoint_catalog.items():
        try:
            tick = int(coordinate)
        except (TypeError, ValueError) as exc:
            raise CampaignProviderError(
                "tick-native checkpoint catalog contains a noncanonical coordinate"
            ) from exc
        if tick not in requested:
            continue
        if maximum_tick is not None and tick > maximum_tick:
            continue
        value = dict(raw)
        value.pop("prior_checkpoint_catalog", None)
        retained[str(tick)] = value
    return retained


def _tick_native_catalog_after_async_publication_v1(
    checkpoint_catalog: Mapping[str, Mapping[str, Any]],
    *,
    published_coordinate: int,
    published_checkpoint: Mapping[str, Any],
    requested_checkpoint_ticks: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Publish a lagging ACS result without rewinding continuation state.

    An asynchronous writer may complete after compute has already crossed a
    later owner boundary. Requested checkpoints remain durable history, while
    every coordinate newer than the completing writer remains live
    continuation authority until a still-newer replacement is published.
    """

    if type(published_coordinate) is not int or published_coordinate < 0:
        raise CampaignProviderError(
            "tick-native asynchronous checkpoint coordinate is invalid"
        )
    requested = {int(value) for value in requested_checkpoint_ticks}
    updated: dict[str, dict[str, Any]] = {}
    for coordinate, raw in checkpoint_catalog.items():
        try:
            tick = int(coordinate)
        except (TypeError, ValueError) as exc:
            raise CampaignProviderError(
                "tick-native checkpoint catalog contains a noncanonical coordinate"
            ) from exc
        if tick not in requested and tick <= published_coordinate:
            continue
        value = dict(raw)
        value.pop("prior_checkpoint_catalog", None)
        updated[str(tick)] = value
    updated[str(published_coordinate)] = dict(published_checkpoint)
    return updated


def _record_tick_native_async_publication_v1(
    context: Any,
    *,
    published_coordinate: int,
    published_checkpoint: Mapping[str, Any],
    postpublication_catalog: Mapping[str, Mapping[str, Any]],
) -> int:
    """Inventory a lagging artifact while keeping recovery at the live frontier."""

    if str(published_coordinate) not in postpublication_catalog:
        raise CampaignProviderError(
            "tick-native asynchronous publication is absent from its catalog"
        )
    frontier = max((int(value) for value in postpublication_catalog), default=-1)
    if frontier < published_coordinate:
        raise CampaignProviderError(
            "tick-native asynchronous publication rewinds its catalog frontier"
        )
    context.record_checkpoint(published_checkpoint)
    if frontier > published_coordinate:
        # record_checkpoint also owns lifecycle inventory binding. Rebind the
        # already-sealed newer continuation after inventorying the lagging
        # artifact so the manifest's recovery pointer cannot move backwards.
        context.record_checkpoint(postpublication_catalog[str(frontier)])
    return frontier


def _checkpoint_payload_paths_v1(checkpoint: Mapping[str, Any]) -> tuple[Path, ...]:
    rows = checkpoint.get("cell_bundles")
    if not isinstance(rows, list):
        return ()
    paths: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        raw_path = row.get("path") if isinstance(row, Mapping) else None
        if not isinstance(raw_path, str) or not raw_path:
            continue
        unresolved = Path(raw_path).expanduser()
        if unresolved.is_symlink():
            raise CampaignProviderError(
                "checkpoint payload path cannot be a symlink"
            )
        path = unresolved.resolve()
        rendered = str(path)
        if rendered not in seen:
            seen.add(rendered)
            paths.append(path)
    return tuple(paths)


def _release_tick_native_transient_storage_v1(
    *,
    context: Any,
    prior_checkpoint_catalog: Mapping[str, Mapping[str, Any]],
    retained_checkpoint_catalog: Mapping[str, Mapping[str, Any]],
    replacement_checkpoint: Mapping[str, Any],
    requested_checkpoint_ticks: Sequence[int],
    release_source_import: bool,
) -> list[dict[str, Any]]:
    """Release exact producer-owned R1 state after durable replacement.

    This is rolling checkpoint/source-store settlement inside the producer,
    not lifecycle pruning.  Every target is an exact child of the current
    physical batch and every release is bound to the published replacement.
    """

    batch_root = Path(context.batch_root).expanduser().resolve()
    checkpoint_root_raw = batch_root / "checkpoint_native_aggregate_v1"
    native_checkpoint_root_raw = batch_root / "checkpoint_native_v1"
    bundle_root_raw = batch_root / "checkpoint_bundles_v2"
    world_checkpoint_root_raw = batch_root / "checkpoint_world_sharded_v1"
    source_root_raw = batch_root / "source_import_v1"
    if any(
        path.is_symlink()
        for path in (
            checkpoint_root_raw,
            native_checkpoint_root_raw,
            bundle_root_raw,
            world_checkpoint_root_raw,
            source_root_raw,
        )
    ):
        raise CampaignProviderError("transient storage root cannot be a symlink")
    checkpoint_root = checkpoint_root_raw.resolve()
    native_checkpoint_root = native_checkpoint_root_raw.resolve()
    bundle_root = bundle_root_raw.resolve()
    world_checkpoint_root = world_checkpoint_root_raw.resolve()
    checkpoint_roots = (
        checkpoint_root,
        native_checkpoint_root,
        bundle_root,
        world_checkpoint_root,
    )
    if any(batch_root not in root.parents for root in checkpoint_roots):
        raise CampaignProviderError("transient checkpoint root escapes its batch")
    requested = {int(value) for value in requested_checkpoint_ticks}
    retained_paths = {
        str(path)
        for checkpoint in retained_checkpoint_catalog.values()
        for path in _checkpoint_payload_paths_v1(checkpoint)
    }
    replacement_paths = {
        str(path) for path in _checkpoint_payload_paths_v1(replacement_checkpoint)
    }
    replacement_identity = replacement_checkpoint.get("checkpoint_identity_sha256")
    if (
        not isinstance(replacement_identity, str)
        or re.fullmatch(r"[0-9a-f]{64}", replacement_identity) is None
    ):
        raise CampaignProviderError(
            "transient storage settlement lacks replacement checkpoint identity"
        )
    replacement_coordinate = replacement_checkpoint.get("next_absolute_tick")
    if type(replacement_coordinate) is not int or replacement_coordinate < 0:
        raise CampaignProviderError(
            "transient storage settlement lacks a tick-native replacement coordinate"
        )
    checkpoint_release_needed = any(
        int(coordinate) not in requested
        and any(
            str(path) not in retained_paths
            and str(path) not in replacement_paths
            and path.exists()
            for path in _checkpoint_payload_paths_v1(checkpoint)
        )
        for coordinate, checkpoint in prior_checkpoint_catalog.items()
    )
    source_release_needed = bool(release_source_import and source_root_raw.exists())
    if not checkpoint_release_needed and not source_release_needed:
        return []
    manifest_path = Path(context.manifest_path).expanduser().resolve()
    run_root = Path(context.run_root).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, list):
        raise CampaignProviderError(
            "batch manifest lacks artifact inventory during rolling release"
        )
    if manifest.get("artifact_inventory_complete") is False:
        raise CampaignProviderError(
            "batch checkpoint inventory is incomplete during rolling release"
        )
    inventoried_rows: dict[str, Mapping[str, Any]] = {}
    for row in inventory:
        if not isinstance(row, Mapping):
            continue
        raw_inventory_path = row.get("path")
        if not isinstance(raw_inventory_path, str) or not raw_inventory_path:
            continue
        unresolved_inventory_path = Path(raw_inventory_path).expanduser()
        if not unresolved_inventory_path.is_absolute():
            unresolved_inventory_path = run_root / unresolved_inventory_path
        if unresolved_inventory_path.is_symlink():
            raise CampaignProviderError(
                "replacement checkpoint inventory cannot contain a symlink"
            )
        resolved_inventory_path = unresolved_inventory_path.resolve()
        if run_root not in resolved_inventory_path.parents:
            raise CampaignProviderError(
                "replacement checkpoint inventory escapes the run root"
            )
        inventoried_rows[str(resolved_inventory_path)] = row
    if not replacement_paths or not replacement_paths.issubset(inventoried_rows):
        raise CampaignProviderError(
            "transient storage settlement requires an inventoried replacement checkpoint"
        )
    for rendered in sorted(replacement_paths):
        row = inventoried_rows[rendered]
        path = Path(rendered)
        if not any(root in path.parents for root in checkpoint_roots):
            raise CampaignProviderError(
                "replacement checkpoint escapes its physical batch roots"
            )
        observed = os.lstat(path)
        sealed_stat = row.get("producer_sealed_stat")
        expected_stat = {
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
            "bytes": int(observed.st_size),
            "mtime_ns": int(observed.st_mtime_ns),
            "ctime_ns": int(observed.st_ctime_ns),
        }
        artifact_type = row.get("artifact_type")
        digest = row.get("sha256") if artifact_type == "file" else None
        if (
            path.is_symlink()
            or not path.is_file()
            or observed.st_nlink != 1
            or row.get("producer_checkpoint_identity_sha256")
            != replacement_identity
            or row.get("artifact_family") != "checkpoint.bundle_v2"
            or sealed_stat != expected_stat
            or type(row.get("bytes")) is not int
            or row["bytes"] < 0
            or row["bytes"] != int(observed.st_size)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise CampaignProviderError(
                "transient storage settlement replacement seal is incomplete or stale"
            )
    released: list[dict[str, Any]] = []
    released_paths: set[str] = set()
    for coordinate, checkpoint in prior_checkpoint_catalog.items():
        tick = int(coordinate)
        if tick in requested:
            continue
        for path in _checkpoint_payload_paths_v1(checkpoint):
            rendered = str(path)
            if rendered in retained_paths or rendered in replacement_paths:
                continue
            if tick > replacement_coordinate and path.exists():
                raise CampaignProviderError(
                    "transient storage settlement cannot release checkpoint "
                    f"{tick} newer than replacement {replacement_coordinate}: {path}"
                )
            if not any(
                path == root or root in path.parents for root in checkpoint_roots
            ):
                raise CampaignProviderError(
                    f"transient checkpoint escapes its batch root: {path}"
                )
            if path.is_symlink():
                raise CampaignProviderError(
                    f"transient checkpoint is a symlink: {path}"
                )
            if not path.exists():
                continue
            if path.is_file():
                observed = os.lstat(path)
                if observed.st_nlink != 1:
                    raise CampaignProviderError(
                        f"transient checkpoint is hard-linked: {path}"
                    )
                bytes_released = int(observed.st_size)
                path.unlink()
            elif path.is_dir():
                checkpoint_members = list(path.rglob("*"))
                if any(member.is_symlink() for member in checkpoint_members):
                    raise CampaignProviderError(
                        f"transient checkpoint tree contains a symlink: {path}"
                    )
                if any(
                    not member.is_file() and not member.is_dir()
                    for member in checkpoint_members
                ):
                    raise CampaignProviderError(
                        f"transient checkpoint tree contains a special file: {path}"
                    )
                checkpoint_files = [
                    member for member in checkpoint_members if member.is_file()
                ]
                if any(os.lstat(member).st_nlink != 1 for member in checkpoint_files):
                    raise CampaignProviderError(
                        f"transient checkpoint tree contains a hard link: {path}"
                    )
                bytes_released = sum(
                    int(os.lstat(member).st_size) for member in checkpoint_files
                )
                shutil.rmtree(path)
            else:
                raise CampaignProviderError(
                    f"transient checkpoint is not a regular file/tree: {path}"
                )
            released_paths.add(rendered)
            released.append(
                {
                    "schema_version": "eiel_tick_native_transient_storage_release_v1",
                    "artifact_kind": "rolling_owner_checkpoint",
                    "coordinate": tick,
                    "path": rendered,
                    "bytes_released": bytes_released,
                    "replacement_checkpoint_identity_sha256": replacement_identity,
                    "release_predicate": (
                        "successor checkpoint published and historical coordinate "
                        "is not a declared checkpoint"
                    ),
                }
            )

    source_root = source_root_raw.resolve()
    if batch_root not in source_root.parents:
        raise CampaignProviderError("source import root escapes its batch")
    if release_source_import and source_root.exists() and not source_root.is_dir():
        raise CampaignProviderError("source import root is not a directory")
    if (
        release_source_import
        and source_root.is_dir()
        and not source_root.is_symlink()
    ):
        bundle_rows = replacement_checkpoint.get("cell_bundles")
        valid_bundle_rows = (
            [
                row
                for row in bundle_rows
                if isinstance(row, Mapping)
                and isinstance(row.get("cell_id"), str)
                and row["cell_id"]
                and isinstance(row.get("path"), str)
                and row["path"]
            ]
            if isinstance(bundle_rows, list)
            else []
        )
        replacement_cell_ids = [str(row["cell_id"]) for row in valid_bundle_rows]
        expected_cell_ids = manifest.get("cell_ids")
        source_digests = replacement_checkpoint.get(
            "source_import_projection_digest_sha256s"
        )
        terminal_digests = replacement_checkpoint.get("terminal_cell_digests")
        if (
            not replacement_cell_ids
            or not isinstance(expected_cell_ids, list)
            or any(
                not isinstance(cell_id, str) or not cell_id
                for cell_id in expected_cell_ids
            )
            or len(set(expected_cell_ids)) != len(expected_cell_ids)
            or replacement_cell_ids != expected_cell_ids
            or not isinstance(bundle_rows, list)
            or len(valid_bundle_rows) != len(bundle_rows)
            or len(set(replacement_cell_ids)) != len(replacement_cell_ids)
            or not isinstance(source_digests, Mapping)
            or set(source_digests) != set(replacement_cell_ids)
            or not isinstance(terminal_digests, Mapping)
            or set(terminal_digests) != set(replacement_cell_ids)
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (
                    *source_digests.values(),
                    *terminal_digests.values(),
                )
            )
        ):
            raise CampaignProviderError(
                "source import release lacks exact source digest coverage"
            )
        source_members = list(source_root.rglob("*"))
        if any(member.is_symlink() for member in source_members):
            raise CampaignProviderError(
                "source import tree contains a symlink"
            )
        if any(
            not member.is_file() and not member.is_dir()
            for member in source_members
        ):
            raise CampaignProviderError(
                "source import tree contains a special file"
            )
        source_files = [member for member in source_members if member.is_file()]
        if any(os.lstat(member).st_nlink != 1 for member in source_files):
            raise CampaignProviderError(
                "source import tree contains a hard link"
            )
        bytes_released = sum(
            int(os.lstat(member).st_size) for member in source_files
        )
        shutil.rmtree(source_root)
        released_paths.add(str(source_root))
        released.append(
            {
                "schema_version": "eiel_tick_native_transient_storage_release_v1",
                "artifact_kind": "initial_source_import_store",
                "path": str(source_root),
                "objects_released": len(source_files),
                "bytes_released": bytes_released,
                "source_projection_digest_sha256s": dict(source_digests),
                "replacement_checkpoint_identity_sha256": replacement_identity,
                "release_predicate": (
                    "first durable successor checkpoint published after bounded "
                    "source import"
                ),
            }
        )

    if released_paths:
        manifest["artifact_inventory"] = [
            row
            for row in inventory
            if not (
                isinstance(row, Mapping)
                and str(
                    (
                        Path(str(row.get("path") or "")).expanduser()
                        if Path(str(row.get("path") or "")).is_absolute()
                        else run_root / str(row.get("path") or "")
                    ).resolve()
                ) in released_paths
            )
        ]
        from engine.campaigns.orchestration import atomic_write_json

        atomic_write_json(manifest_path, manifest)

    for receipt in released:
        core = dict(receipt)
        receipt["receipt_sha256"] = _canonical_sha256(core)
    return released


@dataclass(slots=True)
class ResidentOwnerSpanV1:
    """One native owner retained across private tick-native subspans."""

    cell_ids: tuple[str, ...]
    start_next_tick: int
    end_next_tick: int
    next_absolute_tick: int
    prepared_owner_identity: str
    builds: tuple[Any, ...]
    requests: tuple[Any, ...]
    prepared_owner: Any
    # Tensor-free source cursor/runtime prefix for each physical cell.  This
    # is bounded continuation metadata, not a host staging copy of state.
    tick_native_prefix_by_cell: tuple[dict[str, Any], ...] = ()
    # One physical CUDA allocation may cross several bounded logical owner
    # epochs.  The logical identity changes at every owner-span boundary while
    # this identity remains fixed until the allocation is actually released.
    physical_owner_identity: str = ""
    closed: bool = False

    def __post_init__(self) -> None:
        if not self.physical_owner_identity:
            self.physical_owner_identity = self.prepared_owner_identity

    def assert_cursor(self, *, start_next_tick: int, end_next_tick: int) -> None:
        if self.closed:
            raise CampaignProviderError("resident owner span is already closed")
        if start_next_tick != self.next_absolute_tick:
            raise CampaignProviderError(
                "resident owner span cursor is not contiguous"
            )
        if not (
            self.start_next_tick
            <= start_next_tick
            < end_next_tick
            <= self.end_next_tick
        ):
            raise CampaignProviderError(
                "resident subspan escapes its owner-span bounds"
            )


@dataclass(frozen=True, slots=True)
class CompactSegmentSettlementV1:
    """Tensor-free observation emitted at every native dispatch boundary."""

    schema_version: str
    cell_ids: tuple[str, ...]
    start_next_tick: int
    end_next_tick: int
    owner_identity: str
    compact_result_rows: tuple[dict[str, Any], ...] = ()
    event_history_reference_by_cell: tuple[Mapping[str, Any], ...] = ()
    event_drain_receipts: tuple[Mapping[str, Any], ...] = ()
    event_frontier_by_cell: tuple[int, ...] = ()
    event_pending_rows_by_cell: tuple[int, ...] = ()
    native_execution_receipt: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OwnerReleaseReceiptV1:
    """Proof that one resident owner was released before its successor."""

    schema_version: str
    owner_identity: str
    final_next_absolute_tick: int
    release_synchronization_complete: bool
    live_alias_count: int
    post_release_cuda_allocated_bytes: int
    post_release_cuda_reserved_bytes: int
    released_perf_ns: int
    released_unix_ns: int

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "owner_identity": self.owner_identity,
            "final_next_absolute_tick": self.final_next_absolute_tick,
            "release_synchronization_complete": (
                self.release_synchronization_complete
            ),
            "live_alias_count": self.live_alias_count,
            "post_release_cuda_allocated_bytes": (
                self.post_release_cuda_allocated_bytes
            ),
            "post_release_cuda_reserved_bytes": (
                self.post_release_cuda_reserved_bytes
            ),
            "released_perf_ns": self.released_perf_ns,
            "released_unix_ns": self.released_unix_ns,
        }


@dataclass(frozen=True, slots=True)
class ResidentOwnerEpochTransferReceiptV1:
    """Proof that logical ownership advanced without rebuilding CUDA state."""

    schema_version: str
    physical_owner_identity: str
    prior_owner_identity: str
    successor_owner_identity: str
    boundary_next_absolute_tick: int
    successor_span_end_tick: int
    physical_owner_reused: bool = True
    cuda_allocation_recreated: bool = False
    simultaneous_physical_owner_count: int = 1

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "physical_owner_identity": self.physical_owner_identity,
            "prior_owner_identity": self.prior_owner_identity,
            "successor_owner_identity": self.successor_owner_identity,
            "boundary_next_absolute_tick": self.boundary_next_absolute_tick,
            "successor_span_end_tick": self.successor_span_end_tick,
            "physical_owner_reused": self.physical_owner_reused,
            "cuda_allocation_recreated": self.cuda_allocation_recreated,
            "simultaneous_physical_owner_count": (
                self.simultaneous_physical_owner_count
            ),
        }


def _tick_native_source_prefix_rows_v1(
    prepared: Any,
    *,
    source_projections: Sequence[Mapping[str, Any]] | None,
    cell_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Build one bounded, tensor-free source prefix per resident cell."""

    # Native mechanics bindings return a small engine-owned adapter whose
    # ``physical_window_arguments`` is the actual WDL/Jungle/EED prepared
    # owner.  The adapter deliberately exposes only descriptors and does not
    # duplicate the prepared owner's metadata properties.  Unwrap that
    # boundary here; otherwise a real streamed native owner is incorrectly
    # reported as missing source metadata before its first tick.
    prepared_view = getattr(prepared, "physical_window_arguments", prepared)
    cells = len(tuple(cell_ids))
    metadata_rows = getattr(prepared_view, "metadata_rows", ())
    if not metadata_rows:
        metadata_rows = getattr(prepared, "metadata_rows", ())
    runtime_rows = getattr(prepared_view, "runtime_prefix_rows", ())
    if not runtime_rows:
        runtime_rows = getattr(prepared, "runtime_prefix_rows", ())
    prepared_sources = tuple(
        getattr(prepared_view, "source_projections", ()) or ()
    )
    if not prepared_sources:
        prepared_sources = tuple(
            getattr(prepared, "source_projections", ()) or ()
        )
    if len(metadata_rows) not in {0, cells}:
        raise CampaignProviderError(
            "prepared tick-native metadata rows do not cover the physical batch"
        )
    if len(runtime_rows) not in {0, cells}:
        raise CampaignProviderError(
            "prepared tick-native runtime prefix rows do not cover the physical batch"
        )

    rows: list[dict[str, Any]] = []
    for cell in range(cells):
        projection: Mapping[str, Any] | None = None
        if source_projections is not None and len(source_projections) == cells:
            candidate = source_projections[cell]
            if isinstance(candidate, Mapping):
                projection = candidate
        if projection is None and len(prepared_sources) == cells:
            candidate = prepared_sources[cell]
            if isinstance(candidate, Mapping):
                projection = candidate
        if projection is None and cell == 0:
            candidate = getattr(prepared_view, "initial_projection", None)
            if not isinstance(candidate, Mapping):
                candidate = getattr(prepared, "initial_projection", None)
            if isinstance(candidate, Mapping):
                projection = candidate

        if len(metadata_rows) == cells:
            raw_metadata = metadata_rows[cell]
        elif projection is not None:
            raw_metadata = projection.get("metadata")
        else:
            raw_metadata = None
        if not isinstance(raw_metadata, Mapping):
            raise CampaignProviderError(
                f"prepared tick-native source metadata is missing for cell {cell}"
            )
        metadata = copy.deepcopy(dict(raw_metadata))

        if len(runtime_rows) == cells:
            raw_runtime = runtime_rows[cell]
        elif projection is not None:
            raw_runtime = {
                name: projection.get(name)
                for name in _TICK_NATIVE_RUNTIME_PREFIX_FIELDS_V1
                if name in projection
            }
            active = int(
                metadata.get(
                    "event_tape_active_count",
                    metadata.get("typed_lifecycle_event_active_count", 0),
                )
                or 0
            )
            raw_runtime.setdefault("event_cursor", active)
            raw_runtime.setdefault("runtime_event_cursor", active)
        else:
            raw_runtime = {}
        if not isinstance(raw_runtime, Mapping):
            raise CampaignProviderError(
                f"prepared tick-native runtime prefix is missing for cell {cell}"
            )
        runtime: dict[str, Any] = {}
        active = int(
            metadata.get(
                "event_tape_active_count",
                metadata.get("typed_lifecycle_event_active_count", 0),
            )
            or 0
        )
        defaults: dict[str, Any] = {
            "restore_count": 0,
            "generation_calls": 0,
            "event_cursor": active,
            "runtime_event_cursor": active,
            "last_applied_actions": 0,
            "last_moved": 0,
            "last_gathered": 0,
            "last_policy_refill_calls": 0,
            "last_policy_refill_rows": 0,
            "pending_generation_free_agent_ids": [],
            "pending_generation_free_slots": [],
            "last_world": "none",
        }
        for name in _TICK_NATIVE_RUNTIME_PREFIX_FIELDS_V1:
            value = raw_runtime.get(name, defaults[name])
            if hasattr(value, "device") or hasattr(value, "detach"):
                raise CampaignProviderError(
                    "tick-native source prefix retained a tensor-bearing field: "
                    f"{name}"
                )
            if name in {
                "pending_generation_free_agent_ids",
                "pending_generation_free_slots",
            }:
                if not isinstance(value, (list, tuple)) or any(
                    type(item) is not int for item in value
                ):
                    raise CampaignProviderError(
                        f"tick-native source prefix {name} is malformed"
                    )
                runtime[name] = list(value)
            else:
                runtime[name] = copy.deepcopy(value)
        rows.append({"metadata": metadata, "runtime": runtime})
    return tuple(rows)


def _tick_native_runtime_bookkeeping_v1(
    direct_result: Mapping[str, Any], *, cells: int, span_ticks: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Reduce only the five-column scalar telemetry needed for continuation."""

    totals_tensor = direct_result.get("runtime_bookkeeping_totals")
    last_tensor = direct_result.get("runtime_bookkeeping_last")
    if totals_tensor is not None or last_tensor is not None:
        if not hasattr(totals_tensor, "shape") or not hasattr(last_tensor, "shape"):
            raise CampaignProviderError(
                "tick-native direct result has incomplete bounded bookkeeping"
            )
        if tuple(int(value) for value in totals_tensor.shape) != (cells, 5) or tuple(
            int(value) for value in last_tensor.shape
        ) != (cells, 5):
            raise CampaignProviderError(
                "tick-native bounded bookkeeping shape differs from Bx5"
            )
        tensors = (totals_tensor, last_tensor)
        if any(
            getattr(value, "dtype", None) is not None
            and str(value.dtype) != "torch.int64"
            for value in tensors
        ):
            raise CampaignProviderError(
                "tick-native bounded bookkeeping must use int64 counters"
            )
        totals = totals_tensor.detach().cpu().tolist()
        last = last_tensor.detach().cpu().tolist()
    else:
        runtime = direct_result.get("runtime_bookkeeping_by_tick")
        if not hasattr(runtime, "shape") or not hasattr(runtime, "sum"):
            raise CampaignProviderError(
                "tick-native direct result lacks runtime bookkeeping telemetry"
            )
        shape = tuple(int(value) for value in runtime.shape)
        if shape != (cells, span_ticks, 5):
            raise CampaignProviderError(
                "tick-native runtime bookkeeping shape differs from its dispatch span: "
                f"expected={(cells, span_ticks, 5)} actual={shape}"
            )
        if getattr(runtime, "dtype", None) is not None and str(runtime.dtype) != "torch.int64":
            raise CampaignProviderError(
                "tick-native runtime bookkeeping must use int64 counters"
            )
        totals = runtime.sum(dim=1).detach().cpu().tolist()
        last = runtime[:, -1, :].detach().cpu().tolist()
    normalized_totals: list[tuple[int, ...]] = []
    normalized_last: list[tuple[int, ...]] = []
    for cell in range(cells):
        total_row = tuple(int(value) for value in totals[cell])
        last_row = tuple(int(value) for value in last[cell])
        if any(value < 0 for value in total_row + last_row):
            raise CampaignProviderError(
                "tick-native runtime bookkeeping counters must be nonnegative"
            )
        normalized_totals.append(total_row)
        normalized_last.append(last_row)
    return tuple(normalized_totals), tuple(normalized_last)


def _advance_tick_native_source_prefix_v1(
    owner: ResidentOwnerSpanV1,
    *,
    direct_result: Mapping[str, Any],
    start_next_tick: int,
    end_next_tick: int,
    interval_ticks: int,
) -> None:
    """Advance compact source authority after one resident dispatch."""

    if len(owner.tick_native_prefix_by_cell) != len(owner.cell_ids):
        raise CampaignProviderError(
            "resident owner lacks one compact tick-native prefix per cell"
        )
    span_ticks = end_next_tick - start_next_tick
    runtime = direct_result.get("runtime_bookkeeping_by_tick")
    bounded_runtime = direct_result.get("runtime_bookkeeping_totals")
    if (
        not hasattr(runtime, "shape")
        or not hasattr(runtime, "sum")
    ) and not hasattr(
        bounded_runtime, "shape"
    ) and direct_result.get("absolute_tick_span_native_owner") is not True:
        return
    totals, last = _tick_native_runtime_bookkeeping_v1(
        direct_result,
        cells=len(owner.cell_ids),
        span_ticks=span_ticks,
    )
    frontiers = direct_result.get("event_history_frontier_by_cell")
    if (
        not isinstance(frontiers, Sequence)
        or isinstance(frontiers, (str, bytes, bytearray))
        or len(frontiers) != len(owner.cell_ids)
        or any(type(value) is not int or value < 0 for value in frontiers)
    ):
        raise CampaignProviderError(
            "tick-native direct result lacks exact event-history frontiers"
        )
    for cell, prefix in enumerate(owner.tick_native_prefix_by_cell):
        metadata = prefix["metadata"]
        runtime = prefix["runtime"]
        previous_frontier = metadata.get("event_history_frontier")
        if type(previous_frontier) is not int or previous_frontier < 0:
            raise CampaignProviderError(
                "tick-native source prefix lacks an exact event-history frontier"
            )
        frontier = int(frontiers[cell])
        if frontier < previous_frontier:
            raise CampaignProviderError(
                "tick-native event-history frontier moved backwards"
            )
        appended_events = frontier - previous_frontier
        runtime["generation_calls"] = int(runtime["generation_calls"]) + span_ticks
        runtime["runtime_event_cursor"] = (
            int(runtime["runtime_event_cursor"])
            + totals[cell][0]
            + appended_events
        )
        runtime["last_applied_actions"] = last[cell][0]
        runtime["last_moved"] = last[cell][1]
        runtime["last_gathered"] = last[cell][2]
        runtime["last_policy_refill_calls"] = last[cell][3]
        runtime["last_policy_refill_rows"] = last[cell][4]
        runtime["event_cursor"] = 0
        runtime["last_world"] = "patch"
        metadata["generation"] = end_next_tick // interval_ticks
        metadata["tick"] = end_next_tick % interval_ticks
        metadata["agent_grid_runner_phase"] = (
            "generation_boundary"
            if metadata["tick"] == 0
            else "partial_generation"
        )
        metadata["prediction_tick"] = metadata["tick"]
        metadata["event_history_frontier"] = frontier
        if metadata.get("event_history_storage_mode") == "eiel_resident_event_sink_v1":
            # The external sink owns all rows below the new frontier.  The
            # compact checkpoint projection therefore has an empty pending
            # window whose sequence base starts at that frontier; leaving the
            # old base in place would make next_sequence != base + active_count.
            metadata["typed_lifecycle_event_sequence_base"] = frontier
        metadata["typed_lifecycle_event_next_sequence"] = frontier
        metadata["typed_lifecycle_event_active_count"] = 0
        metadata["event_tape_active_count"] = 0
        metadata["event_tape_cursor"] = 0


def _compact_segment_settlement_v1(
    *,
    cell_ids: tuple[str, ...],
    start_next_tick: int,
    end_next_tick: int,
    owner_identity: str,
    direct_result: Mapping[str, Any],
    native_execution_receipt: Mapping[str, Any],
) -> CompactSegmentSettlementV1:
    """Copy only scalar/event-frontier observations; never retain tensors."""

    if (
        type(start_next_tick) is not int
        or type(end_next_tick) is not int
        or not 0 <= start_next_tick < end_next_tick
        or type(owner_identity) is not str
        or not owner_identity
    ):
        raise CampaignProviderError(
            "compact segment cursor or owner identity is invalid"
        )
    raw_rows = direct_result.get("compact_result_rows", ())
    compact_rows: list[dict[str, Any]] = []
    if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise CampaignProviderError(
                    "compact native metric row is not a mapping"
                )
            if any(
                hasattr(value, "detach") or hasattr(value, "device")
                for value in raw_row.values()
            ):
                raise CampaignProviderError(
                    "compact native metric row retains a tensor-bearing value"
                )
            compact_rows.append(dict(raw_row))
    references = direct_result.get("event_history_prefix_references_by_cell", ())
    event_references = tuple(
        dict(reference)
        for reference in references
        if isinstance(reference, Mapping)
    )
    drain_receipts = direct_result.get("event_drain_receipts", ())
    compact_drains = tuple(
        dict(receipt)
        for receipt in drain_receipts
        if isinstance(receipt, Mapping)
    )

    def exact_nonnegative_vector(name: str) -> tuple[int, ...]:
        values = direct_result.get(name, ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return ()
        return tuple(
            int(value)
            for value in values
            if type(value) is int and value >= 0
        )

    return CompactSegmentSettlementV1(
        schema_version=COMPACT_SEGMENT_SETTLEMENT_SCHEMA_VERSION_V1,
        cell_ids=cell_ids,
        start_next_tick=start_next_tick,
        end_next_tick=end_next_tick,
        owner_identity=owner_identity,
        compact_result_rows=tuple(compact_rows),
        event_history_reference_by_cell=event_references,
        event_drain_receipts=compact_drains,
        event_frontier_by_cell=exact_nonnegative_vector(
            "event_history_frontier_by_cell"
        ),
        event_pending_rows_by_cell=exact_nonnegative_vector(
            "event_pending_rows_by_cell"
        ),
        native_execution_receipt=dict(native_execution_receipt),
    )


def _validated_physical_compact_result_rows_v1(
    receipt: Mapping[str, Any], *, cell_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """Require one tensor-free terminal metric row per physical world."""

    raw_rows = receipt.get("physical_compact_result_rows")
    if (
        not isinstance(raw_rows, Sequence)
        or isinstance(raw_rows, (str, bytes, bytearray))
        or len(raw_rows) != len(cell_ids)
    ):
        raise CampaignProviderError(
            "physical world checkpoint metrics do not cover every cell"
        )
    rows: list[dict[str, Any]] = []
    for expected_cell_id, raw_row in zip(cell_ids, raw_rows, strict=True):
        if (
            not isinstance(raw_row, Mapping)
            or raw_row.get("cell_id") != expected_cell_id
            or not isinstance(raw_row.get("metrics"), Mapping)
            or any(
                hasattr(value, "detach") or hasattr(value, "device")
                for value in raw_row.values()
            )
        ):
            raise CampaignProviderError(
                "physical world checkpoint metric authority differs"
            )
        rows.append(dict(raw_row))
    return rows


def _release_resident_owner_v1(owner: ResidentOwnerSpanV1) -> OwnerReleaseReceiptV1:
    """Synchronize, drop owner references, and measure the released boundary."""

    if not isinstance(owner, ResidentOwnerSpanV1) or owner.closed:
        raise CampaignProviderError("resident owner release is not open")
    devices = {
        str(getattr(request, "cuda_device", ""))
        for request in owner.requests
        if isinstance(getattr(request, "cuda_device", None), str)
        and str(getattr(request, "cuda_device", "")).startswith("cuda")
    }
    # Clear the owner-held references before collection.  The caller clears
    # its local aliases before invoking this function, so a successor cannot
    # overlap the old CUDA allocation.
    owner.prepared_owner = None
    owner.builds = ()
    owner.requests = ()
    gc.collect()
    synchronized = True
    allocated = 0
    reserved = 0
    try:
        import torch

        if torch.cuda.is_available():
            for device in sorted(devices):
                torch.cuda.synchronize(device)
            for device in sorted(devices):
                torch.cuda.empty_cache()
                allocated = max(allocated, int(torch.cuda.memory_allocated(device)))
                reserved = max(reserved, int(torch.cuda.memory_reserved(device)))
    except (ImportError, RuntimeError, ValueError):
        synchronized = False
        raise
    released_perf_ns = time.perf_counter_ns()
    released_unix_ns = time.time_ns()
    owner.closed = True
    return OwnerReleaseReceiptV1(
        schema_version=OWNER_RELEASE_RECEIPT_SCHEMA_VERSION_V1,
        owner_identity=owner.prepared_owner_identity,
        final_next_absolute_tick=owner.next_absolute_tick,
        release_synchronization_complete=synchronized,
        live_alias_count=0,
        post_release_cuda_allocated_bytes=allocated,
        post_release_cuda_reserved_bytes=reserved,
        released_perf_ns=released_perf_ns,
        released_unix_ns=released_unix_ns,
    )


def _reset_terminal_cuda_primary_context_v1(
    *,
    execution_device: str,
    owner_release_receipt: OwnerReleaseReceiptV1,
    terminal_capture: Mapping[str, Any],
    live_owner_count: int,
) -> dict[str, Any]:
    """Destroy the terminal CUDA context before detached host persistence waits."""

    if (
        not isinstance(execution_device, str)
        or not execution_device.startswith("cuda")
        or not isinstance(owner_release_receipt, OwnerReleaseReceiptV1)
        or not owner_release_receipt.release_synchronization_complete
        or owner_release_receipt.live_alias_count != 0
        or type(owner_release_receipt.post_release_cuda_allocated_bytes) is not int
        or owner_release_receipt.post_release_cuda_allocated_bytes < 0
        or type(owner_release_receipt.post_release_cuda_reserved_bytes) is not int
        or owner_release_receipt.post_release_cuda_reserved_bytes < 0
        or type(live_owner_count) is not int
        or live_owner_count != 0
    ):
        raise CampaignProviderError(
            "terminal CUDA primary-context reset lacks an alias-free owner release"
        )
    try:
        from engine.campaigns.acs_native_detached_slots_v1 import (
            validate_detached_capture_host_only_v1,
        )

        detached_capture_authority = validate_detached_capture_host_only_v1(
            terminal_capture
        )
    except (ImportError, RuntimeError) as exc:
        raise CampaignProviderError(
            "terminal CUDA primary-context reset lacks a host-only capture"
        ) from exc
    try:
        import ctypes
        import ctypes.util

        import torch

        device = torch.device(execution_device)
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if torch.cuda.current_device() != device_index:
            torch.cuda.set_device(device_index)
        runtime_library = ctypes.util.find_library("cudart")
        if runtime_library is None:
            raise RuntimeError("CUDA runtime library is unavailable")
        reset = ctypes.CDLL(runtime_library).cudaDeviceReset
        reset.argtypes = []
        reset.restype = ctypes.c_int
        reset_result = int(reset())
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
        raise CampaignProviderError(
            "terminal CUDA primary-context reset failed"
        ) from exc
    if reset_result != 0:
        raise CampaignProviderError(
            "terminal CUDA primary-context reset returned an error"
        )
    released_perf_ns = time.perf_counter_ns()
    released_unix_ns = time.time_ns()
    return {
        "schema_version": "eiel_terminal_cuda_primary_context_release_v1",
        "execution_device": execution_device,
        "cuda_device_reset_result": reset_result,
        "pre_reset_cuda_allocator_residue": {
            "allocated_bytes": (
                owner_release_receipt.post_release_cuda_allocated_bytes
            ),
            "reserved_bytes": (
                owner_release_receipt.post_release_cuda_reserved_bytes
            ),
        },
        "live_owner_count_before_reset": live_owner_count,
        "detached_capture_authority": detached_capture_authority,
        "post_reset_allocator_observation": (
            "not_queried_to_avoid_cuda_primary_context_reinitialization"
        ),
        "owner_release_receipt": owner_release_receipt.payload(),
        "released_perf_ns": released_perf_ns,
        "released_unix_ns": released_unix_ns,
    }


def _transfer_resident_owner_epoch_v1(
    owner: ResidentOwnerSpanV1,
    *,
    successor_owner_identity: str,
    successor_span_end_tick: int,
) -> ResidentOwnerEpochTransferReceiptV1:
    """Advance logical owner authority while retaining one physical CUDA owner.

    Tick-native owner spans are governance and cursor epochs, not a reason to
    serialize state through a checkpoint.  The prepared owner and every CUDA
    allocation stay in place; only the bounded logical identity and span move
    forward.  Durable checkpoints remain independent explicit boundaries.
    """

    if not isinstance(owner, ResidentOwnerSpanV1) or owner.closed:
        raise CampaignProviderError("resident owner epoch transfer is not open")
    boundary = owner.next_absolute_tick
    if boundary != owner.end_next_tick:
        raise CampaignProviderError(
            "resident owner epoch transfer requires its exact terminal cursor"
        )
    if (
        type(successor_owner_identity) is not str
        or not successor_owner_identity
        or successor_owner_identity == owner.prepared_owner_identity
    ):
        raise CampaignProviderError(
            "resident owner epoch transfer requires a distinct successor identity"
        )
    if (
        type(successor_span_end_tick) is not int
        or successor_span_end_tick <= boundary
    ):
        raise CampaignProviderError(
            "resident owner epoch transfer successor span is invalid"
        )
    if owner.prepared_owner is None or not owner.builds or not owner.requests:
        raise CampaignProviderError(
            "resident owner epoch transfer lost its physical prepared owner"
        )
    prior_identity = owner.prepared_owner_identity
    receipt = ResidentOwnerEpochTransferReceiptV1(
        schema_version=OWNER_EPOCH_TRANSFER_RECEIPT_SCHEMA_VERSION_V1,
        physical_owner_identity=owner.physical_owner_identity,
        prior_owner_identity=prior_identity,
        successor_owner_identity=successor_owner_identity,
        boundary_next_absolute_tick=boundary,
        successor_span_end_tick=successor_span_end_tick,
    )
    owner.start_next_tick = boundary
    owner.end_next_tick = successor_span_end_tick
    owner.prepared_owner_identity = successor_owner_identity
    return receipt


def _synchronize_detached_acs_device_boundary_v1(
    execution_device: str | None,
) -> None:
    """Fence all CUDA work before a detached cursor releases its aliases.

    The native controller synchronizes its execution stream, but the
    WDL/Jungle/EED cursor adapters may perform CUDA-backed scalar readbacks on
    another current stream while composing the bounded projection.  A
    detached checkpoint must not drop the projection/result aliases until all
    device work that can still reference the resident owner has completed.
    """

    if not isinstance(execution_device, str) or not execution_device.startswith(
        "cuda"
    ):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(execution_device)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise CampaignProviderError(
            "detached ACS device boundary synchronization failed"
        ) from exc


def _resident_owner_identity_v1(
    *,
    batch_id: str,
    cell_ids: Sequence[str],
    start_next_tick: int,
    end_next_tick: int,
    execution_route_sha256: str | None,
) -> str:
    payload = {
        "schema_version": RESIDENT_OWNER_SPAN_SCHEMA_VERSION_V1,
        "batch_id": batch_id,
        "cell_ids": list(cell_ids),
        "start_next_tick": start_next_tick,
        "end_next_tick": end_next_tick,
        "execution_route_sha256": execution_route_sha256,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _mark_native_streamed_capture_owner_v1(prepared: Any) -> None:
    """Bind native capture to the owner consumed by the selected projector."""

    # Native mechanics bindings wrap the actual prepared owner in
    # PreparedNativeMechanicsOwnerV1.  The projector unwraps that object
    # through physical_window_arguments before reaching the direct-resident
    # Patch/WDL/Jungle owner, so bind the capture mode on that same authority.
    native_capture_owner = getattr(
        prepared, "physical_window_arguments", prepared
    )
    native_capture_owner = getattr(
        native_capture_owner, "patch", native_capture_owner
    )
    native_capture_inputs = getattr(
        native_capture_owner, "checkpoint_projection_inputs", None
    )
    if not isinstance(native_capture_inputs, dict):
        raise CampaignProviderError(
            "native streamed checkpoint owner lacks mutable "
            "checkpoint projection inputs"
        )
    native_capture_inputs["native_streamed_capture"] = True


def _aggregate_native_cuda_source_field_counts_v1(
    source_projections: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Count CUDA-backed fields retained by an aggregate snapshot.

    This is telemetry only.  The canonical checkpoint projector deliberately
    materializes its bounded serialization snapshot on the host, so a zero
    count is valid for a checkpoint projection and does not mean that the
    resident execution owner used CPU request staging.  Route admission for
    that separate concern is enforced by the explicit staging parameter and
    the resident source allocator/native binding.
    """

    counts: list[int] = []
    for index, projection in enumerate(source_projections):
        tensors = projection.get("tensors") if isinstance(projection, Mapping) else None
        if not isinstance(tensors, Mapping):
            raise CampaignProviderError(
                "ACS aggregate owner snapshot lacks its tensor mapping: "
                f"cell_index={index}"
            )
        counts.append(
            sum(
                1
                for tensor in tensors.values()
                if getattr(getattr(tensor, "device", None), "type", None) == "cuda"
            )
        )
    return tuple(counts)


def _validate_aggregate_native_production_owner_route_v1(
    request: Any,
    source_projections: Sequence[Mapping[str, Any]],
) -> tuple[int, ...]:
    """Reject aggregate routes that still enable CPU request staging.

    ``source_projections`` are canonical checkpoint snapshots, not the live
    resident execution owner.  They may therefore be host-materialized by
    the checkpoint contract; demanding CUDA devices here would incorrectly
    reject every valid bounded checkpoint capture.
    """

    counts = _aggregate_native_cuda_source_field_counts_v1(source_projections)
    if getattr(request, "profile_id", None) != PRODUCTION_PROFILE_ID:
        return counts
    parameters = dict(getattr(request, "parameter_arrays", ()))
    staging_values = parameters.get("cuda_architecture_b.cpu_request_staging_enable")
    if not isinstance(staging_values, tuple) or len(staging_values) != len(
        getattr(request, "cell_ids", ())
    ) or any(value is not False for value in staging_values):
        raise CampaignProviderError(
            "ACS aggregate production capture requires explicit "
            "cuda_architecture_b.cpu_request_staging_enable=false for every cell"
        )
    return counts

class _LazyProjectionBundleSequence(Sequence[Mapping[str, Any]]):
    """Load one durable cell projection at a time during owner preparation."""

    __slots__ = (
        "_cache",
        "_cache_results",
        "_loader",
        "_paths",
        "_source_projection_digests",
    )

    def __init__(
        self,
        paths: Sequence[str],
        loader: Callable[[str], Mapping[str, Any]],
        source_projection_digests: Mapping[str, str] | None = None,
        *,
        cache_results: bool = True,
    ) -> None:
        self._paths = tuple(str(path) for path in paths)
        self._loader = loader
        self._cache: dict[int, Mapping[str, Any]] = {}
        self._cache_results = bool(cache_results)
        self._source_projection_digests = {
            str(path): str(digest)
            for path, digest in (source_projection_digests or {}).items()
        }

    @property
    def descriptor_paths(self) -> tuple[str, ...]:
        """Expose durable paths without decoding or caching their payloads."""

        return self._paths

    @property
    def descriptor_digests(self) -> Mapping[str, str]:
        return dict(self._source_projection_digests)

    @property
    def world_sharded_sources(self) -> bool:
        """Whether every descriptor is one independent physical-world object."""

        return bool(self._paths) and all(
            Path(path).suffix == ".acsworld" for path in self._paths
        )

    def load_uncached(self, index: int) -> Mapping[str, Any]:
        """Consume one projection without retaining a batch-sized cache."""

        if type(index) is not int:
            raise TypeError("projection bundle index must be an integer")
        position = index if index >= 0 else len(self._paths) + index
        if position < 0 or position >= len(self._paths):
            raise IndexError("projection bundle index is out of range")
        cached = self._cache.pop(position, None)
        if cached is not None:
            return cached
        projection = self._loader(self._paths[position])
        if not isinstance(projection, Mapping):
            raise CampaignProviderError(
                "lazy checkpoint projection loader returned a non-mapping"
            )
        return projection

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, Any] | tuple[Mapping[str, Any], ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        if type(index) is not int:
            raise TypeError("projection bundle index must be an integer")
        position = index if index >= 0 else len(self._paths) + index
        if position < 0 or position >= len(self._paths):
            raise IndexError("projection bundle index is out of range")
        cached = self._cache.get(position)
        if cached is not None:
            return cached
        projection = self._loader(self._paths[position])
        if not isinstance(projection, Mapping):
            raise CampaignProviderError(
                "lazy checkpoint projection loader returned a non-mapping"
            )
        if self._cache_results:
            self._cache[position] = projection
        return projection

def _aggregate_file_fingerprint(path: str) -> tuple[int, int, int, int, int]:
    try:
        stat = Path(path).stat()
    except OSError as exc:
        raise CampaignProviderError(
            f"aggregate checkpoint stat failed during lazy restore: {path}: {exc}"
        ) from exc
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


class _LazyAggregateProjectionBundleSequence(Sequence[Mapping[str, Any]]):
    """Load cell-indexed projections from one aggregate checkpoint file."""

    __slots__ = (
        "_cell_ids",
        "_loader",
        "_paths",
        "_source_projection_digests",
        "_verified_paths",
    )

    def __init__(
        self,
        paths: Sequence[str],
        cell_ids: Sequence[str],
        loader: Callable[..., Mapping[str, Any]],
        source_projection_digests: Mapping[str, str] | None = None,
    ) -> None:
        self._paths = tuple(str(path) for path in paths)
        if len(cell_ids) != len(self._paths):
            raise CampaignProviderError(
                "aggregate checkpoint path and cell identity counts differ"
            )
        self._cell_ids = tuple(str(cell_id) for cell_id in cell_ids)
        self._loader = loader
        supplied_digests = dict(source_projection_digests or {})
        path_counts: dict[str, int] = {}
        for path in self._paths:
            path_counts[path] = path_counts.get(path, 0) + 1
        self._source_projection_digests: dict[str, str] = {}
        for cell_id, path in zip(self._cell_ids, self._paths, strict=True):
            digest = supplied_digests.get(cell_id)
            if digest is None and path_counts[path] == 1:
                digest = supplied_digests.get(path)
            if digest is not None:
                if not _valid_sha256(digest):
                    raise CampaignProviderError(
                        "aggregate checkpoint source projection digest is invalid"
                    )
                self._source_projection_digests[cell_id] = str(digest)
        self._verified_paths: dict[str, tuple[int, int, int, int, int]] = {}

    @property
    def descriptor_paths(self) -> tuple[str, ...]:
        """Expose the shared durable aggregate path without loading a cell."""

        return self._paths

    @property
    def descriptor_digests_by_cell(self) -> Mapping[str, str]:
        """Return exact per-cell digests even when every path is identical."""

        return dict(self._source_projection_digests)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, Any] | tuple[Mapping[str, Any], ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        if type(index) is not int:
            raise TypeError("aggregate projection bundle index must be an integer")
        position = index if index >= 0 else len(self._paths) + index
        if position < 0 or position >= len(self._paths):
            raise IndexError("aggregate projection bundle index is out of range")
        path = self._paths[position]
        fingerprint = _aggregate_file_fingerprint(path)
        verify_payload = self._verified_paths.get(path) != fingerprint
        projection = self._loader(
            path,
            cell_id=self._cell_ids[position],
            verify_aggregate_payload=verify_payload,
        )
        fingerprint_after = _aggregate_file_fingerprint(path)
        if fingerprint_after != fingerprint:
            raise CampaignProviderError(
                f"aggregate checkpoint changed during lazy restore: {path}"
            )
        if verify_payload:
            self._verified_paths[path] = fingerprint_after
        if not isinstance(projection, Mapping):
            raise CampaignProviderError(
                "lazy aggregate checkpoint projection loader returned a non-mapping"
            )
        return projection


class _LazyDetachedSpoolProjectionSequence(Sequence[Mapping[str, Any]]):
    """Load one successor cell at a time from an immutable ACS producer spool."""

    __slots__ = ("_capture", "_cell_ids", "_source_projection_digests")

    def __init__(self, capture: Mapping[str, Any]) -> None:
        from engine.campaigns.acs_native_detached_slots_v1 import (
            _validate_spool_descriptor_v1,
        )

        _, header, _ = _validate_spool_descriptor_v1(capture)
        self._capture = dict(capture)
        self._cell_ids = tuple(str(value) for value in header["cell_ids"])
        digests = capture.get("source_projection_sha256s")
        if not isinstance(digests, Mapping) or set(digests) != set(self._cell_ids):
            raise CampaignProviderError(
                "detached spool continuation lacks exact per-cell digests"
            )
        self._source_projection_digests = {
            cell_id: str(digests[cell_id]) for cell_id in self._cell_ids
        }

    @property
    def descriptor_capture(self) -> Mapping[str, Any]:
        return dict(self._capture)

    @property
    def descriptor_digests_by_cell(self) -> Mapping[str, str]:
        return dict(self._source_projection_digests)

    @property
    def ordered_cell_ids(self) -> tuple[str, ...]:
        return self._cell_ids

    def __len__(self) -> int:
        return len(self._cell_ids)

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, Any] | tuple[Mapping[str, Any], ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        if type(index) is not int:
            raise TypeError("detached spool index must be an integer")
        position = index if index >= 0 else len(self._cell_ids) + index
        if position < 0 or position >= len(self._cell_ids):
            raise IndexError("detached spool index is out of range")
        cell_id = self._cell_ids[position]
        from engine.campaigns.acs_native_detached_slots_v1 import (
            load_acs_native_detached_spool_cell_v1,
        )

        restored = load_acs_native_detached_spool_cell_v1(
            self._capture,
            cell_id=cell_id,
            expected_source_projection_sha256=self._source_projection_digests[
                cell_id
            ],
        )
        projection = restored.get("projection")
        if not isinstance(projection, Mapping):
            raise CampaignProviderError(
                "detached spool continuation returned no projection"
            )
        return projection


def _resident_owner_compact_source_projections_v1(
    prepared: Any,
    *,
    bounded_streamed_preparation: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Return only producer-owned compact views safe to revisit per subspan.

    Streamed preparation consumes its durable source descriptors exactly once
    while constructing the resident CUDA owner. ``prepared.source_projections``
    may still name that consumed descriptor sequence; iterating it again after
    the ACS source lease is released would reopen an already-settled spool and
    would reconstruct a full host projection bank. The resident owner and its
    bounded terminal cursor are authoritative after import, so the streamed
    route must never revisit those input descriptors.
    """

    if bounded_streamed_preparation:
        return ()
    return tuple(getattr(prepared, "source_projections", ()))


class _BoundedSourceProjectionStoreV1:
    """Durable one-cell source descriptors for tick-native owner preparation."""

    __slots__ = (
        "_provider",
        "_root",
        "_cell_ids",
        "_descriptors",
        "_aggregate_sequence",
        "_detached_spool_sequence",
        "_indexed_source_capture",
    )

    def __init__(
        self,
        provider: Any,
        *,
        root: Path,
        cell_ids: Sequence[str],
        source_projections: Sequence[Mapping[str, Any]] | None,
        indexed_source_capture: bool = False,
    ) -> None:
        self._provider = provider
        self._root = root.resolve()
        self._cell_ids = tuple(cell_ids)
        if type(indexed_source_capture) is not bool:
            raise CampaignProviderError(
                "indexed source capture selection must be an exact bool"
            )
        self._indexed_source_capture = indexed_source_capture
        if not self._cell_ids or len(set(self._cell_ids)) != len(self._cell_ids):
            raise CampaignProviderError(
                "bounded source store requires unique physical cell IDs"
            )
        self._descriptors: list[dict[str, Any]] = []
        self._aggregate_sequence: (
            _LazyAggregateProjectionBundleSequence | None
        ) = None
        self._detached_spool_sequence: (
            _LazyDetachedSpoolProjectionSequence | None
        ) = None
        if source_projections is not None:
            self._bind_existing(source_projections)

    @property
    def has_existing_sources(self) -> bool:
        return bool(self._descriptors)

    @property
    def source_projection_digests(self) -> dict[str, str]:
        return {
            cell_id: descriptor["source_projection_sha256"]
            for cell_id, descriptor in zip(
                self._cell_ids, self._descriptors, strict=True
            )
            if isinstance(descriptor.get("source_projection_sha256"), str)
        }

    def detached_descriptor_recipe_v1(self) -> dict[str, Any]:
        """Export the immutable source set without retaining its loader owner.

        A physical-batch checkpoint worker must be able to reload the
        canonical per-cell skeleton after the resident CUDA owner resumes.
        The worker is deliberately not allowed to retain this store, its
        provider, or the cursor-factory closure.  Export only the already
        durable file/index descriptors and their exact cell identities.
        """

        if len(self._descriptors) != len(self._cell_ids):
            raise CampaignProviderError(
                "bounded source descriptor recipe requested before every source was captured"
            )
        descriptors = tuple(copy.deepcopy(row) for row in self._descriptors)
        if any(row.get("kind") == "detached_spool_index" for row in descriptors):
            raise CampaignProviderError(
                "physical-batch settlement cannot depend on a legacy detached cursor spool"
            )
        recipe = {
            "schema_version": "eiel_bounded_source_descriptor_recipe_v1",
            "cell_ids": list(self._cell_ids),
            "descriptors": list(descriptors),
            "source_projection_sha256s": self.source_projection_digests,
        }
        try:
            json.dumps(
                recipe,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise CampaignProviderError(
                "bounded source descriptor recipe is not deterministic JSON"
            ) from exc
        return recipe

    def _bind_existing(
        self, source_projections: Sequence[Mapping[str, Any]]
    ) -> None:
        if isinstance(source_projections, _LazyDetachedSpoolProjectionSequence):
            if len(source_projections) != len(self._cell_ids):
                raise CampaignProviderError(
                    "detached spool sources do not cover the physical batch"
                )
            digests = dict(source_projections.descriptor_digests_by_cell)
            if set(digests) != set(self._cell_ids):
                raise CampaignProviderError(
                    "detached spool sources lack exact per-cell digests"
                )
            self._detached_spool_sequence = source_projections
            spool_path = str(
                source_projections.descriptor_capture.get("spool_path") or ""
            )
            self._descriptors = [
                {
                    "kind": "detached_spool_index",
                    "path": spool_path,
                    "cell_index": index,
                    "source_projection_sha256": str(digests[cell_id]),
                }
                for index, cell_id in enumerate(self._cell_ids)
            ]
            return
        if isinstance(source_projections, _LazyAggregateProjectionBundleSequence):
            descriptor_paths = source_projections.descriptor_paths
            if len(descriptor_paths) != len(self._cell_ids):
                raise CampaignProviderError(
                    "bounded aggregate source descriptors do not cover the physical batch"
                )
            digests = dict(source_projections.descriptor_digests_by_cell)
            if set(digests) != set(self._cell_ids):
                raise CampaignProviderError(
                    "bounded aggregate source descriptors lack exact per-cell digests"
                )
            self._aggregate_sequence = source_projections
            self._descriptors = [
                {
                    "kind": "aggregate_index",
                    "path": str(path),
                    "cell_index": index,
                    "source_projection_sha256": str(digests[cell_id]),
                }
                for index, (cell_id, path) in enumerate(
                    zip(self._cell_ids, descriptor_paths, strict=True)
                )
            ]
            return
        if (
            isinstance(source_projections, _LazyProjectionBundleSequence)
            and source_projections.world_sharded_sources
        ):
            if len(source_projections) != len(self._cell_ids):
                raise CampaignProviderError(
                    "world-sharded sources do not cover the physical batch"
                )
            # A world object is an independently restorable physical state, not
            # a source component that the next durability worker may embed.
            # Flatten it once, B1-at-a-time, at the restore/source-import
            # boundary.  The next packed capture then exports only component
            # descriptors and its persistence path never reconstructs B1 or
            # depends on the prior checkpoint set.
            for index in range(len(self._cell_ids)):
                projection = source_projections.load_uncached(index)
                self.capture(index, projection)
                del projection
            if (
                len(self._descriptors) != len(self._cell_ids)
                or any(row.get("kind") != "component" for row in self._descriptors)
            ):
                raise CampaignProviderError(
                    "world-sharded source import did not flatten to components"
                )
            return
        descriptor_paths = getattr(source_projections, "descriptor_paths", None)
        if isinstance(descriptor_paths, tuple):
            if len(descriptor_paths) != len(self._cell_ids):
                raise CampaignProviderError(
                    "bounded source descriptor paths do not cover the physical batch"
                )
            digests = dict(
                getattr(source_projections, "descriptor_digests", {}) or {}
            )
            self._descriptors = [
                {
                    "kind": "bundle",
                    "path": str(path),
                    **(
                        {"source_projection_sha256": str(digests[str(path)])}
                        if isinstance(digests.get(str(path)), str)
                        else {}
                    ),
                }
                for path in descriptor_paths
            ]
            return
        if not isinstance(source_projections, Sequence) or isinstance(
            source_projections, (str, bytes, bytearray)
        ):
            raise CampaignProviderError(
                "bounded source input must be a durable descriptor sequence"
            )
        if len(source_projections) != len(self._cell_ids):
            raise CampaignProviderError(
                "bounded source input does not cover the physical batch"
            )
        # This conversion is only a compatibility fence for an already
        # materialized legacy continuation.  New tick-native paths never enter
        # it: they bind bundle descriptors and decode one cell at a time.
        for index, projection in enumerate(source_projections):
            self.capture(index, projection)

    def rebind_detached_spool(self, capture: Mapping[str, Any]) -> None:
        """Roll a resident owner's canonical source cursor to a new spool.

        The caller seals the replacement checkpoint cursor before rebinding,
        so no cursor from the prior descriptor set remains live.  Only the
        immutable file descriptor changes; no projection bank is retained.
        """

        replacement = _LazyDetachedSpoolProjectionSequence(capture)
        if replacement.ordered_cell_ids != self._cell_ids:
            raise CampaignProviderError(
                "detached spool rebind differs from resident physical cells"
            )
        self._descriptors = []
        self._aggregate_sequence = None
        self._detached_spool_sequence = None
        self._bind_existing(replacement)

    def capture(self, index: int, projection: Mapping[str, Any]) -> str:
        if type(index) is not int or not 0 <= index < len(self._cell_ids):
            raise CampaignProviderError("bounded source capture index is invalid")
        if not isinstance(projection, Mapping):
            raise CampaignProviderError("bounded source capture is not a mapping")
        source_digest = self._provider.terminal_projection_digest(projection)
        if index < len(self._descriptors):
            descriptor = self._descriptors[index]
            expected = descriptor.get("source_projection_sha256")
            if expected is not None and expected != source_digest:
                raise CampaignProviderError(
                    "bounded source capture differs from its durable digest"
                )
            descriptor["source_projection_sha256"] = source_digest
            if (
                self._indexed_source_capture
                and descriptor.get("kind") == "component"
                and "indexed_source" not in descriptor
            ):
                from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
                    write_indexed_source_projection_v1,
                )

                component_path = Path(str(descriptor.get("path") or ""))
                descriptor["indexed_source"] = dict(
                    write_indexed_source_projection_v1(
                        projection,
                        component_path.with_name("source_projection.acsindexed"),
                        source_projection_sha256=source_digest,
                    )
                )
            return source_digest
        if index != len(self._descriptors):
            raise CampaignProviderError(
                "bounded source capture arrived out of physical cell order"
            )
        path = (
            self._root
            / "source_import_v1"
            / f"cell-{index:04d}-{self._cell_ids[index]}"
            / "source_projection.pt"
        )
        descriptor = self._provider._write_component_checkpoint(projection, path)
        indexed_source = None
        if self._indexed_source_capture:
            from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
                write_indexed_source_projection_v1,
            )

            indexed_source = write_indexed_source_projection_v1(
                projection,
                path.with_name("source_projection.acsindexed"),
                source_projection_sha256=source_digest,
            )
        self._descriptors.append(
            {
                "kind": "component",
                "path": str(descriptor["path"]),
                "file_sha256": str(descriptor["sha256"]),
                "source_projection_sha256": source_digest,
                **(
                    {"indexed_source": dict(indexed_source)}
                    if indexed_source is not None
                    else {}
                ),
            }
        )
        return source_digest

    @staticmethod
    def _projection_tensor_bytes(projection: Mapping[str, Any]) -> int:
        manifest = projection.get("tensor_manifest")
        if not isinstance(manifest, Mapping):
            return 0
        return sum(
            int(row.get("bytes", 0))
            for row in manifest.values()
            if isinstance(row, Mapping) and type(row.get("bytes")) is int
        )

    def _load(
        self, descriptor: Mapping[str, Any], index: int, cell_id: str
    ) -> Any:
        from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
            DetachedSourcePayloadV1,
            SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1,
        )

        kind = descriptor.get("kind")
        path = str(descriptor.get("path") or "")
        if kind == "detached_spool_index":
            spool_sequence = self._detached_spool_sequence
            descriptor_index = descriptor.get("cell_index")
            if spool_sequence is None or descriptor_index != index:
                raise CampaignProviderError(
                    "bounded detached spool source index is invalid"
                )
            projection = spool_sequence[index]
        elif kind == "aggregate_index":
            aggregate_sequence = self._aggregate_sequence
            descriptor_index = descriptor.get("cell_index")
            if aggregate_sequence is None or descriptor_index != index:
                raise CampaignProviderError(
                    "bounded aggregate source descriptor index is invalid"
                )
            projection = aggregate_sequence[index]
        elif kind == "bundle":
            projection = self._provider._read_projection_bundle(path)
        elif kind == "component":
            projection = self._provider._read_component_checkpoint(
                {
                    "path": path,
                    "sha256": descriptor.get("file_sha256"),
                }
            )
        else:
            raise CampaignProviderError("bounded source descriptor kind is invalid")
        if not isinstance(projection, Mapping):
            raise CampaignProviderError("bounded source loader returned no projection")
        source_digest = self._provider.terminal_projection_digest(projection)
        expected_digest = descriptor.get("source_projection_sha256")
        if expected_digest is not None and source_digest != expected_digest:
            raise CampaignProviderError(
                f"bounded source projection digest differs for cell {cell_id}"
            )
        return DetachedSourcePayloadV1(
            schema_version=SOURCE_IMPORT_CURSOR_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=index,
            payload=projection,
            source_projection_sha256=source_digest,
            live_payload_bytes=self._projection_tensor_bytes(projection),
        )

    def cursor_factory(self) -> Callable[[], Any]:
        from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
            SourceImportCursorV1,
        )

        cell_ids = self._cell_ids

        def factory() -> Any:
            # The streamed owner builder receives this callable while it is
            # still capturing the ordered per-cell source descriptors.  Do
            # not snapshot or validate the descriptor set at callable
            # construction time: Python evaluates the preparer arguments
            # before the builder has requested every cell.  The cursor itself
            # is opened only after capture, at which point this completeness
            # fence is meaningful.
            if len(self._descriptors) != len(cell_ids):
                raise CampaignProviderError(
                    "bounded source cursor requested before every source was captured"
                )
            descriptors = tuple(dict(row) for row in self._descriptors)
            return SourceImportCursorV1.begin(
                descriptors,
                cell_ids,
                loader=self._load,
                importer=(
                    lambda payload, prepared_storage, _index, _cell_id: (
                        prepared_storage
                        if prepared_storage is not None
                        else payload
                    )
                ),
            )

        factory.rebind_detached_spool = self.rebind_detached_spool  # type: ignore[attr-defined]
        factory.detached_descriptor_recipe_v1 = (  # type: ignore[attr-defined]
            self.detached_descriptor_recipe_v1
        )

        return factory


def _lazy_projection_bundle_sequence_v1(
    paths: Sequence[str],
    loader: Callable[..., Mapping[str, Any]],
    *,
    cell_ids: Sequence[str],
    source_projection_digests: Mapping[str, str] | None = None,
) -> Sequence[Mapping[str, Any]]:
    normalized_paths = tuple(str(path) for path in paths)
    if normalized_paths and all(
        Path(path).suffix == ".acsaggregate" for path in normalized_paths
    ):
        return _LazyAggregateProjectionBundleSequence(
            normalized_paths,
            cell_ids,
            loader,
            source_projection_digests=source_projection_digests,
        )
    supplied_digests = dict(source_projection_digests or {})
    path_digests: dict[str, str] = {}
    for cell_id, path in zip(cell_ids, normalized_paths, strict=True):
        digest = supplied_digests.get(str(cell_id), supplied_digests.get(path))
        if isinstance(digest, str):
            path_digests[path] = digest
    return _LazyProjectionBundleSequence(
        normalized_paths,
        loader,
        source_projection_digests=path_digests,
        cache_results=not (
            bool(normalized_paths)
            and all(Path(path).suffix == ".acsworld" for path in normalized_paths)
        ),
    )

def _require_acs_native_nonnegative_int_v1(
    receipt: Mapping[str, Any],
    field: str,
) -> int:
    value = receipt.get(field)
    if type(value) is not int or value < 0:
        raise CampaignProviderError(
            "ACS capture native receipt lacks exact nonnegative " f"{field}"
        )
    return value


PROVIDER_NATIVE_EXPOSURE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_provider_native_exposure_receipt_v1"
)
CHECKPOINT_IDENTITY_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "eiel_provider_checkpoint_identity_contract_v1"
)
OPTION_D_M0_PRIMARY_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_option_d_rng_instrumentation_receipt_v1"
)
OPTION_D_M0_SOURCE_REBIND_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_option_d_m0_source_rebind_v3"
)
OPTION_D_M0_EXPECTED_REFERENCE_PACKAGE_SHA256: Final[str] = (
    "b263fea205b5ec2ae9f071a14340bf6070628bf25633f602677e2bcf771728e5"
)
OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT_NAME: Final[str] = (
    "option_d_m0_source_environment_receipt"
)
OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT_KIND: Final[str] = (
    "option_d_m0_source_environment_receipt"
)
RUNTIME_ARTIFACT_RELOCATION_ROOTS_V1: Final[tuple[tuple[str, str], ...]] = (
    (
        public_runtime_path('evidence_raw/'),
        public_runtime_path('knobs_discovery_evidence_raw/'),
    ),
)

ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1: Final[str] = (
    "acs_native_runtime_extension"
)
ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1: Final[str] = (
    "acs_native_runtime_extension_manifest"
)
ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1: Final[str] = "native_runtime_extension"
ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1: Final[str] = "native_runtime_extension_manifest"
ACS_NATIVE_RUNTIME_ABI_V1: Final[str] = "eiel_native_runtime_abi_v1"
ACS_NATIVE_RUNTIME_VERSION_V1: Final[str] = "eiel_native_runtime_flat_v1"
ACS_NATIVE_RUNTIME_MODULE_NAME_V1: Final[str] = "eiel_native_runtime_ext"
ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1: Final[str] = (
    "EIEL_NATIVE_RUNTIME_EXTENSION"
)


def _runtime_artifact_candidates_v1(declared: Path) -> tuple[Path, ...]:
    rendered = str(declared)
    candidates = [declared]
    for source_root, relocated_root in RUNTIME_ARTIFACT_RELOCATION_ROOTS_V1:
        if rendered.startswith(source_root):
            candidates.append(Path(relocated_root + rendered[len(source_root) :]))
    return tuple(candidates)


def _file_sha256_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_runtime_artifact_path_v1(declared: Path, digest: str) -> Path | None:
    for candidate in _runtime_artifact_candidates_v1(declared):
        if candidate.is_symlink():
            continue
        path = candidate.resolve()
        if path.is_file() and _file_sha256_v1(path) == digest:
            return path
    return None


def _acs_native_runtime_artifact_binding_v1(
    plan: Mapping[str, Any], *, required: bool
) -> dict[str, Any] | None:
    """Validate the exact native ACS runtime selected by a compiled plan.

    The Python ACS writers intentionally load the extension through the
    existing ``EIEL_NATIVE_RUNTIME_EXTENSION`` binding.  That environment
    variable is only an implementation transport; this function is the
    authority that prevents it from selecting an arbitrary or stale binary.
    Both the extension and its adjacent build manifest are plan artifacts and
    both identities are checked before execution.
    """

    execution = plan.get("execution")
    native_selected = (
        execution.get("native_streamed_checkpoint", False)
        if isinstance(execution, Mapping)
        else False
    )
    if native_selected is not True:
        return None
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        if required:
            raise CampaignProviderError(
                "native ACS runtime artifact binding is missing from the plan"
            )
        return None
    extension_row = artifacts.get(ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1)
    manifest_row = artifacts.get(ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1)
    if extension_row is None and manifest_row is None:
        if required:
            raise CampaignProviderError(
                "native ACS runtime artifact binding is missing from the plan"
            )
        return None
    if not isinstance(extension_row, Mapping) or not isinstance(
        manifest_row, Mapping
    ):
        raise CampaignProviderError(
            "native ACS runtime extension and manifest bindings must be paired"
        )

    def resolve(row: Mapping[str, Any], *, artifact_id: str, kind: str) -> Path:
        declared_value = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(declared_value, str)
            or not declared_value
            or not Path(declared_value).is_absolute()
            or Path(declared_value).is_symlink()
            or row.get("kind") != kind
            or row.get("abi") != ACS_NATIVE_RUNTIME_ABI_V1
            or not _valid_sha256(digest)
        ):
            raise CampaignProviderError(
                f"native ACS runtime artifact mapping is invalid: {artifact_id}"
            )
        resolved = _verified_runtime_artifact_path_v1(
            Path(declared_value), str(digest)
        )
        if resolved is None:
            raise CampaignProviderError(
                f"native ACS runtime artifact mapping is stale: {artifact_id}"
            )
        return resolved

    extension_path = resolve(
        extension_row,
        artifact_id=ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
        kind=ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
    )
    manifest_path = resolve(
        manifest_row,
        artifact_id=ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
        kind=ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
    )
    expected_manifest_path = extension_path.with_suffix(
        extension_path.suffix + ".manifest.json"
    )
    if manifest_path != expected_manifest_path:
        raise CampaignProviderError(
            "native ACS runtime manifest is not adjacent to its extension"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            "native ACS runtime build manifest is unreadable"
        ) from exc
    extension_digest = str(extension_row["sha256"])
    manifest_digest = str(manifest_row["sha256"])
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "eiel_native_runtime_build_manifest_v1"
        or manifest.get("stable_module_name") != ACS_NATIVE_RUNTIME_MODULE_NAME_V1
        or manifest.get("extension_abi_version") != ACS_NATIVE_RUNTIME_ABI_V1
        or manifest.get("extension_version") != ACS_NATIVE_RUNTIME_VERSION_V1
        or manifest.get("output_sha256") != extension_digest
        or manifest_digest != _file_sha256_v1(manifest_path)
        or manifest.get("runtime_jit_in_measured_execution") is not False
    ):
        raise CampaignProviderError(
            "native ACS runtime manifest identity is invalid"
        )
    return {
        "extension_artifact_id": ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
        "manifest_artifact_id": ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
        "extension_path": str(extension_path),
        "manifest_path": str(manifest_path),
        "extension_sha256": extension_digest,
        "manifest_sha256": manifest_digest,
        "extension_abi": ACS_NATIVE_RUNTIME_ABI_V1,
        "extension_version": ACS_NATIVE_RUNTIME_VERSION_V1,
        "stable_module_name": ACS_NATIVE_RUNTIME_MODULE_NAME_V1,
        "runtime_jit_in_measured_execution": False,
    }


def _acs_native_runtime_route_component_v1(
    plan: Mapping[str, Any], *, required: bool = False
) -> dict[str, Any] | None:
    binding = _acs_native_runtime_artifact_binding_v1(plan, required=required)
    if binding is None:
        return None
    return {
        "component_id": "acs_native_runtime_binding",
        **binding,
    }


OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT_ABI: Final[str] = (
    "eiel_option_d_m0_source_rebind_v3"
)
OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_option_d_cpu_policy_service_observation_v1"
)
OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_AUTHORITY: Final[str] = (
    "measurement_only_not_option_d_rng_authority"
)
OPTION_D_M0_CPU_POLICY_SERVICE_RAW_METRICS_FIELD: Final[str] = (
    "option_d_m0_exact_cpu_policy_service_raw_metrics"
)
OPTION_D_M0_HOST_ALLOCATION_MEASUREMENT_SCOPE: Final[str] = (
    "lower_bound_of_nonempty_explicit_service_owned_host_tensor_and_vector_outputs"
)
OPTION_D_M0_TRANSFER_OPERATION_MEASUREMENT_SCOPE: Final[str] = (
    "nonempty_tensor_copy_operations_only"
)
OPTION_D_M0_SYNCHRONIZATION_MEASUREMENT_SCOPE: Final[str] = (
    "explicit_final_cuda_stream_synchronizations_and_lower_bound_"
    "blocking_device_to_host_tensor_transfers"
)
OPTION_D_M0_COMBINED_ENVIRONMENT_SCHEMA_VERSION: Final[str] = (
    "eiel_option_d_m0_combined_environment_identity_v1"
)
OPTION_D_M0_D1_POLICY_ROW_WIDTH: Final[int] = 9_415
_OPTION_D_M0_GETRANDBITS_ZERO_SURFACES_V1: Final[tuple[str, ...]] = (
    "engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh",
    "engine/perf/cuda_architecture_b_persistent_turnover_plan.cuh",
    "engine/perf/native/cuda_architecture_b_lifecycle.cu",
    "engine/perf/native/cuda_architecture_b_turnover.cu",
)
OPTION_D_M0_PRIMARY_RECEIPT_FIELD: Final[str] = "option_d_rng_instrumentation_receipt"
OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_FIELD: Final[str] = (
    "exact_cpu_policy_service_observation_receipt"
)

_OPTION_D_M0_PROVIDER_RECEIPT_FIELDS_V1: Final[tuple[str, ...]] = (
    "option_d_m0_telemetry_enabled",
    "native_host_policy_birth_segment_refill_used",
    "policy_birth_segment_refill_calls_inside_tick_loop",
    "policy_birth_segment_refill_receipts",
    "policy_birth_segment_device_to_host_operations_lower_bound",
    "policy_birth_segment_host_to_device_operations_lower_bound",
    "policy_birth_segment_host_wall_time_ns",
    "policy_birth_segment_device_to_host_time_ns",
    "policy_birth_segment_host_to_device_time_ns",
    "policy_birth_segment_torch_random_generation_time_ns",
    "policy_birth_segment_torch_compare_pack_time_ns",
    "policy_birth_segment_allocation_stack_copy_time_ns",
    "native_host_gaussian_segment_refill_used",
    "gaussian_segment_refill_calls_inside_tick_loop",
    "gaussian_segment_refill_receipts",
    "gaussian_segment_device_to_host_operations_lower_bound",
    "gaussian_segment_host_to_device_operations_lower_bound",
    "gaussian_segment_host_wall_time_ns",
    "gaussian_segment_device_to_host_time_ns",
    "gaussian_segment_host_to_device_time_ns",
    "gaussian_segment_raw_generation_time_ns",
    "gaussian_segment_transcendental_conversion_time_ns",
    "gaussian_segment_allocation_copy_time_ns",
    "host_synchronizations_inside_loop",
    "final_host_synchronizations",
    "intermediate_device_to_host_operations",
    "final_device_to_host_operations",
    "intermediate_host_to_device_operations",
    "final_host_to_device_operations",
    "intermediate_host_synchronizations",
    "aten_operations_inside_tick_loop",
    "aten_operations_inside_tick_loop_count_semantics",
    "exact_cpu_policy_service_calls_inside_tick_loop",
    "native_host_policy_service_used",
    OPTION_D_M0_CPU_POLICY_SERVICE_RAW_METRICS_FIELD,
)

_OPTION_D_M0_RUNTIME_PLAN_NATIVE_FIELDS_V1: Final[tuple[str, ...]] = (
    "owner",
    "physical_window_generations",
    "logical_generation_count",
    "external_python_window_boundaries",
    "native_window_loop_owner",
    "internal_pybind_redispatch",
    "direct_cpp_window_core_dispatch",
    "production_native_chunk_loop",
    "hidden_fallback",
    "runtime_jit",
)

_OPTION_D_M0_RESIDENT_WINDOW_ACCOUNTING_INTEGER_FIELDS_V1: Final[tuple[str, ...]] = (
    "resident_window_outer_boundary_device_to_host_operations_lower_bound",
    "resident_window_outer_event_device_to_host_operations_lower_bound",
    "resident_window_outer_refill_device_to_host_operations_lower_bound",
    "resident_window_outer_refill_host_to_device_operations_lower_bound",
)
_OPTION_D_M0_RESIDENT_WINDOW_TRANSFER_SCOPE_V1: Final[str] = (
    "nonempty_cuda_tensor_copy_operations_in_native_resident_window_"
    "boundary_status_summary_event_archive_and_rng_refill_v1"
)
_OPTION_D_M0_RESIDENT_WINDOW_COUNTER_SEMANTICS_V1: Final[str] = (
    "child_final_counters_reclassified_as_intermediate_except_for_"
    "the_single_route_terminal_window_v1"
)

_OPTION_D_M0_REQUIRED_MANIFEST_FIELDS_V1: Final[tuple[str, ...]] = (
    "artifact_sha256",
    "committed_source_snapshot",
    "committed_source_snapshot.build_source_commit",
    "committed_source_snapshot.compiled_source_snapshot_sha256",
    "source_hashes",
    "compiler_command",
    "compiler_path",
    "compiler_version",
    "nvcc_command",
    "nvcc_path",
    "nvcc_version",
    "cuda_home",
    "cxx_flags",
    "nvcc_flags",
    "link_flags",
    "torch_include_paths",
    "torch_library_paths",
    "numpy_random_lib",
    "option_d_m0_telemetry_build",
)
_OPTION_D_M0_REQUIRED_SUPPLEMENTAL_IDENTITIES_V1: Final[tuple[str, ...]] = (
    "artifact_manifest_sha256",
    "loaded_artifact_sha256",
    "compiler_binary_sha256",
    "nvcc_binary_sha256",
    "relevant_linked_library_sha256",
    "artifact_environment_identity_sha256",
    "interpreter_environment_identity_sha256",
    "combined_environment_identity_sha256",
)

_LEGACY_NATIVE_RECEIPT_FIELDS_V1: Final[tuple[str, ...]] = (
    "native_host_gaussian_segment_refill_used",
    "gaussian_segment_refill_calls_inside_tick_loop",
    "gaussian_segment_refill_receipts",
    "gaussian_segment_device_to_host_operations_lower_bound",
    "gaussian_segment_host_to_device_operations_lower_bound",
    "intermediate_device_to_host_operations",
    "intermediate_host_to_device_operations",
)

_OPTION_D_M0_PROVIDER_OWNED_DIRECT_FIELDS_V1: Final[tuple[str, ...]] = (
    "actual_ordinary_birth_count",
    "actual_turnover_birth_count",
    "torch_policy_raw_word_delta",
    "torch_policy_semantic_draw_delta",
    "terminal_digest_sha256",
    "output_digest_sha256",
    "checkpoint_identity_sha256",
    "process_rss_bytes",
    "process_hwm_bytes",
    "resident_vram_bytes",
    "peak_vram_bytes",
    "receipt_construction_time_ns",
    "checkpoint_projection_time_ns",
    "checkpoint_compression_time_ns",
    "checkpoint_compression_time_scope",
)

_OPTION_D_M0_TORCH_POLICY_COUNTER_CONTRACT_V1: Final[dict[str, Any]] = {
    "schema_version": "eiel_option_d_m0_torch_policy_counter_contract_v1",
    "algorithm": "pytorch_cpu_mt19937",
    "logical_stream": "policy_birth_torch_cpu_by_cell",
    "operation_scope": "d1_torch_rand_float32_only",
    "route_scope": "mutation_disabled_fixed_mlp_policy_birth",
    "raw_counter_unit": "mt19937_uint32_words",
    "semantic_counter_unit": "torch_rand_float32_scalars",
    "raw_counter_derivation": ("one_mt19937_uint32_word_per_torch_rand_float32_scalar"),
    "counter_boundary_source": (
        "canonical_checkpoint_policy_birth_tape_cursor_or_metadata_derived_"
        "mutation_torch_cpu_semantic_draws"
    ),
    "logical_policy_row_width": OPTION_D_M0_D1_POLICY_ROW_WIDTH,
    "logical_policy_row_width_source": "canonical_checkpoint_mlp_pack_last_dimension",
    "excluded_algorithms": "cpython_mt19937,numpy_pcg64",
}


def _option_d_m0_torch_policy_counter_contract_v1(
    logical_policy_row_width: int,
    *,
    rng_semantics_version: int = 1,
) -> dict[str, Any]:
    if type(logical_policy_row_width) is not int or logical_policy_row_width <= 0:
        raise CampaignProviderError(
            "Option D D1 logical MLP policy row width is invalid"
        )
    if rng_semantics_version == 1:
        return {
            **_OPTION_D_M0_TORCH_POLICY_COUNTER_CONTRACT_V1,
            "logical_policy_row_width": logical_policy_row_width,
        }
    if rng_semantics_version != 2:
        raise CampaignProviderError("Option D TW RNG semantic version is invalid")
    return {
        "schema_version": "eiel_option_d_m0_torch_policy_counter_contract_v2",
        "algorithm": "philox4x32-10",
        "logical_stream": "policy_birth_formula_by_logical_world",
        "operation_scope": "domain_0x0370_fixed_mlp_crossover_words",
        "route_scope": "mutation_disabled_fixed_mlp_policy_birth",
        "raw_counter_unit": "philox_uint32_words",
        "semantic_counter_unit": "formula_crossover_scalars",
        "raw_counter_derivation": "one_philox_uint32_word_per_crossover_scalar",
        "counter_boundary_source": (
            "canonical_checkpoint_metadata_derived_formula_semantic_draws"
        ),
        "logical_policy_row_width": logical_policy_row_width,
        "logical_policy_row_width_source": (
            "canonical_checkpoint_mlp_pack_last_dimension"
        ),
        "excluded_algorithms": "cpython_mt19937,numpy_pcg64,pytorch_cpu_mt19937",
        "rng_semantics_version": 2,
        "state_advancing_cursor": False,
    }


_OPTION_D_M0_TURNOVER_MLP_SHADOW_ELIDE_PARAMETER_V1: Final[str] = (
    "cuda_architecture_b.turnover_mlp_shadow_elide_enable"
)
_OPTION_D_M0_NATIVE_TURNOVER_REFILL_ROWS_FIELD_V1: Final[str] = (
    "option_d_m0_native_turnover_refill_committed_rows_v1"
)

_OPTION_D_M0_SHARED_IDENTITY_FIELDS_V1: Final[tuple[str, ...]] = (
    "run_identity_sha256",
    "cell_identity_sha256",
    "generation_tick_boundary_identity_sha256",
    "plan_identity_sha256",
    "runtime_plan_binding_identity_sha256",
    "source_commit",
    "environment_identity_sha256",
    "terminal_digest_sha256",
    "checkpoint_identity_sha256",
)

_OPTION_D_M0_PRIMARY_INTEGER_FIELDS_V1: Final[tuple[str, ...]] = (
    "actual_ordinary_birth_count",
    "actual_turnover_birth_count",
    "torch_policy_raw_word_delta",
    "torch_policy_semantic_draw_delta",
    "process_rss_bytes",
    "process_hwm_bytes",
    "resident_vram_bytes",
    "peak_vram_bytes",
    "receipt_construction_time_ns",
    "checkpoint_projection_time_ns",
    "checkpoint_compression_time_ns",
)

_OPTION_D_M0_PRIMARY_DIGEST_FIELDS_V1: Final[tuple[str, ...]] = (
    "output_digest_sha256",
)

_OPTION_D_M0_CPU_POLICY_SERVICE_INTEGER_FIELDS_V1: Final[tuple[str, ...]] = (
    "calls_inside_tick_loop",
    "host_wall_time_ns",
    "device_to_host_time_ns",
    "host_to_device_time_ns",
    "cpu_mlp_time_ns",
    "cpu_sampling_time_ns",
    "final_stream_synchronization_time_ns",
    "device_to_host_bytes_lower_bound",
    "host_to_device_bytes_lower_bound",
    "device_to_host_operations",
    "host_to_device_operations",
    "explicit_final_stream_synchronizations",
    "implicit_blocking_device_to_host_transfer_waits_lower_bound",
    "host_allocation_count_lower_bound",
    "host_allocation_bytes_lower_bound",
    "process_rss_bytes",
    "process_hwm_bytes",
)

_OPTION_D_M0_CPU_POLICY_SERVICE_RAW_INTEGER_FIELDS_V1: Final[tuple[str, ...]] = tuple(
    field
    for field in _OPTION_D_M0_CPU_POLICY_SERVICE_INTEGER_FIELDS_V1
    if field not in {"process_rss_bytes", "process_hwm_bytes"}
)

_OPTION_D_M0_CPU_POLICY_SERVICE_SUMMARY_FIELDS_V1: Final[tuple[str, ...]] = (
    "calls_inside_tick_loop",
    "host_wall_time_ns",
    "device_to_host_time_ns",
    "host_to_device_time_ns",
    "cpu_mlp_time_ns",
    "cpu_sampling_time_ns",
    "final_stream_synchronization_time_ns",
    "device_to_host_bytes_lower_bound",
    "host_to_device_bytes_lower_bound",
    "device_to_host_operations",
    "host_to_device_operations",
    "explicit_final_stream_synchronizations",
    "implicit_blocking_device_to_host_transfer_waits_lower_bound",
    "host_allocation_count_lower_bound",
    "host_allocation_bytes_lower_bound",
    "host_allocation_measurement_scope",
    "transfer_operation_measurement_scope",
    "synchronization_measurement_scope",
)


def _physical_window_generations_from_plan_v1(plan: Mapping[str, Any]) -> int:
    execution = plan.get("execution")
    if (
        isinstance(execution, Mapping)
        and execution.get("execution_route_identity")
        == "tick_native_execution_route_v1"
    ):
        # The legacy source ABI still has a one-window compatibility argument.
        # It is not the native owner or dispatch authority; those are the
        # explicit tick spans in tick_native_control.
        return 1
    value = (
        execution.get("physical_window_generations")
        if isinstance(execution, Mapping)
        else None
    )
    if value is None:
        return 2
    if type(value) is not int or value not in {1, 2}:
        raise CampaignProviderError(
            "physical_window_generations must be exact integer 1 or 2"
        )
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha256_file_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option_d_m0_environment_component_identities_v1(
    environment: Mapping[str, Any],
) -> dict[str, str]:
    required_mappings = ("python", "torch", "numpy", "platform", "module_files")
    if (
        environment.get("schema_version")
        != "eiel_option_d_production_environment_identity_v2"
        or any(
            not isinstance(environment.get(name), Mapping) for name in required_mappings
        )
        or environment.get("module_probe_errors") != {}
        or not isinstance(environment.get("requested_executable"), str)
        or not environment["requested_executable"]
        or not isinstance(environment.get("requested_executable_resolved"), str)
        or not environment["requested_executable_resolved"]
        or environment.get("requested_path_preserved_for_invocation") is not True
    ):
        raise CampaignProviderError(
            "Option D production environment component identity is incomplete"
        )
    torch_identity = environment["torch"]
    python_identity = environment["python"]
    return {
        "supplied_interpreter": _canonical_sha256(
            {
                "requested_executable": environment["requested_executable"],
                "expected_venv_root": environment.get("expected_venv_root"),
            }
        ),
        "resolved_interpreter": _canonical_sha256(
            {
                "requested_executable_resolved": environment[
                    "requested_executable_resolved"
                ],
                "real_executable": python_identity.get("real_executable"),
                "binary_sha256": python_identity.get("binary_sha256"),
            }
        ),
        "python_runtime": _canonical_sha256(dict(python_identity)),
        "pytorch": _canonical_sha256(dict(torch_identity)),
        "numpy": _canonical_sha256(dict(environment["numpy"])),
        "platform_glibc_endian": _canonical_sha256(dict(environment["platform"])),
        "loaded_module_files": _canonical_sha256(
            dict(sorted(environment["module_files"].items()))
        ),
        "cuda_runtime_and_gpu": _canonical_sha256(
            {
                name: torch_identity.get(name)
                for name in (
                    "cuda_available",
                    "cuda_build",
                    "cuda_compiled_version",
                    "cuda_driver_version",
                    "gpu",
                )
            }
        ),
        "compiler_and_abi": _canonical_sha256(
            {
                "python_compiler": python_identity.get("compiler"),
                "torch_build_config_sha256": torch_identity.get("build_config_sha256"),
                "torch_cxx11_abi": torch_identity.get("cxx11_abi"),
            }
        ),
    }


def _option_d_m0_measurement_authority_v1(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    def mapping(name: str) -> Mapping[str, Any]:
        value = payload.get(name)
        if not isinstance(value, Mapping):
            raise CampaignProviderError(
                f"Option D M0 source receipt lacks mapping {name}"
            )
        return value

    package = mapping("package")
    fixtures = mapping("accepted_fixtures")
    source = mapping("source_binding")
    build = mapping("live_build_configuration")
    production_environment = mapping("production_interpreter")
    environment_comparison = mapping("environment_target_comparison")
    regeneration = mapping("production_regeneration")
    inventory = mapping("live_rng_inventory_reconciliation")
    ubuntu = mapping("ubuntu_source_closure")
    final_git = mapping("final_git_snapshot_stability")
    telemetry = mapping("option_d_m0_telemetry_build")
    route_scope = mapping("route_scope_contract")
    gate = mapping("m0_measurement_gate")
    production_gate = mapping("production_reconciliation_gate")
    getrandbits_zero_inventory = mapping("getrandbits_zero_inventory")
    snapshot = build.get("committed_source_snapshot")
    build_context = build.get("build_context")
    build_basis = build.get("build_configuration_identity_basis")
    fixture_components = fixtures.get("components")
    if (
        package.get("manifest_status") != "verified"
        or package.get("zip_sha256") != OPTION_D_M0_EXPECTED_REFERENCE_PACKAGE_SHA256
        or fixtures.get("status") != "verified_immutable_zip_members"
        or fixtures.get("accepted_fixtures_mutable") is not False
        or not isinstance(fixture_components, Mapping)
        or set(fixture_components) != {"D1", "D2", "D3", "INTEGRATION"}
        or any(
            not isinstance(row, Mapping)
            or not _valid_sha256(row.get("manifest_sha256"))
            or not _valid_sha256(row.get("fixture_manifest_sha256"))
            or type(row.get("files_verified")) is not int
            or row["files_verified"] <= 0
            for row in fixture_components.values()
        )
    ):
        raise CampaignProviderError(
            "Option D M0 immutable package or accepted fixture authority is invalid"
        )
    source_commit = source.get("head")
    compiled_snapshot_sha256 = build.get("compiled_source_snapshot_sha256")
    snapshot_rows = snapshot.get("sources") if isinstance(snapshot, Mapping) else None
    validated_snapshot_rows: list[dict[str, str]] = []
    if isinstance(snapshot_rows, list):
        for row in snapshot_rows:
            if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
                continue
            raw_relative = row.get("path")
            digest = row.get("sha256")
            relative = (
                PurePosixPath(raw_relative)
                if isinstance(raw_relative, str)
                else PurePosixPath()
            )
            if (
                not raw_relative
                or relative.is_absolute()
                or relative.as_posix() != raw_relative
                or any(part in {"", ".", ".."} for part in relative.parts)
                or not _valid_sha256(digest)
            ):
                continue
            validated_snapshot_rows.append(
                {"path": str(raw_relative), "sha256": str(digest)}
            )
    source_snapshot_basis = {
        "schema_version": (
            snapshot.get("schema_version") if isinstance(snapshot, Mapping) else None
        ),
        "build_source_commit": (
            snapshot.get("build_source_commit")
            if isinstance(snapshot, Mapping)
            else None
        ),
        "sources": validated_snapshot_rows,
    }
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or source.get("rebind_complete") is not True
        or source.get("dirty") is not False
        or build.get("passed") is not True
        or not isinstance(snapshot, Mapping)
        or snapshot.get("passed") is not True
        or snapshot.get("build_source_commit") != source_commit
        or not _valid_sha256(compiled_snapshot_sha256)
        or validated_snapshot_rows != snapshot_rows
        or validated_snapshot_rows
        != sorted(validated_snapshot_rows, key=lambda row: row["path"])
        or len({row["path"] for row in validated_snapshot_rows})
        != len(validated_snapshot_rows)
        or not validated_snapshot_rows
        or _canonical_sha256(source_snapshot_basis) != compiled_snapshot_sha256
        or snapshot.get("compiled_source_snapshot_sha256") != compiled_snapshot_sha256
        or payload.get("compiled_source_snapshot_sha256") != compiled_snapshot_sha256
        or not isinstance(build_context, Mapping)
        or build_context.get("build_source_commit") != source_commit
        or build_context.get("compiled_source_snapshot_sha256")
        != compiled_snapshot_sha256
        or not isinstance(build_basis, Mapping)
        or build.get("build_context_identity_sha256")
        != _canonical_sha256(dict(build_context))
        or build.get("build_configuration_identity_sha256")
        != _canonical_sha256(dict(build_basis))
        or not isinstance(build.get("artifact_manifest_contract"), Mapping)
        or build["artifact_manifest_contract"].get(
            "provider_must_validate_loaded_artifact"
        )
        is not True
        or build.get("source_inventory_per_file_sha256")
        != {
            row.get("path"): row.get("sha256")
            for row in snapshot.get("sources", [])
            if isinstance(row, Mapping)
        }
    ):
        raise CampaignProviderError(
            "Option D M0 clean source or builder-equivalent source snapshot is invalid"
        )
    authority_checks = environment_comparison.get("authority_checks")

    def observed_environment_identity(name: Any) -> Any:
        aliases = {
            "platform.bits": ("platform", "pointer_bits"),
            "platform.glibc_version": ("platform", "libc", 1),
            "torch.cuda_runtime": ("torch", "cuda_build"),
        }
        path: tuple[Any, ...] = aliases.get(
            name,
            tuple(str(name).split(".")) if isinstance(name, str) else (),
        )
        value: Any = production_environment
        for component in path:
            if isinstance(component, int):
                if not isinstance(value, list) or component >= len(value):
                    return None
                value = value[component]
            elif isinstance(value, Mapping):
                value = value.get(component)
            else:
                return None
        return value

    environment_rows_exact = (
        isinstance(authority_checks, list)
        and bool(authority_checks)
        and all(
            isinstance(row, Mapping)
            and row.get("match") is True
            and row.get("observed")
            == observed_environment_identity(row.get("identity"))
            and row.get("expected") == row.get("observed")
            for row in authority_checks
        )
    )
    component_comparisons = regeneration.get("component_comparisons")
    staged_manifests = regeneration.get("staged_fixture_manifest_validations")
    orchestrator_validation = regeneration.get("orchestrator_receipt_validation")
    if (
        environment_comparison.get("status") != "PASS_EXACT_PRODUCTION_CONTRACT"
        or environment_comparison.get("exact_contract_match") is not True
        or environment_comparison.get("authority_mismatches") != []
        or not _valid_sha256(environment_comparison.get("target_sha256"))
        or not environment_rows_exact
        or regeneration.get("passed") is not True
        or regeneration.get("regeneration_scope") != "d1-d3-plus-integration"
        or regeneration.get("mismatch_components") != []
        or regeneration.get("missing_staged_components") != []
        or regeneration.get("invalid_staged_fixture_manifests") != []
        or regeneration.get("accepted_fixture_copy_unchanged") is not True
        or regeneration.get("accepted_fixture_archive_unchanged") is not True
        or regeneration.get("accepted_fixtures_mutated") is not False
        or regeneration.get("package_zip_sha256_before")
        != regeneration.get("package_zip_sha256_after")
        or regeneration.get("package_zip_sha256_after") != package.get("zip_sha256")
        or not isinstance(component_comparisons, Mapping)
        or set(component_comparisons) != {"D1", "D2", "D3", "INTEGRATION"}
        or any(
            not isinstance(row, Mapping)
            or row.get("algorithm_output_state_match") is not True
            for row in component_comparisons.values()
        )
        or not isinstance(staged_manifests, Mapping)
        or set(staged_manifests) != {"D1", "D2", "D3", "INTEGRATION"}
        or any(
            not isinstance(row, Mapping)
            or row.get("passed") is not True
            or not _valid_sha256(row.get("manifest_sha256"))
            or row.get("manifest_identity_sha256") != row.get("actual_identity_sha256")
            for row in staged_manifests.values()
        )
        or not isinstance(orchestrator_validation, Mapping)
        or orchestrator_validation.get("passed") is not True
    ):
        raise CampaignProviderError(
            "Option D M0 production environment or fixture regeneration is invalid"
        )
    inventory_files = inventory.get("per_file_sha256")
    if (
        inventory.get("passed") is not True
        or inventory.get("head") != source_commit
        or inventory.get("dirty") is not False
        or inventory.get("blocking_reasons") != []
        or inventory.get("unknown_consumer_count") != 0
        or inventory.get("ambiguous_consumer_count") != 0
        or not _valid_sha256(inventory.get("inventory_identity_sha256"))
        or not _valid_sha256(inventory.get("scanned_source_manifest_sha256"))
        or not isinstance(inventory_files, Mapping)
        or not inventory_files
        or any(
            not isinstance(path, str) or not _valid_sha256(digest)
            for path, digest in inventory_files.items()
        )
        or inventory.get("route_exclusion_authority")
        != "provisional_pending_independent_runtime_plan_binding"
        or inventory.get("runtime_plan_binding_present") is not False
    ):
        raise CampaignProviderError(
            "Option D M0 live RNG inventory or provisional exclusions are invalid"
        )
    active_inclusion = inventory.get("active_runtime_inclusion")
    exclusions = inventory.get("excluded_source_categories")
    source_anchors = inventory.get("source_anchors")
    classified_consumers = inventory.get("classified_consumers")
    unknown_consumers = inventory.get("unknown_consumers")
    ambiguous_consumers = inventory.get("ambiguous_consumers")
    support_surfaces = inventory.get("support_surfaces")
    alternate_surfaces = inventory.get("alternate_route_surfaces")
    discovery_only_consumers = (
        inventory.get("broad_discovery", {}).get("discovery_only_consumers")
        if isinstance(inventory.get("broad_discovery"), Mapping)
        else None
    )
    owner_rules = inventory.get("w32_owner_rule_binding")
    consumer_delta = inventory.get("consumer_delta")
    d2_binding = inventory.get("d2_getrandbits_zero_followup")
    provisional_exclusions = inventory.get("provisional_route_exclusion_receipt")
    sequence_fields = (
        source_anchors,
        classified_consumers,
        unknown_consumers,
        ambiguous_consumers,
        support_surfaces,
        alternate_surfaces,
        discovery_only_consumers,
        owner_rules,
    )
    if (
        not isinstance(active_inclusion, Mapping)
        or active_inclusion.get("passed") is not True
        or not isinstance(exclusions, Mapping)
        or not _valid_sha256(exclusions.get("exclusion_identity_sha256"))
        or any(not isinstance(value, list) for value in sequence_fields)
        or not isinstance(consumer_delta, Mapping)
        or not isinstance(d2_binding, Mapping)
        or not isinstance(provisional_exclusions, Mapping)
    ):
        raise CampaignProviderError(
            "Option D M0 RNG inventory identity basis is incomplete"
        )
    alternate_identity = _canonical_sha256(alternate_surfaces)
    alternate_file_hashes = {
        str(row.get("path")): str(row.get("file_sha256"))
        for row in alternate_surfaces
        if isinstance(row, Mapping)
        and isinstance(row.get("path"), str)
        and _valid_sha256(row.get("file_sha256"))
    }
    if (
        inventory.get("scanned_source_manifest_sha256")
        != _canonical_sha256(dict(inventory_files))
        or inventory.get("source_anchor_identity_sha256")
        != _canonical_sha256(source_anchors)
        or inventory.get("support_surface_identity_sha256")
        != _canonical_sha256(support_surfaces)
        or inventory.get("alternate_route_identity_sha256") != alternate_identity
        or inventory.get("w32_owner_rule_binding_sha256")
        != _canonical_sha256(owner_rules)
        or provisional_exclusions.get("schema_version")
        != "eiel_option_d_m0_provisional_route_exclusions_v1"
        or provisional_exclusions.get("surface_count") != len(alternate_surfaces)
        or provisional_exclusions.get("surfaces_identity_sha256") != alternate_identity
        or provisional_exclusions.get("per_file_sha256") != alternate_file_hashes
        or provisional_exclusions.get("provider_runtime_plan_binding_required")
        is not True
        or provisional_exclusions.get("runtime_reachability_proven") is not False
        or any(
            not isinstance(row, Mapping)
            or not _valid_sha256(row.get("file_sha256"))
            or not row.get("route_exclusion")
            or not row.get("exact_symbol_chain")
            or not row.get("declared_call_chains")
            or not row.get("build_condition_context")
            or row.get("runtime_reachability_proven") is not False
            for row in alternate_surfaces
        )
    ):
        raise CampaignProviderError(
            "Option D M0 RNG inventory source or exclusion identities are invalid"
        )
    d2_findings = getrandbits_zero_inventory.get("findings")
    d2_finding_paths = (
        [row.get("path") if isinstance(row, Mapping) else None for row in d2_findings]
        if isinstance(d2_findings, list)
        else None
    )
    source_file_rows = source.get("files")
    source_files_by_path: dict[str, Mapping[str, Any]] = {}
    source_file_rows_valid = isinstance(source_file_rows, list)
    if isinstance(source_file_rows, list):
        for row in source_file_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
                source_file_rows_valid = False
                continue
            path = str(row["path"])
            if path in source_files_by_path:
                source_file_rows_valid = False
                continue
            source_files_by_path[path] = row
    expected_d2_followup = (
        any(
            isinstance(row, Mapping) and row.get("requires_d2_followup") is True
            for row in d2_findings
        )
        if isinstance(d2_findings, list)
        else None
    )
    expected_d2_classification = (
        "known_d2_getrandbits_zero_followup_preserved"
        if expected_d2_followup is True
        else "no_getrandbits_zero_followup_detected"
    )

    def canonical_d2_finding(row: Any) -> bool:
        if not isinstance(row, Mapping):
            return False
        path = row.get("path")
        source_row = source_files_by_path.get(path) if isinstance(path, str) else None
        has_zero_guard = row.get("has_explicit_zero_guard")
        consumes_raw32 = row.get("small_width_path_consumes_raw32")
        requires_followup = row.get("requires_d2_followup")
        expected_followup = consumes_raw32 is True and has_zero_guard is False
        expected_status = (
            "conformant_source_shape"
            if has_zero_guard is True
            else "requires_live_d2_conformance_test"
        )
        return (
            row.get("present") is True
            and _valid_sha256(row.get("source_sha256"))
            and isinstance(source_row, Mapping)
            and source_row.get("present") is True
            and source_row.get("live_sha256") == row.get("source_sha256")
            and type(has_zero_guard) is bool
            and consumes_raw32 is True
            and requires_followup is expected_followup
            and row.get("status") == expected_status
        )

    if (
        getrandbits_zero_inventory.get("schema_version")
        != "eiel_option_d_getrandbits_zero_inventory_v1"
        or getrandbits_zero_inventory.get("complete_inventory") is not True
        or getrandbits_zero_inventory.get("known_surface_count")
        != len(_OPTION_D_M0_GETRANDBITS_ZERO_SURFACES_V1)
        or not isinstance(d2_findings, list)
        or len(d2_findings) != len(_OPTION_D_M0_GETRANDBITS_ZERO_SURFACES_V1)
        or d2_finding_paths != list(_OPTION_D_M0_GETRANDBITS_ZERO_SURFACES_V1)
        or len(set(d2_finding_paths or ()))
        != len(_OPTION_D_M0_GETRANDBITS_ZERO_SURFACES_V1)
        or not source_file_rows_valid
        or any(not canonical_d2_finding(row) for row in d2_findings)
        or getrandbits_zero_inventory.get("followup_required")
        is not expected_d2_followup
        or d2_binding.get("source_inventory_identity_sha256")
        != _canonical_sha256(dict(getrandbits_zero_inventory))
        or d2_binding.get("followup_required") is not expected_d2_followup
        or d2_binding.get("classification") != expected_d2_classification
        or d2_binding.get("repair_performed") is not False
    ):
        raise CampaignProviderError(
            "Option D M0 getrandbits(0) deferred follow-up identity is invalid"
        )
    live_consumer_identity = _canonical_sha256(
        {
            "classified": classified_consumers,
            "unknown": unknown_consumers,
            "ambiguous": ambiguous_consumers,
            "support": support_surfaces,
            "alternate_route": alternate_surfaces,
            "discovery_only": discovery_only_consumers,
        }
    )
    inventory_basis = {
        "package_inventory_identity_sha256": inventory.get(
            "package_inventory_identity_sha256"
        ),
        "head": inventory.get("head"),
        "git_status_sha256": inventory.get("git_status_sha256"),
        "branch": inventory.get("branch"),
        "dirty_state_classification": inventory.get("dirty_state_classification"),
        "build_configuration_identity_sha256": build.get(
            "build_configuration_identity_sha256"
        ),
        "active_runtime_identity_sha256": _canonical_sha256(dict(active_inclusion)),
        "exclusion_identity_sha256": exclusions.get("exclusion_identity_sha256"),
        "scanned_source_manifest_sha256": inventory.get(
            "scanned_source_manifest_sha256"
        ),
        "anchor_identity_sha256": _canonical_sha256(source_anchors),
        "live_consumer_identity_sha256": live_consumer_identity,
        "support_surface_identity_sha256": _canonical_sha256(support_surfaces),
        "alternate_route_identity_sha256": alternate_identity,
        "owner_rule_binding_sha256": _canonical_sha256(owner_rules),
        "bound_source_manifest_sha256": _canonical_sha256(dict(inventory_files)),
        "consumer_delta_sha256": _canonical_sha256(dict(consumer_delta)),
        "d2_followup": dict(d2_binding),
    }
    if inventory.get(
        "live_consumer_identity_sha256"
    ) != live_consumer_identity or inventory.get(
        "inventory_identity_sha256"
    ) != _canonical_sha256(
        inventory_basis
    ):
        raise CampaignProviderError(
            "Option D M0 RNG inventory aggregate identity is invalid"
        )
    ubuntu_members = ubuntu.get("dsc_source_member_identities")
    ubuntu_patches = ubuntu.get("patch_identities")
    ubuntu_rng_comparisons = ubuntu.get("rng_source_comparisons")
    if (
        ubuntu.get("passed") is not True
        or ubuntu.get("operator_trust_anchor_supplied") is not True
        or not _valid_sha256(ubuntu.get("trusted_dsc_sha256"))
        or ubuntu.get("rng_source_byte_identical") is not True
        or not _valid_sha256(ubuntu.get("dsc_source_member_identity_sha256"))
        or not _valid_sha256(ubuntu.get("patch_series_sha256"))
        or not _valid_sha256(ubuntu.get("patch_identity_sha256"))
        or not _valid_sha256(ubuntu.get("rng_source_comparison_identity_sha256"))
        or not isinstance(ubuntu_members, list)
        or not ubuntu_members
        or ubuntu.get("dsc_source_member_identity_sha256")
        != _canonical_sha256(ubuntu_members)
        or not isinstance(ubuntu_patches, list)
        or not ubuntu_patches
        or ubuntu.get("patch_identity_sha256") != _canonical_sha256(ubuntu_patches)
        or not isinstance(ubuntu_rng_comparisons, Mapping)
        or set(ubuntu_rng_comparisons) != {"Modules/_randommodule.c", "Lib/random.py"}
        or any(
            not isinstance(row, Mapping) or row.get("identical") is not True
            for row in ubuntu_rng_comparisons.values()
        )
        or ubuntu.get("rng_source_comparison_identity_sha256")
        != _canonical_sha256(dict(ubuntu_rng_comparisons))
        or final_git.get("status") != "PASS_HEAD_AND_STATUS_STABLE_AFTER_ALL_HASHES"
        or final_git.get("head") != source_commit
        or final_git.get("dirty_state_classification") != "clean"
        or final_git.get("git_status_sha256") != hashlib.sha256(b"").hexdigest()
        or telemetry.get("option_d_m0_telemetry_enabled") is not True
        or build.get("telemetry_build") != telemetry
        or build.get("telemetry_identity_matches") is not True
    ):
        raise CampaignProviderError(
            "Option D M0 Ubuntu source, final Git, or telemetry authority is invalid"
        )
    route_basis_fields = (
        "schema_version",
        "scope_id",
        "accepted_eiel_commit",
        "compiled_source_snapshot_sha256",
        "build_configuration_identity_sha256",
        "build_context_identity_sha256",
        "telemetry_build_identity_sha256",
        "inventory_identity_sha256",
        "declared_owner_rule_binding_sha256",
        "provisional_route_exclusion_identity_sha256",
        "package_inventory_identity_sha256",
    )
    route_basis = {name: route_scope.get(name) for name in route_basis_fields}
    if (
        route_scope.get("schema_version") != "eiel_option_d_m0_route_scope_contract_v1"
        or route_scope.get("scope_id") != "w32-fixed-mlp-g1-m0-measurement-v1"
        or route_scope.get("accepted_eiel_commit") != source_commit
        or route_scope.get("compiled_source_snapshot_sha256")
        != compiled_snapshot_sha256
        or route_scope.get("build_configuration_identity_sha256")
        != build.get("build_configuration_identity_sha256")
        or route_scope.get("build_context_identity_sha256")
        != build.get("build_context_identity_sha256")
        or route_scope.get("telemetry_build_identity_sha256")
        != _canonical_sha256(dict(telemetry))
        or route_scope.get("inventory_identity_sha256")
        != inventory.get("inventory_identity_sha256")
        or route_scope.get("declared_owner_rule_binding_sha256")
        != inventory.get("w32_owner_rule_binding_sha256")
        or route_scope.get("provisional_route_exclusion_identity_sha256")
        != inventory.get("alternate_route_identity_sha256")
        or route_scope.get("package_inventory_identity_sha256")
        != inventory.get("package_inventory_identity_sha256")
        or route_scope.get("passed_for_measurement_identity") is not True
        or route_scope.get("runtime_plan_binding_present") is not False
        or route_scope.get("production_integration_authority") is not False
        or route_scope.get("route_scope_contract_identity_sha256")
        != _canonical_sha256(route_basis)
        or payload.get("route_scope_contract_identity_sha256")
        != route_scope.get("route_scope_contract_identity_sha256")
    ):
        raise CampaignProviderError(
            "Option D M0 declarative route-scope identity is invalid"
        )
    gate_basis = dict(gate)
    gate_identity = gate_basis.pop("decision_identity_sha256", None)
    if (
        gate.get("schema_version") != "eiel_option_d_m0_measurement_gate_v1"
        or gate.get("status") != "PASS_M0_MEASUREMENT_ONLY_GATE"
        or gate.get("decision") != "M0-A"
        or gate.get("passed") is not True
        or gate.get("measurement_authority") is not True
        or gate.get("provider_measurement_authority") is not False
        or gate.get("production_integration_authority") is not False
        or gate.get("blocking_gates") != []
        or gate.get("accepted_eiel_commit") != source_commit
        or gate.get("compiled_source_snapshot_sha256") != compiled_snapshot_sha256
        or gate.get("route_scope_contract_identity_sha256")
        != route_scope.get("route_scope_contract_identity_sha256")
        or gate.get("runtime_plan_binding_present") is not False
        or gate.get("provider_runtime_plan_binding_required") is not True
        or not _valid_sha256(gate_identity)
        or _canonical_sha256(gate_basis) != gate_identity
        or production_gate.get("production_integration_authority") is not False
        or production_gate.get("accepted_eiel_commit") != source_commit
        or production_gate.get("accepted_eiel_commit_match") is not True
    ):
        raise CampaignProviderError(
            "Option D M0 measurement-only gate or withheld production authority is invalid"
        )
    return {
        "source_commit": source_commit,
        "compiled_source_snapshot_sha256": compiled_snapshot_sha256,
        "build_configuration_identity_sha256": build[
            "build_configuration_identity_sha256"
        ],
        "build_context_identity_sha256": build["build_context_identity_sha256"],
        "route_scope_contract_identity_sha256": route_scope[
            "route_scope_contract_identity_sha256"
        ],
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "measurement_gate_identity_sha256": gate_identity,
        "committed_source_snapshot": dict(snapshot),
        "build_context": dict(build_context),
        "build_configuration_identity_basis": dict(build_basis),
        "artifact_manifest_contract": dict(build["artifact_manifest_contract"]),
    }


def _option_d_m0_source_environment_identity_v1(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = plan.get("artifacts")
    binding = (
        artifacts.get(OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT_NAME)
        if isinstance(artifacts, Mapping)
        else None
    )
    if not isinstance(binding, Mapping):
        raise CampaignProviderError(
            "Option D qualification requires the compiled source/environment "
            "receipt artifact"
        )
    raw_path = binding.get("path")
    expected_sha256 = binding.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not _valid_sha256(expected_sha256)
        or binding.get("kind") != OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT_KIND
        or binding.get("abi") != OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT_ABI
    ):
        raise CampaignProviderError(
            "Option D source/environment artifact path, SHA-256, kind, or ABI is "
            "invalid"
        )
    path = Path(raw_path)
    if not path.is_absolute() or not path.is_file():
        raise CampaignProviderError(
            "Option D source/environment receipt must be an existing absolute file"
        )
    if _sha256_file_v1(path) != expected_sha256:
        raise CampaignProviderError(
            "Option D source/environment receipt file SHA-256 differs"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            f"Option D source/environment receipt is unreadable: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CampaignProviderError(
            "Option D source/environment receipt must be an object"
        )
    receipt_identity = payload.get("receipt_identity_sha256")
    identity_payload = dict(payload)
    identity_payload.pop("receipt_identity_sha256", None)
    source = payload.get("source_binding")
    environment = payload.get("production_interpreter")
    if (
        payload.get("schema_version")
        != OPTION_D_M0_SOURCE_REBIND_RECEIPT_SCHEMA_VERSION
        or not _valid_sha256(receipt_identity)
        or _canonical_sha256(identity_payload) != receipt_identity
        or not isinstance(source, Mapping)
        or not isinstance(environment, Mapping)
        or environment.get("status") != "PASS"
    ):
        raise CampaignProviderError(
            "Option D source/environment receipt identity or interpreter is invalid"
        )
    authority = _option_d_m0_measurement_authority_v1(payload)
    source_commit = authority["source_commit"]
    environment_identity = environment.get("identity_sha256")
    environment_identity_payload = dict(environment)
    environment_identity_payload.pop("identity_sha256", None)
    if (
        not _valid_sha256(environment_identity)
        or _canonical_sha256(environment_identity_payload) != environment_identity
    ):
        raise CampaignProviderError(
            "Option D source/environment receipt identities are inconsistent"
        )
    component_identities = _option_d_m0_environment_component_identities_v1(environment)
    return {
        "source_commit": source_commit,
        "environment_identity_sha256": environment_identity,
        "source_interpreter_environment_identity_sha256": environment_identity,
        "path": str(path),
        "sha256": expected_sha256,
        "kind": binding["kind"],
        "abi": binding["abi"],
        "receipt_identity_sha256": receipt_identity,
        "environment_component_identity_sha256s": component_identities,
        "source_interpreter_component_identity_sha256s": component_identities,
        **authority,
    }


def _option_d_m0_manifest_file_identity_v1(
    value: Any,
    *,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignProviderError(f"Option D artifact manifest lacks {field}")
    raw_path = value.get("path")
    digest = value.get("sha256")
    size_bytes = value.get("size_bytes")
    path = Path(raw_path) if isinstance(raw_path, str) else Path()
    if (
        not path.is_absolute()
        or not path.is_file()
        or not _valid_sha256(digest)
        or type(size_bytes) is not int
        or size_bytes <= 0
        or path.stat().st_size != size_bytes
        or _sha256_file_v1(path) != digest
    ):
        raise CampaignProviderError(
            f"Option D artifact manifest {field} identity is invalid"
        )
    return dict(value)


def _option_d_m0_loaded_artifact_contract_v1(
    *,
    module: Any,
    plan: Mapping[str, Any],
    source_environment: Mapping[str, Any],
) -> dict[str, Any]:
    resource_export = getattr(
        module,
        "eiel_cuda_architecture_b_flattened_controller_resource_receipt_v1",
        None,
    )
    provenance_export = getattr(
        module,
        "eiel_authoritative_state_build_provenance_v1",
        None,
    )
    if not callable(resource_export) or not callable(provenance_export):
        raise CampaignProviderError(
            "Option D qualification artifact lacks telemetry or build-provenance "
            "self-description"
        )
    resource = resource_export()
    provenance = provenance_export()
    if not isinstance(resource, Mapping) or not isinstance(provenance, Mapping):
        raise CampaignProviderError(
            "Option D qualification artifact self-description is malformed"
        )
    provenance = dict(provenance)
    if (
        resource.get("option_d_m0_telemetry_enabled") is not True
        or provenance.get("schema_version")
        != "eiel_authoritative_state_binary_build_provenance_v1"
        or not isinstance(provenance.get("build_source_commit"), str)
        or len(provenance["build_source_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in provenance["build_source_commit"]
        )
        or not _valid_sha256(provenance.get("compiled_source_snapshot_sha256"))
        or resource.get("authoritative_state_build_provenance_v1") != provenance
    ):
        raise CampaignProviderError(
            "Option D qualification artifact telemetry or build provenance is invalid"
        )
    if provenance["build_source_commit"] != source_environment.get(
        "source_commit"
    ) or provenance["compiled_source_snapshot_sha256"] != source_environment.get(
        "compiled_source_snapshot_sha256"
    ):
        raise CampaignProviderError(
            "Option D loaded artifact source commit or compiled source snapshot "
            "differs from the clean source/environment receipt"
        )
    artifacts = plan.get("artifacts")
    extension = (
        artifacts.get("cuda_extension") if isinstance(artifacts, Mapping) else None
    )
    if (
        not isinstance(extension, Mapping)
        or not isinstance(extension.get("path"), str)
        or not extension["path"]
        or not _valid_sha256(extension.get("sha256"))
    ):
        raise CampaignProviderError(
            "Option D qualification lacks the loaded CUDA artifact identity"
        )
    extension_path = Path(extension["path"])
    if (
        not extension_path.is_absolute()
        or not extension_path.is_file()
        or _sha256_file_v1(extension_path) != extension["sha256"]
    ):
        raise CampaignProviderError(
            "Option D loaded CUDA artifact path or SHA-256 differs"
        )
    loaded_module_path_raw = getattr(module, "__file__", None)
    loaded_module_path = (
        Path(loaded_module_path_raw).resolve()
        if isinstance(loaded_module_path_raw, str)
        else Path()
    )
    if loaded_module_path != extension_path.resolve():
        raise CampaignProviderError(
            "Option D loaded module file differs from the qualified CUDA artifact"
        )
    qualification = (
        artifacts.get("qualification_receipt")
        if isinstance(artifacts, Mapping)
        else None
    )
    if (
        not isinstance(qualification, Mapping)
        or not isinstance(qualification.get("path"), str)
        or not qualification["path"]
        or not _valid_sha256(qualification.get("sha256"))
    ):
        raise CampaignProviderError(
            "Option D qualification lacks its artifact qualification identity"
        )
    qualification_path = Path(qualification["path"])
    if (
        not qualification_path.is_absolute()
        or not qualification_path.is_file()
        or _sha256_file_v1(qualification_path) != qualification["sha256"]
    ):
        raise CampaignProviderError(
            "Option D artifact qualification receipt path or SHA-256 differs"
        )
    try:
        qualification_payload = json.loads(
            qualification_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            f"Option D artifact qualification receipt is unreadable: {exc}"
        ) from exc
    artifact_row = (
        qualification_payload.get("artifact")
        if isinstance(qualification_payload, Mapping)
        else None
    )
    if (
        not isinstance(qualification_payload, Mapping)
        or qualification_payload.get("schema_version")
        != PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION
        or qualification_payload.get("ok") is not True
        or qualification_payload.get("provider_id") != PRODUCTION_PROVIDER_ID
        or qualification_payload.get("profile_id") != PRODUCTION_PROFILE_ID
    ):
        raise CampaignProviderError(
            "Option D artifact qualification decision or schema is not accepted"
        )
    manifest_path = (
        Path(str(artifact_row.get("manifest_path", "")))
        if isinstance(artifact_row, Mapping)
        else Path()
    )
    manifest_sha256 = (
        artifact_row.get("manifest_sha256")
        if isinstance(artifact_row, Mapping)
        else None
    )
    if (
        not isinstance(artifact_row, Mapping)
        or artifact_row.get("path") != extension["path"]
        or artifact_row.get("sha256") != extension["sha256"]
        or not manifest_path.is_absolute()
        or not manifest_path.is_file()
        or not _valid_sha256(manifest_sha256)
        or _sha256_file_v1(manifest_path) != manifest_sha256
    ):
        raise CampaignProviderError(
            "Option D loaded artifact differs from its qualification manifest"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            f"Option D loaded artifact manifest is unreadable: {exc}"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version")
        != source_environment.get("artifact_manifest_contract", {}).get(
            "schema_version"
        )
        or manifest.get("artifact") != extension["path"]
        or manifest.get("artifact_sha256") != extension["sha256"]
        or manifest.get("authoritative_state_build_provenance_v1") != provenance
        or not isinstance(manifest.get("compiler_version"), str)
        or not isinstance(manifest.get("nvcc_version"), str)
        or not isinstance(manifest.get("cuda_arch"), str)
        or not isinstance(manifest.get("cxx_flags"), list)
        or not isinstance(manifest.get("nvcc_flags"), list)
    ):
        raise CampaignProviderError(
            "Option D loaded artifact manifest provenance or ABI is invalid"
        )
    manifest_contract = source_environment.get("artifact_manifest_contract")
    committed_snapshot = manifest.get("committed_source_snapshot")
    expected_snapshot = source_environment.get("committed_source_snapshot")
    required_manifest_fields = (
        manifest_contract.get("required_identity_fields")
        if isinstance(manifest_contract, Mapping)
        else None
    )
    required_supplemental_fields = (
        manifest_contract.get("required_supplemental_identity_fields")
        if isinstance(manifest_contract, Mapping)
        else None
    )

    def manifest_field(path: str) -> Any:
        value: Any = manifest
        for name in path.split("."):
            if not isinstance(value, Mapping) or name not in value:
                return None
            value = value[name]
        return value

    if (
        not isinstance(manifest_contract, Mapping)
        or manifest_contract.get("provider_must_validate_loaded_artifact") is not True
        or required_manifest_fields != list(_OPTION_D_M0_REQUIRED_MANIFEST_FIELDS_V1)
        or required_supplemental_fields
        != list(_OPTION_D_M0_REQUIRED_SUPPLEMENTAL_IDENTITIES_V1)
        or any(manifest_field(str(name)) is None for name in required_manifest_fields)
        or not isinstance(committed_snapshot, Mapping)
        or not isinstance(expected_snapshot, Mapping)
        or {
            name: committed_snapshot.get(name)
            for name in ("schema_version", "build_source_commit", "sources")
        }
        != {
            name: expected_snapshot.get(name)
            for name in ("schema_version", "build_source_commit", "sources")
        }
        or committed_snapshot.get("build_source_commit")
        != provenance["build_source_commit"]
        or committed_snapshot.get("compiled_source_snapshot_sha256")
        != provenance["compiled_source_snapshot_sha256"]
        or manifest.get("source_snapshot_stable") is not True
    ):
        raise CampaignProviderError(
            "Option D artifact manifest does not satisfy its source-build contract"
        )
    builder_identity = _option_d_m0_manifest_file_identity_v1(
        manifest.get("builder"),
        field="builder",
    )
    build_basis = source_environment.get("build_configuration_identity_basis")
    build_context = source_environment.get("build_context")
    manifest_cache_width = manifest.get("qualified_cache_width")
    conditional_flags = (
        build_context.get("conditional_flags")
        if isinstance(build_context, Mapping)
        else None
    )
    if (
        not isinstance(build_basis, Mapping)
        or not isinstance(build_context, Mapping)
        or not isinstance(manifest_cache_width, Mapping)
        or not isinstance(conditional_flags, Mapping)
        or builder_identity["sha256"] != build_basis.get("build_script_sha256")
        or manifest.get("module_name") != build_context.get("module_name")
        or manifest.get("cxx_flags") != build_context.get("effective_cxx_flags")
        or manifest.get("nvcc_flags") != build_context.get("effective_nvcc_flags")
        or manifest.get("cuda_arch") != build_context.get("cuda_arch")
        or manifest.get("deterministic_environment")
        != build_context.get("deterministic_environment")
        or manifest.get("parallel_patch_policy_rows_challenger")
        is not conditional_flags.get("parallel_patch_policy_rows")
        or manifest_cache_width.get("value")
        != conditional_flags.get("qualified_cache_width")
    ):
        raise CampaignProviderError(
            "Option D artifact builder or build configuration differs from the "
            "reconciled source contract"
        )
    builder_relative_suffix = PurePosixPath(str(build_basis.get("build_script", "")))
    builder_candidates = [
        PurePosixPath(str(row.get("path")))
        for row in committed_snapshot.get("sources", [])
        if isinstance(row, Mapping)
        and row.get("sha256") == builder_identity["sha256"]
        and tuple(
            PurePosixPath(str(row.get("path"))).parts[
                -len(builder_relative_suffix.parts) :
            ]
        )
        == tuple(builder_relative_suffix.parts)
    ]
    if len(builder_candidates) != 1 or not builder_relative_suffix.parts:
        raise CampaignProviderError(
            "Option D artifact builder is not uniquely present in the compiled "
            "source snapshot"
        )
    builder_repository_relative = builder_candidates[0]
    repository_root = Path(builder_identity["path"])
    for _part in builder_repository_relative.parts:
        repository_root = repository_root.parent
    if (repository_root / Path(*builder_repository_relative.parts)).resolve() != Path(
        builder_identity["path"]
    ):
        raise CampaignProviderError(
            "Option D artifact builder path does not match the Git-root-relative "
            "compiled source row"
        )
    source_rows = manifest.get("sources")
    source_hashes = manifest.get("source_hashes")
    expected_source_hashes = {
        row.get("path"): row.get("sha256")
        for row in committed_snapshot.get("sources", [])
        if isinstance(row, Mapping)
    }
    observed_source_hashes: dict[str, str] = {}
    if isinstance(source_rows, list) and isinstance(source_hashes, Mapping):
        for row in source_rows:
            if not isinstance(row, Mapping):
                continue
            raw_source_path = row.get("path")
            digest = row.get("sha256")
            source_path = (
                Path(raw_source_path) if isinstance(raw_source_path, str) else Path()
            )
            try:
                relative = source_path.resolve().relative_to(repository_root).as_posix()
            except ValueError:
                continue
            if (
                source_path.is_file()
                and _valid_sha256(digest)
                and _sha256_file_v1(source_path) == digest
                and source_hashes.get(raw_source_path) == digest
            ):
                observed_source_hashes[relative] = str(digest)
    if observed_source_hashes != expected_source_hashes:
        raise CampaignProviderError(
            "Option D artifact manifest source rows differ from the exact compiled "
            "source inventory"
        )
    compiler_identity = _option_d_m0_manifest_file_identity_v1(
        manifest.get("compiler_binary_identity"),
        field="compiler_binary_identity",
    )
    nvcc_identity = _option_d_m0_manifest_file_identity_v1(
        manifest.get("nvcc_binary_identity"),
        field="nvcc_binary_identity",
    )
    numpy_random_identity = _option_d_m0_manifest_file_identity_v1(
        manifest.get("numpy_random_lib_identity"),
        field="numpy_random_lib_identity",
    )
    linked_rows = manifest.get("direct_link_library_identities")
    if not isinstance(linked_rows, list):
        raise CampaignProviderError(
            "Option D artifact manifest lacks direct link library identities"
        )
    linked_identities = [
        {
            "link_name": row.get("link_name") if isinstance(row, Mapping) else None,
            **_option_d_m0_manifest_file_identity_v1(
                row,
                field="direct_link_library_identity",
            ),
        }
        for row in linked_rows
    ]
    if [row["link_name"] for row in linked_identities] != [
        "c10",
        "c10_cuda",
        "torch_cpu",
        "torch_cuda",
        "torch",
        "torch_python",
        "cudart",
    ]:
        raise CampaignProviderError(
            "Option D artifact manifest direct link library order is invalid"
        )
    linked_payload = {
        "schema_version": "eiel_direct_link_input_identity_v1",
        "libraries": linked_identities,
        "numpy_random_archive": numpy_random_identity,
    }
    artifact_environment = manifest.get("artifact_environment_identity")
    if (
        manifest.get("compiler_path") != compiler_identity["path"]
        or manifest.get("compiler_binary_sha256") != compiler_identity["sha256"]
        or manifest.get("nvcc_path") != nvcc_identity["path"]
        or manifest.get("nvcc_binary_sha256") != nvcc_identity["sha256"]
        or manifest.get("numpy_random_lib") != numpy_random_identity["path"]
        or manifest.get("relevant_linked_library_identity") != linked_payload
        or manifest.get("relevant_linked_library_sha256")
        != _canonical_sha256(linked_payload)
        or not isinstance(artifact_environment, Mapping)
        or manifest.get("artifact_environment_identity_sha256")
        != _canonical_sha256(dict(artifact_environment))
        or artifact_environment.get("compiler_binary_identity") != compiler_identity
        or artifact_environment.get("nvcc_binary_identity") != nvcc_identity
        or artifact_environment.get("direct_link_input_identity_sha256")
        != manifest.get("relevant_linked_library_sha256")
        or artifact_environment.get("compiler_command")
        != manifest.get("compiler_command")
        or artifact_environment.get("compiler_version")
        != manifest.get("compiler_version")
        or artifact_environment.get("nvcc_command") != manifest.get("nvcc_command")
        or artifact_environment.get("nvcc_version") != manifest.get("nvcc_version")
        or artifact_environment.get("cuda_home") != manifest.get("cuda_home")
        or artifact_environment.get("cuda_arch") != manifest.get("cuda_arch")
        or artifact_environment.get("cuda_gencode") != build_context.get("cuda_gencode")
        or artifact_environment.get("cxx_flags") != manifest.get("cxx_flags")
        or artifact_environment.get("nvcc_flags") != manifest.get("nvcc_flags")
        or artifact_environment.get("deterministic_environment")
        != manifest.get("deterministic_environment")
        or artifact_environment.get("python_executable")
        != manifest.get("python_executable")
        or artifact_environment.get("python_version") != manifest.get("python_version")
        or artifact_environment.get("torch_version") != manifest.get("torch_version")
        or artifact_environment.get("torch_cuda_version")
        != manifest.get("torch_cuda_version")
        or artifact_environment.get("numpy_version") != manifest.get("numpy_version")
    ):
        raise CampaignProviderError(
            "Option D artifact compiler, CUDA, linked-library, or build-environment "
            "identity is invalid"
        )
    telemetry_build = manifest.get("option_d_m0_telemetry_build")
    telemetry_identity = (
        telemetry_build.get("identity_sha256")
        if isinstance(telemetry_build, Mapping)
        else None
    )
    telemetry_identity_payload = (
        dict(telemetry_build) if isinstance(telemetry_build, Mapping) else {}
    )
    telemetry_identity_payload.pop("identity_sha256", None)
    compile_definition = "-DEIEL_OPTION_D_M0_TELEMETRY_ENABLED=1"
    if (
        not isinstance(telemetry_build, Mapping)
        or telemetry_build.get("schema_version")
        != "eiel_option_d_m0_telemetry_build_identity_v1"
        or telemetry_build.get("compile_time_macro")
        != "EIEL_OPTION_D_M0_TELEMETRY_ENABLED"
        or telemetry_build.get("receipt_field") != "option_d_m0_telemetry_enabled"
        or telemetry_build.get("default_enabled") is not False
        or telemetry_build.get("option_d_m0_telemetry_enabled") is not True
        or telemetry_build.get("compile_definition") != compile_definition
        or telemetry_build.get("identity_authority")
        != "compiled_binary_resource_receipt"
        or telemetry_build.get("runtime_resource_receipt_confirmed") is not True
        or telemetry_build.get("runtime_resource_receipt_sha256")
        != _canonical_sha256(dict(resource))
        or not _valid_sha256(telemetry_identity)
        or _canonical_sha256(telemetry_identity_payload) != telemetry_identity
        or manifest["cxx_flags"].count(compile_definition) != 1
        or manifest["nvcc_flags"].count(compile_definition) != 1
    ):
        raise CampaignProviderError(
            "Option D loaded artifact telemetry build identity differs from its "
            "runtime resource receipt or compile definition"
        )
    return {
        "telemetry_enabled": True,
        "build_provenance": provenance,
        "cuda_extension_path": str(extension_path),
        "cuda_extension_sha256": extension["sha256"],
        "qualification_receipt_path": str(qualification_path),
        "qualification_receipt_sha256": qualification["sha256"],
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": manifest_sha256,
        "compiler_binary_sha256": compiler_identity["sha256"],
        "nvcc_binary_sha256": nvcc_identity["sha256"],
        "relevant_linked_library_sha256": manifest["relevant_linked_library_sha256"],
        "artifact_environment_identity_sha256": manifest[
            "artifact_environment_identity_sha256"
        ],
        "option_d_m0_telemetry_build_identity_sha256": telemetry_identity,
        "artifact_toolchain_identity_sha256": _canonical_sha256(
            {
                name: manifest.get(name)
                for name in (
                    "compiler_path",
                    "compiler_version",
                    "nvcc_path",
                    "nvcc_version",
                    "cuda_home",
                    "cuda_arch",
                    "torch_cuda_arch_list",
                    "compiler_binary_sha256",
                    "nvcc_binary_sha256",
                    "relevant_linked_library_sha256",
                    "artifact_environment_identity_sha256",
                )
            }
        ),
        "artifact_abi_flags_identity_sha256": _canonical_sha256(
            {
                "cxx_flags": manifest["cxx_flags"],
                "nvcc_flags": manifest["nvcc_flags"],
                "link_flags": manifest.get("link_flags"),
                "torch_version": manifest.get("torch_version"),
                "torch_cuda_version": manifest.get("torch_cuda_version"),
                "numpy_version": manifest.get("numpy_version"),
            }
        ),
        "resource_receipt_build_provenance_exact": True,
    }


def _option_d_m0_combined_environment_identity_v1(
    *,
    source_environment: Mapping[str, Any],
    loaded_artifact_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind interpreter/runtime and the loaded extension into one identity."""

    source_commit = source_environment.get("source_commit")
    source_identity = source_environment.get(
        "source_interpreter_environment_identity_sha256"
    )
    source_components = source_environment.get(
        "source_interpreter_component_identity_sha256s"
    )
    build_provenance = loaded_artifact_contract.get("build_provenance")
    loaded_component_fields = {
        "loaded_extension": "cuda_extension_sha256",
        "artifact_qualification_receipt": "qualification_receipt_sha256",
        "artifact_manifest": "artifact_manifest_sha256",
        "compiler_binary": "compiler_binary_sha256",
        "nvcc_binary": "nvcc_binary_sha256",
        "direct_link_inputs": "relevant_linked_library_sha256",
        "artifact_build_environment": "artifact_environment_identity_sha256",
        "option_d_m0_telemetry_build": ("option_d_m0_telemetry_build_identity_sha256"),
        "artifact_toolchain": "artifact_toolchain_identity_sha256",
        "artifact_abi_flags": "artifact_abi_flags_identity_sha256",
    }
    if (
        not isinstance(source_commit, str)
        or not _valid_sha256(source_identity)
        or not isinstance(source_components, Mapping)
        or not source_components
        or any(
            not isinstance(name, str)
            or not name
            or not _valid_sha256(component_identity)
            for name, component_identity in source_components.items()
        )
        or source_environment.get("environment_identity_sha256") != source_identity
        or source_environment.get("environment_component_identity_sha256s")
        != source_components
        or not _valid_sha256(source_environment.get("sha256"))
        or not _valid_sha256(source_environment.get("receipt_identity_sha256"))
        or loaded_artifact_contract.get("telemetry_enabled") is not True
        or not isinstance(build_provenance, Mapping)
        or build_provenance.get("build_source_commit") != source_commit
        or any(
            not _valid_sha256(loaded_artifact_contract.get(field))
            for field in loaded_component_fields.values()
        )
    ):
        raise CampaignProviderError(
            "Option D combined environment identity lacks a source-bound "
            "interpreter or loaded-artifact component"
        )
    component_identities = {
        "source_interpreter_runtime_manifest": str(source_identity),
        **{
            f"source_interpreter.{name}": str(component_identity)
            for name, component_identity in sorted(source_components.items())
        },
        **{
            name: str(loaded_artifact_contract[field])
            for name, field in loaded_component_fields.items()
        },
        "artifact_build_provenance": _canonical_sha256(dict(build_provenance)),
    }
    manifest = {
        "schema_version": OPTION_D_M0_COMBINED_ENVIRONMENT_SCHEMA_VERSION,
        "source_commit": source_commit,
        "source_environment_receipt_identity_sha256": source_environment[
            "receipt_identity_sha256"
        ],
        "source_environment_receipt_file_sha256": source_environment["sha256"],
        "component_identity_sha256s": dict(sorted(component_identities.items())),
    }
    return {
        "schema_version": OPTION_D_M0_COMBINED_ENVIRONMENT_SCHEMA_VERSION,
        "environment_identity_sha256": _canonical_sha256(manifest),
        "component_identity_sha256s": manifest["component_identity_sha256s"],
        "manifest": manifest,
    }


def _option_d_m0_begin_cuda_memory_measurement_v1(*, execution_device: str) -> str:
    if not execution_device.startswith("cuda"):
        raise CampaignProviderError(
            "Option D M0 qualification requires a CUDA execution device"
        )
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.reset_peak_memory_stats(execution_device)
    except Exception as exc:
        raise CampaignProviderError(
            f"Option D M0 CUDA memory measurement could not start: {exc}"
        ) from exc
    return execution_device


def _option_d_m0_process_memory_measurement_v1() -> tuple[int, int]:
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            name, separator, raw = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                pieces = raw.split()
                if len(pieces) != 2 or pieces[1] != "kB":
                    raise ValueError(f"invalid {name} row")
                fields[name] = int(pieces[0]) * 1024
        rss = fields["VmRSS"]
        hwm = fields["VmHWM"]
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        raise CampaignProviderError(
            f"Option D M0 process memory measurement failed: {exc}"
        ) from exc
    if rss < 0 or hwm < rss:
        raise CampaignProviderError(
            "Option D M0 process RSS/HWM measurement is incoherent"
        )
    return rss, hwm


def _option_d_m0_finish_memory_measurement_v1(
    *, execution_device: str
) -> dict[str, Any]:
    process_rss_bytes, process_hwm_bytes = _option_d_m0_process_memory_measurement_v1()
    try:
        import torch

        resident_vram_bytes = int(torch.cuda.memory_allocated(execution_device))
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(execution_device))
    except Exception as exc:
        raise CampaignProviderError(
            f"Option D M0 CUDA memory measurement failed: {exc}"
        ) from exc
    if resident_vram_bytes < 0 or peak_vram_bytes < resident_vram_bytes:
        raise CampaignProviderError(
            "Option D M0 resident/peak CUDA allocation measurement is incoherent"
        )
    return {
        "process_rss_bytes": process_rss_bytes,
        "process_hwm_bytes": process_hwm_bytes,
        "resident_vram_bytes": resident_vram_bytes,
        "peak_vram_bytes": peak_vram_bytes,
        "process_memory_measurement_scope": (
            "linux_process_boundary_vmrss_and_cumulative_vmhwm"
        ),
        "vram_measurement_scope": (
            "torch_cuda_allocator_segment_final_allocated_and_peak_since_provider_reset"
        ),
    }


def _option_d_m0_runtime_plan_binding_v1(
    *,
    plan: Mapping[str, Any],
    request: "ResidentBatchRequestV1",
    native_receipt: Mapping[str, Any],
    source_environment: Mapping[str, Any],
    start_generation: int,
    end_generation: int,
) -> dict[str, Any]:
    """Bind M0 measurement authority to the executed W32 g1 route."""

    plan_sha256 = plan.get("plan_sha256")
    qualification_key_sha256 = plan.get("qualification_key_sha256")
    execution_route_sha256 = request.execution_route_sha256
    route_scope_identity = source_environment.get(
        "route_scope_contract_identity_sha256"
    )
    measurement_gate_identity = source_environment.get(
        "measurement_gate_identity_sha256"
    )
    cells = plan.get("cells")
    execution = plan.get("execution")
    if (
        not _valid_sha256(plan_sha256)
        or not _valid_sha256(qualification_key_sha256)
        or not _valid_sha256(execution_route_sha256)
        or not _valid_sha256(route_scope_identity)
        or not _valid_sha256(measurement_gate_identity)
        or plan.get("resolved_profile") != PRODUCTION_PROFILE_ID
        or request.profile_id != PRODUCTION_PROFILE_ID
        or not isinstance(cells, list)
        or not isinstance(execution, Mapping)
        or type(execution.get("physical_window_generations")) is not int
        or execution["physical_window_generations"] != 1
        or len(request.cell_ids) != len(set(request.cell_ids))
        or any(
            not isinstance(cell_id, str) or not cell_id for cell_id in request.cell_ids
        )
        or type(request.generations) is not int
        or request.generations <= 0
        or type(request.ticks_per_generation) is not int
        or request.ticks_per_generation <= 0
        or type(start_generation) is not int
        or type(end_generation) is not int
        or start_generation < 0
        or end_generation <= start_generation
    ):
        raise CampaignProviderError(
            "Option D M0 runtime plan lacks exact production W32 g1 authority"
        )

    cells_by_id: dict[str, Mapping[str, Any]] = {}
    for row in cells:
        cell_id = row.get("cell_id") if isinstance(row, Mapping) else None
        if not isinstance(cell_id, str) or not cell_id or cell_id in cells_by_id:
            raise CampaignProviderError(
                "Option D M0 runtime plan cell identities are malformed or duplicated"
            )
        cells_by_id[cell_id] = row
    requested_cells: list[Mapping[str, Any]] = []
    for cell_id in request.cell_ids:
        row = cells_by_id.get(cell_id)
        parameters = row.get("parameters") if isinstance(row, Mapping) else None
        if (
            not isinstance(parameters, Mapping)
            or type(parameters.get("patch.grid_size")) is not int
            or parameters["patch.grid_size"] != 32
        ):
            raise CampaignProviderError(
                "Option D M0 runtime plan is outside the W32 route scope"
            )
        requested_cells.append(row)

    expected_native = {
        "owner": "native_extension",
        "physical_window_generations": 1,
        "logical_generation_count": end_generation - start_generation,
        "external_python_window_boundaries": 0,
        "native_window_loop_owner": "native_extension",
        "internal_pybind_redispatch": False,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    native_types_exact = all(
        type(native_receipt.get(field)) is type(expected)
        and native_receipt[field] == expected
        for field, expected in expected_native.items()
    )
    if not native_types_exact:
        raise CampaignProviderError(
            "Option D M0 native execution differs from the accepted g1 owner route"
        )

    binding: dict[str, Any] = {
        "schema_version": "eiel_option_d_m0_provider_runtime_plan_binding_v1",
        "status": "PASS_EXACT_PROVIDER_RUNTIME_PLAN_BINDING",
        "authority": "provider_observed_plan_request_and_native_result",
        "plan_identity_sha256": plan_sha256,
        "qualification_key_sha256": qualification_key_sha256,
        "provider_id": PRODUCTION_PROVIDER_ID,
        "profile_id": PRODUCTION_PROFILE_ID,
        "campaign_id": plan.get("campaign_id"),
        "batch_id": request.batch_id,
        "cell_ids": list(request.cell_ids),
        "execution_route_sha256": execution_route_sha256,
        "requested_generations": request.generations,
        "ticks_per_generation": request.ticks_per_generation,
        "segment_start_generation": start_generation,
        "segment_end_generation": end_generation,
        "physical_window_generations": 1,
        "world_grid_size": 32,
        "controller_family": "fixed_mlp",
        "policy_weight_mutation_enabled": False,
        "policy_route_authority": (
            "canonical_source_and_terminal_checkpoint_projection_validation"
        ),
        "native_execution": expected_native,
        "source_route_scope_contract_identity_sha256": route_scope_identity,
        "source_measurement_gate_identity_sha256": measurement_gate_identity,
        "production_integration_authority": False,
    }
    if (
        not isinstance(binding["campaign_id"], str)
        or not binding["campaign_id"]
        or len(requested_cells) != len(request.cell_ids)
    ):
        raise CampaignProviderError(
            "Option D M0 runtime plan campaign or cell binding is incomplete"
        )
    binding["runtime_plan_binding_identity_sha256"] = _canonical_sha256(binding)
    return binding


def _validate_option_d_m0_runtime_plan_binding_v1(
    value: Any,
    *,
    identity_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt lacks its runtime plan binding"
        )
    binding = dict(value)
    digest = binding.pop("runtime_plan_binding_identity_sha256", None)
    native = binding.get("native_execution")
    if (
        binding.get("schema_version")
        != "eiel_option_d_m0_provider_runtime_plan_binding_v1"
        or binding.get("status") != "PASS_EXACT_PROVIDER_RUNTIME_PLAN_BINDING"
        or binding.get("authority")
        != "provider_observed_plan_request_and_native_result"
        or binding.get("provider_id") != PRODUCTION_PROVIDER_ID
        or binding.get("profile_id") != PRODUCTION_PROFILE_ID
        or type(binding.get("physical_window_generations")) is not int
        or binding["physical_window_generations"] != 1
        or type(binding.get("world_grid_size")) is not int
        or binding["world_grid_size"] != 32
        or binding.get("controller_family") != "fixed_mlp"
        or binding.get("policy_weight_mutation_enabled") is not False
        or binding.get("production_integration_authority") is not False
        or not isinstance(native, Mapping)
        or native.get("owner") != "native_extension"
        or native.get("native_window_loop_owner") != "native_extension"
        or type(native.get("physical_window_generations")) is not int
        or native["physical_window_generations"] != 1
        or type(native.get("logical_generation_count")) is not int
        or native["logical_generation_count"] <= 0
        or type(native.get("external_python_window_boundaries")) is not int
        or native["external_python_window_boundaries"] != 0
        or native.get("internal_pybind_redispatch") is not False
        or native.get("direct_cpp_window_core_dispatch") is not True
        or native.get("production_native_chunk_loop") is not True
        or native.get("hidden_fallback") is not False
        or native.get("runtime_jit") is not False
        or not _valid_sha256(digest)
        or _canonical_sha256(binding) != digest
        or identity_bindings.get("plan_identity_sha256")
        != binding.get("plan_identity_sha256")
        or identity_bindings.get("runtime_plan_binding_identity_sha256") != digest
    ):
        raise CampaignProviderError(
            "Option D RNG instrumentation runtime plan binding is invalid"
        )
    return {**binding, "runtime_plan_binding_identity_sha256": digest}


def _option_d_m0_provider_identity_bindings_v1(
    *,
    plan: Mapping[str, Any],
    request: "ResidentBatchRequestV1",
    segment_ordinal: int,
    start_generation: int,
    end_generation: int,
    source_commit: str,
    environment_identity_sha256: str,
    terminal_projection_digest_sha256s: Mapping[str, str],
    canonical_terminal_state_digest_sha256s: Mapping[str, str],
    checkpoint_identity_contract: Mapping[str, Any],
    runtime_plan_binding: Mapping[str, Any],
    settlement_event_rows_sha256: str,
) -> tuple[dict[str, Any], str]:
    campaign_id = plan.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise CampaignProviderError("Option D run identity lacks campaign_id")
    run_identity = _canonical_sha256(
        {
            "schema_version": "eiel_option_d_m0_run_identity_v1",
            "campaign_id": campaign_id,
            "batch_id": request.batch_id,
            "profile_id": request.profile_id,
            "execution_route_sha256": request.execution_route_sha256,
            "requested_generations": request.generations,
            "ticks_per_generation": request.ticks_per_generation,
        }
    )
    cell_identity = _canonical_sha256(
        {
            "schema_version": "eiel_option_d_m0_cell_identity_v1",
            "cell_ids": list(request.cell_ids),
        }
    )
    boundary_identity = _canonical_sha256(
        {
            "schema_version": "eiel_option_d_m0_generation_tick_boundary_v1",
            "segment_ordinal": segment_ordinal,
            "start_generation": start_generation,
            "end_generation": end_generation,
            "ticks_per_generation": request.ticks_per_generation,
        }
    )
    terminal_state_identity = _canonical_sha256(
        dict(sorted(canonical_terminal_state_digest_sha256s.items()))
    )
    checkpoint_identity = _canonical_sha256(dict(checkpoint_identity_contract))
    plan_identity = runtime_plan_binding.get("plan_identity_sha256")
    runtime_plan_binding_identity = runtime_plan_binding.get(
        "runtime_plan_binding_identity_sha256"
    )
    if not _valid_sha256(plan_identity) or not _valid_sha256(
        runtime_plan_binding_identity
    ):
        raise CampaignProviderError(
            "Option D provider identity lacks its runtime plan binding"
        )
    if not _valid_sha256(settlement_event_rows_sha256):
        raise CampaignProviderError(
            "Option D provider identity lacks settlement-event evidence"
        )
    output_digest = _canonical_sha256(
        {
            "schema_version": "eiel_option_d_m0_output_identity_v2",
            "terminal_projection_digest_sha256s": dict(
                sorted(terminal_projection_digest_sha256s.items())
            ),
            "settlement_event_rows_sha256": settlement_event_rows_sha256,
        }
    )
    identity = {
        "run_identity_sha256": run_identity,
        "cell_identity_sha256": cell_identity,
        "generation_tick_boundary_identity_sha256": boundary_identity,
        "plan_identity_sha256": plan_identity,
        "runtime_plan_binding_identity_sha256": runtime_plan_binding_identity,
        "source_commit": source_commit,
        "environment_identity_sha256": environment_identity_sha256,
        "terminal_digest_sha256": terminal_state_identity,
        "checkpoint_identity_sha256": checkpoint_identity,
    }
    return (
        _validate_option_d_m0_shared_identity_v1(
            identity,
            receipt_name="provider-finalized Option D M0 receipt",
        ),
        output_digest,
    )


def _validate_option_d_m0_shared_identity_v1(
    value: Any,
    *,
    receipt_name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignProviderError(
            f"{receipt_name} lacks Option D shared identity bindings"
        )
    identity = dict(value)
    for field in _OPTION_D_M0_SHARED_IDENTITY_FIELDS_V1:
        item = identity.get(field)
        valid = _valid_sha256(item)
        if field == "source_commit":
            valid = (
                isinstance(item, str)
                and len(item) == 40
                and all(character in "0123456789abcdef" for character in item)
            )
        if not valid:
            raise CampaignProviderError(
                f"{receipt_name} Option D shared identity field is invalid: {field}"
            )
    return identity


def _validate_option_d_m0_cpu_policy_service_raw_metrics_v1(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignProviderError(
            "Option D qualification requires exact CPU policy-service raw metrics"
        )
    raw = dict(value)
    expected_fields = set(_OPTION_D_M0_CPU_POLICY_SERVICE_RAW_INTEGER_FIELDS_V1) | {
        "host_allocation_measurement_scope",
        "transfer_operation_measurement_scope",
        "synchronization_measurement_scope",
    }
    if set(raw) != expected_fields:
        raise CampaignProviderError(
            "Option D CPU policy-service raw metric fields differ from the "
            "qualification contract"
        )
    for field in _OPTION_D_M0_CPU_POLICY_SERVICE_RAW_INTEGER_FIELDS_V1:
        item = raw[field]
        if type(item) is not int or item < 0:
            raise CampaignProviderError(
                f"Option D CPU policy-service raw metric is invalid: {field}"
            )
    if (
        raw["host_allocation_measurement_scope"]
        != OPTION_D_M0_HOST_ALLOCATION_MEASUREMENT_SCOPE
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service host-allocation measurement scope is invalid"
        )
    if (
        raw["transfer_operation_measurement_scope"]
        != OPTION_D_M0_TRANSFER_OPERATION_MEASUREMENT_SCOPE
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service transfer-operation measurement scope is "
            "invalid"
        )
    if (
        raw["synchronization_measurement_scope"]
        != OPTION_D_M0_SYNCHRONIZATION_MEASUREMENT_SCOPE
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service synchronization measurement scope is invalid"
        )
    return raw


def _option_d_m0_projection_policy_counters_v1(
    projection: Any,
    *,
    cell_id: str,
    generation: int,
    boundary_name: str,
) -> tuple[int, int, int, int]:
    if not isinstance(projection, Mapping):
        raise CampaignProviderError(
            f"Option D {boundary_name} projection is malformed for {cell_id}"
        )
    metadata = projection.get("metadata")
    tensors = projection.get("tensors")
    mutation_std = (
        metadata.get("policy_weight_mutation_std")
        if isinstance(metadata, Mapping)
        else None
    )
    rng_semantics_version = (
        metadata.get("tw_rngv2_semantics_version", 1)
        if isinstance(metadata, Mapping)
        else None
    )
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("generation") != generation
        or metadata.get("tick") != 0
        or metadata.get("native_runtime_fixed_policy_family") != "mlp"
        or metadata.get("policy_weight_mutation_enable") is not False
        or isinstance(mutation_std, bool)
        or not isinstance(mutation_std, (int, float))
        or not math.isfinite(float(mutation_std))
        or float(mutation_std) != 0.0
        or rng_semantics_version not in {1, 2}
        or not isinstance(tensors, Mapping)
    ):
        raise CampaignProviderError(
            "Option D D1 counter derivation requires an exact mutation-disabled "
            f"MLP generation boundary for {cell_id} ({boundary_name})"
        )
    try:
        import torch
    except ImportError as exc:
        raise CampaignProviderError(
            "Option D checkpoint counter derivation requires PyTorch"
        ) from exc
    mlp_pack = tensors.get("mlp_pack")
    if (
        not isinstance(mlp_pack, torch.Tensor)
        or mlp_pack.device.type != "cpu"
        or mlp_pack.dtype != torch.float32
        or not mlp_pack.is_contiguous()
        or mlp_pack.ndim < 1
    ):
        raise CampaignProviderError(
            "Option D canonical checkpoint D1 stream index or logical MLP policy "
            f"row width is invalid for {cell_id} ({boundary_name})"
        )
    logical_width = int(mlp_pack.numel() if mlp_pack.ndim == 1 else mlp_pack.shape[-1])
    if logical_width <= 0:
        raise CampaignProviderError(
            "Option D canonical checkpoint D1 stream index or logical MLP policy "
            f"row width is invalid for {cell_id} ({boundary_name})"
        )

    def optional_scalar_counter(field: str) -> int | None:
        tensor = tensors.get(field)
        if tensor is None:
            return None
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.int64
            or tensor.numel() != 1
            or not tensor.is_contiguous()
        ):
            raise CampaignProviderError(
                "Option D canonical checkpoint counter is malformed: "
                f"{cell_id}.{boundary_name}.{field}"
            )
        item = int(tensor.reshape(()).item())
        if item < 0:
            raise CampaignProviderError(
                "Option D canonical checkpoint counter is negative: "
                f"{cell_id}.{boundary_name}.{field}"
            )
        return item

    explicit_cursor = optional_scalar_counter("policy_birth_tape_cursor")
    explicit_semantic_draws = optional_scalar_counter(
        "mutation_torch_cpu_semantic_draws"
    )
    streams = metadata.get("native_runtime_rng_manager_stream_indices")
    mutation_stream = streams.get("mutation") if isinstance(streams, Mapping) else None
    metadata_semantic_draws = (
        mutation_stream.get("torch_cpu")
        if isinstance(mutation_stream, Mapping)
        else None
    )
    if (
        type(metadata_semantic_draws) is not int
        or metadata_semantic_draws < 0
        or (
            explicit_semantic_draws is not None
            and metadata_semantic_draws != explicit_semantic_draws
        )
    ):
        raise CampaignProviderError(
            "Option D canonical checkpoint D1 stream index or logical MLP policy "
            f"row width is invalid for {cell_id} ({boundary_name})"
        )
    semantic_draws = metadata_semantic_draws
    if explicit_cursor is None:
        if semantic_draws % logical_width != 0:
            raise CampaignProviderError(
                "Option D canonical checkpoint D1 stream index or logical MLP policy "
                f"row width is invalid for {cell_id} ({boundary_name})"
            )
        cursor = semantic_draws // logical_width
    else:
        cursor = explicit_cursor
    for candidate in (
        metadata.get("policy_birth_crossover_mask_logical_width"),
        tensors.get("policy_birth_crossover_mask_logical_width"),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, torch.Tensor):
            if (
                candidate.device.type != "cpu"
                or candidate.dtype != torch.int64
                or candidate.numel() != 1
                or not candidate.is_contiguous()
            ):
                raise CampaignProviderError(
                    "Option D canonical checkpoint explicit crossover width is "
                    "malformed"
                )
            candidate = int(candidate.reshape(()).item())
        if type(candidate) is not int or candidate != logical_width:
            raise CampaignProviderError(
                "Option D canonical checkpoint explicit crossover width differs "
                "from the MLP policy row"
            )
    return cursor, semantic_draws, logical_width, rng_semantics_version


def _option_d_m0_projection_lifecycle_cursor_v1(
    projection: Mapping[str, Any],
    *,
    expected_cursor: int,
) -> Any:
    public_cursor = projection.get("event_cursor")
    if public_cursor == expected_cursor:
        return public_cursor
    runtime_cursor = projection.get("runtime_event_cursor")
    if runtime_cursor == expected_cursor:
        return runtime_cursor
    return public_cursor


def _option_d_m0_project_native_turnover_refill_rows_v1(
    direct_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    counts_by_window = direct_result.get("turnover_counts_by_window")
    terminal_generations = direct_result.get("turnover_count_terminal_generations")
    if counts_by_window is None and terminal_generations is None:
        return []

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure is environment fatal.
        raise CampaignProviderError(
            "Option D M0 cannot project native turnover counters without torch"
        ) from exc

    if (
        not isinstance(counts_by_window, torch.Tensor)
        or counts_by_window.device.type != "cpu"
        or counts_by_window.dtype != torch.int64
        or counts_by_window.ndim != 3
        or counts_by_window.size(-1) != 8
        or not isinstance(terminal_generations, Sequence)
        or isinstance(terminal_generations, (str, bytes, bytearray))
        or len(terminal_generations) != counts_by_window.size(0)
        or any(type(generation) is not int for generation in terminal_generations)
    ):
        raise CampaignProviderError(
            "Option D M0 native turnover refill counters are malformed"
        )
    rows: list[dict[str, Any]] = []
    seen_generations: set[int] = set()
    counts = counts_by_window.contiguous()
    for window_index, generation in enumerate(terminal_generations):
        if generation in seen_generations:
            raise CampaignProviderError(
                "Option D M0 native turnover generation metadata is duplicated"
            )
        seen_generations.add(generation)
        committed = [int(value) for value in counts[window_index, :, 6].tolist()]
        if any(value < 0 for value in committed):
            raise CampaignProviderError(
                "Option D M0 native turnover refill count is negative"
            )
        rows.append(
            {
                "terminal_generation": generation,
                "committed_refill_counts_by_cell": committed,
                "counter_source": "turnover_counts_by_window_column_6",
            }
        )
    return rows


def _option_d_m0_native_turnover_refill_counts_v1(
    native_receipt: Mapping[str, Any],
    *,
    cell_count: int,
    start_generation: int,
    end_generation: int,
) -> list[int]:
    compact_rows = native_receipt.get(_OPTION_D_M0_NATIVE_TURNOVER_REFILL_ROWS_FIELD_V1)
    if compact_rows is not None:
        if not isinstance(compact_rows, list) or any(
            not isinstance(row, Mapping) for row in compact_rows
        ):
            raise CampaignProviderError(
                "Option D M0 native turnover refill rows are malformed"
            )
        totals = [0 for _ in range(cell_count)]
        seen_generations: set[int] = set()
        for row in compact_rows:
            generation = row.get("terminal_generation")
            committed = row.get("committed_refill_counts_by_cell")
            if (
                type(generation) is not int
                or generation in seen_generations
                or not isinstance(committed, list)
                or len(committed) != cell_count
                or any(type(value) is not int or value < 0 for value in committed)
                or row.get("counter_source") != "turnover_counts_by_window_column_6"
            ):
                raise CampaignProviderError(
                    "Option D M0 native turnover refill rows are malformed"
                )
            seen_generations.add(generation)
            if start_generation <= generation < end_generation:
                for cell_index, value in enumerate(committed):
                    totals[cell_index] += value
        return totals

    counts_by_window = native_receipt.get("turnover_counts_by_window")
    terminal_generations = native_receipt.get("turnover_count_terminal_generations")
    if counts_by_window is None and terminal_generations is None:
        return [0 for _ in range(cell_count)]

    try:
        import torch
    except Exception as exc:  # pragma: no cover - import failure is environment fatal.
        raise CampaignProviderError(
            "Option D M0 cannot validate native turnover counters without torch"
        ) from exc

    if (
        not isinstance(counts_by_window, torch.Tensor)
        or counts_by_window.device.type != "cpu"
        or counts_by_window.dtype != torch.int64
        or counts_by_window.ndim not in {2, 3}
        or counts_by_window.size(-1) != 8
    ):
        raise CampaignProviderError(
            "Option D M0 native turnover refill counters are malformed"
        )
    counts = counts_by_window.contiguous()
    if counts.ndim == 2:
        if terminal_generations is not None:
            raise CampaignProviderError(
                "Option D M0 native turnover generation metadata is malformed"
            )
        if counts.size(0) != cell_count:
            raise CampaignProviderError(
                "Option D M0 native turnover counter cell count differs"
            )
        return [int(value) for value in counts[:, 6].tolist()]

    if counts.size(1) != cell_count:
        raise CampaignProviderError(
            "Option D M0 native turnover counter cell count differs"
        )
    if (
        not isinstance(terminal_generations, Sequence)
        or isinstance(terminal_generations, (str, bytes, bytearray))
        or len(terminal_generations) != counts.size(0)
        or any(type(generation) is not int for generation in terminal_generations)
    ):
        raise CampaignProviderError(
            "Option D M0 native turnover generation metadata is malformed"
        )
    totals = [0 for _ in range(cell_count)]
    seen_generations: set[int] = set()
    for window_index, generation in enumerate(terminal_generations):
        if generation in seen_generations:
            raise CampaignProviderError(
                "Option D M0 native turnover generation metadata is duplicated"
            )
        seen_generations.add(generation)
        if start_generation <= generation < end_generation:
            for cell_index, value in enumerate(counts[window_index, :, 6].tolist()):
                count = int(value)
                if count < 0:
                    raise CampaignProviderError(
                        "Option D M0 native turnover refill count is negative"
                    )
                totals[cell_index] += count
    return totals


def _option_d_m0_settled_per_cell_measurements_v1(
    *,
    request: "ResidentBatchRequestV1",
    native_receipt: Mapping[str, Any],
    source_projections: Sequence[Mapping[str, Any]],
    terminal_projections: Sequence[Mapping[str, Any]],
    settlement_event_rows: Sequence[Mapping[str, Any]],
    start_generation: int,
    end_generation: int,
    turnover_policy_crossover_elided: bool,
) -> dict[str, Any]:
    cell_ids = tuple(request.cell_ids)
    generation_count = end_generation - start_generation
    if (
        generation_count <= 0
        or len(source_projections) != len(cell_ids)
        or len(terminal_projections) != len(cell_ids)
        or len(settlement_event_rows) != len(cell_ids) * generation_count
    ):
        raise CampaignProviderError(
            "Option D settled evidence dimensions differ from the generation boundary"
        )
    native_turnover_refill_by_cell = _option_d_m0_native_turnover_refill_counts_v1(
        native_receipt,
        cell_count=len(cell_ids),
        start_generation=start_generation,
        end_generation=end_generation,
    )
    ordinary_by_cell = {cell_id: 0 for cell_id in cell_ids}
    turnover_by_cell = {cell_id: 0 for cell_id in cell_ids}
    lifecycle_cursor_start_by_cell: dict[str, int] = {}
    lifecycle_cursor_end_by_cell: dict[str, int] = {}
    event_type_names = {
        "birth_committed",
        "mutation",
        "birth_failed",
        "death_committed",
    }
    previous_cursor_by_cell: dict[str, int] = {}
    expected_lattice = tuple(
        (cell_id, generation)
        for cell_id in cell_ids
        for generation in range(start_generation, end_generation)
    )
    for row, (expected_cell_id, expected_generation) in zip(
        settlement_event_rows,
        expected_lattice,
        strict=True,
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "schema_version",
            "cell_id",
            "generation",
            "events",
        }:
            raise CampaignProviderError("Option D settlement event rows are malformed")
        cell_id = row.get("cell_id")
        row_generation = row.get("generation")
        events = row.get("events")
        generation_counts = (
            events.get("generation_counts_by_type")
            if isinstance(events, Mapping)
            else None
        )
        turnover_counts = (
            events.get("turnover_counts_by_type")
            if isinstance(events, Mapping)
            else None
        )
        ordinary_births = (
            generation_counts.get("birth_committed")
            if isinstance(generation_counts, Mapping)
            else None
        )
        turnover_births = (
            turnover_counts.get("birth_committed")
            if isinstance(turnover_counts, Mapping)
            else None
        )
        expected_event_fields = {
            "cursor_start",
            "cursor_pre_turnover",
            "cursor_end",
            "authoritative_cursor",
            "event_row_count",
            "generation_event_count",
            "turnover_event_count",
            "generation_counts_by_type",
            "turnover_counts_by_type",
        }
        integer_fields = (
            "cursor_start",
            "cursor_pre_turnover",
            "cursor_end",
            "authoritative_cursor",
            "event_row_count",
            "generation_event_count",
            "turnover_event_count",
        )
        if (
            row.get("schema_version") != "eiel_direct_resident_settlement_event_row_v1"
            or cell_id != expected_cell_id
            or type(row_generation) is not int
            or row_generation != expected_generation
            or not isinstance(events, Mapping)
            or set(events) != expected_event_fields
            or any(
                type(events.get(field)) is not int or events[field] < 0
                for field in integer_fields
            )
            or not isinstance(generation_counts, Mapping)
            or set(generation_counts) != event_type_names
            or not isinstance(turnover_counts, Mapping)
            or set(turnover_counts) != event_type_names
            or any(
                type(value) is not int or value < 0
                for value in (
                    *generation_counts.values(),
                    *turnover_counts.values(),
                )
            )
            or type(ordinary_births) is not int
            or ordinary_births < 0
            or type(turnover_births) is not int
            or turnover_births < 0
        ):
            raise CampaignProviderError(
                "Option D settlement event rows are incomplete, duplicated, "
                "or malformed"
            )
        cursor_start = events["cursor_start"]
        cursor_pre_turnover = events["cursor_pre_turnover"]
        cursor_end = events["cursor_end"]
        previous_cursor = previous_cursor_by_cell.get(cell_id)
        if (
            not cursor_start <= cursor_pre_turnover <= cursor_end
            or events["authoritative_cursor"] != cursor_end
            or events["event_row_count"] != cursor_end - cursor_start
            or events["generation_event_count"] != cursor_pre_turnover - cursor_start
            or events["turnover_event_count"] != cursor_end - cursor_pre_turnover
            or sum(generation_counts.values()) != events["generation_event_count"]
            or sum(turnover_counts.values()) != events["turnover_event_count"]
            or (previous_cursor is not None and cursor_start != previous_cursor)
        ):
            raise CampaignProviderError(
                "Option D settlement event cursor or type counts are contradictory"
            )
        previous_cursor_by_cell[cell_id] = cursor_end
        lifecycle_cursor_start_by_cell.setdefault(cell_id, cursor_start)
        lifecycle_cursor_end_by_cell[cell_id] = cursor_end
        ordinary_by_cell[cell_id] += ordinary_births
        turnover_by_cell[cell_id] += turnover_births

    cursor_starts: list[int] = []
    cursor_ends: list[int] = []
    cursor_deltas: list[int] = []
    semantic_starts: list[int] = []
    semantic_ends: list[int] = []
    semantic_deltas: list[int] = []
    state_advancing_birth_counts: list[int] = []
    policy_rng_counter_modes: list[str] = []
    rng_semantics_versions: list[int] = []
    logical_widths: list[int] = []
    lifecycle_cursor_starts: list[int] = []
    lifecycle_cursor_ends: list[int] = []
    observed_turnover_policy_crossover_elided: bool | None = None
    for cell_id, source, terminal in zip(
        cell_ids,
        source_projections,
        terminal_projections,
        strict=True,
    ):
        cursor_start, semantic_start, source_width, source_rng_version = (
            _option_d_m0_projection_policy_counters_v1(
                source,
                cell_id=cell_id,
                generation=start_generation,
                boundary_name="source",
            )
        )
        cursor_end, semantic_end, terminal_width, terminal_rng_version = (
            _option_d_m0_projection_policy_counters_v1(
                terminal,
                cell_id=cell_id,
                generation=end_generation,
                boundary_name="terminal",
            )
        )
        cursor_delta = cursor_end - cursor_start
        semantic_delta = semantic_end - semantic_start
        semantic_birth_count = (
            semantic_delta // source_width if source_width > 0 else -1
        )
        cell_index = cell_ids.index(cell_id)
        turnover_policy_births = max(
            turnover_by_cell[cell_id],
            native_turnover_refill_by_cell[cell_index],
        )
        total_birth_count = ordinary_by_cell[cell_id] + turnover_policy_births
        possible_state_advancing_birth_counts = {
            ordinary_by_cell[cell_id],
            total_birth_count,
        }
        if source_rng_version != terminal_rng_version:
            raise CampaignProviderError(
                "Option D TW RNG semantic version changed across a checkpoint span"
            )
        formula_rngv2 = source_rng_version == 2
        if turnover_policy_births > 0:
            cell_turnover_elided = semantic_birth_count == (
                0 if formula_rngv2 else ordinary_by_cell[cell_id]
            )
            if observed_turnover_policy_crossover_elided is None:
                observed_turnover_policy_crossover_elided = cell_turnover_elided
            elif observed_turnover_policy_crossover_elided != cell_turnover_elided:
                raise CampaignProviderError(
                    "Option D turnover policy crossover accounting differs across cells"
                )
        expected_lifecycle_cursor_start = lifecycle_cursor_start_by_cell.get(cell_id)
        expected_lifecycle_cursor_end = lifecycle_cursor_end_by_cell.get(cell_id)
        source_lifecycle_cursor = _option_d_m0_projection_lifecycle_cursor_v1(
            source, expected_cursor=expected_lifecycle_cursor_start
        )
        terminal_lifecycle_cursor = _option_d_m0_projection_lifecycle_cursor_v1(
            terminal, expected_cursor=expected_lifecycle_cursor_end
        )
        counter_contract_invalid = (
            cursor_delta < 0
            or semantic_delta < 0
            or semantic_delta % source_width != 0
            or source_width != terminal_width
            or (
                formula_rngv2
                and (
                    cursor_start != 0
                    or cursor_end != 0
                    or cursor_delta != 0
                    or semantic_birth_count not in {0, turnover_policy_births}
                )
            )
            or (
                not formula_rngv2
                and (
                    semantic_birth_count
                    not in possible_state_advancing_birth_counts
                    or cursor_delta != semantic_birth_count
                )
            )
        )
        if (
            counter_contract_invalid
            or type(source_lifecycle_cursor) is not int
            or source_lifecycle_cursor < 0
            or type(terminal_lifecycle_cursor) is not int
            or terminal_lifecycle_cursor < source_lifecycle_cursor
            or expected_lifecycle_cursor_start != source_lifecycle_cursor
            or expected_lifecycle_cursor_end != terminal_lifecycle_cursor
        ):
            raise CampaignProviderError(
                "Option D settled births and canonical Torch policy counters differ "
                f"for {cell_id}: "
                + json.dumps(
                    {
                        "cursor_start": cursor_start,
                        "cursor_end": cursor_end,
                        "cursor_delta": cursor_delta,
                        "semantic_start": semantic_start,
                        "semantic_end": semantic_end,
                        "semantic_delta": semantic_delta,
                        "semantic_birth_count": semantic_birth_count,
                        "ordinary_birth_count": ordinary_by_cell[cell_id],
                        "turnover_birth_count": turnover_by_cell[cell_id],
                        "native_turnover_refill_birth_count": (
                            native_turnover_refill_by_cell[cell_index]
                        ),
                        "turnover_policy_birth_count": turnover_policy_births,
                        "possible_state_advancing_birth_counts": sorted(
                            possible_state_advancing_birth_counts
                        ),
                        "rng_semantics_version": source_rng_version,
                        "source_width": source_width,
                        "terminal_width": terminal_width,
                        "source_lifecycle_cursor": source_lifecycle_cursor,
                        "terminal_lifecycle_cursor": terminal_lifecycle_cursor,
                        "expected_lifecycle_cursor_start": (
                            expected_lifecycle_cursor_start
                        ),
                        "expected_lifecycle_cursor_end": (
                            expected_lifecycle_cursor_end
                        ),
                    },
                    sort_keys=True,
                )
            )
        cursor_starts.append(cursor_start)
        cursor_ends.append(cursor_end)
        cursor_deltas.append(cursor_delta)
        semantic_starts.append(semantic_start)
        semantic_ends.append(semantic_end)
        semantic_deltas.append(semantic_delta)
        state_advancing_birth_counts.append(semantic_birth_count)
        policy_rng_counter_modes.append(
            "stateless_philox_formula_v2"
            if formula_rngv2
            else "state_advancing_torch_tape_v1"
        )
        rng_semantics_versions.append(source_rng_version)
        logical_widths.append(source_width)
        lifecycle_cursor_starts.append(source_lifecycle_cursor)
        lifecycle_cursor_ends.append(terminal_lifecycle_cursor)
    if len(set(logical_widths)) != 1:
        raise CampaignProviderError(
            "Option D canonical checkpoint D1 logical MLP policy row width differs "
            "across cells"
        )
    turnover_policy_crossover_elided_observed = (
        turnover_policy_crossover_elided
        if observed_turnover_policy_crossover_elided is None
        else observed_turnover_policy_crossover_elided
    )
    return {
        "cell_ids": list(cell_ids),
        "settlement_event_row_count": len(settlement_event_rows),
        "settlement_event_rows_sha256": _canonical_sha256(
            [dict(row) for row in settlement_event_rows]
        ),
        "actual_ordinary_birth_counts": [
            ordinary_by_cell[cell_id] for cell_id in cell_ids
        ],
        "actual_turnover_birth_counts": [
            max(turnover_by_cell[cell_id], native_turnover_refill_by_cell[index])
            for index, cell_id in enumerate(cell_ids)
        ],
        "torch_policy_state_advancing_birth_counts": state_advancing_birth_counts,
        "policy_rng_counter_modes": policy_rng_counter_modes,
        "rng_semantics_versions": rng_semantics_versions,
        "turnover_policy_crossover_elided": (turnover_policy_crossover_elided_observed),
        "policy_birth_tape_cursor_starts": cursor_starts,
        "policy_birth_tape_cursor_ends": cursor_ends,
        "policy_birth_tape_cursor_deltas": cursor_deltas,
        "mutation_torch_cpu_semantic_draw_starts": semantic_starts,
        "mutation_torch_cpu_semantic_draw_ends": semantic_ends,
        "policy_birth_crossover_logical_widths": logical_widths,
        "lifecycle_event_cursor_starts": lifecycle_cursor_starts,
        "lifecycle_event_cursor_ends": lifecycle_cursor_ends,
        "torch_policy_raw_word_deltas": list(semantic_deltas),
        "torch_policy_semantic_draw_deltas": semantic_deltas,
        "birth_measurement_scope": (
            "exact_settlement_event_rows_generation_birth_committed_and_native_"
            "turnover_refill_committed_when_available"
        ),
        "torch_counter_measurement_scope": (
            "canonical_checkpoint_source_to_terminal_policy_birth_tape_cursor_or_"
            "metadata_derived_mutation_torch_cpu_semantic_draws_with_configured_"
            "turnover_shadow_elision_accounted"
        ),
    }


def _option_d_m0_turnover_policy_crossover_elided_v1(
    plan: Mapping[str, Any],
) -> bool:
    cells = plan.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)) or not cells:
        raise CampaignProviderError(
            "Option D M0 plan lacks cells for turnover crossover accounting"
        )
    values: set[bool] = set()
    for cell in cells:
        parameters = cell.get("parameters") if isinstance(cell, Mapping) else None
        if not isinstance(parameters, Mapping):
            raise CampaignProviderError(
                "Option D M0 plan cell lacks parameters for turnover accounting"
            )
        value = parameters.get(_OPTION_D_M0_TURNOVER_MLP_SHADOW_ELIDE_PARAMETER_V1)
        if value is None:
            value = False
        if type(value) is not bool:
            raise CampaignProviderError(
                "Option D M0 turnover shadow-elide parameter must be boolean"
            )
        values.add(value)
    if len(values) != 1:
        raise CampaignProviderError(
            "Option D M0 turnover shadow-elide parameter differs across cells"
        )
    return values.pop()


def _validate_option_d_m0_primary_receipt_v1(
    value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise CampaignProviderError(
            "Option D qualification requires its linked RNG instrumentation receipt"
        )
    receipt = dict(value)
    if (
        receipt.get("schema_version") != OPTION_D_M0_PRIMARY_RECEIPT_SCHEMA_VERSION
        or receipt.get("option_d_m0_telemetry_enabled") is not True
    ):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt schema or telemetry binding is "
            "invalid"
        )
    identity = _validate_option_d_m0_shared_identity_v1(
        receipt.get("identity_bindings"),
        receipt_name="linked Option D RNG instrumentation receipt",
    )
    for field in _OPTION_D_M0_PRIMARY_INTEGER_FIELDS_V1:
        item = receipt.get(field)
        if type(item) is not int or item < 0:
            raise CampaignProviderError(
                f"Option D RNG instrumentation receipt field is invalid: {field}"
            )
    for field in _OPTION_D_M0_PRIMARY_DIGEST_FIELDS_V1:
        if not _valid_sha256(receipt.get(field)):
            raise CampaignProviderError(
                f"Option D RNG instrumentation receipt field is invalid: {field}"
            )
    if receipt["process_hwm_bytes"] < receipt["process_rss_bytes"]:
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt HWM is below current RSS"
        )
    if receipt["peak_vram_bytes"] < receipt["resident_vram_bytes"]:
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt peak VRAM is below resident VRAM"
        )
    policy_counter_contract = receipt.get("torch_policy_counter_contract")
    if not isinstance(policy_counter_contract, Mapping):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt Torch policy counter scope is invalid"
        )
    logical_policy_row_width = policy_counter_contract.get("logical_policy_row_width")
    rng_semantics_version = policy_counter_contract.get("rng_semantics_version", 1)
    if policy_counter_contract != _option_d_m0_torch_policy_counter_contract_v1(
        logical_policy_row_width,
        rng_semantics_version=rng_semantics_version,
    ):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt Torch policy counter scope is invalid"
        )
    source_environment = receipt.get("source_environment_receipt")
    loaded_artifact = receipt.get("loaded_artifact_contract")
    combined_environment = receipt.get("combined_environment_identity")
    if (
        not isinstance(source_environment, Mapping)
        or not _valid_sha256(source_environment.get("sha256"))
        or not _valid_sha256(
            source_environment.get("source_interpreter_environment_identity_sha256")
        )
        or not isinstance(
            source_environment.get("source_interpreter_component_identity_sha256s"),
            Mapping,
        )
        or any(
            not _valid_sha256(component_identity)
            for component_identity in source_environment[
                "source_interpreter_component_identity_sha256s"
            ].values()
        )
        or not isinstance(loaded_artifact, Mapping)
        or loaded_artifact.get("telemetry_enabled") is not True
        or not _valid_sha256(loaded_artifact.get("cuda_extension_sha256"))
        or not _valid_sha256(loaded_artifact.get("qualification_receipt_sha256"))
        or not _valid_sha256(loaded_artifact.get("artifact_manifest_sha256"))
        or not _valid_sha256(
            loaded_artifact.get("option_d_m0_telemetry_build_identity_sha256")
        )
        or not _valid_sha256(loaded_artifact.get("artifact_toolchain_identity_sha256"))
        or not _valid_sha256(loaded_artifact.get("artifact_abi_flags_identity_sha256"))
    ):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt source or artifact binding is invalid"
        )
    expected_combined_environment = _option_d_m0_combined_environment_identity_v1(
        source_environment=source_environment,
        loaded_artifact_contract=loaded_artifact,
    )
    if (
        not isinstance(combined_environment, Mapping)
        or dict(combined_environment) != expected_combined_environment
        or identity["environment_identity_sha256"]
        != expected_combined_environment["environment_identity_sha256"]
    ):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt combined environment identity "
            "is missing or mismatched"
        )
    _validate_option_d_m0_runtime_plan_binding_v1(
        receipt.get("runtime_plan_binding"),
        identity_bindings=identity,
    )
    for scope_field in (
        "process_memory_measurement_scope",
        "vram_measurement_scope",
        "checkpoint_projection_time_scope",
        "checkpoint_compression_time_scope",
    ):
        if not isinstance(receipt.get(scope_field), str) or not receipt[scope_field]:
            raise CampaignProviderError(
                "Option D RNG instrumentation receipt measurement scope is invalid: "
                f"{scope_field}"
            )
    per_cell = receipt.get("per_cell_measurements")
    if not isinstance(per_cell, Mapping):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt lacks per-cell measurements"
        )
    cell_ids = per_cell.get("cell_ids")
    if (
        not isinstance(cell_ids, list)
        or not cell_ids
        or len(cell_ids) != len(set(cell_ids))
        or any(not isinstance(cell_id, str) or not cell_id for cell_id in cell_ids)
    ):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt per-cell identities are invalid"
        )
    settlement_evidence = receipt.get("settlement_event_evidence")
    terminal_projection_digests = (
        settlement_evidence.get("terminal_projection_digest_sha256s")
        if isinstance(settlement_evidence, Mapping)
        else None
    )
    if (
        not isinstance(settlement_evidence, Mapping)
        or set(settlement_evidence)
        != {
            "schema_version",
            "row_count",
            "rows_sha256",
            "identity_bindings_sha256",
            "terminal_projection_digest_sha256s",
        }
        or settlement_evidence.get("schema_version")
        != "eiel_option_d_m0_settlement_event_evidence_v1"
        or type(settlement_evidence.get("row_count")) is not int
        or settlement_evidence["row_count"] <= 0
        or settlement_evidence["row_count"]
        != per_cell.get("settlement_event_row_count")
        or not _valid_sha256(settlement_evidence.get("rows_sha256"))
        or settlement_evidence["rows_sha256"]
        != per_cell.get("settlement_event_rows_sha256")
        or settlement_evidence.get("identity_bindings_sha256")
        != _canonical_sha256(identity)
        or not isinstance(terminal_projection_digests, Mapping)
        or set(terminal_projection_digests) != set(cell_ids)
        or any(
            not _valid_sha256(digest) for digest in terminal_projection_digests.values()
        )
        or receipt["output_digest_sha256"]
        != _canonical_sha256(
            {
                "schema_version": "eiel_option_d_m0_output_identity_v2",
                "terminal_projection_digest_sha256s": dict(
                    sorted(terminal_projection_digests.items())
                ),
                "settlement_event_rows_sha256": settlement_evidence["rows_sha256"],
            }
        )
    ):
        raise CampaignProviderError(
            "Option D settlement-event evidence identity is invalid"
        )
    resident_accounting = receipt.get("resident_window_transfer_accounting")
    resident_accounting_payload = (
        {
            key: value
            for key, value in resident_accounting.items()
            if key != "accounting_sha256"
        }
        if isinstance(resident_accounting, Mapping)
        else None
    )
    if (
        not isinstance(resident_accounting, Mapping)
        or set(resident_accounting)
        != {
            "schema_version",
            *_OPTION_D_M0_RESIDENT_WINDOW_ACCOUNTING_INTEGER_FIELDS_V1,
            "transfer_operation_measurement_scope",
            "route_counter_semantics",
            "identity_bindings_sha256",
            "accounting_sha256",
        }
        or resident_accounting.get("schema_version")
        != "eiel_option_d_m0_resident_window_transfer_accounting_v1"
        or any(
            type(resident_accounting.get(field)) is not int
            or resident_accounting[field] < 0
            for field in _OPTION_D_M0_RESIDENT_WINDOW_ACCOUNTING_INTEGER_FIELDS_V1
        )
        or resident_accounting.get("transfer_operation_measurement_scope")
        != _OPTION_D_M0_RESIDENT_WINDOW_TRANSFER_SCOPE_V1
        or resident_accounting.get("route_counter_semantics")
        != _OPTION_D_M0_RESIDENT_WINDOW_COUNTER_SEMANTICS_V1
        or resident_accounting.get("identity_bindings_sha256")
        != _canonical_sha256(identity)
        or resident_accounting.get("accounting_sha256")
        != _canonical_sha256(resident_accounting_payload)
    ):
        raise CampaignProviderError(
            "Option D resident-window transfer accounting is invalid"
        )
    per_cell_to_total = {
        "actual_ordinary_birth_counts": "actual_ordinary_birth_count",
        "actual_turnover_birth_counts": "actual_turnover_birth_count",
        "torch_policy_raw_word_deltas": "torch_policy_raw_word_delta",
        "torch_policy_semantic_draw_deltas": ("torch_policy_semantic_draw_delta"),
    }
    for field, total_field in per_cell_to_total.items():
        items = per_cell.get(field)
        if (
            not isinstance(items, list)
            or len(items) != len(cell_ids)
            or any(type(item) is not int or item < 0 for item in items)
            or sum(items) != receipt[total_field]
        ):
            raise CampaignProviderError(
                "Option D RNG instrumentation receipt per-cell field is invalid: "
                f"{field}"
            )
    if (
        per_cell["torch_policy_raw_word_deltas"]
        != per_cell["torch_policy_semantic_draw_deltas"]
    ):
        raise CampaignProviderError(
            "Option D D1 torch.rand(float32) raw-word and semantic-scalar deltas differ"
        )
    endpoint_fields = (
        "policy_birth_tape_cursor_starts",
        "policy_birth_tape_cursor_ends",
        "policy_birth_tape_cursor_deltas",
        "mutation_torch_cpu_semantic_draw_starts",
        "mutation_torch_cpu_semantic_draw_ends",
    )
    for field in endpoint_fields:
        items = per_cell.get(field)
        if (
            not isinstance(items, list)
            or len(items) != len(cell_ids)
            or any(type(item) is not int or item < 0 for item in items)
        ):
            raise CampaignProviderError(
                "Option D RNG instrumentation receipt per-cell boundary field is "
                f"invalid: {field}"
            )
    logical_widths = per_cell.get("policy_birth_crossover_logical_widths")
    state_advancing_births = per_cell.get("torch_policy_state_advancing_birth_counts")
    counter_modes = per_cell.get("policy_rng_counter_modes")
    rng_semantics_versions = per_cell.get("rng_semantics_versions")
    turnover_elided = per_cell.get("turnover_policy_crossover_elided")
    if (
        not isinstance(logical_widths, list)
        or len(logical_widths) != len(cell_ids)
        or any(
            type(width) is not int or width != logical_policy_row_width
            for width in logical_widths
        )
        or not isinstance(state_advancing_births, list)
        or len(state_advancing_births) != len(cell_ids)
        or any(type(count) is not int or count < 0 for count in state_advancing_births)
        or not isinstance(counter_modes, list)
        or len(counter_modes) != len(cell_ids)
        or not isinstance(rng_semantics_versions, list)
        or len(rng_semantics_versions) != len(cell_ids)
        or any(
            (version, mode)
            not in {
                (1, "state_advancing_torch_tape_v1"),
                (2, "stateless_philox_formula_v2"),
            }
            for version, mode in zip(
                rng_semantics_versions, counter_modes, strict=True
            )
        )
        or type(turnover_elided) is not bool
    ):
        raise CampaignProviderError(
            "Option D RNG instrumentation receipt D1 logical policy counter "
            "contract fields are invalid"
        )
    for index in range(len(cell_ids)):
        formula_rngv2 = rng_semantics_versions[index] == 2
        if formula_rngv2:
            expected_state_advancing_births = (
                0
                if turnover_elided
                else per_cell["actual_turnover_birth_counts"][index]
            )
        else:
            expected_state_advancing_births = per_cell[
                "actual_ordinary_birth_counts"
            ][index] + (
                0
                if turnover_elided
                else per_cell["actual_turnover_birth_counts"][index]
            )
        if (
            per_cell["policy_birth_tape_cursor_ends"][index]
            - per_cell["policy_birth_tape_cursor_starts"][index]
            != per_cell["policy_birth_tape_cursor_deltas"][index]
            or per_cell["mutation_torch_cpu_semantic_draw_ends"][index]
            - per_cell["mutation_torch_cpu_semantic_draw_starts"][index]
            != per_cell["torch_policy_semantic_draw_deltas"][index]
            or state_advancing_births[index] != expected_state_advancing_births
            or (
                formula_rngv2
                and per_cell["policy_birth_tape_cursor_deltas"][index] != 0
            )
            or (
                not formula_rngv2
                and per_cell["policy_birth_tape_cursor_deltas"][index]
                != state_advancing_births[index]
            )
            or per_cell["torch_policy_semantic_draw_deltas"][index]
            != state_advancing_births[index] * logical_widths[index]
        ):
            raise CampaignProviderError(
                "Option D RNG instrumentation receipt per-cell counter boundary "
                "arithmetic differs"
            )
    for scope_field in ("birth_measurement_scope", "torch_counter_measurement_scope"):
        if not isinstance(per_cell.get(scope_field), str) or not per_cell[scope_field]:
            raise CampaignProviderError(
                "Option D RNG instrumentation receipt per-cell scope is invalid: "
                f"{scope_field}"
            )
    return receipt, identity


def _validate_option_d_m0_cpu_policy_service_receipt_v1(
    *,
    linked_option_d_receipt: Any,
    policy_service_receipt: Any,
) -> None:
    """Validate the qualification-only sibling without changing RNG counters."""

    linked_receipt, linked_identity = _validate_option_d_m0_primary_receipt_v1(
        linked_option_d_receipt
    )
    if not isinstance(policy_service_receipt, Mapping):
        raise CampaignProviderError(
            "Option D qualification requires the sibling CPU policy-service receipt"
        )
    sibling = dict(policy_service_receipt)
    if (
        sibling.get("schema_version")
        != OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_SCHEMA_VERSION
        or sibling.get("authority") != OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_AUTHORITY
        or sibling.get("optimization_performed") is not False
        or sibling.get("option_d_m0_telemetry_enabled") is not True
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service receipt schema or authority is invalid"
        )
    sibling_identity = _validate_option_d_m0_shared_identity_v1(
        sibling.get("identity_bindings"),
        receipt_name="Option D CPU policy-service receipt",
    )
    if sibling_identity != linked_identity:
        raise CampaignProviderError(
            "Option D CPU policy-service receipt identity differs from its linked "
            "RNG instrumentation receipt"
        )
    for field in _OPTION_D_M0_CPU_POLICY_SERVICE_INTEGER_FIELDS_V1:
        value = sibling.get(field)
        if type(value) is not int or value < 0:
            raise CampaignProviderError(
                f"Option D CPU policy-service receipt field is invalid: {field}"
            )
    if sibling["process_hwm_bytes"] < sibling["process_rss_bytes"]:
        raise CampaignProviderError(
            "Option D CPU policy-service receipt HWM is below current RSS"
        )
    if (
        sibling.get("host_allocation_measurement_scope")
        != OPTION_D_M0_HOST_ALLOCATION_MEASUREMENT_SCOPE
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service host-allocation measurement scope is invalid"
        )
    if (
        sibling.get("transfer_operation_measurement_scope")
        != OPTION_D_M0_TRANSFER_OPERATION_MEASUREMENT_SCOPE
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service transfer-operation measurement scope is "
            "invalid"
        )
    if (
        sibling.get("synchronization_measurement_scope")
        != OPTION_D_M0_SYNCHRONIZATION_MEASUREMENT_SCOPE
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service synchronization measurement scope is invalid"
        )
    if not isinstance(sibling.get("process_memory_measurement_scope"), str) or not (
        sibling["process_memory_measurement_scope"]
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service process-memory measurement scope is invalid"
        )
    source_environment = linked_receipt["source_environment_receipt"]
    if sibling.get("source_environment_receipt_path") != source_environment.get(
        "path"
    ) or sibling.get("source_environment_receipt_sha256") != source_environment.get(
        "sha256"
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service source/environment receipt identity differs"
        )

    sibling_artifact = linked_receipt.get("cpu_policy_service_receipt_artifact")
    sibling_summary = linked_receipt.get("cpu_policy_service_compact_summary")
    if (sibling_artifact is None) is not (sibling_summary is None):
        raise CampaignProviderError(
            "Option D CPU policy-service artifact and compact summary must be retained "
            "together"
        )
    if sibling_artifact is not None:
        if not isinstance(sibling_artifact, Mapping) or not isinstance(
            sibling_summary, Mapping
        ):
            raise CampaignProviderError(
                "Option D CPU policy-service retained artifact is malformed"
            )
        sibling_path = Path(str(sibling_artifact.get("path", "")))
        if (
            not sibling_path.is_absolute()
            or not sibling_path.is_file()
            or not _valid_sha256(sibling_artifact.get("sha256"))
            or _sha256_file_v1(sibling_path) != sibling_artifact["sha256"]
            or sibling_artifact.get("schema_version") != sibling["schema_version"]
            or sibling_artifact.get("run_identity_sha256")
            != linked_identity["run_identity_sha256"]
            or sibling_artifact.get("generation_tick_boundary_identity_sha256")
            != linked_identity["generation_tick_boundary_identity_sha256"]
            or sibling_artifact.get("identity_bindings") != linked_identity
        ):
            raise CampaignProviderError(
                "Option D CPU policy-service retained artifact identity is invalid"
            )
        expected_summary = {
            name: sibling[name]
            for name in _OPTION_D_M0_CPU_POLICY_SERVICE_SUMMARY_FIELDS_V1
        }
        if dict(sibling_summary) != expected_summary:
            raise CampaignProviderError(
                "Option D CPU policy-service compact summary differs from its sibling"
            )


def _finalize_option_d_m0_provider_receipts_v1(
    *,
    plan: Mapping[str, Any],
    request: "ResidentBatchRequestV1",
    native_receipt: dict[str, Any],
    source_environment: Mapping[str, Any],
    loaded_artifact_contract: Mapping[str, Any],
    memory_measurement: Mapping[str, Any],
    receipt_output_root: Path,
    segment_ordinal: int,
    start_generation: int,
    end_generation: int,
    source_projections: Sequence[Mapping[str, Any]],
    terminal_projections: Sequence[Mapping[str, Any]],
    settlement_event_rows: Sequence[Mapping[str, Any]],
    terminal_projection_digest_sha256s: Mapping[str, str],
    canonical_terminal_state_digest_sha256s: Mapping[str, str],
    checkpoint_identity_contract: Mapping[str, Any],
) -> None:
    """Construct and bind both M0 receipts after authoritative settlement."""

    if native_receipt.get("option_d_m0_telemetry_enabled") is not True:
        return
    construction_started = time.perf_counter_ns()
    source_commit = source_environment.get("source_commit")
    build_provenance = loaded_artifact_contract.get("build_provenance")
    if (
        not isinstance(build_provenance, Mapping)
        or loaded_artifact_contract.get("telemetry_enabled") is not True
        or build_provenance.get("build_source_commit") != source_commit
    ):
        raise CampaignProviderError(
            "Option D finalization lacks the source-bound loaded artifact contract"
        )
    if (
        OPTION_D_M0_PRIMARY_RECEIPT_FIELD in native_receipt
        or OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_FIELD in native_receipt
    ):
        raise CampaignProviderError(
            "Option D native execution must not preconstruct provider-owned receipts"
        )
    raw_policy_metrics = _validate_option_d_m0_cpu_policy_service_raw_metrics_v1(
        native_receipt.get(OPTION_D_M0_CPU_POLICY_SERVICE_RAW_METRICS_FIELD)
    )
    per_cell = _option_d_m0_settled_per_cell_measurements_v1(
        request=request,
        native_receipt=native_receipt,
        source_projections=source_projections,
        terminal_projections=terminal_projections,
        settlement_event_rows=settlement_event_rows,
        start_generation=start_generation,
        end_generation=end_generation,
        turnover_policy_crossover_elided=(
            _option_d_m0_turnover_policy_crossover_elided_v1(plan)
        ),
    )
    runtime_plan_binding = _option_d_m0_runtime_plan_binding_v1(
        plan=plan,
        request=request,
        native_receipt=native_receipt,
        source_environment=source_environment,
        start_generation=start_generation,
        end_generation=end_generation,
    )
    combined_environment = _option_d_m0_combined_environment_identity_v1(
        source_environment=source_environment,
        loaded_artifact_contract=loaded_artifact_contract,
    )
    measured_run_fields: dict[str, Any] = {}
    for field in (
        "process_rss_bytes",
        "process_hwm_bytes",
        "resident_vram_bytes",
        "peak_vram_bytes",
    ):
        value = memory_measurement.get(field)
        if type(value) is not int or value < 0:
            raise CampaignProviderError(
                f"Option D provider memory measurement is invalid: {field}"
            )
        measured_run_fields[field] = value
    for field in ("process_memory_measurement_scope", "vram_measurement_scope"):
        value = memory_measurement.get(field)
        if not isinstance(value, str) or not value:
            raise CampaignProviderError(
                f"Option D provider memory measurement scope is invalid: {field}"
            )
        measured_run_fields[field] = value
    for field in (
        "checkpoint_projection_time_ns",
        "checkpoint_compression_time_ns",
    ):
        value = native_receipt.get(field)
        if type(value) is not int or value < 0:
            raise CampaignProviderError(
                f"Option D provider timing measurement is invalid: {field}"
            )
        measured_run_fields[field] = value
    compression_scope = native_receipt.get("checkpoint_compression_time_scope")
    if not isinstance(compression_scope, str) or not compression_scope:
        raise CampaignProviderError(
            "Option D provider checkpoint compression timing scope is invalid"
        )
    measured_run_fields["checkpoint_compression_time_scope"] = compression_scope
    measured_run_fields["actual_ordinary_birth_count"] = sum(
        per_cell["actual_ordinary_birth_counts"]
    )
    measured_run_fields["actual_turnover_birth_count"] = sum(
        per_cell["actual_turnover_birth_counts"]
    )
    measured_run_fields["torch_policy_raw_word_delta"] = sum(
        per_cell["torch_policy_raw_word_deltas"]
    )
    measured_run_fields["torch_policy_semantic_draw_delta"] = sum(
        per_cell["torch_policy_semantic_draw_deltas"]
    )
    identity, output_digest = _option_d_m0_provider_identity_bindings_v1(
        plan=plan,
        request=request,
        segment_ordinal=segment_ordinal,
        start_generation=start_generation,
        end_generation=end_generation,
        source_commit=str(source_commit),
        environment_identity_sha256=combined_environment["environment_identity_sha256"],
        terminal_projection_digest_sha256s=terminal_projection_digest_sha256s,
        canonical_terminal_state_digest_sha256s=(
            canonical_terminal_state_digest_sha256s
        ),
        checkpoint_identity_contract=checkpoint_identity_contract,
        runtime_plan_binding=runtime_plan_binding,
        settlement_event_rows_sha256=per_cell["settlement_event_rows_sha256"],
    )
    provider_owned_fields = {
        "terminal_digest_sha256": identity["terminal_digest_sha256"],
        "output_digest_sha256": output_digest,
        "checkpoint_identity_sha256": identity["checkpoint_identity_sha256"],
    }
    measured_run_fields.update(provider_owned_fields)
    resident_window_transfer_accounting: dict[str, Any] = {
        "schema_version": ("eiel_option_d_m0_resident_window_transfer_accounting_v1"),
        **{
            field: native_receipt[field]
            for field in _OPTION_D_M0_RESIDENT_WINDOW_ACCOUNTING_INTEGER_FIELDS_V1
        },
        "transfer_operation_measurement_scope": (
            native_receipt["resident_window_outer_transfer_operation_measurement_scope"]
        ),
        "route_counter_semantics": native_receipt[
            "resident_window_route_counter_semantics"
        ],
        "identity_bindings_sha256": _canonical_sha256(identity),
    }
    resident_window_transfer_accounting["accounting_sha256"] = _canonical_sha256(
        resident_window_transfer_accounting
    )

    sibling: dict[str, Any] = {
        "schema_version": OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_SCHEMA_VERSION,
        "authority": OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_AUTHORITY,
        "optimization_performed": False,
        "option_d_m0_telemetry_enabled": True,
        "identity_bindings": dict(identity),
        **raw_policy_metrics,
        "process_rss_bytes": measured_run_fields["process_rss_bytes"],
        "process_hwm_bytes": measured_run_fields["process_hwm_bytes"],
        "process_memory_measurement_scope": (
            "provider_process_boundary_shared_with_linked_m0_run"
        ),
        "source_environment_receipt_path": source_environment.get("path"),
        "source_environment_receipt_sha256": source_environment.get("sha256"),
    }
    logical_policy_row_width = per_cell["policy_birth_crossover_logical_widths"][0]
    rng_semantics_versions = set(per_cell["rng_semantics_versions"])
    if len(rng_semantics_versions) != 1:
        raise CampaignProviderError(
            "Option D TW RNG semantic version differs across physical cells"
        )
    rng_semantics_version = rng_semantics_versions.pop()

    primary: dict[str, Any] = {
        "schema_version": OPTION_D_M0_PRIMARY_RECEIPT_SCHEMA_VERSION,
        "option_d_m0_telemetry_enabled": True,
        "identity_bindings": dict(identity),
        "output_digest_sha256": output_digest,
        "source_environment_receipt": dict(source_environment),
        "loaded_artifact_contract": dict(loaded_artifact_contract),
        "combined_environment_identity": dict(combined_environment),
        "runtime_plan_binding": dict(runtime_plan_binding),
        "torch_policy_counter_contract": (
            _option_d_m0_torch_policy_counter_contract_v1(
                logical_policy_row_width,
                rng_semantics_version=rng_semantics_version,
            )
        ),
        "per_cell_measurements": dict(per_cell),
        "settlement_event_evidence": {
            "schema_version": "eiel_option_d_m0_settlement_event_evidence_v1",
            "row_count": per_cell["settlement_event_row_count"],
            "rows_sha256": per_cell["settlement_event_rows_sha256"],
            "identity_bindings_sha256": _canonical_sha256(identity),
            "terminal_projection_digest_sha256s": dict(
                sorted(terminal_projection_digest_sha256s.items())
            ),
        },
        "resident_window_transfer_accounting": (resident_window_transfer_accounting),
        "process_memory_measurement_scope": measured_run_fields[
            "process_memory_measurement_scope"
        ],
        "vram_measurement_scope": measured_run_fields["vram_measurement_scope"],
        "checkpoint_projection_time_scope": ("provider_settlement_projector_wall_time"),
        "checkpoint_compression_time_scope": compression_scope,
    }
    for field in _OPTION_D_M0_PRIMARY_INTEGER_FIELDS_V1:
        if field != "receipt_construction_time_ns":
            primary[field] = measured_run_fields.get(field)
    primary["receipt_construction_time_ns"] = 0
    _validate_option_d_m0_cpu_policy_service_receipt_v1(
        linked_option_d_receipt=primary,
        policy_service_receipt=sibling,
    )

    from engine.campaigns.orchestration import atomic_write_json

    sibling_root = receipt_output_root / "option_d_m0_receipts"
    sibling_root.mkdir(parents=True, exist_ok=True)
    sibling_path = sibling_root / (
        f"segment-{segment_ordinal:06d}-exact-cpu-policy-service.json"
    )
    atomic_write_json(sibling_path, sibling)
    sibling_artifact = {
        "path": str(sibling_path.resolve()),
        "sha256": _sha256_file_v1(sibling_path),
        "schema_version": sibling["schema_version"],
        "run_identity_sha256": identity["run_identity_sha256"],
        "generation_tick_boundary_identity_sha256": identity[
            "generation_tick_boundary_identity_sha256"
        ],
    }
    sibling_artifact["identity_bindings"] = dict(identity)
    sibling_summary = {
        name: sibling[name]
        for name in _OPTION_D_M0_CPU_POLICY_SERVICE_SUMMARY_FIELDS_V1
    }
    primary["cpu_policy_service_receipt_artifact"] = sibling_artifact
    primary["cpu_policy_service_compact_summary"] = sibling_summary
    construction_time_ns = time.perf_counter_ns() - construction_started
    primary["receipt_construction_time_ns"] = construction_time_ns
    measured_run_fields["receipt_construction_time_ns"] = construction_time_ns

    _validate_option_d_m0_cpu_policy_service_receipt_v1(
        linked_option_d_receipt=primary,
        policy_service_receipt=sibling,
    )
    if native_receipt.get("exact_cpu_policy_service_calls_inside_tick_loop") != sibling[
        "calls_inside_tick_loop"
    ] or native_receipt.get("native_host_policy_service_used") is not (
        sibling["calls_inside_tick_loop"] > 0
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service receipt disagrees with native call counters"
        )
    linked, linked_identity = _validate_option_d_m0_primary_receipt_v1(primary)
    for field in (
        *_OPTION_D_M0_PRIMARY_INTEGER_FIELDS_V1,
        *_OPTION_D_M0_PRIMARY_DIGEST_FIELDS_V1,
    ):
        if measured_run_fields.get(field) != linked[field]:
            raise CampaignProviderError(
                "Option D native run field differs from its RNG instrumentation "
                f"receipt: {field}"
            )
    for field in ("terminal_digest_sha256", "checkpoint_identity_sha256"):
        if measured_run_fields.get(field) != linked_identity[field]:
            raise CampaignProviderError(
                "Option D native run identity differs from its RNG instrumentation "
                f"receipt: {field}"
            )
    native_receipt.update(measured_run_fields)
    native_receipt[OPTION_D_M0_PRIMARY_RECEIPT_FIELD] = primary
    native_receipt[OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_FIELD] = sibling
    native_receipt["exact_cpu_policy_service_observation_receipt_artifact"] = (
        sibling_artifact
    )
    native_receipt["exact_cpu_policy_service_observation_summary"] = sibling_summary
    native_receipt.pop(OPTION_D_M0_CPU_POLICY_SERVICE_RAW_METRICS_FIELD, None)
    native_receipt["option_d_m0_qualification_status"] = "provider_finalized_validated"


def _project_option_d_m0_provider_receipts_v1(
    *,
    direct_result: Mapping[str, Any],
    native_receipt: dict[str, Any],
) -> None:
    """Retain native M0 measurements for post-settlement provider finalization."""

    telemetry_enabled = direct_result.get("option_d_m0_telemetry_enabled", False)
    if type(telemetry_enabled) is not bool:
        raise CampaignProviderError(
            "Option D M0 telemetry enablement must be an exact boolean"
        )
    native_receipt["option_d_m0_telemetry_enabled"] = telemetry_enabled
    for field in _LEGACY_NATIVE_RECEIPT_FIELDS_V1:
        if field in direct_result:
            native_receipt[field] = direct_result[field]
    if telemetry_enabled is not True:
        return
    missing_fields = tuple(
        field
        for field in (
            *_OPTION_D_M0_PROVIDER_RECEIPT_FIELDS_V1,
            *_OPTION_D_M0_RUNTIME_PLAN_NATIVE_FIELDS_V1,
            *_OPTION_D_M0_RESIDENT_WINDOW_ACCOUNTING_INTEGER_FIELDS_V1,
            "resident_window_outer_transfer_operation_measurement_scope",
            "resident_window_route_counter_semantics",
        )
        if field not in direct_result
    )
    if missing_fields:
        raise CampaignProviderError(
            "Option D M0 native telemetry fields are missing: "
            + ", ".join(missing_fields)
        )
    unexpected_provider_fields = tuple(
        field
        for field in _OPTION_D_M0_PROVIDER_OWNED_DIRECT_FIELDS_V1
        if field in direct_result
    )
    if unexpected_provider_fields:
        raise CampaignProviderError(
            "Option D native execution supplied provider-owned post-settlement fields: "
            + ", ".join(unexpected_provider_fields)
        )
    if OPTION_D_M0_PRIMARY_RECEIPT_FIELD in direct_result:
        raise CampaignProviderError(
            "Option D native execution must not preconstruct the provider-owned RNG "
            "instrumentation receipt"
        )
    if OPTION_D_M0_CPU_POLICY_SERVICE_RECEIPT_FIELD in direct_result:
        raise CampaignProviderError(
            "Option D native execution must not preconstruct the provider-owned CPU "
            "policy-service receipt"
        )
    raw_policy_metrics = _validate_option_d_m0_cpu_policy_service_raw_metrics_v1(
        direct_result[OPTION_D_M0_CPU_POLICY_SERVICE_RAW_METRICS_FIELD]
    )
    native_receipt[_OPTION_D_M0_NATIVE_TURNOVER_REFILL_ROWS_FIELD_V1] = (
        _option_d_m0_project_native_turnover_refill_rows_v1(direct_result)
    )
    for field in _OPTION_D_M0_RUNTIME_PLAN_NATIVE_FIELDS_V1:
        native_receipt[field] = direct_result[field]
    for field in _OPTION_D_M0_RESIDENT_WINDOW_ACCOUNTING_INTEGER_FIELDS_V1:
        value = direct_result[field]
        if type(value) is not int or value < 0:
            raise CampaignProviderError(
                f"Option D M0 resident-window accounting is invalid: {field}"
            )
        native_receipt[field] = value
    if (
        direct_result["resident_window_outer_transfer_operation_measurement_scope"]
        != _OPTION_D_M0_RESIDENT_WINDOW_TRANSFER_SCOPE_V1
        or direct_result["resident_window_route_counter_semantics"]
        != _OPTION_D_M0_RESIDENT_WINDOW_COUNTER_SEMANTICS_V1
    ):
        raise CampaignProviderError(
            "Option D M0 resident-window accounting scope is invalid"
        )
    native_receipt["resident_window_outer_transfer_operation_measurement_scope"] = (
        _OPTION_D_M0_RESIDENT_WINDOW_TRANSFER_SCOPE_V1
    )
    native_receipt["resident_window_route_counter_semantics"] = (
        _OPTION_D_M0_RESIDENT_WINDOW_COUNTER_SEMANTICS_V1
    )
    boolean_fields = (
        "option_d_m0_telemetry_enabled",
        "native_host_policy_birth_segment_refill_used",
        "native_host_gaussian_segment_refill_used",
        "native_host_policy_service_used",
    )
    receipt_fields = (
        "policy_birth_segment_refill_receipts",
        "gaussian_segment_refill_receipts",
    )
    for field in boolean_fields:
        if type(direct_result[field]) is not bool:
            raise CampaignProviderError(
                f"Option D M0 native telemetry boolean is invalid: {field}"
            )
    for field in receipt_fields:
        if not isinstance(direct_result[field], list):
            raise CampaignProviderError(
                f"Option D M0 native telemetry receipt list is invalid: {field}"
            )
    semantics = direct_result["aten_operations_inside_tick_loop_count_semantics"]
    if not isinstance(semantics, str) or not semantics:
        raise CampaignProviderError(
            "Option D M0 ATen operation count semantics are invalid"
        )
    non_integer_fields = {
        *boolean_fields,
        *receipt_fields,
        "aten_operations_inside_tick_loop_count_semantics",
        OPTION_D_M0_CPU_POLICY_SERVICE_RAW_METRICS_FIELD,
    }
    for field in _OPTION_D_M0_PROVIDER_RECEIPT_FIELDS_V1:
        if field in non_integer_fields:
            continue
        value = direct_result[field]
        if type(value) is not int or value < 0:
            raise CampaignProviderError(
                f"Option D M0 native telemetry integer is invalid: {field}"
            )
    if direct_result[
        "exact_cpu_policy_service_calls_inside_tick_loop"
    ] != raw_policy_metrics["calls_inside_tick_loop"] or direct_result[
        "native_host_policy_service_used"
    ] is not (
        raw_policy_metrics["calls_inside_tick_loop"] > 0
    ):
        raise CampaignProviderError(
            "Option D CPU policy-service raw metrics disagree with native call counters"
        )
    for field in _OPTION_D_M0_PROVIDER_RECEIPT_FIELDS_V1:
        native_receipt[field] = (
            raw_policy_metrics
            if field == OPTION_D_M0_CPU_POLICY_SERVICE_RAW_METRICS_FIELD
            else direct_result[field]
        )


def _provider_native_exposure_receipt_v1(
    *,
    request: "ResidentBatchRequestV1",
    native_execution_receipts: Sequence[Mapping[str, Any]],
    segment_artifacts: Sequence[Mapping[str, Any]],
    tick_native: bool = False,
) -> dict[str, Any] | None:
    """Summarize generation-mode native reproduction exposure artifacts.

    Tick-native routes expose bounded absolute-tick settlements and, where
    requested, provider-native Chronicle segments.  Their terminal capture is
    owned by a projection cursor, so the legacy generation-segment artifact
    list is intentionally incomplete and must not be forced through this
    receipt contract.
    """

    if tick_native:
        return None
    if not segment_artifacts:
        return None
    cell_ids = tuple(request.cell_ids)
    by_segment: dict[int, dict[str, Mapping[str, Any]]] = {}
    for artifact in segment_artifacts:
        if not isinstance(artifact, Mapping):
            raise CampaignProviderError("native exposure artifact is malformed")
        segment_index = artifact.get("segment_index")
        cell_id = artifact.get("cell_id")
        cell = artifact.get("cell")
        if (
            type(segment_index) is not int
            or not isinstance(cell_id, str)
            or cell_id not in cell_ids
            or not isinstance(cell, Mapping)
            or cell.get("cell_index") != cell_ids.index(cell_id)
            or cell.get("enabled") is not True
            or cell.get("status") != 0
            or not _valid_sha256(cell.get("segment_sha256"))
            or not _valid_sha256(cell.get("previous_chain_sha256"))
            or not _valid_sha256(cell.get("chain_sha256_after_segment"))
            or not _valid_sha256(artifact.get("segment_output_sha256"))
        ):
            raise CampaignProviderError(
                "native exposure artifact identity or status is invalid"
            )
        attempts = cell.get("attempt_count")
        successes = cell.get("success_count")
        failures = cell.get("failure_count")
        snapshots = cell.get("snapshot_count")
        attempt_rows = cell.get("attempts")
        cumulative = cell.get("cumulative_counts_after_segment")
        if (
            type(attempts) is not int
            or type(successes) is not int
            or type(failures) is not int
            or type(snapshots) is not int
            or min(attempts, successes, failures, snapshots) < 0
            or successes + failures != attempts
            or not isinstance(attempt_rows, list)
            or len(attempt_rows) != attempts
            or not isinstance(cumulative, Mapping)
            or any(
                type(cumulative.get(name)) is not int or int(cumulative[name]) < 0
                for name in (
                    "attempts",
                    "successes",
                    "failures",
                    "eligible_snapshots",
                )
            )
        ):
            raise CampaignProviderError(
                "native exposure artifact counters are incoherent"
            )
        segment = by_segment.setdefault(segment_index, {})
        if cell_id in segment:
            raise CampaignProviderError(
                "native exposure artifact duplicates a cell in one segment"
            )
        segment[cell_id] = cell
    segment_ids = tuple(sorted(by_segment))
    if len(segment_ids) != len(native_execution_receipts):
        raise CampaignProviderError(
            "native exposure segment count differs from CUDA dispatch receipts"
        )
    for segment_id in segment_ids:
        if set(by_segment[segment_id]) != set(cell_ids):
            raise CampaignProviderError(
                "native exposure segment lacks exact cell coverage"
            )
    segment_artifact_sha256s = [
        _canonical_sha256(artifact)
        for segment_id in segment_ids
        for artifact in sorted(
            (
                artifact
                for artifact in segment_artifacts
                if artifact.get("segment_index") == segment_id
            ),
            key=lambda value: cell_ids.index(str(value.get("cell_id"))),
        )
    ]
    generation_spans = [
        (
            int(row["start_generation"]),
            int(row["end_generation"]),
        )
        for row in native_execution_receipts
        if isinstance(row, Mapping)
        and type(row.get("start_generation")) is int
        and type(row.get("end_generation")) is int
    ]
    if len(generation_spans) != len(native_execution_receipts) or any(
        end < start for start, end in generation_spans
    ):
        raise CampaignProviderError("native execution receipt spans are invalid")
    exact_tick_spans = [
        (
            int(row["start_next_tick"]),
            int(row["end_next_tick"]),
        )
        for row in native_execution_receipts
        if isinstance(row, Mapping)
        and type(row.get("start_next_tick")) is int
        and type(row.get("end_next_tick")) is int
    ]
    if exact_tick_spans and len(exact_tick_spans) != len(native_execution_receipts):
        raise CampaignProviderError(
            "native execution receipt tick spans are incomplete"
        )
    if any(end <= start for start, end in exact_tick_spans):
        raise CampaignProviderError("native execution receipt tick spans are invalid")
    if any(end == start for start, end in generation_spans if not exact_tick_spans):
        raise CampaignProviderError("native execution receipt spans are invalid")
    scientific_tick_count = (
        sum((end - start) * len(cell_ids) for start, end in exact_tick_spans)
        if exact_tick_spans
        else sum(
            (end - start) * request.ticks_per_generation * len(cell_ids)
            for start, end in generation_spans
        )
    )
    cell_receipts: dict[str, dict[str, Any]] = {}
    total_attempts = 0
    total_successes = 0
    total_failures = 0
    total_snapshots = 0
    for cell_id in cell_ids:
        cells = [by_segment[segment_id][cell_id] for segment_id in segment_ids]
        attempts = sum(int(cell["attempt_count"]) for cell in cells)
        successes = sum(int(cell["success_count"]) for cell in cells)
        failures = sum(int(cell["failure_count"]) for cell in cells)
        snapshots = sum(int(cell["snapshot_count"]) for cell in cells)
        total_attempts += attempts
        total_successes += successes
        total_failures += failures
        total_snapshots += snapshots
        last = cells[-1]
        cell_receipts[cell_id] = {
            "cell_id": cell_id,
            "segment_count": len(cells),
            "attempt_count": attempts,
            "success_count": successes,
            "failure_count": failures,
            "eligible_snapshot_count": snapshots,
            "chain_sha256_after_last_segment": last["chain_sha256_after_segment"],
            "cumulative_counts_after_last_segment": dict(
                last["cumulative_counts_after_segment"]
            ),
        }
    payload = {
        "schema_version": PROVIDER_NATIVE_EXPOSURE_RECEIPT_SCHEMA_VERSION,
        "batch_id": request.batch_id,
        "profile_id": request.profile_id,
        "execution_route_sha256": request.execution_route_sha256,
        "cell_ids": list(cell_ids),
        "segment_indices": list(segment_ids),
        "current_execution_segment_artifacts_sha256": segment_artifact_sha256s,
        "current_execution_generation_spans": [
            {"start_generation": start, "end_generation": end}
            for start, end in generation_spans
        ],
        **(
            {
                "current_execution_tick_spans": [
                    {"start_next_tick": start, "end_next_tick": end}
                    for start, end in exact_tick_spans
                ]
            }
            if exact_tick_spans
            else {}
        ),
        "current_execution_scientific_tick_count": scientific_tick_count,
        "native_execution_receipts_sha256": _canonical_sha256(
            list(native_execution_receipts)
        ),
        "cell_receipts": cell_receipts,
        "totals": {
            "attempt_count": total_attempts,
            "success_count": total_successes,
            "failure_count": total_failures,
            "eligible_snapshot_count": total_snapshots,
            "decision_bearing_drop_count": 0,
        },
    }
    return {**payload, "receipt_sha256": _canonical_sha256(payload)}


_NATIVE_EXECUTION_COMPATIBILITY_FIELDS_V1: Final[tuple[str, ...]] = (
    "generation",
    "generations",
    "start_generation",
    "end_generation",
    "logical_generation_count",
    "physical_window_generations",
    "resident_segment_generations",
    "physical_storage_generations",
    "ticks_per_generation",
)


def _native_public_execution_receipt_v1(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose native dispatch identity without making the legacy lattice authoritative."""

    public = dict(receipt)
    compatibility = {
        field: public.pop(field)
        for field in _NATIVE_EXECUTION_COMPATIBILITY_FIELDS_V1
        if field in public
    }
    if compatibility:
        public["legacy_compatibility"] = compatibility
    public.setdefault("time_authority", "absolute_tick_v1")
    return public


def _native_public_authoritative_state_receipts_v1(
    receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project tick-native authoritative-state receipts for public emission."""

    public_receipts: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            raise CampaignProviderError(
                f"authoritative-state receipt {index} is malformed"
            )
        start_next_tick = receipt.get("start_next_tick")
        end_next_tick = receipt.get("end_next_tick")
        if not (
            type(start_next_tick) is int
            and start_next_tick >= 0
            and type(end_next_tick) is int
            and end_next_tick > start_next_tick
        ):
            raise CampaignProviderError(
                "tick-native authoritative-state receipt lacks an exact "
                "absolute-tick span"
            )
        public_receipts.append(_native_public_execution_receipt_v1(receipt))
    return public_receipts


def _tick_native_performance_timeline_binding_v1(
    *,
    launch_started_perf_ns: int,
    launch_completed_perf_ns: int,
    native_execution_receipts: Sequence[Mapping[str, Any]],
    acs_capture_measurements: Sequence[Mapping[str, Any]],
    acs_catalog_publication_measurements: Sequence[Mapping[str, Any]],
    owner_preparation_measurements: Sequence[Mapping[str, Any]],
    terminal_next_absolute_tick: int,
    acs_close_completed_perf_ns: int,
    restore_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind exact provider clocks to the strict launch timeline contract."""

    from engine.campaigns.tick_native_performance_timeline_v1 import (
        TickNativePerformanceTimelineError,
        derive_tick_native_performance_timeline_v1,
    )

    if not (
        type(launch_started_perf_ns) is int
        and type(launch_completed_perf_ns) is int
        and launch_started_perf_ns < launch_completed_perf_ns
        and type(terminal_next_absolute_tick) is int
        and terminal_next_absolute_tick > 0
        and type(acs_close_completed_perf_ns) is int
        and acs_close_completed_perf_ns <= launch_completed_perf_ns
    ):
        raise CampaignProviderError("tick-native launch timing boundary is invalid")

    simulations: list[dict[str, Any]] = []
    for index, receipt in enumerate(native_execution_receipts):
        if not isinstance(receipt, Mapping):
            raise CampaignProviderError(
                f"tick-native simulation timing receipt {index} is invalid"
            )
        simulations.append(
            {
                "start_perf_ns": receipt.get("segment_dispatch_started_perf_ns"),
                "end_perf_ns": receipt.get("segment_dispatch_finished_perf_ns"),
                "start_next_tick": receipt.get("start_next_tick"),
                "end_next_tick": receipt.get("end_next_tick"),
            }
        )

    publications: dict[int, Mapping[str, Any]] = {}
    for row in acs_catalog_publication_measurements:
        if not isinstance(row, Mapping) or type(row.get("coordinate")) is not int:
            raise CampaignProviderError(
                "tick-native durable publication timing receipt is invalid"
            )
        coordinate = int(row["coordinate"])
        if coordinate in publications:
            raise CampaignProviderError(
                "tick-native durable publication timing coordinate is duplicated"
            )
        publications[coordinate] = row

    checkpoints: list[dict[str, Any]] = []
    terminal_submission: int | None = None
    terminal_gpu_release: int | None = None
    for index, capture in enumerate(acs_capture_measurements):
        if not isinstance(capture, Mapping):
            raise CampaignProviderError(
                f"tick-native checkpoint timing receipt {index} is invalid"
            )
        coordinate = capture.get("coordinate")
        if type(coordinate) is not int or coordinate not in publications:
            raise CampaignProviderError(
                "tick-native checkpoint timing lacks exact durable publication"
            )
        publication = publications[int(coordinate)]
        submitted = capture.get("detached_capture_submitted_perf_ns")
        writer_started = capture.get("detached_writer_started_perf_ns")
        gpu_release = capture.get("gpu_safely_releasable_perf_ns")
        terminal_capture = coordinate == terminal_next_absolute_tick
        context_release = capture.get("terminal_cuda_primary_context_release")
        if terminal_capture and (
            type(gpu_release) is not int
            or type(submitted) is not int
            or type(writer_started) is not int
        ):
            raise CampaignProviderError(
                "terminal checkpoint timing lacks an ordered GPU-safe release"
            )
        if terminal_capture and isinstance(context_release, Mapping):
            if (
                context_release.get("released_perf_ns") != gpu_release
                or gpu_release >= submitted
                or submitted > writer_started
            ):
                raise CampaignProviderError(
                    "terminal checkpoint persistence did not begin after CUDA context release"
                )
        elif terminal_capture and gpu_release < submitted:
            raise CampaignProviderError(
                "terminal checkpoint timing lacks an ordered GPU-safe release"
            )
        checkpoint = {
            "checkpoint_tick": coordinate,
            "owner_span": {
                "start_next_tick": capture.get("owner_span_start_tick"),
                "end_next_tick": capture.get("owner_span_end_tick"),
            },
            "detach": {
                "start_perf_ns": capture.get(
                    "segment_dispatch_finished_perf_ns"
                ),
                "end_perf_ns": gpu_release if terminal_capture else submitted,
            },
            "persist": {
                "start_perf_ns": capture.get("detached_writer_started_perf_ns"),
                "end_perf_ns": publication.get(
                    "durable_acknowledged_perf_ns"
                ),
                "async": True,
            },
        }
        checkpoints.append(checkpoint)
        if terminal_capture:
            if type(submitted) is not int:
                raise CampaignProviderError(
                    "terminal checkpoint timing lacks detached submission"
                )
            terminal_submission = submitted
            terminal_gpu_release = gpu_release

    if terminal_submission is None or terminal_gpu_release is None:
        raise CampaignProviderError(
            "tick-native timeline lacks its terminal submission or GPU release"
        )
    if terminal_gpu_release >= acs_close_completed_perf_ns:
        raise CampaignProviderError(
            "tick-native ACS tail has no positive measured interval"
        )
    simulation_window_start = min(
        int(row["start_perf_ns"]) for row in simulations
    )
    simulation_window_end = max(int(row["end_perf_ns"]) for row in simulations)
    if not simulation_window_start < simulation_window_end <= terminal_gpu_release:
        raise CampaignProviderError(
            "tick-native GPU performance window is not ordered"
        )

    owner_preparations = [dict(row) for row in owner_preparation_measurements]
    restore = (
        {"status": "not_requested"}
        if restore_receipt is None
        else dict(restore_receipt)
    )
    envelope: dict[str, Any] = {
        "clock_domain": "perf_counter_ns",
        "wall_time": {
            "start_perf_ns": launch_started_perf_ns,
            "end_perf_ns": launch_completed_perf_ns,
        },
        "owner_preparation_receipts": owner_preparations,
        "simulation_receipts": simulations,
        "checkpoint_receipts": checkpoints,
        "gpu_performance_window": {
            "start_perf_ns": simulation_window_start,
            "end_perf_ns": terminal_gpu_release,
            "terminal_checkpoint_tick": terminal_next_absolute_tick,
            "post_simulation_acs_tail_excluded": True,
        },
        "acs_tail_receipt": {
            "start_perf_ns": terminal_gpu_release,
            "end_perf_ns": acs_close_completed_perf_ns,
        },
        "restore_receipt": restore,
    }

    # Every exclusive phase and asynchronous persistence interval has an exact
    # provider clock.  Fill only the complement with reason-bearing provider
    # administration; this neither moves phase boundaries nor folds the gaps
    # into T_SIM/T_DETACH/T_PERSIST.
    intervals: list[tuple[int, int]] = []
    for row in simulations:
        intervals.append((int(row["start_perf_ns"]), int(row["end_perf_ns"])))
    for row in owner_preparations:
        intervals.append((int(row["start_perf_ns"]), int(row["end_perf_ns"])))
    if restore.get("status") == "completed":
        intervals.append(
            (int(restore["start_perf_ns"]), int(restore["end_perf_ns"]))
        )
    for row in checkpoints:
        for name in ("detach", "persist"):
            interval = row[name]
            intervals.append(
                (int(interval["start_perf_ns"]), int(interval["end_perf_ns"]))
            )
    intervals.append((terminal_gpu_release, acs_close_completed_perf_ns))
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not (launch_started_perf_ns <= start < end <= launch_completed_perf_ns):
            raise CampaignProviderError(
                "tick-native phase timing lies outside its launch boundary"
            )
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    administrative: list[dict[str, Any]] = []
    cursor = launch_started_perf_ns
    for start, end in merged:
        if cursor < start:
            administrative.append(
                {
                    "start_perf_ns": cursor,
                    "end_perf_ns": start,
                    "reason": "provider_orchestration_outside_declared_phase_v1",
                }
            )
        cursor = max(cursor, end)
    if cursor < launch_completed_perf_ns:
        administrative.append(
            {
                "start_perf_ns": cursor,
                "end_perf_ns": launch_completed_perf_ns,
                "reason": "provider_orchestration_outside_declared_phase_v1",
            }
        )
    envelope["administrative_receipts"] = administrative
    try:
        timeline = derive_tick_native_performance_timeline_v1(envelope)
    except TickNativePerformanceTimelineError as exc:
        raise CampaignProviderError(
            f"tick-native performance timeline is invalid: {exc}"
        ) from exc
    return {
        "schema_version": "eiel_tick_native_performance_timeline_binding_v1",
        "launch_bound_receipt": envelope,
        "timeline": timeline,
    }


def _checkpoint_identity_contract_v1(
    *,
    execution_route_sha256: str,
    generation: int | None = None,
    tick: int | None = None,
    absolute_tick: int | None = None,
    next_absolute_tick: int | None = None,
    compatibility_interval_ticks: int | None = None,
    source_import_projection_digest_sha256s: Mapping[str, str],
    parent_checkpoint_projection_digest_sha256s: Mapping[str, str] | None,
    terminal_projection_digest_sha256s: Mapping[str, str],
    canonical_terminal_state_digest_sha256s: Mapping[str, str],
    canonical_checkpoint_storage_digest_sha256s: Mapping[str, str],
    _legacy_compatibility_in_provenance: bool = False,
) -> dict[str, Any]:
    """Label state and provenance digests without changing checkpoint mechanics."""

    maps: dict[str, Mapping[str, str]] = {
        "source_import_projection_digest_sha256s": (
            source_import_projection_digest_sha256s
        ),
        "terminal_projection_digest_sha256s": terminal_projection_digest_sha256s,
        "canonical_terminal_state_digest_sha256s": (
            canonical_terminal_state_digest_sha256s
        ),
        "canonical_checkpoint_storage_digest_sha256s": (
            canonical_checkpoint_storage_digest_sha256s
        ),
    }
    if parent_checkpoint_projection_digest_sha256s is not None:
        maps["parent_checkpoint_projection_digest_sha256s"] = (
            parent_checkpoint_projection_digest_sha256s
        )
    for name, values in maps.items():
        if (
            not isinstance(values, Mapping)
            or not values
            or any(not isinstance(cell_id, str) or not cell_id for cell_id in values)
            or any(not _valid_sha256(value) for value in values.values())
        ):
            raise CampaignProviderError(f"{name} is invalid")
    cell_ids = set(terminal_projection_digest_sha256s)
    if any(set(values) != cell_ids for values in maps.values()):
        raise CampaignProviderError("checkpoint identity digest cell sets differ")
    if not _valid_sha256(execution_route_sha256):
        raise CampaignProviderError("checkpoint identity execution route is invalid")
    if generation is not None and (type(generation) is not int or generation < 0):
        raise CampaignProviderError("checkpoint identity generation is invalid")
    if tick is not None and (type(tick) is not int or tick < 0):
        raise CampaignProviderError("checkpoint identity tick is invalid")
    if absolute_tick is not None and (
        type(absolute_tick) is not int or absolute_tick < 0
    ):
        raise CampaignProviderError("checkpoint identity absolute tick is invalid")
    if absolute_tick is not None and next_absolute_tick is None:
        next_absolute_tick = absolute_tick
    if absolute_tick is not None and next_absolute_tick != absolute_tick:
        raise CampaignProviderError(
            "checkpoint identity absolute and next ticks differ"
        )
    if next_absolute_tick is not None:
        if type(next_absolute_tick) is not int or next_absolute_tick < 0:
            raise CampaignProviderError("checkpoint absolute-tick identity is invalid")
        if generation is not None or tick is not None:
            if (
                type(compatibility_interval_ticks) is not int
                or compatibility_interval_ticks < 1
                or generation is None
                or tick is None
            ):
                raise CampaignProviderError(
                    "checkpoint compatibility facade is incomplete"
                )
            derived_generation, derived_tick = divmod(
                next_absolute_tick, compatibility_interval_ticks
            )
            if generation != derived_generation or tick != derived_tick:
                raise CampaignProviderError(
                    "checkpoint generation/tick facade differs from absolute next tick"
                )
    provenance_payload = {
        "schema_version": CHECKPOINT_IDENTITY_CONTRACT_SCHEMA_VERSION,
        "execution_route_sha256": execution_route_sha256,
        "source_import_projection_digest_sha256s": dict(
            sorted(source_import_projection_digest_sha256s.items())
        ),
        "parent_checkpoint_projection_digest_sha256s": (
            None
            if parent_checkpoint_projection_digest_sha256s is None
            else dict(sorted(parent_checkpoint_projection_digest_sha256s.items()))
        ),
        "terminal_projection_digest_sha256s": dict(
            sorted(terminal_projection_digest_sha256s.items())
        ),
    }
    if next_absolute_tick is not None:
        provenance_payload.update(
            {
                "time_authority": "absolute_tick_v1",
                "absolute_tick": next_absolute_tick,
                "next_absolute_tick": next_absolute_tick,
            }
        )
        if generation is not None and _legacy_compatibility_in_provenance:
            provenance_payload["legacy_compatibility"] = {
                "generation": generation,
                "tick": tick,
                "interval_ticks": compatibility_interval_ticks,
            }
    contract = {
        "schema_version": CHECKPOINT_IDENTITY_CONTRACT_SCHEMA_VERSION,
        "source_import_projection_digest_sha256s": dict(
            source_import_projection_digest_sha256s
        ),
        "parent_checkpoint_projection_digest_sha256s": (
            None
            if parent_checkpoint_projection_digest_sha256s is None
            else dict(parent_checkpoint_projection_digest_sha256s)
        ),
        "terminal_projection_digest_sha256s": dict(terminal_projection_digest_sha256s),
        "canonical_terminal_state_digest_sha256s": dict(
            canonical_terminal_state_digest_sha256s
        ),
        "canonical_checkpoint_storage_digest_sha256s": dict(
            canonical_checkpoint_storage_digest_sha256s
        ),
        "route_envelope_or_execution_provenance_digest_sha256": _canonical_sha256(
            provenance_payload
        ),
        "q5_terminal_equality_authority": "canonical_terminal_state_digest_sha256s",
        "route_provenance_may_differ_from_terminal_state": True,
        "legacy_import_digest_mapping": (
            "legacy_import_digest_maps_only_to_source_import_projection_digest_sha256"
        ),
    }
    if next_absolute_tick is not None:
        contract.update(
            {
                "time_authority": "absolute_tick_v1",
                "absolute_tick": next_absolute_tick,
                "next_absolute_tick": next_absolute_tick,
            }
        )
        if generation is not None:
            contract["legacy_compatibility"] = {
                "generation": generation,
                "tick": tick,
                "interval_ticks": compatibility_interval_ticks,
            }
    return contract


def _checkpoint_identity_sha256_v1(contract: Mapping[str, Any]) -> str:
    """Digest native identity without optional compatibility projections."""

    canonical = dict(contract)
    if canonical.get("time_authority") == "absolute_tick_v1":
        canonical.pop("legacy_compatibility", None)
    return _canonical_sha256(canonical)


def _validate_tick_native_checkpoint_identity_v1(
    checkpoint: Mapping[str, Any],
    *,
    execution_route_sha256: str,
    compatibility_interval_ticks: int | None = None,
) -> int:
    """Validate the absolute checkpoint authority without a legacy facade."""

    next_absolute_tick = checkpoint.get("next_absolute_tick")
    if type(next_absolute_tick) is not int or next_absolute_tick < 0:
        raise CampaignProviderError(
            "tick-native checkpoint next absolute tick is invalid"
        )
    if (
        checkpoint.get("time_authority") != "absolute_tick_v1"
        or checkpoint.get("absolute_tick") != next_absolute_tick
    ):
        raise CampaignProviderError(
            "tick-native checkpoint absolute-tick authority is invalid"
        )
    identity = checkpoint.get("checkpoint_identity_contract")
    if not isinstance(identity, Mapping):
        raise CampaignProviderError(
            "tick-native checkpoint identity contract is absent"
        )
    canonical = _checkpoint_identity_contract_v1(
        execution_route_sha256=execution_route_sha256,
        next_absolute_tick=next_absolute_tick,
        source_import_projection_digest_sha256s=checkpoint.get(
            "source_import_projection_digest_sha256s"
        ),
        parent_checkpoint_projection_digest_sha256s=checkpoint.get(
            "parent_checkpoint_projection_digest_sha256s"
        ),
        terminal_projection_digest_sha256s=checkpoint.get(
            "terminal_projection_digest_sha256s"
        ),
        canonical_terminal_state_digest_sha256s=checkpoint.get(
            "canonical_terminal_state_digest_sha256s"
        ),
        canonical_checkpoint_storage_digest_sha256s=checkpoint.get(
            "canonical_checkpoint_storage_digest_sha256s"
        ),
    )
    supplied_identity = dict(identity)
    supplied_compatibility = supplied_identity.pop("legacy_compatibility", None)
    if compatibility_interval_ticks is not None and (
        type(compatibility_interval_ticks) is not int
        or compatibility_interval_ticks < 1
    ):
        raise CampaignProviderError(
            "tick-native checkpoint compatibility interval is invalid"
        )
    historical: dict[str, Any] | None = None
    if compatibility_interval_ticks is not None:
        generation, tick = divmod(next_absolute_tick, compatibility_interval_ticks)
        # Historical identities put this derived facade in their provenance
        # digest. Reconstruct it only when the reader supplies the exact positive
        # interval; the checkpoint's absolute next tick remains the authority.
        historical = _checkpoint_identity_contract_v1(
            execution_route_sha256=execution_route_sha256,
            generation=generation,
            tick=tick,
            next_absolute_tick=next_absolute_tick,
            compatibility_interval_ticks=compatibility_interval_ticks,
            source_import_projection_digest_sha256s=checkpoint.get(
                "source_import_projection_digest_sha256s"
            ),
            parent_checkpoint_projection_digest_sha256s=checkpoint.get(
                "parent_checkpoint_projection_digest_sha256s"
            ),
            terminal_projection_digest_sha256s=checkpoint.get(
                "terminal_projection_digest_sha256s"
            ),
            canonical_terminal_state_digest_sha256s=checkpoint.get(
                "canonical_terminal_state_digest_sha256s"
            ),
            canonical_checkpoint_storage_digest_sha256s=checkpoint.get(
                "canonical_checkpoint_storage_digest_sha256s"
            ),
            _legacy_compatibility_in_provenance=True,
        )
        expected_compatibility = historical["legacy_compatibility"]
        if (
            supplied_compatibility is not None
            and supplied_compatibility != expected_compatibility
        ):
            raise CampaignProviderError(
                "tick-native checkpoint compatibility projection is inconsistent"
            )
    elif supplied_compatibility is not None:
        raise CampaignProviderError(
            "tick-native checkpoint compatibility requires an explicit interval"
        )
    supplied_digest = checkpoint.get("checkpoint_identity_sha256")
    canonical_identity = dict(canonical)
    canonical_identity.pop("legacy_compatibility", None)
    canonical_pair = (
        supplied_identity == canonical_identity
        and supplied_digest == _checkpoint_identity_sha256_v1(canonical)
    )
    historical_pair = False
    if historical is not None:
        historical_identity = dict(historical)
        historical_identity.pop("legacy_compatibility", None)
        historical_pair = (
            supplied_identity == historical_identity
            and supplied_digest == _canonical_sha256(historical)
        )
    accepted_pair = canonical_pair or historical_pair
    if not accepted_pair:
        raise CampaignProviderError(
            "tick-native checkpoint identity digest differs from recomputed authority"
        )
    return next_absolute_tick


def _checkpoint_snapshot_sequence_v1(
    *, generation: int, absolute_next_tick: int | None
) -> int:
    """Use the absolute committed frontier whenever the route supplies one."""

    return generation if absolute_next_tick is None else absolute_next_tick


def direct_resident_detached_checkpoint_authority_factory_v1(
    *,
    provider_id: str,
    batch_id: str,
    cell_ids: Sequence[str],
    generation: int,
    tick: int,
    absolute_next_tick: int | None,
    segment_ordinal: int,
) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Build checkpoint authority on the detached host settlement worker.

    The physical producer cannot know terminal projection digests before the
    host-only canonical projector runs.  This pure factory retains only scalar
    route identity and computes the same source, lifecycle, observer, and
    layered checkpoint authorities while each B1 projection is live.
    """

    normalized_ids = tuple(cell_ids)
    if (
        type(provider_id) is not str
        or not provider_id
        or type(batch_id) is not str
        or not batch_id
        or not normalized_ids
        or len(set(normalized_ids)) != len(normalized_ids)
        or any(type(value) is not str or not value for value in normalized_ids)
        or type(generation) is not int
        or generation < 0
        or type(tick) is not int
        or tick < 0
        or (
            absolute_next_tick is not None
            and (type(absolute_next_tick) is not int or absolute_next_tick < 0)
        )
        or type(segment_ordinal) is not int
        or segment_ordinal < 0
    ):
        raise CampaignProviderError(
            "detached physical checkpoint authority coordinates are invalid"
        )
    cell_indices = {cell_id: index for index, cell_id in enumerate(normalized_ids)}
    snapshot_sequence = _checkpoint_snapshot_sequence_v1(
        generation=generation,
        absolute_next_tick=absolute_next_tick,
    )

    def authority_factory(
        cell_id: str, projection: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        cell_index = cell_indices.get(cell_id)
        if cell_index is None or not isinstance(projection, Mapping):
            raise CampaignProviderError(
                "detached physical checkpoint authority cell is invalid"
            )
        from engine.native_runtime.checkpoint_digests import (
            DigestLayer,
            checkpoint_digest_manifest,
            checkpoint_source_projection_authority_manifest,
        )
        from engine.native_runtime.lifecycle_events import (
            typed_lifecycle_events_from_projection,
            validated_typed_lifecycle_event_cursor,
        )

        source_digest = str(
            checkpoint_source_projection_authority_manifest(projection)[
                "source_projection_sha256"
            ]
        )
        event_cursor = validated_typed_lifecycle_event_cursor(
            projection,
            typed_lifecycle_events_from_projection(projection),
        )
        owner_epoch = max(1, snapshot_sequence, segment_ordinal)
        controller_id = f"{provider_id}-{cell_id}"
        identity = {
            "run_id": batch_id,
            "controller_id": controller_id,
            "owner_epoch": owner_epoch,
            "snapshot_id": (
                f"{controller_id}:owner:{owner_epoch}:boundary:{snapshot_sequence}"
            ),
            "snapshot_sequence": snapshot_sequence,
            "snapshot_digest_sha256": source_digest,
            "next_generation": generation,
            "tick": tick,
            "phase": "generation_boundary" if tick == 0 else "partial_generation",
            "source_projection_sha256": source_digest,
        }
        return {
            "source_projection_sha256": source_digest,
            "architecture_b_identity": identity,
            "observer_continuation": ProductionEngineLifecycleV1._observer_continuation(
                generation,
                tick=tick,
                event_cursor=event_cursor,
                snapshot_sequence=snapshot_sequence,
                absolute_tick_authority=absolute_next_tick is not None,
            ),
            "mechanics_checkpoint_digest": str(
                checkpoint_digest_manifest(
                    projection, layer=DigestLayer.TRAJECTORY_STATE
                )["digest"]
            ),
            "canonical_checkpoint_storage_digest": str(
                checkpoint_digest_manifest(
                    projection, layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE
                )["digest"]
            ),
            "cuda_source_field_count": 0,
            "cell_index": cell_index,
        }

    return authority_factory


def _require_tick_native_event_sink_builder_v1(
    *, tick_native: bool, accepts_event_sink: bool
) -> None:
    """Bar cumulative in-memory event history on active tick-native owners."""

    if tick_native and not accepts_event_sink:
        raise CampaignProviderError(
            "tick-native request builder must accept event_history_sink_path"
        )


def _validate_tick_native_direct_event_storage_v1(
    direct_result: object, *, tick_native: bool
) -> None:
    """Require native proof that event history stayed in the resident sink."""

    if tick_native and (
        not isinstance(direct_result, Mapping)
        or direct_result.get("event_history_storage_mode")
        != "eiel_resident_event_sink_v1"
    ):
        raise CampaignProviderError(
            "tick-native dispatch did not bind the resident event-history sink"
        )


def _reconcile_tick_native_resume_event_sink_v1(
    source_projections: Sequence[Mapping[str, Any]],
    *,
    expected_cells: int,
) -> dict[str, Any]:
    """Restore the shared sink to every authenticated checkpoint prefix once."""

    references: dict[int, Mapping[str, Any]] = {}
    sink_paths: set[str] = set()
    for projection in source_projections:
        metadata = (
            projection.get("metadata") if isinstance(projection, Mapping) else None
        )
        reference = (
            metadata.get("event_history_reference")
            if isinstance(metadata, Mapping)
            and metadata.get("event_history_storage_mode")
            == "eiel_resident_event_sink_v1"
            else None
        )
        if not isinstance(reference, Mapping):
            raise CampaignProviderError(
                "tick-native resume projection lacks an authenticated event-history reference"
            )
        cell_index = reference.get("cell_index")
        path = reference.get("path")
        if (
            type(cell_index) is not int
            or cell_index < 0
            or cell_index in references
            or not isinstance(path, str)
            or not path
        ):
            raise CampaignProviderError(
                "tick-native resume event-history references are malformed"
            )
        references[cell_index] = dict(reference)
        sink_paths.add(str(Path(path).expanduser().resolve()))
    if set(references) != set(range(expected_cells)) or len(sink_paths) != 1:
        raise CampaignProviderError(
            "tick-native resume event-history references do not cover one shared sink"
        )
    from engine.perf.cuda_architecture_b_event_sink_v1 import (
        EventSinkError,
        reconcile_event_sink_to_checkpoint_prefixes_v1,
    )

    try:
        receipt = reconcile_event_sink_to_checkpoint_prefixes_v1(
            next(iter(sink_paths)), references_by_cell=references
        )
    except EventSinkError as exc:
        raise CampaignProviderError(
            f"cannot reconcile tick-native event-history checkpoint prefixes: {exc}"
        ) from exc
    if not isinstance(receipt, Mapping):
        raise CampaignProviderError("event-history reconciliation returned no receipt")
    return dict(receipt)


def _generation_zero_projection_payload_v1(value: Any) -> dict[str, Any]:
    """Materialize a complete mapping from the typed G0 import projection."""

    from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (  # noqa: PLC0415
        CudaArchitectureBDirectResidentSourceBuilderError,
        canonical_generation_zero_source_projection_v1,
    )

    try:
        return canonical_generation_zero_source_projection_v1(value)
    except CudaArchitectureBDirectResidentSourceBuilderError as exc:
        raise CampaignProviderError(str(exc)) from exc


def _production_candidate_authority_sha256_v1(
    *,
    provider_id: str,
    profile_id: str,
    qualification_receipt: Mapping[str, Any],
    reporting_receipt: Mapping[str, Any],
    artifact_row: Mapping[str, Any],
) -> str:
    """Derive the prepared-owner authority identity for a qualified artifact."""

    payload = {
        "schema_version": ("eiel_production_engine_candidate_authority_sha256_v1"),
        "authorization_scope": "production_engine_prepared_owner_audit_v1",
        "production_authority": True,
        "provider_id": provider_id,
        "profile_id": profile_id,
        "mechanics_binding_id": qualification_receipt.get("mechanics_binding_id"),
        "artifact_abi": qualification_receipt.get("artifact_abi"),
        "authority_scope": qualification_receipt.get("authority_scope"),
        "artifact_sha256s": {
            "cuda_extension": artifact_row.get("sha256"),
            "extension_manifest": artifact_row.get("manifest_sha256"),
            "qualification_receipt": hashlib.sha256(
                json.dumps(
                    qualification_receipt,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest(),
            "reporting_receipt": hashlib.sha256(
                json.dumps(
                    reporting_receipt,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest(),
        },
    }
    if (
        not isinstance(payload["mechanics_binding_id"], str)
        or not isinstance(payload["artifact_abi"], str)
        or not isinstance(payload["authority_scope"], str)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in payload["artifact_sha256s"].values()
        )
    ):
        raise CampaignProviderError(
            "production-engine candidate authority identity is incomplete"
        )
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION = (
    "eiel_patch_fixed_fresh_owner_restore_receipt_v1"
)
FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION = (
    "eiel_patch_fixed_fresh_owner_checkpoint_rebind_v1"
)
WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1 = "wrc_checkpoint_adaptive_rebinding_v1"


@dataclass(frozen=True, slots=True)
class ResidentBatchRequestV1:
    """Provider-neutral, immutable inputs for one resident physical batch."""

    profile_id: str
    batch_id: str
    cell_ids: tuple[str, ...]
    checkpoint_milestones: tuple[int, ...]
    parameter_arrays: tuple[tuple[str, tuple[Any, ...]], ...]
    seed_arrays: tuple[tuple[str, tuple[int, ...]], ...]
    generations: int | None = None
    ticks_per_generation: int | None = None
    agent_id_allocation_generation_count: int | None = None
    checkpoint_storage_mode: str = "disk"
    reporting_enabled: bool = True
    execution_route_sha256: str | None = None
    genome_v2_checkpoint_authority_binding: tuple[tuple[str, str], ...] = ()
    total_ticks: int | None = None
    start_next_tick: int = 0
    checkpoint_ticks: tuple[int, ...] = ()
    deliberate_restore_tick: int | None = None
    deliberate_restore_ticks: tuple[int, ...] = ()
    internal_lifecycle_interval_ticks: int | None = None
    max_internal_lifecycle_segments_per_dispatch: int | None = None
    acs_checkpoint_slots: int = 0
    acs_capture_instrumentation: bool = False
    native_streamed_checkpoint: bool = False
    world_sharded_checkpoint: bool = False
    # Active tick-native controls. Generation fields are absent on this route.
    resident_owner_span_ticks: int | None = None
    native_dispatch_window_ticks: int | None = None
    report_every_ticks: int | None = None
    science_sample_interval_ticks: int | None = None
    reducer_every_ticks: int | None = None
    archive_every_ticks: int | None = None
    event_window_capacity: int | None = None
    source_window_ticks: int | None = None
    tick_native_control: Mapping[str, Any] = field(default_factory=dict)
    # Optional closed native-control receipt used by science CUDA V2.  Keeping
    # this additive and absent for ordinary providers preserves their payload
    # shape while ensuring the provider/direct-sparse seam cannot lose it.
    native_control: Mapping[str, Any] | None = None
    native_rng_world_uids: tuple[int, ...] = ()
    native_world_controls: tuple[Mapping[str, Any], ...] = ()

    def payload(self) -> dict[str, Any]:
        value = {
            "profile_id": self.profile_id,
            "batch_id": self.batch_id,
            "cell_ids": list(self.cell_ids),
            "parameter_arrays": {
                name: list(values) for name, values in self.parameter_arrays
            },
            "seed_arrays": {name: list(values) for name, values in self.seed_arrays},
            "agent_id_allocation_generation_count": (
                self.agent_id_allocation_generation_count
            ),
            "checkpoint_storage_mode": self.checkpoint_storage_mode,
            "reporting_enabled": self.reporting_enabled,
            "execution_route_sha256": self.execution_route_sha256,
            "genome_v2_checkpoint_authority_binding": dict(
                self.genome_v2_checkpoint_authority_binding
            ),
        }
        if self.acs_checkpoint_slots:
            value["acs_checkpoint_slots"] = self.acs_checkpoint_slots
        if self.acs_capture_instrumentation:
            value["acs_capture_instrumentation"] = True
        if self.native_streamed_checkpoint:
            value["native_streamed_checkpoint"] = True
        if self.world_sharded_checkpoint:
            value["world_sharded_checkpoint"] = True
        if self.native_control is not None:
            value["native_control"] = dict(self.native_control)
        if self.native_rng_world_uids:
            value["native_rng_world_uids"] = list(self.native_rng_world_uids)
        if self.native_world_controls:
            value["native_world_controls"] = [
                dict(control) for control in self.native_world_controls
            ]
        if self.total_ticks is None:
            if self.generations is None or self.ticks_per_generation is None:
                raise CampaignProviderError(
                    "legacy resident requests require generation compatibility fields"
                )
            value.update(
                {
                    "generations": self.generations,
                    "ticks_per_generation": self.ticks_per_generation,
                    "checkpoint_milestones": list(self.checkpoint_milestones),
                }
            )
        else:
            if (
                self.generations is not None
                or self.ticks_per_generation is not None
                or self.internal_lifecycle_interval_ticks is not None
                or self.max_internal_lifecycle_segments_per_dispatch is not None
            ):
                raise CampaignProviderError(
                    "tick-native resident requests forbid generation/lifecycle adapters"
                )
            value.update(
                {
                    "time_authority": "absolute_tick_v1",
                    "total_ticks": self.total_ticks,
                    "start_next_tick": self.start_next_tick,
                    "checkpoint_ticks": list(self.checkpoint_ticks),
                    "deliberate_restore_tick": self.deliberate_restore_tick,
                    "deliberate_restore_ticks": list(self.deliberate_restore_ticks),
                    "tick_native_control": dict(self.tick_native_control),
                }
            )
        return value


class CampaignExecutionProviderV1(Protocol):
    """One coherent runtime ownership boundary registered for a profile."""

    provider_id: str
    profile_id: str

    def translate_batch(
        self, plan: CampaignPlanV1 | Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1: ...

    def readiness_reasons(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]: ...

    def execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1: ...


def _plan_payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise CampaignProviderError("plan must be a CampaignPlanV1 or mapping")
    return dict(plan)


def _callable_accepts_keyword_v1(callable_value: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_value).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


_ROUTE_SOURCE_SUFFIXES_V1 = frozenset(
    {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
)


def _route_source_projection_v1(provider_module: str) -> dict[str, Any]:
    """Conservatively hash broad behavior-owning packages.

    The current native dependency closure is not generated mechanically, so the
    safe first version deliberately invalidates all Patch routes when any source
    in these owning packages changes.
    """

    engine_root = Path(__file__).resolve().parents[1]
    roots = (
        engine_root / "campaigns",
        engine_root / "native_runtime",
        engine_root / "perf",
        engine_root / "simulation",
    )
    module = sys.modules.get(provider_module)
    module_path = Path(str(getattr(module, "__file__", "")))
    candidates: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates.add(root.resolve())
        elif root.is_dir():
            candidates.update(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix in _ROUTE_SOURCE_SUFFIXES_V1
            )
    if module_path.is_file():
        candidates.add(module_path.resolve())
    digest = hashlib.sha256()
    for path in sorted(candidates):
        try:
            relative = path.relative_to(engine_root.parent).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(ProductionEngineLifecycleV1._file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "projection_mode": "conservative_broad_package_v1",
        "source_roots": [
            "engine/campaigns",
            "engine/native_runtime",
            "engine/perf",
            "engine/simulation",
            f"provider_module:{provider_module}",
        ],
        "source_file_count": len(candidates),
        "source_projection_sha256": digest.hexdigest(),
    }


_QUALIFICATION_TENSOR_LAYOUTS_V1: Final[Mapping[str, tuple[int, int, int | None]]] = {
    "runtime_bookkeeping_by_tick": (3, 0, 1),
    "drive_noise_rng_semantic_draws_by_tick": (2, 0, 1),
    "event_tape_active_count": (2, 0, 1),
    "actions_by_tick_slot": (3, 0, 1),
    "patch_stats_by_tick": (3, 0, 1),
}


def _qualification_tensor_row_v1(
    value: Any,
    *,
    name: str,
    cell_index: int,
    tick_index: int,
    batch_cells: int,
) -> Any | None:
    if value is None:
        return None
    layout = _QUALIFICATION_TENSOR_LAYOUTS_V1.get(name)
    if layout is None:
        raise CampaignProviderError(
            f"qualification tensor has no declared axis contract: {name}"
        )
    rank, cell_axis, tick_axis = layout
    if (
        not hasattr(value, "detach")
        or not hasattr(value, "shape")
        or len(value.shape) != rank
        or type(batch_cells) is not int
        or batch_cells < 1
        or not 0 <= cell_index < batch_cells
        or int(value.shape[cell_axis]) != batch_cells
        or type(tick_index) is not int
        or tick_index < 0
        or (tick_axis is not None and tick_index >= int(value.shape[tick_axis]))
    ):
        raise CampaignProviderError(
            f"qualification tensor differs from its declared axis contract: {name}"
        )
    row = value.select(cell_axis, cell_index)
    if tick_axis is not None:
        adjusted_tick_axis = tick_axis - int(tick_axis > cell_axis)
        row = row.select(adjusted_tick_axis, tick_index)
    return row.detach().to(device="cpu", copy=True).contiguous()


def _qualification_tensor_digest_v1(value: Any) -> str | None:
    if value is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(getattr(value, "dtype", "unknown")).encode("ascii"))
    digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _qualification_tensor_integer_v1(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value.sum().item())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _qualification_transition_sentinels_v1(
    direct_result: Any,
    request: ResidentBatchRequestV1,
    *,
    start_generation: int,
    segment_generations: int,
    start_next_tick: int | None = None,
    end_next_tick: int | None = None,
    restored_entry: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Project compact high-risk transition observations from existing outputs."""

    tick_native = request.total_ticks is not None
    ticks = request.source_window_ticks if tick_native else request.ticks_per_generation
    if type(ticks) is not int or ticks < 1:
        raise CampaignProviderError(
            "qualification transition sentinel tick interval is invalid"
        )
    points: list[tuple[str, int, int, int, int]] = []
    if tick_native:
        if (
            type(start_next_tick) is not int
            or type(end_next_tick) is not int
            or not 0 <= start_next_tick < end_next_tick <= request.total_ticks
        ):
            raise CampaignProviderError(
                "tick-native qualification sentinel span is invalid"
            )
        segment_ticks = end_next_tick - start_next_tick

        def compatibility_generation(absolute_tick: int) -> int:
            # RouteTransitionSentinelV1 retains this historical field.  It is
            # a compatibility-only projection; absolute_tick owns time.
            return absolute_tick // ticks + 1

        if start_next_tick > 0:
            entry_generation = compatibility_generation(start_next_tick)
            if restored_entry:
                points.extend(
                    [
                        (
                            "first_tick_after_restore",
                            0,
                            0,
                            entry_generation,
                            start_next_tick,
                        ),
                        (
                            "first_policy_cache_refill_after_restore",
                            0,
                            0,
                            entry_generation,
                            start_next_tick,
                        ),
                    ]
                )
                second_name = "second_tick_after_restore"
            else:
                points.append(
                    (
                        "first_tick_after_boundary",
                        0,
                        0,
                        entry_generation,
                        start_next_tick,
                    )
                )
                second_name = "second_tick_after_boundary"
            if segment_ticks > 1:
                points.append(
                    (
                        second_name,
                        1,
                        1,
                        compatibility_generation(start_next_tick + 1),
                        start_next_tick + 1,
                    )
                )
        terminal_tick = end_next_tick - 1
        terminal_generation = compatibility_generation(terminal_tick)
        points.extend(
            [
                (
                    "terminal_settlement",
                    segment_ticks - 1,
                    segment_ticks - 1,
                    terminal_generation,
                    terminal_tick,
                ),
                (
                    "checkpoint_export_boundary",
                    segment_ticks,
                    segment_ticks,
                    terminal_generation,
                    end_next_tick,
                ),
            ]
        )
    else:
        total_ticks = segment_generations * ticks
        if start_generation > 0:
            restore_generation = start_generation + 1
            points.extend(
                [
                    (
                        "first_tick_after_restore",
                        0,
                        0,
                        restore_generation,
                        start_generation * ticks,
                    ),
                    (
                        "first_policy_cache_refill_after_restore",
                        0,
                        0,
                        restore_generation,
                        start_generation * ticks,
                    ),
                ]
            )
            if total_ticks > 1:
                points.append(
                    (
                        "second_tick_after_restore",
                        1,
                        1,
                        restore_generation,
                        start_generation * ticks + 1,
                    )
                )
        for boundary in range(1, segment_generations):
            before = boundary * ticks - 1
            after = boundary * ticks
            before_generation = start_generation + boundary
            after_generation = before_generation + 1
            points.extend(
                [
                    (
                        "last_tick_before_generation_boundary",
                        before,
                        before,
                        before_generation,
                        start_generation * ticks + before,
                    ),
                    (
                        "generation_boundary",
                        before,
                        after,
                        after_generation,
                        start_generation * ticks + before,
                    ),
                    (
                        "first_tick_after_boundary",
                        after,
                        after,
                        after_generation,
                        start_generation * ticks + after,
                    ),
                    (
                        "first_policy_cache_refill_after_boundary",
                        after,
                        after,
                        after_generation,
                        start_generation * ticks + after,
                    ),
                ]
            )
            if after + 1 < total_ticks:
                points.append(
                    (
                        "second_tick_after_boundary",
                        after + 1,
                        after + 1,
                        after_generation,
                        start_generation * ticks + after + 1,
                    )
                )
        terminal_tick = total_ticks - 1
        terminal_generation = start_generation + segment_generations
        points.extend(
            [
                (
                    "terminal_settlement",
                    terminal_tick,
                    terminal_tick,
                    terminal_generation,
                    start_generation * ticks + terminal_tick,
                ),
                (
                    "checkpoint_export_boundary",
                    terminal_tick,
                    total_ticks,
                    terminal_generation,
                    start_generation * ticks + terminal_tick,
                ),
            ]
        )
    result: dict[str, list[dict[str, Any]]] = {
        cell_id: [] for cell_id in request.cell_ids
    }
    batch_cells = len(request.cell_ids)
    for cell_index, cell_id in enumerate(request.cell_ids):
        for transition, row_tick, window_position, generation, absolute_tick in points:

            def row(name: str) -> Any | None:
                return _qualification_tensor_row_v1(
                    getattr(direct_result, name, None),
                    name=name,
                    cell_index=cell_index,
                    tick_index=row_tick,
                    batch_cells=batch_cells,
                )

            bookkeeping = row("runtime_bookkeeping_by_tick")
            refill_calls = None
            if bookkeeping is not None and bookkeeping.numel() >= 2:
                refill_calls = int(bookkeeping.reshape(-1)[-2].item())
            drive_cursor = _qualification_tensor_integer_v1(
                row("drive_noise_rng_semantic_draws_by_tick")
            )
            event_cursor = _qualification_tensor_integer_v1(
                row("event_tape_active_count")
            )
            result[cell_id].append(
                RouteTransitionSentinelV1(
                    transition_name=transition,
                    absolute_generation=generation,
                    absolute_tick=absolute_tick,
                    physical_window_position=window_position,
                    rng_cursor=drive_cursor,
                    refill_calls=refill_calls,
                    action_digest=_qualification_tensor_digest_v1(
                        row("actions_by_tick_slot")
                    ),
                    world_digest=_qualification_tensor_digest_v1(
                        row("patch_stats_by_tick")
                    ),
                    profile_fields=(
                        {"event_tape_active_count": event_cursor}
                        if event_cursor is not None
                        else {}
                    ),
                ).payload()
            )
    return result


class ProductionEngineLifecycleV1:
    """Single production owner for execution and persistence lifecycle.

    This owner contains batching, segmentation, dispatch orchestration,
    checkpoint publication/reload, fresh-owner restoration, settlement,
    reporting projection, and route qualification. World, controller, and
    genome mechanics are being extracted behind configuration components and
    must not acquire their own copies of this lifecycle.
    """

    provider_id = PRODUCTION_PROVIDER_ID
    profile_id = PRODUCTION_PROFILE_ID
    qualification_schema_version = PRODUCTION_MECHANIC_QUALIFICATION_SCHEMA_VERSION
    checkpoint_schema_version = PRODUCTION_CHECKPOINT_SCHEMA_VERSION
    artifact_abi = "architecture_b_patch_p2_stage3_v2"

    def _generation_series_required(self, plan: Mapping[str, Any]) -> bool:
        """Return whether this configured mechanics path owns exact series data."""

        del plan
        return getattr(self, "generation_series_required", False) is True

    reporting_qualification_schema_version = (
        PRODUCTION_REPORTING_QUALIFICATION_SCHEMA_VERSION
    )
    required_artifact_exports = (
        "eiel_cuda_architecture_b_execute_direct_resident_windows_v1",
    )
    route_controller_contract = "flattened_fixed_mlp_controller_v1"
    route_policy_cache_contract = "flattened_fixed_mlp_policy_cache_v1"
    route_refill_contract = "generation_boundary_policy_cache_refill_v1"
    route_rng_contract = "native_patch_stochastic_rng_and_gaussian_tape_v1"
    route_lifecycle_contract = "direct_resident_patch_lifecycle_owner_v1"
    route_settlement_contract = "compact_summary_terminal_settlement_v1"
    route_requires_policy_cache_refill = True
    route_supports_turnover = True
    reporting_parity_supported = True
    resident_segment_generation_min = 3
    resident_segment_generation_max = 128

    @staticmethod
    def campaign_metric_rows(
        rows: Sequence[Mapping[str, Any]], *, cell_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Project native compact rows onto the generic terminal metric contract."""

        if not rows:
            return []
        expected = set(cell_ids)
        latest: dict[str, tuple[int, dict[str, Any]]] = {}
        for ordinal, raw in enumerate(rows):
            row = dict(raw)
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or cell_id not in expected:
                raise CampaignProviderError(
                    "Patch compact metric row has an unknown cell identity"
                )
            generation = row.get("generation", ordinal)
            if type(generation) is not int or generation < 0:
                raise CampaignProviderError(
                    "Patch compact metric row has an invalid generation"
                )
            supplied = row.get("metrics")
            if isinstance(supplied, Mapping):
                metrics = dict(supplied)
            else:
                native_names = {
                    "population.alive": "alive",
                    "population.births": "births",
                    "population.deaths": "deaths",
                }
                if any(name not in row for name in native_names.values()):
                    raise CampaignProviderError(
                        "Patch compact metric row lacks the generic population projection"
                    )
                metrics = {
                    public: row[native] for public, native in native_names.items()
                }
            if any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or (isinstance(value, float) and not math.isfinite(value))
                )
                for value in metrics.values()
            ):
                raise CampaignProviderError(
                    "Patch campaign metric values must be finite numeric values or null"
                )
            previous = latest.get(cell_id)
            if previous is not None and generation == previous[0]:
                raise CampaignProviderError(
                    "Patch compact metrics contain a duplicate cell generation"
                )
            if previous is None or generation > previous[0]:
                latest[cell_id] = (generation, metrics)
        if set(latest) != expected:
            raise CampaignProviderError(
                "Patch compact metrics do not cover every physical batch cell"
            )
        return [
            {"cell_id": cell_id, "metrics": latest[cell_id][1]} for cell_id in cell_ids
        ]

    def __init__(
        self,
        *,
        artifact_loader: Callable[..., Any] | None = None,
        runner_builder: Callable[..., Any] | None = None,
        request_builder: Callable[..., Any] | None = None,
        preparer: Callable[..., Any] | None = None,
        executor: Callable[..., Mapping[str, Any]] | None = None,
        projector: Callable[..., Any] | None = None,
        checkpoint_writer: Callable[..., str] | None = None,
        checkpoint_loader: Callable[..., Mapping[str, Any]] | None = None,
        checkpoint_decoder: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
        checkpoint_digest: Callable[[Mapping[str, Any]], str] | None = None,
        source_preparer: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        """Create the real provider, with narrow seams for contract testing."""

        self._artifact_loader = artifact_loader
        self._runner_builder = runner_builder
        self._request_builder = request_builder
        self._preparer = preparer
        self._executor = executor
        self._projector = projector
        self._checkpoint_writer = checkpoint_writer
        self._checkpoint_loader = checkpoint_loader
        self._checkpoint_decoder = checkpoint_decoder
        self._checkpoint_digest = checkpoint_digest
        self._source_preparer = source_preparer
        self._active_acs_pipeline: Any | None = None

    def execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        """Describe the selected behavior route independently of campaign extent."""

        payload = _plan_payload(plan)
        execution = payload.get("execution")
        acs_checkpoint_slots = (
            execution.get("acs_checkpoint_slots", 0)
            if isinstance(execution, Mapping)
            else 0
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
            0,
            1,
            2,
            3,
        }:
            raise CampaignProviderError("execution route ACS slots are invalid")
        raw_acs_capture_instrumentation = (
            execution.get("acs_capture_instrumentation", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "execution route ACS capture instrumentation is invalid"
            )
        acs_capture_instrumentation = raw_acs_capture_instrumentation or bool(
            acs_checkpoint_slots
        )
        raw_native_streamed_checkpoint = (
            execution.get("native_streamed_checkpoint", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_native_streamed_checkpoint) is not bool:
            raise CampaignProviderError(
                "execution route native streamed checkpoint selection is invalid"
            )
        raw_world_sharded_checkpoint = (
            execution.get("world_sharded_checkpoint", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_world_sharded_checkpoint) is not bool:
            raise CampaignProviderError(
                "execution route world-sharded checkpoint selection is invalid"
            )
        native_acs_runtime_component = _acs_native_runtime_route_component_v1(
            _plan_payload(plan)
        )
        tick_native = (
            isinstance(execution, Mapping)
            and execution.get("execution_route_identity")
            == "tick_native_execution_route_v1"
        )
        tick_native_control_payload: Mapping[str, Any] | None = None
        if tick_native:
            try:
                tick_native_control_payload = TickNativeControlV1.from_execution(
                    execution
                ).payload()
            except TickNativeControlError as exc:
                raise CampaignProviderError(
                    f"execution route tick-native control is invalid: {exc}"
                ) from exc
            if acs_checkpoint_slots not in {0, 1, 2}:
                raise CampaignProviderError(
                    "tick-native execution route ACS slots must be zero, one, or two"
                )
            if acs_checkpoint_slots and not raw_native_streamed_checkpoint:
                raise CampaignProviderError(
                    "tick-native asynchronous ACS requires native_streamed_checkpoint"
                )
            if raw_world_sharded_checkpoint and (
                not raw_native_streamed_checkpoint or acs_checkpoint_slots not in {1, 2}
            ):
                raise CampaignProviderError(
                    "world-sharded checkpoint route requires tick-native ACS1 or ACS2"
                )
        if request is not None and (
            request.acs_checkpoint_slots != acs_checkpoint_slots
            or request.acs_capture_instrumentation is not acs_capture_instrumentation
            or request.native_streamed_checkpoint is not raw_native_streamed_checkpoint
            or request.world_sharded_checkpoint is not raw_world_sharded_checkpoint
        ):
            raise CampaignProviderError(
                "resident request ACS checkpoint route binding mismatch"
            )

        def seam_id(name: str, default: str) -> str:
            value = getattr(self, name)
            if value is None:
                return default
            module = getattr(value, "__module__", type(value).__module__)
            qualname = getattr(value, "__qualname__", type(value).__qualname__)
            return f"injected:{module}.{qualname}"

        source_projection = _route_source_projection_v1(type(self).__module__)
        components: tuple[Mapping[str, Any], ...] = (
            {
                "component_id": "provider_translation",
                "implementation": (
                    f"{type(self).__module__}.{type(self).__qualname__}"
                ),
                "contract_version": "resident_batch_translation_v1",
            },
            {
                "component_id": "runner_factory",
                "implementation": seam_id(
                    "_runner_builder", f"{self.profile_id}:default_runner_factory"
                ),
                "contract_version": "provider_localized_runner_factory_v1",
            },
            {
                "component_id": "native_dispatch",
                "artifact_abi": self.artifact_abi,
                "entrypoints": sorted(self.required_artifact_exports),
            },
            {
                "component_id": "world_route",
                "contract_version": self.qualification_schema_version,
            },
            {
                "component_id": "controller",
                "contract_version": self.route_controller_contract,
            },
            *(
                (
                    {
                        "component_id": "tick_native_control",
                        "contract_version": "eiel-tick-native-control-v1",
                        "time_authority": "absolute_tick_v1",
                        "control": dict(tick_native_control_payload),
                    },
                )
                if tick_native_control_payload is not None
                else ()
            ),
            {
                "component_id": "lifecycle_owner",
                "contract_version": self.route_lifecycle_contract,
                "implementation": seam_id(
                    "_executor", f"{self.profile_id}:default_window_executor"
                ),
            },
            *(
                (
                    {
                        "component_id": "policy_cache_refill",
                        "cache_contract": self.route_policy_cache_contract,
                        "refill_contract": self.route_refill_contract,
                    },
                )
                if not tick_native
                else (
                    {
                        "component_id": "policy_cache_restore",
                        "cache_contract": self.route_policy_cache_contract,
                        "restore_contract": "fresh_owner_policy_cache_restore_v1",
                        "dispatch_boundary_refill": False,
                    },
                )
            ),
            {
                "component_id": "rng_and_tape",
                "contract_version": self.route_rng_contract,
            },
            {
                "component_id": "birth_and_turnover",
                "contract_version": "native_patch_birth_turnover_v1",
                "turnover_enabled": self.route_supports_turnover,
            },
            {
                "component_id": "checkpoint_restore",
                "checkpoint_contract": self.checkpoint_schema_version,
                "restore_contract": "fresh_owner_exact_projection_restore_v1",
                **(
                    {
                        "persistence_route": (
                            "acs_detached_owner_transition_pipeline_v2"
                            if tick_native
                            else "acs_bounded_single_writer_v1"
                        ),
                        "immutable_snapshot_slots": acs_checkpoint_slots,
                        "backpressure": "block_without_overwrite_or_drop",
                        "frontiers": "compute_and_durable_checkpoint_distinct",
                        **(
                            {
                                "worker_boundary": "path_and_scalar_metadata_only",
                                "producer_capture": (
                                    "bounded_cell_major_file_spool_v2"
                                ),
                                "worker_materialization": (
                                    "field_major_aggregate_v2"
                                ),
                                "successor_source": (
                                    "sealed_cell_major_spool_one_cell_cursor_v1"
                                ),
                                "spool_lifetime": (
                                    "independent_writer_and_successor_source_leases_v1"
                                ),
                                "owner_transition_persistence": (
                                    "detached_without_old_new_cuda_owner_overlap_v1"
                                ),
                                "worker_accepts_cuda_objects": False,
                                "worker_accepts_callbacks": False,
                                "capture_pool": "one_shared_two_slab_not_per_slot",
                            }
                            if tick_native
                            else {}
                        ),
                    }
                    if acs_checkpoint_slots
                    else {}
                ),
                **(
                    {
                        "physical_format": "acs100x_native_streamed_checkpoint_v1",
                        "multicell_physical_format": (
                            "acs100x_native_aggregate_checkpoint_v2"
                        ),
                        "physical_format_selection": (
                            "aggregate_for_multicell_batch_v1"
                        ),
                        "strict_prebuilt_extension": True,
                        "native_extension_abi": "eiel_native_runtime_abi_v1",
                    }
                    if raw_native_streamed_checkpoint
                    else {}
                ),
                **(
                    {"capture_instrumentation": ("acs_capture_barrier_measurement_v1")}
                    if acs_capture_instrumentation
                    else {}
                ),
            },
            {
                "component_id": "settlement_reporting",
                "settlement_contract": self.route_settlement_contract,
                "reporting_contract": self.reporting_qualification_schema_version,
            },
            *(
                (native_acs_runtime_component,)
                if native_acs_runtime_component is not None
                else ()
            ),
            {
                "component_id": "source_dependency_closure",
                **source_projection,
            },
        )
        obligations = {
            "cold_start",
            "first_generation",
            "first_tick",
            "intra_generation",
            "generation_boundary",
            "first_post_boundary_tick",
            "second_post_boundary_tick",
            "ordinary_birth",
            "checkpoint_export",
            "disk_checkpoint_reload",
            "fresh_owner_restore",
            "first_post_restore_tick",
            "terminal_settlement",
        }
        if self.route_requires_policy_cache_refill:
            obligations.update(
                {
                    "first_post_restore_refill",
                    *(
                        {"post_boundary_policy_cache_refill"}
                        if not tick_native
                        else set()
                    ),
                }
            )
        if self.route_supports_turnover:
            obligations.add("turnover_refill")
        multicell_route = "jungle" not in self.artifact_abi
        if multicell_route:
            obligations.update(
                {
                    "multicell_execution",
                    "odd_physical_tail",
                    "alternate_batch_partition",
                }
            )
        if self.reporting_parity_supported:
            obligations.add("reporting_parity")
        if acs_checkpoint_slots:
            obligations.update(
                {
                    "async_checkpoint_backpressure",
                    "async_checkpoint_failure_latch",
                    "durable_frontier_publication",
                }
            )
        if acs_capture_instrumentation:
            obligations.add("acs_capture_barrier_measurement")
        contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=components,
            required_obligations=tuple(sorted(obligations)),
        )
        if (
            request is not None
            and request.execution_route_sha256 is not None
            and request.execution_route_sha256 != contract.execution_route_sha256
        ):
            raise CampaignProviderError(
                "resident request execution-route binding mismatch"
            )
        return contract

    def execute_route_qualification(
        self,
        *,
        campaign_plan: Mapping[str, Any],
        canary: Mapping[str, Any],
        run_root: Path,
    ) -> dict[str, Any]:
        """Run the explicit bounded canary and atomically publish route authority."""

        from .route_qualification import (
            bounded_route_qualification_preflight_v1,
            execute_route_qualification_v1,
        )

        _, qualification_receipt_path, _ = self._artifact_paths(campaign_plan)
        registry = ExecutionProviderRegistryV1()
        registry.register(self)
        return execute_route_qualification_v1(
            campaign_plan=campaign_plan,
            canary=canary,
            run_root=run_root,
            route_contract=self.execution_route_contract(campaign_plan),
            qualification_receipt_path=qualification_receipt_path,
            preflight_runner=lambda plan: bounded_route_qualification_preflight_v1(
                self, plan
            ),
            provider_registry=registry,
        )

    def promote_long_horizon_qualification(
        self,
        *,
        campaign_plan: Mapping[str, Any],
        long_horizon_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach exact long-horizon evidence to an existing bounded receipt."""

        from .route_qualification import (
            promote_execution_route_long_horizon_evidence_v1,
        )

        _, qualification_receipt_path, _ = self._artifact_paths(campaign_plan)
        return promote_execution_route_long_horizon_evidence_v1(
            campaign_plan=campaign_plan,
            route_contract=self.execution_route_contract(campaign_plan),
            qualification_receipt_path=qualification_receipt_path,
            long_horizon_evidence=long_horizon_evidence,
        )

    def promote_science_route_qualification(
        self,
        *,
        campaign_plan: Mapping[str, Any],
        long_horizon_evidence: Mapping[str, Any],
        canonical_full_production_route_attestation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach Gate27 science authority without changing long evidence."""

        from .route_qualification import (
            promote_execution_route_science_evidence_v1,
        )

        _, qualification_receipt_path, _ = self._artifact_paths(campaign_plan)
        return promote_execution_route_science_evidence_v1(
            campaign_plan=campaign_plan,
            route_contract=self.execution_route_contract(campaign_plan),
            qualification_receipt_path=qualification_receipt_path,
            long_horizon_evidence=long_horizon_evidence,
            canonical_full_production_route_attestation=(
                canonical_full_production_route_attestation
            ),
        )

    def estimate_resources(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        requests: Sequence[ResidentBatchRequestV1],
    ) -> dict[str, Any]:
        """Own the fixed-Patch estimator without central profile dispatch."""

        from .resources import (
            PATCH_MAX_LOGICAL_GENERATIONS,
            ResourceEstimateError,
            estimate_patch_fixed_resident_resources_v1,
            resident_request_resource_dimensions_v1,
        )

        dimensions = resident_request_resource_dimensions_v1(
            _plan_payload(plan), requests, include_parameter_dimensions=True
        )
        minimum = dimensions.pop("minimum_resident_segment_generations")
        if minimum < 3 or dimensions["resident_segment_generations"] > (
            PATCH_MAX_LOGICAL_GENERATIONS
        ):
            raise ResourceEstimateError(
                "every resident checkpoint segment must be in the admitted G3..G128 range"
            )
        return estimate_patch_fixed_resident_resources_v1(**dimensions)

    def load_qualified_artifact(
        self,
        artifact_path: str,
        qualification_receipt_path: str,
        reporting_receipt_path: str,
        *,
        loader: Callable[..., Any] | None = None,
        expected_artifact_abi: str | None = None,
        expected_binding_id: str | None = None,
        required_exports: Sequence[str] | None = None,
    ) -> Any:
        """Load one engine-qualified artifact or an explicitly bound adapter."""

        if loader is None:
            receipt_path = Path(qualification_receipt_path).resolve()
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CampaignProviderError(
                    f"production-engine qualification receipt is unreadable: {exc}"
                ) from exc
            if (
                isinstance(receipt, Mapping)
                and receipt.get("schema_version") == self.qualification_schema_version
            ):
                return self._load_declarative_qualified_artifact(
                    artifact_path,
                    receipt,
                    reporting_receipt_path,
                    expected_artifact_abi=expected_artifact_abi,
                    expected_binding_id=expected_binding_id,
                    required_exports=required_exports,
                )
            raise CampaignProviderError(
                "production-engine qualification receipt schema is invalid"
            )
        return loader(
            artifact_path,
            qualification_receipt_path,
            reporting_receipt_path,
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return _file_sha256_v1(path)

    def _load_declarative_qualified_artifact(
        self,
        artifact_path: str,
        receipt: Mapping[str, Any],
        reporting_receipt_path: str,
        *,
        expected_artifact_abi: str | None = None,
        expected_binding_id: str | None = None,
        required_exports: Sequence[str] | None = None,
    ) -> Any:
        artifact = Path(artifact_path).resolve()
        artifact_row = receipt.get("artifact")
        if (
            receipt.get("ok") is not True
            or receipt.get("provider_id") != self.provider_id
            or receipt.get("profile_id") != self.profile_id
            or not isinstance(artifact_row, Mapping)
            or (
                expected_artifact_abi is not None
                and receipt.get("artifact_abi") != expected_artifact_abi
            )
            or (
                expected_binding_id is not None
                and receipt.get("mechanics_binding_id") != expected_binding_id
            )
        ):
            raise CampaignProviderError(
                "production-engine qualification receipt identity is invalid"
            )
        expected_hash = artifact_row.get("sha256")
        expected_artifact = _verified_runtime_artifact_path_v1(
            Path(str(artifact_row.get("path", ""))), str(expected_hash)
        )
        if (
            not artifact.is_file()
            or expected_artifact is None
            or artifact != expected_artifact
            or not isinstance(expected_hash, str)
            or self._file_sha256(artifact) != expected_hash
        ):
            raise CampaignProviderError(
                "production-engine qualified artifact binding is invalid"
            )
        manifest = _verified_runtime_artifact_path_v1(
            Path(str(artifact_row.get("manifest_path", ""))),
            str(artifact_row.get("manifest_sha256")),
        )
        if manifest is None:
            raise CampaignProviderError(
                "production-engine qualified manifest binding is invalid"
            )
        try:
            reporting = json.loads(
                Path(reporting_receipt_path).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CampaignProviderError(
                f"production-engine reporting qualification is unreadable: {exc}"
            ) from exc
        if (
            not isinstance(reporting, Mapping)
            or reporting.get("schema_version")
            != self.reporting_qualification_schema_version
            or reporting.get("ok") is not True
            or reporting.get("provider_id") != self.provider_id
            or reporting.get("profile_id") != self.profile_id
            or reporting.get("artifact_sha256") != expected_hash
            or reporting.get("reporting_is_observational") is not True
            or (
                expected_binding_id is not None
                and reporting.get("mechanics_binding_id") != expected_binding_id
            )
        ):
            raise CampaignProviderError(
                "production-engine reporting qualification identity is invalid"
            )
        module_name = artifact_row.get("module_name")
        if not isinstance(module_name, str) or not module_name.isidentifier():
            raise CampaignProviderError(
                "production-engine qualified module name is invalid"
            )
        try:
            import torch  # noqa: F401 - loads the extension's shared-library owners.
        except ImportError as exc:
            raise CampaignProviderError(
                "PyTorch is required to load the qualified production artifact"
            ) from exc
        spec = importlib.util.spec_from_file_location(module_name, artifact)
        if spec is None or spec.loader is None:
            raise CampaignProviderError(
                "production-engine qualified artifact import failed"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        required = tuple(
            required_exports
            or ("eiel_cuda_architecture_b_execute_direct_resident_windows_v1",)
        )
        missing = tuple(name for name in required if not hasattr(module, name))
        if missing:
            raise CampaignProviderError(
                "production-engine qualified artifact is missing exports: "
                + ", ".join(missing)
            )
        candidate_authority_sha256 = _production_candidate_authority_sha256_v1(
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            qualification_receipt=receipt,
            reporting_receipt=reporting,
            artifact_row=artifact_row,
        )
        return SimpleNamespace(
            module=module,
            candidate_authority_sha256=candidate_authority_sha256,
            production_authority=True,
            qualification_receipt=dict(receipt),
            reporting_receipt=dict(reporting),
        )

    def execute_runner(
        self,
        runner: Any,
        *,
        start_generation: int,
        end_generation: int,
        skip_initialization: bool,
        executor: Callable[..., Any] | None = None,
    ) -> Any:
        """Delegate one accepted request to the public native runner owner."""

        if executor is None:
            from engine.native_runtime.cuda_architecture_b_runner_owner_v1 import (
                execute_cuda_architecture_b_runner_run_v1,
            )

            executor = execute_cuda_architecture_b_runner_run_v1
        return executor(
            runner,
            start_gen=start_generation,
            end_gen=end_generation,
            skip_init=skip_initialization,
        )

    def request_cancellation(
        self,
        runner: Any,
        *,
        request_id: str,
        reason: str = "operator_requested",
        timeout_seconds: float = 5.0,
        requester: Callable[..., Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Delegate cancellation without taking tick/generation ownership."""

        if requester is None:
            from engine.native_runtime.cuda_architecture_b_runner_owner_v1 import (
                request_cuda_architecture_b_runner_cancellation_v1,
            )

            requester = request_cuda_architecture_b_runner_cancellation_v1
        return dict(
            requester(
                runner,
                request_id=request_id,
                reason=reason,
                timeout_seconds=timeout_seconds,
            )
        )

    @staticmethod
    def _artifact_paths(plan: Mapping[str, Any]) -> tuple[str, str, str]:
        artifacts = plan.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise CampaignProviderError("compiled plan is missing artifact bindings")
        paths: list[str] = []
        for name in ("cuda_extension", "qualification_receipt", "reporting_receipt"):
            row = artifacts.get(name)
            path = row.get("path") if isinstance(row, Mapping) else None
            if not isinstance(path, str) or not path:
                raise CampaignProviderError(f"compiled artifact path is absent: {name}")
            paths.append(path)
        return paths[0], paths[1], paths[2]

    @staticmethod
    def _genome_v2_enabled(plan: Mapping[str, Any]) -> bool:
        cells = plan.get("cells")
        if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)):
            for cell in cells:
                parameters = (
                    cell.get("parameters") if isinstance(cell, Mapping) else None
                )
                if (
                    isinstance(parameters, Mapping)
                    and parameters.get("genome_upgrade_v2.enable") is True
                ):
                    return True
        return False

    @staticmethod
    def _genome_v2_checkpoint_authority_binding(
        plan: Mapping[str, Any],
        *,
        execution_route_sha256: str,
    ) -> tuple[tuple[str, str], ...]:
        if not ProductionEngineLifecycleV1._genome_v2_enabled(plan):
            return ()
        artifacts = plan.get("artifacts")
        package_row = (
            artifacts.get("genome_upgrade_v2_package_receipt")
            if isinstance(artifacts, Mapping)
            else None
        )
        if not isinstance(package_row, Mapping):
            return ()
        package_path = package_row.get("path")
        package_receipt_sha256 = package_row.get("sha256")
        if (
            not isinstance(package_path, str)
            or not package_path
            or not _valid_sha256(package_receipt_sha256)
            or not _valid_sha256(execution_route_sha256)
        ):
            raise CampaignProviderError(
                "Genome V2 package authority binding is incomplete"
            )
        path = Path(package_path).expanduser().resolve()
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise CampaignProviderError(
                "Genome V2 package receipt is unreadable"
            ) from exc
        if hashlib.sha256(raw_bytes).hexdigest() != package_receipt_sha256:
            raise CampaignProviderError("Genome V2 package receipt digest mismatch")
        try:
            receipt = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignProviderError(
                "Genome V2 package receipt is malformed"
            ) from exc
        if not isinstance(receipt, Mapping):
            raise CampaignProviderError("Genome V2 package receipt must be an object")
        identity = receipt.get("identity")
        package_id_sha256 = receipt.get("package_id_sha256")
        if not isinstance(identity, Mapping) or not _valid_sha256(package_id_sha256):
            raise CampaignProviderError(
                "Genome V2 package receipt lacks package identity"
            )
        from evolution.genome_upgrade_v2_package_v1 import (
            checkpoint_authority_binding_v1,
            validate_checkpoint_authority_binding_v1,
        )

        binding = validate_checkpoint_authority_binding_v1(
            checkpoint_authority_binding_v1(
                identity,
                package_id_sha256=package_id_sha256,
                package_receipt_sha256=package_receipt_sha256,
                execution_route_sha256=execution_route_sha256,
            )
        )
        return tuple(sorted(binding.items()))

    def _checkpoint_binding(
        self, plan: Mapping[str, Any], request: ResidentBatchRequestV1
    ) -> dict[str, Any]:
        required = {
            "campaign_id": plan.get("campaign_id"),
            "plan_sha256": plan.get("plan_sha256"),
            "qualification_key_sha256": plan.get("qualification_key_sha256"),
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise CampaignProviderError(
                "compiled plan checkpoint identity is incomplete"
            )
        artifacts = plan.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise CampaignProviderError("compiled plan checkpoint artifacts are absent")
        artifact_sha256s: dict[str, str] = {}
        for name, row in sorted(artifacts.items()):
            digest = row.get("sha256") if isinstance(row, Mapping) else None
            if (
                not isinstance(name, str)
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise CampaignProviderError(
                    f"compiled checkpoint artifact digest is invalid: {name}"
                )
            artifact_sha256s[name] = digest
        binding = {
            **required,
            "profile_id": self.profile_id,
            "provider_id": self.provider_id,
            "execution_route_sha256": request.execution_route_sha256,
            "batch_id": request.batch_id,
            "cell_ids": list(request.cell_ids),
            "artifact_sha256s": artifact_sha256s,
        }
        from engine.campaigns.run_contract_v1 import checkpoint_contract_binding_v1

        run_contract_binding = checkpoint_contract_binding_v1(plan)
        if run_contract_binding is not None:
            binding["run_contract"] = run_contract_binding
        return binding

    def prepare_fresh_owner_restore_checkpoint(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        source_request: ResidentBatchRequestV1,
        restore_request: ResidentBatchRequestV1,
        checkpoint: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Rebind one verified checkpoint to a distinct fresh-owner batch ID.

        The source binding remains embedded and hashed.  Only the orchestration
        identity may change; every scientific and route-bearing request field
        must remain exact.
        """

        payload = _plan_payload(plan)
        if checkpoint.get("schema_version") != self.checkpoint_schema_version:
            raise CampaignProviderError(
                "fresh-owner source checkpoint schema is unsupported"
            )
        source_binding = self._checkpoint_binding(payload, source_request)
        if checkpoint.get("checkpoint_binding") != source_binding:
            raise CampaignProviderError(
                "fresh-owner source checkpoint binding mismatch"
            )
        source_payload = source_request.payload()
        restore_payload = restore_request.payload()
        for field in (
            "profile_id",
            "cell_ids",
            "generations",
            "ticks_per_generation",
            "checkpoint_milestones",
            "parameter_arrays",
            "seed_arrays",
            "agent_id_allocation_generation_count",
            "checkpoint_storage_mode",
            "reporting_enabled",
            "execution_route_sha256",
            "genome_v2_checkpoint_authority_binding",
            "total_ticks",
            "start_next_tick",
            "checkpoint_ticks",
            "resident_owner_span_ticks",
            "native_dispatch_window_ticks",
            "report_every_ticks",
            "science_sample_interval_ticks",
            "reducer_every_ticks",
            "archive_every_ticks",
            "event_window_capacity",
            "source_window_ticks",
            "tick_native_control",
        ):
            # Legacy requests predate the absolute-tick fields and omit them
            # entirely. Keep the comparison strict for native requests, but
            # do not turn an absent-on-both-sides legacy default into a
            # KeyError during fresh-owner rebinding.
            if (
                field in source_payload or field in restore_payload
            ) and source_payload.get(field) != restore_payload.get(field):
                raise CampaignProviderError(
                    f"fresh-owner request changes checkpoint semantics: {field}"
                )
        unsigned = dict(checkpoint)
        unsigned.pop("fresh_owner_checkpoint_rebind", None)
        source_checkpoint_sha256 = hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        rebound = dict(checkpoint)
        rebound["checkpoint_binding"] = self._checkpoint_binding(
            payload, restore_request
        )
        rebound["fresh_owner_checkpoint_rebind"] = {
            "schema_version": FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION,
            "source_batch_id": source_request.batch_id,
            "restored_batch_id": restore_request.batch_id,
            "source_checkpoint_binding": source_binding,
            "source_checkpoint_sha256": source_checkpoint_sha256,
        }
        return rebound

    def prepare_wrc_checkpoint_adaptive_rebinding_v1(
        self,
        source_plan: CampaignPlanV1 | Mapping[str, Any],
        restore_plan: CampaignPlanV1 | Mapping[str, Any],
        source_request: ResidentBatchRequestV1,
        restore_request: ResidentBatchRequestV1,
        checkpoint: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind approved WRC experiment-variable arrays to a restored owner."""

        source_payload = _plan_payload(source_plan)
        restore_payload = _plan_payload(restore_plan)
        if source_request.parameter_arrays == restore_request.parameter_arrays:
            if source_payload != restore_payload:
                raise CampaignProviderError(
                    "unchanged adaptive restore requires the exact legacy plan"
                )
            return self.prepare_fresh_owner_restore_checkpoint(
                source_payload,
                source_request,
                restore_request,
                checkpoint,
            )
        if checkpoint.get("schema_version") != self.checkpoint_schema_version:
            raise CampaignProviderError(
                "adaptive source checkpoint schema is unsupported"
            )
        source_binding = self._checkpoint_binding(source_payload, source_request)
        if checkpoint.get("checkpoint_binding") != source_binding:
            raise CampaignProviderError("adaptive source checkpoint binding mismatch")
        from engine.campaigns.registry import default_profile_registry
        from engine.campaigns.wrc_adaptive_rebinding import (
            WRCCheckpointAdaptiveRebindingError,
            validate_wrc_checkpoint_adaptive_rebinding_v1,
        )

        profile = next(
            (
                candidate
                for candidate in default_profile_registry().profiles()
                if candidate.profile_id == self.profile_id
            ),
            None,
        )
        if profile is None:
            raise CampaignProviderError("adaptive WRC profile contract is unavailable")
        try:
            adaptive_receipt = validate_wrc_checkpoint_adaptive_rebinding_v1(
                source_plan=source_payload,
                restore_plan=restore_payload,
                source_request=source_request,
                restore_request=restore_request,
                checkpoint_generation=checkpoint.get("generation"),
                profile=profile,
                generation_min=self.resident_segment_generation_min,
                generation_max=self.resident_segment_generation_max,
            )
        except WRCCheckpointAdaptiveRebindingError as exc:
            raise CampaignProviderError(str(exc)) from exc
        unsigned = dict(checkpoint)
        unsigned.pop("fresh_owner_checkpoint_rebind", None)
        unsigned.pop(WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1, None)
        source_checkpoint_sha256 = hashlib.sha256(
            json.dumps(
                unsigned,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        restore_binding = self._checkpoint_binding(restore_payload, restore_request)
        adaptive_unsigned = {
            **adaptive_receipt,
            "source_checkpoint_binding": source_binding,
            "restore_checkpoint_binding": restore_binding,
            "source_checkpoint_sha256": source_checkpoint_sha256,
        }
        adaptive_unsigned.pop("receipt_sha256")
        from engine.campaigns.canonical import canonical_sha256

        adaptive_bound = {
            **adaptive_unsigned,
            "receipt_sha256": canonical_sha256(adaptive_unsigned),
        }
        rebound = dict(checkpoint)
        rebound["checkpoint_binding"] = restore_binding
        rebound["fresh_owner_checkpoint_rebind"] = {
            "schema_version": FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION,
            "source_batch_id": source_request.batch_id,
            "restored_batch_id": restore_request.batch_id,
            "source_checkpoint_binding": source_binding,
            "source_checkpoint_sha256": source_checkpoint_sha256,
        }
        rebound[WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1] = adaptive_bound
        return rebound

    def _fresh_owner_reload_receipts(
        self,
        *,
        request: ResidentBatchRequestV1,
        checkpoint: Mapping[str, Any],
        bundle_rows: Mapping[str, Mapping[str, Any]],
        projections: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        checkpoint_digests = checkpoint.get("terminal_cell_digests")
        if (
            not isinstance(checkpoint_digests, Mapping)
            or len(checkpoint_digests) != len(request.cell_ids)
            or set(checkpoint_digests) != set(request.cell_ids)
            or len(projections) != len(request.cell_ids)
        ):
            raise CampaignProviderError(
                "fresh-owner checkpoint digest authority is incomplete"
            )
        receipts: list[dict[str, Any]] = []
        for cell_id, projection in zip(request.cell_ids, projections, strict=True):
            row = bundle_rows[cell_id]
            declared = row.get("source_projection_sha256")
            checkpoint_digest = checkpoint_digests.get(cell_id)
            loaded = self.terminal_projection_digest(projection)
            physical_world_state = (
                checkpoint.get("checkpoint_digest_domain")
                == "physical_world_state_v1"
                and row.get("physical_format")
                == "eiel_acs_world_checkpoint_object_v1"
            )
            if (
                not isinstance(declared, str)
                or not isinstance(checkpoint_digest, str)
                or declared != checkpoint_digest
                or (not physical_world_state and loaded != declared)
            ):
                raise CampaignProviderError(
                    f"fresh-owner loaded projection digest differs: {cell_id}"
                )
            receipts.append(
                {
                    "cell_id": cell_id,
                    "bundle_path": str(row["path"]),
                    "source_projection_sha256": declared,
                    "loaded_projection_sha256": loaded,
                    "source_checkpoint_digest_domain": checkpoint.get(
                        "checkpoint_digest_domain", "canonical_projection_v1"
                    ),
                }
            )
        return receipts

    def _fresh_owner_restore_receipt(
        self,
        *,
        request: ResidentBatchRequestV1,
        checkpoint: Mapping[str, Any],
        reload_receipts: Sequence[Mapping[str, Any]],
        terminal_cell_digests: Any,
    ) -> dict[str, Any]:
        source_binding = checkpoint.get("checkpoint_binding")
        rebind = checkpoint.get("fresh_owner_checkpoint_rebind")
        adaptive_rebind = checkpoint.get(WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1)
        if isinstance(rebind, Mapping):
            source_batch_id = rebind.get("source_batch_id")
            restored_batch_id = rebind.get("restored_batch_id")
            if (
                rebind.get("schema_version")
                != FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION
                or restored_batch_id != request.batch_id
                or not isinstance(source_batch_id, str)
                or not source_batch_id
                or not isinstance(source_binding, Mapping)
                or source_binding.get("batch_id") != request.batch_id
            ):
                raise CampaignProviderError(
                    "fresh-owner checkpoint rebind authority is invalid"
                )
        else:
            source_batch_id = (
                source_binding.get("batch_id")
                if isinstance(source_binding, Mapping)
                else None
            )
            restored_batch_id = request.batch_id
        if adaptive_rebind is not None:
            if not isinstance(adaptive_rebind, Mapping):
                raise CampaignProviderError(
                    "adaptive checkpoint rebinding receipt is malformed"
                )
            from engine.campaigns.wrc_adaptive_rebinding import (
                WRCCheckpointAdaptiveRebindingError,
                validate_wrc_checkpoint_adaptive_rebinding_receipt_v1,
            )

            try:
                validated_adaptive = (
                    validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
                        adaptive_rebind
                    )
                )
            except WRCCheckpointAdaptiveRebindingError as exc:
                raise CampaignProviderError(str(exc)) from exc
            reconstructed_source = dict(checkpoint)
            reconstructed_source["checkpoint_binding"] = validated_adaptive.get(
                "source_checkpoint_binding"
            )
            reconstructed_source.pop("fresh_owner_checkpoint_rebind", None)
            reconstructed_source.pop(WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1, None)
            reconstructed_source_sha256 = hashlib.sha256(
                json.dumps(
                    reconstructed_source,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if (
                validated_adaptive.get("restore_checkpoint_binding") != source_binding
                or validated_adaptive.get("execution_route_sha256")
                != request.execution_route_sha256
                or validated_adaptive.get("cell_ids") != list(request.cell_ids)
                or validated_adaptive.get("restore_generation") != request.generations
                or validated_adaptive.get("source_generation")
                != checkpoint.get("generation")
                or validated_adaptive.get("source_checkpoint_binding")
                != rebind.get("source_checkpoint_binding")
                or validated_adaptive.get("source_checkpoint_sha256")
                != rebind.get("source_checkpoint_sha256")
                or reconstructed_source_sha256
                != validated_adaptive.get("source_checkpoint_sha256")
            ):
                raise CampaignProviderError(
                    "adaptive checkpoint rebinding authority differs"
                )
        else:
            validated_adaptive = None
        checkpoint_digests = checkpoint.get("terminal_cell_digests")
        checkpoint_identity = checkpoint.get("checkpoint_identity_contract")
        source_terminal_projection_digests = (
            checkpoint_identity.get("terminal_projection_digest_sha256s")
            if isinstance(checkpoint_identity, Mapping)
            else checkpoint_digests
        )
        if (
            not isinstance(source_batch_id, str)
            or not source_batch_id
            or not isinstance(checkpoint_digests, Mapping)
            or not isinstance(source_terminal_projection_digests, Mapping)
            or not isinstance(terminal_cell_digests, Mapping)
            or len(checkpoint_digests) != len(request.cell_ids)
            or set(checkpoint_digests) != set(request.cell_ids)
            or set(source_terminal_projection_digests) != set(request.cell_ids)
            or len(terminal_cell_digests) != len(request.cell_ids)
            or set(terminal_cell_digests) != set(request.cell_ids)
        ):
            raise CampaignProviderError(
                "fresh-owner restore receipt authority is incomplete"
            )
        receipt = {
            "schema_version": FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION,
            "projection_bundles_reloaded": True,
            "source_batch_id": source_batch_id,
            "restored_batch_id": restored_batch_id,
            "cell_ids": list(request.cell_ids),
            "source_checkpoint_cell_digests": dict(checkpoint_digests),
            "parent_checkpoint_projection_digest_sha256s": dict(
                source_terminal_projection_digests
            ),
            "restored_terminal_cell_digests": dict(terminal_cell_digests),
            "restored_terminal_projection_digest_sha256s": dict(terminal_cell_digests),
            "reload_receipts": [dict(row) for row in reload_receipts],
            "checkpoint_rebind": (
                dict(rebind) if isinstance(rebind, Mapping) else None
            ),
        }
        if request.total_ticks is not None:
            receipt.update(
                {
                    "time_authority": "absolute_tick_v1",
                    "source_checkpoint_absolute_tick": checkpoint.get(
                        "absolute_tick"
                    ),
                    "source_checkpoint_next_absolute_tick": checkpoint.get(
                        "next_absolute_tick"
                    ),
                }
            )
        else:
            receipt["source_checkpoint_generation"] = checkpoint.get(
                "generation"
            )
        if validated_adaptive is not None:
            receipt["wrc_checkpoint_adaptive_rebinding"] = validated_adaptive
        return receipt

    @staticmethod
    def _batch_cells(
        plan: Mapping[str, Any], request: ResidentBatchRequestV1
    ) -> tuple[Mapping[str, Any], ...]:
        raw_cells = plan.get("cells")
        if not isinstance(raw_cells, list):
            raise CampaignProviderError("compiled plan is missing logical cells")
        by_id = {
            row.get("cell_id"): row
            for row in raw_cells
            if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
        }
        if any(
            not cell_id or Path(cell_id).name != cell_id or cell_id in {".", ".."}
            for cell_id in request.cell_ids
        ):
            raise CampaignProviderError("compiled batch contains an unsafe cell id")
        try:
            return tuple(by_id[cell_id] for cell_id in request.cell_ids)
        except KeyError as exc:
            raise CampaignProviderError(
                "compiled batch references an unknown cell"
            ) from exc

    def _prepare_generation_zero_sources_v2(
        self,
        plan: Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
        *,
        runner_builder: Callable[..., Any],
        cells: Sequence[Mapping[str, Any]],
        artifact: str,
        qualification: str,
        reporting: str,
        segment_generations: int | None = None,
        source_capability_mask: Mapping[str, bool] | None = None,
        source_projection_binder: (
            Callable[[Mapping[str, Any], Any], Mapping[str, Any]] | None
        ) = None,
        source_projection_binding_scope: str = "exact_inputs",
    ) -> tuple[Mapping[str, Any], ...]:
        """Materialize exact cached generation-zero projections for real Patch runs."""

        from engine.campaigns.canonical import canonical_sha256
        from engine.campaigns.orchestration import atomic_write_json
        from engine.campaigns.source_factory import (
            prepare_campaign_sources,
            source_identity_parameters_v1,
        )
        from engine.native_runtime.interchange_projection import (
            validate_native_checkpoint_projection,
        )
        from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
            QUALIFIED_CAPABILITY_MASK_V1,
            _bind_life_history_metadata_v1,
            _life_history_config_from_sim_cfg_v1,
        )
        from engine.perf.cxxbackbone10_resident_state_owner import (
            build_import_projection,
        )
        import torch

        run_root = getattr(context, "run_root", None)
        batch_root = getattr(context, "batch_root", None)
        if not isinstance(run_root, Path) or not isinstance(batch_root, Path):
            raise CampaignProviderError(
                "source factory requires durable run and batch roots"
            )
        capability_mask = {
            str(name): bool(value)
            for name, value in dict(
                source_capability_mask or QUALIFIED_CAPABILITY_MASK_V1
            ).items()
        }
        if request.genome_v2_checkpoint_authority_binding:
            capability_mask["genome_v2"] = True
        if source_projection_binding_scope not in {"exact_inputs", "per_cell"}:
            raise CampaignProviderError(
                "source projection binding scope is unsupported"
            )
        if (
            source_projection_binding_scope == "per_cell"
            and source_projection_binder is None
        ):
            raise CampaignProviderError(
                "per-cell source projection binding requires a binder"
            )
        by_identity: dict[str, Mapping[str, Any]] = {}
        for cell in cells:
            source_inputs = {
                "seeds": dict(sorted(dict(cell["seeds"]).items())),
                "parameters": dict(
                    sorted(source_identity_parameters_v1(cell["parameters"]).items())
                ),
            }
            if source_projection_binding_scope == "per_cell":
                source_inputs["logical_cell_id"] = str(cell["cell_id"])
            identity = canonical_sha256(source_inputs)
            prior = by_identity.setdefault(identity, cell)
            if dict(prior["seeds"]) != dict(cell["seeds"]) or dict(
                prior["parameters"]
            ) != dict(cell["parameters"]):
                raise CampaignProviderError("generation-zero source identity collision")

        source_builder_path = Path(
            sys.modules[
                "engine.perf.cuda_architecture_b_direct_resident_source_builder_v1"
            ].__file__
        ).resolve()
        runner_factory_path = Path(
            sys.modules["engine.campaigns.patch_runner_factory"].__file__
        ).resolve()
        provider_path = Path(__file__).resolve()
        source_contract = {
            "schema_version": "eiel_patch_generation_zero_source_contract_v3",
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "provider_source_factory_python_sha256": self._file_sha256(provider_path),
            "generation_zero_life_history_metadata_binding": True,
            "source_builder_python_sha256": self._file_sha256(source_builder_path),
            "runner_factory_python_sha256": self._file_sha256(runner_factory_path),
            "source_capability_mask": dict(sorted(capability_mask.items())),
            "source_projection_binder": (
                None
                if source_projection_binder is None
                else {
                    "module": source_projection_binder.__module__,
                    "qualname": source_projection_binder.__qualname__,
                }
            ),
            "source_projection_binding_scope": source_projection_binding_scope,
        }
        if request.total_ticks is not None:
            (
                source_capacity_window_count,
                native_source_window_ticks,
            ) = _tick_native_runner_source_dimensions_v1(
                request.tick_native_control
            )
            runner_time_kwargs = {
                "tick_native_control": dict(request.tick_native_control),
            }
            source_contract.update(
                {
                    "time_authority": "absolute_tick_v1",
                    "source_capacity_window_count": source_capacity_window_count,
                    "source_window_ticks": native_source_window_ticks,
                }
            )
        else:
            if type(segment_generations) is not int or segment_generations <= 0:
                raise CampaignProviderError(
                    "legacy source preparation requires a positive segment extent"
                )
            runner_time_kwargs = {
                "generations": segment_generations,
                "ticks_per_generation": request.ticks_per_generation,
            }

        def generate(source_request: Any, destination: Path) -> None:
            cell = by_identity.get(source_request.source_identity_sha256)
            if cell is None:
                raise CampaignProviderError(
                    "source factory requested an unknown identity"
                )
            built = None
            projection: Mapping[str, Any] | None = None
            try:
                built = runner_builder(
                    cell,
                    destination.parent / "runner",
                    **runner_time_kwargs,
                    artifact_path=artifact,
                    qualification_receipt_path=qualification,
                    reporting_receipt_path=reporting,
                    **(
                        {
                            "genome_v2_checkpoint_authority_binding": dict(
                                request.genome_v2_checkpoint_authority_binding
                            )
                        }
                        if request.genome_v2_checkpoint_authority_binding
                        else {}
                    ),
                )
                projection = _generation_zero_projection_payload_v1(
                    build_import_projection(
                        built.runner,
                        capability_mask,
                        generation=0,
                        tick=0,
                    )
                )
                projection = _bind_life_history_metadata_v1(
                    dict(projection),
                    life_history=_life_history_config_from_sim_cfg_v1(
                        built.runner.sim_cfg
                    ),
                )
                validate_native_checkpoint_projection(projection)
                if source_projection_binder is not None:
                    bound = source_projection_binder(projection, built.runner)
                    if not isinstance(bound, Mapping):
                        raise CampaignProviderError(
                            "source projection binder returned a non-mapping"
                        )
                    projection = dict(bound)
                torch.save(projection, destination)
            finally:
                built = None
                projection = None
                try:
                    import gc

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        source_plan = dict(plan)
        source_plan["cells"] = [dict(cell) for cell in cells]
        receipt = prepare_campaign_sources(
            source_plan,
            cache_root=(run_root.parent / "_source_cache_v2").resolve(),
            materialization_root=(batch_root / "source_projections_v2").resolve(),
            source_contract=source_contract,
            generator=generate,
            source_identity_scope=source_projection_binding_scope,
        )
        atomic_write_json(batch_root / "source_factory_receipt.json", receipt)
        rows = receipt.get("rows")
        if not isinstance(rows, list):
            raise CampaignProviderError("source factory receipt lacks rows")
        paths = {
            row.get("cell_id"): row.get("projection_path")
            for row in rows
            if isinstance(row, Mapping)
        }
        projections: list[Mapping[str, Any]] = []
        for cell_id in request.cell_ids:
            path = paths.get(cell_id)
            if not isinstance(path, str):
                raise CampaignProviderError(f"source factory omitted cell {cell_id}")
            try:
                # The source cache is the immutable owner of generation-zero
                # state.  Keep its storages file-backed while the direct
                # resident owner is assembled; eager deserialization turns a
                # B136 source set into a second anonymous-RAM simulation.
                value = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
            except Exception as exc:
                raise CampaignProviderError(
                    f"cached generation-zero projection is unreadable: {cell_id}: {exc}"
                ) from exc
            if not isinstance(value, Mapping):
                raise CampaignProviderError(
                    "cached generation-zero projection is invalid"
                )
            projection = dict(value)
            validate_native_checkpoint_projection(projection)
            metadata = projection.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("generation") != 0:
                raise CampaignProviderError(
                    "cached generation-zero projection has a nonzero generation"
                )
            projections.append(projection)
        return tuple(projections)

    @classmethod
    def _segment_ends(
        cls, request: ResidentBatchRequestV1, *, start: int
    ) -> tuple[int, ...]:
        if type(start) is not int or start < 0 or start > request.generations:
            raise CampaignProviderError("checkpoint generation is outside the campaign")
        milestones = cls._rolling_segment_ends(
            generations=request.generations,
            declared_milestones=request.checkpoint_milestones,
        )
        if any(
            type(value) is not int or value <= 0 or value > request.generations
            for value in milestones
        ):
            raise CampaignProviderError(
                "checkpoint milestones are outside the campaign"
            )
        return tuple(value for value in milestones if value > start)

    @staticmethod
    def _absolute_tick_segments_v1(
        request: ResidentBatchRequestV1, *, start_next_tick: int
    ) -> tuple[dict[str, Any], ...]:
        """Decompose public tick targets by owner and dispatch boundaries only."""

        total = request.total_ticks
        source_window = request.source_window_ticks
        if (
            type(total) is not int
            or type(source_window) is not int
            or total < 1
            or source_window < 1
            or type(start_next_tick) is not int
            or not 0 <= start_next_tick <= total
        ):
            raise CampaignProviderError("absolute-tick execution cursor is invalid")
        native_tick_window = request.native_dispatch_window_ticks
        native_owner_span = request.resident_owner_span_ticks
        cadence_fields = {
            "report": request.report_every_ticks,
            "science_sample": request.science_sample_interval_ticks,
            "reducer": request.reducer_every_ticks,
            "archive": request.archive_every_ticks,
        }
        if any(
            value is not None and (type(value) is not int or value < 1)
            for value in cadence_fields.values()
        ):
            raise CampaignProviderError("absolute-tick schedule cadence is invalid")
        schedule_ticks: dict[str, set[int]] = {
            name: (set(range(value, total + 1, value)) if type(value) is int else set())
            for name, value in cadence_fields.items()
        }
        schedule_ticks["checkpoint"] = set(request.checkpoint_ticks)
        if native_owner_span is not None:
            schedule_ticks["owner"] = set(
                range(native_owner_span, total + 1, native_owner_span)
            )
        else:
            schedule_ticks["owner"] = set()
        schedule_ticks["terminal"] = {total}
        public_targets = tuple(sorted(set().union(*schedule_ticks.values())))
        if any(
            type(value) is not int or value <= 0 or value > total
            for value in public_targets
        ):
            raise CampaignProviderError("absolute schedule targets are invalid")
        segments: list[dict[str, Any]] = []
        cursor = start_next_tick
        if native_tick_window is not None:
            if (
                type(native_tick_window) is not int
                or native_tick_window < 1
                or type(native_owner_span) is not int
                or native_owner_span < native_tick_window
            ):
                raise CampaignProviderError(
                    "tick-native owner and dispatch spans are invalid"
                )
            for target in (value for value in public_targets if value > cursor):
                while cursor < target:
                    start = cursor
                    owner_span_start = (start // native_owner_span) * native_owner_span
                    owner_span_end = min(total, owner_span_start + native_owner_span)
                    cursor = min(
                        target,
                        start + native_tick_window,
                        owner_span_end,
                    )
                    count = cursor - start
                    segments.append(
                        {
                            "start_next_tick": start,
                            "end_next_tick": cursor,
                            "next_absolute_tick": cursor,
                            "owner_span_start_tick": owner_span_start,
                            "owner_span_end_tick": owner_span_end,
                            "dispatch_window_ticks": native_tick_window,
                            # The native owner still has bounded source-row
                            # coordinates. They are capacity indexes only and
                            # never split dispatch or own simulation time.
                            "start_generation": start // source_window,
                            "start_tick": start % source_window,
                            "terminal_generation": cursor // source_window,
                            "terminal_tick": cursor % source_window,
                            "logical_generation_count": 1,
                            "absolute_tick_span": count,
                            "publish_checkpoint": (
                                cursor in schedule_ticks["checkpoint"]
                                or cursor == total
                            ),
                            "fired_schedules": tuple(
                                name
                                for name in (
                                    "report",
                                    "science_sample",
                                    "reducer",
                                    "archive",
                                    "checkpoint",
                                    "owner",
                                    "terminal",
                                )
                                if cursor in schedule_ticks[name]
                            ),
                        }
                    )
            return tuple(segments)
        raise CampaignProviderError(
            "tick-native execution requires an explicit native dispatch window"
        )

    @staticmethod
    def _agent_id_allocation_generation_count(
        request: ResidentBatchRequestV1, *, start_generation: int | None
    ) -> int:
        if request.total_ticks is not None:
            control = request.tick_native_control
            if not isinstance(control, Mapping):
                raise CampaignProviderError(
                    "tick-native request lacks its canonical control contract"
                )
            dispatch = control.get("native_dispatch_window_ticks")
            source_window = control.get("source_window_ticks")
            if (
                type(dispatch) is not int
                or type(source_window) is not int
                or dispatch < 1
                or source_window < 1
            ):
                raise CampaignProviderError("tick-native allocation spans are invalid")
            # Identity storage is a reusable owner-local envelope.  It is
            # bounded by one admitted dispatch, never by total_ticks.
            return max(2, (dispatch + source_window - 1) // source_window)
        if type(start_generation) is not int:
            raise CampaignProviderError(
                "legacy resident allocation requires a generation cursor"
            )
        terminal_generation = request.agent_id_allocation_generation_count
        if terminal_generation is None:
            terminal_generation = request.generations
        if (
            type(terminal_generation) is not int
            or (
                request.total_ticks is None
                and terminal_generation < request.generations
            )
            or start_generation < 0
            or start_generation >= terminal_generation
        ):
            raise CampaignProviderError(
                "resident agent-ID allocation terminal generation is invalid"
            )
        remaining = terminal_generation - start_generation
        dispatch_cap = request.max_internal_lifecycle_segments_per_dispatch
        if dispatch_cap is None:
            dispatch_cap = 128
        elif type(dispatch_cap) is not int or dispatch_cap < 1 or dispatch_cap > 128:
            raise CampaignProviderError(
                "resident agent-ID allocation dispatch cap is invalid"
            )
        return max(2, min(remaining, dispatch_cap))

    @staticmethod
    def _mutation_arguments(prepared: Any) -> dict[str, Any]:
        projections = tuple(getattr(prepared, "source_projections", ()) or ())
        if projections:
            projection = projections[0]
        else:
            # Tick-native streamed owners deliberately retain no B-wide
            # source-projection tuple.  Their bounded builder keeps the first
            # canonical projection as compact owner metadata and exposes the
            # rest only through the lazy source cursor.
            physical_window_arguments = getattr(
                prepared, "physical_window_arguments", prepared
            )
            projection = getattr(
                physical_window_arguments, "initial_projection", None
            )
            if projection is None:
                wdl_arguments = getattr(physical_window_arguments, "wdl", None)
                projection = getattr(wdl_arguments, "initial_projection", None)
        try:
            if projection is None:
                raise AttributeError("initial projection is absent")
            metadata = (
                projection.get("metadata")
                if isinstance(projection, Mapping)
                else projection.metadata
            )
        except (AttributeError, IndexError, TypeError) as exc:
            raise CampaignProviderError(
                "prepared Patch owner has no source metadata"
            ) from exc
        if not isinstance(metadata, Mapping):
            raise CampaignProviderError("prepared Patch source metadata is invalid")
        values = {
            "policy_mutation_enable": metadata.get("policy_weight_mutation_enable"),
            "policy_mutation_mode": metadata.get("policy_weight_mutation_mode"),
            "policy_mutation_rate": metadata.get("policy_weight_mutation_rate"),
            "policy_mutation_std": metadata.get("policy_weight_mutation_std"),
        }
        if (
            type(values["policy_mutation_enable"]) is not bool
            or not isinstance(values["policy_mutation_mode"], str)
            or type(values["policy_mutation_rate"]) is not float
            or type(values["policy_mutation_std"]) is not float
        ):
            raise CampaignProviderError(
                "prepared Patch mutation metadata is incomplete"
            )
        return values

    @staticmethod
    def _observer_continuation(
        generation: int,
        *,
        tick: int = 0,
        event_cursor: int,
        snapshot_sequence: int,
        absolute_tick_authority: bool = False,
    ) -> dict[str, Any]:
        if type(absolute_tick_authority) is not bool:
            raise CampaignProviderError(
                "observer continuation absolute tick authority must be a boolean"
            )
        if generation == 0 and tick == 0:
            return {
                "schema_version": "eiel_architecture_b_observer_continuation_v1",
                "checkpoint_next_generation": 0,
                "tick": 0,
                "typed_lifecycle_event_cursor": event_cursor,
                "exact_reporting_boundary_ordinal": 0,
                "behavior_completed_generation_and_row_key": {
                    "completed_generation": -1,
                    "last_row_key": None,
                },
                "metrics_completed_generation": -1,
                "lineage_completed_generation_and_row_key": {
                    "completed_generation": -1,
                    "last_row_key": None,
                },
                "chronicle_observation_ordinal": 0,
                "progress_sequence": 0,
                "schedule_phase": {
                    "origin_generation": 0,
                    "next_generation": 0,
                },
                "deduplication": {"acknowledged_record_keys": []},
                "acknowledged_channel_cursor_ranges": {
                    "exact_reporting": [0, 0],
                    "behaviors": [0, 0],
                    "metrics": [0, 0],
                    "lineage": [0, 0],
                    "typed_lifecycle": [0, event_cursor],
                    "chronicle": [0, 0],
                    "checkpoint_generation": [0, 1],
                },
            }
        completed = generation - 1
        ordinal = snapshot_sequence
        partial = tick > 0 or absolute_tick_authority
        row_cursor = (
            {
                "completed_generation": completed,
                "last_row_key": f"generation:{completed}:agent:0",
            }
            if completed >= 0
            else {"completed_generation": -1, "last_row_key": None}
        )
        lineage_cursor = (
            {
                "completed_generation": completed,
                "last_row_key": f"generation:{completed}:lineage:0",
            }
            if completed >= 0
            else {"completed_generation": -1, "last_row_key": None}
        )
        return {
            "schema_version": "eiel_architecture_b_observer_continuation_v1",
            "checkpoint_next_generation": generation,
            "tick": tick,
            "typed_lifecycle_event_cursor": event_cursor,
            "exact_reporting_boundary_ordinal": ordinal,
            "behavior_completed_generation_and_row_key": row_cursor,
            "metrics_completed_generation": completed,
            "lineage_completed_generation_and_row_key": lineage_cursor,
            "chronicle_observation_ordinal": ordinal,
            "progress_sequence": ordinal,
            "schedule_phase": {
                "origin_generation": (
                    generation if partial else generation - snapshot_sequence
                ),
                "next_generation": generation,
            },
            "deduplication": {
                "acknowledged_record_keys": (
                    [f"generation:{completed}:exact_reporting"]
                    if completed >= 0
                    else []
                )
            },
            "acknowledged_channel_cursor_ranges": {
                "exact_reporting": [max(0, completed), generation],
                "behaviors": [max(0, completed), generation],
                "metrics": [max(0, completed), generation],
                "lineage": [max(0, completed), generation],
                "typed_lifecycle": [0, event_cursor],
                "chronicle": [max(0, ordinal - 1), ordinal],
                "checkpoint_generation": [generation, generation + 1],
            },
        }

    def _write_projection_bundle(
        self,
        projection: Mapping[str, Any],
        bundle_path: Path,
        *,
        generation: int,
        tick: int = 0,
        absolute_next_tick: int | None = None,
        batch_id: str,
        cell_id: str,
        segment_ordinal: int,
        expected_source_projection_sha256: str | None = None,
        performance_telemetry: dict[str, float] | None = None,
        native_streamed_checkpoint: bool = False,
        require_native_extension: bool = False,
        native_receipt: dict[str, Any] | None = None,
        native_capture_session: Any | None = None,
    ) -> str:
        writer = self._checkpoint_writer
        using_default_writer = writer is None
        digest_fn = self._checkpoint_digest
        if writer is None:
            from engine.simulation.checkpoint_bundle_v2 import (
                checkpoint_bundle_v2_save_projection,
            )

            writer = checkpoint_bundle_v2_save_projection
        if digest_fn is None:
            from engine.native_runtime.checkpoint_digests import (
                checkpoint_source_projection_authority_manifest,
            )

            def digest_fn(value: Mapping[str, Any]) -> str:
                return str(
                    checkpoint_source_projection_authority_manifest(value)[
                        "source_projection_sha256"
                    ]
                )

        outer_started = time.perf_counter()
        digest_started = time.perf_counter()
        source_digest = str(digest_fn(projection))
        source_digest_seconds = time.perf_counter() - digest_started
        if (
            expected_source_projection_sha256 is not None
            and source_digest != expected_source_projection_sha256
        ):
            raise CampaignProviderError(
                "ACS immutable checkpoint snapshot changed after capture"
            )
        identity_started = time.perf_counter()
        snapshot_sequence = _checkpoint_snapshot_sequence_v1(
            generation=generation,
            absolute_next_tick=absolute_next_tick,
        )
        try:
            from engine.native_runtime.lifecycle_events import (
                typed_lifecycle_events_from_projection,
                validated_typed_lifecycle_event_cursor,
            )

            event_cursor = validated_typed_lifecycle_event_cursor(
                projection, typed_lifecycle_events_from_projection(projection)
            )
        except Exception as exc:
            if self._checkpoint_writer is None:
                raise CampaignProviderError(
                    f"cannot persist checkpoint with invalid lifecycle authority: {exc}"
                ) from exc
            event_cursor = 0
        owner_epoch = max(1, snapshot_sequence, segment_ordinal)
        controller_id = f"{self.provider_id}-{cell_id}"
        identity = {
            "run_id": batch_id,
            "controller_id": controller_id,
            "owner_epoch": owner_epoch,
            "snapshot_id": (
                f"{controller_id}:owner:{owner_epoch}:boundary:{snapshot_sequence}"
            ),
            "snapshot_sequence": snapshot_sequence,
            "snapshot_digest_sha256": source_digest,
            "next_generation": generation,
            "tick": tick,
            "phase": "generation_boundary" if tick == 0 else "partial_generation",
            "source_projection_sha256": source_digest,
        }
        identity_seconds = time.perf_counter() - identity_started
        writer_telemetry: dict[str, float] = {}
        writer_kwargs: dict[str, Any] = {
            "generation_index": generation,
            "expected_source_projection_sha256": source_digest,
            "architecture_b_identity": identity,
            "observer_continuation": self._observer_continuation(
                generation,
                tick=tick,
                event_cursor=event_cursor,
                snapshot_sequence=snapshot_sequence,
                absolute_tick_authority=absolute_next_tick is not None,
            ),
            "strict_replay_mode": True,
        }
        if using_default_writer:
            writer_kwargs["performance_telemetry"] = writer_telemetry
        writer_started = time.perf_counter()
        if native_streamed_checkpoint:
            if not using_default_writer:
                raise CampaignProviderError(
                    "native streamed checkpoint route cannot use a legacy injected writer"
                )
            from engine.campaigns.acs_native_streamed_checkpoint_v1 import (
                write_acs_native_streamed_checkpoint,
            )

            published = write_acs_native_streamed_checkpoint(
                projection,
                bundle_path,
                cell_id=cell_id,
                generation=generation,
                tick=tick,
                absolute_next_tick=absolute_next_tick,
                batch_id=batch_id,
                segment_ordinal=segment_ordinal,
                source_projection_sha256=source_digest,
                architecture_b_identity=identity,
                observer_continuation=writer_kwargs["observer_continuation"],
                performance_telemetry=writer_telemetry,
                require_native_extension=require_native_extension,
                native_receipt=native_receipt,
                native_capture_session=native_capture_session,
            )
        else:
            published = str(
                writer(
                    projection,
                    str(bundle_path),
                    **writer_kwargs,
                )
            )
        if performance_telemetry is not None:
            performance_telemetry.clear()
            performance_telemetry.update(writer_telemetry)
            performance_telemetry.update(
                {
                    "source_digest_seconds": source_digest_seconds,
                    "checkpoint_identity_seconds": identity_seconds,
                    "writer_call_seconds": time.perf_counter() - writer_started,
                    "outer_total_seconds": time.perf_counter() - outer_started,
                }
            )
        return published

    def terminal_projection_digest(self, projection: Mapping[str, Any]) -> str:
        """Return the strict Bundle v2 source-authority digest for one cell."""

        digest_fn = self._checkpoint_digest
        if digest_fn is None:
            from engine.native_runtime.checkpoint_digests import (
                checkpoint_source_projection_authority_manifest,
            )

            def digest_fn(value: Mapping[str, Any]) -> str:
                return str(
                    checkpoint_source_projection_authority_manifest(value)[
                        "source_projection_sha256"
                    ]
                )

        return str(digest_fn(projection))

    def _layered_checkpoint_digest_v1(
        self,
        projection: Mapping[str, Any],
        layer: Any,
    ) -> str:
        """Return a mechanics/storage digest without mixing source provenance.

        ``_checkpoint_digest`` is the source-authority callback used to bind a
        projection to its immutable import and persistence identity.  It may
        legitimately include batch-local paths, event-sink provenance, and
        other storage metadata.  It must therefore not stand in for the
        trajectory or canonical-storage digest used for exact route parity.

        A few legacy unit-test fixtures intentionally contain only metadata and
        no tensor projection.  Preserve their injected-digest behavior while
        requiring real tensor-bearing projections to use the layered digest
        authority.
        """

        if "tensors" not in projection and self._checkpoint_digest is not None:
            return self.terminal_projection_digest(projection)
        from engine.native_runtime.checkpoint_digests import (
            checkpoint_digest_manifest,
        )

        return str(checkpoint_digest_manifest(projection, layer=layer)["digest"])

    @staticmethod
    def _acs_process_resource_sample_v1() -> dict[str, int | str | None]:
        """Collect non-mutating host-memory authority for ACS0 measurements."""

        sample: dict[str, int | str | None] = {
            "rss_bytes": None,
            "pss_bytes": None,
            "pinned_bytes": None,
            "host_available_bytes": None,
            "swap_free_bytes": None,
            "swap_total_bytes": None,
            "source": "linux_proc_status_and_smaps_rollup_v1",
        }

        def kibibytes(value: str) -> int | None:
            fields = value.split()
            if len(fields) < 2 or fields[1] != "kB":
                return None
            try:
                return int(fields[0]) * 1024
            except ValueError:
                return None

        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                name, separator, raw = line.partition(":")
                if not separator:
                    continue
                if name == "VmRSS":
                    sample["rss_bytes"] = kibibytes(raw)
                elif name == "VmPin":
                    sample["pinned_bytes"] = kibibytes(raw)
        except OSError:
            pass
        try:
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
                name, separator, raw = line.partition(":")
                if separator and name == "Pss":
                    sample["pss_bytes"] = kibibytes(raw)
                    break
        except OSError:
            pass
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, separator, raw = line.partition(":")
                if not separator:
                    continue
                field = {
                    "MemAvailable": "host_available_bytes",
                    "SwapFree": "swap_free_bytes",
                    "SwapTotal": "swap_total_bytes",
                }.get(name)
                if field is not None:
                    sample[field] = kibibytes(raw)
        except OSError:
            pass
        try:
            # Linux reports KiB. This process-lifetime high-water mark is a
            # conservative upper bound for every enclosed dispatch window and
            # cannot miss a transient RSS spike between endpoint samples.
            sample["rss_high_water_bytes"] = (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            )
        except (ValueError, OSError):
            sample["rss_high_water_bytes"] = None
        return sample

    @staticmethod
    def _acs_gpu_resource_sample_v1(
        execution_device: str | None,
    ) -> dict[str, int | str | None]:
        """Read allocator counters without launching CUDA work or synchronizing."""

        sample: dict[str, int | str | None] = {
            "memory_allocated_bytes": None,
            "memory_reserved_bytes": None,
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
            "free_memory_bytes": None,
            "total_memory_bytes": None,
            "source": "torch_cuda_allocator_counters_no_kernel_no_sync_v1",
        }
        if not isinstance(execution_device, str) or not execution_device.startswith(
            "cuda"
        ):
            return sample
        try:
            import torch

            free_bytes, total_bytes = torch.cuda.mem_get_info(execution_device)

            sample.update(
                {
                    "memory_allocated_bytes": int(
                        torch.cuda.memory_allocated(execution_device)
                    ),
                    "memory_reserved_bytes": int(
                        torch.cuda.memory_reserved(execution_device)
                    ),
                    "max_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(execution_device)
                    ),
                    "max_memory_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(execution_device)
                    ),
                    "free_memory_bytes": int(free_bytes),
                    "total_memory_bytes": int(total_bytes),
                }
            )
        except (ImportError, RuntimeError, ValueError):
            pass
        return sample

    @staticmethod
    def _resident_tensor_storage_bytes_v1(value: Any) -> dict[str, int]:
        """Count unique resident tensor storage without copying or synchronizing."""

        try:
            import torch
        except ImportError:
            return {"all_tensor_bytes": 0, "cuda_tensor_bytes": 0}
        seen_objects: set[int] = set()
        seen_storages: set[tuple[str, int]] = set()
        totals = {"all_tensor_bytes": 0, "cuda_tensor_bytes": 0}

        def visit(candidate: Any) -> None:
            object_id = id(candidate)
            if object_id in seen_objects:
                return
            seen_objects.add(object_id)
            if isinstance(candidate, torch.Tensor):
                storage = candidate.untyped_storage()
                key = (str(candidate.device), int(storage.data_ptr()))
                if key in seen_storages:
                    return
                seen_storages.add(key)
                size = int(storage.nbytes())
                totals["all_tensor_bytes"] += size
                if candidate.is_cuda:
                    totals["cuda_tensor_bytes"] += size
                return
            if isinstance(candidate, Mapping):
                for item in candidate.values():
                    visit(item)
                return
            if isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray)
            ):
                for item in candidate:
                    visit(item)

        visit(value)
        return totals

    @staticmethod
    def _acs_projection_tensor_bytes(
        projections: Sequence[Mapping[str, Any]],
    ) -> int:
        """Read detached projection byte authority without touching tensors."""

        total = 0
        for projection in projections:
            manifest = projection.get("tensor_manifest")
            if not isinstance(manifest, Mapping):
                continue
            for row in manifest.values():
                if not isinstance(row, Mapping):
                    raise CampaignProviderError(
                        "ACS projection tensor manifest contains a non-object row"
                    )
                value = row.get("bytes")
                if type(value) is not int or value < 0:
                    raise CampaignProviderError(
                        "ACS projection tensor manifest contains invalid bytes"
                    )
                total += value
        return total

    @staticmethod
    def _acs_published_bundle_bytes_v1(path: str) -> int | None:
        """Measure published checkpoint bytes without reading bundle content."""

        root = Path(path)
        if root.is_file():
            try:
                return int(root.stat().st_size)
            except OSError as exc:
                raise CampaignProviderError(
                    f"ACS published checkpoint file size is unavailable: {exc}"
                ) from exc
        if not root.is_dir():
            return None
        total = 0
        try:
            for candidate in root.rglob("*"):
                if candidate.is_file():
                    total += candidate.stat().st_size
        except OSError as exc:
            raise CampaignProviderError(
                f"ACS published checkpoint bundle size is unavailable: {exc}"
            ) from exc
        return total

    def _aggregate_checkpoint_authorities_v1(
        self,
        *,
        request: ResidentBatchRequestV1,
        source_projections: Sequence[Mapping[str, Any]],
        generation: int,
        tick: int,
        absolute_next_tick: int | None,
        segment_ordinal: int,
        expected_source_projection_digest_sha256s: Mapping[str, str] | None,
    ) -> dict[str, Any]:
        """Build scalar checkpoint authorities before a detached capture.

        This method is intentionally projection-facing only on the producer
        thread.  Its return value contains digests, lifecycle identities, and
        observer continuations, never tensor-bearing state.
        """

        from engine.native_runtime.checkpoint_digests import (
            DigestLayer,
        )

        if len(source_projections) != len(request.cell_ids):
            raise CampaignProviderError(
                "ACS detached aggregate projection count does not match the batch"
            )
        cuda_source_field_counts = (
            _validate_aggregate_native_production_owner_route_v1(
                request,
                source_projections,
            )
        )
        if expected_source_projection_digest_sha256s is not None and set(
            expected_source_projection_digest_sha256s
        ) != set(request.cell_ids):
            raise CampaignProviderError(
                "ACS detached snapshot digest authority does not cover the batch"
            )

        source_digest_started = time.perf_counter()
        source_digests: dict[str, str] = {}
        for cell_id, projection in zip(
            request.cell_ids, source_projections, strict=True
        ):
            expected = (
                expected_source_projection_digest_sha256s.get(cell_id)
                if expected_source_projection_digest_sha256s is not None
                else None
            )
            source_digests[cell_id] = (
                expected
                if isinstance(expected, str)
                else self.terminal_projection_digest(projection)
            )
        source_digest_seconds = time.perf_counter() - source_digest_started

        snapshot_sequence = _checkpoint_snapshot_sequence_v1(
            generation=generation,
            absolute_next_tick=absolute_next_tick,
        )
        identities: dict[str, dict[str, Any]] = {}
        observers: dict[str, dict[str, Any]] = {}
        identity_started = time.perf_counter()
        for cell_id, projection in zip(
            request.cell_ids, source_projections, strict=True
        ):
            try:
                from engine.native_runtime.lifecycle_events import (
                    typed_lifecycle_events_from_projection,
                    validated_typed_lifecycle_event_cursor,
                )

                event_cursor = validated_typed_lifecycle_event_cursor(
                    projection,
                    typed_lifecycle_events_from_projection(projection),
                )
            except Exception as exc:
                if self._checkpoint_writer is None:
                    raise CampaignProviderError(
                        "cannot capture aggregate checkpoint with invalid lifecycle authority: "
                        f"{exc}"
                    ) from exc
                event_cursor = 0
            owner_epoch = max(1, snapshot_sequence, segment_ordinal)
            controller_id = f"{self.provider_id}-{cell_id}"
            source_digest = source_digests[cell_id]
            identities[cell_id] = {
                "run_id": request.batch_id,
                "controller_id": controller_id,
                "owner_epoch": owner_epoch,
                "snapshot_id": (
                    f"{controller_id}:owner:{owner_epoch}:boundary:{snapshot_sequence}"
                ),
                "snapshot_sequence": snapshot_sequence,
                "snapshot_digest_sha256": source_digest,
                "next_generation": generation,
                "tick": tick,
                "phase": "generation_boundary" if tick == 0 else "partial_generation",
                "source_projection_sha256": source_digest,
            }
            observers[cell_id] = self._observer_continuation(
                generation,
                tick=tick,
                event_cursor=event_cursor,
                snapshot_sequence=snapshot_sequence,
                absolute_tick_authority=absolute_next_tick is not None,
            )
        identity_seconds = time.perf_counter() - identity_started

        mechanics_checkpoint_digests: dict[str, str] = {}
        canonical_checkpoint_storage_digests: dict[str, str] = {}
        for cell_id, projection in zip(
            request.cell_ids, source_projections, strict=True
        ):
            mechanics_checkpoint_digests[cell_id] = (
                self._layered_checkpoint_digest_v1(
                    projection,
                    DigestLayer.TRAJECTORY_STATE,
                )
            )
            canonical_checkpoint_storage_digests[cell_id] = (
                self._layered_checkpoint_digest_v1(
                    projection,
                    DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
                )
            )
        return {
            "source_projection_sha256s": source_digests,
            "architecture_b_identities": identities,
            "observer_continuations": observers,
            "mechanics_checkpoint_digests": mechanics_checkpoint_digests,
            "canonical_checkpoint_storage_digests": (
                canonical_checkpoint_storage_digests
            ),
            "source_digest_seconds": source_digest_seconds,
            "checkpoint_identity_seconds": identity_seconds,
            "cuda_source_field_counts": {
                cell_id: count
                for cell_id, count in zip(
                    request.cell_ids,
                    cuda_source_field_counts,
                    strict=True,
                )
            },
        }

    def _aggregate_checkpoint_authority_factory_v1(
        self,
        *,
        request: ResidentBatchRequestV1,
        generation: int,
        tick: int,
        absolute_next_tick: int | None,
        segment_ordinal: int,
        expected_source_projection_digest_sha256s: Mapping[str, str] | None,
    ) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
        """Return a scalar-only authority factory for a bounded cursor.

        Detached ACS capture consumes the projection cursor itself.  Digest,
        lifecycle, observer, and checkpoint authorities must therefore be
        computed while each cursor cell is live rather than by first creating
        a batch-wide projection tuple.
        """

        from engine.native_runtime.checkpoint_digests import DigestLayer

        if expected_source_projection_digest_sha256s is not None and set(
            expected_source_projection_digest_sha256s
        ) != set(request.cell_ids):
            raise CampaignProviderError(
                "ACS detached cursor digest authority does not cover the batch"
            )
        cell_indices = {
            cell_id: index for index, cell_id in enumerate(request.cell_ids)
        }
        snapshot_sequence = _checkpoint_snapshot_sequence_v1(
            generation=generation,
            absolute_next_tick=absolute_next_tick,
        )

        def authority_factory(
            cell_id: str, projection: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            cell_index = cell_indices.get(cell_id)
            if cell_index is None:
                raise CampaignProviderError(
                    "ACS detached cursor authority cell differs from the request"
                )
            counts = _aggregate_native_cuda_source_field_counts_v1((projection,))
            if request.profile_id == PRODUCTION_PROFILE_ID:
                parameters = dict(getattr(request, "parameter_arrays", ()))
                staging_values = parameters.get(
                    "cuda_architecture_b.cpu_request_staging_enable"
                )
                if (
                    not isinstance(staging_values, tuple)
                    or len(staging_values) != len(request.cell_ids)
                    or any(value is not False for value in staging_values)
                ):
                    raise CampaignProviderError(
                        "ACS detached production capture requires explicit "
                        "cuda_architecture_b.cpu_request_staging_enable=false "
                        "for every cell"
                    )

            expected = (
                expected_source_projection_digest_sha256s.get(cell_id)
                if expected_source_projection_digest_sha256s is not None
                else None
            )
            source_digest = (
                expected
                if isinstance(expected, str)
                else self.terminal_projection_digest(projection)
            )
            try:
                from engine.native_runtime.lifecycle_events import (
                    typed_lifecycle_events_from_projection,
                    validated_typed_lifecycle_event_cursor,
                )

                event_cursor = validated_typed_lifecycle_event_cursor(
                    projection,
                    typed_lifecycle_events_from_projection(projection),
                )
            except Exception as exc:
                if self._checkpoint_writer is None:
                    raise CampaignProviderError(
                        "cannot capture detached aggregate checkpoint with invalid "
                        f"lifecycle authority: {exc}"
                    ) from exc
                event_cursor = 0
            owner_epoch = max(1, snapshot_sequence, segment_ordinal)
            controller_id = f"{self.provider_id}-{cell_id}"
            identity = {
                "run_id": request.batch_id,
                "controller_id": controller_id,
                "owner_epoch": owner_epoch,
                "snapshot_id": (
                    f"{controller_id}:owner:{owner_epoch}:boundary:{snapshot_sequence}"
                ),
                "snapshot_sequence": snapshot_sequence,
                "snapshot_digest_sha256": source_digest,
                "next_generation": generation,
                "tick": tick,
                "phase": "generation_boundary" if tick == 0 else "partial_generation",
                "source_projection_sha256": source_digest,
            }
            return {
                "source_projection_sha256": source_digest,
                "architecture_b_identity": identity,
                "observer_continuation": self._observer_continuation(
                    generation,
                    tick=tick,
                    event_cursor=event_cursor,
                    snapshot_sequence=snapshot_sequence,
                    absolute_tick_authority=absolute_next_tick is not None,
                ),
                "mechanics_checkpoint_digest": self._layered_checkpoint_digest_v1(
                    projection, DigestLayer.TRAJECTORY_STATE
                ),
                "canonical_checkpoint_storage_digest": (
                    self._layered_checkpoint_digest_v1(
                        projection, DigestLayer.CANONICAL_CHECKPOINT_STORAGE
                    )
                ),
                "cuda_source_field_count": counts[0],
                "cell_index": cell_index,
            }

        return authority_factory

    def _acs_detached_capture_measurement_v1(
        self,
        *,
        native_receipt: Mapping[str, Any],
        direct_result: Mapping[str, Any],
        checkpoint_projection_time_ns: int,
        capture_started_unix_ns: int | None,
        capture_started_perf_ns: int,
        capture_completed_perf_ns: int,
        capture_cpu_started_ns: int | None,
        coordinate: int,
        captured_bytes: int,
        source_projection_sha256s: Mapping[str, str],
        host_resources_before: Mapping[str, Any] | None,
        gpu_resources_before: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the ACS barrier receipt for cursor-backed detached capture."""

        if type(capture_started_unix_ns) is not int:
            raise CampaignProviderError(
                "detached ACS capture has no start timestamp"
            )
        if type(checkpoint_projection_time_ns) is not int or checkpoint_projection_time_ns < 0:
            raise CampaignProviderError(
                "detached ACS projection time is invalid"
            )
        if type(captured_bytes) is not int or captured_bytes < 0:
            raise CampaignProviderError("detached ACS captured bytes are invalid")
        if not isinstance(source_projection_sha256s, Mapping):
            raise CampaignProviderError(
                "detached ACS source digest authority is invalid"
            )
        native_final_sync_time_ns = _require_acs_native_nonnegative_int_v1(
            direct_result, "final_stream_synchronization_time_ns"
        )
        last_useful_kernel_completion_observed_host_time_ns = (
            _require_acs_native_nonnegative_int_v1(
                direct_result,
                "last_useful_kernel_completion_observed_host_time_ns",
            )
        )
        if last_useful_kernel_completion_observed_host_time_ns == 0:
            raise CampaignProviderError(
                "detached ACS capture lacks the last-useful-kernel boundary"
            )
        final_host_synchronizations = _require_acs_native_nonnegative_int_v1(
            direct_result, "final_host_synchronizations"
        )
        final_device_to_host_operations = _require_acs_native_nonnegative_int_v1(
            direct_result, "final_device_to_host_operations"
        )
        compact_settlement_d2h_operations = _require_acs_native_nonnegative_int_v1(
            direct_result, "compact_settlement_d2h_operations"
        )
        reporting_storage = direct_result.get("compact_summary_reporting_storage")
        if not isinstance(reporting_storage, Mapping):
            raise CampaignProviderError(
                "detached ACS capture lacks compact reporting storage"
            )
        final_transfer = reporting_storage.get("final_transfer")
        if not isinstance(final_transfer, Mapping):
            raise CampaignProviderError(
                "detached ACS capture lacks final transfer contract"
            )
        final_transfer_bytes = final_transfer.get("bytes")
        observed_host_transfer_count = final_transfer.get(
            "observed_host_transfer_count"
        )
        if type(final_transfer_bytes) is not int or final_transfer_bytes < 0:
            raise CampaignProviderError(
                "detached ACS final transfer bytes are invalid"
            )
        if (
            type(observed_host_transfer_count) is not int
            or observed_host_transfer_count < 0
            or observed_host_transfer_count != compact_settlement_d2h_operations
        ):
            raise CampaignProviderError(
                "detached ACS final transfer count differs from compact settlement"
            )
        if final_transfer.get("status") not in {
            "native_stream_ordered_callback_wired",
            "disabled",
        }:
            raise CampaignProviderError(
                "detached ACS final transfer status is invalid"
            )
        if host_resources_before is None or gpu_resources_before is None:
            raise CampaignProviderError(
                "detached ACS capture lacks before-resource measurements"
            )
        capture_completed_unix_ns = time.time_ns()
        host_resources_after = self._acs_process_resource_sample_v1()
        gpu_resources_after = self._acs_gpu_resource_sample_v1(
            native_receipt["execution_device"]
        )
        capture_cpu_time_ns = (
            time.process_time_ns() - capture_cpu_started_ns
            if capture_cpu_started_ns is not None
            else None
        )
        host_capture_freeze_time_ns = max(
            0,
            capture_completed_perf_ns - capture_started_perf_ns,
        )
        return {
            "schema_version": "eiel_acs_capture_barrier_measurement_v1",
            "capture_started_unix_ns": capture_started_unix_ns,
            "capture_completed_unix_ns": capture_completed_unix_ns,
            "capture_started_perf_ns": capture_started_perf_ns,
            "capture_completed_perf_ns": capture_completed_perf_ns,
            "coordinate": coordinate,
            "execution_device": native_receipt["execution_device"],
            "segment_dispatch_started_perf_ns": native_receipt.get(
                "segment_dispatch_started_perf_ns"
            ),
            "segment_dispatch_finished_perf_ns": native_receipt.get(
                "segment_dispatch_finished_perf_ns"
            ),
            "checkpoint_projection_time_ns": checkpoint_projection_time_ns,
            "snapshot_digest_time_ns": 0,
            "host_capture_freeze_time_ns": host_capture_freeze_time_ns,
            "capture_cpu_time_ns": capture_cpu_time_ns,
            "native_final_stream_synchronization_time_ns": native_final_sync_time_ns,
            "last_useful_kernel_completion_observed_host_time_ns": (
                last_useful_kernel_completion_observed_host_time_ns
            ),
            "capture_barrier_time_ns": (
                host_capture_freeze_time_ns + native_final_sync_time_ns
            ),
            "post_sync_gpu_idle_proxy_time_ns": host_capture_freeze_time_ns,
            "captured_bytes": captured_bytes,
            "captured_cell_count": len(source_projection_sha256s),
            "immutable_snapshot_digest_sha256s": dict(
                source_projection_sha256s
            ),
            "host_resources_before": dict(host_resources_before),
            "host_resources_peak": dict(host_resources_after),
            "host_resources_after": host_resources_after,
            "gpu_resources_before": dict(gpu_resources_before),
            "gpu_resources_peak": dict(gpu_resources_after),
            "gpu_resources_after": gpu_resources_after,
            "native_transfer_receipt": {
                "final_host_synchronizations": final_host_synchronizations,
                "final_device_to_host_operations": final_device_to_host_operations,
                "compact_settlement_d2h_operations": compact_settlement_d2h_operations,
                "final_transfer_bytes": final_transfer_bytes,
                "final_transfer_contract": dict(final_transfer),
            },
            "measurement_scope": (
                "native_final_sync_plus_provider_projector_settlement_"
                "cursor_bounded_cell_major_spool_capture"
            ),
            "limitations": [
                "cursor authority digest time is included in bounded capture time and is not separately event-timed",
                "host freeze is represented by the bounded cursor capture interval",
                "post-sync host capture time is a GPU-idle proxy; external utilization telemetry remains required for ACS decision evidence",
            ],
        }

    def _aggregate_checkpoint_set_from_detached_receipt_v1(
        self,
        *,
        request: ResidentBatchRequestV1,
        receipt: Mapping[str, Any],
        authorities: Mapping[str, Any],
        writer_telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind one worker-published aggregate receipt into provider state."""

        if receipt.get("status") != "complete":
            raise CampaignProviderError(
                "detached aggregate checkpoint receipt is not complete"
            )
        published = receipt.get("path")
        if not isinstance(published, str) or not published:
            raise CampaignProviderError(
                "detached aggregate checkpoint receipt lacks its publication path"
            )
        published_bytes = self._acs_published_bundle_bytes_v1(published)
        if published_bytes is None:
            raise CampaignProviderError(
                "detached aggregate checkpoint publication has no measurable file size"
            )
        source_digests = authorities.get("source_projection_sha256s")
        mechanics_digests = authorities.get("mechanics_checkpoint_digests")
        canonical_digests = authorities.get("canonical_checkpoint_storage_digests")
        if not all(
            isinstance(value, Mapping)
            for value in (source_digests, mechanics_digests, canonical_digests)
        ):
            raise CampaignProviderError(
                "detached aggregate checkpoint authorities are incomplete"
            )
        expected_cells = set(request.cell_ids)
        if any(
            set(value) != expected_cells
            for value in (source_digests, mechanics_digests, canonical_digests)
        ):
            raise CampaignProviderError(
                "detached aggregate checkpoint authorities do not exactly cover the batch"
            )
        raw_cells = receipt.get("cells")
        if not isinstance(raw_cells, Mapping) or set(raw_cells) != expected_cells:
            raise CampaignProviderError(
                "detached aggregate receipt does not cover the batch"
            )

        writer_service_time_ns = receipt.get("detached_writer_wall_time_ns", 0)
        writer_cpu_time_ns = receipt.get("detached_writer_cpu_time_ns", 0)
        if type(writer_service_time_ns) is not int or writer_service_time_ns < 0:
            raise CampaignProviderError(
                "detached aggregate writer wall time is invalid"
            )
        if type(writer_cpu_time_ns) is not int or writer_cpu_time_ns < 0:
            raise CampaignProviderError(
                "detached aggregate writer CPU time is invalid"
            )
        cell_stage_telemetry_seconds = {
            cell_id: dict(writer_telemetry) for cell_id in request.cell_ids
        }
        cell_service_times_ns = {cell_id: 0 for cell_id in request.cell_ids}
        cell_cpu_times_ns = {cell_id: 0 for cell_id in request.cell_ids}
        cell_checkpoint_bytes = {cell_id: 0 for cell_id in request.cell_ids}
        cell_physical_attribution: dict[str, int] = {}
        bundle_rows: list[dict[str, Any]] = []
        native_checkpoint_receipts: dict[str, dict[str, Any]] = {}
        for cell_id in request.cell_ids:
            cell_receipt = raw_cells.get(cell_id)
            if not isinstance(cell_receipt, Mapping):
                raise CampaignProviderError(
                    f"detached aggregate receipt cell is invalid: {cell_id}"
                )
            worker_source_digest = cell_receipt.get("source_projection_sha256")
            authoritative_source_digest = source_digests.get(cell_id)
            if (
                not isinstance(worker_source_digest, str)
                or not isinstance(authoritative_source_digest, str)
                or worker_source_digest != authoritative_source_digest
            ):
                raise CampaignProviderError(
                    "detached aggregate worker/source authority digest differs: "
                    f"{cell_id}"
                )
            physical_bytes = cell_receipt.get("physical_bytes")
            if type(physical_bytes) is not int or physical_bytes < 0:
                raise CampaignProviderError(
                    f"detached aggregate physical attribution is invalid: {cell_id}"
                )
            cell_physical_attribution[cell_id] = physical_bytes
            native_receipt = {
                **dict(receipt),
                **dict(cell_receipt),
                "cell_id": cell_id,
                "path": published,
                "physical_format": "acs100x_native_aggregate_checkpoint_v2",
            }
            native_checkpoint_receipts[cell_id] = native_receipt
            bundle_rows.append(
                {
                    "cell_id": cell_id,
                    "path": published,
                    "source_projection_sha256": source_digests[cell_id],
                    "physical_format": "acs100x_native_aggregate_checkpoint_v2",
                    "artifact_sha256": receipt["artifact_sha256"],
                    "payload_sha256": receipt["payload_sha256"],
                    "seal_sha256": receipt["seal_sha256"],
                    "payload_bytes": receipt["payload_bytes"],
                    "chunk_count": cell_receipt["chunk_count"],
                    "aggregate_cell_id": cell_id,
                    "aggregate_physical_bytes": published_bytes,
                    "aggregate_logical_bytes": receipt["logical_bytes"],
                    "aggregate_zero_fill_logical_bytes": receipt[
                        "zero_fill_logical_bytes"
                    ],
                    "cell_physical_bytes": physical_bytes,
                }
            )
        return {
            "bundle_rows": bundle_rows,
            "terminal_cell_digests": dict(source_digests),
            "mechanics_checkpoint_digests": dict(mechanics_digests),
            "canonical_checkpoint_storage_digests": dict(canonical_digests),
            "writer_service_time_ns": writer_service_time_ns,
            "writer_cpu_time_ns": writer_cpu_time_ns,
            "cell_service_times_ns": cell_service_times_ns,
            "cell_cpu_times_ns": cell_cpu_times_ns,
            "cell_stage_telemetry_seconds": cell_stage_telemetry_seconds,
            "cell_checkpoint_bytes": cell_checkpoint_bytes,
            "cell_physical_bytes_attribution": cell_physical_attribution,
            "checkpoint_bytes": published_bytes,
            "aggregate_checkpoint_bytes": published_bytes,
            "aggregate_checkpoint_logical_bytes": receipt["logical_bytes"],
            "aggregate_checkpoint_zero_fill_logical_bytes": receipt[
                "zero_fill_logical_bytes"
            ],
            "aggregate_checkpoint_receipt": dict(receipt),
            "native_checkpoint_receipts": native_checkpoint_receipts,
            "native_capture_group_receipts": {},
            "checkpoint_bytes_measurement_status": "complete",
        }

    def _world_checkpoint_set_from_detached_receipt_v1(
        self,
        *,
        request: ResidentBatchRequestV1,
        receipt: Mapping[str, Any],
        writer_telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind one atomically published world-sharded set into provider state."""

        from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
            SET_MANIFEST_SCHEMA_VERSION,
            load_world_checkpoint_set_manifest_v1,
        )

        if (
            receipt.get("schema_version") != SET_MANIFEST_SCHEMA_VERSION
            or receipt.get("status") != "complete"
            or receipt.get("world_count") != len(request.cell_ids)
        ):
            raise CampaignProviderError(
                "detached world-sharded checkpoint receipt is incomplete"
            )
        manifest_path = receipt.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise CampaignProviderError(
                "detached world-sharded checkpoint lacks its set manifest"
            )
        manifest = load_world_checkpoint_set_manifest_v1(manifest_path)
        if manifest.get("cell_ids") != list(request.cell_ids):
            raise CampaignProviderError(
                "detached world-sharded checkpoint cell order differs"
            )
        set_root = Path(manifest_path).resolve().parent
        rows = manifest["objects"]
        terminal_digests = {
            str(row["cell_id"]): str(row["world_state_sha256"]) for row in rows
        }
        mechanics_digests = {
            str(row["cell_id"]): str(row["world_mechanics_sha256"])
            for row in rows
        }
        storage_digests = {
            str(row["cell_id"]): str(row["content_sha256"]) for row in rows
        }
        bundle_rows: list[dict[str, Any]] = []
        native_receipts: dict[str, dict[str, Any]] = {}
        cell_bytes: dict[str, int] = {}
        for row in rows:
            cell_id = str(row["cell_id"])
            object_path = set_root / str(row["relative_path"])
            physical_bytes = int(row["physical_file_bytes"])
            if not object_path.is_file() or object_path.stat().st_size != physical_bytes:
                raise CampaignProviderError(
                    f"world-sharded checkpoint object is unavailable: {cell_id}"
                )
            cell_bytes[cell_id] = physical_bytes
            bundle_row = {
                "cell_id": cell_id,
                "path": str(object_path),
                "world_object_relative_path": str(row["relative_path"]),
                "source_projection_sha256": str(row["world_state_sha256"]),
                "source_dependency_projection_sha256": str(
                    row["source_projection_sha256"]
                ),
                "terminal_world_state_sha256": str(row["world_state_sha256"]),
                "physical_format": "eiel_acs_world_checkpoint_object_v1",
                "checkpoint_set_id": manifest["checkpoint_set_id"],
                "checkpoint_set_manifest_path": manifest_path,
                "artifact_sha256": str(row["artifact_sha256"]),
                "payload_sha256": str(row["payload_sha256"]),
                "content_sha256": str(row["content_sha256"]),
                "payload_bytes": int(row["payload_bytes"]),
                "cell_physical_bytes": physical_bytes,
            }
            bundle_rows.append(bundle_row)
            native_receipts[cell_id] = dict(bundle_row)
        manifest_bytes = Path(manifest_path).stat().st_size
        total_bytes = sum(cell_bytes.values()) + manifest_bytes
        writer_service_time_ns = receipt.get("detached_writer_wall_time_ns", 0)
        writer_cpu_time_ns = receipt.get("detached_writer_cpu_time_ns", 0)
        if type(writer_service_time_ns) is not int or writer_service_time_ns < 0:
            raise CampaignProviderError("world-sharded writer wall time is invalid")
        if type(writer_cpu_time_ns) is not int or writer_cpu_time_ns < 0:
            raise CampaignProviderError("world-sharded writer CPU time is invalid")
        return {
            "bundle_rows": bundle_rows,
            "terminal_cell_digests": terminal_digests,
            "mechanics_checkpoint_digests": mechanics_digests,
            "canonical_checkpoint_storage_digests": storage_digests,
            "checkpoint_digest_domain": "physical_world_state_v1",
            "writer_service_time_ns": writer_service_time_ns,
            "writer_cpu_time_ns": writer_cpu_time_ns,
            "cell_service_times_ns": {cell_id: 0 for cell_id in request.cell_ids},
            "cell_cpu_times_ns": {cell_id: 0 for cell_id in request.cell_ids},
            "cell_stage_telemetry_seconds": {
                cell_id: dict(writer_telemetry) for cell_id in request.cell_ids
            },
            "cell_checkpoint_bytes": dict(cell_bytes),
            "cell_physical_bytes_attribution": dict(cell_bytes),
            "checkpoint_bytes": total_bytes,
            "aggregate_checkpoint_bytes": 0,
            "aggregate_checkpoint_logical_bytes": 0,
            "aggregate_checkpoint_zero_fill_logical_bytes": 0,
            "aggregate_checkpoint_receipt": {},
            "world_checkpoint_set_receipt": dict(receipt),
            "checkpoint_set_manifest_path": manifest_path,
            "checkpoint_set_manifest_sha256": manifest["manifest_sha256"],
            "native_checkpoint_receipts": native_receipts,
            "native_capture_group_receipts": {
                str(manifest["checkpoint_set_id"]): {
                    "schema_version": SET_MANIFEST_SCHEMA_VERSION,
                    "manifest_path": manifest_path,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "world_count": manifest["world_count"],
                    "physical_file_bytes": total_bytes,
                }
            },
            "checkpoint_bytes_measurement_status": "complete",
        }

    def _persist_aggregate_projection_checkpoint_cursor_set_v1(
        self,
        *,
        batch_root: Path,
        request: ResidentBatchRequestV1,
        projection_cursor: Any,
        generation: int,
        tick: int,
        absolute_next_tick: int | None,
        segment_ordinal: int,
        expected_source_projection_digest_sha256s: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist one multicell aggregate from a one-cell cursor.

        Tick-native settlement deliberately refuses to materialize a B136
        projection sequence.  The aggregate writer therefore consumes the
        cursor directly and keeps only scalar authorities plus one slab-sized
        field spool alive at a time.
        """

        from engine.native_runtime.checkpoint_digests import DigestLayer
        from engine.campaigns.acs_native_aggregate_checkpoint_v1 import (
            write_acs_native_aggregate_checkpoint_from_cursor,
        )

        if expected_source_projection_digest_sha256s is not None and set(
            expected_source_projection_digest_sha256s
        ) != set(request.cell_ids):
            raise CampaignProviderError(
                "ACS cursor snapshot digest authority does not cover the batch"
            )
        cell_indices = {
            cell_id: index for index, cell_id in enumerate(request.cell_ids)
        }
        snapshot_sequence = _checkpoint_snapshot_sequence_v1(
            generation=generation,
            absolute_next_tick=absolute_next_tick,
        )

        def authority_factory(
            cell_id: str, projection: Mapping[str, Any]
        ) -> Mapping[str, Any]:
            cell_index = cell_indices.get(cell_id)
            if cell_index is None:
                raise CampaignProviderError(
                    "ACS cursor authority cell differs from the request"
                )
            counts = _aggregate_native_cuda_source_field_counts_v1((projection,))
            if request.profile_id == PRODUCTION_PROFILE_ID:
                parameters = dict(getattr(request, "parameter_arrays", ()))
                staging_values = parameters.get(
                    "cuda_architecture_b.cpu_request_staging_enable"
                )
                if (
                    not isinstance(staging_values, tuple)
                    or len(staging_values) != len(request.cell_ids)
                    or any(value is not False for value in staging_values)
                ):
                    raise CampaignProviderError(
                        "ACS aggregate production capture requires explicit "
                        "cuda_architecture_b.cpu_request_staging_enable=false "
                        "for every cell"
                    )

            expected = (
                expected_source_projection_digest_sha256s.get(cell_id)
                if expected_source_projection_digest_sha256s is not None
                else None
            )
            source_digest = (
                expected
                if isinstance(expected, str)
                else self.terminal_projection_digest(projection)
            )
            try:
                from engine.native_runtime.lifecycle_events import (
                    typed_lifecycle_events_from_projection,
                    validated_typed_lifecycle_event_cursor,
                )

                event_cursor = validated_typed_lifecycle_event_cursor(
                    projection,
                    typed_lifecycle_events_from_projection(projection),
                )
            except Exception as exc:
                if self._checkpoint_writer is None:
                    raise CampaignProviderError(
                        "cannot persist aggregate checkpoint with invalid "
                        f"lifecycle authority: {exc}"
                    ) from exc
                event_cursor = 0
            owner_epoch = max(1, snapshot_sequence, segment_ordinal)
            controller_id = f"{self.provider_id}-{cell_id}"
            identity = {
                "run_id": request.batch_id,
                "controller_id": controller_id,
                "owner_epoch": owner_epoch,
                "snapshot_id": (
                    f"{controller_id}:owner:{owner_epoch}:boundary:{snapshot_sequence}"
                ),
                "snapshot_sequence": snapshot_sequence,
                "snapshot_digest_sha256": source_digest,
                "next_generation": generation,
                "tick": tick,
                "phase": "generation_boundary" if tick == 0 else "partial_generation",
                "source_projection_sha256": source_digest,
            }
            return {
                "source_projection_sha256": source_digest,
                "architecture_b_identity": identity,
                "observer_continuation": self._observer_continuation(
                    generation,
                    tick=tick,
                    event_cursor=event_cursor,
                    snapshot_sequence=snapshot_sequence,
                    absolute_tick_authority=absolute_next_tick is not None,
                ),
                "mechanics_checkpoint_digest": self._layered_checkpoint_digest_v1(
                    projection, DigestLayer.TRAJECTORY_STATE
                ),
                "canonical_checkpoint_storage_digest": (
                    self._layered_checkpoint_digest_v1(
                        projection, DigestLayer.CANONICAL_CHECKPOINT_STORAGE
                    )
                ),
                "cuda_source_field_count": counts[0],
                "cell_index": cell_index,
            }

        destination = (
            batch_root
            / "checkpoint_native_aggregate_v1"
            / (
                f"checkpoint_tick{absolute_next_tick:012d}.acsaggregate"
                if absolute_next_tick is not None
                else f"checkpoint_gen{generation:06d}.acsaggregate"
            )
        )
        writer_telemetry: dict[str, float] = {}
        writer_started_ns = time.perf_counter_ns()
        writer_started_cpu_ns = time.thread_time_ns()
        stream_result = write_acs_native_aggregate_checkpoint_from_cursor(
            projection_cursor,
            destination,
            cell_ids=request.cell_ids,
            generation=generation,
            tick=tick,
            absolute_next_tick=absolute_next_tick,
            batch_id=request.batch_id,
            segment_ordinal=segment_ordinal,
            authority_factory=authority_factory,
            require_native_extension=(request.profile_id == PRODUCTION_PROFILE_ID),
            performance_telemetry=writer_telemetry,
        )
        writer_service_time_ns = time.perf_counter_ns() - writer_started_ns
        writer_cpu_time_ns = time.thread_time_ns() - writer_started_cpu_ns
        receipt = stream_result.get("receipt")
        authorities = stream_result.get("authorities_by_cell")
        if not isinstance(receipt, Mapping) or not isinstance(authorities, Mapping):
            raise CampaignProviderError(
                "cursor aggregate writer returned incomplete authorities"
            )
        writer_telemetry.update(
            {
                "owner_snapshot_capture_mode": (
                    "bounded_canonical_projection_cursor_v1"
                ),
                "cuda_source_field_counts": {
                    cell_id: int(authorities[cell_id]["cuda_source_field_count"])
                    for cell_id in request.cell_ids
                },
                "cpu_request_staging_enabled": False,
            }
        )
        published = receipt.get("path")
        if not isinstance(published, str) or not published:
            raise CampaignProviderError(
                "cursor aggregate writer returned no publication path"
            )
        published_bytes = self._acs_published_bundle_bytes_v1(published)
        if published_bytes is None:
            raise CampaignProviderError(
                "cursor aggregate checkpoint publication has no measurable file size"
            )
        source_digests = {
            cell_id: str(authorities[cell_id]["source_projection_sha256"])
            for cell_id in request.cell_ids
        }
        mechanics_digests = {
            cell_id: str(authorities[cell_id]["mechanics_checkpoint_digest"])
            for cell_id in request.cell_ids
        }
        canonical_digests = {
            cell_id: str(
                authorities[cell_id]["canonical_checkpoint_storage_digest"]
            )
            for cell_id in request.cell_ids
        }
        bundle_rows: list[dict[str, Any]] = []
        native_checkpoint_receipts: dict[str, dict[str, Any]] = {}
        cell_physical_attribution: dict[str, int] = {}
        cell_stage_telemetry_seconds = {
            cell_id: dict(writer_telemetry) for cell_id in request.cell_ids
        }
        cell_service_times_ns = {cell_id: 0 for cell_id in request.cell_ids}
        cell_cpu_times_ns = {cell_id: 0 for cell_id in request.cell_ids}
        cell_checkpoint_bytes = {cell_id: 0 for cell_id in request.cell_ids}
        raw_cells = receipt.get("cells")
        if not isinstance(raw_cells, Mapping):
            raise CampaignProviderError(
                "cursor aggregate receipt lacks cell physical attribution"
            )
        for cell_id in request.cell_ids:
            cell_receipt = raw_cells.get(cell_id)
            if not isinstance(cell_receipt, Mapping):
                raise CampaignProviderError(
                    f"cursor aggregate receipt lacks cell {cell_id}"
                )
            physical_bytes = cell_receipt.get("physical_bytes")
            if type(physical_bytes) is not int or physical_bytes < 0:
                raise CampaignProviderError(
                    f"cursor aggregate physical attribution is invalid: {cell_id}"
                )
            cell_physical_attribution[cell_id] = physical_bytes
            native_receipt = {
                **dict(receipt),
                **dict(cell_receipt),
                "cell_id": cell_id,
                "path": published,
                "physical_format": "acs100x_native_aggregate_checkpoint_v2",
            }
            native_checkpoint_receipts[cell_id] = native_receipt
            bundle_rows.append(
                {
                    "cell_id": cell_id,
                    "path": published,
                    "source_projection_sha256": source_digests[cell_id],
                    "physical_format": "acs100x_native_aggregate_checkpoint_v2",
                    "artifact_sha256": receipt["artifact_sha256"],
                    "payload_sha256": receipt["payload_sha256"],
                    "seal_sha256": receipt["seal_sha256"],
                    "payload_bytes": receipt["payload_bytes"],
                    "chunk_count": cell_receipt["chunk_count"],
                    "aggregate_cell_id": cell_id,
                    "aggregate_physical_bytes": published_bytes,
                    "aggregate_logical_bytes": receipt["logical_bytes"],
                    "aggregate_zero_fill_logical_bytes": receipt[
                        "zero_fill_logical_bytes"
                    ],
                    "cell_physical_bytes": physical_bytes,
                }
            )
        return {
            "bundle_rows": bundle_rows,
            "terminal_cell_digests": source_digests,
            "mechanics_checkpoint_digests": mechanics_digests,
            "canonical_checkpoint_storage_digests": canonical_digests,
            "writer_service_time_ns": writer_service_time_ns,
            "writer_cpu_time_ns": writer_cpu_time_ns,
            "cell_service_times_ns": cell_service_times_ns,
            "cell_cpu_times_ns": cell_cpu_times_ns,
            "cell_stage_telemetry_seconds": cell_stage_telemetry_seconds,
            "cell_checkpoint_bytes": cell_checkpoint_bytes,
            "cell_physical_bytes_attribution": cell_physical_attribution,
            "checkpoint_bytes": published_bytes,
            "aggregate_checkpoint_bytes": published_bytes,
            "aggregate_checkpoint_logical_bytes": receipt["logical_bytes"],
            "aggregate_checkpoint_zero_fill_logical_bytes": receipt[
                "zero_fill_logical_bytes"
            ],
            "aggregate_checkpoint_receipt": dict(receipt),
            "native_checkpoint_receipts": native_checkpoint_receipts,
            "native_capture_group_receipts": {},
            "checkpoint_bytes_measurement_status": "complete",
            "checkpoint_set_commit": {
                "schema_version": "eiel_checkpoint_set_commit_v1",
                "state": "SEALED",
                "ordered_cell_ids": list(request.cell_ids),
                "writer_concurrency": 1,
                "failure_state": None,
            },
            "checkpoint_projection_cursor_receipt": stream_result.get(
                "checkpoint_projection_cursor_receipt"
            ),
            "bounded_projection_aggregate": stream_result.get(
                "bounded_projection_aggregate"
            ),
        }

    def _persist_aggregate_projection_checkpoint_set_v1(
        self,
        *,
        batch_root: Path,
        request: ResidentBatchRequestV1,
        source_projections: Sequence[Mapping[str, Any]] | None,
        projection_cursor: Any | None = None,
        generation: int,
        tick: int,
        absolute_next_tick: int | None,
        segment_ordinal: int,
        expected_source_projection_digest_sha256s: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist one native physical batch as one indexed checkpoint object."""

        if source_projections is None:
            if projection_cursor is None:
                raise CampaignProviderError(
                    "aggregate checkpoint persistence has no projection source"
                )
            return self._persist_aggregate_projection_checkpoint_cursor_set_v1(
                batch_root=batch_root,
                request=request,
                projection_cursor=projection_cursor,
                generation=generation,
                tick=tick,
                absolute_next_tick=absolute_next_tick,
                segment_ordinal=segment_ordinal,
                expected_source_projection_digest_sha256s=(
                    expected_source_projection_digest_sha256s
                ),
            )
        if projection_cursor is not None:
            raise CampaignProviderError(
                "aggregate checkpoint persistence received two projection sources"
            )

        from engine.native_runtime.checkpoint_digests import (
            DigestLayer,
        )
        from engine.campaigns.acs_native_aggregate_checkpoint_v1 import (
            write_acs_native_aggregate_checkpoint,
        )

        if len(source_projections) != len(request.cell_ids):
            raise CampaignProviderError(
                "ACS aggregate snapshot projection count does not match the batch"
            )
        cuda_source_field_counts = (
            _validate_aggregate_native_production_owner_route_v1(
                request,
                source_projections,
            )
        )
        if expected_source_projection_digest_sha256s is not None and set(
            expected_source_projection_digest_sha256s
        ) != set(request.cell_ids):
            raise CampaignProviderError(
                "ACS immutable snapshot digest authority does not cover the batch"
            )

        started_ns = time.perf_counter_ns()
        started_wall = time.perf_counter()
        started_cpu_ns = time.thread_time_ns()
        source_digest_started = time.perf_counter()
        source_digests: dict[str, str] = {}
        for cell_id, projection in zip(
            request.cell_ids, source_projections, strict=True
        ):
            expected = (
                expected_source_projection_digest_sha256s.get(cell_id)
                if expected_source_projection_digest_sha256s is not None
                else None
            )
            source_digests[cell_id] = (
                expected
                if isinstance(expected, str)
                else self.terminal_projection_digest(projection)
            )
        source_digest_seconds = time.perf_counter() - source_digest_started

        snapshot_sequence = _checkpoint_snapshot_sequence_v1(
            generation=generation,
            absolute_next_tick=absolute_next_tick,
        )
        identities: dict[str, dict[str, Any]] = {}
        observers: dict[str, dict[str, Any]] = {}
        identity_started = time.perf_counter()
        for cell_id, projection in zip(
            request.cell_ids, source_projections, strict=True
        ):
            try:
                from engine.native_runtime.lifecycle_events import (
                    typed_lifecycle_events_from_projection,
                    validated_typed_lifecycle_event_cursor,
                )

                event_cursor = validated_typed_lifecycle_event_cursor(
                    projection,
                    typed_lifecycle_events_from_projection(projection),
                )
            except Exception as exc:
                if self._checkpoint_writer is None:
                    raise CampaignProviderError(
                        f"cannot persist aggregate checkpoint with invalid lifecycle authority: {exc}"
                    ) from exc
                event_cursor = 0
            owner_epoch = max(1, snapshot_sequence, segment_ordinal)
            controller_id = f"{self.provider_id}-{cell_id}"
            source_digest = source_digests[cell_id]
            identities[cell_id] = {
                "run_id": request.batch_id,
                "controller_id": controller_id,
                "owner_epoch": owner_epoch,
                "snapshot_id": (
                    f"{controller_id}:owner:{owner_epoch}:boundary:{snapshot_sequence}"
                ),
                "snapshot_sequence": snapshot_sequence,
                "snapshot_digest_sha256": source_digest,
                "next_generation": generation,
                "tick": tick,
                "phase": "generation_boundary" if tick == 0 else "partial_generation",
                "source_projection_sha256": source_digest,
            }
            observers[cell_id] = self._observer_continuation(
                generation,
                tick=tick,
                event_cursor=event_cursor,
                snapshot_sequence=snapshot_sequence,
                absolute_tick_authority=absolute_next_tick is not None,
            )
        identity_seconds = time.perf_counter() - identity_started

        checkpoint_name = (
            f"checkpoint_tick{absolute_next_tick:012d}.acsaggregate"
            if absolute_next_tick is not None
            else f"checkpoint_gen{generation:06d}.acsaggregate"
        )
        destination = (
            batch_root / "checkpoint_native_aggregate_v1" / checkpoint_name
        )
        writer_telemetry: dict[str, float] = {}
        writer_started = time.perf_counter()
        receipt = write_acs_native_aggregate_checkpoint(
            source_projections,
            destination,
            cell_ids=request.cell_ids,
            generation=generation,
            tick=tick,
            absolute_next_tick=absolute_next_tick,
            batch_id=request.batch_id,
            segment_ordinal=segment_ordinal,
            source_projection_sha256s=source_digests,
            architecture_b_identities=identities,
            observer_continuations=observers,
            require_native_extension=(request.profile_id == PRODUCTION_PROFILE_ID),
            performance_telemetry=writer_telemetry,
        )
        writer_call_seconds = time.perf_counter() - writer_started
        writer_telemetry.update(
            {
                "source_digest_seconds": source_digest_seconds,
                "checkpoint_identity_seconds": identity_seconds,
                "writer_call_seconds": writer_call_seconds,
                "provider_post_digest_seconds": 0.0,
                "outer_total_seconds": time.perf_counter() - started_wall,
                "owner_snapshot_capture_mode": (
                    "producer_boundary_synchronous"
                    if any(cuda_source_field_counts)
                    else "cpu_projection_writer"
                ),
                "cuda_source_field_counts": {
                    cell_id: count
                    for cell_id, count in zip(
                        request.cell_ids,
                        cuda_source_field_counts,
                        strict=True,
                    )
                },
            }
        )
        published = str(receipt["path"])
        published_bytes = self._acs_published_bundle_bytes_v1(published)
        if published_bytes is None:
            raise CampaignProviderError(
                "aggregate native checkpoint publication has no measurable file size"
            )

        terminal_cell_digests = dict(source_digests)
        mechanics_checkpoint_digests: dict[str, str] = {}
        canonical_checkpoint_storage_digests: dict[str, str] = {}
        for cell_id, projection in zip(
            request.cell_ids, source_projections, strict=True
        ):
            mechanics_checkpoint_digests[cell_id] = (
                self._layered_checkpoint_digest_v1(
                    projection,
                    DigestLayer.TRAJECTORY_STATE,
                )
            )
            canonical_checkpoint_storage_digests[cell_id] = (
                self._layered_checkpoint_digest_v1(
                    projection,
                    DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
                )
            )

        bundle_rows: list[dict[str, Any]] = []
        native_checkpoint_receipts: dict[str, dict[str, Any]] = {}
        cell_stage_telemetry_seconds: dict[str, dict[str, float]] = {}
        cell_service_times_ns = {cell_id: 0 for cell_id in request.cell_ids}
        cell_cpu_times_ns = {cell_id: 0 for cell_id in request.cell_ids}
        cell_checkpoint_bytes = {cell_id: 0 for cell_id in request.cell_ids}
        cell_physical_attribution = {
            cell_id: int(receipt["cells"][cell_id]["physical_bytes"])
            for cell_id in request.cell_ids
        }
        for cell_id in request.cell_ids:
            cell_receipt = dict(receipt["cells"][cell_id])
            native_receipt = {
                **receipt,
                **cell_receipt,
                "cell_id": cell_id,
                "path": published,
                "physical_format": "acs100x_native_aggregate_checkpoint_v2",
            }
            native_checkpoint_receipts[cell_id] = native_receipt
            cell_stage_telemetry_seconds[cell_id] = dict(writer_telemetry)
            bundle_rows.append(
                {
                    "cell_id": cell_id,
                    "path": published,
                    "source_projection_sha256": source_digests[cell_id],
                    "physical_format": "acs100x_native_aggregate_checkpoint_v2",
                    "artifact_sha256": receipt["artifact_sha256"],
                    "payload_sha256": receipt["payload_sha256"],
                    "seal_sha256": receipt["seal_sha256"],
                    "payload_bytes": receipt["payload_bytes"],
                    "chunk_count": cell_receipt["chunk_count"],
                    "aggregate_cell_id": cell_id,
                    "aggregate_physical_bytes": published_bytes,
                    "aggregate_logical_bytes": receipt["logical_bytes"],
                    "aggregate_zero_fill_logical_bytes": receipt[
                        "zero_fill_logical_bytes"
                    ],
                    "cell_physical_bytes": cell_receipt["physical_bytes"],
                }
            )

        return {
            "bundle_rows": bundle_rows,
            "terminal_cell_digests": terminal_cell_digests,
            "mechanics_checkpoint_digests": mechanics_checkpoint_digests,
            "canonical_checkpoint_storage_digests": canonical_checkpoint_storage_digests,
            "writer_service_time_ns": time.perf_counter_ns() - started_ns,
            "writer_cpu_time_ns": time.thread_time_ns() - started_cpu_ns,
            "cell_service_times_ns": cell_service_times_ns,
            "cell_cpu_times_ns": cell_cpu_times_ns,
            "cell_stage_telemetry_seconds": cell_stage_telemetry_seconds,
            "cell_checkpoint_bytes": cell_checkpoint_bytes,
            "cell_physical_bytes_attribution": cell_physical_attribution,
            "checkpoint_bytes": published_bytes,
            "aggregate_checkpoint_bytes": published_bytes,
            "aggregate_checkpoint_logical_bytes": receipt["logical_bytes"],
            "aggregate_checkpoint_zero_fill_logical_bytes": receipt[
                "zero_fill_logical_bytes"
            ],
            "aggregate_checkpoint_receipt": receipt,
            "native_checkpoint_receipts": native_checkpoint_receipts,
            "native_capture_group_receipts": {},
            "checkpoint_bytes_measurement_status": "complete",
        }

    def _persist_projection_checkpoint_set_v1(
        self,
        *,
        batch_root: Path,
        request: ResidentBatchRequestV1,
        source_projections: Sequence[Mapping[str, Any]] | None = None,
        projection_cursor: Any | None = None,
        generation: int,
        tick: int,
        absolute_next_tick: int | None,
        segment_ordinal: int,
        expected_source_projection_digest_sha256s: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist one detached multicell snapshot through the existing writer.

        Tick-native callers pass ``projection_cursor``.  That path never
        converts the cursor to a ``Sequence`` and releases each detached cell
        immediately after its immutable bundle and digest authorities are
        durable.  The legacy ``source_projections`` path is retained for
        generation-mode compatibility only.
        """

        if request.native_streamed_checkpoint and len(request.cell_ids) > 1:
            return self._persist_aggregate_projection_checkpoint_set_v1(
                batch_root=batch_root,
                request=request,
                source_projections=source_projections,
                projection_cursor=projection_cursor,
                generation=generation,
                tick=tick,
                absolute_next_tick=absolute_next_tick,
                segment_ordinal=segment_ordinal,
                expected_source_projection_digest_sha256s=(
                    expected_source_projection_digest_sha256s
                ),
            )

        from engine.native_runtime.checkpoint_digests import (
            DigestLayer,
        )

        def layer_digest(projection: Mapping[str, Any], layer: Any) -> str:
            return self._layered_checkpoint_digest_v1(projection, layer)

        started_ns = time.perf_counter_ns()
        started_cpu_ns = time.thread_time_ns()
        bundle_rows: list[dict[str, str]] = []
        terminal_cell_digests: dict[str, str] = {}
        mechanics_checkpoint_digests: dict[str, str] = {}
        canonical_checkpoint_storage_digests: dict[str, str] = {}
        cell_service_times_ns: dict[str, int] = {}
        cell_cpu_times_ns: dict[str, int] = {}
        cell_stage_telemetry_seconds: dict[str, dict[str, float]] = {}
        cell_checkpoint_bytes: dict[str, int] = {}
        cursor_receipt: Any | None = None
        if (source_projections is None) == (projection_cursor is None):
            raise CampaignProviderError(
                "checkpoint persistence requires exactly one projection source"
            )
        native_checkpoint_receipts: dict[str, dict[str, Any]] = {}
        if expected_source_projection_digest_sha256s is not None and set(
            expected_source_projection_digest_sha256s
        ) != set(request.cell_ids):
            raise CampaignProviderError(
                "ACS immutable snapshot digest authority does not cover the batch"
            )
        def persist_cell(cell_id: str, projection: Mapping[str, Any]) -> None:
            cell_started_ns = time.perf_counter_ns()
            cell_started_cpu_ns = time.thread_time_ns()
            writer_stage_telemetry: dict[str, float] = {}
            native_receipt: dict[str, Any] = {}
            checkpoint_name = (
                f"checkpoint_tick{absolute_next_tick:012d}.acsnative"
                if absolute_next_tick is not None
                else f"checkpoint_gen{generation:06d}.acsnative"
            )
            path = (
                batch_root / "checkpoint_native_v1" / cell_id / checkpoint_name
                if request.native_streamed_checkpoint
                else batch_root
                / "checkpoint_bundles_v2"
                / cell_id
                / (
                    f"bundle_tick{absolute_next_tick:012d}"
                    if absolute_next_tick is not None
                    else f"bundle_gen{generation:06d}"
                )
            )
            published = self._write_projection_bundle(
                projection,
                path,
                generation=generation,
                tick=tick,
                absolute_next_tick=absolute_next_tick,
                batch_id=request.batch_id,
                cell_id=cell_id,
                segment_ordinal=segment_ordinal,
                expected_source_projection_sha256=(
                    expected_source_projection_digest_sha256s[cell_id]
                    if expected_source_projection_digest_sha256s is not None
                    else None
                ),
                performance_telemetry=writer_stage_telemetry,
                native_streamed_checkpoint=request.native_streamed_checkpoint,
                require_native_extension=(request.profile_id == PRODUCTION_PROFILE_ID),
                native_receipt=native_receipt,
            )
            if request.native_streamed_checkpoint:
                native_checkpoint_receipts[cell_id] = dict(native_receipt)
            if request.acs_capture_instrumentation:
                published_bytes = self._acs_published_bundle_bytes_v1(published)
                if published_bytes is not None:
                    cell_checkpoint_bytes[cell_id] = published_bytes
            post_digest_started = time.perf_counter()
            source_digest = self.terminal_projection_digest(projection)
            if (
                expected_source_projection_digest_sha256s is not None
                and source_digest != expected_source_projection_digest_sha256s[cell_id]
            ):
                raise CampaignProviderError(
                    "ACS immutable checkpoint snapshot changed during persistence"
                )
            terminal_cell_digests[cell_id] = source_digest
            mechanics_checkpoint_digests[cell_id] = layer_digest(
                projection, DigestLayer.TRAJECTORY_STATE
            )
            canonical_checkpoint_storage_digests[cell_id] = layer_digest(
                projection, DigestLayer.CANONICAL_CHECKPOINT_STORAGE
            )
            writer_stage_telemetry["provider_post_digest_seconds"] = (
                time.perf_counter() - post_digest_started
            )
            bundle_row: dict[str, Any] = {
                "cell_id": cell_id,
                "path": published,
                "source_projection_sha256": source_digest,
            }
            if request.native_streamed_checkpoint:
                bundle_row.update(
                    {
                        "physical_format": (
                            "acs100x_native_streamed_checkpoint_v1"
                        ),
                        "artifact_sha256": native_receipt["artifact_sha256"],
                        "payload_sha256": native_receipt["payload_sha256"],
                        "seal_sha256": native_receipt["seal_sha256"],
                        "payload_bytes": native_receipt["payload_bytes"],
                        "chunk_count": native_receipt["chunk_count"],
                    }
                )
            bundle_rows.append(bundle_row)
            cell_service_times_ns[cell_id] = time.perf_counter_ns() - cell_started_ns
            cell_cpu_times_ns[cell_id] = time.thread_time_ns() - cell_started_cpu_ns
            cell_stage_telemetry_seconds[cell_id] = writer_stage_telemetry

        if projection_cursor is not None:
            cursor_cell_ids = tuple(getattr(projection_cursor, "cell_ids", ()))
            if cursor_cell_ids != tuple(request.cell_ids):
                raise CampaignProviderError(
                    "checkpoint projection cursor cell order differs from request"
                )
            expected_index = 0
            try:
                while True:
                    detached = projection_cursor.next_cell()
                    if detached is None:
                        break
                    if (
                        expected_index >= len(request.cell_ids)
                        or detached.cell_id != request.cell_ids[expected_index]
                        or detached.cell_index != expected_index
                    ):
                        raise CampaignProviderError(
                            "checkpoint projection cursor emitted an unordered cell"
                        )
                    persist_cell(detached.cell_id, detached.require_projection())
                    projection_cursor.release_cell(detached.cell_id)
                    expected_index += 1
                if expected_index != len(request.cell_ids):
                    raise CampaignProviderError(
                        "checkpoint projection cursor ended before all cells"
                    )
                cursor_receipt = projection_cursor.close()
                if getattr(cursor_receipt, "state", None) != "SEALED":
                    raise CampaignProviderError(
                        "checkpoint projection cursor did not seal"
                    )
            except BaseException as exc:
                try:
                    projection_cursor.fail(str(exc))
                except Exception:
                    pass
                raise
        else:
            for cell_id, projection in zip(
                request.cell_ids, source_projections, strict=True
            ):
                persist_cell(cell_id, projection)
        cursor_payload = (
            cursor_receipt.payload()
            if cursor_receipt is not None and hasattr(cursor_receipt, "payload")
            else None
        )
        bounded_projection_aggregate = None
        if projection_cursor is not None:
            aggregate = getattr(projection_cursor, "aggregate_payload", None)
            if not callable(aggregate):
                raise CampaignProviderError(
                    "bounded checkpoint cursor lacks its sealed aggregate"
                )
            bounded_projection_aggregate = aggregate()
        return {
            "bundle_rows": bundle_rows,
            "terminal_cell_digests": terminal_cell_digests,
            "mechanics_checkpoint_digests": mechanics_checkpoint_digests,
            "canonical_checkpoint_storage_digests": (
                canonical_checkpoint_storage_digests
            ),
            "writer_service_time_ns": time.perf_counter_ns() - started_ns,
            "writer_cpu_time_ns": time.thread_time_ns() - started_cpu_ns,
            "cell_service_times_ns": cell_service_times_ns,
            "cell_cpu_times_ns": cell_cpu_times_ns,
            "cell_stage_telemetry_seconds": cell_stage_telemetry_seconds,
            "cell_checkpoint_bytes": cell_checkpoint_bytes,
            "native_checkpoint_receipts": native_checkpoint_receipts,
            "native_capture_group_receipts": {},
            "checkpoint_bytes": sum(cell_checkpoint_bytes.values()),
            "checkpoint_bytes_measurement_status": (
                "complete"
                if len(cell_checkpoint_bytes) == len(request.cell_ids)
                else "unavailable_non_filesystem_writer"
            ),
            "checkpoint_set_commit": {
                "schema_version": "eiel_checkpoint_set_commit_v1",
                "state": "SEALED",
                "ordered_cell_ids": list(request.cell_ids),
                "writer_concurrency": 1,
                "failure_state": None,
            },
            "checkpoint_projection_cursor_receipt": cursor_payload,
            "bounded_projection_aggregate": bounded_projection_aggregate,
        }

    def _checkpoint_bundle_projection_cursor_v1(
        self,
        *,
        batch_root: Path,
        request: ResidentBatchRequestV1,
        bundle_rows: Sequence[Mapping[str, Any]],
    ) -> Any:
        """Open a fresh one-cell cursor over already-published bundles.

        The checkpoint writer owns and seals the settlement cursor.  Consumers
        such as provider-native Chronicle therefore re-open only compact
        bundle descriptors after publication.  Decoding remains ordered and
        one-cell-at-a-time; no terminal projection tuple is constructed.
        """

        if len(bundle_rows) != len(request.cell_ids):
            raise CampaignProviderError(
                "published checkpoint bundles do not cover the physical batch"
            )
        paths: list[str] = []
        digests: dict[str, str] = {}
        for cell_id, row in zip(request.cell_ids, bundle_rows, strict=True):
            if not isinstance(row, Mapping) or row.get("cell_id") != cell_id:
                raise CampaignProviderError(
                    "published checkpoint bundle order differs from request"
                )
            path = row.get("path")
            digest = row.get("source_projection_sha256")
            if type(path) is not str or not path:
                raise CampaignProviderError(
                    "published checkpoint bundle path is missing"
                )
            if not _valid_sha256(digest):
                raise CampaignProviderError(
                    "published checkpoint bundle digest is missing"
                )
            paths.append(path)
            digests[path] = digest
        if len(set(paths)) != len(paths):
            raise CampaignProviderError(
                "published checkpoint bundle paths are duplicated"
            )
        descriptors = _LazyProjectionBundleSequence(
            paths,
            self._read_projection_bundle,
            digests,
        )
        store = _BoundedSourceProjectionStoreV1(
            self,
            root=(batch_root / "chronicle_projection_import_v1"),
            cell_ids=request.cell_ids,
            source_projections=descriptors,
        )
        return store.cursor_factory()()

    def reporting_parity_evidence(
        self, plan: CampaignPlanV1 | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Declare the current exact reporting-parity qualification boundary."""

        del plan
        return {
            "status": "unsupported",
            "evidence_mode": "provider_observational_contract_with_state_mutation_sentinel",
            "observational_contract": "exact_reporting_must_not_mutate_authoritative_state_v1",
            "state_mutation_sentinel": None,
            "reason_code": "reporting_state_mutation_sentinel_unavailable",
            "reason": (
                "the current Patch resident owner exposes exact reporting rows but no "
                "registered reporting-disabled execution or authoritative-state mutation sentinel"
            ),
        }

    def _read_projection_bundle(
        self,
        bundle_path: str,
        *,
        cell_id: str | None = None,
        verify_aggregate_payload: bool = True,
    ) -> Mapping[str, Any]:
        bundle = Path(bundle_path)
        world_set_manifest = bundle / "checkpoint_set_manifest.json"
        if bundle.is_dir() and world_set_manifest.is_file():
            if not isinstance(cell_id, str) or not cell_id:
                raise CampaignProviderError(
                    "world-sharded checkpoint restore requires its world identity"
                )
            from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
                ACSWorldShardedCheckpointError,
                restore_selected_world_projection_from_set_v1,
            )

            try:
                restored = restore_selected_world_projection_from_set_v1(
                    world_set_manifest, world_id=cell_id
                )
            except ACSWorldShardedCheckpointError as exc:
                raise CampaignProviderError(
                    f"strict world-sharded checkpoint restore failed: {exc}"
                ) from exc
            projection = restored.get("projection")
            if not isinstance(projection, Mapping):
                raise CampaignProviderError(
                    "strict world-sharded checkpoint projection is invalid"
                )
            return dict(projection)
        if bundle.is_file() and bundle.suffix == ".acsworld":
            from engine.campaigns.acs_world_sharded_checkpoint_v1 import (
                ACSWorldShardedCheckpointError,
                restore_world_checkpoint_projection_v1,
            )

            try:
                restored = restore_world_checkpoint_projection_v1(bundle_path)
            except ACSWorldShardedCheckpointError as exc:
                raise CampaignProviderError(
                    f"strict world-sharded checkpoint restore failed: {exc}"
                ) from exc
            projection = restored.get("projection")
            if not isinstance(projection, Mapping):
                raise CampaignProviderError(
                    "strict world-sharded checkpoint projection is invalid"
                )
            return dict(projection)
        if bundle.is_file() and bundle.suffix == ".acsaggregate":
            from engine.campaigns.acs_native_aggregate_checkpoint_v1 import (
                ACSNativeAggregateCheckpointError,
                load_acs_native_aggregate_checkpoint,
            )

            try:
                packed = load_acs_native_aggregate_checkpoint(
                    bundle_path,
                    cell_id=cell_id,
                    verify_payload=verify_aggregate_payload,
                )
            except ACSNativeAggregateCheckpointError as exc:
                raise CampaignProviderError(
                    f"strict native aggregate checkpoint load failed: {exc}"
                ) from exc
            projection = packed.get("projection")
            if not isinstance(projection, Mapping):
                raise CampaignProviderError(
                    "strict native aggregate checkpoint projection is invalid"
                )
            normalized = dict(projection)
            actual_digest = self.terminal_projection_digest(normalized)
            expected_digest = packed.get("header", {}).get("cells", ())
            expected_for_cell = next(
                (
                    row.get("source_projection_sha256")
                    for row in expected_digest
                    if isinstance(row, Mapping)
                    and row.get("cell_id") == packed.get("cell_id")
                ),
                None,
            )
            if actual_digest != expected_for_cell:
                raise CampaignProviderError(
                    "native aggregate checkpoint projection does not match its source digest"
                )
            return normalized
        if bundle.is_file() and bundle.suffix == ".acsnative":
            from engine.campaigns.acs_native_streamed_checkpoint_v1 import (
                ACSNativeCheckpointError,
                load_acs_native_streamed_checkpoint,
            )

            try:
                packed = load_acs_native_streamed_checkpoint(bundle_path)
            except ACSNativeCheckpointError as exc:
                raise CampaignProviderError(
                    f"strict native streamed checkpoint load failed: {exc}"
                ) from exc
            projection = packed.get("projection")
            footer = packed.get("footer")
            if not isinstance(projection, Mapping) or not isinstance(footer, Mapping):
                raise CampaignProviderError(
                    "strict native streamed checkpoint envelope is invalid"
                )
            normalized = dict(projection)
            actual_digest = self.terminal_projection_digest(normalized)
            if actual_digest != footer.get("source_projection_sha256"):
                raise CampaignProviderError(
                    "native streamed checkpoint projection does not match its source digest"
                )
            return normalized
        loader = self._checkpoint_loader
        decoder = self._checkpoint_decoder
        if loader is None:
            from engine.simulation.checkpoint_bundle_v2 import (
                ARCHITECTURE_B_BUNDLE_KIND,
                checkpoint_bundle_v2_load,
            )

            payload = checkpoint_bundle_v2_load(
                bundle_path,
                strict=True,
                required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
            )
        else:
            payload = loader(
                bundle_path, strict=True, required_bundle_kind="architecture_b"
            )
        if decoder is None:
            from engine.simulation.checkpoint_bundle_v2 import (
                decode_architecture_b_resume_envelope,
            )

            envelope = decode_architecture_b_resume_envelope(payload)
            checkpoint_projection = envelope.get("checkpoint_projection")
            import_metadata = envelope.get("controller_import_metadata")
            if not isinstance(checkpoint_projection, Mapping) or not isinstance(
                import_metadata, Mapping
            ):
                raise CampaignProviderError(
                    "strict checkpoint resume envelope is invalid"
                )
            projection = dict(checkpoint_projection)
            projection["metadata"] = dict(import_metadata)
        else:
            sidecar = payload.get("native_runtime_checkpoint_projection")
            if not isinstance(sidecar, Mapping):
                raise CampaignProviderError(
                    "strict checkpoint bundle has no native projection"
                )
            projection = decoder(sidecar)
        if not isinstance(projection, Mapping):
            raise CampaignProviderError("decoded checkpoint projection is invalid")
        # Bundle/controller identity and observer cursors describe the durable
        # transport route.  They are verified by Bundle v2, but they are not
        # mechanics state and must not contaminate a fresh owner's source
        # projection or its terminal digest.
        normalized = dict(projection)
        metadata = normalized.get("metadata")
        if isinstance(metadata, Mapping):
            mechanics_metadata = dict(metadata)
            mechanics_metadata.pop("architecture_b_identity", None)
            mechanics_metadata.pop("observer_continuation", None)
            normalized["metadata"] = mechanics_metadata
        if decoder is None:
            expected_digest = envelope.get("source_projection_sha256")
            actual_digest = self.terminal_projection_digest(normalized)
            if expected_digest != actual_digest:
                raise CampaignProviderError(
                    "decoded checkpoint mechanics projection does not match the "
                    "verified Bundle v2 source digest"
                )
        return normalized

    @staticmethod
    def _write_component_checkpoint(
        value: Mapping[str, Any], destination: Path
    ) -> dict[str, str]:
        """Persist a domain sidecar through the engine checkpoint owner."""

        import os
        from uuid import uuid4

        import torch

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        try:
            torch.save(dict(value), temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        return {"path": str(destination), "sha256": digest}

    @staticmethod
    def _read_component_checkpoint(
        descriptor: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import torch

        path = Path(str(descriptor.get("path") or "")).resolve()
        digest = descriptor.get("sha256")
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise CampaignProviderError(
                "engine component checkpoint descriptor is stale"
            )
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, Mapping):
            raise CampaignProviderError(
                "engine component checkpoint payload is malformed"
            )
        return dict(value)

    @staticmethod
    def _prepare_resident_segment_owner_v1(
        *,
        plan: Mapping[str, Any],
        runner_builder: Callable[..., Any],
        request_builder: Callable[..., Any],
        preparer: Callable[..., Any],
        cells: Sequence[Mapping[str, Any]],
        cell_ids: Sequence[str],
        batch_root: Path,
        start_generation: int | None = None,
        end_generation: int | None = None,
        ticks_per_generation: int | None = None,
        owner_start_tick: int | None = None,
        owner_end_tick: int | None = None,
        artifact: str,
        qualification: str,
        reporting: str,
        module: Any,
        source_projections: Sequence[Mapping[str, Any]] | None,
        allocation_generations: int | None = None,
        genome_v2_checkpoint_authority_binding: Mapping[str, str] | None = None,
        tick_native_control: Mapping[str, Any] | None = None,
        allow_short_technical_gate_generation_count: bool = False,
        allow_partial_generation_source_cursor: bool = False,
        physical_window_generations: int | None = None,
        hapb_policy_tile_rows: int | None = None,
        provider: Any | None = None,
        enable_streamed_preparation: bool = False,
        direct_batch_host_import_enable: bool = False,
        indexed_source_capture: bool = False,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
        """Construct one dispatch-ready owner through the production path."""

        execution = plan.get("execution") if isinstance(plan, Mapping) else None
        if tick_native_control is None and isinstance(execution, Mapping):
            if execution.get("execution_route_identity") == (
                "tick_native_execution_route_v1"
            ):
                raw_control = execution.get("tick_native_control")
                if not isinstance(raw_control, Mapping):
                    raise CampaignProviderError(
                        "tick-native plan is missing tick_native_control"
                    )
                tick_native_control = dict(raw_control)
        if tick_native_control is not None:
            if tick_native_control.get("time_authority") != "absolute_tick_v1":
                raise CampaignProviderError(
                    "tick-native owner control must use absolute_tick_v1"
                )
            if any(
                name in tick_native_control
                for name in (
                    "generations",
                    "generation_count",
                    "logical_generation_count",
                    "physical_window_generations",
                    "ticks_per_generation",
                )
            ):
                raise CampaignProviderError(
                    "tick-native owner control contains generation authority"
                )
            if any(
                value is not None
                for value in (start_generation, end_generation, ticks_per_generation)
            ):
                raise CampaignProviderError(
                    "tick-native owner preparation cannot receive generation coordinates"
                )
            if type(owner_start_tick) is not int or type(owner_end_tick) is not int:
                raise CampaignProviderError(
                    "tick-native owner preparation requires absolute owner ticks"
                )
            private_owner = _tick_native_private_owner_adapter_v1(
                tick_native_control,
                owner_start_tick=owner_start_tick,
                owner_end_tick=owner_end_tick,
            )
            if (
                allocation_generations is not None
                or physical_window_generations is not None
            ):
                raise CampaignProviderError(
                    "tick-native owner preparation cannot receive private "
                    "generation-shaped ABI dimensions"
                )
            segment_generations = private_owner.private_source_window_count
            allocation_generations = (
                private_owner.private_agent_id_allocation_generation_count
            )
            physical_window_generations = (
                private_owner.private_physical_window_generation_count
            )
            runner_directory_name = private_owner.runner_directory_name
            runner_time_kwargs = {
                "tick_native_control": dict(tick_native_control),
            }
        else:
            if owner_start_tick is not None or owner_end_tick is not None:
                raise CampaignProviderError(
                    "legacy owner preparation cannot receive absolute owner ticks"
                )
            if (
                type(start_generation) is not int
                or type(end_generation) is not int
                or end_generation <= start_generation
                or type(ticks_per_generation) is not int
                or ticks_per_generation <= 0
            ):
                raise CampaignProviderError(
                    "legacy owner preparation requires valid generation coordinates"
                )
            segment_generations = end_generation - start_generation
            if (
                type(allocation_generations) is not int
                or allocation_generations <= 0
            ):
                raise CampaignProviderError(
                    "legacy owner preparation requires an allocation generation count"
                )
            if physical_window_generations is None:
                physical_window_generations = 2
            runner_directory_name = (
                f"segment-{start_generation:06d}-{end_generation:06d}"
            )
            runner_time_kwargs = {
                "generations": segment_generations,
                "ticks_per_generation": ticks_per_generation,
            }
        if type(
            physical_window_generations
        ) is not int or physical_window_generations not in {1, 2}:
            raise CampaignProviderError(
                "physical_window_generations must be exact integer 1 or 2"
            )
        if hapb_policy_tile_rows is not None and (
            type(hapb_policy_tile_rows) is not int
            or hapb_policy_tile_rows not in {512, 4096}
        ):
            raise CampaignProviderError(
                "hapb_policy_tile_rows must be exactly 512 or 4096"
            )
        if type(direct_batch_host_import_enable) is not bool:
            raise CampaignProviderError(
                "direct_batch_host_import_enable must be an exact bool"
            )
        source_capability_kwargs = (
            {
                "source_capability_mask": {
                    "flat": False,
                    "patch": True,
                    "churn": True,
                    "gru": False,
                    "genome_v2": True,
                }
            }
            if genome_v2_checkpoint_authority_binding
            else {}
        )

        streamed_preparation = (
            enable_streamed_preparation
            and
            tick_native_control is not None
            and _callable_accepts_keyword_v1(preparer, "per_cell_request_factory")
        )
        if streamed_preparation:
            if provider is None:
                raise CampaignProviderError(
                    "streamed preparation requires its provider checkpoint owner"
                )
            if source_projections is not None and not isinstance(
                source_projections, Sequence
            ):
                raise CampaignProviderError(
                    "streamed source continuation is not a durable descriptor sequence"
                )
            source_store = _BoundedSourceProjectionStoreV1(
                provider,
                root=batch_root,
                cell_ids=cell_ids,
                source_projections=source_projections,
                indexed_source_capture=indexed_source_capture,
            )
            source_cursor = (
                source_store.cursor_factory()()
                if source_store.has_existing_sources
                else None
            )
            source_import_receipt: dict[str, Any] = {}
            stream_builds: list[Any] = []
            runner_builder_parameters = inspect.signature(runner_builder).parameters
            runner_builder_accepts_source_initialization = (
                "initialize_fresh_source" in runner_builder_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in runner_builder_parameters.values()
                )
            )
            request_builder_parameters = inspect.signature(request_builder).parameters
            request_builder_accepts_source_callbacks = (
                "source_import_release" in request_builder_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in request_builder_parameters.values()
                )
            )
            if source_cursor is not None and not request_builder_accepts_source_callbacks:
                raise CampaignProviderError(
                    "streamed source import requires request-builder release callbacks"
                )
            request_builder_accepts_event_sink = (
                "event_history_sink_path" in request_builder_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in request_builder_parameters.values()
                )
            )
            request_builder_accepts_tick_control = (
                "tick_native_control" in request_builder_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in request_builder_parameters.values()
                )
            )
            event_history_sink_path = (
                (batch_root / "event_history_v1" / "events.bin")
                .expanduser()
                .resolve()
            )
            def streamed_request_factory(index: int) -> Any:
                if type(index) is not int or not 0 <= index < len(cell_ids):
                    raise CampaignProviderError(
                        "streamed request factory received an invalid cell index"
                    )
                cell = cells[index]
                cell_id = cell_ids[index]
                detached_source = None
                source_projection = None

                def fail_source(reason: str) -> None:
                    if source_cursor is not None:
                        source_cursor.fail(reason)

                try:
                    if source_cursor is not None:
                        detached_source = source_cursor.next_cell()
                        if detached_source is None:
                            raise CampaignProviderError(
                                "streamed source cursor ended before request preparation"
                            )
                        if (
                            detached_source.cell_id != cell_id
                            or detached_source.cell_index != index
                        ):
                            raise CampaignProviderError(
                                "streamed source cursor emitted unordered request input"
                            )
                        source_projection = detached_source.require_payload()
                except BaseException as exc:
                    fail_source(str(exc))
                    raise

                try:
                    build = runner_builder(
                        cell,
                        batch_root
                        / "cells"
                        / cell_id
                        / runner_directory_name,
                        **runner_time_kwargs,
                        artifact_path=artifact,
                        qualification_receipt_path=qualification,
                        reporting_receipt_path=reporting,
                        **(
                            {
                                "genome_v2_checkpoint_authority_binding": dict(
                                    genome_v2_checkpoint_authority_binding
                                )
                            }
                            if genome_v2_checkpoint_authority_binding
                            else {}
                        ),
                        **(
                            {"initialize_fresh_source": False}
                            if source_projection is not None
                            and runner_builder_accepts_source_initialization
                            else {}
                        ),
                    )
                    receipt = getattr(build, "receipt", None)
                    if not isinstance(receipt, Mapping):
                        raise CampaignProviderError(
                            f"streamed runner build {cell_id} lacks a compact receipt"
                        )
                    stream_builds.append(SimpleNamespace(receipt=dict(receipt)))
                except BaseException as exc:
                    fail_source(str(exc))
                    raise

                def release_source(direct_request: Any) -> None:
                    if source_cursor is None or detached_source is None:
                        return
                    imported_digest = getattr(
                        direct_request,
                        "source_import_projection_digest_sha256",
                        None,
                    )
                    if (
                        isinstance(imported_digest, str)
                        and imported_digest != detached_source.source_projection_sha256
                    ):
                        source_cursor.fail(
                            f"streamed source projection digest was substituted for {cell_id}"
                        )
                        raise CampaignProviderError(
                            f"streamed source projection digest was substituted for {cell_id}"
                        )
                    source_cursor.import_and_release(
                        cell_id, direct_request.cuda_boundary
                    )
                    if index == len(cell_ids) - 1:
                        receipt_value = source_cursor.close()
                        source_import_receipt.update(receipt_value.payload())

                request_kwargs: dict[str, Any] = {
                    "qualified_cuda_module": module,
                    "logical_generation_count": segment_generations,
                    "physical_window_generations": physical_window_generations,
                    "agent_id_allocation_generation_count": allocation_generations,
                    "source_checkpoint_projection": source_projection,
                    **(
                        {"allow_short_technical_gate_generation_count": True}
                        if allow_short_technical_gate_generation_count
                        else {}
                    ),
                    **(
                        {"allow_partial_generation_source_cursor": True}
                        if allow_partial_generation_source_cursor
                        else {}
                    ),
                    **(
                        {
                            "event_history_sink_path": str(event_history_sink_path),
                            "event_history_sink_cell_index": index,
                        }
                        if request_builder_accepts_event_sink
                        else {}
                    ),
                    **(
                        {"tick_native_control": dict(tick_native_control)}
                        if request_builder_accepts_tick_control
                        else {}
                    ),
                    **source_capability_kwargs,
                }
                if source_cursor is not None:
                    request_kwargs.update(
                        {
                            "source_import_release": release_source,
                            "source_import_fail": fail_source,
                        }
                    )
                try:
                    return request_builder(build.runner, **request_kwargs)
                except BaseException as exc:
                    fail_source(str(exc))
                    raise

            prepared = preparer(
                cell_ids=tuple(cell_ids),
                per_cell_request_factory=streamed_request_factory,
                logical_generation_count=segment_generations,
                physical_window_generations=physical_window_generations,
                agent_id_allocation_generation_count=allocation_generations,
                resume_projections=None,
                source_projection_cursor_factory=source_store.cursor_factory(),
                source_import_cursor_receipt=source_import_receipt,
                source_projection_capture=source_store.capture,
                **(
                    {"allow_short_technical_gate_generation_count": True}
                    if allow_short_technical_gate_generation_count
                    else {}
                ),
                **(
                    {"allow_partial_generation_source_cursor": True}
                    if allow_partial_generation_source_cursor
                    else {}
                ),
                **(
                    {"hapb_policy_tile_rows": hapb_policy_tile_rows}
                    if hapb_policy_tile_rows is not None
                    else {}
                ),
                **(
                    {"tick_native_control": dict(tick_native_control)}
                    if _callable_accepts_keyword_v1(preparer, "tick_native_control")
                    else {}
                ),
            )
            if source_cursor is not None and (
                source_import_receipt.get("state") != "SEALED"
                or source_import_receipt.get("ordered_cell_ids")
                != list(cell_ids)
                or source_import_receipt.get("covered_cell_ids")
                != list(cell_ids)
                or source_import_receipt.get(
                    "max_live_tensor_bearing_payloads"
                )
                != 1
                or source_import_receipt.get("importer_concurrency") != 1
            ):
                raise CampaignProviderError(
                    "streamed successor owner returned before its bounded source cursor sealed"
                )
            request_descriptors = tuple(
                getattr(prepared, "request_descriptors", ())
            )
            if not request_descriptors:
                raise CampaignProviderError(
                    "streamed native preparer returned no compact request descriptors"
                )
            if len(request_descriptors) != len(cell_ids):
                raise CampaignProviderError(
                    "streamed native preparer request descriptors do not cover B"
                )
            return tuple(stream_builds), request_descriptors, prepared

        builds = tuple(
            runner_builder(
                cell,
                batch_root
                / "cells"
                / cell_id
                / runner_directory_name,
                **runner_time_kwargs,
                artifact_path=artifact,
                qualification_receipt_path=qualification,
                reporting_receipt_path=reporting,
                **(
                    {
                        "genome_v2_checkpoint_authority_binding": dict(
                            genome_v2_checkpoint_authority_binding
                        )
                    }
                    if genome_v2_checkpoint_authority_binding
                    else {}
                ),
            )
            for cell, cell_id in zip(cells, cell_ids, strict=True)
        )
        source_capability_kwargs = (
            {
                "source_capability_mask": {
                    "flat": False,
                    "patch": True,
                    "churn": True,
                    "gru": False,
                    "genome_v2": True,
                }
            }
            if genome_v2_checkpoint_authority_binding
            else {}
        )

        # Lifecycle rows are a streamed output of the resident owner.  Keep a
        # single durable sink for the physical batch so the native controller
        # can append by cell without ever rebuilding a batch-wide history in
        # Python memory.  A resumed projection may already bind an existing
        # sink; preserve that authority instead of silently creating a fork.
        event_history_sink_path = (
            (batch_root / "event_history_v1" / "events.bin").expanduser().resolve()
        )
        referenced_sink_paths: set[str] = set()
        if source_projections is not None:
            for projection in source_projections:
                metadata = (
                    projection.get("metadata", {})
                    if isinstance(projection, Mapping)
                    else {}
                )
                reference = (
                    metadata.get("event_history_reference")
                    if isinstance(metadata, Mapping)
                    else None
                )
                if isinstance(reference, Mapping) and reference.get("path"):
                    referenced_sink_paths.add(
                        str(Path(str(reference["path"])).expanduser().resolve())
                    )
        if len(referenced_sink_paths) > 1:
            raise CampaignProviderError(
                "resident batch source projections bind multiple event sinks"
            )
        if referenced_sink_paths:
            event_history_sink_path = Path(next(iter(referenced_sink_paths)))
        try:
            request_builder_parameters = inspect.signature(request_builder).parameters
            request_builder_accepts_event_sink = (
                "event_history_sink_path" in request_builder_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in request_builder_parameters.values()
                )
            )
        except (TypeError, ValueError):
            request_builder_accepts_event_sink = False
        _require_tick_native_event_sink_builder_v1(
            tick_native=tick_native_control is not None,
            accepts_event_sink=request_builder_accepts_event_sink,
        )
        try:
            request_builder_parameters = inspect.signature(request_builder).parameters
            request_builder_accepts_tick_control = (
                "tick_native_control" in request_builder_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in request_builder_parameters.values()
                )
            )
        except (TypeError, ValueError):
            request_builder_accepts_tick_control = False
        try:
            request_builder_parameters = inspect.signature(request_builder).parameters
            request_builder_accepts_direct_batch_host_import = (
                "direct_batch_host_import_enable" in request_builder_parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in request_builder_parameters.values()
                )
            )
        except (TypeError, ValueError):
            request_builder_accepts_direct_batch_host_import = False
        if (
            direct_batch_host_import_enable
            and not request_builder_accepts_direct_batch_host_import
        ):
            raise CampaignProviderError(
                "resident request builder lacks the required direct batch host-import seam"
            )

        # A disk-backed continuation is intentionally lazy, but a single owner
        # preparation may call the request builder twice while reconciling a
        # batch event-capacity override.  Cache each compact source projection
        # for this preparation so that reconciliation does not reread the
        # durable bundle or create duplicate host deserializations.
        source_projection_cache: dict[int, Mapping[str, Any]] = {}

        def source_projection_for_cell(index: int) -> Mapping[str, Any] | None:
            if source_projections is None:
                return None
            if index not in source_projection_cache:
                source_projection_cache[index] = source_projections[index]
            return source_projection_cache[index]

        def build_requests(
            *, event_capacity_override: int | None = None
        ) -> tuple[Any, ...]:
            return tuple(
                request_builder(
                    build.runner,
                    qualified_cuda_module=module,
                    logical_generation_count=segment_generations,
                    physical_window_generations=physical_window_generations,
                    agent_id_allocation_generation_count=allocation_generations,
                    source_checkpoint_projection=(source_projection_for_cell(index)),
                    **(
                        {"event_capacity_override": event_capacity_override}
                        if event_capacity_override is not None
                        else {}
                    ),
                    **(
                        {"allow_short_technical_gate_generation_count": True}
                        if allow_short_technical_gate_generation_count
                        else {}
                    ),
                    **(
                        {"allow_partial_generation_source_cursor": True}
                        if allow_partial_generation_source_cursor
                        else {}
                    ),
                    **(
                        {
                            "event_history_sink_path": str(event_history_sink_path),
                            "event_history_sink_cell_index": index,
                        }
                        if request_builder_accepts_event_sink
                        else {}
                    ),
                    **(
                        {"tick_native_control": dict(tick_native_control)}
                        if tick_native_control is not None
                        and request_builder_accepts_tick_control
                        else {}
                    ),
                    **(
                        {"direct_batch_host_import_enable": True}
                        if direct_batch_host_import_enable
                        and request_builder_accepts_direct_batch_host_import
                        else {}
                    ),
                    **source_capability_kwargs,
                )
                for index, build in enumerate(builds)
            )

        requests = build_requests()
        if tick_native_control is not None:
            expected_control = dict(tick_native_control)
            for index, request in enumerate(requests):
                actual_control = getattr(request, "tick_native_control", None)
                if not isinstance(actual_control, Mapping):
                    raise CampaignProviderError(
                        f"resident request {index} lost tick_native_control"
                    )
                if dict(actual_control) != expected_control:
                    raise CampaignProviderError(
                        f"resident request {index} changed tick_native_control"
                    )
        event_capacities = tuple(
            getattr(request, "event_capacity", None) for request in requests
        )
        if event_capacities and all(type(value) is int for value in event_capacities):
            max_event_capacity = max(event_capacities)
            if any(value != max_event_capacity for value in event_capacities):
                requests = build_requests(event_capacity_override=max_event_capacity)
                rebuilt_event_capacities = tuple(
                    getattr(request, "event_capacity", None) for request in requests
                )
                if any(
                    value != max_event_capacity for value in rebuilt_event_capacities
                ):
                    raise CampaignProviderError(
                        "resident segment request builder did not honor the "
                        "batch event-capacity override"
                    )
        gc.collect()
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        resume_projections_for_preparer: tuple[Any, ...] | None = None
        if source_projections is not None:
            bound_projections: list[Any] = []
            for cell, prepared_request in enumerate(requests):
                # Integrated WDL/Jungle adapters intentionally return a thin
                # request wrapper and construct the direct allocation request
                # inside the preparer.  The generic Patch adapter returns the
                # direct request itself.  Preserve the exact source object for
                # either public request shape; assuming every adapter exposes
                # import_projection breaks the integrated multicell route.
                projection = getattr(prepared_request, "import_projection", None)
                if projection is None:
                    projection = getattr(
                        prepared_request, "source_checkpoint_projection", None
                    )
                if projection is None:
                    raise CampaignProviderError(
                        "resident request lacks its bound source projection "
                        f"for cell {cell}"
                    )
                bound_projections.append(projection)
            resume_projections_for_preparer = tuple(bound_projections)
        prepared = preparer(
            cell_ids=tuple(cell_ids),
            per_cell_requests=requests,
            logical_generation_count=segment_generations,
            physical_window_generations=physical_window_generations,
            agent_id_allocation_generation_count=allocation_generations,
            # Request builders own the one-cell import.  For resumed bundles
            # they replace the decoded transport mapping with its compact
            # canonical view, so the preparer must validate that view rather
            # than materialize the lazy bundle sequence a second time.
            resume_projections=resume_projections_for_preparer,
            **(
                {"allow_short_technical_gate_generation_count": True}
                if allow_short_technical_gate_generation_count
                else {}
            ),
            **(
                {"allow_partial_generation_source_cursor": True}
                if allow_partial_generation_source_cursor
                else {}
            ),
            **(
                {"hapb_policy_tile_rows": hapb_policy_tile_rows}
                if hapb_policy_tile_rows is not None
                else {}
            ),
            **(
                {"tick_native_control": dict(tick_native_control)}
                if tick_native_control is not None
                and _callable_accepts_keyword_v1(preparer, "tick_native_control")
                else {}
            ),
        )
        return builds, requests, prepared

    def _generation_zero_checkpoint_projections_v1(
        self,
        prepared: Any,
        *,
        projector: Callable[..., Any] | None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return initialized resident G0 projections for durable publication."""

        if callable(projector):
            projections = tuple(projector(prepared))
        elif self._preparer is not None:
            projections = tuple(getattr(prepared, "source_projections", ()))
        else:
            raise CampaignProviderError(
                "native mechanics binding lacks a G0 checkpoint projector"
            )
        if not projections:
            raise CampaignProviderError(
                "prepared owner lacks ordered G0 checkpoint projections"
            )
        return projections

    def measure_prepared_owner_resources_v1(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        *,
        audit_root: Path,
        audit_id: str,
        measurement_authorization: Mapping[str, Any],
        cuda_memory_detector: Callable[[], tuple[int, int]] | None = None,
        cuda_hardware_detector: Callable[[], Mapping[str, Any]] | None = None,
        runtime_environment_detector: (
            Callable[[], Mapping[str, str | None]] | None
        ) = None,
        stage_sampler: Callable[[str], Any] | None = None,
        projection_serializer: Callable[[Any, Path], None] | None = None,
        resource_releaser: Callable[[Any, Any], None] | None = None,
    ) -> dict[str, Any]:
        """Prepare one owner and record structural zero-dispatch counters."""

        from engine.campaigns.canonical import canonical_sha256
        from engine.campaigns.orchestration import atomic_write_json
        from engine.campaigns.resident_prepared_owner_resource_audit_v1 import (
            MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION,
            ResidentPreparedOwnerResourceAuditRequestV1,
            cuda_memory_stage_sampler_v1,
            detect_cuda_hardware_identity_v1,
            execute_prepared_owner_resource_audit_v1,
            prepared_owner_configuration_sha256_v1,
            validate_cuda_hardware_identity_v1,
        )
        from engine.campaigns.preflight import detect_cuda_memory_v1
        from engine.campaigns.qualification import (
            execution_route_qualification_sidecar_path_v1,
            execution_route_qualification_status_v1,
            read_execution_route_qualification_sidecar_v1,
        )
        from engine.campaigns.resources import (
            minimum_residual_headroom_bytes_v1,
            required_runtime_environment_v1,
        )
        from engine.campaigns.patch_runner_factory import build_patch_fixed_runner_v1
        from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
            prepare_direct_resident_patch_multicell_v1,
        )
        from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
            build_direct_resident_source_allocation_request_v1,
        )

        payload = _plan_payload(plan)
        physical_window_generations = _physical_window_generations_from_plan_v1(payload)
        if request.profile_id != self.profile_id:
            raise CampaignProviderError("prepared-owner audit profile binding mismatch")
        authority = getattr(self, "_require_execution_authority", None)
        if callable(authority):
            authority(payload, request)
        contract = self.execution_route_contract(payload, request)
        route_sha256 = contract.execution_route_sha256
        if (
            not isinstance(request.execution_route_sha256, str)
            or request.execution_route_sha256 != route_sha256
        ):
            raise CampaignProviderError(
                "prepared-owner audit execution-route binding mismatch"
            )
        artifact, qualification, reporting = self._artifact_paths(payload)
        resolver = getattr(self, "_resolved_execution_adapters", None)
        resolved = dict(resolver(payload)) if callable(resolver) else {}
        runner_builder = resolved.get("runner_builder") or (
            self._runner_builder or build_patch_fixed_runner_v1
        )
        request_builder = resolved.get("request_builder") or (
            self._request_builder or build_direct_resident_source_allocation_request_v1
        )
        preparer = resolved.get("preparer") or (
            self._preparer or prepare_direct_resident_patch_multicell_v1
        )
        source_preparer = (
            resolved.get("source_preparer", self._source_preparer)
            or self._prepare_generation_zero_sources_v2
        )
        artifact_loader = resolved.get("artifact_loader", self._artifact_loader)
        mechanics_binding = resolved.get("mechanics_binding")
        source_preparer_kwargs: dict[str, Any] = {}
        mechanics_binding_id = getattr(mechanics_binding, "binding_id", None)
        if mechanics_binding_id in {
            "wdl_fixed_mlp_native_binding_v1",
            "wdl_jungle_eed_fixed_mlp_native_binding_v1",
        }:
            from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
                _wdl_source_capabilities_v2,
            )

            genome_v2_enabled = any(
                name == "genome_upgrade_v2.enable"
                and any(value is True for value in values)
                for name, values in request.parameter_arrays
            )
            source_preparer_kwargs["source_capability_mask"] = (
                _wdl_source_capabilities_v2(
                    {"genome_v2": True} if genome_v2_enabled else None,
                    jungle_overlay=(
                        mechanics_binding_id
                        == "wdl_jungle_eed_fixed_mlp_native_binding_v1"
                    ),
                    eed_overlay=(
                        mechanics_binding_id
                        == "wdl_jungle_eed_fixed_mlp_native_binding_v1"
                    ),
                )
            )
        cells = self._batch_cells(payload, request)
        segment_ends = self._segment_ends(request, start=0)
        if not segment_ends:
            raise CampaignProviderError(
                "prepared-owner audit has no physical segment to prepare"
            )
        first_end = segment_ends[0]
        allocation_generations = self._agent_id_allocation_generation_count(
            request, start_generation=0
        )
        root = Path(audit_root).expanduser().resolve()
        holder: dict[str, Any] = {}
        context = SimpleNamespace(
            run_root=root,
            batch_root=root,
            progress_events=[],
        )

        def load_module() -> Any:
            loaded = self.load_qualified_artifact(
                artifact,
                qualification,
                reporting,
                loader=artifact_loader,
            )
            candidate_authority = getattr(loaded, "candidate_authority_sha256", None)
            if (
                not isinstance(candidate_authority, str)
                or len(candidate_authority) != 64
            ):
                raise CampaignProviderError(
                    "provider/artifact does not expose prepared-owner audit authority"
                )
            holder["loaded"] = loaded
            holder["candidate_authority_sha256"] = candidate_authority
            return getattr(loaded, "module", loaded)

        def prepare_owner(module: Any) -> Any:
            sources = tuple(
                source_preparer(
                    payload,
                    request,
                    context,
                    runner_builder=runner_builder,
                    cells=cells,
                    artifact=artifact,
                    qualification=qualification,
                    reporting=reporting,
                    segment_generations=request.generations,
                    **source_preparer_kwargs,
                )
            )
            try:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _, _, prepared = self._prepare_resident_segment_owner_v1(
                plan=payload,
                runner_builder=runner_builder,
                request_builder=request_builder,
                preparer=preparer,
                cells=cells,
                cell_ids=request.cell_ids,
                batch_root=root,
                start_generation=0,
                end_generation=first_end,
                ticks_per_generation=request.ticks_per_generation,
                artifact=artifact,
                qualification=qualification,
                reporting=reporting,
                module=module,
                source_projections=sources,
                allocation_generations=allocation_generations,
                genome_v2_checkpoint_authority_binding=dict(
                    request.genome_v2_checkpoint_authority_binding
                ),
                physical_window_generations=physical_window_generations,
                provider=self,
            )
            return prepared

        def export_projection(owner: Any) -> tuple[Any, str]:
            projections = tuple(getattr(owner, "source_projections", ()))
            if len(projections) != len(request.cell_ids):
                raise CampaignProviderError(
                    "prepared owner lacks ordered source projections"
                )
            digests = [
                self.terminal_projection_digest(projection)
                for projection in projections
            ]
            return projections, canonical_sha256(
                {
                    "cell_ids": list(request.cell_ids),
                    "source_projection_sha256s": digests,
                }
            )

        parameters = {name: values for name, values in request.parameter_arrays}

        def uniform_integer(name: str) -> int:
            values = parameters.get(name)
            if (
                not isinstance(values, tuple)
                or not values
                or any(type(value) is not int or value < 1 for value in values)
                or len(set(values)) != 1
            ):
                raise CampaignProviderError(
                    f"prepared-owner audit requires one exact {name}"
                )
            return int(values[0])

        artifacts = payload.get("artifacts")
        execution = payload.get("execution")
        if not isinstance(artifacts, Mapping) or not isinstance(execution, Mapping):
            raise CampaignProviderError(
                "prepared-owner audit plan bindings are incomplete"
            )
        artifact_sha256s = {
            str(name): str(row.get("sha256"))
            for name, row in artifacts.items()
            if isinstance(row, Mapping)
        }
        route_sidecar_path = (
            execution_route_qualification_sidecar_path_v1(
                qualification,
                contract.execution_route_sha256,
            )
            .expanduser()
            .resolve()
        )
        route_receipt = read_execution_route_qualification_sidecar_v1(
            qualification,
            contract.execution_route_sha256,
        )
        route_status = execution_route_qualification_status_v1(
            contract,
            route_receipt,
        )
        if (
            route_status.get("authority_state") != "route_qualified"
            or not route_sidecar_path.is_file()
            or route_sidecar_path.is_symlink()
            or not isinstance(route_receipt, Mapping)
            or route_receipt.get("artifact_sha256")
            != artifact_sha256s.get("cuda_extension")
        ):
            raise CampaignProviderError(
                "prepared-owner measurement requires exact route qualification"
            )
        route_qualification = {
            "authority_state": "route_qualified",
            "receipt_path": str(route_sidecar_path),
            "receipt_sha256": hashlib.sha256(
                route_sidecar_path.read_bytes()
            ).hexdigest(),
            "artifact_conformance_exact": True,
        }
        budgets = execution.get("budgets")
        vram_budget = (
            budgets.get("vram_bytes") if isinstance(budgets, Mapping) else None
        )
        if type(vram_budget) is not int or vram_budget < 1:
            raise CampaignProviderError(
                "prepared-owner audit requires a positive VRAM budget"
            )
        resource_estimate = self.estimate_resources(payload, (request,))
        resource_estimator_id = resource_estimate.get("resource_estimator_id")
        conservative_projection = resource_estimate.get("vram_bytes_projection")
        if (
            not isinstance(resource_estimator_id, str)
            or not resource_estimator_id
            or type(conservative_projection) is not int
            or conservative_projection <= 0
        ):
            raise CampaignProviderError(
                "prepared-owner audit requires a conservative provider projection"
            )
        runtime_environment_requirements = required_runtime_environment_v1(
            resource_estimate
        )
        runtime_environment = (
            runtime_environment_detector()
            if runtime_environment_detector is not None
            else os.environ
        )
        observed_runtime_environment = {
            key: runtime_environment.get(key)
            for key in runtime_environment_requirements
        }
        if any(
            observed_runtime_environment.get(key) != expected
            for key, expected in runtime_environment_requirements.items()
        ):
            raise CampaignProviderError(
                "prepared-owner audit runtime environment assumptions mismatch"
            )
        minimum_residual_headroom = minimum_residual_headroom_bytes_v1(
            resource_estimate
        )
        configuration_sha256 = prepared_owner_configuration_sha256_v1(
            payload, request.payload()
        )
        teardown_tolerance = int(
            getattr(
                self,
                "prepared_owner_teardown_tolerance_bytes",
                0,
            )
        )
        if not isinstance(measurement_authorization, Mapping):
            raise CampaignProviderError(
                "prepared-owner measurement authorization is required"
            )
        authorization = dict(measurement_authorization)
        authorization_sha256 = authorization.pop("authorization_sha256", None)
        if authorization.get(
            "schema_version"
        ) != MEASUREMENT_AUTHORIZATION_SCHEMA_VERSION or authorization_sha256 != canonical_sha256(
            authorization
        ):
            raise CampaignProviderError(
                "prepared-owner measurement authorization SHA-256 mismatch"
            )
        bindings = authorization.get("request_bindings")
        if not isinstance(bindings, Mapping):
            raise CampaignProviderError(
                "prepared-owner measurement authorization bindings are missing"
            )
        expected_bindings = {
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "campaign_id": str(payload.get("campaign_id")),
            "plan_sha256": str(payload.get("plan_sha256")),
            "qualification_key_sha256": str(payload.get("qualification_key_sha256")),
            "execution_route_sha256": route_sha256,
            "route_qualification": route_qualification,
            "artifact_sha256s": artifact_sha256s,
            "configuration_sha256": configuration_sha256,
            "resource_estimator_id": resource_estimator_id,
            "conservative_vram_projection_bytes": conservative_projection,
            "runtime_assumptions": runtime_environment_requirements,
            "ordered_cell_ids_sha256": canonical_sha256(list(request.cell_ids)),
            "batch_request_sha256": canonical_sha256(request.payload()),
            "dimensions": {
                "batch_cells": len(request.cell_ids),
                "resident_segment_generations": first_end,
                "ticks_per_generation": request.ticks_per_generation,
                "resident_capacity": uniform_integer("patch.capacity"),
                "initial_population": uniform_integer("patch.initial_population"),
                "agent_id_allocation_generation_count": allocation_generations,
                "physical_window_generations": physical_window_generations,
            },
        }
        for key, expected in expected_bindings.items():
            if bindings.get(key) != expected:
                raise CampaignProviderError(
                    f"prepared-owner measurement authorization binding mismatch: {key}"
                )
        hardware_identity_raw = bindings.get("hardware_identity")
        if not isinstance(hardware_identity_raw, Mapping):
            raise CampaignProviderError(
                "prepared-owner measurement hardware identity is missing"
            )
        try:
            hardware_identity = validate_cuda_hardware_identity_v1(
                hardware_identity_raw
            )
        except ValueError as exc:
            raise CampaignProviderError(
                f"prepared-owner measurement hardware identity is invalid: {exc}"
            ) from exc
        if hardware_identity != hardware_identity_raw:
            raise CampaignProviderError(
                "prepared-owner measurement hardware identity is not canonical"
            )
        hardware_total = hardware_identity.get("total_memory_bytes")
        if type(hardware_total) is not int or hardware_total <= 0:
            raise CampaignProviderError(
                "prepared-owner measurement hardware identity lacks total memory"
            )
        physical_headroom = math.ceil(hardware_total * 0.04)
        measurement_bootstrap_floor = physical_headroom + minimum_residual_headroom
        expected_policy = {
            "declared_vram_budget_bytes": vram_budget,
            "minimum_physical_headroom_fraction": 0.04,
            "minimum_physical_headroom_bytes": 0,
            "minimum_residual_headroom_bytes": minimum_residual_headroom,
            "measurement_live_free_policy": {
                "mode": "exact_allocation_only_measurement_bootstrap_v2",
                "minimum_free_bytes": measurement_bootstrap_floor,
                "requires_projection_within_total_memory": False,
                "allows_projection_over_declared_budget": True,
                "zero_dispatch_only": True,
                "production_execution_admission_unchanged": True,
            },
            "teardown_tolerance_bytes": teardown_tolerance,
        }
        if bindings.get("policy") != expected_policy:
            raise CampaignProviderError(
                "prepared-owner measurement authorization binding mismatch: policy"
            )
        if set(bindings) != {*expected_bindings, "hardware_identity", "policy"}:
            raise CampaignProviderError(
                "prepared-owner measurement authorization has unexpected bindings"
            )
        current_hardware = validate_cuda_hardware_identity_v1(
            (cuda_hardware_detector or detect_cuda_hardware_identity_v1)()
        )
        if current_hardware != hardware_identity:
            raise CampaignProviderError(
                "prepared-owner CUDA hardware differs from authorization"
            )
        free_bytes, total_bytes = (cuda_memory_detector or detect_cuda_memory_v1)()
        if (
            type(free_bytes) is not int
            or type(total_bytes) is not int
            or free_bytes < 0
            or total_bytes <= 0
            or free_bytes > total_bytes
            or total_bytes != hardware_total
            or free_bytes < measurement_bootstrap_floor
        ):
            raise CampaignProviderError(
                "prepared-owner live CUDA memory lacks measurement bootstrap floor"
            )
        if root.exists():
            raise CampaignProviderError(
                f"prepared-owner audit root already exists: {root}"
            )
        root.mkdir(parents=True)

        def request_factory(
            source_projection_root_sha256: str,
        ) -> ResidentPreparedOwnerResourceAuditRequestV1:
            return ResidentPreparedOwnerResourceAuditRequestV1(
                audit_id=audit_id,
                provider_id=self.provider_id,
                profile_id=self.profile_id,
                campaign_id=str(payload.get("campaign_id")),
                plan_sha256=str(payload.get("plan_sha256")),
                qualification_key_sha256=str(payload.get("qualification_key_sha256")),
                execution_route_sha256=route_sha256,
                route_qualification=route_qualification,
                candidate_authority_sha256=str(holder["candidate_authority_sha256"]),
                artifact_sha256s=artifact_sha256s,
                configuration_sha256=configuration_sha256,
                resource_estimator_id=resource_estimator_id,
                conservative_vram_projection_bytes=conservative_projection,
                runtime_assumptions=runtime_environment_requirements,
                measurement_authorization_sha256=str(authorization_sha256),
                hardware_identity=dict(hardware_identity),
                source_projection_root_sha256=source_projection_root_sha256,
                ordered_cell_ids_sha256=canonical_sha256(list(request.cell_ids)),
                batch_request_sha256=canonical_sha256(request.payload()),
                batch_cells=len(request.cell_ids),
                resident_segment_generations=first_end,
                ticks_per_generation=request.ticks_per_generation,
                resident_capacity=uniform_integer("patch.capacity"),
                initial_population=uniform_integer("patch.initial_population"),
                agent_id_allocation_generation_count=allocation_generations,
                physical_window_generations=physical_window_generations,
                declared_vram_budget_bytes=vram_budget,
                minimum_residual_headroom_bytes=minimum_residual_headroom,
                teardown_tolerance_bytes=teardown_tolerance,
            )

        try:
            import torch

            sampler = stage_sampler or cuda_memory_stage_sampler_v1(
                torch,
                expected_hardware_identity=hardware_identity,
            )
            serializer = projection_serializer or (
                lambda value, path: torch.save(value, path)
            )

            def release(owner: Any, module: Any) -> None:
                if resource_releaser is not None:
                    resource_releaser(owner, module)
                holder.clear()

            receipt = execute_prepared_owner_resource_audit_v1(
                request_factory=request_factory,
                module_loader=load_module,
                owner_preparer=prepare_owner,
                projection_exporter=export_projection,
                projection_destination=root / "preparation_projection.pt",
                projection_format_id="torch_save_projection_tuple_v1",
                projection_serializer=serializer,
                stage_sampler=sampler,
                resource_releaser=release,
                post_release_finalizer=torch.cuda.empty_cache,
            )
        except Exception:
            holder.clear()
            raise
        receipt_path = root / "prepared_owner_resource_audit_receipt.json"
        atomic_write_json(receipt_path, receipt.payload())
        return {
            **receipt.payload(),
            "receipt_path": str(receipt_path),
            "projection_path": str(root / "preparation_projection.pt"),
        }

    def _execute_segments(
        self,
        plan: Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
        *,
        start_generation: int,
        start_next_tick: int | None = None,
        resume_projections: Sequence[Mapping[str, Any]] | None,
        prior_metrics_rows: Sequence[Mapping[str, Any]] = (),
        prior_checkpoint_catalog: Mapping[str, Mapping[str, Any]] | None = None,
        prior_native_chronicle_receipt: Mapping[str, Any] | None = None,
        prior_native_continuity: Mapping[str, Any] | None = None,
        prior_chronicle_checkpoint_authority_sha256: str | None = None,
        event_sink_reconciliation_receipt: Mapping[str, Any] | None = None,
        preloaded_artifact: Any | None = None,
        execution_artifact_paths: tuple[str, str, str] | None = None,
        segment_artifact_observer: Callable[..., None] | None = None,
        segment_execution_observer: Callable[..., None] | None = None,
        launch_started_perf_ns: int | None = None,
        restore_timing_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        execution_launch_started_perf_ns = (
            time.perf_counter_ns()
            if launch_started_perf_ns is None
            else launch_started_perf_ns
        )
        from engine.campaigns.patch_runner_factory import build_patch_fixed_runner_v1
        from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
            prepare_direct_resident_patch_multicell_v1,
            project_direct_resident_patch_multicell_checkpoints_v1,
        )
        from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
            execute_direct_resident_windows_v1,
        )
        from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
            build_direct_resident_source_allocation_request_v1,
        )

        physical_window_generations = _physical_window_generations_from_plan_v1(plan)
        resolver = getattr(self, "_resolved_execution_adapters", None)
        resolved = dict(resolver(plan)) if callable(resolver) else {}
        runner_builder = resolved.get("runner_builder") or (
            self._runner_builder or build_patch_fixed_runner_v1
        )
        request_builder = resolved.get("request_builder") or (
            self._request_builder or build_direct_resident_source_allocation_request_v1
        )
        preparer = resolved.get("preparer") or (
            self._preparer or prepare_direct_resident_patch_multicell_v1
        )
        executor = resolved.get("executor") or (
            self._executor or execute_direct_resident_windows_v1
        )
        projector = resolved.get("projector") or (
            self._projector or project_direct_resident_patch_multicell_checkpoints_v1
        )
        artifact_loader = resolved.get("artifact_loader", self._artifact_loader)
        source_preparer = resolved.get("source_preparer", self._source_preparer)
        mechanics_binding = resolved.get("mechanics_binding")
        component_checkpoint_exporter = resolved.get("component_checkpoint_exporter")
        component_checkpoint_restorer = resolved.get("component_checkpoint_restorer")
        component_checkpoint_source_restorer = resolved.get(
            "component_checkpoint_source_restorer"
        )
        terminal_snapshot_metrics_merger = resolved.get(
            "terminal_snapshot_metrics_merger"
        )
        generation_zero_checkpoint_projector = resolved.get(
            "generation_zero_checkpoint_projector"
        )
        mechanics_binding = resolved.get("mechanics_binding")
        bounded_streamed_preparation = (
            request.total_ticks is not None
            and mechanics_binding is not None
            and _callable_accepts_keyword_v1(preparer, "per_cell_request_factory")
        )
        direct_batch_host_import_enable = (
            getattr(mechanics_binding, "binding_id", None)
            == "wdl_jungle_eed_fixed_mlp_native_binding_v1"
            and len(request.cell_ids) == 136
            and physical_window_generations == 1
        )
        if preloaded_artifact is None:
            if execution_artifact_paths is not None:
                raise CampaignProviderError(
                    "execution artifact paths require a preloaded artifact"
                )
            if segment_execution_observer is not None:
                raise CampaignProviderError(
                    "segment execution observation requires a preloaded "
                    "qualification artifact"
                )
            artifact_paths = resolved.get("artifact_paths")
            if artifact_paths is None:
                artifact, qualification, reporting = self._artifact_paths(plan)
            else:
                if (
                    not isinstance(artifact_paths, tuple)
                    or len(artifact_paths) != 3
                    or any(not isinstance(path, str) for path in artifact_paths)
                ):
                    raise CampaignProviderError(
                        "engine native binding artifact paths are malformed"
                    )
                artifact, qualification, reporting = artifact_paths
            loaded_artifact = self.load_qualified_artifact(
                artifact,
                qualification,
                reporting,
                loader=artifact_loader,
                expected_artifact_abi=getattr(mechanics_binding, "artifact_abi", None),
                expected_binding_id=getattr(mechanics_binding, "binding_id", None),
                required_exports=getattr(
                    mechanics_binding, "required_artifact_exports", None
                ),
            )
        else:
            if (
                getattr(preloaded_artifact, "production_authority", None) is not False
                or segment_artifact_observer is None
                or not isinstance(execution_artifact_paths, tuple)
                or len(execution_artifact_paths) != 3
                or any(
                    not isinstance(path, str) or not path
                    for path in execution_artifact_paths
                )
            ):
                raise CampaignProviderError(
                    "preloaded qualification execution requires false production "
                    "authority, a segment observer, and three explicit artifact paths"
                )
            artifact, qualification, reporting = execution_artifact_paths
            loaded_artifact = preloaded_artifact
        if segment_artifact_observer is not None and not callable(
            segment_artifact_observer
        ):
            raise CampaignProviderError("segment artifact observer must be callable")
        if segment_execution_observer is not None and not callable(
            segment_execution_observer
        ):
            raise CampaignProviderError("segment execution observer must be callable")
        module = getattr(loaded_artifact, "module", loaded_artifact)
        plan_artifacts = plan.get("artifacts")
        option_d_m0_binding_declared = (
            isinstance(plan_artifacts, Mapping)
            and OPTION_D_M0_SOURCE_ENVIRONMENT_ARTIFACT_NAME in plan_artifacts
        )
        option_d_m0_source_environment: dict[str, Any] | None = None
        option_d_m0_loaded_artifact_contract: dict[str, Any] | None = None
        if option_d_m0_binding_declared:
            option_d_m0_source_environment = (
                _option_d_m0_source_environment_identity_v1(plan)
            )
            option_d_m0_loaded_artifact_contract = (
                _option_d_m0_loaded_artifact_contract_v1(
                    module=module,
                    plan=plan,
                    source_environment=option_d_m0_source_environment,
                )
            )
        authoritative_identity_bindings: dict[str, str] | None = None
        if (
            getattr(self, "authoritative_state_schema_required_v1", False) is True
            and not hapb_host_authoritative_policy_binding_v1(mechanics_binding)
        ):
            from engine.authoritative_state.production_identity import (
                validate_identity_against_local_schema_v1,
            )
            from engine.authoritative_state.schema import load_schema_v1

            identity_export = getattr(
                module, "eiel_authoritative_state_identity_v1", None
            )
            if not callable(identity_export):
                raise CampaignProviderError(
                    "schema-bound artifact lacks binary self-description"
                )
            state_schema = load_schema_v1()
            binary_identity = validate_identity_against_local_schema_v1(
                dict(identity_export()),
                schema=state_schema,
                provider_id=str(self.provider_id),
                profile_id=str(self.profile_id),
            )
            authoritative_identity_bindings = {
                "authoritative_schema_hash": state_schema.schema_hash,
                "binary_schema_hash": binary_identity["state_schema_sha256"],
                "checkpoint_schema_hash": state_schema.components[
                    "checkpoint_schema.json"
                ]["contract_hash"],
            }
        cells = self._batch_cells(plan, request)
        production_parameter_contracts: Mapping[str, Any] | None = None
        production_control_readbacks: list[dict[str, Any]] = []
        if request.profile_id in {PRODUCTION_PROFILE_ID, PRODUCTION_CPU_PROFILE_ID}:
            from engine.campaigns.registry import default_profile_registry

            production_profile = next(
                (
                    profile
                    for profile in default_profile_registry().profiles()
                    if profile.profile_id == request.profile_id
                ),
                None,
            )
            if production_profile is None:
                raise CampaignProviderError(
                    f"production profile contract is unavailable: {request.profile_id}"
                )
            from engine.campaigns.production_control_path_v1 import (
                PRODUCTION_CONTROL_READBACK_PARAMETER_NAMES_V1,
            )

            production_parameter_contracts = {
                name: contract
                for name, contract in production_profile.parameter_contracts.items()
                if name in PRODUCTION_CONTROL_READBACK_PARAMETER_NAMES_V1
            }
        metrics_rows = [dict(row) for row in prior_metrics_rows]
        receipts: list[Mapping[str, Any]] = []
        compact_settlements: list[CompactSegmentSettlementV1] = []
        owner_release_receipts: list[OwnerReleaseReceiptV1] = []
        terminal_cuda_context_release_receipts: list[dict[str, Any]] = []
        owner_epoch_transfer_receipts: list[
            ResidentOwnerEpochTransferReceiptV1
        ] = []
        resident_owner: ResidentOwnerSpanV1 | None = None
        resident_owner_preparations = 0
        resident_owner_live_started_perf_ns: dict[str, int] = {}
        resident_owner_lifetime_measurements: list[dict[str, Any]] = []
        resident_owner_preparation_measurements: list[dict[str, Any]] = []
        resident_owner_max_simultaneous = 0

        def register_resident_owner(
            owner: ResidentOwnerSpanV1,
            *,
            preparation_started_perf_ns: int,
        ) -> None:
            nonlocal resident_owner_max_simultaneous
            identity = owner.prepared_owner_identity
            if owner.closed or identity in resident_owner_live_started_perf_ns:
                raise CampaignProviderError(
                    "resident owner lifecycle registration is invalid"
                )
            if resident_owner_live_started_perf_ns:
                raise CampaignProviderError(
                    "successor CUDA owner overlapped an unreleased resident owner"
                )
            resident_owner_live_started_perf_ns[identity] = (
                preparation_started_perf_ns
            )
            resident_owner_max_simultaneous = max(
                resident_owner_max_simultaneous,
                len(resident_owner_live_started_perf_ns),
            )

        def release_resident_owner(
            owner: ResidentOwnerSpanV1,
        ) -> OwnerReleaseReceiptV1:
            identity = owner.prepared_owner_identity
            started_perf_ns = resident_owner_live_started_perf_ns.get(identity)
            if type(started_perf_ns) is not int:
                raise CampaignProviderError(
                    "resident owner release lacks a measured live registration"
                )
            live_before = len(resident_owner_live_started_perf_ns)
            receipt = _release_resident_owner_v1(owner)
            resident_owner_live_started_perf_ns.pop(identity)
            finished_perf_ns = receipt.released_perf_ns
            resident_owner_lifetime_measurements.append(
                {
                    "owner_identity": identity,
                    "started_perf_ns": started_perf_ns,
                    "released_perf_ns": finished_perf_ns,
                    "lifetime_ns": finished_perf_ns - started_perf_ns,
                    "live_owner_count_before_release": live_before,
                    "live_owner_count_after_release": len(
                        resident_owner_live_started_perf_ns
                    ),
                    "release_receipt": receipt.payload(),
                }
            )
            return receipt

        def transfer_resident_owner_epoch(
            owner: ResidentOwnerSpanV1,
            *,
            successor_owner_identity: str,
            successor_span_end_tick: int,
        ) -> ResidentOwnerEpochTransferReceiptV1:
            prior_identity = owner.prepared_owner_identity
            started_perf_ns = resident_owner_live_started_perf_ns.get(
                prior_identity
            )
            if type(started_perf_ns) is not int:
                raise CampaignProviderError(
                    "resident owner epoch transfer lacks a live registration"
                )
            if len(resident_owner_live_started_perf_ns) != 1:
                raise CampaignProviderError(
                    "resident owner epoch transfer observed multiple physical owners"
                )
            receipt = _transfer_resident_owner_epoch_v1(
                owner,
                successor_owner_identity=successor_owner_identity,
                successor_span_end_tick=successor_span_end_tick,
            )
            resident_owner_live_started_perf_ns.pop(prior_identity)
            if successor_owner_identity in resident_owner_live_started_perf_ns:
                raise CampaignProviderError(
                    "resident owner epoch transfer reused a live logical identity"
                )
            # Preserve the physical allocation lifetime clock.  Epoch transfer
            # is metadata-only and must not masquerade as a release/rebuild.
            resident_owner_live_started_perf_ns[successor_owner_identity] = (
                started_perf_ns
            )
            return receipt

        def transfer_owner_epoch_at_boundary(
            owner: ResidentOwnerSpanV1 | None,
            *,
            boundary_next_tick: int,
            owner_transition: bool,
        ) -> bool:
            if (
                not tick_native
                or not owner_transition
                or boundary_next_tick >= int(request.total_ticks or 0)
            ):
                return False
            if owner is None:
                raise CampaignProviderError(
                    "tick-native logical owner transition lost its physical owner"
                )
            owner_span_ticks = request.resident_owner_span_ticks
            if type(owner_span_ticks) is not int or owner_span_ticks < 1:
                raise CampaignProviderError(
                    "tick-native owner transfer lacks its bounded span"
                )
            successor_span_end_tick = min(
                int(request.total_ticks or 0),
                boundary_next_tick + owner_span_ticks,
            )
            successor_identity = _resident_owner_identity_v1(
                batch_id=request.batch_id,
                cell_ids=request.cell_ids,
                start_next_tick=boundary_next_tick,
                end_next_tick=successor_span_end_tick,
                execution_route_sha256=request.execution_route_sha256,
            )
            owner_epoch_transfer_receipts.append(
                transfer_resident_owner_epoch(
                    owner,
                    successor_owner_identity=successor_identity,
                    successor_span_end_tick=successor_span_end_tick,
                )
            )
            return True
        # Continuation projections are a durable sequence, not an in-memory
        # B136 snapshot.  Keep it lazy until each cell's direct CUDA boundary
        # has consumed it.
        source_projections = resume_projections
        resume_projections = None
        dispatches = 0
        native_execution_receipts: list[dict[str, Any]] = []
        acs_capture_measurements: list[dict[str, Any]] = []
        acs_catalog_publication_measurements: list[dict[str, Any]] = []
        checkpoint_control_plane_transitions: list[dict[str, Any]] = []
        bounded_terminal_capture_receipts: list[dict[str, Any]] = []
        transient_storage_release_receipts: list[dict[str, Any]] = []
        authoritative_state_receipts: list[dict[str, Any]] = []
        authoritative_analysis_sources: list[dict[str, Any]] = []
        generation_series_artifacts: list[dict[str, Any]] = []
        current_segment_artifacts: list[dict[str, Any]] = []
        lifecycle_qacct_receipts: list[dict[str, Any]] = []
        tick_native = request.total_ticks is not None
        interval_ticks = (
            request.source_window_ticks
            if tick_native
            else request.ticks_per_generation
        )
        if type(interval_ticks) is not int or interval_ticks < 1:
            raise CampaignProviderError("bounded source window is invalid")
        current_next_tick = (
            start_generation * interval_ticks
            if start_next_tick is None
            else start_next_tick
        )
        execution_start_next_tick = current_next_tick
        current = current_next_tick // interval_ticks
        restore_schedule = request.deliberate_restore_ticks or (
            (request.deliberate_restore_tick,)
            if type(request.deliberate_restore_tick) is int
            else ()
        )
        deliberate_interruption_tick = next(
            (
                tick
                for tick in restore_schedule
                if tick_native and current_next_tick < tick
            ),
            None,
        )
        deliberately_interrupted = False
        checkpoint_payload: dict[str, Any] | None = None
        checkpoint_catalog: dict[str, dict[str, Any]] = {
            str(coordinate): dict(value)
            for coordinate, value in (
                prior_checkpoint_catalog.items()
                if isinstance(prior_checkpoint_catalog, Mapping)
                else ()
            )
        }
        component_checkpoint: Mapping[str, Any] | None = None
        if checkpoint_catalog:
            latest_generation = max(
                (int(value) for value in checkpoint_catalog),
                default=-1,
            )
            latest_checkpoint = checkpoint_catalog.get(str(latest_generation), {})
            prior_analysis_sources = latest_checkpoint.get(
                "authoritative_analysis_sources", []
            )
            if not isinstance(prior_analysis_sources, list) or any(
                not isinstance(row, Mapping) for row in prior_analysis_sources
            ):
                raise CampaignProviderError(
                    "prior authoritative analysis source catalog is invalid"
                )
            authoritative_analysis_sources.extend(
                dict(row) for row in prior_analysis_sources
            )
            component_descriptor = latest_checkpoint.get("component_checkpoint")
            if component_descriptor is not None:
                if not isinstance(component_descriptor, Mapping):
                    raise CampaignProviderError(
                        "prior engine component checkpoint is malformed"
                    )
                component_checkpoint = self._read_component_checkpoint(
                    component_descriptor
                )
        if component_checkpoint is not None and source_projections is not None:
            if not callable(component_checkpoint_source_restorer):
                raise CampaignProviderError(
                    "component checkpoint exists without a source restore mechanic"
                )
            source_projections = tuple(
                component_checkpoint_source_restorer(
                    source_projections,
                    component_checkpoint,
                    cell_ids=request.cell_ids,
                )
            )
            if len(source_projections) != len(request.cell_ids):
                raise CampaignProviderError(
                    "component checkpoint source restore changed the physical batch"
                )
        if (
            preloaded_artifact is None
            and request.reporting_enabled
            and self._generation_series_required(plan)
            and checkpoint_catalog
        ):
            prior_series_loader = getattr(
                self, "load_prior_generation_series_artifacts", None
            )
            if not callable(prior_series_loader):
                raise CampaignProviderError(
                    "required generation-series continuation loader is absent"
                )
            generation_series_artifacts.extend(
                dict(artifact)
                for artifact in prior_series_loader(
                    checkpoint_catalog=checkpoint_catalog,
                    cell_ids=request.cell_ids,
                )
            )
        qualification_sentinels: dict[str, list[dict[str, Any]]] | None = (
            {cell_id: [] for cell_id in request.cell_ids}
            if isinstance(plan.get("route_qualification_case_id"), str)
            else None
        )
        native_chronicle = None
        native_chronicle_observed = False
        output = plan.get("output")
        if (
            request.reporting_enabled
            and isinstance(output, Mapping)
            and output.get("chronicle_profile") == "scientific_full"
        ):
            from engine.campaigns.provider_native_chronicle_v1 import (
                ProviderNativeChronicleConsumerV1,
                ProviderNativeChronicleReceiptV1,
            )

            continuation_values = (
                prior_native_chronicle_receipt,
                prior_native_continuity,
                prior_chronicle_checkpoint_authority_sha256,
            )
            if all(value is None for value in continuation_values):
                native_chronicle = ProviderNativeChronicleConsumerV1(
                    batch_root=context.batch_root,
                    campaign_id=str(plan.get("campaign_id") or ""),
                    batch_id=request.batch_id,
                    cell_ids=request.cell_ids,
                    profile="scientific_full",
                    authoritative_identity_bindings=(authoritative_identity_bindings),
                )
            elif all(value is not None for value in continuation_values):
                native_chronicle = (
                    ProviderNativeChronicleConsumerV1.from_checkpoint_continuation(
                        batch_root=context.batch_root,
                        campaign_id=str(plan.get("campaign_id") or ""),
                        batch_id=(
                            f"{request.batch_id}-fresh-"
                            f"{str(prior_chronicle_checkpoint_authority_sha256)[:12]}"
                        ),
                        cell_ids=request.cell_ids,
                        profile="scientific_full",
                        prior_receipt=(
                            ProviderNativeChronicleReceiptV1.from_payload(
                                prior_native_chronicle_receipt
                            )
                        ),
                        prior_continuity=prior_native_continuity,
                        prior_authority_sha256=str(
                            prior_chronicle_checkpoint_authority_sha256
                        ),
                        authoritative_identity_bindings=(
                            authoritative_identity_bindings
                        ),
                    )
                )
            else:
                raise CampaignProviderError(
                    "provider-native Chronicle continuation binding is incomplete"
                )
        if request.checkpoint_storage_mode not in {"disk", "disk_and_memory"}:
            raise CampaignProviderError("unsupported checkpoint storage mode")
        if type(
            request.acs_checkpoint_slots
        ) is not int or request.acs_checkpoint_slots not in {0, 1, 2, 3}:
            raise CampaignProviderError(
                "ACS checkpoint slots must be zero, one, two, or three"
            )
        if type(request.acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "ACS capture instrumentation selection must be bool"
            )
        execution = plan.get("execution")
        plan_acs_checkpoint_slots = (
            execution.get("acs_checkpoint_slots", 0)
            if isinstance(execution, Mapping)
            else 0
        )
        raw_plan_acs_capture_instrumentation = (
            execution.get("acs_capture_instrumentation", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_plan_acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "plan ACS capture instrumentation selection must be bool"
            )
        plan_acs_capture_instrumentation = raw_plan_acs_capture_instrumentation or bool(
            plan_acs_checkpoint_slots
        )
        if (
            request.acs_checkpoint_slots != plan_acs_checkpoint_slots
            or request.acs_capture_instrumentation
            is not plan_acs_capture_instrumentation
        ):
            raise CampaignProviderError(
                "resident request ACS checkpoint route binding mismatch"
            )
        acs_measurement_enabled = request.acs_capture_instrumentation
        bounded_terminal_capture_route = bool(
            tick_native
            and request.acs_capture_instrumentation
            and request.acs_checkpoint_slots == 0
        )
        acs_pipeline: Any | None = None
        tick_native_detached_acs = False
        pending_acs_checkpoints: list[tuple[Any, dict[str, Any]]] = []
        source_projection_acs_token: Any | None = None
        source_lease_measurements: dict[int, dict[str, Any]] = {}
        acs_final_telemetry: dict[str, Any] | None = None
        if request.acs_checkpoint_slots:
            if self._active_acs_pipeline is not None:
                raise CampaignProviderError(
                    "ACS checkpoint staging provider is already active"
                )
            if option_d_m0_source_environment is not None:
                raise CampaignProviderError(
                    "ACS checkpoint staging is not admitted during Option D M0 qualification"
                )
            if native_chronicle is not None:
                raise CampaignProviderError(
                    "ACS checkpoint staging is not yet admitted with scientific_full Chronicle"
                )
            if callable(component_checkpoint_exporter):
                raise CampaignProviderError(
                    "ACS checkpoint staging is not yet admitted with component checkpoints"
                )
            if tick_native:
                if not request.native_streamed_checkpoint:
                    raise CampaignProviderError(
                        "tick-native asynchronous ACS requires native_streamed_checkpoint"
                    )
                if request.acs_checkpoint_slots not in {1, 2}:
                    raise CampaignProviderError(
                        "tick-native detached ACS supports exactly one or two slots"
                    )
                from engine.campaigns.acs_native_detached_slots_v1 import (
                    ACSNativeDetachedSlotsV1,
                )

                acs_pipeline = ACSNativeDetachedSlotsV1(
                    slots=request.acs_checkpoint_slots
                )
                tick_native_detached_acs = True
            else:
                from engine.campaigns.acs_checkpoint_staging import ACSCheckpointStaging

                acs_pipeline = ACSCheckpointStaging[dict[str, Any]](
                    slots=request.acs_checkpoint_slots
                )
            self._active_acs_pipeline = acs_pipeline
        physical_batch_detached_capture_route = bool(
            tick_native_detached_acs
            and request.profile_id == PRODUCTION_PROFILE_ID
            and request.native_streamed_checkpoint
            and len(request.cell_ids) == 136
        )
        physical_batch_terminal_capture_route = bool(
            bounded_terminal_capture_route
            and request.profile_id == PRODUCTION_PROFILE_ID
            and request.native_streamed_checkpoint
            and len(request.cell_ids) == 136
        )

        def finalize_oldest_acs_checkpoint(
            *, producer_backpressure: bool = False
        ) -> None:
            nonlocal checkpoint_payload, metrics_rows
            if acs_pipeline is None or not pending_acs_checkpoints:
                raise CampaignProviderError(
                    "ACS checkpoint finalization has no pending work"
                )
            token, capture = pending_acs_checkpoints[0]
            finalization_started_ns = time.perf_counter_ns()
            detached_receipt = None
            if tick_native_detached_acs:
                detached_receipt = acs_pipeline.result(
                    token,
                    producer_backpressure=producer_backpressure,
                )
                capture_measurement = capture.get("capture_barrier_measurement")
                if not isinstance(capture_measurement, dict):
                    raise CampaignProviderError(
                        "detached ACS capture measurement is not mutable"
                    )
                for timing_field in (
                    "detached_slot_reserved_perf_ns",
                    "detached_capture_submitted_perf_ns",
                    "detached_writer_started_perf_ns",
                    "detached_writer_finished_perf_ns",
                    "detached_writer_wall_time_ns",
                    "detached_writer_cpu_time_ns",
                ):
                    timing_value = detached_receipt.get(timing_field)
                    if type(timing_value) is not int or timing_value < 0:
                        raise CampaignProviderError(
                            f"detached ACS receipt has invalid {timing_field}"
                        )
                    capture_measurement[timing_field] = timing_value
                source_consumer_required = detached_receipt.get(
                    "detached_source_consumer_required"
                )
                source_released_perf_ns = detached_receipt.get(
                    "detached_source_released_perf_ns"
                )
                if type(source_consumer_required) is not bool or (
                    source_consumer_required
                    and source_released_perf_ns is not None
                    and (
                        type(source_released_perf_ns) is not int
                        or source_released_perf_ns < 0
                    )
                ):
                    raise CampaignProviderError(
                        "detached ACS receipt has invalid successor-source lease timing"
                    )
                capture_measurement["detached_source_consumer_required"] = (
                    source_consumer_required
                )
                capture_measurement["detached_source_released_perf_ns"] = (
                    source_released_perf_ns
                )
                receipt_authorities = detached_receipt.get(
                    "detached_authorities"
                )
                if isinstance(receipt_authorities, Mapping):
                    terminal_digests = receipt_authorities.get(
                        "source_projection_sha256s"
                    )
                    if not isinstance(terminal_digests, Mapping) or set(
                        terminal_digests
                    ) != set(request.cell_ids):
                        raise CampaignProviderError(
                            "detached physical settlement lacks exact terminal digests"
                        )
                    capture_measurement[
                        "immutable_snapshot_digest_sha256s"
                    ] = dict(terminal_digests)
                    capture_measurement[
                        "terminal_digest_authority_state"
                    ] = "complete_on_detached_host_settlement_v1"
                world_sharded_receipt = (
                    detached_receipt.get("schema_version")
                    == "eiel_acs_world_checkpoint_set_manifest_v1"
                )
                if world_sharded_receipt:
                    world_terminal_digests = {
                        str(row["cell_id"]): str(row["world_state_sha256"])
                        for row in detached_receipt.get("objects", ())
                        if isinstance(row, Mapping)
                    }
                    if set(world_terminal_digests) != set(request.cell_ids):
                        raise CampaignProviderError(
                            "world-sharded receipt lacks exact terminal state digests"
                        )
                    capture_measurement[
                        "immutable_snapshot_digest_sha256s"
                    ] = world_terminal_digests
                    capture_measurement[
                        "terminal_digest_authority_state"
                    ] = "complete_physical_world_state_v1"
                capture_measurement[
                    "detached_spool_materialization_time_ns"
                ] = int(
                    float(
                        detached_receipt.get(
                            "detached_spool_materialization_seconds", 0.0
                        )
                    )
                    * 1_000_000_000
                )
                checkpoint_set = (
                    self._world_checkpoint_set_from_detached_receipt_v1(
                        request=request,
                        receipt=detached_receipt,
                        writer_telemetry={
                            "owner_snapshot_capture_mode": (
                                "direct_physical_to_world_sharded_persistence_v1"
                            ),
                            "source_verification_seconds": float(
                                detached_receipt.get(
                                    "source_verification_seconds", 0.0
                                )
                            ),
                            "set_durability_seconds": float(
                                detached_receipt.get("set_durability_seconds", 0.0)
                            ),
                            "worker_count": int(
                                detached_receipt.get("worker_count", 0)
                            ),
                        },
                    )
                    if world_sharded_receipt
                    else self._aggregate_checkpoint_set_from_detached_receipt_v1(
                        request=request,
                        receipt=detached_receipt,
                        authorities=(
                            detached_receipt.get("detached_authorities")
                            if isinstance(
                                detached_receipt.get("detached_authorities"),
                                Mapping,
                            )
                            else capture.get("detached_authorities")
                        ),
                        writer_telemetry={
                            "owner_snapshot_capture_mode": (
                                "detached_file_backed_native_v1"
                            ),
                            "detached_capture_seconds": float(
                                detached_receipt.get(
                                    "detached_capture_seconds", 0.0
                                )
                            ),
                            "detached_finalize_seconds": float(
                                detached_receipt.get(
                                    "detached_finalize_seconds", 0.0
                                )
                            ),
                            "detached_spool_materialization_seconds": float(
                                detached_receipt.get(
                                    "detached_spool_materialization_seconds",
                                    0.0,
                                )
                            ),
                            "detached_slot_reserved_perf_ns": (
                                detached_receipt.get(
                                    "detached_slot_reserved_perf_ns"
                                )
                            ),
                            "detached_capture_submitted_perf_ns": (
                                detached_receipt.get(
                                    "detached_capture_submitted_perf_ns"
                                )
                            ),
                            "detached_writer_started_perf_ns": (
                                detached_receipt.get(
                                    "detached_writer_started_perf_ns"
                                )
                            ),
                            "detached_writer_finished_perf_ns": (
                                detached_receipt.get(
                                    "detached_writer_finished_perf_ns"
                                )
                            ),
                            "detached_payload_bytes": int(
                                detached_receipt.get("payload_bytes", 0)
                            ),
                            "detached_physical_file_bytes": int(
                                detached_receipt.get("physical_file_bytes", 0)
                            ),
                        },
                    )
                )
                projected_rows = [
                    dict(row)
                    for row in (
                        detached_receipt.get("physical_compact_result_rows", ())
                        or capture.get("metrics_rows", ())
                    )
                    if isinstance(row, Mapping)
                ]
                if projected_rows:
                    if getattr(self, "terminal_snapshot_metrics", False) or getattr(
                        mechanics_binding,
                        "terminal_snapshot_metrics",
                        False,
                    ):
                        if metrics_rows and callable(terminal_snapshot_metrics_merger):
                            metrics_rows = [
                                dict(row)
                                for row in terminal_snapshot_metrics_merger(
                                    metrics_rows,
                                    projected_rows,
                                )
                            ]
                        else:
                            metrics_rows = projected_rows
                    else:
                        metrics_rows.extend(projected_rows)
                capture["metrics_rows"] = [dict(row) for row in metrics_rows]
            else:
                checkpoint_set = acs_pipeline.result(
                    token,
                    producer_backpressure=producer_backpressure,
                )
            result_received_ns = time.perf_counter_ns()
            pending_acs_checkpoints.pop(0)
            parent_checkpoint_projection_digest_sha256s = None
            parent_checkpoint = checkpoint_catalog.get(
                str(capture["parent_coordinate"])
            )
            if isinstance(parent_checkpoint, Mapping):
                raw_parent = parent_checkpoint.get(
                    "terminal_projection_digest_sha256s"
                ) or parent_checkpoint.get("terminal_cell_digests")
                if not isinstance(raw_parent, Mapping):
                    raise CampaignProviderError(
                        "parent checkpoint lacks projection digest authority"
                    )
                parent_checkpoint_projection_digest_sha256s = {
                    str(cell_id): str(digest) for cell_id, digest in raw_parent.items()
                }
            checkpoint_identity_contract = _checkpoint_identity_contract_v1(
                execution_route_sha256=request.execution_route_sha256,
                generation=(
                    None if capture["tick_native"] else capture["generation"]
                ),
                tick=None,
                next_absolute_tick=(
                    capture["coordinate"] if capture["tick_native"] else None
                ),
                source_import_projection_digest_sha256s=(
                    capture["source_import_projection_digest_sha256s"]
                ),
                parent_checkpoint_projection_digest_sha256s=(
                    parent_checkpoint_projection_digest_sha256s
                ),
                terminal_projection_digest_sha256s=(
                    checkpoint_set["terminal_cell_digests"]
                ),
                canonical_terminal_state_digest_sha256s=(
                    checkpoint_set["mechanics_checkpoint_digests"]
                ),
                canonical_checkpoint_storage_digest_sha256s=(
                    checkpoint_set["canonical_checkpoint_storage_digests"]
                ),
            )
            capture["native_receipt"]["checkpoint_compression_time_ns"] = (
                checkpoint_set["writer_service_time_ns"]
            )
            capture["native_receipt"]["checkpoint_compression_time_scope"] = (
                "acs_background_provider_checkpoint_bundle_write_wall_time_"
                "including_serialization_digest_compression_and_io"
            )
            checkpoint_telemetry = acs_pipeline.telemetry()
            checkpoint_telemetry.update(
                {
                    "durable_checkpoint_frontier": token.compute_frontier,
                    "frontier_lag": max(
                        0,
                        checkpoint_telemetry["compute_frontier"]
                        - token.compute_frontier,
                    ),
                }
            )
            retained_prior_catalog = (
                _tick_native_retained_prior_checkpoint_catalog_v1(
                    checkpoint_catalog,
                    requested_checkpoint_ticks=request.checkpoint_ticks,
                    maximum_tick=int(capture["coordinate"]),
                )
                if capture["tick_native"]
                else {
                    coordinate: dict(value)
                    for coordinate, value in checkpoint_catalog.items()
                }
            )
            checkpoint_payload = {
                "schema_version": self.checkpoint_schema_version,
                "execution_route_sha256": request.execution_route_sha256,
                "checkpoint_binding": self._checkpoint_binding(plan, request),
                **(
                    {
                        "time_authority": "absolute_tick_v1",
                        "absolute_tick": capture["coordinate"],
                        "next_absolute_tick": capture["coordinate"],
                    }
                    if capture["tick_native"]
                    else {"generation": capture["generation"]}
                ),
                "continuation_storage_mode": request.checkpoint_storage_mode,
                "cell_bundles": checkpoint_set["bundle_rows"],
                "terminal_cell_digests": checkpoint_set["terminal_cell_digests"],
                "checkpoint_identity_contract": checkpoint_identity_contract,
                "checkpoint_identity_sha256": _checkpoint_identity_sha256_v1(
                    checkpoint_identity_contract
                ),
                "source_import_projection_digest_sha256s": (
                    checkpoint_identity_contract[
                        "source_import_projection_digest_sha256s"
                    ]
                ),
                "parent_checkpoint_projection_digest_sha256s": (
                    checkpoint_identity_contract[
                        "parent_checkpoint_projection_digest_sha256s"
                    ]
                ),
                "terminal_projection_digest_sha256s": (
                    checkpoint_identity_contract["terminal_projection_digest_sha256s"]
                ),
                "canonical_terminal_state_digest_sha256s": (
                    checkpoint_identity_contract[
                        "canonical_terminal_state_digest_sha256s"
                    ]
                ),
                "canonical_checkpoint_storage_digest_sha256s": (
                    checkpoint_identity_contract[
                        "canonical_checkpoint_storage_digest_sha256s"
                    ]
                ),
                "route_envelope_or_execution_provenance_digest_sha256": (
                    checkpoint_identity_contract[
                        "route_envelope_or_execution_provenance_digest_sha256"
                    ]
                ),
                "mechanics_checkpoint_digests": checkpoint_set[
                    "mechanics_checkpoint_digests"
                ],
                "canonical_checkpoint_storage_digests": checkpoint_set[
                    "canonical_checkpoint_storage_digests"
                ],
                "checkpoint_digest_domain": checkpoint_set.get(
                    "checkpoint_digest_domain", "canonical_projection_v1"
                ),
                **(
                    {
                        "checkpoint_set_manifest_path": checkpoint_set[
                            "checkpoint_set_manifest_path"
                        ],
                        "checkpoint_set_manifest_sha256": checkpoint_set[
                            "checkpoint_set_manifest_sha256"
                        ],
                    }
                    if checkpoint_set.get("checkpoint_set_manifest_path")
                    else {}
                ),
                "component_field_trace_status": (
                    "exact_component_field_unavailable_only_terminal_projection_digest"
                ),
                "metrics_rows": capture["metrics_rows"],
                **(
                    {
                        "production_control_readback": capture[
                            "production_control_readback"
                        ]
                    }
                    if capture["production_control_readback"] is not None
                    else {}
                ),
                "authoritative_analysis_sources": capture[
                    "authoritative_analysis_sources"
                ],
                "prior_checkpoint_catalog": retained_prior_catalog,
                "acs_checkpoint_staging": {
                    **checkpoint_telemetry,
                    "capture_mode": (
                        "detached_file_backed_native_v1"
                        if tick_native_detached_acs
                        else "legacy_callable_snapshot_v1"
                    ),
                    **(
                        {
                            "detached_checkpoint_receipt": dict(
                                detached_receipt
                            )
                        }
                        if detached_receipt is not None
                        else {}
                    ),
                    "captured_bytes": capture["capture_barrier_measurement"][
                        "captured_bytes"
                    ],
                    "capture_barrier_measurement": capture[
                        "capture_barrier_measurement"
                    ],
                    "writer_service_time_ns": checkpoint_set["writer_service_time_ns"],
                    "writer_cpu_time_ns": checkpoint_set["writer_cpu_time_ns"],
                    "cell_service_times_ns": checkpoint_set["cell_service_times_ns"],
                    "cell_cpu_times_ns": checkpoint_set["cell_cpu_times_ns"],
                    "cell_stage_telemetry_seconds": checkpoint_set[
                        "cell_stage_telemetry_seconds"
                    ],
                    "cell_checkpoint_bytes": checkpoint_set["cell_checkpoint_bytes"],
                    "checkpoint_bytes": checkpoint_set["checkpoint_bytes"],
                    "checkpoint_bytes_measurement_status": checkpoint_set[
                        "checkpoint_bytes_measurement_status"
                    ],
                    "native_checkpoint_receipts": checkpoint_set[
                        "native_checkpoint_receipts"
                    ],
                    "native_capture_group_receipts": checkpoint_set[
                        "native_capture_group_receipts"
                    ],
                },
            }
            payload_materialized_ns = time.perf_counter_ns()
            context_publication_started_ns = time.perf_counter_ns()
            prior_catalog_snapshot = {
                coordinate: dict(value)
                for coordinate, value in checkpoint_catalog.items()
            }
            postpublication_catalog = (
                _tick_native_catalog_after_async_publication_v1(
                    prior_catalog_snapshot,
                    published_coordinate=int(capture["coordinate"]),
                    published_checkpoint=checkpoint_payload,
                    requested_checkpoint_ticks=request.checkpoint_ticks,
                )
                if capture["tick_native"]
                else {
                    **retained_prior_catalog,
                    str(capture["coordinate"]): dict(checkpoint_payload),
                }
            )
            if capture["tick_native"]:
                manifest_checkpoint_frontier = (
                    _record_tick_native_async_publication_v1(
                        context,
                        published_coordinate=int(capture["coordinate"]),
                        published_checkpoint=checkpoint_payload,
                        postpublication_catalog=postpublication_catalog,
                    )
                )
            else:
                context.record_checkpoint(checkpoint_payload)
                manifest_checkpoint_frontier = int(capture["coordinate"])
            context_published_ns = time.perf_counter_ns()
            checkpoint_catalog.clear()
            checkpoint_catalog.update(postpublication_catalog)
            if capture["tick_native"]:
                transient_storage_release_receipts.extend(
                    _release_tick_native_transient_storage_v1(
                        context=context,
                        prior_checkpoint_catalog=prior_catalog_snapshot,
                        retained_checkpoint_catalog=postpublication_catalog,
                        replacement_checkpoint=checkpoint_payload,
                        requested_checkpoint_ticks=request.checkpoint_ticks,
                        release_source_import=bool(
                            capture.get("release_source_import_after_publication")
                        ),
                    )
                )
            catalog_bound_ns = time.perf_counter_ns()
            acs_pipeline.acknowledge_durable(token)
            durable_acknowledged_ns = time.perf_counter_ns()
            durable_state: dict[str, Any] | None = None
            if request.world_sharded_checkpoint:
                durable_state = {
                    "schema_version": (
                        "eiel_world_sharded_checkpoint_control_plane_state_v1"
                    ),
                    "state": "RUN_DURABLE_COMPLETE",
                    "absolute_next_tick": int(capture["coordinate"]),
                    "entered_perf_ns": durable_acknowledged_ns,
                    "entered_unix_ns": time.time_ns(),
                    "terminal_cuda_owner_released": True,
                    "world_objects_sealed": True,
                    "set_manifest_committed": True,
                    "inventory_complete": True,
                    "restore_admitted": True,
                    "lifecycle_release_subject_to_contract": True,
                    "qualification_closeout_subject_to_gates": True,
                    "checkpoint_set_manifest_path": checkpoint_set.get(
                        "checkpoint_set_manifest_path"
                    ),
                    "checkpoint_set_manifest_sha256": checkpoint_set.get(
                        "checkpoint_set_manifest_sha256"
                    ),
                }
                checkpoint_control_plane_transitions.append(durable_state)
            acs_catalog_publication_measurements.append(
                {
                    "coordinate": int(capture["coordinate"]),
                    "durable_completed_unix_ns": time.time_ns(),
                    "durable_completed_perf_ns": time.perf_counter_ns(),
                    "durable_acknowledged_perf_ns": durable_acknowledged_ns,
                    "writer_result_wait_time_ns": (
                        result_received_ns - finalization_started_ns
                    ),
                    "payload_materialization_time_ns": (
                        payload_materialized_ns - result_received_ns
                    ),
                    "context_publication_time_ns": (
                        context_published_ns - context_publication_started_ns
                    ),
                    "manifest_checkpoint_frontier": manifest_checkpoint_frontier,
                    "catalog_binding_time_ns": (
                        catalog_bound_ns - context_published_ns
                    ),
                    "durable_acknowledgement_time_ns": (
                        durable_acknowledged_ns - catalog_bound_ns
                    ),
                    "total_finalization_time_ns": (
                        durable_acknowledged_ns - finalization_started_ns
                    ),
                    **(
                        {"checkpoint_control_plane_state": durable_state}
                        if durable_state is not None
                        else {}
                    ),
                }
            )

        if tick_native:
            segment_specs = self._absolute_tick_segments_v1(
                request, start_next_tick=current_next_tick
            )
        else:
            legacy_specs: list[dict[str, Any]] = []
            legacy_start = current
            for legacy_end in self._segment_ends(request, start=current):
                legacy_specs.append(
                    {
                        "start_next_tick": legacy_start * interval_ticks,
                        "end_next_tick": legacy_end * interval_ticks,
                        "start_generation": legacy_start,
                        "start_tick": 0,
                        "terminal_generation": legacy_end,
                        "terminal_tick": 0,
                        "logical_generation_count": legacy_end - legacy_start,
                        "absolute_tick_span": None,
                        "publish_checkpoint": True,
                    }
                )
                legacy_start = legacy_end
            segment_specs = tuple(legacy_specs)
        expected_resident_owner_transition_ticks = tuple(
            int(spec["end_next_tick"])
            for spec in segment_specs
            if tick_native
            and int(spec["end_next_tick"]) == int(spec["owner_span_end_tick"])
            and int(spec["end_next_tick"]) < int(request.total_ticks or 0)
        )
        observed_resident_owner_transition_ticks: list[int] = []
        if (
            source_projections is None
            and current == 0
            and not bounded_streamed_preparation
            and (
                source_preparer is not None
                or (
                    (
                        mechanics_binding is None
                        or getattr(mechanics_binding, "binding_id", None)
                        == "patch_fixed_mlp_native_binding_v1"
                    )
                    and self._runner_builder is None
                    and self._request_builder is None
                    and self._preparer is None
                )
            )
            and segment_specs
        ):
            source_preparer = (
                source_preparer or self._prepare_generation_zero_sources_v2
            )
            source_preparer_arguments = {
                "runner_builder": runner_builder,
                "cells": cells,
                "artifact": artifact,
                "qualification": qualification,
                "reporting": reporting,
            }
            if not tick_native:
                source_preparer_arguments["segment_generations"] = (
                    request.generations
                )
            source_projections = tuple(
                source_preparer(
                    plan,
                    request,
                    context,
                    **source_preparer_arguments,
                )
            )
        for segment_ordinal, segment_spec in enumerate(segment_specs, start=1):
            if acs_pipeline is not None:
                while pending_acs_checkpoints and acs_pipeline.ready(
                    pending_acs_checkpoints[0][0]
                ):
                    finalize_oldest_acs_checkpoint()
                acs_pipeline.poll_failure()
            segment_control_readback: dict[str, Any] | None = None
            # These aliases can each retain the full CUDA owner.  Keep their
            # lifecycle explicit so a completed segment is not accidentally
            # co-resident with the successor prepared below.
            abi_owner: Any | None = None
            post_dispatch_owner: Any | None = None
            executor_arguments: dict[str, Any] = {}
            segment_start_next_tick = int(segment_spec["start_next_tick"])
            segment_end_next_tick = int(segment_spec["end_next_tick"])
            current = int(segment_spec["start_generation"])
            start_tick = int(segment_spec["start_tick"])
            end = int(segment_spec["terminal_generation"])
            terminal_tick = int(segment_spec["terminal_tick"])
            segment_generations = int(segment_spec["logical_generation_count"])
            absolute_tick_span = segment_spec["absolute_tick_span"]
            publish_checkpoint = bool(segment_spec["publish_checkpoint"])
            owner_span_start_tick = (
                int(segment_spec["owner_span_start_tick"])
                if tick_native
                else segment_start_next_tick
            )
            owner_span_end_tick = (
                int(segment_spec["owner_span_end_tick"])
                if tick_native
                else segment_end_next_tick
            )
            owner_transition = bool(
                tick_native and segment_end_next_tick == owner_span_end_tick
            )
            # A logical owner-span boundary is not a checkpoint boundary.  The
            # physical B136 allocation is transferred to the successor epoch
            # in place; only explicitly requested checkpoints enter ACS.
            full_checkpoint_boundary = publish_checkpoint
            terminal_physical_capture_route = bool(
                physical_batch_terminal_capture_route
                and full_checkpoint_boundary
                and (
                    segment_end_next_tick == request.total_ticks
                    or segment_end_next_tick == deliberate_interruption_tick
                )
            )
            acs_capture_token: Any | None = None
            prepare_owner = not tick_native or resident_owner is None
            if tick_native and resident_owner is not None:
                resident_owner.assert_cursor(
                    start_next_tick=segment_start_next_tick,
                    end_next_tick=segment_end_next_tick,
                )
                if (
                    resident_owner.start_next_tick != owner_span_start_tick
                    or resident_owner.end_next_tick != owner_span_end_tick
                    or resident_owner.cell_ids != tuple(request.cell_ids)
                ):
                    raise CampaignProviderError(
                        "tick-native segment crossed resident owner identity"
                    )
            # A fresh owner restored from a checkpoint inside a resident
            # owner-span starts at the actual source cursor, not at the
            # historical envelope's lower bound.  The envelope still bounds
            # the owner identity; the imported source cursor owns semantics.
            private_owner_abi_arguments: dict[str, Any] = {}
            if not tick_native:
                private_owner_abi_arguments = {
                    "allocation_generations": (
                        self._agent_id_allocation_generation_count(
                            request, start_generation=current
                        )
                    ),
                    "physical_window_generations": physical_window_generations,
                }
            allow_short_technical_gate_generation_count = (
                request.batch_id
                in {
                    "technical-gate-canary",
                    "technical-gate-fresh-owner-restore",
                }
                and len(request.cell_ids) == 1
                and segment_generations in {1, 2}
            )
            allow_short_technical_gate_generation_count = (
                allow_short_technical_gate_generation_count or tick_native
            )
            successor_checkpoint_capture = next(
                (
                    pending_capture
                    for _pending_token, pending_capture in reversed(
                        pending_acs_checkpoints
                    )
                    if pending_capture.get("coordinate")
                    == segment_start_next_tick
                ),
                None,
            )
            if prepare_owner and source_projection_acs_token is not None:
                source_token_sequence = getattr(
                    source_projection_acs_token, "sequence", None
                )
                measurement = source_lease_measurements.get(
                    source_token_sequence
                )
                if not isinstance(measurement, dict):
                    raise CampaignProviderError(
                        "detached successor-source lease lacks mutable timing evidence"
                    )
                measurement["successor_owner_prepare_started_perf_ns"] = (
                    time.perf_counter_ns()
                )
            if prepare_owner:
                if resident_owner_live_started_perf_ns:
                    raise CampaignProviderError(
                        "resident owner preparation began before prior CUDA owner release"
                    )
                owner_preparation_started_perf_ns = time.perf_counter_ns()
                owner_coordinate_arguments = (
                    {
                        "owner_start_tick": segment_start_next_tick,
                        "owner_end_tick": owner_span_end_tick,
                    }
                    if tick_native
                    else {
                        "start_generation": current,
                        "end_generation": current + segment_generations,
                        "ticks_per_generation": request.ticks_per_generation,
                    }
                )
                builds, requests, prepared = self._prepare_resident_segment_owner_v1(
                    plan=plan,
                    runner_builder=runner_builder,
                    request_builder=request_builder,
                    preparer=preparer,
                    cells=cells,
                    cell_ids=request.cell_ids,
                    batch_root=context.batch_root,
                    **owner_coordinate_arguments,
                    artifact=artifact,
                    qualification=qualification,
                    reporting=reporting,
                    module=module,
                    source_projections=source_projections,
                    **private_owner_abi_arguments,
                    genome_v2_checkpoint_authority_binding=dict(
                        request.genome_v2_checkpoint_authority_binding
                    ),
                    tick_native_control=(
                        dict(request.tick_native_control) if tick_native else None
                    ),
                    allow_short_technical_gate_generation_count=(
                        allow_short_technical_gate_generation_count
                    ),
                    allow_partial_generation_source_cursor=tick_native,
                    provider=self,
                    enable_streamed_preparation=bounded_streamed_preparation,
                    direct_batch_host_import_enable=direct_batch_host_import_enable,
                    indexed_source_capture=bool(
                        terminal_physical_capture_route
                        and request.world_sharded_checkpoint
                    ),
                )
                if tick_native:
                    _trim_cuda_allocator_after_resident_owner_preparation_v1()
                    resident_owner_preparations += 1
                    resident_owner = ResidentOwnerSpanV1(
                        cell_ids=tuple(request.cell_ids),
                        start_next_tick=owner_span_start_tick,
                        end_next_tick=owner_span_end_tick,
                        next_absolute_tick=segment_start_next_tick,
                        prepared_owner_identity=_resident_owner_identity_v1(
                            batch_id=request.batch_id,
                            cell_ids=request.cell_ids,
                            start_next_tick=owner_span_start_tick,
                            end_next_tick=owner_span_end_tick,
                            execution_route_sha256=request.execution_route_sha256,
                        ),
                        builds=builds,
                        requests=requests,
                        prepared_owner=prepared,
                        tick_native_prefix_by_cell=_tick_native_source_prefix_rows_v1(
                            prepared,
                            # The streamed preparer has already consumed and
                            # released every detached spool cell into the new
                            # owner.  Its tensor-free metadata/runtime rows are
                            # now authoritative; rereading 136 tensor payloads
                            # here would be redundant host work.
                            source_projections=(
                                None
                                if isinstance(
                                    source_projections,
                                    _LazyDetachedSpoolProjectionSequence,
                                )
                                else source_projections
                            ),
                            cell_ids=request.cell_ids,
                        ),
                    )
                    register_resident_owner(
                        resident_owner,
                        preparation_started_perf_ns=(
                            owner_preparation_started_perf_ns
                        ),
                    )
                    owner_preparation_completed_perf_ns = time.perf_counter_ns()
                    resident_owner_preparation_measurements.append(
                        {
                            "start_perf_ns": owner_preparation_started_perf_ns,
                            "end_perf_ns": owner_preparation_completed_perf_ns,
                            "owner_id": resident_owner.prepared_owner_identity,
                            "owner_span": {
                                "start_next_tick": owner_span_start_tick,
                                "end_next_tick": owner_span_end_tick,
                            },
                        }
                    )
                    if source_projection_acs_token is not None:
                        if acs_pipeline is None:
                            raise CampaignProviderError(
                                "detached spool source lease lacks its ACS pipeline"
                            )
                        source_token_sequence = getattr(
                            source_projection_acs_token, "sequence", None
                        )
                        measurement = source_lease_measurements.get(
                            source_token_sequence
                        )
                        if not isinstance(measurement, dict):
                            raise CampaignProviderError(
                                "detached successor owner lost its timing evidence"
                            )
                        measurement["successor_owner_prepare_completed_perf_ns"] = (
                            time.perf_counter_ns()
                        )
            elif resident_owner is not None:
                builds = resident_owner.builds
                requests = resident_owner.requests
                prepared = resident_owner.prepared_owner
            else:
                raise CampaignProviderError(
                    "tick-native segment has no resident owner"
                )
            compact_source_projections = (
                _resident_owner_compact_source_projections_v1(
                    prepared,
                    bounded_streamed_preparation=bounded_streamed_preparation,
                )
            )
            prepared_arguments = getattr(
                prepared, "physical_window_arguments", prepared
            )
            source_projection_cursor_factory = getattr(
                prepared_arguments, "source_projection_cursor_factory", None
            )
            if (
                len(compact_source_projections) != len(request.cell_ids)
                and not (
                    bounded_streamed_preparation
                    and callable(
                        getattr(
                            prepared_arguments,
                            "source_projection_cursor_factory",
                            None,
                        )
                    )
                )
            ):
                raise CampaignProviderError(
                    "prepared owner lacks ordered source import projections"
                )
            # The builder has already imported each source into the direct
            # CUDA boundary. Retain only the compact canonical views from the
            # prepared owner; this drops decoded resume bundles and avoids a
            # second B136 host bank.
            if bounded_streamed_preparation:
                # The source cursor was consumed into the resident owner during
                # preparation. Never retain or reopen its detached descriptor
                # sequence across tick-native subspans.
                source_projections = None
            elif compact_source_projections:
                # Re-bind the current resident owner's compact views at every
                # subspan.  Detached ACS releases the previous snapshot after
                # capture; the next boundary must obtain the current owner
                # view again without retaining the old snapshot in a worker.
                source_projections = compact_source_projections
            elif prepare_owner:
                source_projections = None
            prepared_arguments = None
            if component_checkpoint is not None:
                if not callable(component_checkpoint_restorer):
                    raise CampaignProviderError(
                        "component checkpoint exists without a restore mechanic"
                    )
                restored = component_checkpoint_restorer(
                    prepared.physical_window_arguments,
                    component_checkpoint,
                    **(
                        {"absolute_tick": segment_start_next_tick}
                        if tick_native
                        else {"absolute_generation": current}
                    ),
                    cell_ids=request.cell_ids,
                )
                expected_restore_cursor = (
                    segment_start_next_tick if tick_native else current
                )
                if restored != expected_restore_cursor:
                    raise CampaignProviderError(
                        "component checkpoint restored the wrong time cursor"
                    )
            if (
                getattr(self, "authoritative_state_schema_required_v1", False) is True
                and not hapb_host_authoritative_policy_binding_v1(mechanics_binding)
            ):
                from engine.authoritative_state.runtime_abi import (
                    observe_prepared_direct_resident_abi_v1,
                    resolve_prepared_direct_resident_trajectory_v1,
                    validate_runtime_abi_order_v1,
                )
                from engine.authoritative_state.schema import load_schema_v1

                state_schema = load_schema_v1()
                abi_owner = resolve_prepared_direct_resident_trajectory_v1(prepared)
                runtime_receipt = observe_prepared_direct_resident_abi_v1(abi_owner)
                admitted_variants = validate_runtime_abi_order_v1(
                    runtime_receipt,
                    state_schema.abi_variants,
                    require_production_admission=True,
                    prepared=abi_owner,
                )
                authoritative_state_receipts.append(
                    {
                        "schema_version": (
                            "eiel_authoritative_state_pre_dispatch_receipt_v1"
                        ),
                        "state_schema_sha256": state_schema.schema_hash,
                        "segment_ordinal": segment_ordinal,
                        **(
                            {
                                "start_next_tick": segment_start_next_tick,
                                "end_next_tick": segment_end_next_tick,
                            }
                            if tick_native
                            else {}
                        ),
                        "start_generation": current,
                        "end_generation": end,
                        "runtime_abi_receipt_sha256": runtime_receipt["receipt_sha256"],
                        "admitted_abi_variants": admitted_variants,
                        "identity_bindings": dict(
                            authoritative_identity_bindings or {}
                        ),
                    }
                )
            if current == 0 and 0 in request.checkpoint_milestones:
                initial_projection_cursor = None
                if (
                    bounded_streamed_preparation
                    and callable(source_projection_cursor_factory)
                ):
                    initial_projection_cursor = source_projection_cursor_factory()
                    initial_projections: tuple[Mapping[str, Any], ...] = ()
                else:
                    initial_projections = (
                        self._generation_zero_checkpoint_projections_v1(
                            prepared,
                            projector=generation_zero_checkpoint_projector,
                        )
                    )
                    if len(initial_projections) != len(request.cell_ids):
                        raise CampaignProviderError(
                            "prepared owner lacks ordered G0 source projections"
                        )
                initial_bundles: list[dict[str, str]] = []
                initial_digests: dict[str, str] = {}
                initial_canonical_terminal_state_digests: dict[str, str] = {}
                initial_canonical_checkpoint_storage_digests: dict[str, str] = {}
                from engine.native_runtime.checkpoint_digests import (
                    DigestLayer,
                )

                def initial_layer_digest(
                    projection: Mapping[str, Any], layer: Any
                ) -> str:
                    return self._layered_checkpoint_digest_v1(projection, layer)

                for index, cell_id in enumerate(request.cell_ids):
                    detached_initial = None
                    if initial_projection_cursor is not None:
                        detached_initial = initial_projection_cursor.next_cell()
                        if (
                            detached_initial is None
                            or detached_initial.cell_id != cell_id
                            or detached_initial.cell_index != index
                        ):
                            initial_projection_cursor.fail(
                                "G0 source cursor emitted unordered cell"
                            )
                            raise CampaignProviderError(
                                "G0 source cursor emitted unordered cell"
                            )
                        projection = detached_initial.require_payload()
                    else:
                        projection = initial_projections[index]
                    path = (
                        context.batch_root
                        / "checkpoint_bundles_v2"
                        / cell_id
                        / (
                            "bundle_tick000000000000"
                            if tick_native
                            else "bundle_gen000000"
                        )
                    )
                    published = self._write_projection_bundle(
                        projection,
                        path,
                        generation=0,
                        batch_id=request.batch_id,
                        cell_id=cell_id,
                        segment_ordinal=0,
                    )
                    source_digest = self.terminal_projection_digest(projection)
                    initial_digests[cell_id] = source_digest
                    initial_canonical_terminal_state_digests[cell_id] = (
                        initial_layer_digest(projection, DigestLayer.TRAJECTORY_STATE)
                    )
                    initial_canonical_checkpoint_storage_digests[cell_id] = (
                        initial_layer_digest(
                            projection,
                            layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
                        )
                    )
                    initial_bundles.append(
                        {
                            "cell_id": cell_id,
                            "path": published,
                            "source_projection_sha256": source_digest,
                        }
                    )
                    if initial_projection_cursor is not None:
                        initial_projection_cursor.import_and_release(cell_id, None)
                if initial_projection_cursor is not None:
                    initial_cursor_receipt = initial_projection_cursor.close()
                    if (
                        initial_cursor_receipt.state != "SEALED"
                        or initial_cursor_receipt.max_live_tensor_bearing_payloads != 1
                        or initial_cursor_receipt.importer_concurrency != 1
                    ):
                        raise CampaignProviderError(
                            "G0 source cursor did not prove bounded import"
                        )
                initial_identity_contract = _checkpoint_identity_contract_v1(
                    execution_route_sha256=request.execution_route_sha256,
                    generation=None if tick_native else 0,
                    absolute_tick=0 if tick_native else None,
                    next_absolute_tick=0 if tick_native else None,
                    source_import_projection_digest_sha256s=initial_digests,
                    parent_checkpoint_projection_digest_sha256s=None,
                    terminal_projection_digest_sha256s=initial_digests,
                    canonical_terminal_state_digest_sha256s=(
                        initial_canonical_terminal_state_digests
                    ),
                    canonical_checkpoint_storage_digest_sha256s=(
                        initial_canonical_checkpoint_storage_digests
                    ),
                )
                initial_checkpoint = {
                    "schema_version": self.checkpoint_schema_version,
                    "execution_route_sha256": request.execution_route_sha256,
                    "checkpoint_binding": self._checkpoint_binding(plan, request),
                    "continuation_storage_mode": request.checkpoint_storage_mode,
                    "cell_bundles": initial_bundles,
                    "terminal_cell_digests": initial_digests,
                    "checkpoint_identity_contract": initial_identity_contract,
                    "source_import_projection_digest_sha256s": (
                        initial_identity_contract[
                            "source_import_projection_digest_sha256s"
                        ]
                    ),
                    "parent_checkpoint_projection_digest_sha256s": None,
                    "terminal_projection_digest_sha256s": (
                        initial_identity_contract["terminal_projection_digest_sha256s"]
                    ),
                    "canonical_terminal_state_digest_sha256s": (
                        initial_identity_contract[
                            "canonical_terminal_state_digest_sha256s"
                        ]
                    ),
                    "canonical_checkpoint_storage_digest_sha256s": (
                        initial_identity_contract[
                            "canonical_checkpoint_storage_digest_sha256s"
                        ]
                    ),
                    "route_envelope_or_execution_provenance_digest_sha256": (
                        initial_identity_contract[
                            "route_envelope_or_execution_provenance_digest_sha256"
                        ]
                    ),
                    "component_field_trace_status": (
                        "exact_component_field_unavailable_only_terminal_projection_digest"
                    ),
                    "metrics_rows": [dict(row) for row in metrics_rows],
                }
                if tick_native:
                    initial_checkpoint.update(
                        {
                            "time_authority": "absolute_tick_v1",
                            "absolute_tick": 0,
                            "next_absolute_tick": 0,
                        }
                    )
                else:
                    initial_checkpoint["generation"] = 0
                # Generation zero is a producer-published checkpoint, not
                # merely an internal source catalog entry.  Publish its exact
                # bundle inventory at the same boundary as later checkpoints
                # so terminal lifecycle closeout can apply the declared R1
                # role and the pre-admission reclaimer can safely release it
                # after the terminal successor is durable.
                if getattr(context, "lifecycle_contract", None) is not None:
                    context.record_checkpoint(initial_checkpoint)
                checkpoint_catalog["0"] = initial_checkpoint
                if native_chronicle is not None:
                    if initial_projection_cursor is not None:
                        if not callable(source_projection_cursor_factory):
                            raise CampaignProviderError(
                                "Chronicle G0 cursor requires a source cursor factory"
                            )
                        # The cursor used to publish the G0 bundles is sealed
                        # above.  Give Chronicle a fresh cursor over the same
                        # durable source descriptors rather than retaining a
                        # second B136 projection bank or reusing a consumed
                        # cursor.
                        chronicle_g0_cursor = source_projection_cursor_factory()
                        native_chronicle.observe_generation_zero(
                            cell_ids=tuple(request.cell_ids),
                            checkpoint_projections=None,
                            checkpoint_projection_cursor=chronicle_g0_cursor,
                            source_projection_sha256_by_cell=initial_digests,
                            source_projection_digest=(
                                self.terminal_projection_digest
                            ),
                        )
                    else:
                        native_chronicle.observe_generation_zero(
                            cell_ids=tuple(request.cell_ids),
                            checkpoint_projections=initial_projections,
                            source_projection_sha256_by_cell=initial_digests,
                            source_projection_digest=(
                                self.terminal_projection_digest
                            ),
                        )
            # Source import for a successor owner must run before reserving a
            # possibly full ACS queue.  In particular, a one-slot route would
            # otherwise wait for the very source lease this preparation owns.
            # Reservation still precedes GPU dispatch, so no completed
            # checkpoint can be silently dropped when backpressure applies.
            if acs_pipeline is not None and full_checkpoint_boundary:
                while len(pending_acs_checkpoints) >= acs_pipeline.slots:
                    finalize_oldest_acs_checkpoint(producer_backpressure=True)
                acs_capture_token = acs_pipeline.begin_capture(
                    compute_frontier=(segment_end_next_tick if tick_native else end)
                )
            if segment_execution_observer is not None:
                segment_execution_observer(
                    phase="prepared",
                    segment_ordinal=segment_ordinal,
                    start_generation=current,
                    end_generation=end,
                    prepared=prepared,
                    direct_result=None,
                )
            request_devices = {
                str(getattr(item, "cuda_device", "")) for item in requests
            }
            tick_window_host_before = None
            tick_window_gpu_before = None
            tick_window_resident_storage = None
            if tick_native:
                measurement_device = (
                    next(iter(request_devices)) if len(request_devices) == 1 else None
                )
                tick_window_host_before = self._acs_process_resource_sample_v1()
                tick_window_gpu_before = self._acs_gpu_resource_sample_v1(
                    measurement_device
                )
                if isinstance(
                    measurement_device, str
                ) and measurement_device.startswith("cuda"):
                    import torch

                    torch.cuda.reset_peak_memory_stats(measurement_device)
                tick_window_resident_storage = self._resident_tensor_storage_bytes_v1(
                    prepared.physical_window_arguments
                )
            option_d_m0_memory_device: str | None = None
            if option_d_m0_source_environment is not None:
                if len(request_devices) != 1:
                    raise CampaignProviderError(
                        "Option D M0 qualification requires one exact execution device"
                    )
                option_d_m0_memory_device = (
                    _option_d_m0_begin_cuda_memory_measurement_v1(
                        execution_device=next(iter(request_devices))
                    )
                )
            executor_arguments = {
                "physical_window_arguments": prepared.physical_window_arguments,
                "logical_generation_count": segment_generations,
                **self._mutation_arguments(prepared),
            }
            if absolute_tick_span is not None:
                if not _callable_accepts_keyword_v1(executor, "absolute_tick_span"):
                    raise CampaignProviderError(
                        "tick-native executor does not accept an absolute tick span"
                    )
                executor_arguments["absolute_tick_span"] = int(absolute_tick_span)
                if not _callable_accepts_keyword_v1(executor, "next_absolute_tick"):
                    raise CampaignProviderError(
                        "tick-native executor does not accept its requested cursor"
                    )
                if not _callable_accepts_keyword_v1(
                    executor, "authoritative_next_absolute_tick"
                ):
                    raise CampaignProviderError(
                        "tick-native executor does not accept resident-owner cursor "
                        "authority"
                    )
                if resident_owner is None:
                    raise CampaignProviderError(
                        "tick-native dispatch lacks its authoritative resident owner"
                    )
                if resident_owner.next_absolute_tick != segment_start_next_tick:
                    raise CampaignProviderError(
                        "tick-native dispatch cursor differs from the resident-owner "
                        "authority"
                    )
                executor_arguments["next_absolute_tick"] = segment_start_next_tick
                executor_arguments["authoritative_next_absolute_tick"] = (
                    resident_owner.next_absolute_tick
                )
                if _callable_accepts_keyword_v1(executor, "tick_native_control"):
                    executor_arguments["tick_native_control"] = dict(
                        request.tick_native_control
                    )
            dispatch_started_perf_ns = time.perf_counter_ns()
            dispatch_started_unix_ns = time.time_ns()
            if successor_checkpoint_capture is not None:
                measurement = successor_checkpoint_capture.get(
                    "capture_barrier_measurement"
                )
                if not isinstance(measurement, dict):
                    raise CampaignProviderError(
                        "detached successor dispatch lost its timing evidence"
                    )
                measurement["successor_dispatch_started_perf_ns"] = (
                    dispatch_started_perf_ns
                )
                measurement["successor_dispatch_coordinate"] = (
                    segment_end_next_tick
                )
            direct_result = executor(module, **executor_arguments)
            if successor_checkpoint_capture is not None:
                measurement = successor_checkpoint_capture.get(
                    "capture_barrier_measurement"
                )
                if not isinstance(measurement, dict):
                    raise CampaignProviderError(
                        "detached successor completion lost its timing evidence"
                    )
                measurement["successor_dispatch_completed_perf_ns"] = (
                    time.perf_counter_ns()
                )
                if not isinstance(direct_result, Mapping):
                    raise CampaignProviderError(
                        "detached successor dispatch lacks native completion evidence"
                    )
                successor_last_useful_kernel_completion_ns = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result,
                        "last_useful_kernel_completion_observed_host_time_ns",
                    )
                )
                successor_final_stream_synchronization_time_ns = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result,
                        "final_stream_synchronization_time_ns",
                    )
                )
                successor_final_host_synchronizations = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result,
                        "final_host_synchronizations",
                    )
                )
                successor_source_window_count = _require_acs_native_nonnegative_int_v1(
                    direct_result,
                    "source_window_count",
                )
                if (
                    successor_last_useful_kernel_completion_ns == 0
                    or successor_source_window_count < 1
                    or successor_final_host_synchronizations
                    != successor_source_window_count
                ):
                    raise CampaignProviderError(
                        "detached successor native completion evidence is invalid"
                    )
                measurement[
                    "successor_last_useful_kernel_completion_observed_host_time_ns"
                ] = successor_last_useful_kernel_completion_ns
                measurement[
                    "successor_final_stream_synchronization_time_ns"
                ] = successor_final_stream_synchronization_time_ns
                measurement["successor_final_host_synchronizations"] = (
                    successor_final_host_synchronizations
                )
                measurement["successor_source_window_count"] = (
                    successor_source_window_count
                )
            if (
                tick_native
                and owner_transition
                and segment_end_next_tick < int(request.total_ticks or 0)
            ):
                observed_resident_owner_transition_ticks.append(
                    segment_end_next_tick
                )
            if tick_native and isinstance(direct_result, Mapping):
                if resident_owner is None:
                    raise CampaignProviderError(
                        "tick-native dispatch completed without a resident owner"
                    )
                # Bind the provider cursor even for adapters whose native
                # result predates the explicit cursor receipt.  This is
                # metadata only; tensor-bearing result values remain owned by
                # the native boundary and are never copied into compact
                # settlement state.
                direct_result = {
                    **dict(direct_result),
                    "tick_native_source_prefix_by_cell": tuple(
                        {
                            "metadata": dict(prefix["metadata"]),
                            "runtime": dict(prefix["runtime"]),
                        }
                        for prefix in resident_owner.tick_native_prefix_by_cell
                    ),
                    "tick_native_generation_base": (
                        segment_start_next_tick // interval_ticks
                    ),
                    "tick_native_start_next_tick": segment_start_next_tick,
                    "tick_native_end_next_tick": segment_end_next_tick,
                }
            _validate_tick_native_direct_event_storage_v1(
                direct_result, tick_native=tick_native
            )
            tick_window_host_after = (
                self._acs_process_resource_sample_v1() if tick_native else None
            )
            tick_window_gpu_after = (
                self._acs_gpu_resource_sample_v1(
                    next(iter(request_devices)) if len(request_devices) == 1 else None
                )
                if tick_native
                else None
            )
            dispatch_finished_unix_ns = time.time_ns()
            dispatch_finished_perf_ns = time.perf_counter_ns()
            direct_option_d_m0_telemetry_enabled = (
                direct_result.get("option_d_m0_telemetry_enabled", False)
                if isinstance(direct_result, Mapping)
                else False
            )
            if (
                option_d_m0_source_environment is not None
                and direct_option_d_m0_telemetry_enabled is not True
            ):
                raise CampaignProviderError(
                    "Option D M0 qualification artifact did not emit enabled native "
                    "telemetry"
                )
            if (
                direct_option_d_m0_telemetry_enabled is True
                and option_d_m0_source_environment is None
            ):
                raise CampaignProviderError(
                    "Option D M0 native telemetry requires the compiled "
                    "source/environment qualification artifact"
                )
            explicit_native_no_fallback = (
                isinstance(direct_result, Mapping)
                and direct_result.get("owner") == "native_extension"
                and direct_result.get("hidden_fallback") is False
                and direct_result.get("runtime_jit") is False
                and len(request_devices) == 1
                and next(iter(request_devices)).startswith("cuda")
            )
            if request.profile_id == PRODUCTION_CPU_PROFILE_ID:
                explicit_native_no_fallback = (
                    isinstance(direct_result, Mapping)
                    and direct_result.get("owner") == "cpu_native_extension"
                    and direct_result.get("hidden_fallback") is False
                    and direct_result.get("runtime_jit") is False
                    and len(request_devices) == 1
                    and next(iter(request_devices)) == "cpu"
                )
            if "cross_path_hidden_fallback" in direct_result:
                explicit_native_no_fallback = (
                    explicit_native_no_fallback
                    and direct_result.get("cross_path_hidden_fallback") is False
                )
            native_receipt = {
                "segment_ordinal": segment_ordinal,
                "start_generation": current,
                "end_generation": end,
                "owner_identity": (
                    resident_owner.prepared_owner_identity
                    if resident_owner is not None
                    else None
                ),
                "owner_span_start_tick": (
                    owner_span_start_tick if tick_native else None
                ),
                "owner_span_end_tick": owner_span_end_tick if tick_native else None,
                "segment_dispatch_started_unix_ns": dispatch_started_unix_ns,
                "segment_dispatch_finished_unix_ns": dispatch_finished_unix_ns,
                "segment_dispatch_started_perf_ns": dispatch_started_perf_ns,
                "segment_dispatch_finished_perf_ns": dispatch_finished_perf_ns,
                "last_useful_kernel_completion_observed_host_time_ns": (
                    direct_result.get(
                        "last_useful_kernel_completion_observed_host_time_ns"
                    )
                    if isinstance(direct_result, Mapping)
                    else None
                ),
                "execution_device": (
                    next(iter(request_devices)) if len(request_devices) == 1 else None
                ),
                "native_owner": (
                    direct_result.get("owner")
                    if isinstance(direct_result, Mapping)
                    else None
                ),
                "hidden_fallback": (
                    direct_result.get("hidden_fallback")
                    if isinstance(direct_result, Mapping)
                    else None
                ),
                "runtime_jit": (
                    direct_result.get("runtime_jit")
                    if isinstance(direct_result, Mapping)
                    else None
                ),
                "explicit_native_no_fallback": explicit_native_no_fallback,
            }
            if tick_native:
                from engine.perf.cuda_architecture_b_event_sink_v1 import (
                    EVENT_SINK_ROW_BYTES,
                )

                drain_receipts = (
                    direct_result.get("event_drain_receipts", [])
                    if isinstance(direct_result, Mapping)
                    else []
                )
                drain_counts = [
                    int(count)
                    for receipt in drain_receipts
                    if isinstance(receipt, Mapping)
                    for count in receipt.get("active_count_before", [])
                    if type(count) is int and count >= 0
                ]
                drain_occupancies = [
                    sum(
                        int(count)
                        for count in receipt.get("active_count_before", [])
                        if type(count) is int and count >= 0
                    )
                    for receipt in drain_receipts
                    if isinstance(receipt, Mapping)
                ]
                frontiers = (
                    direct_result.get("event_history_frontier_by_cell", [])
                    if isinstance(direct_result, Mapping)
                    else []
                )
                pending = (
                    direct_result.get("event_pending_rows_by_cell", [])
                    if isinstance(direct_result, Mapping)
                    else []
                )
                gpu_samples = [
                    row
                    for row in (tick_window_gpu_before, tick_window_gpu_after)
                    if isinstance(row, Mapping)
                ]
                host_samples = [
                    row
                    for row in (tick_window_host_before, tick_window_host_after)
                    if isinstance(row, Mapping)
                ]

                def measured_values(
                    rows: Sequence[Mapping[str, Any]], name: str
                ) -> list[int]:
                    return [
                        int(row[name])
                        for row in rows
                        if type(row.get(name)) is int and int(row[name]) >= 0
                    ]

                event_bytes = (
                    len(request.cell_ids)
                    * int(request.event_window_capacity or 0)
                    * int(EVENT_SINK_ROW_BYTES)
                )
                cuda_tensor_bytes = int(
                    (tick_window_resident_storage or {}).get("cuda_tensor_bytes", 0)
                )
                conservative_free = measured_values(gpu_samples, "free_memory_bytes")
                if tick_window_gpu_before and tick_window_gpu_after:
                    before_total = tick_window_gpu_before.get("total_memory_bytes")
                    before_free = tick_window_gpu_before.get("free_memory_bytes")
                    before_reserved = tick_window_gpu_before.get(
                        "memory_reserved_bytes"
                    )
                    peak_reserved = tick_window_gpu_after.get(
                        "max_memory_reserved_bytes"
                    )
                    if all(
                        type(value) is int
                        for value in (
                            before_total,
                            before_free,
                            before_reserved,
                            peak_reserved,
                        )
                    ):
                        external_used = max(
                            0, before_total - before_free - before_reserved
                        )
                        conservative_free.append(
                            max(0, before_total - external_used - peak_reserved)
                        )
                native_receipt.update(
                    {
                        "start_next_tick": segment_start_next_tick,
                        "end_next_tick": segment_end_next_tick,
                        "owner_span_start_tick": segment_spec.get(
                            "owner_span_start_tick"
                        ),
                        "owner_span_end_tick": segment_spec.get("owner_span_end_tick"),
                        "dispatch_window_ticks": request.native_dispatch_window_ticks,
                        "resident_event_capacity": request.event_window_capacity,
                        "fired_schedules": list(
                            segment_spec.get("fired_schedules", ())
                        ),
                        "time_authority": "absolute_tick_v1",
                        "first_executed_absolute_tick": segment_start_next_tick,
                        "first_committed_next_tick": segment_start_next_tick + 1,
                        "resource_window_measurement": {
                            "schema_version": (
                                "eiel_tick_native_resource_window_measurement_v1"
                            ),
                            "start_tick": segment_start_next_tick,
                            "end_tick": segment_end_next_tick,
                            "peak_gpu_allocated_bytes": max(
                                measured_values(
                                    (
                                        [tick_window_gpu_after]
                                        if isinstance(tick_window_gpu_after, Mapping)
                                        else []
                                    ),
                                    "max_memory_allocated_bytes",
                                ),
                                default=0,
                            ),
                            "peak_gpu_reserved_bytes": max(
                                measured_values(
                                    (
                                        [tick_window_gpu_after]
                                        if isinstance(tick_window_gpu_after, Mapping)
                                        else []
                                    ),
                                    "max_memory_reserved_bytes",
                                ),
                                default=0,
                            ),
                            "minimum_free_gpu_bytes": min(
                                conservative_free,
                                default=0,
                            ),
                            "host_peak_rss_bytes": max(
                                measured_values(host_samples, "rss_bytes")
                                + measured_values(host_samples, "rss_high_water_bytes"),
                                default=0,
                            ),
                            "host_terminal_rss_bytes": max(
                                measured_values(
                                    (
                                        [tick_window_host_after]
                                        if isinstance(tick_window_host_after, Mapping)
                                        else []
                                    ),
                                    "rss_bytes",
                                ),
                                default=0,
                            ),
                            "minimum_host_available_bytes": min(
                                measured_values(host_samples, "host_available_bytes"),
                                default=0,
                            ),
                            "minimum_swap_free_bytes": min(
                                measured_values(host_samples, "swap_free_bytes"),
                                default=0,
                            ),
                            "swap_total_bytes": max(
                                measured_values(host_samples, "swap_total_bytes"),
                                default=0,
                            ),
                            "resident_event_bytes": event_bytes,
                            "resident_source_bytes": max(
                                0, cuda_tensor_bytes - event_bytes
                            ),
                            "resident_all_tensor_bytes": int(
                                (tick_window_resident_storage or {}).get(
                                    "all_tensor_bytes", 0
                                )
                            ),
                            "event_ring_capacity": (
                                request.event_window_capacity * len(request.cell_ids)
                            ),
                            "event_ring_peak_occupancy": max(
                                drain_occupancies,
                                default=0,
                            ),
                            "events_emitted": sum(drain_counts)
                            + sum(
                                int(value)
                                for value in pending
                                if type(value) is int and value >= 0
                            ),
                            "events_drained": sum(drain_counts),
                            "terminal_event_ring_occupancy": sum(
                                int(value)
                                for value in pending
                                if type(value) is int and value >= 0
                            ),
                            "external_event_history_count": sum(
                                int(value)
                                for value in frontiers
                                if type(value) is int and value >= 0
                            ),
                            "gpu_samples": [dict(row) for row in gpu_samples],
                            "host_samples": [dict(row) for row in host_samples],
                        },
                    }
                )
            if isinstance(direct_result, Mapping):
                _project_option_d_m0_provider_receipts_v1(
                    direct_result=direct_result,
                    native_receipt=native_receipt,
                )
            native_execution_receipts.append(native_receipt)
            if segment_execution_observer is not None:
                segment_execution_observer(
                    phase="executed",
                    segment_ordinal=segment_ordinal,
                    start_generation=current,
                    end_generation=end,
                    prepared=prepared,
                    direct_result=direct_result,
                )
            if qualification_sentinels is not None:
                observed = _qualification_transition_sentinels_v1(
                    direct_result,
                    request,
                    start_generation=current,
                    segment_generations=segment_generations,
                    start_next_tick=(segment_start_next_tick if tick_native else None),
                    end_next_tick=(segment_end_next_tick if tick_native else None),
                    restored_entry=(
                        tick_native
                        and execution_start_next_tick > 0
                        and segment_start_next_tick == execution_start_next_tick
                    ),
                )
                for cell_id in request.cell_ids:
                    qualification_sentinels[cell_id].extend(observed[cell_id])
            dispatches += 1
            if not full_checkpoint_boundary:
                if tick_native:
                    if resident_owner is None:
                        raise CampaignProviderError(
                            "tick-native compact dispatch lacks its resident owner"
                        )
                    _advance_tick_native_source_prefix_v1(
                        resident_owner,
                        direct_result=direct_result,
                        start_next_tick=segment_start_next_tick,
                        end_next_tick=segment_end_next_tick,
                        interval_ticks=interval_ticks,
                    )
                compact_settlement = _compact_segment_settlement_v1(
                    cell_ids=tuple(request.cell_ids),
                    start_next_tick=segment_start_next_tick,
                    end_next_tick=segment_end_next_tick,
                    owner_identity=(
                        resident_owner.prepared_owner_identity
                        if resident_owner is not None
                        else "legacy-segment"
                    ),
                    direct_result=direct_result,
                    native_execution_receipt=native_receipt,
                )
                compact_settlements.append(compact_settlement)
                native_receipt["compact_settlement"] = {
                    "schema_version": compact_settlement.schema_version,
                    "start_next_tick": compact_settlement.start_next_tick,
                    "end_next_tick": compact_settlement.end_next_tick,
                    "owner_identity": compact_settlement.owner_identity,
                    "event_frontier_by_cell": list(
                        compact_settlement.event_frontier_by_cell
                    ),
                    "event_pending_rows_by_cell": list(
                        compact_settlement.event_pending_rows_by_cell
                    ),
                }
                if segment_execution_observer is not None:
                    segment_execution_observer(
                        phase="settled",
                        segment_ordinal=segment_ordinal,
                        start_generation=current,
                        end_generation=end,
                        prepared=prepared,
                        direct_result=direct_result,
                        compact_settlement=compact_settlement,
                    )
                if resident_owner is not None:
                    resident_owner.next_absolute_tick = segment_end_next_tick
                    transfer_owner_epoch_at_boundary(
                        resident_owner,
                        boundary_next_tick=segment_end_next_tick,
                        owner_transition=owner_transition,
                    )
                current_next_tick = segment_end_next_tick
                executor_arguments.clear()
                abi_owner = None
                post_dispatch_owner = None
                del direct_result
                gc.collect()
                continue
            capability = requests[0].cuda_capability
            # A tick-native owner boundary is a full bounded capture even when
            # it is not one of the route's durable checkpoint milestones.  The
            # synchronous slot-zero cursor path still emits the bounded ACS
            # clock and therefore needs the same start/resource authorities.
            boundary_acs_enabled = acs_measurement_enabled and full_checkpoint_boundary
            acs_host_resources_before = (
                self._acs_process_resource_sample_v1()
                if boundary_acs_enabled
                else None
            )
            acs_gpu_resources_before = (
                self._acs_gpu_resource_sample_v1(native_receipt["execution_device"])
                if boundary_acs_enabled
                else None
            )
            acs_capture_cpu_started_ns = (
                time.process_time_ns() if boundary_acs_enabled else None
            )
            if request.native_streamed_checkpoint:
                _mark_native_streamed_capture_owner_v1(prepared)
            checkpoint_projection_started = time.perf_counter_ns()
            # The bounded projection owns its clock independently of ACS
            # instrumentation.  Route-qualification controls may legitimately
            # disable ACS measurements while retaining the same bounded cursor
            # and persistence path, so borrowing this authority from the ACS
            # flag would produce a half-bound capture receipt.
            checkpoint_projection_started_unix_ns = time.time_ns()
            acs_capture_started_unix_ns = (
                checkpoint_projection_started_unix_ns
                if boundary_acs_enabled
                else None
            )
            projector_arguments: dict[str, Any] = {
                "direct_result": direct_result,
                "capability": capability,
                "reporting_enabled": (
                    request.reporting_enabled and absolute_tick_span is None
                ),
            }
            if tick_native and _callable_accepts_keyword_v1(
                projector, "generation_base_override"
            ):
                projector_arguments["generation_base_override"] = (
                    segment_start_next_tick // interval_ticks
                )
            if tick_native and _callable_accepts_keyword_v1(
                projector, "bounded_projection_cursor"
            ):
                projector_arguments["bounded_projection_cursor"] = True
            if (
                physical_batch_detached_capture_route
                or terminal_physical_capture_route
            ):
                # The canonical B1 projector runs on the detached worker from
                # one physical field-major snapshot.  A non-None sentinel
                # preserves the existing bounded-cursor control flow without
                # executing 136 producer-side settlements.
                settlement = SimpleNamespace(
                    terminal_generation=end,
                    terminal_tick=terminal_tick,
                    checkpoint_projection_cursor=None,
                    checkpoint_projections=(),
                    compact_result_rows=(),
                    lifecycle_qacct_receipt=None,
                    lifecycle_qacct_receipts=(),
                    segment_artifacts=(),
                    analysis_source_rows=(),
                )
                projection_cursor = object()
            else:
                settlement = projector(prepared, **projector_arguments)
                projection_cursor = getattr(
                    settlement, "checkpoint_projection_cursor", None
                )
            if bounded_terminal_capture_route and projection_cursor is None:
                raise CampaignProviderError(
                    "tick-native bounded terminal capture requires a projection cursor"
                )
            # Slot-zero capture is synchronous and bounded at every declared
            # absolute-tick checkpoint.  The cursor/persistence contract is
            # coordinate-independent; restricting it to the terminal tick
            # would reject the intermediate owner-boundary checkpoints used
            # by the tick-native canary and the long-horizon campaign.
            if tick_native and (
                getattr(settlement, "terminal_generation", None) != end
                or getattr(settlement, "terminal_tick", 0) != terminal_tick
            ):
                raise CampaignProviderError(
                    "settled absolute-tick cursor differs from the requested span"
                )
            if resident_owner is not None:
                if tick_native:
                    _advance_tick_native_source_prefix_v1(
                        resident_owner,
                        direct_result=direct_result,
                        start_next_tick=segment_start_next_tick,
                        end_next_tick=segment_end_next_tick,
                        interval_ticks=interval_ticks,
                    )
                resident_owner.next_absolute_tick = segment_end_next_tick
            qacct_receipt = getattr(settlement, "lifecycle_qacct_receipt", None)
            if isinstance(qacct_receipt, Mapping):
                lifecycle_qacct_receipts.append(dict(qacct_receipt))
            qacct_receipts = getattr(settlement, "lifecycle_qacct_receipts", ())
            if isinstance(qacct_receipts, Sequence) and not isinstance(
                qacct_receipts, (str, bytes)
            ):
                lifecycle_qacct_receipts.extend(
                    dict(receipt)
                    for receipt in qacct_receipts
                    if isinstance(receipt, Mapping)
                )
            checkpoint_projection_time_ns = (
                time.perf_counter_ns() - checkpoint_projection_started
            )
            if (
                native_receipt.get("option_d_m0_telemetry_enabled") is True
                or boundary_acs_enabled
            ):
                native_receipt["checkpoint_projection_time_ns"] = (
                    checkpoint_projection_time_ns
                )
            if segment_execution_observer is not None:
                segment_execution_observer(
                    phase="settled",
                    segment_ordinal=segment_ordinal,
                    start_generation=current,
                    end_generation=end,
                    prepared=prepared,
                    direct_result=direct_result,
                )
            segment_artifacts = (
                ()
                if projection_cursor is not None
                else tuple(getattr(settlement, "segment_artifacts", ()))
            )
            current_artifacts = [
                dict(artifact)
                for artifact in segment_artifacts
                if isinstance(artifact, Mapping)
            ]
            current_segment_artifacts.extend(current_artifacts)
            analysis_source_rows = tuple(
                getattr(settlement, "analysis_source_rows", ())
            )
            if analysis_source_rows:
                if authoritative_identity_bindings is None:
                    raise CampaignProviderError(
                        "analysis source rows lack authoritative identity bindings"
                    )
                from engine.campaigns.authoritative_state_analysis_projection_v1 import (
                    authoritative_analysis_source_payload_v1,
                    write_immutable_analysis_source_v1,
                )

                for raw_source in analysis_source_rows:
                    if not isinstance(raw_source, Mapping):
                        raise CampaignProviderError(
                            "analysis source row must be an object"
                        )
                    required_source_fields = {
                        "cell_id",
                        "start_generation",
                        "terminal_generation",
                        "population_semantics",
                        "row_agent_ids",
                        "source_values",
                    }
                    if set(raw_source) != required_source_fields:
                        raise CampaignProviderError(
                            "analysis source row field set differs"
                        )
                    cell_id = raw_source["cell_id"]
                    if not isinstance(cell_id, str) or cell_id not in request.cell_ids:
                        raise CampaignProviderError(
                            "analysis source cell differs from resident request"
                        )
                    payload = authoritative_analysis_source_payload_v1(
                        source_values=raw_source["source_values"],
                        identity_bindings=authoritative_identity_bindings,
                        source_coordinates={
                            "cell_id": cell_id,
                            "start_generation": raw_source["start_generation"],
                            "terminal_generation": raw_source["terminal_generation"],
                            "population_semantics": raw_source["population_semantics"],
                        },
                        row_agent_ids=raw_source["row_agent_ids"],
                    )
                    analysis_path = (
                        context.batch_root
                        / "cells"
                        / cell_id
                        / "authoritative_state"
                        / "analysis_sources"
                        / (
                            f"segment-{segment_ordinal:06d}"
                            f"-g{current:06d}-g{end:06d}.json"
                        )
                    )
                    source_payload_sha256 = write_immutable_analysis_source_v1(
                        analysis_path, payload
                    )
                    authoritative_analysis_sources.append(
                        {
                            "schema_version": (
                                "eiel_authoritative_analysis_source_receipt_v1"
                            ),
                            "cell_id": cell_id,
                            "segment_ordinal": segment_ordinal,
                            "start_generation": current,
                            "terminal_generation": end,
                            "population_semantics": raw_source["population_semantics"],
                            "row_count": len(raw_source["row_agent_ids"]),
                            "path": str(analysis_path),
                            "source_payload_sha256": source_payload_sha256,
                        }
                    )
            if (
                preloaded_artifact is None
                and request.reporting_enabled
                and self._generation_series_required(plan)
                and callable(getattr(self, "project_generation_series", None))
            ):
                generation_series_artifacts.extend(
                    dict(artifact) for artifact in current_artifacts
                )
            if segment_artifacts and native_chronicle is not None:
                if projection_cursor is not None:
                    # The terminal checkpoint cursor is consumed by the
                    # checkpoint writer below.  Chronicle is bound after that
                    # publication from a fresh one-cell cursor over the
                    # durable bundles; consuming this cursor here would make
                    # the checkpoint itself unavailable.
                    pass
                else:
                    native_chronicle.observe_segment(
                        segment_ordinal=segment_ordinal,
                        start_generation=current,
                        end_generation=end,
                        cell_ids=tuple(request.cell_ids),
                        segment_artifacts=segment_artifacts,
                        checkpoint_projections=tuple(
                            settlement.checkpoint_projections
                        ),
                    )
                    native_chronicle_observed = True
            if segment_artifact_observer is not None and projection_cursor is None:
                segment_artifact_observer(
                    segment_ordinal=segment_ordinal,
                    start_generation=current,
                    end_generation=end,
                    cell_ids=tuple(request.cell_ids),
                    segment_artifacts=segment_artifacts,
                    checkpoint_projections=(
                        ()
                        if projection_cursor is not None
                        else tuple(settlement.checkpoint_projections)
                    ),
                )
            prepared_source_projections = tuple(
                getattr(prepared, "source_projections", ())
            )
            if (
                not prepared_source_projections
                and bounded_streamed_preparation
            ):
                source_import_projection_digest_sha256s = {
                    cell_id: str(
                        getattr(
                            request_descriptor,
                            "source_import_projection_digest_sha256",
                            "",
                        )
                    )
                    for cell_id, request_descriptor in zip(
                        request.cell_ids, requests, strict=True
                    )
                }
                if any(
                    not _valid_sha256(value)
                    for value in source_import_projection_digest_sha256s.values()
                ):
                    raise CampaignProviderError(
                        "streamed owner lacks exact source import projection digests"
                    )
            else:
                if len(prepared_source_projections) != len(request.cell_ids):
                    raise CampaignProviderError(
                        "prepared owner lacks ordered source import projections"
                    )
                source_import_projection_digest_sha256s = {
                    cell_id: (
                        str(request.source_import_projection_digest_sha256)
                        if isinstance(
                            getattr(
                                request, "source_import_projection_digest_sha256", None
                            ),
                            str,
                        )
                        else self.terminal_projection_digest(projection)
                    )
                    for cell_id, request, projection in zip(
                        request.cell_ids,
                        requests,
                        prepared_source_projections,
                        strict=True,
                    )
                }
            parent_checkpoint_projection_digest_sha256s: dict[str, str] | None = None
            parent_checkpoint = checkpoint_catalog.get(
                str(segment_start_next_tick if tick_native else current)
            )
            if isinstance(parent_checkpoint, Mapping):
                raw_parent = parent_checkpoint.get(
                    "terminal_projection_digest_sha256s"
                ) or parent_checkpoint.get("terminal_cell_digests")
                if not isinstance(raw_parent, Mapping):
                    raise CampaignProviderError(
                        "parent checkpoint lacks projection digest authority"
                    )
                parent_checkpoint_projection_digest_sha256s = {
                    str(cell_id): str(digest) for cell_id, digest in raw_parent.items()
                }
            if projection_cursor is not None:
                source_projections = None
            else:
                source_projections = tuple(settlement.checkpoint_projections)
            if tick_native:
                from engine.perf.cuda_architecture_b_event_sink_v1 import (
                    event_history_digest_from_references_v1,
                    event_sink_prefix_identities_v1,
                )

                if projection_cursor is not None:
                    reference = direct_result.get("event_history_reference")
                    frontiers = direct_result.get(
                        "event_history_frontier_by_cell"
                    )
                    if (
                        not isinstance(reference, Mapping)
                        or type(reference.get("path")) is not str
                        or not isinstance(frontiers, Sequence)
                        or isinstance(frontiers, (str, bytes, bytearray))
                        or len(frontiers) != len(request.cell_ids)
                        or any(type(value) is not int or value < 0 for value in frontiers)
                    ):
                        raise CampaignProviderError(
                            "tick-native bounded settlement lacks authenticated event history"
                        )
                    try:
                        identities = event_sink_prefix_identities_v1(
                            reference["path"],
                            expected_rows_by_cell={
                                cell: int(frontiers[cell])
                                for cell in range(len(request.cell_ids))
                            },
                        )
                        event_references = [
                            dict(identities[cell])
                            for cell in range(len(request.cell_ids))
                        ]
                    except Exception as exc:
                        raise CampaignProviderError(
                            f"cannot authenticate bounded event history: {exc}"
                        ) from exc
                else:
                    event_references = []
                    for projection in source_projections:
                        metadata = (
                            projection.get("metadata")
                            if isinstance(projection, Mapping)
                            else None
                        )
                        reference = (
                            metadata.get("event_history_reference")
                            if isinstance(metadata, Mapping)
                            else None
                        )
                        if not isinstance(reference, Mapping):
                            raise CampaignProviderError(
                                "tick-native settlement lacks authenticated event history"
                            )
                        event_references.append(dict(reference))
                direct_result = {
                    **dict(direct_result),
                    "event_history_prefix_references_by_cell": event_references,
                }
                native_receipt["event_history_prefix_references_by_cell"] = (
                    event_references
                )
                native_receipt["external_event_history_digest"] = (
                    event_history_digest_from_references_v1(event_references)
                )
            projected_rows = [dict(row) for row in settlement.compact_result_rows]
            if (
                production_parameter_contracts is not None
                and absolute_tick_span is None
                and getattr(self, "authoritative_state_schema_required_v1", False)
                is True
                and not hapb_host_authoritative_policy_binding_v1(mechanics_binding)
            ):
                from engine.authoritative_state.runtime_abi import (
                    observe_applied_direct_resident_abi_v1,
                    resolve_prepared_direct_resident_trajectory_v1,
                )
                from engine.campaigns.orchestration import atomic_write_json
                from engine.campaigns.production_control_path_v1 import (
                    build_segment_control_readback_v1,
                )

                post_dispatch_owner = resolve_prepared_direct_resident_trajectory_v1(
                    prepared
                )
                post_dispatch_abi_receipt = observe_applied_direct_resident_abi_v1(
                    post_dispatch_owner
                )
                authoritative_state_receipts.append(
                    {
                        "schema_version": (
                            "eiel_authoritative_state_post_dispatch_receipt_v1"
                        ),
                        "state_schema_sha256": (
                            dict(authoritative_identity_bindings or {}).get(
                                "authoritative_schema_hash"
                            )
                        ),
                        "segment_ordinal": segment_ordinal,
                        **(
                            {
                                "start_next_tick": segment_start_next_tick,
                                "end_next_tick": segment_end_next_tick,
                            }
                            if tick_native
                            else {}
                        ),
                        "start_generation": current,
                        "end_generation": end,
                        "runtime_abi_receipt_sha256": post_dispatch_abi_receipt[
                            "receipt_sha256"
                        ],
                        "identity_bindings": dict(
                            authoritative_identity_bindings or {}
                        ),
                    }
                )
                try:
                    segment_control_readback = build_segment_control_readback_v1(
                        profile_id=request.profile_id,
                        segment_ordinal=segment_ordinal,
                        start_generation=current,
                        end_generation=end,
                        cells=cells,
                        builds=builds,
                        requests=requests,
                        prepared=prepared,
                        checkpoint_projections=(
                            source_projections
                            if source_projections is not None
                            else ()
                        ),
                        compact_result_rows=projected_rows,
                        parameter_contracts=production_parameter_contracts,
                        post_dispatch_abi_receipt=post_dispatch_abi_receipt,
                    )
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise CampaignProviderError(
                        f"production control readback failed closed: {exc}"
                    ) from exc
                receipt_path = (
                    context.batch_root
                    / "control_path_receipts"
                    / (
                        f"segment-{segment_ordinal:06d}"
                        f"-g{current:06d}-g{end:06d}.json"
                    )
                )
                atomic_write_json(receipt_path, segment_control_readback)
                descriptor = {
                    "schema_version": "eiel_production_control_readback_descriptor_v1",
                    "path": str(receipt_path),
                    "receipt_sha256": segment_control_readback["receipt_sha256"],
                    "post_dispatch_abi_receipt_sha256": (
                        post_dispatch_abi_receipt["receipt_sha256"]
                    ),
                }
                native_receipt["production_control_readback"] = dict(descriptor)
                production_control_readbacks.append(
                    {
                        **descriptor,
                        "segment_ordinal": segment_ordinal,
                        "start_generation": current,
                        "end_generation": end,
                    }
                )
                bindings_by_cell = {
                    str(row["cell_id"]): str(row["control_bindings_sha256"])
                    for row in segment_control_readback["cells"]
                }
                for row in projected_rows:
                    cell_id = str(row.get("cell_id", row.get("cell", "")))
                    if cell_id not in bindings_by_cell:
                        raise CampaignProviderError(
                            "production metric row is not bound to a control receipt"
                        )
                    row["production_control_bindings_sha256"] = bindings_by_cell[
                        cell_id
                    ]
                    row["production_control_readback_sha256"] = (
                        segment_control_readback["receipt_sha256"]
                    )
                    row["production_control_readback_status"] = "complete"
            # A bounded tick-native projection cursor publishes its compact
            # rows only after the one-cell checkpoint writer has consumed and
            # sealed it.  Do not merge the intentionally empty pre-persist
            # facade against the previous terminal snapshot; the aggregate
            # below is the authoritative projected row set for this boundary.
            if projection_cursor is None:
                if getattr(self, "terminal_snapshot_metrics", False) or getattr(
                    mechanics_binding,
                    "terminal_snapshot_metrics",
                    False,
                ):
                    if metrics_rows and callable(terminal_snapshot_metrics_merger):
                        metrics_rows = [
                            dict(row)
                            for row in terminal_snapshot_metrics_merger(
                                metrics_rows,
                                projected_rows,
                            )
                        ]
                    else:
                        metrics_rows = projected_rows
                else:
                    metrics_rows.extend(projected_rows)
            receipts.extend(dict(build.receipt) for build in builds)
            if not full_checkpoint_boundary:
                current_next_tick = segment_end_next_tick
                executor_arguments.clear()
                abi_owner = None
                post_dispatch_owner = None
                del builds, requests, prepared, direct_result, settlement
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
                continue
            immutable_source_projections: tuple[Mapping[str, Any], ...] | None = None
            immutable_snapshot_digest_sha256s: dict[str, str] | None = None
            capture_measurement: dict[str, Any] | None = None
            captured_bytes: int | None = None
            detached_capture: dict[str, Any] | None = None
            deferred_terminal_capture: dict[str, Any] | None = None
            detached_authorities: dict[str, Any] | None = None
            direct_physical_receipt: dict[str, Any] | None = None
            if boundary_acs_enabled and projection_cursor is None:
                immutable_source_projections = tuple(source_projections)
                if len(immutable_source_projections) != len(request.cell_ids):
                    raise CampaignProviderError(
                        "ACS checkpoint measurement snapshot does not cover the batch"
                    )
                snapshot_digest_started_ns = time.perf_counter_ns()
                immutable_snapshot_digest_sha256s = {
                    cell_id: self.terminal_projection_digest(projection)
                    for cell_id, projection in zip(
                        request.cell_ids,
                        immutable_source_projections,
                        strict=True,
                    )
                }
                snapshot_digest_time_ns = (
                    time.perf_counter_ns() - snapshot_digest_started_ns
                )
                host_capture_freeze_time_ns = (
                    time.perf_counter_ns() - checkpoint_projection_started
                )
                capture_cpu_time_ns = (
                    time.process_time_ns() - acs_capture_cpu_started_ns
                    if acs_capture_cpu_started_ns is not None
                    else None
                )
                acs_host_resources_after = self._acs_process_resource_sample_v1()
                acs_gpu_resources_after = self._acs_gpu_resource_sample_v1(
                    native_receipt["execution_device"]
                )
                captured_bytes = self._acs_projection_tensor_bytes(
                    immutable_source_projections
                )
                native_final_sync_time_ns = _require_acs_native_nonnegative_int_v1(
                    direct_result, "final_stream_synchronization_time_ns"
                )
                last_useful_kernel_completion_observed_host_time_ns = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result,
                        "last_useful_kernel_completion_observed_host_time_ns",
                    )
                )
                if last_useful_kernel_completion_observed_host_time_ns == 0:
                    raise CampaignProviderError(
                        "ACS capture native last-useful-kernel boundary must be positive"
                    )
                final_host_synchronizations = _require_acs_native_nonnegative_int_v1(
                    direct_result, "final_host_synchronizations"
                )
                final_device_to_host_operations = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result, "final_device_to_host_operations"
                    )
                )
                compact_settlement_d2h_operations = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result, "compact_settlement_d2h_operations"
                    )
                )
                capture_barrier_time_ns = (
                    host_capture_freeze_time_ns + native_final_sync_time_ns
                )
                reporting_storage = direct_result.get(
                    "compact_summary_reporting_storage"
                )
                if not isinstance(reporting_storage, Mapping):
                    raise CampaignProviderError(
                        "ACS capture native receipt lacks compact reporting storage"
                    )
                final_transfer = reporting_storage.get("final_transfer")
                if not isinstance(final_transfer, Mapping):
                    raise CampaignProviderError(
                        "ACS capture native receipt lacks final transfer contract"
                    )
                final_transfer_bytes = final_transfer.get("bytes")
                observed_host_transfer_count = final_transfer.get(
                    "observed_host_transfer_count"
                )
                final_transfer_status = final_transfer.get("status")
                if type(final_transfer_bytes) is not int or final_transfer_bytes < 0:
                    raise CampaignProviderError(
                        "ACS capture final transfer bytes must be an exact nonnegative integer"
                    )
                if (
                    type(observed_host_transfer_count) is not int
                    or observed_host_transfer_count < 0
                    or observed_host_transfer_count != compact_settlement_d2h_operations
                ):
                    raise CampaignProviderError(
                        "ACS capture final transfer count differs from compact settlement"
                    )
                if final_transfer_status not in {
                    "native_stream_ordered_callback_wired",
                    "disabled",
                }:
                    raise CampaignProviderError(
                        "ACS capture final transfer status is invalid"
                    )
                capture_measurement = {
                    "schema_version": "eiel_acs_capture_barrier_measurement_v1",
                    "capture_completed_unix_ns": time.time_ns(),
                    "capture_completed_perf_ns": time.perf_counter_ns(),
                    "segment_ordinal": segment_ordinal,
                    "coordinate": (segment_end_next_tick if tick_native else end),
                    "owner_transition": owner_transition,
                    "owner_span_start_tick": (
                        owner_span_start_tick if tick_native else None
                    ),
                    "owner_span_end_tick": (
                        owner_span_end_tick if tick_native else None
                    ),
                    "execution_device": native_receipt["execution_device"],
                    "checkpoint_projection_time_ns": (checkpoint_projection_time_ns),
                    "snapshot_digest_time_ns": snapshot_digest_time_ns,
                    "host_capture_freeze_time_ns": (host_capture_freeze_time_ns),
                    "capture_cpu_time_ns": capture_cpu_time_ns,
                    "native_final_stream_synchronization_time_ns": (
                        native_final_sync_time_ns
                    ),
                    "last_useful_kernel_completion_observed_host_time_ns": (
                        last_useful_kernel_completion_observed_host_time_ns
                    ),
                    "capture_barrier_time_ns": capture_barrier_time_ns,
                    "post_sync_gpu_idle_proxy_time_ns": (host_capture_freeze_time_ns),
                    "captured_bytes": captured_bytes,
                    "captured_cell_count": len(immutable_source_projections),
                    "immutable_snapshot_digest_sha256s": dict(
                        immutable_snapshot_digest_sha256s
                    ),
                    "host_resources_before": acs_host_resources_before,
                    "host_resources_peak": dict(acs_host_resources_after),
                    "host_resources_after": acs_host_resources_after,
                    "gpu_resources_before": acs_gpu_resources_before,
                    "gpu_resources_peak": dict(acs_gpu_resources_after),
                    "gpu_resources_after": acs_gpu_resources_after,
                    "native_transfer_receipt": {
                        "final_host_synchronizations": (final_host_synchronizations),
                        "final_device_to_host_operations": (
                            final_device_to_host_operations
                        ),
                        "compact_settlement_d2h_operations": (
                            compact_settlement_d2h_operations
                        ),
                        "final_transfer_bytes": final_transfer_bytes,
                        "final_transfer_contract": dict(final_transfer),
                    },
                    "measurement_scope": (
                        "native_final_sync_plus_provider_projector_settlement_"
                        "postprocessing_and_digest_bound_detached_host_snapshot"
                    ),
                    "limitations": [
                        "D2H substage time is bounded by native final synchronization time rather than separately event-timed",
                        "host freeze is represented by digest binding of detached projections",
                        "post-sync host capture time is a GPU-idle proxy; external utilization telemetry remains required for ACS0 decision evidence",
                    ],
                }
                acs_capture_measurements.append(capture_measurement)

            if (
                acs_pipeline is not None or terminal_physical_capture_route
            ) and full_checkpoint_boundary:
                if acs_pipeline is not None and acs_capture_token is None:
                    raise CampaignProviderError(
                        "ACS checkpoint capture lacks a reserved slot"
                    )
                if not (
                    (
                        tick_native_detached_acs
                        or terminal_physical_capture_route
                    )
                    and projection_cursor is not None
                ) and (
                    immutable_source_projections is None
                    or immutable_snapshot_digest_sha256s is None
                    or capture_measurement is None
                    or captured_bytes is None
                ):
                    raise CampaignProviderError(
                        "ACS checkpoint staging lacks capture instrumentation"
                    )
                if (
                    physical_batch_detached_capture_route
                    or terminal_physical_capture_route
                ):
                    if terminal_physical_capture_route and (
                        native_chronicle is not None
                        or callable(component_checkpoint_exporter)
                    ):
                        raise CampaignProviderError(
                            "direct physical terminal capture is not admitted with "
                            "Chronicle or component checkpoint exporters"
                        )
                    checkpoint_name = (
                        f"checkpoint_tick{segment_end_next_tick:012d}.acsaggregate"
                    )
                    destination = (
                        context.batch_root
                        / (
                            "checkpoint_world_sharded_v1"
                            if request.world_sharded_checkpoint
                            else "checkpoint_native_aggregate_v1"
                        )
                        / (
                            f"checkpoint_tick{segment_end_next_tick:012d}"
                            if request.world_sharded_checkpoint
                            else checkpoint_name
                        )
                    )
                    raw_destination = (
                        context.batch_root
                        / "checkpoint_physical_capture_v1"
                        / f"checkpoint_tick{segment_end_next_tick:012d}.acspb"
                    )
                    parameters = dict(getattr(request, "parameter_arrays", ()))
                    staging_values = parameters.get(
                        "cuda_architecture_b.cpu_request_staging_enable"
                    )
                    if (
                        not isinstance(staging_values, tuple)
                        or len(staging_values) != len(request.cell_ids)
                        or any(value is not False for value in staging_values)
                    ):
                        raise CampaignProviderError(
                            "physical B136 capture requires explicit no-staging authority"
                        )
                    if resident_owner is None:
                        raise CampaignProviderError(
                            "physical B136 capture lacks its resident owner identity"
                        )
                    _synchronize_detached_acs_device_boundary_v1(
                        native_receipt.get("execution_device")
                    )
                    capture_started_perf_ns = time.perf_counter_ns()
                    from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
                        build_direct_resident_physical_capture_v1,
                    )

                    detached_capture = build_direct_resident_physical_capture_v1(
                        prepared,
                        direct_result=direct_result,
                        raw_destination=raw_destination,
                        aggregate_destination=destination,
                        provider_id=self.provider_id,
                        batch_id=request.batch_id,
                        segment_ordinal=segment_ordinal,
                        generation=end,
                        tick=terminal_tick,
                        absolute_next_tick=segment_end_next_tick,
                        physical_owner_identity=(
                            resident_owner.prepared_owner_identity
                        ),
                        reporting_enabled=bool(
                            projector_arguments["reporting_enabled"]
                        ),
                        cpu_request_staging_enable=False,
                    )
                    if request.world_sharded_checkpoint:
                        detached_capture["world_sharded_checkpoint"] = True
                        detached_capture["world_sharded_destination_root"] = str(
                            destination
                        )
                        # Indexed source projection removes zero/alias reads,
                        # and Zstandard releases the GIL during bounded field
                        # encoding.  Two workers per admitted host CPU keep
                        # storage/hash work overlapped without changing the
                        # fixed B136 memory bound.
                        detached_capture["world_sharded_worker_count"] = 32
                    capture_completed_perf_ns = time.perf_counter_ns()
                    captured_bytes = int(detached_capture["logical_bytes"])
                    pending_terminal_digests = {
                        cell_id: "pending_host_settlement_v1"
                        for cell_id in request.cell_ids
                    }
                    immutable_snapshot_digest_sha256s = {}
                    capture_measurement = self._acs_detached_capture_measurement_v1(
                        native_receipt=native_receipt,
                        direct_result=direct_result,
                        checkpoint_projection_time_ns=checkpoint_projection_time_ns,
                        capture_started_unix_ns=acs_capture_started_unix_ns,
                        capture_started_perf_ns=capture_started_perf_ns,
                        capture_completed_perf_ns=capture_completed_perf_ns,
                        capture_cpu_started_ns=acs_capture_cpu_started_ns,
                        coordinate=segment_end_next_tick,
                        captured_bytes=captured_bytes,
                        source_projection_sha256s=pending_terminal_digests,
                        host_resources_before=acs_host_resources_before,
                        gpu_resources_before=acs_gpu_resources_before,
                    )
                    capture_measurement.update(
                        {
                            "segment_ordinal": segment_ordinal,
                            "owner_transition": owner_transition,
                            "owner_span_start_tick": owner_span_start_tick,
                            "owner_span_end_tick": owner_span_end_tick,
                            "immutable_snapshot_digest_sha256s": {},
                            "terminal_digest_authority_state": (
                                "pending_host_settlement_v1"
                            ),
                            "physical_batch_capture_descriptor": {
                                "schema_version": detached_capture[
                                    "schema_version"
                                ],
                                "raw_payload_bytes": detached_capture[
                                    "packed_descriptor"
                                ]["payload_bytes"],
                                "slab_pool": dict(
                                    detached_capture["native_slab_pool"]
                                ),
                            },
                        }
                    )
                    acs_capture_measurements.append(capture_measurement)
                    # The placeholder carries no reporting authority.  The
                    # detached world writer reduces terminal metrics directly
                    # from each already-read physical world field and returns
                    # exact ordered rows after publication.
                    cursor_capture = {"compact_result_rows": ()}
                if (
                    tick_native_detached_acs
                    and projection_cursor is not None
                    and not physical_batch_detached_capture_route
                ):
                    from engine.campaigns.acs_native_detached_slots_v1 import (
                        capture_acs_native_aggregate_payload_from_cursor_v1,
                    )

                    checkpoint_name = (
                        f"checkpoint_tick{segment_end_next_tick:012d}.acsaggregate"
                        if tick_native
                        else f"checkpoint_gen{end:06d}.acsaggregate"
                    )
                    destination = (
                        context.batch_root
                        / "checkpoint_native_aggregate_v1"
                        / checkpoint_name
                    )
                    # The native producer owns the resident tensors until the
                    # detached cursor has finished consuming them.  Establish
                    # the producer/device boundary before projection creation
                    # and before any cursor cell can be released.  A fence
                    # placed after cursor capture is too late to order the
                    # producer against the copy stream.
                    _synchronize_detached_acs_device_boundary_v1(
                        native_receipt.get("execution_device")
                    )
                    capture_started_perf_ns = time.perf_counter_ns()
                    authority_factory = self._aggregate_checkpoint_authority_factory_v1(
                        request=request,
                        generation=end,
                        tick=terminal_tick,
                        absolute_next_tick=(
                            segment_end_next_tick if tick_native else None
                        ),
                        segment_ordinal=segment_ordinal,
                        expected_source_projection_digest_sha256s=None,
                    )
                    cursor_capture = (
                        capture_acs_native_aggregate_payload_from_cursor_v1(
                            projection_cursor,
                            destination,
                            cell_ids=request.cell_ids,
                            generation=end,
                            tick=terminal_tick,
                            absolute_next_tick=(
                                segment_end_next_tick if tick_native else None
                            ),
                            batch_id=request.batch_id,
                            segment_ordinal=segment_ordinal,
                            authority_factory=authority_factory,
                            require_native_extension=(
                                request.profile_id == PRODUCTION_PROFILE_ID
                            ),
                        )
                    )
                    capture_completed_perf_ns = time.perf_counter_ns()
                    detached_capture = dict(cursor_capture["capture"])
                    authority_cells = cursor_capture.get("authorities_by_cell")
                    source_digests = cursor_capture.get(
                        "source_projection_sha256s"
                    )
                    if not isinstance(authority_cells, Mapping) or not isinstance(
                        source_digests, Mapping
                    ):
                        raise CampaignProviderError(
                            "detached cursor capture returned incomplete authorities"
                        )
                    detached_authorities = {
                        "source_projection_sha256s": {
                            cell_id: str(authority_cells[cell_id]["source_projection_sha256"])
                            for cell_id in request.cell_ids
                        },
                        "architecture_b_identities": {
                            cell_id: dict(authority_cells[cell_id]["architecture_b_identity"])
                            for cell_id in request.cell_ids
                        },
                        "observer_continuations": {
                            cell_id: dict(authority_cells[cell_id]["observer_continuation"])
                            for cell_id in request.cell_ids
                        },
                        "mechanics_checkpoint_digests": {
                            cell_id: str(authority_cells[cell_id]["mechanics_checkpoint_digest"])
                            for cell_id in request.cell_ids
                        },
                        "canonical_checkpoint_storage_digests": {
                            cell_id: str(
                                authority_cells[cell_id][
                                    "canonical_checkpoint_storage_digest"
                                ]
                            )
                            for cell_id in request.cell_ids
                        },
                        "cuda_source_field_counts": {
                            cell_id: int(authority_cells[cell_id]["cuda_source_field_count"])
                            for cell_id in request.cell_ids
                        },
                    }
                    immutable_authority_digests = {
                        cell_id: str(source_digests[cell_id])
                        for cell_id in request.cell_ids
                    }
                    immutable_snapshot_digest_sha256s = immutable_authority_digests
                    captured_bytes = int(detached_capture["logical_bytes"])
                    capture_measurement = (
                        self._acs_detached_capture_measurement_v1(
                            native_receipt=native_receipt,
                            direct_result=direct_result,
                            checkpoint_projection_time_ns=(
                                checkpoint_projection_time_ns
                            ),
                            capture_started_unix_ns=acs_capture_started_unix_ns,
                            capture_started_perf_ns=capture_started_perf_ns,
                            capture_completed_perf_ns=capture_completed_perf_ns,
                            capture_cpu_started_ns=acs_capture_cpu_started_ns,
                            coordinate=(
                                segment_end_next_tick if tick_native else end
                            ),
                            captured_bytes=captured_bytes,
                            source_projection_sha256s=immutable_authority_digests,
                            host_resources_before=acs_host_resources_before,
                            gpu_resources_before=acs_gpu_resources_before,
                        )
                    )
                    capture_measurement["segment_ordinal"] = segment_ordinal
                    capture_measurement["owner_transition"] = owner_transition
                    capture_measurement["owner_span_start_tick"] = (
                        owner_span_start_tick if tick_native else None
                    )
                    capture_measurement["owner_span_end_tick"] = (
                        owner_span_end_tick if tick_native else None
                    )
                    capture_measurement["cursor_receipt"] = dict(
                        cursor_capture.get("cursor_receipt") or {}
                    )
                    acs_capture_measurements.append(capture_measurement)
                    continuation_requires_source = bool(
                        tick_native
                        and segment_end_next_tick != request.total_ticks
                        and segment_end_next_tick
                        != deliberate_interruption_tick
                    )
                    if continuation_requires_source:
                        rebind_source = getattr(
                            source_projection_cursor_factory,
                            "rebind_detached_spool",
                            None,
                        )
                        if not callable(rebind_source) and bounded_streamed_preparation:
                            raise CampaignProviderError(
                                "same-owner detached checkpoint cannot roll its canonical source cursor"
                            )
                        if callable(rebind_source):
                            rebind_source(detached_capture)
                    if source_projection_acs_token is not None:
                        if acs_pipeline is None:
                            raise CampaignProviderError(
                                "detached source release lacks its ACS pipeline"
                            )
                        source_token_sequence = getattr(
                            source_projection_acs_token, "sequence", None
                        )
                        source_measurement = source_lease_measurements.get(
                            source_token_sequence
                        )
                        if not isinstance(source_measurement, dict):
                            raise CampaignProviderError(
                                "detached source release lost its timing authority"
                            )
                        source_released_perf_ns = acs_pipeline.release_source(
                            source_projection_acs_token
                        )
                        source_measurement[
                            "detached_source_released_perf_ns"
                        ] = source_released_perf_ns
                        source_measurement[
                            "successor_owner_source_consumed_perf_ns"
                        ] = source_released_perf_ns
                        source_lease_measurements.pop(
                            source_token_sequence, None
                        )
                        source_projection_acs_token = None
                capture = {
                    "coordinate": segment_end_next_tick if tick_native else end,
                    "parent_coordinate": (
                        segment_start_next_tick if tick_native else current
                    ),
                    "generation": end,
                    "tick": terminal_tick,
                    "tick_native": tick_native,
                    "release_source_import_after_publication": bool(
                        tick_native
                        and (
                            owner_transition
                            or segment_end_next_tick == request.total_ticks
                            or segment_end_next_tick == deliberate_interruption_tick
                        )
                    ),
                    "source_import_projection_digest_sha256s": dict(
                        source_import_projection_digest_sha256s
                    ),
                    "metrics_rows": [
                        dict(row)
                        for row in (
                            cursor_capture.get("compact_result_rows", ())
                            if tick_native_detached_acs
                            and projection_cursor is not None
                            else metrics_rows
                        )
                    ],
                    "production_control_readback": (
                        dict(segment_control_readback)
                        if segment_control_readback is not None
                        else None
                    ),
                    "authoritative_analysis_sources": [
                        dict(row) for row in authoritative_analysis_sources
                    ],
                    "native_receipt": native_receipt,
                    "capture_barrier_measurement": capture_measurement,
                    "immutable_snapshot_digest_sha256s": dict(
                        immutable_snapshot_digest_sha256s
                    ),
                }

                if tick_native_detached_acs:
                    try:
                        if detached_capture is None:
                            if projection_cursor is not None:
                                raise CampaignProviderError(
                                    "detached ACS cursor did not produce a capture"
                                )
                            if (
                                immutable_source_projections is None
                                or immutable_snapshot_digest_sha256s is None
                            ):
                                raise CampaignProviderError(
                                    "detached ACS sequence capture lacks its snapshot"
                                )
                            detached_authorities = (
                                self._aggregate_checkpoint_authorities_v1(
                                    request=request,
                                    source_projections=immutable_source_projections,
                                    generation=end,
                                    tick=terminal_tick,
                                    absolute_next_tick=(
                                        segment_end_next_tick if tick_native else None
                                    ),
                                    segment_ordinal=segment_ordinal,
                                    expected_source_projection_digest_sha256s=(
                                        immutable_snapshot_digest_sha256s
                                    ),
                                )
                            )
                            checkpoint_name = (
                                f"checkpoint_tick{segment_end_next_tick:012d}.acsaggregate"
                                if tick_native
                                else f"checkpoint_gen{end:06d}.acsaggregate"
                            )
                            destination = (
                                context.batch_root
                                / "checkpoint_native_aggregate_v1"
                                / checkpoint_name
                            )
                            from engine.campaigns.acs_native_detached_slots_v1 import (
                                capture_acs_native_aggregate_payload_v1,
                            )

                            detached_capture = (
                                capture_acs_native_aggregate_payload_v1(
                                    immutable_source_projections,
                                    destination,
                                    cell_ids=request.cell_ids,
                                    generation=end,
                                    tick=terminal_tick,
                                    absolute_next_tick=(
                                        segment_end_next_tick if tick_native else None
                                    ),
                                    batch_id=request.batch_id,
                                    segment_ordinal=segment_ordinal,
                                    source_projection_sha256s=detached_authorities[
                                        "source_projection_sha256s"
                                    ],
                                    architecture_b_identities=detached_authorities[
                                        "architecture_b_identities"
                                    ],
                                    observer_continuations=detached_authorities[
                                        "observer_continuations"
                                    ],
                                    require_native_extension=(
                                        request.profile_id == PRODUCTION_PROFILE_ID
                                    ),
                                )
                            )
                        if (
                            detached_authorities is None
                            and not physical_batch_detached_capture_route
                        ):
                            raise CampaignProviderError(
                                "detached ACS authorities are missing"
                            )
                        source_consumer_required = bool(
                            tick_native
                            and segment_end_next_tick != request.total_ticks
                            and segment_end_next_tick
                            != deliberate_interruption_tick
                            and detached_capture.get("schema_version")
                            == "eiel_acs_native_detached_spool_capture_v2"
                        )
                        if source_consumer_required and detached_capture.get(
                            "schema_version"
                        ) != "eiel_acs_native_detached_spool_capture_v2":
                            raise CampaignProviderError(
                                "owner-transition ACS continuation requires a sealed cursor spool"
                            )
                        detached_capture["source_consumer_required"] = (
                            source_consumer_required
                        )
                        defer_terminal_submission = bool(
                            physical_batch_detached_capture_route
                            and (
                                segment_end_next_tick == request.total_ticks
                                or segment_end_next_tick
                                == deliberate_interruption_tick
                            )
                        )
                        if defer_terminal_submission:
                            deferred_terminal_capture = dict(detached_capture)
                        else:
                            acs_pipeline.submit_capture(
                                acs_capture_token,
                                detached_capture,
                            )
                    except BaseException as exc:
                        acs_pipeline.fail_capture(acs_capture_token, exc)
                        raise
                    capture["detached_authorities"] = detached_authorities
                    capture["detached_capture"] = {
                        key: value
                        for key, value in detached_capture.items()
                        if key
                        not in {
                            "header",
                            "spool_index",
                            "source_projection_sha256s",
                        }
                    }
                    capture["native_receipt"][
                        "checkpoint_capture_mode"
                    ] = (
                        "direct_physical_to_world_sharded_persistence_v1"
                        if request.world_sharded_checkpoint
                        else "direct_physical_capture_release_then_host_settlement_v1"
                        if terminal_physical_capture_route
                        else (
                            "detached_physical_batch_host_settlement_v1"
                            if physical_batch_detached_capture_route
                            else "detached_file_backed_native_v1"
                        )
                    )
                elif acs_pipeline is not None:
                    def persist_immutable_snapshot(
                        projections=immutable_source_projections,
                        generation=end,
                        tick=terminal_tick,
                        next_tick=(segment_end_next_tick if tick_native else None),
                        ordinal=segment_ordinal,
                        expected_digests=immutable_snapshot_digest_sha256s,
                    ):
                        return self._persist_projection_checkpoint_set_v1(
                            batch_root=context.batch_root,
                            request=request,
                            source_projections=projections,
                            generation=generation,
                            tick=tick,
                            absolute_next_tick=next_tick,
                            segment_ordinal=ordinal,
                            expected_source_projection_digest_sha256s=(
                                expected_digests
                            ),
                        )

                    acs_pipeline.seal_capture(
                        acs_capture_token,
                        captured_bytes=captured_bytes,
                        persist=persist_immutable_snapshot,
                    )
                if acs_pipeline is not None:
                    pending_acs_checkpoints.append((acs_capture_token, capture))
                    current_next_tick = segment_end_next_tick
                    current = end
                    source_projections = (
                        None
                        if tick_native_detached_acs
                        else immutable_source_projections
                    )
                    if tick_native_detached_acs:
                        if bool(detached_capture.get("source_consumer_required")):
                            if source_projection_acs_token is not None:
                                raise CampaignProviderError(
                                    "detached spool successor-source lease overlapped another owner"
                                )
                            source_projections = _LazyDetachedSpoolProjectionSequence(
                                detached_capture
                            )
                            source_projection_acs_token = acs_capture_token
                            source_lease_measurements[
                                acs_capture_token.sequence
                            ] = capture_measurement
                        # The worker owns only the file path and scalar header;
                        # release the producer's tensor-bearing tuple before the
                        # successor subspan can be prepared.
                        del immutable_source_projections
                        del detached_capture
                    executor_arguments.clear()
                    abi_owner = None
                    post_dispatch_owner = None
                    source_projection_cursor_factory = None
                    projection_cursor = None
                    compact_source_projections = ()
                    prepared_arguments = None
                    owner_epoch_transferred = transfer_owner_epoch_at_boundary(
                        resident_owner,
                        boundary_next_tick=segment_end_next_tick,
                        owner_transition=(
                            owner_transition
                            and segment_end_next_tick != deliberate_interruption_tick
                        ),
                    )
                    keep_owner = bool(
                        tick_native
                        and (not owner_transition or owner_epoch_transferred)
                        and segment_end_next_tick != deliberate_interruption_tick
                        and segment_end_next_tick != request.total_ticks
                    )
                    owner_to_close = resident_owner if not keep_owner else None
                    if not keep_owner:
                        resident_owner = None
                    del builds, requests, prepared, direct_result, settlement
                    if owner_to_close is not None:
                        owner_release_receipt = release_resident_owner(owner_to_close)
                        owner_release_receipts.append(owner_release_receipt)
                        terminal_release_boundary = bool(
                            segment_end_next_tick == request.total_ticks
                            or segment_end_next_tick == deliberate_interruption_tick
                        )
                        gc.collect()
                        if terminal_release_boundary:
                            if not isinstance(capture_measurement, dict):
                                raise CampaignProviderError(
                                    "terminal detached ACS release lacks capture timing"
                                )
                            if physical_batch_detached_capture_route:
                                if deferred_terminal_capture is None:
                                    raise CampaignProviderError(
                                        "terminal physical capture was submitted before CUDA release"
                                    )
                                context_release_receipt = (
                                    _reset_terminal_cuda_primary_context_v1(
                                        execution_device=str(
                                            native_receipt["execution_device"]
                                        ),
                                        owner_release_receipt=owner_release_receipt,
                                        terminal_capture=deferred_terminal_capture,
                                        live_owner_count=len(
                                            resident_owner_live_started_perf_ns
                                        ),
                                    )
                                )
                                terminal_cuda_context_release_receipts.append(
                                    context_release_receipt
                                )
                                capture_measurement[
                                    "terminal_cuda_primary_context_release"
                                ] = dict(context_release_receipt)
                                try:
                                    acs_pipeline.submit_capture(
                                        acs_capture_token,
                                        deferred_terminal_capture,
                                    )
                                except BaseException as exc:
                                    acs_pipeline.fail_capture(acs_capture_token, exc)
                                    raise
                                deferred_terminal_capture = None
                                safe_release_perf_ns = context_release_receipt[
                                    "released_perf_ns"
                                ]
                                safe_release_unix_ns = context_release_receipt[
                                    "released_unix_ns"
                                ]
                            else:
                                safe_release_perf_ns = (
                                    owner_release_receipt.released_perf_ns
                                )
                                safe_release_unix_ns = (
                                    owner_release_receipt.released_unix_ns
                                )
                            capture_measurement[
                                "gpu_safely_releasable_perf_ns"
                            ] = safe_release_perf_ns
                            capture_measurement[
                                "gpu_safely_releasable_unix_ns"
                            ] = safe_release_unix_ns
                            if request.world_sharded_checkpoint:
                                pending_state = {
                                    "schema_version": (
                                        "eiel_world_sharded_checkpoint_control_plane_state_v1"
                                    ),
                                    "state": (
                                        "COMPUTE_COMPLETE_GPU_RELEASED_"
                                        "PERSISTENCE_PENDING"
                                    ),
                                    "absolute_next_tick": int(
                                        segment_end_next_tick
                                    ),
                                    "entered_perf_ns": int(safe_release_perf_ns),
                                    "entered_unix_ns": int(safe_release_unix_ns),
                                    "terminal_cuda_owner_released": True,
                                    "queued_cuda_references": 0,
                                    "queued_cuda_callbacks": 0,
                                    "restore_admitted": False,
                                    "lifecycle_release_admitted": False,
                                    "qualification_closeout_admitted": False,
                                }
                                capture_measurement[
                                    "checkpoint_control_plane_state"
                                ] = dict(pending_state)
                                checkpoint_control_plane_transitions.append(
                                    pending_state
                                )
                            capture_measurement[
                                "post_simulation_acs_tail_excluded_from_gpu_performance"
                            ] = True
                        else:
                            try:
                                import torch

                                torch.cuda.empty_cache()
                            except (ImportError, RuntimeError):
                                pass
                    if segment_end_next_tick == deliberate_interruption_tick:
                        deliberately_interrupted = True
                        break
                    continue
            bounded_capture_receipt: dict[str, Any] | None = None
            if terminal_physical_capture_route:
                if (
                    detached_capture is None
                    or not isinstance(capture_measurement, dict)
                    or captured_bytes is None
                    or resident_owner is None
                ):
                    raise CampaignProviderError(
                        "direct physical terminal capture lacks release authorities"
                    )
                executor_arguments.clear()
                abi_owner = None
                post_dispatch_owner = None
                source_projection_cursor_factory = None
                projection_cursor = None
                compact_source_projections = ()
                prepared_arguments = None
                source_projections = None
                prepared_source_projections = ()
                immutable_source_projections = None
                owner_to_close = resident_owner
                resident_owner = None
                builds = ()
                requests = ()
                prepared = None
                direct_result = {}
                settlement = None
                owner_release_receipt = release_resident_owner(owner_to_close)
                owner_release_receipts.append(owner_release_receipt)
                if (
                    owner_release_receipt.live_alias_count != 0
                    or resident_owner_live_started_perf_ns
                ):
                    raise CampaignProviderError(
                        "direct physical capture did not release its sole GPU owner"
                    )
                safe_release_perf_ns = owner_release_receipt.released_perf_ns
                safe_release_unix_ns = owner_release_receipt.released_unix_ns
                capture_measurement["gpu_safely_releasable_perf_ns"] = (
                    safe_release_perf_ns
                )
                capture_measurement["gpu_safely_releasable_unix_ns"] = (
                    safe_release_unix_ns
                )
                capture_measurement[
                    "post_simulation_acs_tail_excluded_from_gpu_performance"
                ] = True
                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
                from engine.campaigns.acs_native_detached_slots_v1 import (
                    finalize_direct_resident_physical_capture_v1,
                )

                direct_physical_receipt = (
                    finalize_direct_resident_physical_capture_v1(detached_capture)
                )
                writer_started_perf_ns = direct_physical_receipt.get(
                    "detached_writer_started_perf_ns"
                )
                writer_finished_perf_ns = direct_physical_receipt.get(
                    "detached_writer_finished_perf_ns"
                )
                if (
                    type(writer_started_perf_ns) is not int
                    or writer_started_perf_ns <= safe_release_perf_ns
                    or type(writer_finished_perf_ns) is not int
                    or writer_finished_perf_ns <= writer_started_perf_ns
                ):
                    raise CampaignProviderError(
                        "direct physical persistence began before GPU owner release"
                    )
                gpu_release_to_durable_set_seconds = (
                    writer_finished_perf_ns - safe_release_perf_ns
                ) / 1_000_000_000.0
                detached_authorities = direct_physical_receipt.get(
                    "detached_authorities"
                )
                if not isinstance(detached_authorities, Mapping):
                    raise CampaignProviderError(
                        "direct physical settlement lacks detached authorities"
                    )
                checkpoint_set = (
                    self._aggregate_checkpoint_set_from_detached_receipt_v1(
                        request=request,
                        receipt=direct_physical_receipt,
                        authorities=detached_authorities,
                        writer_telemetry={
                            "owner_snapshot_capture_mode": (
                                "direct_physical_capture_release_then_host_settlement_v1"
                            ),
                            "gpu_safely_releasable_perf_ns": safe_release_perf_ns,
                            "detached_writer_started_perf_ns": writer_started_perf_ns,
                            "detached_writer_finished_perf_ns": (
                                writer_finished_perf_ns
                            ),
                            "gpu_release_to_durable_set_seconds": (
                                gpu_release_to_durable_set_seconds
                            ),
                            "detached_writer_wall_time_ns": (
                                direct_physical_receipt.get(
                                    "detached_writer_wall_time_ns"
                                )
                            ),
                            "detached_writer_cpu_time_ns": (
                                direct_physical_receipt.get(
                                    "detached_writer_cpu_time_ns"
                                )
                            ),
                        },
                    )
                )
                checkpoint_set["checkpoint_set_commit"] = {
                    "schema_version": "eiel_checkpoint_set_commit_v1",
                    "state": "SEALED",
                    "ordered_cell_ids": list(request.cell_ids),
                    "writer_concurrency": 1,
                    "failure_state": None,
                }
                checkpoint_set["checkpoint_projection_cursor_receipt"] = None
                terminal_digests = checkpoint_set["terminal_cell_digests"]
                capture_measurement["immutable_snapshot_digest_sha256s"] = dict(
                    terminal_digests
                )
                capture_measurement[
                    "terminal_digest_authority_state"
                ] = "complete_on_post_release_host_settlement_v1"
                capture_measurement["detached_writer_started_perf_ns"] = (
                    writer_started_perf_ns
                )
                capture_measurement["detached_writer_finished_perf_ns"] = (
                    writer_finished_perf_ns
                )
                capture_measurement["gpu_release_to_durable_set_seconds"] = (
                    gpu_release_to_durable_set_seconds
                )
                capture_measurement["detached_writer_wall_time_ns"] = (
                    direct_physical_receipt.get("detached_writer_wall_time_ns")
                )
                capture_measurement[
                    "persistence_started_after_gpu_release"
                ] = True
                capture_measurement["producer_capture_tensor_free"] = True
                capture_measurement["owner_release_receipt"] = (
                    owner_release_receipt.payload()
                )
                capture_measurement["host_resources_after"] = (
                    self._acs_process_resource_sample_v1()
                )
                capture_measurement["gpu_resources_after"] = (
                    self._acs_gpu_resource_sample_v1(
                        native_receipt["execution_device"]
                    )
                )
                projected_rows = _validated_physical_compact_result_rows_v1(
                    direct_physical_receipt,
                    cell_ids=request.cell_ids,
                )
                if getattr(self, "terminal_snapshot_metrics", False) or getattr(
                    mechanics_binding,
                    "terminal_snapshot_metrics",
                    False,
                ):
                    if metrics_rows and callable(terminal_snapshot_metrics_merger):
                        metrics_rows = [
                            dict(row)
                            for row in terminal_snapshot_metrics_merger(
                                metrics_rows,
                                projected_rows,
                            )
                        ]
                    else:
                        metrics_rows = projected_rows
                else:
                    metrics_rows.extend(projected_rows)
                bounded_capture_receipt = {
                    "schema_version": "eiel_bounded_terminal_capture_receipt_v1",
                    "capture_mode": (
                        "direct_physical_capture_release_then_host_settlement_v1"
                    ),
                    "coordinate": int(segment_end_next_tick),
                    "next_absolute_tick": int(segment_end_next_tick),
                    "ordered_cell_count": len(request.cell_ids),
                    "ordered_cell_ids": list(request.cell_ids),
                    "digest_map": dict(terminal_digests),
                    "writer_concurrency": 1,
                    "max_live_tensor_bearing_projections": 0,
                    "max_live_projection_bytes": 0,
                    "capture_started_unix_ns": acs_capture_started_unix_ns,
                    "capture_completed_unix_ns": capture_measurement.get(
                        "capture_completed_unix_ns"
                    ),
                    "gpu_safely_releasable_unix_ns": safe_release_unix_ns,
                    "gpu_safely_releasable_perf_ns": safe_release_perf_ns,
                    "durable_completed_unix_ns": time.time_ns(),
                    "persistence_started_after_gpu_release": True,
                    "producer_capture_tensor_free": True,
                    "owner_release_receipt": owner_release_receipt.payload(),
                    "detached_writer_started_perf_ns": writer_started_perf_ns,
                    "detached_writer_finished_perf_ns": (
                        writer_finished_perf_ns
                    ),
                    "gpu_release_to_durable_set_seconds": (
                        gpu_release_to_durable_set_seconds
                    ),
                    "catalog_publication_proof": {
                        "state": "SEALED",
                        "context_publication_completed_unix_ns": None,
                        "catalog_bound_unix_ns": None,
                    },
                    "host_resources_before": acs_host_resources_before,
                    "host_resources_after": capture_measurement[
                        "host_resources_after"
                    ],
                    "gpu_resources_before": acs_gpu_resources_before,
                    "gpu_resources_after": capture_measurement[
                        "gpu_resources_after"
                    ],
                    "checkpoint_bytes": checkpoint_set["checkpoint_bytes"],
                    "writer_service_time_ns": checkpoint_set[
                        "writer_service_time_ns"
                    ],
                    "writer_cpu_time_ns": checkpoint_set["writer_cpu_time_ns"],
                    "cell_checkpoint_bytes": dict(
                        checkpoint_set["cell_checkpoint_bytes"]
                    ),
                    "per_stage_times": dict(
                        checkpoint_set["cell_stage_telemetry_seconds"]
                    ),
                    "failure_state": None,
                    "publication_state": "SEALED",
                }
                bounded_terminal_capture_receipts.append(
                    bounded_capture_receipt
                )
            else:
                checkpoint_set = self._persist_projection_checkpoint_set_v1(
                    batch_root=context.batch_root,
                    request=request,
                    source_projections=source_projections,
                    projection_cursor=projection_cursor,
                    generation=end,
                    tick=terminal_tick,
                    absolute_next_tick=(
                        segment_end_next_tick if tick_native else None
                    ),
                    segment_ordinal=segment_ordinal,
                    expected_source_projection_digest_sha256s=(
                        immutable_snapshot_digest_sha256s
                        if boundary_acs_enabled
                        else None
                    ),
                )
            bounded_projection_cursor_present = (
                projection_cursor is not None
                and not terminal_physical_capture_route
            )
            if (
                projection_cursor is not None
                and not terminal_physical_capture_route
            ):
                aggregate = checkpoint_set.get("bounded_projection_aggregate")
                cursor_receipt = checkpoint_set.get(
                    "checkpoint_projection_cursor_receipt"
                )
                if not isinstance(aggregate, Mapping) or not isinstance(
                    cursor_receipt, Mapping
                ):
                    raise CampaignProviderError(
                        "bounded checkpoint persistence omitted cursor authorities"
                    )
                if (
                    cursor_receipt.get("state") != "SEALED"
                    or cursor_receipt.get("ordered_cell_ids")
                    != list(request.cell_ids)
                    or cursor_receipt.get("covered_cell_ids")
                    != list(request.cell_ids)
                    or cursor_receipt.get("max_live_tensor_bearing_projections") != 1
                    or cursor_receipt.get("writer_concurrency") != 1
                ):
                    raise CampaignProviderError(
                        "bounded checkpoint cursor receipt does not prove one-cell sealing"
                    )
                projected_rows = [
                    dict(row)
                    for row in aggregate.get("compact_result_rows", ())
                    if isinstance(row, Mapping)
                ]
                segment_artifacts = tuple(
                    row
                    for row in aggregate.get("segment_artifacts", ())
                    if isinstance(row, Mapping)
                )
                current_artifacts = [dict(row) for row in segment_artifacts]
                current_segment_artifacts.extend(current_artifacts)
                lifecycle_qacct_receipts.extend(
                    dict(row)
                    for row in aggregate.get("lifecycle_qacct_receipts", ())
                    if isinstance(row, Mapping)
                )
                if getattr(self, "terminal_snapshot_metrics", False) or getattr(
                    mechanics_binding,
                    "terminal_snapshot_metrics",
                    False,
                ):
                    if metrics_rows and callable(terminal_snapshot_metrics_merger):
                        metrics_rows = [
                            dict(row)
                            for row in terminal_snapshot_metrics_merger(
                                metrics_rows,
                                projected_rows,
                            )
                        ]
                    else:
                        metrics_rows = projected_rows
                else:
                    metrics_rows.extend(projected_rows)

                if native_chronicle is not None and projection_cursor is not None:
                    chronicle_cursor = self._checkpoint_bundle_projection_cursor_v1(
                        batch_root=context.batch_root,
                        request=request,
                        bundle_rows=checkpoint_set.get("bundle_rows", ()),
                    )
                    native_chronicle.observe_segment(
                        segment_ordinal=segment_ordinal,
                        start_generation=current,
                        end_generation=end,
                        cell_ids=tuple(request.cell_ids),
                        segment_artifacts=segment_artifacts,
                        checkpoint_projections=None,
                        checkpoint_projection_cursor=chronicle_cursor,
                    )
                    native_chronicle_observed = True

                capture_started_unix_ns = checkpoint_projection_started_unix_ns
                capture_completed_unix_ns = time.time_ns()
                host_resources_after = self._acs_process_resource_sample_v1()
                gpu_resources_after = self._acs_gpu_resource_sample_v1(
                    native_receipt["execution_device"]
                )
                native_final_sync_time_ns = _require_acs_native_nonnegative_int_v1(
                    direct_result, "final_stream_synchronization_time_ns"
                )
                last_useful_kernel_completion_observed_host_time_ns = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result,
                        "last_useful_kernel_completion_observed_host_time_ns",
                    )
                )
                if last_useful_kernel_completion_observed_host_time_ns == 0:
                    raise CampaignProviderError(
                        "bounded terminal capture lacks the last-useful-kernel boundary"
                    )
                final_host_synchronizations = _require_acs_native_nonnegative_int_v1(
                    direct_result, "final_host_synchronizations"
                )
                final_device_to_host_operations = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result, "final_device_to_host_operations"
                    )
                )
                compact_settlement_d2h_operations = (
                    _require_acs_native_nonnegative_int_v1(
                        direct_result, "compact_settlement_d2h_operations"
                    )
                )
                reporting_storage = direct_result.get(
                    "compact_summary_reporting_storage"
                )
                if not isinstance(reporting_storage, Mapping):
                    raise CampaignProviderError(
                        "bounded terminal capture lacks compact reporting storage"
                    )
                final_transfer = reporting_storage.get("final_transfer")
                if not isinstance(final_transfer, Mapping):
                    raise CampaignProviderError(
                        "bounded terminal capture lacks final transfer contract"
                    )
                final_transfer_bytes = final_transfer.get("bytes")
                observed_host_transfer_count = final_transfer.get(
                    "observed_host_transfer_count"
                )
                if type(final_transfer_bytes) is not int or final_transfer_bytes < 0:
                    raise CampaignProviderError(
                        "bounded terminal capture transfer bytes are invalid"
                    )
                if (
                    type(observed_host_transfer_count) is not int
                    or observed_host_transfer_count < 0
                    or observed_host_transfer_count != compact_settlement_d2h_operations
                ):
                    raise CampaignProviderError(
                        "bounded terminal capture transfer count differs from compact settlement"
                    )
                if final_transfer.get("status") not in {
                    "native_stream_ordered_callback_wired",
                    "disabled",
                }:
                    raise CampaignProviderError(
                        "bounded terminal capture transfer status is invalid"
                    )
                cursor_max_live_bytes = cursor_receipt.get(
                    "max_live_projection_bytes"
                )
                checkpoint_bytes = checkpoint_set.get("checkpoint_bytes")
                if type(cursor_max_live_bytes) is not int or cursor_max_live_bytes < 0:
                    raise CampaignProviderError(
                        "bounded terminal capture lacks max live projection bytes"
                    )
                if type(checkpoint_bytes) is not int or checkpoint_bytes < 0:
                    raise CampaignProviderError(
                        "bounded terminal capture lacks checkpoint bytes"
                    )
                bounded_capture_receipt = {
                    "schema_version": "eiel_bounded_terminal_capture_receipt_v1",
                    "capture_mode": "bounded_serial_cell_v1",
                    "coordinate": int(segment_end_next_tick),
                    "next_absolute_tick": int(segment_end_next_tick),
                    "ordered_cell_count": len(request.cell_ids),
                    "ordered_cell_ids": list(request.cell_ids),
                    "digest_map": dict(checkpoint_set["terminal_cell_digests"]),
                    "writer_concurrency": 1,
                    "max_live_tensor_bearing_projections": 1,
                    "max_live_projection_bytes": cursor_max_live_bytes,
                    "final_device_to_host_operations": final_device_to_host_operations,
                    "final_device_to_host_bytes": final_transfer_bytes,
                    "capture_started_unix_ns": capture_started_unix_ns,
                    "capture_completed_unix_ns": capture_completed_unix_ns,
                    "gpu_safely_releasable_unix_ns": None,
                    "native_final_sync_proof": {
                        "final_stream_synchronization_time_ns": native_final_sync_time_ns,
                        "last_useful_kernel_completion_observed_host_time_ns": (
                            last_useful_kernel_completion_observed_host_time_ns
                        ),
                        "final_host_synchronizations": final_host_synchronizations,
                    },
                    "durable_completed_unix_ns": time.time_ns(),
                    "catalog_publication_proof": {
                        "state": "SEALED",
                        "context_publication_completed_unix_ns": None,
                        "catalog_bound_unix_ns": None,
                    },
                    "host_resources_before": acs_host_resources_before,
                    "host_resources_peak": dict(host_resources_after),
                    "host_resources_after": host_resources_after,
                    "gpu_resources_before": acs_gpu_resources_before,
                    "gpu_resources_peak": dict(gpu_resources_after),
                    "gpu_resources_after": gpu_resources_after,
                    "checkpoint_bytes": checkpoint_bytes,
                    "writer_service_time_ns": checkpoint_set[
                        "writer_service_time_ns"
                    ],
                    "writer_cpu_time_ns": checkpoint_set["writer_cpu_time_ns"],
                    "cell_checkpoint_bytes": dict(
                        checkpoint_set.get("cell_checkpoint_bytes", {})
                    ),
                    "per_stage_times": dict(
                        checkpoint_set.get("cell_stage_telemetry_seconds", {})
                    ),
                    "failure_state": None,
                    "publication_state": "SEALED",
                }
                # Preserve the existing clock input shape while carrying the
                # stronger bounded receipt additively.
                capture_measurement = {
                    "schema_version": "eiel_acs_capture_barrier_measurement_v1",
                    "capture_completed_unix_ns": capture_completed_unix_ns,
                    "capture_completed_perf_ns": time.perf_counter_ns(),
                    "segment_ordinal": segment_ordinal,
                    "coordinate": int(segment_end_next_tick),
                    "execution_device": native_receipt["execution_device"],
                    "checkpoint_projection_time_ns": checkpoint_projection_time_ns,
                    "snapshot_digest_time_ns": 0,
                    "host_capture_freeze_time_ns": (
                        time.perf_counter_ns() - checkpoint_projection_started
                    ),
                    "capture_cpu_time_ns": (
                        time.process_time_ns() - acs_capture_cpu_started_ns
                        if acs_capture_cpu_started_ns is not None
                        else None
                    ),
                    "native_final_stream_synchronization_time_ns": (
                        native_final_sync_time_ns
                    ),
                    "last_useful_kernel_completion_observed_host_time_ns": (
                        last_useful_kernel_completion_observed_host_time_ns
                    ),
                    "capture_barrier_time_ns": (
                        time.perf_counter_ns() - checkpoint_projection_started
                    ),
                    "post_sync_gpu_idle_proxy_time_ns": 0,
                    "captured_bytes": checkpoint_bytes,
                    "captured_cell_count": len(request.cell_ids),
                    "immutable_snapshot_digest_sha256s": dict(
                        checkpoint_set["terminal_cell_digests"]
                    ),
                    "host_resources_before": acs_host_resources_before,
                    "host_resources_peak": dict(host_resources_after),
                    "host_resources_after": host_resources_after,
                    "gpu_resources_before": acs_gpu_resources_before,
                    "gpu_resources_peak": dict(gpu_resources_after),
                    "gpu_resources_after": gpu_resources_after,
                    "native_transfer_receipt": {
                        "final_host_synchronizations": final_host_synchronizations,
                        "final_device_to_host_operations": (
                            final_device_to_host_operations
                        ),
                        "compact_settlement_d2h_operations": (
                            compact_settlement_d2h_operations
                        ),
                        "final_transfer_bytes": final_transfer_bytes,
                        "final_transfer_contract": dict(final_transfer),
                    },
                    "measurement_scope": "bounded_serial_cell_gpu_capture_then_release",
                    "limitations": [],
                    "bounded_terminal_capture_receipt": bounded_capture_receipt,
                }
                # The terminal capture is synchronous and must release the
                # owner before the checkpoint is published as complete.
                owner_to_close = resident_owner
                resident_owner = None
                if owner_to_close is None:
                    raise CampaignProviderError(
                        "bounded terminal capture has no resident owner to release"
                    )
                executor_arguments.clear()
                abi_owner = None
                post_dispatch_owner = None
                source_projection_cursor_factory = None
                projection_cursor = None
                compact_source_projections = ()
                prepared_arguments = None
                builds = ()
                requests = ()
                prepared = None
                direct_result = {}
                settlement = None
                owner_release_receipt = release_resident_owner(owner_to_close)
                owner_release_receipts.append(owner_release_receipt)
                safe_release_perf_ns = owner_release_receipt.released_perf_ns
                safe_release_unix_ns = owner_release_receipt.released_unix_ns
                # The measurement is assembled before release so it can
                # describe the final native-sync boundary. Bind the actual
                # post-release authority after the owner is gone; otherwise
                # the clock adapter would have to infer release from durable
                # publication.
                capture_measurement["gpu_safely_releasable_unix_ns"] = (
                    safe_release_unix_ns
                )
                capture_measurement["gpu_safely_releasable_perf_ns"] = (
                    safe_release_perf_ns
                )
                capture_measurement[
                    "post_simulation_acs_tail_excluded_from_gpu_performance"
                ] = True
                capture_measurement["host_resources_after"] = (
                    self._acs_process_resource_sample_v1()
                )
                capture_measurement["host_resources_peak"] = dict(
                    capture_measurement["host_resources_after"]
                )
                capture_measurement["gpu_resources_after"] = (
                    self._acs_gpu_resource_sample_v1(
                        native_receipt["execution_device"]
                    )
                )
                capture_measurement["gpu_resources_peak"] = dict(
                    capture_measurement["gpu_resources_after"]
                )
                bounded_capture_receipt["gpu_safely_releasable_unix_ns"] = (
                    safe_release_unix_ns
                )
                bounded_capture_receipt["gpu_safely_releasable_perf_ns"] = (
                    safe_release_perf_ns
                )
                bounded_capture_receipt["host_resources_after"] = (
                    capture_measurement["host_resources_after"]
                )
                bounded_capture_receipt["host_resources_peak"] = dict(
                    capture_measurement["host_resources_peak"]
                )
                bounded_capture_receipt["gpu_resources_after"] = (
                    capture_measurement["gpu_resources_after"]
                )
                bounded_capture_receipt["gpu_resources_peak"] = dict(
                    capture_measurement["gpu_resources_peak"]
                )
                bounded_capture_receipt["publication_state"] = "SEALED"
                if acs_measurement_enabled:
                    acs_capture_measurements.append(capture_measurement)
                bounded_terminal_capture_receipts.append(bounded_capture_receipt)
            bundle_rows = checkpoint_set["bundle_rows"]
            terminal_cell_digests = checkpoint_set["terminal_cell_digests"]
            mechanics_checkpoint_digests = checkpoint_set[
                "mechanics_checkpoint_digests"
            ]
            canonical_checkpoint_storage_digests = checkpoint_set[
                "canonical_checkpoint_storage_digests"
            ]
            if native_receipt.get("option_d_m0_telemetry_enabled") is True:
                native_receipt["checkpoint_compression_time_ns"] = checkpoint_set[
                    "writer_service_time_ns"
                ]
                native_receipt["checkpoint_compression_time_scope"] = (
                    "provider_checkpoint_bundle_write_wall_time_including_"
                    "serialization_digest_compression_and_io"
                )
            checkpoint_identity_contract = _checkpoint_identity_contract_v1(
                execution_route_sha256=request.execution_route_sha256,
                generation=None if tick_native else end,
                tick=None,
                next_absolute_tick=(segment_end_next_tick if tick_native else None),
                source_import_projection_digest_sha256s=(
                    source_import_projection_digest_sha256s
                ),
                parent_checkpoint_projection_digest_sha256s=(
                    parent_checkpoint_projection_digest_sha256s
                ),
                terminal_projection_digest_sha256s=terminal_cell_digests,
                canonical_terminal_state_digest_sha256s=mechanics_checkpoint_digests,
                canonical_checkpoint_storage_digest_sha256s=(
                    canonical_checkpoint_storage_digests
                ),
            )
            if native_receipt.get("option_d_m0_telemetry_enabled") is True:
                if (
                    option_d_m0_source_environment is None
                    or option_d_m0_loaded_artifact_contract is None
                    or option_d_m0_memory_device is None
                ):
                    raise CampaignProviderError(
                        "Option D M0 provider finalization lacks qualification context"
                    )
                option_d_m0_memory_measurement = (
                    _option_d_m0_finish_memory_measurement_v1(
                        execution_device=option_d_m0_memory_device
                    )
                )
                _finalize_option_d_m0_provider_receipts_v1(
                    plan=plan,
                    request=request,
                    native_receipt=native_receipt,
                    source_environment=option_d_m0_source_environment,
                    loaded_artifact_contract=option_d_m0_loaded_artifact_contract,
                    memory_measurement=option_d_m0_memory_measurement,
                    receipt_output_root=Path(context.batch_root),
                    segment_ordinal=segment_ordinal,
                    start_generation=current,
                    end_generation=end,
                    source_projections=prepared_source_projections,
                    terminal_projections=source_projections,
                    settlement_event_rows=tuple(
                        getattr(settlement, "settlement_event_rows", ())
                    ),
                    terminal_projection_digest_sha256s=terminal_cell_digests,
                    canonical_terminal_state_digest_sha256s=(
                        mechanics_checkpoint_digests
                    ),
                    checkpoint_identity_contract=checkpoint_identity_contract,
                )
            retained_prior_catalog = (
                _tick_native_retained_prior_checkpoint_catalog_v1(
                    checkpoint_catalog,
                    requested_checkpoint_ticks=request.checkpoint_ticks,
                )
                if tick_native
                else {
                    coordinate: dict(value)
                    for coordinate, value in checkpoint_catalog.items()
                }
            )
            checkpoint_payload = {
                "schema_version": self.checkpoint_schema_version,
                "execution_route_sha256": request.execution_route_sha256,
                "checkpoint_binding": self._checkpoint_binding(plan, request),
                **(
                    {
                        "time_authority": "absolute_tick_v1",
                        "absolute_tick": segment_end_next_tick,
                        "next_absolute_tick": segment_end_next_tick,
                    }
                    if tick_native
                    else {"generation": end}
                ),
                "continuation_storage_mode": request.checkpoint_storage_mode,
                "cell_bundles": bundle_rows,
                "terminal_cell_digests": terminal_cell_digests,
                "checkpoint_identity_contract": checkpoint_identity_contract,
                "checkpoint_identity_sha256": _checkpoint_identity_sha256_v1(
                    checkpoint_identity_contract
                ),
                "source_import_projection_digest_sha256s": (
                    checkpoint_identity_contract[
                        "source_import_projection_digest_sha256s"
                    ]
                ),
                "parent_checkpoint_projection_digest_sha256s": (
                    checkpoint_identity_contract[
                        "parent_checkpoint_projection_digest_sha256s"
                    ]
                ),
                "terminal_projection_digest_sha256s": (
                    checkpoint_identity_contract["terminal_projection_digest_sha256s"]
                ),
                "canonical_terminal_state_digest_sha256s": (
                    checkpoint_identity_contract[
                        "canonical_terminal_state_digest_sha256s"
                    ]
                ),
                "canonical_checkpoint_storage_digest_sha256s": (
                    checkpoint_identity_contract[
                        "canonical_checkpoint_storage_digest_sha256s"
                    ]
                ),
                "route_envelope_or_execution_provenance_digest_sha256": (
                    checkpoint_identity_contract[
                        "route_envelope_or_execution_provenance_digest_sha256"
                    ]
                ),
                "mechanics_checkpoint_digests": mechanics_checkpoint_digests,
                "canonical_checkpoint_storage_digests": (
                    canonical_checkpoint_storage_digests
                ),
                "component_field_trace_status": (
                    "exact_component_field_unavailable_only_terminal_projection_digest"
                ),
                "metrics_rows": [dict(row) for row in metrics_rows],
                **(
                    {"production_control_readback": dict(segment_control_readback)}
                    if segment_control_readback is not None
                    else {}
                ),
                "authoritative_analysis_sources": [
                    dict(row) for row in authoritative_analysis_sources
                ],
                "prior_checkpoint_catalog": retained_prior_catalog,
                "checkpoint_set_commit": checkpoint_set["checkpoint_set_commit"],
                "checkpoint_projection_cursor_receipt": checkpoint_set.get(
                    "checkpoint_projection_cursor_receipt"
                ),
                **(
                    {
                        "acs_checkpoint_instrumentation": {
                            "mode": "synchronous_instrumentation_only_v1",
                            "capture_barrier_measurement": capture_measurement,
                            "bounded_terminal_capture_receipt": bounded_capture_receipt,
                            "writer_service_time_ns": checkpoint_set[
                                "writer_service_time_ns"
                            ],
                            "writer_cpu_time_ns": checkpoint_set["writer_cpu_time_ns"],
                            "cell_service_times_ns": checkpoint_set[
                                "cell_service_times_ns"
                            ],
                            "cell_cpu_times_ns": checkpoint_set["cell_cpu_times_ns"],
                            "cell_stage_telemetry_seconds": checkpoint_set[
                                "cell_stage_telemetry_seconds"
                            ],
                            "cell_checkpoint_bytes": checkpoint_set[
                                "cell_checkpoint_bytes"
                            ],
                            "checkpoint_bytes": checkpoint_set["checkpoint_bytes"],
                            "checkpoint_bytes_measurement_status": checkpoint_set[
                                "checkpoint_bytes_measurement_status"
                            ],
                            "native_checkpoint_receipts": checkpoint_set[
                                "native_checkpoint_receipts"
                            ],
                            "native_capture_group_receipts": checkpoint_set[
                                "native_capture_group_receipts"
                            ],
                        }
                    }
                    if boundary_acs_enabled
                    else {}
                ),
            }
            if callable(component_checkpoint_exporter):
                component_checkpoint = component_checkpoint_exporter(
                    prepared.physical_window_arguments,
                    **(
                        {"absolute_tick": segment_end_next_tick}
                        if tick_native
                        else {"absolute_generation": end}
                    ),
                    cell_ids=request.cell_ids,
                )
                if not isinstance(component_checkpoint, Mapping):
                    raise CampaignProviderError(
                        "component checkpoint exporter returned no mapping"
                    )
                checkpoint_payload["component_checkpoint"] = (
                    self._write_component_checkpoint(
                        component_checkpoint,
                        context.batch_root
                        / "component_checkpoints"
                        / (
                            f"tick{segment_end_next_tick:012d}.pt"
                            if tick_native
                            else f"g{end:06d}.pt"
                        ),
                    )
                )
            if native_chronicle_observed and native_chronicle is not None:
                from engine.campaigns.canonical import canonical_sha256
                from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
                    CONTINUITY_SCHEMA_VERSION,
                )

                artifact_cells = [
                    artifact.get("cell")
                    for artifact in segment_artifacts
                    if isinstance(artifact, Mapping)
                ]
                if len(artifact_cells) != len(request.cell_ids) or any(
                    not isinstance(cell, Mapping) for cell in artifact_cells
                ):
                    raise CampaignProviderError(
                        "provider-native Chronicle checkpoint lacks ordered "
                        "native exposure cells"
                    )
                continuity = {
                    "schema_version": CONTINUITY_SCHEMA_VERSION,
                    "segment_count": int(segment_artifacts[0]["segment_index"]) + 1,
                    "cell_chain_sha256": [
                        cell["chain_sha256_after_segment"] for cell in artifact_cells
                    ],
                    "cumulative_attempt_counts": [
                        cell["cumulative_counts_after_segment"]["attempts"]
                        for cell in artifact_cells
                    ],
                    "cumulative_success_counts": [
                        cell["cumulative_counts_after_segment"]["successes"]
                        for cell in artifact_cells
                    ],
                    "cumulative_failure_counts": [
                        cell["cumulative_counts_after_segment"]["failures"]
                        for cell in artifact_cells
                    ],
                    "cumulative_snapshot_counts": [
                        cell["cumulative_counts_after_segment"]["eligible_snapshots"]
                        for cell in artifact_cells
                    ],
                }
                checkpoint_authority = canonical_sha256(checkpoint_payload)
                prefix_receipt = native_chronicle.checkpoint_receipt(
                    snapshot_batch_root=(
                        context.batch_root
                        / "provider_native_chronicle_checkpoints"
                        / (
                            f"tick-{segment_end_next_tick:012d}"
                            if tick_native
                            else f"generation-{end:06d}"
                        )
                    )
                )
                checkpoint_payload["provider_native_chronicle_receipt"] = (
                    prefix_receipt.payload()
                )
                checkpoint_payload["provider_native_reproduction_continuity"] = (
                    continuity
                )
                checkpoint_payload[
                    "provider_native_chronicle_checkpoint_authority_sha256"
                ] = checkpoint_authority
            synchronous_finalization_started_ns = time.perf_counter_ns()
            checkpoint_coordinate = str(segment_end_next_tick if tick_native else end)
            prior_catalog_snapshot = {
                coordinate: dict(value)
                for coordinate, value in checkpoint_catalog.items()
            }
            checkpoint_catalog.clear()
            checkpoint_catalog.update(retained_prior_catalog)
            checkpoint_catalog[checkpoint_coordinate] = dict(checkpoint_payload)
            synchronous_catalog_bound_ns = time.perf_counter_ns()
            synchronous_catalog_bound_unix_ns = time.time_ns()
            context.record_checkpoint(checkpoint_payload)
            synchronous_context_published_ns = time.perf_counter_ns()
            synchronous_context_published_unix_ns = time.time_ns()
            if bounded_capture_receipt is not None:
                bounded_capture_receipt["catalog_publication_proof"] = {
                    "state": "PUBLISHED",
                    "context_publication_completed_unix_ns": (
                        synchronous_context_published_unix_ns
                    ),
                    "catalog_bound_unix_ns": synchronous_catalog_bound_unix_ns,
                }
                bounded_capture_receipt["publication_state"] = "PUBLISHED"
            if (
                bounded_projection_cursor_present
                or terminal_physical_capture_route
            ):
                checkpoint_payload["checkpoint_set_commit"] = {
                    **dict(checkpoint_set["checkpoint_set_commit"]),
                    "state": "PUBLISHED",
                    "failure_state": None,
                }
                checkpoint_catalog[checkpoint_coordinate] = dict(checkpoint_payload)
                # The first record is the SEALED boundary; this second record
                # is the explicit catalog/frontier publication transition.
                context.record_checkpoint(checkpoint_payload)
            if tick_native:
                transient_storage_release_receipts.extend(
                    _release_tick_native_transient_storage_v1(
                        context=context,
                        prior_checkpoint_catalog=prior_catalog_snapshot,
                        retained_checkpoint_catalog=retained_prior_catalog,
                        replacement_checkpoint=checkpoint_payload,
                        requested_checkpoint_ticks=request.checkpoint_ticks,
                        release_source_import=bool(
                            owner_transition
                            or segment_end_next_tick == request.total_ticks
                            or segment_end_next_tick == deliberate_interruption_tick
                        ),
                    )
                )
            if acs_measurement_enabled:
                acs_catalog_publication_measurements.append(
                    {
                        "coordinate": int(checkpoint_coordinate),
                        "mode": "synchronous_instrumentation_only_v1",
                        "durable_completed_unix_ns": time.time_ns(),
                        "durable_completed_perf_ns": time.perf_counter_ns(),
                        "catalog_binding_time_ns": (
                            synchronous_catalog_bound_ns
                            - synchronous_finalization_started_ns
                        ),
                        "context_publication_time_ns": (
                            synchronous_context_published_ns
                            - synchronous_catalog_bound_ns
                        ),
                        "total_finalization_time_ns": (
                            synchronous_context_published_ns
                            - synchronous_finalization_started_ns
                        ),
                    }
                )
            current_next_tick = segment_end_next_tick
            current = end
            reload_paths = tuple(row["path"] for row in bundle_rows)
            if request.checkpoint_storage_mode == "disk":
                source_projections = None
            # The durable CPU projection now owns continuation.  Drop the
            # completed owner only at an owner transition, terminal boundary,
            # or deliberate interruption.  A declared checkpoint inside an
            # owner span must resume that same owner.
            executor_arguments.clear()
            abi_owner = None
            post_dispatch_owner = None
            source_projection_cursor_factory = None
            projection_cursor = None
            compact_source_projections = ()
            del prepared_arguments
            keep_owner = bool(
                tick_native
                and not owner_transition
                and segment_end_next_tick != deliberate_interruption_tick
                and segment_end_next_tick != request.total_ticks
            )
            owner_to_close = resident_owner if not keep_owner else None
            if not keep_owner:
                resident_owner = None
            del builds, requests, prepared, direct_result, settlement
            if owner_to_close is not None:
                owner_release_receipts.append(
                    release_resident_owner(owner_to_close)
                )
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            if request.checkpoint_storage_mode == "disk" and (
                current_next_tick < request.total_ticks
                if tick_native
                else current < request.generations
            ):
                source_projections = _lazy_projection_bundle_sequence_v1(
                    reload_paths,
                    self._read_projection_bundle,
                    source_projection_digests={
                        str(row["cell_id"]): str(row["source_projection_sha256"])
                        for row in bundle_rows
                        if isinstance(row, Mapping)
                        and isinstance(row.get("cell_id"), str)
                        and isinstance(row.get("source_projection_sha256"), str)
                        and row.get("physical_format")
                        != "eiel_acs_world_checkpoint_object_v1"
                    },
                    cell_ids=request.cell_ids,
                )
            if segment_end_next_tick == deliberate_interruption_tick:
                deliberately_interrupted = True
                break
        if acs_pipeline is not None:
            while pending_acs_checkpoints:
                finalize_oldest_acs_checkpoint()
            acs_pipeline.close()
            acs_final_telemetry = acs_pipeline.telemetry()
            acs_final_telemetry["capture_mode"] = (
                "detached_file_backed_native_v1"
                if tick_native_detached_acs
                else "legacy_callable_snapshot_v1"
            )
            self._active_acs_pipeline = None
            acs_close_completed_perf_ns = time.perf_counter_ns()
        else:
            acs_close_completed_perf_ns = None
        if checkpoint_payload is None:
            raise CampaignProviderError("no physical segment remained for dispatch")
        if tick_native and resident_owner_live_started_perf_ns:
            raise CampaignProviderError(
                "tick-native execution ended with a live resident CUDA owner"
            )
        generation_series_projector = getattr(self, "project_generation_series", None)
        if (
            preloaded_artifact is None
            and request.reporting_enabled
            and self._generation_series_required(plan)
            and callable(generation_series_projector)
        ):
            metrics_rows = list(
                generation_series_projector(
                    plan=plan,
                    cells=cells,
                    generic_patch_rows=metrics_rows,
                    native_exposure_segment_artifacts=(generation_series_artifacts),
                    checkpoint_catalog=checkpoint_catalog,
                    terminal_generation=current,
                )
            )
        if tick_native:
            native_execution_receipts = [
                _native_public_execution_receipt_v1(receipt)
                for receipt in native_execution_receipts
            ]
            authoritative_state_receipts = (
                _native_public_authoritative_state_receipts_v1(
                    authoritative_state_receipts
                )
            )
        native_exposure_receipt = _provider_native_exposure_receipt_v1(
            request=request,
            native_execution_receipts=native_execution_receipts,
            segment_artifacts=current_segment_artifacts,
            tick_native=tick_native,
        )
        result_payload: dict[str, Any] = {
            **(
                {
                    "time_authority": "absolute_tick_v1",
                    "terminal_next_absolute_tick": current_next_tick,
                    "resumed_from_next_absolute_tick": (
                        start_generation * interval_ticks
                        if start_next_tick is None
                        else start_next_tick
                    ),
                    "deliberate_fresh_owner_restore_pending": (
                        deliberately_interrupted
                    ),
                }
                if tick_native
                else {
                    "terminal_generation": current,
                    "resumed_from_generation": start_generation,
                }
            ),
            "native_dispatches": dispatches,
            "runner_receipts": receipts,
            "checkpoint_catalog": checkpoint_catalog,
            "native_execution_receipts": native_execution_receipts,
            "acs_capture_measurements": acs_capture_measurements,
            "acs_catalog_publication_measurements": (
                acs_catalog_publication_measurements
            ),
            "checkpoint_control_plane_transitions": (
                checkpoint_control_plane_transitions
            ),
            "bounded_terminal_capture_receipts": (
                bounded_terminal_capture_receipts
            ),
            "transient_storage_release_receipts": (
                transient_storage_release_receipts
            ),
            "authoritative_state_receipts": authoritative_state_receipts,
            "authoritative_analysis_sources": authoritative_analysis_sources,
            "lifecycle_qacct_receipts": lifecycle_qacct_receipts,
            "production_control_readbacks": production_control_readbacks,
            "resident_owner_preparations": resident_owner_preparations,
            "resident_owner_release_receipts": [
                receipt.payload() for receipt in owner_release_receipts
            ],
            "terminal_cuda_context_release_receipts": (
                terminal_cuda_context_release_receipts
            ),
            "resident_owner_epoch_transfer_receipts": [
                receipt.payload() for receipt in owner_epoch_transfer_receipts
            ],
            "resident_owner_lifetime_measurements": (
                resident_owner_lifetime_measurements
            ),
            "resident_owner_preparation_measurements": (
                resident_owner_preparation_measurements
            ),
            "resident_owner_max_simultaneous": (
                resident_owner_max_simultaneous
            ),
            "resident_owner_live_count_terminal": len(
                resident_owner_live_started_perf_ns
            ),
            "expected_resident_owner_transition_ticks": list(
                expected_resident_owner_transition_ticks
            ),
            "observed_resident_owner_transition_ticks": list(
                observed_resident_owner_transition_ticks
            ),
            "compact_segment_settlements": [
                {
                    "schema_version": settlement.schema_version,
                    "cell_ids": list(settlement.cell_ids),
                    "start_next_tick": settlement.start_next_tick,
                    "end_next_tick": settlement.end_next_tick,
                    "owner_identity": settlement.owner_identity,
                    "event_frontier_by_cell": list(
                        settlement.event_frontier_by_cell
                    ),
                    "event_pending_rows_by_cell": list(
                        settlement.event_pending_rows_by_cell
                    ),
                }
                for settlement in compact_settlements
            ],
        }
        if event_sink_reconciliation_receipt is not None:
            result_payload["event_sink_reconciliation_receipt"] = dict(
                event_sink_reconciliation_receipt
            )
        if acs_final_telemetry is not None:
            result_payload["acs_checkpoint_staging"] = acs_final_telemetry
        elif bounded_terminal_capture_receipts:
            latest_capture = bounded_terminal_capture_receipts[-1]
            result_payload["acs_checkpoint_staging"] = {
                "schema_version": "eiel_bounded_terminal_capture_staging_v1",
                "mode": "bounded_serial_cell_v1",
                "slots": 0,
                "writer_busy_time_ns": int(
                    latest_capture.get("writer_service_time_ns", 0)
                ),
                "checkpoint_bytes": int(latest_capture["checkpoint_bytes"]),
                "bounded_terminal_capture_receipts": (
                    bounded_terminal_capture_receipts
                ),
            }
        if native_exposure_receipt is not None:
            result_payload["provider_native_exposure_receipt"] = native_exposure_receipt
        if native_execution_receipts and all(
            row["explicit_native_no_fallback"] is True
            for row in native_execution_receipts
        ):
            result_payload["execution_device"] = native_execution_receipts[0][
                "execution_device"
            ]
            result_payload["fallback_used"] = False
        if qualification_sentinels is not None:
            result_payload["transition_sentinels"] = qualification_sentinels
        if native_chronicle is not None and not native_chronicle_observed:
            raise CampaignProviderError(
                "scientific_full requires provider-native Chronicle segments"
            )
        if native_chronicle_observed:
            if native_chronicle is None:
                raise CampaignProviderError(
                    "provider-native Chronicle observer was not initialized"
                )
            result_payload["provider_native_chronicle_required"] = True
            result_payload["provider_native_chronicle"] = (
                native_chronicle.finalize().payload()
            )
        if (
            tick_native
            and request.profile_id == PRODUCTION_PROFILE_ID
            and request.native_streamed_checkpoint
            and request.acs_checkpoint_slots > 0
            and request.acs_capture_instrumentation
        ):
            if type(acs_close_completed_perf_ns) is not int:
                raise CampaignProviderError(
                    "production tick-native timing lacks ACS close completion"
                )
            launch_completed_perf_ns = time.perf_counter_ns()
            result_payload["tick_native_performance_timeline_binding"] = (
                _tick_native_performance_timeline_binding_v1(
                    launch_started_perf_ns=execution_launch_started_perf_ns,
                    launch_completed_perf_ns=launch_completed_perf_ns,
                    native_execution_receipts=native_execution_receipts,
                    acs_capture_measurements=acs_capture_measurements,
                    acs_catalog_publication_measurements=(
                        acs_catalog_publication_measurements
                    ),
                    owner_preparation_measurements=(
                        resident_owner_preparation_measurements
                    ),
                    terminal_next_absolute_tick=current_next_tick,
                    acs_close_completed_perf_ns=acs_close_completed_perf_ns,
                    restore_receipt=restore_timing_receipt,
                )
            )
        return {
            "status": "interrupted" if deliberately_interrupted else "completed",
            "checkpoint": checkpoint_payload,
            "metrics_rows": metrics_rows,
            "result": result_payload,
        }

    def _execute_segments_with_acs_cleanup(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        plan = args[0] if args else kwargs.get("plan")
        request = args[1] if len(args) > 1 else kwargs.get("request")
        prior_native_extension = os.environ.get(
            ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1
        )
        bound_native_extension = False
        try:
            if (
                isinstance(request, ResidentBatchRequestV1)
                and request.native_streamed_checkpoint
            ):
                binding = _acs_native_runtime_artifact_binding_v1(
                    _plan_payload(plan),
                    required=self.profile_id == PRODUCTION_PROFILE_ID,
                )
                if binding is not None:
                    os.environ[ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1] = str(
                        binding["extension_path"]
                    )
                    bound_native_extension = True
            return self._execute_segments(*args, **kwargs)
        except BaseException:
            pipeline = getattr(self, "_active_acs_pipeline", None)
            if pipeline is not None:
                try:
                    pipeline.abort()
                finally:
                    self._active_acs_pipeline = None
            raise
        finally:
            if bound_native_extension:
                if prior_native_extension is None:
                    os.environ.pop(ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1, None)
                else:
                    os.environ[ACS_NATIVE_RUNTIME_ENVIRONMENT_VARIABLE_V1] = (
                        prior_native_extension
                    )

    def execute_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
    ) -> dict[str, Any]:
        """Execute each declared physical segment once under native ownership."""

        return self._execute_segments_with_acs_cleanup(
            _plan_payload(plan),
            request,
            context,
            start_generation=0,
            resume_projections=None,
        )

    def resume_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        checkpoint: Mapping[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        """Load strict bundles and continue in newly constructed owners."""

        resume_launch_started_perf_ns = time.perf_counter_ns()
        if checkpoint.get("schema_version") != self.checkpoint_schema_version:
            raise CampaignProviderError("unsupported provider checkpoint set")
        payload = _plan_payload(plan)
        expected_checkpoint_binding = self._checkpoint_binding(payload, request)
        if checkpoint.get("checkpoint_binding") != expected_checkpoint_binding:
            expected_contract = expected_checkpoint_binding.get("run_contract")
            actual_binding = checkpoint.get("checkpoint_binding")
            actual_contract = (
                actual_binding.get("run_contract")
                if isinstance(actual_binding, Mapping)
                else None
            )
            if expected_contract is not None and actual_contract is None:
                raise CampaignProviderError(
                    "strict run-contract checkpoint binding is absent"
                )
            if expected_contract != actual_contract:
                raise CampaignProviderError(
                    "strict run-contract checkpoint binding mismatch"
                )
            raise CampaignProviderError("provider checkpoint binding mismatch")
        chronicle_extension_fields = (
            "provider_native_chronicle_receipt",
            "provider_native_reproduction_continuity",
            "provider_native_chronicle_checkpoint_authority_sha256",
        )
        chronicle_extension_presence = tuple(
            field in checkpoint for field in chronicle_extension_fields
        )
        if any(chronicle_extension_presence):
            if not all(chronicle_extension_presence):
                raise CampaignProviderError(
                    "provider-native Chronicle checkpoint binding is incomplete"
                )
            from engine.campaigns.canonical import canonical_sha256

            checkpoint_core = dict(checkpoint)
            supplied_authority = checkpoint_core.pop(
                "provider_native_chronicle_checkpoint_authority_sha256"
            )
            checkpoint_core.pop("provider_native_chronicle_receipt")
            checkpoint_core.pop("provider_native_reproduction_continuity")
            if (
                not isinstance(supplied_authority, str)
                or canonical_sha256(checkpoint_core) != supplied_authority
            ):
                raise CampaignProviderError(
                    "provider-native Chronicle checkpoint authority differs"
                )
        checkpoint_storage_mode = checkpoint.get("continuation_storage_mode", "disk")
        if checkpoint_storage_mode not in {"disk", "disk_and_memory"}:
            raise CampaignProviderError(
                "checkpoint continuation storage mode is invalid"
            )
        if checkpoint_storage_mode != request.checkpoint_storage_mode:
            raise CampaignProviderError(
                "checkpoint continuation storage mode does not match the request"
            )
        tick_native = request.total_ticks is not None
        if tick_native:
            source_window = request.source_window_ticks
            if type(source_window) is not int or source_window < 1:
                raise CampaignProviderError(
                    "tick-native restore lacks its bounded source window"
                )
            checkpoint_next_tick = _validate_tick_native_checkpoint_identity_v1(
                checkpoint,
                execution_route_sha256=request.execution_route_sha256,
                compatibility_interval_ticks=source_window,
            )
            generation = checkpoint_next_tick // source_window
        else:
            generation = checkpoint.get("generation")
            checkpoint_next_tick = (
                generation * request.ticks_per_generation
                if type(generation) is int
                else None
            )
        rows = checkpoint.get("cell_bundles")
        if (
            (not tick_native and type(generation) is not int)
            or not isinstance(rows, list)
        ):
            raise CampaignProviderError("Patch checkpoint set is incomplete")
        by_id = {
            row.get("cell_id"): row
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("cell_id"), str)
            and isinstance(row.get("path"), str)
        }
        if (
            len(rows) != len(request.cell_ids)
            or len(by_id) != len(request.cell_ids)
            or tuple(by_id) != request.cell_ids
        ):
            raise CampaignProviderError(
                "Patch checkpoint cell order does not match the batch"
            )
        prior = checkpoint.get("metrics_rows", [])
        if not isinstance(prior, list) or not all(
            isinstance(row, Mapping) for row in prior
        ):
            raise CampaignProviderError(
                "Patch checkpoint metrics continuation is invalid"
            )
        projections = _lazy_projection_bundle_sequence_v1(
            tuple(str(by_id[cell_id]["path"]) for cell_id in request.cell_ids),
            self._read_projection_bundle,
            source_projection_digests={
                str(cell_id): str(by_id[cell_id]["source_projection_sha256"])
                for cell_id in request.cell_ids
                if isinstance(by_id[cell_id].get("source_projection_sha256"), str)
                and by_id[cell_id].get("physical_format")
                != "eiel_acs_world_checkpoint_object_v1"
            },
            cell_ids=request.cell_ids,
        )
        event_sink_reconciliation_receipt = (
            _reconcile_tick_native_resume_event_sink_v1(
                projections, expected_cells=len(request.cell_ids)
            )
            if tick_native
            else None
        )
        reload_receipts = self._fresh_owner_reload_receipts(
            request=request,
            checkpoint=checkpoint,
            bundle_rows=by_id,
            projections=projections,
        )
        if (
            checkpoint_next_tick == request.total_ticks
            if tick_native
            else generation == request.generations
        ):
            checkpoint_catalog = {
                **{
                    str(key): dict(value)
                    for key, value in (
                        checkpoint.get("prior_checkpoint_catalog", {}).items()
                        if isinstance(
                            checkpoint.get("prior_checkpoint_catalog"),
                            Mapping,
                        )
                        else ()
                    )
                },
                str(checkpoint_next_tick if tick_native else generation): dict(
                    checkpoint
                ),
            }
            recovered_metrics = [dict(row) for row in prior]
            generation_series_projector = getattr(
                self, "project_generation_series", None
            )
            if (
                request.reporting_enabled
                and self._generation_series_required(payload)
                and callable(generation_series_projector)
            ):
                prior_series_loader = getattr(
                    self, "load_prior_generation_series_artifacts", None
                )
                if not callable(prior_series_loader):
                    raise CampaignProviderError(
                        "required generation-series recovery loader is absent"
                    )
                recovered_metrics = list(
                    generation_series_projector(
                        plan=payload,
                        cells=self._batch_cells(payload, request),
                        generic_patch_rows=prior,
                        native_exposure_segment_artifacts=(
                            prior_series_loader(
                                checkpoint_catalog=checkpoint_catalog,
                                cell_ids=request.cell_ids,
                            )
                        ),
                        checkpoint_catalog=checkpoint_catalog,
                        terminal_generation=generation,
                    )
                )
            recovered_result: dict[str, Any] = {
                **(
                    {
                        "time_authority": "absolute_tick_v1",
                        "terminal_next_absolute_tick": checkpoint_next_tick,
                        "resumed_from_next_absolute_tick": checkpoint_next_tick,
                    }
                    if tick_native
                    else {
                        "terminal_generation": generation,
                        "resumed_from_generation": generation,
                    }
                ),
                "native_dispatches": 0,
                "runner_receipts": [],
                "terminal_checkpoint_recovered": True,
                "checkpoint_catalog": checkpoint_catalog,
                "fresh_owner_restore_receipt": (
                    self._fresh_owner_restore_receipt(
                        request=request,
                        checkpoint=checkpoint,
                        reload_receipts=reload_receipts,
                        terminal_cell_digests=checkpoint.get("terminal_cell_digests"),
                    )
                ),
            }
            if event_sink_reconciliation_receipt is not None:
                recovered_result["event_sink_reconciliation_receipt"] = dict(
                    event_sink_reconciliation_receipt
                )
            output = payload.get("output")
            if (
                isinstance(output, Mapping)
                and output.get("chronicle_profile") == "scientific_full"
            ):
                chronicle_receipt = checkpoint.get("provider_native_chronicle_receipt")
                if not isinstance(chronicle_receipt, Mapping):
                    raise CampaignProviderError(
                        "terminal checkpoint lacks required native Chronicle"
                    )
                recovered_result["provider_native_chronicle_required"] = True
                recovered_result["provider_native_chronicle"] = dict(chronicle_receipt)
            return {
                "status": "completed",
                "checkpoint": dict(checkpoint),
                "metrics_rows": recovered_metrics,
                "result": recovered_result,
            }
        adaptive_rebinding = checkpoint.get(WRC_ADAPTIVE_REBINDING_CHECKPOINT_FIELD_V1)
        if adaptive_rebinding is not None:
            from engine.campaigns.wrc_adaptive_rebinding import (
                WRCCheckpointAdaptiveRebindingError,
                materialize_wrc_adaptive_runtime_projections_v1,
                validate_wrc_checkpoint_adaptive_rebinding_receipt_v1,
            )

            try:
                validated_adaptive = (
                    validate_wrc_checkpoint_adaptive_rebinding_receipt_v1(
                        adaptive_rebinding
                    )
                )
                if (
                    validated_adaptive.get("execution_route_sha256")
                    != request.execution_route_sha256
                    or validated_adaptive.get("cell_ids") != list(request.cell_ids)
                    or validated_adaptive.get("source_generation") != generation
                    or validated_adaptive.get("restore_generation")
                    != request.generations
                ):
                    raise WRCCheckpointAdaptiveRebindingError(
                        "ADAPTIVE_REBINDING_AUTHORITY_MISMATCH: runtime request differs"
                    )
                projections = materialize_wrc_adaptive_runtime_projections_v1(
                    adaptive_rebinding=validated_adaptive,
                    source_projections=projections,
                )
            except WRCCheckpointAdaptiveRebindingError as exc:
                raise CampaignProviderError(str(exc)) from exc
        prior_catalog = checkpoint.get("prior_checkpoint_catalog")
        checkpoint_catalog = {
            str(key): dict(value)
            for key, value in (
                prior_catalog.items() if isinstance(prior_catalog, Mapping) else ()
            )
        }
        checkpoint_catalog[str(checkpoint_next_tick if tick_native else generation)] = (
            dict(checkpoint)
        )
        continuation_arguments: dict[str, Any] = {}
        if all(chronicle_extension_presence):
            continuation_arguments = {
                "prior_native_chronicle_receipt": checkpoint[
                    "provider_native_chronicle_receipt"
                ],
                "prior_native_continuity": checkpoint[
                    "provider_native_reproduction_continuity"
                ],
                "prior_chronicle_checkpoint_authority_sha256": checkpoint[
                    "provider_native_chronicle_checkpoint_authority_sha256"
                ],
            }
        restore_transport_completed_perf_ns = time.perf_counter_ns()
        result = self._execute_segments_with_acs_cleanup(
            payload,
            request,
            context,
            start_generation=generation,
            start_next_tick=(checkpoint_next_tick if tick_native else None),
            resume_projections=projections,
            prior_metrics_rows=prior,
            prior_checkpoint_catalog=checkpoint_catalog,
            event_sink_reconciliation_receipt=(event_sink_reconciliation_receipt),
            launch_started_perf_ns=resume_launch_started_perf_ns,
            restore_timing_receipt={
                "status": "completed",
                "checkpoint_tick": checkpoint_next_tick,
                "start_perf_ns": resume_launch_started_perf_ns,
                "end_perf_ns": restore_transport_completed_perf_ns,
            },
            **continuation_arguments,
        )
        provider_result = result.get("result")
        terminal_checkpoint = result.get("checkpoint")
        if not isinstance(provider_result, dict) or not isinstance(
            terminal_checkpoint, Mapping
        ):
            raise CampaignProviderError("fresh-owner continuation result is incomplete")
        if adaptive_rebinding is not None:
            from engine.campaigns.wrc_adaptive_rebinding import (
                WRCCheckpointAdaptiveRebindingError,
                validate_wrc_adaptive_runtime_readback_v1,
            )

            try:
                provider_result["wrc_checkpoint_adaptive_runtime_readback"] = (
                    validate_wrc_adaptive_runtime_readback_v1(
                        adaptive_rebinding=adaptive_rebinding,
                        production_control_readbacks=provider_result.get(
                            "production_control_readbacks", ()
                        ),
                    )
                )
            except WRCCheckpointAdaptiveRebindingError as exc:
                raise CampaignProviderError(str(exc)) from exc
        fresh_owner_receipt = self._fresh_owner_restore_receipt(
            request=request,
            checkpoint=checkpoint,
            reload_receipts=reload_receipts,
            terminal_cell_digests=terminal_checkpoint.get("terminal_cell_digests"),
        )
        if tick_native:
            execution_receipts = provider_result.get("native_execution_receipts")
            first_execution = (
                execution_receipts[0]
                if isinstance(execution_receipts, list) and execution_receipts
                else None
            )
            if not isinstance(first_execution, Mapping):
                raise CampaignProviderError(
                    "fresh-owner tick continuation lacks its first native dispatch"
                )
            first_executed = first_execution.get("first_executed_absolute_tick")
            first_committed = first_execution.get("first_committed_next_tick")
            if (
                first_executed != checkpoint_next_tick
                or first_committed != checkpoint_next_tick + 1
            ):
                raise CampaignProviderError(
                    "fresh-owner continuation is off by one at the restore cursor"
                )
            fresh_owner_receipt.update(
                {
                    "restore_next_tick": checkpoint_next_tick,
                    "first_executed_absolute_tick": first_executed,
                    "first_committed_next_tick": first_committed,
                    "off_by_one_check": "PASS",
                }
            )
        provider_result["fresh_owner_restore_receipt"] = fresh_owner_receipt
        return result

    def translate_batch(
        self, plan: CampaignPlanV1 | Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        payload = _plan_payload(plan)
        if payload.get("resolved_profile") != self.profile_id:
            raise CampaignProviderError("Patch fixed provider received another profile")
        raw_batches = payload.get("physical_batches")
        raw_cells = payload.get("cells")
        execution = payload.get("execution")
        if not isinstance(raw_batches, list) or not isinstance(raw_cells, list):
            raise CampaignProviderError("compiled plan is missing cells or batches")
        if not isinstance(execution, Mapping):
            raise CampaignProviderError("compiled plan is missing execution")
        batch = next(
            (
                row
                for row in raw_batches
                if isinstance(row, Mapping) and row.get("batch_id") == batch_id
            ),
            None,
        )
        if batch is None or not isinstance(batch.get("cell_ids"), (list, tuple)):
            raise CampaignProviderError(f"compiled batch is absent: {batch_id}")
        cell_ids = tuple(batch["cell_ids"])
        by_id = {
            row.get("cell_id"): row
            for row in raw_cells
            if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
        }
        try:
            cells = [by_id[cell_id] for cell_id in cell_ids]
        except KeyError as exc:
            raise CampaignProviderError(
                "compiled batch references an unknown cell"
            ) from exc
        parameter_names = sorted(
            {name for cell in cells for name in dict(cell.get("parameters", {}))}
        )
        seed_names = sorted(
            {name for cell in cells for name in dict(cell.get("seeds", {}))}
        )
        try:
            parameter_arrays = tuple(
                (
                    name,
                    tuple(dict(cell["parameters"])[name] for cell in cells),
                )
                for name in parameter_names
            )
            seed_arrays = tuple(
                (name, tuple(int(dict(cell["seeds"])[name]) for cell in cells))
                for name in seed_names
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CampaignProviderError("compiled batch arrays are incomplete") from exc
        tick_native = (
            execution.get("execution_route_identity")
            == "tick_native_execution_route_v1"
        )
        tick_control: TickNativeControlV1 | None = None
        total_ticks: int | None = None
        start_next_tick = 0
        checkpoint_ticks: tuple[int, ...] = ()
        if tick_native:
            try:
                tick_control = TickNativeControlV1.from_execution(execution)
            except TickNativeControlError as exc:
                raise CampaignProviderError(str(exc)) from exc
            total_ticks = tick_control.total_ticks
            start_next_tick = tick_control.start_next_tick
            generations = None
            ticks = None
            milestones: list[int] = []
            checkpoint_ticks = tick_control.resolved_checkpoint_ticks
            deliberate_restore_tick = execution.get("deliberate_restore_tick")
            deliberate_restore_ticks = tuple(
                execution.get("deliberate_restore_ticks") or ()
            )
        else:
            generations = execution.get("generations")
            ticks = execution.get("ticks_per_generation")
            milestones = execution.get("checkpoints")
            deliberate_restore_tick = None
            deliberate_restore_ticks = ()
        checkpoint_storage_mode = execution.get("checkpoint_storage_mode", "disk")
        acs_checkpoint_slots = execution.get("acs_checkpoint_slots", 0)
        raw_acs_capture_instrumentation = execution.get(
            "acs_capture_instrumentation", False
        )
        if not tick_native and (
            type(generations) is not int or type(ticks) is not int
        ):
            raise CampaignProviderError("compiled execution counts are invalid")
        if not tick_native and (
            not isinstance(milestones, list)
            or not all(
            type(item) is int for item in milestones
            )
        ):
            raise CampaignProviderError("compiled checkpoint milestones are invalid")
        if checkpoint_storage_mode not in {"disk", "disk_and_memory"}:
            raise CampaignProviderError("compiled checkpoint storage mode is invalid")
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
            0,
            1,
            2,
            3,
        }:
            raise CampaignProviderError("compiled ACS checkpoint slots are invalid")
        if type(raw_acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "compiled ACS capture instrumentation is invalid"
            )
        acs_capture_instrumentation = raw_acs_capture_instrumentation or bool(
            acs_checkpoint_slots
        )
        raw_native_streamed_checkpoint = execution.get(
            "native_streamed_checkpoint", False
        )
        if type(raw_native_streamed_checkpoint) is not bool:
            raise CampaignProviderError(
                "compiled native streamed checkpoint selection is invalid"
            )
        raw_world_sharded_checkpoint = execution.get(
            "world_sharded_checkpoint", False
        )
        if type(raw_world_sharded_checkpoint) is not bool:
            raise CampaignProviderError(
                "compiled world-sharded checkpoint selection is invalid"
            )
        if not tick_native:
            milestones = list(
                self._rolling_segment_ends(
                    generations=generations,
                    declared_milestones=tuple(milestones),
                )
            )
        execution_route_sha256 = self.execution_route_contract(
            payload
        ).execution_route_sha256
        genome_v2_checkpoint_authority_binding = (
            self._genome_v2_checkpoint_authority_binding(
                payload,
                execution_route_sha256=execution_route_sha256,
            )
        )
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=cell_ids,
            generations=generations,
            ticks_per_generation=ticks,
            checkpoint_milestones=tuple(milestones),
            parameter_arrays=parameter_arrays,
            seed_arrays=seed_arrays,
            agent_id_allocation_generation_count=(None if tick_native else generations),
            checkpoint_storage_mode=checkpoint_storage_mode,
            execution_route_sha256=execution_route_sha256,
            genome_v2_checkpoint_authority_binding=(
                genome_v2_checkpoint_authority_binding
            ),
            deliberate_restore_tick=deliberate_restore_tick,
            deliberate_restore_ticks=deliberate_restore_ticks,
            acs_checkpoint_slots=acs_checkpoint_slots,
            acs_capture_instrumentation=acs_capture_instrumentation,
            native_streamed_checkpoint=raw_native_streamed_checkpoint,
            world_sharded_checkpoint=raw_world_sharded_checkpoint,
            total_ticks=total_ticks,
            start_next_tick=start_next_tick,
            checkpoint_ticks=checkpoint_ticks,
            resident_owner_span_ticks=(
                tick_control.resident_owner_span_ticks
                if tick_control is not None
                else None
            ),
            native_dispatch_window_ticks=(
                tick_control.native_dispatch_window_ticks
                if tick_control is not None
                else None
            ),
            report_every_ticks=(
                tick_control.report_every_ticks if tick_control is not None else None
            ),
            science_sample_interval_ticks=(
                tick_control.science_sample_interval_ticks
                if tick_control is not None
                else None
            ),
            reducer_every_ticks=(
                tick_control.reducer_every_ticks if tick_control is not None else None
            ),
            archive_every_ticks=(
                tick_control.archive_every_ticks if tick_control is not None else None
            ),
            event_window_capacity=(
                tick_control.event_window_capacity if tick_control is not None else None
            ),
            source_window_ticks=(
                tick_control.source_window_ticks if tick_control is not None else None
            ),
            tick_native_control=(
                tick_control.payload() if tick_control is not None else {}
            ),
        )

    @classmethod
    def _rolling_segment_ends(
        cls,
        *,
        generations: int,
        declared_milestones: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Insert value-agnostic rolling boundaries inside the admitted envelope."""

        minimum = cls.resident_segment_generation_min
        maximum = cls.resident_segment_generation_max
        if minimum < 1 or maximum < minimum:
            raise CampaignProviderError("provider resident segment envelope is invalid")
        declared = tuple(sorted(set((*declared_milestones, generations))))
        if any(value < 0 or value > generations for value in declared):
            raise CampaignProviderError(
                "checkpoint milestones are outside the campaign"
            )
        declared = tuple(value for value in declared if value > 0)
        ends: list[int] = []
        left = 0
        for right in declared:
            gap = right - left
            if gap <= maximum:
                ends.append(right)
                left = right
                continue
            parts = (gap + maximum - 1) // maximum
            width, wider = divmod(gap, parts)
            if width < minimum:
                # Preserve the declared boundary; readiness will emit the
                # profile-specific fail-closed reason instead of hiding it.
                ends.append(right)
                left = right
                continue
            current = left
            for ordinal in range(parts):
                current += width + (1 if ordinal < wider else 0)
                ends.append(current)
            left = right
        return tuple(ends)

    def readiness_reasons(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]:
        reasons: list[UnsupportedReasonV1] = []
        payload = _plan_payload(plan)
        artifacts = payload.get("artifacts")
        required_artifacts = {
            "cuda_extension": "cuda_extension",
            "extension_manifest": "extension_manifest",
            "qualification_receipt": "qualification_receipt",
            "reporting_receipt": "reporting_receipt",
        }
        for name, kind in required_artifacts.items():
            row = artifacts.get(name) if isinstance(artifacts, Mapping) else None
            if (
                not isinstance(row, Mapping)
                or row.get("kind") != kind
                or row.get("abi") != "architecture_b_patch_p2_stage3_v2"
            ):
                reasons.append(
                    UnsupportedReasonV1(
                        "runtime_artifact_contract_incomplete",
                        f"artifacts.{name}",
                        "the Patch provider requires exact extension, manifest, "
                        "qualification, and reporting bindings for the stage3 v2 ABI",
                        self.profile_id,
                    )
                )
        qualification_row = (
            artifacts.get("qualification_receipt")
            if isinstance(artifacts, Mapping)
            else None
        )
        qualification_path = (
            qualification_row.get("path")
            if isinstance(qualification_row, Mapping)
            else None
        )
        try:
            qualification = json.loads(
                Path(str(qualification_path)).read_text(encoding="utf-8")
            )
            execution_contract = qualification["execution_contract"]
            artifact_contract = qualification["artifact"]
            generation_range = execution_contract["logical_generation_range"]
            live_batch_range = execution_contract["direct_live_physical_batch_range"]
            generation_min, generation_max = generation_range
            batch_min, batch_max = live_batch_range
            qualified_artifact_hash = artifact_contract["sha256"]
            qualified_manifest_hash = artifact_contract["manifest_sha256"]
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            reasons.append(
                UnsupportedReasonV1(
                    "runtime_qualification_receipt_unavailable",
                    "artifacts.qualification_receipt",
                    f"cannot read the declarative Patch qualification envelope: {exc}",
                    self.profile_id,
                )
            )
            return tuple(reasons)
        artifact_rows = artifacts if isinstance(artifacts, Mapping) else {}
        cuda_row = artifact_rows.get("cuda_extension")
        manifest_row = artifact_rows.get("extension_manifest")
        if (
            not isinstance(cuda_row, Mapping)
            or cuda_row.get("sha256") != qualified_artifact_hash
            or not isinstance(manifest_row, Mapping)
            or manifest_row.get("sha256") != qualified_manifest_hash
        ):
            reasons.append(
                UnsupportedReasonV1(
                    "runtime_qualification_artifact_binding_mismatch",
                    "artifacts",
                    "the campaign artifacts do not match the qualified Patch artifact",
                    self.profile_id,
                )
            )
        if request.total_ticks is not None:
            control = request.tick_native_control
            if not isinstance(control, Mapping):
                reasons.append(
                    UnsupportedReasonV1(
                        "runtime_tick_native_control_missing",
                        "execution.tick_native_control",
                        "the tick-native request lacks its canonical control object",
                        self.profile_id,
                    )
                )
                return tuple(reasons)
            owner_span = control.get("resident_owner_span_ticks")
            dispatch_window = control.get("native_dispatch_window_ticks")
            source_window = control.get("source_window_ticks")
            if (
                any(
                    type(value) is not int or value < 1
                    for value in (owner_span, dispatch_window, source_window)
                )
                or dispatch_window > owner_span
                or source_window > dispatch_window
            ):
                reasons.append(
                    UnsupportedReasonV1(
                        "runtime_tick_native_control_invalid",
                        "execution.tick_native_control",
                        "owner, dispatch, and source-window spans are inconsistent",
                        self.profile_id,
                    )
                )
                return tuple(reasons)
            # The resource receipt admits a bounded source-buffer envelope;
            # total_ticks remains the sole simulation-time authority.
            segment_generations = (
                max(
                    1,
                    (min(request.total_ticks, dispatch_window) + source_window - 1)
                    // source_window,
                ),
            )
        else:
            boundaries = tuple(
                sorted({0, *request.checkpoint_milestones, request.generations})
            )
            segment_generations = tuple(
                right - left for left, right in zip(boundaries, boundaries[1:])
            )
        if (
            type(generation_min) is not int
            or type(generation_max) is not int
            or not segment_generations
            or any(
                segment < generation_min or segment > generation_max
                for segment in segment_generations
            )
        ):
            reasons.append(
                UnsupportedReasonV1(
                    "runtime_generation_contract_out_of_range",
                    "checkpoints.milestones",
                    "each Patch rolling execution segment must contain logical "
                    f"generations in [{generation_min}, {generation_max}]",
                    self.profile_id,
                )
            )
        if (
            type(batch_min) is not int
            or type(batch_max) is not int
            or not batch_min <= len(request.cell_ids) <= batch_max
        ):
            reasons.append(
                UnsupportedReasonV1(
                    "runtime_physical_batch_not_live_admitted",
                    "execution.batch",
                    "the directly live-qualified Patch physical batch range is "
                    f"[{batch_min}, {batch_max}]; larger preparation support is not "
                    "execution authority",
                    self.profile_id,
                )
            )
        if not any(
            reason.code == "runtime_artifact_contract_incomplete" for reason in reasons
        ):
            reasons.extend(self._structural_readiness_reasons(plan, request))
        return tuple(reasons)

    def _structural_readiness_reasons(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]:
        """Validate receipt-owned world dimensions without provider constants."""

        payload = _plan_payload(plan)
        artifacts = payload.get("artifacts")
        row = (
            artifacts.get("qualification_receipt")
            if isinstance(artifacts, Mapping)
            else None
        )
        path = row.get("path") if isinstance(row, Mapping) else None
        try:
            receipt = json.loads(Path(str(path)).read_text(encoding="utf-8"))
            fixed = receipt["fixed_capacity"]
            grid = fixed["grid"]
            initial_population = fixed["initial_population"]
            resident_capacity = fixed["resident_capacity"]
            if "admitted_ticks_per_generation" in fixed:
                raw_ticks = fixed["admitted_ticks_per_generation"]
                if (
                    isinstance(raw_ticks, (str, bytes))
                    or not isinstance(raw_ticks, Sequence)
                    or not raw_ticks
                    or any(type(value) is not int or value < 1 for value in raw_ticks)
                    or len(set(raw_ticks)) != len(raw_ticks)
                ):
                    raise ValueError("qualified admitted tick set is malformed")
                admitted_ticks_per_generation = tuple(sorted(raw_ticks))
            else:
                ticks_per_generation = fixed["ticks_per_generation"]
                if type(ticks_per_generation) is not int or ticks_per_generation < 1:
                    raise ValueError("qualified fixed tick count is malformed")
                admitted_ticks_per_generation = (ticks_per_generation,)
            if (
                not isinstance(grid, list)
                or len(grid) != 2
                or grid[0] != grid[1]
                or any(type(value) is not int or value < 1 for value in grid)
                or any(
                    type(value) is not int or value < 1
                    for value in (initial_population, resident_capacity)
                )
                or resident_capacity < initial_population
            ):
                raise ValueError("qualified fixed capacity is malformed")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            return (
                UnsupportedReasonV1(
                    "runtime_qualification_receipt_unavailable",
                    "artifacts.qualification_receipt",
                    f"cannot read the qualified structural envelope: {exc}",
                    self.profile_id,
                ),
            )
        if (
            receipt.get("schema_version") != self.qualification_schema_version
            or receipt.get("ok") is not True
            or receipt.get("provider_id") != self.provider_id
            or receipt.get("profile_id") != self.profile_id
        ):
            return (
                UnsupportedReasonV1(
                    "runtime_qualification_receipt_identity_mismatch",
                    "artifacts.qualification_receipt",
                    "the qualification receipt does not bind this provider/profile",
                    self.profile_id,
                ),
            )
        reasons: list[UnsupportedReasonV1] = []
        parameter_arrays = dict(request.parameter_arrays)
        for parameter_name, qualified_value, label in (
            ("patch.grid_size", grid[0], "square world side"),
            ("patch.initial_population", initial_population, "initial population"),
            ("patch.capacity", resident_capacity, "resident capacity"),
        ):
            requested_values = parameter_arrays.get(parameter_name, ())
            if not requested_values or any(
                value != qualified_value for value in requested_values
            ):
                reasons.append(
                    UnsupportedReasonV1(
                        "runtime_fixed_capacity_mismatch",
                        f"parameters.{parameter_name}",
                        f"the qualification receipt fixes {label} at "
                        f"{qualified_value}; requested values are "
                        f"{sorted(set(requested_values), key=repr)!r}",
                        self.profile_id,
                    )
                )
        if request.ticks_per_generation not in admitted_ticks_per_generation:
            reasons.append(
                UnsupportedReasonV1(
                    "runtime_ticks_contract_mismatch",
                    "execution.ticks_per_generation",
                    "the qualification receipt admits ticks per generation "
                    f"{list(admitted_ticks_per_generation)!r}",
                    self.profile_id,
                )
            )
        return tuple(reasons)


class ExecutionProviderRegistryV1:
    """Explicit localized provider registry; no discovery or central switch."""

    def __init__(self) -> None:
        self._providers: dict[str, CampaignExecutionProviderV1] = {}

    def register(self, provider: CampaignExecutionProviderV1) -> None:
        if provider.profile_id in self._providers:
            raise ValueError(f"duplicate execution provider: {provider.profile_id}")
        self._providers[provider.profile_id] = provider

    def get(self, profile_id: str) -> CampaignExecutionProviderV1 | None:
        return self._providers.get(profile_id)

    def providers(self) -> tuple[CampaignExecutionProviderV1, ...]:
        return tuple(self._providers[name] for name in sorted(self._providers))


def default_execution_provider_registry() -> ExecutionProviderRegistryV1:
    """Return the production registry containing exactly one engine provider."""

    registry = ExecutionProviderRegistryV1()
    from .production_engine_provider import (
        EielProductionExecutionProviderV1,
    )

    registry.register(EielProductionExecutionProviderV1(bind_committed_defaults=True))
    registry.register(
        EielProductionExecutionProviderV1(
            bind_committed_defaults=True,
            device="cpu",
            require_prebuilt=False,
            profile_id=PRODUCTION_CPU_PROFILE_ID,
        )
    )
    return registry


__all__ = [
    "CampaignExecutionProviderV1",
    "CampaignProviderError",
    "ExecutionProviderRegistryV1",
    "ProductionEngineLifecycleV1",
    "ResidentBatchRequestV1",
    "default_execution_provider_registry",
]
