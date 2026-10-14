"""Durable, provider-owned execution for compiled declarative campaigns.

This module owns only campaign/batch lifecycle and receipts.  Runtime mechanics
remain behind the explicitly registered provider boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import hashlib
import shutil
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from engine.qrr.execution_request_v1 import (
    ExecutionRequestDerivationV1,
    ExecutionRequestError,
    build_execution_request_v1,
    derive_resident_batch_request_v1,
)

from .providers import ExecutionProviderRegistryV1
from .providers import ResidentBatchRequestV1
from .artifact_lifecycle import (
    ArtifactLifecycleError,
    RunStorageGuard,
    StorageBudgetExceeded,
    build_artifact_disposition_manifest_v1,
    build_run_artifact_failure_receipt_v1,
    build_run_passport_v1,
    build_sealed_artifact_inventory_row_v1,
    build_storage_disposition_receipt_v1,
    checkpoint_role_for_generation,
    checkpoint_role_for_tick,
    ensure_lifecycle_contract_v1,
    progressive_storage_reservation_window_bytes_v1,
    validate_artifact_disposition_manifest_v1,
    write_run_artifact_failure_receipt_v1,
    write_run_passport_v1,
    write_storage_disposition_receipt_v1,
    register_pending_durable_archive_v1,
    run_artifact_transaction_fence_v1,
)
from .storage_reclamation import reclaim_before_storage_wait_v1
from .route_sentinels import (
    RouteTransitionSentinelV1,
    first_route_transition_difference,
)


def _tick_native_or_legacy_generation_horizon_v1(
    execution: Any,
) -> tuple[int, list[Any]]:
    """Return a legacy comparison horizon without inventing native generations."""

    if not isinstance(execution, Mapping):
        raise CampaignExecutionError("compiled comparison execution is invalid")
    generations = execution.get("generations")
    checkpoints = execution.get("checkpoints", [])
    if type(generations) is int and isinstance(checkpoints, list):
        return generations, list(checkpoints)
    if execution.get("execution_route_identity") == "tick_native_execution_route_v1":
        raise CampaignExecutionError(
            "tick-native comparison branches require an absolute-tick comparator"
        )
    raise CampaignExecutionError("compiled comparison execution is invalid")


class CampaignExecutionError(RuntimeError):
    """A durable campaign cannot be dispatched or resumed safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignExecutionError(f"{name} must be an object")
    return dict(value)


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignExecutionError(f"cannot read {name}: {path}: {exc}") from exc
    return _json_object(value, name=name)


def _gpu_slot_admission_metadata_v1(
    root: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Project route and Scheduler occupancy into fail-open slot telemetry."""

    execution = plan.get("execution")
    policy = (
        execution.get("gpu_resource_policy") if isinstance(execution, Mapping) else None
    )
    shared = isinstance(policy, Mapping)
    estimate = plan.get("resource_estimate")
    peak_bytes = (
        estimate.get("vram_bytes_upper_bound")
        if isinstance(estimate, Mapping)
        else None
    )
    peak_vram_gib = (
        float(policy["peak_vram_gib"])
        if shared and isinstance(policy.get("peak_vram_gib"), (int, float))
        else (
            float(peak_bytes) / float(1024**3)
            if type(peak_bytes) is int and peak_bytes > 0
            else None
        )
    )
    result: dict[str, Any] = {
        "gpu_admission_mode": "shared_functional" if shared else "exclusive",
        "exclusive_gpu": not shared,
        "peak_vram_gib": peak_vram_gib,
        "workload_class": (
            "cuda_full_evolution_v2_shared_functional"
            if shared
            else "campaign_execution"
        ),
    }
    try:
        pointer = _read_object(
            root / "control" / "crca" / "attempt-authority.json",
            name="operational Scheduler Attempt pointer",
        )
        bundle = _read_object(
            Path(str(pointer["scheduler_bundle_path"])).resolve(),
            name="operational Scheduler Attempt bundle",
        )
        lease = _json_object(
            bundle.get("resource_start_lease"),
            name="operational Scheduler resource lease",
        )
        fence = lease.get("fencing_token_digest")
        claim = _json_object(
            bundle.get("claim"), name="operational Scheduler Attempt claim"
        )
        result.update(
            {
                "gpu_device_id": str(claim["device_id"]),
                "scheduler_resource_lease_id": str(lease["lease_id"]),
                "scheduler_resource_ids": tuple(lease["resource_ids"]),
                "scheduler_fencing_token_digest": (
                    str(fence["hex"]) if isinstance(fence, Mapping) else ""
                ),
            }
        )
    except (CampaignExecutionError, KeyError, OSError, TypeError, ValueError):
        # The heartbeat is operational only; Scheduler remains authoritative.
        pass
    return result


def _absolute_checkpoint_tick_v1(checkpoint: Mapping[str, Any]) -> int | None:
    names = (
        "checkpoint_at_tick",
        "next_absolute_tick",
        "absolute_tick",
        "next_tick",
    )
    supplied = {
        name: checkpoint[name]
        for name in names
        if name in checkpoint and checkpoint[name] is not None
    }
    if any(type(value) is not int for value in supplied.values()):
        raise CampaignExecutionError(
            "absolute-tick checkpoint coordinates must be integers"
        )
    coordinates = set(supplied.values())
    if len(coordinates) > 1:
        raise CampaignExecutionError("absolute-tick checkpoint coordinates conflict")
    return next(iter(coordinates), None)


def _native_checkpoint_next_absolute_tick_v1(
    checkpoint: Mapping[str, Any],
) -> int:
    """Require one exact native cursor; aliases may only mirror that cursor."""

    next_absolute_tick = checkpoint.get("next_absolute_tick")
    if (
        checkpoint.get("time_authority") != "absolute_tick_v1"
        or type(next_absolute_tick) is not int
        or next_absolute_tick < 0
    ):
        raise CampaignExecutionError(
            "tick-native checkpoint lacks canonical next_absolute_tick authority"
        )
    if _absolute_checkpoint_tick_v1(checkpoint) != next_absolute_tick:
        raise CampaignExecutionError(
            "tick-native checkpoint absolute-tick coordinates conflict"
        )
    return next_absolute_tick


def _interrupted_checkpoint_next_tick_v1(
    checkpoint: Mapping[str, Any],
) -> int:
    """Normalize one provider interruption checkpoint for the front door.

    The front door's historical receipt field is named
    ``interrupted_checkpoint_next_tick``.  For an absolute tick checkpoint,
    ``next_absolute_tick`` is the authoritative coordinate; legacy aliases
    are accepted only as equal compatibility projections and any conflict is
    rejected.  A checkpoint without an explicit absolute-tick authority stays
    on the legacy ``next_tick`` contract rather than being guessed into the
    native route.
    """

    if checkpoint.get("time_authority") == "absolute_tick_v1":
        return _native_checkpoint_next_absolute_tick_v1(checkpoint)

    next_tick = checkpoint.get("next_tick")
    if type(next_tick) is not int:
        raise CampaignExecutionError(
            "legacy interrupted checkpoint lacks next_tick authority"
        )
    return next_tick


def _lifecycle_checkpoint_catalog(
    run_root: Path,
    batches: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    lifecycle_checkpoints = lifecycle.get("checkpoints")
    native_lifecycle = (
        isinstance(lifecycle_checkpoints, Mapping)
        and lifecycle_checkpoints.get("authority") == "absolute_tick"
    )
    coordinate_name = "checkpoint_tick" if native_lifecycle else "generation"
    for batch in batches:
        checkpoint = batch.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            continue
        checkpoint_authority = lifecycle_checkpoints.get(
            "authority", "legacy_generation"
        )
        if checkpoint_authority == "absolute_tick":
            terminal_coordinate = _native_checkpoint_next_absolute_tick_v1(checkpoint)
            coordinates = [terminal_coordinate]
            aggregate_coordinates = checkpoint.get("checkpoint_ticks")
            if aggregate_coordinates is not None:
                if (
                    not isinstance(aggregate_coordinates, list)
                    or not aggregate_coordinates
                    or any(
                        type(value) is not int or value < 0
                        for value in aggregate_coordinates
                    )
                    or aggregate_coordinates != sorted(set(aggregate_coordinates))
                ):
                    raise CampaignExecutionError(
                        "tick-native aggregate checkpoint schedule is invalid"
                    )
                lifecycle_coordinates = lifecycle_checkpoints.get("checkpoint_ticks")
                if (
                    aggregate_coordinates != lifecycle_coordinates
                    or aggregate_coordinates[-1] != terminal_coordinate
                ):
                    raise CampaignExecutionError(
                        "tick-native aggregate checkpoint schedule conflicts with lifecycle"
                    )
                coordinates = list(aggregate_coordinates)
        else:
            coordinate = checkpoint.get("generation")
            if type(coordinate) is not int:
                continue
            coordinates = [coordinate]
            terminal_coordinate = coordinate
        for coordinate in coordinates:
            if checkpoint_authority == "absolute_tick":
                role = checkpoint_role_for_tick(lifecycle, coordinate)
                coordinate_row = {"checkpoint_tick": coordinate}
                fallback_identity = f"checkpoint-tick-{coordinate:012d}"
            else:
                role = checkpoint_role_for_generation(lifecycle, coordinate)
                coordinate_row = {"generation": coordinate}
                fallback_identity = f"checkpoint-{coordinate:09d}"
            row = {
                "identity": role.get("identity") if role else fallback_identity,
                **coordinate_row,
                "role": role.get("role") if role else "unknown",
                "retention_class": role.get("retention_class") if role else None,
            }
            if coordinate == terminal_coordinate:
                for source, target in (
                    ("checkpoint_path", "path"),
                    ("checkpoint_file_sha256", "sha256"),
                    ("state_sha256", "state_sha256"),
                    ("terminal_digest", "state_sha256"),
                ):
                    if source in checkpoint and target not in row:
                        row[target] = checkpoint[source]
            if isinstance(row.get("path"), str):
                try:
                    row["path"] = str(
                        Path(row["path"]).resolve().relative_to(run_root.resolve())
                    )
                except ValueError:
                    row["path"] = str(row["path"])
            catalog.append(row)
    unique = {
        (int(row.get(coordinate_name, -1)), str(row.get("identity"))): row
        for row in catalog
    }
    return [unique[key] for key in sorted(unique)]


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace one JSON receipt atomically and durably on the local filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _batch_path(run_root: Path, batch_id: str) -> Path:
    if not batch_id or Path(batch_id).name != batch_id or batch_id in {".", ".."}:
        raise CampaignExecutionError(f"unsafe batch id: {batch_id!r}")
    return run_root / "batches" / batch_id / "manifest.json"


def _execution_route_sha256(provider: Any, plan: Mapping[str, Any]) -> str | None:
    owner = getattr(provider, "execution_route_contract", None)
    if not callable(owner):
        return None
    contract = owner(plan)
    value = getattr(contract, "execution_route_sha256", None)
    if not isinstance(value, str) or len(value) != 64:
        raise CampaignExecutionError(
            f"provider {provider.provider_id!r} returned an invalid execution route"
        )
    return value


def _chronicle_archive_run_id(
    run_root: Path,
    *,
    campaign_id: str,
    plan_sha256: str,
    profile_id: str,
    provider_id: str,
    execution_route_sha256: str | None,
) -> str:
    """Return the collision fallback bound to immutable execution authority."""

    identity = _canonical_sha256(
        {
            "schema_version": "eiel_chronicle_archive_run_identity_v2",
            "campaign_id": campaign_id,
            "plan_sha256": plan_sha256,
            "profile_id": profile_id,
            "provider_id": provider_id,
            "execution_route_sha256": execution_route_sha256,
        }
    )
    return f"{run_root.name}-v2-{identity[:16]}"


def _bind_checkpoint_route(
    checkpoint: Mapping[str, Any],
    execution_route_sha256: str | None,
    *,
    require_existing: bool = False,
) -> dict[str, Any]:
    value = dict(checkpoint)
    if execution_route_sha256 is None:
        return value
    observed = value.get("execution_route_sha256")
    if require_existing and observed is None:
        raise CampaignExecutionError(
            "checkpoint execution-route binding is missing before restore or dispatch"
        )
    if observed is not None and observed != execution_route_sha256:
        raise CampaignExecutionError(
            "checkpoint execution-route binding mismatch before restore or dispatch"
        )
    value["execution_route_sha256"] = execution_route_sha256
    return value


def _compact_resident_checkpoint_settlement_required_v1(
    plan: Mapping[str, Any], batch: Mapping[str, Any]
) -> bool:
    execution = plan.get("execution")
    lifecycle = plan.get("artifact_lifecycle")
    estimate = plan.get("resource_estimate")
    cells = plan.get("cells")
    cell_ids = batch.get("cell_ids")
    physical_batch_cells = (
        estimate.get("physical_batch_cells") if isinstance(estimate, Mapping) else None
    )
    logical_cells = (
        estimate.get("logical_cells") if isinstance(estimate, Mapping) else len(cells)
    )
    resident_batched = (
        type(physical_batch_cells) is int
        and type(logical_cells) is int
        and physical_batch_cells < logical_cells
    )
    return bool(
        isinstance(execution, Mapping)
        and isinstance(lifecycle, Mapping)
        and lifecycle.get("lifecycle_profile") == "SCIENCE_COMPACT"
        and (
            execution.get("backend") == "cuda_resident"
            or (
                execution.get("execution_route_identity")
                == "tick_native_execution_route_v1"
                and resident_batched
            )
        )
        and isinstance(cells, Sequence)
        and not isinstance(cells, (str, bytes))
        and isinstance(cell_ids, Sequence)
        and not isinstance(cell_ids, (str, bytes))
        and (resident_batched or len(cell_ids) < len(cells))
    )


def _rewrite_compact_resident_checkpoint_paths_v1(
    value: Any, source_root: Path, durable_root: Path
) -> Any:
    source = str(source_root.resolve())
    durable = str(durable_root.resolve())
    if isinstance(value, str) and (
        value == source or value.startswith(f"{source}{os.sep}")
    ):
        return f"{durable}{value[len(source) :]}"
    if isinstance(value, Mapping):
        return {
            key: _rewrite_compact_resident_checkpoint_paths_v1(
                item, source_root, durable_root
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_compact_resident_checkpoint_paths_v1(
                item, source_root, durable_root
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _rewrite_compact_resident_checkpoint_paths_v1(
                item, source_root, durable_root
            )
            for item in value
        )
    return value


def _compact_resident_checkpoint_settle_v1(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Move completed resident-batch checkpoints out of the active quota.

    A completed physical batch no longer needs its Bundle-v2 directory or
    native aggregate files in the active checkpoint zone: its terminal
    checkpoint remains authoritative, but the next physical batch owns the
    only live checkpoint materialization.
    Settlement is deliberately a filesystem move plus manifest path rewrite,
    so it does not create a second host staging copy or alter checkpoint bytes.
    """

    if not _compact_resident_checkpoint_settlement_required_v1(plan, batch):
        return dict(manifest)
    if manifest.get("status") != "completed":
        return dict(manifest)
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise CampaignExecutionError(
            f"completed compact resident batch {batch['batch_id']} lacks a checkpoint"
        )
    bundle_rows = checkpoint.get("cell_bundles")
    if (
        not isinstance(bundle_rows, list)
        or not bundle_rows
        or not all(isinstance(row, Mapping) for row in bundle_rows)
    ):
        raise CampaignExecutionError(
            f"completed compact resident batch {batch['batch_id']} lacks Bundle-v2 rows"
        )

    physical_formats = {
        row.get("physical_format")
        for row in bundle_rows
        if isinstance(row.get("physical_format"), str)
    }
    native_aggregate = physical_formats == {"acs100x_native_aggregate_checkpoint_v2"}
    if physical_formats and not native_aggregate:
        raise CampaignExecutionError(
            f"completed compact resident batch {batch['batch_id']} has an "
            "unsupported or mixed checkpoint physical format"
        )

    batch_root = manifest_path.parent.resolve()
    if native_aggregate:
        source_root = batch_root / "checkpoint_native_aggregate_v1"
        durable_root = (
            manifest_path.parents[2]
            / "durable"
            / "checkpoint_native_aggregate_v1"
            / str(batch["batch_id"])
        ).resolve()
    else:
        source_root = batch_root / "checkpoint_bundles_v2"
        durable_root = (
            manifest_path.parents[2] / "durable" / "checkpoint_bundles_v2"
        ).resolve()
    source_exists = source_root.is_dir()
    if not source_exists and not durable_root.is_dir():
        raise CampaignExecutionError(
            f"compact resident checkpoint bundle root is missing for {batch_root}"
        )

    source_root = source_root.resolve()
    durable_root = durable_root.resolve()
    source_paths: list[tuple[str, Path]] = []
    for row in bundle_rows:
        raw_path = row.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise CampaignExecutionError(
                f"compact resident checkpoint row has no path in {batch_root}"
            )
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(source_root if source_exists else durable_root)
        except ValueError as exc:
            raise CampaignExecutionError(
                f"compact resident checkpoint path escapes its storage root: {path}"
            ) from exc
        if not (path.is_file() if native_aggregate else path.is_dir()):
            raise CampaignExecutionError(
                f"compact resident checkpoint authority is missing: {path}"
            )
        source_paths.append((raw_path, path))

    bytes_before = (
        sum(
            file_path.stat().st_size
            for file_path in source_root.rglob("*")
            if file_path.is_file()
        )
        if source_exists
        else 0
    )
    if source_exists:
        durable_root.parent.mkdir(parents=True, exist_ok=True)
        durable_root.mkdir(parents=True, exist_ok=True)
        for child in sorted(source_root.iterdir(), key=lambda item: item.name):
            destination = durable_root / child.name
            if destination.exists():
                raise CampaignExecutionError(
                    f"compact resident checkpoint durable cell already exists: {destination}"
                )
            shutil.move(str(child), str(destination))
        source_root.rmdir()

    rewritten = _rewrite_compact_resident_checkpoint_paths_v1(
        dict(manifest), source_root, durable_root
    )
    if not isinstance(rewritten, dict):
        raise CampaignExecutionError("settled batch manifest is not an object")
    for row in rewritten["checkpoint"]["cell_bundles"]:
        settled_path = Path(str(row["path"]))
        if not (settled_path.is_file() if native_aggregate else settled_path.is_dir()):
            raise CampaignExecutionError(
                f"settled compact resident checkpoint path is missing: {row['path']}"
            )
    atomic_write_json(manifest_path, rewritten)

    receipt_path = batch_root / "checkpoint_settlement_receipt.json"
    if receipt_path.is_file():
        return rewritten
    durable_bytes = sum(
        file_path.stat().st_size
        for file_path in durable_root.rglob("*")
        if file_path.is_file()
    )
    path_bindings = [
        {
            "cell_id": row.get("cell_id"),
            "source_path": raw_path,
            "durable_path": row.get("path"),
        }
        for (raw_path, _), row in zip(
            source_paths, rewritten["checkpoint"]["cell_bundles"], strict=True
        )
    ]
    receipt = {
        "schema_version": "eiel_compact_resident_checkpoint_settlement_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "batch_id": batch["batch_id"],
        "status": "settled",
        "storage_mode": "durable_after_batch",
        "checkpoint_storage_kind": (
            "native_aggregate_file_v1" if native_aggregate else "bundle_v2_directory"
        ),
        "retention_reason": (
            "completed physical batch is no longer active; preserve terminal "
            "checkpoint authority without charging the active checkpoint quota"
        ),
        "source_root": str(source_root),
        "durable_root": str(durable_root),
        "bundle_count": len(path_bindings),
        "bytes_moved": bytes_before,
        "durable_bytes": durable_bytes,
        "path_bindings": path_bindings,
        "verified": True,
        "settled_at_utc": _now(),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    atomic_write_json(receipt_path, receipt)
    return rewritten


def _safe_id(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise CampaignExecutionError(f"unsafe {name}: {value!r}")
    return value


def _comparison_path(run_root: Path, branch_id: str) -> Path:
    return (
        run_root
        / "comparisons"
        / _safe_id(branch_id, name="comparison branch id")
        / "manifest.json"
    )


def _batch_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("physical_batches")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise CampaignExecutionError("compiled plan has no physical batches")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw:
        row = _json_object(value, name="physical batch")
        batch_id = row.get("batch_id")
        cell_ids = row.get("cell_ids")
        if not isinstance(batch_id, str) or batch_id in seen:
            raise CampaignExecutionError(
                "compiled plan has invalid or duplicate batch ids"
            )
        if (
            not isinstance(cell_ids, Sequence)
            or isinstance(cell_ids, (str, bytes))
            or not all(isinstance(item, str) for item in cell_ids)
        ):
            raise CampaignExecutionError(f"batch {batch_id!r} has invalid cell ids")
        _batch_path(Path("."), batch_id)
        seen.add(batch_id)
        rows.append({"batch_id": batch_id, "cell_ids": list(cell_ids)})
    return rows


def _new_batch_manifest(
    *,
    plan: Mapping[str, Any],
    provider: Any,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    now = _now()
    execution_route_sha256 = _execution_route_sha256(provider, plan)
    return {
        "schema_version": "eiel_campaign_batch_manifest_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": execution_route_sha256,
        "batch_id": batch["batch_id"],
        "cell_ids": list(batch["cell_ids"]),
        "status": "pending",
        "attempts": 0,
        "checkpoint": None,
        "artifact_inventory": [],
        "metrics_rows": [],
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }


def _load_batch_manifest(
    path: Path,
    *,
    plan: Mapping[str, Any],
    provider: Any,
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        manifest = _new_batch_manifest(plan=plan, provider=provider, batch=batch)
        atomic_write_json(path, manifest)
        return manifest
    manifest = _read_object(path, name="batch manifest")
    expected = {
        "schema_version": "eiel_campaign_batch_manifest_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "batch_id": batch["batch_id"],
        "cell_ids": list(batch["cell_ids"]),
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise CampaignExecutionError(
                f"batch manifest binding mismatch for {batch['batch_id']}: {name}"
            )
    return manifest


def _lifecycle_inventory_retention_v1(relative: str) -> tuple[str, str, str]:
    """Classify non-checkpoint producer outputs without a second policy owner."""

    normalized = relative.replace("\\", "/")
    lower = normalized.lower()
    name = Path(normalized).name.lower()
    if lower.startswith(("control/", "closeout/")) or any(
        token in name
        for token in (
            "manifest",
            "receipt",
            "passport",
            "contract",
            "preflight",
            "metrics",
            "config",
            "completion",
            "execution_plan",
            "execution_request",
            "canopy_relay_supervision",
        )
    ):
        return "R5", "lifecycle.control", "control/closeout authority is retained"
    if lower.startswith("durable/") or any(
        token in lower
        for token in (
            "compact_generation_summaries",
            "evidence_closeout",
            "chronicle_archive",
        )
    ):
        return (
            "R3",
            "lifecycle.durable",
            "durable replacement/evidence material is retained",
        )
    if (
        lower.startswith(("scratch/", "cache/", "quarantine/"))
        or "/scratch/" in lower
        or any(
            token in lower
            for token in ("trace", "debug", "event_jsonl", "funnel_events")
        )
    ):
        return "R0", "lifecycle.scratch", "terminal unreferenced scratch is releaseable"
    return "RX", "lifecycle.unknown", "producer purpose is unresolved"


def _collect_lifecycle_inventory_v1(
    *,
    run_root: Path,
    batch_manifests: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    terminal_status: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Collect producer-sealed checkpoints plus one bounded terminal sweep.

    The sweep never descends into a checkpoint bundle already sealed by the
    provider.  It only hashes the remaining run-local control/derived files so
    closeout does not re-read the large checkpoint payloads.
    """

    root = run_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    # Keep exact file coverage separate from sealed-tree coverage.  Treating
    # every producer file as a "covered root" makes the terminal sweep compare
    # each run path against every inventory row (O(files * inventory)).  A
    # 10k-tick direct-sparse run exposes that quadratic closeout cost.  Exact
    # file rows are membership lookups; only the small number of sealed trees
    # needs an ancestor check.
    covered_files: set[Path] = set()
    covered_trees: list[Path] = []
    seen: set[str] = set()
    complete = True
    for batch in batch_manifests:
        if batch.get("artifact_inventory_complete") is False:
            complete = False
        raw_inventory = batch.get("artifact_inventory")
        if not isinstance(raw_inventory, list):
            complete = False
            continue
        for row in raw_inventory:
            if not isinstance(row, Mapping):
                complete = False
                continue
            value = dict(row)
            relative = str(value.get("path") or "").replace("\\", "/").strip("/")
            if not relative or relative in seen:
                complete = False
                continue
            seen.add(relative)
            rows.append(value)
            candidate = (root / relative).resolve()
            if value.get("artifact_type") == "sealed_tree" and candidate.is_dir():
                covered_trees.append(candidate)
            elif value.get("artifact_type") == "file" and candidate.is_file():
                covered_files.add(candidate)

        batch_id = batch.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            complete = False
            continue
        batch_root = (root / "batches" / batch_id).resolve()
        checkpoint = batch.get("checkpoint")
        replacement_identity = (
            checkpoint.get("checkpoint_identity_sha256")
            if isinstance(checkpoint, Mapping)
            else None
        )
        producer_completed = batch.get("status") == "completed"
        for directory_name, retention_class, family in (
            ("cells", "R3", "campaign.cell_outputs"),
            ("event_history_v1", "R3", "campaign.event_history"),
            ("source_import_v1", "R1", "campaign.source_import"),
        ):
            unresolved_candidate = batch_root / directory_name
            if unresolved_candidate.is_symlink():
                complete = False
                continue
            candidate = unresolved_candidate.resolve()
            if not candidate.is_dir():
                continue
            try:
                row = build_sealed_artifact_inventory_row_v1(
                    run_root=root,
                    path=candidate,
                    artifact_family=family,
                    retention_class=retention_class,
                    owner_writer=str(batch.get("provider_id") or "campaign-provider"),
                    **(
                        {
                            "live_reference_status": (
                                "none" if producer_completed else "active"
                            ),
                            "audit_hold_status": "not_required",
                            "qualification_hold_status": "not_required",
                            "replacement_verified": bool(
                                producer_completed
                                and isinstance(replacement_identity, str)
                                and len(replacement_identity) == 64
                            ),
                            "replacement_artifact_identity": replacement_identity,
                            "release_predicate": (
                                "durable successor checkpoint published and source "
                                "import is no longer live"
                            ),
                        }
                        if retention_class == "R1"
                        else {}
                    ),
                )
            except (ArtifactLifecycleError, OSError):
                complete = False
                continue
            relative = str(row["path"])
            if relative in seen:
                complete = False
                continue
            rows.append(row)
            seen.add(relative)
            covered_trees.append(candidate)

    def covered(path: Path) -> bool:
        if path in covered_files:
            return True
        return any(path == parent or parent in path.parents for parent in covered_trees)

    for path in (
        sorted(root.rglob("*"), key=lambda item: str(item)) if root.exists() else []
    ):
        if path.is_symlink():
            complete = False
            continue
        if not path.is_file() or covered(path):
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        if relative in seen:
            complete = False
            continue
        retention_class, family, reason = _lifecycle_inventory_retention_v1(relative)
        try:
            rows.append(
                build_sealed_artifact_inventory_row_v1(
                    run_root=root,
                    path=path,
                    artifact_family=family,
                    retention_class=retention_class,
                    owner_writer="campaign-closeout",
                    release_predicate=reason,
                    replacement_verified=False,
                )
            )
        except (ArtifactLifecycleError, OSError):
            complete = False
            continue
        seen.add(relative)

    # A producer-declared inventory is complete only when every regular file
    # is covered and no unclassified/link object was observed.
    if terminal_status != "completed":
        complete = False
    return rows, complete


def _revalidate_disposition_source_rows_v1(
    run_root: Path,
    disposition: Mapping[str, Any],
) -> None:
    """Re-open every exact hot source before publishing replacement authority."""

    root = run_root.expanduser().resolve()
    for raw in disposition.get("artifacts", []):
        if not isinstance(raw, Mapping) or raw.get("exact_object_identity") is not True:
            continue
        row = dict(raw)
        relative = str(row.get("path") or "").replace("\\", "/").strip("/")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise CampaignExecutionError(
                "replacement disposition contains an invalid exact source path"
            )
        try:
            observed = build_sealed_artifact_inventory_row_v1(
                run_root=root,
                path=root / relative,
                artifact_family=str(row.get("artifact_family") or ""),
                retention_class=str(row.get("retention_class") or "RX"),
                owner_writer=str(row.get("owner_writer") or "authority-revalidation"),
            )
        except (ArtifactLifecycleError, OSError) as exc:
            raise CampaignExecutionError(
                f"exact disposition source changed before publication: {relative}"
            ) from exc
        # Automatically walked disposition manifests retain the file suffix
        # (for example, ``json``) as a descriptive artifact_type, while
        # producer-sealed inventories use the canonical ``file`` or
        # ``sealed_tree`` values.  Revalidation must compare the physical
        # identity in both forms without mistaking that descriptive suffix
        # for a changed object type.
        identity_fields = {"path", "bytes"}
        if row.get("artifact_type") == "sealed_tree":
            identity_fields.update({"artifact_type", "tree_sha256"})
        else:
            identity_fields.add("sha256")
            if row.get("artifact_type") == "file":
                identity_fields.add("artifact_type")
        if any(observed.get(name) != row.get(name) for name in identity_fields):
            raise CampaignExecutionError(
                f"exact disposition source changed before publication: {relative}"
            )


def reseal_terminal_lifecycle_inventory_v1(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    refreshed_paths: Sequence[str] = (
        "baseline_conformance_receipt.json",
        "execution_state.json",
        "execution_result.json",
        "run_passport.json",
    ),
    additional_artifacts: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Refresh final front-door control files in an already sealed closeout.

    Campaign orchestration seals its producer inventory before the V2 front
    door writes its final conformance, state, and execution controls.  This
    bounded continuation refreshes or adds only those explicitly named files;
    it never performs a second tree walk or reads producer checkpoint
    payloads.  The disposition and storage receipt are rebuilt together so the
    receipt stays bound to the replacement manifest.  The manifest and receipt
    are intentionally excluded from their own inventory, avoiding a circular
    self-hash.
    """

    root = run_root.expanduser().resolve()
    manifest_path = root / "artifact_disposition_manifest.json"
    manifest = validate_artifact_disposition_manifest_v1(
        _read_object(manifest_path, name="artifact disposition manifest")
    )
    if manifest.get("terminal_status") != "completed":
        raise CampaignExecutionError(
            "terminal lifecycle reseal requires a completed disposition"
        )
    inventory_meta = manifest.get("inventory")
    if (
        not isinstance(inventory_meta, Mapping)
        or inventory_meta.get("complete") is not True
    ):
        raise CampaignExecutionError(
            "terminal lifecycle reseal requires a complete producer inventory"
        )

    normalized_targets: tuple[str, ...] = tuple(
        str(path).replace("\\", "/").strip("/") for path in refreshed_paths
    )
    if not normalized_targets or any(
        not path or Path(path).is_absolute() or ".." in Path(path).parts
        for path in normalized_targets
    ):
        raise CampaignExecutionError("terminal lifecycle reseal path is invalid")
    additional = {
        str(path).replace("\\", "/").strip("/"): dict(binding)
        for path, binding in (additional_artifacts or {}).items()
    }
    expected_additional_fields = {
        "artifact_family",
        "retention_class",
        "owner_writer",
        "release_predicate",
    }
    if any(
        not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or set(binding) != expected_additional_fields
        or any(
            not isinstance(binding[field], str) or not binding[field]
            for field in expected_additional_fields
        )
        for path, binding in additional.items()
    ):
        raise CampaignExecutionError(
            "terminal lifecycle additional artifact binding is invalid"
        )
    target_set = set(normalized_targets) | set(additional)

    def seal_target(relative: str) -> dict[str, Any]:
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise CampaignExecutionError(
                f"terminal lifecycle reseal target is not a regular file: {relative}"
            )
        binding = additional.get(relative)
        if binding is None:
            binding = {
                "artifact_family": "lifecycle.control",
                "retention_class": "R5",
                "owner_writer": "execution-front-door",
                "release_predicate": "terminal front-door authority is retained",
            }
        return build_sealed_artifact_inventory_row_v1(
            run_root=root,
            path=target,
            artifact_family=binding["artifact_family"],
            retention_class=binding["retention_class"],
            owner_writer=binding["owner_writer"],
            release_predicate=binding["release_predicate"],
            replacement_verified=False,
        )

    refreshed: list[dict[str, Any]] = []
    found: set[str] = set()
    for original in manifest.get("artifacts", []):
        if not isinstance(original, Mapping):
            raise CampaignExecutionError("disposition artifact row is malformed")
        row = dict(original)
        relative = str(row.get("path") or "").replace("\\", "/").strip("/")
        if relative in target_set:
            row = seal_target(relative)
            found.add(relative)
        refreshed.append(row)
    for relative in sorted(target_set - found):
        refreshed.append(seal_target(relative))

    retention = manifest.get("retention")
    if not isinstance(retention, Mapping):
        raise CampaignExecutionError("disposition retention metadata is malformed")
    replacement_artifacts = manifest.get("replacement_artifacts", [])
    if not isinstance(replacement_artifacts, list):
        raise CampaignExecutionError("disposition replacement metadata is malformed")
    replacement_verified = any(
        isinstance(row, Mapping) and row.get("verified") is True
        for row in replacement_artifacts
    )
    resealed_manifest = build_artifact_disposition_manifest_v1(
        plan,
        run_root=root,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status=str(retention.get("audit_hold_status") or "active"),
        qualification_hold_status=str(
            retention.get("qualification_hold_status") or "not_required"
        ),
        replacement_verified=replacement_verified,
        replacement_artifacts=replacement_artifacts,
        checkpoint_catalog=(
            manifest.get("checkpoint_catalog", [])
            if isinstance(manifest.get("checkpoint_catalog"), list)
            else []
        ),
        artifact_inventory=refreshed,
        inventory_complete=True,
        # The producer closeout has already admitted this exact complete
        # inventory under its execution-scope policy.  Resealing only replaces
        # bounded terminal control rows, so preserve that accepted cardinality
        # instead of reapplying the builder's unrelated 4096-row default.
        max_objects=max(4096, len(refreshed)),
    )
    with run_artifact_transaction_fence_v1(root):
        _revalidate_disposition_source_rows_v1(root, resealed_manifest)
        atomic_write_json(manifest_path, resealed_manifest)

        produced_bytes = int(resealed_manifest["summary"]["produced_bytes"])
        receipt = build_storage_disposition_receipt_v1(
            manifest=resealed_manifest,
            bytes_before=produced_bytes,
            bytes_retained=produced_bytes,
            bytes_released=0,
            bytes_unresolved_rx=int(
                resealed_manifest["summary"]["unresolved_rx_bytes"]
            ),
            authorization_reason=(
                "terminal_metadata_reseal_only_no_destructive_cleanup_authority"
            ),
            replacement_references=replacement_artifacts,
            retention_references=[str(manifest_path)],
            plan_sha256=str(plan.get("plan_sha256")),
            timestamp_utc=_now(),
            result="dry_run",
        )
        receipt_path = root / "storage_disposition_receipt.json"
        write_storage_disposition_receipt_v1(receipt_path, receipt)
        pointer = register_pending_durable_archive_v1(
            manifest_path,
            _transaction_fence_held=True,
        )
    return {
        "manifest": resealed_manifest,
        "storage_receipt": receipt,
        "manifest_path": str(manifest_path),
        "storage_receipt_path": str(receipt_path),
        "durable_archive_pointer_path": None if pointer is None else str(pointer),
    }


@dataclass(frozen=True, slots=True)
class BatchExecutionContextV1:
    """Provider callback surface for durable mid-batch checkpoint recording."""

    run_root: Path
    batch_root: Path
    manifest_path: Path
    slot_lease: Any | None = None
    execution_route_sha256: str | None = None
    lifecycle_contract: Mapping[str, Any] | None = None
    storage_guard: RunStorageGuard | None = None
    provider_id: str | None = None

    def record_artifact(
        self,
        path: str | Path,
        *,
        artifact_family: str,
        retention_class: str,
        owner_writer: str | None = None,
        **metadata: Any,
    ) -> Mapping[str, Any]:
        """Seal one exact provider output into the running batch inventory."""

        manifest = _read_object(self.manifest_path, name="batch manifest")
        if manifest.get("status") != "running":
            raise CampaignExecutionError(
                "artifact can only be recorded while the batch is running"
            )
        inventory = manifest.get("artifact_inventory")
        if not isinstance(inventory, list):
            inventory = []
        if manifest.get("artifact_inventory_complete") is False:
            raise CampaignExecutionError(
                "artifact inventory is already incomplete for this attempt"
            )

        artifact_path = Path(path).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = self.batch_root / artifact_path
        try:
            row = build_sealed_artifact_inventory_row_v1(
                run_root=self.run_root,
                path=artifact_path,
                artifact_family=artifact_family,
                retention_class=retention_class,
                owner_writer=owner_writer or self.provider_id or "campaign-provider",
                **metadata,
            )
        except (ArtifactLifecycleError, OSError) as exc:
            manifest["artifact_inventory_complete"] = False
            failures = manifest.get("artifact_inventory_failures")
            if not isinstance(failures, list):
                failures = []
            failures.append(
                {
                    "path": os.path.abspath(os.fspath(artifact_path)),
                    "error_type": type(exc).__name__,
                }
            )
            manifest["artifact_inventory_failures"] = failures
            manifest["updated_at"] = _now()
            atomic_write_json(self.manifest_path, manifest)
            raise CampaignExecutionError(
                "provider artifact could not be sealed into the lifecycle inventory"
            ) from exc

        existing = next(
            (
                dict(candidate)
                for candidate in inventory
                if isinstance(candidate, Mapping)
                and candidate.get("path") == row["path"]
            ),
            None,
        )
        if existing is not None:
            if existing == row:
                return existing
            manifest["artifact_inventory_complete"] = False
            failures = manifest.get("artifact_inventory_failures")
            if not isinstance(failures, list):
                failures = []
            failures.append(
                {
                    "path": row["path"],
                    "error_type": "ConflictingArtifactSeal",
                }
            )
            manifest["artifact_inventory_failures"] = failures
            manifest["updated_at"] = _now()
            atomic_write_json(self.manifest_path, manifest)
            raise CampaignExecutionError(
                "provider artifact path has a conflicting lifecycle seal"
            )

        inventory.append(row)
        manifest["artifact_inventory"] = inventory
        manifest["updated_at"] = _now()
        atomic_write_json(self.manifest_path, manifest)
        return row

    def record_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        value = _bind_checkpoint_route(
            _json_object(checkpoint, name="checkpoint"),
            self.execution_route_sha256,
        )
        native_coordinate = None
        if value.get("time_authority") == "absolute_tick_v1":
            native_coordinate = _native_checkpoint_next_absolute_tick_v1(value)
        manifest = _read_object(self.manifest_path, name="batch manifest")
        if manifest.get("status") != "running":
            raise CampaignExecutionError(
                "checkpoint can only be recorded while running"
            )
        manifest["checkpoint"] = value
        if self.lifecycle_contract is not None:
            inventory = manifest.get("artifact_inventory")
            if not isinstance(inventory, list):
                inventory = []
            if manifest.get("artifact_inventory_complete") is False:
                raise CampaignExecutionError(
                    "checkpoint inventory is already incomplete for this attempt"
                )

            def mark_inventory_incomplete(path: str, error_type: str) -> None:
                manifest["artifact_inventory_complete"] = False
                failures = manifest.get("artifact_inventory_failures")
                if not isinstance(failures, list):
                    failures = []
                failures.append({"path": path, "error_type": error_type})
                manifest["artifact_inventory_failures"] = failures

            checkpoint_authority = self.lifecycle_contract["checkpoints"].get(
                "authority", "legacy_generation"
            )
            if checkpoint_authority == "absolute_tick":
                coordinate = (
                    native_coordinate
                    if type(native_coordinate) is int
                    else _native_checkpoint_next_absolute_tick_v1(value)
                )
                role = (
                    checkpoint_role_for_tick(self.lifecycle_contract, coordinate)
                    if type(coordinate) is int
                    else None
                )
                coordinate_metadata = (
                    {"checkpoint_tick": coordinate} if type(coordinate) is int else {}
                )
            else:
                coordinate = value.get("generation")
                role = (
                    checkpoint_role_for_generation(self.lifecycle_contract, coordinate)
                    if type(coordinate) is int
                    else None
                )
                coordinate_metadata = (
                    {"generation": coordinate} if type(coordinate) is int else {}
                )
            bundle_rows = value.get("cell_bundles")
            if not isinstance(bundle_rows, list) or not bundle_rows:
                mark_inventory_incomplete(
                    "<checkpoint.cell_bundles>",
                    "MalformedCheckpointBundleInventory",
                )
            if isinstance(bundle_rows, list):
                inventory_bundle_rows = list(bundle_rows)
                checkpoint_set_manifest_path = value.get("checkpoint_set_manifest_path")
                if (
                    isinstance(checkpoint_set_manifest_path, str)
                    and checkpoint_set_manifest_path
                ):
                    inventory_bundle_rows.append(
                        {
                            "path": checkpoint_set_manifest_path,
                            "artifact_family": "checkpoint.world_set_manifest_v1",
                        }
                    )
                sealed_bundle_paths: set[str] = set()
                inventory_by_path = {
                    str(row.get("path")): row
                    for row in inventory
                    if isinstance(row, Mapping) and isinstance(row.get("path"), str)
                }
                for bundle in inventory_bundle_rows:
                    if not isinstance(bundle, Mapping):
                        mark_inventory_incomplete(
                            "<checkpoint.cell_bundles>",
                            "MalformedCheckpointBundleRow",
                        )
                        continue
                    raw_path = bundle.get("path")
                    if not isinstance(raw_path, str) or not raw_path:
                        mark_inventory_incomplete(
                            "<checkpoint.cell_bundles.path>",
                            "MalformedCheckpointBundlePath",
                        )
                        continue
                    bundle_path = Path(raw_path).expanduser()
                    if not bundle_path.is_absolute():
                        bundle_path = self.batch_root / bundle_path
                    # One physical aggregate is intentionally projected as one
                    # logical bundle row per cell.  Seal the physical object
                    # once; hashing it once per logical row made publication
                    # latency scale with physical B136.
                    bundle_key = os.path.abspath(os.fspath(bundle_path))
                    if bundle_key in sealed_bundle_paths:
                        continue
                    sealed_bundle_paths.add(bundle_key)
                    try:
                        relative_bundle_path = str(
                            bundle_path.resolve(strict=True).relative_to(
                                self.run_root.resolve()
                            )
                        ).replace("\\", "/")
                    except (OSError, ValueError):
                        relative_bundle_path = None
                    existing_inventory_row = (
                        inventory_by_path.get(relative_bundle_path)
                        if relative_bundle_path is not None
                        else None
                    )
                    if existing_inventory_row is not None:
                        try:
                            observed = os.lstat(bundle_path)
                        except OSError as exc:
                            mark_inventory_incomplete(
                                os.path.abspath(os.fspath(bundle_path)),
                                type(exc).__name__,
                            )
                            manifest["updated_at"] = _now()
                            atomic_write_json(self.manifest_path, manifest)
                            raise CampaignExecutionError(
                                "previously sealed checkpoint path is unavailable"
                            ) from exc
                        sealed_stat = existing_inventory_row.get("producer_sealed_stat")
                        artifact_type = existing_inventory_row.get("artifact_type")
                        digest = (
                            existing_inventory_row.get("sha256")
                            if artifact_type == "file"
                            else existing_inventory_row.get("tree_sha256")
                            if artifact_type == "sealed_tree"
                            else None
                        )
                        expected_stat = {
                            "device": int(observed.st_dev),
                            "inode": int(observed.st_ino),
                            "bytes": int(observed.st_size),
                            "mtime_ns": int(observed.st_mtime_ns),
                            "ctime_ns": int(observed.st_ctime_ns),
                        }
                        if (
                            bundle_path.is_symlink()
                            or (bundle_path.is_file() and observed.st_nlink != 1)
                            or sealed_stat != expected_stat
                            or existing_inventory_row.get(
                                "producer_checkpoint_identity_sha256"
                            )
                            != value.get("checkpoint_identity_sha256")
                            or not isinstance(digest, str)
                            or len(digest) != 64
                            or any(
                                character not in "0123456789abcdef"
                                for character in digest
                            )
                        ):
                            mark_inventory_incomplete(
                                os.path.abspath(os.fspath(bundle_path)),
                                "ConflictingCheckpointSeal",
                            )
                            manifest["updated_at"] = _now()
                            atomic_write_json(self.manifest_path, manifest)
                            raise CampaignExecutionError(
                                "previously sealed checkpoint path changed or has "
                                "conflicting checkpoint identity"
                            )
                        continue
                    implicit_owner_transition = (
                        checkpoint_authority == "absolute_tick" and role is None
                    )
                    retention_class = str(
                        (role or {}).get("retention_class")
                        or ("R1" if implicit_owner_transition else "RX")
                    )
                    try:
                        inventory_row = build_sealed_artifact_inventory_row_v1(
                            run_root=self.run_root,
                            path=bundle_path,
                            artifact_family=str(
                                bundle.get("artifact_family") or "checkpoint.bundle_v2"
                            ),
                            retention_class=retention_class,
                            owner_writer=self.provider_id or "campaign-provider",
                            checkpoint_role=(role or {}).get("role")
                            or (
                                "owner_transition"
                                if implicit_owner_transition
                                else None
                            ),
                            checkpoint_identity=(role or {}).get("identity"),
                            producer_checkpoint_identity_sha256=value.get(
                                "checkpoint_identity_sha256"
                            ),
                            checkpoint_run_association={
                                "campaign_id": manifest.get("campaign_id"),
                                "batch_id": manifest.get("batch_id"),
                                **coordinate_metadata,
                            },
                            release_predicate=(
                                "checkpoint superseded and no replay/evidence consumer remains"
                                if (role or {}).get("role") == "intermediate"
                                or implicit_owner_transition
                                else "explicit checkpoint lifecycle role"
                            ),
                            replacement_verified=False,
                        )
                        observed = os.lstat(bundle_path)
                        inventory_row["producer_sealed_stat"] = {
                            "device": int(observed.st_dev),
                            "inode": int(observed.st_ino),
                            "bytes": int(observed.st_size),
                            "mtime_ns": int(observed.st_mtime_ns),
                            "ctime_ns": int(observed.st_ctime_ns),
                        }
                    except (ArtifactLifecycleError, OSError) as exc:
                        # A missing/unstable checkpoint is never converted into
                        # deletion authority; the terminal inventory remains
                        # incomplete and therefore fail-closed.
                        mark_inventory_incomplete(
                            os.path.abspath(os.fspath(bundle_path)),
                            type(exc).__name__,
                        )
                        continue
                    if not any(
                        isinstance(existing, Mapping)
                        and existing.get("path") == inventory_row["path"]
                        for existing in inventory
                    ):
                        inventory.append(inventory_row)
            manifest["artifact_inventory"] = inventory
        manifest["updated_at"] = _now()
        atomic_write_json(self.manifest_path, manifest)
        generation = value.get("generation")
        native_tick = None
        if value.get("time_authority") == "absolute_tick_v1":
            try:
                native_tick = _native_checkpoint_next_absolute_tick_v1(value)
            except CampaignExecutionError:
                # Slot telemetry is observational. The provider/front door owns
                # checkpoint validity and will fail the execution separately.
                native_tick = None
        if self.slot_lease is not None and (
            type(generation) is int or type(native_tick) is int
        ):
            try:
                self.slot_lease.update(
                    **(
                        {"next_absolute_tick": native_tick}
                        if type(native_tick) is int
                        else {"current_generation": generation}
                    ),
                    status="running",
                    stage=(
                        "checkpoint tick "
                        f"{native_tick if type(native_tick) is int else generation * int(self.slot_lease.ticks_per_generation)}"
                    ),
                )
            except Exception:
                # Slot telemetry is operational and cannot fail campaign work.
                pass

    def record_progress(self, absolute_tick: int, *, stage: str = "native dispatch") -> None:
        """Publish one committed native dispatch cursor to operational telemetry.

        This is deliberately outside the lifecycle manifest: the slot monitor
        is a read-only operational projection and must not become a campaign
        failure path.  The provider calls it after each durable native
        dispatch, so a tick-native run is visible between checkpoint
        boundaries as well as at them.
        """

        if self.slot_lease is None or type(absolute_tick) is not int:
            return
        try:
            self.slot_lease.update(
                next_absolute_tick=absolute_tick,
                status="running",
                stage=f"{stage} tick {absolute_tick}",
            )
        except Exception:
            # Slot telemetry is operational and cannot fail campaign work.
            pass


def _campaign_manifest(
    run_root: Path,
    plan: Mapping[str, Any],
    provider: Any,
    batches: list[dict[str, Any]],
    *,
    storage_wait: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    counts = {
        name: 0 for name in ("pending", "running", "interrupted", "failed", "completed")
    }
    for batch in batches:
        manifest = _read_object(
            _batch_path(run_root, batch["batch_id"]), name="batch manifest"
        )
        status = manifest.get("status")
        if status not in counts:
            raise CampaignExecutionError(f"invalid batch status: {status!r}")
        counts[status] += 1
        summaries.append(
            {
                "batch_id": batch["batch_id"],
                "status": status,
                "attempts": manifest.get("attempts"),
                "manifest_path": str(_batch_path(run_root, batch["batch_id"])),
            }
        )
    if counts["completed"] == len(batches):
        status = "completed"
    elif counts["failed"]:
        status = "failed"
    elif counts["interrupted"]:
        status = "interrupted"
    elif counts["running"]:
        status = "running"
    else:
        status = "pending"
    value = {
        "schema_version": "eiel_campaign_execution_manifest_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "status": status,
        "counts": counts,
        "batches": summaries,
        "updated_at": _now(),
    }
    if storage_wait is not None:
        value["status"] = "interrupted"
        value["storage_wait"] = dict(storage_wait)
    atomic_write_json(run_root / "campaign_manifest.json", value)
    return value


def _materialize_storage_wait_v1(
    *,
    plan: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    run_root: Path,
    provider: Any,
    batches: list[dict[str, Any]],
    execution_route_sha256: str | None,
    exc: StorageBudgetExceeded,
) -> dict[str, Any]:
    """Publish an interrupted, resumable run instead of quarantining it.

    Storage pressure is a normal resource wait at a safe batch boundary.  The
    provider is not called again until a later execution request resumes the
    same batch manifests and reservation identity.
    """

    wait_core = dict(exc.receipt)
    wait_core.update(
        {
            "status": "WAIT_STORAGE",
            "resume_safe_boundary": "between_physical_batches",
            "action": "release_or_archive_pending_lifecycle_material_then_resume",
            "updated_at_utc": _now(),
        }
    )
    wait_receipt = {
        **wait_core,
        "receipt_sha256": _canonical_sha256(wait_core),
    }
    wait_path = run_root / "storage_wait_receipt.json"
    atomic_write_json(wait_path, wait_receipt)
    campaign = _campaign_manifest(
        run_root,
        plan,
        provider,
        batches,
        storage_wait=wait_receipt,
    )
    _write_run_contract_artifacts_v1(
        run_root,
        plan,
        batches,
        campaign_status="interrupted",
    )
    manifests = [
        _read_object(_batch_path(run_root, batch["batch_id"]), name="batch manifest")
        for batch in batches
    ]
    inventory, inventory_complete = _collect_lifecycle_inventory_v1(
        run_root=run_root,
        batch_manifests=manifests,
        lifecycle=lifecycle,
        terminal_status="interrupted",
    )
    checkpoint_catalog = _lifecycle_checkpoint_catalog(
        run_root,
        manifests,
        lifecycle,
    )
    disposition = build_artifact_disposition_manifest_v1(
        plan,
        run_root=run_root,
        terminal_status="interrupted",
        live_reference_status="active",
        audit_hold_status=(
            "active"
            if lifecycle["retention"]["audit_hold_binding"] == "required"
            else "not_required"
        ),
        qualification_hold_status=(
            "active"
            if lifecycle["retention"]["qualification_hold_binding"] == "required"
            else "not_required"
        ),
        checkpoint_catalog=checkpoint_catalog,
        artifact_inventory=inventory,
        inventory_complete=inventory_complete,
    )
    disposition_path = run_root / "artifact_disposition_manifest.json"
    atomic_write_json(disposition_path, disposition)
    produced_bytes = int(disposition["summary"]["produced_bytes"])
    disposition_receipt = build_storage_disposition_receipt_v1(
        manifest=disposition,
        bytes_before=produced_bytes,
        bytes_retained=produced_bytes,
        bytes_released=0,
        bytes_unresolved_rx=int(disposition["summary"]["unresolved_rx_bytes"]),
        authorization_reason="storage pressure wait; no destructive action authorized",
        retention_references=[str(disposition_path), str(wait_path)],
        plan_sha256=str(plan["plan_sha256"]),
        timestamp_utc=_now(),
        result="storage_wait",
    )
    disposition_receipt_path = run_root / "storage_disposition_receipt.json"
    write_storage_disposition_receipt_v1(
        disposition_receipt_path,
        disposition_receipt,
    )
    passport = build_run_passport_v1(
        plan,
        run_root=run_root,
        terminal_status="interrupted",
        checkpoint_catalog=checkpoint_catalog,
        provider_identity={
            "provider_id": getattr(provider, "provider_id", None),
            "profile_id": plan["resolved_profile"],
            "execution_route_sha256": execution_route_sha256,
        },
        disposition_manifest_path=str(disposition_path),
        disposition_receipt_path=str(disposition_receipt_path),
    )
    passport_path = run_root / "run_passport.json"
    write_run_passport_v1(passport_path, passport)
    return {
        "ok": False,
        "schema_version": "eiel_campaign_execution_receipt_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": execution_route_sha256,
        "status": "interrupted",
        "storage_wait": True,
        "storage_wait_receipt_path": str(wait_path),
        "counts": campaign["counts"],
        "campaign_manifest_path": str(run_root / "campaign_manifest.json"),
        "artifact_lifecycle_contract_path": str(
            run_root / "artifact_lifecycle_contract.json"
        ),
        "run_passport_path": str(passport_path),
        "artifact_disposition_manifest_path": str(disposition_path),
        "storage_disposition_receipt_path": str(disposition_receipt_path),
    }


def _write_run_contract_artifacts_v1(
    run_root: Path,
    plan: Mapping[str, Any],
    batches: list[dict[str, Any]],
    *,
    campaign_status: str,
) -> None:
    """Materialize diagnostic-only contract sidecars for an opted-in run."""

    contract = plan.get("run_contract")
    if not isinstance(contract, Mapping):
        return
    contract_hash = contract.get("contract_hash")
    if not isinstance(contract_hash, str) or not contract_hash.startswith("sha256:"):
        return
    from .run_contract_v1 import build_activation_report_v1

    atomic_write_json(run_root / "resolved_run_contract.json", contract)
    manifests = [
        _read_object(_batch_path(run_root, batch["batch_id"]), name="batch manifest")
        for batch in batches
    ]
    activation = build_activation_report_v1(
        plan,
        manifests,
        campaign_status=campaign_status,
    )
    if activation is not None:
        atomic_write_json(run_root / "activation_report.json", activation)


def _technical_gate_receipt(
    run_root: Path,
    plan: Mapping[str, Any],
    provider: Any,
    batches: list[dict[str, Any]],
) -> dict[str, Any] | None:
    execution = plan.get("execution")
    gate = execution.get("technical_gate") if isinstance(execution, Mapping) else None
    if not isinstance(gate, Mapping) or gate.get("required") is not True:
        return None
    if gate.get("execution_mode") == "pre_primary_canary":
        return _validated_pre_primary_gate_receipt(run_root, plan, provider)
    gate_batch_ids = gate.get("batch_ids")
    gate_cell_ids = gate.get("cell_ids")
    if (
        not isinstance(gate_batch_ids, list)
        or not gate_batch_ids
        or not isinstance(gate_cell_ids, list)
        or not gate_cell_ids
    ):
        raise CampaignExecutionError("compiled technical gate identity is invalid")
    manifests: list[dict[str, Any]] = []
    for batch_id in gate_batch_ids:
        manifest = _read_object(
            _batch_path(run_root, str(batch_id)), name="technical gate batch manifest"
        )
        if manifest.get("status") != "completed":
            return None
        manifests.append(manifest)
    validator = getattr(provider, "evaluate_technical_gate", None)
    if not callable(validator):
        raise CampaignExecutionError(
            f"provider {provider.provider_id!r} lacks technical-gate evaluation"
        )
    raw = validator(
        plan=plan,
        gate_cell_ids=tuple(str(value) for value in gate_cell_ids),
        batch_manifests=tuple(manifests),
        run_root=run_root,
    )
    if not isinstance(raw, Mapping) or type(raw.get("passed")) is not bool:
        raise CampaignExecutionError(
            "provider technical-gate receipt must contain boolean passed"
        )
    receipt = {
        "schema_version": "eiel_campaign_technical_gate_receipt_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "gate_cell_ids": list(gate_cell_ids),
        "gate_batch_ids": list(gate_batch_ids),
        **dict(raw),
    }
    atomic_write_json(run_root / "technical_gate_receipt.json", receipt)
    if receipt["passed"] is not True:
        raise CampaignExecutionError("provider technical gate did not pass")
    return receipt


def _technical_gate_manifest_path(run_root: Path, role: str) -> Path:
    names = {
        "canary_execute": "canary_manifest.json",
        "fresh_owner_restore": "fresh_owner_restore_manifest.json",
    }
    if role not in names:
        raise CampaignExecutionError(f"invalid technical gate role: {role!r}")
    return run_root / "technical_gate" / names[role]


def _technical_gate_contract(plan: Mapping[str, Any]) -> dict[str, Any] | None:
    execution = plan.get("execution")
    gate = execution.get("technical_gate") if isinstance(execution, Mapping) else None
    if not isinstance(gate, Mapping) or gate.get("required") is not True:
        return None
    if gate.get("execution_mode") != "pre_primary_canary":
        return None
    canary = gate.get("canary")
    if not isinstance(canary, Mapping):
        raise CampaignExecutionError("compiled pre-primary technical gate lacks canary")
    cell_ids = canary.get("cell_ids")
    if execution.get("execution_route_identity") == "tick_native_execution_route_v1":
        total_ticks = canary.get("total_ticks")
        checkpoint_ticks = canary.get("checkpoint_ticks")
        if (
            canary.get("time_authority") != "absolute_tick_v1"
            or type(total_ticks) is not int
            or total_ticks < 1
            or not isinstance(checkpoint_ticks, list)
            or any(
                type(value) is not int or value < 1 or value > total_ticks
                for value in checkpoint_ticks
            )
            or checkpoint_ticks != sorted(set(checkpoint_ticks))
        ):
            raise CampaignExecutionError(
                "compiled pre-primary technical gate has an invalid absolute-tick "
                "or checkpoint contract"
            )
        normalized_checkpoint_ticks = sorted({*checkpoint_ticks, total_ticks})
        if (
            not isinstance(cell_ids, list)
            or not cell_ids
            or len(set(cell_ids)) != len(cell_ids)
            or cell_ids != gate.get("cell_ids")
            or not all(isinstance(value, str) and value for value in cell_ids)
        ):
            raise CampaignExecutionError(
                "compiled pre-primary technical gate has invalid cell identities"
            )
        for name in ("batch_id", "restore_batch_id"):
            _safe_id(canary.get(name), name=f"technical gate {name}")
        profile = gate.get("profile")
        if profile is not None and (
            not isinstance(profile, str) or not profile or profile.strip() != profile
        ):
            raise CampaignExecutionError(
                "compiled pre-primary technical gate profile is invalid"
            )
        return {
            "execution_mode": "pre_primary_canary",
            "cell_ids": list(cell_ids),
            "batch_id": str(canary["batch_id"]),
            "restore_batch_id": str(canary["restore_batch_id"]),
            "time_authority": "absolute_tick_v1",
            "total_ticks": total_ticks,
            "checkpoint_ticks": normalized_checkpoint_ticks,
            **({"profile": profile} if profile is not None else {}),
        }
    generations = canary.get("generations")
    milestones = canary.get("checkpoint_milestones")
    if (
        not isinstance(cell_ids, list)
        or not cell_ids
        or len(set(cell_ids)) != len(cell_ids)
        or cell_ids != gate.get("cell_ids")
        or not all(isinstance(value, str) for value in cell_ids)
        or type(generations) is not int
        or generations <= 0
        or not isinstance(milestones, list)
        or not milestones
        or any(
            type(value) is not int or not 0 <= value <= generations
            for value in milestones
        )
        or milestones != sorted(set(milestones))
        or milestones[-1] != generations
    ):
        raise CampaignExecutionError(
            "compiled pre-primary technical gate has an invalid generation "
            "or checkpoint contract"
        )
    for name in ("batch_id", "restore_batch_id"):
        _safe_id(canary.get(name), name=f"technical gate {name}")
    profile = gate.get("profile")
    if profile is not None and (
        not isinstance(profile, str) or not profile or profile.strip() != profile
    ):
        raise CampaignExecutionError(
            "compiled pre-primary technical gate profile is invalid"
        )
    return {
        "execution_mode": "pre_primary_canary",
        "cell_ids": list(cell_ids),
        "batch_id": str(canary["batch_id"]),
        "restore_batch_id": str(canary["restore_batch_id"]),
        "generations": generations,
        "checkpoint_milestones": list(milestones),
        **({"profile": profile} if profile is not None else {}),
    }


def _technical_gate_binding(
    plan: Mapping[str, Any], provider: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    value = {
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "contract": dict(contract),
    }
    return {**value, "gate_contract_sha256": _canonical_sha256(value)}


def _new_technical_gate_manifest(
    *,
    binding: Mapping[str, Any],
    role: str,
    batch_id: str,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    now = _now()
    return {
        "schema_version": "eiel_campaign_technical_gate_manifest_v1",
        **dict(binding),
        "gate_role": role,
        "batch_id": batch_id,
        "cell_ids": list(cell_ids),
        "status": "pending",
        "attempts": 0,
        "checkpoint": None,
        "metrics_rows": [],
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }


def _load_technical_gate_manifest(
    path: Path,
    *,
    binding: Mapping[str, Any],
    role: str,
    batch_id: str,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    expected = _new_technical_gate_manifest(
        binding=binding,
        role=role,
        batch_id=batch_id,
        cell_ids=cell_ids,
    )
    if not path.exists():
        atomic_write_json(path, expected)
        return expected
    manifest = _read_object(path, name=f"technical gate {role} manifest")
    for name in (
        "schema_version",
        "campaign_id",
        "plan_sha256",
        "profile_id",
        "provider_id",
        "execution_route_sha256",
        "gate_contract_sha256",
        "gate_role",
        "batch_id",
        "cell_ids",
    ):
        if manifest.get(name) != expected[name]:
            raise CampaignExecutionError(
                f"technical gate manifest binding mismatch: {role}/{name}"
            )
    return manifest


def _subset_request(
    request: ResidentBatchRequestV1,
    *,
    batch_id: str,
    cell_ids: Sequence[str],
    generations: int | None,
    checkpoint_milestones: Sequence[int],
    total_ticks: int | None = None,
    checkpoint_ticks: Sequence[int] | None = None,
) -> ResidentBatchRequestV1:
    if request.total_ticks is not None:
        native_total_ticks = request.total_ticks if total_ticks is None else total_ticks
        if (
            type(native_total_ticks) is not int
            or native_total_ticks < 1
            or native_total_ticks > request.total_ticks
        ):
            raise CampaignExecutionError(
                "tick-native technical-gate total_ticks is outside the source request"
            )
        native_checkpoints = tuple(
            sorted(
                {
                    *(
                        request.checkpoint_ticks
                        if checkpoint_ticks is None
                        else checkpoint_ticks
                    ),
                    native_total_ticks,
                }
            )
        )
        if any(
            type(value) is not int or value < 1 or value > native_total_ticks
            for value in native_checkpoints
        ):
            raise CampaignExecutionError(
                "tick-native technical-gate checkpoint_ticks are invalid"
            )
        control = dict(request.tick_native_control)
        control["total_ticks"] = native_total_ticks
        control["checkpoint_ticks"] = list(native_checkpoints)
        return replace(
            request,
            batch_id=batch_id,
            cell_ids=tuple(cell_ids),
            checkpoint_milestones=(),
            reporting_enabled=True,
            total_ticks=native_total_ticks,
            checkpoint_ticks=native_checkpoints,
            deliberate_restore_tick=None,
            deliberate_restore_ticks=(),
            tick_native_control=control,
        )
    if generations is None:
        raise CampaignExecutionError("legacy technical-gate generations are absent")
    try:
        return derive_resident_batch_request_v1(
            request,
            ExecutionRequestDerivationV1(
                batch_id=batch_id,
                cell_ids=tuple(cell_ids),
                generations=generations,
                checkpoint_milestones=tuple(checkpoint_milestones),
                reporting_enabled=True,
            ),
        )
    except ExecutionRequestError as exc:
        raise CampaignExecutionError(str(exc)) from exc


def _persist_gate_provider_result(
    path: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    current = _read_object(path, name="technical gate manifest")
    checkpoint = result.get("checkpoint")
    if checkpoint is not None:
        current["checkpoint"] = dict(checkpoint)
    current["metrics_rows"] = [dict(row) for row in result.get("metrics_rows", [])]
    provider_result = result.get("result")
    current["result"] = None if provider_result is None else dict(provider_result)
    current["status"] = result["status"]
    current["error"] = None
    current["updated_at"] = _now()
    atomic_write_json(path, current)
    return current


def _technical_gate_restore_checkpoint(
    canary: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    execution_route_sha256: str | None,
) -> dict[str, Any]:
    if contract.get("time_authority") == "absolute_tick_v1":
        terminal_tick = int(contract["total_ticks"])
        checkpoints = tuple(int(value) for value in contract["checkpoint_ticks"])
        intermediate = [tick for tick in checkpoints if 0 < tick < terminal_tick]
        if not intermediate:
            checkpoint = canary.get("checkpoint")
            if (
                not isinstance(checkpoint, Mapping)
                or _native_checkpoint_next_absolute_tick_v1(checkpoint) != terminal_tick
            ):
                raise CampaignExecutionError(
                    "technical gate canary lacks its exact terminal tick checkpoint"
                )
            return _bind_checkpoint_route(
                checkpoint,
                execution_route_sha256,
                require_existing=execution_route_sha256 is not None,
            )
        restore_tick = intermediate[-1]
        result = canary.get("result")
        catalog = (
            result.get("checkpoint_catalog") if isinstance(result, Mapping) else None
        )
        checkpoint = (
            catalog.get(str(restore_tick)) if isinstance(catalog, Mapping) else None
        )
        if (
            not isinstance(checkpoint, Mapping)
            or _native_checkpoint_next_absolute_tick_v1(checkpoint) != restore_tick
        ):
            raise CampaignExecutionError(
                "technical gate canary lacks its declared intermediate tick checkpoint"
            )
        return _bind_checkpoint_route(
            checkpoint,
            execution_route_sha256,
            require_existing=execution_route_sha256 is not None,
        )
    terminal_generation = int(contract["generations"])
    milestones = tuple(int(value) for value in contract["checkpoint_milestones"])
    intermediate = [
        generation for generation in milestones if 0 < generation < terminal_generation
    ]
    if not intermediate:
        checkpoint = canary.get("checkpoint")
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("generation") != terminal_generation
        ):
            raise CampaignExecutionError(
                "technical gate canary lacks its exact "
                f"G{terminal_generation} checkpoint"
            )
        return _bind_checkpoint_route(
            checkpoint,
            execution_route_sha256,
            require_existing=execution_route_sha256 is not None,
        )

    restore_generation = intermediate[-1]
    result = canary.get("result")
    catalog = result.get("checkpoint_catalog") if isinstance(result, Mapping) else None
    checkpoint = (
        catalog.get(str(restore_generation)) if isinstance(catalog, Mapping) else None
    )
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("generation") != restore_generation
    ):
        raise CampaignExecutionError(
            "technical gate canary lacks its declared intermediate "
            f"G{restore_generation} checkpoint"
        )
    return _bind_checkpoint_route(
        checkpoint,
        execution_route_sha256,
        require_existing=execution_route_sha256 is not None,
    )


def _validated_pre_primary_gate_receipt(
    run_root: Path, plan: Mapping[str, Any], provider: Any
) -> dict[str, Any]:
    contract = _technical_gate_contract(plan)
    if contract is None:
        raise CampaignExecutionError("campaign has no pre-primary technical gate")
    binding = _technical_gate_binding(plan, provider, contract)
    path = run_root / "technical_gate_receipt.json"
    if not path.is_file():
        raise CampaignExecutionError("pre-primary technical gate receipt is absent")
    receipt = _read_object(path, name="pre-primary technical gate receipt")
    recorded_hash = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    expected = {
        "schema_version": "eiel_campaign_technical_gate_receipt_v2",
        **binding,
        "passed": True,
    }
    if (
        any(receipt.get(name) != value for name, value in expected.items())
        or not isinstance(recorded_hash, str)
        or recorded_hash != _canonical_sha256(unhashed)
    ):
        raise CampaignExecutionError(
            "pre-primary technical gate receipt is stale or binding-invalid"
        )
    return receipt


def _execute_pre_primary_technical_gate(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    provider: Any,
    execute: Any,
    resume: Any,
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    contract = _technical_gate_contract(plan)
    if contract is None:
        return None
    receipt_path = run_root / "technical_gate_receipt.json"
    if receipt_path.exists():
        return _validated_pre_primary_gate_receipt(run_root, plan, provider)
    if not callable(resume):
        raise CampaignExecutionError(
            "pre-primary technical gate requires fresh-owner checkpoint resume"
        )
    containing = [
        batch
        for batch in batches
        if set(contract["cell_ids"]) <= set(batch["cell_ids"])
    ]
    if len(containing) != 1:
        raise CampaignExecutionError(
            "pre-primary technical gate cells must share exactly one primary batch"
        )
    source_batch = containing[0]
    source_bound_request = build_execution_request_v1(
        provider, plan, source_batch["batch_id"]
    )
    source_request = source_bound_request.resident_request
    route_sha256 = _execution_route_sha256(provider, plan)
    if source_request.execution_route_sha256 != route_sha256:
        raise CampaignExecutionError(
            "technical gate source request does not bind the admitted route"
        )
    binding = _technical_gate_binding(plan, provider, contract)
    canary_request = _subset_request(
        source_request,
        batch_id=contract["batch_id"],
        cell_ids=contract["cell_ids"],
        generations=contract.get("generations"),
        checkpoint_milestones=contract.get("checkpoint_milestones", ()),
        total_ticks=contract.get("total_ticks"),
        checkpoint_ticks=contract.get("checkpoint_ticks"),
    )
    canary_path = _technical_gate_manifest_path(run_root, "canary_execute")
    canary = _load_technical_gate_manifest(
        canary_path,
        binding=binding,
        role="canary_execute",
        batch_id=contract["batch_id"],
        cell_ids=contract["cell_ids"],
    )
    if canary.get("status") != "completed":
        checkpoint = canary.get("checkpoint")
        if canary.get("status") in {"running", "interrupted"} and not isinstance(
            checkpoint, Mapping
        ):
            raise CampaignExecutionError(
                "technical gate canary is incomplete without a durable checkpoint"
            )
        canary["status"] = "running"
        canary["attempts"] = int(canary.get("attempts", 0)) + 1
        canary["error"] = None
        canary["updated_at"] = _now()
        atomic_write_json(canary_path, canary)
        context = BatchExecutionContextV1(
            run_root=run_root,
            batch_root=canary_path.parent / "canary",
            manifest_path=canary_path,
            execution_route_sha256=route_sha256,
        )
        try:
            raw = (
                resume(plan, canary_request, dict(checkpoint), context)
                if isinstance(checkpoint, Mapping)
                else execute(plan, canary_request, context)
            )
        except Exception as exc:
            canary["status"] = "failed"
            canary["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            canary["updated_at"] = _now()
            atomic_write_json(canary_path, canary)
            raise
        canary = _persist_gate_provider_result(
            canary_path,
            _provider_result(raw, execution_route_sha256=route_sha256),
        )
        if canary["status"] != "completed":
            terminal_label = (
                f"tick {contract['total_ticks']}"
                if contract.get("time_authority") == "absolute_tick_v1"
                else f"G{contract['generations']}"
            )
            raise CampaignExecutionError(
                f"technical gate canary did not reach {terminal_label}"
            )
    terminal_checkpoint = canary.get("checkpoint")
    if contract.get("time_authority") == "absolute_tick_v1":
        terminal_coordinate_valid = bool(
            isinstance(terminal_checkpoint, Mapping)
            and _native_checkpoint_next_absolute_tick_v1(terminal_checkpoint)
            == contract["total_ticks"]
        )
    else:
        terminal_coordinate_valid = bool(
            isinstance(terminal_checkpoint, Mapping)
            and terminal_checkpoint.get("generation") == contract["generations"]
        )
    if not terminal_coordinate_valid:
        raise CampaignExecutionError(
            "technical gate canary lacks its exact terminal checkpoint"
        )
    checkpoint = _technical_gate_restore_checkpoint(
        canary,
        contract,
        execution_route_sha256=route_sha256,
    )

    restore_request = _subset_request(
        source_request,
        batch_id=contract["restore_batch_id"],
        cell_ids=contract["cell_ids"],
        generations=contract.get("generations"),
        checkpoint_milestones=contract.get("checkpoint_milestones", ()),
        total_ticks=contract.get("total_ticks"),
        checkpoint_ticks=contract.get("checkpoint_ticks"),
    )
    checkpoint_rebinder = getattr(
        provider, "prepare_fresh_owner_restore_checkpoint", None
    )
    if callable(checkpoint_rebinder):
        checkpoint = dict(
            checkpoint_rebinder(
                plan,
                canary_request,
                restore_request,
                checkpoint,
            )
        )
    restore_path = _technical_gate_manifest_path(run_root, "fresh_owner_restore")
    restored = _load_technical_gate_manifest(
        restore_path,
        binding=binding,
        role="fresh_owner_restore",
        batch_id=contract["restore_batch_id"],
        cell_ids=contract["cell_ids"],
    )
    if restored.get("status") != "completed":
        restored["status"] = "running"
        restored["attempts"] = int(restored.get("attempts", 0)) + 1
        restored["checkpoint"] = dict(checkpoint)
        restored["error"] = None
        restored["updated_at"] = _now()
        atomic_write_json(restore_path, restored)
        restore_context = BatchExecutionContextV1(
            run_root=run_root,
            batch_root=restore_path.parent / "fresh_owner_restore",
            manifest_path=restore_path,
            execution_route_sha256=route_sha256,
        )
        restored = _persist_gate_provider_result(
            restore_path,
            _provider_result(
                resume(plan, restore_request, dict(checkpoint), restore_context),
                execution_route_sha256=route_sha256,
            ),
        )
    if restored["status"] != "completed":
        terminal_label = (
            f"tick {contract['total_ticks']}"
            if contract.get("time_authority") == "absolute_tick_v1"
            else f"G{contract['generations']}"
        )
        raise CampaignExecutionError(
            f"technical gate fresh-owner restore did not reach {terminal_label}"
        )
    restored_checkpoint = restored.get("checkpoint")
    if contract.get("time_authority") == "absolute_tick_v1":
        restored_coordinate_valid = bool(
            isinstance(restored_checkpoint, Mapping)
            and _native_checkpoint_next_absolute_tick_v1(restored_checkpoint)
            == contract["total_ticks"]
        )
    else:
        restored_coordinate_valid = bool(
            isinstance(restored_checkpoint, Mapping)
            and restored_checkpoint.get("generation") == contract["generations"]
        )
    if not restored_coordinate_valid:
        raise CampaignExecutionError(
            "technical gate fresh-owner restore lacks its exact terminal checkpoint"
        )

    validator = getattr(provider, "evaluate_technical_gate", None)
    if not callable(validator):
        raise CampaignExecutionError(
            f"provider {provider.provider_id!r} lacks technical-gate evaluation"
        )
    evidence = validator(
        plan=plan,
        gate_cell_ids=tuple(contract["cell_ids"]),
        batch_manifests=(canary, restored),
        run_root=run_root,
    )
    if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
        raise CampaignExecutionError("provider pre-primary technical gate did not pass")
    receipt = {
        "schema_version": "eiel_campaign_technical_gate_receipt_v2",
        **binding,
        "passed": True,
        "canary_manifest_path": str(canary_path),
        "fresh_owner_restore_manifest_path": str(restore_path),
        "evidence": dict(evidence),
        "completed_at": _now(),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def _public_metric_rows(
    rows: Any, *, cell_ids: Sequence[str], path: str
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise CampaignExecutionError(f"{path} must be an array")
    expected = set(cell_ids)
    observed: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != {"cell_id", "metrics"}:
            raise CampaignExecutionError(
                f"{path}[{index}] must contain exactly cell_id and metrics"
            )
        cell_id = raw.get("cell_id")
        metrics = raw.get("metrics")
        if (
            not isinstance(cell_id, str)
            or cell_id not in expected
            or cell_id in observed
        ):
            raise CampaignExecutionError(
                f"{path}[{index}] has an invalid cell identity"
            )
        if not isinstance(metrics, Mapping) or not all(
            isinstance(name, str) for name in metrics
        ):
            raise CampaignExecutionError(f"{path}[{index}].metrics must be an object")
        for name, value in metrics.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (isinstance(value, float) and not math.isfinite(value))
            ):
                raise CampaignExecutionError(
                    f"{path}[{index}].metrics.{name} must be numeric or null"
                )
        observed.add(cell_id)
        result.append({"cell_id": cell_id, "metrics": dict(metrics)})
    if observed != expected:
        raise CampaignExecutionError(
            f"{path} does not cover every batch cell exactly once"
        )
    return result


def _write_metrics(
    run_root: Path,
    plan: Mapping[str, Any],
    batches: list[dict[str, Any]],
    provider: Any,
) -> Path:
    rows: list[Any] = []
    for batch in batches:
        manifest = _read_object(
            _batch_path(run_root, batch["batch_id"]), name="batch manifest"
        )
        if manifest.get("status") != "completed":
            continue
        batch_rows = manifest.get("metrics_rows", [])
        cell_ids = batch.get("cell_ids")
        if not isinstance(cell_ids, list) or not all(
            isinstance(cell_id, str) for cell_id in cell_ids
        ):
            raise CampaignExecutionError("compiled batch cell_ids must be an array")
        projector = getattr(provider, "campaign_metric_rows", None)
        if callable(projector):
            if not isinstance(batch_rows, list) or not all(
                isinstance(row, Mapping) for row in batch_rows
            ):
                raise CampaignExecutionError("batch metrics_rows must be an array")
            batch_rows = projector(batch_rows, cell_ids=cell_ids)
        rows.extend(
            _public_metric_rows(
                batch_rows,
                cell_ids=cell_ids,
                path=f"batch {batch['batch_id']} campaign metrics",
            )
        )
    path = run_root / "cell_metrics.json"
    atomic_write_json(
        path,
        {
            "schema_version": "eiel_campaign_cell_metrics_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            "rows": rows,
        },
    )
    return path


def _write_generation_metrics(
    run_root: Path,
    plan: Mapping[str, Any],
    batches: list[dict[str, Any]],
) -> Path | None:
    """Persist an optional provider-owned generation series without flattening it.

    Providers keep ownership of the row schema.  The harness owns only durable
    identity, uniqueness, ordering, and plan binding so future campaign
    profiles can expose trajectories without changing the terminal metric
    contract.
    """

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    known_cells = {
        str(cell_id)
        for batch in batches
        for cell_id in batch.get("cell_ids", [])
        if isinstance(cell_id, str)
    }
    schema_versions: set[str] = set()
    for batch in batches:
        manifest = _read_object(
            _batch_path(run_root, batch["batch_id"]), name="batch manifest"
        )
        result = manifest.get("result")
        if not isinstance(result, Mapping):
            continue
        supplied = result.get("generation_metrics_rows")
        if supplied is None:
            continue
        if not isinstance(supplied, list):
            raise CampaignExecutionError(
                "provider generation_metrics_rows must be an array"
            )
        for index, raw in enumerate(supplied):
            if not isinstance(raw, Mapping):
                raise CampaignExecutionError(
                    f"provider generation_metrics_rows[{index}] must be an object"
                )
            row = dict(raw)
            cell_id = row.get("cell_id")
            generation = row.get("generation")
            schema_version = row.get("schema_version")
            if (
                not isinstance(cell_id, str)
                or cell_id not in known_cells
                or type(generation) is not int
                or generation < 0
                or not isinstance(schema_version, str)
                or not schema_version
            ):
                raise CampaignExecutionError(
                    "provider generation metric row has invalid identity"
                )
            identity = (cell_id, generation)
            if identity in seen:
                raise CampaignExecutionError(
                    f"duplicate generation metric row: {cell_id}/G{generation}"
                )
            seen.add(identity)
            schema_versions.add(schema_version)
            rows.append(row)
    if not rows:
        return None
    rows.sort(key=lambda row: (str(row["cell_id"]), int(row["generation"])))
    path = run_root / "generation_metrics.json"
    atomic_write_json(
        path,
        {
            "schema_version": "eiel_campaign_generation_series_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            "provider_row_schema_versions": sorted(schema_versions),
            "row_count": len(rows),
            "rows": rows,
        },
    )
    return path


def _write_cell_receipts(
    run_root: Path,
    plan: Mapping[str, Any],
    batches: list[dict[str, Any]],
    provider: Any,
) -> Path:
    """Materialize generic per-cell provenance from authoritative batch results."""

    plan_cells = plan.get("cells")
    if not isinstance(plan_cells, list):
        raise CampaignExecutionError("compiled plan cells must be an array")
    cells_by_id = {
        row.get("cell_id"): dict(row)
        for row in plan_cells
        if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
    }
    route_sha256 = _execution_route_sha256(provider, plan)
    index_rows: list[dict[str, Any]] = []
    for batch in batches:
        manifest_path = _batch_path(run_root, batch["batch_id"])
        manifest = _read_object(manifest_path, name="batch manifest")
        if manifest.get("status") != "completed":
            continue
        checkpoint = manifest.get("checkpoint")
        native_route = (
            isinstance(plan.get("execution"), Mapping)
            and plan["execution"].get("execution_route_identity")
            == "tick_native_execution_route_v1"
        )
        checkpoint_coordinate = (
            (
                _native_checkpoint_next_absolute_tick_v1(checkpoint)
                if isinstance(checkpoint, Mapping)
                else None
            )
            if native_route
            else (
                checkpoint.get("generation")
                if isinstance(checkpoint, Mapping)
                else None
            )
        )
        for cell_id in batch["cell_ids"]:
            if cell_id not in cells_by_id:
                raise CampaignExecutionError(
                    f"batch references unknown compiled cell: {cell_id}"
                )
            root = run_root / "cells" / _safe_id(cell_id, name="cell id")
            cell = cells_by_id[cell_id]
            cell_manifest = {
                "schema_version": "eiel_campaign_cell_manifest_v1",
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "profile_id": plan["resolved_profile"],
                "provider_id": provider.provider_id,
                "execution_route_sha256": route_sha256,
                "batch_id": batch["batch_id"],
                "batch_manifest_path": str(manifest_path),
                "cell": cell,
            }
            completion = {
                "schema_version": "eiel_campaign_cell_completion_v1",
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan["plan_sha256"],
                "cell_id": cell_id,
                "status": "completed",
                (
                    "terminal_checkpoint_tick"
                    if native_route
                    else "terminal_checkpoint_generation"
                ): checkpoint_coordinate,
                "batch_manifest_path": str(manifest_path),
                "execution_route_sha256": route_sha256,
            }
            atomic_write_json(root / "cell_manifest.json", cell_manifest)
            atomic_write_json(root / "config_snapshot.json", cell)
            atomic_write_json(root / "completion.json", completion)
            index_rows.append(
                {
                    "cell_id": cell_id,
                    "cell_manifest_path": str(root / "cell_manifest.json"),
                    "completion_path": str(root / "completion.json"),
                }
            )
    path = run_root / "cell_receipts.json"
    atomic_write_json(
        path,
        {
            "schema_version": "eiel_campaign_cell_receipt_index_v1",
            "campaign_id": plan["campaign_id"],
            "plan_sha256": plan["plan_sha256"],
            "completed_cell_count": len(index_rows),
            "rows": sorted(index_rows, key=lambda row: row["cell_id"]),
        },
    )
    return path


def _completed_provider_cell_results(
    run_root: Path,
    batches: Sequence[Mapping[str, Any]],
    *,
    expected_cell_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """Collect each authoritative provider result exactly once."""

    expected = set(expected_cell_ids)
    results: dict[str, Mapping[str, Any]] = {}
    for batch in batches:
        batch_id = _safe_id(batch.get("batch_id"), name="batch id")
        manifest = _read_object(
            _batch_path(run_root, batch_id),
            name="completed batch manifest",
        )
        if manifest.get("status") != "completed":
            raise CampaignExecutionError(
                f"evidence closeout requires completed batch {batch_id}"
            )
        payload = manifest.get("result")
        cell_results = (
            payload.get("cell_results") if isinstance(payload, Mapping) else None
        )
        if not isinstance(cell_results, Mapping):
            raise CampaignExecutionError(
                f"completed batch {batch_id} lacks provider cell results"
            )
        for cell_id, result in cell_results.items():
            if (
                not isinstance(cell_id, str)
                or cell_id not in expected
                or cell_id in results
                or not isinstance(result, Mapping)
            ):
                raise CampaignExecutionError(
                    "provider cell results are missing, duplicated, or unexpected"
                )
            results[cell_id] = result
    if set(results) != expected:
        raise CampaignExecutionError(
            "provider cell results do not exactly cover the compiled campaign"
        )
    return results


def _comparison_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    comparisons = plan.get("comparisons", {})
    if not isinstance(comparisons, Mapping):
        raise CampaignExecutionError("compiled plan comparisons must be an object")
    raw = comparisons.get("branches", [])
    if not isinstance(raw, list):
        raise CampaignExecutionError("compiled comparison branches must be an array")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, value in enumerate(raw):
        row = _json_object(value, name="comparison branch")
        branch_id = _safe_id(row.get("branch_id"), name="comparison branch id")
        kind = row.get("kind")
        source_cell_id = row.get("source_cell_id")
        if branch_id in seen or kind not in {
            "fresh_repeat",
            "checkpoint_continuation",
            "reporting_parity",
        }:
            raise CampaignExecutionError(
                "compiled plan has invalid or duplicate comparison branches"
            )
        if not isinstance(source_cell_id, str):
            raise CampaignExecutionError(f"comparison {branch_id!r} has no source cell")
        split = row.get("split_generation")
        if kind == "checkpoint_continuation" and (type(split) is not int or split <= 0):
            raise CampaignExecutionError(
                f"comparison {branch_id!r} has an invalid split"
            )
        if kind in {"fresh_repeat", "reporting_parity"} and split is not None:
            raise CampaignExecutionError(
                f"comparison {branch_id!r} has an unexpected split"
            )
        row["ordinal"] = ordinal
        seen.add(branch_id)
        rows.append(row)
    return rows


def _source_batch(
    batches: list[dict[str, Any]], source_cell_id: str
) -> tuple[dict[str, Any], int]:
    matches = [
        (batch, batch["cell_ids"].index(source_cell_id))
        for batch in batches
        if source_cell_id in batch["cell_ids"]
    ]
    if len(matches) != 1:
        raise CampaignExecutionError(
            f"comparison source cell must belong to exactly one physical batch: {source_cell_id}"
        )
    return matches[0]


def _narrow_request(
    request: Any,
    *,
    branch_id: str,
    source_index: int,
    source_cell_id: str,
    generations: int,
    checkpoints: tuple[int, ...],
    reporting_enabled: bool = True,
) -> ResidentBatchRequestV1:
    if not isinstance(request, ResidentBatchRequestV1):
        raise CampaignExecutionError(
            "comparison execution requires the registered ResidentBatchRequestV1 contract"
        )
    if (
        source_index >= len(request.cell_ids)
        or request.cell_ids[source_index] != source_cell_id
    ):
        raise CampaignExecutionError(
            "comparison source index does not bind the declared source cell"
        )
    try:
        narrowed = derive_resident_batch_request_v1(
            request,
            ExecutionRequestDerivationV1(
                batch_id=branch_id,
                cell_ids=(source_cell_id,),
                generations=generations,
                checkpoint_milestones=checkpoints,
                reporting_enabled=reporting_enabled,
            ),
        )
    except ExecutionRequestError as exc:
        raise CampaignExecutionError(str(exc)) from exc
    narrowed = replace(
        narrowed,
        deliberate_restore_tick=None,
        deliberate_restore_ticks=(),
    )
    if request.total_ticks is None:
        return replace(
            narrowed,
            agent_id_allocation_generation_count=generations,
        )
    interval = request.source_window_ticks
    if type(interval) is not int or interval < 1:
        raise CampaignExecutionError(
            "tick-native comparison request lacks a bounded source window"
        )
    # ``generations`` is only the compatibility horizon used to derive a
    # comparison request.  A tick-native campaign may end part-way through
    # its final generation (for example, 5,000 ticks is 31 full generations
    # plus 40 ticks).  Do not silently round that terminal comparison up to
    # the next generation: it would compare the primary run at T=5,000 with
    # a reporting/repeat run at T=5,120 and fail on terminal_checkpoint_digest.
    total_ticks = min(request.total_ticks, generations * interval)
    checkpoint_ticks = tuple(
        sorted(
            {
                value
                for value in request.checkpoint_ticks
                if type(value) is int and 0 < value <= total_ticks
            }
            | {total_ticks}
        )
    )
    return replace(
        narrowed,
        total_ticks=total_ticks,
        checkpoint_ticks=checkpoint_ticks,
        agent_id_allocation_generation_count=generations,
    )


def _new_comparison_manifest(
    *,
    plan: Mapping[str, Any],
    provider: Any,
    branch: Mapping[str, Any],
    source_batch_id: str,
    source_index: int,
) -> dict[str, Any]:
    now = _now()
    return {
        "schema_version": "eiel_campaign_comparison_manifest_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "branch_id": branch["branch_id"],
        "kind": branch["kind"],
        "source_cell_id": branch["source_cell_id"],
        "source_batch_id": source_batch_id,
        "source_cell_ordinal": source_index,
        "split_generation": branch.get("split_generation"),
        "repeat_ordinal": branch.get("repeat_ordinal"),
        "status": "pending",
        "phase": "pending",
        "attempts": 0,
        "checkpoint": None,
        "metrics_rows": [],
        "result": None,
        "comparison": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
    }


def _load_comparison_manifest(
    path: Path,
    *,
    plan: Mapping[str, Any],
    provider: Any,
    branch: Mapping[str, Any],
    source_batch_id: str,
    source_index: int,
) -> dict[str, Any]:
    expected = _new_comparison_manifest(
        plan=plan,
        provider=provider,
        branch=branch,
        source_batch_id=source_batch_id,
        source_index=source_index,
    )
    if not path.exists():
        atomic_write_json(path, expected)
        return expected
    manifest = _read_object(path, name="comparison manifest")
    for name in (
        "schema_version",
        "campaign_id",
        "plan_sha256",
        "profile_id",
        "provider_id",
        "branch_id",
        "kind",
        "source_cell_id",
        "source_batch_id",
        "source_cell_ordinal",
        "split_generation",
        "repeat_ordinal",
    ):
        if manifest.get(name) != expected[name]:
            raise CampaignExecutionError(
                f"comparison manifest binding mismatch for {branch['branch_id']}: {name}"
            )
    return manifest


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cell_metrics(rows: Any, cell_id: str) -> list[dict[str, Any]] | None:
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise CampaignExecutionError(
            "comparison metrics_rows must be an array of objects"
        )
    selected = [dict(row) for row in rows if row.get("cell_id") == cell_id]
    return selected or None


def _cell_value(container: Any, names: tuple[str, ...], cell_id: str) -> Any:
    if not isinstance(container, Mapping):
        return None
    for name in names:
        value = container.get(name)
        if isinstance(value, Mapping):
            return value.get(cell_id)
        if isinstance(value, str):
            return value
    return None


def _comparison_surface(
    manifest: Mapping[str, Any],
    cell_id: str,
    *,
    provider: Any,
    continuation: bool = False,
) -> dict[str, str]:
    surface: dict[str, str] = {}
    metrics = _cell_metrics(manifest.get("metrics_rows", []), cell_id)
    if metrics is not None:
        projector = getattr(provider, "campaign_metric_rows", None)
        if callable(projector):
            metrics = projector(metrics, cell_ids=(cell_id,))
        public_metrics = _public_metric_rows(
            metrics,
            cell_ids=(cell_id,),
            path=f"comparison {cell_id} campaign metrics",
        )
        surface["metrics_rows_sha256"] = _canonical_sha256(public_metrics)
    # Cross-route comparisons bind trajectory mechanics first.  The complete
    # source-projection digest remains authoritative inside Bundle v2, but it
    # also contains route-local Chronicle/overlay audit history and therefore
    # is intentionally not invariant across a fresh repeat or restore path.
    digest_names = (
        "mechanics_checkpoint_digests",
        "terminal_cell_digests",
        "cell_terminal_digests",
        "cell_digests",
    )
    checkpoint_digest = _cell_value(
        manifest.get("checkpoint"),
        digest_names,
        cell_id,
    )
    result = manifest.get("result")
    if checkpoint_digest is None:
        checkpoint_digest = _cell_value(
            result,
            digest_names,
            cell_id,
        )
    if checkpoint_digest is not None:
        if not isinstance(checkpoint_digest, str):
            raise CampaignExecutionError(
                "provider terminal cell digest must be a string"
            )
        surface["terminal_checkpoint_digest"] = checkpoint_digest
    reporting_digest = _cell_value(
        result,
        ("reporting_bytes_sha256", "cell_reporting_sha256"),
        cell_id,
    )
    if reporting_digest is not None:
        if not isinstance(reporting_digest, str):
            raise CampaignExecutionError("provider reporting digest must be a string")
        surface["reporting_bytes_sha256"] = reporting_digest
    sentinels = (
        _continuation_transition_sentinels(manifest, cell_id)
        if continuation
        else _comparable_transition_sentinels(manifest, cell_id)
    )
    if sentinels is not None:
        surface["transition_sentinels_sha256"] = _canonical_sha256(sentinels)
    return surface


def _transition_sentinels(
    manifest: Mapping[str, Any], cell_id: str
) -> list[dict[str, Any]] | None:
    result = manifest.get("result")
    checkpoint = manifest.get("checkpoint")
    for container in (result, checkpoint):
        if not isinstance(container, Mapping):
            continue
        raw = container.get("transition_sentinels")
        if isinstance(raw, Mapping):
            raw = raw.get(cell_id)
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise CampaignExecutionError(
                "provider transition sentinels must be an array"
            )
        try:
            return [
                RouteTransitionSentinelV1.from_mapping(row).payload() for row in raw
            ]
        except (TypeError, ValueError) as exc:
            raise CampaignExecutionError(
                f"provider transition sentinel is invalid: {exc}"
            ) from exc
    return None


def _comparable_transition_sentinels(
    manifest: Mapping[str, Any], cell_id: str
) -> list[dict[str, Any]] | None:
    """Exclude branch-only intermediate checkpoint markers from parity."""

    rows = _transition_sentinels(manifest, cell_id)
    if rows is None:
        return None
    terminal_generation = max(
        (row["absolute_generation"] for row in rows),
        default=0,
    )
    return [
        row
        for row in rows
        if row.get("transition_name") != "checkpoint_export_boundary"
        or row.get("absolute_generation") == terminal_generation
    ]


def _continuation_transition_sentinels(
    manifest: Mapping[str, Any], cell_id: str
) -> list[dict[str, Any]] | None:
    """Compare scientific observations, excluding split-only lifecycle markers."""

    rows = _comparable_transition_sentinels(manifest, cell_id)
    if rows is None:
        return None
    terminal_generation = max(
        (row["absolute_generation"] for row in rows),
        default=0,
    )
    restore_names = {
        "first_tick_after_restore": "first_tick_after_boundary",
        "first_policy_cache_refill_after_restore": (
            "first_policy_cache_refill_after_boundary"
        ),
        "second_tick_after_restore": "second_tick_after_boundary",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        name = raw.get("transition_name")
        if name in {
            "last_tick_before_generation_boundary",
            "generation_boundary",
        }:
            continue
        if (
            name == "terminal_settlement"
            and raw.get("absolute_generation") != terminal_generation
        ):
            continue
        row = dict(raw)
        row["transition_name"] = restore_names.get(str(name), name)
        row["physical_window_position"] = row["absolute_tick"]
        identity = _canonical_sha256(row)
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(row)
    return normalized


def _component_field_trace(
    manifest: Mapping[str, Any], cell_id: str
) -> list[dict[str, Any]] | None:
    for container in (manifest.get("checkpoint"), manifest.get("result")):
        if not isinstance(container, Mapping):
            continue
        raw = container.get("component_field_trace")
        if isinstance(raw, Mapping):
            raw = raw.get(cell_id)
        if raw is None:
            continue
        if not isinstance(raw, list) or not all(
            isinstance(row, Mapping) for row in raw
        ):
            raise CampaignExecutionError(
                "provider component/field trace must be an array of objects"
            )
        rows = [dict(row) for row in raw]
        for row in rows:
            if (
                type(row.get("generation")) is not int
                or not isinstance(row.get("component"), str)
                or not isinstance(row.get("field"), str)
                or not isinstance(row.get("sha256"), str)
            ):
                raise CampaignExecutionError(
                    "provider component/field trace row is invalid"
                )
        return rows
    return None


def _first_trace_difference(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], cell_id: str
) -> dict[str, Any]:
    left = _component_field_trace(baseline, cell_id)
    right = _component_field_trace(candidate, cell_id)
    if left is None or right is None:
        return {
            "status": "exact_component_field_unavailable_only_terminal_projection_digest"
        }
    maximum = max(len(left), len(right))
    for index in range(maximum):
        before = left[index] if index < len(left) else None
        after = right[index] if index < len(right) else None
        if before != after:
            row = before if before is not None else after
            return {
                "status": "localized",
                "generation": row.get("generation"),
                "component": row.get("component"),
                "field": row.get("field"),
                "baseline": before,
                "candidate": after,
            }
    return {"status": "no_component_field_difference"}


def _first_comparison_difference(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cell_id: str,
    *,
    continuation: bool = False,
) -> dict[str, Any]:
    sentinel_reader = (
        _continuation_transition_sentinels
        if continuation
        else _comparable_transition_sentinels
    )
    left = sentinel_reader(baseline, cell_id)
    right = sentinel_reader(candidate, cell_id)
    if left is not None and right is not None:
        difference = first_route_transition_difference(left, right)
        if difference is None:
            return {"status": "no_transition_sentinel_difference"}
        index = difference["sentinel_index"]
        row = left[index] if index < len(left) else right[index]
        path = str(difference["path"])
        return {
            "status": "localized_transition_sentinel",
            "absolute_generation": row.get("absolute_generation"),
            "absolute_tick": row.get("absolute_tick"),
            "physical_window_position": row.get("physical_window_position"),
            "transition": difference.get("transition_name"),
            "component": (
                path.split(".", 1)[1].split(".", 1)[0]
                if "." in path
                else "transition_sentinels"
            ),
            "field": path.rsplit(".", 1)[-1],
            **difference,
        }
    return _first_trace_difference(baseline, candidate, cell_id)


def _compare_branch(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cell_id: str,
    provider: Any,
    continuation: bool = False,
) -> dict[str, Any]:
    left = _comparison_surface(
        baseline,
        cell_id,
        provider=provider,
        continuation=continuation,
    )
    right = _comparison_surface(
        candidate,
        cell_id,
        provider=provider,
        continuation=continuation,
    )
    common = sorted(set(left) & set(right))
    missing = sorted(set(left) ^ set(right))
    mismatches = [name for name in common if left[name] != right[name]]
    if missing:
        evidence_status = "surface_contract_mismatch"
    elif not common:
        evidence_status = "insufficient_comparison_surface"
    elif mismatches:
        evidence_status = "exact_surface_mismatch"
    elif "terminal_checkpoint_digest" in common:
        evidence_status = "terminal_checkpoint_and_observations_match"
    else:
        evidence_status = "observations_only_match"
    receipt = {
        "schema_version": "eiel_campaign_comparison_receipt_v1",
        "source_cell_id": cell_id,
        "compared_surfaces": common,
        "baseline": left,
        "candidate": right,
        "evidence_status": evidence_status,
        "match": bool(common) and not missing and not mismatches,
        "missing_surfaces": missing,
        "mismatches": mismatches,
        "first_difference": (
            _first_comparison_difference(
                baseline,
                candidate,
                cell_id,
                continuation=continuation,
            )
            if mismatches
            else None
        ),
    }
    return receipt


def _compare_reporting_parity(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cell_id: str,
    provider: Any,
) -> dict[str, Any]:
    left_surface = _comparison_surface(baseline, cell_id, provider=provider)
    right_surface = _comparison_surface(candidate, cell_id, provider=provider)
    left = left_surface.get("terminal_checkpoint_digest")
    right = right_surface.get("terminal_checkpoint_digest")
    terminal_present = isinstance(left, str) and isinstance(right, str)
    sentinel_present = (
        "transition_sentinels_sha256" in left_surface,
        "transition_sentinels_sha256" in right_surface,
    )
    missing = [] if terminal_present else ["terminal_checkpoint_digest"]
    if sentinel_present[0] != sentinel_present[1]:
        missing.append("transition_sentinels")
    mismatches: list[str] = []
    if terminal_present and left != right:
        mismatches.append("terminal_checkpoint_digest")
    if (
        sentinel_present == (True, True)
        and left_surface["transition_sentinels_sha256"]
        != right_surface["transition_sentinels_sha256"]
    ):
        mismatches.append("transition_sentinels_sha256")
    match = terminal_present and not missing and not mismatches
    compared_surfaces = ["terminal_checkpoint_digest"] if terminal_present else []
    if sentinel_present == (True, True):
        compared_surfaces.append("transition_sentinels_sha256")
    return {
        "schema_version": "eiel_campaign_reporting_parity_comparison_v1",
        "source_cell_id": cell_id,
        "evidence_mode": "enabled_disabled_exact_terminal_state",
        "compared_surfaces": compared_surfaces,
        "reporting_enabled_terminal_digest": left,
        "reporting_disabled_terminal_digest": right,
        "state_mutation_sentinel": (left if match else None),
        "match": match,
        "evidence_status": (
            "reporting_parity_proven" if match else "reporting_terminal_digest_mismatch"
        ),
        "first_difference": (
            None
            if match
            else _first_comparison_difference(baseline, candidate, cell_id)
        ),
        "mismatches": mismatches,
        "missing_surfaces": missing,
    }


def _provider_result(
    raw: Any, *, execution_route_sha256: str | None = None
) -> dict[str, Any]:
    result = _json_object(raw, name="provider comparison result")
    if result.get("status") not in {"completed", "interrupted"}:
        raise CampaignExecutionError(
            "provider comparison result status must be completed or interrupted"
        )
    checkpoint = result.get("checkpoint")
    if checkpoint is not None and not isinstance(checkpoint, Mapping):
        raise CampaignExecutionError("provider comparison checkpoint must be an object")
    if isinstance(checkpoint, Mapping):
        result["checkpoint"] = _bind_checkpoint_route(
            checkpoint, execution_route_sha256
        )
    rows = result.get("metrics_rows", [])
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise CampaignExecutionError("provider comparison metrics must be objects")
    provider_result = result.get("result")
    if provider_result is not None and not isinstance(provider_result, Mapping):
        raise CampaignExecutionError(
            "provider comparison result payload must be an object"
        )
    return result


def _persist_comparison_result(
    path: Path, result: Mapping[str, Any], *, phase: str
) -> dict[str, Any]:
    current = _read_object(path, name="comparison manifest")
    checkpoint = result.get("checkpoint")
    if checkpoint is not None:
        current["checkpoint"] = dict(checkpoint)
    current["metrics_rows"] = [dict(row) for row in result.get("metrics_rows", [])]
    provider_result = result.get("result")
    previous_result = current.get("result")
    if isinstance(previous_result, Mapping) and isinstance(provider_result, Mapping):
        previous_sentinels = previous_result.get("transition_sentinels")
        next_sentinels = provider_result.get("transition_sentinels")
        if previous_sentinels is not None and next_sentinels is not None:
            provider_result = dict(provider_result)
            provider_result["transition_sentinels"] = (
                _merge_transition_sentinel_surfaces(previous_sentinels, next_sentinels)
            )
    current["result"] = None if provider_result is None else dict(provider_result)
    current["status"] = result["status"]
    current["phase"] = phase
    current["error"] = None
    current["updated_at"] = _now()
    atomic_write_json(path, current)
    return current


def _merge_transition_sentinel_rows(left: Any, right: Any) -> list[Any]:
    if not isinstance(left, list) or not isinstance(right, list):
        raise CampaignExecutionError(
            "provider transition sentinel continuation surfaces must be arrays"
        )
    rows = [*left]
    if rows and right and rows[-1] == right[0]:
        rows.extend(right[1:])
    else:
        rows.extend(right)
    return rows


def _merge_transition_sentinel_surfaces(left: Any, right: Any) -> Any:
    if isinstance(left, list) or isinstance(right, list):
        return _merge_transition_sentinel_rows(left, right)
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise CampaignExecutionError(
            "provider transition sentinel continuation surfaces are incompatible"
        )
    result: dict[str, Any] = {}
    for cell_id in sorted(set(left) | set(right)):
        if cell_id in left and cell_id in right:
            result[cell_id] = _merge_transition_sentinel_rows(
                left[cell_id], right[cell_id]
            )
        elif cell_id in left:
            result[cell_id] = left[cell_id]
        else:
            result[cell_id] = right[cell_id]
    return result


def _execute_comparison_branch(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    provider: Any,
    execute: Any,
    resume: Any,
    branch: Mapping[str, Any],
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    source_batch, source_index = _source_batch(batches, branch["source_cell_id"])
    baseline_path = _batch_path(run_root, source_batch["batch_id"])
    baseline = _read_object(baseline_path, name="source batch manifest")
    if baseline.get("status") != "completed":
        raise CampaignExecutionError(
            f"comparison source batch is not complete: {source_batch['batch_id']}"
        )
    path = _comparison_path(run_root, branch["branch_id"])
    manifest = _load_comparison_manifest(
        path,
        plan=plan,
        provider=provider,
        branch=branch,
        source_batch_id=source_batch["batch_id"],
        source_index=source_index,
    )
    if manifest.get("status") == "completed":
        return manifest

    if (
        branch["kind"] == "reporting_parity"
        and getattr(provider, "reporting_parity_supported", False) is not True
    ):
        manifest["status"] = "completed"
        manifest["phase"] = "unsupported"
        manifest["comparison"] = {
            "schema_version": "eiel_campaign_reporting_parity_comparison_v1",
            "source_cell_id": branch["source_cell_id"],
            "match": False,
            "evidence_status": "provider_reporting_parity_execution_unsupported",
            "first_difference": None,
            "mismatches": [],
            "missing_surfaces": ["reporting_disabled_terminal_checkpoint_digest"],
        }
        manifest["updated_at"] = _now()
        atomic_write_json(path, manifest)
        return manifest

    checkpoint = manifest.get("checkpoint")
    if manifest.get("status") in {"running", "interrupted", "failed"}:
        if not isinstance(checkpoint, Mapping):
            raise CampaignExecutionError(
                f"comparison {branch['branch_id']} has no durable checkpoint; "
                "refusing duplicate dispatch"
            )
        if not callable(resume):
            raise CampaignExecutionError(
                f"provider {provider.provider_id!r} has no batch resume capability"
            )

    base_bound_request = build_execution_request_v1(
        provider, plan, source_batch["batch_id"]
    )
    base_request = base_bound_request.resident_request
    route_sha256 = _execution_route_sha256(provider, plan)
    if route_sha256 is not None and base_request.execution_route_sha256 != route_sha256:
        raise CampaignExecutionError(
            "comparison request does not bind the admitted execution route"
        )
    if isinstance(checkpoint, Mapping):
        checkpoint = _bind_checkpoint_route(
            checkpoint,
            route_sha256,
            require_existing=True,
        )
    terminal_generation, checkpoints = _tick_native_or_legacy_generation_horizon_v1(
        plan.get("execution")
    )
    terminal_checkpoints = tuple(
        value
        for value in checkpoints
        if type(value) is int and value <= terminal_generation
    )
    adaptive_reporting_checkpoint: Mapping[str, Any] | None = None
    if branch["kind"] == "reporting_parity" and checkpoint is None:
        baseline_result = baseline.get("result")
        fresh = (
            baseline_result.get("fresh_owner_restore_receipt")
            if isinstance(baseline_result, Mapping)
            else None
        )
        adaptive = (
            fresh.get("wrc_checkpoint_adaptive_rebinding")
            if isinstance(fresh, Mapping)
            else None
        )
        if adaptive is not None:
            catalog = baseline_result.get("checkpoint_catalog")
            source_generation = (
                adaptive.get("source_generation")
                if isinstance(adaptive, Mapping)
                else None
            )
            candidate = (
                catalog.get(str(source_generation))
                if isinstance(catalog, Mapping) and type(source_generation) is int
                else None
            )
            if (
                not isinstance(adaptive, Mapping)
                or adaptive.get("schema_version")
                != "wrc_checkpoint_adaptive_rebinding_v1"
                or adaptive.get("capability_id")
                != "WRC_CHECKPOINT_ADAPTIVE_REBINDING_V1"
                or adaptive.get("restore_generation") != terminal_generation
                or adaptive.get("cell_ids") != list(base_request.cell_ids)
                or adaptive.get("execution_route_sha256")
                != base_request.execution_route_sha256
                or not isinstance(candidate, Mapping)
                or candidate.get("generation") != source_generation
                or candidate.get("wrc_checkpoint_adaptive_rebinding_v1") != adaptive
            ):
                raise CampaignExecutionError(
                    "adaptive reporting parity parent checkpoint is incomplete"
                )
            adaptive_reporting_checkpoint = dict(candidate)
    if adaptive_reporting_checkpoint is None:
        terminal_request = _narrow_request(
            base_request,
            branch_id=branch["branch_id"],
            source_index=source_index,
            source_cell_id=branch["source_cell_id"],
            generations=terminal_generation,
            checkpoints=terminal_checkpoints,
            reporting_enabled=branch["kind"] != "reporting_parity",
        )
    else:
        # A reporting-disabled adaptive sentinel must resume the exact same
        # parent B48 checkpoint. Starting one cell from G0 would compare a
        # different intervention history and make trajectory divergence look
        # like reporting mutation.
        terminal_request = replace(
            base_request,
            generations=terminal_generation,
            checkpoint_milestones=terminal_checkpoints,
            reporting_enabled=False,
        )
        checkpoint = adaptive_reporting_checkpoint
    context = BatchExecutionContextV1(
        run_root=run_root,
        batch_root=path.parent,
        manifest_path=path,
        execution_route_sha256=route_sha256,
    )

    manifest["status"] = "running"
    manifest["attempts"] = int(manifest.get("attempts", 0)) + 1
    manifest["error"] = None
    manifest["updated_at"] = _now()
    atomic_write_json(path, manifest)
    try:
        if (
            branch["kind"] == "checkpoint_continuation"
            and branch.get("baseline_mode", "fresh_branch") == "fresh_branch"
        ):
            uninterrupted_path = path.parent / "uninterrupted_baseline_manifest.json"
            uninterrupted_request = _narrow_request(
                base_request,
                branch_id=f"{branch['branch_id']}-uninterrupted",
                source_index=source_index,
                source_cell_id=branch["source_cell_id"],
                generations=terminal_generation,
                checkpoints=(terminal_generation,),
            )
            if uninterrupted_path.exists():
                uninterrupted = _read_object(
                    uninterrupted_path, name="uninterrupted comparison baseline"
                )
                if uninterrupted.get("status") != "completed":
                    raise CampaignExecutionError(
                        "uninterrupted comparison baseline is incomplete; refusing "
                        "to reinterpret it as a fresh trajectory"
                    )
            else:
                atomic_write_json(
                    uninterrupted_path,
                    {
                        "schema_version": "eiel_campaign_uninterrupted_baseline_v1",
                        "campaign_id": plan["campaign_id"],
                        "plan_sha256": plan["plan_sha256"],
                        "source_cell_id": branch["source_cell_id"],
                        "checkpoint_milestones": [terminal_generation],
                        "status": "running",
                        "checkpoint": None,
                        "metrics_rows": [],
                        "result": None,
                        "error": None,
                        "updated_at": _now(),
                    },
                )
                uninterrupted_context = BatchExecutionContextV1(
                    run_root=run_root,
                    batch_root=uninterrupted_path.parent / "uninterrupted_baseline",
                    manifest_path=uninterrupted_path,
                    execution_route_sha256=route_sha256,
                )
                uninterrupted_result = _provider_result(
                    execute(plan, uninterrupted_request, uninterrupted_context),
                    execution_route_sha256=route_sha256,
                )
                if uninterrupted_result["status"] != "completed":
                    raise CampaignExecutionError(
                        "uninterrupted comparison baseline did not reach the terminal boundary"
                    )
                uninterrupted = {
                    **_read_object(
                        uninterrupted_path,
                        name="uninterrupted comparison baseline",
                    ),
                    "status": "completed",
                    "checkpoint": dict(uninterrupted_result["checkpoint"]),
                    "metrics_rows": [
                        dict(row) for row in uninterrupted_result["metrics_rows"]
                    ],
                    "result": dict(uninterrupted_result.get("result") or {}),
                    "updated_at": _now(),
                }
                atomic_write_json(uninterrupted_path, uninterrupted)
            split = branch["split_generation"]
            if checkpoint is None and branch.get("baseline_mode") == "campaign_primary":
                baseline_result = baseline.get("result")
                catalog = (
                    baseline_result.get("checkpoint_catalog")
                    if isinstance(baseline_result, Mapping)
                    else None
                )
                source_checkpoint = (
                    catalog.get(str(split)) if isinstance(catalog, Mapping) else None
                )
                if not isinstance(source_checkpoint, Mapping):
                    raise CampaignExecutionError(
                        "campaign-primary continuation baseline lacks the "
                        f"declared G{split} checkpoint"
                    )
                checkpoint = _bind_checkpoint_route(
                    source_checkpoint,
                    route_sha256,
                    require_existing=True,
                )
            generation = (
                checkpoint.get("generation") if isinstance(checkpoint, Mapping) else 0
            )
            if (
                type(generation) is not int
                or generation < 0
                or generation > terminal_generation
            ):
                raise CampaignExecutionError(
                    "comparison checkpoint generation is invalid"
                )
            if generation < split:
                split_request = _narrow_request(
                    base_request,
                    branch_id=branch["branch_id"],
                    source_index=source_index,
                    source_cell_id=branch["source_cell_id"],
                    generations=split,
                    checkpoints=(split,),
                )
                if checkpoint is None:
                    split_result = _provider_result(
                        execute(plan, split_request, context),
                        execution_route_sha256=route_sha256,
                    )
                else:
                    split_result = _provider_result(
                        resume(plan, split_request, dict(checkpoint), context),
                        execution_route_sha256=route_sha256,
                    )
                current = _persist_comparison_result(
                    path, split_result, phase="split_completed"
                )
                if split_result["status"] == "interrupted":
                    return current
                checkpoint = current.get("checkpoint")
                if (
                    not isinstance(checkpoint, Mapping)
                    or checkpoint.get("generation") != split
                ):
                    raise CampaignExecutionError(
                        "provider did not expose the declared continuation split checkpoint"
                    )
                current["status"] = "running"
                current["updated_at"] = _now()
                atomic_write_json(path, current)
            if not isinstance(checkpoint, Mapping):
                raise CampaignExecutionError(
                    "continuation branch lacks its split checkpoint"
                )
            continuation_context = BatchExecutionContextV1(
                run_root=run_root,
                batch_root=path.parent / "continued",
                manifest_path=path,
                execution_route_sha256=route_sha256,
            )
            terminal_result = _provider_result(
                resume(
                    plan,
                    terminal_request,
                    dict(checkpoint),
                    continuation_context,
                ),
                execution_route_sha256=route_sha256,
            )
        elif checkpoint is None:
            terminal_result = _provider_result(
                execute(plan, terminal_request, context),
                execution_route_sha256=route_sha256,
            )
        else:
            terminal_result = _provider_result(
                resume(plan, terminal_request, dict(checkpoint), context),
                execution_route_sha256=route_sha256,
            )
        current = _persist_comparison_result(
            path, terminal_result, phase="terminal_completed"
        )
        if terminal_result["status"] == "interrupted":
            return current
        if branch["kind"] == "reporting_parity":
            current["comparison"] = _compare_reporting_parity(
                baseline=baseline,
                candidate=current,
                cell_id=branch["source_cell_id"],
                provider=provider,
            )
            if adaptive_reporting_checkpoint is not None:
                current["comparison"]["adaptive_parent_checkpoint_generation"] = (
                    adaptive_reporting_checkpoint["generation"]
                )
                current["comparison"]["adaptive_parent_checkpoint_sha256"] = (
                    _canonical_sha256(adaptive_reporting_checkpoint)
                )
                current["comparison"]["candidate_execution"] = (
                    "same_parent_adaptive_reporting_disabled_v1"
                )
        else:
            comparison_baseline = (
                uninterrupted
                if (
                    branch["kind"] == "checkpoint_continuation"
                    and branch.get("baseline_mode", "fresh_branch") == "fresh_branch"
                )
                else baseline
            )
            current["comparison"] = _compare_branch(
                baseline=comparison_baseline,
                candidate=current,
                cell_id=branch["source_cell_id"],
                provider=provider,
                continuation=branch["kind"] == "checkpoint_continuation",
            )
            if branch["kind"] == "checkpoint_continuation":
                campaign_primary = branch.get("baseline_mode") == "campaign_primary"
                current["comparison"]["baseline_execution"] = (
                    "campaign_primary_uninterrupted_v1"
                    if campaign_primary
                    else "fresh_terminal_only_uninterrupted_v1"
                )
                current["comparison"]["baseline_manifest_path"] = str(
                    _batch_path(run_root, source_batch["batch_id"])
                    if campaign_primary
                    else uninterrupted_path
                )
        current["comparison"]["execution_route_sha256"] = route_sha256
        current["comparison"]["branch_role"] = branch["kind"]
        current["status"] = "completed" if current["comparison"]["match"] else "failed"
        current["updated_at"] = _now()
        atomic_write_json(path, current)
        if current["status"] == "failed":
            names = (
                current["comparison"]["mismatches"]
                or current["comparison"]["missing_surfaces"]
                or [current["comparison"]["evidence_status"]]
            )
            raise CampaignExecutionError(
                f"comparison mismatch for {branch['source_cell_id']}: {', '.join(names)}"
            )
        return current
    except Exception as exc:
        current = _read_object(path, name="comparison manifest")
        current["status"] = "failed"
        current["error"] = {"type": type(exc).__name__, "message": str(exc)}
        current["updated_at"] = _now()
        atomic_write_json(path, current)
        if isinstance(exc, CampaignExecutionError):
            raise
        raise CampaignExecutionError(
            f"provider failed comparison {branch['branch_id']}: {exc}"
        ) from exc


def _comparison_campaign_manifest(
    run_root: Path,
    plan: Mapping[str, Any],
    provider: Any,
    branches: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        name: 0 for name in ("pending", "running", "interrupted", "failed", "completed")
    }
    summaries: list[dict[str, Any]] = []
    for branch in branches:
        path = _comparison_path(run_root, branch["branch_id"])
        manifest = _read_object(path, name="comparison manifest")
        status = manifest.get("status")
        if status not in counts:
            raise CampaignExecutionError(f"invalid comparison status: {status!r}")
        counts[status] += 1
        summaries.append(
            {
                "branch_id": branch["branch_id"],
                "kind": branch["kind"],
                "source_cell_id": branch["source_cell_id"],
                "status": status,
                "attempts": manifest.get("attempts"),
                "manifest_path": str(path),
            }
        )
    if not branches or counts["completed"] == len(branches):
        status = "completed"
    elif counts["failed"]:
        status = "failed"
    elif counts["interrupted"]:
        status = "interrupted"
    elif counts["running"]:
        status = "running"
    else:
        status = "pending"
    value = {
        "schema_version": "eiel_campaign_comparisons_manifest_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "status": status,
        "counts": counts,
        "branches": summaries,
        "updated_at": _now(),
    }
    atomic_write_json(run_root / "comparisons_manifest.json", value)
    return value


def _reporting_parity_receipt(
    run_root: Path, plan: Mapping[str, Any], provider: Any
) -> dict[str, Any] | None:
    comparisons = plan.get("comparisons")
    requirement = (
        comparisons.get("reporting_parity")
        if isinstance(comparisons, Mapping)
        else None
    )
    if not isinstance(requirement, Mapping) or requirement.get("required") is not True:
        return None
    accepted = requirement.get("accepted_evidence_modes")
    if not isinstance(accepted, list) or not all(
        isinstance(row, str) for row in accepted
    ):
        raise CampaignExecutionError("compiled reporting parity requirement is invalid")
    reporting_branch = next(
        (
            row
            for row in comparisons.get("branches", [])
            if isinstance(row, Mapping) and row.get("kind") == "reporting_parity"
        ),
        None,
    )
    branch_comparison = None
    if isinstance(reporting_branch, Mapping):
        branch_manifest = _read_object(
            _comparison_path(run_root, reporting_branch.get("branch_id")),
            name="reporting parity branch manifest",
        )
        branch_comparison = branch_manifest.get("comparison")
    if (
        isinstance(branch_comparison, Mapping)
        and branch_comparison.get("match") is True
    ):
        evidence = {
            "status": "supported",
            "evidence_mode": "enabled_disabled_exact_terminal_state",
            "observational_contract": "reporting_disabled_skips_compact_materialization_only_v1",
            "state_mutation_sentinel": branch_comparison.get("state_mutation_sentinel"),
            "reason_code": None,
            "reason": None,
        }
    else:
        evidence_owner = getattr(provider, "reporting_parity_evidence", None)
        if callable(evidence_owner):
            evidence = _json_object(
                evidence_owner(plan), name="provider reporting parity evidence"
            )
        else:
            evidence = {
                "status": "unsupported",
                "evidence_mode": None,
                "observational_contract": None,
                "state_mutation_sentinel": None,
                "reason_code": "provider_reporting_parity_contract_unavailable",
                "reason": "the registered provider declares no reporting parity evidence contract",
            }
    status = evidence.get("status")
    mode = evidence.get("evidence_mode")
    if status not in {"supported", "unsupported"}:
        raise CampaignExecutionError("provider reporting parity status is invalid")
    if status == "supported":
        if mode not in accepted:
            raise CampaignExecutionError(
                "provider reporting parity mode is not admitted"
            )
        if mode == "provider_observational_contract_with_state_mutation_sentinel" and (
            not isinstance(evidence.get("observational_contract"), str)
            or not isinstance(evidence.get("state_mutation_sentinel"), str)
        ):
            raise CampaignExecutionError(
                "provider reporting observational evidence lacks its state-mutation sentinel"
            )
    receipt = {
        "schema_version": "eiel_campaign_reporting_parity_receipt_v1",
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "profile_id": plan["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": _execution_route_sha256(provider, plan),
        "comparison_branch_id": (
            reporting_branch.get("branch_id")
            if isinstance(reporting_branch, Mapping)
            else None
        ),
        "comparison_receipt_sha256": (
            _canonical_sha256(branch_comparison)
            if isinstance(branch_comparison, Mapping)
            else None
        ),
        "comparison_first_difference": (
            branch_comparison.get("first_difference")
            if isinstance(branch_comparison, Mapping)
            else None
        ),
        "qualification_evidence_status": (
            "reporting_parity_proven"
            if status == "supported"
            else "reporting_parity_unsupported"
        ),
        **evidence,
        "updated_at": _now(),
    }
    atomic_write_json(run_root / "reporting_parity_receipt.json", receipt)
    return receipt


def _execute_campaign_impl(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    provider_registry: ExecutionProviderRegistryV1,
    qualification_mode: bool = False,
    execution_scope: str = "qualified_execution",
    nested_storage_parent_run_root: str | Path | None = None,
    before_lifecycle_closeout_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute or safely continue one compiled plan through its provider."""

    payload = _json_object(plan, name="compiled plan")
    for name in ("campaign_id", "plan_sha256", "resolved_profile"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise CampaignExecutionError(f"compiled plan lacks {name}")
    from .run_contract_v1 import validate_resolved_run_contract_v1

    validate_resolved_run_contract_v1(payload)
    root = run_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        lifecycle = ensure_lifecycle_contract_v1(payload)
    except (ArtifactLifecycleError, TypeError, ValueError) as exc:
        raise CampaignExecutionError(
            f"invalid run artifact lifecycle contract: {exc}"
        ) from exc
    estimate = payload.get("resource_estimate")
    planned_storage_bytes = (
        int(estimate["disk_bytes_upper_bound"])
        if isinstance(estimate, Mapping)
        and type(estimate.get("disk_bytes_upper_bound")) is int
        else None
    )
    try:
        reservation_window_bytes = (
            progressive_storage_reservation_window_bytes_v1(
                estimate if isinstance(estimate, Mapping) else None
            )
        )
    except ArtifactLifecycleError as exc:
        raise CampaignExecutionError(
            f"invalid progressive lifecycle storage reservation: {exc}"
        ) from exc
    storage_guard = RunStorageGuard(
        lifecycle,
        run_root=root,
        planned_bytes=planned_storage_bytes,
        reservation_window_bytes=reservation_window_bytes,
        same_owner_nested_parent_run_root=nested_storage_parent_run_root,
    )
    atomic_write_json(root / "artifact_lifecycle_contract.json", lifecycle)
    preflight_path = root / "preflight_receipt.json"
    if not preflight_path.is_file():
        raise CampaignExecutionError(
            "campaign execution requires a durable preflight receipt"
        )
    preflight = _read_object(preflight_path, name="preflight receipt")
    expected_preflight = {
        "schema_version": "eiel_campaign_preflight_v1",
        "campaign_id": payload["campaign_id"],
        "plan_sha256": payload["plan_sha256"],
        (
            "ready_for_route_qualification"
            if qualification_mode
            else "ready_for_execution"
        ): True,
    }
    for name, expected in expected_preflight.items():
        if preflight.get(name) != expected:
            raise CampaignExecutionError(
                f"preflight receipt does not grant execution authority: {name}"
            )
    if execution_scope not in {
        "qualified_execution",
        "qualprep_development_only",
        "engineering_unqualified_real_use",
        "engineering_only",
    }:
        raise CampaignExecutionError("campaign execution scope is unsupported")
    observed_execution_scope = preflight.get("execution_authority_scope")
    if not qualification_mode and (
        (
            execution_scope
            in {
                "qualprep_development_only",
                "engineering_unqualified_real_use",
                "engineering_only",
            }
            and observed_execution_scope != execution_scope
        )
        or (
            observed_execution_scope is not None
            and observed_execution_scope != execution_scope
        )
    ):
        raise CampaignExecutionError(
            "preflight receipt execution-authority scope does not match dispatch"
        )
    authority_reasons = preflight.get("execution_authority_reasons")
    allowed_qualification_codes = {
        "execution_route_obligation_uncovered",
        "execution_route_hash_mismatch",
        "execution_route_profile_mismatch",
        "execution_route_qualification_missing",
        "provider_vram_bytes_upper_bound_unavailable",
        "profile_not_execution_qualified",
    }
    unresolved_authority = (
        [
            row
            for row in authority_reasons
            if not isinstance(row, Mapping)
            or row.get("code") not in allowed_qualification_codes
        ]
        if (
            qualification_mode
            or execution_scope
            in {
                "qualprep_development_only",
                "engineering_unqualified_real_use",
                "engineering_only",
            }
        )
        and isinstance(authority_reasons, list)
        else authority_reasons
    )
    if preflight.get("reasons") not in (None, []) or unresolved_authority not in (
        None,
        [],
    ):
        raise CampaignExecutionError(
            "preflight receipt contains unresolved execution reasons"
        )
    provider = provider_registry.get(payload["resolved_profile"])
    if provider is None:
        raise CampaignExecutionError(
            f"profile {payload['resolved_profile']!r} has no registered execution provider"
        )
    execution_route_sha256 = _execution_route_sha256(provider, payload)
    if execution_route_sha256 is not None:
        route_preflight = preflight.get("execution_route")
        development_unqualified_route = bool(
            not qualification_mode
            and execution_scope
            in {
                "qualprep_development_only",
                "engineering_unqualified_real_use",
                "engineering_only",
            }
            and isinstance(route_preflight, Mapping)
            and route_preflight.get("authority_state")
            in {"legacy_unbound", "qualification_only"}
            and route_preflight.get(f"{execution_scope}_route_authorized") is True
        )
        if (
            not isinstance(route_preflight, Mapping)
            or route_preflight.get("execution_route_sha256") != execution_route_sha256
            or (
                route_preflight.get("authority_state")
                not in (
                    {"legacy_unbound", "qualification_only", "route_qualified"}
                    if (
                        qualification_mode
                        or execution_scope
                        in {
                            "qualprep_development_only",
                            "engineering_unqualified_real_use",
                            "engineering_only",
                        }
                    )
                    else {"route_qualified"}
                )
                and not development_unqualified_route
            )
            or (
                not qualification_mode
                and execution_scope
                not in {
                    "qualprep_development_only",
                    "engineering_unqualified_real_use",
                    "engineering_only",
                }
                and route_preflight.get("artifact_conformance_exact") is not True
                and not development_unqualified_route
            )
        ):
            raise CampaignExecutionError(
                "preflight receipt does not grant execution-route authority"
            )

    batches = _batch_rows(payload)
    # Repair an interrupted/restarted compact resident campaign before the
    # first lifecycle admission check.  Completed batches are authoritative in
    # the durable zone and must not keep consuming the active checkpoint cap.
    for batch in batches:
        path = _batch_path(root, batch["batch_id"])
        manifest = _load_batch_manifest(
            path, plan=payload, provider=provider, batch=batch
        )
        if manifest.get("status") == "completed":
            _compact_resident_checkpoint_settle_v1(
                plan=payload,
                batch=batch,
                manifest_path=path,
                manifest=manifest,
            )
    try:
        storage_guard.enforce(
            stage="before_campaign",
            pre_admission_reclaimer=reclaim_before_storage_wait_v1,
        )
    except StorageBudgetExceeded as exc:
        return _materialize_storage_wait_v1(
            plan=payload,
            lifecycle=lifecycle,
            run_root=root,
            provider=provider,
            batches=batches,
            execution_route_sha256=execution_route_sha256,
            exc=exc,
        )
    initial_passport = build_run_passport_v1(
        payload,
        run_root=root,
        terminal_status="active",
        provider_identity={
            "provider_id": getattr(provider, "provider_id", None),
            "profile_id": payload["resolved_profile"],
            "execution_route_sha256": execution_route_sha256,
        },
        disposition_manifest_path=str(root / "artifact_disposition_manifest.json"),
        disposition_receipt_path=str(root / "storage_disposition_receipt.json"),
    )
    write_run_passport_v1(root / "run_passport.json", initial_passport)
    execute = getattr(provider, "execute_batch", None)
    resume = getattr(provider, "resume_batch", None)
    if not callable(execute):
        raise CampaignExecutionError(
            f"provider {provider.provider_id!r} has no batch execution capability"
        )

    for batch in batches:
        try:
            storage_guard.enforce(
                stage=f"before_batch:{batch['batch_id']}",
                pre_admission_reclaimer=reclaim_before_storage_wait_v1,
            )
        except StorageBudgetExceeded as exc:
            return _materialize_storage_wait_v1(
                plan=payload,
                lifecycle=lifecycle,
                run_root=root,
                provider=provider,
                batches=batches,
                execution_route_sha256=execution_route_sha256,
                exc=exc,
            )
        path = _batch_path(root, batch["batch_id"])
        _load_batch_manifest(path, plan=payload, provider=provider, batch=batch)
    campaign = _campaign_manifest(root, payload, provider, batches)
    _write_run_contract_artifacts_v1(
        root,
        payload,
        batches,
        campaign_status=str(campaign["status"]),
    )
    if not qualification_mode:
        _execute_pre_primary_technical_gate(
            plan=payload,
            run_root=root,
            provider=provider,
            execute=execute,
            resume=resume,
            batches=batches,
        )

    for batch in batches:
        gate_receipt_path = root / "technical_gate_receipt.json"
        gate = payload.get("execution", {}).get("technical_gate", {})
        gate_batch_ids = gate.get("batch_ids", []) if isinstance(gate, Mapping) else []
        if (
            isinstance(gate, Mapping)
            and gate.get("required") is True
            and gate.get("execution_mode") != "pre_primary_canary"
            and batch["batch_id"] not in gate_batch_ids
            and not gate_receipt_path.exists()
        ):
            receipt = _technical_gate_receipt(root, payload, provider, batches)
            if receipt is None:
                raise CampaignExecutionError(
                    "technical gate batches are incomplete before main-stage dispatch"
                )
        path = _batch_path(root, batch["batch_id"])
        manifest = _load_batch_manifest(
            path, plan=payload, provider=provider, batch=batch
        )
        status = manifest.get("status")
        if status == "completed":
            continue
        checkpoint = manifest.get("checkpoint")
        needs_resume = status in {"running", "interrupted", "failed"}
        if needs_resume and not isinstance(checkpoint, Mapping):
            raise CampaignExecutionError(
                f"batch {batch['batch_id']} is {status} without a durable checkpoint; "
                "refusing duplicate dispatch"
            )
        if needs_resume and not callable(resume):
            raise CampaignExecutionError(
                f"provider {provider.provider_id!r} has no batch resume capability"
            )

        bound_request = build_execution_request_v1(provider, payload, batch["batch_id"])
        request = bound_request.resident_request
        preflight_requests = preflight.get("execution_requests")
        if isinstance(preflight_requests, Mapping):
            preflight_request = preflight_requests.get(batch["batch_id"])
            if (
                not isinstance(preflight_request, Mapping)
                or preflight_request.get("bound_request_sha256")
                != bound_request.bound_request_sha256
            ):
                raise CampaignExecutionError(
                    "execution request differs from the durable preflight binding"
                )
        if (
            execution_route_sha256 is not None
            and request.execution_route_sha256 != execution_route_sha256
        ):
            raise CampaignExecutionError(
                "resident request does not bind the admitted execution route"
            )
        if isinstance(checkpoint, Mapping):
            checkpoint = _bind_checkpoint_route(
                checkpoint,
                execution_route_sha256,
                require_existing=True,
            )
        manifest["status"] = "running"
        manifest["attempts"] = int(manifest.get("attempts", 0)) + 1
        manifest["error"] = None
        manifest["updated_at"] = _now()
        atomic_write_json(path, manifest)
        _campaign_manifest(root, payload, provider, batches)
        from engine.monitoring.slot_telemetry import GPU_CAPACITY, start_slot_lease

        tick_native = request.total_ticks is not None
        if tick_native:
            start_absolute_tick = int(request.start_next_tick or 0)
            if needs_resume and isinstance(checkpoint, Mapping):
                start_absolute_tick = _native_checkpoint_next_absolute_tick_v1(
                    checkpoint
                )
            slot_bounds = {
                "start_absolute_tick": start_absolute_tick,
                "target_absolute_tick": request.total_ticks,
            }
        else:
            execution = payload.get("execution")
            generations = (
                execution.get("generations") if isinstance(execution, Mapping) else None
            )
            ticks = (
                execution.get("ticks_per_generation")
                if isinstance(execution, Mapping)
                else None
            )
            start_generation = 0
            if needs_resume and isinstance(checkpoint, Mapping):
                checkpoint_generation = checkpoint.get("generation")
                if type(checkpoint_generation) is int:
                    start_generation = checkpoint_generation
            slot_bounds = {
                "current_generation": start_generation,
                "target_generation": int(generations or request.generations),
                "ticks_per_generation": int(ticks or request.ticks_per_generation),
            }
        slot_cell_ids = tuple(request.cell_ids)
        if len(slot_cell_ids) > GPU_CAPACITY:
            slot_cell_ids = (f"physical-batch:{batch['batch_id']}",)
        slot_lease = start_slot_lease(
            device="gpu",
            run_id=f"{payload['campaign_id']}:{batch['batch_id']}",
            cell_ids=slot_cell_ids,
            **slot_bounds,
            run_root=root,
            campaign_id=payload["campaign_id"],
            batch_id=batch["batch_id"],
            profile_id=payload["resolved_profile"],
            logical_world_count=len(request.cell_ids),
            **_gpu_slot_admission_metadata_v1(root, payload),
        )
        context = BatchExecutionContextV1(
            run_root=root,
            batch_root=path.parent,
            manifest_path=path,
            slot_lease=slot_lease,
            execution_route_sha256=execution_route_sha256,
            lifecycle_contract=lifecycle,
            storage_guard=storage_guard,
            provider_id=getattr(provider, "provider_id", None),
        )
        try:
            slot_lease.update(
                **(
                    {"next_absolute_tick": start_absolute_tick}
                    if tick_native
                    else {"current_generation": start_generation}
                ),
                status="running",
                stage="native batch dispatch",
            )
        except Exception:
            pass
        try:
            if needs_resume:
                raw_result = resume(payload, request, dict(checkpoint), context)
            else:
                raw_result = execute(payload, request, context)
            result = _json_object(raw_result, name="provider batch result")
            terminal_status = result.get("status")
            if terminal_status not in {"completed", "interrupted"}:
                raise CampaignExecutionError(
                    "provider batch result status must be completed or interrupted"
                )
            current = _read_object(path, name="batch manifest")
            returned_checkpoint = result.get("checkpoint")
            if returned_checkpoint is not None:
                current["checkpoint"] = _bind_checkpoint_route(
                    _json_object(
                        returned_checkpoint, name="provider result checkpoint"
                    ),
                    execution_route_sha256,
                )
            if terminal_status == "interrupted" and not isinstance(
                current.get("checkpoint"), Mapping
            ):
                raise CampaignExecutionError(
                    "interrupted provider result lacks a durable checkpoint"
                )
            metrics_rows = result.get("metrics_rows", [])
            if not isinstance(metrics_rows, list) or not all(
                isinstance(row, Mapping) for row in metrics_rows
            ):
                raise CampaignExecutionError(
                    "provider metrics_rows must be an array of objects"
                )
            provider_result = result.get("result")
            if provider_result is not None and not isinstance(provider_result, Mapping):
                raise CampaignExecutionError(
                    "provider result payload must be an object"
                )
            current["status"] = terminal_status
            current["metrics_rows"] = [dict(row) for row in metrics_rows]
            current["result"] = (
                None if provider_result is None else dict(provider_result)
            )
            current["error"] = None
            current["updated_at"] = _now()
            atomic_write_json(path, current)
            if terminal_status == "completed":
                current = _compact_resident_checkpoint_settle_v1(
                    plan=payload,
                    batch=batch,
                    manifest_path=path,
                    manifest=current,
                )
            try:
                slot_lease.close(
                    status=(
                        "complete" if terminal_status == "completed" else "interrupted"
                    )
                )
            except Exception:
                pass
        except Exception as exc:
            current = _read_object(path, name="batch manifest")
            current["status"] = "failed"
            current["error"] = {"type": type(exc).__name__, "message": str(exc)}
            current["updated_at"] = _now()
            atomic_write_json(path, current)
            campaign = _campaign_manifest(root, payload, provider, batches)
            _write_run_contract_artifacts_v1(
                root,
                payload,
                batches,
                campaign_status=str(campaign["status"]),
            )
            try:
                slot_lease.close(status="failed", error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            if isinstance(exc, CampaignExecutionError):
                raise
            raise CampaignExecutionError(
                f"provider failed batch {batch['batch_id']}: {exc}"
            ) from exc
        campaign = _campaign_manifest(root, payload, provider, batches)
        if campaign["status"] == "interrupted":
            break
        try:
            storage_guard.enforce(
                stage=f"after_batch:{batch['batch_id']}",
                pre_admission_reclaimer=reclaim_before_storage_wait_v1,
            )
        except StorageBudgetExceeded as exc:
            return _materialize_storage_wait_v1(
                plan=payload,
                lifecycle=lifecycle,
                run_root=root,
                provider=provider,
                batches=batches,
                execution_route_sha256=execution_route_sha256,
                exc=exc,
            )

    campaign = _campaign_manifest(root, payload, provider, batches)
    _write_run_contract_artifacts_v1(
        root,
        payload,
        batches,
        campaign_status=str(campaign["status"]),
    )
    technical_gate = _technical_gate_receipt(root, payload, provider, batches)
    metrics_path = _write_metrics(root, payload, batches, provider)
    generation_metrics_path = _write_generation_metrics(root, payload, batches)
    cell_receipts_path = _write_cell_receipts(root, payload, batches, provider)
    comparison_branches = _comparison_rows(payload)
    comparisons: dict[str, Any] | None = None
    reporting_parity: dict[str, Any] | None = None
    if campaign["status"] == "completed":
        for branch in comparison_branches:
            source_batch, source_index = _source_batch(
                batches, branch["source_cell_id"]
            )
            _load_comparison_manifest(
                _comparison_path(root, branch["branch_id"]),
                plan=payload,
                provider=provider,
                branch=branch,
                source_batch_id=source_batch["batch_id"],
                source_index=source_index,
            )
        comparisons = _comparison_campaign_manifest(
            root, payload, provider, comparison_branches
        )
        for branch in comparison_branches:
            try:
                result = _execute_comparison_branch(
                    plan=payload,
                    run_root=root,
                    provider=provider,
                    execute=execute,
                    resume=resume,
                    branch=branch,
                    batches=batches,
                )
            except Exception:
                _comparison_campaign_manifest(
                    root, payload, provider, comparison_branches
                )
                raise
            comparisons = _comparison_campaign_manifest(
                root, payload, provider, comparison_branches
            )
            if result.get("status") == "interrupted":
                break
        if comparisons["status"] == "completed":
            reporting_parity = _reporting_parity_receipt(root, payload, provider)
    status = campaign["status"]
    if comparisons is not None and comparisons["status"] != "completed":
        status = comparisons["status"]
    if status == "completed" and before_lifecycle_closeout_fn is not None:
        if not callable(before_lifecycle_closeout_fn):
            raise CampaignExecutionError(
                "before_lifecycle_closeout_fn must be callable"
            )
        try:
            # Consumer-owned derived artifacts (for example an asynchronous
            # science reducer) must finish before this function walks and
            # seals the run inventory.  This is a post-producer consumer
            # boundary; it never creates producer backpressure.
            before_lifecycle_closeout_fn()
        except Exception as exc:
            raise CampaignExecutionError(
                "pre-lifecycle-closeout hook failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    chronicle_archive_receipt: dict[str, Any] | None = None
    evidence_closeout_receipt: dict[str, Any] | None = None
    output = payload.get("output")
    chronicle_archive_enabled = (
        isinstance(output, Mapping) and output.get("chronicle_archive_enabled") is True
    )
    if status == "completed" and chronicle_archive_enabled and not qualification_mode:
        from engine.campaigns.chronicle_closeout import (
            CampaignChronicleCloseoutError,
            materialize_completed_campaign_chronicles,
        )
        from simulation.chronicle.archive_history import (
            ArchiveCollisionError,
            ChronicleArchiveError,
            create_completed_run_archive,
        )

        try:
            batch_manifests = [
                _read_object(
                    _batch_path(root, batch["batch_id"]),
                    name="completed batch manifest",
                )
                for batch in batches
            ]
            world_chronicles = materialize_completed_campaign_chronicles(
                payload,
                run_root=root,
                batch_manifests=batch_manifests,
            )
            history_root = Path(str(output["chronicle_history_root"])).resolve()
            campaign_id = str(payload["campaign_id"])
            archive_metadata = {
                "plan_sha256": payload["plan_sha256"],
                "profile_id": payload["resolved_profile"],
                "provider_id": provider.provider_id,
                "execution_route_sha256": execution_route_sha256,
                "chronicle_profile": output.get("chronicle_profile", "notable"),
                "source_run_root": str(root),
            }
            archive_run_id_scheme = "legacy_basename"
            try:
                record = create_completed_run_archive(
                    history_root,
                    campaign_id=campaign_id,
                    run_id=root.name,
                    worlds=world_chronicles,
                    metadata=archive_metadata,
                )
            except ArchiveCollisionError:
                archive_run_id_scheme = "bound_execution_v2"
                record = create_completed_run_archive(
                    history_root,
                    campaign_id=campaign_id,
                    run_id=_chronicle_archive_run_id(
                        root,
                        campaign_id=campaign_id,
                        plan_sha256=str(payload["plan_sha256"]),
                        profile_id=str(payload["resolved_profile"]),
                        provider_id=provider.provider_id,
                        execution_route_sha256=execution_route_sha256,
                    ),
                    worlds=world_chronicles,
                    metadata=archive_metadata,
                )
            loose_root = root / "chronicle_worlds"
            if loose_root.is_dir() and not loose_root.is_symlink():
                shutil.rmtree(loose_root)
        except (
            CampaignChronicleCloseoutError,
            ChronicleArchiveError,
            OSError,
        ) as exc:
            raise CampaignExecutionError(
                f"completed campaign Chronicle closeout failed: {exc}"
            ) from exc
        chronicle_archive_receipt = {
            "schema_version": "eiel_campaign_chronicle_archive_receipt_v1",
            "campaign_id": record.campaign_id,
            "run_id": record.run_id,
            "run_id_scheme": archive_run_id_scheme,
            "plan_sha256": payload["plan_sha256"],
            "execution_route_sha256": execution_route_sha256,
            "chronicle_profile": output.get("chronicle_profile", "notable"),
            "archive_path": str(record.archive_path),
            "archive_sha256": record.sha256,
            "archive_bytes": record.size_bytes,
            "world_ids": list(record.world_ids),
            "member_count": record.member_count,
            "verified": True,
            "loose_materialized_sources_removed": True,
            "git_payload_policy": "filesystem_only_no_automatic_git_commit",
        }
        atomic_write_json(
            root / "chronicle_archive_receipt.json",
            chronicle_archive_receipt,
        )
    evidence_closeout_enabled = (
        isinstance(output, Mapping) and output.get("evidence_closeout_enabled") is True
    )
    if status == "completed" and evidence_closeout_enabled and not qualification_mode:
        from engine.campaigns.evidence_closeout import (
            CampaignEvidenceCloseoutError,
            closeout_campaign_evidence,
        )

        try:
            expected_cell_ids = [
                str(cell["cell_id"])
                for cell in payload.get("cells", [])
                if isinstance(cell, Mapping)
            ]
            provider_cell_results = _completed_provider_cell_results(
                root,
                batches,
                expected_cell_ids=expected_cell_ids,
            )
            evidence_closeout_receipt = closeout_campaign_evidence(
                payload,
                run_root=root,
                provider_cell_results=provider_cell_results,
            )
            atomic_write_json(
                root / "evidence_closeout_receipt.json",
                evidence_closeout_receipt,
            )
        except (CampaignEvidenceCloseoutError, OSError) as exc:
            raise CampaignExecutionError(
                f"completed campaign evidence closeout failed: {exc}"
            ) from exc
    try:
        completed_batch_manifests = [
            _read_object(
                _batch_path(root, batch["batch_id"]),
                name="batch manifest",
            )
            for batch in batches
        ]
        artifact_inventory, inventory_complete = _collect_lifecycle_inventory_v1(
            run_root=root,
            batch_manifests=completed_batch_manifests,
            lifecycle=lifecycle,
            terminal_status=status,
        )
        checkpoint_catalog = _lifecycle_checkpoint_catalog(
            root,
            completed_batch_manifests,
            lifecycle,
        )
        replacement_references: list[dict[str, Any]] = []
        for receipt in (chronicle_archive_receipt, evidence_closeout_receipt):
            if isinstance(receipt, Mapping):
                if isinstance(receipt.get("replacement_artifacts"), list):
                    replacement_references.extend(
                        dict(row)
                        for row in receipt["replacement_artifacts"]
                        if isinstance(row, Mapping)
                    )
                replacement_references.append(
                    {
                        key: receipt[key]
                        for key in (
                            "schema_version",
                            "archive_path",
                            "archive_sha256",
                            "zip_path",
                            "zip_sha256",
                            "verified",
                        )
                        if key in receipt
                    }
                )
        replacement_verified = any(
            row.get("verified") is True for row in replacement_references
        )
        disposition = build_artifact_disposition_manifest_v1(
            payload,
            run_root=root,
            terminal_status=status,
            live_reference_status="none" if status == "completed" else "active",
            audit_hold_status=(
                "active"
                if lifecycle["retention"]["audit_hold_binding"] == "required"
                else "not_required"
            ),
            qualification_hold_status=(
                "active"
                if qualification_mode
                or lifecycle["retention"]["qualification_hold_binding"] == "required"
                else "not_required"
            ),
            replacement_verified=replacement_verified,
            replacement_artifacts=replacement_references,
            checkpoint_catalog=checkpoint_catalog,
            artifact_inventory=artifact_inventory,
            inventory_complete=inventory_complete,
            # Engineering direct-sparse runs publish one exact file row per
            # one-tick envelope and sink receipt.  Preserve that exact
            # producer inventory for bounded long-horizon rehearsals instead
            # of silently aggregating it (aggregation is advisory/RX).  The
            # qualification and production paths retain the conservative
            # lifecycle default of 4096 objects.
            max_objects=(
                max(4096, len(artifact_inventory))
                if execution_scope
                in {"engineering_unqualified_real_use", "engineering_only"}
                else 4096
            ),
        )
        disposition_path = root / "artifact_disposition_manifest.json"
        with run_artifact_transaction_fence_v1(root):
            _revalidate_disposition_source_rows_v1(root, disposition)
            atomic_write_json(disposition_path, disposition)
            produced_bytes = int(disposition["summary"]["produced_bytes"])
            disposition_receipt = build_storage_disposition_receipt_v1(
                manifest=disposition,
                bytes_before=produced_bytes,
                bytes_retained=produced_bytes,
                bytes_released=0,
                bytes_unresolved_rx=int(disposition["summary"]["unresolved_rx_bytes"]),
                authorization_reason="dry_run_only_no_destructive_cleanup_authority",
                replacement_references=replacement_references,
                retention_references=[
                    str(root / "artifact_disposition_manifest.json"),
                    *(
                        [str(root / "evidence_closeout_receipt.json")]
                        if evidence_closeout_receipt is not None
                        else []
                    ),
                ],
                plan_sha256=str(payload["plan_sha256"]),
                timestamp_utc=_now(),
                result="dry_run",
            )
            disposition_receipt_path = root / "storage_disposition_receipt.json"
            write_storage_disposition_receipt_v1(
                disposition_receipt_path,
                disposition_receipt,
            )
            archive_pointer_path = (
                register_pending_durable_archive_v1(
                    disposition_path,
                    _transaction_fence_held=True,
                )
                if status == "completed"
                else None
            )
        final_passport = build_run_passport_v1(
            payload,
            run_root=root,
            terminal_status=status,
            checkpoint_catalog=checkpoint_catalog,
            provider_identity={
                "provider_id": getattr(provider, "provider_id", None),
                "profile_id": payload["resolved_profile"],
                "execution_route_sha256": execution_route_sha256,
            },
            disposition_manifest_path=str(disposition_path),
            disposition_receipt_path=str(disposition_receipt_path),
        )
        write_run_passport_v1(root / "run_passport.json", final_passport)
    except (ArtifactLifecycleError, OSError, ValueError) as exc:
        raise CampaignExecutionError(
            f"run artifact lifecycle closeout failed: {exc}"
        ) from exc
    interruption: dict[str, Any] = {}
    if status == "interrupted":
        batch_manifests = [
            _read_object(_batch_path(root, batch["batch_id"]), name="batch manifest")
            for batch in batches
        ]
        interrupted = [
            manifest
            for manifest in batch_manifests
            if manifest.get("status") == "interrupted"
        ]
        if len(interrupted) > 1:
            raise CampaignExecutionError(
                "interrupted campaign identifies more than one physical batch"
            )
        if interrupted:
            checkpoint = interrupted[0].get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise CampaignExecutionError(
                    "interrupted batch lacks checkpoint authority"
                )
            interruption = {
                "interrupted_batch_id": interrupted[0].get("batch_id"),
                "interrupted_checkpoint_next_tick": (
                    _interrupted_checkpoint_next_tick_v1(checkpoint)
                ),
            }
    return {
        "ok": status == "completed",
        "schema_version": "eiel_campaign_execution_receipt_v1",
        "campaign_id": payload["campaign_id"],
        "plan_sha256": payload["plan_sha256"],
        "profile_id": payload["resolved_profile"],
        "provider_id": provider.provider_id,
        "execution_route_sha256": execution_route_sha256,
        "status": status,
        **interruption,
        "counts": campaign["counts"],
        "comparison_counts": None if comparisons is None else comparisons["counts"],
        "campaign_manifest_path": str(root / "campaign_manifest.json"),
        "comparisons_manifest_path": (
            None if comparisons is None else str(root / "comparisons_manifest.json")
        ),
        "reporting_parity_status": (
            None
            if reporting_parity is None
            else reporting_parity["qualification_evidence_status"]
        ),
        "reporting_parity_receipt_path": (
            None
            if reporting_parity is None
            else str(root / "reporting_parity_receipt.json")
        ),
        "cell_metrics_path": str(metrics_path),
        "chronicle_archive_receipt_path": (
            None
            if chronicle_archive_receipt is None
            else str(root / "chronicle_archive_receipt.json")
        ),
        "chronicle_archive_path": (
            None
            if chronicle_archive_receipt is None
            else chronicle_archive_receipt["archive_path"]
        ),
        "generation_metrics_path": (
            None if generation_metrics_path is None else str(generation_metrics_path)
        ),
        "cell_receipts_path": str(cell_receipts_path),
        "technical_gate_receipt_path": (
            None
            if technical_gate is None
            else str(root / "technical_gate_receipt.json")
        ),
        "evidence_closeout_receipt": evidence_closeout_receipt,
        "evidence_closeout_receipt_path": (
            None
            if evidence_closeout_receipt is None
            else str(root / "evidence_closeout_receipt.json")
        ),
        "artifact_lifecycle_contract_path": str(
            root / "artifact_lifecycle_contract.json"
        ),
        "run_passport_path": str(root / "run_passport.json"),
        "artifact_disposition_manifest_path": str(
            root / "artifact_disposition_manifest.json"
        ),
        "storage_disposition_receipt_path": str(
            root / "storage_disposition_receipt.json"
        ),
        "durable_archive_pointer_path": (
            None if archive_pointer_path is None else str(archive_pointer_path)
        ),
    }


def execute_campaign(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    provider_registry: ExecutionProviderRegistryV1,
    qualification_mode: bool = False,
    execution_scope: str = "qualified_execution",
    nested_storage_parent_run_root: str | Path | None = None,
    before_lifecycle_closeout_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute one campaign and quarantine abnormal exits before re-raising."""

    root = run_root.expanduser().resolve()
    from .run_contract_v1 import validate_resolved_run_contract_v1

    validate_resolved_run_contract_v1(_json_object(plan, name="compiled plan"))
    lifecycle: dict[str, Any] | None = None
    try:
        lifecycle = ensure_lifecycle_contract_v1(plan)
        return _execute_campaign_impl(
            plan,
            run_root=root,
            provider_registry=provider_registry,
            qualification_mode=qualification_mode,
            execution_scope=execution_scope,
            nested_storage_parent_run_root=nested_storage_parent_run_root,
            before_lifecycle_closeout_fn=before_lifecycle_closeout_fn,
        )
    except Exception as exc:
        disposition_path = root / "artifact_disposition_manifest.json"
        should_write = True
        if disposition_path.is_file():
            try:
                existing = _read_object(
                    disposition_path,
                    name="existing artifact disposition manifest",
                )
                should_write = existing.get("terminal_status") != "completed"
            except CampaignExecutionError:
                should_write = True
        if should_write:
            try:
                receipt = build_run_artifact_failure_receipt_v1(
                    plan,
                    run_root=root,
                    stage="campaign_execution",
                    failure_type=type(exc).__name__,
                    message=str(exc),
                )
                write_run_artifact_failure_receipt_v1(
                    root / "run_artifact_failure_receipt.json",
                    receipt,
                )
            except Exception:
                # The original failure remains authoritative.  A missing
                # receipt is still fail-closed because no disposition authority
                # is created and the run cannot become automatically prunable.
                pass
        raise
    finally:
        if lifecycle is not None:
            try:
                RunStorageGuard.release_for_run(lifecycle, run_root=root)
            except (ArtifactLifecycleError, OSError, ValueError):
                # A stale reservation is fail-closed: future admission will
                # reconcile the recorded PID and no destructive authority is
                # created by reservation cleanup failure.
                pass


def prepare_adaptive_campaign_continuation_v1(
    source_plan: Mapping[str, Any],
    restore_plan: Mapping[str, Any],
    *,
    source_run_root: Path,
    restore_run_root: Path,
    provider_registry: ExecutionProviderRegistryV1,
) -> dict[str, Any]:
    """Seed a preflighted stage root with one validated adaptive checkpoint."""

    source = _json_object(source_plan, name="source execution plan")
    restore = _json_object(restore_plan, name="restore execution plan")
    for plan, label in ((source, "source"), (restore, "restore")):
        for field in ("campaign_id", "plan_sha256", "resolved_profile"):
            if not isinstance(plan.get(field), str) or not plan[field]:
                raise CampaignExecutionError(f"{label} plan lacks {field}")
    if source["resolved_profile"] != restore["resolved_profile"]:
        raise CampaignExecutionError("adaptive continuation changes runtime profile")
    if (
        source.get("qualification_key") != restore.get("qualification_key")
        or source.get("qualification_key_sha256")
        != restore.get("qualification_key_sha256")
        or source.get("artifacts") != restore.get("artifacts")
    ):
        raise CampaignExecutionError(
            "adaptive continuation changes qualification or artifacts"
        )
    source_batches = _batch_rows(source)
    restore_batches = _batch_rows(restore)
    if (
        len(source_batches) != 1
        or len(restore_batches) != 1
        or source_batches[0] != restore_batches[0]
        or len(source_batches[0]["cell_ids"]) != 48
    ):
        raise CampaignExecutionError(
            "adaptive continuation requires one unchanged physical B48"
        )
    provider = provider_registry.get(source["resolved_profile"])
    if provider is None:
        raise CampaignExecutionError("adaptive continuation provider is unavailable")
    source_route = _execution_route_sha256(provider, source)
    restore_route = _execution_route_sha256(provider, restore)
    if source_route is None or source_route != restore_route:
        raise CampaignExecutionError("adaptive continuation changes execution route")
    source_root = source_run_root.expanduser().resolve()
    restore_root = restore_run_root.expanduser().resolve()
    source_plan_path = source_root / "execution_plan.json"
    restore_plan_path = restore_root / "execution_plan.json"
    if _read_object(source_plan_path, name="source persisted plan") != source:
        raise CampaignExecutionError("source persisted plan differs")
    if _read_object(restore_plan_path, name="restore persisted plan") != restore:
        raise CampaignExecutionError("restore persisted plan differs")
    preflight = _read_object(
        restore_root / "preflight_receipt.json", name="restore preflight receipt"
    )
    route_preflight = preflight.get("execution_route")
    if (
        preflight.get("schema_version") != "eiel_campaign_preflight_v1"
        or preflight.get("campaign_id") != restore["campaign_id"]
        or preflight.get("plan_sha256") != restore["plan_sha256"]
        or preflight.get("ready_for_execution") is not True
        or preflight.get("reasons") not in (None, [])
        or preflight.get("execution_authority_reasons") not in (None, [])
        or not isinstance(route_preflight, Mapping)
        or route_preflight.get("authority_state") != "route_qualified"
        or route_preflight.get("execution_route_sha256") != restore_route
        or route_preflight.get("artifact_conformance_exact") is not True
    ):
        raise CampaignExecutionError(
            "restore preflight does not authorize the exact continuation plan"
        )
    batch = source_batches[0]
    source_manifest = _load_batch_manifest(
        _batch_path(source_root, batch["batch_id"]),
        plan=source,
        provider=provider,
        batch=batch,
    )
    source_checkpoint = source_manifest.get("checkpoint")
    if source_manifest.get("status") != "completed" or not isinstance(
        source_checkpoint, Mapping
    ):
        raise CampaignExecutionError(
            "source stage is not completed with a durable checkpoint"
        )
    source_generation = source_checkpoint.get("generation")
    source_execution = source.get("execution")
    restore_execution = restore.get("execution")
    if (
        isinstance(source_execution, Mapping)
        and source_execution.get("execution_route_identity")
        == "tick_native_execution_route_v1"
    ) or (
        isinstance(restore_execution, Mapping)
        and restore_execution.get("execution_route_identity")
        == "tick_native_execution_route_v1"
    ):
        raise CampaignExecutionError(
            "adaptive generation continuation is legacy-route only"
        )
    if (
        not isinstance(source_execution, Mapping)
        or not isinstance(restore_execution, Mapping)
        or source_generation != source_execution.get("generations")
        or type(restore_execution.get("generations")) is not int
        or restore_execution["generations"] <= source_generation
    ):
        raise CampaignExecutionError("adaptive generation transition is invalid")
    source_request = provider.translate_batch(source, batch["batch_id"])
    restore_request = provider.translate_batch(restore, batch["batch_id"])
    prepare = getattr(provider, "prepare_wrc_checkpoint_adaptive_rebinding_v1", None)
    if not callable(prepare):
        raise CampaignExecutionError("provider lacks WRC adaptive checkpoint rebinding")
    rebound = _bind_checkpoint_route(
        prepare(
            source,
            restore,
            source_request,
            restore_request,
            dict(source_checkpoint),
        ),
        restore_route,
        require_existing=True,
    )
    adaptive = rebound.get("wrc_checkpoint_adaptive_rebinding_v1")
    if not isinstance(adaptive, Mapping):
        raise CampaignExecutionError(
            "provider returned no adaptive rebinding authority"
        )
    restore_manifest_path = _batch_path(restore_root, batch["batch_id"])
    restore_manifest = _load_batch_manifest(
        restore_manifest_path,
        plan=restore,
        provider=provider,
        batch=batch,
    )
    if restore_manifest.get("status") not in {"pending", "interrupted"}:
        raise CampaignExecutionError("restore stage manifest is not seedable")
    existing_checkpoint = restore_manifest.get("checkpoint")
    if existing_checkpoint is not None and existing_checkpoint != rebound:
        raise CampaignExecutionError(
            "restore stage already contains a different checkpoint"
        )
    restore_manifest["status"] = "interrupted"
    restore_manifest["checkpoint"] = rebound
    restore_manifest["metrics_rows"] = []
    restore_manifest["result"] = None
    restore_manifest["error"] = None
    restore_manifest["updated_at"] = _now()
    atomic_write_json(restore_manifest_path, restore_manifest)
    receipt = {
        "schema_version": "eiel_wrc_adaptive_campaign_transition_v1",
        "source_campaign_id": source["campaign_id"],
        "restore_campaign_id": restore["campaign_id"],
        "source_plan_sha256": source["plan_sha256"],
        "restore_plan_sha256": restore["plan_sha256"],
        "execution_route_sha256": restore_route,
        "qualification_key_sha256": restore["qualification_key_sha256"],
        "batch_id": batch["batch_id"],
        "cell_ids": list(batch["cell_ids"]),
        "source_generation": source_generation,
        "restore_generation": restore_execution["generations"],
        "adaptive_rebinding_receipt_sha256": adaptive.get("receipt_sha256"),
        "source_batch_manifest_path": str(_batch_path(source_root, batch["batch_id"])),
        "restore_batch_manifest_path": str(restore_manifest_path),
        "qualification_receipt_action": "unchanged",
        "status": "READY_TO_RESUME",
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    receipt_path = restore_root / "adaptive_transition_receipt.json"
    if (
        receipt_path.exists()
        and _read_object(receipt_path, name="adaptive transition receipt") != receipt
    ):
        raise CampaignExecutionError(
            "restore stage already contains a different transition receipt"
        )
    atomic_write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


__all__ = [
    "BatchExecutionContextV1",
    "CampaignExecutionError",
    "atomic_write_json",
    "execute_campaign",
    "prepare_adaptive_campaign_continuation_v1",
]
