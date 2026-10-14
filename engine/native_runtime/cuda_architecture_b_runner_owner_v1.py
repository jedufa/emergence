"""Run-level owner for the exact Patch P2 Architecture B CUDA envelope."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from engine.native_runtime.checkpoint_digests import (
    DigestLayer,
    checkpoint_digest_manifest,
    checkpoint_source_projection_digest_manifest,
)
from engine.native_runtime.cuda_architecture_b_production_adapter import (
    CudaArchitectureBProductionAdapter,
    CudaArchitectureBProductionDispatchError,
    CudaArchitectureBProductionExecutionV2,
    load_qualified_cuda_artifact,
)
from engine.native_runtime.cuda_architecture_b_routing import (
    CPP_CPU_BACKEND,
    CUDA_BACKEND,
    QUALIFIED_ARTIFACT_PATH,
    QUALIFIED_RECEIPT_PATH,
    QUALIFIED_REPORTING_RECEIPT_PATH,
    native_engine_pre_dispatch_rejection_receipt,
    qualified_cuda_pre_dispatch_rejection_receipt,
)
from engine.native_runtime.lifecycle_events import (
    typed_lifecycle_events_from_projection,
    typed_lifecycle_events_to_rows,
    validated_typed_lifecycle_event_cursor,
    write_typed_lifecycle_public_outputs,
)
from engine.native_runtime.scientific_throughput_efficiency_v1 import (
    build_cuda_scientific_throughput_efficiency_report,
    scientific_throughput_efficiency_enabled,
)
from engine.perf.cuda_architecture_b_direct_resident_settlement_v1 import (
    DirectResidentTerminalSettlementV1,
    settle_patch_p2_b1_direct_resident_prefix_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    build_prepared_direct_resident_trajectory_v1,
)
from engine.native_runtime.cuda_architecture_b_backend_factory import (
    derive_cuda_architecture_b_production_runner_envelope,
)
from perf.cxxbackbone10_resident_state_owner import projection_digest
from tools.visualization.living_lineage_canopy.python.default_run_export import (
    materialize_default_terminal_canopy,
)


OWNER_SCHEMA_VERSION: Final[str] = "eiel_cuda_architecture_b_runner_owner_v1"
OUTPUT_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_compact_output_manifest_v1"
)
CHECKPOINT_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_terminal_checkpoint_manifest_v1"
)
COMPACT_FIELD_ORDER: Final[tuple[str, ...]] = (
    "schema_version",
    "generation",
    "world",
    "alive",
    "births",
    "failed_births",
    "deaths",
    "event_cursor",
    "moved",
    "gathered",
)


class CudaArchitectureBRunnerOwnerError(RuntimeError):
    """The native CUDA owner could not settle one exact production run."""


@dataclass(frozen=True, slots=True)
class CudaArchitectureBRunnerRunV1:
    schema_version: str
    run_id: str
    terminal_status: str
    completed_generations: int
    output_manifest_path: str
    checkpoint_path: str
    production_route_receipt: dict[str, Any]
    terminal_result: dict[str, Any]
    python_generation_dispatches: int = 0
    hidden_fallback: bool = False
    runtime_jit: bool = False


def _fail(message: str) -> CudaArchitectureBRunnerOwnerError:
    return CudaArchitectureBRunnerOwnerError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _required_path(sim_cfg: Any, field: str, default: str) -> Path:
    raw = getattr(sim_cfg, field, "")
    if type(raw) is not str:
        raise _fail(f"sim_cfg.{field} must name a tracked qualification artifact")
    raw = raw.strip() or default
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise _fail(f"sim_cfg.{field} is missing: {path}")
    return path


def _stable_run_id(runner: Any) -> str:
    sim_cfg = runner.sim_cfg
    material = {
        "branch_id": str(getattr(sim_cfg, "branch_id", "default") or "default"),
        "campaign_id": str(getattr(sim_cfg, "campaign_id", "") or ""),
        "seed": int(getattr(sim_cfg, "seed", 0) or 0),
        "run_dir": str(Path(runner.run_dir).resolve()),
    }
    return (
        "archb-cuda-" + hashlib.sha256(_canonical_json_bytes(material)).hexdigest()[:24]
    )


def _validate_invocation(
    runner: Any,
    *,
    start_gen: int,
    end_gen: int,
    skip_init: bool,
) -> None:
    if (
        type(start_gen) is not int
        or type(end_gen) is not int
        or type(skip_init) is not bool
    ):
        raise _fail("runner invocation coordinates must use exact scalar types")
    source_checkpoint_projection = getattr(
        runner,
        "cuda_architecture_b_source_checkpoint_projection",
        None,
    )
    if source_checkpoint_projection is None:
        if start_gen != 0 or end_gen != 4 or skip_init:
            raise _fail(
                "the qualified CUDA runner owner requires one fresh generation-0 g4 run"
            )
    else:
        if not isinstance(source_checkpoint_projection, Mapping):
            raise _fail("CUDA runner owner resume source checkpoint must be a mapping")
        metadata = source_checkpoint_projection.get("metadata")
        if not isinstance(metadata, Mapping):
            raise _fail("CUDA runner owner resume source checkpoint lacks metadata")
        generation = metadata.get("generation")
        tick = metadata.get("tick")
        if generation != start_gen or tick != 0 or end_gen != start_gen + 4:
            raise _fail(
                "the qualified CUDA runner owner resume requires a tick-0 g4 "
                "source checkpoint boundary"
            )
        if not skip_init:
            raise _fail("CUDA runner owner resume requires skip_init=True")
    if end_gen <= start_gen:
        raise _fail(
            "CUDA runner owner invocation end_gen must be after start_gen"
        )
    mode = getattr(runner.sim_cfg, "cuda_architecture_b_request_mode", None)
    if mode not in {"auto", "explicit_cuda"}:
        raise _fail(
            "the bounded runner owner requires request_mode='auto' or 'explicit_cuda'"
        )
    unsupported_observers = [
        field
        for field in ("branch_observability_enable",)
        if bool(getattr(runner.sim_cfg, field, False))
    ]
    if unsupported_observers:
        raise _fail(
            "the compact CUDA runner owner cannot finalize Python population observers: "
            + ", ".join(unsupported_observers)
        )


def _compact_payload(rows: Sequence[Mapping[str, Any]]) -> tuple[bytes, bytes]:
    normalized = [dict(row) for row in rows]
    if not normalized:
        raise _fail("native compact summary rows changed their closed public schema")
    field_order = tuple(normalized[0])
    if field_order[: len(COMPACT_FIELD_ORDER)] != COMPACT_FIELD_ORDER:
        raise _fail("native compact summary rows changed their closed public schema")
    if any(tuple(row) != field_order for row in normalized):
        raise _fail("native compact summary rows changed their closed public schema")
    jsonl = b"".join(_canonical_json_bytes(row) for row in normalized)
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_order, lineterminator="\n")
    writer.writeheader()
    writer.writerows(normalized)
    return jsonl, stream.getvalue().encode("ascii")


def _persist_checkpoint(
    root: Path,
    settlement: DirectResidentTerminalSettlementV1,
) -> tuple[Path, dict[str, Any]]:
    checkpoint = settlement.terminal_checkpoint
    projection = checkpoint.projection
    if projection_digest(projection) != checkpoint.projection_digest:
        raise _fail("terminal checkpoint projection digest changed before persistence")
    source_manifest = checkpoint_source_projection_digest_manifest(projection)
    storage_manifest = checkpoint_digest_manifest(
        projection,
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    )
    target = root / "cuda_architecture_b_terminal_checkpoint_v1"
    temporary = root / f".{target.name}.tmp-{os.getpid()}"
    if target.exists():
        raise _fail(f"terminal checkpoint target already exists: {target}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        projection_path = temporary / "projection.pt"
        with projection_path.open("wb") as stream:
            torch.save(projection, stream)
            stream.flush()
            os.fsync(stream.fileno())
        restored = torch.load(projection_path, map_location="cpu", weights_only=False)
        if projection_digest(restored) != checkpoint.projection_digest:
            raise _fail("persisted terminal checkpoint failed semantic verification")
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "generation": settlement.terminal_generation,
            "tick": 0,
            "projection_path": "projection.pt",
            "projection_file_sha256": _sha256(projection_path),
            "projection_digest_sha256": checkpoint.projection_digest,
            "source_projection_digest_sha256": source_manifest["digest"],
            "canonical_checkpoint_storage_digest_sha256": storage_manifest["digest"],
            "device_to_host_operations": checkpoint.device_to_host_operations,
            "device_to_host_bytes": checkpoint.device_to_host_bytes,
            "device_synchronizations": checkpoint.device_synchronizations,
            "resume_supported": True,
            "observer_continuation_supported": False,
        }
        _atomic_write(temporary / "manifest.json", _canonical_json_bytes(manifest))
        _fsync_directory(temporary)
        os.replace(temporary, target)
        _fsync_directory(root)
        return target, manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _persist_outputs(
    runner: Any,
    *,
    settlement: DirectResidentTerminalSettlementV1,
    execution: CudaArchitectureBProductionExecutionV2,
) -> tuple[Path, Path, dict[str, Any]]:
    row_count = len(settlement.compact_summary_rows)
    terminal_generation = settlement.terminal_generation
    first_generation = (
        int(settlement.compact_summary_rows[0]["generation"]) if row_count else -1
    )
    expected_terminal_generation = first_generation + row_count
    if (
        execution.completed_generations != row_count
        or terminal_generation != expected_terminal_generation
        or int(settlement.compact_summary_rows[-1]["generation"])
        != terminal_generation - 1
    ):
        raise _fail("native execution, compact rows, and terminal generation disagree")
    root = Path(runner.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    compact_jsonl = root / "compact_generation_summaries.jsonl"
    compact_csv = root / "compact_generation_summaries.csv"
    jsonl_payload, csv_payload = _compact_payload(settlement.compact_summary_rows)
    _atomic_write(compact_jsonl, jsonl_payload)
    _atomic_write(compact_csv, csv_payload)

    events = typed_lifecycle_events_from_projection(
        settlement.terminal_checkpoint.projection
    )
    cursor = validated_typed_lifecycle_event_cursor(
        settlement.terminal_checkpoint.projection,
        events,
    )
    metrics_root = root / "metrics"
    lifecycle = write_typed_lifecycle_public_outputs(
        metrics_root / "typed_lifecycle_events.jsonl",
        metrics_root / "typed_lifecycle_events.csv",
        events,
        cursor=0,
    )
    if int(lifecycle["end_cursor"]) != cursor:
        raise _fail("typed lifecycle persistence cursor contradicted the checkpoint")
    if int(settlement.compact_summary_rows[-1]["event_cursor"]) != cursor:
        raise _fail("final compact event cursor contradicted the checkpoint")

    checkpoint_path, checkpoint_manifest = _persist_checkpoint(root, settlement)
    canopy_materialization = materialize_default_terminal_canopy(
        run_root=root,
        run_id=execution.run_id,
        terminal_generation=settlement.terminal_generation,
        projection=settlement.terminal_checkpoint.projection,
        source_files=(checkpoint_path / "projection.pt", checkpoint_path / "manifest.json"),
    )
    route_path = root / "cuda_architecture_b_route_receipt.json"
    control_path = root / "cuda_architecture_b_control_before_ack.json"
    _atomic_write(
        route_path,
        _canonical_json_bytes(execution.production_route_receipt),
    )
    _atomic_write(control_path, _canonical_json_bytes(execution.run_control_manifest))
    qacct_artifact = None
    if settlement.lifecycle_qacct_receipt is not None:
        qacct_path = root / "lifecycle_qacct_receipt.json"
        _atomic_write(
            qacct_path,
            _canonical_json_bytes(settlement.lifecycle_qacct_receipt),
        )
        qacct_artifact = {
            "path": qacct_path.name,
            "sha256": _sha256(qacct_path),
        }
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA_VERSION,
        "status": "durable_waiting_native_ack",
        "owner": "native_cuda_run_control_v2",
        "source_generation": first_generation,
        "completed_generations": row_count,
        "terminal_generation": settlement.terminal_generation,
        "compact_summary_rows": len(settlement.compact_summary_rows),
        "typed_lifecycle_event_rows": cursor,
        "terminal_checkpoint_count": 1,
        "artifacts": {
            "compact_jsonl": {
                "path": compact_jsonl.name,
                "sha256": _sha256(compact_jsonl),
            },
            "compact_csv": {
                "path": compact_csv.name,
                "sha256": _sha256(compact_csv),
            },
            "typed_lifecycle_jsonl": {
                "path": "metrics/typed_lifecycle_events.jsonl",
                "sha256": _sha256(metrics_root / "typed_lifecycle_events.jsonl"),
            },
            "typed_lifecycle_csv": {
                "path": "metrics/typed_lifecycle_events.csv",
                "sha256": _sha256(metrics_root / "typed_lifecycle_events.csv"),
            },
            "terminal_checkpoint": {
                "path": checkpoint_path.name,
                "manifest": checkpoint_manifest,
            },
            "default_canopy": canopy_materialization,
            "route_receipt": {
                "path": route_path.name,
                "sha256": _sha256(route_path),
            },
            "control_before_ack": {
                "path": control_path.name,
                "sha256": _sha256(control_path),
            },
        },
        "output_contract": {
            "schema": "eiel_patch_p2_production_compact_summary_v1",
            "legacy_metrics_csv_complete": False,
            "legacy_behaviors_csv_complete": False,
            "legacy_lineage_csv_complete": False,
            "resume_supported": False,
            "observer_continuation_supported": False,
            "reason": "qualified_compact_route_does_not_claim_legacy_full_projection",
        },
        "no_hidden_fallback": True,
        "runtime_jit": False,
        "python_tick_or_generation_ownership": False,
    }
    if scientific_throughput_efficiency_enabled():
        manifest["scientific_throughput_efficiency"] = (
            build_cuda_scientific_throughput_efficiency_report(
                direct_result=execution.direct_result,
                production_route_receipt=execution.production_route_receipt,
                completed_generations=execution.completed_generations,
                compact_summary_rows=len(settlement.compact_summary_rows),
            )
        )
    if qacct_artifact is not None:
        manifest["artifacts"]["lifecycle_qacct_receipt"] = qacct_artifact
    manifest_path = root / "cuda_architecture_b_output_manifest.json"
    _atomic_write(manifest_path, _canonical_json_bytes(manifest))
    return manifest_path, checkpoint_path, manifest


def _materialize_native_terminal_chronicle(
    runner: Any,
    settlement: DirectResidentTerminalSettlementV1,
) -> dict[str, Any] | None:
    """Build Chronicle once from terminal native outputs, never inside GPU generations."""

    recorder = getattr(runner, "scientific_chronicle", None)
    if recorder is None:
        return None
    try:
        lifecycle_rows: list[dict[str, Any]] = []
        if str(getattr(recorder.config, "profile", "notable")) in {
            "scientific_full",
            "debug_trace",
        }:
            events = typed_lifecycle_events_from_projection(
                settlement.terminal_checkpoint.projection
            )
            cursor = validated_typed_lifecycle_event_cursor(
                settlement.terminal_checkpoint.projection,
                events,
            )
            lifecycle_rows = typed_lifecycle_events_to_rows(
                events,
                start=0,
                stop=cursor,
            )
        recorder.observe_native_terminal_summaries(
            settlement.compact_summary_rows,
            typed_lifecycle_rows=lifecycle_rows,
        )
        recorder.finalize(status="complete")
        manifest_path = Path(runner.run_dir) / "chronicle" / "chronicle_manifest.json"
        if not recorder.finalized or not manifest_path.is_file():
            raise _fail("native terminal Chronicle did not finalize")
        return {
            "status": "complete",
            "source": "native_terminal_compact_generation_summaries",
            "generation_count": len(settlement.compact_summary_rows),
            "path": str(manifest_path.relative_to(Path(runner.run_dir))),
            "sha256": _sha256(manifest_path),
        }
    except Exception as exc:
        recorder.disabled_after_write_failure = True
        error = f"{type(exc).__name__}: {exc}"
        setattr(runner, "_scientific_chronicle_last_error", error)
        return {
            "status": "failed",
            "source": "native_terminal_compact_generation_summaries",
            "error": error,
        }


def _durable_boundary(
    manifest: Mapping[str, Any],
    *,
    settlement: DirectResidentTerminalSettlementV1,
    cancellation: bool,
) -> dict[str, Any]:
    boundary = dict(manifest.get("last_boundary") or {})
    frontier = int(boundary.get("trajectory_commit_frontier", -1))
    completed_generation = settlement.terminal_generation - 1
    if frontier != completed_generation:
        raise _fail("native control frontier contradicted the terminal checkpoint")
    sequence = 1
    boundary.update(
        {
            "snapshot_frontier": frontier,
            "output_ack_frontier": frontier,
            "snapshot_sequence": sequence,
            "snapshot_id": (
                f"{manifest['controller_id']}:owner:{manifest['owner_epoch']}:boundary:{sequence}"
            ),
            "snapshot_digest_sha256": (
                settlement.terminal_checkpoint.projection_digest
            ),
            "cancellation_checkpoint_requested": cancellation,
            "cancellation_checkpoint_persisted": cancellation,
        }
    )
    return boundary


def _publish_failure(control: Any, exc: BaseException) -> dict[str, Any]:
    try:
        manifest = dict(control.manifest())
        boundary = dict(manifest.get("last_boundary") or {})
        receipt = dict(
            control.publish_failure(
                "RUNNER_SETTLEMENT_FAILED",
                f"{type(exc).__name__}: {exc}",
                int(manifest.get("active_generation", -1)),
                int(boundary.get("trajectory_commit_frontier", 0)),
            )
        )
        after_failure = dict(control.manifest())
        terminal_raw = control.final_result()
        terminal = dict(terminal_raw) if isinstance(terminal_raw, Mapping) else None
        exact = (
            receipt.get("terminal_status") == "failed"
            and after_failure.get("status") == "failed"
            and after_failure.get("sealed") is True
            and terminal is not None
            and terminal.get("terminal_status") == "failed"
            and terminal.get("sealed") is True
        )
        return {
            "native_failure_settlement_exact": exact,
            "native_failure_receipt": receipt,
            "run_control_manifest": after_failure,
            "native_terminal_result": terminal,
            "failure_settlement_error": None,
        }
    except BaseException as settlement_exc:
        return {
            "native_failure_settlement_exact": False,
            "native_failure_receipt": None,
            "run_control_manifest": None,
            "native_terminal_result": None,
            "failure_settlement_error": {
                "error_type": type(settlement_exc).__name__,
                "error": str(settlement_exc),
            },
        }


def _persist_failure(
    runner: Any,
    *,
    phase: str,
    exc: BaseException,
    run_id: str,
    production_route_receipt: Mapping[str, Any],
    failure_state: Mapping[str, Any],
) -> tuple[Path, Path]:
    root = Path(runner.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    route_path = root / "cuda_architecture_b_route_receipt.json"
    _atomic_write(route_path, _canonical_json_bytes(dict(production_route_receipt)))

    failure_path = root / "cuda_architecture_b_run_failed.json"
    failure_payload = {
        "schema_version": OWNER_SCHEMA_VERSION,
        "status": "failed",
        "phase": phase,
        "run_id": run_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "native_failure_settlement_exact": bool(
            failure_state.get("native_failure_settlement_exact", False)
        ),
        "native_failure_receipt": failure_state.get("native_failure_receipt"),
        "native_terminal_result": failure_state.get("native_terminal_result"),
        "run_control_manifest": failure_state.get("run_control_manifest"),
        "failure_settlement_error": failure_state.get("failure_settlement_error"),
        "fallback_performed": False,
        "success_marker_allowed": False,
        "promotion_result_allowed": False,
    }
    _atomic_write(failure_path, _canonical_json_bytes(failure_payload))

    manifest_path = root / "cuda_architecture_b_output_manifest.json"
    prior: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="ascii"))
            if isinstance(loaded, Mapping):
                prior = dict(loaded)
        except (OSError, UnicodeError, json.JSONDecodeError):
            prior = {}
    artifacts = dict(prior.get("artifacts") or {})
    artifacts.update(
        {
            "route_receipt": {
                "path": route_path.name,
                "sha256": _sha256(route_path),
            },
            "failure_receipt": {
                "path": failure_path.name,
                "sha256": _sha256(failure_path),
            },
        }
    )
    output_manifest = {
        **prior,
        "schema_version": OUTPUT_MANIFEST_SCHEMA_VERSION,
        "status": "failed",
        "terminal_status": "failed",
        "owner": "native_cuda_run_control_v2",
        "run_id": run_id,
        "failure_phase": phase,
        "native_failure_settlement_exact": failure_payload[
            "native_failure_settlement_exact"
        ],
        "artifacts": artifacts,
        "no_hidden_fallback": True,
        "runtime_jit": False,
        "python_tick_or_generation_ownership": False,
        "success_marker_allowed": False,
        "promotion_result_allowed": False,
    }
    _atomic_write(manifest_path, _canonical_json_bytes(output_manifest))
    return failure_path, manifest_path


def request_cuda_architecture_b_runner_cancellation_v1(
    runner: Any,
    *,
    request_id: str,
    reason: str = "operator_requested",
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Request cancellation without taking ownership of a tick or generation."""

    adapter = getattr(runner, "_cuda_architecture_b_production_adapter", None)
    operation = getattr(adapter, "request_cancel", None)
    if not callable(operation):
        raise _fail("runner has no active CUDA Architecture B production adapter")
    receipt = operation(
        request_id=request_id,
        reason=reason,
        timeout_seconds=timeout_seconds,
    )
    if not isinstance(receipt, Mapping):
        raise _fail("native CUDA cancellation returned a non-mapping receipt")
    return dict(receipt)


def execute_cuda_architecture_b_runner_run_v1(
    runner: Any,
    *,
    start_gen: int,
    end_gen: int,
    skip_init: bool,
    artifact_loader: Callable[..., Any] = load_qualified_cuda_artifact,
    adapter_type: Callable[[Any], CudaArchitectureBProductionAdapter] = (
        CudaArchitectureBProductionAdapter
    ),
    source_builder: Callable[..., Any] = build_prepared_direct_resident_trajectory_v1,
    settlement_fn: Callable[..., DirectResidentTerminalSettlementV1] = (
        settle_patch_p2_b1_direct_resident_prefix_v2
    ),
) -> CudaArchitectureBRunnerRunV1 | None:
    """Execute and durably settle one exact native-owned g4 trajectory."""

    _validate_invocation(
        runner,
        start_gen=start_gen,
        end_gen=end_gen,
        skip_init=skip_init,
    )
    sim_cfg = runner.sim_cfg
    mode = str(sim_cfg.cuda_architecture_b_request_mode)
    try:
        artifact_arguments = (
            _required_path(
                sim_cfg,
                "cuda_architecture_b_artifact_path",
                QUALIFIED_ARTIFACT_PATH,
            ),
            _required_path(
                sim_cfg,
                "cuda_architecture_b_qualification_receipt_path",
                QUALIFIED_RECEIPT_PATH,
            ),
            _required_path(
                sim_cfg,
                "cuda_architecture_b_reporting_receipt_path",
                QUALIFIED_REPORTING_RECEIPT_PATH,
            ),
        )
        registry_path = str(
            getattr(sim_cfg, "cuda_architecture_b_qualification_registry_path", "")
            or ""
        )
        promotion_registry_path = str(
            getattr(sim_cfg, "cuda_architecture_b_promotion_registry_path", "")
            or ""
        )
        if registry_path or promotion_registry_path:
            artifact_kwargs: dict[str, str] = {}
            if registry_path:
                artifact_kwargs["registry_path"] = registry_path
            if promotion_registry_path:
                artifact_kwargs["promotion_registry_path"] = promotion_registry_path
            artifact = artifact_loader(*artifact_arguments, **artifact_kwargs)
        else:
            artifact = artifact_loader(*artifact_arguments)
    except BaseException as exc:
        if mode == "auto":
            runner._cuda_architecture_b_route_receipt = qualified_cuda_pre_dispatch_rejection_receipt(
                reason=(
                    f"qualified_cuda_artifact_unavailable:{type(exc).__name__}:{exc}"
                )
            )
            raise _fail(
                f"qualified automatic CUDA artifact failed closed before dispatch: {type(exc).__name__}: {exc}"
            ) from exc
        raise
    adapter = adapter_type(artifact)
    envelope = derive_cuda_architecture_b_production_runner_envelope(runner)
    route = adapter.resolve_once(
        request_mode=mode,
        allow_cpp_cpu_fallback=mode == "auto",
        workload=envelope.routing_workload(),
        allow_explicit_cuda_exactness_override=bool(
            getattr(
                sim_cfg,
                "cuda_architecture_b_allow_explicit_exactness_override",
                False,
            )
        ),
        performance_decision=str(
            getattr(sim_cfg, "cuda_architecture_b_performance_decision", "unmeasured")
        ),
        performance_decision_measured=bool(
            getattr(
                sim_cfg,
                "cuda_architecture_b_performance_decision_measured",
                False,
            )
        ),
    )
    runner._cuda_architecture_b_route_receipt = dict(route)
    if route.get("resolved_backend") == CPP_CPU_BACKEND and mode == "auto":
        reason = "cpp_cpu_owner_not_configured_for_live_phase1_route"
        runner._cuda_architecture_b_route_receipt = (
            native_engine_pre_dispatch_rejection_receipt(
                reason=reason,
                request_mode=mode,
            )
        )
        raise _fail(
            "automatic route selected C++ CPU before dispatch, but the live Phase 1 runner has no configured C++ owner; failed closed"
        )
    if route.get("resolved_backend") != CUDA_BACKEND:
        raise _fail(
            f"explicit CUDA route rejected before dispatch: {route.get('reason', 'unknown')}"
        )
    runner._cuda_architecture_b_production_adapter = adapter
    # RNG V2 owns drive noise by deterministic event address on the device.
    # Only the legacy RNG route may materialize a bounded future-value tape.
    sim_cfg.native_runtime_use_precomputed_drive_noise = (
        int(getattr(sim_cfg, "rng_semantics_version", 1) or 1) != 2
    )
    source_checkpoint_projection = getattr(
        runner,
        "cuda_architecture_b_source_checkpoint_projection",
        None,
    )
    prepared = source_builder(
        runner,
        qualified_cuda_module=artifact.module,
        cuda_device="cuda",
        source_checkpoint_projection=source_checkpoint_projection,
    )
    initial_projection = prepared.checkpoint_projection_inputs.get(
        "pre_generation_canonical_projection"
    )
    if not isinstance(initial_projection, Mapping) or not isinstance(
        initial_projection.get("metadata"), Mapping
    ):
        raise _fail("prepared source omitted canonical policy-mutation metadata")
    initial_metadata = initial_projection["metadata"]
    run_id = _stable_run_id(runner)
    try:
        execution = adapter.execute_prepared(
            run_id=run_id,
            physical_window_arguments=prepared.physical_window_arguments,
            policy_mutation_enable=initial_metadata.get(
                "policy_weight_mutation_enable"
            ),
            policy_mutation_mode=initial_metadata.get("policy_weight_mutation_mode"),
            policy_mutation_rate=initial_metadata.get("policy_weight_mutation_rate"),
            policy_mutation_std=initial_metadata.get("policy_weight_mutation_std"),
        )
    except CudaArchitectureBProductionDispatchError as exc:
        if (
            exc.production_route_receipt.get("contract_status")
            == "native_engine_pre_dispatch_rejected"
            and exc.production_route_receipt.get("dispatch_started") is False
        ):
            runner._cuda_architecture_b_route_receipt = dict(
                exc.production_route_receipt
            )
            raise _fail(
                f"native CUDA owner failed closed during pre-dispatch construction: {exc}"
            ) from exc
        failure_state = {
            "native_failure_settlement_exact": (
                exc.failure_settlement_error is None
                and exc.native_failure_receipt is not None
                and exc.run_control_manifest is not None
                and exc.run_control_manifest.get("status") == "failed"
                and exc.run_control_manifest.get("sealed") is True
                and exc.native_terminal_result is not None
                and exc.native_terminal_result.get("terminal_status") == "failed"
            ),
            "native_failure_receipt": exc.native_failure_receipt,
            "run_control_manifest": exc.run_control_manifest,
            "native_terminal_result": exc.native_terminal_result,
            "failure_settlement_error": exc.failure_settlement_error,
        }
        try:
            _persist_failure(
                runner,
                phase="dispatch",
                exc=exc,
                run_id=exc.run_id,
                production_route_receipt=exc.production_route_receipt,
                failure_state=failure_state,
            )
        except BaseException as persistence_exc:
            raise _fail(
                f"CUDA dispatch failed and its failure artifacts could not be persisted: {type(persistence_exc).__name__}: {persistence_exc}"
            ) from exc
        raise _fail(
            f"CUDA dispatch failed terminally; failure receipt and manifest were persisted without fallback: {exc}"
        ) from exc
    control = execution.run_control
    try:
        settlement = settlement_fn(
            prepared,
            direct_result=execution.direct_result,
            capability=artifact.capability,
        )
        manifest_path, checkpoint_path, output_manifest = _persist_outputs(
            runner,
            settlement=settlement,
            execution=execution,
        )
        before_ack = dict(control.manifest())
        cancelled = settlement.terminal_generation < 4
        boundary = _durable_boundary(
            before_ack,
            settlement=settlement,
            cancellation=cancelled,
        )
        if cancelled:
            acknowledgement = control.acknowledge_cancellation_checkpoint(
                before_ack["controller_id"],
                before_ack["owner_epoch"],
                boundary,
            )
            expected_terminal = "cancelled"
        else:
            acknowledgement = control.acknowledge_run_complete(
                before_ack["controller_id"],
                before_ack["owner_epoch"],
                boundary,
            )
            expected_terminal = "run_complete"
        terminal = control.final_result()
        after_ack = dict(control.manifest())
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("terminal_status") != expected_terminal
            or acknowledgement.get("terminal_status") != expected_terminal
            or after_ack.get("sealed") is not True
            or after_ack.get("awaiting_final_output_ack") is not False
            or terminal.get("success_marker_allowed") is not False
            or terminal.get("promotion_result_allowed") is not False
        ):
            raise _fail("native run control rejected durable terminal settlement")
        terminal_receipt_path = (
            Path(runner.run_dir) / "cuda_architecture_b_terminal_receipt.json"
        )
        terminal_receipt = {
            "schema_version": OWNER_SCHEMA_VERSION,
            "status": expected_terminal,
            "control_before_ack": before_ack,
            "acknowledged_boundary": boundary,
            "acknowledgement": dict(acknowledgement),
            "terminal_result": dict(terminal),
            "control_after_ack": after_ack,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
        }
        _atomic_write(terminal_receipt_path, _canonical_json_bytes(terminal_receipt))
        chronicle_materialization = _materialize_native_terminal_chronicle(
            runner,
            settlement,
        )
        completed_artifacts = {
            **output_manifest["artifacts"],
            "terminal_control_receipt": {
                "path": terminal_receipt_path.name,
                "sha256": _sha256(terminal_receipt_path),
            },
        }
        if (
            chronicle_materialization is not None
            and chronicle_materialization.get("status") == "complete"
        ):
            completed_artifacts["scientific_chronicle"] = {
                "path": chronicle_materialization["path"],
                "sha256": chronicle_materialization["sha256"],
            }
        chronicle_manifest_fields = (
            {"chronicle_materialization": chronicle_materialization}
            if chronicle_materialization is not None
            else {}
        )
        output_manifest = {
            **output_manifest,
            "status": "complete",
            "terminal_status": expected_terminal,
            "native_acknowledgement": dict(acknowledgement),
            "native_terminal_result": dict(terminal),
            **chronicle_manifest_fields,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
            "artifacts": completed_artifacts,
        }
        _atomic_write(manifest_path, _canonical_json_bytes(output_manifest))
        runner._current_gen = settlement.terminal_generation - 1
        try:
            runner.pop._eiel_current_generation_index = runner._current_gen
            runner.pop._current_generation = runner._current_gen
        except AttributeError:
            pass
        runner._cuda_architecture_b_execution = execution
        runner._cuda_architecture_b_settlement = settlement
        result = CudaArchitectureBRunnerRunV1(
            schema_version=OWNER_SCHEMA_VERSION,
            run_id=execution.run_id,
            terminal_status=expected_terminal,
            completed_generations=execution.completed_generations,
            output_manifest_path=str(manifest_path),
            checkpoint_path=str(checkpoint_path),
            production_route_receipt=dict(execution.production_route_receipt),
            terminal_result=dict(terminal),
        )
        runner._cuda_architecture_b_runner_result = result
        return result
    except BaseException as exc:
        failure_state = _publish_failure(control, exc)
        try:
            _persist_failure(
                runner,
                phase="settlement",
                exc=exc,
                run_id=execution.run_id,
                production_route_receipt=execution.production_route_receipt,
                failure_state=failure_state,
            )
        except BaseException as persistence_exc:
            raise _fail(
                f"CUDA settlement failed and its failure artifacts could not be persisted: {type(persistence_exc).__name__}: {persistence_exc}"
            ) from exc
        raise _fail(
            f"CUDA dispatch completed but durable settlement failed; fallback is prohibited: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "OUTPUT_MANIFEST_SCHEMA_VERSION",
    "OWNER_SCHEMA_VERSION",
    "CudaArchitectureBRunnerOwnerError",
    "CudaArchitectureBRunnerRunV1",
    "execute_cuda_architecture_b_runner_run_v1",
    "request_cuda_architecture_b_runner_cancellation_v1",
]
