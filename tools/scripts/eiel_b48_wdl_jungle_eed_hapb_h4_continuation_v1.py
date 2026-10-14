#!/usr/bin/env python3
"""Exact B48 HAPB fresh-owner continuation using an existing source set.

The source root is a prior successful HAPB direct probe.  This harness never
calls the source factory: the immutable generation-zero projections are
verified and reused in place.  It keeps only one evolved midpoint envelope,
whose shared host policy bank replaces the 48 duplicate MLP payloads.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import json
import os
import random
import sys
from time import perf_counter_ns
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.acs_checkpoint_staging import ACSCheckpointStaging
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.compiler import compile_campaign, load_campaign_spec
from engine.campaigns.orchestration import atomic_write_json
from engine.campaigns.tick_native_control import (
    TickNativeControlError,
    TickNativeControlV1,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
    HAPB_DIRECT_COMPONENT_CHECKPOINT_SCHEMA_V1,
    export_direct_resident_wdl_jungle_hapb_checkpoint_v1,
    export_hapb_policy_window_identity_checkpoint_v1,
    project_direct_resident_wdl_jungle_multicell_checkpoints_v1,
    restore_direct_resident_wdl_jungle_hapb_checkpoint_sources_v1,
    restore_direct_resident_wdl_jungle_hapb_component_checkpoint_v1,
    restore_hapb_policy_window_identity_checkpoint_v1,
)
from engine.perf.cxxbackbone10_resident_state_owner import projection_digest
from tools.scripts.eiel_b48_non_eed_wdl_jungle_allocation_diagnostic_v1 import (
    AllocationDiagnosticError,
    AllocationOnlyProvider,
    _diagnostic_descriptor,
    _diagnostic_plan,
    _load_exact_module,
    _sha256,
)
from tools.scripts.eiel_b48_non_eed_wdl_jungle_qualprep_runtime_v1 import (
    _bind_diagnostic_artifact,
    _bind_hapb_artifact_abi,
    _preflight_external_artifact_bindings_v1,
    _runtime_plan,
)


SCHEMA_VERSION = "eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1"
CELLS = 48


def _h4_tick_native_control_v1(
    source_spec: Mapping[str, Any],
    *,
    generation_turnover: bool,
    total_ticks: int | None = None,
    split_tick: int | None = None,
) -> TickNativeControlV1:
    execution = source_spec.get("execution")
    if not isinstance(execution, Mapping):
        raise AllocationDiagnosticError("HAPB H4 source spec lacks execution controls")
    if execution.get("execution_route_identity") != "tick_native_execution_route_v1":
        raise AllocationDiagnosticError("HAPB H4 requires the tick-native execution route")
    if any(
        execution.get(name) is not None
        for name in ("generations", "ticks_per_generation", "physical_window_generations")
    ):
        raise AllocationDiagnosticError(
            "HAPB H4 tick-native execution must not carry generation authority"
        )
    try:
        control = TickNativeControlV1.from_execution(execution)
    except (TickNativeControlError, TypeError, ValueError) as exc:
        raise AllocationDiagnosticError(
            f"HAPB H4 tick-native control is invalid: {exc}"
        ) from exc
    if total_ticks is not None and total_ticks != control.total_ticks:
        raise AllocationDiagnosticError("HAPB H4 total_ticks disagrees with native control")
    if split_tick is not None and split_tick not in control.resolved_checkpoint_ticks:
        raise AllocationDiagnosticError("HAPB H4 split_tick is not an exact checkpoint tick")
    del generation_turnover
    return control


def _h4_private_adapter_bounds_v1(
    *, start_tick: int, end_tick: int, chunk_ticks: int
) -> tuple[int, int]:
    if (
        type(start_tick) is not int
        or type(end_tick) is not int
        or type(chunk_ticks) is not int
        or start_tick < 0
        or end_tick <= start_tick
        or chunk_ticks <= 0
    ):
        raise AllocationDiagnosticError(
            "HAPB private adapter bounds require an ordered absolute tick span"
        )
    return (
        start_tick // chunk_ticks,
        (end_tick + chunk_ticks - 1) // chunk_ticks,
    )


def _h4_tick_dispatch_spans_v1(
    control: TickNativeControlV1,
    *,
    start_tick: int | None = None,
    end_tick: int | None = None,
    start_generation: int | None = None,
    end_generation: int | None = None,
    ticks_per_generation: int | None = None,
) -> tuple[int, ...]:
    if start_tick is None or end_tick is None:
        if (
            start_generation is None
            or end_generation is None
            or ticks_per_generation is None
        ):
            raise AllocationDiagnosticError("HAPB H4 span requires absolute tick bounds")
        # Compatibility-only adapter for the current provider ABI.  The active
        # route and its receipts remain bound to the native tick control.
        start_tick = start_generation * ticks_per_generation
        end_tick = end_generation * ticks_per_generation
    if start_tick < 0 or end_tick > control.total_ticks or end_tick <= start_tick:
        raise AllocationDiagnosticError("HAPB H4 tick cursor is outside its control horizon")
    spans: list[int] = []
    cursor = start_tick
    while cursor < end_tick:
        next_tick = min(end_tick, cursor + control.native_dispatch_window_ticks)
        spans.append(next_tick - cursor)
        cursor = next_tick
    return tuple(spans)

_FRAG_IDENTITY_COUNT_BYTE_FIELDS_V1: Final[tuple[str, ...]] = (
    "policy_birth_identity_ensure_calls_total",
    "policy_birth_identity_materialization_calls_total",
    "policy_birth_identity_physical_refill_calls_total",
    "policy_birth_identity_resident_suffix_reuse_calls_total",
    "policy_birth_identity_device_to_host_operations_lower_bound",
    "policy_birth_identity_host_to_device_operations_lower_bound",
    "policy_birth_identity_device_to_host_bytes_lower_bound",
    "policy_birth_identity_host_to_device_bytes_lower_bound",
)
_FRAG_IDENTITY_TIME_FIELDS_V1: Final[tuple[str, ...]] = (
    "policy_birth_identity_host_wall_time_ns",
    "policy_birth_identity_device_to_host_time_ns",
    "policy_birth_identity_host_to_device_time_ns",
)
_FRAG_IDENTITY_ENVIRONMENT_FIELDS_V1: Final[tuple[str, ...]] = (
    "EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE",
    "EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL",
    "EIEL_HAPB_FRAG_IDENTITY_TRACE",
)


def _frag_identity_runtime_controls_v1(
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the exact native environment switches and their effective arm."""

    raw = {
        name: environ.get(name) for name in _FRAG_IDENTITY_ENVIRONMENT_FIELDS_V1
    }
    enable_requested = (
        raw["EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE"] == "1"
    )
    force_physical_requested = (
        raw["EIEL_HAPB_FRAG_A_FORCE_PHYSICAL_POLICY_REFILL"] == "1"
    )
    trace_enabled = raw["EIEL_HAPB_FRAG_IDENTITY_TRACE"] == "1"
    effective_reuse_enabled = enable_requested and not force_physical_requested
    active_arm = (
        "force_physical"
        if force_physical_requested
        else "identity_bound_reuse"
        if effective_reuse_enabled
        else "legacy_default"
    )
    return {
        "schema_version": "eiel_hapb_frag_identity_runtime_controls_v1",
        "environment": raw,
        "enable_policy_window_reuse_requested": enable_requested,
        "force_physical_policy_refill_requested": force_physical_requested,
        "identity_trace_enabled": trace_enabled,
        "effective_policy_window_reuse_enabled": effective_reuse_enabled,
        "active_arm": active_arm,
    }


def _option_d_m0_telemetry_enabled_v1(
    manifest_payload: Mapping[str, Any],
) -> bool:
    build = manifest_payload.get("option_d_m0_telemetry_build")
    if not isinstance(build, Mapping):
        raise AllocationDiagnosticError(
            "HAPB artifact manifest lacks its Option D M0 build identity"
        )
    enabled = build.get("option_d_m0_telemetry_enabled")
    if type(enabled) is not bool:
        raise AllocationDiagnosticError(
            "HAPB artifact manifest has a malformed Option D M0 build identity"
        )
    return enabled


def _frag_identity_accounting_v1(
    result: Mapping[str, Any],
    *,
    runtime_controls: Mapping[str, Any],
    option_d_m0_telemetry_enabled: bool,
) -> dict[str, Any]:
    """Retain traffic accounting while refusing to promote disabled timers."""

    identity_requested = (
        runtime_controls.get("enable_policy_window_reuse_requested") is True
    )
    native_m0_marker = result.get("option_d_m0_telemetry_enabled", False)
    if type(native_m0_marker) is not bool:
        raise AllocationDiagnosticError(
            "FRAG identity result has a malformed Option D M0 marker"
        )
    if native_m0_marker is not option_d_m0_telemetry_enabled:
        raise AllocationDiagnosticError(
            "FRAG identity result contradicts the artifact Option D M0 identity"
        )
    accounting: dict[str, Any] = {
        "schema_version": "eiel_hapb_frag_identity_accounting_v2",
        "identity_reuse_requested": identity_requested,
        "runtime_controls": dict(runtime_controls),
        "execution_count": 1,
    }
    for field in _FRAG_IDENTITY_COUNT_BYTE_FIELDS_V1:
        value = result.get(field)
        if identity_requested and (type(value) is not int or value < 0):
            raise AllocationDiagnosticError(
                f"FRAG identity result omitted exact accounting field {field}"
            )
        accounting[field] = value
    for field in _FRAG_IDENTITY_TIME_FIELDS_V1:
        raw_value = result.get(field)
        if identity_requested and (
            type(raw_value) is not int or raw_value < 0
        ):
            raise AllocationDiagnosticError(
                f"FRAG identity result omitted exact timing field {field}"
            )
        accounting[field] = (
            raw_value if option_d_m0_telemetry_enabled else None
        )
    accounting["identity_subtime_telemetry"] = {
        "schema_version": "eiel_hapb_frag_identity_subtime_telemetry_v1",
        "measured": option_d_m0_telemetry_enabled,
        "option_d_m0_telemetry_enabled": option_d_m0_telemetry_enabled,
        "authority": "artifact_manifest.option_d_m0_telemetry_build",
        "unavailable_reason": (
            None
            if option_d_m0_telemetry_enabled
            else "artifact_compiled_with_option_d_m0_telemetry_disabled"
        ),
    }
    return accounting


def _aggregate_frag_identity_accounting_v1(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise AllocationDiagnosticError("FRAG identity accounting rows are absent")
    controls = rows[0].get("runtime_controls")
    telemetry = rows[0].get("identity_subtime_telemetry")
    if not isinstance(controls, Mapping) or not isinstance(telemetry, Mapping):
        raise AllocationDiagnosticError("FRAG identity accounting authority is absent")
    if any(row.get("runtime_controls") != controls for row in rows):
        raise AllocationDiagnosticError("FRAG identity controls changed between entries")
    measured = telemetry.get("measured") is True
    if any(
        not isinstance(row.get("identity_subtime_telemetry"), Mapping)
        or row["identity_subtime_telemetry"].get("measured") is not measured
        for row in rows
    ):
        raise AllocationDiagnosticError("FRAG identity timer authority changed between entries")
    aggregated: dict[str, Any] = {
        "schema_version": "eiel_hapb_frag_identity_accounting_v2",
        "identity_reuse_requested": rows[0].get("identity_reuse_requested"),
        "runtime_controls": dict(controls),
        "execution_count": len(rows),
        "identity_subtime_telemetry": dict(telemetry),
    }
    for field in _FRAG_IDENTITY_COUNT_BYTE_FIELDS_V1:
        values = [row.get(field) for row in rows]
        if any(type(value) is not int or value < 0 for value in values):
            raise AllocationDiagnosticError(
                f"FRAG identity entry accounting is invalid for {field}"
            )
        aggregated[field] = sum(values)
    for field in _FRAG_IDENTITY_TIME_FIELDS_V1:
        values = [row.get(field) for row in rows]
        if measured:
            if any(type(value) is not int or value < 0 for value in values):
                raise AllocationDiagnosticError(
                    f"FRAG identity entry timing is invalid for {field}"
                )
            aggregated[field] = sum(values)
        else:
            if any(value is not None for value in values):
                raise AllocationDiagnosticError(
                    f"disabled FRAG identity timing was promoted for {field}"
                )
            aggregated[field] = None
    return aggregated


def _identity_entry_receipts_v1(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    receipts = result.get("policy_birth_identity_entry_ensure_receipts")
    if receipts is None:
        receipt = result.get("policy_birth_identity_entry_ensure_receipt")
        receipts = [receipt] if isinstance(receipt, Mapping) and receipt else []
    if not isinstance(receipts, list) or any(
        not isinstance(receipt, Mapping) for receipt in receipts
    ):
        raise AllocationDiagnosticError("FRAG identity trace entry receipts are malformed")
    return list(receipts)


def _same_owner_entry_result_v1(
    result: Mapping[str, Any],
    *,
    ordinal: int,
    identity_accounting: Mapping[str, Any],
    native_execute_wall_ns: int,
    process_rng_after_entry: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = _identity_entry_receipts_v1(result)
    if len(receipts) != 1:
        raise AllocationDiagnosticError(
            "same-owner diagnostic requires one exact entry receipt per span"
        )
    trace = receipts[0].get("frag_identity_trace")
    if not isinstance(trace, list) or not trace or any(
        not isinstance(row, Mapping) for row in trace
    ):
        raise AllocationDiagnosticError(
            "same-owner diagnostic requires exact entry trace markers"
        )
    raw_markers = {row.get("checkpoint_reentry_marker") for row in trace}
    if any(type(marker) is not str for marker in raw_markers):
        raise AllocationDiagnosticError("same-owner entry marker is malformed")
    markers = sorted(raw_markers)
    segment_receipts = result.get("policy_birth_segment_refill_receipts", [])
    outer_receipts = result.get("policy_refill_receipts", [])
    resume_receipt = result.get("resume_policy_refill_receipt", {})
    if not isinstance(segment_receipts, list) or not isinstance(
        outer_receipts, list
    ):
        raise AllocationDiagnosticError("same-owner materialization receipts are malformed")
    if not isinstance(resume_receipt, Mapping):
        raise AllocationDiagnosticError("same-owner resume receipt is malformed")
    return {
        "schema_version": f"{SCHEMA_VERSION}_same_owner_entry_result_v1",
        "entry_ordinal": ordinal,
        "tick_span_start": result.get("tick_span_start"),
        "tick_span_end": result.get("tick_span_end"),
        "tick_span_count": result.get("tick_span_count"),
        "checkpoint_reentry_markers": markers,
        "frag_identity_result": receipts[0].get("frag_identity_result"),
        "physical_refill_performed": receipts[0].get("physical_refill_performed"),
        "resident_window_reused": receipts[0].get("resident_window_reused"),
        "identity_entry_ensure_receipts": receipts,
        "segment_refill_receipts": list(segment_receipts),
        "outer_policy_materialization_receipts": list(outer_receipts),
        "resume_policy_materialization_receipt": dict(resume_receipt),
        "frag_identity_accounting": dict(identity_accounting),
        "native_execute_wall_ns": native_execute_wall_ns,
        "process_rng_after_entry": dict(process_rng_after_entry),
    }


def _verify_same_owner_reentry_v1(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(entries) != 2:
        raise AllocationDiagnosticError("same-owner diagnostic requires exactly two entries")
    expected = (
        (1, 0, 1, "fresh_allocator_owner", True, False, 1, 0),
        (2, 1, 2, "same_owner_route_reentry", False, True, 0, 1),
    )
    for entry, (
        ordinal,
        start,
        end,
        marker,
        physical,
        reused,
        physical_calls,
        reuse_calls,
    ) in zip(entries, expected, strict=True):
        accounting = entry.get("frag_identity_accounting")
        if not isinstance(accounting, Mapping):
            raise AllocationDiagnosticError("same-owner entry accounting is absent")
        physical_call_count = accounting.get(
            "policy_birth_identity_physical_refill_calls_total"
        )
        reuse_call_count = accounting.get(
            "policy_birth_identity_resident_suffix_reuse_calls_total"
        )
        observed = (
            entry.get("entry_ordinal"),
            entry.get("tick_span_start"),
            entry.get("tick_span_end"),
            entry.get("checkpoint_reentry_markers"),
            entry.get("physical_refill_performed"),
            entry.get("resident_window_reused"),
        )
        expected_observed = (
            ordinal,
            start,
            end,
            [marker],
            physical,
            reused,
        )
        counts_exact = (
            physical_call_count == physical_calls
            and type(reuse_call_count) is int
            and reuse_call_count >= reuse_calls
        )
        if observed != expected_observed or not counts_exact:
            raise AllocationDiagnosticError(
                "same-owner entry "
                f"{ordinal} proof differs: {observed!r}, "
                f"physical_calls={physical_call_count!r}, "
                f"reuse_calls={reuse_call_count!r}"
            )
    return {
        "schema_version": f"{SCHEMA_VERSION}_same_owner_reentry_proof_v1",
        "entry_markers_exact": True,
        "entry_tick_coordinates_exact": True,
        "first_entry_hydration_count": 1,
        "second_entry_hydration_count": 0,
        "second_entry_same_owner_suffix_reuse_count_lower_bound": 1,
        "duplicate_first_entry_hydration": False,
        "entry_results": [
            {
                "entry_ordinal": entry["entry_ordinal"],
                "checkpoint_reentry_markers": entry[
                    "checkpoint_reentry_markers"
                ],
                "frag_identity_result": entry["frag_identity_result"],
                "physical_refill_performed": entry["physical_refill_performed"],
                "resident_window_reused": entry["resident_window_reused"],
            }
            for entry in entries
        ],
    }


def _rebind_same_owner_tick_source_v1(
    prepared: Any,
    projections: Sequence[Mapping[str, Any]],
    *,
    expected_tick: int,
) -> Any:
    """Advance only the checkpoint view while sharing every live owner/tensor."""

    if not projections:
        raise AllocationDiagnosticError("same-owner intermediate projection is absent")
    for projection in projections:
        metadata = projection.get("metadata") if isinstance(projection, Mapping) else None
        if not isinstance(metadata, Mapping) or metadata.get("tick") != expected_tick:
            raise AllocationDiagnosticError(
                "same-owner intermediate projection has the wrong tick cursor"
            )
    wdl = getattr(prepared, "wdl", None)
    patch = getattr(wdl, "patch", None)
    source = getattr(patch, "source", None)
    checkpoint_inputs = getattr(patch, "checkpoint_projection_inputs", None)
    if not isinstance(checkpoint_inputs, Mapping) or source is None:
        raise AllocationDiagnosticError("same-owner prepared checkpoint view is absent")
    rebound_inputs = dict(checkpoint_inputs)
    rebound_inputs["pre_generation_canonical_projection"] = projections[0]
    rebound_inputs["pre_generation_canonical_projections"] = tuple(projections)
    rebound_source = replace(
        source,
        checkpoint_projection_inputs=rebound_inputs,
    )
    rebound_patch = replace(
        patch,
        source=rebound_source,
        checkpoint_projection_inputs=rebound_inputs,
    )
    rebound_wdl = replace(
        wdl,
        patch=rebound_patch,
        initial_projection=projections[0],
        source_projections=tuple(projections),
    )
    rebound = replace(prepared, wdl=rebound_wdl)
    if (
        rebound.policy_window_identity_owner
        is not prepared.policy_window_identity_owner
    ):
        raise AllocationDiagnosticError("same-owner identity capsule was substituted")
    return rebound


def _head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_reused_sources(
    source_root: Path, *, cell_ids: Sequence[str]
) -> tuple[Mapping[str, Any], ...]:
    """Load only receipt-bound source files under the declared source root."""

    import torch

    receipt_path = source_root / "source_factory_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != len(cell_ids):
        raise AllocationDiagnosticError("reused HAPB source receipt is incomplete")
    paths: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AllocationDiagnosticError("reused HAPB source receipt row is malformed")
        cell_id, raw_path, digest = (
            row.get("cell_id"), row.get("projection_path"), row.get("projection_sha256")
        )
        if not isinstance(cell_id, str) or not isinstance(raw_path, str) or not isinstance(digest, str):
            raise AllocationDiagnosticError("reused HAPB source receipt lacks identity")
        path = Path(raw_path).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise AllocationDiagnosticError("reused HAPB source escapes its immutable root") from exc
        if not path.is_file() or _sha256(path) != digest:
            raise AllocationDiagnosticError("reused HAPB source projection digest differs")
        if cell_id in paths:
            raise AllocationDiagnosticError("reused HAPB source receipt repeats a cell")
        paths[cell_id] = path
    if tuple(paths) != tuple(cell_ids):
        raise AllocationDiagnosticError(
            "reused HAPB source cell order differs from the requested physical extent"
        )
    loaded = tuple(torch.load(paths[cell_id], map_location="cpu", weights_only=False) for cell_id in cell_ids)
    if any(not isinstance(value, Mapping) for value in loaded):
        raise AllocationDiagnosticError("reused HAPB source projection is malformed")
    return loaded


def _terminal_digests(projections: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if not projections:
        raise AllocationDiagnosticError("HAPB continuation projection count differs from physical B")
    return tuple(projection_digest(projection) for projection in projections)


def _detach_host_snapshot_v1(value: Any) -> Any:
    """Copy one envelope onto host memory before ACS may touch it.

    ACS admits only an immutable detached snapshot.  Saving live CUDA tensors
    from the writer thread is not that contract and is not bit-stable.
    """

    detach = getattr(value, "detach", None)
    to_cpu = getattr(value, "cpu", None)
    contiguous = getattr(value, "contiguous", None)
    if callable(detach) and callable(to_cpu) and callable(contiguous):
        # Tensor.cpu is a bound method; do not call it as cpu(tensor).
        return detach().cpu().contiguous()
    if isinstance(value, Mapping):
        return {key: _detach_host_snapshot_v1(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_host_snapshot_v1(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_host_snapshot_v1(item) for item in value)
    return value


def persist_hapb_midpoint_via_acs_v1(
    envelope: Any,
    checkpoint_path: Path,
    *,
    slots: int,
    compute_frontier: int,
    save: Any,
) -> dict[str, Any]:
    """Persist one HAPB midpoint through the qualified two-slot ACS writer.

    The restore contract still consumes the same on-disk envelope.  ACS owns
    the write, FIFO consumption, and durable-frontier acknowledgement.
    """

    if type(slots) is not int or slots not in {1, 2, 3}:
        raise AllocationDiagnosticError("ACS checkpoint slots must be 1, 2, or 3")
    if type(compute_frontier) is not int or compute_frontier < 0:
        raise AllocationDiagnosticError("ACS compute frontier must be non-negative")
    host_snapshot = _detach_host_snapshot_v1(envelope)
    staging = ACSCheckpointStaging(slots=slots)

    def persist() -> dict[str, Any]:
        save(host_snapshot, checkpoint_path)
        return {
            "path": str(checkpoint_path),
            "bytes": int(checkpoint_path.stat().st_size),
            "sha256": _sha256(checkpoint_path),
            "acs_checkpoint_slots": slots,
            "acs_compute_frontier": compute_frontier,
            "host_snapshot_detached": True,
        }

    try:
        token = staging.submit(
            compute_frontier=compute_frontier,
            captured_bytes=0,
            persist=persist,
        )
        result = staging.result(token, producer_backpressure=True)
        staging.acknowledge_durable(token)
    except BaseException:
        staging.abort()
        raise
    staging.close()
    payload = dict(result)
    payload["acs_telemetry"] = staging.telemetry()
    return payload


def _process_rng_trace() -> dict[str, Any]:
    """Return payload-free identities for process-global RNGs.

    HAPB checkpoints each simulation-owned stream.  These identities diagnose
    whether an unowned process generator can still influence a source-bound
    native segment; they never retain RNG state bytes.
    """

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    numpy_digest = hashlib.sha256()
    numpy_digest.update(str(numpy_state[0]).encode("utf-8"))
    numpy_digest.update(numpy_state[1].tobytes())
    numpy_digest.update(str(tuple(numpy_state[2:])).encode("utf-8"))

    trace: dict[str, Any] = {
        "python_global_rng_sha256": hashlib.sha256(
            repr(random.getstate()).encode("utf-8")
        ).hexdigest(),
        "numpy_global_rng_sha256": numpy_digest.hexdigest(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_cpu_rng_sha256": hashlib.sha256(
            torch.get_rng_state().detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
    }
    if torch.cuda.is_available():
        trace["torch_cuda_rng_sha256"] = [
            hashlib.sha256(
                state.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ]
    return trace


def _projection_value_digest(value: Any) -> str:
    """Hash one projection value without materializing a second projection."""

    h = hashlib.sha256()
    if hasattr(value, "detach"):
        cpu = value.detach().to("cpu").contiguous()
        array = cpu.numpy()
        h.update(b"tensor")
        h.update(str(array.dtype).encode("utf-8"))
        h.update(json.dumps([int(item) for item in array.shape]).encode("utf-8"))
        h.update(array.tobytes())
    elif isinstance(value, Mapping):
        h.update(b"mapping")
        for key in sorted(str(item) for item in value):
            h.update(key.encode("utf-8"))
            h.update(_projection_value_digest(value[key]).encode("utf-8"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        h.update(b"sequence")
        for item in value:
            h.update(_projection_value_digest(item).encode("utf-8"))
    else:
        h.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


_IGNORED_TRAJECTORY_METADATA: Final[frozenset[str]] = frozenset({
        "import_digest",
        "native_policy_cache_served",
        "native_next_segment_index",
        "native_chain_sha256",
    })


def _projection_signatures(
    projections: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Retain only semantic field hashes before releasing B48 projections."""

    signatures: list[dict[str, Any]] = []
    for projection in projections:
        metadata = dict(projection.get("metadata") or {})
        tensors = dict(projection.get("tensors") or {})
        signatures.append({
            "projection_digest": projection_digest(projection),
            "metadata": {
                key: _projection_value_digest(value)
                for key, value in metadata.items()
                if key not in _IGNORED_TRAJECTORY_METADATA
            },
            "tensors": {
                key: _projection_value_digest(value)
                for key, value in tensors.items()
            },
            "top_level": {
                key: _projection_value_digest(value)
                for key, value in projection.items()
                if key not in {"metadata", "tensors"}
            },
        })
    return tuple(signatures)


def _projection_difference_summary(
    reference_signatures: Sequence[Mapping[str, Any]],
    resumed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep a compact semantic mismatch map; never persist full B48 rows."""

    resumed_signatures = _projection_signatures(resumed)
    mismatches: list[dict[str, Any]] = []
    for cell, (left, right) in enumerate(
        zip(reference_signatures, resumed_signatures, strict=True)
    ):
        if left["projection_digest"] == right["projection_digest"]:
            continue
        metadata_keys = sorted(
            key
            for key in set(left["metadata"]) | set(right["metadata"])
            if left["metadata"].get(key) != right["metadata"].get(key)
        )
        tensor_names = sorted(
            name
            for name in set(left["tensors"]) | set(right["tensors"])
            if left["tensors"].get(name) != right["tensors"].get(name)
        )
        top_level_keys = sorted(
            key
            for key in set(left["top_level"]) | set(right["top_level"])
            if left["top_level"].get(key) != right["top_level"].get(key)
        )
        mismatches.append({
            "cell_index": cell,
            "reference_digest": left["projection_digest"],
            "resumed_digest": right["projection_digest"],
            "metadata_keys": metadata_keys,
            "tensor_names": tensor_names,
            "top_level_keys": top_level_keys,
        })
    return {
        "schema_version": f"{SCHEMA_VERSION}_difference_v1",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _checkpoint_restore_difference_summary(
    checkpoint: Mapping[str, Any],
    restored_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare transport reconstruction without retaining another checkpoint."""

    projections = checkpoint.get("checkpoint_projections")
    if not isinstance(projections, list) or not projections:
        raise AllocationDiagnosticError("HAPB checkpoint projections are malformed")
    if len(restored_sources) != len(projections):
        raise AllocationDiagnosticError("HAPB checkpoint restore inputs are malformed")
    transport = frozenset({
        "hapb_policy_checkpoint_sha256",
        "hapb_policy_checkpoint_schema_version",
        "hapb_policy_restore_pending_cache_rehydrate",
    })
    mismatches: list[dict[str, Any]] = []
    for cell, (raw, restored) in enumerate(zip(projections, restored_sources, strict=True)):
        if not isinstance(raw, Mapping):
            raise AllocationDiagnosticError(f"HAPB checkpoint projection {cell} is malformed")
        expected_metadata = {
            key: value for key, value in dict(raw.get("metadata") or {}).items()
            if key not in transport
        }
        actual_metadata = {
            key: value for key, value in dict(restored.get("metadata") or {}).items()
            if key not in transport
        }
        actual_tensors = dict(restored.get("tensors") or {})
        restored_mlp = actual_tensors.get("mlp_pack")
        if not hasattr(restored_mlp, "detach"):
            raise AllocationDiagnosticError(
                f"HAPB checkpoint restore {cell} lacks its canonical policy view"
            )
        # The serialized bank is a bytes payload, validated during restore.
        # Reuse its admitted restored view here rather than allocating a second
        # four-gigabyte canonical tensor solely for compact comparison.
        expected_tensors = {**dict(raw.get("tensors") or {}), "mlp_pack": restored_mlp}
        metadata_keys = sorted(
            key for key in set(expected_metadata) | set(actual_metadata)
            if _projection_value_digest(expected_metadata.get(key))
            != _projection_value_digest(actual_metadata.get(key))
        )
        tensor_names = sorted(
            key for key in set(expected_tensors) | set(actual_tensors)
            if _projection_value_digest(expected_tensors.get(key))
            != _projection_value_digest(actual_tensors.get(key))
        )
        top_level_keys = sorted(
            key for key in set(raw) | set(restored)
            if key not in {"metadata", "tensors"}
            and _projection_value_digest(raw.get(key))
            != _projection_value_digest(restored.get(key))
        )
        if metadata_keys or tensor_names or top_level_keys:
            mismatches.append({
                "cell_index": cell,
                "metadata_keys": metadata_keys,
                "tensor_names": tensor_names,
                "top_level_keys": top_level_keys,
            })
    return {
        "schema_version": f"{SCHEMA_VERSION}_restore_boundary_difference_v1",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _prepared_restore_difference_summary(
    expected_sources: Sequence[Mapping[str, Any]],
    prepared_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the transport-normalized G3 input after allocator preparation."""

    if not expected_sources or len(expected_sources) != len(prepared_sources):
        raise AllocationDiagnosticError("HAPB prepared restore source count is malformed")
    transport = frozenset({
        "hapb_policy_checkpoint_sha256",
        "hapb_policy_checkpoint_schema_version",
        "hapb_policy_restore_pending_cache_rehydrate",
    })
    mismatches: list[dict[str, Any]] = []
    for cell, (expected, prepared) in enumerate(zip(expected_sources, prepared_sources, strict=True)):
        expected_metadata = {
            key: value for key, value in dict(expected.get("metadata") or {}).items()
            if key not in transport
        }
        prepared_metadata = {
            key: value for key, value in dict(prepared.get("metadata") or {}).items()
            if key not in transport
        }
        expected_tensors, prepared_tensors = (
            dict(expected.get("tensors") or {}),
            dict(prepared.get("tensors") or {}),
        )
        metadata_keys = sorted(
            key for key in set(expected_metadata) | set(prepared_metadata)
            if _projection_value_digest(expected_metadata.get(key))
            != _projection_value_digest(prepared_metadata.get(key))
        )
        tensor_names = sorted(
            key for key in set(expected_tensors) | set(prepared_tensors)
            if _projection_value_digest(expected_tensors.get(key))
            != _projection_value_digest(prepared_tensors.get(key))
        )
        if metadata_keys or tensor_names:
            mismatches.append({
                "cell_index": cell,
                "metadata_keys": metadata_keys,
                "tensor_names": tensor_names,
            })
    return {
        "schema_version": f"{SCHEMA_VERSION}_prepared_restore_difference_v1",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--compare-reference-midpoint", action="store_true")
    parser.add_argument(
        "--disable-content-alias",
        action="store_true",
        help=(
            "run the bounded pre-rescue baseline with the same source and "
            "native artifact; this is a diagnostic comparison only"
        ),
    )
    parser.add_argument(
        "--generation-turnover",
        action="store_true",
        help=(
            "run the legacy synchronized lifecycle ABI (0) with host-authoritative "
            "HAPB turnover transactions"
        ),
    )
    parser.add_argument(
        "--repeat-diagnostic-end-generation",
        type=int,
        help=(
            "run two independent fresh G0-to-G3 segments and "
            "fail closed with a compact terminal difference map"
        ),
    )
    parser.add_argument(
        "--repeat-diagnostic-end-tick",
        type=int,
        help=(
            "run two independent fresh HAPB owners through one exact partial "
            "lifecycle span and compare their per-cell terminal projections"
        ),
    )
    parser.add_argument(
        "--single-diagnostic-end-tick",
        type=int,
        help=(
            "run one fresh HAPB owner through a bounded partial lifecycle span "
            "for performance measurement without a determinism comparison"
        ),
    )
    parser.add_argument(
        "--single-diagnostic-end-generation",
        type=int,
        help=(
            "run one fresh HAPB owner through G0-to-G3 using the official "
            "generation execute path; diagnostic profiling only"
        ),
    )
    parser.add_argument(
        "--compare-fresh-g3-to-checkpoint",
        type=Path,
        help=(
            "run one fresh-process G0-to-G3 segment and compare it with the "
            "declared HAPB checkpoint source boundary"
        ),
    )
    parser.add_argument(
        "--same-owner-reentry-diagnostic",
        action="store_true",
        help=(
            "run a fixed two-by-one-tick H4 diagnostic on one prepared owner "
            "and compare it with one fresh two-tick reference"
        ),
    )
    parser.add_argument(
        "--total-ticks",
        type=int,
        help="absolute native-tick horizon; defaults to the campaign spec",
    )
    parser.add_argument(
        "--split-tick",
        type=int,
        help="absolute native checkpoint tick; defaults to the first non-terminal checkpoint",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="deprecated compatibility alias; never an execution authority",
    )
    parser.add_argument(
        "--split-generations",
        type=int,
        default=None,
        help="deprecated compatibility alias; never an execution authority",
    )
    parser.add_argument(
        "--physical-cells",
        type=int,
        default=CELLS,
        help="physical B for this continuation; 48 is the frozen factory-200 extent",
    )
    parser.add_argument(
        "--tile-rows",
        type=int,
        default=512,
        help=(
            "bounded HAPB overflow tile candidate; 512 is the qualified "
            "default and 4096 is the secondary timing candidate"
        ),
    )
    parser.add_argument(
        "--acs-checkpoint-slots",
        type=int,
        default=2,
        help=(
            "ACS persist slots for the HAPB midpoint envelope; 2 is the "
            "qualified default, 0 is explicit synchronous torch.save"
        ),
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise AllocationDiagnosticError("refusing to overwrite immutable HAPB continuation evidence")
    from engine.perf.cuda_architecture_b_hapb_v1 import hapb_physical_cell_count_v1
    from engine.perf.hapb_physical_extent_v1 import expand_campaign_spec_physical_cells_v1

    try:
        physical_cells = hapb_physical_cell_count_v1(args.physical_cells)
    except Exception as exc:
        raise AllocationDiagnosticError("HAPB physical cells must be in [1, 256]") from exc
    if type(args.tile_rows) is not int or args.tile_rows not in {512, 4096}:
        raise AllocationDiagnosticError("HAPB tile rows must be exactly 512 or 4096")
    if type(args.acs_checkpoint_slots) is not int or args.acs_checkpoint_slots not in {0, 1, 2, 3}:
        raise AllocationDiagnosticError("ACS checkpoint slots must be 0, 1, 2, or 3")
    diagnostic_modes = (
        args.repeat_diagnostic_end_generation is not None,
        args.repeat_diagnostic_end_tick is not None,
        args.single_diagnostic_end_tick is not None,
        args.single_diagnostic_end_generation is not None,
        args.same_owner_reentry_diagnostic,
    )
    if sum(diagnostic_modes) > 1:
        raise AllocationDiagnosticError("HAPB repeat diagnostic modes are exclusive")
    if args.repeat_diagnostic_end_generation is not None:
        if args.repeat_diagnostic_end_generation != args.split_generations:
            raise AllocationDiagnosticError(
                "HAPB repeat diagnostic must preserve the qualified G3 minimum"
            )
        if (
            args.resume_checkpoint is not None
            or args.compare_reference_midpoint
            or args.compare_fresh_g3_to_checkpoint is not None
        ):
            raise AllocationDiagnosticError(
                "HAPB repeat diagnostic cannot consume continuation checkpoints"
            )
    if args.repeat_diagnostic_end_tick is not None:
        if not 1 <= args.repeat_diagnostic_end_tick <= 160:
            raise AllocationDiagnosticError(
                "HAPB repeat tick diagnostic must end inside the first lifecycle interval"
            )
        if (
            args.resume_checkpoint is not None
            or args.compare_reference_midpoint
            or args.compare_fresh_g3_to_checkpoint is not None
        ):
            raise AllocationDiagnosticError(
                "HAPB repeat tick diagnostic cannot consume continuation checkpoints"
            )
    if args.single_diagnostic_end_tick is not None:
        if not 1 <= args.single_diagnostic_end_tick <= 160:
            raise AllocationDiagnosticError(
                "HAPB single tick diagnostic must end inside the first lifecycle interval"
            )
        if (
            args.resume_checkpoint is not None
            or args.compare_reference_midpoint
            or args.compare_fresh_g3_to_checkpoint is not None
        ):
            raise AllocationDiagnosticError(
                "HAPB single tick diagnostic cannot consume continuation checkpoints"
            )
    if args.single_diagnostic_end_generation is not None:
        if args.single_diagnostic_end_generation not in {
            1,
            args.split_generations,
        }:
            raise AllocationDiagnosticError(
                "HAPB single generation diagnostic must be 1 (residual-wall) "
                "or the qualified G3 minimum"
            )
        if (
            args.resume_checkpoint is not None
            or args.compare_reference_midpoint
            or args.compare_fresh_g3_to_checkpoint is not None
        ):
            raise AllocationDiagnosticError(
                "HAPB single generation diagnostic cannot consume continuation checkpoints"
            )
    if args.compare_fresh_g3_to_checkpoint is not None and (
        args.resume_checkpoint is not None or args.compare_reference_midpoint
    ):
        raise AllocationDiagnosticError(
            "HAPB fresh-process checkpoint diagnostic is a distinct execution mode"
        )
    if args.same_owner_reentry_diagnostic and (
        args.resume_checkpoint is not None
        or args.compare_reference_midpoint
        or args.compare_fresh_g3_to_checkpoint is not None
    ):
        raise AllocationDiagnosticError(
            "HAPB same-owner re-entry diagnostic cannot consume continuation "
            "checkpoints"
        )

    spec_path = args.spec.expanduser().resolve()
    if not spec_path.is_file():
        raise AllocationDiagnosticError(f"HAPB continuation spec is absent: {spec_path}")

    import os
    import torch
    os.environ["EIEL_HAPB_ENGINEERING_PROBE"] = "1"
    from engine.perf.hapb_contended_shared_gpu_v1 import (
        admit_hapb_exclusive_cuda_launch_v1,
        snapshot_from_resource_monitor_gpu_v1,
    )
    from tools.resource_monitor import collect_gpu

    gpu_observation = collect_gpu()
    exclusive_cuda_admission = admit_hapb_exclusive_cuda_launch_v1(
        snapshot=snapshot_from_resource_monitor_gpu_v1(gpu_observation),
        cuda_available=bool(torch.cuda.is_available())
        and bool(gpu_observation.get("available")),
    )
    if not exclusive_cuda_admission["admitted"]:
        raise AllocationDiagnosticError(
            "HAPB exclusive CUDA admission refused ("
            + ",".join(exclusive_cuda_admission["refuse_reasons"])
            + "); WAIT-RESOURCE is not PERF-B"
        )
    artifact, manifest, source_root, output_root = (
        args.artifact.resolve(), args.manifest.resolve(), args.source_root.resolve(), args.output_root.resolve()
    )
    if not artifact.is_file() or not manifest.is_file() or not source_root.is_dir():
        raise AllocationDiagnosticError("HAPB continuation inputs are absent")
    resume_checkpoint = (
        args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None
    )
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise AllocationDiagnosticError("reused HAPB midpoint checkpoint is absent")
    diagnostic_checkpoint = (
        args.compare_fresh_g3_to_checkpoint.resolve()
        if args.compare_fresh_g3_to_checkpoint is not None
        else None
    )
    if diagnostic_checkpoint is not None and not diagnostic_checkpoint.is_file():
        raise AllocationDiagnosticError("diagnostic HAPB midpoint checkpoint is absent")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    option_d_m0_telemetry_enabled = _option_d_m0_telemetry_enabled_v1(
        manifest_payload
    )
    artifact_sha256, manifest_sha256 = _sha256(artifact), _sha256(manifest)

    source_spec = load_campaign_spec(spec_path)
    source_spec["campaign_id"] = f"{source_spec['campaign_id']}-hapb-ordinary-birth"
    source_spec["parameters"]["production_lifecycle.ordinary_births_enabled"] = True
    from engine.campaigns.patch_runner_factory import (
        apply_hapb_frag_identity_runtime_controls_v1,
    )

    # Bind the declared force/reuse arm before module loading, owner
    # preparation, and receipt capture.  The provider also applies this
    # boundary during runner construction; doing it here prevents a retained
    # process environment from making the harness observe the wrong arm.
    apply_hapb_frag_identity_runtime_controls_v1(source_spec["parameters"])
    if int(source_spec.get("parameters", {}).get("tw_rngv2.semantics_version", 1) or 1) == 2:
        from types import SimpleNamespace

        from engine.perf.cuda_architecture_b_hapb_v1 import apply_hapb_frag_rng_route_v1

        apply_hapb_frag_rng_route_v1(
            SimpleNamespace(), hapb_rng_semantics_version=2
        )
    if args.generation_turnover:
        source_spec["parameters"]["production_lifecycle.lifecycle_mode"] = (
            "legacy_synchronized_truncation_v1"
        )
        source_spec["parameters"]["production_lifecycle.life_history_profile"] = (
            "legacy_inert_v1"
        )
        source_spec["parameters"]["production_lifecycle.turnover_cull_fraction"] = 0.5
        source_spec["parameters"]["production_lifecycle.turnover_parent_fraction"] = 1.0
    tick_native_control = _h4_tick_native_control_v1(
        source_spec,
        generation_turnover=args.generation_turnover,
        total_ticks=args.total_ticks,
        split_tick=args.split_tick,
    )
    execution_controls = source_spec["execution"]
    ticks_per_generation = int(tick_native_control.refill_chunk_ticks)
    split_tick = args.split_tick
    if split_tick is None:
        split_tick = next(
            (
                value
                for value in tick_native_control.resolved_checkpoint_ticks
                if value < tick_native_control.total_ticks
            ),
            None,
        )
    if split_tick is None or not 0 < split_tick < tick_native_control.total_ticks:
        raise AllocationDiagnosticError(
            "HAPB H4 requires an interior absolute native tick checkpoint"
        )
    # Private adapter values for the current provider ABI.  All active route
    # control, checkpoints, restore cursors, and evidence remain absolute ticks.
    args.generations = (tick_native_control.total_ticks + ticks_per_generation - 1) // ticks_per_generation
    _, args.split_generations = _h4_private_adapter_bounds_v1(
        start_tick=0,
        end_tick=split_tick,
        chunk_ticks=ticks_per_generation,
    )
    resume_adapter_start, _ = _h4_private_adapter_bounds_v1(
        start_tick=split_tick,
        end_tick=tick_native_control.total_ticks,
        chunk_ticks=ticks_per_generation,
    )
    # The historical opt-out is retained only as a CLI compatibility alias;
    # midpoint exactness is mandatory for the active H4 route.
    args.compare_reference_midpoint = True
    if physical_cells > CELLS:
        source_spec = expand_campaign_spec_physical_cells_v1(source_spec, physical_cells)
    factory_receipt = json.loads((source_root / "source_factory_receipt.json").read_text(encoding="utf-8"))
    factory_rows = factory_receipt.get("rows")
    if not isinstance(factory_rows, list) or len(factory_rows) != physical_cells:
        raise AllocationDiagnosticError(
            "HAPB source factory is not the requested physical extent"
        )
    selected_cell_ids = tuple(str(row.get("cell_id")) for row in factory_rows if isinstance(row, Mapping))
    if len(selected_cell_ids) != physical_cells:
        raise AllocationDiagnosticError("HAPB source factory cell identities are incomplete")
    source_plan = compile_campaign(source_spec).payload()
    runtime_plan = _bind_hapb_artifact_abi(_bind_diagnostic_artifact(
        _runtime_plan(
            source_plan,
            cells=physical_cells,
            selected_cell_ids=selected_cell_ids,
        ),
        artifact_override=artifact,
        manifest_override=manifest,
    ))
    _preflight_external_artifact_bindings_v1(runtime_plan)
    output_root.mkdir(parents=True)
    qualification, reporting = _diagnostic_descriptor(
        output_root, artifact_sha256=artifact_sha256, manifest_sha256=manifest_sha256,
        plan_sha256=str(runtime_plan["plan_sha256"]),
    )
    plan = _diagnostic_plan(runtime_plan, qualification=qualification, reporting=reporting)
    progress = output_root / "continuation_progress.json"
    from engine.perf.hapb_contended_shared_gpu_v1 import (
        hapb_host_admission_from_samples_v1,
        hapb_memory_admission_from_samples_v1,
        host_sample_v1,
        memory_sample_from_snapshot_v1,
        snapshot_from_resource_monitor_gpu_v1,
    )

    memory_samples: list[dict[str, Any]] = []
    host_samples: list[dict[str, Any]] = []

    def record(stage: str) -> None:
        atomic_write_json(
            progress,
            {
                "schema_version": SCHEMA_VERSION,
                "stage": stage,
                "cells": physical_cells,
                "time_authority": "absolute_tick_v1",
                "total_ticks": tick_native_control.total_ticks,
                "checkpoint_ticks": list(tick_native_control.resolved_checkpoint_ticks),
            },
        )
        try:
            from tools.resource_monitor import collect_gpu

            memory_samples.append(
                memory_sample_from_snapshot_v1(
                    snapshot_from_resource_monitor_gpu_v1(collect_gpu()),
                    label=stage,
                )
            )
            host_samples.append(host_sample_v1(label=stage))
        except (OSError, RuntimeError, ValueError):
            return

    module = _load_exact_module(artifact, eed_overlay=True, hapb_policy_only=True)
    batch_id = str(runtime_plan["physical_batches"][0]["batch_id"])
    identity_provider = AllocationOnlyProvider(bind_committed_defaults=False)
    identity_request = identity_provider.translate_batch(plan, batch_id)
    cell_ids = tuple(identity_request.cell_ids)
    if len(cell_ids) != physical_cells:
        raise AllocationDiagnosticError(
            f"HAPB continuation requires exactly B{physical_cells} cell identities"
        )
    record("reused_sources_verified")
    boundary_timing_samples: list[dict[str, Any]] = []
    frag_identity_runtime_controls = _frag_identity_runtime_controls_v1(os.environ)
    frag_identity_requested = (
        frag_identity_runtime_controls[
            "enable_policy_window_reuse_requested"
        ]
        is True
    )
    frag_identity_trace_enabled = (
        frag_identity_runtime_controls["identity_trace_enabled"] is True
    )
    if args.same_owner_reentry_diagnostic and (
        not frag_identity_requested
        or not frag_identity_trace_enabled
        or frag_identity_runtime_controls[
            "force_physical_policy_refill_requested"
        ]
        is True
    ):
        raise AllocationDiagnosticError(
            "same-owner re-entry diagnostic requires ENABLE=1, TRACE=1, and FORCE unset"
        )
    frag_identity_execution_ordinal = 0

    def secondary_boundary_timing_receipt() -> dict[str, Any]:
        return {
            "schema_version": f"{SCHEMA_VERSION}_secondary_boundary_timing_v2",
            "samples": list(boundary_timing_samples),
            "frag_identity_runtime_controls": dict(
                frag_identity_runtime_controls
            ),
            "identity_subtime_telemetry": {
                "measured": option_d_m0_telemetry_enabled,
                "option_d_m0_telemetry_enabled": option_d_m0_telemetry_enabled,
                "authority": "artifact_manifest.option_d_m0_telemetry_build",
                "unavailable_reason": (
                    None
                    if option_d_m0_telemetry_enabled
                    else "artifact_compiled_with_option_d_m0_telemetry_disabled"
                ),
            },
            "policy_byte_reduction_credit": False,
        }

    def execute(
        source_projections: Sequence[Mapping[str, Any]], *, start: int, end: int,
        component_checkpoint: Mapping[str, Any] | None = None,
        expected_restore_sources: Sequence[Mapping[str, Any]] | None = None,
        emit_checkpoint: bool = True,
        absolute_tick_start: int | None = None,
        absolute_tick_end: int | None = None,
        absolute_tick_span: int | None = None,
        birth_trace_label: str | None = None,
        sequential_tick_spans: tuple[int, int] | None = None,
    ) -> tuple[
        tuple[Mapping[str, Any], ...],
        Mapping[str, Any] | None,
        Mapping[str, Any],
    ]:
        nonlocal frag_identity_execution_ordinal
        arm_full_h4_started_ns = perf_counter_ns()
        segment_start_tick = (
            start * ticks_per_generation
            if absolute_tick_start is None
            else int(absolute_tick_start)
        )
        segment_end_tick = (
            end * ticks_per_generation
            if absolute_tick_end is None
            else int(absolute_tick_end)
        )
        if segment_start_tick < 0 or segment_end_tick <= segment_start_tick:
            raise AllocationDiagnosticError(
                "HAPB execution requires a positive absolute native tick span"
            )
        if absolute_tick_span is not None and sequential_tick_spans is not None:
            raise AllocationDiagnosticError(
                "HAPB execution cannot combine one-shot and sequential tick spans"
            )
        if sequential_tick_spans is not None and sequential_tick_spans != (1, 1):
            if (
                not sequential_tick_spans
                or any(type(span) is not int or span <= 0 for span in sequential_tick_spans)
                or any(
                    span > tick_native_control.native_dispatch_window_ticks
                    for span in sequential_tick_spans
                )
                or sum(sequential_tick_spans)
                != segment_end_tick - segment_start_tick
            ):
                raise AllocationDiagnosticError(
                    "HAPB sequential dispatch spans do not match the tick-native horizon"
                )
        identity_trace_label = None
        if frag_identity_trace_enabled:
            frag_identity_execution_ordinal += 1
            identity_trace_label = (
                f"execute_{frag_identity_execution_ordinal:02d}_tick"
                f"{segment_start_tick}_to_tick{segment_end_tick}"
            )
        # A native segment is allowed to mutate its own provider, request, and
        # runner graph.  Continuation comparison must therefore never reuse
        # that graph as the next segment's implicit source authority.
        run_plan = copy.deepcopy(plan)
        provider = AllocationOnlyProvider(bind_committed_defaults=False)
        request = provider.translate_batch(run_plan, batch_id)
        if tuple(request.cell_ids) != cell_ids:
            raise AllocationDiagnosticError("fresh HAPB segment changed its physical cell identities")
        adapters = dict(provider._resolved_execution_adapters(run_plan))
        selected_cells = provider._batch_cells(run_plan, request)
        prepared = None
        original_owner = None
        runtime_source = None
        rng_trace: dict[str, Any] = {"before_prepare": _process_rng_trace()}
        segment_tick_native_control = TickNativeControlV1.from_execution(
            {
                "tick_native_control": {
                    **tick_native_control.payload(),
                    "next_absolute_tick": segment_start_tick,
                }
            }
        )
        host_prepare_started_ns = perf_counter_ns()
        try:
            _, _, prepared = provider._prepare_resident_segment_owner_v1(
                plan=run_plan, runner_builder=adapters["runner_builder"], request_builder=adapters["request_builder"],
                preparer=adapters["preparer"], cells=selected_cells, cell_ids=request.cell_ids,
                batch_root=output_root, start_generation=start, end_generation=end,
                ticks_per_generation=request.ticks_per_generation, artifact=str(artifact),
                qualification=str(qualification), reporting=str(reporting), module=module,
                source_projections=source_projections,
                allocation_generations=provider._agent_id_allocation_generation_count(request, start_generation=start),
                genome_v2_checkpoint_authority_binding=dict(request.genome_v2_checkpoint_authority_binding),
                allow_short_technical_gate_generation_count=(
                    absolute_tick_span is not None
                    or sequential_tick_spans is not None
                    or args.single_diagnostic_end_generation is not None
                ),
                allow_partial_generation_source_cursor=(
                    absolute_tick_span is not None
                    or sequential_tick_spans is not None
                    or args.single_diagnostic_end_generation is not None
                ),
                physical_window_generations=1,
                hapb_policy_tile_rows=args.tile_rows,
                tick_native_control=segment_tick_native_control.payload(),
            )
            host_prepare_wall_ns = perf_counter_ns() - host_prepare_started_ns
            rng_trace["after_prepare"] = _process_rng_trace()
            if expected_restore_sources is not None:
                prepared_sources = (
                    prepared.physical_window_arguments.patch.source
                    .checkpoint_projection_inputs["pre_generation_canonical_projections"]
                )
                prepared_restore_difference = _prepared_restore_difference_summary(
                    expected_restore_sources, prepared_sources,
                )
                atomic_write_json(
                    output_root / "prepared_restore_projection_difference.json",
                    prepared_restore_difference,
                )
                if prepared_restore_difference["mismatch_count"]:
                    raise AllocationDiagnosticError(
                        "HAPB allocator preparation does not preserve its G3 source boundary"
                    )
            # Preparation has copied every source-owned tensor into the native
            # owner.  Do not retain an otherwise duplicate 4 GiB source set
            # while projecting a terminal B48 state.
            source_projections = ()
            if component_checkpoint is not None:
                restored = restore_direct_resident_wdl_jungle_hapb_component_checkpoint_v1(
                    prepared.physical_window_arguments, component_checkpoint,
                    absolute_tick=segment_start_tick,
                    cell_ids=request.cell_ids,
                )
                if restored != segment_start_tick:
                    raise AllocationDiagnosticError("HAPB component restore returned the wrong tick cursor")
                identity_checkpoint = component_checkpoint.get(
                    "hapb_policy_window_identity"
                )
                if identity_checkpoint is None:
                    if prepared.policy_window_identity_owner is not None:
                        raise AllocationDiagnosticError(
                            "HAPB component restore lacks its native identity checkpoint"
                        )
                else:
                    restore_hapb_policy_window_identity_checkpoint_v1(
                        module,
                        prepared.physical_window_arguments,
                        identity_checkpoint,
                    )
            record(f"native_tick{segment_start_tick}_to_tick{segment_end_tick}_started")
            executor_arguments = {
                "physical_window_arguments": prepared.physical_window_arguments,
                "logical_generation_count": end - start,
            }
            if absolute_tick_span is not None:
                executor_arguments["absolute_tick_span"] = absolute_tick_span
            sequential_entry_results: list[dict[str, Any]] = []
            sequential_native_results: list[Mapping[str, Any]] = []
            interentry_projection_wall_ns = 0
            projection_prepared = prepared.physical_window_arguments
            birth_trace: list[dict[str, Any]] = []
            direct_runtime = None
            original_commit = None
            runtime_source = prepared.physical_window_arguments.patch.source
            original_owner = runtime_source.live_banks.get(
                "hapb_policy_owner"
            )
            if args.disable_content_alias:
                if original_owner is None or not hasattr(
                    original_owner, "content_alias_enabled"
                ):
                    raise AllocationDiagnosticError(
                        "HAPB baseline owner is unavailable"
                    )
                # The baseline is a same-source diagnostic.  Replacing the
                # immutable owner record keeps the production alias route and
                # its canonical content map untouched while forcing the old
                # overflow-tile path for this one bounded run.
                # Fresh G0 prepare already filled a usable hot prefix; wiping
                # it here fails native execute.  Invalidate only on resume
                # when the restored owner is still alias-on (the 208/209
                # export-order hole).
                invalidate_hot_prefix = (
                    component_checkpoint is not None
                    and original_owner.content_alias_enabled is True
                )
                runtime_source.live_banks["hapb_policy_owner"] = replace(
                    original_owner,
                    content_alias_enabled=False,
                    hot_prefix_valid=(
                        False
                        if invalidate_hot_prefix
                        else original_owner.hot_prefix_valid
                    ),
                )
            if birth_trace_label is not None:
                from engine.perf import (
                    cuda_architecture_b_direct_resident_wdl_jungle_v1 as direct_runtime,
                )

                original_commit = direct_runtime.commit_hapb_ordinary_fixed_mlp_birth_rows_v1

                def traced_commit(owner: Any, **kwargs: Any) -> Any:
                    call_ordinal = len(birth_trace) + 1
                    counts = kwargs["birth_counts"]
                    parents = kwargs["parent_slots"]
                    children = kwargs["child_slots"]
                    deltas = kwargs["policy_birth_delta_tape"]
                    masks = kwargs["policy_birth_mutation_mask_tape"]
                    semantic = kwargs["policy_birth_semantic_draws_tape"]
                    cursors = kwargs["policy_birth_tape_cursor"]
                    active: list[dict[str, Any]] = []
                    expected_rows: dict[int, Any] = {}
                    for cell, raw_count in enumerate(counts.tolist()):
                        count = int(raw_count)
                        if count == 0:
                            continue
                        cursor = int(cursors[cell].item())
                        parent_slots = parents[cell, :count]
                        child_slots = children[cell, :count]
                        parent_rows = owner.canonical_host_bank[cell].index_select(
                            0, parent_slots
                        ).clone()
                        delta = deltas[cell, cursor : cursor + count]
                        if int(delta.shape[1]) == 1:
                            delta = delta.expand(-1, int(parent_rows.shape[1]))
                        expected = torch.add(parent_rows, delta)
                        expected_rows[cell] = (child_slots.clone(), expected)
                        active.append({
                            "cell_index": cell,
                            "birth_count": count,
                            "policy_birth_tape_cursor_start": cursor,
                            "policy_birth_tape_cursor_end": cursor + count,
                            "parent_slots_sha256": _projection_value_digest(parent_slots),
                            "child_slots_sha256": _projection_value_digest(child_slots),
                            "parent_rows_sha256": _projection_value_digest(parent_rows),
                            "policy_delta_sha256": _projection_value_digest(delta),
                            "policy_mutation_mask_sha256": _projection_value_digest(
                                masks[cell, cursor : cursor + count]
                            ),
                            "policy_semantic_draws_sha256": _projection_value_digest(
                                semantic[cell, cursor : cursor + count]
                            ),
                            "cpu_expected_child_rows_sha256": _projection_value_digest(expected),
                        })
                    committed = original_commit(owner, **kwargs)
                    by_cell = {int(item["cell_index"]): item for item in active}
                    for cell, (child_slots, expected) in expected_rows.items():
                        actual = owner.canonical_host_bank[cell].index_select(
                            0, child_slots
                        )
                        by_cell[cell]["committed_child_rows_sha256"] = (
                            _projection_value_digest(actual)
                        )
                        by_cell[cell]["cpu_reference_exact"] = torch.equal(
                            actual, expected
                        )
                    birth_trace.append({
                        "callback_ordinal": call_ordinal,
                        "total_birth_count": int(counts.sum().item()),
                        "active_cells": active,
                    })
                    return committed

                direct_runtime.commit_hapb_ordinary_fixed_mlp_birth_rows_v1 = traced_commit
            try:
                if sequential_tick_spans is None:
                    # This remains the authoritative g0_to_g6 native-only
                    # timer. Do not add preparation, projection, or export.
                    native_execute_started_ns = perf_counter_ns()
                    result = adapters["executor"](module, **executor_arguments)
                    native_execute_wall_ns = (
                        perf_counter_ns() - native_execute_started_ns
                    )
                    sequential_native_results = [result]
                else:
                    entry_native_wall_ns: list[int] = []
                    current_prepared = prepared.physical_window_arguments
                    for ordinal, span in enumerate(
                        sequential_tick_spans, start=1
                    ):
                        entry_start_tick = segment_start_tick + sum(
                            sequential_tick_spans[: ordinal - 1]
                        )
                        entry_arguments = dict(executor_arguments)
                        entry_arguments["physical_window_arguments"] = current_prepared
                        entry_arguments["absolute_tick_span"] = span
                        entry_started_ns = perf_counter_ns()
                        entry_result = adapters["executor"](
                            module, **entry_arguments
                        )
                        entry_wall_ns = perf_counter_ns() - entry_started_ns
                        entry_native_wall_ns.append(entry_wall_ns)
                        entry_accounting = _frag_identity_accounting_v1(
                            entry_result,
                            runtime_controls=frag_identity_runtime_controls,
                            option_d_m0_telemetry_enabled=(
                                option_d_m0_telemetry_enabled
                            ),
                        )
                        entry_rng = _process_rng_trace()
                        sequential_native_results.append(entry_result)
                        sequential_entry_results.append(
                            _same_owner_entry_result_v1(
                                entry_result,
                                ordinal=ordinal,
                                identity_accounting=entry_accounting,
                                native_execute_wall_ns=entry_wall_ns,
                                process_rng_after_entry=entry_rng,
                            )
                        )
                        if ordinal < len(sequential_tick_spans):
                            interentry_projection_started_ns = perf_counter_ns()
                            intermediate_projections = (
                                project_direct_resident_wdl_jungle_multicell_checkpoints_v1(
                                    current_prepared,
                                    direct_result=entry_result,
                                    completed_generation_count=1,
                                    reporting_enabled=False,
                                )
                            )
                            interentry_projection_wall_ns = (
                                perf_counter_ns()
                                - interentry_projection_started_ns
                            )
                            sequential_entry_results[-1][
                                "terminal_projection_digests"
                            ] = list(_terminal_digests(intermediate_projections))
                            current_prepared = _rebind_same_owner_tick_source_v1(
                                current_prepared,
                                intermediate_projections,
                                expected_tick=entry_start_tick + span,
                            )
                    result = sequential_native_results[-1]
                    projection_prepared = current_prepared
                    native_execute_wall_ns = sum(entry_native_wall_ns)
            finally:
                if direct_runtime is not None and original_commit is not None:
                    direct_runtime.commit_hapb_ordinary_fixed_mlp_birth_rows_v1 = original_commit
            traffic = result.get("hapb_policy_traffic_counters")
            if not isinstance(traffic, Mapping):
                raise AllocationDiagnosticError(
                    "HAPB native result omitted physical policy traffic counters"
                )
            if sequential_tick_spans is None:
                identity_accounting = _frag_identity_accounting_v1(
                    result,
                    runtime_controls=frag_identity_runtime_controls,
                    option_d_m0_telemetry_enabled=option_d_m0_telemetry_enabled,
                )
            else:
                identity_accounting = _aggregate_frag_identity_accounting_v1(
                    [
                        entry["frag_identity_accounting"]
                        for entry in sequential_entry_results
                    ]
                )
            identity_trace_artifact = None
            if frag_identity_trace_enabled:
                identity_receipts: list[Any] = []
                identity_entry_receipts: list[Mapping[str, Any]] = []
                outer_materialization_receipts: list[Any] = []
                resume_materialization_receipts: list[dict[str, Any]] = []
                for entry_ordinal, native_result in enumerate(
                    sequential_native_results, start=1
                ):
                    entry_trace_receipts = native_result.get(
                        "policy_birth_segment_refill_receipts"
                    )
                    if not isinstance(entry_trace_receipts, list):
                        raise AllocationDiagnosticError(
                            "FRAG identity trace requested but native receipts are absent"
                        )
                    identity_receipts.extend(entry_trace_receipts)
                    identity_entry_receipts.extend(
                        _identity_entry_receipts_v1(native_result)
                    )
                    raw_outer = native_result.get("policy_refill_receipts", [])
                    if not isinstance(raw_outer, list):
                        raise AllocationDiagnosticError(
                            "FRAG identity outer materialization receipts are malformed"
                        )
                    outer_materialization_receipts.extend(raw_outer)
                    raw_resume = native_result.get(
                        "resume_policy_refill_receipt", {}
                    )
                    if raw_resume:
                        if not isinstance(raw_resume, Mapping):
                            raise AllocationDiagnosticError(
                                "FRAG identity resume materialization receipt is malformed"
                            )
                        resume_materialization_receipts.append(
                            {
                                "entry_ordinal": entry_ordinal,
                                "receipt": dict(raw_resume),
                            }
                        )
                identity_trace_path = (
                    output_root
                    / f"frag_identity_trace_{identity_trace_label}.json"
                )
                atomic_write_json(
                    identity_trace_path,
                    {
                        "schema_version": "eiel_hapb_frag_identity_execution_trace_v1",
                        "label": identity_trace_label,
                        "time_authority": "absolute_tick_v1",
                        "start_next_tick": segment_start_tick,
                        "end_next_tick": segment_end_tick,
                        "absolute_tick_span": absolute_tick_span,
                        "sequential_tick_spans": (
                            list(sequential_tick_spans)
                            if sequential_tick_spans is not None
                            else None
                        ),
                        "checkpoint_restore_input": component_checkpoint is not None,
                        "entry_ensure_receipts": identity_entry_receipts,
                        "outer_policy_materialization_receipts": (
                            outer_materialization_receipts
                        ),
                        "resume_policy_materialization_receipts": (
                            resume_materialization_receipts
                        ),
                        # Retain the pre-existing single-execution field for
                        # default-mode consumers; sequential evidence uses the
                        # ordinal-bound plural form above.
                        "resume_policy_materialization_receipt": (
                            dict(
                                result.get(
                                    "resume_policy_refill_receipt", {}
                                )
                            )
                            if sequential_tick_spans is None
                            and isinstance(
                                result.get("resume_policy_refill_receipt", {}),
                                Mapping,
                            )
                            else {}
                        ),
                        "trace_receipts": identity_receipts,
                        "same_owner_entry_results": sequential_entry_results,
                    },
                )
                identity_trace_artifact = {
                    "path": str(identity_trace_path),
                    "sha256": _sha256(identity_trace_path),
                }
            host_classification_entry_wall_ns: list[int] = []
            host_classification_wall_available = True
            for native_result in sequential_native_results:
                host_boundary_timing = native_result.get(
                    "hapb_host_boundary_timing"
                )
                entry_host_wall_ns = (
                    host_boundary_timing.get(
                        "host_classification_map_construction_wall_ns"
                    )
                    if isinstance(host_boundary_timing, Mapping)
                    else None
                )
                if type(entry_host_wall_ns) is not int:
                    host_classification_wall_available = False
                    continue
                host_classification_entry_wall_ns.append(entry_host_wall_ns)
            host_classification_wall_ns = (
                sum(host_classification_entry_wall_ns)
                if host_classification_wall_available
                else None
            )
            boundary_timing_sample = {
                "schema_version": f"{SCHEMA_VERSION}_secondary_boundary_timing_sample_v1",
                "label": birth_trace_label or (
                    f"tick{segment_start_tick}_to_tick{segment_end_tick}"
                ),
                "tick_native_control": tick_native_control.payload(),
                "tick_native_dispatch_spans": (
                    list(sequential_tick_spans)
                    if sequential_tick_spans is not None
                    else ([absolute_tick_span] if absolute_tick_span is not None else None)
                ),
                "tick_native_start_next_tick": segment_start_tick,
                "tick_native_end_next_tick": segment_end_tick,
                "tile_rows": args.tile_rows,
                "host_prepare_wall_ns": int(host_prepare_wall_ns),
                "host_classification_map_construction_wall_ns": (
                    host_classification_wall_ns
                ),
                "host_classification_map_construction_entry_wall_ns": (
                    host_classification_entry_wall_ns
                ),
                "native_execute_wall_ns": int(native_execute_wall_ns),
                "sequential_entry_native_execute_wall_ns": [
                    int(entry["native_execute_wall_ns"])
                    for entry in sequential_entry_results
                ],
                "interentry_projection_wall_ns": int(
                    interentry_projection_wall_ns
                ),
                "counters": dict(traffic),
                "control_d2h_bytes": int(traffic.get("control_d2h_bytes", -1)),
                "control_h2d_bytes": int(traffic.get("control_h2d_bytes", -1)),
                "sync_count": int(traffic.get("sync_count", -1)),
                "gpu_stall_ns": int(traffic.get("gpu_stall_ns", -1)),
                "hapb_residual_wall": (
                    dict(result["hapb_residual_wall"])
                    if isinstance(result.get("hapb_residual_wall"), Mapping)
                    else None
                ),
                "frag_identity_accounting": identity_accounting,
                "frag_identity_runtime_controls": dict(
                    frag_identity_runtime_controls
                ),
                "frag_identity_trace_artifact": identity_trace_artifact,
                "same_owner_entry_results": sequential_entry_results,
                "policy_byte_reduction_credit": False,
                "scope": (
                    "bounded secondary timing only; wall clock is not a "
                    "policy-byte reduction measurement"
                ),
            }
            boundary_timing_samples.append(boundary_timing_sample)
            if birth_trace_label is not None:
                atomic_write_json(
                    output_root / f"repeat_tick_{birth_trace_label}_policy_traffic.json",
                    {
                        "schema_version": f"{SCHEMA_VERSION}_policy_traffic_v1",
                        "trace_label": birth_trace_label,
                        "content_alias_enabled": result.get(
                            "hapb_content_alias_enabled"
                        ),
                        "content_alias_relation_sha256": result.get(
                            "hapb_content_alias_relation_sha256"
                        ),
                        "counters": dict(traffic),
                    },
                )
            if birth_trace_label is not None:
                atomic_write_json(
                    output_root / f"repeat_tick_{birth_trace_label}_birth_transaction_trace.json",
                    {
                        "schema_version": f"{SCHEMA_VERSION}_birth_transaction_trace_v1",
                        "trace_label": birth_trace_label,
                        "callback_count": len(birth_trace),
                        "events": birth_trace,
                    },
                )
            rng_trace["after_execute"] = _process_rng_trace()
            if any(
                native_result.get("hapb_host_authoritative_ordinary_birth")
                is not True
                or native_result.get("hapb_policy_only_route") is not False
                for native_result in sequential_native_results
            ):
                raise AllocationDiagnosticError(
                    "HAPB continuation lost its ordinary-birth guard"
                )
            record(f"native_tick{segment_start_tick}_to_tick{segment_end_tick}_complete")
            projections = project_direct_resident_wdl_jungle_multicell_checkpoints_v1(
                projection_prepared, direct_result=result,
                completed_generation_count=(
                    end - start
                    if sequential_tick_spans is not None
                    else 1
                    if absolute_tick_span is not None
                    else end - start
                ),
                reporting_enabled=False,
            )
            if sequential_entry_results:
                sequential_entry_results[-1]["terminal_projection_digests"] = list(
                    _terminal_digests(projections)
                )
            envelope = None
            if end == args.split_generations and emit_checkpoint:
                envelope = export_direct_resident_wdl_jungle_hapb_checkpoint_v1(
                    projection_prepared, projections
                )
                if projection_prepared.policy_window_identity_owner is not None:
                    envelope["hapb_policy_window_identity"] = (
                        export_hapb_policy_window_identity_checkpoint_v1(
                            module, projection_prepared
                        )
                    )
            boundary_timing_sample["arm_full_h4_wall_ns"] = int(
                perf_counter_ns() - arm_full_h4_started_ns
            )
            boundary_timing_sample["arm_full_h4_wall_scope"] = {
                "schema_version": f"{SCHEMA_VERSION}_arm_full_h4_wall_scope_v1",
                "starts_before": "plan_copy_provider_translation_and_owner_preparation",
                "ends_after": (
                    "terminal_projection_and_optional_checkpoint_envelope_construction"
                ),
                "includes": [
                    "owner_preparation",
                    "optional_component_restore",
                    "native_execution",
                    "in_execute_resource_sampling_and_trace_artifact_writes",
                    "terminal_projection",
                    "optional_checkpoint_envelope_construction",
                ],
                "excludes": [
                    "caller_source_projection_loading",
                    "checkpoint_persistence_or_external_checkpoint_loading",
                    "post_return_resource_sampling_and_owner_cleanup",
                ],
                "authoritative_native_field": "native_execute_wall_ns",
                "authoritative_g0_to_g6_native_timing_unchanged": True,
                "authority": "secondary_harness_wall_not_native_gate_timing",
            }
            return tuple(projections), envelope, rng_trace
        finally:
            # Restore the production alias-on owner only after project/export
            # so the midpoint envelope records the alias-off flag and maps
            # that the bounded baseline actually executed.
            if args.disable_content_alias and original_owner is not None:
                runtime_source.live_banks["hapb_policy_owner"] = original_owner
            prepared = None
            gc.collect()
            torch.cuda.empty_cache()

    try:
        if args.same_owner_reentry_diagnostic:
            record("same_owner_two_tick_reference_started")
            reference, _, reference_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0,
                end=1,
                emit_checkpoint=False,
                absolute_tick_span=2,
                birth_trace_label="same_owner_reference",
            )
            reference_signatures = _projection_signatures(reference)
            reference_digests = _terminal_digests(reference)
            reference = ()
            gc.collect()
            record("same_owner_two_by_one_tick_started")
            sequential, _, sequential_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0,
                end=1,
                emit_checkpoint=False,
                birth_trace_label="same_owner_reentry",
                sequential_tick_spans=(1, 1),
            )
            sequential_digests = _terminal_digests(sequential)
            difference = _projection_difference_summary(
                reference_signatures, sequential
            )
            difference_path = output_root / "same_owner_reentry_projection_difference.json"
            atomic_write_json(difference_path, difference)
            sequential_sample = boundary_timing_samples[-1]
            raw_entries = sequential_sample.get("same_owner_entry_results")
            if not isinstance(raw_entries, list):
                raise AllocationDiagnosticError(
                    "same-owner diagnostic entry results are absent"
                )
            same_owner_proof = _verify_same_owner_reentry_v1(raw_entries)
            process_rng_exact = (
                reference_rng_trace.get("after_execute")
                == sequential_rng_trace.get("after_execute")
            )
            if (
                reference_digests != sequential_digests
                or difference["mismatch_count"]
                or not process_rng_exact
            ):
                raise AllocationDiagnosticError(
                    "same-owner two-by-one-tick execution differs from the two-tick reference"
                )
            record("same_owner_reentry_exact")
            receipt = {
                "schema_version": f"{SCHEMA_VERSION}_same_owner_reentry_diagnostic_v1",
                "authority": "hapb_fixed_two_by_one_tick_same_owner_diagnostic_v1",
                "host_harness_source_commit": _head(),
                "artifact_source_commit": manifest_payload[
                    "committed_source_snapshot"
                ]["build_source_commit"],
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "reused_generation_zero_source_root": str(source_root),
                "physical_cells": physical_cells,
                "tile_rows": args.tile_rows,
                "reference_tick_spans": [2],
                "same_owner_tick_spans": [1, 1],
                "same_prepared_identity_owner": True,
                "same_owner_reentry_proof": same_owner_proof,
                "same_owner_entry_results": raw_entries,
                "terminal_projection_digests_exact": True,
                "process_rng_after_execute_exact": True,
                "mismatch_count": difference["mismatch_count"],
                "terminal_projection_digests": {
                    "fresh_two_tick_reference": list(reference_digests),
                    "same_owner_two_by_one_tick": list(sequential_digests),
                },
                "process_rng_traces": {
                    "fresh_two_tick_reference": reference_rng_trace,
                    "same_owner_two_by_one_tick": sequential_rng_trace,
                },
                "projection_difference_artifact": {
                    "path": str(difference_path),
                    "sha256": _sha256(difference_path),
                },
                "frag_identity_runtime_controls": dict(
                    frag_identity_runtime_controls
                ),
                "content_alias_enabled": not args.disable_content_alias,
                "production_execution_admission": False,
                "qualification_claim": False,
                "secondary_boundary_timing": secondary_boundary_timing_receipt(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(
                output_root / "same_owner_reentry_diagnostic_receipt.json",
                receipt,
            )
            return 0

        if diagnostic_checkpoint is not None:
            record(f"fresh_process_tick0_to_tick{split_tick}_started")
            fresh, _, fresh_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0,
                end=args.split_generations,
                emit_checkpoint=False,
                birth_trace_label="fresh_process",
                sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                    tick_native_control,
                    start_tick=0,
                    end_tick=split_tick,
                ),
            )
            loaded_diagnostic_envelope = torch.load(
                diagnostic_checkpoint, map_location="cpu", weights_only=False
            )
            checkpoint_sources = restore_direct_resident_wdl_jungle_hapb_checkpoint_sources_v1(
                loaded_diagnostic_envelope
            )
            difference = _prepared_restore_difference_summary(fresh, checkpoint_sources)
            atomic_write_json(
                output_root / "fresh_process_checkpoint_projection_difference.json",
                difference,
            )
            rng_traces = {"fresh_process": fresh_rng_trace}
            if difference["mismatch_count"]:
                raise AllocationDiagnosticError(
                    "fresh-process HAPB G3 state differs from the checkpoint boundary"
                )
            record(f"fresh_process_tick0_to_tick{split_tick}_checkpoint_exact")
            receipt = {
                "schema_version": f"{SCHEMA_VERSION}_fresh_process_checkpoint_diagnostic_v1",
                "authority": "hapb_b48_fresh_process_checkpoint_diagnostic_v1",
                "host_harness_source_commit": _head(),
                "artifact_source_commit": manifest_payload["committed_source_snapshot"]["build_source_commit"],
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "reused_generation_zero_source_root": str(source_root),
                "checkpoint_path": str(diagnostic_checkpoint),
                "checkpoint_sha256": _sha256(diagnostic_checkpoint),
                "physical_cells": physical_cells,
                "tile_rows": args.tile_rows,
                "time_authority": "absolute_tick_v1",
                "start_tick": 0,
                "total_ticks": tick_native_control.total_ticks,
                "checkpoint_ticks": [split_tick],
                "end_tick": split_tick,
                "legacy_compatibility": {
                    "start_generation": 0,
                    "end_generation": args.split_generations,
                    "ticks_per_generation": ticks_per_generation,
                },
                "checkpoint_source_exact": True,
                "mismatch_count": difference["mismatch_count"],
                "process_rng_traces": rng_traces,
                "content_alias_enabled": not args.disable_content_alias,
                "policy_traffic_observations": {
                    "fresh_process": json.loads(
                        (
                            output_root
                            / "repeat_tick_fresh_process_policy_traffic.json"
                        ).read_text(encoding="utf-8")
                    )
                },
                "production_execution_admission": False,
                "qualification_claim": False,
                "secondary_boundary_timing": secondary_boundary_timing_receipt(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(
                output_root / "fresh_process_checkpoint_diagnostic_receipt.json",
                receipt,
            )
            return 0

        if args.repeat_diagnostic_end_tick is not None:
            diagnostic_tick = args.repeat_diagnostic_end_tick
            diagnostic_generation_horizon = 1
            record(f"repeat_tick0_to_tick{diagnostic_tick}_first_started")
            first, _, first_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0, end=diagnostic_generation_horizon, emit_checkpoint=False,
                absolute_tick_span=diagnostic_tick,
                birth_trace_label="first",
            )
            first_signatures = _projection_signatures(first)
            first_digests = _terminal_digests(first)
            first = ()
            gc.collect()
            record(f"repeat_tick0_to_tick{diagnostic_tick}_second_started")
            second, _, second_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0, end=diagnostic_generation_horizon, emit_checkpoint=False,
                absolute_tick_span=diagnostic_tick,
                birth_trace_label="second",
            )
            second_digests = _terminal_digests(second)
            difference = _projection_difference_summary(first_signatures, second)
            atomic_write_json(output_root / "repeat_tick_projection_difference.json", difference)
            rng_traces = {"first": first_rng_trace, "second": second_rng_trace}
            if first_digests != second_digests or difference["mismatch_count"]:
                raise AllocationDiagnosticError(
                    f"HAPB repeated fresh tick-0 to tick-{diagnostic_tick} spans differ"
                )
            record(f"repeat_tick0_to_tick{diagnostic_tick}_exact")
            receipt = {
                "schema_version": f"{SCHEMA_VERSION}_repeat_tick_diagnostic_v1",
                "authority": "hapb_b48_repeat_tick_diagnostic_v1",
                "host_harness_source_commit": _head(),
                "artifact_source_commit": manifest_payload["committed_source_snapshot"]["build_source_commit"],
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "reused_generation_zero_source_root": str(source_root),
                "physical_cells": physical_cells,
                "tile_rows": args.tile_rows,
                "start_next_tick": 0,
                "end_next_tick": diagnostic_tick,
                "repeat_count": 2,
                "terminal_projection_digests_exact": True,
                "mismatch_count": difference["mismatch_count"],
                "terminal_projection_digests": {
                    "first": list(first_digests),
                    "second": list(second_digests),
                },
                "process_rng_traces": rng_traces,
                "content_alias_enabled": not args.disable_content_alias,
                "policy_traffic_observations": {
                    label: json.loads(
                        (output_root / f"repeat_tick_{label}_policy_traffic.json")
                        .read_text(encoding="utf-8")
                    )
                    for label in ("first", "second")
                },
                "production_execution_admission": False,
                "qualification_claim": False,
                "secondary_boundary_timing": secondary_boundary_timing_receipt(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(output_root / "repeat_tick_diagnostic_receipt.json", receipt)
            return 0

        if args.single_diagnostic_end_tick is not None:
            diagnostic_tick = args.single_diagnostic_end_tick
            diagnostic_generation_horizon = 1
            record(f"single_tick0_to_tick{diagnostic_tick}_started")
            single, _, single_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0,
                end=diagnostic_generation_horizon,
                emit_checkpoint=False,
                absolute_tick_span=diagnostic_tick,
                birth_trace_label="single",
            )
            single_digests = _terminal_digests(single)
            record(f"single_tick0_to_tick{diagnostic_tick}_complete")
            receipt = {
                "schema_version": f"{SCHEMA_VERSION}_single_tick_diagnostic_v1",
                "authority": "hapb_b48_single_tick_performance_diagnostic_v1",
                "host_harness_source_commit": _head(),
                "artifact_source_commit": manifest_payload["committed_source_snapshot"]["build_source_commit"],
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "reused_generation_zero_source_root": str(source_root),
                "physical_cells": physical_cells,
                "tile_rows": args.tile_rows,
                "start_next_tick": 0,
                "end_next_tick": diagnostic_tick,
                "repeat_count": 1,
                "determinism_check_performed": False,
                "terminal_projection_digests": {"single": list(single_digests)},
                "process_rng_traces": {"single": single_rng_trace},
                "content_alias_enabled": not args.disable_content_alias,
                "policy_traffic_observations": {
                    "single": json.loads(
                        (output_root / "repeat_tick_single_policy_traffic.json")
                        .read_text(encoding="utf-8")
                    )
                },
                "production_execution_admission": False,
                "qualification_claim": False,
                "secondary_boundary_timing": secondary_boundary_timing_receipt(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(output_root / "single_tick_diagnostic_receipt.json", receipt)
            return 0

        if args.single_diagnostic_end_generation is not None:
            diagnostic_end = args.single_diagnostic_end_generation
            record(
                f"single_tick0_to_tick{diagnostic_end * ticks_per_generation}_started"
            )
            single, _, single_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0,
                end=diagnostic_end,
                emit_checkpoint=False,
                sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                    tick_native_control,
                    start_generation=0,
                    end_generation=diagnostic_end,
                    ticks_per_generation=ticks_per_generation,
                ),
            )
            single_digests = _terminal_digests(single)
            record(
                f"single_tick0_to_tick{diagnostic_end * ticks_per_generation}_complete"
            )
            receipt = {
                "schema_version": f"{SCHEMA_VERSION}_single_tick_diagnostic_v1",
                "authority": "hapb_official_g0_g3_profiler_diagnostic_v1",
                "host_harness_source_commit": _head(),
                "artifact_source_commit": manifest_payload["committed_source_snapshot"]["build_source_commit"],
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "reused_generation_zero_source_root": str(source_root),
                "physical_cells": physical_cells,
                "tile_rows": args.tile_rows,
                "time_authority": "absolute_tick_v1",
                "total_ticks": diagnostic_end * ticks_per_generation,
                "end_next_tick": diagnostic_end * ticks_per_generation,
                "legacy_compatibility": {
                    "end_generation": diagnostic_end,
                    "ticks_per_generation": ticks_per_generation,
                },
                "acs_checkpoint_slots": int(args.acs_checkpoint_slots),
                "generation_turnover": (
                    "host_authoritative_transaction"
                    if args.generation_turnover
                    else "native_lifecycle"
                ),
                "determinism_check_performed": False,
                "terminal_projection_digests": {"single": list(single_digests)},
                "process_rng_traces": {"single": single_rng_trace},
                "content_alias_enabled": not args.disable_content_alias,
                "production_execution_admission": False,
                "qualification_claim": False,
                "secondary_boundary_timing": secondary_boundary_timing_receipt(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(
                output_root / "single_tick_diagnostic_receipt.json",
                receipt,
            )
            return 0

        if args.repeat_diagnostic_end_generation is not None:
            diagnostic_end = args.repeat_diagnostic_end_generation
            record(
                f"repeat_tick0_to_tick{diagnostic_end * ticks_per_generation}_first_started"
            )
            first, _, first_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0,
                end=diagnostic_end,
                emit_checkpoint=False,
                sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                    tick_native_control,
                    start_generation=0,
                    end_generation=diagnostic_end,
                    ticks_per_generation=ticks_per_generation,
                ),
            )
            first_signatures = _projection_signatures(first)
            first_digests = _terminal_digests(first)
            first = ()
            gc.collect()
            record(
                f"repeat_tick0_to_tick{diagnostic_end * ticks_per_generation}_second_started"
            )
            second, _, second_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0,
                end=diagnostic_end,
                emit_checkpoint=False,
                sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                    tick_native_control,
                    start_generation=0,
                    end_generation=diagnostic_end,
                    ticks_per_generation=ticks_per_generation,
                ),
            )
            second_digests = _terminal_digests(second)
            difference = _projection_difference_summary(first_signatures, second)
            atomic_write_json(
                output_root / "repeat_segment_projection_difference.json",
                difference,
            )
            rng_traces = {"first": first_rng_trace, "second": second_rng_trace}
            if first_digests != second_digests or difference["mismatch_count"]:
                raise AllocationDiagnosticError(
                    f"HAPB repeated fresh G0 to G{diagnostic_end} segments differ"
                )
            record(
                f"repeat_tick0_to_tick{diagnostic_end * ticks_per_generation}_exact"
            )
            receipt = {
                "schema_version": f"{SCHEMA_VERSION}_repeat_segment_diagnostic_v1",
                "authority": "hapb_b48_repeat_segment_diagnostic_v1",
                "host_harness_source_commit": _head(),
                "artifact_source_commit": manifest_payload["committed_source_snapshot"]["build_source_commit"],
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "reused_generation_zero_source_root": str(source_root),
                "physical_cells": physical_cells,
                "tile_rows": args.tile_rows,
                "time_authority": "absolute_tick_v1",
                "start_tick": 0,
                "end_tick": diagnostic_end * ticks_per_generation,
                "total_ticks": diagnostic_end * ticks_per_generation,
                "legacy_compatibility": {
                    "start_generation": 0,
                    "end_generation": diagnostic_end,
                    "ticks_per_generation": ticks_per_generation,
                },
                "repeat_count": 2,
                "terminal_projection_digests_exact": True,
                "mismatch_count": difference["mismatch_count"],
                "terminal_projection_digests": {
                    "first": list(first_digests),
                    "second": list(second_digests),
                },
                "process_rng_traces": rng_traces,
                "content_alias_enabled": not args.disable_content_alias,
                "production_execution_admission": False,
                "qualification_claim": False,
                "secondary_boundary_timing": secondary_boundary_timing_receipt(),
            }
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            atomic_write_json(
                output_root / "repeat_segment_diagnostic_receipt.json", receipt
            )
            return 0

        from native_runtime.checkpoint_digests import (
            checkpoint_source_projection_digest_manifest,
        )

        def _action_cache_digest_map_v1(
            projections: Sequence[Mapping[str, Any]],
        ) -> dict[str, str]:
            if len(projections) != len(cell_ids):
                raise AllocationDiagnosticError(
                    "HAPB action-cache observation does not contain one projection per cell"
                )
            return {
                str(cell_id): str(
                    checkpoint_source_projection_digest_manifest(projection)[
                        "action_cache"
                    ]["digest"]
                )
                for cell_id, projection in zip(cell_ids, projections)
            }

        record("reference_started")
        reference, _, reference_rng_trace = execute(
            _load_reused_sources(source_root, cell_ids=cell_ids),
            start=0, end=args.generations,
            absolute_tick_start=0,
            absolute_tick_end=tick_native_control.total_ticks,
            sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                tick_native_control,
                start_tick=0,
                end_tick=tick_native_control.total_ticks,
            ),
        )
        reference_digests = _terminal_digests(reference)
        reference_signatures = _projection_signatures(reference)
        reference_action_cache_digests = _action_cache_digest_map_v1(reference)
        reference = ()
        gc.collect()
        record("reference_digest_complete")

        midpoint_digests: list[str] | None = None
        midpoint_action_cache_digests: dict[str, str] | None = None
        if resume_checkpoint is None:
            midpoint, envelope, midpoint_rng_trace = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0, end=args.split_generations,
                absolute_tick_start=0,
                absolute_tick_end=split_tick,
                sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                    tick_native_control,
                    start_tick=0,
                    end_tick=split_tick,
                ),
            )
            if envelope is None:
                raise AllocationDiagnosticError("HAPB midpoint emitted no checkpoint envelope")
            midpoint_digests = _terminal_digests(midpoint)
            midpoint_action_cache_digests = _action_cache_digest_map_v1(midpoint)
            midpoint = ()
            checkpoint_path = output_root / "hapb_serialized_midpoint.pt"
            persist_started_ns = perf_counter_ns()
            if args.acs_checkpoint_slots:
                acs_persist = persist_hapb_midpoint_via_acs_v1(
                    envelope,
                    checkpoint_path,
                    slots=args.acs_checkpoint_slots,
                    compute_frontier=args.split_generations,
                    save=torch.save,
                )
                atomic_write_json(
                    output_root / "hapb_midpoint_acs_persist_receipt.json",
                    acs_persist,
                )
            else:
                torch.save(envelope, checkpoint_path)
            checkpoint_persist_wall_ns = perf_counter_ns() - persist_started_ns
        else:
            checkpoint_path = resume_checkpoint
            midpoint_rng_trace = {"checkpoint_reused": True}
        checkpoint_sha256 = _sha256(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
        record(
            "midpoint_reused_owner_released"
            if resume_checkpoint is not None
            else "midpoint_serialized_owner_released"
        )
        loaded_envelope = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored_sources = restore_direct_resident_wdl_jungle_hapb_checkpoint_sources_v1(loaded_envelope)
        reference_midpoint_difference: Mapping[str, Any] | None = None
        reference_midpoint_digests: list[str] | None = None
        reference_midpoint_action_cache_digests: dict[str, str] | None = None
        if args.compare_reference_midpoint:
            reference_midpoint, _, _ = execute(
                _load_reused_sources(source_root, cell_ids=cell_ids),
                start=0, end=args.split_generations,
                absolute_tick_start=0,
                absolute_tick_end=split_tick,
                emit_checkpoint=False,
                sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                    tick_native_control,
                    start_tick=0,
                    end_tick=split_tick,
                ),
            )
            reference_midpoint_digests = _terminal_digests(reference_midpoint)
            reference_midpoint_action_cache_digests = _action_cache_digest_map_v1(
                reference_midpoint
            )
            reference_midpoint_difference = _prepared_restore_difference_summary(
                reference_midpoint, restored_sources,
            )
            reference_midpoint = ()
            atomic_write_json(
                output_root / "reference_midpoint_projection_difference.json",
                reference_midpoint_difference,
            )
            if reference_midpoint_difference["mismatch_count"]:
                raise AllocationDiagnosticError(
                    "uninterrupted HAPB G3 state differs from the resume checkpoint"
                )
        component = {
            "schema_version": HAPB_DIRECT_COMPONENT_CHECKPOINT_SCHEMA_V1,
            "time_authority": "absolute_tick_v1",
            "absolute_tick": split_tick,
            "next_absolute_tick": split_tick,
            "cell_ids": list(cell_ids),
            "hapb_policy_bank": loaded_envelope["hapb_policy_bank"],
            "hapb_policy_window_identity": loaded_envelope.get(
                "hapb_policy_window_identity"
            ),
        }
        record("fresh_owner_restore_started")
        restore_boundary_difference = _checkpoint_restore_difference_summary(
            loaded_envelope, restored_sources,
        )
        atomic_write_json(
            output_root / "checkpoint_restore_projection_difference.json",
            restore_boundary_difference,
        )
        if restore_boundary_difference["mismatch_count"]:
            raise AllocationDiagnosticError(
                "HAPB checkpoint restore does not reconstruct its G3 source boundary"
            )
        loaded_envelope = None
        resumed, _, resumed_rng_trace = execute(
            restored_sources, start=resume_adapter_start, end=args.generations,
            component_checkpoint=component,
            expected_restore_sources=restored_sources,
            absolute_tick_start=split_tick,
            absolute_tick_end=tick_native_control.total_ticks,
            sequential_tick_spans=_h4_tick_dispatch_spans_v1(
                tick_native_control,
                start_tick=split_tick,
                end_tick=tick_native_control.total_ticks,
            ),
        )
        resumed_digests = _terminal_digests(resumed)
        resumed_action_cache_digests = _action_cache_digest_map_v1(resumed)
        if resumed_digests != reference_digests:
            atomic_write_json(
                output_root / "continuation_projection_difference.json",
                _projection_difference_summary(reference_signatures, resumed),
            )
            rng_traces = {
                "reference": reference_rng_trace,
                "midpoint": midpoint_rng_trace,
                "resumed": resumed_rng_trace,
            }
            raise AllocationDiagnosticError("HAPB fresh-owner terminal projection digests differ")
        record("continuation_exact")
        def _cell_digest_map_v1(
            digests: Sequence[str] | None,
        ) -> dict[str, str] | None:
            if digests is None:
                return None
            if len(digests) != len(cell_ids):
                raise AllocationDiagnosticError(
                    "HAPB exactness receipt does not contain one projection per cell"
                )
            return {
                str(cell_id): str(digest)
                for cell_id, digest in zip(cell_ids, digests)
            }

        process_rng_traces = {
            "reference": reference_rng_trace,
            "midpoint": midpoint_rng_trace,
            "resumed": resumed_rng_trace,
        }
        policy_traffic_observation = [
            {
                "label": sample.get("label"),
                "counters": sample.get("counters"),
                "frag_identity_accounting": sample.get("frag_identity_accounting"),
                "same_owner_entry_results": sample.get("same_owner_entry_results"),
            }
            for sample in boundary_timing_samples
        ]
        action_cache_observation = {
            "reference_terminal": reference_action_cache_digests,
            "midpoint": midpoint_action_cache_digests,
            "reference_midpoint": reference_midpoint_action_cache_digests,
            "resumed_terminal": resumed_action_cache_digests,
        }
        action_cache_observed = all(
            isinstance(value, Mapping) and len(value) == physical_cells
            for value in action_cache_observation.values()
        )
        compatibility_aliases_absent = all(
            field not in execution_controls
            for field in (
                "generations",
                "ticks_per_generation",
                "physical_window_generations",
            )
        )
        exactness_contract = {
            "schema_version": "eiel_hapb_h4_exactness_contract_v1",
            "cell_count": physical_cells,
            "cell_ids": [str(cell_id) for cell_id in cell_ids],
            "projections": {
                "reference_terminal": _cell_digest_map_v1(reference_digests),
                "midpoint": _cell_digest_map_v1(midpoint_digests),
                "reference_midpoint": _cell_digest_map_v1(reference_midpoint_digests),
                "resumed_terminal": _cell_digest_map_v1(resumed_digests),
            },
            "identity": {
                "host_harness_source_commit": _head(),
                "artifact_source_commit": manifest_payload["committed_source_snapshot"]["build_source_commit"],
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "execution_route_identity": execution_controls["execution_route_identity"],
                "tick_native_control_sha256": canonical_sha256(tick_native_control.payload()),
            },
            "observations": {
                "policy_traffic_sha256": canonical_sha256(policy_traffic_observation),
                "rng_trace_sha256": canonical_sha256(process_rng_traces),
                "action_cache_sha256": (
                    canonical_sha256(action_cache_observation)
                    if action_cache_observed
                    else None
                ),
            },
            "checks": {
                "cell_count_exact": physical_cells == 48,
                "midpoint_reference_exact": (
                    reference_midpoint_difference is not None
                    and reference_midpoint_difference["mismatch_count"] == 0
                ),
                "checkpoint_restore_exact": (
                    restore_boundary_difference["mismatch_count"] == 0
                ),
                "fresh_owner_restore": True,
                "continuation_exact": True,
                "compatibility_alias_deletion_proof": compatibility_aliases_absent,
                "action_cache_observed": action_cache_observed,
            },
        }
        secondary_timing = secondary_boundary_timing_receipt()
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "authority": "hapb_b48_fresh_owner_continuation_development_v1",
            "host_harness_source_commit": _head(),
            "artifact_source_commit": manifest_payload["committed_source_snapshot"]["build_source_commit"],
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": manifest_sha256,
            "reused_generation_zero_source_root": str(source_root),
            "reused_generation_zero_source_receipt_sha256": _sha256(source_root / "source_factory_receipt.json"),
            "physical_cells": physical_cells,
            "tile_rows": args.tile_rows,
            "time_authority": "absolute_tick_v1",
            "total_ticks": tick_native_control.total_ticks,
            "checkpoint_ticks": list(tick_native_control.resolved_checkpoint_ticks),
            "tick_native_control": tick_native_control.payload(),
            "checkpoint_next_absolute_tick": split_tick,
            "cell_ids": list(cell_ids),
            "h4_exactness_contract": exactness_contract,
            "generation_turnover": (
                "host_authoritative_transaction"
                if args.generation_turnover
                else "not_enabled_in_this_probe"
            ),
            "hapb_host_authoritative_generation_turnover": args.generation_turnover,
            "hapb_host_authoritative_ordinary_birth": True,
            "content_alias_enabled": not args.disable_content_alias,
            "midpoint_checkpoint_path": str(checkpoint_path),
            "midpoint_checkpoint_sha256": checkpoint_sha256,
            "midpoint_checkpoint_bytes": checkpoint_bytes,
            "midpoint_checkpoint_reused": resume_checkpoint is not None,
            "acs_checkpoint_slots": int(args.acs_checkpoint_slots),
            "reference_terminal_projection_digests": list(reference_digests),
            "resumed_terminal_projection_digests": list(resumed_digests),
            "fresh_owner_restore": True,
            "continuation_exact": True,
            "checkpoint_restore_boundary_mismatch_count": (
                restore_boundary_difference["mismatch_count"]
            ),
            "reference_midpoint_boundary_mismatch_count": (
                None if reference_midpoint_difference is None
                else reference_midpoint_difference["mismatch_count"]
            ),
            "process_rng_traces": process_rng_traces,
            "production_execution_admission": False,
            "exclusive_cuda_admission": exclusive_cuda_admission,
            "memory": hapb_memory_admission_from_samples_v1(memory_samples),
            "host": hapb_host_admission_from_samples_v1(host_samples),
            "checkpoint_persist_wall_ns": locals().get("checkpoint_persist_wall_ns"),
            "qualification_claim": False,
            "secondary_boundary_timing": secondary_timing,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_write_json(output_root / "hapb_continuation_receipt.json", receipt)
        return 0
    except BaseException as exc:
        atomic_write_json(output_root / "continuation_failure_diagnostic.json", {
            "schema_version": SCHEMA_VERSION,
            "authority": "hapb_b48_fresh_owner_continuation_failure_v1",
            "error": f"{type(exc).__name__}: {exc}",
            "stage": json.loads(progress.read_text(encoding="utf-8")).get("stage"),
            "process_rng_traces": locals().get("rng_traces", {
                "reference": locals().get("reference_rng_trace"),
                "midpoint": locals().get("midpoint_rng_trace"),
                "resumed": locals().get("resumed_rng_trace"),
            }),
        })
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
