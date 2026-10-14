from __future__ import annotations

import csv
from functools import wraps
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .engine import NativeEielEngine
from .lifecycle_events import (
    FIELD_DTYPES,
    complete_reference_typed_lifecycle_events,
    continue_typed_lifecycle_event_accumulator,
    empty_typed_lifecycle_event_tensors,
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_projection,
    validated_typed_lifecycle_event_cursor,
    write_typed_lifecycle_public_outputs,
)
from .safe_default import NativeSafeDefaultDecision, resolve_native_safe_default
from .types import NativeGenerationResult


_BEHAVIOR_CSV_FIELDS = [
    "generation",
    "agent_id",
    "cohort",
    "move_n",
    "move_s",
    "move_e",
    "move_w",
    "gather",
    "rest",
    "attack",
    "lifespan_steps",
    "energy_collected",
    "offspring",
    "attacks_made",
    "fitness",
    "alive_end",
    "eligible_for_selection",
    "kept_next",
    "injury_end",
    "injury_delta",
    "aggression_bias",
]

_CHECKPOINT_TENSOR_OVERRIDES_ATTR = "_cxxbackbone10_checkpoint_tensor_overrides"
_CHECKPOINT_METADATA_OVERRIDES_ATTR = "_cxxbackbone10_checkpoint_metadata_overrides"
_ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_ATTR = "_cxxbackbone10_reference_lifecycle_event_accumulator"
_ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_CURSOR_ATTR = "_typed_lifecycle_event_output_accumulator_cursor"
_REFERENCE_LIFECYCLE_IDENTITY_REGISTRY_ATTR = "_reference_typed_lifecycle_identity_registry"
_LEGACY_EVENT_TAPE_FIELDS = (
    "event_tape_type",
    "event_tape_generation",
    "event_tape_tick",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
)
_REQUIRED_NATIVE_RESULT_FIELDS = (
    "behavior_rows",
    "native_tick_loop_count",
    "alive_count",
    "capacity",
    "free_slots",
    "typed_event_rows",
)


def native_runtime_enabled(runner: Any) -> bool:
    return bool(getattr(getattr(runner, "sim_cfg", None), "native_runtime_enable", False))


def native_runtime_fail_closed(runner: Any) -> bool:
    return bool(getattr(getattr(runner, "sim_cfg", None), "native_runtime_fail_closed", True))


def validate_native_runtime_supported(runner: Any) -> NativeSafeDefaultDecision:
    """Fail closed for surfaces outside the current CPU native foundation envelope."""

    sim_cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    decision = resolve_native_safe_default(sim_cfg, env_cfg, runner)
    try:
        setattr(runner, "_native_safe_default_decision", decision.to_dict())
    except Exception:
        pass
    if not decision.native_supported or not decision.native_execution_active:
        joined = ",".join(decision.unsupported_reasons)
        raise RuntimeError(f"EIEL-NATIVE-RUNTIME-UNSUPPORTED: {joined}")
    return decision


def record_native_runtime_failure(
    runner: Any,
    exc: BaseException,
    *,
    stage: str,
    quarantine_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    existing = getattr(runner, "_native_runtime_failure", None)
    first_cause = (
        str(existing.get("first_cause", ""))
        if isinstance(existing, Mapping) and existing.get("first_cause")
        else repr(exc)
    )
    existing_quarantine = tuple(existing.get("quarantine_paths", ())) if isinstance(existing, Mapping) else ()
    merged_quarantine = tuple(dict.fromkeys((*existing_quarantine, *quarantine_paths)))
    payload = {
        "schema_version": "eiel_native_runtime_failure_v1",
        "status": "failed",
        "ok": False,
        "promotion_eligible": False,
        "native_runtime_active": True,
        "stage": str(stage),
        "first_cause": first_cause,
        "exception": repr(exc),
        "partial_outputs_quarantined": bool(merged_quarantine),
        "quarantine_paths": list(merged_quarantine),
    }
    setattr(runner, "_native_runtime_failure", dict(payload))
    summary = getattr(runner, "_native_runtime_summary", None)
    failed_summary = dict(summary) if isinstance(summary, Mapping) else {}
    failed_summary.update(payload)
    setattr(runner, "_native_runtime_summary", failed_summary)

    run_dir = str(getattr(runner, "run_dir", "") or getattr(runner, "run_dir_path", "") or "")
    if not run_dir:
        return payload
    marker = Path(run_dir) / "native_runtime_failure.json"
    tmp = marker.with_name(marker.name + ".tmp")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, marker)
        directory_fd = os.open(str(marker.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception as marker_exc:
        payload["failure_marker_error"] = repr(marker_exc)
        setattr(runner, "_native_runtime_failure", dict(payload))
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    return payload


def _native_result_failure(message: str) -> RuntimeError:
    return RuntimeError(f"EIEL-NATIVE-GENERATION-FAILED: {message}")


def validate_native_generation_result(
    runner: Any,
    result: NativeGenerationResult,
    engine: NativeEielEngine,
    *,
    generation_index: int,
    steps_per_generation: int,
) -> dict[str, Any]:
    raw = result.raw_result
    if not result.executed or not bool(raw.get("executed", False)):
        raise _native_result_failure("executed=false")
    if bool(raw.get("hidden_fallback", False)):
        raise _native_result_failure("hidden_fallback=true")
    if "ok" in raw and raw.get("ok") is not True:
        raise _native_result_failure("ok=false")
    if "status" in raw and str(raw.get("status", "") or "").strip().lower() != "passed":
        raise _native_result_failure(f"status={raw.get('status')!r}")
    for marker in (
        "generation_complete",
        "finalization_complete",
        "finalized",
        "required_outputs_present",
        "output_flush_complete",
        "extension_hash_verified",
    ):
        if marker in raw and raw.get(marker) is not True:
            raise _native_result_failure(f"{marker}=false")
    if raw.get("native_owner") is not True or raw.get("single_native_generation_call") is not True:
        raise _native_result_failure("inconsistent native owner identity")
    if bool(raw.get("mixed_ownership", False)) or bool(raw.get("python_agent_sync", False)):
        raise _native_result_failure("mixed Python/native ownership")
    if bool(raw.get("runtime_jit", False)) or result.capability.runtime_jit_in_backend:
        raise _native_result_failure("runtime JIT detected")
    if int(result.generation_index) != int(generation_index):
        raise _native_result_failure("generation index mismatch")
    if int(result.steps_per_generation) != int(steps_per_generation):
        raise _native_result_failure("steps-per-generation mismatch")
    if int(result.native_tick_loop_count) != int(steps_per_generation):
        raise _native_result_failure("incomplete generation tick count")
    missing_outputs = [name for name in _REQUIRED_NATIVE_RESULT_FIELDS if name not in raw]
    if missing_outputs:
        raise _native_result_failure(f"required native outputs missing: {missing_outputs}")

    decision = resolve_native_safe_default(runner.sim_cfg, runner.env_cfg, runner)
    if not (
        decision.native_execution_active
        and decision.extension_loaded
        and decision.extension_abi_compatible
        and decision.extension_hash_verified
        and not decision.extension_runtime_jit
    ):
        raise _native_result_failure("extension identity no longer qualified")
    if (
        result.capability.extension_abi_version != decision.extension_abi_version
        or result.capability.extension_version != decision.extension_version
    ):
        raise _native_result_failure("engine extension ABI/version identity mismatch")
    expected_benchmark = "qualified" if decision.benchmark_qualified else "not_qualified"
    if not (
        result.capability.extension_abi_compatible
        and result.capability.production_jit_free
        and result.capability.persistent_engine
        and result.capability.supports_flat_generation
        and result.capability.reporting_support == decision.reporting_qualification
        and result.capability.benchmark_qualification == expected_benchmark
        and result.capability.accepted_default_native_envelope == decision.lifecycle_default_active
    ):
        raise _native_result_failure("malformed or contradictory native capability response")

    projection = engine.export_checkpoint_projection()
    metadata = dict(projection.get("metadata") or {})
    try:
        checkpoint_generation = int(metadata.get("generation", -1))
        checkpoint_tick = int(metadata.get("tick", -1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise _native_result_failure("malformed final checkpoint metadata") from exc
    checkpoint_phase = str(metadata.get("agent_grid_runner_phase", "") or "").strip().lower()
    if (
        checkpoint_generation != int(generation_index) + 1
        or checkpoint_tick != 0
        or checkpoint_phase != "generation_boundary"
    ):
        raise _native_result_failure(
            "partial or unfinalized checkpoint projection: "
            f"generation={checkpoint_generation} tick={checkpoint_tick} phase={checkpoint_phase!r}"
        )

    validation = {
        "schema_version": "eiel_native_generation_validation_v1",
        "ok": True,
        "status": "passed",
        "generation_complete": True,
        "finalization_complete": True,
        "required_outputs_present": True,
        "native_owner": True,
        "mixed_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
        "extension_path": decision.extension_path,
        "extension_sha256": decision.extension_sha256,
        "extension_hash_verified": True,
        "extension_abi_version": decision.extension_abi_version,
        "extension_version": decision.extension_version,
        "checkpoint_generation": checkpoint_generation,
        "checkpoint_tick": checkpoint_tick,
        "checkpoint_phase": checkpoint_phase,
        "public_output_flush_pending": True,
    }
    raw["native_result_validation"] = dict(validation)
    result.compact_result["native_result_validation"] = dict(validation)
    return validation


def get_or_create_native_engine(
    runner: Any,
    *,
    generation_index: int,
    safe_default_decision: NativeSafeDefaultDecision | None = None,
) -> NativeEielEngine:
    engine = getattr(runner, "_native_eiel_engine", None)
    if isinstance(engine, NativeEielEngine):
        decision = safe_default_decision or validate_native_runtime_supported(runner)
        engine.bind_qualified_runner(runner, decision)
        return engine
    build_dir = str(getattr(getattr(runner, "sim_cfg", None), "native_runtime_build_dir", "") or "") or None
    engine = NativeEielEngine.from_runner(
        runner,
        generation=int(generation_index),
        tick=0,
        build_directory=build_dir,
    )
    setattr(runner, "_native_eiel_engine", engine)
    return engine


def _emit_typed_lifecycle_public_outputs(
    runner: Any,
    events: dict[str, Any],
    *,
    jsonl_name: str,
    csv_name: str,
    cursor_attr: str,
    summary_attr: str,
    schema_version: str,
    source: str,
) -> dict[str, Any]:
    total_rows = int(events["event_type"].shape[0])
    metrics_dir = Path(str(getattr(runner, "run_dir", "") or ".")) / "metrics"
    jsonl_path = metrics_dir / jsonl_name
    csv_path = metrics_dir / csv_name
    cursor_value = getattr(runner, cursor_attr, None)
    cursor_path_attr = f"{cursor_attr}_path"
    cursor_path = getattr(runner, cursor_path_attr, None)
    output_path_key = (str(jsonl_path.resolve()), str(csv_path.resolve()))
    path_scoped_cursor = cursor_value if cursor_path == output_path_key else None
    snapshots = {
        path: (path.exists(), path.stat().st_size if path.exists() else 0)
        for path in (jsonl_path, csv_path)
    }
    try:
        output = write_typed_lifecycle_public_outputs(
            jsonl_path,
            csv_path,
            events,
            cursor=int(path_scoped_cursor) if path_scoped_cursor is not None else None,
        )
        for path in (jsonl_path, csv_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    except Exception as exc:
        quarantine_paths: list[str] = []
        for path, (existed, original_size) in snapshots.items():
            try:
                if existed and path.exists():
                    with path.open("r+b") as handle:
                        handle.truncate(original_size)
                        handle.flush()
                        os.fsync(handle.fileno())
                elif path.exists():
                    quarantine = path.with_name(path.name + ".incomplete")
                    os.replace(path, quarantine)
                    quarantine_paths.append(str(quarantine))
            except Exception as quarantine_exc:
                quarantine_paths.append(f"quarantine_failed:{path}:{quarantine_exc!r}")
        decision = getattr(runner, "_native_safe_default_decision", None)
        if isinstance(decision, Mapping) and bool(decision.get("native_execution_active", False)):
            record_native_runtime_failure(
                runner,
                exc,
                stage="typed_lifecycle_public_output",
                quarantine_paths=tuple(quarantine_paths),
            )
        raise
    next_cursor = int(output["end_cursor"])
    summary = {
        "schema_version": schema_version,
        "source": source,
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "cursor_source": "runner_attr" if path_scoped_cursor is not None else str(output["cursor_source"]),
        "start_cursor": int(output["start_cursor"]),
        "end_cursor": int(next_cursor),
        "rows_written": int(output["rows_written"]),
        "total_event_rows": int(total_rows),
        "typed_lifecycle_event_digest": typed_lifecycle_event_digest(events),
        "jsonl_existed": bool(output["jsonl_existed"]),
        "csv_existed": bool(output["csv_existed"]),
        "flush_complete": bool(jsonl_path.is_file() and csv_path.is_file()),
    }
    if not summary["flush_complete"]:
        raise RuntimeError("typed lifecycle public output flush did not produce both required files")
    setattr(runner, cursor_attr, int(next_cursor))
    setattr(runner, cursor_path_attr, output_path_key)
    setattr(runner, summary_attr, dict(summary))
    return summary


def _unprefix_typed_lifecycle_events(
    source: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any] | None:
    prefix = "typed_lifecycle_event_"
    present = [name for name in FIELD_DTYPES if hasattr(source.get(f"{prefix}{name}"), "detach")]
    if not present:
        return None
    if len(present) != len(FIELD_DTYPES):
        missing = sorted(set(FIELD_DTYPES) - set(present))
        raise RuntimeError(f"{label} has an incomplete typed lifecycle event tensor set: {missing}")
    return {name: source[f"{prefix}{name}"] for name in FIELD_DTYPES}


def _reference_checkpoint_lifecycle_boundary(
    runner: Any,
) -> tuple[dict[str, Any] | None, int]:
    overrides = getattr(runner, _CHECKPOINT_TENSOR_OVERRIDES_ATTR, None)
    if not isinstance(overrides, Mapping):
        return None, 0

    boundary = _unprefix_typed_lifecycle_events(overrides, label="reference checkpoint boundary")
    if boundary is None:
        return None, 0

    checkpoint_rows = int(boundary["event_type"].shape[0])
    legacy_present = [name for name in _LEGACY_EVENT_TAPE_FIELDS if hasattr(overrides.get(name), "detach")]
    if legacy_present and len(legacy_present) != len(_LEGACY_EVENT_TAPE_FIELDS):
        missing = sorted(set(_LEGACY_EVENT_TAPE_FIELDS) - set(legacy_present))
        raise RuntimeError(f"reference checkpoint boundary has an incomplete legacy event tape: {missing}")
    if legacy_present:
        legacy_lengths = {int(overrides[name].shape[0]) for name in _LEGACY_EVENT_TAPE_FIELDS}
        if legacy_lengths != {checkpoint_rows}:
            raise RuntimeError(
                "typed lifecycle checkpoint cursor contradiction: "
                f"typed_rows={checkpoint_rows} legacy_rows={sorted(legacy_lengths)}"
            )

    metadata_overrides = getattr(runner, _CHECKPOINT_METADATA_OVERRIDES_ATTR, None)
    recorded_cursor = None
    if isinstance(metadata_overrides, Mapping) and "event_tape_cursor" in metadata_overrides:
        recorded_cursor = metadata_overrides["event_tape_cursor"]
    checkpoint_cursor = checkpoint_rows if recorded_cursor is None else recorded_cursor
    return boundary, checkpoint_cursor


def _typed_lifecycle_prefix_matches(
    events: Mapping[str, Any],
    prefix: Mapping[str, Any],
) -> bool:
    event_rows = int(events["event_type"].shape[0])
    prefix_rows = int(prefix["event_type"].shape[0])
    if event_rows < prefix_rows:
        return False
    return all(
        torch.equal(
            events[name][:prefix_rows].detach().cpu().to(dtype=dtype),
            prefix[name].detach().cpu().to(dtype=dtype),
        )
        for name, dtype in FIELD_DTYPES.items()
    )


def _canonical_reference_lifecycle_events(runner: Any) -> dict[str, Any] | None:
    cursor = getattr(runner, _ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_CURSOR_ATTR, None)
    if cursor is None:
        return None
    accumulated = getattr(runner, _ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_ATTR, None)
    if not isinstance(accumulated, Mapping):
        return None
    events = _unprefix_typed_lifecycle_events(accumulated, label="canonical reference lifecycle tape")
    if events is None:
        return None
    cursor_i = int(cursor)
    rows = int(events["event_type"].shape[0])
    if cursor_i < 0 or cursor_i > rows:
        raise RuntimeError(
            "canonical reference lifecycle cursor is outside the accumulated tape: "
            f"cursor={cursor_i} rows={rows}"
        )
    return {
        name: events[name][:cursor_i].detach().cpu().to(dtype=dtype).contiguous()
        for name, dtype in FIELD_DTYPES.items()
    }


def _reference_typed_lifecycle_completion_required(runner: Any) -> bool:
    sim_cfg = getattr(runner, "sim_cfg", None)
    if sim_cfg is None or not getattr(sim_cfg, "native_runtime_cxx26_scope_lock_enable", False):
        return False
    native_summary = getattr(runner, "_native_runtime_summary", None)
    if isinstance(native_summary, Mapping) and str(native_summary.get("decision", "")) == (
        "EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION"
    ):
        return False
    return True


def _agent_lifecycle_identities(agents: Mapping[Any, Any]) -> dict[int, dict[str, int]]:
    identities: dict[int, dict[str, int]] = {}
    for raw_agent_id, agent in agents.items():
        agent_id = int(getattr(agent, "agent_id", raw_agent_id))
        stats = getattr(agent, "stats", None)
        parent_ids = tuple(getattr(stats, "parent_ids", ()) or ()) if stats is not None else ()
        identities[agent_id] = {
            "slot": int(getattr(agent, "slot", -1)),
            "lineage_founder_id": int(getattr(stats, "founder_id", -1) if stats is not None else -1),
            "lineage_depth": int(getattr(stats, "lineage_depth", -1) if stats is not None else -1),
            "parent_agent_id": int(parent_ids[0]) if parent_ids else -1,
        }
    return identities


def _capture_reference_agent_lifecycle_identities(runner: Any) -> None:
    population = getattr(runner, "pop", None)
    agents = getattr(population, "agents", None)
    if not isinstance(agents, Mapping):
        raise RuntimeError("authoritative reference lifecycle capture requires pop.agents")
    registry = getattr(runner, _REFERENCE_LIFECYCLE_IDENTITY_REGISTRY_ATTR, None)
    merged = dict(registry) if isinstance(registry, Mapping) else {}
    merged.update(_agent_lifecycle_identities(agents))
    setattr(runner, _REFERENCE_LIFECYCLE_IDENTITY_REGISTRY_ATTR, merged)


def _reference_agent_lifecycle_identities(runner: Any) -> dict[int, dict[str, int]]:
    registry = getattr(runner, _REFERENCE_LIFECYCLE_IDENTITY_REGISTRY_ATTR, None)
    identities = dict(registry) if isinstance(registry, Mapping) else {}
    population = getattr(runner, "pop", None)
    agents = getattr(population, "agents", None)
    if isinstance(agents, Mapping):
        identities.update(_agent_lifecycle_identities(agents))
    if not identities:
        raise RuntimeError("authoritative reference lifecycle completion has no captured identities")
    return identities


def _complete_reference_lifecycle_accumulator_before_turnover(runner: Any) -> None:
    from perf.cxxbackbone10_resident_state_owner import (  # noqa: WPS433
        _append_reference_lifecycle_segment,
    )

    _append_reference_lifecycle_segment(runner)
    accumulated = getattr(runner, _ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_ATTR, None)
    if not isinstance(accumulated, Mapping):
        raise RuntimeError("reference lifecycle segment finalization did not create an accumulator")
    events = _unprefix_typed_lifecycle_events(
        accumulated,
        label="pre-turnover reference lifecycle accumulator",
    )
    if events is None:
        return
    completed = complete_reference_typed_lifecycle_events(
        events,
        _reference_agent_lifecycle_identities(runner),
        require_complete=True,
    )
    setattr(
        runner,
        _ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_ATTR,
        {
            f"typed_lifecycle_event_{name}": completed[name].detach().cpu().clone()
            for name in FIELD_DTYPES
        },
    )


def install_reference_typed_lifecycle_projection_hook() -> bool:
    from simulation.runner_pkg.runner import SimulationRunner  # noqa: WPS433

    current = getattr(SimulationRunner, "_run_ticks")
    if bool(getattr(current, "_eiel_reference_typed_lifecycle_projection_hook", False)):
        return False

    @wraps(current)
    def run_ticks_with_reference_lifecycle_identity_capture(runner: Any, *args: Any, **kwargs: Any) -> Any:
        delegate = getattr(
            run_ticks_with_reference_lifecycle_identity_capture,
            "_eiel_reference_typed_lifecycle_projection_delegate",
        )
        if bool(getattr(delegate, "_eiel_reference_committed_death_oracle_hook", False)):
            compatibility_delegate = getattr(
                run_ticks_with_reference_lifecycle_identity_capture,
                "_eiel_reference_run_delegate",
                None,
            )
            if compatibility_delegate is not None:
                setattr(delegate, "_eiel_reference_run_delegate", compatibility_delegate)
        result = delegate(runner, *args, **kwargs)
        if _reference_typed_lifecycle_completion_required(runner):
            _capture_reference_agent_lifecycle_identities(runner)
            _complete_reference_lifecycle_accumulator_before_turnover(runner)
        return result

    setattr(
        run_ticks_with_reference_lifecycle_identity_capture,
        "_eiel_reference_typed_lifecycle_projection_delegate",
        current,
    )
    setattr(
        run_ticks_with_reference_lifecycle_identity_capture,
        "_eiel_reference_typed_lifecycle_projection_hook",
        True,
    )
    SimulationRunner._run_ticks = run_ticks_with_reference_lifecycle_identity_capture
    return True


def _consume_reference_checkpoint_lifecycle_boundary(runner: Any) -> None:
    overrides = getattr(runner, _CHECKPOINT_TENSOR_OVERRIDES_ATTR, None)
    if isinstance(overrides, Mapping):
        kept = {
            name: value
            for name, value in overrides.items()
            if name not in _LEGACY_EVENT_TAPE_FIELDS and not name.startswith("typed_lifecycle_event_")
        }
        setattr(runner, _CHECKPOINT_TENSOR_OVERRIDES_ATTR, kept)
    metadata = getattr(runner, _CHECKPOINT_METADATA_OVERRIDES_ATTR, None)
    if isinstance(metadata, Mapping):
        lifecycle_metadata = {
            "event_tape_active_count",
            "event_tape_cursor",
            "typed_lifecycle_event_active_count",
            "typed_lifecycle_event_schema_version",
            "typed_lifecycle_event_source",
        }
        setattr(
            runner,
            _CHECKPOINT_METADATA_OVERRIDES_ATTR,
            {name: value for name, value in metadata.items() if name not in lifecycle_metadata},
        )


def _install_canonical_reference_lifecycle_events(
    runner: Any,
    events: Mapping[str, Any],
) -> None:
    event_rows = int(events["event_type"].shape[0])
    setattr(
        runner,
        _ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_ATTR,
        {
            f"typed_lifecycle_event_{name}": events[name].detach().cpu().clone()
            for name in FIELD_DTYPES
        },
    )
    setattr(runner, _ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_CURSOR_ATTR, event_rows)


def _continue_reference_typed_lifecycle_events(
    runner: Any,
    reference_events: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    boundary, checkpoint_cursor = _reference_checkpoint_lifecycle_boundary(runner)
    canonical = _canonical_reference_lifecycle_events(runner)
    if boundary is not None:
        if _typed_lifecycle_prefix_matches(reference_events, boundary):
            continued = dict(reference_events)
        else:
            continued = continue_typed_lifecycle_event_accumulator(
                boundary,
                reference_events,
                checkpoint_cursor=checkpoint_cursor,
            )
        continued_from_checkpoint = True
    elif canonical is None or _typed_lifecycle_prefix_matches(reference_events, canonical):
        continued = dict(reference_events)
        continued_from_checkpoint = False
    elif _typed_lifecycle_prefix_matches(canonical, reference_events):
        continued = canonical
        continued_from_checkpoint = False
    else:
        raise RuntimeError(
            "reference lifecycle candidate contradicts the canonical tape prefix"
        )
    return continued, continued_from_checkpoint


def emit_native_typed_lifecycle_public_outputs(
    runner: Any,
    engine: NativeEielEngine,
) -> dict[str, Any]:
    projection = engine.export_checkpoint_projection()
    events = typed_lifecycle_events_from_projection(projection)
    return _emit_typed_lifecycle_public_outputs(
        runner,
        events,
        jsonl_name="native_typed_lifecycle_events.jsonl",
        csv_name="native_typed_lifecycle_events.csv",
        cursor_attr="_native_runtime_typed_lifecycle_event_public_cursor",
        summary_attr="_native_runtime_typed_lifecycle_public_output",
        schema_version="eiel_native_typed_lifecycle_public_output_v1",
        source="native_checkpoint_projection_typed_lifecycle_event_tensors",
    )


def emit_route_neutral_typed_lifecycle_public_outputs(runner: Any) -> dict[str, Any]:
    engine = getattr(runner, "_native_eiel_engine", None)
    export_projection = getattr(engine, "export_checkpoint_projection", None)
    if callable(export_projection):
        projection = export_projection()
        events = typed_lifecycle_events_from_projection(projection)
        event_cursor = validated_typed_lifecycle_event_cursor(projection, events)
        source = "native_checkpoint_projection_typed_lifecycle_event_tensors"
        continued_from_checkpoint = False
    else:
        from perf.cxxbackbone10_resident_state_owner import (  # noqa: WPS433
            _typed_lifecycle_event_tensors_for_runner,
        )

        prefixed = _typed_lifecycle_event_tensors_for_runner(runner)
        if prefixed:
            reference_events = {
                name: prefixed[f"typed_lifecycle_event_{name}"]
                for name in FIELD_DTYPES
            }
        else:
            reference_events = empty_typed_lifecycle_event_tensors()
        require_complete = _reference_typed_lifecycle_completion_required(runner)
        if require_complete:
            reference_events = complete_reference_typed_lifecycle_events(
                reference_events,
                _reference_agent_lifecycle_identities(runner),
                require_complete=True,
            )
        events, continued_from_checkpoint = _continue_reference_typed_lifecycle_events(
            runner,
            reference_events,
        )
        source = (
            "native_checkpoint_boundary_plus_reference_cumulative_lifecycle_event_tensors"
            if continued_from_checkpoint
            else "reference_cumulative_lifecycle_event_tensors"
        )

    summary = _emit_typed_lifecycle_public_outputs(
        runner,
        events,
        jsonl_name="typed_lifecycle_events.jsonl",
        csv_name="typed_lifecycle_events.csv",
        cursor_attr="_typed_lifecycle_event_public_cursor",
        summary_attr="_typed_lifecycle_public_output",
        schema_version="eiel_native_typed_lifecycle_public_output_v1",
        source=source,
    )
    _install_canonical_reference_lifecycle_events(runner, events)
    if continued_from_checkpoint:
        _consume_reference_checkpoint_lifecycle_boundary(runner)
    if callable(export_projection):
        setattr(runner, _ROUTE_NEUTRAL_OUTPUT_ACCUMULATOR_CURSOR_ATTR, event_cursor)
    native_summary = getattr(runner, "_native_runtime_summary", None)
    if isinstance(native_summary, dict):
        native_summary["typed_lifecycle_public_output"] = dict(summary)
        setattr(runner, "_native_runtime_summary", native_summary)
    return summary


def write_native_behavior_public_outputs(
    runner: Any,
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    if not bool(getattr(getattr(runner, "sim_cfg", None), "behavior_log_enable", True)):
        summary = {
            "schema_version": "eiel_native_behavior_public_output_v1",
            "source": "native_owner_behavior_rows",
            "csv_path": str(Path(str(getattr(runner, "run_dir", "") or ".")) / "behaviors.csv"),
            "rows_written": 0,
            "csv_existed": False,
            "disabled": True,
            "flush_complete": False,
        }
        setattr(runner, "_native_runtime_behavior_public_output", dict(summary))
        return summary
    rows = list(raw_result.get("behavior_rows") or [])
    path = Path(str(getattr(runner, "run_dir", "") or ".")) / "behaviors.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    original_size = path.stat().st_size if existed else 0
    try:
        with path.open("a", encoding="utf-8", newline="") as handle:
            # Match BehaviorLogger's csv.writer byte contract so native-owned
            # rows remain interchangeable with reference output.
            writer = csv.writer(handle, lineterminator="\r\n")
            if not existed:
                writer.writerow(_BEHAVIOR_CSV_FIELDS)
            for row in rows:
                writer.writerow(list(row))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        quarantine_paths: tuple[str, ...] = ()
        try:
            if existed and path.exists():
                with path.open("r+b") as handle:
                    handle.truncate(original_size)
                    handle.flush()
                    os.fsync(handle.fileno())
            elif path.exists():
                quarantine = path.with_name(path.name + ".incomplete")
                os.replace(path, quarantine)
                quarantine_paths = (str(quarantine),)
        except Exception as quarantine_exc:
            quarantine_paths = (f"quarantine_failed:{quarantine_exc!r}",)
        record_native_runtime_failure(
            runner,
            exc,
            stage="behavior_public_output",
            quarantine_paths=quarantine_paths,
        )
        raise
    behavior_logger = getattr(runner, "behavior_logger", None)
    if behavior_logger is not None:
        try:
            behavior_logger._gen = None
            behavior_logger._data.clear()
            behavior_logger._ids = []
            behavior_logger._start_ids = set()
            behavior_logger._injury_start = {}
        except Exception:
            pass
    summary = {
        "schema_version": "eiel_native_behavior_public_output_v1",
        "source": "native_owner_behavior_rows",
        "csv_path": str(path),
        "rows_written": int(len(rows)),
        "csv_existed": bool(existed),
        "flush_complete": True,
    }
    validation = raw_result.get("native_result_validation")
    if isinstance(validation, dict):
        validation["behavior_output_flush_complete"] = True
    setattr(runner, "_native_runtime_behavior_public_output", dict(summary))
    return summary


def execute_native_generation_from_runner(
    runner: Any,
    *,
    generation_index: int,
    steps_per_generation: int,
) -> NativeGenerationResult:
    decision = validate_native_runtime_supported(runner)
    engine = get_or_create_native_engine(
        runner,
        generation_index=int(generation_index),
        safe_default_decision=decision,
    )
    max_births = int(getattr(getattr(runner, "sim_cfg", None), "native_runtime_max_births_per_generation", 1) or 1)
    birth_commit = bool(getattr(getattr(runner, "sim_cfg", None), "native_runtime_birth_commit_enable", True))
    result = engine.execute_generation(
        generation_index=int(generation_index),
        steps_per_generation=int(steps_per_generation),
        max_births=max_births,
        birth_commit=birth_commit,
    )
    try:
        validate_native_generation_result(
            runner,
            result,
            engine,
            generation_index=int(generation_index),
            steps_per_generation=int(steps_per_generation),
        )
    except Exception as exc:
        record_native_runtime_failure(runner, exc, stage="generation_result_validation")
        raise
    from .public_metrics_overlay import public_metrics_overlay

    checkpoint_projection = engine.export_checkpoint_projection()
    metrics_overlay_values = public_metrics_overlay(
        result.raw_result,
        result.compact_result,
        checkpoint_projection,
        sim_cfg=getattr(runner, "sim_cfg", None),
        env_cfg=getattr(runner, "env_cfg", None),
        metrics_generation=int(generation_index),
    )
    metrics_overlay = {
        "schema_version": "eiel_native_public_metrics_overlay_v1",
        "source": "native_owner_checkpoint_projection",
        "values": dict(metrics_overlay_values),
    }
    setattr(runner, "_native_runtime_public_metrics_overlay", metrics_overlay)
    result.compact_result["python_agent_sync"] = False
    summary = result.to_diagnostic_dict()
    summary["python_agent_sync"] = False
    summary["safe_default_decision"] = dict(getattr(runner, "_native_safe_default_decision", {}) or {})
    summary["native_result_validation"] = dict(result.raw_result["native_result_validation"])
    summary["public_metrics_overlay"] = {
        "schema_version": metrics_overlay["schema_version"],
        "source": metrics_overlay["source"],
        "field_count": int(len(metrics_overlay_values)),
    }
    result.raw_result["native_result_validation"]["metrics_output_present"] = True
    result.compact_result["native_result_validation"]["metrics_output_present"] = True
    summary["native_result_validation"]["metrics_output_present"] = True
    setattr(runner, "_native_runtime_summary", summary)
    return result


def execute_native_generation_boundary_turnover_from_runner(
    runner: Any,
    *,
    generation_index: int,
) -> dict[str, Any]:
    decision = validate_native_runtime_supported(runner)
    engine = get_or_create_native_engine(
        runner,
        generation_index=int(generation_index),
        safe_default_decision=decision,
    )
    result = dict(engine.execute_generation_boundary_turnover())
    if not bool(result.get("executed", False)) or bool(result.get("hidden_fallback", False)):
        raise RuntimeError(f"EIEL-NATIVE-GENERATION-TURNOVER-FAILED: {result}")
    summary = {
        "schema_version": "eiel_native_generation_boundary_turnover_v1",
        "generation_index": int(generation_index),
        "result": result,
        "python_agent_sync": False,
    }
    setattr(runner, "_native_runtime_generation_boundary_turnover", summary)
    return summary


install_reference_typed_lifecycle_projection_hook()
