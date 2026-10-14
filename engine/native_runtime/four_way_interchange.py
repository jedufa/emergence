from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from perf.cxxbackbone10_resident_state_owner import (
    ResidentRuntimeState,
    compare_projection_digest,
)

from .checkpoint_digests import (
    DigestLayer,
    checkpoint_digest_manifest,
    compare_checkpoint_layer_first_difference,
)
from .engine import NativeEielEngine
from .event_matrix import (
    checkpoint_sidecar_semantics_from_projection,
    final_lifecycle_summary_from_artifact,
    final_lineage_summary_semantics_from_projection,
    final_summary_json_semantics_from_projection,
    lineage_funnel_semantic_rows_from_artifact,
    lifecycle_counts_from_artifact,
    typed_lifecycle_artifact_from_projection,
)
from .interchange_projection import (
    apply_interchange_to_reference_runner,
    interchange_projection_from_native_checkpoint,
    native_checkpoint_projection_from_interchange,
    native_engine_from_interchange,
)


SCHEMA_VERSION = "eiel_four_way_interchange_harness_v1"
NON_AUTHORITATIVE_PROJECTION_METADATA_FIELDS = frozenset(
    {
        "import_digest",
        "native_policy_cache_served",
        "native_runtime_rng_manager_stream_indices",
        "typed_lifecycle_event_source",
    }
)

ReferenceSegmentRunner = Callable[[Any, int, int], Any]
RunnerFingerprint = Callable[[Any], Mapping[str, Any]]


def comparison_artifact_from_projection(
    projection: Mapping[str, Any],
    *,
    source: str,
    final_alive_count: int | None = None,
) -> dict[str, Any]:
    artifact = typed_lifecycle_artifact_from_projection(projection, source=source)
    lifecycle_counts = lifecycle_counts_from_artifact(artifact)
    final_lineage_summary = final_lineage_summary_semantics_from_projection(
        projection,
        lifecycle_counts=lifecycle_counts,
    )
    checkpoint_sidecar_semantics = checkpoint_sidecar_semantics_from_projection(
        projection,
        artifact=artifact,
    )
    trajectory_manifest = checkpoint_digest_manifest(projection, layer=DigestLayer.TRAJECTORY_STATE)
    canonical_manifest = checkpoint_digest_manifest(projection, layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE)
    diagnostic_manifest = checkpoint_digest_manifest(projection, layer=DigestLayer.DIAGNOSTIC_PROJECTION)
    return {
        "schema_version": "eiel_four_way_comparison_artifact_v1",
        "source": str(source),
        "trajectory_digest": trajectory_manifest["digest"],
        "trajectory_field_digests": {
            str(row["field"]): str(row["sha256"]) for row in trajectory_manifest["fields"]
        },
        "canonical_storage_digest": canonical_manifest["digest"],
        "canonical_storage_field_digests": {
            str(row["field"]): str(row["sha256"]) for row in canonical_manifest["fields"]
        },
        "diagnostic_digest": diagnostic_manifest["digest"],
        "diagnostic_field_digests": {
            str(row["field"]): str(row["sha256"]) for row in diagnostic_manifest["fields"]
        },
        "rng_state_digest": {
            key: str(dict(projection.get("metadata") or {}).get(key, ""))
            for key in (
                "native_runtime_numpy_env_state",
                "native_runtime_numpy_runner_state",
                "native_runtime_numpy_population_state",
                "native_runtime_python_random_state",
            )
        },
        "agent_grid_phase_contract": {
            key: dict(projection.get("metadata") or {}).get(key)
            for key in (
                "agent_grid_phase_contract_schema_version",
                "agent_grid_runner_phase",
                "agent_grid_authority_epoch",
            )
        },
        "typed_lifecycle_artifact": artifact.to_summary(),
        "lineage_funnel_semantic_rows": lineage_funnel_semantic_rows_from_artifact(artifact),
        "lifecycle_counts": lifecycle_counts,
        "final_lifecycle_summary": final_lifecycle_summary_from_artifact(
            artifact,
            final_alive_count=final_alive_count,
            lifecycle_counts=lifecycle_counts,
        ),
        "final_lineage_summary": final_lineage_summary,
        "checkpoint_sidecar_semantics": checkpoint_sidecar_semantics,
        "final_summary_json_semantics": final_summary_json_semantics_from_projection(
            projection,
            artifact=artifact,
            lifecycle_counts=lifecycle_counts,
            final_lineage_summary=final_lineage_summary,
            checkpoint_sidecar_semantics=checkpoint_sidecar_semantics,
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def first_mismatch(left: Any, right: Any, *, path: str = "$") -> dict[str, Any] | None:
    left = _jsonable(left)
    right = _jsonable(right)
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right, "reason": "type_mismatch"}
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left:
                return {"path": f"{path}.{key}", "left": "<missing>", "right": right[key]}
            if key not in right:
                return {"path": f"{path}.{key}", "left": left[key], "right": "<missing>"}
            mismatch = first_mismatch(left[key], right[key], path=f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": f"{path}.length", "left": len(left), "right": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatch = first_mismatch(left_item, right_item, path=f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if left != right:
        return {"path": path, "left": left, "right": right}
    return None


def _artifact_without_source(artifact: Mapping[str, Any]) -> dict[str, Any]:
    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): strip(v) for k, v in value.items() if str(k) != "source"}
        if isinstance(value, tuple):
            return tuple(strip(v) for v in value)
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    out = dict(strip(artifact))
    out.pop("source", None)
    out.pop("trajectory_digest", None)
    out.pop("canonical_storage_digest", None)
    out.pop("diagnostic_digest", None)
    summary = out.get("final_summary_json_semantics")
    if isinstance(summary, dict):
        summary = dict(summary)
        summary.pop("deterministic_storage_fields", None)
        summary.pop("diagnostic_only_fields", None)
        summary.pop("implementation_specific_fields", None)
        out["final_summary_json_semantics"] = summary
    return out


def _field_name_from_mismatch(mismatch: Mapping[str, Any]) -> str:
    field = mismatch.get("field")
    if field is not None:
        return str(field)
    path = str(mismatch.get("path", "$"))
    for prefix in ("$.metadata.", "$.tensors."):
        if path.startswith(prefix):
            return path[len(prefix) :].split(".", maxsplit=1)[0].split("[", maxsplit=1)[0]
    return path.removeprefix("$.").split(".", maxsplit=1)[0].split("[", maxsplit=1)[0]


def _projection_digest_first_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any] | None:
    left_metadata = dict(left.get("metadata") or {})
    right_metadata = dict(right.get("metadata") or {})
    for field in NON_AUTHORITATIVE_PROJECTION_METADATA_FIELDS:
        left_metadata.pop(field, None)
        right_metadata.pop(field, None)
    metadata_mismatch = first_mismatch(
        json.loads(json.dumps(left_metadata, sort_keys=True, default=str)),
        json.loads(json.dumps(right_metadata, sort_keys=True, default=str)),
        path="$.metadata",
    )
    if metadata_mismatch is not None:
        return {
            "field": _field_name_from_mismatch(metadata_mismatch),
            **metadata_mismatch,
        }
    mismatch = compare_checkpoint_layer_first_difference(
        left,
        right,
        layer=DigestLayer.TRAJECTORY_STATE,
    )
    if mismatch is not None:
        return mismatch
    left_manifest = checkpoint_digest_manifest(left, layer=DigestLayer.TRAJECTORY_STATE)
    right_manifest = checkpoint_digest_manifest(right, layer=DigestLayer.TRAJECTORY_STATE)
    mismatch = first_mismatch(
        left_manifest["fields"],
        right_manifest["fields"],
        path="$.trajectory_fields",
    )
    if mismatch is not None and "field" not in mismatch:
        return {"field": _field_name_from_mismatch(mismatch), **mismatch}
    return mismatch


def _authoritative_projection_digest_input(projection: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(projection.get("metadata") or {})
    for field in NON_AUTHORITATIVE_PROJECTION_METADATA_FIELDS:
        metadata.pop(field, None)
    manifest = checkpoint_digest_manifest(projection, layer=DigestLayer.TRAJECTORY_STATE)
    return {
        "metadata": {
            "authoritative_metadata": metadata,
            "trajectory_fields": manifest["fields"],
        },
        "tensors": {},
    }


def _canonical_checkpoint_storage_verdict(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_manifest = checkpoint_digest_manifest(
        left,
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    )
    right_manifest = checkpoint_digest_manifest(
        right,
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    )
    ok = str(left_manifest["digest"]) == str(right_manifest["digest"])
    mismatch = None
    if not ok:
        mismatch = compare_checkpoint_layer_first_difference(
            left,
            right,
            layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
        )
        if mismatch is None:
            mismatch = first_mismatch(
                left_manifest["metadata"],
                right_manifest["metadata"],
                path="$.metadata",
            )
        if mismatch is None:
            mismatch = first_mismatch(
                left_manifest["fields"],
                right_manifest["fields"],
                path="$.fields",
            )
        if mismatch is not None and "field" not in mismatch:
            mismatch = {"field": _field_name_from_mismatch(mismatch), **mismatch}
    return {
        "ok": ok,
        "left_digest": str(left_manifest["digest"]),
        "right_digest": str(right_manifest["digest"]),
        "first_mismatch": mismatch,
    }


def _projection_parity_verdict(
    left_projection: Mapping[str, Any],
    right_projection: Mapping[str, Any],
    left_artifact: Mapping[str, Any],
    right_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    projection_compare = compare_projection_digest(
        _authoritative_projection_digest_input(left_projection),
        _authoritative_projection_digest_input(right_projection),
    )
    projection_ok = bool(projection_compare.get("ok", False))
    projection_mismatch = (
        None
        if projection_ok
        else _projection_digest_first_difference(left_projection, right_projection)
    )
    if not projection_ok and projection_mismatch is None:
        projection_mismatch = {
            "field": "projection_digest",
            "path": "$.projection_digest",
            "left": projection_compare.get("left_digest"),
            "right": projection_compare.get("right_digest"),
            "reason": "digest_mismatch_without_field_difference",
        }
    authoritative_projection_digest = {
        "ok": projection_ok,
        "left_digest": projection_compare.get("left_digest"),
        "right_digest": projection_compare.get("right_digest"),
        "first_mismatch": projection_mismatch,
    }
    canonical_checkpoint_storage = _canonical_checkpoint_storage_verdict(
        left_projection,
        right_projection,
    )
    artifact_mismatch = first_mismatch(
        _artifact_without_source(left_artifact),
        _artifact_without_source(right_artifact),
    )
    comparison_artifact = {
        "ok": artifact_mismatch is None,
        "first_mismatch": artifact_mismatch,
    }
    surfaces = {
        "authoritative_projection_digest": authoritative_projection_digest,
        "canonical_checkpoint_storage": canonical_checkpoint_storage,
        "comparison_artifact": comparison_artifact,
    }
    first_failed = next(
        (
            {
                "surface": name,
                "details": surface.get("first_mismatch") or surface,
            }
            for name, surface in surfaces.items()
            if not bool(surface.get("ok", False))
        ),
        None,
    )
    ok = all(bool(surface.get("ok", False)) for surface in surfaces.values())
    return {
        **projection_compare,
        "projection_digest_ok": projection_ok,
        "projection_digest_first_mismatch": projection_mismatch,
        "canonical_checkpoint_storage_ok": bool(canonical_checkpoint_storage["ok"]),
        "canonical_checkpoint_first_mismatch": canonical_checkpoint_storage["first_mismatch"],
        "artifact_ok": bool(comparison_artifact["ok"]),
        "mismatch": artifact_mismatch,
        "surfaces": surfaces,
        "status": "passed" if ok else "failed_closed",
        "ok": ok,
        "first_mismatch": None if first_failed is None else first_failed["details"],
        "first_failed": first_failed,
    }


def _overall_comparison_verdict(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    first_failed = next(
        (
            {
                "path": name,
                "details": comparison.get("first_mismatch")
                or comparison.get("mismatch")
                or comparison,
            }
            for name, comparison in comparisons.items()
            if not bool(comparison.get("ok", False))
        ),
        None,
    )
    return {
        "ok": first_failed is None,
        "first_failed": first_failed,
    }


def _projection_from_reference_runner(
    runner: Any,
    *,
    capability_mask: Mapping[str, bool],
    generation: int,
    tick: int,
    build_directory: str | Path | None,
) -> dict[str, Any]:
    owner = ResidentRuntimeState.from_runner(
        runner,
        capability_mask,
        generation=int(generation),
        tick=int(tick),
        build_directory=build_directory,
    )
    return owner.export_checkpoint_projection()


def _run_reference_segment_with_start_tick(
    runner: Any,
    run_reference_segment: ReferenceSegmentRunner,
    generation_index: int,
    steps: int,
    *,
    start_tick: int,
) -> None:
    previous = getattr(runner, "_eiel_resume_start_tick", None)
    marker_was_present = hasattr(runner, "_eiel_resume_start_tick")
    sim_cfg = getattr(runner, "sim_cfg", None)
    native_route_attrs = (
        "native_safe_default_enable",
        "native_runtime_enable",
        "native_runtime_diagnostic_enable",
    )
    native_route_previous = {
        name: getattr(sim_cfg, name)
        for name in native_route_attrs
        if sim_cfg is not None and hasattr(sim_cfg, name)
    }
    try:
        setattr(runner, "_eiel_resume_start_tick", int(start_tick))
        if sim_cfg is not None:
            for name in native_route_previous:
                setattr(sim_cfg, name, False)
        run_reference_segment(runner, int(generation_index), int(steps))
    finally:
        if sim_cfg is not None:
            for name, value in native_route_previous.items():
                setattr(sim_cfg, name, value)
        if marker_was_present:
            setattr(runner, "_eiel_resume_start_tick", previous)
        else:
            try:
                delattr(runner, "_eiel_resume_start_tick")
            except AttributeError:
                pass


def _generation_tick_after_steps(runner: Any, generation_index: int, steps: int) -> tuple[int, int]:
    configured_steps = int(getattr(getattr(runner, "sim_cfg", None), "steps_per_generation", 0) or 0)
    if configured_steps > 0 and int(steps) >= configured_steps:
        return int(generation_index) + 1, 0
    return int(generation_index), int(steps)


def _clear_reference_checkpoint_tensor_overrides(
    runner: Any,
    *,
    keep_fields: tuple[str, ...] = (),
    keep_prefixes: tuple[str, ...] = (),
) -> None:
    try:
        delattr(runner, "_cxxbackbone10_checkpoint_metadata_overrides")
    except AttributeError:
        pass
    overrides = getattr(runner, "_cxxbackbone10_checkpoint_tensor_overrides", None)
    if (keep_fields or keep_prefixes) and isinstance(overrides, Mapping):
        kept = {
            name: value
            for name, value in overrides.items()
            if name in keep_fields or any(str(name).startswith(prefix) for prefix in keep_prefixes)
        }
        if kept:
            setattr(runner, "_cxxbackbone10_checkpoint_tensor_overrides", kept)
            return
    try:
        delattr(runner, "_cxxbackbone10_checkpoint_tensor_overrides")
    except AttributeError:
        pass


def run_generation_boundary_four_way_interchange(
    *,
    runner_factory: Callable[[str], Any],
    run_reference_segment: ReferenceSegmentRunner,
    runner_fingerprint: RunnerFingerprint,
    capability_mask: Mapping[str, bool],
    generation_index: int,
    total_steps: int,
    boundary_tick: int,
    max_births: int,
    build_directory: str | Path | None = None,
) -> dict[str, Any]:
    remaining_steps = int(total_steps) - int(boundary_tick)
    if remaining_steps < 0:
        raise ValueError("boundary_tick must be <= total_steps")

    reference_uninterrupted_runner = runner_factory("reference_uninterrupted")
    _run_reference_segment_with_start_tick(
        reference_uninterrupted_runner,
        run_reference_segment,
        int(generation_index),
        int(total_steps),
        start_tick=0,
    )
    reference_uninterrupted = runner_fingerprint(reference_uninterrupted_runner)

    native_uninterrupted_runner = runner_factory("native_uninterrupted")
    native_uninterrupted = NativeEielEngine.from_runner(
        native_uninterrupted_runner,
        generation=int(generation_index),
        tick=0,
        build_directory=build_directory,
    )
    native_uninterrupted.execute_generation(
        generation_index=int(generation_index),
        steps_per_generation=int(total_steps),
        max_births=int(max_births),
    )
    native_uninterrupted_projection = native_uninterrupted.export_checkpoint_projection()

    native_resume_runner = runner_factory("native_resume")
    native_first = NativeEielEngine.from_runner(
        native_resume_runner,
        generation=int(generation_index),
        tick=0,
        build_directory=build_directory,
    )
    native_first.execute_generation(
        generation_index=int(generation_index),
        steps_per_generation=int(boundary_tick),
        max_births=int(max_births),
    )
    boundary_generation, boundary_tick_coordinate = _generation_tick_after_steps(
        native_resume_runner,
        int(generation_index),
        int(boundary_tick),
    )
    boundary_checkpoint = f"generation:{boundary_generation}:tick:{boundary_tick_coordinate}"
    native_interchange = interchange_projection_from_native_checkpoint(
        native_first.export_checkpoint_projection(),
        source_runtime="native",
        checkpoint_boundary=boundary_checkpoint,
    )
    native_restored = native_engine_from_interchange(
        native_interchange,
        build_directory=build_directory,
        allow_partial_generation_for_diagnostics=True,
    )
    if remaining_steps > 0:
        native_restored.execute_generation(
            generation_index=int(generation_index),
            steps_per_generation=int(remaining_steps),
            max_births=int(max_births),
        )
    native_resume_projection = native_restored.export_checkpoint_projection()

    reference_to_native_runner = runner_factory("reference_to_native")
    _run_reference_segment_with_start_tick(
        reference_to_native_runner,
        run_reference_segment,
        int(generation_index),
        int(boundary_tick),
        start_tick=0,
    )
    reference_interchange = interchange_projection_from_native_checkpoint(
        _projection_from_reference_runner(
            reference_to_native_runner,
            capability_mask=capability_mask,
            generation=int(boundary_generation),
            tick=int(boundary_tick_coordinate),
            build_directory=build_directory,
        ),
        source_runtime="reference",
        checkpoint_boundary=boundary_checkpoint,
    )
    reference_resumed_native = native_engine_from_interchange(
        reference_interchange,
        build_directory=build_directory,
        allow_partial_generation_for_diagnostics=True,
    )
    if remaining_steps > 0:
        reference_resumed_native.execute_generation(
            generation_index=int(generation_index),
            steps_per_generation=int(remaining_steps),
            max_births=int(max_births),
        )
    reference_resumed_native_projection = reference_resumed_native.export_checkpoint_projection()
    apply_interchange_to_reference_runner(
        reference_to_native_runner,
        interchange_projection_from_native_checkpoint(
            reference_resumed_native_projection,
            source_runtime="reference_resumed_native",
            checkpoint_boundary=f"generation:{int(generation_index)}:tick:{int(total_steps)}",
        ),
    )
    reference_to_native = runner_fingerprint(reference_to_native_runner)

    native_to_reference_runner = runner_factory("native_to_reference")
    apply_result = apply_interchange_to_reference_runner(native_to_reference_runner, native_interchange)
    native_route_capability_mask = dict(
        dict(native_checkpoint_projection_from_interchange(native_interchange).get("metadata") or {}).get(
            "capability_mask"
        )
        or capability_mask
    )
    reimported_projection = _projection_from_reference_runner(
        native_to_reference_runner,
        capability_mask=native_route_capability_mask,
        generation=int(boundary_generation),
        tick=int(boundary_tick_coordinate),
        build_directory=build_directory,
    )
    native_to_reference_reimport_artifact = comparison_artifact_from_projection(
        reimported_projection,
        source="native_checkpoint_consumed_by_reference",
    )
    _clear_reference_checkpoint_tensor_overrides(
        native_to_reference_runner,
        keep_fields=(
            "event_tape_type",
            "event_tape_generation",
            "event_tape_tick",
            "event_tape_actor_id",
            "event_tape_target_id",
            "event_tape_value",
        ),
        keep_prefixes=("typed_lifecycle_event_",),
    )
    if remaining_steps > 0:
        _run_reference_segment_with_start_tick(
            native_to_reference_runner,
            run_reference_segment,
            int(generation_index),
            int(remaining_steps),
            start_tick=int(boundary_tick),
        )
    _clear_reference_checkpoint_tensor_overrides(
        native_to_reference_runner,
        keep_fields=(
            "drive_inertia",
            "drive_noise",
            "policy_attack_bias",
            "policy_action_temperature",
            "policy_drive_logit_gain",
            "policy_use_gru",
            "policy_mem_dim",
            "event_tape_type",
            "event_tape_generation",
            "event_tape_tick",
            "event_tape_actor_id",
            "event_tape_target_id",
            "event_tape_value",
            "genome_hidden_size",
            "genome_memory_dim",
            "genome_sensor_channel_mask",
            "genome_sensor_radius",
            "genome_introspection_channel_mask",
            "genome_self_regulation_control_mask",
            "genome_self_modification_target_mask",
            "genome_directed_inheritance_channel_mask",
            "genome_exploration_bias",
            "genome_aggression_bias",
            "genome_mutation_rate",
            "genome_reproduction_threshold",
            "genome_action_temperature",
            "genome_drive_inertia",
            "genome_drive_noise",
            "genome_drive_logit_gain",
            "genome_pred_lr",
            "genome_drive_sp_hunger",
            "genome_drive_sp_pain",
            "genome_drive_sp_fatigue",
            "genome_drive_sp_fear",
            "genome_exp_prior_strength",
            "genome_exp_prior_noise",
            "genome_drive_birth_noise",
        ),
        keep_prefixes=("typed_lifecycle_event_",),
    )
    native_to_reference = runner_fingerprint(native_to_reference_runner)
    final_generation, final_tick = _generation_tick_after_steps(
        native_to_reference_runner,
        int(generation_index),
        int(total_steps),
    )
    native_to_reference_projection = _projection_from_reference_runner(
        native_to_reference_runner,
        capability_mask=native_route_capability_mask,
        generation=final_generation,
        tick=int(final_tick),
        build_directory=build_directory,
    )
    native_to_reference_final_interchange = interchange_projection_from_native_checkpoint(
        native_to_reference_projection,
        source_runtime="native_checkpoint_resumed_reference",
        checkpoint_boundary=f"generation:{final_generation}:tick:{int(final_tick)}",
    )
    native_to_reference_reimported_native = native_engine_from_interchange(
        native_to_reference_final_interchange,
        build_directory=build_directory,
        allow_partial_generation_for_diagnostics=True,
    )
    native_to_reference_reimported_native_projection = (
        native_to_reference_reimported_native.export_checkpoint_projection()
    )

    native_uninterrupted_artifact = comparison_artifact_from_projection(
        native_uninterrupted_projection,
        source="native_uninterrupted",
    )
    native_resume_artifact = comparison_artifact_from_projection(
        native_resume_projection,
        source="native_save_resume",
    )
    reference_resumed_native_artifact = comparison_artifact_from_projection(
        reference_resumed_native_projection,
        source="reference_checkpoint_resumed_native",
    )
    native_to_reference_resumed_artifact = comparison_artifact_from_projection(
        native_to_reference_projection,
        source="native_checkpoint_resumed_reference",
    )
    native_to_reference_reimported_native_artifact = comparison_artifact_from_projection(
        native_to_reference_reimported_native_projection,
        source="native_checkpoint_resumed_reference_reimported_native",
    )

    native_resume_compare = _projection_parity_verdict(
        native_uninterrupted_projection,
        native_resume_projection,
        native_uninterrupted_artifact,
        native_resume_artifact,
    )

    native_boundary_projection = native_checkpoint_projection_from_interchange(native_interchange)
    native_boundary_artifact = comparison_artifact_from_projection(
        native_boundary_projection,
        source="native_boundary",
    )
    reimport_compare = _projection_parity_verdict(
        native_boundary_projection,
        reimported_projection,
        native_boundary_artifact,
        native_to_reference_reimport_artifact,
    )

    native_to_reference_reimported_native_compare = _projection_parity_verdict(
        native_uninterrupted_projection,
        native_to_reference_reimported_native_projection,
        native_uninterrupted_artifact,
        native_to_reference_reimported_native_artifact,
    )

    comparisons = {
        "native_resume_matches_native_uninterrupted": native_resume_compare,
        "reference_to_native_matches_reference_uninterrupted": {
            "ok": reference_to_native == reference_uninterrupted,
            "mismatch": first_mismatch(reference_uninterrupted, reference_to_native),
        },
        "native_to_reference_reimport_matches_boundary": reimport_compare,
        "native_to_reference_matches_reference_uninterrupted": {
            "ok": native_to_reference == reference_uninterrupted,
            "mismatch": first_mismatch(reference_uninterrupted, native_to_reference),
        },
        "native_to_reference_reimported_native_matches_native_uninterrupted": (
            native_to_reference_reimported_native_compare
        ),
    }
    overall_comparison = _overall_comparison_verdict(comparisons)

    return {
        "schema_version": SCHEMA_VERSION,
        "generation_index": int(generation_index),
        "total_steps": int(total_steps),
        "boundary_tick": int(boundary_tick),
        "directions": {
            "reference_uninterrupted": reference_uninterrupted,
            "native_uninterrupted": native_uninterrupted_artifact,
            "native_save_resume": native_resume_artifact,
            "reference_checkpoint_resumed_native": reference_resumed_native_artifact,
            "native_checkpoint_consumed_by_reference": {
                "apply_result": apply_result,
                "reimport_artifact": native_to_reference_reimport_artifact,
            },
            "native_checkpoint_resumed_reference": {
                "fingerprint": native_to_reference,
                "artifact": native_to_reference_resumed_artifact,
            },
            "native_checkpoint_resumed_reference_reimported_native": (
                native_to_reference_reimported_native_artifact
            ),
        },
        "comparisons": comparisons,
        "ok": bool(overall_comparison["ok"]),
        "first_failed": overall_comparison["first_failed"],
    }
