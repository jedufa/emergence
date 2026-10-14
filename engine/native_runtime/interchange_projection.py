from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from perf.cuda_architecture_b_modular_state_abi import (
    ModularStorageError,
    modular_storage_from_projection,
)

from .checkpoint_digests import DigestLayer, checkpoint_digest_manifest
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


SCHEMA_VERSION = "eiel_reference_native_interchange_projection_v1"
ADAPTER_VERSION = "eiel_reference_native_checkpoint_adapter_v1"

REQUIRED_METADATA_FIELDS = (
    "generation",
    "tick",
    "world",
    "agent_grid_phase_contract_schema_version",
    "agent_grid_runner_phase",
    "agent_grid_authority_epoch",
    "agent_grid_preserve_exact_stored",
    "native_runtime_numpy_env_state",
    "native_runtime_numpy_runner_state",
    "native_runtime_numpy_population_state",
    "native_runtime_python_random_state",
)

REQUIRED_TENSOR_FIELDS = (
    "agent_grid",
    "alive_mask",
    "slot_to_agent_id",
    "agent_id_to_slot_agent_ids",
    "agent_id_to_slot_slots",
    "slot_tokens",
    "slot_free_queue",
    "pos_y",
    "pos_x",
    "energy",
    "health",
    "drives",
    "hmem",
    "mlp_pack",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "action_cache_actions",
    "event_tape_type",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
)


class InterchangeProjectionError(RuntimeError):
    """Raised when a reference/native interchange projection is not authoritative."""


def _copy_projection(projection: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        str(name): value.detach().clone() if hasattr(value, "detach") else deepcopy(value)
        for name, value in dict(projection).items()
        if name not in {"metadata", "tensors"}
    }
    out["schema_version"] = str(projection.get("schema_version", "native_checkpoint_projection") or "")
    out["metadata"] = deepcopy(dict(projection.get("metadata") or {}))
    out["tensors"] = {
        str(name): value.detach().clone() if hasattr(value, "detach") else deepcopy(value)
        for name, value in dict(projection.get("tensors") or {}).items()
    }
    return out


def _tensor_manifest(projection: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = []
    for name, value in sorted(dict(projection.get("tensors") or {}).items()):
        if not hasattr(value, "detach"):
            raise InterchangeProjectionError(f"interchange tensor field is not tensor-like: {name}")
        tensor = value.detach().cpu().contiguous()
        rows.append(
            {
                "field": str(name),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "shape": tuple(int(x) for x in tensor.shape),
            }
        )
    return tuple(rows)


def validate_native_checkpoint_projection(projection: Mapping[str, Any]) -> None:
    metadata = dict(projection.get("metadata") or {})
    tensors = dict(projection.get("tensors") or {})
    missing_metadata = [name for name in REQUIRED_METADATA_FIELDS if name not in metadata]
    missing_tensors = [name for name in REQUIRED_TENSOR_FIELDS if name not in tensors]
    if missing_metadata:
        raise InterchangeProjectionError(f"missing authoritative checkpoint metadata fields: {missing_metadata}")
    if missing_tensors:
        raise InterchangeProjectionError(f"missing authoritative checkpoint tensor fields: {missing_tensors}")
    if str(metadata.get("agent_grid_phase_contract_schema_version", "") or "") != "eiel_agent_grid_phase_contract_v1":
        raise InterchangeProjectionError("unsupported agent-grid phase contract schema")
    if not bool(metadata.get("agent_grid_preserve_exact_stored", False)):
        raise InterchangeProjectionError("agent-grid projection is not exact-stored authoritative")
    runtime_event_cursor = projection.get("runtime_event_cursor")
    if type(runtime_event_cursor) is not int or runtime_event_cursor < 0:
        raise InterchangeProjectionError(
            "runtime_event_cursor must be a non-negative int"
        )
    projection_event_cursor = projection.get("event_cursor")
    if type(projection_event_cursor) is not int or projection_event_cursor < 0:
        raise InterchangeProjectionError("event_cursor must be a non-negative int")
    if runtime_event_cursor < projection_event_cursor:
        raise InterchangeProjectionError(
            "runtime_event_cursor cannot precede event_cursor"
        )
    event_count = int(metadata.get("typed_lifecycle_event_active_count", metadata.get("event_tape_active_count", 0)) or 0)
    event_cursor = int(metadata.get("event_tape_cursor", event_count) or 0)
    if event_cursor != event_count:
        raise InterchangeProjectionError(
            f"event cursor inconsistency: active_count={event_count} cursor={event_cursor}"
        )
    if metadata.get("event_history_storage_mode") == "eiel_resident_event_sink_v1":
        try:
            sequence_base = int(metadata["typed_lifecycle_event_sequence_base"])
            next_sequence = int(metadata["typed_lifecycle_event_next_sequence"])
            frontier = int(metadata["event_history_frontier"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InterchangeProjectionError(
                "external event-history metadata is incomplete"
            ) from exc
        if (
            min(sequence_base, next_sequence, frontier) < 0
            or next_sequence != sequence_base + event_count
            or frontier != next_sequence
        ):
            raise InterchangeProjectionError(
                "external event-history frontier contradicts pending event rows"
            )
    try:
        modular_storage_from_projection(metadata, tensors)
    except ModularStorageError as exc:
        raise InterchangeProjectionError(
            f"invalid Phase E modular storage projection: {exc}"
        ) from exc


def interchange_projection_from_native_checkpoint(
    projection: Mapping[str, Any],
    *,
    source_runtime: str,
    checkpoint_boundary: str,
    public_output_cursor: int | None = None,
) -> dict[str, Any]:
    checkpoint = _copy_projection(projection)
    validate_native_checkpoint_projection(checkpoint)
    artifact = typed_lifecycle_artifact_from_projection(checkpoint, source=str(source_runtime))
    lifecycle_counts = lifecycle_counts_from_artifact(artifact)
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "source_runtime": str(source_runtime),
        "checkpoint_boundary": str(checkpoint_boundary),
        "public_output_cursor": int(public_output_cursor) if public_output_cursor is not None else int(artifact.cursor),
        "checkpoint_projection": checkpoint,
        "tensor_manifest": _tensor_manifest(checkpoint),
        "digests": {
            "trajectory_state": checkpoint_digest_manifest(checkpoint, layer=DigestLayer.TRAJECTORY_STATE)["digest"],
            "canonical_checkpoint_storage": checkpoint_digest_manifest(
                checkpoint,
                layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
            )["digest"],
            "diagnostic_projection": checkpoint_digest_manifest(
                checkpoint,
                layer=DigestLayer.DIAGNOSTIC_PROJECTION,
            )["digest"],
        },
        "typed_lifecycle_artifact": artifact.to_summary(),
        "lineage_funnel_semantic_rows": lineage_funnel_semantic_rows_from_artifact(artifact),
        "lifecycle_counts": lifecycle_counts,
        "final_lifecycle_summary": final_lifecycle_summary_from_artifact(artifact),
        "final_lineage_summary": final_lineage_summary_semantics_from_projection(
            checkpoint,
            lifecycle_counts=lifecycle_counts,
        ),
        "checkpoint_sidecar_semantics": checkpoint_sidecar_semantics_from_projection(checkpoint),
        "final_summary_json_semantics": final_summary_json_semantics_from_projection(checkpoint),
    }


def native_checkpoint_projection_from_interchange(interchange: Mapping[str, Any]) -> dict[str, Any]:
    if str(interchange.get("schema_version", "") or "") != SCHEMA_VERSION:
        raise InterchangeProjectionError("unsupported interchange projection schema")
    if str(interchange.get("adapter_version", "") or "") != ADAPTER_VERSION:
        raise InterchangeProjectionError("unsupported interchange adapter version")
    projection = _copy_projection(dict(interchange.get("checkpoint_projection") or {}))
    validate_native_checkpoint_projection(projection)
    return projection


def native_engine_from_interchange(
    interchange: Mapping[str, Any],
    *,
    build_directory: str | Path | None = None,
    allow_partial_generation_for_diagnostics: bool = False,
) -> NativeEielEngine:
    return NativeEielEngine.from_checkpoint_projection(
        native_checkpoint_projection_from_interchange(interchange),
        build_directory=build_directory,
        allow_partial_generation_for_diagnostics=allow_partial_generation_for_diagnostics,
    )


def apply_interchange_to_reference_runner(runner: Any, interchange: Mapping[str, Any]) -> dict[str, Any]:
    from perf.cxxbackbone10_resident_state_owner import apply_owner_checkpoint_to_runner

    return apply_owner_checkpoint_to_runner(
        runner,
        native_checkpoint_projection_from_interchange(interchange),
    )
