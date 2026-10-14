"""Host-only settlement of a detached direct-resident physical-batch spool."""

from __future__ import annotations

import copy
import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import torch

from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
    SCHEMA_VERSION as PHYSICAL_CAPTURE_SCHEMA_VERSION,
    SETTLEMENT_OUTPUT_CELL_AXES_V1,
    SETTLEMENT_RECIPE_SCHEMA_VERSION,
    SOURCE_CONFIGURATION_RECIPE_SCHEMA_VERSION,
    decode_birth_template_v1,
)
from engine.campaigns.acs_direct_resident_physical_surface_v1 import (
    CheckpointProjectionCursorV1,
    CudaGenerationConfig,
    DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
    DetachedCellProjectionV1,
    EED_STATE_FIELDS_V1,
    EVENT_COLUMN_NAMES,
    JUNGLE_STATE_FIELDS_V4,
    PreparedDirectResidentWdlJungleV1,
    PreparedDirectResidentWdlV2,
    RESULT_SCHEMA_VERSION,
    WDL_STATE_FIELDS_V2,
    project_direct_resident_wdl_jungle_multicell_settlement_v1,
)
from engine.tw_rngv2.world_identity_v1 import (
    LOGICAL_WORLD_UID_DERIVATION_V1,
    logical_world_uid_v1,
)
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT as TW_RNGV2_NAMESPACE_DEFAULT


SCHEMA_VERSION: Final[str] = "eiel_acs_physical_batch_host_settlement_v1"


class DirectResidentPhysicalSettlementError(RuntimeError):
    """Raised when a detached physical state cannot settle canonically."""


@dataclass(frozen=True, slots=True)
class _DetachedPhysicalSettlementTrajectoryV1:
    """Host-only structural authority consumed by the canonical projector."""

    source: Any
    live_banks: Mapping[str, Any]
    checkpoint_projection_inputs: Mapping[str, Any]


def _fail(message: str) -> DirectResidentPhysicalSettlementError:
    return DirectResidentPhysicalSettlementError(message)


def _validate_rngv2_logical_world_identity_v1(
    checkpoint_payload: Mapping[str, Any], *, cell_ids: tuple[str, ...]
) -> None:
    """Rebind captured RNG coordinates to the restored stable cell identities."""

    decoded = _decode_checkpoint_scalar_tree_v1(checkpoint_payload)
    if not isinstance(decoded, Mapping):
        raise _fail("physical checkpoint RNG identity is malformed")
    semantics_version = decoded.get("tw_rngv2_semantics_version", 1)
    if semantics_version == 1:
        return
    if semantics_version != 2:
        raise _fail("physical checkpoint RNG semantic version is invalid")
    configured_world_uid = decoded.get("tw_rngv2_configured_world_uid")
    namespace_key = decoded.get("tw_rngv2_namespace_key64")
    semantic_epoch = decoded.get("tw_rngv2_semantic_epoch")
    derivation = decoded.get("tw_rngv2_logical_world_uid_derivation")
    captured = decoded.get("tw_rngv2_logical_world_uids")
    if (
        type(configured_world_uid) is not int
        or type(namespace_key) is not int
        or type(semantic_epoch) is not int
        or derivation != LOGICAL_WORLD_UID_DERIVATION_V1
        or isinstance(captured, (str, bytes, bytearray))
        or not isinstance(captured, Sequence)
        or any(type(value) is not int for value in captured)
    ):
        raise _fail("physical checkpoint RNG V2 identity is incomplete")
    if namespace_key != TW_RNGV2_NAMESPACE_DEFAULT or semantic_epoch != 1:
        raise _fail("physical checkpoint RNG V2 namespace or epoch is invalid")
    try:
        expected = tuple(
            logical_world_uid_v1(
                configured_world_uid=configured_world_uid,
                cell_id=cell_id,
            )
            for cell_id in cell_ids
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    if len(set(expected)) != len(expected):
        raise _fail("physical checkpoint RNG V2 logical world identity collided")
    if tuple(captured) != expected:
        raise _fail(
            "physical checkpoint RNG V2 logical world identity disagrees "
            "with restored cell IDs"
        )


def _source_projection_digest(projection: Mapping[str, Any]) -> str:
    from engine.native_runtime.checkpoint_digests import (
        checkpoint_source_projection_authority_manifest,
    )

    return str(
        checkpoint_source_projection_authority_manifest(projection)[
            "source_projection_sha256"
        ]
    )


def _tick_native_terminal_metric_rows_v1(
    projections: Sequence[Mapping[str, Any]],
    *,
    cell_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Use the engine-owned tick-native terminal metric authority."""

    from engine.campaigns.production_engine_native_bindings import (
        _tick_native_terminal_metric_rows_v1 as project_terminal_metrics,
    )

    return project_terminal_metrics(projections, cell_ids=cell_ids)


def _load_bundle_projection(
    path: Path, *, cell_id: str | None, verify_aggregate_payload: bool
) -> Mapping[str, Any]:
    if path.suffix == ".acsaggregate":
        from engine.campaigns.acs_native_aggregate_checkpoint_v1 import (
            load_acs_native_aggregate_checkpoint,
        )

        packed = load_acs_native_aggregate_checkpoint(
            path,
            cell_id=cell_id,
            verify_payload=verify_aggregate_payload,
        )
        projection = packed.get("projection")
        if not isinstance(projection, Mapping):
            raise _fail("aggregate source descriptor returned no projection")
        return dict(projection)
    if path.suffix == ".acsnative":
        from engine.campaigns.acs_native_streamed_checkpoint_v1 import (
            load_acs_native_streamed_checkpoint,
        )

        packed = load_acs_native_streamed_checkpoint(path)
        projection = packed.get("projection")
        if not isinstance(projection, Mapping):
            raise _fail("native source descriptor returned no projection")
        return dict(projection)

    from engine.simulation.checkpoint_bundle_v2 import (
        ARCHITECTURE_B_BUNDLE_KIND,
        checkpoint_bundle_v2_load,
        decode_architecture_b_resume_envelope,
    )

    payload = checkpoint_bundle_v2_load(
        path,
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    envelope = decode_architecture_b_resume_envelope(payload)
    checkpoint_projection = envelope.get("checkpoint_projection")
    import_metadata = envelope.get("controller_import_metadata")
    if not isinstance(checkpoint_projection, Mapping) or not isinstance(
        import_metadata, Mapping
    ):
        raise _fail("strict source bundle resume envelope is invalid")
    projection = dict(checkpoint_projection)
    mechanics_metadata = dict(import_metadata)
    mechanics_metadata.pop("architecture_b_identity", None)
    mechanics_metadata.pop("observer_continuation", None)
    projection["metadata"] = mechanics_metadata
    if envelope.get("source_projection_sha256") != _source_projection_digest(
        projection
    ):
        raise _fail("strict source bundle mechanics digest differs")
    return projection


def load_source_projection_from_recipe_v1(
    recipe: Mapping[str, Any],
    *,
    cell_index: int,
    cell_id: str,
    verified_aggregate_paths: set[str] | None = None,
    byte_counter: dict[str, int] | None = None,
) -> Mapping[str, Any]:
    """Load one canonical skeleton from a tensor-free durable descriptor."""

    if recipe.get("schema_version") != "eiel_bounded_source_descriptor_recipe_v1":
        raise _fail("bounded source descriptor recipe schema is invalid")
    cell_ids = tuple(recipe.get("cell_ids") or ())
    descriptors = recipe.get("descriptors")
    if (
        cell_ids != tuple(dict.fromkeys(cell_ids))
        or not 0 <= cell_index < len(cell_ids)
        or cell_ids[cell_index] != cell_id
        or not isinstance(descriptors, list)
        or len(descriptors) != len(cell_ids)
        or not isinstance(descriptors[cell_index], Mapping)
    ):
        raise _fail("bounded source descriptor recipe cell order is invalid")
    descriptor = descriptors[cell_index]
    kind = descriptor.get("kind")
    path = Path(str(descriptor.get("path") or "")).resolve()
    if not path.is_file():
        raise _fail("bounded source descriptor path is unavailable")
    fingerprint = path.stat()
    before = (
        fingerprint.st_dev,
        fingerprint.st_ino,
        fingerprint.st_size,
        fingerprint.st_mtime_ns,
        fingerprint.st_ctime_ns,
    )
    if kind in {"aggregate_index", "bundle"}:
        verify_aggregate = (
            path.suffix == ".acsaggregate"
            and (
                verified_aggregate_paths is None
                or str(path) not in verified_aggregate_paths
            )
        )
        projection = _load_bundle_projection(
            path,
            cell_id=cell_id,
            verify_aggregate_payload=verify_aggregate,
        )
        if verified_aggregate_paths is not None and path.suffix == ".acsaggregate":
            verified_aggregate_paths.add(str(path))
        if byte_counter is not None:
            byte_counter["source_dependency_bytes_read"] = byte_counter.get(
                "source_dependency_bytes_read", 0
            ) + int(path.stat().st_size)
            byte_counter["source_dependencies_read"] = byte_counter.get(
                "source_dependencies_read", 0
            ) + 1
    elif kind == "component":
        expected_file = descriptor.get("file_sha256")
        raw = path.read_bytes()
        if not isinstance(expected_file, str) or hashlib.sha256(raw).hexdigest() != expected_file:
            raise _fail("component source descriptor file digest differs")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, Mapping):
            raise _fail("component source descriptor payload is malformed")
        projection = dict(value)
        if byte_counter is not None:
            byte_counter["source_dependency_bytes_read"] = byte_counter.get(
                "source_dependency_bytes_read", 0
            ) + len(raw)
            byte_counter["source_dependencies_read"] = byte_counter.get(
                "source_dependencies_read", 0
            ) + 1
    else:
        raise _fail("bounded source descriptor kind is unsupported")
    after_stat = path.stat()
    after = (
        after_stat.st_dev,
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
        after_stat.st_ctime_ns,
    )
    if after != before:
        raise _fail("bounded source descriptor changed during load")
    actual = _source_projection_digest(projection)
    expected = descriptor.get("source_projection_sha256")
    recipe_digests = recipe.get("source_projection_sha256s")
    recipe_expected = (
        recipe_digests.get(cell_id) if isinstance(recipe_digests, Mapping) else None
    )
    for authority in (expected, recipe_expected):
        if authority is not None and authority != actual:
            raise _fail("bounded source projection digest differs")
    return projection


def _field_value(
    loaded_fields: Mapping[str, torch.Tensor], descriptor: Mapping[str, Any]
) -> torch.Tensor:
    field = descriptor.get("field")
    alias = descriptor.get("alias_of")
    name = field if isinstance(field, str) else alias
    if not isinstance(name, str) or name not in loaded_fields:
        raise _fail("physical field-layout descriptor cannot be resolved")
    value = loaded_fields[name]
    if not isinstance(value, torch.Tensor):
        raise _fail("physical field-layout target is not a tensor")
    view = descriptor.get("view")
    if isinstance(alias, str):
        if not isinstance(view, Mapping):
            return value
        shape = view.get("shape")
        stride = view.get("stride")
        storage_offset = view.get("storage_offset")
        if (
            not isinstance(shape, list)
            or not isinstance(stride, list)
            or len(shape) != len(stride)
            or any(type(item) is not int or item < 0 for item in shape)
            or any(type(item) is not int or item < 0 for item in stride)
            or type(storage_offset) is not int
            or storage_offset < 0
        ):
            raise _fail("physical live-bank alias view is malformed")
        try:
            return torch.as_strided(
                value,
                size=tuple(shape),
                stride=tuple(stride),
                storage_offset=int(value.storage_offset()) + storage_offset,
            )
        except RuntimeError as exc:
            raise _fail("physical live-bank alias view escapes captured storage") from exc
    return value


def _batch_one(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0).contiguous()


def _configuration_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(
            **{str(key): _configuration_namespace(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_configuration_namespace(child) for child in value)
    return value


def _decode_checkpoint_scalar_tree_v1(value: Any) -> Any:
    """Restore immutable sequence authority while retaining mapping shape."""

    if isinstance(value, Mapping):
        return {
            str(key): _decode_checkpoint_scalar_tree_v1(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return tuple(_decode_checkpoint_scalar_tree_v1(child) for child in value)
    return value


def _decode_source_configuration_v1(value: Any) -> Any:
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        SOURCE_CONFIGURATION_RECIPE_SCHEMA_VERSION
    ):
        return _configuration_namespace(value)
    if value.get("class") != "CudaGenerationConfig":
        raise _fail("physical source configuration class is unsupported")
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise _fail("physical source configuration fields are malformed")
    try:
        return CudaGenerationConfig(
            **{
                str(name): _configuration_namespace(child)
                for name, child in fields.items()
            }
        )
    except (TypeError, ValueError) as exc:
        raise _fail("physical source configuration cannot be reconstructed") from exc


def _projection_tensor_bytes(projection: Mapping[str, Any]) -> int:
    manifest = projection.get("tensor_manifest")
    if not isinstance(manifest, Mapping):
        tensors = projection.get("tensors")
        if not isinstance(tensors, Mapping):
            return 0
        return sum(
            int(value.numel()) * int(value.element_size())
            for value in tensors.values()
            if isinstance(value, torch.Tensor)
        )
    return sum(
        int(row.get("bytes", 0))
        for row in manifest.values()
        if isinstance(row, Mapping) and type(row.get("bytes")) is int
    )


def _slice_scalar_sequence(value: Any, *, cell: int, cells: int, path: str) -> Any:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _fail(f"{path} is not a physical-cell sequence")
    if len(value) != cells:
        raise _fail(f"{path} does not cover the physical batch")
    return copy.deepcopy(value[cell])


def _settlement_source_payload_v1(recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    source = recipe.get("source")
    if not isinstance(source, Mapping):
        raise _fail("settlement recipe source authority is missing")
    if source.get("time_authority") != "absolute_tick_v1":
        return source
    if source.get("absolute_next_tick") != recipe.get("absolute_next_tick") or any(
        name in source
        for name in (
            "generation_base",
            "logical_generation_count",
            "physical_window_generations",
            "ticks_per_generation",
        )
    ):
        raise _fail("physical settlement public source leaked legacy authority")
    adapter = recipe.get("legacy_compatibility")
    private_source = (
        adapter.get("private_settlement_source")
        if isinstance(adapter, Mapping)
        else None
    )
    configuration = source.get("configuration")
    if not isinstance(private_source, Mapping) or configuration is None:
        raise _fail("physical settlement private source adapter is incomplete")
    return {"configuration": configuration, **dict(private_source)}


def _validate_absolute_tick_private_adapter_v1(
    recipe: Mapping[str, Any],
    *,
    public_scalars: Mapping[str, Any],
    private_scalars: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> None:
    """Verify that the legacy projector facade is derived from absolute ticks."""

    if public_scalars.get("absolute_tick_span_native_owner") is not True:
        return
    authority = recipe.get("absolute_tick_contract")
    adapter = recipe.get("legacy_compatibility")
    # Historical pre-contract fixtures are still readable.  Every current
    # physical capture carries physical-window metadata and must bind the new
    # absolute-tick adapter exactly.
    if not isinstance(authority, Mapping) or not isinstance(adapter, Mapping):
        if "physical_window_generations" not in source_payload:
            return
        raise _fail("physical settlement absolute-tick adapter is missing")
    absolute_next_tick = recipe.get("absolute_next_tick")
    start = authority.get("segment_start_next_tick")
    end = authority.get("segment_end_next_tick")
    span = authority.get("span_ticks")
    interval = adapter.get("interval_ticks")
    source_window_ticks = authority.get("source_window_ticks")
    source_window_count = authority.get("source_window_count")
    if (
        authority.get("schema_version")
        != "eiel_absolute_tick_physical_capture_contract_v1"
        or authority.get("time_authority") != "absolute_tick_v1"
        or adapter.get("schema_version")
        != "eiel_absolute_tick_private_settlement_adapter_v1"
        or adapter.get("authority") != "private_legacy_compatibility_only"
        or type(absolute_next_tick) is not int
        or type(start) is not int
        or type(end) is not int
        or type(span) is not int
        or type(interval) is not int
        or type(source_window_ticks) is not int
        or type(source_window_count) is not int
        or not 0 <= start < end == absolute_next_tick
        or span != end - start
        or interval < 1
        or source_window_ticks < 1
        or source_window_count
        != (
            (start % source_window_ticks) + span + source_window_ticks - 1
        ) // source_window_ticks
        or adapter.get("derived_from_absolute_next_tick") != absolute_next_tick
        or public_scalars.get("physical_window_alignment") != "absolute_tick_v1"
        or public_scalars.get("tick_native_start_next_tick") != start
        or public_scalars.get("tick_native_end_next_tick") != end
        or public_scalars.get("tick_span_start") != start
        or public_scalars.get("tick_span_end") != end
        or public_scalars.get("tick_span_count") != span
        or authority.get("native_tick_iterations") != span
        or public_scalars.get("native_tick_iterations") != span
        or authority.get("source_window_ticks")
        != public_scalars.get("source_window_ticks")
        or authority.get("source_window_count")
        != public_scalars.get("source_window_count")
    ):
        raise _fail("physical settlement absolute-tick authority is invalid")
    if any(
        name in public_scalars
        for name in (
            "completed_generations",
            "logical_generation_count",
            "physical_window_generations",
            "settled_window_count",
            "terminal_generation_boundary",
            "tick_native_source_prefix_by_cell",
            "window_count",
        )
    ):
        raise _fail("physical settlement public native contract leaked legacy fields")
    terminal_generation, terminal_tick = divmod(absolute_next_tick, interval)
    if terminal_tick == 0:
        expected_generation = terminal_generation - 1
        private_span_end = interval
        expected_boundary = True
    else:
        expected_generation = terminal_generation
        private_span_end = terminal_tick
        expected_boundary = False
    private_span_start = max(0, private_span_end - span)
    expected_count = private_span_end - private_span_start
    expected_phase = (
        "generation_boundary" if private_span_start == 0 else "partial_generation"
    )
    expected_adapter = {
        "source_generation": expected_generation,
        "source_tick": private_span_start,
        "tick_span_start": private_span_start,
        "tick_span_end": private_span_end,
        "tick_span_count": expected_count,
        "terminal_generation_boundary": expected_boundary,
    }
    if expected_generation < 0 or any(
        type(adapter.get(name)) is not type(value) or adapter.get(name) != value
        for name, value in expected_adapter.items()
    ):
        raise _fail("physical settlement private coordinates are not tick-derived")
    expected_scalars = {
        "logical_generation_count": int(
            source_payload.get("logical_generation_count", 0)
        ),
        "physical_window_generations": int(
            source_payload.get("physical_window_generations", 0)
        ),
        "window_count": int(source_payload.get("logical_generation_count", 0)),
        "completed_generations": 1,
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "tick_span_start": private_span_start,
        "tick_span_end": private_span_end,
        "tick_span_count": expected_count,
        "terminal_generation_boundary": expected_boundary,
    }
    if any(
        type(private_scalars.get(name)) is not type(value)
        or private_scalars.get(name) != value
        for name, value in expected_scalars.items()
    ):
        raise _fail("physical settlement private direct-result facade is invalid")
    prefixes = private_scalars.get("tick_native_source_prefix_by_cell")
    if not isinstance(prefixes, Sequence) or isinstance(
        prefixes, (str, bytes, bytearray)
    ):
        raise _fail("physical settlement private source prefixes are missing")
    for prefix in prefixes:
        metadata = prefix.get("metadata") if isinstance(prefix, Mapping) else None
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("generation") != expected_generation
            or metadata.get("tick") != private_span_start
            or metadata.get("prediction_tick") != private_span_start
            or metadata.get("agent_grid_runner_phase") != expected_phase
        ):
            raise _fail("physical settlement source prefix is not tick-derived")


def _cell_direct_result(
    recipe: Mapping[str, Any],
    *,
    cell: int,
    cell_id: str,
    cells: int,
    loaded_fields: Mapping[str, torch.Tensor],
    layout: Mapping[str, Any],
    live_banks: Mapping[str, Any],
) -> dict[str, Any]:
    public_scalars = recipe.get("direct_result_scalars")
    if not isinstance(public_scalars, Mapping):
        raise _fail("settlement recipe direct-result scalars are missing")
    source_payload = _settlement_source_payload_v1(recipe)
    adapter = recipe.get("legacy_compatibility")
    private_scalars = (
        adapter.get("private_settlement_direct_result_scalars")
        if isinstance(adapter, Mapping)
        else None
    )
    scalars = (
        private_scalars if isinstance(private_scalars, Mapping) else public_scalars
    )
    result = copy.deepcopy(dict(scalars))
    _validate_absolute_tick_private_adapter_v1(
        recipe,
        public_scalars=public_scalars,
        private_scalars=scalars,
        source_payload=source_payload,
    )
    legacy_pre_contract_spool = "physical_window_generations" not in source_payload
    result.setdefault("physical_window_generations", 1)
    logical_generation_count = int(source_payload.get("logical_generation_count", 0))
    result_defaults = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "logical_generation_count": logical_generation_count,
        "window_count": logical_generation_count,
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, value in result_defaults.items():
        if legacy_pre_contract_spool:
            result.setdefault(name, value)
        elif type(result.get(name)) is not type(value) or result.get(name) != value:
            raise _fail("physical direct-result settlement authority is incomplete")
    for name in (
        "event_history_frontier_by_cell",
        "event_history_prefix_references_by_cell",
        "event_rows_by_cell",
        "tick_native_source_prefix_by_cell",
    ):
        if name in scalars:
            result[name] = [
                _slice_scalar_sequence(
                    scalars[name], cell=cell, cells=cells, path=f"direct_result.{name}"
                )
            ]
    result["source_projection_cell_ids"] = [cell_id]
    frontiers = result.get("event_history_frontier_by_cell")
    if isinstance(frontiers, list) and len(frontiers) == 1:
        result["event_history_frontier"] = copy.deepcopy(frontiers[0])
    output_layout = layout.get("settlement_outputs")
    if not isinstance(output_layout, Mapping):
        raise _fail("settlement output field layout is missing")
    for name, axis in SETTLEMENT_OUTPUT_CELL_AXES_V1.items():
        descriptor = output_layout.get(name)
        if not isinstance(descriptor, Mapping):
            continue
        selected = _field_value(loaded_fields, descriptor)
        if (
            name == "runtime_bookkeeping_by_tick"
            and result.get("absolute_tick_span_native_owner") is True
        ):
            span_count = result.get("tick_span_count")
            if (
                type(span_count) is not int
                or span_count < 1
                or selected.ndim < 1
                or selected.shape[0] < span_count
            ):
                raise _fail("physical runtime bookkeeping cannot cover tick span")
            selected = selected.narrow(0, 0, span_count)
        if axis == 1:
            result[name] = selected.unsqueeze(1).contiguous()
        else:
            result[name] = _batch_one(selected)
    sequence_base = live_banks.get("event_sequence_base")
    if isinstance(sequence_base, torch.Tensor):
        result["event_sequence_base"] = sequence_base
    return result


def build_physical_batch_settlement_cursor_v1(
    capture: Mapping[str, Any],
    *,
    reader: Any,
    cancel_event: threading.Event | None = None,
    source_byte_counter: dict[str, int] | None = None,
    source_projection_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> CheckpointProjectionCursorV1:
    """Create a strict B1-at-a-time canonical settlement cursor.

    ``reader`` is the verified packed reader.  It must expose
    ``load_selected_cell(index)`` and return the same mapping shape as the
    packed module's strict selected-cell loader.
    """

    if capture.get("schema_version") != PHYSICAL_CAPTURE_SCHEMA_VERSION:
        raise _fail("physical capture envelope schema is invalid")
    recipe = capture.get("settlement_recipe")
    if not isinstance(recipe, Mapping) or recipe.get("schema_version") != SETTLEMENT_RECIPE_SCHEMA_VERSION:
        raise _fail("physical settlement recipe schema is invalid")
    cell_ids = tuple(recipe.get("cell_ids") or ())
    cells = len(cell_ids)
    if not cell_ids or cells != len(set(cell_ids)):
        raise _fail("physical settlement recipe cell order is invalid")
    checkpoint_payload = recipe.get("checkpoint_projection_inputs")
    if not isinstance(checkpoint_payload, Mapping):
        raise _fail("physical settlement checkpoint recipe is missing")
    _validate_rngv2_logical_world_identity_v1(
        {
            name: value
            for name, value in checkpoint_payload.items()
            if name != "resident_birth_state"
        },
        cell_ids=cell_ids,
    )
    layout = recipe.get("field_layout")
    if not isinstance(layout, Mapping):
        raise _fail("physical settlement recipe field layout is missing")
    source_recipe = recipe.get("source_descriptor_recipe")
    if not isinstance(source_recipe, Mapping):
        raise _fail("physical settlement source recipe is missing")
    verified_source_aggregates: set[str] = set()

    def project_one_cell(cell: int, cell_id: str) -> DetachedCellProjectionV1:
        if cancel_event is not None and cancel_event.is_set():
            raise _fail("physical host settlement was cancelled")
        loaded = reader.load_selected_cell(cell)
        loaded_fields = loaded.get("fields") if isinstance(loaded, Mapping) else None
        if not isinstance(loaded_fields, Mapping):
            raise _fail("verified physical reader returned no selected fields")
        override = (
            source_projection_overrides.get(cell_id)
            if isinstance(source_projection_overrides, Mapping)
            else None
        )
        canonical = (
            dict(override)
            if isinstance(override, Mapping)
            else load_source_projection_from_recipe_v1(
                source_recipe,
                cell_index=cell,
                cell_id=cell_id,
                verified_aggregate_paths=verified_source_aggregates,
                byte_counter=source_byte_counter,
            )
        )
        if isinstance(override, Mapping):
            source_digests = source_recipe.get("source_projection_sha256s")
            expected_source_digest = (
                source_digests.get(cell_id)
                if isinstance(source_digests, Mapping)
                else None
            )
            from engine.native_runtime.checkpoint_digests import (
                checkpoint_source_projection_authority_manifest,
            )

            observed_source_digest = (
                checkpoint_source_projection_authority_manifest(canonical).get(
                    "source_projection_sha256"
                )
            )
            if observed_source_digest != expected_source_digest:
                raise _fail("physical settlement source override digest differs")

        live_layout = layout.get("live_bank_tensors")
        event_layout = layout.get("event_columns")
        if not isinstance(live_layout, Mapping) or not isinstance(event_layout, Mapping):
            raise _fail("physical live-bank field layout is incomplete")
        live_banks = {
            str(name): _batch_one(_field_value(loaded_fields, descriptor))
            for name, descriptor in live_layout.items()
            if isinstance(descriptor, Mapping)
        }
        live_banks["event_columns"] = [
            _batch_one(_field_value(loaded_fields, event_layout[name]))
            for name in EVENT_COLUMN_NAMES
        ]
        live_scalars = recipe.get("live_bank_scalars")
        if not isinstance(live_scalars, Mapping):
            raise _fail("physical live-bank scalar recipe is missing")
        live_banks.update(copy.deepcopy(dict(live_scalars)))
        founder_profiles = live_banks.get("genome_v2_founder_profiles")
        if founder_profiles is not None:
            if (
                isinstance(founder_profiles, (str, bytes, bytearray))
                or not isinstance(founder_profiles, Sequence)
                or len(founder_profiles) != cells
                or not isinstance(founder_profiles[cell], Mapping)
            ):
                raise _fail("physical settlement founder profiles do not cover B136")
            live_banks["genome_v2_founder_profiles"] = (
                copy.deepcopy(dict(founder_profiles[cell])),
            )

        outputs: dict[str, torch.Tensor] = {}
        output_layout = layout.get("settlement_outputs")
        if not isinstance(output_layout, Mapping):
            raise _fail("physical settlement output layout is missing")
        for name, descriptor in output_layout.items():
            if not isinstance(descriptor, Mapping):
                raise _fail("physical settlement output descriptor is malformed")
            selected = _field_value(loaded_fields, descriptor)
            outputs[str(name)] = (
                selected.unsqueeze(1).contiguous()
                if SETTLEMENT_OUTPUT_CELL_AXES_V1.get(name) == 1
                else _batch_one(selected)
            )

        source_payload = _settlement_source_payload_v1(recipe)
        checkpoint_payload = recipe.get("checkpoint_projection_inputs")
        if not isinstance(checkpoint_payload, Mapping):
            raise _fail("physical settlement source/checkpoint recipe is incomplete")
        checkpoint_inputs = _decode_checkpoint_scalar_tree_v1(
            {
                name: value
                for name, value in checkpoint_payload.items()
                if name != "resident_birth_state"
            }
        )
        if not isinstance(checkpoint_inputs, dict):
            raise _fail("physical checkpoint scalar authority is malformed")
        birth_recipe = checkpoint_payload.get("resident_birth_state")
        if not isinstance(birth_recipe, Mapping):
            raise _fail("physical settlement birth template is missing")
        checkpoint_inputs["resident_birth_state"] = decode_birth_template_v1(
            birth_recipe
        )
        checkpoint_inputs["pre_generation_canonical_projection"] = canonical
        checkpoint_inputs["pre_generation_canonical_projections"] = (canonical,)
        selected_physical_cell_index = recipe.get(
            "selected_physical_cell_index", cell
        )
        if (
            type(selected_physical_cell_index) is not int
            or selected_physical_cell_index < 0
        ):
            raise _fail("physical settlement selected world ordinal is invalid")
        checkpoint_inputs["genome_v2_causal_physical_cell_ordinals"] = (
            selected_physical_cell_index,
        )
        source = SimpleNamespace(
            outputs=outputs,
            configuration=_decode_source_configuration_v1(
                source_payload["configuration"]
            ),
            generation_base=int(source_payload["generation_base"]),
            logical_generation_count=int(source_payload["logical_generation_count"]),
            # Physical-capture schema v1 was admitted only on the frozen g1
            # production route.  Retain that exact meaning for pre-field
            # diagnostic spools while all new captures carry it explicitly.
            physical_window_generations=int(
                source_payload.get("physical_window_generations", 1)
            ),
            ticks_per_generation=int(source_payload["ticks_per_generation"]),
        )
        trajectory = _DetachedPhysicalSettlementTrajectoryV1(
            source=source,
            live_banks=live_banks,
            checkpoint_projection_inputs=checkpoint_inputs,
        )
        direct_result = _cell_direct_result(
            recipe,
            cell=cell,
            cell_id=cell_id,
            cells=cells,
            loaded_fields=loaded_fields,
            layout=layout,
            live_banks=live_banks,
        )
        completed = direct_result.get("completed_generations")
        if type(completed) is not int or completed < 1:
            raise _fail("physical settlement completed-generation count is invalid")

        wdl_layout = layout.get("wdl")
        if not isinstance(wdl_layout, Mapping):
            raise _fail("physical WDL settlement layout is missing")
        next_step = _batch_one(
            _field_value(loaded_fields, wdl_layout["next_absolute_step"])
        ).reshape(1)
        agent_keys = _field_value(loaded_fields, wdl_layout["agent_keys"])
        agent_flags = _field_value(loaded_fields, wdl_layout["agent_flags"])
        reporting: list[torch.Tensor] = [torch.empty((0,)) for _ in range(8)]
        reporting[6] = torch.zeros(
            (1, completed, *agent_keys.shape), dtype=agent_keys.dtype
        )
        reporting[7] = torch.zeros(
            (1, completed, *agent_flags.shape), dtype=agent_flags.dtype
        )
        reporting[6][0, completed - 1].copy_(agent_keys)
        reporting[7][0, completed - 1].copy_(agent_flags)
        wdl = PreparedDirectResidentWdlV2(
            patch=trajectory,
            state_tensors=tuple(live_banks[name] for name in WDL_STATE_FIELDS_V2),
            schedule_tensors=(torch.zeros((1,), dtype=torch.int64), next_step),
            reporting_tensors=tuple(reporting),
            config={"cuda_capability": checkpoint_inputs.get("cuda_capability", {})},
            generation_rows=completed,
            initial_projection=canonical,
            cell_ids=(cell_id,),
            source_projections=(canonical,),
        )
        jungle_layout = layout.get("jungle")
        eed_layout = layout.get("eed")
        if not isinstance(jungle_layout, Mapping) or not isinstance(eed_layout, Mapping):
            raise _fail("physical Jungle/EED settlement layout is malformed")
        jungle_state = tuple(
            _batch_one(_field_value(loaded_fields, jungle_layout[name]))
            for name in JUNGLE_STATE_FIELDS_V4
        )
        eed_state = (
            tuple(
                _batch_one(_field_value(loaded_fields, eed_layout[name]))
                for name in EED_STATE_FIELDS_V1
            )
            if eed_layout
            else None
        )
        combined = PreparedDirectResidentWdlJungleV1(
            wdl=wdl,
            jungle_state_tensors=jungle_state,
            jungle_reporting_tensors=(),
            jungle_config={},
            eed_state_tensors=eed_state,
            eed_config={} if eed_state is not None else None,
        )
        source_prefixes = direct_result.get("tick_native_source_prefix_by_cell")
        prefix_generation = None
        if isinstance(source_prefixes, Sequence) and not isinstance(
            source_prefixes, (str, bytes, bytearray)
        ) and len(source_prefixes) == 1 and isinstance(source_prefixes[0], Mapping):
            metadata = source_prefixes[0].get("metadata")
            if isinstance(metadata, Mapping):
                prefix_generation = metadata.get("generation")
        if prefix_generation is not None and (
            type(prefix_generation) is not int or prefix_generation < 0
        ):
            raise _fail("physical source-prefix generation is malformed")
        compatibility = recipe.get("legacy_compatibility")
        fallback_generation = (
            compatibility.get("generation_base_override")
            if isinstance(compatibility, Mapping)
            else recipe.get("generation_base_override")
        )
        if prefix_generation is None and (
            type(fallback_generation) is not int or fallback_generation < 0
        ):
            raise _fail("physical private generation adapter is malformed")
        settlement = project_direct_resident_wdl_jungle_multicell_settlement_v1(
            combined,
            direct_result=direct_result,
            completed_generation_count=completed,
            reporting_enabled=bool(recipe.get("reporting_enabled")),
            generation_base_override=(
                int(prefix_generation)
                if prefix_generation is not None
                else int(fallback_generation)
            ),
            bounded_projection_cursor=False,
        )
        projections = tuple(settlement.checkpoint_projections)
        if len(projections) != 1 or not isinstance(projections[0], Mapping):
            raise _fail("B1 host settlement did not return one canonical projection")
        projection = projections[0]
        raw_compact_rows = (
            _tick_native_terminal_metric_rows_v1(
                projections,
                cell_ids=(cell_id,),
            )
            if direct_result.get("absolute_tick_span_native_owner") is True
            else tuple(settlement.compact_result_rows)
        )
        compact_rows = tuple(
            {"cell_id": cell_id, **dict(row)}
            for row in raw_compact_rows
        )
        qacct_rows = tuple(settlement.lifecycle_qacct_receipts)
        if len(qacct_rows) > 1 or (
            qacct_rows and not isinstance(qacct_rows[0], Mapping)
        ):
            raise _fail("B1 host settlement returned invalid lifecycle QAcct rows")
        qacct = dict(qacct_rows[0]) if qacct_rows else None
        return DetachedCellProjectionV1(
            schema_version=DETACHED_CELL_PROJECTION_SCHEMA_VERSION_V1,
            cell_id=cell_id,
            cell_index=cell,
            projection=projection,
            compact_result_rows=compact_rows,
            segment_artifacts=tuple(settlement.segment_artifacts),
            lifecycle_qacct_receipt=qacct,
            terminal_generation=int(settlement.terminal_generation),
            terminal_tick=int(settlement.terminal_tick),
            live_projection_bytes=_projection_tensor_bytes(projection),
        )

    return CheckpointProjectionCursorV1.begin(
        owner={"schema_version": SCHEMA_VERSION},
        coordinate=int(recipe["absolute_next_tick"]),
        ordered_cell_ids=cell_ids,
        factory=project_one_cell,
    )


__all__ = [
    "DirectResidentPhysicalSettlementError",
    "SCHEMA_VERSION",
    "build_physical_batch_settlement_cursor_v1",
    "load_source_projection_from_recipe_v1",
]
