"""Physical-B checkpoint capture for tick-native direct-resident owners.

This module owns only the producer/device detachment boundary.  It writes one
field-major physical-batch spool and returns a JSON-only settlement recipe.
It does not settle cells, compute canonical checkpoint digests, or publish an
``.acsaggregate``.  Those operations belong to the detached host worker.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from engine.campaigns.acs_direct_resident_field_manifest_v1 import (
    EVENT_COLUMN_ROW,
    build_direct_resident_field_manifest_v1,
)
from engine.campaigns.acs_physical_batch_packed_checkpoint_v1 import (
    SCHEMA_VERSION as PACKED_SCHEMA_VERSION,
    ZeroFillFieldV1,
    capture_physical_batch_packed_checkpoint_v1,
)
from engine.campaigns.acs_direct_resident_physical_surface_v1 import (
    CudaGenerationBirthState,
    CudaGenerationBirthStateReceipt,
    EED_STATE_FIELDS_V1,
    EVENT_COLUMN_NAMES,
    JUNGLE_STATE_FIELDS_V4,
    RESULT_SCHEMA_VERSION,
)


SCHEMA_VERSION: Final[str] = (
    "eiel_acs_direct_resident_physical_batch_capture_v1"
)
SETTLEMENT_RECIPE_SCHEMA_VERSION: Final[str] = (
    "eiel_acs_physical_batch_settlement_recipe_v1"
)
SOURCE_DESCRIPTOR_RECIPE_SCHEMA_VERSION: Final[str] = (
    "eiel_bounded_source_descriptor_recipe_v1"
)
SOURCE_CONFIGURATION_RECIPE_SCHEMA_VERSION: Final[str] = (
    "eiel_direct_resident_source_configuration_recipe_v1"
)
EXECUTION_ROUTE_IDENTITY_V1: Final[str] = "tick_native_execution_route_v1"
PHYSICAL_BATCH_CELLS_V1: Final[int] = 136
EVENT_WINDOW_CAPACITY_V1: Final[int] = 41_804
EVENT_WINDOW_PHYSICAL_BYTES_V1: Final[int] = 801_633_504
EVENT_ROW_BYTES_V1: Final[int] = 141

SETTLEMENT_OUTPUT_CELL_AXES_V1: Final[Mapping[str, int]] = {
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

_DIRECT_RESULT_SCALAR_FIELDS_V1: Final[tuple[str, ...]] = (
    "absolute_tick_span_native_owner",
    "completed_generations",
    "event_history_frontier_by_cell",
    "event_history_prefix_references_by_cell",
    "event_history_reference",
    "event_history_storage_mode",
    "event_rows_by_cell",
    "external_python_window_boundaries",
    "direct_cpp_window_core_dispatch",
    "eed_native_owner",
    "hidden_fallback",
    "logical_generation_count",
    "native_tick_iterations",
    "owner",
    "physical_window_generations",
    "physical_window_alignment",
    "production_native_chunk_loop",
    "runtime_jit",
    "schema_version",
    "settled_window_count",
    "source_projection_cell_ids",
    "source_window_count",
    "source_window_ticks",
    "terminal_generation_boundary",
    "tick_native_start_next_tick",
    "tick_native_end_next_tick",
    "tick_native_source_prefix_by_cell",
    "tick_span_count",
    "tick_span_end",
    "tick_span_start",
    "window_count",
    "wdl_jungle_native_owner",
)

_CHECKPOINT_INPUT_OMISSIONS_V1: Final[frozenset[str]] = frozenset(
    {
        "pre_generation_canonical_projection",
        "pre_generation_canonical_projections",
        "source_projection_cursor_factory",
    }
)


class DirectResidentPhysicalCaptureError(RuntimeError):
    """Raised when the physical owner cannot detach without semantic loss."""


def _fail(message: str) -> DirectResidentPhysicalCaptureError:
    return DirectResidentPhysicalCaptureError(message)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _fail("physical settlement recipe is not deterministic JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _encode_scalar_tree(value: Any, *, path: str) -> Any:
    """Encode a tensor-free deterministic scalar tree.

    Dataclasses are retained as named field mappings.  Only the resident birth
    template may contain tensors; its tensors are encoded separately below.
    """

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _fail(f"{path} contains a non-finite float")
        return value
    if isinstance(value, torch.Tensor):
        raise _fail(f"{path} contains a tensor outside the packed field surface")
    if callable(value):
        raise _fail(f"{path} contains a callable")
    if dataclasses.is_dataclass(value):
        return {
            field.name: _encode_scalar_tree(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or not key:
                raise _fail(f"{path} has a non-string or empty mapping key")
            out[key] = _encode_scalar_tree(child, path=f"{path}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _encode_scalar_tree(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise _fail(f"{path} contains unsupported scalar authority {type(value).__name__}")


def _encode_birth_template(value: Any, *, path: str) -> Any:
    """Encode the known tensor-empty birth schema without tensor ownership."""

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.numel() != 0:
            raise _fail(f"{path} birth template tensor must be empty CPU storage")
        return {
            "kind": "empty_tensor",
            "dtype": str(value.dtype),
        }
    if dataclasses.is_dataclass(value):
        class_id = f"{type(value).__module__}.{type(value).__qualname__}"
        admitted = {
            "engine.perf.cuda_architecture_b_generation_birth_state.CudaGenerationBirthState",
            "engine.perf.cuda_architecture_b_generation_birth_state.CudaGenerationBirthStateReceipt",
            "perf.cuda_architecture_b_generation_birth_state.CudaGenerationBirthState",
            "perf.cuda_architecture_b_generation_birth_state.CudaGenerationBirthStateReceipt",
        }
        if class_id not in admitted:
            raise _fail(f"{path} contains an unsupported birth-template dataclass")
        return {
            "kind": "dataclass",
            "class": class_id.rsplit(".", 1)[-1],
            "fields": {
                field.name: _encode_birth_template(
                    getattr(value, field.name), path=f"{path}.{field.name}"
                )
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": {
                str(key): _encode_birth_template(
                    child, path=f"{path}.{key}"
                )
                for key, child in value.items()
            },
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _encode_birth_template(child, path=f"{path}[{index}]")
                for index, child in enumerate(value)
            ],
        }
    return {
        "kind": "scalar",
        "value": _encode_scalar_tree(value, path=path),
    }


def decode_birth_template_v1(value: Mapping[str, Any]) -> Any:
    """Reconstruct the tensor-empty birth schema on the detached worker."""

    if not isinstance(value, Mapping):
        raise _fail("birth template recipe node must be a mapping")
    kind = value.get("kind")
    if kind == "empty_tensor":
        dtype_name = value.get("dtype")
        dtype = {
            str(candidate): candidate
            for candidate in (
                torch.bool,
                torch.float16,
                torch.float32,
                torch.float64,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.uint16,
                torch.uint32,
                torch.uint64,
            )
        }.get(dtype_name)
        if dtype is None:
            raise _fail("birth template dtype is unsupported")
        return torch.empty((0,), dtype=dtype, device="cpu")
    if kind == "scalar":
        return value.get("value")
    if kind == "mapping":
        items = value.get("items")
        if not isinstance(items, Mapping):
            raise _fail("birth template mapping node is malformed")
        return {str(key): decode_birth_template_v1(child) for key, child in items.items()}
    if kind == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise _fail("birth template tuple node is malformed")
        return tuple(decode_birth_template_v1(child) for child in items)
    if kind == "dataclass":
        fields = value.get("fields")
        if not isinstance(fields, Mapping):
            raise _fail("birth template dataclass node is malformed")
        class_type = {
            "CudaGenerationBirthState": CudaGenerationBirthState,
            "CudaGenerationBirthStateReceipt": CudaGenerationBirthStateReceipt,
        }.get(value.get("class"))
        if class_type is None:
            raise _fail("birth template dataclass identity is unsupported")
        return class_type(
            **{
                str(name): decode_birth_template_v1(child)
                for name, child in fields.items()
            }
        )
    raise _fail("birth template recipe node kind is unsupported")


def _configuration_payload(configuration: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(configuration):
        class_id = f"{type(configuration).__module__}.{type(configuration).__qualname__}"
        if class_id not in {
            "engine.perf.cuda_architecture_b_generation.CudaGenerationConfig",
            "perf.cuda_architecture_b_generation.CudaGenerationConfig",
        }:
            raise _fail("resident source configuration dataclass is unsupported")
        raw = {
            field.name: getattr(configuration, field.name)
            for field in dataclasses.fields(configuration)
        }
        encoded = _encode_scalar_tree(raw, path="source.configuration.fields")
        if not isinstance(encoded, dict):
            raise _fail("resident source configuration fields are not a mapping")
        return {
            "schema_version": SOURCE_CONFIGURATION_RECIPE_SCHEMA_VERSION,
            "class": "CudaGenerationConfig",
            "fields": encoded,
        }
    elif isinstance(configuration, Mapping):
        raw = dict(configuration)
    elif hasattr(configuration, "__dict__"):
        raw = vars(configuration)
    else:
        raise _fail("resident source configuration is not serializable")
    encoded = _encode_scalar_tree(raw, path="source.configuration")
    if not isinstance(encoded, dict):
        raise _fail("resident source configuration payload is not a mapping")
    return encoded


def _resolve_owner(prepared: Any) -> tuple[Any, Any, Any]:
    physical = getattr(prepared, "physical_window_arguments", prepared)
    composite = physical if hasattr(physical, "wdl") else None
    wdl = getattr(physical, "wdl", physical)
    trajectory = getattr(wdl, "patch", None)
    if trajectory is None or not isinstance(getattr(trajectory, "live_banks", None), Mapping):
        raise _fail("physical checkpoint owner is not the WDL/Jungle direct-resident route")
    return composite, wdl, trajectory


def _direct_result_capture_contract_v1(
    direct_result: Mapping[str, Any],
    *,
    source: Any,
    absolute_next_tick: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Validate native authority and derive the private settlement facade.

    Absolute ticks remain authoritative.  The returned direct-result scalars
    contain a generation-shaped facade only because the existing one-cell
    checkpoint projector is a private legacy ABI.  Every facade coordinate is
    derived here from ``absolute_next_tick`` and cannot affect native dispatch.
    """

    direct_scalars = {
        name: _encode_scalar_tree(direct_result[name], path=f"direct_result.{name}")
        for name in _DIRECT_RESULT_SCALAR_FIELDS_V1
        if name in direct_result
    }
    common_expected = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    if any(
        type(direct_scalars.get(name)) is not type(expected)
        or direct_scalars.get(name) != expected
        for name, expected in common_expected.items()
    ):
        raise _fail("physical capture direct-result contract is incomplete")

    tick_native = direct_scalars.get("absolute_tick_span_native_owner") is True
    if not tick_native:
        logical_generation_count = int(getattr(source, "logical_generation_count", 0))
        expected = {
            **common_expected,
            "physical_window_generations": 1,
            "logical_generation_count": logical_generation_count,
            "window_count": logical_generation_count,
            "physical_window_alignment": "absolute_generation_lattice_v1",
        }
        if any(
            type(direct_scalars.get(name)) is not type(value)
            or direct_scalars.get(name) != value
            for name, value in expected.items()
        ):
            raise _fail("physical capture direct-result contract is incomplete")
        return direct_scalars, None, None

    raw_start = direct_scalars.get("tick_native_start_next_tick")
    raw_end = direct_scalars.get("tick_native_end_next_tick")
    span_start = direct_scalars.get("tick_span_start")
    span_end = direct_scalars.get("tick_span_end")
    span_count = direct_scalars.get("tick_span_count")
    native_iterations = direct_scalars.get("native_tick_iterations")
    source_window_ticks = direct_scalars.get("source_window_ticks")
    source_window_count = direct_scalars.get("source_window_count")
    if (
        type(absolute_next_tick) is not int
        or absolute_next_tick < 1
        or type(raw_start) is not int
        or type(raw_end) is not int
        or type(span_start) is not int
        or type(span_end) is not int
        or type(span_count) is not int
        or type(native_iterations) is not int
        or type(source_window_ticks) is not int
        or type(source_window_count) is not int
        or not 0 <= raw_start < raw_end == absolute_next_tick
        or span_start != raw_start
        or span_end != raw_end
        or span_count != raw_end - raw_start
        or native_iterations != span_count
        or source_window_ticks != getattr(source, "source_window_ticks", None)
        or source_window_ticks < 1
        or source_window_count
        != (
            (raw_start % source_window_ticks)
            + span_count
            + source_window_ticks
            - 1
        ) // source_window_ticks
        or direct_scalars.get("physical_window_alignment") != "absolute_tick_v1"
        or type(direct_scalars.get("terminal_generation_boundary")) is not bool
    ):
        raise _fail("physical capture absolute-tick direct-result contract is incomplete")

    interval_ticks = int(getattr(source, "ticks_per_generation", 0))
    if interval_ticks < 1:
        raise _fail("physical capture private settlement interval is invalid")
    terminal_generation, terminal_tick = divmod(absolute_next_tick, interval_ticks)
    if terminal_tick == 0:
        settlement_generation = terminal_generation - 1
        private_span_end = interval_ticks
        settlement_terminal_boundary = True
    else:
        settlement_generation = terminal_generation
        private_span_end = terminal_tick
        settlement_terminal_boundary = False
    private_span_start = max(0, private_span_end - span_count)
    settlement_tick_count = private_span_end - private_span_start
    if settlement_generation < 0 or settlement_tick_count < 1:
        raise _fail("physical capture private settlement coordinates are invalid")

    logical_generation_count = int(getattr(source, "logical_generation_count", 0))
    physical_window_generations = int(
        getattr(source, "physical_window_generations", 0)
    )
    if logical_generation_count < 1 or physical_window_generations != 1:
        raise _fail("physical capture private settlement envelope is invalid")

    prefixes = direct_scalars.get("tick_native_source_prefix_by_cell")
    if not isinstance(prefixes, list):
        raise _fail("physical capture lacks tick-native source prefix authority")
    normalized_prefixes: list[dict[str, Any]] = []
    for prefix in prefixes:
        if not isinstance(prefix, Mapping) or not isinstance(
            prefix.get("metadata"), Mapping
        ):
            raise _fail("physical capture tick-native source prefix is incomplete")
        normalized = dict(prefix)
        metadata = dict(prefix["metadata"])
        metadata.update(
            {
                "generation": settlement_generation,
                "tick": private_span_start,
                "prediction_tick": private_span_start,
                "agent_grid_runner_phase": (
                    "generation_boundary"
                    if private_span_start == 0
                    else "partial_generation"
                ),
            }
        )
        normalized["metadata"] = metadata
        normalized_prefixes.append(normalized)

    public_direct_scalars = dict(direct_scalars)
    for legacy_name in (
        "completed_generations",
        "logical_generation_count",
        "physical_window_generations",
        "settled_window_count",
        "terminal_generation_boundary",
        "tick_native_source_prefix_by_cell",
        "window_count",
    ):
        public_direct_scalars.pop(legacy_name, None)

    private_direct_scalars = dict(direct_scalars)
    private_direct_scalars.update(
        {
            "physical_window_generations": physical_window_generations,
            "logical_generation_count": logical_generation_count,
            "window_count": logical_generation_count,
            "completed_generations": 1,
            "physical_window_alignment": "absolute_generation_lattice_v1",
            "tick_span_start": private_span_start,
            "tick_span_end": private_span_end,
            "tick_span_count": settlement_tick_count,
            "terminal_generation_boundary": settlement_terminal_boundary,
            "tick_native_source_prefix_by_cell": normalized_prefixes,
        }
    )
    absolute_contract = {
        "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
        "time_authority": "absolute_tick_v1",
        "segment_start_next_tick": raw_start,
        "segment_end_next_tick": raw_end,
        "span_ticks": span_count,
        "native_tick_iterations": native_iterations,
        "source_window_ticks": source_window_ticks,
        "source_window_count": source_window_count,
    }
    private_adapter = {
        "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
        "authority": "private_legacy_compatibility_only",
        "derived_from_absolute_next_tick": absolute_next_tick,
        "interval_ticks": interval_ticks,
        "source_generation": settlement_generation,
        "source_tick": private_span_start,
        "tick_span_start": private_span_start,
        "tick_span_end": private_span_end,
        "tick_span_count": settlement_tick_count,
        "terminal_generation_boundary": settlement_terminal_boundary,
        "private_settlement_direct_result_scalars": private_direct_scalars,
    }
    return public_direct_scalars, absolute_contract, private_adapter


def _validated_source_descriptor_recipe(wdl: Any, cell_ids: tuple[str, ...]) -> dict[str, Any]:
    factory = getattr(wdl, "source_projection_cursor_factory", None)
    exporter = getattr(factory, "detached_descriptor_recipe_v1", None)
    if not callable(exporter):
        raise _fail("physical owner lacks a tensor-free source descriptor exporter")
    recipe = exporter()
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("schema_version") != SOURCE_DESCRIPTOR_RECIPE_SCHEMA_VERSION
        or tuple(recipe.get("cell_ids") or ()) != cell_ids
        or not isinstance(recipe.get("descriptors"), list)
        or len(recipe["descriptors"]) != len(cell_ids)
    ):
        raise _fail("physical source descriptor recipe is incomplete")
    normalized = _encode_scalar_tree(recipe, path="source_descriptor_recipe")
    assert isinstance(normalized, dict)
    return normalized


def _tensor_storage_key(value: torch.Tensor) -> tuple[Any, ...]:
    storage = value.untyped_storage()
    return (
        str(value.device),
        int(storage.data_ptr()),
        int(value.storage_offset()),
        tuple(int(dim) for dim in value.shape),
        tuple(int(stride) for stride in value.stride()),
        str(value.dtype),
    )


def _tensor_storage_identity(value: torch.Tensor) -> tuple[Any, ...]:
    storage = value.untyped_storage()
    return (
        str(value.device),
        int(getattr(storage, "_cdata", storage.data_ptr())),
        int(storage.nbytes()),
    )


def _live_bank_alias_descriptor(
    value: torch.Tensor,
    *,
    base_name: str,
    base: torch.Tensor,
) -> dict[str, Any]:
    if (
        value.dtype != base.dtype
        or value.ndim < 1
        or base.ndim < 1
        or int(value.size(0)) != PHYSICAL_BATCH_CELLS_V1
        or int(base.size(0)) != PHYSICAL_BATCH_CELLS_V1
        or int(value.stride(0)) != int(base.stride(0))
    ):
        raise _fail("live-bank alias does not preserve the physical cell stride")
    relative_offset = int(value.storage_offset()) - int(base.storage_offset())
    cell_shape = tuple(int(dim) for dim in value.shape[1:])
    cell_stride = tuple(int(item) for item in value.stride()[1:])
    if relative_offset < 0:
        raise _fail("live-bank alias begins before its captured storage owner")
    maximum_offset = relative_offset
    for size, stride in zip(cell_shape, cell_stride, strict=True):
        if size > 0:
            maximum_offset += (size - 1) * stride
    if maximum_offset >= int(base.stride(0)):
        raise _fail("live-bank alias escapes its captured per-cell storage owner")
    return {
        "alias_of": base_name,
        "view": {
            "storage_offset": relative_offset,
            "shape": list(cell_shape),
            "stride": list(cell_stride),
        },
    }


def build_direct_resident_physical_capture_v1(
    prepared: Any,
    *,
    direct_result: Mapping[str, Any],
    raw_destination: str | Path,
    aggregate_destination: str | Path,
    provider_id: str,
    batch_id: str,
    segment_ordinal: int,
    generation: int,
    tick: int,
    absolute_next_tick: int,
    physical_owner_identity: str,
    reporting_enabled: bool,
    cpu_request_staging_enable: bool,
    fixed_mlp_contract: bool = True,
) -> dict[str, Any]:
    """Capture one exact physical B136 state and return a tensor-free envelope."""

    if not isinstance(direct_result, Mapping):
        raise _fail("direct_result must be a mapping")
    if (
        type(provider_id) is not str
        or not provider_id
        or type(reporting_enabled) is not bool
        or cpu_request_staging_enable is not False
        or fixed_mlp_contract is not True
    ):
        raise _fail("physical B136 capture requires the fixed-MLP production contract")
    composite, wdl, trajectory = _resolve_owner(prepared)
    source = getattr(trajectory, "source", None)
    live_banks = getattr(trajectory, "live_banks", None)
    outputs = getattr(source, "outputs", None)
    cell_ids = tuple(getattr(wdl, "cell_ids", ()) or ())
    if len(cell_ids) != PHYSICAL_BATCH_CELLS_V1 or len(set(cell_ids)) != len(cell_ids):
        raise _fail("physical checkpoint capture requires exact unique B136 cell order")
    if not isinstance(live_banks, Mapping) or not isinstance(outputs, Mapping):
        raise _fail("physical owner lacks live banks or settlement outputs")
    if live_banks.get("hapb_policy_owner") is not None:
        raise _fail("HAPB host-policy ownership is not admitted on this capture route")
    if direct_result.get("event_history_storage_mode") != "eiel_resident_event_sink_v1":
        raise _fail("physical capture requires the resident event-history sink")

    fields: dict[str, Any] = {}
    layout: dict[str, Any] = {
        "live_bank_tensors": {},
        "live_bank_aliases": {},
        "event_columns": {},
        "settlement_outputs": {},
        "wdl": {},
        "jungle": {},
        "eed": {},
    }
    storage_owner: dict[tuple[Any, ...], str] = {}

    def add_tensor(
        field_name: str,
        value: Any,
        *,
        target: dict[str, Any],
        target_name: str,
        allow_contiguous_copy: bool = False,
    ) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise _fail(f"physical capture field {field_name!r} is not a batched tensor")
        if int(value.size(0)) != PHYSICAL_BATCH_CELLS_V1:
            raise _fail(f"physical capture field {field_name!r} does not have B136 first")
        if not value.is_contiguous():
            if not allow_contiguous_copy:
                raise _fail(f"physical capture field {field_name!r} is not contiguous")
            value = value.contiguous()
        key = _tensor_storage_key(value)
        existing = storage_owner.get(key)
        if existing is not None:
            target[target_name] = {"alias_of": existing}
            return
        if field_name in fields:
            raise _fail(f"physical capture field name {field_name!r} is ambiguous")
        storage_owner[key] = field_name
        fields[field_name] = value
        target[target_name] = {"field": field_name}

    zero_fill_names = {"gru_pack", "hmem"}
    live_tensors = {
        str(name): value
        for name, value in live_banks.items()
        if isinstance(value, torch.Tensor)
    }
    canonical_by_storage: dict[tuple[Any, ...], str] = {}
    for name, value in live_tensors.items():
        if name in zero_fill_names or not value.is_contiguous():
            continue
        identity = _tensor_storage_identity(value)
        current_name = canonical_by_storage.get(identity)
        if current_name is None or value.numel() > live_tensors[current_name].numel():
            canonical_by_storage[identity] = name

    for name, value in live_tensors.items():
        if name in zero_fill_names:
            if value.ndim < 1 or int(value.size(0)) != PHYSICAL_BATCH_CELLS_V1:
                raise _fail(f"fixed-MLP zero field {name!r} is not B136")
            fields[name] = ZeroFillFieldV1(
                dtype=str(value.dtype),
                shape=tuple(int(dim) for dim in value.shape),
            )
            layout["live_bank_tensors"][name] = {"field": name}
            continue
        identity = _tensor_storage_identity(value)
        canonical_name = canonical_by_storage.get(identity)
        if canonical_name is None:
            raise _fail(
                f"physical capture field {name!r} is a strided view without "
                "a contiguous resident storage owner"
            )
        if canonical_name == name:
            add_tensor(
                name,
                value,
                target=layout["live_bank_tensors"],
                target_name=name,
            )
            continue
        descriptor = _live_bank_alias_descriptor(
            value,
            base_name=canonical_name,
            base=live_tensors[canonical_name],
        )
        layout["live_bank_tensors"][name] = descriptor
        layout["live_bank_aliases"][name] = descriptor

    event_columns = live_banks.get("event_columns")
    if not isinstance(event_columns, Sequence) or isinstance(
        event_columns, (str, bytes, bytearray)
    ) or len(event_columns) != len(EVENT_COLUMN_NAMES):
        raise _fail("physical owner lacks the exact resident event-column surface")
    event_physical_bytes = 0
    for name, value in zip(EVENT_COLUMN_NAMES, event_columns, strict=True):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 2
            or tuple(value.shape[:2])
            != (PHYSICAL_BATCH_CELLS_V1, EVENT_WINDOW_CAPACITY_V1)
        ):
            raise _fail(f"resident event column {name!r} is not B136x41804")
        event_physical_bytes += int(value.numel()) * int(value.element_size())
        add_tensor(
            name,
            value,
            target=layout["event_columns"],
            target_name=name,
        )
    if event_physical_bytes != EVENT_WINDOW_PHYSICAL_BYTES_V1:
        raise _fail(
            "resident event-column bytes differ from the exact physical B136 contract"
        )

    manifest = build_direct_resident_field_manifest_v1(
        fixed_mlp_contract=True
    )
    for row in manifest.fields:
        if row.provider_kind == EVENT_COLUMN_ROW:
            covered = row.name in layout["event_columns"]
        else:
            covered = row.name in layout["live_bank_tensors"]
        if not covered and row.name in EED_STATE_FIELDS_V1 and composite is not None:
            covered = getattr(composite, "eed_state_tensors", None) is not None
        if not covered:
            raise _fail(f"physical owner cannot resolve manifest field {row.name!r}")

    for name, axis in SETTLEMENT_OUTPUT_CELL_AXES_V1.items():
        value = direct_result.get(name, outputs.get(name))
        if value is None:
            continue
        if axis == 1:
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                raise _fail(f"settlement output {name!r} is invalid")
            value = value.movedim(1, 0)
        add_tensor(
            f"output/{name}",
            value,
            target=layout["settlement_outputs"],
            target_name=name,
            allow_contiguous_copy=True,
        )

    schedule = getattr(wdl, "schedule_tensors", ())
    reporting = getattr(wdl, "reporting_tensors", ())
    completed = (
        1
        if direct_result.get("absolute_tick_span_native_owner") is True
        else direct_result.get("completed_generations")
    )
    if (
        not isinstance(schedule, Sequence)
        or len(schedule) < 2
        or not isinstance(reporting, Sequence)
        or len(reporting) < 8
        or type(completed) is not int
        or completed < 1
    ):
        raise _fail("WDL physical owner lacks settlement schedule/reporting authority")
    add_tensor(
        "wdl/next_absolute_step",
        schedule[1],
        target=layout["wdl"],
        target_name="next_absolute_step",
    )
    for target_name, value in (
        ("agent_keys", reporting[6][:, completed - 1]),
        ("agent_flags", reporting[7][:, completed - 1]),
    ):
        add_tensor(
            f"wdl/{target_name}",
            value,
            target=layout["wdl"],
            target_name=target_name,
            allow_contiguous_copy=True,
        )

    if composite is not None:
        jungle_state = getattr(composite, "jungle_state_tensors", ())
        if not isinstance(jungle_state, Sequence) or len(jungle_state) != len(
            JUNGLE_STATE_FIELDS_V4
        ):
            raise _fail("Jungle physical owner lacks its exact state surface")
        for name, value in zip(JUNGLE_STATE_FIELDS_V4, jungle_state, strict=True):
            add_tensor(
                name,
                value,
                target=layout["jungle"],
                target_name=name,
            )
        eed_state = getattr(composite, "eed_state_tensors", None)
        if eed_state is not None:
            if len(eed_state) != len(EED_STATE_FIELDS_V1):
                raise _fail("EED physical owner lacks its exact state surface")
            for name, value in zip(EED_STATE_FIELDS_V1, eed_state, strict=True):
                add_tensor(
                    name,
                    value,
                    target=layout["eed"],
                    target_name=name,
                )

    live_scalars: dict[str, Any] = {}
    for name in (
        "event_history_drained",
        "hapb_full_cuda_mlp_pack_absent",
        "genome_v2_observability_config",
        "genome_v2_founder_profiles",
    ):
        if name in live_banks:
            live_scalars[name] = _encode_scalar_tree(
                live_banks[name], path=f"live_banks.{name}"
            )

    checkpoint_inputs = getattr(trajectory, "checkpoint_projection_inputs", None)
    if not isinstance(checkpoint_inputs, Mapping):
        raise _fail("physical owner lacks checkpoint projection inputs")
    encoded_checkpoint_inputs: dict[str, Any] = {}
    for name, value in checkpoint_inputs.items():
        if name in _CHECKPOINT_INPUT_OMISSIONS_V1:
            continue
        if name == "resident_birth_state":
            encoded_checkpoint_inputs[name] = _encode_birth_template(
                value, path="checkpoint_projection_inputs.resident_birth_state"
            )
        else:
            encoded_checkpoint_inputs[name] = _encode_scalar_tree(
                value, path=f"checkpoint_projection_inputs.{name}"
            )

    direct_scalars, absolute_tick_contract, private_settlement_adapter = (
        _direct_result_capture_contract_v1(
            direct_result,
            source=source,
            absolute_next_tick=absolute_next_tick,
        )
    )
    if tuple(direct_scalars.get("source_projection_cell_ids", cell_ids)) != cell_ids:
        raise _fail("direct result source projection order differs from B136")
    frontiers = direct_scalars.get("event_history_frontier_by_cell")
    prefixes = direct_scalars.get("event_history_prefix_references_by_cell")
    reference = direct_scalars.get("event_history_reference")
    if (
        not isinstance(reference, Mapping)
        or type(reference.get("path")) is not str
        or not isinstance(frontiers, list)
        or len(frontiers) != PHYSICAL_BATCH_CELLS_V1
        or not isinstance(prefixes, list)
        or len(prefixes) != PHYSICAL_BATCH_CELLS_V1
    ):
        raise _fail("physical capture lacks checkpoint-bound event prefixes")
    for cell, (frontier, prefix) in enumerate(zip(frontiers, prefixes, strict=True)):
        if (
            type(frontier) is not int
            or frontier < 0
            or not isinstance(prefix, Mapping)
            or prefix.get("path") != reference["path"]
            or prefix.get("cell_index") != cell
            or prefix.get("row_count") != frontier
            or prefix.get("row_bytes") != EVENT_ROW_BYTES_V1
            or type(prefix.get("prefix_sha256")) is not str
            or len(prefix["prefix_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in prefix["prefix_sha256"])
        ):
            raise _fail("physical capture event-prefix authority is malformed")
    source_descriptor_recipe = _validated_source_descriptor_recipe(wdl, cell_ids)
    source_configuration = _configuration_payload(source.configuration)
    settlement_direct_scalars = (
        private_settlement_adapter["private_settlement_direct_result_scalars"]
        if private_settlement_adapter is not None
        else direct_scalars
    )
    source_prefixes = settlement_direct_scalars.get(
        "tick_native_source_prefix_by_cell"
    )
    source_generation = (
        private_settlement_adapter["source_generation"]
        if private_settlement_adapter is not None
        else None
    )
    if (
        direct_scalars.get("absolute_tick_span_native_owner") is True
        and source_prefixes is None
    ):
        raise _fail("physical capture lacks tick-native source prefix authority")
    if source_prefixes is not None:
        if (
            not isinstance(source_prefixes, list)
            or len(source_prefixes) != PHYSICAL_BATCH_CELLS_V1
        ):
            raise _fail("physical capture tick-native source prefix is incomplete")
        prefix_generations: list[int] = []
        for row in source_prefixes:
            metadata = row.get("metadata") if isinstance(row, Mapping) else None
            generation_value = (
                metadata.get("generation") if isinstance(metadata, Mapping) else None
            )
            if type(generation_value) is not int or generation_value < 0:
                raise _fail(
                    "physical capture tick-native source generation is malformed"
                )
            prefix_generations.append(generation_value)
        if len(set(prefix_generations)) != 1:
            raise _fail("physical capture tick-native source generations disagree")
        source_generation = prefix_generations[0]
    if source_generation is None:
        source_generation = int(direct_result.get("tick_span_start", 0)) // int(
            getattr(source, "ticks_per_generation", 160)
        )
    route = {
        "execution_route_identity": EXECUTION_ROUTE_IDENTITY_V1,
        "physical_batch_cells": PHYSICAL_BATCH_CELLS_V1,
        "event_window_capacity": EVENT_WINDOW_CAPACITY_V1,
        "event_window_physical_bytes": EVENT_WINDOW_PHYSICAL_BYTES_V1,
        "event_row_bytes": EVENT_ROW_BYTES_V1,
        "resident_owner_span_ticks": getattr(source, "resident_owner_span_ticks", None),
        "native_dispatch_window_ticks": getattr(source, "native_dispatch_window_ticks", None),
        "source_window_ticks": getattr(source, "source_window_ticks", None),
        "cpu_request_staging_enable": False,
        "physical_owner_identity": physical_owner_identity,
    }
    if (
        route["resident_owner_span_ticks"] != 1120
        or route["native_dispatch_window_ticks"] != 1120
        or route["source_window_ticks"] != 160
        or getattr(source, "physical_window_generations", None) != 1
        or getattr(source, "resident_event_capacity", None)
        != EVENT_WINDOW_CAPACITY_V1
    ):
        raise _fail("physical capture owner geometry differs from the frozen route")
    private_source = {
        "generation_base": int(getattr(source, "generation_base", 0)),
        "logical_generation_count": int(
            getattr(source, "logical_generation_count", 0)
        ),
        "physical_window_generations": int(
            getattr(source, "physical_window_generations", 0)
        ),
        "ticks_per_generation": int(getattr(source, "ticks_per_generation", 0)),
    }
    source_contract = {"configuration": source_configuration, **private_source}
    if private_settlement_adapter is not None:
        private_settlement_adapter.update(
            {
                "provider_generation": generation,
                "provider_tick": tick,
                "generation_base_override": source_generation,
                "private_settlement_source": private_source,
            }
        )
        source_contract = {
            "configuration": source_configuration,
            "time_authority": "absolute_tick_v1",
            "absolute_next_tick": absolute_next_tick,
        }
    recipe = {
        "schema_version": SETTLEMENT_RECIPE_SCHEMA_VERSION,
        "provider_id": provider_id,
        "route": route,
        "batch_id": batch_id,
        "cell_ids": list(cell_ids),
        "segment_ordinal": segment_ordinal,
        "absolute_next_tick": absolute_next_tick,
        "reporting_enabled": reporting_enabled,
        "source": source_contract,
        "checkpoint_projection_inputs": encoded_checkpoint_inputs,
        "source_descriptor_recipe": source_descriptor_recipe,
        "direct_result_scalars": direct_scalars,
        "live_bank_scalars": live_scalars,
        "field_layout": layout,
        "field_manifest": manifest.to_dict(),
        "field_manifest_sha256": manifest.canonical_digest,
        "source_configuration_sha256": _sha256(source_configuration),
    }
    if absolute_tick_contract is not None:
        recipe["time_authority"] = "absolute_tick_v1"
        recipe["absolute_tick_contract"] = absolute_tick_contract
        recipe["legacy_compatibility"] = private_settlement_adapter
    else:
        recipe["generation"] = generation
        recipe["tick"] = tick
        recipe["generation_base_override"] = source_generation
    recipe["recipe_sha256"] = _sha256(recipe)
    _canonical_json(recipe)

    packed_descriptor = capture_physical_batch_packed_checkpoint_v1(
        fields,
        raw_destination,
        cell_ids=cell_ids,
        batch_id=batch_id,
        scalar_metadata={"settlement_recipe": recipe},
        require_native_pool=True,
    )
    slab_pool = packed_descriptor.get("slab_pool")
    if (
        not isinstance(slab_pool, Mapping)
        or slab_pool.get("native_pool") is not True
        or slab_pool.get("slab_count") != 2
        or slab_pool.get("pinned") is not True
        or slab_pool.get("preallocated") is not True
    ):
        Path(str(packed_descriptor.get("capture_path") or "")).unlink(
            missing_ok=True
        )
        raise _fail(
            "physical B136 capture requires one native preallocated pinned two-slab pool"
        )
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "source_consumer_required": False,
        "require_native_extension": True,
        "capture_path": packed_descriptor["capture_path"],
        "spool_path": packed_descriptor["capture_path"],
        "destination": str(Path(aggregate_destination).resolve()),
        "raw_publication_path": packed_descriptor["requested_publication_path"],
        "packed_descriptor": packed_descriptor,
        "packed_schema_version": PACKED_SCHEMA_VERSION,
        "batch_id": batch_id,
        "cell_ids": list(cell_ids),
        "segment_ordinal": segment_ordinal,
        "absolute_next_tick": absolute_next_tick,
        "logical_bytes": int(packed_descriptor["payload_bytes"])
        + int(packed_descriptor["zero_fill_logical_bytes"]),
        "zero_fill_logical_bytes": int(
            packed_descriptor["zero_fill_logical_bytes"]
        ),
        "native_slab_pool": dict(packed_descriptor["slab_pool"]),
        "settlement_recipe": recipe,
    }
    if absolute_tick_contract is not None:
        envelope["time_authority"] = "absolute_tick_v1"
    else:
        envelope["generation"] = generation
        envelope["tick"] = tick
    _canonical_json(envelope)
    return envelope


__all__ = [
    "DirectResidentPhysicalCaptureError",
    "EVENT_WINDOW_CAPACITY_V1",
    "EVENT_WINDOW_PHYSICAL_BYTES_V1",
    "EXECUTION_ROUTE_IDENTITY_V1",
    "PHYSICAL_BATCH_CELLS_V1",
    "SCHEMA_VERSION",
    "SETTLEMENT_RECIPE_SCHEMA_VERSION",
    "SOURCE_CONFIGURATION_RECIPE_SCHEMA_VERSION",
    "build_direct_resident_physical_capture_v1",
    "decode_birth_template_v1",
]
