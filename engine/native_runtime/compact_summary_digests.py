"""Strict digest producers for compact generation-summary authority planes."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ACTIVE_AGENT_IDS_SCHEMA_VERSION = "eiel_active_agent_ids_digest_v1"
POLICY_MUTATION_TAPE_SCHEMA_VERSION = (
    "eiel_policy_mutation_tape_consumed_prefix_v1"
)
BEHAVIOR_ROWS_SCHEMA_VERSION = "eiel_behavior_rows_digest_v1"
LINEAGE_RECORDS_SCHEMA_VERSION = "eiel_lineage_records_digest_v1"
RECOVERABLE_ERROR_SCHEMA_VERSION = "eiel_recoverable_error_digest_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

BEHAVIOR_ROW_FIELDS = (
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
)
LINEAGE_RECORD_FIELDS = (
    "agent_id",
    "energy",
    "health",
    "injury",
    "drives",
    "y",
    "x",
    "stats",
    "genome",
)
LINEAGE_DRIVE_FIELDS = ("hunger", "pain", "fatigue", "fear")
LINEAGE_STATS_FIELDS = (
    "lineage_id",
    "founder_id",
    "parent_ids",
    "parent_lineage_ids",
    "birth_generation",
    "birth_tick",
    "lineage_depth",
    "offspring_count",
    "survival_steps",
    "energy_collected",
    "first_repro_step",
    "pred_err_count",
    "pred_err_sum",
    "episode0_pred_err_mean",
    "identity_drift_ema",
    "drive_jitter",
    "ecological_role",
    "role_source",
    "role_confidence",
    "patch_habitat_lineage_id",
)
LINEAGE_GENOME_FIELDS = (
    "hidden_size",
    "memory_dim",
    "use_gru",
    "aggression_bias",
    "exploration_bias",
)

_BEHAVIOR_NONNEGATIVE_INT_FIELDS = frozenset(
    {
        "generation",
        "agent_id",
        "move_n",
        "move_s",
        "move_e",
        "move_w",
        "gather",
        "rest",
        "attack",
        "lifespan_steps",
        "offspring",
        "attacks_made",
    }
)
_BEHAVIOR_BINARY_FIELDS = frozenset(
    {"alive_end", "eligible_for_selection", "kept_next"}
)
_BEHAVIOR_FLOAT_FIELDS = frozenset(
    {
        "energy_collected",
        "fitness",
        "injury_end",
        "injury_delta",
        "aggression_bias",
    }
)
_POLICY_TAPE_DTYPES_AND_RANKS = {
    "policy_birth_crossover_mask_tape": (np.dtype(np.bool_), 2),
    "policy_birth_delta_tape": (np.dtype(np.float32), 2),
    "policy_birth_mutation_mask_tape": (np.dtype(np.bool_), 2),
    "policy_birth_post_rng_state_tape": (np.dtype(np.uint8), 2),
    "policy_birth_semantic_draws_tape": (np.dtype(np.int64), 1),
}
_POLICY_TAPE_CURSOR_FIELD = "policy_birth_tape_cursor"
_POLICY_TAPE_PACKED_POST_FIELD = "policy_birth_post_rng_state_packed_tape"
_TORCH_CPU_RNG_FULL_STATE_BYTES = 5_056
_TORCH_CPU_RNG_PACKED_STATE_BYTES = 2_560
_TORCH_CPU_RNG_HEAD_BYTES = 24
_TORCH_CPU_RNG_MIDDLE_LANES = 624
_TORCH_CPU_RNG_LANE_BYTES = 8
_TORCH_CPU_RNG_LOW_HALF_BYTES = 4
_TORCH_CPU_RNG_MIDDLE_FULL_BYTES = (
    _TORCH_CPU_RNG_MIDDLE_LANES * _TORCH_CPU_RNG_LANE_BYTES
)
_TORCH_CPU_RNG_MIDDLE_PACKED_BYTES = (
    _TORCH_CPU_RNG_MIDDLE_LANES * _TORCH_CPU_RNG_LOW_HALF_BYTES
)
_TORCH_CPU_RNG_TAIL_FULL_OFFSET = (
    _TORCH_CPU_RNG_HEAD_BYTES + _TORCH_CPU_RNG_MIDDLE_FULL_BYTES
)
_TORCH_CPU_RNG_TAIL_PACKED_OFFSET = (
    _TORCH_CPU_RNG_HEAD_BYTES + _TORCH_CPU_RNG_MIDDLE_PACKED_BYTES
)

__all__ = [
    "active_agent_ids_digest_sha256",
    "active_agent_ids_digest_from_tensor_sha256",
    "behavior_rows_digest_sha256",
    "checkpoint_layer_digest_from_tensor_manifest",
    "checkpoint_rng_digest_from_tensor_manifest",
    "checkpoint_source_projection_digest_from_tensor_manifest",
    "lineage_records_digest_sha256",
    "policy_mutation_tape_digest_from_tensor_sha256",
    "policy_mutation_tape_digest_sha256",
    "recoverable_error_digest_sha256",
]


def _require_little_endian_host() -> None:
    if sys.byteorder != "little":
        raise RuntimeError("compact summary digests require a little-endian host")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_exact_fields(
    value: Any,
    expected: Sequence[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    actual = set(value)
    required = set(expected)
    missing = sorted(required - actual)
    extra = sorted(actual - required, key=repr)
    if missing or extra:
        raise ValueError(f"{context} fields mismatch: missing={missing} extra={extra}")
    if any(not isinstance(name, str) for name in value):
        raise TypeError(f"{context} field names must be strings")
    return value


def _tensor_array(
    value: Any,
    *,
    field: str,
    dtype: np.dtype[Any],
    rank: int,
) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach") and hasattr(value, "to"):
        try:
            array = value.detach().to("cpu").contiguous().numpy()
        except Exception as exc:
            raise TypeError(f"{field} must be a dense tensor") from exc
    else:
        raise TypeError(f"{field} must be a NumPy array or tensor")
    if array.dtype != dtype:
        raise ValueError(f"{field} dtype must be {dtype}, got {array.dtype}")
    if array.ndim != rank:
        raise ValueError(f"{field} rank must be {rank}, got {array.ndim}")
    return np.ascontiguousarray(array)


def _tensor_manifest_entry(field: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "field": field,
        "dtype": str(array.dtype),
        "shape": [int(dim) for dim in array.shape],
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _unpack_torch_cpu_rng_state_rows_v1(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 2 or packed.shape[1] != _TORCH_CPU_RNG_PACKED_STATE_BYTES:
        raise ValueError(
            f"{_POLICY_TAPE_PACKED_POST_FIELD} shape must be "
            f"[rows,{_TORCH_CPU_RNG_PACKED_STATE_BYTES}]"
        )
    rows = packed.shape[0]
    middle = np.zeros(
        (rows, _TORCH_CPU_RNG_MIDDLE_LANES, _TORCH_CPU_RNG_LANE_BYTES),
        dtype=np.uint8,
    )
    middle[:, :, :_TORCH_CPU_RNG_LOW_HALF_BYTES] = packed[
        :, _TORCH_CPU_RNG_HEAD_BYTES:_TORCH_CPU_RNG_TAIL_PACKED_OFFSET
    ].reshape(rows, _TORCH_CPU_RNG_MIDDLE_LANES, _TORCH_CPU_RNG_LOW_HALF_BYTES)
    return np.ascontiguousarray(
        np.concatenate(
            (
                packed[:, :_TORCH_CPU_RNG_HEAD_BYTES],
                middle.reshape(rows, _TORCH_CPU_RNG_MIDDLE_FULL_BYTES),
                packed[:, _TORCH_CPU_RNG_TAIL_PACKED_OFFSET:],
            ),
            axis=1,
        )
    )


def _strict_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _tensor_manifest_entry_from_digest(
    field: str,
    *,
    dtype: np.dtype[Any],
    shape: Sequence[int],
    sha256: Any,
) -> dict[str, Any]:
    dimensions = [
        _strict_int(dimension, context=f"{field}.shape[{index}]")
        for index, dimension in enumerate(shape)
    ]
    return {
        "field": field,
        "dtype": str(dtype),
        "shape": dimensions,
        "sha256": _strict_sha256(sha256, context=f"{field}.sha256"),
    }


def _strict_int(value: Any, *, context: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an int")
    if value < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return value


def _strict_float(value: Any, *, context: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{context} must be a float")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return value


def _strict_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    if not value.strip():
        raise ValueError(f"{context} must be non-empty")
    return value


def _strict_json_value(value: Any, *, context: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{context} object keys must be strings")
        return {
            key: _strict_json_value(child, context=f"{context}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _strict_json_value(child, context=f"{context}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(f"{context} contains a non-JSON value {type(value).__name__}")


def recoverable_error_digest_sha256(code: Any, details: Any) -> str:
    """Digest the exact settled recoverable-error envelope without mutation."""

    code_value = _strict_nonempty_string(code, context="recoverable error code")
    if not isinstance(details, Mapping):
        raise TypeError("recoverable error details must be a mapping")
    details_value = _strict_json_value(details, context="recoverable error details")
    envelope = {
        "schema_version": RECOVERABLE_ERROR_SCHEMA_VERSION,
        "code": code_value,
        "details": details_value,
    }
    return _canonical_digest(envelope)


def active_agent_ids_digest_sha256(active_agent_ids: Any) -> str:
    """Digest the exact ascending active-ID authority tensor."""

    _require_little_endian_host()
    array = _tensor_array(
        active_agent_ids,
        field="agent_ids_by_alive_order",
        dtype=np.dtype(np.int64),
        rank=1,
    )
    if array.size and bool(np.any(array < 0)):
        raise ValueError("agent_ids_by_alive_order must contain nonnegative IDs")
    if array.size > 1 and not bool(np.all(array[1:] > array[:-1])):
        raise ValueError(
            "agent_ids_by_alive_order must be strictly ascending and unique"
        )
    envelope = {
        "schema_version": ACTIVE_AGENT_IDS_SCHEMA_VERSION,
        "fields": [_tensor_manifest_entry("agent_ids_by_alive_order", array)],
    }
    return _canonical_digest(envelope)


def active_agent_ids_digest_from_tensor_sha256(
    active_count: Any,
    tensor_sha256: Any,
) -> str:
    """Rebuild the active-ID envelope from an exact device tensor digest."""

    _require_little_endian_host()
    count = _strict_int(active_count, context="active_agent_count")
    envelope = {
        "schema_version": ACTIVE_AGENT_IDS_SCHEMA_VERSION,
        "fields": [
            _tensor_manifest_entry_from_digest(
                "agent_ids_by_alive_order",
                dtype=np.dtype(np.int64),
                shape=(count,),
                sha256=tensor_sha256,
            )
        ],
    }
    return _canonical_digest(envelope)


def _policy_tape_cursor(value: Any) -> int:
    if type(value) is int:
        return _strict_int(value, context=_POLICY_TAPE_CURSOR_FIELD)
    array = _tensor_array(
        value,
        field=_POLICY_TAPE_CURSOR_FIELD,
        dtype=np.dtype(np.int64),
        rank=1,
    )
    if array.shape != (1,):
        raise ValueError(f"{_POLICY_TAPE_CURSOR_FIELD} shape must be [1]")
    return _strict_int(int(array[0]), context=_POLICY_TAPE_CURSOR_FIELD)


def policy_mutation_tape_digest_sha256(tape: Any) -> str:
    """Digest only the consumed prefix of the exact five-field mutation tape."""

    _require_little_endian_host()
    if not isinstance(tape, Mapping):
        raise TypeError("policy mutation tape must be a mapping")
    expected_base = set(_POLICY_TAPE_DTYPES_AND_RANKS) - {
        "policy_birth_post_rng_state_tape"
    }
    required = expected_base | {_POLICY_TAPE_CURSOR_FIELD}
    has_dense_post = "policy_birth_post_rng_state_tape" in tape
    has_packed_post = _POLICY_TAPE_PACKED_POST_FIELD in tape
    actual = set(tape)
    missing = sorted(required - actual)
    extra = sorted(
        actual
        - required
        - {"policy_birth_post_rng_state_tape", _POLICY_TAPE_PACKED_POST_FIELD},
        key=repr,
    )
    if missing or extra:
        raise ValueError(
            f"policy mutation tape fields mismatch: missing={missing} extra={extra}"
        )
    if has_dense_post == has_packed_post:
        raise ValueError(
            "policy mutation tape must contain exactly one post-RNG state field"
        )
    mapping = tape
    cursor = _policy_tape_cursor(mapping[_POLICY_TAPE_CURSOR_FIELD])

    arrays: dict[str, np.ndarray] = {}
    capacity: int | None = None
    for field, (dtype, rank) in _POLICY_TAPE_DTYPES_AND_RANKS.items():
        if field == "policy_birth_post_rng_state_tape" and has_packed_post:
            continue
        array = _tensor_array(mapping[field], field=field, dtype=dtype, rank=rank)
        if capacity is None:
            capacity = int(array.shape[0])
        elif int(array.shape[0]) != capacity:
            raise ValueError("policy mutation tape field capacities disagree")
        arrays[field] = array
    if has_packed_post:
        packed_post = _tensor_array(
            mapping[_POLICY_TAPE_PACKED_POST_FIELD],
            field=_POLICY_TAPE_PACKED_POST_FIELD,
            dtype=np.dtype(np.uint8),
            rank=2,
        )
        if capacity is None:
            capacity = int(packed_post.shape[0])
        elif int(packed_post.shape[0]) != capacity:
            raise ValueError("policy mutation tape field capacities disagree")
        arrays[_POLICY_TAPE_PACKED_POST_FIELD] = packed_post
    assert capacity is not None
    if cursor > capacity:
        raise ValueError(
            f"{_POLICY_TAPE_CURSOR_FIELD} exceeds tape capacity: "
            f"cursor={cursor} capacity={capacity}"
        )
    delta_shape = arrays["policy_birth_delta_tape"].shape
    mask_shape = arrays["policy_birth_mutation_mask_tape"].shape
    crossover_shape = arrays["policy_birth_crossover_mask_tape"].shape
    if mask_shape != delta_shape:
        raise ValueError(
            "policy_birth_mutation_mask_tape shape must match "
            "policy_birth_delta_tape"
        )
    if crossover_shape[0] != delta_shape[0]:
        raise ValueError(
            "policy_birth_crossover_mask_tape capacity must match "
            "policy_birth_delta_tape"
        )
    if delta_shape[1] not in (1, crossover_shape[1]):
        raise ValueError(
            "policy_birth_delta_tape width must be compact width 1 or match "
            "policy_birth_crossover_mask_tape"
        )

    consumed_delta = arrays["policy_birth_delta_tape"][:cursor]
    if not bool(np.isfinite(consumed_delta).all()):
        raise ValueError("consumed policy_birth_delta_tape values must be finite")
    entries = []
    for field in sorted(arrays):
        consumed = np.ascontiguousarray(arrays[field][:cursor])
        if field == _POLICY_TAPE_PACKED_POST_FIELD:
            entries.append(
                _tensor_manifest_entry(
                    "policy_birth_post_rng_state_tape",
                    _unpack_torch_cpu_rng_state_rows_v1(consumed),
                )
            )
        else:
            entries.append(_tensor_manifest_entry(field, consumed))
    envelope = {
        "schema_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
        "cursor": cursor,
        "fields": entries,
    }
    return _canonical_digest(envelope)


def policy_mutation_tape_digest_from_tensor_sha256(
    cursor: Any,
    consumed_shapes: Mapping[str, Sequence[int]],
    tensor_sha256: Mapping[str, str],
) -> str:
    """Rebuild the consumed-tape envelope from exact device field digests."""

    _require_little_endian_host()
    cursor_value = _strict_int(cursor, context=_POLICY_TAPE_CURSOR_FIELD)
    expected = set(_POLICY_TAPE_DTYPES_AND_RANKS)
    if set(consumed_shapes) != expected or set(tensor_sha256) != expected:
        raise ValueError("policy mutation tape digest fields disagree with registry")

    entries: list[dict[str, Any]] = []
    normalized_shapes: dict[str, tuple[int, ...]] = {}
    for field in sorted(expected):
        dtype, rank = _POLICY_TAPE_DTYPES_AND_RANKS[field]
        shape = consumed_shapes[field]
        if isinstance(shape, (str, bytes, bytearray)) or not isinstance(
            shape, Sequence
        ):
            raise TypeError(f"{field}.shape must be a sequence")
        dimensions = tuple(
            _strict_int(value, context=f"{field}.shape[{index}]")
            for index, value in enumerate(shape)
        )
        if len(dimensions) != rank:
            raise ValueError(f"{field}.shape rank must be {rank}")
        if dimensions[0] != cursor_value:
            raise ValueError(f"{field}.shape must describe exactly the consumed prefix")
        normalized_shapes[field] = dimensions
        entries.append(
            _tensor_manifest_entry_from_digest(
                field,
                dtype=dtype,
                shape=dimensions,
                sha256=tensor_sha256[field],
            )
        )
    delta_shape = normalized_shapes["policy_birth_delta_tape"]
    mask_shape = normalized_shapes["policy_birth_mutation_mask_tape"]
    crossover_shape = normalized_shapes["policy_birth_crossover_mask_tape"]
    if mask_shape != delta_shape:
        raise ValueError(
            "policy_birth_mutation_mask_tape.shape must match "
            "policy_birth_delta_tape"
        )
    if crossover_shape[0] != delta_shape[0]:
        raise ValueError(
            "policy_birth_crossover_mask_tape.shape capacity must match "
            "policy_birth_delta_tape"
        )
    if delta_shape[1] not in (1, crossover_shape[1]):
        raise ValueError(
            "policy_birth_delta_tape.shape width must be compact width 1 or "
            "match policy_birth_crossover_mask_tape"
        )
    envelope = {
        "schema_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
        "cursor": cursor_value,
        "fields": entries,
    }
    return _canonical_digest(envelope)


def checkpoint_layer_digest_from_tensor_manifest(
    *,
    layer: Any,
    projection_schema_version: Any,
    world: Any,
    generation: Any,
    tick: Any,
    fields: Sequence[Mapping[str, Any]],
) -> str:
    """Rebuild one accepted layered checkpoint digest from device field hashes."""

    _require_little_endian_host()
    from .checkpoint_digests import (
        CANONICAL_STORAGE_EQUIVALENT_PROJECTION_SCHEMAS,
        SCHEMA_VERSION,
        DigestLayer,
    )

    resolved_layer = DigestLayer(layer)
    schema = _strict_nonempty_string(
        projection_schema_version,
        context="projection_schema_version",
    )
    if (
        resolved_layer is DigestLayer.CANONICAL_CHECKPOINT_STORAGE
        and schema in CANONICAL_STORAGE_EQUIVALENT_PROJECTION_SCHEMAS
    ):
        schema = SCHEMA_VERSION
    world_value = _strict_nonempty_string(world, context="checkpoint world")
    generation_value = _strict_int(generation, context="checkpoint generation")
    tick_value = _strict_int(tick, context="checkpoint tick")
    if isinstance(fields, (str, bytes, bytearray)) or not isinstance(
        fields, Sequence
    ):
        raise TypeError("checkpoint tensor manifest fields must be a sequence")

    entries: list[dict[str, Any]] = []
    previous_name: str | None = None
    for index, raw_entry in enumerate(fields):
        entry = _require_exact_fields(
            raw_entry,
            ("field", "dtype", "shape", "sha256"),
            context=f"checkpoint tensor manifest[{index}]",
        )
        name = _strict_nonempty_string(
            entry["field"], context=f"checkpoint tensor manifest[{index}].field"
        )
        if previous_name is not None and name <= previous_name:
            raise ValueError(
                "checkpoint tensor manifest fields must be strictly lexicographic"
            )
        previous_name = name
        dtype = _strict_nonempty_string(
            entry["dtype"], context=f"checkpoint tensor manifest[{index}].dtype"
        )
        shape = entry["shape"]
        if isinstance(shape, (str, bytes, bytearray)) or not isinstance(
            shape, Sequence
        ):
            raise TypeError(
                f"checkpoint tensor manifest[{index}].shape must be a sequence"
            )
        entries.append(
            {
                "field": name,
                "dtype": dtype,
                "shape": [
                    _strict_int(
                        dimension,
                        context=(
                            f"checkpoint tensor manifest[{index}].shape[{dimension_index}]"
                        ),
                    )
                    for dimension_index, dimension in enumerate(shape)
                ],
                "sha256": _strict_sha256(
                    entry["sha256"],
                    context=f"checkpoint tensor manifest[{index}].sha256",
                ),
            }
        )
    payload = {
        "layer": str(resolved_layer),
        "fields": entries,
        "metadata": {
            "schema_version": schema,
            "world": world_value,
            "generation": generation_value,
            "tick": tick_value,
        },
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def checkpoint_source_projection_digest_from_tensor_manifest(
    *,
    projection_fields: Mapping[str, Any],
    tensor_manifest: Mapping[str, Mapping[str, Any]],
) -> str:
    """Rebuild a source-projection digest without retaining tensor bytes."""

    _require_little_endian_host()
    from .checkpoint_digests import (
        SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
        SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
        SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
        SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS,
        _pack_source_projection_value,
        _packed_source_projection_digest,
        _source_projection_patch_profile,
    )

    if not isinstance(projection_fields, Mapping):
        raise TypeError("projection_fields must be a mapping")
    if any(type(name) is not str for name in projection_fields):
        raise TypeError("projection_fields names must be strings")
    if "tensors" in projection_fields:
        raise ValueError("projection_fields must omit tensors and use tensor_manifest")
    required = set(SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS) - {"tensors"}
    missing = sorted(required - set(projection_fields))
    if missing:
        raise ValueError(f"projection_fields is missing authority fields: {missing}")
    metadata = projection_fields.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("projection_fields.metadata must be a mapping")
    world = metadata.get("world")
    if type(world) is not str or world not in {"flat", "patch"}:
        raise ValueError("projection_fields.metadata.world must be 'flat' or 'patch'")
    if not isinstance(tensor_manifest, Mapping):
        raise TypeError("tensor_manifest must be a mapping")
    if any(type(name) is not str for name in tensor_manifest):
        raise TypeError("tensor_manifest names must be strings")

    packed_tensors: dict[str, dict[str, Any]] = {}
    for name in sorted(tensor_manifest):
        entry = _require_exact_fields(
            tensor_manifest[name],
            ("device", "dtype", "shape", "sha256"),
            context=f"source tensor manifest.{name}",
        )
        device = _strict_nonempty_string(
            entry["device"], context=f"source tensor manifest.{name}.device"
        )
        dtype = _strict_nonempty_string(
            entry["dtype"], context=f"source tensor manifest.{name}.dtype"
        )
        shape = entry["shape"]
        if isinstance(shape, (str, bytes, bytearray)) or not isinstance(
            shape, Sequence
        ):
            raise TypeError(f"source tensor manifest.{name}.shape must be a sequence")
        packed_tensors[name] = {
            "kind": "tensor",
            "device": device,
            "dtype": dtype,
            "shape": [
                _strict_int(
                    dimension,
                    context=f"source tensor manifest.{name}.shape[{index}]",
                )
                for index, dimension in enumerate(shape)
            ],
            "sha256": _strict_sha256(
                entry["sha256"],
                context=f"source tensor manifest.{name}.sha256",
            ),
        }

    canonical_projection_fields = projection_fields
    if world == "patch":
        profile = _source_projection_patch_profile(metadata)
        if profile != "P2":
            raise ValueError(
                "prehashed source-projection reconstruction supports Patch P2 only"
            )

        required_world_fields = (
            *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
            *SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
        )
        required_patch_fields = {
            "alive_mask",
            "agent_grid",
            "food_grid",
            *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
            *required_world_fields,
            *SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
        }
        missing_patch_fields = sorted(required_patch_fields - set(packed_tensors))
        if missing_patch_fields:
            raise ValueError(
                "source tensor manifest is missing required Patch P2 fields: "
                f"{missing_patch_fields}"
            )

        allowed_patch_fields = {
            *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
            *SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
            *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
            *SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
        }
        unsupported_patch_fields = sorted(
            name
            for name in packed_tensors
            if name.startswith("patch_") and name not in allowed_patch_fields
        )
        if unsupported_patch_fields:
            raise ValueError(
                "source tensor manifest contains unsupported Patch state: "
                f"{unsupported_patch_fields}"
            )

        def _patch_descriptor(
            name: str,
            *,
            dtype: str,
            rank: int,
            cpu_required: bool,
        ) -> dict[str, Any]:
            descriptor = packed_tensors[name]
            if cpu_required and descriptor["device"] != "cpu":
                raise ValueError(
                    f"source tensor manifest.{name}.device must be 'cpu' for Patch P2"
                )
            if descriptor["dtype"] != dtype:
                raise ValueError(
                    f"source tensor manifest.{name}.dtype must be {dtype!r}"
                )
            if len(descriptor["shape"]) != rank:
                raise ValueError(
                    f"source tensor manifest.{name}.shape must have rank {rank}"
                )
            return descriptor

        alive_mask = _patch_descriptor(
            "alive_mask", dtype="bool", rank=1, cpu_required=False
        )
        agent_grid = _patch_descriptor(
            "agent_grid", dtype="int32", rank=2, cpu_required=True
        )
        world_shape = agent_grid["shape"]
        for name in ("food_grid",):
            descriptor = _patch_descriptor(
                name, dtype="int32", rank=2, cpu_required=True
            )
            if descriptor["shape"] != world_shape:
                raise ValueError(
                    f"source tensor manifest.{name}.shape must match agent_grid"
                )
        for name in (*required_world_fields, *SOURCE_PATCH_DERIVED_TENSOR_FIELDS):
            descriptor = _patch_descriptor(
                name, dtype="float32", rank=2, cpu_required=True
            )
            if descriptor["shape"] != world_shape:
                raise ValueError(
                    f"source tensor manifest.{name}.shape must match agent_grid"
                )
        for name in SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS:
            descriptor = _patch_descriptor(
                name, dtype="int32", rank=1, cpu_required=True
            )
            if descriptor["shape"] != alive_mask["shape"]:
                raise ValueError(
                    f"source tensor manifest.{name}.shape must match alive_mask"
                )

        stored_manifest = projection_fields["tensor_manifest"]
        if not isinstance(stored_manifest, Mapping):
            raise TypeError(
                "projection_fields.tensor_manifest must be a mapping for Patch P2"
            )
        canonical_stored_manifest = dict(stored_manifest)
        for name in SOURCE_PATCH_DERIVED_TENSOR_FIELDS:
            canonical_stored_manifest.pop(name, None)
        canonical_projection_fields = dict(projection_fields)
        canonical_projection_fields["tensor_manifest"] = canonical_stored_manifest

    packed_tensor_mapping = {
        "kind": "mapping",
        "items": [
            (_pack_source_projection_value(name), packed_tensors[name])
            for name in sorted(packed_tensors)
        ],
    }
    top_level_fields = sorted((*canonical_projection_fields, "tensors"))
    packed_fields = {
        name: (
            packed_tensor_mapping
            if name == "tensors"
            else _pack_source_projection_value(canonical_projection_fields[name])
        )
        for name in top_level_fields
    }
    packed_projection = {
        "kind": "mapping",
        "items": [
            (_pack_source_projection_value(name), packed_fields[name])
            for name in top_level_fields
        ],
    }
    return _packed_source_projection_digest(packed_projection)


def checkpoint_rng_digest_from_tensor_manifest(
    *,
    metadata: Mapping[str, Any],
    tensor_manifest: Mapping[str, Mapping[str, Any]],
) -> str:
    """Rebuild the complete checkpoint RNG digest from retained authorities."""

    _require_little_endian_host()
    from .checkpoint_digests import (
        SOURCE_RNG_OPTIONAL_METADATA_FIELDS,
        SOURCE_RNG_REQUIRED_METADATA_FIELDS,
        SOURCE_RNG_TENSOR_FIELDS,
        _pack_source_projection_value,
        _packed_source_projection_digest,
    )

    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint RNG metadata must be a mapping")
    missing = sorted(set(SOURCE_RNG_REQUIRED_METADATA_FIELDS) - set(metadata))
    if missing:
        raise ValueError(f"checkpoint RNG metadata is incomplete: {missing}")
    if not isinstance(tensor_manifest, Mapping):
        raise TypeError("checkpoint RNG tensor_manifest must be a mapping")
    metadata_fields = [
        name
        for name in (
            *SOURCE_RNG_REQUIRED_METADATA_FIELDS,
            *SOURCE_RNG_OPTIONAL_METADATA_FIELDS,
        )
        if name in metadata
    ]
    if bool(metadata["native_runtime_env_policy_numpy_alias"]):
        env = _pack_source_projection_value(
            metadata["native_runtime_numpy_env_state"]
        )
        policy = _pack_source_projection_value(
            metadata["native_runtime_numpy_policy_state"]
        )
        if env != policy:
            raise ValueError("checkpoint RNG env/policy alias state contradiction")

    packed_tensors: dict[str, Any] = {}
    for name in SOURCE_RNG_TENSOR_FIELDS:
        if name not in tensor_manifest:
            continue
        entry = _require_exact_fields(
            tensor_manifest[name],
            ("device", "dtype", "shape", "sha256"),
            context=f"checkpoint RNG tensor manifest.{name}",
        )
        shape = entry["shape"]
        if isinstance(shape, (str, bytes, bytearray)) or not isinstance(
            shape, Sequence
        ):
            raise TypeError(
                f"checkpoint RNG tensor manifest.{name}.shape must be a sequence"
            )
        packed_tensors[name] = {
            "kind": "tensor",
            "device": _strict_nonempty_string(
                entry["device"],
                context=f"checkpoint RNG tensor manifest.{name}.device",
            ),
            "dtype": _strict_nonempty_string(
                entry["dtype"],
                context=f"checkpoint RNG tensor manifest.{name}.dtype",
            ),
            "shape": [
                _strict_int(
                    dimension,
                    context=f"checkpoint RNG tensor manifest.{name}.shape[{index}]",
                )
                for index, dimension in enumerate(shape)
            ],
            "sha256": _strict_sha256(
                entry["sha256"],
                context=f"checkpoint RNG tensor manifest.{name}.sha256",
            ),
        }
    packed = {
        "metadata": {
            name: _pack_source_projection_value(metadata[name])
            for name in metadata_fields
        },
        "tensors": packed_tensors,
    }
    return _packed_source_projection_digest(
        _pack_source_projection_value(packed)
    )


def behavior_rows_digest_sha256(rows: Any) -> str:
    """Digest strict behavior mappings as canonical 21-value row arrays."""

    _require_little_endian_host()
    if isinstance(rows, (Mapping, str, bytes, bytearray)) or not isinstance(
        rows, Sequence
    ):
        raise TypeError("behavior rows must be a sequence of mappings")

    encoded_rows: list[list[Any]] = []
    previous_key: tuple[int, int] | None = None
    for index, raw_row in enumerate(rows):
        context = f"behavior rows[{index}]"
        row = _require_exact_fields(raw_row, BEHAVIOR_ROW_FIELDS, context=context)
        generation = _strict_int(row["generation"], context=f"{context}.generation")
        agent_id = _strict_int(row["agent_id"], context=f"{context}.agent_id")
        key = (generation, agent_id)
        if previous_key is not None and key <= previous_key:
            raise ValueError(
                "behavior rows must be sorted by generation then agent_id with "
                "unique agent IDs per generation"
            )
        previous_key = key

        encoded: list[Any] = []
        for field in BEHAVIOR_ROW_FIELDS:
            value = row[field]
            field_context = f"{context}.{field}"
            if field in _BEHAVIOR_NONNEGATIVE_INT_FIELDS:
                value = _strict_int(value, context=field_context)
            elif field in _BEHAVIOR_BINARY_FIELDS:
                value = _strict_int(value, context=field_context)
                if value not in (0, 1):
                    raise ValueError(f"{field_context} must be 0 or 1")
            elif field in _BEHAVIOR_FLOAT_FIELDS:
                value = _strict_float(value, context=field_context)
            elif field == "cohort":
                value = _strict_nonempty_string(value, context=field_context)
            else:  # pragma: no cover - constants and validation must advance together.
                raise RuntimeError(f"unclassified behavior field {field!r}")
            encoded.append(value)
        encoded_rows.append(encoded)

    envelope = {
        "schema_version": BEHAVIOR_ROWS_SCHEMA_VERSION,
        "fields": list(BEHAVIOR_ROW_FIELDS),
        "rows": encoded_rows,
    }
    return _canonical_digest(envelope)


def _strict_int_list(value: Any, *, context: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return [
        _strict_int(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    ]


def _strict_string_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return [
        _strict_nonempty_string(item, context=f"{context}[{index}]")
        for index, item in enumerate(value)
    ]


def _validated_lineage_stats(value: Any, *, context: str) -> dict[str, Any]:
    stats = _require_exact_fields(value, LINEAGE_STATS_FIELDS, context=context)
    out: dict[str, Any] = {}
    out["lineage_id"] = _strict_nonempty_string(
        stats["lineage_id"], context=f"{context}.lineage_id"
    )
    out["founder_id"] = _strict_int(
        stats["founder_id"], context=f"{context}.founder_id"
    )
    out["parent_ids"] = _strict_int_list(
        stats["parent_ids"], context=f"{context}.parent_ids"
    )
    out["parent_lineage_ids"] = _strict_string_list(
        stats["parent_lineage_ids"], context=f"{context}.parent_lineage_ids"
    )
    for field in (
        "birth_generation",
        "birth_tick",
        "lineage_depth",
        "offspring_count",
    ):
        out[field] = _strict_int(stats[field], context=f"{context}.{field}")
    for field in ("survival_steps", "energy_collected"):
        out[field] = _strict_float(stats[field], context=f"{context}.{field}")
    first_repro_step = stats["first_repro_step"]
    out["first_repro_step"] = (
        None
        if first_repro_step is None
        else _strict_int(first_repro_step, context=f"{context}.first_repro_step")
    )
    out["pred_err_count"] = _strict_int(
        stats["pred_err_count"], context=f"{context}.pred_err_count"
    )
    for field in (
        "pred_err_sum",
        "episode0_pred_err_mean",
        "identity_drift_ema",
        "drive_jitter",
    ):
        out[field] = _strict_float(stats[field], context=f"{context}.{field}")
    out["ecological_role"] = _strict_nonempty_string(
        stats["ecological_role"], context=f"{context}.ecological_role"
    )
    out["role_source"] = _strict_nonempty_string(
        stats["role_source"], context=f"{context}.role_source"
    )
    role_confidence = stats["role_confidence"]
    out["role_confidence"] = (
        None
        if role_confidence is None
        else _strict_float(role_confidence, context=f"{context}.role_confidence")
    )
    out["patch_habitat_lineage_id"] = _strict_int(
        stats["patch_habitat_lineage_id"],
        context=f"{context}.patch_habitat_lineage_id",
    )
    return out


def _validated_lineage_genome(value: Any, *, context: str) -> dict[str, Any]:
    genome = _require_exact_fields(value, LINEAGE_GENOME_FIELDS, context=context)
    use_gru = genome["use_gru"]
    if type(use_gru) is not bool:
        raise TypeError(f"{context}.use_gru must be a bool")
    return {
        "hidden_size": _strict_int(
            genome["hidden_size"], context=f"{context}.hidden_size"
        ),
        "memory_dim": _strict_int(
            genome["memory_dim"], context=f"{context}.memory_dim"
        ),
        "use_gru": use_gru,
        "aggression_bias": _strict_float(
            genome["aggression_bias"], context=f"{context}.aggression_bias"
        ),
        "exploration_bias": _strict_float(
            genome["exploration_bias"], context=f"{context}.exploration_bias"
        ),
    }


def lineage_records_digest_sha256(generation: Any, records: Any) -> str:
    """Digest strict lineage records for one generation in ascending ID order."""

    _require_little_endian_host()
    generation_value = _strict_int(generation, context="lineage generation")
    if isinstance(records, (Mapping, str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TypeError("lineage records must be a sequence of mappings")

    encoded_records: list[dict[str, Any]] = []
    previous_agent_id: int | None = None
    for index, raw_record in enumerate(records):
        context = f"lineage records[{index}]"
        record = _require_exact_fields(raw_record, LINEAGE_RECORD_FIELDS, context=context)
        agent_id = _strict_int(record["agent_id"], context=f"{context}.agent_id")
        if previous_agent_id is not None and agent_id <= previous_agent_id:
            raise ValueError(
                "lineage records must have strictly ascending unique agent_id values"
            )
        previous_agent_id = agent_id

        drives = record["drives"]
        if not isinstance(drives, list) or len(drives) != len(LINEAGE_DRIVE_FIELDS):
            raise TypeError(
                "lineage record drives must be a four-value list ordered as "
                + ", ".join(LINEAGE_DRIVE_FIELDS)
            )
        encoded_records.append(
            {
                "agent_id": agent_id,
                "energy": _strict_float(record["energy"], context=f"{context}.energy"),
                "health": _strict_float(record["health"], context=f"{context}.health"),
                "injury": _strict_float(record["injury"], context=f"{context}.injury"),
                "drives": [
                    _strict_float(value, context=f"{context}.drives.{field}")
                    for field, value in zip(LINEAGE_DRIVE_FIELDS, drives, strict=True)
                ],
                "y": _strict_int(record["y"], context=f"{context}.y"),
                "x": _strict_int(record["x"], context=f"{context}.x"),
                "stats": _validated_lineage_stats(
                    record["stats"], context=f"{context}.stats"
                ),
                "genome": _validated_lineage_genome(
                    record["genome"], context=f"{context}.genome"
                ),
            }
        )

    envelope = {
        "schema_version": LINEAGE_RECORDS_SCHEMA_VERSION,
        "generation": generation_value,
        "records": encoded_records,
    }
    return _canonical_digest(envelope)
