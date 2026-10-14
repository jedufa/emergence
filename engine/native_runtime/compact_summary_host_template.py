"""Pre-dispatch byte templates for compact-summary host digest settlement.

The compiler consumes explicit token placeholders, never settled CUDA values.  Each
compiled program is a JSON-serializable sequence of immutable UTF-8 literals and
bounded dynamic-token references.  The reference renderer is intentionally simple;
the same program can be executed by a C++ callback into a preallocated byte buffer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

from .checkpoint_digests import (
    CANONICAL_STORAGE_EQUIVALENT_PROJECTION_SCHEMAS,
    SCHEMA_VERSION as CHECKPOINT_LAYER_SCHEMA_VERSION,
    SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS,
    SOURCE_RNG_OPTIONAL_METADATA_FIELDS,
    SOURCE_RNG_REQUIRED_METADATA_FIELDS,
    SOURCE_RNG_SCHEMA_VERSION,
    SOURCE_RNG_TENSOR_FIELDS,
    DigestLayer,
    _source_projection_patch_profile,
)
from .compact_summary_checkpoint_descriptor_plan import (
    PATCH_P2_PROFILE,
    PLAN_SCHEMA_VERSION,
    WORLD_FLAT,
    WORLD_PATCH,
    CompactSummaryCheckpointDescriptorPlanError,
    validate_compact_summary_checkpoint_descriptor_plan,
)
from .compact_summary_digests import (
    ACTIVE_AGENT_IDS_SCHEMA_VERSION,
    POLICY_MUTATION_TAPE_SCHEMA_VERSION,
)


HOST_TEMPLATE_SCHEMA_VERSION: Final[str] = "eiel_compact_summary_host_template_v1"
HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_host_template_bundle_v1"
)
TOKEN_PLACEHOLDER_KEY: Final[str] = "$eiel_compact_summary_host_token"

TOKEN_INTEGER: Final[str] = "integer"
TOKEN_BOOLEAN: Final[str] = "boolean"
TOKEN_PCG_UINT128: Final[str] = "pcg_uint128"
TOKEN_PYTHON_MT_WORD: Final[str] = "python_mt_word"
TOKEN_TENSOR_SHAPE: Final[str] = "tensor_shape"
TOKEN_SHA256_HEX: Final[str] = "sha256_hex"
TOKEN_NULLABLE_FLOAT64: Final[str] = "nullable_float64"
TOKEN_TYPES: Final[tuple[str, ...]] = (
    TOKEN_INTEGER,
    TOKEN_BOOLEAN,
    TOKEN_PCG_UINT128,
    TOKEN_PYTHON_MT_WORD,
    TOKEN_TENSOR_SHAPE,
    TOKEN_SHA256_HEX,
    TOKEN_NULLABLE_FLOAT64,
)

SURFACE_CHECKPOINT_TRAJECTORY: Final[str] = "checkpoint_trajectory"
SURFACE_CANONICAL_STORAGE: Final[str] = "canonical_checkpoint_storage"
SURFACE_NEXT_GENERATION_SOURCE: Final[str] = "next_generation_source_projection"
SURFACE_RNG: Final[str] = "rng"
SURFACE_ACTIVE_AGENT_IDS: Final[str] = "active_agent_ids"
SURFACE_POLICY_MUTATION_TAPE: Final[str] = "policy_mutation_tape"
SURFACE_ORDER: Final[tuple[str, ...]] = (
    SURFACE_CHECKPOINT_TRAJECTORY,
    SURFACE_CANONICAL_STORAGE,
    SURFACE_NEXT_GENERATION_SOURCE,
    SURFACE_RNG,
    SURFACE_ACTIVE_AGENT_IDS,
    SURFACE_POLICY_MUTATION_TAPE,
)
WIRE_FIELDS: Final[dict[str, str]] = {
    SURFACE_CHECKPOINT_TRAJECTORY: (
        "state_digests.authoritative_state_digest_sha256"
    ),
    SURFACE_CANONICAL_STORAGE: (
        "state_digests.canonical_checkpoint_storage_digest_sha256"
    ),
    SURFACE_NEXT_GENERATION_SOURCE: (
        "state_digests.next_generation_start_state_digest_sha256"
    ),
    SURFACE_RNG: "rng.authority_digest_sha256",
    SURFACE_ACTIVE_AGENT_IDS: "identity.active_agent_ids_digest_sha256",
    SURFACE_POLICY_MUTATION_TAPE: "rng.policy_mutation_tape_digest_sha256",
}

UINT32_MAX: Final[int] = (1 << 32) - 1
UINT64_MAX: Final[int] = (1 << 64) - 1
UINT128_MAX: Final[int] = (1 << 128) - 1
MAX_COORDINATES: Final[int] = 4096
MAX_TOKENS_PER_TEMPLATE: Final[int] = 8192
MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE: Final[int] = 8 * 1024 * 1024
MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE: Final[int] = 16 * 1024 * 1024
MAX_SOURCE_PATH_UTF8_BYTES: Final[int] = 2048

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_FIELDS: Final[frozenset[str]] = frozenset(
    {"token_type", "source_path", "rank", "render_pack_depth"}
)
_PROJECTION_DYNAMIC_TOP_LEVEL: Final[tuple[str, ...]] = (
    "restore_count",
    "generation_calls",
    "event_cursor",
    "runtime_event_cursor",
    "last_applied_actions",
    "last_moved",
    "last_gathered",
    "last_policy_refill_calls",
    "last_policy_refill_rows",
)
_PROJECTION_DYNAMIC_METADATA_INTEGERS: Final[tuple[str, ...]] = (
    "configured_steps_per_generation",
    "alive_count",
    "event_tape_active_count",
    "event_tape_cursor",
    "event_tape_inactive_tail_count",
    "native_runtime_rng_state",
    "policy_energy_observation_finite_live_count",
    "policy_energy_observation_live_count",
    "population_next_id",
    "typed_lifecycle_event_active_count",
    "typed_lifecycle_event_next_sequence",
)
_PROJECTION_DYNAMIC_METADATA_BOOLEANS: Final[tuple[str, ...]] = (
    "native_policy_cache_served",
)
_PRE_DISPATCH_COORDINATE_PATHS: Final[frozenset[str]] = frozenset(
    {"/checkpoint/projection_fields/metadata/generation"}
)
_STATIC_PRE_DISPATCH_SOURCE_TENSORS: Final[tuple[str, ...]] = (
    "drive_noise_samples_by_tick_slot",
    "drive_noise_samples_valid_by_tick_slot",
)


class CompactSummaryHostTemplateError(ValueError):
    """The template input or a settled token value violates the frozen contract."""


class CompactSummaryHostTemplateBlocker(CompactSummaryHostTemplateError):
    """An exact canonical surface needs a value class unavailable pre-dispatch."""


def _fail(path: str, message: str) -> None:
    raise CompactSummaryHostTemplateError(f"{path}: {message}")


def _block(path: str, message: str) -> None:
    raise CompactSummaryHostTemplateBlocker(f"{path}: {message}")


def _pointer(base: str, component: str | int) -> str:
    escaped = str(component).replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}" if base else f"/{escaped}"


def _rule(token_type: str, *, rank: int | None = None) -> dict[str, Any]:
    if token_type == TOKEN_INTEGER:
        return {"kind": "integer_range", "minimum": 0, "maximum": UINT64_MAX}
    if token_type == TOKEN_BOOLEAN:
        return {"kind": "exact_boolean"}
    if token_type == TOKEN_PCG_UINT128:
        return {"kind": "integer_range", "minimum": 0, "maximum": UINT128_MAX}
    if token_type == TOKEN_PYTHON_MT_WORD:
        return {"kind": "integer_range", "minimum": 0, "maximum": UINT32_MAX}
    if token_type == TOKEN_TENSOR_SHAPE:
        if type(rank) is not int or rank < 0:
            raise ValueError("tensor_shape placeholders require a nonnegative exact rank")
        return {
            "kind": "tensor_shape",
            "rank": rank,
            "minimum_dimension": 0,
            "maximum_dimension": UINT64_MAX,
        }
    if token_type == TOKEN_SHA256_HEX:
        return {"kind": "lowercase_sha256_hex", "length": 64}
    if token_type == TOKEN_NULLABLE_FLOAT64:
        return {"kind": "finite_optional_float64"}
    raise ValueError(f"unsupported compact-summary host token type {token_type!r}")


def dynamic_token_placeholder(
    token_type: str,
    source_path: str,
    *,
    rank: int | None = None,
) -> dict[str, Any]:
    """Return one JSON-serializable pre-dispatch retained-authority placeholder."""

    _rule(token_type, rank=rank)
    if type(source_path) is not str or not source_path.startswith("/"):
        raise ValueError("source_path must be an absolute JSON Pointer")
    marker: dict[str, Any] = {
        "token_type": token_type,
        "source_path": source_path,
    }
    if token_type == TOKEN_TENSOR_SHAPE:
        marker["rank"] = rank
    return {TOKEN_PLACEHOLDER_KEY: marker}


def _placeholder_spec(
    value: Any,
    *,
    path: str | None = None,
    token_type: str | None = None,
    rank: int | None = None,
) -> dict[str, Any] | None:
    if not (
        isinstance(value, Mapping)
        and set(value) == {TOKEN_PLACEHOLDER_KEY}
        and isinstance(value[TOKEN_PLACEHOLDER_KEY], Mapping)
    ):
        return None
    marker = value[TOKEN_PLACEHOLDER_KEY]
    actual_fields = set(marker)
    required_fields = {"token_type", "source_path"}
    allowed_fields = set(_PLACEHOLDER_FIELDS)
    if not required_fields <= actual_fields or not actual_fields <= allowed_fields:
        _fail(str(path or "placeholder"), "placeholder fields are not exact")
    actual_type = marker["token_type"]
    source_path = marker["source_path"]
    if actual_type not in TOKEN_TYPES:
        _fail(str(path or "placeholder"), f"unsupported token type {actual_type!r}")
    if type(source_path) is not str or not source_path.startswith("/"):
        _fail(str(path or "placeholder"), "source_path must be an absolute JSON Pointer")
    if len(source_path.encode("utf-8")) > MAX_SOURCE_PATH_UTF8_BYTES:
        _fail(str(path or "placeholder"), "source_path exceeds the bounded contract")
    if path is not None and source_path != path:
        _fail(path, f"placeholder source_path must equal {path!r}")
    if token_type is not None and actual_type != token_type:
        _fail(str(path or source_path), f"token type must equal {token_type!r}")
    marker_rank = marker.get("rank")
    pack_depth = marker.get("render_pack_depth", 0)
    if type(pack_depth) is not int or not 0 <= pack_depth <= 2:
        _fail(str(path or source_path), "render_pack_depth must be 0, 1, or 2")
    if actual_type == TOKEN_TENSOR_SHAPE:
        if type(marker_rank) is not int or marker_rank < 0:
            _fail(str(path or source_path), "tensor_shape rank must be a nonnegative int")
        if rank is not None and marker_rank != rank:
            _fail(str(path or source_path), f"tensor_shape rank must equal {rank}")
    elif "rank" in marker:
        _fail(str(path or source_path), "rank is valid only for tensor_shape tokens")
    validation_rule = _rule(actual_type, rank=marker_rank)
    validation_rule["render_pack_depth"] = pack_depth
    return {
        "token_type": actual_type,
        "source_path": source_path,
        "validation_rule": validation_rule,
    }


def _expect_placeholder(
    value: Any,
    *,
    path: str,
    token_type: str,
    rank: int | None = None,
) -> None:
    spec = _placeholder_spec(value, path=path, token_type=token_type, rank=rank)
    if spec is None:
        _fail(path, f"must be an explicit {token_type!r} pre-dispatch placeholder")
    if spec["validation_rule"]["render_pack_depth"] != 0:
        _fail(path, "input placeholders must not specify an internal render pack depth")


def _strict_count(value: Any, *, path: str, minimum: int = 1) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an int")
    if value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _exact_mapping(value: Any, fields: Sequence[str], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{path} keys must be strings")
    missing = sorted(set(fields) - set(value))
    extra = sorted(set(value) - set(fields))
    if missing or extra:
        _fail(path, f"fields mismatch: missing={missing} extra={extra}")
    return value


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{path} keys must be strings")
    return value


def _descriptor_rows(
    plan: Mapping[str, Any],
    *,
    world: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        _fail("descriptor_plan.schema_version", f"must equal {PLAN_SCHEMA_VERSION!r}")
    try:
        validate_compact_summary_checkpoint_descriptor_plan(plan, world=world)
    except CompactSummaryCheckpointDescriptorPlanError as exc:
        _fail("descriptor_plan", str(exc))
    raw_rows = plan.get("descriptors")
    if not isinstance(raw_rows, list):
        raise TypeError("descriptor_plan.descriptors must be a list")
    rows: list[Mapping[str, Any]] = []
    previous: str | None = None
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw, path=f"descriptor_plan.descriptors[{index}]")
        name = row.get("tensor_name")
        if type(name) is not str or not name:
            _fail(f"descriptor_plan.descriptors[{index}].tensor_name", "must be non-empty")
        if previous is not None and name <= previous:
            _fail("descriptor_plan.descriptors", "must be strictly lexicographic")
        if row.get("descriptor_index") != index:
            _fail(f"descriptor_plan.descriptors[{index}]", "descriptor_index drift")
        previous = name
        rows.append(row)
    checkpoint = [row for row in rows if row.get("source_scope") == "checkpoint_projection"]
    policy = [row for row in rows if row.get("source_scope") == "policy_mutation_tape"]
    if len(checkpoint) + len(policy) != len(rows):
        _fail("descriptor_plan.descriptors", "contains an unsupported source_scope")
    return checkpoint, policy


def _projection_world(
    projection: Mapping[str, Any],
    *,
    flat_only: bool = False,
) -> str:
    metadata = _mapping(
        projection.get("metadata"),
        path="checkpoint_projection_skeleton.metadata",
    )
    world = metadata.get("world")
    last_world = projection.get("last_world")
    if flat_only and (world != WORLD_FLAT or last_world != WORLD_FLAT):
        _fail(
            "checkpoint_projection_skeleton",
            "only accepted Flat projections are supported",
        )
    if type(world) is not str or world not in {WORLD_FLAT, WORLD_PATCH}:
        _fail(
            "checkpoint_projection_skeleton.metadata.world",
            "must be exactly 'flat' or 'patch'",
        )
    if type(last_world) is not str or last_world != world:
        _fail(
            "checkpoint_projection_skeleton.last_world",
            "must exactly match metadata.world",
        )
    if world == WORLD_PATCH:
        try:
            profile = _source_projection_patch_profile(metadata)
        except (TypeError, ValueError) as exc:
            _fail("checkpoint_projection_skeleton.metadata", str(exc))
        if profile != PATCH_P2_PROFILE:
            _fail(
                "checkpoint_projection_skeleton.metadata.native_runtime_patch_profile",
                "compact-summary host compilation requires exact Patch P2",
            )
    return world


def _load_projection_skeleton(
    skeleton: Mapping[str, Any] | str | Path,
    plan: Mapping[str, Any],
    *,
    flat_only: bool = False,
) -> Mapping[str, Any]:
    raw: bytes | None = None
    if isinstance(skeleton, (str, Path)):
        path = Path(skeleton)
        raw = path.read_bytes()
        loaded = json.loads(raw)
    elif isinstance(skeleton, Mapping):
        loaded = skeleton
    else:
        raise TypeError("checkpoint_projection_skeleton must be a mapping or path")
    projection = loaded.get("projection") if isinstance(loaded, Mapping) else None
    if isinstance(projection, Mapping):
        resolved = projection
    else:
        resolved = _mapping(loaded, path="checkpoint_projection_skeleton")
    world = _projection_world(resolved, flat_only=flat_only)
    if raw is not None and world == WORLD_FLAT:
        expected = plan["source_contract"]["accepted_flat_projection_fixture"]
        if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            _fail("checkpoint_projection_skeleton", "accepted fixture SHA-256 drift")
    return resolved


def _validate_projection_skeleton(
    projection: Mapping[str, Any],
    plan: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    *,
    world: str,
) -> None:
    missing = sorted(set(SOURCE_PROJECTION_REQUIRED_TOP_LEVEL_FIELDS) - set(projection))
    if missing:
        _fail("checkpoint_projection_skeleton", f"missing fields {missing}")
    if _projection_world(projection) != world:
        _fail("checkpoint_projection_skeleton", "world contradicts descriptor plan")
    manifest = _mapping(
        projection["tensor_manifest"],
        path="checkpoint_projection_skeleton.tensor_manifest",
    )
    accepted = plan["source_contract"]["accepted_flat_projection_fixture"]
    additions = set(accepted["current_registry_additions"])
    descriptor_names = {row["tensor_name"] for row in checkpoint_rows}
    expected_names = descriptor_names - additions
    actual_names = set(manifest)
    accepted_name_sets = {frozenset(expected_names)}
    if world == WORLD_PATCH:
        accepted_name_sets.add(frozenset(descriptor_names))
    static_names = set(_STATIC_PRE_DISPATCH_SOURCE_TENSORS)
    allowed_name_sets = {
        *accepted_name_sets,
        *(frozenset(names | static_names) for names in accepted_name_sets),
    }
    if frozenset(actual_names) not in allowed_name_sets:
        _fail("checkpoint_projection_skeleton.tensor_manifest", "field registry drift")
    if world == WORLD_PATCH:
        tensors = _mapping(
            projection.get("tensors"),
            path="checkpoint_projection_skeleton.tensors",
        )
        if set(tensors) != actual_names:
            _fail(
                "checkpoint_projection_skeleton.tensors",
                "Patch tensor fields must exactly match tensor_manifest",
            )
    for field in ("pending_generation_free_agent_ids", "pending_generation_free_slots"):
        value = projection[field]
        if not isinstance(value, list):
            raise TypeError(f"checkpoint_projection_skeleton.{field} must be a list")
        if value:
            _block(
                f"checkpoint_projection_skeleton.{field}",
                "variable-cardinality pending-free authority cannot be fixed before dispatch",
            )


def _dynamic_projection_specs(projection: Mapping[str, Any]) -> dict[str, tuple[str, int | None]]:
    specs: dict[str, tuple[str, int | None]] = {}
    base = "/checkpoint/projection_fields"
    for name in _PROJECTION_DYNAMIC_TOP_LEVEL:
        if name in projection:
            specs[_pointer(base, name)] = (TOKEN_INTEGER, None)
    metadata = _mapping(projection["metadata"], path="checkpoint_projection_skeleton.metadata")
    metadata_base = _pointer(base, "metadata")
    generation = metadata.get("generation")
    if type(generation) is not int or generation < 0:
        _fail("checkpoint_projection_skeleton.metadata.generation", "must be a nonnegative int")
    specs[_pointer(metadata_base, "generation")] = (TOKEN_INTEGER, None)
    for name in _PROJECTION_DYNAMIC_METADATA_INTEGERS:
        if name in metadata:
            specs[_pointer(metadata_base, name)] = (TOKEN_INTEGER, None)
    for name in _PROJECTION_DYNAMIC_METADATA_BOOLEANS:
        if name in metadata:
            specs[_pointer(metadata_base, name)] = (TOKEN_BOOLEAN, None)

    for name, value in metadata.items():
        path = _pointer(metadata_base, name)
        if (
            type(value) is str
            and _SHA256_RE.fullmatch(value) is not None
            and (name.endswith("sha256") or name.endswith("digest"))
        ):
            specs[path] = (TOKEN_SHA256_HEX, None)
        if name.endswith("_shape") and isinstance(value, list):
            specs[path] = (TOKEN_TENSOR_SHAPE, len(value))

    for name, value in metadata.items():
        if not (
            name.startswith("native_runtime_numpy_")
            and name.endswith("_state")
            and isinstance(value, Mapping)
            and value.get("bit_generator") == "PCG64"
        ):
            continue
        path = _pointer(metadata_base, name)
        state = _mapping(value.get("state"), path=f"checkpoint_projection_skeleton.metadata.{name}.state")
        if set(state) != {"state", "inc"}:
            _block(path, "PCG64 state structure is not the frozen two-word authority")
        specs[_pointer(_pointer(path, "state"), "state")] = (TOKEN_PCG_UINT128, None)
        specs[_pointer(_pointer(path, "state"), "inc")] = (TOKEN_PCG_UINT128, None)
        for scalar in ("has_uint32", "uinteger"):
            if scalar in value:
                specs[_pointer(path, scalar)] = (TOKEN_INTEGER, None)

    python_name = "native_runtime_python_random_state"
    if python_name in metadata:
        value = metadata[python_name]
        path = _pointer(metadata_base, python_name)
        if isinstance(value, Mapping):
            gauss_path = _pointer(path, "gauss_next")
            if "gauss_next" not in value:
                _block(gauss_path, "Python Gaussian cache field is required")
            gauss = value["gauss_next"]
            if gauss is not None and (
                type(gauss) is not float or not math.isfinite(gauss)
            ):
                _block(gauss_path, "Python Gaussian cache must be null or a finite float")
            specs[gauss_path] = (TOKEN_NULLABLE_FLOAT64, None)
            state = value.get("state")
            if not isinstance(state, list):
                _block(_pointer(path, "state"), "Python MT state must have fixed list structure")
            for index in range(len(state)):
                specs[_pointer(_pointer(path, "state"), index)] = (
                    TOKEN_PYTHON_MT_WORD,
                    None,
                )
            if "index" in value:
                specs[_pointer(path, "index")] = (TOKEN_INTEGER, None)
        elif isinstance(value, list) and len(value) == 3 and isinstance(value[1], list):
            gauss_path = _pointer(path, 2)
            if value[2] is not None and (
                type(value[2]) is not float or not math.isfinite(value[2])
            ):
                _block(gauss_path, "Python Gaussian cache must be null or a finite float")
            specs[gauss_path] = (TOKEN_NULLABLE_FLOAT64, None)
            if not value[1]:
                _block(_pointer(path, 1), "Python MT state list is empty")
            words_path = _pointer(path, 1)
            for index in range(len(value[1]) - 1):
                specs[_pointer(words_path, index)] = (TOKEN_PYTHON_MT_WORD, None)
            specs[_pointer(words_path, len(value[1]) - 1)] = (TOKEN_INTEGER, None)
        else:
            _block(path, "Python MT state structure is not fixed before dispatch")

    streams_name = "native_runtime_rng_manager_stream_indices"
    if streams_name in metadata:
        root = _pointer(metadata_base, streams_name)

        def add_stream_indices(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    add_stream_indices(child, _pointer(path, key))
            elif type(value) is int:
                specs[path] = (TOKEN_INTEGER, None)
            else:
                _block(path, "RNG stream index authority must contain only mappings and ints")

        add_stream_indices(metadata[streams_name], root)
    return specs


def _validate_static_tree(
    actual: Any,
    expected: Any,
    *,
    path: str,
    dynamic_specs: Mapping[str, tuple[str, int | None]],
) -> None:
    spec = dynamic_specs.get(path)
    if spec is not None:
        if (
            path in _PRE_DISPATCH_COORDINATE_PATHS
            and actual == expected
            and type(actual) is type(expected)
        ):
            return
        _expect_placeholder(actual, path=path, token_type=spec[0], rank=spec[1])
        return
    if _placeholder_spec(actual) is not None:
        _block(path, "the frozen contract does not classify this value as tokenizable")
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise TypeError(f"{path} must be a mapping")
        if set(actual) != set(expected):
            _fail(path, "static mapping structure differs from the accepted skeleton")
        for key in expected:
            _validate_static_tree(
                actual[key],
                expected[key],
                path=_pointer(path, key),
                dynamic_specs=dynamic_specs,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            raise TypeError(f"{path} must be a list")
        if len(actual) != len(expected):
            _block(path, "dynamic list cardinality cannot be selected before dispatch")
        for index, child in enumerate(expected):
            _validate_static_tree(
                actual[index],
                child,
                path=_pointer(path, index),
                dynamic_specs=dynamic_specs,
            )
        return
    if actual != expected or type(actual) is not type(expected):
        if type(expected) in (float, str):
            _block(path, f"generation-varying {type(expected).__name__} is not tokenizable")
        _fail(path, "must remain equal to the accepted pre-dispatch skeleton")


def _shape_rank(row: Mapping[str, Any], output: str) -> int:
    key = "canonical_rank" if output == "canonical" else "raw_rank"
    rank = row.get(key)
    if type(rank) is not int or rank < 0:
        _fail(f"descriptor_plan.{row.get('tensor_name')}.{key}", "invalid rank")
    return rank


def _validate_source_manifests(
    checkpoint: Mapping[str, Any],
    projection_fields: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    world: str,
) -> None:
    expected_names = [row["tensor_name"] for row in rows]
    manifest = _mapping(checkpoint.get("tensor_manifest"), path="/checkpoint/tensor_manifest")
    stored = _mapping(
        projection_fields.get("tensor_manifest"),
        path="/checkpoint/projection_fields/tensor_manifest",
    )
    expected_field_set = set(expected_names)
    stored_expected_field_set = (
        expected_field_set - set(SOURCE_PATCH_DERIVED_TENSOR_FIELDS)
        if world == WORLD_PATCH
        else expected_field_set
    )
    static_fields = set(_STATIC_PRE_DISPATCH_SOURCE_TENSORS)
    manifest_fields = set(manifest)
    stored_fields = set(stored)
    if (
        manifest_fields not in (
            expected_field_set,
            expected_field_set | static_fields,
        )
        or stored_fields not in (
            stored_expected_field_set,
            stored_expected_field_set | static_fields,
        )
        or manifest_fields - expected_field_set
        != stored_fields - stored_expected_field_set
    ):
        _fail(
            "/checkpoint/tensor_manifest",
            f"fields must exactly match the frozen {world} descriptors",
        )
    row_by_name = {row["tensor_name"]: row for row in rows}
    for name in expected_names:
        row = row_by_name[name]
        output = row["digest_inclusion"]["source_projection_output"]
        rank = _shape_rank(row, output)
        source_path = _pointer("/checkpoint/tensor_manifest", name)
        entry = _exact_mapping(
            manifest[name],
            ("device", "dtype", "shape", "sha256"),
            path=source_path,
        )
        if entry["device"] != "cpu" or entry["dtype"] != row["dtype"]:
            _fail(source_path, "device/dtype contradict the descriptor plan")
        _expect_placeholder(
            entry["shape"],
            path=_pointer(source_path, "shape"),
            token_type=TOKEN_TENSOR_SHAPE,
            rank=rank,
        )
        _expect_placeholder(
            entry["sha256"],
            path=_pointer(source_path, "sha256"),
            token_type=TOKEN_SHA256_HEX,
        )

        if name not in stored_expected_field_set:
            continue
        stored_path = _pointer("/checkpoint/projection_fields/tensor_manifest", name)
        stored_entry = _exact_mapping(
            stored[name],
            (
                "shape",
                "dtype",
                "device",
                "numel",
                "element_size",
                "bytes",
                "is_cuda",
                "is_contiguous",
            ),
            path=stored_path,
        )
        fixed = {
            "dtype": row["torch_tensor_manifest_dtype"],
            "device": "cpu",
            "element_size": row["dtype_bytes"],
            "is_cuda": False,
            "is_contiguous": True,
        }
        for field, expected in fixed.items():
            if stored_entry[field] != expected or type(stored_entry[field]) is not type(expected):
                _fail(_pointer(stored_path, field), "contradicts the descriptor plan")
        _expect_placeholder(
            stored_entry["shape"],
            path=_pointer(stored_path, "shape"),
            token_type=TOKEN_TENSOR_SHAPE,
            rank=rank,
        )
        for field in ("numel", "bytes"):
            _expect_placeholder(
                stored_entry[field],
                path=_pointer(stored_path, field),
                token_type=TOKEN_INTEGER,
            )
    static_contracts = {
        "drive_noise_samples_by_tick_slot": ("float32", "Float", 3, 4),
        "drive_noise_samples_valid_by_tick_slot": ("bool", "Bool", 2, 1),
    }
    for name in _STATIC_PRE_DISPATCH_SOURCE_TENSORS:
        if name not in manifest_fields:
            continue
        numpy_dtype, torch_dtype, rank, element_size = static_contracts[name]
        source_path = _pointer("/checkpoint/tensor_manifest", name)
        entry = _exact_mapping(
            manifest[name],
            ("device", "dtype", "shape", "sha256"),
            path=source_path,
        )
        shape = entry["shape"]
        if (
            entry["device"] != "cpu"
            or entry["dtype"] != numpy_dtype
            or not isinstance(shape, list)
            or len(shape) != rank
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or type(entry["sha256"]) is not str
            or _SHA256_RE.fullmatch(entry["sha256"]) is None
        ):
            _fail(source_path, "static source tensor manifest is malformed")
        stored_path = _pointer(
            "/checkpoint/projection_fields/tensor_manifest", name
        )
        stored_entry = _exact_mapping(
            stored[name],
            (
                "shape",
                "dtype",
                "device",
                "numel",
                "element_size",
                "bytes",
                "is_cuda",
                "is_contiguous",
            ),
            path=stored_path,
        )
        numel = 1
        for dimension in shape:
            numel *= dimension
        if stored_entry != {
            "shape": shape,
            "dtype": torch_dtype,
            "device": "cpu",
            "numel": numel,
            "element_size": element_size,
            "bytes": numel * element_size,
            "is_cuda": False,
            "is_contiguous": True,
        }:
            _fail(stored_path, "static stored tensor manifest is malformed")


def _validate_layer_fields(
    checkpoint: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    authority_name: str,
    layer: DigestLayer,
) -> Sequence[Mapping[str, Any]]:
    value = checkpoint.get(authority_name)
    if not isinstance(value, list):
        raise TypeError(f"/checkpoint/{authority_name} must be a list")
    expected_rows = [
        row for row in rows if str(layer) in row["digest_inclusion"]["checkpoint_layers"]
    ]
    expected_names = [row["tensor_name"] for row in expected_rows]
    if len(value) != len(expected_rows):
        _fail(f"/checkpoint/{authority_name}", "field count contradicts the descriptor plan")
    for index, (entry, row, expected_name) in enumerate(zip(value, expected_rows, expected_names)):
        path = _pointer(_pointer("/checkpoint", authority_name), index)
        resolved = _exact_mapping(entry, ("field", "dtype", "shape", "sha256"), path=path)
        if resolved["field"] != expected_name or resolved["dtype"] != row["dtype"]:
            _fail(path, "field order or dtype contradicts the descriptor plan")
        _expect_placeholder(
            resolved["shape"],
            path=_pointer(path, "shape"),
            token_type=TOKEN_TENSOR_SHAPE,
            rank=_shape_rank(row, "canonical"),
        )
        _expect_placeholder(
            resolved["sha256"],
            path=_pointer(path, "sha256"),
            token_type=TOKEN_SHA256_HEX,
        )
    return value


def _validate_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("source_version", ACTIVE_AGENT_IDS_SCHEMA_VERSION) != ACTIVE_AGENT_IDS_SCHEMA_VERSION:
        _fail("/identity/source_version", "schema drift")
    _expect_placeholder(
        identity.get("active_agent_count"),
        path="/identity/active_agent_count",
        token_type=TOKEN_INTEGER,
    )
    _expect_placeholder(
        identity.get("tensor_sha256"),
        path="/identity/tensor_sha256",
        token_type=TOKEN_SHA256_HEX,
    )


def _validate_policy(
    policy: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if policy.get("source_version", POLICY_MUTATION_TAPE_SCHEMA_VERSION) != POLICY_MUTATION_TAPE_SCHEMA_VERSION:
        _fail("/policy_mutation_tape/source_version", "schema drift")
    _expect_placeholder(
        policy.get("cursor"),
        path="/policy_mutation_tape/cursor",
        token_type=TOKEN_INTEGER,
    )
    shapes = _mapping(
        policy.get("consumed_shapes"),
        path="/policy_mutation_tape/consumed_shapes",
    )
    hashes = _mapping(
        policy.get("tensor_sha256"),
        path="/policy_mutation_tape/tensor_sha256",
    )
    names = [row["tensor_name"] for row in rows]
    if set(shapes) != set(names) or set(hashes) != set(names):
        _fail("/policy_mutation_tape", "fields contradict the descriptor plan")
    for row in rows:
        name = row["tensor_name"]
        _expect_placeholder(
            shapes[name],
            path=_pointer("/policy_mutation_tape/consumed_shapes", name),
            token_type=TOKEN_TENSOR_SHAPE,
            rank=_shape_rank(row, "canonical"),
        )
        _expect_placeholder(
            hashes[name],
            path=_pointer("/policy_mutation_tape/tensor_sha256", name),
            token_type=TOKEN_SHA256_HEX,
        )


def _coordinate_authority(
    raw: Any,
    *,
    generation: int,
    cell: int,
    skeleton: Mapping[str, Any],
    dynamic_specs: Mapping[str, tuple[str, int | None]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    world: str,
) -> dict[str, Any]:
    authority = _mapping(raw, path=f"placeholder_retained_authorities[{generation}][{cell}]")
    for field, expected in (("generation", generation), ("cell", cell)):
        value = authority.get(field)
        if value == expected and type(value) is type(expected):
            continue
        if _placeholder_spec(value) is not None:
            _expect_placeholder(value, path=f"/{field}", token_type=TOKEN_INTEGER)
            continue
        _fail(f"/{field}", f"must equal coordinate value {expected!r} or be its typed placeholder")
    if authority.get("world") != world or type(authority.get("world")) is not str:
        _fail("/world", f"must equal coordinate value {world!r}")
    checkpoint = _mapping(authority.get("checkpoint"), path="/checkpoint")
    projection_schema = checkpoint.get("projection_schema_version")
    if projection_schema != skeleton.get("schema_version"):
        _fail("/checkpoint/projection_schema_version", "contradicts the accepted skeleton")
    projection_fields = _mapping(
        checkpoint.get("projection_fields"),
        path="/checkpoint/projection_fields",
    )
    expected_projection = deepcopy({key: value for key, value in skeleton.items() if key != "tensors"})
    expected_projection["metadata"]["generation"] = generation + 1
    expected_projection["metadata"]["tick"] = 0
    expected_projection["metadata"]["agent_grid_authority_epoch"] = (
        f"generation:{generation + 1}:tick:0"
    )
    expected_projection["last_world"] = world
    if set(projection_fields) != set(expected_projection):
        _fail("/checkpoint/projection_fields", "top-level structure differs from skeleton")
    for field in ("pending_generation_free_agent_ids", "pending_generation_free_slots"):
        if projection_fields[field] != []:
            _block(
                _pointer("/checkpoint/projection_fields", field),
                "variable-cardinality pending-free authority cannot be fixed before dispatch",
            )
    for key in expected_projection:
        if key == "tensor_manifest":
            continue
        _validate_static_tree(
            projection_fields[key],
            expected_projection[key],
            path=_pointer("/checkpoint/projection_fields", key),
            dynamic_specs=dynamic_specs,
        )
    _validate_source_manifests(
        checkpoint,
        projection_fields,
        checkpoint_rows,
        world=world,
    )
    trajectory = _validate_layer_fields(
        checkpoint,
        checkpoint_rows,
        authority_name="trajectory_state_fields",
        layer=DigestLayer.TRAJECTORY_STATE,
    )
    storage = _validate_layer_fields(
        checkpoint,
        checkpoint_rows,
        authority_name="canonical_checkpoint_storage_fields",
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    )
    identity = _mapping(authority.get("identity"), path="/identity")
    policy = _mapping(authority.get("policy_mutation_tape"), path="/policy_mutation_tape")
    _validate_identity(identity)
    _validate_policy(policy, policy_rows)
    return {
        "projection_schema": projection_schema,
        "projection_fields": projection_fields,
        "tensor_manifest": checkpoint["tensor_manifest"],
        "trajectory": trajectory,
        "storage": storage,
        "identity": identity,
        "policy": policy,
        "world": world,
    }


def _canonical_literal(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _placeholder_with_pack_depth(value: Mapping[str, Any], depth: int) -> dict[str, Any]:
    marker = dict(value[TOKEN_PLACEHOLDER_KEY])
    marker["render_pack_depth"] = marker.get("render_pack_depth", 0) + depth
    return {TOKEN_PLACEHOLDER_KEY: marker}


def _pack_concrete_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = [
            (_pack_concrete_value(key), _pack_concrete_value(child))
            for key, child in value.items()
        ]
        items.sort(key=lambda row: _canonical_literal(row[0]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_pack_concrete_value(child) for child in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_pack_concrete_value(child) for child in value],
        }
    if isinstance(value, (bytes, bytearray)):
        return {"kind": "bytes", "hex": bytes(value).hex()}
    if type(value) in (str, int, float, bool) or value is None:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("packed source projection contains a non-finite float")
        return {"kind": type(value).__name__, "value": value}
    raise TypeError(f"unsupported retained authority value type: {type(value).__name__}")


def _pack_value(value: Any) -> Any:
    if _placeholder_spec(value) is not None:
        return _placeholder_with_pack_depth(value, 1)
    if isinstance(value, Mapping):
        items = [(_pack_value(key), _pack_value(child)) for key, child in value.items()]
        items.sort(key=lambda row: _canonical_literal(row[0]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_pack_value(child) for child in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_pack_value(child) for child in value]}
    if isinstance(value, (bytes, bytearray)):
        return {"kind": "bytes", "hex": bytes(value).hex()}
    if type(value) in (str, int, float, bool) or value is None:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("packed source projection contains a non-finite float")
        return {"kind": type(value).__name__, "value": value}
    raise TypeError(f"unsupported retained authority value type: {type(value).__name__}")


def _checkpoint_layer_payload(
    *,
    layer: DigestLayer,
    projection_schema: str,
    generation: int,
    fields: Sequence[Mapping[str, Any]],
    world: str,
) -> dict[str, Any]:
    schema = projection_schema
    if (
        layer is DigestLayer.CANONICAL_CHECKPOINT_STORAGE
        and schema in CANONICAL_STORAGE_EQUIVALENT_PROJECTION_SCHEMAS
    ):
        schema = CHECKPOINT_LAYER_SCHEMA_VERSION
    return {
        "layer": str(layer),
        "fields": list(fields),
        "metadata": {
            "schema_version": schema,
            "world": world,
            "generation": generation + 1,
            "tick": 0,
        },
    }


def _source_projection_payload(authority: Mapping[str, Any]) -> dict[str, Any]:
    projection_fields = authority["projection_fields"]
    manifest = authority["tensor_manifest"]
    packed_tensors = {
        name: {
            "kind": "tensor",
            "device": entry["device"],
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "sha256": entry["sha256"],
        }
        for name, entry in sorted(manifest.items())
    }
    packed_tensor_mapping = {
        "kind": "mapping",
        "items": [
            (_pack_value(name), packed_tensors[name]) for name in sorted(packed_tensors)
        ],
    }
    top_level_fields = sorted((*projection_fields, "tensors"))
    packed_fields = {
        name: (
            packed_tensor_mapping
            if name == "tensors"
            else _pack_value(projection_fields[name])
        )
        for name in top_level_fields
    }
    return {
        "kind": "mapping",
        "items": [
            (_pack_value(name), packed_fields[name]) for name in top_level_fields
        ],
    }


def _rng_payload(authority: Mapping[str, Any]) -> Any:
    metadata = authority["projection_fields"]["metadata"]
    fields = [
        name
        for name in (
            *SOURCE_RNG_REQUIRED_METADATA_FIELDS,
            *SOURCE_RNG_OPTIONAL_METADATA_FIELDS,
        )
        if name in metadata
    ]
    missing = sorted(set(SOURCE_RNG_REQUIRED_METADATA_FIELDS) - set(metadata))
    if missing:
        _fail("/checkpoint/projection_fields/metadata", f"missing RNG fields {missing}")
    tensors = {
        name: {
            "kind": "tensor",
            "device": authority["tensor_manifest"][name]["device"],
            "dtype": authority["tensor_manifest"][name]["dtype"],
            "shape": authority["tensor_manifest"][name]["shape"],
            "sha256": authority["tensor_manifest"][name]["sha256"],
        }
        for name in SOURCE_RNG_TENSOR_FIELDS
        if name in authority["tensor_manifest"]
    }
    packed = {
        "metadata": {name: _pack_value(metadata[name]) for name in fields},
        "tensors": tensors,
    }
    return _pack_value(packed)


def _active_ids_payload(authority: Mapping[str, Any]) -> dict[str, Any]:
    identity = authority["identity"]
    return {
        "schema_version": ACTIVE_AGENT_IDS_SCHEMA_VERSION,
        "fields": [
            {
                "field": "agent_ids_by_alive_order",
                "dtype": "int64",
                "shape": [identity["active_agent_count"]],
                "sha256": identity["tensor_sha256"],
            }
        ],
    }


def _policy_payload(authority: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policy = authority["policy"]
    return {
        "schema_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
        "cursor": policy["cursor"],
        "fields": [
            {
                "field": row["tensor_name"],
                "dtype": row["dtype"],
                "shape": policy["consumed_shapes"][row["tensor_name"]],
                "sha256": policy["tensor_sha256"][row["tensor_name"]],
            }
            for row in rows
        ],
    }


def _token_max_bytes(spec: Mapping[str, Any]) -> int:
    token_type = spec["token_type"]
    rule = spec["validation_rule"]
    if token_type == TOKEN_INTEGER:
        representative: Any = UINT64_MAX
    elif token_type == TOKEN_BOOLEAN:
        representative = False
    elif token_type == TOKEN_PCG_UINT128:
        representative = UINT128_MAX
    elif token_type == TOKEN_PYTHON_MT_WORD:
        representative = UINT32_MAX
    elif token_type == TOKEN_SHA256_HEX:
        representative = "f" * 64
    elif token_type == TOKEN_NULLABLE_FLOAT64:
        representative = -float.fromhex("0x1.fffffffffffffp+1023")
    elif token_type == TOKEN_TENSOR_SHAPE:
        representative = [UINT64_MAX] * rule["rank"]
    else:
        raise AssertionError(token_type)
    for _ in range(rule["render_pack_depth"]):
        representative = _pack_concrete_value(representative)
    return len(_canonical_literal(representative).encode("utf-8"))


def _compile_json_template(
    payload: Any,
    *,
    surface: str,
    generation: int,
    cell: int,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    token_indices: dict[str, int] = {}

    def literal(text: str) -> None:
        if not text:
            return
        if parts and parts[-1]["kind"] == "literal_utf8":
            parts[-1]["value"] += text
        else:
            parts.append({"kind": "literal_utf8", "value": text})

    def token(value: Any) -> None:
        spec = _placeholder_spec(value)
        assert spec is not None
        source_path = spec["source_path"]
        index = token_indices.get(source_path)
        if index is None:
            index = len(tokens)
            token_indices[source_path] = index
            token_id = f"g{generation}.c{cell}.{surface}.t{index}"
            resolved = {
                "token_index": index,
                "token_id": token_id,
                "token_type": spec["token_type"],
                "source_path": source_path,
                "generation": generation,
                "cell": cell,
                "validation_rule": spec["validation_rule"],
            }
            resolved["maximum_utf8_bytes"] = _token_max_bytes(resolved)
            tokens.append(resolved)
        elif (
            tokens[index]["token_type"] != spec["token_type"]
            or tokens[index]["validation_rule"] != spec["validation_rule"]
        ):
            _fail(source_path, "one source path has contradictory token declarations")
        parts.append({"kind": "dynamic_token", "token_index": index})

    def emit(value: Any) -> None:
        if _placeholder_spec(value) is not None:
            token(value)
            return
        if isinstance(value, Mapping):
            if any(type(key) is not str for key in value):
                raise TypeError("canonical template mapping keys must be strings")
            literal("{")
            for index, key in enumerate(sorted(value)):
                if index:
                    literal(",")
                literal(_canonical_literal(key))
                literal(":")
                emit(value[key])
            literal("}")
            return
        if isinstance(value, (list, tuple)):
            literal("[")
            for index, child in enumerate(value):
                if index:
                    literal(",")
                emit(child)
            literal("]")
            return
        literal(_canonical_literal(value))

    emit(payload)
    if len(tokens) > MAX_TOKENS_PER_TEMPLATE:
        _fail(surface, "token count exceeds the bounded callback contract")
    literal_bytes = sum(
        len(part["value"].encode("utf-8"))
        for part in parts
        if part["kind"] == "literal_utf8"
    )
    maximum = literal_bytes + sum(token["maximum_utf8_bytes"] for token in tokens)
    if literal_bytes > MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE:
        _fail(surface, "literal bytes exceed the bounded callback contract")
    if maximum > MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE:
        _fail(surface, "rendered bytes exceed the bounded callback contract")
    return {
        "schema_version": HOST_TEMPLATE_SCHEMA_VERSION,
        "surface": surface,
        "wire_field": WIRE_FIELDS[surface],
        "generation": generation,
        "cell": cell,
        "encoding": "utf-8",
        "literal_utf8_bytes": literal_bytes,
        "token_count": len(tokens),
        "maximum_rendered_utf8_bytes": maximum,
        "tokens": tokens,
        "parts": parts,
        "constraints": [],
    }


def _attach_relational_constraints(
    template: dict[str, Any],
    *,
    authority: Mapping[str, Any],
    policy_rows: Sequence[Mapping[str, Any]],
) -> None:
    surface = template["surface"]
    constraints: list[dict[str, Any]] = []
    if surface == SURFACE_POLICY_MUTATION_TAPE:
        cursor = "/policy_mutation_tape/cursor"
        shape_paths = {
            row["tensor_name"]: _pointer(
                "/policy_mutation_tape/consumed_shapes", row["tensor_name"]
            )
            for row in policy_rows
        }
        constraints.extend(
            {
                "kind": "shape_first_dimension_equals_integer",
                "shape_source_path": path,
                "integer_source_path": cursor,
            }
            for path in shape_paths.values()
        )
        delta = shape_paths["policy_birth_delta_tape"]
        for name in (
            "policy_birth_crossover_mask_tape",
            "policy_birth_mutation_mask_tape",
        ):
            constraints.append(
                {
                    "kind": "values_equal",
                    "left_source_path": shape_paths[name],
                    "right_source_path": delta,
                }
            )
    elif surface == SURFACE_RNG:
        metadata = authority["projection_fields"]["metadata"]
        if metadata["native_runtime_env_policy_numpy_alias"] is True:
            env_prefix = (
                "/checkpoint/projection_fields/metadata/"
                "native_runtime_numpy_env_state"
            )
            policy_prefix = (
                "/checkpoint/projection_fields/metadata/"
                "native_runtime_numpy_policy_state"
            )
            paths = {token["source_path"] for token in template["tokens"]}
            for env_path in sorted(path for path in paths if path.startswith(env_prefix)):
                policy_path = policy_prefix + env_path[len(env_prefix) :]
                if policy_path not in paths:
                    _fail(env_path, "RNG alias token structure contradicts policy state")
                constraints.append(
                    {
                        "kind": "values_equal",
                        "left_source_path": env_path,
                        "right_source_path": policy_path,
                    }
                )
    template["constraints"] = constraints


def _normalize_authorities(
    value: Any,
    *,
    generation_count: int,
    cell_count: int,
) -> list[list[Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if generation_count == cell_count == 1:
            return [[value]]
        raise TypeError("multiple coordinates require a generation-major authority sequence")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("placeholder_retained_authorities must be a sequence")
    if len(value) == generation_count and all(
        isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray, Mapping))
        for row in value
    ):
        matrix = [list(row) for row in value]
        if any(len(row) != cell_count for row in matrix):
            _fail("placeholder_retained_authorities", "cell count mismatch")
        return matrix
    if len(value) == generation_count * cell_count and all(
        isinstance(row, Mapping) for row in value
    ):
        flat = list(value)
        return [
            flat[generation * cell_count : (generation + 1) * cell_count]
            for generation in range(generation_count)
        ]
    _fail("placeholder_retained_authorities", "generation/cell shape mismatch")
    raise AssertionError("unreachable")


def compile_compact_summary_host_templates(
    checkpoint_projection_skeleton: Mapping[str, Any] | str | Path,
    descriptor_plan: Mapping[str, Any],
    generation_count: int,
    cell_count: int,
    placeholder_retained_authorities: Any,
) -> dict[str, Any]:
    """Compile all six Flat or Patch P2 outer digest preimages before dispatch."""

    generations = _strict_count(generation_count, path="generation_count")
    cells = _strict_count(cell_count, path="cell_count")
    if generations * cells > MAX_COORDINATES:
        _fail("generation_count/cell_count", "coordinate product exceeds bounded contract")
    plan = _mapping(descriptor_plan, path="descriptor_plan")
    skeleton = _load_projection_skeleton(checkpoint_projection_skeleton, plan)
    world = _projection_world(skeleton)
    checkpoint_rows, policy_rows = _descriptor_rows(plan, world=world)
    _validate_projection_skeleton(
        skeleton,
        plan,
        checkpoint_rows,
        world=world,
    )
    dynamic_specs = _dynamic_projection_specs(skeleton)
    matrix = _normalize_authorities(
        placeholder_retained_authorities,
        generation_count=generations,
        cell_count=cells,
    )

    templates: list[dict[str, Any]] = []
    for generation in range(generations):
        for cell in range(cells):
            authority = _coordinate_authority(
                matrix[generation][cell],
                generation=generation,
                cell=cell,
                skeleton=skeleton,
                dynamic_specs=dynamic_specs,
                checkpoint_rows=checkpoint_rows,
                policy_rows=policy_rows,
                world=world,
            )
            payloads = {
                SURFACE_CHECKPOINT_TRAJECTORY: _checkpoint_layer_payload(
                    layer=DigestLayer.TRAJECTORY_STATE,
                    projection_schema=authority["projection_schema"],
                    generation=generation,
                    fields=authority["trajectory"],
                    world=world,
                ),
                SURFACE_CANONICAL_STORAGE: _checkpoint_layer_payload(
                    layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
                    projection_schema=authority["projection_schema"],
                    generation=generation,
                    fields=authority["storage"],
                    world=world,
                ),
                SURFACE_NEXT_GENERATION_SOURCE: _source_projection_payload(authority),
                SURFACE_RNG: _rng_payload(authority),
                SURFACE_ACTIVE_AGENT_IDS: _active_ids_payload(authority),
                SURFACE_POLICY_MUTATION_TAPE: _policy_payload(authority, policy_rows),
            }
            for surface in SURFACE_ORDER:
                template = _compile_json_template(
                    payloads[surface],
                    surface=surface,
                    generation=generation,
                    cell=cell,
                )
                _attach_relational_constraints(
                    template,
                    authority=authority,
                    policy_rows=policy_rows,
                )
                template["template_index"] = len(templates)
                templates.append(template)
    return {
        "schema_version": HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION,
        "representation": "immutable_utf8_segments_plus_bounded_dynamic_tokens",
        "ordering": "generation_major_cell_major_surface_order",
        "surface_order": list(SURFACE_ORDER),
        "generation_count": generations,
        "cell_count": cells,
        "coordinate_count": generations * cells,
        "template_count": len(templates),
        "bounds": {
            "maximum_coordinates": MAX_COORDINATES,
            "maximum_tokens_per_template": MAX_TOKENS_PER_TEMPLATE,
            "maximum_literal_utf8_bytes_per_template": (
                MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE
            ),
            "maximum_rendered_utf8_bytes_per_template": (
                MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE
            ),
        },
        "templates": templates,
    }


def _assign_pointer(root: Any, pointer: str, value: Any) -> None:
    components = [
        raw.replace("~1", "/").replace("~0", "~")
        for raw in pointer.split("/")[1:]
    ]
    current = root
    for component in components[:-1]:
        current = current[int(component)] if isinstance(current, list) else current[component]
    final = components[-1]
    if isinstance(current, list):
        current[int(final)] = value
    else:
        current[final] = value


def _placeholder_manifest_entry(
    row: Mapping[str, Any],
    *,
    base_path: str,
) -> dict[str, Any]:
    output = row["digest_inclusion"]["source_projection_output"]
    rank = _shape_rank(row, output)
    return {
        "device": "cpu",
        "dtype": row["dtype"],
        "shape": dynamic_token_placeholder(
            TOKEN_TENSOR_SHAPE,
            _pointer(base_path, "shape"),
            rank=rank,
        ),
        "sha256": dynamic_token_placeholder(
            TOKEN_SHA256_HEX,
            _pointer(base_path, "sha256"),
        ),
    }


def _placeholder_stored_manifest_entry(
    row: Mapping[str, Any],
    *,
    base_path: str,
) -> dict[str, Any]:
    output = row["digest_inclusion"]["source_projection_output"]
    rank = _shape_rank(row, output)
    return {
        "shape": dynamic_token_placeholder(
            TOKEN_TENSOR_SHAPE,
            _pointer(base_path, "shape"),
            rank=rank,
        ),
        "dtype": row["torch_tensor_manifest_dtype"],
        "device": "cpu",
        "numel": dynamic_token_placeholder(
            TOKEN_INTEGER,
            _pointer(base_path, "numel"),
        ),
        "element_size": row["dtype_bytes"],
        "bytes": dynamic_token_placeholder(
            TOKEN_INTEGER,
            _pointer(base_path, "bytes"),
        ),
        "is_cuda": False,
        "is_contiguous": True,
    }


def _static_source_manifest_entry(value: Any, *, path: str) -> dict[str, Any]:
    if not all(hasattr(value, name) for name in ("detach", "to", "contiguous")):
        raise TypeError(f"{path} must be a tensor")
    source_device = str(getattr(value, "device", "cpu"))
    tensor = value.detach().to("cpu").contiguous()
    array = tensor.numpy()
    return {
        "device": source_device,
        "dtype": str(array.dtype),
        "shape": [int(dimension) for dimension in array.shape],
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _placeholder_layer_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    authority_name: str,
    layer: DigestLayer,
) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for row in rows:
        if str(layer) not in row["digest_inclusion"]["checkpoint_layers"]:
            continue
        path = _pointer(_pointer("/checkpoint", authority_name), len(fields))
        fields.append(
            {
                "field": row["tensor_name"],
                "dtype": row["dtype"],
                "shape": dynamic_token_placeholder(
                    TOKEN_TENSOR_SHAPE,
                    _pointer(path, "shape"),
                    rank=_shape_rank(row, "canonical"),
                ),
                "sha256": dynamic_token_placeholder(
                    TOKEN_SHA256_HEX,
                    _pointer(path, "sha256"),
                ),
            }
        )
    return fields


def _placeholder_authority_for_coordinate(
    skeleton: Mapping[str, Any],
    *,
    generation: int,
    cell: int,
    dynamic_specs: Mapping[str, tuple[str, int | None]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    world: str,
) -> dict[str, Any]:
    projection = deepcopy({key: value for key, value in skeleton.items() if key != "tensors"})
    projection["metadata"]["generation"] = generation + 1
    projection["metadata"]["tick"] = 0
    projection["metadata"]["agent_grid_authority_epoch"] = (
        f"generation:{generation + 1}:tick:0"
    )
    projection["last_world"] = world

    source_manifest: dict[str, dict[str, Any]] = {}
    stored_manifest: dict[str, dict[str, Any]] = {}
    for row in checkpoint_rows:
        name = row["tensor_name"]
        source_path = _pointer("/checkpoint/tensor_manifest", name)
        stored_path = _pointer(
            "/checkpoint/projection_fields/tensor_manifest",
            name,
        )
        source_manifest[name] = _placeholder_manifest_entry(row, base_path=source_path)
        if not (
            world == WORLD_PATCH and name in SOURCE_PATCH_DERIVED_TENSOR_FIELDS
        ):
            stored_manifest[name] = _placeholder_stored_manifest_entry(
                row,
                base_path=stored_path,
            )
    skeleton_tensors = _mapping(
        skeleton["tensors"], path="checkpoint_projection_skeleton.tensors"
    )
    skeleton_manifest = _mapping(
        skeleton["tensor_manifest"],
        path="checkpoint_projection_skeleton.tensor_manifest",
    )
    for name in _STATIC_PRE_DISPATCH_SOURCE_TENSORS:
        if name not in skeleton_manifest:
            continue
        if name not in skeleton_tensors:
            _fail(
                f"checkpoint_projection_skeleton.tensors.{name}",
                "static source tensor is absent",
            )
        source_manifest[name] = _static_source_manifest_entry(
            skeleton_tensors[name],
            path=f"checkpoint_projection_skeleton.tensors.{name}",
        )
        stored_manifest[name] = deepcopy(skeleton_manifest[name])
    projection["tensor_manifest"] = stored_manifest

    policy_shapes: dict[str, Any] = {}
    policy_hashes: dict[str, Any] = {}
    for row in policy_rows:
        name = row["tensor_name"]
        policy_shapes[name] = dynamic_token_placeholder(
            TOKEN_TENSOR_SHAPE,
            _pointer("/policy_mutation_tape/consumed_shapes", name),
            rank=_shape_rank(row, "canonical"),
        )
        policy_hashes[name] = dynamic_token_placeholder(
            TOKEN_SHA256_HEX,
            _pointer("/policy_mutation_tape/tensor_sha256", name),
        )

    authority: dict[str, Any] = {
        "generation": dynamic_token_placeholder(TOKEN_INTEGER, "/generation"),
        "cell": dynamic_token_placeholder(TOKEN_INTEGER, "/cell"),
        "world": world,
        "checkpoint": {
            "source_version": CHECKPOINT_LAYER_SCHEMA_VERSION,
            "rng_source_version": SOURCE_RNG_SCHEMA_VERSION,
            "projection_schema_version": projection["schema_version"],
            "projection_fields": projection,
            "tensor_manifest": source_manifest,
            "trajectory_state_fields": _placeholder_layer_fields(
                checkpoint_rows,
                authority_name="trajectory_state_fields",
                layer=DigestLayer.TRAJECTORY_STATE,
            ),
            "canonical_checkpoint_storage_fields": _placeholder_layer_fields(
                checkpoint_rows,
                authority_name="canonical_checkpoint_storage_fields",
                layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
            ),
        },
        "identity": {
            "source_version": ACTIVE_AGENT_IDS_SCHEMA_VERSION,
            "active_agent_count": dynamic_token_placeholder(
                TOKEN_INTEGER,
                "/identity/active_agent_count",
            ),
            "tensor_sha256": dynamic_token_placeholder(
                TOKEN_SHA256_HEX,
                "/identity/tensor_sha256",
            ),
        },
        "policy_mutation_tape": {
            "source_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
            "cursor": dynamic_token_placeholder(
                TOKEN_INTEGER,
                "/policy_mutation_tape/cursor",
            ),
            "consumed_shapes": policy_shapes,
            "tensor_sha256": policy_hashes,
        },
    }
    for path, (token_type, rank) in dynamic_specs.items():
        _assign_pointer(
            authority,
            path,
            dynamic_token_placeholder(token_type, path, rank=rank),
        )
    return authority


def _compile_device_neutral_compact_summary_host_programs(
    checkpoint_projection_skeleton: Mapping[str, Any] | str | Path,
    descriptor_plan: Mapping[str, Any],
    generation_count: int,
    cell_count: int,
    *,
    flat_only: bool,
) -> dict[str, Any]:
    generations = _strict_count(generation_count, path="generation_count")
    cells = _strict_count(cell_count, path="cell_count")
    if generations * cells > MAX_COORDINATES:
        _fail("generation_count/cell_count", "coordinate product exceeds bounded contract")
    plan = _mapping(descriptor_plan, path="descriptor_plan")
    skeleton = _load_projection_skeleton(
        checkpoint_projection_skeleton,
        plan,
        flat_only=flat_only,
    )
    world = _projection_world(skeleton, flat_only=flat_only)
    checkpoint_rows, policy_rows = _descriptor_rows(plan, world=world)
    _validate_projection_skeleton(
        skeleton,
        plan,
        checkpoint_rows,
        world=world,
    )
    dynamic_specs = _dynamic_projection_specs(skeleton)
    authorities = [
        [
            _placeholder_authority_for_coordinate(
                skeleton,
                generation=generation,
                cell=cell,
                dynamic_specs=dynamic_specs,
                checkpoint_rows=checkpoint_rows,
                policy_rows=policy_rows,
                world=world,
            )
            for cell in range(cells)
        ]
        for generation in range(generations)
    ]
    return compile_compact_summary_host_templates(
        checkpoint_projection_skeleton,
        plan,
        generations,
        cells,
        authorities,
    )


def compile_device_neutral_compact_summary_host_programs(
    checkpoint_projection_skeleton: Mapping[str, Any] | str | Path,
    descriptor_plan: Mapping[str, Any],
    generation_count: int,
    cell_count: int,
) -> dict[str, Any]:
    """Compile exact Flat or Patch P2 programs without reading live device state."""

    return _compile_device_neutral_compact_summary_host_programs(
        checkpoint_projection_skeleton,
        descriptor_plan,
        generation_count,
        cell_count,
        flat_only=False,
    )


def compile_compact_summary_host_programs(
    checkpoint_projection_skeleton: Mapping[str, Any] | str | Path,
    descriptor_plan: Mapping[str, Any],
    generation_count: int,
    cell_count: int,
) -> dict[str, Any]:
    """Public concise alias for the device-neutral Flat/Patch P2 compiler."""

    return compile_device_neutral_compact_summary_host_programs(
        checkpoint_projection_skeleton,
        descriptor_plan,
        generation_count,
        cell_count,
    )


def compile_flat_compact_summary_host_programs(
    checkpoint_projection_skeleton: Mapping[str, Any] | str | Path,
    descriptor_plan: Mapping[str, Any],
    generation_count: int,
    cell_count: int,
) -> dict[str, Any]:
    """Compile Flat host programs only; never inspect live post-dispatch state."""

    return _compile_device_neutral_compact_summary_host_programs(
        checkpoint_projection_skeleton,
        descriptor_plan,
        generation_count,
        cell_count,
        flat_only=True,
    )


def _validated_token_bytes(token: Mapping[str, Any], value: Any) -> bytes:
    token_type = token["token_type"]
    rule = token["validation_rule"]
    path = token["source_path"]
    if token_type in (TOKEN_INTEGER, TOKEN_PCG_UINT128, TOKEN_PYTHON_MT_WORD):
        if type(value) is not int:
            raise TypeError(f"{path} must be an int")
        if not rule["minimum"] <= value <= rule["maximum"]:
            _fail(path, f"must be in [{rule['minimum']}, {rule['maximum']}]")
        rendered_value: Any = value
    elif token_type == TOKEN_BOOLEAN:
        if type(value) is not bool:
            raise TypeError(f"{path} must be a bool")
        rendered_value = value
    elif token_type == TOKEN_TENSOR_SHAPE:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a JSON list")
        if len(value) != rule["rank"]:
            _fail(path, f"shape rank must equal {rule['rank']}")
        for index, dimension in enumerate(value):
            if type(dimension) is not int:
                raise TypeError(f"{path}[{index}] must be an int")
            if not rule["minimum_dimension"] <= dimension <= rule["maximum_dimension"]:
                _fail(path, f"shape dimension {index} is outside the bounded range")
        rendered_value = value
    elif token_type == TOKEN_SHA256_HEX:
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            _fail(path, "must be exactly 64 lowercase SHA-256 hex characters")
        rendered_value = value
    elif token_type == TOKEN_NULLABLE_FLOAT64:
        if value is not None and type(value) is not float:
            raise TypeError(f"{path} must be None or a finite Python float")
        if type(value) is float and not math.isfinite(value):
            _fail(path, "must be None or a finite Python float")
        rendered_value = value
    else:
        raise AssertionError(token_type)
    for _ in range(rule["render_pack_depth"]):
        rendered_value = _pack_concrete_value(rendered_value)
    return _canonical_literal(rendered_value).encode("ascii")


def render_compact_summary_host_template(
    template: Mapping[str, Any],
    token_values: Mapping[str, Any] | Sequence[Any],
) -> bytes:
    """Pure reference rendering of one compiled preimage after settlement."""

    if template.get("schema_version") != HOST_TEMPLATE_SCHEMA_VERSION:
        _fail("template.schema_version", "unsupported template schema")
    tokens = template.get("tokens")
    parts = template.get("parts")
    if not isinstance(tokens, list) or not isinstance(parts, list):
        raise TypeError("template tokens and parts must be lists")
    if isinstance(token_values, Mapping):
        ids = [token["token_id"] for token in tokens]
        paths = [token["source_path"] for token in tokens]
        keys = set(token_values)
        if keys == set(ids):
            values = [token_values[token_id] for token_id in ids]
        elif len(set(paths)) == len(paths) and keys == set(paths):
            values = [token_values[path] for path in paths]
        else:
            _fail("token_values", "keys must exactly match token_ids or source_paths")
    elif isinstance(token_values, Sequence) and not isinstance(
        token_values, (str, bytes, bytearray)
    ):
        values = list(token_values)
        if len(values) != len(tokens):
            _fail("token_values", "sequence length must equal template token_count")
    else:
        raise TypeError("token_values must be a mapping or sequence")
    rendered_tokens = [
        _validated_token_bytes(token, value) for token, value in zip(tokens, values)
    ]
    path_values = {
        token["source_path"]: value for token, value in zip(tokens, values)
    }
    constraints = template.get("constraints", [])
    if not isinstance(constraints, list):
        raise TypeError("template constraints must be a list")
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, Mapping):
            raise TypeError(f"template.constraints[{index}] must be a mapping")
        kind = constraint.get("kind")
        if kind == "values_equal":
            left = constraint.get("left_source_path")
            right = constraint.get("right_source_path")
            if set(constraint) != {"kind", "left_source_path", "right_source_path"}:
                _fail(f"template.constraints[{index}]", "fields are not exact")
            if path_values[left] != path_values[right]:
                _fail(str(left), f"must equal aliased authority {right}")
        elif kind == "shape_first_dimension_equals_integer":
            shape_path = constraint.get("shape_source_path")
            integer_path = constraint.get("integer_source_path")
            if set(constraint) != {
                "kind",
                "shape_source_path",
                "integer_source_path",
            }:
                _fail(f"template.constraints[{index}]", "fields are not exact")
            shape = path_values[shape_path]
            if not shape or shape[0] != path_values[integer_path]:
                _fail(str(shape_path), f"first dimension must equal {integer_path}")
        else:
            _fail(f"template.constraints[{index}]", "unsupported constraint kind")
    chunks: list[bytes] = []
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise TypeError(f"template.parts[{index}] must be a mapping")
        if part.get("kind") == "literal_utf8" and set(part) == {"kind", "value"}:
            value = part["value"]
            if type(value) is not str:
                raise TypeError(f"template.parts[{index}].value must be a string")
            chunks.append(value.encode("utf-8"))
        elif part.get("kind") == "dynamic_token" and set(part) == {
            "kind",
            "token_index",
        }:
            token_index = part["token_index"]
            if type(token_index) is not int or not 0 <= token_index < len(tokens):
                _fail(f"template.parts[{index}].token_index", "is out of range")
            chunks.append(rendered_tokens[token_index])
        else:
            _fail(f"template.parts[{index}]", "has an unsupported part shape")
    rendered = b"".join(chunks)
    if len(rendered) > template["maximum_rendered_utf8_bytes"]:
        _fail("rendered preimage", "exceeds the compiled byte bound")
    return rendered


def _resolve_pointer(value: Any, pointer: str) -> Any:
    current = value
    for raw in pointer.split("/")[1:]:
        component = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current[component]
        elif isinstance(current, list):
            current = current[int(component)]
        else:
            raise KeyError(pointer)
    return current


def retained_authority_token_values(
    template: Mapping[str, Any],
    settled_retained_authority: Mapping[str, Any],
) -> list[Any]:
    """Resolve a template's ordered token values from one settled authority row."""

    values: list[Any] = []
    for token in template["tokens"]:
        path = token["source_path"]
        if path == "/identity/active_agent_ids_shape":
            count = _resolve_pointer(settled_retained_authority, "/identity/active_agent_count")
            values.append([count])
        else:
            values.append(_resolve_pointer(settled_retained_authority, path))
    return values


def compact_summary_host_template_json_bytes(bundle: Mapping[str, Any]) -> bytes:
    """Return deterministic canonical JSON bytes for persistence or C++ loading."""

    if bundle.get("schema_version") != HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION:
        _fail("bundle.schema_version", "unsupported bundle schema")
    return _canonical_literal(bundle).encode("utf-8")


__all__ = [
    "HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION",
    "HOST_TEMPLATE_SCHEMA_VERSION",
    "SURFACE_ACTIVE_AGENT_IDS",
    "SURFACE_CANONICAL_STORAGE",
    "SURFACE_CHECKPOINT_TRAJECTORY",
    "SURFACE_NEXT_GENERATION_SOURCE",
    "SURFACE_ORDER",
    "SURFACE_POLICY_MUTATION_TAPE",
    "SURFACE_RNG",
    "TOKEN_BOOLEAN",
    "TOKEN_INTEGER",
    "TOKEN_PCG_UINT128",
    "TOKEN_PYTHON_MT_WORD",
    "TOKEN_NULLABLE_FLOAT64",
    "TOKEN_SHA256_HEX",
    "TOKEN_TENSOR_SHAPE",
    "CompactSummaryHostTemplateBlocker",
    "CompactSummaryHostTemplateError",
    "compact_summary_host_template_json_bytes",
    "compile_compact_summary_host_programs",
    "compile_compact_summary_host_templates",
    "compile_device_neutral_compact_summary_host_programs",
    "compile_flat_compact_summary_host_programs",
    "dynamic_token_placeholder",
    "render_compact_summary_host_template",
    "retained_authority_token_values",
]
