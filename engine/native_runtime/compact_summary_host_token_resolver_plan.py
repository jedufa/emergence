"""Pinned source-resolution rows for Flat compact-summary host tokens.

The compiler records how native code can obtain each dynamic host-template token;
it never records a settled value.  Rows are program-major and token-major, exactly
matching the token traversal used by ``compact_summary_host_template_wire``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .compact_summary_checkpoint_descriptor_plan import (
    PLAN_SCHEMA_VERSION,
    validate_compact_summary_checkpoint_descriptor_plan,
)
from .compact_summary_host_template import (
    HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION,
    HOST_TEMPLATE_SCHEMA_VERSION,
    MAX_COORDINATES,
    MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE,
    MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE,
    MAX_SOURCE_PATH_UTF8_BYTES,
    MAX_TOKENS_PER_TEMPLATE,
    SURFACE_ORDER,
    TOKEN_BOOLEAN,
    TOKEN_INTEGER,
    TOKEN_NULLABLE_FLOAT64,
    TOKEN_PCG_UINT128,
    TOKEN_PYTHON_MT_WORD,
    TOKEN_SHA256_HEX,
    TOKEN_TENSOR_SHAPE,
    UINT32_MAX,
    UINT64_MAX,
    UINT128_MAX,
    WIRE_FIELDS,
)


HOST_TOKEN_RESOLVER_PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_host_token_resolver_plan_v1"
)
HOST_TOKEN_RESOLVER_ABI_VERSION: Final[int] = 1
RESOLVER_PLAN_COLUMNS: Final[tuple[str, ...]] = (
    "abi_version",
    "source_kind_code",
    "primary_index",
    "secondary_code",
    "rank",
    "reserved_zero",
)
MAX_DESCRIPTOR_PLAN_JSON_BYTES: Final[int] = 16 * 1024 * 1024
MAX_PYTHON_MT_WORDS: Final[int] = 624
MAX_RESOLVER_PLAN_TOKENS: Final[int] = (
    MAX_COORDINATES * len(SURFACE_ORDER) * MAX_TOKENS_PER_TEMPLATE
)

SOURCE_KIND_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "checkpoint_trajectory_digest": 1,
        "checkpoint_trajectory_shape": 2,
        "checkpoint_canonical_digest": 3,
        "checkpoint_canonical_shape": 4,
        "checkpoint_source_manifest_digest": 5,
        "checkpoint_source_manifest_shape": 6,
        "checkpoint_stored_manifest_shape": 7,
        "checkpoint_stored_manifest_numel": 8,
        "checkpoint_stored_manifest_bytes": 9,
        "projection_top_level_integer": 10,
        "projection_metadata_integer": 11,
        "projection_metadata_boolean": 12,
        "pcg64_state": 13,
        "pcg64_inc": 14,
        "pcg64_buffer_flag": 15,
        "pcg64_buffer_value": 16,
        "python_mt_word": 17,
        "python_mt_index": 18,
        "semantic_stream_counter": 19,
        "metadata_import_digest": 20,
        "policy_energy_digest": 21,
        "policy_energy_shape": 22,
        "identity_count": 23,
        "identity_digest": 24,
        "policy_mutation_tape_cursor": 25,
        "policy_mutation_tape_digest": 26,
        "policy_mutation_tape_shape": 27,
        "python_gauss_cache": 28,
    }
)
SECONDARY_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {"none": 0, "raw": 1, "canonical": 2}
)
PROJECTION_TOP_LEVEL_FIELD_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        name: index + 1
        for index, name in enumerate(
            (
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
        )
    }
)
PROJECTION_METADATA_FIELD_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        name: index + 1
        for index, name in enumerate(
            (
                "generation",
                "alive_count",
                "event_tape_active_count",
                "event_tape_cursor",
                "event_tape_inactive_tail_count",
                "native_policy_cache_served",
                "native_runtime_rng_state",
                "policy_energy_observation_finite_live_count",
                "policy_energy_observation_live_count",
                "population_next_id",
                "typed_lifecycle_event_active_count",
                "typed_lifecycle_event_next_sequence",
                "configured_steps_per_generation",
            )
        )
    }
)
PCG64_AUTHORITY_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        name: index + 1
        for index, name in enumerate(
            (
                "native_runtime_numpy_birth_spawn_env_state",
                "native_runtime_numpy_env_state",
                "native_runtime_numpy_policy_state",
                "native_runtime_numpy_population_state",
                "native_runtime_numpy_runner_state",
            )
        )
    }
)
SEMANTIC_STREAM_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        name: index + 1
        for index, name in enumerate(
            f"{semantic}/{backend}"
            for semantic in ("metrics", "mutation", "policy", "stress")
            for backend in ("py", "np", "torch_cpu", "torch_cuda/cuda:0")
        )
    }
)

CODEBOOKS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "source_kind_codes": SOURCE_KIND_CODES,
        "secondary_codes": SECONDARY_CODES,
        "projection_top_level_field_codes": PROJECTION_TOP_LEVEL_FIELD_CODES,
        "projection_metadata_field_codes": PROJECTION_METADATA_FIELD_CODES,
        "pcg64_authority_codes": PCG64_AUTHORITY_CODES,
        "semantic_stream_codes": SEMANTIC_STREAM_CODES,
        "descriptor_primary_index": "descriptor_index",
        "python_mt_word_primary_index": "zero_based_word_index",
    }
)

_BUNDLE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "representation",
    "ordering",
    "surface_order",
    "generation_count",
    "cell_count",
    "coordinate_count",
    "template_count",
    "bounds",
    "templates",
)
_BOUND_FIELDS: Final[tuple[str, ...]] = (
    "maximum_coordinates",
    "maximum_tokens_per_template",
    "maximum_literal_utf8_bytes_per_template",
    "maximum_rendered_utf8_bytes_per_template",
)
_TEMPLATE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "surface",
    "wire_field",
    "generation",
    "cell",
    "encoding",
    "literal_utf8_bytes",
    "token_count",
    "maximum_rendered_utf8_bytes",
    "tokens",
    "parts",
    "constraints",
    "template_index",
)
_TOKEN_FIELDS: Final[tuple[str, ...]] = (
    "token_index",
    "token_id",
    "token_type",
    "source_path",
    "generation",
    "cell",
    "validation_rule",
    "maximum_utf8_bytes",
)
_MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "source_schema_version",
    "descriptor_plan_schema_version",
    "abi_version",
    "representation",
    "ordering",
    "columns",
    "dtype",
    "shape",
    "token_count",
    "unique_source_path_count",
    "template_count",
    "generation_count",
    "cell_count",
    "descriptor_count",
    "source_kind_counts",
    "pinned_memory",
    "test_only_unpinned_fallback",
    "limitations",
)


class CompactSummaryHostTokenResolverPlanError(ValueError):
    """The bundle, descriptor plan, or resolver tensor violates the frozen ABI."""


def _fail(path: str, message: str) -> None:
    raise CompactSummaryHostTokenResolverPlanError(f"{path}: {message}")


def _exact_mapping(value: Any, fields: tuple[str, ...], *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(path, "must be a string-keyed mapping")
    missing = sorted(set(fields) - set(value))
    extra = sorted(set(value) - set(fields))
    if missing or extra:
        _fail(path, f"fields mismatch: missing={missing} extra={extra}")
    return value


def _exact_list(value: Any, *, path: str) -> list[Any]:
    if type(value) is not list:
        _fail(path, "must be a list")
    return value


def _uint(value: Any, *, path: str, maximum: int = UINT64_MAX) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        _fail(path, f"must be an integer in [0, {maximum}]")
    return value


def _pointer(base: str, component: str | int) -> str:
    escaped = str(component).replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}"


def _load_descriptor_plan(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        plan = value
    elif isinstance(value, (str, Path)):
        path = Path(value)
        try:
            size = path.stat().st_size
        except OSError as exc:
            _fail("descriptor_plan", f"cannot stat JSON file: {exc}")
        if size > MAX_DESCRIPTOR_PLAN_JSON_BYTES:
            _fail("descriptor_plan", "JSON file exceeds the bounded input size")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _fail("descriptor_plan", f"cannot load JSON file: {exc}")
        if not isinstance(parsed, Mapping):
            _fail("descriptor_plan", "JSON root must be a mapping")
        plan = parsed
    else:
        raise TypeError("descriptor_plan must be a mapping or JSON path")
    try:
        validate_compact_summary_checkpoint_descriptor_plan(plan)
    except (TypeError, ValueError) as exc:
        _fail("descriptor_plan", f"frozen plan drift: {exc}")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        _fail("descriptor_plan.schema_version", "unsupported descriptor plan schema")
    return plan


@dataclass(frozen=True, slots=True)
class _SourceSpec:
    kind: int
    primary: int
    secondary: int
    rank: int
    token_type: str

    def row(self) -> list[int]:
        return [
            HOST_TOKEN_RESOLVER_ABI_VERSION,
            self.kind,
            self.primary,
            self.secondary,
            self.rank,
            0,
        ]


def _spec(
    kind: str,
    *,
    primary: int = 0,
    output: str = "none",
    rank: int = 0,
    token_type: str,
) -> _SourceSpec:
    return _SourceSpec(
        SOURCE_KIND_CODES[kind],
        primary,
        SECONDARY_CODES[output],
        rank,
        token_type,
    )


def _add_spec(specs: dict[str, _SourceSpec], path: str, spec: _SourceSpec) -> None:
    if path in specs and specs[path] != spec:
        _fail(path, "source path has contradictory resolver declarations")
    specs[path] = spec


def _descriptor_output(row: Mapping[str, Any], field: str) -> str:
    inclusion = row.get("digest_inclusion")
    if not isinstance(inclusion, Mapping):
        _fail("descriptor_plan.descriptors", "digest_inclusion must be a mapping")
    output = inclusion.get(field)
    if output not in SECONDARY_CODES or output == "none":
        _fail(
            f"descriptor_plan.descriptors[{row.get('descriptor_index')}].digest_inclusion.{field}",
            "must select raw or canonical output",
        )
    return str(output)


def _descriptor_rank(row: Mapping[str, Any], output: str) -> int:
    return _uint(
        row.get(f"{output}_rank"),
        path=f"descriptor_plan.descriptors[{row.get('descriptor_index')}].{output}_rank",
        maximum=UINT32_MAX,
    )


def _source_specs(plan: Mapping[str, Any]) -> Mapping[str, _SourceSpec]:
    raw_rows = plan.get("descriptors")
    if type(raw_rows) is not list:
        _fail("descriptor_plan.descriptors", "must be a list")
    rows: list[Mapping[str, Any]] = []
    names: dict[str, Mapping[str, Any]] = {}
    for expected_index, value in enumerate(raw_rows):
        if not isinstance(value, Mapping):
            _fail(f"descriptor_plan.descriptors[{expected_index}]", "must be a mapping")
        if value.get("descriptor_index") != expected_index:
            _fail(f"descriptor_plan.descriptors[{expected_index}]", "descriptor_index drift")
        name = value.get("tensor_name")
        if type(name) is not str or not name or name in names:
            _fail(f"descriptor_plan.descriptors[{expected_index}].tensor_name", "name drift")
        rows.append(value)
        names[name] = value

    checkpoint = [row for row in rows if row.get("source_scope") == "checkpoint_projection"]
    policy = [row for row in rows if row.get("source_scope") == "policy_mutation_tape"]
    if len(checkpoint) + len(policy) != len(rows):
        _fail("descriptor_plan.descriptors", "unsupported source_scope")

    specs: dict[str, _SourceSpec] = {}
    for row in checkpoint:
        name = str(row["tensor_name"])
        descriptor_index = int(row["descriptor_index"])
        output = _descriptor_output(row, "source_projection_output")
        rank = _descriptor_rank(row, output)
        source_base = _pointer("/checkpoint/tensor_manifest", name)
        _add_spec(
            specs,
            _pointer(source_base, "sha256"),
            _spec(
                "checkpoint_source_manifest_digest",
                primary=descriptor_index,
                output=output,
                token_type=TOKEN_SHA256_HEX,
            ),
        )
        _add_spec(
            specs,
            _pointer(source_base, "shape"),
            _spec(
                "checkpoint_source_manifest_shape",
                primary=descriptor_index,
                output=output,
                rank=rank,
                token_type=TOKEN_TENSOR_SHAPE,
            ),
        )
        stored_base = _pointer(
            "/checkpoint/projection_fields/tensor_manifest", name
        )
        for leaf, kind, leaf_rank, token_type in (
            ("shape", "checkpoint_stored_manifest_shape", rank, TOKEN_TENSOR_SHAPE),
            ("numel", "checkpoint_stored_manifest_numel", 0, TOKEN_INTEGER),
            ("bytes", "checkpoint_stored_manifest_bytes", 0, TOKEN_INTEGER),
        ):
            _add_spec(
                specs,
                _pointer(stored_base, leaf),
                _spec(
                    kind,
                    primary=descriptor_index,
                    output=output,
                    rank=leaf_rank,
                    token_type=token_type,
                ),
            )

    for authority, inclusion_name, digest_kind, shape_kind in (
        (
            "trajectory_state_fields",
            "trajectory_state",
            "checkpoint_trajectory_digest",
            "checkpoint_trajectory_shape",
        ),
        (
            "canonical_checkpoint_storage_fields",
            "canonical_checkpoint_storage",
            "checkpoint_canonical_digest",
            "checkpoint_canonical_shape",
        ),
    ):
        layer_rows = [
            row
            for row in checkpoint
            if inclusion_name in row["digest_inclusion"]["checkpoint_layers"]
        ]
        for layer_index, row in enumerate(layer_rows):
            output = _descriptor_output(row, "checkpoint_layers_output")
            rank = _descriptor_rank(row, output)
            base = _pointer(_pointer("/checkpoint", authority), layer_index)
            descriptor_index = int(row["descriptor_index"])
            _add_spec(
                specs,
                _pointer(base, "sha256"),
                _spec(
                    digest_kind,
                    primary=descriptor_index,
                    output=output,
                    token_type=TOKEN_SHA256_HEX,
                ),
            )
            _add_spec(
                specs,
                _pointer(base, "shape"),
                _spec(
                    shape_kind,
                    primary=descriptor_index,
                    output=output,
                    rank=rank,
                    token_type=TOKEN_TENSOR_SHAPE,
                ),
            )

    projection_base = "/checkpoint/projection_fields"
    for name, code in PROJECTION_TOP_LEVEL_FIELD_CODES.items():
        _add_spec(
            specs,
            _pointer(projection_base, name),
            _spec("projection_top_level_integer", primary=code, token_type=TOKEN_INTEGER),
        )
    metadata_base = _pointer(projection_base, "metadata")
    for name, code in PROJECTION_METADATA_FIELD_CODES.items():
        kind = (
            "projection_metadata_boolean"
            if name == "native_policy_cache_served"
            else "projection_metadata_integer"
        )
        token_type = TOKEN_BOOLEAN if kind.endswith("boolean") else TOKEN_INTEGER
        _add_spec(
            specs,
            _pointer(metadata_base, name),
            _spec(kind, primary=code, token_type=token_type),
        )

    for name, authority_code in PCG64_AUTHORITY_CODES.items():
        base = _pointer(metadata_base, name)
        for suffix, kind, token_type in (
            ("state/state", "pcg64_state", TOKEN_PCG_UINT128),
            ("state/inc", "pcg64_inc", TOKEN_PCG_UINT128),
            ("has_uint32", "pcg64_buffer_flag", TOKEN_INTEGER),
            ("uinteger", "pcg64_buffer_value", TOKEN_INTEGER),
        ):
            _add_spec(
                specs,
                f"{base}/{suffix}",
                _spec(kind, primary=authority_code, token_type=token_type),
            )

    python_base = _pointer(metadata_base, "native_runtime_python_random_state")
    for word_index in range(MAX_PYTHON_MT_WORDS):
        _add_spec(
            specs,
            _pointer(_pointer(python_base, "state"), word_index),
            _spec("python_mt_word", primary=word_index, token_type=TOKEN_PYTHON_MT_WORD),
        )
    _add_spec(
        specs,
        _pointer(python_base, "index"),
        _spec("python_mt_index", token_type=TOKEN_INTEGER),
    )
    for gauss_path in (_pointer(python_base, "gauss_next"), _pointer(python_base, 2)):
        _add_spec(
            specs,
            gauss_path,
            _spec("python_gauss_cache", token_type=TOKEN_NULLABLE_FLOAT64),
        )

    streams_base = _pointer(metadata_base, "native_runtime_rng_manager_stream_indices")
    for name, stream_code in SEMANTIC_STREAM_CODES.items():
        _add_spec(
            specs,
            f"{streams_base}/{name}",
            _spec("semantic_stream_counter", primary=stream_code, token_type=TOKEN_INTEGER),
        )

    _add_spec(
        specs,
        _pointer(metadata_base, "import_digest"),
        _spec("metadata_import_digest", token_type=TOKEN_SHA256_HEX),
    )
    try:
        policy_energy = names["policy_energy_observation"]
        identity = names["agent_ids_by_alive_order"]
    except KeyError as exc:
        _fail("descriptor_plan.descriptors", f"missing required descriptor {exc.args[0]!r}")
    energy_output = _descriptor_output(policy_energy, "source_projection_output")
    energy_index = int(policy_energy["descriptor_index"])
    _add_spec(
        specs,
        _pointer(metadata_base, "policy_energy_observation_tensor_sha256"),
        _spec(
            "policy_energy_digest",
            primary=energy_index,
            output=energy_output,
            token_type=TOKEN_SHA256_HEX,
        ),
    )
    _add_spec(
        specs,
        _pointer(metadata_base, "policy_energy_observation_shape"),
        _spec(
            "policy_energy_shape",
            primary=energy_index,
            output=energy_output,
            rank=_descriptor_rank(policy_energy, energy_output),
            token_type=TOKEN_TENSOR_SHAPE,
        ),
    )

    identity_output = _descriptor_output(identity, "active_agent_ids_output")
    identity_index = int(identity["descriptor_index"])
    _add_spec(
        specs,
        "/identity/active_agent_count",
        _spec(
            "identity_count",
            primary=identity_index,
            output=identity_output,
            token_type=TOKEN_INTEGER,
        ),
    )
    _add_spec(
        specs,
        "/identity/tensor_sha256",
        _spec(
            "identity_digest",
            primary=identity_index,
            output=identity_output,
            token_type=TOKEN_SHA256_HEX,
        ),
    )

    _add_spec(
        specs,
        "/policy_mutation_tape/cursor",
        _spec("policy_mutation_tape_cursor", token_type=TOKEN_INTEGER),
    )
    for row in policy:
        name = str(row["tensor_name"])
        descriptor_index = int(row["descriptor_index"])
        output = _descriptor_output(row, "policy_mutation_tape_output")
        _add_spec(
            specs,
            _pointer("/policy_mutation_tape/tensor_sha256", name),
            _spec(
                "policy_mutation_tape_digest",
                primary=descriptor_index,
                output=output,
                token_type=TOKEN_SHA256_HEX,
            ),
        )
        _add_spec(
            specs,
            _pointer("/policy_mutation_tape/consumed_shapes", name),
            _spec(
                "policy_mutation_tape_shape",
                primary=descriptor_index,
                output=output,
                rank=_descriptor_rank(row, output),
                token_type=TOKEN_TENSOR_SHAPE,
            ),
        )
    return MappingProxyType(specs)


def _token_rank(token: Mapping[str, Any], *, path: str) -> int:
    token_type = token["token_type"]
    rule = token["validation_rule"]
    if not isinstance(rule, Mapping) or any(type(key) is not str for key in rule):
        _fail(f"{path}.validation_rule", "must be a string-keyed mapping")
    depth = rule.get("render_pack_depth")
    if type(depth) is not int or not 0 <= depth <= 2:
        _fail(f"{path}.validation_rule.render_pack_depth", "must be 0, 1, or 2")
    if token_type == TOKEN_INTEGER:
        expected = {
            "kind": "integer_range",
            "minimum": 0,
            "maximum": UINT64_MAX,
            "render_pack_depth": depth,
        }
        rank = 0
    elif token_type == TOKEN_BOOLEAN:
        expected = {"kind": "exact_boolean", "render_pack_depth": depth}
        rank = 0
    elif token_type == TOKEN_PCG_UINT128:
        expected = {
            "kind": "integer_range",
            "minimum": 0,
            "maximum": UINT128_MAX,
            "render_pack_depth": depth,
        }
        rank = 0
    elif token_type == TOKEN_PYTHON_MT_WORD:
        expected = {
            "kind": "integer_range",
            "minimum": 0,
            "maximum": UINT32_MAX,
            "render_pack_depth": depth,
        }
        rank = 0
    elif token_type == TOKEN_TENSOR_SHAPE:
        rank = _uint(rule.get("rank"), path=f"{path}.validation_rule.rank", maximum=UINT32_MAX)
        expected = {
            "kind": "tensor_shape",
            "rank": rank,
            "minimum_dimension": 0,
            "maximum_dimension": UINT64_MAX,
            "render_pack_depth": depth,
        }
    elif token_type == TOKEN_SHA256_HEX:
        expected = {
            "kind": "lowercase_sha256_hex",
            "length": 64,
            "render_pack_depth": depth,
        }
        rank = 0
    elif token_type == TOKEN_NULLABLE_FLOAT64:
        expected = {
            "kind": "finite_optional_float64",
            "render_pack_depth": depth,
        }
        rank = 0
    else:
        _fail(f"{path}.token_type", f"unsupported token type {token_type!r}")
    if dict(rule) != expected:
        _fail(f"{path}.validation_rule", "does not exactly match token type")
    return rank


def _validate_bundle_and_compile_rows(
    bundle: Mapping[str, Any],
    specs: Mapping[str, _SourceSpec],
) -> tuple[list[list[int]], set[str]]:
    source = _exact_mapping(bundle, _BUNDLE_FIELDS, path="bundle")
    if source["schema_version"] != HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION:
        _fail("bundle.schema_version", "unsupported compiled bundle schema")
    if source["representation"] != "immutable_utf8_segments_plus_bounded_dynamic_tokens":
        _fail("bundle.representation", "unsupported compiled representation")
    if source["ordering"] != "generation_major_cell_major_surface_order":
        _fail("bundle.ordering", "unsupported template ordering")
    if _exact_list(source["surface_order"], path="bundle.surface_order") != list(SURFACE_ORDER):
        _fail("bundle.surface_order", "surface order drift")
    generations = _uint(
        source["generation_count"],
        path="bundle.generation_count",
        maximum=UINT32_MAX,
    )
    cells = _uint(source["cell_count"], path="bundle.cell_count", maximum=UINT32_MAX)
    if generations == 0 or cells == 0 or generations * cells > MAX_COORDINATES:
        _fail("bundle", "coordinate matrix is outside the bounded contract")
    coordinate_count = generations * cells
    if source["coordinate_count"] != coordinate_count:
        _fail("bundle.coordinate_count", "coordinate count drift")
    template_count = coordinate_count * len(SURFACE_ORDER)
    if source["template_count"] != template_count:
        _fail("bundle.template_count", "template count drift")
    bounds = _exact_mapping(source["bounds"], _BOUND_FIELDS, path="bundle.bounds")
    if dict(bounds) != {
        "maximum_coordinates": MAX_COORDINATES,
        "maximum_tokens_per_template": MAX_TOKENS_PER_TEMPLATE,
        "maximum_literal_utf8_bytes_per_template": MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE,
        "maximum_rendered_utf8_bytes_per_template": MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE,
    }:
        _fail("bundle.bounds", "compiled bounds drift")
    templates = _exact_list(source["templates"], path="bundle.templates")
    if len(templates) != template_count:
        _fail("bundle.templates", "template count mismatch")

    rows: list[list[int]] = []
    paths: set[str] = set()
    repeated: dict[str, _SourceSpec] = {}
    for template_index, value in enumerate(templates):
        path = f"bundle.templates[{template_index}]"
        template = _exact_mapping(value, _TEMPLATE_FIELDS, path=path)
        coordinate_index, surface_index = divmod(template_index, len(SURFACE_ORDER))
        generation, cell = divmod(coordinate_index, cells)
        surface = SURFACE_ORDER[surface_index]
        expected_template = {
            "schema_version": HOST_TEMPLATE_SCHEMA_VERSION,
            "surface": surface,
            "wire_field": WIRE_FIELDS[surface],
            "generation": generation,
            "cell": cell,
            "encoding": "utf-8",
            "template_index": template_index,
        }
        for field, expected in expected_template.items():
            if template[field] != expected or type(template[field]) is not type(expected):
                _fail(f"{path}.{field}", f"must equal {expected!r}")
        _uint(template["literal_utf8_bytes"], path=f"{path}.literal_utf8_bytes")
        _uint(
            template["maximum_rendered_utf8_bytes"],
            path=f"{path}.maximum_rendered_utf8_bytes",
        )
        token_values = _exact_list(template["tokens"], path=f"{path}.tokens")
        token_count = _uint(
            template["token_count"], path=f"{path}.token_count", maximum=UINT32_MAX
        )
        if token_count != len(token_values) or token_count > MAX_TOKENS_PER_TEMPLATE:
            _fail(f"{path}.token_count", "token count mismatch or bound exceeded")
        local_paths: set[str] = set()
        for token_index, token_value in enumerate(token_values):
            token_path = f"{path}.tokens[{token_index}]"
            token = _exact_mapping(token_value, _TOKEN_FIELDS, path=token_path)
            expected_token = {
                "token_index": token_index,
                "token_id": f"g{generation}.c{cell}.{surface}.t{token_index}",
                "generation": generation,
                "cell": cell,
            }
            for field, expected in expected_token.items():
                if token[field] != expected or type(token[field]) is not type(expected):
                    _fail(f"{token_path}.{field}", f"must equal {expected!r}")
            source_path = token["source_path"]
            if type(source_path) is not str or not source_path.startswith("/"):
                _fail(f"{token_path}.source_path", "must be an absolute JSON Pointer")
            try:
                source_bytes = source_path.encode("utf-8")
            except UnicodeEncodeError as exc:
                _fail(f"{token_path}.source_path", f"must be valid UTF-8: {exc.reason}")
            if len(source_bytes) > MAX_SOURCE_PATH_UTF8_BYTES:
                _fail(f"{token_path}.source_path", "exceeds the bounded path size")
            if source_path in local_paths:
                _fail(f"{token_path}.source_path", "must be unique per program")
            local_paths.add(source_path)
            spec = specs.get(source_path)
            if spec is None:
                _fail(f"{token_path}.source_path", f"unknown Flat g2 source path {source_path!r}")
            rank = _token_rank(token, path=token_path)
            if token["token_type"] != spec.token_type or rank != spec.rank:
                _fail(token_path, "source path token type or rank contradicts resolver plan")
            _uint(token["maximum_utf8_bytes"], path=f"{token_path}.maximum_utf8_bytes")
            previous = repeated.setdefault(source_path, spec)
            if previous != spec:
                _fail(f"{token_path}.source_path", "repeated source path classification drift")
            rows.append(spec.row())
            paths.add(source_path)
        parts = _exact_list(template["parts"], path=f"{path}.parts")
        seen_tokens: set[int] = set()
        for part_index, part in enumerate(parts):
            part_path = f"{path}.parts[{part_index}]"
            if not isinstance(part, Mapping):
                _fail(part_path, "must be a mapping")
            if part.get("kind") == "dynamic_token" and set(part) == {"kind", "token_index"}:
                index = _uint(
                    part["token_index"],
                    path=f"{part_path}.token_index",
                    maximum=UINT32_MAX,
                )
                if index >= token_count:
                    _fail(f"{part_path}.token_index", "is outside token rows")
                seen_tokens.add(index)
            elif part.get("kind") == "literal_utf8" and set(part) == {"kind", "value"}:
                if type(part["value"]) is not str:
                    _fail(f"{part_path}.value", "must be a string")
            else:
                _fail(part_path, "unsupported part shape")
        if seen_tokens != set(range(token_count)):
            _fail(f"{path}.parts", "must reference every token row")
        _exact_list(template["constraints"], path=f"{path}.constraints")
    if not rows or len(rows) > MAX_RESOLVER_PLAN_TOKENS:
        _fail("bundle.tokens", "total token count is outside the bounded contract")
    return rows, paths


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def _immutable_manifest(
    *,
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
    rows: list[list[int]],
    unique_path_count: int,
    pinned: bool,
    fallback: bool,
) -> Mapping[str, Any]:
    reverse = {code: name for name, code in SOURCE_KIND_CODES.items()}
    counts = Counter(reverse[row[1]] for row in rows)
    return MappingProxyType(
        {
            "schema_version": HOST_TOKEN_RESOLVER_PLAN_SCHEMA_VERSION,
            "source_schema_version": HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION,
            "descriptor_plan_schema_version": PLAN_SCHEMA_VERSION,
            "abi_version": HOST_TOKEN_RESOLVER_ABI_VERSION,
            "representation": "descriptor_indexed_flat_g2_token_source_resolver_rows",
            "ordering": "program_major_token_order_identical_to_host_template_wire_tokens",
            "columns": RESOLVER_PLAN_COLUMNS,
            "dtype": "uint64",
            "shape": (len(rows), len(RESOLVER_PLAN_COLUMNS)),
            "token_count": len(rows),
            "unique_source_path_count": unique_path_count,
            "template_count": bundle["template_count"],
            "generation_count": bundle["generation_count"],
            "cell_count": bundle["cell_count"],
            "descriptor_count": len(plan["descriptors"]),
            "source_kind_counts": MappingProxyType(
                {name: counts.get(name, 0) for name in SOURCE_KIND_CODES}
            ),
            "pinned_memory": pinned,
            "test_only_unpinned_fallback": fallback,
            "limitations": (
                "accepted Flat g2 host-template source paths only",
                "contains resolver metadata and no settled token values",
                "shape values remain descriptor/source-tensor lookups after D2H",
                "Python MT mapping form is bounded to 624 words",
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class CompactSummaryHostTokenResolverPlan:
    """Frozen wrapper around one ordered resolver tensor and immutable metadata."""

    resolver_plan: Any
    manifest: Mapping[str, Any]
    codebooks: Mapping[str, Any]

    @property
    def ordered_tensors(self) -> tuple[Any]:
        return (self.resolver_plan,)

    @property
    def tensor_tuple(self) -> tuple[Any]:
        return self.ordered_tensors

    def as_tuple(self) -> tuple[Any]:
        return self.ordered_tensors


def compile_flat_compact_summary_host_token_resolver_plan(
    bundle: Mapping[str, Any],
    descriptor_plan: Mapping[str, Any] | str | Path,
    torch: Any,
    *,
    allow_unpinned_test_fallback: bool = False,
) -> CompactSummaryHostTokenResolverPlan:
    """Compile one exact CPU uint64[token_count, 6] Flat g2 resolver tensor.

    Pinned allocation is mandatory.  The fallback is test-only, is attempted only
    after pinned allocation fails, and is rejected while CUDA is available.
    """

    if type(allow_unpinned_test_fallback) is not bool:
        _fail("allow_unpinned_test_fallback", "must be a bool")
    plan = _load_descriptor_plan(descriptor_plan)
    specs = _source_specs(plan)
    rows, paths = _validate_bundle_and_compile_rows(bundle, specs)
    used_fallback = False
    try:
        torch.empty((1,), dtype=torch.uint8, device="cpu", pin_memory=True)
    except RuntimeError as exc:
        if not allow_unpinned_test_fallback or _cuda_available(torch):
            _fail("resolver_plan.pin_memory", f"pinned CPU allocation failed: {exc}")
        used_fallback = True
    try:
        tensor = torch.tensor(
            rows,
            dtype=torch.uint64,
            device="cpu",
            pin_memory=not used_fallback,
        )
    except RuntimeError as exc:
        _fail("resolver_plan.allocation", f"CPU tensor allocation failed: {exc}")
    manifest = _immutable_manifest(
        bundle=bundle,
        plan=plan,
        rows=rows,
        unique_path_count=len(paths),
        pinned=not used_fallback,
        fallback=used_fallback,
    )
    compiled = CompactSummaryHostTokenResolverPlan(tensor, manifest, CODEBOOKS)
    validate_compact_summary_host_token_resolver_plan(
        compiled,
        bundle,
        plan,
        torch,
    )
    return compiled


def validate_compact_summary_host_token_resolver_plan(
    compiled: CompactSummaryHostTokenResolverPlan,
    bundle: Mapping[str, Any],
    descriptor_plan: Mapping[str, Any] | str | Path,
    torch: Any,
) -> None:
    """Recompute and validate row order, codebooks, manifest, and tensor storage."""

    if not isinstance(compiled, CompactSummaryHostTokenResolverPlan):
        _fail("resolver_plan", "must use CompactSummaryHostTokenResolverPlan")
    plan = _load_descriptor_plan(descriptor_plan)
    rows, paths = _validate_bundle_and_compile_rows(bundle, _source_specs(plan))
    tensor = compiled.resolver_plan
    expected_shape = (len(rows), len(RESOLVER_PLAN_COLUMNS))
    if (
        getattr(tensor, "dtype", None) != torch.uint64
        or getattr(getattr(tensor, "device", None), "type", None) != "cpu"
        or tuple(getattr(tensor, "shape", ())) != expected_shape
        or not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or bool(tensor.requires_grad)
    ):
        _fail("resolver_plan.tensor", "dtype, shape, device, or storage contract drift")
    manifest = _exact_mapping(compiled.manifest, _MANIFEST_FIELDS, path="resolver_plan.manifest")
    pinned = manifest["pinned_memory"] is True
    fallback = manifest["test_only_unpinned_fallback"] is True
    if pinned == fallback:
        _fail("resolver_plan.manifest", "must declare exactly one allocation mode")
    if pinned and not tensor.is_pinned():
        _fail("resolver_plan.tensor", "declared pinned tensor is not pinned")
    if fallback and (tensor.is_pinned() or _cuda_available(torch)):
        _fail("resolver_plan.tensor", "test-only fallback allocation is not admissible")
    expected_manifest = _immutable_manifest(
        bundle=bundle,
        plan=plan,
        rows=rows,
        unique_path_count=len(paths),
        pinned=pinned,
        fallback=fallback,
    )
    if compiled.manifest != expected_manifest:
        _fail("resolver_plan.manifest", "manifest drift")
    if compiled.codebooks != CODEBOOKS:
        _fail("resolver_plan.codebooks", "codebook drift")
    if tensor.tolist() != rows:
        _fail("resolver_plan.tensor", "row content or ordering drift")


__all__ = [
    "CODEBOOKS",
    "HOST_TOKEN_RESOLVER_ABI_VERSION",
    "HOST_TOKEN_RESOLVER_PLAN_SCHEMA_VERSION",
    "MAX_PYTHON_MT_WORDS",
    "PCG64_AUTHORITY_CODES",
    "PROJECTION_METADATA_FIELD_CODES",
    "PROJECTION_TOP_LEVEL_FIELD_CODES",
    "RESOLVER_PLAN_COLUMNS",
    "SECONDARY_CODES",
    "SEMANTIC_STREAM_CODES",
    "SOURCE_KIND_CODES",
    "CompactSummaryHostTokenResolverPlan",
    "CompactSummaryHostTokenResolverPlanError",
    "compile_flat_compact_summary_host_token_resolver_plan",
    "validate_compact_summary_host_token_resolver_plan",
]
