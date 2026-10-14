"""Pointer-free pinned tensor wire form for compiled host templates.

The returned tensors contain only integer rows and one byte pool.  Native code
can therefore rebind byte spans and row slices immediately before dispatch
without retaining pointers into Python objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

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


HOST_TEMPLATE_WIRE_SCHEMA_VERSION: Final[str] = (
    "eiel_compact_summary_host_template_wire_v1"
)
HOST_TEMPLATE_RENDERER_ABI_VERSION: Final[int] = 1

PROGRAM_COLUMNS: Final[tuple[str, ...]] = (
    "abi_version",
    "encoding_code",
    "surface_code",
    "generation",
    "cell",
    "token_offset",
    "token_count",
    "part_offset",
    "part_count",
    "constraint_offset",
    "constraint_count",
    "literal_utf8_bytes",
    "maximum_rendered_utf8_bytes",
)
TOKEN_COLUMNS: Final[tuple[str, ...]] = (
    "token_index",
    "token_type_code",
    "validation_kind_code",
    "rank",
    "render_pack_depth",
    "maximum_utf8_bytes",
    "source_path_pool_offset",
    "source_path_utf8_bytes",
    "rule_parameter",
    "minimum_low",
    "minimum_high",
    "maximum_low",
    "maximum_high",
)
PART_COLUMNS: Final[tuple[str, ...]] = (
    "kind_code",
    "token_index",
    "literal_pool_offset",
    "literal_utf8_bytes",
    "reserved_zero",
)
CONSTRAINT_COLUMNS: Final[tuple[str, ...]] = (
    "kind_code",
    "left_path_pool_offset",
    "left_path_utf8_bytes",
    "right_path_pool_offset",
    "right_path_utf8_bytes",
)
TENSOR_ORDER: Final[tuple[str, ...]] = (
    "programs",
    "tokens",
    "parts",
    "constraints",
    "byte_pool",
)
TENSOR_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "programs": PROGRAM_COLUMNS,
        "tokens": TOKEN_COLUMNS,
        "parts": PART_COLUMNS,
        "constraints": CONSTRAINT_COLUMNS,
        "byte_pool": ("byte",),
    }
)

_SURFACE_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {surface: index + 1 for index, surface in enumerate(SURFACE_ORDER)}
)
_TOKEN_TYPE_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        TOKEN_INTEGER: 1,
        TOKEN_BOOLEAN: 2,
        TOKEN_PCG_UINT128: 3,
        TOKEN_PYTHON_MT_WORD: 4,
        TOKEN_TENSOR_SHAPE: 5,
        TOKEN_SHA256_HEX: 6,
        TOKEN_NULLABLE_FLOAT64: 7,
    }
)
_VALIDATION_KIND_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "integer_range": 1,
        "exact_boolean": 2,
        "tensor_shape": 3,
        "lowercase_sha256_hex": 4,
        "finite_optional_float64": 5,
    }
)
_PART_KIND_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {"literal_utf8": 1, "dynamic_token": 2}
)
_CONSTRAINT_KIND_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "values_equal": 1,
        "shape_first_dimension_equals_integer": 2,
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
    "abi_version",
    "representation",
    "ordering",
    "surface_order",
    "generation_count",
    "cell_count",
    "coordinate_count",
    "template_count",
    "bounds",
    "tensor_order",
    "tensor_columns",
    "tensor_dtypes",
    "tensor_shapes",
    "byte_pool_traversal",
    "pinned_memory",
    "test_only_unpinned_fallback",
)


class CompactSummaryHostTemplateWireError(ValueError):
    """A compiled bundle or lowered tensor violates the frozen wire ABI."""


def _fail(path: str, message: str) -> None:
    raise CompactSummaryHostTemplateWireError(f"{path}: {message}")


def _exact_mapping(
    value: Any,
    fields: tuple[str, ...],
    *,
    path: str,
) -> Mapping[str, Any]:
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


def _uint(value: Any, *, bits: int, path: str) -> int:
    maximum = UINT32_MAX if bits == 32 else UINT64_MAX
    if type(value) is not int or not 0 <= value <= maximum:
        _fail(path, f"must be a uint{bits}")
    return value


def _uint128(value: Any, *, path: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT128_MAX:
        _fail(path, "must be a uint128")
    return value


def _checked_add(left: int, right: int, *, path: str) -> int:
    if right < 0 or left > UINT64_MAX - right:
        _fail(path, "overflows uint64")
    return left + right


def _utf8(value: Any, *, path: str, source_path: bool = False) -> bytes:
    if type(value) is not str:
        _fail(path, "must be a string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        _fail(path, f"is not valid UTF-8: {exc.reason}")
    if source_path:
        if not encoded or encoded[0] != ord("/"):
            _fail(path, "must be a non-empty absolute JSON Pointer")
        if len(encoded) > MAX_SOURCE_PATH_UTF8_BYTES:
            _fail(path, "exceeds the source-path UTF-8 bound")
        _uint(len(encoded), bits=32, path=f"{path}.utf8_bytes")
    return encoded


def _split_uint128(value: int) -> tuple[int, int]:
    return value & UINT64_MAX, value >> 64


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _packed_representative(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = [
            (_packed_representative(key), _packed_representative(child))
            for key, child in value.items()
        ]
        items.sort(key=lambda row: _canonical_json(row[0]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_packed_representative(child) for child in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_packed_representative(child) for child in value],
        }
    return {"kind": type(value).__name__, "value": value}


def _maximum_token_utf8_bytes(token_type: str, rank: int, depth: int) -> int:
    if token_type == TOKEN_TENSOR_SHAPE:
        def encoded_shape_size(sample_rank: int) -> int:
            sample: Any = [UINT64_MAX] * sample_rank
            for _ in range(depth):
                sample = _packed_representative(sample)
            return len(_canonical_json(sample).encode("ascii"))

        empty_size = encoded_shape_size(0)
        if rank == 0:
            return empty_size
        one_size = encoded_shape_size(1)
        return empty_size + rank * (one_size - empty_size) + rank - 1
    if token_type == TOKEN_INTEGER:
        value: Any = UINT64_MAX
    elif token_type == TOKEN_BOOLEAN:
        value = False
    elif token_type == TOKEN_PCG_UINT128:
        value = UINT128_MAX
    elif token_type == TOKEN_PYTHON_MT_WORD:
        value = UINT32_MAX
    elif token_type == TOKEN_SHA256_HEX:
        value = "f" * 64
    elif token_type == TOKEN_NULLABLE_FLOAT64:
        value = -float.fromhex("0x1.fffffffffffffp+1023")
    else:  # pragma: no cover - guarded by the codebook lookup
        raise AssertionError(token_type)
    for _ in range(depth):
        value = _packed_representative(value)
    return len(_canonical_json(value).encode("ascii"))


def _validation_words(
    token_type: str,
    raw_rule: Any,
    *,
    path: str,
) -> tuple[int, int, int, int, int, int, int, int]:
    if not isinstance(raw_rule, Mapping):
        _fail(path, "must be a mapping")
    kind = raw_rule.get("kind")
    depth = raw_rule.get("render_pack_depth")
    _uint(depth, bits=32, path=f"{path}.render_pack_depth")
    if depth > 2:
        _fail(f"{path}.render_pack_depth", "must be 0, 1, or 2")

    rank = 0
    parameter = 0
    minimum = 0
    maximum = 0
    if token_type == TOKEN_INTEGER:
        _exact_mapping(
            raw_rule,
            ("kind", "minimum", "maximum", "render_pack_depth"),
            path=path,
        )
        expected_kind, maximum = "integer_range", UINT64_MAX
    elif token_type == TOKEN_BOOLEAN:
        _exact_mapping(raw_rule, ("kind", "render_pack_depth"), path=path)
        expected_kind = "exact_boolean"
    elif token_type == TOKEN_PCG_UINT128:
        _exact_mapping(
            raw_rule,
            ("kind", "minimum", "maximum", "render_pack_depth"),
            path=path,
        )
        expected_kind, maximum = "integer_range", UINT128_MAX
    elif token_type == TOKEN_PYTHON_MT_WORD:
        _exact_mapping(
            raw_rule,
            ("kind", "minimum", "maximum", "render_pack_depth"),
            path=path,
        )
        expected_kind, maximum = "integer_range", UINT32_MAX
    elif token_type == TOKEN_TENSOR_SHAPE:
        _exact_mapping(
            raw_rule,
            (
                "kind",
                "rank",
                "minimum_dimension",
                "maximum_dimension",
                "render_pack_depth",
            ),
            path=path,
        )
        expected_kind = "tensor_shape"
        rank = _uint(raw_rule["rank"], bits=32, path=f"{path}.rank")
        maximum = UINT64_MAX
    elif token_type == TOKEN_SHA256_HEX:
        _exact_mapping(
            raw_rule,
            ("kind", "length", "render_pack_depth"),
            path=path,
        )
        expected_kind, parameter = "lowercase_sha256_hex", 64
    elif token_type == TOKEN_NULLABLE_FLOAT64:
        _exact_mapping(raw_rule, ("kind", "render_pack_depth"), path=path)
        expected_kind = "finite_optional_float64"
    else:  # pragma: no cover - guarded by the codebook lookup
        raise AssertionError(token_type)

    if kind != expected_kind:
        _fail(f"{path}.kind", f"must equal {expected_kind!r}")
    for name, expected in (
        ("minimum", minimum),
        ("maximum", maximum),
        ("minimum_dimension", minimum),
        ("maximum_dimension", maximum),
    ):
        if name in raw_rule:
            actual = _uint128(raw_rule[name], path=f"{path}.{name}")
            if actual != expected:
                _fail(f"{path}.{name}", f"must equal {expected}")
    if "length" in raw_rule:
        length = _uint(raw_rule["length"], bits=32, path=f"{path}.length")
        if length != parameter:
            _fail(f"{path}.length", f"must equal {parameter}")
    minimum_low, minimum_high = _split_uint128(minimum)
    maximum_low, maximum_high = _split_uint128(maximum)
    return (
        _VALIDATION_KIND_CODES[expected_kind],
        rank,
        depth,
        parameter,
        minimum_low,
        minimum_high,
        maximum_low,
        maximum_high,
    )


@dataclass(frozen=True, slots=True)
class CompactSummaryHostTemplateWire:
    """Frozen wrapper around the five ordered CPU tensor wire buffers."""

    programs: Any
    tokens: Any
    parts: Any
    constraints: Any
    byte_pool: Any
    manifest: Mapping[str, Any]

    @property
    def ordered_tensors(self) -> tuple[Any, Any, Any, Any, Any]:
        return (
            self.programs,
            self.tokens,
            self.parts,
            self.constraints,
            self.byte_pool,
        )

    @property
    def tensor_tuple(self) -> tuple[Any, Any, Any, Any, Any]:
        return self.ordered_tensors

    def as_tuple(self) -> tuple[Any, Any, Any, Any, Any]:
        return self.ordered_tensors


def _append_pool(pool: bytearray, value: bytes, *, path: str) -> int:
    offset = len(pool)
    _checked_add(offset, len(value), path=path)
    pool.extend(value)
    return offset


def _lower_token(
    raw: Any,
    *,
    path: str,
    expected_index: int,
    generation: int,
    cell: int,
    surface: str,
    pool: bytearray,
) -> tuple[list[int], str, int, int]:
    token = _exact_mapping(raw, _TOKEN_FIELDS, path=path)
    token_index = _uint(token["token_index"], bits=32, path=f"{path}.token_index")
    if token_index != expected_index:
        _fail(f"{path}.token_index", f"must equal {expected_index}")
    token_generation = _uint(
        token["generation"], bits=32, path=f"{path}.generation"
    )
    token_cell = _uint(token["cell"], bits=32, path=f"{path}.cell")
    if token_generation != generation or token_cell != cell:
        _fail(path, "token coordinates contradict its program")
    expected_id = f"g{generation}.c{cell}.{surface}.t{expected_index}"
    if token["token_id"] != expected_id:
        _fail(f"{path}.token_id", f"must equal {expected_id!r}")
    token_type = token["token_type"]
    if token_type not in _TOKEN_TYPE_CODES:
        _fail(f"{path}.token_type", "uses an unsupported renderer enum")
    source_path = token["source_path"]
    source_bytes = _utf8(source_path, path=f"{path}.source_path", source_path=True)
    source_offset = _append_pool(pool, source_bytes, path=f"{path}.source_path")
    (
        validation_code,
        rank,
        depth,
        parameter,
        minimum_low,
        minimum_high,
        maximum_low,
        maximum_high,
    ) = _validation_words(
        token_type,
        token["validation_rule"],
        path=f"{path}.validation_rule",
    )
    maximum_utf8_bytes = _uint(
        token["maximum_utf8_bytes"],
        bits=32,
        path=f"{path}.maximum_utf8_bytes",
    )
    expected_maximum = _maximum_token_utf8_bytes(token_type, rank, depth)
    if maximum_utf8_bytes != expected_maximum:
        _fail(
            f"{path}.maximum_utf8_bytes",
            f"must equal the renderer bound {expected_maximum}",
        )
    return (
        [
            token_index,
            _TOKEN_TYPE_CODES[token_type],
            validation_code,
            rank,
            depth,
            maximum_utf8_bytes,
            source_offset,
            len(source_bytes),
            parameter,
            minimum_low,
            minimum_high,
            maximum_low,
            maximum_high,
        ],
        source_path,
        _TOKEN_TYPE_CODES[token_type],
        rank,
    )


def _lower_template(
    raw: Any,
    *,
    template_index: int,
    generation: int,
    cell: int,
    surface: str,
    token_rows: list[list[int]],
    part_rows: list[list[int]],
    constraint_rows: list[list[int]],
    pool: bytearray,
) -> list[int]:
    path = f"bundle.templates[{template_index}]"
    template = _exact_mapping(raw, _TEMPLATE_FIELDS, path=path)
    if template["schema_version"] != HOST_TEMPLATE_SCHEMA_VERSION:
        _fail(f"{path}.schema_version", "unsupported template schema")
    actual_template_index = _uint(
        template["template_index"], bits=64, path=f"{path}.template_index"
    )
    if actual_template_index != template_index:
        _fail(f"{path}.template_index", f"must equal {template_index}")
    if template["surface"] != surface:
        _fail(f"{path}.surface", f"must equal {surface!r}")
    if template["wire_field"] != WIRE_FIELDS[surface]:
        _fail(f"{path}.wire_field", "contradicts the frozen surface field")
    if template["encoding"] != "utf-8":
        _fail(f"{path}.encoding", "must equal 'utf-8'")
    template_generation = _uint(
        template["generation"], bits=32, path=f"{path}.generation"
    )
    template_cell = _uint(template["cell"], bits=32, path=f"{path}.cell")
    if template_generation != generation or template_cell != cell:
        _fail(path, "template coordinates contradict bundle ordering")

    raw_tokens = _exact_list(template["tokens"], path=f"{path}.tokens")
    token_count = _uint(template["token_count"], bits=32, path=f"{path}.token_count")
    if token_count != len(raw_tokens) or token_count > MAX_TOKENS_PER_TEMPLATE:
        _fail(f"{path}.token_count", "does not match the bounded token rows")
    token_offset = len(token_rows)
    token_paths: list[str] = []
    token_declarations: dict[str, tuple[int, int]] = {}
    token_maximum_sum = 0
    for index, raw_token in enumerate(raw_tokens):
        row, source_path, type_code, rank = _lower_token(
            raw_token,
            path=f"{path}.tokens[{index}]",
            expected_index=index,
            generation=generation,
            cell=cell,
            surface=surface,
            pool=pool,
        )
        if source_path in token_declarations:
            _fail(f"{path}.tokens[{index}].source_path", "must be unique per program")
        token_paths.append(source_path)
        token_declarations[source_path] = (type_code, rank)
        token_maximum_sum = _checked_add(
            token_maximum_sum,
            row[5],
            path=f"{path}.maximum_rendered_utf8_bytes",
        )
        token_rows.append(row)

    raw_parts = _exact_list(template["parts"], path=f"{path}.parts")
    if not raw_parts or len(raw_parts) > MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE:
        _fail(f"{path}.parts", "part count is outside the renderer bound")
    _uint(len(raw_parts), bits=32, path=f"{path}.parts.count")
    part_offset = len(part_rows)
    literal_bytes = 0
    seen_tokens: set[int] = set()
    previous_literal = False
    for index, raw_part in enumerate(raw_parts):
        part_path = f"{path}.parts[{index}]"
        if not isinstance(raw_part, Mapping):
            _fail(part_path, "must be a mapping")
        kind = raw_part.get("kind")
        if kind == "literal_utf8":
            part = _exact_mapping(raw_part, ("kind", "value"), path=part_path)
            if previous_literal:
                _fail(part_path, "adjacent literals must be coalesced")
            literal = _utf8(part["value"], path=f"{part_path}.value")
            if not literal:
                _fail(f"{part_path}.value", "must not be empty")
            literal_offset = _append_pool(pool, literal, path=f"{part_path}.value")
            literal_bytes = _checked_add(
                literal_bytes,
                len(literal),
                path=f"{path}.literal_utf8_bytes",
            )
            part_rows.append(
                [_PART_KIND_CODES[kind], UINT32_MAX, literal_offset, len(literal), 0]
            )
            previous_literal = True
        elif kind == "dynamic_token":
            part = _exact_mapping(raw_part, ("kind", "token_index"), path=part_path)
            token_index = _uint(
                part["token_index"], bits=32, path=f"{part_path}.token_index"
            )
            if token_index >= token_count:
                _fail(f"{part_path}.token_index", "is outside the program token rows")
            part_rows.append([_PART_KIND_CODES[kind], token_index, 0, 0, 0])
            seen_tokens.add(token_index)
            previous_literal = False
        else:
            _fail(f"{part_path}.kind", "uses an unsupported renderer enum")
    if seen_tokens != set(range(token_count)):
        _fail(f"{path}.parts", "must reference every declared token")

    declared_literal_bytes = _uint(
        template["literal_utf8_bytes"],
        bits=64,
        path=f"{path}.literal_utf8_bytes",
    )
    if (
        declared_literal_bytes != literal_bytes
        or literal_bytes > MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE
    ):
        _fail(f"{path}.literal_utf8_bytes", "contradicts the bounded literal rows")
    maximum_rendered = _uint(
        template["maximum_rendered_utf8_bytes"],
        bits=64,
        path=f"{path}.maximum_rendered_utf8_bytes",
    )
    expected_maximum = _checked_add(
        literal_bytes,
        token_maximum_sum,
        path=f"{path}.maximum_rendered_utf8_bytes",
    )
    if (
        maximum_rendered != expected_maximum
        or maximum_rendered > MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE
    ):
        _fail(
            f"{path}.maximum_rendered_utf8_bytes",
            "contradicts the compiled token and literal bounds",
        )

    raw_constraints = _exact_list(
        template["constraints"], path=f"{path}.constraints"
    )
    if len(raw_constraints) > token_count:
        _fail(f"{path}.constraints", "count exceeds the program token count")
    _uint(len(raw_constraints), bits=32, path=f"{path}.constraints.count")
    if raw_constraints and surface not in {"rng", "policy_mutation_tape"}:
        _fail(f"{path}.constraints", "surface does not admit constraints")
    constraint_offset = len(constraint_rows)
    for index, raw_constraint in enumerate(raw_constraints):
        constraint_path = f"{path}.constraints[{index}]"
        if not isinstance(raw_constraint, Mapping):
            _fail(constraint_path, "must be a mapping")
        kind = raw_constraint.get("kind")
        if kind == "values_equal":
            constraint = _exact_mapping(
                raw_constraint,
                ("kind", "left_source_path", "right_source_path"),
                path=constraint_path,
            )
            left_name = constraint["left_source_path"]
            right_name = constraint["right_source_path"]
        elif kind == "shape_first_dimension_equals_integer":
            constraint = _exact_mapping(
                raw_constraint,
                ("kind", "shape_source_path", "integer_source_path"),
                path=constraint_path,
            )
            left_name = constraint["shape_source_path"]
            right_name = constraint["integer_source_path"]
        else:
            _fail(f"{constraint_path}.kind", "uses an unsupported renderer enum")
        left = _utf8(left_name, path=f"{constraint_path}.left_path", source_path=True)
        right = _utf8(
            right_name, path=f"{constraint_path}.right_path", source_path=True
        )
        if (
            left_name == right_name
            or left_name not in token_declarations
            or right_name not in token_declarations
        ):
            _fail(constraint_path, "paths must name two distinct program tokens")
        left_type, left_rank = token_declarations[left_name]
        right_type, right_rank = token_declarations[right_name]
        if kind == "values_equal":
            if (left_type, left_rank) != (right_type, right_rank):
                _fail(constraint_path, "values_equal token declarations must match")
        elif (
            surface != "policy_mutation_tape"
            or left_type != _TOKEN_TYPE_CODES[TOKEN_TENSOR_SHAPE]
            or left_rank == 0
            or right_type != _TOKEN_TYPE_CODES[TOKEN_INTEGER]
        ):
            _fail(constraint_path, "shape/integer constraint types contradict the ABI")
        left_offset = _append_pool(pool, left, path=f"{constraint_path}.left_path")
        right_offset = _append_pool(pool, right, path=f"{constraint_path}.right_path")
        constraint_rows.append(
            [
                _CONSTRAINT_KIND_CODES[kind],
                left_offset,
                len(left),
                right_offset,
                len(right),
            ]
        )

    return [
        HOST_TEMPLATE_RENDERER_ABI_VERSION,
        1,
        _SURFACE_CODES[surface],
        generation,
        cell,
        token_offset,
        token_count,
        part_offset,
        len(raw_parts),
        constraint_offset,
        len(raw_constraints),
        literal_bytes,
        maximum_rendered,
    ]


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def _tensor_from_rows(
    torch: Any,
    rows: list[list[int]],
    *,
    width: int,
    pin_memory: bool,
) -> Any:
    if rows:
        return torch.tensor(
            rows,
            dtype=torch.uint64,
            device="cpu",
            pin_memory=pin_memory,
        )
    return torch.empty(
        (1, width),
        dtype=torch.uint64,
        device="cpu",
        pin_memory=pin_memory,
    )[:0]


def _allocate_tensors(
    torch: Any,
    program_rows: list[list[int]],
    token_rows: list[list[int]],
    part_rows: list[list[int]],
    constraint_rows: list[list[int]],
    pool: bytearray,
    *,
    pin_memory: bool,
) -> tuple[Any, Any, Any, Any, Any]:
    programs = _tensor_from_rows(
        torch, program_rows, width=len(PROGRAM_COLUMNS), pin_memory=pin_memory
    )
    tokens = _tensor_from_rows(
        torch, token_rows, width=len(TOKEN_COLUMNS), pin_memory=pin_memory
    )
    parts = _tensor_from_rows(
        torch, part_rows, width=len(PART_COLUMNS), pin_memory=pin_memory
    )
    constraints = _tensor_from_rows(
        torch,
        constraint_rows,
        width=len(CONSTRAINT_COLUMNS),
        pin_memory=pin_memory,
    )
    if pool:
        byte_pool = torch.tensor(
            pool,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
    else:
        byte_pool = torch.empty(
            (1,), dtype=torch.uint8, device="cpu", pin_memory=pin_memory
        )[:0]
    return programs, tokens, parts, constraints, byte_pool


def _immutable_manifest(
    *,
    bundle: Mapping[str, Any],
    shapes: Mapping[str, tuple[int, ...]],
    pinned: bool,
    fallback: bool,
) -> Mapping[str, Any]:
    tensor_columns = MappingProxyType(dict(TENSOR_COLUMNS))
    tensor_dtypes = MappingProxyType(
        {name: ("uint8" if name == "byte_pool" else "uint64") for name in TENSOR_ORDER}
    )
    tensor_shapes = MappingProxyType(dict(shapes))
    bounds = MappingProxyType(dict(bundle["bounds"]))
    return MappingProxyType(
        {
            "schema_version": HOST_TEMPLATE_WIRE_SCHEMA_VERSION,
            "source_schema_version": HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION,
            "abi_version": HOST_TEMPLATE_RENDERER_ABI_VERSION,
            "representation": bundle["representation"],
            "ordering": bundle["ordering"],
            "surface_order": tuple(bundle["surface_order"]),
            "generation_count": bundle["generation_count"],
            "cell_count": bundle["cell_count"],
            "coordinate_count": bundle["coordinate_count"],
            "template_count": bundle["template_count"],
            "bounds": bounds,
            "tensor_order": TENSOR_ORDER,
            "tensor_columns": tensor_columns,
            "tensor_dtypes": tensor_dtypes,
            "tensor_shapes": tensor_shapes,
            "byte_pool_traversal": (
                "program_major_token_paths_then_literal_parts_then_constraint_paths"
            ),
            "pinned_memory": pinned,
            "test_only_unpinned_fallback": fallback,
        }
    )


def lower_compact_summary_host_template_bundle_to_wire(
    bundle: Mapping[str, Any],
    torch: Any,
    *,
    allow_unpinned_test_fallback: bool = False,
) -> CompactSummaryHostTemplateWire:
    """Validate and lower one compiled bundle into five ordered CPU tensors.

    Pinned allocation is mandatory.  ``allow_unpinned_test_fallback`` is an
    explicit test-only escape hatch: it is honored only after pinned allocation
    fails and the supplied Torch module reports that CUDA is unavailable.
    """

    if type(allow_unpinned_test_fallback) is not bool:
        _fail("allow_unpinned_test_fallback", "must be a bool")
    source = _exact_mapping(bundle, _BUNDLE_FIELDS, path="bundle")
    if source["schema_version"] != HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION:
        _fail("bundle.schema_version", "unsupported compiled bundle schema")
    if source["representation"] != "immutable_utf8_segments_plus_bounded_dynamic_tokens":
        _fail("bundle.representation", "unsupported compiled representation")
    if source["ordering"] != "generation_major_cell_major_surface_order":
        _fail("bundle.ordering", "unsupported template ordering")
    surface_order = _exact_list(source["surface_order"], path="bundle.surface_order")
    if surface_order != list(SURFACE_ORDER):
        _fail("bundle.surface_order", "contradicts the renderer surface enums")

    generations = _uint(source["generation_count"], bits=32, path="bundle.generation_count")
    cells = _uint(source["cell_count"], bits=32, path="bundle.cell_count")
    if generations == 0 or cells == 0:
        _fail("bundle", "generation_count and cell_count must be positive")
    coordinate_count = _checked_add(0, generations * cells, path="bundle.coordinate_count")
    declared_coordinate_count = _uint(
        source["coordinate_count"], bits=64, path="bundle.coordinate_count"
    )
    if coordinate_count > MAX_COORDINATES or declared_coordinate_count != coordinate_count:
        _fail("bundle.coordinate_count", "contradicts the bounded coordinate matrix")
    template_count = _checked_add(
        0, coordinate_count * len(SURFACE_ORDER), path="bundle.template_count"
    )
    declared_template_count = _uint(
        source["template_count"], bits=64, path="bundle.template_count"
    )
    if declared_template_count != template_count:
        _fail("bundle.template_count", "contradicts coordinates and surfaces")

    bounds = _exact_mapping(source["bounds"], _BOUND_FIELDS, path="bundle.bounds")
    expected_bounds = {
        "maximum_coordinates": MAX_COORDINATES,
        "maximum_tokens_per_template": MAX_TOKENS_PER_TEMPLATE,
        "maximum_literal_utf8_bytes_per_template": MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE,
        "maximum_rendered_utf8_bytes_per_template": MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE,
    }
    for name, expected in expected_bounds.items():
        actual = _uint(bounds[name], bits=64, path=f"bundle.bounds.{name}")
        if actual != expected:
            _fail(f"bundle.bounds.{name}", f"must equal {expected}")
    raw_templates = _exact_list(source["templates"], path="bundle.templates")
    if len(raw_templates) != template_count:
        _fail("bundle.templates", "count contradicts bundle.template_count")

    program_rows: list[list[int]] = []
    token_rows: list[list[int]] = []
    part_rows: list[list[int]] = []
    constraint_rows: list[list[int]] = []
    pool = bytearray()
    for template_index, raw_template in enumerate(raw_templates):
        coordinate_index, surface_index = divmod(template_index, len(SURFACE_ORDER))
        generation, cell = divmod(coordinate_index, cells)
        program_rows.append(
            _lower_template(
                raw_template,
                template_index=template_index,
                generation=generation,
                cell=cell,
                surface=SURFACE_ORDER[surface_index],
                token_rows=token_rows,
                part_rows=part_rows,
                constraint_rows=constraint_rows,
                pool=pool,
            )
        )
    for name, count in (
        ("tokens", len(token_rows)),
        ("parts", len(part_rows)),
        ("constraints", len(constraint_rows)),
        ("byte_pool", len(pool)),
    ):
        _uint(count, bits=64, path=f"wire.{name}.count")

    used_fallback = False
    try:
        torch.empty(
            (1,), dtype=torch.uint8, device="cpu", pin_memory=True
        )
    except RuntimeError as exc:
        if not allow_unpinned_test_fallback or _cuda_available(torch):
            _fail("wire.pin_memory", f"pinned CPU allocation failed: {exc}")
        used_fallback = True
    try:
        tensors = _allocate_tensors(
            torch,
            program_rows,
            token_rows,
            part_rows,
            constraint_rows,
            pool,
            pin_memory=not used_fallback,
        )
    except RuntimeError as exc:
        _fail("wire.allocation", f"CPU tensor allocation failed: {exc}")

    shapes = {
        "programs": (len(program_rows), len(PROGRAM_COLUMNS)),
        "tokens": (len(token_rows), len(TOKEN_COLUMNS)),
        "parts": (len(part_rows), len(PART_COLUMNS)),
        "constraints": (len(constraint_rows), len(CONSTRAINT_COLUMNS)),
        "byte_pool": (len(pool),),
    }
    manifest = _immutable_manifest(
        bundle=source,
        shapes=shapes,
        pinned=not used_fallback,
        fallback=used_fallback,
    )
    wire = CompactSummaryHostTemplateWire(*tensors, manifest)
    validate_compact_summary_host_template_wire(wire, torch=torch)
    return wire


def _pool_span(pool: bytes, offset: int, size: int, *, path: str) -> bytes:
    if offset > len(pool) or size > len(pool) - offset:
        _fail(path, "byte-pool span is out of bounds")
    return pool[offset : offset + size]


def _valid_utf8_bytes(value: bytes, *, path: str, source_path: bool = False) -> None:
    try:
        value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail(path, f"contains invalid UTF-8: {exc.reason}")
    if source_path and (
        not value or value[0] != ord("/") or len(value) > MAX_SOURCE_PATH_UTF8_BYTES
    ):
        _fail(path, "is not a bounded absolute source path")


def validate_compact_summary_host_template_wire(
    wire: CompactSummaryHostTemplateWire,
    *,
    torch: Any,
) -> None:
    """Fail closed unless a wire object is zero-copy safe for native rebinding."""

    if not isinstance(wire, CompactSummaryHostTemplateWire):
        _fail("wire", "must be a CompactSummaryHostTemplateWire")
    manifest = _exact_mapping(wire.manifest, _MANIFEST_FIELDS, path="wire.manifest")
    if (
        manifest["schema_version"] != HOST_TEMPLATE_WIRE_SCHEMA_VERSION
        or manifest["source_schema_version"] != HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION
        or manifest["abi_version"] != HOST_TEMPLATE_RENDERER_ABI_VERSION
        or manifest["representation"]
        != "immutable_utf8_segments_plus_bounded_dynamic_tokens"
        or manifest["ordering"] != "generation_major_cell_major_surface_order"
        or manifest["surface_order"] != SURFACE_ORDER
        or manifest["tensor_order"] != TENSOR_ORDER
        or manifest["tensor_columns"] != TENSOR_COLUMNS
        or manifest["byte_pool_traversal"]
        != "program_major_token_paths_then_literal_parts_then_constraint_paths"
    ):
        _fail("wire.manifest", "frozen schema metadata drift")
    expected_dtypes = {
        name: ("uint8" if name == "byte_pool" else "uint64")
        for name in TENSOR_ORDER
    }
    if manifest["tensor_dtypes"] != expected_dtypes:
        _fail("wire.manifest.tensor_dtypes", "tensor dtype metadata drift")
    generations = _uint(
        manifest["generation_count"],
        bits=32,
        path="wire.manifest.generation_count",
    )
    cells = _uint(
        manifest["cell_count"], bits=32, path="wire.manifest.cell_count"
    )
    coordinates = _uint(
        manifest["coordinate_count"],
        bits=64,
        path="wire.manifest.coordinate_count",
    )
    template_count = _uint(
        manifest["template_count"], bits=64, path="wire.manifest.template_count"
    )
    if (
        generations == 0
        or cells == 0
        or generations * cells != coordinates
        or coordinates > MAX_COORDINATES
        or coordinates * len(SURFACE_ORDER) != template_count
    ):
        _fail("wire.manifest", "coordinate counts drift")
    expected_bounds = {
        "maximum_coordinates": MAX_COORDINATES,
        "maximum_tokens_per_template": MAX_TOKENS_PER_TEMPLATE,
        "maximum_literal_utf8_bytes_per_template": (
            MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE
        ),
        "maximum_rendered_utf8_bytes_per_template": (
            MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE
        ),
    }
    if manifest["bounds"] != expected_bounds:
        _fail("wire.manifest.bounds", "renderer bounds drift")
    expected_shapes = manifest["tensor_shapes"]
    if not isinstance(expected_shapes, Mapping):
        _fail("wire.manifest.tensor_shapes", "must be a mapping")
    if set(expected_shapes) != set(TENSOR_ORDER):
        _fail("wire.manifest.tensor_shapes", "tensor names are not exact")
    pinned_required = manifest["pinned_memory"] is True
    fallback = manifest["test_only_unpinned_fallback"] is True
    if not pinned_required and not fallback:
        _fail("wire.manifest", "must declare pinned memory or the test-only fallback")
    if fallback and _cuda_available(torch):
        _fail("wire.manifest", "test-only fallback requires unavailable CUDA pinning")
    for name, tensor in zip(TENSOR_ORDER, wire.ordered_tensors):
        expected_dtype = torch.uint8 if name == "byte_pool" else torch.uint64
        if tensor.dtype != expected_dtype:
            _fail(f"wire.{name}.dtype", f"must be {expected_dtype}")
        if tensor.device.type != "cpu" or tensor.requires_grad:
            _fail(f"wire.{name}", "must be a non-gradient CPU tensor")
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            _fail(f"wire.{name}", "must be base-offset contiguous storage")
        if tuple(tensor.shape) != tuple(expected_shapes.get(name, ())):
            _fail(f"wire.{name}.shape", "contradicts the immutable manifest")
        if pinned_required and not tensor.is_pinned():
            _fail(f"wire.{name}", "must use pinned CPU storage")
        if fallback and tensor.is_pinned():
            _fail(f"wire.{name}", "test fallback must consistently be unpinned")

    programs = wire.programs.tolist()
    tokens = wire.tokens.tolist()
    parts = wire.parts.tolist()
    constraints = wire.constraints.tolist()
    pool = bytes(wire.byte_pool.tolist())
    expected_programs = manifest["template_count"]
    if type(expected_programs) is not int or len(programs) != expected_programs:
        _fail("wire.programs", "row count contradicts the manifest")
    if type(cells) is not int or cells <= 0:
        _fail("wire.manifest.cell_count", "must be positive")

    token_cursor = 0
    part_cursor = 0
    constraint_cursor = 0
    pool_cursor = 0
    for program_index, row in enumerate(programs):
        if len(row) != len(PROGRAM_COLUMNS):
            _fail(f"wire.programs[{program_index}]", "column count mismatch")
        (
            abi,
            encoding,
            surface,
            generation,
            cell,
            token_offset,
            token_count,
            part_offset,
            part_count,
            constraint_offset,
            constraint_count,
            literal_bytes,
            maximum_rendered,
        ) = row
        coordinate_index, expected_surface_index = divmod(
            program_index, len(SURFACE_ORDER)
        )
        expected_generation, expected_cell = divmod(coordinate_index, cells)
        if (abi, encoding, surface, generation, cell) != (
            1,
            1,
            expected_surface_index + 1,
            expected_generation,
            expected_cell,
        ):
            _fail(f"wire.programs[{program_index}]", "ABI, surface, or coordinate drift")
        if (token_offset, part_offset, constraint_offset) != (
            token_cursor,
            part_cursor,
            constraint_cursor,
        ):
            _fail(f"wire.programs[{program_index}]", "row offsets are not contiguous")
        if (
            generation > UINT32_MAX
            or cell > UINT32_MAX
            or token_count > MAX_TOKENS_PER_TEMPLATE
            or not 0 < part_count <= MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE
            or part_count > UINT32_MAX
            or constraint_count > token_count
            or literal_bytes > MAX_LITERAL_UTF8_BYTES_PER_TEMPLATE
            or maximum_rendered > MAX_RENDERED_UTF8_BYTES_PER_TEMPLATE
            or token_count > len(tokens) - token_cursor
            or part_count > len(parts) - part_cursor
            or constraint_count > len(constraints) - constraint_cursor
        ):
            _fail(f"wire.programs[{program_index}]", "row slice is out of bounds")

        local_paths: list[bytes] = []
        local_token_rows: list[list[int]] = []
        token_maximum_sum = 0
        for local_index in range(token_count):
            token = tokens[token_cursor + local_index]
            if len(token) != len(TOKEN_COLUMNS) or token[0] != local_index:
                _fail(f"wire.tokens[{token_cursor + local_index}]", "token index drift")
            token_path = f"wire.tokens[{token_cursor + local_index}]"
            if (
                token[1] not in _TOKEN_TYPE_CODES.values()
                or token[2] not in _VALIDATION_KIND_CODES.values()
            ):
                _fail(token_path, "unsupported enum code")
            if (
                token[0] > UINT32_MAX
                or token[1] > UINT32_MAX
                or token[2] > UINT32_MAX
                or token[3] > UINT32_MAX
                or token[4] > 2
                or token[5] > UINT32_MAX
                or token[7] == 0
                or token[7] > UINT32_MAX
                or token[8] > UINT32_MAX
            ):
                _fail(token_path, "declaration bound drift")
            expected_rule_words: tuple[int, int, int, int, int, int]
            if token[1] == _TOKEN_TYPE_CODES[TOKEN_INTEGER]:
                expected_rule_words = (1, 0, 0, 0, UINT64_MAX, 0)
            elif token[1] == _TOKEN_TYPE_CODES[TOKEN_BOOLEAN]:
                expected_rule_words = (2, 0, 0, 0, 0, 0)
            elif token[1] == _TOKEN_TYPE_CODES[TOKEN_PCG_UINT128]:
                expected_rule_words = (1, 0, 0, 0, UINT64_MAX, UINT64_MAX)
            elif token[1] == _TOKEN_TYPE_CODES[TOKEN_PYTHON_MT_WORD]:
                expected_rule_words = (1, 0, 0, 0, UINT32_MAX, 0)
            elif token[1] == _TOKEN_TYPE_CODES[TOKEN_TENSOR_SHAPE]:
                expected_rule_words = (3, token[3], 0, 0, UINT64_MAX, 0)
            elif token[1] == _TOKEN_TYPE_CODES[TOKEN_SHA256_HEX]:
                expected_rule_words = (4, 0, 64, 0, 0, 0)
            else:
                expected_rule_words = (5, 0, 0, 0, 0, 0)
            actual_rule_words = (
                token[2],
                token[3],
                token[8],
                token[9] | token[10],
                token[11],
                token[12],
            )
            if actual_rule_words != expected_rule_words:
                _fail(token_path, "token type and validation rule contradict the ABI")
            token_type = next(
                name for name, code in _TOKEN_TYPE_CODES.items() if code == token[1]
            )
            if token[5] != _maximum_token_utf8_bytes(
                token_type, token[3], token[4]
            ):
                _fail(token_path, "maximum UTF-8 bytes contradict the renderer")
            if token[6] != pool_cursor:
                _fail(f"wire.tokens[{token_cursor + local_index}]", "pool traversal drift")
            source = _pool_span(
                pool,
                token[6],
                token[7],
                path=f"wire.tokens[{token_cursor + local_index}].source_path",
            )
            _valid_utf8_bytes(
                source,
                path=f"wire.tokens[{token_cursor + local_index}].source_path",
                source_path=True,
            )
            if source in local_paths:
                _fail(f"wire.tokens[{token_cursor + local_index}]", "duplicate source path")
            local_paths.append(source)
            local_token_rows.append(token)
            pool_cursor += token[7]
            token_maximum_sum = _checked_add(
                token_maximum_sum,
                token[5],
                path=f"wire.programs[{program_index}].maximum_rendered_utf8_bytes",
            )

        actual_literal_bytes = 0
        seen_tokens: set[int] = set()
        previous_literal = False
        for local_index in range(part_count):
            part = parts[part_cursor + local_index]
            part_path = f"wire.parts[{part_cursor + local_index}]"
            if len(part) != len(PART_COLUMNS) or part[4] != 0:
                _fail(part_path, "column count or reserved field drift")
            if part[0] == _PART_KIND_CODES["literal_utf8"]:
                if (
                    previous_literal
                    or part[1] != UINT32_MAX
                    or part[3] == 0
                    or part[2] != pool_cursor
                ):
                    _fail(part_path, "literal row drift")
                literal = _pool_span(pool, part[2], part[3], path=part_path)
                _valid_utf8_bytes(literal, path=part_path)
                pool_cursor += part[3]
                actual_literal_bytes = _checked_add(
                    actual_literal_bytes, part[3], path=part_path
                )
                previous_literal = True
            elif part[0] == _PART_KIND_CODES["dynamic_token"]:
                if part[1] >= token_count or part[2:] != [0, 0, 0]:
                    _fail(part_path, "dynamic-token row drift")
                seen_tokens.add(part[1])
                previous_literal = False
            else:
                _fail(part_path, "unsupported part enum")
        if seen_tokens != set(range(token_count)) or actual_literal_bytes != literal_bytes:
            _fail(f"wire.programs[{program_index}]", "part rows contradict the program")
        if maximum_rendered != literal_bytes + token_maximum_sum:
            _fail(f"wire.programs[{program_index}]", "rendered bound drift")
        if constraint_count and surface not in (
            _SURFACE_CODES["rng"],
            _SURFACE_CODES["policy_mutation_tape"],
        ):
            _fail(f"wire.programs[{program_index}]", "constraint surface drift")

        for local_index in range(constraint_count):
            constraint = constraints[constraint_cursor + local_index]
            constraint_path = f"wire.constraints[{constraint_cursor + local_index}]"
            if (
                len(constraint) != len(CONSTRAINT_COLUMNS)
                or constraint[0] not in _CONSTRAINT_KIND_CODES.values()
            ):
                _fail(constraint_path, "column count or enum drift")
            if constraint[1] != pool_cursor:
                _fail(constraint_path, "left-path traversal drift")
            left = _pool_span(pool, constraint[1], constraint[2], path=constraint_path)
            _valid_utf8_bytes(left, path=constraint_path, source_path=True)
            pool_cursor += constraint[2]
            if constraint[3] != pool_cursor:
                _fail(constraint_path, "right-path traversal drift")
            right = _pool_span(pool, constraint[3], constraint[4], path=constraint_path)
            _valid_utf8_bytes(right, path=constraint_path, source_path=True)
            pool_cursor += constraint[4]
            if left == right or left not in local_paths or right not in local_paths:
                _fail(constraint_path, "paths do not name distinct local tokens")
            left_token = local_token_rows[local_paths.index(left)]
            right_token = local_token_rows[local_paths.index(right)]
            if constraint[0] == _CONSTRAINT_KIND_CODES["values_equal"]:
                if (left_token[1], left_token[3]) != (
                    right_token[1],
                    right_token[3],
                ):
                    _fail(constraint_path, "values_equal declaration drift")
            elif (
                surface != _SURFACE_CODES["policy_mutation_tape"]
                or left_token[1] != _TOKEN_TYPE_CODES[TOKEN_TENSOR_SHAPE]
                or left_token[3] == 0
                or right_token[1] != _TOKEN_TYPE_CODES[TOKEN_INTEGER]
            ):
                _fail(constraint_path, "shape/integer declaration drift")

        token_cursor += token_count
        part_cursor += part_count
        constraint_cursor += constraint_count
    if (token_cursor, part_cursor, constraint_cursor, pool_cursor) != (
        len(tokens),
        len(parts),
        len(constraints),
        len(pool),
    ):
        _fail("wire", "one or more rows or byte-pool spans are unowned")


__all__ = [
    "CONSTRAINT_COLUMNS",
    "CompactSummaryHostTemplateWire",
    "CompactSummaryHostTemplateWireError",
    "HOST_TEMPLATE_RENDERER_ABI_VERSION",
    "HOST_TEMPLATE_WIRE_SCHEMA_VERSION",
    "PART_COLUMNS",
    "PROGRAM_COLUMNS",
    "TENSOR_COLUMNS",
    "TENSOR_ORDER",
    "TOKEN_COLUMNS",
    "lower_compact_summary_host_template_bundle_to_wire",
    "validate_compact_summary_host_template_wire",
]
