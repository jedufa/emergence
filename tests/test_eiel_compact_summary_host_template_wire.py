from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from typing import Any, Callable

import pytest
import torch

from engine.native_runtime.compact_summary_host_template import (
    HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION,
    HOST_TEMPLATE_SCHEMA_VERSION,
    SURFACE_ORDER,
    TOKEN_NULLABLE_FLOAT64,
    WIRE_FIELDS,
    compile_compact_summary_host_templates,
    dynamic_token_placeholder,
)
from engine.native_runtime.compact_summary_host_template_wire import (
    CONSTRAINT_COLUMNS,
    PART_COLUMNS,
    PROGRAM_COLUMNS,
    TENSOR_ORDER,
    TOKEN_COLUMNS,
    CompactSummaryHostTemplateWireError,
    lower_compact_summary_host_template_bundle_to_wire,
    validate_compact_summary_host_template_wire,
)
from tests.test_eiel_cuda_architecture_b_compact_summary_host_template_renderer_v1 import (
    _placeholder_authority,
    _real_settled_authority,
)


UINT32_MAX = (1 << 32) - 1
_SURFACES = {index + 1: value for index, value in enumerate(SURFACE_ORDER)}
_TOKEN_TYPES = {
    1: "integer",
    2: "boolean",
    3: "pcg_uint128",
    4: "python_mt_word",
    5: "tensor_shape",
    6: "sha256_hex",
    7: "nullable_float64",
}
_VALIDATION_KINDS = {
    1: "integer_range",
    2: "exact_boolean",
    3: "tensor_shape",
    4: "lowercase_sha256_hex",
    5: "finite_optional_float64",
}


@pytest.fixture(scope="module")
def real_compiled_bundle() -> dict[str, Any]:
    plan, fixture_path, settled = _real_settled_authority()
    placeholder = _placeholder_authority(settled, plan)
    path = (
        "/checkpoint/projection_fields/metadata/"
        "native_runtime_python_random_state/gauss_next"
    )
    placeholder["checkpoint"]["projection_fields"]["metadata"][
        "native_runtime_python_random_state"
    ]["gauss_next"] = dynamic_token_placeholder(TOKEN_NULLABLE_FLOAT64, path)
    return compile_compact_summary_host_templates(
        fixture_path,
        plan,
        1,
        1,
        placeholder,
    )


def _decode_pool(pool: bytes, offset: int, size: int) -> str:
    return pool[offset : offset + size].decode("utf-8")


def _decode_rule(row: list[int]) -> dict[str, Any]:
    kind = _VALIDATION_KINDS[row[2]]
    rule: dict[str, Any] = {"kind": kind}
    minimum = row[9] | (row[10] << 64)
    maximum = row[11] | (row[12] << 64)
    if kind == "integer_range":
        rule.update(minimum=minimum, maximum=maximum)
    elif kind == "tensor_shape":
        rule.update(
            rank=row[3],
            minimum_dimension=minimum,
            maximum_dimension=maximum,
        )
    elif kind == "lowercase_sha256_hex":
        rule["length"] = row[8]
    rule["render_pack_depth"] = row[4]
    return rule


def _reconstruct_bundle(wire: Any) -> dict[str, Any]:
    programs = wire.programs.tolist()
    tokens = wire.tokens.tolist()
    parts = wire.parts.tolist()
    constraints = wire.constraints.tolist()
    pool = bytes(wire.byte_pool.tolist())
    templates: list[dict[str, Any]] = []
    for template_index, program in enumerate(programs):
        surface = _SURFACES[program[2]]
        token_offset, token_count = program[5:7]
        part_offset, part_count = program[7:9]
        constraint_offset, constraint_count = program[9:11]
        decoded_tokens = []
        for row in tokens[token_offset : token_offset + token_count]:
            decoded_tokens.append(
                {
                    "token_index": row[0],
                    "token_id": (
                        f"g{program[3]}.c{program[4]}.{surface}.t{row[0]}"
                    ),
                    "token_type": _TOKEN_TYPES[row[1]],
                    "source_path": _decode_pool(pool, row[6], row[7]),
                    "generation": program[3],
                    "cell": program[4],
                    "validation_rule": _decode_rule(row),
                    "maximum_utf8_bytes": row[5],
                }
            )
        decoded_parts = []
        for row in parts[part_offset : part_offset + part_count]:
            if row[0] == 1:
                assert row[1] == UINT32_MAX and row[4] == 0
                decoded_parts.append(
                    {
                        "kind": "literal_utf8",
                        "value": _decode_pool(pool, row[2], row[3]),
                    }
                )
            else:
                assert row[0] == 2 and row[2:] == [0, 0, 0]
                decoded_parts.append(
                    {"kind": "dynamic_token", "token_index": row[1]}
                )
        decoded_constraints = []
        for row in constraints[
            constraint_offset : constraint_offset + constraint_count
        ]:
            left = _decode_pool(pool, row[1], row[2])
            right = _decode_pool(pool, row[3], row[4])
            if row[0] == 1:
                decoded_constraints.append(
                    {
                        "kind": "values_equal",
                        "left_source_path": left,
                        "right_source_path": right,
                    }
                )
            else:
                assert row[0] == 2
                decoded_constraints.append(
                    {
                        "kind": "shape_first_dimension_equals_integer",
                        "shape_source_path": left,
                        "integer_source_path": right,
                    }
                )
        templates.append(
            {
                "schema_version": HOST_TEMPLATE_SCHEMA_VERSION,
                "surface": surface,
                "wire_field": WIRE_FIELDS[surface],
                "generation": program[3],
                "cell": program[4],
                "encoding": "utf-8",
                "literal_utf8_bytes": program[11],
                "token_count": program[6],
                "maximum_rendered_utf8_bytes": program[12],
                "tokens": decoded_tokens,
                "parts": decoded_parts,
                "constraints": decoded_constraints,
                "template_index": template_index,
            }
        )
    manifest = wire.manifest
    return {
        "schema_version": manifest["source_schema_version"],
        "representation": manifest["representation"],
        "ordering": manifest["ordering"],
        "surface_order": list(manifest["surface_order"]),
        "generation_count": manifest["generation_count"],
        "cell_count": manifest["cell_count"],
        "coordinate_count": manifest["coordinate_count"],
        "template_count": manifest["template_count"],
        "bounds": dict(manifest["bounds"]),
        "templates": templates,
    }


def test_real_six_surface_fixture_round_trips_every_compiler_field(
    real_compiled_bundle: dict[str, Any],
) -> None:
    wire = lower_compact_summary_host_template_bundle_to_wire(
        real_compiled_bundle, torch
    )

    assert real_compiled_bundle["schema_version"] == HOST_TEMPLATE_BUNDLE_SCHEMA_VERSION
    assert [row["surface"] for row in real_compiled_bundle["templates"]] == list(
        SURFACE_ORDER
    )
    assert _reconstruct_bundle(wire) == real_compiled_bundle
    assert wire.ordered_tensors == wire.tensor_tuple == wire.as_tuple()
    assert wire.manifest["tensor_order"] == TENSOR_ORDER
    assert wire.manifest["tensor_columns"] == {
        "programs": PROGRAM_COLUMNS,
        "tokens": TOKEN_COLUMNS,
        "parts": PART_COLUMNS,
        "constraints": CONSTRAINT_COLUMNS,
        "byte_pool": ("byte",),
    }
    assert [tuple(value.shape) for value in wire.ordered_tensors] == [
        (6, 13),
        (2495, 13),
        (4996, 5),
        (11, 5),
        (313767,),
    ]
    for tensor in wire.ordered_tensors:
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()
        assert tensor.storage_offset() == 0
        assert tensor.is_pinned()
        assert not tensor.requires_grad
    validate_compact_summary_host_template_wire(wire, torch=torch)


def test_repeated_lowering_is_byte_identical_and_metadata_is_frozen(
    real_compiled_bundle: dict[str, Any],
) -> None:
    first = lower_compact_summary_host_template_bundle_to_wire(
        real_compiled_bundle, torch
    )
    second = lower_compact_summary_host_template_bundle_to_wire(
        copy.deepcopy(real_compiled_bundle), torch
    )

    assert first.manifest == second.manifest
    for left, right in zip(first.ordered_tensors, second.ordered_tensors):
        assert torch.equal(left, right)
        assert left.numpy().tobytes(order="C") == right.numpy().tobytes(order="C")
    with pytest.raises(FrozenInstanceError):
        first.programs = second.programs
    with pytest.raises(TypeError):
        first.manifest["template_count"] = 0
    with pytest.raises(TypeError):
        first.manifest["bounds"]["maximum_coordinates"] = 0


def test_nullable_float64_uses_frozen_wire_codes_and_pack_depths(
    real_compiled_bundle: dict[str, Any],
) -> None:
    wire = lower_compact_summary_host_template_bundle_to_wire(
        real_compiled_bundle, torch
    )
    source_path = (
        "/checkpoint/projection_fields/metadata/"
        "native_runtime_python_random_state/gauss_next"
    )
    source_rows = [
        row
        for token, row in zip(
            (
                token
                for template in real_compiled_bundle["templates"]
                for token in template["tokens"]
            ),
            wire.tokens.tolist(),
            strict=True,
        )
        if token["source_path"] == source_path
    ]

    assert [(row[1], row[2], row[3], row[4]) for row in source_rows] == [
        (7, 5, 0, 1),
        (7, 5, 0, 2),
    ]


def _extra_bundle_field(bundle: dict[str, Any]) -> None:
    bundle["extra"] = 1


def _surface_order_drift(bundle: dict[str, Any]) -> None:
    bundle["surface_order"] = list(reversed(bundle["surface_order"]))


def _coordinate_count_drift(bundle: dict[str, Any]) -> None:
    bundle["coordinate_count"] += 1


def _bound_drift(bundle: dict[str, Any]) -> None:
    bundle["bounds"]["maximum_tokens_per_template"] -= 1


def _template_coordinate_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["cell"] = 1


def _template_schema_extra(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["part_count"] = len(bundle["templates"][0]["parts"])


def _token_count_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["token_count"] -= 1


def _token_index_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["tokens"][0]["token_index"] = 1


def _token_bound_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["tokens"][0]["maximum_utf8_bytes"] -= 1


def _token_rule_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["tokens"][0]["validation_rule"]["maximum"] = 7


def _boolean_rule_word(bundle: dict[str, Any]) -> None:
    token = next(
        token
        for template in bundle["templates"]
        for token in template["tokens"]
        if token["validation_rule"]["kind"] == "integer_range"
    )
    token["validation_rule"]["minimum"] = False


def _maximum_shape_rank(bundle: dict[str, Any]) -> None:
    token = next(
        token
        for template in bundle["templates"]
        for token in template["tokens"]
        if token["validation_rule"]["kind"] == "tensor_shape"
    )
    token["validation_rule"]["rank"] = UINT32_MAX


def _invalid_source_utf8(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["tokens"][0]["source_path"] = "/bad\ud800path"


def _invalid_literal_utf8(bundle: dict[str, Any]) -> None:
    literal = next(
        part
        for part in bundle["templates"][0]["parts"]
        if part["kind"] == "literal_utf8"
    )
    literal["value"] = "bad\ud800literal"


def _literal_count_drift(bundle: dict[str, Any]) -> None:
    bundle["templates"][0]["literal_utf8_bytes"] -= 1


def _constraint_path_drift(bundle: dict[str, Any]) -> None:
    constraint = bundle["templates"][3]["constraints"][0]
    constraint["right_source_path"] = "/missing"


@pytest.mark.parametrize(
    "mutate",
    [
        _extra_bundle_field,
        _surface_order_drift,
        _coordinate_count_drift,
        _bound_drift,
        _template_coordinate_drift,
        _template_schema_extra,
        _token_count_drift,
        _token_index_drift,
        _token_bound_drift,
        _token_rule_drift,
        _boolean_rule_word,
        _maximum_shape_rank,
        _invalid_source_utf8,
        _invalid_literal_utf8,
        _literal_count_drift,
        _constraint_path_drift,
    ],
)
def test_malformed_compiler_bundles_fail_closed(
    real_compiled_bundle: dict[str, Any],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    malformed = copy.deepcopy(real_compiled_bundle)
    mutate(malformed)
    with pytest.raises(CompactSummaryHostTemplateWireError):
        lower_compact_summary_host_template_bundle_to_wire(malformed, torch)


def test_post_pack_validator_rejects_mutated_tensor_declarations(
    real_compiled_bundle: dict[str, Any],
) -> None:
    wire = lower_compact_summary_host_template_bundle_to_wire(
        real_compiled_bundle, torch
    )
    wire.tokens[0, 1] = 2

    with pytest.raises(
        CompactSummaryHostTemplateWireError,
        match="token type and validation rule",
    ):
        validate_compact_summary_host_template_wire(wire, torch=torch)


class _UnavailableCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _NoPinTorch:
    uint64 = torch.uint64
    uint8 = torch.uint8
    cuda = _UnavailableCuda()

    @staticmethod
    def tensor(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("pin_memory"):
            raise RuntimeError("test allocator has no CUDA pin-memory allocator")
        return torch.tensor(*args, **kwargs)

    @staticmethod
    def empty(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("pin_memory"):
            raise RuntimeError("test allocator has no CUDA pin-memory allocator")
        return torch.empty(*args, **kwargs)


def test_unpinned_fallback_is_explicit_and_only_for_cuda_unavailable_tests(
    real_compiled_bundle: dict[str, Any],
) -> None:
    no_pin_torch = _NoPinTorch()
    with pytest.raises(CompactSummaryHostTemplateWireError, match="pinned CPU"):
        lower_compact_summary_host_template_bundle_to_wire(
            real_compiled_bundle, no_pin_torch
        )

    wire = lower_compact_summary_host_template_bundle_to_wire(
        real_compiled_bundle,
        no_pin_torch,
        allow_unpinned_test_fallback=True,
    )
    assert wire.manifest["pinned_memory"] is False
    assert wire.manifest["test_only_unpinned_fallback"] is True
    assert not any(tensor.is_pinned() for tensor in wire.ordered_tensors)
