from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from engine.native_runtime import compact_summary_planes as planes


def test_exact_v2_registries_and_logical_bytes() -> None:
    assert planes.ABI_VERSION == "eiel_compact_summary_native_abi_v2"
    assert len(planes.STATUS_FIELDS) == 12
    assert len(planes.INTEGER_FIELDS) == 76
    assert len(planes.FLOAT_FIELDS) == 21
    assert len(planes.DIGEST_FIELDS) == 10
    assert planes.TEXT_FIELDS == ("lineage_metrics.dominant_lineage_id",)
    assert len(planes.DEFINED_FIELDS) == 32
    assert tuple(planes.DEFINED_FIELD_BITS.values()) == tuple(range(32))
    assert planes.DEFINED_MASK_WORD_COUNT == 1
    assert planes.LOGICAL_BYTES_PER_GENERATION_CELL == 1_218
    assert planes.LOGICAL_BYTES_PER_GENERATION_CELL == (
        1 + 8 + 12 * 4 + 76 * 8 + 21 * 8 + 10 * 32 + 65
    )


def test_status_codebooks_include_finite_terminal_strings() -> None:
    assert planes.status_code(
        "terminal.recoverable_error_code", "exact_reporting_failed"
    ) == 1
    assert planes.status_code(
        "terminal.recoverable_error_code", "output_flush_failed"
    ) == 3
    assert planes.status_code("terminal.failure_stage", "birth_event_append") == 4
    assert planes.status_code("terminal.failure_stage", "api_state") == 13
    assert planes.status_value("terminal.failure_stage", 4) == "birth_event_append"
    with pytest.raises(ValueError, match="unsupported"):
        planes.status_code("terminal.failure_stage", "made_up_stage")
    with pytest.raises(ValueError, match="unknown"):
        planes.status_value("terminal.failure_stage", 0)
    with pytest.raises(TypeError):
        planes.STATUS_CODEBOOKS["world"]["flat"] = 9  # type: ignore[index]


def test_allocation_has_exact_shapes_dtypes_zeroes_and_size() -> None:
    value = planes.allocate_compact_summary_planes(3, 2)
    assert value.validity.shape == (3, 2)
    assert value.validity.dtype == np.uint8
    assert value.defined_mask.shape == (3, 2, 1)
    assert value.defined_mask.dtype == np.uint64
    assert value.status_codes.shape == (3, 2, 12)
    assert value.status_codes.dtype == np.int32
    assert value.integer_values.shape == (3, 2, 76)
    assert value.integer_values.dtype == np.int64
    assert value.float_values.shape == (3, 2, 21)
    assert value.float_values.dtype == np.float64
    assert value.digest_values.shape == (3, 2, 10, 32)
    assert value.text_values.shape == (3, 2, 1, 65)
    assert sum(item.nbytes for item in value.as_mapping().values()) == 3 * 2 * 1_218
    assert all(not np.any(item.view(np.uint8)) for item in value.as_mapping().values())
    assert planes.validate_compact_summary_planes(value) == (3, 2)


@pytest.mark.parametrize("generations,cells", [(0, 1), (1, 0), (-1, 1)])
def test_allocation_rejects_invalid_dimensions(generations: int, cells: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        planes.allocate_compact_summary_planes(generations, cells)


def test_layout_validation_is_strict_and_nonmutating() -> None:
    value = planes.allocate_compact_summary_planes(2, 1)
    before = {name: item.copy() for name, item in value.as_mapping().items()}
    assert planes.validate_compact_summary_plane_layout(value) == (2, 1)
    assert all(
        np.array_equal(item, before[name])
        for name, item in value.as_mapping().items()
    )
    with pytest.raises(ValueError, match="dtype"):
        planes.validate_compact_summary_plane_layout(
            replace(value, defined_mask=value.defined_mask.astype(np.uint32))
        )
    with pytest.raises(ValueError, match="shape"):
        planes.validate_compact_summary_plane_layout(
            replace(value, status_codes=value.status_codes[:, :, :-1])
        )
    noncontiguous = np.zeros((2, 1, 42), dtype=np.float64)[:, :, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        planes.validate_compact_summary_plane_layout(
            replace(value, float_values=noncontiguous)
        )


def test_defined_mask_covers_all_paths_and_rejects_reserved_bits() -> None:
    assert planes.DEFINED_FIELD_BITS[planes.PATCH_OBJECT_FIELD] == 17
    with pytest.raises(ValueError, match="reserved high bits"):
        planes.validate_defined_mask(1 << 32)
    with pytest.raises(ValueError, match="fit uint64"):
        planes.validate_defined_mask(-1)
    with pytest.raises(ValueError, match="Patch child bits"):
        planes.set_field_defined(0, planes.PATCH_DEFINED_FIELDS[0], True)
    mask = planes.set_field_defined(0, planes.PATCH_OBJECT_FIELD, True)
    mask = planes.set_field_defined(mask, planes.PATCH_DEFINED_FIELDS[0], True)
    assert planes.is_field_defined(mask, planes.PATCH_DEFINED_FIELDS[0])
    with pytest.raises(ValueError, match="Patch child bits"):
        planes.set_field_defined(mask, planes.PATCH_OBJECT_FIELD, False)


def test_undefined_fields_are_sorted_and_patch_object_aware() -> None:
    flat = planes.undefined_fields_from_mask(0)
    assert len(flat) == 18
    assert flat == tuple(sorted(flat))
    assert "patch_metrics" in flat
    assert not any(field.startswith("patch_metrics.") for field in flat)
    patch_mask = planes.set_field_defined(0, planes.PATCH_OBJECT_FIELD, True)
    patch = planes.undefined_fields_from_mask(patch_mask)
    assert len(patch) == 31
    assert "patch_metrics" not in patch
    assert set(planes.PATCH_DEFINED_FIELDS).issubset(patch)


def test_single_text_value_round_trip_boundary_and_no_mutation() -> None:
    text = "L" + "1" * 63
    record = planes.encode_text_value(text)
    before = record.copy()
    assert int(record[0]) == 64
    assert planes.decode_text_value(record) == text
    assert np.array_equal(record, before)
    assert record.shape == (65,)


def test_text_value_rejects_empty_overlong_invalid_utf8_and_dirty_tail() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        planes.encode_text_value("")
    with pytest.raises(ValueError, match="exceeds 64"):
        planes.encode_text_value("x" * 65)
    with pytest.raises(ValueError, match="valid UTF-8"):
        planes.encode_text_value("\ud800")
    invalid = np.zeros(65, dtype=np.uint8)
    invalid[0] = 1
    invalid[1] = 0xFF
    with pytest.raises(ValueError, match="valid UTF-8"):
        planes.decode_text_value(invalid)
    dirty = planes.encode_text_value("L1")
    dirty[-1] = 1
    with pytest.raises(ValueError, match="zero-filled"):
        planes.decode_text_value(dirty)


def test_signed_zero_is_bit_preserving() -> None:
    negative = planes.float64_to_bits(-0.0)
    positive = planes.float64_to_bits(0.0)
    assert negative != positive
    restored = planes.float64_from_bits(negative)
    assert restored == 0.0
    assert np.signbit(restored)
    value = planes.allocate_compact_summary_planes(1, 1)
    value.float_values[0, 0, 0] = restored
    assert int(value.float_values[0, 0].view(np.uint64)[0]) == negative


def _populate_minimal_valid_row(
    value: planes.CompactSummaryPlanes, *, world: str = "flat"
) -> int:
    value.validity[0, 0] = 1
    statuses = {
        "summary_status": "complete",
        "world": world,
        "turnover.status": "committed",
        "rng.authority_complete": True,
        "rng.export_non_consuming": True,
        "terminal.status": "not_terminal",
        "terminal.recoverable_error_status": "none",
        "terminal.sticky_failure": False,
        "checkpoint_due": False,
    }
    for field, item in statuses.items():
        value.status_codes[0, 0, planes.STATUS_FIELD_INDEX[field]] = planes.status_code(
            field, item
        )
    for field in planes.POSITIVE_INTEGER_FIELDS:
        value.integer_values[0, 0, planes.INTEGER_FIELD_INDEX[field]] = 1
    mask = 0
    if world == "patch":
        mask = planes.set_field_defined(mask, planes.PATCH_OBJECT_FIELD, True)
    value.defined_mask[0, 0, 0] = mask
    return mask


def test_minimal_valid_flat_row_and_validation_do_not_mutate() -> None:
    value = planes.allocate_compact_summary_planes(1, 1)
    _populate_minimal_valid_row(value)
    before = {name: item.copy() for name, item in value.as_mapping().items()}
    assert planes.validate_compact_summary_planes(value) == (1, 1)
    assert all(
        np.array_equal(item, before[name])
        for name, item in value.as_mapping().items()
    )


def test_invalid_row_requires_every_plane_to_be_bit_zero() -> None:
    value = planes.allocate_compact_summary_planes(1, 1)
    value.float_values[0, 0, 0] = -0.0
    with pytest.raises(ValueError, match="frozen no-op"):
        planes.validate_compact_summary_planes(value)


def test_valid_row_rejects_unknown_status_nonfinite_and_integer_bounds() -> None:
    value = planes.allocate_compact_summary_planes(1, 1)
    _populate_minimal_valid_row(value)
    value.status_codes[0, 0, 0] = 999
    with pytest.raises(ValueError, match="unknown"):
        planes.validate_compact_summary_planes(value)
    value.status_codes[0, 0, 0] = planes.status_code("summary_status", "complete")
    value.float_values[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        planes.validate_compact_summary_planes(value)
    value.float_values[0, 0, 0] = 0.0
    value.integer_values[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        planes.validate_compact_summary_planes(value)
    value.integer_values[0, 0, 0] = planes.INT64_MAX
    with pytest.raises(ValueError, match="cannot be incremented"):
        planes.validate_compact_summary_planes(value)


def test_patch_movement_energy_totals_allow_signed_terrain_delta() -> None:
    value = planes.allocate_compact_summary_planes(1, 1)
    mask = _populate_minimal_valid_row(value, world="patch")
    mask = planes.set_field_defined(mask, planes.PATCH_OBJECT_FIELD, True)
    for field in planes.PATCH_DEFINED_FIELDS:
        mask = planes.set_field_defined(mask, field, True)
    value.defined_mask[0, 0, 0] = mask
    value.float_values[
        0,
        0,
        planes.FLOAT_FIELD_INDEX["patch_metrics.patch_movement_energy_total"],
    ] = -0.25
    value.float_values[
        0,
        0,
        planes.FLOAT_FIELD_INDEX[
            "patch_metrics.patch_movement_energy_extra_total"
        ],
    ] = -0.25

    assert planes.validate_compact_summary_planes(value) == (1, 1)


def test_dynamic_lineage_text_requires_defined_bit_and_exact_record() -> None:
    value = planes.allocate_compact_summary_planes(1, 1)
    mask = _populate_minimal_valid_row(value)
    value.text_values[0, 0, 0] = planes.encode_text_value("lineage:custom")
    with pytest.raises(ValueError, match="undefined"):
        planes.validate_compact_summary_planes(value)
    mask = planes.set_field_defined(
        mask, "lineage_metrics.dominant_lineage_id", True
    )
    value.defined_mask[0, 0, 0] = mask
    assert planes.validate_compact_summary_planes(value) == (1, 1)


def test_recoverable_error_code_and_digest_must_be_defined_together() -> None:
    value = planes.allocate_compact_summary_planes(1, 1)
    mask = _populate_minimal_valid_row(value)
    value.status_codes[
        0, 0, planes.STATUS_FIELD_INDEX["terminal.recoverable_error_status"]
    ] = planes.status_code("terminal.recoverable_error_status", "present")
    mask = planes.set_field_defined(mask, "terminal.recoverable_error_code", True)
    value.defined_mask[0, 0, 0] = mask
    value.status_codes[
        0, 0, planes.STATUS_FIELD_INDEX["terminal.recoverable_error_code"]
    ] = planes.status_code(
        "terminal.recoverable_error_code", "exact_reporting_failed"
    )
    with pytest.raises(ValueError, match="code and digest"):
        planes.validate_compact_summary_planes(value)
    mask = planes.set_field_defined(
        mask, "terminal.recoverable_error_digest_sha256", True
    )
    value.defined_mask[0, 0, 0] = mask
    assert planes.validate_compact_summary_planes(value) == (1, 1)


def test_failed_terminal_requires_coordinates_stage_and_sticky_status() -> None:
    value = planes.allocate_compact_summary_planes(1, 1)
    mask = _populate_minimal_valid_row(value)
    value.status_codes[0, 0, planes.STATUS_FIELD_INDEX["terminal.status"]] = (
        planes.status_code("terminal.status", "failed")
    )
    with pytest.raises(ValueError, match="sticky failure"):
        planes.validate_compact_summary_planes(value)
    for field in (
        "terminal.failure_generation",
        "terminal.failure_tick",
        "terminal.failure_stage",
    ):
        mask = planes.set_field_defined(mask, field, True)
    value.defined_mask[0, 0, 0] = mask
    value.status_codes[0, 0, planes.STATUS_FIELD_INDEX["terminal.sticky_failure"]] = (
        planes.status_code("terminal.sticky_failure", True)
    )
    value.status_codes[0, 0, planes.STATUS_FIELD_INDEX["terminal.failure_stage"]] = (
        planes.status_code("terminal.failure_stage", "birth_event_append")
    )
    assert planes.validate_compact_summary_planes(value) == (1, 1)
