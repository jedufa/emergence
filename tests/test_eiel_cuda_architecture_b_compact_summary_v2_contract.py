from __future__ import annotations

import json
import re
from pathlib import Path

from engine.native_runtime import compact_summary_planes as host_planes


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "40_eiel_design"
HEADER = ROOT / "engine" / "perf" / "cuda_architecture_b_compact_summary_v2.cuh"
COMPILE_TEST = (
    ROOT
    / "tests"
    / "perf"
    / "test_eiel_cuda_architecture_b_compact_summary_v2_compile.cu"
)

NATIVE_V2 = DOCS / "compact_summary_native_abi_v2.json"
PLANE_LAYOUT_V2 = DOCS / "compact_summary_plane_layout_v2.json"
PROJECTION_V2 = DOCS / "compact_summary_projection_contract_v2.json"
INVALID_ROW_V2 = DOCS / "compact_summary_invalid_row_contract_v2.json"
DEFINED_MASK_V1 = DOCS / "compact_summary_defined_mask_v1.json"
NULLABLE_REGISTRY_V1 = DOCS / "compact_nullable_field_registry_v1.json"
PLANE_REGISTRY_V1 = DOCS / "compact_plane_index_registry_v1.json"
CODEBOOKS_V1 = DOCS / "compact_summary_codebooks_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _enum_members(source: str, enum_name: str) -> list[tuple[str, int]]:
    match = re.search(
        rf"\benum\s+{re.escape(enum_name)}\s*:[^{{]+\{{(?P<body>.*?)\}};",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing enum {enum_name}"
    body = re.sub(
        r"//[^\n]*|/\*.*?\*/",
        "",
        match.group("body"),
        flags=re.DOTALL,
    )
    members: list[tuple[str, int]] = []
    value = -1
    for raw_entry in body.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, raw_value = (part.strip() for part in entry.split("=", 1))
            value = int(raw_value, 0)
        else:
            name = entry
            value += 1
        members.append((name, value))
    return members


def _symbol_suffix(path: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[._]", path))


def _expected_members(
    prefix: str,
    paths: list[str],
    sentinel: str,
) -> list[tuple[str, int]]:
    members = [
        (prefix + _symbol_suffix(path), ordinal)
        for ordinal, path in enumerate(paths)
    ]
    members.append((sentinel, len(paths)))
    return members


def _version_literal(source: str, name: str) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*\[\]\s*=\s*\n?\s*\"([^\"]+)\"\s*;",
        source,
    )
    assert match is not None, f"missing version literal {name}"
    return match.group(1)


def test_v2_documents_freeze_exact_plane_counts_and_byte_math() -> None:
    native = _load(NATIVE_V2)
    layout = _load(PLANE_LAYOUT_V2)

    assert native["schema_version"] == "eiel_compact_summary_native_abi_v2"
    assert layout["native_abi_version"] == native["schema_version"]
    assert native["plane_layout_version"] == layout["schema_version"]
    assert layout["axis_order"] == ["generation", "cell", "field"]

    expected = {
        "validity": ("uint8", 1, 1),
        "defined_mask": ("uint64", 1, 8),
        "status_codes": ("int32", 12, 48),
        "integer_values": ("int64", 76, 608),
        "float_values": ("float64", 21, 168),
        "digest_values": ("uint8", 10, 320),
        "text_values": ("uint8", 1, 65),
    }
    native_planes = {plane["name"]: plane for plane in native["planes"]}
    layout_planes = {plane["name"]: plane for plane in layout["planes"]}
    assert list(native_planes) == list(expected)
    assert list(layout_planes) == list(expected)

    for name, (dtype, field_count, logical_bytes) in expected.items():
        native_plane = native_planes[name]
        layout_plane = layout_planes[name]
        assert native_plane["dtype"] == layout_plane["dtype"] == dtype
        assert layout_plane["field_count"] == field_count
        native_field_count = 32 if name == "defined_mask" else field_count
        assert native_plane.get("field_count", 1) == native_field_count
        assert native_plane["logical_bytes_per_row"] == logical_bytes

    assert sum(item[2] for item in expected.values()) == 1218
    assert native["logical_bytes_per_generation_cell"] == 1218
    assert layout["logical_bytes_per_generation_cell"] == 1218
    assert native_planes["digest_values"]["shape"][-2:] == [10, 32]
    assert native_planes["text_values"]["shape"][-2:] == [1, 65]


def test_ordered_field_enums_match_the_v1_registries_and_v2_text_decision() -> None:
    source = HEADER.read_text(encoding="utf-8")
    native_v2 = _load(NATIVE_V2)
    nullable = _load(NULLABLE_REGISTRY_V1)
    registry = _load(PLANE_REGISTRY_V1)["field_registries"]
    status_paths = [entry["path"] for entry in registry["status_codes"]]
    assert status_paths == [
        "summary_status",
        "world",
        "turnover.status",
        "rng.authority_complete",
        "rng.export_non_consuming",
        "terminal.status",
        "terminal.recoverable_error_status",
        "terminal.sticky_failure",
        "checkpoint_due",
        "lineage_metrics.lineage_collapse_warning",
        "terminal.recoverable_error_code",
        "terminal.failure_stage",
    ]
    assert native_v2["representation_decisions"]["bounded_text_fields"] == [
        registry["text_values"][0]["path"]
    ]

    expected_enums = {
        "StatusFieldV2": _expected_members(
            "kStatus", status_paths, "kStatusFieldCountV2"
        ),
        "IntegerFieldV2": _expected_members(
            "kInteger",
            [entry["path"] for entry in registry["integer_values"]],
            "kIntegerFieldCountV2",
        ),
        "FloatFieldV2": _expected_members(
            "kFloat",
            [entry["path"] for entry in registry["float_values"]],
            "kFloatFieldCountV2",
        ),
        "DigestFieldV2": _expected_members(
            "kDigest",
            [entry["path"] for entry in registry["digest_values"]],
            "kDigestFieldCountV2",
        ),
        "TextFieldV2": _expected_members(
            "kText",
            native_v2["representation_decisions"]["bounded_text_fields"],
            "kTextFieldCountV2",
        ),
        "DefinedBitV1": _expected_members(
            "kDefined",
            [subject["path"] for subject in nullable["subjects"]],
            "kDefinedBitCountV1",
        ),
    }
    for enum_name, expected_members in expected_enums.items():
        assert _enum_members(source, enum_name) == expected_members


def test_header_freezes_versions_general_mask_and_invalid_row_metadata() -> None:
    source = HEADER.read_text(encoding="utf-8")
    native = _load(NATIVE_V2)
    layout = _load(PLANE_LAYOUT_V2)
    projection = _load(PROJECTION_V2)
    invalid = _load(INVALID_ROW_V2)
    defined_mask = _load(DEFINED_MASK_V1)

    expected_versions = {
        "kNativeAbiSchemaVersionV2": native["schema_version"],
        "kWireSchemaVersionNameV1": native["wire_schema_version"],
        "kCodebookSchemaVersionV1": native["codebook_version"],
        "kDefinedMaskSchemaVersionV1": native["defined_mask_version"],
        "kPlaneLayoutSchemaVersionV2": layout["schema_version"],
        "kProjectionContractSchemaVersionV2": projection["schema_version"],
        "kInvalidRowContractSchemaVersionV2": invalid["schema_version"],
    }
    for constant, expected_value in expected_versions.items():
        assert _version_literal(source, constant) == expected_value

    assert defined_mask["dtype"] == "uint64"
    assert defined_mask["used_bit_range"] == [0, 31]
    assert defined_mask["reserved_bit_range"] == [32, 63]
    assert "0x00000000FFFFFFFFULL" in source
    assert "0xFFFFFFFF00000000ULL" in source
    assert "kReservedInvalidStatusCodeV2 = 0" in source
    assert "set_defined_bit(" in source
    assert "test_defined_bit(" in source
    assert "defined_mask_has_only_frozen_bits(" in source

    assert invalid["planes"] == {
        "validity": 0,
        "defined_mask": "all_zero",
        "status_codes": "all_reserved_invalid_code_zero",
        "integer_values": "all_zero",
        "float_values": "all_positive_zero",
        "digest_values": "all_zero_bytes",
        "text_values": "zero_length_and_zero_tail",
    }
    assert layout["validity_written_last"] is True
    assert invalid["canonical_projection_permitted"] is False
    assert invalid["default_values_never_imply_definedness"] is True
    assert projection["invalid_row"] == "must_not_project_to_canonical_wire_row"
    for member in (
        "float_is_positive_zero",
        "validity_written_last",
        "canonical_projection_permitted",
        "completed_generation_count_includes_row",
        "completed_rows_before_failure_remain_valid_and_immutable",
        "later_simulation_mutation_after_sticky_failure",
        "default_values_imply_definedness",
        "phase_a_nan_telemetry_is_separate_authority",
    ):
        assert member in source


def test_header_is_device_neutral_pod_only_and_compile_probe_launches_nothing() -> None:
    header = HEADER.read_text(encoding="utf-8")
    compile_test = COMPILE_TEST.read_text(encoding="utf-8")

    for forbidden in (
        "#include <torch/",
        "#include <ATen/",
        "at::",
        "torch::",
        "cudaMalloc",
        "cudaFree",
        "<<<",
        "__global__",
    ):
        assert forbidden not in header
        assert forbidden not in compile_test

    assert "struct StridedPlaneViewV2" in header
    assert "struct CompactSummaryViewV2" in header
    assert "std::is_trivially_copyable_v<CompactSummaryViewV2>" in header
    assert "EIEL_COMPACT_SUMMARY_V2_HOST_DEVICE" in header
    assert "row_offset(" in header
    assert "row_pointer(" in header
    assert "__device__" in compile_test


def test_host_registries_and_codebooks_match_the_frozen_json_artifacts() -> None:
    registry = _load(PLANE_REGISTRY_V1)["field_registries"]
    expected_fields = {
        "status_codes": host_planes.STATUS_FIELDS,
        "integer_values": host_planes.INTEGER_FIELDS,
        "float_values": host_planes.FLOAT_FIELDS,
        "digest_values": host_planes.DIGEST_FIELDS,
        "text_values": host_planes.TEXT_FIELDS,
    }
    for plane, expected in expected_fields.items():
        assert tuple(entry["path"] for entry in registry[plane]) == expected

    codebook_artifact = _load(CODEBOOKS_V1)
    assert tuple(entry["path"] for entry in codebook_artifact["codebooks"]) == (
        host_planes.STATUS_FIELDS
    )
    for entry in codebook_artifact["codebooks"]:
        expected = host_planes.STATUS_CODEBOOKS[entry["path"]]
        assert {
            item["wire_value"]: item["code"] for item in entry["entries"]
        } == dict(expected)
