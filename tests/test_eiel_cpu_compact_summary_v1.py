from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pybind11
import pytest

from engine.native_runtime import compact_summary_planes as abi
from engine.native_runtime import compact_summary_projection as projection


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "engine" / "perf" / "native" / "cpu_compact_summary_v1.hpp"
SOURCE = HEADER.with_suffix(".cpp")
ABI_HEADER = (
    ROOT / "engine" / "perf" / "cuda_architecture_b_compact_summary_v2.cuh"
)
BINDING = (
    ROOT / "tests" / "perf" / "test_eiel_cpu_compact_summary_v1_binding.cpp"
)
MODULE_NAME = "eiel_cpu_compact_summary_v1_test"


@pytest.fixture(scope="module")
def native_encoder(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is unavailable")
    build = tmp_path_factory.mktemp("cpu_compact_summary_v1")
    extension = build / f"{MODULE_NAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            "-Wall",
            "-Wextra",
            f"-I{pybind11.get_include()}",
            f"-I{sysconfig.get_paths()['include']}",
            f"-I{ROOT / 'engine' / 'perf' / 'native'}",
            f"-I{ROOT / 'engine' / 'perf'}",
            str(BINDING),
            str(SOURCE),
            "-o",
            str(extension),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    spec = importlib.util.spec_from_file_location(MODULE_NAME, extension)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enum_members(header: str, enum_name: str) -> tuple[str, ...]:
    match = re.search(
        rf"enum {re.escape(enum_name)}\s*:[^{{]+\{{(?P<body>.*?)\}};",
        header,
        flags=re.DOTALL,
    )
    assert match is not None
    return tuple(
        re.findall(r"^\s*(k[A-Za-z0-9_]+)\s*(?:=[^,]+)?\s*,", match["body"], re.MULTILINE)
    )


def _field_symbol(prefix: str, field: str) -> str:
    suffix = "".join(
        token[:1].upper() + token[1:]
        for token in re.split(r"[._]", field)
    )
    return f"k{prefix}{suffix}"


def test_cpp_abi_counts_and_field_order_match_python_registry(
    native_encoder: ModuleType,
) -> None:
    header = ABI_HEADER.read_text(encoding="ascii")
    receipt = native_encoder.abi_receipt()
    plane_names = tuple(abi.allocate_compact_summary_planes(1, 1).as_mapping())

    assert _enum_members(header, "CompactSummaryPlaneV2") == (
        *(_field_symbol("Plane", name) for name in plane_names),
        "kPlaneCountV2",
    )
    for enum_name, prefix, fields, count_name in (
        ("StatusFieldV2", "Status", abi.STATUS_FIELDS, "kStatusFieldCountV2"),
        ("IntegerFieldV2", "Integer", abi.INTEGER_FIELDS, "kIntegerFieldCountV2"),
        ("FloatFieldV2", "Float", abi.FLOAT_FIELDS, "kFloatFieldCountV2"),
        ("DigestFieldV2", "Digest", abi.DIGEST_FIELDS, "kDigestFieldCountV2"),
        ("TextFieldV2", "Text", abi.TEXT_FIELDS, "kTextFieldCountV2"),
        ("DefinedBitV1", "Defined", abi.DEFINED_FIELDS, "kDefinedBitCountV1"),
    ):
        assert _enum_members(header, enum_name) == (
            *(_field_symbol(prefix, field) for field in fields),
            count_name,
        )

    assert receipt["plane_count"] == 7
    assert receipt["status_count"] == len(abi.STATUS_FIELDS)
    assert receipt["integer_count"] == len(abi.INTEGER_FIELDS)
    assert receipt["float_count"] == len(abi.FLOAT_FIELDS)
    assert receipt["digest_count"] == len(abi.DIGEST_FIELDS)
    assert receipt["text_count"] == len(abi.TEXT_FIELDS)
    assert receipt["defined_bit_count"] == len(abi.DEFINED_FIELDS)


def test_exact_fixed_plane_sizes_and_framework_free_cxx17_compile(
    native_encoder: ModuleType, tmp_path: Path
) -> None:
    receipt = native_encoder.abi_receipt()
    expected_plane_bytes = {
        "validity": 1,
        "defined_mask": 8,
        "status_codes": len(abi.STATUS_FIELDS) * 4,
        "integer_values": len(abi.INTEGER_FIELDS) * 8,
        "float_values": len(abi.FLOAT_FIELDS) * 8,
        "digest_values": len(abi.DIGEST_FIELDS) * 32,
        "text_values": len(abi.TEXT_FIELDS) * abi.TEXT_RECORD_BYTES,
    }
    assert receipt["plane_bytes"] == expected_plane_bytes
    assert receipt["raw_digest_bytes"] == 32
    assert receipt["utf8_text_value_bytes"] == abi.TEXT_RECORD_BYTES
    assert receipt["logical_row_bytes"] == abi.LOGICAL_BYTES_PER_GENERATION_CELL
    assert sum(expected_plane_bytes.values()) == abi.LOGICAL_BYTES_PER_GENERATION_CELL

    production = HEADER.read_text(encoding="ascii") + SOURCE.read_text(encoding="ascii")
    for forbidden in (
        "#include <cuda",
        "cuda_runtime",
        "#include <torch",
        "ATen/",
        "Python.h",
        "pybind11",
        "std::vector",
        "malloc(",
        "calloc(",
        "operator new",
    ):
        assert forbidden not in production

    compiler = shutil.which("g++") or shutil.which("clang++")
    assert compiler is not None
    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O2",
            "-fno-exceptions",
            "-fno-rtti",
            f"-I{ROOT / 'engine' / 'perf' / 'native'}",
            f"-I{ROOT / 'engine' / 'perf'}",
            "-c",
            str(SOURCE),
            "-o",
            str(tmp_path / "cpu_compact_summary_v1.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    implementation = SOURCE.read_text(encoding="ascii")
    encode_body = implementation.split("EncodeStatusV1 encode_completed_row(", 1)[1]
    encode_body = encode_body.split("abi::CompactSummaryViewV2", 1)[0]
    validity_write = encode_body.index("row.validity[0] = 1;")
    assert encode_body.index("row.defined_mask[0] = source.defined_mask;") < validity_write
    assert encode_body.index("row.status_codes = source.status_codes;") < validity_write
    assert encode_body.index("row.integer_values = source.integer_values;") < validity_write
    assert encode_body.index("row.float_values = source.float_values;") < validity_write
    assert encode_body.index("row.digest_values[") < validity_write
    assert encode_body.index("row.text_values[") < validity_write


_DTYPES = {
    "validity": np.dtype(np.uint8),
    "defined_mask": np.dtype(np.uint64),
    "status_codes": np.dtype(np.int32),
    "integer_values": np.dtype(np.int64),
    "float_values": np.dtype(np.float64),
    "digest_values": np.dtype(np.uint8),
    "text_values": np.dtype(np.uint8),
}


def _install_raw_row(
    target: abi.CompactSummaryPlanes,
    raw: dict[str, bytes],
    generation: int,
) -> None:
    for name, plane in target.as_mapping().items():
        destination = np.asarray(plane[generation, 0])
        decoded = np.frombuffer(raw[name], dtype=_DTYPES[name])
        assert decoded.nbytes == destination.nbytes
        plane[generation, 0] = decoded.reshape(destination.shape)


def _path(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        value = value[part]
    return value


def test_complete_flat_fixture_and_invalid_row_match_projection_raw_planes(
    native_encoder: ModuleType,
) -> None:
    output = native_encoder.fixture_output()
    assert output["status"] == "success"
    planes = abi.allocate_compact_summary_planes(2, 1)
    _install_raw_row(planes, output["completed"], 0)
    _install_raw_row(planes, output["invalid"], 1)

    assert abi.validate_compact_summary_planes(planes) == (2, 1)
    assert planes.validity[:, 0].tolist() == [1, 0]
    assert all(value == 0 for value in output["invalid"].values() for value in value)
    with pytest.raises(ValueError, match="invalid/no-op"):
        projection.project_compact_summary_row(planes, generation=1, cell=0)

    canonical = projection.project_compact_summary_row(
        planes, generation=0, cell=0
    )
    assert canonical["world"] == "flat"
    assert canonical["patch_metrics"] is None
    assert canonical["lineage_metrics"]["dominant_lineage_id"] == "L00000001"
    native_digests = np.frombuffer(
        output["completed"]["digest_values"], dtype=np.uint8
    ).reshape(len(abi.DIGEST_FIELDS), 32)
    for field, index in abi.DIGEST_FIELD_INDEX.items():
        digest = _path(canonical, field)
        if digest is None:
            assert not np.any(native_digests[index])
        else:
            assert len(digest) == 64
            assert digest == digest.lower()
            assert bytes.fromhex(digest) == native_digests[index].tobytes()

    projected_planes = abi.allocate_compact_summary_planes(1, 1)
    projection.encode_compact_summary_row(
        canonical, projected_planes, generation=0, cell=0
    )
    for name, plane in projected_planes.as_mapping().items():
        assert np.asarray(plane[0, 0]).tobytes() == output["completed"][name]

    view = output["view"]
    assert view == {
        "generation_count": 1,
        "cell_count": 1,
        "validity_stride": 1,
        "status_stride": len(abi.STATUS_FIELDS),
        "integer_stride": len(abi.INTEGER_FIELDS),
        "float_stride": len(abi.FLOAT_FIELDS),
        "digest_stride": len(abi.DIGEST_FIELDS) * 32,
        "text_stride": len(abi.TEXT_FIELDS) * abi.TEXT_RECORD_BYTES,
        "pointers_match": True,
    }


def test_malformed_sources_fail_closed_to_exact_invalid_rows(
    native_encoder: ModuleType,
) -> None:
    failures = native_encoder.validation_failures()
    assert [item["status"] for item in failures] == [
        "source_authority_version_mismatch",
        "reserved_defined_bit",
        "patch_child_without_object",
        "status_code_invalid",
        "world_patch_presence_mismatch",
        "integer_range_invalid",
        "float_non_finite",
        "float_range_invalid",
        "undefined_storage_nonzero",
        "flat_patch_storage_nonzero",
        "recoverable_state_inconsistent",
        "terminal_state_inconsistent",
        "text_length_invalid",
        "text_utf8_invalid",
        "text_tail_nonzero",
        "required_digest_missing",
    ]
    assert all(item["name"] == item["status"] for item in failures)
    assert all(item["row_bit_zero"] is True for item in failures)
