from __future__ import annotations

import importlib.util
import runpy
import shutil
import subprocess
import sysconfig
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pybind11
import pytest

from engine.native_runtime import compact_summary_cpu as cpu
from engine.native_runtime import compact_summary_planes as abi
from engine.native_runtime import compact_summary_projection as projection


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/native/cpu_compact_summary_source_composer_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
ENCODER_SOURCE = ROOT / "engine/perf/native/cpu_compact_summary_v1.cpp"
BINDING = (
    ROOT
    / "tests/perf/"
    "test_eiel_cpu_compact_summary_source_composer_v1_binding.cpp"
)
MODULE_NAME = "eiel_cpu_compact_summary_source_composer_v1_test"


@pytest.fixture(scope="module")
def native_composer(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++17 compiler is unavailable")
    build = tmp_path_factory.mktemp("cpu_compact_summary_source_composer_v1")
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
            f"-I{ROOT / 'engine/perf/native'}",
            f"-I{ROOT / 'engine/perf'}",
            str(BINDING),
            str(SOURCE),
            str(ENCODER_SOURCE),
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


@pytest.fixture(scope="module")
def certified_source_builder() -> Callable[..., dict[str, Any]]:
    namespace = runpy.run_path(str(ROOT / "tests/test_eiel_compact_summary_cpu.py"))
    return namespace["_source"]


_DTYPES = {
    "validity": np.dtype(np.uint8),
    "defined_mask": np.dtype(np.uint64),
    "status_codes": np.dtype(np.int32),
    "integer_values": np.dtype(np.int64),
    "float_values": np.dtype(np.float64),
    "digest_values": np.dtype(np.uint8),
    "text_values": np.dtype(np.uint8),
}


def _expected_planes(row: dict[str, Any]) -> abi.CompactSummaryPlanes:
    planes = abi.allocate_compact_summary_planes(1, 1)
    projection.encode_compact_summary_row(
        row, planes, generation=0, cell=0
    )
    return planes


def _raw_row(planes: abi.CompactSummaryPlanes) -> dict[str, bytes]:
    return {
        name: np.asarray(plane[0, 0]).tobytes()
        for name, plane in planes.as_mapping().items()
    }


def _digest_authorities(planes: abi.CompactSummaryPlanes) -> list[bytes]:
    return [
        np.asarray(planes.digest_values[0, 0, index]).tobytes()
        for index in range(len(abi.DIGEST_FIELDS))
    ]


def _install_raw_row(raw: dict[str, bytes]) -> abi.CompactSummaryPlanes:
    planes = abi.allocate_compact_summary_planes(1, 1)
    for name, target in planes.as_mapping().items():
        destination = np.asarray(target[0, 0])
        decoded = np.frombuffer(raw[name], dtype=_DTYPES[name])
        assert decoded.nbytes == destination.nbytes
        target[0, 0] = decoded.reshape(destination.shape)
    return planes


@pytest.mark.parametrize(
    ("world", "lineage_defined"),
    [
        ("flat", True),
        ("patch", True),
        ("flat", False),
    ],
)
def test_exact_planes_and_decoded_row_match_certified_cpu_builder(
    native_composer: ModuleType,
    certified_source_builder: Callable[..., dict[str, Any]],
    world: str,
    lineage_defined: bool,
) -> None:
    source = certified_source_builder(world=world)
    if not lineage_defined:
        source["lineage"].update(
            active_lineages=None,
            dominant_lineage_id=None,
            dominant_lineage_population_share=None,
            top3_lineage_population_share=None,
            lineage_population_entropy=None,
            lineage_reproduction_entropy=None,
            dominant_lineage_reproduction_share=None,
            rare_persistent_lineages=None,
            lineage_collapse_warning=None,
            lineage_collapse_streak=None,
        )

    expected_row = cpu.build_compact_summary_row(source)
    expected_planes = _expected_planes(expected_row)
    native = native_composer.compose_fixture(
        world == "patch",
        lineage_defined,
        _digest_authorities(expected_planes),
    )

    assert native["status"] == "success"
    assert native["row"] == _raw_row(expected_planes)
    native_planes = _install_raw_row(native["row"])
    assert abi.validate_compact_summary_planes(native_planes) == (1, 1)
    decoded = projection.project_compact_summary_row(
        native_planes, generation=0, cell=0
    )
    assert decoded == expected_row


def test_malformed_missing_digest_and_terminal_inputs_fail_closed(
    native_composer: ModuleType,
) -> None:
    failures = native_composer.failure_cases()
    assert [case["name"] for case in failures] == [
        "input_version",
        "lifecycle_cursor",
        "identity_capacity",
        "missing_lifecycle_digest",
        "missing_state_digest",
        "recoverable_terminal_mismatch",
        "failed_terminal_mismatch",
        "host_canonical_failure",
    ]
    assert [case["row_status"] for case in failures] == [
        "input_version_mismatch",
        "lifecycle_authority_mismatch",
        "identity_authority_mismatch",
        "encoder_rejected",
        "encoder_rejected",
        "terminal_authority_mismatch",
        "terminal_authority_mismatch",
        "host_canonical_authority_mismatch",
    ]
    for case in failures:
        assert case["source_status"] == case["row_status"]
        assert case["source_bit_zero"] is True
        assert case["row_bit_zero"] is True


def test_framework_free_allocation_free_strict_cxx17_compile(
    native_composer: ModuleType,
    tmp_path: Path,
) -> None:
    receipt = native_composer.abi_receipt()
    assert receipt["input_bytes"] > 0
    assert receipt["input_standard_layout"] is True
    assert receipt["input_trivially_copyable"] is True
    assert receipt["plane_count"] == 7

    production = HEADER.read_text(encoding="ascii") + SOURCE.read_text(
        encoding="ascii"
    )
    for forbidden in (
        "#include <cuda",
        "cuda_runtime",
        "#include <torch",
        "ATen/",
        "Python.h",
        "pybind11",
        "std::vector",
        "std::string",
        "std::map",
        "std::unordered_map",
        "malloc(",
        "calloc(",
        "operator new",
    ):
        assert forbidden not in production
    assert "encoder::encode_completed_row(source, row)" in production

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
            f"-I{ROOT / 'engine/perf/native'}",
            f"-I{ROOT / 'engine/perf'}",
            "-c",
            str(SOURCE),
            "-o",
            str(tmp_path / "cpu_compact_summary_source_composer_v1.o"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
