from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from engine.native_runtime.compact_summary_host_template import (
    SURFACE_ORDER,
    compile_flat_compact_summary_host_programs,
)
from engine.native_runtime.compact_summary_host_template_wire import (
    lower_compact_summary_host_template_bundle_to_wire,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json"
RUNTIME_HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_host_template_runtime_v1.hpp"
)
RUNTIME_SOURCE = RUNTIME_HEADER.with_suffix(".cpp")
RENDERER_SOURCE = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_host_template_renderer_v1.cpp"
)
PROBE_SOURCE = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_host_template_runtime_v1.cpp"
)


@pytest.fixture(scope="module")
def exact_g2_b1_wire() -> Any:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    fixture = Path(plan["source_contract"]["accepted_flat_projection_fixture"]["path"])
    bundle = compile_flat_compact_summary_host_programs(fixture, plan, 2, 1)
    assert bundle["template_count"] == 12
    assert [
        (template["generation"], template["cell"], template["surface"])
        for template in bundle["templates"]
    ] == [
        (generation, 0, surface)
        for generation in range(2)
        for surface in SURFACE_ORDER
    ]
    return lower_compact_summary_host_template_bundle_to_wire(
        bundle,
        torch,
        allow_unpinned_test_fallback=True,
    )


@pytest.fixture(scope="module")
def runtime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("host_template_runtime_v1") / "probe"
    completed = subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            f"-I{ROOT / 'engine/perf'}",
            str(PROBE_SOURCE),
            str(RUNTIME_SOURCE),
            str(RENDERER_SOURCE),
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return output


def _cloned_tensors(wire: Any) -> dict[str, torch.Tensor]:
    return {
        "programs": wire.programs.clone(),
        "tokens": wire.tokens.clone(),
        "parts": wire.parts.clone(),
        "constraints": wire.constraints.clone(),
        "byte_pool": wire.byte_pool.clone(),
    }


def _write_wire(root: Path, tensors: dict[str, torch.Tensor]) -> None:
    root.mkdir()
    for name, tensor in tensors.items():
        (root / f"{name}.bin").write_bytes(
            tensor.contiguous().numpy().tobytes(order="C")
        )


def _run_probe(
    probe: Path,
    tmp_path: Path,
    wire: Any,
    mutate: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    tensors = _cloned_tensors(wire)
    if mutate is not None:
        mutate(tensors)
    wire_root = tmp_path / "wire"
    _write_wire(wire_root, tensors)
    return subprocess.run(
        [str(probe), str(wire_root)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_exact_flat_g2_b1_wire_rehydrates_and_callback_renders(
    runtime_probe: Path,
    exact_g2_b1_wire: Any,
    tmp_path: Path,
) -> None:
    completed = _run_probe(runtime_probe, tmp_path, exact_g2_b1_wire)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("status=success programs=12 callbacks=12 ")


def test_nullable_float64_wire_rule_is_exact_at_all_pack_depths(
    exact_g2_b1_wire: Any,
) -> None:
    nullable = exact_g2_b1_wire.tokens[
        exact_g2_b1_wire.tokens[:, 1] == 7
    ]
    assert nullable.shape[0] > 0
    assert set(nullable[:, 4].tolist()) <= {0, 1, 2}
    assert set(nullable[:, 2].tolist()) == {5}
    assert not nullable[:, 3].any()
    assert not nullable[:, 8:13].any()


def _program_abi(tensors: dict[str, torch.Tensor]) -> None:
    tensors["programs"][0, 0] = 2


def _program_coordinate(tensors: dict[str, torch.Tensor]) -> None:
    tensors["programs"][6, 3] = 0


def _row_slice_offset(tensors: dict[str, torch.Tensor]) -> None:
    tensors["programs"][1, 5] = int(tensors["programs"][1, 5]) + 1


def _token_enum(tensors: dict[str, torch.Tensor]) -> None:
    tensors["tokens"][0, 1] = 99


def _token_pool_offset(tensors: dict[str, torch.Tensor]) -> None:
    tensors["tokens"][0, 6] = int(tensors["tokens"][0, 6]) + 1


def _invalid_path_utf8(tensors: dict[str, torch.Tensor]) -> None:
    offset = int(tensors["tokens"][0, 6])
    tensors["byte_pool"][offset] = 0xFF


def _part_reserved(tensors: dict[str, torch.Tensor]) -> None:
    tensors["parts"][0, 4] = 1


def _constraint_enum(tensors: dict[str, torch.Tensor]) -> None:
    assert tensors["constraints"].shape[0] > 0
    tensors["constraints"][0, 0] = 99


def _renderer_token_bound(tensors: dict[str, torch.Tensor]) -> None:
    tensors["tokens"][0, 5] = int(tensors["tokens"][0, 5]) + 1
    tensors["programs"][0, 12] = int(tensors["programs"][0, 12]) + 1


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (_program_abi, "status=unsupported_program_abi component=2 row=0 column=0"),
        (_program_coordinate, "status=coordinate_order component=2 row=6 column=3"),
        (_row_slice_offset, "status=row_slice_order component=2 row=1"),
        (_token_enum, "status=unsupported_token_type component=3 row=0 column=1"),
        (_token_pool_offset, "status=byte_pool_traversal component=3 row=0 column=6"),
        (_invalid_path_utf8, "status=invalid_utf8 component=3 row=0 column=6"),
        (_part_reserved, "status=reserved_field_nonzero component=4 row=0 column=4"),
        (
            _constraint_enum,
            "status=unsupported_constraint_kind component=5 row=0 column=0",
        ),
        (
            _renderer_token_bound,
            "status=renderer_rejected component=8 row=0 column=4294967295 "
            "renderer_status=11",
        ),
    ],
    ids=[
        "program-abi",
        "coordinate-order",
        "row-slice-order",
        "token-enum",
        "pool-traversal",
        "utf8",
        "reserved-zero",
        "constraint-enum",
        "renderer-declaration",
    ],
)
def test_malformed_wire_fails_closed_with_precise_status(
    runtime_probe: Path,
    exact_g2_b1_wire: Any,
    tmp_path: Path,
    mutate: Callable[[dict[str, torch.Tensor]], None],
    expected: str,
) -> None:
    completed = _run_probe(runtime_probe, tmp_path, exact_g2_b1_wire, mutate)
    assert completed.returncode == 0, completed.stderr
    assert expected in completed.stdout


def test_runtime_source_contract_is_bounded_to_wire_rehydration() -> None:
    header = RUNTIME_HEADER.read_text(encoding="utf-8")
    source = RUNTIME_SOURCE.read_text(encoding="utf-8")

    assert "prepare_host_template_runtime_v1" in header
    assert "lookup_host_template_program_v1" in header
    assert "lookup_host_template_token_storage_v1" in header
    assert "std::vector<renderer::HostTemplateTokenValueV1>" in header
    assert "std::vector<std::uint64_t> shape_dimensions_" in header
    assert "std::vector<std::uint8_t> sha256_hex_" in header
    assert "render_and_hash_host_template_v1" in source
    for forbidden in (
        "pybind11",
        "cudaLaunchHostFunc",
        "settle_compact_summary",
        "retained_authority_token_values",
    ):
        assert forbidden not in header
        assert forbidden not in source
