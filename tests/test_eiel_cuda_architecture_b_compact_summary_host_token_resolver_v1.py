from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from engine.native_runtime.compact_summary_host_template import (
    compile_flat_compact_summary_host_programs,
)
from engine.native_runtime.compact_summary_checkpoint_descriptor_plan import (
    WORLD_PATCH,
    generate_compact_summary_checkpoint_descriptor_plan,
)
from engine.native_runtime.compact_summary_host_template_wire import (
    lower_compact_summary_host_template_bundle_to_wire,
)
from engine.native_runtime.compact_summary_host_token_resolver_plan import (
    SOURCE_KIND_CODES,
    compile_flat_compact_summary_host_token_resolver_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json"
HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_host_token_resolver_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
RUNTIME_SOURCE = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_host_template_runtime_v1.cpp"
)
RENDERER_SOURCE = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_host_template_renderer_v1.cpp"
)
PROBE_SOURCE = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_host_token_resolver_v1.cpp"
)

_DESCRIPTOR_SOURCE_KINDS = frozenset(
    {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        21,
        22,
        23,
        24,
        26,
        27,
    }
)


def _patch_p2_resolver_rows(flat_rows: Any, flat_plan: Any, patch_plan: Any) -> Any:
    flat_names = {
        int(row["descriptor_index"]): str(row["tensor_name"])
        for row in flat_plan["descriptors"]
    }
    patch_indices = {
        str(row["tensor_name"]): int(row["descriptor_index"])
        for row in patch_plan["descriptors"]
    }
    rows = flat_rows.clone()
    for row_index, row in enumerate(rows.tolist()):
        if int(row[1]) not in _DESCRIPTOR_SOURCE_KINDS:
            continue
        flat_index = int(row[2])
        rows[row_index, 2] = patch_indices[flat_names[flat_index]]
    return rows


@pytest.fixture(scope="module")
def real_g2_inputs() -> tuple[Any, Any, Any]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    fixture = Path(plan["source_contract"]["accepted_flat_projection_fixture"]["path"])
    if not fixture.is_file():
        pytest.skip(f"accepted Flat projection fixture is unavailable: {fixture}")
    bundle = compile_flat_compact_summary_host_programs(fixture, plan, 2, 1)
    wire = lower_compact_summary_host_template_bundle_to_wire(
        bundle,
        torch,
        allow_unpinned_test_fallback=True,
    )
    resolver_plan = compile_flat_compact_summary_host_token_resolver_plan(
        bundle,
        plan,
        torch,
        allow_unpinned_test_fallback=True,
    )
    assert tuple(resolver_plan.resolver_plan.shape) == (4992, 6)
    assert SOURCE_KIND_CODES["python_gauss_cache"] == 28
    assert set(resolver_plan.resolver_plan[:, 1].tolist()) == set(
        SOURCE_KIND_CODES.values()
    )
    patch_plan = generate_compact_summary_checkpoint_descriptor_plan(
        world=WORLD_PATCH
    )
    patch_rows = _patch_p2_resolver_rows(
        resolver_plan.resolver_plan,
        plan,
        patch_plan,
    )
    assert tuple(patch_rows.shape) == (4992, 6)
    assert sorted(
        set(
            patch_rows[
                patch_rows[:, 1] == SOURCE_KIND_CODES["policy_energy_digest"],
                2,
            ].tolist()
        )
    ) == [110]
    assert sorted(
        set(
            patch_rows[
                patch_rows[:, 1]
                == SOURCE_KIND_CODES["policy_mutation_tape_digest"],
                2,
            ].tolist()
        )
    ) == [104, 105, 106, 107, 108]
    return wire, resolver_plan, patch_rows


@pytest.fixture(scope="module")
def resolver_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("host_token_resolver_v1") / "probe"
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
            str(SOURCE),
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


def _write_tensor(path: Path, tensor: Any) -> None:
    path.write_bytes(tensor.contiguous().numpy().tobytes(order="C"))


def test_real_g2_plan_resolves_all_sources_without_allocation(
    resolver_probe: Path,
    real_g2_inputs: tuple[Any, Any, Any],
    tmp_path: Path,
) -> None:
    wire, resolver_plan, patch_rows = real_g2_inputs
    root = tmp_path / "fixture"
    root.mkdir()
    for name in ("programs", "tokens", "parts", "constraints", "byte_pool"):
        _write_tensor(root / f"{name}.bin", getattr(wire, name))
    _write_tensor(root / "resolver.bin", resolver_plan.resolver_plan)
    _write_tensor(root / "resolver_patch_p2.bin", patch_rows)

    completed = subprocess.run(
        [str(resolver_probe), str(root)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (
        "status=success source_kinds=28 resolver_rows=4992 "
        "malformed=16 patch_malformed=7 allocation_count=0\n"
    )


def test_source_contract_is_allocation_free_and_bounded() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")

    assert "resolve_compact_summary_host_tokens_v1" in header
    assert "HostTokenResolverRequestV1" in header
    assert "lookup_host_template_token_storage_v1" in source
    assert "kPatchP2DescriptorPlanV1" in source
    assert "is_supported_descriptor_count_v1" in source
    assert "Ten explicit zero streams" in source
    assert "== kPatchP2DescriptorPlanV1.descriptor_count" in source
    assert "Frozen Flat envelope constant" not in source
    for forbidden in (
        "std::vector",
        "std::string",
        "new ",
        "malloc",
        "calloc",
        "realloc",
        "pybind11",
        "cudaLaunchHostFunc",
    ):
        assert forbidden not in header
        assert forbidden not in source
