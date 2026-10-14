from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from types import ModuleType

import pytest
import torch

from native_runtime.lifecycle_events import (
    FIELD_DTYPES,
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_reference_contexts,
)


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_lifecycle_digest_v1.cuh"
)
CUDA_SOURCE = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_lifecycle_digest_v1.cu"
)
BINDING_SOURCE = (
    ROOT
    / "tests/perf/test_eiel_cuda_architecture_b_compact_summary_lifecycle_digest_v1_binding.cpp"
)
ENGINE_PERF = ROOT / "engine/perf"
FIELD_ORDER = tuple(sorted(FIELD_DTYPES))
EMPTY_PREFIX_DIGEST = "7a3adbac68eed07450028e3678677caab10fd56a98307ec5319904faf203b280"


def _contexts(count: int, *, identity_base: int, generation: int) -> list[dict]:
    contexts: list[dict] = []
    for index in range(count):
        actor = identity_base + index * 3
        child = actor + 1
        common = {
            "generation": generation,
            "tick": index,
            "lineage_founder_id": identity_base,
            "lineage_depth": index + 1,
        }
        kind = index % 4
        if kind == 0:
            contexts.append(
                {
                    **common,
                    "event_type": "birth_success",
                    "parent_id": actor,
                    "child_id": child,
                    "parent_slot": index + 2,
                    "child_slot": index + 20,
                    "child_energy": 12.5 + index,
                }
            )
        elif kind == 1:
            contexts.append(
                {
                    **common,
                    "event_type": "mutation",
                    "actor_agent_id": child,
                    "target_agent_id": child,
                    "parent_agent_id": actor,
                    "child_id": child,
                    "child_slot": index + 20,
                    "mutation_delta": -0.125 * index,
                }
            )
        elif kind == 2:
            contexts.append(
                {
                    **common,
                    "event_type": "birth_fail_no_space",
                    "parent_id": actor,
                    "parent_slot": index + 2,
                }
            )
        else:
            contexts.append(
                {
                    **common,
                    "event_type": "death_committed",
                    "actor_agent_id": actor,
                    "target_agent_id": -1,
                    "actor_slot": index + 2,
                    "reason_code": 1 if index % 2 else 2,
                    "value": 1.0 if index % 2 else 2.0,
                }
            )
    return contexts


def _fixture_fields(
    counts: list[int],
    *,
    capacity: int,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], list[str]]:
    event_sets = [
        typed_lifecycle_events_from_reference_contexts(
            _contexts(count, identity_base=1000 * (cell + 1), generation=cell + 3)
        )
        for cell, count in enumerate(counts)
    ]
    matrices: dict[str, torch.Tensor] = {}
    for name in FIELD_ORDER:
        dtype = FIELD_DTYPES[name]
        if dtype.is_floating_point:
            matrix = torch.full((len(counts), capacity), float("nan"), dtype=dtype)
        else:
            matrix = torch.full((len(counts), capacity), 73, dtype=dtype)
        for cell, events in enumerate(event_sets):
            count = counts[cell]
            if count:
                matrix[cell, :count].copy_(events[name])
        matrices[name] = matrix
    expected = [
        typed_lifecycle_event_digest(
            {
                name: matrices[name][cell, :count].contiguous()
                for name in FIELD_ORDER
            }
        )
        for cell, count in enumerate(counts)
    ]
    return [matrices[name].cuda() for name in FIELD_ORDER], matrices, expected


def _build_extension(build_directory: Path) -> ModuleType:
    from torch.utils.cpp_extension import load

    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    old_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["MAX_JOBS"] = "1"
    try:
        return load(
            name="eiel_compact_summary_lifecycle_digest_v1_test",
            sources=[str(BINDING_SOURCE), str(CUDA_SOURCE)],
            extra_include_paths=[str(ENGINE_PERF)],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
            build_directory=str(build_directory),
            verbose=False,
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch
        if old_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = old_jobs


@pytest.fixture(scope="module")
def module(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        pytest.skip("CUDA compiler and device are required")
    return _build_extension(tmp_path_factory.mktemp("lifecycle_digest_v1_build"))


def test_contract_is_raw_async_allocation_free_and_lexicographic() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    combined = f"{header}\n{source}"
    assert '#include "cuda_architecture_b_sha256_v1.cuh"' in source
    assert "struct FlatTypedLifecycleColumnsV1" in header
    assert "const std::int64_t* event_cursor" in header
    assert "LifecycleDigestOutputsV1" in header
    assert "query_compact_summary_lifecycle_digest_v1_resources" in header
    assert "outputs.generation_index * outputs.cell_count + cell" in source
    assert re.search(
        r"\b(?:ATen|Torch|torch|Python|malloc|calloc|new|cudaMalloc|cudaMemcpy|cudaMemset|cudaDeviceSynchronize|cudaStreamSynchronize)\b",
        combined,
    ) is None

    digest_section = source.split("// FIELD_DTYPES lexicographic name order", 1)[1]
    positions = [digest_section.index(f'&context, "{name}"') for name in FIELD_ORDER]
    assert positions == sorted(positions)
    assert FIELD_ORDER == (
        "actor_agent_id",
        "actor_slot",
        "child_agent_id",
        "event_type",
        "generation",
        "lineage_depth",
        "lineage_founder_id",
        "parent_agent_id",
        "reason_code",
        "sequence",
        "subtype_code",
        "success",
        "target_agent_id",
        "target_slot",
        "tick",
        "value",
    )


def test_multiple_real_prefix_counts_match_python_and_generation_rows_are_isolated(
    module: ModuleType,
) -> None:
    generation_zero_counts = [0, 1, 3, 5]
    fields_zero, _matrices_zero, expected_zero = _fixture_fields(
        generation_zero_counts,
        capacity=6,
    )
    cursor_zero = torch.tensor(generation_zero_counts, dtype=torch.int64, device="cuda")
    digests = torch.full((2, 4, 32), 0xA5, dtype=torch.uint8, device="cuda")
    status = torch.full((2, 4), -99, dtype=torch.int32, device="cuda")
    validity = torch.zeros((2, 4), dtype=torch.uint8, device="cuda")

    module.digest_into(
        fields_zero,
        cursor_zero,
        0,
        digests,
        status,
        validity,
    )
    actual_zero = [bytes(row.cpu().tolist()).hex() for row in digests[0]]
    assert actual_zero == expected_zero
    assert actual_zero[0] == EMPTY_PREFIX_DIGEST
    assert status[0].cpu().tolist() == [module.status_success] * 4
    assert validity[0].cpu().tolist() == [1, 1, 1, 1]
    generation_zero_snapshot = digests[0].clone()

    generation_one_counts = [2, 4, 6, 3]
    fields_one, _matrices_one, expected_one = _fixture_fields(
        generation_one_counts,
        capacity=6,
    )
    cursor_one = torch.tensor(generation_one_counts, dtype=torch.int64, device="cuda")
    module.digest_into(
        fields_one,
        cursor_one,
        1,
        digests,
        status,
        validity,
    )

    assert torch.equal(digests[0], generation_zero_snapshot)
    assert [bytes(row.cpu().tolist()).hex() for row in digests[1]] == expected_one
    assert status.cpu().tolist() == [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert validity.cpu().tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]


@pytest.mark.parametrize(
    ("fault", "expected_status"),
    [
        ("negative_capacity", "status_invalid_capacity"),
        ("negative_count", "status_count_out_of_range"),
        ("count_above_capacity", "status_count_out_of_range"),
        ("null_column", "status_null_column"),
        ("sequence_gap", "status_sequence_mismatch"),
        ("nan_value", "status_nonfinite_value"),
        ("infinite_value", "status_nonfinite_value"),
        ("generation_reversal", "status_row_order_invalid"),
        ("tick_reversal", "status_row_order_invalid"),
    ],
)
def test_invalid_prefixes_fail_closed(
    module: ModuleType,
    fault: str,
    expected_status: str,
) -> None:
    fields, _matrices, _expected = _fixture_fields([4], capacity=4)
    cursor = torch.tensor([4], dtype=torch.int64, device="cuda")
    capacity_override = -2
    null_field_index = -1
    if fault == "negative_capacity":
        capacity_override = -1
    elif fault == "negative_count":
        cursor[0] = -1
    elif fault == "count_above_capacity":
        cursor[0] = 5
    elif fault == "null_column":
        null_field_index = FIELD_ORDER.index("value")
    elif fault == "sequence_gap":
        fields[FIELD_ORDER.index("sequence")][0, 2] = 9
    elif fault == "nan_value":
        fields[FIELD_ORDER.index("value")][0, 1] = float("nan")
    elif fault == "infinite_value":
        fields[FIELD_ORDER.index("value")][0, 1] = float("inf")
    elif fault == "generation_reversal":
        fields[FIELD_ORDER.index("generation")][0, 2] = -1
    elif fault == "tick_reversal":
        fields[FIELD_ORDER.index("tick")][0, 2] = -1
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(f"unknown fault: {fault}")

    digests = torch.full((1, 1, 32), 0xA5, dtype=torch.uint8, device="cuda")
    status = torch.full((1, 1), -99, dtype=torch.int32, device="cuda")
    validity = torch.full((1, 1), 7, dtype=torch.uint8, device="cuda")
    module.digest_into(
        fields,
        cursor,
        0,
        digests,
        status,
        validity,
        capacity_override,
        null_field_index,
    )

    assert validity.item() == 0
    assert status.item() == getattr(module, expected_status)
    assert torch.count_nonzero(digests).item() == 0


def test_resource_query_reports_sm75_kernel_contract(module: ModuleType) -> None:
    receipt = module.resource_receipt()
    assert module.contract_version == (
        "eiel_cuda_architecture_b_compact_summary_lifecycle_digest_v1"
    )
    assert receipt["binary_version"] == 75
    assert receipt["max_threads_per_block"] >= 128
    assert receipt["registers_per_thread"] > 0
    assert receipt["occupancy_blocks_per_multiprocessor"] > 0
    assert receipt["constant_bytes"] >= 0
    assert receipt["local_bytes_per_thread"] >= 0
    assert receipt["shared_bytes_per_block"] == 0
