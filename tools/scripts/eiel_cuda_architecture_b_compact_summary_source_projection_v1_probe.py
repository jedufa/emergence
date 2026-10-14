#!/usr/bin/env python3
"""Build and probe the standalone native compact source-plane projection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np
import torch


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "engine"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.native_runtime import compact_summary_planes as host_abi
from engine.perf import cuda_architecture_b_compact_summary as torch_abi


CUDA_SOURCE: Final[Path] = (
    ROOT
    / "engine/perf/"
    "cuda_architecture_b_compact_summary_source_projection_v1.cu"
)
BINDING_SOURCE: Final[Path] = (
    ROOT
    / "tests/perf/"
    "eiel_cuda_architecture_b_compact_summary_source_projection_v1_binding.cpp"
)
MODULE_NAME: Final[str] = "eiel_cuda_compact_summary_source_projection_v1_probe"
GENERATIONS: Final[int] = 2
CELLS: Final[int] = 3
ROWS: Final[int] = GENERATIONS * CELLS
AUTHORITY_BYTE_WIDTH: Final[int] = 1256

ROW_COMPLETE: Final[int] = 1
ROW_FAILED_GENERATION: Final[int] = 2
ROW_STICKY_NOOP: Final[int] = 3
ROW_UNEXECUTED: Final[int] = 4

STATUS_SUCCESS: Final[int] = 0
STATUS_FAILED_GENERATION_NOOP: Final[int] = 1
STATUS_STICKY_NOOP: Final[int] = 2
STATUS_UNEXECUTED_NOOP: Final[int] = 3
STATUS_INDEX_INVALID: Final[int] = 7
STATUS_CODE_INVALID: Final[int] = 11
STATUS_UNDEFINED_STORAGE: Final[int] = 12
STATUS_FLOAT_NONFINITE: Final[int] = 14

AUTHORITY_DTYPE: Final[np.dtype[Any]] = np.dtype(
    [
        ("source_version", np.uint32),
        ("disposition", np.uint32),
        ("generation_index", np.int64),
        ("cell_index", np.int64),
        ("validity", np.uint8),
        ("header_reserved", np.uint8, (7,)),
        ("defined_mask", np.uint64),
        ("status_codes", np.int32, (12,)),
        ("integer_values", np.int64, (76,)),
        ("float_values", np.float64, (21,)),
        ("digest_values", np.uint8, (320,)),
        ("text_values", np.uint8, (65,)),
        ("tail_reserved", np.uint8, (7,)),
    ],
    align=False,
)
assert AUTHORITY_DTYPE.itemsize == AUTHORITY_BYTE_WIDTH

NULLABLE_INTEGER_BITS: Final[dict[int, int]] = {
    30: 0,
    31: 1,
    34: 4,
    35: 5,
    69: 7,
    70: 14,
    71: 16,
    72: 17,
    73: 18,
    74: 19,
    75: 20,
}
NULLABLE_FLOAT_BITS: Final[dict[int, int]] = {
    1: 9,
    2: 10,
    3: 11,
    4: 12,
    5: 13,
    6: 17,
    7: 17,
    8: 17,
    9: 17,
    10: 21,
    11: 22,
    12: 23,
    13: 24,
    14: 25,
    15: 26,
    16: 27,
    17: 28,
    18: 29,
    19: 30,
    20: 31,
}

_MODULE_CACHE: dict[Path, ModuleType] = {}


def build_extension(build_directory: Path, *, verbose: bool = False) -> ModuleType:
    """Compile the standalone sm_75 test extension."""

    if not torch.cuda.is_available():
        raise RuntimeError("an available CUDA device is required")
    build_directory = build_directory.expanduser().resolve()
    cached = _MODULE_CACHE.get(build_directory)
    if cached is not None:
        return cached
    build_directory.mkdir(parents=True, exist_ok=True)
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    previous_jobs = os.environ.get("MAX_JOBS")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["MAX_JOBS"] = "1"
    try:
        from torch.utils.cpp_extension import load

        module = load(
            name=MODULE_NAME,
            sources=[str(BINDING_SOURCE), str(CUDA_SOURCE)],
            extra_include_paths=[str(ROOT / "engine/perf")],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"],
            build_directory=str(build_directory),
            verbose=verbose,
            with_cuda=True,
            is_python_module=True,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        if previous_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous_jobs
    _MODULE_CACHE[build_directory] = module
    return module


def _set_valid_payload(
    row: np.void,
    *,
    generation: int,
    cell: int,
    defined_mask: int,
    world_code: int,
    seed: int,
) -> None:
    row["source_version"] = 1
    row["disposition"] = ROW_COMPLETE
    row["generation_index"] = generation
    row["cell_index"] = cell
    row["validity"] = 1
    row["defined_mask"] = defined_mask

    statuses = row["status_codes"]
    statuses[:] = [1, world_code, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0]
    if defined_mask & (1 << 15):
        statuses[9] = 1
    if defined_mask & (1 << 2):
        statuses[10] = 1
    if defined_mask & (1 << 6):
        statuses[11] = 1

    integers = row["integer_values"]
    integers[:] = np.arange(76, dtype=np.int64) + seed + 1
    integers[0] = generation
    integers[1] = 0
    integers[2] = generation
    integers[3] = seed + 1
    integers[4] = generation + 1
    integers[5] = 0
    integers[12] = 1
    for index, bit in NULLABLE_INTEGER_BITS.items():
        if not defined_mask & (1 << bit):
            integers[index] = 0

    floats = row["float_values"]
    floats.fill(0.25)
    floats[0] = -0.0 if seed % 2 == 0 else 0.0
    for index, bit in NULLABLE_FLOAT_BITS.items():
        if not defined_mask & (1 << bit):
            floats[index] = 0.0

    digests = row["digest_values"].reshape(10, 32)
    digest_seed = np.arange(32, dtype=np.uint8)
    for index in range(10):
        digests[index] = digest_seed + np.uint8(seed + index)
    if not defined_mask & (1 << 3):
        digests[4].fill(0)

    text_record = row["text_values"]
    text_record.fill(0)
    if defined_mask & (1 << 8):
        payload = f"lineage-{seed}-\N{GREEK CAPITAL LETTER OMEGA}".encode("utf-8")
        text_record[0] = len(payload)
        text_record[1 : len(payload) + 1] = np.frombuffer(payload, dtype=np.uint8)


def make_generation_major_fixture() -> np.ndarray:
    """Create g2/b3 complete, nullable, malformed, and sticky-tail rows."""

    authority = np.zeros(ROWS, dtype=AUTHORITY_DTYPE)
    for flat_index in range(ROWS):
        generation, cell = divmod(flat_index, CELLS)
        _set_valid_payload(
            authority[flat_index],
            generation=generation,
            cell=cell,
            defined_mask=0xFFFFFFFF,
            world_code=2,
            seed=flat_index + 1,
        )

    _set_valid_payload(
        authority[1],
        generation=0,
        cell=1,
        defined_mask=0,
        world_code=1,
        seed=2,
    )
    authority[3]["status_codes"][1] = 99

    for row_index, disposition in (
        (4, ROW_STICKY_NOOP),
        (5, ROW_UNEXECUTED),
    ):
        authority[row_index]["disposition"] = disposition
        authority[row_index]["validity"] = 0
        authority[row_index]["status_codes"].fill(99)
        authority[row_index]["integer_values"].fill(-1)
        authority[row_index]["float_values"].fill(np.nan)
        authority[row_index]["digest_values"].fill(0xA5)
        authority[row_index]["text_values"].fill(0x5A)
    return authority


def _authority_bytes(authority: np.ndarray, device: torch.device) -> torch.Tensor:
    raw = np.ascontiguousarray(authority.view(np.uint8).reshape(len(authority), -1))
    return torch.from_numpy(raw).to(device=device)


def _allocate_prefilled_output(
    device: torch.device,
) -> torch_abi.CompactSummaryTensors:
    output = torch_abi.allocate_compact_summary_tensors(
        GENERATIONS,
        CELLS,
        device=device,
    )
    for tensor in output.as_mapping().values():
        tensor.fill_(1)
    return output


def _to_host_planes(
    output: torch_abi.CompactSummaryTensors,
) -> host_abi.CompactSummaryPlanes:
    values = {
        name: tensor.detach().to(device="cpu", copy=True).contiguous().numpy()
        for name, tensor in output.as_mapping().items()
    }
    return host_abi.CompactSummaryPlanes(**values)


def project_authority(
    module: ModuleType,
    authority: np.ndarray,
) -> tuple[host_abi.CompactSummaryPlanes, np.ndarray]:
    device = torch.device("cuda", torch.cuda.current_device())
    authority_device = _authority_bytes(authority, device)
    output = _allocate_prefilled_output(device)
    producer_status = torch.zeros(
        (GENERATIONS, CELLS),
        dtype=torch.int32,
        device=device,
    )
    module.project_into(
        authority_device,
        list(output.as_mapping().values()),
        producer_status,
    )
    return (
        _to_host_planes(output),
        producer_status.detach().to(device="cpu", copy=True).numpy(),
    )


def expected_planes(authority: np.ndarray) -> host_abi.CompactSummaryPlanes:
    expected = host_abi.allocate_compact_summary_planes(GENERATIONS, CELLS)
    for flat_index in (0, 1, 2):
        generation, cell = divmod(flat_index, CELLS)
        row = authority[flat_index]
        expected.validity[generation, cell] = row["validity"]
        expected.defined_mask[generation, cell, 0] = row["defined_mask"]
        expected.status_codes[generation, cell] = row["status_codes"]
        expected.integer_values[generation, cell] = row["integer_values"]
        expected.float_values[generation, cell] = row["float_values"]
        expected.digest_values[generation, cell] = row["digest_values"].reshape(10, 32)
        expected.text_values[generation, cell] = row["text_values"].reshape(1, 65)
    return expected


def _byte_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left.view(np.uint8), right.view(np.uint8)))


def _planes_byte_exact(
    actual: host_abi.CompactSummaryPlanes,
    expected: host_abi.CompactSummaryPlanes,
) -> dict[str, bool]:
    return {
        name: _byte_exact(actual.as_mapping()[name], expected.as_mapping()[name])
        for name in actual.as_mapping()
    }


def _row_is_bit_zero(
    planes: host_abi.CompactSummaryPlanes,
    row_index: int,
) -> bool:
    generation, cell = divmod(row_index, CELLS)
    return all(
        not np.any(values[generation, cell].view(np.uint8))
        for values in planes.as_mapping().values()
    )


def _validation_case(
    module: ModuleType,
    mutate: Any,
    *,
    row_index: int,
    expected_status: int,
) -> dict[str, Any]:
    authority = make_generation_major_fixture()
    mutate(authority[row_index])
    actual, statuses = project_authority(module, authority)
    generation, cell = divmod(row_index, CELLS)
    return {
        "producer_status": int(statuses[generation, cell]),
        "expected_status": expected_status,
        "status_exact": int(statuses[generation, cell]) == expected_status,
        "bit_zero": _row_is_bit_zero(actual, row_index),
    }


def run_probe(module: ModuleType) -> dict[str, Any]:
    """Run exact g2/b3 projection and fail-closed malformed variants."""

    authority = make_generation_major_fixture()
    actual, statuses = project_authority(module, authority)
    expected = expected_planes(authority)
    plane_exact = _planes_byte_exact(actual, expected)
    expected_statuses = np.asarray(
        [
            [STATUS_SUCCESS, STATUS_SUCCESS, STATUS_SUCCESS],
            [STATUS_CODE_INVALID, STATUS_STICKY_NOOP, STATUS_UNEXECUTED_NOOP],
        ],
        dtype=np.int32,
    )

    validations = {
        "nonfinite_float": _validation_case(
            module,
            lambda row: row["float_values"].__setitem__(0, np.inf),
            row_index=2,
            expected_status=STATUS_FLOAT_NONFINITE,
        ),
        "invalid_index": _validation_case(
            module,
            lambda row: row.__setitem__("cell_index", 99),
            row_index=2,
            expected_status=STATUS_INDEX_INVALID,
        ),
        "undefined_storage": _validation_case(
            module,
            lambda row: row["integer_values"].__setitem__(30, 7),
            row_index=1,
            expected_status=STATUS_UNDEFINED_STORAGE,
        ),
        "failed_generation": _validation_case(
            module,
            lambda row: (
                row.__setitem__("disposition", ROW_FAILED_GENERATION),
                row.__setitem__("validity", 0),
            ),
            row_index=2,
            expected_status=STATUS_FAILED_GENERATION_NOOP,
        ),
    }
    validations_exact = all(
        case["status_exact"] and case["bit_zero"]
        for case in validations.values()
    )
    status_exact = bool(np.array_equal(statuses, expected_statuses))
    invalid_rows_bit_zero = all(_row_is_bit_zero(actual, row) for row in (3, 4, 5))
    negative_zero_bits = actual.float_values[0, 1, 0].view(np.uint64).item()
    all_exact = (
        all(plane_exact.values())
        and status_exact
        and invalid_rows_bit_zero
        and negative_zero_bits == 0x8000000000000000
        and validations_exact
    )
    return {
        "schema_version": (
            "eiel_cuda_architecture_b_compact_summary_source_projection_v1_probe_v1"
        ),
        "all_exact": all_exact,
        "fixture": "generation_major_g2_b3",
        "shape": [GENERATIONS, CELLS],
        "authority_byte_width": AUTHORITY_BYTE_WIDTH,
        "source_plane_logical_bytes": host_abi.LOGICAL_BYTES_PER_GENERATION_CELL,
        "plane_byte_exact": plane_exact,
        "producer_status": statuses.tolist(),
        "producer_status_exact": status_exact,
        "null_defined_mask_exact": int(actual.defined_mask[0, 1, 0]) == 0,
        "negative_zero_preserved": negative_zero_bits == 0x8000000000000000,
        "invalid_rows_bit_zero": invalid_rows_bit_zero,
        "validations": validations,
        "contract_receipt": dict(module.contract_receipt()),
        "resource_receipt": dict(module.resource_receipt()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=Path("/tmp/eiel_cuda_compact_summary_source_projection_v1"),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    module = build_extension(args.build_directory, verbose=args.verbose)
    receipt = run_probe(module)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
