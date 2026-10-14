from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from engine.native_runtime.compact_summary_checkpoint_descriptor_plan import (
    WORLD_PATCH,
    generate_compact_summary_checkpoint_descriptor_plan,
)
from engine.perf import cuda_architecture_b_compact_summary_checkpoint_runtime as runtime_v1


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/cuda_architecture_b_compact_summary_checkpoint_runtime_v1.hpp"
)

_CPP_COLUMNS = (
    "kDataTensorIndexV1",
    "kAliveMaskTensorIndexV1",
    "kActiveLengthTensorIndexV1",
    "kCellStrideBytesV1",
    "kBytesPerCellV1",
    "kRowCapacityV1",
    "kRowBytesV1",
    "kAliveMaskCellStrideV1",
    "kDtypeBytesV1",
    "kCanonicalizationModeV1",
    "kRawOutputIndexV1",
    "kCanonicalOutputIndexV1",
    "kDescriptorIndexV1",
    "kDescriptorSpecWidthV1",
)
_PYTHON_COLUMNS = (
    "data_tensor_index",
    "alive_mask_tensor_index",
    "active_length_tensor_index",
    "cell_stride_bytes",
    "bytes_per_cell",
    "row_capacity",
    "row_bytes",
    "alive_mask_cell_stride",
    "dtype_bytes",
    "canonicalization_mode",
    "raw_output_index",
    "canonical_output_index",
    "descriptor_index",
)


class _NoValueReadsArray(np.ndarray):
    def __getitem__(self, key: Any) -> Any:
        raise AssertionError(f"source tensor value read through {key!r}")


@pytest.fixture(scope="module")
def ready_plan() -> dict[str, Any]:
    plan = generate_compact_summary_checkpoint_descriptor_plan()
    assert plan["status"] == runtime_v1.READY_FLAT_PLAN_STATUS
    assert plan["counts"]["descriptors"] == 132
    return plan


@pytest.fixture(scope="module")
def ready_patch_p2_plan() -> dict[str, Any]:
    plan = generate_compact_summary_checkpoint_descriptor_plan(world=WORLD_PATCH)
    assert plan["status"] == runtime_v1.READY_PATCH_P2_PLAN_STATUS
    assert plan["source_contract"]["descriptor_world"] == WORLD_PATCH
    assert plan["source_contract"]["patch_profile"] == "P2"
    assert plan["counts"]["descriptors"] == 144
    return plan


def _symbol_sizes(
    plan: dict[str, Any],
    *,
    cells: int,
) -> dict[str, int]:
    names = {
        symbol
        for descriptor in plan["descriptors"]
        for symbol in descriptor["resident_shape_symbols"]
    }
    values = {name: 2 for name in names}
    values.update(
        {
            "cell": cells,
            "capacity": 3,
            "event_capacity": 4,
            "packed_torch_rng_state_bytes": 2560,
            "policy_birth_tape_capacity": 4,
        }
    )
    return values


def _sources(
    plan: dict[str, Any],
    *,
    cells: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    sizes = _symbol_sizes(plan, cells=cells)
    resident: dict[str, np.ndarray] = {}
    policy: dict[str, np.ndarray] = {}
    for descriptor in plan["descriptors"]:
        shape = tuple(sizes[name] for name in descriptor["resident_shape_symbols"])
        value = np.zeros(shape, dtype=np.dtype(descriptor["dtype"]))
        target = (
            resident
            if descriptor["source_scope"] == "checkpoint_projection"
            else policy
        )
        physical_name = descriptor["physical_source_name"]
        previous = target.get(physical_name)
        if previous is None:
            target[physical_name] = value
        else:
            assert previous.shape == value.shape
            assert previous.dtype == value.dtype

    resident.update(
        {
            "active_count_state": np.zeros((cells,), dtype=np.int64),
            "event_active_count": np.zeros((cells,), dtype=np.int64),
            "slot_free_queue_count": np.zeros((cells,), dtype=np.int64),
        }
    )
    policy["policy_birth_tape_cursor"] = np.zeros((cells,), dtype=np.int64)
    resident["energy"] = resident["energy"].view(_NoValueReadsArray)
    for value in (*resident.values(), *policy.values()):
        value.setflags(write=False)
    return resident, policy, sizes


def _descriptor_index(plan: dict[str, Any], name: str) -> int:
    return next(
        row["descriptor_index"]
        for row in plan["descriptors"]
        if row["tensor_name"] == name
    )


def _spec_tensor(
    bound: runtime_v1.CompactSummaryCheckpointRuntimeV1,
    descriptor_index: int,
    column: int,
) -> Any:
    tensor_index = int(bound.descriptor_specs[descriptor_index, column])
    return bound.tensor_table[tensor_index]


def test_python_descriptor_spec_registry_matches_parent_header_exactly() -> None:
    text = HEADER.read_text(encoding="utf-8")
    contract = re.search(r'kContractVersionV1\[\]\s*=\s*\n\s*"([^"]+)"', text)
    enum = re.search(
        r"enum DescriptorSpecColumnV1\s*:\s*std::int64_t\s*\{(.*?)\};",
        text,
        flags=re.DOTALL,
    )

    assert contract is not None
    assert contract.group(1) == runtime_v1.CONTRACT_VERSION_V1
    assert enum is not None
    assert tuple(re.findall(r"\b(k[A-Za-z0-9]+V1)\b", enum.group(1))) == (
        _CPP_COLUMNS
    )
    assert runtime_v1.DESCRIPTOR_SPEC_COLUMNS_V1 == _PYTHON_COLUMNS
    assert runtime_v1.DESCRIPTOR_SPEC_WIDTH_V1 == 13
    assert tuple(runtime_v1.DESCRIPTOR_SPEC_COLUMN_INDEX_V1.values()) == tuple(
        range(13)
    )
    for index, cpp_name in enumerate(_CPP_COLUMNS[:-1]):
        python_constant = re.sub(r"(?<!^)(?=[A-Z])", "_", cpp_name[1:-2]).upper()
        assert getattr(runtime_v1, f"{python_constant}_V1") == index


def test_g2_b1_binds_physical_sources_batches_and_generation_major_outputs(
    ready_plan: dict[str, Any],
) -> None:
    resident, policy, sizes = _sources(ready_plan, cells=1)

    bound = runtime_v1.bind_compact_summary_checkpoint_runtime(
        ready_plan,
        resident,
        policy,
        symbol_sizes=sizes,
        generation_count=2,
        cell_count=1,
    )

    assert bound.descriptor_count == 132
    assert bound.descriptor_specs.shape == (132, 13)
    assert bound.descriptor_specs.dtype == np.int64
    assert hashlib.sha256(bound.descriptor_specs.tobytes(order="C")).hexdigest() == (
        "50fae97a5e1a98401987d50ea28ce8be07c7b2aefc782fa22d9eb8f45f152c09"
    )
    assert bound.raw_digests.shape == (2, 1, 132, 32)
    assert bound.canonical_digests.shape == (2, 1, 132, 32)
    assert bound.status.shape == (2, 1, 132)
    assert bound.validity.shape == (2, 1, 132)
    assert bound.raw_digests.dtype == np.uint8
    assert bound.canonical_digests.dtype == np.uint8
    assert bound.status.dtype == np.int32
    assert bound.validity.dtype == np.uint8
    assert bound.allocation.total_nbytes == 18_216
    assert [batch.count for batch in bound.batches] == [16] * 8 + [4]
    assert all(batch.descriptor_specs.base is bound.descriptor_specs for batch in bound.batches)

    active = _descriptor_index(ready_plan, "agent_ids_by_alive_order")
    assert _spec_tensor(bound, active, runtime_v1.DATA_TENSOR_INDEX_V1) is resident[
        "active_agent_ids"
    ]
    assert _spec_tensor(
        bound, active, runtime_v1.ACTIVE_LENGTH_TENSOR_INDEX_V1
    ) is resident["active_count_state"]
    assert bound.descriptor_specs[active, runtime_v1.ROW_CAPACITY_V1] == 3
    assert bound.descriptor_specs[active, runtime_v1.ROW_BYTES_V1] == 8
    compact_ids = _descriptor_index(ready_plan, "agent_id_to_slot_agent_ids")
    compact_slots = _descriptor_index(ready_plan, "agent_id_to_slot_slots")
    assert _spec_tensor(
        bound, compact_ids, runtime_v1.DATA_TENSOR_INDEX_V1
    ) is resident["active_agent_ids"]
    assert _spec_tensor(
        bound, compact_slots, runtime_v1.DATA_TENSOR_INDEX_V1
    ) is resident["active_agent_slots"]
    for descriptor in (compact_ids, compact_slots):
        assert _spec_tensor(
            bound, descriptor, runtime_v1.ACTIVE_LENGTH_TENSOR_INDEX_V1
        ) is resident["active_count_state"]
    assert "agent_id_to_slot_prefix_length" not in resident

    event = _descriptor_index(ready_plan, "event_tape_type")
    assert _spec_tensor(
        bound, event, runtime_v1.ACTIVE_LENGTH_TENSOR_INDEX_V1
    ) is resident["event_active_count"]
    policy_delta = _descriptor_index(ready_plan, "policy_birth_delta_tape")
    assert _spec_tensor(
        bound, policy_delta, runtime_v1.ACTIVE_LENGTH_TENSOR_INDEX_V1
    ) is policy["policy_birth_tape_cursor"]
    energy = _descriptor_index(ready_plan, "energy")
    assert _spec_tensor(
        bound, energy, runtime_v1.ALIVE_MASK_TENSOR_INDEX_V1
    ) is resident["alive_mask"]

    assert not any(value.flags.writeable for value in resident.values())
    assert not any(value.flags.writeable for value in policy.values())
    runtime_v1.validate_compact_summary_checkpoint_runtime(bound)
    with pytest.raises(runtime_v1.CompactSummaryCheckpointRuntimeError, match="CUDA"):
        runtime_v1.validate_compact_summary_checkpoint_runtime(
            bound, strict_live=True
        )


def test_patch_p2_plan_binds_exact_descriptor_set(
    ready_patch_p2_plan: dict[str, Any],
) -> None:
    resident, policy, sizes = _sources(ready_patch_p2_plan, cells=1)

    bound = runtime_v1.bind_compact_summary_checkpoint_runtime(
        ready_patch_p2_plan,
        resident,
        policy,
        symbol_sizes=sizes,
        generation_count=2,
        cell_count=1,
    )

    assert bound.descriptor_count == 144
    assert bound.descriptor_specs.shape == (144, 13)
    assert bound.raw_digests.shape == (2, 1, 144, 32)
    assert bound.canonical_digests.shape == (2, 1, 144, 32)
    assert bound.status.shape == (2, 1, 144)
    assert bound.validity.shape == (2, 1, 144)
    assert [batch.count for batch in bound.batches] == [16] * 9
    assert bound.descriptor_names == tuple(
        descriptor["tensor_name"]
        for descriptor in ready_patch_p2_plan["descriptors"]
    )
    runtime_v1.validate_compact_summary_checkpoint_runtime(bound)


def test_g32_b16_allocation_receipt_is_bounded_without_source_allocation(
    ready_plan: dict[str, Any],
) -> None:
    allocation = runtime_v1.checkpoint_digest_output_allocation(
        32, 16, ready_plan["counts"]["descriptors"]
    )

    assert allocation.raw_digest_shape == (32, 16, 132, 32)
    assert allocation.canonical_digest_shape == (32, 16, 132, 32)
    assert allocation.status_shape == (32, 16, 132)
    assert allocation.validity_shape == (32, 16, 132)
    assert allocation.raw_digest_nbytes == 2_162_688
    assert allocation.canonical_digest_nbytes == 2_162_688
    assert allocation.status_nbytes == 270_336
    assert allocation.validity_nbytes == 67_584
    assert allocation.total_nbytes == 4_663_296


def test_metadata_is_immutable_while_native_outputs_remain_writable(
    ready_plan: dict[str, Any],
) -> None:
    resident, policy, sizes = _sources(ready_plan, cells=1)
    bound = runtime_v1.bind_compact_summary_checkpoint_runtime(
        ready_plan,
        resident,
        policy,
        symbol_sizes=sizes,
        generation_count=2,
        cell_count=1,
    )

    with pytest.raises(FrozenInstanceError):
        bound.cell_count = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        bound.tensor_aliases["new"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        bound.symbol_sizes["cell"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        bound.outputs()["status"] = bound.status  # type: ignore[index]
    with pytest.raises(ValueError):
        bound.descriptor_specs[0, 0] = 99
    with pytest.raises(ValueError):
        bound.batches[0].descriptor_specs[0, 0] = 99

    bound.validity[0, 0, 0] = 1
    assert bound.validity[0, 0, 0] == 1


def test_duplicate_runtime_aliases_must_name_one_physical_authority(
    ready_plan: dict[str, Any],
) -> None:
    resident, policy, sizes = _sources(ready_plan, cells=1)
    resident["agent_ids_by_alive_order"] = np.zeros(
        (1, sizes["capacity"]), dtype=np.int64
    )

    with pytest.raises(
        runtime_v1.CompactSummaryCheckpointRuntimeError,
        match="must alias resident.active_agent_ids",
    ):
        runtime_v1.bind_compact_summary_checkpoint_runtime(
            ready_plan,
            resident,
            policy,
            symbol_sizes=sizes,
            generation_count=2,
            cell_count=1,
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda resident: resident.pop("energy"), "missing required tensor 'energy'"),
        (
            lambda resident: resident.__setitem__(
                "energy", np.zeros((1, 3), dtype=np.float64)
            ),
            "dtype must be float32",
        ),
        (
            lambda resident: resident.__setitem__(
                "energy", np.zeros((1, 6), dtype=np.float32)[:, ::2]
            ),
            "must be C-contiguous",
        ),
        (
            lambda resident: resident.__setitem__(
                "energy", np.zeros((1, 2), dtype=np.float32)
            ),
            "shape must be",
        ),
        (
            lambda resident: resident.__setitem__(
                "active_count_state", np.zeros((2,), dtype=np.int64)
            ),
            r"shape must be \(1,\)",
        ),
    ],
)
def test_malformed_resident_sources_fail_closed(
    ready_plan: dict[str, Any],
    mutator: Any,
    match: str,
) -> None:
    resident, policy, sizes = _sources(ready_plan, cells=1)
    mutator(resident)

    with pytest.raises(runtime_v1.CompactSummaryCheckpointRuntimeError, match=match):
        runtime_v1.bind_compact_summary_checkpoint_runtime(
            ready_plan,
            resident,
            policy,
            symbol_sizes=sizes,
            generation_count=2,
            cell_count=1,
        )


def test_unsupported_mode_and_patch_pending_plan_are_rejected(
    ready_plan: dict[str, Any],
) -> None:
    resident, policy, sizes = _sources(ready_plan, cells=1)
    malformed = copy.deepcopy(ready_plan)
    malformed["descriptors"][0]["raw_canonical_mode"] = "implicit-default"
    with pytest.raises(
        runtime_v1.CompactSummaryCheckpointRuntimeError,
        match="descriptor validation failed",
    ):
        runtime_v1.bind_compact_summary_checkpoint_runtime(
            malformed,
            resident,
            policy,
            symbol_sizes=sizes,
            generation_count=2,
            cell_count=1,
        )

    patch_pending = copy.deepcopy(ready_plan)
    patch_pending["world"] = "patch"
    patch_pending["status"] = "patch_source_projection_pending"
    with pytest.raises(
        runtime_v1.CompactSummaryCheckpointRuntimeError,
        match="Patch pending plans are unsupported",
    ):
        runtime_v1.bind_compact_summary_checkpoint_runtime(
            patch_pending,
            resident,
            policy,
            symbol_sizes=sizes,
            generation_count=2,
            cell_count=1,
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda plan: plan["source_contract"].__setitem__("patch_profile", "P1"),
            "patch_profile: must equal 'P2'",
        ),
        (
            lambda plan: plan.__setitem__(
                "status", runtime_v1.READY_FLAT_PLAN_STATUS
            ),
            "ready Patch P2 plan required",
        ),
        (
            lambda plan: plan["descriptors"][0].__setitem__(
                "dtype_bytes", plan["descriptors"][0]["dtype_bytes"] + 1
            ),
            "descriptor validation failed",
        ),
    ],
)
def test_malformed_or_mismatched_patch_p2_plans_fail_closed(
    ready_patch_p2_plan: dict[str, Any],
    mutator: Any,
    match: str,
) -> None:
    malformed = copy.deepcopy(ready_patch_p2_plan)
    mutator(malformed)
    resident, policy, sizes = _sources(ready_patch_p2_plan, cells=1)

    with pytest.raises(runtime_v1.CompactSummaryCheckpointRuntimeError, match=match):
        runtime_v1.bind_compact_summary_checkpoint_runtime(
            malformed,
            resident,
            policy,
            symbol_sizes=sizes,
            generation_count=2,
            cell_count=1,
        )
