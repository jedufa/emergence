from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_child_drives import (
    CHILD_DRIVE_CAPABILITY_RECEIPT,
    CHILD_DRIVE_STATUS,
    POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_OPERATOR_ID_V1,
    POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_RECEIPT_SCHEMA_V1,
    POPULATION_NEWBORN_DRIVE_RAW64_BLOCKER_V1,
    execute_fixed_mlp_child_drives_cuda,
    population_newborn_drive_standard_normal_receipt_v1,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors


CAPACITY = 8
MAX_ENERGY = 100.0


def _pcg64_state_from_tensors(
    state_words: torch.Tensor,
    buffer_words: torch.Tensor,
) -> dict[str, object]:
    state = [int(value) for value in state_words.detach().cpu().tolist()]
    buffer = [int(value) for value in buffer_words.detach().cpu().tolist()]
    return {
        "bit_generator": "PCG64",
        "state": {
            "state": state[0] | (state[1] << 64),
            "inc": state[2] | (state[3] << 64),
        },
        "has_uint32": buffer[0],
        "uinteger": buffer[1],
    }


def _canon_genome_double(value: np.float32) -> float:
    return float(np.rint(float(np.float32(value)) * 100_000_000.0) / 100_000_000.0)


def _clamp01_f32(value: np.float32) -> np.float32:
    if value < np.float32(0.0):
        return np.float32(0.0)
    if value > np.float32(1.0):
        return np.float32(1.0)
    return np.float32(value)


def _cpu_resident_owner_oracle(
    state: dict[str, torch.Tensor],
    initial_pcg64_state: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object], int, int, int]:
    """Committed Flat+Patch fixed-MLP newborn-drive order, without policy work."""
    drives = state["drives"].detach().cpu().numpy().copy()
    shocks = state["birth_reset_shock"].detach().cpu().numpy().copy()
    energy = state["policy_energy_observation"].detach().cpu().numpy()
    hunger = state["genome_drive_sp_hunger"].detach().cpu().numpy()
    pain = state["genome_drive_sp_pain"].detach().cpu().numpy()
    fatigue = state["genome_drive_sp_fatigue"].detach().cpu().numpy()
    fear = state["genome_drive_sp_fear"].detach().cpu().numpy()
    exp_prior_noise = state["genome_exp_prior_noise"].detach().cpu().numpy()
    birth_noise = state["genome_drive_birth_noise"].detach().cpu().numpy()
    children = state["child_slots"].detach().cpu().tolist()

    bit_generator = np.random.PCG64()
    bit_generator.state = copy.deepcopy(initial_pcg64_state)
    rng = np.random.Generator(bit_generator)
    birth_rows = 0
    exp_prior_rows = 0
    semantic_draws = 0
    max_energy = float(np.float32(MAX_ENERGY))
    normal64 = np.empty((4,), dtype=np.float64)
    normal32 = np.empty((4,), dtype=np.float32)

    for raw_child_slot in children:
        child_slot = int(raw_child_slot)
        hunger_from_energy = min(
            1.0,
            max(0.0, 1.0 - float(energy[child_slot]) / max_energy),
        )
        drives[child_slot, 0] = _clamp01_f32(np.float32(
            0.60 * _canon_genome_double(hunger[child_slot])
            + 0.40 * hunger_from_energy
        ))
        drives[child_slot, 1] = _clamp01_f32(pain[child_slot])
        drives[child_slot, 2] = _clamp01_f32(fatigue[child_slot])
        drives[child_slot, 3] = _clamp01_f32(fear[child_slot])

        clipped_birth_noise = min(
            0.20,
            max(0.0, _canon_genome_double(birth_noise[child_slot])),
        )
        if clipped_birth_noise > 0.0:
            rng.standard_normal(size=(4,), dtype=np.float64, out=normal64)
            normal32[...] = normal64
            for drive_index in range(4):
                delta = np.float32(
                    np.float32(clipped_birth_noise) * normal32[drive_index]
                )
                drives[child_slot, drive_index] = _clamp01_f32(np.float32(
                    drives[child_slot, drive_index] + delta
                ))
            birth_rows += 1
            semantic_draws += 4

        shock0 = np.float32(abs(np.float32(
            drives[child_slot, 0] - hunger[child_slot]
        )))
        shock1 = np.float32(abs(np.float32(
            drives[child_slot, 1] - pain[child_slot]
        )))
        shock2 = np.float32(abs(np.float32(
            drives[child_slot, 2] - fatigue[child_slot]
        )))
        shock3 = np.float32(abs(np.float32(
            drives[child_slot, 3] - fear[child_slot]
        )))
        shock_sum01 = np.float32(shock0 + shock1)
        shock_sum012 = np.float32(shock_sum01 + shock2)
        shock_sum0123 = np.float32(shock_sum012 + shock3)
        shocks[child_slot] = np.float32(shock_sum0123 / np.float32(4.0))

        if float(exp_prior_noise[child_slot]) > 0.0:
            rng.standard_normal(size=(7,), dtype=np.float64)
            rng.standard_normal(size=(7,), dtype=np.float64)
            exp_prior_rows += 2
            semantic_draws += 14

    return (
        drives,
        shocks,
        copy.deepcopy(bit_generator.state),
        birth_rows,
        exp_prior_rows,
        semantic_draws,
    )


def _state(
    *,
    seed: int = 15,
    buffered_uint32: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    bit_generator = np.random.PCG64(seed)
    if buffered_uint32:
        np.random.Generator(bit_generator).integers(0, 2, dtype=np.int32)
        assert int(bit_generator.state["has_uint32"]) == 1
    initial_pcg64_state = copy.deepcopy(bit_generator.state)
    pcg64_state_words, pcg64_buffer_words = pcg64_state_tensors(
        initial_pcg64_state,
        device="cuda",
    )

    slot_to_agent_id = torch.full(
        (CAPACITY,), -1, dtype=torch.int64, device="cuda"
    )
    slot_to_agent_id[[0, 1, 4, 5]] = torch.tensor(
        [10, 20, 21, 22], dtype=torch.int64, device="cuda"
    )
    alive_mask = torch.zeros((CAPACITY,), dtype=torch.bool, device="cuda")
    alive_mask[[0, 1, 4, 5]] = True
    policy_energy_observation = torch.full(
        (CAPACITY,), float("nan"), dtype=torch.float64, device="cuda"
    )
    policy_energy_observation[[0, 1, 4, 5]] = torch.tensor(
        [60.0, 84.0, 30.0, 42.0], dtype=torch.float64, device="cuda"
    )

    state = {
        "slot_to_agent_id": slot_to_agent_id,
        "alive_mask": alive_mask,
        "policy_use_gru": torch.zeros(
            (CAPACITY,), dtype=torch.bool, device="cuda"
        ),
        "parent_slots": torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        "child_slots": torch.tensor([4, 5], dtype=torch.int64, device="cuda"),
        "policy_energy_observation": policy_energy_observation,
        "genome_drive_sp_hunger": torch.tensor(
            [0.55, 0.55, 0.55, 0.55, 0.52, 0.73, 0.55, 0.55],
            dtype=torch.float32,
            device="cuda",
        ),
        "genome_drive_sp_pain": torch.tensor(
            [0.05, 0.05, 0.05, 0.05, 0.06, 0.16, 0.05, 0.05],
            dtype=torch.float32,
            device="cuda",
        ),
        "genome_drive_sp_fatigue": torch.tensor(
            [0.15, 0.15, 0.15, 0.15, 0.17, 0.27, 0.15, 0.15],
            dtype=torch.float32,
            device="cuda",
        ),
        "genome_drive_sp_fear": torch.tensor(
            [0.10, 0.10, 0.10, 0.10, 0.12, 0.22, 0.10, 0.10],
            dtype=torch.float32,
            device="cuda",
        ),
        "genome_exp_prior_noise": torch.tensor(
            [0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.05, 0.05],
            dtype=torch.float32,
            device="cuda",
        ),
        "genome_drive_birth_noise": torch.tensor(
            [0.03, 0.03, 0.03, 0.03, 0.02, 0.00, 0.03, 0.03],
            dtype=torch.float32,
            device="cuda",
        ),
        "drives": torch.full(
            (CAPACITY, 4), -9.0, dtype=torch.float32, device="cuda"
        ),
        "birth_reset_shock": torch.full(
            (CAPACITY,), -7.0, dtype=torch.float32, device="cuda"
        ),
        "pcg64_state_words": pcg64_state_words,
        "pcg64_buffer_words": pcg64_buffer_words,
        "mutation_np_semantic_draws": torch.tensor(
            [37], dtype=torch.int64, device="cuda"
        ),
    }
    return state, initial_pcg64_state


def _execute(
    state: dict[str, torch.Tensor],
    build_directory: Path,
) -> dict[str, torch.Tensor]:
    return execute_fixed_mlp_child_drives_cuda(
        **state,
        max_energy=MAX_ENERGY,
        build_directory=build_directory,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_child_drives_and_all_numpy_pcg64_transitions_match_cpu_owner(
    tmp_path: Path,
) -> None:
    state, initial_pcg64_state = _state(seed=15, buffered_uint32=True)
    expected = _cpu_resident_owner_oracle(state, initial_pcg64_state)
    before_drives = state["drives"].clone()
    before_shock = state["birth_reset_shock"].clone()
    before_state_words = state["pcg64_state_words"].clone()
    before_buffer_words = state["pcg64_buffer_words"].clone()
    before_semantic_draws = state["mutation_np_semantic_draws"].clone()
    pointers = {
        name: state[name].data_ptr()
        for name in (
            "drives",
            "birth_reset_shock",
            "pcg64_state_words",
            "pcg64_buffer_words",
            "mutation_np_semantic_draws",
        )
    }

    result = _execute(state, tmp_path / "build")

    assert int(result["status"].item()) == 0
    assert int(result["committed_count"].item()) == 2
    assert int(result["population_birth_noise_rows_consumed"].item()) == expected[3]
    assert int(result["population_exp_prior_noise_rows_consumed"].item()) == expected[4]
    assert int(result["native_population_standard_normal_rows_generated"].item()) == expected[3]
    assert int(result["semantic_draws_added"].item()) == expected[5]
    assert int(result["raw64_draws_added"].item()) >= expected[5]
    assert int(state["mutation_np_semantic_draws"].item()) == 37 + expected[5]
    operation_receipt = population_newborn_drive_standard_normal_receipt_v1(
        result=result,
        pcg64_state_words_before=before_state_words,
        pcg64_buffer_words_before=before_buffer_words,
        pcg64_state_words_after=state["pcg64_state_words"],
        pcg64_buffer_words_after=state["pcg64_buffer_words"],
        mutation_np_semantic_draws_before=before_semantic_draws,
        mutation_np_semantic_draws_after=state["mutation_np_semantic_draws"],
    )
    assert (
        operation_receipt["schema_version"]
        == POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_RECEIPT_SCHEMA_V1
    )
    assert (
        operation_receipt["operator_id"]
        == POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_OPERATOR_ID_V1
    )
    assert operation_receipt["stream_id"] == "numpy.population"
    assert operation_receipt["semantic_draws_added"] == expected[5]
    assert operation_receipt["birth_noise_rows_consumed"] == expected[3]
    assert operation_receipt["exp_prior_noise_rows_consumed"] == expected[4]
    assert operation_receipt["standard_normal_rows_generated"] == expected[3]
    assert operation_receipt["raw64_draws_added"] == int(
        result["raw64_draws_added"].item()
    )
    assert operation_receipt["raw_counter_unit"] == "numpy_pcg64_raw64_words"
    assert operation_receipt["raw_pcg64_delta_available"] is True
    assert operation_receipt["promotion_blockers"] == []
    assert operation_receipt["promotion_qualified"] is True
    assert operation_receipt["pcg64_state_changed"] is True
    assert len(operation_receipt["receipt_sha256"]) == 64
    assert _pcg64_state_from_tensors(
        state["pcg64_state_words"], state["pcg64_buffer_words"]
    ) == expected[2]
    for name, pointer in pointers.items():
        assert state[name].data_ptr() == pointer
        assert result[name].data_ptr() == pointer

    expected_drives = torch.from_numpy(expected[0]).to(device="cuda")
    expected_shock = torch.from_numpy(expected[1]).to(device="cuda")
    assert torch.equal(state["drives"].view(torch.int32), expected_drives.view(torch.int32))
    assert torch.equal(
        state["birth_reset_shock"].view(torch.int32),
        expected_shock.view(torch.int32),
    )
    untouched = torch.tensor([0, 1, 2, 3, 6, 7], device="cuda")
    assert torch.equal(state["drives"][untouched], before_drives[untouched])
    assert torch.equal(state["birth_reset_shock"][untouched], before_shock[untouched])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_zero_noise_rows_initialize_without_advancing_population_rng(
    tmp_path: Path,
) -> None:
    state, initial_pcg64_state = _state(seed=303, buffered_uint32=True)
    state["genome_drive_birth_noise"][state["child_slots"]] = 0.0
    state["genome_exp_prior_noise"][state["child_slots"]] = 0.0
    expected = _cpu_resident_owner_oracle(state, initial_pcg64_state)

    result = _execute(state, tmp_path / "build")

    assert int(result["status"].item()) == 0
    assert int(result["population_birth_noise_rows_consumed"].item()) == 0
    assert int(result["population_exp_prior_noise_rows_consumed"].item()) == 0
    assert int(result["semantic_draws_added"].item()) == 0
    assert int(result["raw64_draws_added"].item()) == 0
    assert int(state["mutation_np_semantic_draws"].item()) == 37
    assert _pcg64_state_from_tensors(
        state["pcg64_state_words"], state["pcg64_buffer_words"]
    ) == initial_pcg64_state
    assert torch.equal(
        state["drives"].view(torch.int32),
        torch.from_numpy(expected[0]).to(device="cuda").view(torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batch_and_split_child_drive_continuation_are_identical(
    tmp_path: Path,
) -> None:
    batch, _ = _state(seed=61, buffered_uint32=False)
    split, _ = _state(seed=61, buffered_uint32=False)

    batch_result = _execute(batch, tmp_path / "build")
    first_parents = split["parent_slots"]
    first_children = split["child_slots"]
    split["parent_slots"] = first_parents[:1]
    split["child_slots"] = first_children[:1]
    first_result = _execute(split, tmp_path / "build")
    split["parent_slots"] = first_parents[1:]
    split["child_slots"] = first_children[1:]
    second_result = _execute(split, tmp_path / "build")

    assert int(batch_result["status"].item()) == 0
    assert int(first_result["status"].item()) == 0
    assert int(second_result["status"].item()) == 0
    for name in (
        "drives",
        "birth_reset_shock",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "mutation_np_semantic_draws",
    ):
        assert torch.equal(batch[name], split[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("duplicate_child", 1),
        ("parent_order", 2),
        ("child_identity", 3),
        ("gru", 4),
        ("scalar", 5),
        ("pcg64", 6),
    ],
)
def test_preflight_failures_are_atomic(
    failure: str,
    expected_status: int,
    tmp_path: Path,
) -> None:
    state, _ = _state(seed=15, buffered_uint32=True)
    if failure == "duplicate_child":
        state["child_slots"] = torch.tensor(
            [4, 4], dtype=torch.int64, device="cuda"
        )
    elif failure == "parent_order":
        state["parent_slots"] = torch.tensor(
            [1, 0], dtype=torch.int64, device="cuda"
        )
    elif failure == "child_identity":
        state["alive_mask"][5] = False
    elif failure == "gru":
        state["policy_use_gru"][4] = True
    elif failure == "scalar":
        state["genome_drive_birth_noise"][5] = 0.25
    elif failure == "pcg64":
        state["pcg64_buffer_words"][0] = 2
    else:
        raise AssertionError(f"unknown failure: {failure}")

    mutable_names = (
        "drives",
        "birth_reset_shock",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "mutation_np_semantic_draws",
    )
    before = {name: state[name].clone() for name in mutable_names}
    pointers = {name: state[name].data_ptr() for name in mutable_names}

    result = _execute(state, tmp_path / "build")

    assert int(result["status"].item()) == expected_status
    assert CHILD_DRIVE_STATUS[expected_status] != "ok"
    assert int(result["committed_count"].item()) == 0
    assert int(result["semantic_draws_added"].item()) == 0
    for name in mutable_names:
        assert state[name].data_ptr() == pointers[name]
        assert result[name].data_ptr() == pointers[name]
        assert torch.equal(state[name], before[name])


def test_child_drive_claims_are_standalone_and_bounded() -> None:
    receipt: dict[str, Any] = CHILD_DRIVE_CAPABILITY_RECEIPT
    assert receipt["fixed_mlp_flat_patch_child_drives"] is True
    assert receipt["numpy_pcg64_standard_normal_float64_then_float32"] == "bit_exact"
    assert receipt["whole_batch_preflight_fail_closed"] is True
    for nonclaim in (
        "birth_identity",
        "scalar_inheritance",
        "policy_rows",
        "generation_integration",
        "controlled_birth_commit",
        "gru",
        "jungle",
        "wdl",
        "promotion_qualified",
    ):
        assert receipt[nonclaim] is False


def test_population_newborn_drive_standard_normal_receipt_is_named_and_bounded() -> None:
    before_state = torch.tensor([1, 2, 3, 4], dtype=torch.uint64)
    before_buffer = torch.tensor([0, 0], dtype=torch.int64)
    after_state = torch.tensor([5, 6, 7, 8], dtype=torch.uint64)
    after_buffer = torch.tensor([1, 123], dtype=torch.int64)
    before_draws = torch.tensor([37], dtype=torch.int64)
    after_draws = torch.tensor([55], dtype=torch.int64)
    result = {
        "status": torch.tensor([0], dtype=torch.int32),
        "committed_count": torch.tensor([2], dtype=torch.int64),
        "population_birth_noise_rows_consumed": torch.tensor([1], dtype=torch.int64),
        "population_exp_prior_noise_rows_consumed": torch.tensor([2], dtype=torch.int64),
        "native_population_standard_normal_rows_generated": torch.tensor([1], dtype=torch.int64),
        "semantic_draws_added": torch.tensor([18], dtype=torch.int64),
        "raw64_draws_added": torch.tensor([21], dtype=torch.int64),
    }

    receipt = population_newborn_drive_standard_normal_receipt_v1(
        result=result,
        pcg64_state_words_before=before_state,
        pcg64_buffer_words_before=before_buffer,
        pcg64_state_words_after=after_state,
        pcg64_buffer_words_after=after_buffer,
        mutation_np_semantic_draws_before=before_draws,
        mutation_np_semantic_draws_after=after_draws,
    )

    assert (
        receipt["schema_version"]
        == POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_RECEIPT_SCHEMA_V1
    )
    assert (
        receipt["operator_id"]
        == POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_OPERATOR_ID_V1
    )
    assert receipt["stream_id"] == "numpy.population"
    assert receipt["status_name"] == "ok"
    assert receipt["semantic_draws_added"] == 18
    assert receipt["semantic_counter_unit"] == "numpy_standard_normal_float64_values"
    assert receipt["raw64_draws_added"] == 21
    assert receipt["raw_counter_unit"] == "numpy_pcg64_raw64_words"
    assert receipt["raw_pcg64_delta_available"] is True
    assert receipt["promotion_blockers"] == []
    assert receipt["promotion_qualified"] is True
    assert receipt["pcg64_state_changed"] is True
    assert len(receipt["pcg64_state_before"]["sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64


def test_population_newborn_drive_receipt_fails_closed_without_raw64() -> None:
    before_state = torch.tensor([1, 2, 3, 4], dtype=torch.uint64)
    before_buffer = torch.tensor([0, 0], dtype=torch.int64)
    after_state = torch.tensor([5, 6, 7, 8], dtype=torch.uint64)
    after_buffer = torch.tensor([1, 123], dtype=torch.int64)
    result = {
        "status": torch.tensor([0], dtype=torch.int32),
        "committed_count": torch.tensor([2], dtype=torch.int64),
        "population_birth_noise_rows_consumed": torch.tensor([1], dtype=torch.int64),
        "population_exp_prior_noise_rows_consumed": torch.tensor([2], dtype=torch.int64),
        "native_population_standard_normal_rows_generated": torch.tensor([1], dtype=torch.int64),
        "semantic_draws_added": torch.tensor([18], dtype=torch.int64),
    }

    receipt = population_newborn_drive_standard_normal_receipt_v1(
        result=result,
        pcg64_state_words_before=before_state,
        pcg64_buffer_words_before=before_buffer,
        pcg64_state_words_after=after_state,
        pcg64_buffer_words_after=after_buffer,
        mutation_np_semantic_draws_before=torch.tensor([37], dtype=torch.int64),
        mutation_np_semantic_draws_after=torch.tensor([55], dtype=torch.int64),
    )

    assert receipt["raw_pcg64_delta_available"] is False
    assert receipt["promotion_blockers"] == [
        POPULATION_NEWBORN_DRIVE_RAW64_BLOCKER_V1
    ]
    assert receipt["promotion_qualified"] is False
