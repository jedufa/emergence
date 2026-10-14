from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_birth_transaction import (
    BIRTH_TRANSACTION_CAPABILITIES,
    DRIVE_STATE_FIELDS,
    GENOME_FLOAT32_FIELDS,
    GENOME_INT32_FIELDS,
    IDENTITY_STATE_FIELDS,
    POLICY_STATE_FIELDS,
    SCALAR_RNG_STATE_FIELDS,
    execute_fixed_mlp_birth_transaction_cuda,
    load_cuda_birth_transaction_extension,
)
from perf.cuda_architecture_b_child_drives import (
    execute_fixed_mlp_child_drives_cuda,
)
from perf.cuda_architecture_b_child_policy import (
    execute_fixed_mlp_child_policy_cuda,
    prepare_fixed_mlp_turnover_child_policy_tape,
)
from perf.cuda_architecture_b_lifecycle import (
    execute_controlled_birth_commit_cuda,
    execute_fixed_mlp_child_scalar_genome_cuda,
    pack_cpython_random_state_cuda,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors
from tests.perf.test_eiel_cuda_architecture_b_lifecycle import (
    _birth_state,
    _scalar_genome_state,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-birth-transaction-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path) -> Any:
    return load_cuda_birth_transaction_extension(build_directory=cuda_build_directory)


def _make_bundle(*, policy_episode_capacity: int = 2) -> dict[str, Any]:
    identity = _birth_state()
    identity.update(
        {
            "last_action": torch.arange(8, dtype=torch.int32, device="cuda"),
            "last_last_action": torch.arange(
                10, 18, dtype=torch.int32, device="cuda"
            ),
            "action_switches": torch.arange(
                20, 28, dtype=torch.int32, device="cuda"
            ),
            "action_oscillations": torch.arange(
                30, 38, dtype=torch.int32, device="cuda"
            ),
            "episode0_switches": torch.arange(
                40, 48, dtype=torch.int32, device="cuda"
            ),
            "episode0_oscillations": torch.arange(
                50, 58, dtype=torch.int32, device="cuda"
            ),
        }
    )
    genome_all = _scalar_genome_state()
    identity["repro_threshold"] = genome_all["repro_threshold"]
    genome = {
        name: genome_all[name]
        for name in (*GENOME_INT32_FIELDS, *GENOME_FLOAT32_FIELDS)
    }

    python_rng = random.Random(717)
    scalar_rng = pack_cpython_random_state_cuda(
        python_rng.getstate(),
        device="cuda",
        semantic_draws=23,
        gauss_tape_raw_capacity=4096,
    )

    bit_generator = np.random.PCG64(303)
    np.random.Generator(bit_generator).integers(0, 2, dtype=np.int32)
    state_words, buffer_words = pcg64_state_tensors(
        copy.deepcopy(bit_generator.state),
        device="cuda",
    )
    drive = {
        "drives": torch.full((8, 4), -9.0, dtype=torch.float32, device="cuda"),
        "birth_reset_shock": torch.full(
            (8,), -7.0, dtype=torch.float32, device="cuda"
        ),
        "pcg64_state_words": state_words,
        "pcg64_buffer_words": buffer_words,
        "mutation_np_semantic_draws": torch.tensor(
            [37], dtype=torch.int64, device="cuda"
        ),
        "chemistry": torch.full(
            (8, 6), -6.0, dtype=torch.float32, device="cuda"
        ),
    }

    width = 17
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(909)
    policy_tape = prepare_fixed_mlp_turnover_child_policy_tape(
        torch_generator.get_state(),
        episode_capacity=policy_episode_capacity,
        row_width=width,
        device="cuda",
        policy_weight_mutation_enable=True,
        policy_weight_mutation_mode="sparse",
        policy_weight_mutation_rate=0.35,
        policy_weight_mutation_std=0.075,
        semantic_draws=11,
    )
    policy = {
        "mlp_pack": (
            torch.arange(8 * width, dtype=torch.float32)
            .reshape(8, width)
            .div(100.0)
            .cuda()
            .contiguous()
        ),
        "hmem": torch.full((8, 4), 9.0, dtype=torch.float32, device="cuda"),
        **policy_tape,
    }
    return {
        "identity": {name: identity[name] for name in IDENTITY_STATE_FIELDS},
        "parent_slots": identity["parent_slots"],
        "spawn_positions": identity["spawn_positions"],
        "scalar_rng": scalar_rng,
        "genome": genome,
        "policy_mem_dim": genome_all["policy_mem_dim"],
        "policy_use_gru": genome_all["policy_use_gru"],
        "drive": drive,
        "policy": policy,
    }


def _clone_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    clone = {
        "identity": {name: tensor.clone() for name, tensor in bundle["identity"].items()},
        "parent_slots": bundle["parent_slots"].clone(),
        "spawn_positions": bundle["spawn_positions"].clone(),
        "scalar_rng": {
            name: tensor.clone() for name, tensor in bundle["scalar_rng"].items()
        },
        "genome": {name: tensor.clone() for name, tensor in bundle["genome"].items()},
        "policy_mem_dim": bundle["policy_mem_dim"].clone(),
        "policy_use_gru": bundle["policy_use_gru"].clone(),
        "drive": {name: tensor.clone() for name, tensor in bundle["drive"].items()},
        "policy": {name: tensor.clone() for name, tensor in bundle["policy"].items()},
    }
    clone["genome"]["repro_threshold"] = clone["identity"]["repro_threshold"]
    return clone


def _execute_transaction(
    bundle: dict[str, Any],
    build_directory: Path,
) -> dict[str, torch.Tensor]:
    return execute_fixed_mlp_birth_transaction_cuda(
        identity_state=bundle["identity"],
        parent_slots=bundle["parent_slots"],
        spawn_positions=bundle["spawn_positions"],
        scalar_rng_state=bundle["scalar_rng"],
        genome_state=bundle["genome"],
        policy_mem_dim=bundle["policy_mem_dim"],
        policy_use_gru=bundle["policy_use_gru"],
        drive_state=bundle["drive"],
        policy_state=bundle["policy"],
        parent_energy_share=0.25,
        minimum_energy_buffer=2.0,
        start_health=1.5,
        cooldown_steps=7,
        generation=4,
        tick=12,
        sensor_radius_max=3,
        max_energy=100.0,
        build_directory=build_directory,
    )


def _execute_standalone_composition(
    bundle: dict[str, Any],
    build_directory: Path,
) -> dict[str, dict[str, torch.Tensor]]:
    identity = execute_controlled_birth_commit_cuda(
        **{
            name: tensor
            for name, tensor in bundle["identity"].items()
            if name
            not in {
                "last_action",
                "last_last_action",
                "action_switches",
                "action_oscillations",
                "episode0_switches",
                "episode0_oscillations",
            }
        },
        parent_slots=bundle["parent_slots"],
        spawn_positions=bundle["spawn_positions"],
        parent_energy_share=0.25,
        minimum_energy_buffer=2.0,
        start_health=1.5,
        cooldown_steps=7,
        generation=4,
        tick=12,
        build_directory=build_directory / "identity",
    )
    child_slots = identity["child_slots"]
    scalar = execute_fixed_mlp_child_scalar_genome_cuda(
        slot_to_agent_id=bundle["identity"]["slot_to_agent_id"],
        alive_mask=bundle["identity"]["alive_mask"],
        parent_slots=bundle["parent_slots"],
        child_slots=child_slots,
        **bundle["scalar_rng"],
        **bundle["genome"],
        policy_mem_dim=bundle["policy_mem_dim"],
        policy_use_gru=bundle["policy_use_gru"],
        sensor_radius_max=3,
        build_directory=build_directory / "scalar",
    )
    drives = execute_fixed_mlp_child_drives_cuda(
        slot_to_agent_id=bundle["identity"]["slot_to_agent_id"],
        alive_mask=bundle["identity"]["alive_mask"],
        policy_use_gru=bundle["policy_use_gru"],
        parent_slots=bundle["parent_slots"],
        child_slots=child_slots,
        policy_energy_observation=bundle["identity"]["policy_energy_observation"],
        genome_drive_sp_hunger=bundle["genome"]["genome_drive_sp_hunger"],
        genome_drive_sp_pain=bundle["genome"]["genome_drive_sp_pain"],
        genome_drive_sp_fatigue=bundle["genome"]["genome_drive_sp_fatigue"],
        genome_drive_sp_fear=bundle["genome"]["genome_drive_sp_fear"],
        genome_exp_prior_noise=bundle["genome"]["genome_exp_prior_noise"],
        genome_drive_birth_noise=bundle["genome"]["genome_drive_birth_noise"],
        **{
            name: tensor
            for name, tensor in bundle["drive"].items()
            if name != "chemistry"
        },
        max_energy=100.0,
        build_directory=build_directory / "drives",
    )
    policy = execute_fixed_mlp_child_policy_cuda(
        slot_to_agent_id=bundle["identity"]["slot_to_agent_id"],
        alive_mask=bundle["identity"]["alive_mask"],
        parent_slots=bundle["parent_slots"],
        child_slots=child_slots,
        policy_use_gru=bundle["policy_use_gru"],
        **{
            name: tensor
            for name, tensor in bundle["policy"].items()
            if name
            not in {
                "policy_birth_crossover_mask_packed_tape",
                "policy_birth_crossover_mask_logical_width",
                "policy_birth_crossover_mask_encoding",
            }
        },
        build_directory=build_directory / "policy",
    )
    return {
        "identity": identity,
        "scalar": scalar,
        "drives": drives,
        "policy": policy,
    }


def _mutable_tensors(bundle: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for group_name in ("identity", "scalar_rng", "genome", "drive", "policy"):
        for name, tensor in bundle[group_name].items():
            tensors[f"{group_name}.{name}"] = tensor
    tensors["policy_mem_dim"] = bundle["policy_mem_dim"]
    tensors["policy_use_gru"] = bundle["policy_use_gru"]
    return tensors


def _assert_bit_exact(actual: torch.Tensor, expected: torch.Tensor, field: str) -> None:
    assert actual.dtype == expected.dtype, field
    assert actual.shape == expected.shape, field
    if actual.dtype == torch.float32:
        equal = torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    elif actual.dtype == torch.float64:
        equal = torch.equal(actual.view(torch.int64), expected.view(torch.int64))
    else:
        equal = torch.equal(actual, expected)
    assert equal, field


def test_transaction_matches_standalone_component_composition_exactly(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    base = _make_bundle()
    transaction_bundle = _clone_bundle(base)
    standalone_bundle = _clone_bundle(base)
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in _mutable_tensors(transaction_bundle).items()
    }

    transaction = _execute_transaction(transaction_bundle, cuda_build_directory)
    standalone = _execute_standalone_composition(
        standalone_bundle,
        cuda_build_directory / "standalone",
    )
    torch.cuda.synchronize()

    assert int(transaction["status"].cpu()[0]) == 0
    assert int(transaction["committed_count"].cpu()[0]) == 2
    for component_status in (
        "identity_status",
        "scalar_status",
        "drive_status",
        "policy_status",
    ):
        assert int(transaction[component_status].cpu()[0]) == 0
    assert pointers == {
        name: tensor.data_ptr()
        for name, tensor in _mutable_tensors(transaction_bundle).items()
    }
    for name, actual in _mutable_tensors(transaction_bundle).items():
        if name == "drive.chemistry" or name in {
            "identity.last_action",
            "identity.last_last_action",
            "identity.action_switches",
            "identity.action_oscillations",
            "identity.episode0_switches",
            "identity.episode0_oscillations",
        }:
            continue
        _assert_bit_exact(actual, _mutable_tensors(standalone_bundle)[name], name)

    child_slots = transaction["child_slots"].detach().cpu().tolist()
    for field in ("last_action", "last_last_action"):
        assert transaction_bundle["identity"][field][child_slots].detach().cpu().tolist() == [
            -1,
            -1,
        ]
    for field in (
        "action_switches",
        "action_oscillations",
        "episode0_switches",
        "episode0_oscillations",
    ):
        assert transaction_bundle["identity"][field][child_slots].detach().cpu().tolist() == [
            0,
            0,
        ]

    expected_chemistry = standalone_bundle["drive"]["chemistry"].clone()
    for slot in transaction["child_slots"].detach().cpu().tolist():
        child_energy = float(
            transaction_bundle["identity"]["policy_energy_observation"][slot]
            .detach()
            .cpu()
            .item()
        )
        expected_chemistry[slot] = torch.tensor(
            (
                np.float32(0.40 + 0.40 * np.clip(child_energy / 100.0, 0.0, 1.0)),
                np.float32(0.55),
                np.float32(0.0),
                np.float32(0.0),
                np.float32(0.50),
                np.float32(0.25),
            ),
            dtype=torch.float32,
            device="cuda",
        )
    _assert_bit_exact(
        transaction_bundle["drive"]["chemistry"],
        expected_chemistry,
        "drive.chemistry",
    )

    output_pairs = {
        "parent_agent_ids": standalone["identity"]["parent_agent_ids"],
        "child_agent_ids": standalone["identity"]["child_agent_ids"],
        "child_slots": standalone["identity"]["child_slots"],
        "alive_agent_ids_by_order": standalone["identity"]["alive_agent_ids_by_order"],
        "scalar_semantic_draws_added": standalone["scalar"]["semantic_draws_added"],
        "population_birth_noise_rows_consumed": standalone["drives"][
            "population_birth_noise_rows_consumed"
        ],
        "population_exp_prior_noise_rows_consumed": standalone["drives"][
            "population_exp_prior_noise_rows_consumed"
        ],
        "native_population_standard_normal_rows_generated": standalone["drives"][
            "native_population_standard_normal_rows_generated"
        ],
        "drive_semantic_draws_added": standalone["drives"]["semantic_draws_added"],
        "drive_raw64_draws_added": standalone["drives"]["raw64_draws_added"],
        "policy_semantic_draws_added": standalone["policy"]["semantic_draws_added"],
        "policy_mutation_events": standalone["policy"]["mutation_events"],
        "policy_mutated_elements": standalone["policy"]["mutated_elements"],
        "policy_mutation_delta_total": standalone["policy"]["mutation_delta_total"],
        "policy_mutation_occurred_by_birth": standalone["policy"][
            "mutation_occurred_by_birth"
        ],
        "policy_mutation_delta_by_birth": standalone["policy"][
            "mutation_delta_by_birth"
        ],
    }
    for name, expected in output_pairs.items():
        _assert_bit_exact(transaction[name], expected, name)


def test_policy_tape_exhaustion_rolls_back_all_four_stages_atomically(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    bundle = _make_bundle(policy_episode_capacity=1)
    before = {name: tensor.clone() for name, tensor in _mutable_tensors(bundle).items()}
    pointers = {
        name: tensor.data_ptr() for name, tensor in _mutable_tensors(bundle).items()
    }

    result = _execute_transaction(bundle, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 4
    assert int(result["identity_status"].cpu()[0]) == 0
    assert int(result["scalar_status"].cpu()[0]) == 0
    assert int(result["drive_status"].cpu()[0]) == 0
    assert int(result["policy_status"].cpu()[0]) == 2
    assert int(result["committed_count"].cpu()[0]) == 0
    for name, tensor in _mutable_tensors(bundle).items():
        assert tensor.data_ptr() == pointers[name]
        _assert_bit_exact(tensor, before[name], name)


def test_invalid_identity_plan_fails_before_authoritative_commit(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    bundle = _make_bundle()
    bundle["spawn_positions"][1].copy_(bundle["spawn_positions"][0])
    before = {name: tensor.clone() for name, tensor in _mutable_tensors(bundle).items()}

    result = _execute_transaction(bundle, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 1
    assert int(result["identity_status"].cpu()[0]) != 0
    assert int(result["committed_count"].cpu()[0]) == 0
    for name, tensor in _mutable_tensors(bundle).items():
        _assert_bit_exact(tensor, before[name], name)


def test_capability_receipt_excludes_unimplemented_surfaces() -> None:
    for capability in (
        "resident_cuda_state",
        "one_native_transaction_call",
        "stable_authoritative_tensor_pointers",
        "whole_batch_atomic_commit",
        "exact_cpython_rng_continuation",
        "exact_numpy_pcg64_continuation",
        "exact_torch_cpu_mutation_rng_continuation",
        "fixed_mlp_flat_patch",
    ):
        assert BIRTH_TRANSACTION_CAPABILITIES[capability] is True
    for unsupported in (
        "python_per_birth_loop",
        "typed_birth_events",
        "generation_integration",
        "fresh_agent_stats_reset",
        "selection_stat_accumulator_reset",
        "gru",
        "jungle",
        "wdl",
        "multicell",
        "promotion_qualified",
    ):
        assert BIRTH_TRANSACTION_CAPABILITIES[unsupported] is False
