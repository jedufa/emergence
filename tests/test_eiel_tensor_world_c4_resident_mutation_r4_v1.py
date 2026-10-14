"""Focused CPU/static contract tests for the isolated resident C4 R4 component."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_adversarial_fixture_v1
from engine.tensor_world.c4_genome import preflight_c4_scalar_genome_mutations_v1
from engine.tensor_world.c4_mating import (
    TensorWorldC4MateBirthConfigV1,
    commit_c4_mate_births_oracle_v1,
    plan_c4_mate_births_oracle_v1,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    C4_RESIDENT_MUTATION_R4_DECISION_V1,
    _validate_replay_structure,
    c4_resident_mutation_r4_source_receipt_v1,
    execute_c4_resident_mutation_r4_cuda_v1,
    make_c4_resident_mutation_r4_staging_v1,
)
from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu"


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            item = getattr(owner, field.name)
            if isinstance(item, torch.Tensor):
                setattr(owner, field.name, item.to(device=device))
            elif is_dataclass(item):
                move(item)

    move(clone)
    return clone


def _authoritative(bundle, replay) -> list[torch.Tensor]:
    scalar = bundle.scalar_rng
    assert scalar is not None
    assert replay.rng_buffer_words is not None
    assert replay.semantic_draw_counters is not None
    return [
        *_primary_tensors(bundle), *_inherit_tensors(bundle),
        *_zero_tensors(bundle), *_minus_one_tensors(bundle),
        *_half_tensors(bundle), *_one_tensors(bundle),
        scalar.python_random_words, scalar.python_random_index,
        scalar.python_random_has_gauss, scalar.python_random_gauss_next,
        scalar.python_random_gauss_primary_tape,
        scalar.python_random_gauss_secondary_tape,
        scalar.python_random_raw_draws, scalar.mutation_py_semantic_draws,
        replay.rng_state_words, replay.rng_buffer_words, replay.rng_cursors,
        replay.semantic_draw_counters, replay.event_sequence,
        replay.event_order_keys,
    ]


def _r4_fixture():
    bundle, _ = make_c4_adversarial_fixture_v1("recombination_path", seed=19)
    core = bundle.base.base.core
    for slot in range(bundle.capacity):
        if not bool(core.agents.alive[0, slot]):
            continue
        y = int(core.agents.pos_y[0, slot])
        x = int(core.agents.pos_x[0, slot])
        agent_id = int(core.agents.agent_id[0, slot])
        core.occupancy.agent_id_grid[0, y, x] = agent_id
        core.occupancy.slot_grid[0, y, x] = slot
        bundle.base.base.patch.agent_grid_i32[0, y, x] = agent_id
    genome = bundle.base.genome
    genome.hidden_size[0, 0] = 32
    genome.hidden_size[0, 1] = 64
    genome.mutation_rate[0, :2] = 1.0
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    bundle.base.base.core.policy.packed_weights[0, 0].fill_(0.125)
    bundle.base.base.core.policy.packed_weights[0, 1].fill_(0.875)
    bundle.base.base.policy.packed_weights[0, 0].fill_(0.25)
    bundle.base.base.policy.packed_weights[0, 1].fill_(0.75)
    return bundle


def test_r4_receipt_binds_two_rng_owners_without_gate_or_seed_claim() -> None:
    receipt = c4_resident_mutation_r4_source_receipt_v1()

    assert receipt["decision"] == C4_RESIDENT_MUTATION_R4_DECISION_V1
    assert receipt["continuation_owners"] == [
        "C5_PCG64_MATE_PLACEMENT",
        "C4_CPYTHON_MT19937_SCALAR",
    ]
    assert receipt["hidden_and_policy_inheritance"] == "primary_parent_copy"
    assert receipt["policy_vector_recombination"] == "unsupported_fail_closed"
    assert receipt["production_seed_binding_claim"] is False
    assert receipt["gate_claim"] is False
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())


def test_r4_oracle_budget_and_primary_hidden_policy_contract() -> None:
    bundle, _ = make_c4_adversarial_fixture_v1("recombination_path", seed=19)
    genome = bundle.base.genome
    genome.hidden_size[0, 0] = 32
    genome.hidden_size[0, 1] = 64
    genome.mutation_rate[0, :2] = 1.0
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    core_policy = bundle.base.base.core.policy.packed_weights
    c2_policy = bundle.base.base.policy.packed_weights
    core_policy[0, 0].fill_(0.125)
    core_policy[0, 1].fill_(0.875)
    c2_policy[0, 0].fill_(0.25)
    c2_policy[0, 1].fill_(0.75)
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=19)
    config = TensorWorldC4MateBirthConfigV1(max_births=1, allow_mutation=True)

    plan_c4_mate_births_oracle_v1(bundle, config, rng_context=replay)

    population = bundle.population
    assert population.parent_slots[0, 0].item() == 0
    assert population.mate_slots[0, 0].item() == 1
    assert population.child_slots[0, 0].item() == 3
    assert population.runner_rng_draws_added.item() == 0
    assert population.runner_rng_semantic_draws_added.item() == 1
    assert population.spawn_rng_draws_added.item() == 7
    assert population.spawn_rng_semantic_draws_added.item() == 8

    trial, outputs = preflight_c4_scalar_genome_mutations_v1(
        bundle, world=0, parent_slots=[0], mate_slots=[1]
    )
    assert outputs[0][0] is True
    assert trial.python_random_raw_draws.item() == 42
    assert trial.mutation_py_semantic_draws.item() == 20

    commit_c4_mate_births_oracle_v1(bundle, config, rng_context=replay)

    child = 3
    assert genome.hidden_size[0, child].item() == 32
    assert torch.equal(core_policy[0, child], core_policy[0, 0])
    assert torch.equal(c2_policy[0, child], c2_policy[0, 0])
    assert not torch.equal(core_policy[0, child], core_policy[0, 1])
    assert not torch.equal(c2_policy[0, child], c2_policy[0, 1])


def test_r4_oracle_addresses_refilled_gaussian_tape_relative_to_window_base() -> None:
    bundle = _r4_fixture()
    scalar = bundle.scalar_rng
    assert scalar is not None
    window_base = scalar.tape_capacity
    scalar.python_random_raw_draws.fill_(window_base)

    trial, outputs = preflight_c4_scalar_genome_mutations_v1(
        bundle,
        world=0,
        parent_slots=[0],
        mate_slots=[1],
        gaussian_window_base_raw_draws=window_base,
    )

    assert outputs[0][0] is True
    assert trial.python_random_raw_draws.item() > window_base
    with pytest.raises(TensorWorldContractError, match="exhausted"):
        preflight_c4_scalar_genome_mutations_v1(
            bundle, world=0, parent_slots=[0], mate_slots=[1]
        )
    with pytest.raises(TensorWorldContractError, match="window base"):
        preflight_c4_scalar_genome_mutations_v1(
            bundle,
            world=0,
            parent_slots=[0],
            mate_slots=[1],
            gaussian_window_base_raw_draws=window_base + 1,
        )


def test_r4_replay_validator_is_structural_and_cpu_staging_is_forbidden() -> None:
    bundle, _ = make_c4_adversarial_fixture_v1("recombination_path", seed=7)
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=11)

    assert _validate_replay_structure(bundle, replay) == 2
    replay.rng_buffer_words = None
    with pytest.raises(TensorWorldContractError, match="buffer and semantic"):
        _validate_replay_structure(bundle, replay)
    with pytest.raises(TensorWorldContractError, match="CPU fallback"):
        make_c4_resident_mutation_r4_staging_v1(
            batch=1, capacity=8, streams=2, device="cpu"
        )


def test_r4_unsupported_modes_fail_before_native_load(monkeypatch) -> None:
    bundle, _ = make_c4_adversarial_fixture_v1("recombination_path", seed=7)
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=11)

    def forbidden_loader(**_kwargs):
        raise AssertionError("native loader must not run")

    monkeypatch.setattr(
        "engine.tensor_world.c4_resident_mutation_r4."
        "load_tensor_world_c4_resident_mutation_r4_extension",
        forbidden_loader,
    )
    with pytest.raises(TensorWorldContractError, match="requires source-current"):
        execute_c4_resident_mutation_r4_cuda_v1(
            bundle, object(), TensorWorldC4MateBirthConfigV1(allow_mutation=False),
            rng_context=replay,
        )
    with pytest.raises(TensorWorldContractError, match="policy-vector"):
        execute_c4_resident_mutation_r4_cuda_v1(
            bundle, object(), TensorWorldC4MateBirthConfigV1(allow_mutation=True),
            rng_context=replay, policy_vector_recombination=True,
        )
    with pytest.raises(TensorWorldContractError, match="life-history"):
        execute_c4_resident_mutation_r4_cuda_v1(
            bundle, object(),
            TensorWorldC4MateBirthConfigV1(allow_mutation=True, life_history_mode=1),
            rng_context=replay,
        )


def test_r4_native_source_uses_exact_components_and_avoids_divergent_paths() -> None:
    source = KERNEL.read_text()

    assert 'cuda_architecture_b_persistent_mate_birth_plan_v1.cuh' in source
    assert 'cuda_architecture_b_persistent_birth_scalar_genome.cuh' in source
    assert "CPU C4 consumes this decision but inherits primary hidden size" in source
    assert "stage.python_words" in source
    assert "stage.pcg_state" in source
    assert "output[field] = static_cast<float>(value);" in source
    assert "output[8] = static_cast<float>(child_rate);" in source
    assert "all_ok(stage.base, state.base.batch)" in source
    assert "flattened_mate_birth_scalar_genome" not in source
    assert "policy_birth_crossover" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r4_b1_cuda_exact_parity_and_gaussian_rollback(tmp_path: Path) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R4_BUILD_DIR", tmp_path / "build"))
    build_dir.mkdir(parents=True, exist_ok=True)
    cpu_bundle = _r4_fixture()
    cpu_replay = make_c5_pcg64_context_v1(
        cpu_bundle, streams=2, seed=19, typed=True,
        source_commit="c5-r4-exact-parity-test",
    )
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    cuda_replay = _move_dataclass_tensors(cpu_replay, torch.device("cuda"))
    config = TensorWorldC4MateBirthConfigV1(max_births=1, allow_mutation=True)

    plan_c4_mate_births_oracle_v1(cpu_bundle, config, rng_context=cpu_replay)
    commit_c4_mate_births_oracle_v1(cpu_bundle, config, rng_context=cpu_replay)
    staging = make_c4_resident_mutation_r4_staging_v1(
        batch=1,
        capacity=cpu_bundle.capacity,
        streams=int(cpu_replay.rng_state_words.shape[1]),
        device="cuda",
    )
    execute_c4_resident_mutation_r4_cuda_v1(
        cuda_bundle, staging, config, rng_context=cuda_replay,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()

    assert staging.birth.status.cpu().tolist() == [0]
    assert staging.birth.planned_count.cpu().tolist() == [1]
    assert staging.birth.parent_slots[0, 0].item() == 0
    assert staging.mate_slots[0, 0].item() == 1
    assert staging.birth.child_slots[0, 0].item() == 3
    assert staging.runner_rng_draws_added.item() == 0
    assert staging.runner_rng_semantic_draws_added.item() == 1
    assert staging.birth.spawn_rng_draws_added.item() == 7
    assert staging.birth.spawn_rng_semantic_draws_added.item() == 8
    for index, (expected, actual) in enumerate(zip(
        _authoritative(cpu_bundle, cpu_replay),
        _authoritative(cuda_bundle, cuda_replay),
        strict=True,
    )):
        assert torch.equal(actual.cpu(), expected), (
            f"authoritative tensor index {index} is not byte exact"
        )

    failed_bundle = _r4_fixture()
    failed_replay = make_c5_pcg64_context_v1(failed_bundle, streams=2, seed=19)
    failed_bundle.scalar_rng.python_random_raw_draws.fill_(
        failed_bundle.scalar_rng.tape_capacity
    )
    failed_cuda = _move_dataclass_tensors(failed_bundle, torch.device("cuda"))
    failed_replay_cuda = _move_dataclass_tensors(
        failed_replay, torch.device("cuda")
    )
    failed_staging = make_c4_resident_mutation_r4_staging_v1(
        batch=1, capacity=failed_bundle.capacity, streams=2, device="cuda"
    )
    before = [
        tensor.clone() for tensor in _authoritative(failed_cuda, failed_replay_cuda)
    ]
    execute_c4_resident_mutation_r4_cuda_v1(
        failed_cuda, failed_staging, config, rng_context=failed_replay_cuda,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()
    assert failed_staging.birth.status.cpu().tolist() == [36]
    for expected, actual in zip(
        before, _authoritative(failed_cuda, failed_replay_cuda), strict=True
    ):
        torch.testing.assert_close(actual, expected, equal_nan=True)
