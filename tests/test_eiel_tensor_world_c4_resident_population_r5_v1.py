"""Focused contract and deferred B4 parity tests for resident C4 R5."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_mating import (
    TensorWorldC4MateBirthConfigV1,
    commit_c4_mate_births_oracle_v1,
    plan_c4_mate_births_oracle_v1,
)
from engine.tensor_world.c4_population import apply_c4_death_settlement_oracle_v1
from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
)
from engine.tensor_world.c4_resident_population_r5 import (
    C4_RESIDENT_POPULATION_R5_DECISION_V1,
    C4_RESIDENT_POPULATION_R5_STATUS_V1,
    c4_resident_population_r5_source_receipt_v1,
    execute_c4_resident_population_r5_cuda_v1,
    make_c4_resident_population_r5_staging_v1,
)
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu"
BINDING = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r5_binding.cpp"


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
    return [
        *_primary_tensors(bundle), *_death_authoritative(bundle),
        *_inherit_tensors(bundle), *_zero_tensors(bundle),
        *_minus_one_tensors(bundle), *_half_tensors(bundle), *_one_tensors(bundle),
        *_scalar_rng_tensors(bundle), *_replay_tensors(replay),
    ]


def _b4_fixture():
    bundle, _ = make_c4_fixture_v1(batch=4, capacity=8, active=4, width=8)
    core = bundle.base.base.core
    patch = bundle.base.base.patch.agent_grid_i32
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    patch.fill_(-1)
    positions = (
        ((2, 2), (2, 1), (0, 4), (6, 6)),
        ((2, 2), (5, 5), (2, 1), (6, 6)),
        ((0, 0), (0, 3), (3, 0), (3, 3)),
        ((2, 2), (2, 1), (5, 5), (6, 6)),
    )
    for world, rows in enumerate(positions):
        for slot, (y, x) in enumerate(rows):
            agent_id = int(core.agents.agent_id[world, slot])
            core.agents.pos_y[world, slot] = y
            core.agents.pos_x[world, slot] = x
            core.occupancy.agent_id_grid[world, y, x] = agent_id
            core.occupancy.slot_grid[world, y, x] = slot
            patch[world, y, x] = agent_id
    bundle.population.free_queue[1].fill_(-1)
    bundle.population.free_queue_count[1] = 0
    bundle.base.lifecycle.death_candidate[1, 1] = True
    bundle.base.lifecycle.death_candidate[3, 2:4] = True
    core.agents.health[1, 1] = 0.0
    core.agents.health[3, 2:4] = 0.0
    core.agents.energy[2, :4] = 0.1
    bundle.base.agent.policy_energy_observation[2, :4] = 0.1
    bundle.base.genome.mutation_rate[:, :4] = 1.0
    scalar = bundle.scalar_rng
    assert scalar is not None
    scalar.python_random_gauss_primary_tape.fill_(0.01)
    scalar.python_random_gauss_secondary_tape.zero_()
    return bundle


def test_r5_receipt_records_b4_atomicity_and_identity_limit() -> None:
    receipt = c4_resident_population_r5_source_receipt_v1()

    assert receipt["decision"] == C4_RESIDENT_POPULATION_R5_DECISION_V1
    assert receipt["route"] == "R5_B4_MIXED_WORLD_DEATH_MATE_MUTATION"
    assert receipt["batch"] == 4
    assert receipt["atomicity"] == "batch_global_guarded_multi_kernel_commit_boundary"
    assert receipt["runtime_fault_atomicity"] == (
        "not_claimed_for_multi_kernel_launch_faults"
    )
    assert receipt["event_order"] == "death_then_mutation_then_birth_per_world"
    assert receipt["identity_capacity"] == (
        "bounded_agent_id_to_slot_domain_fail_closed_no_wrap"
    )
    assert receipt["gate_claim"] is False
    assert C4_RESIDENT_POPULATION_R5_STATUS_V1[43] == (
        "unsupported_identity_capacity"
    )
    assert all(len(source["sha256"]) == 64 for source in receipt["sources"].values())


def test_r5_staging_and_unsupported_modes_fail_before_native_load(monkeypatch) -> None:
    with pytest.raises(TensorWorldContractError, match="bounded to B4"):
        make_c4_resident_population_r5_staging_v1(
            batch=1, capacity=8, grid_size=8, streams=2, device="cpu"
        )
    with pytest.raises(TensorWorldContractError, match="CPU fallback"):
        make_c4_resident_population_r5_staging_v1(
            batch=4, capacity=8, grid_size=8, streams=2, device="cpu"
        )
    bundle = _b4_fixture()
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=19)

    def forbidden_loader(**_kwargs):
        raise AssertionError("native loader must not run")

    monkeypatch.setattr(
        "engine.tensor_world.c4_resident_population_r5."
        "load_tensor_world_c4_resident_population_r5_extension",
        forbidden_loader,
    )
    with pytest.raises(TensorWorldContractError, match="requires source-current"):
        execute_c4_resident_population_r5_cuda_v1(
            bundle, object(), TensorWorldC4MateBirthConfigV1(allow_mutation=False),
            rng_context=replay,
        )
    with pytest.raises(TensorWorldContractError, match="policy-vector"):
        execute_c4_resident_population_r5_cuda_v1(
            bundle, object(), TensorWorldC4MateBirthConfigV1(allow_mutation=True),
            rng_context=replay, policy_vector_recombination=True,
        )
    with pytest.raises(TensorWorldContractError, match="life-history"):
        execute_c4_resident_population_r5_cuda_v1(
            bundle, object(),
            TensorWorldC4MateBirthConfigV1(allow_mutation=True, life_history_mode=1),
            rng_context=replay,
        )


def test_r5_native_source_has_shadow_plan_and_batch_global_commit_guard() -> None:
    source = KERNEL.read_text()
    binding = BINDING.read_text()

    assert '#include "tensor_world_c4_resident_mutation_r4.cu"' in source
    assert "prepare_post_death_shadow<<<4, 1" in source
    assert "r4::preflight_kernel<<<4, 1" in source
    assert "merge_all_status<<<1, 1" in source
    assert "r5::validate_abi(" in source
    assert source.index("merge_all_status<<<") < source.index("dispatch_inherit(")
    assert source.index("commit_deaths<<<") < source.index("r4::commit_kernel<<<")
    assert "if (turn.status[other] != 0 || stage.base.status[other] != 0) return" in source
    assert "kIdentityCapacity" in source
    assert "tensor_world_c4_resident_turnover_r3_cuda(" not in source
    assert "tensor_world_c4_resident_mutation_r4_cuda(" not in source
    assert "population_transaction" in binding


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r5_b4_mixed_exact_parity_and_batch_global_rollback(tmp_path: Path) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R5_BUILD_DIR", tmp_path / "build"))
    config = TensorWorldC4MateBirthConfigV1(
        max_births=1, allow_mutation=True, generation=7
    )
    cpu_bundle = _b4_fixture()
    cpu_replay = make_c5_pcg64_context_v1(cpu_bundle, streams=2, seed=19)
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    cuda_replay = _move_dataclass_tensors(cpu_replay, torch.device("cuda"))

    apply_c4_death_settlement_oracle_v1(
        cpu_bundle, event_sequence=cpu_replay.event_sequence
    )
    plan_c4_mate_births_oracle_v1(cpu_bundle, config, rng_context=cpu_replay)
    commit_c4_mate_births_oracle_v1(cpu_bundle, config, rng_context=cpu_replay)
    staging = make_c4_resident_population_r5_staging_v1(
        batch=4, capacity=8, grid_size=8, streams=2, device="cuda"
    )
    execute_c4_resident_population_r5_cuda_v1(
        cuda_bundle, staging, config, rng_context=cuda_replay,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()

    assert staging.turnover.death_status.cpu().tolist() == [0, 0, 0, 0]
    assert staging.mutation.birth.status.cpu().tolist() == [0, 0, 0, 0]
    assert staging.turnover.death_count.cpu().tolist() == [0, 1, 0, 2]
    assert staging.mutation.birth.planned_count.cpu().tolist() == [1, 1, 0, 1]
    for index, (expected, actual) in enumerate(zip(
        _authoritative(cpu_bundle, cpu_replay),
        _authoritative(cuda_bundle, cuda_replay),
        strict=True,
    )):
        torch.testing.assert_close(
            actual.cpu(), expected, equal_nan=True,
            msg=f"authoritative tensor index {index}",
        )

    failed = _b4_fixture()
    failed_replay = make_c5_pcg64_context_v1(failed, streams=2, seed=19)
    failed.population.next_agent_id[0] = failed.capacity
    failed_cuda = _move_dataclass_tensors(failed, torch.device("cuda"))
    failed_replay_cuda = _move_dataclass_tensors(failed_replay, torch.device("cuda"))
    failed_staging = make_c4_resident_population_r5_staging_v1(
        batch=4, capacity=8, grid_size=8, streams=2, device="cuda"
    )
    before = [
        tensor.clone() for tensor in _authoritative(failed_cuda, failed_replay_cuda)
    ]
    execute_c4_resident_population_r5_cuda_v1(
        failed_cuda, failed_staging, config, rng_context=failed_replay_cuda,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()
    assert failed_staging.turnover.death_status[0].item() == 43
    assert all(value != 0 for value in failed_staging.mutation.birth.status.cpu().tolist())
    for expected, actual in zip(
        before, _authoritative(failed_cuda, failed_replay_cuda), strict=True
    ):
        torch.testing.assert_close(actual, expected, equal_nan=True)
