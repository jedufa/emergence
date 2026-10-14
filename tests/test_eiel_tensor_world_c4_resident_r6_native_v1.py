"""Focused native R6 identity/slot adversarial rollback over the R5 route."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

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
    execute_c4_resident_population_r5_cuda_v1,
    make_c4_resident_population_r5_staging_v1,
)
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from tests.test_eiel_tensor_world_c4_resident_population_r5_v1 import _b4_fixture


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


def _corrupt_later_world(bundle, case: str) -> int:
    core = bundle.base.base.core
    population = bundle.population
    if case == "duplicate_free_queue":
        world = 2
        population.free_queue[world, :2] = 4
        population.free_queue_count[world] = 2
        return world
    if case == "conflicting_occupancy":
        world = 3
        y = int(core.agents.pos_y[world, 2])
        x = int(core.agents.pos_x[world, 2])
        core.occupancy.slot_grid[world, y, x] = 0
        return world
    if case == "conflicting_identity_map":
        world = 3
        agent_id = int(core.agents.agent_id[world, 2])
        population.agent_id_to_slot[world, agent_id] = 0
        return world
    raise AssertionError(f"unknown R6 corruption: {case}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r6_native_b4_identity_slot_adversarial_rollback(tmp_path: Path) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R6_BUILD_DIR", tmp_path / "build"))
    config = TensorWorldC4MateBirthConfigV1(
        max_births=1, allow_mutation=True, generation=7
    )

    # Positive sentinel: a fragmented older queue remains authoritative in
    # world 0, while world 1 immediately reuses its newly deceased slot.
    cpu_bundle = _b4_fixture()
    cpu_bundle.population.free_queue[0, :4] = torch.tensor([7, 4, 6, 5])
    cpu_bundle.population.free_queue_count[0] = 4
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
    assert staging.mutation.birth.status.cpu().tolist() == [0, 0, 0, 0]
    assert staging.mutation.birth.child_slots[:, 0].cpu().tolist() == [7, 1, -1, 4]
    assert cuda_bundle.population.free_queue[0, :3].cpu().tolist() == [4, 6, 5]
    for expected, actual in zip(
        _authoritative(cpu_bundle, cpu_replay),
        _authoritative(cuda_bundle, cuda_replay),
        strict=True,
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

    for case in (
        "duplicate_free_queue",
        "conflicting_occupancy",
        "conflicting_identity_map",
    ):
        failed = _b4_fixture()
        invalid_world = _corrupt_later_world(failed, case)
        assert invalid_world > 0
        failed_replay = make_c5_pcg64_context_v1(failed, streams=2, seed=19)
        failed_cuda = _move_dataclass_tensors(failed, torch.device("cuda"))
        failed_replay_cuda = _move_dataclass_tensors(
            failed_replay, torch.device("cuda")
        )
        failed_staging = make_c4_resident_population_r5_staging_v1(
            batch=4, capacity=8, grid_size=8, streams=2, device="cuda"
        )
        before = [
            tensor.clone()
            for tensor in _authoritative(failed_cuda, failed_replay_cuda)
        ]
        execute_c4_resident_population_r5_cuda_v1(
            failed_cuda, failed_staging, config,
            rng_context=failed_replay_cuda, build_directory=build_dir,
        )
        torch.cuda.synchronize()
        death_status = failed_staging.turnover.death_status.cpu().tolist()
        birth_status = failed_staging.mutation.birth.status.cpu().tolist()
        assert death_status[invalid_world] != 0 or birth_status[invalid_world] != 0
        assert all(status != 0 for status in birth_status)
        for expected, actual in zip(
            before,
            _authoritative(failed_cuda, failed_replay_cuda),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, equal_nan=True)
