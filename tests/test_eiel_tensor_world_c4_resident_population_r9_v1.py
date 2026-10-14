"""Static, CPU-oracle, and deferred CUDA tests for the B48 C4 R9 seam."""

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
    _half_tensors, _inherit_tensors, _minus_one_tensors, _one_tensors,
    _primary_tensors, _zero_tensors,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors, _scalar_rng_tensors,
)
from engine.tensor_world.c4_resident_population_r9 import (
    C4_RESIDENT_POPULATION_R9_BATCH_V1,
    C4_RESIDENT_POPULATION_R9_DECISION_V1,
    C4_RESIDENT_POPULATION_R9_SCHEMA_VERSION_V1,
    c4_resident_population_r9_source_receipt_v1,
    execute_c4_resident_population_r9_cuda_v1,
    make_c4_resident_population_r9_staging_v1,
)
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu"
BINDING = ROOT / (
    "engine/tensor_world/native/tensor_world_c4_resident_population_r9_binding.cpp"
)


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


def _b48_fixture():
    bundle, _ = make_c4_fixture_v1(
        batch=48, capacity=8, active=3, width=8, seed=59,
        world_local_identity=True,
    )
    core = bundle.base.base.core
    patch = bundle.base.base.patch.agent_grid_i32
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    patch.fill_(-1)
    positions = ((2, 2), (5, 5), (2, 1))
    for world in range(bundle.batch):
        for slot, (y, x) in enumerate(positions):
            agent_id = int(core.agents.agent_id[world, slot])
            core.agents.pos_y[world, slot] = y
            core.agents.pos_x[world, slot] = x
            core.occupancy.agent_id_grid[world, y, x] = agent_id
            core.occupancy.slot_grid[world, y, x] = slot
            patch[world, y, x] = agent_id
        mode = world % 4
        if mode in (1, 3):
            bundle.base.lifecycle.death_candidate[world, 1] = True
            core.agents.health[world, 1] = 0.0
        if mode == 2:
            core.agents.energy[world, :3] = 0.1
            bundle.base.agent.policy_energy_observation[world, :3] = 0.1
        if mode == 3:
            bundle.population.free_queue[world].fill_(-1)
            bundle.population.free_queue_count[world] = 0
    bundle.base.genome.mutation_rate[:, :3] = 1.0
    assert bundle.scalar_rng is not None
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=59)
    config = TensorWorldC4MateBirthConfigV1(
        max_births=1, allow_mutation=True, generation=9
    )
    return bundle, replay, config


def test_c4_r9_receipt_binds_b48_dynamic_launch_without_gate_claim() -> None:
    receipt = c4_resident_population_r9_source_receipt_v1()

    assert receipt["schema_version"] == C4_RESIDENT_POPULATION_R9_SCHEMA_VERSION_V1
    assert receipt["route"] == "R9_B48_ONE_TICK_ALL_LIVE_PREFLIGHT_MATE_MUTATION"
    assert receipt["decision"] == C4_RESIDENT_POPULATION_R9_DECISION_V1
    assert receipt["batch"] == C4_RESIDENT_POPULATION_R9_BATCH_V1 == 48
    assert receipt["launch_geometry"] == "dynamic_batch_blocks"
    assert receipt["r5_r6_source_modified"] is False
    assert receipt["execution_classification"] == (
        "host_launched_device_local_not_cuda_graph_bound"
    )
    assert receipt["runtime_boundary_counters"] == "unmeasured"
    assert receipt["gate_claim"] is False
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())


def test_c4_r9_staging_and_cpu_execute_fail_before_native_load(monkeypatch) -> None:
    with pytest.raises(TensorWorldContractError, match="bounded to B48"):
        make_c4_resident_population_r9_staging_v1(
            batch=4, capacity=8, grid_size=8, streams=2, device="cpu"
        )
    with pytest.raises(TensorWorldContractError, match="CPU fallback"):
        make_c4_resident_population_r9_staging_v1(
            batch=48, capacity=8, grid_size=8, streams=2, device="cpu"
        )
    bundle, replay, config = _b48_fixture()

    def forbidden_loader(**_kwargs):
        raise AssertionError("native loader must not run")

    monkeypatch.setattr(
        "engine.tensor_world.c4_resident_population_r9."
        "load_tensor_world_c4_resident_population_r9_extension",
        forbidden_loader,
    )
    with pytest.raises(TensorWorldContractError, match="requires a CUDA"):
        execute_c4_resident_population_r9_cuda_v1(
            bundle, object(), config, rng_context=replay
        )


def test_c4_r9_native_source_replaces_fixed_four_with_dynamic_batch_launches() -> None:
    source = KERNEL.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")

    assert '#include "tensor_world_c4_resident_preflight_r6.cu"' in source
    assert "batch == 48" in source
    assert "<<<4, 1" not in source
    for kernel in (
        "r6::all_live_semantic_preflight",
        "r5::prepare_post_death_shadow",
        "r4::preflight_kernel",
        "r5::commit_deaths",
        "r4::commit_kernel",
    ):
        assert f"{kernel}<<<batch, 1" in source
    assert "r6::merge_preflight_status<<<1, 1" in source
    assert source.index("r6::merge_preflight_status<<<") < source.index(
        "r5::commit_deaths<<<"
    )
    assert "population_transaction" in binding


def test_c4_r9_cpu_b48_oracle_freezes_mixed_one_tick_sentinels() -> None:
    bundle, replay, config = _b48_fixture()

    apply_c4_death_settlement_oracle_v1(
        bundle, event_sequence=replay.event_sequence
    )
    plan_c4_mate_births_oracle_v1(bundle, config, rng_context=replay)
    commit_c4_mate_births_oracle_v1(bundle, config, rng_context=replay)

    assert bundle.population.death_count.tolist() == [0, 1, 0, 1] * 12
    assert bundle.population.birth_count.tolist() == [1, 1, 0, 1] * 12
    assert bundle.population.child_slots[:, 0].tolist() == [3, 3, -1, 1] * 12
    assert bundle.population.next_agent_id.tolist() == [5, 5, 4, 5] * 12
    assert bundle.population.slot_tokens[:, 1].tolist()[3::4] == [2] * 12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r9_cuda_b48_one_tick_exact_cpu_parity(tmp_path: Path) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R9_BUILD_DIR", tmp_path / "build"))
    cpu_bundle, cpu_replay, config = _b48_fixture()
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    cuda_replay = _move_dataclass_tensors(cpu_replay, torch.device("cuda"))

    apply_c4_death_settlement_oracle_v1(
        cpu_bundle, event_sequence=cpu_replay.event_sequence
    )
    plan_c4_mate_births_oracle_v1(cpu_bundle, config, rng_context=cpu_replay)
    commit_c4_mate_births_oracle_v1(cpu_bundle, config, rng_context=cpu_replay)
    staging = make_c4_resident_population_r9_staging_v1(
        batch=48, capacity=8, grid_size=8, streams=2, device="cuda"
    )
    execute_c4_resident_population_r9_cuda_v1(
        cuda_bundle, staging, config, rng_context=cuda_replay,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()

    assert staging.preflight_status.cpu().tolist() == [0] * 48
    assert staging.turnover.death_count.cpu().tolist() == [0, 1, 0, 1] * 12
    assert staging.mutation.birth.planned_count.cpu().tolist() == [1, 1, 0, 1] * 12
    for expected, actual in zip(
        _authoritative(cpu_bundle, cpu_replay),
        _authoritative(cuda_bundle, cuda_replay),
        strict=True,
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
