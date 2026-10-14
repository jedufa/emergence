"""Focused static and opt-in CUDA proof for the R11 newborn fanout."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_event_ring_r11 import make_c4_event_ring_r11_v1
from engine.tensor_world.c4_events import (
    C4_EVENT_BIRTH_COMMITTED_V1,
    C4_EVENT_DEATH_COMMITTED_V1,
)
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_identity_r11 import bind_c4_identity_authority_r11_v1
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
from engine.tensor_world.c4_resident_composed_r11 import (
    _load_r11_graph_extension_v1,
    _r11_population_call_v1,
    make_c4_resident_composed_r11_staging_v1,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
)
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
BINDING = ROOT / (
    "engine/tensor_world/native/tensor_world_c4_resident_graph_r11_binding.cpp"
)
R5_KERNEL = ROOT / (
    "engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu"
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


def _poison_free_slot_fanout(bundle, *, slot: int) -> None:
    groups = (
        (_inherit_tensors(bundle), 9.0),
        (_zero_tensors(bundle), 7.0),
        (_minus_one_tensors(bundle), 7.0),
        (_half_tensors(bundle), 0.125),
        (_one_tensors(bundle), 0.25),
    )
    for tensors, value in groups:
        for tensor in tensors:
            row = tensor[:, slot]
            if tensor.dtype == torch.bool:
                row.fill_(bool(value))
            else:
                row.fill_(value)


def _mixed_b48_fixture():
    bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=32,
        active=3,
        width=8,
        seed=109,
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
    _poison_free_slot_fanout(bundle, slot=3)
    bundle.base.genome.mutation_rate[:, :3] = 1.0
    assert bundle.scalar_rng is not None
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    bundle.base.base.core.tick = 911
    replay = make_c5_pcg64_context_v1(
        bundle, streams=5, seed=109, typed=True
    )
    config = TensorWorldC4MateBirthConfigV1(
        max_births=1, allow_mutation=True, generation=37
    )
    return bundle, replay, config


def _authoritative_without_local_event_cursors(bundle, replay):
    # Event rows remain comparable; R11 alone moves the two local tape cursor
    # scalars into its durable ring after the transaction.
    return [
        *_primary_tensors(bundle)[:99],
        *_death_authoritative(bundle),
        *_inherit_tensors(bundle),
        *_zero_tensors(bundle),
        *_minus_one_tensors(bundle),
        *_half_tensors(bundle),
        *_one_tensors(bundle),
        *_scalar_rng_tensors(bundle),
        *_replay_tensors(replay),
    ]


def test_r11_fanout_source_is_one_guarded_parallel_descriptor_kernel() -> None:
    native = KERNEL.read_text(encoding="utf-8")
    binding = BINDING.read_text(encoding="utf-8")
    transaction = native.split(
        "void tensor_world_c4_resident_population_r11_graph_cuda", 1
    )[1].split(
        "std::vector<int64_t> tensor_world_c4_resident_r11_newborn_fanout_contract",
        1,
    )[0]

    assert "kNewbornFanoutFieldCount = 97" in native
    assert "std::is_trivially_copyable_v<NewbornFanoutFields>" in native
    assert "batch_ok = all_ok(stage, batch)" in native
    assert "persistent_window_all_ok(persistent_outer_status, batch)" in native
    assert "transaction-local success and the" in native
    assert "newborn fanout tensors must not alias" in native
    assert "__syncthreads();" in native
    assert "element += blockDim.x" in native
    assert transaction.count("fused_newborn_fanout_kernel<<<") == 1
    assert "dispatch_inherit(" not in transaction
    assert "dispatch_fill(" not in transaction
    assert "newborn_fanout_contract" in binding
    assert "captured_graph_node_count" in binding


def test_r11_fanout_descriptor_count_matches_the_registered_reset_abi() -> None:
    bundle, _, _ = _mixed_b48_fixture()
    counts = (
        len(_inherit_tensors(bundle)),
        len(_zero_tensors(bundle)),
        len(_minus_one_tensors(bundle)),
        len(_half_tensors(bundle)),
        len(_one_tensors(bundle)),
    )
    assert counts == (29, 55, 11, 1, 1)
    assert sum(counts) == 97


def test_r11_transaction_start_clears_capture_status_before_semantic_preflight() -> None:
    native = KERNEL.read_text(encoding="utf-8")
    reset = native.split("__global__ void reset_local_event_cursor", 1)[1].split(
        "__global__ void sync_stage_event_cursor", 1
    )[0]
    assert reset.index("status[world] = 0;") < reset.index(
        "replay.event_sequence[world] != ring.producer_cursor[world]"
    )
    assert "status[world] = kRingCursor;" in reset


def test_r11_device_time_death_events_preserve_legacy_r5_scalar_projection() -> None:
    r11 = KERNEL.read_text(encoding="utf-8")
    r5 = R5_KERNEL.read_text(encoding="utf-8")

    assert "generation.data_ptr<int64_t>(), tick.data_ptr<int64_t>()" in r11
    assert "if (config.generation_device != nullptr)" in r5
    assert "event_config.generation = config_generation(config, world);" in r5
    legacy_config = r5.split("Config config{max_births", 1)[1].split(
        "const auto stream", 1
    )[0]
    assert "generation, tick};" in legacy_config
    assert "generation_device" not in legacy_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    os.environ.get("EIEL_C4_R11_FANOUT_PROBE") != "1",
    reason="R11 fused newborn fanout probe is explicitly opt-in",
)
def test_r11_fused_fanout_exact_cpu_parity_and_graph_node_count(
    tmp_path: Path,
) -> None:
    cpu_bundle, cpu_replay, config = _mixed_b48_fixture()
    cuda_bundle = _move_dataclass_tensors(cpu_bundle, torch.device("cuda"))
    cuda_replay = _move_dataclass_tensors(cpu_replay, torch.device("cuda"))

    apply_c4_death_settlement_oracle_v1(
        cpu_bundle, event_sequence=cpu_replay.event_sequence
    )
    plan_c4_mate_births_oracle_v1(
        cpu_bundle, config, rng_context=cpu_replay
    )
    commit_c4_mate_births_oracle_v1(
        cpu_bundle, config, rng_context=cpu_replay
    )
    # R2-R9 intentionally retain generation zero on death rows.  R11 owns
    # resident device time, so its exact expected projection changes only
    # those death-generation cells while every other oracle field is frozen.
    assert cpu_bundle.events is not None
    death_rows = (
        cpu_bundle.events.typed_event_type == C4_EVENT_DEATH_COMMITTED_V1
    )
    cpu_bundle.events.generation.masked_fill_(death_rows, 37)
    cpu_bundle.events.typed_generation.masked_fill_(death_rows, 37)

    assert cuda_bundle.events is not None
    cuda_bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48,
        capacity=32,
        event_capacity=cuda_bundle.events.capacity,
        device="cuda",
    )
    bind_c4_identity_authority_r11_v1(cuda_bundle, semantic=False)
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=32,
        grid_size=8,
        streams=int(cuda_replay.rng_state_words.shape[1]),
        device="cuda",
    )
    staging.generation.fill_(37)
    staging.tick.fill_(911)
    staging.requested_ticks.fill_(1)
    staging.completed_ticks.zero_()
    # Reproduce the post-capture sentinel which previously caused status=1
    # before R6 could begin semantic validation.
    staging.r9.preflight_status.fill_(1)
    build_dir = Path(os.environ.get("EIEL_C4_R11_BUILD_DIR", tmp_path / "build"))
    extension = _load_r11_graph_extension_v1(build_directory=build_dir)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        _r11_population_call_v1(
            extension, cuda_bundle, cuda_replay, staging, config
        )
    graph.instantiate()
    graph_nodes = extension.captured_graph_node_count(graph.raw_cuda_graph())
    graph.replay()
    torch.cuda.synchronize()

    assert extension.newborn_fanout_contract() == [97, 97, 1, 96, 256]
    # Typed RNG continuation adds exactly three guarded nodes: remaining-window
    # admission, counter-shadow preparation, and post-merge counter commit.
    assert graph_nodes == 19
    assert staging.r9.preflight_status.cpu().tolist() == [0] * 48
    assert staging.r9.turnover.death_count.cpu().tolist() == [0, 1, 0, 1] * 12
    assert staging.r9.mutation.birth.planned_count.cpu().tolist() == [1, 1, 0, 1] * 12

    for expected, actual in zip(
        _authoritative_without_local_event_cursors(cpu_bundle, cpu_replay),
        _authoritative_without_local_event_cursors(cuda_bundle, cuda_replay),
        strict=True,
    ):
        torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

    ring = cuda_bundle.event_ring
    assert ring is not None and cpu_bundle.events is not None
    for world in range(48):
        count = int(cpu_bundle.events.active_count[world])
        assert int(ring.bank_counts[world].sum()) == count
        bank = ring.banks[int(ring.active_bank[world])]
        torch.testing.assert_close(
            bank.typed_event_type[world, :count].cpu(),
            cpu_bundle.events.typed_event_type[world, :count],
        )
        torch.testing.assert_close(
            bank.typed_generation[world, :count].cpu(),
            cpu_bundle.events.typed_generation[world, :count],
        )
        torch.testing.assert_close(
            bank.typed_tick[world, :count].cpu(),
            cpu_bundle.events.typed_tick[world, :count],
        )
    mixed_types = ring.banks[0].typed_event_type[3, : int(ring.bank_counts[3, 0])]
    assert C4_EVENT_DEATH_COMMITTED_V1 in mixed_types.cpu().tolist()
    assert C4_EVENT_BIRTH_COMMITTED_V1 in mixed_types.cpu().tolist()
    assert set(ring.banks[0].typed_generation[3, : len(mixed_types)].cpu().tolist()) == {37}
    assert set(ring.banks[0].typed_tick[3, : len(mixed_types)].cpu().tolist()) == {911}
