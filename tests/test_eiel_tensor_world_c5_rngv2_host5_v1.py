from __future__ import annotations

import os

import pytest
import torch

from engine.tensor_world.c4_resident_composed_r11 import (
    C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from engine.tensor_world.c5_rngv2_cpu import mutate_planned_births_v2
from engine.tensor_world.c5_rngv2_graph import (
    GRAPH_WINDOW_TICKS,
    bind_v2_graph_fusion_owner,
    run_v2_graph_windows,
)
from engine.tensor_world.c5_rngv2_route import V1_ROUTE_ID, resolve_rng_route


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

BUILD = "/tmp/emergence-test/runtime/build_cache/tw_rngv2_graph_fusion"
C2 = (
    "/tmp/emergence-test/runtime/evidence_raw/tensor_world/c7_current/"
    "20260817-c7-fullroute-source-1d2bd319b-05/bin/tensor_world_c2_ext.so"
)


def _owner(*, seed: int = 41):
    os.environ.setdefault("TENSOR_WORLD_C2_PREBUILT", C2)
    return bind_v2_graph_fusion_owner(
        capacity=32,
        active=4,
        width=8,
        seed=seed,
        source_commit="host5-q2",
        max_births=1,
        native_build_directory=BUILD,
    )


def test_default_v1_route_unchanged() -> None:
    assert resolve_rng_route()["route_id"] == V1_ROUTE_ID
    assert GRAPH_WINDOW_TICKS == 5
    assert C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1 == 5


def test_two_window_replay_single_host_seed() -> None:
    owner = _owner()
    run = run_v2_graph_windows(owner, windows=2, start_tick=0, generation=0)
    assert run["ticks"] == 10
    assert run["graph_replays"] == 2
    assert run["graph_recaptures"] == 0
    assert run["refills"] == 0
    assert run["host_prepare_calls"] == 1
    assert run["host_event_reclaim_copies"] == 0
    assert run["mutation_birth_tape_bytes"] == 0
    assert int(owner.shadow_staging.tick[0].item()) == 10
    assert int(owner.shadow_staging.generation[0].item()) == 5


def test_twenty_five_tick_generation_cycle() -> None:
    owner = _owner(seed=7)
    prepare_c4_resident_composed_r11_window_control_v1(owner, generation=0, tick=0)
    assert owner.v2_host_prepare_calls == 1
    ticks = []
    generations = []
    for _ in range(5):
        execute_c4_resident_composed_r11_window_cuda_v1(owner)
        torch.cuda.synchronize()
        ticks.append(int(owner.shadow_staging.tick[0].item()))
        generations.append(int(owner.shadow_staging.generation[0].item()))
    assert ticks == [5, 10, 15, 20, 25]
    assert generations == [5, 5, 5, 5, 5]
    assert owner.v2_host_prepare_calls == 1
    assert owner.v2_host_reclaim_copies == 0


def test_device_reclaim_matches_drop_policy() -> None:
    owner = _owner(seed=11)
    prepare_c4_resident_composed_r11_window_control_v1(owner, generation=0, tick=0)
    execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    producer = int(owner.bundle.event_ring.producer_cursor[0].item())
    execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    # After the next replay's first node, previous producer was dropped.
    consumer = int(owner.bundle.event_ring.consumer_cursor[0].item())
    assert consumer >= producer
    assert owner.v2_host_reclaim_copies == 0


def test_sticky_status_blocks_authoritative_mutation() -> None:
    owner = _owner(seed=13)
    prepare_c4_resident_composed_r11_window_control_v1(owner, generation=0, tick=0)
    execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    live_ids = owner.bundle.population.next_agent_id.detach().clone()
    live_tick = owner.shadow_staging.tick.detach().clone()
    live_expl = owner.bundle.base.genome.exploration_bias.detach().clone()
    owner.shadow_staging.global_status.fill_(7)
    execute_c4_resident_composed_r11_window_cuda_v1(owner)
    torch.cuda.synchronize()
    # merge_window_control latches any failure to 1; the injected code need not survive.
    assert int(owner.shadow_staging.global_status.max().item()) != 0
    assert torch.equal(live_ids, owner.bundle.population.next_agent_id)
    assert torch.equal(live_tick, owner.shadow_staging.tick)
    assert torch.equal(live_expl, owner.bundle.base.genome.exploration_bias)


def test_qualified_oracle_still_directed() -> None:
    genes = (0.1, -0.2, 10.0, 1.0, 0.4, 0.02, 0.5, 0.1)
    out = mutate_planned_births_v2(
        world_uid=0,
        absolute_tick=9,
        plans=((2, 5, genes, genes, 0.2, 0.2, 0), (5, 2, genes, genes, 0.2, 0.2, 0)),
    )
    assert out[0].child_genes != out[1].child_genes
    owner = _owner()
    run = run_v2_graph_windows(owner, windows=2, start_tick=0, generation=0)
    assert run["mutation_birth_h2d_bytes"] == 0
    assert run["host_prepare_calls"] == 1
