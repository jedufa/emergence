"""C4 typed event and C5 source-current PCG64 contract tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_genome import _load_cursor
from engine.tensor_world.c4_population import (
    TensorWorldC4BirthConfigV1,
    apply_c4_death_settlement_oracle_v1,
    commit_c4_ordinary_births_oracle_v1,
    plan_c4_ordinary_births_oracle_v1,
)
from engine.tensor_world.c4_mating import (
    TensorWorldC4MateBirthConfigV1,
    commit_c4_mate_births_oracle_v1,
    plan_c4_mate_births_oracle_v1,
)
from engine.tensor_world.c5_determinism import (
    c5_pcg64_bounded_uint32_v1,
    c5_pcg64_uniform_float32_v1,
    make_c5_pcg64_context_v1,
    run_c5_replay_resume_parity_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def test_c4_typed_events_match_source_order_and_payloads() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    config = TensorWorldC4BirthConfigV1(max_births=3)
    plan_c4_ordinary_births_oracle_v1(
        bundle, config, spawn_uniforms=torch.zeros((1, 8, 3), dtype=torch.float64)
    )
    commit_c4_ordinary_births_oracle_v1(bundle, config)

    tape = bundle.events
    assert tape is not None
    assert tape.active_count.tolist() == [3]
    # Decision order is accepted birth, no-space failure, accepted birth.
    assert tape.typed_event_type[0, :3].tolist() == [1, 3, 1]
    assert tape.typed_sequence[0, :3].tolist() == [0, 1, 2]
    assert tape.typed_actor_id[0, :3].tolist() == [1, 2, 3]
    assert tape.typed_target_id[0, :3].tolist() == [4, -1, 5]
    assert tape.typed_reason[0, :3].tolist() == [-1, 1, -1]
    assert tape.typed_success[0, :3].tolist() == [1, 0, 1]


def test_c4_cpython_mt19937_cursor_matches_native_random_continuation() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    cursor = _load_cursor(bundle.scalar_rng, 0)
    reference = random.Random(0)
    assert [cursor.raw32() for _ in range(5)] == [
        reference.getrandbits(32) for _ in range(5)
    ]

    cursor = _load_cursor(bundle.scalar_rng, 0)
    reference = random.Random(0)
    assert [cursor.random_double() for _ in range(5)] == [
        reference.random() for _ in range(5)
    ]


def test_c4_scalar_mutation_advances_cpython_state_and_publishes_mutation_rows() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    bundle.base.genome.mutation_rate[0, :3] = 1.0
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    config = TensorWorldC4BirthConfigV1(max_births=3, allow_mutation=True)
    plan_c4_ordinary_births_oracle_v1(
        bundle, config, spawn_uniforms=torch.zeros((1, 8, 3), dtype=torch.float64)
    )
    before_raw = bundle.scalar_rng.python_random_raw_draws.clone()
    before_semantic = bundle.scalar_rng.mutation_py_semantic_draws.clone()
    before_exploration = bundle.base.genome.exploration_bias.clone()

    commit_c4_ordinary_births_oracle_v1(bundle, config)

    assert bundle.population.mutation_occurred[0, :2].tolist() == [True, True]
    assert torch.all(bundle.population.mutation_delta_total[0, :2] > 0)
    assert int(bundle.scalar_rng.python_random_raw_draws[0]) > int(before_raw[0])
    assert int(bundle.scalar_rng.mutation_py_semantic_draws[0]) > int(before_semantic[0])
    assert not torch.equal(before_exploration[0, 3:5], bundle.base.genome.exploration_bias[0, 3:5])
    assert bundle.events is not None
    assert bundle.events.typed_event_type[0, :5].tolist() == [2, 1, 3, 2, 1]


def test_c4_scalar_mutation_tape_exhaustion_is_precommit() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    bundle.base.genome.mutation_rate[0, :3] = 1.0
    bundle.scalar_rng.python_random_gauss_primary_tape = torch.empty(
        (1, 0), dtype=torch.float64
    )
    bundle.scalar_rng.python_random_gauss_secondary_tape = torch.empty(
        (1, 0), dtype=torch.float64
    )
    config = TensorWorldC4BirthConfigV1(max_births=3, allow_mutation=True)
    plan_c4_ordinary_births_oracle_v1(
        bundle, config, spawn_uniforms=torch.zeros((1, 8, 3), dtype=torch.float64)
    )
    before_ids = bundle.base.base.core.agents.agent_id.clone()
    before_rng = bundle.scalar_rng.python_random_raw_draws.clone()
    with pytest.raises(TensorWorldContractError, match="Gaussian tape"):
        commit_c4_ordinary_births_oracle_v1(bundle, config)
    assert torch.equal(before_ids, bundle.base.base.core.agents.agent_id)
    assert torch.equal(before_rng, bundle.scalar_rng.python_random_raw_draws)


def test_c4_death_event_is_sealed_before_slot_release_and_keeps_lineage() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=4, width=8)
    bundle.base.base.core.agents.health[0, 2] = 0.0
    bundle.base.lifecycle.death_candidate[0, 2] = True
    apply_c4_death_settlement_oracle_v1(bundle)

    tape = bundle.events
    assert tape is not None
    assert tape.active_count.tolist() == [1]
    assert tape.typed_event_type[0, 0].item() == 4
    assert tape.typed_actor_id[0, 0].item() == 3
    assert tape.typed_actor_slot[0, 0].item() == 2
    assert tape.typed_founder_id[0, 0].item() == 3
    assert tape.typed_lineage_depth[0, 0].item() == 0
    assert tape.typed_reason[0, 0].item() == 2
    assert tape.typed_success[0, 0].item() == 1


def test_c4_event_capacity_preflight_does_not_commit_birth_state() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    config = TensorWorldC4BirthConfigV1(max_births=3)
    plan_c4_ordinary_births_oracle_v1(
        bundle, config, spawn_uniforms=torch.zeros((1, 8, 3), dtype=torch.float64)
    )
    assert bundle.events is not None
    bundle.events.active_count.fill_(bundle.events.capacity)
    before_ids = bundle.base.base.core.agents.agent_id.clone()
    with pytest.raises(TensorWorldContractError, match="event tape capacity"):
        commit_c4_ordinary_births_oracle_v1(bundle, config)
    assert torch.equal(bundle.base.base.core.agents.agent_id, before_ids)


def test_c5_pcg64_float32_matches_numpy_reference_and_cursors() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=41)
    reference = np.random.Generator(np.random.PCG64(1234))
    state = reference.bit_generator.state
    state_words = [state["state"]["state"] & ((1 << 64) - 1), state["state"]["state"] >> 64,
                   state["state"]["inc"] & ((1 << 64) - 1), state["state"]["inc"] >> 64]
    signed = [value - (1 << 64) if value >= (1 << 63) else value for value in state_words]
    context.rng_state_words[0, 0] = torch.tensor(signed, dtype=torch.int64)
    values = c5_pcg64_uniform_float32_v1(
        context, bundle, world=0, stream=0, count=6
    )
    expected = reference.random(6, dtype=np.float32)
    assert torch.equal(values, torch.from_numpy(expected))
    assert context.rng_cursors[0, 0].item() == 6


def test_c5_pcg64_bounded_draw_advances_raw_and_semantic_cursors() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    context = make_c5_pcg64_context_v1(bundle, streams=1, seed=7)
    value = c5_pcg64_bounded_uint32_v1(
        context, bundle, world=0, stream=0, max_value=3
    )
    assert 0 <= value <= 3
    assert context.rng_cursors[0, 0].item() >= 1
    assert context.semantic_draw_counters[0, 0].item() == 1


def test_c5_replay_resume_harness_matches_fresh_digest_trajectory() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=4, active=2, width=4)
    context = make_c5_pcg64_context_v1(bundle, streams=1, seed=9)

    def advance(state, replay):
        state.base.base.core.agents.energy[0, 0] += 0.25
        c5_pcg64_bounded_uint32_v1(
            replay, state, world=0, stream=0, max_value=7
        )
        state.events.active_count[0] += 0

    result = run_c5_replay_resume_parity_v1(
        bundle,
        context,
        ticks=3,
        checkpoint_ticks=(1, 2),
        advance=advance,
    )
    assert result["all_equal"] is True
    assert result["comparisons"] == {"1": True, "2": True}


def test_c4_mate_plan_preserves_world_local_identity_and_rng_obligations() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    core = bundle.base.base.core
    # Make biological IDs 1 and 2 adjacent without changing slot identity.
    core.occupancy.agent_id_grid[0, 0, 2] = -1
    core.occupancy.slot_grid[0, 0, 2] = -1
    core.occupancy.agent_id_grid[0, 0, 1] = 2
    core.occupancy.slot_grid[0, 0, 1] = 1
    bundle.base.base.patch.agent_grid_i32[0, 0, 2] = -1
    bundle.base.base.patch.agent_grid_i32[0, 0, 1] = 2
    core.agents.pos_x[0, 1] = 1
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=7)
    before_ids = core.agents.agent_id.clone()

    plan_c4_mate_births_oracle_v1(
        bundle,
        TensorWorldC4MateBirthConfigV1(max_births=2),
        rng_context=context,
    )

    assert torch.equal(core.agents.agent_id, before_ids)
    assert bundle.population.decision_count.tolist() == [3]
    assert bundle.population.decision_parent_slots[0, 0].item() == 0
    assert bundle.population.decision_mate_slots[0, 0].item() == 1
    assert bundle.population.planned_count.item() == 1
    assert bundle.population.mate_slots[0, 0].item() == 1
    assert bundle.population.runner_rng_semantic_draws_added.item() > 0
    assert bundle.population.spawn_rng_semantic_draws_added.item() > 0


def test_c4_mate_commit_debits_both_parents_and_keeps_primary_lineage() -> None:
    bundle, _ = make_c4_fixture_v1(batch=1, capacity=8, active=3, width=8)
    core = bundle.base.base.core
    core.occupancy.agent_id_grid[0, 0, 2] = -1
    core.occupancy.slot_grid[0, 0, 2] = -1
    core.occupancy.agent_id_grid[0, 0, 1] = 2
    core.occupancy.slot_grid[0, 0, 1] = 1
    bundle.base.base.patch.agent_grid_i32[0, 0, 2] = -1
    bundle.base.base.patch.agent_grid_i32[0, 0, 1] = 2
    core.agents.pos_x[0, 1] = 1
    context = make_c5_pcg64_context_v1(bundle, streams=2, seed=7)
    config = TensorWorldC4MateBirthConfigV1(max_births=2)
    plan_c4_mate_births_oracle_v1(bundle, config, rng_context=context)
    parent_energy = core.agents.energy[0, :2].clone()

    commit_c4_mate_births_oracle_v1(bundle, config, rng_context=context)

    assert bundle.population.active_count.tolist() == [4]
    assert core.agents.agent_id[0, 3].item() == 4
    assert core.agents.energy[0, :2].tolist() == [40.0, 40.0]
    assert parent_energy.tolist() == [80.0, 80.0]
    assert bundle.population.lineage_parent_agent_id[0, 3].item() == 1
    assert bundle.events is not None
    assert bundle.events.typed_event_type[0, 0].item() == 1
