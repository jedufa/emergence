from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from native_runtime.lifecycle_events import (
    LifecycleEventType,
    committed_death_semantic_rows,
    compare_committed_death_semantics,
    empty_typed_lifecycle_event_tensors,
    typed_lifecycle_events_from_reference_contexts,
)
from native_runtime.runner_integration import _REFERENCE_LIFECYCLE_IDENTITY_REGISTRY_ATTR
from perf.cxxbackbone10_resident_state_owner import (
    REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR,
    REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR,
    _legacy_event_tape_from_prefixed_typed_events,
    _merge_prefixed_typed_lifecycle_events,
    _refresh_event_tape_metadata,
    _sync_checkpoint_lineage_identity,
    _typed_lifecycle_event_tensors_for_runner,
    reference_committed_death_oracle,
)
from simulation.runner import SimulationRunner


class _ReferenceEnv:
    def __init__(self, agent_ids: tuple[int, ...]) -> None:
        self.agent_alive = np.zeros((max(agent_ids) + 1,), dtype=np.bool_)
        self.agent_alive[list(agent_ids)] = True

    def remove_agent(self, agent_id: int) -> None:
        self.agent_alive[int(agent_id)] = False


def _reference_runner(*agents: SimpleNamespace) -> SimpleNamespace:
    ids = tuple(int(agent.agent_id) for agent in agents)
    return SimpleNamespace(
        env=_ReferenceEnv(ids),
        pop=SimpleNamespace(agents={int(agent.agent_id): agent for agent in agents}),
        _eiel_fdp_cur_gen=3,
        _eiel_fdp_cur_tick=0,
    )


def _agent(agent_id: int, *, slot: int, energy: float = 5.0, health: float = 5.0) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        slot=int(slot),
        energy=float(energy),
        health=float(health),
        alive=True,
        stats=SimpleNamespace(
            founder_id=int(agent_id),
            lineage_depth=0,
            parent_ids=(),
        ),
    )


def _unprefixed(events: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "typed_lifecycle_event_"
    return {name.removeprefix(prefix): tensor for name, tensor in events.items()}


def _legacy_death_projection(
    *,
    actors: list[int],
    generations: list[int],
    ticks: list[int],
    reasons: list[int],
    targets: list[int] | None = None,
) -> dict[str, object]:
    count = len(actors)
    return {
        "metadata": {"event_tape_active_count": count},
        "tensors": {
            "event_tape_type": torch.full((count,), int(LifecycleEventType.DEATH_COMMITTED), dtype=torch.int32),
            "event_tape_generation": torch.tensor(generations, dtype=torch.int64),
            "event_tape_tick": torch.tensor(ticks, dtype=torch.int64),
            "event_tape_actor_id": torch.tensor(actors, dtype=torch.int64),
            "event_tape_target_id": torch.tensor(targets or [-1] * count, dtype=torch.int64),
            "event_tape_value": torch.tensor(reasons, dtype=torch.float32),
        },
    }


def test_reference_oracle_captures_exact_committed_deaths_in_stable_order() -> None:
    energy_death = _agent(9, slot=4)
    health_death = _agent(3, slot=1)
    second_health_death = _agent(6, slot=2)
    runner = _reference_runner(energy_death, health_death, second_health_death)

    with reference_committed_death_oracle(runner) as contexts:
        runner._eiel_fdp_cur_tick = 5
        energy_death.energy = 0.0
        energy_death.alive = False
        runner.env.remove_agent(energy_death.agent_id)

        runner._eiel_fdp_cur_tick = 2
        health_death.health = 0.0
        health_death.alive = False
        runner.env.remove_agent(health_death.agent_id)

        second_health_death.health = 0.0
        second_health_death.alive = False
        runner.env.remove_agent(second_health_death.agent_id)

    assert [(row["agent_id"], row["tick"], row["reason_code"]) for row in contexts] == [
        (9, 5, 1),
        (3, 2, 2),
        (6, 2, 2),
    ]

    prefixed = _typed_lifecycle_event_tensors_for_runner(runner)
    events = _unprefixed(prefixed)
    assert events["event_type"].tolist() == [4, 4, 4]
    assert events["generation"].tolist() == [3, 3, 3]
    assert events["tick"].tolist() == [2, 2, 5]
    assert events["sequence"].tolist() == [0, 1, 2]
    assert events["actor_agent_id"].tolist() == [3, 6, 9]
    assert events["target_agent_id"].tolist() == [-1, -1, -1]
    assert events["value"].tolist() == [2.0, 2.0, 1.0]
    assert [row["reason_code"] for row in committed_death_semantic_rows(events)] == [2, 2, 1]
    assert "remove_agent" not in runner.env.__dict__


def test_reference_oracle_rejects_dead_removal_without_native_terminal_reason() -> None:
    agent = _agent(7, slot=2)
    runner = _reference_runner(agent)
    agent.alive = False

    with reference_committed_death_oracle(runner):
        with pytest.raises(RuntimeError, match="without terminal energy/health"):
            runner.env.remove_agent(agent.agent_id)

    assert bool(runner.env.agent_alive[agent.agent_id]) is True


def test_reference_oracle_requires_timestamp_before_committing_removal() -> None:
    agent = _agent(7, slot=2, energy=0.0)
    runner = _reference_runner(agent)
    agent.alive = False
    del runner._eiel_fdp_cur_gen
    del runner._eiel_fdp_cur_tick

    with reference_committed_death_oracle(runner):
        with pytest.raises(RuntimeError, match="no authoritative generation"):
            runner.env.remove_agent(agent.agent_id)

    assert bool(runner.env.agent_alive[agent.agent_id]) is True


def test_production_reference_tick_hook_captures_death_and_bypasses_native(monkeypatch) -> None:
    reference_agent = _agent(12, slot=5)
    reference = _reference_runner(reference_agent)
    reference.sim_cfg = SimpleNamespace(
        native_runtime_cxx26_scope_lock_enable=True,
        native_safe_default_enable=False,
        native_runtime_enable=False,
        native_runtime_diagnostic_enable=False,
    )

    def execute_tick_segment(runner, *args, **kwargs):
        runner._eiel_fdp_cur_gen = int(kwargs["gen"])
        runner._eiel_fdp_cur_tick = 17
        agent = runner.pop.agents[12]
        agent.energy = 0.0
        agent.alive = False
        runner.env.remove_agent(agent.agent_id)
        return "reference-segment-complete"

    production_hook = SimulationRunner._run_ticks
    assert bool(getattr(production_hook, "_eiel_reference_committed_death_oracle_hook", False))
    monkeypatch.setattr(production_hook, "_eiel_reference_run_delegate", execute_tick_segment)

    assert production_hook(reference, gen=6) == "reference-segment-complete"
    contexts = getattr(reference, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR)
    assert [(row["generation"], row["tick"], row["agent_id"], row["reason_code"]) for row in contexts] == [
        (6, 17, 12, 1)
    ]
    assert _unprefixed(_typed_lifecycle_event_tensors_for_runner(reference))["event_type"].tolist() == [4]

    native_agent = _agent(12, slot=5)
    native = _reference_runner(native_agent)
    native.sim_cfg = SimpleNamespace(
        native_runtime_cxx26_scope_lock_enable=True,
        native_safe_default_enable=True,
        native_runtime_enable=True,
        native_runtime_diagnostic_enable=False,
    )
    assert production_hook(native, gen=6) == "reference-segment-complete"
    assert not hasattr(native, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR)


def test_production_reference_hooks_finalize_reset_boundary_without_duplication(monkeypatch) -> None:
    first_death = _agent(12, slot=5)
    second_death = _agent(13, slot=6)
    reference = _reference_runner(first_death, second_death)
    setattr(
        reference,
        _REFERENCE_LIFECYCLE_IDENTITY_REGISTRY_ATTR,
        {
            agent_id: {
                "slot": slot,
                "lineage_founder_id": founder_id,
                "lineage_depth": depth,
                "parent_agent_id": parent_id,
            }
            for agent_id, slot, founder_id, depth, parent_id in (
                (100, 20, 100, 0, -1),
                (200, 21, 100, 1, 100),
                (101, 22, 101, 0, -1),
                (201, 23, 101, 1, 101),
                (150, 24, 150, 0, -1),
                (250, 25, 150, 1, 150),
            )
        },
    )
    reference.sim_cfg = SimpleNamespace(
        native_runtime_cxx26_scope_lock_enable=True,
        native_safe_default_enable=False,
        native_runtime_enable=False,
        native_runtime_diagnostic_enable=False,
        lineage_reproduction_funnel_reporting_enable=False,
        lineage_reproduction_funnel_write_events=False,
    )

    def execute_tick_segment(runner, *args, **kwargs):
        assert runner.sim_cfg.lineage_reproduction_funnel_reporting_enable is True
        assert runner.sim_cfg.lineage_reproduction_funnel_write_events is True
        generation = int(kwargs["gen"])
        tick = 17 + generation
        runner._eiel_fdp_cur_gen = generation
        runner._eiel_fdp_cur_tick = tick
        runner._lineage_reproduction_funnel_event_contexts_gen = [
            {
                "generation": generation,
                "tick": tick,
                "event_type": "birth_success",
                "parent_id": 100 + generation,
                "child_id": 200 + generation,
                "birth_success": 1,
            }
        ]
        agent = runner.pop.agents[12 + generation]
        agent.health = 0.0
        agent.alive = False
        runner.env.remove_agent(agent.agent_id)
        return generation

    production_hook = SimulationRunner._run_ticks
    monkeypatch.setattr(production_hook, "_eiel_reference_run_delegate", execute_tick_segment)

    assert production_hook(reference, gen=0) == 0

    accumulated = getattr(reference, REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR)
    assert accumulated["typed_lifecycle_event_event_type"].tolist() == [4, 1]

    reference._lineage_reproduction_funnel_event_contexts_gen.append(
        {
            "generation": 0,
            "tick": 18,
            "event_type": "birth_success",
            "parent_id": 150,
            "child_id": 250,
            "birth_success": 1,
        }
    )
    first_projection = _unprefixed(_typed_lifecycle_event_tensors_for_runner(reference))
    second_projection = _unprefixed(_typed_lifecycle_event_tensors_for_runner(reference))
    assert first_projection["event_type"].tolist() == [4, 1, 1]
    assert second_projection["event_type"].tolist() == [4, 1, 1]
    assert accumulated["typed_lifecycle_event_event_type"].tolist() == [4, 1]

    def reset_generation_contexts(runner, *args, **kwargs):
        runner._lineage_reproduction_funnel_event_contexts_gen = []
        return "generation-reset"

    start_hook = SimulationRunner._start_generation
    assert bool(getattr(start_hook, "_eiel_reference_lifecycle_pre_reset_hook", False))
    monkeypatch.setattr(
        start_hook,
        "_eiel_reference_start_generation_delegate",
        reset_generation_contexts,
    )
    assert start_hook(reference, gen=1) == "generation-reset"
    accumulated = getattr(reference, REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR)
    assert accumulated["typed_lifecycle_event_event_type"].tolist() == [4, 1, 1]
    assert accumulated["typed_lifecycle_event_actor_agent_id"].tolist() == [12, 100, 150]

    assert production_hook(reference, gen=1) == 1

    accumulated = getattr(reference, REFERENCE_LIFECYCLE_EVENT_ACCUMULATOR_ATTR)
    projected = _unprefixed(_typed_lifecycle_event_tensors_for_runner(reference))
    assert accumulated["typed_lifecycle_event_event_type"].tolist() == [4, 1, 1, 4, 1]
    assert projected["event_type"].tolist() == [4, 1, 1, 4, 1]
    assert projected["generation"].tolist() == [0, 0, 0, 1, 1]
    assert projected["tick"].tolist() == [17, 17, 18, 18, 18]
    assert projected["sequence"].tolist() == [0, 1, 2, 3, 4]
    assert projected["actor_agent_id"].tolist() == [12, 100, 150, 13, 101]
    assert len(getattr(reference, REFERENCE_COMMITTED_DEATH_CONTEXT_ATTR)) == 2
    assert len(reference._lineage_reproduction_funnel_event_contexts_gen) == 1
    assert reference.sim_cfg.lineage_reproduction_funnel_reporting_enable is False
    assert reference.sim_cfg.lineage_reproduction_funnel_write_events is False


def test_committed_death_semantic_compare_is_non_vacuous_and_reports_exact_field() -> None:
    reference = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "event_type": "death_committed",
                "generation": 4,
                "tick": 6,
                "agent_id": 12,
                "target_agent_id": -1,
                "reason_code": 2,
                "value": 2.0,
            }
        ]
    )
    matching_native = _legacy_death_projection(
        actors=[12],
        generations=[4],
        ticks=[6],
        reasons=[2],
    )

    matching = compare_committed_death_semantics(reference, matching_native)
    assert matching["ok"] is True
    assert matching["non_vacuous"] is True
    assert matching["reference_count"] == matching["candidate_count"] == 1

    wrong_actor = _legacy_death_projection(
        actors=[13],
        generations=[4],
        ticks=[6],
        reasons=[2],
    )
    mismatch = compare_committed_death_semantics(reference, wrong_actor)
    assert mismatch["ok"] is False
    assert mismatch["first_mismatch"] == {
        "path": "$.committed_deaths[0].actor_agent_id",
        "reference": 12,
        "candidate": 13,
    }

    omitted = compare_committed_death_semantics(reference, empty_typed_lifecycle_event_tensors())
    assert omitted["ok"] is False
    assert omitted["first_mismatch"]["path"] == "$.committed_deaths.length"

    vacuous = compare_committed_death_semantics(
        empty_typed_lifecycle_event_tensors(),
        empty_typed_lifecycle_event_tensors(),
    )
    assert vacuous["ok"] is False
    assert vacuous["non_vacuous"] is False
    assert vacuous["first_mismatch"]["path"] == "$.committed_deaths.reference_count"


def test_committed_death_semantics_preserve_authoritative_same_tick_order() -> None:
    reference = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "event_type": "death_committed",
                "generation": 0,
                "tick": 52,
                "agent_id": actor_id,
                "reason_code": 2,
            }
            for actor_id in (129, 3)
        ]
    )
    sorted_candidate = _legacy_death_projection(
        actors=[3, 129],
        generations=[0, 0],
        ticks=[52, 52],
        reasons=[2, 2],
    )

    comparison = compare_committed_death_semantics(reference, sorted_candidate)

    assert committed_death_semantic_rows(reference)[0]["actor_agent_id"] == 129
    assert comparison["ok"] is False
    assert comparison["first_mismatch"] == {
        "path": "$.committed_deaths[0].actor_agent_id",
        "reference": 129,
        "candidate": 3,
    }


def test_committed_death_semantics_fail_closed_on_non_sentinel_target() -> None:
    reference = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "event_type": "death_committed",
                "generation": 1,
                "tick": 2,
                "agent_id": 8,
                "reason_code": 1,
            }
        ]
    )
    malformed = _legacy_death_projection(
        actors=[8],
        generations=[1],
        ticks=[2],
        reasons=[1],
        targets=[99],
    )

    comparison = compare_committed_death_semantics(reference, malformed)

    assert comparison["ok"] is False
    assert "target_agent_id must be -1" in comparison["errors"]["candidate"]


def test_checkpoint_lifecycle_merge_continues_sequence_and_event_cursor() -> None:
    boundary = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "event_type": "death_committed",
                "generation": 0,
                "tick": 3,
                "agent_id": 10,
                "reason_code": 1,
            },
            {
                "event_type": "death_committed",
                "generation": 1,
                "tick": 4,
                "agent_id": 11,
                "reason_code": 2,
            },
        ]
    )
    local = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "event_type": "death_committed",
                "generation": 2,
                "tick": 7,
                "agent_id": 12,
                "reason_code": 2,
            }
        ]
    )
    boundary_prefixed = {f"typed_lifecycle_event_{name}": tensor for name, tensor in boundary.items()}
    local_prefixed = {f"typed_lifecycle_event_{name}": tensor for name, tensor in local.items()}

    merged = _merge_prefixed_typed_lifecycle_events(boundary_prefixed, local_prefixed)
    tensors = {**merged, **_legacy_event_tape_from_prefixed_typed_events(merged)}
    metadata: dict[str, object] = {}
    _refresh_event_tape_metadata(metadata, tensors)

    assert merged["typed_lifecycle_event_sequence"].tolist() == [0, 1, 2]
    assert merged["typed_lifecycle_event_actor_agent_id"].tolist() == [10, 11, 12]
    assert metadata["event_tape_active_count"] == 3
    assert metadata["event_tape_cursor"] == 3
    assert metadata["typed_lifecycle_event_active_count"] == 3


def test_reference_death_context_requires_identity_timestamp_and_reason() -> None:
    with pytest.raises(ValueError, match="generation and tick"):
        typed_lifecycle_events_from_reference_contexts(
            [{"event_type": "death_committed", "agent_id": 1, "reason_code": 1}]
        )
    with pytest.raises(ValueError, match="actor agent identity"):
        typed_lifecycle_events_from_reference_contexts(
            [{"event_type": "death_committed", "generation": 0, "tick": 0, "reason_code": 1}]
        )
    with pytest.raises(ValueError, match="reason_code/value"):
        typed_lifecycle_events_from_reference_contexts(
            [{"event_type": "death_committed", "generation": 0, "tick": 0, "agent_id": 1}]
        )


def test_checkpoint_apply_syncs_flat_asexual_lineage_id_from_founder_only() -> None:
    stats = SimpleNamespace(founder_id=256, lineage_id="L00000256")

    synced = _sync_checkpoint_lineage_identity(
        stats,
        41,
        metadata={"world": "flat", "reproduce_requires_mate": False},
    )

    assert synced is True
    assert stats.founder_id == 41
    assert stats.lineage_id == "L00000041"

    mating_stats = SimpleNamespace(founder_id=256, lineage_id="MATING-LINEAGE")
    synced_mating = _sync_checkpoint_lineage_identity(
        mating_stats,
        41,
        metadata={"world": "flat", "reproduce_requires_mate": True},
    )
    assert synced_mating is False
    assert mating_stats.founder_id == 41
    assert mating_stats.lineage_id == "MATING-LINEAGE"
