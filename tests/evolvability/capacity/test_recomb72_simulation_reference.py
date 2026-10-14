from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from agents.genome import Genome
from evolution.population_pkg.core import Population
from evolution.recomb72_inheritance import (
    AUDITED_LOCI,
    BOUNDED_INTERPOLATION,
    CURRENT_AUTHORITATIVE,
    INHERITANCE_MODES,
    LOCUS_BOUNDS,
    PARENT_COPY_PER_LOCUS,
    WHOLE_PACKAGE_SINGLE_PARENT,
    founder_packages,
)
from evolution.reproduction import crossover_genome
from simulation.runner_pkg.runner_ticks_reproduction import TickReproductionMixin


class CountingRandom(random.Random):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return super().random()


def _genome(package: dict[str, float], *, hidden_size: int = 16) -> Genome:
    genome = Genome(
        hidden_size=hidden_size,
        exploration_bias=package["exploration_bias"],
        aggression_bias=package["aggression_bias"],
        reproduction_threshold=40.0,
        mutation_rate=0.17,
        action_temperature=package["action_temperature"],
        use_gru=False,
        memory_dim=0,
        drive_inertia=package["drive_inertia"],
        drive_noise=0.02,
        drive_logit_gain=1.0,
        pred_lr=package["pred_lr"],
        developmental_program={},
    )
    genome.resource_handling_axis = package["resource_handling_axis"]
    return genome


def _audit_cfg(mode: str = CURRENT_AUTHORITATIVE, **overrides):
    values = {
        "recomb72_audit_enable": True,
        "recomb72_inheritance_mode": mode,
        "recomb72_founder_pattern": "correlated_multi_trait_packages",
        "recomb72_audit_seed": 91,
        "recomb72_disable_ordinary_tick_births": True,
        "recomb72_mutation_off": True,
        "recomb72_neutral_actions_enable": True,
        "recomb72_neutral_action_id": 5,
        "recomb72_neutral_environment_enable": True,
        "policy_weight_mutation_enable": False,
        "mutation_weather_enabled": False,
        "generations": 30,
        "steps_per_generation": 80,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mode", INHERITANCE_MODES)
def test_actual_crossover_boundary_uses_fixed_six_draw_budget(mode: str) -> None:
    package_a, package_b = founder_packages("correlated_multi_trait_packages")
    rng = CountingRandom(117)
    parent_a = _genome(package_a)
    parent_b = _genome(package_b, hidden_size=32)
    parent_a.mutation_rate = 0.0
    parent_b.mutation_rate = 0.0

    child = crossover_genome(
        parent_a,
        parent_b,
        rng,
        sim_cfg=_audit_cfg(mode),
    )

    if mode == CURRENT_AUTHORITATIVE:
        assert rng.draw_count > len(AUDITED_LOCI)
    else:
        assert rng.draw_count == len(AUDITED_LOCI)
    assert child.mutation_rate == 0.0
    event = child._recomb72_inheritance_event
    assert event["mode"] == mode
    assert event["mutation"]["mutation_events_required"] == 0
    if mode == CURRENT_AUTHORITATIVE:
        assert event["production_current_path"] is True
        assert event["child"] == pytest.approx({
            locus: 0.5 * (
                float(getattr(parent_a, locus)) + float(getattr(parent_b, locus))
            )
            for locus in AUDITED_LOCI
        })
    elif mode == PARENT_COPY_PER_LOCUS:
        assert event["fixed_draw_budget"] == len(AUDITED_LOCI)
        assert all(
            event["child"][locus] in {package_a[locus], package_b[locus]}
            for locus in AUDITED_LOCI
        )
    elif mode == BOUNDED_INTERPOLATION:
        assert event["fixed_draw_budget"] == len(AUDITED_LOCI)
        assert all(
            LOCUS_BOUNDS[locus][0]
            <= event["child"][locus]
            <= LOCUS_BOUNDS[locus][1]
            for locus in AUDITED_LOCI
        )
    elif mode == WHOLE_PACKAGE_SINGLE_PARENT:
        assert event["fixed_draw_budget"] == len(AUDITED_LOCI)
        assert event["child"] in (package_a, package_b)


def test_default_off_flag_preserves_production_output_and_rng_state() -> None:
    package_a, package_b = founder_packages("correlated_multi_trait_packages")
    kwargs = {
        "genome_c1_enable": False,
        "genome_c2_enable": False,
        "resource_handling_representation_enable": False,
        "policy_family_head_residual_genome_enable": False,
    }
    rng_absent = random.Random(443)
    rng_false = random.Random(443)

    absent = crossover_genome(
        _genome(package_a),
        _genome(package_b),
        rng_absent,
        sim_cfg=SimpleNamespace(**kwargs),
    )
    explicit_false = crossover_genome(
        _genome(package_a),
        _genome(package_b),
        rng_false,
        sim_cfg=SimpleNamespace(recomb72_audit_enable=False, **kwargs),
    )

    assert absent.to_dict() == explicit_false.to_dict()
    assert rng_absent.getstate() == rng_false.getstate()


def _fake_population(mode: str = CURRENT_AUTHORITATIVE) -> Population:
    package_a, package_b = founder_packages("correlated_multi_trait_packages")
    pop = Population.__new__(Population)
    pop.env_cfg = SimpleNamespace(width=16, height=16, start_energy=20.0)
    pop.pop_cfg = SimpleNamespace(population_size=64)
    pop.sim_cfg = _audit_cfg(mode)
    pop.rng = random.Random(991)
    pop._eiel_current_generation_index = 4
    pop.next_id = 65
    pop.agents = {}
    for agent_id in range(1, 65):
        package = package_a if agent_id <= 32 else package_b
        pop.agents[agent_id] = SimpleNamespace(
            agent_id=agent_id,
            alive=True,
            genome=_genome(package),
            stats=SimpleNamespace(offspring_count=0),
            _recomb72_ancestry={
                "founder_a" if agent_id <= 32 else "founder_b": 1.0
            },
        )

    def remove_dead() -> None:
        pop.agents = {
            agent_id: agent
            for agent_id, agent in pop.agents.items()
            if agent.alive
        }

    calls: list[tuple[int, int]] = []

    def make_child(parent_a, parent_b, child_energy):
        calls.append((parent_a.agent_id, parent_b.agent_id))
        child_id = pop.next_id
        pop.next_id += 1
        child = SimpleNamespace(
            agent_id=child_id,
            alive=True,
            genome=crossover_genome(
                parent_a.genome,
                parent_b.genome,
                pop.rng,
                sim_cfg=pop.sim_cfg,
            ),
            stats=SimpleNamespace(offspring_count=0),
            energy=child_energy,
        )
        pop.agents[child_id] = child
        return child

    pop.remove_dead = remove_dead
    pop.make_child = make_child
    pop._test_make_child_calls = calls
    return pop


def test_neutral_boundary_culls_32_and_refills_through_make_child() -> None:
    pop = _fake_population()

    pop.reproduce_and_cull()

    event = pop._recomb72_generation_boundary_event
    assert len(event["culled_ids"]) == 32
    assert len(event["survivor_ids"]) == 32
    assert event["eligible_parent_ids"] == event["survivor_ids"]
    assert len(event["reproduction_events"]) == 32
    assert len(pop._test_make_child_calls) == 32
    assert len(pop.alive_agents()) == 64
    assert event["mutation_events"] == 0
    assert event["ordinary_tick_births"] == 0
    assert all(
        row["inheritance"]["production_current_path"] is True
        for row in event["reproduction_events"]
    )


def test_neutral_schedule_is_mode_independent() -> None:
    events = []
    for mode in INHERITANCE_MODES:
        pop = _fake_population(mode)
        rng_state = pop.rng.getstate()
        pop.reproduce_and_cull()
        assert pop.rng.getstate() == rng_state
        events.append(pop._recomb72_generation_boundary_event)

    assert len({tuple(event["culled_ids"]) for event in events}) == 1
    assert len(
        {
            tuple(
                (row["parent_a_id"], row["parent_b_id"])
                for row in event["reproduction_events"]
            )
            for event in events
        }
    ) == 1


def test_founder_helper_assigns_exact_32_32_packages() -> None:
    pop = _fake_population()

    receipt = pop.apply_recomb72_founder_packages()

    assert receipt["founder_split"] == {"a": 32, "b": 32}
    agents = pop.alive_agents()
    assert [agent._recomb72_founder_group for agent in agents[:32]] == ["a"] * 32
    assert [agent._recomb72_founder_group for agent in agents[32:]] == ["b"] * 32
    assert all(agent.genome.mutation_rate == 0.0 for agent in agents)


def test_neutral_boundary_fails_closed_outside_t80() -> None:
    pop = _fake_population()
    pop.sim_cfg.steps_per_generation = 79

    with pytest.raises(RuntimeError, match="requires w16p64"):
        pop.reproduce_and_cull()


def test_audit_tick_birth_disable_returns_before_rng_access() -> None:
    class Fixture(TickReproductionMixin):
        sim_cfg = _audit_cfg()

        @property
        def rng_manager(self):
            raise AssertionError("tick-birth audit path touched RNG")

    assert Fixture()._handle_births([], 3, 2, {}) == (3, 2)
