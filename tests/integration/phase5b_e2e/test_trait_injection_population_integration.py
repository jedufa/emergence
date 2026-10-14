"""tests/integration/phase5b_e2e/test_trait_injection_population_integration.py

Integration regression test:
- TraitInjectionCommand is ACCEPTED + APPLIED into the ControlPlane event log.
- Population.reproduce_and_cull() must apply active injections even when it takes
  the hard-cap early-return path (len(alive) > target).

Why this exists
---------------
In high-birth regimes the population can exceed target size every generation.
The reproduce_and_cull() function then does a hard-cap kill and returns early.
If trait injection hooks are placed *after* that return, injections silently
never run, and the UI shows "PROCESSED" commands with no trait changes.

This test keeps the behavior honest.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from evolution.population_pkg.core import Population
from simulation.control_plane_pkg import ControlPlane
from simulation.control_plane_pkg.commands import TraitInjectionCommand


@dataclass
class DummySimCfg:
    """Minimal sim_cfg stub for trait injection phase freezing."""

    trait_injection_apply_phase: str = "post-selection-pre-mutation"


@dataclass
class DummySim:
    """Minimal sim stub compatible with ControlPlane.process_pending."""

    run_dir: str
    sim_cfg: DummySimCfg
    pop: Population


def _mk_trait_injection_spec(*, apply_phase: str, duration: int = 10):
    return {
        "schema_version": "trait_injection_v2",
        "duration_generations": int(duration),
        "apply_phase": str(apply_phase),
        "locus_tags": ["mutation_rate"],
        "allele_values": {"mutation_rate": 0.2},
        # Empty list means "all agents" in the runtime injector.
        "target_agent_ids": [],
    }


def test_population_reproduce_and_cull_applies_injection_on_hardcap(tmp_path):
    run_dir = str(tmp_path / "run")

    # 1) ControlPlane: create an applied TraitInjectionCommand event.
    cp = ControlPlane(
        run_dir=run_dir,
        branch_id="b0",
        campaign_id="c0",
        initial_generation=0,
        trait_injection_apply_phase_frozen="post-selection-pre-mutation",
    )

    cmd = TraitInjectionCommand(
        "cmd_v2",
        "00000000-0000-0000-0000-00000000HARD",
        1,
        trait_injection=_mk_trait_injection_spec(apply_phase="post-selection-pre-mutation", duration=10),
    )

    ack = cp.submit_command(cmd)
    assert ack.status == "ACCEPTED"

    # Dummy sim is fine: TraitInjectionCommand.apply_to_sim is a no-op.
    env_cfg = EnvConfig()
    pop_cfg = PopConfig(population_size=4, parent_fraction=0.5, cull_fraction=0.0, slot_factor=6)

    rng = random.Random(123)
    np_rng = np.random.Generator(np.random.PCG64(123))

    pop = Population(env_cfg=env_cfg, pop_cfg=pop_cfg, rng=rng, np_rng=np_rng)
    pop.init_random()

    sim_cfg = DummySimCfg(trait_injection_apply_phase="post-selection-pre-mutation")
    sim = DummySim(run_dir=run_dir, sim_cfg=sim_cfg, pop=pop)

    cp.process_pending(1, sb_ctx=None, sim=sim)

    # 2) Force hard-cap path: exceed target population size.
    for _ in range(4):
        extra = pop._spawn_random_agent()
        assert extra is not None

    assert len(pop.alive_agents()) > int(pop_cfg.population_size)

    # 3) Wire the runner-injected fields used by the integration hook.
    pop.run_dir = run_dir
    pop.sim_cfg = sim_cfg
    pop._eiel_current_generation_index = 1

    # Before: mutation_rate should not already be 0.2 everywhere.
    before = [float(a.genome.mutation_rate) for a in pop.alive_agents()]
    assert any(abs(x - 0.2) > 1e-9 for x in before)

    # Act: reproduce_and_cull triggers hard-cap, then must still apply injection.
    pop.reproduce_and_cull()

    after = [float(a.genome.mutation_rate) for a in pop.alive_agents()]
    assert len(after) == int(pop_cfg.population_size)
    assert all(abs(x - 0.2) < 1e-9 for x in after)
