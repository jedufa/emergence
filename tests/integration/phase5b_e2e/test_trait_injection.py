"""tests/integration/phase5b_e2e/test_trait_injection.py

Phase 5B validation tests.

Required tests:
- test_injection_applies_only_at_allowed_phase()
- test_injection_idempotent_by_uuid()
- test_cost_model_duration_and_application_phase()
- test_allele_frequency_multi_allelic_and_sparse()

These tests are intentionally small and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from simulation.control_plane_pkg import ControlPlane
from simulation.control_plane_pkg.commands import TraitInjectionCommand
from simulation.trait_injection_pkg import apply_trait_injections
from simulation.trait_injection_pkg.allele_frequency import compute_allele_frequencies, SPARSE_BUCKET_KEY
from simulation.trait_injection_pkg.cost_model import trait_injection_cost


@dataclass
class DummySimCfg:
    """Minimal sim_cfg stub for Phase 5B tests."""

    trait_injection_apply_phase: str = "post-selection-pre-mutation"


@dataclass
class DummyGenome:
    """Minimal genome stub."""

    memory_dim: int = 16


@dataclass
class DummyAgent:
    """Minimal agent stub."""

    agent_id: int
    genome: DummyGenome
    alive: bool = True


class DummyPopulation:
    """Minimal population stub."""

    def __init__(self, agents: Dict[int, DummyAgent]):
        self.agents = dict(agents)

    def alive_agents(self):
        return [self.agents[k] for k in sorted(self.agents.keys()) if bool(self.agents[k].alive)]


@dataclass
class DummySim:
    """Minimal sim stub compatible with apply_trait_injections."""

    run_dir: str
    sim_cfg: DummySimCfg
    pop: DummyPopulation


def _mk_trait_injection_spec(*, apply_phase: str, duration: int = 1):
    return {
        "schema_version": "trait_injection_v2",
        "duration_generations": int(duration),
        "apply_phase": str(apply_phase),
        "locus_tags": ["memory_dim"],
        "allele_values": {"memory_dim": 64},
        "target_agent_ids": [0, 1],
    }


def test_injection_applies_only_at_allowed_phase(tmp_path):
    run_dir = str(tmp_path / "run")

    pop = DummyPopulation(
        {
            0: DummyAgent(agent_id=0, genome=DummyGenome(memory_dim=16)),
            1: DummyAgent(agent_id=1, genome=DummyGenome(memory_dim=16)),
            2: DummyAgent(agent_id=2, genome=DummyGenome(memory_dim=16)),
        }
    )
    sim_cfg = DummySimCfg(trait_injection_apply_phase="post-selection-pre-mutation")
    sim = DummySim(run_dir=run_dir, sim_cfg=sim_cfg, pop=pop)

    cp = ControlPlane(
        run_dir=run_dir,
        branch_id="b0",
        campaign_id="c0",
        initial_generation=0,
        trait_injection_apply_phase_frozen="post-selection-pre-mutation",
    )

    cmd = TraitInjectionCommand(
        "cmd_v2",
        "00000000-0000-0000-0000-000000000001",
        1,
        trait_injection=_mk_trait_injection_spec(apply_phase="post-selection-pre-mutation", duration=1),
    )

    ack = cp.submit_command(cmd)
    assert ack.status == "ACCEPTED"

    cp.process_pending(1, sb_ctx=None, sim=sim)

    # Wrong phase: no change.
    applied_wrong = apply_trait_injections(sim=sim, generation_index=1, apply_phase="post-reproduction-pre-selection")
    assert applied_wrong == []
    assert sim.pop.agents[0].genome.memory_dim == 16
    assert sim.pop.agents[1].genome.memory_dim == 16

    # Right phase: applied.
    applied_right = apply_trait_injections(sim=sim, generation_index=1, apply_phase="post-selection-pre-mutation")
    assert applied_right == ["00000000-0000-0000-0000-000000000001"]
    assert sim.pop.agents[0].genome.memory_dim == 64
    assert sim.pop.agents[1].genome.memory_dim == 64
    assert sim.pop.agents[2].genome.memory_dim == 16


def test_injection_idempotent_by_uuid(tmp_path):
    run_dir = str(tmp_path / "run")

    pop = DummyPopulation(
        {
            0: DummyAgent(agent_id=0, genome=DummyGenome(memory_dim=16)),
            1: DummyAgent(agent_id=1, genome=DummyGenome(memory_dim=16)),
        }
    )
    sim_cfg = DummySimCfg(trait_injection_apply_phase="post-selection-pre-mutation")
    sim = DummySim(run_dir=run_dir, sim_cfg=sim_cfg, pop=pop)

    cp = ControlPlane(
        run_dir=run_dir,
        branch_id="b0",
        campaign_id="c0",
        initial_generation=0,
        trait_injection_apply_phase_frozen="post-selection-pre-mutation",
    )

    uuid = "00000000-0000-0000-0000-000000000002"
    cmd = TraitInjectionCommand(
        "cmd_v2",
        uuid,
        1,
        trait_injection=_mk_trait_injection_spec(apply_phase="post-selection-pre-mutation", duration=1),
    )

    ack1 = cp.submit_command(cmd)
    ack2 = cp.submit_command(cmd)

    assert ack1.status == "ACCEPTED"
    assert ack2.status == "IDEMPOTENT_ACK"

    cp.process_pending(1, sb_ctx=None, sim=sim)

    # One application.
    applied1 = apply_trait_injections(sim=sim, generation_index=1, apply_phase="post-selection-pre-mutation")
    assert applied1 == [uuid]
    assert sim.pop.agents[0].genome.memory_dim == 64

    # Second call (same gen/phase) should be idempotent by command_uuid.
    applied2 = apply_trait_injections(sim=sim, generation_index=1, apply_phase="post-selection-pre-mutation")
    assert applied2 == []


def test_cost_model_duration_and_application_phase():
    c1 = trait_injection_cost(3, "post-selection-pre-mutation")
    c2 = trait_injection_cost(3, "post-reproduction-pre-selection")
    assert c1 > 0
    assert c2 > 0
    assert c2 != c1

    # Duration must matter.
    assert trait_injection_cost(1, "post-selection-pre-mutation") != trait_injection_cost(2, "post-selection-pre-mutation")


def test_allele_frequency_multi_allelic_and_sparse():
    genomes = [
        DummyGenome(memory_dim=16),
        DummyGenome(memory_dim=16),
        DummyGenome(memory_dim=32),
        DummyGenome(memory_dim=32),
        DummyGenome(memory_dim=32),
        DummyGenome(memory_dim=64),
    ]

    freqs = compute_allele_frequencies(genomes, locus_tags=["memory_dim"], sparse_threshold=0.25)

    assert "memory_dim" in freqs
    md = freqs["memory_dim"]

    # Multi-allelic: 16 and 32 are present.
    assert "16" in md
    assert "32" in md

    # Sparse threshold should bucket the rare allele (64 is 1/6 ~= 0.166).
    assert SPARSE_BUCKET_KEY in md

    # Sanity: frequencies should sum to ~1.0
    s = sum(md.values())
    assert abs(s - 1.0) < 1e-9
