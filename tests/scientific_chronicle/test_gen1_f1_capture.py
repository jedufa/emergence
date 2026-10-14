from __future__ import annotations

import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np

from agents.genome import Genome
from agents.modular_genome_v1 import canonical_hash_from_genome_dict
from env.env_config import EnvConfig
from evolution.reproduction import crossover_genome
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.chronicle.gen1_f1 import Gen1F1Capture
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _genome(value: float) -> Genome:
    return Genome(
        hidden_size=8,
        exploration_bias=value,
        aggression_bias=0.1,
        reproduction_threshold=20.0,
        mutation_rate=0.25,
        action_temperature=1.0,
        use_gru=False,
        memory_dim=0,
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.0,
        pred_lr=0.08,
        developmental_program={},
    )


def _cfg(*, enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        gen1_f1_capture_enable=enabled,
        genome_c1_enable=True,
        genome_c2_enable=True,
        campaign_id="campaign-a",
        branch_id="main",
        chronicle_run_id="run-a",
        seed=17,
        scenario="flat",
        code_commit_hash="a" * 40,
        execution_route_hash="b" * 64,
    )


def _agent(agent_id: int, genome: Genome, lineage: str) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        genome=genome,
        stats=SimpleNamespace(lineage_id=lineage),
    )


def test_capture_instrumentation_is_rng_and_genome_neutral() -> None:
    parent_a = _genome(0.1)
    parent_b = _genome(0.4)
    rng_off = random.Random(991)
    rng_on = random.Random(991)

    child_off = crossover_genome(parent_a, parent_b, rng_off, sim_cfg=_cfg(enabled=False))
    child_on = crossover_genome(parent_a, parent_b, rng_on, sim_cfg=_cfg(enabled=True))

    assert child_on.to_dict() == child_off.to_dict()
    assert rng_on.getstate() == rng_off.getstate()
    stages = child_on._gen1_f1_canonical_stages
    assert stages["stage_boundary"] == "post_c2_pre_c1_to_post_c1"
    assert canonical_hash_from_genome_dict(stages["child_post_mutation_genome"]) == (
        canonical_hash_from_genome_dict(child_on.to_dict())
    )
    assert not hasattr(child_off, "_gen1_f1_canonical_stages")


def test_birth_capture_reconciles_objects_and_feeds_attribution(tmp_path: Path) -> None:
    cfg = _cfg(enabled=True)
    sim = SimpleNamespace(sim_cfg=cfg, run_dir=tmp_path / "run-a")
    capture = Gen1F1Capture(sim)
    parent_a = _agent(1, _genome(0.1), "L00000001")
    parent_b = _agent(2, _genome(0.4), "L00000002")
    child_genome = crossover_genome(
        parent_a.genome,
        parent_b.genome,
        random.Random(88),
        sim_cfg=cfg,
    )
    child = _agent(3, child_genome, "L00000001")

    birth_id = capture.record_birth(
        parent_a=parent_a,
        parent_b=parent_b,
        child=child,
        generation=4,
        step=9,
        population_size=3,
    )
    assert birth_id
    capture.flush()
    births = [
        json.loads(line)
        for line in (capture.root / "gen1_f1_birth_records.jsonl").read_text().splitlines()
    ]
    assert len(births) == 1
    record = births[0]
    assert record["birth_event_id"] == birth_id
    assert record["parent_agent_ids"] == [1, 2]
    for field in (
        "child_pre_mutation_genome_object_ref",
        "child_post_mutation_genome_object_ref",
        "final_attached_child_genome_object_ref",
    ):
        object_path = capture.root / record[field]
        assert object_path.is_file()
        assert canonical_hash_from_genome_dict(json.loads(object_path.read_text())) == (
            record[field.replace("_object_ref", "_hash")]
        )
    mutation_events = capture.root / "attribution" / "mutation_events.jsonl"
    assert mutation_events.is_file()
    assert all(
        json.loads(line)["causal_status"] == "not_assessed"
        for line in mutation_events.read_text().splitlines()
    )


def test_state_roundtrip_is_duplicate_safe_and_carriers_are_stable(tmp_path: Path) -> None:
    cfg = _cfg(enabled=True)
    parent_a = _agent(1, _genome(0.1), "L00000001")
    parent_b = _agent(2, _genome(0.4), "L00000002")
    child = _agent(
        3,
        crossover_genome(parent_a.genome, parent_b.genome, random.Random(77), sim_cfg=cfg),
        "L00000001",
    )
    first = Gen1F1Capture(SimpleNamespace(sim_cfg=cfg, run_dir=tmp_path / "first"))
    birth_id = first.record_birth(
        parent_a=parent_a,
        parent_b=parent_b,
        child=child,
        generation=1,
        step=2,
    )
    first.observe_carriers(generation=1, agents=[parent_a, parent_b, child])
    state = first.export_state()

    restored = Gen1F1Capture(SimpleNamespace(sim_cfg=cfg, run_dir=tmp_path / "restored"))
    restored.import_state(state)
    duplicate_id = restored.record_birth(
        parent_a=parent_a,
        parent_b=parent_b,
        child=child,
        generation=1,
        step=2,
    )
    restored.observe_carriers(generation=1, agents=[child, parent_b, parent_a])

    assert duplicate_id == birth_id
    roundtrip = restored.export_state()
    assert roundtrip["cursor"] == 1
    assert len(roundtrip["birth_records"]) == 1
    assert len(roundtrip["carrier_records"]) == 3
    assert roundtrip["attribution_state"] == state["attribution_state"]


def test_missing_canonical_stage_fails_closed_without_birth_record(tmp_path: Path) -> None:
    cfg = _cfg(enabled=True)
    capture = Gen1F1Capture(SimpleNamespace(sim_cfg=cfg, run_dir=tmp_path / "run"))
    child = _agent(3, _genome(0.2), "L00000001")
    assert capture.record_birth(
        parent_a=_agent(1, _genome(0.1), "L00000001"),
        parent_b=_agent(2, _genome(0.4), "L00000002"),
        child=child,
        generation=1,
        step=1,
    ) is None
    assert capture.export_state()["birth_records"] == []
    assert capture.capture_errors == 1


def test_runner_capture_on_off_preserves_trajectory_and_rng(tmp_path: Path) -> None:
    def run(enabled: bool) -> tuple[dict, Path]:
        cfg = SimConfig(
            seed=812,
            generations=1,
            steps_per_generation=2,
            device="cpu",
            bank_device="cpu",
            torch_threads=1,
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            chronicle_enable=False,
            gen1_f1_capture_enable=enabled,
        )
        root = tmp_path / ("on" if enabled else "off")
        runner = SimulationRunner(
            env_cfg=EnvConfig(width=8, height=8, view_radius=1),
            pop_cfg=PopConfig(population_size=6),
            sim_cfg=cfg,
            logger=NoOpMetricsLogger(),
            run_dir=str(root),
            perf=False,
        )
        runner.run()
        agents = [
            (
                int(agent.agent_id),
                canonical_hash_from_genome_dict(agent.genome.to_dict()),
                float(agent.energy),
                float(agent.health),
            )
            for agent in sorted(runner.pop.alive_agents(), key=lambda value: int(value.agent_id))
        ]
        return (
            {
                "agents": agents,
                "grid": np.asarray(runner.env.grid).tolist(),
                "agent_grid": np.asarray(runner.env.agent_grid).tolist(),
                "rng": runner.rng_manager.capture_snapshot(generation_index=1),
            },
            root,
        )

    disabled, off_root = run(False)
    enabled, on_root = run(True)
    assert enabled == disabled
    assert not (off_root / "chronicle").exists()
    carrier_path = on_root / "chronicle" / "gen1_f1" / "gen1_f1_carrier_records.jsonl"
    assert carrier_path.is_file()
    assert len(carrier_path.read_text().splitlines()) == 6


def test_checkpoint_bundle_roundtrips_gen1_cursor_and_carriers(tmp_path: Path) -> None:
    def runner(root: Path) -> SimulationRunner:
        return SimulationRunner(
            env_cfg=EnvConfig(width=8, height=8, view_radius=1),
            pop_cfg=PopConfig(population_size=6),
            sim_cfg=SimConfig(
                seed=91,
                generations=1,
                steps_per_generation=2,
                device="cpu",
                bank_device="cpu",
                torch_threads=1,
                behavior_log_enable=False,
                policy_deterministic_actions=True,
                chronicle_enable=False,
                gen1_f1_capture_enable=True,
            ),
            logger=NoOpMetricsLogger(),
            run_dir=str(root),
            perf=False,
        )

    first = runner(tmp_path / "first")
    first.run()
    bundle = checkpoint_bundle_v2_save(
        first,
        str(tmp_path / "bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    state = payload["runner_state"]["gen1_f1_capture_state"]
    assert len(state["carrier_records"]) == 6

    restored = runner(tmp_path / "restored")
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)
    assert restored.gen1_f1_capture.export_state() == state
