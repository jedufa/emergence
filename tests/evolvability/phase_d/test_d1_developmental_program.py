from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agents.developmental_program_v1 import (
    apply_irreversible_stage_locks,
    canonicalize_developmental_program,
    lock_commitments_for_stage,
    resolve_module_active,
)
from agents.genome import Genome
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_base import RunnerBase
from simulation.runner_pkg.runner_config import SimConfig


def _make_sim(tmp_path: Path, *, seed: int, generations: int = 2, steps: int = 4, pop_size: int = 8) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=int(pop_size))
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )
    sim_cfg.policy_deterministic_actions = True
    return SimulationRunner(env_cfg=env_cfg, pop_cfg=pop_cfg, sim_cfg=sim_cfg, logger=NoOpMetricsLogger(), run_dir=str(tmp_path), perf=False)


def _dev_program() -> dict:
    return {
        "schema_version": "developmental_program_v1",
        "stages": [
            {"stage_id": "juvenile", "start_step": 0, "end_step": 5},
            {"stage_id": "adult", "start_step": 6, "end_step": -1},
        ],
        "module_windows": [
            {"module_id": "m010_behavior_bias", "stage_id": "juvenile", "active": False},
            {"module_id": "m010_behavior_bias", "stage_id": "adult", "active": True},
            {"module_id": "m030_identity_regulation", "stage_id": "juvenile", "active": True},
            {"module_id": "m030_identity_regulation", "stage_id": "adult", "active": False},
        ],
    }


def _install_program(sim: SimulationRunner) -> None:
    for a in sim.pop.agents.values():
        d = a.genome.to_dict()
        d["developmental_program"] = _dev_program()
        mods = list(d.get("modules", []))
        if not mods:
            mods = [
                {"module_id": "m000_policy_core", "module_type": "policy_core", "params": {"hidden_size": 64, "use_gru": False, "memory_dim": 0}},
                {"module_id": "m010_behavior_bias", "module_type": "behavior_bias", "params": {"aggression_bias": 0.1, "action_temperature": 1.0}},
                {"module_id": "m030_identity_regulation", "module_type": "identity_regulation", "params": {"drive_logit_gain": 1.0, "pred_lr": 0.08}},
            ]
        for m in mods:
            mid = str(m.get("module_id", ""))
            params = m.get("params", {}) if isinstance(m.get("params", {}), dict) else {}
            if mid == "m010_behavior_bias":
                params["aggression_bias"] = 0.95
                params["action_temperature"] = 0.55
            if mid == "m030_identity_regulation":
                params["drive_logit_gain"] = 1.9
                params["pred_lr"] = 0.01
            m["params"] = params
        d["modules"] = mods
        a.genome = Genome.from_dict(d)


def _trace_lines(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics" / "developmental_stage_trace.jsonl"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


def _lock_trace_lines(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics" / "developmental_lock_trace.jsonl"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


def _dev_runtime_state(sim: SimulationRunner) -> dict[int, tuple[str, str, int]]:
    out: dict[int, tuple[str, str, int]] = {}
    for aid, a in sim.pop.agents.items():
        st = getattr(a, "stats", None)
        if st is None:
            continue
        out[int(aid)] = (
            str(getattr(st, "_dev_module_structure_signature", "") or ""),
            str(getattr(st, "_dev_stage_id", "") or ""),
            RunnerBase._dev_previous_active_module_count(st),
        )
    return out


def _normalize_trace_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        r = dict(row)
        # Resume apply can legitimately report generation=-1 before first tick updates
        # generation mirrors; compare semantic trace payload fields.
        r.pop("generation", None)
        out.append(r)
    return out


def _canonical_trace_rows(rows: list[dict]) -> list[str]:
    return sorted(
        json.dumps(dict(r), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for r in rows
    )


def _d2_program() -> dict:
    return {
        "schema_version": "developmental_program_v1",
        "stages": [
            {"stage_id": "juvenile", "start_step": 0, "end_step": 1},
            {"stage_id": "adult", "start_step": 2, "end_step": -1},
        ],
        "module_windows": [
            {"module_id": "m010_behavior_bias", "stage_id": "juvenile", "active": True},
            {"module_id": "m010_behavior_bias", "stage_id": "adult", "active": True},
            {"module_id": "m030_identity_regulation", "stage_id": "juvenile", "active": False},
            {"module_id": "m030_identity_regulation", "stage_id": "adult", "active": False},
        ],
        "lock_commitments": [
            {"stage_id": "juvenile", "module_id": "m010_behavior_bias", "locked_active": False},
            {"stage_id": "adult", "module_id": "m030_identity_regulation", "locked_active": False},
        ],
        "lock_dependencies": [
            {
                "module_id": "m010_behavior_bias",
                "depends_on_module_id": "m030_identity_regulation",
                "requires_active": True,
            }
        ],
    }


def _d2_illegal_reverse_program() -> dict:
    return {
        "schema_version": "developmental_program_v1",
        "stages": [
            {"stage_id": "juvenile", "start_step": 0, "end_step": 1},
            {"stage_id": "adult", "start_step": 2, "end_step": -1},
        ],
        "module_windows": [
            {"module_id": "m010_behavior_bias", "stage_id": "juvenile", "active": True},
            {"module_id": "m010_behavior_bias", "stage_id": "adult", "active": True},
        ],
        "lock_commitments": [
            {"stage_id": "juvenile", "module_id": "m010_behavior_bias", "locked_active": False},
            {"stage_id": "adult", "module_id": "m010_behavior_bias", "locked_active": True},
        ],
        "lock_dependencies": [],
    }


def _run_phase_d_intent_pair(
    tmp_path: Path,
    *,
    seed: int = 9521,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    dict[int, str],
    dict[int, str],
    dict[int, dict[str, bool]],
    dict[int, dict[str, bool]],
]:
    run_fresh = tmp_path / "phase_d_intent_fresh"
    sim_fresh = _make_sim(run_fresh, seed=seed, generations=3, steps=4, pop_size=8)
    sim_fresh.sim_cfg.generations = 1
    sim_fresh.run(start_gen=0, skip_init=False)
    for a in sim_fresh.pop.agents.values():
        d = a.genome.to_dict()
        d["developmental_program"] = _d2_program()
        a.genome = Genome.from_dict(d)
    sim_fresh.sim_cfg.generations = 1
    sim_fresh.run(start_gen=1, skip_init=True)

    bundle = run_fresh / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(sim_fresh, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_fresh.sim_cfg.generations = 1
    sim_fresh.run(start_gen=2, skip_init=True)
    trace_fresh = _trace_lines(run_fresh)
    lock_trace_fresh = _lock_trace_lines(run_fresh)
    hashes_fresh = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_fresh.pop.agents.items()}
    lock_state_fresh = {
        int(aid): dict(getattr(getattr(a, "stats", None), "_dev_d2_lock_state_by_module_id", {}) or {})
        for aid, a in sim_fresh.pop.agents.items()
    }

    run_resume = tmp_path / "phase_d_intent_resume"
    sim_resume = _make_sim(run_resume, seed=seed, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_resume, payload, restore_rng=True)
    sim_resume.sim_cfg.generations = 1
    sim_resume.run(start_gen=2, skip_init=True)
    trace_resume = _trace_lines(run_resume)
    lock_trace_resume = _lock_trace_lines(run_resume)
    hashes_resume = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_resume.pop.agents.items()}
    lock_state_resume = {
        int(aid): dict(getattr(getattr(a, "stats", None), "_dev_d2_lock_state_by_module_id", {}) or {})
        for aid, a in sim_resume.pop.agents.items()
    }

    return (
        trace_fresh,
        trace_resume,
        lock_trace_fresh,
        lock_trace_resume,
        hashes_fresh,
        hashes_resume,
        lock_state_fresh,
        lock_state_resume,
    )


def _run_d2_replay_pair(tmp_path: Path, *, seed: int = 9401) -> tuple[list[dict], list[dict], list[dict], list[dict], dict[int, str], dict[int, str]]:
    run_a = tmp_path / "d2_run_a"
    sim_a = _make_sim(run_a, seed=seed, generations=2, steps=4, pop_size=8)
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=0, skip_init=False)
    for a in sim_a.pop.agents.values():
        d = a.genome.to_dict()
        d["developmental_program"] = _d2_program()
        a.genome = Genome.from_dict(d)
    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=1, skip_init=True)
    t_a = _trace_lines(run_a)
    lt_a = _lock_trace_lines(run_a)
    h_a = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_a.pop.agents.items()}

    run_b = tmp_path / "d2_run_b"
    sim_b = _make_sim(run_b, seed=seed, generations=2, steps=4, pop_size=8)
    sim_b.sim_cfg.generations = 1
    sim_b.run(start_gen=0, skip_init=False)
    for a in sim_b.pop.agents.values():
        d = a.genome.to_dict()
        d["developmental_program"] = _d2_program()
        a.genome = Genome.from_dict(d)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=1, skip_init=True)
    t_b = _trace_lines(run_b)
    lt_b = _lock_trace_lines(run_b)
    h_b = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_b.pop.agents.items()}
    return t_a, t_b, lt_a, lt_b, h_a, h_b


def _run_d1_replay_pair(tmp_path: Path, *, seed: int = 9101) -> tuple[list[dict], list[dict], dict[int, str], dict[int, str]]:
    run_a = tmp_path / "d1_run_a"
    sim_a = _make_sim(run_a, seed=seed, generations=2, steps=4, pop_size=8)
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=0, skip_init=False)
    _install_program(sim_a)
    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=1, skip_init=True)
    t_a = _trace_lines(run_a)
    h_a = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_a.pop.agents.items()}

    run_b = tmp_path / "d1_run_b"
    sim_b = _make_sim(run_b, seed=seed, generations=2, steps=4, pop_size=8)
    sim_b.sim_cfg.generations = 1
    sim_b.run(start_gen=0, skip_init=False)
    _install_program(sim_b)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=1, skip_init=True)
    t_b = _trace_lines(run_b)
    h_b = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_b.pop.agents.items()}
    return t_a, t_b, h_a, h_b


def test_d1_unit_stage_window_boundaries_pre_in_post() -> None:
    prog = canonicalize_developmental_program(
        {
            "stages": [{"stage_id": "s1", "start_step": 5, "end_step": 10}],
            "module_windows": [{"module_id": "m1", "stage_id": "s1", "active": True}],
        }
    )
    pre = resolve_module_active(prog, module_id="m1", age_steps=4)
    inside = resolve_module_active(prog, module_id="m1", age_steps=7)
    post = resolve_module_active(prog, module_id="m1", age_steps=11)
    assert pre[0] is False
    assert inside[0] is True
    assert post[0] is False


def test_d1_unit_midlife_structural_mutation_is_ignored_with_reason(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "structural_guard", seed=9001, generations=1, steps=1, pop_size=4)
    sim.pop.init_random()
    _install_program(sim)
    a = sim.pop.alive_agents()[0]
    e1 = sim.resolve_developmental_expression_for_agent(a)
    a.genome.modules.append({"module_id": "m999_new", "module_type": "generic", "params": {}, "io": {}, "latent_pos": [0.1, 0.2]})
    sim._dev_expr_cache.clear()
    e2 = sim.resolve_developmental_expression_for_agent(a)
    assert e2.get("structural_change_ignored", False) is True
    assert str(e2.get("structural_change_reason", "")) == "runtime_structure_change_ignored"
    assert dict(e2.get("effective", {})) == dict(e1.get("effective", {}))
    assert int(e2.get("module_count", -1)) == int(e1.get("module_count", -1))
    assert int(e2.get("active_module_count", -1)) == int(e1.get("active_module_count", -1))
    assert dict(e2.get("active_by_module_id", {})) == dict(e1.get("active_by_module_id", {}))
    assert dict(e2.get("inactive_reason_by_module_id", {})) == dict(e1.get("inactive_reason_by_module_id", {}))
    assert "m999_new" not in dict(e2.get("active_by_module_id", {}))


def test_d1_unit_genome_developmental_program_roundtrip_stable() -> None:
    g = Genome.random(random.Random(1234))
    d = g.to_dict()
    d["developmental_program"] = {
        "schema_version": "developmental_program_v1",
        "stages": [
            {"stage_id": "adult", "start_step": 3, "end_step": -1},
            {"stage_id": "juvenile", "start_step": 0, "end_step": 2},
        ],
        "module_windows": [
            {"module_id": "m030_identity_regulation", "stage_id": "adult", "active": False},
            {"module_id": "m010_behavior_bias", "stage_id": "juvenile", "active": False},
        ],
    }
    g2 = Genome.from_dict(d)
    g3 = Genome.from_dict(g2.to_dict())
    assert g2.to_dict()["developmental_program"] == g3.to_dict()["developmental_program"]
    assert g2.canonical_hash() == g3.canonical_hash()


def test_d1_integration_fixed_seed_replay_equivalence_stage_traces_and_outcomes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")
    t_a, t_b, h_a, h_b = _run_d1_replay_pair(tmp_path, seed=9101)

    assert _canonical_trace_rows(t_a) == _canonical_trace_rows(t_b)
    assert h_a == h_b
    assert len({str(x.get("stage_id", "")) for x in t_a}) >= 2
    assert len({int(x.get("active_module_count", 0)) for x in t_a}) >= 2


def test_d1_integration_fixed_seed_replay_equivalence_stress_20x(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")
    for i in range(20):
        t_a, t_b, h_a, h_b = _run_d1_replay_pair(tmp_path / f"stress_{i:02d}", seed=9101)
        assert _canonical_trace_rows(t_a) == _canonical_trace_rows(t_b)
        assert h_a == h_b


def test_d1_integration_backward_compat_without_developmental_program(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "d1_backcompat", seed=9201, generations=1, steps=3, pop_size=6)
    sim.run(start_gen=0, skip_init=False)
    for a in sim.pop.agents.values():
        dp = a.genome.to_dict().get("developmental_program", {})
        assert str(dp.get("schema_version", "")) == "developmental_program_v1"


def test_d1_integration_checkpoint_resume_preserves_dev_runtime_state_and_trace_parity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")

    run_a = tmp_path / "d1_resume_fresh"
    sim_a = _make_sim(run_a, seed=9301, generations=3, steps=4, pop_size=8)
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=0, skip_init=False)
    _install_program(sim_a)
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=1, skip_init=True)

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    pop_payload = payload.get("population", {}) if isinstance(payload, dict) else {}
    sample_stats = ((pop_payload.get("agents") or [{}])[0]).get("stats", {}) if isinstance(pop_payload, dict) else {}
    assert "_dev_module_structure_signature" in sample_stats
    assert "_dev_last_effective" in sample_stats
    assert "_dev_last_expr_snapshot" in sample_stats

    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=2, skip_init=True)
    trace_a = [x for x in _trace_lines(run_a) if int(x.get("generation", -1)) == 2]
    hashes_a = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_a.pop.agents.items()}
    state_a = _dev_runtime_state(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=888)

    run_b = tmp_path / "d1_resume_applied"
    sim_b = _make_sim(run_b, seed=9301, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 1
    sim_b.run(start_gen=2, skip_init=True)
    trace_b = _trace_lines(run_b)
    hashes_b = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_b.pop.agents.items()}
    state_b = _dev_runtime_state(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=888)

    assert _canonical_trace_rows(_normalize_trace_rows(trace_a)) == _canonical_trace_rows(_normalize_trace_rows(trace_b))
    assert hashes_a == hashes_b
    assert state_a == state_b
    assert rng_a == rng_b


def test_d1_regression_stage_gated_module_evaluation_order_is_deterministic() -> None:
    base = Genome.random(random.Random(99)).to_dict()
    base["developmental_program"] = _dev_program()
    base["modules"] = list(reversed(list(base.get("modules", []))))
    g1 = Genome.from_dict(base)
    p2 = dict(base)
    dp2 = dict(p2.get("developmental_program", {}))
    dp2["module_windows"] = list(reversed(list(dp2.get("module_windows", []))))
    p2["developmental_program"] = dp2
    g2 = Genome.from_dict(p2)
    e1 = g1.resolve_developmental_expression(age_steps=2)
    e2 = g2.resolve_developmental_expression(age_steps=2)
    assert e1["stage_id"] == e2["stage_id"]
    assert e1["active_by_module_id"] == e2["active_by_module_id"]
    assert e1["effective"] == e2["effective"]


def test_d2_unit_one_way_lock_enforcement_allows_lock_and_rejects_reverse(monkeypatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    program = canonicalize_developmental_program(
        {
            "stages": [
                {"stage_id": "juvenile", "start_step": 0, "end_step": 0},
                {"stage_id": "adult", "start_step": 1, "end_step": -1},
            ],
            "lock_commitments": [
                {"stage_id": "juvenile", "module_id": "m1", "locked_active": False},
                {"stage_id": "adult", "module_id": "m1", "locked_active": True},
            ],
        }
    )
    pre = apply_irreversible_stage_locks(
        program=program,
        stage_id="juvenile",
        prev_stage_id="default",
        active_by_module_id={"m1": True},
        inactive_reason_by_module_id={},
        previous_lock_state={},
        strict=True,
    )
    assert pre["lock_state_by_module_id"]["m1"] is False
    with pytest.raises(RuntimeError, match="DET_DEV_D2_ILLEGAL_REVERSE_LOCK"):
        apply_irreversible_stage_locks(
            program=program,
            stage_id="adult",
            prev_stage_id="juvenile",
            active_by_module_id={"m1": True},
            inactive_reason_by_module_id={},
            previous_lock_state=pre["lock_state_by_module_id"],
            strict=True,
        )


def test_d2_unit_dependency_conflict_resolution_is_deterministic_and_stable_order() -> None:
    program = canonicalize_developmental_program(
        {
            "lock_commitments": [
                {"stage_id": "adult", "module_id": "mB", "locked_active": True},
                {"stage_id": "adult", "module_id": "mA", "locked_active": True},
            ],
            "lock_dependencies": [
                {"module_id": "mA", "depends_on_module_id": "mZ", "requires_active": True},
                {"module_id": "mB", "depends_on_module_id": "mA", "requires_active": True},
            ],
        }
    )
    res_1 = apply_irreversible_stage_locks(
        program=program,
        stage_id="adult",
        prev_stage_id="juvenile",
        active_by_module_id={"mA": True, "mB": True, "mZ": False},
        inactive_reason_by_module_id={},
        previous_lock_state={},
        strict=False,
    )
    res_2 = apply_irreversible_stage_locks(
        program=program,
        stage_id="adult",
        prev_stage_id="juvenile",
        active_by_module_id={"mB": True, "mZ": False, "mA": True},
        inactive_reason_by_module_id={},
        previous_lock_state={},
        strict=False,
    )
    assert res_1["lock_state_by_module_id"] == res_2["lock_state_by_module_id"]
    assert res_1["lock_state_by_module_id"]["mA"] is False
    assert res_1["lock_state_by_module_id"]["mB"] is False
    assert res_1["active_by_module_id"]["mA"] is False
    assert res_1["active_by_module_id"]["mB"] is False


def test_d2_unit_schema_state_canonicalization_roundtrip_stable() -> None:
    g = Genome.random(random.Random(4321))
    d = g.to_dict()
    d["developmental_program"] = {
        "schema_version": "developmental_program_v1",
        "stages": [
            {"stage_id": "adult", "start_step": 3, "end_step": -1},
            {"stage_id": "juvenile", "start_step": 0, "end_step": 2},
        ],
        "module_windows": [
            {"module_id": "m030_identity_regulation", "stage_id": "adult", "active": False},
            {"module_id": "m010_behavior_bias", "stage_id": "juvenile", "active": False},
        ],
        "lock_commitments": [
            {"stage_id": "adult", "module_id": "m010_behavior_bias", "locked_active": True},
            {"stage_id": "juvenile", "module_id": "m010_behavior_bias", "locked_active": False},
        ],
        "lock_dependencies": [
            {"module_id": "m010_behavior_bias", "depends_on_module_id": "m030_identity_regulation", "requires_active": True}
        ],
    }
    g2 = Genome.from_dict(d)
    g3 = Genome.from_dict(g2.to_dict())
    dp2 = canonicalize_developmental_program(g2.to_dict()["developmental_program"])
    dp3 = canonicalize_developmental_program(g3.to_dict()["developmental_program"])
    assert dp2 == dp3
    assert g2.canonical_hash() == g3.canonical_hash()
    assert lock_commitments_for_stage(dp2, stage_id="adult")[0]["module_id"] == "m010_behavior_bias"


def test_d2_unit_strict_negative_illegal_irreversible_transition_raises(monkeypatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    program = canonicalize_developmental_program(
        {
            "lock_commitments": [
                {"stage_id": "juvenile", "module_id": "m1", "locked_active": True},
                {"stage_id": "adult", "module_id": "m1", "locked_active": False},
            ]
        }
    )
    pre = apply_irreversible_stage_locks(
        program=program,
        stage_id="juvenile",
        prev_stage_id="default",
        active_by_module_id={"m1": True},
        inactive_reason_by_module_id={},
        previous_lock_state={},
        strict=True,
    )
    with pytest.raises(RuntimeError, match="DET_DEV_D2_ILLEGAL_REVERSE_LOCK"):
        apply_irreversible_stage_locks(
            program=program,
            stage_id="adult",
            prev_stage_id="juvenile",
            active_by_module_id={"m1": False},
            inactive_reason_by_module_id={},
            previous_lock_state=pre["lock_state_by_module_id"],
            strict=True,
        )


def test_d2_integration_stage_transition_applies_irreversible_lock_at_boundary(tmp_path: Path) -> None:
    sim = _make_sim(tmp_path / "d2_stage_transition", seed=9407, generations=1, steps=1, pop_size=4)
    sim.pop.init_random()
    _install_program(sim)
    a = sim.pop.alive_agents()[0]
    d = a.genome.to_dict()
    d["developmental_program"] = _d2_program()
    a.genome = Genome.from_dict(d)
    st = getattr(a, "stats")
    st.survival_steps = 0
    e0 = sim.resolve_developmental_expression_for_agent(a)
    sim._dev_expr_cache.clear()
    st.survival_steps = 2
    e1 = sim.resolve_developmental_expression_for_agent(a)
    assert e0["lock_state_by_module_id"].get("m010_behavior_bias") is False
    assert e1["lock_state_by_module_id"].get("m010_behavior_bias") is False
    assert e1["active_by_module_id"].get("m010_behavior_bias") is False


def test_d2_integration_checkpoint_resume_parity_for_lock_runtime_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    run_a = tmp_path / "d2_resume_fresh"
    sim_a = _make_sim(run_a, seed=9411, generations=3, steps=4, pop_size=8)
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=0, skip_init=False)
    for a in sim_a.pop.agents.values():
        d = a.genome.to_dict()
        d["developmental_program"] = _d2_program()
        a.genome = Genome.from_dict(d)
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=1, skip_init=True)
    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000002"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=2, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)
    pop_payload = payload.get("population", {}) if isinstance(payload, dict) else {}
    sample_stats = ((pop_payload.get("agents") or [{}])[0]).get("stats", {}) if isinstance(pop_payload, dict) else {}
    assert "_dev_d2_lock_state_by_module_id" in sample_stats
    assert "_dev_d2_lock_state_sha256" in sample_stats
    sim_a.sim_cfg.generations = 1
    sim_a.run(start_gen=2, skip_init=True)
    hashes_a = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_a.pop.agents.items()}
    lock_state_a = {
        int(aid): dict(getattr(getattr(a, "stats", None), "_dev_d2_lock_state_by_module_id", {}) or {})
        for aid, a in sim_a.pop.agents.items()
    }

    run_b = tmp_path / "d2_resume_applied"
    sim_b = _make_sim(run_b, seed=9411, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 1
    sim_b.run(start_gen=2, skip_init=True)
    hashes_b = {int(aid): str(a.genome.canonical_hash()) for aid, a in sim_b.pop.agents.items()}
    lock_state_b = {
        int(aid): dict(getattr(getattr(a, "stats", None), "_dev_d2_lock_state_by_module_id", {}) or {})
        for aid, a in sim_b.pop.agents.items()
    }
    assert hashes_a == hashes_b
    assert lock_state_a == lock_state_b


def test_d2_integration_fixed_seed_replay_equivalence_stage_lock_traces_and_outcomes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")
    t_a, t_b, lt_a, lt_b, h_a, h_b = _run_d2_replay_pair(tmp_path, seed=9421)
    assert _canonical_trace_rows(t_a) == _canonical_trace_rows(t_b)
    assert _canonical_trace_rows(lt_a) == _canonical_trace_rows(lt_b)
    assert h_a == h_b
    assert len(lt_a) > 0


def test_d2_integration_fixed_seed_replay_equivalence_stress_20x(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")
    for i in range(20):
        t_a, t_b, lt_a, lt_b, h_a, h_b = _run_d2_replay_pair(tmp_path / f"d2_stress_{i:02d}", seed=9421)
        assert _canonical_trace_rows(t_a) == _canonical_trace_rows(t_b)
        assert _canonical_trace_rows(lt_a) == _canonical_trace_rows(lt_b)
        assert h_a == h_b


def test_phase_d_e2e_intent_validation_fresh_vs_resume_with_irreversible_lockin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")
    (
        trace_fresh,
        trace_resume,
        lock_trace_fresh,
        lock_trace_resume,
        hashes_fresh,
        hashes_resume,
        lock_state_fresh,
        lock_state_resume,
    ) = _run_phase_d_intent_pair(tmp_path, seed=9521)

    trace_fresh_resumed_segment = [x for x in trace_fresh if int(x.get("generation", -1)) == 2]
    lock_trace_fresh_resumed_segment = [x for x in lock_trace_fresh if int(x.get("generation", -1)) == 2]
    assert _canonical_trace_rows(_normalize_trace_rows(trace_fresh_resumed_segment)) == _canonical_trace_rows(_normalize_trace_rows(trace_resume))
    assert _canonical_trace_rows(_normalize_trace_rows(lock_trace_fresh_resumed_segment)) == _canonical_trace_rows(_normalize_trace_rows(lock_trace_resume))
    assert hashes_fresh == hashes_resume
    assert lock_state_fresh == lock_state_resume

    stage_ids = {str(r.get("stage_id", "")) for r in trace_fresh}
    assert "juvenile" in stage_ids or "default" in stage_ids
    assert len(lock_trace_fresh) > 0
    all_events = [str(ev.get("event", "")) for row in lock_trace_fresh for ev in list(row.get("events", []) or [])]
    assert "lock_committed" in all_events
    assert "illegal_reverse_lock_ignored" not in all_events
    for _aid, state in lock_state_fresh.items():
        if "m010_behavior_bias" in state:
            assert bool(state.get("m010_behavior_bias")) is False

    stage_sim = _make_sim(tmp_path / "phase_d_stage_transition_probe", seed=9525, generations=1, steps=1, pop_size=4)
    stage_sim.pop.init_random()
    stage_agent = stage_sim.pop.alive_agents()[0]
    d_stage = stage_agent.genome.to_dict()
    d_stage["developmental_program"] = _d2_program()
    stage_agent.genome = Genome.from_dict(d_stage)
    st_stage = getattr(stage_agent, "stats")
    st_stage.survival_steps = 0
    e0 = stage_sim.resolve_developmental_expression_for_agent(stage_agent)
    stage_sim._dev_expr_cache.clear()
    st_stage.survival_steps = 2
    e1 = stage_sim.resolve_developmental_expression_for_agent(stage_agent)
    assert str(e0.get("stage_id", "")) == "juvenile"
    assert str(e1.get("stage_id", "")) == "adult"

    illegal_sim = _make_sim(tmp_path / "phase_d_illegal_reverse", seed=9527, generations=1, steps=1, pop_size=4)
    illegal_sim.pop.init_random()
    agent = illegal_sim.pop.alive_agents()[0]
    d = agent.genome.to_dict()
    d["developmental_program"] = _d2_illegal_reverse_program()
    agent.genome = Genome.from_dict(d)
    st = getattr(agent, "stats")
    st.survival_steps = 0
    illegal_sim.resolve_developmental_expression_for_agent(agent)
    illegal_sim._dev_expr_cache.clear()
    st.survival_steps = 3
    with pytest.raises(RuntimeError, match="DET_DEV_D2_ILLEGAL_REVERSE_LOCK"):
        illegal_sim.resolve_developmental_expression_for_agent(agent)


def test_phase_d_e2e_intent_validation_stress_20x(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")
    for i in range(20):
        (
            trace_fresh,
            trace_resume,
            lock_trace_fresh,
            lock_trace_resume,
            hashes_fresh,
            hashes_resume,
            lock_state_fresh,
            lock_state_resume,
        ) = _run_phase_d_intent_pair(tmp_path / f"phase_d_e2e_stress_{i:02d}", seed=9521)
        trace_fresh_resumed_segment = [x for x in trace_fresh if int(x.get("generation", -1)) == 2]
        lock_trace_fresh_resumed_segment = [x for x in lock_trace_fresh if int(x.get("generation", -1)) == 2]
        assert _canonical_trace_rows(_normalize_trace_rows(trace_fresh_resumed_segment)) == _canonical_trace_rows(_normalize_trace_rows(trace_resume))
        assert _canonical_trace_rows(_normalize_trace_rows(lock_trace_fresh_resumed_segment)) == _canonical_trace_rows(_normalize_trace_rows(lock_trace_resume))
        assert hashes_fresh == hashes_resume
        assert lock_state_fresh == lock_state_resume
