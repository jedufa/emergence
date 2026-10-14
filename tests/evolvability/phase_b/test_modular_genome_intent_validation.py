from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from pathlib import Path

import pytest

from agents.genome import Genome
from agents.modular_genome_v1 import B3_FROZEN_LOCALITY, MODULAR_GENOME_SCHEMA_VERSION, canonical_payload_from_genome_dict
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _artifact_dir(tmp_path: Path, exp_id: str) -> Path:
    override = os.environ.get("EIEL_PHASEB_VALIDATION_ARTIFACT_ROOT", "").strip()
    base = Path(override) if override else (tmp_path / "artifacts")
    out = base / exp_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def _line_modules(n: int = 8) -> list[dict]:
    mods = []
    for i in range(n):
        x = float(i) / float(max(1, n - 1))
        mods.append(
            {
                "module_id": f"m{i}",
                "module_type": "generic",
                "latent_pos": [x, 0.0],
                "io": {
                    "inputs": [{"slot_id": "in", "value_type": "scalar"}],
                    "outputs": [{"slot_id": "out", "value_type": "scalar"}],
                    "caps": {"max_fanin": 16, "max_fanout": 16},
                },
            }
        )
    return mods


def _dense_connections(n: int = 8) -> list[dict]:
    out = []
    cidx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out.append(
                {
                    "connection_id": f"c{cidx:04d}",
                    "src_module_id": f"m{i}",
                    "dst_module_id": f"m{j}",
                    "src_slot": "out",
                    "dst_slot": "in",
                    "value_type": "scalar",
                    "weight": 1.0,
                }
            )
            cidx += 1
    return out


def _phaseb_payload() -> dict:
    return {
        "schema_version": MODULAR_GENOME_SCHEMA_VERSION,
        "hidden_size": 64,
        "exploration_bias": 0.1,
        "aggression_bias": 0.1,
        "reproduction_threshold": 30.0,
        "mutation_rate": 0.05,
        "action_temperature": 1.0,
        "use_gru": False,
        "memory_dim": 0,
        "drive_inertia": 0.8,
        "drive_noise": 0.01,
        "drive_logit_gain": 1.0,
        "pred_lr": 0.08,
        "b2_invalid_policy": "clip",
        "modules": _line_modules(6),
        "connections": _dense_connections(6),
    }


def _distance(src: list[float], dst: list[float]) -> float:
    raw = math.sqrt((float(src[0]) - float(dst[0])) ** 2 + (float(src[1]) - float(dst[1])) ** 2)
    return max(0.0, min(1.0, raw / math.sqrt(2.0)))


def _locality_metrics(payload: dict) -> dict[str, float]:
    modules = {str(m["module_id"]): list(m["latent_pos"]) for m in payload["modules"]}
    raw_dist = [
        _distance(modules[c["src_module_id"]], modules[c["dst_module_id"]])
        for c in payload.get("connections", [])
    ]
    can = canonical_payload_from_genome_dict(payload)
    kept = can.get("connections", [])
    kept_dist = [float(c.get("locality_distance", 0.0) or 0.0) for c in kept]
    short_thr = 0.25
    raw_short = float(sum(1 for d in raw_dist if d <= short_thr)) / float(max(1, len(raw_dist)))
    kept_short = float(sum(1 for d in kept_dist if d <= short_thr)) / float(max(1, len(kept_dist)))
    return {
        "raw_edge_count": float(len(raw_dist)),
        "kept_edge_count": float(len(kept_dist)),
        "raw_mean_distance": float(sum(raw_dist) / max(1, len(raw_dist))),
        "kept_mean_distance": float(sum(kept_dist) / max(1, len(kept_dist))),
        "raw_short_ratio": raw_short,
        "kept_short_ratio": kept_short,
        "dropped_nonlocal": float(can.get("b3_locality_dropped_connections", 0)),
    }


def _make_sim(tmp_path: Path, *, seed: int, generations: int, steps: int, pop_size: int = 10) -> SimulationRunner:
    env_cfg = EnvConfig(width=10, height=10)
    pop_cfg = PopConfig(population_size=pop_size)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=int(generations),
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
    )
    sim_cfg.policy_deterministic_actions = True
    sim_cfg.checkpoint_bundle_v2_period_gen = 1
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _genome_hashes(sim: SimulationRunner) -> dict[int, str]:
    out: dict[int, str] = {}
    for aid, a in sim.pop.agents.items():
        g = getattr(a, "genome", None)
        if g is None:
            continue
        out[int(aid)] = str(g.canonical_hash())
    return out


def _mean_locality_metric(sim: SimulationRunner, field_name: str) -> float:
    vals = []
    for a in sim.pop.agents.values():
        g = getattr(a, "genome", None)
        if g is None:
            continue
        vals.append(float(getattr(g, field_name, 0.0) or 0.0))
    return float(sum(vals) / max(1, len(vals)))


def test_phaseb_e1_canonical_equivalence_permuted_ordering(tmp_path: Path) -> None:
    base = _phaseb_payload()
    p1 = deepcopy(base)
    p2 = deepcopy(base)
    p1["modules"] = list(reversed(p1["modules"]))
    p2["connections"] = list(reversed(p2["connections"]))
    g0 = Genome.from_dict(base)
    g1 = Genome.from_dict(p1)
    g2 = Genome.from_dict(p2)
    assert g0.canonical_bytes() == g1.canonical_bytes() == g2.canonical_bytes()
    assert g0.canonical_hash() == g1.canonical_hash() == g2.canonical_hash()

    art = {
        "experiment": "E1",
        "hash": str(g0.canonical_hash()),
        "canonical_bytes_len": len(g0.canonical_bytes()),
        "variants": 3,
    }
    _write_json(_artifact_dir(tmp_path, "b_ic_e1") / "canonical_equivalence.json", art)


def test_phaseb_e2_legacy_migration_idempotent_rng_neutral(tmp_path: Path) -> None:
    legacy = {
        "hidden_size": 64,
        "exploration_bias": -0.12,
        "aggression_bias": 0.34,
        "reproduction_threshold": 33.0,
        "mutation_rate": 0.04,
        "action_temperature": 1.1,
        "use_gru": True,
        "memory_dim": 16,
        "drive_inertia": 0.88,
        "drive_noise": 0.02,
        "drive_logit_gain": 1.2,
        "pred_lr": 0.09,
    }
    rm = RNGManager(global_seed=1701)
    pre = rm.capture_snapshot(generation_index=0)
    hashes: list[str] = []
    canonical_payloads: list[str] = []
    for _ in range(32):
        g = Genome.from_dict(legacy)
        hashes.append(str(g.canonical_hash()))
        canonical_payloads.append(json.dumps(g.to_dict(), sort_keys=True, separators=(",", ":")))
    post = rm.capture_snapshot(generation_index=0)

    assert len(set(hashes)) == 1
    assert len(set(canonical_payloads)) == 1
    assert pre == post

    _write_json(
        _artifact_dir(tmp_path, "b_ic_e2") / "legacy_migration_determinism.json",
        {
            "experiment": "E2",
            "runs": len(hashes),
            "unique_hashes": len(set(hashes)),
            "rng_snapshot_equal": bool(pre == post),
        },
    )


def test_phaseb_e3_constraint_enforcement_deterministic_repair_or_reject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _phaseb_payload()
    payload["connections"].extend(
        [
            {
                "connection_id": "bad_type",
                "src_module_id": "m0",
                "dst_module_id": "m1",
                "src_slot": "out",
                "dst_slot": "in",
                "value_type": "vector",
                "weight": 1.0,
            },
            {
                "connection_id": "bad_missing",
                "src_module_id": "missing",
                "dst_module_id": "m1",
                "src_slot": "out",
                "dst_slot": "in",
                "value_type": "scalar",
                "weight": 1.0,
            },
        ]
    )

    can_runs = [Genome.from_dict(deepcopy(payload)).canonical_hash() for _ in range(10)]
    assert len(set(can_runs)) == 1

    strict_payload = deepcopy(payload)
    strict_payload["b2_invalid_policy"] = "reject"
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")
    with pytest.raises(RuntimeError, match="DET_GENOME_B2_(INVALID_CONNECTION|TYPE_MISMATCH)"):
        Genome.from_dict(strict_payload)

    _write_json(
        _artifact_dir(tmp_path, "b_ic_e3") / "constraint_determinism.json",
        {
            "experiment": "E3",
            "repair_unique_hashes": len(set(str(x) for x in can_runs)),
            "strict_reject_codes": ["DET_GENOME_B2_INVALID_CONNECTION", "DET_GENOME_B2_TYPE_MISMATCH"],
        },
    )


def test_phaseb_e4_locality_bias_metrics_shift_to_short_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(B3_FROZEN_LOCALITY, "max_connection_distance", 0.28)
    payload = _phaseb_payload()
    m = _locality_metrics(payload)

    assert m["kept_edge_count"] > 0.0
    assert m["kept_mean_distance"] < m["raw_mean_distance"]
    assert m["kept_short_ratio"] > m["raw_short_ratio"]
    assert m["dropped_nonlocal"] > 0.0

    _write_json(_artifact_dir(tmp_path, "b_ic_e4") / "locality_bias_summary.json", {"experiment": "E4", **m})


def test_phaseb_e5_locality_ablation_contrast_on_vs_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _phaseb_payload()

    monkeypatch.setitem(B3_FROZEN_LOCALITY, "max_connection_distance", 0.28)
    on = _locality_metrics(payload)

    monkeypatch.setitem(B3_FROZEN_LOCALITY, "max_connection_distance", 1.0)
    off = _locality_metrics(payload)

    delta_mean = float(off["kept_mean_distance"] - on["kept_mean_distance"])
    delta_short = float(on["kept_short_ratio"] - off["kept_short_ratio"])

    assert delta_mean > 0.05
    assert delta_short > 0.10

    _write_json(
        _artifact_dir(tmp_path, "b_ic_e5") / "locality_ablation_compare.json",
        {
            "experiment": "E5",
            "on": on,
            "off": off,
            "delta_mean_distance_off_minus_on": delta_mean,
            "delta_short_ratio_on_minus_off": delta_short,
        },
    )


def test_phaseb_e6_fresh_vs_resume_structural_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EIEL_STRICT_DETERMINISM", "1")

    run_a = tmp_path / "b_ic_e6_a"
    sim_a = _make_sim(run_a, seed=9306, generations=3, steps=4, pop_size=8)
    sim_a.run(start_gen=0, skip_init=False)

    for a in sim_a.pop.agents.values():
        a.genome = Genome.from_dict(_phaseb_payload())

    bundle = run_a / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim_a, str(bundle), generation_index=3, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(bundle, strict=True)

    sim_a.sim_cfg.generations = 2
    sim_a.run(start_gen=3, skip_init=True)
    hash_a = _genome_hashes(sim_a)
    rng_a = sim_a.rng_manager.capture_snapshot(generation_index=999)
    mean_d_a = _mean_locality_metric(sim_a, "b3_locality_mean_distance")

    run_b = tmp_path / "b_ic_e6_b"
    sim_b = _make_sim(run_b, seed=9306, generations=3, steps=4, pop_size=8)
    checkpoint_bundle_v2_apply(sim_b, payload, restore_rng=True)
    sim_b.sim_cfg.generations = 2
    sim_b.run(start_gen=3, skip_init=True)
    hash_b = _genome_hashes(sim_b)
    rng_b = sim_b.rng_manager.capture_snapshot(generation_index=999)
    mean_d_b = _mean_locality_metric(sim_b, "b3_locality_mean_distance")

    assert hash_a == hash_b
    assert rng_a == rng_b
    assert mean_d_a == mean_d_b

    _write_json(
        _artifact_dir(tmp_path, "b_ic_e6") / "fresh_resume_parity.json",
        {
            "experiment": "E6",
            "agents": len(hash_a),
            "hashes_equal": bool(hash_a == hash_b),
            "rng_equal": bool(rng_a == rng_b),
            "mean_locality_distance": mean_d_a,
        },
    )
