from __future__ import annotations

import json
from pathlib import Path

from tests.evolvability.phase_e.test_e1_metrics_engine import _make_sim, _read_jsonl


def test_e1_metric_and_sentinel_schema_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "schema_lock"
    sim = _make_sim(run_dir, seed=111, generations=2)
    sim.sim_cfg.phase_e_e1_enable = True
    sim.sim_cfg.phase_e_e1_cadence_gen = 1
    sim.sim_cfg.phase_e_e1_knockout_every_gen = 1
    sim.run(start_gen=0, skip_init=False)

    latest = json.loads((run_dir / "phase_e_e1_latest.json").read_text(encoding="utf-8"))
    metrics_rows = _read_jsonl(run_dir / "metrics" / "phase_e_e1_metrics.jsonl")
    sentinel_rows = _read_jsonl(run_dir / "metrics" / "phase_e_e1_sentinel.jsonl")
    assert metrics_rows and sentinel_rows

    required_latest_keys = {
        "schema_version",
        "generation",
        "cadence_gen",
        "retention_k_gen",
        "population_alive",
        "unique_structures",
        "introduced_this_gen",
        "retained_ge_k",
        "sir",
        "neutrality_mean",
        "neutrality_trend",
        "functional_density",
        "redundancy_index",
        "recovery_taxonomy",
        "knockout_cadence_gen",
        "determinism_sentinel",
    }
    assert required_latest_keys.issubset(set(latest.keys()))

    required_sentinel_keys = {
        "schema_version",
        "generation",
        "rng_indices_before",
        "rng_indices_after",
        "rng_indices_delta",
        "population_hash_before",
        "population_hash_after",
        "population_hash_equal",
    }
    assert required_sentinel_keys.issubset(set(sentinel_rows[-1].keys()))

