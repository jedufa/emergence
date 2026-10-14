from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.habacclim1 import ARM_NAMES, HABACCLIM1R_SELECTED_POLICY_HASHES, policy_hashes
from simulation.metrics import MetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=10, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = True
    cfg.patch_world_observation_enable = True
    cfg.patch_food_regrow_per_step = 0.02
    cfg.habacclim_enable = True
    cfg.habacclim_observation_enable = True
    cfg.habacclim_control_mode = "acclimation_tradeoff"
    return cfg


def _sim_cfg(seed: int, arm: str) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=2,
        steps_per_generation=500,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.habacclim1_probe_enable = True
    cfg.habacclim1_probe_schema_version = 1
    cfg.habacclim1_probe_arm = str(arm)
    cfg.habacclim1_probe_controller_version = "habacclim1_mlp_v1"
    cfg.habacclim1_telemetry_enable = True
    cfg.habacclim1_telemetry_schema_version = 1
    return cfg


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty csv: {path}")
    return rows[-1]


def _num(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def run_validation(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    for idx, arm in enumerate(ARM_NAMES):
        run_dir = out_dir / f"runtime_{arm}"
        sim = SimulationRunner(
            env_cfg=_env_cfg(),
            pop_cfg=PopConfig(population_size=16),
            sim_cfg=_sim_cfg(7100 + idx, arm),
            logger=MetricsLogger(str(run_dir)),
            run_dir=str(run_dir),
            perf=False,
        )
        sim.run(start_gen=0, skip_init=False)
        summary_path = run_dir / "metrics" / "habacclim1_generation_summary.csv"
        contract_path = run_dir / "metrics" / "habacclim1_probe_contract.json"
        row = _last_csv_row(summary_path)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        rows[arm] = {
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "contract_path": str(contract_path),
            "last_summary": row,
            "policy_hashes": contract.get("policy_hashes", {}),
            "obs_size": int(contract.get("obs_size", 0)),
            "scalar_index": int(contract.get("habitat_acclimation_state_norm_index", -1)),
            "gather_attempts": _num(row, "gather_attempts"),
            "gather_successes": _num(row, "gather_successes"),
            "effective_food_gain": _num(row, "effective_food_gain"),
            "transitions": _num(row, "transitions"),
            "habitat_a_ticks": _num(row, "habitat_a_ticks"),
            "habitat_b_ticks": _num(row, "habitat_b_ticks"),
            "cumulative_mismatch": _num(row, "cumulative_mismatch"),
        }
    hashes = policy_hashes()
    hash_ok = hashes == HABACCLIM1R_SELECTED_POLICY_HASHES and all(
        rows[arm]["policy_hashes"] == HABACCLIM1R_SELECTED_POLICY_HASHES for arm in ARM_NAMES
    )
    construct_ok = (
        hash_ok
        and rows["habitat_a_committed"]["habitat_a_ticks"] > rows["habitat_a_committed"]["habitat_b_ticks"]
        and rows["habitat_b_committed"]["habitat_b_ticks"] > rows["habitat_b_committed"]["habitat_a_ticks"]
        and rows["balanced_switcher"]["transitions"] > rows["habitat_a_committed"]["transitions"]
        and rows["balanced_switcher"]["transitions"] > rows["habitat_b_committed"]["transitions"]
        and rows["balanced_switcher"]["gather_attempts"] > 0
        and rows["balanced_switcher"]["gather_successes"] > 0
    )
    result = {
        "schema_version": "habacclim1r_runtime_probe_validation_v1",
        "ok": bool(construct_ok),
        "hash_ok": bool(hash_ok),
        "policy_hashes": hashes,
        "arms": rows,
        "decision_if_failed": None if construct_ok else "HABACCLIM-1R-E_AUTHORITATIVE_RUNTIME_CONSTRUCT_FAILURE",
    }
    (out_dir / "habacclim1r_runtime_probe_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = run_validation(Path(args.out_dir))
    print(json.dumps({"ok": result["ok"], "hash_ok": result["hash_ok"], "decision_if_failed": result["decision_if_failed"]}, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
