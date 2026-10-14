from __future__ import annotations

import csv
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from env.env_config import EnvConfig
from env.gridworld_pkg.patch_world import (
    RESPEC0B_RESOURCE_A,
    RESPEC0B_RESOURCE_B,
    initialize_patch_world_arrays,
    respec_n2_capacity_balance_contract,
    respec_n2_capacity_budget_snapshot,
)
from simulation.respec_n1_zone_telemetry import finish_generation
from simulation.respec_n1_zone_telemetry import record_tick


def _load_n2():
    path = Path(__file__).resolve().parents[3] / "tools" / "bin" / "respec_n2_capacity_balanced_niches.py"
    spec = importlib.util.spec_from_file_location("respec_n2_capacity_balanced_niches", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("respec_n2_capacity_balanced_niches", module)
    spec.loader.exec_module(module)
    return module


n2 = _load_n2()


def _cfg() -> EnvConfig:
    cfg = EnvConfig(width=4, height=4)
    cfg.patch_world_enable = True
    cfg.respec0b_resource_specialization_enable = True
    cfg.respec0b_resource_class_mode = "vertical_split_ab"
    cfg.respec0b_resource_equalized_sham_enable = False
    return cfg


def test_respec_n2_capacity_balanced_overlay_is_default_off() -> None:
    cfg = _cfg()
    grid = np.ones((4, 4), dtype=np.int8)

    default_arrays = initialize_patch_world_arrays(cfg, grid)
    cfg.respec_n2_capacity_balanced_mixed_enable = True
    balanced_arrays = initialize_patch_world_arrays(cfg, grid)

    default_fertility = np.asarray(default_arrays["patch_fertility"], dtype=np.float32)
    balanced_fertility = np.asarray(balanced_arrays["patch_fertility"], dtype=np.float32)
    classes = np.asarray(balanced_arrays["respec0b_resource_class"], dtype=np.int32)

    assert np.all(default_fertility == np.float32(1.0))
    assert np.all(balanced_fertility[classes == RESPEC0B_RESOURCE_A] == np.float32(1.0))
    assert np.all(balanced_fertility[classes == RESPEC0B_RESOURCE_B] == np.float32(0.5))


def test_respec_n2_capacity_balanced_overlay_does_not_apply_to_controls() -> None:
    grid = np.ones((4, 4), dtype=np.int8)

    sham_cfg = _cfg()
    sham_cfg.respec0b_resource_equalized_sham_enable = True
    sham_cfg.respec_n2_capacity_balanced_mixed_enable = True
    sham_arrays = initialize_patch_world_arrays(sham_cfg, grid)

    all_a_cfg = _cfg()
    all_a_cfg.respec0b_resource_class_mode = "all_a"
    all_a_cfg.respec_n2_capacity_balanced_mixed_enable = True
    all_a_arrays = initialize_patch_world_arrays(all_a_cfg, grid)

    assert np.all(np.asarray(sham_arrays["patch_fertility"], dtype=np.float32) == np.float32(1.0))
    assert np.all(np.asarray(all_a_arrays["patch_fertility"], dtype=np.float32) == np.float32(1.0))


def test_respec_n2_capacity_balance_contract_equalizes_energy_budget() -> None:
    cfg = _cfg()
    cfg.respec_n2_capacity_balanced_mixed_enable = True
    contract = respec_n2_capacity_balance_contract(cfg)

    assert contract["formula_version"] == "energy_return_regrowth_balance_v1"
    assert contract["resource_a_return"] == 0.75
    assert contract["resource_b_return"] == 1.5
    assert contract["resource_a_fertility"] == 1.0
    assert contract["resource_b_fertility"] == 0.5
    assert contract["resource_a_energy_regrowth_budget_per_cell"] == contract[
        "resource_b_energy_regrowth_budget_per_cell"
    ]


def test_respec_n2_capacity_budget_snapshot_reports_equal_energy_opportunity() -> None:
    cfg = _cfg()
    cfg.respec_n2_capacity_balanced_mixed_enable = True
    grid = np.ones((4, 4), dtype=np.int8)
    arrays = initialize_patch_world_arrays(cfg, grid)
    classes = np.asarray(arrays["respec0b_resource_class"], dtype=np.int32)
    fertility = np.asarray(arrays["patch_fertility"], dtype=np.float32)
    before = np.zeros((4, 4), dtype=np.float32)
    cap = np.ones((4, 4), dtype=np.float32)
    effective_delta = np.where(classes == RESPEC0B_RESOURCE_A, 0.02, 0.01).astype(np.float32)
    after = before + effective_delta

    snapshot = respec_n2_capacity_budget_snapshot(
        cfg,
        classes=classes,
        fertility=fertility,
        before=before,
        after=after,
        cap=cap,
        base_delta=effective_delta,
        effective_delta=effective_delta,
    )
    rows = {row["zone"]: row for row in snapshot["zones"]}

    assert snapshot["capacity_balanced_mixed_enabled"] is True
    assert rows["resource_a_zone"]["cell_count"] == 8
    assert rows["resource_b_zone"]["cell_count"] == 8
    assert rows["resource_a_zone"]["energy_opportunity_added"] == rows["resource_b_zone"][
        "energy_opportunity_added"
    ]


def test_respec_n2_capacity_budget_telemetry_writes_compact_csv(tmp_path) -> None:
    env = SimpleNamespace(cfg=_cfg())
    env.cfg.respec_n1_zone_telemetry_enable = True
    env.respec0b_resource_class = np.asarray(
        [
            [RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B, RESPEC0B_RESOURCE_B],
            [RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B, RESPEC0B_RESOURCE_B],
        ],
        dtype=np.float32,
    )
    env.patch_food_amount = np.ones((2, 4), dtype=np.float32)
    env.agent_pos_y = np.zeros((0,), dtype=np.int32)
    env.agent_pos_x = np.zeros((0,), dtype=np.int32)
    env.respec_n2_capacity_budget_last = {
        "schema_version": "respec_n2_capacity_budget_snapshot_v1",
        "capacity_balanced_mixed_enabled": True,
        "contract": {"formula_version": "energy_return_regrowth_balance_v1"},
        "zones": [
            {
                "zone": "resource_a_zone",
                "resource_class": RESPEC0B_RESOURCE_A,
                "cell_count": 4,
                "fertility_mean": 1.0,
                "amount_before": 0.0,
                "amount_after": 0.08,
                "cap_total": 4.0,
                "available_cells_before": 0,
                "available_cells_after": 0,
                "available_amount_before": 0.0,
                "available_amount_after": 0.0,
                "base_regrowth_added": 0.08,
                "effective_regrowth_added": 0.08,
                "energy_opportunity_added": 0.06,
                "return_per_completion": 0.75,
                "stockout_cells_after": 4,
            }
        ],
    }
    runner = SimpleNamespace(env=env, env_cfg=env.cfg, run_dir=str(tmp_path))

    finish_generation(runner, generation=0, alive_after=[], selected_parent_agents=[], birth_contexts=[])

    budget_path = tmp_path / "metrics" / "respec_n2_capacity_budget_generation_summary.csv"
    payload = json.loads((tmp_path / "metrics" / "respec_n1_zone_run_summary.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(budget_path.open(newline="", encoding="utf-8")))

    assert budget_path.exists()
    assert rows[0]["formula_version"] == "energy_return_regrowth_balance_v1"
    assert rows[0]["zone"] == "resource_a_zone"
    assert payload["capacity_budget_history_rows"] == 1


def test_respec_n1_zone_telemetry_reports_parent_pool_denominator(tmp_path) -> None:
    cfg = _cfg()
    cfg.respec_n1_zone_telemetry_enable = True
    env = SimpleNamespace(cfg=cfg)
    env.respec0b_resource_class = np.asarray([[RESPEC0B_RESOURCE_A, RESPEC0B_RESOURCE_B]], dtype=np.int32)
    env.patch_food_amount = np.ones((1, 2), dtype=np.float32)
    env.agent_pos_y = np.asarray([0, 0], dtype=np.int32)
    env.agent_pos_x = np.asarray([0, 1], dtype=np.int32)

    a_agent = SimpleNamespace(
        agent_id=0,
        genome=SimpleNamespace(resource_handling_axis=0.75),
        stats=SimpleNamespace(survival_steps=0, lineage_id="L0"),
    )
    neutral_agent = SimpleNamespace(
        agent_id=1,
        genome=SimpleNamespace(resource_handling_axis=0.0),
        stats=SimpleNamespace(survival_steps=120, lineage_id="L1"),
    )
    runner = SimpleNamespace(env=env, env_cfg=cfg, run_dir=str(tmp_path))

    record_tick(runner, generation=0, tick=0, alive=[a_agent, neutral_agent], died_mask=np.asarray([False, False]))
    finish_generation(
        runner,
        generation=0,
        alive_after=[a_agent, neutral_agent],
        eligible_parent_agents=[a_agent, neutral_agent],
        selected_parent_agents=[a_agent],
        birth_contexts=[],
    )

    payload = json.loads((tmp_path / "metrics" / "respec_n1_zone_run_summary.json").read_text(encoding="utf-8"))
    rows = {(row["arm"], row["zone"]): row for row in payload["arm_zone"]}

    a_row = rows[("a_favored", "resource_a_zone")]
    neutral_row = rows[("neutral", "resource_b_zone")]
    assert a_row["eligible_parent_events"] == 1.0
    assert a_row["selected_parent_events_per_1000_eligible_parent_events"] == 1000.0
    assert a_row["juvenile_ticks"] == 1.0
    assert neutral_row["eligible_parent_events"] == 1.0
    assert neutral_row["mature_ticks"] == 1.0


def test_respec_n2_registry_and_selected_run_contract() -> None:
    registry = n2.registry_verification()

    assert registry["status"] == "PASS"
    assert registry["full_matrix_cells"] == 84
    assert registry["selected_validation_cells"] == 38
    assert len(n2.selected_rehearsal_run_ids()) == 3
    assert len(n2.selected_repeat_run_ids()) == 4
    assert n2.selected_validation_run_ids()[0].endswith("__world_capacity_balanced_mixed__arm_a_favored__seed000101")


def _write_synthetic_run(
    root: Path,
    *,
    world: str,
    arm: str,
    seed: int,
    local_pass: bool = True,
) -> dict:
    run_id = n2._run_id(world, arm, seed)
    run_dir = root / "runs" / run_id
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    attempts = 0.0 if world in {"flat", "reporting_only"} else 10.0
    rows = [
        {
            "generation": "0",
            "alive": "10",
            "successful_births": "10",
            "failed_births": "0",
            "respec0b_resource_gather_attempts": str(attempts),
            "respec0b_resource_gather_successes": "6",
            "respec0b_resource_gather_failed_handling": "0",
            "respec0b_resource_gather_resource_a_successes": "3",
            "respec0b_resource_gather_resource_b_successes": "3",
            "respec0b_resource_gather_energy_resource_a": "3",
            "respec0b_resource_gather_energy_resource_b": "3",
            "respec0b_resource_gather_handling_ticks": str(attempts),
            "respec0b_resource_gather_interruptions": "0",
            "respec0b_resource_a_cell_frac": "0.5" if world not in {"flat", "reporting_only", "a_only", "b_only"} else ("1.0" if world == "a_only" else "0.0"),
            "respec0b_resource_b_cell_frac": "0.5" if world not in {"flat", "reporting_only", "a_only", "b_only"} else ("1.0" if world == "b_only" else "0.0"),
        }
    ]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    zone_a = 6.0
    zone_b = 2.0
    if arm == "b_favored":
        zone_a, zone_b = 2.0, 6.0
    if not local_pass and arm in {"a_favored", "b_favored"}:
        zone_a = zone_b = 4.0
    zone_rows = [
        {
            "generation": 47,
            "arm": arm,
            "zone": "resource_a_zone",
            "selected_parent_events_per_1000_agent_ticks": zone_a,
            "births_per_1000_agent_ticks": zone_a,
        },
        {
            "generation": 47,
            "arm": arm,
            "zone": "resource_b_zone",
            "selected_parent_events_per_1000_agent_ticks": zone_b,
            "births_per_1000_agent_ticks": zone_b,
        },
    ]
    (metrics_dir / "respec_n1_zone_run_summary.json").write_text(
        json.dumps({"arm_zone": zone_rows}, sort_keys=True),
        encoding="utf-8",
    )
    budget_rows = [
        {
            "generation": "47",
            "schema_version": "respec_n2_capacity_budget_snapshot_v1",
            "capacity_balanced_mixed_enabled": "True",
            "formula_version": "energy_return_regrowth_balance_v1",
            "zone": "resource_a_zone",
            "resource_class": "1",
            "cell_count": "128",
            "fertility_mean": "1.0",
            "return_per_completion": "0.75",
        },
        {
            "generation": "47",
            "schema_version": "respec_n2_capacity_budget_snapshot_v1",
            "capacity_balanced_mixed_enabled": "True",
            "formula_version": "energy_return_regrowth_balance_v1",
            "zone": "resource_b_zone",
            "resource_class": "2",
            "cell_count": "128",
            "fertility_mean": "0.5",
            "return_per_completion": "1.5",
        },
    ]
    with (metrics_dir / "respec_n2_capacity_budget_generation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(budget_rows)
    return {"run_id": run_id, "status": "ok", "returncode": 0, "run_dir": str(run_dir)}


def _write_synthetic_campaign(root: Path, *, local_pass: bool = True, run_ids: list[str] | None = None) -> None:
    results = []
    ids = run_ids or n2.selected_validation_run_ids()
    for run_id in ids:
        parsed = n2.parse_run_id(run_id)
        results.append(
            _write_synthetic_run(
                root,
                world=parsed["world_family"],
                arm=parsed["fixed_arm"],
                seed=parsed["seed"],
                local_pass=local_pass,
            )
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "parallel_campaign_summary.json").write_text(
        json.dumps({"planned_runs": len(results), "failures": [], "results": results}, sort_keys=True),
        encoding="utf-8",
    )


def test_respec_n2_synthetic_positive_campaign_reaches_a(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    repeat_a = tmp_path / "repeat_a"
    repeat_b = tmp_path / "repeat_b"
    _write_synthetic_campaign(campaign)
    _write_synthetic_campaign(repeat_a, run_ids=n2.selected_repeat_run_ids())
    _write_synthetic_campaign(repeat_b, run_ids=n2.selected_repeat_run_ids())

    payloads = n2.build_payloads(campaign_root=campaign, repeat_a_root=repeat_a, repeat_b_root=repeat_b, runtime_roots={})

    assert payloads["respec_n2_campaign_completion.json"]["status"] == "PASS"
    assert payloads["respec_n2_capacity_budget_summary.json"]["status"] == "PASS"
    assert payloads["respec_n2_control_leakage_summary.json"]["status"] == "PASS"
    assert payloads["respec_n2_local_zone_direction_summary.json"]["status"] == "PASS"
    assert payloads["respec_n2_final_decision.json"]["decision"] == "RESPEC-N2-A"


def test_respec_n2_synthetic_local_failure_downgrades_to_c(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    repeat_a = tmp_path / "repeat_a"
    repeat_b = tmp_path / "repeat_b"
    _write_synthetic_campaign(campaign, local_pass=False)
    _write_synthetic_campaign(repeat_a, run_ids=n2.selected_repeat_run_ids())
    _write_synthetic_campaign(repeat_b, run_ids=n2.selected_repeat_run_ids())

    payloads = n2.build_payloads(campaign_root=campaign, repeat_a_root=repeat_a, repeat_b_root=repeat_b, runtime_roots={})

    assert payloads["respec_n2_capacity_budget_summary.json"]["status"] == "PASS"
    assert payloads["respec_n2_local_zone_direction_summary.json"]["status"] == "FAIL"
    assert payloads["respec_n2_final_decision.json"]["decision"] == "RESPEC-N2-C"
