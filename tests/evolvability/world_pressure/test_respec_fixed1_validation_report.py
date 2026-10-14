from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.bin import respec_fixed1_validation_report as fixed1


def _write_run(root: Path, *, world: str, arm: str, seed: int, attempts: float, a_succ: float, b_succ: float) -> dict:
    run_id = fixed1._run_id(world, arm, seed)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "generation": "0",
            "alive": "8",
            "deaths": "0",
            "successful_births": "1",
            "failed_births": "0",
            "respec0b_resource_gather_attempts": str(attempts),
            "respec0b_resource_gather_successes": str(a_succ + b_succ),
            "respec0b_resource_gather_failed_handling": "0",
            "respec0b_resource_gather_resource_a_successes": str(a_succ),
            "respec0b_resource_gather_resource_b_successes": str(b_succ),
            "respec0b_resource_gather_energy_resource_a": str(a_succ),
            "respec0b_resource_gather_energy_resource_b": str(b_succ),
            "respec0b_resource_gather_handling_ticks": str(max(1.0, attempts)),
            "respec0b_resource_gather_interruptions": "0",
            "respec0b_resource_a_cell_frac": "1.0" if world == "a_only" else ("0.0" if world == "b_only" else "0.5"),
            "respec0b_resource_b_cell_frac": "0.0" if world == "a_only" else ("1.0" if world == "b_only" else "0.5"),
        }
    ]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "env_cfg": {
                    "patch_world_enable": world != "flat",
                    "patch_world_observation_enable": world != "flat",
                    "respec0b_resource_specialization_enable": world not in {"flat", "reporting_only"},
                    "respec0b_resource_class_mode": {
                        "a_only": "all_a",
                        "b_only": "all_b",
                    }.get(world, "vertical_split_ab"),
                    "respec0b_resource_equalized_sham_enable": world == "equalized_sham",
                },
                "sim_cfg": {
                    "resource_handling_representation_enable": True,
                    "resource_handling_representation_reporting_enable": True,
                    "checkpoint_bundle_v2_period_gen": 0,
                    "lineage_qd_archive_enable": False,
                    "lineage_founder_balance_enable": False,
                    "lro1_reproductive_opportunity_enable": False,
                    "lro4_offspring_dispersal_enable": False,
                    "pressure_autopilot_enable": False,
                },
                "genome_cfg": {"overrides": {"resource_handling_axis": str(fixed1.FIXED_ARMS[arm]), "mutation_rate": "0.0"}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {"run_id": run_id, "status": "ok", "returncode": 0, "run_dir": str(run_dir)}


def _write_summary(root: Path, results: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "parallel_campaign_summary.json").write_text(
        json.dumps({"status": "complete", "planned_runs": len(results), "failures": [], "results": results}, sort_keys=True),
        encoding="utf-8",
    )


def test_respec_fixed1_contract_is_frozen_and_names_decisions() -> None:
    contract = fixed1.decision_contract()
    assert contract["contract_status"] == "FROZEN_BEFORE_EXECUTION"
    assert contract["decision_classes"]["RESPEC-FIXED-A"]
    assert contract["seed_consistency_rule"]["mandatory_support"] == "at least 3 of 4 seeds support each mandatory reciprocal direction"


def test_respec_fixed1_run_id_sets_are_deterministic() -> None:
    assert len(fixed1.selected_rehearsal_run_ids()) == 18
    assert len(fixed1.selected_parity_run_ids()) == 6
    assert fixed1.selected_rehearsal_run_ids()[0].endswith("__world_flat__arm_a_favored__seed000101")


def test_respec_fixed1_synthetic_positive_campaign_reaches_a(tmp_path: Path) -> None:
    results = []
    for world in fixed1.WORLD_FAMILIES:
        for arm in fixed1.FIXED_ARMS:
            for seed in fixed1.SEEDS:
                attempts = 10.0 if world not in {"flat", "reporting_only"} else 0.0
                a_succ = 0.0
                b_succ = 0.0
                if world == "a_only":
                    a_succ = {"a_favored": 9.0, "neutral": 6.0, "b_favored": 2.0}[arm]
                elif world == "b_only":
                    b_succ = {"a_favored": 2.0, "neutral": 6.0, "b_favored": 9.0}[arm]
                elif world in {"equalized_sham", "mixed_tradeoff"}:
                    a_succ = 3.0
                    b_succ = 3.0
                results.append(_write_run(tmp_path, world=world, arm=arm, seed=seed, attempts=attempts, a_succ=a_succ, b_succ=b_succ))
    _write_summary(tmp_path, results)

    stage = fixed1.collect_stage_runs(tmp_path)
    integrity = fixed1.stage_integrity(stage, expected_runs=72, expected_seeds=fixed1.SEEDS)
    deltas = fixed1.paired_deltas(stage)
    direction = fixed1.direction_summary(deltas)
    decision = fixed1.final_decision(
        rehearsal_integrity={"status": "PASS"},
        campaign_integrity=integrity,
        repeat_parity={"status": "PASS"},
        direction=direction,
        equalized=fixed1.equalized_sham_audit(stage),
        flat_reporting=fixed1.flat_reporting_audit(stage),
        observation=fixed1.observation_action_audit(stage),
        neutral=fixed1.neutral_generalist_summary(stage),
    )

    assert integrity["status"] == "PASS"
    assert direction["status"] == "PASS"
    assert decision["decision"] == "RESPEC-FIXED-A"


def test_respec_fixed1_equalized_sham_failure_precedes_positive_cells(tmp_path: Path) -> None:
    results = []
    for world in fixed1.WORLD_FAMILIES:
        for arm in fixed1.FIXED_ARMS:
            for seed in fixed1.SEEDS:
                attempts = 10.0 if world not in {"flat", "reporting_only"} else 0.0
                a_succ = 0.0
                b_succ = 0.0
                if world == "a_only":
                    a_succ = {"a_favored": 9.0, "neutral": 6.0, "b_favored": 2.0}[arm]
                elif world == "b_only":
                    b_succ = {"a_favored": 2.0, "neutral": 6.0, "b_favored": 9.0}[arm]
                elif world == "equalized_sham":
                    a_succ = {"a_favored": 9.0, "neutral": 4.0, "b_favored": 1.0}[arm]
                elif world == "mixed_tradeoff":
                    a_succ = 3.0
                    b_succ = 3.0
                results.append(_write_run(tmp_path, world=world, arm=arm, seed=seed, attempts=attempts, a_succ=a_succ, b_succ=b_succ))
    _write_summary(tmp_path, results)
    stage = fixed1.collect_stage_runs(tmp_path)
    integrity = fixed1.stage_integrity(stage, expected_runs=72, expected_seeds=fixed1.SEEDS)
    decision = fixed1.final_decision(
        rehearsal_integrity={"status": "PASS"},
        campaign_integrity=integrity,
        repeat_parity={"status": "PASS"},
        direction=fixed1.direction_summary(fixed1.paired_deltas(stage)),
        equalized=fixed1.equalized_sham_audit(stage),
        flat_reporting=fixed1.flat_reporting_audit(stage),
        observation=fixed1.observation_action_audit(stage),
        neutral=fixed1.neutral_generalist_summary(stage),
    )
    assert decision["decision"] == "RESPEC-FIXED-C"
