#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCIENTIFIC_KEYS = {
    ("sim", "generations"),
    ("sim", "steps_per_generation"),
    ("sim", "seed"),
    ("sim", "policy_tick_stride"),
    ("sim", "policy_weight_mutation_enable"),
    ("sim", "policy_weight_mutation_mode"),
    ("sim", "policy_weight_mutation_rate"),
    ("sim", "policy_weight_mutation_std"),
    ("sim", "genome_c1_enable"),
    ("sim", "genome_c2_enable"),
    ("sim", "chemistry_functional_coupling_enable"),
    ("sim", "maintenance_enable"),
    ("env", "width"),
    ("env", "height"),
    ("env", "view_radius"),
    ("env", "food_spawn_per_step"),
    ("env", "max_food"),
    ("env", "max_energy"),
    ("env", "food_energy"),
    ("env", "move_cost"),
    ("env", "gather_cost"),
    ("env", "rest_cost"),
    ("env", "attack_cost"),
    ("env", "start_health"),
    ("env", "reproduction_threshold"),
    ("env", "reproduce_try_prob"),
    ("env", "reproduce_requires_mate"),
    ("env", "reproduce_parent_energy_share"),
    ("env", "reproduce_min_energy_buffer"),
    ("env", "reproduce_cooldown_steps"),
    ("env", "enable_injury"),
    ("env", "enable_scar"),
    ("env", "injury_repro_block"),
    ("env", "scar_repro_block"),
    ("env", "injury_repro_k_injury"),
    ("env", "injury_repro_k_scar"),
    ("env", "injury_birth_fail_scar"),
    ("env", "injury_birth_fail_injury"),
    ("env", "enable_rest_charge"),
    ("env", "patch_world_enable"),
    ("env", "jungle_predator_pressure_enable"),
    ("env", "jungle_ecological_recovery_enable"),
    ("env", "wdl_delayed_hazard_enable"),
}

NATIVE_EXECUTION_CONTROL_KEYS = {
    ("sim", "native_runtime_enable"),
    ("sim", "native_runtime_fail_closed"),
    ("sim", "native_runtime_birth_commit_enable"),
    ("sim", "native_runtime_max_births_per_generation"),
    ("sim", "native_runtime_build_dir"),
    ("sim", "native_runtime_allow_runtime_jit_build"),
    ("sim", "native_runtime_fixed_policy_family"),
    ("sim", "native_runtime_cxx26_scope_lock_enable"),
    ("sim", "native_runtime_write_summary"),
    ("sim", "native_safe_default_enable"),
    ("sim", "native_execution_profile"),
}

REPORTING_STORAGE_KEYS = {
    ("sim", "campaign_output_profile"),
    ("sim", "allow_large_output_profile"),
    ("sim", "allow_generation_checkpointing"),
    ("sim", "checkpoint_bundle_v2_period_gen"),
    ("sim", "checkpoint_bundle_v2_strict_replay_mode"),
    ("sim", "lineage_reproduction_funnel_write_events"),
    ("sim", "lineage_reproduction_funnel_write_per_generation"),
    ("sim", "behavior_log_enable"),
}


def _read_run_config(run_dir: Path) -> dict[tuple[str, str], str]:
    path = run_dir / "run_config.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            out[(str(row["section"]), str(row["key"]))] = str(row["value"])
    return out


def _diff_for_keys(
    reference: dict[tuple[str, str], str],
    native: dict[tuple[str, str], str],
    keys: set[tuple[str, str]],
) -> list[dict[str, str]]:
    diffs: list[dict[str, str]] = []
    for section, key in sorted(keys):
        ref_value = reference.get((section, key), "<missing>")
        native_value = native.get((section, key), "<missing>")
        if ref_value != native_value:
            diffs.append(
                {
                    "section": section,
                    "key": key,
                    "reference": ref_value,
                    "native": native_value,
                }
            )
    return diffs


def _metric_row(run_dir: Path) -> dict[str, str]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    return dict(rows[0]) if rows else {}


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except Exception:
        return None


def compare_scientific_config(reference_run: Path, native_run: Path) -> dict[str, Any]:
    reference = _read_run_config(reference_run)
    native = _read_run_config(native_run)
    scientific_diffs = _diff_for_keys(reference, native, SCIENTIFIC_KEYS)
    native_control_diffs = _diff_for_keys(reference, native, NATIVE_EXECUTION_CONTROL_KEYS)
    reporting_diffs = _diff_for_keys(reference, native, REPORTING_STORAGE_KEYS)
    reference_metric = _metric_row(reference_run)
    native_metric = _metric_row(native_run)
    capacity = int(float(reference.get(("env", "width"), "0"))) * int(float(reference.get(("env", "height"), "0")))
    start_pop = 64
    try:
        start_pop = int(float(reference.get(("sim", "population_size"), reference.get(("sim", "pop"), "64"))))
    except Exception:
        pass
    ref_births = _float(reference_metric, "successful_births")
    native_births = _float(native_metric, "successful_births")
    reference_effective_free_slot_birth_capacity = capacity - start_pop + int(_float(reference_metric, "deaths") or 0)
    native_effective_free_slot_birth_capacity = capacity - start_pop + int(_float(native_metric, "deaths") or 0)
    native_raw_cap = int(float(native.get(("sim", "native_runtime_max_births_per_generation"), "1") or 1))
    effective_birth_cap = {
        "schema_version": "eiel_effective_birth_cap_contract_v1",
        "capacity": capacity,
        "initial_population": start_pop,
        "reference_free_slot_birth_capacity": reference_effective_free_slot_birth_capacity,
        "native_free_slot_birth_capacity": native_effective_free_slot_birth_capacity,
        "native_raw_max_births_per_generation": native_raw_cap,
        "reference_successful_births": ref_births,
        "native_successful_births": native_births,
        "native_raw_cap_binds": bool(native_births is not None and native_births >= native_raw_cap),
        "classification": "capacity_and_free_slots_are_effective_birth_limit_in_this_envelope",
    }
    return {
        "schema_version": "eiel_scientific_config_comparison_v1",
        "reference_run_dir": str(reference_run),
        "native_run_dir": str(native_run),
        "scientific_fields_checked": [f"{section}.{key}" for section, key in sorted(SCIENTIFIC_KEYS)],
        "scientific_differences": scientific_diffs,
        "native_execution_control_differences": native_control_diffs,
        "reporting_storage_differences": reporting_diffs,
        "effective_birth_cap": effective_birth_cap,
        "ok": len(scientific_diffs) == 0,
        "fail_closed": len(scientific_diffs) != 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare EIEL reference/native effective scientific config.")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare_scientific_config(args.reference_run, args.native_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "fail_closed": result["fail_closed"]}, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
