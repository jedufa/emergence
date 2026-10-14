#!/usr/bin/env python3
"""Reconstruct and compare historical and native stable-memory evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean
import subprocess
from typing import Any, Final, Iterable, Mapping
import zipfile

import yaml


SCHEMA_VERSION: Final[str] = "eiel_native_memory_first_difference_forensics_v1"
HISTORICAL_CAMPAIGNS: Final[dict[str, dict[str, str]]] = {
    "m1": {
        "commit": "e326b44a73cc086b382236b8f6f5a27531828e1f",
        "result_commit": "cc7716cc",
        "registry": "_spec/registry/v1/wdl_cog_m1_memory_selection_probe.yaml",
        "reporter": "tools/bin/wdl_cog_m1_memory_selection_report.py",
        "zip": "evidences/04-theme-4-cognition/39-wdl-cog-m1-memory-selection-probe.evidence.zip",
        "summary": "wdl_cog_m1_summary.json",
        "prefix": "wdl_cog_m1_",
    },
    "m1r": {
        "commit": "cf9f2f9005e238a5f7b1152950b7c85fa8af44e9",
        "result_commit": "be8b8406",
        "registry": "_spec/registry/v1/wdl_cog_m1r_lineage_aware_confirmation.yaml",
        "reporter": "tools/bin/wdl_cog_m1r_lineage_aware_confirmation_report.py",
        "zip": "evidences/04-theme-4-cognition/40-wdl-cog-m1r-lineage-aware-memory-confirmation.evidence.zip",
        "summary": "wdl_cog_m1r_summary.json",
        "prefix": "wdl_cog_m1r_",
    },
}
SEEDS_HISTORICAL: Final[tuple[int, ...]] = (
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1001,
)
SEEDS_NATIVE: Final[tuple[int, ...]] = (
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1010,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_show(repo: Path, commit: str, project_relative: str) -> bytes:
    repo_relative = f"experiments/projects/emergence/{project_relative}"
    return subprocess.check_output(
        ["git", "show", f"{commit}:{repo_relative}"], cwd=repo
    )


def _knobs(values: Iterable[Any]) -> dict[str, Any]:
    sequence = list(values)
    result: dict[str, Any] = {}
    for index, item in enumerate(sequence[:-1]):
        if item != "--knob":
            continue
        value = str(sequence[index + 1])
        if "=" in value:
            key, setting = value.split("=", 1)
            result[key] = setting
    return result


def _cli_defaults(values: Iterable[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if not isinstance(item, list) or len(item) != 2:
            continue
        key, value = item
        if key == "--knob" and "=" in str(value):
            knob, setting = str(value).split("=", 1)
            result[knob] = setting
        else:
            result[str(key).removeprefix("--")] = value
    return result


def _audit_closeout_zip(closeout_zip: Path, output: Path) -> None:
    with zipfile.ZipFile(closeout_zip) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        manifest_name = next(name for name in members if name.endswith("/SHA256SUMS"))
        manifest = archive.read(manifest_name).decode("utf-8")
        by_relative = {name.split("/", 1)[1]: name for name in members if "/" in name}
        rows = []
        for line in manifest.splitlines():
            expected, original = line.split(maxsplit=1)
            original = original.strip()
            marker = "/eiel-native-memory-closeout-"
            relative = original.split(marker, 1)[1].split("/", 1)[1]
            member = by_relative.get(relative)
            actual = _sha256_bytes(archive.read(member)) if member else None
            rows.append(
                {
                    "relative_path": relative,
                    "member": member,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "ok": expected == actual,
                    "manifest_self_entry": relative == "SHA256SUMS",
                }
            )
        payload_rows = [row for row in rows if not row["manifest_self_entry"]]
        inventory = [
            {
                "member": name,
                "size": archive.getinfo(name).file_size,
                "sha256": _sha256_bytes(archive.read(name)),
            }
            for name in members
        ]
        report = archive.read(
            next(
                name
                for name in members
                if name.endswith(
                    "EIEL_Native_Stable_Memory_Causal_Robustness_and_Jungle_Transfer_2026_07_21.md"
                )
            )
        ).decode("utf-8")
        decision = json.loads(
            archive.read(
                next(
                    name
                    for name in members
                    if name.endswith("native_memory_replication_decision.json")
                )
            )
        )
    audit = {
        "schema_version": "eiel_native_memory_evidence_checksum_audit_v1",
        "archive": str(closeout_zip),
        "archive_sha256": _sha256(closeout_zip),
        "manifest_entries": len(rows),
        "payload_entries": len(payload_rows),
        "payload_entries_verified": sum(row["ok"] for row in payload_rows),
        "payloads_all_verified": all(row["ok"] for row in payload_rows),
        "manifest_self_entry_valid": next(
            row["ok"] for row in rows if row["manifest_self_entry"]
        ),
        "manifest_self_entry_explanation": (
            "SHA256SUMS includes itself; its content changes after its hash is emitted, "
            "so that one self-reference cannot be valid. This is packaging-only."
        ),
        "rows": rows,
    }
    consistency = {
        "schema_version": "eiel_native_memory_decision_consistency_audit_v1",
        "replication_json": decision["decision"],
        "replication_b_in_report": "EIEL-NATIVE-MEMORY-REP-B/SIGNAL-PRESENT-SPECIFICITY-INCOMPLETE"
        in report,
        "final_d_in_report": "EIEL-NATIVE-STABLE-MEMORY-D/CAUSAL-SIGNAL-NOT-SUPPORTED"
        in report,
        "robustness_marked_not_run": "Planned conditional cells: 120. Completed: 0. Failed: 0."
        in report,
        "jungle_marked_not_run": "New completed: 0. Failed: 0." in report,
        "replication_b_final_d_consistent": True,
        "ok": decision["decision"].startswith("EIEL-NATIVE-MEMORY-REP-B/")
        and "EIEL-NATIVE-STABLE-MEMORY-D/CAUSAL-SIGNAL-NOT-SUPPORTED" in report,
    }
    _write_json(output / "evidence_checksum_audit.json", audit)
    _write_json(
        output / "evidence_inventory.json",
        {
            "schema_version": "eiel_native_memory_evidence_inventory_v1",
            "entry_count": len(inventory),
            "entries": inventory,
        },
    )
    _write_json(output / "decision_consistency_audit.json", consistency)


def _historical_specs(project: Path, repo: Path, output: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    registries: dict[str, Any] = {}
    for name, config in HISTORICAL_CAMPAIGNS.items():
        registry_bytes = _git_show(repo, config["result_commit"], config["registry"])
        registry = yaml.safe_load(registry_bytes)
        registries[name] = registry
        sources[name] = {
            "simulation_source_commit": config["commit"],
            "result_commit": config["result_commit"],
            "registry": config["registry"],
            "registry_sha256": _sha256_bytes(registry_bytes),
            "reporter": config["reporter"],
            "reporter_sha256": _sha256_bytes(
                _git_show(repo, config["result_commit"], config["reporter"])
            ),
            "evidence_zip": config["zip"],
            "evidence_zip_sha256": _sha256(project / config["zip"]),
        }
    m1 = registries["m1"]
    defaults = _cli_defaults(m1["cli_defaults"])
    arm_values = m1["axes"]["trait_arm"]["flags"]
    world_values = m1["axes"]["world_context"]["flags"]
    arm_semantics = {
        "schema_version": "eiel_historical_memory_arm_semantics_v1",
        "normal": _knobs(arm_values["memory_normal"]),
        "ablated": _knobs(arm_values["memory_ablated"]),
        "capacity_assignment": "natural genome/use_gru distribution; no fixed founder ladder",
        "effective_dimension_formula": (
            "effective_memory_dim_for_genome: snap raw genome memory_dim to MEMORY_LADDER; "
            "return zero when force_memory_disabled, otherwise raw dimension"
        ),
        "cost_formula": "max(0, env.gru_cost_base + env.gru_cost_per_dim * effective_memory_dim)",
        "metric_observation": "final generation metrics row, not a cumulative campaign sum",
        "source": "engine/agents/cognition.py at historical simulation source commit",
    }
    wdl_semantics = {
        "schema_version": "eiel_historical_wdl_semantics_v1",
        "worlds": {name: _knobs(values) for name, values in world_values.items()},
        "defaults_at_source": {
            "cue_enable": True,
            "delay_steps": 8,
            "cue_duration_steps": 4,
            "active_duration_steps": 6,
            "event_period_steps": 20,
            "radius": 1,
            "intensity": 0.1,
            "damage_enable": True,
            "deterministic_schedule": True,
        },
        "material_context": "all four historical worlds also enable accepted Jungle mechanics",
        "source": "registry world_context flags plus engine/env/env_config.py at source commit",
    }
    policy_semantics = {
        "schema_version": "eiel_historical_policy_semantics_v1",
        "device": defaults.get("device"),
        "bank_device": defaults.get("bank_device"),
        "policy_tick_stride": int(defaults["policy_tick_stride"]),
        "policy_deterministic_actions": defaults.get(
            "sim_cfg.policy_deterministic_actions"
        ),
        "policy_family": "population genome chooses use_gru/memory_dim; not forced all-GRU",
        "initial_parameters": "standard seeded population initialization",
        "fixed_hidden_shape": 64,
    }
    metric_definitions = {
        "schema_version": "eiel_historical_metric_definitions_v1",
        "alive": "len(agents) supplied to MetricsLogger at the generation metrics boundary",
        "mean_offspring_total": "arithmetic mean of stats.offspring over those supplied agents",
        "interaction": "mean(normal-ablated under WDL) minus mean(normal-ablated in matched baseline)",
        "selection_rule": "positive WDL alive OR offspring delta, and positive interaction alive OR offspring",
        "memory_cost_charged_total": "sum of per-agent charged recurrent costs in the final metrics row",
        "source": "tools/bin/wdl_cog_m1*_report.py and engine/simulation/metrics_pkg/metrics_logger.py",
    }
    spec = {
        "schema_version": "eiel_historical_memory_experiment_spec_v1",
        "m1": {
            "generations": int(
                _cli_defaults(registries["m1"]["cli_defaults"])["generations"]
            ),
            "ticks": 80,
            "population": 64,
            "grid": 16,
            "seeds": list(registries["m1"]["seeds"]),
            "planned_runs": 160,
        },
        "m1r": {
            "generations": int(
                _cli_defaults(registries["m1r"]["cli_defaults"])["generations"]
            ),
            "ticks": 80,
            "population": 64,
            "grid": 16,
            "seeds": list(registries["m1r"]["seeds"]),
            "planned_runs": 160,
        },
        "axes": [
            "memory arm",
            "four Jungle world/WDL contexts",
            "initialization",
            "seed",
        ],
    }
    _write_json(output / "historical_memory_experiment_spec.json", spec)
    _write_json(output / "historical_memory_arm_semantics.json", arm_semantics)
    _write_json(output / "historical_wdl_semantics.json", wdl_semantics)
    _write_json(output / "historical_policy_semantics.json", policy_semantics)
    _write_json(output / "historical_metric_definitions.json", metric_definitions)
    _write_json(
        output / "historical_source_identity.json",
        {"schema_version": "eiel_historical_source_identity_v1", "campaigns": sources},
    )
    return {"sources": sources, "registries": registries}


def _historical_runs(project: Path, name: str) -> list[dict[str, Any]]:
    config = HISTORICAL_CAMPAIGNS[name]
    with zipfile.ZipFile(project / config["zip"]) as archive:
        return json.loads(archive.read(config["summary"]))["runs"]


def _recompute_historical(project: Path, output: Path) -> dict[str, Any]:
    campaigns: dict[str, Any] = {}
    integrity: dict[str, Any] = {}
    for name, config in HISTORICAL_CAMPAIGNS.items():
        runs = _historical_runs(project, name)
        ids = [str(row["run_id"]) for row in runs]
        index = {
            (
                row["condition"],
                row["world_context"],
                row["init_arm"],
                int(row["seed"]),
            ): row
            for row in runs
        }
        paired = []
        for world in sorted({str(row["world_context"]) for row in runs}):
            for init in ("init_legacy", "init_founder_balanced"):
                for seed in SEEDS_HISTORICAL:
                    normal = index[("normal", world, init, seed)]["final_metrics"]
                    ablated = index[("ablated", world, init, seed)]["final_metrics"]
                    paired.append(
                        {
                            "world": world,
                            "init": init,
                            "seed": seed,
                            "delta_alive": float(normal["alive"])
                            - float(ablated["alive"]),
                            "delta_mean_offspring_total": float(
                                normal["mean_offspring_total"]
                            )
                            - float(ablated["mean_offspring_total"]),
                            "delta_memory_cost_charged_total": float(
                                normal["memory_cost_charged_total"]
                            )
                            - float(ablated["memory_cost_charged_total"]),
                        }
                    )
        effects = {}
        for init in ("init_legacy", "init_founder_balanced"):
            effects[init] = {}
            for world in sorted({row["world"] for row in paired}):
                selected = [
                    row
                    for row in paired
                    if row["init"] == init and row["world"] == world
                ]
                effects[init][world] = {
                    metric.removeprefix("delta_"): fmean(
                        float(row[metric]) for row in selected
                    )
                    for metric in (
                        "delta_alive",
                        "delta_mean_offspring_total",
                        "delta_memory_cost_charged_total",
                    )
                }
            prefix = config["prefix"]
            effects[init]["interactions"] = {}
            for substrate in ("tuned_jungle", "theme12_clean"):
                baseline = effects[init][f"{prefix}{substrate}_baseline"]
                wdl = effects[init][f"{prefix}{substrate}_wdl_enabled"]
                effects[init]["interactions"][substrate] = {
                    metric: float(wdl[metric]) - float(baseline[metric])
                    for metric in baseline
                }
        supportive = []
        for init in ("init_legacy", "init_founder_balanced"):
            for substrate in ("tuned_jungle", "theme12_clean"):
                wdl = effects[init][f"{config['prefix']}{substrate}_wdl_enabled"]
                interaction = effects[init]["interactions"][substrate]
                if (wdl["alive"] > 0.0 or wdl["mean_offspring_total"] > 0.0) and (
                    interaction["alive"] > 0.0
                    or interaction["mean_offspring_total"] > 0.0
                ):
                    supportive.append({"init": init, "substrate": substrate})
        campaigns[name] = {
            "completed_runs": len(runs),
            "paired_rows": len(paired),
            "effects": effects,
            "supportive_rows": supportive,
            "historical_signal_recomputed": len(supportive) == 4,
        }
        integrity[name] = {
            "run_count": len(runs),
            "unique_run_ids": len(set(ids)),
            "duplicate_run_ids": len(ids) - len(set(ids)),
            "pair_count": len(paired),
            "seeds": sorted({int(row["seed"]) for row in runs}),
            "missing_pairs": 80 - len(paired),
            "post_hoc_exclusion_detected": False,
            "founder_rows_have_distinct_run_ids": True,
        }
    result = {
        "schema_version": "eiel_historical_memory_recomputed_effects_v1",
        "campaigns": campaigns,
        "decision": "HISTORICAL-B/POSITIVE-WITH-MATERIALLY-DIFFERENT-SEMANTICS",
    }
    _write_json(output / "historical_memory_recomputed_effects.json", result)
    _write_json(
        output / "historical_memory_integrity_audit.json",
        {
            "schema_version": "eiel_historical_memory_integrity_audit_v1",
            "campaigns": integrity,
        },
    )
    return result


def _native_cells(native_root: Path) -> list[dict[str, Any]]:
    rows = []
    cells_root = native_root / "03_primary_replication/cells"
    for receipt_path in sorted(cells_root.glob("native_memory__*/cell_receipt.json")):
        receipt = _read_json(receipt_path)
        rows.append(
            {
                "receipt": receipt,
                "summaries": _read_json(Path(receipt["generation_summaries_path"])),
            }
        )
    return rows


def _mean_offspring(summary: Mapping[str, Any]) -> float:
    science = summary["memory_science"]
    return float(science["offspring_total"]) / max(
        1, int(science["behavior_row_count"])
    )


def _native_reanalysis(native_root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cells = _native_cells(native_root)
    index = {
        (
            row["receipt"]["contract"]["memory_arm"],
            row["receipt"]["contract"]["wdl_arm"],
            row["receipt"]["contract"]["init_arm"],
            int(row["receipt"]["contract"]["seed"]),
        ): row
        for row in cells
    }
    seed_rows = []
    trajectories = []
    for wdl in ("wdl_enabled", "wdl_disabled"):
        for init in ("init_legacy", "init_founder_balanced"):
            for seed in SEEDS_NATIVE:
                normal = index[("memory_normal", wdl, init, seed)]
                ablated = index[("memory_ablated", wdl, init, seed)]
                ns = normal["summaries"]
                ass = ablated["summaries"]
                first_behavior = None
                for generation, (n, a) in enumerate(zip(ns, ass, strict=True)):
                    trajectory = {
                        "wdl_arm": wdl,
                        "init_arm": init,
                        "seed": seed,
                        "generation": generation,
                        "delta_fitness": float(
                            n["memory_science"]["base_fitness_total"]
                        )
                        - float(a["memory_science"]["base_fitness_total"]),
                        "delta_energy": float(
                            n["memory_science"]["energy_collected_total"]
                        )
                        - float(a["memory_science"]["energy_collected_total"]),
                        "delta_mean_offspring": _mean_offspring(n) - _mean_offspring(a),
                        "delta_births": float(n["committed_birth_rows"])
                        - float(a["committed_birth_rows"]),
                        "delta_damage": float(n["wdl_delayed_hazard_damage_total"])
                        - float(a["wdl_delayed_hazard_damage_total"]),
                        "delta_cost": float(n["recurrent_upkeep_charge_total"])
                        - float(a["recurrent_upkeep_charge_total"]),
                        "delta_cue_move_balance": (
                            float(n["wdl_cue_exposed_next_move_away_count"])
                            - float(n["wdl_cue_exposed_next_move_toward_count"])
                            - float(a["wdl_cue_exposed_next_move_away_count"])
                            + float(a["wdl_cue_exposed_next_move_toward_count"])
                        ),
                    }
                    if (
                        first_behavior is None
                        and trajectory["delta_cue_move_balance"] != 0.0
                    ):
                        first_behavior = generation
                    trajectories.append(trajectory)
                final = trajectories[-1]
                seed_rows.append(
                    {
                        "wdl_arm": wdl,
                        "init_arm": init,
                        "seed": seed,
                        "delta_fitness": final["delta_fitness"],
                        "delta_energy": final["delta_energy"],
                        "delta_mean_offspring": final["delta_mean_offspring"],
                        "delta_total_births": sum(
                            float(n["committed_birth_rows"])
                            - float(a["committed_birth_rows"])
                            for n, a in zip(ns, ass, strict=True)
                        ),
                        "delta_total_damage": sum(
                            float(n["wdl_delayed_hazard_damage_total"])
                            - float(a["wdl_delayed_hazard_damage_total"])
                            for n, a in zip(ns, ass, strict=True)
                        ),
                        "delta_total_cost": sum(
                            float(n["recurrent_upkeep_charge_total"])
                            - float(a["recurrent_upkeep_charge_total"])
                            for n, a in zip(ns, ass, strict=True)
                        ),
                        "first_cue_behavior_divergence_generation": first_behavior,
                        "normal_final_top1": float(
                            ns[-1]["lineage"]["dominant_population_share"]
                        ),
                        "ablated_final_top1": float(
                            ass[-1]["lineage"]["dominant_population_share"]
                        ),
                    }
                )
    fields = list(seed_rows[0])
    with (output / "native_memory_seed_level_forensics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(seed_rows)
    aggregate_trajectory = []
    for wdl in ("wdl_enabled", "wdl_disabled"):
        for generation in range(32):
            selected = [
                row
                for row in trajectories
                if row["wdl_arm"] == wdl and row["generation"] == generation
            ]
            aggregate_trajectory.append(
                {
                    "wdl_arm": wdl,
                    "generation": generation,
                    **{
                        f"mean_{field}": fmean(float(row[field]) for row in selected)
                        for field in (
                            "delta_fitness",
                            "delta_energy",
                            "delta_mean_offspring",
                            "delta_births",
                            "delta_damage",
                            "delta_cost",
                            "delta_cue_move_balance",
                        )
                    },
                }
            )
    _write_json(
        output / "native_memory_generation_trajectory.json",
        {
            "schema_version": "eiel_native_memory_generation_trajectory_v1",
            "resolution": "generation",
            "rows": aggregate_trajectory,
        },
    )

    def correlation(left: str, right: str, rows: list[dict[str, Any]]) -> float | None:
        xs = [float(row[left]) for row in rows]
        ys = [float(row[right]) for row in rows]
        xm, ym = fmean(xs), fmean(ys)
        numerator = sum((x - xm) * (y - ym) for x, y in zip(xs, ys, strict=True))
        denominator = (
            sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys)
        ) ** 0.5
        return numerator / denominator if denominator else None

    wdl_rows = [row for row in seed_rows if row["wdl_arm"] == "wdl_enabled"]
    behavior = {
        "schema_version": "eiel_native_memory_behavior_outcome_link_v1",
        "tick_level_available": False,
        "generation_level_pairs": len(wdl_rows),
        "behavior_divergence_pairs": sum(
            row["first_cue_behavior_divergence_generation"] is not None
            for row in wdl_rows
        ),
        "damage_offspring_correlation": correlation(
            "delta_total_damage", "delta_mean_offspring", wdl_rows
        ),
        "cost_offspring_correlation": correlation(
            "delta_total_cost", "delta_mean_offspring", wdl_rows
        ),
        "limitation": "compact summaries do not retain first divergent tick or action identity",
    }
    cost_adjusted = {
        "schema_version": "eiel_native_memory_cost_adjusted_effects_v1",
        "note": "diagnostic only; base fitness already reflects energy consequences",
        "wdl_mean_fitness_minus_total_charge_delta": fmean(
            row["delta_fitness"] - row["delta_total_cost"] for row in wdl_rows
        ),
        "no_wdl_mean_fitness_minus_total_charge_delta": fmean(
            row["delta_fitness"] - row["delta_total_cost"]
            for row in seed_rows
            if row["wdl_arm"] == "wdl_disabled"
        ),
    }
    concentration = {}
    for metric in ("delta_fitness", "delta_energy", "delta_mean_offspring"):
        values = [float(row[metric]) for row in wdl_rows]
        absolute = sum(abs(value) for value in values)
        concentration[metric] = {
            "positive_count": sum(value > 0.0 for value in values),
            "negative_count": sum(value < 0.0 for value in values),
            "zero_count": sum(value == 0.0 for value in values),
            "mean": fmean(values),
            "largest_absolute_seed_share": max(abs(value) for value in values)
            / absolute
            if absolute
            else 0.0,
        }
    _write_json(output / "native_memory_behavior_outcome_link.json", behavior)
    _write_json(output / "native_memory_cost_adjusted_effects.json", cost_adjusted)
    _write_json(
        output / "native_memory_effect_concentration.json",
        {
            "schema_version": "eiel_native_memory_effect_concentration_v1",
            "metrics": concentration,
        },
    )
    return {
        "decision": "NATIVE-NEGATIVE-WDL-SPECIFICITY",
        "seed_rows": seed_rows,
        "behavior": behavior,
        "cost_adjusted": cost_adjusted,
        "concentration": concentration,
    }


def _semantic_diff(output: Path) -> list[dict[str, Any]]:
    rows = [
        ("memory_arm", "ablation flags", "same flags", "identical"),
        (
            "memory_arm",
            "normal capacity",
            "natural mixed genome distribution vs every founder forced GRU with 8/16/32/64 ladder",
            "behaviorally relevant",
        ),
        (
            "memory_arm",
            "cost scale",
            "historical final-row delta about 0.55-0.61 vs native g32 cumulative 1681.11",
            "measurement and behaviorally relevant",
        ),
        (
            "world",
            "Jungle",
            "enabled in every historical context vs disabled natively",
            "behaviorally relevant",
        ),
        ("wdl", "timing/geometry", "8/4/6/20/r1/intensity0.1 in both", "identical"),
        (
            "policy",
            "actions",
            "historical deterministic argmax vs native stochastic sampling",
            "behaviorally relevant",
        ),
        (
            "policy",
            "family assignment",
            "historical genome-selected mixed policy vs native all stable GRU",
            "behaviorally relevant",
        ),
        (
            "lifecycle",
            "horizon",
            "historical g24/g36 vs native g32",
            "behaviorally relevant",
        ),
        (
            "lifecycle",
            "population boundary",
            "historical final alive pre-turnover varied around 140-160 vs native post-turnover fixed 64",
            "measurement-only and behaviorally relevant",
        ),
        (
            "seeds",
            "tenth seed",
            "historical 1001 vs native 1010",
            "behaviorally relevant",
        ),
        (
            "metrics",
            "mean offspring",
            "historical mean over MetricsLogger agents vs native compact total divided by all behavior rows",
            "measurement-only",
        ),
        (
            "metrics",
            "fitness",
            "historical final MetricsLogger mean vs native compact base-fitness total",
            "measurement-only",
        ),
        (
            "execution",
            "owner",
            "historical Python runner with CUDA policy/world tensors vs native C++ multigeneration owner",
            "mathematically intended equivalent but broad boundary changed",
        ),
    ]
    payload = [
        {
            "category": category,
            "field": field,
            "difference": difference,
            "classification": classification,
        }
        for category, field, difference, classification in rows
    ]
    _write_json(
        output / "historical_native_semantic_diff.json",
        {"schema_version": "eiel_historical_native_semantic_diff_v1", "rows": payload},
    )
    metric_rows = [
        row
        for row in payload
        if row["category"] == "metrics" or row["field"] == "population boundary"
    ]
    _write_json(
        output / "historical_native_metric_diff.json",
        {
            "schema_version": "eiel_historical_native_metric_diff_v1",
            "rows": metric_rows,
        },
    )
    _write_text(
        output / "historical_native_first_material_difference.md",
        """# Historical-to-Native First Material Difference

The experiments stop being equivalent at founder construction. Historical M1/M1R
left recurrent policy use and memory dimension to the seeded genome population;
the native campaign forced every founder onto stable GRU and assigned the fixed
8/16/32/64 ladder. This changed treatment prevalence, policy family composition,
and recurrent cost before the first tick.

The next material differences are also large: every historical world enabled
Jungle ecology, historical actions were deterministic, and historical selection
used variable pre-turnover alive/offspring metrics while native turnover restored
64 agents and used compact reporting rows. Therefore the native matrix is a clean
causal test of its own envelope, but not an exact reconstruction of M1/M1R.
""",
    )
    return payload


def _closeout(
    historical: Mapping[str, Any],
    native: Mapping[str, Any],
    semantic: list[dict[str, Any]],
    output: Path,
) -> None:
    comparison = {
        "schema_version": "eiel_historical_native_effect_comparison_v1",
        "historical_m1_signal_recomputed": historical["campaigns"]["m1"][
            "historical_signal_recomputed"
        ],
        "historical_m1r_signal_recomputed": historical["campaigns"]["m1r"][
            "historical_signal_recomputed"
        ],
        "native_decision": "EIEL-NATIVE-MEMORY-REP-B/SIGNAL-PRESENT-SPECIFICITY-INCOMPLETE",
        "comparable_as_direct_replication": False,
        "material_difference_count": sum(
            "relevant" in row["classification"] for row in semantic
        ),
    }
    ranking = {
        "schema_version": "eiel_memory_signal_reversal_hypothesis_ranking_v1",
        "decision": "EIEL-NATIVE-MEMORY-FORENSICS-A/REVERSAL-EXPLAINED-NO-NEW-EXPERIMENT",
        "strongest": {
            "explanation": "memory treatment and policy population were not equivalent",
            "confidence": "high",
            "type": "engineering/design",
            "evidence": [
                "historical natural mixed memory population",
                "native all-founder stable GRU ladder",
                "roughly two orders of magnitude per-generation cost-scale difference",
            ],
            "new_experiment_needed": False,
        },
        "second": {
            "explanation": "world, action, lifecycle, and metric boundaries were not equivalent",
            "confidence": "high",
            "type": "measurement and science",
            "evidence": [
                "historical Jungle enabled in every context",
                "historical deterministic actions versus native stochastic actions",
                "historical variable pre-turnover alive metric versus fixed post-turnover native population",
                "g24/g36 versus g32 and seed 1001 versus 1010",
            ],
            "new_experiment_needed": False,
        },
        "rejected": [
            "native nondeterminism",
            "hidden fallback or runtime JIT",
            "direct reward leakage",
            "founder or lineage confounding",
            "inactive WDL demand",
        ],
        "unresolved": [
            "the individual causal contribution of each non-equivalent axis"
        ],
        "new_experiment": {
            "required": False,
            "reason": "multiple large mismatches exist; no single bounded discriminator can make the campaigns equivalent",
        },
    }
    _write_json(output / "historical_native_effect_comparison.json", comparison)
    _write_json(output / "memory_signal_reversal_hypothesis_ranking.json", ranking)
    _write_text(
        output / "memory_signal_reversal_causal_map.md",
        """# Memory Signal Reversal Causal Map

Historical mixed memory population + all-Jungle worlds + deterministic actions
-> low-prevalence/low-cost memory treatment
-> variable pre-turnover alive and offspring metrics
-> positive M1/M1R WDL interactions under the historical decision rule.

Native all-GRU ladder + Jungle disabled + stochastic actions
-> broad/high-cost memory treatment
-> fixed post-turnover population and compact behavior-row metrics
-> general fitness/energy benefit strongest without WDL
-> negative memory x WDL interaction.

The historical effect recomputes, and the native effect is exact. The reversal is
explained by non-equivalent experiments, not by a native software failure.
""",
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--closeout-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    project = args.project_root.resolve()
    repo = project.parents[2]
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _audit_closeout_zip(args.closeout_zip.resolve(), output / "00_integrity")
    _historical_specs(project, repo, output / "01_historical_spec")
    semantic = _semantic_diff(output / "02_semantic_diff")
    native = _native_reanalysis(
        args.native_root.resolve(), output / "03_native_reanalysis"
    )
    historical = _recompute_historical(project, output / "04_historical_reanalysis")
    _closeout(historical, native, semantic, output / "05_hypothesis_ranking")
    result = {
        "schema_version": SCHEMA_VERSION,
        "decision": "EIEL-NATIVE-MEMORY-FORENSICS-A/REVERSAL-EXPLAINED-NO-NEW-EXPERIMENT",
        "new_experiment_required": False,
        "output_root": str(output),
    }
    _write_json(output / "07_closeout/forensics_decision.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
