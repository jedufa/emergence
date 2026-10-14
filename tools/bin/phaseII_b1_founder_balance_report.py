#!/usr/bin/env python3
"""Phase II-B1 founder-balance validation helper.

Reporting and replay orchestration only. This helper reuses the existing
LBC-1/R3a checkpoint-continuation replay path and adds B1-specific
legacy-vs-founder-balanced summaries.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


csv.field_size_limit(sys.maxsize)

LBC1_PATH = Path(__file__).resolve().parent / "lbc1_lineage_bottleneck_closure_report.py"
SPEC = importlib.util.spec_from_file_location("lbc1_lineage_bottleneck_closure_report", LBC1_PATH)
assert SPEC is not None and SPEC.loader is not None
lbc1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lbc1)

B1_SCENARIOS: tuple[str, ...] = (
    "b1v_00_tuned_full_jungle_recovery_legacy",
    "b1v_01_tuned_full_jungle_recovery_founder_balanced",
    "b1v_02_default_full_jungle_legacy",
    "b1v_03_default_full_jungle_founder_balanced",
    "b1v_04_theme12_clean_pressure_legacy",
    "b1v_05_theme12_clean_pressure_founder_balanced",
)

SCENARIO_META: dict[str, tuple[str, str]] = {
    "b1v_00_tuned_full_jungle_recovery_legacy": ("tuned_full_jungle_recovery", "legacy"),
    "b1v_01_tuned_full_jungle_recovery_founder_balanced": ("tuned_full_jungle_recovery", "founder_balanced"),
    "b1v_02_default_full_jungle_legacy": ("default_full_jungle", "legacy"),
    "b1v_03_default_full_jungle_founder_balanced": ("default_full_jungle", "founder_balanced"),
    "b1v_04_theme12_clean_pressure_legacy": ("theme12_clean_pressure", "legacy"),
    "b1v_05_theme12_clean_pressure_founder_balanced": ("theme12_clean_pressure", "founder_balanced"),
}

SOURCE_METRICS: tuple[str, ...] = (
    "founder_initial_effective_lineages",
    "founder_initial_top1_share",
    "early_lineage_effective_gen_1",
    "early_lineage_effective_gen_2",
    "lineage_effective_alive_lineages",
    "lineage_top1_alive_share",
    "alive",
    "mean_offspring_total",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_damage_prevented_total",
    "patch_regrowth_suppressed_total",
    "patch_recovery_activity_mean",
    "patch_recovery_added_regrowth_total",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _scenario_runs(source_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for summary_path in sorted(source_root.rglob("summary.json")):
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        for row in summary.get("runs") or []:
            scenario = str((row.get("axes") or {}).get("scenario") or "")
            if scenario in B1_SCENARIOS:
                item = dict(row)
                item["scenario"] = scenario
                item["summary_path"] = str(summary_path)
                runs.append(item)
    runs.sort(key=lambda item: (B1_SCENARIOS.index(str(item["scenario"])), int(item.get("seed", 0) or 0)))
    return runs


def source_summary(source_root: Path) -> dict[str, Any]:
    rows = []
    for run in _scenario_runs(source_root):
        run_dir = Path(str(run.get("run_dir")))
        metrics_path = run_dir / "metrics.csv"
        metrics_rows = _read_csv(metrics_path)
        first = metrics_rows[0] if metrics_rows else {}
        final = metrics_rows[-1] if metrics_rows else {}
        checkpoint_gens = []
        for bundle in sorted((run_dir / "checkpoint_bundles_v2").glob("bundle_gen*")):
            try:
                checkpoint_gens.append(int(bundle.name.replace("bundle_gen", "")))
            except ValueError:
                pass
        ecology, mode = SCENARIO_META[str(run["scenario"])]
        config = _read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        rows.append(
            {
                "scenario": run.get("scenario"),
                "ecology": ecology,
                "mode": mode,
                "seed": run.get("seed"),
                "status": run.get("status"),
                "run_dir": str(run_dir),
                "metrics_exists": metrics_path.exists(),
                "metrics_non_empty": metrics_path.exists() and metrics_path.stat().st_size > 0 and bool(metrics_rows),
                "has_checkpoint_generations_0_to_6": all(gen in checkpoint_gens for gen in range(7)),
                "theme4_disabled": all(
                    not bool((config.get("sim_cfg") or {}).get(field, False))
                    for field in lbc1.r3a.THEME4_DISABLED_FIELDS
                ),
                "founder_balance_enable_config": bool((config.get("sim_cfg") or {}).get("lineage_founder_balance_enable", False)),
                **{f"first_{key}": _num(first.get(key)) for key in SOURCE_METRICS},
                **{f"final_{key}": _num(final.get(key)) for key in SOURCE_METRICS},
            }
        )
    return {
        "schema_version": "phaseII_b1_source_summary_v1",
        "planned": 30,
        "completed": sum(1 for row in rows if row["status"] == "ok"),
        "failed": sum(1 for row in rows if row["status"] != "ok"),
        "runs": rows,
    }


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def source_scenario_summary(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in source.get("runs") or []:
        grouped[(str(row.get("ecology")), str(row.get("mode")))].append(row)
    out = []
    for (ecology, mode), rows in sorted(grouped.items()):
        item: dict[str, Any] = {"ecology": ecology, "mode": mode, "count": len(rows)}
        for metric in SOURCE_METRICS:
            item[f"mean_first_{metric}"] = _mean([_num(row.get(f"first_{metric}")) or 0.0 for row in rows])
            item[f"mean_final_{metric}"] = _mean([_num(row.get(f"final_{metric}")) or 0.0 for row in rows])
        out.append(item)
    return out


def legacy_vs_balanced_summary(scenario_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(row.get("ecology")), str(row.get("mode"))): row for row in scenario_rows}
    comparisons = []
    for ecology in ("tuned_full_jungle_recovery", "default_full_jungle", "theme12_clean_pressure"):
        legacy = by_key.get((ecology, "legacy"), {})
        balanced = by_key.get((ecology, "founder_balanced"), {})
        item: dict[str, Any] = {"ecology": ecology}
        for metric in SOURCE_METRICS:
            lval = _num(legacy.get(f"mean_final_{metric}")) or 0.0
            bval = _num(balanced.get(f"mean_final_{metric}")) or 0.0
            item[f"legacy_final_{metric}"] = lval
            item[f"balanced_final_{metric}"] = bval
            item[f"delta_final_{metric}"] = bval - lval
        item["legacy_first_founder_initial_effective_lineages"] = _num(legacy.get("mean_first_founder_initial_effective_lineages")) or 0.0
        item["balanced_first_founder_initial_effective_lineages"] = _num(balanced.get("mean_first_founder_initial_effective_lineages")) or 0.0
        item["delta_first_founder_initial_effective_lineages"] = item["balanced_first_founder_initial_effective_lineages"] - item["legacy_first_founder_initial_effective_lineages"]
        item["legacy_first_founder_initial_top1_share"] = _num(legacy.get("mean_first_founder_initial_top1_share")) or 0.0
        item["balanced_first_founder_initial_top1_share"] = _num(balanced.get("mean_first_founder_initial_top1_share")) or 0.0
        item["delta_first_founder_initial_top1_share"] = item["balanced_first_founder_initial_top1_share"] - item["legacy_first_founder_initial_top1_share"]
        comparisons.append(item)
    return comparisons


def select_candidate_cohort(source_root: Path, output_root: Path) -> dict[str, Any]:
    old = lbc1.SCENARIOS
    try:
        lbc1.SCENARIOS = B1_SCENARIOS
        cohort = lbc1.select_candidate_cohort(source_root, output_root)
    finally:
        lbc1.SCENARIOS = old
    cohort = dict(cohort)
    cohort["schema_version"] = "phaseII_b1_candidate_cohort_v1"
    cohort["target_candidate_count"] = 24
    for item in cohort.get("candidates") or []:
        scenario = str(item.get("source_scenario_id") or "")
        ecology, mode = SCENARIO_META.get(scenario, ("unknown", "unknown"))
        item["source_ecology"] = ecology
        item["source_mode"] = mode
        item["manifest_id"] = "phaseII_b1_founder_balance_replay"
        item["source_campaign_id"] = "phaseII_b1_founder_balance_validation"
        item["entry_id"] = str(item.get("entry_id", "")).replace("lbc1_", "b1v_", 1)
        item["candidate_id"] = item["entry_id"]
        item["output_root"] = str(output_root / "replays" / "primary" / item["entry_id"])
        item["output_dir"] = str(output_root / "replays" / "primary" / item["entry_id"] / "replay")
    return cohort


def build_manifest(cohort: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    return {
        "manifest_version": "phaseII_b1_branch_replay_manifest_set_v1",
        "manifest_id": "phaseII_b1_founder_balance_replay",
        "source_campaign_id": "phaseII_b1_founder_balance_validation",
        "source_output_root": str(output_root),
        "replay_mode": "checkpoint_continuation",
        "output_root": str(output_root),
        "entries": [dict(item) for item in cohort.get("candidates") or []],
    }


def run_repeat_parity(manifest: Mapping[str, Any], output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    entries = list(manifest.get("entries") or [])
    wanted = [
        ("dominant", "founder_balanced", "tuned_full_jungle_recovery"),
        ("non_dominant", "founder_balanced", "tuned_full_jungle_recovery"),
        ("non_dominant", "founder_balanced", "default_full_jungle"),
        ("non_dominant", "founder_balanced", "theme12_clean_pressure"),
    ]
    selected = []
    for dominance, mode, ecology in wanted:
        match = next(
            (
                entry
                for entry in entries
                if entry.get("candidate_dominance_class") == dominance
                and entry.get("source_mode") == mode
                and entry.get("source_ecology") == ecology
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    checks = []
    for entry in selected:
        a = lbc1.run_replay_entry(entry, run_root=output_root / "replays", label="parity_a", overwrite=overwrite)
        b = lbc1.run_replay_entry(entry, run_root=output_root / "replays", label="parity_b", overwrite=overwrite)
        passed = (
            a.get("status") == "ok"
            and b.get("status") == "ok"
            and a.get("metrics_hash") == b.get("metrics_hash")
            and a.get("candidate_lineage_tracking_hash") == b.get("candidate_lineage_tracking_hash")
            and a.get("dominant_tracking_hash") == b.get("dominant_tracking_hash")
            and a.get("config_hash") == b.get("config_hash")
            and a.get("final_metrics_row") == b.get("final_metrics_row")
        )
        checks.append(
            {
                "entry_id": entry.get("entry_id"),
                "source_ecology": entry.get("source_ecology"),
                "source_mode": entry.get("source_mode"),
                "candidate_dominance_class": entry.get("candidate_dominance_class"),
                "passed": bool(passed),
            }
        )
    return {
        "schema_version": "phaseII_b1_repeat_parity_summary_v1",
        "executed": True,
        "passed": bool(checks) and all(item.get("passed") for item in checks),
        "checks": checks,
    }


def replay_mode_comparison(fates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fates:
        grouped[(str(row.get("source_scenario_id")), str(row.get("candidate_dominance_class")), str(row.get("source_mode", "")))].append(row)
    # Enrich rows with source_mode/source_ecology from scenario id when summarize_replays did not include them.
    normalized = []
    for row in fates:
        item = dict(row)
        ecology, mode = SCENARIO_META.get(str(item.get("source_scenario_id")), ("unknown", "unknown"))
        item["source_ecology"] = ecology
        item["source_mode"] = mode
        normalized.append(item)
    by_group: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_group[(str(row["source_ecology"]), str(row["source_mode"]), str(row.get("candidate_dominance_class")))].append(row)
    out = []
    for (ecology, mode, dominance), rows in sorted(by_group.items()):
        starts = [_num(row.get("alive_share_start")) or 0.0 for row in rows]
        ends = [_num(row.get("alive_share_end")) or 0.0 for row in rows]
        out.append(
            {
                "ecology": ecology,
                "mode": mode,
                "candidate_dominance_class": dominance,
                "count": len(rows),
                "mean_alive_share_change": (_mean(ends) - _mean(starts)) if rows else 0.0,
                "extinction_or_sweep_rate": sum(1 for row in rows if row.get("fate") in {"extinct_after_start", "dominant_sweep"}) / len(rows) if rows else 0.0,
                "persisted_or_recovered_rate": sum(1 for row in rows if row.get("fate") in {"persisted", "recovered"}) / len(rows) if rows else 0.0,
                "mean_births_during_replay": _mean([float(row.get("births_during_replay") or 0) for row in rows]),
            }
        )
    return out


def decide_b1(
    *,
    source: Mapping[str, Any],
    validation: Mapping[str, Any],
    parity: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    legacy_balanced: Sequence[Mapping[str, Any]],
    replay_comparison: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fates = list(replay_summary.get("candidate_fates") or [])
    for row in fates:
        ecology, mode = SCENARIO_META.get(str(row.get("source_scenario_id")), ("unknown", "unknown"))
        row["source_ecology"] = ecology
        row["source_mode"] = mode
    invalid = (
        source.get("failed", 0) != 0
        or not validation.get("valid")
        or not parity.get("passed")
        or replay_summary.get("failed", 0) != 0
    )
    balanced_runs = [row for row in source.get("runs") or [] if row.get("mode") == "founder_balanced"]
    legacy_runs = [row for row in source.get("runs") or [] if row.get("mode") == "legacy"]
    balanced_initial_ok = bool(balanced_runs) and all((_num(row.get("first_founder_initial_effective_lineages")) or 0.0) >= 7.5 for row in balanced_runs)
    legacy_noop_ok = bool(legacy_runs) and all(not bool(row.get("founder_balance_enable_config")) for row in legacy_runs)
    viability_ok = all((_num(row.get("final_alive")) or 0.0) > 0.0 for row in source.get("runs") or [])
    pressure_ok = any((_num(row.get("final_jungle_predator_events")) or 0.0) > 0.0 for row in source.get("runs") or []) and any((_num(row.get("final_patch_regrowth_suppressed_total")) or 0.0) > 0.0 for row in source.get("runs") or [])
    non_dom_balanced = [row for row in fates if row.get("candidate_dominance_class") == "non_dominant" and row.get("source_mode") == "founder_balanced"]
    non_dom_legacy = [row for row in fates if row.get("candidate_dominance_class") == "non_dominant" and row.get("source_mode") == "legacy"]
    balanced_improved_fates = sum(1 for row in non_dom_balanced if row.get("fate") in {"persisted", "recovered", "declined"})
    legacy_improved_fates = sum(1 for row in non_dom_legacy if row.get("fate") in {"persisted", "recovered", "declined"})
    balanced_births = _mean([float(row.get("births_during_replay") or 0) for row in non_dom_balanced])
    legacy_births = _mean([float(row.get("births_during_replay") or 0) for row in non_dom_legacy])
    early_delta = _mean([float(row.get("delta_final_early_lineage_effective_gen_2") or 0.0) for row in legacy_balanced])
    if invalid:
        classification = "B1-INVALIDATED"
    elif not balanced_initial_ok or not legacy_noop_ok or not viability_ok or not pressure_ok:
        classification = "B1-FAIL"
    elif early_delta > 0.25 and (balanced_improved_fates > legacy_improved_fates or balanced_births > legacy_births):
        classification = "B1-PASS"
    elif early_delta > 0.25 or balanced_improved_fates > legacy_improved_fates or balanced_births > legacy_births:
        classification = "B1-LIMITED"
    else:
        classification = "B1-FAIL"
    next_step = {
        "B1-PASS": "Revalidate Phase II closure with B1 available, then consider CRB-1 with caveats.",
        "B1-LIMITED": "Decide whether to accept collapse as a V0 limitation with B1 available, or keep B1 disabled and proceed with documented limitation.",
        "B1-FAIL": "Do not tune additional fixes in this task; reconsider whether to accept the limitation or stop before CRB-1.",
        "B1-INVALIDATED": "Repair the invalidating infrastructure or flag mismatch before interpreting B1.",
    }[classification]
    return {
        "schema_version": "phaseII_b1_decision_v1",
        "classification": classification,
        "next_step": next_step,
        "evidence": {
            "balanced_initial_ok": balanced_initial_ok,
            "legacy_noop_ok": legacy_noop_ok,
            "viability_ok": viability_ok,
            "pressure_ok": pressure_ok,
            "early_lineage_effective_gen2_delta_mean": early_delta,
            "non_dominant_balanced_interpretable_count": len(non_dom_balanced),
            "non_dominant_legacy_interpretable_count": len(non_dom_legacy),
            "non_dominant_balanced_persisted_recovered_or_declined": balanced_improved_fates,
            "non_dominant_legacy_persisted_recovered_or_declined": legacy_improved_fates,
            "non_dominant_balanced_mean_births": balanced_births,
            "non_dominant_legacy_mean_births": legacy_births,
        },
        "replay_comparison": replay_comparison,
    }


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    work_info: Mapping[str, Any],
    source: Mapping[str, Any],
    scenario_rows: Sequence[Mapping[str, Any]],
    legacy_balanced: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    parity: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    replay_comparison: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    tests_run: Sequence[str],
    files_changed: Sequence[str],
    evidence_path: str,
) -> None:
    scenario_table = [
        [
            row.get("ecology"),
            row.get("mode"),
            row.get("count"),
            row.get("mean_first_founder_initial_effective_lineages"),
            row.get("mean_first_founder_initial_top1_share"),
            row.get("mean_final_early_lineage_effective_gen_2"),
            row.get("mean_final_alive"),
            row.get("mean_final_jungle_predator_events"),
            row.get("mean_final_patch_regrowth_suppressed_total"),
        ]
        for row in scenario_rows
    ]
    comparison_table = [
        [
            row.get("ecology"),
            row.get("delta_first_founder_initial_effective_lineages"),
            row.get("delta_first_founder_initial_top1_share"),
            row.get("delta_final_early_lineage_effective_gen_2"),
            row.get("delta_final_alive"),
            row.get("delta_final_jungle_predator_events"),
            row.get("delta_final_patch_regrowth_suppressed_total"),
        ]
        for row in legacy_balanced
    ]
    fate_table = [
        [
            row.get("entry_id"),
            row.get("source_scenario_id"),
            row.get("candidate_dominance_class"),
            row.get("fate"),
            row.get("alive_at_replay_start"),
            row.get("alive_at_replay_end"),
            row.get("births_during_replay"),
            row.get("cause_proxy"),
        ]
        for row in replay_summary.get("candidate_fates") or []
    ]
    replay_comp_table = [
        [
            row.get("ecology"),
            row.get("mode"),
            row.get("candidate_dominance_class"),
            row.get("count"),
            row.get("mean_alive_share_change"),
            row.get("extinction_or_sweep_rate"),
            row.get("persisted_or_recovered_rate"),
            row.get("mean_births_during_replay"),
        ]
        for row in replay_comparison
    ]
    text = f"""# EIEL Phase II-B1 Founder / Lineage Balance Fix Report

## 1. Purpose

Phase II-B1 implements and validates the single selected Phase II fix:
feature-gated initial founder / lineage balance.

## 2. Phase II Decision B Context

The LBC-1 rerun selected Decision B: one targeted ecology/lineage fix. The
selected fix was B1, initial founder / lineage balance.

## 3. Working Directory And Base Commit

Working directory: `{work_info.get('working_directory')}`

Base commit: `{work_info.get('base_commit')}`

Git status before work:

```text
{str(work_info.get('git_status_before', '')).strip()}
```

Git status after work:

```text
{str(work_info.get('git_status_after', '')).strip()}
```

Device used: `{work_info.get('device_used')}`

## 4. Fix Design

B1 adds an initialization-only founder-balance mode. When enabled, the initial
population is assigned to deterministic biological founder groups after initial
agent spawning. The assignment is sorted by agent id and round-robin across
founder groups. No lineage quotas or protections are applied after
initialization.

## 5. Feature Flags

- `lineage_founder_balance_enable`
- `lineage_founder_group_count`
- `lineage_founder_min_agents_per_group`
- `lineage_founder_assignment_mode`
- `lineage_founder_balance_schema_version`

## 6. Default No-Op Behavior

The feature is disabled by default. Disabled mode preserves one founder lineage
per initial agent and does not require new config.

## 7. Founder Balance Initialization Behavior

B1 validation used 8 founder groups with 8 initial agents per group for a
population of 64.

## 8. Metrics Added

Founder metrics and early lineage generation snapshots were added to
`metrics.csv`. They are reporting-only and are not used for selection.

## 9. Tests Added

Focused tests cover disabled no-op behavior, deterministic balanced assignment,
invalid config rejection, checkpoint-style lineage preservation, metrics
headers/values, and no intervention flags.

## 10. Preflight Results

Preflight and verification commands are listed in the tests section below.

## 11. Registry And Campaign Settings

Registry: `_spec/registry/v1/phaseII_b1_founder_balance_validation.yaml`

Settings: 6 scenarios, 5 paired seeds, 8 generations, 60 steps/generation,
population 64, grid 16, CUDA device/bank device.

## 12. Repeat Parity

Repeat parity passed: `{parity.get('passed')}`.

## 13. Source Campaign Completion

Source runs planned/completed/failed: `{source.get('planned')} / {source.get('completed')} / {source.get('failed')}`.

## 14. Checkpoint Availability

All completed source runs were checked for `checkpoint_bundles_v2` generations
0 through 6.

## 15. Replay Validation

Replay runs planned/completed/failed: `{replay_summary.get('planned')} / {replay_summary.get('completed')} / {replay_summary.get('failed')}`.

## 16. Legacy Vs Founder-Balanced Comparison

{_md_table(('Ecology', 'Mode', 'N', 'Founder eff initial', 'Founder top1 initial', 'Early eff gen2', 'Alive', 'Pred events', 'Regrowth suppressed'), scenario_table)}

{_md_table(('Ecology', 'Delta founder eff initial', 'Delta top1 initial', 'Delta early eff gen2', 'Delta alive', 'Delta pred events', 'Delta regrowth suppressed'), comparison_table)}

## 17. Effect On Early Lineage Collapse

The B1 decision JSON records the mean legacy-vs-balanced delta for early
generation effective lineages.

## 18. Effect On Non-Dominant Replay Fate

{_md_table(('Entry', 'Scenario', 'Dominance', 'Fate', 'Alive start', 'Alive end', 'Births', 'Proxy'), fate_table)}

{_md_table(('Ecology', 'Mode', 'Dominance', 'N', 'Mean share change', 'Ext/sweep rate', 'Persist/recover rate', 'Mean births'), replay_comp_table)}

## 19. Effect On Population Viability

Population viability is summarized by final alive counts in the source tables.

## 20. Effect On Patch/Jungle Pressure Metrics

Predator events, predator damage, prevented damage, regrowth suppression, and
recovery metrics were retained in source summaries to check that pressure was
not erased.

## 21. Final B1-V Classification

`{decision.get('classification')}`

## 22. Recommended Next Step

{decision.get('next_step')}

## 23. Files Changed

{chr(10).join(f'- `{item}`' for item in files_changed)}

## 24. Evidence Artifact Path

`{evidence_path}`

## 25. Tests Run

{chr(10).join(f'- `{item}`' for item in tests_run)}

## 26. No-Claims Statement

Phase II-B1 implements and validates a single feature-gated founder/lineage initialization balance fix.
It does not add lineage protection, diversity rewards, archive reseeding, or ongoing lineage intervention.
It does not activate Theme 4 cognition.
It does not prove cognition adaptiveness, intelligence, emergence, or solved lineage diversity.
It only tests whether initial founder balance reduces the early lineage bottleneck enough to support Phase II closure.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source = source_summary(source_root)
    scenario_rows = source_scenario_summary(source)
    legacy_balanced = legacy_vs_balanced_summary(scenario_rows)
    cohort = select_candidate_cohort(source_root, output_root)
    manifest = build_manifest(cohort, output_root)
    validation = lbc1.validate_manifest(manifest)
    _write_json(Path(args.source_summary_json), source)
    _write_csv(output_root / "b1v_source_scenario_summary.csv", scenario_rows, list(scenario_rows[0].keys()) if scenario_rows else ("ecology", "mode"))
    _write_json(Path(args.cohort_json), cohort)
    _write_json(Path(args.manifest_json), manifest)
    _write_yaml(Path(args.manifest_yaml), manifest)
    _write_json(Path(args.validation_json), validation)
    if not validation.get("valid"):
        execution = {"schema_version": "phaseII_b1_replay_execution_v1", "planned": 0, "completed": 0, "failed": 0, "replay_runs": []}
        parity = {"schema_version": "phaseII_b1_repeat_parity_summary_v1", "executed": False, "passed": False, "checks": []}
    else:
        execution = lbc1.run_replays(manifest, output_root, overwrite=bool(args.overwrite))
        parity = run_repeat_parity(manifest, output_root, overwrite=bool(args.overwrite)) if execution.get("failed") == 0 else {
            "schema_version": "phaseII_b1_repeat_parity_summary_v1",
            "executed": False,
            "passed": False,
            "checks": [],
        }
    _write_json(output_root / "b1v_replay_execution_summary.json", execution)
    replay_summary = lbc1.summarize_replays(manifest, execution)
    for row in replay_summary.get("candidate_fates") or []:
        ecology, mode = SCENARIO_META.get(str(row.get("source_scenario_id")), ("unknown", "unknown"))
        row["source_ecology"] = ecology
        row["source_mode"] = mode
    replay_comparison = replay_mode_comparison(replay_summary.get("candidate_fates") or [])
    decision = decide_b1(
        source=source,
        validation=validation,
        parity=parity,
        replay_summary=replay_summary,
        legacy_balanced=legacy_balanced,
        replay_comparison=replay_comparison,
    )
    _write_json(Path(args.repeat_parity_json), parity)
    _write_json(Path(args.replay_summary_json), replay_summary)
    _write_csv(
        Path(args.replay_summary_csv),
        replay_summary.get("candidate_fates") or [],
        (
            "entry_id",
            "status",
            "candidate_dominance_class",
            "candidate_class",
            "source_scenario_id",
            "source_ecology",
            "source_mode",
            "source_seed",
            "candidate_lineage_id",
            "candidate_category",
            "fate",
            "cause_proxy",
        ),
    )
    fate_fields = [
        "entry_id",
        "candidate_dominance_class",
        "candidate_class",
        "source_scenario_id",
        "source_ecology",
        "source_mode",
        "source_seed",
        "candidate_lineage_id",
        "alive_at_replay_start",
        "alive_at_replay_end",
        "alive_share_start",
        "alive_share_end",
        "births_during_replay",
        "birth_share_end",
        "extinct_generation",
        "persistence_duration_generations",
        "fate",
        "cause_proxy",
    ] + [f"{field}_mean" for field in lbc1.EXPOSURE_FIELDS]
    _write_csv(Path(args.fate_csv), replay_summary.get("candidate_fates") or [], fate_fields)
    _write_csv(
        Path(args.legacy_vs_balanced_csv),
        legacy_balanced,
        list(legacy_balanced[0].keys()) if legacy_balanced else ("ecology",),
    )
    _write_csv(
        Path(args.replay_comparison_csv),
        replay_comparison,
        ("ecology", "mode", "candidate_dominance_class", "count", "mean_alive_share_change", "extinction_or_sweep_rate", "persisted_or_recovered_rate", "mean_births_during_replay"),
    )
    _write_json(Path(args.decision_json), decision)
    if args.report_md:
        work_info = _read_json(Path(args.work_info_json)) if args.work_info_json else {}
        write_report(
            Path(args.report_md),
            work_info=work_info,
            source=source,
            scenario_rows=scenario_rows,
            legacy_balanced=legacy_balanced,
            validation=validation,
            parity=parity,
            replay_summary=replay_summary,
            replay_comparison=replay_comparison,
            decision=decision,
            tests_run=[line for line in str(args.tests_run or "").splitlines() if line.strip()],
            files_changed=[line for line in str(args.files_changed or "").splitlines() if line.strip()],
            evidence_path=str(args.evidence_path or ""),
        )
    return {
        "classification": decision.get("classification"),
        "source_completed": source.get("completed"),
        "candidate_count": cohort.get("selected_count"),
        "manifest_valid": validation.get("valid"),
        "replay_completed": replay_summary.get("completed"),
        "repeat_parity_passed": parity.get("passed"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase II-B1 founder-balance validation report helper")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_summary_json", required=True)
    parser.add_argument("--cohort_json", required=True)
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--manifest_yaml", required=True)
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--repeat_parity_json", required=True)
    parser.add_argument("--replay_summary_json", required=True)
    parser.add_argument("--replay_summary_csv", required=True)
    parser.add_argument("--fate_csv", required=True)
    parser.add_argument("--legacy_vs_balanced_csv", required=True)
    parser.add_argument("--replay_comparison_csv", required=True)
    parser.add_argument("--decision_json", required=True)
    parser.add_argument("--work_info_json", default="")
    parser.add_argument("--report_md", default="")
    parser.add_argument("--tests_run", default="")
    parser.add_argument("--files_changed", default="")
    parser.add_argument("--evidence_path", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
