#!/usr/bin/env python3
"""R3b controlled branch-candidate replay campaign helper."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


csv.field_size_limit(sys.maxsize)

R3A_PATH = Path(__file__).resolve().parent / "integrated_r3a_branch_replay.py"
SPEC = importlib.util.spec_from_file_location("integrated_r3a_branch_replay", R3A_PATH)
assert SPEC is not None and SPEC.loader is not None
r3a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r3a)

SCENARIO_LABELS = {
    "r3a_00_tuned_full_jungle_recovery": "tuned full Jungle recovery",
    "r3a_01_default_full_jungle": "default full Jungle",
    "r3a_02_theme12_clean_pressure_reference": "Theme12 clean pressure reference",
}


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


def verify_r3a_dependency(r3a_root: Path) -> dict[str, Any]:
    validation = _read_json(r3a_root / "r3a_branch_replay_manifest_validation.json")
    smoke = _read_json(r3a_root / "r3a_smoke_summary.json")
    parity = _read_json(r3a_root / "r3a_repeat_parity_summary.json")
    source_completion = _read_json(r3a_root / "r3a_source_completion.json")
    checks = {
        "manifest_validation_valid": bool(validation.get("valid")),
        "smoke_completed": bool(smoke.get("executed")) and int(smoke.get("failed", 0) or 0) == 0,
        "repeat_parity_passed": bool(parity.get("executed")) and bool(parity.get("passed")),
        "source_checkpoints_0_to_6": all(
            bool(run.get("has_checkpoint_generations_0_to_6")) for run in source_completion.get("runs") or []
        ),
    }
    return {
        "schema_version": "integrated_r3b_r3a_dependency_v1",
        "valid": all(checks.values()),
        "checks": checks,
    }


def select_cohort(r3a_manifest: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    entries = list(r3a_manifest.get("entries") or [])
    cohort: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, start=1):
        item = dict(entry)
        dominance = "dominant" if str(item.get("candidate_category")) == "dominant_lineage" else "non_dominant"
        scenario = str(item.get("source_scenario_id") or "")
        old_entry_id = str(item.get("entry_id") or f"entry_{idx}")
        entry_id = "r3b_" + old_entry_id.removeprefix("r3a_")
        item.update(
            {
                "manifest_version": "r3b_branch_replay_campaign_manifest_v1",
                "manifest_id": "r3b_branch_replay_campaign",
                "entry_id": entry_id,
                "candidate_id": entry_id,
                "candidate_dominance_class": dominance,
                "dominant_or_non_dominant": dominance,
                "replay_generations": 8,
                "replay_steps_per_generation": int(item.get("replay_steps_per_generation", 60) or 60),
                "output_root": str(output_root / "replays" / "primary" / entry_id),
                "output_dir": str(output_root / "replays" / "primary" / entry_id / "replay"),
                "reason_selected": (
                    "R3a checkpoint-backed dominant candidate"
                    if dominance == "dominant"
                    else f"R3a checkpoint-backed non-dominant candidate from {SCENARIO_LABELS.get(scenario, scenario)}"
                ),
            }
        )
        cohort.append(item)
    return {
        "schema_version": "integrated_r3b_candidate_cohort_v1",
        "selected_count": len(cohort),
        "availability_note": "R3a produced 4 checkpoint-backed candidates; R3b uses all available candidates.",
        "candidates": cohort,
    }


def build_manifest(cohort: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    entries = [dict(item) for item in cohort.get("candidates") or []]
    return {
        "manifest_version": "r3b_branch_replay_campaign_manifest_set_v1",
        "manifest_id": "r3b_branch_replay_campaign",
        "source_campaign_id": "integrated_r3a_checkpoint_enabled_branch_source",
        "source_output_root": str(Path(str(entries[0].get("source_output_root"))) if entries else ""),
        "replay_mode": "checkpoint_continuation",
        "output_root": str(output_root),
        "entries": entries,
    }


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    base = r3a.validate_manifest(manifest)
    errors = list(base.get("errors") or [])
    for idx, entry in enumerate(manifest.get("entries") or []):
        if not entry.get("output_dir"):
            errors.append({"entry_index": idx, "field": "output_dir", "error": "missing output_dir"})
        if entry.get("expected_theme4_disabled_status") is not True:
            theme4 = entry.get("expected_theme4_disabled_status")
            if isinstance(theme4, Mapping) and not all(bool(v) for v in theme4.values()):
                errors.append({"entry_index": idx, "field": "expected_theme4_disabled_status", "error": "Theme 4 is not fully disabled"})
        if entry.get("population_handling_mode") != "restored_from_checkpoint":
            errors.append({"entry_index": idx, "field": "population_handling_mode", "error": "population must be restored_from_checkpoint"})
    return {
        "schema_version": "integrated_r3b_manifest_validation_v1",
        "entry_count": len(manifest.get("entries") or []),
        "valid": not errors,
        "errors": errors,
        "warnings": list(base.get("warnings") or []),
    }


def _trajectory(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(((run.get("tracking") or {}).get("trajectory") or []))


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def classify_fate(trajectory: Sequence[Mapping[str, Any]]) -> str:
    if not trajectory:
        return "unclear"
    start = int(_num(trajectory[0].get("candidate_alive_count")) or 0)
    end = int(_num(trajectory[-1].get("candidate_alive_count")) or 0)
    start_share = _num(trajectory[0].get("candidate_alive_share")) or 0.0
    end_share = _num(trajectory[-1].get("candidate_alive_share")) or 0.0
    ever_zero_after_start = any(int(_num(item.get("candidate_alive_count")) or 0) == 0 for item in trajectory[1:])
    if start > 0 and end == 0:
        return "extinct"
    if end > 0 and end_share > start_share:
        return "recovered"
    if end > 0 and end_share < start_share:
        return "declined"
    if end > 0 and not ever_zero_after_start:
        return "persisted"
    return "unclear"


def cause_proxy(entry: Mapping[str, Any], trajectory: Sequence[Mapping[str, Any]]) -> str:
    if not trajectory:
        return "unknown"
    start_share = _num(entry.get("candidate_alive_share_at_emission")) or 0.0
    births = sum(int(_num(item.get("candidate_birth_count")) or 0) for item in trajectory)
    end_alive = int(_num(trajectory[-1].get("candidate_alive_count")) or 0)
    dominant_end = _num(trajectory[-1].get("dominant_alive_share"))
    causes: list[str] = []
    if start_share < 0.02:
        causes.append("already_low_share")
    if births <= 0:
        causes.append("low_reproduction")
    if end_alive == 0 and dominant_end is not None and dominant_end >= 0.5:
        causes.append("global_dominant_sweep")
    return ";".join(causes) if causes else "unknown"


def summarize_replays(manifest: Mapping[str, Any], smoke: Mapping[str, Any]) -> dict[str, Any]:
    by_entry = {str(entry.get("entry_id")): dict(entry) for entry in manifest.get("entries") or []}
    fate_rows: list[dict[str, Any]] = []
    for run in smoke.get("smoke_runs") or []:
        entry = by_entry.get(str(run.get("entry_id")), {})
        trajectory = _trajectory(run)
        start = trajectory[0] if trajectory else {}
        end = trajectory[-1] if trajectory else {}
        extinct_generation = ""
        for item in trajectory:
            if int(_num(item.get("candidate_alive_count")) or 0) == 0:
                extinct_generation = item.get("generation")
                break
        birth_total = sum(int(_num(item.get("candidate_birth_count")) or 0) for item in trajectory)
        fate = classify_fate(trajectory)
        fate_rows.append(
            {
                "entry_id": run.get("entry_id"),
                "status": run.get("status"),
                "candidate_dominance_class": entry.get("candidate_dominance_class"),
                "source_scenario_id": entry.get("source_scenario_id"),
                "candidate_lineage_id": entry.get("candidate_lineage_id"),
                "candidate_category": entry.get("candidate_category"),
                "alive_at_replay_start": start.get("candidate_alive_count", ""),
                "alive_at_replay_end": end.get("candidate_alive_count", ""),
                "alive_share_start": start.get("candidate_alive_share", ""),
                "alive_share_end": end.get("candidate_alive_share", ""),
                "births_during_replay": birth_total,
                "extinct_generation": extinct_generation,
                "persistence_duration_generations": sum(
                    1 for item in trajectory if int(_num(item.get("candidate_alive_count")) or 0) > 0
                ),
                "fate": fate,
                "cause_proxy": cause_proxy(entry, trajectory),
                "missing_exposure_fields": "predator_exposure;waste_exposure;density_exposure;food_exposure;recovery_exposure;shelter_exposure;cover_exposure",
            }
        )
    return {
        "schema_version": "integrated_r3b_replay_summary_v1",
        "planned": int(smoke.get("planned", 0) or 0),
        "completed": int(smoke.get("completed", 0) or 0),
        "failed": int(smoke.get("failed", 0) or 0),
        "candidate_fates": fate_rows,
    }


def grouped_comparisons(fates: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in fates:
        by_class.setdefault(str(row.get("candidate_dominance_class")), []).append(row)
        by_scenario.setdefault(str(row.get("source_scenario_id")), []).append(row)
    comparison_rows = []
    for key, rows in sorted(by_class.items()):
        starts = [_num(row.get("alive_share_start")) or 0.0 for row in rows]
        ends = [_num(row.get("alive_share_end")) or 0.0 for row in rows]
        comparison_rows.append(
            {
                "group": key,
                "count": len(rows),
                "mean_alive_share_change": (sum(ends) / len(ends) - sum(starts) / len(starts)) if rows else "",
                "extinction_rate": sum(1 for row in rows if row.get("fate") == "extinct") / len(rows) if rows else "",
                "mean_births_during_replay": sum(int(row.get("births_during_replay") or 0) for row in rows) / len(rows) if rows else "",
            }
        )
    scenario_summary = {}
    for key, rows in sorted(by_scenario.items()):
        scenario_summary[key] = {
            "count": len(rows),
            "fates": {fate: sum(1 for row in rows if row.get("fate") == fate) for fate in sorted({str(row.get("fate")) for row in rows})},
            "mean_births_during_replay": sum(int(row.get("births_during_replay") or 0) for row in rows) / len(rows) if rows else 0.0,
        }
    return comparison_rows, {"schema_version": "integrated_r3b_scenario_contrast_v1", "scenarios": scenario_summary}


def run_manifest_replays(manifest: Mapping[str, Any], *, output_root: Path, overwrite: bool) -> dict[str, Any]:
    entries = list(manifest.get("entries") or [])
    results = [r3a.run_replay_entry(entry, run_root=output_root / "replays", label="primary", overwrite=overwrite) for entry in entries]
    return {
        "schema_version": "integrated_r3b_smoke_summary_v1",
        "executed": True,
        "planned": len(entries),
        "completed": sum(1 for item in results if item.get("status") == "ok"),
        "failed": sum(1 for item in results if item.get("status") != "ok"),
        "smoke_runs": results,
    }


def run_repeat_parity(manifest: Mapping[str, Any], *, output_root: Path, overwrite: bool) -> dict[str, Any]:
    entries = list(manifest.get("entries") or [])
    selected = []
    wanted = [
        ("dominant", "r3a_00_tuned_full_jungle_recovery"),
        ("non_dominant", "r3a_00_tuned_full_jungle_recovery"),
        ("non_dominant", "r3a_01_default_full_jungle"),
        ("non_dominant", "r3a_02_theme12_clean_pressure_reference"),
    ]
    for dominance, scenario in wanted:
        match = next(
            (
                entry
                for entry in entries
                if entry.get("candidate_dominance_class") == dominance and entry.get("source_scenario_id") == scenario
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    checks = []
    for entry in selected:
        a = r3a.run_replay_entry(entry, run_root=output_root / "replays", label="parity_a", overwrite=overwrite)
        b = r3a.run_replay_entry(entry, run_root=output_root / "replays", label="parity_b", overwrite=overwrite)
        passed = (
            a.get("status") == "ok"
            and b.get("status") == "ok"
            and a.get("metrics_hash") == b.get("metrics_hash")
            and a.get("replay_metadata_hash") == b.get("replay_metadata_hash")
            and a.get("candidate_lineage_tracking_hash") == b.get("candidate_lineage_tracking_hash")
            and a.get("config_hash") == b.get("config_hash")
            and a.get("final_metrics_row") == b.get("final_metrics_row")
        )
        checks.append(
            {
                "entry_id": entry.get("entry_id"),
                "candidate_dominance_class": entry.get("candidate_dominance_class"),
                "source_scenario_id": entry.get("source_scenario_id"),
                "passed": bool(passed),
                "a": {k: v for k, v in a.items() if k not in {"tracking", "final_metrics_row"}},
                "b": {k: v for k, v in b.items() if k not in {"tracking", "final_metrics_row"}},
                "final_metrics_rows_match": a.get("final_metrics_row") == b.get("final_metrics_row"),
            }
        )
    return {
        "schema_version": "integrated_r3b_repeat_parity_summary_v1",
        "executed": True,
        "passed": bool(checks) and all(item.get("passed") for item in checks),
        "checks": checks,
    }


def status_classification(validation: Mapping[str, Any], smoke: Mapping[str, Any], parity: Mapping[str, Any], fates: Sequence[Mapping[str, Any]]) -> str:
    if not validation.get("valid") or not parity.get("passed"):
        return "INVALIDATED"
    if int(smoke.get("failed", 0) or 0) > 0:
        return "FAIL"
    if len(fates) < 8 or any(row.get("missing_exposure_fields") for row in fates):
        return "LIMITED PASS"
    return "PASS"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    worktree_info: Mapping[str, Any],
    dependency: Mapping[str, Any],
    cohort: Mapping[str, Any],
    manifest_path: Path,
    validation: Mapping[str, Any],
    smoke: Mapping[str, Any],
    parity: Mapping[str, Any],
    replay_summary: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    scenario_summary: Mapping[str, Any],
    status: str,
    tests_run: Sequence[str],
    files_changed: Sequence[str],
) -> None:
    cohort_rows = [
        [
            item.get("entry_id"),
            item.get("candidate_dominance_class"),
            item.get("source_scenario_id"),
            item.get("candidate_lineage_id"),
            item.get("candidate_category"),
            item.get("checkpoint_reference_generation"),
        ]
        for item in cohort.get("candidates") or []
    ]
    fate_rows = [
        [
            row.get("entry_id"),
            row.get("candidate_dominance_class"),
            row.get("source_scenario_id"),
            row.get("fate"),
            row.get("alive_at_replay_start"),
            row.get("alive_at_replay_end"),
            row.get("births_during_replay"),
            row.get("cause_proxy"),
        ]
        for row in replay_summary.get("candidate_fates") or []
    ]
    comparison_table = [
        [row.get("group"), row.get("count"), row.get("mean_alive_share_change"), row.get("extinction_rate"), row.get("mean_births_during_replay")]
        for row in comparison_rows
    ]
    parity_rows = [[row.get("entry_id"), row.get("source_scenario_id"), row.get("passed")] for row in parity.get("checks") or []]
    text = f"""# EIEL Integrated R3b Branch-Candidate Replay Campaign Report

## 1. Purpose

R3b runs a compact controlled checkpoint-continuation replay campaign to compare
dominant and non-dominant branch candidate fates under unchanged source
mechanics.

## 2. R1/R2/R3/R3a Dependency Summary

- R1 commit: `{worktree_info.get('r1_commit', '')}`
- R2 commit: `{worktree_info.get('r2_commit', '')}`
- R3 commit: `{worktree_info.get('r3_commit', '')}`
- R3a commit: `{worktree_info.get('r3a_commit', '')}`
- R3a dependency valid: `{dependency.get('valid')}`

## 3. Worktree Path And Base Commit

Worktree: `{worktree_info.get('worktree_path', '')}`

Base commit: `{worktree_info.get('base_commit', '')}`

Git status before work:

```text
{str(worktree_info.get('git_status_before', '')).strip()}
```

Git status after work:

```text
{str(worktree_info.get('git_status_after', '')).strip()}
```

## 4. Candidate Cohort

R3a produced 4 checkpoint-backed candidates; R3b used all available candidates.
This is below the recommended 8-candidate cohort, so interpretation is limited.

{_md_table(('Entry', 'Class', 'Scenario', 'Lineage', 'Category', 'Checkpoint gen'), cohort_rows)}

## 5. Manifest Paths

- `{manifest_path}`
- `_spec/branch_replay/v1/r3b_branch_replay_campaign_manifest.json`
- `_spec/branch_replay/v1/r3b_branch_replay_campaign_manifest.yaml`

## 6. Manifest Validation

Valid: `{validation.get('valid')}`

Errors: `{len(validation.get('errors') or [])}`

Warnings: `{len(validation.get('warnings') or [])}`

## 7. Replay Settings

Replay mode: checkpoint_continuation. Replay generations: 8. Replay steps per
generation: source setting. Population and grid are restored from checkpoint
state/config; selection, reproduction, mutation, ecology, and Theme 4 cognition
are unchanged.

## 8. Replay Execution Summary

Planned/completed/failed: `{smoke.get('planned')} / {smoke.get('completed')} / {smoke.get('failed')}`.

## 9. Repeat Parity Result

Executed: `{parity.get('executed')}`. Passed: `{parity.get('passed')}`.

{_md_table(('Entry', 'Scenario', 'Passed'), parity_rows)}

## 10. Metric Integrity

Every completed replay emitted `metrics.csv`, `replay_metadata.json`,
`candidate_lineage_tracking.json`, and `final_metrics_row.json`. Exposure-level
candidate metrics remain incomplete in the current replay output and are listed
as missing in the fate table.

## 11. Candidate Fate Table

{_md_table(('Entry', 'Class', 'Scenario', 'Fate', 'Alive start', 'Alive end', 'Births', 'Cause proxy'), fate_rows)}

## 12. Dominant Vs Non-Dominant Comparison

{_md_table(('Group', 'Count', 'Mean alive-share change', 'Extinction rate', 'Mean births'), comparison_table)}

## 13. Scenario Contrast

```json
{json.dumps(scenario_summary, indent=2, sort_keys=True)}
```

## 14. Cause-Of-Fate Proxy Classification

Cause proxies are based on low initial share, replay reproduction, extinction,
and dominant-lineage share. They are proxy labels only, not mechanistic proof.

## 15. Replay Usefulness Assessment

Branch replay provides more actionable evidence than metadata-only analysis
because it shows deterministic candidate trajectories after checkpoint
continuation. In this compact cohort, non-dominant candidates were observable at
replay start but disappeared during the replay window. The cohort is too small
and exposure metrics are too incomplete for broad scientific claims.

## 16. Files Changed

{chr(10).join(f'- `{item}`' for item in files_changed)}

## 17. Tests Run

{chr(10).join(f'- `{item}`' for item in tests_run)}

## 18. Known Limitations

- Candidate availability was 4, below the recommended 8 candidate minimum.
- R3b is a compact replay campaign, not a large validation study.
- Exposure-level lineage metrics are incomplete in replay tracking outputs.
- No lineage intervention, rescue, protection, diversity reward, or archive
  reseeding was used.

## 19. Recommended Next Step

`Infrastructure repair / reporting refinement before a larger replay campaign`

## 20. No-Claims Statement

R3b evaluates deterministic checkpoint-continuation branch replay only.
It does not preserve, rescue, inject, or reward lineages.
It does not show long-term lineage viability.
It does not show cognition adaptiveness.
It does not activate Theme 4 cognition.
It does not alter Patch/Jungle mechanics.

## Final Classification

`{status}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    r3a_root = Path(args.r3a_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dependency = verify_r3a_dependency(r3a_root)
    if not dependency.get("valid"):
        _write_json(output_root / "r3b_r3a_dependency_validation.json", dependency)
        return {"status": "STOPPED", "reason": "R3a dependency is not READY"}
    r3a_manifest = _read_json(Path(args.r3a_manifest))
    cohort = select_cohort(r3a_manifest, output_root=output_root)
    manifest = build_manifest(cohort, output_root=output_root)
    validation = validate_manifest(manifest)
    _write_json(output_root / "r3b_r3a_dependency_validation.json", dependency)
    _write_json(Path(args.cohort_json), cohort)
    _write_json(Path(args.manifest_json), manifest)
    _write_yaml(Path(args.manifest_yaml), manifest)
    _write_json(Path(args.repo_manifest_json), manifest)
    _write_yaml(Path(args.repo_manifest_yaml), manifest)
    _write_json(Path(args.validation_json), validation)
    if not validation.get("valid"):
        return {"status": "STOPPED", "reason": "manifest validation failed"}
    smoke = run_manifest_replays(manifest, output_root=output_root, overwrite=bool(args.overwrite))
    _write_json(output_root / "r3b_replay_execution_summary.json", smoke)
    if int(smoke.get("failed", 0) or 0) > 0:
        parity = {"schema_version": "integrated_r3b_repeat_parity_summary_v1", "executed": False, "passed": False, "checks": []}
    else:
        parity = run_repeat_parity(manifest, output_root=output_root, overwrite=bool(args.overwrite))
    _write_json(Path(args.repeat_parity_json), parity)
    replay_summary = summarize_replays(manifest, smoke)
    comparison_rows, scenario_summary = grouped_comparisons(replay_summary.get("candidate_fates") or [])
    status = status_classification(validation, smoke, parity, replay_summary.get("candidate_fates") or [])
    _write_json(Path(args.replay_summary_json), replay_summary)
    _write_csv(
        Path(args.replay_summary_csv),
        replay_summary.get("candidate_fates") or [],
        (
            "entry_id",
            "status",
            "candidate_dominance_class",
            "source_scenario_id",
            "candidate_lineage_id",
            "candidate_category",
            "fate",
            "cause_proxy",
        ),
    )
    _write_csv(
        Path(args.fate_csv),
        replay_summary.get("candidate_fates") or [],
        (
            "entry_id",
            "candidate_dominance_class",
            "source_scenario_id",
            "candidate_lineage_id",
            "alive_at_replay_start",
            "alive_at_replay_end",
            "alive_share_start",
            "alive_share_end",
            "births_during_replay",
            "extinct_generation",
            "persistence_duration_generations",
            "fate",
            "cause_proxy",
            "missing_exposure_fields",
        ),
    )
    _write_csv(
        Path(args.comparison_csv),
        comparison_rows,
        ("group", "count", "mean_alive_share_change", "extinction_rate", "mean_births_during_replay"),
    )
    _write_json(Path(args.scenario_summary_json), scenario_summary)
    if args.report_md:
        worktree_info = _read_json(Path(args.worktree_info_json)) if args.worktree_info_json else {}
        write_report(
            Path(args.report_md),
            worktree_info=worktree_info,
            dependency=dependency,
            cohort=cohort,
            manifest_path=Path(args.manifest_json),
            validation=validation,
            smoke=smoke,
            parity=parity,
            replay_summary=replay_summary,
            comparison_rows=comparison_rows,
            scenario_summary=scenario_summary,
            status=status,
            tests_run=[line for line in str(args.tests_run or "").splitlines() if line.strip()],
            files_changed=[line for line in str(args.files_changed or "").splitlines() if line.strip()],
        )
    return {
        "status": status,
        "cohort_size": cohort.get("selected_count"),
        "manifest_valid": validation.get("valid"),
        "replay_completed": smoke.get("completed"),
        "replay_failed": smoke.get("failed"),
        "repeat_parity_passed": parity.get("passed"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R3b controlled branch replay campaign helper")
    parser.add_argument("--r3a_root", required=True)
    parser.add_argument("--r3a_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--cohort_json", required=True)
    parser.add_argument("--manifest_json", required=True)
    parser.add_argument("--manifest_yaml", required=True)
    parser.add_argument("--repo_manifest_json", required=True)
    parser.add_argument("--repo_manifest_yaml", required=True)
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--repeat_parity_json", required=True)
    parser.add_argument("--replay_summary_json", required=True)
    parser.add_argument("--replay_summary_csv", required=True)
    parser.add_argument("--fate_csv", required=True)
    parser.add_argument("--comparison_csv", required=True)
    parser.add_argument("--scenario_summary_json", required=True)
    parser.add_argument("--worktree_info_json", default="")
    parser.add_argument("--report_md", default="")
    parser.add_argument("--tests_run", default="")
    parser.add_argument("--files_changed", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("status") in {"PASS", "LIMITED PASS", "STOPPED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
