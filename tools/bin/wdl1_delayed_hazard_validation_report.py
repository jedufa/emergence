#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping


KEY_METRICS = (
    "alive",
    "successful_births",
    "wdl_delayed_hazard_enable",
    "wdl_delayed_hazard_cue_cell_count",
    "wdl_delayed_hazard_active_cell_count",
    "wdl_delayed_hazard_event_count",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_delayed_hazard_damage_mean",
    "wdl_delayed_hazard_damage_max",
    "wdl_delayed_hazard_agents_damaged",
    "wdl_delayed_hazard_cue_to_active_delay_mean",
    "wdl_delayed_hazard_active_without_prior_cue_count",
    "wdl_cue_seen_agents",
    "wdl_pre_hazard_exit_count",
    "wdl_hazard_enter_count",
    "wdl_hazard_avoidance_proxy",
    "wdl_hazard_exposure_after_cue_count",
    "jungle_predator_damage_total",
    "patch_regrowth_suppressed_total",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        csv.field_size_limit(sys.maxsize)
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _matrix_run_rows(output_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    matrices = sorted(output_root.rglob("matrix.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for matrix_path in matrices:
        for row in _read_csv(matrix_path):
            run_dir = row.get("run_dir") or ""
            if not run_dir or run_dir in seen:
                continue
            if not (Path(run_dir) / "metrics.csv").exists():
                continue
            row["_matrix_path"] = str(matrix_path)
            rows.append(row)
            seen.add(run_dir)
    return rows


def _run_record(metrics_path: Path, *, scenario: str = "", seed: int | None = None, planned_run_id: str = "") -> dict[str, Any]:
    run_dir = metrics_path.parent
    config = _read_json(run_dir / "config.json")
    rows = _read_csv(metrics_path)
    final = rows[-1] if rows else {}
    run_id = planned_run_id or run_dir.name
    if not scenario:
        for part in run_id.split("__"):
            if part.startswith("scenario_"):
                scenario = part.removeprefix("scenario_")
            if part.startswith("seed") and seed is None:
                try:
                    seed = int(part.removeprefix("seed"))
                except Exception:
                    seed = None
    metrics = {key: _num(final.get(key)) for key in KEY_METRICS}
    return {
        "run_id": run_id,
        "runtime_run_id": run_dir.name,
        "run_dir": str(run_dir),
        "scenario": scenario,
        "seed": seed,
        "metrics_rows": len(rows),
        "metrics_hash": _hash_file(metrics_path),
        "config_hash": _hash_file(run_dir / "config.json"),
        "final_metrics": metrics,
        "config_flags": {
            "wdl_delayed_hazard_enable": bool((config.get("env_cfg") or {}).get("wdl_delayed_hazard_enable", False)),
            "patch_world_enable": bool((config.get("env_cfg") or {}).get("patch_world_enable", False)),
            "enable_cognitive_trait_registry": bool((config.get("sim_cfg") or {}).get("enable_cognitive_trait_registry", False)),
            "lineage_founder_balance_enable": bool((config.get("sim_cfg") or {}).get("lineage_founder_balance_enable", False)),
        },
        "checkpoint_bundle_count": len(list((run_dir / "checkpoint_bundles_v2").glob("bundle_gen*"))),
    }


def _runs(output_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    matrix_rows = _matrix_run_rows(output_root)
    if matrix_rows:
        for row in matrix_rows:
            seed = None
            try:
                seed = int(row.get("seed") or "")
            except Exception:
                seed = None
            runs.append(
                _run_record(
                    Path(row["run_dir"]) / "metrics.csv",
                    scenario=row.get("scenario") or "",
                    seed=seed,
                    planned_run_id=row.get("run_id") or "",
                )
            )
        return sorted(runs, key=lambda r: (str(r.get("scenario") or ""), int(r.get("seed") or -1), str(r.get("runtime_run_id") or "")))

    for metrics_path in sorted(output_root.rglob("metrics.csv")):
        runs.append(_run_record(metrics_path))
    return runs


def _mean(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return sum(clean) / float(len(clean)) if clean else None


def _scenario_summary(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scenarios = sorted({str(r.get("scenario") or "") for r in runs})
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        subset = [r for r in runs if r.get("scenario") == scenario]
        row: dict[str, Any] = {"scenario": scenario, "run_count": len(subset)}
        for metric in KEY_METRICS:
            row[f"{metric}_mean"] = _mean([(r.get("final_metrics") or {}).get(metric) for r in subset])
        rows.append(row)
    return rows


def _paired_deltas(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = (
        ("wdl1v_00_tuned_jungle_baseline", "wdl1v_01_tuned_jungle_delayed_hazard"),
        ("wdl1v_02_theme12_clean_pressure_baseline", "wdl1v_03_theme12_clean_pressure_delayed_hazard"),
    )
    by_key = {(r.get("scenario"), r.get("seed")): r for r in runs}
    out: list[dict[str, Any]] = []
    for base, variant in pairs:
        seeds = sorted({s for (sc, s) in by_key if sc in (base, variant)})
        row: dict[str, Any] = {"comparison": f"{variant}_minus_{base}", "paired_seed_count": 0}
        deltas: dict[str, list[float]] = {m: [] for m in KEY_METRICS}
        for seed in seeds:
            b = by_key.get((base, seed))
            v = by_key.get((variant, seed))
            if not b or not v:
                continue
            row["paired_seed_count"] += 1
            for metric in KEY_METRICS:
                bv = (b.get("final_metrics") or {}).get(metric)
                vv = (v.get("final_metrics") or {}).get(metric)
                if bv is not None and vv is not None:
                    deltas[metric].append(float(vv) - float(bv))
        for metric, values in deltas.items():
            row[f"delta_{metric}_mean"] = sum(values) / float(len(values)) if values else None
        out.append(row)
    return out


def _decision(integrity: Mapping[str, Any], consequence: Mapping[str, Any]) -> dict[str, Any]:
    failures = list(integrity.get("failures") or [])
    completed = int(integrity.get("completed_runs") or 0)
    planned = int(integrity.get("planned_runs") or 25)
    damage_total = float(consequence.get("damage_total") or 0.0)
    if failures:
        decision = "WDL-1-D"
        wording = "invalidated; infrastructure, flags, metrics, replay, or run integrity failed"
    elif damage_total <= 0.0:
        decision = "WDL-1-B"
        wording = "mechanism valid but hazard consequence too weak for demand-loop validation"
    elif completed == planned:
        decision = "WDL-1-A"
        wording = "delayed hazard demand loop validated as substrate"
    else:
        decision = "WDL-1-C"
        wording = "reporting/control insufficient; one reporting repair required"
    return {
        "schema_version": "wdl1_decision_v1",
        "decision": decision,
        "decision_wording": wording,
        "recommended_next_step": "WDL-2: Demand Loop Cognition Readiness Bridge Design",
    }


def _report(summary: Mapping[str, Any], decision: Mapping[str, Any], *, base_commit: str, workdir: str, device: str) -> str:
    return f"""# EIEL Phase IV WDL-1 Delayed Hazard Demand Loop Report

## Purpose
Validate a feature-gated delayed hazard / partial observability world-demand loop.

## TDR-1 dependency
WDL-1 follows TDR-1 decision TDR-A: implement one bounded world/action demand loop.

## Working directory and base commit
- Working directory: `{workdir}`
- Base commit: `{base_commit}`
- Device: `{device}`

## Design summary
WDL-1 adds deterministic cue, hidden-gap, active-hazard, and bounded-damage state to Patch World. It is disabled by default.

## Feature flags
See `wdl1_summary.json` and run `config.json` files.

## Observation handling
The cue observation channel is added only when Patch World observation and `wdl_delayed_hazard_observation_enable` are both true.

## Metrics added
Cue cells, active cells, event counts, cue/action context, damage, timing, lineage JSON diagnostics, and integrity checks.

## Checkpoint/replay support
WDL state is stored under checkpoint bundle v2 `patch_world.wdl_delayed_hazard`.

## WDL-1V campaign settings
5 scenarios x 5 seeds, 8 generations, 80 steps/generation, population 64, grid 16.

## Completion table
- Planned runs: `{summary.get('planned_runs')}`
- Completed runs: `{summary.get('completed_runs')}`
- Failed or missing runs: `{summary.get('missing_or_empty_runs')}`

## Hazard timing analysis
See `wdl1_hazard_timing_summary.json`.

## Hazard consequence analysis
See `wdl1_hazard_consequence_summary.json`.

## Demand-loop evidence
See `wdl1_action_context_summary.json`.

## Final WDL-1 decision
{decision.get('decision')}: {decision.get('decision_wording')}

## No-claims statement
WDL-1 implements a feature-gated delayed hazard / partial observability demand loop. It does not show intelligence, emergence, consciousness, perception adaptiveness, memory selection, prediction selection, or general cognition. It does not activate Theme 4 trait campaigns. It does not add cognition rewards or trait rewards. It does not make B1 founder balance the default.
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--report-path", required=True)
    p.add_argument("--evidence-zip", required=True)
    p.add_argument("--base-commit", default="unknown")
    p.add_argument("--workdir", default=os.getcwd())
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    output_root = Path(args.output_root)
    runs = _runs(output_root)
    completed = [r for r in runs if int(r.get("metrics_rows") or 0) > 0]
    failures: list[str] = []
    planned = 25
    if len(completed) != planned:
        failures.append(f"completed_runs={len(completed)} planned_runs={planned}")
    for r in completed:
        flags = r.get("config_flags") or {}
        scenario = str(r.get("scenario") or "")
        wdl_expected = scenario.endswith("delayed_hazard")
        if bool(flags.get("wdl_delayed_hazard_enable")) != wdl_expected:
            failures.append(f"{r.get('run_id')} WDL flag mismatch")
        final = r.get("final_metrics") or {}
        if wdl_expected and (final.get("wdl_delayed_hazard_active_cell_count") or 0.0) <= 0.0:
            failures.append(f"{r.get('run_id')} missing active hazard cells")
        if (final.get("wdl_delayed_hazard_active_without_prior_cue_count") or 0.0) > 0.0:
            failures.append(f"{r.get('run_id')} active hazard without prior cue")
        if bool(flags.get("enable_cognitive_trait_registry")):
            failures.append(f"{r.get('run_id')} cognitive trait registry enabled")
        if bool(flags.get("lineage_founder_balance_enable")):
            failures.append(f"{r.get('run_id')} founder balance enabled")

    scenario_rows = _scenario_summary(completed)
    deltas = _paired_deltas(completed)
    summary = {
        "schema_version": "wdl1_summary_v1",
        "planned_runs": planned,
        "discovered_runs": len(runs),
        "completed_runs": len(completed),
        "missing_or_empty_runs": max(0, planned - len(completed)),
        "runs": completed,
    }
    integrity = {
        "schema_version": "wdl1_integrity_summary_v1",
        "planned_runs": planned,
        "completed_runs": len(completed),
        "failures": failures,
    }
    timing = {
        "schema_version": "wdl1_hazard_timing_summary_v1",
        "cue_to_active_delay_mean": _mean([(r.get("final_metrics") or {}).get("wdl_delayed_hazard_cue_to_active_delay_mean") for r in completed]),
        "active_without_prior_cue_total": sum(float((r.get("final_metrics") or {}).get("wdl_delayed_hazard_active_without_prior_cue_count") or 0.0) for r in completed),
    }
    consequence = {
        "schema_version": "wdl1_hazard_consequence_summary_v1",
        "damage_total": sum(float((r.get("final_metrics") or {}).get("wdl_delayed_hazard_damage_total") or 0.0) for r in completed),
        "agents_damaged_total": sum(float((r.get("final_metrics") or {}).get("wdl_delayed_hazard_agents_damaged") or 0.0) for r in completed),
    }
    action = {
        "schema_version": "wdl1_action_context_summary_v1",
        "cue_seen_agents_total": sum(float((r.get("final_metrics") or {}).get("wdl_cue_seen_agents") or 0.0) for r in completed),
        "pre_hazard_exit_total": sum(float((r.get("final_metrics") or {}).get("wdl_pre_hazard_exit_count") or 0.0) for r in completed),
        "hazard_exposure_after_cue_total": sum(float((r.get("final_metrics") or {}).get("wdl_hazard_exposure_after_cue_count") or 0.0) for r in completed),
    }
    checkpoint = {
        "schema_version": "wdl1_checkpoint_replay_summary_v1",
        "runs_with_checkpoints": sum(1 for r in completed if int(r.get("checkpoint_bundle_count") or 0) > 0),
    }
    decision = _decision(integrity, consequence)

    paths = {
        "summary": output_root / "wdl1_summary.json",
        "integrity": output_root / "wdl1_integrity_summary.json",
        "scenario": output_root / "wdl1_scenario_summary.csv",
        "deltas": output_root / "wdl1_paired_deltas.csv",
        "timing": output_root / "wdl1_hazard_timing_summary.json",
        "consequence": output_root / "wdl1_hazard_consequence_summary.json",
        "action": output_root / "wdl1_action_context_summary.json",
        "checkpoint": output_root / "wdl1_checkpoint_replay_summary.json",
        "decision": output_root / "wdl1_decision.json",
    }
    _write_json(paths["summary"], summary)
    _write_json(paths["integrity"], integrity)
    _write_csv(paths["scenario"], scenario_rows, sorted({k for row in scenario_rows for k in row.keys()}))
    _write_csv(paths["deltas"], deltas, sorted({k for row in deltas for k in row.keys()}))
    _write_json(paths["timing"], timing)
    _write_json(paths["consequence"], consequence)
    _write_json(paths["action"], action)
    _write_json(paths["checkpoint"], checkpoint)
    _write_json(paths["decision"], decision)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summary, decision, base_commit=args.base_commit, workdir=args.workdir, device=args.device), encoding="utf-8")

    evidence_zip = Path(args.evidence_zip)
    evidence_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(evidence_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [report_path, *paths.values()]:
            if path.exists():
                zf.write(path, arcname=path.name)
        for pattern in (
            str(output_root / "wdl1_repeat_parity_summary.json"),
            str(output_root / "e" / "*" / "*" / "matrix.csv"),
            str(output_root / "e" / "*" / "*" / "summary.json"),
            str(Path(args.workdir) / "_spec" / "registry" / "v1" / "wdl1_delayed_hazard_validation.yaml"),
        ):
            for raw in sorted(glob.glob(pattern)):
                path = Path(raw)
                if path.exists():
                    try:
                        arcname = str(path.relative_to(output_root))
                    except ValueError:
                        arcname = str(path.relative_to(Path(args.workdir)))
                    zf.write(path, arcname=arcname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
