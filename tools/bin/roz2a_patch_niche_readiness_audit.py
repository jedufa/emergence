#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "roz2a_patch_niche_readiness_audit_v1"
RUN_ID_RE = re.compile(
    r"__arm_(?P<arm>.*?)__context_(?P<context>.*?)__init_(?P<init_arm>.*?)__seed(?P<seed>\d+)"
)
NO_CLAIMS = (
    "ROZ-2A is a bounded mechanics-readiness package for a default-off, lineage-blind static "
    "Patch World niche overlay. It does not show intelligence, emergence, consciousness, broad "
    "cognition, adaptive memory, prediction selection, solved lineage diversity, beneficial open-ended "
    "search, or general open-ended evolution."
)
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_branch4() -> Any:
    path = _project_root() / "tools/bin/branch4_niche_strategy_audit.py"
    spec = importlib.util.spec_from_file_location("branch4_niche_strategy_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _run_label(run_dir: Path) -> str:
    meta = _run_metadata(run_dir)
    if meta.get("run_id"):
        return str(meta["run_id"])
    manifest = _read_json(run_dir / "manifest.json", {})
    if isinstance(manifest, Mapping) and manifest.get("run_id"):
        return str(manifest["run_id"])
    return run_dir.name


def _campaign_index_candidates(run_dir: Path) -> Iterable[Path]:
    for parent in [run_dir, *run_dir.parents]:
        for name in ("parallel_campaign_summary.json", "campaign_index.json", "campaign_manifest.json"):
            path = parent / name
            if path.exists():
                yield path


def _run_metadata(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json", {})
    out: dict[str, Any] = {}
    if isinstance(manifest, Mapping):
        out.update({k: v for k, v in manifest.items() if k in {"run_id", "seed", "run_tag"}})
        axes = manifest.get("axes")
        if isinstance(axes, Mapping):
            out.update({k: v for k, v in axes.items() if k in {"arm", "context", "init_arm"}})
    for path in _campaign_index_candidates(run_dir):
        payload = _read_json(path, {})
        rows = []
        if isinstance(payload, Mapping):
            rows.extend(payload.get("results") or [])
            rows.extend(payload.get("selected_cases") or [])
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("run_dir") or "") and str(row.get("run_dir")) != str(run_dir):
                continue
            if not str(row.get("run_id") or ""):
                continue
            out.setdefault("run_id", str(row.get("run_id")))
            axes = row.get("axes")
            if isinstance(axes, Mapping):
                out.update({k: v for k, v in axes.items() if k in {"arm", "context", "init_arm"}})
            out.setdefault("seed", row.get("seed", ""))
    run_id = str(out.get("run_id") or "")
    match = RUN_ID_RE.search(run_id)
    if match:
        out.update({k: v for k, v in match.groupdict().items() if v})
    return out


def _final_metrics_row(run_dir: Path) -> dict[str, str]:
    rows = _read_csv_rows(run_dir / "metrics.csv")
    return rows[-1] if rows else {}


def _niche_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "branch_observability" / "branch_observability_niche_lifecycle.jsonl"
    zone_counter: Counter[str] = Counter()
    dominant_counter: Counter[str] = Counter()
    rows = 0
    total_ticks = 0
    shelter_ticks = 0
    food_collected = 0.0
    damage = 0.0
    if not path.exists():
        return {
            "niche_rows": 0,
            "zone_classes": 0,
            "dominant_zone_classes": 0,
            "total_observed_ticks": 0,
            "shelter_cover_fraction": 0.0,
            "food_collected": 0.0,
            "damage": 0.0,
            "top_zone_classes": {},
            "top_dominant_zone_classes": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            total_ticks += _safe_int(row.get("total_observed_ticks"), 0)
            shelter_ticks += _safe_int(row.get("shelter_cover_exposure_ticks"), 0)
            for key, value in dict(row.get("zone_exposure_counts") or {}).items():
                zone_counter[str(key)] += _safe_int(value, 0)
            dominant = str(row.get("dominant_zone_class") or "")
            if dominant:
                dominant_counter[dominant] += 1
            for value in dict(row.get("food_collected_by_zone") or {}).values():
                food_collected += _safe_float(value, 0.0)
            for value in dict(row.get("damage_by_zone") or {}).values():
                damage += _safe_float(value, 0.0)
    return {
        "niche_rows": rows,
        "zone_classes": len(zone_counter),
        "dominant_zone_classes": len(dominant_counter),
        "total_observed_ticks": total_ticks,
        "shelter_cover_fraction": round(float(shelter_ticks) / max(1.0, float(total_ticks)), 6),
        "food_collected": round(food_collected, 6),
        "damage": round(damage, 6),
        "top_zone_classes": dict(zone_counter.most_common(8)),
        "top_dominant_zone_classes": dict(dominant_counter.most_common(8)),
    }


def _run_row(run_dir: Path) -> dict[str, Any]:
    meta = _run_metadata(run_dir)
    run_id = str(meta.get("run_id") or _run_label(run_dir))
    match = RUN_ID_RE.search(run_id)
    axes = match.groupdict() if match else {}
    axes.update({k: str(v) for k, v in meta.items() if k in {"arm", "context", "init_arm", "seed"} and str(v)})
    funnel = _read_json(run_dir / "metrics" / "lineage_reproduction_funnel_run_summary.json", {})
    branch_summary = _read_json(run_dir / "branch_observability" / "branch_observability_run_summary.json", {})
    metrics = _final_metrics_row(run_dir)
    niche = _niche_summary(run_dir)
    path2_event_rows = _safe_int((funnel or {}).get("path2_roz_event_rows"), 0)
    path2_active_rows = _safe_int((funnel or {}).get("path2_roz_window_active_rows"), 0)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "arm": str(axes.get("arm", "")),
        "context": str(axes.get("context", "")),
        "init_arm": str(axes.get("init_arm", "")),
        "seed": str(axes.get("seed", "")),
        "path2_roz_event_rows": path2_event_rows,
        "path2_roz_window_active_rows": path2_active_rows,
        "path2_roz_parent_in_zone_rows": _safe_int((funnel or {}).get("path2_roz_parent_in_zone_rows"), 0),
        "path2_roz_selected_parent_scan_in_zone_rows": _safe_int(
            (funnel or {}).get("path2_roz_selected_parent_scan_in_zone_rows"), 0
        ),
        "branch_niche_rows": _safe_int((branch_summary or {}).get("niche_lifecycle_rows"), 0),
        "final_population": _safe_float(metrics.get("population_alive", metrics.get("alive_count", 0.0)), 0.0),
        "patch_food_amount_mean": _safe_float(metrics.get("patch_food_amount_mean"), 0.0),
        "patch_cover_mean": _safe_float(metrics.get("patch_cover_mean"), 0.0),
        "patch_shelter_mean": _safe_float(metrics.get("patch_shelter_mean"), 0.0),
        "patch_movement_cost_mean": _safe_float(metrics.get("patch_movement_cost_mean"), 0.0),
        "patch_waste_mean": _safe_float(metrics.get("patch_waste_mean"), 0.0),
        "patch_recovery_activity_mean": _safe_float(metrics.get("patch_recovery_activity_mean"), 0.0),
        **niche,
    }


def _aggregate(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("arm", "")), str(row.get("context", "")))].append(row)
    out = []
    for (arm, context), items in sorted(grouped.items()):
        n = max(1, len(items))
        out.append(
            {
                "arm": arm,
                "context": context,
                "runs": len(items),
                "path2_roz_event_rows": sum(_safe_int(row.get("path2_roz_event_rows"), 0) for row in items),
                "path2_roz_window_active_rows": sum(_safe_int(row.get("path2_roz_window_active_rows"), 0) for row in items),
                "branch_niche_rows": sum(_safe_int(row.get("branch_niche_rows"), 0) for row in items),
                "avg_zone_classes": round(sum(_safe_float(row.get("zone_classes"), 0.0) for row in items) / n, 6),
                "avg_dominant_zone_classes": round(
                    sum(_safe_float(row.get("dominant_zone_classes"), 0.0) for row in items) / n,
                    6,
                ),
                "avg_shelter_cover_fraction": round(
                    sum(_safe_float(row.get("shelter_cover_fraction"), 0.0) for row in items) / n,
                    6,
                ),
                "avg_patch_cover_mean": round(sum(_safe_float(row.get("patch_cover_mean"), 0.0) for row in items) / n, 6),
                "avg_patch_shelter_mean": round(sum(_safe_float(row.get("patch_shelter_mean"), 0.0) for row in items) / n, 6),
                "avg_patch_movement_cost_mean": round(
                    sum(_safe_float(row.get("patch_movement_cost_mean"), 0.0) for row in items) / n,
                    6,
                ),
                "avg_patch_waste_mean": round(sum(_safe_float(row.get("patch_waste_mean"), 0.0) for row in items) / n, 6),
                "avg_food_collected": round(sum(_safe_float(row.get("food_collected"), 0.0) for row in items) / n, 6),
                "avg_damage": round(sum(_safe_float(row.get("damage"), 0.0) for row in items) / n, 6),
            }
        )
    return out


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def analyze(runs_root: Path, output_dir: Path, *, expected_runs: int = 54, repeat_runs_root: Path | None = None) -> dict[str, Any]:
    branch4 = _load_branch4()
    run_dirs = branch4.discover_run_dirs([runs_root])
    rows = [_run_row(path) for path in run_dirs]
    aggregate = _aggregate(rows)
    branch4_result = branch4.analyze_runs(run_dirs, profile="nominal")
    branch4_paths = branch4.write_outputs(branch4_result, output_dir)

    baseline_rows = [row for row in rows if row.get("arm") == "baseline"]
    reporting_rows = [row for row in rows if row.get("arm") == "reporting_only"]
    active_rows = [row for row in rows if row.get("arm") == "active_roz2a"]
    baseline_path2_leakage = sum(_safe_int(row.get("path2_roz_event_rows"), 0) for row in baseline_rows)
    active_context = [row for row in active_rows if row.get("context") == "roz2a_niche_context"]
    reporting_context = [row for row in reporting_rows if row.get("context") == "roz2a_niche_context"]
    active_avg_zone_classes = sum(_safe_float(row.get("zone_classes"), 0.0) for row in active_context) / max(1, len(active_context))
    reporting_avg_zone_classes = sum(_safe_float(row.get("zone_classes"), 0.0) for row in reporting_context) / max(1, len(reporting_context))
    active_avg_shelter = sum(_safe_float(row.get("shelter_cover_fraction"), 0.0) for row in active_context) / max(1, len(active_context))
    reporting_avg_shelter = sum(_safe_float(row.get("shelter_cover_fraction"), 0.0) for row in reporting_context) / max(1, len(reporting_context))
    active_static_separation = bool(
        active_context
        and reporting_context
        and active_avg_zone_classes > reporting_avg_zone_classes
        and active_avg_shelter > reporting_avg_shelter
    )

    repeat_parity = {"checked": False, "passed": False, "reason": "repeat_runs_root_not_supplied"}
    if repeat_runs_root is not None:
        repeat_dirs = branch4.discover_run_dirs([repeat_runs_root])
        repeat_rows = [_run_row(path) for path in repeat_dirs]
        main_by_run_id = {str(row.get("run_id", "")): dict(row) for row in rows if str(row.get("run_id", ""))}
        mismatches = []
        missing_main_rows = []
        keys = [
            "path2_roz_event_rows",
            "path2_roz_window_active_rows",
            "path2_roz_parent_in_zone_rows",
            "path2_roz_selected_parent_scan_in_zone_rows",
            "branch_niche_rows",
            "zone_classes",
            "dominant_zone_classes",
            "food_collected",
            "damage",
            "final_population",
        ]
        for row in repeat_rows:
            run_id = str(row.get("run_id", ""))
            main = main_by_run_id.get(run_id)
            if main is None:
                missing_main_rows.append(run_id)
                continue
            for key in keys:
                if row.get(key) != main.get(key):
                    mismatches.append({"run_id": run_id, "key": key, "main": main.get(key), "repeat": row.get(key)})
        repeat_parity = {
            "checked": True,
            "passed": not mismatches and not missing_main_rows and bool(repeat_rows),
            "repeat_rows": len(repeat_rows),
            "compared_to_main_rows": len(repeat_rows) - len(missing_main_rows),
            "missing_main_rows": missing_main_rows,
            "mismatches": mismatches,
        }

    blockers = []
    if len(rows) != int(expected_runs):
        blockers.append(f"expected_{expected_runs}_runs_got_{len(rows)}")
    if baseline_path2_leakage:
        blockers.append("baseline_path2_roz_disabled_leakage")
    if not active_static_separation:
        blockers.append("active_static_zone_separation_not_detected")
    if not bool(branch4_result["three_layer_gate"].get("method_ready")):
        blockers.append("branch4_strategy_gate_not_method_ready")
    if repeat_runs_root is not None and not repeat_parity.get("passed"):
        blockers.append("repeat_parity_failed")
    decision = "ROZ-2A-A" if not blockers else "ROZ-2A-C"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "runs_root": str(runs_root),
        "run_count": len(rows),
        "expected_runs": int(expected_runs),
        "decision": decision,
        "blockers": blockers,
        "baseline_path2_roz_disabled_leakage_rows": baseline_path2_leakage,
        "active_static_separation": {
            "passed": active_static_separation,
            "active_avg_zone_classes": round(active_avg_zone_classes, 6),
            "reporting_avg_zone_classes": round(reporting_avg_zone_classes, 6),
            "active_avg_shelter_cover_fraction": round(active_avg_shelter, 6),
            "reporting_avg_shelter_cover_fraction": round(reporting_avg_shelter, 6),
        },
        "repeat_parity": repeat_parity,
        "branch4_strategy_gate": branch4_result["three_layer_gate"],
        "no_claims": NO_CLAIMS,
    }
    _write_csv(
        output_dir / "roz2a_run_summary.csv",
        rows,
        [
            "run_id",
            "arm",
            "context",
            "init_arm",
            "seed",
            "path2_roz_event_rows",
            "path2_roz_window_active_rows",
            "path2_roz_parent_in_zone_rows",
            "path2_roz_selected_parent_scan_in_zone_rows",
            "branch_niche_rows",
            "zone_classes",
            "dominant_zone_classes",
            "shelter_cover_fraction",
            "food_collected",
            "damage",
            "patch_cover_mean",
            "patch_shelter_mean",
            "patch_movement_cost_mean",
            "patch_waste_mean",
            "patch_recovery_activity_mean",
            "run_dir",
        ],
    )
    _write_csv(
        output_dir / "roz2a_arm_context_summary.csv",
        aggregate,
        [
            "arm",
            "context",
            "runs",
            "path2_roz_event_rows",
            "path2_roz_window_active_rows",
            "branch_niche_rows",
            "avg_zone_classes",
            "avg_dominant_zone_classes",
            "avg_shelter_cover_fraction",
            "avg_patch_cover_mean",
            "avg_patch_shelter_mean",
            "avg_patch_movement_cost_mean",
            "avg_patch_waste_mean",
            "avg_food_collected",
            "avg_damage",
        ],
    )
    _write_json(output_dir / "roz2a_readiness_decision.json", summary)
    _write_json(output_dir / "roz2a_branch4_output_paths.json", branch4_paths)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ROZ-2A patch niche mechanics readiness outputs.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-runs", type=int, default=54)
    parser.add_argument("--repeat-runs-root", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = analyze(
        Path(args.runs_root),
        Path(args.output_dir),
        expected_runs=int(args.expected_runs),
        repeat_runs_root=Path(args.repeat_runs_root) if str(args.repeat_runs_root).strip() else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
