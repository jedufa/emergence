#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "branch5_specialization_metric_audit_v1"
NO_CLAIMS = (
    "BRANCH-5 is a reporting, metric-repair, reinterpretation, and world-demand selection task for "
    "ecological specialization, descendant branches, and persistent strategies. It does not show "
    "intelligence, emergence, consciousness, broad cognition, memory selection, prediction selection, "
    "solved lineage diversity, beneficial open-ended search, culture, superintelligence, or general "
    "open-ended evolution. It does not add direct cognition reward, lineage rescue, branch reward, "
    "strategy-diversity reward, protected reproduction, forced equal reproduction, archive reseeding, "
    "or survival subsidy."
)
RUN_RE = re.compile(
    r"__arm_(?P<arm>.*?)__context_(?P<context>.*?)__init_(?P<init_arm>.*?)__seed(?P<seed>\d+)"
)
PROFILE_COMPONENTS = (
    "exposure",
    "movement",
    "gather",
    "stay",
    "food",
    "damage",
    "recovery",
    "birth",
    "transition_out",
)
THRESHOLDS = {
    "conservative": {"min_agents": 8, "min_exposure_ticks": 1024, "smoothing_alpha": 0.5},
    "nominal": {"min_agents": 6, "min_exposure_ticks": 384, "smoothing_alpha": 0.5},
    "permissive": {"min_agents": 2, "min_exposure_ticks": 64, "smoothing_alpha": 0.5},
}


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


branch4 = _load_branch4()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _counter(value: Any) -> Counter[str]:
    out: Counter[str] = Counter()
    if isinstance(value, Mapping):
        for key, raw in value.items():
            amount = _num(raw, 0.0)
            if amount > 0.0:
                out[str(key)] += amount
    return out


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _parse_run_id(run_id: str) -> dict[str, str]:
    match = RUN_RE.search(str(run_id))
    if not match:
        return {"arm": "", "context": "", "init_arm": "", "seed": ""}
    return {
        "arm": match.group("arm"),
        "context": match.group("context"),
        "init_arm": match.group("init_arm"),
        "seed": str(int(match.group("seed"))),
    }


def _run_dirs(paths: Iterable[Path]) -> list[Path]:
    return branch4.discover_run_dirs(paths)


def _normalize(counter: Mapping[str, float], keys: Iterable[str], *, smoothing_alpha: float = 0.0) -> dict[str, float]:
    values = {str(key): max(0.0, _num(counter.get(str(key), 0.0), 0.0)) + float(smoothing_alpha) for key in keys}
    total = sum(values.values())
    if total <= 0.0:
        return {key: 0.0 for key in values}
    return {key: value / total for key, value in sorted(values.items())}


def _entropy(dist: Mapping[str, float]) -> float:
    total = sum(max(0.0, float(v)) for v in dist.values())
    if total <= 0.0:
        return 0.0
    n = len([v for v in dist.values() if v > 0.0])
    if n <= 1:
        return 0.0
    ent = 0.0
    for raw in dist.values():
        p = max(0.0, float(raw)) / total
        if p > 0.0:
            ent -= p * math.log(p, 2)
    return round(ent / math.log(n, 2), 6)


def jsd(a: Mapping[str, float], b: Mapping[str, float], *, smoothing_alpha: float = 1.0e-6) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    pa = _normalize(a, keys, smoothing_alpha=smoothing_alpha)
    pb = _normalize(b, keys, smoothing_alpha=smoothing_alpha)
    mid = {key: (pa[key] + pb[key]) / 2.0 for key in keys}

    def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
        total = 0.0
        for key in keys:
            pv = p[key]
            qv = q[key]
            if pv > 0.0 and qv > 0.0:
                total += pv * math.log(pv / qv, 2)
        return total

    return round((kl(pa, mid) + kl(pb, mid)) / 2.0, 6)


def hellinger(a: Mapping[str, float], b: Mapping[str, float], *, smoothing_alpha: float = 1.0e-6) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    pa = _normalize(a, keys, smoothing_alpha=smoothing_alpha)
    pb = _normalize(b, keys, smoothing_alpha=smoothing_alpha)
    return round(math.sqrt(sum((math.sqrt(pa[k]) - math.sqrt(pb[k])) ** 2 for k in keys)) / math.sqrt(2.0), 6)


def total_variation(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    pa = _normalize(a, keys)
    pb = _normalize(b, keys)
    return round(0.5 * sum(abs(pa[k] - pb[k]) for k in keys), 6)


def weighted_median(pairs: Iterable[tuple[float, float]]) -> float:
    rows = sorted((float(value), max(0.0, float(weight))) for value, weight in pairs if math.isfinite(float(value)))
    total = sum(weight for _, weight in rows)
    if not rows or total <= 0.0:
        return 0.0
    acc = 0.0
    for value, weight in rows:
        acc += weight
        if acc >= total / 2.0:
            return round(value, 6)
    return round(rows[-1][0], 6)


def _transition_source_counts(transitions: Mapping[str, Any]) -> Counter[str]:
    out: Counter[str] = Counter()
    for key, raw in dict(transitions or {}).items():
        left = str(key).split("->", 1)[0]
        amount = _num(raw, 0.0)
        if left and amount > 0.0:
            out[left] += amount
    return out


def build_strategy_profile(run_id: str, strategy_id: str, rows: list[Mapping[str, Any]], *, profile: str = "nominal") -> dict[str, Any]:
    thresholds = THRESHOLDS[profile]
    zones: set[str] = set()
    exposure: Counter[str] = Counter()
    movement: Counter[str] = Counter()
    gather: Counter[str] = Counter()
    stay: Counter[str] = Counter()
    food: Counter[str] = Counter()
    damage: Counter[str] = Counter()
    recovery: Counter[str] = Counter()
    birth: Counter[str] = Counter()
    transition_out: Counter[str] = Counter()
    agent_ids: set[int] = set()
    birth_generations: list[int] = []
    final_generations: list[int] = []
    for row in rows:
        agent_ids.add(_int(row.get("agent_id"), -1))
        birth_generations.append(_int(row.get("birth_generation"), 0))
        final_generations.append(_int(row.get("final_generation"), 0))
        counters = {
            "exposure": _counter(row.get("zone_exposure_counts")),
            "movement": _counter(row.get("movement_actions_by_zone")),
            "gather": _counter(row.get("gather_actions_by_zone")),
            "stay": _counter(row.get("stay_actions_by_zone")),
            "food": _counter(row.get("food_collected_by_zone")),
            "damage": _counter(row.get("damage_by_zone")) + _counter(row.get("injury_by_zone")),
            "recovery": _counter(row.get("recovery_by_zone")),
            "birth": _counter(row.get("successful_births_by_zone")),
            "transition_out": _transition_source_counts(row.get("transitions_between_zone_classes")),
        }
        for counter in counters.values():
            zones.update(counter.keys())
        exposure.update(counters["exposure"])
        movement.update(counters["movement"])
        gather.update(counters["gather"])
        stay.update(counters["stay"])
        food.update(counters["food"])
        damage.update(counters["damage"])
        recovery.update(counters["recovery"])
        birth.update(counters["birth"])
        transition_out.update(counters["transition_out"])
    zone_keys = sorted(zones | set(exposure.keys()))
    total_ticks = sum(exposure.values())
    raw_dist = _normalize(exposure, zone_keys, smoothing_alpha=0.0)
    smooth_dist = _normalize(exposure, zone_keys, smoothing_alpha=float(thresholds["smoothing_alpha"]))
    sorted_shares = sorted(raw_dist.values(), reverse=True)
    dominant_share = sorted_shares[0] if sorted_shares else 0.0
    second_share = sorted_shares[1] if len(sorted_shares) > 1 else 0.0
    confidence = "high"
    if len(agent_ids) < int(thresholds["min_agents"]) or total_ticks < int(thresholds["min_exposure_ticks"]):
        confidence = "low_exposure"
    component_counters = {
        "exposure": exposure,
        "movement": movement,
        "gather": gather,
        "stay": stay,
        "food": food,
        "damage": damage,
        "recovery": recovery,
        "birth": birth,
        "transition_out": transition_out,
    }
    vector: dict[str, float] = {}
    for component in PROFILE_COMPONENTS:
        counter = component_counters[component]
        for zone in zone_keys:
            if component == "exposure":
                value = raw_dist.get(zone, 0.0)
            else:
                value = _num(counter.get(zone), 0.0) / max(1.0, _num(exposure.get(zone), 0.0))
            if value > 0.0:
                vector[f"{component}:{zone}"] = round(value, 9)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "strategy_id": strategy_id,
        "profile": profile,
        "agent_count": len([aid for aid in agent_ids if aid >= 0]),
        "total_exposure_ticks": round(total_ticks, 6),
        "zone_count": len(zone_keys),
        "zone_exposure_distribution": smooth_dist,
        "raw_zone_exposure_distribution": raw_dist,
        "dominant_zone_share": round(dominant_share, 6),
        "second_zone_share": round(second_share, 6),
        "zone_preference_entropy": _entropy(raw_dist),
        "zone_preference_concentration": round(1.0 - _entropy(raw_dist), 6),
        "successful_births": round(sum(birth.values()), 6),
        "generation_span": max(final_generations or [0]) - min(birth_generations or [0]) + 1,
        "confidence": confidence,
        "profile_vector": vector,
        "birth_distribution": _normalize(birth, zone_keys, smoothing_alpha=0.0),
    }


def profile_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    return {
        "full_profile_jsd": jsd(a.get("zone_exposure_distribution", {}), b.get("zone_exposure_distribution", {})),
        "hellinger": hellinger(a.get("zone_exposure_distribution", {}), b.get("zone_exposure_distribution", {})),
        "behavior_distance": total_variation(a.get("profile_vector", {}), b.get("profile_vector", {})),
        "reproductive_jsd": jsd(a.get("birth_distribution", {}), b.get("birth_distribution", {})),
    }


def _strategy_rows(run_dir: Path, *, profile: str = "nominal") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_id = str(branch4._run_label(run_dir))
    merged = branch4._merged_agent_rows(run_dir)
    by_strategy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in merged:
        sig = branch4.niche_signature(row, profile=profile)
        if sig.get("strategy_status") == "available":
            row = dict(row)
            row["agent_id"] = _int(row.get("agent_id"), -1)
            by_strategy[str(sig["strategy_id"])].append(row)
    profiles = [build_strategy_profile(run_id, sid, list(rows), profile=profile) for sid, rows in sorted(by_strategy.items())]
    return profiles, merged


def _pairwise_summary(profiles: list[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [p for p in profiles if p.get("confidence") == "high"]
    distance_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for idx, left in enumerate(eligible):
        for right in eligible[idx + 1 :]:
            dist = profile_distance(left, right)
            weight = math.sqrt(max(1.0, _num(left.get("agent_count"), 1.0)) * max(1.0, _num(right.get("agent_count"), 1.0)))
            for key, value in dist.items():
                distance_pairs[key].append((value, weight))
    return {
        "persistent_strategy_count": len(eligible),
        "low_confidence_strategy_count": len(profiles) - len(eligible),
        "weighted_median_full_profile_jsd": weighted_median(distance_pairs["full_profile_jsd"]),
        "weighted_median_hellinger": weighted_median(distance_pairs["hellinger"]),
        "weighted_median_behavior_distance": weighted_median(distance_pairs["behavior_distance"]),
        "weighted_median_reproductive_jsd": weighted_median(distance_pairs["reproductive_jsd"]),
        "weighted_median_preference_concentration": weighted_median(
            (_num(p.get("zone_preference_concentration"), 0.0), _num(p.get("agent_count"), 1.0)) for p in eligible
        ),
        "weighted_median_dominant_zone_share": weighted_median(
            (_num(p.get("dominant_zone_share"), 0.0), _num(p.get("agent_count"), 1.0)) for p in eligible
        ),
        "strategies_with_births": sum(1 for p in eligible if _num(p.get("successful_births"), 0.0) > 0.0),
    }


def analyze_runs(run_dirs: list[Path], *, profile: str = "nominal") -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    coverage = {
        "run_count": len(run_dirs),
        "rows_with_zone_exposure_counts": 0,
        "rows_with_food_by_zone": 0,
        "rows_with_damage_by_zone": 0,
        "rows_with_recovery_by_zone": 0,
        "rows_with_births_by_zone": 0,
        "rows_with_transitions": 0,
        "agent_rows": 0,
    }
    for run_dir in run_dirs:
        run_id = str(branch4._run_label(run_dir))
        parsed = _parse_run_id(run_id)
        profiles, merged = _strategy_rows(run_dir, profile=profile)
        for row in merged:
            coverage["agent_rows"] += 1
            coverage["rows_with_zone_exposure_counts"] += int(bool(row.get("zone_exposure_counts")))
            coverage["rows_with_food_by_zone"] += int(bool(row.get("food_collected_by_zone")))
            coverage["rows_with_damage_by_zone"] += int(bool(row.get("damage_by_zone") or row.get("injury_by_zone")))
            coverage["rows_with_recovery_by_zone"] += int(bool(row.get("recovery_by_zone")))
            coverage["rows_with_births_by_zone"] += int(bool(row.get("successful_births_by_zone")))
            coverage["rows_with_transitions"] += int(bool(row.get("transitions_between_zone_classes")))
        summary = _pairwise_summary(profiles)
        summary_rows.append({"run_id": run_id, **parsed, **summary, "run_dir": str(run_dir)})
        for p in profiles:
            profile_rows.append(
                {
                    "run_id": run_id,
                    **parsed,
                    "strategy_id": p["strategy_id"],
                    "agent_count": p["agent_count"],
                    "total_exposure_ticks": p["total_exposure_ticks"],
                    "zone_count": p["zone_count"],
                    "dominant_zone_share": p["dominant_zone_share"],
                    "second_zone_share": p["second_zone_share"],
                    "zone_preference_entropy": p["zone_preference_entropy"],
                    "zone_preference_concentration": p["zone_preference_concentration"],
                    "successful_births": p["successful_births"],
                    "generation_span": p["generation_span"],
                    "confidence": p["confidence"],
                }
            )
    return {"summary_rows": summary_rows, "profile_rows": profile_rows, "coverage": coverage}


def _paired(rows: list[Mapping[str, Any]], active_arm: str = "active_roz2a", control_arm: str = "reporting_only") -> list[dict[str, Any]]:
    controls = {
        (r.get("context"), r.get("init_arm"), r.get("seed")): r
        for r in rows
        if r.get("arm") == control_arm
    }
    out: list[dict[str, Any]] = []
    metrics = [
        "weighted_median_full_profile_jsd",
        "weighted_median_hellinger",
        "weighted_median_behavior_distance",
        "weighted_median_reproductive_jsd",
        "weighted_median_preference_concentration",
        "weighted_median_dominant_zone_share",
        "persistent_strategy_count",
        "strategies_with_births",
    ]
    for row in rows:
        if row.get("arm") != active_arm:
            continue
        key = (row.get("context"), row.get("init_arm"), row.get("seed"))
        control = controls.get(key)
        if not control:
            continue
        for metric in metrics:
            active = _num(row.get(metric), 0.0)
            baseline = _num(control.get(metric), 0.0)
            out.append(
                {
                    "context": row.get("context", ""),
                    "init_arm": row.get("init_arm", ""),
                    "seed": row.get("seed", ""),
                    "metric": metric,
                    "active": round(active, 6),
                    "control": round(baseline, 6),
                    "delta": round(active - baseline, 6),
                    "relative_delta": round((active - baseline) / max(1.0, abs(baseline)), 6),
                }
            )
    return out


def _metric_summary(pair_rows: list[Mapping[str, Any]], metric: str, *, context: str = "roz2a_niche_context") -> dict[str, Any]:
    rows = [r for r in pair_rows if r.get("metric") == metric and r.get("context") == context]
    deltas = [_num(r.get("delta"), 0.0) for r in rows]
    rels = [_num(r.get("relative_delta"), 0.0) for r in rows]
    return {
        "metric": metric,
        "context": context,
        "pairs": len(rows),
        "median_delta": weighted_median((d, 1.0) for d in deltas),
        "median_relative_delta": weighted_median((d, 1.0) for d in rels),
        "improved_count": sum(1 for d in deltas if d > 0.0),
        "worse_count": sum(1 for d in deltas if d < 0.0),
    }


def _reinterpretation(pair_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    full = _metric_summary(pair_rows, "weighted_median_full_profile_jsd")
    behavior = _metric_summary(pair_rows, "weighted_median_behavior_distance")
    concentration = _metric_summary(pair_rows, "weighted_median_preference_concentration")
    births = _metric_summary(pair_rows, "strategies_with_births")
    real_specialization = (
        full["pairs"] == 10
        and full["median_delta"] >= 0.03
        and behavior["median_delta"] >= 0.03
        and concentration["median_delta"] >= 0.02
    )
    if real_specialization:
        decision = "ROZ-REINTERPRET-A"
        meaning = "ROZ creates repaired-metric ecological specialization but remains branch-neutral under ROZ-2B."
    else:
        decision = "ROZ-REINTERPRET-B"
        meaning = "ROZ remains specialization-neutral or weak under the repaired metric; retain as clean substrate only."
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "meaning": meaning,
        "full_profile_jsd": full,
        "behavior_distance": behavior,
        "preference_concentration": concentration,
        "strategies_with_births": births,
        "branch_benefit": "not_supported_by_ROZ2B_final_branch_classification",
        "no_claims": NO_CLAIMS,
    }


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    summary_rows = list(result["summary_rows"])
    pair_rows = _paired(summary_rows)
    reinterpretation = _reinterpretation(pair_rows)
    paths = {
        "reanalysis": output_dir / "branch5_roz2b_reanalysis.csv",
        "paired": output_dir / "branch5_roz2b_repaired_metric_paired.csv",
        "summary": output_dir / "branch5_roz2b_specialization_summary.json",
        "reinterpretation": output_dir / "branch5_roz2b_reinterpretation.json",
        "coverage": output_dir / "branch5_stage0_data_coverage.json",
    }
    _write_csv(
        paths["reanalysis"],
        summary_rows,
        [
            "run_id",
            "arm",
            "context",
            "init_arm",
            "seed",
            "persistent_strategy_count",
            "low_confidence_strategy_count",
            "weighted_median_full_profile_jsd",
            "weighted_median_hellinger",
            "weighted_median_behavior_distance",
            "weighted_median_reproductive_jsd",
            "weighted_median_preference_concentration",
            "weighted_median_dominant_zone_share",
            "strategies_with_births",
            "run_dir",
        ],
    )
    _write_csv(paths["paired"], pair_rows, ["context", "init_arm", "seed", "metric", "active", "control", "delta", "relative_delta"])
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "profile": result.get("profile", "nominal"),
        "run_count": len(summary_rows),
        "paired_metric_summaries": {
            metric: _metric_summary(pair_rows, metric)
            for metric in [
                "weighted_median_full_profile_jsd",
                "weighted_median_hellinger",
                "weighted_median_behavior_distance",
                "weighted_median_reproductive_jsd",
                "weighted_median_preference_concentration",
                "weighted_median_dominant_zone_share",
                "persistent_strategy_count",
                "strategies_with_births",
            ]
        },
        "repaired_metric_method_ready": True,
        "metric_saturation_repaired": True,
        "no_claims": NO_CLAIMS,
    }
    _write_json(paths["summary"], summary_payload)
    _write_json(paths["reinterpretation"], reinterpretation)
    coverage = dict(result["coverage"])
    coverage.update(
        {
            "schema_version": SCHEMA_VERSION,
            "archived_roz2b_supports_repaired_metric": coverage.get("rows_with_zone_exposure_counts", 0) > 0
            and coverage.get("rows_with_food_by_zone", 0) > 0
            and coverage.get("rows_with_transitions", 0) > 0,
            "available_profile_fields": [
                "zone_exposure_counts",
                "movement_actions_by_zone",
                "gather_actions_by_zone",
                "stay_actions_by_zone",
                "food_collected_by_zone",
                "damage_by_zone",
                "injury_by_zone",
                "recovery_by_zone",
                "successful_births_by_zone",
                "transitions_between_zone_classes",
                "total_observed_ticks",
            ],
        }
    )
    _write_json(paths["coverage"], coverage)
    return {key: str(path) for key, path in paths.items()}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BRANCH-5 repaired ecological specialization metric audit.")
    parser.add_argument("--runs-root", action="append", default=[], help="ROZ-2B campaign root or extracted archive root.")
    parser.add_argument("--output-dir", required=True, help="Compact evidence output directory.")
    parser.add_argument("--profile", choices=sorted(THRESHOLDS), default="nominal")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dirs = _run_dirs([Path(p) for p in args.runs_root])
    result = analyze_runs(run_dirs, profile=str(args.profile))
    result["profile"] = str(args.profile)
    paths = write_outputs(result, Path(args.output_dir))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_dirs": len(run_dirs),
        "paths": paths,
        "coverage": result["coverage"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
