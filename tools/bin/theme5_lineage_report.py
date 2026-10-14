#!/usr/bin/env python3
from __future__ import annotations

"""Generate a basic Theme 5 biological lineage report from run outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return int(default)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _latest_records_by_generation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda r: _int(r.get("generation", 0), 0))


def _read_branch_manifests(run_dir: Path) -> List[Dict[str, Any]]:
    manifest_dir = run_dir / "archives" / "lineage_qd" / "branch_manifests"
    if not manifest_dir.exists():
        return []
    manifests = [_read_json(path) for path in sorted(manifest_dir.glob("*.json"), key=lambda p: p.name)]
    return [m for m in manifests if m]


def render_report(run_dir: Path) -> str:
    metrics_dir = run_dir / "metrics"
    summary_rows = _read_summary(metrics_dir / "lineage_summary.csv")
    metric_rows = _read_jsonl(metrics_dir / "lineage_metrics.jsonl")
    event_rows = _read_jsonl(metrics_dir / "lineage_events.jsonl")
    descriptor_rows = _read_jsonl(metrics_dir / "lineage_descriptors.jsonl")
    candidate_rows = _read_jsonl(metrics_dir / "branch_candidates.jsonl")
    founder_spatial_rows = _read_jsonl(metrics_dir / "lineage_founder_spatial_survival.jsonl")
    founder_spatial_init = _read_json(metrics_dir / "lineage_founder_spatial_initialization.json")
    archive_rows = _read_jsonl(run_dir / "archives" / "lineage_qd" / "archive_index.jsonl")
    manifest_rows = _read_branch_manifests(run_dir)

    latest = summary_rows[-1] if summary_rows else {}
    warnings = [r for r in summary_rows if _int(r.get("lineage_collapse_warning", 0), 0) != 0]
    active_values = [_int(r.get("active_lineages", 0), 0) for r in summary_rows]
    entropy_values = [_float(r.get("lineage_population_entropy", 0.0), 0.0) for r in summary_rows]

    latest_by_lineage: Dict[str, Dict[str, Any]] = {}
    for row in metric_rows:
        lid = str(row.get("lineage_id", "") or "")
        gen = _int(row.get("generation", 0), 0)
        prev = latest_by_lineage.get(lid)
        if lid and (prev is None or gen >= _int(prev.get("generation", 0), 0)):
            latest_by_lineage[lid] = row

    event_counts: Dict[str, int] = defaultdict(int)
    for event in event_rows:
        event_counts[str(event.get("event", "unknown") or "unknown")] += 1

    latest_descriptor_record = _latest_records_by_generation(descriptor_rows)
    latest_descriptors = list(latest_descriptor_record.get("lineages", []) if latest_descriptor_record else [])
    descriptor_summary = latest_descriptor_record.get("global_descriptor_summary", {}) if latest_descriptor_record else {}
    latest_candidate_record = _latest_records_by_generation(candidate_rows)
    latest_candidates = list(latest_candidate_record.get("candidates", []) if latest_candidate_record else [])
    candidate_type_counts: Dict[str, int] = defaultdict(int)
    for candidate in latest_candidates:
        candidate_type_counts[str(candidate.get("candidate_type", "unknown") or "unknown")] += 1
    archive_type_counts: Dict[str, int] = defaultdict(int)
    role_counts: Dict[str, int] = defaultdict(int)
    for row in archive_rows:
        archive_type_counts[str(row.get("candidate_type", "unknown") or "unknown")] += 1
        role_counts[str(row.get("ecological_role", "unknown") or "unknown")] += 1
    archive_lineages = {str(row.get("lineage_id", "") or "") for row in archive_rows if str(row.get("lineage_id", "") or "")}
    live_lineages = set(latest_by_lineage.keys())
    for row in metric_rows:
        role_counts[str(row.get("ecological_role", "unknown") or "unknown")] += 1
    for row in latest_descriptors:
        role_counts[str(row.get("ecological_role", "unknown") or "unknown")] += 1
    manifest_type_counts: Dict[str, int] = defaultdict(int)
    for manifest in manifest_rows:
        manifest_type_counts[str(manifest.get("manifest_type", "unknown") or "unknown")] += 1

    lines: List[str] = []
    lines.append("# Theme 5 Lineage Report")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append("- Schema: `theme5_lineage_report_v2`")
    lines.append("")
    lines.append("## Summary")
    if not summary_rows:
        lines.append("")
        lines.append("No lineage summary rows were found.")
    else:
        lines.append("")
        lines.append(f"- Generations reported: {len(summary_rows)}")
        lines.append(f"- Latest generation: {_int(latest.get('generation', 0), 0)}")
        lines.append(f"- Latest active lineages: {_int(latest.get('active_lineages', 0), 0)}")
        lines.append(f"- Active lineage range: {min(active_values)} to {max(active_values)}")
        lines.append(f"- Dominant lineage: `{latest.get('dominant_lineage_id', '') or 'n/a'}`")
        lines.append(f"- Dominant population share: {_fmt(latest.get('dominant_lineage_population_share'))}")
        lines.append(f"- Top-3 population share: {_fmt(latest.get('top3_lineage_population_share'))}")
        lines.append(f"- Latest lineage entropy: {_fmt(latest.get('lineage_population_entropy'))}")
        lines.append(f"- Entropy range: {_fmt(min(entropy_values))} to {_fmt(max(entropy_values))}")
        lines.append(f"- Collapse warning rows: {len(warnings)}")
        lines.append(f"- Latest collapse streak: {_int(latest.get('lineage_collapse_streak', 0), 0)}")

    lines.append("")
    lines.append("## Collapse Events")
    if event_counts:
        for key in sorted(event_counts.keys()):
            lines.append(f"- {key}: {event_counts[key]}")
    else:
        lines.append("- No lineage events were emitted.")

    lines.append("")
    lines.append("## Current Lineages")
    if latest_by_lineage:
        lines.append("")
        lines.append("| Lineage | Founder | Population | Pop share | Offspring | Repro share | Max depth |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for lid, row in sorted(
            latest_by_lineage.items(),
            key=lambda kv: (-_float(kv[1].get("population_share", 0.0), 0.0), str(kv[0])),
        )[:20]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{lid}`",
                        str(_int(row.get("founder_id", 0), 0)),
                        str(_int(row.get("population_count", 0), 0)),
                        _fmt(row.get("population_share")),
                        str(_int(row.get("offspring_count", 0), 0)),
                        _fmt(row.get("reproduction_share")),
                        str(_int(row.get("max_lineage_depth", 0), 0)),
                    ]
                )
                + " |"
            )
    else:
        lines.append("")
        lines.append("No per-lineage metric rows were found.")

    lines.append("")
    lines.append("## Behavioral Descriptors")
    if latest_descriptors:
        available = descriptor_summary.get("available_numeric_fields", [])
        unavailable = descriptor_summary.get("unavailable_numeric_fields", [])
        lines.append("")
        lines.append(f"- Latest descriptor generation: {_int(latest_descriptor_record.get('generation', 0), 0)}")
        lines.append(f"- Described lineages: {len(latest_descriptors)}")
        lines.append(f"- Available numeric fields: {', '.join(str(x) for x in available) if available else 'none'}")
        lines.append(f"- Missing or unavailable fields: {', '.join(str(x) for x in unavailable) if unavailable else 'none'}")
        lines.append("")
        lines.append("| Lineage | Pop share | Survival | Offspring | Memory | GRU frac | Injury | Distance proxy |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        dominant_id = str(latest.get("dominant_lineage_id", "") or "")
        dominant = next((d for d in latest_descriptors if str(d.get("lineage_id", "")) == dominant_id), None)
        for desc in sorted(
            latest_descriptors,
            key=lambda d: (-_float(d.get("population_share", 0.0), 0.0), str(d.get("lineage_id", ""))),
        )[:20]:
            if dominant is not None:
                try:
                    from simulation.metrics_pkg.lineage import descriptor_distance

                    distance = descriptor_distance(desc, dominant)
                except Exception:
                    distance = 0.0
            else:
                distance = 0.0
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{desc.get('lineage_id', '')}`",
                        _fmt(desc.get("population_share")),
                        _fmt(desc.get("mean_survival_steps")),
                        _fmt(desc.get("mean_offspring_count")),
                        _fmt(desc.get("mean_memory_dim")),
                        _fmt(desc.get("gru_fraction")),
                        _fmt(desc.get("mean_injury")),
                        _fmt(distance),
                    ]
                )
                + " |"
            )
    else:
        lines.append("")
        lines.append("No behavioral descriptor rows were found. Descriptor files may be absent for older runs.")

    lines.append("")
    lines.append("## Branch Candidates")
    if latest_candidates:
        lines.append("")
        lines.append(f"- Latest candidate generation: {_int(latest_candidate_record.get('generation', 0), 0)}")
        lines.append(f"- Candidate records: {len(latest_candidates)}")
        for key in sorted(candidate_type_counts.keys()):
            lines.append(f"- {key}: {candidate_type_counts[key]}")
        lines.append("")
        lines.append("| Lineage | Type | Score | Pop share | Reasons |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for candidate in latest_candidates[:25]:
            reasons = candidate.get("reasons", [])
            if isinstance(reasons, list):
                reason_text = "; ".join(str(x) for x in reasons[:2])
            else:
                reason_text = str(reasons)
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{candidate.get('lineage_id', '')}`",
                        str(candidate.get("candidate_type", "")),
                        _fmt(candidate.get("candidate_score")),
                        _fmt(candidate.get("population_share")),
                        reason_text.replace("|", "/"),
                    ]
                )
                + " |"
            )
    else:
        lines.append("")
        lines.append("No branch candidate rows were found. Candidate detection may be absent for older runs or no candidates were detected.")

    lines.append("")
    lines.append("## Founder Spatial Separation")
    lines.append("")
    if founder_spatial_init:
        lines.append(f"- Enabled: {bool(founder_spatial_init.get('enabled'))}")
        lines.append(f"- Assignment mode: `{founder_spatial_init.get('assignment_mode', 'none')}`")
        lines.append(f"- Founder spatial groups: {_int(founder_spatial_init.get('group_count', 0), 0)}")
        lines.append(f"- Min-distance satisfied: {bool(founder_spatial_init.get('min_distance_satisfied', True))}")
        lines.append(f"- Founder collision index: {_fmt(founder_spatial_init.get('founder_collision_index', 0.0))}")
        warning = str(founder_spatial_init.get("warning", "") or "")
        if warning:
            lines.append(f"- Warning: `{warning}`")
    else:
        lines.append("- No founder spatial initialization artifact was found.")
    latest_spatial = founder_spatial_rows[-1] if founder_spatial_rows else {}
    latest_spatial_groups = list(latest_spatial.get("groups", []) if isinstance(latest_spatial, dict) else [])
    if latest_spatial_groups:
        lines.append("")
        lines.append("| Lineage | Initial | Alive | Survival fraction | Alive / initial |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in latest_spatial_groups[:20]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.get('lineage_id', '')}`",
                        str(_int(row.get("initial_count", 0), 0)),
                        str(_int(row.get("alive_count", 0), 0)),
                        _fmt(row.get("survival_fraction", 0.0)),
                        _fmt(row.get("alive_fraction_relative_to_initial", row.get("survival_fraction", 0.0))),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Quality-Diversity Archive")
    lines.append("")
    if archive_rows:
        latest_archive_generation = max(_int(r.get("generation", 0), 0) for r in archive_rows)
        lines.append(f"- Archive entries: {len(archive_rows)}")
        lines.append(f"- Latest archived generation: {latest_archive_generation}")
        lines.append("- Archive policy: passive analytical archive; no automatic reseeding or live intervention.")
        lines.append("- Boundary: live diversity != archive diversity.")
        lines.append(f"- Current live lineages: {len(live_lineages)}")
        lines.append(f"- Archived lineages: {len(archive_lineages)}")
        lines.append(f"- Archive-only lineage IDs: {len(archive_lineages - live_lineages)}")
        lines.append(f"- Live-only lineage IDs: {len(live_lineages - archive_lineages)}")
        replay_ready = [row for row in archive_rows if bool((row.get("branch_replay_eligibility") or {}).get("eligible"))]
        lines.append(f"- Replay-eligible checkpoint-backed entries: {len(replay_ready)}")
        for key in sorted(archive_type_counts.keys()):
            lines.append(f"- {key}: {archive_type_counts[key]}")
        lines.append("")
        lines.append("| Entry | Lineage | Type | Score | Generation | Replay eligible |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        top_entries = sorted(
            archive_rows,
            key=lambda r: (
                -_float(r.get("candidate_score", 0.0), 0.0),
                str(r.get("candidate_type", "")),
                str(r.get("lineage_id", "")),
                str(r.get("archive_entry_id", "")),
            ),
        )[:20]
        for entry in top_entries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{entry.get('archive_entry_id', '')}`",
                        f"`{entry.get('lineage_id', '')}`",
                        str(entry.get("candidate_type", "")),
                        _fmt(entry.get("candidate_score")),
                        str(_int(entry.get("generation", 0), 0)),
                        str(bool((entry.get("branch_replay_eligibility") or {}).get("eligible"))),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No quality-diversity archive entries were found. Older runs or runs without candidates may not have archive output.")
        lines.append("Archive behavior is analytical only and does not reseed archived lineages.")
        lines.append("Boundary: live diversity != archive diversity.")

    lines.append("")
    lines.append("## Branch/Restart Manifests")
    lines.append("")
    if manifest_rows:
        lines.append(f"- Manifests found: {len(manifest_rows)}")
        for key in sorted(manifest_type_counts.keys()):
            lines.append(f"- {key}: {manifest_type_counts[key]}")
        lines.append("- Execution status: manifest-only. This report found no automatic branch execution or reseeding.")
        lines.append("")
        lines.append("| Manifest | Type | Lineages | Archive entries | Warning |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for manifest in manifest_rows[:20]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{manifest.get('manifest_id', '')}`",
                        str(manifest.get("manifest_type", "")),
                        str(len(manifest.get("selected_lineage_ids", []) or [])),
                        str(len(manifest.get("selected_archive_entry_ids", []) or [])),
                        str(manifest.get("explicit_warning", "")).replace("|", "/"),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No branch/restart experiment manifests were found.")
        lines.append("Stage 8 support is explicit manifest generation and validation; ordinary simulation runs are not altered.")

    lines.append("")
    lines.append("## Co-evolution Readiness")
    lines.append("")
    if role_counts:
        lines.append("- Ecological role fields are present in Theme 5 lineage outputs.")
        for key in sorted(role_counts.keys()):
            lines.append(f"- {key}: {role_counts[key]}")
    else:
        lines.append("- Ecological role fields were absent in the inspected outputs.")
    lines.append("- Current default role is `general_agent` unless future systems provide role assignment.")
    lines.append("- Predator/prey/parasite/niche-specialist lineage metrics are schema readiness only; no mechanics are implemented here.")

    lines.append("")
    lines.append("## Stage Completion Summary")
    lines.append("- Behavioral descriptors: implemented as diagnostic lineage descriptors; they do not redefine biological lineage.")
    lines.append("- Branch candidate detection: implemented as analytical recommendations only; branch execution is not implemented.")
    lines.append("- Quality-diversity archive: implemented as metadata-only analytical archive output.")
    lines.append("- Branch/restart experiments from lineage candidates: implemented as explicit manifest generation and validation; execution remains deferred.")
    lines.append("- Co-evolution readiness hooks: implemented as optional role fields in Theme 5 outputs and report sections.")
    lines.append("- Formal Theme 5 verification campaigns are deferred for the grouped campaign pack.")
    lines.append("- No archive reseed, population quota, branch execution, or live intervention behavior is implemented by this report.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed run directory containing metrics/ lineage outputs.")
    parser.add_argument("--out", default="", help="Output markdown path. Defaults to <run-dir>/lineage_report.md.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if str(args.out or "").strip() else (run_dir / "lineage_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(run_dir), encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
