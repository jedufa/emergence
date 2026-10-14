"""Deterministic campaign decision report generation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tools.evidence.contracts import check_campaign_schema, dumps_result
from tools.evidence.decisions import analyze_profile
from tools.evidence.manifest import dumps_manifest, load_manifest
from tools.evidence.runs import RUN_STATUS_FIELDS, run_status_csv_rows


DEFAULT_DETERMINISTIC_TIMESTAMP = "1970-01-01T00:00:00Z"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: tuple(str(item.get(field, "")) for field in fields)):
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_run_status(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUN_STATUS_FIELDS))
        writer.writeheader()
        for row in sorted(run_status_csv_rows(rows), key=lambda item: tuple(str(item.get(field, "")) for field in RUN_STATUS_FIELDS)):
            writer.writerow(row)


def _section_list(items: list[Any]) -> str:
    if not items:
        return "- Not available in supplied outputs.\n"
    return "".join(f"- {item}\n" for item in items)


def render_markdown(schema_status: Mapping[str, Any], decision: Mapping[str, Any], generated_at: str) -> str:
    lines: list[str] = [
        "# Campaign Decision Report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Executive Verdict",
        "",
        f"Decision: **{decision.get('decision', 'not available')}**",
        "",
        _section_list(list(decision.get("rationale") or [])),
        "## Campaign Design",
        "",
        f"- Campaign id: `{schema_status.get('campaign_id', 'not available')}`",
        f"- Profile: `{schema_status.get('profile', 'not available')}`",
        f"- Expected runs: `{schema_status.get('run_count_expected', 'not available')}`",
        f"- Observed runs: `{schema_status.get('run_count_observed', 'not available')}`",
        "",
        "## Run Completeness",
        "",
        f"- Paired coverage: `{decision.get('paired_coverage', {}).get('observed', 'not available')}` / `{decision.get('paired_coverage', {}).get('expected', 'not available')}`",
        f"- Anomaly groups: `{len((schema_status.get('anomaly_summary') or {}).get('affected') or {})}`",
        "",
        "## Schema Contract Status",
        "",
        f"- Schema ok: `{bool(schema_status.get('ok'))}`",
        "- Errors:",
        _section_list(list(schema_status.get("errors") or [])),
        "- Warnings:",
        _section_list(list(schema_status.get("warnings") or [])),
        "## Determinism/Repeat Status",
        "",
        "- Not available in supplied outputs.",
        "",
        "## Primary Deltas",
        "",
    ]
    for row in decision.get("normal_vs_ablated", {}).get("summary", []):
        alive = ((row.get("alive") or {}).get("mean_delta")) if isinstance(row.get("alive"), Mapping) else None
        births = ((row.get("mean_offspring_total") or {}).get("mean_delta")) if isinstance(row.get("mean_offspring_total"), Mapping) else None
        lines.append(f"- `{row.get('ecology_context')}` / `{row.get('init_arm')}`: alive `{alive}`, births `{births}`")
    if not decision.get("normal_vs_ablated", {}).get("summary"):
        lines.append("- Not available in supplied outputs.")
    lines.extend(["", "## Pressure-Specificity Result", ""])
    for row in decision.get("pressure_specificity", []):
        lines.append(
            f"- `{row.get('ecology_context')}`: alive pressure-minus-flat `{row.get('pressure_minus_flat_delta_alive')}`, births pressure-minus-flat `{row.get('pressure_minus_flat_delta_mean_offspring_total')}`"
        )
    if not decision.get("pressure_specificity"):
        lines.append("- Not available in supplied outputs.")
    lines.extend(["", "## Lineage/Founder Arm Summary", ""])
    for row in decision.get("founder_summary", []):
        lines.append(f"- `{row.get('ecology_context')}`: founder-minus-legacy alive delta `{row.get('founder_balanced_minus_legacy_delta_alive')}`")
    if not decision.get("founder_summary"):
        lines.append("- Not available in supplied outputs.")
    lines.extend(
        [
            "",
            "## Behavior Proxy Summary",
            "",
            "- Behavior proxy deltas are included in `paired_comparisons.csv` when supplied by metrics.",
            "",
            "## Cost Summary",
            "",
            f"- Mean perception cost charged: `{(decision.get('cost_summary') or {}).get('mean_perception_cost_charged', 'not available')}`",
            f"- Observed cost rows: `{(decision.get('cost_summary') or {}).get('observed_count', 'not available')}`",
            "",
            "## Anomalies/Failures",
            "",
            "See `anomaly_summary.json` and `run_status.csv` for the complete deterministic tables.",
            "",
            "## A/B/C/D Decision",
            "",
            f"- `{decision.get('decision', 'not available')}`",
            "",
            "## Claims Allowed",
            "",
            _section_list(list(decision.get("claims_allowed") or [])),
            "## Claims Not Allowed",
            "",
            _section_list(list(decision.get("claims_not_allowed") or [])),
            "## Recommended Next Action",
            "",
            f"- {decision.get('recommended_next_action', 'Not available in supplied outputs.')}",
            "",
        ]
    )
    return "\n".join(lines)


def generate_campaign_decision_report(
    manifest_path: str | Path,
    profile_name: str,
    out_dir: str | Path,
    *,
    generated_at: str | None = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    timestamp = generated_at or (
        DEFAULT_DETERMINISTIC_TIMESTAMP if deterministic else datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evidence_manifest = load_manifest(manifest_path)
    manifest = evidence_manifest.data
    schema_status = check_campaign_schema(manifest_path)
    decision = analyze_profile(schema_status, profile_name)

    summary = {
        "campaign_id": schema_status.get("campaign_id"),
        "profile": profile_name,
        "decision": decision["decision"],
        "generated_at": timestamp,
        "git_commit": manifest.get("git_commit", ""),
        "branch": manifest.get("branch", ""),
        "recommended_next_action": decision["recommended_next_action"],
        "claims_allowed": decision["claims_allowed"],
        "claims_not_allowed": decision["claims_not_allowed"],
        "reward_audit": decision["reward_audit"],
    }

    (out / "campaign_evidence_manifest.json").write_text(dumps_manifest(manifest), encoding="utf-8")
    (out / "decision_report.md").write_text(render_markdown(schema_status, decision, timestamp), encoding="utf-8")
    _write_json(out / "decision_summary.json", summary)
    _write_json(out / "anomaly_summary.json", schema_status.get("anomaly_summary") or {})
    (out / "schema_status.json").write_text(dumps_result(dict(schema_status)), encoding="utf-8")
    _write_run_status(out / "run_status.csv", list(schema_status.get("run_status") or []))
    _write_csv(out / "paired_comparisons.csv", list(decision.get("normal_vs_ablated", {}).get("per_pair_rows") or []))
    return {
        "out_dir": str(out),
        "schema_status": schema_status,
        "decision": decision,
        "summary": summary,
        "artifacts": {
            "decision_report": str(out / "decision_report.md"),
            "decision_summary": str(out / "decision_summary.json"),
            "paired_comparisons": str(out / "paired_comparisons.csv"),
            "run_status": str(out / "run_status.csv"),
            "anomaly_summary": str(out / "anomaly_summary.json"),
            "schema_status": str(out / "schema_status.json"),
            "manifest": str(out / "campaign_evidence_manifest.json"),
        },
    }
