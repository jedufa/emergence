"""Generate a deterministic campaign evidence index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACTS = (
    ("manifest", "campaign_evidence_manifest.json"),
    ("decision_report", "decision_report.md"),
    ("summary_json", "decision_summary.json"),
    ("paired_comparison_csv", "paired_comparisons.csv"),
    ("anomaly_summary", "anomaly_summary.json"),
    ("schema_status", "schema_status.json"),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _link(root: Path, path: Path) -> str:
    if not path.exists():
        return "missing"
    return f"[{path.name}]({path.relative_to(root).as_posix()})"


def discover_campaigns(evidence_root: str | Path) -> list[dict[str, Any]]:
    root = Path(evidence_root)
    rows: list[dict[str, Any]] = []
    for folder in sorted([item for item in root.iterdir() if item.is_dir()] if root.exists() else []):
        summary = _read_json(folder / "decision_summary.json")
        schema = _read_json(folder / "schema_status.json")
        manifest = _read_json(folder / "campaign_evidence_manifest.json")
        campaign_id = summary.get("campaign_id") or schema.get("campaign_id") or manifest.get("campaign_id") or folder.name
        rows.append(
            {
                "folder": folder.name,
                "campaign_id": campaign_id,
                "profile": summary.get("profile") or schema.get("profile") or manifest.get("profile") or "missing",
                "decision": summary.get("decision", "missing"),
                "status": "ok" if summary.get("decision") else "missing_artifacts",
                "git_commit": summary.get("git_commit") or manifest.get("git_commit") or "missing",
                "branch": summary.get("branch") or manifest.get("branch") or "missing",
                "timestamp": summary.get("generated_at") or manifest.get("timestamp_utc") or "missing",
                "paths": {name: _link(root, folder / filename) for name, filename in ARTIFACTS},
            }
        )
    rows.sort(key=lambda row: (str(row["campaign_id"]), str(row["timestamp"]), str(row["folder"])))
    return rows


def render_index(evidence_root: str | Path) -> str:
    root = Path(evidence_root)
    rows = discover_campaigns(root)
    header = (
        "| Campaign | Profile | Decision | Status | Commit | Branch | Timestamp | Manifest | Report | Summary | Pairs | Anomalies | Schema |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    body = []
    for row in rows:
        paths = row["paths"]
        body.append(
            "| {campaign_id} | {profile} | {decision} | {status} | {git_commit} | {branch} | {timestamp} | {manifest} | {decision_report} | {summary_json} | {paired_comparison_csv} | {anomaly_summary} | {schema_status} |".format(
                **row,
                **paths,
            )
        )
    if not body:
        body.append("| No campaigns found | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |")
    return "# Campaign Evidence Index\n\n" + header + "\n".join(body) + "\n"


def update_campaign_index(evidence_root: str | Path) -> Path:
    root = Path(evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign_index.md"
    path.write_text(render_index(root), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update campaign evidence index.")
    parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args(argv)
    print(update_campaign_index(args.evidence_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
