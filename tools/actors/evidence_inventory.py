"""Read-only inventory of attached and orphaned Evidence Decision records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tools.actors.canonical import canonical_sha256
from tools.actors.portfolio import evidence_store, work_store


def evidence_attachment_inventory(state_root: str | Path | None = None) -> dict[str, Any]:
    work_rows = []
    works = work_store(state_root)
    attached_by: dict[str, list[str]] = {}
    for summary in works.list_read_only():
        try:
            record = works.get_read_only(str(summary.get("record_id") or ""))
        except Exception:
            continue
        work_id = str(record.get("record_id") or "")
        refs = [str(value).strip().upper() for value in dict(record.get("payload") or {}).get("evidence_assessment_ids") or [] if str(value).strip()]
        work_rows.append({"work_id": work_id, "evidence_assessment_ids": sorted(set(refs))})
        for assessment_id in refs:
            attached_by.setdefault(assessment_id, []).append(work_id)

    rows: list[dict[str, Any]] = []
    store = evidence_store(state_root)
    for summary in store.list_read_only():
        assessment_id = str(summary.get("record_id") or "")
        try:
            record = store.get_read_only(assessment_id)
            payload = dict(record.get("payload") or {})
        except Exception:
            payload = {}
            record = dict(summary)
        attached = sorted(set(attached_by.get(assessment_id, [])))
        tool_validation = bool(payload.get("synthetic_fixture") or payload.get("tool_validation"))
        rows.append({
            "assessment_id": assessment_id,
            "status": record.get("status"),
            "record_sha256": record.get("record_sha256"),
            "attachment_status": "ATTACHED" if attached else ("TOOL_VALIDATION_EXCEPTION" if tool_validation else "ORPHAN"),
            "attached_work_ids": attached,
            "attach_command": None if attached or tool_validation else f"python -m tools.eiel workManager update <WORK-ID> --evidence-ref {assessment_id}",
        })
    rows.sort(key=lambda row: str(row.get("assessment_id") or ""))
    body = {
        "schema_version": "eiel_evidence_attachment_inventory_v1",
        "records": rows,
        "counts": {
            "attached": sum(1 for row in rows if row["attachment_status"] == "ATTACHED"),
            "orphan": sum(1 for row in rows if row["attachment_status"] == "ORPHAN"),
            "tool_validation_exception": sum(1 for row in rows if row["attachment_status"] == "TOOL_VALIDATION_EXCEPTION"),
        },
        "work_reference_rows": work_rows,
    }
    body["inventory_sha256"] = canonical_sha256(body)
    return body


__all__ = ["evidence_attachment_inventory"]
