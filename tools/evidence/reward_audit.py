"""Reward leakage audit helpers for evidence reports."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from tools.evidence.schema import truthy_number


AMBIGUOUS_REWARD_TOKENS = ("reward", "selection_pressure")


def _is_disabled_control_field(name: str) -> bool:
    return name.endswith("_disabled_flag") or name.endswith("_disabled")


def audit_reward_leakage(rows: Iterable[Mapping[str, Any]], leakage_columns: Iterable[str]) -> dict[str, Any]:
    expected = tuple(leakage_columns)
    active: list[dict[str, str]] = []
    inactive_controls: set[str] = set()
    missing: set[str] = set(expected)
    ambiguous: set[str] = set()
    checked: set[str] = set()

    for row in rows:
        metrics = row.get("final_metrics") if isinstance(row.get("final_metrics"), Mapping) else {}
        run_id = str(row.get("run_id") or row.get("scenario_id") or "")
        for column in expected:
            if column in metrics:
                missing.discard(column)
                checked.add(column)
                if truthy_number(metrics.get(column)):
                    active.append({"run_id": run_id, "field": column, "value": str(metrics.get(column))})
        for column in metrics:
            name = str(column)
            if name not in expected and any(token in name for token in AMBIGUOUS_REWARD_TOKENS):
                if _is_disabled_control_field(name):
                    if truthy_number(metrics.get(column)):
                        inactive_controls.add(name)
                    else:
                        active.append({"run_id": run_id, "field": name, "value": str(metrics.get(column))})
                    continue
                ambiguous.add(name)
                if truthy_number(metrics.get(column)):
                    active.append({"run_id": run_id, "field": name, "value": str(metrics.get(column))})

    severity = "ok"
    if active:
        severity = "fatal"
    elif missing or ambiguous:
        severity = "ambiguous"

    return {
        "ok": severity == "ok",
        "severity": severity,
        "checked_fields": sorted(checked),
        "missing_fields": sorted(missing),
        "active_fields": sorted(active, key=lambda item: (item["field"], item["run_id"])),
        "inactive_control_fields": sorted(inactive_controls),
        "ambiguous_fields": sorted(ambiguous),
    }
