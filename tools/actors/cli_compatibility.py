"""Versioned compatibility contract for the operator-facing EIEL CLI.

The manifest distinguishes canonical commands, compatibility aliases, and
removed commands.  It is deliberately data-only: compatibility never grants
lifecycle authority and never guesses a replacement for an unknown spelling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CLI_COMPATIBILITY_SCHEMA_VERSION = "eiel_cli_compatibility_manifest_v1"

CLI_COMPATIBILITY_MANIFEST: dict[str, Any] = {
    "schema_version": CLI_COMPATIBILITY_SCHEMA_VERSION,
    "tool": "python -m tools.eiel",
    "commands": [
        {
            "canonical_command": "start",
            "aliases": [],
            "operation": "INSPECTION_ONLY",
            "deprecated_since": None,
            "replacement": None,
            "warning_period": None,
            "removal_not_before": None,
        },
        {
            "canonical_command": "resume-status",
            "aliases": ["resume"],
            "operation": "INSPECTION_ONLY",
            "deprecated_since": "1.18.0-i-interim",
            "replacement": "resume-status",
            "warning_period": "one minor release",
            "removal_not_before": "1.19.0",
        },
        {
            "canonical_command": "status",
            "aliases": [],
            "operation": "INSPECTION_ONLY",
            "deprecated_since": None,
            "replacement": None,
            "warning_period": None,
            "removal_not_before": None,
        },
        {
            "canonical_command": "audit",
            "aliases": [],
            "operation": "INSPECTION_ONLY",
            "deprecated_since": None,
            "replacement": None,
            "warning_period": None,
            "removal_not_before": None,
        },
        {
            "canonical_command": "handoff",
            "aliases": [],
            "operation": "READ_ONLY_EXPORT",
            "deprecated_since": None,
            "replacement": None,
            "warning_period": None,
            "removal_not_before": None,
        },
    ],
    "removed_commands": [
        {
            "command": "release-dependencies",
            "removed_since": "1.16.0",
            "replacement": "workManager refresh",
            "reason": "Dependency readiness is a read-only projection; refresh reports it and does not release Work.",
        }
    ],
}


def manifest_snapshot() -> dict[str, Any]:
    return {
        "schema_version": CLI_COMPATIBILITY_MANIFEST["schema_version"],
        "tool": CLI_COMPATIBILITY_MANIFEST["tool"],
        "commands": [dict(row) for row in CLI_COMPATIBILITY_MANIFEST["commands"]],
        "removed_commands": [dict(row) for row in CLI_COMPATIBILITY_MANIFEST["removed_commands"]],
    }


def command_entry(command: str) -> dict[str, Any] | None:
    value = str(command or "").strip()
    for row in CLI_COMPATIBILITY_MANIFEST["commands"]:
        item = dict(row)
        if value == item["canonical_command"] or value in item.get("aliases", []):
            item["invoked_as"] = value
            item["is_compatibility_alias"] = value != item["canonical_command"]
            return item
    return None


def removed_command_entry(command: str) -> dict[str, Any] | None:
    value = str(command or "").strip()
    for row in CLI_COMPATIBILITY_MANIFEST["removed_commands"]:
        if value == row.get("command"):
            return dict(row)
    return None


def compatibility_warning(entry: Mapping[str, Any] | None) -> str | None:
    if not entry or not bool(entry.get("is_compatibility_alias")):
        return None
    replacement = str(entry.get("replacement") or entry.get("canonical_command") or "").strip()
    removal = str(entry.get("removal_not_before") or "a future release")
    return (
        f"{entry.get('invoked_as')} is a compatibility alias for read-only inspection; "
        f"use {replacement}. It will not be removed before {removal}."
    )


__all__ = [
    "CLI_COMPATIBILITY_MANIFEST",
    "CLI_COMPATIBILITY_SCHEMA_VERSION",
    "command_entry",
    "compatibility_warning",
    "manifest_snapshot",
    "removed_command_entry",
]
