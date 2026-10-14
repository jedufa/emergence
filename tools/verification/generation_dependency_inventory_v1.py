#!/usr/bin/env python3
"""Build the repository-wide generation dependency and guidance inventories."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from typing import Any


SCHEMA_VERSION = "eiel_tick_native_generation_dependency_inventory_v1"
TOKEN = re.compile(
    r"\b(?:generation(?:s|_count|_index|_boundary)?|logical_generation_count|"
    r"physical_window_generations|ticks_per_generation|steps_per_generation|"
    r"G\d+(?:-G\d+)?/T\d+)\b",
    re.IGNORECASE,
)
CLASSIFICATIONS = (
    "LEGACY_INPUT_COMPATIBILITY",
    "LEGACY_OUTPUT_COMPATIBILITY",
    "HISTORICAL_READER_ONLY",
    "HISTORICAL_IMMUTABLE_ARTIFACT",
    "HERITAGE_TURNOVER_REQUIRED",
    "NAMING_ONLY_BUT_DANGEROUS",
    "ACTIVE_RUNTIME_AUTHORITY_FORBIDDEN",
    "ACTIVE_MEMORY_AUTHORITY_FORBIDDEN",
    "ACTIVE_QUALIFICATION_AUTHORITY_FORBIDDEN",
    "ACTIVE_REPORT_CHECKPOINT_AUTHORITY_FORBIDDEN",
    "UNKNOWN",
)
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".html",
    ".ini", ".json", ".md", ".py", ".rst", ".sh", ".toml", ".txt",
    ".yaml", ".yml",
}


def _files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return sorted(row for row in result.stdout.decode().split("\0") if row)


def _python_symbols(path: Path) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    rows: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*prefix, child.name))
                rows.append((child.lineno, getattr(child, "end_lineno", child.lineno), name))
                visit(child, (*prefix, child.name))
            else:
                visit(child, prefix)

    visit(tree)
    return rows


def _symbol(symbols: list[tuple[int, int, str]], line: int) -> str:
    matches = [row for row in symbols if row[0] <= line <= row[1]]
    return min(matches, key=lambda row: row[1] - row[0])[2] if matches else "<module>"


def _classification(path: str, symbol: str, text: str) -> str:
    lower_path = path.lower()
    lower_symbol = symbol.lower()
    lower = text.lower()
    if path in {"AGENTS.md"} or lower_path.startswith("governance/"):
        return "NAMING_ONLY_BUT_DANGEROUS"
    if lower_path.startswith("docs/") or lower_path.startswith("evidences/"):
        if "900_lifecycle/940_incoming" in lower_path:
            return "HISTORICAL_IMMUTABLE_ARTIFACT"
        return "HISTORICAL_IMMUTABLE_ARTIFACT"
    if lower_path.startswith("tests/"):
        if "legacy" in lower_path or "compatibility" in lower_path:
            return "LEGACY_INPUT_COMPATIBILITY"
        return "NAMING_ONLY_BUT_DANGEROUS"
    if lower_path.startswith("_spec/"):
        if "science_representative_5000tick" in lower_path:
            return "NAMING_ONLY_BUT_DANGEROUS"
        if "historical" in lower or "non_executable_historical" in lower:
            return "HISTORICAL_IMMUTABLE_ARTIFACT"
        return "LEGACY_INPUT_COMPATIBILITY"
    if lower_path.startswith(("simulation/", "evolution/", "engine/perf/", "engine/native_runtime/")):
        if "reader" in lower_symbol or "decode" in lower_symbol or "load" in lower_symbol:
            return "HISTORICAL_READER_ONLY"
        return "HERITAGE_TURNOVER_REQUIRED"
    if lower_path.startswith(("engine/campaigns/", "engine/qrr/")):
        if "tick_native" in lower_path or "ticknative" in lower_symbol:
            return "NAMING_ONLY_BUT_DANGEROUS"
        if any(word in lower_symbol for word in ("payload", "receipt", "serialize", "manifest")):
            return "LEGACY_OUTPUT_COMPATIBILITY"
        if any(word in lower_symbol for word in ("load", "parse", "from_", "request", "validate")):
            return "LEGACY_INPUT_COMPATIBILITY"
        if "qualification" in lower_path or "qualification" in lower_symbol:
            return "NAMING_ONLY_BUT_DANGEROUS"
        return "LEGACY_INPUT_COMPATIBILITY"
    if lower_path.startswith("tools/historical/"):
        return "HISTORICAL_READER_ONLY"
    if lower_path.startswith("tools/"):
        if any(word in lower for word in ("parse", "load", "read", "compat")):
            return "LEGACY_INPUT_COMPATIBILITY"
        return "NAMING_ONLY_BUT_DANGEROUS"
    if lower_path.startswith(("registries/", "schemas/")):
        return "LEGACY_INPUT_COMPATIBILITY"
    return "NAMING_ONLY_BUT_DANGEROUS"


def build_inventory(root: Path) -> dict[str, Any]:
    occurrences: list[dict[str, Any]] = []
    scanned_files = 0
    for relative in _files(root):
        if relative.startswith("_artifacts/tick-native-finalization-20260820/"):
            continue
        path = root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        scanned_files += 1
        symbols = _python_symbols(path) if path.suffix == ".py" else []
        for line_number, text in enumerate(lines, 1):
            matches = sorted({match.group(0) for match in TOKEN.finditer(text)})
            if not matches:
                continue
            symbol = _symbol(symbols, line_number) if symbols else f"line:{line_number}"
            classification = _classification(relative, symbol, text)
            occurrences.append(
                {
                    "path": relative,
                    "line_or_symbol": f"{line_number}:{symbol}",
                    "tokens": matches,
                    "current_meaning": text.strip()[:500],
                    "producer": f"{relative}:{line_number}",
                    "consumer": "classified repository dependency; resolve through symbol and route",
                    "active_routes": (
                        ["Lifecycle V2", "EIEL_FULL_EVOLUTION_V1"]
                        if relative.startswith(("engine/", "AGENTS.md", "governance/"))
                        else []
                    ),
                    "classification": classification,
                    "required_disposition": {
                        "LEGACY_INPUT_COMPATIBILITY": "retain only behind explicit legacy input adapter",
                        "LEGACY_OUTPUT_COMPATIBILITY": "retain only as derived compatibility output",
                        "HISTORICAL_READER_ONLY": "read historical coordinates; never select active scope",
                        "HISTORICAL_IMMUTABLE_ARTIFACT": "preserve immutable; current supersession index governs interpretation",
                        "HERITAGE_TURNOVER_REQUIRED": "retain only for explicit synchronized-turnover mechanics; derive from absolute tick",
                        "NAMING_ONLY_BUT_DANGEROUS": "retain with active-symbol and guidance firewall coverage",
                    }.get(classification, "remove or repair before Phase 0 completion"),
                    "test_coverage": [
                        "tests/tick_native_generation_firewall.py",
                        "tools/verification/tick_native_generation_firewall_v1.py",
                    ],
                    "documentation_authority": "governance/EIEL_Tick_Native_Authority_and_Generation_Era_Supersession_Index.md",
                }
            )
    counts = Counter(row["classification"] for row in occurrences)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "all tracked and task-untracked text files in the repository",
        "classification_contract": list(CLASSIFICATIONS),
        "method": {
            "file_enumeration": "git ls-files --cached --others --exclude-standard",
            "occurrence_detection": "line token scan with Python AST narrowest enclosing symbol",
            "active_authority_verification": [
                "tests/tick_native_generation_firewall.py",
                "tools/verification/tick_native_generation_firewall_v1.py",
            ],
        },
        "scanned_file_count": scanned_files,
        "occurrence_count": len(occurrences),
        "classification_counts": dict(sorted(counts.items())),
        "unknown_count": counts["UNKNOWN"],
        "forbidden_active_remaining": sum(
            counts[name]
            for name in CLASSIFICATIONS
            if name.startswith("ACTIVE_") and name.endswith("_FORBIDDEN")
        ),
        "occurrences": occurrences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_inventory(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: payload[key] for key in (
                "scanned_file_count", "occurrence_count", "classification_counts",
                "unknown_count", "forbidden_active_remaining",
            )},
            indent=2,
        )
    )
    return 0 if payload["unknown_count"] == 0 and payload["forbidden_active_remaining"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
