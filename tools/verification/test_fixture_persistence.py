"""Static gate for explicit test-only actor-store persistence exceptions."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

ALLOWED_CLASSES = {
    "intentional_fault_injection",
    "historical_or_cross_manager_fixture",
    "store_contract_unit",
}


def direct_put_calls(repo_root: str | Path) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "tests").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "put"
            ):
                continue
            current: ast.AST = node
            function = "<module>"
            while current in parents:
                current = parents[current]
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function = current.name
                    break
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "function": function,
                "line": int(node.lineno),
            })
    return rows


def validate_allowlist(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = root / "governance" / "EIEL_TEST_DIRECT_STORE_ALLOWLIST_V1_19.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = list(raw.get("entries") or [])
    observed = direct_put_calls(root)
    key = lambda row: (str(row.get("path")), str(row.get("function")), int(row.get("line") or 0))
    allowed = {key(row): row for row in entries}
    observed_map = {key(row): row for row in observed}
    invalid_classes = [row for row in entries if row.get("classification") not in ALLOWED_CLASSES]
    return {
        "schema_version": "eiel_test_fixture_persistence_audit_v1",
        "observed_count": len(observed),
        "allowlisted_count": len(entries),
        "unapproved": [observed_map[item] for item in sorted(set(observed_map) - set(allowed))],
        "stale_allowlist_entries": [allowed[item] for item in sorted(set(allowed) - set(observed_map))],
        "invalid_classifications": invalid_classes,
        "ok": not invalid_classes and set(observed_map) == set(allowed),
    }


__all__ = ["ALLOWED_CLASSES", "direct_put_calls", "validate_allowlist"]
