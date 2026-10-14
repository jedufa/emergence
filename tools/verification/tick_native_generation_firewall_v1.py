#!/usr/bin/env python3
"""Path- and symbol-aware firewall for active tick-native authority.

This deliberately does not grep the repository for the word ``generation``.
It parses the active Python symbols, checks their field/attribute dependencies,
and verifies the exact C++ compatibility guard that bounds one native tick
span. Historical readers and named heritage adapters remain outside this
active-symbol fence and are governed by the dependency inventory.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "eiel_tick_native_generation_firewall_receipt_v1"
FORBIDDEN_PUBLIC_FIELD_FRAGMENTS = (
    "generation",
    "ticks_per_generation",
    "steps_per_generation",
)


@dataclass(frozen=True)
class Finding:
    path: str
    symbol: str
    code: str
    detail: str

    def payload(self) -> dict[str, str]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "code": self.code,
            "detail": self.detail,
        }


def _tree(root: Path, relative: str) -> ast.Module:
    path = root / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definition(tree: ast.AST, qualified_name: str) -> ast.AST:
    parts = qualified_name.split(".")
    nodes: Iterable[ast.AST] = getattr(tree, "body", ())
    current: ast.AST | None = None
    for part in parts:
        current = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if current is None:
            raise LookupError(qualified_name)
        nodes = getattr(current, "body", ())
    return current


def _attribute_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _chains(node: ast.AST) -> set[str]:
    return {
        chain
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Attribute)
        and (chain := _attribute_chain(candidate)) is not None
    }


def _class_fields(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.ClassDef):
        return set()
    return {
        statement.target.id
        for statement in node.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
    }


def _function_parameters(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    args = node.args
    return {
        argument.arg
        for argument in (*args.posonlyargs, *args.args, *args.kwonlyargs)
    }


def _if_branch(node: ast.AST, marker: str) -> ast.If:
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.If) and marker in ast.unparse(candidate.test):
            return candidate
    raise LookupError(f"conditional branch containing {marker!r}")


def _check_public_contracts(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    contracts = (
        (
            "engine/campaigns/tick_native.py",
            "TickNativeExecutionRequestV1",
            {"legacy_turnover_interval_ticks"},
        ),
        (
            "engine/campaigns/tick_native_control.py",
            "TickNativeControlV1",
            {"legacy_turnover_interval_ticks"},
        ),
    )
    for relative, symbol, heritage_fields in contracts:
        node = _definition(_tree(root, relative), symbol)
        for field in sorted(_class_fields(node)):
            if field in heritage_fields:
                continue
            if any(fragment in field for fragment in FORBIDDEN_PUBLIC_FIELD_FRAGMENTS):
                findings.append(
                    Finding(
                        relative,
                        symbol,
                        "ACTIVE_RUNTIME_AUTHORITY_FORBIDDEN",
                        f"public tick-native field {field!r} is generation-shaped",
                    )
                )
    return findings


def _check_provider_symbols(root: Path) -> list[Finding]:
    relative = "engine/campaigns/providers.py"
    tree = _tree(root, relative)
    findings: list[Finding] = []
    planner_name = "ProductionEngineLifecycleV1._absolute_tick_segments_v1"
    planner = _definition(tree, planner_name)
    planner_chains = _chains(planner)
    native_planner_branch = _if_branch(planner, "native_tick_window is not None")
    native_planner_chains = _chains(native_planner_branch)
    required = {
        "request.total_ticks",
        "request.source_window_ticks",
        "request.native_dispatch_window_ticks",
        "request.resident_owner_span_ticks",
    }
    for chain in sorted(required - planner_chains):
        findings.append(
            Finding(relative, planner_name, "TICK_BOUNDARY_DEPENDENCY_MISSING", chain)
        )
    for chain in sorted(
        native_planner_chains.intersection(
            {
                "request.generations",
                "request.checkpoint_milestones",
                "request.max_internal_lifecycle_segments_per_dispatch",
            }
        )
    ):
        findings.append(
            Finding(
                relative,
                planner_name,
                "ACTIVE_RUNTIME_AUTHORITY_FORBIDDEN",
                chain,
            )
        )

    allocation_name = "ProductionEngineLifecycleV1._agent_id_allocation_generation_count"
    allocation = _definition(tree, allocation_name)
    native_allocation_branch = _if_branch(allocation, "request.total_ticks is not None")
    allocation_chains = _chains(native_allocation_branch)
    for chain in sorted(
        allocation_chains.intersection(
            {
                "request.generations",
                "request.max_internal_lifecycle_segments_per_dispatch",
            }
        )
    ):
        findings.append(
            Finding(
                relative,
                allocation_name,
                "ACTIVE_MEMORY_AUTHORITY_FORBIDDEN",
                chain,
            )
        )
    for chain in (
        "control.native_dispatch_window_ticks",
        "control.source_window_ticks",
    ):
        # The implementation may use mapping access rather than attributes;
        # require the literal field name in that exact AST symbol either way.
        if chain.rsplit(".", 1)[1] not in ast.unparse(allocation):
            findings.append(
                Finding(relative, allocation_name, "TICK_ALLOCATION_BOUND_MISSING", chain)
            )

    identity_name = "_checkpoint_identity_contract_v1"
    identity = _definition(tree, identity_name)
    parameters = _function_parameters(identity)
    if "next_absolute_tick" not in parameters:
        findings.append(
            Finding(
                relative,
                identity_name,
                "ACTIVE_REPORT_CHECKPOINT_AUTHORITY_FORBIDDEN",
                "checkpoint identity lacks next_absolute_tick",
            )
        )
    identity_source = ast.unparse(identity)
    for marker in ("absolute_tick_v1", "next_absolute_tick"):
        if marker not in identity_source:
            findings.append(
                Finding(relative, identity_name, "CHECKPOINT_IDENTITY_BINDING_MISSING", marker)
            )
    return findings


def _check_recursive_payload_firewall(root: Path) -> list[Finding]:
    relative = "engine/campaigns/tick_native_control.py"
    node = _definition(_tree(root, relative), "assert_tick_native_plan_payload_v1")
    source = ast.unparse(node)
    findings: list[Finding] = []
    if "walk" not in source and "recurse" not in source:
        findings.append(
            Finding(
                relative,
                "assert_tick_native_plan_payload_v1",
                "NESTED_PAYLOAD_FIREWALL_MISSING",
                "active payload validation is not recursive",
            )
        )
    for container in ("Mapping", "list", "tuple"):
        if container not in source:
            findings.append(
                Finding(
                    relative,
                    "assert_tick_native_plan_payload_v1",
                    "NESTED_PAYLOAD_CONTAINER_MISSING",
                    container,
                )
            )
    return findings


def _check_native_span_guard(root: Path) -> list[Finding]:
    relative = "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
    source = (root / relative).read_text(encoding="utf-8")
    # Strip comments before inspecting the exact guard; comments are not
    # executable authority and must not satisfy the firewall.
    source = re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.MULTILINE | re.DOTALL)
    start = source.find("const bool exact_partial_tick_span")
    end = source.find("TORCH_CHECK", start)
    guard = source[start:end] if start >= 0 and end > start else ""
    required_patterns = (
        r"start_tick\s*>=\s*0",
        r"ticks\s*<=\s*160\s*-\s*start_tick",
        r"generation_count\s*==\s*1",
    )
    return [
        Finding(relative, "exact_partial_tick_span", "NATIVE_TICK_SPAN_GUARD_MISSING", pattern)
        for pattern in required_patterns
        if re.search(pattern, guard) is None
    ]


def run_firewall(root: Path) -> dict[str, object]:
    checks = (
        _check_public_contracts,
        _check_provider_symbols,
        _check_recursive_payload_firewall,
        _check_native_span_guard,
    )
    findings: list[Finding] = []
    for check in checks:
        try:
            findings.extend(check(root))
        except (LookupError, OSError, SyntaxError) as exc:
            findings.append(
                Finding("<firewall>", check.__name__, "FIREWALL_INSPECTION_FAILED", str(exc))
            )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "repository_root": str(root),
        "status": "PASS" if not findings else "FAIL",
        "path_symbol_aware": True,
        "finding_count": len(findings),
        "findings": [finding.payload() for finding in findings],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_firewall(args.repo_root.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
