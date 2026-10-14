#!/usr/bin/env python3
"""Fail-closed AST import ratchet for the frozen EIEL modular target DAG."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


LEGACY_ENGINE_NAMESPACES = frozenset(
    {
        "agents",
        "artifact_catalog",
        "authoritative_state",
        "campaigns",
        "capacity_profiles",
        "device_selection",
        "dispatch_planning",
        "env",
        "evolution",
        "monitoring",
        "native_runtime",
        "perf",
        "planning_receipts",
        "qrr",
        "qualification_plans",
        "runtime_roots",
        "simulation",
        "validation_plans",
        "wrc",
    }
)

KNOWN_TARGET_PACKAGES = frozenset(
    {
        "emergence_contracts",
        "emergence_runtime",
        "emergence_worldkit",
        "emergence_agentkit",
        "emergence_adaptation",
        "emergence_lab",
        "emergence_evidence",
        "emergence_assurance",
        "emergence_ops",
        "eiel_composition",
    }
)

REQUIRED_EXCEPTION_FIELDS = frozenset(
    {
        "source",
        "target",
        "path",
        "ceiling",
        "owner",
        "reason",
        "expires_after_slice",
        "validation_obligation",
    }
)


class GuardConfigurationError(ValueError):
    """The ratchet baseline is incomplete, contradictory, or non-monotonic."""


@dataclass(frozen=True, slots=True)
class ImportOccurrence:
    source_module: str
    target_module: str
    source_family: str
    target_family: str
    source_path: str
    line: int


@dataclass(frozen=True, slots=True)
class AuditResult:
    module_count: int
    resolved_internal_import_statements: int
    legacy_module_count: int
    target_module_count: int
    legacy_edges: tuple[tuple[str, str, int], ...]
    target_edges: tuple[tuple[str, str, int], ...]
    legacy_sccs: tuple[tuple[str, ...], ...]
    target_sccs: tuple[tuple[str, ...], ...]
    parse_errors: tuple[tuple[str, str], ...]
    unknown_target_roots: tuple[str, ...]
    target_external_imports: tuple[tuple[str, str, str, int], ...]
    occurrences: tuple[ImportOccurrence, ...]
    compatibility_edges: tuple[tuple[str, str, int], ...] = ()

    def deterministic_dict(self) -> dict[str, object]:
        return {
            "module_count": self.module_count,
            "resolved_internal_import_statements": self.resolved_internal_import_statements,
            "legacy_module_count": self.legacy_module_count,
            "target_module_count": self.target_module_count,
            "legacy_edges": [
                {"source": source, "target": target, "count": count}
                for source, target, count in self.legacy_edges
            ],
            "target_edges": [
                {"source": source, "target": target, "count": count}
                for source, target, count in self.target_edges
            ],
            "legacy_sccs": [list(component) for component in self.legacy_sccs],
            "target_sccs": [list(component) for component in self.target_sccs],
            "compatibility_edges": [
                {"source": source, "target": target, "count": count}
                for source, target, count in self.compatibility_edges
            ],
            "parse_errors": [
                {"path": path, "error": error} for path, error in self.parse_errors
            ],
            "unknown_target_roots": list(self.unknown_target_roots),
            "target_external_imports": [
                {
                    "source": source,
                    "import_root": import_root,
                    "path": path,
                    "line": line,
                }
                for source, import_root, path, line in self.target_external_imports
            ],
        }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def edge_key(source: str, target: str) -> str:
    return f"{source} -> {target}"


def split_edge_key(value: str) -> tuple[str, str]:
    parts = value.split(" -> ")
    if len(parts) != 2 or not all(parts):
        raise GuardConfigurationError(f"invalid legacy edge key: {value!r}")
    return parts[0], parts[1]


def module_name(repo_root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(repo_root)
    if relative.parts[0] == "engine":
        parts = relative.with_suffix("").parts[1:]
    else:
        parts = relative.with_suffix("").parts
    module = ".".join(parts)
    if module.endswith(".__init__"):
        module = module[:-9]
    return module, relative.parts[0]


def resolve_from(current: str, path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    parts = current.split(".")
    if path.name != "__init__.py":
        parts = parts[:-1]
    base = parts[: max(0, len(parts) - (node.level - 1))]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def best_repo_match(name: str, modules: set[str]) -> str | None:
    if name.startswith("engine."):
        name = name[len("engine.") :]
    parts = name.split(".")
    for index in range(len(parts), 0, -1):
        candidate = ".".join(parts[:index])
        if candidate in modules or any(
            module.startswith(candidate + ".") for module in modules
        ):
            return candidate
    return None


def module_family(module: str, origin: Mapping[str, str]) -> str:
    first = module.split(".", 1)[0]
    if first in KNOWN_TARGET_PACKAGES:
        return first
    if origin.get(module) == "engine" or first in LEGACY_ENGINE_NAMESPACES:
        return first
    if first == "tools" and "." in module:
        return ".".join(module.split(".")[:2])
    return first


def strongly_connected_components(
    nodes: Iterable[str], adjacency: Mapping[str, set[str]]
) -> tuple[tuple[str, ...], ...]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)
        for destination in sorted(adjacency.get(node, set())):
            if destination not in index:
                visit(destination)
                low[node] = min(low[node], low[destination])
            elif destination in on_stack:
                low[node] = min(low[node], index[destination])
        if low[node] == index[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                components.append(tuple(sorted(component)))

    for node in sorted(nodes):
        if node not in index:
            visit(node)
    return tuple(sorted(components, key=lambda item: (-len(item), item)))


def sccs_from_edges(
    edges: Iterable[tuple[str, str, int]],
) -> tuple[tuple[str, ...], ...]:
    nodes: set[str] = set()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target, count in edges:
        if count <= 0:
            continue
        nodes.update((source, target))
        adjacency[source].add(target)
    return strongly_connected_components(nodes, adjacency)


def _unknown_target_roots(repo_root: Path) -> tuple[str, ...]:
    unknown: list[str] = []
    for child in sorted(repo_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        looks_like_target = child.name.startswith("emergence_") or child.name.startswith(
            "eiel_"
        )
        if (
            looks_like_target
            and child.name not in KNOWN_TARGET_PACKAGES
            and any(child.rglob("*.py"))
        ):
            unknown.append(child.name)
    return tuple(unknown)


def _compatibility_aliases(root: Path, files: Sequence[Path]) -> dict[str, str]:
    """Validate exact, inert re-exports; never exempt implementation imports.

    The legacy graph still counts every caller of a compatibility module. Its
    one declared outward re-export is reported separately; the canonical owner
    is audited under the unchanged target DAG. No legacy ceiling is increased.
    """
    manifest = root / "tools/architecture/eiel_modular_compatibility_aliases_v1.json"
    if not manifest.exists():
        return {}
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if set(raw) != {"schema_version", "aliases"} or raw["schema_version"] != "eiel_modular_compatibility_aliases_v1":
            raise ValueError("unsupported alias manifest")
        aliases = raw["aliases"]
        if not isinstance(aliases, list):
            raise ValueError("aliases must be a list")
        covered = {str(path.relative_to(root)) for path in files}
        result: dict[str, str] = {}
        for row in aliases:
            if set(row) != {"path", "target", "names", "sha256"}:
                raise ValueError("invalid alias fields")
            path, target, names = row["path"], row["target"], row["names"]
            if path not in covered or path in result:
                raise ValueError(f"missing, unaudited or duplicate alias: {path}")
            if path.split("/", 1)[0] in KNOWN_TARGET_PACKAGES:
                raise ValueError("target packages cannot masquerade as legacy aliases")
            if target.split(".")[0] not in KNOWN_TARGET_PACKAGES:
                raise ValueError("alias must point to an accepted target package")
            if target.split(".")[0] == "eiel_composition":
                raise ValueError("aliases cannot target the legacy-dependent composition boundary")
            if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n.isidentifier() for n in names):
                raise ValueError("alias names must be explicit unique identifiers")
            owner = root.joinpath(*target.split(".")).with_suffix(".py")
            if str(owner.relative_to(root)) not in covered:
                raise ValueError(f"canonical alias owner is not audited: {target}")
            data = (root / path).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"compatibility alias content changed: {path}")
            body = ast.parse(data).body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            if len(body) != 2:
                raise ValueError(f"compatibility alias contains executable code: {path}")
            imp, export = body
            if not isinstance(imp, ast.ImportFrom) or imp.level or imp.module != target or [(a.name, a.asname) for a in imp.names] != [(n, None) for n in names]:
                raise ValueError(f"compatibility alias imports changed: {path}")
            if not isinstance(export, ast.Assign) or len(export.targets) != 1 or not isinstance(export.targets[0], ast.Name) or export.targets[0].id != "__all__" or ast.literal_eval(export.value) != names:
                raise ValueError(f"compatibility alias exports changed: {path}")
            result[path] = target
        return result
    except (OSError, ValueError, TypeError, KeyError, SyntaxError) as error:
        raise GuardConfigurationError(f"invalid compatibility aliases: {error}") from error


def audit_repository(repo_root: Path, *, audit_roots: Sequence[str]) -> AuditResult:
    root = repo_root.resolve()
    include_roots = set(audit_roots) | {
        name for name in KNOWN_TARGET_PACKAGES if (root / name).is_dir()
    }
    files: list[Path] = []
    module_by_file: dict[Path, str] = {}
    origin: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] not in include_roots:
            continue
        module, top = module_name(root, path)
        files.append(path)
        module_by_file[path] = module
        origin[module] = top

    compatibility_aliases = _compatibility_aliases(root, files)
    modules = set(module_by_file.values())
    parse_errors: list[tuple[str, str]] = []
    occurrences: list[ImportOccurrence] = []
    target_external_imports: list[tuple[str, str, str, int]] = []
    for path in files:
        current = module_by_file[path]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError) as error:
            parse_errors.append((str(path.relative_to(root)), str(error)))
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = resolve_from(current, path, node)
                if base:
                    names = [base]
            for name in names:
                target = best_repo_match(name, modules)
                if target is None:
                    source_family = module_family(current, origin)
                    import_root = name.split(".", 1)[0]
                    if (
                        source_family in KNOWN_TARGET_PACKAGES
                        and import_root not in sys.stdlib_module_names
                        and import_root != "__future__"
                    ):
                        target_external_imports.append(
                            (
                                source_family,
                                import_root,
                                str(path.relative_to(root)),
                                getattr(node, "lineno", 0),
                            )
                        )
                    continue
                occurrences.append(
                    ImportOccurrence(
                        source_module=current,
                        target_module=target,
                        source_family=module_family(current, origin),
                        target_family=module_family(target, origin),
                        source_path=str(path.relative_to(root)),
                        line=getattr(node, "lineno", 0),
                    )
                )

    legacy_counter: Counter[tuple[str, str]] = Counter()
    target_counter: Counter[tuple[str, str]] = Counter()
    compatibility_counter: Counter[tuple[str, str]] = Counter()
    for occurrence in occurrences:
        if occurrence.source_family == occurrence.target_family:
            continue
        key = (occurrence.source_family, occurrence.target_family)
        if compatibility_aliases.get(occurrence.source_path) == occurrence.target_module:
            compatibility_counter[key] += 1
        elif occurrence.source_family in KNOWN_TARGET_PACKAGES:
            target_counter[key] += 1
        else:
            legacy_counter[key] += 1

    legacy_edges = tuple(
        (source, target, count)
        for (source, target), count in sorted(legacy_counter.items())
    )
    target_edges = tuple(
        (source, target, count)
        for (source, target), count in sorted(target_counter.items())
    )
    target_modules = sum(
        1 for module in modules if module_family(module, origin) in KNOWN_TARGET_PACKAGES
    )
    return AuditResult(
        module_count=len(files),
        resolved_internal_import_statements=len(occurrences),
        legacy_module_count=len(files) - target_modules,
        target_module_count=target_modules,
        legacy_edges=legacy_edges,
        target_edges=target_edges,
        compatibility_edges=tuple(
            (source, target, count)
            for (source, target), count in sorted(compatibility_counter.items())
        ),
        legacy_sccs=sccs_from_edges(legacy_edges),
        target_sccs=sccs_from_edges(target_edges),
        parse_errors=tuple(sorted(parse_errors)),
        unknown_target_roots=_unknown_target_roots(root),
        target_external_imports=tuple(sorted(target_external_imports)),
        occurrences=tuple(
            sorted(
                occurrences,
                key=lambda item: (
                    item.source_path,
                    item.line,
                    item.source_module,
                    item.target_module,
                ),
            )
        ),
    )


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _baseline_edges(baseline: Mapping[str, object]) -> dict[tuple[str, str], int]:
    raw_edges = baseline.get("legacy_edges")
    if not isinstance(raw_edges, dict):
        raise GuardConfigurationError("legacy_edges must be an object")
    edges: dict[tuple[str, str], int] = {}
    for raw_key, raw_record in raw_edges.items():
        if not isinstance(raw_key, str) or not isinstance(raw_record, dict):
            raise GuardConfigurationError("legacy edge entries must be objects")
        source, target = split_edge_key(raw_key)
        initial = raw_record.get("initial_ceiling")
        ceiling = raw_record.get("ceiling")
        history = raw_record.get("ceiling_history")
        if (
            not isinstance(initial, int)
            or isinstance(initial, bool)
            or initial < 0
            or not isinstance(ceiling, int)
            or isinstance(ceiling, bool)
            or ceiling < 0
            or not isinstance(history, list)
            or not history
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in history
            )
        ):
            raise GuardConfigurationError(f"invalid ceiling record for {raw_key}")
        if history[0] != initial or history[-1] != ceiling:
            raise GuardConfigurationError(f"ceiling history endpoints mismatch for {raw_key}")
        if any(later > earlier for earlier, later in zip(history, history[1:])):
            raise GuardConfigurationError(f"ceiling history rises for {raw_key}")
        key = (source, target)
        if key in edges:
            raise GuardConfigurationError(f"duplicate legacy edge: {raw_key}")
        edges[key] = ceiling
    return edges


def _validate_exceptions(baseline: Mapping[str, object]) -> list[dict[str, object]]:
    exceptions = baseline.get("temporary_exceptions")
    if not isinstance(exceptions, list):
        raise GuardConfigurationError("temporary_exceptions must be an array")
    validated: list[dict[str, object]] = []
    for index, exception in enumerate(exceptions):
        if not isinstance(exception, dict):
            raise GuardConfigurationError(f"exception {index} must be an object")
        missing = REQUIRED_EXCEPTION_FIELDS - set(exception)
        if missing:
            raise GuardConfigurationError(
                f"exception {index} missing metadata: {', '.join(sorted(missing))}"
            )
        if any(
            not _nonempty_string(exception[field])
            for field in REQUIRED_EXCEPTION_FIELDS - {"ceiling"}
        ):
            raise GuardConfigurationError(f"exception {index} has empty metadata")
        ceiling = exception["ceiling"]
        if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
            raise GuardConfigurationError(f"exception {index} has invalid ceiling")
        path = Path(str(exception["path"]))
        if path.is_absolute() or ".." in path.parts:
            raise GuardConfigurationError(f"exception {index} path must be repository-relative")
        validated.append(exception)
    return validated


def validate_baseline(baseline: Mapping[str, object]) -> dict[tuple[str, str], int]:
    if baseline.get("schema_version") != "eiel_modular_legacy_import_ratchet_v1":
        raise GuardConfigurationError("unsupported baseline schema_version")
    if not _nonempty_string(baseline.get("baseline_commit")):
        raise GuardConfigurationError("baseline_commit is required")
    if not _nonempty_string(baseline.get("source_inventory")):
        raise GuardConfigurationError("source_inventory is required")
    source_digest = baseline.get("source_inventory_sha256")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        raise GuardConfigurationError("source_inventory_sha256 must be lowercase SHA-256")
    audit_roots = baseline.get("audit_roots")
    if not isinstance(audit_roots, list) or not audit_roots or any(
        not _nonempty_string(item) for item in audit_roots
    ):
        raise GuardConfigurationError("audit_roots must be a non-empty string array")
    policy = baseline.get("target_allowed_dependencies")
    if not isinstance(policy, dict) or set(policy) != KNOWN_TARGET_PACKAGES:
        raise GuardConfigurationError("target dependency policy has unknown or missing roots")
    for source, allowed in policy.items():
        if not isinstance(allowed, list) or any(
            target not in KNOWN_TARGET_PACKAGES and target != "legacy_eiel"
            for target in allowed
        ):
            raise GuardConfigurationError(f"invalid target dependency policy for {source}")

    edges = _baseline_edges(baseline)
    _validate_exceptions(baseline)
    classified = baseline.get("classified_debt")
    if not isinstance(classified, list):
        raise GuardConfigurationError("classified_debt must be an array")
    for index, record in enumerate(classified):
        if not isinstance(record, dict) or any(
            not _nonempty_string(record.get(field))
            for field in ("source", "target", "owner", "reason", "remove_by")
        ):
            raise GuardConfigurationError(f"classified debt {index} is incomplete")
        key = (str(record["source"]), str(record["target"]))
        if key not in edges or edges[key] <= 0:
            raise GuardConfigurationError(
                f"classified debt edge is absent from baseline: {edge_key(*key)}"
            )

    expected = baseline.get("baseline_audit")
    if not isinstance(expected, dict):
        raise GuardConfigurationError("baseline_audit must be an object")
    for field in (
        "python_modules_audited",
        "resolved_internal_import_statements",
        "legacy_edge_count",
        "parse_errors",
    ):
        value = expected.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise GuardConfigurationError(f"baseline_audit.{field} must be non-negative")
    if expected["parse_errors"] != 0:
        raise GuardConfigurationError("a ratchet baseline with parse errors is forbidden")
    nonzero_edges = sorted(
        (source, target, count)
        for (source, target), count in edges.items()
        if count > 0
    )
    if expected.get("legacy_edge_count") != len(nonzero_edges):
        raise GuardConfigurationError("baseline legacy edge count is inconsistent")
    expected_sccs = tuple(tuple(item) for item in expected.get("legacy_sccs", []))
    if expected_sccs != sccs_from_edges(nonzero_edges):
        raise GuardConfigurationError("baseline SCC inventory is inconsistent")
    digest_payload = {
        edge_key(source, target): count
        for source, target, count in nonzero_edges
    }
    digest = hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()
    if expected.get("legacy_edges_digest_sha256") != digest:
        raise GuardConfigurationError("baseline legacy edge digest is inconsistent")
    return edges


def _exception_allows(
    exceptions: Sequence[Mapping[str, object]],
    *,
    source: str,
    target: str,
    count: int,
    occurrences: Sequence[ImportOccurrence],
) -> bool:
    paths = {
        occurrence.source_path
        for occurrence in occurrences
        if occurrence.source_family == source and occurrence.target_family == target
    }
    return any(
        exception["source"] == source
        and exception["target"] == target
        and count <= exception["ceiling"]
        and exception["path"] in paths
        for exception in exceptions
    )


def evaluate_audit(
    audit: AuditResult, baseline: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    ceilings = validate_baseline(baseline)
    exceptions = _validate_exceptions(baseline)
    findings: list[dict[str, object]] = []
    for path, error in audit.parse_errors:
        findings.append({"code": "AST_PARSE_ERROR", "path": path, "error": error})
    for root in audit.unknown_target_roots:
        findings.append({"code": "UNKNOWN_TARGET_ROOT", "root": root})
    for source, import_root, path, line in audit.target_external_imports:
        if source == "emergence_contracts":
            findings.append(
                {
                    "code": "FORBIDDEN_CONTRACT_EXTERNAL_IMPORT",
                    "source": source,
                    "target": import_root,
                    "path": path,
                    "line": line,
                    "count": 1,
                }
            )

    allowed_policy = baseline["target_allowed_dependencies"]
    assert isinstance(allowed_policy, dict)
    for source, target, count in audit.target_edges:
        policy_target = target
        if source == "eiel_composition" and target not in KNOWN_TARGET_PACKAGES:
            policy_target = "legacy_eiel"
        if policy_target not in allowed_policy[source]:
            findings.append(
                {
                    "code": "FORBIDDEN_TARGET_DEPENDENCY",
                    "source": source,
                    "target": target,
                    "count": count,
                }
            )

    observed = {(source, target): count for source, target, count in audit.legacy_edges}
    for source, target in sorted(set(ceilings) | set(observed)):
        ceiling = ceilings.get((source, target), 0)
        count = observed.get((source, target), 0)
        if count == ceiling:
            continue
        if count > ceiling and _exception_allows(
            exceptions,
            source=source,
            target=target,
            count=count,
            occurrences=audit.occurrences,
        ):
            continue
        findings.append(
            {
                "code": (
                    "LEGACY_EDGE_GROWTH"
                    if count > ceiling
                    else "LEGACY_CEILING_NOT_LOWERED"
                ),
                "source": source,
                "target": target,
                "ceiling": ceiling,
                "observed": count,
            }
        )
    return tuple(
        sorted(findings, key=lambda item: canonical_json_bytes(item))
    )


def check_repository(repo_root: Path, baseline: Mapping[str, object]) -> dict[str, object]:
    validate_baseline(baseline)
    audit_roots = baseline["audit_roots"]
    assert isinstance(audit_roots, list)
    audit = audit_repository(repo_root, audit_roots=audit_roots)
    findings = evaluate_audit(audit, baseline)
    forbidden_contract_imports = sum(
        int(finding.get("count", 0))
        for finding in findings
        if (
            finding.get("code") == "FORBIDDEN_TARGET_DEPENDENCY"
            and finding.get("source") == "emergence_contracts"
        )
        or finding.get("code") == "FORBIDDEN_CONTRACT_EXTERNAL_IMPORT"
    )
    result = {
        "schema_version": "eiel_modular_import_guard_result_v1",
        "status": "PASS" if not findings else "FAIL",
        "baseline_commit": baseline["baseline_commit"],
        "audit": audit.deterministic_dict(),
        "finding_count": len(findings),
        "findings": list(findings),
        "target_package_violation_count": sum(
            finding.get("code")
            in {"FORBIDDEN_TARGET_DEPENDENCY", "FORBIDDEN_CONTRACT_EXTERNAL_IMPORT"}
            for finding in findings
        ),
        "emergence_contracts_forbidden_imports": forbidden_contract_imports,
    }
    return result


def load_baseline(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GuardConfigurationError(f"cannot load baseline: {error}") from error
    if not isinstance(value, dict):
        raise GuardConfigurationError("baseline root must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--check", action="store_true", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = check_repository(
            arguments.repo_root.resolve(), load_baseline(arguments.baseline.resolve())
        )
    except GuardConfigurationError as error:
        result = {
            "schema_version": "eiel_modular_import_guard_result_v1",
            "status": "FAIL",
            "finding_count": 1,
            "findings": [
                {"code": "BASELINE_CONFIGURATION_INVALID", "error": str(error)}
            ],
            "target_package_violation_count": 0,
            "emergence_contracts_forbidden_imports": 0,
        }
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
