"""Hash-graph and exact-inventory validation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..io import InputStore, ReviewInputError, json_pointer
from ..model import CheckResult


def _has_cycle(nodes: set[str], edges: list[tuple[str, str]]) -> bool:
    graph: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        graph[source].append(target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in graph[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in nodes)


def validate_hash_graph(spec: dict, store: InputStore) -> CheckResult:
    result = CheckResult(spec["id"], "hash_graph")
    documents: dict[str, object] = {}
    digests: dict[str, str] = {}
    try:
        node_ids = [node["id"] for node in spec["nodes"]]
        duplicate_ids = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
        if duplicate_ids:
            result.fail(f"hash graph node IDs must be unique: {duplicate_ids}")
        for node in spec["nodes"]:
            node_id = node["id"]
            path = store.resolve(node["path"])
            digest = store.hash_file(path)
            digests[node_id] = digest
            if digest != node["sha256"]:
                result.fail(
                    f"node {node_id!r} hash mismatch: expected {node['sha256']}, got {digest}"
                )
        edges: list[tuple[str, str]] = []
        for edge in spec.get("edges", []):
            source = edge["from"]
            target = edge["to"]
            if source not in digests or target not in digests:
                result.fail(f"edge references unknown node: {source!r} -> {target!r}")
                continue
            if source not in documents:
                source_node = next(node for node in spec["nodes"] if node["id"] == source)
                documents[source] = store.read_json(source_node["path"])
            bound = json_pointer(documents[source], edge["json_pointer"])
            if bound != digests[target]:
                result.fail(
                    f"edge {source!r} -> {target!r} does not bind target hash at "
                    f"{edge['json_pointer']}"
                )
            edges.append((source, target))
        if spec.get("require_acyclic", True) and _has_cycle(set(digests), edges):
            result.fail("hash graph contains a cycle")
        result.details = {
            "node_count": len(digests),
            "edge_count": len(edges),
            "digests": dict(sorted(digests.items())),
        }
    except (ReviewInputError, StopIteration) as exc:
        result.fail(str(exc))
    return result


def _inventory_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReviewInputError(f"inventory contains symlink: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files)


def validate_inventory(spec: dict, store: InputStore) -> CheckResult:
    result = CheckResult(spec["id"], "exact_inventory")
    try:
        root = store.resolve(spec["root"], require_file=False)
        if not root.is_dir():
            raise ReviewInputError(f"inventory root is not a directory: {spec['root']}")
        actual_paths: list[str] = []
        actual_hashes: dict[str, str] = {}
        for path in _inventory_files(root):
            relative = path.relative_to(root).as_posix()
            actual_paths.append(relative)
            actual_hashes[relative] = store.hash_file(path)
        expected_paths = sorted(spec["expected_paths"])
        missing = sorted(set(expected_paths) - set(actual_paths))
        unexpected = sorted(set(actual_paths) - set(expected_paths))
        if missing:
            result.fail(f"missing inventory paths: {missing}")
        if unexpected:
            result.fail(f"unexpected inventory paths: {unexpected}")
        for relative, expected_hash in sorted(spec.get("expected_sha256", {}).items()):
            actual_hash = actual_hashes.get(relative)
            if actual_hash != expected_hash:
                result.fail(
                    f"inventory hash mismatch for {relative!r}: expected {expected_hash}, "
                    f"got {actual_hash}"
                )
        result.details = {
            "root": spec["root"],
            "expected_count": len(expected_paths),
            "actual_count": len(actual_paths),
            "missing": missing,
            "unexpected": unexpected,
        }
    except ReviewInputError as exc:
        result.fail(str(exc))
    return result
