from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "eiel_composition" / "adaptive_expedition"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.append(node.module)
    return tuple(result)


def test_package_instructions_freeze_authority_and_stop_boundaries() -> None:
    guidance = (PACKAGE_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "not a\nscheduler, provider, evidence authority" in guidance
    assert "Absolute ticks are the only scientific time coordinate" in guidance
    assert "RetainedParentBankV1" in guidance
    assert "ENGINEERING_G5_COMPLETE_AWAITING_HUMAN_REVIEW" in guidance
    assert "Do not launch 24-hour" in guidance


def test_policy_modules_cannot_import_execution_or_ambient_io() -> None:
    forbidden_roots = {
        "engine",
        "tools",
        "emergence_evidence",
        "emergence_ops",
        "emergence_runtime",
        "pathlib",
        "random",
        "secrets",
        "socket",
        "subprocess",
        "time",
        "urllib",
    }
    policy_paths = [PACKAGE_ROOT / "policy_protocol.py"]
    policy_paths.extend(sorted((PACKAGE_ROOT / "policies").glob("*.py")))
    for path in policy_paths:
        if not path.exists():
            continue
        imported_roots = {name.split(".", 1)[0] for name in _imports(path)}
        assert not imported_roots.intersection(forbidden_roots), path


def test_core_never_imports_the_reference_policy_directly() -> None:
    for name in ("coordinator.py", "ledger.py", "atlas.py", "parent_bank.py"):
        path = PACKAGE_ROOT / name
        if not path.exists():
            continue
        assert "adaptive_illumination_v1" not in _imports(path)


def test_no_expedition_module_imports_provider_or_launchpad_children() -> None:
    forbidden_fragments = (
        "cuda_full_evolution_v2_provider",
        "launchpad_attempts",
        "engine.campaigns.providers",
        "tools.actors.resource_manager",
    )
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        imports = _imports(path)
        assert not any(
            fragment in imported
            for imported in imports
            for fragment in forbidden_fragments
        ), path


def test_only_named_adapters_may_import_the_science_front_door() -> None:
    allowed = {"founder_loop.py", "continuation_adapter.py"}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        imports = _imports(path)
        if any("science_request_v1_front_door" in imported for imported in imports):
            assert path.name in allowed, path


def test_package_does_not_create_a_manager_scheduler_or_database() -> None:
    forbidden_names = {"manager.py", "scheduler.py", "database.py"}
    assert not forbidden_names.intersection(
        path.name for path in PACKAGE_ROOT.glob("*.py")
    )


def test_outer_frontier_is_an_ordinal_not_a_scientific_tick_or_index() -> None:
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert '"frontier_index"' not in source, path
        assert "'frontier_index'" not in source, path


def test_public_schema_property_names_are_tick_native() -> None:
    schema_root = REPO_ROOT / "_spec" / "adaptive_science_expedition" / "v1"

    def property_names(value: object) -> list[str]:
        if isinstance(value, dict):
            result = list(value.get("properties", {}))
            for child in value.values():
                result.extend(property_names(child))
            return result
        if isinstance(value, list):
            result: list[str] = []
            for child in value:
                result.extend(property_names(child))
            return result
        return []

    forbidden = {"generation", "ticks_per_generation", "frontier_index"}
    for path in sorted(schema_root.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        assert forbidden.isdisjoint(property_names(document)), path
