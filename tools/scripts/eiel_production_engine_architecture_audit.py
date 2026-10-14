"""Emit bounded, machine-derived EIEL production-engine architecture evidence.

This audit deliberately separates structural facts that can be derived from the
current source tree and live registries from architectural ownership
declarations.  It does not claim to prove semantic uniqueness of mechanics.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Final, Iterable

from engine.campaigns.production_engine_architecture_audit import (
    OWNERSHIP_ROWS,
    REMOVED_PROVIDER_MODULES,
    direct_lifecycle_construction_sites_v1,
    missing_component_implementation_modules_v1,
)
from engine.campaigns.production_engine_identity import (
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
)
from engine.campaigns.providers import default_execution_provider_registry
from engine.campaigns.registry import default_profile_registry


SCHEMA_VERSION: Final[str] = (
    "eiel_production_engine_architecture_evidence_v1"
)

FORMER_PROVIDER_CLASSES: Final[tuple[str, ...]] = (
    "GmnwGruExecutionProviderV1",
    "InheritanceCrossPathGpuExecutionProviderV1",
    "PatchFixedExecutionProviderV1",
    "PatchJungleFixedMlpExecutionProviderV1",
    "PatchStableGruExecutionProviderV1",
    "PatchWdlFixedMlpExecutionProviderV1",
    "PatchWdlStableGruExecutionProviderV1",
    "Recomb72ExecutionProviderV1",
    "Recomb72GpuExecutionProviderV1",
    "Recomb72LiveExecutionProviderV1",
)

EXPECTED_PRODUCTION_PROVIDER_CLASS: Final[str] = (
    "engine/campaigns/production_engine_provider.py:"
    "EielProductionExecutionProviderV1"
)


def _python_paths(project_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for source_root in (project_root / "engine", project_root / "tools"):
        if source_root.is_dir():
            paths.extend(source_root.rglob("*.py"))
    return tuple(sorted(path.resolve() for path in paths if path.is_file()))


def _base_name(base: ast.expr) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _class_declarations_v1(
    project_root: Path,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    declarations: list[tuple[str, str]] = []
    errors: list[str] = []
    for path in _python_paths(project_root):
        relative = path.relative_to(project_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{relative}:{type(exc).__name__}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                declarations.append((relative, node.name))
    return tuple(sorted(declarations)), tuple(sorted(errors))


def _execution_provider_declarations_v1(
    project_root: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    providers: list[str] = []
    errors: list[str] = []
    for path in _python_paths(project_root):
        relative = path.relative_to(project_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{relative}:{type(exc).__name__}")
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name.endswith("ExecutionProviderV1")
                and "Protocol" not in {
                    name for base in node.bases if (name := _base_name(base))
                }
            ):
                providers.append(f"{relative}:{node.name}")
    return tuple(sorted(providers)), tuple(sorted(errors))


def former_provider_source_surfaces_v1(
    project_root: Path,
) -> dict[str, Any]:
    """Derive whether any former provider implementation surface still exists.

    A zero count proves only that the former provider modules and class bodies
    are absent. It does not prove that every migrated mechanic is semantically
    unique elsewhere in the production call graph.
    """

    declarations, scan_errors = _class_declarations_v1(project_root)
    by_name: dict[str, list[str]] = {}
    for relative, name in declarations:
        by_name.setdefault(name, []).append(relative)

    campaign_root = project_root / "engine" / "campaigns"
    module_rows = [
        {
            "module": module,
            "present": (campaign_root / module).exists(),
        }
        for module in REMOVED_PROVIDER_MODULES
    ]
    class_rows = [
        {
            "class": name,
            "declaration_sites": sorted(by_name.get(name, ())),
        }
        for name in FORMER_PROVIDER_CLASSES
    ]
    remaining_module_count = sum(row["present"] is True for row in module_rows)
    remaining_class_count = sum(
        len(row["declaration_sites"]) for row in class_rows
    )
    derived_zero = (
        not scan_errors
        and remaining_module_count == 0
        and remaining_class_count == 0
    )
    return {
        "scope": (
            "former provider module files and exact former execution-provider "
            "class declarations in engine/ and tools/"
        ),
        "former_provider_modules": module_rows,
        "former_provider_classes": class_rows,
        "scan_errors": list(scan_errors),
        "remaining_module_count": remaining_module_count,
        "remaining_class_declaration_count": remaining_class_count,
        "mechanics_remaining_in_former_provider_implementations": (
            0 if derived_zero else None
        ),
        "derived_zero": derived_zero,
        "semantic_mechanic_uniqueness_proven": False,
    }


def _declared_ownership_payload_v1() -> dict[str, Any]:
    rows = [
        {
            "feature": row.feature,
            "declared_owner": row.owner,
            "declared_implementation_count": row.implementation_count,
            "evidence_kind": "architectural_declaration",
        }
        for row in OWNERSHIP_ROWS
    ]
    features = [row["feature"] for row in rows]
    declarations_well_formed = (
        len(features) == len(set(features))
        and all(row["declared_owner"] for row in rows)
        and all(row["declared_implementation_count"] == 1 for row in rows)
    )
    return {
        "rows": rows,
        "declarations_well_formed": declarations_well_formed,
        "implementation_counts_machine_derived": False,
        "semantic_mechanic_uniqueness_proven": False,
    }


def architecture_evidence_payload_v1(project_root: Path) -> dict[str, Any]:
    """Return structural evidence without promoting declarations into proof."""

    project_root = project_root.resolve()
    provider_declarations, provider_scan_errors = (
        _execution_provider_declarations_v1(project_root)
    )
    former = former_provider_source_surfaces_v1(project_root)
    providers = default_execution_provider_registry().providers()
    profiles = default_profile_registry().profiles()
    registry_provider_rows = [
        {
            "provider_id": provider.provider_id,
            "profile_id": provider.profile_id,
            "implementation": (
                f"{type(provider).__module__}.{type(provider).__qualname__}"
            ),
        }
        for provider in providers
    ]
    registry_profile_rows = [
        {
            "profile_id": profile.profile_id,
            "execution_owner": profile.execution_owner,
        }
        for profile in profiles
    ]
    lifecycle_sites = direct_lifecycle_construction_sites_v1(project_root)
    missing_component_modules = missing_component_implementation_modules_v1()
    derived_checks = {
        "one_registered_provider": (
            len(registry_provider_rows) == 1
            and registry_provider_rows[0]["provider_id"]
            == PRODUCTION_PROVIDER_ID
            and registry_provider_rows[0]["profile_id"]
            == PRODUCTION_PROFILE_ID
        ),
        "one_registered_runtime_profile": (
            len(registry_profile_rows) == 1
            and registry_profile_rows[0]["profile_id"]
            == PRODUCTION_PROFILE_ID
        ),
        "one_concrete_execution_provider_class": (
            provider_declarations == (EXPECTED_PRODUCTION_PROVIDER_CLASS,)
        ),
        "no_provider_source_scan_errors": not provider_scan_errors,
        "no_direct_lifecycle_construction": not lifecycle_sites,
        "all_production_component_modules_importable": (
            not missing_component_modules
        ),
        "former_provider_implementation_surfaces_absent": former[
            "derived_zero"
        ],
    }
    structural_ok = all(derived_checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "structural_ok": structural_ok,
        "production_provider_id": PRODUCTION_PROVIDER_ID,
        "production_profile_id": PRODUCTION_PROFILE_ID,
        "derived_checks": derived_checks,
        "registered_providers": registry_provider_rows,
        "registered_runtime_profiles": registry_profile_rows,
        "concrete_execution_provider_classes": list(provider_declarations),
        "provider_source_scan_errors": list(provider_scan_errors),
        "direct_lifecycle_construction_sites": list(lifecycle_sites),
        "missing_component_implementation_modules": list(
            missing_component_modules
        ),
        "former_provider_structural_closure": former,
        "declared_ownership": _declared_ownership_payload_v1(),
        "semantic_mechanic_uniqueness": {
            "status": "not_machine_proven",
            "implementation_count": None,
            "reason": (
                "AST and registry scans cannot establish semantic equivalence "
                "or uniqueness across domain mechanic implementations"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = architecture_evidence_payload_v1(args.project_root)
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))
    return 0 if payload["structural_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

