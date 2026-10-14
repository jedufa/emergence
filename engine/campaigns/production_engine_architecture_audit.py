"""Machine-readable architectural invariants for the EIEL production engine."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Final

from .production_engine_identity import (
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
)

@dataclass(frozen=True, slots=True)
class OwnershipRowV1:
    feature: str
    owner: str
    implementation_count: int = 1

    def payload(self) -> dict[str, str | int]:
        return {
            "feature": self.feature,
            "owner": self.owner,
            "implementation_count": self.implementation_count,
        }


OWNERSHIP_ROWS: Final[tuple[OwnershipRowV1, ...]] = (
    OwnershipRowV1(
        "execution lifecycle",
        "engine.campaigns.providers.ProductionEngineLifecycleV1",
    ),
    OwnershipRowV1(
        "dispatch and batching",
        "engine.campaigns.providers.ProductionEngineLifecycleV1",
    ),
    OwnershipRowV1(
        "checkpoint export and restore",
        "engine.campaigns.providers.ProductionEngineLifecycleV1",
    ),
    OwnershipRowV1(
        "Chronicle projection",
        "engine.campaigns.providers.ProductionEngineLifecycleV1",
    ),
    OwnershipRowV1(
        "reporting and analysis projection",
        "engine.campaigns.providers.ProductionEngineLifecycleV1",
    ),
    OwnershipRowV1(
        "route qualification",
        "engine.campaigns.production_engine_provider.EielProductionExecutionProviderV1",
    ),
    OwnershipRowV1(
        "genome inheritance and mutation registry",
        "engine.evolution.global_genome_registry_v1",
    ),
    OwnershipRowV1(
        "world component registry",
        "engine.campaigns.production_engine_components.WORLD_COMPONENTS",
    ),
    OwnershipRowV1(
        "controller component registry",
        "engine.campaigns.production_engine_components.CONTROLLER_COMPONENTS",
    ),
    OwnershipRowV1(
        "native mechanics composition",
        "engine.campaigns.production_engine_native_bindings",
    ),
)


REMOVED_PROVIDER_MODULES: Final[tuple[str, ...]] = (
    "gmnw_provider.py",
    "inheritance_cross_path_gpu_provider.py",
    "jungle_provider.py",
    "recomb72_gpu_provider.py",
    "recomb72_live_provider.py",
    "recomb72_provider.py",
)


def execution_provider_classes_v1(campaign_root: Path) -> tuple[str, ...]:
    """Return concrete classes whose names declare an execution-provider role."""

    rows: list[str] = []
    for path in sorted(campaign_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.ClassDef)
                and node.name.endswith("ExecutionProviderV1")
                and not any(
                    isinstance(base, ast.Name) and base.id == "Protocol"
                    for base in node.bases
                )
            ):
                rows.append(f"{path.name}:{node.name}")
    return tuple(rows)


def direct_lifecycle_construction_sites_v1(project_root: Path) -> tuple[str, ...]:
    """Find non-test code that constructs the lifecycle behind the provider."""

    rows: list[str] = []
    for source_root in (project_root / "engine", project_root / "tools"):
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                name = (
                    function.id
                    if isinstance(function, ast.Name)
                    else function.attr
                    if isinstance(function, ast.Attribute)
                    else None
                )
                if name == "ProductionEngineLifecycleV1":
                    rows.append(
                        f"{path.relative_to(project_root)}:{node.lineno}"
                    )
    return tuple(rows)


def missing_component_implementation_modules_v1() -> tuple[str, ...]:
    """Prove component registry ownership points at importable source modules."""

    from .production_engine_components import production_domain_components_v1

    return tuple(
        component.implementation
        for component in production_domain_components_v1()
        if importlib.util.find_spec(component.implementation) is None
    )


def architecture_audit_payload_v1(campaign_root: Path) -> dict[str, object]:
    project_root = campaign_root.parents[1]
    provider_classes = execution_provider_classes_v1(campaign_root)
    direct_lifecycle_construction_sites = direct_lifecycle_construction_sites_v1(
        project_root
    )
    missing_component_modules = missing_component_implementation_modules_v1()
    removed_modules_present = tuple(
        name for name in REMOVED_PROVIDER_MODULES if (campaign_root / name).exists()
    )
    duplicate_owners = tuple(
        row.feature for row in OWNERSHIP_ROWS if row.implementation_count != 1
    )
    expected_provider = (
        "production_engine_provider.py:EielProductionExecutionProviderV1",
    )
    return {
        "schema_version": "eiel_production_engine_architecture_audit_v1",
        "ok": (
            provider_classes == expected_provider
            and not removed_modules_present
            and not duplicate_owners
            and not direct_lifecycle_construction_sites
            and not missing_component_modules
        ),
        "production_provider_id": PRODUCTION_PROVIDER_ID,
        "production_profile_id": PRODUCTION_PROFILE_ID,
        "execution_provider_classes": list(provider_classes),
        "removed_provider_modules_present": list(removed_modules_present),
        "ownership": [row.payload() for row in OWNERSHIP_ROWS],
        "duplicate_owners": list(duplicate_owners),
        "direct_lifecycle_construction_sites": list(
            direct_lifecycle_construction_sites
        ),
        "missing_component_implementation_modules": list(
            missing_component_modules
        ),
    }


__all__ = [
    "OWNERSHIP_ROWS",
    "OwnershipRowV1",
    "PRODUCTION_PROFILE_ID",
    "PRODUCTION_PROVIDER_ID",
    "REMOVED_PROVIDER_MODULES",
    "architecture_audit_payload_v1",
    "direct_lifecycle_construction_sites_v1",
    "execution_provider_classes_v1",
    "missing_component_implementation_modules_v1",
]
