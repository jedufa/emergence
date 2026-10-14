from pathlib import Path

from engine.campaigns.production_engine_architecture_audit import (
    OWNERSHIP_ROWS,
    architecture_audit_payload_v1,
)
from engine.campaigns.providers import default_execution_provider_registry
from engine.campaigns.registry import default_profile_registry
from tools.scripts.eiel_production_engine_architecture_audit import (
    FORMER_PROVIDER_CLASSES,
    architecture_evidence_payload_v1,
    former_provider_source_surfaces_v1,
)


CAMPAIGN_ROOT = Path(__file__).parents[1] / "engine" / "campaigns"
PROJECT_ROOT = Path(__file__).parents[1]


def test_source_tree_contains_exactly_one_execution_provider_class() -> None:
    payload = architecture_audit_payload_v1(CAMPAIGN_ROOT)

    assert payload["ok"] is True
    assert payload["execution_provider_classes"] == [
        "production_engine_provider.py:EielProductionExecutionProviderV1"
    ]
    assert payload["removed_provider_modules_present"] == []
    assert payload["direct_lifecycle_construction_sites"] == []
    assert payload["missing_component_implementation_modules"] == []


def test_every_production_subsystem_has_exactly_one_declared_owner() -> None:
    assert len({row.feature for row in OWNERSHIP_ROWS}) == len(OWNERSHIP_ROWS)
    assert all(row.implementation_count == 1 for row in OWNERSHIP_ROWS)
    assert all(row.owner for row in OWNERSHIP_ROWS)


def test_declared_identity_matches_the_only_registered_provider() -> None:
    payload = architecture_audit_payload_v1(CAMPAIGN_ROOT)
    providers = default_execution_provider_registry().providers()

    assert len(providers) == 1
    assert providers[0].provider_id == payload["production_provider_id"]
    assert providers[0].profile_id == payload["production_profile_id"]
    profiles = default_profile_registry().profiles()
    assert len(profiles) == 1
    assert profiles[0].profile_id == payload["production_profile_id"]


def test_derived_evidence_separates_declarations_from_structural_proof() -> None:
    payload = architecture_evidence_payload_v1(PROJECT_ROOT)

    assert payload["structural_ok"] is True
    assert all(payload["derived_checks"].values())
    assert len(payload["registered_providers"]) == 1
    assert len(payload["registered_runtime_profiles"]) == 1
    assert payload["concrete_execution_provider_classes"] == [
        "engine/campaigns/production_engine_provider.py:"
        "EielProductionExecutionProviderV1"
    ]
    declared = payload["declared_ownership"]
    assert declared["declarations_well_formed"] is True
    assert declared["implementation_counts_machine_derived"] is False
    assert declared["semantic_mechanic_uniqueness_proven"] is False
    assert {
        row["evidence_kind"] for row in declared["rows"]
    } == {"architectural_declaration"}
    assert payload["semantic_mechanic_uniqueness"] == {
        "status": "not_machine_proven",
        "implementation_count": None,
        "reason": (
            "AST and registry scans cannot establish semantic equivalence "
            "or uniqueness across domain mechanic implementations"
        ),
    }


def test_former_provider_implementation_surfaces_are_derived_absent() -> None:
    evidence = former_provider_source_surfaces_v1(PROJECT_ROOT)

    assert evidence["derived_zero"] is True
    assert evidence["scan_errors"] == []
    assert evidence["remaining_module_count"] == 0
    assert evidence["remaining_class_declaration_count"] == 0
    assert evidence["mechanics_remaining_in_former_provider_implementations"] == 0
    assert evidence["semantic_mechanic_uniqueness_proven"] is False
    assert {
        row["class"] for row in evidence["former_provider_classes"]
    } == set(FORMER_PROVIDER_CLASSES)
    assert all(
        row["declaration_sites"] == []
        for row in evidence["former_provider_classes"]
    )
    assert all(
        row["present"] is False
        for row in evidence["former_provider_modules"]
    )


def test_former_provider_surface_scan_fails_closed_on_remaining_class(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "engine" / "campaigns"
    source_root.mkdir(parents=True)
    (source_root / "residue.py").write_text(
        "class GmnwGruExecutionProviderV1:\n"
        "    pass\n",
        encoding="utf-8",
    )

    evidence = former_provider_source_surfaces_v1(tmp_path)

    assert evidence["derived_zero"] is False
    assert evidence["remaining_class_declaration_count"] == 1
    assert evidence["mechanics_remaining_in_former_provider_implementations"] is None
