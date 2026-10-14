from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.scripts.genome_upgrade_v2_state_inventory import (
    inventory_rows,
    write_artifacts,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    PACKAGE_OBLIGATIONS,
    PRE_ROUTE_PACKAGE_OBLIGATIONS,
    ROUTE_OWNED_OBLIGATIONS,
)
from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
)
from engine.qualification_plans.v1 import EXECUTION_ROUTE_OBLIGATIONS


def test_inventory_covers_mandatory_authority_categories() -> None:
    rows = inventory_rows()
    names = {row["name"] for row in rows}
    assert {
        "genome_v2_alleles_q10",
        "genome_v2_modifiers_q10",
        "genome_v2_products_q10",
        "genome_v2_phenotypes_q10",
        "genome_v2_context_q10",
        "genome_v2_runtime_projection",
        "genome_v2_regulatory_projection",
        "genome_v2_staged_children",
        "genome_v2_turnover_staging_and_shadows",
        "canonical_hereditary_digest",
        "linkage_and_inheritance_rng",
        "ordinary_birth_identity_lineage_and_reservations",
        "energy_health_reproduction_and_memory_targets",
        "inheritance_sidecar_identity",
        "checkpoint_cursors_and_cut_point",
        "cancellation_and_sticky_failure",
        "compact_generation_summary",
    } <= names
    required = {
        "owner",
        "type_shape",
        "mutation_point",
        "reset_rule",
        "checkpoint_inclusion",
        "restore_rule",
        "rollback_rule",
        "equality_rule",
        "route_support",
        "evidence_field",
    }
    assert all(required <= row.keys() for row in rows)


def test_artifacts_are_consistent_and_reviewable(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    inventory = json.loads(
        (tmp_path / "genome_v2_authoritative_state_inventory.json").read_text(
            encoding="ascii"
        )
    )
    executable_contract = json.loads(
        (tmp_path / "genome_v2_executable_contract.json").read_text(
            encoding="ascii"
        )
    )
    with (
        tmp_path / "genome_v2_mutable_state_ownership.csv"
    ).open(encoding="ascii", newline="") as stream:
        ownership = list(csv.DictReader(stream))
    with (
        tmp_path / "genome_v2_checkpoint_surface_matrix.csv"
    ).open(encoding="ascii", newline="") as stream:
        checkpoint = list(csv.DictReader(stream))
    names = [row["name"] for row in inventory["states"]]
    assert names == [row["name"] for row in ownership]
    assert names == [row["state"] for row in checkpoint]
    assert inventory["production_route"] == "unified_native_sm75_cuda"
    assert executable_contract["provider_id"] == (
        "eiel-production-provider-v1"
    )
    assert executable_contract["global_genome_profile_id"] == (
        "genome_upgrade_v2"
    )
    assert executable_contract["capability_contract_sha256"] == (
        GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
    )
    capability_contract = json.loads(
        (tmp_path / "genome_v2_p10_capability_contract.json").read_text(
            encoding="ascii"
        )
    )
    assert capability_contract["contract_sha256"] == (
        GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
    )
    assert tuple(executable_contract["required_obligations"]) == (
        PRE_ROUTE_PACKAGE_OBLIGATIONS
    )
    assert len(executable_contract["promotion_test_contract_sha256"]) == 8
    assert len(executable_contract["promotion_test_matrix_sha256"]) == 9
    assert all(
        len(value) == 64
        for value in (
            *executable_contract[
                "promotion_test_contract_sha256"
            ].values(),
            *executable_contract[
                "promotion_test_matrix_sha256"
            ].values(),
        )
    )
    package_contract = json.loads(
        (
            tmp_path / "genome_v2_executable_package_contract_v1.json"
        ).read_text(encoding="ascii")
    )
    obligation_dictionary = json.loads(
        (
            tmp_path / "genome_v2_package_obligation_dictionary.json"
        ).read_text(encoding="ascii")
    )
    invalidation_rules = json.loads(
        (
            tmp_path / "genome_v2_package_invalidation_rules.json"
        ).read_text(encoding="ascii")
    )
    build_policy = json.loads(
        (
            tmp_path / "genome_v2_build_admission_policy.json"
        ).read_text(encoding="ascii")
    )
    assert tuple(package_contract["complete_promotion_obligations"]) == (
        PACKAGE_OBLIGATIONS
    )
    assert tuple(package_contract["route_owned_obligations"]) == (
        ROUTE_OWNED_OBLIGATIONS
    )
    assert len(obligation_dictionary["obligations"]) == 25
    assert invalidation_rules["partial_reuse"] == "forbidden"
    assert build_policy["maximum_monitored_build_trees"] == 1


def test_package_obligations_do_not_expand_route_canary_vocabulary() -> None:
    assert set(PRE_ROUTE_PACKAGE_OBLIGATIONS).isdisjoint(
        EXECUTION_ROUTE_OBLIGATIONS
    )
    assert set(ROUTE_OWNED_OBLIGATIONS).isdisjoint(
        EXECUTION_ROUTE_OBLIGATIONS
    )
