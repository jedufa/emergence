from __future__ import annotations

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
    PREREGISTRATION_SHA256 as ECOLOGICAL_E1_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as ECOLOGICAL_E1_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (
    PREREGISTRATION_SHA256 as ENVIRONMENT_SWITCH_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as ENVIRONMENT_SWITCH_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
    PREREGISTRATION_SHA256 as HEREDITARY_CAPACITY_PREREGISTRATION_SHA256,
    SCHEMA_VERSION as HEREDITARY_CAPACITY_SCHEMA_VERSION,
)
from engine.evolution.genome_upgrade_v2_package_v1 import (
    capacity_layout_contract_v1,
    checkpoint_authority_contract_v1,
    executable_package_contract_v1,
    package_invalidation_decision_v1,
    package_invalidation_field_map_v1,
    package_core_identity_sha256_v1,
    package_identity_sha256_v1,
    route_input_contract_v1,
    supported_checkpoint_cut_ticks_v1,
    supported_world_controller_config_envelope_v1,
    validate_capacity_layout_admission_v1,
)


def _identity() -> dict[str, object]:
    contract = executable_package_contract_v1()
    return {
        field: f"value:{field}"
        for field in contract["required_identity_fields"]
    }


def test_package_identity_covers_prompt_defining_surfaces() -> None:
    fields = set(executable_package_contract_v1()["required_identity_fields"])
    assert {
        "cuda_artifact_sha256",
        "loaded_artifact_sha256",
        "cuda_manifest_sha256",
        "capability_contract_sha256",
        "causal_settlement_schema_version",
        "causal_settlement_contract_sha256",
        "causal_gate_schema_version",
        "causal_gate_preregistration_sha256",
        "ecological_e1_schema_version",
        "ecological_e1_preregistration_sha256",
        "hereditary_capacity_schema_version",
        "hereditary_capacity_preregistration_sha256",
        "environment_switch_schema_version",
        "environment_switch_preregistration_sha256",
        "cancellation_abi_sha256",
        "capacity_layout_contract_sha256",
        "route_contract_inputs_sha256",
        "critical_build_flags_sha256",
        "target_architecture",
        "no_jit",
        "no_fallback",
    } <= fields
    assert (
        executable_package_contract_v1()["causal_evidence_binding"]["post_route"]
        == (
            "event, terminal, and strict-join evidence is mandatory at C1 "
            "with the exact issued package and execution route"
        )
    )


def test_package_identity_field_set_is_exactly_bound() -> None:
    required = executable_package_contract_v1()["required_identity_fields"]
    bound = {
        field
        for fields in package_invalidation_field_map_v1().values()
        for field in fields
    }
    assert len(required) == len(set(required))
    assert set(required) == bound | {
        "diagnostic_only",
        "compiler_cuda_torch_environment",
    }


def test_e1_contract_and_patch_world_envelope_are_package_bound() -> None:
    fields = package_invalidation_field_map_v1()["ecological_authority"]
    assert fields == (
        "ecological_e1_schema_version",
        "ecological_e1_preregistration_sha256",
    )
    assert ECOLOGICAL_E1_SCHEMA_VERSION == (
        "genome_v2_p10_e1_ecological_gate_v1"
    )
    assert len(ECOLOGICAL_E1_PREREGISTRATION_SHA256) == 64
    assert supported_world_controller_config_envelope_v1() == {
        "world_base": "patch",
        "world_overlays": [
            "eed_resource_impact",
            "jungle",
            "none",
            "wdl_delayed_hazard",
        ],
        "controller_family": "fixed_mlp",
        "maximum_physical_cells": 48,
        "maximum_resident_capacity": 512,
        "target_architecture": "sm_75",
    }
    assert supported_checkpoint_cut_ticks_v1() == [
        "terminal_at_total_ticks",
        "cancellation_at_absolute_tick",
    ]


def test_terminal_science_preregistrations_are_package_bound() -> None:
    fields = package_invalidation_field_map_v1()[
        "terminal_science_authority"
    ]
    assert fields == (
        "hereditary_capacity_schema_version",
        "hereditary_capacity_preregistration_sha256",
        "environment_switch_schema_version",
        "environment_switch_preregistration_sha256",
    )
    assert HEREDITARY_CAPACITY_SCHEMA_VERSION == (
        "genome_v2_p10_hereditary_capacity_gate_v1"
    )
    assert ENVIRONMENT_SWITCH_SCHEMA_VERSION == (
        "genome_v2_p10_environment_switch_gate_v1"
    )
    assert len(HEREDITARY_CAPACITY_PREREGISTRATION_SHA256) == 64
    assert len(ENVIRONMENT_SWITCH_PREREGISTRATION_SHA256) == 64


def test_capability_contract_change_invalidates_package() -> None:
    baseline = _identity()
    candidate = {
        **baseline,
        "capability_contract_sha256": "changed:capability",
    }
    decision = package_invalidation_decision_v1(baseline, candidate)
    assert decision["action"] == "requalify"
    assert "global_genome_profile" in decision["changed_surfaces"]


def test_every_bound_surface_change_invalidates_package() -> None:
    baseline = _identity()
    for surface, fields in package_invalidation_field_map_v1().items():
        candidate = dict(baseline)
        candidate[fields[0]] = f"changed:{surface}"
        decision = package_invalidation_decision_v1(baseline, candidate)
        assert decision["reusable"] is False
        assert decision["action"] == "requalify"
        assert surface in decision["changed_surfaces"]


def test_unbound_documentation_change_does_not_invalidate_package() -> None:
    baseline = _identity()
    candidate = {**baseline, "documentation_note": "changed"}
    decision = package_invalidation_decision_v1(baseline, candidate)
    assert decision["reusable"] is True
    assert decision["action"] == "reuse"
    assert decision["unbound_metadata_ignored"] is True


def test_capacity_and_route_contracts_are_fixed_and_hashable() -> None:
    capacity = capacity_layout_contract_v1()
    route = route_input_contract_v1()
    assert capacity["bytes_per_cell_slot"] == 1160
    assert capacity["qualification_envelope"]["maximum_physical_cells"] == 48
    assert capacity["qualification_envelope"]["maximum_resident_capacity"] == 512
    assert "ticks_per_generation" not in capacity["qualification_envelope"]
    assert "maximum_logical_generations" not in capacity["qualification_envelope"]
    assert "maximum_total_ticks" not in capacity["qualification_envelope"]
    assert checkpoint_authority_contract_v1() == {
        "schema_version": "genome_v2_absolute_tick_checkpoint_authority_v1",
        "authority": "absolute_tick_v1",
        "request_fields": ["total_ticks", "checkpoint_ticks"],
        "restore_cursor_field": "next_absolute_tick",
        "span_semantics": "half_open_absolute_tick_span",
        "compatibility_policy": "legacy_compatibility_optional_non_authoritative",
    }
    assert route["provider_id"] == "eiel-production-provider-v1"
    assert route["runtime_profile_id"] == "eiel-production-runtime-v1"
    assert route["jit_allowed"] is False
    assert route["fallback_allowed"] is False
    assert len(content_sha256(capacity)) == 64
    assert len(content_sha256(route)) == 64


def test_capacity_admission_accepts_boundary_and_rejects_overflow() -> None:
    assert validate_capacity_layout_admission_v1(
        cells=48,
        capacity=512,
        total_ticks=10_000_000_000,
    ) == {
        "physical_cells": 48,
        "resident_capacity": 512,
        "total_ticks": 10_000_000_000,
    }
    for override in (
        {"cells": 49},
        {"capacity": 513},
    ):
        values = {
            "cells": 48,
            "capacity": 512,
            "total_ticks": 10_000_000_000,
            **override,
        }
        with pytest.raises(ValueError, match="envelope exceeded"):
            validate_capacity_layout_admission_v1(**values)
    with pytest.raises(ValueError, match="positive integers"):
        validate_capacity_layout_admission_v1(
            cells=48,
            capacity=512,
            total_ticks=0,
        )


def test_legacy_compatibility_metadata_is_not_hash_excluded() -> None:
    baseline_identity = _identity()
    baseline_core = package_core_identity_sha256_v1(baseline_identity)
    compatibility_identity = {
        **baseline_identity,
        "legacy_compatibility": {
            "ticks_per_generation": 160,
            "maximum_logical_generations": 200,
        },
    }
    assert package_core_identity_sha256_v1(compatibility_identity) != baseline_core

    baseline_receipt = {
        "identity": baseline_identity,
        "obligations": {"required": {"status": "passed"}},
    }
    baseline_package_id = package_identity_sha256_v1(baseline_receipt)
    compatibility_receipt = {
        **baseline_receipt,
        "legacy_compatibility": {
            "ticks_per_generation": 160,
            "maximum_logical_generations": 200,
        },
        "identity": compatibility_identity,
    }
    assert package_identity_sha256_v1(compatibility_receipt) != baseline_package_id


def test_active_package_identity_has_no_generation_authority() -> None:
    required = executable_package_contract_v1()["required_identity_fields"]
    assert all(
        "generation" not in field and "ticks_per_generation" not in field
        for field in required
    )
    assert "supported_world_controller_absolute_tick_envelope" in required
    assert "supported_checkpoint_cut_ticks" in required
    assert "checkpoint_authority_contract_sha256" in required
    assert "supported_world_controller_config_envelope" not in required
    assert "supported_checkpoint_cut_points" not in required
