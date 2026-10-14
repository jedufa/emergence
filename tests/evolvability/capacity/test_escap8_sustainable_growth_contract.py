from __future__ import annotations

from copy import deepcopy

import pytest

from agents.structural_identity_v1 import new_native_identity
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    neutral_developmental_expression_record,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_OPERATOR,
    GATED_RECORD_KEY,
    neutral_gated_recurrent_record,
)
from evolution.structural_observation_routing_v1 import (
    ROUTING_BLOCK_KEY,
    normalize_routing_block,
)
from evolution.structural_output_routing_v1 import (
    OUTPUT_ROUTING_BLOCK_KEY,
    normalize_output_routing_block,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    COST_COEFFICIENTS,
    DUPLICATION_NEUTRALIZATION_VERSION,
    FIXED_ENVELOPE_SHAPE,
    FUSION_POLICY_VERSION,
    GROWTH_MUTATION_BINS,
    GROWTH_MUTATION_DRAWS,
    REFINEMENT_OPERATOR_VERSION,
    SPECIALIZATION_TELEMETRY_VERSION,
    SustainableGrowthContractError,
    classify_bloat,
    hereditary_no_output,
    is_duplicate_descendant,
    module_is_dormant,
    mutate_sustainable_growth,
    normalize_activity_module_params,
    specialization_difference_summary,
    structural_cost_summary,
)


def _identity(index: int, *, parent: dict | None = None, operator: str = "founder") -> dict:
    return new_native_identity(
        context={"lineage_token": f"escap8-contract-{index}", "generation": index},
        operator_id=operator,
        attempt_index=0,
        operator_role=f"role-{index}",
        parent_module=parent,
        inherited_homology=False,
    )


def _module(
    index: int,
    *,
    module_id: str | None = None,
    output_gain: float = 1.0,
    expression: float = 1.0,
    dormant: bool | None = None,
    parent: dict | None = None,
    operator: str = "founder",
    module_type: str = "generic",
) -> dict:
    params: dict = {
        "expression_scale": expression,
        "live_operator": "bounded_scalar_integrator_v1",
        "live_retention": 0.5,
        "live_input_gain": 1.0,
        "live_output_gain": output_gain,
    }
    if dormant is not None:
        params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": dormant,
        }
    return {
        "module_id": module_id or f"m{index:03d}",
        "module_type": module_type,
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **_identity(index, parent=parent, operator=operator),
    }


def _gated_module(index: int, *, output_gain: float, dormant: bool | None = None) -> dict:
    module = _module(index, output_gain=0.0, dormant=dormant)
    record = neutral_gated_recurrent_record()
    record["output_gain"] = output_gain
    module["params"] = {
        "expression_scale": 1.0,
        "live_operator": GATED_OPERATOR,
        GATED_RECORD_KEY: record,
        **(
            {
                ACTIVITY_RECORD_KEY: {
                    "schema_version": ACTIVITY_SCHEMA_VERSION,
                    "dormant": dormant,
                }
            }
            if dormant is not None
            else {}
        ),
    }
    return module


def _duplicate(index: int, source: dict, *, output_gain: float = 0.0) -> dict:
    return _module(
        index,
        output_gain=output_gain,
        parent=source,
        operator="duplicate_with_decay",
    )


def _connection(cid: str, src: str, dst: str, *, weight: float = 0.5) -> dict:
    return {
        "connection_id": cid,
        "src_module_id": src,
        "dst_module_id": dst,
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "escap8-test",
        "weight": weight,
    }


def _payload(modules: list[dict], connections: list[dict] | None = None) -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": deepcopy(modules),
        "connections": deepcopy(connections or []),
    }


def test_frozen_versions_shape_bins_and_reporting_coefficients() -> None:
    assert DUPLICATION_NEUTRALIZATION_VERSION == "escap8_exact_output_neutral_duplication_v1"
    assert REFINEMENT_OPERATOR_VERSION == "escap8_bounded_duplicate_refinement_v1"
    assert CHECKPOINT_SCHEMA_VERSION == "escap8_sustainable_growth_checkpoint_metadata_v1"
    assert SPECIALIZATION_TELEMETRY_VERSION == "escap8_source_descendant_specialization_metrics_v1"
    assert "deferred" in FUSION_POLICY_VERSION
    assert GROWTH_MUTATION_DRAWS == 2
    assert GROWTH_MUTATION_BINS == (
        (0.0, 0.25, "toggle_dormancy"),
        (0.25, 0.5, "prune_eligible_module"),
        (0.5, 0.75, "refine_descendant_parameter"),
        (0.75, 1.0, "refine_descendant_routing_or_expression"),
    )
    assert FIXED_ENVELOPE_SHAPE["max_module_slots"] == 12
    assert FIXED_ENVELOPE_SHAPE["max_connection_slots"] == 32
    assert FIXED_ENVELOPE_SHAPE["state_width_per_module"] == 1
    assert FIXED_ENVELOPE_SHAPE["observation_width"] == 27
    assert FIXED_ENVELOPE_SHAPE["action_width"] == 7
    assert FIXED_ENVELOPE_SHAPE["mlp_hidden_width"] == 64
    assert FIXED_ENVELOPE_SHAPE["central_gru_hidden_width"] == 64
    assert COST_COEFFICIENTS["active_output_binding"] == 2.0


def test_optional_activity_absence_is_active_and_explicit_record_is_strict() -> None:
    base = {"expression_scale": 1.0}
    assert normalize_activity_module_params(base) == base
    assert ACTIVITY_RECORD_KEY not in normalize_activity_module_params(base)
    assert module_is_dormant({"params": base}) is False
    dormant = {
        **base,
        ACTIVITY_RECORD_KEY: {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": True,
        },
    }
    assert module_is_dormant({"params": dormant}) is True
    normalized = normalize_activity_module_params(dormant)
    assert normalized == dormant
    assert normalized is not dormant


@pytest.mark.parametrize(
    "record",
    [
        None,
        {"schema_version": ACTIVITY_SCHEMA_VERSION},
        {"schema_version": ACTIVITY_SCHEMA_VERSION, "dormant": 1},
        {"schema_version": "wrong", "dormant": True},
        {"schema_version": ACTIVITY_SCHEMA_VERSION, "dormant": True, "extra": 0},
    ],
)
def test_activity_record_rejects_nonmapping_missing_extra_type_and_version(record) -> None:
    with pytest.raises(SustainableGrowthContractError):
        normalize_activity_module_params({ACTIVITY_RECORD_KEY: record})


def test_exact_no_output_and_duplicate_descendant_are_hereditary_only() -> None:
    source = _module(1)
    duplicate = _duplicate(2, source, output_gain=0.0)
    assert hereditary_no_output(duplicate) is True
    assert is_duplicate_descendant(duplicate) is True
    assert is_duplicate_descendant(source) is False
    assert hereditary_no_output(_module(3, output_gain=1.0, expression=0.0)) is True
    assert hereditary_no_output(_gated_module(4, output_gain=0.0)) is True
    assert hereditary_no_output(_gated_module(5, output_gain=1.0)) is False
    assert hereditary_no_output(_module(6, output_gain=1.0, dormant=True)) is True


@pytest.mark.parametrize("draw0", [0.0, 0.249999, 0.25, 0.499999, 0.5, 0.749999, 0.75, 0.999999])
def test_disabled_every_bin_preserves_parent_and_declares_exact_two_draws(draw0: float) -> None:
    parent = _payload([_module(1)])
    result = mutate_sustainable_growth(parent, draw0, 0.5, enabled=False)
    assert result.telemetry["status"] == "disabled"
    assert result.telemetry["draw_budget"] == {
        "declared_supplied_draws": 2,
        "internal_rng_draws": 0,
    }
    assert result.payload["modules"][0]["module_instance_id"] == parent["modules"][0]["module_instance_id"]


def test_toggle_dormancy_is_reversible_hereditary_and_identity_preserving() -> None:
    module = _module(1)
    parent = _payload([module])
    first = mutate_sustainable_growth(parent, 0.10, 0.20)
    assert first.telemetry["status"] == "accepted"
    changed = first.payload["modules"][0]
    assert module_is_dormant(changed) is True
    assert changed["module_instance_id"] == module["module_instance_id"]
    assert changed["homology_id"] == module["homology_id"]
    second = mutate_sustainable_growth(first.payload, 0.10, 0.20)
    assert second.telemetry["status"] == "accepted"
    assert module_is_dormant(second.payload["modules"][0]) is False


def test_atomic_prune_removes_incident_edges_repacks_and_preserves_remaining_identity() -> None:
    left = _module(10, module_id="m900")
    target = _module(11, module_id="m400", dormant=True)
    right = _module(12, module_id="m700")
    parent = _payload(
        [right, target, left],
        [
            _connection("c900", "m900", "m400"),
            _connection("c700", "m900", "m700"),
        ],
    )
    result = mutate_sustainable_growth(parent, 0.30, 0.50)
    assert result.telemetry["status"] == "accepted"
    modules = result.payload["modules"]
    assert [module["module_id"] for module in modules] == ["m000", "m001"]
    assert {module["module_instance_id"] for module in modules} == {
        left["module_instance_id"],
        right["module_instance_id"],
    }
    assert result.telemetry["selected_module_instance_id"] == target["module_instance_id"]
    assert len(result.payload["connections"]) == 1
    assert result.payload["connections"][0]["connection_id"] == "c000"
    assert result.payload["connections"][0]["src_module_id"] in {"m000", "m001"}
    assert result.payload["connections"][0]["dst_module_id"] in {"m000", "m001"}


def test_prune_rejects_dependency_target_and_preserves_canonical_parent() -> None:
    target = _module(1, output_gain=0.0)
    dependent = _module(2)
    record = neutral_developmental_expression_record()
    record["prerequisite_homology_ids"] = [target["homology_id"]]
    dependent["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = record
    parent = _payload([target, dependent])
    result = mutate_sustainable_growth(parent, 0.30, 0.10)
    assert result.telemetry["status"] == "dependency_blocked"
    assert result.telemetry["dependency_blockers"] == [dependent["homology_id"]]
    assert len(result.payload["modules"]) == 2


def test_prune_rejects_last_generic_even_when_non_generic_record_remains() -> None:
    target = _module(1, output_gain=0.0)
    non_generic = _module(2, module_type="policy_core")
    parent = _payload([target, non_generic])
    result = mutate_sustainable_growth(parent, 0.30, 0.50)
    assert result.telemetry["status"] == "ineligible"
    assert result.telemetry["reason"] == "minimum_module_invariant"
    assert len(result.payload["modules"]) == 2


def test_parameter_refinement_diverges_neutral_duplicate_without_identity_change() -> None:
    source = _module(1)
    duplicate = _duplicate(2, source, output_gain=0.0)
    parent = _payload([source, duplicate])
    result = mutate_sustainable_growth(parent, 0.60, 0.95)
    assert result.telemetry["status"] == "accepted"
    changed = next(
        module
        for module in result.payload["modules"]
        if module["module_instance_id"] == duplicate["module_instance_id"]
    )
    assert result.telemetry["selected_field"] == "live_output_gain"
    assert changed["params"]["live_output_gain"] != 0.0
    assert changed["homology_id"] == duplicate["homology_id"]
    assert source["params"]["live_output_gain"] == 1.0
    specialization = result.telemetry["specialization_difference"]
    assert specialization["schema_version"] == SPECIALIZATION_TELEMETRY_VERSION
    assert specialization["reporting_only"] is True
    assert specialization["selection_input"] is False
    assert specialization["descendant_is_duplicate"] is True
    assert specialization["dimensions"]["parameter"]["changed"] is True
    assert specialization["dimensions"]["causal_output"]["measured"] is False
    assert specialization["specialized"] is True


def test_routing_refinement_changes_existing_duplicate_binding_only() -> None:
    source = _module(1)
    duplicate = _duplicate(2, source)
    output = normalize_output_routing_block(None)
    output["bindings"] = [
        {"semantic_id": "rest", "source_slot": "out", "gain": 0.0, "enabled": True}
    ]
    duplicate["params"][OUTPUT_ROUTING_BLOCK_KEY] = normalize_output_routing_block(output)
    parent = _payload([source, duplicate])
    result = mutate_sustainable_growth(parent, 0.90, 0.75)
    assert result.telemetry["status"] == "accepted"
    assert result.telemetry["selected_field"] == "output_binding_gain"
    changed = next(
        module
        for module in result.payload["modules"]
        if module["module_instance_id"] == duplicate["module_instance_id"]
    )
    binding = changed["params"][OUTPUT_ROUTING_BLOCK_KEY]["bindings"][0]
    assert binding["gain"] == pytest.approx(0.5)
    assert OUTPUT_ROUTING_BLOCK_KEY not in source["params"]


def test_specialization_metrics_cover_every_frozen_difference_dimension() -> None:
    source = _module(10)
    descendant = _duplicate(11, source)
    source["params"][ROUTING_BLOCK_KEY] = {
        "bindings": [{"semantic_id": "self_energy", "target_slot": "in", "gain": 0.5}]
    }
    descendant["params"][ROUTING_BLOCK_KEY] = {
        "bindings": [{"semantic_id": "self_energy", "target_slot": "in", "gain": 0.75}]
    }
    source["params"][OUTPUT_ROUTING_BLOCK_KEY] = {
        "bindings": [{"semantic_id": "rest", "source_slot": "out", "gain": 0.5}]
    }
    descendant["params"][OUTPUT_ROUTING_BLOCK_KEY] = {
        "bindings": [{"semantic_id": "rest", "source_slot": "out", "gain": 0.25}]
    }
    source_development = neutral_developmental_expression_record()
    descendant_development = neutral_developmental_expression_record()
    descendant_development["stage_gains"] = [0.5, 1.0, 1.0]
    descendant_development["prerequisite_homology_ids"] = [source["homology_id"]]
    source["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = source_development
    descendant["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = descendant_development
    descendant["params"]["expression_scale"] = 0.75
    descendant["params"]["live_retention"] = 0.25
    connections = [
        {
            "src_module_id": source["module_id"],
            "dst_module_id": descendant["module_id"],
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "channel": "fixture",
            "weight": 0.5,
        }
    ]

    summary = specialization_difference_summary(
        source,
        descendant,
        connections=connections,
        source_causal_output=0.0,
        descendant_causal_output=0.25,
    )

    assert summary["changed_dimensions"] == [
        "causal_output",
        "connection",
        "dependency",
        "development",
        "expression",
        "input_routing",
        "output_routing",
        "parameter",
    ]
    assert summary["changed_dimension_count"] == 8
    assert summary["specialized"] is True


def test_invalid_draw_and_invalid_parent_are_atomic_classified_rejections() -> None:
    parent = _payload([_module(1)])
    bad_draw = mutate_sustainable_growth(parent, 1.0, 0.0)
    assert bad_draw.telemetry["status"] == "rejected"
    assert bad_draw.telemetry["reason"].startswith("invalid_draw:")
    malformed = deepcopy(parent)
    malformed["modules"][0]["params"][ACTIVITY_RECORD_KEY] = {"dormant": True}
    bad_parent = mutate_sustainable_growth(malformed, 0.1, 0.1)
    assert bad_parent.telemetry["status"] == "rejected"
    assert bad_parent.telemetry["reason"].startswith("invalid_parent:")
    assert bad_parent.payload == malformed


def test_reporting_only_cost_is_exact_additive_and_output_binding_cost_is_two() -> None:
    summary = structural_cost_summary(
        allocated_module_count=2,
        executing_integrator_count=1,
        executing_gated_count=1,
        evaluated_connection_count=3,
        active_input_binding_count=2,
        active_output_binding_count=1,
        active_dependency_edge_count=1,
        dormant_module_count=1,
        serialized_genome_bytes=100,
        saturated_or_rejected_activity_count=2,
    )
    assert summary["reporting_only"] is True
    assert summary["components"]["active_output_binding"] == 2.0
    assert summary["components"]["dormant_module_maintenance"] == 0.0
    assert summary["components"]["serialized_genome_byte"] == 0.0
    assert summary["total"] == 49.0
    with pytest.raises(SustainableGrowthContractError):
        structural_cost_summary(
            allocated_module_count=1,
            executing_integrator_count=0,
            executing_gated_count=0,
            evaluated_connection_count=0,
            active_input_binding_count=0,
            active_output_binding_count=0,
            active_dependency_edge_count=0,
            dormant_module_count=2,
            serialized_genome_bytes=0,
            saturated_or_rejected_activity_count=0,
        )


def _classification(**updates) -> str:
    values = {
        "allocated_module_count": 0,
        "module_capacity": 12,
        "active_module_count": 0,
        "dormant_module_count": 0,
        "causally_active_module_count": 0,
        "neutral_duplicate_count": 0,
        "redundant_structure_count": 0,
        "prunable_module_count": 0,
        "blocked_duplication_attempt_count": 0,
        "blocked_prune_attempt_count": 0,
        "cap_contact_count": 0,
        "sustained_cap_pressure": False,
    }
    values.update(updates)
    result = classify_bloat(**values)
    assert result["reporting_only"] is True
    return result["classification"]


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, "healthy_headroom"),
        (
            {
                "allocated_module_count": 2,
                "active_module_count": 2,
                "causally_active_module_count": 1,
            },
            "active_growth",
        ),
        (
            {"allocated_module_count": 2, "dormant_module_count": 1},
            "dormant_reserve",
        ),
        (
            {
                "allocated_module_count": 2,
                "active_module_count": 2,
                "neutral_duplicate_count": 1,
            },
            "growth_without_activity",
        ),
        (
            {
                "allocated_module_count": 12,
                "active_module_count": 12,
                "causally_active_module_count": 8,
                "prunable_module_count": 1,
                "sustained_cap_pressure": True,
            },
            "pruning_pressure",
        ),
        (
            {
                "allocated_module_count": 12,
                "active_module_count": 12,
                "causally_active_module_count": 8,
                "blocked_duplication_attempt_count": 2,
                "sustained_cap_pressure": True,
            },
            "capacity_limited",
        ),
        (
            {
                "allocated_module_count": 12,
                "active_module_count": 2,
                "causally_active_module_count": 1,
                "neutral_duplicate_count": 5,
                "redundant_structure_count": 2,
                "sustained_cap_pressure": True,
            },
            "possible_bloat",
        ),
    ],
)
def test_bloat_reporting_labels_are_deterministic(updates: dict, expected: str) -> None:
    assert _classification(**updates) == expected


def test_single_cap_touch_alone_is_not_capacity_pressure() -> None:
    assert _classification(
        allocated_module_count=12,
        active_module_count=12,
        causally_active_module_count=12,
        cap_contact_count=1,
        sustained_cap_pressure=False,
    ) == "active_growth"
