from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from emergence_contracts import (
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
    strategy_attribution_configuration_v1,
    validate_strategy_attribution_configuration_v1,
)
from engine.perf.cuda_full_evolution_v2.science_native_world_runtime_v1 import (
    founder_profile_policy_bindings_v1,
)
from emergence_evidence.science_strategy_attribution_v1 import (
    _StreamingAgent,
    _stream_add_offspring_survival,
    classify_causal_class_v1,
    strategy_group_id_v1,
    strategy_union_v1,
)


def _binding() -> dict[str, object]:
    return {
        "schema_version": SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
        "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
        "identity_mode": "declared_policy_mapping_v1",
        "lineage_mode": "parent_ancestry_v1",
        "strategy_ids": ["alpha", "beta"],
        "policy_id_to_strategy_id": {
            "sha256:" + "a" * 64: "alpha",
            "sha256:" + "b" * 64: "beta",
        },
        "denominator_fields": list(SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1),
    }


def test_explicit_binding_is_valid_only_with_all_generic_observation_tokens() -> None:
    configuration = {"strategy_attribution": _binding()}
    required = list(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1)
    assert strategy_attribution_configuration_v1(configuration) == _binding()
    assert validate_strategy_attribution_configuration_v1(
        configuration, required_observations=required
    ) == ()

    for omitted in SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1:
        reduced = [token for token in required if token != omitted]
        errors = validate_strategy_attribution_configuration_v1(
            configuration, required_observations=reduced
        )
        assert f"required_observations:{omitted}" in errors


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_kinds", ["birth"]),
        ("identity_mode", "unknown_identity_mode_v1"),
        ("lineage_mode", "founder_only_v1"),
        ("denominator_fields", ["birth_attempt_count"]),
        ("policy_id_to_strategy_id", {}),
    ],
)
def test_binding_rejects_incomplete_or_ambiguous_join_contract(
    field: str, value: object
) -> None:
    binding = _binding()
    binding[field] = value
    errors = validate_strategy_attribution_configuration_v1(
        {"strategy_attribution": binding},
        required_observations=list(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1),
    )
    assert any(field in error for error in errors)


def test_unconfigured_reducers_remain_backward_compatible() -> None:
    assert strategy_attribution_configuration_v1({"ordinary": True}) is None
    assert validate_strategy_attribution_configuration_v1(
        {"ordinary": True}, required_observations=[]
    ) == ()


def test_policy_bindings_are_derived_from_reusable_founder_registry() -> None:
    bindings = founder_profile_policy_bindings_v1(
        (
            "native-neutral-v1",
            "native-resource-specialist-v1",
            "native-movement-specialist-v1",
            "native-risk-resistant-v1",
        )
    )
    assert set(bindings.values()) == {
        "native-neutral-v1",
        "native-resource-specialist-v1",
        "native-movement-specialist-v1",
        "native-risk-resistant-v1",
    }
    assert all(key.startswith("sha256:") and len(key) == 71 for key in bindings)


def test_required_observation_type_is_fail_closed() -> None:
    configuration = {"strategy_attribution": _binding()}
    errors = validate_strategy_attribution_configuration_v1(
        configuration, required_observations={"not": "a token list"}
    )
    assert "required_observations.type" in errors


def _result(
    *,
    active_count: int,
    births: int,
    eligible: int,
    first_birth: int | None = None,
    first_death: int | None = None,
    extinction: int | None = None,
    pressure_onset: int | None = None,
    mixed: bool = False,
) -> dict:
    return {
        "time_to_first_birth": {"absolute_tick": first_birth},
        "time_to_first_death": {"absolute_tick": first_death},
        "extinction": {"absolute_tick": extinction},
        "survival_curve": [{"absolute_tick": 10_000, "active_count": active_count}],
        "event_counts": {"birth": births, "death": 0},
        "reproduction_funnel": {
            "all_agents": {
                "reproduction_eligible_count": eligible,
                "mate_candidate_count": eligible,
                "birth_attempt_count": births,
            }
        },
        "initial_composition": {"mixed": mixed},
        "pressure_exposure": {"onset_absolute_tick": pressure_onset},
    }


def test_strategy_union_is_deterministic_and_preserves_mixed_identity() -> None:
    assert strategy_union_v1(("z", "a"), ("a",)) == ("a", "z")
    assert strategy_group_id_v1(("z", "a")) == "a+z"


def test_streaming_persistence_keeps_founders_and_excludes_them_from_offspring_denominator() -> None:
    state = SimpleNamespace(
        persistence_to_early=defaultdict(int),
        offspring_survived=defaultdict(lambda: defaultdict(int)),
    )
    founder = _StreamingAgent(("alpha",), birth_tick=0, lineage_depth=0, founder=True)
    child = _StreamingAgent(("alpha", "beta"), birth_tick=2, lineage_depth=1)

    _stream_add_offspring_survival(
        state, founder, death_tick=None, thresholds=(10,), early_life_tick=10
    )
    _stream_add_offspring_survival(
        state, child, death_tick=None, thresholds=(10,), early_life_tick=10
    )

    assert dict(state.persistence_to_early) == {"alpha": 2, "beta": 1}
    assert dict(state.offspring_survived["alpha"]) == {"10": 1}
    assert dict(state.offspring_survived["beta"]) == {"10": 1}


def test_causal_classification_a_baseline_death() -> None:
    result = classify_causal_class_v1(
        _result(active_count=0, births=0, eligible=0, first_death=4, extinction=4)
    )
    assert result["class"] == "A"


def test_causal_classification_b_survives_without_reproduction_opportunity() -> None:
    result = classify_causal_class_v1(_result(active_count=256, births=0, eligible=0))
    assert result["class"] == "B"


def test_causal_classification_c_births_but_no_early_persistence() -> None:
    result = classify_causal_class_v1(
        _result(active_count=0, births=3, eligible=4, first_birth=2, extinction=30)
    )
    assert result["class"] == "C"


def test_causal_classification_d_pressure_contrast() -> None:
    result = classify_causal_class_v1(
        _result(
            active_count=0,
            births=3,
            eligible=4,
            first_birth=2,
            extinction=2_000,
            pressure_onset=1_000,
        ),
        pressure_control_survives=True,
    )
    assert result["class"] == "D"


def test_causal_classification_e_mixed_persistence() -> None:
    result = classify_causal_class_v1(
        _result(active_count=12, births=3, eligible=4, mixed=True)
    )
    assert result["class"] == "E"
