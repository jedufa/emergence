from __future__ import annotations

from copy import deepcopy

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.inheritance_cross_path_qualification_controls_v1 import (
    InheritanceCrossPathQualificationControlError,
    ResidentCapacityFaultControlV1,
    ResidentCapacityFaultObservationV1,
    ResidentFixedActionControlV1,
    ResidentFixedActionObservationV1,
)
from engine.env.gridworld_pkg.constants import ACT_REST


def _fault_control() -> ResidentCapacityFaultControlV1:
    return ResidentCapacityFaultControlV1(
        target_occurrence_id="occurrence-2",
        target_occurrence_authority_sha256="1" * 64,
        target_cell_index=1,
        target_cell_authority_sha256="2" * 64,
        capacity_kind="attempt",
        nominal_bound=8,
        effective_bound=1,
        expected_status_code=2,
        pre_fault_authority_sha256="3" * 64,
    )


def _fault_observation(
    control: ResidentCapacityFaultControlV1,
) -> ResidentCapacityFaultObservationV1:
    return ResidentCapacityFaultObservationV1(
        control_authority_sha256=control.control_authority_sha256,
        target_occurrence_authority_sha256=(
            control.target_occurrence_authority_sha256
        ),
        target_cell_index=control.target_cell_index,
        target_cell_authority_sha256=control.target_cell_authority_sha256,
        pre_fault_authority_sha256=control.pre_fault_authority_sha256,
        status_before_code=0,
        status_after_code=control.expected_status_code,
        status_after_drain_code=control.expected_status_code,
        mechanics_before_sha256="4" * 64,
        mechanics_after_sha256="4" * 64,
        peer_count=2,
        peer_set_before_sha256="5" * 64,
        peer_set_after_sha256="5" * 64,
    )


def _fixed_control() -> ResidentFixedActionControlV1:
    return ResidentFixedActionControlV1(
        cell_mask_sha256="6" * 64,
        selected_cell_count=3,
    )


def _fixed_observation(
    control: ResidentFixedActionControlV1,
) -> ResidentFixedActionObservationV1:
    return ResidentFixedActionObservationV1(
        control_authority_sha256=control.control_authority_sha256,
        cell_mask_sha256=control.cell_mask_sha256,
        action_id=int(ACT_REST),
        observed_action_count=12,
        all_rest_actions_sha256=canonical_sha256([int(ACT_REST)] * 12),
        policy_evaluation_count_before=9,
        policy_evaluation_count_after=9,
        policy_evaluation_count_delta=0,
        semantic_draw_count_before=27,
        semantic_draw_count_after=27,
        semantic_draw_count_delta=0,
    )


def test_fault_contracts_round_trip_and_bind_target_authority() -> None:
    control = _fault_control()
    parsed_control = ResidentCapacityFaultControlV1.from_payload(
        control.payload()
    )
    observation = _fault_observation(parsed_control)
    parsed_observation = ResidentCapacityFaultObservationV1.from_payload(
        observation.payload()
    )

    parsed_observation.validate_against(parsed_control)
    assert parsed_control == control
    assert parsed_observation == observation


def test_fault_control_rejects_digest_tamper_and_extra_fields() -> None:
    payload = _fault_control().payload()
    payload["effective_bound"] = 2

    with pytest.raises(
        InheritanceCrossPathQualificationControlError,
        match="does not bind",
    ):
        ResidentCapacityFaultControlV1.from_payload(payload)

    payload = _fault_control().payload()
    payload["campaign_name"] = "not-authorized"
    with pytest.raises(
        InheritanceCrossPathQualificationControlError,
        match="unexpected",
    ):
        ResidentCapacityFaultControlV1.from_payload(payload)


@pytest.mark.parametrize(
    ("changed_field", "message"),
    (
        ("status_after_drain_code", "sticky"),
        ("mechanics_after_sha256", "mechanics changed"),
        ("peer_set_after_sha256", "peers changed"),
    ),
)
def test_fault_observation_rejects_nonisolated_or_nonsticky_result(
    changed_field: str,
    message: str,
) -> None:
    payload = _fault_observation(_fault_control()).payload()
    if changed_field == "status_after_drain_code":
        payload[changed_field] = 3
    else:
        payload[changed_field] = "9" * 64
    payload.pop("observation_authority_sha256")

    with pytest.raises(
        InheritanceCrossPathQualificationControlError,
        match=message,
    ):
        ResidentCapacityFaultObservationV1(
            **{
                key: value
                for key, value in payload.items()
                if key not in {"schema_version", "authority_scope"}
            }
        )


def test_fault_observation_rejects_another_target_control() -> None:
    control = _fault_control()
    observation = _fault_observation(control)
    other_payload = deepcopy(control.payload())
    other_payload["target_cell_index"] = 2
    other_payload.pop("control_authority_sha256")
    other = ResidentCapacityFaultControlV1(
        **{
            key: value
            for key, value in other_payload.items()
            if key not in {"schema_version", "authority_scope"}
        }
    )

    with pytest.raises(
        InheritanceCrossPathQualificationControlError,
        match="does not match",
    ):
        observation.validate_against(other)


def test_fixed_action_contracts_round_trip_and_bind_rest_mask() -> None:
    control = _fixed_control()
    parsed_control = ResidentFixedActionControlV1.from_payload(
        control.payload()
    )
    observation = _fixed_observation(parsed_control)
    parsed_observation = ResidentFixedActionObservationV1.from_payload(
        observation.payload()
    )

    parsed_observation.validate_against(parsed_control)
    assert parsed_control.action_id == int(ACT_REST)
    assert parsed_observation.policy_evaluation_count_delta == 0
    assert parsed_observation.semantic_draw_count_delta == 0


@pytest.mark.parametrize(
    ("field", "after", "delta", "message"),
    (
        (
            "policy_evaluation_count",
            10,
            1,
            "policy evaluation delta",
        ),
        ("semantic_draw_count", 28, 1, "semantic draw delta"),
    ),
)
def test_fixed_action_observation_rejects_policy_work(
    field: str,
    after: int,
    delta: int,
    message: str,
) -> None:
    control = _fixed_control()
    observation = _fixed_observation(control)
    values = {
        key: value
        for key, value in observation.authority_payload().items()
        if key not in {"schema_version", "authority_scope"}
    }
    values[f"{field}_after"] = after
    values[f"{field}_delta"] = delta

    with pytest.raises(
        InheritanceCrossPathQualificationControlError,
        match=message,
    ):
        ResidentFixedActionObservationV1(**values)


def test_fixed_action_observation_rejects_non_rest_action_digest() -> None:
    control = _fixed_control()
    values = {
        key: value
        for key, value in _fixed_observation(control).authority_payload().items()
        if key not in {"schema_version", "authority_scope"}
    }
    values["all_rest_actions_sha256"] = canonical_sha256(
        [int(ACT_REST), 0] + [int(ACT_REST)] * 10
    )

    with pytest.raises(
        InheritanceCrossPathQualificationControlError,
        match="all-REST action vector",
    ):
        ResidentFixedActionObservationV1(**values)


def test_fixed_action_observation_rejects_wrong_mask_control() -> None:
    control = _fixed_control()
    observation = _fixed_observation(control)
    other = ResidentFixedActionControlV1(
        cell_mask_sha256="a" * 64,
        selected_cell_count=control.selected_cell_count,
    )

    with pytest.raises(
        InheritanceCrossPathQualificationControlError,
        match="does not match",
    ):
        observation.validate_against(other)
