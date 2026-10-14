from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from evolution.structural_gated_recurrent_v1 import (
    GATED_DRAW1_BINS,
    GATED_NUMERIC_FIELDS,
    GATED_OPERATOR,
    GATED_PARAMETER_BOUNDS,
    GATED_RECORD_FIELDS,
    GATED_RECORD_KEY,
    GATED_RECORD_SCHEMA_VERSION,
    GATED_STATE_WIDTH,
    GatedRecordError,
    hereditary_neutral_integrator_eligible,
    nested_interval_choice,
    neutral_gated_recurrent_record,
    normalize_gated_module_params,
    normalize_gated_recurrent_record,
    stable_sigmoid_f32,
    stable_tanh_f32,
    step_gated_scalar,
)


def _f32(value: float) -> float:
    return float(np.float32(value))


def _clip(value: float, lower: float, upper: float) -> float:
    return _f32(min(_f32(upper), max(_f32(lower), _f32(value))))


def _sigmoid(value: float) -> float:
    x = _clip(value, -16.0, 16.0)
    if x >= 0.0:
        exponent = _f32(math.exp(-x))
        denominator = _f32(_f32(1.0) + exponent)
        return _f32(_f32(1.0) / denominator)
    exponent = _f32(math.exp(x))
    denominator = _f32(_f32(1.0) + exponent)
    return _f32(exponent / denominator)


def _expected_step(
    *,
    external: float,
    routed: float,
    previous: float,
    expression: float,
    development: float,
    record: dict,
) -> dict[str, float]:
    external_f = _f32(external)
    routed_f = _f32(routed)
    previous_f = _f32(previous)
    x = _clip(_f32(external_f + routed_f), -1.0, 1.0)

    reset_input = _f32(_f32(record["reset_input_gain"]) * x)
    reset_state = _f32(_f32(record["reset_state_gain"]) * previous_f)
    reset_pre = _clip(
        _f32(_f32(_f32(record["reset_bias"]) + reset_input) + reset_state),
        -16.0,
        16.0,
    )
    reset = _sigmoid(reset_pre)

    update_input = _f32(_f32(record["update_input_gain"]) * x)
    update_state = _f32(_f32(record["update_state_gain"]) * previous_f)
    update_pre = _clip(
        _f32(_f32(_f32(record["update_bias"]) + update_input) + update_state),
        -16.0,
        16.0,
    )
    update = _sigmoid(update_pre)

    reset_previous = _f32(reset * previous_f)
    candidate_input = _f32(_f32(record["candidate_input_gain"]) * x)
    candidate_state = _f32(
        _f32(record["candidate_state_gain"]) * reset_previous
    )
    candidate_pre = _clip(
        _f32(
            _f32(_f32(record["candidate_bias"]) + candidate_input)
            + candidate_state
        ),
        -16.0,
        16.0,
    )
    candidate = _f32(math.tanh(candidate_pre))

    keep = _f32(_f32(_f32(1.0) - update) * previous_f)
    write = _f32(update * candidate)
    state = _clip(_f32(keep + write), -1.0, 1.0)
    base = _clip(_f32(_f32(record["output_gain"]) * state), -1.0, 1.0)
    expressed = _clip(_f32(_f32(expression) * base), -1.0, 1.0)
    effective = (
        expressed
        if _f32(development) == 1.0
        else _clip(_f32(_f32(development) * expressed), -1.0, 1.0)
    )
    return {
        "external_input": external_f,
        "routed_previous": routed_f,
        "input_aggregate": x,
        "reset_pre": reset_pre,
        "reset_gate": reset,
        "update_pre": update_pre,
        "update_gate": update,
        "candidate_pre": candidate_pre,
        "candidate": candidate,
        "state_before": previous_f,
        "state_after": state,
        "base_output": base,
        "expressed_output": expressed,
        "effective_output": effective,
    }


def _record(**updates: float) -> dict:
    record = neutral_gated_recurrent_record()
    record.update(updates)
    return record


def _integrator_module(*, expression: float, output_gain: float) -> dict:
    return {
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": expression,
            "live_retention": 0.5,
            "live_input_gain": 1.0,
            "live_output_gain": output_gain,
        },
    }


def test_strict_record_has_exact_fixed_fields_and_float32_values() -> None:
    raw = _record(reset_bias=1.125, output_gain=-1.75)
    normalized = normalize_gated_recurrent_record(raw)

    assert tuple(normalized) == ("schema_version", *GATED_NUMERIC_FIELDS)
    assert set(normalized) == set(GATED_RECORD_FIELDS)
    assert normalized["schema_version"] == GATED_RECORD_SCHEMA_VERSION
    assert normalized["reset_bias"] == _f32(1.125)
    assert normalized["output_gain"] == _f32(-1.75)
    assert GATED_STATE_WIDTH == 1


@pytest.mark.parametrize(
    "mutator",
    (
        lambda record: record.pop("reset_bias"),
        lambda record: record.__setitem__("unknown", 0.0),
        lambda record: record.__setitem__("schema_version", "wrong"),
        lambda record: record.__setitem__("reset_bias", float("nan")),
        lambda record: record.__setitem__("update_bias", float("inf")),
        lambda record: record.__setitem__("candidate_bias", -4.01),
        lambda record: record.__setitem__("output_gain", 2.01),
        lambda record: record.__setitem__("output_gain", True),
    ),
)
def test_strict_record_rejects_missing_extra_version_nonfinite_and_bounds(mutator) -> None:
    record = _record()
    mutator(record)
    with pytest.raises(GatedRecordError):
        normalize_gated_recurrent_record(record)


def test_stale_record_on_integrator_and_missing_record_on_gated_fail_closed() -> None:
    with pytest.raises(GatedRecordError, match="STALE_OPERATOR_PAYLOAD"):
        normalize_gated_module_params(
            {
                "live_operator": "bounded_scalar_integrator_v1",
                GATED_RECORD_KEY: _record(),
            }
        )
    with pytest.raises(GatedRecordError, match="MALFORMED_RECORD"):
        normalize_gated_module_params({"live_operator": GATED_OPERATOR})


@pytest.mark.parametrize(
    ("value", "sigmoid", "tanh"),
    (
        (-16, 1.1253516163378663e-07, -1.0),
        (-4, 0.01798621006309986, -0.9993293285369873),
        (-1, 0.2689414322376251, -0.7615941762924194),
        (0, 0.5, 0.0),
        (1, 0.7310585975646973, 0.7615941762924194),
        (4, 0.9820137619972229, 0.9993293285369873),
        (16, 0.9999998807907104, 1.0),
    ),
)
def test_stable_nonlinear_reference_values_are_exact(
    value: int,
    sigmoid: float,
    tanh: float,
) -> None:
    assert stable_sigmoid_f32(value) == sigmoid
    assert stable_tanh_f32(value) == tanh


@pytest.mark.parametrize(
    ("external", "routed", "previous"),
    (
        (0.0, 0.0, 0.0),
        (0.75, 0.5, -0.4),
        (-0.75, -0.5, 0.4),
        (0.25, -0.5, -0.75),
    ),
)
def test_exact_gated_equation_matches_frozen_float_order(
    external: float,
    routed: float,
    previous: float,
) -> None:
    record = _record(
        reset_bias=-0.75,
        reset_input_gain=1.25,
        reset_state_gain=-2.0,
        update_bias=0.5,
        update_input_gain=-1.5,
        update_state_gain=0.75,
        candidate_bias=-0.25,
        candidate_input_gain=2.5,
        candidate_state_gain=-3.0,
        output_gain=1.5,
    )
    actual = step_gated_scalar(
        external_input=external,
        routed_previous=routed,
        previous_state=previous,
        expression_scale=-1.25,
        developmental_gain=0.5,
        record=record,
    )
    expected = _expected_step(
        external=external,
        routed=routed,
        previous=previous,
        expression=-1.25,
        development=0.5,
        record=record,
    )
    assert actual.__dict__ == expected


def test_parameter_boundaries_and_input_preactivation_state_output_clips() -> None:
    record = _record(
        **{
            field: bounds[1]
            for field, bounds in GATED_PARAMETER_BOUNDS.items()
        }
    )
    result = step_gated_scalar(
        external_input=1.0,
        routed_previous=1.0,
        previous_state=1.0,
        expression_scale=2.0,
        developmental_gain=1.0,
        record=record,
    )
    assert result.input_aggregate == 1.0
    assert result.reset_pre == 12.0
    assert result.update_pre == 12.0
    assert -1.0 <= result.state_after <= 1.0
    assert -1.0 <= result.base_output <= 1.0
    assert -1.0 <= result.effective_output <= 1.0
    assert stable_sigmoid_f32(1000.0) == stable_sigmoid_f32(16.0)
    assert stable_tanh_f32(-1000.0) == stable_tanh_f32(-16.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("external_input", float("nan")),
        ("routed_previous", float("inf")),
        ("previous_state", float("nan")),
        ("expression_scale", float("inf")),
        ("developmental_gain", float("nan")),
    ),
)
def test_equation_rejects_every_nonfinite_dynamic_surface(field: str, value: float) -> None:
    kwargs = {
        "external_input": 0.0,
        "routed_previous": 0.0,
        "previous_state": 0.0,
        "expression_scale": 1.0,
        "developmental_gain": 1.0,
        "record": _record(),
    }
    kwargs[field] = value
    with pytest.raises(GatedRecordError, match="NONFINITE"):
        step_gated_scalar(**kwargs)


def test_neutral_acquisition_eligibility_is_hereditary_not_runtime_based() -> None:
    assert hereditary_neutral_integrator_eligible(
        _integrator_module(expression=0.0, output_gain=1.0)
    )
    assert hereditary_neutral_integrator_eligible(
        _integrator_module(expression=1.0, output_gain=0.0)
    )
    assert not hereditary_neutral_integrator_eligible(
        _integrator_module(expression=1.0, output_gain=1.0)
    )
    gated = _integrator_module(expression=0.0, output_gain=0.0)
    gated["params"]["live_operator"] = GATED_OPERATOR
    gated["params"][GATED_RECORD_KEY] = _record()
    assert not hereditary_neutral_integrator_eligible(gated)


def test_neutral_gated_record_is_public_output_neutral_with_latent_state_evolution() -> None:
    neutral = neutral_gated_recurrent_record()
    first = step_gated_scalar(
        external_input=1.0,
        routed_previous=1.0,
        previous_state=0.8,
        expression_scale=2.0,
        developmental_gain=1.0,
        record=neutral,
    )
    assert first.reset_gate == 0.5
    assert first.update_gate == 0.5
    assert first.candidate == 0.0
    assert first.state_after == _f32(0.5 * _f32(0.8))
    assert first.base_output == first.expressed_output == first.effective_output == 0.0

    exposed_record = deepcopy(neutral)
    exposed_record["output_gain"] = 1.0
    exposed = step_gated_scalar(
        external_input=0.0,
        routed_previous=0.0,
        previous_state=first.state_after,
        expression_scale=1.0,
        developmental_gain=1.0,
        record=exposed_record,
    )
    assert exposed.state_after == _f32(0.5 * first.state_after)
    assert exposed.effective_output == exposed.state_after


def test_five_draw_bins_are_contiguous_equal_and_have_no_revert() -> None:
    assert GATED_DRAW1_BINS == (
        (0.0, 0.2, "acquire_neutral"),
        (0.2, 0.4, "mutate_reset_parameter"),
        (0.4, 0.6, "mutate_update_parameter"),
        (0.6, 0.8, "mutate_candidate_parameter"),
        (0.8, 1.0, "mutate_output_gain"),
    )
    assert all("revert" not in operation for _, _, operation in GATED_DRAW1_BINS)


@pytest.mark.parametrize(
    ("draw", "count", "index", "remainder"),
    (
        (0.0, 3, 0, 0.0),
        (np.nextafter(np.float32(1.0 / 3.0), np.float32(0.0)), 3, 0, 0.9999999403953552),
        (np.float32(1.0 / 3.0), 3, 1, 2.9802322387695312e-08),
        (np.nextafter(np.float32(2.0 / 3.0), np.float32(0.0)), 3, 1, 0.9999998807907104),
        (np.float32(2.0 / 3.0), 3, 2, 5.960464477539063e-08),
        (np.nextafter(np.float32(1.0), np.float32(0.0)), 3, 2, 0.9999998211860657),
    ),
)
def test_nested_interval_decomposition_boundaries_are_exact(
    draw: float,
    count: int,
    index: int,
    remainder: float,
) -> None:
    assert nested_interval_choice(draw, count, field="test") == (index, remainder)


@pytest.mark.parametrize("draw", (-0.1, 1.0, float("nan")))
def test_nested_interval_decomposition_rejects_invalid_draws(draw: float) -> None:
    with pytest.raises(GatedRecordError):
        nested_interval_choice(draw, 3, field="test")
