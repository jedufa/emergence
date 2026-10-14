from __future__ import annotations

"""ESCAP-7 fixed-width gated scalar recurrent contract.

This module owns the strict hereditary record and the pure CPU/reference
equation.  It does not allocate runtime state and consumes no RNG.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


GATED_OPERATOR = "gated_scalar_recurrent_v1"
GATED_RECORD_KEY = "gated_recurrent"
GATED_RECORD_SCHEMA_VERSION = "escap7_gated_scalar_recurrent_params_v1"
GATED_EQUATION_VERSION = "escap7_reset_update_candidate_scalar_v1"
GATED_NUMERIC_VERSION = "escap7_stable_f32_clip16_v1"
GATED_STATE_WIDTH = 1
GATED_PROFILE = "escap7_fixed_width_gated_recurrent_v1"
GATED_RUNTIME_SCHEMA_VERSION = "escap7_fixed_width_gated_recurrent_runtime_v1"
GATED_NEUTRAL_TEMPLATE_VERSION = "escap7_gated_neutral_output_v1"
GATED_NEUTRAL_ELIGIBILITY_VERSION = "escap7_hereditary_output_neutral_integrator_v1"
GATED_LATENT_STATE_POLICY_VERSION = "escap7_public_output_neutral_latent_state_evolves_v1"
GATED_DRAW_DECOMPOSITION_VERSION = "escap7_nested_interval_decomposition_v1"
GATED_MUTATION_PROFILE = "escap7_gated_scalar_operator_mutation_v1"
GATED_MUTATION_BUDGET_VERSION = "escap7_post_escap6_one_attempt_two_draws_v1"
GATED_OPERATION_BUDGET_VERSION = "escap7_gated_scalar_32_ops_reporting_v1"
GATED_DUPLICATION_POLICY_VERSION = "escap7_duplicate_output_gain_zero_v1"
GATED_PROMOTER_POLICY_VERSION = "escap7_preserve_gated_operator_v1"
GATED_FUSION_POLICY_VERSION = "escap7_full_proposal_dependency_validation_v1"
GATED_C2_COMPATIBILITY_VERSION = "escap7_strict_gated_atomic_module_v1"
GATED_SUPPORTED_LANE = "escap7_python_reference_cpu_stride1_v1"
GATED_EXECUTION_OPERATION_ESTIMATE = 32

GATED_RESET_FIELDS = ("reset_bias", "reset_input_gain", "reset_state_gain")
GATED_UPDATE_FIELDS = ("update_bias", "update_input_gain", "update_state_gain")
GATED_CANDIDATE_FIELDS = (
    "candidate_bias",
    "candidate_input_gain",
    "candidate_state_gain",
)
GATED_OUTPUT_FIELDS = ("output_gain",)
GATED_NUMERIC_FIELDS = (
    *GATED_RESET_FIELDS,
    *GATED_UPDATE_FIELDS,
    *GATED_CANDIDATE_FIELDS,
    *GATED_OUTPUT_FIELDS,
)
GATED_RECORD_FIELDS = ("schema_version", *GATED_NUMERIC_FIELDS)
GATED_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    **{name: (-4.0, 4.0) for name in (*GATED_RESET_FIELDS, *GATED_UPDATE_FIELDS, *GATED_CANDIDATE_FIELDS)},
    "output_gain": (-2.0, 2.0),
}
GATED_PREACTIVATION_BOUND = 16.0
GATED_STATE_BOUND = 1.0
GATED_DRAW1_BINS = (
    (0.0, 0.2, "acquire_neutral"),
    (0.2, 0.4, "mutate_reset_parameter"),
    (0.4, 0.6, "mutate_update_parameter"),
    (0.6, 0.8, "mutate_candidate_parameter"),
    (0.8, 1.0, "mutate_output_gain"),
)


class GatedRecordError(ValueError):
    """A deterministic ESCAP-7 hereditary-record rejection."""


def _record_fail(code: str, detail: str) -> None:
    raise GatedRecordError(f"{code} {detail}")


def f32(value: Any, *, field: str = "value") -> float:
    if isinstance(value, (bool, np.bool_)):
        _record_fail("ESCAP7_INVALID_NUMERIC", f"field={field};reason=boolean")
    try:
        raw = float(value)
    except Exception as exc:
        raise GatedRecordError(f"ESCAP7_INVALID_NUMERIC field={field}") from exc
    if not math.isfinite(raw):
        _record_fail("ESCAP7_NONFINITE", f"field={field}")
    out = float(np.float32(raw))
    if not math.isfinite(out):
        _record_fail("ESCAP7_NONFINITE", f"field={field};boundary=f32")
    return out


def clip_f32(value: Any, lower: float, upper: float, *, field: str = "value") -> float:
    current = f32(value, field=field)
    lo = f32(lower, field=f"{field}.lower")
    hi = f32(upper, field=f"{field}.upper")
    if lo > hi:
        _record_fail("ESCAP7_INVALID_BOUND", f"field={field};lower={lo};upper={hi}")
    return f32(min(hi, max(lo, current)), field=f"{field}.clipped")


def stable_sigmoid_f32(value: Any) -> float:
    x = clip_f32(
        value,
        -GATED_PREACTIVATION_BOUND,
        GATED_PREACTIVATION_BOUND,
        field="sigmoid_input",
    )
    if x >= 0.0:
        exp_term = f32(math.exp(-x), field="sigmoid_exp")
        denominator = f32(f32(1.0) + exp_term, field="sigmoid_denominator")
        return f32(f32(1.0) / denominator, field="sigmoid_result")
    exp_term = f32(math.exp(x), field="sigmoid_exp")
    denominator = f32(f32(1.0) + exp_term, field="sigmoid_denominator")
    return f32(exp_term / denominator, field="sigmoid_result")


def stable_tanh_f32(value: Any) -> float:
    x = clip_f32(
        value,
        -GATED_PREACTIVATION_BOUND,
        GATED_PREACTIVATION_BOUND,
        field="tanh_input",
    )
    return f32(math.tanh(x), field="tanh_result")


def neutral_gated_recurrent_record() -> dict[str, Any]:
    return {
        "schema_version": GATED_RECORD_SCHEMA_VERSION,
        **{name: 0.0 for name in GATED_NUMERIC_FIELDS},
    }


def normalize_gated_recurrent_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        _record_fail("ESCAP7_MALFORMED_RECORD", "reason=not_mapping")
    actual = {str(key) for key in record}
    expected = set(GATED_RECORD_FIELDS)
    if actual != expected:
        _record_fail(
            "ESCAP7_MALFORMED_RECORD",
            f"missing={sorted(expected - actual)!r};extra={sorted(actual - expected)!r}",
        )
    schema = str(record.get("schema_version", "") or "")
    if schema != GATED_RECORD_SCHEMA_VERSION:
        _record_fail("ESCAP7_SCHEMA_MISMATCH", f"schema_version={schema}")
    normalized: dict[str, Any] = {"schema_version": GATED_RECORD_SCHEMA_VERSION}
    for name in GATED_NUMERIC_FIELDS:
        value = f32(record.get(name), field=name)
        lower, upper = GATED_PARAMETER_BOUNDS[name]
        if not lower <= value <= upper:
            _record_fail("ESCAP7_PARAM_OUT_OF_RANGE", f"field={name};value={value}")
        normalized[name] = value
    return normalized


def normalize_gated_module_params(params: Any) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        _record_fail("ESCAP7_MALFORMED_PARAMS", "reason=not_mapping")
    normalized = dict(params)
    operator = str(normalized.get("live_operator", "bounded_scalar_integrator_v1") or "bounded_scalar_integrator_v1")
    if operator == GATED_OPERATOR:
        normalized["live_operator"] = GATED_OPERATOR
        normalized[GATED_RECORD_KEY] = normalize_gated_recurrent_record(
            normalized.get(GATED_RECORD_KEY)
        )
    elif GATED_RECORD_KEY in normalized:
        _record_fail(
            "ESCAP7_STALE_OPERATOR_PAYLOAD",
            f"operator={operator};record_key={GATED_RECORD_KEY}",
        )
    return normalized


def hereditary_neutral_integrator_eligible(module: Any) -> bool:
    if not isinstance(module, Mapping) or str(module.get("module_type", "")) != "generic":
        return False
    params = module.get("params", {})
    if not isinstance(params, Mapping) or GATED_RECORD_KEY in params:
        return False
    operator = str(params.get("live_operator", "bounded_scalar_integrator_v1") or "bounded_scalar_integrator_v1")
    if operator != "bounded_scalar_integrator_v1":
        return False
    try:
        expression = f32(params.get("expression_scale", 0.0), field="expression_scale")
        output_gain = f32(params.get("live_output_gain", 0.0), field="live_output_gain")
    except GatedRecordError:
        return False
    return expression == 0.0 or output_gain == 0.0


def _lane_device_type(value: Any) -> str:
    return str(getattr(value, "type", value) or "").strip().lower()


def escap7_unsupported_lane_reasons(
    cfg: Any,
    *,
    policy_device: Any,
    bank_device: Any,
) -> list[str]:
    """Return every enabled non-reference owner or execution lane.

    The scan is deliberately stricter than the older ESCAP validators.  The
    ESCAP-7 contract owns exactly one Python CPU reference lane, so any active
    RunnerCore or CXX/native execution switch is rejected, including shadow or
    pilot switches whose ownership semantics could otherwise drift later.
    """

    reasons: list[str] = []
    policy = _lane_device_type(policy_device)
    bank = _lane_device_type(bank_device)
    if policy != "cpu":
        reasons.append(f"policy_device={policy or 'unknown'}")
    if bank != "cpu":
        reasons.append(f"bank_device={bank or 'unknown'}")
    configured_device = str(getattr(cfg, "device", "auto") or "auto").strip().lower()
    configured_bank = str(getattr(cfg, "bank_device", "auto") or "auto").strip().lower()
    if configured_device in {"cuda", "gpu"} or configured_device.startswith("cuda:"):
        reasons.append(f"configured_device={configured_device}")
    if configured_bank in {"cuda", "gpu"} or configured_bank.startswith("cuda:"):
        reasons.append(f"configured_bank_device={configured_bank}")
    if int(getattr(cfg, "policy_tick_stride", 1) or 1) != 1:
        reasons.append(f"policy_tick_stride={getattr(cfg, 'policy_tick_stride', None)!r}")
    if bool(getattr(cfg, "torch_compile_policy", False)):
        reasons.append("torch_compile_policy")

    dataclass_fields = getattr(type(cfg), "__dataclass_fields__", {}) or {}
    for name, value in sorted(vars(cfg).items()):
        normalized = "".join(ch for ch in name.lower() if ch.isalnum())
        runnercore_family = "runnercore" in normalized
        cxx_family = "cxx" in normalized
        native_cuda_family = any(
            token in normalized for token in ("native", "cuda", "gpu")
        )
        if not (runnercore_family or cxx_family or native_cuda_family):
            continue
        field = dataclass_fields.get(name)
        active_enable_switch = normalized.endswith("enable")
        explicit_lane_boolean = normalized in {
            "perfsynccuda",
            "nativeruntimeenable",
        }
        if type(value) is bool and value and (
            active_enable_switch or explicit_lane_boolean
        ) and (field is None or field.default is not True):
            reasons.append(name)
            continue
        if not any(
            token in normalized
            for token in (
                "ownership",
                "owner",
                "authority",
                "authoritative",
                "profile",
                "candidate",
                "native",
                "directaction",
                "optimization",
                "backend",
                "pilot",
            )
        ):
            continue
        inert_unknown = value is None or value is False or value == 0 or (
            isinstance(value, str)
            and value.strip().lower()
            in {"", "none", "disabled", "off", "false", "0"}
        )
        if field is None and not inert_unknown:
            reasons.append(f"{name}=unknown_non_default")
        elif field is not None and value != field.default:
            reasons.append(f"{name}=non_default")
    return sorted(set(reasons))


def nested_interval_choice(value: Any, count: int, *, field: str) -> tuple[int, float]:
    if isinstance(value, (bool, np.bool_)):
        _record_fail("ESCAP7_INVALID_NUMERIC", f"field={field};reason=boolean")
    try:
        draw = float(value)
    except Exception as exc:
        raise GatedRecordError(f"ESCAP7_INVALID_NUMERIC field={field}") from exc
    size = int(count)
    if size <= 0:
        _record_fail("ESCAP7_EMPTY_CHOICE", f"field={field};count={size}")
    if not math.isfinite(draw):
        _record_fail("ESCAP7_NONFINITE", f"field={field}")
    if not 0.0 <= draw < 1.0:
        _record_fail("ESCAP7_DRAW_OUT_OF_RANGE", f"field={field};value={draw}")
    # Keep the supplied RNG wrapper draw in its native [0,1) precision while
    # partitioning intervals.  Premature float32 rounding would incorrectly
    # turn legal values immediately below one into the forbidden endpoint.
    scaled = draw * size
    index = int(math.floor(scaled))
    if index >= size:
        index = size - 1
    remainder = scaled - index
    if not 0.0 <= remainder < 1.0:
        _record_fail("ESCAP7_DECOMPOSITION_RANGE", f"field={field};remainder={remainder}")
    return index, remainder


@dataclass(frozen=True)
class GatedScalarStep:
    external_input: float
    routed_previous: float
    input_aggregate: float
    reset_pre: float
    reset_gate: float
    update_pre: float
    update_gate: float
    candidate_pre: float
    candidate: float
    state_before: float
    state_after: float
    base_output: float
    expressed_output: float
    effective_output: float


def step_gated_scalar(
    *,
    external_input: Any,
    routed_previous: Any,
    previous_state: Any,
    expression_scale: Any,
    developmental_gain: Any,
    record: Any,
) -> GatedScalarStep:
    params = normalize_gated_recurrent_record(record)
    external = f32(external_input, field="external_input")
    routed = f32(routed_previous, field="routed_previous")
    h_prev = f32(previous_state, field="previous_state")
    if not -GATED_STATE_BOUND <= h_prev <= GATED_STATE_BOUND:
        _record_fail("ESCAP7_STATE_OUT_OF_RANGE", f"value={h_prev}")
    expression = f32(expression_scale, field="expression_scale")
    if not -2.0 <= expression <= 2.0:
        _record_fail("ESCAP7_PARAM_OUT_OF_RANGE", f"field=expression_scale;value={expression}")
    development = f32(developmental_gain, field="developmental_gain")
    if not 0.0 <= development <= 1.0:
        _record_fail("ESCAP7_DEVELOPMENT_GAIN_OUT_OF_RANGE", f"value={development}")

    x_sum = f32(external + routed, field="input_sum")
    x = clip_f32(x_sum, -1.0, 1.0, field="input_aggregate")

    reset_input_term = f32(params["reset_input_gain"] * x, field="reset_input_term")
    reset_state_term = f32(params["reset_state_gain"] * h_prev, field="reset_state_term")
    reset_sum_1 = f32(params["reset_bias"] + reset_input_term, field="reset_sum_1")
    reset_sum_2 = f32(reset_sum_1 + reset_state_term, field="reset_sum_2")
    reset_pre = clip_f32(reset_sum_2, -16.0, 16.0, field="reset_pre")
    reset = stable_sigmoid_f32(reset_pre)

    update_input_term = f32(params["update_input_gain"] * x, field="update_input_term")
    update_state_term = f32(params["update_state_gain"] * h_prev, field="update_state_term")
    update_sum_1 = f32(params["update_bias"] + update_input_term, field="update_sum_1")
    update_sum_2 = f32(update_sum_1 + update_state_term, field="update_sum_2")
    update_pre = clip_f32(update_sum_2, -16.0, 16.0, field="update_pre")
    update = stable_sigmoid_f32(update_pre)

    reset_state = f32(reset * h_prev, field="reset_state")
    candidate_input_term = f32(params["candidate_input_gain"] * x, field="candidate_input_term")
    candidate_state_term = f32(
        params["candidate_state_gain"] * reset_state,
        field="candidate_state_term",
    )
    candidate_sum_1 = f32(params["candidate_bias"] + candidate_input_term, field="candidate_sum_1")
    candidate_sum_2 = f32(candidate_sum_1 + candidate_state_term, field="candidate_sum_2")
    candidate_pre = clip_f32(candidate_sum_2, -16.0, 16.0, field="candidate_pre")
    candidate = stable_tanh_f32(candidate_pre)

    one_minus_update = f32(f32(1.0) - update, field="one_minus_update")
    keep_term = f32(one_minus_update * h_prev, field="keep_term")
    write_term = f32(update * candidate, field="write_term")
    state_sum = f32(keep_term + write_term, field="state_sum")
    h_new = clip_f32(state_sum, -1.0, 1.0, field="state_after")

    raw_output = f32(params["output_gain"] * h_new, field="raw_output")
    base_output = clip_f32(raw_output, -1.0, 1.0, field="base_output")
    expression_product = f32(expression * base_output, field="expression_product")
    expressed_output = clip_f32(expression_product, -1.0, 1.0, field="expressed_output")
    effective_output = (
        expressed_output
        if development == 1.0
        else clip_f32(
            f32(development * expressed_output, field="development_product"),
            -1.0,
            1.0,
            field="effective_output",
        )
    )
    return GatedScalarStep(
        external_input=external,
        routed_previous=routed,
        input_aggregate=x,
        reset_pre=reset_pre,
        reset_gate=reset,
        update_pre=update_pre,
        update_gate=update,
        candidate_pre=candidate_pre,
        candidate=candidate,
        state_before=h_prev,
        state_after=h_new,
        base_output=base_output,
        expressed_output=expressed_output,
        effective_output=effective_output,
    )


__all__ = [
    "GATED_CANDIDATE_FIELDS",
    "GATED_C2_COMPATIBILITY_VERSION",
    "GATED_DRAW1_BINS",
    "GATED_DRAW_DECOMPOSITION_VERSION",
    "GATED_DUPLICATION_POLICY_VERSION",
    "GATED_EQUATION_VERSION",
    "GATED_EXECUTION_OPERATION_ESTIMATE",
    "GATED_FUSION_POLICY_VERSION",
    "GATED_LATENT_STATE_POLICY_VERSION",
    "GATED_MUTATION_BUDGET_VERSION",
    "GATED_MUTATION_PROFILE",
    "GATED_NEUTRAL_ELIGIBILITY_VERSION",
    "GATED_NEUTRAL_TEMPLATE_VERSION",
    "GATED_NUMERIC_FIELDS",
    "GATED_NUMERIC_VERSION",
    "GATED_OPERATION_BUDGET_VERSION",
    "GATED_OPERATOR",
    "GATED_OUTPUT_FIELDS",
    "GATED_PARAMETER_BOUNDS",
    "GATED_PROFILE",
    "GATED_PROMOTER_POLICY_VERSION",
    "GATED_RECORD_FIELDS",
    "GATED_RECORD_KEY",
    "GATED_RECORD_SCHEMA_VERSION",
    "GATED_RESET_FIELDS",
    "GATED_RUNTIME_SCHEMA_VERSION",
    "GATED_STATE_WIDTH",
    "GATED_SUPPORTED_LANE",
    "GATED_UPDATE_FIELDS",
    "GatedRecordError",
    "GatedScalarStep",
    "clip_f32",
    "f32",
    "escap7_unsupported_lane_reasons",
    "hereditary_neutral_integrator_eligible",
    "nested_interval_choice",
    "neutral_gated_recurrent_record",
    "normalize_gated_module_params",
    "normalize_gated_recurrent_record",
    "stable_sigmoid_f32",
    "stable_tanh_f32",
    "step_gated_scalar",
]
