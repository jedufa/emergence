"""CPU-only Phase F v2 formal-oracle expectation adapter.

The adapter executes the public Phase E formal composition for all sixteen
fixture agents.  It converts only public source planes and formal raw outputs
into the exact comparison inventory frozen by the current Phase F matrix.  It
does not import CUDA owners, launch simulations, or claim checkpoint execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
import math
import re
from typing import Any, Final

import torch

from evolution.phase_f_v2.canonical import canonical_bytes, canonical_sha256
from evolution.phase_f_v2.matrix import validate_phase_f_v2_matrix
from perf.cuda_architecture_b_phase_e_formal_composition import (
    PHASE_E_FORMAL_COMPOSITION_SCHEMA_VERSION,
    PHASE_E_FORMAL_COMPOSITION_SCOPE,
    PhaseEFormalCompositionRaw,
    execute_phase_e_cpu_formal_composition,
)
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    PANEL_AGENT_COUNT,
    validate_phase_f_fixture_case_v2,
)
from perf.eiel_phase_f_forced_combination_gate_v2 import (
    EVIDENCE_ONLY_AUTHORITY,
    build_cpu_expectation_evidence_v2,
)
from perf.eiel_phase_f_forced_combination_source_v2 import (
    PhaseFV2SourceState,
    validate_phase_f_source_v2,
)
from perf.eiel_phase_f_execution_source_v2 import (
    validate_prepared_phase_f_execution_source_v2,
)
from perf.cuda_architecture_b_modular_state_abi import pack_modular_storage


ORACLE_PLAN_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_cpu_oracle_plan_v1"
ORACLE_EXPECTATION_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_f_v2_cpu_oracle_expectation_v1"
)
EXPECTED_AGENT_COUNT: Final[int] = 16
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")

_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "authority",
        "execution_authorized",
        "matrix_id",
        "matrix_sha256",
        "case_id",
        "case_sha256",
        "fixture_blueprint_sha256",
        "fixture_sha256",
        "profile_id",
        "source_seed",
        "owner_id",
        "lane_id",
        "checkpoint_kind",
        "exact_fields",
        "admitted_float_fields",
        "plan_sha256",
    }
)
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "authority",
        "execution_authorized",
        "matrix_id",
        "matrix_sha256",
        "case_id",
        "case_sha256",
        "fixture_blueprint_sha256",
        "fixture_sha256",
        "source_sha256",
        "execution_source_sha256",
        "profile_id",
        "source_seed",
        "owner_id",
        "lane_id",
        "expectation_kind",
        "checkpoint_kind",
        "fresh_owner_restored",
        "agent_count",
        "agent_ids",
        "agent_slots",
        "per_agent_raw_sha256",
        "fallback_count",
        "modular_rng_draw_count",
        "expectation",
        "expectation_sha256",
        "cpu_expectation_evidence",
        "receipt_sha256",
    }
)


class PhaseFV2OracleError(ValueError):
    """The source, formal raw evidence, or expectation contract is not exact."""


def _exact_mapping(
    value: Any, *, fields_: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PhaseFV2OracleError(f"{label} must be a string-keyed mapping")
    result = dict(value)
    missing = sorted(fields_ - set(result))
    extra = sorted(set(result) - fields_)
    if missing or extra:
        raise PhaseFV2OracleError(
            f"{label} fields are not exact; missing={missing!r} extra={extra!r}"
        )
    return result


def _sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PhaseFV2OracleError(f"{field} must be a lowercase SHA-256")
    return value


def _checked_matrix(value: Any) -> dict[str, Any]:
    try:
        return validate_phase_f_v2_matrix(value)
    except (TypeError, ValueError) as exc:
        raise PhaseFV2OracleError("matrix is not the current canonical matrix") from exc


def _case_for_fixture(
    matrix: Mapping[str, Any], fixture: Mapping[str, Any]
) -> dict[str, Any]:
    matches = [
        case for case in matrix["cases"] if case["case_id"] == fixture["case_id"]
    ]
    if len(matches) != 1:
        raise PhaseFV2OracleError("fixture case is absent from the current matrix")
    case = dict(matches[0])
    expected = {
        "case_sha256": case["case_sha256"],
        "matrix_sha256": matrix["matrix_sha256"],
        "fixture_blueprint_sha256": case["fixture_blueprint"][
            "fixture_blueprint_sha256"
        ],
        "profile_id": case["profile_id"],
    }
    for field, required in expected.items():
        if fixture.get(field) != required:
            raise PhaseFV2OracleError(f"fixture {field} is not current")
    return case


def _comparison_fields(
    matrix: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    contract = matrix["comparison_contract"]
    exact = tuple(contract["exact_fields"])
    admitted = tuple(contract["admitted_float_fields"])
    if (
        not exact
        or not admitted
        or len(set((*exact, *admitted))) != len((*exact, *admitted))
    ):
        raise PhaseFV2OracleError("matrix comparison inventory is malformed")
    if contract["categorical_rule"] != "exact" or contract["float_rule"] != {
        "absolute_tolerance": 2e-7,
        "relative_tolerance": 0.0,
    }:
        raise PhaseFV2OracleError("matrix comparison rules are not accepted")
    return exact, admitted


def build_phase_f_cpu_oracle_plan_v2(
    *, matrix: Any, fixture_case: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind one current fixture to the exact oracle inventory without execution."""

    checked_matrix = _checked_matrix(matrix)
    try:
        fixture = validate_phase_f_fixture_case_v2(fixture_case, matrix=checked_matrix)
    except Exception as exc:
        raise PhaseFV2OracleError("fixture case is not current and exact") from exc
    case = _case_for_fixture(checked_matrix, fixture)
    exact, admitted = _comparison_fields(checked_matrix)
    owner = checked_matrix["owner_contract"]
    plan: dict[str, Any] = {
        "schema_version": ORACLE_PLAN_SCHEMA_VERSION,
        "authority": EVIDENCE_ONLY_AUTHORITY,
        "execution_authorized": False,
        "matrix_id": checked_matrix["matrix_id"],
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "fixture_blueprint_sha256": case["fixture_blueprint"][
            "fixture_blueprint_sha256"
        ],
        "fixture_sha256": fixture["fixture_sha256"],
        "profile_id": case["profile_id"],
        "source_seed": case["fixture_blueprint"]["construction_seed"],
        "owner_id": owner["accelerated_owner"],
        "lane_id": owner["modular_tick_owner"],
        "checkpoint_kind": case["checkpoint_schedule"]["kind"],
        "exact_fields": list(exact),
        "admitted_float_fields": list(admitted),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return validate_phase_f_cpu_oracle_plan_v2(
        plan, matrix=checked_matrix, fixture_case=fixture
    )


def validate_phase_f_cpu_oracle_plan_v2(
    value: Any, *, matrix: Any, fixture_case: Mapping[str, Any]
) -> dict[str, Any]:
    checked_matrix = _checked_matrix(matrix)
    plan = _exact_mapping(value, fields_=_PLAN_FIELDS, label="oracle plan")
    if plan["schema_version"] != ORACLE_PLAN_SCHEMA_VERSION:
        raise PhaseFV2OracleError("oracle plan schema mismatch")
    if (
        plan["authority"] != EVIDENCE_ONLY_AUTHORITY
        or plan["execution_authorized"] is not False
    ):
        raise PhaseFV2OracleError("oracle plan overclaims authority")
    try:
        fixture = validate_phase_f_fixture_case_v2(fixture_case, matrix=checked_matrix)
    except Exception as exc:
        raise PhaseFV2OracleError("fixture case is not current and exact") from exc
    case = _case_for_fixture(checked_matrix, fixture)
    exact, admitted = _comparison_fields(checked_matrix)
    owner = checked_matrix["owner_contract"]
    required = {
        "matrix_id": checked_matrix["matrix_id"],
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "fixture_blueprint_sha256": case["fixture_blueprint"][
            "fixture_blueprint_sha256"
        ],
        "fixture_sha256": fixture["fixture_sha256"],
        "profile_id": case["profile_id"],
        "source_seed": case["fixture_blueprint"]["construction_seed"],
        "owner_id": owner["accelerated_owner"],
        "lane_id": owner["modular_tick_owner"],
        "checkpoint_kind": case["checkpoint_schedule"]["kind"],
        "exact_fields": list(exact),
        "admitted_float_fields": list(admitted),
    }
    for field, expected_value in required.items():
        if plan[field] != expected_value:
            raise PhaseFV2OracleError(f"oracle plan {field} binding mismatch")
    observed = _sha256(plan["plan_sha256"], field="plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256")
    if canonical_sha256(unsigned) != observed:
        raise PhaseFV2OracleError("oracle plan self digest mismatch")
    return plan


def _tensor_manifest(value: torch.Tensor) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise PhaseFV2OracleError("formal raw tensors must remain on CPU")
    tensor = value.detach().contiguous()
    if tensor.requires_grad:
        raise PhaseFV2OracleError("formal raw tensors must not require gradients")
    if tensor.is_floating_point() and not bool(torch.all(torch.isfinite(tensor))):
        raise PhaseFV2OracleError("formal raw tensor contains a nonfinite value")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": sha256(tensor.numpy().tobytes(order="C")).hexdigest(),
    }


def _canonical_raw_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_manifest(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_raw_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_raw_value(child) for child in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise PhaseFV2OracleError("formal raw value contains a nonfinite float")
        return value
    raise PhaseFV2OracleError(f"formal raw value has unsupported type: {type(value)!r}")


def _raw_sha256(raw: PhaseEFormalCompositionRaw) -> str:
    payload = {
        field.name: _canonical_raw_value(getattr(raw, field.name))
        for field in fields(PhaseEFormalCompositionRaw)
    }
    return canonical_sha256(payload)


def _tensor_json(value: torch.Tensor) -> Any:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise PhaseFV2OracleError("expectation tensors must remain on CPU")
    tensor = value.detach().contiguous()
    if tensor.is_floating_point() and not bool(torch.all(torch.isfinite(tensor))):
        raise PhaseFV2OracleError("expectation tensor contains a nonfinite value")
    result = tensor.tolist()
    canonical_bytes(result)
    return result


def _agent_values(raws: Sequence[PhaseEFormalCompositionRaw], name: str) -> list[Any]:
    return [_tensor_json(raw.modular_audit_tensors[name].squeeze(0)) for raw in raws]


def _plane_values(
    raws: Sequence[PhaseEFormalCompositionRaw], name: str, *, after: bool
) -> list[Any]:
    return [
        _tensor_json(
            (raw.planes_after if after else raw.planes_before)[name].squeeze(0)
        )
        for raw in raws
    ]


def _plane_manifests(
    raws: Sequence[PhaseEFormalCompositionRaw], *, after: bool
) -> list[dict[str, Any]]:
    return [
        {
            name: _tensor_manifest(tensor)
            for name, tensor in sorted(
                (raw.planes_after if after else raw.planes_before).items()
            )
        }
        for raw in raws
    ]


def _rng_equal(raw: PhaseEFormalCompositionRaw) -> bool:
    return (
        raw.rng_stream_indices_before == raw.rng_stream_indices_after
        and torch.equal(raw.torch_rng_state_before, raw.torch_rng_state_after)
        and canonical_bytes(_canonical_raw_value(raw.numpy_rng_state_before))
        == canonical_bytes(_canonical_raw_value(raw.numpy_rng_state_after))
        and raw.python_rng_state_before == raw.python_rng_state_after
    )


def _validate_raw(raw: Any) -> PhaseEFormalCompositionRaw:
    if not isinstance(raw, PhaseEFormalCompositionRaw):
        raise PhaseFV2OracleError("formal result has the wrong public raw type")
    if (
        raw.schema_version != PHASE_E_FORMAL_COMPOSITION_SCHEMA_VERSION
        or raw.scope != PHASE_E_FORMAL_COMPOSITION_SCOPE
    ):
        raise PhaseFV2OracleError("formal result schema or scope mismatch")
    required_audit = {
        "state_before",
        "state_after",
        "module_output",
        "development_effective_gain",
        "development_dependency_ready",
        "routed_observation",
        "previous_tick_connection_input",
        "action_residuals",
        "structural_cost_components",
        "structural_cost_total",
        "rng_draw_count",
    }
    missing = required_audit - set(raw.modular_audit_tensors)
    if missing:
        raise PhaseFV2OracleError(
            f"formal raw audit surfaces missing: {sorted(missing)!r}"
        )
    if (
        raw.modular_receipt.get("rng_draws") != 0
        or raw.modular_receipt.get("hidden_fallback_count") != 0
    ):
        raise PhaseFV2OracleError("formal raw used RNG or fallback")
    if int(raw.modular_audit_tensors["rng_draw_count"].item()) != 0:
        raise PhaseFV2OracleError("formal raw audit reports modular RNG")
    if not _rng_equal(raw):
        raise PhaseFV2OracleError("formal composition changed an RNG state")
    _raw_sha256(raw)
    return raw


def _execute_formal_input(
    formal_input: Any, *, modular_enabled: bool
) -> PhaseEFormalCompositionRaw:
    kwargs = formal_input.call_kwargs()
    if modular_enabled:
        return execute_phase_e_cpu_formal_composition(**kwargs)

    # Phase E enablement is a whole-invocation boundary.  For the disabled
    # legacy control, run the independent base policy against an empty modular
    # envelope to obtain exact zero-contribution audit shapes, then preserve
    # the real source bank unchanged as the before/after owner projection.
    source_planes = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in kwargs["planes"].items()
    }
    empty = pack_modular_storage(
        [None], agent_capacity=1, profile_id=formal_input.profile_id
    )
    kwargs["planes"] = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in empty.tensors.items()
    }
    raw = execute_phase_e_cpu_formal_composition(**kwargs)
    return replace(
        raw,
        planes_before={name: tensor.clone() for name, tensor in source_planes.items()},
        planes_after={name: tensor.clone() for name, tensor in source_planes.items()},
    )


def _expectation_mapping(
    *,
    matrix: Mapping[str, Any],
    source: PhaseFV2SourceState,
    execution_source: PhaseFV2SourceState,
    raws: Sequence[PhaseEFormalCompositionRaw],
    expectation_kind: str,
    modular_enabled: bool,
) -> dict[str, Any]:
    exact_fields, admitted_fields = _comparison_fields(matrix)
    if len(raws) != EXPECTED_AGENT_COUNT:
        raise PhaseFV2OracleError("formal raw count must equal 16 agents")
    actions = [int(raw.deterministic_actions.item()) for raw in raws]
    fallback_count = sum(
        int(raw.modular_receipt["hidden_fallback_count"]) for raw in raws
    )
    rng_draw_count = sum(int(raw.modular_receipt["rng_draws"]) for raw in raws)
    owner = matrix["owner_contract"]
    profile_id = execution_source.profile_id

    exact: dict[str, Any] = {
        "actions": actions,
        "checkpoint_projection": {
            "expectation_kind": expectation_kind,
            "source_sha256": source.source_sha256,
            "execution_source_sha256": execution_source.source_sha256,
            "planes_before": _plane_manifests(raws, after=False),
            "planes_after": _plane_manifests(raws, after=True),
        },
        "dependency_readiness": _agent_values(raws, "development_dependency_ready"),
        "developmental_stage": [int(raw.stage_indices.item()) for raw in raws],
        "dormancy": _plane_values(raws, "module_dormant", after=False),
        "fallback_count": fallback_count,
        "free_slot_maps": [
            {
                "module_free": [
                    index
                    for index, occupied in enumerate(
                        raw.planes_after["module_occupied"].squeeze(0).tolist()
                    )
                    if not occupied
                ],
                "connection_free": [
                    index
                    for index, occupied in enumerate(
                        raw.planes_after["connection_occupied"].squeeze(0).tolist()
                    )
                    if not occupied
                ],
            }
            for raw in raws
        ],
        "homology": _plane_values(raws, "module_homology_id_index", after=True),
        "identities": [
            {
                "agent_id": int(formal.agent_id),
                "agent_slot": int(formal.agent_slot),
                "agent_slot_token": int(formal.agent_slot_token),
                "module_instance_id_index": _tensor_json(
                    raw.planes_after["module_instance_id_index"].squeeze(0)
                ),
                "identity_origin_code": _tensor_json(
                    raw.planes_after["identity_origin_code"].squeeze(0)
                ),
            }
            for formal, raw in zip(
                execution_source.formal_oracle_inputs, raws, strict=True
            )
        ],
        "modular_rng_draw_count": rng_draw_count,
        "module_and_connection_presence": [
            {
                "module_occupied": _tensor_json(
                    raw.planes_after["module_occupied"].squeeze(0)
                ),
                "connection_occupied": _tensor_json(
                    raw.planes_after["connection_occupied"].squeeze(0)
                ),
                "module_state_present": _tensor_json(
                    raw.planes_after["module_state_present"].squeeze(0)
                ),
                "module_output_present": _tensor_json(
                    raw.planes_after["module_output_present"].squeeze(0)
                ),
            }
            for raw in raws
        ],
        "profile_and_owner_ids": {
            "profile_id": profile_id,
            "owner_id": owner["accelerated_owner"],
            "lane_id": owner["modular_tick_owner"],
            "case_id": source.case_id,
            "modular_enabled": modular_enabled,
        },
        "semantic_ids": [
            {
                name: _tensor_json(raw.planes_after[name].squeeze(0))
                for name in (
                    "input_semantic_code",
                    "input_semantic_id_index",
                    "output_semantic_code",
                    "output_semantic_id_index",
                    "output_action_index",
                )
            }
            for raw in raws
        ],
        "slot_maps": [
            {
                name: _tensor_json(raw.planes_after[name].squeeze(0))
                for name in (
                    "module_execution_order",
                    "dependency_topological_order",
                    "connection_execution_order",
                    "connection_src_module_slot",
                    "connection_dst_module_slot",
                    "connection_src_slot_index",
                    "connection_dst_slot_index",
                )
            }
            for raw in raws
        ],
        "validity_masks": [
            {
                "validity_before": _tensor_json(raw.validity_mask_before.squeeze(0)),
                "validity_after": _tensor_json(raw.validity_mask_after.squeeze(0)),
                "authority_before": _tensor_json(raw.authority_mask_before.squeeze(0)),
                "authority_after": _tensor_json(raw.authority_mask_after.squeeze(0)),
                "effective": _tensor_json(raw.effective_action_mask.squeeze(0)),
            }
            for raw in raws
        ],
    }
    admitted: dict[str, Any] = {
        "action_logits": [_tensor_json(raw.composed_logits.squeeze(0)) for raw in raws],
        "developmental_effective_gain": _agent_values(
            raws, "development_effective_gain"
        ),
        "module_output": _agent_values(raws, "module_output"),
        "module_state": _plane_values(raws, "module_state", after=True),
        "previous_output": _plane_values(raws, "module_previous_output", after=False),
        "previous_tick_connection_input": _agent_values(
            raws, "previous_tick_connection_input"
        ),
        "routed_action_residual": [
            _tensor_json(raw.modular_action_residuals.squeeze(0)) for raw in raws
        ],
        "routed_observation": _agent_values(raws, "routed_observation"),
        "state_before_and_after": [
            {
                "before": _tensor_json(
                    raw.modular_audit_tensors["state_before"].squeeze(0)
                ),
                "after": _tensor_json(
                    raw.modular_audit_tensors["state_after"].squeeze(0)
                ),
            }
            for raw in raws
        ],
        "structural_cost_components_and_total": [
            {
                "components": _tensor_json(
                    raw.modular_audit_tensors["structural_cost_components"].squeeze(0)
                ),
                "total": _tensor_json(
                    raw.modular_audit_tensors["structural_cost_total"].squeeze(0)
                ),
            }
            for raw in raws
        ],
    }
    if tuple(exact) != exact_fields or tuple(admitted) != admitted_fields:
        raise PhaseFV2OracleError("oracle expectation inventory drifted from matrix")
    expectation = {**exact, **admitted}
    _validate_expectation_values(
        expectation, exact_fields=exact_fields, admitted_fields=admitted_fields
    )
    return expectation


def _walk_floats(value: Any) -> list[float]:
    if type(value) is float:
        return [value]
    if type(value) is int:
        return [float(value)]
    if isinstance(value, list):
        return [number for child in value for number in _walk_floats(child)]
    if isinstance(value, Mapping):
        return [number for key in sorted(value) for number in _walk_floats(value[key])]
    raise PhaseFV2OracleError("admitted float surface contains a nonnumeric value")


def _validate_expectation_values(
    expectation: Any,
    *,
    exact_fields: Sequence[str],
    admitted_fields: Sequence[str],
) -> dict[str, Any]:
    required = frozenset((*exact_fields, *admitted_fields))
    values = _exact_mapping(expectation, fields_=required, label="expectation")
    for field in exact_fields:
        try:
            canonical_bytes(values[field])
        except (TypeError, ValueError) as exc:
            raise PhaseFV2OracleError(
                f"exact surface is not canonical: {field}"
            ) from exc
    for field in admitted_fields:
        numbers = _walk_floats(values[field])
        if not numbers or not all(math.isfinite(number) for number in numbers):
            raise PhaseFV2OracleError(f"float surface is empty or nonfinite: {field}")
    return values


def build_phase_f_cpu_expectation_v2(
    *,
    matrix: Any,
    fixture_case: Mapping[str, Any],
    source_state: PhaseFV2SourceState,
    expectation_kind: str = "uninterrupted",
    restored_source_state: PhaseFV2SourceState | None = None,
    fixture_registry: Mapping[str, Any] | None = None,
    source_preparation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute all sixteen CPU formal inputs and build one bound expectation."""

    checked_matrix = _checked_matrix(matrix)
    try:
        fixture = validate_phase_f_fixture_case_v2(fixture_case, matrix=checked_matrix)
    except Exception as exc:
        raise PhaseFV2OracleError("fixture case is not current and exact") from exc
    case = _case_for_fixture(checked_matrix, fixture)
    try:
        if source_preparation_receipt is None:
            source = validate_phase_f_source_v2(
                source_state,
                fixture_case=fixture,
                source_seed=int(case["fixture_blueprint"]["construction_seed"]),
            )
        else:
            if fixture_registry is None:
                raise PhaseFV2OracleError(
                    "prepared source requires the complete fixture registry"
                )
            source = validate_prepared_phase_f_execution_source_v2(
                source_state,
                matrix=checked_matrix,
                fixture_registry=fixture_registry,
                case_id=str(case["case_id"]),
                preparation_receipt=source_preparation_receipt,
            )
    except Exception as exc:
        raise PhaseFV2OracleError("ordinary source is not current and exact") from exc
    source_seed = int(source.source_seed)
    if expectation_kind == "uninterrupted":
        if restored_source_state is not None:
            raise PhaseFV2OracleError(
                "ordinary expectation cannot accept restored source"
            )
        execution_source = source
        fresh_owner_restored = False
    elif expectation_kind == "resumed":
        if case["checkpoint_schedule"]["kind"] == "none":
            raise PhaseFV2OracleError(
                "non-checkpoint case cannot request resumed oracle"
            )
        if restored_source_state is None or restored_source_state is source_state:
            raise PhaseFV2OracleError(
                "resumed oracle requires a distinct restored source"
            )
        try:
            if source_preparation_receipt is None:
                execution_source = validate_phase_f_source_v2(
                    restored_source_state,
                    fixture_case=fixture,
                    source_seed=source_seed,
                )
            else:
                assert fixture_registry is not None
                execution_source = validate_prepared_phase_f_execution_source_v2(
                    restored_source_state,
                    matrix=checked_matrix,
                    fixture_registry=fixture_registry,
                    case_id=str(case["case_id"]),
                    preparation_receipt=source_preparation_receipt,
                )
        except Exception as exc:
            raise PhaseFV2OracleError(
                "restored source is not current and exact"
            ) from exc
        fresh_owner_restored = True
    else:
        raise PhaseFV2OracleError("expectation_kind must be uninterrupted or resumed")

    modular_enabled = fixture["execution_recipe"]["modular_enabled"]
    if type(modular_enabled) is not bool:
        raise PhaseFV2OracleError("fixture modular enablement must be an exact bool")
    raw_results = tuple(
        _validate_raw(
            _execute_formal_input(formal_input, modular_enabled=modular_enabled)
        )
        for formal_input in execution_source.formal_oracle_inputs
    )
    if (
        len(raw_results) != PANEL_AGENT_COUNT
        or len(raw_results) != EXPECTED_AGENT_COUNT
    ):
        raise PhaseFV2OracleError(
            "formal execution did not cover all 16 fixture agents"
        )
    expectation = _expectation_mapping(
        matrix=checked_matrix,
        source=source,
        execution_source=execution_source,
        raws=raw_results,
        expectation_kind=expectation_kind,
        modular_enabled=modular_enabled,
    )
    raw_digests = [_raw_sha256(raw) for raw in raw_results]
    owner = checked_matrix["owner_contract"]
    evidence = build_cpu_expectation_evidence_v2(
        case=case,
        matrix_sha256=checked_matrix["matrix_sha256"],
        fixture_sha256=fixture["fixture_sha256"],
        source_sha256=execution_source.source_sha256,
        expectation=expectation,
        expectation_kind=expectation_kind,
        status="passed",
    )
    receipt: dict[str, Any] = {
        "schema_version": ORACLE_EXPECTATION_SCHEMA_VERSION,
        "authority": EVIDENCE_ONLY_AUTHORITY,
        "execution_authorized": False,
        "matrix_id": checked_matrix["matrix_id"],
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "fixture_blueprint_sha256": case["fixture_blueprint"][
            "fixture_blueprint_sha256"
        ],
        "fixture_sha256": fixture["fixture_sha256"],
        "source_sha256": source.source_sha256,
        "execution_source_sha256": execution_source.source_sha256,
        "profile_id": source.profile_id,
        "source_seed": source_seed,
        "owner_id": owner["accelerated_owner"],
        "lane_id": owner["modular_tick_owner"],
        "expectation_kind": expectation_kind,
        "checkpoint_kind": case["checkpoint_schedule"]["kind"],
        "fresh_owner_restored": fresh_owner_restored,
        "agent_count": EXPECTED_AGENT_COUNT,
        "agent_ids": [item.agent_id for item in execution_source.formal_oracle_inputs],
        "agent_slots": [
            item.agent_slot for item in execution_source.formal_oracle_inputs
        ],
        "per_agent_raw_sha256": raw_digests,
        "fallback_count": expectation["fallback_count"],
        "modular_rng_draw_count": expectation["modular_rng_draw_count"],
        "expectation": expectation,
        "expectation_sha256": canonical_sha256(expectation),
        "cpu_expectation_evidence": evidence,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return validate_phase_f_cpu_expectation_v2(
        receipt,
        matrix=checked_matrix,
        fixture_case=fixture,
        source_state=source,
        restored_source_state=(execution_source if fresh_owner_restored else None),
        fixture_registry=fixture_registry,
        source_preparation_receipt=source_preparation_receipt,
    )


def validate_phase_f_cpu_expectation_v2(
    value: Any,
    *,
    matrix: Any,
    fixture_case: Mapping[str, Any],
    source_state: PhaseFV2SourceState,
    restored_source_state: PhaseFV2SourceState | None = None,
    fixture_registry: Mapping[str, Any] | None = None,
    source_preparation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checked_matrix = _checked_matrix(matrix)
    try:
        fixture = validate_phase_f_fixture_case_v2(fixture_case, matrix=checked_matrix)
    except Exception as exc:
        raise PhaseFV2OracleError("fixture case is not current and exact") from exc
    case = _case_for_fixture(checked_matrix, fixture)
    try:
        if source_preparation_receipt is None:
            source = validate_phase_f_source_v2(
                source_state,
                fixture_case=fixture,
                source_seed=int(case["fixture_blueprint"]["construction_seed"]),
            )
        else:
            if fixture_registry is None:
                raise PhaseFV2OracleError(
                    "prepared source requires the complete fixture registry"
                )
            source = validate_prepared_phase_f_execution_source_v2(
                source_state,
                matrix=checked_matrix,
                fixture_registry=fixture_registry,
                case_id=str(case["case_id"]),
                preparation_receipt=source_preparation_receipt,
            )
    except Exception as exc:
        raise PhaseFV2OracleError("source is not current and exact") from exc
    source_seed = int(source.source_seed)
    receipt = _exact_mapping(
        value, fields_=_CONTRACT_FIELDS, label="CPU oracle expectation"
    )
    if receipt["schema_version"] != ORACLE_EXPECTATION_SCHEMA_VERSION:
        raise PhaseFV2OracleError("CPU oracle expectation schema mismatch")
    if (
        receipt["authority"] != EVIDENCE_ONLY_AUTHORITY
        or receipt["execution_authorized"] is not False
    ):
        raise PhaseFV2OracleError("CPU oracle expectation overclaims authority")
    kind = receipt["expectation_kind"]
    if kind not in {"uninterrupted", "resumed"}:
        raise PhaseFV2OracleError("CPU oracle expectation kind is unsupported")
    if kind == "resumed":
        if restored_source_state is None or restored_source_state is source_state:
            raise PhaseFV2OracleError(
                "resumed expectation lacks distinct restored source"
            )
        try:
            if source_preparation_receipt is None:
                execution_source = validate_phase_f_source_v2(
                    restored_source_state,
                    fixture_case=fixture,
                    source_seed=source_seed,
                )
            else:
                assert fixture_registry is not None
                execution_source = validate_prepared_phase_f_execution_source_v2(
                    restored_source_state,
                    matrix=checked_matrix,
                    fixture_registry=fixture_registry,
                    case_id=str(case["case_id"]),
                    preparation_receipt=source_preparation_receipt,
                )
        except Exception as exc:
            raise PhaseFV2OracleError(
                "restored source is not current and exact"
            ) from exc
    else:
        if restored_source_state is not None:
            raise PhaseFV2OracleError("ordinary expectation supplied restored source")
        execution_source = source
    owner = checked_matrix["owner_contract"]
    required = {
        "matrix_id": checked_matrix["matrix_id"],
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "fixture_blueprint_sha256": case["fixture_blueprint"][
            "fixture_blueprint_sha256"
        ],
        "fixture_sha256": fixture["fixture_sha256"],
        "source_sha256": source.source_sha256,
        "execution_source_sha256": execution_source.source_sha256,
        "profile_id": source.profile_id,
        "source_seed": source_seed,
        "owner_id": owner["accelerated_owner"],
        "lane_id": owner["modular_tick_owner"],
        "checkpoint_kind": case["checkpoint_schedule"]["kind"],
        "fresh_owner_restored": kind == "resumed",
        "agent_count": EXPECTED_AGENT_COUNT,
        "agent_ids": [item.agent_id for item in execution_source.formal_oracle_inputs],
        "agent_slots": [
            item.agent_slot for item in execution_source.formal_oracle_inputs
        ],
        "fallback_count": 0,
        "modular_rng_draw_count": 0,
    }
    for field, expected in required.items():
        if receipt[field] != expected:
            raise PhaseFV2OracleError(f"CPU oracle {field} binding mismatch")
    raw_digests = receipt["per_agent_raw_sha256"]
    if (
        type(raw_digests) is not list
        or len(raw_digests) != EXPECTED_AGENT_COUNT
        or len(set(raw_digests)) != EXPECTED_AGENT_COUNT
    ):
        raise PhaseFV2OracleError("per-agent raw digest inventory is not exact")
    for index, digest in enumerate(raw_digests):
        _sha256(digest, field=f"per_agent_raw_sha256[{index}]")
    exact, admitted = _comparison_fields(checked_matrix)
    expectation = _validate_expectation_values(
        receipt["expectation"], exact_fields=exact, admitted_fields=admitted
    )
    expectation_digest = canonical_sha256(expectation)
    if receipt["expectation_sha256"] != expectation_digest:
        raise PhaseFV2OracleError("expectation self digest mismatch")
    evidence = receipt["cpu_expectation_evidence"]
    expected_evidence = build_cpu_expectation_evidence_v2(
        case=case,
        matrix_sha256=checked_matrix["matrix_sha256"],
        fixture_sha256=fixture["fixture_sha256"],
        source_sha256=execution_source.source_sha256,
        expectation=expectation,
        expectation_kind=kind,
        status="passed",
    )
    if evidence != expected_evidence:
        raise PhaseFV2OracleError("CPU expectation evidence binding mismatch")
    observed = _sha256(receipt["receipt_sha256"], field="receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    if canonical_sha256(unsigned) != observed:
        raise PhaseFV2OracleError("CPU oracle receipt self digest mismatch")
    return deepcopy(receipt)


__all__ = [
    "EXPECTED_AGENT_COUNT",
    "ORACLE_EXPECTATION_SCHEMA_VERSION",
    "ORACLE_PLAN_SCHEMA_VERSION",
    "PhaseFV2OracleError",
    "build_phase_f_cpu_expectation_v2",
    "build_phase_f_cpu_oracle_plan_v2",
    "validate_phase_f_cpu_expectation_v2",
    "validate_phase_f_cpu_oracle_plan_v2",
]
