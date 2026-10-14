from __future__ import annotations

"""Fail-closed validation for bounded Phase E formal composition.

The receipt does not claim a monolithic CPU modular-generation oracle.  It
binds an independently reviewed set of component and seam gates to one exact
source identity.  The resulting preparation remains caller-locked and does not
select the bounded Stage1 owner.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
import re
from typing import Any, Final, Mapping

import torch

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_RUNTIME_MUTABLE_FIELD_NAMES,
)
from perf.cuda_architecture_b_phase_e_formal_composition import (
    PHASE_E_FORMAL_COMPOSITION_SCHEMA_VERSION,
    PHASE_E_FORMAL_COMPOSITION_SCOPE,
    PhaseEFormalCompositionError,
    PhaseEFormalCompositionRaw,
    execute_phase_e_cpu_formal_composition,
)


FORMAL_GATE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_phase_e_formal_composition_receipt_v1"
)
FORMAL_GATE_SCOPE: Final[str] = (
    "phase_e_stage1_single_cell_no_birth_no_turnover_formal_composition_v1"
)
FORMAL_GATE_LANE_OWNER: Final[str] = "single_cell_native_generation_phase_e_v1"
FORMAL_GATE_NUMERIC_RULE_ID: Final[str] = (
    "phase_e_named_float_fields_atol_2e_7_v1"
)
FORMAL_GATE_ERROR: Final[str] = "EIEL_PHASE_E_FORMAL_GATE_REJECTED"
FORMAL_GATE_PLANE_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_modular_storage_overlay_v1"
)
FORMAL_GATE_PLANE_COUNT: Final[int] = 57
FORMAL_GATE_ACTION_VOCABULARY_VERSION: Final[str] = "eiel_action_vocabulary_v1"
FORMAL_GATE_ACTION_WIDTH: Final[int] = 7
FORMAL_GATE_OBSERVATION_SCHEMA_VERSION: Final[str] = (
    "radius2_local25_plus_energy_health_v1"
)
FORMAL_GATE_OBSERVATION_WIDTH: Final[int] = 27
FORMAL_GATE_POLICY_PROFILE: Final[str] = "fixed_mlp"
FORMAL_GATE_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "cxxbackbone10_checkpoint_projection_v1"
)
FORMAL_GATE_CATEGORICAL_RULE_ID: Final[str] = "phase_e_categorical_exact_v1"
FORMAL_GATE_RNG_RULE_ID: Final[str] = "phase_e_rng_zero_draw_indices_exact_v1"
FORMAL_GATE_COMPOSITION_ORDER_VERSION: Final[str] = (
    "phase_e_modular_tick_residual_mlp_action_v1"
)

SUPPORTED_LOGICAL_PROFILES: Final[tuple[str, ...]] = (
    "b2-12x32-v1",
    "b2-16x48-v1",
)
REQUIRED_PHASE_E_ARMS: Final[tuple[str, ...]] = (
    "disabled_legacy",
    "integrator",
    "gated_module",
    "input_routing",
    "output_routing",
    "development",
    "dependency_chain",
    "dormancy",
    "prune_and_reuse",
    "structural_cost",
    "checkpoint_resume",
    "capacity_migration",
)
REQUIRED_GATE_IDS: Final[tuple[str, ...]] = (
    "disabled_legacy_exact_v1",
    "storage_abi_exact_v1",
    "modular_equations_formal_parity_v1",
    "routing_formal_parity_v1",
    "development_dormancy_formal_parity_v1",
    "policy_composition_exact_v1",
    "generation_seam_exact_v1",
    "death_cleanup_exact_v1",
    "prune_reuse_exact_v1",
    "checkpoint_resume_exact_v1",
    "capacity_migration_exact_v1",
    "repeat_determinism_and_fallback_exact_v1",
    "numeric_rule_formally_accepted_v1",
)
NAMED_FORMAL_TOLERANCE_FIELDS: Final[tuple[str, ...]] = (
    "module_state",
    "module_previous_output",
    "state_before",
    "state_after",
    "module_output",
    "routed_observation",
    "previous_tick_connection_input",
    "development_effective_gain",
    "legacy_rest_residual",
    "routed_action_residual_before_rest_combine",
    "action_residuals",
    "structural_cost_components",
    "structural_cost_total",
    "action_logits",
)
EXACT_REQUIREMENTS: Final[tuple[str, ...]] = (
    "categorical_fields",
    "module_and_connection_presence",
    "module_state_presence",
    "module_output_presence",
    "development_latch_presence",
    "development_dependency_masks",
    "routing_masks",
    "dormancy_masks",
    "action_masks",
    "chosen_actions",
    "rng_stream_indices",
    "rng_snapshot_digest",
    "identity_maps",
    "module_instance_ids",
    "module_homology_ids",
    "slot_maps",
    "inactive_module_state_canonical",
    "inactive_module_output_canonical",
    "checkpoint_projection",
    "checkpoint_resume",
)
_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "scope",
        "lane_owner",
        "source_branch",
        "source_commit",
        "test_commit",
        "native_source_bundle_sha256",
        "gpu_artifact_present",
        "native_artifact_sha256",
        "capability_sha256",
        "fixture_source_sha256",
        "test_source_sha256",
        "plane_schema_version",
        "plane_count",
        "action_vocabulary_version",
        "action_width",
        "observation_schema_version",
        "observation_width",
        "policy_profile",
        "checkpoint_schema_version",
        "component_implementation_sha256",
        "validator_implementation_sha256",
        "raw_composition_sha256",
        "categorical_rule_id",
        "rng_rule_id",
        "composition_order_version",
        "logical_profiles",
        "arms",
        "excluded_arms",
        "gates",
        "hidden_fallback_count",
        "modular_rng_draw_count",
        "exact_requirements",
        "numeric_rule",
        "claims",
        "receipt_sha256",
    }
)
_ARM_FIELDS: Final[frozenset[str]] = frozenset(
    {"status", "comparison_basis", "evidence_sha256", "first_mismatch"}
)
_GATE_FIELDS: Final[frozenset[str]] = frozenset(
    {"status", "evidence_sha256", "first_mismatch"}
)
_NUMERIC_RULE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "rule_id",
        "review_status",
        "atol",
        "rtol",
        "named_float_tolerance_fields",
        "all_other_fields_exact",
    }
)
_CLAIMS: Final[dict[str, bool]] = {
    "stage1_gpu_ready_but_locked": True,
    "caller_locked": True,
    "formal_composition_receipt_accepted": True,
    "bounded_stage1_owner_qualified": False,
    "bounded_gpu_formal_composition_qualified": False,
    "gpu_parity_present": False,
    "whole_generation_cpu_oracle_present": False,
    "full_phase_e_qualified": False,
    "phase_f_authorized": False,
    "campaign_qualified": False,
    "speed_claimed": False,
}
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}")
_BRANCH_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
_RAW_TOLERANCE_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "base_logits",
        "modular_action_residuals",
        "composed_logits",
        "masked_logits",
    }
)
_RAW_TOLERANCE_PLANE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "module_state",
        "module_previous_output",
        "development_effective_gain",
        "structural_cost_components",
        "structural_cost_total",
    }
)
_RAW_TOLERANCE_AUDIT_FIELDS: Final[frozenset[str]] = frozenset(
    set(NAMED_FORMAL_TOLERANCE_FIELDS) - {"module_state", "module_previous_output"}
)
_POLICY_FORWARD_INPUT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "policy_observations",
        "w1",
        "b1",
        "w2",
        "b2",
        "w3",
        "b3",
        "attack_bias",
        "action_temperature",
    }
)


class PhaseEFormalGateError(RuntimeError):
    """The formal-composition receipt is absent, malformed, or contradicted."""


@dataclass(frozen=True)
class PhaseEFormalGateBindings:
    source_branch: str
    source_commit: str
    test_commit: str
    native_source_bundle_sha256: str
    gpu_artifact_present: bool
    native_artifact_sha256: str | None
    capability_sha256: str
    fixture_source_sha256: str
    test_source_sha256: str
    plane_schema_version: str
    plane_count: int
    action_vocabulary_version: str
    action_width: int
    observation_schema_version: str
    observation_width: int
    policy_profile: str
    checkpoint_schema_version: str
    component_implementation_sha256: Mapping[str, str]
    validator_implementation_sha256: str
    raw_composition_sha256: str
    categorical_rule_id: str
    rng_rule_id: str
    composition_order_version: str
    receipt_sha256: str


@dataclass(frozen=True)
class PhaseEFormalGateValidation:
    schema_version: str
    scope: str
    lane_owner: str
    source_branch: str
    source_commit: str
    test_commit: str
    native_source_bundle_sha256: str
    gpu_artifact_present: bool
    native_artifact_sha256: str | None
    capability_sha256: str
    fixture_source_sha256: str
    test_source_sha256: str
    validator_implementation_sha256: str
    raw_composition_sha256: str
    plane_schema_version: str
    plane_count: int
    action_vocabulary_version: str
    action_width: int
    observation_schema_version: str
    observation_width: int
    policy_profile: str
    checkpoint_schema_version: str
    component_implementation_sha256: dict[str, str]
    categorical_rule_id: str
    rng_rule_id: str
    composition_order_version: str
    receipt_sha256: str
    logical_profiles: tuple[str, ...]
    arms: tuple[str, ...]
    gate_ids: tuple[str, ...]
    numeric_rule_id: str
    stage1_gpu_ready_but_locked: bool = True
    caller_locked: bool = True
    formal_composition_receipt_accepted: bool = True
    bounded_stage1_owner_qualified: bool = False
    bounded_gpu_formal_composition_qualified: bool = False
    gpu_parity_present: bool = False
    whole_generation_cpu_oracle_present: bool = False
    full_phase_e_qualified: bool = False
    phase_f_authorized: bool = False
    campaign_qualified: bool = False
    speed_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fail(message: str) -> PhaseEFormalGateError:
    return PhaseEFormalGateError(f"{FORMAL_GATE_ERROR}: {message}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _fail("receipt must be canonical JSON data") from exc


def phase_e_formal_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Digest every receipt field except its self-authenticating digest."""

    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    payload = deepcopy(dict(receipt))
    payload.pop("receipt_sha256", None)
    return sha256(_canonical_bytes(payload)).hexdigest()


def _raw_canonical_value(value: Any, *, path: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise _fail(f"raw composition tensor must remain on CPU at {path}")
        if value.requires_grad or not value.is_contiguous():
            raise _fail(f"raw composition tensor contract is invalid at {path}")
        tensor = value.detach()
        raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data_sha256": sha256(raw_bytes).hexdigest(),
        }
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key in sorted(value):
            if type(key) is not str or not key:
                raise _fail(f"raw composition mapping key is invalid at {path}")
            out[key] = _raw_canonical_value(value[key], path=f"{path}.{key}")
        return {"kind": "mapping", "items": out}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _raw_canonical_value(item, path=f"{path}.{index}")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _raw_canonical_value(item, path=f"{path}.{index}")
                for index, item in enumerate(value)
            ],
        }
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise _fail(f"raw composition contains noncanonical data at {path}")


def phase_e_formal_composition_raw_sha256(
    observed: PhaseEFormalCompositionRaw,
) -> str:
    """Return the deterministic digest of externally supplied raw CPU evidence."""

    if not isinstance(observed, PhaseEFormalCompositionRaw):
        raise TypeError("observed must be PhaseEFormalCompositionRaw")
    payload = {
        field.name: _raw_canonical_value(
            getattr(observed, field.name),
            path=field.name,
        )
        for field in fields(PhaseEFormalCompositionRaw)
    }
    return sha256(_canonical_bytes(payload)).hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a lowercase SHA-256")
    return value


def _require_commit(value: Any, *, name: str) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise _fail(f"{name} must be a full lowercase Git commit")
    return value


def _require_branch(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or _BRANCH_RE.fullmatch(value) is None
        or ".." in value
        or value.endswith("/")
    ):
        raise _fail(f"{name} must be an exact repository branch name")
    return value


def _require_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise _fail(f"{name} must be a nonempty exact string")
    return value


def _require_component_hashes(value: Any, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise _fail(f"{name} must be a nonempty component hash mapping")
    out: dict[str, str] = {}
    for component, digest in value.items():
        component_name = _require_string(component, name=f"{name}.component")
        if component_name in out:
            raise _fail(f"{name} contains a duplicate component")
        out[component_name] = _require_sha256(
            digest,
            name=f"{name}.{component_name}",
        )
    return out


def _require_exact_fields(
    value: Any,
    *,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a mapping")
    actual = frozenset(str(field) for field in value)
    if actual != expected:
        raise _fail(
            f"{name} fields mismatch missing={sorted(expected-actual)!r} "
            f"extra={sorted(actual-expected)!r}"
        )
    return value


def _validate_bindings(
    receipt: Mapping[str, Any],
    bindings: PhaseEFormalGateBindings,
) -> None:
    if not isinstance(bindings, PhaseEFormalGateBindings):
        raise TypeError("bindings must be PhaseEFormalGateBindings")
    expected_branch = _require_branch(
        bindings.source_branch,
        name="bindings.source_branch",
    )
    actual_branch = _require_branch(
        receipt.get("source_branch"),
        name="receipt.source_branch",
    )
    if actual_branch != expected_branch:
        raise _fail("source branch contradicts the approved binding")
    expected_commit = _require_commit(
        bindings.source_commit,
        name="bindings.source_commit",
    )
    actual_commit = _require_commit(
        receipt.get("source_commit"),
        name="receipt.source_commit",
    )
    if actual_commit != expected_commit:
        raise _fail("source commit contradicts the approved binding")
    expected_test_commit = _require_commit(
        bindings.test_commit,
        name="bindings.test_commit",
    )
    actual_test_commit = _require_commit(
        receipt.get("test_commit"),
        name="receipt.test_commit",
    )
    if actual_test_commit != expected_test_commit:
        raise _fail("test commit contradicts the approved binding")
    for field in (
        "native_source_bundle_sha256",
        "capability_sha256",
        "fixture_source_sha256",
        "test_source_sha256",
    ):
        expected = _require_sha256(getattr(bindings, field), name=f"bindings.{field}")
        actual = _require_sha256(receipt.get(field), name=f"receipt.{field}")
        if actual != expected:
            raise _fail(f"{field} contradicts the approved binding")
    if (
        type(bindings.gpu_artifact_present) is not bool
        or bindings.gpu_artifact_present is not False
        or receipt.get("gpu_artifact_present") is not False
    ):
        raise _fail("GPU artifact must be explicitly absent in the locked CPU receipt")
    if (
        bindings.native_artifact_sha256 is not None
        or receipt.get("native_artifact_sha256") is not None
    ):
        raise _fail("locked CPU receipt must not bind a native artifact hash")
    validator_expected = _require_sha256(
        bindings.validator_implementation_sha256,
        name="bindings.validator_implementation_sha256",
    )
    validator_actual = _require_sha256(
        receipt.get("validator_implementation_sha256"),
        name="receipt.validator_implementation_sha256",
    )
    if validator_actual != validator_expected:
        raise _fail("validator implementation hash contradicts the approved binding")
    raw_expected = _require_sha256(
        bindings.raw_composition_sha256,
        name="bindings.raw_composition_sha256",
    )
    raw_actual = _require_sha256(
        receipt.get("raw_composition_sha256"),
        name="receipt.raw_composition_sha256",
    )
    if raw_actual != raw_expected:
        raise _fail("raw composition hash contradicts the approved binding")

    string_contracts = {
        "plane_schema_version": FORMAL_GATE_PLANE_SCHEMA_VERSION,
        "action_vocabulary_version": FORMAL_GATE_ACTION_VOCABULARY_VERSION,
        "observation_schema_version": FORMAL_GATE_OBSERVATION_SCHEMA_VERSION,
        "policy_profile": FORMAL_GATE_POLICY_PROFILE,
        "checkpoint_schema_version": FORMAL_GATE_CHECKPOINT_SCHEMA_VERSION,
        "categorical_rule_id": FORMAL_GATE_CATEGORICAL_RULE_ID,
        "rng_rule_id": FORMAL_GATE_RNG_RULE_ID,
        "composition_order_version": FORMAL_GATE_COMPOSITION_ORDER_VERSION,
    }
    for field, frozen_value in string_contracts.items():
        expected = _require_string(getattr(bindings, field), name=f"bindings.{field}")
        actual = _require_string(receipt.get(field), name=f"receipt.{field}")
        if expected != frozen_value or actual != expected:
            raise _fail(f"{field} contradicts the frozen approved binding")

    integer_contracts = {
        "plane_count": FORMAL_GATE_PLANE_COUNT,
        "action_width": FORMAL_GATE_ACTION_WIDTH,
        "observation_width": FORMAL_GATE_OBSERVATION_WIDTH,
    }
    for field, frozen_value in integer_contracts.items():
        expected = getattr(bindings, field)
        actual = receipt.get(field)
        if (
            type(expected) is not int
            or type(actual) is not int
            or expected != frozen_value
            or actual != expected
        ):
            raise _fail(f"{field} contradicts the frozen approved binding")

    expected_components = _require_component_hashes(
        bindings.component_implementation_sha256,
        name="bindings.component_implementation_sha256",
    )
    actual_components = _require_component_hashes(
        receipt.get("component_implementation_sha256"),
        name="receipt.component_implementation_sha256",
    )
    if actual_components != expected_components:
        raise _fail("component implementation hashes contradict the approved binding")


def _validate_arms(value: Any) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(
        REQUIRED_PHASE_E_ARMS
    ):
        raise _fail("receipt must contain the exact 12-arm Phase E matrix")
    for name in REQUIRED_PHASE_E_ARMS:
        arm = _require_exact_fields(
            value[name],
            expected=_ARM_FIELDS,
            name=f"arms.{name}",
        )
        if arm.get("status") != "cpu_reference_passed_gpu_pending":
            raise _fail(f"Phase E arm must remain locked pending GPU parity: {name}")
        if (
            arm.get("comparison_basis")
            != "cpu_reference_component_and_static_preparation"
        ):
            raise _fail(f"Phase E arm comparison basis is invalid: {name}")
        _require_sha256(arm.get("evidence_sha256"), name=f"arms.{name}.evidence_sha256")
        if arm.get("first_mismatch") is not None:
            raise _fail(f"Phase E arm reports a first mismatch: {name}")


def _validate_gates(value: Any) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(
        REQUIRED_GATE_IDS
    ):
        raise _fail("receipt gate IDs are incomplete or expanded")
    for gate_id in REQUIRED_GATE_IDS:
        gate = _require_exact_fields(
            value[gate_id],
            expected=_GATE_FIELDS,
            name=f"gates.{gate_id}",
        )
        if gate.get("status") != "cpu_reference_or_static_ready_gpu_pending":
            raise _fail(f"required formal gate must remain GPU-pending: {gate_id}")
        _require_sha256(
            gate.get("evidence_sha256"),
            name=f"gates.{gate_id}.evidence_sha256",
        )
        if gate.get("first_mismatch") is not None:
            raise _fail(f"required formal gate reports a mismatch: {gate_id}")


def _validate_exact_requirements(value: Any) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != frozenset(
        EXACT_REQUIREMENTS
    ):
        raise _fail("exact categorical/RNG/identity/slot/checkpoint requirements changed")
    weakened = [name for name in EXACT_REQUIREMENTS if value.get(name) is not True]
    if weakened:
        raise _fail(f"exact requirements were weakened: {weakened!r}")


def _validate_numeric_rule(value: Any) -> None:
    rule = _require_exact_fields(
        value,
        expected=_NUMERIC_RULE_FIELDS,
        name="numeric_rule",
    )
    if rule.get("rule_id") != FORMAL_GATE_NUMERIC_RULE_ID:
        raise _fail("numeric rule ID is not the formally accepted rule")
    if rule.get("review_status") != "formally_accepted":
        raise _fail("numeric rule is not formally accepted")
    atol = rule.get("atol")
    rtol = rule.get("rtol")
    if type(atol) is not float or atol != 2.0e-7:
        raise _fail("numeric rule must use atol=2e-7")
    if type(rtol) is not float or rtol != 0.0:
        raise _fail("numeric rule must use rtol=0")
    if tuple(rule.get("named_float_tolerance_fields") or ()) != (
        NAMED_FORMAL_TOLERANCE_FIELDS
    ):
        raise _fail("numeric tolerance is not limited to the accepted float fields")
    if rule.get("all_other_fields_exact") is not True:
        raise _fail(
            "presence/mask/action/categorical/RNG/identity/slot/inactive/checkpoint "
            "fields must remain exact"
        )


def _raw_path_uses_formal_tolerance(path: tuple[str, ...]) -> bool:
    if len(path) == 1 and path[0] in _RAW_TOLERANCE_TOP_LEVEL_FIELDS:
        return True
    if len(path) == 2 and path[0] == "planes_after":
        return path[1] in _RAW_TOLERANCE_PLANE_FIELDS
    if len(path) == 2 and path[0] == "modular_audit_tensors":
        return path[1] in _RAW_TOLERANCE_AUDIT_FIELDS
    return False


def _compare_raw_value(expected: Any, observed: Any, *, path: tuple[str, ...]) -> None:
    name = ".".join(path)
    if isinstance(expected, torch.Tensor):
        if not isinstance(observed, torch.Tensor):
            raise _fail(f"formal CPU oracle type mismatch at {name}")
        if (
            expected.dtype != observed.dtype
            or expected.device.type != "cpu"
            or observed.device.type != "cpu"
            or tuple(expected.shape) != tuple(observed.shape)
            or expected.is_contiguous() != observed.is_contiguous()
            or expected.requires_grad != observed.requires_grad
        ):
            raise _fail(f"formal CPU oracle tensor contract mismatch at {name}")
        if _raw_path_uses_formal_tolerance(path):
            if not bool(torch.all(torch.isfinite(expected))) or not bool(
                torch.all(torch.isfinite(observed))
            ):
                raise _fail(f"formal CPU oracle nonfinite value at {name}")
            if not torch.allclose(expected, observed, atol=2.0e-7, rtol=0.0):
                raise _fail(f"formal CPU oracle float mismatch at {name}")
        elif not torch.equal(expected, observed):
            raise _fail(f"formal CPU oracle exact mismatch at {name}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or frozenset(expected) != frozenset(
            observed
        ):
            raise _fail(f"formal CPU oracle mapping mismatch at {name}")
        for key in expected:
            if type(key) is not str:
                raise _fail(f"formal CPU oracle mapping key is not a string at {name}")
            _compare_raw_value(
                expected[key],
                observed[key],
                path=(*path, key),
            )
        return
    if isinstance(expected, (tuple, list)):
        if type(observed) is not type(expected) or len(expected) != len(observed):
            raise _fail(f"formal CPU oracle sequence mismatch at {name}")
        for index, (expected_value, observed_value) in enumerate(
            zip(expected, observed, strict=True)
        ):
            _compare_raw_value(
                expected_value,
                observed_value,
                path=(*path, str(index)),
            )
        return
    if type(observed) is not type(expected) or observed != expected:
        raise _fail(f"formal CPU oracle exact mismatch at {name}")


def _validate_observed_cpu_composition(observed: Any) -> None:
    if not isinstance(observed, PhaseEFormalCompositionRaw):
        raise _fail("observed_composition must be raw Phase E CPU composition data")
    if observed.schema_version != PHASE_E_FORMAL_COMPOSITION_SCHEMA_VERSION:
        raise _fail("observed CPU composition schema mismatch")
    if observed.scope != PHASE_E_FORMAL_COMPOSITION_SCOPE:
        raise _fail("observed CPU composition scope mismatch")
    if observed.runtime_mutable_plane_fields != MODULAR_RUNTIME_MUTABLE_FIELD_NAMES:
        raise _fail("observed CPU composition mutable-plane ownership changed")
    if not isinstance(observed.policy_forward_inputs, Mapping) or frozenset(
        observed.policy_forward_inputs
    ) != _POLICY_FORWARD_INPUT_FIELDS:
        raise _fail("observed CPU composition policy inputs changed")

    try:
        expected = execute_phase_e_cpu_formal_composition(
            planes=observed.planes_before,
            modular_observations=observed.modular_observations,
            fatigue=observed.fatigue,
            stage_indices=observed.stage_indices,
            policy_observations=observed.policy_forward_inputs[
                "policy_observations"
            ],
            w1=observed.policy_forward_inputs["w1"],
            b1=observed.policy_forward_inputs["b1"],
            w2=observed.policy_forward_inputs["w2"],
            b2=observed.policy_forward_inputs["b2"],
            w3=observed.policy_forward_inputs["w3"],
            b3=observed.policy_forward_inputs["b3"],
            attack_bias=observed.policy_forward_inputs["attack_bias"],
            action_temperature=observed.policy_forward_inputs[
                "action_temperature"
            ],
            validity_mask=observed.validity_mask_before,
            authority_mask=observed.authority_mask_before,
            rng_stream_indices=observed.rng_stream_indices_before,
        )
    except PhaseEFormalCompositionError as exc:
        raise _fail(f"formal CPU oracle recomputation rejected its inputs: {exc}") from exc

    if observed.plane_field_names != expected.plane_field_names:
        raise _fail("observed CPU composition plane field list changed")

    if not torch.equal(observed.validity_mask_before, observed.validity_mask_after):
        raise _fail("formal CPU composition changed the validity mask")
    if not torch.equal(observed.authority_mask_before, observed.authority_mask_after):
        raise _fail("formal CPU composition changed the authority mask")
    if observed.rng_stream_indices_before != observed.rng_stream_indices_after:
        raise _fail("formal CPU composition changed an RNG stream index")
    mutable_fields = frozenset(MODULAR_RUNTIME_MUTABLE_FIELD_NAMES)
    for plane_name in observed.plane_field_names:
        if plane_name not in mutable_fields and not torch.equal(
            observed.planes_before[plane_name], observed.planes_after[plane_name]
        ):
            raise _fail(f"formal CPU composition changed unowned plane {plane_name}")

    for field in fields(PhaseEFormalCompositionRaw):
        _compare_raw_value(
            getattr(expected, field.name),
            getattr(observed, field.name),
            path=(field.name,),
        )


def validate_phase_e_formal_composition_receipt(
    receipt: Mapping[str, Any],
    *,
    bindings: PhaseEFormalGateBindings,
    observed_composition: PhaseEFormalCompositionRaw,
) -> PhaseEFormalGateValidation:
    """Validate one locked bounded formal-composition preparation receipt."""

    observed_raw_digest = phase_e_formal_composition_raw_sha256(
        observed_composition
    )
    approved_raw_digest = _require_sha256(
        bindings.raw_composition_sha256,
        name="bindings.raw_composition_sha256",
    )
    if observed_raw_digest != approved_raw_digest:
        raise _fail("supplied raw CPU composition is not the approved evidence")
    _validate_observed_cpu_composition(observed_composition)

    candidate = _require_exact_fields(
        receipt,
        expected=_TOP_LEVEL_FIELDS,
        name="receipt",
    )
    if candidate.get("schema_version") != FORMAL_GATE_SCHEMA_VERSION:
        raise _fail("receipt schema mismatch")
    if candidate.get("scope") != FORMAL_GATE_SCOPE:
        raise _fail("receipt scope mismatch")
    if candidate.get("lane_owner") != FORMAL_GATE_LANE_OWNER:
        raise _fail("receipt lane owner mismatch")
    _validate_bindings(candidate, bindings)

    observed_digest = _require_sha256(
        candidate.get("receipt_sha256"),
        name="receipt.receipt_sha256",
    )
    computed_digest = phase_e_formal_receipt_sha256(candidate)
    approved_digest = _require_sha256(
        bindings.receipt_sha256,
        name="bindings.receipt_sha256",
    )
    if observed_digest != computed_digest:
        raise _fail("canonical receipt digest mismatch")
    if observed_digest != approved_digest:
        raise _fail("canonical receipt digest is not the approved digest")

    if tuple(candidate.get("logical_profiles") or ()) != SUPPORTED_LOGICAL_PROFILES:
        raise _fail("receipt must bind both exact logical B2 profiles")
    if candidate.get("excluded_arms") != ["reproduction"]:
        raise _fail("the bounded receipt must exclude only the reproduction arm")
    _validate_arms(candidate.get("arms"))
    _validate_gates(candidate.get("gates"))
    if type(candidate.get("hidden_fallback_count")) is not int or candidate.get(
        "hidden_fallback_count"
    ) != 0:
        raise _fail("hidden_fallback_count must be exact integer zero")
    if type(candidate.get("modular_rng_draw_count")) is not int or candidate.get(
        "modular_rng_draw_count"
    ) != 0:
        raise _fail("modular_rng_draw_count must be exact integer zero")
    _validate_exact_requirements(candidate.get("exact_requirements"))
    _validate_numeric_rule(candidate.get("numeric_rule"))
    if candidate.get("claims") != _CLAIMS:
        raise _fail("bounded formal-composition claims are missing or overbroad")

    return PhaseEFormalGateValidation(
        schema_version=FORMAL_GATE_SCHEMA_VERSION,
        scope=FORMAL_GATE_SCOPE,
        lane_owner=FORMAL_GATE_LANE_OWNER,
        source_branch=str(candidate["source_branch"]),
        source_commit=str(candidate["source_commit"]),
        test_commit=str(candidate["test_commit"]),
        native_source_bundle_sha256=str(
            candidate["native_source_bundle_sha256"]
        ),
        gpu_artifact_present=False,
        native_artifact_sha256=None,
        capability_sha256=str(candidate["capability_sha256"]),
        fixture_source_sha256=str(candidate["fixture_source_sha256"]),
        test_source_sha256=str(candidate["test_source_sha256"]),
        validator_implementation_sha256=str(
            candidate["validator_implementation_sha256"]
        ),
        raw_composition_sha256=str(candidate["raw_composition_sha256"]),
        plane_schema_version=FORMAL_GATE_PLANE_SCHEMA_VERSION,
        plane_count=FORMAL_GATE_PLANE_COUNT,
        action_vocabulary_version=FORMAL_GATE_ACTION_VOCABULARY_VERSION,
        action_width=FORMAL_GATE_ACTION_WIDTH,
        observation_schema_version=FORMAL_GATE_OBSERVATION_SCHEMA_VERSION,
        observation_width=FORMAL_GATE_OBSERVATION_WIDTH,
        policy_profile=FORMAL_GATE_POLICY_PROFILE,
        checkpoint_schema_version=FORMAL_GATE_CHECKPOINT_SCHEMA_VERSION,
        component_implementation_sha256=dict(
            candidate["component_implementation_sha256"]
        ),
        categorical_rule_id=FORMAL_GATE_CATEGORICAL_RULE_ID,
        rng_rule_id=FORMAL_GATE_RNG_RULE_ID,
        composition_order_version=FORMAL_GATE_COMPOSITION_ORDER_VERSION,
        receipt_sha256=observed_digest,
        logical_profiles=SUPPORTED_LOGICAL_PROFILES,
        arms=REQUIRED_PHASE_E_ARMS,
        gate_ids=REQUIRED_GATE_IDS,
        numeric_rule_id=FORMAL_GATE_NUMERIC_RULE_ID,
    )


__all__ = [
    "EXACT_REQUIREMENTS",
    "FORMAL_GATE_ACTION_VOCABULARY_VERSION",
    "FORMAL_GATE_ACTION_WIDTH",
    "FORMAL_GATE_CATEGORICAL_RULE_ID",
    "FORMAL_GATE_CHECKPOINT_SCHEMA_VERSION",
    "FORMAL_GATE_COMPOSITION_ORDER_VERSION",
    "FORMAL_GATE_ERROR",
    "FORMAL_GATE_LANE_OWNER",
    "FORMAL_GATE_NUMERIC_RULE_ID",
    "FORMAL_GATE_OBSERVATION_SCHEMA_VERSION",
    "FORMAL_GATE_OBSERVATION_WIDTH",
    "FORMAL_GATE_PLANE_COUNT",
    "FORMAL_GATE_PLANE_SCHEMA_VERSION",
    "FORMAL_GATE_POLICY_PROFILE",
    "FORMAL_GATE_RNG_RULE_ID",
    "FORMAL_GATE_SCHEMA_VERSION",
    "FORMAL_GATE_SCOPE",
    "NAMED_FORMAL_TOLERANCE_FIELDS",
    "REQUIRED_GATE_IDS",
    "REQUIRED_PHASE_E_ARMS",
    "SUPPORTED_LOGICAL_PROFILES",
    "PhaseEFormalGateBindings",
    "PhaseEFormalGateError",
    "PhaseEFormalGateValidation",
    "phase_e_formal_receipt_sha256",
    "phase_e_formal_composition_raw_sha256",
    "validate_phase_e_formal_composition_receipt",
]
