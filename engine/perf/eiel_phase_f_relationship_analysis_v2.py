"""Derived Phase F v2 repeat, causal, checkpoint, and terminal evidence.

This module is deliberately execution-free.  It consumes canonical CPU
expectations and persisted accelerated observations already bound to comparison
receipts.  Relationship outcomes are calculated from those artifacts; callers
cannot supply pass/fail statuses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final

from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.matrix import validate_phase_f_v2_matrix


CPU_BINDING_SCHEMA: Final[str] = "eiel_phase_f_v2_cpu_observation_binding_v1"
OBSERVATION_BINDING_SCHEMA: Final[str] = (
    "eiel_phase_f_v2_accelerated_observation_binding_v1"
)
REPEAT_SCHEMA: Final[str] = "eiel_phase_f_v2_repeat_relationship_v1"
CAUSAL_SCHEMA: Final[str] = "eiel_phase_f_v2_causal_relationship_v1"
CHECKPOINT_SCHEMA: Final[str] = "eiel_phase_f_v2_checkpoint_relationship_v1"
TERMINAL_SCHEMA: Final[str] = "eiel_phase_f_v2_relationship_terminal_ledger_v1"

DECISION_A: Final[str] = "PHASE-F-V2-A"
DECISION_B: Final[str] = "PHASE-F-V2-B"
DECISION_C: Final[str] = "PHASE-F-V2-C"
DECISION_D: Final[str] = "PHASE-F-V2-D"

EXPECTED_CPU_BINDINGS: Final[int] = 134
EXPECTED_ACCELERATED_BINDINGS: Final[int] = 268
EXPECTED_REPEAT_RELATIONSHIPS: Final[int] = 134
EXPECTED_CAUSAL_RELATIONSHIPS: Final[int] = 42
EXPECTED_CHECKPOINT_RELATIONSHIPS: Final[int] = 14

_ADMIN_CHECKPOINT_KEYS = frozenset(
    {"expectation_kind", "source_sha256", "execution_source_sha256"}
)

# The current matrix's broad surface names are interpreted through this frozen
# field inventory.  Dormancy is a module-trace property for causal analysis.
CAUSAL_SURFACE_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "named_module_trace": (
        "dependency_readiness",
        "developmental_effective_gain",
        "dormancy",
        "module_output",
        "previous_output",
        "previous_tick_connection_input",
        "routed_observation",
        "state_before_and_after",
    ),
    "named_routed_residual": (
        "actions",
        "action_logits",
        "routed_action_residual",
    ),
    "named_state_projection": (
        "module_state",
        "state_before_and_after",
    ),
}

_REQUIRED_SURFACE_BY_ACTION: Final[dict[str, tuple[str, ...]]] = {
    "remove_input_binding": ("named_module_trace",),
    "remove_output_binding": ("named_routed_residual",),
    "remove_downstream_output": ("named_routed_residual",),
    "reset_state": ("named_state_projection",),
    "reset_member_state": ("named_state_projection",),
    "remove_chain_connection": ("named_module_trace",),
    "remove_first_structural_connection": ("named_module_trace",),
    "remove_recurrent_connection": ("named_module_trace",),
    "block_dependency": ("named_module_trace",),
    "block_prerequisite": ("named_module_trace",),
    "enable_downstream_dormancy": ("named_module_trace",),
    "enable_member_dormancy": ("named_module_trace",),
    "enable_target_dormancy": ("named_module_trace",),
    "enable_dormancy_on_active_twin": ("named_module_trace",),
    "set_development_expression_zero": ("named_module_trace",),
}

# Exact effect slots implied by the frozen family variants.  This table avoids
# calling inapplicable no-op variants causal interventions.
_EFFECT_SLOT_OVERRIDES: Final[dict[tuple[str, str], tuple[int, ...]]] = {
    ("f02", "remove_input_binding"): (0, 1, 2, 3, 6, 7),
    ("f03", "remove_input_binding"): (0, 1, 2, 3, 4, 5, 6),
    ("f03", "block_dependency"): (6, 7),
    ("f04", "remove_downstream_output"): (1, 2, 3, 4, 5, 6, 7),
    ("f05", "block_prerequisite"): (3,),
    ("f07", "set_development_expression_zero"): tuple(range(8)),
}

_NON_EFFECT_SLOT_FIELDS: Final[tuple[str, ...]] = (
    "actions",
    "dependency_readiness",
    "developmental_stage",
    "dormancy",
    "free_slot_maps",
    "homology",
    "identities",
    "module_and_connection_presence",
    "semantic_ids",
    "slot_maps",
    "validity_masks",
    "action_logits",
    "developmental_effective_gain",
    "module_output",
    "module_state",
    "previous_output",
    "previous_tick_connection_input",
    "routed_action_residual",
    "routed_observation",
    "state_before_and_after",
    "structural_cost_components_and_total",
)


class PhaseFRelationshipAnalysisError(ValueError):
    """Relationship input is malformed, contradictory, or unbound."""


def _self_digest(value: Mapping[str, Any], *, field: str = "receipt_sha256") -> str:
    digest = value.get(field)
    if not isinstance(digest, str) or len(digest) != 64:
        raise PhaseFRelationshipAnalysisError(f"{field} is not a SHA-256")
    unsigned = dict(value)
    unsigned.pop(field)
    if canonical_sha256(unsigned) != digest:
        raise PhaseFRelationshipAnalysisError(f"{field} self digest mismatch")
    return digest


def _finish(value: dict[str, Any]) -> dict[str, Any]:
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _case_map(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): dict(case) for case in matrix["cases"]}


def _fixture_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = registry.get("cases")
    if not isinstance(cases, list):
        raise PhaseFRelationshipAnalysisError("fixture registry cases are missing")
    result = {str(case["case_id"]): dict(case) for case in cases}
    if len(result) != len(cases):
        raise PhaseFRelationshipAnalysisError("fixture registry has duplicate cases")
    return result


def expected_effect_slots_v2(case: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the exact variants on which the named intervention can operate."""

    family = str(case["family_id"])
    action = str(case["role_contract"]["action"])
    override = _EFFECT_SLOT_OVERRIDES.get((family, action))
    if override is not None:
        return override
    slots = case["role_contract"]["target_slots"]
    if not isinstance(slots, list) or any(type(slot) is not int for slot in slots):
        raise PhaseFRelationshipAnalysisError("case target_slots are malformed")
    return tuple(slots)


def bind_phase_f_cpu_expectation_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach the actual expectation and bind it to its oracle receipt."""

    if not isinstance(value, Mapping):
        raise PhaseFRelationshipAnalysisError("CPU expectation must be a mapping")
    expectation = deepcopy(value.get("expectation"))
    expectation_sha = canonical_sha256(expectation)
    if value.get("expectation_sha256") != expectation_sha:
        raise PhaseFRelationshipAnalysisError("CPU expectation digest mismatch")
    oracle_sha = _self_digest(value)
    evidence = value.get("cpu_expectation_evidence")
    if not isinstance(evidence, Mapping):
        raise PhaseFRelationshipAnalysisError("CPU expectation evidence is missing")
    if evidence.get("status") not in {"passed", "failed"}:
        raise PhaseFRelationshipAnalysisError("CPU evidence status is malformed")
    if evidence.get("expectation_sha256") != expectation_sha:
        raise PhaseFRelationshipAnalysisError("CPU evidence expectation mismatch")
    result = {
        "schema_version": CPU_BINDING_SCHEMA,
        "case_id": value.get("case_id"),
        "case_sha256": value.get("case_sha256"),
        "expectation_kind": value.get("expectation_kind"),
        "fixture_sha256": value.get("fixture_sha256"),
        "source_sha256": value.get("execution_source_sha256"),
        "expectation_sha256": expectation_sha,
        "oracle_receipt_sha256": oracle_sha,
        "outcome": evidence["status"],
        "expectation": expectation,
    }
    return _finish(result)


def bind_phase_f_accelerated_observation_v2(
    *, invocation_receipt: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind one persisted raw observation to its comparison receipt."""

    if not isinstance(invocation_receipt, Mapping) or not isinstance(
        observation, Mapping
    ):
        raise PhaseFRelationshipAnalysisError(
            "invocation receipt and observation must be mappings"
        )
    receipt_sha = _self_digest(invocation_receipt)
    observed = deepcopy(dict(observation))
    observed_sha = canonical_sha256(observed)
    if invocation_receipt.get("observed_sha256") != observed_sha:
        raise PhaseFRelationshipAnalysisError(
            "persisted observation contradicts invocation receipt"
        )
    raw_outcome = invocation_receipt.get("outcome")
    try:
        outcome = {
            "no_mismatch": "passed",
            "first_difference": "failed",
        }[raw_outcome]
    except (KeyError, TypeError) as exc:
        raise PhaseFRelationshipAnalysisError(
            "invocation comparison outcome is unsupported"
        ) from exc
    result = {
        "schema_version": OBSERVATION_BINDING_SCHEMA,
        "case_id": invocation_receipt.get("case_id"),
        "case_sha256": invocation_receipt.get("case_sha256"),
        "execution_id": invocation_receipt.get("execution_id"),
        "invocation_kind": invocation_receipt.get("invocation_kind"),
        "repeat_index": invocation_receipt.get("repeat_index"),
        "fixture_sha256": invocation_receipt.get("fixture_sha256"),
        "source_sha256": invocation_receipt.get("source_sha256"),
        "expected_sha256": invocation_receipt.get("expected_sha256"),
        "observed_sha256": observed_sha,
        "owner_id": invocation_receipt.get("owner_id"),
        "profile_id": invocation_receipt.get("profile_id"),
        "lane_id": invocation_receipt.get("lane_id"),
        "outcome": outcome,
        "invocation_receipt_sha256": receipt_sha,
        "observation": observed,
    }
    return _finish(result)


def _validate_binding(value: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema_version") != schema:
        raise PhaseFRelationshipAnalysisError(f"unsupported {schema} binding")
    result = deepcopy(dict(value))
    _self_digest(result)
    if schema == CPU_BINDING_SCHEMA:
        if canonical_sha256(result["expectation"]) != result["expectation_sha256"]:
            raise PhaseFRelationshipAnalysisError("CPU binding expectation changed")
    else:
        if canonical_sha256(result["observation"]) != result["observed_sha256"]:
            raise PhaseFRelationshipAnalysisError(
                "accelerated binding observation changed"
            )
    return result


def _cpu_inventory(
    values: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in values:
        row = _validate_binding(raw, schema=CPU_BINDING_SCHEMA)
        key = (str(row["case_id"]), str(row["expectation_kind"]))
        if key in result:
            raise PhaseFRelationshipAnalysisError("duplicate CPU binding")
        result[key] = row
    return result


def _accelerated_inventory(
    values: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in values:
        row = _validate_binding(raw, schema=OBSERVATION_BINDING_SCHEMA)
        key = (
            str(row["case_id"]),
            str(row["invocation_kind"]),
            int(row["repeat_index"]),
        )
        if key in result:
            raise PhaseFRelationshipAnalysisError(
                "duplicate accelerated observation binding"
            )
        result[key] = row
    return result


def build_phase_f_repeat_relationships_v2(
    *,
    matrix: Mapping[str, Any],
    accelerated_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive every available repeat-0/repeat-1 relationship exactly."""

    checked = validate_phase_f_v2_matrix(matrix)
    inventory = _accelerated_inventory(accelerated_bindings)
    relationships: list[dict[str, Any]] = []
    for case in checked["cases"]:
        case_id = str(case["case_id"])
        for kind in ("uninterrupted", "resumed"):
            if case["execution_accounting"][f"{kind}_repeats"] == 0:
                continue
            first = inventory.get((case_id, kind, 0))
            second = inventory.get((case_id, kind, 1))
            if first is None or second is None:
                continue
            checks = {
                "both_invocations_passed": (
                    first["outcome"] == second["outcome"] == "passed"
                ),
                "fixture_binding_exact": (
                    first["fixture_sha256"] == second["fixture_sha256"]
                ),
                "source_binding_exact": first["source_sha256"]
                == second["source_sha256"],
                "expectation_binding_exact": (
                    first["expected_sha256"] == second["expected_sha256"]
                ),
                "owner_profile_lane_exact": (
                    (
                        first["owner_id"],
                        first["profile_id"],
                        first["lane_id"],
                    )
                    == (
                        second["owner_id"],
                        second["profile_id"],
                        second["lane_id"],
                    )
                ),
                "observation_exact": (
                    first["observed_sha256"] == second["observed_sha256"]
                ),
            }
            result = {
                "schema_version": REPEAT_SCHEMA,
                "case_id": case_id,
                "case_sha256": case["case_sha256"],
                "invocation_kind": kind,
                "repeat_zero_receipt_sha256": first[
                    "invocation_receipt_sha256"
                ],
                "repeat_one_receipt_sha256": second[
                    "invocation_receipt_sha256"
                ],
                "repeat_zero_observed_sha256": first["observed_sha256"],
                "repeat_one_observed_sha256": second["observed_sha256"],
                "checks": checks,
                "outcome": "passed" if all(checks.values()) else "failed",
            }
            relationships.append(_finish(result))
    return relationships


def _normalize_scientific_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    checkpoint = result.get("checkpoint_projection")
    if isinstance(checkpoint, Mapping):
        result["checkpoint_projection"] = {
            key: deepcopy(item)
            for key, item in checkpoint.items()
            if key not in _ADMIN_CHECKPOINT_KEYS
        }
    owner = result.get("profile_and_owner_ids")
    if isinstance(owner, Mapping):
        result["profile_and_owner_ids"] = {
            key: deepcopy(item) for key, item in owner.items() if key != "case_id"
        }
    return result


def _slot_value(value: Any, slots: Sequence[int]) -> Any:
    if isinstance(value, list) and len(value) == 16:
        return [deepcopy(value[slot]) for slot in slots]
    return deepcopy(value)


def _surface_projection(
    value: Mapping[str, Any], surface: str, slots: Sequence[int]
) -> dict[str, Any]:
    fields = CAUSAL_SURFACE_FIELDS[surface]
    return {
        field: _slot_value(value.get(field), slots)
        for field in fields
    }


def _surface_changed(
    base: Mapping[str, Any],
    neutral: Mapping[str, Any],
    lesion: Mapping[str, Any],
    *,
    surface: str,
    slots: Sequence[int],
) -> bool:
    base_projection = _surface_projection(base, surface, slots)
    neutral_projection = _surface_projection(neutral, surface, slots)
    lesion_projection = _surface_projection(lesion, surface, slots)
    return (
        canonical_sha256(base_projection) == canonical_sha256(neutral_projection)
        and canonical_sha256(lesion_projection) != canonical_sha256(base_projection)
    )


def _non_effect_projection(
    value: Mapping[str, Any], slots: Sequence[int]
) -> dict[str, Any]:
    return {
        field: _slot_value(value.get(field), slots)
        for field in _NON_EFFECT_SLOT_FIELDS
    }


def _fixture_checks(
    *,
    base: Mapping[str, Any],
    neutral: Mapping[str, Any],
    lesion: Mapping[str, Any],
    effect_slots: Sequence[int],
    action: str,
) -> dict[str, bool]:
    base_agents = base.get("agents")
    neutral_agents = neutral.get("agents")
    lesion_agents = lesion.get("agents")
    if not all(
        isinstance(agents, list) and len(agents) == 16
        for agents in (base_agents, neutral_agents, lesion_agents)
    ):
        raise PhaseFRelationshipAnalysisError("fixture agent inventory is malformed")
    assert isinstance(base_agents, list)
    assert isinstance(neutral_agents, list)
    assert isinstance(lesion_agents, list)
    effect = set(effect_slots)
    unrelated_slots = [slot for slot in range(16) if slot not in effect]
    intact_equal = all(
        base_agents[slot]["intact_genome_sha256"]
        == neutral_agents[slot]["intact_genome_sha256"]
        == lesion_agents[slot]["intact_genome_sha256"]
        for slot in effect_slots
    )
    base_neutral_records = all(
        canonical_sha256(base_agents[slot]["record"])
        == canonical_sha256(neutral_agents[slot]["record"])
        for slot in range(16)
    )
    unrelated_equal = all(
        canonical_sha256(base_agents[slot]["record"])
        == canonical_sha256(neutral_agents[slot]["record"])
        == canonical_sha256(lesion_agents[slot]["record"])
        for slot in unrelated_slots
    )
    operations_real = True
    for slot in effect_slots:
        intervention = lesion_agents[slot].get("intervention")
        if not isinstance(intervention, Mapping) or intervention.get("action") != action:
            operations_real = False
            break
        operations = intervention.get("operations")
        if (
            not isinstance(operations, list)
            or not operations
            or any(not isinstance(operation, Mapping) for operation in operations)
            or all(
                operation.get("operation")
                in {"boundary_or_reporting_recipe", "prepared_boundary_recipe"}
                for operation in operations
            )
        ):
            operations_real = False
            break
    return {
        "base_neutral_fixture_records_exact": base_neutral_records,
        "effect_slot_intact_genomes_exact": intact_equal,
        "named_intervention_executed_on_every_effect_slot": operations_real,
        "unrelated_fixture_bytes_exact": unrelated_equal,
    }


def _repeat_map(
    values: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping) or raw.get("schema_version") != REPEAT_SCHEMA:
            raise PhaseFRelationshipAnalysisError("repeat relationship is malformed")
        row = deepcopy(dict(raw))
        _self_digest(row)
        key = (str(row["case_id"]), str(row["invocation_kind"]))
        if key in result:
            raise PhaseFRelationshipAnalysisError("duplicate repeat relationship")
        result[key] = row
    return result


def build_phase_f_causal_relationships_v2(
    *,
    matrix: Mapping[str, Any],
    fixture_registry: Mapping[str, Any],
    cpu_bindings: Sequence[Mapping[str, Any]],
    accelerated_bindings: Sequence[Mapping[str, Any]],
    repeat_relationships: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive all available intact/sham/lesion relationships."""

    checked = validate_phase_f_v2_matrix(matrix)
    cases = _case_map(checked)
    fixtures = _fixture_map(fixture_registry)
    cpu = _cpu_inventory(cpu_bindings)
    accelerated = _accelerated_inventory(accelerated_bindings)
    repeats = _repeat_map(repeat_relationships)
    results: list[dict[str, Any]] = []
    for case in checked["cases"]:
        if case["role_contract"]["claim_kind"] != "matched_causal_lesion":
            continue
        case_id = str(case["case_id"])
        base_id = str(case["relationship"]["base_case_id"])
        neutral_id = str(case["relationship"]["neutral_case_id"])
        required_ids = (base_id, neutral_id, case_id)
        if any(
            (candidate, "uninterrupted") not in cpu
            or (candidate, "uninterrupted", 0) not in accelerated
            or (candidate, "uninterrupted") not in repeats
            for candidate in required_ids
        ):
            continue
        if any(candidate not in fixtures for candidate in required_ids):
            continue
        cpu_rows = [cpu[(candidate, "uninterrupted")] for candidate in required_ids]
        gpu_rows = [
            accelerated[(candidate, "uninterrupted", 0)]
            for candidate in required_ids
        ]
        repeat_rows = [
            repeats[(candidate, "uninterrupted")] for candidate in required_ids
        ]
        effect_slots = expected_effect_slots_v2(case)
        action = str(case["role_contract"]["action"])
        required_surfaces = _REQUIRED_SURFACE_BY_ACTION.get(action)
        if required_surfaces is None:
            raise PhaseFRelationshipAnalysisError(
                f"causal action lacks a frozen effect rule: {action}"
            )
        base_cpu, neutral_cpu, lesion_cpu = [
            _normalize_scientific_observation(row["expectation"])
            for row in cpu_rows
        ]
        base_gpu, neutral_gpu, lesion_gpu = [
            _normalize_scientific_observation(row["observation"])
            for row in gpu_rows
        ]
        unrelated_slots = [
            slot for slot in range(16) if slot not in set(effect_slots)
        ]
        fixture_checks = _fixture_checks(
            base=fixtures[base_id],
            neutral=fixtures[neutral_id],
            lesion=fixtures[case_id],
            effect_slots=effect_slots,
            action=action,
        )
        checks: dict[str, bool] = {
            "matrix_target_slots_exact": tuple(
                case["role_contract"]["target_slots"]
            )
            == effect_slots,
            "fixture_receipt_bindings_exact": all(
                cpu_row["fixture_sha256"] == fixtures[candidate]["fixture_sha256"]
                and gpu_row["fixture_sha256"]
                == fixtures[candidate]["fixture_sha256"]
                for candidate, cpu_row, gpu_row in zip(
                    required_ids, cpu_rows, gpu_rows, strict=True
                )
            ),
            "all_cpu_expectations_passed": all(
                row["outcome"] == "passed" for row in cpu_rows
            ),
            "all_accelerated_invocations_passed": all(
                row["outcome"] == "passed" for row in gpu_rows
            ),
            "all_repeat_relationships_passed": all(
                row["outcome"] == "passed" for row in repeat_rows
            ),
            "profile_identity_exact": len(
                {row["profile_id"] for row in gpu_rows}
            )
            == 1
            and gpu_rows[0]["profile_id"] == case["profile_id"],
            "owner_lane_identity_exact": len(
                {(row["owner_id"], row["lane_id"]) for row in gpu_rows}
            )
            == 1,
            "rng_schedule_zero": all(
                row["expectation"].get("modular_rng_draw_count") == 0
                for row in cpu_rows
            )
            and all(
                row["observation"].get("modular_rng_draw_count") == 0
                for row in gpu_rows
            ),
            "cpu_base_neutral_scientific_projection_exact": (
                canonical_sha256(base_cpu) == canonical_sha256(neutral_cpu)
            ),
            "gpu_base_neutral_scientific_projection_exact": (
                canonical_sha256(base_gpu) == canonical_sha256(neutral_gpu)
            ),
            "cpu_unrelated_slots_exact": canonical_sha256(
                _non_effect_projection(base_cpu, unrelated_slots)
            )
            == canonical_sha256(
                _non_effect_projection(neutral_cpu, unrelated_slots)
            )
            == canonical_sha256(
                _non_effect_projection(lesion_cpu, unrelated_slots)
            ),
            "gpu_unrelated_slots_exact": canonical_sha256(
                _non_effect_projection(base_gpu, unrelated_slots)
            )
            == canonical_sha256(
                _non_effect_projection(neutral_gpu, unrelated_slots)
            )
            == canonical_sha256(
                _non_effect_projection(lesion_gpu, unrelated_slots)
            ),
        }
        checks.update(fixture_checks)
        for surface in required_surfaces:
            checks[f"cpu_required_effect:{surface}"] = _surface_changed(
                base_cpu,
                neutral_cpu,
                lesion_cpu,
                surface=surface,
                slots=effect_slots,
            )
            checks[f"gpu_required_effect:{surface}"] = _surface_changed(
                base_gpu,
                neutral_gpu,
                lesion_gpu,
                surface=surface,
                slots=effect_slots,
            )
        result = {
            "schema_version": CAUSAL_SCHEMA,
            "case_id": case_id,
            "case_sha256": case["case_sha256"],
            "base_case_id": base_id,
            "base_case_sha256": cases[base_id]["case_sha256"],
            "neutral_case_id": neutral_id,
            "neutral_case_sha256": cases[neutral_id]["case_sha256"],
            "family_id": case["family_id"],
            "profile_id": case["profile_id"],
            "action": action,
            "matrix_target_slots": list(case["role_contract"]["target_slots"]),
            "effect_slots": list(effect_slots),
            "required_changed_surfaces": list(required_surfaces),
            "preregistered_changed_surfaces": list(
                case["role_contract"]["expected_changed_surfaces"]
            ),
            "preregistered_unchanged_surfaces": list(
                case["role_contract"]["expected_unchanged_surfaces"]
            ),
            "cpu_binding_sha256": [
                row["receipt_sha256"] for row in cpu_rows
            ],
            "accelerated_binding_sha256": [
                row["receipt_sha256"] for row in gpu_rows
            ],
            "repeat_relationship_sha256": [
                row["receipt_sha256"] for row in repeat_rows
            ],
            "checks": checks,
            "outcome": "passed" if all(checks.values()) else "failed",
        }
        results.append(_finish(result))
    return results


def build_phase_f_checkpoint_relationships_v2(
    *,
    matrix: Mapping[str, Any],
    cpu_bindings: Sequence[Mapping[str, Any]],
    accelerated_bindings: Sequence[Mapping[str, Any]],
    repeat_relationships: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive all available uninterrupted/fresh-owner checkpoint pairs."""

    checked = validate_phase_f_v2_matrix(matrix)
    cpu = _cpu_inventory(cpu_bindings)
    accelerated = _accelerated_inventory(accelerated_bindings)
    repeats = _repeat_map(repeat_relationships)
    results: list[dict[str, Any]] = []
    for case in checked["cases"]:
        if case["checkpoint_schedule"]["kind"] == "none":
            continue
        case_id = str(case["case_id"])
        required = (
            (case_id, "uninterrupted"),
            (case_id, "resumed"),
        )
        if any(key not in cpu or key not in repeats for key in required):
            continue
        if any((*key, 0) not in accelerated for key in required):
            continue
        cpu_rows = [cpu[key] for key in required]
        gpu_rows = [accelerated[(*key, 0)] for key in required]
        repeat_rows = [repeats[key] for key in required]
        cpu_normalized = [
            _normalize_scientific_observation(row["expectation"])
            for row in cpu_rows
        ]
        gpu_normalized = [
            _normalize_scientific_observation(row["observation"])
            for row in gpu_rows
        ]
        checks = {
            "both_cpu_expectations_passed": all(
                row["outcome"] == "passed" for row in cpu_rows
            ),
            "both_accelerated_invocations_passed": all(
                row["outcome"] == "passed" for row in gpu_rows
            ),
            "both_repeat_relationships_passed": all(
                row["outcome"] == "passed" for row in repeat_rows
            ),
            "cpu_checkpoint_projection_exact": canonical_sha256(cpu_normalized[0])
            == canonical_sha256(cpu_normalized[1]),
            "gpu_checkpoint_projection_exact": canonical_sha256(gpu_normalized[0])
            == canonical_sha256(gpu_normalized[1]),
            "fixture_binding_exact": cpu_rows[0]["fixture_sha256"]
            == cpu_rows[1]["fixture_sha256"]
            == gpu_rows[0]["fixture_sha256"]
            == gpu_rows[1]["fixture_sha256"],
            "owner_profile_lane_exact": (
                (
                    gpu_rows[0]["owner_id"],
                    gpu_rows[0]["profile_id"],
                    gpu_rows[0]["lane_id"],
                )
                == (
                    gpu_rows[1]["owner_id"],
                    gpu_rows[1]["profile_id"],
                    gpu_rows[1]["lane_id"],
                )
            ),
            "rng_schedule_zero": all(
                row["expectation"].get("modular_rng_draw_count") == 0
                for row in cpu_rows
            )
            and all(
                row["observation"].get("modular_rng_draw_count") == 0
                for row in gpu_rows
            ),
        }
        result = {
            "schema_version": CHECKPOINT_SCHEMA,
            "case_id": case_id,
            "case_sha256": case["case_sha256"],
            "checkpoint_kind": case["checkpoint_schedule"]["kind"],
            "uninterrupted_cpu_binding_sha256": cpu_rows[0]["receipt_sha256"],
            "resumed_cpu_binding_sha256": cpu_rows[1]["receipt_sha256"],
            "uninterrupted_accelerated_binding_sha256": gpu_rows[0][
                "receipt_sha256"
            ],
            "resumed_accelerated_binding_sha256": gpu_rows[1]["receipt_sha256"],
            "repeat_relationship_sha256": [
                row["receipt_sha256"] for row in repeat_rows
            ],
            "checks": checks,
            "outcome": "passed" if all(checks.values()) else "failed",
        }
        results.append(_finish(result))
    return results


def select_phase_f_terminal_decision_v2(
    *,
    integrity_failed: bool,
    composition_failed: bool,
    incomplete: bool,
) -> str:
    """Apply the frozen C > B > D > A precedence."""

    if any(
        type(value) is not bool
        for value in (integrity_failed, composition_failed, incomplete)
    ):
        raise PhaseFRelationshipAnalysisError("decision flags must be exact booleans")
    if integrity_failed:
        return DECISION_C
    if composition_failed:
        return DECISION_B
    if incomplete:
        return DECISION_D
    return DECISION_A


def _relationship_inventory(
    values: Sequence[Mapping[str, Any]], *, schema: str
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for raw in values:
        if not isinstance(raw, Mapping) or raw.get("schema_version") != schema:
            raise PhaseFRelationshipAnalysisError(
                f"relationship does not use {schema}"
            )
        row = deepcopy(dict(raw))
        _self_digest(row)
        key: Any = str(row["case_id"])
        if schema == REPEAT_SCHEMA:
            key = (key, str(row["invocation_kind"]))
        if key in result:
            raise PhaseFRelationshipAnalysisError("duplicate relationship receipt")
        if row.get("outcome") not in {"passed", "failed"}:
            raise PhaseFRelationshipAnalysisError(
                "relationship outcome is not derived pass/fail"
            )
        checks = row.get("checks")
        if (
            not isinstance(checks, Mapping)
            or not checks
            or any(type(value) is not bool for value in checks.values())
            or row["outcome"]
            != ("passed" if all(checks.values()) else "failed")
        ):
            raise PhaseFRelationshipAnalysisError(
                "relationship outcome does not derive from exact checks"
            )
        result[key] = row
    return result


def aggregate_phase_f_relationship_terminal_v2(
    *,
    matrix: Mapping[str, Any],
    cpu_bindings: Sequence[Mapping[str, Any]],
    accelerated_bindings: Sequence[Mapping[str, Any]],
    repeat_relationships: Sequence[Mapping[str, Any]],
    causal_relationships: Sequence[Mapping[str, Any]],
    checkpoint_relationships: Sequence[Mapping[str, Any]],
    blocked: bool = False,
) -> dict[str, Any]:
    """Validate exact inventories and derive one terminal decision."""

    checked = validate_phase_f_v2_matrix(matrix)
    cpu = _cpu_inventory(cpu_bindings)
    accelerated = _accelerated_inventory(accelerated_bindings)
    repeats = _relationship_inventory(repeat_relationships, schema=REPEAT_SCHEMA)
    causal = _relationship_inventory(causal_relationships, schema=CAUSAL_SCHEMA)
    checkpoint = _relationship_inventory(
        checkpoint_relationships, schema=CHECKPOINT_SCHEMA
    )
    if type(blocked) is not bool:
        raise PhaseFRelationshipAnalysisError("blocked must be an exact boolean")

    expected_cpu = {
        (str(case["case_id"]), kind)
        for case in checked["cases"]
        for kind in ("uninterrupted", "resumed")
        if kind == "uninterrupted"
        or case["execution_accounting"]["resumed_repeats"] > 0
    }
    expected_accelerated = {
        (str(case["case_id"]), kind, repeat)
        for case in checked["cases"]
        for kind in ("uninterrupted", "resumed")
        for repeat in range(case["execution_accounting"][f"{kind}_repeats"])
    }
    expected_repeats = {(case_id, kind) for case_id, kind in expected_cpu}
    expected_causal = {
        str(case["case_id"])
        for case in checked["cases"]
        if case["role_contract"]["claim_kind"] == "matched_causal_lesion"
    }
    expected_checkpoint = {
        str(case["case_id"])
        for case in checked["cases"]
        if case["checkpoint_schedule"]["kind"] != "none"
    }
    for observed, expected, label in (
        (set(cpu), expected_cpu, "CPU"),
        (set(accelerated), expected_accelerated, "accelerated"),
        (set(repeats), expected_repeats, "repeat"),
        (set(causal), expected_causal, "causal"),
        (set(checkpoint), expected_checkpoint, "checkpoint"),
    ):
        if observed - expected:
            raise PhaseFRelationshipAnalysisError(f"invented {label} inventory slot")

    missing = {
        "cpu_bindings": len(expected_cpu - set(cpu)),
        "accelerated_bindings": len(expected_accelerated - set(accelerated)),
        "repeat_relationships": len(expected_repeats - set(repeats)),
        "causal_relationships": len(expected_causal - set(causal)),
        "checkpoint_relationships": len(expected_checkpoint - set(checkpoint)),
    }
    integrity_failed = (
        any(row["outcome"] != "passed" for row in cpu.values())
        or any(row["outcome"] != "passed" for row in accelerated.values())
        or any(row["outcome"] != "passed" for row in repeats.values())
        or any(row["outcome"] != "passed" for row in checkpoint.values())
    )
    composition_failed = any(
        row["outcome"] != "passed" for row in causal.values()
    )
    incomplete = blocked or any(missing.values())
    decision = select_phase_f_terminal_decision_v2(
        integrity_failed=integrity_failed,
        composition_failed=composition_failed,
        incomplete=incomplete,
    )
    result = {
        "schema_version": TERMINAL_SCHEMA,
        "matrix_id": checked["matrix_id"],
        "matrix_sha256": checked["matrix_sha256"],
        "decision": decision,
        "integrity_failed": integrity_failed,
        "composition_failed": composition_failed,
        "incomplete": incomplete,
        "blocked": blocked,
        "expected_counts": {
            "cpu_bindings": EXPECTED_CPU_BINDINGS,
            "accelerated_bindings": EXPECTED_ACCELERATED_BINDINGS,
            "repeat_relationships": EXPECTED_REPEAT_RELATIONSHIPS,
            "causal_relationships": EXPECTED_CAUSAL_RELATIONSHIPS,
            "checkpoint_relationships": EXPECTED_CHECKPOINT_RELATIONSHIPS,
        },
        "observed_counts": {
            "cpu_bindings": len(cpu),
            "accelerated_bindings": len(accelerated),
            "repeat_relationships": len(repeats),
            "causal_relationships": len(causal),
            "checkpoint_relationships": len(checkpoint),
        },
        "missing_counts": missing,
        "claims": {
            "bounded_composition_supported": decision == DECISION_A,
            "campaign_qualified": False,
            "speed_claimed": False,
        },
    }
    return _finish(result)


__all__ = [
    "CAUSAL_SCHEMA",
    "CAUSAL_SURFACE_FIELDS",
    "CHECKPOINT_SCHEMA",
    "CPU_BINDING_SCHEMA",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "EXPECTED_CAUSAL_RELATIONSHIPS",
    "EXPECTED_CHECKPOINT_RELATIONSHIPS",
    "EXPECTED_REPEAT_RELATIONSHIPS",
    "OBSERVATION_BINDING_SCHEMA",
    "PhaseFRelationshipAnalysisError",
    "REPEAT_SCHEMA",
    "aggregate_phase_f_relationship_terminal_v2",
    "bind_phase_f_accelerated_observation_v2",
    "bind_phase_f_cpu_expectation_v2",
    "build_phase_f_causal_relationships_v2",
    "build_phase_f_checkpoint_relationships_v2",
    "build_phase_f_repeat_relationships_v2",
    "expected_effect_slots_v2",
    "select_phase_f_terminal_decision_v2",
]
