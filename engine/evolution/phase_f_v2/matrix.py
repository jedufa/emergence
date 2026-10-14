"""Canonical planning-only matrix for the Phase F v2 combination battery.

This module deliberately owns no simulation, CUDA, checkpoint, or result
authority.  It freezes the smallest honest Phase F v2 matrix: twelve families,
two logical profiles, and five matched roles per family/profile block.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence

from .canonical import (
    PhaseFCanonicalJsonError,
    canonical_bytes,
    canonical_sha256 as _canonical_sha256,
)


SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_forced_combination_matrix_v2"
MATRIX_ID: Final[str] = "phase-f-v2-forced-combination-120-v2"
AUTHORIZED_BASE_COMMIT: Final[str] = "f65725b23eac636ea6afbf1743ab09dca418a186"
MATRIX_STATUS: Final[str] = "planning_only_preregistered"
CASE_COUNT: Final[int] = 120
CHECKPOINT_CASE_COUNT: Final[int] = 14
EXPECTED_CPU_ORACLE_EVALUATIONS: Final[int] = 134
EXPECTED_ACCELERATED_INVOCATIONS: Final[int] = 268
EXPECTED_TOTAL_REQUIRED_EVALUATIONS: Final[int] = 402
UNINTERRUPTED_REPEATS: Final[int] = 2
RESUMED_REPEATS: Final[int] = 2

ACCELERATED_OWNER: Final[str] = "single_cell_native_generation_phase_e_v1"
MODULAR_TICK_OWNER: Final[str] = "native_aten_device_resident_modular_tick_v1"
OWNER_CONTRACT_ID: Final[str] = "phase-f-v2-qualified-modular-owner-v1"
COMPARISON_CONTRACT_ID: Final[str] = "phase-e-named-parity-rule-v1"
ORACLE_RECIPE_ID: Final[str] = "phase-e-formal-composition-oracle-v1"

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_FORBIDDEN_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {"actual", "observed", "passed", "result", "results", "verdict"}
)

_PROFILES: Final[tuple[dict[str, Any], ...]] = (
    {
        "profile_code": "b12",
        "profile_id": "b2-12x32-v1",
        "profile_index": 0,
        "logical_module_slots": 12,
        "logical_connection_slots": 32,
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
    },
    {
        "profile_code": "b16",
        "profile_id": "b2-16x48-v1",
        "profile_index": 1,
        "logical_module_slots": 16,
        "logical_connection_slots": 48,
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
    },
)

_ROLES: Final[tuple[dict[str, Any], ...]] = (
    {"role_id": "r0", "ordinal": 0, "name": "intact_reference"},
    {"role_id": "r1", "ordinal": 1, "name": "neutral_sham_control"},
    {"role_id": "r2", "ordinal": 2, "name": "primary_lesion_or_control"},
    {"role_id": "r3", "ordinal": 3, "name": "secondary_lesion_or_control"},
    {"role_id": "r4", "ordinal": 4, "name": "state_or_boundary_control"},
)

_FAMILY_ROWS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    (
        "f01",
        "legacy_and_neutrality",
        (
            "empty record for the batch-level modular-disabled legacy arm",
            "empty record for the batch-level modular-enabled empty-envelope arm",
            "neutral scalar integrator",
            "neutral gated scalar module",
            "zero input binding",
            "zero output and development expression",
            "dormant module with retained private state",
            "neutral duplicate plus canonical clean free slot",
        ),
    ),
    (
        "f02",
        "one_active_integrator",
        (
            "self-energy positive input and positive prior state",
            "self-energy negative input and negative prior state",
            "self-health positive input and negative prior state",
            "self-health negative input and positive prior state",
            "positive imported previous-tick connection aggregate",
            "negative imported previous-tick connection aggregate",
            "partial developmental gain with movement routing",
            "REST routing with signed state and input",
        ),
    ),
    (
        "f03",
        "one_gated_recurrent_module",
        (
            "strong retention with update bias at most -4 and zero update gains",
            "rapid update with update bias at least +4 and zero update gains",
            "reset-driven forgetting with reset bias at most -4 and update bias at least +4",
            "positive candidate",
            "negative candidate",
            "self-energy routing",
            "self-health mature dependency-ready routing with one neutral prerequisite provider",
            "previous-tick input with blocked dependency, REST route, and one neutral prerequisite provider",
        ),
    ),
    (
        "f04",
        "two_module_chains",
        (
            "integrator-to-integrator positive first-only output",
            "integrator-to-gated positive second-only output",
            "gated-to-integrator positive shared-action output",
            "gated-to-gated positive different-action output",
            "gated-to-gated negative connection",
            "cancellation-sensitive shared-action connection",
            "dormant first module",
            "development-blocked second module",
        ),
    ),
    (
        "f05",
        "three_module_acyclic_structures",
        (
            "A-to-B-to-C chain",
            "A-and-B converge on C",
            "A branches to B and C",
            "two prerequisites for C",
            "dormant predecessor",
            "developmentally inactive predecessor",
            "mixed integrator/gated operators and inputs",
            "mixed outputs with movement/REST arbitration",
        ),
    ),
    (
        "f06",
        "previous_tick_recurrent_motifs",
        (
            "positive reciprocal pair",
            "negative reciprocal pair",
            "zero-sum signed reciprocal pair",
            "positive three-module previous-tick loop",
            "negative three-module previous-tick loop",
            "zero-sum signed three-module previous-tick loop",
            "loop with dormant member",
            "signed explicitly imported previous outputs",
        ),
    ),
    (
        "f07",
        "development_and_dependencies",
        (
            "early active",
            "mature activation",
            "late imported inactive state",
            "dependency blocked",
            "dependency ready",
            "latched readiness",
            "dormant despite latched readiness",
            "reactivated partial-gain and exact-zero expression",
        ),
    ),
    (
        "f08",
        "dormancy",
        (
            "dormant integrator with stored state",
            "dormant gated module with stored state",
            "dormant module with old previous output",
            "dormant module with active routing records",
            "dormant predecessor in a chain",
            "dormant output target",
            "matched active integrator",
            "matched active gated module",
        ),
    ),
    (
        "f09",
        "prune_and_reuse_boundary",
        (
            "eligible dormant module",
            "eligible output-neutral module",
            "dependency-blocked prune request",
            "last-generic-module prune request",
            "incident input and output bindings",
            "incident recurrent connections",
            "deterministic repack ordering",
            "freed slot prepared for new biological identity",
        ),
    ),
    (
        "f10",
        "structural_cost",
        (
            "one action-neutral active integrator",
            "one action-neutral active gated module",
            "several action-neutral active modules",
            "counted zero-contribution connections",
            "counted input and output bindings",
            "counted dependency edge",
            "dormant and pruned reserve",
            "near-cap action-neutral structure",
        ),
    ),
    (
        "f11",
        "capacity_migration",
        (
            "legacy-profile mixed source structure",
            "no-migration matched control",
            "unauthorized or insufficient-pressure request",
            "wrong boundary profile or digest request",
            "accepted migration with canonical neutral tail",
            "accepted migration with routing and connections",
            "accepted migration with development dependency dormancy and cost",
            "first continuation with preserved identity and slots",
        ),
    ),
    (
        "f12",
        "checkpoint_continuation",
        (
            "neutral or empty representative",
            "single gated representative",
            "two-module chain representative",
            "development and dependency representative",
            "dormancy representative",
            "structural-cost representative",
            "prune and reuse representative",
            "capacity-migration representative",
        ),
    ),
)

_COMMON_UNCHANGED_SURFACES: Final[tuple[str, ...]] = (
    "action_vocabulary",
    "deterministic_action_binding",
    "fixed_envelope_shapes",
    "owner_identity",
    "profile_identity",
    "unrelated_fixture_bytes",
    "unrelated_identity_and_slot_maps",
    "unrelated_rng_schedule",
)

_COMMON_ROLE_INTERVENTIONS: Final[dict[str, dict[str, str]]] = {
    "f02": {
        "r2": "remove_input_binding",
        "r3": "remove_output_binding",
        "r4": "reset_state",
    },
    "f03": {
        "r2": "remove_input_binding",
        "r3": "block_dependency",
        "r4": "reset_state",
    },
    "f04": {
        "r2": "remove_chain_connection",
        "r3": "remove_downstream_output",
        "r4": "enable_downstream_dormancy",
    },
    "f05": {
        "r2": "remove_first_structural_connection",
        "r3": "block_prerequisite",
        "r4": "enable_member_dormancy",
    },
    "f06": {
        "r2": "remove_recurrent_connection",
        "r3": "enable_member_dormancy",
        "r4": "reset_member_state",
    },
    "f07": {
        "r2": "set_development_expression_zero",
        "r3": "block_dependency",
        "r4": "enable_target_dormancy",
    },
    "f08": {
        "r2": "enable_dormancy_on_active_twin",
        "r3": "reset_state",
        "r4": "remove_output_binding",
    },
    "f10": {
        "r2": "dormant_output_neutral_module",
        "r3": "remove_zero_contribution_connection",
        "r4": "prune_output_neutral_structure",
    },
}

_F11_P12: Final[dict[str, str]] = {
    "r0": "source_mixed_structure_intact_tick",
    "r1": "no_migration_sham_control",
    "r2": "reject_unauthorized_request",
    "r3": "reject_wrong_safe_boundary",
    "r4": "source_checkpoint_fresh_owner_before_migration",
}
_F11_P16: Final[dict[str, str]] = {
    "r0": "accepted_migration_exact_first_tick",
    "r1": "accepted_migration_canonical_neutral_tail",
    "r2": "accepted_migration_integrator_gated_routing_connections",
    "r3": "accepted_migration_development_dependency_dormancy_cost",
    "r4": "post_migration_checkpoint_fresh_owner_first_tick",
}
_F12_P12: Final[dict[str, str]] = {
    "r0": "neutral_empty",
    "r1": "single_gated",
    "r2": "two_module_chain",
    "r3": "development_dependency",
    "r4": "dormancy",
}
_F12_P16: Final[dict[str, str]] = {
    "r0": "structural_cost",
    "r1": "prune_reuse",
    "r2": "capacity_migration_post_boundary",
    "r3": "gated_chain_composite",
    "r4": "development_dormancy_cost_composite",
}


class PhaseFV2MatrixError(ValueError):
    """The planning-only matrix is malformed, non-canonical, or contradicted."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the exact digest preimage used by matrix and case receipts."""

    try:
        return canonical_bytes(value)
    except PhaseFCanonicalJsonError as exc:
        raise PhaseFV2MatrixError("value is not canonical ASCII JSON") from exc


def canonical_sha256(value: Any) -> str:
    try:
        return _canonical_sha256(value)
    except PhaseFCanonicalJsonError as exc:
        raise PhaseFV2MatrixError("value is not canonical ASCII JSON") from exc


def _with_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload[field] = canonical_sha256(payload)
    return payload


def _family_specs() -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    for ordinal, (family_id, name, descriptions) in enumerate(_FAMILY_ROWS, start=1):
        variants = [
            {"variant_id": f"v{index}", "description": description}
            for index, description in enumerate(descriptions)
        ]
        families.append(
            _with_digest(
                {
                    "family_id": family_id,
                    "ordinal": ordinal,
                    "name": name,
                    "panel_variants": variants,
                },
                "panel_sha256",
            )
        )
    return families


def _case_id(family_id: str, profile_code: str, role_id: str) -> str:
    return f"pf2_{family_id}_{profile_code}_{role_id}"


def _construction_seed(family_ordinal: int, profile_index: int, role_index: int) -> int:
    return 620_000 + family_ordinal * 100 + profile_index * 10 + role_index


def _slot_plan() -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for variant in range(8):
        slots.append(
            {
                "slot": variant,
                "variant_id": f"v{variant}",
                "twin_slot": variant + 8,
                "intervention_eligible": True,
            }
        )
    for variant in range(8):
        slots.append(
            {
                "slot": variant + 8,
                "variant_id": f"v{variant}",
                "twin_slot": variant,
                "intervention_eligible": False,
            }
        )
    return slots


def _checkpoint_schedule(
    family_id: str, profile_code: str, role_id: str
) -> dict[str, Any]:
    kind = "none"
    if family_id == "f12":
        if profile_code == "b16" and role_id == "r1":
            kind = "post_reuse_pre_tick"
        elif profile_code == "b16" and role_id == "r2":
            kind = "post_migration_pre_tick"
        else:
            kind = "pre_tick_fresh_owner"
    elif family_id == "f09" and role_id == "r4":
        kind = "post_reuse_pre_tick"
    elif family_id == "f11" and role_id == "r4":
        kind = (
            "pre_migration_source"
            if profile_code == "b12"
            else "post_migration_pre_tick"
        )
    return {
        "kind": kind,
        "resumed_repeats": 0 if kind == "none" else RESUMED_REPEATS,
    }


def _role_action(family_id: str, profile_code: str, role_id: str) -> str:
    if family_id == "f01":
        return {
            "r0": "modular_disabled_legacy",
            "r1": "modular_enabled_empty",
            "r2": "neutral_operator_and_zero_binding_panel",
            "r3": "zero_development_and_dormancy_panel",
            "r4": "neutral_duplicate_and_clean_free_slot_panel",
        }[role_id]
    if family_id == "f09":
        return {
            "r0": "pre_boundary_reference",
            "r1": "no_op_prune_sham",
            "r2": "accepted_same_slot_prune_reuse_cleanup",
            "r3": "rejected_prune_request",
            "r4": "reuse_freed_slot_with_new_identity",
        }[role_id]
    if family_id == "f11":
        return (_F11_P12 if profile_code == "b12" else _F11_P16)[role_id]
    if family_id == "f12":
        return (_F12_P12 if profile_code == "b12" else _F12_P16)[role_id]
    if role_id == "r0":
        return "intact_variation_panel"
    if role_id == "r1":
        return "byte_identical_no_op_sham"
    return _COMMON_ROLE_INTERVENTIONS[family_id][role_id]


def _claim_kind(family_id: str, role_id: str) -> str:
    if family_id == "f01":
        return "legacy_neutrality"
    if family_id == "f09":
        return (
            "boundary_integrity"
            if role_id in {"r2", "r3", "r4"}
            else "boundary_control"
        )
    if family_id == "f11":
        return "migration_integrity"
    if family_id == "f12":
        return "checkpoint_equivalence"
    if role_id == "r0":
        return "intact_reference"
    if role_id == "r1":
        return "neutral_control"
    if family_id == "f10":
        return "reporting_only_cost_isolation"
    return "matched_causal_lesion"


def _target_slots(family_id: str, role_id: str) -> list[int]:
    if role_id == "r0" or family_id == "f12":
        return []
    sparse = {
        ("f01", "r2"): [4, 5, 6],
        ("f01", "r3"): [2, 3, 4, 5, 6, 7],
        ("f01", "r4"): [2, 3, 4, 5, 6, 7],
        ("f02", "r2"): [0, 1, 2, 3, 6, 7],
        ("f03", "r2"): [0, 1, 2, 3, 4, 5, 6],
        ("f03", "r3"): [6, 7],
        ("f04", "r3"): [1, 2, 3, 4, 5, 6, 7],
        ("f05", "r3"): [3],
        ("f09", "r2"): [0, 1, 4, 5, 6, 7],
        ("f09", "r3"): [2, 3],
        ("f09", "r4"): [0, 1, 4, 5, 6, 7],
        ("f10", "r3"): [2, 3, 4, 5, 6, 7],
    }
    return list(sparse.get((family_id, role_id), range(8)))


def _changed_surfaces(family_id: str, profile_code: str, role_id: str) -> list[str]:
    if role_id in {"r0", "r1"}:
        return []
    if family_id == "f01":
        return ["modular_structure_metadata", "structural_cost"]
    if family_id == "f09":
        return [
            "modular_identity_map",
            "modular_slot_map",
            "modular_structure_metadata",
        ]
    if family_id == "f10":
        return ["structural_cost"]
    if family_id == "f11":
        if profile_code == "b12":
            # Source, rejected-request, and pre-migration checkpoint cases are
            # required to be exact state/profile/shape/RNG no-ops.
            return []
        return ["capacity_profile_metadata", "expanded_shape_digest"]
    if family_id == "f12":
        return []
    action = _role_action(family_id, profile_code, role_id)
    if "dormancy" in action or "dormant" in action:
        return [
            "activity_projection",
            "named_module_trace",
            "named_routed_residual",
        ]
    if "output" in action:
        return ["named_routed_residual"]
    if "reset_state" in action or "reset_member_state" in action:
        return ["named_state_projection"]
    if (
        "input" in action
        or "connection" in action
        or "dependency" in action
        or "prerequisite" in action
        or "development" in action
    ):
        return ["named_module_trace"]
    raise PhaseFV2MatrixError(
        f"causal action lacks an exact changed-surface contract: {action}"
    )


def _boundary_source_case_id(
    family_id: str, profile_code: str, role_id: str
) -> str | None:
    if family_id == "f11" and profile_code == "b16":
        return _case_id(family_id, "b12", role_id)
    if family_id == "f12" and profile_code == "b16" and role_id == "r2":
        return _case_id(family_id, "b12", role_id)
    return None


def _role_contract(family_id: str, profile_code: str, role_id: str) -> dict[str, Any]:
    return {
        "claim_kind": _claim_kind(family_id, role_id),
        "action": _role_action(family_id, profile_code, role_id),
        "target_slots": _target_slots(family_id, role_id),
        "boundary_source_case_id": _boundary_source_case_id(
            family_id, profile_code, role_id
        ),
        "expected_changed_surfaces": _changed_surfaces(
            family_id, profile_code, role_id
        ),
        "expected_unchanged_surfaces": list(_COMMON_UNCHANGED_SURFACES),
    }


def _relationship(family_id: str, profile_code: str, role_id: str) -> dict[str, Any]:
    current = _case_id(family_id, profile_code, role_id)
    base = _case_id(family_id, profile_code, "r0")
    neutral = _case_id(family_id, profile_code, "r1")
    if family_id == "f12":
        return {
            "kind": "checkpoint_self_pair",
            "base_case_id": current,
            "neutral_case_id": None,
        }
    if role_id == "r0":
        return {"kind": "reference", "base_case_id": None, "neutral_case_id": neutral}
    if role_id == "r1":
        return {
            "kind": "neutral_control",
            "base_case_id": base,
            "neutral_case_id": current,
        }
    return {
        "kind": "matched_intervention",
        "base_case_id": base,
        "neutral_case_id": neutral,
    }


def _fixture_blueprint(
    *,
    family: Mapping[str, Any],
    profile: Mapping[str, Any],
    role: Mapping[str, Any],
) -> dict[str, Any]:
    material = {
        "panel_id": f"{family['family_id']}_{profile['profile_code']}_panel_v1",
        "panel_sha256": family["panel_sha256"],
        "construction_seed": _construction_seed(
            int(family["ordinal"]), int(profile["profile_index"]), int(role["ordinal"])
        ),
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
        "slot_plan": _slot_plan(),
    }
    return _with_digest(material, "fixture_blueprint_sha256")


def _case(
    *,
    family: Mapping[str, Any],
    profile: Mapping[str, Any],
    role: Mapping[str, Any],
) -> dict[str, Any]:
    family_id = str(family["family_id"])
    profile_code = str(profile["profile_code"])
    role_id = str(role["role_id"])
    checkpoint = _checkpoint_schedule(family_id, profile_code, role_id)
    resumed = int(checkpoint["resumed_repeats"])
    material = {
        "case_id": _case_id(family_id, profile_code, role_id),
        "family_id": family_id,
        "profile_id": profile["profile_id"],
        "role_id": role_id,
        "fixture_blueprint": _fixture_blueprint(
            family=family, profile=profile, role=role
        ),
        "relationship": _relationship(family_id, profile_code, role_id),
        "role_contract": _role_contract(family_id, profile_code, role_id),
        "checkpoint_schedule": checkpoint,
        "oracle_recipe_id": ORACLE_RECIPE_ID,
        "owner_contract_id": OWNER_CONTRACT_ID,
        "comparison_contract_id": COMPARISON_CONTRACT_ID,
        "execution_accounting": {
            "uninterrupted_repeats": UNINTERRUPTED_REPEATS,
            "resumed_repeats": resumed,
            "cpu_oracle_evaluations": 1 + (1 if resumed else 0),
            "accelerated_invocations": UNINTERRUPTED_REPEATS + resumed,
        },
    }
    return _with_digest(material, "case_sha256")


def _protocol() -> dict[str, Any]:
    return {
        "simulation_cells_per_invocation": 1,
        "generations_per_invocation": 1,
        "accelerated_ticks_per_invocation": 1,
        "population": 16,
        "grid_width": 8,
        "grid_height": 8,
        "action_width": 7,
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
        "module_state_width": 1,
        "deterministic_actions": True,
        "births": False,
        "deaths": False,
        "cull": False,
        "refill": False,
        "turnover": False,
        "selection": False,
        "reproduction": False,
        "in_tick_topology_growth": False,
        "multi_cell_owner": False,
    }


def _comparison_contract() -> dict[str, Any]:
    return {
        "contract_id": COMPARISON_CONTRACT_ID,
        "categorical_rule": "exact",
        "float_rule": {"absolute_tolerance": 2e-7, "relative_tolerance": 0.0},
        "exact_fields": [
            "actions",
            "checkpoint_projection",
            "dependency_readiness",
            "developmental_stage",
            "dormancy",
            "fallback_count",
            "free_slot_maps",
            "homology",
            "identities",
            "modular_rng_draw_count",
            "module_and_connection_presence",
            "profile_and_owner_ids",
            "semantic_ids",
            "slot_maps",
            "validity_masks",
        ],
        "admitted_float_fields": [
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
        ],
        "expected_hidden_fallback_count": 0,
        "expected_modular_rng_draw_count": 0,
    }


def _owner_contract() -> dict[str, Any]:
    return {
        "contract_id": OWNER_CONTRACT_ID,
        "accelerated_owner": ACCELERATED_OWNER,
        "modular_tick_owner": MODULAR_TICK_OWNER,
        "logical_profiles": [profile["profile_id"] for profile in _PROFILES],
        "deterministic_action_binding_required": True,
        "single_cell_invocation_required": True,
        "hidden_fallback_forbidden": True,
        "modular_rng_forbidden": True,
    }


def _canonicalization_contract() -> dict[str, Any]:
    return {
        "encoding": "ascii",
        "json_format": "sort-keys-compact-no-nan-v1",
        "digest_algorithm": "sha256",
        "case_order": "family-ordinal_profile-index_role-ordinal",
        "result_fields_forbidden": True,
    }


def _build_matrix_unchecked() -> dict[str, Any]:
    families = _family_specs()
    cases = [
        _case(family=family, profile=profile, role=role)
        for family in families
        for profile in _PROFILES
        for role in _ROLES
    ]
    material = {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "status": MATRIX_STATUS,
        "bindings": {
            "authorized_base_commit": AUTHORIZED_BASE_COMMIT,
            "frozen_execution_commit": None,
            "entry_receipt_sha256": None,
            "cpu_expectation_manifest_sha256": None,
        },
        "canonicalization": _canonicalization_contract(),
        "protocol": _protocol(),
        "profiles": deepcopy(list(_PROFILES)),
        "families": families,
        "roles": deepcopy(list(_ROLES)),
        "owner_contract": _owner_contract(),
        "comparison_contract": _comparison_contract(),
        "case_count": CASE_COUNT,
        "checkpoint_case_count": CHECKPOINT_CASE_COUNT,
        "expected_cpu_oracle_evaluations": EXPECTED_CPU_ORACLE_EVALUATIONS,
        "expected_accelerated_invocations": EXPECTED_ACCELERATED_INVOCATIONS,
        "expected_total_required_evaluations": EXPECTED_TOTAL_REQUIRED_EVALUATIONS,
        "cases": cases,
    }
    return _with_digest(material, "matrix_sha256")


def build_phase_f_v2_matrix() -> dict[str, Any]:
    """Build a fresh canonical 120-case planning matrix."""

    matrix = _build_matrix_unchecked()
    validate_phase_f_v2_matrix(matrix)
    return matrix


def _exact_object(value: Any, keys: frozenset[str], *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PhaseFV2MatrixError(f"{path} must be an object with string keys")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise PhaseFV2MatrixError(
            f"{path} fields are not exact; missing={missing!r} extra={extra!r}"
        )
    return dict(value)


def _reject_result_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = sorted(_FORBIDDEN_RESULT_KEYS & set(value))
        if forbidden:
            raise PhaseFV2MatrixError(
                f"{path} contains observed-result fields: {forbidden}"
            )
        for key, child in value.items():
            _reject_result_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_result_keys(child, path=f"{path}[{index}]")


def _validate_nested_shapes(matrix: Mapping[str, Any]) -> None:
    _exact_object(
        matrix["bindings"],
        frozenset(
            {
                "authorized_base_commit",
                "frozen_execution_commit",
                "entry_receipt_sha256",
                "cpu_expectation_manifest_sha256",
            }
        ),
        path="$.bindings",
    )
    for field in (
        "frozen_execution_commit",
        "entry_receipt_sha256",
        "cpu_expectation_manifest_sha256",
    ):
        if matrix["bindings"][field] is not None:
            raise PhaseFV2MatrixError(
                f"$.bindings.{field} must remain a planning placeholder"
            )
    if _COMMIT_RE.fullmatch(str(matrix["bindings"]["authorized_base_commit"])) is None:
        raise PhaseFV2MatrixError("$.bindings.authorized_base_commit is not a commit")

    for index, family in enumerate(matrix["families"]):
        _exact_object(
            family,
            frozenset(
                {"family_id", "ordinal", "name", "panel_variants", "panel_sha256"}
            ),
            path=f"$.families[{index}]",
        )
        for variant_index, variant in enumerate(family["panel_variants"]):
            _exact_object(
                variant,
                frozenset({"variant_id", "description"}),
                path=f"$.families[{index}].panel_variants[{variant_index}]",
            )
    for index, case in enumerate(matrix["cases"]):
        _exact_object(
            case,
            frozenset(
                {
                    "case_id",
                    "family_id",
                    "profile_id",
                    "role_id",
                    "fixture_blueprint",
                    "relationship",
                    "role_contract",
                    "checkpoint_schedule",
                    "oracle_recipe_id",
                    "owner_contract_id",
                    "comparison_contract_id",
                    "execution_accounting",
                    "case_sha256",
                }
            ),
            path=f"$.cases[{index}]",
        )
        fixture = _exact_object(
            case["fixture_blueprint"],
            frozenset(
                {
                    "panel_id",
                    "panel_sha256",
                    "construction_seed",
                    "physical_module_slots",
                    "physical_connection_slots",
                    "slot_plan",
                    "fixture_blueprint_sha256",
                }
            ),
            path=f"$.cases[{index}].fixture_blueprint",
        )
        for slot_index, slot in enumerate(fixture["slot_plan"]):
            _exact_object(
                slot,
                frozenset({"slot", "variant_id", "twin_slot", "intervention_eligible"}),
                path=f"$.cases[{index}].fixture_blueprint.slot_plan[{slot_index}]",
            )
        _exact_object(
            case["relationship"],
            frozenset({"kind", "base_case_id", "neutral_case_id"}),
            path=f"$.cases[{index}].relationship",
        )
        _exact_object(
            case["role_contract"],
            frozenset(
                {
                    "claim_kind",
                    "action",
                    "target_slots",
                    "boundary_source_case_id",
                    "expected_changed_surfaces",
                    "expected_unchanged_surfaces",
                }
            ),
            path=f"$.cases[{index}].role_contract",
        )
        _exact_object(
            case["checkpoint_schedule"],
            frozenset({"kind", "resumed_repeats"}),
            path=f"$.cases[{index}].checkpoint_schedule",
        )
        _exact_object(
            case["execution_accounting"],
            frozenset(
                {
                    "uninterrupted_repeats",
                    "resumed_repeats",
                    "cpu_oracle_evaluations",
                    "accelerated_invocations",
                }
            ),
            path=f"$.cases[{index}].execution_accounting",
        )


def validate_phase_f_v2_matrix(value: Any) -> dict[str, Any]:
    """Validate the exact planning matrix and return a detached copy."""

    matrix = _exact_object(
        value,
        frozenset(
            {
                "schema_version",
                "matrix_id",
                "status",
                "bindings",
                "canonicalization",
                "protocol",
                "profiles",
                "families",
                "roles",
                "owner_contract",
                "comparison_contract",
                "case_count",
                "checkpoint_case_count",
                "expected_cpu_oracle_evaluations",
                "expected_accelerated_invocations",
                "expected_total_required_evaluations",
                "cases",
                "matrix_sha256",
            }
        ),
        path="$",
    )
    _reject_result_keys(matrix)
    try:
        _validate_nested_shapes(matrix)
    except (KeyError, TypeError) as exc:
        raise PhaseFV2MatrixError("matrix contains malformed nested values") from exc

    digest = matrix.get("matrix_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise PhaseFV2MatrixError("$.matrix_sha256 is not lowercase SHA-256")
    unhashed = dict(matrix)
    unhashed.pop("matrix_sha256")
    if digest != canonical_sha256(unhashed):
        raise PhaseFV2MatrixError("matrix SHA-256 does not match canonical payload")

    expected = _build_matrix_unchecked()
    if matrix != expected:
        raise PhaseFV2MatrixError(
            "matrix does not match the frozen 12x2x5 construction"
        )

    cases = matrix["cases"]
    if len(cases) != CASE_COUNT or matrix["case_count"] != CASE_COUNT:
        raise PhaseFV2MatrixError("matrix must contain exactly 120 logical cases")
    checkpoint_cases = [
        case for case in cases if case["checkpoint_schedule"]["kind"] != "none"
    ]
    if len(checkpoint_cases) != CHECKPOINT_CASE_COUNT:
        raise PhaseFV2MatrixError("matrix must contain exactly 14 checkpoint cases")
    if (
        sum(case["execution_accounting"]["cpu_oracle_evaluations"] for case in cases)
        != EXPECTED_CPU_ORACLE_EVALUATIONS
    ):
        raise PhaseFV2MatrixError("CPU oracle evaluation arithmetic is inconsistent")
    if (
        sum(case["execution_accounting"]["accelerated_invocations"] for case in cases)
        != EXPECTED_ACCELERATED_INVOCATIONS
    ):
        raise PhaseFV2MatrixError("accelerated invocation arithmetic is inconsistent")
    return deepcopy(matrix)


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise PhaseFV2MatrixError(f"duplicate JSON field {key!r}")
        result[key] = item
    return result


def _reject_json_constant(value: str) -> None:
    raise PhaseFV2MatrixError(f"non-finite JSON number {value!r} is forbidden")


def load_phase_f_v2_matrix(path: str | Path) -> dict[str, Any]:
    """Load one strict JSON matrix without duplicate keys or non-finite values."""

    source = Path(path)
    try:
        text = source.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise PhaseFV2MatrixError(f"cannot read ASCII matrix: {source}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise PhaseFV2MatrixError(f"matrix is not valid JSON: {source}") from exc
    return validate_phase_f_v2_matrix(value)


__all__ = [
    "AUTHORIZED_BASE_COMMIT",
    "CASE_COUNT",
    "CHECKPOINT_CASE_COUNT",
    "EXPECTED_ACCELERATED_INVOCATIONS",
    "EXPECTED_CPU_ORACLE_EVALUATIONS",
    "EXPECTED_TOTAL_REQUIRED_EVALUATIONS",
    "MATRIX_ID",
    "SCHEMA_VERSION",
    "PhaseFV2MatrixError",
    "build_phase_f_v2_matrix",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_phase_f_v2_matrix",
    "validate_phase_f_v2_matrix",
]
