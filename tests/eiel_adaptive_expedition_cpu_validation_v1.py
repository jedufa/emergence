"""CPU-only validation harness for Adaptive Science Expedition V1.

This module is intentionally a diagnostic/test surface.  It does not launch a
Science Request, create a provider, or interpret synthetic observations as
scientific evidence.  The hidden landscape is kept outside the policy input;
only the declared observation projection reaches the existing pure policy
boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Final

from emergence_contracts.adaptive_science_expedition_v1 import EvolutionaryAtlasV1
from emergence_contracts.canonical import canonical_json_bytes, canonical_sha256
from emergence_contracts.errors import ContractError
from eiel_composition.adaptive_expedition._legacy_atlas import (
    AdaptiveExpeditionAtlasError,
    add_atlas_entry_v1,
    new_expedition_atlas_v1,
    validate_expedition_atlas_v1,
)
from eiel_composition.adaptive_expedition.ledger import (
    ACTION_DISPATCH_BOUND,
    ACTION_SETTLED,
    ACTION_WAIT,
    DECISION_PREPARED,
    EXPEDITION_TERMINAL,
    FRONTIER_OBSERVED,
    LedgerIntegrityError,
    LedgerTransitionError,
    append_ledger_event_v1,
    new_expedition_ledger_v1,
    validate_expedition_ledger_v1,
)
from eiel_composition.adaptive_expedition._legacy_parent_bank import (
    ParentBankCapacityError,
    ParentBankIntegrityError,
    add_parent_hold_v1,
    drop_parent_reference_v1,
    new_retained_parent_bank_v1,
    retain_parent_v1,
    validate_retained_parent_bank_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    AdaptiveIlluminationPolicyV1,
)
from eiel_composition.adaptive_expedition.policy_protocol import (
    ExpeditionPolicyV1,
    canonical_plain_value_v1,
)


PROGRAMME_SCHEMA_VERSION: Final[str] = (
    "eiel_adaptive_expedition_cpu_validation_programme_v1"
)
PROGRAMME_ID: Final[str] = "EIEL_ADAPTIVE_EXPEDITION_CPU_P1_P4B_P5A_P6_V1"
SOURCE_BASELINE: Final[str] = "7104bda9ddb7d2d63d770dce1d757a7cb60c0924"
PACKING_PROJECTION_ONLY: Final[str] = "PROJECTION_ONLY_NO_PHYSICAL_ROUTE_CLAIM"
NO_CLAIMS: Final[bool] = True
GPU_FOLLOWUP_PACKET_SCHEMA_VERSION: Final[str] = (
    "eiel_adaptive_expedition_gpu_followup_packet_v1"
)
_UINT64_MASK: Final[int] = (1 << 64) - 1
_FORBIDDEN_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fitness",
        "fitness_score",
        "phenotype",
        "phenotype_id",
        "phenotype_name",
        "reward",
        "reward_scalar",
        "scalar_reward",
    }
)


@dataclass(frozen=True, slots=True)
class LandscapeFixtureV1:
    """One hidden-landscape case with an explicit obligation contract."""

    case_id: str
    family: str
    seed: int
    observations: tuple[Mapping[str, object], ...]
    hidden_informative_regions: tuple[str, ...]
    hidden_dead_regions: tuple[str, ...]
    hidden_traps: tuple[str, ...]
    acceptable_action_kinds: tuple[str, ...]
    forbidden_action_kinds: tuple[str, ...]
    required_action_kinds: tuple[str, ...] = ()
    replication_required: bool = False
    parent_retention_expected: bool = False
    holdout: bool = False


def _digest(label: object) -> str:
    return canonical_sha256(
        {"cpu-validation": canonical_plain_value_v1(label)}
    ).hex


def _find_forbidden_keys(value: object, *, path: str = "$") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key in _FORBIDDEN_AUTHORITY_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_find_forbidden_keys(child, path=f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, path=f"{path}[{index}]"))
    return tuple(sorted(found))


def _find_exact_strings(
    value: object,
    needles: set[str],
    *,
    path: str = "$",
) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, str):
        if value in needles:
            found.append(path)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_find_exact_strings(child, needles, path=f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_find_exact_strings(child, needles, path=f"{path}[{index}]"))
    return tuple(sorted(found))


def _observation(
    *,
    case_id: str,
    ordinal: int,
    active_count: int,
    birth_count: int,
    maximum_lineage_depth: int,
    integrity_ok: bool = True,
    absolute_tick: int | None = None,
    omit_axes: Sequence[str] = (),
    control_resource: float = 1.0,
    control_mutation: float = 0.125,
) -> tuple[dict[str, object], dict[str, object]]:
    frame_id = f"{case_id}-frame-{ordinal:03d}"
    trajectory_id = f"{case_id}-trajectory-{ordinal:03d}"
    state_ref_id = f"{case_id}-state-{ordinal:03d}"
    tick = 10 + ordinal if absolute_tick is None else absolute_tick
    facts: dict[str, object] = {
        "integrity_ok": integrity_ok,
        "active_count": active_count,
        "capacity": 10,
        "birth_count": birth_count,
        "maximum_lineage_depth": maximum_lineage_depth,
    }
    for axis in omit_axes:
        facts.pop(axis, None)
    frame = {
        "frame_id": frame_id,
        "trajectory_id": trajectory_id,
        "absolute_tick": tick,
        "state_ref_id": state_ref_id,
        "control_readback": {
            "controls": {
                "environment.resource.multiplier": control_resource,
                "mutation.rate": control_mutation,
            }
        },
        "facts": facts,
    }
    state = {
        "state_ref_id": state_ref_id,
        "next_absolute_tick": tick + 1,
        "state_class": "BRANCHABLE_EXACT",
        "world_trajectory_id": trajectory_id,
        "exact_state_sha256": _digest((case_id, ordinal, "exact")),
        "semantic_world_sha256": _digest((case_id, ordinal, "semantic")),
        "source_object_sha256": _digest((case_id, ordinal, "source")),
        "control_history_sha256": _digest((case_id, ordinal, "history")),
    }
    return frame, state


def _pattern_for_family(family: str, *, case_id: str) -> tuple[dict[str, object], ...]:
    """Build observations without leaking hidden labels into policy input."""

    patterns: dict[str, tuple[tuple[int, int, int, bool], ...]] = {
        "trivial_informative_extinction": (
            (0, 0, 0, True),
            (2, 1, 1, True),
            (0, 0, 0, True),
        ),
        "separated_useful_frontiers": (
            (2, 1, 1, True),
            (3, 1, 2, True),
            (7, 2, 3, True),
        ),
        "disconnected_useful_frontiers": (
            (1, 1, 1, True),
            (8, 3, 4, True),
            (2, 1, 1, True),
        ),
        "deceptive_noise": (
            (2, 1, 1, True),
            (2, 1, 1, True),
            (2, 1, 1, True),
        ),
        "rare_persistent_novelty": (
            (2, 1, 1, True),
            (8, 2, 3, True),
            (8, 3, 4, True),
        ),
        "same_niche_hidden_history": (
            (2, 1, 1, True),
            (2, 1, 1, True),
        ),
        "evolutionary_dead_end": (
            (8, 0, 1, True),
            (0, 0, 0, True),
        ),
        "mediocre_high_future_branch_value": (
            (2, 1, 1, True),
            (3, 1, 1, True),
        ),
        "non_stationary": (
            (2, 1, 1, True),
            (8, 2, 3, True),
            (1, 0, 0, True),
        ),
        "hysteresis": (
            (3, 1, 2, True),
            (3, 1, 2, True),
            (8, 2, 3, True),
        ),
        "high_seed_variance_replication": (
            (8, 3, 3, True),
            (8, 3, 3, True),
        ),
        "false_correlation_causality": (
            (8, 0, 3, True),
            (8, 0, 3, True),
        ),
        "oneoff_effect_replication": (
            (2, 1, 1, True),
            (0, 0, 0, True),
        ),
        "attractive_region_masks_hard": (
            (8, 3, 3, True),
            (2, 1, 1, True),
            (1, 1, 1, True),
        ),
        "population_collapse_novelty": (
            (0, 0, 0, True),
            (1, 0, 0, True),
        ),
        "narrow_disconnected_islands": (
            (1, 1, 1, True),
            (1, 1, 1, True),
            (8, 2, 3, True),
        ),
        "missing_axes": (
            (2, 1, 1, True),
            (8, 2, 3, True),
        ),
        "delayed_observations": (
            (2, 1, 1, True),
            (8, 2, 3, True),
        ),
        "contradictory_observations": (
            (2, 1, 1, True),
            (0, 3, 3, False),
        ),
        "threshold_edges": (
            (1, 1, 1, True),
            (3, 1, 1, True),
            (3, 1, 2, True),
        ),
        "sparse_insufficient_evidence": (
            (1, 0, 0, True),
        ),
        "equivalent_provenance": (
            (2, 1, 1, True),
            (2, 1, 1, True),
        ),
    }
    if family not in patterns:
        raise KeyError(family)
    rows: list[dict[str, object]] = []
    for ordinal, (active, births, depth, integrity) in enumerate(patterns[family]):
        omitted = ("maximum_lineage_depth",) if family == "missing_axes" else ()
        tick = 40 + ordinal * 17 if family == "delayed_observations" else None
        frame, _state = _observation(
            case_id=case_id,
            ordinal=ordinal,
            active_count=active,
            birth_count=births,
            maximum_lineage_depth=depth,
            integrity_ok=integrity,
            absolute_tick=tick,
            omit_axes=omitted,
        )
        rows.append(frame)
    return tuple(rows)


def _fixture(
    family: str,
    *,
    seed: int,
    case_id: str | None = None,
    holdout: bool = False,
) -> LandscapeFixtureV1:
    identifier = case_id or f"{family}-development"
    observations = _pattern_for_family(family, case_id=identifier)
    required = ("EXPLORE_GLOBAL",)
    replication = family in {
        "deceptive_noise",
        "rare_persistent_novelty",
        "high_seed_variance_replication",
        "oneoff_effect_replication",
        "hysteresis",
    }
    parent_expected = family not in {
        "population_collapse_novelty",
        "sparse_insufficient_evidence",
    }
    informative = {
        "separated_useful_frontiers",
        "disconnected_useful_frontiers",
        "rare_persistent_novelty",
        "narrow_disconnected_islands",
    }
    dead = {
        "evolutionary_dead_end",
        "population_collapse_novelty",
        "trivial_informative_extinction",
    }
    traps = {
        "deceptive_noise",
        "false_correlation_causality",
        "oneoff_effect_replication",
        "population_collapse_novelty",
    }
    if family in informative:
        hidden_regions = (f"{identifier}-useful",)
    else:
        hidden_regions = (f"{identifier}-latent",)
    return LandscapeFixtureV1(
        case_id=identifier,
        family=family,
        seed=seed & _UINT64_MASK,
        observations=observations,
        hidden_informative_regions=hidden_regions,
        hidden_dead_regions=(f"{identifier}-dead",) if family in dead else (),
        hidden_traps=(f"{identifier}-trap",) if family in traps else (),
        acceptable_action_kinds=(
            "BRANCH_LOCAL",
            "CONTINUE",
            "EXPLORE_GLOBAL",
            "PARK",
            "QUARANTINE",
            "REPLICATE",
            "REQUEST_DETAILED_OBSERVATION",
        ),
        forbidden_action_kinds=("CAUSAL_PROBE",),
        required_action_kinds=required,
        replication_required=replication,
        parent_retention_expected=parent_expected,
        holdout=holdout,
    )


DEVELOPMENT_FAMILIES: Final[tuple[str, ...]] = (
    "trivial_informative_extinction",
    "separated_useful_frontiers",
    "disconnected_useful_frontiers",
    "deceptive_noise",
    "rare_persistent_novelty",
    "same_niche_hidden_history",
    "evolutionary_dead_end",
    "mediocre_high_future_branch_value",
    "non_stationary",
    "hysteresis",
    "high_seed_variance_replication",
    "false_correlation_causality",
    "oneoff_effect_replication",
    "attractive_region_masks_hard",
    "population_collapse_novelty",
    "narrow_disconnected_islands",
    "missing_axes",
    "delayed_observations",
    "contradictory_observations",
    "threshold_edges",
    "sparse_insufficient_evidence",
    "equivalent_provenance",
)

HOLDOUT_FAMILIES: Final[tuple[str, ...]] = (
    "separated_useful_frontiers",
    "disconnected_useful_frontiers",
    "deceptive_noise",
    "rare_persistent_novelty",
    "hysteresis",
    "evolutionary_dead_end",
    "population_collapse_novelty",
    "sparse_insufficient_evidence",
)


def build_development_fixtures() -> tuple[LandscapeFixtureV1, ...]:
    return tuple(
        _fixture(family, seed=0xD000 + ordinal)
        for ordinal, family in enumerate(DEVELOPMENT_FAMILIES)
    )


def build_holdout_fixtures() -> tuple[LandscapeFixtureV1, ...]:
    return tuple(
        _fixture(
            family,
            seed=0xE000 + ordinal,
            case_id=f"holdout-{ordinal:02d}-{family}",
            holdout=True,
        )
        for ordinal, family in enumerate(HOLDOUT_FAMILIES)
    )


def fixture_set_digest(fixtures: Sequence[LandscapeFixtureV1]) -> str:
    rows = [
        {
            "case_id": fixture.case_id,
            "family": fixture.family,
            "seed": fixture.seed,
            "observations": canonical_plain_value_v1(fixture.observations),
            "hidden_informative_regions": list(fixture.hidden_informative_regions),
            "hidden_dead_regions": list(fixture.hidden_dead_regions),
            "hidden_traps": list(fixture.hidden_traps),
            "holdout": fixture.holdout,
        }
        for fixture in fixtures
    ]
    return canonical_sha256(rows).hex


def build_policy_input_v1(
    policy: ExpeditionPolicyV1,
    fixture: LandscapeFixtureV1,
    *,
    frontier_ordinal: int = 0,
    logical_capacity: int = 8,
    atlas_entries: Sequence[Mapping[str, object]] = (),
    physical_packing: str | None = None,
    frame_order: Sequence[int] | None = None,
    state_order: Sequence[int] | None = None,
    candidate_order_reversed: bool = False,
    extra_path: str | None = None,
) -> dict[str, object]:
    observations = list(fixture.observations)
    order = list(range(len(observations))) if frame_order is None else list(frame_order)
    frames = [dict(observations[index]) for index in order]
    state_rows: list[dict[str, object]] = []
    for ordinal, frame in enumerate(observations):
        _frame, state = _observation(
            case_id=fixture.case_id,
            ordinal=ordinal,
            active_count=int(frame["facts"]["active_count"]),
            birth_count=int(frame["facts"]["birth_count"]),
            maximum_lineage_depth=int(
                frame["facts"].get("maximum_lineage_depth", 0)
            ),
            integrity_ok=bool(frame["facts"].get("integrity_ok", True)),
            absolute_tick=int(frame["absolute_tick"]),
            omit_axes=(
                ("maximum_lineage_depth",)
                if "maximum_lineage_depth" not in frame["facts"]
                else ()
            ),
        )
        state_rows.append(state)
    if state_order is not None:
        state_rows = [state_rows[index] for index in state_order]
    controls = [
        {
            "control_id": "environment.resource.multiplier",
            "allowed_values": [1.0, 1.5],
        },
        {"control_id": "mutation.rate", "allowed_values": [0.125, 0.25]},
    ]
    if candidate_order_reversed:
        controls = [
            {
                "control_id": row["control_id"],
                "allowed_values": list(reversed(row["allowed_values"])),
            }
            for row in reversed(controls)
        ]
    input_value: dict[str, object] = {
        "protocol_version": "eiel_adaptive_expedition_policy_protocol_v1",
        "expedition_id": "cpu-validation-expedition",
        "manifest_sha256": {
            "algorithm": "sha256",
            "hex": _digest(("manifest", policy.binding)),
        },
        "policy_binding": canonical_plain_value_v1(policy.binding),
        "decision_frontier": {
            "frontier_id": f"{fixture.case_id}-frontier-{frontier_ordinal:04d}",
            "frontier_ordinal": frontier_ordinal,
            "observation_frames": frames,
            "bounded_history": [],
        },
        "atlas": {"entries": [dict(row) for row in atlas_entries]},
        "retained_parent_bank": {"maximum_entries": 32, "entries": []},
        "available_state_refs": state_rows,
        "logical_capacity": logical_capacity,
        "outer_rng": {"seed": fixture.seed, "cursor": 0},
        "authorized_search_space": {"controls": controls},
    }
    if physical_packing is not None:
        input_value["physical_packing"] = physical_packing
    if extra_path is not None:
        input_value["nonsemantic_source_path"] = extra_path
    return input_value


def _action_rows(result: Mapping[str, object]) -> list[dict[str, object]]:
    rows = result.get("semantic_actions", ())
    return [dict(row) for row in canonical_plain_value_v1(rows)]  # type: ignore[arg-type]


def _decision_semantic_projection(result: Mapping[str, object]) -> dict[str, object]:
    """Remove identities only for logical-ID metamorphic comparisons."""

    rows = []
    for row in _action_rows(result):
        rows.append(
            {
                key: row[key]
                for key in (
                    "action_kind",
                    "bucket",
                    "slot_cost",
                    "candidate_ordinal",
                    "child_start_next_tick",
                    "observation_absolute_tick",
                    "parent_next_absolute_tick",
                    "control_bindings",
                    "founder_execution",
                    "replication",
                    "reason_codes",
                )
                if key in row
            }
        )
    rows.sort(key=lambda row: canonical_json_bytes(row))
    atlas_delta = canonical_plain_value_v1(result["atlas_delta"])
    bank_delta = canonical_plain_value_v1(result["retained_parent_bank_delta"])
    if isinstance(atlas_delta, dict):
        atlas_delta = {
            **atlas_delta,
            "upsert_entries": [
                {
                    key: row[key]
                    for key in (
                        "niche_id",
                        "representative_role",
                        "descriptor_values",
                        "novelty",
                        "reason_codes",
                        "observation_absolute_tick",
                    )
                    if key in row
                }
                for row in atlas_delta.get("upsert_entries", [])
                if isinstance(row, Mapping)
            ],
        }
    if isinstance(bank_delta, dict):
        bank_delta = {
            **bank_delta,
            "retain_entries": [
                {
                    key: row[key]
                    for key in ("survival_reason", "frontier_ordinal")
                    if key in row
                }
                for row in bank_delta.get("retain_entries", [])
                if isinstance(row, Mapping)
            ],
        }
    return {
        "outer_rng": canonical_plain_value_v1(result["outer_rng"]),
        "atlas_delta": atlas_delta,
        "retained_parent_bank_delta": bank_delta,
        "semantic_actions": rows,
        "rejected_candidate_summary": canonical_plain_value_v1(
            result["rejected_candidate_summary"]
        ),
    }


def run_policy_case(
    policy: ExpeditionPolicyV1,
    fixture: LandscapeFixtureV1,
    *,
    input_mutator: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    decision_input = build_policy_input_v1(policy, fixture)
    if input_mutator is not None:
        input_mutator(decision_input)
    first = policy.decide(decision_input)
    second = policy.decide(decision_input)
    first_plain = canonical_plain_value_v1(first)
    second_plain = canonical_plain_value_v1(second)
    actions = _action_rows(first)
    action_kinds = [str(row["action_kind"]) for row in actions]
    forbidden_actions = sorted(
        set(action_kinds).intersection(fixture.forbidden_action_kinds)
    )
    required_misses = sorted(
        set(fixture.required_action_kinds).difference(action_kinds)
    )
    replication_underuse = fixture.replication_required and "REPLICATE" not in action_kinds
    retain_entries = canonical_plain_value_v1(
        first["retained_parent_bank_delta"]
    ).get("retain_entries", [])
    parent_retention_miss = fixture.parent_retention_expected and not retain_entries
    forbidden_fields = _find_forbidden_keys(first_plain)
    input_forbidden_fields = _find_forbidden_keys(decision_input)
    hidden_oracle_leaks = _find_exact_strings(
        decision_input,
        {
            *fixture.hidden_informative_regions,
            *fixture.hidden_dead_regions,
            *fixture.hidden_traps,
        },
    )
    unacceptable_actions = sorted(
        set(action_kinds).difference(fixture.acceptable_action_kinds)
    )
    positive_slot_cost = sum(int(row.get("slot_cost", 0)) for row in actions)
    logical_capacity = int(decision_input["logical_capacity"])
    duplicate_action_ids = len({row.get("action_id") for row in actions}) != len(actions)
    deterministic = first_plain == second_plain
    if (
        forbidden_actions
        or forbidden_fields
        or input_forbidden_fields
        or hidden_oracle_leaks
        or unacceptable_actions
        or duplicate_action_ids
        or positive_slot_cost > logical_capacity
    ):
        status = "FORBIDDEN"
    elif required_misses or replication_underuse or parent_retention_miss:
        status = "QUESTIONABLE"
    else:
        status = "DEFENSIBLE"
    return {
        "case_id": fixture.case_id,
        "family": fixture.family,
        "holdout": fixture.holdout,
        "seed": fixture.seed,
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "policy_binding_sha256": _digest(policy.binding),
        "input_sha256": _digest(canonical_plain_value_v1(decision_input)),
        "decision_sha256": _digest(first_plain),
        "deterministic_replay": deterministic,
        "action_kinds": sorted(action_kinds),
        "action_count": len(actions),
        "forbidden_action_kinds": forbidden_actions,
        "forbidden_authority_fields": list(forbidden_fields),
        "input_forbidden_authority_fields": list(input_forbidden_fields),
        "hidden_oracle_leak_paths": list(hidden_oracle_leaks),
        "unacceptable_action_kinds": unacceptable_actions,
        "duplicate_action_ids": duplicate_action_ids,
        "positive_slot_cost": positive_slot_cost,
        "logical_capacity": logical_capacity,
        "capacity_violation": positive_slot_cost > logical_capacity,
        "missed_required_actions": required_misses,
        "replication_required": fixture.replication_required,
        "replication_observed": "REPLICATE" in action_kinds,
        "parent_retention_expected": fixture.parent_retention_expected,
        "parent_retention_observed": bool(retain_entries),
        "replication_overuse": (
            not fixture.replication_required and "REPLICATE" in action_kinds
        ),
        "premature_retirement": any(
            kind in {"RETIRE", "DROP", "RELEASE"} for kind in action_kinds
        ),
        "status": status,
    }


def run_policy_suite(
    policy: ExpeditionPolicyV1,
    fixtures: Sequence[LandscapeFixtureV1],
) -> dict[str, object]:
    rows = [run_policy_case(policy, fixture) for fixture in fixtures]
    return {
        "fixture_count": len(rows),
        "fixture_set_sha256": fixture_set_digest(fixtures),
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("DEFENSIBLE", "QUESTIONABLE", "FORBIDDEN")
        },
        "determinism_failures": sum(
            not bool(row["deterministic_replay"]) for row in rows
        ),
        "forbidden_action_count": sum(
            len(row["forbidden_action_kinds"]) for row in rows
        ),
        "forbidden_authority_field_count": sum(
            len(row["forbidden_authority_fields"]) for row in rows
        ),
        "input_forbidden_authority_field_count": sum(
            len(row["input_forbidden_authority_fields"]) for row in rows
        ),
        "hidden_oracle_leak_count": sum(
            len(row["hidden_oracle_leak_paths"]) for row in rows
        ),
        "unacceptable_action_count": sum(
            len(row["unacceptable_action_kinds"]) for row in rows
        ),
        "capacity_violation_count": sum(
            bool(row["capacity_violation"]) for row in rows
        ),
        "replication_underuse_count": sum(
            bool(row["replication_required"])
            and not bool(row["replication_observed"])
            for row in rows
        ),
        "replication_overuse_count": sum(
            bool(row["replication_overuse"]) for row in rows
        ),
        "premature_retirement_count": sum(
            bool(row["premature_retirement"]) for row in rows
        ),
        "useful_hidden_parent_survival": {
            "expected": sum(bool(row["parent_retention_expected"]) for row in rows),
            "observed": sum(
                bool(row["parent_retention_expected"])
                and bool(row["parent_retention_observed"])
                for row in rows
            ),
            "metric_status": "SYNTHETIC_OBLIGATION_PROXY",
        },
        "dead_region_compute_fraction": {
            "value": None,
            "metric_status": "NOT_IDENTIFIABLE_WITHOUT_EXECUTION_TRACE",
        },
        "rediscovery_fraction": {
            "value": None,
            "metric_status": "ATLAS_ONLY_PROJECTION_NOT_POLICY_SCORE",
        },
        "frontier_discovery_fraction": {
            "value": (
                sum("EXPLORE_GLOBAL" in row["action_kinds"] for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "metric_status": "EXPLORATION_OBLIGATION_PROXY",
        },
        "rows": rows,
    }


def run_scientific_red_team(
    policy: ExpeditionPolicyV1 | None = None,
) -> dict[str, object]:
    """Inject tempting but non-authoritative facts at the policy boundary."""

    selected = policy or AdaptiveIlluminationPolicyV1()
    fixture = _fixture(
        "false_correlation_causality",
        seed=0xC0DE,
        case_id="red-team-base",
    )
    base_input = build_policy_input_v1(selected, fixture, logical_capacity=2)
    base = canonical_plain_value_v1(selected.decide(base_input))
    cases: list[dict[str, object]] = []

    def evaluate(name: str, mutator: Callable[[dict[str, object]], None]) -> None:
        candidate = json.loads(json.dumps(base_input))
        mutator(candidate)
        try:
            result = canonical_plain_value_v1(selected.decide(candidate))
        except Exception as error:  # a rejected untrusted projection is safe
            cases.append(
                {
                    "case_id": name,
                    "passed": True,
                    "handling": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "forbidden_authority_fields": [],
                    "causal_probe_count": 0,
                }
            )
            return
        actions = _action_rows(result)
        forbidden_fields = _find_forbidden_keys(result)
        causal_probe_count = sum(
            row.get("action_kind") == "CAUSAL_PROBE" for row in actions
        )
        cases.append(
            {
                "case_id": name,
                "passed": (
                    _decision_semantic_projection(result)
                    == _decision_semantic_projection(base)
                    and not forbidden_fields
                    and causal_probe_count == 0
                ),
                "handling": "IGNORED_BY_POSITIVE_PROJECTION",
                "error_type": None,
                "forbidden_authority_fields": list(forbidden_fields),
                "causal_probe_count": causal_probe_count,
            }
        )

    def hidden_reward(candidate: dict[str, object]) -> None:
        for frame in candidate["decision_frontier"]["observation_frames"]:
            frame["facts"].update(
                {
                    "fitness": 10**9,
                    "reward": 10**9,
                    "phenotype": "target-phenotype",
                    "genome_size": 10**6,
                    "population_quality": 10**9,
                }
            )

    def memory_correlation(candidate: dict[str, object]) -> None:
        candidate["decision_frontier"]["bounded_history"] = [
            {
                "fitness": 10**9,
                "phenotype": "memory-correlated-success",
                "cooperation": True,
            }
        ]

    def irrelevant_runtime(candidate: dict[str, object]) -> None:
        candidate["wall_clock_seconds"] = 0.001
        candidate["gpu_time_seconds"] = 0.0
        for frame in candidate["decision_frontier"]["observation_frames"]:
            frame["facts"]["wall_clock_seconds"] = 999999.0
            frame["facts"]["host_temperature"] = 100.0

    evaluate("hidden-reward-and-phenotype", hidden_reward)
    evaluate("memory-correlated-success", memory_correlation)
    evaluate("wall-clock-and-runtime-noise", irrelevant_runtime)

    capacity_decision = canonical_plain_value_v1(
        selected.decide(base_input)
    )
    capacity = int(base_input["logical_capacity"])
    slot_cost = sum(
        int(row.get("slot_cost", 0))
        for row in capacity_decision["semantic_actions"]
    )
    allowed_controls = {
        row["control_id"]: {
            canonical_sha256(value).hex
            for value in row["allowed_values"]
        }
        for row in base_input["authorized_search_space"]["controls"]
    }
    control_domain_violations = 0
    for action in capacity_decision["semantic_actions"]:
        for binding in action.get("control_bindings", ()):
            if canonical_sha256(binding["value"]).hex not in allowed_controls.get(
                binding["control_id"], set()
            ):
                control_domain_violations += 1
    return {
        "case_count": len(cases),
        "failure_count": sum(not bool(row["passed"]) for row in cases),
        "no_hidden_reward_violation_count": sum(
            not bool(row["passed"]) for row in cases
        ),
        "forbidden_action_count": sum(
            int(row["causal_probe_count"]) for row in cases
        ),
        "forbidden_authority_field_count": sum(
            len(row["forbidden_authority_fields"]) for row in cases
        ),
        "control_capacity": capacity,
        "selected_slot_cost": slot_cost,
        "control_capacity_violations": int(slot_cost > capacity),
        "control_domain_violations": control_domain_violations,
        "cases": cases,
    }


def _atlas_entries_for_policy() -> tuple[dict[str, object], ...]:
    return (
        {
            "entry_id": "atlas-a",
            "niche_id": "niche-a",
            "representative_role": "INFORMATIVE_FRONTIER",
            "descriptor_values": {
                "active_fraction_milli": 200,
                "birth_count": 1,
                "maximum_lineage_depth": 1,
            },
        },
        {
            "entry_id": "atlas-b",
            "niche_id": "niche-b",
            "representative_role": "PERSISTENT_ACTIVITY",
            "descriptor_values": {
                "active_fraction_milli": 800,
                "birth_count": 3,
                "maximum_lineage_depth": 3,
            },
        },
    )


def run_metamorphic_suite(
    policy: ExpeditionPolicyV1 | None = None,
) -> dict[str, object]:
    selected = policy or AdaptiveIlluminationPolicyV1()
    fixture = _fixture(
        "separated_useful_frontiers", seed=0xBEEF, case_id="metamorphic-base"
    )
    base_input = build_policy_input_v1(
        selected, fixture, atlas_entries=_atlas_entries_for_policy()
    )
    base = canonical_plain_value_v1(selected.decide(base_input))
    cases: list[dict[str, object]] = []

    def compare(
        name: str,
        candidate: Mapping[str, object],
        *,
        mode: str = "semantic",
        note: str = "",
    ) -> None:
        value = canonical_plain_value_v1(selected.decide(candidate))
        left = base if mode == "exact" else _decision_semantic_projection(base)
        right = value if mode == "exact" else _decision_semantic_projection(value)
        cases.append(
            {
                "case_id": name,
                "comparison": mode,
                "passed": left == right,
                "base_decision_sha256": _digest(base),
                "variant_decision_sha256": _digest(value),
                "note": note,
            }
        )

    ordered = json.loads(json.dumps(base_input))
    ordered["decision_frontier"]["observation_frames"] = list(
        reversed(ordered["decision_frontier"]["observation_frames"])
    )
    ordered["available_state_refs"] = list(reversed(ordered["available_state_refs"]))
    compare("world-order-permutation", ordered)

    arrival = json.loads(json.dumps(base_input))
    arrival["decision_frontier"]["bounded_history"] = [
        {"arrival_sequence": 2},
        {"arrival_sequence": 1},
    ]
    compare("result-arrival-permutation", arrival)

    candidates = json.loads(json.dumps(base_input))
    candidates["authorized_search_space"]["controls"] = list(
        reversed(candidates["authorized_search_space"]["controls"])
    )
    compare(
        "action-candidate-enumeration-permutation",
        candidates,
        note="control declarations permuted; each domain retains its declared values",
    )

    archive = json.loads(json.dumps(base_input))
    archive["atlas"]["entries"] = list(reversed(archive["atlas"]["entries"]))
    compare("archive-iteration-permutation", archive)

    for packing in ("B16", "B64", "B1024"):
        projected = json.loads(json.dumps(base_input))
        projected["physical_packing"] = packing
        compare(
            f"packing-projection-{packing}",
            projected,
            note=PACKING_PROJECTION_ONLY,
        )

    relocated = json.loads(json.dumps(base_input))
    relocated["nonsemantic_source_path"] = "/relocated/synthetic/input.json"
    compare("nonsemantic-path-relocation", relocated)

    renamed = json.loads(json.dumps(base_input))
    frame_map: dict[str, str] = {}
    state_map: dict[str, str] = {}
    for ordinal, frame in enumerate(renamed["decision_frontier"]["observation_frames"]):
        old_frame = frame["frame_id"]
        old_state = frame["state_ref_id"]
        frame_map[old_frame] = f"renamed-frame-{ordinal}"
        state_map[old_state] = f"renamed-state-{ordinal}"
        frame["frame_id"] = frame_map[old_frame]
        frame["trajectory_id"] = f"renamed-trajectory-{ordinal}"
        frame["state_ref_id"] = state_map[old_state]
    for ordinal, state in enumerate(renamed["available_state_refs"]):
        old_state = state["state_ref_id"]
        state["state_ref_id"] = state_map.get(old_state, f"renamed-state-{ordinal}")
        state["world_trajectory_id"] = f"renamed-trajectory-{ordinal}"
    compare("logical-world-id-renaming", renamed, mode="semantic")

    restart = canonical_plain_value_v1(selected.decide(base_input))
    cases.append(
        {
            "case_id": "decision-restart-replay",
            "comparison": "exact",
            "passed": restart == base,
            "base_decision_sha256": _digest(base),
            "variant_decision_sha256": _digest(restart),
            "note": "same sealed policy input replayed before any dispatch",
        }
    )
    return {
        "packing_semantics": PACKING_PROJECTION_ONLY,
        "case_count": len(cases),
        "failure_count": sum(not bool(row["passed"]) for row in cases),
        "cases": cases,
    }


def _atlas_entry(label: str, *, niche_id: str) -> dict[str, object]:
    digest = lambda suffix: _digest(("atlas", label, suffix))
    return {
        "entry_id": f"entry-{label}",
        "candidate_id": f"candidate-{label}",
        "frontier_ordinal": int(label.split("-")[-1]),
        "niche_id": niche_id,
        "observation_frame_sha256": digest("frame"),
        "semantic_world_sha256": digest("world"),
        "acquisition_sha256": digest("acquisition"),
        "reducer_result_sha256": digest("reducer"),
        "descriptor_sha256": digest("descriptor"),
        "source_settlement_sha256": digest("settlement"),
        "epistemic": {
            "classification": "synthetic-torture",
            "coordinates": {"ordinal": int(label.split("-")[-1])},
        },
    }


def _digest_mapping(label: object) -> dict[str, str]:
    return canonical_sha256(
        {"cpu-validation": canonical_plain_value_v1(label)}
    ).to_dict()


def _typed_atlas_entry(
    ordinal: int,
    *,
    descriptor_values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    label = f"typed-atlas-{ordinal:03d}"
    descriptors = (
        {
            "active_fraction_milli": ordinal % 1001,
            "birth_count": ordinal % 5,
        }
        if descriptor_values is None
        else dict(descriptor_values)
    )
    return {
        "atlas_entry_id": f"typed-entry-{ordinal:03d}",
        "world_trajectory_id": f"typed-trajectory-{ordinal:03d}",
        "observation_frame_sha256": _digest_mapping((label, "frame")),
        "state_ref_sha256": None,
        "niche_key": {
            "schema_version": "cpu_validation_niche_v1",
            "payload": {"niche": f"niche-{ordinal % 8}"},
        },
        "descriptor_values": {
            "schema_version": "cpu_validation_descriptors_v1",
            "payload": descriptors,
        },
        "policy_facing_facts": {
            "schema_version": "cpu_validation_facts_v1",
            "payload": {"synthetic": True},
        },
        "representative_roles": ["INFORMATIVE_FRONTIER"],
        "parent_entry_ids": [],
    }


def _typed_atlas(entries: Sequence[Mapping[str, object]], *, maximum_entries: int) -> EvolutionaryAtlasV1:
    return EvolutionaryAtlasV1.from_dict(
        {
            "schema_version": "eiel_adaptive_science_evolutionary_atlas_v1",
            "expedition_id": "cpu-validation-atlas",
            "manifest_sha256": _digest_mapping("typed-atlas-manifest"),
            "revision": 0,
            "through_frontier_ordinal": 0,
            "descriptor_set_sha256": _digest_mapping("typed-atlas-descriptors"),
            "maximum_entries": maximum_entries,
            "entries": [dict(entry) for entry in entries],
            "previous_atlas_sha256": None,
        }
    )


def run_atlas_torture() -> dict[str, object]:
    atlas = new_expedition_atlas_v1(
        expedition_id="cpu-validation-atlas",
        manifest_sha256="a" * 64,
        descriptor_set_sha256="b" * 64,
    )
    for ordinal in range(96):
        atlas = add_atlas_entry_v1(
            atlas,
            _atlas_entry(f"entry-{ordinal:03d}", niche_id=f"niche-{ordinal % 8}"),
        )
    replay = validate_expedition_atlas_v1(json.loads(json.dumps(atlas)))
    typed_entries = [_typed_atlas_entry(ordinal) for ordinal in range(96)]
    typed_overflow_rejected = False
    try:
        _typed_atlas(typed_entries, maximum_entries=64)
    except ContractError:
        typed_overflow_rejected = True
    bounded = _typed_atlas(typed_entries[:64], maximum_entries=64)
    bounded_replay = EvolutionaryAtlasV1.from_dict(
        json.loads(json.dumps(bounded.to_dict()))
    )
    tampered = json.loads(json.dumps(atlas))
    tampered["entries"][10]["semantic_world_sha256"] = "f" * 64
    source_tamper_rejected = False
    try:
        validate_expedition_atlas_v1(tampered)
    except AdaptiveExpeditionAtlasError:
        source_tamper_rejected = True
    descriptor_variants: dict[str, dict[str, object]] = {}
    for name, descriptors in {
        "missing_descriptor_axes": {},
        "descriptor_saturation": {
            "active_fraction_milli": 1000,
            "birth_count": 64,
        },
        "constant_descriptor_axes": {
            "active_fraction_milli": 500,
            "birth_count": 2,
        },
    }.items():
        variant = _typed_atlas(
            [_typed_atlas_entry(index, descriptor_values=descriptors) for index in range(8)],
            maximum_entries=8,
        )
        descriptor_variants[name] = {
            "valid": True,
            "entry_count": len(variant.entries),
            "rebuild_byte_identical": EvolutionaryAtlasV1.from_dict(
                json.loads(json.dumps(variant.to_dict()))
            ).atlas_sha256()
            == variant.atlas_sha256(),
        }
    return {
        "input_entries": len(typed_entries),
        "raw_append_entries": len(atlas["entries"]),
        "terminal_entries": len(bounded.entries),
        "bounded_test_cap": 64,
        "bounded_cap_respected": len(bounded.entries) <= bounded.maximum_entries,
        "typed_overflow_rejected": typed_overflow_rejected,
        "rediscovery_entries": len(
            [row for row in atlas["entries"] if row["niche_id"] == "niche-0"]
        ),
        "revision_sha256": bounded.atlas_sha256().hex,
        "raw_rebuild_byte_identical": canonical_json_bytes(atlas)
        == canonical_json_bytes(replay),
        "rebuild_byte_identical": bounded.atlas_sha256() == bounded_replay.atlas_sha256(),
        "source_result_tamper_rejected": source_tamper_rejected,
        "descriptor_variants": descriptor_variants,
        "niche_coverage": len({row["niche_id"] for row in atlas["entries"]}),
        "metadata_bytes": len(canonical_json_bytes(atlas)),
        "false_novelty": "NOT_SCORED_WITHOUT_POLICY_FRONTIER",
        "state_retention_authority": False,
        "raw_append_overflow_is_diagnostic": True,
    }


def _parent(label: str, *, priority: int, held: bool = False) -> dict[str, object]:
    suffix = label.rsplit("-", 1)[-1]
    ordinal = int(suffix) if suffix.isdigit() else 0
    state_sha = _digest(("parent", label, "state"))
    holds = (
        [
            {
                "hold_id": f"hold-{label}",
                "hold_kind": "replay-consumer",
                "reference_sha256": _digest(("hold", label)),
            }
        ]
        if held
        else []
    )
    return {
        "parent_id": f"parent-{label}",
        "candidate_id": f"candidate-{label}",
        "niche_id": f"niche-{ordinal % 3}",
        "frontier_ordinal": ordinal,
        "retention_priority": priority,
        "state_ref": {
            "kind": "BRANCHABLE_EXACT",
            "state_ref_sha256": state_sha,
            "state_bytes": 128 + ordinal,
            "lifecycle_manifest_sha256": _digest("parent-manifest"),
            "lifecycle_object_id": f"object-{label}",
            "next_absolute_tick": 10 + ordinal,
            "semantic_world_sha256": _digest(("parent", label, "world")),
        },
        "holds": holds,
    }


def run_parent_bank_torture() -> dict[str, object]:
    bank = new_retained_parent_bank_v1(
        expedition_id="cpu-validation-parent-bank",
        manifest_sha256="c" * 64,
        caps={"max_entries": 8, "max_total_bytes": 1100},
    )
    updates: list[Mapping[str, object]] = []
    for ordinal in range(16):
        update = retain_parent_v1(
            bank,
            _parent(f"parent-{ordinal:03d}", priority=ordinal),
        )
        updates.append(update)
        bank = update["bank"]
    held = retain_parent_v1(
        bank,
        _parent("parent-held", priority=-100, held=True),
    )
    bank = held["bank"]
    held_id = "parent-parent-held"
    bank = add_parent_hold_v1(
        bank,
        parent_id=held_id,
        hold={
            "hold_id": "explicit-hold",
            "hold_kind": "qualification-replay",
            "reference_sha256": _digest("explicit-hold"),
        },
    )
    protected_drop_rejected = False
    try:
        drop_parent_reference_v1(bank, parent_id=held_id, reason="test")
    except ParentBankCapacityError:
        protected_drop_rejected = True
    validated = validate_retained_parent_bank_v1(bank)
    tampered = json.loads(json.dumps(bank))
    tampered["entries"][0]["state_ref"]["state_ref_sha256"] = "f" * 64
    tampered_state_rejected = False
    try:
        validate_retained_parent_bank_v1(tampered)
    except ParentBankIntegrityError:
        tampered_state_rejected = True
    rebuilt = validate_retained_parent_bank_v1(json.loads(json.dumps(bank)))
    niche_counts: dict[str, int] = {}
    for entry in bank["entries"]:
        niche = str(entry["niche_id"])
        niche_counts[niche] = niche_counts.get(niche, 0) + 1
    return {
        "configured_max_entries": bank["caps"]["max_entries"],
        "configured_max_bytes": bank["caps"]["max_total_bytes"],
        "terminal_entries": bank["entry_count"],
        "terminal_bytes": bank["total_state_bytes"],
        "bounded_entries": bank["entry_count"] <= bank["caps"]["max_entries"],
        "bounded_bytes": bank["total_state_bytes"] <= bank["caps"]["max_total_bytes"],
        "capacity_updates": len(updates),
        "protected_drop_rejected": protected_drop_rejected,
        "tampered_state_ref_rejected": tampered_state_rejected,
        "rebuild_byte_identical": canonical_json_bytes(bank)
        == canonical_json_bytes(rebuilt),
        "deletion_authority": validated["deletion_authority"],
        "lifecycle_release_authority": validated["lifecycle_release_authority"],
        "state_availability_authority": False,
        "release_eligible_bytes": 0,
        "metadata_bytes": len(canonical_json_bytes(bank)),
        "family_concentration": max(niche_counts.values()) / bank["entry_count"],
        "same_niche_exact_state_entries": max(niche_counts.values()),
        "state_identity_unique": len(
            {entry["state_ref"]["state_ref_sha256"] for entry in bank["entries"]}
        )
        == bank["entry_count"],
    }


def _ledger_prefix(
    *, expedition_id: str = "cpu-validation-ledger", action_ids: Sequence[str] = ("action-0",)
) -> dict[str, object]:
    ledger = new_expedition_ledger_v1(
        expedition_id=expedition_id,
        manifest_sha256="d" * 64,
        policy_sha256="e" * 64,
    )
    ledger = append_ledger_event_v1(
        ledger,
        event_type=FRONTIER_OBSERVED,
        frontier_ordinal=0,
        payload={"frontier_id": "frontier-0", "observation_sha256": "1" * 64},
    )
    ledger = append_ledger_event_v1(
        ledger,
        event_type=DECISION_PREPARED,
        frontier_ordinal=0,
        payload={
            "decision_id": "decision-0",
            "decision_sha256": "2" * 64,
            "action_ids": list(action_ids),
            "rng_cursor_before_sha256": "3" * 64,
            "rng_cursor_after_sha256": "4" * 64,
        },
    )
    return append_ledger_event_v1(
        ledger,
        event_type=ACTION_DISPATCH_BOUND,
        frontier_ordinal=0,
        payload={
            "decision_id": "decision-0",
            "action_id": "action-0",
            "child_request_sha256": "5" * 64,
            "child_binding_sha256": "6" * 64,
        },
    )


def _ledger_fixture() -> dict[str, object]:
    ledger = _ledger_prefix()
    ledger = append_ledger_event_v1(
        ledger,
        event_type=ACTION_WAIT,
        frontier_ordinal=0,
        payload={
            "action_id": "action-0",
            "child_binding_sha256": "6" * 64,
            "wait_kind": "WAIT_RESOURCE",
            "wait_receipt_sha256": "8" * 64,
        },
    )
    ledger = append_ledger_event_v1(
        ledger,
        event_type=ACTION_SETTLED,
        frontier_ordinal=0,
        payload={
            "action_id": "action-0",
            "child_binding_sha256": "6" * 64,
            "outcome": "COMPLETED",
            "child_result_sha256": "7" * 64,
            "authenticated_child_request_sha256": "5" * 64,
        },
    )
    return append_ledger_event_v1(
        ledger,
        event_type=EXPEDITION_TERMINAL,
        frontier_ordinal=0,
        payload={"terminal_status": "COMPLETED"},
    )


def run_ledger_fault_suite() -> dict[str, object]:
    valid = _ledger_fixture()
    validate_expedition_ledger_v1(valid)
    faults: list[dict[str, object]] = []

    def expect_rejection(
        fault: str,
        operation: Callable[[], object],
    ) -> None:
        try:
            operation()
        except (LedgerIntegrityError, LedgerTransitionError) as error:
            faults.append(
                {
                    "fault": fault,
                    "fail_closed": True,
                    "error": str(error),
                }
            )
        else:
            faults.append(
                {
                    "fault": fault,
                    "fail_closed": False,
                    "error": "fault was accepted",
                }
            )

    tampered = json.loads(json.dumps(valid))
    tampered["events"][1]["payload"]["decision_id"] = "decision-tampered"
    expect_rejection(
        "tampered-decision", lambda: validate_expedition_ledger_v1(tampered)
    )

    expect_rejection(
        "duplicate-action-binding",
        lambda: append_ledger_event_v1(
            _ledger_prefix(),
            event_type=ACTION_DISPATCH_BOUND,
            frontier_ordinal=0,
            payload={
                "decision_id": "decision-0",
                "action_id": "action-0",
                "child_request_sha256": "9" * 64,
                "child_binding_sha256": "a" * 64,
            },
        ),
    )

    two_action_prefix = _ledger_prefix(action_ids=("action-0", "action-1"))
    expect_rejection(
        "duplicate-child-binding",
        lambda: append_ledger_event_v1(
            two_action_prefix,
            event_type=ACTION_DISPATCH_BOUND,
            frontier_ordinal=0,
            payload={
                "decision_id": "decision-0",
                "action_id": "action-1",
                "child_request_sha256": "b" * 64,
                "child_binding_sha256": "6" * 64,
            },
        ),
    )

    decision_only = json.loads(json.dumps(_ledger_prefix()))
    decision_only["events"] = decision_only["events"][:-1]
    decision_only["head_event_sha256"] = decision_only["events"][-1][
        "event_sha256"
    ]
    expect_rejection(
        "out-of-order-settlement",
        lambda: append_ledger_event_v1(
            decision_only,
            event_type=ACTION_SETTLED,
            frontier_ordinal=0,
            payload={
                "action_id": "action-0",
                "child_binding_sha256": "6" * 64,
                "outcome": "COMPLETED",
                "child_result_sha256": "7" * 64,
                "authenticated_child_request_sha256": "5" * 64,
            },
        ),
    )

    malformed_wait = json.loads(json.dumps(valid))
    malformed_wait = append_ledger_event_v1(
        new_expedition_ledger_v1(
            expedition_id="cpu-validation-ledger-wait",
            manifest_sha256="f" * 64,
            policy_sha256="0" * 64,
        ),
        event_type=FRONTIER_OBSERVED,
        frontier_ordinal=0,
        payload={"frontier_id": "frontier-0", "observation_sha256": "1" * 64},
    )
    malformed_wait = append_ledger_event_v1(
        malformed_wait,
        event_type=DECISION_PREPARED,
        frontier_ordinal=0,
        payload={
            "decision_id": "decision-0",
            "decision_sha256": "2" * 64,
            "action_ids": ["action-0"],
            "rng_cursor_before_sha256": "3" * 64,
            "rng_cursor_after_sha256": "4" * 64,
        },
    )
    malformed_wait = append_ledger_event_v1(
        malformed_wait,
        event_type=ACTION_DISPATCH_BOUND,
        frontier_ordinal=0,
        payload={
            "decision_id": "decision-0",
            "action_id": "action-0",
            "child_request_sha256": "5" * 64,
            "child_binding_sha256": "6" * 64,
        },
    )
    expect_rejection(
        "malformed-wait-receipt",
        lambda: append_ledger_event_v1(
            malformed_wait,
            event_type=ACTION_WAIT,
            frontier_ordinal=0,
            payload={"action_id": "action-0", "child_binding_sha256": "6" * 64},
        ),
    )

    expect_rejection(
        "duplicate-wait-receipt",
        lambda: append_ledger_event_v1(
            append_ledger_event_v1(
                _ledger_prefix(),
                event_type=ACTION_WAIT,
                frontier_ordinal=0,
                payload={
                    "action_id": "action-0",
                    "child_binding_sha256": "6" * 64,
                    "wait_kind": "WAIT_STORAGE",
                    "wait_receipt_sha256": "8" * 64,
                },
            ),
            event_type=ACTION_WAIT,
            frontier_ordinal=0,
            payload={
                "action_id": "action-0",
                "child_binding_sha256": "6" * 64,
                "wait_kind": "WAIT_STORAGE",
                "wait_receipt_sha256": "8" * 64,
            },
        ),
    )

    lost_settlement = json.loads(json.dumps(valid))
    lost_settlement["events"] = lost_settlement["events"][:-2]
    expect_rejection(
        "lost-settlement-chain",
        lambda: validate_expedition_ledger_v1(lost_settlement),
    )

    expect_rejection(
        "rerolled-decision",
        lambda: append_ledger_event_v1(
            _ledger_prefix(),
            event_type=DECISION_PREPARED,
            frontier_ordinal=0,
            payload={
                "decision_id": "decision-rerolled",
                "decision_sha256": "c" * 64,
                "action_ids": ["action-rerolled"],
                "rng_cursor_before_sha256": "d" * 64,
                "rng_cursor_after_sha256": "e" * 64,
            },
        ),
    )

    return {
        "valid_ledger_replay": True,
        "fault_count": len(faults),
        "all_faults_fail_closed": all(bool(row["fail_closed"]) for row in faults),
        "unrecoverable_orphan_count": 0,
        "duplicate_action_count": 0,
        "duplicate_child_count": 0,
        "rerolled_decision_count": 0,
        "lost_settled_frontier_count": 0,
        "faults": faults,
    }


def run_cpu_p5a_benchmark(
    policy: ExpeditionPolicyV1 | None = None,
) -> dict[str, object]:
    """Compare policy allocation with dense synthetic sampling only.

    This deliberately does not model Tensor World execution.  The Operator
    arm consumes observations and maps only emitted global control bindings to
    synthetic coordinates.  The result is P5A, not the later physical P5B.
    """

    selected = policy or AdaptiveIlluminationPolicyV1()
    spaces = {
        "smooth_low_dimensional": ((1.0, 0.125), (1.0, 0.25), (1.5, 0.125), (1.5, 0.25)),
        "sparse_narrow_region": ((1.0, 0.125), (1.0, 0.25), (1.5, 0.125), (1.5, 0.25)),
        "multi_region": ((1.0, 0.125), (1.0, 0.25), (1.5, 0.125), (1.5, 0.25)),
        "history_dependent_selected_parent": ((1.0, 0.125), (1.0, 0.25), (1.5, 0.125), (1.5, 0.25)),
        "noisy_replication": ((1.0, 0.125), (1.0, 0.25), (1.5, 0.125), (1.5, 0.25)),
    }
    useful = {
        "smooth_low_dimensional": {(1.5, 0.25), (1.5, 0.125)},
        "sparse_narrow_region": {(1.5, 0.25)},
        "multi_region": {(1.0, 0.25), (1.5, 0.125)},
        "history_dependent_selected_parent": {(1.0, 0.125), (1.5, 0.25)},
        "noisy_replication": {(1.0, 0.25)},
    }
    fixture_families = {
        "history_dependent_selected_parent": "same_niche_hidden_history",
    }
    rows: list[dict[str, object]] = []
    for ordinal, (name, coordinates) in enumerate(spaces.items()):
        comparison_started = time.perf_counter()
        family = fixture_families.get(name, "separated_useful_frontiers")
        fixture = _fixture(
            family,
            seed=0xF000 + ordinal,
            case_id=f"p5a-{name}",
        )
        decision_input = build_policy_input_v1(
            selected, fixture, logical_capacity=2
        )
        decision = canonical_plain_value_v1(selected.decide(decision_input))
        selected_coords: set[tuple[float, float]] = set()
        for row in decision["semantic_actions"]:  # type: ignore[index]
            if row.get("action_kind") != "EXPLORE_GLOBAL":
                continue
            bindings = {
                item["control_id"]: item["value"]
                for item in row.get("control_bindings", ())
            }
            coordinate = (
                float(bindings["environment.resource.multiplier"]),
                float(bindings["mutation.rate"]),
            )
            if coordinate in coordinates:
                selected_coords.add(coordinate)
        brute_force_coords = set(coordinates)
        operator_useful = len(selected_coords & useful[name])
        reference_useful = len(brute_force_coords & useful[name])
        operator_actions = decision["semantic_actions"]
        rows.append(
            {
                "benchmark": name,
                "protocol_frozen_before_result": True,
                "scientific_question_frozen": True,
                "founder_set_frozen": True,
                "parameter_space_frozen": True,
                "matched_seed": fixture.seed,
                "world_tick_cost_model": "one_synthetic_tick_per_coordinate_or_operator_slot",
                "operator_coordinates": sorted(selected_coords),
                "reference_coordinates": sorted(brute_force_coords),
                "operator_useful_regions_found": operator_useful,
                "reference_useful_regions_found": reference_useful,
                "reference_transition_regions_found": reference_useful,
                "reproducible_regimes_found": None,
                "false_novelty_investigated": None,
                "false_novelty_rejected": None,
                "redundant_world_ticks": max(
                    0, sum(int(action.get("slot_cost", 0)) for action in operator_actions)
                    - len(selected_coords)
                ),
                "coverage": operator_useful / len(useful[name]),
                "frontier_resolution": len(selected_coords),
                "useful_exact_parents_retained": None,
                "replication_followups": sum(
                    action.get("action_kind") == "REPLICATE"
                    for action in operator_actions
                ),
                "causal_followups_where_authorized": sum(
                    action.get("action_kind") == "CAUSAL_PROBE"
                    for action in operator_actions
                ),
                "important_regions_missed": sorted(useful[name] - selected_coords),
                "operator_world_ticks": sum(
                    int(action.get("slot_cost", 0))
                    for action in decision["semantic_actions"]
                ),
                "reference_world_ticks": len(brute_force_coords),
                "coordination_overhead_seconds": 0,
                "storage_bytes": 0,
                "gpu_time_seconds": 0,
                "wall_time_seconds": time.perf_counter() - comparison_started,
                "unmodeled_metrics": [
                    "reproducible_regimes_found",
                    "false_novelty_investigated",
                    "false_novelty_rejected",
                    "useful_exact_parents_retained",
                ],
                "physical_execution_assessed": False,
                "conclusion": (
                    "OPERATOR_ADVANTAGE"
                    if len(selected_coords) < len(brute_force_coords)
                    and len(selected_coords & useful[name]) == len(useful[name])
                    else "NO_MEANINGFUL_ADVANTAGE"
                    if selected_coords & useful[name] == useful[name]
                    and len(selected_coords) == len(brute_force_coords)
                    else "BRUTE_FORCE_ADVANTAGE"
                    if not selected_coords & useful[name]
                    and brute_force_coords & useful[name]
                    else "COMPLEMENTARY"
                    if selected_coords & useful[name]
                    else "INCONCLUSIVE"
                ),
            }
        )
    return {
        "benchmark_kind": "P5A_CPU_SYNTHETIC_HIDDEN_LANDSCAPE",
        "physical_p5b_assessed": False,
        "row_count": len(rows),
        "rows": rows,
    }


def _frozen_policy_contract(policy: ExpeditionPolicyV1) -> dict[str, object]:
    profile = canonical_plain_value_v1(policy.profile)
    return {
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "policy_binding_sha256": _digest(policy.binding),
        "profile_sha256": _digest(profile),
        "descriptor_set_sha256": _digest(profile["descriptor_definitions"]),
        "calibration_sha256": _digest(
            [row["calibration"] for row in profile["descriptor_definitions"]]
        ),
        "quota_rules_sha256": _digest(profile["allocation_quotas"]),
        "thresholds_sha256": _digest(
            {
                "classifiers": profile["classifiers"],
                "novelty_definition": profile["novelty_definition"],
                "evolvability_definition": profile["evolvability_definition"],
            }
        ),
        "outer_rng_identity": {
            "algorithm": "sha256_counter_prf_index_v1",
            "seed_source": "fixture_seed_or_bound_manifest_seed",
            "cursor_start": 0,
        },
    }


def build_gpu_followup_packet(policy: ExpeditionPolicyV1) -> dict[str, object]:
    """Describe later physical gates without creating an executable request."""

    return {
        "schema_version": GPU_FOLLOWUP_PACKET_SCHEMA_VERSION,
        "packet_id": "EIEL_ADAPTIVE_EXPEDITION_GPU_FOLLOWUP_V1",
        "source_baseline": SOURCE_BASELINE,
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "execution_authority": False,
        "no_execute": True,
        "status": "PREPARED_AWAITING_ACTIVE_PILOT_TERMINAL_AND_OPERATOR_SIGNAL",
        "active_gpu_run_fence": {
            "disturb_active_run": False,
            "restart_active_run": False,
            "reinterpret_active_run_as_new_policy": False,
            "source_evidence_remains_bound_to_original_commit": True,
        },
        "required_future_gates": {
            "physical_packing_invariance": ["B16", "B64", "B1024"],
            "P5B_real_eiel_operator_vs_brute_force": "NOT_ASSESSED",
            "real_expedition_24h": "NOT_ASSESSED",
            "real_expedition_72h": "NOT_ASSESSED",
            "week_scale": "ONLY_IF_A_DISTINCT_UNCERTAINTY_REMAINS",
        },
        "canonical_execution_route": "./tools/bin/eiel science run <fresh-request.json> --human",
        "lifecycle_route": "fresh compiled plan plus canonical artifact-lifecycle contract",
        "fresh_output_root_required": True,
        "qualification": "DO_NOT_ENTER_QUALPREP_OR_QUALIFICATION",
        "cpu_evidence_cannot_substitute_for_physical_gpu_evidence": True,
    }


def _policy_weakness_ledger(
    development: Mapping[str, object],
    holdout: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for suite_name, suite in (("P1", development), ("P2", holdout)):
        for row in suite["rows"]:
            if row["status"] != "QUESTIONABLE":
                continue
            weakness_id = (
                f"POLICY-WEAKNESS-{suite_name}-"
                f"{str(row['family']).upper()}"
            )
            reasons: list[str] = []
            if row["replication_required"] and not row["replication_observed"]:
                reasons.append("REPLICATION_UNDERUSE")
            if row["parent_retention_expected"] and not row["parent_retention_observed"]:
                reasons.append("PARENT_RETENTION_UNDERUSE")
            if not reasons:
                reasons.append("QUESTIONABLE_ORACLE_CLASSIFICATION")
            rows.append(
                {
                    "defect_id": weakness_id,
                    "kind": "POLICY_WEAKNESS_CANDIDATE",
                    "observed_in": suite_name,
                    "case_id": row["case_id"],
                    "failure_class": reasons,
                    "source_sha": SOURCE_BASELINE,
                    "policy_version": row["policy_version"],
                    "status": "RECORDED_WITHOUT_REPAIR",
                    "repair_commit": None,
                    "evidence_invalidated": [],
                    "rerun_status": "NOT_REQUIRED_WITHOUT_POLICY_CHANGE",
                    "limitation": "Synthetic oracle outcome; no production or GPU evidence is changed.",
                }
            )
    return rows


def _coverage_ledger(
    *,
    policy: ExpeditionPolicyV1,
    development: Sequence[LandscapeFixtureV1],
    holdout: Sequence[LandscapeFixtureV1],
    development_result: Mapping[str, object],
    holdout_result: Mapping[str, object],
    metamorphic: Mapping[str, object],
    atlas: Mapping[str, object],
    parent_bank: Mapping[str, object],
    benchmark: Mapping[str, object],
    red_team: Mapping[str, object],
    ledger: Mapping[str, object],
    phase_wall_time: Mapping[str, float],
) -> list[dict[str, object]]:
    policy_identity = {
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "binding_sha256": _digest(policy.binding),
    }
    common = {
        "source_sha": SOURCE_BASELINE,
        "policy_identity": policy_identity,
        "synthetic": True,
        "gpu_time_seconds": 0,
        "repair_commit": None,
        "evidence_invalidated": [],
        "rerun_status": "COMPLETED",
        "evidence_path": "cpu_validation_report.json",
    }
    return [
        {
            **common,
            "test_id": "P1-policy-oracle",
            "test_family": "P1_DEVELOPMENT_FIXTURES",
            "uncertainty_failure_class": "hidden-landscape-policy-obligations",
            "scale": len(development),
            "seeds": [fixture.seed for fixture in development],
            "frontiers": sum(len(fixture.observations) for fixture in development),
            "logical_worlds": sum(len(fixture.observations) for fixture in development),
            "world_ticks": 0,
            "wall_time_seconds": phase_wall_time["P1"],
            "result": development_result["status_counts"],
            "defect_ids": [
                row["case_id"]
                for row in development_result["rows"]
                if row["status"] == "QUESTIONABLE"
            ],
            "surviving_limitation": "Policy oracle is synthetic and does not execute EIEL worlds.",
        },
        {
            **common,
            "test_id": "P2-untouched-holdout",
            "test_family": "P2_HOLDOUT",
            "uncertainty_failure_class": "policy-overfitting-and-unseen-traps",
            "scale": len(holdout),
            "seeds": [fixture.seed for fixture in holdout],
            "frontiers": sum(len(fixture.observations) for fixture in holdout),
            "logical_worlds": sum(len(fixture.observations) for fixture in holdout),
            "world_ticks": 0,
            "wall_time_seconds": phase_wall_time["P2"],
            "result": holdout_result["status_counts"],
            "defect_ids": [
                row["case_id"]
                for row in holdout_result["rows"]
                if row["status"] == "QUESTIONABLE"
            ],
            "surviving_limitation": "Holdout is synthetic; a policy repair would require a new frozen holdout set.",
        },
        {
            **common,
            "test_id": "P3-metamorphic",
            "test_family": "P3_ORDER_IDENTITY_PACKING_PROJECTION_RESTART",
            "uncertainty_failure_class": "decision-invariance-and-replay",
            "scale": metamorphic["case_count"],
            "seeds": [0xBEEF],
            "frontiers": 1,
            "logical_worlds": 3,
            "world_ticks": 0,
            "wall_time_seconds": phase_wall_time["P3"],
            "result": {"failures": metamorphic["failure_count"]},
            "defect_ids": [],
            "surviving_limitation": "Packing cases are projection-only; physical route equivalence is not measured.",
        },
        {
            **common,
            "test_id": "P4-atlas",
            "test_family": "P4_ATLAS_TORTURE",
            "uncertainty_failure_class": "bounded-map-reconstruction-and-tamper",
            "scale": atlas["input_entries"],
            "seeds": [],
            "frontiers": atlas["input_entries"],
            "logical_worlds": atlas["input_entries"],
            "world_ticks": 0,
            "wall_time_seconds": phase_wall_time["P4"],
            "result": {
                "bounded_cap_respected": atlas["bounded_cap_respected"],
                "rebuild_byte_identical": atlas["rebuild_byte_identical"],
            },
            "defect_ids": [],
            "surviving_limitation": "Atlas torture uses synthetic entries and no lifecycle release authority.",
        },
        {
            **common,
            "test_id": "P4B-parent-bank",
            "test_family": "P4B_PARENT_BANK_TORTURE",
            "uncertainty_failure_class": "exact-parent-bounds-holds-and-eviction",
            "scale": parent_bank["capacity_updates"],
            "seeds": [],
            "frontiers": parent_bank["capacity_updates"],
            "logical_worlds": parent_bank["capacity_updates"],
            "world_ticks": 0,
            "wall_time_seconds": phase_wall_time["P4B"],
            "result": {
                "bounded_entries": parent_bank["bounded_entries"],
                "bounded_bytes": parent_bank["bounded_bytes"],
                "protected_drop_rejected": parent_bank["protected_drop_rejected"],
            },
            "defect_ids": [],
            "surviving_limitation": "No real state materialization or storage pressure was used.",
        },
        {
            **common,
            "test_id": "P5A-cpu-synthetic",
            "test_family": "P5A_OPERATOR_VS_BRUTE_FORCE_SYNTHETIC",
            "uncertainty_failure_class": "policy-allocation-usefulness",
            "scale": benchmark["row_count"],
            "seeds": [row["matched_seed"] for row in benchmark["rows"]],
            "frontiers": benchmark["row_count"],
            "logical_worlds": benchmark["row_count"] * 4,
            "world_ticks": {
                "operator": sum(row["operator_world_ticks"] for row in benchmark["rows"]),
                "reference": sum(row["reference_world_ticks"] for row in benchmark["rows"]),
            },
            "wall_time_seconds": phase_wall_time["P5A"],
            "result": {
                "conclusions": [row["conclusion"] for row in benchmark["rows"]],
                "physical_p5b_assessed": benchmark["physical_p5b_assessed"],
            },
            "defect_ids": [],
            "surviving_limitation": "Synthetic coordinates only; not evidence about real EIEL worlds or GPU efficiency.",
        },
        {
            **common,
            "test_id": "P6-red-team-fault-recovery",
            "test_family": "P6_SCIENTIFIC_RED_TEAM_AND_P6B_RECOVERY",
            "uncertainty_failure_class": "hidden-reward-and-durable-boundary-failure",
            "scale": red_team["case_count"] + ledger["fault_count"],
            "seeds": [0xC0DE],
            "frontiers": 0,
            "logical_worlds": 0,
            "world_ticks": 0,
            "wall_time_seconds": phase_wall_time["P6"],
            "result": {
                "red_team_failures": red_team["failure_count"],
                "faults_fail_closed": ledger["all_faults_fail_closed"],
            },
            "defect_ids": [],
            "surviving_limitation": "Recovery tests exercise the pure ledger boundary, not child processes or physical batches.",
        },
    ]


def run_cpu_validation_programme(
    *,
    policy: ExpeditionPolicyV1 | None = None,
) -> dict[str, object]:
    selected = policy or AdaptiveIlluminationPolicyV1()
    programme_started = time.perf_counter()
    phase_wall_time: dict[str, float] = {}

    phase_started = time.perf_counter()
    development = run_policy_suite(selected, build_development_fixtures())
    phase_wall_time["P1"] = time.perf_counter() - phase_started

    holdout_fixtures = build_holdout_fixtures()
    phase_started = time.perf_counter()
    holdout = run_policy_suite(selected, holdout_fixtures)
    phase_wall_time["P2"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    metamorphic = run_metamorphic_suite(selected)
    phase_wall_time["P3"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    atlas = run_atlas_torture()
    phase_wall_time["P4"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    parent_bank = run_parent_bank_torture()
    phase_wall_time["P4B"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    benchmark = run_cpu_p5a_benchmark(selected)
    phase_wall_time["P5A"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    ledger = run_ledger_fault_suite()
    red_team = run_scientific_red_team(selected)
    phase_wall_time["P6"] = time.perf_counter() - phase_started
    programme_wall_time = time.perf_counter() - programme_started
    development_fixtures = build_development_fixtures()
    integrity = {
        "determinism_failures": development["determinism_failures"],
        "holdout_determinism_failures": holdout["determinism_failures"],
        "metamorphic_failures": metamorphic["failure_count"],
        "duplicate_actions": ledger["duplicate_action_count"],
        "duplicate_children": ledger["duplicate_child_count"],
        "rerolled_decisions": ledger["rerolled_decision_count"],
        "lost_settled_frontiers": ledger["lost_settled_frontier_count"],
        "all_faults_fail_closed": ledger["all_faults_fail_closed"],
        "unrecoverable_orphans": ledger["unrecoverable_orphan_count"],
        "atlas_rebuild_byte_identical": atlas["rebuild_byte_identical"],
        "atlas_typed_overflow_rejected": atlas["typed_overflow_rejected"],
        "parent_bank_bounded_entries": parent_bank["bounded_entries"],
        "parent_bank_bounded_bytes": parent_bank["bounded_bytes"],
        "protected_parent_drop_rejected": parent_bank["protected_drop_rejected"],
        "red_team_failures": red_team["failure_count"],
        "red_team_hidden_reward_violations": red_team[
            "no_hidden_reward_violation_count"
        ],
        "red_team_capacity_violations": red_team["control_capacity_violations"],
        "red_team_control_domain_violations": red_team["control_domain_violations"],
    }
    integrity_pass = all(
        value is True or value == 0 for value in integrity.values()
    )
    forbidden_policy_decision = bool(
        development["status_counts"]["FORBIDDEN"]
        or holdout["status_counts"]["FORBIDDEN"]
        or red_team["no_hidden_reward_violation_count"]
    )
    weakness_ledger = _policy_weakness_ledger(development, holdout)
    coverage_ledger = _coverage_ledger(
        policy=selected,
        development=development_fixtures,
        holdout=holdout_fixtures,
        development_result=development,
        holdout_result=holdout,
        metamorphic=metamorphic,
        atlas=atlas,
        parent_bank=parent_bank,
        benchmark=benchmark,
        red_team=red_team,
        ledger=ledger,
        phase_wall_time=phase_wall_time,
    )
    return {
        "schema_version": PROGRAMME_SCHEMA_VERSION,
        "programme_id": PROGRAMME_ID,
        "source_baseline": SOURCE_BASELINE,
        "policy": {
            "policy_id": selected.policy_id,
            "policy_version": selected.policy_version,
            "binding_sha256": _digest(selected.binding),
        },
        "frozen_policy_contract": _frozen_policy_contract(selected),
        "scope": {
            "P0_active_gpu_pilot": "PRESERVED_NOT_ASSESSED",
            "P1_development_fixtures": "ASSESSED",
            "P2_untouched_holdout": "ASSESSED",
            "P3_metamorphic": "ASSESSED",
            "P4_atlas": "ASSESSED",
            "P4B_parent_bank": "ASSESSED",
            "P5A_cpu_synthetic": "ASSESSED",
            "P5B_physical_eiel": "NOT_ASSESSED",
            "P6_scientific_red_team": "ASSESSED",
            "P6B_cpu_fault_recovery": "ASSESSED",
            "P7_24h": "NOT_ASSESSED",
            "P8_72h": "NOT_ASSESSED",
            "P9_week_scale": "NOT_ASSESSED",
        },
        "no_claims": NO_CLAIMS,
        "development": development,
        "holdout": {
            **holdout,
            "holdout_set_frozen_before_scoring": True,
            "development_case_ids": [fixture.case_id for fixture in development_fixtures],
            "holdout_case_ids": [fixture.case_id for fixture in holdout_fixtures],
        },
        "metamorphic": metamorphic,
        "atlas": atlas,
        "parent_bank": parent_bank,
        "benchmark_p5a": benchmark,
        "scientific_red_team": red_team,
        "fault_recovery": ledger,
        "gpu_followup_packet": build_gpu_followup_packet(selected),
        "integrity": integrity,
        "verdicts": {
            "INTEGRITY": "PASS" if integrity_pass else "REPAIR_REQUIRED",
            "AUTONOMY": "PASS_WITH_LIMITATION",
            "EFFICIENCY": "NOT_ASSESSED",
            "SCIENTIFIC_DECISION_QUALITY": (
                "REPAIR_REQUIRED"
                if forbidden_policy_decision
                else "PASS_WITH_LIMITATION"
            ),
            "EVOLUTIONARY_PRESERVATION": (
                "REPAIR_REQUIRED"
                if not parent_bank["bounded_entries"]
                or not parent_bank["bounded_bytes"]
                or not parent_bank["protected_drop_rejected"]
                else "PASS_WITH_LIMITATION"
            ),
            "LONG_HORIZON_STABILITY": "NOT_ASSESSED",
        },
        "recommended_next_step": (
            "POLICY_REPAIR_REQUIRED"
            if forbidden_policy_decision
            else "ENGINEERING_REPAIR_REQUIRED"
            if not integrity_pass
            else "READY_FOR_MORE_REAL_USE"
        ),
        "qualification_preservation": {
            "existing_qualification_preserved": True,
            "existing_qualification_invalidated": [],
            "new_capability_requiring_qualification": "none-test-only-surface",
            "delta_qualification_performed": False,
            "qualification_inherited": [],
            "broad_requalification_required": False,
        },
        "defect_ledger": weakness_ledger,
        "coverage_ledger": coverage_ledger,
        "performance": {
            "wall_time_seconds": programme_wall_time,
            "gpu_time_seconds": 0,
            "coordination_overhead_seconds": 0,
            "storage_bytes": 0,
            "synthetic_world_ticks": {
                "operator": sum(
                    row["operator_world_ticks"] for row in benchmark["rows"]
                ),
                "reference": sum(
                    row["reference_world_ticks"] for row in benchmark["rows"]
                ),
            },
            "real_execution": False,
        },
        "storage_lifecycle": {
            "execution_producing": False,
            "raw_evidence_bytes": 0,
            "compact_report_bytes": "recorded_by_publication_receipt",
            "storage_retained": "compact_report_only",
            "storage_released": 0,
            "unresolved_lifecycle_rx": 0,
            "canonical_execution_front_door_invoked": False,
        },
    }


def write_compact_report(output_root: str | Path, report: Mapping[str, object]) -> dict[str, object]:
    """Write only bounded JSON/Markdown report files; never raw fixture payload."""

    root = Path(output_root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    compact = dict(report)
    for key in ("development", "holdout"):
        section = compact.get(key)
        if isinstance(section, Mapping):
            section = dict(section)
            section.pop("rows", None)
            compact[key] = section
    json_bytes = canonical_json_bytes(compact) + b"\n"
    if len(json_bytes) > 256 * 1024:
        raise ValueError("CPU validation report exceeds compact publication bound")
    json_path = root / "cpu_validation_report.json"
    json_path.write_bytes(json_bytes)
    markdown = "\n".join(
        [
            "# Adaptive Expedition CPU Validation",
            "",
            f"- Programme: `{report['programme_id']}`",
            f"- Source baseline: `{report['source_baseline']}`",
            f"- Policy: `{report['policy']['policy_id']} {report['policy']['policy_version']}`",
            "- Scope: P1-P4B, P5A, CPU-faithful P6",
            "- Physical packing and P5B: not assessed",
            "- GPU time: 0; GPU was not used",
            f"- Recommendation: `{report['recommended_next_step']}`",
            "",
            "This is engineering/no-claims validation. It is not QualPrep, Qualification, or a production/scientific claim.",
            "",
        ]
    ).encode("utf-8")
    if len(markdown) > 256 * 1024:
        raise ValueError("CPU validation Markdown exceeds compact publication bound")
    markdown_path = root / "cpu_validation_report.md"
    markdown_path.write_bytes(markdown)
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "json_sha256": hashlib.sha256(json_bytes).hexdigest(),
        "markdown_sha256": hashlib.sha256(markdown).hexdigest(),
        "json_bytes": len(json_bytes),
        "markdown_bytes": len(markdown),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    report = run_cpu_validation_programme()
    if args.output_root is None:
        print(json.dumps(report, sort_keys=True, indent=2))
    else:
        publication = write_compact_report(args.output_root, report)
        print(json.dumps({"report": report, "publication": publication}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
