"""Faithful founder-only child generation through the Science Request front door.

This is the intentionally cheaper G2.5 lane.  It may generate fresh finite
campaigns from a manifest-bound founder request, but it cannot continue or
approximate an evolved trajectory.  Every child starts at absolute tick zero
and enters through ``run_science_request_v1``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any, Final

from emergence_contracts.adaptive_science_expedition_v1 import (
    ChildRealizationKindV1,
    DecisionV1,
    ExpeditionManifestV1,
    ExperimentActionKindV1,
    GeneratedChildContinuationV1,
    StateClassV1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.ids import RunId, SchemaVersion
from emergence_contracts.payloads import CanonicalPayloadV1
from emergence_contracts.science_request_v1 import (
    EIEL_SCIENCE_REQUEST_SCHEMA_VERSION,
    EielScienceRequestV1,
    validate_eiel_science_request_v1,
)

from eiel_composition.science_request_v1_front_door import run_science_request_v1


FOUNDER_ACTION_PAYLOAD_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_science_policy_action_v1"
)
FOUNDER_EXECUTION_PROFILE_SCHEMA_V1: Final[str] = (
    "eiel_adaptive_founder_execution_profile_v1"
)
FOUNDER_SEED_DERIVATION_V1: Final[str] = (
    "sha256_manifest_action_replicate_ordinal_unique_u63_v1"
)
_FOUNDER_ACTION_KINDS: Final[frozenset[ExperimentActionKindV1]] = frozenset(
    {
        ExperimentActionKindV1.EXPLORE_GLOBAL,
        ExperimentActionKindV1.REPLICATE,
        ExperimentActionKindV1.CAUSAL_PROBE,
    }
)
_PLACEHOLDER_RE = re.compile(r"^\$\{[A-Za-z][A-Za-z0-9_]*\}$")


class FounderLoopError(ValueError):
    """A policy action cannot become a faithful founder-basis child."""


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise FounderLoopError(f"{field} must be a canonical mapping")
    assert isinstance(value, dict)
    return deepcopy(value)


def _sequence(value: object, *, field: str) -> list[object]:
    if type(value) not in {list, tuple}:
        raise FounderLoopError(f"{field} must be a canonical sequence")
    return deepcopy(list(value))


def _contains_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return _PLACEHOLDER_RE.fullmatch(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_placeholder(item) for item in value)
    return False


def _founder_template_v1(
    manifest: ExpeditionManifestV1,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
) -> tuple[EielScienceRequestV1, dict[str, object]]:
    request = validate_eiel_science_request_v1(founder_request)
    if request.expect_worlds != 1:
        raise FounderLoopError(
            "founder-basis request must describe exactly one declared founder world"
        )
    if not any(
        ref.artifact_kind == "eiel_science_request_v1"
        and ref.content_digest == request.request_sha256()
        for ref in manifest.founder_basis_refs
    ):
        raise FounderLoopError(
            "founder request identity is absent from the frozen manifest basis"
        )
    expanded = request.expand_worlds()
    if len(expanded) != 1:
        raise FounderLoopError("founder request did not resolve to one world")
    resolved = expanded[0].world_template.to_value()
    template = _mapping(resolved, field="resolved founder world template")
    if _contains_placeholder(template):
        raise FounderLoopError("resolved founder template still contains a placeholder")
    return request, template


def _founder_nominal_batch_worlds_v1(
    founder_request: Mapping[str, object] | EielScienceRequestV1,
) -> int:
    """Derive the request-owned nominal packing without a machine default."""

    try:
        return validate_eiel_science_request_v1(founder_request).batch_plan().worlds
    except (TypeError, ValueError) as error:
        raise FounderLoopError(
            "founder-basis request has no valid nominal batch projection"
        ) from error


def _proposal_action_v1(decision: DecisionV1, *, action_id: str) -> dict[str, object]:
    proposal = decision.policy_proposal.to_value()
    body = _mapping(proposal, field="policy proposal")
    actions = _sequence(body.get("semantic_actions"), field="policy semantic actions")
    selected = [
        _mapping(item, field="policy semantic action")
        for item in actions
        if isinstance(item, Mapping) and item.get("action_id") == action_id
    ]
    if len(selected) != 1:
        raise FounderLoopError("decision does not contain exactly one requested action")
    action = selected[0]
    try:
        kind = ExperimentActionKindV1(action.get("action_kind"))
    except (TypeError, ValueError) as error:
        raise FounderLoopError("policy action kind is unsupported") from error
    if kind not in _FOUNDER_ACTION_KINDS:
        raise FounderLoopError(
            "historical/state-local action cannot enter the founder-only loop"
        )
    if action.get("parent_state_ref_id") is not None:
        raise FounderLoopError(
            "founder-only action must not carry a historical parent StateRef"
        )
    if action.get("child_start_next_tick", 0) != 0:
        raise FounderLoopError("founder-only action must begin at absolute tick zero")
    return action


def _control_bindings_v1(
    action: Mapping[str, object], manifest: ExpeditionManifestV1
) -> list[dict[str, object]]:
    rows = _sequence(action.get("control_bindings"), field="control bindings")
    bindings: list[dict[str, object]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, field=f"control_bindings[{index}]")
        if set(row) != {"control_id", "value"}:
            raise FounderLoopError("control binding fields must be exact")
        control_id = row.get("control_id")
        if type(control_id) is not str or not control_id:
            raise FounderLoopError("control_id must be non-empty text")
        bindings.append(row)
    bindings.sort(key=lambda row: str(row["control_id"]))
    if len(bindings) < 2:
        raise FounderLoopError(
            "founder child needs two declared non-replicate Science Request axes"
        )
    if len({str(row["control_id"]) for row in bindings}) != len(bindings):
        raise FounderLoopError("founder action contains duplicate controls")

    search_space = manifest.search_space.to_value()
    search = _mapping(search_space, field="manifest search space")
    domains: dict[str, list[object]] = {}
    for raw in _sequence(search.get("controls"), field="manifest controls"):
        row = _mapping(raw, field="manifest control")
        control_id = row.get("control_id")
        values = _sequence(row.get("allowed_values"), field="allowed control values")
        if type(control_id) is str:
            domains[control_id] = values
    for row in bindings:
        control_id = str(row["control_id"])
        if control_id not in domains or row["value"] not in domains[control_id]:
            raise FounderLoopError(
                "policy action left the manifest-authorized control domain"
            )
    return bindings


def founder_world_count_for_action_v1(action: Mapping[str, object]) -> int:
    execution = _mapping(
        action.get("founder_execution"), field="founder execution profile"
    )
    if set(execution) != {
        "schema_version",
        "world_count",
        "seed_derivation",
    }:
        raise FounderLoopError("founder execution profile fields must be exact")
    if execution.get("schema_version") != FOUNDER_EXECUTION_PROFILE_SCHEMA_V1:
        raise FounderLoopError("founder execution profile version is unsupported")
    if execution.get("seed_derivation") != FOUNDER_SEED_DERIVATION_V1:
        raise FounderLoopError("founder execution seed derivation is unsupported")
    world_count = execution.get("world_count")
    if type(world_count) is not int or not 1 <= world_count <= 1_000_000:
        raise FounderLoopError("founder execution world_count is invalid")
    return world_count


def build_founder_child_continuation_v1(
    *,
    manifest: ExpeditionManifestV1,
    decision: DecisionV1,
    action_id: str,
    founder_request: Mapping[str, object] | EielScienceRequestV1,
    replicate_seeds: Sequence[int],
    physical_batch_worlds: int | None = None,
) -> GeneratedChildContinuationV1:
    """Generate one strict, packed, tick-zero Science Request child."""

    if decision.expedition_id != manifest.expedition_id:
        raise FounderLoopError("decision and manifest belong to different expeditions")
    if decision.manifest_sha256 != manifest.manifest_sha256():
        raise FounderLoopError("decision does not bind the supplied manifest")
    if physical_batch_worlds is None:
        physical_batch_worlds = _founder_nominal_batch_worlds_v1(founder_request)
    if type(physical_batch_worlds) is not int or physical_batch_worlds < 1:
        raise FounderLoopError("nominal batch size must be a positive integer")
    base, resolved_template = _founder_template_v1(manifest, founder_request)
    action = _proposal_action_v1(decision, action_id=action_id)
    world_count = founder_world_count_for_action_v1(action)
    seeds = list(replicate_seeds)
    if (
        any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise FounderLoopError("replicate seeds must be unique non-negative integers")
    if len(seeds) != world_count:
        raise FounderLoopError(
            "replicate seed count differs from the policy-bound founder world_count"
        )
    bindings = _control_bindings_v1(action, manifest)
    control_dimensions = [
        f"expedition_control_{index:02d}" for index in range(len(bindings))
    ]
    dimensions: dict[str, list[object]] = {
        dimension: [binding["value"]]
        for dimension, binding in zip(control_dimensions, bindings, strict=True)
    }
    dimensions["replicate_seed"] = seeds

    initialization = _mapping(
        resolved_template.get("initialization"), field="founder initialization"
    )
    controls = _mapping(resolved_template.get("controls"), field="founder controls")
    for dimension, binding in zip(control_dimensions, bindings, strict=True):
        controls[str(binding["control_id"])] = "${" + dimension + "}"
    seeds_template = {
        owner: "${replicate_seed}"
        for owner in (
            "ecology",
            "founder",
            "initialization",
            "intervention",
            "mutation",
            "policy",
            "task_environment",
        )
    }
    source = base.to_dict()
    source_reduce = _mapping(source["reduce"], field="founder reduce")
    source_configuration = _mapping(
        source_reduce["configuration"], field="founder reducer configuration"
    )
    source_test = source.get("test")
    if type(source_test) is not str or not source_test:
        raise FounderLoopError("founder Science Request test identity is invalid")
    decision_family = source_configuration.get(
        "decision_family", f"EXPEDITION_{manifest.expedition_id}"
    )
    reporting_group = source_configuration.get(
        "reporting_group", f"EXPEDITION_{manifest.expedition_id}"
    )
    if (
        type(decision_family) is not str
        or not decision_family
        or type(reporting_group) is not str
        or not reporting_group
    ):
        raise FounderLoopError("founder reducer reporting identities are invalid")
    source_observe = _mapping(source["observe"], field="founder observe")
    identity_core = {
        "expedition_id": manifest.expedition_id,
        "manifest_sha256": manifest.manifest_sha256().to_dict(),
        "decision_sha256": decision.decision_sha256().to_dict(),
        "action_id": action_id,
        "replicate_seeds": seeds,
        "physical_batch_worlds": physical_batch_worlds,
    }
    identity = canonical_sha256(identity_core).hex
    campaign_id = f"EIEL-EXP-{identity[:24]}"
    request_value: dict[str, object] = {
        "schema_version": str(EIEL_SCIENCE_REQUEST_SCHEMA_VERSION),
        "campaign_id": campaign_id,
        # Preserve the instrument/test identity declared by the immutable
        # founder Science Request.  The Expedition coordinator does not own a
        # permanent scientific test name.
        "test": source_test,
        "question": {
            "description": manifest.question_description,
            "no_claims": True,
        },
        "world": deepcopy(source["world"]),
        "run": {
            "ticks": manifest.epoch_ticks,
            "batch": physical_batch_worlds,
            "checkpoint_ticks": [manifest.epoch_ticks],
        },
        "dimensions": dimensions,
        "world_template": {
            "initialization": initialization,
            "controls": controls,
            "seeds": seeds_template,
        },
        "replicate_dimension": "replicate_seed",
        "observe": {
            "profile": source_observe["profile"],
            "events": source_observe["events"],
            "state": source_observe["state"],
            "runtime_counters": source_observe["runtime_counters"],
            "snapshot_ticks": [0, manifest.epoch_ticks],
        },
        "reduce": {
            "reducer": source_reduce["reducer"],
            "configuration": {
                "analysis_dimensions": [*control_dimensions, "replicate_seed"],
                "axis_a": control_dimensions[0],
                "axis_b": control_dimensions[1],
                "decision_family": decision_family,
                "reporting_group": reporting_group,
                "tracked_policy_loci": source_configuration["tracked_policy_loci"],
                "tracked_trait_loci": source_configuration["tracked_trait_loci"],
            },
        },
        "expect_worlds": len(seeds),
    }
    child_request = validate_eiel_science_request_v1(request_value)
    payload = CanonicalPayloadV1.from_value(
        SchemaVersion(str(EIEL_SCIENCE_REQUEST_SCHEMA_VERSION)),
        child_request.to_dict(),
    )
    kind = ExperimentActionKindV1(action["action_kind"])
    child = GeneratedChildContinuationV1(
        expedition_id=manifest.expedition_id,
        manifest_sha256=manifest.manifest_sha256(),
        decision_sha256=decision.decision_sha256(),
        action_id=action_id,
        action_kind=kind,
        realization_kind=ChildRealizationKindV1.FOUNDER_BASIS_REQUEST,
        parent_state_ref=None,
        founder_basis_sha256=manifest.founder_basis_sha256(),
        child_trajectory_id=f"trajectory-founder-{identity[:24]}",
        child_run_id=RunId(f"run-founder-{identity[:24]}"),
        retry_of_run_id=None,
        effective_next_absolute_tick=0,
        control_schedule_sha256=canonical_sha256(
            {
                "effective_next_absolute_tick": 0,
                "control_bindings": bindings,
            }
        ),
        child_request=payload,
        child_request_sha256=payload.content_digest(),
    )
    return GeneratedChildContinuationV1.from_dict(child.to_dict())


def run_founder_child_v1(
    continuation: GeneratedChildContinuationV1,
    *,
    repo_root: str | None = None,
    materialization_parent: str | None = None,
    runtime_parent: str | None = None,
    resume_root: str | None = None,
    **front_door_options: Any,
) -> dict[str, Any]:
    """Submit or resume one already sealed founder child through the front door."""

    if continuation.realization_kind is not ChildRealizationKindV1.FOUNDER_BASIS_REQUEST:
        raise FounderLoopError("founder loop refuses an exact-state continuation")
    if continuation.parent_state_ref is not None:
        raise FounderLoopError("founder loop refuses a historical parent StateRef")
    if continuation.effective_next_absolute_tick != 0:
        raise FounderLoopError("founder loop child does not start at tick zero")
    request = validate_eiel_science_request_v1(
        continuation.child_request.to_value()
    )
    return run_science_request_v1(
        request.to_dict(),
        repo_root=repo_root,
        materialization_parent=materialization_parent,
        runtime_parent=runtime_parent,
        resume_root=resume_root,
        **front_door_options,
    )


def run_exact_state_child_v1(
    continuation: GeneratedChildContinuationV1,
    *,
    repo_root: str | None = None,
    materialization_parent: str | None = None,
    runtime_parent: str | None = None,
    resume_root: str | None = None,
    **front_door_options: Any,
) -> dict[str, Any]:
    """Submit or resume one sealed exact-state child through the front door."""

    parent = continuation.parent_state_ref
    if (
        continuation.realization_kind
        is not ChildRealizationKindV1.EXACT_STATE_REQUEST
        or parent is None
        or parent.state_class is not StateClassV1.BRANCHABLE_EXACT
        or continuation.effective_next_absolute_tick != parent.next_absolute_tick
        or continuation.retry_of_run_id is not None
    ):
        raise FounderLoopError(
            "exact child lacks a direct selected-world continuation"
        )
    request = validate_eiel_science_request_v1(
        continuation.child_request.to_value()
    )
    request_continuation = request.to_dict().get("continuation")
    if (
        not isinstance(request_continuation, Mapping)
        or request_continuation.get("state_ref") != parent.to_dict()
    ):
        raise FounderLoopError(
            "exact child request differs from its sealed parent StateRef"
        )
    return run_science_request_v1(
        request.to_dict(),
        repo_root=repo_root,
        materialization_parent=materialization_parent,
        runtime_parent=runtime_parent,
        resume_root=resume_root,
        **front_door_options,
    )


__all__ = (
    "FOUNDER_ACTION_PAYLOAD_SCHEMA_V1",
    "FOUNDER_EXECUTION_PROFILE_SCHEMA_V1",
    "FOUNDER_SEED_DERIVATION_V1",
    "FounderLoopError",
    "build_founder_child_continuation_v1",
    "founder_world_count_for_action_v1",
    "run_exact_state_child_v1",
    "run_founder_child_v1",
)
