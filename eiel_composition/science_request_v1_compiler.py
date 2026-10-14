"""Route-neutral compiler for the compact EIEL Science Request V1.

The compact request is an authoring contract.  This module resolves its
factorial rows into the existing immutable ``EIEL_SCIENCE_WORLD_BUNDLE_V1``
execution truth and records deterministic *nominal* batch ownership.  It does
not select a provider, admit a route, create an ExecutionRequest, or grant
science/Qualification authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Final

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ContractError,
    SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_GENOTYPE_IDENTITY_MODE_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1,
    SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1,
    SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
    SchemaVersion,
    ScienceWorldBundleV1,
    ScienceWorldRecordV1,
    canonical_json_bytes,
    canonical_sha256,
    validate_hereditary_attribution_configuration_v1,
)
from emergence_contracts.adaptive_science_expedition_v1 import StateRefV1
from emergence_contracts.artifacts import RunArtifactRefV1
from emergence_contracts.science_request_v1 import (
    EielScienceRequestV1,
    ExpandedScienceWorldV1,
    ScienceRequestBatchPlanV1,
    expand_eiel_science_request_v1,
    validate_eiel_science_request_v1,
)
from eiel_composition.science_cuda_v2_generic_host_context_v1 import (
    generic_cuda_v2_science_control_descriptors_v1,
)
from eiel_composition.science_instrument import science_control_catalog_sha256
from emergence_evidence.science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_REDUCER_ID,
    compact_world_atlas_reducer_descriptor_v1,
)
from engine.perf.cuda_full_evolution_v2.science_native_world_runtime_v1 import (
    NATIVE_FOUNDER_REGISTRY_RELATIVE_PATH_V1,
    ScienceNativeWorldRuntimeError,
    founder_profile_policy_bindings_v1,
)
from .exact_natural_source_state_v1 import (
    EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1,
    load_and_authenticate_source_state_bank_v1,
    materialize_exact_natural_source_initialization_v1,
)


SCIENCE_REQUEST_V1_COMPILER_ID: Final[str] = "eiel_science_request_v1_compiler_v1"
SCIENCE_REQUEST_V1_COMPILATION_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_science_request_v1_compilation_receipt_v1"
)
SCIENCE_REQUEST_V1_NOMINAL_BATCH_PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_science_request_v1_nominal_batch_plan_v1"
)
_SCIENCE_OWNERSHIP_SCHEMA = SchemaVersion("eiel_science_ownership_v1")
_RESOLVED_INPUTS_SCHEMA = SchemaVersion("eiel_science_resolved_inputs_v1")
_ACQUISITION_SCHEMA = SchemaVersion("eiel_science_acquisition_contract_v1")
_REDUCERS_SCHEMA = SchemaVersion("eiel_science_world_reducers_v1")
_SOURCE_SPECS_SCHEMA = SchemaVersion("eiel_science_compact_source_specs_v1")
_COMPILATION_RECEIPT_SCHEMA = SchemaVersion(
    SCIENCE_REQUEST_V1_COMPILATION_RECEIPT_SCHEMA_VERSION
)
_NORMALIZED_REQUEST_SCHEMA = SchemaVersion("eiel_science_request_v1")
_TEST_ID = re.compile(r"^(.+)_V([1-9][0-9]*)$")
_SEED_OWNERS: Final[frozenset[str]] = frozenset(
    {
        "ecology",
        "founder",
        "policy",
        "mutation",
        "task_environment",
        "initialization",
        "intervention",
    }
)
_WORLD_PROFILE_IDS: Final[dict[tuple[int, int, int, int], str]] = {
    (32, 32, 256, 512): "eiel-w32-native-science-v1",
}


class ScienceRequestV1CompilerError(ValueError):
    """A normalized compact request cannot become exact semantic worlds."""


@dataclass(frozen=True, slots=True)
class ScienceRequestV1CompilationV1:
    """One content-bound bundle plus its non-authoritative batch projection."""

    _normalized_request: CanonicalPayloadV1
    world_bundle: ScienceWorldBundleV1
    _compilation_receipt: CanonicalPayloadV1

    @property
    def normalized_request(self) -> dict[str, Any]:
        value = self._normalized_request.to_value()
        assert isinstance(value, dict)
        return value

    @property
    def compilation_receipt(self) -> dict[str, Any]:
        value = self._compilation_receipt.to_value()
        assert isinstance(value, dict)
        return value

    @property
    def nominal_batches(self) -> tuple[dict[str, Any], ...]:
        plan = self.compilation_receipt["nominal_batch_plan"]
        return tuple(deepcopy(row) for row in plan["batches"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_request": deepcopy(self.normalized_request),
            "world_bundle": self.world_bundle.to_dict(),
            "compilation_receipt": deepcopy(self.compilation_receipt),
        }


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScienceRequestV1CompilerError(f"{field} must be an object")
    return dict(value)


def _list(value: object, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ScienceRequestV1CompilerError(f"{field} must be an array")
    return list(value)


def _normalized_request(value: object) -> EielScienceRequestV1:
    """Cross the authoring boundary through the canonical V1 validator."""

    try:
        return validate_eiel_science_request_v1(value)
    except ContractError as error:
        raise ScienceRequestV1CompilerError(
            f"compact science request is invalid: {error}"
        ) from error


def _resolved_expansion(
    request: EielScienceRequestV1,
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    """Return canonical typed bindings and fully substituted templates."""

    try:
        expanded = expand_eiel_science_request_v1(request)
    except ContractError as error:
        raise ScienceRequestV1CompilerError(
            f"compact science request expansion failed: {error}"
        ) from error
    if len(expanded) != request.expect_worlds:
        raise ScienceRequestV1CompilerError(
            "contract expansion differs from expect_worlds"
        )
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for expected_ordinal, row in enumerate(expanded):
        if not isinstance(row, ExpandedScienceWorldV1):
            raise ScienceRequestV1CompilerError(
                "contract expansion returned an unsupported world row"
            )
        if row.world_ordinal != expected_ordinal:
            raise ScienceRequestV1CompilerError(
                "contract expansion ordinals are not contiguous"
            )
        dimensions = _mapping(
            row.dimension_values.to_value(),
            field=f"expanded_worlds[{expected_ordinal}].dimension_values",
        )
        template = _mapping(
            row.world_template.to_value(),
            field=f"expanded_worlds[{expected_ordinal}].world_template",
        )
        result.append((dimensions, template))
    return tuple(result)


def _test_identity(value: object) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ScienceRequestV1CompilerError("test must be a versioned string")
    match = _TEST_ID.fullmatch(value)
    if match is None or not match.group(1):
        raise ScienceRequestV1CompilerError("test must end in _V<positive integer>")
    return match.group(1), match.group(2)


def _world_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    width = raw.get("width")
    height = raw.get("height")
    population = raw.get("population")
    capacity = raw.get("capacity")
    key = (width, height, population, capacity)
    if any(type(item) is not int or item <= 0 for item in key):
        raise ScienceRequestV1CompilerError(
            "world width, height, population, and capacity must be positive integers"
        )
    profile_id = _WORLD_PROFILE_IDS.get(key)
    if profile_id is None:
        profile_id = (
            f"eiel-science-world-shape-w{width}x{height}-p{population}-c{capacity}-v1"
        )
    return {
        "world_profile_id": profile_id,
        "overlay_profile_ids": [],
        "geometry": {"width": width, "height": height},
        "resident_capacity": capacity,
    }


def _lower_schedule(
    *,
    control_id: str,
    target: object,
    schedule: Mapping[str, Any],
    total_ticks: int,
    start_next_tick: int = 0,
) -> tuple[object, dict[str, Any]]:
    kind = schedule.get("kind")
    if kind == "static" and set(schedule) == {"kind"}:
        return target, {"kind": "static"}
    if kind != "step_absolute_tick" or set(schedule) != {
        "kind",
        "onset_tick",
        "baseline",
    }:
        raise ScienceRequestV1CompilerError(
            f"control {control_id!r} uses an unsupported compact schedule"
        )
    onset = schedule["onset_tick"]
    baseline = schedule["baseline"]
    if type(onset) is not int or not start_next_tick <= onset <= total_ticks:
        raise ScienceRequestV1CompilerError(
            f"control {control_id!r} step onset is outside the tick horizon"
        )
    if canonical_json_bytes(target) == canonical_json_bytes(baseline):
        return target, {"kind": "static"}
    if onset == start_next_tick:
        return target, {
            "kind": "piecewise_constant_absolute_ticks",
            "segments": [
                {"start_absolute_tick": start_next_tick, "value": target}
            ],
        }
    return baseline, {
        "kind": "piecewise_constant_absolute_ticks",
        "segments": [
            {"start_absolute_tick": start_next_tick, "value": baseline},
            {"start_absolute_tick": onset, "value": target},
        ],
    }


def _control_bindings(
    raw: Mapping[str, Any],
    *,
    total_ticks: int,
    descriptors: Mapping[str, Any],
    start_next_tick: int = 0,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for control_id in sorted(raw):
        descriptor = descriptors.get(control_id)
        if descriptor is None:
            raise ScienceRequestV1CompilerError(
                f"control {control_id!r} is absent from the canonical native catalog"
            )
        expanded = raw[control_id]
        if isinstance(expanded, Mapping):
            binding = dict(expanded)
            if set(binding) != {"value", "schedule"}:
                raise ScienceRequestV1CompilerError(
                    f"structured control {control_id!r} must contain value and schedule"
                )
            target = binding["value"]
            schedule = _mapping(
                binding["schedule"], field=f"controls.{control_id}.schedule"
            )
        else:
            target = expanded
            schedule = {"kind": "static"}
        resolved, lowered = _lower_schedule(
            control_id=control_id,
            target=target,
            schedule=schedule,
            total_ticks=total_ticks,
            start_next_tick=start_next_tick,
        )
        if lowered["kind"] not in descriptor.schedule_kinds:
            raise ScienceRequestV1CompilerError(
                f"control {control_id!r} lowers to an unsupported canonical schedule"
            )
        value_contract = descriptor.value_contract.to_value()
        if not isinstance(value_contract, Mapping):
            raise ScienceRequestV1CompilerError(
                f"control {control_id!r} has an invalid value descriptor"
            )

        def validate_value(value: object) -> None:
            kind = value_contract.get("kind")
            if kind == "boolean":
                valid = type(value) is bool
            elif kind == "numeric":
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and (not isinstance(value, float) or math.isfinite(value))
                )
            else:
                raise ScienceRequestV1CompilerError(
                    f"control {control_id!r} has an unsupported value contract"
                )
            if not valid:
                raise ScienceRequestV1CompilerError(
                    f"control {control_id!r} differs from its canonical value contract"
                )

        validate_value(resolved)
        for segment in lowered.get("segments", ()):
            validate_value(segment["value"])
        result.append(
            {
                "control_id": control_id,
                "resolved_value": resolved,
                "schedule": lowered,
                "descriptor_sha256": descriptor.descriptor_sha256().to_dict(),
            }
        )
    return result


def _seed_bindings(raw: Mapping[str, Any]) -> dict[str, int]:
    if set(raw) != _SEED_OWNERS:
        missing = sorted(_SEED_OWNERS - set(raw))
        extra = sorted(set(raw) - _SEED_OWNERS)
        raise ScienceRequestV1CompilerError(
            f"world_template.seeds must contain exactly seven owners; "
            f"missing={missing}, extra={extra}"
        )
    expanded = {owner: raw[owner] for owner in sorted(raw)}
    if any(
        type(seed) is not int or not 0 <= seed < (1 << 63) for seed in expanded.values()
    ):
        raise ScienceRequestV1CompilerError(
            "resolved seeds must be nonnegative signed-int64 integers"
        )
    return expanded


def _atlas_configuration(
    value: object, *, dimension_names: tuple[str, ...]
) -> dict[str, Any]:
    configuration = _mapping(value, field="reduce.configuration")
    analysis = configuration.get("analysis_dimensions")
    if (
        not isinstance(analysis, list)
        or len(analysis) != len(set(analysis))
        or set(analysis) != set(dimension_names)
    ):
        raise ScienceRequestV1CompilerError(
            "reducer analysis_dimensions must be an exact dimension permutation"
        )
    for axis in ("axis_a", "axis_b"):
        if configuration.get(axis) not in dimension_names:
            raise ScienceRequestV1CompilerError(
                f"reducer {axis} must name a request dimension"
            )
    for field, limit in (("tracked_policy_loci", 112), ("tracked_trait_loci", 16)):
        values = configuration.get(field, [])
        if (
            not isinstance(values, list)
            or any(type(item) is not int or not 0 <= item < limit for item in values)
            or values != sorted(set(values))
        ):
            raise ScienceRequestV1CompilerError(
                f"reducer {field} must be sorted, unique, and in range"
            )
    return configuration


def _strategy_attribution_configuration(
    configuration: Mapping[str, Any],
    *,
    founder_policy_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the requested reproductive attribution to the native carrier."""

    direct = configuration.get(
        "require_direct_genotype_reproductive_attribution", False
    )
    funnel = configuration.get("require_reproduction_funnel_denominators", False)
    if type(direct) is not bool or type(funnel) is not bool:
        raise ScienceRequestV1CompilerError(
            "reproductive attribution requirements must be exact booleans"
        )
    result = deepcopy(dict(configuration))
    if not (direct or funnel):
        return result
    if not founder_policy_bindings:
        raise ScienceRequestV1CompilerError(
            "strategy attribution requirements require founder policy bindings"
        )
    mapping = dict(sorted(founder_policy_bindings.items()))
    strategy_ids = sorted(set(mapping.values()))
    if not strategy_ids or len(strategy_ids) > 64:
        raise ScienceRequestV1CompilerError(
            "direct reproductive attribution strategy identity set is invalid"
        )
    strategy_binding: dict[str, Any] = {
        "schema_version": SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1,
        "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
        "identity_mode": "declared_policy_mapping_v1",
        "lineage_mode": "parent_ancestry_v1",
        "strategy_ids": strategy_ids,
        "policy_id_to_strategy_id": mapping,
        "denominator_fields": list(SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1),
    }
    if direct:
        strategy_binding.update(
            {
                "direct_genotype_attribution": True,
                "genotype_identity_mode": SCIENCE_GENOTYPE_IDENTITY_MODE_V1,
                "genotype_denominator_fields": list(
                    SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
                ),
            }
        )
    result["strategy_attribution"] = strategy_binding
    return result


def _exact_natural_source_policy_bindings(
    initializations: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Bind exact-source policy bytes to content identities.

    Exact natural states are intentionally not registered founder profiles.
    When a request explicitly asks for the generic reproductive-attribution
    contract, their complete policy payload still provides an unambiguous
    policy identity.  Using that identity as its own strategy ID keeps this
    route neutral and avoids inventing a registry/profile alias or selecting
    from a tracked-locus projection.
    """

    bindings: dict[str, str] = {}
    for initialization in initializations:
        states = initialization.get("exact_natural_source_states", [])
        if states is None:
            continue
        if not isinstance(states, list):
            raise ScienceRequestV1CompilerError(
                "exact natural source states must be a list"
            )
        for index, state in enumerate(states):
            if not isinstance(state, Mapping):
                raise ScienceRequestV1CompilerError(
                    f"exact natural source state {index} must be an object"
                )
            genome = state.get("genome")
            if not isinstance(genome, Mapping):
                raise ScienceRequestV1CompilerError(
                    "exact natural source state genome is missing"
                )
            policy_hex = genome.get("policy_hex")
            if (
                not isinstance(policy_hex, str)
                or len(policy_hex) != 448
                or any(character not in "0123456789abcdef" for character in policy_hex)
            ):
                raise ScienceRequestV1CompilerError(
                    "exact natural source policy payload is not canonical"
                )
            policy_id = "sha256:" + hashlib.sha256(
                bytes.fromhex(policy_hex)
            ).hexdigest()
            prior = bindings.get(policy_id)
            if prior is not None and prior != policy_id:
                raise ScienceRequestV1CompilerError(
                    "exact natural source policy identity is ambiguous"
                )
            bindings[policy_id] = policy_id
    return dict(sorted(bindings.items()))


def _hereditary_attribution_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind full hereditary identity and focal frequency observations."""

    full = configuration.get("require_full_hereditary_attribution", False)
    frequency = configuration.get("require_genotype_frequency_trajectory", False)
    if type(full) is not bool or type(frequency) is not bool:
        raise ScienceRequestV1CompilerError(
            "hereditary attribution requirements must be exact booleans"
        )
    result = deepcopy(dict(configuration))
    if not (full or frequency):
        return result
    focal_ids = configuration.get("focal_tracked_genotype_state_ids")
    if (
        not isinstance(focal_ids, list)
        or not focal_ids
        or focal_ids != sorted(set(focal_ids))
        or any(
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
            for value in focal_ids
        )
    ):
        raise ScienceRequestV1CompilerError(
            "full hereditary attribution requires sorted focal tracked identities"
        )
    binding = {
        "schema_version": SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1,
        "event_kinds": list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1),
        "identity_mode": SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1,
        "tracked_identity_mode": SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1,
        "lineage_mode": "parent_ancestry_v1",
        "focal_tracked_genotype_state_ids": list(focal_ids),
        "denominator_fields": list(SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1),
        "frequency_trajectory": frequency,
    }
    result["hereditary_attribution"] = binding
    return result


def _nominal_batch_plan(
    worlds: tuple[ScienceWorldRecordV1, ...],
    *,
    request_plan: ScienceRequestBatchPlanV1,
) -> dict[str, Any]:
    if len(worlds) != request_plan.logical_world_count:
        raise ScienceRequestV1CompilerError(
            "request batch plan does not cover the compiled world set"
        )
    batches = []
    offset = 0
    for index, size in enumerate(request_plan.batch_sizes):
        selected = worlds[offset : offset + size]
        if len(selected) != size:
            raise ScienceRequestV1CompilerError(
                "request batch plan overran the compiled world set"
            )
        batches.append(
            {
                "batch_index": index,
                "batch_id": f"science-request-batch-{index:04d}",
                "world_count": len(selected),
                "world_ids": [world.world_id for world in selected],
            }
        )
        offset += size
    if offset != len(worlds):
        raise ScienceRequestV1CompilerError(
            "request batch plan did not consume the compiled world set"
        )
    core = {
        "schema_version": SCIENCE_REQUEST_V1_NOMINAL_BATCH_PLAN_SCHEMA_VERSION,
        "policy": request_plan.policy,
        "requested_worlds_per_batch": request_plan.worlds,
        "logical_world_count": len(worlds),
        "nominal_batch_count": len(batches),
        "batch_sizes": list(request_plan.batch_sizes),
        "nominal_only": True,
        "execution_capability_claimed": False,
        "batches": batches,
    }
    return {**core, "plan_sha256": canonical_sha256(core).to_dict()}


def _compiler_component_sha256() -> ContentDigest:
    return ContentDigest(
        "sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )


def compile_science_request_v1(request: object) -> ScienceRequestV1CompilationV1:
    """Compile a normalized compact request into explicit semantic worlds.

    The returned nominal batches are deterministic ownership projections only.
    In particular, a structurally recognized W128x80 request acquires no
    executable Workhorse observation/capture claim here.
    """

    normalized_request = _normalized_request(request)
    payload = normalized_request.to_dict()
    request_sha256 = normalized_request.request_sha256()
    expanded_worlds = _resolved_expansion(normalized_request)
    dimensions = _mapping(payload.get("dimensions"), field="dimensions")
    dimension_names = normalized_request.analysis_dimensions
    replicate_dimension = payload.get("replicate_dimension")
    if replicate_dimension not in dimensions:
        raise ScienceRequestV1CompilerError(
            "replicate_dimension must name a request dimension"
        )
    test_id, test_version = _test_identity(payload.get("test"))
    campaign_id = payload.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise ScienceRequestV1CompilerError("campaign_id must be non-empty text")

    raw_world = _mapping(payload.get("world"), field="world")
    world_payload = _world_payload(raw_world)
    continuation_raw = payload.get("continuation")
    continuation_state_ref: StateRefV1 | None = None
    continuation_authority_ref: RunArtifactRefV1 | None = None
    if continuation_raw is not None:
        continuation = _mapping(continuation_raw, field="continuation")
        try:
            continuation_state_ref = StateRefV1.from_dict(
                continuation["state_ref"]
            )
            continuation_authority_ref = RunArtifactRefV1.from_dict(
                continuation["authority_receipt_ref"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ScienceRequestV1CompilerError(
                f"selected-world continuation identity is invalid: {error}"
            ) from error
    raw_run = _mapping(payload.get("run"), field="run")
    target_next_tick = raw_run.get("ticks")
    if type(target_next_tick) is not int or target_next_tick <= 0:
        raise ScienceRequestV1CompilerError("run.ticks must be positive")
    start_next_tick = (
        0
        if continuation_state_ref is None
        else continuation_state_ref.next_absolute_tick
    )
    total_ticks = target_next_tick - start_next_tick
    if total_ticks <= 0:
        raise ScienceRequestV1CompilerError(
            "selected-world continuation target must follow its parent frontier"
        )
    checkpoint_ticks = _list(
        raw_run.get("checkpoint_ticks"), field="run.checkpoint_ticks"
    )
    if (
        any(
            type(tick) is not int
            or not start_next_tick < tick <= target_next_tick
            for tick in checkpoint_ticks
        )
        or checkpoint_ticks != sorted(set(checkpoint_ticks))
        or not checkpoint_ticks
        or checkpoint_ticks[-1] != target_next_tick
    ):
        raise ScienceRequestV1CompilerError(
            "checkpoint_ticks must be unique ascending absolute ticks ending at run.ticks"
        )
    time_payload = {
        "authority": "absolute_tick_v1",
        "start_next_tick": start_next_tick,
        "total_ticks": total_ticks,
        "checkpoint_ticks": checkpoint_ticks,
        "deliberate_restore_ticks": (
            [] if continuation_state_ref is None else [start_next_tick]
        ),
    }

    template = _mapping(payload.get("world_template"), field="world_template")
    template_controls = _mapping(
        template.get("controls"), field="world_template.controls"
    )
    control_descriptors = generic_cuda_v2_science_control_descriptors_v1()
    descriptors_by_id = {
        descriptor.control_id: descriptor for descriptor in control_descriptors
    }
    reducer = _mapping(payload.get("reduce"), field="reduce")
    if reducer.get("reducer") != COMPACT_WORLD_ATLAS_REDUCER_ID:
        raise ScienceRequestV1CompilerError(
            "reduce.reducer is absent from the compact World Atlas registry"
        )
    reducer_descriptor = compact_world_atlas_reducer_descriptor_v1()
    base_reducer_configuration = _atlas_configuration(
        reducer.get("configuration"), dimension_names=dimension_names
    )

    observe = _mapping(payload.get("observe"), field="observe")
    snapshot_ticks = _list(
        observe.get("snapshot_ticks"), field="observe.snapshot_ticks"
    )
    if (
        any(
            type(tick) is not int
            or not start_next_tick <= tick <= target_next_tick
            for tick in snapshot_ticks
        )
        or snapshot_ticks != sorted(set(snapshot_ticks))
        or start_next_tick not in snapshot_ticks
        or target_next_tick not in snapshot_ticks
    ):
        raise ScienceRequestV1CompilerError(
            "snapshot_ticks must be unique ascending absolute ticks covering the run frontier"
        )
    acquisition = {
        "profile_id": observe.get("profile"),
        "event_observations": sorted(
            _list(observe.get("events"), field="observe.events")
        ),
        "state_observations": sorted(
            _list(observe.get("state"), field="observe.state")
        ),
        "runtime_counters": sorted(
            _list(
                observe.get("runtime_counters"),
                field="observe.runtime_counters",
            )
        ),
        "snapshot_ticks": snapshot_ticks,
        "windows": [],
    }

    # Founder, selected-world, and exact natural source initialization are
    # disjoint authority paths. Exact source states are authenticated before
    # the founder registry projection so no proxy profile can be substituted.
    source_bank_binding: dict[str, Any] | None = None
    source_bank: dict[str, Any] | None = None
    source_bank_pointer = payload.get("exact_natural_source_state_bank")
    resolved_initializations: list[dict[str, Any]] = []
    if continuation_state_ref is None:
        for index, (_assignment, resolved_template) in enumerate(expanded_worlds):
            initialization = _mapping(
                resolved_template.get("initialization"),
                field=f"expanded_worlds[{index}].world_template.initialization",
            )
            if "exact_natural_source_composition" in initialization:
                if not isinstance(source_bank_pointer, Mapping):
                    raise ScienceRequestV1CompilerError(
                        "exact natural source composition requires a bound source-state bank"
                    )
                if source_bank is None or source_bank_binding is None:
                    try:
                        source_bank, source_bank_binding = (
                            load_and_authenticate_source_state_bank_v1(
                                source_bank_pointer
                            )
                        )
                    except (TypeError, ValueError) as error:
                        raise ScienceRequestV1CompilerError(
                            f"exact natural source-state bank is not authenticated: {error}"
                        ) from error
                try:
                    initialization = materialize_exact_natural_source_initialization_v1(
                        initialization,
                        bank=source_bank,
                        bank_binding=source_bank_binding,
                    )
                except (TypeError, ValueError) as error:
                    raise ScienceRequestV1CompilerError(
                        f"exact natural source initialization cannot be materialized: {error}"
                    ) from error
            resolved_initializations.append(initialization)
        founder_initializations = resolved_initializations
    else:
        assert continuation_authority_ref is not None
        selected_initialization = {
            "schema_version": "eiel_science_selected_world_initialization_v1",
            "state_ref": continuation_state_ref.to_dict(),
            "authority_receipt_ref": continuation_authority_ref.to_dict(),
            "expected_initial_population": raw_world["population"],
        }
        resolved_initializations = [
            deepcopy(selected_initialization) for _row in expanded_worlds
        ]
        founder_initializations = []
    founder_id_set: set[str] = set()
    exact_profile_ids: set[str] = set()
    for initialization in founder_initializations:
        declared = _list(
            initialization.get("founder_profiles"),
            field="initialization.founder_profiles",
        )
        composition = _list(
            initialization.get("exact_founder_composition"),
            field="initialization.exact_founder_composition",
        )
        candidates = [*declared]
        for index, row in enumerate(composition):
            if not isinstance(row, Mapping):
                raise ScienceRequestV1CompilerError(
                    f"initialization.exact_founder_composition[{index}] must be an object"
                )
            candidates.append(row.get("founder_profile_id"))
        if any(not isinstance(item, str) or not item for item in candidates):
            raise ScienceRequestV1CompilerError(
                "resolved founder profile IDs must be non-empty strings"
            )
        founder_id_set.update(candidates)
        exact_states = initialization.get("exact_natural_source_states")
        if isinstance(exact_states, list):
            exact_profile_ids.update(
                row.get("materialization", {}).get("materialization_profile_id")
                for row in exact_states
                if isinstance(row, Mapping)
                and isinstance(row.get("materialization"), Mapping)
                and isinstance(
                    row.get("materialization", {}).get("materialization_profile_id"),
                    str,
                )
            )
    founder_ids = tuple(sorted(founder_id_set))
    registered_founder_ids = tuple(
        item for item in founder_ids if item not in exact_profile_ids
    )
    repository_root = Path(__file__).resolve().parents[1]
    founder_policy_bindings: dict[str, Any] = {}
    founder_registry_sha256: ContentDigest | None = None
    if continuation_state_ref is None:
        try:
            founder_policy_bindings = founder_profile_policy_bindings_v1(
                registered_founder_ids, repository_root=repository_root
            )
        except ScienceNativeWorldRuntimeError as error:
            raise ScienceRequestV1CompilerError(
                f"founder profiles are not canonically registered: {error}"
            ) from error
        founder_registry_path = (
            repository_root / NATIVE_FOUNDER_REGISTRY_RELATIVE_PATH_V1
        ).resolve()
        founder_registry_sha256 = ContentDigest(
            "sha256", hashlib.sha256(founder_registry_path.read_bytes()).hexdigest()
        )
    exact_source_policy_bindings = _exact_natural_source_policy_bindings(
        resolved_initializations
    )
    combined_policy_bindings = dict(founder_policy_bindings)
    for policy_id, strategy_id in exact_source_policy_bindings.items():
        prior = combined_policy_bindings.get(policy_id)
        if prior is not None and prior != strategy_id:
            raise ScienceRequestV1CompilerError(
                "registered and exact natural source policy bindings conflict"
            )
        combined_policy_bindings[policy_id] = strategy_id
    combined_policy_bindings = dict(sorted(combined_policy_bindings.items()))
    base_reducer_configuration = _strategy_attribution_configuration(
        base_reducer_configuration,
        founder_policy_bindings=combined_policy_bindings,
    )
    base_reducer_configuration = _hereditary_attribution_configuration(
        base_reducer_configuration
    )

    worlds: list[ScienceWorldRecordV1] = []
    for (assignment, resolved_template), initialization in zip(
        expanded_worlds, resolved_initializations, strict=True
    ):
        if continuation_state_ref is None and (
            initialization.get("initial_population") != raw_world.get("population")
        ):
            raise ScienceRequestV1CompilerError(
                "resolved initialization population differs from world.population"
            )
        initial_live_count = initialization.get("initial_live_count")
        if continuation_state_ref is not None:
            initial_live_count = None
        if continuation_state_ref is None and (
            type(initial_live_count) is not int or initial_live_count < 1
        ):
            raise ScienceRequestV1CompilerError(
                "resolved initialization live count is invalid"
            )
        composition = (
            []
            if continuation_state_ref is not None
            else _list(
                initialization.get("exact_founder_composition"),
                field="initialization.exact_founder_composition",
            )
        )
        declared_founders = (
            []
            if continuation_state_ref is not None
            else _list(
                initialization.get("founder_profiles"),
                field="initialization.founder_profiles",
            )
        )
        composition_founders: list[str] = []
        composition_count = 0
        for index, row in enumerate(composition):
            if not isinstance(row, Mapping) or set(row) != {
                "founder_profile_id",
                "count",
            }:
                raise ScienceRequestV1CompilerError(
                    f"exact founder composition row {index} is not exact"
                )
            profile_id = row["founder_profile_id"]
            count = row["count"]
            if (
                not isinstance(profile_id, str)
                or not profile_id
                or type(count) is not int
                or count < 0
            ):
                raise ScienceRequestV1CompilerError(
                    f"exact founder composition row {index} is invalid"
                )
            composition_founders.append(profile_id)
            composition_count += count
        if continuation_state_ref is None and (
            not declared_founders
            or len(declared_founders) != len(set(declared_founders))
            or len(composition_founders) != len(set(composition_founders))
            or composition_founders != declared_founders
        ):
            raise ScienceRequestV1CompilerError(
                "exact founder composition must cover each declared founder once"
            )
        if continuation_state_ref is None and composition_count != initial_live_count:
            raise ScienceRequestV1CompilerError(
                "exact founder composition does not sum to initial_live_count"
            )
        if continuation_state_ref is None and not any(
            count > 0 for count in [row["count"] for row in composition]
        ):
            raise ScienceRequestV1CompilerError(
                "exact founder composition must contain a live founder"
            )
        resolved_controls = _mapping(
            resolved_template.get("controls"),
            field="resolved world_template.controls",
        )
        seeds = (
            {}
            if continuation_state_ref is not None
            else _seed_bindings(
                _mapping(
                    resolved_template.get("seeds"),
                    field="resolved world_template.seeds",
                )
            )
        )
        controls = _control_bindings(
            resolved_controls,
            total_ticks=target_next_tick,
            descriptors=descriptors_by_id,
            start_next_tick=start_next_tick,
        )
        reducer_configuration = deepcopy(base_reducer_configuration)
        required_observations = set(
            reducer_descriptor.required_observation_capability_ids
        )
        if "strategy_attribution" in reducer_configuration:
            required_observations.update(SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1)
            if reducer_configuration["strategy_attribution"].get(
                "direct_genotype_attribution", False
            ):
                required_observations.update(
                    SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1
                )
        if "hereditary_attribution" in reducer_configuration:
            required_observations.update(
                SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1
            )
            validation_errors = validate_hereditary_attribution_configuration_v1(
                reducer_configuration,
                required_observations=required_observations,
            )
            if validation_errors:
                raise ScienceRequestV1CompilerError(
                    "invalid hereditary attribution binding: "
                    + ", ".join(validation_errors)
                )
        if "exact_natural_source_states" in initialization:
            required_observations.add(EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1)
        reducers = [
            {
                "reducer_id": reducer_descriptor.reducer_id,
                "contract_version": str(reducer_descriptor.contract_version),
                "required_observations": sorted(required_observations),
                "output_contract_id": reducer_descriptor.output_contract_id,
                "configuration": reducer_configuration,
            }
        ]
        arm_dimensions = {
            name: assignment[name]
            for name in dimension_names
            if name != replicate_dimension
        }
        science = {
            "test_id": test_id,
            "test_version": test_version,
            "study_id": campaign_id,
            "arm_id": f"arm-{canonical_sha256(arm_dimensions).hex[:32]}",
            "replicate": assignment[replicate_dimension],
        }
        semantic_inputs = {
            "world": world_payload,
            "initialization": initialization,
            "dimension_bindings": {name: assignment[name] for name in dimension_names},
            "control_bindings": controls,
            "seed_bindings": seeds,
            "time": time_payload,
            "initial_state_identity": canonical_sha256(
                {
                    "world": world_payload,
                    "initialization": initialization,
                    "seed_bindings": seeds,
                }
            ).to_dict(),
        }
        semantic_sha256 = canonical_sha256(
            {
                "schema_version": "eiel_science_world_record_v1",
                "science": science,
                "semantic_inputs": semantic_inputs,
            }
        )
        acquisition_sha256 = canonical_sha256(
            {
                "schema_version": "eiel_science_world_record_v1",
                "semantic_world_sha256": semantic_sha256.to_dict(),
                "acquisition": acquisition,
                "reducers": reducers,
            }
        )
        worlds.append(
            ScienceWorldRecordV1(
                world_id=f"world-{semantic_sha256.hex[:32]}",
                science=CanonicalPayloadV1.from_value(
                    _SCIENCE_OWNERSHIP_SCHEMA, science
                ),
                semantic_inputs=CanonicalPayloadV1.from_value(
                    _RESOLVED_INPUTS_SCHEMA, semantic_inputs
                ),
                acquisition=CanonicalPayloadV1.from_value(
                    _ACQUISITION_SCHEMA, acquisition
                ),
                reducers=CanonicalPayloadV1.from_value(_REDUCERS_SCHEMA, reducers),
                semantic_world_sha256=semantic_sha256,
                acquisition_sha256=acquisition_sha256,
            )
        )
    worlds.sort(key=lambda world: world.world_id)
    compiler_component_sha256 = _compiler_component_sha256()
    registry_bindings = {
        "founder_registry_sha256": (
            None
            if founder_registry_sha256 is None
            else founder_registry_sha256.to_dict()
        ),
        "founder_profile_ids": list(registered_founder_ids),
        "founder_policy_bindings": combined_policy_bindings,
        "control_catalog_sha256": science_control_catalog_sha256(
            control_descriptors
        ).to_dict(),
        "control_descriptor_sha256s": {
            descriptor.control_id: descriptor.descriptor_sha256().to_dict()
            for descriptor in control_descriptors
            if descriptor.control_id in template_controls
        },
        "reducer_descriptor_sha256": (reducer_descriptor.descriptor_sha256().to_dict()),
    }
    if source_bank_binding is not None:
        registry_bindings["exact_natural_source_state_bank"] = dict(
            source_bank_binding
        )
        registry_bindings["exact_natural_source_state_profile_ids"] = sorted(
            exact_profile_ids
        )
    source_spec_core = {
        "schema_version": "eiel_science_request_v1_source_spec_v1",
        "campaign_id": campaign_id,
        "request_sha256": request_sha256.to_dict(),
        "dimension_ids": list(dimension_names),
        "post_exclusion_world_count": len(worlds),
        "registry_bindings": registry_bindings,
    }
    source_spec = {
        **source_spec_core,
        "source_spec_sha256": canonical_sha256(source_spec_core).to_dict(),
    }
    bundle = ScienceWorldBundleV1(
        bundle_id=f"SCIENCE-BUNDLE-{request_sha256.hex[:32].upper()}",
        compact_source_specs=CanonicalPayloadV1.from_value(
            _SOURCE_SPECS_SCHEMA, [source_spec]
        ),
        compiler_component_sha256=compiler_component_sha256,
        worlds=tuple(worlds),
    )
    request_batch_plan = normalized_request.batch_plan()
    nominal_batch_plan = _nominal_batch_plan(
        bundle.worlds, request_plan=request_batch_plan
    )
    receipt_core = {
        "schema_version": SCIENCE_REQUEST_V1_COMPILATION_RECEIPT_SCHEMA_VERSION,
        "compiler_id": SCIENCE_REQUEST_V1_COMPILER_ID,
        "request_sha256": request_sha256.to_dict(),
        "compiler_component_sha256": compiler_component_sha256.to_dict(),
        "bundle_id": bundle.bundle_id,
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "semantic_world_count": bundle.world_count,
        "expect_worlds": payload.get("expect_worlds"),
        "registry_bindings": registry_bindings,
        "nominal_batch_plan": nominal_batch_plan,
        "time_authority": "absolute_tick_v1",
        "execution_capability_claimed": False,
        "qualification_authority_effect": False,
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core).to_dict(),
    }
    return ScienceRequestV1CompilationV1(
        _normalized_request=CanonicalPayloadV1.from_value(
            _NORMALIZED_REQUEST_SCHEMA, payload
        ),
        world_bundle=bundle,
        _compilation_receipt=CanonicalPayloadV1.from_value(
            _COMPILATION_RECEIPT_SCHEMA, receipt
        ),
    )


compile_eiel_science_request_v1 = compile_science_request_v1


__all__ = (
    "SCIENCE_REQUEST_V1_COMPILATION_RECEIPT_SCHEMA_VERSION",
    "SCIENCE_REQUEST_V1_COMPILER_ID",
    "SCIENCE_REQUEST_V1_NOMINAL_BATCH_PLAN_SCHEMA_VERSION",
    "ScienceRequestV1CompilationV1",
    "ScienceRequestV1CompilerError",
    "compile_eiel_science_request_v1",
    "compile_science_request_v1",
)
