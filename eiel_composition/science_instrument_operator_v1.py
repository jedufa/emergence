"""Thin operator surface for the EIEL science instrument over CUDA V2.

This module is deliberately a composition boundary, not another campaign
manager.  The maintained rich-science normalizer/compiler, explicit world
bundle contract, CUDA V2 carrier family, and Launchpad execution front door
remain the authorities.  The operator surface only loads those contracts,
materializes their content-addressed projections, and (for one already
materialized batch) calls :func:`execute_bound_campaign_plan_v2`.  For a
declared compact world-atlas reducer, execution also starts an independent
sealed-artifact follower before that call and joins it only after the front
door returns.  The reducer never calls a provider or simulator.

The execution command accepts admitted engineering demonstrations,
exploratory science, and scientific execution.  It constructs an
``ExecutionRequestV2`` with the corresponding non-qualification purpose and
requires a canonical science sidecar.  This module does not perform PREQUAL,
QualPrep, or Qualification.  The compact reducer's native C++ build is the
default technical sidecar backend; its checked-in manifest is still
development-only and cannot grant production-routing or publication authority.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Final

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ScienceEvidencePartitionV1,
    ScienceAdmissionOutcomeV1,
    ScienceExecutionPlanV1,
    ScienceObservationProfileV1,
    ScienceRecordKindV1,
    SchemaVersion,
    canonical_json_bytes,
    canonical_sha256,
    derive_science_semantic_execution_id_v1,
    validate_science_native_control_receipt_v1,
)
from engine.campaigns.baseline_v1 import BaselineDeclarationV1
from engine.campaigns.cuda_full_evolution_v2_provider import (
    CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1,
)
from engine.campaigns.execution_front_door_v1 import (
    ExecutionRequestV2,
    ResourceWaitPolicyV1,
    execute_bound_campaign_plan_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse import (
    WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
    validate_workhorse_selected_world_observation_v1,
)

from eiel_composition.science_cuda_v2_carrier_v1 import (
    SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
    SCIENCE_CUDA_V2_OBSERVATION_OFF_V1,
    SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
    SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
    SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1,
    SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1,
    CudaV2ScienceCarrierFamilyAdapterV1,
)
from engine.campaigns.science_cuda_v2_native_world_v1 import (
    ScienceCudaV2NativeWorldError,
    project_science_cuda_v2_native_world_v1,
)
from eiel_composition.science_instrument import (
    ScienceCompilationBundleV1,
    compile_science_campaign_v1,
)
from eiel_composition.science_instrument_admission_v1 import (
    ScienceAdmissionDecisionV1,
    classify_science_campaign_v1,
)
from eiel_composition.science_instrument_artifacts_v1 import (
    ScienceInstrumentArtifactError,
    load_science_host_context_v1,
    parse_science_host_context_v1,
)
from eiel_composition.science_instrument_host_front_door_v1 import (
    ScienceInstrumentHostContextV1,
)
from eiel_composition.science_cuda_v2_native_control_v1 import (
    ScienceCudaV2NativeControlError,
    project_science_cuda_v2_native_control_v1,
)
from eiel_composition.exact_natural_source_state_v1 import (
    EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1,
    validate_materialized_exact_natural_source_initialization_v1,
)
from eiel_composition.science_request_v2 import (
    NormalizedRichScienceRequestV2,
    normalize_rich_production_science_request_v2,
)
from eiel_composition.science_world_bundle_v1 import (
    ScienceWorldBundleAdmissionV1,
    ScienceWorldAdmissionFindingV1,
    ScienceWorldBundleError,
    ScienceWorldBundleV1,
    admit_science_world_bundle_v1,
    load_science_world_bundle_v1,
    materialize_science_world_bundle_v1,
    write_science_world_bundle_v1,
)
from emergence_evidence.science_mutsup_direct_retention_v1 import (
    MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
    MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
    MUTSUP_DIRECT_RETENTION_REQUIRED_CAPABILITY_IDS,
    aggregate_mutsup_direct_retention_v1,
)
from emergence_evidence.science_compact_world_atlas_v1 import (
    COMPACT_WORLD_ATLAS_CONTRACT_VERSION,
    COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID,
    COMPACT_WORLD_ATLAS_REDUCER_ID,
    CompactWorldAtlasAsyncSidecarV1,
    CompactWorldAtlasIncrementalReducerV1,
    CompactWorldAtlasReducerError,
    NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1,
    NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1,
    NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1,
)
from emergence_evidence.native_compact_reducer_v1 import (
    default_native_compact_reducer_binding_v1,
    load_native_compact_reducer_v1,
)
from emergence_evidence.science_native_sidecar_v1 import (
    SCIENCE_NATIVE_SIDECAR_MODE_V1,
    SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1,
    NativeScienceSidecarV1,
    load_native_science_sidecar_v1,
)
from emergence_evidence.science_reducer_registry_v1 import (
    execute_registered_science_reducer_v1,
)

SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION: Final[str] = (
    "eiel_science_instrument_operator_v1"
)
SCIENCE_INSTRUMENT_MATERIALIZATION_SCHEMA_VERSION: Final[str] = (
    "eiel_science_instrument_materialization_v1"
)
SCIENCE_INSTRUMENT_EXECUTION_LINEAGE_SCHEMA_VERSION: Final[str] = (
    "eiel_science_instrument_execution_lineage_v1"
)
SCIENCE_COMPACT_PROFILE_ID: Final[str] = "SCIENCE_COMPACT"
SCIENCE_MUTSUP_REDUCER_ID: Final[str] = "science.mutsup.direct_retention.v1"
SCIENCE_MUTSUP_RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_science_mutsup_engineering_result_v1"
)
SCIENCE_MUTSUP_APPLIED_OUTCOMES_ONLY_COMPLETE: Final[str] = (
    "applied_outcomes_only_complete"
)
SCIENCE_COMPACT_REDUCER_DEFAULT_WORKERS: Final[int] = 4
SCIENCE_COMPACT_REDUCER_MIN_TAIL_SECONDS: Final[float] = 60.0
SCIENCE_COMPACT_REDUCER_MAX_TAIL_SECONDS: Final[float] = 3600.0
SCIENCE_COMPACT_REDUCER_REFERENCE_WORLD_COUNT: Final[int] = 64
SCIENCE_COMPACT_REDUCER_REFERENCE_TICK_HORIZON: Final[int] = 100_000


class ScienceInstrumentOperatorError(ValueError):
    """The operator surface cannot prove an exact instrument binding."""


JsonInputV1 = str | Path | Mapping[str, Any]
ExecutionFrontDoorV1 = Callable[..., Mapping[str, Any]]


def resolve_compact_reducer_tail_seconds_v1(
    requested: float | None,
    *,
    world_count: int,
    total_ticks: int,
) -> tuple[float, str]:
    """Choose a safe reducer wait without hiding explicit overrides.

    A fixed 60-second ceiling is adequate for tiny demonstrations but makes
    large scientific batches fail after a successful native execution.  The
    automatic budget scales with the exact logical batch and tick horizon;
    an explicit caller value remains a bounded opt-in override.
    """

    if type(world_count) is not int or world_count < 1:
        raise ScienceInstrumentOperatorError(
            "compact reducer world_count must be a positive integer"
        )
    if type(total_ticks) is not int or total_ticks < 1:
        raise ScienceInstrumentOperatorError(
            "compact reducer total_ticks must be a positive integer"
        )
    if requested is not None:
        if (
            isinstance(requested, bool)
            or not isinstance(requested, (int, float))
            or not math.isfinite(float(requested))
            or not 0 < float(requested) <= SCIENCE_COMPACT_REDUCER_MAX_TAIL_SECONDS
        ):
            raise ScienceInstrumentOperatorError(
                "compact_reducer_tail_seconds must be positive and at most "
                f"{int(SCIENCE_COMPACT_REDUCER_MAX_TAIL_SECONDS)}"
            )
        return float(requested), "explicit"

    world_scale = max(
        1.0,
        float(world_count) / float(SCIENCE_COMPACT_REDUCER_REFERENCE_WORLD_COUNT),
    )
    horizon_scale = max(
        0.25,
        float(total_ticks) / float(SCIENCE_COMPACT_REDUCER_REFERENCE_TICK_HORIZON),
    )
    automatic = math.ceil(
        SCIENCE_COMPACT_REDUCER_MIN_TAIL_SECONDS * world_scale * horizon_scale
    )
    return (
        float(
            min(
                SCIENCE_COMPACT_REDUCER_MAX_TAIL_SECONDS,
                max(SCIENCE_COMPACT_REDUCER_MIN_TAIL_SECONDS, automatic),
            )
        ),
        "auto_world_count_tick_horizon",
    )


@dataclass(frozen=True, slots=True)
class _LoadedJsonV1:
    value: dict[str, Any]
    path: Path | None
    file_sha256: str


def _as_sha256(value: object, *, field: str) -> str:
    if isinstance(value, ContentDigest):
        return value.hex
    if isinstance(value, Mapping):
        if set(value) != {"algorithm", "hex"} or value.get("algorithm") != "sha256":
            raise ScienceInstrumentOperatorError(f"{field} is not a SHA-256 digest")
        candidate = value.get("hex")
    else:
        candidate = value
    if (
        not isinstance(candidate, str)
        or len(candidate) != 64
        or any(character not in "0123456789abcdef" for character in candidate)
    ):
        raise ScienceInstrumentOperatorError(f"{field} is not a SHA-256 digest")
    return candidate


def _read_json_input(value: JsonInputV1, *, name: str) -> _LoadedJsonV1:
    if isinstance(value, Mapping):
        raw = dict(value)
        try:
            encoded = canonical_json_bytes(raw)
        except (TypeError, ValueError) as error:
            raise ScienceInstrumentOperatorError(
                f"{name} is not canonical JSON"
            ) from error
        return _LoadedJsonV1(raw, None, hashlib.sha256(encoded).hexdigest())
    source = Path(value).expanduser()
    if source.is_symlink():
        raise ScienceInstrumentOperatorError(
            f"{name} must not be a symbolic link: {source}"
        )
    source = source.resolve()
    if not source.is_file():
        raise ScienceInstrumentOperatorError(f"{name} is not a regular file: {source}")
    try:
        encoded = source.read_bytes()
        raw_value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceInstrumentOperatorError(
            f"{name} is unreadable: {source}"
        ) from error
    if not isinstance(raw_value, Mapping):
        raise ScienceInstrumentOperatorError(f"{name} must be a JSON object")
    return _LoadedJsonV1(dict(raw_value), source, hashlib.sha256(encoded).hexdigest())


def _load_context(
    value: JsonInputV1,
) -> tuple[ScienceInstrumentHostContextV1, _LoadedJsonV1]:
    loaded = _read_json_input(value, name="science host context")
    try:
        context = (
            load_science_host_context_v1(loaded.path)
            if loaded.path is not None
            else parse_science_host_context_v1(loaded.value)
        )
    except (ScienceInstrumentArtifactError, TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"science host context is invalid: {error}"
        ) from error
    return context, loaded


def _load_rich_request(
    value: JsonInputV1,
) -> tuple[NormalizedRichScienceRequestV2, _LoadedJsonV1]:
    loaded = _read_json_input(value, name="rich ProductionScienceRequestV2")
    try:
        normalized = normalize_rich_production_science_request_v2(loaded.value)
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"rich ProductionScienceRequestV2 is invalid: {error}"
        ) from error
    lifecycle = normalized.campaign.artifact_lifecycle.to_value()
    if (
        not isinstance(lifecycle, Mapping)
        or lifecycle.get("profile_id") != SCIENCE_COMPACT_PROFILE_ID
    ):
        raise ScienceInstrumentOperatorError(
            "rich science request must use artifact_lifecycle.profile_id=SCIENCE_COMPACT"
        )
    return normalized, loaded


def _load_base_campaign_plan(value: JsonInputV1) -> _LoadedJsonV1:
    loaded = _read_json_input(value, name="compiled base campaign plan")
    plan = loaded.value
    if plan.get("schema_version") != "eiel_campaign_plan_v1":
        raise ScienceInstrumentOperatorError(
            "compiled base campaign plan must use eiel_campaign_plan_v1"
        )
    declared = plan.get("plan_sha256")
    if (
        _as_sha256(declared, field="base_campaign_plan.plan_sha256")
        != canonical_sha256(
            {key: item for key, item in plan.items() if key != "plan_sha256"}
        ).hex
    ):
        raise ScienceInstrumentOperatorError(
            "compiled base campaign plan plan_sha256 does not match canonical content"
        )
    lifecycle = plan.get("artifact_lifecycle")
    profile = (
        lifecycle.get("lifecycle_profile") if isinstance(lifecycle, Mapping) else None
    )
    if profile is None and isinstance(lifecycle, Mapping):
        profile = lifecycle.get("profile_id")
    if profile != SCIENCE_COMPACT_PROFILE_ID:
        raise ScienceInstrumentOperatorError(
            "compiled base campaign plan must use artifact_lifecycle SCIENCE_COMPACT"
        )
    if not isinstance(plan.get("execution"), Mapping):
        raise ScienceInstrumentOperatorError(
            "compiled base campaign plan execution is absent"
        )
    if not isinstance(plan.get("output"), Mapping):
        raise ScienceInstrumentOperatorError(
            "compiled base campaign plan output is absent"
        )
    return loaded


def _profile_id(value: object) -> str:
    candidate = getattr(value, "value", value)
    if not isinstance(candidate, str) or not candidate:
        raise ScienceInstrumentOperatorError("context observation profile is invalid")
    return candidate


def _requested_scientific_capabilities(world: object) -> tuple[str, ...]:
    """Mirror the existing campaign capability projection for one materialized world."""

    semantic = world.semantic_inputs.to_value()
    if not isinstance(semantic, Mapping):
        raise ScienceInstrumentOperatorError(
            f"{world.world_id}: semantic inputs are invalid"
        )
    world_input = semantic.get("world")
    initialization = semantic.get("initialization")
    time = semantic.get("time")
    if not isinstance(world_input, Mapping) or not isinstance(initialization, Mapping):
        raise ScienceInstrumentOperatorError(
            f"{world.world_id}: world inputs are invalid"
        )
    if not isinstance(time, Mapping):
        raise ScienceInstrumentOperatorError(
            f"{world.world_id}: time inputs are invalid"
        )
    capabilities: set[str] = set()
    world_profile = world_input.get("world_profile_id")
    if isinstance(world_profile, str):
        capabilities.add(f"world_profile:{world_profile}")
    overlays = world_input.get("overlay_profile_ids", [])
    if not isinstance(overlays, list):
        raise ScienceInstrumentOperatorError(
            f"{world.world_id}: overlay profiles are invalid"
        )
    capabilities.update(
        f"world_overlay:{item}" for item in overlays if isinstance(item, str)
    )
    exact_source_mode = isinstance(
        initialization.get("exact_natural_source_states"), list
    )
    if exact_source_mode:
        capabilities.add(EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1)
    for field, prefix in (
        ("founder_profiles", "founder_profile"),
        ("spatial_placement_profile", "placement_profile"),
        ("standing_variation_profile", "variation_profile"),
    ):
        if field == "founder_profiles" and not exact_source_mode:
            values = initialization.get(field, [])
            if not isinstance(values, list):
                raise ScienceInstrumentOperatorError(
                    f"{world.world_id}: initialization.{field} is invalid"
                )
            capabilities.update(
                f"{prefix}:{item}" for item in values if isinstance(item, str)
            )
        else:
            value = initialization.get(field)
            if isinstance(value, str):
                capabilities.add(f"{prefix}:{value}")
    controls = semantic.get("control_bindings", [])
    if not isinstance(controls, list):
        raise ScienceInstrumentOperatorError(
            f"{world.world_id}: control bindings are invalid"
        )
    for binding in controls:
        if not isinstance(binding, Mapping):
            raise ScienceInstrumentOperatorError(
                f"{world.world_id}: control binding is invalid"
            )
        control_id = binding.get("control_id")
        schedule = binding.get("schedule")
        if isinstance(control_id, str):
            capabilities.add(f"control:{control_id}")
        if isinstance(schedule, Mapping) and isinstance(schedule.get("kind"), str):
            capabilities.add(f"schedule:{schedule['kind']}")
    branches = time.get("branches", [])
    if branches:
        capabilities.add("scientific_checkpoint_fork_v1")
    if time.get("restore") is not None:
        capabilities.add("absolute_tick_restore_v1")
    return tuple(sorted(capabilities))


def _native_w32_dimension_findings(
    world: object,
    context: ScienceInstrumentHostContextV1,
    *,
    test_id: str,
) -> list[ScienceWorldAdmissionFindingV1]:
    """Validate a closed bundle-driven native projection generically."""

    physical = _payload_mapping(
        context.physical_execution_core, field="context.physical_execution_core"
    )
    native_w32 = physical.get("carrier_profile_id") == "w32x32p256c512_gpuv2"
    selected_workhorse = (
        physical.get("carrier_profile_id")
        in {
            "w64x64p1024c2048_gpuv2",
            "w128x80p2560c5120_gpuv2",
        }
        and physical.get("selected_world_only") is True
        and physical.get("observation_mode")
        == "OFF_AUTHENTICATED_HOST_BOUNDARY_V1"
        and physical.get("acs_checkpoint_slots") == 2
    )
    if not native_w32 and not selected_workhorse:
        return []
    try:
        semantic_inputs = _payload_mapping(
            getattr(world, "semantic_inputs", None),
            field=f"{world.world_id}.semantic_inputs",
        )
        initialization = _payload_mapping(
            semantic_inputs.get("initialization"),
            field=f"{world.world_id}.initialization",
        )
        if "exact_natural_source_states" in initialization:
            validate_materialized_exact_natural_source_initialization_v1(
                initialization
            )
        project_science_cuda_v2_native_world_v1(
            {
                "semantic_inputs": semantic_inputs,
                "science": _payload_mapping(
                    getattr(world, "science", None),
                    field=f"{world.world_id}.science",
                ),
            }
        )
    except (ScienceCudaV2NativeWorldError, TypeError, ValueError) as error:
        return [
            ScienceWorldAdmissionFindingV1(
                outcome=ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY,
                world_id=world.world_id,
                test_id=test_id,
                field="native_projection.generic_world_contract",
                detail=f"unsupported bundle-driven native world projection: {error}",
            )
        ]
    return []


def _engineering_bundle_admission(
    bundle: ScienceWorldBundleV1,
    context: ScienceInstrumentHostContextV1,
    campaign: object | None,
    *,
    maximum_total_ticks: int,
) -> ScienceWorldBundleAdmissionV1:
    if type(maximum_total_ticks) is not int or maximum_total_ticks <= 0:
        raise ScienceInstrumentOperatorError(
            "maximum_total_ticks must be a positive integer"
        )
    profile_id = _profile_id(context.resource_envelope.profile)
    physical = _payload_mapping(
        context.physical_execution_core, field="context.physical_execution_core"
    )
    bundle_driven_native_w32 = (
        physical.get("carrier_profile_id") == "w32x32p256c512_gpuv2"
    )
    bundle_driven_selected_workhorse = (
        physical.get("carrier_profile_id")
        in {
            "w64x64p1024c2048_gpuv2",
            "w128x80p2560c5120_gpuv2",
        }
        and physical.get("selected_world_only") is True
        and physical.get("observation_mode")
        == "OFF_AUTHENTICATED_HOST_BOUNDARY_V1"
        and physical.get("acs_checkpoint_slots") == 2
    )
    admission = admit_science_world_bundle_v1(
        bundle,
        supported_observations=context.available_observation_capability_ids,
        supported_profiles=(profile_id,),
        maximum_total_ticks=maximum_total_ticks,
        nonexecuted_reducer_ids=(
            (COMPACT_WORLD_ATLAS_REDUCER_ID,)
            if bundle_driven_selected_workhorse
            else ()
        ),
    )
    findings = list(admission.findings)
    available_science = set(context.available_scientific_capability_ids)
    for world in bundle.worlds:
        science = world.science.to_value()
        test_id = f"{science['test_id']}_V{science['test_version']}"
        semantic_inputs = world.semantic_inputs.to_value()
        initialization = semantic_inputs.get("initialization")
        selected_world = (
            isinstance(initialization, Mapping)
            and initialization.get("schema_version")
            == "eiel_science_selected_world_initialization_v1"
        )
        if bundle_driven_selected_workhorse and not selected_world:
            findings.append(
                ScienceWorldAdmissionFindingV1(
                    outcome=ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY,
                    world_id=world.world_id,
                    test_id=test_id,
                    field="native_projection.selected_world_initialization",
                    detail=(
                        "The Workhorse host context admits only exact selected-world "
                        "initialization."
                    ),
                )
            )
        # The scalar mutation projector is a retained compatibility subset for
        # older routes. Generic native W32 bundles and the exact selected-world
        # Workhorse boundary are admitted by the closed per-world projector
        # below; heterogeneous controls must not be forced through the former
        # MUTSUP-only shape.
        if not bundle_driven_native_w32 and not bundle_driven_selected_workhorse:
            try:
                project_science_cuda_v2_native_control_v1(world)
            except ScienceCudaV2NativeControlError as error:
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY,
                        world_id=world.world_id,
                        test_id=test_id,
                        field="capabilities.native_control_projection",
                        detail=f"unsupported native scientific projection: {error}",
                    )
                )
        if not bundle_driven_selected_workhorse or selected_world:
            findings.extend(
                _native_w32_dimension_findings(world, context, test_id=test_id)
            )
        for capability in _requested_scientific_capabilities(world):
            if not bundle_driven_native_w32 and capability not in available_science:
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY,
                        world_id=world.world_id,
                        test_id=test_id,
                        field=f"capabilities.scientific.{capability}",
                        detail="Requested scientific capability is absent from the host context.",
                    )
                )
    # The campaign-level required capability list is deliberately checked in
    # addition to concrete event/state kinds.  The materialized world record
    # owns concrete acquisition names, while this list names host/profile
    # capabilities such as control readback.
    observations = {} if campaign is None else campaign.observations.to_value()
    if not isinstance(observations, Mapping):
        raise ScienceInstrumentOperatorError("campaign observations are invalid")
    required_observation_capabilities = observations.get("required_capability_ids", [])
    if not isinstance(required_observation_capabilities, list):
        raise ScienceInstrumentOperatorError(
            "campaign observations.required_capability_ids is invalid"
        )
    available_observations = set(context.available_observation_capability_ids)
    for world in bundle.worlds:
        for capability in required_observation_capabilities:
            if capability not in available_observations:
                science = world.science.to_value()
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
                        world_id=world.world_id,
                        test_id=f"{science['test_id']}_V{science['test_version']}",
                        field="observations.required_capability_ids",
                        detail=f"missing host observation capability: {capability}",
                    )
                )
    # Profile capability IDs are the only additional observation authority
    # available in the active host context.
    active_profile = next(
        (
            descriptor
            for descriptor in context.observation_profile_descriptors
            if _profile_id(descriptor.profile) == profile_id
        ),
        None,
    )
    if active_profile is None:
        raise ScienceInstrumentOperatorError(
            "active observation profile is absent from the host context"
        )
    active_capabilities = set(active_profile.capability_ids)
    # Concrete acquisition capability IDs are already represented by the
    # canonical bundle admission's event/state/reducer checks.  The active
    # profile's declared capabilities are retained as an integrity assertion:
    # a requested host capability must be present in both context and profile.
    for capability in required_observation_capabilities:
        if capability not in active_capabilities:
            for world in bundle.worlds:
                science = world.science.to_value()
                findings.append(
                    ScienceWorldAdmissionFindingV1(
                        outcome=ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY,
                        world_id=world.world_id,
                        test_id=f"{science['test_id']}_V{science['test_version']}",
                        field="observations.required_capability_ids",
                        detail=f"missing active profile capability: {capability}",
                    )
                )
    if not findings:
        return admission
    findings.sort(
        key=lambda item: (item.outcome.value, item.world_id, item.field, item.detail)
    )
    return ScienceWorldBundleAdmissionV1(
        bundle_content_sha256=bundle.bundle_content_sha256(),
        admitted_world_ids=(),
        findings=tuple(findings),
    )


def _qualified_envelope_classification(
    campaign: object,
    compilation: ScienceCompilationBundleV1,
    context: ScienceInstrumentHostContextV1,
) -> ScienceAdmissionDecisionV1:
    """Project the existing qualification boundary without granting authority.

    Engineering materialization deliberately does not invoke QRR or any
    qualification lifecycle.  The effect-free classifier is retained as an
    explicit receipt projection so an engineering-admitted bundle can still
    say ``OUTSIDE_QUALIFIED_ENVELOPE`` when its requested world/control set is
    outside the currently accepted envelope.
    """

    if not hasattr(campaign, "world"):
        raise ScienceInstrumentOperatorError(
            "campaign is not a normalized science campaign"
        )
    try:
        return classify_science_campaign_v1(
            campaign,
            compilation,
            instrument_manifest=context.instrument_manifest,
            route_binding=context.route_binding,
            qualified_envelope=context.qualified_envelope,
            available_scientific_capability_ids=tuple(
                context.available_scientific_capability_ids
            ),
            available_observation_capability_ids=tuple(
                context.available_observation_capability_ids
            ),
        )
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"qualification-envelope classification failed: {error}"
        ) from error


def _write_canonical_json(
    path: Path, value: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    target = path.expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise ScienceInstrumentOperatorError(
            f"{name} must be a fresh regular path: {target}"
        )
    try:
        encoded = canonical_json_bytes(dict(value))
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(0o444)
    except (OSError, TypeError, ValueError) as error:
        try:
            target.unlink()
        except OSError:
            pass
        raise ScienceInstrumentOperatorError(f"cannot write {name}: {error}") from error
    return {
        "path": str(target),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "immutable": True,
    }


def _compact_reducer_execution_receipt_core_v1(
    *,
    compact_reducer: Mapping[str, Any],
    front_door_result: Mapping[str, Any],
    batch_index: int,
    runtime_root: Path,
    science_use_class: str,
) -> dict[str, Any]:
    """Project durable reducer operations without copying the atlas payload."""

    atlas = compact_reducer.get("atlas_result")
    strategy = compact_reducer.get("strategy_attribution_result")
    atlas_identity = {
        key: atlas.get(key)
        for key in (
            "schema_version",
            "reducer_id",
            "reducer_contract_version",
            "result_sha256",
            "capture_index_sha256",
            "logical_world_count",
            "physical_batch_count",
            "physical_composition_used",
        )
        if isinstance(atlas, Mapping) and key in atlas
    }
    strategy_identity = None
    if isinstance(strategy, Mapping):
        strategy_identity = {
            "schema_version": strategy.get("schema_version"),
            "result_sha256": strategy.get("result_sha256"),
            "result_identity_sha256": strategy.get("result_identity_sha256"),
            "canonical_sha256": canonical_sha256(dict(strategy)).hex,
        }
    reducer_projection = {
        key: value
        for key, value in compact_reducer.items()
        if key not in {"atlas_result", "strategy_attribution_result"}
    }
    core = {
        "schema_version": "eiel_native_compact_reducer_execution_receipt_v1",
        "technical_only": True,
        "qualification_started": False,
        "science_use_class": science_use_class,
        "batch_index": batch_index,
        "runtime_root": str(runtime_root),
        "producer": {
            "ok": front_door_result.get("ok"),
            "status": front_door_result.get("status"),
            "result_class": front_door_result.get("result_class"),
        },
        "reducer": reducer_projection,
        "atlas_identity": atlas_identity,
        "strategy_identity": strategy_identity,
    }
    core["receipt_sha256"] = canonical_sha256(core).hex
    return core


def _front_door_result_is_resumable_wait_v1(
    result: Mapping[str, Any],
) -> bool:
    """Identify a pre-execution wait that must remain retryable.

    A canonical front door may return a structured storage/resource wait
    before the producer or reducer has executed. Such a result is a journal
    state, not a completed reducer execution. Do not publish the normal
    attempt-local reducer receipt in that case: the same bound attempt is
    intentionally resumable and the receipt path must remain fresh.
    """

    if result.get("ok") is not False:
        return False
    statuses = {
        str(result.get(key) or "").upper()
        for key in ("status", "result_class")
    }
    return bool(
        statuses
        & {
            "WAIT_RESOURCE",
            "RESOURCE_WAIT",
            "WAIT_STORAGE",
            "STORAGE_WAIT",
        }
    )


def _output_directory(value: str | Path) -> Path:
    output = Path(value).expanduser()
    if output.is_symlink():
        raise ScienceInstrumentOperatorError(
            "materialization output must not be a symbolic link"
        )
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise ScienceInstrumentOperatorError(
            f"materialization output is not a directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _source_ref(
    loaded: _LoadedJsonV1, *, logical_sha256: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": None if loaded.path is None else str(loaded.path),
        "file_sha256": loaded.file_sha256,
    }
    if logical_sha256 is not None:
        result["logical_sha256"] = logical_sha256
    return result


def materialize_science_instrument_v1(
    request: JsonInputV1,
    science_context: JsonInputV1,
    base_campaign_plan: JsonInputV1,
    *,
    output_directory: str | Path,
    maximum_total_ticks: int,
    physical_batch_capacity: int | Mapping[str, int] = 1024,
    temporal_k: int | Mapping[str, int] = 1,
    execution_quantum_id: str | None = None,
    observation_mode: str = SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
    science_use_class: str = SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1,
) -> dict[str, Any]:
    """Materialize and lower one rich request without executing it.

    The function is effect-free with respect to simulation, providers, GPU,
    and lifecycle runs.  It writes only its own immutable compact projection
    files under ``output_directory``.  Admission rejects before carrier
    planning or lowering when the context cannot satisfy the explicit world
    contracts.
    """

    normalized, request_source = _load_rich_request(request)
    context, context_source = _load_context(science_context)
    base_source = _load_base_campaign_plan(base_campaign_plan)
    try:
        compilation = compile_science_campaign_v1(
            normalized.campaign,
            instrument_manifest=context.instrument_manifest,
            route_binding=context.route_binding,
            control_descriptors=context.control_descriptors,
            reducer_descriptors=context.reducer_descriptors,
            crca_semantic_binding=context.crca_semantic_binding,
            physical_execution_core=context.physical_execution_core,
            compiler_component_sha256=context.compiler_component_sha256,
        )
        bundle = materialize_science_world_bundle_v1(
            ((normalized.campaign, compilation),),
            bundle_id=(
                "SCIENCE-BUNDLE-"
                + str(normalized.normalized_request["request_sha256"])[:32]
            ),
        )
        qualification_classification = _qualified_envelope_classification(
            normalized.campaign, compilation, context
        )
        admission = _engineering_bundle_admission(
            bundle,
            context,
            normalized.campaign,
            maximum_total_ticks=maximum_total_ticks,
        )
    except (ScienceWorldBundleError, TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"science instrument materialization failed before lowering: {error}"
        ) from error
    if admission.outcome is not ScienceAdmissionOutcomeV1.ADMITTED:
        detail = json.dumps(admission.to_dict(), sort_keys=True, separators=(",", ":"))
        raise ScienceInstrumentOperatorError(
            f"science world bundle is not admitted: {admission.outcome.value}: {detail}"
        )

    carrier = CudaV2ScienceCarrierFamilyAdapterV1()
    try:
        carrier_plan = carrier.plan(
            bundle,
            admission,
            physical_batch_capacity=physical_batch_capacity,
            temporal_k=temporal_k,
            execution_quantum_id=execution_quantum_id,
        )
        lowered_batches = carrier.lower_all(
            base_source.value,
            bundle,
            admission,
            carrier_plan,
            observation_mode=observation_mode,
            science_use_class=science_use_class,
            instrument_manifest_sha256=context.instrument_manifest.manifest_sha256(),
            crca_semantic_run_digest=context.crca_semantic_binding.semantic_run_digest,
        )
    except (ScienceWorldBundleError, TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"science instrument carrier lowering failed: {error}"
        ) from error

    output = _output_directory(output_directory)
    normalized_artifact = _write_canonical_json(
        output / "normalized_request.json",
        normalized.normalized_request,
        name="normalized request artifact",
    )
    compilation_artifact = _write_canonical_json(
        output / "compilation_bundle.json",
        compilation.to_dict(),
        name="compilation bundle artifact",
    )
    bundle_artifact = write_science_world_bundle_v1(
        bundle, output / "world_bundle.json"
    )
    admission_artifact = _write_canonical_json(
        output / "bundle_admission.json",
        admission.to_dict(),
        name="bundle admission artifact",
    )
    carrier_artifact = _write_canonical_json(
        output / "carrier_execution_plan.json",
        carrier_plan.to_dict(),
        name="carrier execution plan artifact",
    )
    lowered_artifacts = [
        _write_canonical_json(
            output / f"lowered_batch_{index:04d}.json",
            lowered,
            name=f"lowered batch {index} artifact",
        )
        for index, lowered in enumerate(lowered_batches)
    ]
    identities = {
        "request_sha256": normalized.normalized_request["request_sha256"],
        "context_sha256": context_source.value.get("context_sha256"),
        "campaign_spec_sha256": compilation.campaign_spec_sha256.to_dict(),
        "compilation_bundle_sha256": compilation.compilation_bundle_sha256().to_dict(),
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "bundle_admission_sha256": canonical_sha256(admission.to_dict()).to_dict(),
        "carrier_plan_sha256": carrier_plan.plan_sha256().to_dict(),
        "base_campaign_plan_sha256": base_source.value["plan_sha256"],
    }
    lineage = {
        "rich_request": _source_ref(
            request_source,
            logical_sha256=str(normalized.normalized_request["request_sha256"]),
        ),
        "host_context": _source_ref(
            context_source,
            logical_sha256=(
                context_source.value.get("context_sha256", {}).get("hex")
                if isinstance(context_source.value.get("context_sha256"), Mapping)
                else None
            ),
        ),
        "campaign": {
            "campaign_id": normalized.campaign.campaign_id,
            "campaign_spec_sha256": compilation.campaign_spec_sha256.to_dict(),
        },
        "compilation": {
            "compilation_bundle_sha256": compilation.compilation_bundle_sha256().to_dict(),
            "compiler_component_sha256": compilation.compiler_component_sha256.to_dict(),
        },
        "bundle": {
            "bundle_id": bundle.bundle_id,
            "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
            "world_count": bundle.world_count,
        },
        "admission": {
            "bundle_admission_sha256": canonical_sha256(admission.to_dict()).to_dict(),
            "outcome": admission.outcome.value,
        },
        "carrier": {
            "carrier_plan_sha256": carrier_plan.plan_sha256().to_dict(),
            "logical_bundle_size": carrier_plan.logical_bundle_size,
            "batch_count": len(carrier_plan.batches),
        },
        "base_campaign_plan": {
            "path": None if base_source.path is None else str(base_source.path),
            "file_sha256": base_source.file_sha256,
            "plan_sha256": base_source.value["plan_sha256"],
        },
    }
    receipt_core: dict[str, Any] = {
        "schema_version": SCIENCE_INSTRUMENT_MATERIALIZATION_SCHEMA_VERSION,
        "operator_schema_version": SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION,
        "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
        "engineering_only": (
            science_use_class != SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
        ),
        "scientific_execution": (
            science_use_class == SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
        ),
        "prequal_started": False,
        "qualification_started": False,
        "science_use_class": science_use_class,
        "observation_mode": observation_mode,
        # This is a read-only projection of the existing qualification
        # boundary.  It is intentionally separate from engineering bundle
        # admission and has no Qualification/QRR authority.
        "qualified_envelope_classification": qualification_classification.to_dict(),
        "qualification_authority_effect": False,
        "maximum_total_ticks": maximum_total_ticks,
        "identities": identities,
        "lineage": lineage,
        "artifacts": {
            "normalized_request": normalized_artifact,
            "compilation_bundle": compilation_artifact,
            "world_bundle": bundle_artifact,
            "bundle_admission": admission_artifact,
            "carrier_plan": carrier_artifact,
            "lowered_batches": lowered_artifacts,
        },
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core).to_dict(),
    }
    receipt_artifact = _write_canonical_json(
        output / "materialization_receipt.json",
        receipt,
        name="materialization receipt",
    )
    return {
        **receipt,
        "receipt_path": receipt_artifact["path"],
        "receipt_file_sha256": receipt_artifact["file_sha256"],
        "normalized_request": normalized.normalized_request,
        "compilation_bundle": compilation.to_dict(),
        "world_bundle": bundle.to_dict(),
        "bundle_admission": admission.to_dict(),
        "qualified_envelope_classification": qualification_classification.to_dict(),
        "carrier_plan": carrier_plan.to_dict(),
        "lowered_batches": list(lowered_batches),
    }


def materialize_explicit_science_world_bundle_v1(
    world_bundle: JsonInputV1,
    science_context: JsonInputV1,
    base_campaign_plan: JsonInputV1,
    *,
    baseline: Mapping[str, Any],
    output_directory: str | Path,
    maximum_total_ticks: int,
    physical_batch_capacity: int | Mapping[str, int] = 1024,
    temporal_k: int | Mapping[str, int] = 64,
    execution_quantum_id: str | None = None,
    observation_mode: str = SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
    science_use_class: str = SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1,
) -> dict[str, Any]:
    """Admit and lower an already-explicit heterogeneous world bundle.

    This is the reusable native-science boundary for multi-study bundles.  It
    does not reinterpret panel, arm, test, seed, observation, or reducer
    semantics; those remain owned by each ``ScienceWorldRecordV1``.
    """

    bundle_source = _read_json_input(world_bundle, name="explicit science world bundle")
    try:
        bundle = ScienceWorldBundleV1.from_dict(bundle_source.value)
        baseline_declaration = BaselineDeclarationV1.from_value(baseline)
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"explicit science execution declaration is invalid: {error}"
        ) from error
    if baseline_declaration.intent == "qualification":
        raise ScienceInstrumentOperatorError(
            "qualification Baseline is forbidden by the science bundle operator"
        )
    context, context_source = _load_context(science_context)
    base_source = _load_base_campaign_plan(base_campaign_plan)
    admission = _engineering_bundle_admission(
        bundle, context, None, maximum_total_ticks=maximum_total_ticks
    )
    if admission.outcome is not ScienceAdmissionOutcomeV1.ADMITTED:
        detail = json.dumps(admission.to_dict(), sort_keys=True, separators=(",", ":"))
        raise ScienceInstrumentOperatorError(
            f"science world bundle is not admitted: {admission.outcome.value}: {detail}"
        )
    carrier = CudaV2ScienceCarrierFamilyAdapterV1()
    bundle_semantic_digest = canonical_sha256(
        {
            "schema_version": "eiel_science_bundle_semantic_execution_v1",
            "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
            "world_semantic_identities": [
                {
                    "world_id": world.world_id,
                    "semantic_world_sha256": world.semantic_world_sha256.to_dict(),
                    "science": world.science.to_value(),
                }
                for world in bundle.worlds
            ],
        }
    )
    try:
        carrier_plan = carrier.plan(
            bundle,
            admission,
            physical_batch_capacity=physical_batch_capacity,
            temporal_k=temporal_k,
            execution_quantum_id=execution_quantum_id,
        )
        lowered_batches = carrier.lower_all(
            base_source.value,
            bundle,
            admission,
            carrier_plan,
            observation_mode=observation_mode,
            science_use_class=science_use_class,
            instrument_manifest_sha256=context.instrument_manifest.manifest_sha256(),
            crca_semantic_run_digest=bundle_semantic_digest,
        )
    except (ScienceWorldBundleError, TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"explicit science bundle lowering failed: {error}"
        ) from error

    output = _output_directory(output_directory)
    execution_declaration = {
        "schema_version": "eiel_science_bundle_execution_declaration_v1",
        "baseline": baseline_declaration.payload(),
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "science_use_class": science_use_class,
        "observation_mode": observation_mode,
    }
    normalized_artifact = _write_canonical_json(
        output / "normalized_request.json",
        execution_declaration,
        name="science bundle execution declaration",
    )
    compilation_projection = {
        "schema_version": "eiel_explicit_science_world_bundle_projection_v1",
        "compiler_boundary": "EIEL_SCIENCE_WORLD_BUNDLE_V1",
        "bundle_id": bundle.bundle_id,
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "compiler_component_sha256": bundle.compiler_component_sha256.to_dict(),
        "source_bundle_file_sha256": bundle_source.file_sha256,
    }
    compilation_artifact = _write_canonical_json(
        output / "compilation_bundle.json",
        compilation_projection,
        name="explicit bundle compilation projection",
    )
    bundle_artifact = write_science_world_bundle_v1(
        bundle, output / "world_bundle.json"
    )
    admission_artifact = _write_canonical_json(
        output / "bundle_admission.json",
        admission.to_dict(),
        name="bundle admission artifact",
    )
    carrier_artifact = _write_canonical_json(
        output / "carrier_execution_plan.json",
        carrier_plan.to_dict(),
        name="carrier execution plan artifact",
    )
    lowered_artifacts = [
        _write_canonical_json(
            output / f"lowered_batch_{index:04d}.json",
            lowered,
            name=f"lowered batch {index} artifact",
        )
        for index, lowered in enumerate(lowered_batches)
    ]
    identities = {
        "request_sha256": canonical_sha256(execution_declaration).hex,
        "context_sha256": context_source.value.get("context_sha256"),
        "campaign_spec_sha256": None,
        "compilation_bundle_sha256": canonical_sha256(compilation_projection).to_dict(),
        "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
        "bundle_admission_sha256": canonical_sha256(admission.to_dict()).to_dict(),
        "carrier_plan_sha256": carrier_plan.plan_sha256().to_dict(),
        "base_campaign_plan_sha256": base_source.value["plan_sha256"],
        "bundle_semantic_execution_sha256": bundle_semantic_digest.to_dict(),
    }
    lineage = {
        "rich_request": _source_ref(
            bundle_source, logical_sha256=bundle.bundle_content_sha256().hex
        ),
        "host_context": _source_ref(context_source),
        "campaign": None,
        "compilation": compilation_projection,
        "bundle": {
            "bundle_id": bundle.bundle_id,
            "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
            "world_count": bundle.world_count,
        },
        "admission": {
            "bundle_admission_sha256": canonical_sha256(admission.to_dict()).to_dict(),
            "outcome": admission.outcome.value,
        },
        "carrier": {
            "carrier_plan_sha256": carrier_plan.plan_sha256().to_dict(),
            "logical_bundle_size": carrier_plan.logical_bundle_size,
            "batch_count": len(carrier_plan.batches),
        },
        "base_campaign_plan": {
            "path": None if base_source.path is None else str(base_source.path),
            "file_sha256": base_source.file_sha256,
            "plan_sha256": base_source.value["plan_sha256"],
        },
    }
    receipt_core = {
        "schema_version": SCIENCE_INSTRUMENT_MATERIALIZATION_SCHEMA_VERSION,
        "operator_schema_version": SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION,
        "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
        "engineering_only": (
            science_use_class != SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
        ),
        "scientific_execution": (
            science_use_class == SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
        ),
        "prequal_started": False,
        "qualification_started": False,
        "science_use_class": science_use_class,
        "observation_mode": observation_mode,
        "qualified_envelope_classification": {
            "primary_outcome": "NOT_EVALUATED_DIRECT_EXPLICIT_BUNDLE",
            "qualification_authority_effect": False,
        },
        "qualification_authority_effect": False,
        "maximum_total_ticks": maximum_total_ticks,
        "identities": identities,
        "lineage": lineage,
        "artifacts": {
            "normalized_request": normalized_artifact,
            "compilation_bundle": compilation_artifact,
            "world_bundle": bundle_artifact,
            "bundle_admission": admission_artifact,
            "carrier_plan": carrier_artifact,
            "lowered_batches": lowered_artifacts,
        },
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core).to_dict(),
    }
    receipt_artifact = _write_canonical_json(
        output / "materialization_receipt.json",
        receipt,
        name="materialization receipt",
    )
    return {
        **receipt,
        "receipt_path": receipt_artifact["path"],
        "receipt_file_sha256": receipt_artifact["file_sha256"],
        "world_bundle": bundle.to_dict(),
        "bundle_admission": admission.to_dict(),
        "carrier_plan": carrier_plan.to_dict(),
        "lowered_batches": list(lowered_batches),
    }


def _read_regular_json(
    path: Path, *, name: str, canonical: bool = True
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ScienceInstrumentOperatorError(f"{name} is not a regular file: {path}")
    try:
        encoded = path.read_bytes()
        raw = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceInstrumentOperatorError(f"{name} is unreadable: {path}") from error
    if not isinstance(raw, Mapping):
        raise ScienceInstrumentOperatorError(f"{name} must be an object")
    result = dict(raw)
    if canonical:
        try:
            is_canonical = canonical_json_bytes(result) == encoded
        except (TypeError, ValueError) as error:
            raise ScienceInstrumentOperatorError(
                f"{name} is not canonical JSON: {path}"
            ) from error
        if not is_canonical:
            raise ScienceInstrumentOperatorError(
                f"{name} is not canonical JSON: {path}"
            )
    return result


def _verify_materialization_artifact(
    root: Path, info: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    if not isinstance(info, Mapping):
        raise ScienceInstrumentOperatorError(f"{name} artifact reference is invalid")
    raw_path = info.get("path")
    expected = _as_sha256(info.get("file_sha256"), field=f"{name}.file_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ScienceInstrumentOperatorError(f"{name} artifact path is invalid")
    path = Path(raw_path).expanduser()
    if path.is_symlink():
        raise ScienceInstrumentOperatorError(
            f"{name} artifact must not be a symbolic link"
        )
    path = path.resolve()
    if path.parent != root or not path.is_file():
        raise ScienceInstrumentOperatorError(
            f"{name} artifact is outside its materialization directory"
        )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ScienceInstrumentOperatorError(f"{name} artifact file SHA-256 mismatch")
    return _read_regular_json(path, name=name)


def load_science_instrument_materialization_v1(
    materialization: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]]:
    """Load and verify a materialization receipt plus its lowered batches."""

    source = Path(materialization).expanduser()
    if source.is_symlink():
        raise ScienceInstrumentOperatorError(
            "materialization must not be a symbolic link"
        )
    source = source.resolve()
    root = source.parent if source.is_file() else source
    if not root.is_dir():
        raise ScienceInstrumentOperatorError(
            f"materialization directory is absent: {root}"
        )
    receipt_path = source if source.is_file() else root / "materialization_receipt.json"
    receipt = _read_regular_json(receipt_path, name="materialization receipt")
    if (
        receipt.get("schema_version")
        != SCIENCE_INSTRUMENT_MATERIALIZATION_SCHEMA_VERSION
    ):
        raise ScienceInstrumentOperatorError(
            "materialization receipt schema is unsupported"
        )
    declared = _as_sha256(receipt.get("receipt_sha256"), field="receipt_sha256")
    core = dict(receipt)
    core.pop("receipt_sha256", None)
    if canonical_sha256(core).hex != declared:
        raise ScienceInstrumentOperatorError("materialization receipt SHA-256 mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ScienceInstrumentOperatorError(
            "materialization receipt artifacts are absent"
        )
    normalized = _verify_materialization_artifact(
        root, artifacts.get("normalized_request"), name="normalized request"
    )
    # Validate every persisted input projection before permitting a run.  This
    # catches a stale/partial directory before Launchpad creates any runtime.
    _verify_materialization_artifact(
        root, artifacts.get("compilation_bundle"), name="compilation bundle"
    )
    _verify_materialization_artifact(
        root, artifacts.get("world_bundle"), name="world bundle"
    )
    _verify_materialization_artifact(
        root, artifacts.get("bundle_admission"), name="bundle admission"
    )
    _verify_materialization_artifact(
        root, artifacts.get("carrier_plan"), name="carrier plan"
    )
    raw_lowered = artifacts.get("lowered_batches")
    if not isinstance(raw_lowered, list) or not raw_lowered:
        raise ScienceInstrumentOperatorError("materialization has no lowered batches")
    lowered = tuple(
        _verify_materialization_artifact(root, info, name=f"lowered batch {index}")
        for index, info in enumerate(raw_lowered)
    )
    return root, receipt, normalized, lowered


def _runtime_root(value: str | Path, *, repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise ScienceInstrumentOperatorError(f"repo_root is not a directory: {root}")
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ScienceInstrumentOperatorError("runtime_root must not be a symbolic link")
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate == root:
        raise ScienceInstrumentOperatorError(
            "runtime_root may not be the repository root"
        )
    if candidate.exists() and candidate.is_file():
        raise ScienceInstrumentOperatorError("runtime_root is an existing regular file")
    return candidate


def _bound_plan_sha256(plan: Mapping[str, Any], runtime_root: Path) -> str:
    output = plan.get("output")
    if not isinstance(output, Mapping):
        raise ScienceInstrumentOperatorError("lowered plan output is absent")
    bound = dict(plan)
    bound["output"] = {**dict(output), "runtime_root": str(runtime_root)}
    bound.pop("plan_sha256", None)
    return canonical_sha256(bound).hex


def _compact_world_atlas_sidecar_target_v1(
    materialization_root: Path,
    receipt: Mapping[str, Any],
    lowered: Mapping[str, Any],
    runtime_root: Path,
) -> tuple[ScienceWorldBundleV1, Path, str] | None:
    """Resolve one exact batch bundle without granting reducer authority."""

    materialization = lowered.get("science_materialization")
    if (
        not isinstance(materialization, Mapping)
        or materialization.get("observation_mode")
        != SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1
    ):
        return None
    raw_world_ids = materialization.get("semantic_world_ids")
    if (
        not isinstance(raw_world_ids, list)
        or not raw_world_ids
        or any(not isinstance(value, str) or not value for value in raw_world_ids)
        or len(raw_world_ids) != len(set(raw_world_ids))
    ):
        raise ScienceInstrumentOperatorError(
            "compact reducer batch world ownership is invalid"
        )
    world_ids = tuple(raw_world_ids)
    if world_ids != tuple(sorted(world_ids)):
        raise ScienceInstrumentOperatorError(
            "compact reducer batch world order is not canonical"
        )

    artifacts = receipt.get("artifacts")
    bundle_info = (
        artifacts.get("world_bundle") if isinstance(artifacts, Mapping) else None
    )
    raw_bundle_path = (
        bundle_info.get("path") if isinstance(bundle_info, Mapping) else None
    )
    if not isinstance(raw_bundle_path, str):
        raise ScienceInstrumentOperatorError(
            "compact reducer world bundle artifact is absent"
        )
    bundle_path = Path(raw_bundle_path).expanduser()
    if bundle_path.is_symlink():
        raise ScienceInstrumentOperatorError(
            "compact reducer world bundle must not be a symbolic link"
        )
    bundle_path = bundle_path.resolve()
    if bundle_path.parent != materialization_root:
        raise ScienceInstrumentOperatorError(
            "compact reducer world bundle is outside its materialization"
        )
    try:
        source_bundle = load_science_world_bundle_v1(bundle_path)
    except (OSError, TypeError, ValueError, ScienceWorldBundleError) as error:
        raise ScienceInstrumentOperatorError(
            f"compact reducer world bundle is invalid: {error}"
        ) from error
    expected_bundle_sha256 = _as_sha256(
        bundle_info.get("bundle_content_sha256"),
        field="world_bundle.bundle_content_sha256",
    )
    if source_bundle.bundle_content_sha256().hex != expected_bundle_sha256:
        raise ScienceInstrumentOperatorError(
            "compact reducer world bundle identity differs from materialization"
        )
    worlds_by_id = {world.world_id: world for world in source_bundle.worlds}
    if any(world_id not in worlds_by_id for world_id in world_ids):
        raise ScienceInstrumentOperatorError(
            "compact reducer batch names a world outside the bound bundle"
        )
    selected_worlds = tuple(worlds_by_id[world_id] for world_id in world_ids)
    requested: list[bool] = []
    for world in selected_worlds:
        reducers = world.reducers.to_value()
        matches = [
            row
            for row in reducers
            if isinstance(row, Mapping)
            and row.get("reducer_id") == COMPACT_WORLD_ATLAS_REDUCER_ID
        ]
        if len(matches) > 1:
            raise ScienceInstrumentOperatorError(
                "compact world atlas reducer is duplicated for one world"
            )
        if matches and (
            matches[0].get("contract_version")
            != str(COMPACT_WORLD_ATLAS_CONTRACT_VERSION)
            or matches[0].get("output_contract_id")
            != COMPACT_WORLD_ATLAS_OUTPUT_CONTRACT_ID
        ):
            raise ScienceInstrumentOperatorError(
                "compact world atlas reducer contract differs"
            )
        requested.append(bool(matches))
    if not any(requested):
        return None
    if not all(requested):
        raise ScienceInstrumentOperatorError(
            "compact world atlas reduction must own the complete physical batch"
        )

    physical_batches = lowered.get("physical_batches")
    if not isinstance(physical_batches, list) or len(physical_batches) != 1:
        raise ScienceInstrumentOperatorError(
            "compact reducer requires one lowered physical batch"
        )
    physical_batch = physical_batches[0]
    batch_id = (
        physical_batch.get("batch_id") if isinstance(physical_batch, Mapping) else None
    )
    if (
        not isinstance(batch_id, str)
        or not batch_id
        or Path(batch_id).name != batch_id
        or physical_batch.get("cell_ids") != list(world_ids)
    ):
        raise ScienceInstrumentOperatorError(
            "compact reducer physical batch identity differs"
        )
    batch_bundle = (
        source_bundle
        if world_ids == tuple(world.world_id for world in source_bundle.worlds)
        else ScienceWorldBundleV1(
            bundle_id=f"{source_bundle.bundle_id}--{batch_id}",
            compact_source_specs=source_bundle.compact_source_specs,
            compiler_component_sha256=source_bundle.compiler_component_sha256,
            worlds=selected_worlds,
        )
    )
    compact_root = runtime_root / "batches" / batch_id / "science-compact"
    return batch_bundle, compact_root, batch_id


def _is_selected_world_host_observation_batch_v1(
    materialization: Mapping[str, Any],
) -> bool:
    return (
        materialization.get("schema_version")
        == "eiel_science_selected_world_technical_lowering_v1"
    )


def _authenticated_selected_world_host_observation_binding_v1(
    front_door_result: Mapping[str, Any],
    *,
    runtime_root: Path,
    expected_world_id: str,
) -> dict[str, Any]:
    """Reopen the provider observation without upgrading runtime authority."""

    terminal = front_door_result.get(
        "workhorse_terminalized_indexed_continuation"
    )
    generic_terminal = front_door_result.get("workhorse_terminalized_operation")
    binding = (
        terminal.get("host_boundary_observation")
        if isinstance(terminal, Mapping)
        else None
    )
    expected_fields = {
        "observation_mode",
        "path",
        "file_sha256",
        "content_sha256",
        "file_bytes",
        "checkpoint_occurrence_sha256",
        "checkpoint_occurrence_projection_sha256",
        "terminally_reopened",
        "runtime_observation_mode",
        "host_adapter_observation_only",
    }
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("operation") != "INDEXED_ACS_CONTINUATION"
        or generic_terminal != terminal
        or type(binding) is not dict
        or set(binding) != expected_fields
        or binding.get("observation_mode")
        != WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
        or binding.get("terminally_reopened") is not True
        or binding.get("runtime_observation_mode") != "OFF"
        or binding.get("host_adapter_observation_only") is not True
    ):
        raise ScienceInstrumentOperatorError(
            "selected-world child lacks its terminally reopened host observation"
        )
    try:
        observation_path = Path(str(binding["path"]))
        observation = validate_workhorse_selected_world_observation_v1(
            observation_path,
            expected_checkpoint_occurrence_sha256=str(
                binding["checkpoint_occurrence_sha256"]
            ),
            expected_checkpoint_occurrence_projection_sha256=str(
                binding["checkpoint_occurrence_projection_sha256"]
            ),
        )
        resolved_path = observation_path.resolve(strict=True)
        resolved_path.relative_to(runtime_root.resolve())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"selected-world host observation cannot be authenticated: {error}"
        ) from error
    integrity = observation.get("integrity")
    event_coverage = observation.get("event_observation_coverage")
    worlds = observation.get("world_observations")
    if (
        observation.get("cell_id") != expected_world_id
        or observation.get("execution_authority_granted") is not False
        or observation.get("qualification_authority_granted") is not False
        or observation.get("named_phenotype_reward_authority_granted") is not False
        or not isinstance(integrity, Mapping)
        or integrity.get("world_count") != 1
        or integrity.get("dropped_record_count") != 0
        or integrity.get("runtime_observation_mode_remains_off") is not True
        or integrity.get("host_boundary_adapter_is_observation_only") is not True
        or not isinstance(event_coverage, Mapping)
        or event_coverage.get("event_kind_rows_available") is not False
        or event_coverage.get("birth_death_causal_rows_claimed") is not False
        or not isinstance(worlds, list)
        or len(worlds) != 1
        or binding["path"] != observation["observation_path"]
        or binding["file_sha256"] != observation["observation_file_sha256"]
        or binding["content_sha256"]
        != observation["observation_content_sha256"]
        or binding["file_bytes"] != observation["observation_file_bytes"]
    ):
        raise ScienceInstrumentOperatorError(
            "selected-world host observation changed identity or claim boundary"
        )
    return dict(binding)


def execute_materialized_science_batch_v1(
    materialization: str | Path,
    *,
    batch_index: int,
    runtime_root: str | Path,
    repo_root: str | Path,
    resource_wait: ResourceWaitPolicyV1 | Mapping[str, Any] | None = None,
    crca_adapter: object | None = None,
    state_root: str | Path | None = None,
    execution_front_door: ExecutionFrontDoorV1 | None = None,
    compact_reducer_workers: int = SCIENCE_COMPACT_REDUCER_DEFAULT_WORKERS,
    compact_reducer_tail_seconds: float | None = None,
    compact_reducer_native_artifact: str | Path | None = None,
    compact_reducer_native_manifest: str | Path | None = None,
    compact_reducer_native_manifest_sha256: str | None = None,
    compact_reducer_native_use_class: str | None = None,
    nested_storage_parent_run_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one batch while the default native reducer follows it."""

    if type(compact_reducer_workers) is not int or compact_reducer_workers < 0:
        raise ScienceInstrumentOperatorError(
            "compact_reducer_workers must be a nonnegative integer"
        )
    native_reducer_values = (
        compact_reducer_native_artifact,
        compact_reducer_native_manifest,
        compact_reducer_native_manifest_sha256,
    )
    if any(value is not None for value in native_reducer_values) and not all(
        value is not None for value in native_reducer_values
    ):
        raise ScienceInstrumentOperatorError(
            "native compact reducer requires artifact, manifest, and digest"
        )
    if compact_reducer_native_manifest_sha256 is not None and (
        len(compact_reducer_native_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in compact_reducer_native_manifest_sha256
        )
    ):
        raise ScienceInstrumentOperatorError(
            "native compact reducer manifest digest is invalid"
        )
    if compact_reducer_native_use_class is None:
        compact_reducer_native_use_class = (
            NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1
            if compact_reducer_native_artifact is None
            else NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1
        )
    if compact_reducer_native_use_class not in {
        NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1,
        NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1,
        NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1,
    }:
        raise ScienceInstrumentOperatorError(
            "native compact reducer use class is invalid"
        )
    if compact_reducer_native_artifact is None and (
        compact_reducer_native_use_class
        != NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1
    ):
        raise ScienceInstrumentOperatorError(
            "native compact reducer use class requires a native artifact"
        )

    root, receipt, normalized, lowered_batches = (
        load_science_instrument_materialization_v1(materialization)
    )
    if type(batch_index) is not int or not 0 <= batch_index < len(lowered_batches):
        raise ScienceInstrumentOperatorError(
            "batch_index is outside the materialized batches"
        )
    lowered = lowered_batches[batch_index]
    science_use_class = lowered.get("science_use_class")
    if science_use_class not in {
        SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1,
        SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1,
        SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1,
    }:
        raise ScienceInstrumentOperatorError(
            "lowered batch is not an admitted science execution class"
        )
    if (
        compact_reducer_native_artifact is not None
        and compact_reducer_native_use_class
        == NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1
        and science_use_class
        != SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1
    ):
        raise ScienceInstrumentOperatorError(
            "engineering-real-use native reducer artifacts are restricted to "
            "engineering-demonstration campaigns"
        )
    materialization_meta = lowered.get("science_materialization")
    if not isinstance(materialization_meta, Mapping):
        raise ScienceInstrumentOperatorError(
            "lowered batch lacks science materialization metadata"
        )
    observation_mode = materialization_meta.get("observation_mode")
    selected_world_host_observation = (
        _is_selected_world_host_observation_batch_v1(materialization_meta)
    )
    if selected_world_host_observation:
        selected_world_ids = materialization_meta.get("semantic_world_ids")
        if selected_world_ids is None:
            selected_world_ids = [materialization_meta.get("semantic_world_id")]
        if (
            observation_mode != SCIENCE_CUDA_V2_OBSERVATION_OFF_V1
            or science_use_class
            != SCIENCE_CUDA_V2_USE_CLASS_ENGINEERING_DEMONSTRATION_V1
            or not isinstance(selected_world_ids, list)
            or len(selected_world_ids) != 1
            or not isinstance(selected_world_ids[0], str)
            or not selected_world_ids[0]
        ):
            raise ScienceInstrumentOperatorError(
                "selected-world host observation requires one engineering/no-claims "
                "world with runtime observation OFF"
            )
    elif observation_mode not in {
        SCIENCE_CUDA_V2_OBSERVATION_TYPED_SIDECAR_V1,
        SCIENCE_CUDA_V2_OBSERVATION_COMPACT_SIDECAR_V1,
    }:
        raise ScienceInstrumentOperatorError(
            "lowered batch does not use a canonical science sidecar"
        )
    if (
        receipt.get("science_use_class") != science_use_class
        or receipt.get("observation_mode") != observation_mode
    ):
        raise ScienceInstrumentOperatorError(
            "lowered science class/mode differs from materialization receipt"
        )
    source = receipt.get("lineage")
    if not isinstance(source, Mapping):
        raise ScienceInstrumentOperatorError("materialization lineage is absent")
    base = source.get("base_campaign_plan")
    if not isinstance(base, Mapping) or not isinstance(base.get("path"), str):
        raise ScienceInstrumentOperatorError(
            "execution requires a materialization made from a file-backed base campaign plan"
        )
    # The lowered plan is the execution input, but the original base plan is
    # still part of its provenance.  Re-read and bind that source before
    # handing anything to Launchpad so a post-materialization replacement
    # cannot silently change the campaign source named by the request.
    base_source = _load_base_campaign_plan(base["path"])
    expected_base_file_sha256 = _as_sha256(
        base.get("file_sha256"), field="lineage.base_campaign_plan.file_sha256"
    )
    if base_source.file_sha256 != expected_base_file_sha256:
        raise ScienceInstrumentOperatorError(
            "base campaign plan file changed after materialization"
        )
    expected_base_plan_sha256 = _as_sha256(
        base.get("plan_sha256"), field="lineage.base_campaign_plan.plan_sha256"
    )
    if (
        _as_sha256(base_source.value.get("plan_sha256"), field="base_plan.plan_sha256")
        != expected_base_plan_sha256
    ):
        raise ScienceInstrumentOperatorError(
            "base campaign plan identity changed after materialization"
        )
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping):
        raise ScienceInstrumentOperatorError("materialization identities are absent")
    if (
        _as_sha256(
            identities.get("base_campaign_plan_sha256"),
            field="identities.base_campaign_plan_sha256",
        )
        != expected_base_plan_sha256
    ):
        raise ScienceInstrumentOperatorError(
            "materialization identity does not bind the base campaign plan"
        )
    runtime = _runtime_root(runtime_root, repo_root=repo_root)
    lowered_plan_sha256 = _as_sha256(
        lowered.get("plan_sha256"), field="lowered_plan.plan_sha256"
    )
    bound_plan_sha256 = _bound_plan_sha256(lowered, runtime)
    source_lineage = {
        **dict(source),
        "compiled_plan_sha256": lowered_plan_sha256,
        "bound_plan_sha256": bound_plan_sha256,
    }
    world_ids = (
        list(selected_world_ids)
        if selected_world_host_observation
        else list(materialization_meta.get("semantic_world_ids", []))
    )
    science_lineage = {
        "schema_version": SCIENCE_INSTRUMENT_EXECUTION_LINEAGE_SCHEMA_VERSION,
        "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
        "science_use_class": science_use_class,
        "observation_mode": observation_mode,
        "request_sha256": identities.get("request_sha256"),
        "campaign_spec_sha256": identities.get("campaign_spec_sha256"),
        "compilation_bundle_sha256": identities.get("compilation_bundle_sha256"),
        "bundle_content_sha256": identities.get("bundle_content_sha256"),
        "bundle_admission_sha256": identities.get("bundle_admission_sha256"),
        "carrier_plan_sha256": identities.get("carrier_plan_sha256"),
        "base_campaign_plan_sha256": identities.get("base_campaign_plan_sha256"),
        "lowered_plan_sha256": lowered_plan_sha256,
        "batch_index": batch_index,
        "world_ids": world_ids,
    }
    baseline_raw = normalized.get("baseline")
    try:
        baseline = BaselineDeclarationV1.from_value(baseline_raw)
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            "normalized request baseline is invalid"
        ) from error
    if baseline.intent == "qualification":
        raise ScienceInstrumentOperatorError(
            "qualification Baseline is forbidden by the engineering operator"
        )
    execution = lowered.get("execution")
    if not isinstance(execution, Mapping):
        raise ScienceInstrumentOperatorError("lowered batch execution is absent")
    total_ticks = execution.get("total_ticks")
    if type(total_ticks) is not int or total_ticks < 1:
        raise ScienceInstrumentOperatorError(
            "lowered batch execution total_ticks must be a positive integer"
        )
    compact_reducer_tail_seconds, compact_reducer_tail_source = (
        resolve_compact_reducer_tail_seconds_v1(
            compact_reducer_tail_seconds,
            world_count=len(world_ids),
            total_ticks=total_ticks,
        )
    )
    acs_checkpoint_slots = execution.get("acs_checkpoint_slots", 0)
    if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
        0,
        1,
        2,
    }:
        raise ScienceInstrumentOperatorError(
            "lowered batch acs_checkpoint_slots must be 0, 1, or 2"
        )
    expected_deviation_ids = tuple(
        requirement_id
        for requirement_id in (
            CUDA_FULL_EVOLUTION_V2_BASELINE_DEVIATION_REQUIREMENT_IDS_V1
        )
        if not (
            requirement_id == "checkpoint.acs_two_slot"
            and acs_checkpoint_slots == 2
        )
    )
    declared_deviation_ids = tuple(row.requirement_id for row in baseline.deviations)
    if (
        baseline.intent != "development"
        or declared_deviation_ids != expected_deviation_ids
    ):
        expected_count = "four" if acs_checkpoint_slots == 2 else "five"
        raise ScienceInstrumentOperatorError(
            f"CUDA V2 exploratory execution requires the exact {expected_count} "
            "explicit engineering Baseline deviations for its ACS plan"
        )
    if isinstance(resource_wait, ResourceWaitPolicyV1):
        wait_policy = resource_wait
    else:
        try:
            wait_policy = ResourceWaitPolicyV1.from_value(resource_wait)
        except (TypeError, ValueError) as error:
            raise ScienceInstrumentOperatorError(
                f"resource_wait is invalid: {error}"
            ) from error
    request = ExecutionRequestV2(
        purpose=(
            "simulation"
            if science_use_class == SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
            else "engineering"
        ),
        campaign_spec=str(base["path"]),
        output_root=str(runtime),
        resource_wait=wait_policy,
        qualification_profile=None,
        baseline=baseline,
    )
    adapter = crca_adapter
    if adapter is None:
        from eiel_composition.cuda_full_evolution_v2_launchpad_adapter import (
            CanonicalCudaFullEvolutionV2LaunchpadAdapterV1,
        )

        adapter = CanonicalCudaFullEvolutionV2LaunchpadAdapterV1(
            repo_root=repo_root,
            state_root=state_root,
        )
    using_canonical_front_door = execution_front_door is None
    front_door = execution_front_door or execute_bound_campaign_plan_v2
    reducer_workers = compact_reducer_workers
    reducer_target = (
        None
        if reducer_workers == 0
        else _compact_world_atlas_sidecar_target_v1(
            root,
            receipt,
            lowered,
            runtime,
        )
    )
    reducer_sidecar: CompactWorldAtlasAsyncSidecarV1 | None = None
    bound_native_artifact: str | Path | None = compact_reducer_native_artifact
    bound_native_manifest: str | Path | None = compact_reducer_native_manifest
    bound_native_manifest_sha256: str | None = (
        compact_reducer_native_manifest_sha256
    )
    native_binding = "explicit"
    compact_reducer: dict[str, Any] = {
        "schema_version": "eiel_compact_world_atlas_async_execution_v1",
        "enabled": False,
        "asynchronous": True,
        "producer_notification_required": False,
        "producer_backpressure_path": False,
        "producer_waits_for_reducer_progress": False,
        "tail_budget_seconds": float(compact_reducer_tail_seconds),
        "tail_budget_source": compact_reducer_tail_source,
        "status": ("DISABLED" if reducer_workers == 0 else "NOT_REQUESTED_BY_BATCH"),
    }
    if reducer_target is not None:
        reducer_bundle, compact_root, reducer_batch_id = reducer_target
        if bound_native_artifact is None:
            try:
                (
                    bound_native_artifact,
                    bound_native_manifest,
                    bound_native_manifest_sha256,
                ) = default_native_compact_reducer_binding_v1()
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ScienceInstrumentOperatorError(
                    f"default native compact reducer is unavailable: {error}"
                ) from error
            native_binding = "checked_in_default"
        try:
            sidecar_kwargs: dict[str, Any] = {"max_workers": reducer_workers}
            if bound_native_artifact is not None:
                sidecar_kwargs.update(
                    {
                        "native_artifact_path": bound_native_artifact,
                        "native_manifest_path": bound_native_manifest,
                        "native_manifest_sha256": (
                            bound_native_manifest_sha256
                        ),
                        "native_artifact_use_class": (
                            compact_reducer_native_use_class
                        ),
                    }
                )
            if using_canonical_front_door:
                # The canonical execution front door binds its request and
                # plan receipts before invoking the producer.  Defer the
                # observer process until that binding exists so its
                # batch-local capture directory cannot violate the fresh-root
                # invariant.
                sidecar_kwargs["start_immediately"] = False
            reducer_sidecar = CompactWorldAtlasAsyncSidecarV1(
                compact_root,
                reducer_bundle,
                **sidecar_kwargs,
            )
        except (OSError, TypeError, ValueError, CompactWorldAtlasReducerError) as error:
            raise ScienceInstrumentOperatorError(
                f"compact reducer sidecar could not start: {error}"
            ) from error
        compact_reducer = {
            **compact_reducer,
            "enabled": True,
            "status": "FOLLOWING_SEALED_PREFIX",
            "batch_id": reducer_batch_id,
            "science_compact_root": str(compact_root),
            "max_workers": reducer_workers,
            "sidecar_pid": reducer_sidecar.pid,
            "execution_route": "native_cpp_v1",
            "native_binding": native_binding,
            "native_manifest_sha256": (
                bound_native_manifest_sha256
            ),
            "native_artifact_use_class": (
                compact_reducer_native_use_class
            ),
        }

    def reducer_progress_snapshot() -> dict[str, Any] | None:
        if reducer_sidecar is None:
            return None
        method = getattr(reducer_sidecar, "progress_snapshot", None)
        if not callable(method):
            return None
        try:
            value = method()
        except Exception:
            return None
        return dict(value) if isinstance(value, Mapping) else None

    def reduce_after_tail_timeout_v1() -> tuple[
        dict[str, Any], Mapping[str, Any] | None, float, int
    ]:
        """Finish a sealed capture locally when the observer misses its SLO.

        The producer has already returned successfully at this boundary.  A
        synchronous replay is therefore a reducer-only fallback: it does not
        relaunch the provider, alter the capture, or weaken the scientific
        contract.  The receipt records the missed asynchronous SLO separately.
        """

        if reducer_target is None or reducer_bundle is None or compact_root is None:
            raise CompactWorldAtlasReducerError(
                "tail-timeout fallback lacks its exact reducer target"
            )
        if (
            bound_native_artifact is None
            or bound_native_manifest is None
            or bound_native_manifest_sha256 is None
        ):
            raise CompactWorldAtlasReducerError(
                "tail-timeout fallback lacks native reducer identity"
            )
        fallback_started = time.monotonic()
        native_binding = load_native_compact_reducer_v1(
            bound_native_artifact,
            bound_native_manifest,
            require_production_qualified=(
                compact_reducer_native_use_class
                == NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1
            ),
            expected_manifest_sha256=bound_native_manifest_sha256,
        )
        fallback_reducer: CompactWorldAtlasIncrementalReducerV1 | None = None
        # Preserve the caller's configured parallelism when the asynchronous
        # observer misses its bounded tail SLO.  A serial fallback can turn a
        # successful B1024 producer into an hours-long reducer tail and is not
        # a faithful continuation of the requested execution contract.
        fallback_workers = max(1, reducer_workers)
        try:
            fallback_reducer = CompactWorldAtlasIncrementalReducerV1(
                reducer_bundle,
                max_workers=fallback_workers,
                native_artifact=native_binding,
                output_root=runtime,
            )
            atlas = fallback_reducer.finalize(
                compact_root / "capture-index-terminal.json"
            )
            strategy = fallback_reducer.strategy_attribution_result
        finally:
            if fallback_reducer is not None:
                fallback_reducer.close()
        return dict(atlas), strategy, time.monotonic() - fallback_started, fallback_workers

    reducer_completion: dict[str, Any] | None = None

    def await_reducer_before_lifecycle_closeout_v1() -> None:
        """Finish the consumer reducer before the producer seals its inventory."""

        nonlocal reducer_completion
        if reducer_sidecar is None or reducer_completion is not None:
            return
        tail_started = time.monotonic()
        try:
            result_bundle_method = getattr(
                reducer_sidecar, "result_bundle", None
            )
            if callable(result_bundle_method):
                bundle = dict(
                    result_bundle_method(
                        timeout_seconds=float(compact_reducer_tail_seconds),
                    )
                )
            else:
                atlas_result = reducer_sidecar.result(
                    timeout_seconds=float(compact_reducer_tail_seconds),
                )
                bundle = {
                    "schema_version": (
                        "eiel_native_compact_reducer_execution_bundle_v1"
                    ),
                    "atlas_result": atlas_result,
                    "strategy_attribution_result": getattr(
                        reducer_sidecar, "strategy_attribution_result", None
                    ),
                    "second_python_attribution_pass": False,
                    "operational_metrics": reducer_progress_snapshot(),
                }
        except CompactWorldAtlasReducerError as error:
            reducer_sidecar.cancel()
            tail_seconds = time.monotonic() - tail_started
            if "timed out" not in str(error):
                raise
            try:
                (
                    fallback_atlas_result,
                    fallback_strategy_attribution_result,
                    fallback_seconds,
                    fallback_workers,
                ) = reduce_after_tail_timeout_v1()
            except (
                CompactWorldAtlasReducerError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as fallback_error:
                raise CompactWorldAtlasReducerError(
                    f"{error}; automatic sealed-capture fallback failed: "
                    f"{fallback_error}"
                ) from fallback_error
            reducer_completion = {
                "status": "COMPLETED_AFTER_TAIL_FALLBACK",
                "tail_seconds": tail_seconds,
                "tail_slo_met": False,
                "tail_timeout_error": str(error),
                "fallback_reduction_seconds": fallback_seconds,
                "fallback_worker_count": fallback_workers,
                "fallback_reduction_completed": True,
                "bundle": {
                    "schema_version": (
                        "eiel_native_compact_reducer_execution_bundle_v1"
                    ),
                    "atlas_result": fallback_atlas_result,
                    "strategy_attribution_result": (
                        fallback_strategy_attribution_result
                    ),
                    "second_python_attribution_pass": False,
                    "operational_metrics": reducer_progress_snapshot(),
                },
            }
        else:
            reducer_completion = {
                "status": "COMPLETED",
                "tail_seconds": time.monotonic() - tail_started,
                "tail_slo_met": True,
                "bundle": bundle,
            }

    kwargs: dict[str, Any] = {
        "source_lineage": source_lineage,
        "science_lineage": science_lineage,
        "repo_root": repo_root,
    }
    if nested_storage_parent_run_root is not None:
        kwargs["nested_storage_parent_run_root"] = nested_storage_parent_run_root
    if adapter is not None:
        kwargs["cuda_full_evolution_v2_crca_adapter"] = adapter
        kwargs["cuda_full_evolution_v2_checkpoint_occurrence_adapter"] = adapter
    if using_canonical_front_door and reducer_sidecar is not None:
        kwargs["before_execute_fn"] = reducer_sidecar.start
        kwargs[
            "before_lifecycle_closeout_fn"
        ] = await_reducer_before_lifecycle_closeout_v1
    try:
        result = front_door(request, lowered, **kwargs)
    except BaseException:
        if reducer_sidecar is not None:
            reducer_sidecar.cancel()
        raise
    if not isinstance(result, Mapping):
        if reducer_sidecar is not None:
            reducer_sidecar.cancel()
        raise ScienceInstrumentOperatorError(
            "execution front door returned a non-object result"
        )
    # Keep the canonical front-door decision visible at the operator boundary.
    # In particular, a front door failure is a valid structured result (with
    # diagnostics and a receipt) and must not be mistaken for successful
    # operator invocation merely because the wrapper was constructed.
    front_door_result = dict(result)
    overall_ok = front_door_result.get("ok")
    overall_status = front_door_result.get("status")
    overall_result_class = front_door_result.get("result_class")
    host_observation_binding: dict[str, Any] | None = None
    if selected_world_host_observation and overall_ok is True:
        host_observation_binding = (
            _authenticated_selected_world_host_observation_binding_v1(
                front_door_result,
                runtime_root=runtime,
                expected_world_id=str(selected_world_ids[0]),
            )
        )
    if reducer_sidecar is not None:
        compact_reducer = {
            **compact_reducer,
            "sidecar_pid": reducer_sidecar.pid,
        }
        if reducer_completion is not None:
            bundle = reducer_completion.get("bundle")
            if not isinstance(bundle, Mapping):
                raise ScienceInstrumentOperatorError(
                    "pre-closeout reducer completion bundle is invalid"
                )
            completion_projection = {
                key: reducer_completion[key]
                for key in (
                    "status",
                    "tail_seconds",
                    "tail_slo_met",
                    "tail_timeout_error",
                    "fallback_reduction_seconds",
                    "fallback_worker_count",
                    "fallback_reduction_completed",
                )
                if key in reducer_completion
            }
            compact_reducer = {
                **compact_reducer,
                **completion_projection,
                "atlas_result": bundle.get("atlas_result"),
                "strategy_attribution_result": bundle.get(
                    "strategy_attribution_result"
                ),
                "second_python_attribution_pass": bundle.get(
                    "second_python_attribution_pass", False
                ),
                "operational_metrics": bundle.get("operational_metrics"),
                "producer_completed_before_tail_wait": True,
            }
        elif overall_ok is not True:
            reducer_sidecar.cancel()
            compact_reducer = {
                **compact_reducer,
                "status": "CANCELLED_AFTER_FRONT_DOOR_FAILURE",
                "operational_metrics": reducer_progress_snapshot(),
            }
        else:
            tail_started = time.monotonic()
            try:
                result_bundle_method = getattr(
                    reducer_sidecar, "result_bundle", None
                )
                if callable(result_bundle_method):
                    reducer_execution_bundle = result_bundle_method(
                        timeout_seconds=float(compact_reducer_tail_seconds),
                    )
                    atlas_result = reducer_execution_bundle["atlas_result"]
                    strategy_attribution_result = reducer_execution_bundle.get(
                        "strategy_attribution_result"
                    )
                    second_python_attribution_pass = reducer_execution_bundle.get(
                        "second_python_attribution_pass"
                    )
                    operational_metrics = reducer_execution_bundle.get(
                        "operational_metrics"
                    )
                else:
                    atlas_result = reducer_sidecar.result(
                        timeout_seconds=float(compact_reducer_tail_seconds),
                    )
                    strategy_attribution_result = getattr(
                        reducer_sidecar, "strategy_attribution_result", None
                    )
                    second_python_attribution_pass = False
                    operational_metrics = reducer_progress_snapshot()
            except CompactWorldAtlasReducerError as error:
                reducer_sidecar.cancel()
                tail_seconds = time.monotonic() - tail_started
                timed_out = "timed out" in str(error)
                if timed_out:
                    try:
                        (
                            fallback_atlas_result,
                            fallback_strategy_attribution_result,
                            fallback_seconds,
                            fallback_workers,
                        ) = reduce_after_tail_timeout_v1()
                    except (
                        CompactWorldAtlasReducerError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as fallback_error:
                        compact_reducer = {
                            **compact_reducer,
                            "status": "TAIL_TIMEOUT",
                            "tail_seconds": tail_seconds,
                            "tail_slo_met": False,
                            "error_type": type(error).__name__,
                            "error": (
                                f"{error}; automatic sealed-capture fallback "
                                f"failed: {fallback_error}"
                            ),
                            "fallback_error_type": type(fallback_error).__name__,
                            "operational_metrics": reducer_progress_snapshot(),
                        }
                        overall_ok = False
                        overall_status = "compact_reducer_tail_timeout"
                        overall_result_class = "REDUCER_TAIL_SLO_MISS"
                    else:
                        compact_reducer = {
                            **compact_reducer,
                            "status": "COMPLETED_AFTER_TAIL_FALLBACK",
                            "tail_seconds": tail_seconds,
                            "tail_slo_met": False,
                            "tail_timeout_error": str(error),
                            "fallback_reduction_seconds": fallback_seconds,
                            "fallback_worker_count": fallback_workers,
                            "fallback_reduction_completed": True,
                            "atlas_result": fallback_atlas_result,
                            "strategy_attribution_result": (
                                fallback_strategy_attribution_result
                            ),
                            "second_python_attribution_pass": False,
                            "operational_metrics": reducer_progress_snapshot(),
                            "producer_completed_before_tail_wait": True,
                        }
                else:
                    compact_reducer = {
                        **compact_reducer,
                        "status": "FAILED",
                        "tail_seconds": tail_seconds,
                        "tail_slo_met": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "operational_metrics": reducer_progress_snapshot(),
                    }
                    overall_ok = False
                    overall_status = "compact_reducer_failed"
                    overall_result_class = "REDUCER_ERROR"
            else:
                tail_seconds = time.monotonic() - tail_started
                compact_reducer = {
                    **compact_reducer,
                    "status": "COMPLETED",
                    "tail_seconds": tail_seconds,
                    "tail_slo_met": True,
                    "atlas_result": atlas_result,
                    "strategy_attribution_result": strategy_attribution_result,
                    "second_python_attribution_pass": (
                        second_python_attribution_pass
                    ),
                    "operational_metrics": operational_metrics,
                    "producer_completed_before_tail_wait": True,
                }
        if reducer_sidecar is not None and not _front_door_result_is_resumable_wait_v1(
            front_door_result
        ):
            try:
                reducer_receipt_core = (
                    _compact_reducer_execution_receipt_core_v1(
                        compact_reducer=compact_reducer,
                        front_door_result=front_door_result,
                        batch_index=batch_index,
                        runtime_root=runtime,
                        science_use_class=str(science_use_class),
                    )
                )
                reducer_receipt_artifact = _write_canonical_json(
                    runtime / "compact_reducer_execution_receipt.json",
                    reducer_receipt_core,
                    name="compact reducer execution receipt",
                )
            except (OSError, TypeError, ValueError) as error:
                raise ScienceInstrumentOperatorError(
                    f"cannot persist compact reducer execution receipt: {error}"
                ) from error
            compact_reducer = {
                **compact_reducer,
                "execution_receipt_path": reducer_receipt_artifact["path"],
                "execution_receipt_file_sha256": reducer_receipt_artifact[
                    "file_sha256"
                ],
            }
    return {
        "schema_version": "eiel_science_instrument_execution_result_v1",
        "operator_schema_version": SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION,
        "engineering_only": (
            science_use_class != SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
        ),
        "scientific_execution": (
            science_use_class == SCIENCE_CUDA_V2_USE_CLASS_SCIENTIFIC_EXECUTION_V1
        ),
        "prequal_started": False,
        "qualification_started": False,
        "batch_index": batch_index,
        "runtime_root": str(runtime),
        "source_lineage": source_lineage,
        "science_lineage": science_lineage,
        "ok": overall_ok,
        "status": overall_status,
        "result_class": overall_result_class,
        "front_door_result": front_door_result,
        "compact_reducer": compact_reducer,
        **(
            {"selected_world_host_observation": host_observation_binding}
            if host_observation_binding is not None
            else {}
        ),
    }


@dataclass(frozen=True, slots=True)
class _LoadedScienceBatchReceiptV1:
    """The small, identity-bearing projection needed by offline reduction."""

    path: Path
    file_sha256: str
    batch_id: str
    cell_ids: tuple[str, ...]
    plan_sha256: str
    execution_route_sha256: str | None
    records_path: Path
    records_sha256: str
    next_absolute_tick: int
    frontier: dict[str, Any]
    result: dict[str, Any]
    wrapper_path: Path | None = None


def _input_sequence(
    value: JsonInputV1 | Sequence[JsonInputV1] | None,
    *,
    name: str,
    required: bool = True,
) -> tuple[_LoadedJsonV1, ...]:
    """Normalize one path/object or a sequence without treating strings as lists."""

    if value is None:
        if required:
            raise ScienceInstrumentOperatorError(f"{name} is required")
        return ()
    if isinstance(value, (str, Path, Mapping)):
        values: Sequence[JsonInputV1] = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ScienceInstrumentOperatorError(f"{name} must be a JSON input or sequence")
    if not values and required:
        raise ScienceInstrumentOperatorError(f"{name} cannot be empty")
    loaded: list[_LoadedJsonV1] = []
    for index, item in enumerate(values):
        if not isinstance(item, (str, Path, Mapping)):
            raise ScienceInstrumentOperatorError(
                f"{name}[{index}] must be a path or JSON object"
            )
        loaded.append(_read_json_input(item, name=f"{name}[{index}]"))
    return tuple(loaded)


def _resolve_receipt_path(raw: object, *, anchor: Path | None, field: str) -> Path:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise ScienceInstrumentOperatorError(f"{field} must be a non-empty path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        if anchor is None:
            raise ScienceInstrumentOperatorError(
                f"{field} must be absolute when its receipt is not file-backed"
            )
        path = anchor / path
    if path.is_symlink():
        raise ScienceInstrumentOperatorError(f"{field} must not be a symbolic link")
    return path.resolve()


def _load_completed_science_batch_receipt(
    value: JsonInputV1,
    *,
    index: int,
) -> _LoadedScienceBatchReceiptV1:
    """Load one completed batch manifest (or one cell-completion pointer)."""

    loaded = _read_json_input(value, name=f"completed batch receipt {index}")
    wrapper_path: Path | None = None
    body = loaded.value
    if body.get("schema_version") == "eiel_campaign_cell_completion_v1":
        wrapper_path = loaded.path
        if body.get("status") != "completed":
            raise ScienceInstrumentOperatorError(
                f"completed batch receipt {index} cell completion is not completed"
            )
        batch_path = _resolve_receipt_path(
            body.get("batch_manifest_path"),
            anchor=None if loaded.path is None else loaded.path.parent,
            field=f"completed batch receipt {index}.batch_manifest_path",
        )
        loaded = _read_json_input(
            batch_path, name=f"completed batch receipt {index} batch manifest"
        )
        body = loaded.value
        if body.get("schema_version") != "eiel_campaign_batch_manifest_v1":
            raise ScienceInstrumentOperatorError(
                f"completed batch receipt {index} pointer does not name a batch manifest"
            )
    if body.get("schema_version") != "eiel_campaign_batch_manifest_v1":
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} must use eiel_campaign_batch_manifest_v1"
        )
    if body.get("status") != "completed":
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} is not terminal completed"
        )
    batch_id = body.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id or batch_id.strip() != batch_id:
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index}.batch_id is invalid"
        )
    raw_cell_ids = body.get("cell_ids")
    if (
        not isinstance(raw_cell_ids, list)
        or not raw_cell_ids
        or any(not isinstance(item, str) or not item for item in raw_cell_ids)
        or len(raw_cell_ids) != len(set(raw_cell_ids))
    ):
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index}.cell_ids are invalid"
        )
    cell_ids = tuple(raw_cell_ids)
    plan_sha256 = _as_sha256(
        body.get("plan_sha256"), field=f"completed batch receipt {index}.plan_sha256"
    )
    route_value = body.get("execution_route_sha256")
    route_sha256 = (
        None
        if route_value is None
        else _as_sha256(
            route_value,
            field=f"completed batch receipt {index}.execution_route_sha256",
        )
    )
    raw_result = body.get("result")
    if not isinstance(raw_result, Mapping):
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index}.result is absent"
        )
    result = dict(raw_result)
    if result.get("technical_only") is not True:
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} is not technical-only"
        )
    if result.get("formal_qualification_authority") not in (None, False):
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} claims Qualification authority"
        )
    if result.get("time_authority") not in (None, "absolute_tick_v1"):
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} does not use absolute-tick authority"
        )
    raw_frontier = result.get("science_observation_frontier")
    if raw_frontier is None:
        raw_frontier = body.get("science_observation_frontier")
    if not isinstance(raw_frontier, Mapping):
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} lacks science observation frontier"
        )
    frontier = dict(raw_frontier)
    if frontier.get("schema_version") != (
        "eiel_cuda_full_evolution_science_observation_frontier_v1"
    ):
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} frontier schema is unsupported"
        )
    if frontier.get("observation_mode") != SCIENCE_NATIVE_SIDECAR_MODE_V1:
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} frontier observation mode differs"
        )
    if frontier.get("observation_abi") != SCIENCE_NATIVE_SIDECAR_OBSERVATION_ABI_V1:
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} frontier observation ABI differs"
        )
    next_tick = frontier.get("next_absolute_tick")
    if type(next_tick) is not int or next_tick < 0:
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} frontier absolute tick is invalid"
        )
    if frontier.get("terminal") is not True:
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} frontier is not terminal"
        )
    records_path = _resolve_receipt_path(
        frontier.get("records_path"),
        anchor=None if loaded.path is None else loaded.path.parent,
        field=f"completed batch receipt {index}.frontier.records_path",
    )
    records_sha256 = _as_sha256(
        frontier.get("records_sha256"),
        field=f"completed batch receipt {index}.frontier.records_sha256",
    )
    counts = result.get("science_observation_record_counts")
    if counts is not None and (
        not isinstance(counts, Mapping)
        or any(
            not isinstance(key, str) or type(item) is not int or item < 0
            for key, item in counts.items()
        )
    ):
        raise ScienceInstrumentOperatorError(
            f"completed batch receipt {index} observation counts are invalid"
        )
    if wrapper_path is not None:
        wrapper = _read_regular_json(
            wrapper_path,
            name=f"completed batch receipt {index} cell completion",
            canonical=False,
        )
        wrapper_cell = wrapper.get("cell_id")
        if wrapper_cell is not None and wrapper_cell not in cell_ids:
            raise ScienceInstrumentOperatorError(
                f"completed batch receipt {index} cell completion names another cell"
            )
        wrapper_tick = wrapper.get("terminal_checkpoint_tick")
        if wrapper_tick is not None and wrapper_tick != next_tick:
            raise ScienceInstrumentOperatorError(
                f"completed batch receipt {index} completion tick differs from frontier"
            )
        wrapper_plan = wrapper.get("plan_sha256")
        if (
            wrapper_plan is not None
            and _as_sha256(
                wrapper_plan,
                field=f"completed batch receipt {index}.completion.plan_sha256",
            )
            != plan_sha256
        ):
            raise ScienceInstrumentOperatorError(
                f"completed batch receipt {index} completion plan identity differs"
            )
    return _LoadedScienceBatchReceiptV1(
        path=(
            loaded.path
            if loaded.path is not None
            else Path(f"<in-memory-batch-{index}>")
        ),
        file_sha256=loaded.file_sha256,
        batch_id=batch_id,
        cell_ids=cell_ids,
        plan_sha256=plan_sha256,
        execution_route_sha256=route_sha256,
        records_path=records_path,
        records_sha256=records_sha256,
        next_absolute_tick=next_tick,
        frontier=frontier,
        result=result,
        wrapper_path=wrapper_path,
    )


def _phase_registry(
    value: Mapping[str, int] | str | Path | None,
) -> dict[str, int]:
    if value is None:
        from engine.native_runtime.cuda_v2_science_capture_v1 import (
            science_capture_producer_phase_registry_v1,
        )

        return dict(science_capture_producer_phase_registry_v1())
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        raw = _read_json_input(value, name="producer phase registry").value
        if "producer_phase_registry_ordinals" in raw:
            nested = raw.get("producer_phase_registry_ordinals")
            if not isinstance(nested, Mapping):
                raise ScienceInstrumentOperatorError(
                    "producer phase registry nested mapping is invalid"
                )
            raw = dict(nested)
    if not raw or any(
        not isinstance(key, str) or not key or type(ordinal) is not int or ordinal < 0
        for key, ordinal in raw.items()
    ):
        raise ScienceInstrumentOperatorError("producer phase registry is invalid")
    values = list(raw.values())
    if len(values) != len(set(values)):
        raise ScienceInstrumentOperatorError(
            "producer phase registry ordinals duplicate"
        )
    return {key: int(value) for key, value in sorted(raw.items())}


def _digest_object(value: object, *, field: str) -> ContentDigest:
    return ContentDigest("sha256", _as_sha256(value, field=field))


def _canonical_digest_of_without_identity(
    value: Mapping[str, Any], *, identity_key: str, field: str
) -> str:
    declared = _as_sha256(value.get(identity_key), field=f"{field}.{identity_key}")
    core = {key: item for key, item in value.items() if key != identity_key}
    observed = canonical_sha256(core).hex
    if observed != declared:
        raise ScienceInstrumentOperatorError(
            f"{field} identity does not match canonical content"
        )
    return declared


def _payload_mapping(value: object, *, field: str) -> dict[str, Any]:
    if hasattr(value, "to_value"):
        value = value.to_value()
    if not isinstance(value, Mapping):
        raise ScienceInstrumentOperatorError(f"{field} is not an object")
    return dict(value)


def _expected_science_batch_plan_sha256(
    lowered: Mapping[str, Any], receipt: _LoadedScienceBatchReceiptV1
) -> str:
    """Return the only plan identity a completed batch may claim.

    Materialized plans are content addressed before execution.  The canonical
    Launchpad front door then binds the plan's runtime root, so real batch
    manifests under ``<run-root>/batches/<batch>/manifest.json`` carry that
    bound identity.  The small in-memory/direct manifest shape used by offline
    engineering fixtures carries the unbound materialized identity.  Accept
    exactly the identity appropriate to the receipt shape; never accept an
    arbitrary receipt plan hash.
    """

    lowered_plan_sha256 = _as_sha256(
        lowered.get("plan_sha256"), field="lowered plan.plan_sha256"
    )
    path = receipt.path
    if (
        path.name == "manifest.json"
        and len(path.parents) >= 3
        and path.parents[1].name == "batches"
    ):
        # path = run_root/batches/<batch>/manifest.json
        expected = _bound_plan_sha256(lowered, path.parents[2])
    else:
        expected = lowered_plan_sha256
    if receipt.plan_sha256 != expected:
        raise ScienceInstrumentOperatorError(
            "completed batch plan identity differs from its materialized lowered plan"
        )
    return expected


def _science_observation_contract_for_cell(
    cell: Mapping[str, Any],
    lowered: Mapping[str, Any],
    sidecar: NativeScienceSidecarV1,
    *,
    cell_id: str,
) -> tuple[ScienceObservationProfileV1, SchemaVersion]:
    """Resolve the profile/ABI from the materialized world acquisition.

    The world acquisition contract owns the profile selection.  The lowered
    acquisition projection records the concrete ABI selected for that
    acquisition, and the sealed sidecar must report the same ABI.  This keeps
    reducers independent of a process-wide/default profile while remaining
    compatible with the current V1 acquisition payload (which predates the
    optional explicit ABI field).
    """

    acquisition = _payload_mapping(
        cell.get("science_acquisition"),
        field=f"lowered world {cell_id}.science_acquisition",
    )
    raw_profile = acquisition.get("profile_id")
    try:
        profile = ScienceObservationProfileV1(raw_profile)
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"lowered world {cell_id} acquisition profile is invalid"
        ) from error

    materialization = lowered.get("science_materialization")
    if not isinstance(materialization, Mapping):
        raise ScienceInstrumentOperatorError(
            f"lowered batch for {cell_id} lacks science materialization metadata"
        )
    raw_materialized_abi = materialization.get("observation_abi")
    if not isinstance(raw_materialized_abi, str) or not raw_materialized_abi:
        raise ScienceInstrumentOperatorError(
            f"lowered world {cell_id} lacks an explicit observation ABI"
        )
    try:
        materialized_abi = SchemaVersion(raw_materialized_abi)
        sidecar_abi = SchemaVersion(str(sidecar.frontier["observation_abi"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"world {cell_id} observation ABI is invalid"
        ) from error
    if sidecar_abi != materialized_abi:
        raise ScienceInstrumentOperatorError(
            f"sidecar observation ABI differs from world {cell_id} acquisition"
        )

    # Newer acquisition contracts may carry the ABI directly.  If present,
    # it is authoritative and must agree with both the lowered projection and
    # the sidecar.  Existing V1 bundles resolve it through the lowered
    # acquisition projection above.
    raw_acquisition_abi = acquisition.get("observation_abi_version")
    if raw_acquisition_abi is None:
        raw_acquisition_abi = acquisition.get("observation_abi")
    if raw_acquisition_abi is not None:
        try:
            acquisition_abi = SchemaVersion(str(raw_acquisition_abi))
        except (TypeError, ValueError) as error:
            raise ScienceInstrumentOperatorError(
                f"world {cell_id} acquisition observation ABI is invalid"
            ) from error
        if acquisition_abi != materialized_abi:
            raise ScienceInstrumentOperatorError(
                f"world {cell_id} acquisition observation ABI differs from its lowered ABI"
            )
    return profile, materialized_abi


def _validate_sidecar_world_binding(
    sidecar: NativeScienceSidecarV1,
    cell: Mapping[str, Any],
    lowered: Mapping[str, Any],
    science_plan: ScienceExecutionPlanV1,
    *,
    cell_id: str,
) -> tuple[ScienceObservationProfileV1, SchemaVersion]:
    """Bind every canonical sidecar row to one materialized world contract."""

    rows = sidecar.records_for_cell(cell_id)
    if not rows:
        raise ScienceInstrumentOperatorError(
            f"MUTSUP sidecar has no canonical rows for {cell_id}"
        )
    expected_manifest = science_plan.instrument_manifest_sha256
    expected_crca = science_plan.crca_semantic_run_digest
    try:
        cell_manifest = _digest_object(
            cell.get("science_instrument_manifest_sha256"),
            field=f"lowered world {cell_id}.science_instrument_manifest_sha256",
        )
        cell_crca = _digest_object(
            cell.get("science_crca_semantic_run_digest"),
            field=f"lowered world {cell_id}.science_crca_semantic_run_digest",
        )
        expected_acquisition = _digest_object(
            cell.get("science_acquisition_sha256"),
            field=f"lowered world {cell_id}.science_acquisition_sha256",
        )
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentOperatorError(
            f"lowered world {cell_id} observation identity is invalid"
        ) from error
    if cell_manifest != expected_manifest:
        raise ScienceInstrumentOperatorError(
            f"lowered world {cell_id} instrument manifest differs from execution plan"
        )
    if cell_crca != expected_crca:
        raise ScienceInstrumentOperatorError(
            f"lowered world {cell_id} CRCA semantic digest differs from execution plan"
        )
    logical_world_id = cell.get("science_logical_world_id")
    if type(logical_world_id) is not int or logical_world_id < 0:
        raise ScienceInstrumentOperatorError(
            f"lowered world {cell_id} logical science identity is invalid"
        )
    try:
        expected_native_control = validate_science_native_control_receipt_v1(
            cell.get("native_control"), field=f"lowered world {cell_id}.native_control"
        )
    except Exception as error:
        raise ScienceInstrumentOperatorError(
            f"lowered world {cell_id} native control is invalid"
        ) from error
    profile, observation_abi = _science_observation_contract_for_cell(
        cell, lowered, sidecar, cell_id=cell_id
    )

    control_rows = tuple(
        row for row in rows if row.record_kind is ScienceRecordKindV1.CONTROL_BINDING
    )
    if len(control_rows) != 1:
        raise ScienceInstrumentOperatorError(
            f"sidecar must contain exactly one control binding for {cell_id}"
        )
    control_payload = _payload_mapping(
        control_rows[0].payload,
        field=f"sidecar[{cell_id}].control_binding.payload",
    )
    registered_profiles = control_payload.get("registered_profile_ids")
    if registered_profiles != [profile.value]:
        raise ScienceInstrumentOperatorError(
            f"sidecar profile binding differs from world {cell_id} acquisition"
        )
    try:
        applied_native_control = validate_science_native_control_receipt_v1(
            control_payload.get("applied_native_control"),
            field=f"sidecar[{cell_id}].applied_native_control",
        )
    except Exception as error:
        raise ScienceInstrumentOperatorError(
            f"sidecar native control binding is invalid for {cell_id}"
        ) from error
    if applied_native_control != expected_native_control:
        raise ScienceInstrumentOperatorError(
            f"sidecar native control differs from materialized world {cell_id}"
        )

    expected_semantic_execution = derive_science_semantic_execution_id_v1(
        expected_crca, expected_acquisition
    )
    for row in rows:
        if row.instrument_manifest_sha256 != expected_manifest:
            raise ScienceInstrumentOperatorError(
                f"sidecar instrument manifest differs for {cell_id}"
            )
        if row.science_semantic_acquisition_sha256 != expected_acquisition:
            raise ScienceInstrumentOperatorError(
                f"sidecar acquisition identity differs for {cell_id}"
            )
        if row.crca_semantic_run_digest != expected_crca:
            raise ScienceInstrumentOperatorError(
                f"sidecar CRCA semantic digest differs for {cell_id}"
            )
        if row.logical_world_id != logical_world_id:
            raise ScienceInstrumentOperatorError(
                f"sidecar logical world identity differs for {cell_id}"
            )
        if row.semantic_execution_id != expected_semantic_execution:
            raise ScienceInstrumentOperatorError(
                f"sidecar semantic execution identity differs for {cell_id}"
            )
    return profile, observation_abi


def _materialization_reduction_inputs(
    materialization: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    ScienceWorldBundleV1,
    ScienceExecutionPlanV1,
    tuple[dict[str, Any], ...],
]:
    root, receipt, _normalized, lowered_batches = (
        load_science_instrument_materialization_v1(materialization)
    )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ScienceInstrumentOperatorError("materialization artifacts are absent")
    compilation = _verify_materialization_artifact(
        root, artifacts.get("compilation_bundle"), name="compilation bundle"
    )
    bundle_raw = _verify_materialization_artifact(
        root, artifacts.get("world_bundle"), name="world bundle"
    )
    admission = _verify_materialization_artifact(
        root, artifacts.get("bundle_admission"), name="bundle admission"
    )
    carrier = _verify_materialization_artifact(
        root, artifacts.get("carrier_plan"), name="carrier plan"
    )
    bundle_info = artifacts.get("world_bundle")
    if not isinstance(bundle_info, Mapping) or not isinstance(
        bundle_info.get("path"), str
    ):
        raise ScienceInstrumentOperatorError("world bundle artifact path is absent")
    try:
        bundle = load_science_world_bundle_v1(Path(bundle_info["path"]))
        science_plan = ScienceExecutionPlanV1.from_dict(
            _payload_mapping(
                compilation.get("bundle_core"), field="compilation.bundle_core"
            )["execution_plan"]
        )
    except (ScienceWorldBundleError, TypeError, ValueError, KeyError) as error:
        raise ScienceInstrumentOperatorError(
            f"materialization reduction identity is invalid: {error}"
        ) from error
    if bundle_raw != bundle.to_dict():
        raise ScienceInstrumentOperatorError(
            "world bundle artifact bytes differ from parsed bundle"
        )
    _canonical_digest_of_without_identity(
        compilation,
        identity_key="compilation_bundle_sha256",
        field="compilation bundle",
    )
    _canonical_digest_of_without_identity(
        carrier,
        identity_key="physical_execution_plan_sha256",
        field="carrier plan",
    )
    if carrier.get("fallback_policy") != "FORBIDDEN":
        raise ScienceInstrumentOperatorError(
            "carrier plan fallback policy is not FORBIDDEN"
        )
    if admission.get("outcome") != ScienceAdmissionOutcomeV1.ADMITTED.value:
        raise ScienceInstrumentOperatorError("materialized bundle is not admitted")
    if (
        _digest_object(
            admission.get("bundle_content_sha256"),
            field="bundle admission.bundle_content_sha256",
        )
        != bundle.bundle_content_sha256()
    ):
        raise ScienceInstrumentOperatorError("bundle admission binds another bundle")
    if tuple(admission.get("admitted_world_ids", ())) != tuple(
        sorted(world.world_id for world in bundle.worlds)
    ):
        raise ScienceInstrumentOperatorError(
            "bundle admission world set differs from materialized bundle"
        )
    identities = receipt.get("identities")
    if not isinstance(identities, Mapping):
        raise ScienceInstrumentOperatorError("materialization identities are absent")
    if (
        _digest_object(
            identities.get("bundle_content_sha256"),
            field="identities.bundle_content_sha256",
        )
        != bundle.bundle_content_sha256()
    ):
        raise ScienceInstrumentOperatorError("materialization bundle identity differs")
    if _digest_object(
        identities.get("compilation_bundle_sha256"),
        field="identities.compilation_bundle_sha256",
    ) != _digest_object(
        compilation.get("compilation_bundle_sha256"), field="compilation_bundle_sha256"
    ):
        raise ScienceInstrumentOperatorError(
            "materialization compilation identity differs"
        )
    if _digest_object(
        identities.get("carrier_plan_sha256"), field="identities.carrier_plan_sha256"
    ) != _digest_object(
        carrier.get("physical_execution_plan_sha256"),
        field="carrier physical execution identity",
    ):
        raise ScienceInstrumentOperatorError("materialization carrier identity differs")
    compilation_core = _payload_mapping(
        compilation.get("bundle_core"), field="compilation.bundle_core"
    )
    if _digest_object(
        compilation_core.get("campaign_spec_sha256"),
        field="compilation campaign identity",
    ) != _digest_object(
        identities.get("campaign_spec_sha256"),
        field="identities.campaign_spec_sha256",
    ):
        raise ScienceInstrumentOperatorError("compilation campaign identity differs")
    if science_plan.instrument_manifest_sha256 != _digest_object(
        compilation_core["execution_plan"]["execution_core"][
            "instrument_manifest_sha256"
        ],
        field="execution plan instrument identity",
    ):
        raise ScienceInstrumentOperatorError(
            "execution plan instrument identity differs"
        )
    lower_ids: set[str] = set()
    for index, lowered in enumerate(lowered_batches):
        if (
            lowered.get("science_use_class")
            != SCIENCE_CUDA_V2_USE_CLASS_EXPLORATORY_SCIENCE_V1
        ):
            raise ScienceInstrumentOperatorError(
                f"lowered batch {index} is not exploratory engineering science"
            )
        meta = lowered.get("science_materialization")
        if not isinstance(meta, Mapping):
            raise ScienceInstrumentOperatorError(
                f"lowered batch {index} lacks science materialization metadata"
            )
        if meta.get("observation_mode") != SCIENCE_NATIVE_SIDECAR_MODE_V1:
            raise ScienceInstrumentOperatorError(
                f"lowered batch {index} does not use typed sidecar observations"
            )
        if meta.get("fallback_policy") != "FORBIDDEN":
            raise ScienceInstrumentOperatorError(
                f"lowered batch {index} fallback policy is not FORBIDDEN"
            )
        ids = meta.get("semantic_world_ids")
        cells = lowered.get("cells")
        if (
            not isinstance(ids, list)
            or not isinstance(cells, list)
            or tuple(ids)
            != tuple(cell.get("cell_id") for cell in cells if isinstance(cell, Mapping))
            or len(ids) != len(set(ids))
        ):
            raise ScienceInstrumentOperatorError(
                f"lowered batch {index} semantic world set is invalid"
            )
        if lower_ids.intersection(ids):
            raise ScienceInstrumentOperatorError(
                "materialized lowered batches duplicate worlds"
            )
        lower_ids.update(ids)
        if _digest_object(
            meta.get("physical_execution_plan_sha256"),
            field=f"lowered batch {index}.physical_execution_plan_sha256",
        ) != _digest_object(
            carrier.get("physical_execution_plan_sha256"),
            field="carrier physical execution identity",
        ):
            raise ScienceInstrumentOperatorError(
                f"lowered batch {index} binds another physical carrier plan"
            )
    if lower_ids != {world.world_id for world in bundle.worlds}:
        raise ScienceInstrumentOperatorError(
            "lowered batches do not exactly cover the bundle worlds"
        )
    return (
        root,
        receipt,
        compilation,
        carrier,
        admission,
        bundle,
        science_plan,
        lowered_batches,
    )


def _materialized_mutsup_cells(
    bundle: ScienceWorldBundleV1,
    carrier: Mapping[str, Any],
    lowered_batches: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    """Verify the semantic/carrying projections and return cells by world ID."""

    world_by_id = {world.world_id: world for world in bundle.worlds}
    if len(world_by_id) != len(bundle.worlds) or not world_by_id:
        raise ScienceInstrumentOperatorError(
            "materialized bundle world identities are invalid"
        )
    for world in bundle.worlds:
        science = _payload_mapping(world.science, field=f"{world.world_id}.science")
        if (
            science.get("test_id") != "MUTSUP"
            or str(science.get("test_version")) != "1"
        ):
            raise ScienceInstrumentOperatorError(
                f"reduce-mutsup requires only MUTSUP_V1 worlds; got {world.world_id}"
            )

    mapping_by_id: dict[str, Mapping[str, Any]] = {}
    raw_batches = carrier.get("batches")
    if not isinstance(raw_batches, list) or len(raw_batches) != len(lowered_batches):
        raise ScienceInstrumentOperatorError("carrier and lowered batch counts differ")
    for batch_index, raw_batch in enumerate(raw_batches):
        if not isinstance(raw_batch, Mapping):
            raise ScienceInstrumentOperatorError(
                f"carrier batch {batch_index} is invalid"
            )
        if raw_batch.get("physical_batch_index") != batch_index:
            raise ScienceInstrumentOperatorError(
                "carrier batch indices are not canonical"
            )
        slots = raw_batch.get("world_slots")
        if not isinstance(slots, list) or not slots:
            raise ScienceInstrumentOperatorError(
                f"carrier batch {batch_index} has no world slots"
            )
        for slot_index, slot in enumerate(slots):
            if not isinstance(slot, Mapping) or slot.get("physical_slot") != slot_index:
                raise ScienceInstrumentOperatorError(
                    f"carrier batch {batch_index} physical slots are invalid"
                )
            mapping = slot.get("world")
            if not isinstance(mapping, Mapping):
                raise ScienceInstrumentOperatorError(
                    f"carrier batch {batch_index} world mapping is invalid"
                )
            world_id = mapping.get("world_id")
            if not isinstance(world_id, str) or world_id in mapping_by_id:
                raise ScienceInstrumentOperatorError(
                    f"carrier batch {batch_index} duplicates or lacks world identity"
                )
            if world_id not in world_by_id:
                raise ScienceInstrumentOperatorError(
                    f"carrier batch names unknown materialized world {world_id}"
                )
            mapping_by_id[world_id] = mapping
    if set(mapping_by_id) != set(world_by_id):
        raise ScienceInstrumentOperatorError(
            "carrier plan does not cover the MUTSUP bundle"
        )

    cells_by_id: dict[str, dict[str, Any]] = {}
    for batch_index, lowered in enumerate(lowered_batches):
        cells = lowered.get("cells")
        meta = lowered.get("science_materialization")
        if not isinstance(cells, list) or not isinstance(meta, Mapping):
            raise ScienceInstrumentOperatorError(
                f"lowered batch {batch_index} cells are invalid"
            )
        expected_ids = meta.get("semantic_world_ids")
        if not isinstance(expected_ids, list):
            raise ScienceInstrumentOperatorError(
                f"lowered batch {batch_index} semantic world IDs are absent"
            )
        actual_ids: list[str] = []
        for cell_index, raw_cell in enumerate(cells):
            if not isinstance(raw_cell, Mapping):
                raise ScienceInstrumentOperatorError(
                    f"lowered batch {batch_index} cell {cell_index} is invalid"
                )
            cell = dict(raw_cell)
            cell_id = cell.get("cell_id")
            if not isinstance(cell_id, str) or not cell_id:
                raise ScienceInstrumentOperatorError(
                    f"lowered batch {batch_index} cell identity is invalid"
                )
            if cell_id in cells_by_id:
                raise ScienceInstrumentOperatorError(
                    f"duplicate lowered world {cell_id}"
                )
            actual_ids.append(cell_id)
            world = world_by_id.get(cell_id)
            if world is None:
                raise ScienceInstrumentOperatorError(
                    f"lowered batch names unknown materialized world {cell_id}"
                )
            mapping = mapping_by_id.get(cell_id)
            if mapping is None:
                raise ScienceInstrumentOperatorError(
                    f"carrier mapping is missing lowered world {cell_id}"
                )
            expected_semantic = world.semantic_world_sha256.to_dict()
            expected_acquisition = world.acquisition_sha256.to_dict()
            if cell.get("science_world_sha256") != expected_semantic:
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} semantic identity differs"
                )
            if cell.get("science_acquisition_sha256") != expected_acquisition:
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} acquisition identity differs"
                )
            if mapping.get("semantic_world_sha256") != expected_semantic:
                raise ScienceInstrumentOperatorError(
                    f"carrier world {cell_id} semantic identity differs"
                )
            if mapping.get("acquisition_sha256") != expected_acquisition:
                raise ScienceInstrumentOperatorError(
                    f"carrier world {cell_id} acquisition identity differs"
                )
            logical_id = cell.get("science_logical_world_id")
            if type(logical_id) is not int or logical_id < 0:
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} logical science identity is invalid"
                )
            if mapping.get("world_uid") != logical_id:
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} logical identity differs from carrier mapping"
                )
            if cell.get("science_ownership") != world.science.to_value():
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} science ownership differs"
                )
            if cell.get("science_acquisition") != world.acquisition.to_value():
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} acquisition contract differs"
                )
            native_control = cell.get("native_control")
            if not isinstance(native_control, Mapping):
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} lacks exact native control"
                )
            control_vector = native_control.get("control_vector")
            if (
                not isinstance(control_vector, Mapping)
                or set(control_vector) != {"mutation.enabled"}
                or type(control_vector.get("mutation.enabled")) is not bool
            ):
                raise ScienceInstrumentOperatorError(
                    f"lowered world {cell_id} native mutation control is invalid"
                )
            cell["_batch_index"] = batch_index
            cells_by_id[cell_id] = cell
        if tuple(expected_ids) != tuple(actual_ids):
            raise ScienceInstrumentOperatorError(
                f"lowered batch {batch_index} semantic world order differs"
            )
    if set(cells_by_id) != set(world_by_id):
        raise ScienceInstrumentOperatorError(
            "lowered cells do not exactly cover materialized worlds"
        )
    return cells_by_id


def _resolve_mutsup_world_reference(
    value: object,
    *,
    field: str,
    cells_by_id: Mapping[str, Mapping[str, Any]],
    semantic_by_hex: Mapping[str, str],
    physical_by_hex: Mapping[str, str],
) -> str:
    """Resolve an explicit owner/pair reference to a materialized cell."""

    if isinstance(value, Mapping):
        if set(value) == {"algorithm", "hex"}:
            digest = _as_sha256(value, field=field)
            if digest in semantic_by_hex:
                return semantic_by_hex[digest]
            if digest in physical_by_hex:
                return physical_by_hex[digest]
            raise ScienceInstrumentOperatorError(
                f"{field} does not identify a materialized or observed world"
            )
        if "world_uid" in value:
            return _resolve_mutsup_world_reference(
                value["world_uid"],
                field=f"{field}.world_uid",
                cells_by_id=cells_by_id,
                semantic_by_hex=semantic_by_hex,
                physical_by_hex=physical_by_hex,
            )
        for key in ("world_id", "cell_id", "world_key"):
            if key in value:
                return _resolve_mutsup_world_reference(
                    value[key],
                    field=f"{field}.{key}",
                    cells_by_id=cells_by_id,
                    semantic_by_hex=semantic_by_hex,
                    physical_by_hex=physical_by_hex,
                )
        raise ScienceInstrumentOperatorError(f"{field} does not identify a world")
    if isinstance(value, str) and value.strip() == value and value:
        if value in cells_by_id:
            return value
        if value in semantic_by_hex:
            return semantic_by_hex[value]
        if value in physical_by_hex:
            return physical_by_hex[value]
    raise ScienceInstrumentOperatorError(
        f"{field} does not identify a materialized world"
    )


def _owner_rows_from_configuration(
    raw: object,
    *,
    field: str,
) -> list[tuple[str | None, dict[str, Any]]]:
    if isinstance(raw, list):
        result: list[tuple[str | None, dict[str, Any]]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ScienceInstrumentOperatorError(
                    f"{field}[{index}] is not an object"
                )
            result.append((None, dict(item)))
        return result
    if isinstance(raw, Mapping):
        data = dict(raw)
        if {"study_id", "arm_id", "replicate"}.issubset(data):
            return [(None, data)]
        result = []
        for alias, item in data.items():
            if not isinstance(item, Mapping):
                raise ScienceInstrumentOperatorError(
                    f"{field}[{alias}] is not an object"
                )
            result.append((alias, dict(item)))
        return result
    raise ScienceInstrumentOperatorError(f"{field} must be an array or object")


def _nonempty_mutsup_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ScienceInstrumentOperatorError(
            f"{field} must be a non-empty exact string"
        )
    return value


def _normalise_mutsup_configuration(
    configuration: Mapping[str, Any],
    *,
    cells_by_id: Mapping[str, Mapping[str, Any]],
    semantic_by_hex: Mapping[str, str],
    physical_by_hex: Mapping[str, str],
    physical_world_uid_by_cell: Mapping[str, ContentDigest],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Bind authored ownership/pairs to the observed physical world UIDs.

    The host configuration may name semantic world digests (which are stable
    across carriers), while the reducer receives runtime world UIDs from the
    sidecar.  This conversion is explicit and recorded; it never treats slot
    or batch position as scientific ownership.
    """

    config = dict(configuration)
    schema = config.get("schema_version")
    if schema is not None and schema != str(
        MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION
    ):
        raise ScienceInstrumentOperatorError(
            "MUTSUP reducer configuration schema differs"
        )
    semantics = config.get("mutation_observation_semantics")
    if (
        semantics is not None
        and semantics != SCIENCE_MUTSUP_APPLIED_OUTCOMES_ONLY_COMPLETE
    ):
        raise ScienceInstrumentOperatorError(
            "unsupported mutation_observation_semantics; expected "
            f"{SCIENCE_MUTSUP_APPLIED_OUTCOMES_ONLY_COMPLETE}"
        )
    raw_owners = config.get("world_ownership")
    if raw_owners is None:
        raise ScienceInstrumentOperatorError(
            "MUTSUP reducer configuration requires explicit world_ownership"
        )
    owner_rows = _owner_rows_from_configuration(
        raw_owners, field="configuration.world_ownership"
    )
    if not owner_rows:
        raise ScienceInstrumentOperatorError("MUTSUP world_ownership cannot be empty")
    owner_by_cell: dict[str, dict[str, Any]] = {}
    owner_source_by_cell: dict[str, str] = {}
    for index, (alias, raw_owner) in enumerate(owner_rows):
        nested = raw_owner.get("science_ownership", raw_owner.get("science"))
        if isinstance(nested, Mapping):
            owner = {**dict(nested), **raw_owner}
        else:
            owner = dict(raw_owner)
        reference: object | None = None
        for key in ("world_uid", "world_id", "cell_id", "world_key"):
            if key in owner:
                reference = owner[key]
                break
        if reference is None:
            reference = alias
        cell_id = _resolve_mutsup_world_reference(
            reference,
            field=f"configuration.world_ownership[{index}]",
            cells_by_id=cells_by_id,
            semantic_by_hex=semantic_by_hex,
            physical_by_hex=physical_by_hex,
        )
        if cell_id in owner_by_cell:
            raise ScienceInstrumentOperatorError(
                f"duplicate world ownership for materialized world {cell_id}"
            )
        cell_science = _payload_mapping(
            cells_by_id[cell_id].get("science_ownership"),
            field=f"lowered.{cell_id}.science_ownership",
        )
        for key in ("study_id", "arm_id", "replicate"):
            if key not in owner:
                raise ScienceInstrumentOperatorError(
                    f"configuration.world_ownership[{index}] lacks {key}"
                )
            if owner[key] != cell_science.get(key):
                raise ScienceInstrumentOperatorError(
                    f"MUTSUP ownership differs for {cell_id}: {key}"
                )
        if owner.get("pair_id") is not None:
            _nonempty_mutsup_text(
                owner["pair_id"],
                field=f"configuration.world_ownership[{index}].pair_id",
            )
        source_ref = (
            reference
            if isinstance(reference, str)
            else (
                _as_sha256(
                    reference, field=f"configuration.world_ownership[{index}].world_uid"
                )
                if isinstance(reference, Mapping)
                else str(reference)
            )
        )
        owner_source_by_cell[cell_id] = str(source_ref)
        physical_uid = physical_world_uid_by_cell[cell_id]
        effective_owner = {
            "world_uid": physical_uid.to_dict(),
            "cell_id": cell_id,
            "study_id": owner["study_id"],
            "arm_id": owner["arm_id"],
            "replicate": owner["replicate"],
        }
        if owner.get("pair_id") is not None:
            effective_owner["pair_id"] = owner["pair_id"]
        owner_by_cell[cell_id] = effective_owner
    if set(owner_by_cell) != set(cells_by_id):
        missing = sorted(set(cells_by_id) - set(owner_by_cell))
        raise ScienceInstrumentOperatorError(
            f"MUTSUP world_ownership does not cover all worlds: {missing}"
        )

    pairing = config.get("pairing")
    if not isinstance(pairing, Mapping):
        raise ScienceInstrumentOperatorError(
            "MUTSUP reducer configuration requires explicit pairing object"
        )
    pairing_data = dict(pairing)
    off = pairing_data.get("off_arm_ids", pairing_data.get("off_arms"))
    on = pairing_data.get("on_arm_ids", pairing_data.get("on_arms"))
    if isinstance(off, str):
        off = [off]
    if isinstance(on, str):
        on = [on]
    if (
        not isinstance(off, list)
        or not isinstance(on, list)
        or not off
        or not on
        or any(not isinstance(item, str) or not item for item in off + on)
        or set(off) & set(on)
    ):
        raise ScienceInstrumentOperatorError(
            "MUTSUP pairing requires disjoint non-empty off_arm_ids and on_arm_ids"
        )
    raw_pairs = pairing_data.get("pairs", pairing_data.get("pair_map"))
    if isinstance(raw_pairs, Mapping):
        raw_pairs = [
            (
                {
                    **dict(item),
                    "pair_id": pair_id,
                }
                if isinstance(item, Mapping)
                else item
            )
            for pair_id, item in raw_pairs.items()
        ]
    if raw_pairs is None:
        # The canonical MUTSUP authoring form carries an explicit pair_id on
        # each owner and declares the OFF/ON arm sets.  Materialize that
        # compact form into the same explicit endpoint rows consumed by the
        # aggregator; never fall back to physical order or slot adjacency.
        grouped: dict[tuple[str, str | int, str], dict[str, str]] = {}
        for cell_id, owner in owner_by_cell.items():
            pair_id = owner.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise ScienceInstrumentOperatorError(
                    "MUTSUP pairing without pairs requires pair_id on every world ownership row"
                )
            key = (owner["study_id"], owner["replicate"], pair_id)
            role = (
                "off"
                if owner["arm_id"] in off
                or str(owner["arm_id"]).lower() in {str(item).lower() for item in off}
                else (
                    "on"
                    if owner["arm_id"] in on
                    or str(owner["arm_id"]).lower()
                    in {str(item).lower() for item in on}
                    else None
                )
            )
            if role is None:
                raise ScienceInstrumentOperatorError(
                    f"MUTSUP pairing owner {cell_id} has no OFF/ON arm role"
                )
            row = grouped.setdefault(key, {})
            if role in row:
                raise ScienceInstrumentOperatorError(
                    f"MUTSUP pairing group {key!r} has duplicate {role.upper()} world"
                )
            row[role] = cell_id
        raw_pairs = [
            {
                "pair_id": pair_id,
                "study_id": study_id,
                "replicate": replicate,
                "off_world": endpoints["off"],
                "on_world": endpoints["on"],
            }
            for (study_id, replicate, pair_id), endpoints in sorted(
                grouped.items(),
                key=lambda item: (item[0][0], str(item[0][1]), item[0][2]),
            )
            if set(endpoints) == {"off", "on"}
        ]
        if len(raw_pairs) != len(grouped):
            raise ScienceInstrumentOperatorError(
                "MUTSUP pairing owner groups must contain exactly one OFF and one ON world"
            )
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ScienceInstrumentOperatorError(
            "MUTSUP pairing requires an explicit non-empty pairs list"
        )
    effective_pairs: list[dict[str, Any]] = []
    used: set[str] = set()
    for index, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, Mapping):
            raise ScienceInstrumentOperatorError(
                f"configuration.pairing.pairs[{index}] is not an object"
            )
        pair = dict(raw_pair)
        off_ref = next(
            (
                pair[key]
                for key in ("off_world_uid", "off_world_id", "off_world", "off")
                if key in pair
            ),
            None,
        )
        on_ref = next(
            (
                pair[key]
                for key in ("on_world_uid", "on_world_id", "on_world", "on")
                if key in pair
            ),
            None,
        )
        if off_ref is None or on_ref is None:
            raise ScienceInstrumentOperatorError(
                f"configuration.pairing.pairs[{index}] requires OFF and ON worlds"
            )
        off_cell = _resolve_mutsup_world_reference(
            off_ref,
            field=f"configuration.pairing.pairs[{index}].off_world",
            cells_by_id=cells_by_id,
            semantic_by_hex=semantic_by_hex,
            physical_by_hex=physical_by_hex,
        )
        on_cell = _resolve_mutsup_world_reference(
            on_ref,
            field=f"configuration.pairing.pairs[{index}].on_world",
            cells_by_id=cells_by_id,
            semantic_by_hex=semantic_by_hex,
            physical_by_hex=physical_by_hex,
        )
        if off_cell == on_cell or off_cell in used or on_cell in used:
            raise ScienceInstrumentOperatorError(
                f"configuration.pairing.pairs[{index}] duplicates a world"
            )
        used.update((off_cell, on_cell))
        off_owner = owner_by_cell[off_cell]
        on_owner = owner_by_cell[on_cell]
        if off_owner["arm_id"] not in off and str(off_owner["arm_id"]).lower() not in {
            str(item).lower() for item in off
        }:
            raise ScienceInstrumentOperatorError(
                f"explicit OFF endpoint {off_cell} does not have an OFF arm role"
            )
        if on_owner["arm_id"] not in on and str(on_owner["arm_id"]).lower() not in {
            str(item).lower() for item in on
        }:
            raise ScienceInstrumentOperatorError(
                f"explicit ON endpoint {on_cell} does not have an ON arm role"
            )
        if (off_owner["study_id"], off_owner["replicate"]) != (
            on_owner["study_id"],
            on_owner["replicate"],
        ):
            raise ScienceInstrumentOperatorError(
                f"explicit pair {index} crosses study or replicate ownership"
            )
        effective_pair = {
            "off_world_uid": physical_world_uid_by_cell[off_cell].to_dict(),
            "on_world_uid": physical_world_uid_by_cell[on_cell].to_dict(),
        }
        for key in ("pair_id", "study_id", "replicate"):
            if key in pair:
                effective_pair[key] = pair[key]
        effective_pairs.append(effective_pair)
    if used != set(cells_by_id):
        raise ScienceInstrumentOperatorError(
            "explicit MUTSUP pairing does not cover every materialized world"
        )
    effective_pairing = {
        **pairing_data,
        "off_arm_ids": list(off),
        "on_arm_ids": list(on),
        "pairs": effective_pairs,
    }
    effective_config = {
        **config,
        "world_ownership": [owner_by_cell[key] for key in sorted(owner_by_cell)],
        "pairing": effective_pairing,
    }
    return effective_config, owner_by_cell, owner_source_by_cell


def _mutsup_configuration_from_bundle(
    bundle: ScienceWorldBundleV1,
) -> dict[str, Any]:
    configurations: list[dict[str, Any]] = []
    for world in bundle.worlds:
        bindings = world.reducers.to_value()
        if not isinstance(bindings, list):
            raise ScienceInstrumentOperatorError(
                f"world {world.world_id} reducer bindings are invalid"
            )
        selected = [
            row
            for row in bindings
            if isinstance(row, Mapping)
            and row.get("reducer_id") == SCIENCE_MUTSUP_REDUCER_ID
        ]
        if len(selected) != 1 or not isinstance(
            selected[0].get("configuration"), Mapping
        ):
            raise ScienceInstrumentOperatorError(
                f"world {world.world_id} lacks one exact MUTSUP reducer binding"
            )
        configurations.append(dict(selected[0]["configuration"]))
    if not configurations:
        raise ScienceInstrumentOperatorError(
            "MUTSUP bundle has no reducer configuration"
        )
    canonical = canonical_json_bytes(configurations[0])
    if any(canonical_json_bytes(item) != canonical for item in configurations[1:]):
        raise ScienceInstrumentOperatorError(
            "MUTSUP worlds carry different study reducer configurations"
        )
    return configurations[0]


def _mutsup_partitions_for_world(
    sidecar: NativeScienceSidecarV1,
    *,
    cell_id: str,
    observation_abi: SchemaVersion,
    profile: ScienceObservationProfileV1,
) -> tuple[ScienceEvidencePartitionV1, ...]:
    rows = sidecar.records_for_cell(cell_id)
    if not rows:
        raise ScienceInstrumentOperatorError(
            f"MUTSUP sidecar has no canonical rows for {cell_id}"
        )
    if not isinstance(observation_abi, SchemaVersion):
        raise ScienceInstrumentOperatorError(
            f"MUTSUP observation ABI is not resolved for {cell_id}"
        )
    if not isinstance(profile, ScienceObservationProfileV1):
        raise ScienceInstrumentOperatorError(
            f"MUTSUP observation profile is not resolved for {cell_id}"
        )
    end_tick = sidecar.next_absolute_tick + 1
    return tuple(
        ScienceEvidencePartitionV1(
            instrument_manifest_sha256=rows[0].instrument_manifest_sha256,
            science_semantic_acquisition_sha256=(
                rows[0].science_semantic_acquisition_sha256
            ),
            observation_abi_version=observation_abi,
            profile=profile,
            world_uid=rows[0].world_uid,
            record_kind=kind,
            absolute_tick_start=0,
            absolute_tick_end=end_tick,
            rows=tuple(row for row in rows if row.record_kind is kind),
        )
        for kind in ScienceRecordKindV1
    )


def reduce_mutsup_science_instrument_v1(
    materialization: str | Path,
    batch_receipts: Sequence[JsonInputV1],
    *,
    output_directory: str | Path,
    producer_phase_registry: Mapping[str, int] | str | Path | None = None,
) -> dict[str, Any]:
    """Reduce one completed paired MUTSUP bundle with exact restart equivalence."""

    (
        materialization_root,
        materialization_receipt,
        compilation,
        carrier,
        _admission,
        bundle,
        science_plan,
        lowered_batches,
    ) = _materialization_reduction_inputs(materialization)
    cells_by_id = _materialized_mutsup_cells(bundle, carrier, lowered_batches)
    loaded_receipts = tuple(
        _load_completed_science_batch_receipt(value, index=index)
        for index, value in enumerate(batch_receipts)
    )
    if len(loaded_receipts) != len(lowered_batches):
        raise ScienceInstrumentOperatorError(
            "completed batch receipt count differs from materialized batch count"
        )

    registry = _phase_registry(producer_phase_registry)
    sidecars_by_cell: dict[str, NativeScienceSidecarV1] = {}
    observation_contract_by_cell: dict[
        str, tuple[ScienceObservationProfileV1, SchemaVersion]
    ] = {}
    receipts_by_batch_index: dict[int, _LoadedScienceBatchReceiptV1] = {}
    expected_batch_by_cells = {
        tuple(str(cell["cell_id"]) for cell in lowered.get("cells", ())): index
        for index, lowered in enumerate(lowered_batches)
    }
    for receipt in loaded_receipts:
        batch_index = expected_batch_by_cells.get(receipt.cell_ids)
        if batch_index is None or batch_index in receipts_by_batch_index:
            raise ScienceInstrumentOperatorError(
                "completed batch receipt world set/order differs or duplicates"
            )
        lowered = lowered_batches[batch_index]
        _expected_science_batch_plan_sha256(lowered, receipt)
        sidecar = load_native_science_sidecar_v1(
            receipt.records_path,
            expected_sha256=receipt.records_sha256,
            expected_batch_id=receipt.batch_id,
            expected_cell_ids=receipt.cell_ids,
            expected_next_absolute_tick=receipt.next_absolute_tick,
            producer_phase_registry_ordinals=registry,
        )
        receipt_frontier = {
            key: value
            for key, value in receipt.frontier.items()
            if key not in {"records_path", "records_sha256", "terminal"}
        }
        if sidecar.frontier != receipt_frontier:
            raise ScienceInstrumentOperatorError(
                f"completed batch {receipt.batch_id} frontier differs from its sidecar"
            )
        for cell_id in receipt.cell_ids:
            if cell_id in sidecars_by_cell:
                raise ScienceInstrumentOperatorError(
                    f"completed sidecars duplicate world {cell_id}"
                )
            observation_contract_by_cell[cell_id] = _validate_sidecar_world_binding(
                sidecar,
                cells_by_id[cell_id],
                lowered,
                science_plan,
                cell_id=cell_id,
            )
            sidecars_by_cell[cell_id] = sidecar
        receipts_by_batch_index[batch_index] = receipt
    if set(sidecars_by_cell) != set(cells_by_id):
        raise ScienceInstrumentOperatorError(
            "completed sidecars do not exactly cover the materialized MUTSUP worlds"
        )

    physical_world_uid_by_cell: dict[str, ContentDigest] = {}
    for cell_id, sidecar in sidecars_by_cell.items():
        rows = sidecar.records_for_cell(cell_id)
        physical_world_uid_by_cell[cell_id] = rows[0].world_uid
    if len({item.hex for item in physical_world_uid_by_cell.values()}) != len(
        physical_world_uid_by_cell
    ):
        raise ScienceInstrumentOperatorError(
            "completed sidecars reuse a physical world identity"
        )
    semantic_by_hex = {
        world.semantic_world_sha256.hex: world.world_id for world in bundle.worlds
    }
    physical_by_hex = {
        digest.hex: cell_id for cell_id, digest in physical_world_uid_by_cell.items()
    }
    authored_configuration = _mutsup_configuration_from_bundle(bundle)
    effective_configuration, owner_by_cell, owner_source_by_cell = (
        _normalise_mutsup_configuration(
            authored_configuration,
            cells_by_id=cells_by_id,
            semantic_by_hex=semantic_by_hex,
            physical_by_hex=physical_by_hex,
            physical_world_uid_by_cell=physical_world_uid_by_cell,
        )
    )

    compilation_core = _payload_mapping(
        compilation.get("bundle_core"), field="compilation.bundle_core"
    )
    campaign_spec_sha256 = _digest_object(
        compilation_core.get("campaign_spec_sha256"),
        field="compilation.campaign_spec_sha256",
    )
    physical_execution_plan_sha256 = _digest_object(
        carrier.get("physical_execution_plan_sha256"),
        field="carrier.physical_execution_plan_sha256",
    )
    science_execution_plan_sha256 = science_plan.science_execution_plan_sha256()
    output = _output_directory(output_directory)
    per_world_results: list[CanonicalPayloadV1] = []
    per_world_receipts: list[dict[str, Any]] = []
    per_world_artifacts: list[dict[str, Any]] = []
    restart_artifacts: list[dict[str, Any]] = []

    for world_index, cell_id in enumerate(sorted(cells_by_id)):
        sidecar = sidecars_by_cell[cell_id]
        rows = sidecar.records_for_cell(cell_id)
        observation_profile, observation_abi = observation_contract_by_cell[cell_id]
        partitions = _mutsup_partitions_for_world(
            sidecar,
            cell_id=cell_id,
            observation_abi=observation_abi,
            profile=observation_profile,
        )
        world_configuration_value = {
            **effective_configuration,
            "world_ownership": [owner_by_cell[cell_id]],
        }
        world_configuration = CanonicalPayloadV1.from_value(
            MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
            world_configuration_value,
        )
        reduction_plan_sha256 = canonical_sha256(
            {
                "schema_version": "eiel_science_mutsup_reduction_plan_v1",
                "reducer_id": SCIENCE_MUTSUP_REDUCER_ID,
                "configuration": world_configuration.to_dict(),
                "partitioning": "one_world_one_partition_per_canonical_record_kind_v1",
                "producer_phase_registry_ordinals": registry,
                "restart_gate": "uninterrupted_result_equals_fresh_reducer_resume_result",
            }
        )
        arguments = {
            "evidence_capability_ids": MUTSUP_DIRECT_RETENTION_REQUIRED_CAPABILITY_IDS,
            "campaign_spec_sha256": campaign_spec_sha256,
            "instrument_manifest_sha256": rows[0].instrument_manifest_sha256,
            "route_binding_sha256": science_plan.route_binding_sha256,
            "science_semantic_acquisition_sha256": (
                rows[0].science_semantic_acquisition_sha256
            ),
            "physical_execution_plan_sha256": physical_execution_plan_sha256,
            "science_execution_plan_sha256": science_execution_plan_sha256,
            "reduction_plan_sha256": reduction_plan_sha256,
        }
        uninterrupted = execute_registered_science_reducer_v1(
            SCIENCE_MUTSUP_REDUCER_ID,
            world_configuration,
            partitions,
            **arguments,
        )
        split = max(1, len(partitions) // 2)
        partial = execute_registered_science_reducer_v1(
            SCIENCE_MUTSUP_REDUCER_ID,
            world_configuration,
            partitions,
            stop_after_partitions=split,
            **arguments,
        )
        if partial.complete or partial.checkpoint is None:
            raise ScienceInstrumentOperatorError(
                f"MUTSUP reducer did not produce a restart checkpoint for {cell_id}"
            )
        restarted = execute_registered_science_reducer_v1(
            SCIENCE_MUTSUP_REDUCER_ID,
            world_configuration,
            partitions,
            resume=partial.checkpoint,
            **arguments,
        )
        if (
            not uninterrupted.complete
            or not restarted.complete
            or uninterrupted.result_payload != restarted.result_payload
            or uninterrupted.result_receipt is None
            or restarted.result_receipt is None
            or uninterrupted.result_receipt.result_sha256
            != restarted.result_receipt.result_sha256
        ):
            raise ScienceInstrumentOperatorError(
                f"MUTSUP reducer restart differs from uninterrupted reduction for {cell_id}"
            )
        result_payload = uninterrupted.result_payload
        assert result_payload is not None
        per_world_results.append(result_payload)
        world_result = {
            "schema_version": "eiel_science_mutsup_world_reduction_v1",
            "cell_id": cell_id,
            "semantic_world_sha256": next(
                world.semantic_world_sha256.to_dict()
                for world in bundle.worlds
                if world.world_id == cell_id
            ),
            "physical_world_uid": physical_world_uid_by_cell[cell_id].to_dict(),
            "authored_owner_reference": owner_source_by_cell[cell_id],
            "effective_owner": owner_by_cell[cell_id],
            "sidecar_sha256": sidecar.sidecar_sha256.to_dict(),
            "result": result_payload.to_dict(),
            "uninterrupted_result_receipt": uninterrupted.result_receipt.to_dict(),
            "restarted_result_receipt": restarted.result_receipt.to_dict(),
            "restart_checkpoint_sha256": partial.checkpoint.checkpoint_sha256().to_dict(),
            "restart_equivalent": True,
        }
        per_world_artifacts.append(
            _write_canonical_json(
                output / f"world-{world_index:04d}-reduction.json",
                world_result,
                name=f"MUTSUP world reduction {cell_id}",
            )
        )
        restart_artifacts.append(
            _write_canonical_json(
                output / f"world-{world_index:04d}-reducer-checkpoint.json",
                partial.checkpoint.to_dict(),
                name=f"MUTSUP reducer checkpoint {cell_id}",
            )
        )
        per_world_receipts.append(world_result)

    aggregate_configuration = CanonicalPayloadV1.from_value(
        MUTSUP_DIRECT_RETENTION_CONFIGURATION_SCHEMA_VERSION,
        effective_configuration,
    )
    aggregate = aggregate_mutsup_direct_retention_v1(
        per_world_results,
        pairing=aggregate_configuration,
    )
    aggregate_artifact = _write_canonical_json(
        output / "mutsup_paired_result.json",
        aggregate.to_dict(),
        name="paired MUTSUP result",
    )
    batch_lineage = [
        {
            "batch_index": index,
            "batch_id": receipts_by_batch_index[index].batch_id,
            "batch_manifest_path": str(receipts_by_batch_index[index].path),
            "batch_manifest_file_sha256": receipts_by_batch_index[index].file_sha256,
            "plan_sha256": receipts_by_batch_index[index].plan_sha256,
            "execution_route_sha256": receipts_by_batch_index[
                index
            ].execution_route_sha256,
            "sidecar_path": str(receipts_by_batch_index[index].records_path),
            "sidecar_sha256": receipts_by_batch_index[index].records_sha256,
            "next_absolute_tick": receipts_by_batch_index[index].next_absolute_tick,
            "cell_ids": list(receipts_by_batch_index[index].cell_ids),
        }
        for index in sorted(receipts_by_batch_index)
    ]
    receipt_core = {
        "schema_version": SCIENCE_MUTSUP_RESULT_SCHEMA_VERSION,
        "instrument_contract": "EIEL_SCIENCE_INSTRUMENT_V1",
        "science_status": "EXPLORATORY_SCIENCE",
        "engineering_only": True,
        "prequal_started": False,
        "qualification_started": False,
        "claim_boundary": MUTSUP_DIRECT_RETENTION_CLAIM_BOUNDARY,
        "materialization": {
            "path": str(materialization_root / "materialization_receipt.json"),
            "receipt_sha256": materialization_receipt["receipt_sha256"],
            "bundle_content_sha256": bundle.bundle_content_sha256().to_dict(),
            "physical_execution_plan_sha256": physical_execution_plan_sha256.to_dict(),
            "science_execution_plan_sha256": science_execution_plan_sha256.to_dict(),
        },
        "batch_lineage": batch_lineage,
        "world_count": len(per_world_results),
        "pair_count": len(aggregate.to_value().get("paired_contrasts", [])),
        "reducer_restart_equivalent": True,
        "aggregate_result_identity_sha256": aggregate.to_value()[
            "result_identity_sha256"
        ],
        "aggregate_artifact": aggregate_artifact,
        "world_artifacts": per_world_artifacts,
        "restart_checkpoint_artifacts": restart_artifacts,
        "scientific_claims_made": False,
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": canonical_sha256(receipt_core).to_dict(),
    }
    receipt_artifact = _write_canonical_json(
        output / "mutsup_reduction_receipt.json",
        receipt,
        name="MUTSUP reduction receipt",
    )
    return {
        **receipt,
        "status": "COMPLETED",
        "receipt_path": receipt_artifact["path"],
        "receipt_file_sha256": receipt_artifact["file_sha256"],
        "aggregate_result": aggregate.to_dict(),
        "per_world_results": per_world_receipts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eiel-science-instrument",
        description="Materialize and execute engineering-only EIEL science instrument batches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser(
        "materialize", help="compile, admit, plan, and lower"
    )
    materialize.add_argument(
        "--request", "--rich-request", dest="request", required=True, type=Path
    )
    materialize.add_argument(
        "--context", "--science-context", dest="context", required=True, type=Path
    )
    materialize.add_argument(
        "--base-plan",
        "--base-campaign-plan",
        dest="base_plan",
        required=True,
        type=Path,
    )
    materialize.add_argument(
        "--output", "--output-directory", dest="output", required=True, type=Path
    )
    materialize.add_argument(
        "--max-ticks",
        "--maximum-total-ticks",
        dest="maximum_total_ticks",
        required=True,
        type=int,
    )
    materialize.add_argument("--physical-batch-capacity", type=int, default=1024)
    materialize.add_argument("--temporal-k", type=int, default=1)
    materialize.add_argument("--execution-quantum-id")
    materialize.add_argument("--human", action="store_true")
    execute = subparsers.add_parser(
        "execute", help="execute one materialized batch through Launchpad"
    )
    execute.add_argument("--materialization", required=True, type=Path)
    execute.add_argument("--batch-index", required=True, type=int)
    execute.add_argument("--runtime-root", required=True, type=Path)
    execute.add_argument("--repo-root", type=Path, default=Path.cwd())
    execute.add_argument("--state-root", type=Path)
    execute.add_argument(
        "--compact-reducer-workers",
        type=int,
        default=SCIENCE_COMPACT_REDUCER_DEFAULT_WORKERS,
        help=("parallel compact-reducer workers; default 4, 0 disables"),
    )
    execute.add_argument(
        "--compact-reducer-tail-seconds",
        type=float,
        default=None,
        help=(
            "post-front-door reducer wait; defaults to an exact-batch-size and "
            "tick-horizon budget, capped at 3600 seconds"
        ),
    )
    execute.add_argument(
        "--compact-reducer-native-artifact",
        type=Path,
        help="prebuilt native reducer shared object",
    )
    execute.add_argument(
        "--compact-reducer-native-manifest",
        type=Path,
        help="native reducer build manifest",
    )
    execute.add_argument(
        "--compact-reducer-native-manifest-sha256",
        help="plan-bound SHA-256 of the native reducer manifest",
    )
    execute.add_argument(
        "--compact-reducer-native-use-class",
        choices=(
            NATIVE_REDUCER_ARTIFACT_USE_DEFAULT_TECHNICAL_V1,
            NATIVE_REDUCER_ARTIFACT_USE_PRODUCTION_V1,
            NATIVE_REDUCER_ARTIFACT_USE_ENGINEERING_REAL_USE_V1,
        ),
        default=None,
        help=(
            "artifact authority; default_technical_v1 uses the checked-in "
            "native build without granting publication or qualification "
            "authority; production_routing_v1 requires a qualified artifact"
        ),
    )
    execute.add_argument("--human", action="store_true")
    reduce_mutsup = subparsers.add_parser(
        "reduce-mutsup",
        help="reduce completed paired MUTSUP sidecars with restart equivalence",
    )
    reduce_mutsup.add_argument("--materialization", required=True, type=Path)
    reduce_mutsup.add_argument(
        "--batch-receipt",
        required=True,
        action="append",
        type=Path,
        help="completed batch manifest or cell-completion pointer; repeat per batch",
    )
    reduce_mutsup.add_argument("--output", required=True, type=Path)
    reduce_mutsup.add_argument("--producer-phase-registry", type=Path)
    reduce_mutsup.add_argument("--human", action="store_true")
    return parser


def _print_result(result: Mapping[str, Any], *, human: bool) -> None:
    if not human:
        print(json.dumps(dict(result), indent=2, sort_keys=True))
        return
    print(f"status: {result.get('status', 'MATERIALIZED')}")
    print(f"schema_version: {result.get('schema_version')}")
    print(f"receipt_path: {result.get('receipt_path')}")
    print(f"runtime_root: {result.get('runtime_root')}")
    print(f"batch_index: {result.get('batch_index')}")
    print(f"prequal_started: {result.get('prequal_started')}")
    print(f"qualification_started: {result.get('qualification_started')}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "materialize":
            result = materialize_science_instrument_v1(
                args.request,
                args.context,
                args.base_plan,
                output_directory=args.output,
                maximum_total_ticks=args.maximum_total_ticks,
                physical_batch_capacity=args.physical_batch_capacity,
                temporal_k=args.temporal_k,
                execution_quantum_id=args.execution_quantum_id,
            )
        elif args.command == "execute":
            result = execute_materialized_science_batch_v1(
                args.materialization,
                batch_index=args.batch_index,
                runtime_root=args.runtime_root,
                repo_root=args.repo_root,
                state_root=args.state_root,
                compact_reducer_workers=args.compact_reducer_workers,
                compact_reducer_tail_seconds=(args.compact_reducer_tail_seconds),
                compact_reducer_native_artifact=(args.compact_reducer_native_artifact),
                compact_reducer_native_manifest=(args.compact_reducer_native_manifest),
                compact_reducer_native_manifest_sha256=(
                    args.compact_reducer_native_manifest_sha256
                ),
                compact_reducer_native_use_class=(
                    args.compact_reducer_native_use_class
                ),
            )
        else:
            result = reduce_mutsup_science_instrument_v1(
                args.materialization,
                args.batch_receipt,
                output_directory=args.output,
                producer_phase_registry=args.producer_phase_registry,
            )
    except (OSError, TypeError, ValueError, ScienceInstrumentOperatorError) as error:
        result = {
            "schema_version": "eiel_science_instrument_operator_error_v1",
            "ok": False,
            "engineering_only": True,
            "prequal_started": False,
            "qualification_started": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _print_result(result, human=args.human)
        return 2
    _print_result(result, human=args.human)
    # The execute command is a decision-bearing adapter over the canonical
    # front door.  A structured front-door rejection/failure must therefore
    # be reflected in the process exit status as well as in JSON; otherwise a
    # shell orchestrator could advance after a failed native launch.
    if args.command == "execute" and result.get("ok") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "SCIENCE_COMPACT_PROFILE_ID",
    "SCIENCE_COMPACT_REDUCER_MAX_TAIL_SECONDS",
    "SCIENCE_COMPACT_REDUCER_MIN_TAIL_SECONDS",
    "SCIENCE_INSTRUMENT_EXECUTION_LINEAGE_SCHEMA_VERSION",
    "SCIENCE_INSTRUMENT_MATERIALIZATION_SCHEMA_VERSION",
    "SCIENCE_INSTRUMENT_OPERATOR_SCHEMA_VERSION",
    "ScienceInstrumentOperatorError",
    "execute_materialized_science_batch_v1",
    "load_science_instrument_materialization_v1",
    "main",
    "materialize_explicit_science_world_bundle_v1",
    "materialize_science_instrument_v1",
    "reduce_mutsup_science_instrument_v1",
    "resolve_compact_reducer_tail_seconds_v1",
)
