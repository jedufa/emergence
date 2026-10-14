"""Effect-free semantic admission for EIEL_SCIENCE_INSTRUMENT_V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emergence_contracts import (
    ContentDigest,
    EEDReleaseStatusV1,
    ScienceAdmissionOutcomeV1,
    ScienceAdmissionReceiptV1,
    ScienceCampaignV1,
    ScienceFindingV1,
    ScienceInstrumentManifestV1,
    ScienceRouteBindingV1,
    canonical_json_bytes,
    canonical_sha256,
)

from .science_instrument import ScienceCompilationBundleV1

from .science_qualification_envelope_projection_v1 import (
    ScienceQualifiedEnvelopeProjectionV1,
)


@dataclass(frozen=True, slots=True)
class ScienceAdmissionDecisionV1:
    """Effect-free pre-QRR classification; it grants no admission authority."""

    findings: tuple[ScienceFindingV1, ...]
    primary_outcome: ScienceAdmissionOutcomeV1
    effects_performed: bool = False

    def __post_init__(self) -> None:
        if self.effects_performed is not False:
            raise ValueError("science classification must be effect-free")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "eiel_science_admission_decision_v1",
            "findings": [item.to_dict() for item in self.findings],
            "primary_outcome": self.primary_outcome.value,
            "effects_performed": self.effects_performed,
        }

    def decision_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return list(value)


def _requested_scientific_capabilities(
    campaign: ScienceCampaignV1,
    additional: tuple[str, ...],
) -> tuple[str, ...]:
    world = _mapping(campaign.world.to_value(), field="world")
    initialization = _mapping(
        campaign.initialization.to_value(), field="initialization"
    )
    time = _mapping(campaign.time.to_value(), field="time")
    capabilities = set(additional)
    capabilities.add(f"world_profile:{world['world_profile_id']}")
    capabilities.update(
        f"world_overlay:{profile_id}"
        for profile_id in _list(
            world.get("overlay_profile_ids", []), field="world.overlay_profile_ids"
        )
    )
    capabilities.update(
        f"founder_profile:{profile_id}"
        for profile_id in _list(
            initialization.get("founder_profiles", []),
            field="initialization.founder_profiles",
        )
    )
    capabilities.add(f"placement_profile:{initialization['spatial_placement_profile']}")
    capabilities.add(
        f"variation_profile:{initialization['standing_variation_profile']}"
    )
    for index, raw in enumerate(_list(campaign.controls.to_value(), field="controls")):
        control = _mapping(raw, field=f"controls[{index}]")
        capabilities.add(f"control:{control['control_id']}")
        schedule = _mapping(
            control.get("schedule"), field=f"controls[{index}].schedule"
        )
        capabilities.add(f"schedule:{schedule['kind']}")
    if _list(time.get("branches", []), field="time.branches"):
        capabilities.add("scientific_checkpoint_fork_v1")
    if time.get("restore") is not None:
        capabilities.add("absolute_tick_restore_v1")
    return tuple(sorted(capabilities))


def _requires_eed(capabilities: tuple[str, ...]) -> bool:
    return any(
        token == "eed" or token.startswith("eed_") or ":eed" in token
        for token in (item.lower() for item in capabilities)
    )


def _requested_observation_capabilities(
    campaign: ScienceCampaignV1, bundle: ScienceCompilationBundleV1
) -> tuple[str, ...]:
    observations = _mapping(campaign.observations.to_value(), field="observations")
    required = set(
        _list(
            observations.get("required_capability_ids", []),
            field="observations.required_capability_ids",
        )
    )
    evidence = _mapping(
        bundle.reduction_plan.evidence_requirements.to_value(),
        field="reduction_plan.evidence_requirements",
    )
    required.update(
        _list(
            evidence.get("required_observation_capability_ids", []),
            field="reduction_plan.required_observation_capability_ids",
        )
    )
    return tuple(sorted(required))


def science_envelope_subjects_v1(
    campaign: ScienceCampaignV1, bundle: ScienceCompilationBundleV1
) -> dict[str, object]:
    """Project the exact qualification subjects requested by one compilation."""

    world = _mapping(campaign.world.to_value(), field="world")
    geometry = _mapping(world.get("geometry"), field="world.geometry")
    initialization = _mapping(
        campaign.initialization.to_value(), field="initialization"
    )
    time = _mapping(campaign.time.to_value(), field="time")
    observations = _mapping(campaign.observations.to_value(), field="observations")
    execution = _mapping(campaign.execution.to_value(), field="execution")
    lifecycle = _mapping(
        campaign.artifact_lifecycle.to_value(), field="artifact_lifecycle"
    )
    control_values: dict[str, list[object]] = {}
    control_schedules: dict[str, set[str]] = {}
    for cell in bundle.semantic_cells.to_value():
        for binding in cell["control_bindings"]:
            control_id = binding["control_id"]
            values = control_values.setdefault(control_id, [])
            value = binding["resolved_value"]
            if canonical_json_bytes(value) not in {
                canonical_json_bytes(item) for item in values
            }:
                values.append(value)
            control_schedules.setdefault(control_id, set()).add(
                binding["schedule"]["kind"]
            )
    for values in control_values.values():
        values.sort(key=canonical_json_bytes)
    subjects: dict[str, object] = {
        "execution.machine_profile_id": execution["machine_profile_id"],
        "execution.logical_world_count": execution["logical_world_count"],
        "execution.batching_policy": execution["batching_policy"],
        "world.world_profile_id": world["world_profile_id"],
        "world.geometry.width": geometry["width"],
        "world.geometry.height": geometry["height"],
        "world.resident_capacity": world["resident_capacity"],
        "initialization.initial_population": initialization["initial_population"],
        "initialization.initial_live_count": initialization["initial_live_count"],
        "time.start_next_tick": time["start_next_tick"],
        "time.total_ticks": time["total_ticks"],
        "time.observation_ticks": time.get("observation_ticks", []),
        "time.checkpoint_ticks": time.get("checkpoint_ticks", []),
        "observations.profile_id": observations["profile_id"],
        "artifact_lifecycle.profile_id": lifecycle["profile_id"],
        "interaction.control_ids": sorted(control_values),
    }
    for control_id, values in sorted(control_values.items()):
        subjects[f"controls.{control_id}.values"] = values
        subjects[f"controls.{control_id}.schedule_kinds"] = sorted(
            control_schedules[control_id]
        )
    return subjects


def _integrity_findings(
    campaign: ScienceCampaignV1,
    bundle: ScienceCompilationBundleV1,
    instrument_manifest: ScienceInstrumentManifestV1,
    route_binding: ScienceRouteBindingV1,
    envelope: ScienceQualifiedEnvelopeProjectionV1,
) -> list[ScienceFindingV1]:
    findings: list[ScienceFindingV1] = []

    def require_equal(
        actual: object, expected: object, *, path: str, detail: str
    ) -> None:
        if actual != expected:
            findings.append(
                ScienceFindingV1(
                    code="BOUND_IDENTITY_MISMATCH",
                    subject_path=path,
                    detail=detail,
                )
            )

    manifest_sha256 = instrument_manifest.manifest_sha256()
    require_equal(
        bundle.campaign_spec_sha256,
        campaign.campaign_spec_sha256(),
        path="compilation.campaign_spec_sha256",
        detail="Compilation bundle does not bind this campaign.",
    )
    require_equal(
        bundle.execution_plan.instrument_manifest_sha256,
        manifest_sha256,
        path="compilation.instrument_manifest_sha256",
        detail="Execution plan does not bind this instrument manifest.",
    )
    require_equal(
        route_binding.instrument_manifest_sha256,
        manifest_sha256,
        path="route_binding.instrument_manifest_sha256",
        detail="Route binding does not bind this instrument manifest.",
    )
    require_equal(
        bundle.execution_plan.route_binding_sha256,
        route_binding.route_binding_sha256(),
        path="compilation.route_binding_sha256",
        detail="Execution plan does not bind the selected route.",
    )
    require_equal(
        envelope.instrument_manifest_sha256,
        manifest_sha256,
        path="qualified_envelope.instrument_manifest_sha256",
        detail="Qualified envelope does not bind this instrument.",
    )
    require_equal(
        envelope.route_id,
        route_binding.route_id,
        path="qualified_envelope.route_id",
        detail="Qualified envelope does not name the selected route.",
    )
    require_equal(
        envelope.machine_profile_id,
        route_binding.machine_profile_id,
        path="qualified_envelope.machine_profile_id",
        detail="Qualified envelope and selected route machine profiles differ.",
    )
    require_equal(
        envelope.qualification_authority_sha256,
        route_binding.qualification_authority_sha256,
        path="qualified_envelope.qualification_authority_sha256",
        detail="Qualified envelope authority differs from the route binding.",
    )
    require_equal(
        envelope.projection_sha256(),
        route_binding.qualified_envelope_sha256,
        path="route_binding.qualified_envelope_sha256",
        detail="Route binding does not bind the supplied qualified envelope.",
    )
    return findings


def classify_science_campaign_v1(
    campaign: ScienceCampaignV1,
    bundle: ScienceCompilationBundleV1,
    *,
    instrument_manifest: ScienceInstrumentManifestV1,
    route_binding: ScienceRouteBindingV1,
    qualified_envelope: ScienceQualifiedEnvelopeProjectionV1,
    available_scientific_capability_ids: tuple[str, ...],
    available_observation_capability_ids: tuple[str, ...],
    additional_required_scientific_capability_ids: tuple[str, ...] = (),
    upstream_integrity_findings: tuple[ScienceFindingV1, ...] = (),
    upstream_invalid_campaign_findings: tuple[ScienceFindingV1, ...] = (),
) -> ScienceAdmissionDecisionV1:
    """Classify before QRR lowering/resolution without creating a receipt."""

    integrity = list(upstream_integrity_findings)
    integrity.extend(
        _integrity_findings(
            campaign,
            bundle,
            instrument_manifest,
            route_binding,
            qualified_envelope,
        )
    )
    invalid = list(upstream_invalid_campaign_findings)

    required_science = _requested_scientific_capabilities(
        campaign, additional_required_scientific_capability_ids
    )
    available_science = set(available_scientific_capability_ids)
    missing_science: list[ScienceFindingV1] = []
    for capability in required_science:
        if capability not in available_science:
            extension = (
                "B136_SCIENCE_INSTRUMENT_V2_EXTENSION"
                if capability == "scientific_checkpoint_fork_v1"
                else None
            )
            missing_science.append(
                ScienceFindingV1(
                    code="SCIENTIFIC_CAPABILITY_ABSENT",
                    subject_path=f"capabilities.scientific.{capability}",
                    detail="Requested scientific capability is absent.",
                    extension_target=extension,
                )
            )
    if (
        instrument_manifest.eed_release_status is EEDReleaseStatusV1.EXCLUDED
        and _requires_eed(required_science)
        and not any(
            "eed" in finding.subject_path.lower() for finding in missing_science
        )
    ):
        missing_science.append(
            ScienceFindingV1(
                code="EED_RELEASE_EXCLUDED",
                subject_path="instrument.eed_release_status",
                detail="EED is excluded from this V1 instrument release.",
            )
        )

    required_observations = _requested_observation_capabilities(campaign, bundle)
    available_observations = set(available_observation_capability_ids)
    missing_observations = [
        ScienceFindingV1(
            code="OBSERVATION_CAPABILITY_ABSENT",
            subject_path=f"capabilities.observation.{capability}",
            detail="Requested observation capability is absent.",
        )
        for capability in required_observations
        if capability not in available_observations
    ]
    envelope_findings = list(
        qualified_envelope.findings_for(science_envelope_subjects_v1(campaign, bundle))
    )

    if integrity:
        outcome = ScienceAdmissionOutcomeV1.INSTRUMENT_INTEGRITY_FAILURE
    elif invalid:
        outcome = ScienceAdmissionOutcomeV1.INVALID_CAMPAIGN
    elif missing_science:
        outcome = ScienceAdmissionOutcomeV1.MISSING_SCIENTIFIC_CAPABILITY
    elif missing_observations:
        outcome = ScienceAdmissionOutcomeV1.MISSING_OBSERVATION_CAPABILITY
    elif envelope_findings:
        outcome = ScienceAdmissionOutcomeV1.OUTSIDE_QUALIFIED_ENVELOPE
    else:
        outcome = ScienceAdmissionOutcomeV1.ADMITTED

    findings = tuple(
        sorted(
            set(
                integrity
                + invalid
                + missing_science
                + missing_observations
                + envelope_findings
            ),
            key=lambda item: (
                item.subject_path,
                item.code,
                item.detail,
                item.extension_target or "",
                item.evidence_refs,
            ),
        )
    )
    return ScienceAdmissionDecisionV1(
        findings=findings,
        primary_outcome=outcome,
    )


def admit_science_campaign_v1(
    campaign: ScienceCampaignV1,
    bundle: ScienceCompilationBundleV1,
    *,
    instrument_manifest: ScienceInstrumentManifestV1,
    route_binding: ScienceRouteBindingV1,
    qualified_envelope: ScienceQualifiedEnvelopeProjectionV1,
    available_scientific_capability_ids: tuple[str, ...],
    available_observation_capability_ids: tuple[str, ...],
    machine_profile_sha256: ContentDigest,
    lifecycle_contract_sha256: ContentDigest,
    active_qrr_resolution_receipt_sha256: ContentDigest,
    admission_kernel_receipt_sha256: ContentDigest,
    additional_required_scientific_capability_ids: tuple[str, ...] = (),
    upstream_integrity_findings: tuple[ScienceFindingV1, ...] = (),
    upstream_invalid_campaign_findings: tuple[ScienceFindingV1, ...] = (),
    rich_to_flat_projection_receipt_sha256: ContentDigest | None = None,
    science_qrr_binding_receipt_sha256: ContentDigest | None = None,
) -> ScienceAdmissionReceiptV1:
    """Bind one classified campaign to QRR and canonical admission authority."""

    decision = classify_science_campaign_v1(
        campaign,
        bundle,
        instrument_manifest=instrument_manifest,
        route_binding=route_binding,
        qualified_envelope=qualified_envelope,
        available_scientific_capability_ids=available_scientific_capability_ids,
        available_observation_capability_ids=available_observation_capability_ids,
        additional_required_scientific_capability_ids=(
            additional_required_scientific_capability_ids
        ),
        upstream_integrity_findings=upstream_integrity_findings,
        upstream_invalid_campaign_findings=upstream_invalid_campaign_findings,
    )
    return ScienceAdmissionReceiptV1(
        campaign_spec_sha256=campaign.campaign_spec_sha256(),
        instrument_manifest_sha256=instrument_manifest.manifest_sha256(),
        route_binding_sha256=route_binding.route_binding_sha256(),
        qualified_envelope_projection_sha256=qualified_envelope.projection_sha256(),
        machine_profile_id=route_binding.machine_profile_id,
        machine_profile_sha256=machine_profile_sha256,
        lifecycle_contract_sha256=lifecycle_contract_sha256,
        campaign_compiler_component_sha256=bundle.compiler_component_sha256,
        science_semantic_acquisition_sha256=(
            bundle.execution_plan.science_semantic_acquisition_sha256()
        ),
        physical_execution_plan_sha256=(
            bundle.execution_plan.physical_execution_plan_sha256()
        ),
        science_execution_plan_sha256=(
            bundle.execution_plan.science_execution_plan_sha256()
        ),
        reduction_plan_sha256=bundle.reduction_plan.reduction_plan_sha256(),
        compilation_bundle_sha256=bundle.compilation_bundle_sha256(),
        active_qrr_resolution_receipt_sha256=active_qrr_resolution_receipt_sha256,
        canonical_crca_semantic_identity=(
            bundle.execution_plan.crca_semantic_run_digest
        ),
        admission_kernel_receipt_sha256=admission_kernel_receipt_sha256,
        findings=decision.findings,
        primary_outcome=decision.primary_outcome,
        legacy_campaign_plan_sha256=(bundle.execution_plan.legacy_campaign_plan_sha256),
        rich_to_flat_projection_receipt_sha256=(rich_to_flat_projection_receipt_sha256),
        science_qrr_binding_receipt_sha256=science_qrr_binding_receipt_sha256,
    )


__all__ = (
    "ScienceAdmissionDecisionV1",
    "admit_science_campaign_v1",
    "classify_science_campaign_v1",
    "science_envelope_subjects_v1",
)
