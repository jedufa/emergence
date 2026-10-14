"""Effect-free end-to-end host preparation for rich science requests.

This composes normalization, deterministic compilation, pre-QRR classification,
route-owned lowering, read-only active-QRR resolution, additive QRR binding, and
the final science-admission receipt.  It never writes a run root, builds native
code, reserves resources, launches a provider, or touches a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    ProfileResourceEnvelopeV1,
    ScienceAdmissionOutcomeV1,
    ScienceAdmissionReceiptV1,
    ScienceControlDescriptorV1,
    ScienceInstrumentManifestV1,
    ScienceObservationProfileDescriptorV1,
    ScienceReducerDescriptorV1,
    ScienceRouteBindingV1,
    canonical_sha256,
    science_observation_profile_catalog_sha256_v1,
)

from .science_instrument import (
    ScienceCompilationBundleV1,
    ScienceCRCASemanticBindingV1,
    compile_science_campaign_v1,
)

from .science_instrument_admission_v1 import (
    ScienceAdmissionDecisionV1,
    admit_science_campaign_v1,
    classify_science_campaign_v1,
)
from .science_qrr_projection_v1 import (
    ScienceQRRRouteAdapterV1,
    ScienceRichToFlatProjectionV1,
    bind_science_qrr_resolution_v1,
    project_rich_science_request_to_flat_v2,
)
from .science_qualification_envelope_projection_v1 import (
    ScienceQualifiedEnvelopeProjectionV1,
)
from .science_request_v2 import (
    NormalizedRichScienceRequestV2,
    ScienceCampaignResolverV1,
    normalize_rich_production_science_request_v2,
)

SCIENCE_HOST_PREPARATION_SCHEMA_VERSION = "eiel_science_host_preparation_v1"


class ScienceInstrumentHostPreparationError(ValueError):
    """The host-only preparation chain is incomplete or identity-inconsistent."""


class ScienceActiveQRRResolverV1(Protocol):
    """Read-only resolution against the canonical active-QRR authority."""

    def resolve(self, flat_request: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ScienceInstrumentHostContextV1:
    instrument_manifest: ScienceInstrumentManifestV1
    route_binding: ScienceRouteBindingV1
    route_adapter: ScienceQRRRouteAdapterV1
    control_descriptors: tuple[ScienceControlDescriptorV1, ...]
    reducer_descriptors: tuple[ScienceReducerDescriptorV1, ...]
    observation_profile_descriptors: tuple[ScienceObservationProfileDescriptorV1, ...]
    crca_semantic_binding: ScienceCRCASemanticBindingV1
    physical_execution_core: CanonicalPayloadV1
    resource_envelope: ProfileResourceEnvelopeV1
    compiler_component_sha256: ContentDigest
    qualified_envelope: ScienceQualifiedEnvelopeProjectionV1
    available_scientific_capability_ids: tuple[str, ...]
    available_observation_capability_ids: tuple[str, ...]
    machine_profile_sha256: ContentDigest
    admission_kernel_receipt_sha256: ContentDigest

    def __post_init__(self) -> None:
        for field_name in (
            "available_scientific_capability_ids",
            "available_observation_capability_ids",
        ):
            value = getattr(self, field_name)
            if value != tuple(sorted(set(value))):
                raise ScienceInstrumentHostPreparationError(
                    f"{field_name} must be sorted and unique"
                )
        manifest_sha256 = self.instrument_manifest.manifest_sha256()
        if self.route_binding.instrument_manifest_sha256 != manifest_sha256:
            raise ScienceInstrumentHostPreparationError(
                "route binding names a different instrument manifest"
            )
        if self.route_binding.route_id != self.route_adapter.route_id:
            raise ScienceInstrumentHostPreparationError(
                "route adapter names a different route"
            )
        if self.route_binding.execution_adapter_sha256 != (
            self.route_adapter.execution_adapter_sha256()
        ) or self.route_binding.control_adapter_sha256 != (
            self.route_adapter.control_adapter_sha256()
        ):
            raise ScienceInstrumentHostPreparationError(
                "route adapter identity differs from route binding"
            )
        if self.route_binding.resource_envelope_sha256 != canonical_sha256(
            self.resource_envelope.to_dict()
        ):
            raise ScienceInstrumentHostPreparationError(
                "resource envelope identity differs from route binding"
            )
        if self.instrument_manifest.observation_profile_catalog_sha256 != (
            science_observation_profile_catalog_sha256_v1(
                self.observation_profile_descriptors
            )
        ):
            raise ScienceInstrumentHostPreparationError(
                "observation profile catalog differs from instrument manifest"
            )
        if self.resource_envelope.profile not in {
            item.profile for item in self.observation_profile_descriptors
        }:
            raise ScienceInstrumentHostPreparationError(
                "resource envelope profile is absent from the instrument catalog"
            )
        active_profile = next(
            item
            for item in self.observation_profile_descriptors
            if item.profile is self.resource_envelope.profile
        )
        count_capacity = dict(
            self.resource_envelope.max_records_per_tick_per_world_by_kind
        )
        byte_capacity = dict(self.resource_envelope.max_record_bytes_by_kind)
        if any(
            count_capacity.get(kind, 0) < 1 or byte_capacity.get(kind, 0) < 1
            for kind in active_profile.record_kinds
        ):
            raise ScienceInstrumentHostPreparationError(
                "resource envelope does not cover its observation profile"
            )
        if (
            self.qualified_envelope.instrument_manifest_sha256 != manifest_sha256
            or self.qualified_envelope.route_id != self.route_binding.route_id
            or self.qualified_envelope.machine_profile_id
            != self.route_binding.machine_profile_id
            or self.qualified_envelope.qualification_authority_sha256
            != self.route_binding.qualification_authority_sha256
            or self.qualified_envelope.projection_sha256()
            != self.route_binding.qualified_envelope_sha256
        ):
            raise ScienceInstrumentHostPreparationError(
                "qualified envelope identity differs from route binding"
            )


@dataclass(frozen=True, slots=True)
class ScienceInstrumentHostPreparationV1:
    normalized: NormalizedRichScienceRequestV2
    compilation_bundle: ScienceCompilationBundleV1
    classification: ScienceAdmissionDecisionV1
    rich_to_flat_projection: ScienceRichToFlatProjectionV1 | None
    active_qrr_resolution_receipt: dict[str, Any] | None
    science_qrr_binding_receipt: dict[str, Any] | None
    admission_receipt: ScienceAdmissionReceiptV1 | None
    effects_performed: bool = False

    def __post_init__(self) -> None:
        admitted = (
            self.classification.primary_outcome is ScienceAdmissionOutcomeV1.ADMITTED
        )
        final_items = (
            self.rich_to_flat_projection,
            self.active_qrr_resolution_receipt,
            self.science_qrr_binding_receipt,
            self.admission_receipt,
        )
        if admitted != all(item is not None for item in final_items):
            raise ScienceInstrumentHostPreparationError(
                "host preparation has an invalid admitted/rejected shape"
            )
        if self.effects_performed is not False:
            raise ScienceInstrumentHostPreparationError(
                "host preparation must be effect-free"
            )

    def to_dict(self) -> dict[str, object]:
        projection = self.rich_to_flat_projection
        return {
            "schema_version": SCIENCE_HOST_PREPARATION_SCHEMA_VERSION,
            "normalized_request": self.normalized.normalized_request,
            "campaign_resolution_receipt": (self.normalized.content_resolution_receipt),
            "compilation_bundle": self.compilation_bundle.to_dict(),
            "classification": self.classification.to_dict(),
            "flat_request": None if projection is None else projection.flat_request,
            "rich_to_flat_projection_receipt": (
                None if projection is None else projection.projection_receipt
            ),
            "active_qrr_resolution_receipt": self.active_qrr_resolution_receipt,
            "science_qrr_binding_receipt": self.science_qrr_binding_receipt,
            "admission_receipt": (
                None
                if self.admission_receipt is None
                else self.admission_receipt.to_dict()
            ),
            "effects_performed": self.effects_performed,
        }


def _digest_from_value(value: object, *, field: str) -> ContentDigest:
    if (
        not isinstance(value, dict)
        or value.get("algorithm") != "sha256"
        or set(value) != {"algorithm", "hex"}
    ):
        raise ScienceInstrumentHostPreparationError(f"{field} is not a SHA-256 digest")
    try:
        return ContentDigest(value["algorithm"], value["hex"])
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentHostPreparationError(f"{field} is invalid") from error


def prepare_rich_science_request_v2(
    request: dict[str, Any],
    context: ScienceInstrumentHostContextV1,
    *,
    active_qrr_resolver: ScienceActiveQRRResolverV1 | None = None,
    science_campaign_resolvers: dict[str, ScienceCampaignResolverV1] | None = None,
) -> ScienceInstrumentHostPreparationV1:
    """Prepare and admit when supported; rejected classifications stop before QRR."""

    normalized = normalize_rich_production_science_request_v2(
        request, resolvers=science_campaign_resolvers
    )
    bundle = compile_science_campaign_v1(
        normalized.campaign,
        instrument_manifest=context.instrument_manifest,
        route_binding=context.route_binding,
        control_descriptors=context.control_descriptors,
        reducer_descriptors=context.reducer_descriptors,
        crca_semantic_binding=context.crca_semantic_binding,
        physical_execution_core=context.physical_execution_core,
        compiler_component_sha256=context.compiler_component_sha256,
    )
    classification = classify_science_campaign_v1(
        normalized.campaign,
        bundle,
        instrument_manifest=context.instrument_manifest,
        route_binding=context.route_binding,
        qualified_envelope=context.qualified_envelope,
        available_scientific_capability_ids=(
            context.available_scientific_capability_ids
        ),
        available_observation_capability_ids=(
            context.available_observation_capability_ids
        ),
    )
    if classification.primary_outcome is not ScienceAdmissionOutcomeV1.ADMITTED:
        return ScienceInstrumentHostPreparationV1(
            normalized=normalized,
            compilation_bundle=bundle,
            classification=classification,
            rich_to_flat_projection=None,
            active_qrr_resolution_receipt=None,
            science_qrr_binding_receipt=None,
            admission_receipt=None,
        )
    projection = project_rich_science_request_to_flat_v2(
        normalized.normalized_request,
        normalized.campaign,
        bundle,
        route_binding=context.route_binding,
        route_adapter=context.route_adapter,
    )
    if active_qrr_resolver is None:
        raise ScienceInstrumentHostPreparationError(
            "admitted rich request requires a read-only active-QRR resolver"
        )
    resolution = active_qrr_resolver.resolve(projection.flat_request)
    if not isinstance(resolution, dict):
        raise ScienceInstrumentHostPreparationError(
            "active-QRR resolver must return an exact receipt object"
        )
    qrr_binding = bind_science_qrr_resolution_v1(projection, resolution)
    admission = admit_science_campaign_v1(
        normalized.campaign,
        bundle,
        instrument_manifest=context.instrument_manifest,
        route_binding=context.route_binding,
        qualified_envelope=context.qualified_envelope,
        available_scientific_capability_ids=(
            context.available_scientific_capability_ids
        ),
        available_observation_capability_ids=(
            context.available_observation_capability_ids
        ),
        machine_profile_sha256=context.machine_profile_sha256,
        lifecycle_contract_sha256=_digest_from_value(
            projection.projection_receipt["lifecycle_contract_sha256"],
            field="lifecycle_contract_sha256",
        ),
        active_qrr_resolution_receipt_sha256=ContentDigest(
            "sha256", resolution["receipt_sha256"]
        ),
        admission_kernel_receipt_sha256=context.admission_kernel_receipt_sha256,
        rich_to_flat_projection_receipt_sha256=_digest_from_value(
            projection.projection_receipt["receipt_sha256"],
            field="rich_to_flat_projection_receipt_sha256",
        ),
        science_qrr_binding_receipt_sha256=_digest_from_value(
            qrr_binding["receipt_sha256"],
            field="science_qrr_binding_receipt_sha256",
        ),
    )
    if admission.primary_outcome is not ScienceAdmissionOutcomeV1.ADMITTED:
        raise ScienceInstrumentHostPreparationError(
            "final admission differs from pre-QRR classification"
        )
    return ScienceInstrumentHostPreparationV1(
        normalized=normalized,
        compilation_bundle=bundle,
        classification=classification,
        rich_to_flat_projection=projection,
        active_qrr_resolution_receipt=resolution,
        science_qrr_binding_receipt=qrr_binding,
        admission_receipt=admission,
    )


__all__ = (
    "SCIENCE_HOST_PREPARATION_SCHEMA_VERSION",
    "ScienceActiveQRRResolverV1",
    "ScienceInstrumentHostContextV1",
    "ScienceInstrumentHostPreparationError",
    "ScienceInstrumentHostPreparationV1",
    "prepare_rich_science_request_v2",
)
