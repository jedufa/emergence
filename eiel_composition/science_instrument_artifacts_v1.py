"""Content-addressed host-context artifacts for the science instrument."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    EEDReleaseStatusV1,
    ProfileResourceEnvelopeV1,
    SchemaVersion,
    ScienceControlDescriptorV1,
    ScienceInstrumentManifestV1,
    ScienceObservationProfileDescriptorV1,
    ScienceReducerDescriptorV1,
    ScienceRouteBindingV1,
    canonical_sha256,
)

from .science_instrument import ScienceCRCASemanticBindingV1

from .science_instrument_host_front_door_v1 import ScienceInstrumentHostContextV1
from .science_qrr_projection_v1 import ScienceQRRRouteAdapterV1
from .science_qualification_envelope_projection_v1 import (
    ScienceQualifiedDomainRowV1,
    ScienceQualifiedEnvelopeProjectionV1,
)

SCIENCE_HOST_CONTEXT_ARTIFACT_SCHEMA_VERSION = "eiel_science_host_context_artifact_v1"


class ScienceInstrumentArtifactError(ValueError):
    """A host context is unreadable, malformed, or hash-inconsistent."""


def _mapping(
    value: object, *, field: str, exact_keys: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScienceInstrumentArtifactError(f"{field} must be an object")
    result = dict(value)
    if exact_keys is not None and set(result) != exact_keys:
        raise ScienceInstrumentArtifactError(f"{field} has invalid fields")
    return result


def _digest(value: object, *, field: str) -> ContentDigest:
    data = _mapping(value, field=field, exact_keys={"algorithm", "hex"})
    try:
        return ContentDigest(data["algorithm"], data["hex"])
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentArtifactError(f"{field} is invalid") from error


def _payload(value: object, *, field: str) -> CanonicalPayloadV1:
    try:
        return CanonicalPayloadV1.from_dict(value)
    except (TypeError, ValueError) as error:
        raise ScienceInstrumentArtifactError(f"{field} is invalid") from error


def _manifest(value: object) -> ScienceInstrumentManifestV1:
    data = _mapping(
        value,
        field="instrument_manifest",
        exact_keys={
            "schema_version",
            "product_id",
            "component_graph_sha256",
            "control_catalog_sha256",
            "world_profile_catalog_sha256",
            "founder_profile_catalog_sha256",
            "policy_profile_catalog_sha256",
            "mutation_operator_catalog_sha256",
            "observation_abi_sha256",
            "observation_profile_catalog_sha256",
            "reducer_protocol_version",
            "route_binding_protocol_version",
            "eed_release_status",
        },
    )
    return ScienceInstrumentManifestV1(
        component_graph_sha256=_digest(
            data["component_graph_sha256"], field="component_graph_sha256"
        ),
        control_catalog_sha256=_digest(
            data["control_catalog_sha256"], field="control_catalog_sha256"
        ),
        world_profile_catalog_sha256=_digest(
            data["world_profile_catalog_sha256"],
            field="world_profile_catalog_sha256",
        ),
        founder_profile_catalog_sha256=_digest(
            data["founder_profile_catalog_sha256"],
            field="founder_profile_catalog_sha256",
        ),
        policy_profile_catalog_sha256=_digest(
            data["policy_profile_catalog_sha256"],
            field="policy_profile_catalog_sha256",
        ),
        mutation_operator_catalog_sha256=_digest(
            data["mutation_operator_catalog_sha256"],
            field="mutation_operator_catalog_sha256",
        ),
        observation_abi_sha256=_digest(
            data["observation_abi_sha256"], field="observation_abi_sha256"
        ),
        observation_profile_catalog_sha256=_digest(
            data["observation_profile_catalog_sha256"],
            field="observation_profile_catalog_sha256",
        ),
        reducer_protocol_version=SchemaVersion(data["reducer_protocol_version"]),
        route_binding_protocol_version=SchemaVersion(
            data["route_binding_protocol_version"]
        ),
        eed_release_status=EEDReleaseStatusV1(data["eed_release_status"]),
        product_id=data["product_id"],
        schema_version=SchemaVersion(data["schema_version"]),
    )


def _control(value: object) -> ScienceControlDescriptorV1:
    data = _mapping(
        value,
        field="control_descriptor",
        exact_keys={
            "schema_version",
            "control_id",
            "value_contract",
            "unit",
            "activation_phases",
            "heterogeneity_modes",
            "schedule_kinds",
            "readback_capability_ids",
            "interaction_ids",
            "component_id",
            "qualified_envelope_subjects",
        },
    )
    return ScienceControlDescriptorV1(
        control_id=data["control_id"],
        value_contract=_payload(data["value_contract"], field="value_contract"),
        unit=data["unit"],
        activation_phases=tuple(data["activation_phases"]),
        heterogeneity_modes=tuple(data["heterogeneity_modes"]),
        schedule_kinds=tuple(data["schedule_kinds"]),
        readback_capability_ids=tuple(data["readback_capability_ids"]),
        interaction_ids=tuple(data["interaction_ids"]),
        component_id=data["component_id"],
        qualified_envelope_subjects=tuple(data["qualified_envelope_subjects"]),
        schema_version=SchemaVersion(data["schema_version"]),
    )


def _reducer(value: object) -> ScienceReducerDescriptorV1:
    data = _mapping(
        value,
        field="reducer_descriptor",
        exact_keys={
            "schema_version",
            "reducer_id",
            "contract_version",
            "implementation_sha256",
            "configuration_schema_sha256",
            "required_observation_capability_ids",
            "output_contract_id",
            "purity_declarations",
        },
    )
    return ScienceReducerDescriptorV1(
        reducer_id=data["reducer_id"],
        contract_version=SchemaVersion(data["contract_version"]),
        implementation_sha256=_digest(
            data["implementation_sha256"], field="implementation_sha256"
        ),
        configuration_schema_sha256=_digest(
            data["configuration_schema_sha256"],
            field="configuration_schema_sha256",
        ),
        required_observation_capability_ids=tuple(
            data["required_observation_capability_ids"]
        ),
        output_contract_id=data["output_contract_id"],
        purity_declarations=tuple(data["purity_declarations"]),
        schema_version=SchemaVersion(data["schema_version"]),
    )


def _qualified_envelope(value: object) -> ScienceQualifiedEnvelopeProjectionV1:
    data = _mapping(
        value,
        field="qualified_envelope",
        exact_keys={
            "schema_version",
            "instrument_manifest_sha256",
            "route_id",
            "machine_profile_id",
            "qualification_authority_sha256",
            "rows",
        },
    )
    if not isinstance(data["rows"], list):
        raise ScienceInstrumentArtifactError("qualified_envelope.rows must be an array")
    rows: list[ScienceQualifiedDomainRowV1] = []
    for raw in data["rows"]:
        row = _mapping(
            raw,
            field="qualified_envelope.rows[]",
            exact_keys={
                "schema_version",
                "subject_path",
                "accepted_values",
                "envelope_component_id",
                "evidence_refs",
            },
        )
        rows.append(
            ScienceQualifiedDomainRowV1(
                subject_path=row["subject_path"],
                accepted_values=_payload(
                    row["accepted_values"], field="accepted_values"
                ),
                envelope_component_id=row["envelope_component_id"],
                evidence_refs=tuple(row["evidence_refs"]),
                schema_version=SchemaVersion(row["schema_version"]),
            )
        )
    return ScienceQualifiedEnvelopeProjectionV1(
        instrument_manifest_sha256=_digest(
            data["instrument_manifest_sha256"],
            field="instrument_manifest_sha256",
        ),
        route_id=data["route_id"],
        machine_profile_id=data["machine_profile_id"],
        qualification_authority_sha256=_digest(
            data["qualification_authority_sha256"],
            field="qualification_authority_sha256",
        ),
        rows=tuple(rows),
        schema_version=SchemaVersion(data["schema_version"]),
    )


def serialize_science_host_context_v1(
    context: ScienceInstrumentHostContextV1,
) -> dict[str, object]:
    crca = context.crca_semantic_binding
    core: dict[str, object] = {
        "schema_version": SCIENCE_HOST_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "instrument_manifest": context.instrument_manifest.to_dict(),
        "route_binding": context.route_binding.to_dict(),
        "route_adapter": context.route_adapter.to_dict(),
        "control_descriptors": [item.to_dict() for item in context.control_descriptors],
        "reducer_descriptors": [item.to_dict() for item in context.reducer_descriptors],
        "observation_profile_descriptors": [
            item.to_dict() for item in context.observation_profile_descriptors
        ],
        "crca_semantic_binding": {
            "semantic_run_digest": crca.semantic_run_digest.to_dict(),
            "projection_sha256": crca.projection_sha256.to_dict(),
            "absolute_start_next_tick": crca.absolute_start_next_tick,
            "absolute_target_next_tick": crca.absolute_target_next_tick,
            "trajectory_checkpoint_ticks": list(crca.trajectory_checkpoint_ticks),
            "authority": crca.authority,
        },
        "physical_execution_core": context.physical_execution_core.to_dict(),
        "resource_envelope": context.resource_envelope.to_dict(),
        "compiler_component_sha256": context.compiler_component_sha256.to_dict(),
        "qualified_envelope": context.qualified_envelope.to_dict(),
        "available_scientific_capability_ids": list(
            context.available_scientific_capability_ids
        ),
        "available_observation_capability_ids": list(
            context.available_observation_capability_ids
        ),
        "machine_profile_sha256": context.machine_profile_sha256.to_dict(),
        "admission_kernel_receipt_sha256": (
            context.admission_kernel_receipt_sha256.to_dict()
        ),
    }
    return {**core, "context_sha256": canonical_sha256(core).to_dict()}


def parse_science_host_context_v1(value: object) -> ScienceInstrumentHostContextV1:
    data = _mapping(value, field="science_host_context")
    context_digest = _digest(data.pop("context_sha256", None), field="context_sha256")
    expected_keys = {
        "schema_version",
        "instrument_manifest",
        "route_binding",
        "route_adapter",
        "control_descriptors",
        "reducer_descriptors",
        "observation_profile_descriptors",
        "crca_semantic_binding",
        "physical_execution_core",
        "resource_envelope",
        "compiler_component_sha256",
        "qualified_envelope",
        "available_scientific_capability_ids",
        "available_observation_capability_ids",
        "machine_profile_sha256",
        "admission_kernel_receipt_sha256",
    }
    if (
        set(data) != expected_keys
        or data.get("schema_version") != SCIENCE_HOST_CONTEXT_ARTIFACT_SCHEMA_VERSION
    ):
        raise ScienceInstrumentArtifactError("science host context fields are invalid")
    if canonical_sha256(data) != context_digest:
        raise ScienceInstrumentArtifactError("science host context SHA-256 mismatch")
    crca = _mapping(
        data["crca_semantic_binding"],
        field="crca_semantic_binding",
        exact_keys={
            "semantic_run_digest",
            "projection_sha256",
            "absolute_start_next_tick",
            "absolute_target_next_tick",
            "trajectory_checkpoint_ticks",
            "authority",
        },
    )
    if (
        not isinstance(data["control_descriptors"], list)
        or not isinstance(data["reducer_descriptors"], list)
        or not isinstance(data["observation_profile_descriptors"], list)
    ):
        raise ScienceInstrumentArtifactError("descriptor catalogs must be arrays")
    context = ScienceInstrumentHostContextV1(
        instrument_manifest=_manifest(data["instrument_manifest"]),
        route_binding=ScienceRouteBindingV1.from_dict(data["route_binding"]),
        route_adapter=ScienceQRRRouteAdapterV1.from_dict(data["route_adapter"]),
        control_descriptors=tuple(
            _control(item) for item in data["control_descriptors"]
        ),
        reducer_descriptors=tuple(
            _reducer(item) for item in data["reducer_descriptors"]
        ),
        observation_profile_descriptors=tuple(
            ScienceObservationProfileDescriptorV1.from_dict(item)
            for item in data["observation_profile_descriptors"]
        ),
        crca_semantic_binding=ScienceCRCASemanticBindingV1(
            semantic_run_digest=_digest(
                crca["semantic_run_digest"], field="semantic_run_digest"
            ),
            projection_sha256=_digest(
                crca["projection_sha256"], field="projection_sha256"
            ),
            absolute_start_next_tick=crca["absolute_start_next_tick"],
            absolute_target_next_tick=crca["absolute_target_next_tick"],
            trajectory_checkpoint_ticks=tuple(crca["trajectory_checkpoint_ticks"]),
            authority=crca["authority"],
        ),
        physical_execution_core=_payload(
            data["physical_execution_core"], field="physical_execution_core"
        ),
        resource_envelope=ProfileResourceEnvelopeV1.from_dict(
            data["resource_envelope"]
        ),
        compiler_component_sha256=_digest(
            data["compiler_component_sha256"], field="compiler_component_sha256"
        ),
        qualified_envelope=_qualified_envelope(data["qualified_envelope"]),
        available_scientific_capability_ids=tuple(
            data["available_scientific_capability_ids"]
        ),
        available_observation_capability_ids=tuple(
            data["available_observation_capability_ids"]
        ),
        machine_profile_sha256=_digest(
            data["machine_profile_sha256"], field="machine_profile_sha256"
        ),
        admission_kernel_receipt_sha256=_digest(
            data["admission_kernel_receipt_sha256"],
            field="admission_kernel_receipt_sha256",
        ),
    )
    if serialize_science_host_context_v1(context) != value:
        raise ScienceInstrumentArtifactError(
            "science host context does not round-trip canonically"
        )
    return context


def load_science_host_context_v1(path: Path) -> ScienceInstrumentHostContextV1:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ScienceInstrumentArtifactError(
            f"science host context must not be a symbolic link: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ScienceInstrumentArtifactError(
            f"science host context is not a regular file: {resolved}"
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScienceInstrumentArtifactError(
            f"science host context is unreadable: {resolved}"
        ) from error
    return parse_science_host_context_v1(value)


__all__ = (
    "SCIENCE_HOST_CONTEXT_ARTIFACT_SCHEMA_VERSION",
    "ScienceInstrumentArtifactError",
    "load_science_host_context_v1",
    "parse_science_host_context_v1",
    "serialize_science_host_context_v1",
)
