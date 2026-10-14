"""CRCA-G2 deterministic shadow projection over current execution locks.

The current campaign compiler, provider request builder, Baseline, lifecycle,
and route contracts remain authoritative.  This module only composes their
already-resolved payloads into the frozen CRCA identity DTOs and emits a
discardable shadow receipt.  It cannot admit, launch, retry, retain, publish,
or qualify an execution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from emergence_contracts import (
    AdmissionAssuranceBindingV1,
    AttemptAdmissionRequestV1,
    BackendId,
    ConcretizedExecutionV1,
    ContentDigest,
    ExecutionImplementationV1,
    ExperimentId,
    LifecycleBindingV1,
    NamedDigestV1,
    RunId,
    RunOccurrenceV1,
    SemanticRunSpecV1,
    canonical_sha256,
    derive_attempt_id,
    derive_run_id,
)


SHADOW_PROJECTION_INPUT_SCHEMA_VERSION = "crca_shadow_projection_input_v1"
SHADOW_PROJECTION_SCHEMA_VERSION = "crca_shadow_execution_projection_v1"
SHADOW_PROJECTION_AUTHORITY = (
    "SHADOW_ONLY_NO_LAUNCH_RETRY_RETENTION_PUBLICATION_OR_QUALIFICATION_AUTHORITY"
)


class ShadowProjectionError(ValueError):
    """A current execution lock cannot be projected exactly and safely."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowProjectionError(f"{field} must be an object")
    return dict(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ShadowProjectionError(f"{field} must be a non-empty exact string")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ShadowProjectionError(f"{field} must be an integer >= {minimum}")
    return value


def _exact(value: Mapping[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ShadowProjectionError(f"{field} has unsupported fields: {unknown}")


def _content_digest(value: Any, *, field: str) -> ContentDigest:
    if not isinstance(value, Mapping) or set(value) != {"algorithm", "hex"}:
        raise ShadowProjectionError(f"{field} must be a ContentDigest object")
    try:
        return ContentDigest(str(value["algorithm"]), str(value["hex"]))
    except Exception as exc:
        raise ShadowProjectionError(f"{field} is not a valid SHA-256 digest") from exc


def _sha256_hex(value: Any, *, field: str) -> ContentDigest:
    try:
        return ContentDigest("sha256", _text(value, field=field))
    except Exception as exc:
        raise ShadowProjectionError(f"{field} must be a lowercase SHA-256") from exc


def _without(value: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in names}


def _verify_declared_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
    excluded_fields: tuple[str, ...] = (),
    field: str,
) -> ContentDigest:
    declared = _sha256_hex(value.get(digest_field), field=f"{field}.{digest_field}")
    actual = canonical_sha256(_without(value, digest_field, *excluded_fields))
    if actual != declared:
        raise ShadowProjectionError(f"{field}.{digest_field} differs from its payload")
    return declared


def _named_hashes(value: Any, *, field: str) -> tuple[NamedDigestV1, ...]:
    rows = _mapping(value, field=field)
    return tuple(
        NamedDigestV1(name=name, digest=canonical_sha256(rows[name]))
        for name in sorted(rows)
    )


def _named_digest_tuple(value: Any, *, field: str) -> tuple[NamedDigestV1, ...]:
    if not isinstance(value, list):
        raise ShadowProjectionError(f"{field} must be an array")
    rows: list[NamedDigestV1] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, field=f"{field}[{index}]")
        if set(item) != {"name", "digest"}:
            raise ShadowProjectionError(f"{field}[{index}] has an invalid shape")
        rows.append(
            NamedDigestV1(
                name=_text(item.get("name"), field=f"{field}[{index}].name"),
                digest=_content_digest(
                    item.get("digest"), field=f"{field}[{index}].digest"
                ),
            )
        )
    return tuple(rows)


def _optional_content_digest(value: Any, *, field: str) -> ContentDigest | None:
    return None if value is None else _content_digest(value, field=field)


def _reconstruct_layer_digest(name: str, payload: Mapping[str, Any]) -> ContentDigest:
    """Rebuild a frozen DTO so a rehashed malformed payload cannot pass."""

    try:
        if name == "semantic":
            _exact(
                payload,
                {
                    "schema_version",
                    "experiment_id",
                    "system_assembly_digest",
                    "semantic_parameters_digest",
                    "rng_contract_digest",
                    "seed_digest",
                    "absolute_start_next_tick",
                    "absolute_target_next_tick",
                    "trajectory_checkpoint_ticks",
                    "measurement_behavior_digest",
                    "control_and_branch_semantics_digest",
                },
                field="layers.semantic.payload",
            )
            raw_ticks = payload.get("trajectory_checkpoint_ticks")
            if not isinstance(raw_ticks, list):
                raise ShadowProjectionError(
                    "semantic trajectory checkpoint ticks must be an array"
                )
            dto: Any = SemanticRunSpecV1(
                experiment_id=ExperimentId(
                    _text(payload.get("experiment_id"), field="semantic.experiment_id")
                ),
                system_assembly_digest=_content_digest(
                    payload.get("system_assembly_digest"),
                    field="semantic.system_assembly_digest",
                ),
                semantic_parameters_digest=_content_digest(
                    payload.get("semantic_parameters_digest"),
                    field="semantic.semantic_parameters_digest",
                ),
                rng_contract_digest=_content_digest(
                    payload.get("rng_contract_digest"),
                    field="semantic.rng_contract_digest",
                ),
                seed_digest=_content_digest(
                    payload.get("seed_digest"), field="semantic.seed_digest"
                ),
                absolute_start_next_tick=_integer(
                    payload.get("absolute_start_next_tick"),
                    field="semantic.absolute_start_next_tick",
                ),
                absolute_target_next_tick=_integer(
                    payload.get("absolute_target_next_tick"),
                    field="semantic.absolute_target_next_tick",
                    minimum=1,
                ),
                trajectory_checkpoint_ticks=tuple(
                    _integer(
                        tick, field=f"semantic.trajectory_checkpoint_ticks[{index}]"
                    )
                    for index, tick in enumerate(raw_ticks)
                ),
                measurement_behavior_digest=_content_digest(
                    payload.get("measurement_behavior_digest"),
                    field="semantic.measurement_behavior_digest",
                ),
                control_and_branch_semantics_digest=_content_digest(
                    payload.get("control_and_branch_semantics_digest"),
                    field="semantic.control_and_branch_semantics_digest",
                ),
            )
            digest = dto.semantic_spec_digest()
        elif name == "implementation":
            _exact(
                payload,
                {
                    "schema_version",
                    "source_commit",
                    "source_tree_digest",
                    "dirty_patch_digest",
                    "toolchain_digest",
                    "environment_digest",
                    "provider_id",
                    "backend_id",
                    "runtime_profile_id",
                    "native_artifact_digests",
                    "abi_digest",
                    "device_conformance_envelope_digest",
                },
                field="layers.implementation.payload",
            )
            dto = ExecutionImplementationV1(
                source_commit=_text(
                    payload.get("source_commit"), field="implementation.source_commit"
                ),
                source_tree_digest=_content_digest(
                    payload.get("source_tree_digest"),
                    field="implementation.source_tree_digest",
                ),
                dirty_patch_digest=_optional_content_digest(
                    payload.get("dirty_patch_digest"),
                    field="implementation.dirty_patch_digest",
                ),
                toolchain_digest=_content_digest(
                    payload.get("toolchain_digest"),
                    field="implementation.toolchain_digest",
                ),
                environment_digest=_content_digest(
                    payload.get("environment_digest"),
                    field="implementation.environment_digest",
                ),
                provider_id=_text(
                    payload.get("provider_id"), field="implementation.provider_id"
                ),
                backend_id=BackendId(
                    _text(payload.get("backend_id"), field="implementation.backend_id")
                ),
                runtime_profile_id=_text(
                    payload.get("runtime_profile_id"),
                    field="implementation.runtime_profile_id",
                ),
                native_artifact_digests=_named_digest_tuple(
                    payload.get("native_artifact_digests"),
                    field="implementation.native_artifact_digests",
                ),
                abi_digest=_content_digest(
                    payload.get("abi_digest"), field="implementation.abi_digest"
                ),
                device_conformance_envelope_digest=_content_digest(
                    payload.get("device_conformance_envelope_digest"),
                    field="implementation.device_conformance_envelope_digest",
                ),
            )
            digest = dto.implementation_digest()
        elif name == "admission_assurance":
            _exact(
                payload,
                {
                    "schema_version",
                    "intended_claim_id",
                    "qualified_envelope_digest",
                    "obligation_digests",
                    "authority_reference_digests",
                    "review_reference_digests",
                },
                field="layers.admission_assurance.payload",
            )
            dto = AdmissionAssuranceBindingV1(
                intended_claim_id=_text(
                    payload.get("intended_claim_id"),
                    field="admission_assurance.intended_claim_id",
                ),
                qualified_envelope_digest=_optional_content_digest(
                    payload.get("qualified_envelope_digest"),
                    field="admission_assurance.qualified_envelope_digest",
                ),
                obligation_digests=_named_digest_tuple(
                    payload.get("obligation_digests"),
                    field="admission_assurance.obligation_digests",
                ),
                authority_reference_digests=_named_digest_tuple(
                    payload.get("authority_reference_digests"),
                    field="admission_assurance.authority_reference_digests",
                ),
                review_reference_digests=_named_digest_tuple(
                    payload.get("review_reference_digests"),
                    field="admission_assurance.review_reference_digests",
                ),
            )
            digest = dto.admission_assurance_digest()
        elif name == "lifecycle":
            _exact(
                payload,
                {
                    "schema_version",
                    "output_root",
                    "artifact_lifecycle_contract_digest",
                    "inventory_role_digests",
                    "initial_hold_digests",
                    "retention_profile_id",
                },
                field="layers.lifecycle.payload",
            )
            dto = LifecycleBindingV1(
                output_root=_text(
                    payload.get("output_root"), field="lifecycle.output_root"
                ),
                artifact_lifecycle_contract_digest=_content_digest(
                    payload.get("artifact_lifecycle_contract_digest"),
                    field="lifecycle.artifact_lifecycle_contract_digest",
                ),
                inventory_role_digests=_named_digest_tuple(
                    payload.get("inventory_role_digests"),
                    field="lifecycle.inventory_role_digests",
                ),
                initial_hold_digests=_named_digest_tuple(
                    payload.get("initial_hold_digests"),
                    field="lifecycle.initial_hold_digests",
                ),
                retention_profile_id=_text(
                    payload.get("retention_profile_id"),
                    field="lifecycle.retention_profile_id",
                ),
            )
            digest = dto.lifecycle_digest()
        elif name == "attempt_admission_request":
            _exact(
                payload,
                {
                    "schema_version",
                    "resource_requirements_digest",
                    "executor_constraints_digest",
                    "requested_restore_checkpoint_digest",
                    "observational_persistence_cadence_digest",
                    "proposed_reason",
                },
                field="layers.attempt_admission_request.payload",
            )
            dto = AttemptAdmissionRequestV1(
                resource_requirements_digest=_content_digest(
                    payload.get("resource_requirements_digest"),
                    field="attempt_admission_request.resource_requirements_digest",
                ),
                executor_constraints_digest=_content_digest(
                    payload.get("executor_constraints_digest"),
                    field="attempt_admission_request.executor_constraints_digest",
                ),
                requested_restore_checkpoint_digest=_optional_content_digest(
                    payload.get("requested_restore_checkpoint_digest"),
                    field=(
                        "attempt_admission_request.requested_restore_checkpoint_digest"
                    ),
                ),
                observational_persistence_cadence_digest=_optional_content_digest(
                    payload.get("observational_persistence_cadence_digest"),
                    field=(
                        "attempt_admission_request."
                        "observational_persistence_cadence_digest"
                    ),
                ),
                proposed_reason=_text(
                    payload.get("proposed_reason"),
                    field="attempt_admission_request.proposed_reason",
                ),
            )
            digest = dto.attempt_admission_request_digest()
        elif name == "concretized_execution":
            digest_fields = {
                "semantic_spec_digest",
                "implementation_digest",
                "admission_assurance_digest",
                "lifecycle_digest",
                "attempt_admission_request_digest",
                "campaign_plan_digest",
                "bound_execution_request_digest",
                "run_contract_digest",
                "baseline_digest",
                "route_contract_digest",
                "assembly_digest",
                "qualification_key_digest",
            }
            _exact(
                payload,
                {"schema_version", *digest_fields},
                field="layers.concretized_execution.payload",
            )
            dto = ConcretizedExecutionV1(
                **{
                    field_name: _content_digest(
                        payload.get(field_name),
                        field=f"concretized_execution.{field_name}",
                    )
                    for field_name in digest_fields
                }
            )
            digest = dto.concretized_execution_digest()
        else:
            raise ShadowProjectionError(f"unknown CRCA layer: {name}")
    except ShadowProjectionError:
        raise
    except (TypeError, ValueError) as exc:
        raise ShadowProjectionError(f"layers.{name}.payload is invalid: {exc}") from exc
    if dto.to_dict() != dict(payload):
        raise ShadowProjectionError(
            f"layers.{name}.payload is not an exact frozen DTO reconstruction"
        )
    return digest


def _artifact_digests(plan: Mapping[str, Any]) -> tuple[NamedDigestV1, ...]:
    artifacts = _mapping(plan.get("artifacts", {}), field="campaign_plan.artifacts")
    rows: list[NamedDigestV1] = []
    for name in sorted(artifacts):
        artifact = _mapping(artifacts[name], field=f"campaign_plan.artifacts.{name}")
        rows.append(
            NamedDigestV1(
                name=name,
                digest=_sha256_hex(
                    artifact.get("sha256"),
                    field=f"campaign_plan.artifacts.{name}.sha256",
                ),
            )
        )
    return tuple(rows)


def _absolute_span_and_boundaries(
    execution: Mapping[str, Any],
) -> tuple[int, int, tuple[int, ...], str]:
    route = execution.get("execution_route_identity")
    if route == "tick_native_execution_route_v1":
        if execution.get("time_authority") != "absolute_tick_v1":
            raise ShadowProjectionError(
                "tick-native campaign_plan.execution lacks absolute_tick_v1 authority"
            )
        start = _integer(
            execution.get("start_next_tick", 0),
            field="campaign_plan.execution.start_next_tick",
        )
        target = _integer(
            execution.get("total_ticks"),
            field="campaign_plan.execution.total_ticks",
            minimum=1,
        )
        raw = execution.get("checkpoint_ticks", [])
        authority = "absolute_tick_v1"
    elif route is None:
        generations = _integer(
            execution.get("generations"),
            field="campaign_plan.execution.generations",
            minimum=1,
        )
        ticks_per_generation = _integer(
            execution.get("ticks_per_generation"),
            field="campaign_plan.execution.ticks_per_generation",
            minimum=1,
        )
        start = 0
        target = generations * ticks_per_generation
        raw_generations = execution.get("checkpoints", [])
        if not isinstance(raw_generations, list):
            raise ShadowProjectionError(
                "campaign_plan.execution.checkpoints must be an array"
            )
        raw = [
            _integer(
                item,
                field=f"campaign_plan.execution.checkpoints[{index}]",
            )
            * ticks_per_generation
            for index, item in enumerate(raw_generations)
        ]
        authority = "private_legacy_generation_projection_v1"
    else:
        raise ShadowProjectionError(
            f"unsupported campaign execution route identity: {route!r}"
        )
    if not isinstance(raw, list):
        raise ShadowProjectionError(
            "campaign_plan.execution checkpoint coordinates must be an array"
        )
    checkpoints = [
        _integer(item, field=f"campaign_plan.execution.checkpoints[{index}]")
        for index, item in enumerate(raw)
    ]
    boundaries = tuple(sorted({start, *checkpoints, target}))
    if target <= start or any(item < start or item > target for item in boundaries):
        raise ShadowProjectionError("campaign_plan absolute execution span is invalid")
    return start, target, boundaries, authority


def _semantic_layer(plan: Mapping[str, Any]) -> SemanticRunSpecV1:
    execution = _mapping(plan.get("execution"), field="campaign_plan.execution")
    qualification = _mapping(
        plan.get("qualification_key"), field="campaign_plan.qualification_key"
    )
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ShadowProjectionError("campaign_plan.cells must be a non-empty array")
    start, target, checkpoints, source_time_authority = _absolute_span_and_boundaries(
        execution
    )
    semantic_cells: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(cells):
        cell = _mapping(raw, field=f"campaign_plan.cells[{index}]")
        semantic_cells.append(
            {
                "cell_id": cell.get("cell_id"),
                "dimensions": cell.get("dimensions"),
                "parameters": cell.get("parameters"),
            }
        )
        seed_rows.append(
            {
                "cell_id": cell.get("cell_id"),
                "rng_lineage_id": cell.get("rng_lineage_id"),
                "seeds": cell.get("seeds"),
            }
        )
    assembly_payload = {
        "resolved_profile": plan.get("resolved_profile"),
        "backend": qualification.get("backend"),
        "profile_id": qualification.get("profile_id"),
        "profile_version": qualification.get("profile_version"),
        "fixed_invariants": qualification.get("fixed_invariants"),
        "state_components": qualification.get("state_components"),
        "checkpoint_components": qualification.get("checkpoint_components"),
    }
    rng_payload = {
        "source_time_authority": source_time_authority,
        "qualification_fixed_invariants": qualification.get("fixed_invariants"),
        "tick_native_control": execution.get("tick_native_control"),
    }
    measurement_payload = {
        "analysis": plan.get("analysis"),
        "reporting_parity": _mapping(
            plan.get("comparisons", {}), field="campaign_plan.comparisons"
        ).get("reporting_parity"),
        "observation_contract": qualification.get("observation_contract"),
    }
    control_payload = {
        "source_time_authority": source_time_authority,
        "technical_gate": execution.get("technical_gate"),
        "comparison_branches": _mapping(
            plan.get("comparisons", {}), field="campaign_plan.comparisons"
        ).get("branches"),
        "trajectory_checkpoint_ticks": list(checkpoints),
    }
    return SemanticRunSpecV1(
        experiment_id=ExperimentId(
            _text(plan.get("campaign_id"), field="campaign_plan.campaign_id")
        ),
        system_assembly_digest=canonical_sha256(assembly_payload),
        semantic_parameters_digest=canonical_sha256(semantic_cells),
        rng_contract_digest=canonical_sha256(rng_payload),
        seed_digest=canonical_sha256(seed_rows),
        absolute_start_next_tick=start,
        absolute_target_next_tick=target,
        trajectory_checkpoint_ticks=checkpoints,
        measurement_behavior_digest=canonical_sha256(measurement_payload),
        control_and_branch_semantics_digest=canonical_sha256(control_payload),
    )


def _implementation_layer(
    plan: Mapping[str, Any],
    bound_request: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> ExecutionImplementationV1:
    _exact(
        source_identity,
        {
            "source_commit",
            "source_tree_identity",
            "dirty_patch_sha256",
            "toolchain",
            "environment",
            "device_conformance_envelope",
        },
        field="source_identity",
    )
    source_commit = _text(
        source_identity.get("source_commit"), field="source_identity.source_commit"
    )
    if len(source_commit) != 40 or any(
        ch not in "0123456789abcdef" for ch in source_commit
    ):
        raise ShadowProjectionError(
            "source_identity.source_commit must be a full Git ID"
        )
    dirty = source_identity.get("dirty_patch_sha256")
    qualification = _mapping(
        plan.get("qualification_key"), field="campaign_plan.qualification_key"
    )
    backend = _text(qualification.get("backend"), field="qualification_key.backend")
    abi_payload = {
        "state_components": qualification.get("state_components"),
        "checkpoint_components": qualification.get("checkpoint_components"),
        "observation_contract": qualification.get("observation_contract"),
        "run_contract": plan.get("run_contract"),
    }
    return ExecutionImplementationV1(
        source_commit=source_commit,
        source_tree_digest=canonical_sha256(
            _mapping(
                source_identity.get("source_tree_identity"),
                field="source_identity.source_tree_identity",
            )
        ),
        dirty_patch_digest=(
            None
            if dirty is None
            else _sha256_hex(dirty, field="source_identity.dirty_patch_sha256")
        ),
        toolchain_digest=canonical_sha256(
            _mapping(
                source_identity.get("toolchain"), field="source_identity.toolchain"
            )
        ),
        environment_digest=canonical_sha256(
            _mapping(
                source_identity.get("environment"), field="source_identity.environment"
            )
        ),
        provider_id=_text(
            bound_request.get("provider_id"),
            field="bound_execution_request.provider_id",
        ),
        backend_id=BackendId(backend),
        runtime_profile_id=_text(
            bound_request.get("profile_id"), field="bound_execution_request.profile_id"
        ),
        native_artifact_digests=_artifact_digests(plan),
        abi_digest=canonical_sha256(abi_payload),
        device_conformance_envelope_digest=canonical_sha256(
            _mapping(
                source_identity.get("device_conformance_envelope"),
                field="source_identity.device_conformance_envelope",
            )
        ),
    )


def _assurance_layer(
    assurance: Mapping[str, Any],
) -> AdmissionAssuranceBindingV1:
    _exact(
        assurance,
        {
            "intended_claim_id",
            "qualified_envelope",
            "obligations",
            "authority_references",
            "review_references",
            "initial_holds",
        },
        field="assurance",
    )
    qualified = assurance.get("qualified_envelope")
    return AdmissionAssuranceBindingV1(
        intended_claim_id=_text(
            assurance.get("intended_claim_id"), field="assurance.intended_claim_id"
        ),
        qualified_envelope_digest=(
            None
            if qualified is None
            else canonical_sha256(
                _mapping(qualified, field="assurance.qualified_envelope")
            )
        ),
        obligation_digests=_named_hashes(
            assurance.get("obligations", {}), field="assurance.obligations"
        ),
        authority_reference_digests=_named_hashes(
            assurance.get("authority_references", {}),
            field="assurance.authority_references",
        ),
        review_reference_digests=_named_hashes(
            assurance.get("review_references", {}),
            field="assurance.review_references",
        ),
    )


def _lifecycle_layer(
    plan: Mapping[str, Any], assurance: Mapping[str, Any]
) -> LifecycleBindingV1:
    lifecycle = _mapping(
        plan.get("artifact_lifecycle"), field="campaign_plan.artifact_lifecycle"
    )
    output = _mapping(plan.get("output"), field="campaign_plan.output")
    checkpoints = _mapping(
        lifecycle.get("checkpoints"),
        field="campaign_plan.artifact_lifecycle.checkpoints",
    )
    role_rows = checkpoints.get("roles", [])
    if not isinstance(role_rows, list):
        raise ShadowProjectionError(
            "artifact lifecycle checkpoint roles must be an array"
        )
    roles: dict[str, Any] = {}
    for index, raw in enumerate(role_rows):
        row = _mapping(raw, field=f"artifact_lifecycle.checkpoints.roles[{index}]")
        name = _text(row.get("identity"), field=f"checkpoint role {index}.identity")
        if name in roles:
            raise ShadowProjectionError(
                "artifact lifecycle checkpoint role repeats identity"
            )
        roles[name] = row
    return LifecycleBindingV1(
        output_root=_text(
            output.get("runtime_root"), field="campaign_plan.output.runtime_root"
        ),
        artifact_lifecycle_contract_digest=canonical_sha256(lifecycle),
        inventory_role_digests=_named_hashes(roles, field="lifecycle.roles"),
        initial_hold_digests=_named_hashes(
            assurance.get("initial_holds", {}), field="assurance.initial_holds"
        ),
        retention_profile_id=_text(
            lifecycle.get("lifecycle_profile"),
            field="artifact_lifecycle.lifecycle_profile",
        ),
    )


def _attempt_admission_layer(
    plan: Mapping[str, Any],
    bound_request: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    *,
    proposed_reason: str,
    child_restore_checkpoint_digest: ContentDigest | None = None,
) -> AttemptAdmissionRequestV1:
    resident = _mapping(
        bound_request.get("resident_request"),
        field="bound_execution_request.resident_request",
    )
    execution = _mapping(plan.get("execution"), field="campaign_plan.execution")
    restore_ticks = resident.get("deliberate_restore_ticks", [])
    restore_tick = resident.get("deliberate_restore_tick")
    requested_restore = None
    if not isinstance(restore_ticks, list):
        raise ShadowProjectionError(
            "bound_execution_request.resident_request.deliberate_restore_ticks "
            "must be an array"
        )
    if restore_ticks:
        exact_restore_ticks = [
            _integer(
                tick,
                field=(
                    "bound_execution_request.resident_request."
                    f"deliberate_restore_ticks[{index}]"
                ),
            )
            for index, tick in enumerate(restore_ticks)
        ]
        if exact_restore_ticks != sorted(set(exact_restore_ticks)):
            raise ShadowProjectionError(
                "bound execution request restore ticks must be sorted and unique"
            )
        if restore_tick is not None and restore_tick not in exact_restore_ticks:
            raise ShadowProjectionError(
                "singular restore tick differs from the exact restore tick set"
            )
        requested_restore = {"absolute_ticks": exact_restore_ticks}
    elif restore_tick is not None:
        restore_tick = _integer(
            restore_tick,
            field="bound_execution_request.resident_request.deliberate_restore_tick",
        )
        requested_restore = {"absolute_tick": restore_tick}
    cadence = {
        "reporting_enabled": resident.get("reporting_enabled"),
        "checkpoint_storage_mode": resident.get("checkpoint_storage_mode"),
        "artifact_observability": _mapping(
            plan.get("artifact_lifecycle"), field="campaign_plan.artifact_lifecycle"
        ).get("observability"),
    }
    executor = {
        "provider_id": bound_request.get("provider_id"),
        "profile_id": bound_request.get("profile_id"),
        "backend": _mapping(
            plan.get("qualification_key"), field="campaign_plan.qualification_key"
        ).get("backend"),
        "device_conformance_envelope": source_identity.get(
            "device_conformance_envelope"
        ),
    }
    derived_restore_digest = (
        None if requested_restore is None else canonical_sha256(requested_restore)
    )
    if (
        child_restore_checkpoint_digest is not None
        and derived_restore_digest is not None
        and child_restore_checkpoint_digest != derived_restore_digest
    ):
        raise ShadowProjectionError(
            "child checkpoint identity differs from deliberate restore identity"
        )
    return AttemptAdmissionRequestV1(
        resource_requirements_digest=canonical_sha256(
            {
                "budgets": execution.get("budgets"),
                "resource_estimate": plan.get("resource_estimate"),
                "bound_resident_request": resident,
                "bound_request_derivation": bound_request.get("derivation"),
                "bound_physical_batch_set": bound_request.get(
                    "physical_batch_set"
                ),
            }
        ),
        executor_constraints_digest=canonical_sha256(executor),
        requested_restore_checkpoint_digest=(
            child_restore_checkpoint_digest or derived_restore_digest
        ),
        observational_persistence_cadence_digest=canonical_sha256(cadence),
        proposed_reason=_text(proposed_reason, field="proposed_reason"),
    )


def _bound_request_digest(bound_request: Mapping[str, Any]) -> ContentDigest:
    declared = _sha256_hex(
        bound_request.get("bound_request_sha256"),
        field="bound_execution_request.bound_request_sha256",
    )
    actual = canonical_sha256(
        _without(
            bound_request,
            "bound_request_sha256",
            "execution_route_contract",
        )
    )
    if actual != declared:
        raise ShadowProjectionError(
            "bound_execution_request.bound_request_sha256 differs from identity payload"
        )
    return declared


def project_current_execution_v1(
    campaign_plan: Mapping[str, Any],
    bound_execution_request: Mapping[str, Any],
    baseline: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    assurance: Mapping[str, Any],
    *,
    cell_or_repeat_ordinal: int = 0,
    attempt_ordinal: int = 1,
    attempt_nonce: str = "shadow-projection",
    proposed_reason: str = "shadow identity projection",
    parent_checkpoint: Mapping[str, Any] | None = None,
    branch_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project current locks without changing any owning execution decision."""

    plan = _mapping(campaign_plan, field="campaign_plan")
    bound = _mapping(bound_execution_request, field="bound_execution_request")
    baseline_value = _mapping(baseline, field="baseline")
    source = _mapping(source_identity, field="source_identity")
    assurance_value = _mapping(assurance, field="assurance")
    ordinal = _integer(cell_or_repeat_ordinal, field="cell_or_repeat_ordinal")
    attempt_number = _integer(attempt_ordinal, field="attempt_ordinal", minimum=1)
    nonce = _text(attempt_nonce, field="attempt_nonce")

    plan_digest = _verify_declared_digest(
        plan,
        digest_field="plan_sha256",
        field="campaign_plan",
    )
    bound_digest = _bound_request_digest(bound)
    if bound.get("plan_sha256") != plan_digest.hex:
        raise ShadowProjectionError("bound request is for another campaign plan")
    if bound.get("profile_id") != plan.get("resolved_profile"):
        raise ShadowProjectionError("bound request profile differs from campaign plan")
    resident = _mapping(
        bound.get("resident_request"),
        field="bound_execution_request.resident_request",
    )
    if resident.get("profile_id") != bound.get("profile_id"):
        raise ShadowProjectionError("bound resident request profile differs")
    qualification_digest = _sha256_hex(
        plan.get("qualification_key_sha256"),
        field="campaign_plan.qualification_key_sha256",
    )
    if canonical_sha256(plan.get("qualification_key")).hex != qualification_digest.hex:
        raise ShadowProjectionError("campaign qualification key digest differs")
    if bound.get("qualification_key_sha256") != qualification_digest.hex:
        raise ShadowProjectionError("bound request qualification key differs")
    route_contract = _mapping(
        bound.get("execution_route_contract"),
        field="bound_execution_request.execution_route_contract",
    )
    route_digest = _sha256_hex(
        bound.get("execution_route_sha256"),
        field="bound_execution_request.execution_route_sha256",
    )
    if canonical_sha256(route_contract) != route_digest:
        raise ShadowProjectionError("bound execution route digest differs")
    if resident.get("execution_route_sha256") != route_digest.hex:
        raise ShadowProjectionError("bound resident request route differs")

    semantic = _semantic_layer(plan)
    implementation = _implementation_layer(plan, bound, source)
    admission_assurance = _assurance_layer(assurance_value)
    lifecycle = _lifecycle_layer(plan, assurance_value)
    parent_digest = (
        None if parent_checkpoint is None else canonical_sha256(dict(parent_checkpoint))
    )
    attempt_admission = _attempt_admission_layer(
        plan,
        bound,
        source,
        proposed_reason=proposed_reason,
        child_restore_checkpoint_digest=parent_digest,
    )
    run_contract = plan.get("run_contract")
    run_contract_digest = canonical_sha256(
        {"absent": True} if run_contract is None else run_contract
    )
    baseline_digest = canonical_sha256(baseline_value)
    concretized = ConcretizedExecutionV1(
        semantic_spec_digest=semantic.semantic_spec_digest(),
        implementation_digest=implementation.implementation_digest(),
        admission_assurance_digest=admission_assurance.admission_assurance_digest(),
        lifecycle_digest=lifecycle.lifecycle_digest(),
        attempt_admission_request_digest=(
            attempt_admission.attempt_admission_request_digest()
        ),
        campaign_plan_digest=plan_digest,
        bound_execution_request_digest=bound_digest,
        run_contract_digest=run_contract_digest,
        baseline_digest=baseline_digest,
        route_contract_digest=route_digest,
        assembly_digest=semantic.system_assembly_digest,
        qualification_key_digest=qualification_digest,
    )
    branch_digest = None if branch_spec is None else canonical_sha256(dict(branch_spec))
    run = RunOccurrenceV1.create(
        experiment_id=semantic.experiment_id,
        cell_or_repeat_ordinal=ordinal,
        semantic_spec_digest=semantic.semantic_spec_digest(),
        parent_checkpoint_digest=parent_digest,
        branch_spec_digest=branch_digest,
    )
    attempt_id = derive_attempt_id(
        run_id=run.run_id,
        monotonic_ordinal=attempt_number,
        nonce=nonce,
    )

    payload: dict[str, Any] = {
        "schema_version": SHADOW_PROJECTION_SCHEMA_VERSION,
        "authority": SHADOW_PROJECTION_AUTHORITY,
        "layers": {
            "semantic": {
                "payload": semantic.to_dict(),
                "digest": semantic.semantic_spec_digest().to_dict(),
            },
            "implementation": {
                "payload": implementation.to_dict(),
                "digest": implementation.implementation_digest().to_dict(),
            },
            "admission_assurance": {
                "payload": admission_assurance.to_dict(),
                "digest": admission_assurance.admission_assurance_digest().to_dict(),
            },
            "lifecycle": {
                "payload": lifecycle.to_dict(),
                "digest": lifecycle.lifecycle_digest().to_dict(),
            },
            "attempt_admission_request": {
                "payload": attempt_admission.to_dict(),
                "digest": attempt_admission.attempt_admission_request_digest().to_dict(),
            },
            "concretized_execution": {
                "payload": concretized.to_dict(),
                "digest": concretized.concretized_execution_digest().to_dict(),
            },
        },
        "current_lock_bindings": {
            "campaign_plan_digest": plan_digest.to_dict(),
            "bound_execution_request_digest": bound_digest.to_dict(),
            "run_contract_digest": run_contract_digest.to_dict(),
            "baseline_digest": baseline_digest.to_dict(),
            "route_contract_digest": route_digest.to_dict(),
            "assembly_digest": semantic.system_assembly_digest.to_dict(),
            "qualification_key_digest": qualification_digest.to_dict(),
        },
        "shadow_run_projection": {
            "occurrence_created": False,
            "reason": "shadow projection does not create an authoritative Run record",
            "candidate": run.to_dict(),
        },
        "shadow_attempt_projection": {
            "attempt_created": False,
            "execution_side_effect_observed": False,
            "state": "NOT_STARTED_SHADOW",
            "candidate_attempt_id": str(attempt_id),
            "run_id": str(run.run_id),
            "monotonic_ordinal": attempt_number,
            "nonce": nonce,
            "concretized_execution_digest": concretized.concretized_execution_digest().to_dict(),
            "reason": "no admitted lease or execution side effect exists",
        },
    }
    payload["projection_sha256"] = canonical_sha256(payload).hex
    return validate_shadow_projection_v1(payload)


def validate_shadow_projection_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct every derived digest and candidate identity exactly."""

    receipt = _mapping(value, field="shadow_projection")
    _exact(
        receipt,
        {
            "schema_version",
            "authority",
            "layers",
            "current_lock_bindings",
            "shadow_run_projection",
            "shadow_attempt_projection",
            "projection_sha256",
        },
        field="shadow_projection",
    )
    if receipt.get("schema_version") != SHADOW_PROJECTION_SCHEMA_VERSION:
        raise ShadowProjectionError("unsupported shadow projection schema")
    if receipt.get("authority") != SHADOW_PROJECTION_AUTHORITY:
        raise ShadowProjectionError("shadow projection authority marker differs")
    declared_projection = _sha256_hex(
        receipt.get("projection_sha256"), field="shadow_projection.projection_sha256"
    )
    if canonical_sha256(_without(receipt, "projection_sha256")) != declared_projection:
        raise ShadowProjectionError("shadow projection SHA-256 differs")

    layers = _mapping(receipt.get("layers"), field="shadow_projection.layers")
    required_layers = {
        "semantic",
        "implementation",
        "admission_assurance",
        "lifecycle",
        "attempt_admission_request",
        "concretized_execution",
    }
    if set(layers) != required_layers:
        raise ShadowProjectionError("shadow projection layer set differs")
    layer_digests: dict[str, ContentDigest] = {}
    for name in sorted(required_layers):
        row = _mapping(layers[name], field=f"layers.{name}")
        if set(row) != {"payload", "digest"}:
            raise ShadowProjectionError(f"layers.{name} has an invalid shape")
        payload = _mapping(row["payload"], field=f"layers.{name}.payload")
        declared = _content_digest(row["digest"], field=f"layers.{name}.digest")
        if _reconstruct_layer_digest(name, payload) != declared:
            raise ShadowProjectionError(f"layers.{name} digest differs")
        layer_digests[name] = declared

    concretized = _mapping(
        layers["concretized_execution"]["payload"],
        field="layers.concretized_execution.payload",
    )
    layer_links = {
        "semantic_spec_digest": "semantic",
        "implementation_digest": "implementation",
        "admission_assurance_digest": "admission_assurance",
        "lifecycle_digest": "lifecycle",
        "attempt_admission_request_digest": "attempt_admission_request",
    }
    for field_name, layer_name in layer_links.items():
        if (
            _content_digest(
                concretized.get(field_name), field=f"concretized_execution.{field_name}"
            )
            != layer_digests[layer_name]
        ):
            raise ShadowProjectionError(
                f"concretized_execution.{field_name} differs from its layer"
            )
    locks = _mapping(
        receipt.get("current_lock_bindings"), field="current_lock_bindings"
    )
    lock_fields = {
        "campaign_plan_digest",
        "bound_execution_request_digest",
        "run_contract_digest",
        "baseline_digest",
        "route_contract_digest",
        "assembly_digest",
        "qualification_key_digest",
    }
    if set(locks) != lock_fields:
        raise ShadowProjectionError("current lock binding set differs")
    for field_name in sorted(lock_fields):
        if _content_digest(locks[field_name], field=f"locks.{field_name}") != (
            _content_digest(
                concretized.get(field_name),
                field=f"concretized_execution.{field_name}",
            )
        ):
            raise ShadowProjectionError(f"current lock {field_name} differs")

    run_projection = _mapping(
        receipt.get("shadow_run_projection"), field="shadow_run_projection"
    )
    _exact(
        run_projection,
        {"occurrence_created", "reason", "candidate"},
        field="shadow_run_projection",
    )
    if run_projection.get("occurrence_created") is not False:
        raise ShadowProjectionError("shadow projection cannot create a Run occurrence")
    run = _mapping(
        run_projection.get("candidate"), field="shadow_run_projection.candidate"
    )
    _exact(
        run,
        {
            "schema_version",
            "run_id",
            "experiment_id",
            "cell_or_repeat_ordinal",
            "semantic_spec_digest",
            "parent_checkpoint_digest",
            "branch_spec_digest",
        },
        field="shadow_run_projection.candidate",
    )
    if run.get("schema_version") != "crca_run_occurrence_v1":
        raise ShadowProjectionError("shadow Run candidate schema differs")
    semantic_payload = _mapping(
        layers["semantic"]["payload"], field="layers.semantic.payload"
    )
    semantic_digest = layer_digests["semantic"]
    parent = run.get("parent_checkpoint_digest")
    branch = run.get("branch_spec_digest")
    expected_run = derive_run_id(
        experiment_id=ExperimentId(
            _text(run.get("experiment_id"), field="run.experiment_id")
        ),
        cell_or_repeat_ordinal=_integer(
            run.get("cell_or_repeat_ordinal"), field="run.cell_or_repeat_ordinal"
        ),
        semantic_spec_digest=semantic_digest,
        parent_checkpoint_digest=(
            None
            if parent is None
            else _content_digest(parent, field="run.parent_checkpoint_digest")
        ),
        branch_spec_digest=(
            None
            if branch is None
            else _content_digest(branch, field="run.branch_spec_digest")
        ),
    )
    if run.get("run_id") != str(expected_run):
        raise ShadowProjectionError("shadow Run candidate identity differs")
    if run.get("semantic_spec_digest") != semantic_digest.to_dict():
        raise ShadowProjectionError("shadow Run semantic digest differs")
    if semantic_payload.get("experiment_id") != run.get("experiment_id"):
        raise ShadowProjectionError("shadow Run experiment differs from semantic layer")

    attempt = _mapping(
        receipt.get("shadow_attempt_projection"), field="shadow_attempt_projection"
    )
    _exact(
        attempt,
        {
            "attempt_created",
            "execution_side_effect_observed",
            "state",
            "candidate_attempt_id",
            "run_id",
            "monotonic_ordinal",
            "nonce",
            "concretized_execution_digest",
            "reason",
        },
        field="shadow_attempt_projection",
    )
    if (
        attempt.get("attempt_created") is not False
        or attempt.get("execution_side_effect_observed") is not False
        or attempt.get("state") != "NOT_STARTED_SHADOW"
    ):
        raise ShadowProjectionError("shadow projection claims a runtime Attempt")
    if attempt.get("run_id") != str(expected_run):
        raise ShadowProjectionError("shadow Attempt RunId differs")
    expected_attempt = derive_attempt_id(
        run_id=RunId(str(expected_run)),
        monotonic_ordinal=_integer(
            attempt.get("monotonic_ordinal"),
            field="attempt.monotonic_ordinal",
            minimum=1,
        ),
        nonce=_text(attempt.get("nonce"), field="attempt.nonce"),
    )
    if attempt.get("candidate_attempt_id") != str(expected_attempt):
        raise ShadowProjectionError("shadow Attempt candidate identity differs")
    if (
        _content_digest(
            attempt.get("concretized_execution_digest"),
            field="attempt.concretized_execution_digest",
        )
        != layer_digests["concretized_execution"]
    ):
        raise ShadowProjectionError("shadow Attempt concretized digest differs")
    return json.loads(json.dumps(receipt, sort_keys=True))


def project_from_input_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project one portable current-lock bundle used by the CLI and tests."""

    raw = _mapping(value, field="projection_input")
    _exact(
        raw,
        {
            "schema_version",
            "campaign_plan",
            "bound_execution_request",
            "baseline",
            "source_identity",
            "assurance",
            "cell_or_repeat_ordinal",
            "attempt_ordinal",
            "attempt_nonce",
            "proposed_reason",
            "parent_checkpoint",
            "branch_spec",
        },
        field="projection_input",
    )
    if raw.get("schema_version") != SHADOW_PROJECTION_INPUT_SCHEMA_VERSION:
        raise ShadowProjectionError("unsupported shadow projection input schema")
    return project_current_execution_v1(
        _mapping(raw.get("campaign_plan"), field="campaign_plan"),
        _mapping(raw.get("bound_execution_request"), field="bound_execution_request"),
        _mapping(raw.get("baseline"), field="baseline"),
        _mapping(raw.get("source_identity"), field="source_identity"),
        _mapping(raw.get("assurance"), field="assurance"),
        cell_or_repeat_ordinal=raw.get("cell_or_repeat_ordinal", 0),
        attempt_ordinal=raw.get("attempt_ordinal", 1),
        attempt_nonce=raw.get("attempt_nonce", "shadow-projection"),
        proposed_reason=raw.get("proposed_reason", "shadow identity projection"),
        parent_checkpoint=(
            None
            if raw.get("parent_checkpoint") is None
            else _mapping(raw["parent_checkpoint"], field="parent_checkpoint")
        ),
        branch_spec=(
            None
            if raw.get("branch_spec") is None
            else _mapping(raw["branch_spec"], field="branch_spec")
        ),
    )


def write_shadow_projection_v1(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically write a validated shadow receipt outside execution authority."""

    value = validate_shadow_projection_v1(receipt)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def inspect_shadow_projection_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = validate_shadow_projection_v1(value)
    layers = receipt["layers"]
    run = receipt["shadow_run_projection"]["candidate"]
    attempt = receipt["shadow_attempt_projection"]
    semantic = layers["semantic"]["payload"]
    return {
        "schema_version": "crca_shadow_projection_inspection_v1",
        "status": "VALID_SHADOW_PROJECTION",
        "authority": receipt["authority"],
        "projection_sha256": receipt["projection_sha256"],
        "campaign_id": semantic["experiment_id"],
        "absolute_span": {
            "start_next_tick": semantic["absolute_start_next_tick"],
            "target_next_tick": semantic["absolute_target_next_tick"],
            "trajectory_checkpoint_ticks": semantic["trajectory_checkpoint_ticks"],
        },
        "run_id_candidate": run["run_id"],
        "attempt_id_candidate": attempt["candidate_attempt_id"],
        "run_created": False,
        "attempt_created": False,
        "layer_digests": {name: row["digest"] for name, row in sorted(layers.items())},
        "current_lock_bindings": receipt["current_lock_bindings"],
    }


def _changed_paths(left: Any, right: Any, path: tuple[str, ...] = ()) -> list[str]:
    if type(left) is not type(right):
        return [".".join(path) or "<root>"]
    if isinstance(left, dict):
        findings: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = (*path, key)
            if key not in left or key not in right:
                findings.append(".".join(child))
            else:
                findings.extend(_changed_paths(left[key], right[key], child))
        return findings
    if isinstance(left, list):
        if len(left) != len(right):
            return [".".join(path) or "<root>"]
        findings = []
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            findings.extend(_changed_paths(left_item, right_item, (*path, str(index))))
        return findings
    return [] if left == right else [".".join(path) or "<root>"]


def diff_shadow_projections_v1(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    before = validate_shadow_projection_v1(left)
    after = validate_shadow_projection_v1(right)
    changed_layers = [
        name
        for name in sorted(before["layers"])
        if before["layers"][name]["digest"] != after["layers"][name]["digest"]
    ]
    semantic_changed = "semantic" in changed_layers
    implementation_changed = "implementation" in changed_layers
    admission_changed = "attempt_admission_request" in changed_layers
    lifecycle_changed = "lifecycle" in changed_layers
    assurance_changed = "admission_assurance" in changed_layers
    if (
        not changed_layers
        and before["current_lock_bindings"] == after["current_lock_bindings"]
    ):
        decision = "IDENTICAL_RECONSTRUCTION"
    elif semantic_changed:
        decision = "SUCCESSOR_OR_SIBLING_RUN_REQUIRED"
    elif implementation_changed:
        decision = "SAME_RUN_NEW_ATTEMPT_REQUIRES_EXACT_CONFORMANCE"
    elif admission_changed or lifecycle_changed:
        decision = "SAME_RUN_NEW_ATTEMPT_CANDIDATE"
    elif (
        assurance_changed
        or before["current_lock_bindings"] != after["current_lock_bindings"]
    ):
        decision = "PREADMISSION_REEVALUATION_NO_ATTEMPT"
    else:
        decision = "SHADOW_ONLY_DIFFERENCE"
    paths = _changed_paths(
        _without(before, "projection_sha256"),
        _without(after, "projection_sha256"),
    )
    return {
        "schema_version": "crca_shadow_projection_diff_v1",
        "authority": SHADOW_PROJECTION_AUTHORITY,
        "left_projection_sha256": before["projection_sha256"],
        "right_projection_sha256": after["projection_sha256"],
        "identical": not paths,
        "changed_layers": changed_layers,
        "changed_paths": paths,
        "semantic_run_id_preserved": (
            before["shadow_run_projection"]["candidate"]["run_id"]
            == after["shadow_run_projection"]["candidate"]["run_id"]
        ),
        "decision": decision,
        "execution_authority": False,
        "qualification_authority": False,
    }


def explain_shadow_projection_diff_v1(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    diff = diff_shadow_projections_v1(left, right)
    layer_text = (
        "no causal layers changed"
        if not diff["changed_layers"]
        else "changed causal layers: " + ", ".join(diff["changed_layers"])
    )
    explanation = [
        f"Decision: {diff['decision']}.",
        f"Identity comparison found {layer_text}.",
        (
            "The declared Run identity is preserved."
            if diff["semantic_run_id_preserved"]
            else "The declared Run identity changes."
        ),
        "This is a shadow explanation and grants no execution or Qualification authority.",
    ]
    return {
        "schema_version": "crca_shadow_projection_explanation_v1",
        "authority": SHADOW_PROJECTION_AUTHORITY,
        "decision": diff["decision"],
        "explanation": explanation,
        "diff": diff,
    }


__all__ = [
    "SHADOW_PROJECTION_AUTHORITY",
    "SHADOW_PROJECTION_INPUT_SCHEMA_VERSION",
    "SHADOW_PROJECTION_SCHEMA_VERSION",
    "ShadowProjectionError",
    "diff_shadow_projections_v1",
    "explain_shadow_projection_diff_v1",
    "inspect_shadow_projection_v1",
    "project_current_execution_v1",
    "project_from_input_v1",
    "validate_shadow_projection_v1",
    "write_shadow_projection_v1",
]
