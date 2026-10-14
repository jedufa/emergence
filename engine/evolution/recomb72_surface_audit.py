"""Machine-readable production inheritance-surface audit for RECOMB-72.

The payload binds the existing RECOMB-1 production registry verbatim.  The
additional rows classify source ownership and evidence coverage; they do not
define or execute an alternative inheritance mechanic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from agents.modular_genome_v1 import B2_FROZEN_CAPS
from evolution.inheritance_registry_v1 import (
    INHERITANCE_REGISTRY_VERSION,
    LINKED_INHERITANCE_PROFILE,
    LINKED_PROFILE_DRAW_COUNT,
    SUPPORTED_VALIDATION_LANE,
    inheritance_registry_metadata,
)


SCHEMA_VERSION: Final[str] = "recomb72_production_inheritance_surface_audit_v3"
CLASSIFICATIONS: Final[tuple[str, ...]] = (
    "measured_directly",
    "inferred",
    "not_tested",
)

REFERENCE_REPRODUCTION_OWNER: Final[dict[str, str]] = {
    "path": "engine/evolution/reproduction.py",
    "symbol": "crossover_genome",
}
REFERENCE_POLICY_OWNER: Final[dict[str, str]] = {
    "path": "engine/evolution/reproduction.py",
    "symbol": "crossover_policy",
}
REFERENCE_STRUCTURAL_OWNER: Final[dict[str, str]] = {
    "path": "engine/evolution/recombination_c2.py",
    "symbol": "recombine_lineage_payloads",
}
RESIDENT_NATIVE_OWNER: Final[dict[str, str]] = {
    "path": "engine/perf/native/cxxbackbone10_resident_state_owner.cpp",
    "symbol": "ResidentRuntimeStateV10::execute_population_generation_turnover_cpu",
}
CUDA_NATIVE_OWNER: Final[dict[str, str]] = {
    "path": "engine/perf/native/cuda_architecture_b_lifecycle.cu",
    "symbol": "eiel_cuda_architecture_b_turnover_fixed_mlp_child_scalar_genome_v1",
}
RECOMB72_GPU_NATIVE_OWNER: Final[dict[str, str]] = {
    "path": "engine/perf/native/recomb72_gpu_native.cu",
    "symbol": "recomb72_gpu_advance_v1",
}
PRODUCTION_ENGINE_OWNER: Final[dict[str, str]] = {
    "path": "engine/campaigns/production_engine_provider.py",
    "symbol": "EielProductionExecutionProviderV1",
}
RECOMB72_GPU_PRODUCTION_PARITY_TEST: Final[dict[str, str]] = {
    "path": "tests/perf/test_recomb72_gpu_native.py",
    "symbol": "test_gpu_current_mode_matches_production_crossover_for_every_g1_event",
}
AUDITED_SCALAR_LOCI: Final[tuple[str, ...]] = (
    "exploration_bias",
    "aggression_bias",
    "action_temperature",
    "resource_handling_axis",
    "drive_inertia",
    "pred_lr",
)


class Recomb72SurfaceAuditError(RuntimeError):
    """Raised when the production registry cannot satisfy the frozen audit."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _source(
    classification: str,
    owner: Mapping[str, str] | None,
    *,
    finding: str,
) -> dict[str, Any]:
    if classification not in CLASSIFICATIONS:
        raise Recomb72SurfaceAuditError(
            f"unsupported evidence classification: {classification!r}"
        )
    return {
        "classification": classification,
        "owner": dict(owner) if owner is not None else None,
        "finding": str(finding),
    }


def _registry_index(
    registry: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_loci = registry.get("loci")
    raw_blocks = registry.get("blocks")
    if not isinstance(raw_loci, list) or not isinstance(raw_blocks, list):
        raise Recomb72SurfaceAuditError(
            "production inheritance registry lacks loci or blocks"
        )
    loci: dict[str, dict[str, Any]] = {}
    for raw in raw_loci:
        if not isinstance(raw, Mapping):
            raise Recomb72SurfaceAuditError("production registry locus is not a mapping")
        locus_id = str(raw.get("semantic_locus_id", ""))
        if not locus_id or locus_id in loci:
            raise Recomb72SurfaceAuditError(
                f"production registry has invalid locus id: {locus_id!r}"
            )
        loci[locus_id] = dict(raw)
    blocks: dict[str, dict[str, Any]] = {}
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            raise Recomb72SurfaceAuditError("production registry block is not a mapping")
        block_id = str(raw.get("id", ""))
        if not block_id or block_id in blocks:
            raise Recomb72SurfaceAuditError(
                f"production registry has invalid block id: {block_id!r}"
            )
        blocks[block_id] = dict(raw)
    return loci, blocks


def _select_loci(
    loci: Mapping[str, Mapping[str, Any]],
    locus_ids: Sequence[str],
) -> list[dict[str, Any]]:
    missing = [locus_id for locus_id in locus_ids if locus_id not in loci]
    if missing:
        raise Recomb72SurfaceAuditError(
            f"production registry is missing required loci: {missing!r}"
        )
    return [dict(loci[locus_id]) for locus_id in locus_ids]


def _surface(
    *,
    surface_id: str,
    registry_locus_ids: Sequence[str],
    loci: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Any],
    resident_native: Mapping[str, Any],
    cuda_native: Mapping[str, Any],
    campaign_measurement: Mapping[str, Any],
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "surface_id": surface_id,
        "registry_locus_ids": list(registry_locus_ids),
        "registry_rows": _select_loci(loci, registry_locus_ids),
        "source_classification": {
            "python_reference": dict(reference),
            "resident_native_cpu": dict(resident_native),
            "cuda_lifecycle": dict(cuda_native),
        },
        "recomb72_campaign_measurement": dict(campaign_measurement),
        "notes": list(notes),
    }


def surface_audit_payload() -> dict[str, Any]:
    """Return a deterministic audit payload bound to the live registry."""

    registry = inheritance_registry_metadata()
    if registry.get("registry_version") != INHERITANCE_REGISTRY_VERSION:
        raise Recomb72SurfaceAuditError("production registry version mismatch")
    loci, blocks = _registry_index(registry)

    scalar_ids = tuple(
        f"flat.{field}"
        for block_id in (
            "behavioral_strategy_v1",
            "regulation_prediction_v1",
            "life_history_v1",
            "mutation_rate_v1",
        )
        for field in blocks[block_id]["fields"]
    )
    audited_scalar_ids = tuple(
        f"flat.{locus}" for locus in AUDITED_SCALAR_LOCI
    )
    other_scalar_ids = tuple(
        locus_id
        for locus_id in scalar_ids
        if locus_id not in audited_scalar_ids
    )
    memory_ids = tuple(
        f"flat.{field}"
        for field in ("hidden_size", "memory_dim", "use_gru")
    )
    mask_ids = tuple(
        locus_id
        for locus_id in loci
        if locus_id.startswith("categorical.")
    )
    module_ids = (
        "structural.module_record",
        "structural.module_live_operator_parameters",
        "structural.input_routing_record",
        "structural.output_routing_record",
        "structural.dormancy_record",
    )
    connection_ids = ("structural.connection_record",)
    developmental_ids = (
        "developmental.top_level_program",
        "structural.developmental_record",
    )

    inferred_reference = _source(
        "inferred",
        REFERENCE_REPRODUCTION_OWNER,
        finding="semantics classified from the production source and bound registry",
    )
    not_measured = _source(
        "not_tested",
        None,
        finding="the current RECOMB-72 campaign does not invoke this production surface",
    )
    measured_gpu_scalars = {
        **_source(
            "measured_directly",
            RECOMB72_GPU_NATIVE_OWNER,
            finding=(
                "the qualified native GPU fixture directly records parent, mate, "
                "child, population, and ancestry tensors for exactly these six loci"
            ),
        ),
        "audited_loci": list(AUDITED_SCALAR_LOCI),
        "historical_execution_path": {
            "status": "removed_from_production",
            "provider_id": "recomb72-gpu-native-cuda-resident-provider-v1",
            "profile_id": "recomb72-neutral-inheritance-cuda-resident-v1",
        },
        "qualification_contract": {
            "profile_id": "eiel-production-runtime-v1",
            "provider_id": "eiel-production-provider-v1",
            "qualification_status": "component_not_migrated_fail_closed",
            "native_abi": "recomb72_gpu_native_v1",
            "route_contract_owner": {
                "path": "engine/campaigns/production_engine_provider.py",
                "symbol": "EielProductionExecutionProviderV1.execution_route_contract",
            },
            "required_obligations": [],
        },
        "production_current_parity": {
            "classification": "measured_directly",
            "test": dict(RECOMB72_GPU_PRODUCTION_PARITY_TEST),
            "production_contract": {
                "path": "engine/evolution/reproduction.py",
                "symbol": "crossover_genome",
                "inheritance_profile": "legacy_midpoint_v1",
                "event_contract": "production_current_path_v1",
            },
            "native_semantics": "exact_float32_legacy_parent_midpoint",
            "exact_every_g1_event": True,
            "general_production_crossover_invoked_by_campaign": False,
            "general_production_crossover_used_as_test_oracle": True,
        },
    }

    surfaces = [
        _surface(
            surface_id="recomb72_audited_scalar_loci",
            registry_locus_ids=audited_scalar_ids,
            loci=loci,
            reference=inferred_reference,
            resident_native=_source(
                "inferred",
                RESIDENT_NATIVE_OWNER,
                finding="midpoint inheritance followed by bounded mutation",
            ),
            cuda_native=_source(
                "inferred",
                CUDA_NATIVE_OWNER,
                finding="midpoint inheritance followed by bounded mutation",
            ),
            campaign_measurement=measured_gpu_scalars,
            notes=(
                "dedicated_native_fixture_reproduces_exact_legacy_midpoint_semantics",
                "campaign_does_not_invoke_general_production_crossover_function",
            ),
        ),
        _surface(
            surface_id="other_bounded_scalar_genes",
            registry_locus_ids=other_scalar_ids,
            loci=loci,
            reference=inferred_reference,
            resident_native=_source(
                "inferred",
                RESIDENT_NATIVE_OWNER,
                finding="midpoint inheritance followed by bounded mutation",
            ),
            cuda_native=_source(
                "inferred",
                CUDA_NATIVE_OWNER,
                finding="midpoint inheritance followed by bounded mutation",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="memory_architecture",
            registry_locus_ids=memory_ids,
            loci=loci,
            reference=inferred_reference,
            resident_native=_source(
                "inferred",
                RESIDENT_NATIVE_OWNER,
                finding="hidden-size donor plus midpoint-snapped memory mutation",
            ),
            cuda_native=_source(
                "inferred",
                CUDA_NATIVE_OWNER,
                finding="fixed-MLP turnover forces memory disabled after inherited draw schedule",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="categorical_masks_and_sensor_radius",
            registry_locus_ids=mask_ids,
            loci=loci,
            reference=inferred_reference,
            resident_native=_source(
                "inferred",
                RESIDENT_NATIVE_OWNER,
                finding="sanitized parent-A copy in resident turnover",
            ),
            cuda_native=_source(
                "inferred",
                CUDA_NATIVE_OWNER,
                finding="sanitized parent-A copy in fixed-MLP lifecycle",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="family_head_residual_record",
            registry_locus_ids=("flat.family_head_residuals",),
            loci=loci,
            reference=_source(
                "inferred",
                {
                    "path": "engine/agents/family_head_residuals.py",
                    "symbol": "recombine_family_head_residual_blocks",
                },
                finding="legacy midpoint-center-project or linked whole-record donor",
            ),
            resident_native=_source(
                "not_tested",
                RESIDENT_NATIVE_OWNER,
                finding="gap: resident turnover has no family-head residual record",
            ),
            cuda_native=_source(
                "not_tested",
                CUDA_NATIVE_OWNER,
                finding="gap: CUDA lifecycle has no family-head residual record",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="mlp_policy_controller",
            registry_locus_ids=("flat.policy_controller",),
            loci=loci,
            reference=_source(
                "inferred",
                REFERENCE_POLICY_OWNER,
                finding="legacy parameter-wise masks; linked profile whole-controller donor",
            ),
            resident_native=_source(
                "inferred",
                RESIDENT_NATIVE_OWNER,
                finding="resident fixed-MLP turnover copies parent-A row then mutates by policy tape",
            ),
            cuda_native=_source(
                "inferred",
                CUDA_NATIVE_OWNER,
                finding="fixed-MLP scalar lifecycle delegates policy-row transaction",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="gru_policy_controller",
            registry_locus_ids=("flat.policy_controller",),
            loci=loci,
            reference=_source(
                "inferred",
                {
                    "path": "engine/evolution/population_pkg/spawn.py",
                    "symbol": "make_child_row_spec",
                },
                finding="legacy single-GRU-parent copy; linked whole-controller donor",
            ),
            resident_native=_source(
                "inferred",
                RESIDENT_NATIVE_OWNER,
                finding="single compatible GRU donor row; newborn hidden memory is zeroed",
            ),
            cuda_native=_source(
                "not_tested",
                CUDA_NATIVE_OWNER,
                finding="fixed-MLP lifecycle entrypoint does not test GRU controller inheritance",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="structural_modules_and_routing",
            registry_locus_ids=module_ids,
            loci=loci,
            reference=_source(
                "inferred",
                REFERENCE_STRUCTURAL_OWNER,
                finding="homology-aligned whole-module donor with unmatched union and caps",
            ),
            resident_native=_source(
                "not_tested",
                RESIDENT_NATIVE_OWNER,
                finding="gap: generation turnover rejects C1/C2 structural genetics",
            ),
            cuda_native=_source(
                "not_tested",
                CUDA_NATIVE_OWNER,
                finding="gap: scalar lifecycle does not carry structural module records",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="structural_connections",
            registry_locus_ids=connection_ids,
            loci=loci,
            reference=_source(
                "inferred",
                REFERENCE_STRUCTURAL_OWNER,
                finding="homology-endpoint-aligned whole-connection donor",
            ),
            resident_native=_source(
                "not_tested",
                RESIDENT_NATIVE_OWNER,
                finding="gap: generation turnover rejects C1/C2 structural genetics",
            ),
            cuda_native=_source(
                "not_tested",
                CUDA_NATIVE_OWNER,
                finding="gap: scalar lifecycle does not carry structural connections",
            ),
            campaign_measurement=not_measured,
        ),
        _surface(
            surface_id="developmental_records",
            registry_locus_ids=developmental_ids,
            loci=loci,
            reference=_source(
                "inferred",
                REFERENCE_STRUCTURAL_OWNER,
                finding="top-level whole-record donor plus module-local atomic donor",
            ),
            resident_native=_source(
                "not_tested",
                RESIDENT_NATIVE_OWNER,
                finding="gap: no developmental record turnover under C1/C2 rejection",
            ),
            cuda_native=_source(
                "not_tested",
                CUDA_NATIVE_OWNER,
                finding="gap: scalar lifecycle does not carry developmental records",
            ),
            campaign_measurement=not_measured,
        ),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "classification_vocabulary": list(CLASSIFICATIONS),
        "classification_meanings": {
            "measured_directly": "bound or compared by a direct machine assertion",
            "inferred": "classified from production source and registry ownership",
            "not_tested": "not executed or measured by the current RECOMB-72 route",
        },
        "production_registry_binding": {
            "classification": "measured_directly",
            "owner": {
                "path": "engine/evolution/inheritance_registry_v1.py",
                "symbol": "inheritance_registry_metadata",
            },
            "registry_version": registry["registry_version"],
            "registry_metadata_sha256": registry["registry_metadata_sha256"],
            "registry": registry,
        },
        "existing_repair": {
            "profile": LINKED_INHERITANCE_PROFILE,
            "classification": "measured_directly",
            "draws_per_offspring": LINKED_PROFILE_DRAW_COUNT,
            "supported_lane": SUPPORTED_VALIDATION_LANE,
            "native_supported": False,
            "description": (
                "existing linked_parent_copy_v1 repair uses seven linkage-block "
                "donors and is qualified only on the CPU Python reference lane"
            ),
        },
        "surfaces": surfaces,
        "homology_and_capacity_metadata": {
            "classification": "inferred",
            "owner": {
                "path": "engine/agents/modular_genome_v1.py",
                "symbol": "resolve_b2_caps",
            },
            "registry_structural_linkage_rule": registry["structural_linkage_rule"],
            "homology_fields": [
                "module_instance_id",
                "homology_id",
                "inherited_homology_id",
                "structural_ancestry",
            ],
            "capacity_fields": [
                "b2_caps",
                *[f"b2_caps.{field}" for field in B2_FROZEN_CAPS],
            ],
            "frozen_capacity_values": dict(B2_FROZEN_CAPS),
            "native_turnover": {
                "classification": "not_tested",
                "finding": "resident turnover rejects C1/C2 structural genetics",
            },
            "recomb72_campaign_measurement": dict(not_measured),
        },
        "native_gaps": [
            {
                "gap_id": "resident_resource_axis_absent",
                "classification": "inferred",
                "owner": dict(RESIDENT_NATIVE_OWNER),
                "effect": (
                    "the general resident turnover does not carry this field; "
                    "the dedicated RECOMB-72 GPU fixture owns the measured route"
                ),
            },
            {
                "gap_id": "resident_family_head_absent",
                "classification": "inferred",
                "owner": dict(RESIDENT_NATIVE_OWNER),
                "effect": "family-head production inheritance remains reference-only",
            },
            {
                "gap_id": "resident_structural_genetics_rejected",
                "classification": "inferred",
                "owner": dict(RESIDENT_NATIVE_OWNER),
                "effect": "modules, connections, development, homology, and caps are excluded",
            },
            {
                "gap_id": "linked_parent_copy_native_rejected",
                "classification": "measured_directly",
                "owner": {
                    "path": "engine/evolution/inheritance_registry_v1.py",
                    "symbol": "validate_inheritance_execution_lane",
                },
                "effect": "linked_parent_copy_v1 cannot qualify a native RECOMB-72 route",
            },
        ],
        "campaign_authority_decision": {
            "classification": "measured_directly",
            "current_route_invokes_production_inheritance": False,
            "current_route_directly_measures_audited_scalar_loci": True,
            "audited_loci": list(AUDITED_SCALAR_LOCI),
            "native_fixture_reproduces_legacy_midpoint_semantics": True,
            "general_production_crossover_invoked_by_campaign": False,
            "general_production_crossover_used_as_parity_oracle": True,
            "production_current_parity_test": dict(
                RECOMB72_GPU_PRODUCTION_PARITY_TEST
            ),
            "finding": (
                "the qualified dedicated GPU fixture directly measures exactly six "
                "scalar loci and reproduces the legacy production-current midpoint "
                "semantics exactly; it does not invoke the general production "
                "crossover function during campaign execution"
            ),
            "owner": dict(PRODUCTION_ENGINE_OWNER),
        },
    }


def surface_audit_sha256() -> str:
    """Return the SHA-256 of the canonical payload."""

    return hashlib.sha256(_canonical_bytes(surface_audit_payload())).hexdigest()


def surface_audit_artifact() -> dict[str, Any]:
    """Return the payload with its independently reproducible canonical hash."""

    payload = surface_audit_payload()
    return {
        **payload,
        "payload_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }


def write_surface_audit(path: str | Path) -> Path:
    """Write one canonical JSON artifact and return its resolved path."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_bytes(surface_audit_artifact()) + b"\n")
    return target


__all__ = [
    "AUDITED_SCALAR_LOCI",
    "CLASSIFICATIONS",
    "Recomb72SurfaceAuditError",
    "SCHEMA_VERSION",
    "surface_audit_artifact",
    "surface_audit_payload",
    "surface_audit_sha256",
    "write_surface_audit",
]
