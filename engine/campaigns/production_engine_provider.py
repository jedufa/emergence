"""The single EIEL production execution provider.

The engine owns lifecycle, batching, native/CUDA dispatch, checkpoints,
Chronicle handoff, reporting and analysis projections, and qualification.
World, controller, and inheritance identities are configuration. During
migration, configurations whose components are not yet owned here fail closed;
this module never forwards production execution to a legacy provider object.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from dataclasses import replace
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Final, Mapping, Sequence

from engine.evolution.genome_upgrade_v2_package_v1 import (
    PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION,
)

from .canonical import canonical_sha256
from .contracts import CampaignPlanV1, ExecutionRouteContractV1, UnsupportedReasonV1
from .gpu_resident_support import (
    GpuResidentSupportError,
    validate_safe_cell_id,
    verified_prebuilt_artifact_binding,
)
from .providers import (
    CampaignProviderError,
    FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION,
    FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION,
    PROVIDER_NATIVE_EXPOSURE_RECEIPT_SCHEMA_VERSION,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
    _acs_native_runtime_artifact_binding_v1,
    _acs_native_runtime_route_component_v1,
)
from .inheritance_cross_path_exposure_projection_contract_v1 import (
    AGENT_ID_ALLOCATION_SCHEMA_VERSION,
    InheritanceCrossPathExposureProjectionContractError,
    required_reproduction_exposure_aggregate_capacity_v1,
)
from .qualification import EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION
from .production_engine_identity import (
    PRODUCTION_CHECKPOINT_SCHEMA_VERSION,
    PRODUCTION_CPU_PROFILE_ID,
    PRODUCTION_LIFECYCLE_OWNER_ID,
    PRODUCTION_PROFILE_ID,
    PRODUCTION_PROVIDER_ID,
)
from .production_engine_configuration import production_engine_configuration_v1
from .production_engine_lifecycle_v1 import (
    production_engine_lifecycle_control_v1,
)
from .tick_native import (
    TICK_NATIVE_EXECUTION_ROUTE_V1,
    TickNativeError,
    TickNativeExecutionRequestV1,
)
from .tick_native_control import TickNativeControlError, TickNativeControlV1

PROFILE_ID: Final[str] = PRODUCTION_PROFILE_ID
CPU_PROFILE_ID: Final[str] = PRODUCTION_CPU_PROFILE_ID
PROVIDER_ID: Final[str] = PRODUCTION_PROVIDER_ID
GLOBAL_GENOME_PROFILE_SCHEMA_VERSION: Final[str] = "inheritance_cross_path_registry_v2"
# Historical import compatibility for existing artifact builders.
REGISTRY_SCHEMA_VERSION: Final[str] = GLOBAL_GENOME_PROFILE_SCHEMA_VERSION
REPRODUCTION_EXPOSURE_CAPACITY_SCHEMA_VERSION: Final[str] = (
    "eiel_resident_reproduction_exposure_capacity_v3"
)
MECHANIC_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_qualification_v2"
)
CHECKPOINT_SCHEMA_VERSION: Final[str] = PRODUCTION_CHECKPOINT_SCHEMA_VERSION
TECHNICAL_GATE_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_technical_gate_evidence_v1"
)
GENOME_V2_PACKAGE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    PACKAGE_IDENTITY_RECEIPT_SCHEMA_VERSION
)
RUNTIME_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {"world_base", "world_overlays", "policy_family", "selection_mode"}
)
RUNTIME_ARTIFACT_RELOCATION_ROOTS_V1: Final[tuple[tuple[str, str], ...]] = (
    (
        public_runtime_path('evidence_raw/'),
        public_runtime_path('knobs_discovery_evidence_raw/'),
    ),
)


def _native_api() -> ModuleType:
    return importlib.import_module("engine.perf.inheritance_cross_path_gpu_native")


def _torch() -> ModuleType:
    return importlib.import_module("torch")


def _payload(plan: CampaignPlanV1 | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, CampaignPlanV1):
        return plan.payload()
    if not isinstance(plan, Mapping):
        raise CampaignProviderError("cross-path plan must be a mapping")
    return dict(plan)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tick_native_checkpoint_coordinate_exact_v1(
    checkpoint: Mapping[str, Any],
    *,
    expected_next_absolute_tick: int,
) -> bool:
    """Validate a native checkpoint coordinate without promoting aliases."""

    absolute_tick = checkpoint.get("absolute_tick")
    next_absolute_tick = checkpoint.get("next_absolute_tick")
    if (
        type(expected_next_absolute_tick) is not int
        or expected_next_absolute_tick < 0
        or checkpoint.get("time_authority") != "absolute_tick_v1"
        or type(absolute_tick) is not int
        or absolute_tick != expected_next_absolute_tick
        or type(next_absolute_tick) is not int
        or next_absolute_tick != expected_next_absolute_tick
    ):
        return False
    return all(
        type(checkpoint.get(name)) is int
        and checkpoint.get(name) == expected_next_absolute_tick
        for name in ("checkpoint_at_tick", "next_tick")
        if name in checkpoint
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _stable_resident_owner_prebuilt_route_status(status: Any) -> dict[str, Any]:
    payload = dict(status.to_dict())
    payload.pop("loaded", None)
    return payload


def _artifact_rows(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = plan.get("artifacts")
    if not isinstance(raw, Mapping):
        raise CampaignProviderError("cross-path plan lacks frozen artifacts")
    return {str(name): row for name, row in raw.items() if isinstance(row, Mapping)}


def _artifact_path_candidates_v1(declared: Path) -> tuple[Path, ...]:
    """Return the declared artifact location and verified relocation candidates.

    A plan's declared path remains part of its frozen provenance.  This only
    supports an externally relocated runtime evidence root, and callers still
    require the declared SHA-256 before accepting the relocated file.
    """

    rendered = str(declared)
    candidates = [declared]
    for source_root, relocated_root in RUNTIME_ARTIFACT_RELOCATION_ROOTS_V1:
        if rendered.startswith(source_root):
            candidates.append(Path(relocated_root + rendered[len(source_root) :]))
    return tuple(candidates)


def _verified_hashed_path_v1(declared: Path, digest: str) -> Path | None:
    """Resolve a declared file or relocation candidate only by exact digest."""

    for candidate in _artifact_path_candidates_v1(declared):
        if candidate.is_symlink():
            continue
        path = candidate.resolve()
        if path.is_file() and _file_sha256(path) == digest:
            return path
    return None


def _verified_artifact(
    plan: Mapping[str, Any], artifact_id: str
) -> tuple[Path, Mapping[str, Any]]:
    row = _artifact_rows(plan).get(artifact_id)
    if not isinstance(row, Mapping):
        raise CampaignProviderError(f"cross-path artifact is missing: {artifact_id}")
    declared = Path(str(row.get("path") or "")).expanduser()
    digest = row.get("sha256")
    abi = row.get("abi")
    if (
        not declared.is_absolute()
        or declared.is_symlink()
        or not _valid_sha256(digest)
        or not isinstance(abi, str)
        or not abi
    ):
        raise CampaignProviderError(
            f"cross-path artifact mapping is invalid: {artifact_id}"
        )
    path = _verified_hashed_path_v1(declared, digest)
    if path is None:
        raise CampaignProviderError(
            f"cross-path artifact mapping is stale: {artifact_id}"
        )
    return path, row


def _string_int_map(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise CampaignProviderError(f"cross-path registry {field} must be an object")
    result = {
        str(name): raw
        for name, raw in value.items()
        if isinstance(name, str) and type(raw) is int and raw >= 0
    }
    if (
        len(result) != len(value)
        or not result
        or len(set(result.values())) != len(result)
    ):
        raise CampaignProviderError(
            f"cross-path registry {field} must be a non-empty unique integer map"
        )
    return result


def _semantic_types(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise CampaignProviderError(
            "cross-path registry semantic_types must be a non-empty object"
        )
    result = {str(name): kind for name, kind in value.items()}
    if (
        len(result) != len(value)
        or any(not name for name in result)
        or any(
            kind not in {"string", "boolean", "integer", "number"}
            for kind in result.values()
        )
    ):
        raise CampaignProviderError("cross-path registry semantic types are invalid")
    return result


def _semantic_token(value: Any, kind: str, *, field: str) -> str:
    if kind == "string" and isinstance(value, str) and value:
        return value
    if kind == "boolean" and type(value) is bool:
        return "true" if value else "false"
    if kind == "integer" and type(value) is int:
        return str(value)
    if kind == "number" and type(value) in {int, float} and math.isfinite(float(value)):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    raise CampaignProviderError(f"cross-path semantic value has wrong type: {field}")


def _decode_semantic_token(token: str, kind: str, *, field: str) -> Any:
    try:
        if kind == "boolean":
            if token not in {"false", "true"}:
                raise ValueError
            return token == "true"
        if kind == "integer":
            return int(token)
        if kind == "number":
            return float(token)
        return token
    except (TypeError, ValueError, OverflowError) as exc:
        raise CampaignProviderError(
            f"cross-path semantic ID token is invalid: {field}"
        ) from exc


def _semantic_id_maps(
    value: Any, semantic_types: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or not value:
        raise CampaignProviderError(
            "cross-path registry semantic_ids must be a non-empty object"
        )
    result: dict[str, dict[str, int]] = {}
    for semantic, values in value.items():
        if (
            not isinstance(semantic, str)
            or not semantic
            or semantic not in semantic_types
        ):
            raise CampaignProviderError(
                "cross-path registry semantic IDs require non-empty axis names"
            )
        rows = _string_int_map(values, field=f"semantic_ids.{semantic}")
        for token in rows:
            if (
                _semantic_token(
                    _decode_semantic_token(
                        token,
                        semantic_types[semantic],
                        field=f"semantic_ids.{semantic}.{token}",
                    ),
                    semantic_types[semantic],
                    field=f"semantic_ids.{semantic}.{token}",
                )
                != token
            ):
                raise CampaignProviderError(
                    "cross-path semantic ID token is not canonical"
                )
        result[semantic] = rows
    if set(result) != set(semantic_types):
        raise CampaignProviderError("cross-path semantic types and IDs differ")
    return result


def _runtime_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != RUNTIME_IDENTITY_FIELDS:
        raise CampaignProviderError(
            "cross-path registry runtime identity shape mismatch"
        )
    overlays = value["world_overlays"]
    if (
        any(
            not isinstance(value[name], str) or not value[name]
            for name in ("world_base", "policy_family", "selection_mode")
        )
        or not isinstance(overlays, list)
        or any(not isinstance(item, str) or not item for item in overlays)
        or len(set(overlays)) != len(overlays)
        or overlays != sorted(overlays)
    ):
        raise CampaignProviderError("cross-path registry runtime identity is invalid")
    return {
        "world_base": str(value["world_base"]),
        "world_overlays": list(overlays),
        "policy_family": str(value["policy_family"]),
        "selection_mode": str(value["selection_mode"]),
    }


def _validate_compiled_runtime_identity(
    plan: Mapping[str, Any], runtime_identity: Mapping[str, Any]
) -> None:
    qualification = plan.get("qualification_key")
    invariants = (
        qualification.get("fixed_invariants")
        if isinstance(qualification, Mapping)
        else None
    )
    expected = {
        "runtime.world_base": runtime_identity["world_base"],
        "runtime.world_overlays": json.dumps(
            runtime_identity["world_overlays"],
            separators=(",", ":"),
        ),
        "runtime.policy_family": runtime_identity["policy_family"],
        "runtime.selection_mode": runtime_identity["selection_mode"],
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(name) != value for name, value in expected.items()
    ):
        raise CampaignProviderError(
            "cross-path registry runtime identity differs from compiled authority"
        )


def _global_genome_profile_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    path, _row = _verified_artifact(plan, "inheritance_registry")
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            "cross-path inheritance registry is not valid JSON"
        ) from exc
    required = {
        "schema_version",
        "native_module",
        "native_abi",
        "native_checkpoint_schema",
        "runtime_identity",
        "parameter_bindings",
        "fixed_parameters",
        "seed_owners",
        "semantic_types",
        "semantic_ids",
        "inheritance_semantic",
        "draw_budgets",
        "audited_loci",
        "linkage_blocks",
        "max_admitted_batch",
        "max_generations",
        "ticks_per_generation",
        "required_obligations",
    }
    optional = {"reproduction_exposure_capacity"}
    if (
        not isinstance(contract, Mapping)
        or not required <= set(contract)
        or set(contract) - required - optional
    ):
        raise CampaignProviderError("cross-path inheritance registry shape mismatch")
    if contract["schema_version"] != GLOBAL_GENOME_PROFILE_SCHEMA_VERSION:
        raise CampaignProviderError("cross-path inheritance registry version mismatch")
    bindings = contract["parameter_bindings"]
    if (
        not isinstance(bindings, Mapping)
        or not bindings
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in bindings.items()
        )
        or len(set(bindings.values())) != len(bindings)
    ):
        raise CampaignProviderError(
            "cross-path registry parameter bindings are invalid"
        )
    runtime_identity = _runtime_identity(contract["runtime_identity"])
    _validate_compiled_runtime_identity(plan, runtime_identity)
    semantic_types = _semantic_types(contract["semantic_types"])
    semantics = _semantic_id_maps(contract["semantic_ids"], semantic_types)
    if set(bindings) != set(semantics):
        raise CampaignProviderError(
            "cross-path registry semantic axes and parameter bindings differ"
        )
    inheritance_semantic = contract["inheritance_semantic"]
    if (
        not isinstance(inheritance_semantic, str)
        or inheritance_semantic not in semantics
    ):
        raise CampaignProviderError(
            "cross-path registry inheritance semantic is invalid"
        )
    fixed = contract["fixed_parameters"]
    owners = contract["seed_owners"]
    loci = contract["audited_loci"]
    blocks = contract["linkage_blocks"]
    obligations = contract["required_obligations"]
    if (
        not isinstance(fixed, Mapping)
        or not isinstance(owners, list)
        or not owners
        or any(not isinstance(value, str) or not value for value in owners)
        or len(set(owners)) != len(owners)
        or not isinstance(loci, list)
        or not loci
        or any(not isinstance(value, str) or not value for value in loci)
        or len(set(loci)) != len(loci)
        or not isinstance(blocks, Mapping)
        or not blocks
        or any(
            not isinstance(values, list) or any(locus not in loci for locus in values)
            for values in blocks.values()
        )
        or any(not isinstance(name, str) or not name for name in blocks)
        or (len([locus for values in blocks.values() for locus in values]) != len(loci))
        or set(locus for values in blocks.values() for locus in values) != set(loci)
        or not isinstance(obligations, list)
        or not obligations
        or any(not isinstance(value, str) or not value for value in obligations)
        or len(set(obligations)) != len(obligations)
    ):
        raise CampaignProviderError("cross-path registry semantic contract is invalid")
    budgets = contract["draw_budgets"]
    if (
        not isinstance(budgets, Mapping)
        or set(budgets) != set(semantics[inheritance_semantic])
        or any(type(value) is not int or value < 0 for value in budgets.values())
    ):
        raise CampaignProviderError(
            "cross-path registry draw budgets mismatch inheritance values"
        )
    for field in ("max_admitted_batch", "max_generations", "ticks_per_generation"):
        if type(contract[field]) is not int or contract[field] < 1:
            raise CampaignProviderError(f"cross-path registry {field} must be positive")
    exposure_capacity = contract.get("reproduction_exposure_capacity")
    if exposure_capacity is not None and (
        not isinstance(exposure_capacity, Mapping)
        or set(exposure_capacity)
        != {
            "schema_version",
            "storage_mode",
            "attempt_capacity",
            "snapshot_capacity",
            "aggregate_capacity",
            "agent_id_allocation",
        }
        or exposure_capacity.get("schema_version")
        != REPRODUCTION_EXPOSURE_CAPACITY_SCHEMA_VERSION
        or any(
            type(exposure_capacity.get(field)) is not int
            or exposure_capacity[field] < 1
            for field in ("attempt_capacity", "snapshot_capacity")
        )
        or exposure_capacity.get("storage_mode")
        not in {"exact_full", "compact_complete"}
        or type(exposure_capacity.get("aggregate_capacity")) is not int
        or exposure_capacity["aggregate_capacity"] < 1
    ):
        raise CampaignProviderError(
            "cross-path reproduction exposure capacity is invalid"
        )
    if exposure_capacity is not None:
        allocation = exposure_capacity["agent_id_allocation"]
        allocation_fields = {
            "schema_version",
            "current_capacity",
            "population_next_id",
            "max_births_per_tick",
            "slot_capacity",
            "allocation_generation_count",
            "retained_capacity",
            "required_aggregate_capacity",
        }
        if (
            not isinstance(allocation, Mapping)
            or set(allocation) != allocation_fields
            or allocation.get("schema_version") != AGENT_ID_ALLOCATION_SCHEMA_VERSION
            or any(
                type(allocation.get(field)) is not int or allocation[field] < 1
                for field in allocation_fields - {"schema_version"}
            )
        ):
            raise CampaignProviderError(
                "cross-path reproduction exposure agent-ID allocation is invalid"
            )
        try:
            required_aggregate_capacity = (
                required_reproduction_exposure_aggregate_capacity_v1(
                    allocation["current_capacity"],
                    allocation["population_next_id"],
                    allocation["max_births_per_tick"],
                    allocation["slot_capacity"],
                    allocation_generation_count=allocation[
                        "allocation_generation_count"
                    ],
                    retained_capacity=allocation["retained_capacity"],
                )
            )
        except InheritanceCrossPathExposureProjectionContractError as exc:
            raise CampaignProviderError(
                "cross-path reproduction exposure agent-ID allocation is invalid"
            ) from exc
        if (
            allocation["required_aggregate_capacity"] != required_aggregate_capacity
            or exposure_capacity["aggregate_capacity"] < required_aggregate_capacity
            or allocation["allocation_generation_count"] < contract["max_generations"]
        ):
            raise CampaignProviderError(
                "cross-path reproduction exposure agent-ID allocation is stale "
                "or insufficient"
            )
    if (
        not isinstance(contract["native_module"], str)
        or not contract["native_module"]
        or not isinstance(contract["native_abi"], str)
        or not contract["native_abi"]
        or not isinstance(contract["native_checkpoint_schema"], str)
        or not contract["native_checkpoint_schema"]
    ):
        raise CampaignProviderError("cross-path registry native binding is invalid")
    return dict(contract)


def _validate_genome_v2_package_v1(
    plan: Mapping[str, Any],
    *,
    required_obligations: Sequence[str],
) -> dict[str, Any]:
    executable_path, executable_row = _verified_artifact(
        plan, "genome_upgrade_v2_executable_contract"
    )
    capability_contract_path, _capability_contract_row = _verified_artifact(
        plan, "genome_upgrade_v2_capability_contract"
    )
    package_contract_path, package_contract_row = _verified_artifact(
        plan, "genome_upgrade_v2_package_contract"
    )
    obligation_dictionary_path, obligation_dictionary_row = _verified_artifact(
        plan, "genome_upgrade_v2_obligation_dictionary"
    )
    invalidation_rules_path, invalidation_rules_row = _verified_artifact(
        plan, "genome_upgrade_v2_invalidation_rules"
    )
    chronicle_schema_path, _chronicle_schema_row = _verified_artifact(
        plan, "genome_upgrade_v2_chronicle_schema"
    )
    reporting_schema_path, _reporting_schema_row = _verified_artifact(
        plan, "genome_upgrade_v2_reporting_schema"
    )
    storage_budget_path, _storage_budget_row = _verified_artifact(
        plan, "genome_upgrade_v2_observability_storage_budget"
    )
    receipt_path, _receipt_row = _verified_artifact(
        plan, "genome_upgrade_v2_package_receipt"
    )
    try:
        executable = json.loads(executable_path.read_text(encoding="utf-8"))
        capability_contract = json.loads(
            capability_contract_path.read_text(encoding="utf-8")
        )
        package_contract = json.loads(package_contract_path.read_text(encoding="utf-8"))
        obligation_dictionary = json.loads(
            obligation_dictionary_path.read_text(encoding="utf-8")
        )
        invalidation_rules = json.loads(
            invalidation_rules_path.read_text(encoding="utf-8")
        )
        chronicle_schema = json.loads(chronicle_schema_path.read_text(encoding="utf-8"))
        reporting_schema = json.loads(reporting_schema_path.read_text(encoding="utf-8"))
        storage_budget = json.loads(storage_budget_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            "Genome V2 executable package artifacts are unreadable"
        ) from exc
    from engine.evolution.genome_upgrade_v2 import CONTRACT_SHA256
    from engine.evolution.genome_upgrade_v2_package_v1 import (
        PRE_ROUTE_PACKAGE_OBLIGATIONS,
        capacity_layout_contract_v1,
        checkpoint_authority_contract_v1,
        executable_package_contract_v1,
        invalidation_rules_v1,
        obligation_dictionary_v1,
        package_core_identity_sha256_v1,
        package_identity_sha256_v1,
        route_input_contract_v1,
        supported_checkpoint_cut_ticks_v1,
        supported_world_controller_absolute_tick_envelope_v1,
    )
    from engine.evolution.genome_upgrade_v2_cancellation_v1 import (
        cancellation_contract_v1,
    )
    from engine.evolution.genome_upgrade_v2_capability_contract_v1 import (
        GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
        GenomeV2P10CapabilityContract,
    )
    from engine.evolution.genome_upgrade_v2_causal_gate_v1 import (
        PREREGISTRATION_SHA256 as CAUSAL_GATE_PREREGISTRATION_SHA256,
        SCHEMA_VERSION as CAUSAL_GATE_SCHEMA_VERSION,
    )
    from engine.evolution.genome_upgrade_v2_causal_settlement_v1 import (
        CONTRACT_SHA256 as CAUSAL_SETTLEMENT_CONTRACT_SHA256,
        SCHEMA_VERSION as CAUSAL_SETTLEMENT_SCHEMA_VERSION,
    )
    from engine.evolution.genome_upgrade_v2_e1_gate_v1 import (
        PREREGISTRATION_SHA256 as ECOLOGICAL_E1_PREREGISTRATION_SHA256,
        SCHEMA_VERSION as ECOLOGICAL_E1_SCHEMA_VERSION,
    )
    from engine.evolution.genome_upgrade_v2_environment_switch_gate_v1 import (
        PREREGISTRATION_SHA256 as ENVIRONMENT_SWITCH_PREREGISTRATION_SHA256,
        SCHEMA_VERSION as ENVIRONMENT_SWITCH_SCHEMA_VERSION,
    )
    from engine.evolution.genome_upgrade_v2_hereditary_capacity_gate_v1 import (
        PREREGISTRATION_SHA256 as HEREDITARY_CAPACITY_PREREGISTRATION_SHA256,
        SCHEMA_VERSION as HEREDITARY_CAPACITY_SCHEMA_VERSION,
    )
    from engine.evolution.global_genome_registry_v1 import (
        global_genome_profile_v1,
    )
    from engine.authoritative_state.schema import load_schema_v1
    from engine.evolution.genome_upgrade_v2_telemetry import (
        chronicle_schema_v1,
        observability_storage_budget_v1,
        reporting_schema_v1,
    )

    expected = set(required_obligations)
    canonical_expected = set(PRE_ROUTE_PACKAGE_OBLIGATIONS)
    obligations = receipt.get("obligations") if isinstance(receipt, Mapping) else None
    identity = receipt.get("identity") if isinstance(receipt, Mapping) else None
    expected_package_contract = executable_package_contract_v1()
    expected_dictionary = obligation_dictionary_v1()
    expected_invalidation = invalidation_rules_v1()
    expected_chronicle_schema = chronicle_schema_v1()
    expected_reporting_schema = reporting_schema_v1()
    expected_storage_budget = observability_storage_budget_v1()
    expected_supported_absolute_tick_envelope = (
        supported_world_controller_absolute_tick_envelope_v1()
    )
    expected_checkpoint_authority_contract = checkpoint_authority_contract_v1()
    expected_supported_checkpoint_cut_ticks = supported_checkpoint_cut_ticks_v1()
    expected_supported_campaign_classes = [
        "qualification",
        "fixed_genotype_causal",
        "bounded_ecological",
    ]
    genome_profile = global_genome_profile_v1("genome_upgrade_v2")
    schema = load_schema_v1()
    component_hashes = schema.root["component_hashes"]
    cuda_artifact_path, cuda_artifact = _verified_artifact(plan, "cuda_extension")
    cuda_manifest_path = cuda_artifact_path.with_suffix(
        cuda_artifact_path.suffix + ".manifest.json"
    )
    try:
        cuda_manifest = json.loads(cuda_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignProviderError(
            "Genome V2 CUDA artifact manifest is unreadable"
        ) from exc
    required_identity_fields = set(
        expected_package_contract["required_identity_fields"]
    )
    identity_keys_are_valid = isinstance(identity, Mapping) and (
        set(identity) == required_identity_fields
    )
    if (
        expected != canonical_expected
        or not isinstance(executable, Mapping)
        or executable.get("schema_version")
        != "genome_upgrade_v2_executable_contract_v1"
        or executable.get("provider_id") != PROVIDER_ID
        or executable.get("runtime_profile_id") != PROFILE_ID
        or executable.get("global_genome_profile_id") != "genome_upgrade_v2"
        or executable.get("mechanic_contract_sha256") != CONTRACT_SHA256
        or executable.get("capability_contract_sha256")
        != GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
        or capability_contract != GenomeV2P10CapabilityContract.v1().to_dict()
        or set(executable.get("required_obligations") or ()) != expected
        or package_contract != expected_package_contract
        or obligation_dictionary != expected_dictionary
        or invalidation_rules != expected_invalidation
        or chronicle_schema != expected_chronicle_schema
        or reporting_schema != expected_reporting_schema
        or storage_budget != expected_storage_budget
        or not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != GENOME_V2_PACKAGE_RECEIPT_SCHEMA_VERSION
        or receipt.get("ok") is not True
        or receipt.get("authority_state") != "package_candidate_complete"
        or "legacy_compatibility" in receipt
        or receipt.get("package_contract_sha256") != package_contract_row["sha256"]
        or receipt.get("obligation_dictionary_sha256")
        != obligation_dictionary_row["sha256"]
        or receipt.get("invalidation_rules_sha256") != invalidation_rules_row["sha256"]
        or receipt.get("executable_contract_sha256") != executable_row["sha256"]
        or not isinstance(identity, Mapping)
        or not identity_keys_are_valid
        or not isinstance(obligations, Mapping)
        or set(obligations) != expected
        or receipt.get("package_id_sha256") != package_identity_sha256_v1(receipt)
    ):
        raise CampaignProviderError(
            "Genome V2 executable package receipt binding is invalid"
        )
    hash_fields = {
        name for name in required_identity_fields if name.endswith("_sha256")
    }
    exact_identity = {
        "source_clean": True,
        "diagnostic_only": False,
        "provider_id": PROVIDER_ID,
        "runtime_profile_id": PROFILE_ID,
        "authoritative_state_schema_version": schema.root["schema_version"],
        "authoritative_state_schema_sha256": schema.schema_hash,
        "global_genome_profile_id": "genome_upgrade_v2",
        "global_genome_profile_sha256": canonical_sha256(genome_profile.payload()),
        "mechanic_contract_sha256": CONTRACT_SHA256,
        "capability_contract_sha256": (GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256),
        "causal_settlement_schema_version": (CAUSAL_SETTLEMENT_SCHEMA_VERSION),
        "causal_settlement_contract_sha256": (CAUSAL_SETTLEMENT_CONTRACT_SHA256),
        "causal_gate_schema_version": CAUSAL_GATE_SCHEMA_VERSION,
        "causal_gate_preregistration_sha256": (CAUSAL_GATE_PREREGISTRATION_SHA256),
        "ecological_e1_schema_version": ECOLOGICAL_E1_SCHEMA_VERSION,
        "ecological_e1_preregistration_sha256": (ECOLOGICAL_E1_PREREGISTRATION_SHA256),
        "hereditary_capacity_schema_version": (HEREDITARY_CAPACITY_SCHEMA_VERSION),
        "hereditary_capacity_preregistration_sha256": (
            HEREDITARY_CAPACITY_PREREGISTRATION_SHA256
        ),
        "environment_switch_schema_version": (ENVIRONMENT_SWITCH_SCHEMA_VERSION),
        "environment_switch_preregistration_sha256": (
            ENVIRONMENT_SWITCH_PREREGISTRATION_SHA256
        ),
        "checkpoint_schema_version": (
            schema.components["checkpoint_schema.json"]["schema_version"]
        ),
        "checkpoint_schema_sha256": component_hashes["checkpoint_schema.json"],
        "chronicle_schema_version": (
            schema.components["chronicle_schema.json"]["schema_version"]
        ),
        "chronicle_schema_sha256": component_hashes["chronicle_schema.json"],
        "genome_v2_chronicle_schema_version": (
            expected_chronicle_schema["schema_version"]
        ),
        "genome_v2_chronicle_schema_sha256": canonical_sha256(
            expected_chronicle_schema
        ),
        "reporting_schema_version": expected_reporting_schema["schema_version"],
        "reporting_schema_sha256": canonical_sha256(expected_reporting_schema),
        "observability_storage_budget_schema_version": (
            expected_storage_budget["schema_version"]
        ),
        "observability_storage_budget_sha256": canonical_sha256(
            expected_storage_budget
        ),
        "executable_contract_sha256": executable_row["sha256"],
        "obligation_set_sha256": expected_dictionary["obligation_set_sha256"],
        "cuda_artifact_sha256": (
            cuda_artifact.get("sha256") if isinstance(cuda_artifact, Mapping) else None
        ),
        "loaded_artifact_sha256": _file_sha256(cuda_artifact_path),
        "cuda_manifest_sha256": _file_sha256(cuda_manifest_path),
        "cancellation_abi_schema": cancellation_contract_v1()["schema_version"],
        "cancellation_abi_sha256": canonical_sha256(cancellation_contract_v1()),
        "capacity_layout_contract_sha256": canonical_sha256(
            capacity_layout_contract_v1()
        ),
        "checkpoint_authority_contract_sha256": canonical_sha256(
            expected_checkpoint_authority_contract
        ),
        "route_contract_inputs_sha256": canonical_sha256(route_input_contract_v1()),
        "target_architecture": "sm_75",
        "no_jit": True,
        "no_fallback": True,
        "supported_world_controller_absolute_tick_envelope": (
            expected_supported_absolute_tick_envelope
        ),
        "supported_checkpoint_cut_ticks": expected_supported_checkpoint_cut_ticks,
        "supported_campaign_classes": expected_supported_campaign_classes,
    }
    if (
        any(identity.get(name) != value for name, value in exact_identity.items())
        or any(not _valid_sha256(identity.get(name)) for name in hash_fields)
        or not isinstance(identity.get("source_commit"), str)
        or len(identity["source_commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in identity["source_commit"]
        )
        or identity.get("evidence_inventory_sha256") != canonical_sha256(obligations)
        or any(
            not isinstance(identity.get(name), str) or not identity[name]
            for name in (
                "ordinary_birth_abi_schema",
                "turnover_abi_schema",
                "elf_build_id",
            )
        )
        or not isinstance(identity.get("compiler_cuda_torch_environment"), Mapping)
        or identity.get("compiler_cuda_torch_environment_sha256")
        != canonical_sha256(identity["compiler_cuda_torch_environment"])
        or identity.get("critical_build_flags_sha256")
        != canonical_sha256(
            {
                "cxx_flags": cuda_manifest.get("cxx_flags"),
                "nvcc_flags": cuda_manifest.get("nvcc_flags"),
                "link_flags": cuda_manifest.get("link_flags"),
            }
        )
        or cuda_manifest.get("artifact_sha256") != identity.get("cuda_artifact_sha256")
        or cuda_manifest.get("elf_build_id") != identity.get("elf_build_id")
        or cuda_manifest.get("cuda_arch") != identity.get("target_architecture")
        or cuda_manifest.get("runtime_jit_compilation") is not False
        or cuda_manifest.get("sass_only") is not True
        or identity.get("executable_package_core_sha256")
        != package_core_identity_sha256_v1(identity)
        or identity.get("supported_world_controller_absolute_tick_envelope")
        != expected_supported_absolute_tick_envelope
        or identity.get("supported_checkpoint_cut_ticks")
        != expected_supported_checkpoint_cut_ticks
        or identity.get("supported_campaign_classes")
        != expected_supported_campaign_classes
    ):
        raise CampaignProviderError(
            "Genome V2 executable package identity is invalid or stale"
        )
    for obligation, evidence in obligations.items():
        evidence_path = (
            Path(str(evidence.get("evidence_path") or "")).expanduser()
            if isinstance(evidence, Mapping)
            else Path()
        )
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("status") != "passed"
            or not _valid_sha256(evidence.get("evidence_sha256"))
            or not evidence_path.is_absolute()
            or evidence_path.is_symlink()
            or _verified_hashed_path_v1(
                evidence_path, str(evidence.get("evidence_sha256"))
            )
            is None
        ):
            raise CampaignProviderError(
                "Genome V2 package obligation is not evidence-bound: " f"{obligation}"
            )
    return dict(receipt)


# Historical import compatibility. Production code uses the engine-global name.
_registry_contract = _global_genome_profile_contract


def _cell_lookup(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = plan.get("cells")
    if not isinstance(raw, list):
        raise CampaignProviderError("cross-path plan lacks cells")
    cells = {
        str(row["cell_id"]): dict(row)
        for row in raw
        if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
    }
    if len(cells) != len(raw):
        raise CampaignProviderError("cross-path cells are malformed")
    return cells


def _validate_native(native: ModuleType, contract: Mapping[str, Any]) -> None:
    if (
        dict(native.SEMANTIC_TYPES) != dict(contract["semantic_types"])
        or {
            str(semantic): {
                str(name): int(value) for name, value in dict(values).items()
            }
            for semantic, values in dict(native.SEMANTIC_IDS).items()
        }
        != dict(contract["semantic_ids"])
        or str(native.INHERITANCE_SEMANTIC) != str(contract["inheritance_semantic"])
        or dict(native.DRAW_BUDGETS) != dict(contract["draw_budgets"])
        or tuple(native.AUDITED_LOCI) != tuple(contract["audited_loci"])
        or {
            str(name): list(values)
            for name, values in dict(native.LINKAGE_BLOCKS).items()
        }
        != dict(contract["linkage_blocks"])
    ):
        raise CampaignProviderError("cross-path native/registry contract drift")


def _validate_cell(
    cell: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    parameters = cell.get("parameters")
    seeds = cell.get("seeds")
    if not isinstance(parameters, Mapping) or not isinstance(seeds, Mapping):
        raise CampaignProviderError("cross-path cell parameters or seeds are malformed")
    for name, expected in dict(contract["fixed_parameters"]).items():
        if parameters.get(name) != expected:
            raise CampaignProviderError(f"cross-path fixed parameter differs: {name}")
    bindings = dict(contract["parameter_bindings"])
    allowed = dict(contract["semantic_ids"])
    semantic_types = dict(contract["semantic_types"])
    for semantic, parameter_name in bindings.items():
        token = _semantic_token(
            parameters.get(parameter_name),
            semantic_types[semantic],
            field=f"{semantic}.{parameter_name}",
        )
        if token not in allowed[semantic]:
            raise CampaignProviderError(f"cross-path cell has unsupported {semantic}")
    if set(seeds) != set(contract["seed_owners"]) or any(
        type(seed) is not int or seed < 0 for seed in seeds.values()
    ):
        raise CampaignProviderError("cross-path seed ownership mismatch")


def _is_qualification(plan: Mapping[str, Any]) -> bool:
    value = plan.get("route_qualification_case_id")
    return isinstance(value, str) and bool(value.strip())


def _minimum_segment_attempt_capacity_v1(
    *,
    segment_generations: int,
    ticks_per_generation: int,
    resident_capacity: int,
) -> int:
    """Conservative production-loop bound: decisions plus refill per segment."""

    if any(
        type(value) is not int or value < 1
        for value in (
            segment_generations,
            ticks_per_generation,
            resident_capacity,
        )
    ):
        raise CampaignProviderError(
            "reproduction exposure bound inputs must be positive integers"
        )
    return segment_generations * resident_capacity * (ticks_per_generation + 1)


class EielProductionExecutionProviderV1(ProductionEngineLifecycleV1):
    """The sole production provider, currently admitting cross-path config.

    The engine lifecycle owns physical segmentation, Bundle v2 publication,
    disk reload, and fresh-owner continuation. The first admitted engine
    configuration binds the existing global inheritance registry. Other
    world/controller configurations remain fail-closed until their mechanics
    are migrated into engine components.
    """

    provider_id = PROVIDER_ID
    profile_id = PROFILE_ID
    reporting_parity_supported = True
    # The qualified Architecture B runner admits segment generation counts
    # in [3, 128].  Route canaries use this value to choose continuation
    # splits, so it must describe the runner's real lower bound.
    resident_segment_generation_min = 3
    resident_segment_generation_max = 128
    # The loaded CUDA extension retains a bounded process-lifetime allocation
    # after an audit-only owner is released. This tolerance applies only to
    # live allocated bytes; allocator reservation is diagnostic.
    prepared_owner_teardown_tolerance_bytes = 64 * 1024 * 1024
    checkpoint_schema_version = CHECKPOINT_SCHEMA_VERSION
    artifact_abi = "inheritance_cross_path_registry_bound_v1"
    route_lifecycle_contract = "inheritance_cross_path_live_birth_lifecycle_v1"
    route_rng_contract = "inheritance_cross_path_seed_owner_rng_v1"
    route_settlement_contract = "inheritance_cross_path_settlement_v1"
    required_artifact_exports = ()
    authoritative_state_schema_required_v1 = True

    @staticmethod
    def campaign_metric_rows(
        rows: Sequence[Mapping[str, Any]], *, cell_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Project configured mechanics rows without provider-specific dispatch."""

        if rows and all(
            row.get("schema_version")
            in {
                "eiel_eed_native_cell_metrics_v2",
                "eiel_genome_v2_e1_native_cell_metrics_v1",
            }
            and isinstance(row.get("metrics"), Mapping)
            for row in rows
        ):
            eed_rows: list[dict[str, Any]] = []
            for raw in rows:
                row = dict(raw)
                metrics = dict(row["metrics"])
                if metrics.get("population.deaths") == 0:
                    metrics.pop("population.deaths")
                eed_rows.append({"cell_id": row.get("cell_id"), "metrics": metrics})
            return ProductionEngineLifecycleV1.campaign_metric_rows(
                eed_rows,
                cell_ids=cell_ids,
            )

        generic_rows: list[dict[str, Any]] = []
        for raw in rows:
            row = dict(raw)
            population = row.get("population")
            reproduction = row.get("reproduction")
            if not isinstance(population, Mapping) or not isinstance(
                reproduction, Mapping
            ):
                return ProductionEngineLifecycleV1.campaign_metric_rows(
                    rows,
                    cell_ids=cell_ids,
                )
            row["metrics"] = {
                "population.alive": population.get("alive"),
                "population.births": reproduction.get("successful_births"),
                # This generation-series contract does not publish a death
                # count. Null is the generic contract's explicit unavailable
                # value and is never interpreted as zero.
                "population.deaths": None,
            }
            generic_rows.append(row)
        return ProductionEngineLifecycleV1.campaign_metric_rows(
            generic_rows,
            cell_ids=cell_ids,
        )

    @staticmethod
    def project_generation_series(
        *,
        plan: Mapping[str, Any],
        cells: Sequence[Mapping[str, Any]],
        generic_patch_rows: Sequence[Mapping[str, Any]],
        native_exposure_segment_artifacts: Sequence[Mapping[str, Any]],
        checkpoint_catalog: Mapping[str, Mapping[str, Any]],
        terminal_generation: int,
    ) -> list[dict[str, Any]]:
        """Project the exact G0..G inheritance series from resident evidence."""

        del plan
        from .inheritance_generation_series_v1 import (
            project_inheritance_generation_series_v1,
        )

        return project_inheritance_generation_series_v1(
            cells=cells,
            generic_patch_rows=generic_patch_rows,
            native_exposure_segment_artifacts=(native_exposure_segment_artifacts),
            checkpoint_catalog=checkpoint_catalog,
            terminal_generation=terminal_generation,
        )

    def load_prior_generation_series_artifacts(
        self,
        *,
        checkpoint_catalog: Mapping[str, Mapping[str, Any]],
        cell_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Reload exact native segment artifacts from durable milestone bundles."""

        from .inheritance_cross_path_patch_execution_v1 import (
            EXPOSURE_SEGMENT_FIELD,
        )

        ordered_cells = tuple(cell_ids)
        artifacts: list[dict[str, Any]] = []
        try:
            generations = sorted(int(value) for value in checkpoint_catalog)
        except (TypeError, ValueError) as exc:
            raise CampaignProviderError(
                "generation-series checkpoint catalog is malformed"
            ) from exc
        if any(str(generation) not in checkpoint_catalog for generation in generations):
            raise CampaignProviderError(
                "generation-series checkpoint catalog keys are noncanonical"
            )
        for generation in generations:
            if generation == 0:
                continue
            checkpoint = checkpoint_catalog[str(generation)]
            bundles = checkpoint.get("cell_bundles")
            if (
                not isinstance(bundles, list)
                or tuple(
                    row.get("cell_id") for row in bundles if isinstance(row, Mapping)
                )
                != ordered_cells
                or len(bundles) != len(ordered_cells)
            ):
                raise CampaignProviderError(
                    "generation-series checkpoint bundle order differs"
                )
            for cell_id, bundle in zip(ordered_cells, bundles, strict=True):
                projection = self._read_projection_bundle(str(bundle["path"]))
                artifact = projection.get(EXPOSURE_SEGMENT_FIELD)
                if (
                    not isinstance(artifact, Mapping)
                    or artifact.get("cell_id") != cell_id
                ):
                    raise CampaignProviderError(
                        "generation-series native segment artifact is absent"
                    )
                artifacts.append(dict(artifact))
        return artifacts

    def __init__(
        self,
        *,
        native_api: ModuleType | None = None,
        torch_module: ModuleType | None = None,
        device: str = "cuda",
        require_prebuilt: bool = True,
        artifact_loader: Callable[..., Any] | None = None,
        runner_builder: Callable[..., Any] | None = None,
        request_builder: Callable[..., Any] | None = None,
        preparer: Callable[..., Any] | None = None,
        executor: Callable[..., Mapping[str, Any]] | None = None,
        projector: Callable[..., Any] | None = None,
        checkpoint_writer: Callable[..., str] | None = None,
        checkpoint_loader: Callable[..., Mapping[str, Any]] | None = None,
        checkpoint_decoder: (
            Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
        ) = None,
        checkpoint_digest: Callable[[Mapping[str, Any]], str] | None = None,
        source_preparer: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
        bind_committed_defaults: bool = False,
        generation_series_required: bool | None = None,
        profile_id: str = PROFILE_ID,
    ) -> None:
        super().__init__(
            artifact_loader=artifact_loader,
            runner_builder=runner_builder,
            request_builder=request_builder,
            preparer=preparer,
            executor=executor,
            projector=projector,
            checkpoint_writer=checkpoint_writer,
            checkpoint_loader=checkpoint_loader,
            checkpoint_decoder=checkpoint_decoder,
            checkpoint_digest=checkpoint_digest,
            source_preparer=source_preparer,
        )
        self._native_override = native_api
        self._torch_override = torch_module
        self.profile_id = str(profile_id)
        self.device = str(device)
        self.require_prebuilt = bool(require_prebuilt)
        if self.profile_id == CPU_PROFILE_ID:
            self.authoritative_state_schema_required_v1 = False
        self._bind_committed_defaults = bool(bind_committed_defaults)
        self._generation_series_required_override = generation_series_required
        self.generation_series_required = bool(generation_series_required)
        self._committed_adapter_registry_sha256: str | None = None
        self._last_mechanic_receipt_sha256: str | None = None
        self._cross_path_artifact_chain_loader_v1: Any | None = None

    def _generation_series_required(self, plan: Mapping[str, Any]) -> bool:
        """Bind series ownership to inheritance mechanics, not provider identity."""

        if self._generation_series_required_override is not None:
            return bool(self._generation_series_required_override)
        configuration = production_engine_configuration_v1(plan)
        return (
            configuration.world_base == "inheritance_cross_path_neutral"
            and configuration.controller_family == "audit_only"
        )

    @property
    def execution_owner_qualified(self) -> bool:
        """Expose only the most recently verified plan-scoped receipt state.

        Execution never trusts this informational latch; execute/resume validate
        the current plan again so a receipt for one plan cannot authorize
        another plan.
        """

        return self._last_mechanic_receipt_sha256 is not None

    @property
    def torch(self) -> ModuleType:
        return self._torch_override or _torch()

    def _native(self, contract: Mapping[str, Any]) -> ModuleType:
        if self._native_override is not None:
            return self._native_override
        return importlib.import_module(str(contract["native_module"]))

    def _ensure_committed_default_adapters(
        self, plan: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Reject activation of the retired standalone cross-path adapters."""

        raise CampaignProviderError(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "standalone cross-path adapter activation is disabled"
        )

    def _adapter_identities(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for public_name, private_name in (
            ("artifact_loader", "_artifact_loader"),
            ("runner_builder", "_runner_builder"),
            ("request_builder", "_request_builder"),
            ("preparer", "_preparer"),
            ("executor", "_executor"),
            ("projector", "_projector"),
            ("source_preparer", "_source_preparer"),
        ):
            value = getattr(self, private_name)
            if value is None:
                result[public_name] = "unbound"
                continue
            module = getattr(value, "__module__", type(value).__module__)
            qualname = getattr(value, "__qualname__", type(value).__qualname__)
            result[public_name] = f"{module}.{qualname}"
        return result

    @staticmethod
    def _component_binding(plan: Mapping[str, Any]) -> Any:
        from .production_engine_components import (
            production_engine_component_binding_v1,
        )

        return production_engine_component_binding_v1(
            production_engine_configuration_v1(plan)
        )

    @staticmethod
    def _retired_cross_path_configuration(plan: Mapping[str, Any]) -> bool:
        configuration = production_engine_configuration_v1(plan)
        return (
            configuration.world_base == "inheritance_cross_path_neutral"
            and configuration.controller_family == "audit_only"
        )

    @staticmethod
    def _direct_resident_binding_unmigrated_message(plan: Mapping[str, Any]) -> str:
        configuration = production_engine_configuration_v1(plan)
        binding = configuration.component_binding()
        genome = binding["genome_inheritance"]["configuration_id"]
        mutation_regimes: set[tuple[int, int]] = set()
        founder_profiles: set[str] = set()
        cells = plan.get("cells")
        if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)):
            for cell in cells:
                parameters = (
                    cell.get("parameters") if isinstance(cell, Mapping) else None
                )
                if not isinstance(parameters, Mapping):
                    continue
                rate = parameters.get("genome_upgrade_v2.mutation_rate_q10")
                step = parameters.get("genome_upgrade_v2.mutation_step_q10")
                if type(rate) is int and type(step) is int and rate >= 0 and step >= 0:
                    mutation_regimes.add((rate, step))
                founder = parameters.get("genome_upgrade_v2.founder_profile")
                if isinstance(founder, str) and founder:
                    founder_profiles.add(founder)
        mutation_fragment = ""
        if mutation_regimes:
            mutation_fragment = (
                " genome_v2_mutation_regimes_q10="
                f"{json.dumps([list(row) for row in sorted(mutation_regimes)])}"
            )
        recombination_profiles = sorted(
            profile
            for profile in founder_profiles
            if profile.startswith("recomb72_gv2_v1.")
        )
        recombination_fragment = ""
        if recombination_profiles:
            recombination_fragment = (
                " recombination_profiles=" f"{json.dumps(recombination_profiles)}"
            )
        return (
            "EIEL_DIRECT_RESIDENT_BINDING_UNMIGRATED "
            f"world_base={configuration.world_base!r} "
            f"world_overlays={json.dumps(list(configuration.world_overlays))} "
            f"controller={configuration.controller_family!r} "
            f"genome={genome!r}"
            f"{mutation_fragment}"
            f"{recombination_fragment}: production engine has no admitted "
            "direct-resident execution binding for this combined mechanics "
            "envelope"
        )

    @staticmethod
    def _wdl_jungle_direct_resident_execution_unqualified_message() -> str:
        return (
            "WDL_JUNGLE_DIRECT_RESIDENT_EXECUTION_UNQUALIFIED: combined "
            "Patch+WDL+Jungle fixed-MLP runner configuration is named, but "
            "the CPU-native production profile still lacks exact "
            "direct-resident execution, checkpoint projection, RNG/event "
            "lineage, public-output parity, Genome V2 mutation, and live "
            "recombination semantics for this combined route"
        )

    @staticmethod
    def _hapb_ordinary_birth_route(plan: Mapping[str, Any]) -> bool:
        """Select HAPB only for an explicit all-cell ordinary-birth envelope."""

        artifacts = plan.get("artifacts")
        cuda_extension = (
            artifacts.get("cuda_extension") if isinstance(artifacts, Mapping) else None
        )
        if not isinstance(cuda_extension, Mapping) or cuda_extension.get("abi") != (
            "architecture_b_wdl_jungle_eed_hapb_fixed_mlp_native_v1"
        ):
            return False
        cells = plan.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            return False
        if not cells:
            return False
        return all(
            isinstance(cell, Mapping)
            and isinstance(cell.get("parameters"), Mapping)
            and cell["parameters"].get("production_lifecycle.ordinary_births_enabled")
            is True
            for cell in cells
        )

    @staticmethod
    def _direct_resident_binding(plan: Mapping[str, Any]) -> Any | None:
        configuration = production_engine_configuration_v1(plan)
        if configuration.world_base == "patch":
            from .production_engine_native_bindings import (
                eed_fixed_mlp_native_binding_v1,
                gmnw_policy_arms_native_binding_v1,
                jungle_fixed_mlp_native_binding_v1,
                patch_fixed_mlp_native_binding_v1,
                patch_stable_gru_native_binding_v1,
                wdl_jungle_eed_fixed_mlp_native_binding_v1,
                wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1,
                wdl_jungle_fixed_mlp_native_binding_v1,
                wdl_fixed_mlp_native_binding_v1,
                wdl_stable_gru_native_binding_v1,
            )

            if (
                configuration.world_overlays == ("eed_resource_impact",)
                and configuration.controller_family == "fixed_mlp"
                and any(
                    name.startswith("genome_upgrade_v2.")
                    for name in configuration.parameter_names
                )
            ):
                return eed_fixed_mlp_native_binding_v1()
            if (
                not configuration.world_overlays
                and configuration.controller_family == "fixed_mlp"
            ):
                return patch_fixed_mlp_native_binding_v1()
            if (
                configuration.world_overlays == ("wdl_delayed_hazard",)
                and configuration.controller_family == "fixed_mlp"
            ):
                return wdl_fixed_mlp_native_binding_v1()
            if (
                configuration.world_overlays == ("jungle",)
                and configuration.controller_family == "fixed_mlp"
            ):
                return jungle_fixed_mlp_native_binding_v1()
            if (
                configuration.world_overlays == ("jungle", "wdl_delayed_hazard")
                and configuration.controller_family == "fixed_mlp"
                and any(
                    name.startswith("genome_upgrade_v2.")
                    for name in configuration.parameter_names
                )
            ):
                return wdl_jungle_fixed_mlp_native_binding_v1()
            if (
                configuration.world_overlays
                == ("eed_resource_impact", "jungle", "wdl_delayed_hazard")
                and configuration.controller_family == "fixed_mlp"
                and any(
                    name.startswith("genome_upgrade_v2.")
                    for name in configuration.parameter_names
                )
            ):
                if EielProductionExecutionProviderV1._hapb_ordinary_birth_route(plan):
                    # Production admission remains retired. An explicit HAPB ABI
                    # campaign still uses the HAPB binding so Launchpad QualPrep
                    # does not silently retarget the generic EED artifact pin.
                    return wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1()
                return wdl_jungle_eed_fixed_mlp_native_binding_v1()
            if (
                not configuration.world_overlays
                and configuration.controller_family == "stable_gru"
            ):
                return patch_stable_gru_native_binding_v1()
            if (
                configuration.world_overlays == ("wdl_delayed_hazard",)
                and configuration.controller_family == "stable_gru"
            ):
                return wdl_stable_gru_native_binding_v1()
            if (
                configuration.world_overlays == ("gmnw_memory_necessity",)
                and configuration.controller_family == "gmnw_policy_arms"
            ):
                return gmnw_policy_arms_native_binding_v1()
        return None

    def _resolved_execution_adapters(
        self, plan: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        configuration = production_engine_configuration_v1(plan)
        direct_binding = self._direct_resident_binding(plan)
        if direct_binding is not None:
            if self.profile_id == CPU_PROFILE_ID:
                if (
                    getattr(direct_binding, "binding_id", None)
                    == "wdl_jungle_fixed_mlp_native_binding_v1"
                    or getattr(direct_binding, "binding_id", None)
                    == "wdl_jungle_eed_fixed_mlp_native_binding_v1"
                ):
                    from .production_engine_native_bindings import (
                        wdl_jungle_fixed_mlp_cpu_native_adapters_v1,
                    )

                    adapters = wdl_jungle_fixed_mlp_cpu_native_adapters_v1()
                    adapters["mechanics_binding"] = direct_binding
                    return adapters
                if (
                    getattr(direct_binding, "binding_id", None)
                    != "patch_fixed_mlp_native_binding_v1"
                ):
                    raise CampaignProviderError(
                        "CPU-native production profile currently admits only "
                        "Patch fixed-MLP mechanics"
                    )
                from .production_engine_native_bindings import (
                    patch_fixed_mlp_cpu_native_adapters_v1,
                )

                adapters = patch_fixed_mlp_cpu_native_adapters_v1()
                adapters["mechanics_binding"] = direct_binding
                return adapters
            adapters = direct_binding.adapters()
            adapters["mechanics_binding"] = direct_binding
            return adapters
        if (
            configuration.world_base == "inheritance_cross_path_neutral"
            and configuration.controller_family == "audit_only"
        ):
            self._ensure_committed_default_adapters(plan)
            return {
                "artifact_loader": self._artifact_loader,
                "runner_builder": self._runner_builder,
                "request_builder": self._request_builder,
                "preparer": self._preparer,
                "executor": self._executor,
                "projector": self._projector,
                "source_preparer": self._source_preparer,
            }
        raise CampaignProviderError(
            "production engine native mechanics binding is not migrated"
        )

    def prepared_owner_measurement_support_v1(
        self,
        plan: Mapping[str, Any],
    ) -> Mapping[str, str | bool]:
        """State whether the generic zero-dispatch audit is safe for ``plan``."""

        return {"supported": True, "code": "supported", "message": "supported"}

    def execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        payload = _payload(plan)
        execution = payload.get("execution")
        plan_acs_checkpoint_slots = (
            execution.get("acs_checkpoint_slots", 0)
            if isinstance(execution, Mapping)
            else 0
        )
        raw_plan_acs_capture_instrumentation = (
            execution.get("acs_capture_instrumentation", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_plan_acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "production route ACS capture instrumentation is invalid"
            )
        plan_acs_capture_instrumentation = raw_plan_acs_capture_instrumentation or bool(
            plan_acs_checkpoint_slots
        )
        plan_native_streamed_checkpoint = (
            execution.get("native_streamed_checkpoint", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(plan_native_streamed_checkpoint) is not bool:
            raise CampaignProviderError(
                "production route native streamed checkpoint selection is invalid"
            )
        plan_world_sharded_checkpoint = (
            execution.get("world_sharded_checkpoint", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(plan_world_sharded_checkpoint) is not bool:
            raise CampaignProviderError(
                "production route world-sharded checkpoint selection is invalid"
            )
        if request is not None and (
            request.acs_checkpoint_slots != plan_acs_checkpoint_slots
            or request.acs_capture_instrumentation
            is not plan_acs_capture_instrumentation
            or request.native_streamed_checkpoint is not plan_native_streamed_checkpoint
            or request.world_sharded_checkpoint is not plan_world_sharded_checkpoint
        ):
            raise CampaignProviderError(
                "resident request ACS checkpoint route binding mismatch"
            )
        route_identity = (
            execution.get("execution_route_identity")
            if isinstance(execution, Mapping)
            else None
        )
        if route_identity == TICK_NATIVE_EXECUTION_ROUTE_V1:
            tick_request = self.tick_native_execution_request_contract(payload)
            tick_control = self.tick_native_control_contract(payload)
            base = self._patch_fixed_execution_route_contract(
                payload, request=None, tick_native_route=True
            )

            public_forbidden_fields = {
                "generation",
                "generations",
                "generation_min",
                "generation_max",
                "generation_count",
                "logical_generation_count",
                "initial_generation",
                "terminal_generation",
                "start_generation",
                "end_generation",
                "absolute_generation",
                "ticks_per_generation",
                "admitted_ticks_per_generation",
                "request_ticks_per_generation",
                "physical_window_generations",
                "resident_segment_generations",
                "physical_storage_generations",
                "fresh_owner_partial_generation_restore",
                "legacy_generation_authority",
                "refill_chunk_ticks",
                "legacy_turnover_interval_ticks",
            }

            def public_projection(value: Any) -> Any:
                if isinstance(value, Mapping):
                    return {
                        key: public_projection(item)
                        for key, item in value.items()
                        if key not in public_forbidden_fields
                    }
                if isinstance(value, tuple):
                    return tuple(public_projection(item) for item in value)
                if isinstance(value, list):
                    return [public_projection(item) for item in value]
                return value

            public_components = tuple(
                public_projection(component) for component in base.components
            )
            contract = replace(
                base,
                components=(
                    *public_components,
                    {
                        "component_id": "absolute_tick_time_contract",
                        "time_authority": "absolute_tick_v1",
                        "cursor_semantics": "committed_half_open_[0,next_tick)",
                        "arbitrary_positive_terminal_tick": True,
                        "arbitrary_checkpoint_ticks": True,
                        "fresh_owner_partial_tick_restore": True,
                        "source_window_ticks": tick_request.source_window_ticks,
                        "resident_owner_span_ticks": (
                            tick_request.resident_owner_span_ticks
                        ),
                        "native_dispatch_window_ticks": (
                            tick_request.native_dispatch_window_ticks
                        ),
                        "event_window_capacity": tick_request.event_window_capacity,
                        "legacy_horizon_authority": False,
                    },
                    {
                        "component_id": "tick_native_control",
                        "contract_version": "eiel-tick-native-control-v2",
                        "time_authority": "absolute_tick_v1",
                        "control": tick_request.control_payload(),
                    },
                ),
                required_obligations=tuple(
                    sorted(
                        {
                            *(
                                obligation
                                for obligation in base.required_obligations
                                if "generation" not in obligation
                            ),
                            "non_lattice_divisible_terminal_tick",
                            "non_lattice_divisible_checkpoint_tick",
                            "exact_absolute_tick_checkpoint_export",
                            "fresh_owner_partial_tick_restore",
                            "first_committed_tick_after_partial_restore",
                        }
                    )
                ),
            )
            if (
                request is not None
                and request.execution_route_sha256 is not None
                and request.execution_route_sha256 != contract.execution_route_sha256
            ):
                raise CampaignProviderError(
                    "resident tick request execution-route binding mismatch"
                )
            if request is not None and dict(request.tick_native_control) != (
                tick_control.payload()
            ):
                raise CampaignProviderError(
                    "resident tick request control binding mismatch"
                )
            return contract
        if route_identity not in (None, "legacy_generation_route_v1"):
            raise CampaignProviderError(
                "production provider only admits legacy_generation_route_v1; "
                "tick-native provider/native execution is not qualified or implemented"
            )
        if self._direct_resident_binding(payload) is not None:
            return self._patch_fixed_execution_route_contract(payload, request=request)
        if not self._retired_cross_path_configuration(payload):
            raise CampaignProviderError(
                self._direct_resident_binding_unmigrated_message(payload)
            )
        return self._cross_path_execution_route_contract(payload, request=request)

    @staticmethod
    def tick_native_execution_request_contract(
        plan: CampaignPlanV1 | Mapping[str, Any],
    ) -> TickNativeExecutionRequestV1:
        """Parse a direct tick request without manufacturing G/T authority."""

        execution = _payload(plan).get("execution")
        try:
            return TickNativeExecutionRequestV1.from_execution(execution)
        except TickNativeError as exc:
            raise CampaignProviderError(
                f"invalid tick-native provider request: {exc}"
            ) from exc

    @staticmethod
    def tick_native_control_contract(
        plan: CampaignPlanV1 | Mapping[str, Any],
    ) -> TickNativeControlV1:
        """Parse the single authoritative resident tick control."""

        execution = _payload(plan).get("execution")
        try:
            control = TickNativeControlV1.from_execution(execution)
        except (TickNativeControlError, TypeError, ValueError) as exc:
            raise CampaignProviderError(
                f"invalid tick-native provider control: {exc}"
            ) from exc
        return control

    def _patch_fixed_execution_route_contract(
        self,
        plan: Mapping[str, Any],
        *,
        request: ResidentBatchRequestV1 | None,
        tick_native_route: bool = False,
    ) -> ExecutionRouteContractV1:
        configuration = production_engine_configuration_v1(plan)
        binding = configuration.component_binding()
        native_binding = self._direct_resident_binding(plan)
        if native_binding is None:
            raise CampaignProviderError(
                "direct-resident component binding is not migrated"
            )
        from engine.evolution.global_genome_registry_v1 import (
            global_genome_profile_v1,
        )

        genome_profile = global_genome_profile_v1(
            binding["genome_inheritance"]["configuration_id"]
        )
        lifecycle_control = production_engine_lifecycle_control_v1(plan)
        execution = plan.get("execution")
        acs_checkpoint_slots = (
            execution.get("acs_checkpoint_slots", 0)
            if isinstance(execution, Mapping)
            else 0
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
            0,
            1,
            2,
            3,
        }:
            raise CampaignProviderError(
                "Patch engine route ACS checkpoint slots are invalid"
            )
        raw_acs_capture_instrumentation = (
            execution.get("acs_capture_instrumentation", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "Patch engine route ACS capture instrumentation is invalid"
            )
        acs_capture_instrumentation = raw_acs_capture_instrumentation or bool(
            acs_checkpoint_slots
        )
        raw_native_streamed_checkpoint = (
            execution.get("native_streamed_checkpoint", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_native_streamed_checkpoint) is not bool:
            raise CampaignProviderError(
                "Patch engine route native streamed checkpoint selection is invalid"
            )
        raw_world_sharded_checkpoint = (
            execution.get("world_sharded_checkpoint", False)
            if isinstance(execution, Mapping)
            else False
        )
        if type(raw_world_sharded_checkpoint) is not bool:
            raise CampaignProviderError(
                "Patch engine route world-sharded checkpoint selection is invalid"
            )
        native_acs_runtime_component = _acs_native_runtime_route_component_v1(
            plan
        )
        if tick_native_route:
            if acs_checkpoint_slots not in {0, 1, 2}:
                raise CampaignProviderError(
                    "tick-native Patch engine route ACS slots must be zero, one, or two"
                )
            if acs_checkpoint_slots and not raw_native_streamed_checkpoint:
                raise CampaignProviderError(
                    "tick-native asynchronous ACS requires native_streamed_checkpoint"
                )
            if raw_world_sharded_checkpoint and (
                not raw_native_streamed_checkpoint or acs_checkpoint_slots not in {1, 2}
            ):
                raise CampaignProviderError(
                    "world-sharded checkpoint route requires tick-native ACS1 or ACS2"
                )
        if request is not None and request.native_streamed_checkpoint is not raw_native_streamed_checkpoint:
            raise CampaignProviderError(
                "resident request native checkpoint route binding mismatch"
            )
        if request is not None and request.world_sharded_checkpoint is not raw_world_sharded_checkpoint:
            raise CampaignProviderError(
                "resident request world-sharded checkpoint route binding mismatch"
            )
        physical_window_declared = False
        physical_window_generations = None
        if isinstance(execution, Mapping):
            physical_window_generations = execution.get("physical_window_generations")
            physical_window_declared = physical_window_generations is not None
        if physical_window_generations is None:
            physical_window_generations = 2
        if type(physical_window_generations) is not int:
            raise CampaignProviderError(
                "Patch engine route physical_window_generations must be an integer"
            )
        resident_window_route_component: tuple[Mapping[str, Any], ...] = ()
        if physical_window_declared:
            from .resources import (
                fixed_mlp_policy_window_bytes_per_cell_v1,
                patch_resident_window_route_id_v1,
            )

            try:
                resident_window_route_id = patch_resident_window_route_id_v1(
                    physical_window_generations=physical_window_generations
                )
                policy_window = fixed_mlp_policy_window_bytes_per_cell_v1(
                    physical_window_generations=physical_window_generations
                )
            except ValueError as exc:
                raise CampaignProviderError(
                    "Patch engine route has unsupported physical_window_generations"
                ) from exc
            resident_window_route_component = (
                {
                    "component_id": "resident_window_route",
                    "route_identity": resident_window_route_id,
                    "physical_window_generations": physical_window_generations,
                    "max_ticks_per_generation": 160,
                    "turnover_episodes_per_generation": 62,
                    "ordinary_birth_episode_capacity": (
                        policy_window["physical_window_ticks"] * 64
                    ),
                    "turnover_episode_capacity": policy_window["turnover_episodes"],
                    "policy_episode_capacity": policy_window["policy_episode_capacity"],
                    "packed_crossover_codec": "bitpack_lsb0_v1",
                    "packed_torch_state_codec": (
                        "torch_cpu_rng_state_5056_u32_middle_v1"
                    ),
                    "cpython_gauss_dtype": "float64",
                    "cpython_gauss_raw_capacity_per_birth": 512,
                    "provider_runtime_identity": self.profile_id,
                    "checkpoint_schema": self.checkpoint_schema_version,
                    "descriptor_schema": (
                        "eiel_resident_prepared_owner_resource_audit_v1"
                    ),
                },
            )
        artifact_rows = _artifact_rows(plan)
        cpu_native_route = self.profile_id == CPU_PROFILE_ID
        native_artifact_ids = (
            () if cpu_native_route else native_binding.required_artifact_ids
        )
        required_artifacts = tuple(
            dict.fromkeys(
                (
                    *native_artifact_ids,
                    *genome_profile.required_artifact_ids,
                )
            )
        )
        if any(
            artifact_id not in artifact_rows
            or not _valid_sha256(artifact_rows[artifact_id].get("sha256"))
            or (
                artifact_id in native_binding.abi_bound_artifact_ids
                and artifact_rows[artifact_id].get("abi") != native_binding.artifact_abi
            )
            for artifact_id in required_artifacts
        ):
            raise CampaignProviderError(
                "Patch engine route lacks its declared artifact identities"
            )
        artifacts = {
            artifact_id: artifact_rows[artifact_id]["sha256"]
            for artifact_id in required_artifacts
        }
        genome_package = None
        if genome_profile.profile_id == "genome_upgrade_v2":
            genome_package = _validate_genome_v2_package_v1(
                plan,
                required_obligations=genome_profile.required_obligations,
            )
        source_dependency_closure: dict[str, Any] = {
            "provider_sha256": _file_sha256(Path(__file__).resolve()),
        }
        if cpu_native_route:
            from engine.perf.cxxbackbone10_resident_state_owner import (
                NATIVE_EXTENSION_SOURCES,
                resident_state_owner_extension_status,
            )

            campaigns_root = Path(__file__).resolve().parent
            prebuilt_status = resident_state_owner_extension_status()
            prebuilt_status_payload = _stable_resident_owner_prebuilt_route_status(
                prebuilt_status
            )
            prebuilt_path = (
                Path(prebuilt_status.extension_path)
                if prebuilt_status.available and prebuilt_status.extension_path
                else None
            )
            source_dependency_closure.update(
                {
                    "native_bindings_sha256": _file_sha256(
                        campaigns_root / "production_engine_native_bindings.py"
                    ),
                    "patch_runner_factory_sha256": _file_sha256(
                        campaigns_root / "patch_runner_factory.py"
                    ),
                    "native_runtime_engine_sha256": _file_sha256(
                        Path(__file__).resolve().parents[1]
                        / "native_runtime"
                        / "engine.py"
                    ),
                    "resident_state_owner_sha256": _file_sha256(
                        Path(__file__).resolve().parents[1]
                        / "perf"
                        / "cxxbackbone10_resident_state_owner.py"
                    ),
                    "resident_state_owner_native_sources": tuple(
                        {
                            "path": str(Path(path).resolve()),
                            "sha256": _file_sha256(Path(path).resolve()),
                        }
                        for path in NATIVE_EXTENSION_SOURCES
                    ),
                    "prebuilt_extension_path": (
                        str(prebuilt_path) if prebuilt_path is not None else None
                    ),
                    "prebuilt_extension_sha256": (
                        _file_sha256(prebuilt_path)
                        if prebuilt_path is not None
                        else None
                    ),
                    "prebuilt_extension_status": prebuilt_status_payload,
                }
            )
        else:
            extension_row = artifact_rows["cuda_extension"]
            extension_path = Path(str(extension_row.get("path") or "")).expanduser()
            if not extension_path.is_absolute():
                raise CampaignProviderError(
                    "Patch engine route requires an absolute artifact path"
                )
            if extension_row.get("abi") != native_binding.artifact_abi:
                raise CampaignProviderError(
                    "native artifact ABI differs from its engine binding"
                )
            source_dependency_closure.update(
                {
                    "prebuilt_extension_path": str(extension_path),
                    "prebuilt_extension_sha256": extension_row["sha256"],
                }
            )
        qualification_key = plan.get("qualification_key")
        admitted_range = (
            qualification_key.get("admitted_batch_range")
            if isinstance(qualification_key, Mapping)
            else None
        )
        multicell_qualified = bool(
            isinstance(admitted_range, Sequence)
            and not isinstance(admitted_range, (str, bytes))
            and len(admitted_range) == 2
            and type(admitted_range[1]) is int
            and admitted_range[1] > 1
            and native_binding.batch_max > 1
        )
        if lifecycle_control is None:
            birth_obligations = {"ordinary_birth", "turnover_refill"}
        else:
            birth_obligations = (
                {"ordinary_birth_disabled"}
                if lifecycle_control.ordinary_births_enabled is False
                else {"ordinary_birth"}
            )
            if (
                lifecycle_control.turnover_enabled is True
                and lifecycle_control.turnover_cull_fraction == 0.5
                and lifecycle_control.turnover_parent_fraction == 1.0
            ):
                birth_obligations.add("turnover_refill")
        contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                {
                    "component_id": "production_engine_lifecycle",
                    "implementation": (
                        "engine.campaigns.providers.ProductionEngineLifecycleV1"
                    ),
                    "contract_version": "eiel_production_lifecycle_v1",
                },
                *(
                    (
                        {
                            "component_id": "acs_checkpoint_persistence",
                            "contract_version": (
                                "acs_detached_file_backed_native_v1"
                                if tick_native_route
                                else "acs_bounded_single_writer_v1"
                            ),
                            "immutable_snapshot_slots": acs_checkpoint_slots,
                            "backpressure": "block_without_overwrite_or_drop",
                            "frontiers": "compute_and_durable_checkpoint_distinct",
                            **(
                                {
                                    "worker_boundary": "path_and_scalar_metadata_only",
                                    "worker_accepts_cuda_objects": False,
                                    "worker_accepts_callbacks": False,
                                    "capture_pool": "one_shared_two_slab_not_per_slot",
                                }
                                if tick_native_route
                                else {}
                            ),
                        },
                    )
                    if acs_checkpoint_slots
                    else ()
                ),
                *(
                    (
                        {
                            "component_id": ("acs_checkpoint_capture_instrumentation"),
                            "contract_version": ("acs_capture_barrier_measurement_v1"),
                            "persistence_mode": (
                                (
                                    "detached_file_backed_native_v1"
                                    if tick_native_route
                                    else "bounded_async_single_writer_v1"
                                )
                                if acs_checkpoint_slots
                                else "synchronous_instrumentation_only_v1"
                            ),
                        },
                    )
                    if acs_capture_instrumentation
                    else ()
                ),
                *(
                    (
                        {
                            "component_id": "acs_native_checkpoint_format",
                            "contract_version": "acs_native_checkpoint_route_v1",
                            "physical_format": (
                                "acs100x_native_streamed_checkpoint_v1"
                            ),
                            "multicell_physical_format": (
                                "acs100x_native_aggregate_checkpoint_v2"
                            ),
                            "physical_format_selection": (
                                "aggregate_for_multicell_batch_v1"
                            ),
                            "strict_prebuilt_extension": True,
                            "native_extension_abi": "eiel_native_runtime_abi_v1",
                        },
                    )
                    if raw_native_streamed_checkpoint
                    else ()
                ),
                *(
                    (native_acs_runtime_component,)
                    if native_acs_runtime_component is not None
                    else ()
                ),
                {
                    "component_id": "runtime_execution_backend",
                    "profile_id": self.profile_id,
                    "device": self.device,
                    "backend": "cpu_native" if cpu_native_route else "cuda_resident",
                    "fallback_allowed": False,
                },
                {
                    "component_id": "production_engine_configuration",
                    "configuration": configuration.payload(),
                    "configuration_sha256": configuration.configuration_sha256,
                },
                {
                    "component_id": "production_engine_component_binding",
                    "binding": binding,
                    "binding_sha256": canonical_sha256(binding),
                },
                {
                    "component_id": "qualification_identity",
                    "qualification_key_sha256": plan.get("qualification_key_sha256"),
                    "fixed_invariants": dict(
                        sorted(
                            dict(
                                qualification_key.get("fixed_invariants", {})
                                if isinstance(qualification_key, Mapping)
                                else {}
                            ).items()
                        )
                    ),
                },
                *(
                    (
                        {
                            "component_id": "production_engine_lifecycle_control",
                            "control": lifecycle_control.payload(),
                            "control_sha256": lifecycle_control.control_sha256,
                            "native_binding": (
                                "direct_resident_outer_172_neutral_turnover"
                            ),
                        },
                    )
                    if lifecycle_control is not None
                    else ()
                ),
                {
                    "component_id": "native_mechanics_binding",
                    "binding_id": native_binding.binding_id,
                    "artifact_abi": native_binding.artifact_abi,
                    "batch_max": native_binding.batch_max,
                    "generation_min": native_binding.generation_min,
                    "generation_max": native_binding.generation_max,
                    "ticks_per_generation": native_binding.ticks_per_generation,
                    "admitted_ticks_per_generation": tuple(
                        getattr(
                            native_binding,
                            "admitted_ticks_per_generation",
                            (),
                        )
                        or (native_binding.ticks_per_generation,)
                    ),
                    "request_ticks_per_generation": dict(plan.get("execution", {})).get(
                        "ticks_per_generation"
                    ),
                },
                *(
                    (
                        {
                            "component_id": "policy_cache_restore",
                            "cache_contract": "flattened_fixed_mlp_policy_cache_v1",
                            "restore_contract": "fresh_owner_policy_cache_restore_v1",
                            "dispatch_boundary_refill": False,
                        },
                    )
                    if tick_native_route and native_binding.policy_cache_refill
                    else ()
                ),
                *resident_window_route_component,
                {
                    "component_id": "global_genome_profile",
                    "profile": genome_profile.payload(),
                    "profile_sha256": canonical_sha256(genome_profile.payload()),
                },
                *(
                    (
                        {
                            "component_id": ("genome_upgrade_v2_executable_package"),
                            "receipt_sha256": artifact_rows[
                                "genome_upgrade_v2_package_receipt"
                            ]["sha256"],
                            "mechanic_contract_sha256": genome_package["identity"][
                                "mechanic_contract_sha256"
                            ],
                            "capability_contract_sha256": genome_package["identity"][
                                "capability_contract_sha256"
                            ],
                        },
                    )
                    if genome_package is not None
                    else ()
                ),
                {
                    "component_id": "production_artifact_chain",
                    "artifact_sha256": artifacts,
                },
                {
                    "component_id": "source_dependency_closure",
                    **source_dependency_closure,
                },
            ),
            required_obligations=tuple(
                sorted(
                    {
                        "cold_start",
                        "first_generation",
                        "first_tick",
                        "intra_generation",
                        "generation_boundary",
                        "first_post_boundary_tick",
                        "second_post_boundary_tick",
                        *birth_obligations,
                        "checkpoint_export",
                        "disk_checkpoint_reload",
                        "fresh_owner_restore",
                        "first_post_restore_tick",
                        "terminal_settlement",
                        "reporting_parity",
                        *(
                            {
                                "async_checkpoint_backpressure",
                                "async_checkpoint_failure_latch",
                                "durable_frontier_publication",
                            }
                            if acs_checkpoint_slots
                            else set()
                        ),
                        *(
                            {"acs_capture_barrier_measurement"}
                            if acs_capture_instrumentation
                            else set()
                        ),
                        *(
                            {
                                "first_post_restore_refill",
                                *(
                                    {"post_boundary_policy_cache_refill"}
                                    if not tick_native_route
                                    else set()
                                ),
                            }
                            if native_binding.policy_cache_refill
                            else set()
                        ),
                        *(
                            {
                                "multicell_execution",
                                "odd_physical_tail",
                                "alternate_batch_partition",
                            }
                            if multicell_qualified
                            else set()
                        ),
                    }
                )
            ),
        )
        if (
            request is not None
            and request.execution_route_sha256 is not None
            and request.execution_route_sha256 != contract.execution_route_sha256
        ):
            raise CampaignProviderError(
                "resident request execution-route binding mismatch"
            )
        return contract

    def _cross_path_execution_route_contract(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1 | None = None,
    ) -> ExecutionRouteContractV1:
        payload = _payload(plan)
        engine_configuration = production_engine_configuration_v1(payload)
        component_binding = engine_configuration.component_binding()
        registry = self._ensure_committed_default_adapters(payload)
        _validate_native(self._native(registry), registry)
        delegated = ProductionEngineLifecycleV1.execution_route_contract(self, payload)
        extension_path, extension_row = _verified_artifact(payload, "cuda_extension")
        artifact_chain = {
            artifact_id: _verified_artifact(payload, artifact_id)[1]["sha256"]
            for artifact_id in (
                "cuda_extension",
                "extension_manifest",
                "inheritance_registry",
                "telemetry_contract",
                "qualification_receipt",
                "reporting_receipt",
            )
        }
        if extension_row["abi"] != registry["native_abi"]:
            raise CampaignProviderError("cross-path CUDA ABI disagrees with registry")
        contract = ExecutionRouteContractV1(
            schema_version=EXECUTION_ROUTE_CONTRACT_SCHEMA_VERSION,
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            components=(
                *tuple(
                    component
                    for component in delegated.components
                    if component["component_id"] != "source_dependency_closure"
                ),
                {
                    "component_id": "production_engine_configuration",
                    "configuration": engine_configuration.payload(),
                    "configuration_sha256": (engine_configuration.configuration_sha256),
                },
                {
                    "component_id": "production_engine_component_binding",
                    "binding": component_binding,
                    "binding_sha256": canonical_sha256(component_binding),
                },
                {
                    "component_id": "inheritance_cross_path_registry",
                    "registry_sha256": _artifact_rows(payload)["inheritance_registry"][
                        "sha256"
                    ],
                    "runtime_identity": registry["runtime_identity"],
                    "parameter_bindings": registry["parameter_bindings"],
                    "semantic_types": registry["semantic_types"],
                    "semantic_ids": registry["semantic_ids"],
                    "inheritance_semantic": registry["inheritance_semantic"],
                    "draw_budgets": registry["draw_budgets"],
                    "audited_loci": registry["audited_loci"],
                    "linkage_blocks": registry["linkage_blocks"],
                },
                {
                    "component_id": "inheritance_cross_path_execution_adapters",
                    "fallback": False,
                    "implementations": self._adapter_identities(),
                },
                {
                    "component_id": "inheritance_cross_path_artifact_chain",
                    "loader_contract": (
                        "inheritance_cross_path_qualified_artifact_loader_v1"
                    ),
                    "artifact_sha256": artifact_chain,
                },
                {
                    "component_id": "source_dependency_closure",
                    "provider_sha256": _file_sha256(Path(__file__).resolve()),
                    "native_abi": registry["native_abi"],
                    "prebuilt_extension_path": str(extension_path),
                    "prebuilt_extension_sha256": extension_row["sha256"],
                },
            ),
            required_obligations=tuple(
                sorted(
                    {
                        *delegated.required_obligations,
                        *registry["required_obligations"],
                    }
                )
            ),
        )
        if request is not None and request.execution_route_sha256 not in {
            None,
            contract.execution_route_sha256,
        }:
            raise CampaignProviderError("cross-path route hash mismatch")
        return contract

    def _mechanic_receipt_contract(
        self,
        plan: Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> Mapping[str, Any]:
        """Validate the receipt binding without granting process-global authority."""

        self._last_mechanic_receipt_sha256 = None
        path, row = _verified_artifact(plan, "qualification_receipt")
        registry = _global_genome_profile_contract(plan)
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignProviderError(
                "cross-path mechanic qualification receipt is unreadable"
            ) from exc
        execution = (
            receipt.get("execution_contract") if isinstance(receipt, Mapping) else None
        )
        artifact = receipt.get("artifact") if isinstance(receipt, Mapping) else None
        registry_row = _artifact_rows(plan)["inheritance_registry"]
        extension_row = _artifact_rows(plan)["cuda_extension"]
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("schema_version") != MECHANIC_QUALIFICATION_SCHEMA_VERSION
            or receipt.get("ok") is not True
            or receipt.get("provider_id") != self.provider_id
            or receipt.get("profile_id") != self.profile_id
            or receipt.get("registry_sha256") != registry_row.get("sha256")
            or not isinstance(artifact, Mapping)
            or artifact.get("sha256") != extension_row.get("sha256")
            or artifact.get("abi") != registry["native_abi"]
            or not isinstance(execution, Mapping)
            or execution.get("lifecycle_owner") != PRODUCTION_LIFECYCLE_OWNER_ID
            or execution.get("checkpoint_schema_version")
            != self.checkpoint_schema_version
            or execution.get("ticks_per_generation") != registry["ticks_per_generation"]
        ):
            raise CampaignProviderError(
                "cross-path mechanic qualification receipt binding is invalid"
            )
        batch_range = execution.get("physical_batch_range")
        generation_range = execution.get("logical_generation_range")
        if (
            not isinstance(batch_range, list)
            or len(batch_range) != 2
            or any(type(value) is not int for value in batch_range)
            or batch_range[0] < 1
            or batch_range[1] > registry["max_admitted_batch"]
            or batch_range[0] > len(request.cell_ids)
            or len(request.cell_ids) > batch_range[1]
            or not isinstance(generation_range, list)
            or len(generation_range) != 2
            or any(type(value) is not int for value in generation_range)
            or generation_range[0] < 1
            or generation_range[1] > registry["max_generations"]
        ):
            raise CampaignProviderError(
                "cross-path mechanic qualification execution envelope is invalid"
            )
        boundaries = tuple(
            sorted({0, *request.checkpoint_milestones, request.generations})
        )
        segments = tuple(
            right - left for left, right in zip(boundaries, boundaries[1:])
        )
        if (
            request.ticks_per_generation != registry["ticks_per_generation"]
            or not segments
            or any(
                value < generation_range[0] or value > generation_range[1]
                for value in segments
            )
        ):
            raise CampaignProviderError(
                "cross-path request exceeds the qualified mechanic envelope"
            )
        self._last_mechanic_receipt_sha256 = str(row["sha256"])
        return dict(receipt)

    def _execution_adapter_reason(self) -> str | None:
        missing = [
            name.removeprefix("_")
            for name in (
                "_artifact_loader",
                "_runner_builder",
                "_request_builder",
                "_preparer",
                "_executor",
                "_projector",
                "_source_preparer",
            )
            if getattr(self, name) is None
        ]
        if missing:
            return "cross-path execution adapters are not bound: " + ", ".join(missing)
        return None

    @classmethod
    def _rolling_segment_ends(
        cls,
        *,
        generations: int,
        declared_milestones: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Keep the declarative G0 snapshot marker out of physical dispatches."""

        return super()._rolling_segment_ends(
            generations=generations,
            declared_milestones=tuple(
                value for value in declared_milestones if value != 0
            ),
        )

    def _configured_rolling_segment_ends(
        self,
        plan: Mapping[str, Any],
        *,
        generations: int,
        declared_milestones: tuple[int, ...],
    ) -> tuple[int, ...]:
        native_binding = self._direct_resident_binding(plan)
        if native_binding is None:
            raise CampaignProviderError(
                "direct-resident component binding is not migrated"
            )
        maximum = native_binding.generation_max
        minimum = native_binding.generation_min
        declared = tuple(
            sorted(
                {
                    generations,
                    *(value for value in declared_milestones if value != 0),
                }
            )
        )
        if any(value < 0 or value > generations for value in declared):
            raise CampaignProviderError(
                "checkpoint milestones are outside the campaign"
            )
        ends: list[int] = []
        left = 0
        for right in (value for value in declared if value > 0):
            gap = right - left
            if gap <= maximum:
                ends.append(right)
                left = right
                continue
            parts = (gap + maximum - 1) // maximum
            width, wider = divmod(gap, parts)
            if width < minimum:
                ends.append(right)
                left = right
                continue
            current = left
            for ordinal in range(parts):
                current += width + (1 if ordinal < wider else 0)
                ends.append(current)
            left = right
        return tuple(ends)

    def _require_execution_authority(
        self,
        plan: Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> None:
        if self._direct_resident_binding(plan) is not None:
            reasons = self._patch_fixed_readiness_reasons(plan, request)
            if reasons:
                raise CampaignProviderError(reasons[0].message)
            return
        if not self._retired_cross_path_configuration(plan):
            raise CampaignProviderError(
                self._direct_resident_binding_unmigrated_message(plan)
            )
        self._require_cross_path_execution_authority(plan, request)

    def _require_cross_path_execution_authority(
        self,
        plan: Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> None:
        self._mechanic_receipt_contract(plan, request)
        self.execution_route_contract(plan, request)
        adapter_reason = self._execution_adapter_reason()
        if adapter_reason is not None:
            raise CampaignProviderError(adapter_reason)
        for artifact_id in ("extension_manifest", "reporting_receipt"):
            _verified_artifact(plan, artifact_id)
        if self._cross_path_artifact_chain_loader_v1 is not None:
            try:
                self._cross_path_artifact_chain_loader_v1.validate_chain()
            except RuntimeError as exc:
                raise CampaignProviderError(
                    f"cross-path qualified artifact chain is invalid: {exc}"
                ) from exc

    def _patch_fixed_readiness_reasons(
        self,
        plan: Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]:
        reasons: list[UnsupportedReasonV1] = []
        native_binding = self._direct_resident_binding(plan)
        if native_binding is None:
            raise CampaignProviderError(
                "direct-resident component binding is not migrated"
            )
        try:
            production_engine_lifecycle_control_v1(plan)
        except CampaignProviderError as exc:
            return (
                UnsupportedReasonV1(
                    "production_engine_lifecycle_invalid",
                    "runtime.lifecycle",
                    str(exc),
                    self.profile_id,
                ),
            )
        parameter_arrays = dict(request.parameter_arrays)
        initial_populations = parameter_arrays.get("patch.initial_population", ())
        resident_capacities = parameter_arrays.get("patch.capacity", ())
        lifecycle_control = production_engine_lifecycle_control_v1(plan)
        ordinary_births_enabled = (
            True
            if lifecycle_control is None
            else bool(lifecycle_control.ordinary_births_enabled)
        )
        if (
            ordinary_births_enabled
            and initial_populations
            and resident_capacities
            and all(type(value) is int for value in initial_populations)
            and all(type(value) is int for value in resident_capacities)
            and max(initial_populations) >= min(resident_capacities)
        ):
            reasons.append(
                UnsupportedReasonV1(
                    "production_engine_ordinary_birth_headroom_absent",
                    "parameters.patch.capacity",
                    "ordinary-birth Patch routes require declared resident "
                    "capacity greater than initial population; c must equal "
                    "actual allocated resident slots",
                    self.profile_id,
                )
            )
        if request.profile_id != self.profile_id:
            reasons.append(
                UnsupportedReasonV1(
                    "production_engine_profile_mismatch",
                    "runtime.profile",
                    "resident request does not bind the production runtime",
                    self.profile_id,
                )
            )
        if len(request.cell_ids) > native_binding.batch_max:
            reasons.append(
                UnsupportedReasonV1(
                    "production_engine_batch_out_of_range",
                    "execution.batch",
                    "configured native mechanics do not admit this batch",
                    self.profile_id,
                )
            )
        tick_native = request.total_ticks is not None
        if not tick_native:
            raw_admitted_ticks = getattr(
                native_binding,
                "admitted_ticks_per_generation",
                (),
            ) or (native_binding.ticks_per_generation,)
            admitted_ticks = tuple(
                sorted(
                    {
                        int(value)
                        for value in raw_admitted_ticks
                        if type(value) is int and value > 0
                    }
                )
            )
            if not admitted_ticks or len(admitted_ticks) != len(
                tuple(raw_admitted_ticks)
            ):
                raise CampaignProviderError(
                    "configured native mechanics expose an invalid tick envelope"
                )
            if request.ticks_per_generation not in admitted_ticks:
                reasons.append(
                    UnsupportedReasonV1(
                        "production_engine_ticks_mismatch",
                        "execution.ticks_per_generation",
                        "configured native mechanics admit ticks per generation "
                        f"{list(admitted_ticks)!r}",
                        self.profile_id,
                    )
                )
        if tick_native:
            segments = tuple(
                int(segment["logical_generation_count"])
                for segment in ProductionEngineLifecycleV1._absolute_tick_segments_v1(
                    request,
                    start_next_tick=request.start_next_tick,
                )
            )
        else:
            boundaries = tuple(
                sorted({0, *request.checkpoint_milestones, request.generations})
            )
            segments = tuple(
                right - left for left, right in zip(boundaries, boundaries[1:])
            )
        technical_gate_short_probe = (
            request.batch_id
            in {"technical-gate-canary", "technical-gate-fresh-owner-restore"}
            and len(request.cell_ids) == 1
            and bool(segments)
            and all(
                1 <= segment <= native_binding.generation_max for segment in segments
            )
        )
        if tick_native:
            segment_out_of_range = not segments or any(
                segment < 1 or segment > native_binding.generation_max
                for segment in segments
            )
        else:
            segment_out_of_range = not segments or any(
                segment < native_binding.generation_min
                or segment > native_binding.generation_max
                for segment in segments
            )
        if segment_out_of_range:
            if not technical_gate_short_probe:
                reasons.append(
                    UnsupportedReasonV1(
                        "production_engine_segment_out_of_range",
                        "checkpoints.milestones",
                        "configured native mechanics do not admit this segment",
                        self.profile_id,
                    )
                )
        if (
            not tick_native
            and native_binding.generation_min < 3
            and any(segment <= 2 for segment in segments)
            and len(request.cell_ids) != 1
        ):
            reasons.append(
                UnsupportedReasonV1(
                    "production_engine_short_segment_requires_b1",
                    "checkpoints.milestones",
                    "configured recurrent g1/g2 segments require B1",
                    self.profile_id,
                )
            )
        device_family = self.device.split(":", 1)[0]
        if self.profile_id == CPU_PROFILE_ID:
            if device_family != "cpu":
                reasons.append(
                    UnsupportedReasonV1(
                        "production_engine_cpu_native_profile_device_mismatch",
                        "runtime.cpu",
                        "CPU-native Patch production requires the cpu device with no fallback",
                        self.profile_id,
                    )
                )
            from engine.perf.cxxbackbone10_resident_state_owner import (
                resident_state_owner_extension_status,
            )

            prebuilt_status = resident_state_owner_extension_status()
            if not prebuilt_status.available:
                reasons.append(
                    UnsupportedReasonV1(
                        "production_engine_cpu_native_prebuilt_unavailable",
                        "runtime.cpu.prebuilt_extension",
                        "CPU-native Patch production requires a verified "
                        "prebuilt resident owner extension with no runtime JIT: "
                        f"{prebuilt_status.error}",
                        self.profile_id,
                    )
                )
        elif device_family != "cuda" or not self.torch.cuda.is_available():
            reasons.append(
                UnsupportedReasonV1(
                    "production_engine_cuda_unavailable",
                    "runtime.cuda",
                    "Patch production execution requires CUDA with no fallback",
                    self.profile_id,
                )
            )
        if self.profile_id != CPU_PROFILE_ID:
            for artifact_id in (*native_binding.required_artifact_ids,):
                try:
                    _path, row = _verified_artifact(plan, artifact_id)
                    if (
                        artifact_id in native_binding.abi_bound_artifact_ids
                        and row.get("abi") != native_binding.artifact_abi
                    ):
                        raise CampaignProviderError(
                            f"{artifact_id} ABI differs from Patch engine binding"
                        )
                except CampaignProviderError as exc:
                    reasons.append(
                        UnsupportedReasonV1(
                            "production_engine_artifact_unqualified",
                            f"artifacts.{artifact_id}",
                            str(exc),
                            self.profile_id,
                        )
                    )
        if (
            request.native_streamed_checkpoint
            and self.profile_id == PRODUCTION_PROFILE_ID
        ):
            try:
                _acs_native_runtime_artifact_binding_v1(plan, required=True)
            except CampaignProviderError as exc:
                reasons.append(
                    UnsupportedReasonV1(
                        "production_engine_native_acs_runtime_unbound",
                        "artifacts.acs_native_runtime_extension",
                        str(exc),
                        self.profile_id,
                    )
                )
        try:
            self.execution_route_contract(plan, request)
        except CampaignProviderError as exc:
            reasons.append(
                UnsupportedReasonV1(
                    "production_engine_route_unqualified",
                    "runtime.execution_route",
                    str(exc),
                    self.profile_id,
                )
            )
        return tuple(reasons)

    def execute_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        context: Any,
    ) -> dict[str, Any]:
        payload = _payload(plan)
        self._require_execution_authority(payload, request)
        return ProductionEngineLifecycleV1.execute_batch(
            self, payload, request, context
        )

    def resume_batch(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
        checkpoint: Mapping[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        payload = _payload(plan)
        self._require_execution_authority(payload, request)
        return ProductionEngineLifecycleV1.resume_batch(
            self, payload, request, checkpoint, context
        )

    def _evaluate_tick_native_technical_gate_v1(
        self,
        plan: Mapping[str, Any],
        gate_cell_ids: Sequence[str],
        batch_manifests: Sequence[Mapping[str, Any]],
        run_root: Path,
    ) -> dict[str, Any]:
        """Evaluate a pre-primary gate using absolute-tick evidence only."""

        del run_root
        payload = _payload(plan)
        execution = payload.get("execution")
        gate = (
            execution.get("technical_gate") if isinstance(execution, Mapping) else None
        )
        canary = gate.get("canary") if isinstance(gate, Mapping) else None
        reasons: list[str] = []
        checks: dict[str, Any] = {}

        def fail(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        expected_ids: tuple[str, ...] = ()
        total_ticks: int | None = None
        expected_ticks: tuple[int, ...] = ()
        canary_batch_id: str | None = None
        restore_batch_id: str | None = None
        if (
            not isinstance(gate, Mapping)
            or gate.get("required") is not True
            or gate.get("execution_mode") != "pre_primary_canary"
            or not isinstance(canary, Mapping)
            or canary.get("time_authority") != "absolute_tick_v1"
        ):
            fail("compiled plan lacks a required absolute-tick technical gate")
        else:
            raw_ids = canary.get("cell_ids")
            raw_total = canary.get("total_ticks")
            raw_ticks = canary.get("checkpoint_ticks")
            if (
                not isinstance(raw_ids, list)
                or not raw_ids
                or any(not isinstance(value, str) or not value for value in raw_ids)
                or len(set(raw_ids)) != len(raw_ids)
                or raw_ids != gate.get("cell_ids")
            ):
                fail(
                    "compiled absolute-tick technical-gate cell identities are invalid"
                )
            else:
                expected_ids = tuple(raw_ids)
            if (
                type(raw_total) is not int
                or raw_total < 1
                or not isinstance(raw_ticks, list)
                or any(
                    type(value) is not int or value < 1 or value > raw_total
                    for value in raw_ticks
                )
                or raw_ticks != sorted(set(raw_ticks))
            ):
                fail(
                    "compiled absolute-tick technical-gate checkpoint contract is invalid"
                )
            else:
                total_ticks = raw_total
                expected_ticks = tuple(sorted({*raw_ticks, raw_total}))
            canary_batch_id = canary.get("batch_id")
            restore_batch_id = canary.get("restore_batch_id")
            if any(
                not isinstance(value, str) or not value
                for value in (canary_batch_id, restore_batch_id)
            ):
                fail(
                    "compiled absolute-tick technical-gate batch identities are invalid"
                )

        requested = tuple(gate_cell_ids)
        checks["selected_cells_exact"] = (
            bool(expected_ids) and requested == expected_ids
        )
        if not checks["selected_cells_exact"]:
            fail(
                "requested gate cells differ from the compiled ordered cell identities"
            )
        try:
            cells = _cell_lookup(payload)
        except CampaignProviderError as exc:
            cells = {}
            fail(str(exc))
        checks["selected_cells_present_once"] = len(expected_ids) > 0 and len(
            {cell_id for cell_id in expected_ids if cell_id in cells}
        ) == len(expected_ids)
        if not checks["selected_cells_present_once"]:
            fail("one or more compiled gate cells are absent from the plan")

        manifests = [row for row in batch_manifests if isinstance(row, Mapping)]
        by_role: dict[str, Mapping[str, Any]] = {}
        for manifest in manifests:
            role = manifest.get("gate_role")
            if not isinstance(role, str) or role in by_role:
                fail("technical-gate manifests contain an invalid or duplicate role")
                continue
            by_role[role] = manifest
        expected_roles = {"canary_execute", "fresh_owner_restore"}
        checks["manifest_roles_exact"] = (
            len(manifests) == len(batch_manifests) == 2
            and set(by_role) == expected_roles
        )
        if not checks["manifest_roles_exact"]:
            fail("technical gate requires exactly canary and fresh-owner manifests")

        artifact_rows = (
            _artifact_rows(payload)
            if isinstance(payload.get("artifacts"), Mapping)
            else {}
        )
        artifact_sha256s = {
            name: row.get("sha256")
            for name, row in sorted(artifact_rows.items())
            if isinstance(name, str) and isinstance(row, Mapping)
        }
        artifact_binding_valid = bool(artifact_sha256s) and all(
            _valid_sha256(value) for value in artifact_sha256s.values()
        )
        if not artifact_binding_valid:
            fail("compiled plan artifact digests are absent or invalid")

        plan_binding_fields = {
            "campaign_id": payload.get("campaign_id"),
            "plan_sha256": payload.get("plan_sha256"),
            "profile_id": payload.get("resolved_profile"),
            "provider_id": self.provider_id,
        }
        if (
            not all(
                isinstance(value, str) and value
                for value in plan_binding_fields.values()
            )
            or not _valid_sha256(plan_binding_fields["plan_sha256"])
            or not isinstance(payload.get("qualification_key_sha256"), str)
            or not payload.get("qualification_key_sha256")
        ):
            fail("compiled plan identity is incomplete")

        route_sha256: str | None = None
        contract_payload = {
            "execution_mode": "pre_primary_canary",
            "cell_ids": list(expected_ids),
            "batch_id": canary_batch_id,
            "restore_batch_id": restore_batch_id,
            "time_authority": "absolute_tick_v1",
            "total_ticks": total_ticks,
            "checkpoint_ticks": list(expected_ticks),
            **(
                {"profile": gate.get("profile")}
                if isinstance(gate, Mapping) and gate.get("profile") is not None
                else {}
            ),
        }
        manifests_bound = bool(by_role)
        for role in sorted(expected_roles):
            manifest = by_role.get(role)
            if not isinstance(manifest, Mapping):
                manifests_bound = False
                continue
            candidate_route = manifest.get("execution_route_sha256")
            if not _valid_sha256(candidate_route):
                fail(f"{role} manifest lacks a valid execution-route binding")
                manifests_bound = False
            elif route_sha256 is None:
                route_sha256 = candidate_route
            elif route_sha256 != candidate_route:
                fail("technical-gate manifests bind different execution routes")
                manifests_bound = False
            expected_binding = {
                **plan_binding_fields,
                "execution_route_sha256": candidate_route,
                "contract": contract_payload,
            }
            expected_gate_hash = _canonical_sha256(expected_binding)
            expected_batch_id = (
                canary_batch_id if role == "canary_execute" else restore_batch_id
            )
            if (
                any(
                    manifest.get(name) != value
                    for name, value in plan_binding_fields.items()
                )
                or manifest.get("gate_contract_sha256") != expected_gate_hash
                or manifest.get("batch_id") != expected_batch_id
                or manifest.get("cell_ids") != list(expected_ids)
            ):
                fail(f"{role} manifest plan or gate-contract binding is invalid")
                manifests_bound = False
        checks["plan_and_gate_binding_exact"] = (
            manifests_bound and route_sha256 is not None and artifact_binding_valid
        )

        def checkpoint_binding(batch_id: str | None) -> dict[str, Any]:
            binding = {
                "campaign_id": payload.get("campaign_id"),
                "plan_sha256": payload.get("plan_sha256"),
                "qualification_key_sha256": payload.get("qualification_key_sha256"),
                "profile_id": self.profile_id,
                "provider_id": self.provider_id,
                "execution_route_sha256": route_sha256,
                "batch_id": batch_id,
                "cell_ids": list(expected_ids),
                "artifact_sha256s": artifact_sha256s,
            }
            from engine.campaigns.run_contract_v1 import checkpoint_contract_binding_v1

            run_contract = checkpoint_contract_binding_v1(payload)
            if run_contract is not None:
                binding["run_contract"] = run_contract
            return binding

        def checkpoint_valid(
            checkpoint: Any,
            *,
            next_tick: int,
            batch_id: str | None,
            label: str,
        ) -> bool:
            if not isinstance(checkpoint, Mapping):
                fail(f"{label} checkpoint evidence is absent")
                return False
            rows = checkpoint.get("cell_bundles")
            digests = checkpoint.get("terminal_cell_digests")
            valid = (
                checkpoint.get("schema_version") == self.checkpoint_schema_version
                and checkpoint.get("execution_route_sha256") == route_sha256
                and _tick_native_checkpoint_coordinate_exact_v1(
                    checkpoint, expected_next_absolute_tick=next_tick
                )
                and checkpoint.get("checkpoint_binding") == checkpoint_binding(batch_id)
                and isinstance(rows, list)
                and [row.get("cell_id") for row in rows if isinstance(row, Mapping)]
                == list(expected_ids)
                and all(
                    isinstance(row.get("path"), str)
                    and bool(row["path"])
                    and _valid_sha256(row.get("source_projection_sha256"))
                    for row in rows
                    if isinstance(row, Mapping)
                )
                and isinstance(digests, Mapping)
                and set(digests) == set(expected_ids)
                and all(_valid_sha256(value) for value in digests.values())
            )
            if not valid:
                fail(f"{label} checkpoint evidence is incomplete or binding-invalid")
            return valid

        canary_manifest = by_role.get("canary_execute")
        restore_manifest = by_role.get("fresh_owner_restore")
        canary_result = (
            canary_manifest.get("result")
            if isinstance(canary_manifest, Mapping)
            else None
        )
        restore_result = (
            restore_manifest.get("result")
            if isinstance(restore_manifest, Mapping)
            else None
        )
        canary_complete = (
            isinstance(canary_manifest, Mapping)
            and canary_manifest.get("status") == "completed"
            and isinstance(canary_result, Mapping)
            and canary_result.get("time_authority") == "absolute_tick_v1"
            and canary_result.get("terminal_next_absolute_tick") == total_ticks
        )
        restore_complete = (
            isinstance(restore_manifest, Mapping)
            and restore_manifest.get("status") == "completed"
            and isinstance(restore_result, Mapping)
            and restore_result.get("time_authority") == "absolute_tick_v1"
            and restore_result.get("terminal_next_absolute_tick") == total_ticks
        )
        checks["canary_execute_completed"] = canary_complete
        checks["fresh_owner_restore_completed"] = restore_complete
        if not canary_complete:
            fail("canary did not complete the compiled absolute terminal tick")
        if not restore_complete:
            fail(
                "fresh-owner restore did not complete the compiled absolute terminal tick"
            )

        catalog = (
            canary_result.get("checkpoint_catalog")
            if isinstance(canary_result, Mapping)
            else None
        )
        catalog_complete = isinstance(catalog, Mapping) and set(catalog) == {
            str(value) for value in expected_ticks
        }
        if isinstance(catalog, Mapping) and catalog_complete:
            catalog_complete = all(
                checkpoint_valid(
                    catalog.get(str(tick)),
                    next_tick=tick,
                    batch_id=canary_batch_id,
                    label=f"canary tick {tick}",
                )
                for tick in expected_ticks
            )
        if not catalog_complete:
            fail("canary lacks exact absolute-tick checkpoint-catalog coverage")
        checks["checkpoint_ticks_exact"] = catalog_complete

        terminal_checkpoint_valid = checkpoint_valid(
            (
                restore_manifest.get("checkpoint")
                if isinstance(restore_manifest, Mapping)
                else None
            ),
            next_tick=total_ticks if total_ticks is not None else -1,
            batch_id=restore_batch_id,
            label="fresh-owner terminal tick",
        )
        checks["fresh_owner_terminal_checkpoint_bound"] = terminal_checkpoint_valid

        def native_no_fallback(result: Any) -> bool:
            if not isinstance(result, Mapping):
                return False
            receipts = result.get("native_execution_receipts")
            return (
                (
                    result.get("execution_device", "").startswith("cuda")
                    if isinstance(result.get("execution_device"), str)
                    else False
                )
                and result.get("fallback_used") is False
                and isinstance(receipts, list)
                and bool(receipts)
                and all(
                    isinstance(row, Mapping)
                    and row.get("explicit_native_no_fallback") is True
                    for row in receipts
                )
            )

        checks["cuda_without_fallback"] = native_no_fallback(
            canary_result
        ) and native_no_fallback(restore_result)
        if not checks["cuda_without_fallback"]:
            fail("technical gate did not prove CUDA execution without fallback")

        canary_terminal = (
            canary_manifest.get("checkpoint")
            if isinstance(canary_manifest, Mapping)
            else None
        )
        restore_terminal = (
            restore_manifest.get("checkpoint")
            if isinstance(restore_manifest, Mapping)
            else None
        )
        source_ticks = tuple(
            tick
            for tick in expected_ticks
            if total_ticks is not None and tick < total_ticks
        )
        source_tick = source_ticks[-1] if source_ticks else total_ticks
        source_checkpoint = (
            catalog.get(str(source_tick))
            if isinstance(catalog, Mapping) and source_tick is not None
            else None
        )
        source_digests = (
            canary_terminal.get("terminal_cell_digests")
            if isinstance(canary_terminal, Mapping)
            else None
        )
        restored_digests = (
            restore_terminal.get("terminal_cell_digests")
            if isinstance(restore_terminal, Mapping)
            else None
        )
        source_checkpoint_digests = (
            source_checkpoint.get("terminal_cell_digests")
            if isinstance(source_checkpoint, Mapping)
            else None
        )
        restore_receipt = (
            restore_result.get("fresh_owner_restore_receipt")
            if isinstance(restore_result, Mapping)
            else None
        )
        reload_rows = (
            restore_receipt.get("reload_receipts")
            if isinstance(restore_receipt, Mapping)
            else None
        )
        checkpoint_rebind = (
            restore_receipt.get("checkpoint_rebind")
            if isinstance(restore_receipt, Mapping)
            else None
        )
        restore_proven = (
            isinstance(restore_receipt, Mapping)
            and restore_receipt.get("schema_version")
            == FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION
            and restore_receipt.get("projection_bundles_reloaded") is True
            and restore_receipt.get("time_authority") == "absolute_tick_v1"
            and restore_receipt.get("source_batch_id") == canary_batch_id
            and restore_receipt.get("restored_batch_id") == restore_batch_id
            and restore_receipt.get("cell_ids") == list(expected_ids)
            and isinstance(source_digests, Mapping)
            and source_digests == restored_digests
            and isinstance(source_checkpoint_digests, Mapping)
            and restore_receipt.get("source_checkpoint_cell_digests")
            == source_checkpoint_digests
            and restore_receipt.get("restored_terminal_cell_digests")
            == restored_digests
            and isinstance(checkpoint_rebind, Mapping)
            and checkpoint_rebind.get("schema_version")
            == FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION
            and checkpoint_rebind.get("source_batch_id") == canary_batch_id
            and checkpoint_rebind.get("restored_batch_id") == restore_batch_id
            and checkpoint_rebind.get("source_checkpoint_binding")
            == (
                source_checkpoint.get("checkpoint_binding")
                if isinstance(source_checkpoint, Mapping)
                else None
            )
            and checkpoint_rebind.get("source_checkpoint_sha256")
            == (
                _canonical_sha256(source_checkpoint)
                if isinstance(source_checkpoint, Mapping)
                else None
            )
            and isinstance(reload_rows, list)
            and [row.get("cell_id") for row in reload_rows if isinstance(row, Mapping)]
            == list(expected_ids)
            and all(
                _valid_sha256(row.get("source_projection_sha256"))
                and row.get("source_projection_sha256")
                == row.get("loaded_projection_sha256")
                and row.get("source_projection_sha256")
                == source_checkpoint_digests.get(row.get("cell_id"))
                for row in reload_rows
                if isinstance(row, Mapping)
            )
        )
        checks["fresh_owner_bundle_reload_and_equality_proven"] = restore_proven
        if not restore_proven:
            fail(
                "fresh-owner bundle reload/equality receipt is absent or binding-invalid"
            )

        return {
            "schema_version": TECHNICAL_GATE_EVIDENCE_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "route_family_profile": "tick_native_absolute_tick_v1",
            "gate_cell_ids": list(expected_ids),
            "checkpoint_ticks": list(expected_ticks),
            "terminal_next_absolute_tick": total_ticks,
            "checks": checks,
            "reasons": reasons,
            "passed": not reasons and all(value is True for value in checks.values()),
        }

    def evaluate_technical_gate(
        self,
        plan: Mapping[str, Any],
        gate_cell_ids: Sequence[str],
        batch_manifests: Sequence[Mapping[str, Any]],
        run_root: Path,
    ) -> dict[str, Any]:
        """Evaluate only evidence that proves the compiled gate contract.

        The evaluator deliberately does not infer G0 checkpoint publication from
        a source projection, nor fresh-owner restoration from a terminal
        fast-path flag.  Both require explicit, bound evidence.  This lets the
        orchestration surface evolve without weakening the certification
        boundary or embedding a campaign's batch size, generation count, matrix,
        seed set, or checkpoint schedule here.
        """

        payload = _payload(plan)
        execution = payload.get("execution")
        if (
            isinstance(execution, Mapping)
            and execution.get("execution_route_identity")
            == TICK_NATIVE_EXECUTION_ROUTE_V1
        ):
            return self._evaluate_tick_native_technical_gate_v1(
                payload,
                gate_cell_ids,
                batch_manifests,
                run_root,
            )
        reasons: list[str] = []
        checks: dict[str, Any] = {}

        def fail(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        execution = payload.get("execution")
        gate = (
            execution.get("technical_gate") if isinstance(execution, Mapping) else None
        )
        canary = gate.get("canary") if isinstance(gate, Mapping) else None
        expected_ids: tuple[str, ...] = ()
        expected_milestones: tuple[int, ...] = ()
        terminal_generation: int | None = None
        canary_batch_id: str | None = None
        restore_batch_id: str | None = None
        if (
            not isinstance(gate, Mapping)
            or gate.get("required") is not True
            or gate.get("execution_mode") != "pre_primary_canary"
            or not isinstance(canary, Mapping)
        ):
            fail("compiled plan lacks a required pre-primary technical gate")
        else:
            raw_ids = canary.get("cell_ids")
            raw_milestones = canary.get("checkpoint_milestones")
            raw_generation = canary.get("generations")
            if (
                not isinstance(raw_ids, list)
                or not raw_ids
                or any(not isinstance(value, str) or not value for value in raw_ids)
                or len(set(raw_ids)) != len(raw_ids)
                or raw_ids != gate.get("cell_ids")
            ):
                fail("compiled technical-gate cell identities are invalid")
            else:
                expected_ids = tuple(raw_ids)
            if (
                type(raw_generation) is not int
                or raw_generation < 1
                or not isinstance(raw_milestones, list)
                or not raw_milestones
                or any(
                    type(value) is not int or not 0 <= value <= raw_generation
                    for value in raw_milestones
                )
                or raw_milestones != sorted(set(raw_milestones))
                or raw_milestones[0] != 0
                or raw_milestones[-1] != raw_generation
            ):
                fail("compiled technical-gate checkpoint contract is invalid")
            else:
                terminal_generation = raw_generation
                expected_milestones = tuple(raw_milestones)
            canary_batch_id = canary.get("batch_id")
            restore_batch_id = canary.get("restore_batch_id")
            if any(
                not isinstance(value, str) or not value
                for value in (canary_batch_id, restore_batch_id)
            ):
                fail("compiled technical-gate batch identities are invalid")

        requested = tuple(gate_cell_ids)
        checks["selected_cells_exact"] = (
            bool(expected_ids) and requested == expected_ids
        )
        if not checks["selected_cells_exact"]:
            fail(
                "requested gate cells differ from the compiled ordered cell identities"
            )

        try:
            cells = _cell_lookup(payload)
        except CampaignProviderError as exc:
            cells = {}
            fail(str(exc))
        selected_cells = {
            cell_id: cells[cell_id] for cell_id in expected_ids if cell_id in cells
        }
        checks["selected_cells_present_once"] = len(selected_cells) == len(expected_ids)
        if not checks["selected_cells_present_once"]:
            fail("one or more compiled gate cells are absent from the plan")

        gv2_wdl_dimension_names = (
            "ecology_seed",
            "task_seed",
            "founder_profile",
            "mutation_rate_q10",
            "mutation_step_q10",
        )
        cross_path_dimension_names = (
            "inheritance_mode",
            "founder_pattern",
            "reproduction_path",
            "paired_seed",
        )

        def cell_dimensions(cell_id: str) -> Mapping[str, Any]:
            cell = selected_cells.get(cell_id)
            dimensions = cell.get("dimensions") if isinstance(cell, Mapping) else None
            return dimensions if isinstance(dimensions, Mapping) else {}

        def cell_parameters(cell_id: str) -> Mapping[str, Any]:
            cell = selected_cells.get(cell_id)
            parameters = cell.get("parameters") if isinstance(cell, Mapping) else None
            return parameters if isinstance(parameters, Mapping) else {}

        def is_genome_v2_wdl_gate() -> bool:
            if not expected_ids or len(selected_cells) != len(expected_ids):
                return False
            for cell_id in expected_ids:
                dimensions = cell_dimensions(cell_id)
                parameters = cell_parameters(cell_id)
                if (
                    any(name not in dimensions for name in gv2_wdl_dimension_names)
                    or any(name in dimensions for name in cross_path_dimension_names)
                    or parameters.get("genome_upgrade_v2.enable") is not True
                    or parameters.get("production_lifecycle.ordinary_births_enabled")
                    is not True
                    or parameters.get("genome_upgrade_v2.founder_profile")
                    != dimensions.get("founder_profile")
                    or parameters.get("genome_upgrade_v2.mutation_rate_q10")
                    != dimensions.get("mutation_rate_q10")
                    or parameters.get("genome_upgrade_v2.mutation_step_q10")
                    != dimensions.get("mutation_step_q10")
                ):
                    return False
            return True

        explicit_gate_profile = (
            gate.get("profile") if isinstance(gate, Mapping) else None
        )
        gv2_wdl_profile = "genome_v2_wdl_fixed_mutation_persistence_v1"
        cross_path_profile = "inheritance_cross_path_recomb_v1"
        if explicit_gate_profile == gv2_wdl_profile:
            route_family_profile = gv2_wdl_profile
            if not is_genome_v2_wdl_gate():
                fail(
                    "Genome V2 WDL technical gate profile conflicts with "
                    "compiled mechanics"
                )
        elif explicit_gate_profile == cross_path_profile:
            route_family_profile = cross_path_profile
        elif explicit_gate_profile is None:
            route_family_profile = cross_path_profile
            if is_genome_v2_wdl_gate():
                fail(
                    "Genome V2 WDL technical gate lacks the explicit route "
                    "family profile"
                )
        else:
            route_family_profile = cross_path_profile
            fail("compiled technical-gate route family profile is unsupported")

        manifest_rows = [
            manifest for manifest in batch_manifests if isinstance(manifest, Mapping)
        ]
        by_role: dict[str, Mapping[str, Any]] = {}
        for manifest in manifest_rows:
            role = manifest.get("gate_role")
            if not isinstance(role, str) or role in by_role:
                fail("technical-gate manifests contain an invalid or duplicate role")
                continue
            by_role[role] = manifest
        expected_roles = {"canary_execute", "fresh_owner_restore"}
        checks["manifest_roles_exact"] = (
            len(manifest_rows) == len(batch_manifests) == 2
            and set(by_role) == expected_roles
        )
        if not checks["manifest_roles_exact"]:
            fail("technical gate requires exactly canary and fresh-owner manifests")

        artifact_rows = (
            _artifact_rows(payload)
            if isinstance(payload.get("artifacts"), Mapping)
            else {}
        )
        artifact_sha256s = {
            name: row.get("sha256")
            for name, row in sorted(artifact_rows.items())
            if isinstance(name, str) and isinstance(row, Mapping)
        }
        artifact_binding_valid = bool(artifact_sha256s) and all(
            _valid_sha256(value) for value in artifact_sha256s.values()
        )
        if not artifact_binding_valid:
            fail("compiled plan artifact digests are absent or invalid")

        plan_binding_fields = {
            "campaign_id": payload.get("campaign_id"),
            "plan_sha256": payload.get("plan_sha256"),
            "profile_id": payload.get("resolved_profile"),
            "provider_id": self.provider_id,
        }
        if (
            not all(
                isinstance(value, str) and value
                for value in plan_binding_fields.values()
            )
            or not _valid_sha256(plan_binding_fields["plan_sha256"])
            or not isinstance(payload.get("qualification_key_sha256"), str)
            or not payload.get("qualification_key_sha256")
        ):
            fail("compiled plan identity is incomplete")

        route_sha256: str | None = None
        contract_payload = {
            "execution_mode": "pre_primary_canary",
            "cell_ids": list(expected_ids),
            "batch_id": canary_batch_id,
            "restore_batch_id": restore_batch_id,
            "generations": terminal_generation,
            "checkpoint_milestones": list(expected_milestones),
            **(
                {"profile": explicit_gate_profile}
                if explicit_gate_profile is not None
                else {}
            ),
        }
        manifests_bound = bool(by_role)
        for role in sorted(expected_roles):
            manifest = by_role.get(role)
            if not isinstance(manifest, Mapping):
                manifests_bound = False
                continue
            candidate_route = manifest.get("execution_route_sha256")
            if not _valid_sha256(candidate_route):
                fail(f"{role} manifest lacks a valid execution-route binding")
                manifests_bound = False
            elif route_sha256 is None:
                route_sha256 = candidate_route
            elif route_sha256 != candidate_route:
                fail("technical-gate manifests bind different execution routes")
                manifests_bound = False
            expected_binding = {
                **plan_binding_fields,
                "execution_route_sha256": candidate_route,
                "contract": contract_payload,
            }
            expected_gate_hash = _canonical_sha256(expected_binding)
            expected_batch_id = (
                canary_batch_id if role == "canary_execute" else restore_batch_id
            )
            if (
                any(
                    manifest.get(name) != value
                    for name, value in plan_binding_fields.items()
                )
                or manifest.get("gate_contract_sha256") != expected_gate_hash
                or manifest.get("batch_id") != expected_batch_id
                or manifest.get("cell_ids") != list(expected_ids)
            ):
                fail(f"{role} manifest plan or gate-contract binding is invalid")
                manifests_bound = False
        checks["plan_and_gate_binding_exact"] = (
            manifests_bound and route_sha256 is not None and artifact_binding_valid
        )

        def checkpoint_binding(
            batch_id: str | None,
        ) -> dict[str, Any]:
            return {
                "campaign_id": payload.get("campaign_id"),
                "plan_sha256": payload.get("plan_sha256"),
                "qualification_key_sha256": payload.get("qualification_key_sha256"),
                "profile_id": self.profile_id,
                "provider_id": self.provider_id,
                "execution_route_sha256": route_sha256,
                "batch_id": batch_id,
                "cell_ids": list(expected_ids),
                "artifact_sha256s": artifact_sha256s,
            }

        def checkpoint_valid(
            checkpoint: Any,
            *,
            generation: int,
            batch_id: str | None,
            label: str,
        ) -> bool:
            if not isinstance(checkpoint, Mapping):
                fail(f"{label} checkpoint evidence is absent")
                return False
            rows = checkpoint.get("cell_bundles")
            digests = checkpoint.get("terminal_cell_digests")
            valid = (
                checkpoint.get("schema_version") == self.checkpoint_schema_version
                and checkpoint.get("execution_route_sha256") == route_sha256
                and checkpoint.get("generation") == generation
                and checkpoint.get("checkpoint_binding") == checkpoint_binding(batch_id)
                and isinstance(rows, list)
                and [row.get("cell_id") for row in rows if isinstance(row, Mapping)]
                == list(expected_ids)
                and all(
                    isinstance(row.get("path"), str)
                    and bool(row["path"])
                    and _valid_sha256(row.get("source_projection_sha256"))
                    for row in rows
                    if isinstance(row, Mapping)
                )
                and isinstance(digests, Mapping)
                and set(digests) == set(expected_ids)
                and all(_valid_sha256(value) for value in digests.values())
            )
            if not valid:
                fail(f"{label} checkpoint evidence is incomplete or binding-invalid")
            return valid

        canary_manifest = by_role.get("canary_execute")
        restore_manifest = by_role.get("fresh_owner_restore")
        for role, manifest in (
            ("canary_execute", canary_manifest),
            ("fresh_owner_restore", restore_manifest),
        ):
            if not isinstance(manifest, Mapping):
                continue
            result = manifest.get("result")
            complete = (
                manifest.get("status") == "completed"
                and isinstance(result, Mapping)
                and result.get("terminal_generation") == terminal_generation
            )
            checks[f"{role}_completed"] = complete
            if not complete:
                fail(f"{role} did not complete the compiled terminal generation")

        canary_result = (
            canary_manifest.get("result")
            if isinstance(canary_manifest, Mapping)
            else None
        )
        restore_result = (
            restore_manifest.get("result")
            if isinstance(restore_manifest, Mapping)
            else None
        )

        catalog = (
            canary_result.get("checkpoint_catalog")
            if isinstance(canary_result, Mapping)
            else None
        )
        catalog_keys = (
            {str(value) for value in expected_milestones}
            if expected_milestones
            else set()
        )
        checkpoints_complete = (
            isinstance(catalog, Mapping) and set(catalog) == catalog_keys
        )
        if not checkpoints_complete:
            missing = sorted(
                catalog_keys - (set(catalog) if isinstance(catalog, Mapping) else set())
            )
            fail(
                "canary lacks exact checkpoint-catalog coverage"
                + (
                    f": {', '.join('G' + value for value in missing)}"
                    if missing
                    else ""
                )
            )
        if isinstance(catalog, Mapping):
            checkpoints_complete = (
                all(
                    checkpoint_valid(
                        catalog.get(str(generation)),
                        generation=generation,
                        batch_id=canary_batch_id,
                        label=f"canary G{generation}",
                    )
                    for generation in expected_milestones
                )
                and checkpoints_complete
            )
        checks["checkpoint_milestones_exact"] = checkpoints_complete
        if (
            expected_milestones
            and expected_milestones[0] == 0
            and (not isinstance(catalog, Mapping) or "0" not in catalog)
        ):
            fail(
                "G0 durable checkpoint proof is absent; initial telemetry is not "
                "checkpoint evidence"
            )

        terminal_checkpoint_valid = False
        if isinstance(restore_manifest, Mapping) and terminal_generation is not None:
            terminal_checkpoint_valid = checkpoint_valid(
                restore_manifest.get("checkpoint"),
                generation=terminal_generation,
                batch_id=restore_batch_id,
                label="fresh-owner terminal",
            )
        checks["fresh_owner_terminal_checkpoint_bound"] = terminal_checkpoint_valid

        def metric_rows(manifest: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
            raw = (
                manifest.get("metrics_rows") if isinstance(manifest, Mapping) else None
            )
            return (
                [row for row in raw if isinstance(row, Mapping)]
                if isinstance(raw, list)
                else []
            )

        canary_rows = metric_rows(canary_manifest)
        restore_rows = metric_rows(restore_manifest)
        expected_generations = (
            set(range(terminal_generation + 1))
            if terminal_generation is not None
            else set()
        )

        if route_family_profile == "genome_v2_wdl_fixed_mutation_persistence_v1":
            expected_transition_generations = (
                set(range(terminal_generation))
                if terminal_generation is not None
                else set()
            )

            def gv2_rows_by_cell(
                rows: Sequence[Mapping[str, Any]], *, role: str
            ) -> dict[str, Mapping[str, Any]]:
                indexed: dict[str, Mapping[str, Any]] = {}
                malformed = False
                for row in rows:
                    cell_id = row.get("cell_id")
                    generation_rows = row.get("generation_metrics")
                    genome_v2 = row.get("genome_v2_attribution")
                    metrics = row.get("metrics")
                    generations = (
                        {
                            item.get("generation")
                            for item in generation_rows
                            if isinstance(item, Mapping)
                        }
                        if isinstance(generation_rows, list)
                        else set()
                    )
                    if (
                        row.get("schema_version")
                        != "eiel_genome_v2_e1_native_cell_metrics_v1"
                        or cell_id not in expected_ids
                        or cell_id in indexed
                        or not isinstance(generation_rows, list)
                        or generations != expected_transition_generations
                        or any(
                            not isinstance(item, Mapping)
                            or type(item.get("generation")) is not int
                            or type(item.get("alive")) is not int
                            or type(item.get("successful_births")) is not int
                            or item["alive"] < 0
                            or item["successful_births"] < 0
                            for item in generation_rows
                        )
                        or not isinstance(metrics, Mapping)
                        or type(metrics.get("population.births")) is not int
                        or metrics["population.births"] < 0
                        or not isinstance(genome_v2, Mapping)
                        or genome_v2.get("schema_version")
                        != "genome_v2_e1_metric_attribution_v1"
                        or genome_v2.get("cell_id") != cell_id
                        or genome_v2.get("generation") != terminal_generation
                    ):
                        malformed = True
                        continue
                    indexed[str(cell_id)] = row
                if malformed:
                    fail(
                        f"{role} telemetry contains unknown, duplicate, "
                        "or malformed rows"
                    )
                return indexed

            canary_gv2_index = gv2_rows_by_cell(canary_rows, role="canary")
            restore_gv2_index = gv2_rows_by_cell(restore_rows, role="fresh-owner")
            telemetry_complete = (
                bool(expected_ids)
                and set(canary_gv2_index) == set(expected_ids)
                and set(restore_gv2_index) == set(expected_ids)
            )
            checks["telemetry_generation_coverage_exact"] = telemetry_complete
            if not telemetry_complete:
                fail(
                    "per-cell Genome V2 telemetry does not cover every "
                    "compiled gate cell and transition generation"
                )

            identity_exact = True
            mutation_contract = True
            runtime_identity_exact = True

            def runner_receipts_by_cell(
                result: Mapping[str, Any] | None, *, role: str
            ) -> dict[str, Mapping[str, Any]] | None:
                receipts = (
                    result.get("runner_receipts")
                    if isinstance(result, Mapping)
                    else None
                )
                indexed: dict[str, Mapping[str, Any]] = {}
                malformed = False
                if receipts is None or receipts == []:
                    return None
                if not isinstance(receipts, list):
                    fail(f"{role} runner receipts are malformed")
                    return {}
                for row in receipts:
                    cell_id = row.get("cell_id") if isinstance(row, Mapping) else None
                    if cell_id not in expected_ids:
                        malformed = True
                        continue
                    indexed[str(cell_id)] = row
                if malformed or set(indexed) != set(expected_ids):
                    fail(
                        f"{role} runner receipts do not cover exact Genome V2 "
                        "WDL gate cells"
                    )
                return indexed

            canary_runner_receipts = runner_receipts_by_cell(
                canary_result, role="canary"
            )
            restore_runner_receipts = runner_receipts_by_cell(
                restore_result, role="fresh-owner"
            )

            for cell_id in expected_ids:
                dimensions = cell_dimensions(cell_id)
                parameters = cell_parameters(cell_id)
                if (
                    any(
                        dimensions.get(name) is None for name in gv2_wdl_dimension_names
                    )
                    or parameters.get("genome_upgrade_v2.founder_profile")
                    != dimensions.get("founder_profile")
                    or parameters.get("genome_upgrade_v2.mutation_rate_q10")
                    != dimensions.get("mutation_rate_q10")
                    or parameters.get("genome_upgrade_v2.mutation_step_q10")
                    != dimensions.get("mutation_step_q10")
                    or parameters.get("genome_upgrade_v2.enable") is not True
                    or parameters.get("production_lifecycle.ordinary_births_enabled")
                    is not True
                ):
                    identity_exact = False
                    mutation_contract = False
                    fail(
                        "compiled Genome V2 WDL identity or mutation arm is "
                        f"invalid: {cell_id}"
                    )
                    continue
                rate = dimensions.get("mutation_rate_q10")
                step = dimensions.get("mutation_step_q10")
                if (
                    type(rate) is not int
                    or type(step) is not int
                    or rate < 0
                    or step < 0
                    or (rate == 0) != (step == 0)
                ):
                    mutation_contract = False
                    fail(f"Genome V2 mutation arm is incoherent: {cell_id}")
                    continue
                ecology_seed = dimensions.get("ecology_seed")
                task_seed = dimensions.get("task_seed")
                receipt_maps = (
                    ("canary", canary_runner_receipts),
                    ("fresh-owner", restore_runner_receipts),
                )
                for role, receipts_by_cell in receipt_maps:
                    if receipts_by_cell is None:
                        continue
                    receipt = receipts_by_cell.get(cell_id)
                    if not isinstance(receipt, Mapping):
                        runtime_identity_exact = False
                        mutation_contract = False
                        fail(
                            f"{role} Genome V2 WDL runtime identity is absent "
                            f"for {cell_id}"
                        )
                        continue
                    genome = receipt.get("genome")
                    lifecycle = receipt.get("production_lifecycle")
                    seeds = receipt.get("seed_ownership")
                    owners = (
                        seeds.get("named_owners")
                        if isinstance(seeds, Mapping)
                        else None
                    )
                    if (
                        not isinstance(genome, Mapping)
                        or genome.get("enabled") is not True
                        or genome.get("profile") != "genome_upgrade_v2_fixed_q10_v1"
                        or genome.get("founder_profile_id")
                        != dimensions.get("founder_profile")
                        or genome.get("mutation_rate_q10") != rate
                        or genome.get("mutation_step_q10") != step
                        or not isinstance(lifecycle, Mapping)
                        or lifecycle.get("enabled") is not True
                        or lifecycle.get("ordinary_births_enabled") is not True
                        or lifecycle.get("genome_v2_causal_capture_enabled") is not True
                        or lifecycle.get("genome_v2_mutation_enabled") is not (rate > 0)
                        or lifecycle.get("diploid_mutation_enabled") is not (rate > 0)
                        or lifecycle.get("legacy_scalar_mutation_off")
                        is not (rate == 0)
                        or lifecycle.get("genome_v2_mutation_rate_q10") != rate
                        or lifecycle.get("genome_v2_mutation_step_q10") != step
                        or not isinstance(owners, Mapping)
                        or owners.get("ecology") != ecology_seed
                        or owners.get("task") != task_seed
                    ):
                        runtime_identity_exact = False
                        mutation_contract = False
                        fail(
                            f"{role} Genome V2 WDL runtime identity differs "
                            f"for {cell_id}"
                        )
            identity_exact = identity_exact and runtime_identity_exact
            checks["path_and_mode_identity_exact"] = identity_exact
            checks["mutation_zero"] = mutation_contract
            exposure_complete = False
        else:

            def rows_by_cell(
                rows: Sequence[Mapping[str, Any]], *, role: str
            ) -> dict[str, dict[int, Mapping[str, Any]]]:
                indexed = {cell_id: {} for cell_id in expected_ids}
                malformed = False
                for row in rows:
                    cell_id = row.get("cell_id")
                    generation = row.get("generation")
                    if (
                        cell_id not in indexed
                        or type(generation) is not int
                        or generation in indexed[cell_id]
                    ):
                        malformed = True
                        continue
                    indexed[cell_id][generation] = row
                if malformed:
                    fail(
                        f"{role} telemetry contains unknown, duplicate, "
                        "or malformed rows"
                    )
                return indexed

            canary_index = rows_by_cell(canary_rows, role="canary")
            restore_index = rows_by_cell(restore_rows, role="fresh-owner")
            telemetry_complete = bool(expected_generations) and all(
                set(canary_index.get(cell_id, {})) == expected_generations
                and set(restore_index.get(cell_id, {})) == expected_generations
                for cell_id in expected_ids
            )
            checks["telemetry_generation_coverage_exact"] = telemetry_complete
            if not telemetry_complete:
                fail("per-cell telemetry does not cover every compiled gate generation")

            identity_exact = True
            mutation_zero = True
            exposure_by_cell = {cell_id: 0 for cell_id in expected_ids}
            for cell_id in expected_ids:
                dimensions = cell_dimensions(cell_id)
                parameters = cell_parameters(cell_id)
                expected_identity = {
                    name: dimensions.get(name) for name in cross_path_dimension_names
                }
                if (
                    len(expected_identity) != 4
                    or any(value is None for value in expected_identity.values())
                    or parameters.get("inheritance_cross_path.mutation_enabled")
                    is not False
                ):
                    identity_exact = False
                    mutation_zero = False
                    fail(
                        "compiled gate cell identity or mutation arm is "
                        f"invalid: {cell_id}"
                    )
                    continue
                for role, index in (
                    ("canary", canary_index),
                    ("fresh-owner", restore_index),
                ):
                    for row in index.get(cell_id, {}).values():
                        if row.get("identity") != expected_identity:
                            identity_exact = False
                            fail(f"{role} telemetry identity differs for {cell_id}")
                        mutation = row.get("mutation_events")
                        if mutation != 0:
                            mutation_zero = False
                            fail(f"{role} telemetry records mutation for {cell_id}")
                        reproduction = row.get("reproduction")
                        if not isinstance(reproduction, Mapping):
                            fail(
                                f"{role} reproduction telemetry is absent for {cell_id}"
                            )
                            continue
                        for field in (
                            "birth_opportunities",
                            "successful_births",
                            "failed_birth_attempts",
                            "refill_child_count",
                            "ordinary_birth_count",
                        ):
                            if type(reproduction.get(field)) is not int:
                                fail(
                                    f"{role} reproduction telemetry field is absent: "
                                    f"{cell_id}/{field}"
                                )
                        if role == "canary":
                            opportunities = reproduction.get("birth_opportunities")
                            if type(opportunities) is int and opportunities > 0:
                                exposure_by_cell[cell_id] += opportunities
            checks["path_and_mode_identity_exact"] = identity_exact
            checks["mutation_zero"] = mutation_zero
            exposure_complete = bool(exposure_by_cell) and all(
                value > 0 for value in exposure_by_cell.values()
            )
        checks["real_reproduction_exposure"] = exposure_complete
        if not exposure_complete and route_family_profile != (
            "genome_v2_wdl_fixed_mutation_persistence_v1"
        ):
            fail("one or more gate cells lack measured reproduction exposure")

        intermediate_milestones = tuple(
            generation
            for generation in expected_milestones
            if terminal_generation is not None and 0 < generation < terminal_generation
        )
        restore_start_generation = (
            intermediate_milestones[-1] if intermediate_milestones else None
        )

        def explicit_cuda_without_fallback(
            result: Mapping[str, Any] | None, *, role: str
        ) -> bool:
            if not isinstance(result, Mapping):
                return False
            receipts = result.get("native_execution_receipts")
            if route_family_profile == "genome_v2_wdl_fixed_mutation_persistence_v1":
                if (
                    not isinstance(receipts, list)
                    or not receipts
                    or terminal_generation is None
                ):
                    return False
                expected_start = (
                    0 if role == "canary_execute" else restore_start_generation
                )
                if type(expected_start) is not int:
                    return False
                previous_end = expected_start
                for row in receipts:
                    if (
                        not isinstance(row, Mapping)
                        or row.get("start_generation") != previous_end
                        or type(row.get("end_generation")) is not int
                        or row["end_generation"] <= previous_end
                        or isinstance(row.get("execution_device"), str) is False
                        or not row["execution_device"].startswith("cuda")
                        or row.get("native_owner") != "native_extension"
                        or row.get("hidden_fallback") is not False
                        or row.get("runtime_jit") is not False
                        or row.get("explicit_native_no_fallback") is not True
                    ):
                        return False
                    previous_end = int(row["end_generation"])
                return previous_end == terminal_generation
            if (
                isinstance(result.get("execution_device"), str)
                and result["execution_device"].startswith("cuda")
                and result.get("fallback_used") is False
            ):
                return True
            return (
                isinstance(receipts, list)
                and bool(receipts)
                and all(
                    isinstance(row, Mapping)
                    and type(row.get("start_generation")) is int
                    and type(row.get("end_generation")) is int
                    and row["end_generation"] > row["start_generation"]
                    and isinstance(row.get("execution_device"), str)
                    and row["execution_device"].startswith("cuda")
                    and row.get("native_owner") == "native_extension"
                    and row.get("hidden_fallback") is False
                    and row.get("runtime_jit") is False
                    and row.get("explicit_native_no_fallback") is True
                    for row in receipts
                )
            )

        no_fallback = True
        for role, result in (
            ("canary_execute", canary_result),
            ("fresh_owner_restore", restore_result),
        ):
            if not explicit_cuda_without_fallback(result, role=role):
                no_fallback = False
                fail(f"{role} lacks explicit CUDA/no-fallback evidence")
        checks["cuda_without_fallback"] = no_fallback

        if route_family_profile == "genome_v2_wdl_fixed_mutation_persistence_v1":

            def provider_native_exposure_receipt_valid(
                result: Mapping[str, Any] | None,
                *,
                batch_id: str | None,
            ) -> bool:
                receipt = (
                    result.get("provider_native_exposure_receipt")
                    if isinstance(result, Mapping)
                    else None
                )
                native_receipts = (
                    result.get("native_execution_receipts")
                    if isinstance(result, Mapping)
                    else None
                )
                cell_receipts = (
                    receipt.get("cell_receipts")
                    if isinstance(receipt, Mapping)
                    else None
                )
                segment_indices = (
                    receipt.get("segment_indices")
                    if isinstance(receipt, Mapping)
                    else None
                )
                totals = receipt.get("totals") if isinstance(receipt, Mapping) else None
                payload = (
                    {
                        name: value
                        for name, value in receipt.items()
                        if name != "receipt_sha256"
                    }
                    if isinstance(receipt, Mapping)
                    else None
                )
                valid = (
                    isinstance(receipt, Mapping)
                    and receipt.get("schema_version")
                    == PROVIDER_NATIVE_EXPOSURE_RECEIPT_SCHEMA_VERSION
                    and receipt.get("batch_id") == batch_id
                    and receipt.get("profile_id") == self.profile_id
                    and receipt.get("execution_route_sha256") == route_sha256
                    and receipt.get("cell_ids") == list(expected_ids)
                    and isinstance(native_receipts, list)
                    and isinstance(segment_indices, list)
                    and len(segment_indices) == len(native_receipts)
                    and all(type(value) is int for value in segment_indices)
                    and len(set(segment_indices)) == len(segment_indices)
                    and receipt.get("native_execution_receipts_sha256")
                    == canonical_sha256(native_receipts)
                    and isinstance(cell_receipts, Mapping)
                    and set(cell_receipts) == set(expected_ids)
                    and isinstance(totals, Mapping)
                    and totals.get("decision_bearing_drop_count") == 0
                    and type(totals.get("attempt_count")) is int
                    and totals["attempt_count"] > 0
                    and type(totals.get("success_count")) is int
                    and totals["success_count"] > 0
                    and type(totals.get("failure_count")) is int
                    and totals["failure_count"] >= 0
                    and type(totals.get("eligible_snapshot_count")) is int
                    and totals["eligible_snapshot_count"] >= 0
                    and all(
                        isinstance(row, Mapping)
                        and row.get("cell_id") == cell_id
                        and type(row.get("segment_count")) is int
                        and row["segment_count"] > 0
                        and type(row.get("attempt_count")) is int
                        and row["attempt_count"] > 0
                        and type(row.get("success_count")) is int
                        and row["success_count"] >= 0
                        and type(row.get("failure_count")) is int
                        and row["failure_count"] >= 0
                        and _valid_sha256(row.get("chain_sha256_after_last_segment"))
                        for cell_id, row in cell_receipts.items()
                    )
                    and isinstance(payload, Mapping)
                    and receipt.get("receipt_sha256") == canonical_sha256(payload)
                )
                return valid

            def _projection_for_bundle(
                bundle: Mapping[str, Any],
            ) -> Mapping[str, Any] | None:
                inline = bundle.get("genome_v2_checkpoint_projection")
                if isinstance(inline, Mapping):
                    return inline
                path_value = bundle.get("path")
                if not isinstance(path_value, str) or not path_value:
                    return None
                try:
                    projection = self._read_projection_bundle(path_value)
                except (OSError, ValueError, CampaignProviderError):
                    return None
                genome_projection = (
                    projection.get("genome_upgrade_v2_checkpoint_projection")
                    if isinstance(projection, Mapping)
                    else None
                )
                if not isinstance(genome_projection, Mapping):
                    nested = (
                        projection.get("projection")
                        if isinstance(projection, Mapping)
                        else None
                    )
                    genome_projection = (
                        nested.get("genome_upgrade_v2_checkpoint_projection")
                        if isinstance(nested, Mapping)
                        else None
                    )
                return (
                    genome_projection
                    if isinstance(genome_projection, Mapping)
                    else None
                )

            def causal_projection_exposure_valid(
                manifest: Mapping[str, Any] | None,
                indexed_rows: Mapping[str, Mapping[str, Any]],
                *,
                role: str,
            ) -> bool:
                checkpoint = (
                    manifest.get("checkpoint")
                    if isinstance(manifest, Mapping)
                    else None
                )
                bundles = (
                    checkpoint.get("cell_bundles")
                    if isinstance(checkpoint, Mapping)
                    else None
                )
                if not isinstance(bundles, list):
                    fail(f"{role} Genome V2 checkpoint bundles are absent")
                    return False
                indexed: dict[str, Mapping[str, Any]] = {}
                malformed = False
                for bundle in bundles:
                    cell_id = (
                        bundle.get("cell_id") if isinstance(bundle, Mapping) else None
                    )
                    row = (
                        indexed_rows.get(str(cell_id))
                        if isinstance(cell_id, str)
                        else None
                    )
                    genome_projection = (
                        _projection_for_bundle(bundle)
                        if isinstance(bundle, Mapping)
                        else None
                    )
                    identity = (
                        genome_projection.get("identity")
                        if isinstance(genome_projection, Mapping)
                        else None
                    )
                    causal = (
                        genome_projection.get("causal_transaction_projection")
                        if isinstance(genome_projection, Mapping)
                        else None
                    )
                    metrics = row.get("metrics") if isinstance(row, Mapping) else None
                    metric_births = (
                        metrics.get("population.births")
                        if isinstance(metrics, Mapping)
                        else None
                    )
                    captured = (
                        causal.get("captured_counts")
                        if isinstance(causal, Mapping)
                        else None
                    )
                    dropped = (
                        causal.get("dropped_counts")
                        if isinstance(causal, Mapping)
                        else None
                    )
                    records = (
                        causal.get("record_counts")
                        if isinstance(causal, Mapping)
                        else None
                    )
                    authority = (
                        causal.get("checkpoint_authority_binding")
                        if isinstance(causal, Mapping)
                        else None
                    )
                    event_receipts = (
                        causal.get("event_receipts")
                        if isinstance(causal, Mapping)
                        else None
                    )
                    captured_births = (
                        captured.get("birth") if isinstance(captured, Mapping) else None
                    )
                    births = captured_births
                    if (
                        cell_id not in expected_ids
                        or cell_id in indexed
                        or not isinstance(identity, Mapping)
                        or identity.get("cell_id") != cell_id
                        or identity.get("generation") != terminal_generation
                        or type(births) is not int
                        or births <= 0
                        or not isinstance(causal, Mapping)
                        or causal.get("capture_status") != "capture_enabled_joined"
                        or causal.get("evaluable") is not True
                        or not isinstance(captured, Mapping)
                        or (
                            type(metric_births) is int
                            and type(births) is int
                            and metric_births < births
                        )
                        or captured.get("birth") != births
                        or captured.get("total") != births
                        or not isinstance(dropped, Mapping)
                        or dropped.get("birth") != 0
                        or dropped.get("total") != 0
                        or not isinstance(records, Mapping)
                        or records.get("event") != births
                        or records.get("joined") != births
                        or records.get("terminal") != births
                        or not isinstance(event_receipts, list)
                        or len(event_receipts) != births
                        or any(
                            not isinstance(event, Mapping)
                            or event.get("schema_version")
                            != "genome_v2_causal_event_receipt_v2"
                            or not _valid_sha256(event.get("receipt_sha256"))
                            for event in event_receipts
                        )
                        or not isinstance(authority, Mapping)
                        or authority.get("provider_id") != self.provider_id
                        or authority.get("runtime_profile_id") != self.profile_id
                        or authority.get("execution_route_sha256") != route_sha256
                        or authority.get("artifact_sha256")
                        != artifact_sha256s.get("cuda_extension")
                        or authority.get("manifest_sha256")
                        != artifact_sha256s.get("extension_manifest")
                        or authority.get("package_receipt_sha256")
                        != artifact_sha256s.get("genome_upgrade_v2_package_receipt")
                        or not _valid_sha256(authority.get("package_id_sha256"))
                    ):
                        malformed = True
                        continue
                    indexed[str(cell_id)] = causal
                if malformed or set(indexed) != set(expected_ids):
                    fail(
                        f"{role} Genome V2 causal projection exposure receipts "
                        "are incomplete or binding-invalid"
                    )
                    return False
                return True

            canary_exposure = provider_native_exposure_receipt_valid(
                canary_result,
                batch_id=canary_batch_id,
            ) or causal_projection_exposure_valid(
                canary_manifest,
                canary_gv2_index,
                role="canary",
            )
            restore_exposure = provider_native_exposure_receipt_valid(
                restore_result,
                batch_id=restore_batch_id,
            ) or causal_projection_exposure_valid(
                restore_manifest,
                restore_gv2_index,
                role="fresh-owner",
            )
            exposure_complete = canary_exposure and restore_exposure
            checks["real_reproduction_exposure"] = exposure_complete
            checks["provider_native_exposure_receipts_complete"] = exposure_complete
            if not exposure_complete:
                fail("provider-native reproduction exposure receipts are incomplete")
        else:
            chronicle = (
                canary_result.get("provider_native_chronicle")
                if isinstance(canary_result, Mapping)
                else None
            )
            chronicle_cells = (
                chronicle.get("cell_receipts")
                if isinstance(chronicle, Mapping)
                else None
            )
            chronicle_complete = (
                isinstance(chronicle, Mapping)
                and chronicle.get("finalized") is True
                and chronicle.get("source_completeness")
                == "provider_native_segment_artifact_complete_v1"
                and isinstance(chronicle_cells, Mapping)
                and list(chronicle_cells) == list(expected_ids)
                and all(
                    isinstance(row, Mapping)
                    and row.get("cell_id") == cell_id
                    and row.get("finalized") is True
                    and type(row.get("segment_count")) is int
                    and row["segment_count"] > 0
                    and type(row.get("attempt_cursor")) is int
                    and row["attempt_cursor"] > 0
                    for cell_id, row in chronicle_cells.items()
                )
            )
            checks["provider_native_exposure_receipts_complete"] = chronicle_complete
            if not chronicle_complete:
                fail("provider-native reproduction exposure receipts are incomplete")

        restore_receipt = (
            restore_result.get("fresh_owner_restore_receipt")
            if isinstance(restore_result, Mapping)
            else None
        )
        source_digests = (
            canary_manifest.get("checkpoint", {}).get("terminal_cell_digests")
            if isinstance(canary_manifest, Mapping)
            and isinstance(canary_manifest.get("checkpoint"), Mapping)
            else None
        )
        restored_digests = (
            restore_manifest.get("checkpoint", {}).get("terminal_cell_digests")
            if isinstance(restore_manifest, Mapping)
            and isinstance(restore_manifest.get("checkpoint"), Mapping)
            else None
        )
        reload_rows = (
            restore_receipt.get("reload_receipts")
            if isinstance(restore_receipt, Mapping)
            else None
        )
        checkpoint_rebind = (
            restore_receipt.get("checkpoint_rebind")
            if isinstance(restore_receipt, Mapping)
            else None
        )
        source_generation = (
            max(
                (
                    generation
                    for generation in expected_milestones
                    if terminal_generation is not None
                    and 0 < generation < terminal_generation
                ),
                default=terminal_generation,
            )
            if terminal_generation is not None
            else None
        )
        source_checkpoint = (
            catalog.get(str(source_generation))
            if isinstance(catalog, Mapping) and source_generation is not None
            else None
        )
        source_checkpoint_digests = (
            source_checkpoint.get("terminal_cell_digests")
            if isinstance(source_checkpoint, Mapping)
            else None
        )
        restore_proven = (
            isinstance(restore_receipt, Mapping)
            and restore_receipt.get("schema_version")
            == FRESH_OWNER_RESTORE_RECEIPT_SCHEMA_VERSION
            and restore_receipt.get("projection_bundles_reloaded") is True
            and restore_receipt.get("source_batch_id") == canary_batch_id
            and restore_receipt.get("restored_batch_id") == restore_batch_id
            and restore_receipt.get("cell_ids") == list(expected_ids)
            and isinstance(source_digests, Mapping)
            and source_digests == restored_digests
            and restore_receipt.get("source_checkpoint_generation") == source_generation
            and isinstance(source_checkpoint_digests, Mapping)
            and restore_receipt.get("source_checkpoint_cell_digests")
            == source_checkpoint_digests
            and restore_receipt.get("restored_terminal_cell_digests")
            == restored_digests
            and isinstance(checkpoint_rebind, Mapping)
            and checkpoint_rebind.get("schema_version")
            == FRESH_OWNER_CHECKPOINT_REBIND_SCHEMA_VERSION
            and checkpoint_rebind.get("source_batch_id") == canary_batch_id
            and checkpoint_rebind.get("restored_batch_id") == restore_batch_id
            and checkpoint_rebind.get("source_checkpoint_binding")
            == (
                source_checkpoint.get("checkpoint_binding")
                if isinstance(source_checkpoint, Mapping)
                else None
            )
            and checkpoint_rebind.get("source_checkpoint_sha256")
            == (
                _canonical_sha256(source_checkpoint)
                if isinstance(source_checkpoint, Mapping)
                else None
            )
            and isinstance(reload_rows, list)
            and [row.get("cell_id") for row in reload_rows if isinstance(row, Mapping)]
            == list(expected_ids)
            and all(
                _valid_sha256(row.get("source_projection_sha256"))
                and row.get("source_projection_sha256")
                == row.get("loaded_projection_sha256")
                and row.get("source_projection_sha256")
                == source_checkpoint_digests.get(row.get("cell_id"))
                for row in reload_rows
                if isinstance(row, Mapping)
            )
        )
        checks["fresh_owner_bundle_reload_and_equality_proven"] = restore_proven
        if not restore_proven:
            fail(
                "fresh-owner bundle reload/equality receipt is absent or "
                "binding-invalid; terminal fast-path recovery is insufficient"
            )

        if canary_rows != restore_rows:
            fail("fresh-owner telemetry differs from the canary evidence")
            checks["fresh_owner_telemetry_equal"] = False
        else:
            checks["fresh_owner_telemetry_equal"] = bool(canary_rows)

        return {
            "schema_version": TECHNICAL_GATE_EVIDENCE_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "route_family_profile": route_family_profile,
            "gate_cell_ids": list(expected_ids),
            "checkpoint_milestones": list(expected_milestones),
            "terminal_generation": terminal_generation,
            "checks": checks,
            "reasons": reasons,
            "passed": not reasons and all(value is True for value in checks.values()),
        }

    @staticmethod
    def _genome_v2_checkpoint_binding_v1(
        plan: Mapping[str, Any],
        *,
        execution_route_sha256: str,
    ) -> tuple[tuple[str, str], ...]:
        """Bind Genome V2 package authority to the exact execution route."""

        configuration = production_engine_configuration_v1(plan)
        genome_configuration = configuration.component_binding()[
            "genome_inheritance"
        ]["configuration_id"]
        if genome_configuration != "genome_upgrade_v2":
            return ()
        from engine.evolution.genome_upgrade_v2_package_v1 import (
            checkpoint_authority_binding_v1,
        )
        from engine.evolution.global_genome_registry_v1 import (
            global_genome_profile_v1,
        )

        profile = global_genome_profile_v1(genome_configuration)
        receipt = _validate_genome_v2_package_v1(
            plan,
            required_obligations=profile.required_obligations,
        )
        receipt_row = _artifact_rows(plan)["genome_upgrade_v2_package_receipt"]
        return tuple(
            sorted(
                checkpoint_authority_binding_v1(
                    receipt["identity"],
                    package_id_sha256=receipt["package_id_sha256"],
                    package_receipt_sha256=receipt_row["sha256"],
                    execution_route_sha256=execution_route_sha256,
                ).items()
            )
        )

    def translate_batch(
        self, plan: CampaignPlanV1 | Mapping[str, Any], batch_id: str
    ) -> ResidentBatchRequestV1:
        payload = _payload(plan)
        if payload.get("resolved_profile") != self.profile_id:
            raise CampaignProviderError("production engine received another profile")
        execution = payload.get("execution")
        if (
            isinstance(execution, Mapping)
            and execution.get("execution_route_identity")
            == TICK_NATIVE_EXECUTION_ROUTE_V1
        ):
            tick_request = self.tick_native_execution_request_contract(payload)
            tick_control = self.tick_native_control_contract(payload)
            native_request = self._translate_tick_native_configuration(
                payload,
                batch_id,
                tick_request=tick_request,
                tick_control=tick_control,
            )
            route_sha256 = self.execution_route_contract(payload).execution_route_sha256
            return replace(
                native_request,
                execution_route_sha256=route_sha256,
                genome_v2_checkpoint_authority_binding=(
                    self._genome_v2_checkpoint_binding_v1(
                        payload,
                        execution_route_sha256=route_sha256,
                    )
                ),
            )
        migration_reason = self._configuration_migration_reason(payload)
        if migration_reason is not None:
            return self._translate_unmigrated_configuration(payload, batch_id)
        if self._direct_resident_binding(payload) is not None:
            request = self._translate_unmigrated_configuration(payload, batch_id)
            publish_generation_zero = 0 in request.checkpoint_milestones
            milestones = self._configured_rolling_segment_ends(
                payload,
                generations=request.generations,
                declared_milestones=request.checkpoint_milestones,
            )
            if publish_generation_zero:
                milestones = (0, *milestones)
            request = replace(
                request,
                checkpoint_milestones=milestones,
            )
            route_sha256 = self.execution_route_contract(
                payload, request
            ).execution_route_sha256
            return replace(
                request,
                execution_route_sha256=route_sha256,
                genome_v2_checkpoint_authority_binding=(
                    self._genome_v2_checkpoint_binding_v1(
                        payload,
                        execution_route_sha256=route_sha256,
                    )
                ),
            )
        if not self._retired_cross_path_configuration(payload):
            raise CampaignProviderError(
                self._direct_resident_binding_unmigrated_message(payload)
            )
        registry = self._ensure_committed_default_adapters(payload)
        _validate_native(self._native(registry), registry)
        batches = payload.get("physical_batches")
        if not isinstance(batches, list) or not batches:
            raise CampaignProviderError(
                "cross-path provider requires physical resident batches"
            )
        batch = next(
            (
                row
                for row in batches
                if isinstance(row, Mapping) and row.get("batch_id") == batch_id
            ),
            None,
        )
        raw_ids = batch.get("cell_ids") if isinstance(batch, Mapping) else None
        if (
            not isinstance(batch, Mapping)
            or isinstance(raw_ids, (str, bytes))
            or not isinstance(raw_ids, Sequence)
        ):
            raise CampaignProviderError(f"cross-path batch is absent: {batch_id}")
        try:
            ids = [validate_safe_cell_id(value) for value in raw_ids]
        except GpuResidentSupportError as exc:
            raise CampaignProviderError("cross-path cell identity is unsafe") from exc
        lookup = _cell_lookup(payload)
        if not 1 <= len(ids) <= int(registry["max_admitted_batch"]) or len(
            set(ids)
        ) != len(ids):
            raise CampaignProviderError("cross-path batch exceeds its registry bound")
        try:
            cells = [lookup[cell_id] for cell_id in ids]
        except KeyError as exc:
            raise CampaignProviderError(
                "cross-path batch contains unknown cells"
            ) from exc
        for cell in cells:
            _validate_cell(cell, registry)
        execution = payload.get("execution")
        if not isinstance(execution, Mapping):
            raise CampaignProviderError("cross-path execution is malformed")
        generations = execution.get("generations")
        ticks = execution.get("ticks_per_generation")
        checkpoints = execution.get("checkpoints")
        if (
            type(generations) is not int
            or not 1 <= generations <= registry["max_generations"]
            or ticks != registry["ticks_per_generation"]
            or not isinstance(checkpoints, list)
            or any(
                type(value) is not int or not 0 < value <= generations
                for value in checkpoints
            )
        ):
            raise CampaignProviderError(
                "cross-path execution exceeds its registry contract"
            )
        milestones = tuple(sorted({0, generations, *checkpoints}))
        exposure_capacity = registry.get("reproduction_exposure_capacity")
        if exposure_capacity is not None:
            resident_capacities = {
                dict(cell["parameters"]).get("patch.capacity") for cell in cells
            }
            segment_generations = max(
                right - left
                for left, right in zip(milestones[:-1], milestones[1:], strict=True)
            )
            if (
                len(resident_capacities) != 1
                or type(next(iter(resident_capacities))) is not int
            ):
                raise CampaignProviderError(
                    "cross-path resident capacity must be one exact integer"
                )
            required_attempts = _minimum_segment_attempt_capacity_v1(
                segment_generations=segment_generations,
                ticks_per_generation=ticks,
                resident_capacity=int(next(iter(resident_capacities))),
            )
            if int(exposure_capacity["attempt_capacity"]) < required_attempts:
                raise CampaignProviderError(
                    "reproduction exposure attempt capacity is below the "
                    "declarative segment decision/refill bound"
                )
        parameter_names = sorted(
            {
                *dict(registry["fixed_parameters"]),
                *dict(registry["parameter_bindings"]).values(),
            }
        )
        route = self.execution_route_contract(payload).execution_route_sha256
        acs_checkpoint_slots = execution.get("acs_checkpoint_slots", 0)
        raw_acs_capture_instrumentation = execution.get(
            "acs_capture_instrumentation", False
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
            0,
            1,
            2,
            3,
        }:
            raise CampaignProviderError("cross-path ACS checkpoint slots are invalid")
        if type(raw_acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "cross-path ACS capture instrumentation is invalid"
            )
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=tuple(ids),
            generations=generations,
            ticks_per_generation=ticks,
            checkpoint_milestones=milestones,
            parameter_arrays=tuple(
                (
                    name,
                    tuple(dict(cell["parameters"])[name] for cell in cells),
                )
                for name in parameter_names
            ),
            seed_arrays=tuple(
                (
                    owner,
                    tuple(int(dict(cell["seeds"])[owner]) for cell in cells),
                )
                for owner in sorted(registry["seed_owners"])
            ),
            checkpoint_storage_mode=str(
                execution.get("checkpoint_storage_mode", "disk")
            ),
            reporting_enabled=bool(execution.get("reporting_enabled", True)),
            execution_route_sha256=route,
            acs_checkpoint_slots=acs_checkpoint_slots,
            acs_capture_instrumentation=(
                raw_acs_capture_instrumentation or bool(acs_checkpoint_slots)
            ),
        )

    @staticmethod
    def _configuration_migration_reason(plan: Mapping[str, Any]) -> str | None:
        from .production_engine_components import (
            production_engine_component_binding_v1,
        )
        from .production_engine_configuration import (
            production_engine_configuration_v1,
        )

        binding = production_engine_component_binding_v1(
            production_engine_configuration_v1(plan)
        )
        if not binding.unmigrated_components:
            return None
        return (
            "configuration is bound to the single production engine, but these "
            "domain mechanics have not migrated: "
            + ", ".join(binding.unmigrated_components)
        )

    def _translate_tick_native_configuration(
        self,
        plan: Mapping[str, Any],
        batch_id: str,
        *,
        tick_request: TickNativeExecutionRequestV1,
        tick_control: TickNativeControlV1,
    ) -> ResidentBatchRequestV1:
        """Translate absolute-tick production input without a legacy G/T plan."""

        batches = plan.get("physical_batches")
        cells = plan.get("cells")
        execution = plan.get("execution")
        if (
            not isinstance(batches, list)
            or not isinstance(cells, list)
            or not isinstance(execution, Mapping)
        ):
            raise CampaignProviderError("tick-native production plan is malformed")
        batch = next(
            (
                row
                for row in batches
                if isinstance(row, Mapping) and row.get("batch_id") == batch_id
            ),
            None,
        )
        raw_ids = batch.get("cell_ids") if isinstance(batch, Mapping) else None
        if (
            isinstance(raw_ids, (str, bytes))
            or not isinstance(raw_ids, Sequence)
            or not raw_ids
        ):
            raise CampaignProviderError(
                f"tick-native production batch is absent: {batch_id}"
            )
        cell_by_id = {
            cell.get("cell_id"): cell
            for cell in cells
            if isinstance(cell, Mapping) and isinstance(cell.get("cell_id"), str)
        }
        try:
            selected = [cell_by_id[str(cell_id)] for cell_id in raw_ids]
        except KeyError as exc:
            raise CampaignProviderError(
                "tick-native production batch references an unknown cell"
            ) from exc
        parameter_names = sorted(
            {str(name) for cell in selected for name in cell["parameters"]}
        )
        seed_names = sorted(
            {str(name) for cell in selected for name in cell["seeds"]}
        )
        acs_checkpoint_slots = execution.get("acs_checkpoint_slots", 0)
        acs_capture_instrumentation = execution.get(
            "acs_capture_instrumentation", False
        )
        native_streamed_checkpoint = execution.get(
            "native_streamed_checkpoint", False
        )
        world_sharded_checkpoint = execution.get(
            "world_sharded_checkpoint", False
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
            0,
            1,
            2,
            3,
        }:
            raise CampaignProviderError(
                "tick-native production ACS checkpoint slots are invalid"
            )
        if type(acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "tick-native ACS capture instrumentation is invalid"
            )
        if type(native_streamed_checkpoint) is not bool:
            raise CampaignProviderError(
                "tick-native native-streamed checkpoint selection is invalid"
            )
        if type(world_sharded_checkpoint) is not bool:
            raise CampaignProviderError(
                "tick-native world-sharded checkpoint selection is invalid"
            )
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=tuple(str(value) for value in raw_ids),
            generations=None,
            ticks_per_generation=None,
            checkpoint_milestones=(),
            parameter_arrays=tuple(
                (name, tuple(cell["parameters"][name] for cell in selected))
                for name in parameter_names
            ),
            seed_arrays=tuple(
                (name, tuple(int(cell["seeds"][name]) for cell in selected))
                for name in seed_names
            ),
            checkpoint_storage_mode=str(
                execution.get("checkpoint_storage_mode", "disk")
            ),
            reporting_enabled=bool(execution.get("reporting_enabled", True)),
            total_ticks=tick_request.total_ticks,
            start_next_tick=tick_request.start_next_tick,
            checkpoint_ticks=tick_control.resolved_checkpoint_ticks,
            deliberate_restore_tick=tick_request.deliberate_restore_tick,
            deliberate_restore_ticks=tick_request.canonical_deliberate_restore_ticks,
            acs_checkpoint_slots=acs_checkpoint_slots,
            acs_capture_instrumentation=(
                acs_capture_instrumentation or bool(acs_checkpoint_slots)
            ),
            native_streamed_checkpoint=native_streamed_checkpoint,
            world_sharded_checkpoint=world_sharded_checkpoint,
            resident_owner_span_ticks=tick_control.resident_owner_span_ticks,
            native_dispatch_window_ticks=tick_control.native_dispatch_window_ticks,
            report_every_ticks=tick_control.report_every_ticks,
            science_sample_interval_ticks=(
                tick_control.science_sample_interval_ticks
            ),
            reducer_every_ticks=tick_control.reducer_every_ticks,
            archive_every_ticks=tick_control.archive_every_ticks,
            event_window_capacity=tick_control.event_window_capacity,
            source_window_ticks=tick_control.source_window_ticks,
            tick_native_control=tick_control.payload(),
        )

    def _translate_unmigrated_configuration(
        self,
        plan: Mapping[str, Any],
        batch_id: str,
    ) -> ResidentBatchRequestV1:
        batches = plan.get("physical_batches")
        cells = plan.get("cells")
        execution = plan.get("execution")
        if (
            not isinstance(batches, list)
            or not isinstance(cells, list)
            or not isinstance(execution, Mapping)
        ):
            raise CampaignProviderError(
                "unmigrated production engine plan is malformed"
            )
        batch = next(
            (
                row
                for row in batches
                if isinstance(row, Mapping) and row.get("batch_id") == batch_id
            ),
            None,
        )
        raw_ids = batch.get("cell_ids") if isinstance(batch, Mapping) else None
        if (
            isinstance(raw_ids, (str, bytes))
            or not isinstance(raw_ids, Sequence)
            or not raw_ids
        ):
            raise CampaignProviderError(
                f"production engine batch is absent: {batch_id}"
            )
        cell_by_id = {
            cell.get("cell_id"): cell
            for cell in cells
            if isinstance(cell, Mapping) and isinstance(cell.get("cell_id"), str)
        }
        try:
            selected = [cell_by_id[str(cell_id)] for cell_id in raw_ids]
        except KeyError as exc:
            raise CampaignProviderError(
                "production engine batch references an unknown cell"
            ) from exc
        parameter_names = sorted(
            {str(name) for cell in selected for name in cell["parameters"]}
        )
        seed_names = sorted({str(name) for cell in selected for name in cell["seeds"]})
        generations = execution.get("generations")
        ticks = execution.get("ticks_per_generation")
        checkpoints = execution.get("checkpoints")
        if (
            type(generations) is not int
            or type(ticks) is not int
            or not isinstance(checkpoints, list)
        ):
            raise CampaignProviderError(
                "production engine execution dimensions are malformed"
            )
        acs_checkpoint_slots = execution.get("acs_checkpoint_slots", 0)
        raw_acs_capture_instrumentation = execution.get(
            "acs_capture_instrumentation", False
        )
        if type(acs_checkpoint_slots) is not int or acs_checkpoint_slots not in {
            0,
            1,
            2,
            3,
        }:
            raise CampaignProviderError(
                "production engine ACS checkpoint slots are invalid"
            )
        if type(raw_acs_capture_instrumentation) is not bool:
            raise CampaignProviderError(
                "production engine ACS capture instrumentation is invalid"
            )
        raw_native_streamed_checkpoint = execution.get(
            "native_streamed_checkpoint", False
        )
        if type(raw_native_streamed_checkpoint) is not bool:
            raise CampaignProviderError(
                "production engine native streamed checkpoint selection is invalid"
            )
        raw_world_sharded_checkpoint = execution.get(
            "world_sharded_checkpoint", False
        )
        if type(raw_world_sharded_checkpoint) is not bool:
            raise CampaignProviderError(
                "production engine world-sharded checkpoint selection is invalid"
            )
        return ResidentBatchRequestV1(
            profile_id=self.profile_id,
            batch_id=batch_id,
            cell_ids=tuple(str(value) for value in raw_ids),
            generations=generations,
            ticks_per_generation=ticks,
            checkpoint_milestones=tuple(int(value) for value in checkpoints),
            parameter_arrays=tuple(
                (
                    name,
                    tuple(cell["parameters"][name] for cell in selected),
                )
                for name in parameter_names
            ),
            seed_arrays=tuple(
                (
                    name,
                    tuple(int(cell["seeds"][name]) for cell in selected),
                )
                for name in seed_names
            ),
            checkpoint_storage_mode=str(
                execution.get("checkpoint_storage_mode", "disk")
            ),
            reporting_enabled=bool(execution.get("reporting_enabled", True)),
            acs_checkpoint_slots=acs_checkpoint_slots,
            acs_capture_instrumentation=(
                raw_acs_capture_instrumentation or bool(acs_checkpoint_slots)
            ),
            native_streamed_checkpoint=raw_native_streamed_checkpoint,
            world_sharded_checkpoint=raw_world_sharded_checkpoint,
        )

    def readiness_reasons(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        request: ResidentBatchRequestV1,
    ) -> tuple[UnsupportedReasonV1, ...]:
        payload = _payload(plan)
        migration_reason = self._configuration_migration_reason(payload)
        if migration_reason is not None:
            return (
                UnsupportedReasonV1(
                    "production_engine_components_not_migrated",
                    "runtime.components",
                    migration_reason,
                    self.profile_id,
                ),
            )
        if self._direct_resident_binding(payload) is not None:
            return self._patch_fixed_readiness_reasons(payload, request)
        if not self._retired_cross_path_configuration(payload):
            return (
                UnsupportedReasonV1(
                    "production_engine_direct_resident_binding_unmigrated",
                    "runtime.components",
                    self._direct_resident_binding_unmigrated_message(payload),
                    self.profile_id,
                ),
            )
        self._ensure_committed_default_adapters(payload)
        reasons: list[UnsupportedReasonV1] = []
        try:
            self._mechanic_receipt_contract(payload, request)
        except (CampaignProviderError, ImportError, AttributeError) as exc:
            reasons.append(
                UnsupportedReasonV1(
                    "inheritance_cross_path_execution_owner_unqualified",
                    "runtime.execution_owner",
                    str(exc),
                    self.profile_id,
                )
            )
        else:
            adapter_reason = self._execution_adapter_reason()
            if adapter_reason is not None:
                reasons.append(
                    UnsupportedReasonV1(
                        "inheritance_cross_path_execution_adapters_unbound",
                        "runtime.execution_adapters",
                        adapter_reason,
                        self.profile_id,
                    )
                )
            for artifact_id in ("extension_manifest", "reporting_receipt"):
                try:
                    _verified_artifact(payload, artifact_id)
                except CampaignProviderError as exc:
                    reasons.append(
                        UnsupportedReasonV1(
                            "inheritance_cross_path_execution_artifact_unbound",
                            f"artifacts.{artifact_id}",
                            str(exc),
                            self.profile_id,
                        )
                    )
            if self._cross_path_artifact_chain_loader_v1 is not None:
                try:
                    self._cross_path_artifact_chain_loader_v1.validate_chain()
                except RuntimeError as exc:
                    reasons.append(
                        UnsupportedReasonV1(
                            "inheritance_cross_path_artifact_chain_invalid",
                            "artifacts",
                            str(exc),
                            self.profile_id,
                        )
                    )
        if self.device.split(":", 1)[0] != "cuda":
            reasons.append(
                UnsupportedReasonV1(
                    "inheritance_cross_path_non_cuda_device",
                    "execution.device",
                    "cross-path provider rejects CPU execution and fallback",
                    self.profile_id,
                )
            )
        try:
            cuda_available = bool(self.torch.cuda.is_available())
        except (AttributeError, RuntimeError):
            cuda_available = False
        if not cuda_available:
            reasons.append(
                UnsupportedReasonV1(
                    "inheritance_cross_path_cuda_unavailable",
                    "runtime.cuda",
                    "CUDA runtime is unavailable",
                    self.profile_id,
                )
            )
        try:
            registry = _global_genome_profile_contract(payload)
            self.execution_route_contract(payload, request)
            if self.require_prebuilt:
                for artifact_id in _artifact_rows(payload):
                    _verified_artifact(payload, artifact_id)
                extension_path, extension_row = _verified_artifact(
                    payload, "cuda_extension"
                )
                if extension_row["abi"] != registry["native_abi"]:
                    raise CampaignProviderError("cross-path CUDA ABI mismatch")
                verified_prebuilt_artifact_binding(
                    extension_path,
                    expected_sha256=str(extension_row["sha256"]),
                )
        except (
            CampaignProviderError,
            GpuResidentSupportError,
            ImportError,
            AttributeError,
        ) as exc:
            reasons.append(
                UnsupportedReasonV1(
                    "inheritance_cross_path_contract_invalid",
                    "artifacts",
                    str(exc),
                    self.profile_id,
                )
            )
        return tuple(reasons)

    def estimate_resources(
        self,
        plan: CampaignPlanV1 | Mapping[str, Any],
        requests: Sequence[ResidentBatchRequestV1],
    ) -> dict[str, Any]:
        payload = _payload(plan)
        native_binding = self._direct_resident_binding(payload)
        if native_binding is not None:
            from .production_engine_native_bindings import (
                EED_FIXED_MLP_BINDING_ID,
                GMNW_POLICY_ARMS_BINDING_ID,
                JUNGLE_FIXED_MLP_BINDING_ID,
                PATCH_FIXED_MLP_BINDING_ID,
                PATCH_STABLE_GRU_BINDING_ID,
                WDL_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_EED_HAPB_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_FIXED_MLP_BINDING_ID,
                WDL_STABLE_GRU_BINDING_ID,
            )
            from .resources import (
                PATCH_MIN_LOGICAL_GENERATIONS,
                estimate_patch_fixed_resident_resources_v1,
                estimate_patch_gmnw_gru_resident_resources_v1,
                estimate_patch_jungle_fixed_mlp_resident_resources_v1,
                estimate_patch_stable_gru_resident_resources_v1,
                estimate_patch_wdl_fixed_mlp_resident_resources_v1,
                estimate_patch_wdl_stable_gru_resident_resources_v1,
                resident_request_resource_dimensions_v1,
            )

            estimators = {
                PATCH_FIXED_MLP_BINDING_ID: (
                    estimate_patch_fixed_resident_resources_v1
                ),
                EED_FIXED_MLP_BINDING_ID: (estimate_patch_fixed_resident_resources_v1),
                WDL_FIXED_MLP_BINDING_ID: (
                    estimate_patch_wdl_fixed_mlp_resident_resources_v1
                ),
                WDL_JUNGLE_FIXED_MLP_BINDING_ID: (
                    estimate_patch_wdl_fixed_mlp_resident_resources_v1
                ),
                WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID: (
                    estimate_patch_wdl_fixed_mlp_resident_resources_v1
                ),
                WDL_JUNGLE_EED_HAPB_FIXED_MLP_BINDING_ID: (
                    estimate_patch_wdl_fixed_mlp_resident_resources_v1
                ),
                JUNGLE_FIXED_MLP_BINDING_ID: (
                    estimate_patch_jungle_fixed_mlp_resident_resources_v1
                ),
                PATCH_STABLE_GRU_BINDING_ID: (
                    estimate_patch_stable_gru_resident_resources_v1
                ),
                WDL_STABLE_GRU_BINDING_ID: (
                    estimate_patch_wdl_stable_gru_resident_resources_v1
                ),
                GMNW_POLICY_ARMS_BINDING_ID: (
                    estimate_patch_gmnw_gru_resident_resources_v1
                ),
            }
            include_parameter_dimensions = native_binding.binding_id in {
                PATCH_FIXED_MLP_BINDING_ID,
                EED_FIXED_MLP_BINDING_ID,
                WDL_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_EED_HAPB_FIXED_MLP_BINDING_ID,
            }
            dimensions = resident_request_resource_dimensions_v1(
                payload,
                requests,
                include_parameter_dimensions=include_parameter_dimensions,
            )
            dimensions.pop("minimum_resident_segment_generations", None)
            time_authority = dimensions.pop("time_authority", None)
            total_ticks = dimensions.pop("total_ticks", None)
            tick_native_projection: dict[str, Any] | None = None
            native_estimator_dimensions: dict[str, Any] | None = None
            if time_authority == "absolute_tick_v1":
                native_owner_span = dimensions.pop("resident_owner_span_ticks")
                native_dispatch_window = dimensions.pop("native_dispatch_window_ticks")
                native_source_window = dimensions.pop("source_window_ticks")
                native_event_capacity = dimensions.pop("event_window_capacity")
                dimensions.pop("resident_dispatch_window_ticks", None)
                dimensions.pop("resident_event_bytes_horizon_invariant", None)
                dimensions.pop("resident_source_bytes_horizon_invariant", None)
                source_window_ticks = int(native_source_window)
                requested_source_windows = max(
                    1,
                    (int(native_dispatch_window) + source_window_ticks - 1)
                    // source_window_ticks,
                )
                requested_owner_source_windows = max(
                    1,
                    (int(native_owner_span) + source_window_ticks - 1)
                    // source_window_ticks,
                )
                native_estimator_dimensions = {
                    **dimensions,
                    "time_authority": "absolute_tick_v1",
                    "total_ticks": total_ticks,
                    "resident_owner_span_ticks": native_owner_span,
                    "native_dispatch_window_ticks": native_dispatch_window,
                    "source_window_ticks": native_source_window,
                    "event_window_capacity": native_event_capacity,
                }
                tick_native_projection = {
                    "time_authority": "absolute_tick_v1",
                    "requested_total_ticks": total_ticks,
                    "resident_owner_span_ticks": native_owner_span,
                    "native_dispatch_window_ticks": native_dispatch_window,
                    "source_window_ticks": native_source_window,
                    "event_window_capacity": native_event_capacity,
                    "requested_source_windows": requested_source_windows,
                    "requested_owner_source_windows": (
                        requested_owner_source_windows
                    ),
                    "conservative_estimator_source_windows": (
                        max(
                            PATCH_MIN_LOGICAL_GENERATIONS,
                            requested_source_windows,
                        )
                    ),
                    "conservative_estimator_owner_source_windows": max(
                        PATCH_MIN_LOGICAL_GENERATIONS,
                        requested_owner_source_windows,
                    ),
                    "terminal_tick_reinterpreted": False,
                }
                for name in (
                    "resident_owner_span_ticks",
                    "native_dispatch_window_ticks",
                    "event_window_capacity",
                    "source_window_ticks",
                    "resident_dispatch_window_ticks",
                    "resident_event_bytes_horizon_invariant",
                    "resident_source_bytes_horizon_invariant",
                ):
                    dimensions.pop(name, None)

            def finalize_tick_native_estimate(
                estimate: Mapping[str, Any],
            ) -> dict[str, Any]:
                if tick_native_projection is None:
                    return dict(estimate)
                normalized = dict(estimate)
                for retired_name in (
                    "logical_generations",
                    "ticks_per_generation",
                    "resident_segment_generations",
                    "physical_storage_generations",
                    "physical_window_generations",
                ):
                    # Older non-target estimators may still use private row
                    # dimensions.  They are discarded at this boundary and
                    # can never enter the active route/resource contract.
                    normalized.pop(retired_name, None)
                normalized["time_authority"] = "absolute_tick_v1"
                normalized["total_ticks"] = total_ticks
                normalized["tick_native_resource_projection"] = dict(
                    tick_native_projection
                )
                return normalized
            from .eiel_full_evolution_v2_packed_resources_v1 import (
                estimate_eiel_full_evolution_v2_packed_resident_resources_v1,
                is_eiel_full_evolution_v2_packed_identity_v1,
            )

            if (
                native_binding.binding_id == WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID
                and is_eiel_full_evolution_v2_packed_identity_v1(dimensions)
            ):
                estimator_dimensions = (
                    native_estimator_dimensions
                    if native_estimator_dimensions is not None
                    else dimensions
                )
                estimate = estimate_eiel_full_evolution_v2_packed_resident_resources_v1(
                    **estimator_dimensions
                )
                if tick_native_projection is not None:
                    estimate = finalize_tick_native_estimate(estimate)
                return estimate
            if native_estimator_dimensions is not None:
                # Other estimator implementations still consume bounded row
                # counts. Never derive those private capacities from total_ticks.
                dimensions["logical_generations"] = max(
                    PATCH_MIN_LOGICAL_GENERATIONS,
                    requested_source_windows,
                )
                dimensions["resident_segment_generations"] = max(
                    PATCH_MIN_LOGICAL_GENERATIONS,
                    requested_owner_source_windows,
                )
                dimensions["ticks_per_generation"] = source_window_ticks
            estimate = estimators[native_binding.binding_id](**dimensions)
            if native_binding.binding_id in {
                WDL_JUNGLE_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_EED_FIXED_MLP_BINDING_ID,
                WDL_JUNGLE_EED_HAPB_FIXED_MLP_BINDING_ID,
            }:
                jungle_estimate = estimate_patch_jungle_fixed_mlp_resident_resources_v1(
                    **dimensions
                )
                vram_projections = (
                    estimate.get("vram_bytes_projection"),
                    jungle_estimate.get("vram_bytes_projection"),
                )
                wall_projections = (
                    estimate.get("wall_seconds_projection"),
                    jungle_estimate.get("wall_seconds_projection"),
                )
                if not all(
                    type(value) is int and value > 0 for value in vram_projections
                ):
                    raise CampaignProviderError(
                        "WDL+Jungle resource projections must be positive integers"
                    )
                if not all(
                    type(value) is int and value > 0 for value in wall_projections
                ):
                    raise CampaignProviderError(
                        "WDL+Jungle wall projections must be positive integers"
                    )
                vram_bounds = (
                    estimate.get("vram_bytes_upper_bound"),
                    jungle_estimate.get("vram_bytes_upper_bound"),
                )
                estimate = {
                    **estimate,
                    "resource_estimator_id": (
                        "patch_wdl_jungle_fixed_mlp_candidate_resources_v1"
                    ),
                    "vram_bytes_upper_bound": (
                        max(vram_bounds)
                        if all(
                            type(value) is int and value >= 0 for value in vram_bounds
                        )
                        else "unavailable"
                    ),
                    "vram_bytes_projection": max(vram_projections),
                    "vram_bytes_unavailable_reason": (
                        None
                        if all(
                            type(value) is int and value >= 0 for value in vram_bounds
                        )
                        else "WDL+Jungle VRAM uses a conservative combined projection "
                        "until the bounded route canary records exact runtime evidence"
                    ),
                    "disk_bytes_upper_bound": max(
                        int(estimate.get("disk_bytes_upper_bound", 0)),
                        int(jungle_estimate.get("disk_bytes_upper_bound", 0)),
                    ),
                    # Jungle deliberately withholds an ordinary-execution B2
                    # wall bound.  Both estimators still publish conservative
                    # projections, which are sufficient to bound a Q6 canary
                    # without granting ordinary execution authority.
                    "wall_seconds_upper_bound": max(wall_projections),
                    "wall_seconds_unavailable_reason": None,
                    "admission_note": (
                        "conservative resource visibility for named "
                        "WDL+Jungle binding candidate only; execution remains "
                        "unqualified"
                    ),
                }
            if tick_native_projection is not None:
                estimate = finalize_tick_native_estimate(estimate)
            return estimate
        budgets = dict(dict(payload.get("execution") or {}).get("budgets") or {})
        return {
            "resource_estimator_id": "inheritance_cross_path_declared_budgets_v1",
            "vram_bytes_upper_bound": int(budgets.get("vram_bytes", 0)),
            "wall_seconds_upper_bound": int(budgets.get("wall_seconds", 0)),
            "disk_bytes_upper_bound": int(budgets.get("disk_bytes", 0)),
        }
