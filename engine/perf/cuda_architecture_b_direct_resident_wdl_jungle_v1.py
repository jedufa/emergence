"""Execute the shared WDL/Jungle owner with an optional explicit EED overlay."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from time import perf_counter_ns
from typing import Any, Callable, Final

import torch

from engine.perf.composite_runtime_state_v1 import (
    CompositeRuntimeStateError,
    compose_composite_runtime_state_v1,
)
from engine.perf.cxxbackbone10_resident_state_owner import projection_digest
from engine.native_runtime.cuda_architecture_b_backend_factory import (
    CudaArchitectureBRunnerEnvelope,
    derive_cuda_architecture_b_runner_envelope,
)
from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    create_run_control_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    DetachedCellProjectionV1,
    DirectResidentPatchMulticellSettlementV1,
)
from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
    EED_ABI_VERSION_V1,
    EED_STATE_FIELDS_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
    JUNGLE_ABI_VERSION_V4,
    JUNGLE_STATE_FIELDS_V4,
    JUNGLE_SUPPORTED_TICKS_PER_GENERATION_V4,
    _allocate_reporting,
    _normalized_jungle_config,
    _required_jungle_metadata,
    _validated_capability as _validated_jungle_capability,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    RESULT_SCHEMA_VERSION,
    resident_window_count_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_ARGUMENT_INDEX_V1,
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
)
from engine.perf.cuda_architecture_b_hapb_v1 import (
    HAPB_DIRECT_ENTRYPOINT_V1,
    HAPBContractError,
    HostAuthoritativePolicyBankV1,
    POLICY_CROSSOVER_PACKED_WIDTH_V1,
    commit_hapb_ordinary_fixed_mlp_birth_rows_v1,
    commit_hapb_turnover_fixed_mlp_birth_rows_v1,
    hapb_physical_cell_count_v1,
    export_hapb_policy_bank_checkpoint_v1,
    restore_hapb_policy_bank_checkpoint_v1,
    restore_hapb_policy_owner_transient_state_v1,
)
from perf.inheritance_sidecar_identity_v1 import (
    INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
    inheritance_sidecar_payload_sha256_v1,
)

from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    WDL_ABI_VERSION_V3,
    WDL_RESIDENT_GENERATION_MAX_V2,
    WDL_STATE_FIELDS_V2,
    PreparedDirectResidentWdlV2,
    PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V2,
    PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V2,
    PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V2,
    TICKS_ARGUMENT_INDEX_V2,
    build_prepared_direct_resident_wdl_v2,
    build_prepared_direct_resident_wdl_v2_streaming,
    project_direct_resident_wdl_multicell_settlement_v2,
    project_direct_resident_wdl_generation_zero_v2,
)


WDL_JUNGLE_DIRECT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_v1"
)
HAPB_HOST_BOUNDARY_TIMING_SCHEMA_V1: Final[str] = (
    "eiel_b48_wdl_jungle_eed_hapb_host_boundary_timing_v1"
)
WDL_JUNGLE_EED_DIRECT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_v1"
)
WDL_JUNGLE_EED_TICK_DISPATCH_ENTRYPOINT_V2: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2"
)
WDL_JUNGLE_EED_HAPB_TICK_SPAN_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1"
)
HAPB_POLICY_WINDOW_IDENTITY_COMMIT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_hapb_commit_policy_window_identity_owner_entry_v1"
)
HAPB_POLICY_WINDOW_IDENTITY_EXPORT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_hapb_export_policy_window_identity_owner_checkpoint_v1"
)
HAPB_POLICY_WINDOW_IDENTITY_RESTORE_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_hapb_restore_policy_window_identity_owner_checkpoint_v1"
)
HAPB_POLICY_WINDOW_IDENTITY_CHECKPOINT_SCHEMA_V1: Final[str] = (
    "eiel_hapb_policy_window_identity_checkpoint_v1"
)
HAPB_POLICY_WINDOW_IDENTITY_VALIDATION_TOKEN_FIELD_V1: Final[str] = (
    "_hapb_policy_window_identity_validation_token_v1"
)
START_TICK_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "config.start_tick"
)
PATCH_P2_CONFIG_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "patch_p2_config"
)
TURNOVER_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "config.generation_turnover_enabled"
)
TURNOVER_STATE_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "turnover_state_tensors"
)
TURNOVER_SHADOW_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "turnover_shadow_tensors"
)
TURNOVER_OUTPUT_ARGUMENT_INDEX_V1: Final[int] = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(
    "turnover_output_tensors"
)
TURNOVER_POINTER_TABLE_ARGUMENT_INDEX_V1: Final[int] = (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("turnover_pointer_table_tensors")
)
WDL_JUNGLE_RESIDENT_GENERATION_MIN_V1: Final[int] = 3
_JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1: Final[tuple[str, ...]] = (
    "predator_base_risk",
    "predator_health_damage",
    "predator_risk_cap",
    "recovery_base_growth",
    "recovery_decay_factor",
    "recovery_fertility_weight",
    "recovery_waste_penalty_weight",
)


HAPB_DIRECT_CHECKPOINT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_architecture_b_direct_wdl_jungle_hapb_checkpoint_v1"
)
HAPB_DIRECT_COMPONENT_CHECKPOINT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_architecture_b_direct_wdl_jungle_hapb_component_checkpoint_v1"
)


class CudaArchitectureBDirectResidentWdlJungleV1Error(RuntimeError):
    """The exact shared non-EED WDL/Jungle resident owner is unavailable."""


def _fail(message: str) -> CudaArchitectureBDirectResidentWdlJungleV1Error:
    return CudaArchitectureBDirectResidentWdlJungleV1Error(message)


def _bind_tick_span_turnover_arguments_v1(
    arguments: list[Any],
    *,
    generation_turnover_configured: bool,
    commit_turnover: bool,
) -> None:
    """Project a configured lifecycle onto one exact partial native span.

    A nonterminal span retains the turnover-enabled plan identity but cannot
    expose mutable turnover authority to the native controller.  The 16 status
    outputs remain present because they are resident reporting, not mutation
    authority.
    """

    native_turnover_enabled = generation_turnover_configured and commit_turnover
    arguments[TURNOVER_ARGUMENT_INDEX_V1] = native_turnover_enabled
    if generation_turnover_configured and not commit_turnover:
        arguments[TURNOVER_STATE_ARGUMENT_INDEX_V1] = []
        arguments[TURNOVER_SHADOW_ARGUMENT_INDEX_V1] = []
        arguments[TURNOVER_POINTER_TABLE_ARGUMENT_INDEX_V1] = []


def _tick_native_private_reporting_configs_v1(
    prepared: "PreparedDirectResidentWdlJungleV1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebase invocation-local reporting rows without changing time authority.

    Tick-native execution passes the absolute cursor through ``start_tick`` and
    uses generation zero only as the private flattened-controller row selector.
    A prepared checkpoint source can retain a nonzero legacy generation label,
    so its WDL/Jungle reporting contexts must be copied onto the invocation-local
    row origin before entering that private ABI.  The source projection and its
    compatibility metadata remain unchanged and non-authoritative.
    """

    configs = (
        ("WDL", prepared.wdl.config),
        ("Jungle", prepared.jungle_config),
    )
    rebound: list[dict[str, Any]] = []
    for name, config in configs:
        if not isinstance(config, Mapping):
            raise _fail(f"tick-native {name} reporting config is malformed")
        legacy_base = config.get("generation_base")
        if type(legacy_base) is not int or legacy_base < 0:
            raise _fail(
                f"tick-native {name} private reporting row base is malformed"
            )
        copied = dict(config)
        copied["generation_base"] = 0
        rebound.append(copied)
    return rebound[0], rebound[1]


def _require_no_native_failure_v1(result: Mapping[str, Any]) -> None:
    """Fail closed on the controller's sticky first-failure authority."""

    surfaces: list[Mapping[str, Any]]
    direct_sentinel = result.get("first_failure_no_failure_sentinel")
    direct_failures = result.get("first_failure_by_cell")
    if direct_sentinel is not None or direct_failures is not None:
        surfaces = [result]
    else:
        window_receipts = result.get("window_receipts")
        if not isinstance(window_receipts, list) or not window_receipts:
            raise _fail("WDL/Jungle/EED result has malformed native failure records")
        if not all(isinstance(receipt, Mapping) for receipt in window_receipts):
            raise _fail("WDL/Jungle/EED result has malformed native failure records")
        surfaces = window_receipts

    for surface in surfaces:
        sentinel = surface.get("first_failure_no_failure_sentinel")
        failures = surface.get("first_failure_by_cell")
        if type(sentinel) is not int or not isinstance(failures, list):
            raise _fail("WDL/Jungle/EED result has malformed native failure records")
        _require_no_native_failure_records_v1(failures, sentinel=sentinel)


def _require_no_native_failure_records_v1(
    failures: list[Any], *, sentinel: int
) -> None:
    for failure in failures:
        if not isinstance(failure, Mapping):
            raise _fail("WDL/Jungle/EED result has malformed native failure records")
        status = failure.get("status")
        if type(status) is not int:
            raise _fail("WDL/Jungle/EED result has malformed native failure status")
        if status not in {0, sentinel}:
            diagnostic = ""
            if (
                status == 1
                and failure.get("stage") == 1
                and type(failure.get("tick_core_predicate_mask")) is int
            ):
                diagnostic = (
                    " tick_core_predicate_mask="
                    f"{failure['tick_core_predicate_mask']}"
                    " tick_core_predicate_names="
                    f"{failure.get('tick_core_predicate_names', [])!r}"
                )
            raise _fail(
                "WDL/Jungle/EED native execution failed: "
                f"cell={failure.get('cell')!r} "
                f"generation={failure.get('generation')!r} "
                f"tick={failure.get('tick')!r} status={status}{diagnostic}"
            )


_INHERITANCE_SIDECAR_FIELDS_V1: Final[tuple[str, ...]] = (
    "diploid_checkpoint_projection",
    "genome_upgrade_v2_checkpoint_projection",
)


def _sidecar_identity_v1(
    payload: Mapping[str, Any],
    *,
    base_projection: Mapping[str, Any],
    metadata: Mapping[str, Any],
    cell_id: str,
) -> dict[str, Any]:
    raw_payload = dict(payload)
    raw_payload.pop("identity", None)
    return {
        "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
        "cell_id": cell_id,
        "seed": metadata.get("seed"),
        "generation": metadata.get("generation"),
        "tick": metadata.get("tick"),
        "base_projection_sha256": projection_digest(base_projection),
        "payload_sha256": inheritance_sidecar_payload_sha256_v1(raw_payload),
    }

def _rebind_hapb_restored_inheritance_sidecars_v1(
    projection: Mapping[str, Any],
    *,
    pre_hapb_metadata: Mapping[str, Any],
    cell_id: str,
) -> dict[str, Any]:
    """Verify the sealed pre-export identity, then bind the restored base."""

    current = dict(projection)
    metadata = current.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("HAPB restored checkpoint lacks metadata")
    pre_hapb_base = dict(current)
    pre_hapb_base["metadata"] = dict(pre_hapb_metadata)
    restored_base = dict(current)
    # The checkpoint digest/schema/rehydration markers authenticate HAPB
    # transport only.  They are deliberately absent from the canonical next
    # generation projection, so they must not become part of an inheritance
    # sidecar identity that survives the restore boundary.
    restored_base["metadata"] = dict(pre_hapb_metadata)
    for name in _INHERITANCE_SIDECAR_FIELDS_V1:
        pre_hapb_base.pop(name, None)
        restored_base.pop(name, None)
    for name in _INHERITANCE_SIDECAR_FIELDS_V1:
        payload = current.get(name)
        if payload is None:
            continue
        if not isinstance(payload, Mapping) or not isinstance(payload.get("identity"), Mapping):
            raise _fail(f"HAPB restored {name} sidecar identity is missing")
        expected_pre_hapb = _sidecar_identity_v1(
            payload,
            base_projection=pre_hapb_base,
            metadata=pre_hapb_metadata,
            cell_id=cell_id,
        )
        if dict(payload["identity"]) != expected_pre_hapb:
            raise _fail(f"HAPB restored {name} sidecar pre-export identity differs")
        rebound = dict(payload)
        rebound["identity"] = _sidecar_identity_v1(
            rebound,
            base_projection=restored_base,
            metadata=pre_hapb_metadata,
            cell_id=cell_id,
        )
        current[name] = rebound
    return current

def export_direct_resident_wdl_jungle_hapb_checkpoint_v1(
    prepared: "PreparedDirectResidentWdlJungleV1",
    checkpoint_projections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze one HAPB host bank alongside the ordinary B48 checkpoints.

    The normal checkpoint projections deliberately omit ``mlp_pack`` here:
    the single host-policy sidecar is its sole authority.  Its digest is
    repeated in projection metadata so a projection cannot be resumed against
    a different bank, but the nearly-1GiB bytes are serialized exactly once.
    """

    if not isinstance(prepared, PreparedDirectResidentWdlJungleV1):
        raise _fail("HAPB checkpoint export requires a prepared WDL/Jungle owner")
    try:
        hapb_physical_cell_count_v1(len(prepared.cell_ids))
    except HAPBContractError as exc:
        raise _fail("HAPB checkpoint export requires physical B in [1, 256]") from exc
    if len(checkpoint_projections) != len(prepared.cell_ids):
        raise _fail("HAPB checkpoint export requires one physical batch")
    owner = prepared.patch.source.live_banks.get("hapb_policy_owner")
    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise _fail("HAPB checkpoint export lacks its host-authoritative policy owner")
    policy_checkpoint = export_hapb_policy_bank_checkpoint_v1(owner)
    policy_sha256 = policy_checkpoint["sha256"]
    if not isinstance(policy_sha256, str):
        raise _fail("HAPB checkpoint export produced no canonical policy digest")
    normalized: list[dict[str, Any]] = []
    for cell, projection in enumerate(checkpoint_projections):
        if not isinstance(projection, Mapping):
            raise _fail(f"HAPB checkpoint projection {cell} is not a mapping")
        metadata = dict(projection.get("metadata") or {})
        tensors = dict(projection.get("tensors") or {})
        # A tiny ABI sentinel is neither canonical policy state nor a valid
        # continuation source.  The sidecar below is the one authority.
        tensors.pop("mlp_pack", None)
        metadata["hapb_policy_checkpoint_sha256"] = policy_sha256
        metadata["hapb_policy_checkpoint_schema_version"] = (
            policy_checkpoint["schema_version"]
        )
        normalized.append({
            **{
                str(name): value
                for name, value in dict(projection).items()
                if name not in {"metadata", "tensors"}
            },
            "metadata": metadata,
            "tensors": tensors,
        })
    return {
        "schema_version": HAPB_DIRECT_CHECKPOINT_SCHEMA_V1,
        "cell_ids": list(prepared.cell_ids),
        "checkpoint_projections": normalized,
        "hapb_policy_bank": policy_checkpoint,
    }

def restore_direct_resident_wdl_jungle_hapb_checkpoint_sources_v1(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Restore source projections with views of one pageable canonical bank.

    This function intentionally performs no CUDA allocation.  The allocator
    recognizes the contiguous B48 views and adopts their shared storage rather
    than stacking a second host bank; cache/tile rehydration remains explicit.
    """

    if not isinstance(checkpoint, Mapping):
        raise _fail("HAPB checkpoint restore payload must be a mapping")
    if checkpoint.get("schema_version") != HAPB_DIRECT_CHECKPOINT_SCHEMA_V1:
        raise _fail("HAPB checkpoint restore schema is incompatible")
    cell_ids = checkpoint.get("cell_ids")
    projections = checkpoint.get("checkpoint_projections")
    policy_checkpoint = checkpoint.get("hapb_policy_bank")
    if (
        not isinstance(cell_ids, list)
        or any(type(cell_id) is not str or not cell_id for cell_id in cell_ids)
        or not isinstance(projections, list)
        or len(projections) != len(cell_ids)
    ):
        raise _fail("HAPB checkpoint restore requires one ordered physical batch")
    try:
        hapb_physical_cell_count_v1(len(cell_ids))
    except HAPBContractError as exc:
        raise _fail("HAPB checkpoint restore requires physical B in [1, 256]") from exc
    canonical = restore_hapb_policy_bank_checkpoint_v1(policy_checkpoint)
    policy_sha256 = policy_checkpoint.get("sha256") if isinstance(policy_checkpoint, dict) else None
    restored: list[Mapping[str, Any]] = []
    for cell, projection in enumerate(projections):
        if not isinstance(projection, Mapping):
            raise _fail(f"HAPB checkpoint projection {cell} is not a mapping")
        metadata = dict(projection.get("metadata") or {})
        if metadata.get("hapb_policy_checkpoint_sha256") != policy_sha256:
            raise _fail(f"HAPB checkpoint projection {cell} has a stale policy digest")
        metadata["hapb_policy_restore_pending_cache_rehydrate"] = True
        tensors = dict(projection.get("tensors") or {})
        if "mlp_pack" in tensors:
            raise _fail("HAPB checkpoint projection retained a duplicate MLP authority")
        tensors["mlp_pack"] = canonical[cell]
        pre_hapb_metadata = dict(metadata)
        for name in (
            "hapb_policy_checkpoint_sha256",
            "hapb_policy_checkpoint_schema_version",
            "hapb_policy_restore_pending_cache_rehydrate",
        ):
            pre_hapb_metadata.pop(name, None)
        rebound = _rebind_hapb_restored_inheritance_sidecars_v1({
            **{
                str(name): value
                for name, value in dict(projection).items()
                if name not in {"metadata", "tensors"}
            },
            "metadata": metadata,
            "tensors": tensors,
        }, pre_hapb_metadata=pre_hapb_metadata, cell_id=cell_ids[cell])
        restored.append(rebound)
    return tuple(restored)

def export_direct_resident_wdl_jungle_hapb_component_checkpoint_v1(
    prepared: "PreparedDirectResidentWdlJungleV1",
    *,
    absolute_generation: int | None = None,
    absolute_tick: int | None = None,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Persist the one policy authority used by the engine checkpoint owner."""

    if tuple(cell_ids) != prepared.cell_ids:
        raise _fail("HAPB component checkpoint cell order is incompatible")
    try:
        hapb_physical_cell_count_v1(len(cell_ids))
    except HAPBContractError as exc:
        raise _fail("HAPB component checkpoint requires physical B in [1, 256]") from exc
    if (absolute_generation is None) == (absolute_tick is None):
        raise _fail(
            "HAPB component checkpoint requires exactly one time coordinate"
        )
    if absolute_tick is not None:
        if type(absolute_tick) is not int or absolute_tick < 0:
            raise _fail("HAPB component checkpoint absolute tick is invalid")
    elif type(absolute_generation) is not int or absolute_generation < 0:
        raise _fail("HAPB component checkpoint generation is invalid")
    owner = prepared.patch.source.live_banks.get("hapb_policy_owner")
    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise _fail("HAPB component checkpoint lacks its host-policy owner")
    checkpoint = {
        "schema_version": HAPB_DIRECT_COMPONENT_CHECKPOINT_SCHEMA_V1,
        "cell_ids": list(cell_ids),
        "hapb_policy_bank": export_hapb_policy_bank_checkpoint_v1(owner),
    }
    if absolute_tick is not None:
        checkpoint.update(
            {
                "time_authority": "absolute_tick_v1",
                "absolute_tick": absolute_tick,
                "next_absolute_tick": absolute_tick,
            }
        )
    else:
        checkpoint["absolute_generation"] = absolute_generation
    return checkpoint

def restore_direct_resident_wdl_jungle_hapb_component_sources_v1(
    source_projections: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    *,
    cell_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Inject one restored bank into the ordinary per-cell source boundary."""

    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version")
        != HAPB_DIRECT_COMPONENT_CHECKPOINT_SCHEMA_V1
        or checkpoint.get("cell_ids") != list(cell_ids)
        or len(source_projections) != len(cell_ids)
    ):
        raise _fail("HAPB component checkpoint source contract is incompatible")
    try:
        hapb_physical_cell_count_v1(len(cell_ids))
    except HAPBContractError as exc:
        raise _fail("HAPB component checkpoint requires physical B in [1, 256]") from exc
    policy_checkpoint = checkpoint.get("hapb_policy_bank")
    policy_sha256 = (
        policy_checkpoint.get("sha256")
        if isinstance(policy_checkpoint, Mapping)
        else None
    )
    if not isinstance(policy_sha256, str):
        raise _fail("HAPB component checkpoint lacks its policy digest")
    normalized = []
    for projection in source_projections:
        if not isinstance(projection, Mapping):
            raise _fail("HAPB component source projection is not a mapping")
        metadata = dict(projection.get("metadata") or {})
        metadata["hapb_policy_checkpoint_sha256"] = policy_sha256
        normalized.append({
            **{
                str(name): value
                for name, value in dict(projection).items()
                if name != "metadata"
            },
            "metadata": metadata,
        })
    return restore_direct_resident_wdl_jungle_hapb_checkpoint_sources_v1({
        "schema_version": HAPB_DIRECT_CHECKPOINT_SCHEMA_V1,
        "cell_ids": list(cell_ids),
        "checkpoint_projections": normalized,
        "hapb_policy_bank": policy_checkpoint,
    })

def restore_direct_resident_wdl_jungle_hapb_component_checkpoint_v1(
    prepared: "PreparedDirectResidentWdlJungleV1",
    checkpoint: Mapping[str, Any],
    *,
    absolute_generation: int | None = None,
    absolute_tick: int | None = None,
    cell_ids: Sequence[str],
) -> int:
    """Verify restored authority, then explicitly rehydrate its bounded cache."""

    common_valid = (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version")
        != HAPB_DIRECT_COMPONENT_CHECKPOINT_SCHEMA_V1
        or checkpoint.get("cell_ids") != list(cell_ids)
        or tuple(cell_ids) != prepared.cell_ids
    )
    if (absolute_generation is None) == (absolute_tick is None):
        raise _fail(
            "HAPB component restore requires exactly one time coordinate"
        )
    if absolute_tick is not None:
        checkpoint_tick = checkpoint.get("next_absolute_tick")
        if checkpoint_tick is None:
            checkpoint_tick = checkpoint.get("absolute_tick")
        valid_coordinate = (
            type(absolute_tick) is int
            and absolute_tick >= 0
            and checkpoint.get("time_authority") == "absolute_tick_v1"
            and checkpoint.get("absolute_tick") == absolute_tick
            and checkpoint_tick == absolute_tick
        )
    else:
        valid_coordinate = checkpoint.get("absolute_generation") == absolute_generation
    if common_valid or not valid_coordinate:
        raise _fail("HAPB component checkpoint restore contract is incompatible")
    owner = prepared.patch.source.live_banks.get("hapb_policy_owner")
    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise _fail("HAPB component checkpoint restore lacks its host-policy owner")
    expected = checkpoint.get("hapb_policy_bank")
    if not isinstance(expected, Mapping):
        raise _fail("HAPB component checkpoint lacks its policy bank")
    actual = export_hapb_policy_bank_checkpoint_v1(owner)
    if actual.get("sha256") != expected.get("sha256"):
        raise _fail("HAPB component checkpoint restored different canonical policy bytes")
    try:
        restored_owner = restore_hapb_policy_owner_transient_state_v1(
            owner, expected
        )
        prepared.patch.source.live_banks["hapb_policy_owner"] = (
            restored_owner.rehydrate_hot_prefix_cache_v1()
        )
    except (HAPBContractError, KeyError, TypeError) as exc:
        raise _fail("HAPB component checkpoint transient state is incompatible") from exc
    return absolute_tick if absolute_tick is not None else absolute_generation

def patch_envelope_with_wdl_jungle_masked_v1(
    runner: Any,
    *,
    allow_short_technical_gate_generation_count: bool = False,
    eed_overlay: bool = False,
    tick_native_control: Mapping[str, Any] | None = None,
) -> CudaArchitectureBRunnerEnvelope:
    """Classify the base Patch envelope without suppressing overlay metadata."""

    proxy = copy.copy(runner)
    proxy.env_cfg = copy.copy(runner.env_cfg)
    proxy.env_cfg.wdl_delayed_hazard_enable = False
    proxy.env_cfg.wdl_delayed_hazard_observation_enable = False
    for name in (
        "jungle_predator_pressure_enable",
        "jungle_reproduction_emissions_enable",
        "jungle_juvenile_predator_vulnerability_enable",
        "jungle_terrain_protection_enable",
        "jungle_ecological_recovery_enable",
    ):
        setattr(proxy.env_cfg, name, False)
    if eed_overlay:
        for name in (
            "respec0b_resource_specialization_enable",
            "eed_resource_impact_enable",
            "eed_resource_impact_apply_enable",
            "eed_ecology_frozen_enable",
            "eed_phenotype_fixed_enable",
        ):
            setattr(proxy.env_cfg, name, False)
    return derive_cuda_architecture_b_runner_envelope(
        proxy,
        allow_short_technical_gate_generation_count=(
            allow_short_technical_gate_generation_count
        ),
        tick_native_control=tick_native_control,
    )


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentWdlJungleV1:
    """One canonical Patch/WDL source plus Jungle-only overlay allocations."""

    wdl: PreparedDirectResidentWdlV2
    jungle_state_tensors: tuple[torch.Tensor, ...]
    jungle_reporting_tensors: tuple[torch.Tensor, ...]
    jungle_config: dict[str, Any]
    eed_state_tensors: tuple[torch.Tensor, ...] | None = None
    eed_config: dict[str, Any] | None = None
    # Opaque native-only identity state. It starts invalid, persists across
    # same-owner entries, and is absent from every checkpoint projection.
    policy_window_identity_owner: Any | None = None

    @property
    def patch(self) -> Any:
        return self.wdl.patch

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self.wdl.cell_ids

    @property
    def source_projections(self) -> tuple[Any, ...]:
        return self.wdl.source_projections

    @property
    def metadata_rows(self) -> tuple[Mapping[str, Any], ...]:
        """Forward compact per-cell source metadata for tick-native prefixes."""

        return self.wdl.metadata_rows

    @property
    def runtime_prefix_rows(self) -> tuple[Mapping[str, Any], ...]:
        """Forward scalar continuation prefixes without retaining tensors."""

        return self.wdl.runtime_prefix_rows

    @property
    def source_projection_cursor_factory(self) -> Callable[[], Any] | None:
        """Expose the bounded source cursor through the Jungle wrapper.

        Streamed physical owners intentionally keep no B-wide projection tuple.
        The campaign provider therefore discovers the lazy source importer on
        the physical-window argument wrapper.  Keep this as a forwarding
        property so adding the Jungle/EED overlay does not erase the bounded
        cursor contract or force host staging.
        """

        return self.wdl.source_projection_cursor_factory


_POLICY_RNG_STATE_INDEX_V1: Final[int] = 85
_POLICY_DELTA_TAPE_INDEX_V1: Final[int] = 86
_POLICY_MUTATION_MASK_TAPE_INDEX_V1: Final[int] = 87
_POLICY_CROSSOVER_TAPE_INDEX_V1: Final[int] = 88
_POLICY_POST_STATE_TAPE_INDEX_V1: Final[int] = 89
_POLICY_SEMANTIC_TAPE_INDEX_V1: Final[int] = 90
_POLICY_ABSOLUTE_CURSOR_INDEX_V1: Final[int] = 91
_POLICY_SEMANTIC_COUNTER_INDEX_V1: Final[int] = 92
_POLICY_WINDOW_BASE_INDEX_V1: Final[int] = 111
_POLICY_WINDOW_IDENTITY_TENSOR_FIELDS_V1: Final[tuple[tuple[str, int], ...]] = (
    ("source_state", _POLICY_RNG_STATE_INDEX_V1),
    ("delta", _POLICY_DELTA_TAPE_INDEX_V1),
    ("mutation_mask", _POLICY_MUTATION_MASK_TAPE_INDEX_V1),
    ("crossover_mask_packed", _POLICY_CROSSOVER_TAPE_INDEX_V1),
    ("post_state_packed", _POLICY_POST_STATE_TAPE_INDEX_V1),
    ("semantic_tape", _POLICY_SEMANTIC_TAPE_INDEX_V1),
    ("absolute_cursor", _POLICY_ABSOLUTE_CURSOR_INDEX_V1),
    ("semantic_counter", _POLICY_SEMANTIC_COUNTER_INDEX_V1),
    ("window_base", _POLICY_WINDOW_BASE_INDEX_V1),
)


def _frag_identity_reuse_requested_v1() -> bool:
    return os.environ.get("EIEL_HAPB_FRAG_A_ENABLE_POLICY_WINDOW_REUSE") == "1"


def _create_untrusted_policy_window_identity_owner_v1(
    module: Any,
    wdl: PreparedDirectResidentWdlV2,
    *,
    checkpoint_reconstructed: bool,
) -> Any:
    factory = getattr(
        module,
        "eiel_cuda_architecture_b_hapb_create_untrusted_policy_window_identity_owner_v1",
        None,
    )
    if not callable(factory):
        raise _fail("HAPB FRAG identity build lacks its native owner factory")
    state = wdl.patch.source.ordinary_birth_state_tensors
    if len(state) <= _POLICY_WINDOW_BASE_INDEX_V1:
        raise _fail("HAPB FRAG identity build lacks its policy-window tensor ABI")
    metadata = wdl.initial_projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("HAPB FRAG identity build lacks source metadata")
    policy_row_width = metadata.get("policy_param_count")
    if type(policy_row_width) is not int or policy_row_width <= 0:
        policy_row_width = 9_991
    boundary = (
        "checkpoint_reconstructed_allocator_owner"
        if checkpoint_reconstructed
        else "fresh_allocator_owner"
    )
    return factory(
        state[_POLICY_RNG_STATE_INDEX_V1],
        state[_POLICY_DELTA_TAPE_INDEX_V1],
        state[_POLICY_MUTATION_MASK_TAPE_INDEX_V1],
        state[_POLICY_CROSSOVER_TAPE_INDEX_V1],
        state[_POLICY_POST_STATE_TAPE_INDEX_V1],
        state[_POLICY_SEMANTIC_TAPE_INDEX_V1],
        state[_POLICY_ABSOLUTE_CURSOR_INDEX_V1].reshape(-1),
        state[_POLICY_SEMANTIC_COUNTER_INDEX_V1].reshape(-1),
        state[_POLICY_WINDOW_BASE_INDEX_V1].reshape(-1),
        policy_row_width,
        boundary,
    )


def _commit_policy_window_identity_owner_entry_v1(
    module: Any,
    owner: Any | None,
    validation_token: int | None,
) -> None:
    if owner is None:
        if validation_token is not None:
            raise _fail("HAPB FRAG identity returned a token without an owner")
        return
    if type(validation_token) is not int or validation_token <= 0:
        raise _fail("HAPB FRAG identity result lacks its exact validation token")
    commit = getattr(
        module,
        HAPB_POLICY_WINDOW_IDENTITY_COMMIT_ENTRYPOINT_V1,
        None,
    )
    if not callable(commit):
        raise _fail("HAPB FRAG identity build lacks its validation commit")
    commit(owner, validation_token)


def export_hapb_policy_window_identity_checkpoint_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
) -> dict[str, Any]:
    """Persist native identity state and its exact policy-window contents."""

    owner = prepared.policy_window_identity_owner
    export = getattr(module, HAPB_POLICY_WINDOW_IDENTITY_EXPORT_ENTRYPOINT_V1, None)
    if owner is None or not callable(export):
        raise _fail("HAPB identity checkpoint export lacks its native owner")
    raw_state = export(owner)
    if not isinstance(raw_state, Mapping):
        raise _fail("HAPB identity checkpoint export returned no state mapping")
    state = {str(key): value for key, value in raw_state.items()}
    configuration = state.get("configuration")
    if not isinstance(configuration, Mapping):
        raise _fail("HAPB identity checkpoint export lacks configuration")
    state["configuration"] = {
        str(key): value for key, value in configuration.items()
    }
    source_state = prepared.patch.source.ordinary_birth_state_tensors
    tensors: dict[str, torch.Tensor] = {}
    for name, index in _POLICY_WINDOW_IDENTITY_TENSOR_FIELDS_V1:
        if index >= len(source_state) or not isinstance(source_state[index], torch.Tensor):
            raise _fail(f"HAPB identity checkpoint export lacks tensor {name}")
        tensors[name] = source_state[index].detach().cpu().contiguous().clone()
    return {
        "schema_version": HAPB_POLICY_WINDOW_IDENTITY_CHECKPOINT_SCHEMA_V1,
        "time_authority": "absolute_tick_v1",
        "state": state,
        "tensors": tensors,
    }


def restore_hapb_policy_window_identity_checkpoint_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
    checkpoint: Mapping[str, Any],
) -> None:
    """Rebind a fresh owner after restoring the exact native policy window."""

    owner = prepared.policy_window_identity_owner
    restore = getattr(module, HAPB_POLICY_WINDOW_IDENTITY_RESTORE_ENTRYPOINT_V1, None)
    if owner is None or not callable(restore):
        raise _fail("HAPB identity checkpoint restore lacks its native owner")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != HAPB_POLICY_WINDOW_IDENTITY_CHECKPOINT_SCHEMA_V1
        or checkpoint.get("time_authority") != "absolute_tick_v1"
    ):
        raise _fail("HAPB identity checkpoint schema is incompatible")
    state = checkpoint.get("state")
    tensors = checkpoint.get("tensors")
    if not isinstance(state, Mapping) or not isinstance(tensors, Mapping):
        raise _fail("HAPB identity checkpoint lacks state or tensors")
    configuration = state.get("configuration")
    policy_row_width = (
        configuration.get("policy_row_width")
        if isinstance(configuration, Mapping)
        else None
    )
    if type(policy_row_width) is not int or policy_row_width <= 0:
        raise _fail("HAPB identity checkpoint lacks its policy row width")
    source_state = prepared.patch.source.ordinary_birth_state_tensors
    restored: dict[str, torch.Tensor] = {}
    for name, index in _POLICY_WINDOW_IDENTITY_TENSOR_FIELDS_V1:
        saved = tensors.get(name)
        if not isinstance(saved, torch.Tensor) or saved.device.type != "cpu":
            raise _fail(f"HAPB identity checkpoint tensor {name} is not CPU-bound")
        if index >= len(source_state) or not isinstance(source_state[index], torch.Tensor):
            raise _fail(f"HAPB identity restore lacks target tensor {name}")
        target = source_state[index]
        if saved.dtype != target.dtype or tuple(saved.shape) != tuple(target.shape):
            raise _fail(f"HAPB identity checkpoint tensor {name} shape or dtype differs")
        with torch.no_grad():
            target.copy_(saved.to(device=target.device), non_blocking=False)
        restored[name] = target
    restore(
        owner,
        restored["source_state"],
        restored["delta"],
        restored["mutation_mask"],
        restored["crossover_mask_packed"],
        restored["post_state_packed"],
        restored["semantic_tape"],
        restored["absolute_cursor"].reshape(-1),
        restored["semantic_counter"].reshape(-1),
        restored["window_base"].reshape(-1),
        policy_row_width,
        dict(state),
    )


def _take_policy_window_identity_validation_token_v1(
    result: dict[str, Any],
    owner: Any | None,
) -> int | None:
    token = result.pop(HAPB_POLICY_WINDOW_IDENTITY_VALIDATION_TOKEN_FIELD_V1, None)
    if owner is None:
        if token is not None:
            raise _fail("HAPB FRAG identity returned a token without an owner")
        return None
    if type(token) is not int or token <= 0:
        raise _fail("HAPB FRAG identity result lacks its exact validation token")
    return token


def _checkpoint_reconstructed_policy_window_identity_v1(
    metadata_rows: Sequence[Mapping[str, Any]],
) -> bool:
    flags = tuple(
        row.get("hapb_policy_restore_pending_cache_rehydrate") is True
        for row in metadata_rows
    )
    if not flags:
        raise _fail("HAPB FRAG identity requires a nonempty source batch")
    if any(flags) and not all(flags):
        raise _fail(
            "HAPB FRAG identity source batch mixes checkpoint and fresh owners"
        )
    return all(flags)


def build_prepared_direct_resident_wdl_jungle_v1(
    *,
    qualified_cuda_module: Any,
    generation_rows: int,
    initial_generation: int,
    per_cell_requests: Sequence[Any] | None = None,
    source_checkpoint_projections: Sequence[Mapping[str, Any]] | None = None,
    cell_ids: Sequence[str],
    agent_id_allocation_generation_count: int | None = None,
    allow_short_technical_gate_generation_count: bool = False,
    allow_partial_generation_source_cursor: bool = False,
    eed_overlay: bool = False,
    hapb_host_policy_enable: bool = False,
    per_cell_request_factory: Callable[[int], Any] | None = None,
    source_projection_cursor_factory: Callable[[], Any] | None = None,
    source_import_cursor_receipt: Mapping[str, Any] | None = None,
    source_capability_mask: Mapping[str, bool] | None = None,
) -> PreparedDirectResidentWdlJungleV1:
    """Prepare one Patch/WDL source and explicit overlays."""

    if per_cell_request_factory is not None:
        if per_cell_requests is not None or source_checkpoint_projections is not None:
            raise _fail(
                "streamed WDL/Jungle preparation cannot also receive request tuples"
            )
        if hapb_host_policy_enable:
            raise _fail("HAPB host-policy preparation is not admitted on the streamed route")
        ordered_ids = tuple(cell_ids)
        if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
            raise _fail("streamed WDL/Jungle cell IDs must be unique")
        _validated_jungle_capability(qualified_cuda_module)
        extra_names = (*WDL_STATE_FIELDS_V2, "patch_visibility")
        if eed_overlay:
            extra_names += EED_STATE_FIELDS_V1
        wdl = build_prepared_direct_resident_wdl_v2_streaming(
            request_factory=per_cell_request_factory,
            qualified_cuda_module=qualified_cuda_module,
            generation_rows=generation_rows,
            initial_generation=initial_generation,
            cell_ids=ordered_ids,
            agent_id_allocation_generation_count=(
                agent_id_allocation_generation_count
            ),
            allow_short_technical_gate_generation_count=(
                allow_short_technical_gate_generation_count
            ),
            allow_partial_generation_source_cursor=(
                allow_partial_generation_source_cursor
            ),
            source_projection_cursor_factory=source_projection_cursor_factory,
            source_import_cursor_receipt=source_import_cursor_receipt,
            source_capability_mask=source_capability_mask,
            jungle_overlay=True,
            eed_overlay=eed_overlay,
            extra_tensor_names=extra_names,
        )
        metadata_rows = tuple(wdl.metadata_rows)
        if len(metadata_rows) != len(ordered_ids):
            raise _fail("streamed WDL/Jungle metadata count differs from cells")
        extra = dict(wdl.streamed_extra_tensors or {})
        visibility = extra.get("patch_visibility")
        recovery = wdl.patch.source.live_state.get("patch_recovery_activity")
        if not isinstance(visibility, torch.Tensor) or visibility.dtype != torch.float32:
            raise _fail("streamed WDL/Jungle source lacks patch_visibility")
        if not isinstance(recovery, torch.Tensor) or recovery.dtype != torch.float32:
            raise _fail("streamed WDL/Jungle source lacks recovery state")
        if visibility.shape != recovery.shape:
            raise _fail("streamed WDL/Jungle visibility does not match recovery state")
        live_banks = dict(wdl.patch.live_banks)
        live_banks["patch_visibility"] = visibility
        source = replace(wdl.patch.source, live_banks=live_banks)
        wdl = replace(
            wdl,
            patch=replace(
                wdl.patch,
                source=source,
                live_banks=source.live_banks,
            ),
        )
        ticks_per_generation = wdl.config.get("ticks_per_generation")
        if ticks_per_generation not in JUNGLE_SUPPORTED_TICKS_PER_GENERATION_V4:
            raise _fail("canonical streamed WDL ticks_per_generation is unsupported")
        jungle_config_rows = [
            _normalized_jungle_config(
                _required_jungle_metadata(metadata),
                ticks_per_generation=ticks_per_generation,
            )
            for metadata in metadata_rows
        ]
        jungle_config = dict(jungle_config_rows[0])
        shared_jungle_config = {
            name: value
            for name, value in jungle_config.items()
            if name not in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1
        }
        if any(
            {
                name: value
                for name, value in row.items()
                if name not in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1
            }
            != shared_jungle_config
            for row in jungle_config_rows[1:]
        ):
            raise _fail("mixed streamed Jungle mechanics config is not representable")
        experiment_keys_present = sum(
            name in jungle_config for name in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1
        )
        if experiment_keys_present not in {
            0,
            len(_JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1),
        }:
            raise _fail("streamed Jungle experiment controls are incomplete")
        if experiment_keys_present:
            for name in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1:
                jungle_config[f"{name}_by_cell"] = torch.tensor(
                    [float(row[name]) for row in jungle_config_rows],
                    dtype=torch.float64,
                    device=recovery.device,
                )
        cells, capacity = int(recovery.size(0)), int(recovery.size(1))
        reporting = _allocate_reporting(
            device=recovery.device,
            cells=cells,
            height=int(recovery.size(1)),
            width=int(recovery.size(2)),
            capacity=capacity,
            logical_generation_count=generation_rows,
            ticks_per_generation=ticks_per_generation,
        )
        eed_kwargs = (
            _eed_context_from_requests_v1(
                None,
                metadata_rows,
                device=recovery.device,
                streamed_state_tensors={
                    name: extra[name] for name in EED_STATE_FIELDS_V1
                },
            )
            if eed_overlay
            else {}
        )
        return PreparedDirectResidentWdlJungleV1(
            wdl=wdl,
            jungle_state_tensors=(recovery, visibility),
            jungle_reporting_tensors=reporting,
            jungle_config=jungle_config,
            **eed_kwargs,
        )

    if hapb_host_policy_enable:
        try:
            hapb_physical_cell_count_v1(len(per_cell_requests))
        except HAPBContractError as exc:
            raise _fail(
                "HAPB WDL/Jungle owner requires physical B in [1, 256]"
            ) from exc
    elif not 1 <= len(per_cell_requests) <= 256:
        raise _fail("shared WDL/Jungle owner requires B1..B256 requests")
    if len(source_checkpoint_projections) != len(per_cell_requests):
        raise _fail("WDL/Jungle source projections do not match the request count")
    if tuple(cell_ids) != tuple(str(value) for value in cell_ids):
        raise _fail("WDL/Jungle cell identifiers must be exact strings")
    _validated_jungle_capability(qualified_cuda_module)
    wdl = build_prepared_direct_resident_wdl_v2(
        None,
        qualified_cuda_module=qualified_cuda_module,
        generation_rows=generation_rows,
        initial_generation=initial_generation,
        per_cell_requests=per_cell_requests,
        source_checkpoint_projections=source_checkpoint_projections,
        cell_ids=cell_ids,
        agent_id_allocation_generation_count=agent_id_allocation_generation_count,
        allow_short_technical_gate_generation_count=(
            allow_short_technical_gate_generation_count
        ),
        allow_partial_generation_source_cursor=(
            allow_partial_generation_source_cursor
        ),
        jungle_overlay=True,
        eed_overlay=eed_overlay,
        hapb_host_policy_enable=hapb_host_policy_enable,
    )
    ticks_per_generation = wdl.config.get("ticks_per_generation")
    if ticks_per_generation not in JUNGLE_SUPPORTED_TICKS_PER_GENERATION_V4:
        raise _fail("canonical WDL owner ticks_per_generation must be 80 or 160")
    if getattr(wdl.patch.source, "ticks_per_generation", None) != ticks_per_generation:
        raise _fail("canonical WDL source and WDL overlay ticks_per_generation differ")
    requests = tuple(per_cell_requests)
    jungle_config_rows: list[dict[str, Any]] = []
    metadata_rows: list[Mapping[str, Any]] = []
    for cell, _request in enumerate(requests):
        projection = source_checkpoint_projections[cell]
        metadata = projection.get("metadata")
        if not isinstance(metadata, Mapping):
            raise _fail(f"WDL/Jungle projection {cell} lacks metadata")
        metadata_rows.append(metadata)
        # Per-cell source metadata intentionally contains independent RNG and
        # source-projection identities.  Only the normalized Jungle mechanics
        # config crosses the shared native overlay boundary, so equality of the
        # full metadata map would incorrectly reject a legitimate B48 matrix.
        jungle_config_rows.append(
            _normalized_jungle_config(
                _required_jungle_metadata(metadata),
                ticks_per_generation=ticks_per_generation,
            )
        )
    recovery = wdl.patch.source.live_state.get("patch_recovery_activity")
    if not isinstance(recovery, torch.Tensor) or recovery.dtype != torch.float32:
        raise _fail("canonical Patch/WDL owner lacks patch_recovery_activity")
    visibility = wdl.patch.live_banks.get("patch_visibility")
    if not isinstance(visibility, torch.Tensor) or visibility.dtype != torch.float32:
        raise _fail("canonical WDL/Jungle owner lacks CUDA patch_visibility")
    if visibility.shape != recovery.shape:
        raise _fail("Jungle visibility does not match canonical Patch recovery storage")
    if (
        recovery.data_ptr()
        != wdl.patch.source.live_state["patch_recovery_activity"].data_ptr()
    ):
        raise _fail("Jungle recovery must reuse canonical Patch/WDL storage")
    jungle_config = dict(jungle_config_rows[0])
    shared_jungle_config = {
        name: value
        for name, value in jungle_config.items()
        if name not in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1
    }
    if any(
        {
            name: value
            for name, value in row.items()
            if name not in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1
        }
        != shared_jungle_config
        for row in jungle_config_rows[1:]
    ):
        raise _fail("mixed per-cell Jungle mechanics config is not representable")
    experiment_keys_present = sum(
        name in jungle_config for name in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1
    )
    if experiment_keys_present not in {0, len(_JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1)}:
        raise _fail("Jungle experiment controls must be present as one complete set")
    if experiment_keys_present:
        for name in _JUNGLE_EXPERIMENT_FLOAT_BY_CELL_FIELDS_V1:
            jungle_config[f"{name}_by_cell"] = torch.tensor(
                [float(row[name]) for row in jungle_config_rows],
                dtype=torch.float64,
                device=recovery.device,
            )
    cells, capacity = (
        int(recovery.size(0)),
        int(wdl.patch.source.live_state["energy"].size(1)),
    )
    reporting = _allocate_reporting(
        device=recovery.device,
        cells=cells,
        height=int(recovery.size(1)),
        width=int(recovery.size(2)),
        capacity=capacity,
        logical_generation_count=generation_rows,
        ticks_per_generation=ticks_per_generation,
    )
    policy_window_identity_owner = (
        _create_untrusted_policy_window_identity_owner_v1(
            qualified_cuda_module,
            wdl,
            checkpoint_reconstructed=(
                _checkpoint_reconstructed_policy_window_identity_v1(
                    metadata_rows
                )
            ),
        )
        if hapb_host_policy_enable and _frag_identity_reuse_requested_v1()
        else None
    )
    return PreparedDirectResidentWdlJungleV1(
        wdl=wdl,
        jungle_state_tensors=(recovery, visibility),
        jungle_reporting_tensors=reporting,
        jungle_config=jungle_config,
        policy_window_identity_owner=policy_window_identity_owner,
        **(
            _eed_context_from_requests_v1(
                requests,
                metadata_rows,
                device=recovery.device,
            )
            if eed_overlay
            else {}
        ),
    )


def _eed_context_from_requests_v1(
    requests: Sequence[Any] | None,
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    streamed_state_tensors: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Materialize EED overlay tensors beside, never instead of, WDL/Jungle."""

    if (
        streamed_state_tensors is None
        and (requests is None or len(requests) != len(metadata_rows) or not requests)
    ):
        raise _fail("EED overlay requests and metadata must have one row per cell")
    if not metadata_rows:
        raise _fail("EED overlay metadata must cover the physical batch")
    required = (
        "patch_food_presence_threshold",
        "patch_food_gather_amount",
        "patch_food_regrow_per_step",
        "respec0b_resource_a_handling_threshold",
        "respec0b_resource_b_handling_threshold",
        "respec0b_resource_a_return",
        "respec0b_resource_b_return",
        "respec0b_resource_equalized_sham_enable",
        "eed_resource_impact_apply_enable",
        "eed_ecology_frozen_enable",
        "eed_phenotype_fixed_enable",
        "eed_resource_impact_per_success",
        "eed_resource_impact_decay_per_step",
        "eed_resource_impact_max_load",
        "eed_phenotype_fixed_axis_profile",
    )
    first = metadata_rows[0]
    missing = [name for name in required if name not in first]
    if missing:
        raise _fail(f"EED overlay metadata is missing: {missing}")
    shared_required = tuple(
        name
        for name in required[:-1]
        if name
        not in {"patch_food_gather_amount", "patch_food_regrow_per_step"}
    )
    if any(
        any(row.get(name) != first[name] for row in metadata_rows[1:])
        for name in shared_required
    ):
        raise _fail("mixed EED mechanics config is not representable by one shared owner")
    if streamed_state_tensors is None:
        missing_tensors = {
            cell: [
                name
                for name in EED_STATE_FIELDS_V1
                if name not in request.cuda_boundary.tensors
            ]
            for cell, request in enumerate(requests or ())
        }
        missing_tensors = {
            cell: names for cell, names in missing_tensors.items() if names
        }
        if missing_tensors:
            raise _fail(f"EED CUDA boundaries are missing EED state: {missing_tensors}")
        state_tensors = tuple(
            torch.stack(
                tuple(request.cuda_boundary.tensors[name] for request in requests or ())
            )
            .to(device=device)
            .contiguous()
            for name in EED_STATE_FIELDS_V1
        )
    else:
        missing = [
            name for name in EED_STATE_FIELDS_V1 if name not in streamed_state_tensors
        ]
        if missing:
            raise _fail(f"streamed EED state is missing: {missing}")
        state_tensors = tuple(
            streamed_state_tensors[name].to(device=device).contiguous()
            for name in EED_STATE_FIELDS_V1
        )
    fixed_axis_modes = {"neutral": 0, "a_high": 1, "b_high": 2, "symmetric_mix": 3}
    try:
        fixed_axis_mode_by_cell = torch.tensor(
            [
                fixed_axis_modes[str(row["eed_phenotype_fixed_axis_profile"])]
                for row in metadata_rows
            ],
            dtype=torch.int32,
            device=device,
        )
    except KeyError as exc:
        raise _fail("EED fixed phenotype profile is unsupported") from exc
    return {
        "eed_state_tensors": state_tensors,
        "eed_config": {
            "enabled": True,
            "schema_version": EED_ABI_VERSION_V1,
            "food_presence_threshold": float(first["patch_food_presence_threshold"]),
            "gather_amount": float(first["patch_food_gather_amount"]),
            "gather_amount_by_cell": torch.tensor(
                [float(row["patch_food_gather_amount"]) for row in metadata_rows],
                dtype=torch.float64,
                device=device,
            ),
            "regrow_amount_by_cell": torch.tensor(
                [float(row["patch_food_regrow_per_step"]) for row in metadata_rows],
                dtype=torch.float64,
                device=device,
            ),
            "resource_a_handling_threshold": float(
                first["respec0b_resource_a_handling_threshold"]
            ),
            "resource_b_handling_threshold": float(
                first["respec0b_resource_b_handling_threshold"]
            ),
            "resource_a_return": float(first["respec0b_resource_a_return"]),
            "resource_b_return": float(first["respec0b_resource_b_return"]),
            "equalized_sham": bool(first["respec0b_resource_equalized_sham_enable"]),
            "impact_per_success": float(first["eed_resource_impact_per_success"]),
            "decay_per_step": float(first["eed_resource_impact_decay_per_step"]),
            "maximum_load": float(first["eed_resource_impact_max_load"]),
            "impact_applied": bool(first["eed_resource_impact_apply_enable"]),
            "ecology_frozen": bool(first["eed_ecology_frozen_enable"]),
            "phenotype_fixed": bool(first["eed_phenotype_fixed_enable"]),
            "fixed_axis_mode": int(fixed_axis_mode_by_cell[0].item()),
            "fixed_axis_mode_by_cell": fixed_axis_mode_by_cell,
        },
    }


def execute_direct_resident_wdl_jungle_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    logical_generation_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Dispatch exactly one non-EED native owner for all cells and overlays."""

    if not isinstance(prepared, PreparedDirectResidentWdlJungleV1):
        raise _fail("prepared must be PreparedDirectResidentWdlJungleV1")
    if (
        not WDL_JUNGLE_RESIDENT_GENERATION_MIN_V1
        <= logical_generation_count
        <= WDL_RESIDENT_GENERATION_MAX_V2
    ):
        raise _fail("shared WDL/Jungle logical generation count must be in [3,128]")
    if logical_generation_count > prepared.wdl.generation_rows:
        raise _fail("WDL/Jungle reporting allocation does not cover the trajectory")
    if prepared.patch.source.logical_generation_count != logical_generation_count:
        raise _fail("WDL/Jungle source identity capacity does not match the trajectory")
    eed_enabled = (
        prepared.eed_state_tensors is not None and prepared.eed_config is not None
    )
    entrypoint_name = (
        WDL_JUNGLE_EED_DIRECT_ENTRYPOINT_V1
        if eed_enabled
        else WDL_JUNGLE_DIRECT_ENTRYPOINT_V1
    )
    entrypoint = getattr(module, entrypoint_name, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {entrypoint_name}")
    metadata = prepared.wdl.initial_projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("WDL/Jungle initial projection metadata is not a mapping")
    control = create_run_control_v2(module, run_id=run_id)
    arguments = (
        control,
        list(prepared.wdl.state_tensors),
        list(prepared.wdl.schedule_tensors),
        list(prepared.wdl.reporting_tensors),
        dict(prepared.wdl.config),
        list(prepared.jungle_state_tensors),
        list(prepared.jungle_reporting_tensors),
        dict(prepared.jungle_config),
    )
    if eed_enabled:
        arguments += (list(prepared.eed_state_tensors), dict(prepared.eed_config))
    raw = entrypoint(
        *arguments,
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    if not isinstance(raw, Mapping):
        raise _fail("WDL/Jungle CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    _require_no_native_failure_v1(result)
    expected = {
        "wdl_jungle_native_owner": True,
        "wdl_native_controller_abi_version": WDL_ABI_VERSION_V3,
        "jungle_native_controller_abi_version": JUNGLE_ABI_VERSION_V4,
        "wdl_jungle_python_tick_or_generation_ownership": False,
        "completed_generations": logical_generation_count,
        "logical_generation_count": logical_generation_count,
        "physical_window_generations": (
            prepared.patch.source.physical_window_generations
        ),
        "window_count": resident_window_count_v1(
            initial_generation=prepared.patch.source.generation_base,
            logical_generation_count=logical_generation_count,
            physical_window_generations=(
                prepared.patch.source.physical_window_generations
            ),
        ),
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"WDL/Jungle resident result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    if eed_enabled:
        if (
            result.get("eed_native_owner") is not True
            or result.get("eed_native_controller_abi_version") != EED_ABI_VERSION_V1
        ):
            raise _fail("WDL/Jungle/EED resident result lacks the EED native owner")
    manifest = dict(control.manifest())
    if (
        manifest.get("status") != "generation_complete"
        or manifest.get("requested_generations") != logical_generation_count
        or manifest.get("active_generation")
        != prepared.patch.source.generation_base + logical_generation_count - 1
    ):
        raise _fail("WDL/Jungle native run-control is not at the final boundary")
    result["native_run_control_v2"] = manifest
    return result


def execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    span_ticks: int,
    start_tick: int | None = None,
    generation_base: int | None = None,
    absolute_start_next_tick: int | None = None,
    source_window_ticks: int | None = None,
) -> dict[str, Any]:
    """Execute one absolute span using a reusable bounded source/output window."""

    return _execute_direct_resident_wdl_jungle_eed_absolute_dispatch_v2(
        module,
        prepared,
        span_ticks=span_ticks,
        start_tick=start_tick,
        generation_base=generation_base,
        absolute_start_next_tick=absolute_start_next_tick,
        source_window_ticks=source_window_ticks,
        entrypoint_name=WDL_JUNGLE_EED_TICK_DISPATCH_ENTRYPOINT_V2,
    )


def _bind_tick_native_event_sink_arguments_v1(
    prepared: Any,
    arguments: list[Any],
) -> tuple[str, bool]:
    """Bind the authenticated event sink to the exact native ABI argument.

    The provider creates the sink authority before owner preparation, while
    the WDL/Jungle adapter rebuilds a direct allocation request inside that
    preparation. Keep the native boundary fail-closed, but recover a missing
    ABI copy from the already-authenticated source projection instead of
    silently falling back to a bounded Python archive.
    """

    source = getattr(getattr(prepared, "patch", None), "source", None)
    source_config = getattr(source, "patch_p2_config", None)
    if not isinstance(source_config, Mapping):
        raise _fail("tick-native source Patch P2 config is not a mapping")
    source_path = source_config.get("event_history_sink_path")
    if source_path is not None and (
        type(source_path) is not str or not source_path.strip()
    ):
        raise _fail("tick-native source event sink path is malformed")

    projection_paths: set[str] = set()
    for cell, projection in enumerate(
        tuple(getattr(prepared, "source_projections", ()) or ())
    ):
        metadata = (
            projection.get("metadata") if isinstance(projection, Mapping) else None
        )
        if not isinstance(metadata, Mapping):
            raise _fail(
                f"tick-native source projection {cell} lacks event sink metadata"
            )
        if metadata.get("event_history_storage_mode") != "eiel_resident_event_sink_v1":
            continue
        reference = metadata.get("event_history_reference")
        path = reference.get("path") if isinstance(reference, Mapping) else None
        if type(path) is not str or not path.strip():
            raise _fail(
                f"tick-native source projection {cell} lacks an authenticated event sink path"
            )
        projection_paths.add(os.path.realpath(os.path.expanduser(path)))

    if len(projection_paths) > 1:
        raise _fail("tick-native source projections bind multiple event sinks")
    authenticated_path = next(iter(projection_paths), None)
    normalized_source_path = (
        os.path.realpath(os.path.expanduser(source_path))
        if source_path is not None
        else None
    )
    if (
        authenticated_path is not None
        and normalized_source_path is not None
        and authenticated_path != normalized_source_path
    ):
        raise _fail("tick-native source config and projection bind different event sinks")
    sink_path = authenticated_path or normalized_source_path
    if sink_path is None:
        raise _fail("tick-native source has no authenticated event sink binding")

    if not 0 <= PATCH_P2_CONFIG_ARGUMENT_INDEX_V1 < len(arguments):
        raise _fail("tick-native ABI lacks the Patch P2 config argument")
    abi_config = arguments[PATCH_P2_CONFIG_ARGUMENT_INDEX_V1]
    if not isinstance(abi_config, Mapping):
        raise _fail("tick-native ABI Patch P2 config is not a mapping")
    abi_path = abi_config.get("event_history_sink_path")
    if abi_path is not None and (
        type(abi_path) is not str or not abi_path.strip()
    ):
        raise _fail("tick-native ABI event sink path is malformed")
    normalized_abi_path = (
        os.path.realpath(os.path.expanduser(abi_path))
        if abi_path is not None
        else None
    )
    if normalized_abi_path is not None and normalized_abi_path != sink_path:
        raise _fail("tick-native ABI and source projections bind different event sinks")
    repaired = normalized_abi_path is None
    if repaired:
        rebound_config = dict(abi_config)
        rebound_config["event_history_sink_path"] = sink_path
        arguments[PATCH_P2_CONFIG_ARGUMENT_INDEX_V1] = rebound_config
    return sink_path, repaired


def _require_tick_native_event_sink_result_v1(
    result: Mapping[str, Any],
    *,
    sink_path: str,
    require_native_attestation: bool,
) -> None:
    """Require the native owner to return only bounded sink-backed history."""

    if result.get("event_history_storage_mode") != "eiel_resident_event_sink_v1":
        raise _fail(
            "tick-native native result lacks the resident event-history sink mode"
        )
    if "event_histories" in result:
        raise _fail(
            "tick-native native result returned inline cumulative event histories"
        )
    reference = result.get("event_history_reference")
    reference_path = reference.get("path") if isinstance(reference, Mapping) else None
    if type(reference_path) is not str or not reference_path.strip():
        raise _fail("tick-native native result lacks its canonical event sink reference")
    if os.path.realpath(os.path.expanduser(reference_path)) != sink_path:
        raise _fail(
            "tick-native native result event sink reference differs from the bound ABI path"
        )
    if type(result.get("host_event_materialized_bytes")) is not int or result.get(
        "host_event_materialized_bytes"
    ) != 0:
        raise _fail(
            "tick-native native result materialized event history on the host"
        )
    for field in (
        "event_drain_receipts",
        "event_history_frontier_by_cell",
        "event_pending_rows_by_cell",
    ):
        value = result.get(field)
        if not isinstance(value, list):
            raise _fail(f"tick-native native result lacks bounded {field}")
    if not result["event_drain_receipts"]:
        raise _fail("tick-native native result lacks its final event drain receipt")
    if require_native_attestation and (
        result.get("native_event_sink_attestation_schema_version")
        != "eiel_native_event_sink_attestation_v1"
        or result.get("native_event_sink_last_attested_stage")
        != "resident_window_controller_v1"
        or result.get("native_event_sink_abi_argument_index")
        != PATCH_P2_CONFIG_ARGUMENT_INDEX_V1
    ):
        raise _fail(
            "tick-native native result lacks the resident-controller sink attestation"
        )
    native_path = result.get("native_event_sink_path")
    if native_path is not None and (
        type(native_path) is not str
        or os.path.realpath(os.path.expanduser(native_path)) != sink_path
    ):
        raise _fail("tick-native native sink attestation path differs from the ABI path")


def _execute_direct_resident_wdl_jungle_eed_absolute_dispatch_v2(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    span_ticks: int,
    start_tick: int | None = None,
    generation_base: int | None = None,
    absolute_start_next_tick: int | None = None,
    source_window_ticks: int | None = None,
    entrypoint_name: str,
    entrypoint_prefix_arguments: tuple[Any, ...] = (),
    entrypoint_keyword_arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one exact EED span through an explicitly selected native owner."""

    if not isinstance(prepared, PreparedDirectResidentWdlJungleV1):
        raise _fail("prepared must be PreparedDirectResidentWdlJungleV1")
    if prepared.eed_state_tensors is None or prepared.eed_config is None:
        raise _fail("tick-span production execution requires the full EED overlay")
    source = prepared.patch.source
    configuration = source.configuration
    if source.time_authority != "absolute_tick_v1":
        raise _fail("tick-native production requires absolute-tick source authority")
    if source.patch_p2_config.get("tw_rngv2_semantics_version") != 2:
        raise _fail("tick-native production requires TW RNG V2")
    forbidden_host_services = {
        "cpu_request_staging": getattr(
            configuration,
            "cuda_architecture_b_cpu_request_staging_enable",
            False,
        ),
        "policy_birth_segment_refill": getattr(
            configuration,
            "cuda_architecture_b_policy_birth_segment_refill_enable",
            False,
        ),
        "cpython_gaussian_segment_capacity": getattr(
            configuration,
            "cuda_architecture_b_cpython_gauss_segment_capacity",
            0,
        ),
    }
    if forbidden_host_services != {
        "cpu_request_staging": False,
        "policy_birth_segment_refill": False,
        "cpython_gaussian_segment_capacity": 0,
    }:
        raise _fail(
            "tick-native RNG V2 forbids CPU staging and host refill services: "
            f"{forbidden_host_services!r}"
        )
    metadata = prepared.wdl.initial_projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("tick-span source metadata is malformed")
    if absolute_start_next_tick is None:
        raise _fail("tick-native execution requires an explicit absolute start tick")
    if start_tick is None:
        start_tick = absolute_start_next_tick
    if generation_base is None:
        generation_base = 0
    if (
        type(start_tick) is not int
        or start_tick != absolute_start_next_tick
        or type(generation_base) is not int
        or generation_base != 0
        or type(absolute_start_next_tick) is not int
        or absolute_start_next_tick < 0
        or type(span_ticks) is not int
        or span_ticks < 1
    ):
        raise _fail("tick span must use one exact nonnegative absolute-tick cursor")
    if type(source_window_ticks) is not int or source_window_ticks < 1:
        raise _fail("tick-native execution requires a positive source-window capacity")
    terminal_tick = start_tick + span_ticks
    commit_turnover = False
    tick_native_wdl_config, tick_native_jungle_config = (
        _tick_native_private_reporting_configs_v1(prepared)
    )
    entrypoint = getattr(module, entrypoint_name, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {entrypoint_name}")
    base_arguments = list(prepared.patch.arguments)
    event_sink_path, event_sink_binding_repaired = (
        _bind_tick_native_event_sink_arguments_v1(prepared, base_arguments)
    )
    runtime_capacity = base_arguments[
        PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V2
    ]
    if (
        not hasattr(runtime_capacity, "shape")
        or len(runtime_capacity.shape) != 3
        or int(runtime_capacity.shape[1]) < source_window_ticks
    ):
        raise _fail("tick-native runtime buffer is smaller than source_window_ticks")

    source_window_arguments: list[tuple[Any, ...]] = []
    cursor = start_tick
    remaining = span_ticks
    while remaining:
        # Keep each native chunk inside one absolute source window.  The
        # native multi-window entrypoint returns the final chunk's per-tick
        # bookkeeping, which is also the private settlement prefix.  Splitting
        # a restored cursor at a fixed width could cross the source-window
        # lattice and leave that final tensor unable to represent the terminal
        # generation prefix.
        ticks_until_source_boundary = source_window_ticks - (
            cursor % source_window_ticks
        )
        chunk_ticks = min(remaining, ticks_until_source_boundary)
        arguments = list(base_arguments)
        arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V2] = (
            runtime_capacity[:, :chunk_ticks].contiguous()
        )
        full_tick_outputs = list(
            arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V2]
        )
        full_tick_outputs[-1] = full_tick_outputs[-1][:1]
        arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V2] = full_tick_outputs
        patch_stats = list(arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V2])
        patch_stats[0] = patch_stats[0][:, :chunk_ticks].contiguous()
        if len(patch_stats) == 2:
            patch_stats[1] = patch_stats[1][:1]
        arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V2] = patch_stats
        arguments[TICKS_ARGUMENT_INDEX_V2] = chunk_ticks
        arguments[START_TICK_ARGUMENT_INDEX_V1] = cursor
        arguments[GENERATION_ARGUMENT_INDEX_V1] = generation_base
        _bind_tick_span_turnover_arguments_v1(
            arguments,
            generation_turnover_configured=(
                prepared.patch.source.configuration.generation_turnover_enabled
            ),
            commit_turnover=commit_turnover,
        )
        arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = 1
        source_window_arguments.append(
            (
                *arguments,
                1,
                metadata.get("policy_weight_mutation_enable"),
                metadata.get("policy_weight_mutation_mode"),
                metadata.get("policy_weight_mutation_rate"),
                metadata.get("policy_weight_mutation_std"),
            )
        )
        cursor += chunk_ticks
        remaining -= chunk_ticks

    if not source_window_arguments:
        raise _fail("tick-native dispatch produced no bounded source window")
    if entrypoint_name == WDL_JUNGLE_EED_TICK_DISPATCH_ENTRYPOINT_V2:
        raw = entrypoint(
            list(prepared.wdl.state_tensors),
            list(prepared.wdl.schedule_tensors),
            list(prepared.wdl.reporting_tensors),
            tick_native_wdl_config,
            list(prepared.jungle_state_tensors),
            list(prepared.jungle_reporting_tensors),
            tick_native_jungle_config,
            list(prepared.eed_state_tensors),
            dict(prepared.eed_config),
            source_window_arguments,
        )
    else:
        if len(source_window_arguments) != 1:
            raise _fail(
                "this native policy owner lacks bounded multi-window dispatch"
            )
        raw = entrypoint(
            *entrypoint_prefix_arguments,
            list(prepared.wdl.state_tensors),
            list(prepared.wdl.schedule_tensors),
            list(prepared.wdl.reporting_tensors),
            tick_native_wdl_config,
            list(prepared.jungle_state_tensors),
            list(prepared.jungle_reporting_tensors),
            tick_native_jungle_config,
            list(prepared.eed_state_tensors),
            dict(prepared.eed_config),
            *source_window_arguments[0],
            **dict(entrypoint_keyword_arguments or {}),
        )
    if not isinstance(raw, Mapping):
        raise _fail("WDL/Jungle/EED tick dispatch returned a non-mapping")
    result = dict(raw)
    _require_no_native_failure_v1(result)
    _require_tick_native_event_sink_result_v1(
        result,
        sink_path=event_sink_path,
        require_native_attestation=(
            entrypoint_name == WDL_JUNGLE_EED_TICK_DISPATCH_ENTRYPOINT_V2
        ),
    )
    if (
        result.get("absolute_tick_span_native_owner") is not True
        or result.get("native_tick_iterations") != span_ticks
        or result.get("wdl_jungle_native_owner") is not True
        or result.get("eed_native_owner") is not True
    ):
        raise _fail("WDL/Jungle/EED tick dispatch ownership receipt differs")
    result.update(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "owner": "native_extension",
            "physical_window_generations": (
                prepared.patch.source.physical_window_generations
            ),
            "completed_generations": 1,
            "logical_generation_count": 1,
            "window_count": resident_window_count_v1(
                initial_generation=prepared.patch.source.generation_base,
                logical_generation_count=1,
                physical_window_generations=(
                    prepared.patch.source.physical_window_generations
                ),
            ),
            "physical_window_alignment": "absolute_tick_v1",
            "external_python_window_boundaries": 0,
            "direct_cpp_window_core_dispatch": True,
            "production_native_chunk_loop": True,
            "hidden_fallback": False,
            "runtime_jit": False,
            "tick_span_start": start_tick,
            "tick_span_end": terminal_tick,
            "tick_span_count": span_ticks,
            "source_window_ticks": source_window_ticks,
            "source_window_count": len(source_window_arguments),
            "tick_native_start_next_tick": absolute_start_next_tick,
            "tick_native_end_next_tick": absolute_start_next_tick + span_ticks,
            # Deprecated private compatibility metadata only; derive it from
            # the absolute end instead of publishing a stale constant.
            "terminal_generation_boundary": (
                terminal_tick % source_window_ticks == 0
            ),
            "tick_native_event_sink_path": event_sink_path,
            "tick_native_event_sink_binding_repaired": (
                event_sink_binding_repaired
            ),
        }
    )
    return result


def execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    span_ticks: int,
    start_tick: int | None = None,
    generation_base: int | None = None,
    absolute_start_next_tick: int | None = None,
    source_window_ticks: int | None = None,
) -> dict[str, Any]:
    """Execute one exact partial span under the sole HAPB policy owner."""

    if not isinstance(prepared, PreparedDirectResidentWdlJungleV1):
        raise _fail("prepared must be PreparedDirectResidentWdlJungleV1")
    source = prepared.patch.source
    try:
        hapb_physical_cell_count_v1(len(prepared.cell_ids))
    except HAPBContractError as exc:
        raise _fail("HAPB tick-span route requires physical B in [1, 256]") from exc
    owner = source.live_banks.get("hapb_policy_owner")
    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise _fail("HAPB tick-span route lacks its host-authoritative policy owner")
    if source.live_banks.get("hapb_full_cuda_mlp_pack_absent") is not True:
        raise _fail("HAPB tick-span route cannot prove the CUDA policy bank is absent")
    if (
        _frag_identity_reuse_requested_v1()
        and prepared.policy_window_identity_owner is None
    ):
        raise _fail("HAPB FRAG identity tick-span route lacks its transient owner")
    sentinel = source.live_state.get("mlp_pack")
    if (
        not isinstance(sentinel, torch.Tensor)
        or sentinel.device.type != "cuda"
        or sentinel.dtype != torch.float32
        or tuple(sentinel.shape) != (len(prepared.cell_ids), 1, 9991)
    ):
        raise _fail("HAPB tick-span route requires the nonresident MLP ABI sentinel")
    configuration = source.configuration
    if getattr(configuration, "ordinary_births_enabled", None) is not True:
        raise _fail("HAPB tick-span route requires ordinary births to be enabled")
    birth_transaction = _hapb_ordinary_birth_transaction_v1(module, prepared)
    turnover_transaction = (
        _hapb_turnover_transaction_v1(module, prepared)
        if getattr(configuration, "generation_turnover_enabled", None) is True
        else None
    )
    reset_residual = getattr(
        module, "eiel_cuda_architecture_b_hapb_residual_wall_reset_v1", None
    )
    if callable(reset_residual):
        reset_residual()
    result = _execute_direct_resident_wdl_jungle_eed_absolute_dispatch_v2(
        module,
        prepared,
        span_ticks=span_ticks,
        start_tick=start_tick,
        generation_base=generation_base,
        absolute_start_next_tick=absolute_start_next_tick,
        source_window_ticks=source_window_ticks,
        entrypoint_name=WDL_JUNGLE_EED_HAPB_TICK_SPAN_ENTRYPOINT_V1,
        entrypoint_prefix_arguments=(
            owner.canonical_host_bank,
            owner.pinned_staging_buffers[0],
            owner.require_hot_prefix_cache_v1(),
            owner.cuda_tile_arena,
            owner.content_row_map,
            owner.policy_traffic_counters,
            owner.content_alias_enabled,
            birth_transaction,
            turnover_transaction,
        ),
        entrypoint_keyword_arguments={
            "policy_window_identity_owner": prepared.policy_window_identity_owner,
        },
    )
    validation_token = _take_policy_window_identity_validation_token_v1(
        result, prepared.policy_window_identity_owner
    )
    boundary_samples = tuple(
        getattr(birth_transaction, "hapb_boundary_timing_samples", ())
    )
    result["hapb_host_boundary_timing"] = {
        "schema_version": HAPB_HOST_BOUNDARY_TIMING_SCHEMA_V1,
        "samples": [dict(sample) for sample in boundary_samples],
        "host_classification_map_construction_wall_ns": sum(
            int(sample["host_classification_map_construction_wall_ns"])
            for sample in boundary_samples
        ),
        "control_d2h_bytes": sum(
            int(sample["control_d2h_bytes"]) for sample in boundary_samples
        ),
        "control_h2d_bytes": sum(
            int(sample["control_h2d_bytes"]) for sample in boundary_samples
        ),
        "scope": (
            "successful host birth-boundary map/commit callbacks only; "
            "does not include native GPU execution"
        ),
    }
    result["hapb_policy_only_route"] = False
    result["hapb_host_authoritative_ordinary_birth"] = True
    result["hapb_host_authoritative_generation_turnover"] = bool(
        result.get("terminal_generation_boundary") is True
        and getattr(configuration, "generation_turnover_enabled", None) is True
    )
    result["hapb_turnover_boundary_timing"] = {
        "samples": [
            dict(sample)
            for sample in getattr(turnover_transaction, "hapb_turnover_timing_samples", ())
        ],
        "scope": (
            "successful host turnover plan/commit callbacks only; "
            "does not include native GPU execution"
        ),
    }
    result["hapb_policy_traffic_counters"] = owner.policy_traffic_snapshot_v1()
    snapshot_residual = getattr(
        module, "eiel_cuda_architecture_b_hapb_residual_wall_snapshot_v1", None
    )
    if callable(snapshot_residual):
        raw_residual = snapshot_residual()
        result["hapb_residual_wall"] = {
            str(key): int(value) for key, value in dict(raw_residual).items()
        }
    result["hapb_content_alias_enabled"] = owner.content_alias_enabled
    result["hapb_content_alias_relation_sha256"] = (
        owner.content_alias_relation_sha256_v1()
    )
    _commit_policy_window_identity_owner_entry_v1(
        module, prepared.policy_window_identity_owner, validation_token
    )
    return result


def execute_direct_resident_wdl_jungle_eed_hapb_policy_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    logical_generation_count: int,
    run_id: str,
    _ordinary_birth_transaction: object | None = None,
    _turnover_transaction: object | None = None,
) -> dict[str, Any]:
    """Run the explicit policy-only HAPB B1 parity route without fallback.

    This accepts the physical B48 HAPB allocation while B1 is the semantic
    trajectory under comparison. Ordinary births and generation turnover are
    admitted only through their distinct host-authoritative transactions.
    """

    if not isinstance(prepared, PreparedDirectResidentWdlJungleV1):
        raise _fail("prepared must be PreparedDirectResidentWdlJungleV1")
    if prepared.eed_state_tensors is None or prepared.eed_config is None:
        raise _fail("HAPB policy route requires the explicit EED overlay")
    source = prepared.patch.source
    try:
        hapb_physical_cell_count_v1(len(prepared.cell_ids))
    except HAPBContractError as exc:
        raise _fail("HAPB policy route requires physical B in [1, 256]") from exc
    owner = source.live_banks.get("hapb_policy_owner")
    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise _fail("HAPB policy route lacks its host-authoritative policy owner")
    if source.live_banks.get("hapb_full_cuda_mlp_pack_absent") is not True:
        raise _fail("HAPB policy route cannot prove the CUDA policy bank is absent")
    if (
        _frag_identity_reuse_requested_v1()
        and prepared.policy_window_identity_owner is None
    ):
        raise _fail("HAPB FRAG identity route lacks its transient owner")
    sentinel = source.live_state.get("mlp_pack")
    if (
        not isinstance(sentinel, torch.Tensor)
        or sentinel.device.type != "cuda"
        or sentinel.dtype != torch.float32
        or tuple(sentinel.shape) != (len(prepared.cell_ids), 1, 9991)
    ):
        raise _fail("HAPB policy route requires the nonresident MLP ABI sentinel")
    configuration = source.configuration
    ordinary_births_enabled = getattr(configuration, "ordinary_births_enabled", None)
    if _ordinary_birth_transaction is None:
        if ordinary_births_enabled is not False:
            raise _fail("HAPB policy route refuses ordinary births before HAPB birth commit")
    elif ordinary_births_enabled is not True:
        raise _fail("HAPB ordinary-birth route requires ordinary births to be enabled")
    turnover_enabled = getattr(configuration, "generation_turnover_enabled", None)
    if turnover_enabled is True and _turnover_transaction is None:
        raise _fail(
            "HAPB policy route requires its host-authoritative turnover transaction"
        )
    if len(source.full_tick_output_tensors) <= 17:
        raise _fail("HAPB policy route requires exactly one resident output tick")
    jungle_state_cursor = source.full_tick_output_tensors[16]
    jungle_buffer_cursor = source.full_tick_output_tensors[17]
    if (
        not isinstance(jungle_state_cursor, torch.Tensor)
        or not isinstance(jungle_buffer_cursor, torch.Tensor)
        or jungle_state_cursor.size(1) != 1
        or jungle_buffer_cursor.size(1) != 1
    ):
        raise _fail("HAPB policy route requires exactly one resident output tick")
    if not WDL_JUNGLE_RESIDENT_GENERATION_MIN_V1 <= logical_generation_count <= WDL_RESIDENT_GENERATION_MAX_V2:
        raise _fail("HAPB policy route logical generation count must be in [3,128]")
    if logical_generation_count > prepared.wdl.generation_rows:
        raise _fail("HAPB policy route reporting allocation does not cover the trajectory")
    if source.logical_generation_count != logical_generation_count:
        raise _fail("HAPB policy route source identity capacity does not match the trajectory")
    entrypoint = getattr(module, HAPB_DIRECT_ENTRYPOINT_V1, None)
    if not callable(entrypoint):
        raise _fail(f"CUDA extension is missing callable {HAPB_DIRECT_ENTRYPOINT_V1}")
    metadata = prepared.wdl.initial_projection.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _fail("HAPB policy route initial projection metadata is not a mapping")
    control = create_run_control_v2(module, run_id=run_id)
    arguments = (
        control,
        list(prepared.wdl.state_tensors),
        list(prepared.wdl.schedule_tensors),
        list(prepared.wdl.reporting_tensors),
        dict(prepared.wdl.config),
        list(prepared.jungle_state_tensors),
        list(prepared.jungle_reporting_tensors),
        dict(prepared.jungle_config),
        list(prepared.eed_state_tensors),
        dict(prepared.eed_config),
        *prepared.patch.arguments,
        logical_generation_count,
        metadata.get("policy_weight_mutation_enable"),
        metadata.get("policy_weight_mutation_mode"),
        metadata.get("policy_weight_mutation_rate"),
        metadata.get("policy_weight_mutation_std"),
    )
    raw = entrypoint(
        owner.canonical_host_bank,
        owner.pinned_staging_buffers[0],
        owner.require_hot_prefix_cache_v1(),
        owner.cuda_tile_arena,
        owner.content_row_map,
        owner.policy_traffic_counters,
        owner.content_alias_enabled,
        _ordinary_birth_transaction,
        _turnover_transaction,
        *arguments,
        policy_window_identity_owner=prepared.policy_window_identity_owner,
    )
    if not isinstance(raw, Mapping):
        raise _fail("HAPB policy CUDA entrypoint returned a non-mapping result")
    result = dict(raw)
    validation_token = _take_policy_window_identity_validation_token_v1(
        result, prepared.policy_window_identity_owner
    )
    _require_no_native_failure_v1(result)
    result["hapb_policy_traffic_counters"] = owner.policy_traffic_snapshot_v1()
    result["hapb_content_alias_enabled"] = owner.content_alias_enabled
    result["hapb_content_alias_relation_sha256"] = (
        owner.content_alias_relation_sha256_v1()
    )
    result["hapb_host_authoritative_generation_turnover"] = bool(
        turnover_enabled is True
    )
    result["hapb_turnover_boundary_timing"] = {
        "samples": [
            dict(sample)
            for sample in getattr(_turnover_transaction, "hapb_turnover_timing_samples", ())
        ],
        "scope": (
            "successful host turnover plan/commit callbacks only; "
            "does not include native GPU execution"
        ),
    }
    expected = {
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "wdl_native_controller_abi_version": WDL_ABI_VERSION_V3,
        "jungle_native_controller_abi_version": JUNGLE_ABI_VERSION_V4,
        "eed_native_controller_abi_version": EED_ABI_VERSION_V1,
        "wdl_jungle_python_tick_or_generation_ownership": False,
        "completed_generations": logical_generation_count,
        "logical_generation_count": logical_generation_count,
        "physical_window_generations": source.physical_window_generations,
        "window_count": resident_window_count_v1(
            initial_generation=source.generation_base,
            logical_generation_count=logical_generation_count,
            physical_window_generations=source.physical_window_generations,
        ),
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"HAPB policy resident result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    manifest = dict(control.manifest())
    if (
        manifest.get("status") != "generation_complete"
        or manifest.get("requested_generations") != logical_generation_count
        or manifest.get("active_generation")
        != source.generation_base + logical_generation_count - 1
    ):
        raise _fail("HAPB policy native run-control is not at the final boundary")
    result["native_run_control_v2"] = manifest
    result["hapb_policy_only_route"] = _ordinary_birth_transaction is None
    result["hapb_host_authoritative_ordinary_birth"] = (
        _ordinary_birth_transaction is not None
    )
    _commit_policy_window_identity_owner_entry_v1(
        module, prepared.policy_window_identity_owner, validation_token
    )
    return result

def execute_direct_resident_wdl_jungle_eed_hapb_birth_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    logical_generation_count: int,
    run_id: str,
) -> dict[str, Any]:
    """Run direct HAPB with the explicit pre-eligibility birth transaction.

    Native prepare preserves the existing plan, mutation-tape, and RNG
    ordering.  The bridge below gathers only the planned rows to the one HAPB
    workbench and returns only after the child bytes are canonical on host;
    the following native commit skips solely its resident ``mlp_pack`` write.
    """

    birth_transaction = _hapb_ordinary_birth_transaction_v1(module, prepared)
    turnover_transaction = (
        _hapb_turnover_transaction_v1(module, prepared)
        if getattr(prepared.patch.source.configuration,
                   "generation_turnover_enabled", None) is True
        else None
    )

    return execute_direct_resident_wdl_jungle_eed_hapb_policy_v1(
        module,
        prepared,
        logical_generation_count=logical_generation_count,
        run_id=run_id,
        _ordinary_birth_transaction=birth_transaction,
        _turnover_transaction=turnover_transaction,
    )


def _hapb_ordinary_birth_transaction_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
) -> Any:
    """Bind the exact host-policy birth commit used by full and partial spans."""

    source = prepared.patch.source
    owner = source.live_banks.get("hapb_policy_owner")
    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise _fail("HAPB ordinary-birth route lacks its host-authoritative policy owner")
    mutate_rows = getattr(
        module, "eiel_cuda_architecture_b_hapb_ordinary_birth_mutate_rows_v1", None
    )
    if not callable(mutate_rows):
        raise _fail("HAPB ordinary-birth route lacks its bounded CUDA mutation export")

    boundary_timing_samples: list[dict[str, int]] = []

    def _cuda_nbytes(value: Any) -> int:
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            return 0
        return int(value.numel()) * int(value.element_size())

    def host_birth_transaction(
        planned_parent_slots: torch.Tensor,
        planned_child_slots: torch.Tensor,
        planned_count: torch.Tensor,
        policy_delta_tape: torch.Tensor,
        policy_mutation_mask_tape: torch.Tensor,
        policy_semantic_draws_tape: torch.Tensor,
        policy_absolute_cursor: torch.Tensor,
        policy_window_base: torch.Tensor,
    ) -> None:
        # The native launch has synchronized at this exact boundary.  The
        # source policy cursor is absolute; HAPB's bounded resident window is
        # addressed physically relative to its stable window base.
        boundary_started_ns = perf_counter_ns()
        control_d2h_bytes = sum(
            _cuda_nbytes(value)
            for value in (
                planned_parent_slots,
                planned_child_slots,
                planned_count,
                policy_mutation_mask_tape,
                policy_semantic_draws_tape,
                policy_absolute_cursor,
                policy_window_base,
            )
        )
        owner.record_control_d2h_bytes_v1(control_d2h_bytes)
        # The callback currently returns no control tensor to the native
        # caller; parent/delta/child policy buffers are intentionally excluded
        # from this non-policy counter.
        owner.record_control_h2d_bytes_v1(0)
        parent_cpu = planned_parent_slots.detach().cpu().contiguous()
        child_cpu = planned_child_slots.detach().cpu().contiguous()
        count_cpu = planned_count.detach().cpu().contiguous()
        absolute_cursor_cpu = policy_absolute_cursor.detach().cpu().contiguous()
        window_base_cpu = policy_window_base.detach().cpu().contiguous()
        physical_cursor_cpu = (absolute_cursor_cpu - window_base_cpu).contiguous()
        if torch.any(physical_cursor_cpu < 0):
            raise _fail("HAPB ordinary-birth policy cursor precedes its resident window")
        committed = commit_hapb_ordinary_fixed_mlp_birth_rows_v1(
            owner,
            cuda_mutate_export=mutate_rows,
            parent_slots=parent_cpu,
            child_slots=child_cpu,
            birth_counts=count_cpu,
            policy_birth_delta_tape=policy_delta_tape.detach().cpu().contiguous(),
            policy_birth_mutation_mask_tape=(
                policy_mutation_mask_tape.detach().cpu().contiguous()
            ),
            policy_birth_semantic_draws_tape=(
                policy_semantic_draws_tape.detach().cpu().contiguous()
            ),
            policy_birth_tape_cursor=physical_cursor_cpu,
        )
        for cell, count in enumerate(count_cpu.tolist()):
            if count and not torch.equal(
                committed[cell, :count], child_cpu[cell, :count]
            ):
                raise _fail("HAPB ordinary-birth host transaction lost its planned child rows")
        boundary_timing_samples.append(
            {
                "host_classification_map_construction_wall_ns": int(
                    perf_counter_ns() - boundary_started_ns
                ),
                "control_d2h_bytes": int(control_d2h_bytes),
                # The current birth callback returns no control tensor to the
                # native caller.  Policy parent/delta/child buffers remain
                # outside this control-byte counter by contract.
                "control_h2d_bytes": 0,
            }
        )

    host_birth_transaction.hapb_boundary_timing_samples = boundary_timing_samples

    return host_birth_transaction


def _hapb_turnover_transaction_v1(
    module: Any,
    prepared: PreparedDirectResidentWdlJungleV1,
) -> Any:
    """Bind the exact post-identity host-policy turnover transaction."""

    source = prepared.patch.source
    owner = source.live_banks.get("hapb_policy_owner")
    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise _fail("HAPB turnover route lacks its host-authoritative policy owner")
    mutate_rows = getattr(
        module, "eiel_cuda_architecture_b_hapb_ordinary_birth_mutate_rows_v1", None
    )
    if not callable(mutate_rows):
        raise _fail("HAPB turnover route lacks its bounded CUDA mutation export")

    turnover_timing_samples: list[dict[str, int]] = []

    def _cuda_nbytes(value: Any) -> int:
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            return 0
        return int(value.numel()) * int(value.element_size())

    def host_turnover_transaction(
        planned_parent_slots: torch.Tensor,
        planned_mate_slots: torch.Tensor,
        planned_child_slots: torch.Tensor,
        committed_count: torch.Tensor,
        policy_delta_tape: torch.Tensor,
        policy_mutation_mask_tape: torch.Tensor,
        policy_crossover_mask_packed_tape: torch.Tensor,
        policy_absolute_cursor: torch.Tensor,
        policy_window_base: torch.Tensor,
    ) -> None:
        boundary_started_ns = perf_counter_ns()
        parent_cpu = planned_parent_slots.detach().cpu().contiguous()
        mate_cpu = planned_mate_slots.detach().cpu().contiguous()
        child_cpu = planned_child_slots.detach().cpu().contiguous()
        count_cpu = committed_count.detach().cpu().contiguous()
        # The callback runs after native shadow commit but before publish, so
        # the source cursor is pre-turnover.  Native publish will expose this
        # cursor plus the committed turnover rows.
        absolute_cpu = policy_absolute_cursor.detach().cpu().contiguous()
        absolute_after_cpu = (absolute_cpu + count_cpu).contiguous()
        window_cpu = policy_window_base.detach().cpu().contiguous()
        # Copy only the bounded policy-tape slices needed for this turnover.
        # A full birth tape can be much larger than one generation; transferring
        # its unused suffix would violate the bounded HAPB boundary contract.
        max_width = int(policy_delta_tape.shape[2])
        delta_rows: list[torch.Tensor] = []
        mutation_rows: list[torch.Tensor] = []
        crossover_rows: list[torch.Tensor] = []
        compact_window = torch.empty_like(absolute_cpu)
        compact_window.copy_(absolute_cpu)
        control_d2h_bytes = sum(
            _cuda_nbytes(value)
            for value in (
                planned_parent_slots,
                planned_mate_slots,
                planned_child_slots,
                committed_count,
                policy_absolute_cursor,
                policy_window_base,
            )
        )
        # The tape metadata is represented as control traffic; only the
        # bounded slices needed by this turnover cross PCIe.  Full resident
        # tape tensors must not be charged here because they remain on device.
        for cell, count in enumerate(count_cpu.tolist()):
            count = int(count)
            absolute_before = int(absolute_cpu[cell].item())
            window_base = int(window_cpu[cell].item())
            absolute_start = absolute_before
            physical_start = absolute_start - window_base
            if count < 0 or physical_start < 0:
                raise _fail(
                    "HAPB turnover policy cursor precedes its resident window: "
                    f"cell={cell} pre_cursor={absolute_before} count={count} "
                    f"window_base={window_base}"
                )
            if count == 0:
                delta_rows.append(torch.empty((0, max_width), dtype=torch.float32))
                mutation_rows.append(torch.empty((0, max_width), dtype=torch.bool))
                crossover_rows.append(
                    torch.empty(
                        (0, POLICY_CROSSOVER_PACKED_WIDTH_V1), dtype=torch.uint8
                    )
                )
                continue
            delta_rows.append(
                policy_delta_tape[
                    cell, physical_start : physical_start + count
                ].detach().cpu().contiguous()
            )
            mutation_rows.append(
                policy_mutation_mask_tape[
                    cell, physical_start : physical_start + count
                ].detach().cpu().contiguous()
            )
            crossover_rows.append(
                policy_crossover_mask_packed_tape[
                    cell, physical_start : physical_start + count
                ].detach().cpu().contiguous()
            )
            control_d2h_bytes += sum(
                _cuda_nbytes(value)
                for value in (
                    policy_delta_tape[
                        cell, physical_start : physical_start + count
                    ],
                    policy_mutation_mask_tape[
                        cell, physical_start : physical_start + count
                    ],
                    policy_crossover_mask_packed_tape[
                        cell, physical_start : physical_start + count
                    ],
                )
            )
            compact_window[cell] = absolute_start

        owner.record_control_d2h_bytes_v1(control_d2h_bytes)
        owner.record_control_h2d_bytes_v1(0)

        committed = commit_hapb_turnover_fixed_mlp_birth_rows_v1(
            owner,
            cuda_mutate_export=mutate_rows,
            parent_slots=parent_cpu,
            mate_slots=mate_cpu,
            child_slots=child_cpu,
            committed_counts=count_cpu,
            policy_delta_tape=tuple(delta_rows),
            policy_mutation_mask_tape=tuple(mutation_rows),
            policy_crossover_mask_packed_tape=tuple(crossover_rows),
            policy_absolute_cursor=absolute_after_cpu,
            policy_window_base=compact_window,
        )
        for cell, count in enumerate(count_cpu.tolist()):
            count = int(count)
            if count and not torch.equal(
                committed[cell, :count], child_cpu[cell, :count]
            ):
                raise _fail("HAPB turnover host transaction lost committed child rows")
        turnover_timing_samples.append(
            {
                "host_classification_map_construction_wall_ns": int(
                    perf_counter_ns() - boundary_started_ns
                ),
                "control_d2h_bytes": int(control_d2h_bytes),
                "control_h2d_bytes": 0,
                "committed_rows": int(sum(int(value) for value in count_cpu.tolist())),
            }
        )

    host_turnover_transaction.hapb_turnover_timing_samples = turnover_timing_samples
    return host_turnover_transaction


def project_direct_resident_wdl_jungle_multicell_settlement_v1(
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    reporting_enabled: bool,
    generation_base_override: int | None = None,
    bounded_projection_cursor: bool = False,
) -> DirectResidentPatchMulticellSettlementV1:
    """Attach Jungle/EED state while retaining WDL/Patch settlement evidence."""

    wdl_prepared = prepared.wdl
    if prepared.eed_state_tensors is not None:
        live_banks = dict(wdl_prepared.patch.live_banks)
        for name, value in zip(
            EED_STATE_FIELDS_V1, prepared.eed_state_tensors, strict=True
        ):
            if name in live_banks:
                raise _fail(f"WDL/Jungle checkpoint live banks already own {name!r}")
            # The Patch settlement is the canonical checkpoint projector.  It
            # must receive EED's resident terminal banks before it resolves
            # the source projection's required tensor schema.
            live_banks[name] = value
        source = wdl_prepared.patch.source
        if is_dataclass(source):
            source = replace(source, live_banks=live_banks)
            patch = replace(
                wdl_prepared.patch,
                source=source,
                live_banks=source.live_banks,
            )
        else:
            # Lightweight projector fixtures may use an opaque source stub;
            # the production source is the dataclass path above.
            patch = replace(wdl_prepared.patch, live_banks=live_banks)
        wdl_prepared = replace(wdl_prepared, patch=patch)
    settlement = project_direct_resident_wdl_multicell_settlement_v2(
        wdl_prepared,
        direct_result=direct_result,
        completed_generation_count=completed_generation_count,
        reporting_enabled=reporting_enabled,
        generation_base_override=generation_base_override,
        bounded_projection_cursor=bounded_projection_cursor,
    )
    if bounded_projection_cursor:
        cursor = settlement.checkpoint_projection_cursor
        if cursor is None:
            raise _fail("bounded WDL/Jungle projection lacks a checkpoint cursor")

        def transform(
            detached: DetachedCellProjectionV1,
        ) -> DetachedCellProjectionV1:
            cell = detached.cell_index
            expected_recovery = prepared.jungle_state_tensors[0][cell].detach().cpu()
            contributions: dict[str, dict[str, Any]] = {
                "jungle": {
                    JUNGLE_STATE_FIELDS_V4[1]: prepared.jungle_state_tensors[1][cell]
                }
            }
            if prepared.eed_state_tensors is not None:
                contributions["eed"] = {
                    name: value[cell]
                    for name, value in zip(
                        EED_STATE_FIELDS_V1,
                        prepared.eed_state_tensors,
                        strict=True,
                    )
                }
            try:
                composite = compose_composite_runtime_state_v1(
                    detached.require_projection(),
                    base_component_id="patch_wdl",
                    component_tensors=contributions,
                    shared_tensor_owners={
                        JUNGLE_STATE_FIELDS_V4[0]: ("patch_wdl", "jungle")
                    },
                    expected_existing_tensors={
                        JUNGLE_STATE_FIELDS_V4[0]: expected_recovery
                    },
                    execution_markers=direct_result,
                    expected_cell_id=prepared.cell_ids[cell],
                    projection_digest_fn=projection_digest,
                )
            except CompositeRuntimeStateError as exc:
                raise _fail(str(exc)) from exc
            projection = composite.wire_projection()
            return DetachedCellProjectionV1(
                schema_version=detached.schema_version,
                cell_id=detached.cell_id,
                cell_index=detached.cell_index,
                projection=projection,
                compact_result_rows=detached.compact_result_rows,
                segment_artifacts=detached.segment_artifacts,
                lifecycle_qacct_receipt=detached.lifecycle_qacct_receipt,
                terminal_generation=detached.terminal_generation,
                terminal_tick=detached.terminal_tick,
                live_projection_bytes=detached.live_projection_bytes,
            )

        mapped = cursor.map(transform)
        return replace(
            settlement,
            checkpoint_projections=(),
            checkpoint_projection_cursor=mapped,
        )
    out: list[Mapping[str, Any]] = []
    for cell, checkpoint in enumerate(settlement.checkpoint_projections):
        expected_recovery = prepared.jungle_state_tensors[0][cell].detach().cpu()
        contributions: dict[str, dict[str, Any]] = {
            "jungle": {
                JUNGLE_STATE_FIELDS_V4[1]: prepared.jungle_state_tensors[1][cell]
            }
        }
        if prepared.eed_state_tensors is not None:
            contributions["eed"] = {
                name: value[cell]
                for name, value in zip(
                    EED_STATE_FIELDS_V1,
                    prepared.eed_state_tensors,
                    strict=True,
                )
            }
        try:
            composite = compose_composite_runtime_state_v1(
                checkpoint,
                base_component_id="patch_wdl",
                component_tensors=contributions,
                shared_tensor_owners={
                    JUNGLE_STATE_FIELDS_V4[0]: ("patch_wdl", "jungle")
                },
                expected_existing_tensors={
                    JUNGLE_STATE_FIELDS_V4[0]: expected_recovery
                },
                execution_markers=direct_result,
                expected_cell_id=prepared.cell_ids[cell],
                projection_digest_fn=projection_digest,
            )
        except CompositeRuntimeStateError as exc:
            raise _fail(str(exc)) from exc
        out.append(composite.wire_projection())
    return replace(settlement, checkpoint_projections=tuple(out))


def project_direct_resident_wdl_jungle_multicell_checkpoints_v1(
    prepared: PreparedDirectResidentWdlJungleV1,
    *,
    direct_result: Mapping[str, Any],
    completed_generation_count: int,
    reporting_enabled: bool,
) -> tuple[Mapping[str, Any], ...]:
    """Project WDL authority first, then attach Jungle/EED state."""

    settlement = project_direct_resident_wdl_jungle_multicell_settlement_v1(
        prepared,
        direct_result=direct_result,
        completed_generation_count=completed_generation_count,
        reporting_enabled=reporting_enabled,
    )
    return tuple(settlement.checkpoint_projections)


def project_direct_resident_wdl_jungle_generation_zero_v1(
    prepared: Any,
) -> tuple[Mapping[str, Any], ...]:
    """Export the initial singular owner while retaining Jungle visibility."""

    combined = getattr(prepared, "physical_window_arguments", prepared)
    if not isinstance(combined, PreparedDirectResidentWdlJungleV1):
        raise _fail("prepared must be PreparedDirectResidentWdlJungleV1")
    checkpoints = project_direct_resident_wdl_generation_zero_v2(combined.wdl)
    if len(checkpoints) != len(combined.cell_ids):
        raise _fail("WDL/Jungle generation-zero checkpoint count differs from cells")
    out: list[Mapping[str, Any]] = []
    for cell, checkpoint in enumerate(checkpoints):
        contributions: dict[str, dict[str, Any]] = {
            "jungle": {
                JUNGLE_STATE_FIELDS_V4[1]: combined.jungle_state_tensors[1][cell]
            }
        }
        if combined.eed_state_tensors is not None:
            contributions["eed"] = {
                name: value[cell]
                for name, value in zip(
                    EED_STATE_FIELDS_V1,
                    combined.eed_state_tensors,
                    strict=True,
                )
            }
        try:
            composite = compose_composite_runtime_state_v1(
                checkpoint,
                base_component_id="patch_wdl",
                component_tensors=contributions,
                shared_tensor_owners={
                    JUNGLE_STATE_FIELDS_V4[0]: ("patch_wdl", "jungle")
                },
                expected_existing_tensors={
                    JUNGLE_STATE_FIELDS_V4[0]: combined.jungle_state_tensors[0][cell]
                },
                expected_cell_id=combined.cell_ids[cell],
                projection_digest_fn=projection_digest,
            )
        except CompositeRuntimeStateError as exc:
            raise _fail(str(exc)) from exc
        out.append(composite.wire_projection())
    return tuple(out)


__all__ = [
    "CudaArchitectureBDirectResidentWdlJungleV1Error",
    "PreparedDirectResidentWdlJungleV1",
    "WDL_JUNGLE_DIRECT_ENTRYPOINT_V1",
    "WDL_JUNGLE_EED_DIRECT_ENTRYPOINT_V1",
    "HAPB_DIRECT_CHECKPOINT_SCHEMA_V1",
    "build_prepared_direct_resident_wdl_jungle_v1",
    "export_direct_resident_wdl_jungle_hapb_checkpoint_v1",
    "execute_direct_resident_wdl_jungle_eed_hapb_birth_v1",
    "execute_direct_resident_wdl_jungle_eed_hapb_policy_v1",
    "execute_direct_resident_wdl_jungle_v1",
    "execute_direct_resident_wdl_jungle_eed_tick_dispatch_v2",
    "execute_direct_resident_wdl_jungle_eed_hapb_tick_span_v1",
    "patch_envelope_with_wdl_jungle_masked_v1",
    "project_direct_resident_wdl_jungle_multicell_settlement_v1",
    "project_direct_resident_wdl_jungle_multicell_checkpoints_v1",
    "project_direct_resident_wdl_jungle_generation_zero_v1",
    "restore_direct_resident_wdl_jungle_hapb_checkpoint_sources_v1",
]
