"""Deterministic CPU boundary actions for the frozen Phase F v2 matrix.

This module joins existing production authorities; it does not implement a
second lifecycle, migration, or checkpoint format.  It deliberately stops at
the modular-storage owner boundary.  A complete SimulationRunner continuation
still requires the runner checkpoint and RNG envelopes owned elsewhere.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Final, Mapping, Sequence

import torch

from agents.modular_genome_v1 import (
    B2_CAPACITY_PROFILES,
    canonical_hash_from_genome_dict,
    canonical_payload_from_genome_dict,
)
from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    neutral_developmental_expression_record,
    normalize_developmental_expression_record,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    normalize_activity_module_params,
)
from perf.cuda_architecture_b_modular_capacity_epoch import (
    ModularCapacityEpochError,
    ModularCapacityEpochMigration,
    capacity_epoch_authorization_receipt_sha256,
    migrate_modular_capacity_epoch_projection,
)
from perf.cuda_architecture_b_modular_lifecycle import (
    REPLACEMENT_FIELD_NAMES,
    ModularLifecycleResult,
    prune_and_reuse_modular_slots,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_PROJECTION_PREFIX,
    PackedModularStorageV1,
    modular_storage_from_projection,
    modular_storage_projection_payload,
    pack_modular_storage,
    validate_modular_storage,
)


BOUNDARY_ACTION_SCHEMA_VERSION: Final[str] = "eiel_phase_f_boundary_action_v2"
PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_f_phase_e_receipt_binding_placeholder_v1"
)
CHECKPOINT_SCHEDULE_SCHEMA_VERSION: Final[str] = "eiel_phase_f_checkpoint_schedule_v1"
MODULAR_CHECKPOINT_SCHEMA_VERSION: Final[str] = "eiel_phase_f_modular_checkpoint_v1"

RECORD_ACTIONS: Final[frozenset[str]] = frozenset(
    {"reset_member_state", "block_dependency", "enable_member_dormancy"}
)
CHECKPOINT_SCHEDULE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "pre_tick_fresh_owner",
        "post_reuse_pre_tick",
        "post_migration_pre_tick",
        "pre_migration_source",
    }
)
FULL_RUNTIME_RESUME_PREREQUISITES: Final[tuple[str, ...]] = (
    "resident_owner_checkpoint_projection",
    "simulation_runner_checkpoint_bundle",
    "authoritative_rng_state_and_tensor_projection",
    "generation_and_tick_boundary_metadata",
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_INDEX_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "module_local_id_index",
        "module_instance_id_index",
        "module_homology_id_index",
        "module_type_index",
        "input_semantic_id_index",
        "output_semantic_id_index",
        "development_prerequisite_homology_index",
    }
)
_BIOLOGICAL_INDEX_FIELDS: Final[tuple[str, ...]] = (
    "module_local_id_index",
    "module_instance_id_index",
    "module_homology_id_index",
)

BOUNDARY_ACTION_SCHEMA_SHA256: Final[str] = canonical_sha256(
    {
        "schema_version": BOUNDARY_ACTION_SCHEMA_VERSION,
        "record_actions": sorted(RECORD_ACTIONS),
        "checkpoint_schedule_kinds": sorted(CHECKPOINT_SCHEDULE_KINDS),
        "rng_draw_count": 0,
        "full_runtime_resume_prerequisites": list(FULL_RUNTIME_RESUME_PREREQUISITES),
    }
)


class PhaseFBoundaryActionError(RuntimeError):
    """A requested Phase F boundary action is incomplete or unsafe."""


@dataclass(frozen=True)
class RecordBoundaryActionResult:
    record: dict[str, Any]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class PreparedPruneReuseResult:
    lifecycle: ModularLifecycleResult
    receipt: dict[str, Any]


@dataclass(frozen=True)
class BoundCapacityMigrationResult:
    migration: ModularCapacityEpochMigration
    receipt: dict[str, Any]


@dataclass(frozen=True)
class PhaseFModularCheckpoint:
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class FreshModularOwnerRestore:
    owner: PackedModularStorageV1
    receipt: dict[str, Any]


def _exact_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PhaseFBoundaryActionError(f"{field} must be an exact SHA-256")
    return value


def _receipt(material: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": BOUNDARY_ACTION_SCHEMA_VERSION,
        "schema_sha256": BOUNDARY_ACTION_SCHEMA_SHA256,
        **deepcopy(dict(material)),
    }
    payload["action_sha256"] = canonical_sha256(payload)
    return payload


def _validate_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseFBoundaryActionError("boundary receipt must be a mapping")
    payload = deepcopy(dict(value))
    supplied = payload.pop("action_sha256", None)
    if payload.get("schema_version") != BOUNDARY_ACTION_SCHEMA_VERSION:
        raise PhaseFBoundaryActionError("boundary receipt schema mismatch")
    if payload.get("schema_sha256") != BOUNDARY_ACTION_SCHEMA_SHA256:
        raise PhaseFBoundaryActionError("boundary receipt schema digest mismatch")
    if supplied != canonical_sha256(payload):
        raise PhaseFBoundaryActionError("boundary receipt action digest mismatch")
    payload["action_sha256"] = supplied
    return payload


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise PhaseFBoundaryActionError("Phase F digest tensors must be CPU tensors")
    return sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def _planes_sha256(planes: Mapping[str, torch.Tensor]) -> str:
    manifest = [
        {
            "name": name,
            "dtype": str(planes[name].dtype).removeprefix("torch."),
            "shape": list(planes[name].shape),
            "sha256": _tensor_sha256(planes[name]),
        }
        for name in sorted(planes)
    ]
    return canonical_sha256(manifest)


def _projection_sha256(
    metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]
) -> str:
    return canonical_sha256(
        {
            "metadata": deepcopy(dict(metadata)),
            "tensor_manifest": [
                {
                    "name": name,
                    "dtype": str(tensors[name].dtype).removeprefix("torch."),
                    "shape": list(tensors[name].shape),
                    "sha256": _tensor_sha256(tensors[name]),
                }
                for name in sorted(tensors)
            ],
        }
    )


def _exact_nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _exact_nested_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _profile_record(record: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    if profile_id not in B2_CAPACITY_PROFILES:
        raise PhaseFBoundaryActionError(f"unsupported profile {profile_id!r}")
    if not isinstance(record, Mapping):
        raise PhaseFBoundaryActionError("agent record must be a mapping")
    result = deepcopy(dict(record))
    genome = result.get("genome")
    if not isinstance(genome, dict):
        raise PhaseFBoundaryActionError("agent record genome is missing")
    if genome.get("b2_caps") != B2_CAPACITY_PROFILES[profile_id]:
        raise PhaseFBoundaryActionError("agent record capacity profile mismatch")
    # Cross the production packer before and after every record action.
    pack_modular_storage([result], agent_capacity=1, profile_id=profile_id)
    return result


def _module_by_instance(
    modules: Sequence[Mapping[str, Any]], module_instance_id: str
) -> dict[str, Any]:
    matches = [
        module
        for module in modules
        if str(module.get("module_instance_id", "")) == module_instance_id
    ]
    if len(matches) != 1:
        raise PhaseFBoundaryActionError(
            f"module instance must resolve exactly once: {module_instance_id!r}"
        )
    return matches[0]


def apply_record_boundary_action(
    record: Mapping[str, Any],
    *,
    profile_id: str,
    action: str,
    target_module_instance_id: str,
    prerequisite_module_instance_id: str | None = None,
) -> RecordBoundaryActionResult:
    """Apply one deterministic F6-F8 record-level action with zero RNG."""

    if action not in RECORD_ACTIONS:
        raise PhaseFBoundaryActionError(
            f"unsupported record boundary action {action!r}"
        )
    target_id = str(target_module_instance_id or "").strip()
    if not target_id:
        raise PhaseFBoundaryActionError("target module instance ID is required")
    result = _profile_record(record, profile_id)
    genome = deepcopy(result["genome"])
    modules = [deepcopy(dict(module)) for module in genome["modules"]]
    target = _module_by_instance(modules, target_id)
    affected = [target_id]
    prerequisite_id: str | None = None

    if action == "reset_member_state":
        states = result.get("structural_live_module_state")
        outputs = result.get("structural_live_module_outputs")
        if not isinstance(states, dict) or target_id not in states:
            raise PhaseFBoundaryActionError("target member has no live state to reset")
        if not isinstance(outputs, dict) or target_id not in outputs:
            raise PhaseFBoundaryActionError("target member has no live output to reset")
        states[target_id] = 0.0
        outputs[target_id] = 0.0

    elif action == "enable_member_dormancy":
        params = deepcopy(dict(target.get("params", {})))
        params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": True,
        }
        target["params"] = normalize_activity_module_params(params)

    else:
        prerequisite_id = str(prerequisite_module_instance_id or "").strip()
        if not prerequisite_id or prerequisite_id == target_id:
            raise PhaseFBoundaryActionError(
                "a distinct prerequisite module instance ID is required"
            )
        prerequisite = _module_by_instance(modules, prerequisite_id)
        affected.append(prerequisite_id)
        prerequisite_homology = str(prerequisite.get("homology_id", "")).strip()
        if not prerequisite_homology:
            raise PhaseFBoundaryActionError("prerequisite homology identity is missing")

        target_params = deepcopy(dict(target.get("params", {})))
        raw_development = target_params.get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
        if raw_development is None:
            raw_development = neutral_developmental_expression_record()
        development = normalize_developmental_expression_record(
            raw_development,
            module_homology_id=str(target.get("homology_id", "")),
        )
        assert development is not None
        prerequisites = set(development["prerequisite_homology_ids"])
        prerequisites.add(prerequisite_homology)
        development["prerequisite_homology_ids"] = sorted(prerequisites)
        development["latch_dependency_readiness"] = False
        target_params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = (
            normalize_developmental_expression_record(
                development,
                module_homology_id=str(target.get("homology_id", "")),
            )
        )
        target["params"] = target_params

        prerequisite_params = deepcopy(dict(prerequisite.get("params", {})))
        prerequisite_params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": True,
        }
        prerequisite["params"] = normalize_activity_module_params(prerequisite_params)
        latch = result.get("structural_development_latch_by_instance_id")
        effective = result.get("structural_development_effective_gain_by_instance_id")
        if not isinstance(latch, dict) or not isinstance(effective, dict):
            raise PhaseFBoundaryActionError(
                "dependency block requires explicit runtime development maps"
            )
        latch[target_id] = False
        effective[target_id] = 0.0

    source_genome_sha256 = canonical_hash_from_genome_dict(result["genome"])
    genome["modules"] = modules
    result["genome"] = canonical_payload_from_genome_dict(genome)
    packed = pack_modular_storage([result], agent_capacity=1, profile_id=profile_id)
    normalized = packed.records_sidecar[0]
    assert normalized is not None
    receipt = _receipt(
        {
            "action": action,
            "status": "applied_cpu",
            "source_profile_id": profile_id,
            "target_profile_id": profile_id,
            "agent_id": int(normalized["agent_id"]),
            "agent_slot_token": int(normalized["agent_slot_token"]),
            "target_module_instance_id": target_id,
            "prerequisite_module_instance_id": prerequisite_id,
            "affected_module_instance_ids": sorted(affected),
            "source_genome_sha256": source_genome_sha256,
            "target_genome_sha256": canonical_hash_from_genome_dict(
                normalized["genome"]
            ),
            "target_packed_sha256": packed.packed_sha256,
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )
    return RecordBoundaryActionResult(record=deepcopy(normalized), receipt=receipt)


def _slot_for_instance(
    packed: PackedModularStorageV1, *, agent_row: int, module_instance_id: str
) -> int:
    if not 0 <= agent_row < packed.layout.agent_capacity:
        raise PhaseFBoundaryActionError("agent row is outside packed capacity")
    try:
        identity_index = packed.identity_table.index(module_instance_id)
    except ValueError as exc:
        raise PhaseFBoundaryActionError(
            f"module instance identity is absent: {module_instance_id!r}"
        ) from exc
    matches = torch.nonzero(
        torch.logical_and(
            packed.tensors["module_occupied"][agent_row],
            packed.tensors["module_instance_id_index"][agent_row] == identity_index,
        ),
        as_tuple=False,
    )
    if tuple(matches.shape) != (1, 1):
        raise PhaseFBoundaryActionError(
            f"module instance must occupy exactly one slot: {module_instance_id!r}"
        )
    return int(matches[0, 0].item())


def _identity_at(packed: PackedModularStorageV1, index: int) -> str:
    if not 0 <= index < len(packed.identity_table):
        raise PhaseFBoundaryActionError("replacement identity index is unresolved")
    return packed.identity_table[index]


def _identity_index_map(
    source: PackedModularStorageV1,
    admitted_identity_indices: Mapping[str, int],
) -> dict[str, int]:
    if not isinstance(admitted_identity_indices, Mapping):
        raise PhaseFBoundaryActionError("admitted identity indices must be a mapping")
    resolved = {value: index for index, value in enumerate(source.identity_table)}
    used = set(resolved.values())
    for raw_identity, raw_index in admitted_identity_indices.items():
        identity = str(raw_identity)
        if not identity or type(raw_index) is not int:
            raise PhaseFBoundaryActionError("admitted identity binding is malformed")
        if identity in resolved:
            if resolved[identity] != raw_index:
                raise PhaseFBoundaryActionError(
                    "admitted identity contradicts the source identity table"
                )
            continue
        if raw_index < len(source.identity_table) or raw_index in used:
            raise PhaseFBoundaryActionError(
                "new identity index must be fresh and collision-free"
            )
        resolved[identity] = raw_index
        used.add(raw_index)
    return resolved


def _remap_index_tensor(
    value: torch.Tensor,
    *,
    replacement_authority: PackedModularStorageV1,
    identity_indices: Mapping[str, int],
) -> torch.Tensor:
    result = value.clone()
    for index in sorted({int(item) for item in value.reshape(-1).tolist()}):
        if index < 0:
            continue
        identity = _identity_at(replacement_authority, index)
        if identity not in identity_indices:
            raise PhaseFBoundaryActionError(
                f"replacement identity was not admitted: {identity!r}"
            )
        result[value == index] = identity_indices[identity]
    return result


def prepare_prune_reuse_boundary(
    source: PackedModularStorageV1,
    replacement_authority: PackedModularStorageV1,
    *,
    decisions: Sequence[Mapping[str, Any]],
    admitted_identity_indices: Mapping[str, int],
) -> PreparedPruneReuseResult:
    """Execute prepared same-slot reuse through the public lifecycle authority."""

    validate_modular_storage(source)
    validate_modular_storage(replacement_authority)
    if source.layout.profile_id != replacement_authority.layout.profile_id:
        raise PhaseFBoundaryActionError("prune/reuse profile mismatch")
    if not decisions:
        raise PhaseFBoundaryActionError("at least one prune/reuse decision is required")
    identity_indices = _identity_index_map(source, admitted_identity_indices)
    rows: list[int] = []
    source_slots: list[int] = []
    replacement_rows: list[int] = []
    replacement_slots: list[int] = []
    bindings: list[dict[str, Any]] = []
    for raw in decisions:
        if not isinstance(raw, Mapping) or set(raw) != {
            "agent_row",
            "pruned_module_instance_id",
            "replacement_agent_row",
            "replacement_module_instance_id",
        }:
            raise PhaseFBoundaryActionError("prune/reuse decision fields must be exact")
        row = raw["agent_row"]
        replacement_row = raw["replacement_agent_row"]
        if type(row) is not int or type(replacement_row) is not int:
            raise PhaseFBoundaryActionError("prune/reuse rows must be integers")
        pruned_id = str(raw["pruned_module_instance_id"])
        replacement_id = str(raw["replacement_module_instance_id"])
        source_slot = _slot_for_instance(
            source, agent_row=row, module_instance_id=pruned_id
        )
        replacement_slot = _slot_for_instance(
            replacement_authority,
            agent_row=replacement_row,
            module_instance_id=replacement_id,
        )
        biological: dict[str, str] = {}
        for name in _BIOLOGICAL_INDEX_FIELDS:
            replacement_index = int(
                replacement_authority.tensors[name][
                    replacement_row, replacement_slot
                ].item()
            )
            identity = _identity_at(replacement_authority, replacement_index)
            biological[name] = identity
            if identity not in identity_indices:
                raise PhaseFBoundaryActionError(
                    f"replacement biological identity was not admitted: {identity!r}"
                )
        rows.append(row)
        source_slots.append(source_slot)
        replacement_rows.append(replacement_row)
        replacement_slots.append(replacement_slot)
        bindings.append(
            {
                "agent_row": row,
                "agent_id": int(source.tensors["agent_id"][row].item()),
                "agent_slot_token": int(source.tensors["agent_slot_token"][row].item()),
                "module_slot": source_slot,
                "pruned_module_instance_id": pruned_id,
                "replacement_module_instance_id": replacement_id,
                "replacement_biological_identities": biological,
                "replacement_identity_indices": {
                    identity: identity_indices[identity]
                    for identity in sorted(biological.values())
                },
            }
        )

    if len(rows) != len(set(rows)):
        raise PhaseFBoundaryActionError("duplicate per-agent prune/reuse decision")
    replacement: dict[str, torch.Tensor] = {}
    for name in REPLACEMENT_FIELD_NAMES:
        if name.startswith("structural_cost_"):
            value = replacement_authority.tensors[name][replacement_rows].clone()
        else:
            value = replacement_authority.tensors[name][
                replacement_rows, replacement_slots
            ].clone()
        if name in _INDEX_FIELDS:
            value = _remap_index_tensor(
                value,
                replacement_authority=replacement_authority,
                identity_indices=identity_indices,
            )
        replacement[name] = value.contiguous()

    decision_count = len(rows)
    for name in ("module_state_present", "module_output_present"):
        replacement[name].fill_(False)
    for name in ("module_state", "module_previous_output"):
        replacement[name].zero_()
    replacement["development_prerequisite_count"].zero_()
    replacement["development_prerequisite_homology_index"].fill_(-1)
    replacement["development_prerequisite_module_slot"].fill_(-1)
    for name in (
        "development_latch_enabled",
        "development_latch_present",
        "development_latch_state",
        "development_effective_gain_present",
    ):
        replacement[name].fill_(False)
    replacement["development_effective_gain"].zero_()
    assert replacement["module_occupied"].shape[0] == decision_count

    owner_bindings = {
        "slot_to_agent_id": source.tensors["agent_id"].clone(),
        "slot_tokens": source.tensors["agent_slot_token"].clone(),
        "alive_mask": source.tensors["agent_occupied"].clone(),
    }
    lifecycle = prune_and_reuse_modular_slots(
        {name: value.clone() for name, value in source.tensors.items()},
        owner_bindings,
        profile_id=source.layout.profile_id,
        agent_rows=torch.tensor(rows, dtype=torch.int64),
        module_slots=torch.tensor(source_slots, dtype=torch.int64),
        replacement=replacement,
        cpu_validation_mode=True,
    )
    lifecycle_receipt = lifecycle.receipt.to_dict()
    receipt = _receipt(
        {
            "action": "prepared_prune_reuse",
            "status": "executed_public_cpu_lifecycle",
            "source_profile_id": source.layout.profile_id,
            "target_profile_id": source.layout.profile_id,
            "source_packed_sha256": source.packed_sha256,
            "source_identity_table_sha256": canonical_sha256(
                list(source.identity_table)
            ),
            "decision_bindings": bindings,
            "lifecycle_receipt_sha256": canonical_sha256(lifecycle_receipt),
            "result_planes_sha256": _planes_sha256(lifecycle.planes),
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )
    return PreparedPruneReuseResult(lifecycle=lifecycle, receipt=receipt)


def _validate_phase_e_receipt_binding(
    binding: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    source_packed_sha256: str,
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "verification_status",
        "phase_e_final_receipt_sha256",
        "phase_e_execution_commit",
        "matrix_sha256",
        "case_sha256",
        "authorization_receipt_sha256",
        "rollback_checkpoint_digest",
        "source_profile_id",
        "target_profile_id",
    }
    if not isinstance(binding, Mapping) or set(binding) != expected:
        raise PhaseFBoundaryActionError(
            "external Phase E receipt binding fields must be exact"
        )
    payload = deepcopy(dict(binding))
    if payload["schema_version"] != PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION:
        raise PhaseFBoundaryActionError("Phase E receipt binding schema mismatch")
    if payload["verification_status"] != "externally_verified":
        raise PhaseFBoundaryActionError(
            "Phase E final receipt has not been externally verified"
        )
    for field in (
        "phase_e_final_receipt_sha256",
        "matrix_sha256",
        "case_sha256",
        "authorization_receipt_sha256",
        "rollback_checkpoint_digest",
    ):
        _exact_sha256(payload[field], field=field)
    if (
        type(payload["phase_e_execution_commit"]) is not str
        or _COMMIT_RE.fullmatch(payload["phase_e_execution_commit"]) is None
    ):
        raise PhaseFBoundaryActionError(
            "phase_e_execution_commit must be an exact Git commit"
        )
    if payload["source_profile_id"] != "b2-12x32-v1":
        raise PhaseFBoundaryActionError("Phase E binding source profile mismatch")
    if payload["target_profile_id"] != "b2-16x48-v1":
        raise PhaseFBoundaryActionError("Phase E binding target profile mismatch")
    if payload["rollback_checkpoint_digest"] != source_packed_sha256:
        raise PhaseFBoundaryActionError(
            "Phase E binding does not name the source packed checkpoint"
        )
    authorization_sha = capacity_epoch_authorization_receipt_sha256(authorization)
    if payload["authorization_receipt_sha256"] != authorization_sha:
        raise PhaseFBoundaryActionError(
            "Phase E binding does not name the supplied authorization"
        )
    if authorization.get("authorization_receipt_sha256") != authorization_sha:
        raise PhaseFBoundaryActionError("authorization self-digest mismatch")
    if authorization.get("rollback_checkpoint_digest") != source_packed_sha256:
        raise PhaseFBoundaryActionError(
            "authorization does not name the source packed checkpoint"
        )
    if authorization.get("source_profile_id") != payload["source_profile_id"]:
        raise PhaseFBoundaryActionError("authorization source profile mismatch")
    if authorization.get("target_profile_id") != payload["target_profile_id"]:
        raise PhaseFBoundaryActionError("authorization target profile mismatch")
    return payload


def execute_bound_capacity_migration(
    *,
    source_metadata: Mapping[str, Any],
    source_tensors: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
    phase_e_receipt_binding: Mapping[str, Any] | None,
    rng_state_before: Mapping[str, Any],
    rng_state_after: Mapping[str, Any],
    rng_tensors_before: Mapping[str, Any],
    rng_tensors_after: Mapping[str, Any],
) -> BoundCapacityMigrationResult:
    """Execute migration only with externally supplied and bound authority."""

    source = modular_storage_from_projection(source_metadata, source_tensors)
    if source is None:
        raise PhaseFBoundaryActionError("source modular projection is missing")
    if source.layout.profile_id != "b2-12x32-v1":
        raise PhaseFBoundaryActionError("migration source must use b2-12x32-v1")
    if authorization is None:
        raise PhaseFBoundaryActionError("capacity migration authorization is required")
    if phase_e_receipt_binding is None:
        raise PhaseFBoundaryActionError(
            "externally verified Phase E final receipt binding is required"
        )
    binding = _validate_phase_e_receipt_binding(
        phase_e_receipt_binding,
        authorization=authorization,
        source_packed_sha256=source.packed_sha256,
    )
    migration = migrate_modular_capacity_epoch_projection(
        source_metadata=source_metadata,
        source_tensors=source_tensors,
        authorization=authorization,
        rng_state_before=rng_state_before,
        rng_state_after=rng_state_after,
        rng_tensors_before=rng_tensors_before,
        rng_tensors_after=rng_tensors_after,
    )
    migration_receipt = migration.receipt.to_dict()
    target = modular_storage_from_projection(migration.metadata, migration.tensors)
    if target is None:
        raise PhaseFBoundaryActionError(
            "migration target modular projection is missing"
        )
    receipt = _receipt(
        {
            "action": "accepted_capacity_migration",
            "status": "executed_public_cpu_migration",
            "source_profile_id": "b2-12x32-v1",
            "target_profile_id": "b2-16x48-v1",
            "source_packed_sha256": source.packed_sha256,
            "phase_e_receipt_binding_sha256": canonical_sha256(binding),
            "phase_e_final_receipt_sha256": binding["phase_e_final_receipt_sha256"],
            "authorization_receipt_sha256": binding["authorization_receipt_sha256"],
            "migration_receipt_sha256": canonical_sha256(migration_receipt),
            "target_packed_sha256": target.packed_sha256,
            "target_projection_sha256": _projection_sha256(
                migration.metadata, migration.tensors
            ),
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )
    return BoundCapacityMigrationResult(migration=migration, receipt=receipt)


def assert_capacity_migration_rejected(
    *,
    source_metadata: Mapping[str, Any],
    source_tensors: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
    expected_rejection: str,
    rng_state_before: Mapping[str, Any],
    rng_state_after: Mapping[str, Any],
    rng_tensors_before: Mapping[str, Any],
    rng_tensors_after: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove an intended rejected F11 request is rejected by production code."""

    if expected_rejection not in {"unauthorized_request", "wrong_safe_boundary"}:
        raise PhaseFBoundaryActionError("unsupported migration rejection category")
    source = modular_storage_from_projection(source_metadata, source_tensors)
    if source is None:
        raise PhaseFBoundaryActionError("source modular projection is missing")
    if not _exact_nested_equal(rng_state_before, rng_state_after) or not (
        _exact_nested_equal(rng_tensors_before, rng_tensors_after)
    ):
        raise PhaseFBoundaryActionError(
            "rejected migration must preserve the exact RNG state"
        )
    source_projection_before = _projection_sha256(source_metadata, source_tensors)
    try:
        migrate_modular_capacity_epoch_projection(
            source_metadata=source_metadata,
            source_tensors=source_tensors,
            authorization=authorization,  # type: ignore[arg-type]
            rng_state_before=rng_state_before,
            rng_state_after=rng_state_after,
            rng_tensors_before=rng_tensors_before,
            rng_tensors_after=rng_tensors_after,
        )
    except ModularCapacityEpochError as exc:
        message = str(exc)
    else:
        raise PhaseFBoundaryActionError(
            "migration unexpectedly succeeded when rejection was required"
        )
    if _projection_sha256(source_metadata, source_tensors) != source_projection_before:
        raise PhaseFBoundaryActionError(
            "rejected migration mutated the source projection"
        )
    lowered = message.lower()
    if expected_rejection == "unauthorized_request":
        accepted_reason = "authorization" in lowered or "not authorized" in lowered
    else:
        accepted_reason = "safe boundary" in lowered
    if not accepted_reason:
        raise PhaseFBoundaryActionError(
            "migration failed for a different reason than the frozen recipe"
        )
    request = None if authorization is None else deepcopy(dict(authorization))
    return _receipt(
        {
            "action": "rejected_capacity_migration",
            "status": "rejected_by_public_cpu_migration",
            "expected_rejection": expected_rejection,
            "observed_error": message,
            "source_profile_id": source.layout.profile_id,
            "target_profile_id": "b2-16x48-v1",
            "source_packed_sha256": source.packed_sha256,
            "request_sha256": canonical_sha256(request),
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )


def build_checkpoint_schedule_descriptor(
    *,
    case_id: str,
    case_sha256: str,
    matrix_sha256: str,
    profile_id: str,
    schedule_kind: str,
    prerequisite_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a frozen checkpoint point and require prior boundary authority."""

    if schedule_kind not in CHECKPOINT_SCHEDULE_KINDS:
        raise PhaseFBoundaryActionError("unsupported checkpoint schedule kind")
    _exact_sha256(case_sha256, field="case_sha256")
    _exact_sha256(matrix_sha256, field="matrix_sha256")
    if profile_id not in B2_CAPACITY_PROFILES:
        raise PhaseFBoundaryActionError("unsupported checkpoint profile")
    required_action = {
        "post_reuse_pre_tick": "prepared_prune_reuse",
        "post_migration_pre_tick": "accepted_capacity_migration",
    }.get(schedule_kind)
    prerequisite_sha256: str | None = None
    prerequisite_result_planes_sha256: str | None = None
    prerequisite_target_packed_sha256: str | None = None
    if required_action is not None:
        if prerequisite_receipt is None:
            raise PhaseFBoundaryActionError(
                f"{schedule_kind} requires a {required_action} receipt"
            )
        validated = _validate_receipt(prerequisite_receipt)
        if validated.get("action") != required_action:
            raise PhaseFBoundaryActionError(
                f"checkpoint prerequisite must be {required_action}"
            )
        prerequisite_sha256 = str(validated["action_sha256"])
        if required_action == "prepared_prune_reuse":
            prerequisite_result_planes_sha256 = _exact_sha256(
                validated.get("result_planes_sha256"),
                field="result_planes_sha256",
            )
        else:
            prerequisite_target_packed_sha256 = _exact_sha256(
                validated.get("target_packed_sha256"),
                field="target_packed_sha256",
            )
    if schedule_kind == "post_migration_pre_tick" and profile_id != "b2-16x48-v1":
        raise PhaseFBoundaryActionError(
            "post-migration checkpoint requires b2-16x48-v1"
        )
    if schedule_kind == "pre_migration_source" and profile_id != "b2-12x32-v1":
        raise PhaseFBoundaryActionError("pre-migration checkpoint requires b2-12x32-v1")
    return _receipt(
        {
            "action": "checkpoint_schedule",
            "schedule_schema_version": CHECKPOINT_SCHEDULE_SCHEMA_VERSION,
            "status": "prepared_modular_checkpoint_schedule",
            "case_id": str(case_id),
            "case_sha256": case_sha256,
            "matrix_sha256": matrix_sha256,
            "schedule_kind": schedule_kind,
            "source_profile_id": profile_id,
            "target_profile_id": profile_id,
            "prerequisite_action_sha256": prerequisite_sha256,
            "prerequisite_result_planes_sha256": (prerequisite_result_planes_sha256),
            "prerequisite_target_packed_sha256": (prerequisite_target_packed_sha256),
            "fresh_owner_required": True,
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )


def capture_modular_checkpoint(
    authority: PackedModularStorageV1,
    schedule: Mapping[str, Any],
) -> PhaseFModularCheckpoint:
    """Capture an exact modular-only checkpoint at a validated schedule point."""

    validate_modular_storage(authority)
    validated_schedule = _validate_receipt(schedule)
    if validated_schedule.get("action") != "checkpoint_schedule":
        raise PhaseFBoundaryActionError("checkpoint schedule receipt is required")
    if validated_schedule.get("source_profile_id") != authority.layout.profile_id:
        raise PhaseFBoundaryActionError("checkpoint schedule profile mismatch")
    if validated_schedule.get("schedule_kind") == "post_reuse_pre_tick":
        if _planes_sha256(authority.tensors) != validated_schedule.get(
            "prerequisite_result_planes_sha256"
        ):
            raise PhaseFBoundaryActionError(
                "post-reuse checkpoint authority does not match lifecycle result"
            )
    if validated_schedule.get("schedule_kind") == "post_migration_pre_tick":
        if authority.packed_sha256 != validated_schedule.get(
            "prerequisite_target_packed_sha256"
        ):
            raise PhaseFBoundaryActionError(
                "post-migration checkpoint authority does not match migration target"
            )
    projection = modular_storage_projection_payload(authority)
    projection_sha = _projection_sha256(projection["metadata"], projection["tensors"])
    receipt = _receipt(
        {
            "action": "capture_modular_checkpoint",
            "checkpoint_schema_version": MODULAR_CHECKPOINT_SCHEMA_VERSION,
            "status": "captured_modular_storage_owner",
            "checkpoint_scope": "fresh_modular_storage_owner_only",
            "schedule_action_sha256": validated_schedule["action_sha256"],
            "schedule_kind": validated_schedule["schedule_kind"],
            "source_profile_id": authority.layout.profile_id,
            "target_profile_id": authority.layout.profile_id,
            "source_packed_sha256": authority.packed_sha256,
            "identity_table_sha256": canonical_sha256(list(authority.identity_table)),
            "slot_bindings_sha256": canonical_sha256(
                {
                    "agent_id": authority.tensors["agent_id"].tolist(),
                    "agent_slot_token": authority.tensors["agent_slot_token"].tolist(),
                    "agent_occupied": authority.tensors["agent_occupied"].tolist(),
                }
            ),
            "projection_sha256": projection_sha,
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )
    return PhaseFModularCheckpoint(
        metadata=deepcopy(projection["metadata"]),
        tensors={name: value.clone() for name, value in projection["tensors"].items()},
        receipt=receipt,
    )


def restore_fresh_modular_owner(
    checkpoint: PhaseFModularCheckpoint,
) -> FreshModularOwnerRestore:
    """Restore a detached modular owner; no tick or runner resume is implied."""

    if not isinstance(checkpoint, PhaseFModularCheckpoint):
        raise PhaseFBoundaryActionError("Phase F modular checkpoint is required")
    validated = _validate_receipt(checkpoint.receipt)
    if validated.get("action") != "capture_modular_checkpoint":
        raise PhaseFBoundaryActionError("modular checkpoint receipt is required")
    observed_projection_sha = _projection_sha256(
        checkpoint.metadata, checkpoint.tensors
    )
    if observed_projection_sha != validated.get("projection_sha256"):
        raise PhaseFBoundaryActionError("modular checkpoint projection digest mismatch")
    owner = modular_storage_from_projection(checkpoint.metadata, checkpoint.tensors)
    if owner is None:
        raise PhaseFBoundaryActionError("modular checkpoint projection is missing")
    if owner.packed_sha256 != validated.get("source_packed_sha256"):
        raise PhaseFBoundaryActionError("restored modular owner digest mismatch")
    if any(
        owner.tensors[name.removeprefix(MODULAR_STORAGE_PROJECTION_PREFIX)].data_ptr()
        == tensor.data_ptr()
        for name, tensor in checkpoint.tensors.items()
    ):
        raise PhaseFBoundaryActionError("restored owner aliases checkpoint tensors")
    receipt = _receipt(
        {
            "action": "restore_fresh_modular_owner",
            "status": "restored_fresh_modular_storage_owner",
            "checkpoint_scope": "fresh_modular_storage_owner_only",
            "checkpoint_action_sha256": validated["action_sha256"],
            "source_profile_id": owner.layout.profile_id,
            "target_profile_id": owner.layout.profile_id,
            "restored_packed_sha256": owner.packed_sha256,
            "identity_table_sha256": canonical_sha256(list(owner.identity_table)),
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )
    return FreshModularOwnerRestore(owner=owner, receipt=receipt)


def require_full_runtime_resume_prerequisites(
    supplied: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed: this adapter cannot execute a complete runner continuation."""

    supplied_names = set() if supplied is None else set(supplied)
    missing = [
        name for name in FULL_RUNTIME_RESUME_PREREQUISITES if name not in supplied_names
    ]
    if missing:
        raise PhaseFBoundaryActionError(
            "full runtime continuation prerequisites are missing: " + ", ".join(missing)
        )
    raise PhaseFBoundaryActionError(
        "full runtime continuation is outside the modular boundary adapter; "
        "invoke the public ResidentRuntimeState and SimulationRunner checkpoint owners"
    )


__all__ = [
    "BOUNDARY_ACTION_SCHEMA_SHA256",
    "BOUNDARY_ACTION_SCHEMA_VERSION",
    "CHECKPOINT_SCHEDULE_KINDS",
    "CHECKPOINT_SCHEDULE_SCHEMA_VERSION",
    "FULL_RUNTIME_RESUME_PREREQUISITES",
    "MODULAR_CHECKPOINT_SCHEMA_VERSION",
    "PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION",
    "BoundCapacityMigrationResult",
    "FreshModularOwnerRestore",
    "PhaseFBoundaryActionError",
    "PhaseFModularCheckpoint",
    "PreparedPruneReuseResult",
    "RecordBoundaryActionResult",
    "apply_record_boundary_action",
    "assert_capacity_migration_rejected",
    "build_checkpoint_schedule_descriptor",
    "capture_modular_checkpoint",
    "execute_bound_capacity_migration",
    "prepare_prune_reuse_boundary",
    "require_full_runtime_resume_prerequisites",
    "restore_fresh_modular_owner",
]
