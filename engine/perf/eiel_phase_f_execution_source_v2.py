"""Shared CPU-only execution-source preparation for Phase F v2.

The frozen fixtures describe both ordinary tick inputs and actions that must
occur at a public lifecycle or capacity boundary before the one-tick owner is
entered.  This module performs those boundary actions once and returns the
same detached source to the CPU formal oracle and to the accelerated owner.
It never imports a CUDA extension or executes a simulation tick.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import re
from typing import Any, Final, Mapping, Sequence

import torch

from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.matrix import validate_phase_f_v2_matrix
from evolution.structural_sustainable_growth_v1 import (
    hereditary_no_output,
    module_is_dormant,
    mutate_sustainable_growth,
)
from perf.cuda_architecture_b_modular_capacity_epoch import (
    MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION,
    capacity_epoch_authorization_receipt_sha256,
)
from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    PackedModularStorageV1,
    modular_storage_from_projection,
    modular_storage_projection_payload,
    pack_modular_storage,
    validate_modular_storage,
)
from perf.cuda_architecture_b_modular_state_bank import (
    validate_modular_cpu_boundary_projection,
)
from perf.eiel_phase_f_boundary_actions_v2 import (
    BOUNDARY_ACTION_SCHEMA_SHA256,
    BOUNDARY_ACTION_SCHEMA_VERSION,
    PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION,
    assert_capacity_migration_rejected,
    execute_bound_capacity_migration,
    prepare_prune_reuse_boundary,
)
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    phase_f_fixture_sha256_v2,
    validate_phase_f_fixture_registry_v2,
)
from perf.eiel_phase_f_forced_combination_source_v2 import (
    AGENT_CAPACITY,
    PANEL_AGENT_COUNT,
    PhaseFV2FormalOracleInput,
    PhaseFV2SourceState,
    materialize_phase_f_source_v2,
    validate_phase_f_source_v2,
)
from perf import eiel_phase_f_forced_combination_source_v2 as _source_v2
from simulation.capacity_epoch import SAFE_BOUNDARY_KIND


SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_execution_source_preparation_v1"
RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_f_v2_execution_source_preparation_receipt_v1"
)
PROVENANCE_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_execution_source_provenance_v1"
MIGRATION_CONTEXT_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_f_v2_exact_migration_context_v1"
)

_MIGRATION_CONTEXT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authorization",
        "phase_e_receipt_binding",
        "rng_state",
        "rng_tensors",
    }
)
_IDENTITY_INDEX_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "module_local_id_index",
        "module_instance_id_index",
        "module_homology_id_index",
        "module_type_index",
        "input_semantic_id_index",
        "output_semantic_id_index",
        "development_prerequisite_homology_index",
        "connection_local_id_index",
        "connection_src_slot_index",
        "connection_dst_slot_index",
        "connection_value_type_index",
        "connection_channel_index",
    }
)
_RUNTIME_ID_MAP_FIELDS: Final[tuple[str, ...]] = (
    "structural_live_module_state",
    "structural_live_module_outputs",
    "structural_development_latch_by_instance_id",
    "structural_development_effective_gain_by_instance_id",
)
_POST_BOUNDARY_CHECKPOINT_KINDS: Final[frozenset[str]] = frozenset(
    {"post_reuse_pre_tick", "post_migration_pre_tick"}
)
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_VALIDATED_INPUT_CACHE: dict[
    tuple[str, str], tuple[dict[str, Any], dict[str, Any]]
] = {}


class PhaseFV2ExecutionSourceError(RuntimeError):
    """The frozen case cannot be prepared through its required public boundary."""


@dataclass(frozen=True)
class PreparedPhaseFV2ExecutionSource:
    """One boundary-prepared source shared by both Phase F execution lanes."""

    schema_version: str
    source_state: PhaseFV2SourceState
    modular_owner: PackedModularStorageV1
    boundary_receipts: tuple[dict[str, Any], ...]
    checkpoint_prerequisite_receipt: dict[str, Any] | None
    preparation_receipt: dict[str, Any]

    @property
    def source(self) -> PhaseFV2SourceState:
        """Short public alias used by battery callers."""

        return self.source_state

    @property
    def owner(self) -> PackedModularStorageV1:
        """The actual packed owner used for checkpoint scheduling."""

        return self.modular_owner

    @property
    def boundary_receipt(self) -> dict[str, Any] | None:
        """The terminal boundary receipt, when the case has one."""

        if not self.boundary_receipts:
            return None
        return deepcopy(self.boundary_receipts[-1])


def _clone_tensors(
    values: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise PhaseFV2ExecutionSourceError(
                f"execution-source tensor must be CPU authority: {name}"
            )
        result[str(name)] = value.detach().clone().contiguous()
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise PhaseFV2ExecutionSourceError("receipt tensor must be CPU authority")
    return sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def _planes_sha256(values: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256(
        [
            {
                "name": name,
                "dtype": str(values[name].dtype).removeprefix("torch."),
                "shape": list(values[name].shape),
                "sha256": _tensor_sha256(values[name]),
            }
            for name in sorted(values)
        ]
    )


def _boundary_receipt(material: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": BOUNDARY_ACTION_SCHEMA_VERSION,
        "schema_sha256": BOUNDARY_ACTION_SCHEMA_SHA256,
        **deepcopy(dict(material)),
    }
    payload["action_sha256"] = canonical_sha256(payload)
    return payload


def _preparation_receipt(material: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        **deepcopy(dict(material)),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _case_map(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["case_id"]): {
            **dict(row),
            "matrix_sha256": str(matrix["matrix_sha256"]),
        }
        for row in matrix["cases"]
    }


def _validated_inputs(
    matrix: Any, fixture_registry: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate once, then cheaply recheck the registry self-digest."""

    try:
        checked_matrix = validate_phase_f_v2_matrix(matrix)
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError("matrix is not current and exact") from exc
    if not isinstance(fixture_registry, Mapping):
        raise PhaseFV2ExecutionSourceError("fixture registry must be a mapping")
    claimed = fixture_registry.get("registry_sha256")
    if type(claimed) is not str or _SHA256_RE.fullmatch(claimed) is None:
        raise PhaseFV2ExecutionSourceError(
            "fixture registry digest is missing or malformed"
        )
    unsigned = dict(fixture_registry)
    unsigned.pop("registry_sha256", None)
    try:
        observed = phase_f_fixture_sha256_v2(unsigned)
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError("fixture registry is not canonical") from exc
    if observed != claimed:
        raise PhaseFV2ExecutionSourceError("fixture registry self digest mismatch")
    key = (str(checked_matrix["matrix_sha256"]), claimed)
    cached = _VALIDATED_INPUT_CACHE.get(key)
    if cached is not None:
        if (
            fixture_registry.get("matrix_sha256") != checked_matrix["matrix_sha256"]
            or fixture_registry.get("matrix_id") != checked_matrix["matrix_id"]
            or fixture_registry.get("case_count") != len(checked_matrix["cases"])
        ):
            raise PhaseFV2ExecutionSourceError(
                "fixture registry cached binding mismatch"
            )
        return cached
    try:
        registry = validate_phase_f_fixture_registry_v2(
            fixture_registry, matrix=checked_matrix
        )
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError(
            "fixture registry is not current and exact"
        ) from exc
    stored = (checked_matrix, registry)
    _VALIDATED_INPUT_CACHE[key] = stored
    return stored


def _fixture_map(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): dict(row) for row in registry["cases"]}


def _source_seed(case: Mapping[str, Any]) -> int:
    return int(case["fixture_blueprint"]["construction_seed"])


def _owner_from_source(source: PhaseFV2SourceState) -> PackedModularStorageV1:
    try:
        owner = modular_storage_from_projection(
            source.owner_metadata, source.generation_tensors
        )
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError(
            "source modular owner projection is invalid"
        ) from exc
    if owner is None:
        raise PhaseFV2ExecutionSourceError("source modular owner projection is missing")
    validate_modular_storage(owner)
    return owner


def _validate_owner_membership(
    source: PhaseFV2SourceState, owner: PackedModularStorageV1
) -> None:
    expected = {
        "agent_occupied": source.generation_tensors["alive_mask"],
        "agent_id": source.generation_tensors["slot_to_agent_id"],
        "agent_slot_token": torch.where(
            source.generation_tensors["alive_mask"],
            source.generation_tensors["slot_tokens"],
            torch.full_like(source.generation_tensors["slot_tokens"], -1),
        ),
    }
    for name, value in expected.items():
        if not torch.equal(owner.tensors[name], value):
            raise PhaseFV2ExecutionSourceError(
                f"packed owner contradicts generation membership: {name}"
            )


def _formal_inputs_from_owner(
    *,
    template: PhaseFV2SourceState,
    owner: PackedModularStorageV1,
    target_case: Mapping[str, Any],
    target_fixture: Mapping[str, Any],
) -> tuple[PhaseFV2FormalOracleInput, ...]:
    if len(template.formal_oracle_inputs) != PANEL_AGENT_COUNT:
        raise PhaseFV2ExecutionSourceError("formal input count is not exactly 16")
    outputs: list[PhaseFV2FormalOracleInput] = []
    for slot, prior in enumerate(template.formal_oracle_inputs):
        if (
            int(owner.tensors["agent_id"][slot].item()) != prior.agent_id
            or int(owner.tensors["agent_slot_token"][slot].item())
            != prior.agent_slot_token
        ):
            raise PhaseFV2ExecutionSourceError(
                "formal input owner identity changed at the boundary"
            )
        outputs.append(
            replace(
                prior,
                case_id=str(target_case["case_id"]),
                case_sha256=str(target_case["case_sha256"]),
                fixture_sha256=str(target_fixture["fixture_sha256"]),
                profile_id=str(target_case["profile_id"]),
                planes={
                    name: value[slot : slot + 1].detach().clone().contiguous()
                    for name, value in owner.tensors.items()
                },
            )
        )
    return tuple(outputs)


def rebuild_phase_f_source_from_modular_owner_v2(
    *,
    source_state: PhaseFV2SourceState,
    modular_owner: PackedModularStorageV1,
    target_case: Mapping[str, Any],
    target_fixture_case: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> PhaseFV2SourceState:
    """Rebind one real packed owner into both execution planes and both oracles."""

    if not isinstance(source_state, PhaseFV2SourceState):
        raise PhaseFV2ExecutionSourceError("source_state has the wrong type")
    validate_modular_storage(modular_owner)
    if modular_owner.layout.agent_capacity != AGENT_CAPACITY:
        raise PhaseFV2ExecutionSourceError("packed owner capacity is not Phase F exact")
    if modular_owner.layout.profile_id != target_case.get("profile_id"):
        raise PhaseFV2ExecutionSourceError("packed owner target profile mismatch")
    required_case_bindings = {
        "case_id": target_fixture_case.get("case_id"),
        "case_sha256": target_fixture_case.get("case_sha256"),
        "matrix_sha256": target_fixture_case.get("matrix_sha256"),
        "fixture_blueprint_sha256": target_fixture_case.get("fixture_blueprint_sha256"),
        "profile_id": target_fixture_case.get("profile_id"),
    }
    expected_case_bindings = {
        "case_id": target_case.get("case_id"),
        "case_sha256": target_case.get("case_sha256"),
        "matrix_sha256": target_case.get("matrix_sha256"),
        "fixture_blueprint_sha256": dict(
            target_case.get("fixture_blueprint") or {}
        ).get("fixture_blueprint_sha256"),
        "profile_id": target_case.get("profile_id"),
    }
    if required_case_bindings != expected_case_bindings:
        raise PhaseFV2ExecutionSourceError(
            "target fixture lacks its frozen matrix source binding"
        )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or provenance.get("target_case_id") != target_case.get("case_id")
        or provenance.get("target_fixture_sha256")
        != target_fixture_case.get("fixture_sha256")
    ):
        raise PhaseFV2ExecutionSourceError("execution provenance is incomplete")

    projection = modular_storage_projection_payload(modular_owner)
    nonmodular = {
        name: value
        for name, value in source_state.generation_tensors.items()
        if not name.startswith(MODULAR_STORAGE_PROJECTION_PREFIX)
    }
    generation_tensors = {
        **_clone_tensors(nonmodular),
        **_clone_tensors(projection["tensors"]),
    }
    owner_metadata = deepcopy(source_state.owner_metadata)
    owner_metadata.update(
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": str(target_case["case_id"]),
            "case_sha256": str(target_case["case_sha256"]),
            "fixture_sha256": str(target_fixture_case["fixture_sha256"]),
            "matrix_sha256": str(target_case["matrix_sha256"]),
            "profile_id": str(target_case["profile_id"]),
            "role_action": str(target_case["role_contract"]["action"]),
            "packed_storage_sha256": modular_owner.packed_sha256,
            "execution_source_provenance": deepcopy(dict(provenance)),
            MODULAR_STORAGE_METADATA_KEY: deepcopy(
                projection["metadata"][MODULAR_STORAGE_METADATA_KEY]
            ),
        }
    )
    _validate_owner_membership(
        replace(source_state, generation_tensors=generation_tensors),
        modular_owner,
    )
    try:
        overlay = validate_modular_cpu_boundary_projection(
            owner_metadata, generation_tensors, agent_capacity=AGENT_CAPACITY
        )
        _source_v2._validate_generation_tensors(  # noqa: SLF001
            generation_tensors,
            profile_id=str(target_case["profile_id"]),
        )
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError(
            "rebuilt source failed the public generation entry"
        ) from exc
    if overlay is None or overlay.packed.packed_sha256 != modular_owner.packed_sha256:
        raise PhaseFV2ExecutionSourceError("rebuilt owner overlay changed authority")

    formal_inputs = _formal_inputs_from_owner(
        template=source_state,
        owner=modular_owner,
        target_case=target_case,
        target_fixture=target_fixture_case,
    )
    header = {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(target_case["case_id"]),
        "case_sha256": str(target_case["case_sha256"]),
        "fixture_sha256": str(target_fixture_case["fixture_sha256"]),
        "matrix_sha256": str(target_case["matrix_sha256"]),
        "profile_id": str(target_case["profile_id"]),
        "role_action": str(target_case["role_contract"]["action"]),
        # A migrated owner keeps the source-side RNG schedule.  The target
        # construction seed is bound separately in provenance.
        "source_seed": int(source_state.source_seed),
    }
    source_sha256 = _source_v2._source_digest(  # noqa: SLF001
        header=header,
        owner_metadata=owner_metadata,
        config=source_state.generation_config,
        generation_tensors=generation_tensors,
        formal_inputs=formal_inputs,
    )
    rebuilt = PhaseFV2SourceState(
        **header,
        owner_metadata=owner_metadata,
        generation_config=source_state.generation_config,
        generation_tensors=_clone_tensors(generation_tensors),
        formal_oracle_inputs=formal_inputs,
        source_sha256=source_sha256,
    )
    for slot, formal in enumerate(rebuilt.formal_oracle_inputs):
        for name, plane in formal.planes.items():
            projected = rebuilt.generation_tensors[
                f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"
            ][slot : slot + 1]
            if not torch.equal(plane, projected):
                raise PhaseFV2ExecutionSourceError(
                    "formal and generation owner planes diverged"
                )
            if plane.data_ptr() == projected.data_ptr():
                raise PhaseFV2ExecutionSourceError(
                    "formal and generation owner planes alias"
                )
    return rebuilt


def _module_key(module: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(module.get("homology_id", "")),
        str(module.get("module_instance_id", "")),
        str(module.get("module_id", "")),
    )


def _target_module_for_reuse(
    fixture_agent: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    modules = [dict(value) for value in record["genome"]["modules"]]
    if not modules:
        raise PhaseFV2ExecutionSourceError("reuse target has no generic module")
    variant_id = str(fixture_agent["variant_id"])
    candidates = [module for module in modules if hereditary_no_output(module)]
    if variant_id == "v0":
        candidates = [module for module in candidates if module_is_dormant(module)]
    elif variant_id == "v4":
        candidates = [
            module
            for module in candidates
            if "observation_routing" in dict(module.get("params") or {})
            and "output_routing" in dict(module.get("params") or {})
        ]
    elif variant_id == "v5":
        incident_ids = {
            str(connection["src_module_id"])
            for connection in record["genome"]["connections"]
        } | {
            str(connection["dst_module_id"])
            for connection in record["genome"]["connections"]
        }
        candidates = [
            module for module in candidates if str(module["module_id"]) in incident_ids
        ]
    if not candidates:
        raise PhaseFV2ExecutionSourceError(
            f"no exact eligible reuse module for variant {variant_id}"
        )
    return max(candidates, key=_module_key)


def _fresh_identity_suffix(
    *, case_id: str, agent_row: int, prior_instance_id: str
) -> str:
    digest = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "agent_row": agent_row,
            "prior_module_instance_id": prior_instance_id,
        }
    )
    return f"__phase_f_v2_reuse_{digest[:20]}"


def _fresh_reuse_owner(
    *,
    source: PackedModularStorageV1,
    fixture: Mapping[str, Any],
    target_slots: Sequence[int],
) -> tuple[PackedModularStorageV1, list[dict[str, Any]]]:
    records = deepcopy(list(source.records_sidecar))
    decisions: list[dict[str, Any]] = []
    for row in target_slots:
        if type(row) is not int or not 0 <= row < PANEL_AGENT_COUNT:
            raise PhaseFV2ExecutionSourceError("reuse target slot is not exact")
        record = records[row]
        if record is None:
            raise PhaseFV2ExecutionSourceError("reuse target owner is empty")
        fixture_agent = fixture["agents"][row]
        prior = _target_module_for_reuse(fixture_agent, record)
        prior_local_id = str(prior["module_id"])
        prior_instance_id = str(prior["module_instance_id"])
        prior_homology_id = str(prior["homology_id"])
        suffix = _fresh_identity_suffix(
            case_id=str(fixture["case_id"]),
            agent_row=row,
            prior_instance_id=prior_instance_id,
        )
        replacement = deepcopy(prior)
        replacement["module_id"] = f"{prior_local_id}{suffix}"
        replacement["module_instance_id"] = f"{prior_instance_id}{suffix}"
        replacement["homology_id"] = f"{prior_homology_id}{suffix}"
        ancestry = deepcopy(dict(replacement.get("ancestry") or {}))
        ancestry.update(
            {
                "inherited_homology_id": prior_homology_id,
                "origin_operator": "phase_f_v2_prepared_same_slot_reuse",
                "parent_module_instance_ids": [prior_instance_id],
                "origin_mutation_event_id": f"sme_{canonical_sha256({'suffix': suffix, 'kind': 'event'})}",
                "provenance_id": f"sp_{canonical_sha256({'suffix': suffix, 'kind': 'provenance'})}",
            }
        )
        replacement["ancestry"] = ancestry
        record["genome"]["modules"] = [
            replacement
            if str(module["module_instance_id"]) == prior_instance_id
            else module
            for module in record["genome"]["modules"]
        ]
        record["genome"]["connections"] = [
            connection
            for connection in record["genome"]["connections"]
            if str(connection["src_module_id"]) != prior_local_id
            and str(connection["dst_module_id"]) != prior_local_id
        ]
        for field in _RUNTIME_ID_MAP_FIELDS:
            record[field].pop(prior_instance_id, None)
        record["cost_summary"] = None
        decisions.append(
            {
                "agent_row": row,
                "pruned_module_instance_id": prior_instance_id,
                "replacement_agent_row": row,
                "replacement_module_instance_id": str(
                    replacement["module_instance_id"]
                ),
            }
        )
    target = pack_modular_storage(
        records,
        agent_capacity=source.layout.agent_capacity,
        profile_id=source.layout.profile_id,
    )
    return target, decisions


def _canonical_identity_projection(
    value: torch.Tensor,
    *,
    identity_table: tuple[str, ...],
    vocabulary: Mapping[str, int],
) -> torch.Tensor:
    result = value.detach().clone().to(dtype=torch.int64)
    for raw_index in sorted({int(item) for item in value.reshape(-1).tolist()}):
        if raw_index < 0:
            continue
        if not 0 <= raw_index < len(identity_table):
            raise PhaseFV2ExecutionSourceError(
                "lifecycle identity index is out of range"
            )
        result[value == raw_index] = vocabulary[identity_table[raw_index]]
    return result


def _assert_lifecycle_matches_canonical_owner(
    *,
    lifecycle_planes: Mapping[str, torch.Tensor],
    lifecycle_identity_table: tuple[str, ...],
    target: PackedModularStorageV1,
) -> None:
    if set(lifecycle_planes) != set(target.tensors):
        raise PhaseFV2ExecutionSourceError(
            "public lifecycle returned an incomplete plane set"
        )
    vocabulary = {
        identity: index
        for index, identity in enumerate(
            sorted(set(lifecycle_identity_table) | set(target.identity_table))
        )
    }
    for name, expected in target.tensors.items():
        observed = lifecycle_planes[name]
        if name in _IDENTITY_INDEX_FIELDS:
            equal = torch.equal(
                _canonical_identity_projection(
                    observed,
                    identity_table=lifecycle_identity_table,
                    vocabulary=vocabulary,
                ),
                _canonical_identity_projection(
                    expected,
                    identity_table=target.identity_table,
                    vocabulary=vocabulary,
                ),
            )
        else:
            equal = torch.equal(observed, expected)
        if not equal:
            raise PhaseFV2ExecutionSourceError(
                f"public lifecycle diverged from canonical packed owner: {name}"
            )


def _execute_prune_reuse(
    *,
    source: PackedModularStorageV1,
    fixture: Mapping[str, Any],
    target_slots: Sequence[int],
) -> tuple[PackedModularStorageV1, dict[str, Any]]:
    if not target_slots:
        raise PhaseFV2ExecutionSourceError("prune/reuse target slots are empty")
    target, decisions = _fresh_reuse_owner(
        source=source,
        fixture=fixture,
        target_slots=target_slots,
    )
    missing = sorted(set(target.identity_table) - set(source.identity_table))
    admissions = {
        identity: len(source.identity_table) + index
        for index, identity in enumerate(missing)
    }
    try:
        public = prepare_prune_reuse_boundary(
            source,
            target,
            decisions=decisions,
            admitted_identity_indices=admissions,
        )
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError(
            "frozen accepted prune/reuse action failed"
        ) from exc
    lifecycle_identity_table = (*source.identity_table, *missing)
    _assert_lifecycle_matches_canonical_owner(
        lifecycle_planes=public.lifecycle.planes,
        lifecycle_identity_table=lifecycle_identity_table,
        target=target,
    )
    receipt = _boundary_receipt(
        {
            "action": "prepared_prune_reuse",
            "status": "executed_public_cpu_lifecycle_and_canonical_repack",
            "source_profile_id": source.layout.profile_id,
            "target_profile_id": target.layout.profile_id,
            "source_packed_sha256": source.packed_sha256,
            "target_packed_sha256": target.packed_sha256,
            "decision_bindings": deepcopy(public.receipt["decision_bindings"]),
            "public_lifecycle_action_sha256": public.receipt["action_sha256"],
            "public_result_planes_sha256": public.receipt["result_planes_sha256"],
            "result_planes_sha256": _planes_sha256(target.tensors),
            "canonical_identity_repack": True,
            "canonical_identity_table_sha256": canonical_sha256(
                list(target.identity_table)
            ),
            "semantic_repack_verified": True,
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )
    return target, receipt


def _execute_rejected_prune_requests(
    *,
    source: PackedModularStorageV1,
    fixture: Mapping[str, Any],
    target_slots: Sequence[int],
) -> dict[str, Any]:
    if tuple(target_slots) != (2, 3):
        raise PhaseFV2ExecutionSourceError("frozen rejected prune target slots changed")
    before_sha = source.packed_sha256
    torch_rng_before = torch.get_rng_state().clone()
    rejections: list[dict[str, Any]] = []
    expected = {
        2: ("dependency_blocked", "required_by_development"),
        3: ("ineligible", "minimum_module_invariant"),
    }
    for row in target_slots:
        record = source.records_sidecar[row]
        if record is None:
            raise PhaseFV2ExecutionSourceError("rejected prune source is empty")
        result = mutate_sustainable_growth(record["genome"], 0.30, 0.0)
        status, reason = expected[row]
        if (
            result.payload != record["genome"]
            or result.telemetry.get("status") != status
            or result.telemetry.get("reason") != reason
            or result.telemetry.get("draw_budget")
            != {"declared_supplied_draws": 2, "internal_rng_draws": 0}
        ):
            raise PhaseFV2ExecutionSourceError(
                "public sustainable-growth rejection changed semantics"
            )
        rejections.append(
            {
                "agent_row": row,
                "variant_id": fixture["agents"][row]["variant_id"],
                "status": status,
                "reason": reason,
                "telemetry_sha256": canonical_sha256(result.telemetry),
            }
        )
    if not torch.equal(torch_rng_before, torch.get_rng_state()):
        raise PhaseFV2ExecutionSourceError("rejected prune request changed torch RNG")
    validate_modular_storage(source)
    if source.packed_sha256 != before_sha:
        raise PhaseFV2ExecutionSourceError("rejected prune request changed source")
    return _boundary_receipt(
        {
            "action": "rejected_prune_requests",
            "status": "rejected_by_public_sustainable_growth",
            "source_profile_id": source.layout.profile_id,
            "target_profile_id": source.layout.profile_id,
            "source_packed_sha256": source.packed_sha256,
            "result_packed_sha256": source.packed_sha256,
            "rejections": rejections,
            "result_planes_sha256": _planes_sha256(source.tensors),
            "rng_draw_count": 0,
            "runtime_execution_performed": False,
        }
    )


def _rng_authority(
    source: PhaseFV2SourceState,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "schema_version": "eiel_phase_f_v2_boundary_rng_state_v1",
            "case_id": source.case_id,
            "source_seed": source.source_seed,
        },
        {
            "pcg64_state_words": source.generation_tensors["pcg64_state_words"]
            .detach()
            .clone()
            .contiguous(),
            "pcg64_buffer_words": source.generation_tensors["pcg64_buffer_words"]
            .detach()
            .clone()
            .contiguous(),
        },
    )


def _wrong_boundary_authorization(owner: PackedModularStorageV1) -> dict[str, Any]:
    payload = {
        "schema_version": MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_receipt_id": "phase-f-v2-frozen-wrong-boundary-request",
        "authorization_receipt_sha256": "",
        "authorized": True,
        "authorized_generation": 0,
        "next_generation": 1,
        "source_profile_id": "b2-12x32-v1",
        "target_profile_id": "b2-16x48-v1",
        "safe_boundary": "mid_generation",
        "rollback_checkpoint_digest": owner.packed_sha256,
        "rollback_checkpoint_verified": True,
    }
    payload["authorization_receipt_sha256"] = (
        capacity_epoch_authorization_receipt_sha256(payload)
    )
    return payload


def _execute_rejected_migration(
    *,
    source: PhaseFV2SourceState,
    owner: PackedModularStorageV1,
    expected_rejection: str,
) -> dict[str, Any]:
    rng_state, rng_tensors = _rng_authority(source)
    authorization = (
        None
        if expected_rejection == "unauthorized_request"
        else _wrong_boundary_authorization(owner)
    )
    try:
        receipt = assert_capacity_migration_rejected(
            source_metadata=source.owner_metadata,
            source_tensors=source.generation_tensors,
            authorization=authorization,
            expected_rejection=expected_rejection,
            rng_state_before=deepcopy(rng_state),
            rng_state_after=deepcopy(rng_state),
            rng_tensors_before=_clone_tensors(rng_tensors),
            rng_tensors_after=_clone_tensors(rng_tensors),
        )
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError(
            "frozen migration rejection did not occur exactly"
        ) from exc
    validate_modular_storage(owner)
    if receipt.get("source_packed_sha256") != owner.packed_sha256:
        raise PhaseFV2ExecutionSourceError(
            "migration rejection receipt lost source authority"
        )
    return receipt


def _migration_context(
    value: Mapping[str, Any] | None,
    *,
    matrix: Mapping[str, Any],
    source_case: Mapping[str, Any],
    source_owner: PackedModularStorageV1,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MIGRATION_CONTEXT_FIELDS:
        raise PhaseFV2ExecutionSourceError(
            "accepted migration requires an exact Phase E migration context"
        )
    result = deepcopy(dict(value))
    if result["schema_version"] != MIGRATION_CONTEXT_SCHEMA_VERSION:
        raise PhaseFV2ExecutionSourceError("migration context schema mismatch")
    for field in (
        "authorization",
        "phase_e_receipt_binding",
        "rng_state",
        "rng_tensors",
    ):
        if not isinstance(result[field], Mapping):
            raise PhaseFV2ExecutionSourceError(
                f"migration context {field} must be a mapping"
            )
    authorization = result["authorization"]
    binding = result["phase_e_receipt_binding"]
    if (
        authorization.get("rollback_checkpoint_digest") != source_owner.packed_sha256
        or authorization.get("source_profile_id") != "b2-12x32-v1"
        or authorization.get("target_profile_id") != "b2-16x48-v1"
        or authorization.get("safe_boundary") != SAFE_BOUNDARY_KIND
        or authorization.get("authorized") is not True
    ):
        raise PhaseFV2ExecutionSourceError(
            "migration authorization does not bind the exact source owner"
        )
    if (
        binding.get("schema_version") != PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION
        or binding.get("matrix_sha256") != matrix["matrix_sha256"]
        or binding.get("case_sha256") != source_case["case_sha256"]
        or binding.get("rollback_checkpoint_digest") != source_owner.packed_sha256
        or binding.get("authorization_receipt_sha256")
        != authorization.get("authorization_receipt_sha256")
    ):
        raise PhaseFV2ExecutionSourceError(
            "Phase E receipt binding does not name the exact source case"
        )
    return result


def _provenance(
    *,
    preparation_kind: str,
    target_case: Mapping[str, Any],
    target_fixture: Mapping[str, Any],
    source_case: Mapping[str, Any],
    source_fixture: Mapping[str, Any],
    source_state: PhaseFV2SourceState,
    source_owner: PackedModularStorageV1,
    target_owner: PackedModularStorageV1,
    boundary_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "preparation_kind": preparation_kind,
        "target_case_id": target_case["case_id"],
        "target_case_sha256": target_case["case_sha256"],
        "target_fixture_sha256": target_fixture["fixture_sha256"],
        "target_construction_seed": _source_seed(target_case),
        "source_case_id": source_case["case_id"],
        "source_case_sha256": source_case["case_sha256"],
        "source_fixture_sha256": source_fixture["fixture_sha256"],
        "source_construction_seed": _source_seed(source_case),
        "source_state_sha256": source_state.source_sha256,
        "source_packed_sha256": source_owner.packed_sha256,
        "target_packed_sha256": target_owner.packed_sha256,
        "boundary_action_sha256s": [
            receipt["action_sha256"] for receipt in boundary_receipts
        ],
        "rng_draw_count": 0,
        "simulation_executed": False,
    }


def build_phase_f_migration_context_v2(
    *,
    matrix: Any,
    fixture_registry: Any,
    source_case_id: str,
    phase_e_final_receipt_sha256: str,
    phase_e_execution_commit: str,
) -> dict[str, Any]:
    """Build the exact accepted-migration authority for one frozen b12 owner.

    The caller supplies the already-verified Phase E final receipt digest and
    its exact execution commit.  This builder binds those external facts to
    the matrix case, rollback owner, safe boundary, authorization digest, and
    the source's existing RNG authority without drawing randomness.
    """

    checked_matrix, registry = _validated_inputs(matrix, fixture_registry)
    if (
        type(phase_e_final_receipt_sha256) is not str
        or _SHA256_RE.fullmatch(phase_e_final_receipt_sha256) is None
    ):
        raise PhaseFV2ExecutionSourceError(
            "Phase E final receipt digest must be an exact SHA-256"
        )
    if (
        type(phase_e_execution_commit) is not str
        or _COMMIT_RE.fullmatch(phase_e_execution_commit) is None
    ):
        raise PhaseFV2ExecutionSourceError(
            "Phase E execution commit must be an exact Git commit"
        )
    cases = _case_map(checked_matrix)
    fixtures = _fixture_map(registry)
    if (
        type(source_case_id) is not str
        or source_case_id not in cases
        or source_case_id not in fixtures
    ):
        raise PhaseFV2ExecutionSourceError(
            "migration context source case is not frozen"
        )
    source_case = cases[source_case_id]
    if source_case["profile_id"] != "b2-12x32-v1" or not (
        source_case["family_id"] == "f11"
        or (source_case["family_id"] == "f12" and source_case["role_id"] == "r2")
    ):
        raise PhaseFV2ExecutionSourceError(
            "migration context source must be a frozen F11/F12 b12 source"
        )
    fixture = fixtures[source_case_id]
    source = materialize_phase_f_source_v2(
        fixture, source_seed=_source_seed(source_case)
    )
    source = validate_phase_f_source_v2(
        source,
        fixture_case=fixture,
        source_seed=_source_seed(source_case),
    )
    owner = _owner_from_source(source)
    authorization: dict[str, Any] = {
        "schema_version": MODULAR_CAPACITY_EPOCH_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_receipt_id": (
            f"phase-f-v2-{source_case_id}-accepted-capacity-migration"
        ),
        "authorization_receipt_sha256": "",
        "authorized": True,
        "authorized_generation": 0,
        "next_generation": 1,
        "source_profile_id": "b2-12x32-v1",
        "target_profile_id": "b2-16x48-v1",
        "safe_boundary": SAFE_BOUNDARY_KIND,
        "rollback_checkpoint_digest": owner.packed_sha256,
        "rollback_checkpoint_verified": True,
    }
    authorization["authorization_receipt_sha256"] = (
        capacity_epoch_authorization_receipt_sha256(authorization)
    )
    binding = {
        "schema_version": PHASE_E_RECEIPT_BINDING_SCHEMA_VERSION,
        "verification_status": "externally_verified",
        "phase_e_final_receipt_sha256": phase_e_final_receipt_sha256,
        "phase_e_execution_commit": phase_e_execution_commit,
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "case_sha256": source_case["case_sha256"],
        "authorization_receipt_sha256": authorization["authorization_receipt_sha256"],
        "rollback_checkpoint_digest": owner.packed_sha256,
        "source_profile_id": "b2-12x32-v1",
        "target_profile_id": "b2-16x48-v1",
    }
    rng_state, rng_tensors = _rng_authority(source)
    return {
        "schema_version": MIGRATION_CONTEXT_SCHEMA_VERSION,
        "authorization": authorization,
        "phase_e_receipt_binding": binding,
        "rng_state": rng_state,
        "rng_tensors": _clone_tensors(rng_tensors),
    }


def prepare_phase_f_execution_source_v2(
    *,
    matrix: Any,
    fixture_registry: Any,
    case_id: str,
    migration_context: Mapping[str, Any] | None = None,
) -> PreparedPhaseFV2ExecutionSource:
    """Prepare one current case before uninterrupted/resumed execution splits."""

    checked_matrix, registry = _validated_inputs(matrix, fixture_registry)
    cases = _case_map(checked_matrix)
    fixtures = _fixture_map(registry)
    if type(case_id) is not str or case_id not in cases or case_id not in fixtures:
        raise PhaseFV2ExecutionSourceError(
            "requested case lacks a frozen matrix/fixture source binding"
        )
    target_case = cases[case_id]
    target_fixture = fixtures[case_id]
    family = str(target_case["family_id"])
    profile = str(target_case["profile_id"])
    role = str(target_case["role_id"])
    action = str(target_case["role_contract"]["action"])
    source_case = target_case
    source_fixture = target_fixture
    preparation_kind = "unchanged"

    accepted_migration = (family == "f11" and profile == "b2-16x48-v1") or (
        family == "f12" and profile == "b2-16x48-v1" and role == "r2"
    )
    if accepted_migration:
        source_case_id = target_case["role_contract"].get("boundary_source_case_id")
        if (
            type(source_case_id) is not str
            or source_case_id not in cases
            or source_case_id not in fixtures
        ):
            raise PhaseFV2ExecutionSourceError(
                "frozen accepted migration lacks its source binding"
            )
        source_case = cases[source_case_id]
        source_fixture = fixtures[source_case_id]
        if (
            source_case["profile_id"] != "b2-12x32-v1"
            or source_case["family_id"] != target_case["family_id"]
            or source_case["role_id"] != target_case["role_id"]
        ):
            raise PhaseFV2ExecutionSourceError(
                "migration boundary source is not the same-role b12 case"
            )

    source = materialize_phase_f_source_v2(
        source_fixture, source_seed=_source_seed(source_case)
    )
    source = validate_phase_f_source_v2(
        source,
        fixture_case=source_fixture,
        source_seed=_source_seed(source_case),
    )
    source_owner = _owner_from_source(source)
    _validate_owner_membership(source, source_owner)
    target_owner = source_owner
    boundary_receipts: list[dict[str, Any]] = []

    if family == "f09":
        expected_actions = {
            "r0": "pre_boundary_reference",
            "r1": "no_op_prune_sham",
            "r2": "accepted_same_slot_prune_reuse_cleanup",
            "r3": "rejected_prune_request",
            "r4": "reuse_freed_slot_with_new_identity",
        }
        if expected_actions.get(role) != action:
            raise PhaseFV2ExecutionSourceError("frozen F9 role action changed")
        target_slots = tuple(target_case["role_contract"]["target_slots"])
        if role in {"r2", "r4"}:
            target_owner, receipt = _execute_prune_reuse(
                source=source_owner,
                fixture=target_fixture,
                target_slots=target_slots,
            )
            boundary_receipts.append(receipt)
            preparation_kind = "public_prune_reuse"
        elif role == "r3":
            boundary_receipts.append(
                _execute_rejected_prune_requests(
                    source=source_owner,
                    fixture=target_fixture,
                    target_slots=target_slots,
                )
            )
            preparation_kind = "exact_rejected_prune_requests"
    elif family == "f11" and profile == "b2-12x32-v1":
        if role == "r2":
            if action != "reject_unauthorized_request":
                raise PhaseFV2ExecutionSourceError(
                    "frozen unauthorized migration action changed"
                )
            boundary_receipts.append(
                _execute_rejected_migration(
                    source=source,
                    owner=source_owner,
                    expected_rejection="unauthorized_request",
                )
            )
            preparation_kind = "exact_rejected_migration"
        elif role == "r3":
            if action != "reject_wrong_safe_boundary":
                raise PhaseFV2ExecutionSourceError(
                    "frozen wrong-boundary migration action changed"
                )
            boundary_receipts.append(
                _execute_rejected_migration(
                    source=source,
                    owner=source_owner,
                    expected_rejection="wrong_safe_boundary",
                )
            )
            preparation_kind = "exact_rejected_migration"
        elif role not in {"r0", "r1", "r4"}:
            raise PhaseFV2ExecutionSourceError("unsupported frozen F11 b12 role")
    elif accepted_migration:
        context = _migration_context(
            migration_context,
            matrix=checked_matrix,
            source_case=source_case,
            source_owner=source_owner,
        )
        try:
            migrated = execute_bound_capacity_migration(
                source_metadata=source.owner_metadata,
                source_tensors=source.generation_tensors,
                authorization=context["authorization"],
                phase_e_receipt_binding=context["phase_e_receipt_binding"],
                rng_state_before=deepcopy(dict(context["rng_state"])),
                rng_state_after=deepcopy(dict(context["rng_state"])),
                rng_tensors_before=_clone_tensors(context["rng_tensors"]),
                rng_tensors_after=_clone_tensors(context["rng_tensors"]),
            )
            target_owner = modular_storage_from_projection(
                migrated.migration.metadata, migrated.migration.tensors
            )
        except Exception as exc:
            raise PhaseFV2ExecutionSourceError(
                "frozen accepted migration failed"
            ) from exc
        if target_owner is None:
            raise PhaseFV2ExecutionSourceError("migration returned no packed owner")
        validate_modular_storage(target_owner)
        boundary_receipts.append(migrated.receipt)
        preparation_kind = "public_capacity_migration"
    elif family == "f12" and profile == "b2-16x48-v1" and role == "r1":
        if action != "prune_reuse":
            raise PhaseFV2ExecutionSourceError("frozen F12 reuse action changed")
        target_owner, receipt = _execute_prune_reuse(
            source=source_owner,
            fixture=target_fixture,
            target_slots=(6,),
        )
        boundary_receipts.append(receipt)
        preparation_kind = "public_prune_reuse"

    if migration_context is not None and not accepted_migration:
        raise PhaseFV2ExecutionSourceError(
            "migration context supplied to a non-migration case"
        )

    if target_owner.layout.profile_id != target_case["profile_id"]:
        raise PhaseFV2ExecutionSourceError("prepared owner profile is not target exact")
    provenance = _provenance(
        preparation_kind=preparation_kind,
        target_case=target_case,
        target_fixture=target_fixture,
        source_case=source_case,
        source_fixture=source_fixture,
        source_state=source,
        source_owner=source_owner,
        target_owner=target_owner,
        boundary_receipts=boundary_receipts,
    )
    if (
        target_owner.packed_sha256 != source_owner.packed_sha256
        or target_case["case_id"] != source_case["case_id"]
    ):
        execution_source = rebuild_phase_f_source_from_modular_owner_v2(
            source_state=source,
            modular_owner=target_owner,
            target_case=target_case,
            target_fixture_case=target_fixture,
            provenance=provenance,
        )
    else:
        execution_source = source

    checkpoint_kind = str(target_case["checkpoint_schedule"]["kind"])
    prerequisite = (
        deepcopy(boundary_receipts[-1])
        if checkpoint_kind in _POST_BOUNDARY_CHECKPOINT_KINDS and boundary_receipts
        else None
    )
    if checkpoint_kind in _POST_BOUNDARY_CHECKPOINT_KINDS and prerequisite is None:
        raise PhaseFV2ExecutionSourceError(
            "post-boundary checkpoint lacks its exact prerequisite"
        )
    receipt = _preparation_receipt(
        {
            "status": "prepared_cpu_only_execution_source",
            "case_id": target_case["case_id"],
            "case_sha256": target_case["case_sha256"],
            "fixture_sha256": target_fixture["fixture_sha256"],
            "matrix_sha256": checked_matrix["matrix_sha256"],
            "profile_id": target_case["profile_id"],
            "role_action": action,
            "preparation_kind": preparation_kind,
            "source_case_id": source_case["case_id"],
            "source_state_sha256": source.source_sha256,
            "source_packed_sha256": source_owner.packed_sha256,
            "execution_source_sha256": execution_source.source_sha256,
            "execution_packed_sha256": target_owner.packed_sha256,
            "boundary_action_sha256s": [
                value["action_sha256"] for value in boundary_receipts
            ],
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_prerequisite_action_sha256": (
                None if prerequisite is None else prerequisite["action_sha256"]
            ),
            "formal_and_generation_owner_shared": True,
            "generation_entry_validated": True,
            "tensors_detached": True,
            "rng_draw_count": 0,
            "simulation_executed": False,
        }
    )
    return PreparedPhaseFV2ExecutionSource(
        schema_version=SCHEMA_VERSION,
        source_state=execution_source,
        modular_owner=target_owner,
        boundary_receipts=tuple(deepcopy(boundary_receipts)),
        checkpoint_prerequisite_receipt=prerequisite,
        preparation_receipt=receipt,
    )


def validate_prepared_phase_f_execution_source_v2(
    source_state: PhaseFV2SourceState,
    *,
    matrix: Any,
    fixture_registry: Any,
    case_id: str,
    preparation_receipt: Mapping[str, Any],
) -> PhaseFV2SourceState:
    """Validate a boundary-prepared source without rematerializing the raw fixture."""

    checked_matrix, registry = _validated_inputs(matrix, fixture_registry)
    cases = _case_map(checked_matrix)
    fixtures = _fixture_map(registry)
    if case_id not in cases or case_id not in fixtures:
        raise PhaseFV2ExecutionSourceError("prepared source case is not frozen")
    if not isinstance(source_state, PhaseFV2SourceState):
        raise PhaseFV2ExecutionSourceError("prepared source has the wrong type")
    if not isinstance(preparation_receipt, Mapping):
        raise PhaseFV2ExecutionSourceError("preparation receipt must be a mapping")
    receipt = deepcopy(dict(preparation_receipt))
    supplied = receipt.pop("receipt_sha256", None)
    if receipt.get(
        "schema_version"
    ) != RECEIPT_SCHEMA_VERSION or supplied != canonical_sha256(receipt):
        raise PhaseFV2ExecutionSourceError("preparation receipt digest mismatch")
    case = cases[case_id]
    fixture = fixtures[case_id]
    expected = {
        "case_id": case_id,
        "case_sha256": case["case_sha256"],
        "fixture_sha256": fixture["fixture_sha256"],
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "profile_id": case["profile_id"],
        "role_action": case["role_contract"]["action"],
        "execution_source_sha256": source_state.source_sha256,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise PhaseFV2ExecutionSourceError(
                f"preparation receipt {field} binding mismatch"
            )
    if (
        receipt.get("status") != "prepared_cpu_only_execution_source"
        or receipt.get("formal_and_generation_owner_shared") is not True
        or receipt.get("generation_entry_validated") is not True
        or receipt.get("rng_draw_count") != 0
        or receipt.get("simulation_executed") is not False
    ):
        raise PhaseFV2ExecutionSourceError(
            "preparation receipt overclaims or is incomplete"
        )
    if (
        source_state.case_id != case_id
        or source_state.case_sha256 != case["case_sha256"]
        or source_state.fixture_sha256 != fixture["fixture_sha256"]
        or source_state.matrix_sha256 != checked_matrix["matrix_sha256"]
        or source_state.profile_id != case["profile_id"]
        or source_state.role_action != case["role_contract"]["action"]
    ):
        raise PhaseFV2ExecutionSourceError("prepared source case binding mismatch")
    owner = _owner_from_source(source_state)
    if owner.packed_sha256 != receipt.get("execution_packed_sha256"):
        raise PhaseFV2ExecutionSourceError("prepared source owner digest mismatch")
    _validate_owner_membership(source_state, owner)
    try:
        overlay = validate_modular_cpu_boundary_projection(
            source_state.owner_metadata,
            source_state.generation_tensors,
            agent_capacity=AGENT_CAPACITY,
        )
        _source_v2._validate_generation_tensors(  # noqa: SLF001
            source_state.generation_tensors,
            profile_id=source_state.profile_id,
        )
    except Exception as exc:
        raise PhaseFV2ExecutionSourceError(
            "prepared source failed the public generation entry"
        ) from exc
    if overlay is None or overlay.packed.packed_sha256 != owner.packed_sha256:
        raise PhaseFV2ExecutionSourceError("prepared source overlay changed authority")
    header = {
        "schema_version": source_state.schema_version,
        "case_id": source_state.case_id,
        "case_sha256": source_state.case_sha256,
        "fixture_sha256": source_state.fixture_sha256,
        "matrix_sha256": source_state.matrix_sha256,
        "profile_id": source_state.profile_id,
        "role_action": source_state.role_action,
        "source_seed": source_state.source_seed,
    }
    observed_sha = _source_v2._source_digest(  # noqa: SLF001
        header=header,
        owner_metadata=source_state.owner_metadata,
        config=source_state.generation_config,
        generation_tensors=source_state.generation_tensors,
        formal_inputs=source_state.formal_oracle_inputs,
    )
    if observed_sha != source_state.source_sha256:
        raise PhaseFV2ExecutionSourceError("prepared source digest mismatch")
    for slot, formal in enumerate(source_state.formal_oracle_inputs):
        for name, plane in formal.planes.items():
            projected = source_state.generation_tensors[
                f"{MODULAR_STORAGE_PROJECTION_PREFIX}{name}"
            ][slot : slot + 1]
            if not torch.equal(plane, projected):
                raise PhaseFV2ExecutionSourceError(
                    "prepared formal and generation planes diverged"
                )
    return source_state


__all__ = [
    "MIGRATION_CONTEXT_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "PhaseFV2ExecutionSourceError",
    "PreparedPhaseFV2ExecutionSource",
    "build_phase_f_migration_context_v2",
    "prepare_phase_f_execution_source_v2",
    "rebuild_phase_f_source_from_modular_owner_v2",
    "validate_prepared_phase_f_execution_source_v2",
]
