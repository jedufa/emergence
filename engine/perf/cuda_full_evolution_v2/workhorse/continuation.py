"""Packing-neutral ACS2 continuation contracts for Workhorse V2.

The science carrier identifies which world states continue and the absolute
checkpoint schedule. CUDA packing remains a separately sealed physical plan:
target ``B``, preferred/repair ``K`` values, physical order, source files,
backend, artifact, device, and Attempt never enter the science identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any, Final

from .contract import (
    FullEvolutionCudaWorkhorseContractV2,
    FullEvolutionWorkhorseContractError,
    WORKHORSE_TIME_AUTHORITY_V2,
    canonical_sha256,
)
from .dispatch_plan import (
    WorkhorseDispatchPlanError,
    validate_workhorse_absolute_dispatch_plan_v2,
)
from .indexed_checkpoint import (
    WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1,
    WorkhorseIndexedCheckpointError,
    inspect_workhorse_selected_world_checkpoint_v1,
    validate_workhorse_repack_binding_v2,
    validate_workhorse_repack_binding_seal_v2,
)
from .scientific_control import (
    WORKHORSE_SCIENTIFIC_CONTROL_APPLICATION_SCHEMA_V1,
    WORKHORSE_SCIENTIFIC_CONTROL_CONTRACT_SCHEMA_V1,
    WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_ABI_V1,
    WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1,
    WorkhorseScientificControlError,
    validate_workhorse_scientific_control_contract_v1 as _validate_scientific_control,
    workhorse_scientific_control_contract_v1 as _scientific_control_contract,
)


WORKHORSE_INDEXED_CONTINUATION_CONTRACT_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_indexed_continuation_contract_v2"
)
WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2: Final[str] = (
    "INDEXED_ACS_CONTINUATION"
)
class WorkhorseIndexedContinuationError(RuntimeError):
    """Raised when logical continuation and physical packing are conflated."""


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WorkhorseIndexedContinuationError(f"{label} is not a SHA-256")
    return value


def _checkpoint_ticks(
    values: Sequence[int], *, start_next_tick: int, target_next_tick: int
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise WorkhorseIndexedContinuationError(
            "continuation checkpoint ticks must be an ordered sequence"
        )
    ticks = tuple(values)
    if (
        not ticks
        or any(type(tick) is not int for tick in ticks)
        or ticks != tuple(sorted(set(ticks)))
        or ticks[-1] != target_next_tick
        or any(not start_next_tick < tick <= target_next_tick for tick in ticks)
    ):
        raise WorkhorseIndexedContinuationError(
            "continuation checkpoint ticks must be unique absolute ticks in "
            "the run span and include the terminal tick"
        )
    return ticks


def workhorse_scientific_control_contract_v1(
    *,
    effective_next_absolute_tick: int,
    resource_multiplier: float,
    mutation_rate: float,
) -> dict[str, Any]:
    try:
        return _scientific_control_contract(
            effective_next_absolute_tick=effective_next_absolute_tick,
            resource_multiplier=resource_multiplier,
            mutation_rate=mutation_rate,
        )
    except WorkhorseScientificControlError as exc:
        raise WorkhorseIndexedContinuationError(str(exc)) from exc


def validate_workhorse_scientific_control_contract_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _validate_scientific_control(value)
    except WorkhorseScientificControlError as exc:
        raise WorkhorseIndexedContinuationError(str(exc)) from exc


def workhorse_indexed_continuation_contract_v2(
    *,
    science_continuation_contract_sha256: str,
    repack_binding: Mapping[str, Any],
    dispatch_plan: Mapping[str, Any],
    checkpoint_ticks: Sequence[int],
    scientific_control_contract: Mapping[str, Any] | None = None,
    require_current_source_match: bool = True,
    reopen_source_payloads: bool = True,
) -> dict[str, Any]:
    """Bind one logical continuation to one replaceable physical CUDA plan."""

    science_sha256 = _sha256(
        science_continuation_contract_sha256,
        "science continuation contract",
    )
    try:
        repack = (
            validate_workhorse_repack_binding_v2(
                repack_binding,
                require_current_source_match=require_current_source_match,
            )
            if reopen_source_payloads
            else validate_workhorse_repack_binding_seal_v2(repack_binding)
        )
        dispatch = validate_workhorse_absolute_dispatch_plan_v2(dispatch_plan)
        target = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            repack["target_contract"]
        )
    except (
        WorkhorseIndexedCheckpointError,
        WorkhorseDispatchPlanError,
        FullEvolutionWorkhorseContractError,
        TypeError,
        ValueError,
    ) as exc:
        raise WorkhorseIndexedContinuationError(
            f"indexed continuation physical binding is invalid: {exc}"
        ) from exc
    target_next_tick = target.start_next_tick + target.total_ticks
    ticks = _checkpoint_ticks(
        checkpoint_ticks,
        start_next_tick=target.start_next_tick,
        target_next_tick=target_next_tick,
    )
    scientific_control = (
        None
        if scientific_control_contract is None
        else validate_workhorse_scientific_control_contract_v1(
            scientific_control_contract
        )
    )
    if (
        scientific_control is not None
        and scientific_control["effective_next_absolute_tick"]
        != target.start_next_tick
    ):
        raise WorkhorseIndexedContinuationError(
            "changed scientific controls must become effective at the branch tick"
        )
    if (
        dispatch["start_next_tick"] != target.start_next_tick
        or dispatch["target_next_tick"] != target_next_tick
        or dispatch["total_ticks"] != target.total_ticks
        or dispatch["preferred_temporal_k"] != target.temporal_k
        or dispatch["boundary_next_ticks"] != list(ticks)
        or repack["checkpoint_at_tick"] != target.start_next_tick
        or repack["selected_world_count"] != target.B
    ):
        raise WorkhorseIndexedContinuationError(
            "indexed continuation target, repack, dispatch, and checkpoint "
            "frontiers disagree"
        )
    selected = repack["selected_world_state_set_identity"]
    source_state_set_sha256 = _sha256(
        selected.get("world_state_set_sha256"),
        "selected source world-state set",
    )
    selected_authorities: list[dict[str, Any]] = []
    seen_selected_manifests: set[str] = set()
    for locator in repack["source_locators"]:
        manifest_path = str(locator["source_index_manifest_path"])
        if manifest_path in seen_selected_manifests:
            continue
        try:
            raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise WorkhorseIndexedContinuationError(
                f"continuation source manifest cannot be read: {exc}"
            ) from exc
        if raw.get("schema_version") == WORKHORSE_SELECTED_WORLD_CHECKPOINT_SCHEMA_V1:
            selected_manifest, _state_path = (
                inspect_workhorse_selected_world_checkpoint_v1(manifest_path)
            )
            selected_authority = dict(selected_manifest["authority"])
            selected_authority["active_control_state_sha256"] = selected_manifest[
                "active_scientific_control_state"
            ]["active_control_state_sha256"]
            selected_authorities.append(selected_authority)
            seen_selected_manifests.add(manifest_path)
    if selected_authorities and (
        len(selected_authorities) != 1 or repack["selected_world_count"] != 1
    ):
        raise WorkhorseIndexedContinuationError(
            "V1 selected-world continuation requires exactly one retained parent"
        )
    if selected_authorities and scientific_control is None:
        raise WorkhorseIndexedContinuationError(
            "selected-world continuation requires explicit resource and mutation controls"
        )
    logical_body = {
        "schema_version": "eiel_workhorse_science_continuation_identity_v2",
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "science_continuation_contract_sha256": science_sha256,
        "source_world_state_set_sha256": source_state_set_sha256,
        "shape_id": selected["shape_id"],
        "semantic_transition_sha256": selected["semantic_transition_sha256"],
        "start_next_tick": target.start_next_tick,
        "target_next_tick": target_next_tick,
        "checkpoint_ticks": list(ticks),
        "world_count": selected["world_count"],
        "continuation_scope": repack["continuation_scope"],
        "packing_excluded_fields": [
            "physical_B",
            "physical_K",
            "physical_order",
            "source_manifest",
            "source_physical_checkpoint",
            "backend",
            "artifact",
            "device",
            "attempt",
        ],
    }
    if scientific_control is not None:
        logical_body["scientific_control_contract_sha256"] = scientific_control[
            "control_contract_sha256"
        ]
    if selected_authorities:
        selected_authority = selected_authorities[0]
        logical_body["source_checkpoint_occurrence_sha256"] = selected_authority[
            "checkpoint_occurrence_sha256"
        ]
        logical_body["source_parent_occurrence_sha256"] = selected_authority[
            "parent_occurrence_sha256"
        ]
        logical_body["source_semantic_world_sha256"] = selected_authority[
            "semantic_world_sha256"
        ]
        logical_body["source_control_history_sha256"] = selected_authority[
            "control_history_sha256"
        ]
        logical_body["source_rng_state_sha256"] = selected_authority[
            "rng_state_sha256"
        ]
        logical_body["source_event_prefix_sha256"] = selected_authority[
            "event_prefix_sha256"
        ]
        logical_body["source_active_control_state_sha256"] = selected_authority[
            "active_control_state_sha256"
        ]
    logical_identity = {
        **logical_body,
        "science_continuation_identity_sha256": canonical_sha256(logical_body),
    }
    physical_body = {
        "schema_version": "eiel_workhorse_physical_continuation_plan_v2",
        "target_contract": target.to_dict(),
        "repack_binding_sha256": repack["repack_binding_sha256"],
        "dispatch_plan_sha256": dispatch["dispatch_plan_sha256"],
        "source_world_state_set_sha256": source_state_set_sha256,
        "identity_scope": "PHYSICAL_EXECUTION_ONLY",
    }
    physical_plan = {
        **physical_body,
        "physical_continuation_plan_sha256": canonical_sha256(physical_body),
    }
    body = {
        "schema_version": WORKHORSE_INDEXED_CONTINUATION_CONTRACT_SCHEMA_V2,
        "operation": WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "science_continuation_identity": logical_identity,
        "physical_continuation_plan": physical_plan,
        "repack_binding": repack,
        "dispatch_plan": dispatch,
        "checkpoint_ticks": list(ticks),
    }
    if scientific_control is not None:
        body["scientific_control_contract"] = scientific_control
    return {**body, "continuation_contract_sha256": canonical_sha256(body)}


def validate_workhorse_indexed_continuation_contract_v2(
    value: Mapping[str, Any],
    *,
    require_current_source_match: bool = True,
    reopen_source_payloads: bool = True,
) -> dict[str, Any]:
    required_fields = {
        "schema_version",
        "operation",
        "time_authority",
        "science_continuation_identity",
        "physical_continuation_plan",
        "repack_binding",
        "dispatch_plan",
        "checkpoint_ticks",
        "continuation_contract_sha256",
    }
    if type(value) is not dict or frozenset(value) not in {
        frozenset(required_fields),
        frozenset(required_fields | {"scientific_control_contract"}),
    }:
        raise WorkhorseIndexedContinuationError(
            "indexed continuation contract fields are inexact"
        )
    logical = value.get("science_continuation_identity")
    if type(logical) is not dict:
        raise WorkhorseIndexedContinuationError(
            "indexed continuation science identity is absent"
        )
    expected = workhorse_indexed_continuation_contract_v2(
        science_continuation_contract_sha256=logical.get(
            "science_continuation_contract_sha256"
        ),
        repack_binding=value["repack_binding"],
        dispatch_plan=value["dispatch_plan"],
        checkpoint_ticks=value["checkpoint_ticks"],
        scientific_control_contract=value.get("scientific_control_contract"),
        require_current_source_match=require_current_source_match,
        reopen_source_payloads=reopen_source_payloads,
    )
    if dict(value) != expected:
        raise WorkhorseIndexedContinuationError(
            "indexed continuation contract does not recompute"
        )
    return expected


WORKHORSE_INDEXED_CONTINUATION_RESULT_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_indexed_continuation_result_v2"
)
WORKHORSE_TERMINALIZED_INDEXED_CONTINUATION_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_terminalized_indexed_continuation_receipt_v2"
)
WORKHORSE_INDEXED_CONTINUATION_AGGREGATE_WTPS_METRIC_V2: Final[str] = (
    "workhorse.indexed_continuation_aggregate_world_ticks_per_second"
)
WORKHORSE_INDEXED_CHECKPOINT_PUBLICATION_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_indexed_checkpoint_publication_seal_v2"
)


def validate_workhorse_indexed_checkpoint_publication_seals_v2(
    values: Sequence[Mapping[str, Any]],
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    checkpoint_ticks: Sequence[int],
) -> list[dict[str, Any]]:
    """Validate compact output identities after physical payload release."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise WorkhorseIndexedContinuationError(
            "indexed checkpoint publications must be an ordered sequence"
        )
    ticks = _checkpoint_ticks(
        checkpoint_ticks,
        start_next_tick=contract.start_next_tick,
        target_next_tick=contract.start_next_tick + contract.total_ticks,
    )
    if len(values) != len(ticks):
        raise WorkhorseIndexedContinuationError(
            "indexed checkpoint publication coverage is incomplete"
        )
    fields = {
        "schema_version",
        "checkpoint_at_tick",
        "canonical_state_bytes",
        "canonical_state_sha256",
        "checkpoint_bundle_body_sha256",
        "indexed_manifest_file_sha256",
        "indexed_manifest_body_sha256",
        "world_state_set_identity",
        "checkpoint_occurrence_projection_content_sha256",
        "absolute_tick_bundle_digest",
        "occurrence_digest",
        "publication_seal_sha256",
    }
    identity_fields = {
        "schema_version",
        "time_authority",
        "shape_id",
        "state_layout_sha256",
        "semantic_transition_id",
        "semantic_transition_sha256",
        "checkpoint_at_tick",
        "rng_epoch",
        "world_count",
        "worlds_by_uid",
        "packing_excluded_fields",
        "world_state_set_sha256",
    }
    world_fields = {"world_uid", "rng_key", "state_bytes", "state_sha256"}
    normalized: list[dict[str, Any]] = []
    for value, tick in zip(values, ticks, strict=True):
        if type(value) is not dict or set(value) != fields:
            raise WorkhorseIndexedContinuationError(
                "indexed checkpoint publication fields are inexact"
            )
        identity = value.get("world_state_set_identity")
        worlds = identity.get("worlds_by_uid") if isinstance(identity, Mapping) else None
        identity_body = (
            {key: item for key, item in identity.items() if key != "world_state_set_sha256"}
            if isinstance(identity, Mapping)
            else {}
        )
        publication_body = {
            key: item for key, item in value.items() if key != "publication_seal_sha256"
        }
        if (
            value["schema_version"]
            != WORKHORSE_INDEXED_CHECKPOINT_PUBLICATION_SCHEMA_V2
            or value["checkpoint_at_tick"] != tick
            or type(value["canonical_state_bytes"]) is not int
            or value["canonical_state_bytes"] <= 0
            or any(
                _sha256(value[field], field) != value[field]
                for field in (
                    "canonical_state_sha256",
                    "checkpoint_bundle_body_sha256",
                    "indexed_manifest_file_sha256",
                    "indexed_manifest_body_sha256",
                    "checkpoint_occurrence_projection_content_sha256",
                    "absolute_tick_bundle_digest",
                    "occurrence_digest",
                )
            )
            or not isinstance(identity, Mapping)
            or set(identity) != identity_fields
            or type(worlds) is not list
            or len(worlds) != contract.B
            or any(type(row) is not dict or set(row) != world_fields for row in worlds)
            or [row["world_uid"] for row in worlds]
            != sorted(row["world_uid"] for row in worlds)
            or len({row["world_uid"] for row in worlds}) != contract.B
            or any(
                type(row["world_uid"]) is not int
                or row["world_uid"] <= 0
                or type(row["rng_key"]) is not int
                or row["rng_key"] < 0
                or type(row["state_bytes"]) is not int
                or row["state_bytes"] <= 0
                or not _sha256(row["state_sha256"], "world state digest")
                for row in worlds
            )
            or identity["time_authority"] != WORKHORSE_TIME_AUTHORITY_V2
            or identity["shape_id"] != contract.shape_id
            or identity["semantic_transition_id"] != contract.semantic_transition_id
            or identity["semantic_transition_sha256"]
            != contract.semantic_transition_sha256
            or identity["checkpoint_at_tick"] != tick
            or identity["rng_epoch"] != 4
            or identity["world_count"] != contract.B
            or identity["world_state_set_sha256"] != canonical_sha256(identity_body)
            or value["publication_seal_sha256"] != canonical_sha256(publication_body)
        ):
            raise WorkhorseIndexedContinuationError(
                "indexed checkpoint publication identity changed"
            )
        normalized.append(dict(value))
    if len(normalized) != len(ticks):
        raise WorkhorseIndexedContinuationError(
            "indexed checkpoint publication coverage is incomplete"
        )
    return normalized


def run_workhorse_indexed_continuation_v2(
    artifact_path: str | Path,
    *,
    envelope_receipt_path: str | Path,
    canonical_parent_authority_binding: Mapping[str, Any],
    authoritative_micro_receipt_path: str | Path,
    semantic_gate_receipt_path: str | Path,
    continuation_contract: Mapping[str, Any],
    event_sink_root: str | Path,
    execution_result_path: str | Path,
    checkpoint_boundary_handler: Callable[..., object] | None = None,
) -> dict[str, Any]:
    """Restore a fresh physical owner and continue to exact absolute boundaries.

    The callback is provider-owned. It receives the resident state only after
    the event sink has acknowledged the boundary and is responsible for ACS2
    capture/settlement policy. This keeps lifecycle authority outside the CUDA
    engine while allowing the provider to drain the two-slot FIFO when needed.
    """

    continuation = validate_workhorse_indexed_continuation_contract_v2(
        continuation_contract
    )
    target = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        continuation["physical_continuation_plan"]["target_contract"]
    )
    dispatch = continuation["dispatch_plan"]
    logical = continuation["science_continuation_identity"]

    from .canonical_rows import (
        _envelope_row,
        _owner_components,
        _runtime_observation,
        _write_sealed_json,
        validate_workhorse_canonical_parent_authority_binding_v2,
        validate_workhorse_device_envelope_calibration_receipt_v2,
        workhorse_artifact_binding_v2,
    )
    from .authority import (
        validate_authoritative_micro_denominator_v2,
        validate_workhorse_semantic_gate_receipt_v2,
    )
    from .layout import admit_workhorse_vram_v2, estimate_workhorse_resources_v2
    from .probe import (
        _acknowledge_events,
        _advance,
        _allocate_owner,
        _load,
        _prepare_dispatch_v2,
        _prepare_graph_if_needed,
        _release_graph,
        _restore_owner_from_repack_binding_v2,
        _torch,
    )
    from ..benchmark import cuda_device_identity_v2

    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        canonical_parent_authority_binding,
        expected_operation=WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2,
    )
    if (
        parent["contract"] != target.to_dict()
        or parent["science_continuation_contract_sha256"]
        != logical["science_continuation_contract_sha256"]
        or parent["source_checkpoint_identity_sha256"]
        != logical["source_world_state_set_sha256"]
        or parent["compiled_temporal_k"] != dispatch["admitted_temporal_k"]
    ):
        raise WorkhorseIndexedContinuationError(
            "canonical parent differs from the indexed continuation carrier"
        )
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        envelope_receipt_path
    )
    artifact = workhorse_artifact_binding_v2(artifact_path)
    if artifact != envelope["artifact"] or (
        artifact["sha256"] != parent["artifact_sha256"]
        or artifact["sealed_artifact_identity_sha256"]
        != parent["sealed_artifact_identity_sha256"]
    ):
        raise WorkhorseIndexedContinuationError(
            "continuation artifact differs from envelope or parent"
        )
    semantic = validate_workhorse_semantic_gate_receipt_v2(
        semantic_gate_receipt_path,
        expected_shape_id=target.shape_id,
        expected_candidate_artifact_sha256=artifact["sha256"],
        expected_candidate_backend_id=target.backend_id,
    )

    torch = _torch()
    loaded = _load(artifact_path, torch)
    runtime = dict(loaded.module.runtime_inspect())
    device = cuda_device_identity_v2(torch, torch.cuda.current_device())
    if device != envelope["device_identity"] or device["uuid"] != parent["device_uuid"]:
        raise WorkhorseIndexedContinuationError(
            "continuation runtime device differs from envelope or parent"
        )
    micro = validate_authoritative_micro_denominator_v2(
        authoritative_micro_receipt_path,
        expected_device_identity=device,
    )
    envelope_rows = {
        (row["B"], row["temporal_k"]): row for row in envelope["expected_rows"]
    }
    for temporal_k in dispatch["admitted_temporal_k"]:
        row = envelope_rows.get((target.B, temporal_k))
        if row is None or row["planned_admission"].get("admitted") is not True:
            raise WorkhorseIndexedContinuationError(
                f"continuation physical K{temporal_k} is not envelope-admitted"
            )
    components = _owner_components(
        envelope["owner_components_by_k"][str(target.temporal_k)],
        backend_id=target.backend_id,
        temporal_k=target.temporal_k,
    )
    ledger = estimate_workhorse_resources_v2(
        target, owner_components=components
    ).to_dict()
    free_before, total_observed = torch.cuda.mem_get_info(torch.cuda.current_device())
    runtime_observation = _runtime_observation(
        free_bytes=int(free_before),
        total_device_bytes=int(total_observed),
        device_uuid=str(device["uuid"]),
    )
    admission = admit_workhorse_vram_v2(
        target,
        free_bytes=int(free_before),
        total_device_bytes=int(total_observed),
        owner_components=components,
        reserve_bytes=envelope["capacity_basis"]["reserve_bytes_requested"],
    ).to_dict()
    planned = _envelope_row(
        envelope, B=target.B, temporal_k=target.temporal_k
    )["planned_admission"]
    common = {
        "schema_version": WORKHORSE_INDEXED_CONTINUATION_RESULT_SCHEMA_V2,
        "operation": WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2,
        "claim_class": "TECHNICAL_ONLY_NO_SCIENCE_OR_QUALIFICATION_AUTHORITY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "continuation_contract": continuation,
        "canonical_parent_authority_binding": dict(parent),
        "artifact": artifact,
        "envelope_binding": {
            "receipt_path": envelope["receipt_path"],
            "receipt_file_sha256": envelope["receipt_file_sha256"],
            "receipt_content_sha256": envelope["receipt_content_sha256"],
        },
        "semantic_binding": semantic,
        "micro_denominator_binding": micro,
        "resource_ledger": ledger,
        "planned_admission": planned,
        "runtime_admission": admission,
        "runtime_observation": runtime_observation,
        "runtime_inspection": runtime,
    }
    controlled = continuation.get("scientific_control_contract")
    if controlled is not None:
        common["control_application"] = None
    if admission["admitted"] is not True:
        body = {
            **common,
            "outcome": "RUNTIME_HEADROOM_REJECTED",
            "absolute_frontiers": {
                "start_next_tick": target.start_next_tick,
                "target_next_tick": target.start_next_tick + target.total_ticks,
                "committed_next_tick": target.start_next_tick,
            },
            "restored_world_uids": [],
            "dispatch_measurements": [],
            "checkpoint_capture_ticks": [],
            "graph_inspections": [],
            "cuda_event_elapsed_ms_sum": None,
            "aggregate_world_ticks_per_second": None,
        }
        receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
        return validate_workhorse_indexed_continuation_result_v2(
            _write_sealed_json(receipt, execution_result_path)
        )

    owner = None
    measurements: list[dict[str, Any]] = []
    checkpoint_capture_ticks: list[int] = []
    graph_inspections: list[dict[str, Any]] = []
    control_application: dict[str, Any] | None = None
    try:
        owner = _allocate_owner(torch, loaded, target.B, target.temporal_k)
        owner.attempt_authority_binding = parent
        restored = _restore_owner_from_repack_binding_v2(
            torch,
            loaded,
            owner,
            repack_binding=continuation["repack_binding"],
        )
        if restored != continuation["repack_binding"]:
            raise WorkhorseIndexedContinuationError(
                "fresh owner restored another repack binding"
            )
        if controlled is not None:
            native = controlled["native_mapping"]
            runtime_abi = native["runtime_abi"]
            requested_row = list(native["ordered_values"])
            if runtime_abi != {
                "abi_id": WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_ABI_V1,
                "required_exports": list(
                    WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1
                ),
                "application_rule": (
                    "FRESH_RESTORED_OWNER_AT_BOUND_BRANCH_TICK_ONCE"
                ),
            } or any(
                not hasattr(loaded.module, export)
                for export in WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1
            ):
                raise WorkhorseIndexedContinuationError(
                    "sealed Workhorse artifact lacks the scientific-control ABI"
                )
            if loaded.module.science_control_abi_version_v1() != 1:
                raise WorkhorseIndexedContinuationError(
                    "sealed Workhorse artifact reports another scientific-control ABI"
                )
            if (
                owner.committed_next_tick != target.start_next_tick
                or owner.pending is not None
                or owner.poisoned_status
            ):
                raise WorkhorseIndexedContinuationError(
                    "scientific controls require a fresh exact owner at the branch tick"
                )
            _release_graph(loaded)
            requested_rows = [requested_row for _ in range(target.B)]
            control_tensor = torch.tensor(
                requested_rows,
                dtype=torch.int64,
                device=owner.scratch.device,
            )
            loaded.module.apply_science_controls_v1(
                owner.state,
                owner.scratch,
                control_tensor,
                target.start_next_tick,
            )
            raw_readback_rows = (
                loaded.module.science_controls_readback_v1(owner.scratch)
                .detach()
                .cpu()
                .tolist()
            )
            readback_rows = [row[:2] for row in raw_readback_rows]
            application_flags = [row[2] for row in raw_readback_rows]
            if readback_rows != requested_rows:
                raise WorkhorseIndexedContinuationError(
                    "Workhorse scientific-control readback differs before dispatch"
                )
            if application_flags != [1] * target.B:
                raise WorkhorseIndexedContinuationError(
                    "Workhorse scientific-control one-shot application was not observed"
                )
            application_body = {
                "schema_version": WORKHORSE_SCIENTIFIC_CONTROL_APPLICATION_SCHEMA_V1,
                "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
                "control_contract_sha256": controlled["control_contract_sha256"],
                "applied_next_absolute_tick": owner.committed_next_tick,
                "first_permitted_divergence_tick": target.start_next_tick,
                "world_uids": list(owner.world_uids),
                "ordered_native_fields": list(native["ordered_fields"]),
                "requested_native_rows": requested_rows,
                "readback_native_rows": readback_rows,
                "native_application_flags": application_flags,
                "runtime_entrypoints": list(
                    WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1
                ),
            }
            control_application = {
                **application_body,
                "control_application_sha256": canonical_sha256(application_body),
            }
        checkpoint_ticks_set = set(continuation["checkpoint_ticks"])
        sample_index = 0
        for segment_index, segment in enumerate(dispatch["segments"]):
            temporal_k = segment["temporal_k"]
            _release_graph(loaded)
            inspection = _prepare_graph_if_needed(loaded, owner, temporal_k, 0)
            if inspection is not None:
                graph_inspections.append(dict(inspection))
            for dispatch_index in range(segment["dispatch_count"]):
                start_tick = owner.committed_next_tick
                _prepare_dispatch_v2(loaded, owner, temporal_k, 0)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _advance(loaded, owner, temporal_k, 0)
                end.record()
                end.synchronize()
                elapsed_ms = float(start.elapsed_time(end))
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise WorkhorseIndexedContinuationError(
                        "continuation CUDA event duration is not positive"
                    )
                ack = _acknowledge_events(
                    owner,
                    torch,
                    sink_root=Path(event_sink_root)
                    / f"segment-{segment_index}-dispatch-{dispatch_index}",
                    sink_id="workhorse-indexed-continuation-v2",
                    retain_sink_payload=False,
                )
                measurement_body = {
                    "sample_index": sample_index,
                    "segment_index": segment_index,
                    "dispatch_index_in_segment": dispatch_index,
                    "start_next_tick": start_tick,
                    "end_next_tick": owner.committed_next_tick,
                    "temporal_k": temporal_k,
                    "cuda_event_elapsed_ms": elapsed_ms,
                    "committed_world_ticks": target.B * temporal_k,
                    "world_ticks_per_second": target.B * temporal_k * 1000.0 / elapsed_ms,
                    "event_sink_receipt": dict(ack),
                }
                measurements.append(
                    {
                        **measurement_body,
                        "measurement_sha256": canonical_sha256(measurement_body),
                    }
                )
                sample_index += 1
                if owner.committed_next_tick in checkpoint_ticks_set:
                    if checkpoint_boundary_handler is not None:
                        checkpoint_boundary_handler(
                            state=owner.state,
                            contract=target,
                            checkpoint_at_tick=owner.committed_next_tick,
                            event_sink_receipt_path=ack["receipt_path"],
                        )
                    checkpoint_capture_ticks.append(owner.committed_next_tick)
        if owner.committed_next_tick != target.start_next_tick + target.total_ticks:
            raise WorkhorseIndexedContinuationError(
                "continuation did not reach its exact terminal frontier"
            )
        restored_world_uids = list(owner.world_uids)
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elapsed_sum = math.fsum(
        float(row["cuda_event_elapsed_ms"]) for row in measurements
    )
    body = {
        **common,
        "outcome": "CONTINUED",
        "absolute_frontiers": {
            "start_next_tick": target.start_next_tick,
            "target_next_tick": target.start_next_tick + target.total_ticks,
            "committed_next_tick": target.start_next_tick + target.total_ticks,
        },
        "restored_world_uids": restored_world_uids,
        "dispatch_measurements": measurements,
        "checkpoint_capture_ticks": checkpoint_capture_ticks,
        "graph_inspections": graph_inspections,
        "cuda_event_elapsed_ms_sum": elapsed_sum,
        "aggregate_world_ticks_per_second": (
            target.B * target.total_ticks * 1000.0 / elapsed_sum
        ),
    }
    if controlled is not None:
        body["control_application"] = control_application
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    return validate_workhorse_indexed_continuation_result_v2(
        _write_sealed_json(receipt, execution_result_path)
    )


def validate_workhorse_indexed_continuation_result_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen a sealed result and recompute exact schedule/frontier coverage."""

    from .canonical_rows import (
        _envelope_row,
        _owner_components,
        _read_sealed_receipt,
        _reopen_row_event_sink_receipts_v2,
        validate_workhorse_canonical_parent_authority_binding_v2,
        validate_workhorse_device_envelope_calibration_receipt_v2,
    )
    from .layout import admit_workhorse_vram_v2, estimate_workhorse_resources_v2

    _, value = _read_sealed_receipt(
        receipt_path,
        schema_version=WORKHORSE_INDEXED_CONTINUATION_RESULT_SCHEMA_V2,
        label="Workhorse indexed continuation result",
    )
    continuation = validate_workhorse_indexed_continuation_contract_v2(
        value.get("continuation_contract"),
        reopen_source_payloads=False,
    )
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        value.get("canonical_parent_authority_binding"),
        expected_operation=WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2,
    )
    target = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        continuation["physical_continuation_plan"]["target_contract"]
    )
    stored_fields = {
        "schema_version",
        "operation",
        "claim_class",
        "science_authority_granted",
        "qualification_authority_granted",
        "continuation_contract",
        "canonical_parent_authority_binding",
        "artifact",
        "envelope_binding",
        "semantic_binding",
        "micro_denominator_binding",
        "resource_ledger",
        "planned_admission",
        "runtime_admission",
        "runtime_observation",
        "runtime_inspection",
        "outcome",
        "absolute_frontiers",
        "restored_world_uids",
        "dispatch_measurements",
        "checkpoint_capture_ticks",
        "graph_inspections",
        "cuda_event_elapsed_ms_sum",
        "aggregate_world_ticks_per_second",
        "receipt_content_sha256",
    }
    controlled = continuation.get("scientific_control_contract")
    if controlled is not None:
        stored_fields.add("control_application")
    if set(value) != stored_fields | {"receipt_path", "receipt_file_sha256"}:
        raise WorkhorseIndexedContinuationError(
            "indexed continuation result fields are inexact"
        )
    expected_frontiers = {
        "start_next_tick": target.start_next_tick,
        "target_next_tick": target.start_next_tick + target.total_ticks,
        "committed_next_tick": (
            target.start_next_tick
            if value.get("outcome") == "RUNTIME_HEADROOM_REJECTED"
            else target.start_next_tick + target.total_ticks
        ),
    }
    if (
        value.get("operation") != WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2
        or value.get("claim_class")
        != "TECHNICAL_ONLY_NO_SCIENCE_OR_QUALIFICATION_AUTHORITY"
        or value.get("science_authority_granted") is not False
        or value.get("qualification_authority_granted") is not False
        or value.get("absolute_frontiers") != expected_frontiers
        or parent["contract"] != target.to_dict()
        or parent["source_checkpoint_identity_sha256"]
        != continuation["science_continuation_identity"][
            "source_world_state_set_sha256"
        ]
        or parent["science_continuation_contract_sha256"]
        != continuation["science_continuation_identity"][
            "science_continuation_contract_sha256"
        ]
        or parent["compiled_temporal_k"]
        != continuation["dispatch_plan"]["admitted_temporal_k"]
    ):
        raise WorkhorseIndexedContinuationError(
            "indexed continuation result changed authority or frontier"
        )
    control_application = value.get("control_application")
    if controlled is not None:
        if value.get("outcome") == "RUNTIME_HEADROOM_REJECTED":
            if control_application is not None:
                raise WorkhorseIndexedContinuationError(
                    "headroom-rejected continuation claims control application"
                )
        else:
            application_body = (
                {
                    key: item
                    for key, item in control_application.items()
                    if key != "control_application_sha256"
                }
                if isinstance(control_application, Mapping)
                else {}
            )
            native = controlled["native_mapping"]
            requested_rows = [
                list(native["ordered_values"]) for _ in range(target.B)
            ]
            if (
                type(control_application) is not dict
                or set(control_application)
                != {
                    "schema_version",
                    "time_authority",
                    "control_contract_sha256",
                    "applied_next_absolute_tick",
                    "first_permitted_divergence_tick",
                    "world_uids",
                    "ordered_native_fields",
                    "requested_native_rows",
                    "readback_native_rows",
                    "native_application_flags",
                    "runtime_entrypoints",
                    "control_application_sha256",
                }
                or control_application.get("schema_version")
                != WORKHORSE_SCIENTIFIC_CONTROL_APPLICATION_SCHEMA_V1
                or control_application.get("time_authority")
                != WORKHORSE_TIME_AUTHORITY_V2
                or control_application.get("control_contract_sha256")
                != controlled["control_contract_sha256"]
                or control_application.get("applied_next_absolute_tick")
                != target.start_next_tick
                or control_application.get("first_permitted_divergence_tick")
                != target.start_next_tick
                or control_application.get("world_uids")
                != [
                    row["world_uid"]
                    for row in continuation["repack_binding"][
                        "selected_worlds_in_physical_order"
                    ]
                ]
                or control_application.get("ordered_native_fields")
                != native["ordered_fields"]
                or control_application.get("requested_native_rows")
                != requested_rows
                or control_application.get("readback_native_rows")
                != requested_rows
                or control_application.get("native_application_flags")
                != [1] * target.B
                or control_application.get("runtime_entrypoints")
                != list(WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1)
                or control_application.get("control_application_sha256")
                != canonical_sha256(application_body)
            ):
                raise WorkhorseIndexedContinuationError(
                    "scientific-control application/readback does not recompute"
                )
    elif control_application is not None:
        raise WorkhorseIndexedContinuationError(
            "uncontrolled continuation gained a control application"
        )
    envelope_binding = value.get("envelope_binding")
    if type(envelope_binding) is not dict or set(envelope_binding) != {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
    }:
        raise WorkhorseIndexedContinuationError(
            "indexed continuation envelope binding is inexact"
        )
    try:
        envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
            envelope_binding["receipt_path"]
        )
        components = _owner_components(
            envelope["owner_components_by_k"][str(target.temporal_k)],
            backend_id=target.backend_id,
            temporal_k=target.temporal_k,
        )
        expected_ledger = estimate_workhorse_resources_v2(
            target, owner_components=components
        ).to_dict()
        expected_planned = _envelope_row(
            envelope, B=target.B, temporal_k=target.temporal_k
        )["planned_admission"]
    except Exception as exc:
        raise WorkhorseIndexedContinuationError(
            f"indexed continuation envelope/resource binding is invalid: {exc}"
        ) from exc
    runtime_observation = value.get("runtime_observation")
    observation_body = (
        {
            key: item
            for key, item in runtime_observation.items()
            if key != "observation_sha256"
        }
        if isinstance(runtime_observation, Mapping)
        else {}
    )
    if (
        envelope_binding["receipt_file_sha256"]
        != envelope["receipt_file_sha256"]
        or envelope_binding["receipt_content_sha256"]
        != envelope["receipt_content_sha256"]
        or value.get("artifact") != envelope["artifact"]
        or parent["artifact_sha256"] != envelope["artifact"]["sha256"]
        or parent["sealed_artifact_identity_sha256"]
        != envelope["artifact"]["sealed_artifact_identity_sha256"]
        or value.get("planned_admission") != expected_planned
        or value.get("resource_ledger") != expected_ledger
        or type(runtime_observation) is not dict
        or runtime_observation.get("schema_version")
        != "eiel_workhorse_row_runtime_observation_v2"
        or runtime_observation.get("source")
        != "CUDA_MEM_GET_INFO_UNDER_CANONICAL_EXCLUSIVE_PARENT"
        or runtime_observation.get("device_uuid")
        != envelope["device_identity"]["uuid"]
        or runtime_observation.get("total_device_bytes")
        != envelope["device_identity"]["total_memory_bytes"]
        or runtime_observation.get("observation_sha256")
        != canonical_sha256(observation_body)
        or type(runtime_observation.get("free_bytes")) is not int
        or not 0
        <= runtime_observation["free_bytes"]
        <= runtime_observation["total_device_bytes"]
        or type(value.get("semantic_binding")) is not dict
        or type(value.get("micro_denominator_binding")) is not dict
        or type(value.get("runtime_inspection")) is not dict
    ):
        raise WorkhorseIndexedContinuationError(
            "indexed continuation prerequisite/resource evidence changed"
        )
    expected_admission = admit_workhorse_vram_v2(
        target,
        free_bytes=runtime_observation["free_bytes"],
        total_device_bytes=runtime_observation["total_device_bytes"],
        owner_components=components,
        reserve_bytes=envelope["capacity_basis"]["reserve_bytes_requested"],
    ).to_dict()
    if value.get("runtime_admission") != expected_admission:
        raise WorkhorseIndexedContinuationError(
            "indexed continuation runtime admission changed"
        )
    if value.get("outcome") == "RUNTIME_HEADROOM_REJECTED":
        if (
            value.get("runtime_admission", {}).get("admitted") is not False
            or value.get("dispatch_measurements") != []
            or value.get("checkpoint_capture_ticks") != []
            or value.get("restored_world_uids") != []
            or value.get("graph_inspections") != []
            or value.get("cuda_event_elapsed_ms_sum") is not None
            or value.get("aggregate_world_ticks_per_second") is not None
        ):
            raise WorkhorseIndexedContinuationError(
                "headroom rejection gained execution evidence"
            )
        return value
    measurements = value.get("dispatch_measurements")
    if type(measurements) is not list:
        raise WorkhorseIndexedContinuationError(
            "continuation dispatch measurements are absent"
        )
    expected_dispatches: list[tuple[int, int, int, int]] = []
    cursor = target.start_next_tick
    sample_index = 0
    for segment_index, segment in enumerate(continuation["dispatch_plan"]["segments"]):
        for dispatch_index in range(segment["dispatch_count"]):
            expected_dispatches.append(
                (sample_index, segment_index, dispatch_index, cursor)
            )
            sample_index += 1
            cursor += segment["temporal_k"]
    if len(measurements) != len(expected_dispatches):
        raise WorkhorseIndexedContinuationError(
            "continuation dispatch coverage is incomplete"
        )
    elapsed: list[float] = []
    acknowledged_sinks: list[dict[str, Any]] = []
    for row, expected in zip(measurements, expected_dispatches, strict=True):
        sample, segment_index, dispatch_index, start_tick = expected
        temporal_k = continuation["dispatch_plan"]["segments"][segment_index][
            "temporal_k"
        ]
        row_body = {key: item for key, item in row.items() if key != "measurement_sha256"}
        duration = row.get("cuda_event_elapsed_ms")
        if (
            row.get("sample_index") != sample
            or row.get("segment_index") != segment_index
            or row.get("dispatch_index_in_segment") != dispatch_index
            or row.get("start_next_tick") != start_tick
            or row.get("end_next_tick") != start_tick + temporal_k
            or row.get("temporal_k") != temporal_k
            or row.get("committed_world_ticks") != target.B * temporal_k
            or type(duration) not in {int, float}
            or not math.isfinite(float(duration))
            or float(duration) <= 0
            or row.get("measurement_sha256") != canonical_sha256(row_body)
        ):
            raise WorkhorseIndexedContinuationError(
                "continuation dispatch measurement changed"
            )
        elapsed.append(float(duration))
        sink = row.get("event_sink_receipt")
        if not isinstance(sink, Mapping):
            raise WorkhorseIndexedContinuationError(
                "continuation dispatch lacks its event acknowledgement"
            )
        acknowledged_sinks.append(dict(sink))
    try:
        _reopen_row_event_sink_receipts_v2(acknowledged_sinks)
    except Exception as exc:
        raise WorkhorseIndexedContinuationError(
            f"continuation event acknowledgement did not reopen: {exc}"
        ) from exc
    elapsed_sum = math.fsum(elapsed)
    expected_uids = [
        row["world_uid"]
        for row in continuation["repack_binding"][
            "selected_worlds_in_physical_order"
        ]
    ]
    expected_rate = target.B * target.total_ticks * 1000.0 / elapsed_sum
    graph_inspections = value.get("graph_inspections")
    reported_elapsed = value.get("cuda_event_elapsed_ms_sum")
    reported_rate = value.get("aggregate_world_ticks_per_second")
    if (
        value.get("outcome") != "CONTINUED"
        or value.get("runtime_admission", {}).get("admitted") is not True
        or value.get("restored_world_uids") != expected_uids
        or value.get("checkpoint_capture_ticks")
        != continuation["checkpoint_ticks"]
        or type(graph_inspections) is not list
        or any(type(row) is not dict for row in graph_inspections)
        or type(reported_elapsed) not in {int, float}
        or type(reported_rate) not in {int, float}
        or not math.isclose(
            float(reported_elapsed), elapsed_sum, rel_tol=1e-12
        )
        or not math.isclose(
            float(reported_rate),
            expected_rate,
            rel_tol=1e-12,
        )
    ):
        raise WorkhorseIndexedContinuationError(
            "continuation result does not recompute"
        )
    return value


def seal_workhorse_terminalized_indexed_continuation_receipt_v2(
    output_path: str | Path,
    *,
    execution_result_path: str | Path,
    canonical_terminal_binding: Mapping[str, Any],
    indexed_checkpoint_publications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the technical continuation result to completed CRCA terminalization."""

    from .canonical_rows import (
        _write_sealed_json,
        validate_workhorse_canonical_terminal_binding_v2,
    )

    result = validate_workhorse_indexed_continuation_result_v2(
        execution_result_path
    )
    if result["outcome"] != "CONTINUED":
        raise WorkhorseIndexedContinuationError(
            "only a completed indexed continuation can terminalize"
        )
    parent = result["canonical_parent_authority_binding"]
    terminal = validate_workhorse_canonical_terminal_binding_v2(
        canonical_terminal_binding,
        parent=parent,
    )
    target = result["absolute_frontiers"]["target_next_tick"]
    if terminal["attempt_terminal_status"] != "COMPLETED" or any(
        terminal["final_frontiers"][field] != target
        for field in (
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        )
    ):
        raise WorkhorseIndexedContinuationError(
            "indexed continuation requires an exact-target completed terminal"
        )
    continuation = result["continuation_contract"]
    publications = validate_workhorse_indexed_checkpoint_publication_seals_v2(
        indexed_checkpoint_publications,
        contract=FullEvolutionCudaWorkhorseContractV2.from_mapping(
            continuation["physical_continuation_plan"]["target_contract"]
        ),
        checkpoint_ticks=continuation["checkpoint_ticks"],
    )
    body = {
        "schema_version": WORKHORSE_TERMINALIZED_INDEXED_CONTINUATION_SCHEMA_V2,
        "operation": WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2,
        "claim_class": "TECHNICAL_ONLY_NO_SCIENCE_OR_QUALIFICATION_AUTHORITY",
        "science_authority_granted": False,
        "qualification_authority_granted": False,
        "science_continuation_identity": continuation[
            "science_continuation_identity"
        ],
        "physical_continuation_plan": continuation[
            "physical_continuation_plan"
        ],
        "absolute_frontiers": result["absolute_frontiers"],
        "checkpoint_ticks": continuation["checkpoint_ticks"],
        "indexed_checkpoint_publications": publications,
        "aggregate_world_ticks_per_second": result[
            "aggregate_world_ticks_per_second"
        ],
        "execution_result": {
            "path": result["receipt_path"],
            "file_sha256": result["receipt_file_sha256"],
            "content_sha256": result["receipt_content_sha256"],
        },
        "canonical_parent_authority_binding": parent,
        "canonical_terminal_binding": terminal,
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    return validate_workhorse_terminalized_indexed_continuation_receipt_v2(
        _write_sealed_json(receipt, output_path)
    )


def validate_workhorse_terminalized_indexed_continuation_receipt_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    from .canonical_rows import (
        _read_sealed_receipt,
        validate_workhorse_canonical_terminal_binding_v2,
    )

    _, value = _read_sealed_receipt(
        receipt_path,
        schema_version=WORKHORSE_TERMINALIZED_INDEXED_CONTINUATION_SCHEMA_V2,
        label="terminalized Workhorse indexed continuation",
    )
    stored_fields = {
        "schema_version",
        "operation",
        "claim_class",
        "science_authority_granted",
        "qualification_authority_granted",
        "science_continuation_identity",
        "physical_continuation_plan",
        "absolute_frontiers",
        "checkpoint_ticks",
        "indexed_checkpoint_publications",
        "aggregate_world_ticks_per_second",
        "execution_result",
        "canonical_parent_authority_binding",
        "canonical_terminal_binding",
        "receipt_content_sha256",
    }
    if set(value) != stored_fields | {"receipt_path", "receipt_file_sha256"}:
        raise WorkhorseIndexedContinuationError(
            "terminalized continuation receipt fields are inexact"
        )
    result_binding = value.get("execution_result")
    if type(result_binding) is not dict or set(result_binding) != {
        "path",
        "file_sha256",
        "content_sha256",
    }:
        raise WorkhorseIndexedContinuationError(
            "terminalized continuation result binding is inexact"
        )
    result = validate_workhorse_indexed_continuation_result_v2(
        result_binding["path"]
    )
    parent = result["canonical_parent_authority_binding"]
    terminal = validate_workhorse_canonical_terminal_binding_v2(
        value.get("canonical_terminal_binding"), parent=parent
    )
    target = result["absolute_frontiers"]["target_next_tick"]
    continuation = result["continuation_contract"]
    publications = validate_workhorse_indexed_checkpoint_publication_seals_v2(
        value.get("indexed_checkpoint_publications"),
        contract=FullEvolutionCudaWorkhorseContractV2.from_mapping(
            continuation["physical_continuation_plan"]["target_contract"]
        ),
        checkpoint_ticks=continuation["checkpoint_ticks"],
    )
    if (
        value.get("operation") != WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2
        or value.get("claim_class")
        != "TECHNICAL_ONLY_NO_SCIENCE_OR_QUALIFICATION_AUTHORITY"
        or value.get("science_authority_granted") is not False
        or value.get("qualification_authority_granted") is not False
        or result_binding["file_sha256"] != result["receipt_file_sha256"]
        or result_binding["content_sha256"] != result["receipt_content_sha256"]
        or value.get("science_continuation_identity")
        != continuation["science_continuation_identity"]
        or value.get("physical_continuation_plan")
        != continuation["physical_continuation_plan"]
        or value.get("absolute_frontiers") != result["absolute_frontiers"]
        or value.get("checkpoint_ticks") != continuation["checkpoint_ticks"]
        or value.get("indexed_checkpoint_publications") != publications
        or value.get("aggregate_world_ticks_per_second")
        != result["aggregate_world_ticks_per_second"]
        or value.get("canonical_parent_authority_binding") != parent
        or terminal["attempt_terminal_status"] != "COMPLETED"
        or any(
            terminal["final_frontiers"][field] != target
            for field in (
                "last_dispatched_next_tick",
                "last_computed_next_tick",
                "durable_checkpoint_next_tick",
                "published_artifact_next_tick",
            )
        )
    ):
        raise WorkhorseIndexedContinuationError(
            "terminalized indexed continuation does not reopen exactly"
        )
    return value


__all__ = [
    "WORKHORSE_INDEXED_CONTINUATION_CONTRACT_SCHEMA_V2",
    "WORKHORSE_INDEXED_CONTINUATION_OPERATION_V2",
    "WORKHORSE_INDEXED_CONTINUATION_RESULT_SCHEMA_V2",
    "WORKHORSE_INDEXED_CONTINUATION_AGGREGATE_WTPS_METRIC_V2",
    "WORKHORSE_INDEXED_CHECKPOINT_PUBLICATION_SCHEMA_V2",
    "WORKHORSE_SCIENTIFIC_CONTROL_APPLICATION_SCHEMA_V1",
    "WORKHORSE_SCIENTIFIC_CONTROL_CONTRACT_SCHEMA_V1",
    "WORKHORSE_TERMINALIZED_INDEXED_CONTINUATION_SCHEMA_V2",
    "WorkhorseIndexedContinuationError",
    "validate_workhorse_indexed_continuation_contract_v2",
    "validate_workhorse_indexed_continuation_result_v2",
    "validate_workhorse_indexed_checkpoint_publication_seals_v2",
    "validate_workhorse_scientific_control_contract_v1",
    "run_workhorse_indexed_continuation_v2",
    "seal_workhorse_terminalized_indexed_continuation_receipt_v2",
    "validate_workhorse_terminalized_indexed_continuation_receipt_v2",
    "workhorse_indexed_continuation_contract_v2",
    "workhorse_scientific_control_contract_v1",
]
