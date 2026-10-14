"""Runtime-inspected semantic and scaling probes for CUDA workhorse V2.

Semantic audits deliberately use B<=2 while executing the exact large-shape
artifact. Scaling receipts retain the complete attempted B/K envelope so a
future route can identify topology and VRAM regime changes, not only a winner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Final, Iterator, Mapping, Sequence

from .attempt_authority import (
    AttemptAuthorityValidatorV2,
    WorkhorseAttemptAuthorityError,
    resolve_injected_workhorse_attempt_authority_v2,
    validate_workhorse_attempt_authority_binding_v2,
    workhorse_attempt_authority_request_v2,
)

from .authority import (
    FullEvolutionWorkhorseAuthorityError,
    publish_workhorse_sparse_event_sink_receipt_v3,
    reopen_workhorse_checkpoint_bundle_v2,
    validate_authoritative_micro_denominator_v2,
    validate_workhorse_event_sink_receipt_v2,
    validate_workhorse_semantic_gate_receipt_v2,
    workhorse_governing_contract_bindings_v2,
    write_workhorse_checkpoint_bundle_v2,
    write_workhorse_semantic_gate_receipt_v2,
)

from .contract import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    BACKEND_PHASE_GRAPH_V2,
    BACKEND_REFERENCE_PHASED_V2,
    FullEvolutionCudaWorkhorseContractV2,
    FullEvolutionWorkhorseContractError,
    canonical_sha256,
    validate_workhorse_world_identity_binding_v2,
    workhorse_backend_v2,
    workhorse_shape_v2,
)
from .layout import (
    WorkhorseLayoutV2,
    WorkhorseOwnerComponentsV2,
    admit_workhorse_vram_v2,
    estimate_workhorse_resources_v2,
    workhorse_allocator_owner_overhead_bound_v2,
    workhorse_attempt_identity_v2,
    workhorse_candidate_envelope_v2,
    workhorse_layout_v2,
    workhorse_topology_resource_binding_v2,
    validate_workhorse_topology_resource_binding_v2,
)
from .indexed_checkpoint import (
    WorkhorseIndexedCheckpointError,
    validate_workhorse_repack_binding_v2,
)
from .native_loader import (
    file_sha256,
    load_workhorse_prebuilt_v2,
    validate_workhorse_prebuilt_v2,
)
from .semantic_evidence import (
    derive_workhorse_semantic_checks_v2,
    write_workhorse_semantic_evidence_bundle_v2,
)
from .semantic_cases import (
    science_compatible_base_checkpoint_identity_v2,
    validate_external_checkpoint_source_binding_v2,
    validate_workhorse_science_checkpoint_compatibility_binding_v2,
    validate_workhorse_semantic_case_parent_binding_v2,
    workhorse_semantic_case_law_v2,
    workhorse_semantic_case_terminal_contract_v2,
    write_workhorse_semantic_case_fragment_v2,
    reopen_workhorse_semantic_case_fragment_v2,
)


class FullEvolutionWorkhorseProbeError(RuntimeError):
    """Raised when a physical workhorse claim lacks an exact runtime witness."""


DISABLE_PATCH: Final[int] = 1 << 0
DISABLE_WDL: Final[int] = 1 << 1
DISABLE_JUNGLE: Final[int] = 1 << 2
DISABLE_EED: Final[int] = 1 << 3
DISABLE_BIRTH: Final[int] = 1 << 4
DISABLE_MUTATION: Final[int] = 1 << 5
DISABLE_RECOMBINATION: Final[int] = 1 << 6
_EVENT_STRUCT: Final[struct.Struct] = struct.Struct("<QQQQQIHBB")


@dataclass(frozen=True, slots=True)
class _LoadedArtifact:
    artifact: Path
    manifest_path: Path
    manifest: dict[str, Any]
    shape: Any
    backend: Any
    layout: WorkhorseLayoutV2
    module: Any


@dataclass(slots=True)
class _OwnedBuffers:
    """Trusted event-frontier owner; a pending dispatch cannot be reused."""

    loaded: _LoadedArtifact
    state: Any
    scratch: Any
    events: Any
    counts: Any
    temporal_k: int
    world_uids: tuple[int, ...]
    rng_keys: tuple[int, ...]
    committed_next_tick: int
    attempt_authority_binding: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    last_sink_ack: dict[str, Any] | None = None
    last_event_canonicalization_tail_ms: float | None = None
    last_dense_events_cpu: Any | None = None
    poisoned_status: tuple[int, ...] = ()

    def __iter__(self) -> Iterator[Any]:
        return iter((self.state, self.scratch, self.events, self.counts))

    def __getitem__(self, index: int) -> Any:
        return (self.state, self.scratch, self.events, self.counts)[index]


def _torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise FullEvolutionWorkhorseProbeError(
            f"PyTorch is unavailable: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise FullEvolutionWorkhorseProbeError("physical CUDA device is unavailable")
    return torch


def _load(path: str | Path, torch: Any) -> _LoadedArtifact:
    artifact, manifest_path, manifest = validate_workhorse_prebuilt_v2(path)
    shape = workhorse_shape_v2(str(manifest["shape_id"]))
    backend = workhorse_backend_v2(str(manifest["backend_id"]))
    return _LoadedArtifact(
        artifact=artifact,
        manifest_path=manifest_path,
        manifest=manifest,
        shape=shape,
        backend=backend,
        layout=workhorse_layout_v2(shape),
        module=load_workhorse_prebuilt_v2(artifact, device=torch.cuda.current_device()),
    )


def _field(tensor: Any, layout: WorkhorseLayoutV2, name: str, torch: Any) -> Any:
    field = layout.field(name)
    dtype = {
        "u8": torch.uint8,
        "u16": torch.uint16,
        "u32": torch.uint32,
        "i32": torch.int32,
        "u64": torch.int64,
    }[field.dtype]
    return (
        tensor[:, field.offset : field.offset + field.nbytes]
        .view(dtype)
        .reshape(tensor.size(0), field.count)
    )


def _sha_tensor(tensor: Any) -> str:
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def _tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _allocate_owner(
    torch: Any, loaded: _LoadedArtifact, B: int, temporal_k: int
) -> _OwnedBuffers:
    layout = loaded.layout
    device = torch.device("cuda", torch.cuda.current_device())
    buffers = (
        torch.empty(
            (B, layout.state_bytes_per_world), dtype=torch.uint8, device=device
        ),
        torch.empty(
            (B, layout.scratch_bytes_per_world), dtype=torch.uint8, device=device
        ),
        torch.zeros(
            (B, 2 * layout.shape.capacity, 48),
            dtype=torch.uint8,
            device=device,
        ),
        torch.zeros((B,), dtype=torch.int64, device=device),
    )
    return _OwnedBuffers(
        loaded=loaded,
        state=buffers[0],
        scratch=buffers[1],
        events=buffers[2],
        counts=buffers[3],
        temporal_k=temporal_k,
        world_uids=(),
        rng_keys=(),
        committed_next_tick=0,
    )


def _initialize_owner(
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    *,
    world_uid_base: int,
    start_tick: int,
    rng_key: int,
    fixture: bool,
) -> None:
    state, scratch, _, _ = owner
    loaded.module.initialize(state, scratch, world_uid_base, start_tick, rng_key)
    if fixture:
        loaded.module.force_causal_fixture(state, scratch)
    owner.world_uids = tuple(world_uid_base + world for world in range(state.size(0)))
    owner.rng_keys = (rng_key,) * state.size(0)
    owner.committed_next_tick = start_tick
    owner.pending = None
    owner.last_sink_ack = None
    owner.last_event_canonicalization_tail_ms = None
    owner.last_dense_events_cpu = None
    owner.poisoned_status = ()


def _initialize_owner_exact(
    torch: Any,
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    *,
    world_identity_binding: Mapping[str, Any],
    start_tick: int,
    fixture: bool,
) -> None:
    """Initialize one physical owner from stable per-world carrier identity."""

    binding = validate_workhorse_world_identity_binding_v2(
        world_identity_binding,
        expected_world_count=int(owner.state.size(0)),
    )
    world_uids = torch.tensor(
        binding["world_uids"],
        dtype=torch.int64,
        device=owner.state.device,
    )
    rng_keys = torch.tensor(
        binding["rng_keys"],
        dtype=torch.int64,
        device=owner.state.device,
    )
    loaded.module.initialize_exact(
        owner.state,
        owner.scratch,
        world_uids,
        start_tick,
        rng_keys,
    )
    if fixture:
        loaded.module.force_causal_fixture(owner.state, owner.scratch)
    owner.world_uids = tuple(binding["world_uids"])
    owner.rng_keys = tuple(binding["rng_keys"])
    owner.committed_next_tick = start_tick
    owner.pending = None
    owner.last_sink_ack = None
    owner.last_event_canonicalization_tail_ms = None
    owner.last_dense_events_cpu = None
    owner.poisoned_status = ()


def _restore_owner_from_repack_binding_v2(
    torch: Any,
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    *,
    repack_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore selected canonical rows without materializing an aggregate copy."""

    try:
        binding = validate_workhorse_repack_binding_v2(repack_binding)
        target = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            binding["target_contract"]
        )
    except (WorkhorseIndexedCheckpointError, FullEvolutionWorkhorseContractError) as exc:
        owner.poisoned_status = (-1,)
        raise FullEvolutionWorkhorseProbeError(
            f"indexed checkpoint repack is invalid: {exc}"
        ) from exc
    if (
        owner.loaded is not loaded
        or target.shape_id != loaded.shape.shape_id
        or target.backend_id != loaded.backend.backend_id
        or target.B != int(owner.state.size(0))
        or target.temporal_k != owner.temporal_k
        or tuple(owner.state.shape)
        != (target.B, loaded.layout.state_bytes_per_world)
        or tuple(owner.scratch.shape)
        != (target.B, loaded.layout.scratch_bytes_per_world)
        or target.start_next_tick != binding["checkpoint_at_tick"]
    ):
        owner.poisoned_status = (-1,)
        raise FullEvolutionWorkhorseProbeError(
            "indexed checkpoint repack targets another owner, shape, backend, B, K, or tick"
        )
    locators = binding["source_locators"]
    expected_slots = list(range(target.B))
    if [locator["target_physical_slot"] for locator in locators] != expected_slots:
        owner.poisoned_status = (-1,)
        raise FullEvolutionWorkhorseProbeError(
            "indexed checkpoint repack does not exactly cover target physical slots"
        )
    handles: dict[Path, Any] = {}
    try:
        for locator in locators:
            source_path = Path(locator["source_state_path"]).resolve()
            if source_path.is_symlink() or not source_path.is_file():
                raise FullEvolutionWorkhorseProbeError(
                    "indexed checkpoint source state is not a regular file"
                )
            handle = handles.get(source_path)
            if handle is None:
                handle = source_path.open("rb")
                handles[source_path] = handle
            handle.seek(locator["source_state_offset"])
            state_bytes = handle.read(locator["state_bytes"])
            if (
                len(state_bytes) != loaded.layout.state_bytes_per_world
                or hashlib.sha256(state_bytes).hexdigest()
                != locator["state_sha256"]
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "indexed checkpoint source row is truncated or changed"
                )
            header = struct.unpack("<32Q", state_bytes[: 32 * 8])
            if (
                header[0] != locator["world_uid"]
                or header[1] != binding["checkpoint_at_tick"]
                or header[4] != 0
                or header[13] != locator["rng_key"]
                or header[14] != 4
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "indexed checkpoint source row has another carrier or frontier"
                )
            owner.state[locator["target_physical_slot"]].copy_(
                torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8)
            )
        owner.scratch.zero_()
        owner.events.zero_()
        owner.counts.zero_()
        loaded.module.restore_derived(owner.state, owner.scratch)
        for locator in locators:
            restored_sha256 = hashlib.sha256(
                _tensor_bytes(owner.state[locator["target_physical_slot"]])
            ).hexdigest()
            if restored_sha256 != locator["state_sha256"]:
                raise FullEvolutionWorkhorseProbeError(
                    "derived-state reconstruction changed canonical state"
                )
    except Exception:
        owner.poisoned_status = (-1,)
        raise
    finally:
        for handle in handles.values():
            handle.close()
    owner.world_uids = tuple(
        locator["world_uid"] for locator in binding["source_locators"]
    )
    owner.rng_keys = tuple(
        locator["rng_key"] for locator in binding["source_locators"]
    )
    owner.committed_next_tick = binding["checkpoint_at_tick"]
    owner.pending = None
    owner.last_sink_ack = None
    owner.last_event_canonicalization_tail_ms = None
    owner.last_dense_events_cpu = None
    owner.poisoned_status = ()
    return binding


def _authorize_owner_v2(
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    *,
    temporal_k: int,
    phase: str,
    validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
    candidate_envelope_sha256: str | None = None,
    prior_lineage_sha256: str | None = None,
    source_checkpoint_identity_sha256: str | None = None,
    total_ticks: int | None = None,
) -> dict[str, Any]:
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=loaded.shape.shape_id,
        backend_id=loaded.backend.backend_id,
        B=owner.state.size(0),
        temporal_k=temporal_k,
        total_ticks=temporal_k if total_ticks is None else total_ticks,
        start_next_tick=owner.committed_next_tick,
    )
    binding = _resolve_contract_authority_v2(
        loaded,
        contract=contract,
        phase=phase,
        validator=validator,
        device_uuid=device_uuid,
        candidate_envelope_sha256=candidate_envelope_sha256,
        prior_lineage_sha256=prior_lineage_sha256,
        source_checkpoint_identity_sha256=source_checkpoint_identity_sha256,
    )
    if phase != "PLAN_REJECTION":
        owner.attempt_authority_binding = binding
    return binding


def _resolve_contract_authority_v2(
    loaded: _LoadedArtifact,
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    phase: str,
    validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
    candidate_envelope_sha256: str | None = None,
    prior_lineage_sha256: str | None = None,
    source_checkpoint_identity_sha256: str | None = None,
) -> dict[str, Any]:
    request = workhorse_attempt_authority_request_v2(
        phase=phase,
        contract=contract,
        artifact_sha256=file_sha256(loaded.artifact),
        sealed_artifact_identity_sha256=loaded.manifest["sealed_artifact_identity"][
            "sealed_artifact_identity_sha256"
        ],
        device_uuid=device_uuid,
        candidate_envelope_sha256=candidate_envelope_sha256,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
        prior_lineage_sha256=prior_lineage_sha256,
        source_checkpoint_identity_sha256=source_checkpoint_identity_sha256,
    )
    try:
        binding = resolve_injected_workhorse_attempt_authority_v2(
            validator, request=request
        )
    except WorkhorseAttemptAuthorityError as exc:
        raise FullEvolutionWorkhorseProbeError(str(exc)) from exc
    return binding


def _closeout_attempt_authority_v2(
    loaded: _LoadedArtifact,
    *,
    started_binding: Mapping[str, Any],
    validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> dict[str, Any]:
    """Close one exact started lineage after all device ownership is released."""
    try:
        started = validate_workhorse_attempt_authority_binding_v2(
            started_binding, historical=True
        )
        request = started["request"]
        if request["phase"] not in {"CALIBRATION", "EXECUTION_ROW"}:
            raise WorkhorseAttemptAuthorityError(
                "closeout source must be a started calibration or execution row"
            )
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            request["contract"]
        )
    except (KeyError, TypeError, ValueError, WorkhorseAttemptAuthorityError) as exc:
        raise FullEvolutionWorkhorseProbeError(
            f"cannot reopen started Attempt for closeout: {exc}"
        ) from exc
    closeout = _resolve_contract_authority_v2(
        loaded,
        contract=contract,
        phase="CLOSEOUT",
        validator=validator,
        device_uuid=device_uuid,
        candidate_envelope_sha256=request["candidate_envelope_sha256"],
        prior_lineage_sha256=started["attempt_lineage_sha256"],
    )
    if closeout["attempt_lineage_sha256"] != started["attempt_lineage_sha256"]:
        raise FullEvolutionWorkhorseProbeError(
            "Attempt closeout changed the exact started lineage"
        )
    return closeout


def _attempt_lifecycle_row_v2(
    role: str,
    execution_authority: Mapping[str, Any],
    closeout_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if type(role) is not str or not role:
        raise FullEvolutionWorkhorseProbeError("Attempt lifecycle role is required")
    body = {
        "schema_version": "eiel_workhorse_attempt_lifecycle_v2",
        "role": role,
        "execution_attempt_authority": dict(execution_authority),
        "closeout_attempt_authority": dict(closeout_authority),
    }
    return {**body, "attempt_lifecycle_sha256": canonical_sha256(body)}


def _prepare_dispatch_v2(
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    temporal_k: int,
    disable_mask: int,
) -> None:
    if owner.loaded is not loaded:
        raise FullEvolutionWorkhorseProbeError("event owner/artifact mismatch")
    if owner.pending is not None:
        raise FullEvolutionWorkhorseProbeError(
            "event buffer reuse requires an external exact-content sink acknowledgement"
        )
    if owner.poisoned_status:
        raise FullEvolutionWorkhorseProbeError(
            f"event owner is poisoned by status {owner.poisoned_status}"
        )
    if owner.attempt_authority_binding is None:
        raise FullEvolutionWorkhorseProbeError(
            "dispatch requires an injected live Attempt/exclusivity binding"
        )
    raw_binding = owner.attempt_authority_binding
    if (
        isinstance(raw_binding, Mapping)
        and raw_binding.get("schema_version")
        == "eiel_workhorse_canonical_parent_attempt_binding_v2"
    ):
        try:
            from .canonical_rows import (
                validate_workhorse_canonical_parent_authority_binding_v2,
            )

            operation = raw_binding.get("operation")
            if operation not in {
                "EXECUTION_ROW",
                "SEMANTIC_EVIDENCE_CASE",
                "SCIENCE_COMPATIBILITY_DEMO",
                "INDEXED_ACS_CONTINUATION",
            }:
                raise FullEvolutionWorkhorseProbeError(
                    "canonical parent operation cannot dispatch this owner"
                )
            binding = validate_workhorse_canonical_parent_authority_binding_v2(
                raw_binding, expected_operation=operation
            )
            contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
                binding["contract"]
            )
        except Exception as exc:
            raise FullEvolutionWorkhorseProbeError(
                f"canonical parent Attempt binding is invalid: {exc}"
            ) from exc
        temporal_k_admitted = (
            temporal_k in binding["compiled_temporal_k"]
            if operation == "INDEXED_ACS_CONTINUATION"
            else contract.temporal_k == temporal_k
        )
        span = binding["absolute_tick_span"]
        if (
            contract.B != owner.state.size(0)
            or not temporal_k_admitted
            or owner.committed_next_tick + temporal_k > span["target_next_tick"]
            or binding["artifact_sha256"] != file_sha256(loaded.artifact)
        ):
            raise FullEvolutionWorkhorseProbeError(
                "dispatch B/K/artifact differs from canonical parent Attempt"
            )
        authority_binding_sha256 = binding["binding_sha256"]
    else:
        try:
            binding = validate_workhorse_attempt_authority_binding_v2(raw_binding)
        except WorkhorseAttemptAuthorityError as exc:
            raise FullEvolutionWorkhorseProbeError(str(exc)) from exc
        request = binding["request"]
        if (
            request["B"] != owner.state.size(0)
            or request["temporal_k"] != temporal_k
            or request["artifact_sha256"] != file_sha256(loaded.artifact)
        ):
            raise FullEvolutionWorkhorseProbeError(
                "dispatch B/K/artifact differs from injected Attempt authority"
            )
        authority_binding_sha256 = binding["authority_binding_sha256"]
    owner.last_event_canonicalization_tail_ms = None
    pre_state_bytes = _tensor_bytes(owner.state)
    pre_state_sha256 = hashlib.sha256(pre_state_bytes).hexdigest()
    owner.pending = {
        "start_next_absolute_tick": owner.committed_next_tick,
        "end_next_absolute_tick": owner.committed_next_tick + temporal_k,
        "temporal_k": temporal_k,
        "disable_mask": disable_mask,
        "pre_state_sha256": pre_state_sha256,
        "pre_state_bytes": pre_state_bytes,
        "physical_segments": [],
        "overflow_recovery_count": 0,
        "attempt_authority_binding_sha256": authority_binding_sha256,
    }


def _validate_graph_inspection_v2(
    inspection: object,
    *,
    shape: Any,
    backend_id: str,
    B: int,
    temporal_k: int,
    disable_mask: int,
) -> None:
    """Recompute graph topology and launch geometry from inspected node rows."""
    if backend_id != BACKEND_PHASE_GRAPH_V2:
        if inspection is not None:
            raise FullEvolutionWorkhorseProbeError(
                "non-graph backend returned a CUDA Graph inspection"
            )
        return
    if type(inspection) is not dict:
        raise FullEvolutionWorkhorseProbeError(
            "phase_graph_v2 requires a runtime graph inspection"
        )
    expected_tick_sequence = workhorse_backend_v2(backend_id).topology_payload(shape)[
        "kernels_per_tick"
    ]
    expected_sequence = expected_tick_sequence * temporal_k
    nodes = inspection.get("nodes")
    if (
        inspection.get("schema_version")
        != "eiel_full_evolution_cuda_graph_inspection_v2"
        or inspection.get("runtime_inspected") is not True
        or inspection.get("shape_id") != shape.shape_id
        or inspection.get("backend_id") != BACKEND_PHASE_GRAPH_V2
        or inspection.get("B") != B
        or inspection.get("temporal_k") != temporal_k
        or inspection.get("disable_mask") != disable_mask
        or inspection.get("linear_chain") is not True
        or inspection.get("kernel_sequence_exact") is not True
        or inspection.get("kernel_geometry_exact") is not True
        or inspection.get("kernel_sequence") != expected_sequence
        or inspection.get("expected_kernel_sequence") != expected_sequence
        or inspection.get("kernel_node_count") != 15 * temporal_k
        or inspection.get("memset_node_count") != 1
        or inspection.get("node_count") != 15 * temporal_k + 1
        or inspection.get("edge_count") != 15 * temporal_k
        or type(inspection.get("graph_object_device_bytes")) is not int
        or inspection["graph_object_device_bytes"] < 0
        or type(inspection.get("graph_exec_device_bytes")) is not int
        or inspection["graph_exec_device_bytes"] < 0
        or type(nodes) is not list
        or len(nodes) != 15 * temporal_k + 1
    ):
        raise FullEvolutionWorkhorseProbeError(
            "runtime CUDA Graph identity/topology is invalid"
        )
    if any(
        type(node) is not dict or node.get("topological_rank") != rank
        for rank, node in enumerate(nodes)
    ):
        raise FullEvolutionWorkhorseProbeError(
            "runtime CUDA Graph topological ranks are invalid"
        )
    memset_nodes = [node for node in nodes if node.get("node_type") == "memset"]
    kernel_nodes = [node for node in nodes if node.get("node_type") == "kernel"]
    if (
        len(memset_nodes) != 1
        or len(kernel_nodes) != len(expected_sequence)
        or [node.get("kernel") for node in kernel_nodes] != expected_sequence
        or set(node.get("node_type") for node in nodes) != {"memset", "kernel"}
    ):
        raise FullEvolutionWorkhorseProbeError(
            "runtime CUDA Graph node types/sequence are invalid"
        )
    for node in kernel_nodes:
        kernel = node["kernel"]
        expected_grid_x = (
            1
            if kernel
            in {
                "rank_prefix_phase_v2",
                "mate_spawn_phase_v2",
                "compact_death_events_v3",
                "compact_birth_events_v3",
                "publish_phase_v2",
            }
            else (
                shape.site_tiles
                if kernel == "environment_phase_v2"
                else shape.agent_tiles
            )
        )
        if (
            tuple(node.get("grid", ())) != (expected_grid_x, B, 1)
            or tuple(node.get("block", ())) != (256, 1, 1)
            or any(
                type(node.get(field)) is not int or node[field] < 0
                for field in (
                    "dynamic_shared_bytes",
                    "registers_per_thread",
                    "static_shared_bytes",
                    "local_bytes_per_thread",
                    "maximum_threads_per_block",
                )
            )
        ):
            raise FullEvolutionWorkhorseProbeError(
                "runtime CUDA Graph kernel geometry/resources are invalid"
            )


def _prepare_graph_if_needed(
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    temporal_k: int,
    disable_mask: int,
) -> dict[str, Any] | None:
    if loaded.backend.backend_id != BACKEND_PHASE_GRAPH_V2:
        return None
    loaded.module.prepare_graph(*owner, temporal_k, disable_mask)
    inspection = dict(loaded.module.graph_inspect())
    try:
        _validate_graph_inspection_v2(
            inspection,
            shape=loaded.shape,
            backend_id=loaded.backend.backend_id,
            B=owner.state.size(0),
            temporal_k=temporal_k,
            disable_mask=disable_mask,
        )
    except FullEvolutionWorkhorseProbeError:
        loaded.module.release_graph()
        raise
    return inspection


def _release_graph(loaded: _LoadedArtifact) -> None:
    if loaded.backend.backend_id == BACKEND_PHASE_GRAPH_V2:
        loaded.module.release_graph()


def _advance(
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    temporal_k: int,
    disable_mask: int,
) -> None:
    if owner.pending is None:
        _prepare_dispatch_v2(loaded, owner, temporal_k, disable_mask)
    elif (
        owner.pending.get("temporal_k") != temporal_k
        or owner.pending.get("disable_mask") != disable_mask
    ):
        raise FullEvolutionWorkhorseProbeError(
            "prepared dispatch K/ablation identity changed before launch"
        )
    if disable_mask:
        loaded.module.advance_diagnostic_ablation(*owner, temporal_k, disable_mask)
    else:
        loaded.module.advance(*owner, temporal_k)


def _sparse_page_rows_v3(
    page: Any,
    counts: Sequence[int],
    *,
    start_tick: int,
    end_tick: int,
    capacity: int,
    sites: int,
) -> list[list[bytes]]:
    page_rows = int(page.shape[1])
    if len(counts) != int(page.shape[0]):
        raise FullEvolutionWorkhorseProbeError(
            "sparse page count projection has another B"
        )
    raw = page.contiguous().numpy().tobytes()
    per_world: list[list[bytes]] = []
    for world, required_count in enumerate(counts):
        active: list[tuple[tuple[int, int, int], bytes]] = []
        base = world * page_rows * 48
        for row_index in range(page_rows):
            row = raw[base + row_index * 48 : base + (row_index + 1) * 48]
            values = _EVENT_STRUCT.unpack(row)
            if values[7] == 0:
                continue
            event_id = int(values[1])
            event_kind = int(values[6])
            slot = event_id & ((1 << 48) - 1)
            key = (int(values[0]), event_kind, slot)
            if (
                not start_tick <= key[0] < end_tick
                or event_kind not in (1, 2)
                or event_id != (event_kind << 48) | slot
                or slot >= capacity
                or int(values[5]) >= sites
                or int(values[2]) <= 0
                or (event_kind == 2 and (int(values[3]) <= 0 or int(values[4]) <= 0))
                or values[7] != 1
                or values[8] != 0
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "native sparse page contains an invalid active event"
                )
            active.append((key, row))
        active.sort(key=lambda item: item[0])
        if len({key for key, _row in active}) != len(active):
            raise FullEvolutionWorkhorseProbeError(
                "native sparse page contains duplicate event keys"
            )
        if required_count <= page_rows and len(active) != required_count:
            raise FullEvolutionWorkhorseProbeError(
                "native sparse page active rows differ from its complete count"
            )
        if required_count > page_rows and len(active) != page_rows:
            raise FullEvolutionWorkhorseProbeError(
                "native sparse overflow page is not an exact full prefix"
            )
        per_world.append([row for _key, row in active])
    return per_world


def _drain_sparse_page_v3(
    owner: _OwnedBuffers,
    torch: Any,
    *,
    start_tick: int,
    end_tick: int,
) -> tuple[list[list[bytes]], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    torch.cuda.synchronize()
    owner.loaded.module.canonicalize_events(owner.events)
    torch.cuda.synchronize()
    page = owner.events.detach().cpu().contiguous()
    counts = tuple(int(value) for value in owner.counts.detach().cpu().tolist())
    header = _field(owner.state, owner.loaded.layout, "header", torch).detach().cpu()
    statuses = tuple(int(value) for value in header[:, 4].tolist())
    next_ticks = tuple(int(value) for value in header[:, 1].tolist())
    if any(count < 0 for count in counts):
        raise FullEvolutionWorkhorseProbeError(
            "native sparse page returned a negative required count"
        )
    return (
        _sparse_page_rows_v3(
            page,
            counts,
            start_tick=start_tick,
            end_tick=end_tick,
            capacity=owner.loaded.shape.capacity,
            sites=owner.loaded.shape.sites,
        ),
        counts,
        statuses,
        next_ticks,
    )


def _restore_owner_state_bytes_v3(
    owner: _OwnedBuffers, torch: Any, state_bytes: bytes
) -> None:
    if len(state_bytes) != int(owner.state.numel()):
        raise FullEvolutionWorkhorseProbeError(
            "sparse overflow rollback state byte extent changed"
        )
    canonical = torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8).view(
        tuple(owner.state.shape)
    )
    owner.state.copy_(canonical)
    owner.loaded.module.restore_derived(owner.state, owner.scratch)
    torch.cuda.synchronize()
    if _tensor_bytes(owner.state) != state_bytes:
        raise FullEvolutionWorkhorseProbeError(
            "sparse overflow rollback did not restore exact canonical state"
        )


def _recover_sparse_overflow_v3(
    owner: _OwnedBuffers,
    torch: Any,
    *,
    start_tick: int,
    temporal_k: int,
    disable_mask: int,
    snapshot_bytes: bytes,
) -> tuple[list[list[bytes]], list[dict[str, Any]], int]:
    page_rows = int(owner.events.shape[1])
    loaded = owner.loaded

    def run_segment(
        segment_start: int,
        segment_ticks: int,
        segment_snapshot: bytes,
    ) -> tuple[list[list[bytes]], list[dict[str, Any]], int]:
        if loaded.backend.backend_id == BACKEND_PHASE_GRAPH_V2:
            loaded.module.release_graph()
            _prepare_graph_if_needed(loaded, owner, segment_ticks, disable_mask)
        if disable_mask:
            loaded.module.advance_diagnostic_ablation(
                *owner, segment_ticks, disable_mask
            )
        else:
            loaded.module.advance(*owner, segment_ticks)
        segment_end = segment_start + segment_ticks
        rows, counts, statuses, next_ticks = _drain_sparse_page_v3(
            owner,
            torch,
            start_tick=segment_start,
            end_tick=segment_end,
        )
        overflow = any(count > page_rows for count in counts)
        if any(statuses):
            raise FullEvolutionWorkhorseProbeError(
                "overflow recovery encountered a semantic transaction failure"
            )
        if overflow:
            if segment_ticks == 1:
                raise FullEvolutionWorkhorseProbeError(
                    "one absolute tick exceeded the audited 2*C event page"
                )
            _restore_owner_state_bytes_v3(owner, torch, segment_snapshot)
            first_ticks = segment_ticks // 2
            first_rows, first_segments, first_retries = run_segment(
                segment_start, first_ticks, segment_snapshot
            )
            second_snapshot = _tensor_bytes(owner.state)
            second_rows, second_segments, second_retries = run_segment(
                segment_start + first_ticks,
                segment_ticks - first_ticks,
                second_snapshot,
            )
            combined = [
                first_rows[world] + second_rows[world]
                for world in range(len(first_rows))
            ]
            return (
                combined,
                first_segments + second_segments,
                1 + first_retries + second_retries,
            )
        if next_ticks != (segment_end,) * len(next_ticks):
            raise FullEvolutionWorkhorseProbeError(
                "overflow recovery segment did not publish its exact tick frontier"
            )
        segment = {
            "segment_index": 0,
            "start_next_absolute_tick": segment_start,
            "end_next_absolute_tick": segment_end,
            "tick_count": segment_ticks,
            "event_counts": list(counts),
            "packed_event_count": sum(counts),
        }
        return rows, [segment], 0

    if temporal_k == 1:
        raise FullEvolutionWorkhorseProbeError(
            "one absolute tick exceeded the audited 2*C event page"
        )
    first_ticks = temporal_k // 2
    first_rows, first_segments, first_retries = run_segment(
        start_tick, first_ticks, snapshot_bytes
    )
    second_snapshot = _tensor_bytes(owner.state)
    second_rows, second_segments, second_retries = run_segment(
        start_tick + first_ticks,
        temporal_k - first_ticks,
        second_snapshot,
    )
    rows = [first_rows[world] + second_rows[world] for world in range(len(first_rows))]
    segments = first_segments + second_segments
    retries = 1 + first_retries + second_retries
    for index, segment in enumerate(segments):
        segment["segment_index"] = index
    if loaded.backend.backend_id == BACKEND_PHASE_GRAPH_V2:
        loaded.module.release_graph()
        _prepare_graph_if_needed(loaded, owner, temporal_k, disable_mask)
    return rows, segments, retries


def _dense_event_oracle_v3(
    owner: _OwnedBuffers,
    torch: Any,
    per_world_rows: Sequence[Sequence[bytes]],
    *,
    start_tick: int,
    end_tick: int,
) -> Any | None:
    dense_bytes = (
        owner.state.size(0)
        * (end_tick - start_tick)
        * 2
        * owner.loaded.shape.capacity
        * 48
    )
    if dense_bytes > 128 * 1024 * 1024:
        return None
    dense = torch.zeros(
        (
            owner.state.size(0),
            end_tick - start_tick,
            2 * owner.loaded.shape.capacity,
            48,
        ),
        dtype=torch.uint8,
    )
    for world, rows in enumerate(per_world_rows):
        for raw in rows:
            values = _EVENT_STRUCT.unpack(raw)
            local_tick = int(values[0]) - start_tick
            event_kind = int(values[6])
            slot = int(values[1]) & ((1 << 48) - 1)
            dense_index = (
                slot if event_kind == 1 else owner.loaded.shape.capacity + slot
            )
            dense[world, local_tick, dense_index].copy_(
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            )
    return dense


def _prepare_sparse_ack_payload_v3(
    owner: _OwnedBuffers,
    torch: Any,
    pending: Mapping[str, Any],
) -> dict[str, Any]:
    start_tick = pending["start_next_absolute_tick"]
    requested_end_tick = pending["end_next_absolute_tick"]
    canonicalization_started = time.perf_counter()
    rows, counts, statuses, next_ticks = _drain_sparse_page_v3(
        owner,
        torch,
        start_tick=start_tick,
        end_tick=requested_end_tick,
    )
    page_rows = int(owner.events.shape[1])
    overflow = any(count > page_rows for count in counts)
    if overflow and not any(statuses):
        _restore_owner_state_bytes_v3(owner, torch, pending["pre_state_bytes"])
        rows, physical_segments, recovery_count = _recover_sparse_overflow_v3(
            owner,
            torch,
            start_tick=start_tick,
            temporal_k=pending["temporal_k"],
            disable_mask=pending["disable_mask"],
            snapshot_bytes=pending["pre_state_bytes"],
        )
        counts = tuple(len(world_rows) for world_rows in rows)
        header = (
            _field(owner.state, owner.loaded.layout, "header", torch).detach().cpu()
        )
        statuses = tuple(int(value) for value in header[:, 4].tolist())
        next_ticks = tuple(int(value) for value in header[:, 1].tolist())
    else:
        physical_segments = [
            {
                "segment_index": 0,
                "start_next_absolute_tick": start_tick,
                "end_next_absolute_tick": requested_end_tick,
                "tick_count": requested_end_tick - start_tick,
                "event_counts": list(counts),
                "packed_event_count": sum(counts),
            }
        ]
        recovery_count = 0
    owner.last_event_canonicalization_tail_ms = (
        time.perf_counter() - canonicalization_started
    ) * 1000.0
    owner.last_dense_events_cpu = _dense_event_oracle_v3(
        owner,
        torch,
        rows,
        start_tick=start_tick,
        end_tick=requested_end_tick,
    )
    return {
        "start_tick": start_tick,
        "requested_end_tick": requested_end_tick,
        "counts": counts,
        "statuses": statuses,
        "next_ticks": next_ticks,
        "physical_segments": physical_segments,
        "recovery_count": recovery_count,
        "packed_events": b"".join(row for world_rows in rows for row in world_rows),
        "post_state_sha256": _sha_tensor(owner.state),
    }


def _acknowledge_events(
    owner: _OwnedBuffers,
    torch: Any,
    *,
    sink_root: Path,
    sink_id: str,
    allow_failed_dispatch: bool = False,
    compress_sink_payloads: bool = False,
    retain_sink_payload: bool = True,
) -> dict[str, Any]:
    if owner.pending is None:
        raise FullEvolutionWorkhorseProbeError("no pending event dispatch to drain")
    if compress_sink_payloads:
        raise FullEvolutionWorkhorseProbeError(
            "sparse V3 sink is already packed and does not accept dense compression"
        )
    if type(retain_sink_payload) is not bool:
        raise FullEvolutionWorkhorseProbeError("retain_sink_payload must be boolean")
    pending = owner.pending
    try:
        payload = _prepare_sparse_ack_payload_v3(owner, torch, pending)
    except Exception:
        owner.poisoned_status = (-1,)
        _restore_owner_state_bytes_v3(owner, torch, pending["pre_state_bytes"])
        raise
    start_tick = payload["start_tick"]
    requested_end_tick = payload["requested_end_tick"]
    counts = payload["counts"]
    statuses = payload["statuses"]
    next_ticks = payload["next_ticks"]
    physical_segments = payload["physical_segments"]
    recovery_count = payload["recovery_count"]
    packed_events = payload["packed_events"]
    post_state_sha256 = payload["post_state_sha256"]
    try:
        receipt_path = publish_workhorse_sparse_event_sink_receipt_v3(
            sink_root,
            sink_id=sink_id,
            packed_events=packed_events,
            event_counts=counts,
            physical_segments=physical_segments,
            overflow_recovery_count=recovery_count,
            world_uids=owner.world_uids,
            start_next_absolute_tick=start_tick,
            end_next_absolute_tick=requested_end_tick,
            artifact_identity_sha256=owner.loaded.manifest["identity"][
                "identity_sha256"
            ],
            artifact_sha256=file_sha256(owner.loaded.artifact),
            manifest_sha256=owner.loaded.manifest["manifest_sha256"],
            attempt_authority_binding=owner.attempt_authority_binding,
            rng_keys=owner.rng_keys,
            pre_state_sha256=pending["pre_state_sha256"],
            post_state_sha256=post_state_sha256,
            retain_packed_events=retain_sink_payload,
        )
        ack = validate_workhorse_event_sink_receipt_v2(
            receipt_path,
            expected_artifact_identity_sha256=owner.loaded.manifest["identity"][
                "identity_sha256"
            ],
            expected_end_next_absolute_tick=requested_end_tick,
            expected_attempt_authority_binding_sha256=pending[
                "attempt_authority_binding_sha256"
            ],
            expected_post_state_sha256=post_state_sha256,
        )
    except Exception:
        owner.poisoned_status = (-1,)
        _restore_owner_state_bytes_v3(owner, torch, pending["pre_state_bytes"])
        raise
    owner.last_sink_ack = ack
    if any(statuses) or next_ticks != (requested_end_tick,) * len(next_ticks):
        owner.poisoned_status = statuses
        if not allow_failed_dispatch:
            raise FullEvolutionWorkhorseProbeError(
                "failed dispatch cannot advance the acknowledged owner frontier"
            )
        return ack
    owner.committed_next_tick = requested_end_tick
    owner.pending = None
    return ack


def _owner_cpu_snapshot_v2(owner: _OwnedBuffers) -> dict[str, Any]:
    import torch

    dense_events = owner.last_dense_events_cpu
    if dense_events is None:
        dense_events = torch.zeros(
            (
                owner.state.size(0),
                owner.temporal_k,
                2 * owner.loaded.shape.capacity,
                48,
            ),
            dtype=torch.uint8,
        )
    device_counts = owner.counts.detach().cpu()
    if bool((device_counts < 0).any().item()) or bool(
        (device_counts > (1 << 31) - 1).any().item()
    ):
        raise FullEvolutionWorkhorseProbeError(
            "tiny dense semantic oracle cannot encode its device event counts as i32"
        )
    return {
        "state": owner.state.detach().cpu().clone(),
        "scratch": owner.scratch.detach().cpu().clone(),
        "events": dense_events.clone(),
        "counts": device_counts.to(dtype=torch.int32).clone(),
    }


def _raw_dataset_v2(
    snapshot: Mapping[str, Any], *, start_next_absolute_tick: int
) -> dict[str, Any]:
    events = snapshot["events"].contiguous()
    input_snapshot = snapshot.get("input")
    if type(input_snapshot) is not dict:
        raise FullEvolutionWorkhorseProbeError(
            "raw semantic dataset requires persisted dispatch input bytes"
        )
    input_events = input_snapshot["events"].contiguous()
    if tuple(input_events.shape) != tuple(events.shape):
        raise FullEvolutionWorkhorseProbeError(
            "raw semantic input/output event tensors differ in shape"
        )
    return {
        "B": int(events.shape[0]),
        "temporal_k": int(events.shape[1]),
        "start_next_absolute_tick": start_next_absolute_tick,
        "input_state": input_snapshot["state"].contiguous().numpy().tobytes(),
        "input_scratch": input_snapshot["scratch"].contiguous().numpy().tobytes(),
        "input_events": input_events.numpy().tobytes(),
        "input_counts": input_snapshot["counts"].contiguous().numpy().tobytes(),
        "state": snapshot["state"].contiguous().numpy().tobytes(),
        "scratch": snapshot["scratch"].contiguous().numpy().tobytes(),
        "events": events.numpy().tobytes(),
        "counts": snapshot["counts"].contiguous().numpy().tobytes(),
    }


def _dispatch_snapshot(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    B: int,
    temporal_k: int,
    world_uid_base: int = 90_001,
    start_tick: int = 71,
    rng_key: int = 17,
    fixture: bool = True,
    disable_mask: int = 0,
    sink_root: Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> dict[str, Any]:
    owner = _allocate_owner(torch, loaded, B, temporal_k)
    _initialize_owner(
        loaded,
        owner,
        world_uid_base=world_uid_base,
        start_tick=start_tick,
        rng_key=rng_key,
        fixture=fixture,
    )
    input_snapshot = _owner_cpu_snapshot_v2(owner)
    execution_authority = _authorize_owner_v2(
        loaded,
        owner,
        temporal_k=temporal_k,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
    )
    graph = None
    try:
        graph = _prepare_graph_if_needed(loaded, owner, temporal_k, disable_mask)
        _advance(loaded, owner, temporal_k, disable_mask)
        sink_ack = _acknowledge_events(
            owner,
            torch,
            sink_root=sink_root / f"snapshot-{id(owner)}",
            sink_id="workhorse-semantic-snapshot-v2",
        )
        final_snapshot = _owner_cpu_snapshot_v2(owner)
        final_state = final_snapshot["state"]
        final_scratch = final_snapshot["scratch"]
        final_events = final_snapshot["events"]
        final_counts = final_snapshot["counts"]
    finally:
        try:
            _release_graph(loaded)
        finally:
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            closeout_authority = _closeout_attempt_authority_v2(
                loaded,
                started_binding=execution_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )
    return {
        "state": final_state,
        "scratch": final_scratch,
        "events": final_events,
        "counts": final_counts,
        "input": input_snapshot,
        "graph": graph,
        "event_sink_ack": sink_ack,
        "execution_attempt_authority": execution_authority,
        "closeout_attempt_authority": closeout_authority,
    }


def _tick_partition_snapshot(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    ticks: int,
    world_uid_base: int = 90_001,
    start_tick: int = 71,
    rng_key: int = 17,
    fixture: bool = True,
    disable_mask: int = 0,
    sink_root: Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> dict[str, Any]:
    owner = _allocate_owner(torch, loaded, 1, 1)
    _initialize_owner(
        loaded,
        owner,
        world_uid_base=world_uid_base,
        start_tick=start_tick,
        rng_key=rng_key,
        fixture=fixture,
    )
    initial_snapshot = _owner_cpu_snapshot_v2(owner)
    execution_authority = _authorize_owner_v2(
        loaded,
        owner,
        temporal_k=1,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
        total_ticks=ticks,
    )
    graph = None
    event_parts: list[Any] = []
    count_total = torch.zeros((1,), dtype=torch.int32)
    try:
        graph = _prepare_graph_if_needed(loaded, owner, 1, disable_mask)
        for _ in range(ticks):
            _advance(loaded, owner, 1, disable_mask)
            _acknowledge_events(
                owner,
                torch,
                sink_root=sink_root / f"partition-{id(owner)}",
                sink_id="workhorse-semantic-partition-v2",
            )
            tick_snapshot = _owner_cpu_snapshot_v2(owner)
            event_parts.append(tick_snapshot["events"])
            count_total += tick_snapshot["counts"]
        final_state = owner[0].detach().cpu()
        final_scratch = owner[1].detach().cpu()
    finally:
        try:
            _release_graph(loaded)
        finally:
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            closeout_authority = _closeout_attempt_authority_v2(
                loaded,
                started_binding=execution_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )
    return {
        "state": final_state,
        "scratch": final_scratch,
        "events": torch.cat(event_parts, dim=1),
        "counts": count_total,
        "input": {
            "state": initial_snapshot["state"],
            "scratch": initial_snapshot["scratch"],
            "events": torch.zeros_like(torch.cat(event_parts, dim=1)),
            "counts": initial_snapshot["counts"],
        },
        "graph": graph,
        "execution_attempt_authority": execution_authority,
        "closeout_attempt_authority": closeout_authority,
    }


def _event_checks(
    snapshot: Mapping[str, Any],
    layout: WorkhorseLayoutV2,
    *,
    start_tick: int,
    require_birth_and_death: bool,
    torch: Any,
) -> tuple[dict[str, bool], dict[str, Any]]:
    events = snapshot["events"].contiguous()
    counts = snapshot["counts"].to(torch.int64)
    state = snapshot["state"].contiguous()
    B, temporal_k, rows_per_tick, row_bytes = events.shape
    if row_bytes != 48 or rows_per_tick != 2 * layout.shape.capacity:
        raise FullEvolutionWorkhorseProbeError(
            "event tensor violates exact 2*C*K*48 ABI"
        )
    event_blob = events.numpy().tobytes()
    world_stride = temporal_k * rows_per_tick * 48
    tick_stride = rows_per_tick * 48
    active_rows = death_rows = birth_rows = 0
    inactive_zero = codec_exact = tick_exact = site_exact = identities_exact = True
    full_keys: set[tuple[int, int, int]] = set()
    birth_slots: list[int] = []
    headers = _field(state, layout, "header", torch).to(torch.int64)
    agent_ids = _field(state, layout, "agent_id", torch).to(torch.int64)
    parent_a = _field(state, layout, "parent_a_id", torch).to(torch.int64)
    parent_b = _field(state, layout, "parent_b_id", torch).to(torch.int64)
    positions = _field(state, layout, "position", torch).to(torch.int64)
    per_world = [0 for _ in range(B)]
    for world in range(B):
        world_uid = int(headers[world, 0].item())
        for local_tick in range(temporal_k):
            for row_index in range(rows_per_tick):
                offset = (
                    world * world_stride + local_tick * tick_stride + row_index * 48
                )
                row = event_blob[offset : offset + 48]
                (
                    tick,
                    event_id,
                    agent_id,
                    parent_a_id,
                    parent_b_id,
                    site,
                    kind,
                    status,
                    reserved,
                ) = _EVENT_STRUCT.unpack(row)
                if status == 0:
                    inactive_zero = inactive_zero and row == bytes(48)
                    continue
                active_rows += 1
                per_world[world] += 1
                death_rows += kind == 1
                birth_rows += kind == 2
                slot = row_index if kind == 1 else row_index - layout.shape.capacity
                codec_exact = (
                    codec_exact
                    and status == 1
                    and reserved == 0
                    and (
                        kind in (1, 2)
                        and 0 <= slot < layout.shape.capacity
                        and event_id == (kind << 48) | slot
                    )
                )
                tick_exact = tick_exact and tick == start_tick + local_tick
                site_exact = site_exact and site < layout.shape.sites
                full_keys.add((world_uid, tick, event_id))
                if kind == 2:
                    birth_slots.append(slot)
                    identities_exact = identities_exact and (
                        int(agent_ids[world, slot]) == agent_id
                        and int(parent_a[world, slot]) == parent_a_id
                        and int(parent_b[world, slot]) == parent_b_id
                        and int(positions[world, slot]) == site
                    )
    checks = {
        "event_inactive_rows_zero": inactive_zero,
        "event_count_exact": per_world == [int(value) for value in counts.tolist()],
        "event_id_codec_exact": codec_exact,
        "event_full_key_unique": len(full_keys) == active_rows,
        "event_absolute_ticks_exact": tick_exact,
        "event_u32_sites_in_shape": site_exact,
        "birth_event_state_identity_exact": identities_exact,
        "death_event_present": death_rows > 0 if require_birth_and_death else True,
        "birth_event_present": birth_rows > 0 if require_birth_and_death else True,
    }
    return checks, {
        "active_rows": active_rows,
        "death_rows": death_rows,
        "birth_rows": birth_rows,
        "birth_slots": birth_slots,
    }


def _mutation_and_recombination_checks(
    baseline: Mapping[str, Any],
    mutation_disabled: Mapping[str, Any],
    layout: WorkhorseLayoutV2,
    birth_slots: Sequence[int],
    torch: Any,
) -> tuple[dict[str, bool], dict[str, Any]]:
    capacity = layout.shape.capacity
    ga = _field(baseline["state"], layout, "genome_a", torch).reshape(capacity, 32)
    gb = _field(baseline["state"], layout, "genome_b", torch).reshape(capacity, 32)
    expressed = _field(baseline["state"], layout, "expressed", torch).reshape(
        capacity, 32
    )
    ga_nomut = _field(mutation_disabled["state"], layout, "genome_a", torch).reshape(
        capacity, 32
    )
    gb_nomut = _field(mutation_disabled["state"], layout, "genome_b", torch).reshape(
        capacity, 32
    )
    expressed_nomut = _field(
        mutation_disabled["state"], layout, "expressed", torch
    ).reshape(capacity, 32)
    alive = _field(baseline["state"], layout, "alive", torch).reshape(capacity).bool()
    expected_cache = (
        (ga.to(torch.int16) + gb.to(torch.int16))
        .div(2, rounding_mode="floor")
        .to(torch.uint8)
    )
    mutation_locus_differences = mutation_cache_differences = 0
    mutation_isolated = bool(birth_slots)
    for child in birth_slots:
        genome_diff = ga[child] != ga_nomut[child]
        cache_diff = expressed[child] != expressed_nomut[child]
        mutation_locus_differences += int(genome_diff.sum().item())
        mutation_cache_differences += int(cache_diff.sum().item())
        mutation_isolated = mutation_isolated and int(genome_diff.sum().item()) == 1
        mutation_isolated = mutation_isolated and bool(
            torch.equal(gb[child], gb_nomut[child])
            and torch.equal(cache_diff, genome_diff)
        )

    ids = _field(mutation_disabled["state"], layout, "agent_id", torch).reshape(
        capacity
    )
    parent_a_ids = _field(
        mutation_disabled["state"], layout, "parent_a_id", torch
    ).reshape(capacity)
    parent_b_ids = _field(
        mutation_disabled["state"], layout, "parent_b_id", torch
    ).reshape(capacity)
    donor_full_vector = bool(birth_slots)
    donor_a_both = donor_b_both = high_locus_witness = False
    for child in birth_slots:
        pa_matches = (ids == parent_a_ids[child]).nonzero().flatten()
        pb_matches = (ids == parent_b_ids[child]).nonzero().flatten()
        if pa_matches.numel() != 1 or pb_matches.numel() != 1:
            donor_full_vector = False
            continue
        pa = int(pa_matches[0])
        pb = int(pb_matches[0])
        child_a, child_b = ga_nomut[child], gb_nomut[child]
        pa_a, pa_b = ga_nomut[pa], gb_nomut[pa]
        pb_a, pb_b = ga_nomut[pb], gb_nomut[pb]
        donor_full_vector = donor_full_vector and bool(
            ((child_a == pa_a) | (child_a == pa_b)).all().item()
            and ((child_b == pb_a) | (child_b == pb_b)).all().item()
        )
        differentiable_a = pa_a != pa_b
        differentiable_b = pb_a != pb_b
        donor_a_both = donor_a_both or bool(
            (differentiable_a & (child_a == pa_a)).any().item()
            and (differentiable_a & (child_a == pa_b)).any().item()
        )
        donor_b_both = donor_b_both or bool(
            (differentiable_b & (child_b == pb_a)).any().item()
            and (differentiable_b & (child_b == pb_b)).any().item()
        )
        high_locus_witness = high_locus_witness or bool(
            ((child_a[16:] == pa_a[16:]) | (child_a[16:] == pa_b[16:])).all().item()
            and ((child_b[16:] == pb_a[16:]) | (child_b[16:] == pb_b[16:])).all().item()
        )
    checks = {
        "expressed_cache_matches_diploid_genome": bool(
            torch.equal(expressed[alive], expected_cache[alive])
        ),
        "mutation_isolated_to_one_child_locus_and_cache": mutation_isolated,
        "mutation_disabled_genome_b_unchanged": bool(torch.equal(gb, gb_nomut)),
        "recombination_full_32_locus_vector_valid": donor_full_vector,
        "recombination_parent_a_both_homologs": donor_a_both,
        "recombination_parent_b_both_homologs": donor_b_both,
        "recombination_high_loci_witness": high_locus_witness,
    }
    return checks, {
        "mutation_locus_differences": mutation_locus_differences,
        "mutation_cache_differences": mutation_cache_differences,
        "birth_slots": list(birth_slots),
    }


def _causal_checks(
    candidate: _LoadedArtifact,
    torch: Any,
    *,
    sink_root: Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, dict[str, Any]]]:
    authority_kwargs = {
        "attempt_authority_validator": attempt_authority_validator,
        "device_uuid": device_uuid,
    }
    baseline = _tick_partition_snapshot(
        candidate,
        torch,
        ticks=1,
        sink_root=sink_root / "baseline",
        **authority_kwargs,
    )
    disabled = {
        "patch": _tick_partition_snapshot(
            candidate,
            torch,
            ticks=1,
            disable_mask=DISABLE_PATCH,
            sink_root=sink_root / "disable-patch",
            **authority_kwargs,
        ),
        "wdl": _tick_partition_snapshot(
            candidate,
            torch,
            ticks=1,
            disable_mask=DISABLE_WDL,
            sink_root=sink_root / "disable-wdl",
            **authority_kwargs,
        ),
        "jungle": _tick_partition_snapshot(
            candidate,
            torch,
            ticks=1,
            disable_mask=DISABLE_JUNGLE,
            sink_root=sink_root / "disable-jungle",
            **authority_kwargs,
        ),
        "eed": _tick_partition_snapshot(
            candidate,
            torch,
            ticks=1,
            disable_mask=DISABLE_EED,
            sink_root=sink_root / "disable-eed",
            **authority_kwargs,
        ),
        "birth": _tick_partition_snapshot(
            candidate,
            torch,
            ticks=1,
            disable_mask=DISABLE_BIRTH,
            sink_root=sink_root / "disable-birth",
            **authority_kwargs,
        ),
        "mutation": _tick_partition_snapshot(
            candidate,
            torch,
            ticks=1,
            disable_mask=DISABLE_MUTATION,
            sink_root=sink_root / "disable-mutation",
            **authority_kwargs,
        ),
        "recombination": _tick_partition_snapshot(
            candidate,
            torch,
            ticks=1,
            disable_mask=DISABLE_MUTATION | DISABLE_RECOMBINATION,
            sink_root=sink_root / "disable-recombination",
            **authority_kwargs,
        ),
    }
    layout = candidate.layout

    def changed(name: str, other: Mapping[str, Any]) -> bool:
        return not torch.equal(
            _field(baseline["state"], layout, name, torch),
            _field(other["state"], layout, name, torch),
        )

    event_checks, event_witness = _event_checks(
        baseline,
        layout,
        start_tick=71,
        require_birth_and_death=True,
        torch=torch,
    )
    mutation_checks, genome_witness = _mutation_and_recombination_checks(
        baseline,
        disabled["mutation"],
        layout,
        event_witness["birth_slots"],
        torch,
    )
    eed_enabled = _tick_partition_snapshot(
        candidate,
        torch,
        ticks=2,
        disable_mask=DISABLE_MUTATION,
        sink_root=sink_root / "eed-enabled",
        **authority_kwargs,
    )
    eed_disabled = _tick_partition_snapshot(
        candidate,
        torch,
        ticks=2,
        disable_mask=DISABLE_MUTATION | DISABLE_EED,
        sink_root=sink_root / "eed-disabled",
        **authority_kwargs,
    )
    child_slots = event_witness["birth_slots"]
    eed_heritable = bool(child_slots)
    if child_slots:
        energy_enabled = _field(eed_enabled["state"], layout, "energy", torch).reshape(
            -1
        )
        energy_disabled = _field(
            eed_disabled["state"], layout, "energy", torch
        ).reshape(-1)
        expressed = _field(eed_enabled["state"], layout, "expressed", torch).reshape(
            layout.shape.capacity, 32
        )
        eed_heritable = bool(
            (expressed[child_slots, 31] > 0).all().item()
            and (energy_enabled[child_slots] != energy_disabled[child_slots])
            .any()
            .item()
        )

    owner = _allocate_owner(torch, candidate, 1, 1)
    _initialize_owner(
        candidate,
        owner,
        world_uid_base=91_000,
        start_tick=0,
        rng_key=31,
        fixture=False,
    )
    wdl_execution_authority = _authorize_owner_v2(
        candidate,
        owner,
        temporal_k=1,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
        total_ticks=3,
    )
    wdl_active_sums: list[int] = []
    wdl_snapshots: list[dict[str, Any]] = []
    try:
        _prepare_graph_if_needed(candidate, owner, 1, 0)
        for _ in range(3):
            wdl_input = _owner_cpu_snapshot_v2(owner)
            _advance(candidate, owner, 1, 0)
            _acknowledge_events(
                owner,
                torch,
                sink_root=sink_root / f"wdl-{id(owner)}",
                sink_id="workhorse-wdl-delay-v2",
            )
            wdl_active_sums.append(
                int(_field(owner[0], layout, "wdl_active", torch).sum().item())
            )
            wdl_snapshot = _owner_cpu_snapshot_v2(owner)
            wdl_snapshot["input"] = wdl_input
            wdl_snapshots.append(wdl_snapshot)
    finally:
        try:
            _release_graph(candidate)
        finally:
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            wdl_closeout_authority = _closeout_attempt_authority_v2(
                candidate,
                started_binding=wdl_execution_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )

    checks = {
        **event_checks,
        **mutation_checks,
        "patch_paired_state_effect": changed("patch_field", disabled["patch"]),
        "patch_paired_downstream_resource_or_agent_effect": (
            changed("resource", disabled["patch"])
            or changed("energy", disabled["patch"])
        ),
        "wdl_paired_physiology_effect": changed("health", disabled["wdl"]),
        "jungle_paired_physiology_effect": changed("health", disabled["jungle"]),
        "eed_paired_downstream_energy_effect": changed("energy", disabled["eed"]),
        "ordinary_birth_paired_alive_effect": changed("alive", disabled["birth"]),
        "recombination_paired_genome_effect": not torch.equal(
            _field(disabled["mutation"]["state"], layout, "genome_a", torch),
            _field(disabled["recombination"]["state"], layout, "genome_a", torch),
        ),
        "eed_is_heritable_and_changes_child_outcome": eed_heritable,
        "wdl_new_emission_delay_is_exactly_two_ticks": (
            wdl_active_sums[0] == 0
            and wdl_active_sums[1] == 0
            and wdl_active_sums[2] > 0
        ),
    }
    downstream_patch_hashes = {
        "baseline_resource_sha256": _sha_tensor(
            _field(baseline["state"], layout, "resource", torch)
        ),
        "patch_disabled_resource_sha256": _sha_tensor(
            _field(disabled["patch"]["state"], layout, "resource", torch)
        ),
        "baseline_energy_sha256": _sha_tensor(
            _field(baseline["state"], layout, "energy", torch)
        ),
        "patch_disabled_energy_sha256": _sha_tensor(
            _field(disabled["patch"]["state"], layout, "energy", torch)
        ),
    }
    witness = {
        "events": event_witness,
        "genome": genome_witness,
        "wdl_active_sum_after_ticks_1_2_3": wdl_active_sums,
        "patch_downstream_hashes": downstream_patch_hashes,
        "wdl_execution_attempt_authority": wdl_execution_authority,
        "wdl_closeout_attempt_authority": wdl_closeout_authority,
        "attempt_lifecycles": [
            _attempt_lifecycle_row_v2(
                "causal_baseline",
                baseline["execution_attempt_authority"],
                baseline["closeout_attempt_authority"],
            ),
            *[
                _attempt_lifecycle_row_v2(
                    f"causal_{name}_disabled",
                    snapshot["execution_attempt_authority"],
                    snapshot["closeout_attempt_authority"],
                )
                for name, snapshot in disabled.items()
            ],
            _attempt_lifecycle_row_v2(
                "eed_enabled",
                eed_enabled["execution_attempt_authority"],
                eed_enabled["closeout_attempt_authority"],
            ),
            _attempt_lifecycle_row_v2(
                "eed_disabled",
                eed_disabled["execution_attempt_authority"],
                eed_disabled["closeout_attempt_authority"],
            ),
            _attempt_lifecycle_row_v2(
                "wdl_ticks_1_2_3",
                wdl_execution_authority,
                wdl_closeout_authority,
            ),
        ],
    }
    raw_datasets = {
        "causal_baseline": _raw_dataset_v2(baseline, start_next_absolute_tick=71),
        **{
            f"causal_{name}_disabled": _raw_dataset_v2(
                snapshot, start_next_absolute_tick=71
            )
            for name, snapshot in disabled.items()
        },
        "eed_enabled": _raw_dataset_v2(eed_enabled, start_next_absolute_tick=71),
        "eed_disabled": _raw_dataset_v2(eed_disabled, start_next_absolute_tick=71),
        **{
            f"wdl_tick{index}": _raw_dataset_v2(
                snapshot, start_next_absolute_tick=index - 1
            )
            for index, snapshot in enumerate(wdl_snapshots, start=1)
        },
    }
    return checks, witness, raw_datasets


def _overflow_parity_check(
    reference: _LoadedArtifact,
    candidate: _LoadedArtifact,
    torch: Any,
    *,
    sink_root: Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, dict[str, Any]]]:
    def execute_overflow(
        loaded: _LoadedArtifact, *, label: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        owner = _allocate_owner(torch, loaded, 1, 1)
        _initialize_owner(
            loaded,
            owner,
            world_uid_base=95_000,
            start_tick=9,
            rng_key=43,
            fixture=False,
        )
        loaded.module.force_birth_overflow_fixture(owner[0], owner[1])
        fixture_input = _owner_cpu_snapshot_v2(owner)
        execution_authority = _authorize_owner_v2(
            loaded,
            owner,
            temporal_k=1,
            phase="EXECUTION_ROW",
            validator=attempt_authority_validator,
            device_uuid=device_uuid,
        )
        try:
            _prepare_graph_if_needed(loaded, owner, 1, 0)
            _advance(loaded, owner, 1, 0)
            _acknowledge_events(
                owner,
                torch,
                sink_root=sink_root / label,
                sink_id=f"workhorse-overflow-{label}-v2",
            )
            snapshot = _owner_cpu_snapshot_v2(owner)
            snapshot["input"] = fixture_input
        finally:
            try:
                _release_graph(loaded)
            finally:
                owner = None
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                closeout_authority = _closeout_attempt_authority_v2(
                    loaded,
                    started_binding=execution_authority,
                    validator=attempt_authority_validator,
                    device_uuid=device_uuid,
                )
        return snapshot, execution_authority, closeout_authority

    reference_snapshot, reference_authority, reference_closeout = execute_overflow(
        reference, label="reference"
    )
    candidate_snapshot, candidate_authority, candidate_closeout = execute_overflow(
        candidate, label="candidate"
    )
    ref_state = reference_snapshot["state"]
    ref_scratch = reference_snapshot["scratch"]
    ref_events = reference_snapshot["events"]
    ref_counts = reference_snapshot["counts"]
    cand_state = candidate_snapshot["state"]
    cand_scratch = candidate_snapshot["scratch"]
    cand_events = candidate_snapshot["events"]
    cand_counts = candidate_snapshot["counts"]
    header = _field(cand_state, candidate.layout, "header", torch)
    birth_count = int(
        (cand_events[:, :, candidate.shape.capacity :, 46] == 1).sum().item()
    )
    checks = {
        "overflow_reference_candidate_state_parity": bool(
            torch.equal(ref_state, cand_state)
        ),
        "overflow_reference_candidate_scratch_parity": bool(
            torch.equal(ref_scratch, cand_scratch)
        ),
        "overflow_reference_candidate_event_parity": bool(
            torch.equal(ref_events, cand_events)
            and torch.equal(ref_counts, cand_counts)
        ),
        "overflow_truncates_to_single_free_slot": birth_count == 1,
        "overflow_is_not_status_error": int(header[0, 4]) == 0,
        "overflow_publishes_one_tick": int(header[0, 1]) == 10,
    }
    return (
        checks,
        {
            "birth_rows": birth_count,
            "event_count": int(cand_counts[0]),
            "reference_execution_attempt_authority": reference_authority,
            "reference_closeout_attempt_authority": reference_closeout,
            "candidate_execution_attempt_authority": candidate_authority,
            "candidate_closeout_attempt_authority": candidate_closeout,
            "attempt_lifecycles": [
                _attempt_lifecycle_row_v2(
                    "overflow_reference", reference_authority, reference_closeout
                ),
                _attempt_lifecycle_row_v2(
                    "overflow_candidate", candidate_authority, candidate_closeout
                ),
            ],
        },
        {
            "overflow_reference": _raw_dataset_v2(
                reference_snapshot, start_next_absolute_tick=9
            ),
            "overflow_candidate": _raw_dataset_v2(
                candidate_snapshot, start_next_absolute_tick=9
            ),
        },
    )


def _middle_survivor_death_pairing_check(
    reference: _LoadedArtifact,
    candidate: _LoadedArtifact,
    torch: Any,
    *,
    sink_root: Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, dict[str, Any]]]:
    """Prove births pair the post-death compacted UID order, not old ordinals."""

    def execute(
        loaded: _LoadedArtifact, *, label: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        owner = _allocate_owner(torch, loaded, 1, 1)
        _initialize_owner(
            loaded,
            owner,
            world_uid_base=95_500,
            start_tick=31,
            rng_key=45,
            fixture=False,
        )
        loaded.module.force_middle_survivor_death_fixture(owner.state, owner.scratch)
        fixture_input = _owner_cpu_snapshot_v2(owner)
        execution_authority = _authorize_owner_v2(
            loaded,
            owner,
            temporal_k=1,
            phase="EXECUTION_ROW",
            validator=attempt_authority_validator,
            device_uuid=device_uuid,
        )
        try:
            _prepare_graph_if_needed(loaded, owner, 1, 0)
            _advance(loaded, owner, 1, 0)
            _acknowledge_events(
                owner,
                torch,
                sink_root=sink_root / label,
                sink_id=f"workhorse-middle-survivor-{label}-v2",
            )
            snapshot = _owner_cpu_snapshot_v2(owner)
            snapshot["input"] = fixture_input
        finally:
            try:
                _release_graph(loaded)
            finally:
                owner = None
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                closeout_authority = _closeout_attempt_authority_v2(
                    loaded,
                    started_binding=execution_authority,
                    validator=attempt_authority_validator,
                    device_uuid=device_uuid,
                )
        return snapshot, execution_authority, closeout_authority

    reference_snapshot, reference_authority, reference_closeout = execute(
        reference, label="reference"
    )
    candidate_snapshot, candidate_authority, candidate_closeout = execute(
        candidate, label="candidate"
    )
    layout = candidate.layout
    input_state = candidate_snapshot["input"]["state"]
    input_header = _field(input_state, layout, "header", torch)[0]
    input_order = _field(input_state, layout, "active_slots_by_uid", torch)[
        0, : int(input_header[3])
    ].to(torch.int64)
    input_ids = _field(input_state, layout, "agent_id", torch)[0].to(torch.int64)
    input_health = _field(input_state, layout, "health", torch)[0]
    input_energy = _field(input_state, layout, "energy", torch)[0]
    output_state = candidate_snapshot["state"]
    output_header = _field(output_state, layout, "header", torch)[0]
    output_order = _field(output_state, layout, "active_slots_by_uid", torch)[
        0, : int(output_header[3])
    ].to(torch.int64)
    output_ids = _field(output_state, layout, "agent_id", torch)[0].to(torch.int64)
    rows = []
    event_blob = candidate_snapshot["events"].contiguous().numpy().tobytes()
    event_struct = struct.Struct("<QQQQQIHBB")
    for row_index in range(2 * layout.shape.capacity):
        row = event_struct.unpack_from(event_blob, row_index * 48)
        if row[7]:
            rows.append(row)
    deaths = [row for row in rows if row[6] == 1]
    births = [row for row in rows if row[6] == 2]
    expected_survivor_ids = input_ids.index_select(0, input_order[[0, 2, 3]])
    expected_parent_ids = (
        int(expected_survivor_ids[0]),
        int(expected_survivor_ids[1]),
    )
    correct_pairing = bool(
        len(input_order) == 4
        and int(input_health[input_order[1]]) == 0
        and int(input_energy[input_order[3]]) < 256
        and len(deaths) == 1
        and deaths[0][2] == int(input_ids[input_order[1]])
        and len(births) == 1
        and (births[0][3], births[0][4]) == expected_parent_ids
        and int(candidate_snapshot["counts"][0]) == 2
        and torch.equal(
            output_ids.index_select(0, output_order[:3]), expected_survivor_ids
        )
        and int(output_ids[output_order[-1]]) == births[0][2]
    )
    parity = all(
        torch.equal(reference_snapshot[component], candidate_snapshot[component])
        and torch.equal(
            reference_snapshot["input"][component],
            candidate_snapshot["input"][component],
        )
        for component in ("state", "scratch", "events", "counts")
    )
    return (
        {
            "middle_survivor_reference_candidate_byte_parity": parity,
            "birth_pairs_adjacent_compacted_survivor_uid_order": correct_pairing,
        },
        {
            "input_uid_order": [int(input_ids[slot]) for slot in input_order],
            "post_death_survivor_uid_order": [
                int(value) for value in expected_survivor_ids
            ],
            "observed_birth_parent_uids": (
                [int(births[0][3]), int(births[0][4])] if births else []
            ),
            "reference_execution_attempt_authority": reference_authority,
            "reference_closeout_attempt_authority": reference_closeout,
            "candidate_execution_attempt_authority": candidate_authority,
            "candidate_closeout_attempt_authority": candidate_closeout,
            "attempt_lifecycles": [
                _attempt_lifecycle_row_v2(
                    "middle_survivor_reference",
                    reference_authority,
                    reference_closeout,
                ),
                _attempt_lifecycle_row_v2(
                    "middle_survivor_candidate",
                    candidate_authority,
                    candidate_closeout,
                ),
            ],
        },
        {
            "middle_survivor_reference": _raw_dataset_v2(
                reference_snapshot, start_next_absolute_tick=31
            ),
            "middle_survivor_candidate": _raw_dataset_v2(
                candidate_snapshot, start_next_absolute_tick=31
            ),
        },
    )


def _transaction_fail_stop_check(
    candidate: _LoadedArtifact,
    torch: Any,
    *,
    sink_root: Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, dict[str, Any]]]:
    owner = _allocate_owner(torch, candidate, 1, 2)
    _initialize_owner(
        candidate,
        owner,
        world_uid_base=96_000,
        start_tick=13,
        rng_key=47,
        fixture=False,
    )
    candidate.module.force_transaction_error_fixture(owner[0], owner[1])
    transaction_input = _owner_cpu_snapshot_v2(owner)
    execution_authority = _authorize_owner_v2(
        candidate,
        owner,
        temporal_k=2,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
    )
    try:
        _prepare_graph_if_needed(candidate, owner, 2, 0)
        _advance(candidate, owner, 2, 0)
        failed_ack = _acknowledge_events(
            owner,
            torch,
            sink_root=sink_root,
            sink_id="workhorse-failed-dispatch-forensic-sink-v2",
            allow_failed_dispatch=True,
        )
        header = _field(owner[0], candidate.layout, "header", torch)
        status = int(header[0, 4])
        next_tick = int(header[0, 1])
        dense_failed_events = owner.last_dense_events_cpu
        if dense_failed_events is None:
            raise FullEvolutionWorkhorseProbeError(
                "tiny failed dispatch did not materialize its dense event oracle"
            )
        remaining_ticks_fail_stop = (
            int(torch.count_nonzero(dense_failed_events[:, 1:])) == 0
        )
        pending_preserved = owner.pending is not None
        transaction_snapshot = _owner_cpu_snapshot_v2(owner)
        transaction_snapshot["input"] = transaction_input
    finally:
        try:
            _release_graph(candidate)
        finally:
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            closeout_authority = _closeout_attempt_authority_v2(
                candidate,
                started_binding=execution_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )
    checks = {
        "first_error_status_is_deterministic": status == 1,
        "failed_dispatch_does_not_publish_frontier": next_tick == 13,
        "remaining_temporal_ticks_fail_stop": remaining_ticks_fail_stop,
        "failed_event_buffer_cannot_be_reused": pending_preserved,
    }
    return (
        checks,
        {
            "status": status,
            "next_absolute_tick": next_tick,
            "forensic_sink_ack": failed_ack,
            "execution_attempt_authority": execution_authority,
            "closeout_attempt_authority": closeout_authority,
            "attempt_lifecycles": [
                _attempt_lifecycle_row_v2(
                    "transaction_fail_stop",
                    execution_authority,
                    closeout_authority,
                )
            ],
        },
        {
            "transaction_after": _raw_dataset_v2(
                transaction_snapshot, start_next_absolute_tick=13
            )
        },
    )


def _restore_and_checkpoint_check(
    candidate: _LoadedArtifact,
    torch: Any,
    *,
    sink_root: Path,
    checkpoint_root: Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, dict[str, Any]]]:
    layout = candidate.layout
    owner = _allocate_owner(torch, candidate, 1, 1)
    restore_nonce = id(owner)
    _initialize_owner(
        candidate,
        owner,
        world_uid_base=97_000,
        start_tick=21,
        rng_key=53,
        fixture=True,
    )
    checkpoint_dispatch_input = _owner_cpu_snapshot_v2(owner)
    checkpoint_attempt_authority = _authorize_owner_v2(
        candidate,
        owner,
        temporal_k=1,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
    )
    try:
        _prepare_graph_if_needed(candidate, owner, 1, 0)
        _advance(candidate, owner, 1, 0)
        event_sink_ack = _acknowledge_events(
            owner,
            torch,
            sink_root=sink_root / "checkpoint-dispatch",
            sink_id="workhorse-checkpoint-events-v2",
        )
        checkpoint_state = owner.state.detach().cpu().clone()
        checkpoint_snapshot = _owner_cpu_snapshot_v2(owner)
        checkpoint_snapshot["input"] = checkpoint_dispatch_input
        checkpoint_state_bytes = checkpoint_state.contiguous().numpy().tobytes()
        header = _field(checkpoint_state, layout, "header", torch)
        next_tick = int(header[0, 1])
        active_count = int(header[0, 3])
        order = _field(checkpoint_state, layout, "active_slots_by_uid", torch)[
            0, :active_count
        ].to(torch.int64)
        ids = _field(checkpoint_state, layout, "agent_id", torch)[0].to(torch.int64)
        ordered_ids = ids.index_select(0, order)
        dense_checkpoint_events = owner.last_dense_events_cpu
        if dense_checkpoint_events is None:
            raise FullEvolutionWorkhorseProbeError(
                "tiny checkpoint dispatch did not materialize its dense event oracle"
            )
        birth_slots = (
            (dense_checkpoint_events[0, 0, layout.shape.capacity :, 46] == 1)
            .nonzero()
            .flatten()
        )
        low_slot_reuse = bool(
            birth_slots.numel() > 0
            and (birth_slots < layout.shape.initial_population).any().item()
            and torch.equal(
                order[-birth_slots.numel() :].cpu(), birth_slots.to(torch.int64)
            )
        )
        restored_world_uids = owner.world_uids
        restored_rng_keys = owner.rng_keys
    finally:
        try:
            _release_graph(candidate)
        finally:
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            checkpoint_closeout_authority = _closeout_attempt_authority_v2(
                candidate,
                started_binding=checkpoint_attempt_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )

    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=candidate.shape.shape_id,
        backend_id=candidate.backend.backend_id,
        B=1,
        temporal_k=1,
        total_ticks=1,
        start_next_tick=21,
    )
    checkpoint_bundle = write_workhorse_checkpoint_bundle_v2(
        checkpoint_root / f"bundle-{restore_nonce}",
        contract=contract,
        checkpoint_at_tick=next_tick,
        state_bytes=checkpoint_state_bytes,
        artifact_path=candidate.artifact,
        event_sink_receipt_path=event_sink_ack["receipt_path"],
        attempt_authority_binding=checkpoint_attempt_authority,
    )
    checkpoint, reopened_state = reopen_workhorse_checkpoint_bundle_v2(
        checkpoint_bundle
    )
    source_checkpoint_identity_sha256 = checkpoint["checkpoint_identity"][
        "checkpoint_identity_sha256"
    ]

    reconstruction_owner = _allocate_owner(torch, candidate, 1, 1)
    reconstruction_owner.world_uids = restored_world_uids
    reconstruction_owner.rng_keys = restored_rng_keys
    reconstruction_owner.committed_next_tick = next_tick
    reconstruction_execution_authority = _authorize_owner_v2(
        candidate,
        reconstruction_owner,
        temporal_k=1,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
        source_checkpoint_identity_sha256=source_checkpoint_identity_sha256,
    )
    try:
        reconstruction_owner.state.copy_(checkpoint_state)
        reconstruction_owner.scratch.zero_()
        occupancy_field = layout.field("occupancy")
        reconstruction_owner.state[
            :,
            occupancy_field.offset : occupancy_field.offset + occupancy_field.nbytes,
        ].zero_()
        reconstruction_owner.scratch.fill_(0xA5)
        corrupted_snapshot = _owner_cpu_snapshot_v2(reconstruction_owner)
        candidate.module.restore_derived(
            reconstruction_owner.state, reconstruction_owner.scratch
        )
        restored_exact = bool(
            torch.equal(reconstruction_owner.state.detach().cpu(), checkpoint_state)
        )
        reconstructed_snapshot = _owner_cpu_snapshot_v2(reconstruction_owner)
        reconstructed_snapshot["input"] = corrupted_snapshot
    finally:
        reconstruction_owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        reconstruction_closeout_authority = _closeout_attempt_authority_v2(
            candidate,
            started_binding=reconstruction_execution_authority,
            validator=attempt_authority_validator,
            device_uuid=device_uuid,
        )

    restored_owner = _allocate_owner(torch, candidate, 1, 1)
    restored_owner.world_uids = restored_world_uids
    restored_owner.rng_keys = restored_rng_keys
    restored_owner.committed_next_tick = next_tick
    restored_execution_authority = _authorize_owner_v2(
        candidate,
        restored_owner,
        temporal_k=1,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
        source_checkpoint_identity_sha256=source_checkpoint_identity_sha256,
    )
    try:
        restored_owner.state.copy_(reconstructed_snapshot["state"])
        restored_owner.scratch.copy_(reconstructed_snapshot["scratch"])
        restored_continuation_input = _owner_cpu_snapshot_v2(restored_owner)
        _prepare_graph_if_needed(candidate, restored_owner, 1, 0)
        _advance(candidate, restored_owner, 1, 0)
        _acknowledge_events(
            restored_owner,
            torch,
            sink_root=sink_root / "restored-continuation",
            sink_id="workhorse-restored-continuation-v2",
        )
        restored_continuation_snapshot = _owner_cpu_snapshot_v2(restored_owner)
        restored_continuation_snapshot["input"] = restored_continuation_input
    finally:
        try:
            _release_graph(candidate)
        finally:
            restored_owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            restored_closeout_authority = _closeout_attempt_authority_v2(
                candidate,
                started_binding=restored_execution_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )

    clean_owner = _allocate_owner(torch, candidate, 1, 1)
    clean_owner.world_uids = restored_world_uids
    clean_owner.rng_keys = restored_rng_keys
    clean_owner.committed_next_tick = next_tick
    clean_execution_authority = _authorize_owner_v2(
        candidate,
        clean_owner,
        temporal_k=1,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
        source_checkpoint_identity_sha256=source_checkpoint_identity_sha256,
    )
    try:
        clean_owner.state.copy_(checkpoint_state)
        clean_owner.scratch.zero_()
        candidate.module.restore_derived(clean_owner.state, clean_owner.scratch)
        _prepare_graph_if_needed(candidate, clean_owner, 1, 0)
        _advance(candidate, clean_owner, 1, 0)
        _acknowledge_events(
            clean_owner,
            torch,
            sink_root=sink_root / "clean-continuation",
            sink_id="workhorse-clean-continuation-v2",
        )
        clean_continuation_snapshot = _owner_cpu_snapshot_v2(clean_owner)
    finally:
        try:
            _release_graph(candidate)
        finally:
            clean_owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            clean_closeout_authority = _closeout_attempt_authority_v2(
                candidate,
                started_binding=clean_execution_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )

    worker_output = checkpoint_root / f"fresh-worker-{restore_nonce}.json"
    worker_sink = sink_root / f"fresh-worker-{restore_nonce}"
    worker_authority_request = (
        checkpoint_root / f"fresh-worker-{restore_nonce}.request.json"
    )
    worker_authority_binding = (
        checkpoint_root / f"fresh-worker-{restore_nonce}.authority.json"
    )
    worker_closeout_request = (
        checkpoint_root / f"fresh-worker-{restore_nonce}.closeout-request.json"
    )
    worker_closeout_binding = (
        checkpoint_root / f"fresh-worker-{restore_nonce}.closeout-authority.json"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "engine.perf.cuda_full_evolution_v2.workhorse.probe",
            "--restore-worker",
            str(checkpoint_bundle),
            "--worker-output",
            str(worker_output),
            "--worker-sink-root",
            str(worker_sink),
            "--worker-authority-request",
            str(worker_authority_request),
            "--worker-authority-binding",
            str(worker_authority_binding),
            "--worker-closeout-request",
            str(worker_closeout_request),
            "--worker-closeout-binding",
            str(worker_closeout_binding),
        ],
        cwd=Path(__file__).resolve().parents[4],
    )
    deadline = time.monotonic() + 120.0
    while not worker_authority_request.is_file():
        if process.poll() is not None:
            raise FullEvolutionWorkhorseProbeError(
                "fresh restore worker exited before requesting Attempt authority"
            )
        if time.monotonic() >= deadline:
            process.kill()
            raise FullEvolutionWorkhorseProbeError(
                "fresh restore worker authority handshake timed out"
            )
        time.sleep(0.05)
    worker_request = json.loads(worker_authority_request.read_text(encoding="utf-8"))
    try:
        worker_binding = resolve_injected_workhorse_attempt_authority_v2(
            attempt_authority_validator,
            request=worker_request,
            enforce_current_process=False,
        )
    except Exception:
        process.kill()
        process.wait()
        raise
    _write_sealed_json_v2(worker_binding, worker_authority_binding)
    while not worker_closeout_request.is_file():
        if process.poll() is not None:
            raise FullEvolutionWorkhorseProbeError(
                "fresh restore worker exited before requesting Attempt closeout"
            )
        if time.monotonic() >= deadline:
            process.kill()
            raise FullEvolutionWorkhorseProbeError(
                "fresh restore worker closeout handshake timed out"
            )
        time.sleep(0.05)
    worker_closeout_request_value = json.loads(
        worker_closeout_request.read_text(encoding="utf-8")
    )
    try:
        worker_closeout_authority = resolve_injected_workhorse_attempt_authority_v2(
            attempt_authority_validator,
            request=worker_closeout_request_value,
            enforce_current_process=False,
        )
    except Exception:
        process.kill()
        process.wait()
        raise
    if (
        worker_closeout_authority["attempt_lineage_sha256"]
        != worker_binding["attempt_lineage_sha256"]
    ):
        process.kill()
        process.wait()
        raise FullEvolutionWorkhorseProbeError(
            "fresh restore worker closeout changed Attempt lineage"
        )
    _write_sealed_json_v2(worker_closeout_authority, worker_closeout_binding)
    try:
        return_code = process.wait(timeout=max(1.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise FullEvolutionWorkhorseProbeError(
            "fresh restore worker continuation timed out"
        ) from exc
    if return_code:
        raise FullEvolutionWorkhorseProbeError(
            f"fresh restore worker failed with exit code {return_code}"
        )
    worker_path = worker_output.resolve()
    worker_receipt = json.loads(worker_path.read_text(encoding="utf-8"))
    worker_body = {
        key: value
        for key, value in worker_receipt.items()
        if key != "worker_receipt_sha256"
    }
    worker_seal_exact = worker_receipt.get("worker_receipt_sha256") == canonical_sha256(
        worker_body
    )
    same_process_continuation = {
        "state_sha256": _sha_tensor(clean_continuation_snapshot["state"]),
        "scratch_sha256": _sha_tensor(clean_continuation_snapshot["scratch"]),
        "events_sha256": _sha_tensor(clean_continuation_snapshot["events"]),
        "counts_sha256": _sha_tensor(clean_continuation_snapshot["counts"]),
        "next_absolute_tick": next_tick + 1,
    }

    bad_owner = _allocate_owner(torch, candidate, 1, 1)
    bad_owner.world_uids = restored_world_uids
    bad_owner.rng_keys = restored_rng_keys
    bad_owner.committed_next_tick = next_tick
    invalid_execution_authority = _authorize_owner_v2(
        candidate,
        bad_owner,
        temporal_k=1,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
        source_checkpoint_identity_sha256=source_checkpoint_identity_sha256,
    )
    try:
        bad_state = bytearray(checkpoint_state_bytes)
        order_field = layout.field("active_slots_by_uid")
        struct.pack_into("<I", bad_state, order_field.offset + 4, 0)
        bad_owner.state.copy_(
            torch.frombuffer(bad_state, dtype=torch.uint8).reshape_as(bad_owner.state)
        )
        bad_owner.scratch.zero_()
        restore_rejected = False
        try:
            candidate.module.restore_derived(bad_owner.state, bad_owner.scratch)
        except RuntimeError:
            restore_rejected = True
        bad_header = _field(bad_owner.state, layout, "header", torch).detach().cpu()
        preserved_status = int(bad_header[0, 4])
        bad_frontier = int(bad_header[0, 1])
        invalid_before_snapshot = _owner_cpu_snapshot_v2(bad_owner)
        invalid_before_snapshot["input"] = {
            key: value.clone() for key, value in invalid_before_snapshot.items()
        }
        _prepare_graph_if_needed(candidate, bad_owner, 1, 0)
        candidate.module.advance(*bad_owner, 1)
        torch.cuda.synchronize()
        bad_header_after = (
            _field(bad_owner.state, layout, "header", torch).detach().cpu()
        )
        invalid_after_snapshot = _owner_cpu_snapshot_v2(bad_owner)
        invalid_after_snapshot["input"] = {
            key: value.clone()
            for key, value in invalid_before_snapshot.items()
            if key != "input"
        }
    finally:
        try:
            _release_graph(candidate)
        finally:
            bad_owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            invalid_closeout_authority = _closeout_attempt_authority_v2(
                candidate,
                started_binding=invalid_execution_authority,
                validator=attempt_authority_validator,
                device_uuid=device_uuid,
            )
    checks = {
        "active_order_is_strict_biological_uid_order": bool(
            (ordered_ids[1:] > ordered_ids[:-1]).all().item()
        ),
        "birth_reuses_low_slot_but_appends_uid_order": low_slot_reuse,
        "restore_reconstructs_after_arbitrary_derived_corruption": restored_exact,
        "restore_continuation_state_exact": bool(
            torch.equal(
                restored_continuation_snapshot["state"],
                clean_continuation_snapshot["state"],
            )
        ),
        "restore_continuation_scratch_exact": bool(
            torch.equal(
                restored_continuation_snapshot["scratch"],
                clean_continuation_snapshot["scratch"],
            )
        ),
        "restore_continuation_events_exact": bool(
            torch.equal(
                restored_continuation_snapshot["events"],
                clean_continuation_snapshot["events"],
            )
            and torch.equal(
                restored_continuation_snapshot["counts"],
                clean_continuation_snapshot["counts"],
            )
        ),
        "checkpoint_binds_external_exact_content_sink_ack": (
            checkpoint["event_sink_ack"] == event_sink_ack
            and checkpoint["checkpoint_identity"]["event_sink_ack_sha256"]
            == event_sink_ack["receipt_file_sha256"]
        ),
        "sealed_checkpoint_reopens_exact_state_bytes": (
            reopened_state == checkpoint_state_bytes
        ),
        "fresh_process_restore_continuation_exact": (
            worker_seal_exact
            and worker_receipt.get("source_pid") != os.getpid()
            and worker_receipt.get("state_sha256")
            == same_process_continuation["state_sha256"]
            and worker_receipt.get("scratch_sha256")
            == same_process_continuation["scratch_sha256"]
            and worker_receipt.get("events_sha256")
            == same_process_continuation["events_sha256"]
            and worker_receipt.get("counts_sha256")
            == same_process_continuation["counts_sha256"]
            and worker_receipt.get("next_absolute_tick")
            == same_process_continuation["next_absolute_tick"]
        ),
        "invalid_restore_status_preserved_and_dispatch_fail_stops": (
            restore_rejected
            and preserved_status != 0
            and int(bad_header_after[0, 4]) == preserved_status
            and int(bad_header_after[0, 1]) == bad_frontier
        ),
    }
    witness = {
        "checkpoint_bundle": checkpoint,
        "checkpoint_bundle_receipt_path": str(checkpoint_bundle.resolve()),
        "checkpoint_bundle_receipt_file_sha256": file_sha256(checkpoint_bundle),
        "fresh_process_receipt": worker_receipt,
        "fresh_process_receipt_path": str(worker_path),
        "fresh_process_receipt_file_sha256": file_sha256(worker_path),
        "same_process_continuation": same_process_continuation,
        "invalid_restore_status": preserved_status,
        "birth_slots": [int(value) for value in birth_slots.tolist()],
        "active_count": active_count,
        "first_uid": int(ordered_ids[0]),
        "last_uid": int(ordered_ids[-1]),
        "checkpoint_closeout_attempt_authority": checkpoint_closeout_authority,
        "reconstruction_execution_attempt_authority": (
            reconstruction_execution_authority
        ),
        "reconstruction_closeout_attempt_authority": (
            reconstruction_closeout_authority
        ),
        "restored_execution_attempt_authority": restored_execution_authority,
        "restored_closeout_attempt_authority": restored_closeout_authority,
        "clean_execution_attempt_authority": clean_execution_authority,
        "clean_closeout_attempt_authority": clean_closeout_authority,
        "invalid_execution_attempt_authority": invalid_execution_authority,
        "invalid_closeout_attempt_authority": invalid_closeout_authority,
        "attempt_lifecycles": [
            _attempt_lifecycle_row_v2(
                "restore_checkpoint_dispatch",
                checkpoint_attempt_authority,
                checkpoint_closeout_authority,
            ),
            _attempt_lifecycle_row_v2(
                "restore_reconstruction",
                reconstruction_execution_authority,
                reconstruction_closeout_authority,
            ),
            _attempt_lifecycle_row_v2(
                "restore_same_process_continuation",
                restored_execution_authority,
                restored_closeout_authority,
            ),
            _attempt_lifecycle_row_v2(
                "restore_clean_continuation",
                clean_execution_authority,
                clean_closeout_authority,
            ),
            _attempt_lifecycle_row_v2(
                "restore_fresh_process_continuation",
                worker_receipt["attempt_authority_binding"],
                worker_receipt["closeout_attempt_authority"],
            ),
            _attempt_lifecycle_row_v2(
                "restore_invalid_fail_stop",
                invalid_execution_authority,
                invalid_closeout_authority,
            ),
        ],
    }
    fresh_raw = _require_worker_raw_continuation_v2(worker_receipt, worker_path.parent)
    raw_datasets = {
        "restore_checkpoint": _raw_dataset_v2(
            checkpoint_snapshot, start_next_absolute_tick=21
        ),
        "restore_reconstructed": _raw_dataset_v2(
            reconstructed_snapshot, start_next_absolute_tick=21
        ),
        "restore_same_continuation": _raw_dataset_v2(
            restored_continuation_snapshot, start_next_absolute_tick=next_tick
        ),
        "restore_fresh_continuation": fresh_raw,
        "restore_invalid_before": _raw_dataset_v2(
            invalid_before_snapshot, start_next_absolute_tick=next_tick
        ),
        "restore_invalid_after": _raw_dataset_v2(
            invalid_after_snapshot, start_next_absolute_tick=next_tick
        ),
    }
    return checks, witness, raw_datasets


def _semantic_case_initialize_owner_v2(
    loaded: _LoadedArtifact,
    owner: _OwnedBuffers,
    *,
    fixture_law: str,
    start_tick: int,
) -> None:
    if fixture_law == "DETERMINISTIC_SEMANTIC_WORLD_1":
        world_uid_base, rng_key, fixture = 90_002, 17, True
    elif fixture_law in {
        "DETERMINISTIC_SEMANTIC",
        "DETERMINISTIC_SEMANTIC_WORLD_0",
    }:
        world_uid_base, rng_key, fixture = 90_001, 17, True
    elif fixture_law == "ZERO_WORLD":
        world_uid_base, rng_key, fixture = 91_000, 31, False
    elif fixture_law == "BIRTH_OVERFLOW":
        world_uid_base, rng_key, fixture = 95_000, 43, False
    elif fixture_law == "MIDDLE_SURVIVOR_DEATH":
        world_uid_base, rng_key, fixture = 95_500, 45, False
    elif fixture_law == "TRANSACTION_ERROR":
        world_uid_base, rng_key, fixture = 96_000, 47, False
    elif fixture_law == "TRANSACTION_ERROR_AFTER_SUCCESS":
        world_uid_base, rng_key, fixture = 98_000, 59, True
    else:
        raise FullEvolutionWorkhorseProbeError(
            f"semantic case fixture {fixture_law!r} requires a specialized path"
        )
    _initialize_owner(
        loaded,
        owner,
        world_uid_base=world_uid_base,
        start_tick=start_tick,
        rng_key=rng_key,
        fixture=fixture,
    )
    if fixture_law == "BIRTH_OVERFLOW":
        loaded.module.force_birth_overflow_fixture(owner.state, owner.scratch)
    elif fixture_law == "MIDDLE_SURVIVOR_DEATH":
        loaded.module.force_middle_survivor_death_fixture(owner.state, owner.scratch)
    elif fixture_law == "TRANSACTION_ERROR":
        loaded.module.force_transaction_error_fixture(owner.state, owner.scratch)


def _semantic_parent_dispatch_snapshot_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    parent: Mapping[str, Any],
    B: int,
    temporal_k: int,
    start_tick: int,
    fixture_law: str,
    disable_mask: int,
    sink_root: Path,
) -> dict[str, Any]:
    owner = _allocate_owner(torch, loaded, B, temporal_k)
    _semantic_case_initialize_owner_v2(
        loaded, owner, fixture_law=fixture_law, start_tick=start_tick
    )
    owner.attempt_authority_binding = dict(parent)
    input_snapshot = _owner_cpu_snapshot_v2(owner)
    try:
        _prepare_graph_if_needed(loaded, owner, temporal_k, disable_mask)
        _advance(loaded, owner, temporal_k, disable_mask)
        sink_ack = _acknowledge_events(
            owner,
            torch,
            sink_root=sink_root,
            sink_id="workhorse-canonical-semantic-case-v2",
        )
        snapshot = _owner_cpu_snapshot_v2(owner)
        snapshot["input"] = input_snapshot
        snapshot["event_sink_ack"] = sink_ack
        return snapshot
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _semantic_parent_k1_sequence_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    parent: Mapping[str, Any],
    ticks: int,
    start_tick: int,
    fixture_law: str,
    disable_mask: int,
    sink_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    owner = _allocate_owner(torch, loaded, int(parent["contract"]["B"]), 1)
    _semantic_case_initialize_owner_v2(
        loaded, owner, fixture_law=fixture_law, start_tick=start_tick
    )
    owner.attempt_authority_binding = dict(parent)
    initial = _owner_cpu_snapshot_v2(owner)
    parts: list[dict[str, Any]] = []
    events: list[Any] = []
    count_total = torch.zeros((owner.state.size(0),), dtype=torch.int32)
    try:
        _prepare_graph_if_needed(loaded, owner, 1, disable_mask)
        for index in range(ticks):
            dispatch_input = _owner_cpu_snapshot_v2(owner)
            _advance(loaded, owner, 1, disable_mask)
            _acknowledge_events(
                owner,
                torch,
                sink_root=sink_root / f"tick-{index:03d}",
                sink_id="workhorse-canonical-semantic-k1-sequence-v2",
            )
            snapshot = _owner_cpu_snapshot_v2(owner)
            snapshot["input"] = dispatch_input
            parts.append(snapshot)
            events.append(snapshot["events"])
            count_total += snapshot["counts"]
        combined_events = torch.cat(events, dim=1)
        combined = {
            "state": parts[-1]["state"],
            "scratch": parts[-1]["scratch"],
            "events": combined_events,
            "counts": count_total,
            "input": {
                "state": initial["state"],
                "scratch": initial["scratch"],
                "events": torch.zeros_like(combined_events),
                "counts": initial["counts"],
            },
        }
        return combined, parts
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _semantic_checkpoint_source_v2(
    source_checkpoint_binding: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    binding = validate_external_checkpoint_source_binding_v2(
        source_checkpoint_binding,
        expected_identity_sha256=expected_identity_sha256,
    )
    fragment, _ = reopen_workhorse_semantic_case_fragment_v2(
        binding["base_checkpoint_manifest_path"]
    )
    checkpoint = fragment["base_checkpoint"]
    if (
        fragment["case_id"] != "checkpoint_same_restore_sequence"
        or checkpoint is None
        or checkpoint["identity"]["base_checkpoint_identity_sha256"]
        != expected_identity_sha256
    ):
        raise FullEvolutionWorkhorseProbeError(
            "external checkpoint binding resolves to another semantic base"
        )
    state_path = (
        Path(binding["base_checkpoint_manifest_path"]).parent
        / checkpoint["state_file_name"]
    )
    return checkpoint["identity"], state_path.read_bytes()


def _semantic_restore_owner_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    parent: Mapping[str, Any],
    identity: Mapping[str, Any],
    state_bytes: bytes,
) -> _OwnedBuffers:
    owner = _allocate_owner(torch, loaded, int(identity["B"]), 1)
    owner.world_uids = tuple(identity["world_uids"])
    owner.rng_keys = tuple(identity["rng_keys"])
    owner.committed_next_tick = int(identity["checkpoint_at_tick"])
    owner.attempt_authority_binding = dict(parent)
    owner.state.copy_(
        torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8).reshape_as(
            owner.state
        )
    )
    owner.scratch.zero_()
    return owner


def _semantic_checkpoint_same_restore_case_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    parent: Mapping[str, Any],
    sink_root: Path,
    science_checkpoint_compatibility_binding: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    science_compatibility = (
        validate_workhorse_science_checkpoint_compatibility_binding_v2(
            science_checkpoint_compatibility_binding
        )
    )
    if (
        parent["science_continuation_contract_sha256"]
        != science_compatibility["science_consumer_extension_contract_sha256"]
    ):
        raise FullEvolutionWorkhorseProbeError(
            "checkpoint parent binds another science continuation contract"
        )
    owner = _allocate_owner(torch, loaded, 1, 1)
    _initialize_owner(
        loaded,
        owner,
        world_uid_base=97_000,
        start_tick=21,
        rng_key=53,
        fixture=True,
    )
    owner.attempt_authority_binding = dict(parent)
    checkpoint_input = _owner_cpu_snapshot_v2(owner)
    try:
        _prepare_graph_if_needed(loaded, owner, 1, 0)
        _advance(loaded, owner, 1, 0)
        checkpoint_ack = _acknowledge_events(
            owner,
            torch,
            sink_root=sink_root / "checkpoint",
            sink_id="workhorse-canonical-semantic-checkpoint-v2",
        )
        checkpoint_snapshot = _owner_cpu_snapshot_v2(owner)
        checkpoint_snapshot["input"] = checkpoint_input
        checkpoint_state = checkpoint_snapshot["state"]
        checkpoint_bytes = checkpoint_state.contiguous().numpy().tobytes()
        checkpoint_tick = owner.committed_next_tick
        world_uids, rng_keys = owner.world_uids, owner.rng_keys
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    reconstruction = _allocate_owner(torch, loaded, 1, 1)
    reconstruction.world_uids = world_uids
    reconstruction.rng_keys = rng_keys
    reconstruction.committed_next_tick = checkpoint_tick
    reconstruction.attempt_authority_binding = dict(parent)
    reconstruction.state.copy_(checkpoint_state)
    reconstruction.scratch.zero_()
    occupancy = loaded.layout.field("occupancy")
    reconstruction.state[
        :, occupancy.offset : occupancy.offset + occupancy.nbytes
    ].zero_()
    reconstruction.scratch.fill_(0xA5)
    corrupted = _owner_cpu_snapshot_v2(reconstruction)
    loaded.module.restore_derived(reconstruction.state, reconstruction.scratch)
    torch.cuda.synchronize()
    reconstructed = _owner_cpu_snapshot_v2(reconstruction)
    reconstructed["input"] = corrupted
    if not torch.equal(reconstructed["state"], checkpoint_state):
        raise FullEvolutionWorkhorseProbeError(
            "semantic checkpoint reconstruction did not restore exact state"
        )
    continuation_input = _owner_cpu_snapshot_v2(reconstruction)
    try:
        _prepare_graph_if_needed(loaded, reconstruction, 1, 0)
        _advance(loaded, reconstruction, 1, 0)
        _acknowledge_events(
            reconstruction,
            torch,
            sink_root=sink_root / "same-continuation",
            sink_id="workhorse-canonical-semantic-same-restore-v2",
        )
        continuation = _owner_cpu_snapshot_v2(reconstruction)
        continuation["input"] = continuation_input
    finally:
        _release_graph(loaded)
        reconstruction = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    base_identity = science_compatible_base_checkpoint_identity_v2(
        shape_id=loaded.shape.shape_id,
        source_artifact_identity_sha256=loaded.manifest["identity"]["identity_sha256"],
        checkpoint_at_tick=checkpoint_tick,
        state_bytes=checkpoint_bytes,
        world_uids=world_uids,
        rng_keys=rng_keys,
        event_sink_acknowledgement=checkpoint_ack,
        science_checkpoint_compatibility_binding=(science_compatibility),
    )
    return (
        {
            "restore_checkpoint": _raw_dataset_v2(
                checkpoint_snapshot, start_next_absolute_tick=21
            ),
            "restore_reconstructed": _raw_dataset_v2(
                reconstructed, start_next_absolute_tick=21
            ),
            "restore_same_continuation": _raw_dataset_v2(
                continuation, start_next_absolute_tick=checkpoint_tick
            ),
        },
        {"identity": base_identity, "state": checkpoint_bytes},
    )


def _semantic_restore_case_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    parent: Mapping[str, Any],
    case_id: str,
    source_checkpoint_binding: Mapping[str, Any],
    sink_root: Path,
) -> dict[str, dict[str, Any]]:
    source_identity = parent["source_checkpoint_identity_sha256"]
    identity, state_bytes = _semantic_checkpoint_source_v2(
        source_checkpoint_binding,
        expected_identity_sha256=source_identity,
    )
    if (
        identity["source_artifact_identity_sha256"]
        != loaded.manifest["identity"]["identity_sha256"]
        or parent["science_continuation_contract_sha256"]
        != identity["science_checkpoint_compatibility"][
            "science_consumer_extension_contract_sha256"
        ]
    ):
        raise FullEvolutionWorkhorseProbeError(
            "fresh restore artifact differs from the base checkpoint identity"
        )
    owner = _semantic_restore_owner_v2(
        loaded,
        torch,
        parent=parent,
        identity=identity,
        state_bytes=state_bytes,
    )
    if case_id == "fresh_restore_continuation":
        loaded.module.restore_derived(owner.state, owner.scratch)
        continuation_input = _owner_cpu_snapshot_v2(owner)
        try:
            _prepare_graph_if_needed(loaded, owner, 1, 0)
            _advance(loaded, owner, 1, 0)
            _acknowledge_events(
                owner,
                torch,
                sink_root=sink_root,
                sink_id="workhorse-canonical-semantic-fresh-restore-v2",
            )
            continuation = _owner_cpu_snapshot_v2(owner)
            continuation["input"] = continuation_input
        finally:
            _release_graph(loaded)
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return {
            "restore_fresh_continuation": _raw_dataset_v2(
                continuation,
                start_next_absolute_tick=int(identity["checkpoint_at_tick"]),
            )
        }

    order = loaded.layout.field("active_slots_by_uid")
    owner.state[:, order.offset + 4 : order.offset + 8].zero_()
    owner.scratch.zero_()
    restore_rejected = False
    try:
        loaded.module.restore_derived(owner.state, owner.scratch)
    except RuntimeError:
        restore_rejected = True
    torch.cuda.synchronize()
    invalid_before = _owner_cpu_snapshot_v2(owner)
    invalid_before["input"] = {
        key: value.clone() for key, value in invalid_before.items()
    }
    header_before = _field(owner.state, loaded.layout, "header", torch)
    status_before = int(header_before[0, 4])
    frontier_before = int(header_before[0, 1])
    try:
        _prepare_graph_if_needed(loaded, owner, 1, 0)
        _advance(loaded, owner, 1, 0)
        torch.cuda.synchronize()
        invalid_after = _owner_cpu_snapshot_v2(owner)
        invalid_after["input"] = {
            key: value.clone()
            for key, value in invalid_before.items()
            if key != "input"
        }
        header_after = _field(owner.state, loaded.layout, "header", torch)
        if (
            not restore_rejected
            or status_before == 0
            or int(header_after[0, 4]) != status_before
            or int(header_after[0, 1]) != frontier_before
        ):
            raise FullEvolutionWorkhorseProbeError(
                "invalid restore did not produce the exact expected fail-stop"
            )
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {
        "restore_invalid_before": _raw_dataset_v2(
            invalid_before, start_next_absolute_tick=frontier_before
        ),
        "restore_invalid_after": _raw_dataset_v2(
            invalid_after, start_next_absolute_tick=frontier_before
        ),
    }


def _semantic_transaction_case_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    parent: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    owner = _allocate_owner(torch, loaded, 1, 2)
    _semantic_case_initialize_owner_v2(
        loaded, owner, fixture_law="TRANSACTION_ERROR", start_tick=13
    )
    owner.attempt_authority_binding = dict(parent)
    transaction_input = _owner_cpu_snapshot_v2(owner)
    try:
        _prepare_graph_if_needed(loaded, owner, 2, 0)
        _advance(loaded, owner, 2, 0)
        rows, _counts, _statuses, _next_ticks = _drain_sparse_page_v3(
            owner,
            torch,
            start_tick=13,
            end_tick=15,
        )
        owner.last_dense_events_cpu = _dense_event_oracle_v3(
            owner,
            torch,
            rows,
            start_tick=13,
            end_tick=15,
        )
        if owner.last_dense_events_cpu is None:
            raise FullEvolutionWorkhorseProbeError(
                "tiny transaction fixture did not materialize its dense event oracle"
            )
        transaction = _owner_cpu_snapshot_v2(owner)
        transaction["input"] = transaction_input
        header = _field(owner.state, loaded.layout, "header", torch)
        if (
            int(header[0, 4]) != 1
            or int(header[0, 1]) != 13
            or owner.pending is None
            or int(torch.count_nonzero(owner.last_dense_events_cpu[:, 1:])) != 0
        ):
            raise FullEvolutionWorkhorseProbeError(
                "transaction fixture did not produce the exact expected fail-stop"
            )
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {
        "transaction_after": _raw_dataset_v2(transaction, start_next_absolute_tick=13)
    }


def _active_event_row_bytes_v2(events: Any, torch: Any) -> tuple[int, bytes]:
    rows = events.reshape(-1, 48).detach().cpu().contiguous()
    mask = rows[:, 46] != 0
    active = rows[mask].contiguous()
    return int(mask.sum().item()), active.numpy().tobytes()


def _semantic_p2_reused_buffer_case_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    *,
    parent: Mapping[str, Any],
    temporal_k: int,
    sink_root: Path,
) -> dict[str, Any]:
    owner = _allocate_owner(torch, loaded, 1, temporal_k)
    _semantic_case_initialize_owner_v2(
        loaded,
        owner,
        fixture_law="TRANSACTION_ERROR_AFTER_SUCCESS",
        start_tick=41,
    )
    owner.attempt_authority_binding = dict(parent)
    storage_ptr = int(owner.events.data_ptr())
    try:
        _prepare_graph_if_needed(loaded, owner, temporal_k, 0)
        _advance(loaded, owner, temporal_k, 0)
        successful_ack = _acknowledge_events(
            owner,
            torch,
            sink_root=sink_root / "successful-k-dispatch",
            sink_id="workhorse-canonical-semantic-p2-success-v2",
        )
        if owner.committed_next_tick != 41 + temporal_k or owner.pending is not None:
            raise FullEvolutionWorkhorseProbeError(
                "P2 setup dispatch was not acknowledged exactly"
            )
        loaded.module.force_event_then_transaction_error_fixture(
            owner.state, owner.scratch
        )
        _advance(loaded, owner, temporal_k, 0)
        torch.cuda.synchronize()
        active_before, active_bytes_before = _active_event_row_bytes_v2(
            owner.events, torch
        )
        loaded.module.canonicalize_events(owner.events)
        torch.cuda.synchronize()
        active_after, active_bytes_after = _active_event_row_bytes_v2(
            owner.events, torch
        )
        failed_start_tick = 41 + temporal_k
        failed_end_tick = failed_start_tick + temporal_k
        rows, _counts, _statuses, _next_ticks = _drain_sparse_page_v3(
            owner,
            torch,
            start_tick=failed_start_tick,
            end_tick=failed_end_tick,
        )
        failed_dense_events = _dense_event_oracle_v3(
            owner,
            torch,
            rows,
            start_tick=failed_start_tick,
            end_tick=failed_end_tick,
        )
        if failed_dense_events is None:
            raise FullEvolutionWorkhorseProbeError(
                "tiny reused-buffer fixture did not materialize its dense event oracle"
            )
        active_ticks = [
            struct.unpack_from("<Q", active_bytes_after, offset)[0]
            for offset in range(0, len(active_bytes_after), 48)
        ]
        flat = owner.events.reshape(-1, 48)
        inactive = flat[:, 46] == 0
        inactive_zero = bool((flat[inactive] == 0).all().item())
        later_zero = bool((failed_dense_events[:, 1:] == 0).all().item())
        header = _field(owner.state, loaded.layout, "header", torch)
        status = int(header[0, 4])
        frontier = int(header[0, 1])
        owner.poisoned_status = (status,)
        reuse_rejected = False
        try:
            _prepare_dispatch_v2(loaded, owner, temporal_k, 0)
        except FullEvolutionWorkhorseProbeError:
            reuse_rejected = True
        p2_conditions = {
            "active_rows_emitted_before_canonicalization": active_before > 0,
            "active_row_count_preserved": active_after == active_before,
            "active_row_bytes_preserved": active_bytes_after == active_bytes_before,
            "active_rows_match_failed_absolute_tick": all(
                tick == 41 + temporal_k for tick in active_ticks
            ),
            "inactive_rows_exact_zero_after_canonicalization": inactive_zero,
            "later_k_slices_exact_zero_after_canonicalization": later_zero,
            "failed_status_nonzero": status != 0,
            "failed_frontier_uncommitted": frontier == 41 + temporal_k,
            "same_owner_storage_data_ptr": (
                int(owner.events.data_ptr()) == storage_ptr
            ),
            "failed_dispatch_pending_preserved": owner.pending is not None,
            "post_failure_reuse_rejected": reuse_rejected,
        }
        if not all(p2_conditions.values()):
            raise FullEvolutionWorkhorseProbeError(
                "P2 reused-buffer law failed: "
                + json.dumps(p2_conditions, sort_keys=True, separators=(",", ":"))
            )
        witness_body = {
            "schema_version": (
                "eiel_workhorse_reused_event_buffer_canonicalization_witness_v2"
            ),
            "B": 1,
            "temporal_k": temporal_k,
            "successful_dispatch_start_tick": 41,
            "successful_dispatch_end_tick": 41 + temporal_k,
            "successful_ack_receipt_path": successful_ack["receipt_path"],
            "successful_ack_receipt_file_sha256": successful_ack["receipt_file_sha256"],
            "failed_event_acknowledged": False,
            "failed_event_publication_path": None,
            "failed_dispatch_pending_preserved": owner.pending is not None,
            "poisoned_status": list(owner.poisoned_status),
            "post_failure_reuse_rejected": reuse_rejected,
            "failed_event_buffer_cannot_be_reused": reuse_rejected,
            "canonicalization_law": (
                "NATIVE_CANONICALIZATION_WITHOUT_ACK_OR_PUBLICATION"
            ),
            "failed_dispatch_start_tick": 41 + temporal_k,
            "failed_dispatch_requested_end_tick": 41 + 2 * temporal_k,
            "failed_status": status,
            "failed_frontier": frontier,
            "active_rows_before_canonicalization": active_before,
            "active_rows_after_canonicalization": active_after,
            "active_row_bytes_sha256_before_canonicalization": hashlib.sha256(
                active_bytes_before
            ).hexdigest(),
            "active_row_bytes_sha256_after_canonicalization": hashlib.sha256(
                active_bytes_after
            ).hexdigest(),
            "active_row_bytes_hex_before_canonicalization": active_bytes_before.hex(),
            "active_row_bytes_hex_after_canonicalization": active_bytes_after.hex(),
            "active_row_absolute_ticks": active_ticks,
            "active_rows_match_failed_absolute_tick": all(
                tick == 41 + temporal_k for tick in active_ticks
            ),
            "inactive_rows_exact_zero_after_canonicalization": inactive_zero,
            "later_k_slices_exact_zero_after_canonicalization": later_zero,
            "active_rows_byte_preserved": active_bytes_after == active_bytes_before,
            "same_owner_storage_data_ptr": int(owner.events.data_ptr()) == storage_ptr,
            "successful_ack_preceded_failed_reuse": True,
        }
        return {
            **witness_body,
            "witness_sha256": canonical_sha256(witness_body),
        }
    finally:
        _release_graph(loaded)
        owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_workhorse_semantic_case_v2(
    artifact_path: str | Path,
    *,
    case_id: str,
    semantic_temporal_k: int,
    event_sink_root: str | Path,
    fragment_output_root: str | Path,
    canonical_parent_authority_binding: Mapping[str, Any],
    external_source_checkpoint_binding: Mapping[str, Any] | None = None,
    science_checkpoint_compatibility_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one closed semantic case under one already-canonical Attempt.

    Expected-negative cases return normally with a sealed provisional result
    and an explicit failed-Attempt terminalization requirement.  They are not
    VALID_NEGATIVE until the outer adapter reopens the matching Scheduler
    FAILED terminal and invokes the postterminal authority writer.  Unexpected
    infrastructure failures still raise and cannot acquire that classification.
    """

    law = workhorse_semantic_case_law_v2(
        case_id, semantic_temporal_k=semantic_temporal_k
    )
    parent = validate_workhorse_semantic_case_parent_binding_v2(
        canonical_parent_authority_binding,
        case_id=case_id,
        semantic_temporal_k=semantic_temporal_k,
    )
    torch = _torch()
    loaded = _load(artifact_path, torch)
    if (law.artifact_role == "REFERENCE") != (
        loaded.backend.backend_id == BACKEND_REFERENCE_PHASED_V2
    ) or file_sha256(loaded.artifact) != parent["artifact_sha256"]:
        raise FullEvolutionWorkhorseProbeError(
            "semantic case artifact role/hash differs from the parent Attempt"
        )
    if law.artifact_role == "CANDIDATE" and loaded.backend.backend_id not in {
        BACKEND_PHASE_GRAPH_V2,
        BACKEND_FUSED_CTA_TEMPORAL_V2,
    }:
        raise FullEvolutionWorkhorseProbeError(
            "candidate semantic case requires a performance backend"
        )
    sink_root = Path(event_sink_root).resolve()
    datasets: dict[str, dict[str, Any]] = {}
    base_checkpoint = None
    p2_witness = None
    if law.sequence_law == "ONE_K_DISPATCH" or law.sequence_law == "ONE_K1_DISPATCH":
        snapshot = _semantic_parent_dispatch_snapshot_v2(
            loaded,
            torch,
            parent=parent,
            B=law.B,
            temporal_k=law.dispatch_temporal_k,
            start_tick=law.start_next_absolute_tick,
            fixture_law=law.fixture_law,
            disable_mask=law.disable_mask,
            sink_root=sink_root,
        )
        datasets[law.fragment_datasets[0]] = _raw_dataset_v2(
            snapshot,
            start_next_absolute_tick=law.start_next_absolute_tick,
        )
    elif law.sequence_law == "K1_SEQUENCE_COMBINED":
        combined, _ = _semantic_parent_k1_sequence_v2(
            loaded,
            torch,
            parent=parent,
            ticks=law.total_ticks,
            start_tick=law.start_next_absolute_tick,
            fixture_law=law.fixture_law,
            disable_mask=law.disable_mask,
            sink_root=sink_root,
        )
        datasets[law.fragment_datasets[0]] = _raw_dataset_v2(
            combined,
            start_next_absolute_tick=law.start_next_absolute_tick,
        )
    elif law.sequence_law == "K1_SEQUENCE_PREFIX_AND_COMBINED":
        combined, parts = _semantic_parent_k1_sequence_v2(
            loaded,
            torch,
            parent=parent,
            ticks=law.total_ticks,
            start_tick=law.start_next_absolute_tick,
            fixture_law=law.fixture_law,
            disable_mask=law.disable_mask,
            sink_root=sink_root,
        )
        datasets["causal_mutation_disabled"] = _raw_dataset_v2(
            parts[0], start_next_absolute_tick=law.start_next_absolute_tick
        )
        datasets["eed_enabled"] = _raw_dataset_v2(
            combined, start_next_absolute_tick=law.start_next_absolute_tick
        )
    elif law.sequence_law == "K1_SEQUENCE_PREFIXES":
        _, parts = _semantic_parent_k1_sequence_v2(
            loaded,
            torch,
            parent=parent,
            ticks=law.total_ticks,
            start_tick=law.start_next_absolute_tick,
            fixture_law=law.fixture_law,
            disable_mask=law.disable_mask,
            sink_root=sink_root,
        )
        datasets = {
            f"wdl_tick{index}": _raw_dataset_v2(
                snapshot,
                start_next_absolute_tick=law.start_next_absolute_tick + index - 1,
            )
            for index, snapshot in enumerate(parts, start=1)
        }
    elif law.sequence_law == "CHECKPOINT_RECONSTRUCT_AND_CONTINUE":
        if science_checkpoint_compatibility_binding is None:
            raise FullEvolutionWorkhorseProbeError(
                "checkpoint case requires the science compatibility binding"
            )
        datasets, base_checkpoint = _semantic_checkpoint_same_restore_case_v2(
            loaded,
            torch,
            parent=parent,
            sink_root=sink_root,
            science_checkpoint_compatibility_binding=(
                science_checkpoint_compatibility_binding
            ),
        )
    elif law.sequence_law in {
        "EXTERNAL_CHECKPOINT_RESTORE_AND_CONTINUE",
        "INVALID_RESTORE_THEN_FAIL_STOP",
    }:
        if external_source_checkpoint_binding is None:
            raise FullEvolutionWorkhorseProbeError(
                "restore semantic case requires an external checkpoint binding"
            )
        datasets = _semantic_restore_case_v2(
            loaded,
            torch,
            parent=parent,
            case_id=case_id,
            source_checkpoint_binding=external_source_checkpoint_binding,
            sink_root=sink_root,
        )
    elif law.sequence_law == "ONE_K2_FAILED_DISPATCH":
        datasets = _semantic_transaction_case_v2(loaded, torch, parent=parent)
    elif law.sequence_law == "REUSED_OWNER_SUCCESS_ACK_THEN_FAILED_K_DISPATCH":
        p2_witness = _semantic_p2_reused_buffer_case_v2(
            loaded,
            torch,
            parent=parent,
            temporal_k=semantic_temporal_k,
            sink_root=sink_root,
        )
    else:  # pragma: no cover - the closed table prevents an unknown law
        raise FullEvolutionWorkhorseProbeError(
            f"semantic case sequence law {law.sequence_law!r} is unsupported"
        )
    fragment_path = write_workhorse_semantic_case_fragment_v2(
        fragment_output_root,
        case_id=case_id,
        semantic_temporal_k=semantic_temporal_k,
        shape_id=loaded.shape.shape_id,
        artifact_sha256=file_sha256(loaded.artifact),
        canonical_parent_binding=parent,
        native_outcome=law.expected_native_outcome,
        datasets=datasets,
        base_checkpoint=base_checkpoint,
        p2_witness=p2_witness,
        external_source_checkpoint_binding=external_source_checkpoint_binding,
    )
    fragment, _ = reopen_workhorse_semantic_case_fragment_v2(fragment_path)
    return {
        "schema_version": "eiel_workhorse_semantic_case_provisional_result_v2",
        "claim_class": "TECHNICAL_ONLY_PROVISIONAL",
        "qualification_authority_granted": False,
        "science_authority_granted": False,
        "case_id": case_id,
        "expected_attempt_terminal_status": law.expected_attempt_terminal_status,
        "attempt_terminal_contract": workhorse_semantic_case_terminal_contract_v2(law),
        "provisional_result_class": (
            "SUCCESS_PENDING_COMPLETED_TERMINAL"
            if law.expected_attempt_terminal_status == "COMPLETED"
            else "EXPECTED_NEGATIVE_PENDING_FAILED_TERMINAL"
        ),
        "provider_terminalization_action": (
            "RETURN_FOR_COMPLETED_TERMINALIZATION"
            if law.expected_attempt_terminal_status == "COMPLETED"
            else "RETURN_FOR_FAILED_ATTEMPT_TERMINALIZATION"
        ),
        "case_fragment_path": str(fragment_path.resolve()),
        "case_fragment_file_sha256": file_sha256(fragment_path),
        "case_fragment_body_sha256": fragment["fragment_body_sha256"],
        "source_checkpoint_identity_sha256": parent[
            "source_checkpoint_identity_sha256"
        ],
    }


def run_workhorse_semantic_gate_v2(
    reference_artifact: str | Path,
    candidate_artifact: str | Path,
    *,
    temporal_k: int = 2,
    event_sink_root: str | Path,
    checkpoint_root: str | Path,
    semantic_evidence_root: str | Path,
    receipt_output_path: str | Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
) -> dict[str, Any]:
    """Retired direct entry; canonical semantic-gate provider is required."""

    raise FullEvolutionWorkhorseProbeError(
        "direct Workhorse semantic GPU execution is retired; use the canonical "
        "ExecutionRequestV2 semantic-gate provider"
    )

    torch = _torch()
    reference = _load(reference_artifact, torch)
    candidate = _load(candidate_artifact, torch)
    if reference.backend.backend_id != BACKEND_REFERENCE_PHASED_V2:
        raise FullEvolutionWorkhorseProbeError(
            "reference artifact is not reference_phased_v2"
        )
    if candidate.backend.backend_id not in (
        BACKEND_PHASE_GRAPH_V2,
        BACKEND_FUSED_CTA_TEMPORAL_V2,
    ):
        raise FullEvolutionWorkhorseProbeError("candidate is not a performance backend")
    if reference.shape != candidate.shape:
        raise FullEvolutionWorkhorseProbeError(
            "same-shape reference/candidate is required"
        )
    if temporal_k not in candidate.shape.temporal_k_candidates or temporal_k == 1:
        raise FullEvolutionWorkhorseProbeError(
            "semantic gate requires a compiled K>1 candidate"
        )
    sink_root = Path(event_sink_root).resolve()
    checkpoint_directory = Path(checkpoint_root).resolve()
    runtime = dict(candidate.module.runtime_inspect())
    reference_runtime = dict(reference.module.runtime_inspect())
    device_uuid = runtime.get("device_uuid")
    if (
        type(device_uuid) is not str
        or not device_uuid
        or reference_runtime.get("device_uuid") != device_uuid
    ):
        raise FullEvolutionWorkhorseProbeError(
            "semantic reference/candidate device UUID differs"
        )
    authority_kwargs = {
        "attempt_authority_validator": attempt_authority_validator,
        "device_uuid": device_uuid,
    }

    candidate_k = _dispatch_snapshot(
        candidate,
        torch,
        B=2,
        temporal_k=temporal_k,
        sink_root=sink_root / "candidate-k",
        **authority_kwargs,
    )
    ref_owner = _allocate_owner(torch, reference, 2, 1)
    _initialize_owner(
        reference,
        ref_owner,
        world_uid_base=90_001,
        start_tick=71,
        rng_key=17,
        fixture=True,
    )
    reference_input = _owner_cpu_snapshot_v2(ref_owner)
    reference_execution_authority = _authorize_owner_v2(
        reference,
        ref_owner,
        temporal_k=1,
        total_ticks=temporal_k,
        phase="EXECUTION_ROW",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
    )
    ref_events: list[Any] = []
    ref_counts = torch.zeros((2,), dtype=torch.int32)
    try:
        for local_tick in range(temporal_k):
            _advance(reference, ref_owner, 1, 0)
            _acknowledge_events(
                ref_owner,
                torch,
                sink_root=sink_root / f"reference-k-{local_tick}",
                sink_id="workhorse-reference-k-partition-v2",
            )
            reference_tick = _owner_cpu_snapshot_v2(ref_owner)
            ref_events.append(reference_tick["events"])
            ref_counts += reference_tick["counts"]
        combined_reference_events = torch.cat(ref_events, dim=1)
        reference_k = {
            "state": ref_owner[0].detach().cpu(),
            "scratch": ref_owner[1].detach().cpu(),
            "events": combined_reference_events,
            "counts": ref_counts,
            "input": {
                "state": reference_input["state"],
                "scratch": reference_input["scratch"],
                "events": torch.zeros_like(combined_reference_events),
                "counts": reference_input["counts"],
            },
            "execution_attempt_authority": reference_execution_authority,
        }
    finally:
        ref_owner = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        reference_closeout_authority = _closeout_attempt_authority_v2(
            reference,
            started_binding=reference_execution_authority,
            validator=attempt_authority_validator,
            device_uuid=device_uuid,
        )
    reference_k["closeout_attempt_authority"] = reference_closeout_authority
    parity_checks = {
        "same_shape_reference_state_byte_parity": bool(
            torch.equal(candidate_k["state"], reference_k["state"])
        ),
        "same_shape_reference_scratch_byte_parity": bool(
            torch.equal(candidate_k["scratch"], reference_k["scratch"])
        ),
        "same_shape_reference_event_byte_parity": bool(
            torch.equal(candidate_k["events"], reference_k["events"])
        ),
        "same_shape_reference_event_count_parity": bool(
            torch.equal(candidate_k["counts"], reference_k["counts"])
        ),
    }

    whole_b = _dispatch_snapshot(
        candidate,
        torch,
        B=2,
        temporal_k=temporal_k,
        sink_root=sink_root / "whole-b",
        **authority_kwargs,
    )
    split_b = [
        _dispatch_snapshot(
            candidate,
            torch,
            B=1,
            temporal_k=temporal_k,
            world_uid_base=90_001 + world,
            sink_root=sink_root / f"split-b-{world}",
            **authority_kwargs,
        )
        for world in range(2)
    ]
    partition_checks = {
        "B_partition_state_invariance": bool(
            torch.equal(whole_b["state"], torch.cat([row["state"] for row in split_b]))
        ),
        "B_partition_scratch_invariance": bool(
            torch.equal(
                whole_b["scratch"], torch.cat([row["scratch"] for row in split_b])
            )
        ),
        "B_partition_event_invariance": bool(
            torch.equal(
                whole_b["events"], torch.cat([row["events"] for row in split_b])
            )
            and torch.equal(
                whole_b["counts"], torch.cat([row["counts"] for row in split_b])
            )
        ),
    }
    split_k = _tick_partition_snapshot(
        candidate,
        torch,
        ticks=temporal_k,
        sink_root=sink_root / "split-k",
        **authority_kwargs,
    )
    partition_checks.update(
        {
            "K_partition_state_invariance": bool(
                torch.equal(candidate_k["state"][0:1], split_k["state"])
            ),
            "K_partition_scratch_invariance": bool(
                torch.equal(candidate_k["scratch"][0:1], split_k["scratch"])
            ),
            "K_partition_event_invariance": bool(
                torch.equal(candidate_k["events"][0:1], split_k["events"])
                and torch.equal(candidate_k["counts"][0:1], split_k["counts"])
            ),
        }
    )

    causal_checks, causal_witness, causal_raw = _causal_checks(
        candidate,
        torch,
        sink_root=sink_root / "causal",
        **authority_kwargs,
    )
    restore_checks, restore_witness, restore_raw = _restore_and_checkpoint_check(
        candidate,
        torch,
        sink_root=sink_root / "restore",
        checkpoint_root=checkpoint_directory,
        **authority_kwargs,
    )
    overflow_checks, overflow_witness, overflow_raw = _overflow_parity_check(
        reference,
        candidate,
        torch,
        sink_root=sink_root / "birth-overflow",
        **authority_kwargs,
    )
    middle_checks, middle_witness, middle_raw = _middle_survivor_death_pairing_check(
        reference,
        candidate,
        torch,
        sink_root=sink_root / "middle-survivor-death",
        **authority_kwargs,
    )
    transaction_checks, transaction_witness, transaction_raw = (
        _transaction_fail_stop_check(
            candidate,
            torch,
            sink_root=sink_root / "transaction-fail-stop",
            **authority_kwargs,
        )
    )
    attempt_lifecycles = [
        _attempt_lifecycle_row_v2(
            "parity_candidate",
            candidate_k["execution_attempt_authority"],
            candidate_k["closeout_attempt_authority"],
        ),
        _attempt_lifecycle_row_v2(
            "parity_reference",
            reference_k["execution_attempt_authority"],
            reference_k["closeout_attempt_authority"],
        ),
        _attempt_lifecycle_row_v2(
            "b_whole",
            whole_b["execution_attempt_authority"],
            whole_b["closeout_attempt_authority"],
        ),
        *[
            _attempt_lifecycle_row_v2(
                f"b_split_{world}",
                snapshot["execution_attempt_authority"],
                snapshot["closeout_attempt_authority"],
            )
            for world, snapshot in enumerate(split_b)
        ],
        _attempt_lifecycle_row_v2(
            "k_split",
            split_k["execution_attempt_authority"],
            split_k["closeout_attempt_authority"],
        ),
        *causal_witness["attempt_lifecycles"],
        *restore_witness["attempt_lifecycles"],
        *overflow_witness["attempt_lifecycles"],
        *middle_witness["attempt_lifecycles"],
        *transaction_witness["attempt_lifecycles"],
    ]
    split_b_combined = {
        "state": torch.cat([row["state"] for row in split_b]),
        "scratch": torch.cat([row["scratch"] for row in split_b]),
        "events": torch.cat([row["events"] for row in split_b]),
        "counts": torch.cat([row["counts"] for row in split_b]),
        "input": {
            component: torch.cat([row["input"][component] for row in split_b])
            for component in ("state", "scratch", "events", "counts")
        },
    }
    raw_datasets = {
        "parity_reference": _raw_dataset_v2(reference_k, start_next_absolute_tick=71),
        "parity_candidate": _raw_dataset_v2(candidate_k, start_next_absolute_tick=71),
        "b_whole": _raw_dataset_v2(whole_b, start_next_absolute_tick=71),
        "b_split": _raw_dataset_v2(split_b_combined, start_next_absolute_tick=71),
        "k_split": _raw_dataset_v2(split_k, start_next_absolute_tick=71),
        **causal_raw,
        **restore_raw,
        **overflow_raw,
        **middle_raw,
        **transaction_raw,
    }
    evidence_path = write_workhorse_semantic_evidence_bundle_v2(
        semantic_evidence_root,
        shape_id=candidate.shape.shape_id,
        temporal_k=temporal_k,
        reference_artifact_sha256=file_sha256(reference.artifact),
        candidate_artifact_sha256=file_sha256(candidate.artifact),
        datasets=raw_datasets,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    evidence_value = json.loads(evidence_path.read_text(encoding="utf-8"))
    derived_checks, _ = derive_workhorse_semantic_checks_v2(evidence_path)
    measured_checks = {
        **parity_checks,
        **partition_checks,
        **causal_checks,
        **restore_checks,
        **overflow_checks,
        **middle_checks,
        **transaction_checks,
    }
    for name, measured in measured_checks.items():
        if name in derived_checks and derived_checks[name] is not measured:
            raise FullEvolutionWorkhorseProbeError(
                f"raw semantic evidence disagrees with measured check {name}"
            )
    receipt = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_semantic_gate_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pid": os.getpid(),
        "shape_id": candidate.shape.shape_id,
        "semantic_transition_sha256": candidate.manifest["identity"][
            "semantic_transition_sha256"
        ],
        "reference": {
            "artifact": str(reference.artifact),
            "artifact_sha256": file_sha256(reference.artifact),
            "manifest_sha256": reference.manifest["manifest_sha256"],
            "identity": reference.manifest["identity"],
            "build_identity": reference.manifest["build_identity"],
            "sealed_artifact_identity": reference.manifest["sealed_artifact_identity"],
        },
        "candidate": {
            "artifact": str(candidate.artifact),
            "artifact_sha256": file_sha256(candidate.artifact),
            "manifest_sha256": candidate.manifest["manifest_sha256"],
            "identity": candidate.manifest["identity"],
            "build_identity": candidate.manifest["build_identity"],
            "sealed_artifact_identity": candidate.manifest["sealed_artifact_identity"],
        },
        "runtime": runtime,
        "graph_inspection": candidate_k["graph"],
        "B": 2,
        "temporal_k": temporal_k,
        "semantic_evidence": {
            "manifest_path": str(evidence_path.resolve()),
            "manifest_file_sha256": file_sha256(evidence_path),
            "bundle_body_sha256": evidence_value["bundle_body_sha256"],
        },
        "attempt_lifecycles": attempt_lifecycles,
        "witnesses": {
            "causal": causal_witness,
            "restore": restore_witness,
            "birth_overflow": overflow_witness,
            "middle_survivor_death": middle_witness,
            "transaction_fail_stop": transaction_witness,
        },
    }
    output = write_workhorse_semantic_gate_receipt_v2(receipt, receipt_output_path)
    return json.loads(output.read_text(encoding="utf-8"))


def _p10(values: Sequence[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.10
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    )


def _authority_accounted_device_bytes_v2(binding: Mapping[str, Any]) -> int:
    response = binding["response"]
    field = (
        "owner_process_bytes"
        if response["accounting_mode"] == "NVML_PER_PROCESS"
        else "global_device_bytes"
    )
    value = response[field]
    if type(value) is not int or value < 0:
        raise FullEvolutionWorkhorseProbeError(
            "injected live device accounting value is invalid"
        )
    return value


def _calibrate_owner_components_v2(
    loaded: _LoadedArtifact,
    torch: Any,
    temporal_k: int,
    *,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    device_uuid: str,
    total_device_bytes: int,
) -> tuple[WorkhorseOwnerComponentsV2, dict[str, Any]]:
    """Measure exact non-tensor owner components before sealing the envelope."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    calibration_contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=loaded.shape.shape_id,
        backend_id=loaded.backend.backend_id,
        B=1,
        temporal_k=temporal_k,
        total_ticks=temporal_k,
    )
    baseline_authority = _resolve_contract_authority_v2(
        loaded,
        contract=calibration_contract,
        phase="CALIBRATION",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
    )
    lineage = baseline_authority["attempt_lineage_sha256"]
    process_baseline = _authority_accounted_device_bytes_v2(baseline_authority)
    reserved_baseline = int(torch.cuda.memory_reserved(device))
    owner = None
    calibration_error: Exception | None = None
    try:
        owner = _allocate_owner(torch, loaded, 1, temporal_k)
        owner.attempt_authority_binding = baseline_authority
        _initialize_owner(
            loaded,
            owner,
            world_uid_base=109_000,
            start_tick=0,
            rng_key=57,
            fixture=False,
        )
        torch.cuda.synchronize()
        tensor_bytes = (
            loaded.layout.state_bytes_per_world
            + loaded.layout.scratch_bytes_per_world
            + loaded.layout.event_bytes_per_world_tick * temporal_k
            + 4
        )
        reserved_after_owner = int(torch.cuda.memory_reserved(device))
        allocator_observed_reserved_overhead = max(
            0, reserved_after_owner - reserved_baseline - tensor_bytes
        )
        allocator_explicit_bound_floor = workhorse_allocator_owner_overhead_bound_v2(0)
        allocator_overhead = workhorse_allocator_owner_overhead_bound_v2(
            allocator_observed_reserved_overhead
        )
        graph_object_bytes = 0
        graph_exec_bytes = 0
        graph_inspection = None
        if loaded.backend.backend_id == BACKEND_PHASE_GRAPH_V2:
            graph_inspection = _prepare_graph_if_needed(loaded, owner, temporal_k, 0)
            torch.cuda.synchronize()
            for field_name in (
                "graph_object_device_bytes",
                "graph_exec_device_bytes",
            ):
                value = graph_inspection.get(field_name)
                if type(value) is not int or value < 0:
                    raise FullEvolutionWorkhorseProbeError(
                        f"runtime graph owner metric {field_name} is invalid"
                    )
            graph_object_bytes = graph_inspection["graph_object_device_bytes"]
            graph_exec_bytes = graph_inspection["graph_exec_device_bytes"]
        owner_authority = _resolve_contract_authority_v2(
            loaded,
            contract=calibration_contract,
            phase="CALIBRATION",
            validator=attempt_authority_validator,
            device_uuid=device_uuid,
            prior_lineage_sha256=lineage,
        )
        if owner_authority["attempt_lineage_sha256"] != lineage:
            raise FullEvolutionWorkhorseProbeError(
                "calibration authority changed Attempt lineage"
            )
        process_with_owner = _authority_accounted_device_bytes_v2(owner_authority)
        accounted_owner_bytes = (
            tensor_bytes + allocator_overhead + graph_object_bytes + graph_exec_bytes
        )
        driver_unattributed_owner_bytes = max(
            0,
            process_with_owner - process_baseline - accounted_owner_bytes,
        )
    except Exception as exc:
        calibration_error = exc
    finally:
        try:
            _release_graph(loaded)
            if owner is not None:
                del owner
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as exc:
            if calibration_error is None:
                calibration_error = exc
    closeout_authority = _resolve_contract_authority_v2(
        loaded,
        contract=calibration_contract,
        phase="CLOSEOUT",
        validator=attempt_authority_validator,
        device_uuid=device_uuid,
        prior_lineage_sha256=lineage,
    )
    if closeout_authority["attempt_lineage_sha256"] != lineage:
        raise FullEvolutionWorkhorseProbeError(
            "calibration closeout changed Attempt lineage"
        )
    if calibration_error is not None:
        raise calibration_error
    if any(
        binding["response"]["device_total_bytes"] != total_device_bytes
        for binding in (baseline_authority, owner_authority, closeout_authority)
    ):
        raise FullEvolutionWorkhorseProbeError(
            "calibration authority device total differs from inspected runtime"
        )
    body = {
        "schema_version": "eiel_workhorse_owner_component_measurement_v2",
        "artifact_sha256": file_sha256(loaded.artifact),
        "backend_id": loaded.backend.backend_id,
        "temporal_k": temporal_k,
        "process_baseline_bytes": process_baseline,
        "process_with_owner_bytes": process_with_owner,
        "tensor_resident_bytes": tensor_bytes,
        "allocator_reserved_baseline_bytes": reserved_baseline,
        "allocator_reserved_with_owner_bytes": reserved_after_owner,
        "allocator_observed_reserved_overhead_bytes": (
            allocator_observed_reserved_overhead
        ),
        "allocator_explicit_bound_floor_bytes": allocator_explicit_bound_floor,
        "allocator_owner_overhead_bytes": allocator_overhead,
        "driver_unattributed_owner_bytes": driver_unattributed_owner_bytes,
        "graph_object_bytes": graph_object_bytes,
        "graph_exec_bytes": graph_exec_bytes,
        "graph_inspection": graph_inspection,
        "measurement_source": "injected_attempt_bound_process_or_lease_accounting_v2",
        "accounting_mode": baseline_authority["response"]["accounting_mode"],
        "baseline_attempt_authority": baseline_authority,
        "owner_attempt_authority": owner_authority,
        "closeout_attempt_authority": closeout_authority,
        "attempt_lineage_sha256": lineage,
    }
    receipt_sha256 = canonical_sha256(body)
    components = WorkhorseOwnerComponentsV2(
        backend_id=loaded.backend.backend_id,
        temporal_k=temporal_k,
        graph_object_bytes=graph_object_bytes,
        graph_exec_bytes=graph_exec_bytes,
        allocator_owner_overhead_bytes=allocator_overhead,
        driver_unattributed_owner_bytes=driver_unattributed_owner_bytes,
        process_baseline_bytes=process_baseline,
        checkpoint_device_staging_bytes_per_world=0,
        measurement_receipt_sha256=receipt_sha256,
    )
    return components, {**body, "measurement_receipt_sha256": receipt_sha256}


def _write_sealed_json_v2(value: Mapping[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FullEvolutionWorkhorseProbeError("receipt output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o444)
    return output


def classify_workhorse_tuning_plan_v2(
    *,
    B_candidates: Sequence[int] | None,
    temporal_k_candidates: Sequence[int] | None,
) -> str:
    """Only a live-derived, unmodified B/K envelope may be authoritative."""
    if B_candidates is None and temporal_k_candidates is None:
        return "CANONICAL_LIVE_DERIVED_COMPLETE"
    return "EXPLICIT_PARTIAL_PRELIMINARY"


def validate_workhorse_scaling_receipt_v2(
    receipt_path: str | Path,
    *,
    expected_backend_id: str,
) -> dict[str, Any]:
    """Reopen and recompute one complete per-mode scaling receipt."""
    supplied = Path(receipt_path)
    if supplied.is_symlink() or not supplied.is_file():
        raise FullEvolutionWorkhorseProbeError(
            "scaling receipt must be a regular non-symlink"
        )
    path = supplied.resolve()
    if path.stat().st_mode & 0o222:
        raise FullEvolutionWorkhorseProbeError("scaling receipt must be read-only")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullEvolutionWorkhorseProbeError(
            f"cannot reopen scaling receipt: {exc}"
        ) from exc
    if type(value) is not dict:
        raise FullEvolutionWorkhorseProbeError("scaling receipt must be an object")
    body = {key: item for key, item in value.items() if key != "receipt_content_sha256"}
    if (
        value.get("schema_version")
        != "eiel_full_evolution_cuda_workhorse_scaling_receipt_v2"
        or value.get("receipt_content_sha256") != canonical_sha256(body)
        or value.get("governing_contracts")
        != workhorse_governing_contract_bindings_v2()
    ):
        raise FullEvolutionWorkhorseProbeError(
            "scaling receipt schema/source seal is invalid"
        )
    artifact_row = value.get("artifact")
    if type(artifact_row) is not dict:
        raise FullEvolutionWorkhorseProbeError("scaling artifact row is missing")
    artifact, _, manifest = validate_workhorse_prebuilt_v2(artifact_row.get("path"))
    if (
        manifest["backend_id"] != expected_backend_id
        or artifact_row.get("path") != str(artifact)
        or artifact_row.get("sha256") != file_sha256(artifact)
        or artifact_row.get("manifest_sha256") != manifest["manifest_sha256"]
        or artifact_row.get("identity") != manifest["identity"]
        or artifact_row.get("build_identity") != manifest["build_identity"]
        or artifact_row.get("sealed_artifact_identity")
        != manifest["sealed_artifact_identity"]
    ):
        raise FullEvolutionWorkhorseProbeError(
            "scaling artifact identity does not reopen"
        )
    shape = workhorse_shape_v2(manifest["shape_id"])
    if value.get("shape") != shape.identity_fields():
        raise FullEvolutionWorkhorseProbeError("scaling shape identity changed")
    layout = workhorse_layout_v2(shape)
    if value.get("layout") != layout.to_dict():
        raise FullEvolutionWorkhorseProbeError("scaling layout identity changed")
    runtime = value.get("runtime")
    device_identity = value.get("device_identity")
    if (
        type(runtime) is not dict
        or runtime.get("runtime_inspected") is not True
        or runtime.get("backend_id") != expected_backend_id
        or runtime.get("profile_id") != shape.profile_id
        or runtime.get("route_id") != shape.route_id
        or type(runtime.get("multiprocessor_count")) is not int
        or runtime["multiprocessor_count"] <= 0
        or any(
            type(runtime.get(field)) is not int or runtime[field] <= 0
            for field in (
                "maximum_grid_x",
                "maximum_grid_y",
                "maximum_threads_per_block",
                "maximum_threads_per_sm",
                "registers_per_sm",
                "shared_memory_per_sm",
                "maximum_blocks_per_sm",
                "warp_size",
            )
        )
        or type(device_identity) is not dict
        or runtime.get("device_uuid") != device_identity.get("uuid")
        or runtime.get("target_sm") != device_identity.get("sm")
        or runtime.get("multiprocessor_count")
        != device_identity.get("multiprocessor_count")
        or runtime.get("total_device_bytes")
        != device_identity.get("total_memory_bytes")
        or value.get("compiled_kernel_resources")
        != runtime.get("compiled_kernel_resources")
        or value.get("backend_topology")
        != workhorse_backend_v2(expected_backend_id).topology_payload(shape)
    ):
        raise FullEvolutionWorkhorseProbeError(
            "scaling runtime/topology evidence is invalid"
        )
    try:
        micro = validate_authoritative_micro_denominator_v2(
            value["authoritative_micro_denominator"]["receipt_path"],
            expected_device_identity=device_identity,
        )
        semantic = validate_workhorse_semantic_gate_receipt_v2(
            value["semantic_gate"]["receipt_path"],
            expected_shape_id=shape.shape_id,
            expected_candidate_artifact_sha256=file_sha256(artifact),
            expected_candidate_backend_id=expected_backend_id,
        )
    except (KeyError, FullEvolutionWorkhorseAuthorityError) as exc:
        raise FullEvolutionWorkhorseProbeError(
            f"scaling prerequisite authority does not reopen: {exc}"
        ) from exc
    if (
        value.get("authoritative_micro_denominator") != micro
        or value.get("semantic_gate") != semantic
        or semantic.get("runtime_device_identity")
        != {
            "device_uuid": runtime["device_uuid"],
            "target_sm": runtime["target_sm"],
            "multiprocessor_count": runtime["multiprocessor_count"],
            "total_device_bytes": runtime["total_device_bytes"],
        }
    ):
        raise FullEvolutionWorkhorseProbeError(
            "scaling prerequisite projection changed"
        )
    warmup_dispatches = value.get("warmup_dispatches")
    measured_dispatches = value.get("measured_dispatches")
    if (
        type(warmup_dispatches) is not int
        or warmup_dispatches < 1
        or type(measured_dispatches) is not int
        or measured_dispatches < 5
        or value.get("timing_source") != "cuda_event_exact_dispatch_v2"
        or value.get("sink_boundary")
        != "external_exact_content_ack_outside_timed_interval"
    ):
        raise FullEvolutionWorkhorseProbeError(
            "scaling timing/acknowledgement protocol is invalid"
        )

    measurements = value.get("owner_component_measurements")
    if type(measurements) is not dict or set(measurements) != {
        str(temporal_k) for temporal_k in shape.temporal_k_candidates
    }:
        raise FullEvolutionWorkhorseProbeError(
            "owner measurements do not cover every compiled temporal K"
        )
    owner_components_by_k: dict[int, WorkhorseOwnerComponentsV2] = {}
    for temporal_k in shape.temporal_k_candidates:
        measurement = measurements[str(temporal_k)]
        if type(measurement) is not dict:
            raise FullEvolutionWorkhorseProbeError(
                "owner measurement must be an object"
            )
        measurement_body = {
            key: item
            for key, item in measurement.items()
            if key != "measurement_receipt_sha256"
        }
        receipt_sha = measurement.get("measurement_receipt_sha256")
        if (
            measurement.get("schema_version")
            != "eiel_workhorse_owner_component_measurement_v2"
            or receipt_sha != canonical_sha256(measurement_body)
            or measurement.get("artifact_sha256") != file_sha256(artifact)
            or measurement.get("backend_id") != expected_backend_id
            or measurement.get("temporal_k") != temporal_k
            or measurement.get("measurement_source")
            != "injected_attempt_bound_process_or_lease_accounting_v2"
            or measurement.get("allocator_observed_reserved_overhead_bytes")
            != max(
                0,
                measurement.get("allocator_reserved_with_owner_bytes", -1)
                - measurement.get("allocator_reserved_baseline_bytes", -1)
                - measurement.get("tensor_resident_bytes", -1),
            )
            or measurement.get("allocator_explicit_bound_floor_bytes")
            != workhorse_allocator_owner_overhead_bound_v2(0)
            or measurement.get("allocator_owner_overhead_bytes")
            != workhorse_allocator_owner_overhead_bound_v2(
                measurement.get("allocator_observed_reserved_overhead_bytes", -1)
            )
            or type(measurement.get("process_with_owner_bytes")) is not int
            or measurement["process_with_owner_bytes"]
            < measurement.get("process_baseline_bytes", -1)
            or measurement.get("driver_unattributed_owner_bytes")
            != max(
                0,
                measurement["process_with_owner_bytes"]
                - measurement["process_baseline_bytes"]
                - measurement["tensor_resident_bytes"]
                - measurement["allocator_owner_overhead_bytes"]
                - measurement["graph_object_bytes"]
                - measurement["graph_exec_bytes"],
            )
        ):
            raise FullEvolutionWorkhorseProbeError(
                "owner measurement seal/source binding is invalid"
            )
        try:
            baseline_authority = validate_workhorse_attempt_authority_binding_v2(
                measurement.get("baseline_attempt_authority"), historical=True
            )
            owner_authority = validate_workhorse_attempt_authority_binding_v2(
                measurement.get("owner_attempt_authority"), historical=True
            )
            closeout_authority = validate_workhorse_attempt_authority_binding_v2(
                measurement.get("closeout_attempt_authority"), historical=True
            )
        except WorkhorseAttemptAuthorityError as exc:
            raise FullEvolutionWorkhorseProbeError(
                f"owner measurement Attempt authority is invalid: {exc}"
            ) from exc
        lineage = baseline_authority["attempt_lineage_sha256"]
        calibration_contract = FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape.shape_id,
            backend_id=expected_backend_id,
            B=1,
            temporal_k=temporal_k,
            total_ticks=temporal_k,
        )
        if (
            baseline_authority["request"].get("phase") != "CALIBRATION"
            or owner_authority["request"].get("phase") != "CALIBRATION"
            or closeout_authority["request"].get("phase") != "CLOSEOUT"
            or baseline_authority["request"].get("contract")
            != calibration_contract.to_dict()
            or owner_authority["request"].get("contract")
            != calibration_contract.to_dict()
            or closeout_authority["request"].get("contract")
            != calibration_contract.to_dict()
            or baseline_authority["request"].get("prior_lineage_sha256") is not None
            or owner_authority["request"].get("prior_lineage_sha256") != lineage
            or closeout_authority["request"].get("prior_lineage_sha256") != lineage
            or any(
                binding["request"].get("candidate_envelope_sha256") is not None
                for binding in (
                    baseline_authority,
                    owner_authority,
                    closeout_authority,
                )
            )
            or owner_authority["attempt_lineage_sha256"] != lineage
            or closeout_authority["attempt_lineage_sha256"] != lineage
            or measurement.get("attempt_lineage_sha256") != lineage
            or measurement.get("accounting_mode")
            != baseline_authority["response"]["accounting_mode"]
            or owner_authority["response"]["accounting_mode"]
            != measurement.get("accounting_mode")
            or closeout_authority["response"]["accounting_mode"]
            != measurement.get("accounting_mode")
            or measurement.get("process_baseline_bytes")
            != _authority_accounted_device_bytes_v2(baseline_authority)
            or measurement.get("process_with_owner_bytes")
            != _authority_accounted_device_bytes_v2(owner_authority)
        ):
            raise FullEvolutionWorkhorseProbeError(
                "owner measurement phase/lineage/accounting binding changed"
            )
        _validate_graph_inspection_v2(
            measurement.get("graph_inspection"),
            shape=shape,
            backend_id=expected_backend_id,
            B=1,
            temporal_k=temporal_k,
            disable_mask=0,
        )
        graph_inspection = measurement.get("graph_inspection")
        if expected_backend_id == BACKEND_PHASE_GRAPH_V2 and (
            measurement.get("graph_object_bytes")
            != graph_inspection.get("graph_object_device_bytes")
            or measurement.get("graph_exec_bytes")
            != graph_inspection.get("graph_exec_device_bytes")
        ):
            raise FullEvolutionWorkhorseProbeError(
                "owner graph byte measurements do not match graph inspection"
            )
        owner_components_by_k[temporal_k] = WorkhorseOwnerComponentsV2(
            backend_id=expected_backend_id,
            temporal_k=temporal_k,
            graph_object_bytes=measurement["graph_object_bytes"],
            graph_exec_bytes=measurement["graph_exec_bytes"],
            allocator_owner_overhead_bytes=measurement[
                "allocator_owner_overhead_bytes"
            ],
            driver_unattributed_owner_bytes=measurement[
                "driver_unattributed_owner_bytes"
            ],
            process_baseline_bytes=measurement["process_baseline_bytes"],
            checkpoint_device_staging_bytes_per_world=0,
            measurement_receipt_sha256=receipt_sha,
        )

    topology_rows = value.get("topology_resource_bindings_by_k")
    if type(topology_rows) is not dict or set(topology_rows) != {
        str(temporal_k) for temporal_k in shape.temporal_k_candidates
    }:
        raise FullEvolutionWorkhorseProbeError(
            "topology/resource bindings do not cover every compiled K"
        )
    topology_resource_bindings_by_k: dict[int, dict[str, Any]] = {}
    topology_device_fields = (
        "multiprocessor_count",
        "maximum_grid_x",
        "maximum_grid_y",
        "maximum_threads_per_block",
        "maximum_threads_per_sm",
        "registers_per_sm",
        "shared_memory_per_sm",
        "maximum_blocks_per_sm",
        "warp_size",
    )
    for temporal_k in shape.temporal_k_candidates:
        try:
            topology = validate_workhorse_topology_resource_binding_v2(
                topology_rows[str(temporal_k)],
                shape=shape,
                backend_id=expected_backend_id,
                temporal_k=temporal_k,
            )
        except Exception as exc:
            raise FullEvolutionWorkhorseProbeError(
                f"topology/resource binding K{temporal_k} is invalid: {exc}"
            ) from exc
        graph_geometry = topology["graph_geometry"]
        measurement_graph = measurements[str(temporal_k)]["graph_inspection"]
        if (
            topology["device_limits"]
            != {field: runtime[field] for field in topology_device_fields}
            or topology["compiled_kernel_resources"]
            != runtime["compiled_kernel_resources"]
            or (
                expected_backend_id == BACKEND_PHASE_GRAPH_V2
                and (
                    graph_geometry["nodes_sha256"]
                    != canonical_sha256(measurement_graph["nodes"])
                    or graph_geometry["graph_object_device_bytes"]
                    != measurement_graph["graph_object_device_bytes"]
                    or graph_geometry["graph_exec_device_bytes"]
                    != measurement_graph["graph_exec_device_bytes"]
                )
            )
            or (
                expected_backend_id != BACKEND_PHASE_GRAPH_V2
                and (graph_geometry is not None or measurement_graph is not None)
            )
        ):
            raise FullEvolutionWorkhorseProbeError(
                f"topology/resource binding K{temporal_k} differs from live calibration"
            )
        topology_resource_bindings_by_k[temporal_k] = topology
    envelope = value.get("candidate_envelope")
    if type(envelope) is not dict:
        raise FullEvolutionWorkhorseProbeError("candidate envelope is missing")
    try:
        envelope_authority = validate_workhorse_attempt_authority_binding_v2(
            value.get("candidate_envelope_attempt_authority"), historical=True
        )
    except WorkhorseAttemptAuthorityError as exc:
        raise FullEvolutionWorkhorseProbeError(
            f"candidate envelope Attempt authority is invalid: {exc}"
        ) from exc
    final_calibration = measurements[str(shape.temporal_k_candidates[-1])]
    if (
        envelope_authority != final_calibration["closeout_attempt_authority"]
        or envelope_authority["request"].get("phase") != "CLOSEOUT"
        or envelope_authority["response"].get("device_free_bytes")
        != envelope.get("free_bytes")
        or envelope_authority["response"].get("device_total_bytes")
        != envelope.get("total_device_bytes")
        or envelope_authority["response"].get("device_total_bytes")
        != runtime["total_device_bytes"]
    ):
        raise FullEvolutionWorkhorseProbeError(
            "candidate envelope is not the final live exclusive calibration observation"
        )
    expected_envelope = workhorse_candidate_envelope_v2(
        shape,
        backend_id=expected_backend_id,
        free_bytes=envelope.get("free_bytes"),
        total_device_bytes=envelope.get("total_device_bytes"),
        owner_components_by_k=owner_components_by_k,
        topology_resource_bindings_by_k=topology_resource_bindings_by_k,
        live_observation_authority_binding_sha256=envelope_authority[
            "authority_binding_sha256"
        ],
        reserve_bytes=envelope.get("reserve_bytes_requested"),
    )
    for key, expected in expected_envelope.items():
        if envelope.get(key) != expected:
            raise FullEvolutionWorkhorseProbeError(
                f"candidate envelope does not recompute at {key}"
            )
    expected_pairs = [
        (k_row["temporal_k"], B)
        for k_row in expected_envelope["per_k"]
        for B in k_row["B_candidates"]
    ]
    rows = value.get("rows")
    if (
        envelope.get("plan_mode") != "CANONICAL_LIVE_DERIVED_COMPLETE"
        or envelope.get("complete_cartesian_rows") != len(expected_pairs)
        or envelope.get("measured_or_rejected_rows") != len(expected_pairs)
        or type(rows) is not list
        or [(row.get("temporal_k"), row.get("B")) for row in rows] != expected_pairs
    ):
        raise FullEvolutionWorkhorseProbeError(
            "scaling receipt is not the complete canonical B/K envelope"
        )

    ideal = micro["p10_world_ticks_per_second"] / (shape.sites / micro["sites"])
    if (
        value.get("surface_scale") != shape.sites / micro["sites"]
        or value.get("ideal_linear_world_ticks_per_second") != ideal
        or value.get("performance_viability_threshold") != 0.25
    ):
        raise FullEvolutionWorkhorseProbeError(
            "scaling baseline or viability threshold changed"
        )
    measured_rows = []
    governing = workhorse_governing_contract_bindings_v2()
    for row in rows:
        temporal_k = row["temporal_k"]
        B = row["B"]
        contract = FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape.shape_id,
            backend_id=expected_backend_id,
            B=B,
            temporal_k=temporal_k,
            total_ticks=(warmup_dispatches + measured_dispatches) * temporal_k,
        )
        components = owner_components_by_k[temporal_k]
        ledger = estimate_workhorse_resources_v2(contract, owner_components=components)
        admission = admit_workhorse_vram_v2(
            contract,
            free_bytes=envelope["free_bytes"],
            total_device_bytes=envelope["total_device_bytes"],
            owner_components=components,
            reserve_bytes=envelope["reserve_bytes_requested"],
        )
        if (
            row.get("resource_ledger") != ledger.to_dict()
            or row.get("admission") != admission.to_dict()
            or row.get("free_vram_before_bytes") != envelope["free_bytes"]
            or row.get("total_vram_bytes") != envelope["total_device_bytes"]
        ):
            raise FullEvolutionWorkhorseProbeError(
                "scaling row identity/admission does not recompute"
            )
        if (
            row.get("site_ctas_per_world") != runtime["site_ctas_per_world"]
            or row.get("agent_ctas_per_world") != runtime["agent_ctas_per_world"]
            or row.get("environment_ctas_per_world")
            != runtime["environment_ctas_per_world"]
        ):
            raise FullEvolutionWorkhorseProbeError("scaling row CTA geometry changed")
        samples = row.get("world_ticks_per_second_samples")
        if samples is None:
            if not admission.admitted:
                try:
                    rejection = validate_workhorse_attempt_authority_binding_v2(
                        row.get("plan_rejection_authority"), historical=True
                    )
                except WorkhorseAttemptAuthorityError as exc:
                    raise FullEvolutionWorkhorseProbeError(
                        f"plan rejection authority is invalid: {exc}"
                    ) from exc
                if (
                    row.get("outcome") != "REJECTED_VRAM"
                    or row.get("attempt_identity") is not None
                    or rejection["request"].get("phase") != "PLAN_REJECTION"
                    or rejection["request"].get("contract") != contract.to_dict()
                    or rejection["request"].get("candidate_envelope_sha256")
                    != envelope["candidate_envelope_sha256"]
                    or rejection["request"].get("prior_lineage_sha256") is not None
                    or rejection["request"].get("source_checkpoint_identity_sha256")
                    is not None
                    or rejection["response"].get("execution_authority_granted")
                    is not False
                    or row.get("execution_attempt_authority") is not None
                    or row.get("closeout_attempt_authority") is not None
                    or row.get("execution_authority_refreshes") is not None
                    or row.get("attempt_admission") is not None
                ):
                    raise FullEvolutionWorkhorseProbeError(
                        "non-admitted row did not remain PREPARED/no-effect"
                    )
                continue
            if admission.admitted and row.get("outcome") != "RUNTIME_REJECTED":
                raise FullEvolutionWorkhorseProbeError(
                    "admitted unmeasured row is not a runtime rejection"
                )
            try:
                execution_authority = validate_workhorse_attempt_authority_binding_v2(
                    row.get("execution_attempt_authority"), historical=True
                )
                closeout_authority = validate_workhorse_attempt_authority_binding_v2(
                    row.get("closeout_attempt_authority"), historical=True
                )
                refreshes = [
                    validate_workhorse_attempt_authority_binding_v2(
                        binding, historical=True
                    )
                    for binding in row.get("execution_authority_refreshes", [])
                ]
            except (TypeError, WorkhorseAttemptAuthorityError) as exc:
                raise FullEvolutionWorkhorseProbeError(
                    f"runtime-rejected row Attempt authority is invalid: {exc}"
                ) from exc
            lineage = execution_authority["attempt_lineage_sha256"]
            if (
                type(row.get("execution_authority_refreshes")) is not list
                or execution_authority["request"].get("phase") != "EXECUTION_ROW"
                or execution_authority["request"].get("contract") != contract.to_dict()
                or execution_authority["request"].get("candidate_envelope_sha256")
                != envelope["candidate_envelope_sha256"]
                or closeout_authority["request"].get("phase") != "CLOSEOUT"
                or closeout_authority["request"].get("contract") != contract.to_dict()
                or closeout_authority["request"].get("candidate_envelope_sha256")
                != envelope["candidate_envelope_sha256"]
                or closeout_authority["request"].get("prior_lineage_sha256") != lineage
                or closeout_authority["attempt_lineage_sha256"] != lineage
                or closeout_authority["response"].get("device_total_bytes")
                != runtime["total_device_bytes"]
                or any(
                    binding["request"].get("phase") != "EXECUTION_ROW"
                    or binding["request"].get("contract") != contract.to_dict()
                    or binding["request"].get("candidate_envelope_sha256")
                    != envelope["candidate_envelope_sha256"]
                    or binding["request"].get("prior_lineage_sha256") != lineage
                    or binding["attempt_lineage_sha256"] != lineage
                    or binding["response"].get("device_total_bytes")
                    != runtime["total_device_bytes"]
                    for binding in refreshes
                )
                or type(row.get("error")) is not str
                or not row["error"]
                or "plan_rejection_authority" in row
                or "closeout_error" in row
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "runtime-rejected row phase/lineage binding changed"
                )
            attempt_admission = admit_workhorse_vram_v2(
                contract,
                free_bytes=execution_authority["response"]["device_free_bytes"],
                total_device_bytes=execution_authority["response"][
                    "device_total_bytes"
                ],
                owner_components=components,
                reserve_bytes=envelope["reserve_bytes_requested"],
            )
            if (
                execution_authority["response"]["device_total_bytes"]
                != runtime["total_device_bytes"]
                or row.get("attempt_admission") != attempt_admission.to_dict()
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "runtime-rejected row live VRAM admission changed"
                )
            attempt = workhorse_attempt_identity_v2(
                contract,
                artifact_sha256=file_sha256(artifact),
                sealed_artifact_identity_sha256=manifest["sealed_artifact_identity"][
                    "sealed_artifact_identity_sha256"
                ],
                device_uuid=runtime["device_uuid"],
                target_sm=runtime["target_sm"],
                resource_ledger=ledger,
                admission=attempt_admission,
                governing_contracts=governing,
                attempt_authority_binding=execution_authority,
                historical_authority=True,
            )
            if row.get("attempt_identity") != attempt:
                raise FullEvolutionWorkhorseProbeError(
                    "runtime-rejected row Attempt identity changed"
                )
            continue
        try:
            execution_authority = validate_workhorse_attempt_authority_binding_v2(
                row.get("execution_attempt_authority"), historical=True
            )
            closeout_authority = validate_workhorse_attempt_authority_binding_v2(
                row.get("closeout_attempt_authority"), historical=True
            )
        except WorkhorseAttemptAuthorityError as exc:
            raise FullEvolutionWorkhorseProbeError(
                f"measured row Attempt authority is invalid: {exc}"
            ) from exc
        lineage = execution_authority["attempt_lineage_sha256"]
        attempt_admission = admit_workhorse_vram_v2(
            contract,
            free_bytes=execution_authority["response"]["device_free_bytes"],
            total_device_bytes=execution_authority["response"]["device_total_bytes"],
            owner_components=components,
            reserve_bytes=envelope["reserve_bytes_requested"],
        )
        refresh_rows = row.get("execution_authority_refreshes")
        if (
            type(refresh_rows) is not list
            or len(refresh_rows) != 1 + warmup_dispatches + measured_dispatches
        ):
            raise FullEvolutionWorkhorseProbeError(
                "measured row live authority refresh coverage is incomplete"
            )
        try:
            refreshes = [
                validate_workhorse_attempt_authority_binding_v2(
                    binding, historical=True
                )
                for binding in refresh_rows
            ]
        except WorkhorseAttemptAuthorityError as exc:
            raise FullEvolutionWorkhorseProbeError(
                f"measured row authority refresh is invalid: {exc}"
            ) from exc
        if (
            execution_authority["request"].get("phase") != "EXECUTION_ROW"
            or execution_authority["request"].get("contract") != contract.to_dict()
            or execution_authority["request"].get("candidate_envelope_sha256")
            != envelope["candidate_envelope_sha256"]
            or closeout_authority["request"].get("phase") != "CLOSEOUT"
            or closeout_authority["request"].get("contract") != contract.to_dict()
            or closeout_authority["request"].get("candidate_envelope_sha256")
            != envelope["candidate_envelope_sha256"]
            or closeout_authority["request"].get("prior_lineage_sha256") != lineage
            or closeout_authority["attempt_lineage_sha256"] != lineage
            or closeout_authority["response"].get("device_total_bytes")
            != runtime["total_device_bytes"]
            or execution_authority["response"].get("device_total_bytes")
            != runtime["total_device_bytes"]
            or row.get("attempt_admission") != attempt_admission.to_dict()
            or attempt_admission.admitted is not True
            or any(
                binding["request"].get("phase") != "EXECUTION_ROW"
                or binding["request"].get("contract") != contract.to_dict()
                or binding["request"].get("candidate_envelope_sha256")
                != envelope["candidate_envelope_sha256"]
                or binding["request"].get("prior_lineage_sha256") != lineage
                or binding["attempt_lineage_sha256"] != lineage
                or binding["response"].get("device_total_bytes")
                != runtime["total_device_bytes"]
                for binding in refreshes
            )
        ):
            raise FullEvolutionWorkhorseProbeError(
                "measured row phase/lineage binding changed"
            )
        attempt = workhorse_attempt_identity_v2(
            contract,
            artifact_sha256=file_sha256(artifact),
            sealed_artifact_identity_sha256=manifest["sealed_artifact_identity"][
                "sealed_artifact_identity_sha256"
            ],
            device_uuid=runtime["device_uuid"],
            target_sm=runtime["target_sm"],
            resource_ledger=ledger,
            admission=attempt_admission,
            governing_contracts=governing,
            attempt_authority_binding=execution_authority,
            historical_authority=True,
        )
        if row.get("attempt_identity") != attempt:
            raise FullEvolutionWorkhorseProbeError(
                "measured row Attempt identity changed"
            )
        warmup_acks = row.get("warmup_sink_acknowledgements")
        sample_evidence = row.get("measurement_samples")
        if (
            type(samples) is not list
            or len(samples) != measured_dispatches
            or any(
                type(sample) not in (int, float)
                or not math.isfinite(float(sample))
                or sample <= 0
                for sample in samples
            )
            or type(warmup_acks) is not list
            or len(warmup_acks) != warmup_dispatches
            or type(sample_evidence) is not list
            or len(sample_evidence) != measured_dispatches
        ):
            raise FullEvolutionWorkhorseProbeError(
                "scaling CUDA-event sample/acknowledgement set is invalid"
            )
        expected_next_tick = 0
        expected_world_uids = list(range(110_000, 110_000 + B))
        previous_post_state_sha256 = None
        expected_rng_identity_sha256 = None
        for warmup_index, acknowledgement in enumerate(warmup_acks):
            if type(acknowledgement) is not dict:
                raise FullEvolutionWorkhorseProbeError(
                    "warmup sink acknowledgement is invalid"
                )
            reopened_ack = validate_workhorse_event_sink_receipt_v2(
                acknowledgement.get("receipt_path"),
                expected_artifact_identity_sha256=manifest["identity"][
                    "identity_sha256"
                ],
                expected_end_next_absolute_tick=expected_next_tick + temporal_k,
            )
            if (
                acknowledgement != reopened_ack
                or acknowledgement.get("artifact_sha256") != file_sha256(artifact)
                or acknowledgement.get("manifest_sha256") != manifest["manifest_sha256"]
                or acknowledgement.get("world_uids") != expected_world_uids
                or acknowledgement.get("start_next_absolute_tick") != expected_next_tick
                or acknowledgement.get("tick_count") != temporal_k
                or acknowledgement.get("attempt_lineage_sha256") != lineage
                or acknowledgement.get("attempt_authority_binding_sha256")
                != refreshes[1 + warmup_index]["authority_binding_sha256"]
                or (
                    previous_post_state_sha256 is not None
                    and acknowledgement.get("pre_state_sha256")
                    != previous_post_state_sha256
                )
                or (
                    expected_rng_identity_sha256 is not None
                    and acknowledgement.get("rng_identity_sha256")
                    != expected_rng_identity_sha256
                )
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "warmup sink acknowledgement chain does not reopen"
                )
            previous_post_state_sha256 = acknowledgement["post_state_sha256"]
            expected_rng_identity_sha256 = acknowledgement["rng_identity_sha256"]
            expected_next_tick += temporal_k
        derived_rates = []
        for sample_index, sample_row in enumerate(sample_evidence):
            if type(sample_row) is not dict:
                raise FullEvolutionWorkhorseProbeError(
                    "CUDA-event measurement evidence is invalid"
                )
            sample_body = {
                key: item
                for key, item in sample_row.items()
                if key != "measurement_sha256"
            }
            elapsed_ms = sample_row.get("cuda_event_elapsed_ms")
            committed_ticks = sample_row.get("committed_world_ticks")
            acknowledgement = sample_row.get("event_sink_ack")
            if (
                sample_row.get("measurement_sha256") != canonical_sha256(sample_body)
                or sample_row.get("sample_index") != sample_index
                or sample_row.get("timing_source") != "cuda_event"
                or type(elapsed_ms) not in (int, float)
                or not math.isfinite(float(elapsed_ms))
                or elapsed_ms <= 0
                or committed_ticks != B * temporal_k
                or type(acknowledgement) is not dict
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "CUDA-event measurement evidence seal is invalid"
                )
            reopened_ack = validate_workhorse_event_sink_receipt_v2(
                acknowledgement.get("receipt_path"),
                expected_artifact_identity_sha256=manifest["identity"][
                    "identity_sha256"
                ],
                expected_end_next_absolute_tick=expected_next_tick + temporal_k,
            )
            rate = committed_ticks * 1000.0 / float(elapsed_ms)
            if (
                acknowledgement != reopened_ack
                or acknowledgement.get("artifact_sha256") != file_sha256(artifact)
                or acknowledgement.get("manifest_sha256") != manifest["manifest_sha256"]
                or acknowledgement.get("world_uids") != expected_world_uids
                or acknowledgement.get("start_next_absolute_tick") != expected_next_tick
                or acknowledgement.get("tick_count") != temporal_k
                or acknowledgement.get("attempt_lineage_sha256") != lineage
                or acknowledgement.get("attempt_authority_binding_sha256")
                != refreshes[1 + warmup_dispatches + sample_index][
                    "authority_binding_sha256"
                ]
                or acknowledgement.get("pre_state_sha256") != previous_post_state_sha256
                or acknowledgement.get("rng_identity_sha256")
                != expected_rng_identity_sha256
                or sample_row.get("world_ticks_per_second") != rate
            ):
                raise FullEvolutionWorkhorseProbeError(
                    "CUDA-event measurement/sink frontier does not recompute"
                )
            derived_rates.append(rate)
            previous_post_state_sha256 = acknowledgement["post_state_sha256"]
            expected_next_tick += temporal_k
        if samples != derived_rates:
            raise FullEvolutionWorkhorseProbeError(
                "scaling rates do not derive from raw CUDA-event evidence"
            )
        _validate_graph_inspection_v2(
            row.get("graph_inspection"),
            shape=shape,
            backend_id=expected_backend_id,
            B=B,
            temporal_k=temporal_k,
            disable_mask=0,
        )
        p10 = _p10(samples)
        efficiency = p10 / ideal
        resource_observations = [
            execution_authority,
            *refreshes,
            closeout_authority,
        ]
        accounting_modes = {
            binding["response"]["accounting_mode"] for binding in resource_observations
        }
        expected_process_baseline = _authority_accounted_device_bytes_v2(
            execution_authority
        )
        expected_process_peak = max(
            _authority_accounted_device_bytes_v2(binding)
            for binding in resource_observations
        )
        process_peak = row.get("driver_process_peak_bytes")
        process_baseline = row.get("driver_process_baseline_bytes")
        process_peak_delta = row.get("driver_process_peak_delta_bytes")
        observed_free_delta = row.get("observed_free_vram_delta_bytes")
        free_after = row.get("free_vram_after_measurement_bytes")
        allocator_baseline = row.get("allocator_baseline_bytes")
        allocator_peak = row.get("allocator_peak_bytes")
        allocator_peak_delta = row.get("allocator_peak_delta_bytes")
        resource_ledger_conforms = (
            type(process_peak) is int
            and process_peak >= 0
            and len(accounting_modes) == 1
            and process_peak == expected_process_peak
            and type(process_baseline) is int
            and process_baseline >= 0
            and process_baseline == expected_process_baseline
            and process_peak >= process_baseline
            and process_peak_delta == process_peak - process_baseline
            and type(observed_free_delta) is int
            and observed_free_delta >= 0
            and type(free_after) is int
            and 0 <= free_after <= envelope["total_device_bytes"]
            and observed_free_delta == max(0, attempt_admission.free_bytes - free_after)
            and type(allocator_baseline) is int
            and allocator_baseline >= 0
            and type(allocator_peak) is int
            and allocator_peak >= allocator_baseline
            and allocator_peak_delta == allocator_peak - allocator_baseline
            and process_peak <= ledger.projected_process_peak_bytes
            and observed_free_delta <= ledger.incremental_device_bytes
        )
        expected_outcome = (
            "RESOURCE_LEDGER_REJECTED"
            if not resource_ledger_conforms
            else (
                "MODE_MEETS_25_PERCENT"
                if efficiency >= 0.25
                else "PERFORMANCE_RECOVERY"
            )
        )
        if (
            row.get("outcome") != expected_outcome
            or row.get("resource_ledger_conforms") is not resource_ledger_conforms
            or row.get("p10_world_ticks_per_second") != p10
            or row.get("p10_site_ticks_per_second") != p10 * shape.sites
            or row.get("p10_agent_capacity_ticks_per_second") != p10 * shape.capacity
            or row.get("ideal_linear_world_ticks_per_second") != ideal
            or row.get("normalized_site_throughput_efficiency") != efficiency
        ):
            raise FullEvolutionWorkhorseProbeError(
                "scaling metric/outcome does not recompute"
            )
        if resource_ledger_conforms:
            measured_rows.append(row)
    winner = max(
        measured_rows,
        key=lambda row: row["p10_world_ticks_per_second"],
        default=None,
    )
    if value.get("winner") != winner:
        raise FullEvolutionWorkhorseProbeError("scaling winner does not recompute")
    return value


def run_workhorse_tuning_v2(
    artifact_path: str | Path,
    *,
    authoritative_micro_receipt_path: str | Path,
    semantic_gate_receipt_path: str | Path,
    event_sink_root: str | Path,
    receipt_output_path: str | Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    B_candidates: Sequence[int] | None = None,
    temporal_k_candidates: Sequence[int] | None = None,
    warmup_dispatches: int = 3,
    measured_dispatches: int = 10,
) -> dict[str, Any]:
    """Retired direct entry; canonical one-row requests are required."""

    raise FullEvolutionWorkhorseProbeError(
        "direct Workhorse tuning is retired; use one canonical "
        "ExecutionRequestV2 and Scheduler Attempt per exact B/K row"
    )

    torch = _torch()
    loaded = _load(artifact_path, torch)
    if loaded.backend.backend_id == BACKEND_REFERENCE_PHASED_V2:
        raise FullEvolutionWorkhorseProbeError("tuning requires a performance backend")
    if type(warmup_dispatches) is not int or warmup_dispatches < 1:
        raise FullEvolutionWorkhorseProbeError("warmup_dispatches must be positive")
    if type(measured_dispatches) is not int or measured_dispatches < 5:
        raise FullEvolutionWorkhorseProbeError(
            "measured_dispatches must be at least five"
        )
    runtime = dict(loaded.module.runtime_inspect())
    from ..benchmark import cuda_device_identity_v2

    device_identity = cuda_device_identity_v2(torch, torch.cuda.current_device())
    try:
        micro = validate_authoritative_micro_denominator_v2(
            authoritative_micro_receipt_path,
            expected_device_identity=device_identity,
        )
        semantic_gate = validate_workhorse_semantic_gate_receipt_v2(
            semantic_gate_receipt_path,
            expected_shape_id=loaded.shape.shape_id,
            expected_candidate_artifact_sha256=file_sha256(loaded.artifact),
            expected_candidate_backend_id=loaded.backend.backend_id,
        )
    except FullEvolutionWorkhorseAuthorityError as exc:
        raise FullEvolutionWorkhorseProbeError(str(exc)) from exc
    live_semantic_identity = {
        "device_uuid": runtime.get("device_uuid"),
        "target_sm": runtime.get("target_sm"),
        "multiprocessor_count": runtime.get("multiprocessor_count"),
        "total_device_bytes": runtime.get("total_device_bytes"),
    }
    if (
        semantic_gate.get("runtime_device_identity") != live_semantic_identity
        or runtime.get("device_uuid") != device_identity.get("uuid")
        or runtime.get("target_sm") != device_identity.get("sm")
        or runtime.get("multiprocessor_count")
        != device_identity.get("multiprocessor_count")
        or runtime.get("total_device_bytes")
        != device_identity.get("total_memory_bytes")
    ):
        raise FullEvolutionWorkhorseProbeError(
            "micro, semantic gate, and tuning runtime are not the same device"
        )
    owner_components_by_k: dict[int, WorkhorseOwnerComponentsV2] = {}
    owner_component_measurements: dict[str, dict[str, Any]] = {}
    topology_resource_bindings_by_k: dict[int, dict[str, Any]] = {}
    for temporal_k in loaded.shape.temporal_k_candidates:
        components, measurement = _calibrate_owner_components_v2(
            loaded,
            torch,
            temporal_k,
            attempt_authority_validator=attempt_authority_validator,
            device_uuid=str(runtime["device_uuid"]),
            total_device_bytes=int(runtime["total_device_bytes"]),
        )
        owner_components_by_k[temporal_k] = components
        owner_component_measurements[str(temporal_k)] = measurement
        topology_resource_bindings_by_k[temporal_k] = (
            workhorse_topology_resource_binding_v2(
                loaded.shape,
                backend_id=loaded.backend.backend_id,
                temporal_k=temporal_k,
                runtime_inspection=runtime,
                graph_inspection=measurement["graph_inspection"],
            )
        )
    envelope_authority = owner_component_measurements[
        str(loaded.shape.temporal_k_candidates[-1])
    ]["closeout_attempt_authority"]
    free_vram = envelope_authority["response"]["device_free_bytes"]
    total_vram = envelope_authority["response"]["device_total_bytes"]
    envelope = workhorse_candidate_envelope_v2(
        loaded.shape,
        backend_id=loaded.backend.backend_id,
        free_bytes=int(free_vram),
        total_device_bytes=int(total_vram),
        owner_components_by_k=owner_components_by_k,
        topology_resource_bindings_by_k=topology_resource_bindings_by_k,
        live_observation_authority_binding_sha256=envelope_authority[
            "authority_binding_sha256"
        ],
    )
    plan_mode = classify_workhorse_tuning_plan_v2(
        B_candidates=B_candidates,
        temporal_k_candidates=temporal_k_candidates,
    )
    explicit_partial = plan_mode == "EXPLICIT_PARTIAL_PRELIMINARY"
    requested_b = None if B_candidates is None else tuple(B_candidates)
    requested_k = (
        tuple(loaded.shape.temporal_k_candidates)
        if temporal_k_candidates is None
        else tuple(temporal_k_candidates)
    )
    if (
        not requested_k
        or len(set(requested_k)) != len(requested_k)
        or any(value not in loaded.shape.temporal_k_candidates for value in requested_k)
        or (
            requested_b is not None
            and (
                not requested_b
                or len(set(requested_b)) != len(requested_b)
                or any(type(value) is not int or value <= 0 for value in requested_b)
            )
        )
    ):
        raise FullEvolutionWorkhorseProbeError("explicit B/K subset is invalid")

    p10_micro = float(micro["p10_world_ticks_per_second"])
    surface_scale = loaded.shape.sites / int(micro["sites"])
    ideal_linear_wtps = p10_micro / surface_scale
    rows: list[dict[str, Any]] = []
    for k_row in envelope["per_k"]:
        temporal_k = k_row["temporal_k"]
        if temporal_k not in requested_k:
            continue
        for B in k_row["B_candidates"]:
            if requested_b is not None and B not in requested_b:
                continue
            contract = FullEvolutionCudaWorkhorseContractV2(
                shape_id=loaded.shape.shape_id,
                backend_id=loaded.backend.backend_id,
                B=B,
                temporal_k=temporal_k,
                total_ticks=(warmup_dispatches + measured_dispatches) * temporal_k,
            )
            owner_components = owner_components_by_k[temporal_k]
            admission = admit_workhorse_vram_v2(
                contract,
                free_bytes=int(free_vram),
                total_device_bytes=int(total_vram),
                owner_components=owner_components,
            )
            ledger = estimate_workhorse_resources_v2(
                contract, owner_components=owner_components
            )
            row: dict[str, Any] = {
                "B": B,
                "temporal_k": temporal_k,
                "resource_ledger": ledger.to_dict(),
                "admission": admission.to_dict(),
                "free_vram_before_bytes": int(free_vram),
                "total_vram_bytes": int(total_vram),
                "site_ctas_per_world": runtime["site_ctas_per_world"],
                "agent_ctas_per_world": runtime["agent_ctas_per_world"],
                "environment_ctas_per_world": runtime["environment_ctas_per_world"],
            }
            if not admission.admitted:
                rejection_authority = _resolve_contract_authority_v2(
                    loaded,
                    contract=contract,
                    phase="PLAN_REJECTION",
                    validator=attempt_authority_validator,
                    device_uuid=str(runtime["device_uuid"]),
                    candidate_envelope_sha256=envelope["candidate_envelope_sha256"],
                )
                row["outcome"] = "REJECTED_VRAM"
                row["plan_rejection_authority"] = rejection_authority
                row["attempt_identity"] = None
                rows.append(row)
                continue
            allocator_before = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            owner = None
            execution_authority = None
            closeout_authority = None
            lineage = None
            authority_refreshes: list[dict[str, Any]] = []

            def release_row_owner() -> None:
                nonlocal owner
                _release_graph(loaded)
                owner = None
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            try:
                execution_authority = _resolve_contract_authority_v2(
                    loaded,
                    contract=contract,
                    phase="EXECUTION_ROW",
                    validator=attempt_authority_validator,
                    device_uuid=str(runtime["device_uuid"]),
                    candidate_envelope_sha256=envelope["candidate_envelope_sha256"],
                )
                lineage = execution_authority["attempt_lineage_sha256"]
                row["execution_attempt_authority"] = execution_authority
                if execution_authority["response"]["device_total_bytes"] != total_vram:
                    raise FullEvolutionWorkhorseProbeError(
                        "row execution device total differs from sealed envelope"
                    ) from WorkhorseAttemptAuthorityError(
                        "row execution device identity changed"
                    )
                attempt_admission = admit_workhorse_vram_v2(
                    contract,
                    free_bytes=execution_authority["response"]["device_free_bytes"],
                    total_device_bytes=execution_authority["response"][
                        "device_total_bytes"
                    ],
                    owner_components=owner_components,
                    reserve_bytes=envelope["reserve_bytes_requested"],
                )
                row["attempt_admission"] = attempt_admission.to_dict()
                row["attempt_identity"] = workhorse_attempt_identity_v2(
                    contract,
                    artifact_sha256=file_sha256(loaded.artifact),
                    sealed_artifact_identity_sha256=str(
                        loaded.manifest["sealed_artifact_identity"][
                            "sealed_artifact_identity_sha256"
                        ]
                    ),
                    device_uuid=str(runtime["device_uuid"]),
                    target_sm=str(runtime["target_sm"]),
                    resource_ledger=ledger,
                    admission=attempt_admission,
                    governing_contracts=workhorse_governing_contract_bindings_v2(),
                    attempt_authority_binding=execution_authority,
                )
                if not attempt_admission.admitted:
                    raise FullEvolutionWorkhorseProbeError(
                        "live execution Attempt lost required VRAM headroom"
                    )
                process_before = _authority_accounted_device_bytes_v2(
                    execution_authority
                )
                owner = _allocate_owner(torch, loaded, B, temporal_k)
                _initialize_owner(
                    loaded,
                    owner,
                    world_uid_base=110_000,
                    start_tick=0,
                    rng_key=59,
                    fixture=False,
                )
                owner.attempt_authority_binding = execution_authority
                graph_inspection = _prepare_graph_if_needed(
                    loaded, owner, temporal_k, 0
                )

                def refresh_execution_authority() -> dict[str, Any]:
                    binding = _resolve_contract_authority_v2(
                        loaded,
                        contract=contract,
                        phase="EXECUTION_ROW",
                        validator=attempt_authority_validator,
                        device_uuid=str(runtime["device_uuid"]),
                        candidate_envelope_sha256=envelope["candidate_envelope_sha256"],
                        prior_lineage_sha256=lineage,
                    )
                    if binding["attempt_lineage_sha256"] != lineage:
                        raise FullEvolutionWorkhorseProbeError(
                            "row authority refresh changed Attempt lineage"
                        )
                    owner.attempt_authority_binding = binding
                    authority_refreshes.append(binding)
                    return binding

                owner_authority = refresh_execution_authority()
                process_peak = max(
                    process_before,
                    _authority_accounted_device_bytes_v2(owner_authority),
                )
                warmup_sink_acknowledgements = []
                for warmup in range(warmup_dispatches):
                    refresh = refresh_execution_authority()
                    process_peak = max(
                        process_peak, _authority_accounted_device_bytes_v2(refresh)
                    )
                    _advance(loaded, owner, temporal_k, 0)
                    warmup_sink_acknowledgements.append(
                        _acknowledge_events(
                            owner,
                            torch,
                            sink_root=Path(event_sink_root)
                            / loaded.backend.backend_id
                            / f"b{B}-k{temporal_k}-warmup-{warmup}",
                            sink_id="workhorse-tuning-warmup-v2",
                        )
                    )
                samples: list[float] = []
                measurement_samples: list[dict[str, Any]] = []
                for sample_index in range(measured_dispatches):
                    refresh = refresh_execution_authority()
                    process_peak = max(
                        process_peak, _authority_accounted_device_bytes_v2(refresh)
                    )
                    # State readback/hash and live authority checks occur before
                    # the CUDA start event and are excluded from throughput.
                    _prepare_dispatch_v2(loaded, owner, temporal_k, 0)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _advance(loaded, owner, temporal_k, 0)
                    end.record()
                    end.synchronize()
                    milliseconds = float(start.elapsed_time(end))
                    if milliseconds <= 0:
                        raise FullEvolutionWorkhorseProbeError(
                            "CUDA event returned non-positive duration"
                        )
                    rate = B * temporal_k * 1000.0 / milliseconds
                    acknowledgement = _acknowledge_events(
                        owner,
                        torch,
                        sink_root=Path(event_sink_root)
                        / loaded.backend.backend_id
                        / f"b{B}-k{temporal_k}-sample-{sample_index}",
                        sink_id="workhorse-tuning-measurement-v2",
                    )
                    sample_body = {
                        "sample_index": sample_index,
                        "timing_source": "cuda_event",
                        "cuda_event_elapsed_ms": milliseconds,
                        "event_canonicalization_timing_source": (
                            "host_monotonic_synchronized_post_transition_tail"
                        ),
                        "event_canonicalization_tail_ms": (
                            owner.last_event_canonicalization_tail_ms
                        ),
                        "committed_world_ticks": B * temporal_k,
                        "world_ticks_per_second": rate,
                        "event_sink_ack": acknowledgement,
                    }
                    samples.append(rate)
                    measurement_samples.append(
                        {
                            **sample_body,
                            "measurement_sha256": canonical_sha256(sample_body),
                        }
                    )
                peak = int(torch.cuda.max_memory_allocated())
                release_row_owner()
                closeout_authority = _resolve_contract_authority_v2(
                    loaded,
                    contract=contract,
                    phase="CLOSEOUT",
                    validator=attempt_authority_validator,
                    device_uuid=str(runtime["device_uuid"]),
                    candidate_envelope_sha256=envelope["candidate_envelope_sha256"],
                    prior_lineage_sha256=lineage,
                )
                if closeout_authority["attempt_lineage_sha256"] != lineage:
                    raise FullEvolutionWorkhorseProbeError(
                        "row closeout changed Attempt lineage"
                    )
                process_peak = max(
                    process_peak,
                    _authority_accounted_device_bytes_v2(closeout_authority),
                )
                free_after = closeout_authority["response"]["device_free_bytes"]
                if closeout_authority["response"]["device_total_bytes"] != total_vram:
                    raise FullEvolutionWorkhorseProbeError(
                        "row closeout device total differs from sealed envelope"
                    ) from WorkhorseAttemptAuthorityError(
                        "row closeout device identity changed"
                    )
                measured_p10 = _p10(samples)
                efficiency = measured_p10 / ideal_linear_wtps
                observed_free_delta = max(
                    0, int(attempt_admission.free_bytes - free_after)
                )
                resource_ledger_conforms = (
                    process_peak <= ledger.projected_process_peak_bytes
                    and observed_free_delta <= ledger.incremental_device_bytes
                )
                row.update(
                    {
                        "outcome": (
                            "PRELIMINARY_EXPLICIT_SUBSET"
                            if explicit_partial
                            else (
                                "RESOURCE_LEDGER_REJECTED"
                                if not resource_ledger_conforms
                                else (
                                    "MODE_MEETS_25_PERCENT"
                                    if efficiency >= 0.25
                                    else "PERFORMANCE_RECOVERY"
                                )
                            )
                        ),
                        "world_ticks_per_second_samples": samples,
                        "warmup_sink_acknowledgements": (warmup_sink_acknowledgements),
                        "measurement_samples": measurement_samples,
                        "p10_world_ticks_per_second": measured_p10,
                        "p10_site_ticks_per_second": measured_p10 * loaded.shape.sites,
                        "p10_agent_capacity_ticks_per_second": (
                            measured_p10 * loaded.shape.capacity
                        ),
                        "ideal_linear_world_ticks_per_second": ideal_linear_wtps,
                        "normalized_site_throughput_efficiency": efficiency,
                        "allocator_baseline_bytes": allocator_before,
                        "allocator_peak_bytes": peak,
                        "allocator_peak_delta_bytes": max(0, peak - allocator_before),
                        "free_vram_after_measurement_bytes": int(free_after),
                        "observed_free_vram_delta_bytes": observed_free_delta,
                        "driver_process_baseline_bytes": process_before,
                        "driver_process_peak_bytes": process_peak,
                        "driver_process_peak_delta_bytes": max(
                            0, process_peak - process_before
                        ),
                        "resource_ledger_conforms": resource_ledger_conforms,
                        "graph_inspection": graph_inspection,
                        "execution_authority_refreshes": authority_refreshes,
                        "closeout_attempt_authority": closeout_authority,
                    }
                )
            except Exception as exc:
                authority_failure = isinstance(
                    exc.__cause__, WorkhorseAttemptAuthorityError
                )
                if execution_authority is None:
                    raise
                try:
                    release_row_owner()
                except Exception as cleanup_exc:
                    if not authority_failure:
                        exc = cleanup_exc
                if execution_authority is not None and closeout_authority is None:
                    try:
                        closeout_authority = _resolve_contract_authority_v2(
                            loaded,
                            contract=contract,
                            phase="CLOSEOUT",
                            validator=attempt_authority_validator,
                            device_uuid=str(runtime["device_uuid"]),
                            candidate_envelope_sha256=envelope[
                                "candidate_envelope_sha256"
                            ],
                            prior_lineage_sha256=lineage,
                        )
                        row["closeout_attempt_authority"] = closeout_authority
                    except FullEvolutionWorkhorseProbeError as closeout_exc:
                        raise FullEvolutionWorkhorseProbeError(
                            "runtime-rejected row could not reopen exact Attempt closeout"
                        ) from closeout_exc
                if (
                    closeout_authority is None
                    or closeout_authority["attempt_lineage_sha256"] != lineage
                ):
                    raise FullEvolutionWorkhorseProbeError(
                        "runtime-rejected row closeout changed Attempt lineage"
                    )
                if authority_failure:
                    raise
                row.update(
                    {
                        "outcome": "RUNTIME_REJECTED",
                        "error": str(exc),
                        "execution_authority_refreshes": authority_refreshes,
                        "closeout_attempt_authority": closeout_authority,
                    }
                )
            finally:
                if owner is not None:
                    release_row_owner()
            rows.append(row)

    measured_rows = [
        row
        for row in rows
        if "p10_world_ticks_per_second" in row
        and row.get("resource_ledger_conforms") is True
    ]
    expected_complete_rows = sum(len(row["B_candidates"]) for row in envelope["per_k"])
    if not explicit_partial and len(rows) != expected_complete_rows:
        raise FullEvolutionWorkhorseProbeError(
            "canonical live-derived candidate envelope was not completely retained"
        )
    winner = max(
        measured_rows,
        key=lambda row: row["p10_world_ticks_per_second"],
        default=None,
    )
    # The micro projection is writable under its owning V2 writer contract.
    # Reopen it after the physical sweep so a mid-run replacement cannot be
    # smuggled into the sealed workhorse decision.  Reopen semantic evidence
    # at the same boundary even though it is already immutable.
    final_micro = validate_authoritative_micro_denominator_v2(
        authoritative_micro_receipt_path,
        expected_device_identity=device_identity,
    )
    final_semantic_gate = validate_workhorse_semantic_gate_receipt_v2(
        semantic_gate_receipt_path,
        expected_shape_id=loaded.shape.shape_id,
        expected_candidate_artifact_sha256=file_sha256(loaded.artifact),
        expected_candidate_backend_id=loaded.backend.backend_id,
    )
    if final_micro != micro or final_semantic_gate != semantic_gate:
        raise FullEvolutionWorkhorseProbeError(
            "micro or semantic authority changed during the tuning attempt"
        )
    final_artifact, _, final_manifest = validate_workhorse_prebuilt_v2(loaded.artifact)
    if (
        file_sha256(final_artifact) != file_sha256(loaded.artifact)
        or final_manifest != loaded.manifest
    ):
        raise FullEvolutionWorkhorseProbeError(
            "sealed workhorse artifact changed during the tuning attempt"
        )
    body = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_scaling_receipt_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": str(loaded.artifact),
            "sha256": file_sha256(loaded.artifact),
            "manifest_sha256": loaded.manifest["manifest_sha256"],
            "identity": loaded.manifest["identity"],
            "build_identity": loaded.manifest["build_identity"],
            "sealed_artifact_identity": loaded.manifest["sealed_artifact_identity"],
        },
        "shape": loaded.shape.identity_fields(),
        "layout": loaded.layout.to_dict(),
        "backend_topology": loaded.backend.topology_payload(loaded.shape),
        "runtime": runtime,
        "compiled_kernel_resources": runtime["compiled_kernel_resources"],
        "device_identity": device_identity,
        "authoritative_micro_denominator": micro,
        "semantic_gate": semantic_gate,
        "governing_contracts": workhorse_governing_contract_bindings_v2(),
        "surface_scale": surface_scale,
        "ideal_linear_world_ticks_per_second": ideal_linear_wtps,
        "performance_viability_threshold": 0.25,
        "warmup_dispatches": warmup_dispatches,
        "measured_dispatches": measured_dispatches,
        "timing_source": "cuda_event_exact_dispatch_v2",
        "sink_boundary": "external_exact_content_ack_outside_timed_interval",
        "candidate_envelope": {
            **envelope,
            "plan_mode": plan_mode,
            "measured_or_rejected_rows": len(rows),
            "complete_cartesian_rows": (
                expected_complete_rows if not explicit_partial else 0
            ),
        },
        "owner_component_measurements": owner_component_measurements,
        "topology_resource_bindings_by_k": {
            str(key): value for key, value in topology_resource_bindings_by_k.items()
        },
        "candidate_envelope_attempt_authority": envelope_authority,
        "rows": rows,
        "winner": winner,
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json_v2(receipt, receipt_output_path)
    if not explicit_partial:
        return validate_workhorse_scaling_receipt_v2(
            output, expected_backend_id=loaded.backend.backend_id
        )
    return json.loads(output.read_text(encoding="utf-8"))


def validate_workhorse_shape_scaling_receipt_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen both independently tuned modes before accepting shape viability."""
    supplied = Path(receipt_path)
    if supplied.is_symlink() or not supplied.is_file():
        raise FullEvolutionWorkhorseProbeError(
            "shape scaling receipt must be a regular non-symlink"
        )
    path = supplied.resolve()
    if path.stat().st_mode & 0o222:
        raise FullEvolutionWorkhorseProbeError(
            "shape scaling receipt must be read-only"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullEvolutionWorkhorseProbeError(
            f"cannot reopen shape scaling receipt: {exc}"
        ) from exc
    if type(value) is not dict:
        raise FullEvolutionWorkhorseProbeError(
            "shape scaling receipt must be an object"
        )
    body = {key: item for key, item in value.items() if key != "receipt_content_sha256"}
    expected_modes = {
        BACKEND_PHASE_GRAPH_V2,
        BACKEND_FUSED_CTA_TEMPORAL_V2,
    }
    mode_files = value.get("mode_receipt_files")
    embedded_modes = value.get("mode_receipts")
    if (
        value.get("schema_version") != "eiel_workhorse_shape_mode_scaling_receipt_v2"
        or value.get("receipt_content_sha256") != canonical_sha256(body)
        or value.get("governing_contracts")
        != workhorse_governing_contract_bindings_v2()
        or type(mode_files) is not dict
        or set(mode_files) != expected_modes
        or type(embedded_modes) is not dict
        or set(embedded_modes) != expected_modes
    ):
        raise FullEvolutionWorkhorseProbeError(
            "shape scaling schema/source/mode seal is invalid"
        )
    mode_receipts = {}
    for backend_id in (
        BACKEND_PHASE_GRAPH_V2,
        BACKEND_FUSED_CTA_TEMPORAL_V2,
    ):
        file_row = mode_files[backend_id]
        if type(file_row) is not dict or set(file_row) != {
            "path",
            "sha256",
            "receipt_content_sha256",
        }:
            raise FullEvolutionWorkhorseProbeError(
                "shape mode receipt file identity is invalid"
            )
        mode_receipt = validate_workhorse_scaling_receipt_v2(
            file_row["path"], expected_backend_id=backend_id
        )
        if (
            file_row["sha256"] != file_sha256(file_row["path"])
            or file_row["receipt_content_sha256"]
            != mode_receipt["receipt_content_sha256"]
            or embedded_modes[backend_id] != mode_receipt
        ):
            raise FullEvolutionWorkhorseProbeError(
                "shape mode receipt does not reopen exactly"
            )
        mode_receipts[backend_id] = mode_receipt
    shape_ids = {receipt["shape"]["shape_id"] for receipt in mode_receipts.values()}
    devices = {
        canonical_sha256(receipt["device_identity"])
        for receipt in mode_receipts.values()
    }
    if (
        len(shape_ids) != 1
        or len(devices) != 1
        or value.get("shape_id") != next(iter(shape_ids))
    ):
        raise FullEvolutionWorkhorseProbeError(
            "shape modes are not the exact same shape/device"
        )
    all_rows = [
        {"backend_id": backend_id, **row}
        for backend_id, receipt in mode_receipts.items()
        for row in receipt["rows"]
    ]
    measured_rows = [
        row
        for row in all_rows
        if "p10_world_ticks_per_second" in row
        and row.get("resource_ledger_conforms") is True
    ]
    winner = max(
        measured_rows,
        key=lambda row: row["p10_world_ticks_per_second"],
        default=None,
    )
    complete = all(
        receipt["candidate_envelope"]["plan_mode"] == "CANONICAL_LIVE_DERIVED_COMPLETE"
        and receipt["candidate_envelope"]["complete"] is True
        for receipt in mode_receipts.values()
    )
    decision = (
        "VIABLE"
        if complete
        and winner is not None
        and winner["normalized_site_throughput_efficiency"] >= 0.25
        else "PERFORMANCE_RECOVERY"
    )
    if (
        value.get("performance_viability_threshold") != 0.25
        or value.get("mode_envelope_complete") is not complete
        or value.get("rows") != all_rows
        or value.get("winner") != winner
        or value.get("decision") != decision
    ):
        raise FullEvolutionWorkhorseProbeError(
            "shape scaling viability decision does not recompute"
        )
    return value


def run_workhorse_shape_tuning_v2(
    phase_graph_artifact: str | Path,
    fused_temporal_artifact: str | Path,
    *,
    authoritative_micro_receipt_path: str | Path,
    semantic_gate_receipts: Mapping[str, str | Path],
    event_sink_root: str | Path,
    receipt_root: str | Path,
    output_path: str | Path,
    attempt_authority_validator: AttemptAuthorityValidatorV2,
    warmup_dispatches: int = 3,
    measured_dispatches: int = 10,
) -> dict[str, Any]:
    """Retired direct entry; aggregate canonical row receipts offline."""

    raise FullEvolutionWorkhorseProbeError(
        "direct multi-row Workhorse tuning is retired; aggregate independently "
        "terminalized canonical row receipts offline"
    )

    artifacts = {
        BACKEND_PHASE_GRAPH_V2: phase_graph_artifact,
        BACKEND_FUSED_CTA_TEMPORAL_V2: fused_temporal_artifact,
    }
    if set(semantic_gate_receipts) != set(artifacts):
        raise FullEvolutionWorkhorseProbeError(
            "both performance modes require their own passed semantic receipt"
        )
    mode_receipts = {}
    mode_receipt_files = {}
    for backend_id, artifact in artifacts.items():
        mode_path = Path(receipt_root) / f"{backend_id}.json"
        run_workhorse_tuning_v2(
            artifact,
            authoritative_micro_receipt_path=authoritative_micro_receipt_path,
            semantic_gate_receipt_path=semantic_gate_receipts[backend_id],
            event_sink_root=Path(event_sink_root) / backend_id,
            receipt_output_path=mode_path,
            attempt_authority_validator=attempt_authority_validator,
            warmup_dispatches=warmup_dispatches,
            measured_dispatches=measured_dispatches,
        )
        mode_receipts[backend_id] = validate_workhorse_scaling_receipt_v2(
            mode_path, expected_backend_id=backend_id
        )
        mode_receipt_files[backend_id] = {
            "path": str(mode_path.resolve()),
            "sha256": file_sha256(mode_path),
            "receipt_content_sha256": mode_receipts[backend_id][
                "receipt_content_sha256"
            ],
        }
    shape_ids = {receipt["shape"]["shape_id"] for receipt in mode_receipts.values()}
    devices = {
        canonical_sha256(receipt["device_identity"])
        for receipt in mode_receipts.values()
    }
    if len(shape_ids) != 1 or len(devices) != 1:
        raise FullEvolutionWorkhorseProbeError(
            "performance modes are not the same shape/device"
        )
    all_rows = [
        {"backend_id": backend_id, **row}
        for backend_id, receipt in mode_receipts.items()
        for row in receipt["rows"]
    ]
    measured_rows = [
        row
        for row in all_rows
        if "p10_world_ticks_per_second" in row
        and row.get("resource_ledger_conforms") is True
    ]
    winner = max(
        measured_rows,
        key=lambda row: row["p10_world_ticks_per_second"],
        default=None,
    )
    complete = all(
        receipt["candidate_envelope"]["plan_mode"] == "CANONICAL_LIVE_DERIVED_COMPLETE"
        and receipt["candidate_envelope"]["complete"] is True
        for receipt in mode_receipts.values()
    )
    decision = (
        "VIABLE"
        if complete
        and winner is not None
        and winner["normalized_site_throughput_efficiency"] >= 0.25
        else "PERFORMANCE_RECOVERY"
    )
    body = {
        "schema_version": "eiel_workhorse_shape_mode_scaling_receipt_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shape_id": next(iter(shape_ids)),
        "decision": decision,
        "performance_viability_threshold": 0.25,
        "mode_envelope_complete": complete,
        "mode_receipt_files": mode_receipt_files,
        "mode_receipts": mode_receipts,
        "rows": all_rows,
        "winner": winner,
        "governing_contracts": workhorse_governing_contract_bindings_v2(),
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = _write_sealed_json_v2(receipt, output_path)
    return validate_workhorse_shape_scaling_receipt_v2(output)


def _write_worker_raw_continuation_v2(
    snapshot: Mapping[str, Any], *, output_path: Path, start_next_absolute_tick: int
) -> dict[str, Any]:
    dataset = _raw_dataset_v2(
        snapshot, start_next_absolute_tick=start_next_absolute_tick
    )
    components: dict[str, Any] = {}
    for name in (
        "input_state",
        "input_scratch",
        "input_events",
        "input_counts",
        "state",
        "scratch",
        "events",
        "counts",
    ):
        path = output_path.with_name(f"{output_path.stem}.{name}.bin")
        payload = dataset[name]
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
        components[name] = {
            "path": str(path.resolve()),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "B": dataset["B"],
        "temporal_k": dataset["temporal_k"],
        "start_next_absolute_tick": dataset["start_next_absolute_tick"],
        "components": components,
    }


def _require_worker_raw_continuation_v2(
    worker_receipt: Mapping[str, Any], receipt_parent: Path
) -> dict[str, Any]:
    raw = worker_receipt.get("raw_continuation")
    if type(raw) is not dict or set(raw) != {
        "B",
        "temporal_k",
        "start_next_absolute_tick",
        "components",
    }:
        raise FullEvolutionWorkhorseProbeError(
            "fresh worker raw continuation projection is missing"
        )
    components = raw["components"]
    if type(components) is not dict or set(components) != {
        "input_state",
        "input_scratch",
        "input_events",
        "input_counts",
        "state",
        "scratch",
        "events",
        "counts",
    }:
        raise FullEvolutionWorkhorseProbeError(
            "fresh worker raw component inventory is invalid"
        )
    result = {
        "B": raw["B"],
        "temporal_k": raw["temporal_k"],
        "start_next_absolute_tick": raw["start_next_absolute_tick"],
    }
    for name, row in components.items():
        if type(row) is not dict or set(row) != {"path", "bytes", "sha256"}:
            raise FullEvolutionWorkhorseProbeError(
                "fresh worker raw component identity is invalid"
            )
        path = Path(row["path"])
        if (
            path.parent.resolve() != receipt_parent.resolve()
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_mode & 0o222
        ):
            raise FullEvolutionWorkhorseProbeError(
                "fresh worker raw component locator is unsafe"
            )
        payload = path.read_bytes()
        if (
            row["bytes"] != len(payload)
            or row["sha256"] != hashlib.sha256(payload).hexdigest()
        ):
            raise FullEvolutionWorkhorseProbeError(
                "fresh worker raw component exact-content seal failed"
            )
        result[name] = payload
    return result


def _fresh_restore_worker_v2(
    checkpoint_path: Path,
    output_path: Path,
    sink_root: Path,
    authority_request_path: Path,
    authority_binding_path: Path,
    closeout_request_path: Path,
    closeout_binding_path: Path,
) -> None:
    """Fresh interpreter/process continuation used by the semantic gate."""
    checkpoint, state_bytes = reopen_workhorse_checkpoint_bundle_v2(checkpoint_path)
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(checkpoint["contract"])
    worker_contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=contract.shape_id,
        backend_id=contract.backend_id,
        B=contract.B,
        temporal_k=1,
        total_ticks=1,
        start_next_tick=checkpoint["checkpoint_at_tick"],
    )
    authority_request = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=worker_contract,
        artifact_sha256=checkpoint["artifact"]["sha256"],
        sealed_artifact_identity_sha256=checkpoint["artifact"][
            "sealed_artifact_identity_sha256"
        ],
        device_uuid=checkpoint["event_sink_ack"]["device_uuid"],
        candidate_envelope_sha256=None,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
        source_checkpoint_identity_sha256=checkpoint["checkpoint_identity"][
            "checkpoint_identity_sha256"
        ],
    )
    _write_sealed_json_v2(authority_request, authority_request_path)
    deadline = time.monotonic() + 120.0
    while not authority_binding_path.is_file():
        if time.monotonic() >= deadline:
            raise FullEvolutionWorkhorseProbeError(
                "fresh worker did not receive Attempt authority"
            )
        time.sleep(0.05)
    authority_binding = json.loads(authority_binding_path.read_text(encoding="utf-8"))
    try:
        authority_binding = validate_workhorse_attempt_authority_binding_v2(
            authority_binding
        )
    except WorkhorseAttemptAuthorityError as exc:
        raise FullEvolutionWorkhorseProbeError(str(exc)) from exc
    torch = None
    loaded = None
    owner = None
    execution_error: Exception | None = None
    try:
        torch = _torch()
        loaded = _load(checkpoint["artifact"]["path"], torch)
        owner = _allocate_owner(torch, loaded, contract.B, 1)
        owner.state.copy_(
            torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8).reshape(
                contract.B, loaded.layout.state_bytes_per_world
            )
        )
        owner.scratch.zero_()
        owner.world_uids = tuple(checkpoint["world_uids"])
        owner.rng_keys = tuple(checkpoint["rng_keys"])
        owner.committed_next_tick = checkpoint["checkpoint_at_tick"]
        owner.attempt_authority_binding = authority_binding
        loaded.module.restore_derived(owner.state, owner.scratch)
        continuation_input = _owner_cpu_snapshot_v2(owner)
        _prepare_graph_if_needed(loaded, owner, 1, 0)
        _advance(loaded, owner, 1, 0)
        sink_ack = _acknowledge_events(
            owner,
            torch,
            sink_root=sink_root,
            sink_id="workhorse-fresh-process-continuation-v2",
        )
        continuation_snapshot = _owner_cpu_snapshot_v2(owner)
        continuation_snapshot["input"] = continuation_input
        raw_continuation = _write_worker_raw_continuation_v2(
            continuation_snapshot,
            output_path=output_path,
            start_next_absolute_tick=checkpoint["checkpoint_at_tick"],
        )
        continuation_identity = {
            "state_sha256": _sha_tensor(owner.state),
            "scratch_sha256": _sha_tensor(owner.scratch),
            "events_sha256": _sha_tensor(owner.events),
            "counts_sha256": _sha_tensor(owner.counts),
            "next_absolute_tick": owner.committed_next_tick,
        }
    except Exception as exc:
        execution_error = exc
    finally:
        try:
            if loaded is not None:
                _release_graph(loaded)
        except Exception as exc:
            if execution_error is None:
                execution_error = exc
        finally:
            owner = None
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as exc:
                    if execution_error is None:
                        execution_error = exc
    closeout_request = workhorse_attempt_authority_request_v2(
        phase="CLOSEOUT",
        contract=worker_contract,
        artifact_sha256=checkpoint["artifact"]["sha256"],
        sealed_artifact_identity_sha256=checkpoint["artifact"][
            "sealed_artifact_identity_sha256"
        ],
        device_uuid=checkpoint["event_sink_ack"]["device_uuid"],
        candidate_envelope_sha256=None,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
        prior_lineage_sha256=authority_binding["attempt_lineage_sha256"],
    )
    _write_sealed_json_v2(closeout_request, closeout_request_path)
    while not closeout_binding_path.is_file():
        if time.monotonic() >= deadline:
            raise FullEvolutionWorkhorseProbeError(
                "fresh worker did not receive Attempt closeout authority"
            )
        time.sleep(0.05)
    closeout_authority = json.loads(closeout_binding_path.read_text(encoding="utf-8"))
    try:
        closeout_authority = validate_workhorse_attempt_authority_binding_v2(
            closeout_authority
        )
    except WorkhorseAttemptAuthorityError as exc:
        raise FullEvolutionWorkhorseProbeError(str(exc)) from exc
    if (
        closeout_authority["request"] != closeout_request
        or closeout_authority["attempt_lineage_sha256"]
        != authority_binding["attempt_lineage_sha256"]
    ):
        raise FullEvolutionWorkhorseProbeError(
            "fresh worker Attempt closeout request or lineage was substituted"
        )
    if execution_error is not None:
        raise FullEvolutionWorkhorseProbeError(
            f"fresh worker execution failed after exact Attempt closeout: {execution_error}"
        ) from execution_error
    body = {
        "schema_version": "eiel_workhorse_fresh_process_restore_receipt_v2",
        "source_pid": os.getpid(),
        "checkpoint_bundle_sha256": checkpoint["bundle_body_sha256"],
        "artifact_sha256": file_sha256(loaded.artifact),
        "state_sha256": continuation_identity["state_sha256"],
        "scratch_sha256": continuation_identity["scratch_sha256"],
        "events_sha256": continuation_identity["events_sha256"],
        "counts_sha256": continuation_identity["counts_sha256"],
        "event_sink_ack": sink_ack,
        "attempt_authority_binding": authority_binding,
        "closeout_attempt_authority": closeout_authority,
        "raw_continuation": raw_continuation,
        "next_absolute_tick": continuation_identity["next_absolute_tick"],
    }
    _write_sealed_json_v2(
        {**body, "worker_receipt_sha256": canonical_sha256(body)}, output_path
    )


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--restore-worker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-sink-root", type=Path)
    parser.add_argument("--worker-authority-request", type=Path)
    parser.add_argument("--worker-authority-binding", type=Path)
    parser.add_argument("--worker-closeout-request", type=Path)
    parser.add_argument("--worker-closeout-binding", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.restore_worker is None
        or arguments.worker_output is None
        or arguments.worker_sink_root is None
        or arguments.worker_authority_request is None
        or arguments.worker_authority_binding is None
        or arguments.worker_closeout_request is None
        or arguments.worker_closeout_binding is None
    ):
        parser.error(
            "the sealed restore worker requires all authority and output paths"
        )
    _fresh_restore_worker_v2(
        arguments.restore_worker,
        arguments.worker_output,
        arguments.worker_sink_root,
        arguments.worker_authority_request,
        arguments.worker_authority_binding,
        arguments.worker_closeout_request,
        arguments.worker_closeout_binding,
    )
    return 0


__all__ = [
    "FullEvolutionWorkhorseProbeError",
    "run_workhorse_semantic_gate_v2",
    "run_workhorse_shape_tuning_v2",
    "run_workhorse_tuning_v2",
]


if __name__ == "__main__":  # pragma: no cover - physical fresh-process path
    raise SystemExit(_main())
