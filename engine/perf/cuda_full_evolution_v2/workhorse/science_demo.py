"""Bounded technical science-compatibility demo for Workhorse V2.

The runtime remains observation ``OFF``.  Six compact host-boundary records are
projected only after an exact event acknowledgement (or from the initialized
tick-zero state), and therefore cannot perturb the CUDA transition.  The
records intentionally use ``next_absolute_tick`` boundary semantics rather
than pretending to be native Science Instrument event records.

This module grants neither science nor qualification authority.  It proves a
small, restartable data-extraction shape and carries the identities needed by a
future qualified science adapter without claiming that adapter is integrated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from ..science_extension import (
    seal_science_route_binding_adapter_v1,
    seal_science_world_mapping_v1,
    seal_static_science_control_table_v1,
)
from .authority import (
    reopen_workhorse_checkpoint_bundle_v2,
    validate_workhorse_event_sink_receipt_v2,
    write_workhorse_checkpoint_bundle_v2,
)
from .canonical_rows import (
    validate_workhorse_canonical_parent_authority_binding_v2,
    validate_workhorse_canonical_terminal_binding_v2,
    validate_workhorse_device_envelope_calibration_receipt_v2,
    workhorse_artifact_binding_v2,
)
from .contract import (
    BACKEND_REFERENCE_PHASED_V2,
    FULL_EVOLUTION_CAPABILITIES_V2,
    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    FullEvolutionCudaWorkhorseContractV2,
    canonical_sha256,
)
from .native_loader import file_sha256
from .semantic_cases import (
    validate_workhorse_science_checkpoint_compatibility_binding_v2,
)


WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1: Final[str] = (
    "SCIENCE_COMPATIBILITY_DEMO"
)
WORKHORSE_SCIENCE_DEMO_CONTRACT_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_science_compatibility_demo_contract_v1"
)
WORKHORSE_SCIENCE_BOUNDARY_RECORD_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_science_host_boundary_record_v1"
)
WORKHORSE_SCIENCE_CHECKPOINT_SIDECAR_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_science_checkpoint_compatibility_sidecar_v1"
)
WORKHORSE_SCIENCE_RESTORE_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_science_fresh_owner_restore_receipt_v1"
)
WORKHORSE_SCIENCE_DEMO_RESULT_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_science_compatibility_demo_result_v1"
)
WORKHORSE_SCIENCE_DEMO_TERMINAL_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_terminalized_science_compatibility_demo_receipt_v1"
)
WORKHORSE_SCIENCE_SEMANTIC_INHERITANCE_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_science_demo_semantic_inheritance_v1"
)

SCIENCE_DEMO_SHAPE_ID_V1: Final[str] = "w128h80p2560c5120"
SCIENCE_DEMO_BACKEND_ID_V1: Final[str] = BACKEND_REFERENCE_PHASED_V2
SCIENCE_DEMO_B_V1: Final[int] = 10
SCIENCE_DEMO_TEMPORAL_K_V1: Final[int] = 1
SCIENCE_DEMO_START_NEXT_TICK_V1: Final[int] = 0
SCIENCE_DEMO_TOTAL_TICKS_V1: Final[int] = 40
SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1: Final[tuple[int, ...]] = (
    0,
    8,
    16,
    24,
    32,
    40,
)
SCIENCE_DEMO_CHECKPOINT_TICKS_V1: Final[tuple[int, ...]] = (16, 32, 40)
SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1: Final[tuple[int, ...]] = (16, 32)
SCIENCE_DEMO_WORLD_UID_BASE_V1: Final[int] = 2_026_082_700_000
SCIENCE_DEMO_RNG_KEY_V1: Final[int] = 0xE1E1_2026_0827

_CLAIM_CLASS: Final[str] = "TECHNICAL_ONLY"
_NO_AUTHORITY: Final[dict[str, bool]] = {
    "science_authority_granted": False,
    "qualification_authority_granted": False,
}


class WorkhorseScienceDemoError(RuntimeError):
    """Raised when the bounded science-compatibility proof is not exact."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": canonical_sha256(value)}


def _is_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise WorkhorseScienceDemoError(f"{label} fields are inexact")
    return dict(value)


def _write_sealed_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise WorkhorseScienceDemoError(f"fresh output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("sealed JSON write made no progress")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return output


def _read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise WorkhorseScienceDemoError(f"{label} must be a regular file")
    resolved = supplied.resolve()
    try:
        value = json.loads(resolved.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseScienceDemoError(f"cannot reopen {label}: {exc}") from exc
    if type(value) is not dict:
        raise WorkhorseScienceDemoError(f"{label} must be an object")
    return resolved, value


def _projected_file_sha256(path: str | Path) -> str:
    """Hash one JSON-projected path through the Path-only native helper."""

    return file_sha256(Path(path).expanduser().resolve())


def workhorse_science_semantic_inheritance_binding_v1(
    semantic_gate_receipt_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    """Reopen the prior all-component gate by its own exact candidate identity."""

    path, raw = _read_json(semantic_gate_receipt_path, "semantic gate")
    candidate = raw.get("candidate")
    if not isinstance(candidate, Mapping):
        raise WorkhorseScienceDemoError("semantic gate candidate is absent")
    candidate_artifact = candidate.get("artifact_sha256")
    identity = candidate.get("identity")
    backend_id = identity.get("backend_id") if isinstance(identity, Mapping) else None
    shape_id = raw.get("shape_id")
    if not _is_sha(candidate_artifact) or type(backend_id) is not str:
        raise WorkhorseScienceDemoError("semantic gate candidate identity is invalid")
    try:
        from .authority import validate_workhorse_semantic_gate_receipt_v2

        gate = validate_workhorse_semantic_gate_receipt_v2(
            path,
            expected_shape_id=str(shape_id),
            expected_candidate_artifact_sha256=str(candidate_artifact),
            expected_candidate_backend_id=backend_id,
            require_current_source_match=require_current_source_match,
        )
    except Exception as exc:
        raise WorkhorseScienceDemoError(
            f"semantic inheritance gate cannot reopen: {exc}"
        ) from exc
    if shape_id != SCIENCE_DEMO_SHAPE_ID_V1 or gate.get("shape_id") != (
        SCIENCE_DEMO_SHAPE_ID_V1
    ):
        raise WorkhorseScienceDemoError(
            "semantic inheritance gate does not cover the exact 10x shape"
        )
    body = {
        "schema_version": WORKHORSE_SCIENCE_SEMANTIC_INHERITANCE_SCHEMA_V1,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "inheritance_scope": (
            "UNCHANGED_WORKHORSE_TRANSITION_AND_NATIVE_SEMANTIC_CONTRACT"
        ),
        "shape_id": SCIENCE_DEMO_SHAPE_ID_V1,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "semantic_gate_receipt_path": str(path),
        "semantic_gate_receipt_file_sha256": file_sha256(path),
        "semantic_gate_receipt_content_sha256": gate["semantic_gate_receipt_sha256"],
        "source_candidate_artifact_sha256": gate["candidate_artifact_sha256"],
        "source_candidate_backend_id": gate["candidate_backend_id"],
        "all_required_semantic_checks_passed": True,
        "new_artifact_byte_identity_inherited": False,
        "native_semantics_inherited": True,
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def validate_workhorse_science_semantic_inheritance_binding_v1(
    value: Mapping[str, Any],
    *,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "claim_class",
        "science_authority_granted",
        "qualification_authority_granted",
        "inheritance_scope",
        "shape_id",
        "semantic_transition_sha256",
        "semantic_gate_receipt_path",
        "semantic_gate_receipt_file_sha256",
        "semantic_gate_receipt_content_sha256",
        "source_candidate_artifact_sha256",
        "source_candidate_backend_id",
        "all_required_semantic_checks_passed",
        "new_artifact_byte_identity_inherited",
        "native_semantics_inherited",
        "binding_sha256",
    }
    supplied = _exact(value, fields, "semantic inheritance binding")
    rebuilt = workhorse_science_semantic_inheritance_binding_v1(
        supplied["semantic_gate_receipt_path"],
        require_current_source_match=require_current_source_match,
    )
    if supplied != rebuilt:
        raise WorkhorseScienceDemoError("semantic inheritance binding changed")
    return rebuilt


def workhorse_science_demo_contract_v1(
    *,
    artifact_binding: Mapping[str, Any],
    envelope_binding: Mapping[str, Any],
    semantic_inheritance_binding: Mapping[str, Any],
    science_checkpoint_compatibility_binding: Mapping[str, Any],
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    artifact = workhorse_artifact_binding_v2(
        str(artifact_binding.get("path")),
        require_current_source_match=require_current_source_match,
    )
    if artifact != dict(artifact_binding):
        raise WorkhorseScienceDemoError("science demo artifact binding changed")
    if (
        artifact["identity"]["shape_id"] != SCIENCE_DEMO_SHAPE_ID_V1
        or artifact["identity"]["backend_id"] != SCIENCE_DEMO_BACKEND_ID_V1
    ):
        raise WorkhorseScienceDemoError("science demo requires the exact 10x reference")
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        str(envelope_binding.get("path"))
    )
    expected_envelope = {
        "path": envelope["receipt_path"],
        "file_sha256": envelope["receipt_file_sha256"],
        "content_sha256": envelope["receipt_content_sha256"],
    }
    if expected_envelope != dict(envelope_binding) or envelope["artifact"] != artifact:
        raise WorkhorseScienceDemoError("science demo envelope changed artifact")
    semantic = validate_workhorse_science_semantic_inheritance_binding_v1(
        semantic_inheritance_binding,
        require_current_source_match=require_current_source_match,
    )
    science = validate_workhorse_science_checkpoint_compatibility_binding_v2(
        science_checkpoint_compatibility_binding
    )
    world_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for logical_world_id in range(SCIENCE_DEMO_B_V1):
        cuda_uid = SCIENCE_DEMO_WORLD_UID_BASE_V1 + logical_world_id
        semantic_world = {
            "study_id": "tw-workhorse-science-compatibility-v1",
            "test_id": "ten-world-boundary-observation",
            "version_id": "v1",
            "arm_id": "STATIC-CONTROL",
            "replicate_id": f"r{logical_world_id + 1:02d}",
            "reducer_id": "population-change-detector-v1",
            "logical_world_id": logical_world_id,
        }
        science_uid = _digest(semantic_world)
        world_rows.append(
            {
                "cuda_world_uid": cuda_uid,
                "science_world_uid": science_uid,
                "semantic_world_id": (
                    "tw-workhorse-science-compatibility-v1:"
                    f"static-control:r{logical_world_id + 1:02d}:"
                    f"w{logical_world_id:02d}"
                ),
                "cell_id": f"science-demo-world-{logical_world_id:02d}",
                "logical_world_id": logical_world_id,
                **{
                    key: str(value)
                    for key, value in semantic_world.items()
                    if key
                    in {
                        "study_id",
                        "test_id",
                        "version_id",
                        "arm_id",
                        "replicate_id",
                        "reducer_id",
                    }
                },
            }
        )
        control_rows.append(
            {
                "cuda_world_uid": cuda_uid,
                "intervals": [
                    {
                        "start_absolute_tick": 0,
                        "end_absolute_tick": SCIENCE_DEMO_TOTAL_TICKS_V1,
                        "control_payload_sha256": _digest(
                            {
                                "schema_version": (
                                    "eiel_workhorse_static_control_payload_v1"
                                ),
                                "arm_id": "STATIC-CONTROL",
                                "disable_mask": 0,
                                "runtime_control_adapter_integrated": False,
                            }
                        ),
                    }
                ],
            }
        )
    world_mapping = seal_science_world_mapping_v1(world_rows)
    control_table = seal_static_science_control_table_v1(
        control_rows,
        terminal_next_absolute_tick=SCIENCE_DEMO_TOTAL_TICKS_V1,
    )
    route_binding = {
        "schema_version": "eiel_science_route_binding_v1",
        "instrument_manifest_sha256": _digest(science),
        "route_id": (
            "tensor_world_cuda_full_evolution_workhorse_v2__"
            "w128h80p2560c5120__science_compatibility_demo_v1"
        ),
        "machine_profile_id": "EIEL_PRODUCTION_BASELINE_V1",
        "route_profile_sha256": _digest(
            {
                "shape_id": SCIENCE_DEMO_SHAPE_ID_V1,
                "backend_id": SCIENCE_DEMO_BACKEND_ID_V1,
                "B": SCIENCE_DEMO_B_V1,
                "temporal_k": SCIENCE_DEMO_TEMPORAL_K_V1,
            }
        ),
        "capability_projection_sha256": _digest(sorted(FULL_EVOLUTION_CAPABILITIES_V2)),
        "native_build_closure_sha256": _digest(artifact["source_inventory_sha256"]),
        "execution_adapter_sha256": _digest(WORKHORSE_SCIENCE_DEMO_RESULT_SCHEMA_V1),
        "control_adapter_sha256": control_table["control_table_sha256"],
        "observation_adapter_sha256": _digest(
            WORKHORSE_SCIENCE_BOUNDARY_RECORD_SCHEMA_V1
        ),
        "checkpoint_adapter_sha256": _digest(
            WORKHORSE_SCIENCE_CHECKPOINT_SIDECAR_SCHEMA_V1
        ),
        "semantic_conformance_sha256": _digest(semantic),
        "resource_envelope_sha256": _digest(expected_envelope),
        "qualified_envelope_sha256": _digest({"status": "UNQUALIFIED_TECHNICAL_DEMO"}),
        "qualification_authority_sha256": _digest(
            {"qualification_authority_granted": False}
        ),
    }
    route_adapter = seal_science_route_binding_adapter_v1(route_binding)
    body = {
        "schema_version": WORKHORSE_SCIENCE_DEMO_CONTRACT_SCHEMA_V1,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "operation": WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
        "time_authority": "absolute_tick_v1",
        "shape_id": SCIENCE_DEMO_SHAPE_ID_V1,
        "backend_id": SCIENCE_DEMO_BACKEND_ID_V1,
        "B": SCIENCE_DEMO_B_V1,
        "temporal_k": SCIENCE_DEMO_TEMPORAL_K_V1,
        "start_next_tick": SCIENCE_DEMO_START_NEXT_TICK_V1,
        "total_ticks": SCIENCE_DEMO_TOTAL_TICKS_V1,
        "target_next_tick": SCIENCE_DEMO_TOTAL_TICKS_V1,
        "boundary_next_ticks": list(SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1),
        "checkpoint_ticks": list(SCIENCE_DEMO_CHECKPOINT_TICKS_V1),
        "continuation_restore_ticks": list(SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1),
        "terminal_restore_validation_tick": SCIENCE_DEMO_TOTAL_TICKS_V1,
        "initialization": {
            "world_uid_base": SCIENCE_DEMO_WORLD_UID_BASE_V1,
            "rng_key": SCIENCE_DEMO_RNG_KEY_V1,
            "population_fixture": False,
        },
        "full_evolution_components": sorted(FULL_EVOLUTION_CAPABILITIES_V2),
        "disable_mask": 0,
        "runtime_observation_mode": "OFF",
        "runtime_control_adapter_integrated": False,
        "native_capture_runtime_integrated": False,
        "science_route_qualified": False,
        "science_continuable": False,
        "artifact_lifecycle_profile": "SCIENCE_COMPACT",
        "raw_tick_or_event_payloads_retained": False,
        "boundary_record_schema": WORKHORSE_SCIENCE_BOUNDARY_RECORD_SCHEMA_V1,
        "checkpoint_sidecar_schema": WORKHORSE_SCIENCE_CHECKPOINT_SIDECAR_SCHEMA_V1,
        "restore_receipt_schema": WORKHORSE_SCIENCE_RESTORE_RECEIPT_SCHEMA_V1,
        "science_checkpoint_compatibility_binding": science,
        "semantic_inheritance_binding": semantic,
        "science_route_binding_adapter": route_adapter,
        "world_mapping": world_mapping,
        "static_control_table": control_table,
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def validate_workhorse_science_demo_contract_v1(
    value: Mapping[str, Any],
    *,
    artifact_binding: Mapping[str, Any],
    envelope_binding: Mapping[str, Any],
    semantic_inheritance_binding: Mapping[str, Any],
    science_checkpoint_compatibility_binding: Mapping[str, Any],
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    rebuilt = workhorse_science_demo_contract_v1(
        artifact_binding=artifact_binding,
        envelope_binding=envelope_binding,
        semantic_inheritance_binding=semantic_inheritance_binding,
        science_checkpoint_compatibility_binding=(
            science_checkpoint_compatibility_binding
        ),
        require_current_source_match=require_current_source_match,
    )
    if dict(value) != rebuilt:
        raise WorkhorseScienceDemoError("science demo contract is not canonical")
    return rebuilt


def _artifact_projection(path: Path, *, content_sha256: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "content_sha256": content_sha256,
    }


def _state_projection(
    *,
    state_bytes: bytes,
    owner: Any,
    loaded: Any,
    torch: Any,
    world_mapping: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    from .probe import _field

    header = _field(owner.state, loaded.layout, "header", torch).detach().cpu()
    alive = _field(owner.state, loaded.layout, "alive", torch).detach().cpu()
    occupancy = _field(owner.state, loaded.layout, "occupancy", torch).detach().cpu()
    aggregate_fields = {
        name: _field(owner.state, loaded.layout, name, torch).detach().cpu()
        for name in (
            "resource",
            "patch_field",
            "jungle",
            "eed_field",
            "energy",
            "health",
            "age",
        )
    }
    bytes_per_world = loaded.layout.state_bytes_per_world
    mapping_by_uid = {row["cuda_world_uid"]: row for row in world_mapping["rows"]}
    rows: list[dict[str, Any]] = []
    for world_index, cuda_uid in enumerate(owner.world_uids):
        state_row = state_bytes[
            world_index * bytes_per_world : (world_index + 1) * bytes_per_world
        ]
        mapped = mapping_by_uid[cuda_uid]
        active_population = int(header[world_index, 3].item())
        alive_count = int(alive[world_index].to(torch.int64).sum().item())
        if active_population != alive_count:
            raise WorkhorseScienceDemoError(
                "active population differs from the exact alive projection"
            )
        rows.append(
            {
                "cuda_world_uid": cuda_uid,
                "science_world_uid": mapped["science_world_uid"],
                "semantic_world_id": mapped["semantic_world_id"],
                "cell_id": mapped["cell_id"],
                "logical_world_id": mapped["logical_world_id"],
                "next_absolute_tick": int(header[world_index, 1].item()),
                "status": int(header[world_index, 4].item()),
                "active_population": active_population,
                "alive_count": alive_count,
                "occupied_site_count": int((occupancy[world_index] != 0).sum().item()),
                **{
                    f"{name}_total": int(
                        tensor[world_index].to(torch.int64).sum().item()
                    )
                    for name, tensor in aggregate_fields.items()
                },
                "canonical_world_state_sha256": _sha_bytes(state_row),
            }
        )
    return rows, _sha_bytes(state_bytes)


def _boundary_record(
    *,
    next_absolute_tick: int,
    state_bytes: bytes,
    owner: Any,
    loaded: Any,
    torch: Any,
    parent: Mapping[str, Any],
    demo_contract: Mapping[str, Any],
    event_sink_ack: Mapping[str, Any] | None,
    prior_boundary_record_sha256: str | None,
) -> dict[str, Any]:
    rows, state_sha = _state_projection(
        state_bytes=state_bytes,
        owner=owner,
        loaded=loaded,
        torch=torch,
        world_mapping=demo_contract["world_mapping"],
    )
    if any(row["next_absolute_tick"] != next_absolute_tick for row in rows):
        raise WorkhorseScienceDemoError("boundary world frontier changed")
    if next_absolute_tick == 0:
        if event_sink_ack is not None:
            raise WorkhorseScienceDemoError("tick-zero boundary gained an ACK")
        ack_projection = None
    else:
        if event_sink_ack is None:
            raise WorkhorseScienceDemoError("post-tick boundary lacks its ACK")
        ack_projection = {
            "receipt_path": event_sink_ack["receipt_path"],
            "receipt_file_sha256": event_sink_ack["receipt_file_sha256"],
            "dispatch_identity_sha256": event_sink_ack["dispatch_identity_sha256"],
            "dispatch_content_sha256": event_sink_ack["dispatch_content_sha256"],
            "event_counts": list(event_sink_ack["event_counts"]),
            "post_state_sha256": event_sink_ack["post_state_sha256"],
            "raw_payload_retained": event_sink_ack["raw_payload_retained"],
        }
        if (
            event_sink_ack["end_next_absolute_tick"] != next_absolute_tick
            or event_sink_ack["post_state_sha256"] != state_sha
            or event_sink_ack["raw_payload_retained"] is not False
        ):
            raise WorkhorseScienceDemoError(
                "boundary ACK/state/compact-retention identity changed"
            )
    body = {
        "schema_version": WORKHORSE_SCIENCE_BOUNDARY_RECORD_SCHEMA_V1,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "record_kind": "world_state_snapshot",
        "record_kind_version": 1,
        "time_authority": "absolute_tick_v1",
        "boundary_semantics": "STATE_AFTER_HALF_OPEN_SPAN_0_TO_NEXT_TICK",
        "next_absolute_tick": next_absolute_tick,
        "covered_tick_span": {
            "start_next_tick": 0,
            "end_next_tick": next_absolute_tick,
        },
        "demo_contract_sha256": demo_contract["contract_sha256"],
        "canonical_parent_binding_sha256": parent["binding_sha256"],
        "artifact_sha256": parent["artifact_sha256"],
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "world_mapping_sha256": demo_contract["world_mapping"]["mapping_sha256"],
        "control_table_sha256": demo_contract["static_control_table"][
            "control_table_sha256"
        ],
        "science_route_binding_sha256": demo_contract["science_route_binding_adapter"][
            "route_binding_sha256"
        ],
        "prior_boundary_record_sha256": prior_boundary_record_sha256,
        "event_sink_ack": ack_projection,
        "canonical_state_sha256": state_sha,
        "world_count": len(rows),
        "world_rows": rows,
        "runtime_observation_mode": "OFF",
        "host_projection_after_ack": True,
        "native_capture_runtime_integrated": False,
        "runtime_control_adapter_integrated": False,
    }
    return {**body, "record_sha256": canonical_sha256(body)}


def _write_boundary_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    output = _write_sealed_json(path, record)
    return _artifact_projection(output, content_sha256=str(record["record_sha256"]))


def _checkpoint_sidecar(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_manifest_path: Path,
    demo_contract: Mapping[str, Any],
    parent: Mapping[str, Any],
    boundary_records: Sequence[Mapping[str, Any]],
    continuation_parent_sidecar_sha256: str | None,
) -> dict[str, Any]:
    tick = int(checkpoint["checkpoint_at_tick"])
    prefix = [
        {
            "next_absolute_tick": int(row["next_absolute_tick"]),
            "content_sha256": str(row["content_sha256"]),
        }
        for row in boundary_records
        if int(row["next_absolute_tick"]) <= tick
    ]
    if not prefix or prefix[-1]["next_absolute_tick"] != tick:
        raise WorkhorseScienceDemoError(
            "checkpoint sidecar lacks its exact boundary record"
        )
    body = {
        "schema_version": WORKHORSE_SCIENCE_CHECKPOINT_SIDECAR_SCHEMA_V1,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": tick,
        "base_checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "base_checkpoint_manifest_file_sha256": file_sha256(checkpoint_manifest_path),
        "base_checkpoint_bundle_body_sha256": checkpoint["bundle_body_sha256"],
        "base_checkpoint_identity_sha256": checkpoint["checkpoint_identity"][
            "checkpoint_identity_sha256"
        ],
        "canonical_state_sha256": checkpoint["canonical_state"]["sha256"],
        "canonical_parent_binding_sha256": parent["binding_sha256"],
        "science_checkpoint_compatibility_binding_sha256": demo_contract[
            "science_checkpoint_compatibility_binding"
        ]["binding_sha256"],
        "science_route_binding_sha256": demo_contract["science_route_binding_adapter"][
            "route_binding_sha256"
        ],
        "world_mapping_sha256": demo_contract["world_mapping"]["mapping_sha256"],
        "control_table_sha256": demo_contract["static_control_table"][
            "control_table_sha256"
        ],
        "control_cursor_next_absolute_tick": tick,
        "boundary_record_prefix": prefix,
        "boundary_record_prefix_sha256": canonical_sha256(prefix),
        "typed_capture_inventory_schema": (
            "eiel_workhorse_science_host_boundary_inventory_v1"
        ),
        "typed_capture_inventory_sha256": canonical_sha256(prefix),
        "typed_capture_capacity_rows": len(SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1),
        "typed_capture_committed_rows": len(prefix),
        "capture_overflowed": False,
        "continuation_parent_sidecar_sha256": (continuation_parent_sidecar_sha256),
        "compatibility_identity_fields_complete": True,
        "canonical_science_checkpoint_extension_emitted": False,
        "native_capture_runtime_integrated": False,
        "science_continuable": False,
        "artifact_lifecycle_profile": "SCIENCE_COMPACT",
    }
    return {**body, "sidecar_sha256": canonical_sha256(body)}


def _restore_fresh_owner(
    *,
    loaded: Any,
    torch: Any,
    checkpoint_manifest_path: Path,
    parent: Mapping[str, Any],
    admitted_state_bytes: bytes,
    checkpoint_occurrence_projection: Mapping[str, Any],
    continuation_will_dispatch: bool,
) -> tuple[Any, dict[str, Any]]:
    from .probe import _allocate_owner, _field, _tensor_bytes

    checkpoint, state_bytes = reopen_workhorse_checkpoint_bundle_v2(
        checkpoint_manifest_path
    )
    if (
        state_bytes != admitted_state_bytes
        or checkpoint_occurrence_projection.get("checkpoint_at_tick")
        != checkpoint["checkpoint_at_tick"]
        or checkpoint_occurrence_projection.get("canonical_state_sha256")
        != checkpoint["canonical_state"]["sha256"]
        or checkpoint_occurrence_projection.get("canonical_parent_binding_sha256")
        != parent["binding_sha256"]
        or checkpoint_occurrence_projection.get("canonical_visible") is not True
        or checkpoint_occurrence_projection.get("restore_admitted_exact") is not True
    ):
        raise WorkhorseScienceDemoError(
            "fresh-owner restore did not receive the CRCA-admitted checkpoint"
        )
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(checkpoint["contract"])
    owner = _allocate_owner(torch, loaded, contract.B, contract.temporal_k)
    owner.state.copy_(
        torch.frombuffer(bytearray(admitted_state_bytes), dtype=torch.uint8).reshape_as(
            owner.state
        )
    )
    owner.scratch.zero_()
    owner.world_uids = tuple(checkpoint["world_uids"])
    owner.rng_keys = tuple(checkpoint["rng_keys"])
    owner.committed_next_tick = int(checkpoint["checkpoint_at_tick"])
    owner.attempt_authority_binding = dict(parent)
    loaded.module.restore_derived(owner.state, owner.scratch)
    torch.cuda.synchronize()
    restored = _tensor_bytes(owner.state)
    header = _field(owner.state, loaded.layout, "header", torch).detach().cpu()
    tick = int(checkpoint["checkpoint_at_tick"])
    exact = restored == state_bytes
    frontiers_exact = all(int(value) == tick for value in header[:, 1].tolist())
    statuses_clear = all(int(value) == 0 for value in header[:, 4].tolist())
    if not exact or not frontiers_exact or not statuses_clear:
        raise WorkhorseScienceDemoError("fresh-owner restore is not exact")
    body = {
        "schema_version": WORKHORSE_SCIENCE_RESTORE_RECEIPT_SCHEMA_V1,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "time_authority": "absolute_tick_v1",
        "checkpoint_at_tick": tick,
        "checkpoint_manifest_path": str(checkpoint_manifest_path.resolve()),
        "checkpoint_manifest_file_sha256": file_sha256(checkpoint_manifest_path),
        "checkpoint_bundle_body_sha256": checkpoint["bundle_body_sha256"],
        "checkpoint_identity_sha256": checkpoint["checkpoint_identity"][
            "checkpoint_identity_sha256"
        ],
        "checkpoint_occurrence_digest": dict(
            checkpoint_occurrence_projection["occurrence_digest"]
        ),
        "checkpoint_occurrence_projection_sha256": (
            checkpoint_occurrence_projection["projection_content_sha256"]
        ),
        "crca_restore_admitted": True,
        "source_canonical_state_sha256": checkpoint["canonical_state"]["sha256"],
        "restored_canonical_state_sha256": _sha_bytes(restored),
        "canonical_parent_binding_sha256": parent["binding_sha256"],
        "fresh_device_owner_allocated": True,
        "derived_state_reconstructed": True,
        "canonical_state_exact": exact,
        "world_uids_exact": list(owner.world_uids) == checkpoint["world_uids"],
        "rng_keys_exact": list(owner.rng_keys) == checkpoint["rng_keys"],
        "frontiers_exact": frontiers_exact,
        "statuses_clear": statuses_clear,
        "continuation_will_dispatch": continuation_will_dispatch,
        "terminal_restore_without_further_dispatch": (not continuation_will_dispatch),
        "post_restore_next_absolute_tick": owner.committed_next_tick,
        "runtime_observation_mode": "OFF",
        "native_capture_runtime_integrated": False,
    }
    return owner, {**body, "restore_receipt_sha256": canonical_sha256(body)}


def _population_change_analysis(
    boundary_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trajectories: list[dict[str, Any]] = []
    for world_index in range(SCIENCE_DEMO_B_V1):
        points = [
            {
                "next_absolute_tick": int(record["next_absolute_tick"]),
                "active_population": int(
                    record["world_rows"][world_index]["active_population"]
                ),
                "resource_total": int(
                    record["world_rows"][world_index]["resource_total"]
                ),
                "canonical_world_state_sha256": record["world_rows"][world_index][
                    "canonical_world_state_sha256"
                ],
            }
            for record in boundary_records
        ]
        populations = [point["active_population"] for point in points]
        trajectories.append(
            {
                "logical_world_id": world_index,
                "cuda_world_uid": boundary_records[0]["world_rows"][world_index][
                    "cuda_world_uid"
                ],
                "points": points,
                "population_changed": len(set(populations)) > 1,
                "population_delta": populations[-1] - populations[0],
                "state_changed": len(
                    {point["canonical_world_state_sha256"] for point in points}
                )
                > 1,
            }
        )
    body = {
        "schema_version": "eiel_workhorse_population_change_analysis_v1",
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "analysis_kind": "BOUNDED_TECHNICAL_DEMO_NOT_SCIENCE_RESULT",
        "boundary_next_ticks": list(SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1),
        "world_count": SCIENCE_DEMO_B_V1,
        "population_change_detected_world_count": sum(
            row["population_changed"] for row in trajectories
        ),
        "state_change_detected_world_count": sum(
            row["state_changed"] for row in trajectories
        ),
        "trajectories": trajectories,
    }
    return {**body, "analysis_sha256": canonical_sha256(body)}


def run_and_seal_workhorse_science_compatibility_demo_v1(
    artifact_path: str | Path,
    *,
    envelope_receipt_path: str | Path,
    semantic_inheritance_binding: Mapping[str, Any],
    science_checkpoint_compatibility_binding: Mapping[str, Any],
    demo_contract: Mapping[str, Any],
    canonical_parent_authority_binding: Mapping[str, Any],
    operation_contract_binding: Mapping[str, Any],
    checkpoint_occurrence_adapter: Any,
    event_sink_root: str | Path,
    checkpoint_root: str | Path,
    boundary_record_root: str | Path,
    restore_receipt_root: str | Path,
    result_path: str | Path,
    checkpoint_transport: Any | None = None,
    checkpoint_commit_callback: (
        Callable[[Any, Path, Mapping[str, Any]], None] | None
    ) = None,
) -> dict[str, Any]:
    """Run the exact B10/K1/[0,40) technical compatibility proof."""

    from .canonical_rows import validate_workhorse_live_device_under_parent_v2
    from .probe import (
        _acknowledge_events,
        _advance,
        _allocate_owner,
        _initialize_owner,
        _load,
        _prepare_graph_if_needed,
        _release_graph,
        _tensor_bytes,
        _torch,
    )

    artifact = workhorse_artifact_binding_v2(artifact_path)
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        envelope_receipt_path
    )
    envelope_binding = {
        "path": envelope["receipt_path"],
        "file_sha256": envelope["receipt_file_sha256"],
        "content_sha256": envelope["receipt_content_sha256"],
    }
    contract = validate_workhorse_science_demo_contract_v1(
        demo_contract,
        artifact_binding=artifact,
        envelope_binding=envelope_binding,
        semantic_inheritance_binding=semantic_inheritance_binding,
        science_checkpoint_compatibility_binding=(
            science_checkpoint_compatibility_binding
        ),
    )
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        canonical_parent_authority_binding,
        expected_operation=WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
    )
    if (checkpoint_transport is None) != (checkpoint_commit_callback is None):
        raise WorkhorseScienceDemoError(
            "ACS checkpoint transport and external commit callback must be "
            "configured together"
        )
    operation_binding = dict(operation_contract_binding)
    operation_contract_sha256 = operation_binding.pop("operation_contract_sha256", None)
    expected_operation_artifact = {
        "path": artifact["path"],
        "sha256": artifact["sha256"],
        "sealed_artifact_identity_sha256": artifact["sealed_artifact_identity_sha256"],
    }
    if (
        not _is_sha(operation_contract_sha256)
        or canonical_sha256(operation_binding) != operation_contract_sha256
        or parent["operation_contract_sha256"] != operation_contract_sha256
        or operation_binding.get("operation")
        != WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1
        or operation_binding.get("contract") != parent["contract"]
        or operation_binding.get("artifact") != expected_operation_artifact
        or operation_binding.get("envelope") != envelope_binding
        or operation_binding.get("semantic_inheritance_binding")
        != semantic_inheritance_binding
        or operation_binding.get("science_checkpoint_compatibility_binding")
        != science_checkpoint_compatibility_binding
        or operation_binding.get("science_demo_contract") != contract
        or operation_binding.get("science_continuation_contract_sha256")
        != parent["science_continuation_contract_sha256"]
        or operation_binding.get("source_checkpoint_identity_sha256") is not None
    ):
        raise WorkhorseScienceDemoError(
            "science demo operation differs from its Launchpad-bound contract"
        )
    operation_projection_body = {
        "schema_version": "eiel_workhorse_science_demo_operation_projection_v1",
        "operation": WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
        "operation_contract_sha256": operation_contract_sha256,
        "demo_contract_sha256": contract["contract_sha256"],
        "semantic_inheritance_binding_sha256": semantic_inheritance_binding[
            "binding_sha256"
        ],
        "science_checkpoint_compatibility_binding_sha256": (
            science_checkpoint_compatibility_binding["binding_sha256"]
        ),
        "artifact_sha256": artifact["sha256"],
        "envelope_receipt_content_sha256": envelope_binding["content_sha256"],
        "canonical_parent_binding_sha256": parent["binding_sha256"],
    }
    operation_projection = {
        **operation_projection_body,
        "projection_sha256": canonical_sha256(operation_projection_body),
    }
    execution_contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        parent["contract"]
    )
    expected_execution_contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=SCIENCE_DEMO_SHAPE_ID_V1,
        backend_id=SCIENCE_DEMO_BACKEND_ID_V1,
        B=SCIENCE_DEMO_B_V1,
        temporal_k=SCIENCE_DEMO_TEMPORAL_K_V1,
        total_ticks=SCIENCE_DEMO_TOTAL_TICKS_V1,
        start_next_tick=SCIENCE_DEMO_START_NEXT_TICK_V1,
    )
    if (
        execution_contract != expected_execution_contract
        or parent["artifact_sha256"] != artifact["sha256"]
        or envelope["artifact"] != artifact
        or parent["science_continuation_contract_sha256"]
        != contract["science_checkpoint_compatibility_binding"][
            "science_consumer_extension_contract_sha256"
        ]
    ):
        raise WorkhorseScienceDemoError(
            "science demo parent/artifact/envelope/contract identity changed"
        )
    validate_workhorse_live_device_under_parent_v2(
        parent,
        expected_operation=WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
    )
    for raw_root, label in (
        (event_sink_root, "event sink root"),
        (checkpoint_root, "checkpoint root"),
        (boundary_record_root, "boundary record root"),
        (restore_receipt_root, "restore receipt root"),
    ):
        root = Path(raw_root).expanduser().resolve()
        if checkpoint_transport is not None and label == "checkpoint root":
            if (
                root != checkpoint_transport.checkpoint_root
                or root.is_symlink()
                or not root.is_dir()
                or any(root.iterdir())
            ):
                raise WorkhorseScienceDemoError(
                    "ACS checkpoint transport owns another or nonfresh root"
                )
            continue
        if root.exists() or root.is_symlink():
            raise WorkhorseScienceDemoError(f"fresh {label} already exists")
        root.mkdir(parents=True)

    torch = _torch()
    loaded = _load(artifact["path"], torch)
    owner = _allocate_owner(
        torch, loaded, SCIENCE_DEMO_B_V1, SCIENCE_DEMO_TEMPORAL_K_V1
    )
    _initialize_owner(
        loaded,
        owner,
        world_uid_base=SCIENCE_DEMO_WORLD_UID_BASE_V1,
        start_tick=SCIENCE_DEMO_START_NEXT_TICK_V1,
        rng_key=SCIENCE_DEMO_RNG_KEY_V1,
        fixture=False,
    )
    owner.attempt_authority_binding = dict(parent)
    graph_prepared = False
    boundary_records: list[dict[str, Any]] = []
    boundary_artifacts: list[dict[str, Any]] = []
    event_ack_artifacts: list[dict[str, Any]] = []
    checkpoint_artifacts: list[dict[str, Any]] = []
    checkpoint_sidecar_artifacts: list[dict[str, Any]] = []
    checkpoint_occurrence_artifacts: list[dict[str, Any]] = []
    restore_artifacts: list[dict[str, Any]] = []
    prior_boundary_sha: str | None = None
    prior_sidecar_sha: str | None = None
    prior_occurrence_digest: dict[str, str] | None = None
    last_ack: dict[str, Any] | None = None
    try:
        initial_state = _tensor_bytes(owner.state)
        initial_record = _boundary_record(
            next_absolute_tick=0,
            state_bytes=initial_state,
            owner=owner,
            loaded=loaded,
            torch=torch,
            parent=parent,
            demo_contract=contract,
            event_sink_ack=None,
            prior_boundary_record_sha256=None,
        )
        initial_path = (
            Path(boundary_record_root).resolve()
            / "boundary-next-tick-000000000000.json"
        )
        initial_projection = _write_boundary_record(initial_path, initial_record)
        initial_projection["next_absolute_tick"] = 0
        boundary_records.append(initial_record)
        boundary_artifacts.append(initial_projection)
        prior_boundary_sha = initial_record["record_sha256"]
        _prepare_graph_if_needed(loaded, owner, 1, 0)
        graph_prepared = loaded.backend.backend_id != BACKEND_REFERENCE_PHASED_V2
        for next_tick in range(1, SCIENCE_DEMO_TOTAL_TICKS_V1 + 1):
            _advance(loaded, owner, 1, 0)
            ack = _acknowledge_events(
                owner,
                torch,
                sink_root=Path(event_sink_root).resolve()
                / f"dispatch-{next_tick - 1:012d}-{next_tick:012d}",
                sink_id="workhorse-science-compatibility-demo-v1",
                retain_sink_payload=False,
            )
            last_ack = ack
            event_ack_artifacts.append(
                {
                    "start_next_absolute_tick": next_tick - 1,
                    "end_next_absolute_tick": next_tick,
                    "path": ack["receipt_path"],
                    "file_sha256": ack["receipt_file_sha256"],
                    "content_sha256": ack["receipt_body_sha256"],
                    "raw_payload_retained": ack["raw_payload_retained"],
                }
            )
            if next_tick in SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1:
                state_bytes = _tensor_bytes(owner.state)
                record = _boundary_record(
                    next_absolute_tick=next_tick,
                    state_bytes=state_bytes,
                    owner=owner,
                    loaded=loaded,
                    torch=torch,
                    parent=parent,
                    demo_contract=contract,
                    event_sink_ack=ack,
                    prior_boundary_record_sha256=prior_boundary_sha,
                )
                record_path = Path(boundary_record_root).resolve() / (
                    f"boundary-next-tick-{next_tick:012d}.json"
                )
                projection = _write_boundary_record(record_path, record)
                projection["next_absolute_tick"] = next_tick
                boundary_records.append(record)
                boundary_artifacts.append(projection)
                prior_boundary_sha = record["record_sha256"]
            if next_tick in SCIENCE_DEMO_CHECKPOINT_TICKS_V1:
                if next_tick not in SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1:
                    raise WorkhorseScienceDemoError(
                        "checkpoint lacks a boundary observation"
                    )
                if checkpoint_transport is None:
                    checkpoint_path = write_workhorse_checkpoint_bundle_v2(
                        Path(checkpoint_root).resolve()
                        / f"checkpoint-next-tick-{next_tick:012d}",
                        contract=expected_execution_contract,
                        checkpoint_at_tick=next_tick,
                        state_bytes=state_bytes,
                        artifact_path=artifact["path"],
                        event_sink_receipt_path=ack["receipt_path"],
                        attempt_authority_binding=parent,
                    )
                else:
                    token = checkpoint_transport.capture_workhorse_checkpoint(
                        owner.state,
                        contract=expected_execution_contract,
                        checkpoint_at_tick=next_tick,
                        event_sink_receipt_path=ack["receipt_path"],
                    )
                    if graph_prepared:
                        _release_graph(loaded)
                        graph_prepared = False
                    owner = None
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    detached = checkpoint_transport.settle_checkpoint(token)
                    checkpoint_path = Path(detached["manifest_path"])
                checkpoint, reopened_state = reopen_workhorse_checkpoint_bundle_v2(
                    checkpoint_path
                )
                if reopened_state != state_bytes:
                    raise WorkhorseScienceDemoError(
                        "checkpoint durable reopen changed canonical state"
                    )
                checkpoint_artifacts.append(
                    {
                        "checkpoint_at_tick": next_tick,
                        "manifest_path": str(checkpoint_path.resolve()),
                        "manifest_file_sha256": file_sha256(checkpoint_path),
                        "state_path": str(
                            (checkpoint_path.parent / "canonical_state.bin").resolve()
                        ),
                        "state_file_sha256": file_sha256(
                            checkpoint_path.parent / "canonical_state.bin"
                        ),
                        "bundle_body_sha256": checkpoint["bundle_body_sha256"],
                        "checkpoint_identity_sha256": checkpoint["checkpoint_identity"][
                            "checkpoint_identity_sha256"
                        ],
                    }
                )
                sidecar = _checkpoint_sidecar(
                    checkpoint=checkpoint,
                    checkpoint_manifest_path=checkpoint_path,
                    demo_contract=contract,
                    parent=parent,
                    boundary_records=boundary_artifacts,
                    continuation_parent_sidecar_sha256=prior_sidecar_sha,
                )
                sidecar_path = Path(checkpoint_root).resolve() / (
                    f"science-checkpoint-sidecar-{next_tick:012d}.json"
                )
                _write_sealed_json(sidecar_path, sidecar)
                checkpoint_sidecar_artifacts.append(
                    {
                        "checkpoint_at_tick": next_tick,
                        **_artifact_projection(
                            sidecar_path,
                            content_sha256=sidecar["sidecar_sha256"],
                        ),
                    }
                )
                prior_sidecar_sha = sidecar["sidecar_sha256"]
                occurrence_result = checkpoint_occurrence_adapter.materialize_workhorse_science_checkpoint_occurrence_v1(
                    canonical_parent_authority_binding=parent,
                    checkpoint_manifest_path=checkpoint_path,
                    absolute_bundle_root=Path(checkpoint_root).resolve()
                    / "crca-bundles"
                    / f"checkpoint-tick-{next_tick:012d}",
                    occurrence_root=Path(checkpoint_root).resolve()
                    / "crca-occurrences"
                    / f"checkpoint-tick-{next_tick:012d}",
                    parent_checkpoint_digest=prior_occurrence_digest,
                )
                admitted_state = occurrence_result.get("canonical_state")
                occurrence_projection = occurrence_result.get("projection")
                if (
                    not isinstance(admitted_state, bytes)
                    or admitted_state != state_bytes
                    or not isinstance(occurrence_projection, Mapping)
                    or occurrence_projection.get("checkpoint_at_tick") != next_tick
                ):
                    raise WorkhorseScienceDemoError(
                        "CRCA checkpoint occurrence admission changed state identity"
                    )
                occurrence_projection = dict(occurrence_projection)
                checkpoint_occurrence_artifacts.append(occurrence_projection)
                prior_occurrence_digest = dict(
                    occurrence_projection["occurrence_digest"]
                )
                if checkpoint_transport is not None:
                    assert checkpoint_commit_callback is not None
                    checkpoint_commit_callback(
                        token,
                        checkpoint_path,
                        occurrence_projection,
                    )
                if owner is not None:
                    if graph_prepared:
                        _release_graph(loaded)
                        graph_prepared = False
                    owner = None
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                owner, restore_receipt = _restore_fresh_owner(
                    loaded=loaded,
                    torch=torch,
                    checkpoint_manifest_path=checkpoint_path,
                    parent=parent,
                    admitted_state_bytes=admitted_state,
                    checkpoint_occurrence_projection=occurrence_projection,
                    continuation_will_dispatch=(
                        next_tick in SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1
                    ),
                )
                restore_path = Path(restore_receipt_root).resolve() / (
                    f"restore-next-tick-{next_tick:012d}.json"
                )
                _write_sealed_json(restore_path, restore_receipt)
                restore_artifacts.append(
                    {
                        "checkpoint_at_tick": next_tick,
                        **_artifact_projection(
                            restore_path,
                            content_sha256=restore_receipt["restore_receipt_sha256"],
                        ),
                        "continuation_will_dispatch": restore_receipt[
                            "continuation_will_dispatch"
                        ],
                    }
                )
                if next_tick in SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1:
                    _prepare_graph_if_needed(loaded, owner, 1, 0)
                    graph_prepared = (
                        loaded.backend.backend_id != BACKEND_REFERENCE_PHASED_V2
                    )
                elif next_tick != SCIENCE_DEMO_TOTAL_TICKS_V1:
                    raise WorkhorseScienceDemoError(
                        "nonterminal checkpoint omitted continuation restore"
                    )
        if owner is None or owner.committed_next_tick != SCIENCE_DEMO_TOTAL_TICKS_V1:
            raise WorkhorseScienceDemoError("science demo did not reach tick 40")
        if last_ack is None or last_ack["end_next_absolute_tick"] != 40:
            raise WorkhorseScienceDemoError("science demo terminal ACK is absent")
        analysis = _population_change_analysis(boundary_records)
        body = {
            "schema_version": WORKHORSE_SCIENCE_DEMO_RESULT_SCHEMA_V1,
            "claim_class": _CLAIM_CLASS,
            **_NO_AUTHORITY,
            "operation": WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
            "time_authority": "absolute_tick_v1",
            "runtime_observation_mode": "OFF",
            "native_capture_runtime_integrated": False,
            "runtime_control_adapter_integrated": False,
            "science_route_qualified": False,
            "science_continuable": False,
            "artifact_lifecycle_profile": "SCIENCE_COMPACT",
            "raw_tick_or_event_payloads_retained": False,
            "demo_contract": contract,
            "artifact": artifact,
            "envelope_binding": envelope_binding,
            "semantic_inheritance_binding": dict(semantic_inheritance_binding),
            "operation_contract_projection": operation_projection,
            "canonical_parent_authority_binding": parent,
            "absolute_frontiers": {
                "start_next_tick": 0,
                "target_next_tick": 40,
                "committed_next_tick": 40,
            },
            "boundary_records": boundary_artifacts,
            "event_sink_receipts": event_ack_artifacts,
            "checkpoint_bundles": checkpoint_artifacts,
            "checkpoint_occurrences": checkpoint_occurrence_artifacts,
            "checkpoint_sidecars": checkpoint_sidecar_artifacts,
            "restore_receipts": restore_artifacts,
            "population_change_analysis": analysis,
            "component_preservation": {
                "semantic_transition_sha256": (WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2),
                "full_evolution_components": sorted(FULL_EVOLUTION_CAPABILITIES_V2),
                "disable_mask": 0,
                "inherited_all_component_semantic_gate": True,
                "physical_transition_executed": True,
            },
            "checkpoint_restore_summary": {
                "checkpoint_ticks": list(SCIENCE_DEMO_CHECKPOINT_TICKS_V1),
                "continuation_restore_ticks": list(
                    SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1
                ),
                "terminal_restore_validation_tick": 40,
                "terminal_tick_41_dispatched": False,
                "all_restores_exact": True,
                "all_checkpoint_occurrences_canonical_visible": True,
            },
            **(
                {
                    "acs_checkpoint_transport": {
                        "schema_version": (
                            "eiel_workhorse_science_demo_acs_transport_v1"
                        ),
                        "slots": checkpoint_transport.slots,
                        "capture_mode": "direct_resident_cuda_state_v1",
                        "checkpoint_count": len(checkpoint_artifacts),
                        "owner_released_before_worker_settlement": True,
                    }
                }
                if checkpoint_transport is not None
                else {}
            ),
            "result_class": "TECHNICAL_COMPATIBILITY_DEMO_PASSED",
        }
        result = {**body, "result_content_sha256": canonical_sha256(body)}
        output = _write_sealed_json(result_path, result)
        return validate_workhorse_science_demo_result_v1(output)
    finally:
        try:
            if graph_prepared:
                _release_graph(loaded)
        finally:
            owner = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def validate_workhorse_science_demo_result_v1(
    result_path: str | Path,
) -> dict[str, Any]:
    path, value = _read_json(result_path, "science demo result")
    body = {key: item for key, item in value.items() if key != "result_content_sha256"}
    if (
        value.get("schema_version") != WORKHORSE_SCIENCE_DEMO_RESULT_SCHEMA_V1
        or value.get("claim_class") != _CLAIM_CLASS
        or value.get("operation") != WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1
        or value.get("science_authority_granted") is not False
        or value.get("qualification_authority_granted") is not False
        or value.get("runtime_observation_mode") != "OFF"
        or value.get("native_capture_runtime_integrated") is not False
        or value.get("runtime_control_adapter_integrated") is not False
        or value.get("science_route_qualified") is not False
        or value.get("science_continuable") is not False
        or value.get("artifact_lifecycle_profile") != "SCIENCE_COMPACT"
        or value.get("raw_tick_or_event_payloads_retained") is not False
        or value.get("result_class") != "TECHNICAL_COMPATIBILITY_DEMO_PASSED"
        or value.get("result_content_sha256") != canonical_sha256(body)
    ):
        raise WorkhorseScienceDemoError("science demo result authority/seal changed")
    acs_transport = value.get("acs_checkpoint_transport")
    if acs_transport is not None:
        acs_transport = _exact(
            acs_transport,
            {
                "schema_version",
                "slots",
                "capture_mode",
                "checkpoint_count",
                "owner_released_before_worker_settlement",
            },
            "science demo ACS checkpoint transport",
        )
        if (
            acs_transport["schema_version"]
            != "eiel_workhorse_science_demo_acs_transport_v1"
            or acs_transport["slots"] not in {1, 2}
            or acs_transport["capture_mode"]
            != "direct_resident_cuda_state_v1"
            or acs_transport["checkpoint_count"] != 3
            or acs_transport["owner_released_before_worker_settlement"] is not True
        ):
            raise WorkhorseScienceDemoError(
                "science demo ACS checkpoint transport changed"
            )
    parent = validate_workhorse_canonical_parent_authority_binding_v2(
        value["canonical_parent_authority_binding"],
        expected_operation=WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
    )
    operation_projection = _exact(
        value.get("operation_contract_projection"),
        {
            "schema_version",
            "operation",
            "operation_contract_sha256",
            "demo_contract_sha256",
            "semantic_inheritance_binding_sha256",
            "science_checkpoint_compatibility_binding_sha256",
            "artifact_sha256",
            "envelope_receipt_content_sha256",
            "canonical_parent_binding_sha256",
            "projection_sha256",
        },
        "science operation projection",
    )
    operation_projection_body = {
        key: item
        for key, item in operation_projection.items()
        if key != "projection_sha256"
    }
    artifact = workhorse_artifact_binding_v2(
        value["artifact"]["path"], require_current_source_match=False
    )
    if artifact != value["artifact"] or parent["artifact_sha256"] != artifact["sha256"]:
        raise WorkhorseScienceDemoError("science demo result artifact changed")
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        value["envelope_binding"]["path"]
    )
    if (
        value["envelope_binding"]
        != {
            "path": envelope["receipt_path"],
            "file_sha256": envelope["receipt_file_sha256"],
            "content_sha256": envelope["receipt_content_sha256"],
        }
        or envelope["artifact"] != artifact
    ):
        raise WorkhorseScienceDemoError("science demo result envelope changed")
    validate_workhorse_science_semantic_inheritance_binding_v1(
        value["semantic_inheritance_binding"], require_current_source_match=False
    )
    contract = validate_workhorse_science_demo_contract_v1(
        value["demo_contract"],
        artifact_binding=artifact,
        envelope_binding=value["envelope_binding"],
        semantic_inheritance_binding=value["semantic_inheritance_binding"],
        science_checkpoint_compatibility_binding=value["demo_contract"][
            "science_checkpoint_compatibility_binding"
        ],
        require_current_source_match=False,
    )
    if (
        operation_projection["schema_version"]
        != "eiel_workhorse_science_demo_operation_projection_v1"
        or operation_projection["operation"]
        != WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1
        or operation_projection["operation_contract_sha256"]
        != parent["operation_contract_sha256"]
        or operation_projection["demo_contract_sha256"] != contract["contract_sha256"]
        or operation_projection["semantic_inheritance_binding_sha256"]
        != value["semantic_inheritance_binding"]["binding_sha256"]
        or operation_projection["science_checkpoint_compatibility_binding_sha256"]
        != contract["science_checkpoint_compatibility_binding"]["binding_sha256"]
        or operation_projection["artifact_sha256"] != artifact["sha256"]
        or operation_projection["envelope_receipt_content_sha256"]
        != value["envelope_binding"]["content_sha256"]
        or operation_projection["canonical_parent_binding_sha256"]
        != parent["binding_sha256"]
        or operation_projection["projection_sha256"]
        != canonical_sha256(operation_projection_body)
    ):
        raise WorkhorseScienceDemoError("science operation projection changed")
    if value.get("absolute_frontiers") != {
        "start_next_tick": 0,
        "target_next_tick": 40,
        "committed_next_tick": 40,
    }:
        raise WorkhorseScienceDemoError("science demo result frontiers changed")
    boundaries = value.get("boundary_records")
    if type(boundaries) is not list or [
        row.get("next_absolute_tick") for row in boundaries
    ] != list(SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1):
        raise WorkhorseScienceDemoError("science demo boundary inventory changed")
    prior_sha: str | None = None
    boundary_values: list[dict[str, Any]] = []
    for projection in boundaries:
        record_path, record = _read_json(projection["path"], "science boundary record")
        record_body = {
            key: item for key, item in record.items() if key != "record_sha256"
        }
        if (
            file_sha256(record_path) != projection["file_sha256"]
            or record.get("record_sha256") != projection["content_sha256"]
            or record.get("record_sha256") != canonical_sha256(record_body)
            or record.get("schema_version")
            != WORKHORSE_SCIENCE_BOUNDARY_RECORD_SCHEMA_V1
            or record.get("next_absolute_tick") != projection["next_absolute_tick"]
            or record.get("prior_boundary_record_sha256") != prior_sha
            or record.get("demo_contract_sha256") != contract["contract_sha256"]
            or record.get("world_count") != SCIENCE_DEMO_B_V1
            or len(record.get("world_rows") or []) != SCIENCE_DEMO_B_V1
            or record.get("runtime_observation_mode") != "OFF"
            or record.get("native_capture_runtime_integrated") is not False
        ):
            raise WorkhorseScienceDemoError("science boundary record changed")
        prior_sha = record["record_sha256"]
        boundary_values.append(record)
    acks = value.get("event_sink_receipts")
    if type(acks) is not list or len(acks) != SCIENCE_DEMO_TOTAL_TICKS_V1:
        raise WorkhorseScienceDemoError("science demo ACK inventory changed")
    validated_acks: list[dict[str, Any]] = []
    for index, projection in enumerate(acks):
        ack = validate_workhorse_event_sink_receipt_v2(
            projection["path"],
            expected_artifact_identity_sha256=artifact["identity"]["identity_sha256"],
            expected_end_next_absolute_tick=index + 1,
            expected_attempt_authority_binding_sha256=parent["binding_sha256"],
        )
        if (
            projection["start_next_absolute_tick"] != index
            or projection["end_next_absolute_tick"] != index + 1
            or projection["file_sha256"] != ack["receipt_file_sha256"]
            or projection["content_sha256"] != ack["receipt_body_sha256"]
            or projection["raw_payload_retained"] is not False
            or ack["raw_payload_retained"] is not False
        ):
            raise WorkhorseScienceDemoError("science demo ACK projection changed")
        validated_acks.append(ack)
    for record in boundary_values:
        tick = int(record["next_absolute_tick"])
        if tick == 0:
            if record["event_sink_ack"] is not None:
                raise WorkhorseScienceDemoError("tick-zero boundary gained an ACK")
            continue
        ack = validated_acks[tick - 1]
        expected_ack_projection = {
            "receipt_path": ack["receipt_path"],
            "receipt_file_sha256": ack["receipt_file_sha256"],
            "dispatch_identity_sha256": ack["dispatch_identity_sha256"],
            "dispatch_content_sha256": ack["dispatch_content_sha256"],
            "event_counts": list(ack["event_counts"]),
            "post_state_sha256": ack["post_state_sha256"],
            "raw_payload_retained": False,
        }
        if (
            record["event_sink_ack"] != expected_ack_projection
            or ack["end_next_absolute_tick"] != tick
            or ack["post_state_sha256"] != record["canonical_state_sha256"]
        ):
            raise WorkhorseScienceDemoError(
                "science boundary record is not bound to its exact ACK and state"
            )
    checkpoints = value.get("checkpoint_bundles")
    occurrences = value.get("checkpoint_occurrences")
    sidecars = value.get("checkpoint_sidecars")
    restores = value.get("restore_receipts")
    expected_ticks = list(SCIENCE_DEMO_CHECKPOINT_TICKS_V1)
    if any(
        type(rows) is not list
        or [row.get("checkpoint_at_tick") for row in rows] != expected_ticks
        for rows in (checkpoints, occurrences, sidecars, restores)
    ):
        raise WorkhorseScienceDemoError(
            "science demo checkpoint/sidecar/restore inventory changed"
        )
    prior_sidecar: str | None = None
    prior_occurrence_digest: dict[str, str] | None = None
    for (
        checkpoint_projection,
        occurrence_projection,
        sidecar_projection,
        restore_projection,
    ) in zip(checkpoints, occurrences, sidecars, restores, strict=True):
        checkpoint, state = reopen_workhorse_checkpoint_bundle_v2(
            checkpoint_projection["manifest_path"],
            require_current_source_match=False,
        )
        tick = checkpoint_projection["checkpoint_at_tick"]
        if (
            checkpoint["checkpoint_at_tick"] != tick
            or checkpoint_projection["manifest_file_sha256"]
            != _projected_file_sha256(checkpoint_projection["manifest_path"])
            or checkpoint_projection["state_file_sha256"] != _sha_bytes(state)
            or checkpoint_projection["bundle_body_sha256"]
            != checkpoint["bundle_body_sha256"]
            or checkpoint_projection["checkpoint_identity_sha256"]
            != checkpoint["checkpoint_identity"]["checkpoint_identity_sha256"]
        ):
            raise WorkhorseScienceDemoError("science checkpoint bundle changed")
        occurrence_fields = {
            "schema_version",
            "claim_class",
            "science_authority_granted",
            "qualification_authority_granted",
            "time_authority",
            "checkpoint_at_tick",
            "checkpoint_manifest_path",
            "checkpoint_manifest_file_sha256",
            "checkpoint_bundle_body_sha256",
            "checkpoint_identity_sha256",
            "canonical_state_sha256",
            "canonical_parent_binding_sha256",
            "operation_contract_sha256",
            "absolute_tick_bundle_path",
            "absolute_tick_bundle_digest",
            "occurrence_root",
            "occurrence_digest",
            "occurrence_descriptor_path",
            "occurrence_descriptor_file_sha256",
            "occurrence_commit_path",
            "occurrence_commit_file_sha256",
            "parent_checkpoint_digest",
            "canonical_visible",
            "restore_admitted_exact",
            "projection_content_sha256",
        }
        occurrence = _exact(
            occurrence_projection,
            occurrence_fields,
            "science checkpoint occurrence projection",
        )
        occurrence_body = {
            key: item
            for key, item in occurrence.items()
            if key != "projection_content_sha256"
        }
        bundle_path = Path(occurrence["absolute_tick_bundle_path"]).resolve()
        occurrence_root = Path(occurrence["occurrence_root"]).resolve()
        descriptor_path = Path(occurrence["occurrence_descriptor_path"]).resolve()
        commit_path = Path(occurrence["occurrence_commit_path"]).resolve()
        if (
            occurrence["schema_version"]
            != "eiel_workhorse_science_checkpoint_occurrence_projection_v1"
            or occurrence["claim_class"] != _CLAIM_CLASS
            or occurrence["science_authority_granted"] is not False
            or occurrence["qualification_authority_granted"] is not False
            or occurrence["time_authority"] != "absolute_tick_v1"
            or occurrence["checkpoint_at_tick"] != tick
            or occurrence["checkpoint_manifest_path"]
            != str(Path(checkpoint_projection["manifest_path"]).resolve())
            or occurrence["checkpoint_manifest_file_sha256"]
            != checkpoint_projection["manifest_file_sha256"]
            or occurrence["checkpoint_bundle_body_sha256"]
            != checkpoint["bundle_body_sha256"]
            or occurrence["checkpoint_identity_sha256"]
            != checkpoint["checkpoint_identity"]["checkpoint_identity_sha256"]
            or occurrence["canonical_state_sha256"]
            != checkpoint["canonical_state"]["sha256"]
            or occurrence["canonical_parent_binding_sha256"] != parent["binding_sha256"]
            or occurrence["operation_contract_sha256"]
            != parent["operation_contract_sha256"]
            or bundle_path.name != f"checkpoint-tick-{tick:012d}"
            or descriptor_path.parent != occurrence_root
            or descriptor_path.name != "absolute_tick_checkpoint_occurrence_v1.json"
            or commit_path.parent != occurrence_root
            or commit_path.name != "absolute_tick_checkpoint_occurrence_commit_v1.json"
            or occurrence["occurrence_descriptor_file_sha256"]
            != file_sha256(descriptor_path)
            or occurrence["occurrence_commit_file_sha256"] != file_sha256(commit_path)
            or occurrence["parent_checkpoint_digest"] != prior_occurrence_digest
            or occurrence["canonical_visible"] is not True
            or occurrence["restore_admitted_exact"] is not True
            or occurrence["projection_content_sha256"]
            != canonical_sha256(occurrence_body)
            or not isinstance(occurrence["absolute_tick_bundle_digest"], dict)
            or occurrence["absolute_tick_bundle_digest"].get("algorithm") != "sha256"
            or not _is_sha(occurrence["absolute_tick_bundle_digest"].get("hex"))
            or not isinstance(occurrence["occurrence_digest"], dict)
            or occurrence["occurrence_digest"].get("algorithm") != "sha256"
            or not _is_sha(occurrence["occurrence_digest"].get("hex"))
        ):
            raise WorkhorseScienceDemoError(
                "science checkpoint occurrence projection changed"
            )
        prior_occurrence_digest = dict(occurrence["occurrence_digest"])
        sidecar_path, sidecar = _read_json(
            sidecar_projection["path"], "science checkpoint sidecar"
        )
        sidecar_body = {
            key: item for key, item in sidecar.items() if key != "sidecar_sha256"
        }
        expected_prefix = [
            {
                "next_absolute_tick": row["next_absolute_tick"],
                "content_sha256": row["content_sha256"],
            }
            for row in boundaries
            if row["next_absolute_tick"] <= tick
        ]
        if (
            file_sha256(sidecar_path) != sidecar_projection["file_sha256"]
            or sidecar["sidecar_sha256"] != sidecar_projection["content_sha256"]
            or sidecar["sidecar_sha256"] != canonical_sha256(sidecar_body)
            or sidecar["checkpoint_at_tick"] != tick
            or sidecar["continuation_parent_sidecar_sha256"] != prior_sidecar
            or sidecar["base_checkpoint_manifest_path"]
            != str(Path(checkpoint_projection["manifest_path"]).resolve())
            or sidecar["base_checkpoint_manifest_file_sha256"]
            != checkpoint_projection["manifest_file_sha256"]
            or sidecar["base_checkpoint_bundle_body_sha256"]
            != checkpoint["bundle_body_sha256"]
            or sidecar["base_checkpoint_identity_sha256"]
            != checkpoint["checkpoint_identity"]["checkpoint_identity_sha256"]
            or sidecar["canonical_state_sha256"]
            != checkpoint["canonical_state"]["sha256"]
            or sidecar["canonical_parent_binding_sha256"] != parent["binding_sha256"]
            or sidecar["science_checkpoint_compatibility_binding_sha256"]
            != contract["science_checkpoint_compatibility_binding"]["binding_sha256"]
            or sidecar["science_route_binding_sha256"]
            != contract["science_route_binding_adapter"]["route_binding_sha256"]
            or sidecar["world_mapping_sha256"]
            != contract["world_mapping"]["mapping_sha256"]
            or sidecar["control_table_sha256"]
            != contract["static_control_table"]["control_table_sha256"]
            or sidecar["control_cursor_next_absolute_tick"] != tick
            or sidecar["boundary_record_prefix"] != expected_prefix
            or sidecar["boundary_record_prefix_sha256"]
            != canonical_sha256(expected_prefix)
            or sidecar["typed_capture_inventory_sha256"]
            != canonical_sha256(expected_prefix)
            or sidecar["typed_capture_capacity_rows"]
            != len(SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1)
            or sidecar["typed_capture_committed_rows"] != len(expected_prefix)
            or sidecar["capture_overflowed"] is not False
            or sidecar["canonical_science_checkpoint_extension_emitted"] is not False
            or sidecar["science_continuable"] is not False
        ):
            raise WorkhorseScienceDemoError("science checkpoint sidecar changed")
        prior_sidecar = sidecar["sidecar_sha256"]
        restore_path, restore = _read_json(
            restore_projection["path"], "science restore receipt"
        )
        restore_body = {
            key: item
            for key, item in restore.items()
            if key != "restore_receipt_sha256"
        }
        should_continue = tick in SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1
        if (
            file_sha256(restore_path) != restore_projection["file_sha256"]
            or restore["restore_receipt_sha256"] != restore_projection["content_sha256"]
            or restore["restore_receipt_sha256"] != canonical_sha256(restore_body)
            or restore["checkpoint_at_tick"] != tick
            or restore["checkpoint_occurrence_digest"]
            != occurrence["occurrence_digest"]
            or restore["checkpoint_occurrence_projection_sha256"]
            != occurrence["projection_content_sha256"]
            or restore["crca_restore_admitted"] is not True
            or restore["canonical_state_exact"] is not True
            or restore["world_uids_exact"] is not True
            or restore["rng_keys_exact"] is not True
            or restore["frontiers_exact"] is not True
            or restore["statuses_clear"] is not True
            or restore["continuation_will_dispatch"] is not should_continue
            or restore["terminal_restore_without_further_dispatch"] is should_continue
            or restore["post_restore_next_absolute_tick"] != tick
        ):
            raise WorkhorseScienceDemoError("science restore receipt changed")
    if value.get("population_change_analysis") != _population_change_analysis(
        boundary_values
    ):
        raise WorkhorseScienceDemoError("science population analysis changed")
    if value.get("checkpoint_restore_summary") != {
        "checkpoint_ticks": expected_ticks,
        "continuation_restore_ticks": list(SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1),
        "terminal_restore_validation_tick": 40,
        "terminal_tick_41_dispatched": False,
        "all_restores_exact": True,
        "all_checkpoint_occurrences_canonical_visible": True,
    }:
        raise WorkhorseScienceDemoError("science restore summary changed")
    return {
        **value,
        "result_path": str(path),
        "result_file_sha256": file_sha256(path),
    }


def seal_workhorse_terminalized_science_demo_receipt_v1(
    output_path: str | Path,
    *,
    science_demo_result_path: str | Path,
    canonical_terminal_binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = validate_workhorse_science_demo_result_v1(science_demo_result_path)
    parent = result["canonical_parent_authority_binding"]
    terminal = validate_workhorse_canonical_terminal_binding_v2(
        canonical_terminal_binding, parent=parent
    )
    target = SCIENCE_DEMO_TOTAL_TICKS_V1
    if terminal["attempt_terminal_status"] != "COMPLETED" or any(
        terminal["final_frontiers"][field] != target
        for field in (
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        )
    ):
        raise WorkhorseScienceDemoError(
            "science demo requires exact completed tick-40 terminalization"
        )
    body = {
        "schema_version": WORKHORSE_SCIENCE_DEMO_TERMINAL_RECEIPT_SCHEMA_V1,
        "claim_class": _CLAIM_CLASS,
        **_NO_AUTHORITY,
        "operation": WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
        "result_class": "TECHNICAL_COMPATIBILITY_DEMO_PASSED",
        "time_authority": "absolute_tick_v1",
        "artifact_lifecycle_profile": "SCIENCE_COMPACT",
        "science_demo_result": {
            "path": result["result_path"],
            "file_sha256": result["result_file_sha256"],
            "content_sha256": result["result_content_sha256"],
        },
        "canonical_parent_authority_binding": parent,
        "canonical_terminal_binding": terminal,
        "boundary_next_ticks": list(SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1),
        "checkpoint_ticks": list(SCIENCE_DEMO_CHECKPOINT_TICKS_V1),
        "restore_ticks": list(SCIENCE_DEMO_CHECKPOINT_TICKS_V1),
        "terminal_tick_41_dispatched": False,
        "runtime_observation_mode": "OFF",
        "native_capture_runtime_integrated": False,
        "runtime_control_adapter_integrated": False,
        "science_route_qualified": False,
        "science_continuable": False,
        "population_change_analysis": result["population_change_analysis"],
        "component_preservation": result["component_preservation"],
    }
    receipt = {**body, "receipt_content_sha256": canonical_sha256(body)}
    path = _write_sealed_json(output_path, receipt)
    return validate_workhorse_terminalized_science_demo_receipt_v1(path)


def validate_workhorse_terminalized_science_demo_receipt_v1(
    receipt_path: str | Path,
) -> dict[str, Any]:
    path, value = _read_json(receipt_path, "terminalized science demo receipt")
    body = {key: item for key, item in value.items() if key != "receipt_content_sha256"}
    if (
        value.get("schema_version") != WORKHORSE_SCIENCE_DEMO_TERMINAL_RECEIPT_SCHEMA_V1
        or value.get("claim_class") != _CLAIM_CLASS
        or value.get("science_authority_granted") is not False
        or value.get("qualification_authority_granted") is not False
        or value.get("operation") != WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1
        or value.get("result_class") != "TECHNICAL_COMPATIBILITY_DEMO_PASSED"
        or value.get("receipt_content_sha256") != canonical_sha256(body)
        or value.get("terminal_tick_41_dispatched") is not False
        or value.get("runtime_observation_mode") != "OFF"
        or value.get("native_capture_runtime_integrated") is not False
        or value.get("runtime_control_adapter_integrated") is not False
        or value.get("science_route_qualified") is not False
        or value.get("science_continuable") is not False
    ):
        raise WorkhorseScienceDemoError(
            "terminalized science demo authority or seal changed"
        )
    result = validate_workhorse_science_demo_result_v1(
        value["science_demo_result"]["path"]
    )
    if value["science_demo_result"] != {
        "path": result["result_path"],
        "file_sha256": result["result_file_sha256"],
        "content_sha256": result["result_content_sha256"],
    }:
        raise WorkhorseScienceDemoError("terminal science result binding changed")
    terminal = validate_workhorse_canonical_terminal_binding_v2(
        value["canonical_terminal_binding"],
        parent=result["canonical_parent_authority_binding"],
    )
    if terminal["attempt_terminal_status"] != "COMPLETED" or any(
        terminal["final_frontiers"][field] != 40
        for field in (
            "last_dispatched_next_tick",
            "last_computed_next_tick",
            "durable_checkpoint_next_tick",
            "published_artifact_next_tick",
        )
    ):
        raise WorkhorseScienceDemoError("terminal science frontiers changed")
    if (
        value.get("population_change_analysis") != result["population_change_analysis"]
        or value.get("component_preservation") != result["component_preservation"]
    ):
        raise WorkhorseScienceDemoError("terminal science summary changed")
    return {
        **value,
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha256(path),
    }


__all__ = [
    "SCIENCE_DEMO_B_V1",
    "SCIENCE_DEMO_BOUNDARY_NEXT_TICKS_V1",
    "SCIENCE_DEMO_CHECKPOINT_TICKS_V1",
    "SCIENCE_DEMO_CONTINUATION_RESTORE_TICKS_V1",
    "SCIENCE_DEMO_TOTAL_TICKS_V1",
    "WORKHORSE_SCIENCE_BOUNDARY_RECORD_SCHEMA_V1",
    "WORKHORSE_SCIENCE_CHECKPOINT_SIDECAR_SCHEMA_V1",
    "WORKHORSE_SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1",
    "WORKHORSE_SCIENCE_DEMO_CONTRACT_SCHEMA_V1",
    "WORKHORSE_SCIENCE_DEMO_RESULT_SCHEMA_V1",
    "WORKHORSE_SCIENCE_DEMO_TERMINAL_RECEIPT_SCHEMA_V1",
    "WORKHORSE_SCIENCE_RESTORE_RECEIPT_SCHEMA_V1",
    "WorkhorseScienceDemoError",
    "run_and_seal_workhorse_science_compatibility_demo_v1",
    "seal_workhorse_terminalized_science_demo_receipt_v1",
    "validate_workhorse_science_demo_contract_v1",
    "validate_workhorse_science_demo_result_v1",
    "validate_workhorse_science_semantic_inheritance_binding_v1",
    "validate_workhorse_terminalized_science_demo_receipt_v1",
    "workhorse_science_demo_contract_v1",
    "workhorse_science_semantic_inheritance_binding_v1",
]
