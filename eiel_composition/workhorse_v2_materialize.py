#!/usr/bin/env python3
"""Compose fresh canonical Launchpad requests for Workhorse V2 operations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from engine.campaigns.cuda_full_evolution_v2_workhorse_provider import (
    ACS_NATIVE_RUNTIME_ABI_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1,
    ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
    DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
    INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2,
    INDEXED_ACS_CONTINUATION_OPERATION_V2,
    FullEvolutionCudaWorkhorseContractV2,
    OPERATION_EXTENSION_ID,
    OPERATION_SCHEMA_V2,
    OPERATION_SCHEMA_V3,
    SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
    SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
    WORLD_BASE,
    canonical_sha256,
    compile_workhorse_absolute_dispatch_plan_v2,
    cuda_full_evolution_workhorse_v2_profile_registry_v1,
    cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2,
    cuda_device_identity_v2,
    validate_authoritative_micro_denominator_v2,
    validate_workhorse_device_envelope_calibration_receipt_v2,
    inspect_workhorse_indexed_checkpoint_set_v1,
    validate_workhorse_semantic_gate_receipt_v2,
    validate_workhorse_terminalized_semantic_case_receipt_v2,
    workhorse_artifact_binding_v2,
    workhorse_backend_temporal_k_candidates_v2,
    workhorse_fixed_tick_measurement_schedule_v3,
    workhorse_exact_founder_bootstrap_contract_v1,
    workhorse_indexed_continuation_contract_v2,
    workhorse_repack_binding_v2,
    workhorse_science_checkpoint_compatibility_binding_v2,
    workhorse_science_demo_contract_v1,
    workhorse_science_semantic_inheritance_binding_v1,
    workhorse_semantic_case_law_v2,
    workhorse_semantic_case_operation_contract_v2,
    write_workhorse_semantic_case_aggregate_gate_receipt_v2,
)
from engine.campaigns.execution_front_door_v1 import ExecutionRequestV2
from engine.campaigns.compiler import compile_campaign
from engine.campaigns.cuda_full_evolution_v2_workhorse_resources import (
    workhorse_checkpoint_storage_accounting_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.continuation import (
    workhorse_scientific_control_contract_v1,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / (
    "_spec/campaigns/v1/tw_cuda_full_evolution_v2/micro_qualprep_v1.template.yaml"
)


def _workhorse_wall_seconds_upper_bound_v2(total_ticks: int) -> int:
    """Mirror the provider's fail-closed bound for a semantic or timing row."""

    if type(total_ticks) is not int or total_ticks <= 0:
        raise ValueError("Workhorse total_ticks must be a positive integer")
    return max(300, total_ticks * 60)


def _row_checkpoint_ticks_v1(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    exact_founder_bootstrap: bool,
    acs_slots: int,
    checkpoint_retention_class: str,
) -> tuple[int, ...]:
    """Return the explicit regular-row checkpoint schedule.

    Ordinary rows retain their existing terminal-only bytes.  The opt-in
    founder bootstrap publishes one real first-dispatch occurrence and one
    terminal occurrence, both protected while an exact selected parent is
    materialized from the completed canonical run.
    """

    target_tick = contract.start_next_tick + contract.total_ticks
    if not exact_founder_bootstrap:
        return (target_tick,)
    if (
        contract.start_next_tick != 0
        or acs_slots != 2
        or checkpoint_retention_class != "R4"
    ):
        raise ValueError(
            "exact founder bootstrap requires start-next-tick 0, ACS2, and R4"
        )
    genesis_tick = contract.temporal_k
    if genesis_tick >= target_tick:
        raise ValueError(
            "exact founder bootstrap requires a terminal tick after first dispatch"
        )
    return (genesis_tick, target_tick)


def _baseline(*, acs_slots: int = 0) -> dict[str, Any]:
    if type(acs_slots) is not int or acs_slots not in {0, 1, 2}:
        raise ValueError("Workhorse ACS slots must be zero, one, or two")
    deviations = [
        {
            "requirement_id": "evolution.full_evolution",
            "reason": (
                "Workhorse V2 preserves all seven capabilities under a new "
                "unqualified shape-specialized transition route."
            ),
        },
        {
            "requirement_id": "runtime.production_cuda",
            "reason": (
                "The Workhorse V2 technical provider is not the qualified "
                "production provider."
            ),
        },
        {
            "requirement_id": "science.wrc_variability",
            "reason": (
                "This calibration or semantic route grants no WRC science "
                "authority and runs with observation OFF."
            ),
        },
    ]
    if acs_slots != 2:
        deviations.append(
            {
                "requirement_id": "checkpoint.acs_two_slot",
                "reason": (
                    "Workhorse V2 uses one detached ACS slot for this "
                    "technical execution; it is not the two-slot Baseline."
                    if acs_slots == 1
                    else (
                        "The Workhorse V2 absolute-tick checkpoint route is "
                        "running synchronously with ACS disabled."
                    )
                ),
            }
        )
    deviations.append(
        {
            "requirement_id": "authority.device_fallback",
            "reason": (
                "A sealed exact-device Workhorse receipt replaces fallback "
                "authority for this technical-only route."
            ),
        }
    )
    return {
        "schema_version": "eiel_baseline_declaration_v1",
        "profile_id": "EIEL_PRODUCTION_BASELINE_V1",
        "intent": "development",
        "deviations": deviations,
    }


def _live_device_identity() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return dict(cuda_device_identity_v2(torch, torch.cuda.current_device()))


def _fresh_control_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root.exists() or root.is_symlink():
        raise RuntimeError(f"control root must be fresh: {root}")
    root.mkdir(parents=True, mode=0o700)
    return root


def _write_outputs(
    control_root: Path,
    *,
    spec: dict[str, Any],
    run_root: Path,
    purpose: str,
    restore_process_isolation: str = "inline",
    acs_slots: int = 0,
) -> None:
    spec_path = control_root / "campaign.yaml"
    with spec_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)
    request_mapping: dict[str, Any] = {
        "schema_version": "eiel_execution_request_v2",
        "purpose": purpose,
        "campaign_spec": str(spec_path),
        "output_root": str(run_root),
        "resource_wait": {
            "enabled": True,
            "poll_seconds": 300,
            "max_wait_seconds": None,
        },
        "qualification_profile": None,
        "baseline": _baseline(acs_slots=acs_slots),
    }
    if restore_process_isolation != "inline":
        request_mapping["restore_process_isolation"] = restore_process_isolation
    request = ExecutionRequestV2.from_mapping(request_mapping)
    request_path = control_root / "execution-request-v2.json"
    with request_path.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(request.payload(), sort_keys=True) + "\n")
    print(request_path)


def _write_science_authority_base_plan_v1(
    control_root: Path,
    *,
    spec: dict[str, Any],
    shape_id: str,
) -> Path:
    """Publish one no-execution Workhorse plan for existing science authority.

    The output is an ordinary compiled campaign plan.  It creates no execution
    request, installs no authority, and introduces no new contract.  The
    existing Science Request authority installer remains the only activation
    surface.
    """

    spec_path = control_root / "campaign.yaml"
    with spec_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)
    plan = compile_campaign(
        spec,
        registry=cuda_full_evolution_workhorse_v2_profile_registry_v1(
            shape_id=shape_id,
            operation="EXECUTION_ROW",
        ),
        planning_resource_accounting=(
            cuda_full_evolution_workhorse_v2_signed_planning_resource_accounting_v2
        ),
    ).payload()
    if (
        plan.get("artifact_lifecycle", {}).get("lifecycle_profile")
        != "SCIENCE_COMPACT"
        or plan.get("execution", {}).get("acs_checkpoint_slots") != 2
        or plan.get("execution", {}).get("native_streamed_checkpoint") is not True
        or plan.get("analysis", {}).get("extension", {}).get("contract", {}).get(
            "operation"
        )
        != "EXECUTION_ROW"
    ):
        raise RuntimeError(
            "science authority base plan lost its Workhorse SCIENCE_COMPACT boundary"
        )
    plan_path = control_root / "science-authority-base-plan.json"
    with plan_path.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n")
    print(plan_path)
    return plan_path


def _base_spec(
    *,
    campaign_id: str,
    artifact_binding: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    spec = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    spec["campaign_id"] = campaign_id
    spec["description"] = (
        "Fresh Workhorse V2 shape-specialized technical operation through the "
        "canonical ExecutionRequestV2 route."
    )
    spec["no_claims"] = (
        "TECHNICAL_ONLY. No production, Qualification, or science authority."
    )
    spec["world"] = {"base": WORLD_BASE, "overlays": []}
    spec["matrix"]["dimensions"] = [
        {"id": "ecology_seed", "values": [101]},
        {"id": "task_seed", "values": [100001]},
    ]
    spec["matrix"]["expected_cells"] = 1
    spec["matrix"]["max_cells"] = 1
    spec["execution"]["batch"] = {"mode": "explicit", "cells": 1}
    spec["artifacts"] = {
        "workhorse_artifact": {
            "path": artifact_binding["path"],
            "sha256": artifact_binding["sha256"],
            "kind": "cuda_extension",
            "abi": "eiel_full_evolution_cuda_workhorse_abi_v2",
        }
    }
    spec["output"]["runtime_root"] = str(run_root)
    return spec


def _apply_workhorse_storage_accounting_v2(
    spec: dict[str, Any], accounting: dict[str, Any]
) -> None:
    """Bind provider-equivalent checkpoint storage to execution/lifecycle quotas."""

    execution = spec.get("execution")
    lifecycle = spec.get("artifact_lifecycle")
    storage = lifecycle.get("storage") if isinstance(lifecycle, dict) else None
    budgets = execution.get("budgets") if isinstance(execution, dict) else None
    if not isinstance(storage, dict) or not isinstance(budgets, dict):
        raise RuntimeError("Workhorse storage budgets are unavailable")
    scratch = max(
        1, int(accounting["acs_pending_capture_file_bytes_upper_bound"])
    )
    checkpoint = max(1, int(accounting["checkpoint_bytes_upper_bound"]))
    durable = max(
        1, int(accounting["durable_noncheckpoint_bytes_upper_bound"])
    )
    reserve = int(storage.get("emergency_reserve_bytes", 0))
    if reserve <= 0:
        raise RuntimeError("Workhorse lifecycle emergency reserve is unavailable")
    disk_payload_total = scratch + checkpoint + durable
    if disk_payload_total < int(accounting["disk_bytes_upper_bound"]):
        raise RuntimeError("Workhorse lifecycle storage partition lost disk bytes")
    # The lifecycle contract defines total_max_bytes as the complete partition,
    # including its protected emergency reserve.  The execution budget must bind
    # that same value even though the provider's disk upper bound describes only
    # bytes the run may materialize.
    total = disk_payload_total + reserve
    budgets["disk_bytes"] = total
    storage.update(
        {
            "total_max_bytes": total,
            "scratch_max_bytes": scratch,
            "checkpoint_max_bytes": checkpoint,
            "durable_max_bytes": durable,
        }
    )


def _build_indexed_continuation_operation_v2(
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    artifact: dict[str, Any],
    envelope: dict[str, Any],
    authoritative_micro: dict[str, Any],
    semantic_gate: dict[str, Any],
    science_continuation_sha256: str,
    continuation: dict[str, Any],
) -> dict[str, Any]:
    """Construct the exact provider object shared by both canonical carriers.

    Inputs are already reopened and validated by the caller.  Keeping this
    constructor pure prevents the CLI and Science Request lowering paths from
    drifting without adding another public or durable contract.
    """

    return {
        "schema_version": OPERATION_SCHEMA_V3,
        "operation": INDEXED_ACS_CONTINUATION_OPERATION_V2,
        "contract": contract.to_dict(),
        "artifact": {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "sealed_artifact_identity_sha256": artifact[
                "sealed_artifact_identity_sha256"
            ],
        },
        "envelope": {
            "path": envelope["receipt_path"],
            "file_sha256": envelope["receipt_file_sha256"],
            "content_sha256": envelope["receipt_content_sha256"],
        },
        "authoritative_micro_receipt": {
            "path": authoritative_micro["receipt_path"],
            "file_sha256": authoritative_micro["receipt_file_sha256"],
            "content_sha256": authoritative_micro["receipt_content_sha256"],
        },
        "semantic_gate_receipt": {
            "path": semantic_gate["receipt_path"],
            "file_sha256": semantic_gate["receipt_file_sha256"],
            "content_sha256": semantic_gate["semantic_gate_receipt_sha256"],
        },
        "science_continuation_contract_sha256": science_continuation_sha256,
        "source_checkpoint_identity_sha256": continuation[
            "science_continuation_identity"
        ]["source_world_state_set_sha256"],
        "continuation_contract": continuation,
    }


def _bind_acs_native_runtime_artifacts(
    spec: dict[str, Any],
    *,
    acs_slots: int,
    extension_path: str | None,
) -> None:
    """Bind the exact AOT native transport selected by an ACS Workhorse run."""

    if acs_slots == 0:
        if extension_path is not None:
            raise ValueError(
                "--acs-native-runtime-extension requires --acs-slots 1 or 2"
            )
        return
    if not extension_path:
        raise ValueError(
            "ACS Workhorse execution requires --acs-native-runtime-extension"
        )
    declared = Path(extension_path).expanduser()
    if declared.is_symlink() or not declared.is_file():
        raise ValueError("ACS native runtime extension must be a regular file")
    extension = declared.resolve()
    manifest = extension.with_suffix(extension.suffix + ".manifest.json")
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("ACS native runtime extension manifest is missing")
    artifacts = spec.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Workhorse campaign artifact mapping is unavailable")
    artifacts[ACS_NATIVE_RUNTIME_EXTENSION_ARTIFACT_ID_V1] = {
        "path": str(extension),
        "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "kind": ACS_NATIVE_RUNTIME_EXTENSION_KIND_V1,
        "abi": ACS_NATIVE_RUNTIME_ABI_V1,
    }
    artifacts[ACS_NATIVE_RUNTIME_MANIFEST_ARTIFACT_ID_V1] = {
        "path": str(manifest.resolve()),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "kind": ACS_NATIVE_RUNTIME_MANIFEST_KIND_V1,
        "abi": ACS_NATIVE_RUNTIME_ABI_V1,
    }


def materialize_calibration(args: argparse.Namespace) -> None:
    artifact = workhorse_artifact_binding_v2(args.artifact)
    identity = artifact["identity"]
    shape_id = str(identity["shape_id"])
    backend_id = str(identity["backend_id"])
    compiled_k = list(workhorse_backend_temporal_k_candidates_v2(shape_id, backend_id))
    total_ticks = len(compiled_k)
    run_root = Path(args.run_root).expanduser().resolve()
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(f"runtime root must be fresh: {run_root}")
    control_root = _fresh_control_root(args.control_root)
    device_identity = _live_device_identity()
    effective_reserve = max(
        args.reserve_bytes,
        (device_identity["total_memory_bytes"] + 3) // 4,
    )
    spec = _base_spec(
        campaign_id=args.campaign_id,
        artifact_binding=artifact,
        run_root=run_root,
    )
    spec["execution"]["budgets"]["vram_bytes"] = (
        device_identity["total_memory_bytes"] - effective_reserve
    )
    spec["execution"]["budgets"]["minimum_free_gpu_bytes"] = effective_reserve
    operation = {
        "schema_version": OPERATION_SCHEMA_V2,
        "operation": DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
        "shape_id": shape_id,
        "backend_id": backend_id,
        "artifact": {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "sealed_artifact_identity_sha256": artifact[
                "sealed_artifact_identity_sha256"
            ],
        },
        "compiled_temporal_k": compiled_k,
        "device_identity": device_identity,
        "reserve_policy": {
            "mode": "STABLE_TOTAL_MINUS_EFFECTIVE_RESERVE_V2",
            "reserve_bytes_requested": args.reserve_bytes,
            "minimum_fraction_numerator": 1,
            "minimum_fraction_denominator": 4,
            "transient_free_bytes_is_plan_authority": False,
        },
        "absolute_tick_span": {
            "time_authority": "absolute_tick_v1",
            "start_next_tick": 0,
            "target_next_tick": total_ticks,
            "total_ticks": total_ticks,
        },
        "observation_mode": "OFF",
        "science_continuation_claimed": False,
        "science_authority_granted": False,
        "qualification_authority_granted": False,
    }
    spec["parameters"] = {
        "workhorse.operation": DEVICE_ENVELOPE_CALIBRATION_OPERATION_V2,
        "workhorse.shape_id": shape_id,
        "workhorse.backend_id": backend_id,
        "workhorse.compiled_temporal_k_sha256": canonical_sha256(compiled_k),
        "workhorse.reserve_bytes": args.reserve_bytes,
    }
    spec["execution"].update(
        {
            "start_next_tick": 0,
            "total_ticks": total_ticks,
            "checkpoint_ticks": [total_ticks],
            "resident_owner_span_ticks": total_ticks,
            "native_dispatch_window_ticks": total_ticks,
            "source_window_ticks": total_ticks,
            "event_window_capacity": 1,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": total_ticks, "checkpoint_ticks": [total_ticks]}
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [total_ticks],
        "roles": [
            {
                "identity": f"checkpoint-tick-{total_ticks:012d}",
                "checkpoint_tick": total_ticks,
                "role": "terminal",
                "retention_class": "R2",
            }
        ],
    }
    spec["analysis"] = {
        "required_metrics": ["workhorse.device_envelope_calibration_completed"],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": deepcopy(operation),
            "contract_sha256": canonical_sha256(operation),
        },
    }
    _write_outputs(
        control_root,
        spec=spec,
        run_root=run_root,
        purpose="benchmark",
    )


def materialize_semantic_case(args: argparse.Namespace) -> None:
    law = workhorse_semantic_case_law_v2(
        args.case_id, semantic_temporal_k=args.semantic_temporal_k
    )
    artifact_path = (
        args.reference_artifact
        if law.artifact_role == "REFERENCE"
        else args.candidate_artifact
    )
    envelope_path = (
        args.reference_envelope
        if law.artifact_role == "REFERENCE"
        else args.candidate_envelope
    )
    artifact = workhorse_artifact_binding_v2(artifact_path)
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(envelope_path)
    if envelope["artifact"] != artifact:
        raise RuntimeError("semantic artifact differs from its device envelope")
    shape_id = str(artifact["identity"]["shape_id"])
    backend_id = str(artifact["identity"]["backend_id"])
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape_id,
        backend_id=backend_id,
        B=law.B,
        temporal_k=law.dispatch_temporal_k,
        total_ticks=law.total_ticks,
        start_next_tick=law.start_next_absolute_tick,
    )
    external_source = None
    source_identity = None
    if law.requires_source_checkpoint:
        if args.source_case_receipt is None:
            raise RuntimeError("restore semantic case requires case-20 receipt")
        source_receipt = validate_workhorse_terminalized_semantic_case_receipt_v2(
            args.source_case_receipt
        )
        if source_receipt["case_id"] != "checkpoint_same_restore_sequence":
            raise RuntimeError("restore source must be the exact case-20 receipt")
        external_source = source_receipt["produced_external_source_checkpoint_binding"]
        if not isinstance(external_source, dict):
            raise RuntimeError("case-20 receipt lacks its external source binding")
        source_identity = external_source["base_checkpoint_identity_sha256"]
    elif args.source_case_receipt is not None:
        raise RuntimeError("non-restore semantic case cannot bind a source receipt")
    science = workhorse_science_checkpoint_compatibility_binding_v2(REPO_ROOT)
    semantic_case = workhorse_semantic_case_operation_contract_v2(
        case_id=law.case_id,
        shape_id=shape_id,
        semantic_temporal_k=args.semantic_temporal_k,
    )
    operation = {
        "schema_version": OPERATION_SCHEMA_V2,
        "operation": SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
        "semantic_case_contract": semantic_case,
        "contract": contract.to_dict(),
        "artifact": {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "sealed_artifact_identity_sha256": artifact[
                "sealed_artifact_identity_sha256"
            ],
        },
        "envelope": {
            "path": envelope["receipt_path"],
            "file_sha256": envelope["receipt_file_sha256"],
            "content_sha256": envelope["receipt_content_sha256"],
        },
        "science_checkpoint_compatibility_binding": science,
        "science_continuation_contract_sha256": science[
            "science_consumer_extension_contract_sha256"
        ],
        "source_checkpoint_identity_sha256": source_identity,
        "external_source_checkpoint_binding": external_source,
    }
    run_root = Path(args.run_root).expanduser().resolve()
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(f"runtime root must be fresh: {run_root}")
    control_root = _fresh_control_root(args.control_root)
    spec = _base_spec(
        campaign_id=args.campaign_id,
        artifact_binding=artifact,
        run_root=run_root,
    )
    spec["execution"]["budgets"]["wall_seconds"] = max(
        int(spec["execution"]["budgets"]["wall_seconds"]),
        _workhorse_wall_seconds_upper_bound_v2(contract.total_ticks),
    )
    target_tick = law.start_next_absolute_tick + law.total_ticks
    spec["parameters"] = {
        "workhorse.operation": SEMANTIC_EVIDENCE_CASE_OPERATION_V2,
        "workhorse.shape_id": shape_id,
        "workhorse.backend_id": backend_id,
        "workhorse.semantic_case_id": law.case_id,
        "workhorse.B": law.B,
        "workhorse.temporal_k": law.dispatch_temporal_k,
        "workhorse.semantic_temporal_k": args.semantic_temporal_k,
        "workhorse.attempt_terminal_status": law.expected_attempt_terminal_status,
    }
    spec["execution"].update(
        {
            "start_next_tick": law.start_next_absolute_tick,
            "total_ticks": target_tick,
            "checkpoint_ticks": [target_tick],
            "resident_owner_span_ticks": law.total_ticks,
            "native_dispatch_window_ticks": law.dispatch_temporal_k,
            "source_window_ticks": law.dispatch_temporal_k,
            "event_window_capacity": 2 * contract.shape.capacity,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": target_tick, "checkpoint_ticks": [target_tick]}
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [target_tick],
        "roles": [
            {
                "identity": f"checkpoint-tick-{target_tick:012d}",
                "checkpoint_tick": target_tick,
                "role": "terminal",
                "retention_class": "R2",
            }
        ],
    }
    spec["analysis"] = {
        "required_metrics": ["workhorse.semantic_case_provisional_completed"],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": deepcopy(operation),
            "contract_sha256": canonical_sha256(operation),
        },
    }
    _write_outputs(
        control_root,
        spec=spec,
        run_root=run_root,
        purpose="validation",
    )


def materialize_semantic_gate(args: argparse.Namespace) -> None:
    output = write_workhorse_semantic_case_aggregate_gate_receipt_v2(
        args.output,
        aggregate_path=args.aggregate,
        reference_artifact_path=args.reference_artifact,
        candidate_artifact_path=args.candidate_artifact,
    )
    print(output)


def materialize_row(args: argparse.Namespace) -> None:
    artifact = workhorse_artifact_binding_v2(args.artifact)
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(args.envelope)
    if envelope["artifact"] != artifact:
        raise RuntimeError("row artifact differs from its device envelope")
    identity = artifact["identity"]
    shape_id = str(identity["shape_id"])
    backend_id = str(identity["backend_id"])
    semantic = validate_workhorse_semantic_gate_receipt_v2(
        args.semantic_gate,
        expected_shape_id=shape_id,
        expected_candidate_artifact_sha256=artifact["sha256"],
        expected_candidate_backend_id=backend_id,
    )
    micro = validate_authoritative_micro_denominator_v2(
        args.micro_receipt,
        expected_device_identity=envelope["device_identity"],
    )
    science = workhorse_science_checkpoint_compatibility_binding_v2(REPO_ROOT)
    schedule = workhorse_fixed_tick_measurement_schedule_v3(
        temporal_k=args.temporal_k,
        warmup_ticks=args.warmup_ticks,
        measurement_window_ticks=args.measurement_window_ticks,
        measured_windows=args.measured_windows,
    )
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape_id,
        backend_id=backend_id,
        B=args.B,
        temporal_k=args.temporal_k,
        total_ticks=schedule["total_ticks"],
        start_next_tick=args.start_next_tick,
    )
    science_authority_base = bool(
        getattr(args, "science_authority_base_plan", False)
    )
    if science_authority_base and (
        contract.B != 1
        or contract.start_next_tick != 0
        or args.acs_slots != 2
        or bool(getattr(args, "exact_founder_bootstrap", False))
        or args.checkpoint_retention_class != "R2"
    ):
        raise ValueError(
            "science authority base plan requires B1, start-next-tick 0, "
            "ACS2, ordinary row mode, and R2"
        )
    admitted = next(
        (
            row
            for row in envelope["expected_rows"]
            if row["B"] == contract.B and row["temporal_k"] == contract.temporal_k
        ),
        None,
    )
    if admitted is None or admitted["planned_admission"].get("admitted") is not True:
        raise RuntimeError("row is not admitted by the exact device envelope")
    checkpoint_ticks = _row_checkpoint_ticks_v1(
        contract,
        exact_founder_bootstrap=bool(
            getattr(args, "exact_founder_bootstrap", False)
        ),
        acs_slots=args.acs_slots,
        checkpoint_retention_class=args.checkpoint_retention_class,
    )
    exact_founder_bootstrap = bool(
        getattr(args, "exact_founder_bootstrap", False)
    )
    operation = {
        "schema_version": (
            OPERATION_SCHEMA_V3 if exact_founder_bootstrap else OPERATION_SCHEMA_V2
        ),
        "operation": "EXECUTION_ROW",
        "contract": contract.to_dict(),
        "artifact": {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "sealed_artifact_identity_sha256": artifact[
                "sealed_artifact_identity_sha256"
            ],
        },
        "envelope": {
            "path": envelope["receipt_path"],
            "file_sha256": envelope["receipt_file_sha256"],
            "content_sha256": envelope["receipt_content_sha256"],
        },
        "authoritative_micro_receipt": {
            "path": micro["receipt_path"],
            "file_sha256": micro["receipt_file_sha256"],
            "content_sha256": micro["receipt_content_sha256"],
        },
        "semantic_gate_receipt": {
            "path": semantic["receipt_path"],
            "file_sha256": semantic["receipt_file_sha256"],
            "content_sha256": semantic["semantic_gate_receipt_sha256"],
        },
        "science_continuation_contract_sha256": science[
            "science_consumer_extension_contract_sha256"
        ],
        "source_checkpoint_identity_sha256": None,
        "measurement_schedule": schedule,
    }
    if exact_founder_bootstrap:
        operation["exact_founder_bootstrap"] = (
            workhorse_exact_founder_bootstrap_contract_v1()
        )
    run_root = Path(args.run_root).expanduser().resolve()
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(f"runtime root must be fresh: {run_root}")
    control_root = _fresh_control_root(args.control_root)
    spec = _base_spec(
        campaign_id=args.campaign_id,
        artifact_binding=artifact,
        run_root=run_root,
    )
    if science_authority_base:
        spec["description"] = (
            "No-execution Workhorse V2 SCIENCE_COMPACT base plan for the "
            "existing Science Request authority resolver."
        )
        spec["artifact_lifecycle"]["lifecycle_profile"] = "SCIENCE_COMPACT"
    _bind_acs_native_runtime_artifacts(
        spec,
        acs_slots=args.acs_slots,
        extension_path=args.acs_native_runtime_extension,
    )
    capacity = envelope["capacity_basis"]
    spec["execution"]["budgets"]["vram_bytes"] = capacity["planning_available_bytes"]
    spec["execution"]["budgets"]["minimum_free_gpu_bytes"] = capacity[
        "effective_reserve_bytes"
    ]
    spec["execution"]["budgets"]["wall_seconds"] = max(
        int(spec["execution"]["budgets"]["wall_seconds"]),
        _workhorse_wall_seconds_upper_bound_v2(contract.total_ticks),
    )
    spec["parameters"] = {
        "workhorse.operation": "EXECUTION_ROW",
        "workhorse.shape_id": shape_id,
        "workhorse.backend_id": backend_id,
        "workhorse.B": contract.B,
        "workhorse.temporal_k": contract.temporal_k,
    }
    target_tick = checkpoint_ticks[-1]
    spec["execution"].update(
        {
            "start_next_tick": contract.start_next_tick,
            "total_ticks": target_tick,
            "checkpoint_ticks": list(checkpoint_ticks),
            "resident_owner_span_ticks": contract.total_ticks,
            "native_dispatch_window_ticks": contract.temporal_k,
            "source_window_ticks": contract.temporal_k,
            "event_window_capacity": 2 * contract.shape.capacity,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": target_tick, "checkpoint_ticks": list(checkpoint_ticks)}
    )
    spec["checkpoints"].update(
        {
            "acs_slots": args.acs_slots,
            "acs_capture_instrumentation": bool(args.acs_slots),
            "native_streamed_checkpoint": True,
        }
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": list(checkpoint_ticks),
        "roles": [
            {
                "identity": f"checkpoint-tick-{tick:012d}",
                "checkpoint_tick": tick,
                "role": "terminal" if tick == target_tick else "intermediate",
                "retention_class": args.checkpoint_retention_class,
            }
            for tick in checkpoint_ticks
        ],
    }
    storage_accounting = workhorse_checkpoint_storage_accounting_v2(
        contract=contract,
        checkpoint_ticks=checkpoint_ticks,
        acs_checkpoint_slots=args.acs_slots,
        dispatch_count=(
            schedule["warmup_dispatches"] + schedule["measured_dispatches"]
        ),
    )
    _apply_workhorse_storage_accounting_v2(spec, storage_accounting)
    spec["analysis"] = {
        "required_metrics": ["workhorse.p10_world_ticks_per_second"],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": deepcopy(operation),
            "contract_sha256": canonical_sha256(operation),
        },
    }
    if science_authority_base:
        _write_science_authority_base_plan_v1(
            control_root,
            spec=spec,
            shape_id=shape_id,
        )
    else:
        _write_outputs(
            control_root,
            spec=spec,
            run_root=run_root,
            purpose="benchmark",
            acs_slots=args.acs_slots,
        )


def materialize_science_demo(args: argparse.Namespace) -> None:
    artifact = workhorse_artifact_binding_v2(args.artifact)
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(args.envelope)
    if envelope["artifact"] != artifact:
        raise RuntimeError("science demo artifact differs from its device envelope")
    identity = artifact["identity"]
    if (
        identity["shape_id"] != "w128h80p2560c5120"
        or identity["backend_id"] != "reference_phased_v2"
    ):
        raise RuntimeError("science demo requires the exact 10x reference artifact")
    envelope_binding = {
        "path": envelope["receipt_path"],
        "file_sha256": envelope["receipt_file_sha256"],
        "content_sha256": envelope["receipt_content_sha256"],
    }
    semantic = workhorse_science_semantic_inheritance_binding_v1(args.semantic_gate)
    science = workhorse_science_checkpoint_compatibility_binding_v2(REPO_ROOT)
    demo = workhorse_science_demo_contract_v1(
        artifact_binding=artifact,
        envelope_binding=envelope_binding,
        semantic_inheritance_binding=semantic,
        science_checkpoint_compatibility_binding=science,
    )
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id="w128h80p2560c5120",
        backend_id="reference_phased_v2",
        B=10,
        temporal_k=1,
        total_ticks=40,
        start_next_tick=0,
    )
    operation = {
        "schema_version": OPERATION_SCHEMA_V3,
        "operation": SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
        "contract": contract.to_dict(),
        "artifact": {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "sealed_artifact_identity_sha256": artifact[
                "sealed_artifact_identity_sha256"
            ],
        },
        "envelope": envelope_binding,
        "semantic_inheritance_binding": semantic,
        "science_checkpoint_compatibility_binding": science,
        "science_demo_contract": demo,
        "science_continuation_contract_sha256": science[
            "science_consumer_extension_contract_sha256"
        ],
        "source_checkpoint_identity_sha256": None,
    }
    run_root = Path(args.run_root).expanduser().resolve()
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(f"runtime root must be fresh: {run_root}")
    control_root = _fresh_control_root(args.control_root)
    spec = _base_spec(
        campaign_id=args.campaign_id,
        artifact_binding=artifact,
        run_root=run_root,
    )
    _bind_acs_native_runtime_artifacts(
        spec,
        acs_slots=args.acs_slots,
        extension_path=args.acs_native_runtime_extension,
    )
    spec["description"] = (
        "Ten-world, 40-absolute-tick Workhorse V2 science-compatibility "
        "technical demo with compact host-boundary records and real restore."
    )
    spec["artifact_lifecycle"]["lifecycle_profile"] = "SCIENCE_COMPACT"
    spec["artifact_lifecycle"]["observability"]["raw_tick_trace_policy"] = "forbidden"
    spec["artifact_lifecycle"]["observability"]["debug_event_trace_policy"] = (
        "forbidden"
    )
    capacity = envelope["capacity_basis"]
    spec["execution"]["budgets"]["vram_bytes"] = capacity["planning_available_bytes"]
    spec["execution"]["budgets"]["minimum_free_gpu_bytes"] = capacity[
        "effective_reserve_bytes"
    ]
    spec["execution"]["budgets"]["wall_seconds"] = max(
        int(spec["execution"]["budgets"]["wall_seconds"]),
        _workhorse_wall_seconds_upper_bound_v2(40),
    )
    spec["parameters"] = {
        "workhorse.operation": SCIENCE_COMPATIBILITY_DEMO_OPERATION_V1,
        "workhorse.shape_id": contract.shape_id,
        "workhorse.backend_id": contract.backend_id,
        "workhorse.B": contract.B,
        "workhorse.temporal_k": contract.temporal_k,
    }
    spec["execution"].update(
        {
            "start_next_tick": 0,
            "total_ticks": 40,
            "checkpoint_ticks": [16, 32, 40],
            "resident_owner_span_ticks": 40,
            "native_dispatch_window_ticks": 1,
            "source_window_ticks": 1,
            "event_window_capacity": 2 * contract.shape.capacity,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": 40, "checkpoint_ticks": [16, 32, 40]}
    )
    spec["checkpoints"].update(
        {
            "acs_slots": args.acs_slots,
            "acs_capture_instrumentation": bool(args.acs_slots),
            "native_streamed_checkpoint": True,
        }
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": [16, 32, 40],
        "roles": [
            {
                "identity": "checkpoint-tick-000000000016",
                "checkpoint_tick": 16,
                "role": "transit_durable",
                "retention_class": "R3",
            },
            {
                "identity": "checkpoint-tick-000000000032",
                "checkpoint_tick": 32,
                "role": "transit_durable",
                "retention_class": "R3",
            },
            {
                "identity": "checkpoint-tick-000000000040",
                "checkpoint_tick": 40,
                "role": "terminal",
                "retention_class": "R3",
            },
        ],
    }
    spec["analysis"] = {
        "required_metrics": ["workhorse.science_compatibility_demo_completed"],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": deepcopy(operation),
            "contract_sha256": canonical_sha256(operation),
        },
    }
    _write_outputs(
        control_root,
        spec=spec,
        run_root=run_root,
        purpose="simulation",
        acs_slots=args.acs_slots,
    )


def materialize_indexed_continuation(args: argparse.Namespace) -> None:
    artifact = workhorse_artifact_binding_v2(args.artifact)
    envelope = validate_workhorse_device_envelope_calibration_receipt_v2(
        args.envelope
    )
    if envelope["artifact"] != artifact:
        raise RuntimeError("continuation artifact differs from its device envelope")
    identity = artifact["identity"]
    shape_id = str(identity["shape_id"])
    backend_id = str(identity["backend_id"])
    semantic = validate_workhorse_semantic_gate_receipt_v2(
        args.semantic_gate,
        expected_shape_id=shape_id,
        expected_candidate_artifact_sha256=artifact["sha256"],
        expected_candidate_backend_id=backend_id,
    )
    micro = validate_authoritative_micro_denominator_v2(
        args.micro_receipt,
        expected_device_identity=envelope["device_identity"],
    )
    science = workhorse_science_checkpoint_compatibility_binding_v2(REPO_ROOT)
    indexed_paths = [Path(value).expanduser().resolve() for value in args.indexed]
    indexed = [
        inspect_workhorse_indexed_checkpoint_set_v1(path)[0]
        for path in indexed_paths
    ]
    source_ticks = {
        row["world_state_set_identity"]["checkpoint_at_tick"] for row in indexed
    }
    if len(source_ticks) != 1:
        raise RuntimeError("indexed continuation sources have different frontiers")
    start_next_tick = source_ticks.pop()
    if args.world_uids_json is None:
        world_uids = sorted(
            row["world_uid"]
            for manifest in indexed
            for row in manifest["world_index_by_uid"]
        )
    else:
        value = json.loads(
            Path(args.world_uids_json).expanduser().read_text(encoding="utf-8")
        )
        world_uids = value.get("world_uids") if isinstance(value, dict) else value
        if not isinstance(world_uids, list):
            raise RuntimeError("world UID file must contain a list or world_uids list")
    checkpoint_ticks = list(args.checkpoint_ticks)
    if (
        not checkpoint_ticks
        or checkpoint_ticks != sorted(set(checkpoint_ticks))
        or checkpoint_ticks[-1] <= start_next_tick
    ):
        raise RuntimeError(
            "continuation checkpoints must be increasing absolute ticks after source"
        )
    target_next_tick = checkpoint_ticks[-1]
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=shape_id,
        backend_id=backend_id,
        B=len(world_uids),
        temporal_k=args.temporal_k,
        total_ticks=target_next_tick - start_next_tick,
        start_next_tick=start_next_tick,
    )
    admitted_rows = {
        row["temporal_k"]
        for row in envelope["expected_rows"]
        if row["B"] == contract.B
        and row["temporal_k"] <= contract.temporal_k
        and row["planned_admission"].get("admitted") is True
    }
    admitted_temporal_k = (
        sorted(set(args.admitted_temporal_k))
        if args.admitted_temporal_k
        else sorted(admitted_rows)
    )
    if (
        1 not in admitted_temporal_k
        or contract.temporal_k not in admitted_temporal_k
        or any(value not in admitted_rows for value in admitted_temporal_k)
    ):
        raise RuntimeError(
            "continuation admitted K list must be envelope-admitted and include K1 "
            "and the preferred K"
        )
    repack = workhorse_repack_binding_v2(
        indexed_paths,
        ordered_world_uids=world_uids,
        target_contract=contract,
    )
    dispatch = compile_workhorse_absolute_dispatch_plan_v2(
        start_next_tick=start_next_tick,
        target_next_tick=target_next_tick,
        preferred_temporal_k=contract.temporal_k,
        admitted_temporal_k=admitted_temporal_k,
        boundary_next_ticks=checkpoint_ticks,
    )
    if (args.resource_multiplier is None) != (args.mutation_rate is None):
        raise RuntimeError(
            "selected-world scientific controls must be supplied as an exact pair"
        )
    scientific_control = (
        None
        if args.resource_multiplier is None
        else workhorse_scientific_control_contract_v1(
            effective_next_absolute_tick=start_next_tick,
            resource_multiplier=args.resource_multiplier,
            mutation_rate=args.mutation_rate,
        )
    )
    continuation = workhorse_indexed_continuation_contract_v2(
        science_continuation_contract_sha256=science[
            "science_consumer_extension_contract_sha256"
        ],
        repack_binding=repack,
        dispatch_plan=dispatch,
        checkpoint_ticks=checkpoint_ticks,
        scientific_control_contract=scientific_control,
    )
    operation = _build_indexed_continuation_operation_v2(
        contract=contract,
        artifact=artifact,
        envelope=envelope,
        authoritative_micro=micro,
        semantic_gate=semantic,
        science_continuation_sha256=science[
            "science_consumer_extension_contract_sha256"
        ],
        continuation=continuation,
    )
    run_root = Path(args.run_root).expanduser().resolve()
    if run_root.exists() or run_root.is_symlink():
        raise RuntimeError(f"runtime root must be fresh: {run_root}")
    control_root = _fresh_control_root(args.control_root)
    spec = _base_spec(
        campaign_id=args.campaign_id,
        artifact_binding=artifact,
        run_root=run_root,
    )
    _bind_acs_native_runtime_artifacts(
        spec,
        acs_slots=2,
        extension_path=args.acs_native_runtime_extension,
    )
    capacity = envelope["capacity_basis"]
    spec["execution"]["budgets"]["vram_bytes"] = capacity[
        "planning_available_bytes"
    ]
    spec["execution"]["budgets"]["minimum_free_gpu_bytes"] = capacity[
        "effective_reserve_bytes"
    ]
    spec["execution"]["budgets"]["wall_seconds"] = max(
        int(spec["execution"]["budgets"]["wall_seconds"]),
        _workhorse_wall_seconds_upper_bound_v2(contract.total_ticks),
    )
    spec["parameters"] = {
        "workhorse.operation": INDEXED_ACS_CONTINUATION_OPERATION_V2,
        "workhorse.shape_id": shape_id,
        "workhorse.backend_id": backend_id,
        "workhorse.B": contract.B,
        "workhorse.temporal_k": contract.temporal_k,
    }
    spec["execution"].update(
        {
            "start_next_tick": start_next_tick,
            "total_ticks": target_next_tick,
            "checkpoint_ticks": checkpoint_ticks,
            "resident_owner_span_ticks": contract.total_ticks,
            "native_dispatch_window_ticks": contract.temporal_k,
            "source_window_ticks": contract.temporal_k,
            "event_window_capacity": 2 * contract.shape.capacity,
        }
    )
    spec["execution"]["technical_gate"].update(
        {"total_ticks": target_next_tick, "checkpoint_ticks": checkpoint_ticks}
    )
    spec["checkpoints"].update(
        {
            "acs_slots": 2,
            "acs_capture_instrumentation": True,
            "native_streamed_checkpoint": True,
        }
    )
    spec["artifact_lifecycle"]["checkpoints"] = {
        "authority": "absolute_tick",
        "checkpoint_ticks": checkpoint_ticks,
        "roles": [
            {
                "identity": f"checkpoint-tick-{tick:012d}",
                "checkpoint_tick": tick,
                "role": "terminal" if tick == target_next_tick else "intermediate",
                "retention_class": args.checkpoint_retention_class,
            }
            for tick in checkpoint_ticks
        ],
    }
    storage = workhorse_checkpoint_storage_accounting_v2(
        contract=contract,
        checkpoint_ticks=checkpoint_ticks,
        acs_checkpoint_slots=2,
        dispatch_count=dispatch["dispatch_count"],
    )
    _apply_workhorse_storage_accounting_v2(spec, storage)
    spec["analysis"] = {
        "required_metrics": [
            INDEXED_ACS_CONTINUATION_AGGREGATE_WTPS_METRIC_V2
        ],
        "optional_metrics": [],
        "rules": [],
        "extension": {
            "id": OPERATION_EXTENSION_ID,
            "contract": deepcopy(operation),
            "contract_sha256": canonical_sha256(operation),
        },
    }
    _write_outputs(
        control_root,
        spec=spec,
        run_root=run_root,
        purpose="simulation",
        acs_slots=2,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration = subparsers.add_parser("calibration")
    calibration.add_argument("--artifact", required=True)
    calibration.add_argument("--control-root", required=True)
    calibration.add_argument("--run-root", required=True)
    calibration.add_argument("--campaign-id", required=True)
    calibration.add_argument("--reserve-bytes", type=int, default=512 * 1024**2)
    calibration.set_defaults(func=materialize_calibration)
    semantic = subparsers.add_parser("semantic")
    semantic.add_argument("--case-id", required=True)
    semantic.add_argument("--semantic-temporal-k", type=int, required=True)
    semantic.add_argument("--reference-artifact", required=True)
    semantic.add_argument("--candidate-artifact", required=True)
    semantic.add_argument("--reference-envelope", required=True)
    semantic.add_argument("--candidate-envelope", required=True)
    semantic.add_argument("--source-case-receipt")
    semantic.add_argument("--control-root", required=True)
    semantic.add_argument("--run-root", required=True)
    semantic.add_argument("--campaign-id", required=True)
    semantic.set_defaults(func=materialize_semantic_case)
    gate = subparsers.add_parser("semantic-gate")
    gate.add_argument("--aggregate", required=True)
    gate.add_argument("--reference-artifact", required=True)
    gate.add_argument("--candidate-artifact", required=True)
    gate.add_argument("--output", required=True)
    gate.set_defaults(func=materialize_semantic_gate)
    row = subparsers.add_parser("row")
    row.add_argument("--artifact", required=True)
    row.add_argument("--envelope", required=True)
    row.add_argument("--micro-receipt", required=True)
    row.add_argument("--semantic-gate", required=True)
    row.add_argument("--B", type=int, required=True)
    row.add_argument("--temporal-k", type=int, required=True)
    row.add_argument("--warmup-ticks", type=int, default=192)
    row.add_argument("--measurement-window-ticks", type=int, default=64)
    row.add_argument("--measured-windows", type=int, default=20)
    row.add_argument("--start-next-tick", type=int, default=0)
    row.add_argument("--acs-slots", type=int, choices=(0, 1, 2), default=0)
    row.add_argument("--acs-native-runtime-extension")
    row.add_argument(
        "--exact-founder-bootstrap",
        action="store_true",
        help=(
            "opt in to a first-dispatch plus terminal ACS2/R4 schedule from "
            "absolute tick zero for exact selected-parent materialization"
        ),
    )
    row.add_argument(
        "--checkpoint-retention-class",
        choices=("R2", "R4"),
        default="R2",
        help=(
            "R2 for releasable technical material or R4 while the checkpoint "
            "remains a science/replay continuation source"
        ),
    )
    row.add_argument(
        "--science-authority-base-plan",
        action="store_true",
        help=(
            "write a no-execution B1/ACS2 SCIENCE_COMPACT compiled base plan; "
            "do not write an ExecutionRequest"
        ),
    )
    row.add_argument("--control-root", required=True)
    row.add_argument("--run-root", required=True)
    row.add_argument("--campaign-id", required=True)
    row.set_defaults(func=materialize_row)
    science = subparsers.add_parser("science-demo")
    science.add_argument("--artifact", required=True)
    science.add_argument("--envelope", required=True)
    science.add_argument("--semantic-gate", required=True)
    science.add_argument("--control-root", required=True)
    science.add_argument("--run-root", required=True)
    science.add_argument("--campaign-id", required=True)
    science.add_argument("--acs-slots", type=int, choices=(0, 1, 2), default=0)
    science.add_argument("--acs-native-runtime-extension")
    science.set_defaults(func=materialize_science_demo)
    continuation = subparsers.add_parser("indexed-continuation")
    continuation.add_argument("--artifact", required=True)
    continuation.add_argument("--envelope", required=True)
    continuation.add_argument("--micro-receipt", required=True)
    continuation.add_argument("--semantic-gate", required=True)
    continuation.add_argument("--indexed", action="append", required=True)
    continuation.add_argument("--world-uids-json")
    continuation.add_argument("--temporal-k", type=int, required=True)
    continuation.add_argument(
        "--admitted-temporal-k", type=int, action="append", default=[]
    )
    continuation.add_argument(
        "--checkpoint-ticks", type=int, nargs="+", required=True
    )
    continuation.add_argument("--acs-native-runtime-extension", required=True)
    continuation.add_argument(
        "--checkpoint-retention-class",
        choices=("R2", "R4"),
        default="R2",
        help=(
            "R2 for releasable technical material or R4 while the checkpoint "
            "remains a science/replay continuation source"
        ),
    )
    continuation.add_argument("--resource-multiplier", type=float)
    continuation.add_argument("--mutation-rate", type=float)
    continuation.add_argument("--control-root", required=True)
    continuation.add_argument("--run-root", required=True)
    continuation.add_argument("--campaign-id", required=True)
    continuation.set_defaults(func=materialize_indexed_continuation)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
